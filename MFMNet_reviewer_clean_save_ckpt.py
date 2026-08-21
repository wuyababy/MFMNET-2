#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFMNet reviewer-revision runner — clean single-file version.

Purpose
-------
This script replaces the scattered loader/model/ablation/resource scripts with one
reproducible entry point for the reviewer-requested experiments.

It intentionally does NOT try to force the full model to outperform ablations.
It focuses on experimental correctness and traceability:
  1) loads paired drug1/drug2 data from raw CSVs in one Dataset;
  2) keeps drug pairs aligned inside one DataLoader;
  3) uses all samples in five outer folds (no discarded remainder samples);
  4) creates an inner validation split inside each outer-training fold;
  5) selects epochs only by inner-validation AUROC;
  6) evaluates the outer test fold once after model selection;
  7) uses the same outer folds and fold-level seed for every ablation variant;
  8) instantiates only the graph branches needed by each graph ablation, so model
     parameter count and checkpoint size are meaningful;
  9) writes fold metrics, learning curves, predictions, paired tests, resource
     measurements, split assignments, environment information, and publication tables;
 10) supports resume and selective reruns.

Expected directory layout
-------------------------
Run this file from the original codes directory, or point --data-root to the data folder:

codes/
  MFMNet_reviewer_clean_onefile.py
  data/
    raw/
      all_ddi.csv
      all_drug_embedding.csv

The script no longer depends on loader.py, enhanced_model.py, transformer.py, model.py,
or gene_vector.csv for the executed reviewer-analysis path.

Default variants
----------------
full, gcn_only, gin_only, gat_only, no_router, simple_concatenation,
no_drug_embedding, no_cell_line

Important
---------
The raw all_ddi.csv is expected to contain:
  drug1, drug2, label, and 954 cell-line feature columns beginning at column index 9.
The raw all_drug_embedding.csv is expected to contain 1536 numeric columns:
  first 768 = drug1 embedding; last 768 = drug2 embedding.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, GraphNorm

from rdkit import Chem


CODE_VERSION = "2026-08-11-clean-onefile-v1"
DEFAULT_SEED = 0
ATOM_DIM = 78
MAX_ATOMS = 100
DRUG_EMBED_DIM = 768
CELL_DIM = 954
GRAPH_DIM = 128

VARIANTS = [
    "full",
    "gcn_only",
    "gin_only",
    "gat_only",
    "no_router",
    "simple_concatenation",
    "no_drug_embedding",
    "no_cell_line",
]

DISPLAY = {
    "full": "Full MFMNet backbone",
    "gcn_only": "GCN-only",
    "gin_only": "GIN-only",
    "gat_only": "GAT-only",
    "no_router": "No router (mean fusion)",
    "simple_concatenation": "Simple concatenation",
    "no_drug_embedding": "Without drug embedding",
    "no_cell_line": "Without cell-line profile",
}


# -----------------------------------------------------------------------------
# Reproducibility and device helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    cuda_ok = torch.cuda.is_available()

    if requested == "xpu":
        if not xpu_ok:
            raise RuntimeError("--device xpu requested, but torch.xpu.is_available() is False")
        return torch.device("xpu:0")
    if requested == "cuda":
        if not cuda_ok:
            raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise ValueError(f"Unsupported --device value: {requested}")

    if xpu_ok:
        return torch.device("xpu:0")
    if cuda_ok:
        return torch.device("cuda:0")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def empty_accelerator_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        try:
            torch.xpu.memory.empty_cache()
        except AttributeError:
            torch.xpu.empty_cache()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu":
        try:
            torch.xpu.memory.reset_peak_memory_stats(device)
        except AttributeError:
            try:
                torch.xpu.reset_peak_memory_stats(device)
            except AttributeError:
                pass


def max_memory_allocated_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    if device.type == "xpu":
        try:
            return float(torch.xpu.memory.max_memory_allocated(device) / (1024 ** 2))
        except AttributeError:
            try:
                return float(torch.xpu.max_memory_allocated(device) / (1024 ** 2))
            except AttributeError:
                return float("nan")
    return float("nan")


def environment_dict(device: torch.device) -> dict:
    d = {
        "code_version": CODE_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
        "device": str(device),
    }
    try:
        import torch_geometric
        d["torch_geometric"] = torch_geometric.__version__
    except Exception as exc:
        d["torch_geometric_error"] = str(exc)
    try:
        import rdkit
        d["rdkit"] = rdkit.__version__
    except Exception as exc:
        d["rdkit_error"] = str(exc)

    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        d.update({
            "accelerator_type": "CUDA",
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_gb": p.total_memory / (1024 ** 3),
            "cudnn": torch.backends.cudnn.version(),
        })
    elif device.type == "xpu":
        d.update({
            "accelerator_type": "XPU",
            "gpu_name": torch.xpu.get_device_name(device),
        })
        try:
            props = torch.xpu.get_device_properties(device)
            total = getattr(props, "total_memory", None)
            if total is not None:
                d["gpu_total_memory_gb"] = total / (1024 ** 3)
            d["xpu_properties"] = str(props)
        except Exception as exc:
            d["xpu_properties_error"] = str(exc)
    else:
        d["accelerator_type"] = "CPU"
    return d


# -----------------------------------------------------------------------------
# Data loading: one paired Dataset, no independent drug1/drug2 shuffling
# -----------------------------------------------------------------------------

def one_of_k_encoding(x, allowable_set: Sequence) -> List[bool]:
    if x not in allowable_set:
        raise ValueError(f"input {x!r} not in allowable set {allowable_set!r}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set: Sequence) -> List[bool]:
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom) -> np.ndarray:
    feature = (
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
                "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag",
                "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd",
                "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()]
    )
    arr = np.asarray(feature, dtype=np.float32)
    if arr.shape[0] != ATOM_DIM:
        raise RuntimeError(f"Expected {ATOM_DIM} atom features, got {arr.shape[0]}")
    denom = float(arr.sum())
    if denom > 0:
        arr /= denom
    return arr


def smiles_to_fixed_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    n_atoms = mol.GetNumAtoms()
    if n_atoms > MAX_ATOMS:
        raise ValueError(
            f"Molecule contains {n_atoms} atoms, exceeding MAX_ATOMS={MAX_ATOMS}: {smiles!r}"
        )

    x = torch.zeros((MAX_ATOMS, ATOM_DIM), dtype=torch.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = torch.from_numpy(atom_features(atom))

    edges: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


@dataclass
class PairSample:
    index: int
    left_graph: Data
    right_graph: Data
    left_embedding: torch.Tensor
    right_embedding: torch.Tensor
    cell: torch.Tensor
    label: int


@dataclass
class PairBatch:
    index: torch.Tensor
    left_graph: Batch
    right_graph: Batch
    left_embedding: torch.Tensor
    right_embedding: torch.Tensor
    cell: torch.Tensor
    label: torch.Tensor

    def to(self, device: torch.device) -> "PairBatch":
        return PairBatch(
            index=self.index.to(device),
            left_graph=self.left_graph.to(device),
            right_graph=self.right_graph.to(device),
            left_embedding=self.left_embedding.to(device),
            right_embedding=self.right_embedding.to(device),
            cell=self.cell.to(device),
            label=self.label.to(device),
        )


class DrugPairDataset(Dataset):
    def __init__(self, data_root: Path):
        raw = data_root / "raw"
        ddi_path = raw / "all_ddi.csv"
        emb_path = raw / "all_drug_embedding.csv"
        if not ddi_path.exists():
            raise FileNotFoundError(ddi_path)
        if not emb_path.exists():
            raise FileNotFoundError(emb_path)

        print(f"Loading {ddi_path}")
        self.ddi = pd.read_csv(ddi_path)
        required = {"drug1", "drug2", "label"}
        missing = required.difference(self.ddi.columns)
        if missing:
            raise ValueError(f"all_ddi.csv missing required columns: {sorted(missing)}")

        if self.ddi.shape[1] < 9 + CELL_DIM:
            raise ValueError(
                f"Expected at least {9 + CELL_DIM} columns in all_ddi.csv; got {self.ddi.shape[1]}"
            )

        print(f"Loading {emb_path}")
        emb_df = pd.read_csv(emb_path)
        if len(emb_df) != len(self.ddi):
            raise ValueError(
                f"Row count mismatch: all_ddi={len(self.ddi)}, all_drug_embedding={len(emb_df)}"
            )
        if emb_df.shape[1] != DRUG_EMBED_DIM * 2:
            raise ValueError(
                f"Expected {DRUG_EMBED_DIM * 2} embedding columns, got {emb_df.shape[1]}"
            )

        self.left_embeddings = emb_df.iloc[:, :DRUG_EMBED_DIM].to_numpy(dtype=np.float32, copy=True)
        self.right_embeddings = emb_df.iloc[:, DRUG_EMBED_DIM:].to_numpy(dtype=np.float32, copy=True)
        self.cells = self.ddi.iloc[:, 9:9 + CELL_DIM].to_numpy(dtype=np.float32, copy=True)
        self.labels = self.ddi["label"].to_numpy(dtype=np.int64, copy=True)
        self.smiles_left = self.ddi["drug1"].astype(str).tolist()
        self.smiles_right = self.ddi["drug2"].astype(str).tolist()

        unique_labels = sorted(np.unique(self.labels).tolist())
        if unique_labels != [0, 1]:
            raise ValueError(f"Expected binary labels [0, 1], got {unique_labels}")
        if not np.isfinite(self.left_embeddings).all():
            raise ValueError("Non-finite values found in left drug embeddings")
        if not np.isfinite(self.right_embeddings).all():
            raise ValueError("Non-finite values found in right drug embeddings")
        if not np.isfinite(self.cells).all():
            raise ValueError("Non-finite values found in cell-line features")

        unique_smiles = sorted(set(self.smiles_left).union(self.smiles_right))
        print(f"Building fixed molecular graphs for {len(unique_smiles)} unique SMILES ...")
        self.graph_cache: Dict[str, Data] = {}
        for i, smi in enumerate(unique_smiles, start=1):
            self.graph_cache[smi] = smiles_to_fixed_graph(smi)
            if i % 20 == 0 or i == len(unique_smiles):
                print(f"  graph cache {i}/{len(unique_smiles)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> PairSample:
        return PairSample(
            index=int(idx),
            left_graph=self.graph_cache[self.smiles_left[idx]],
            right_graph=self.graph_cache[self.smiles_right[idx]],
            left_embedding=torch.from_numpy(self.left_embeddings[idx]),
            right_embedding=torch.from_numpy(self.right_embeddings[idx]),
            cell=torch.from_numpy(self.cells[idx]),
            label=int(self.labels[idx]),
        )

    def audit(self) -> dict:
        out = {
            "n_records": int(len(self)),
            "n_positive": int((self.labels == 1).sum()),
            "n_negative": int((self.labels == 0).sum()),
            "positive_fraction": float((self.labels == 1).mean()),
            "n_unique_smiles": int(len(self.graph_cache)),
            "n_unique_drug1_smiles": int(len(set(self.smiles_left))),
            "n_unique_drug2_smiles": int(len(set(self.smiles_right))),
            "drug_embedding_dim_each": DRUG_EMBED_DIM,
            "cell_feature_dim": CELL_DIM,
            "atom_feature_dim": ATOM_DIM,
            "max_atoms": MAX_ATOMS,
        }
        for col in ["drug1Name", "drug2Name", "cell", "fold"]:
            if col in self.ddi.columns:
                if col == "fold":
                    out["source_fold_counts"] = {
                        str(k): int(v) for k, v in self.ddi[col].value_counts(dropna=False).sort_index().items()
                    }
                else:
                    out[f"n_unique_{col}"] = int(self.ddi[col].nunique(dropna=True))
        return out


def collate_pair_samples(samples: Sequence[PairSample]) -> PairBatch:
    return PairBatch(
        index=torch.tensor([s.index for s in samples], dtype=torch.long),
        left_graph=Batch.from_data_list([s.left_graph for s in samples]),
        right_graph=Batch.from_data_list([s.right_graph for s in samples]),
        left_embedding=torch.stack([s.left_embedding for s in samples], dim=0),
        right_embedding=torch.stack([s.right_embedding for s in samples], dim=0),
        cell=torch.stack([s.cell for s in samples], dim=0),
        label=torch.tensor([s.label for s in samples], dtype=torch.long),
    )


class IndexDataset(Dataset):
    def __init__(self, base: DrugPairDataset, indices: Sequence[int]):
        self.base = base
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> PairSample:
        return self.base[int(self.indices[i])]


def make_loader(
    dataset: DrugPairDataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> TorchDataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return TorchDataLoader(
        IndexDataset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pair_samples,
        generator=generator,
        pin_memory=False,
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class DrugEmbeddingProcessor(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CellLineEncoder(nn.Module):
    def __init__(self, input_dim: int = 954, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = self.weight_net(features)
        return self.projection(features * weights)


class MFMNetReviewerModel(nn.Module):
    VALID_VARIANTS = set(VARIANTS)

    def __init__(self, variant: str, dropout_rate: float = 0.3, out_channels: int = 2):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.dropout_rate = dropout_rate
        self.relu = nn.ReLU()

        if variant in {"full", "gcn_only", "no_router", "simple_concatenation", "no_drug_embedding", "no_cell_line"}:
            self.gcn_conv1 = GCNConv(ATOM_DIM, 512)
            self.gcn_conv2 = GCNConv(512, GRAPH_DIM)
        else:
            self.gcn_conv1 = None
            self.gcn_conv2 = None

        if variant in {"full", "gin_only", "no_router", "simple_concatenation", "no_drug_embedding", "no_cell_line"}:
            gin_nn1 = nn.Sequential(
                nn.Linear(ATOM_DIM, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
            gin_nn2 = nn.Sequential(
                nn.Linear(512, GRAPH_DIM),
                nn.ReLU(),
                nn.Linear(GRAPH_DIM, GRAPH_DIM),
            )
            self.gin_conv1 = GINConv(gin_nn1)
            self.gin_conv2 = GINConv(gin_nn2)
            self.gnorm1 = GraphNorm(512)
        else:
            self.gin_conv1 = None
            self.gin_conv2 = None
            self.gnorm1 = None

        if variant in {"full", "gat_only", "no_router", "simple_concatenation", "no_drug_embedding", "no_cell_line"}:
            self.gat_conv1 = GATv2Conv(ATOM_DIM, 256, heads=2, dropout=dropout_rate)
            self.gat_conv2 = GATv2Conv(512, GRAPH_DIM, heads=1, dropout=dropout_rate)
        else:
            self.gat_conv1 = None
            self.gat_conv2 = None

        if variant in {"full", "no_drug_embedding", "no_cell_line"}:
            self.graph_fusion = AdaptiveFeatureFusion(GRAPH_DIM * 3, GRAPH_DIM)
        else:
            self.graph_fusion = None

        if variant == "simple_concatenation":
            self.simple_projection = nn.Linear(GRAPH_DIM * 3, GRAPH_DIM)
        else:
            self.simple_projection = None

        self.use_drug_embedding = variant != "no_drug_embedding"
        self.use_cell_line = variant != "no_cell_line"

        self.drug_embedding_processor = (
            DrugEmbeddingProcessor(DRUG_EMBED_DIM, 512, 256) if self.use_drug_embedding else None
        )
        self.cell_encoder = CellLineEncoder(CELL_DIM, 512, 256) if self.use_cell_line else None

        # Keep the classifier input fixed at 1024 for all modality ablations.
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, out_channels),
        )

    def _gcn(self, batch: Batch) -> torch.Tensor:
        h = self.gcn_conv1(batch.x, batch.edge_index)
        h = self.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.gcn_conv2(h, batch.edge_index)

    def _gin(self, batch: Batch) -> torch.Tensor:
        h = self.gin_conv1(batch.x, batch.edge_index)
        h = self.relu(h)
        h = self.gnorm1(h, batch.batch)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.gin_conv2(h, batch.edge_index)

    def _gat(self, batch: Batch) -> torch.Tensor:
        h = self.gat_conv1(batch.x, batch.edge_index)
        h = self.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.gat_conv2(h, batch.edge_index)

    def _graph_features(self, batch: Batch, batch_size: int) -> torch.Tensor:
        v = self.variant
        if v == "gcn_only":
            fused = self._gcn(batch)
        elif v == "gin_only":
            fused = self._gin(batch)
        elif v == "gat_only":
            fused = self._gat(batch)
        else:
            gcn = self._gcn(batch)
            gin = self._gin(batch)
            gat = self._gat(batch)
            if v == "no_router":
                fused = (gcn + gin + gat) / 3.0
            else:
                multi = torch.cat([gcn, gin, gat], dim=1)
                if v == "simple_concatenation":
                    fused = self.simple_projection(multi)
                else:
                    fused = self.graph_fusion(multi)

        expected_nodes = batch_size * MAX_ATOMS
        if fused.shape[0] != expected_nodes:
            raise RuntimeError(
                f"Expected {expected_nodes} nodes ({batch_size} x {MAX_ATOMS}), got {fused.shape[0]}"
            )
        return fused.reshape(batch_size, MAX_ATOMS, GRAPH_DIM)

    def forward(self, batch: PairBatch) -> torch.Tensor:
        batch_size = int(batch.label.shape[0])

        if self.use_drug_embedding:
            emb1 = self.drug_embedding_processor(batch.left_embedding)
            emb2 = self.drug_embedding_processor(batch.right_embedding)
        else:
            emb1 = torch.zeros((batch_size, 256), device=batch.label.device)
            emb2 = torch.zeros((batch_size, 256), device=batch.label.device)

        if self.use_cell_line:
            cell = self.cell_encoder(batch.cell)
        else:
            cell = torch.zeros((batch_size, 256), device=batch.label.device)

        graph1 = self._graph_features(batch.left_graph, batch_size).mean(dim=1)
        graph2 = self._graph_features(batch.right_graph, batch_size).mean(dim=1)

        combined = torch.cat([emb1, emb2, cell, graph1, graph2], dim=1)
        return self.classifier(combined)


# -----------------------------------------------------------------------------
# Metrics, training, checkpoints, resources
# -----------------------------------------------------------------------------

def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def theoretical_weight_size_mb(model: nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return float(total_bytes / (1024 ** 2))


def save_checkpoint_size_mb(model: nn.Module, path: Path) -> float:
    torch.save(model.state_dict(), path)
    mb = float(path.stat().st_size / (1024 ** 2))
    path.unlink(missing_ok=True)
    return mb


def metric_dict(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y, score)
    return {
        "AUROC": float(roc_auc_score(y, score)),
        "PR_AUC": float(metrics.auc(recall_curve, precision_curve)),
        "ACC": float(accuracy_score(y, pred)),
        "BACC": float(balanced_accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "TPR": float(recall_score(y, pred, zero_division=0)),
        "KAPPA": float(cohen_kappa_score(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


def predict(
    model: nn.Module,
    device: torch.device,
    loader: TorchDataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = (prob >= 0.5).long()
            ys.append(batch.label.detach().cpu().numpy())
            scores.append(prob.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            indices.append(batch.index.detach().cpu().numpy())
    return (
        np.concatenate(ys),
        np.concatenate(scores),
        np.concatenate(preds),
        np.concatenate(indices),
    )


def original_l2_penalty(model: nn.Module) -> torch.Tensor:
    terms = [torch.norm(p, 2) for p in model.parameters() if p.requires_grad]
    if not terms:
        raise RuntimeError("No trainable parameters")
    return torch.stack(terms).sum()


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    loader: TorchDataLoader,
    n_samples: int,
    repeats: int,
) -> dict:
    empty_accelerator_cache(device)
    _ = predict(model, device, loader)  # warm-up
    sync(device)
    reset_peak_memory(device)
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = predict(model, device, loader)
        sync(device)
        times.append(time.perf_counter() - t0)
    sec = float(np.median(times))
    return {
        "inference_seconds": sec,
        "inference_ms_per_sample": float(1000.0 * sec / n_samples),
        "throughput_samples_per_second": float(n_samples / sec),
        "inference_peak_gpu_memory_mb": max_memory_allocated_mb(device),
    }


def benchmark_scalability(
    model: nn.Module,
    device: torch.device,
    dataset: DrugPairDataset,
    test_idx: np.ndarray,
    batch_sizes: Sequence[int],
    repeats: int,
    seed: int,
    num_workers: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for bs in batch_sizes:
        try:
            loader = make_loader(dataset, test_idx, int(bs), False, seed, num_workers)
            empty_accelerator_cache(device)
            _ = predict(model, device, loader)
            sync(device)
            reset_peak_memory(device)
            times: List[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                _ = predict(model, device, loader)
                sync(device)
                times.append(time.perf_counter() - t0)
            sec = float(np.median(times))
            rows.append({
                "batch_size": int(bs),
                "n_samples": int(len(test_idx)),
                "inference_seconds": sec,
                "inference_ms_per_sample": float(1000.0 * sec / len(test_idx)),
                "throughput_samples_per_second": float(len(test_idx) / sec),
                "peak_gpu_memory_mb": max_memory_allocated_mb(device),
                "status": "ok",
            })
        except RuntimeError as exc:
            rows.append({
                "batch_size": int(bs),
                "n_samples": int(len(test_idx)),
                "status": f"failed: {exc}",
            })
            empty_accelerator_cache(device)
    return pd.DataFrame(rows)


def training_timing_summary(epoch_seconds: Sequence[float], wall_seconds: float) -> dict:
    arr = np.asarray(epoch_seconds, dtype=float)
    if len(arr) == 0:
        return {
            "training_active_seconds": 0.0,
            "median_epoch_seconds": float("nan"),
            "max_epoch_seconds": float("nan"),
            "wall_to_active_ratio": float("nan"),
            "timing_anomaly": True,
        }
    active = float(arr.sum())
    median_epoch = float(np.median(arr))
    max_epoch = float(arr.max())
    ratio = float(wall_seconds / active) if active > 0 else float("inf")
    within_epoch_anomaly = bool(median_epoch > 0 and max_epoch > 5.0 * median_epoch and max_epoch - median_epoch > 60.0)
    between_epoch_anomaly = bool(ratio > 1.20 and wall_seconds - active > 60.0)
    return {
        "training_active_seconds": active,
        "median_epoch_seconds": median_epoch,
        "max_epoch_seconds": max_epoch,
        "wall_to_active_ratio": ratio,
        "timing_anomaly": bool(within_epoch_anomaly or between_epoch_anomaly),
    }


def train_variant(
    variant: str,
    fold_number: int,
    dataset: DrugPairDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    outdir: Path,
    device: torch.device,
) -> dict:
    # Same fold-level seed for every variant. This is intentionally not variant-dependent.
    fold_seed = args.seed + 1000 * fold_number
    set_seed(fold_seed)

    train_loader = make_loader(
        dataset, train_idx, args.batch_size, args.shuffle_train, fold_seed + 1, args.num_workers
    )
    val_loader = make_loader(dataset, val_idx, args.batch_size, False, fold_seed + 2, args.num_workers)
    test_loader = make_loader(dataset, test_idx, args.batch_size, False, fold_seed + 3, args.num_workers)

    model = MFMNetReviewerModel(variant=variant, dropout_rate=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    n_params = parameter_count(model)
    theoretical_mb = theoretical_weight_size_mb(model)
    best_val_auc = -np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0
    epoch_rows: List[dict] = []
    epoch_seconds: List[float] = []

    empty_accelerator_cache(device)
    reset_peak_memory(device)
    sync(device)
    wall_start = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        sync(device)
        epoch_start = time.perf_counter()
        model.train()
        losses: List[float] = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = F.cross_entropy(out, batch.label)
            if args.l2 > 0:
                loss = loss + args.l2 * original_l2_penalty(model)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        mean_train_loss = float(np.mean(losses))
        scheduler.step(mean_train_loss)

        yv, sv, _, _ = predict(model, device, val_loader)
        val_auc = float(roc_auc_score(yv, sv))
        sync(device)
        epoch_sec = float(time.perf_counter() - epoch_start)
        epoch_seconds.append(epoch_sec)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_rows.append({
            "variant": variant,
            "fold": fold_number,
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_AUROC": val_auc,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_sec,
        })

        if val_auc > best_val_auc + args.min_delta:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or patience_counter >= args.early_stopping_patience:
            print(
                f"[{variant} fold {fold_number}] epoch={epoch} "
                f"loss={mean_train_loss:.5f} val_AUROC={val_auc:.5f} "
                f"best={best_val_auc:.5f}@{best_epoch} patience={patience_counter}"
            )

        if patience_counter >= args.early_stopping_patience:
            break

    sync(device)
    training_wall_seconds = float(time.perf_counter() - wall_start)
    training_peak_mb = max_memory_allocated_mb(device)


    if best_state is None:
        raise RuntimeError("Training ended without a best checkpoint")

    model.load_state_dict(best_state)

    if variant == "full":
        ckpt_dir = outdir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": best_state,
                "variant": variant,
                "fold": int(fold_number),
                "best_epoch": int(best_epoch),
                "best_val_AUROC": float(best_val_auc),
                "seed": int(fold_seed),
            },
            ckpt_dir / f"mfmnet_full_fold{fold_number}.pt",
        )   

    yt, st, pt, it = predict(model, device, test_loader)
    m = metric_dict(yt, st, pt)
    bench = benchmark_inference(model, device, test_loader, len(test_idx), args.inference_repeats)

    tmp_ckpt = outdir / f"_tmp_{variant}_fold{fold_number}.pt"
    checkpoint_mb = save_checkpoint_size_mb(model, tmp_ckpt)
    timing = training_timing_summary(epoch_seconds, training_wall_seconds)

    if variant == "full" and not args.no_scalability:
        scalability_df = benchmark_scalability(
            model=model,
            device=device,
            dataset=dataset,
            test_idx=test_idx,
            batch_sizes=args.scalability_batch_sizes,
            repeats=args.inference_repeats,
            seed=fold_seed + 4,
            num_workers=args.num_workers,
        )
        scalability_df.insert(0, "fold", fold_number)
        scalability_df.to_csv(outdir / f"scalability_full_fold{fold_number}.csv", index=False)

    pd.DataFrame({
        "sample_index": it.astype(int),
        "label": yt.astype(int),
        "score": st,
        "predicted_label": pt.astype(int),
    }).sort_values("sample_index").to_csv(
        outdir / f"pred_{variant}_fold{fold_number}.csv", index=False
    )
    pd.DataFrame(epoch_rows).to_csv(
        outdir / f"learning_{variant}_fold{fold_number}.csv", index=False
    )

    row = {
        "variant": variant,
        "display_name": DISPLAY[variant],
        "fold": fold_number,
        **m,
        "best_epoch": int(best_epoch),
        "best_val_AUROC": float(best_val_auc),
        "trainable_parameters": int(n_params),
        "theoretical_trainable_weight_size_mb": theoretical_mb,
        "checkpoint_size_mb": checkpoint_mb,
        "training_wall_seconds": training_wall_seconds,
        "training_wall_minutes": training_wall_seconds / 60.0,
        **timing,
        "training_active_minutes": timing["training_active_seconds"] / 60.0,
        "training_peak_gpu_memory_mb": training_peak_mb,
        **bench,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "seed": int(fold_seed),
    }
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row


# -----------------------------------------------------------------------------
# Fold construction and statistics
# -----------------------------------------------------------------------------

def make_outer_folds(n: int, seed: int, n_folds: int = 5) -> List[np.ndarray]:
    # Preserve the original seeded random-permutation spirit, but use np.array_split so
    # every record belongs to exactly one outer test fold.
    rr = random.Random(seed)
    perm = np.asarray(rr.sample(range(n), n), dtype=int)
    return [np.asarray(x, dtype=int) for x in np.array_split(perm, n_folds)]


def make_fold_indices(labels: np.ndarray, outer_folds: List[np.ndarray], fold_number: int, val_fraction: float, seed: int):
    test_idx = outer_folds[fold_number - 1]
    all_idx = np.arange(len(labels), dtype=int)
    mask = np.ones(len(labels), dtype=bool)
    mask[test_idx] = False
    outer_train = all_idx[mask]

    train_idx, val_idx = train_test_split(
        outer_train,
        test_size=val_fraction,
        random_state=seed + fold_number,
        stratify=labels[outer_train],
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def write_outer_assignment(outdir: Path, labels: np.ndarray, outer_folds: List[np.ndarray]) -> None:
    assignment = np.full(len(labels), -1, dtype=int)
    for fold_number, idx in enumerate(outer_folds, start=1):
        assignment[idx] = fold_number
    if (assignment < 1).any():
        raise RuntimeError("At least one sample was not assigned to an outer fold")
    pd.DataFrame({
        "sample_index": np.arange(len(labels), dtype=int),
        "outer_fold": assignment,
        "label": labels,
    }).to_csv(outdir / "outer_fold_assignments.csv", index=False)


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def paired_ablation_tests(fold_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    full = fold_df[fold_df["variant"] == "full"].sort_values("fold")
    rows: List[dict] = []
    for variant in VARIANTS[1:]:
        ab = fold_df[fold_df["variant"] == variant].sort_values("fold")
        if len(full) != 5 or len(ab) != 5:
            continue
        if not np.array_equal(full["fold"].to_numpy(), ab["fold"].to_numpy()):
            raise RuntimeError(f"Fold mismatch between Full and {variant}")
        d = full[metric].to_numpy(dtype=float) - ab[metric].to_numpy(dtype=float)
        t_stat, p_value = stats.ttest_rel(full[metric], ab[metric])
        if np.allclose(d, d[0]):
            ci_low = ci_high = float(d.mean())
        else:
            ci_low, ci_high = stats.t.interval(
                0.95, len(d) - 1, loc=float(d.mean()), scale=float(stats.sem(d))
            )
        rows.append({
            "comparison": f"Full vs {DISPLAY[variant]}",
            "variant": variant,
            "metric": metric,
            "mean_paired_difference_full_minus_ablation": float(d.mean()),
            "ci95_lower": float(ci_low),
            "ci95_upper": float(ci_high),
            "t_statistic": float(t_stat),
            "raw_p": float(p_value),
            "full_better_on_mean": bool(d.mean() > 0),
        })
    if rows:
        adjusted = holm_adjust([r["raw_p"] for r in rows])
        for r, p_adj in zip(rows, adjusted):
            r["holm_adjusted_p"] = float(p_adj)
    return pd.DataFrame(rows)


def t_ci(values: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float("nan"), float("nan")
    if np.allclose(a, a[0]):
        return float(a[0]), float(a[0])
    lo, hi = stats.t.interval(0.95, len(a) - 1, loc=float(a.mean()), scale=float(stats.sem(a)))
    return float(lo), float(hi)


def build_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    metrics_to_summarize = ["AUROC", "PR_AUC", "ACC", "BACC", "Precision", "TPR", "KAPPA", "F1"]
    for variant, g in fold_df.groupby("variant", sort=False):
        row = {"variant": variant, "display_name": DISPLAY.get(variant, variant), "n_folds": int(len(g))}
        for metric in metrics_to_summarize:
            vals = g[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_SD"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            lo, hi = t_ci(vals)
            row[f"{metric}_CI95_lower"] = lo
            row[f"{metric}_CI95_upper"] = hi
        for col in [
            "best_epoch",
            "trainable_parameters",
            "checkpoint_size_mb",
            "training_active_minutes",
            "training_wall_minutes",
            "training_peak_gpu_memory_mb",
            "inference_ms_per_sample",
            "throughput_samples_per_second",
            "inference_peak_gpu_memory_mb",
        ]:
            vals = g[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_median"] = float(np.nanmedian(vals))
        row["timing_anomaly_folds"] = ",".join(str(int(f)) for f in g.loc[g["timing_anomaly"] == True, "fold"].tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def build_publication_tables(fold_df: pd.DataFrame, tests: pd.DataFrame, outdir: Path) -> None:
    pub_rows = []
    full_auc = float(fold_df[fold_df.variant == "full"]["AUROC"].mean()) if (fold_df.variant == "full").any() else float("nan")
    for variant, g in fold_df.groupby("variant", sort=False):
        pub_rows.append({
            "variant": variant,
            "display_name": DISPLAY.get(variant, variant),
            "AUROC_mean": float(g["AUROC"].mean()),
            "AUROC_SD": float(g["AUROC"].std(ddof=1)),
            "PR_AUC_mean": float(g["PR_AUC"].mean()),
            "PR_AUC_SD": float(g["PR_AUC"].std(ddof=1)),
            "ACC_mean": float(g["ACC"].mean()),
            "ACC_SD": float(g["ACC"].std(ddof=1)),
            "F1_mean": float(g["F1"].mean()),
            "F1_SD": float(g["F1"].std(ddof=1)),
            "delta_AUROC_vs_full": float(full_auc - g["AUROC"].mean()),
        })
    pub = pd.DataFrame(pub_rows)

    if not tests.empty:
        auc_tests = tests[tests.metric == "AUROC"]
        pmap = dict(zip(auc_tests["variant"], auc_tests["holm_adjusted_p"]))
        dmap = dict(zip(auc_tests["variant"], auc_tests["mean_paired_difference_full_minus_ablation"]))
        pub["holm_adjusted_p_AUROC"] = pub["variant"].map(pmap)
        pub["paired_difference_full_minus_ablation"] = pub["variant"].map(dmap)
    pub.loc[pub["variant"] == "full", ["holm_adjusted_p_AUROC", "paired_difference_full_minus_ablation"]] = np.nan
    pub.to_csv(outdir / "publication_table_ablation.csv", index=False)

    resource = fold_df.groupby(["variant", "display_name"], sort=False).agg(
        trainable_parameters=("trainable_parameters", "mean"),
        checkpoint_size_mb=("checkpoint_size_mb", "mean"),
        training_active_min_mean=("training_active_minutes", "mean"),
        training_active_min_median=("training_active_minutes", "median"),
        training_wall_min_mean=("training_wall_minutes", "mean"),
        inference_ms_per_sample=("inference_ms_per_sample", "mean"),
        throughput_samples_per_second=("throughput_samples_per_second", "mean"),
        training_peak_gpu_memory_mb=("training_peak_gpu_memory_mb", "mean"),
        inference_peak_gpu_memory_mb=("inference_peak_gpu_memory_mb", "mean"),
        timing_anomaly_count=("timing_anomaly", "sum"),
    ).reset_index()
    resource.to_csv(outdir / "publication_table_resources.csv", index=False)


# -----------------------------------------------------------------------------
# Scalability summary
# -----------------------------------------------------------------------------
def build_scalability_summary(outdir: Path) -> None:
    frames: List[pd.DataFrame] = []
    for path in sorted(outdir.glob("scalability_full_fold*.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass
    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / "scalability_full_all_folds.csv", index=False)
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    if ok.empty:
        return

    pub = ok.groupby("batch_size").agg(
        n_folds=("fold", "nunique"),
        inference_ms_per_sample_mean=("inference_ms_per_sample", "mean"),
        inference_ms_per_sample_SD=("inference_ms_per_sample", "std"),
        throughput_samples_per_second_mean=("throughput_samples_per_second", "mean"),
        throughput_samples_per_second_SD=("throughput_samples_per_second", "std"),
        peak_gpu_memory_mb_mean=("peak_gpu_memory_mb", "mean"),
        peak_gpu_memory_mb_SD=("peak_gpu_memory_mb", "std"),
    ).reset_index()
    pub.to_csv(outdir / "publication_table_scalability.csv", index=False)


# -----------------------------------------------------------------------------
# Resume / orchestration
# -----------------------------------------------------------------------------

def load_existing_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "variant" not in df.columns or "fold" not in df.columns:
        return pd.DataFrame()
    return df


def upsert_metric_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    new = pd.DataFrame([row])
    if df.empty:
        return new
    keep = ~((df["variant"] == row["variant"]) & (df["fold"].astype(int) == int(row["fold"])))
    return pd.concat([df.loc[keep], new], ignore_index=True).sort_values(["variant", "fold"]).reset_index(drop=True)


def is_completed(existing: pd.DataFrame, variant: str, fold_number: int) -> bool:
    if existing.empty:
        return False
    return bool(((existing["variant"] == variant) & (existing["fold"].astype(int) == fold_number)).any())


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--output", default="./result/reviewer_clean")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--early-stopping-patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--lr-patience", type=int, default=10)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--inference-repeats", type=int, default=3)
    ap.add_argument("--scalability-batch-sizes", nargs="*", type=int, default=[32, 64, 128, 256, 512])
    ap.add_argument("--no-scalability", action="store_true", help="Skip full-model batch-size scalability benchmark")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--shuffle-train", action="store_true", help="Shuffle paired training samples each epoch")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--force", action="store_true", help="Rerun selected variant/fold even if already present")
    ap.add_argument(
        "--sanity",
        action="store_true",
        help="Quick validation: Full model, fold 1, 3 epochs, patience 3; does not represent paper results",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.sanity:
        args.variants = ["full"]
        args.folds = [1]
        args.max_epochs = 3
        args.early_stopping_patience = 3

    invalid_variants = [v for v in args.variants if v not in VARIANTS]
    if invalid_variants:
        raise ValueError(f"Unknown variants: {invalid_variants}; valid={VARIANTS}")
    invalid_folds = [f for f in args.folds if f not in [1, 2, 3, 4, 5]]
    if invalid_folds:
        raise ValueError(f"Invalid folds: {invalid_folds}; use 1..5")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    set_seed(args.seed)

    print("=" * 80)
    print("CODE_VERSION:", CODE_VERSION)
    print("Device:", device)
    if device.type == "xpu":
        print("Accelerator:", torch.xpu.get_device_name(device))
    elif device.type == "cuda":
        print("Accelerator:", torch.cuda.get_device_name(device))
    print("Output:", outdir.resolve())
    print("=" * 80)

    with open(outdir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(environment_dict(device), f, ensure_ascii=False, indent=2)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    dataset = DrugPairDataset(Path(args.data_root))
    with open(outdir / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(dataset.audit(), f, ensure_ascii=False, indent=2)

    outer_folds = make_outer_folds(len(dataset), args.seed, 5)
    write_outer_assignment(outdir, dataset.labels, outer_folds)

    # Save fold sizes and class counts for direct reviewer reporting.
    fold_audit_rows = []
    for fold_number in range(1, 6):
        tr, va, te = make_fold_indices(dataset.labels, outer_folds, fold_number, args.val_fraction, args.seed)
        fold_audit_rows.append({
            "fold": fold_number,
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "train_positive": int(dataset.labels[tr].sum()),
            "val_positive": int(dataset.labels[va].sum()),
            "test_positive": int(dataset.labels[te].sum()),
        })
    pd.DataFrame(fold_audit_rows).to_csv(outdir / "fold_audit.csv", index=False)

    metrics_path = outdir / "ablation_fold_metrics.csv"
    existing = load_existing_metrics(metrics_path)

    for variant in args.variants:
        for fold_number in args.folds:
            if is_completed(existing, variant, fold_number) and not args.force:
                print(f"SKIP completed: {variant} fold {fold_number}")
                continue

            train_idx, val_idx, test_idx = make_fold_indices(
                dataset.labels, outer_folds, fold_number, args.val_fraction, args.seed
            )
            row = train_variant(
                variant,
                fold_number,
                dataset,
                train_idx,
                val_idx,
                test_idx,
                args,
                outdir,
                device,
            )
            existing = upsert_metric_row(existing, row)
            existing.to_csv(metrics_path, index=False)

    fold_df = load_existing_metrics(metrics_path)
    if fold_df.empty:
        raise RuntimeError("No fold metrics available")

    summary = build_summary(fold_df)
    summary.to_csv(outdir / "ablation_summary.csv", index=False)

    tests_auc = paired_ablation_tests(fold_df, "AUROC")
    tests_pr = paired_ablation_tests(fold_df, "PR_AUC")
    tests = pd.concat([tests_auc, tests_pr], ignore_index=True)
    tests.to_csv(outdir / "ablation_paired_tests.csv", index=False)

    build_publication_tables(fold_df, tests, outdir)
    build_scalability_summary(outdir)

    print("\nFinished.")
    print("Primary outputs:")
    for name in [
        "ablation_fold_metrics.csv",
        "ablation_summary.csv",
        "ablation_paired_tests.csv",
        "publication_table_ablation.csv",
        "publication_table_resources.csv",
        "publication_table_scalability.csv",
        "outer_fold_assignments.csv",
        "fold_audit.csv",
        "data_audit.json",
        "environment.json",
        "run_config.json",
    ]:
        print(" -", outdir / name)
    if fold_df["timing_anomaly"].astype(bool).any():
        bad = fold_df.loc[fold_df["timing_anomaly"].astype(bool), ["variant", "fold"]]
        print("\nWARNING: timing anomalies detected. Rerun these rows before using resource times:")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
