#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MFMNet reviewer revision — FINAL CONTROLLED ABLATION RUNNER.

This file is derived from the supplied MFMNet_reviewer_clean_save_ckpt-pt.py
runner and keeps its strict five-fold protocol, inner-validation checkpoint
selection, fixed hyperparameters, paired drug loading, metrics, and resource
measurement path.

Controlled-ablation design
--------------------------
Primary confirmatory variants:
  full
  without_gcn
  without_gin
  without_gat
  no_router
  simple_concatenation

Supplementary branch diagnostics:
  gcn_only
  gin_only
  gat_only

Key controls
------------
1. The outer folds, inner validation split, fold seed, optimization budget,
   threshold, and checkpoint-selection rule are unchanged across variants.
2. For each fold, one full-model reference initialization is created.
   Every semantically shared parameter/buffer in every ablation is copied from
   that exact reference state before training.
3. Leave-one-branch-out variants preserve the full 384-dimensional CSAR layout.
   The removed branch occupies a zero-filled 128-dimensional slot, allowing the
   complete CSAR block and all surviving shared layers to start identically.
4. The primary Holm family contains exactly five prespecified controlled
   comparisons per metric: full vs w/o GCN, w/o GIN, w/o GAT, w/o CSAR, and
   simple concatenation. Single-branch comparisons are reported separately as
   supplementary descriptive analyses.
5. The validation-selected full models export branch-representation cosine
   similarities and CSAR gate summaries. If all three single-branch variants are
   run, sample-level prediction disagreement and unique-correct patterns are
   also exported as complementary-error diagnostics.
6. No code path forces the full model to outperform an ablation. No result is
   used to choose seed, fold, threshold, hyperparameter, or stopping rule.

Expected input files
--------------------
data/raw/all_ddi.csv
data/raw/all_drug_embedding.csv

The raw all_ddi.csv must contain drug1, drug2, label, and 954 cell-line
features beginning at column index 9. all_drug_embedding.csv must contain 1536
numeric columns: first 768 for drug1 and last 768 for drug2.
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


CODE_VERSION = "2026-08-15-controlled-ablation-masked-final-v1"
DEFAULT_SEED = 0
ATOM_DIM = 78
MAX_ATOMS = 100
DRUG_EMBED_DIM = 768
CELL_DIM = 954
GRAPH_DIM = 128

# Primary confirmatory ablations: each removes/replaces exactly one architectural
# component while retaining the same data split and training protocol.
PRIMARY_VARIANTS = [
    "full",
    "without_gcn",
    "without_gin",
    "without_gat",
    "no_router",
    "simple_concatenation",
]

# Supplementary single-branch models are included by default so that prediction-
# error complementarity among GCN/GIN/GAT can be quantified independently of
# the primary leave-one-branch-out comparisons. They are NOT included in the
# primary Holm family unless --include-singles-in-holm is explicitly requested.
SUPPLEMENTARY_VARIANTS = [
    "gcn_only",
    "gin_only",
    "gat_only",
]

VARIANTS = PRIMARY_VARIANTS + SUPPLEMENTARY_VARIANTS

DISPLAY = {
    "full": "Full MFMNet backbone",
    "without_gcn": "w/o GCN",
    "without_gin": "w/o GIN",
    "without_gat": "w/o GAT",
    "no_router": "w/o CSAR (mean fusion)",
    "simple_concatenation": "Simple concatenation + projection",
    "gcn_only": "GCN-only",
    "gin_only": "GIN-only",
    "gat_only": "GAT-only",
}

BRANCH_ORDER = ("gcn", "gin", "gat")


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
    valid_mask = torch.zeros(MAX_ATOMS, dtype=torch.bool)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = torch.from_numpy(atom_features(atom))
        valid_mask[i] = True

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

    data = Data(x=x, edge_index=edge_index)
    # Preserve the fixed 100-position representation, but explicitly mark real
    # atom positions so padded nodes never contribute to graph normalization or
    # graph-level pooling.
    data.valid_mask = valid_mask
    return data


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
            "padding_mask_used_in_graph_path": True,
            "pooling_rule": "mean_over_real_atoms_only",
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
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(input_dim, output_dim)

    def weights(self, features: torch.Tensor) -> torch.Tensor:
        return self.weight_net(features)

    def forward(self, features: torch.Tensor, return_weights: bool = False):
        weights = self.weights(features)
        fused = self.projection(features * weights)
        if return_weights:
            return fused, weights
        return fused


def variant_branches(variant: str) -> Tuple[str, ...]:
    if variant in {"full", "no_router", "simple_concatenation"}:
        return BRANCH_ORDER
    if variant == "without_gcn":
        return ("gin", "gat")
    if variant == "without_gin":
        return ("gcn", "gat")
    if variant == "without_gat":
        return ("gcn", "gin")
    if variant == "gcn_only":
        return ("gcn",)
    if variant == "gin_only":
        return ("gin",)
    if variant == "gat_only":
        return ("gat",)
    raise ValueError(f"Unknown variant: {variant}")


def variant_fusion_mode(variant: str) -> str:
    if variant in {"full", "without_gcn", "without_gin", "without_gat"}:
        return "adaptive"
    if variant == "no_router":
        return "mean"
    if variant == "simple_concatenation":
        return "concat_projection"
    if variant in {"gcn_only", "gin_only", "gat_only"}:
        return "identity"
    raise ValueError(f"Unknown variant: {variant}")


class MFMNetReviewerModel(nn.Module):
    VALID_VARIANTS = set(VARIANTS)

    def __init__(self, variant: str, dropout_rate: float = 0.3, out_channels: int = 2):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.dropout_rate = float(dropout_rate)
        self.relu = nn.ReLU()
        self.active_branches = variant_branches(variant)
        self.fusion_mode = variant_fusion_mode(variant)

        if "gcn" in self.active_branches:
            self.gcn_conv1 = GCNConv(ATOM_DIM, 512)
            self.gcn_conv2 = GCNConv(512, GRAPH_DIM)
        else:
            self.gcn_conv1 = None
            self.gcn_conv2 = None

        if "gin" in self.active_branches:
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

        if "gat" in self.active_branches:
            self.gat_conv1 = GATv2Conv(ATOM_DIM, 256, heads=2, dropout=dropout_rate)
            self.gat_conv2 = GATv2Conv(512, GRAPH_DIM, heads=1, dropout=dropout_rate)
        else:
            self.gat_conv1 = None
            self.gat_conv2 = None

        if self.fusion_mode == "adaptive":
            # Preserve the full 3 x 128 CSAR input layout for leave-one-branch-out
            # variants. A removed branch occupies a zero-filled slot, so the
            # surviving branches and the complete CSAR block remain structurally
            # identical to the full model.
            self.graph_fusion = AdaptiveFeatureFusion(
                GRAPH_DIM * len(BRANCH_ORDER), GRAPH_DIM
            )
        else:
            self.graph_fusion = None

        if self.fusion_mode == "concat_projection":
            self.simple_projection = nn.Linear(
                GRAPH_DIM * len(self.active_branches), GRAPH_DIM
            )
        else:
            self.simple_projection = None

        # Architecture-focused controlled ablation: drug and cell-line encoders
        # remain present in every variant.
        self.drug_embedding_processor = DrugEmbeddingProcessor(DRUG_EMBED_DIM, 512, 256)
        self.cell_encoder = CellLineEncoder(CELL_DIM, 512, 256)

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

    def _valid_mask(self, batch: Batch) -> torch.Tensor:
        if not hasattr(batch, "valid_mask"):
            raise RuntimeError(
                "Molecular graph is missing valid_mask; padded nodes must be masked "
                "for the controlled ablation."
            )
        return batch.valid_mask.view(-1).bool()

    def _zero_padding(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        mask = self._valid_mask(batch).to(dtype=x.dtype).unsqueeze(1)
        return x * mask

    def _masked_graphnorm(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Apply GraphNorm only to real atoms, then scatter back into 100-node layout."""
        mask = self._valid_mask(batch)
        idx = torch.nonzero(mask, as_tuple=False).view(-1)
        real_x = x.index_select(0, idx)
        real_batch = batch.batch.index_select(0, idx)
        normed = self.gnorm1(real_x, real_batch)
        out = torch.zeros_like(x)
        return out.index_copy(0, idx, normed)

    def _gcn(self, batch: Batch) -> torch.Tensor:
        h = self.gcn_conv1(batch.x, batch.edge_index)
        h = self._zero_padding(h, batch)
        h = self.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.gcn_conv2(h, batch.edge_index)
        return self._zero_padding(h, batch)

    def _gin(self, batch: Batch) -> torch.Tensor:
        h = self.gin_conv1(batch.x, batch.edge_index)
        h = self._zero_padding(h, batch)
        h = self.relu(h)
        h = self._masked_graphnorm(h, batch)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.gin_conv2(h, batch.edge_index)
        return self._zero_padding(h, batch)

    def _gat(self, batch: Batch) -> torch.Tensor:
        h = self.gat_conv1(batch.x, batch.edge_index)
        h = self._zero_padding(h, batch)
        h = self.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        h = self.gat_conv2(h, batch.edge_index)
        return self._zero_padding(h, batch)

    def _branch_node_features(self, batch: Batch) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name in self.active_branches:
            if name == "gcn":
                out[name] = self._gcn(batch)
            elif name == "gin":
                out[name] = self._gin(batch)
            elif name == "gat":
                out[name] = self._gat(batch)
            else:
                raise RuntimeError(name)
        return out

    def _fuse_nodes(
        self,
        branch_features: Dict[str, torch.Tensor],
        return_weights: bool = False,
    ):
        ordered_active = [branch_features[name] for name in self.active_branches]
        if self.fusion_mode == "identity":
            fused = ordered_active[0]
            weights = None
        elif self.fusion_mode == "mean":
            fused = torch.stack(ordered_active, dim=0).mean(dim=0)
            weights = None
        elif self.fusion_mode == "concat_projection":
            multi = torch.cat(ordered_active, dim=1)
            fused = self.simple_projection(multi)
            weights = None
        elif self.fusion_mode == "adaptive":
            # Full and leave-one-out models share an identical 384-d CSAR.
            # Missing branch slots are zeros rather than changing the router
            # architecture, isolating the branch-removal intervention.
            template = ordered_active[0]
            slotted = [
                branch_features[name] if name in branch_features
                else torch.zeros_like(template)
                for name in BRANCH_ORDER
            ]
            multi = torch.cat(slotted, dim=1)
            fused, weights = self.graph_fusion(multi, return_weights=True)
        else:
            raise RuntimeError(self.fusion_mode)
        if return_weights:
            return fused, weights
        return fused

    def _reshape_nodes(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        expected_nodes = batch_size * MAX_ATOMS
        if x.shape[0] != expected_nodes:
            raise RuntimeError(
                f"Expected {expected_nodes} nodes ({batch_size} x {MAX_ATOMS}), got {x.shape[0]}"
            )
        return x.reshape(batch_size, MAX_ATOMS, GRAPH_DIM)

    def _graph_features(self, batch: Batch, batch_size: int) -> torch.Tensor:
        branch_features = self._branch_node_features(batch)
        fused = self._fuse_nodes(branch_features)
        return self._reshape_nodes(fused, batch_size)

    def _masked_mean_pool(
        self,
        node_features: torch.Tensor,
        graph_batch: Batch,
        batch_size: int,
    ) -> torch.Tensor:
        """Mean-pool only real atom positions; zero-padded positions are excluded."""
        if node_features.shape[:2] != (batch_size, MAX_ATOMS):
            raise RuntimeError(
                f"Unexpected node feature shape for masked pooling: {tuple(node_features.shape)}"
            )
        mask = self._valid_mask(graph_batch).reshape(batch_size, MAX_ATOMS)
        mask_f = mask.to(dtype=node_features.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (node_features * mask_f).sum(dim=1) / denom

    def forward(self, batch: PairBatch) -> torch.Tensor:
        batch_size = int(batch.label.shape[0])
        emb1 = self.drug_embedding_processor(batch.left_embedding)
        emb2 = self.drug_embedding_processor(batch.right_embedding)
        cell = self.cell_encoder(batch.cell)

        graph1_nodes = self._graph_features(batch.left_graph, batch_size)
        graph2_nodes = self._graph_features(batch.right_graph, batch_size)
        graph1 = self._masked_mean_pool(graph1_nodes, batch.left_graph, batch_size)
        graph2 = self._masked_mean_pool(graph2_nodes, batch.right_graph, batch_size)

        combined = torch.cat([emb1, emb2, cell, graph1, graph2], dim=1)
        return self.classifier(combined)

    def forward_with_diagnostics(self, batch: PairBatch):
        """
        Full-model-only diagnostic forward pass.

        This does not change model predictions. It exposes:
          - pooled GCN/GIN/GAT representations for each drug;
          - feature-wise CSAR gate weights aggregated by branch segment.
        """
        if self.variant != "full":
            raise RuntimeError("forward_with_diagnostics is defined for variant='full' only")

        batch_size = int(batch.label.shape[0])
        emb1 = self.drug_embedding_processor(batch.left_embedding)
        emb2 = self.drug_embedding_processor(batch.right_embedding)
        cell = self.cell_encoder(batch.cell)

        diag = {}
        pooled_fused = []
        for side, graph in (("drug1", batch.left_graph), ("drug2", batch.right_graph)):
            branches = self._branch_node_features(graph)
            fused, weights = self._fuse_nodes(branches, return_weights=True)

            branch_pooled = {
                name: self._masked_mean_pool(
                    self._reshape_nodes(branches[name], batch_size),
                    graph,
                    batch_size,
                )
                for name in BRANCH_ORDER
            }
            diag[f"{side}_branch_pooled"] = branch_pooled

            if weights is None or weights.shape[1] != GRAPH_DIM * 3:
                raise RuntimeError("Unexpected CSAR weight shape in full model")
            w = weights.reshape(batch_size, MAX_ATOMS, 3, GRAPH_DIM)
            atom_mask = self._valid_mask(graph).reshape(batch_size, MAX_ATOMS)

            gate_means = []
            gate_sds = []
            for bi in range(batch_size):
                real_w = w[bi, atom_mask[bi], :, :]  # n_real_atoms x 3 x 128
                gate_means.append(real_w.mean(dim=(0, 2)))
                gate_sds.append(real_w.std(dim=(0, 2), unbiased=False))
            diag[f"{side}_gate_mean"] = torch.stack(gate_means, dim=0)
            diag[f"{side}_gate_sd"] = torch.stack(gate_sds, dim=0)

            pooled_fused.append(
                self._masked_mean_pool(
                    self._reshape_nodes(fused, batch_size),
                    graph,
                    batch_size,
                )
            )

        graph1, graph2 = pooled_fused
        combined = torch.cat([emb1, emb2, cell, graph1, graph2], dim=1)
        logits = self.classifier(combined)
        return logits, diag


def make_full_reference_state(fold_seed: int, dropout_rate: float) -> Dict[str, torch.Tensor]:
    """
    Create one full-model initialization per fold. Every ablation copies all
    semantically shared parameters with identical names/shapes from this state.
    """
    set_seed(fold_seed)
    ref = MFMNetReviewerModel("full", dropout_rate=dropout_rate).cpu()
    return {k: v.detach().clone() for k, v in ref.state_dict().items()}


def apply_controlled_initialization(
    model: MFMNetReviewerModel,
    reference_state: Dict[str, torch.Tensor],
) -> dict:
    """
    Copy shared parameters/buffers from the same-fold full reference.

    Leave-one-branch-out variants preserve the same 384-d CSAR input layout
    using a zero-filled slot for the removed branch. Therefore the complete
    CSAR block, surviving graph branches, drug encoder, cell encoder, and
    classifier can all be copied exactly from the same-fold full reference.
    """
    own = model.state_dict()
    copied = []
    skipped_shape = []
    skipped_architecture_specific = []
    missing_from_reference = []

    for key, value in own.items():
        if key not in reference_state:
            missing_from_reference.append(key)
            continue
        ref = reference_state[key]
        if tuple(ref.shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        own[key] = ref.detach().clone()
        copied.append(key)

    model.load_state_dict(own, strict=True)

    # Verify every copied tensor is bitwise identical to the reference.
    verified = True
    now = model.state_dict()
    for key in copied:
        if not torch.equal(now[key].cpu(), reference_state[key].cpu()):
            verified = False
            break

    return {
        "variant": model.variant,
        "n_state_entries": len(own),
        "n_copied_from_full_reference": len(copied),
        "n_architecture_specific_not_copied": len(skipped_architecture_specific),
        "n_shape_mismatch_not_copied": len(skipped_shape),
        "n_missing_from_reference": len(missing_from_reference),
        "controlled_shared_initialization_verified": bool(verified),
        "copied_keys": copied,
        "architecture_specific_keys": skipped_architecture_specific,
        "shape_mismatch_keys": skipped_shape,
        "missing_reference_keys": missing_from_reference,
    }


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



def _cosine_np(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return F.cosine_similarity(a, b, dim=1, eps=1e-8).detach().cpu().numpy()


def export_full_diagnostics(
    model: MFMNetReviewerModel,
    device: torch.device,
    loader: TorchDataLoader,
    fold_number: int,
    outdir: Path,
) -> dict:
    """
    Export descriptive evidence for branch representation complementarity and
    CSAR routing behavior from the validation-selected full model.

    These diagnostics are never used for checkpoint, seed, or hyperparameter
    selection.
    """
    model.eval()
    rows: List[dict] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, diag = model.forward_with_diagnostics(batch)
            score = torch.softmax(logits, dim=1)[:, 1]
            pred = (score >= 0.5).long()

            d1 = diag["drug1_branch_pooled"]
            d2 = diag["drug2_branch_pooled"]

            cos = {}
            for side_name, reps in (("drug1", d1), ("drug2", d2)):
                for a, b in (("gcn", "gin"), ("gcn", "gat"), ("gin", "gat")):
                    cos[f"{side_name}_{a}_{b}_cosine"] = _cosine_np(reps[a], reps[b])

            gate_values = {}
            for side_name in ("drug1", "drug2"):
                gm = diag[f"{side_name}_gate_mean"].detach().cpu().numpy()
                gs = diag[f"{side_name}_gate_sd"].detach().cpu().numpy()
                for j, branch in enumerate(BRANCH_ORDER):
                    gate_values[f"{side_name}_{branch}_gate_mean"] = gm[:, j]
                    gate_values[f"{side_name}_{branch}_gate_sd"] = gs[:, j]

            idx = batch.index.detach().cpu().numpy()
            y = batch.label.detach().cpu().numpy()
            sc = score.detach().cpu().numpy()
            pr = pred.detach().cpu().numpy()

            for i in range(len(idx)):
                row = {
                    "fold": int(fold_number),
                    "sample_index": int(idx[i]),
                    "label": int(y[i]),
                    "score": float(sc[i]),
                    "predicted_label": int(pr[i]),
                }
                for key, arr in cos.items():
                    row[key] = float(arr[i])
                for key, arr in gate_values.items():
                    row[key] = float(arr[i])

                row["mean_gcn_gin_cosine"] = float(
                    0.5 * (
                        row["drug1_gcn_gin_cosine"] +
                        row["drug2_gcn_gin_cosine"]
                    )
                )
                row["mean_gcn_gat_cosine"] = float(
                    0.5 * (
                        row["drug1_gcn_gat_cosine"] +
                        row["drug2_gcn_gat_cosine"]
                    )
                )
                row["mean_gin_gat_cosine"] = float(
                    0.5 * (
                        row["drug1_gin_gat_cosine"] +
                        row["drug2_gin_gat_cosine"]
                    )
                )
                for branch in BRANCH_ORDER:
                    row[f"{branch}_gate_mean_across_drugs"] = float(
                        0.5 * (
                            row[f"drug1_{branch}_gate_mean"] +
                            row[f"drug2_{branch}_gate_mean"]
                        )
                    )
                rows.append(row)

    df = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
    df.to_csv(outdir / f"full_diagnostics_fold{fold_number}.csv", index=False)

    summary = {
        "fold": int(fold_number),
        "n_samples": int(len(df)),
    }
    for col in [
        "mean_gcn_gin_cosine",
        "mean_gcn_gat_cosine",
        "mean_gin_gat_cosine",
        "gcn_gate_mean_across_drugs",
        "gin_gate_mean_across_drugs",
        "gat_gate_mean_across_drugs",
    ]:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_SD"] = float(df[col].std(ddof=1))
        summary[f"{col}_min"] = float(df[col].min())
        summary[f"{col}_max"] = float(df[col].max())

    return summary


def build_full_diagnostic_summary(outdir: Path) -> None:
    frames = []
    for p in sorted(outdir.glob("full_diagnostics_fold*.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / "full_diagnostics_all_folds.csv", index=False)

    rows = []
    for col in [
        "mean_gcn_gin_cosine",
        "mean_gcn_gat_cosine",
        "mean_gin_gat_cosine",
        "gcn_gate_mean_across_drugs",
        "gin_gate_mean_across_drugs",
        "gat_gate_mean_across_drugs",
    ]:
        rows.append({
            "quantity": col,
            "n": int(df[col].notna().sum()),
            "mean": float(df[col].mean()),
            "SD": float(df[col].std(ddof=1)),
            "median": float(df[col].median()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        })
    pd.DataFrame(rows).to_csv(
        outdir / "publication_table_branch_csar_diagnostics.csv", index=False
    )


def build_single_branch_error_complementarity(outdir: Path) -> None:
    """
    If GCN-only, GIN-only and GAT-only predictions were run, quantify whether
    their errors differ on the same outer-test records. This is descriptive and
    is not used to select any model.
    """
    detail_frames = []
    summary_rows = []

    for fold in range(1, 6):
        paths = {
            b: outdir / f"pred_{b}_only_fold{fold}.csv"
            for b in BRANCH_ORDER
        }
        if not all(p.exists() for p in paths.values()):
            continue

        merged = None
        for branch, path in paths.items():
            d = pd.read_csv(path)[
                ["sample_index", "label", "score", "predicted_label"]
            ].copy()
            d = d.rename(columns={
                "score": f"{branch}_score",
                "predicted_label": f"{branch}_pred",
            })
            if merged is None:
                merged = d
            else:
                merged = merged.merge(
                    d,
                    on=["sample_index", "label"],
                    how="inner",
                    validate="one_to_one",
                )

        if merged is None or merged.empty:
            continue

        for branch in BRANCH_ORDER:
            merged[f"{branch}_correct"] = (
                merged[f"{branch}_pred"].astype(int) == merged["label"].astype(int)
            )

        correct_cols = [f"{b}_correct" for b in BRANCH_ORDER]
        merged["n_branches_correct"] = merged[correct_cols].sum(axis=1)
        merged["any_branch_correct"] = merged["n_branches_correct"] > 0
        merged["all_branches_correct"] = merged["n_branches_correct"] == 3

        for branch in BRANCH_ORDER:
            other = [f"{b}_correct" for b in BRANCH_ORDER if b != branch]
            merged[f"{branch}_unique_correct"] = (
                merged[f"{branch}_correct"] & ~merged[other].any(axis=1)
            )

        for a, b in (("gcn", "gin"), ("gcn", "gat"), ("gin", "gat")):
            merged[f"{a}_{b}_prediction_disagreement"] = (
                merged[f"{a}_pred"].astype(int) != merged[f"{b}_pred"].astype(int)
            )

        merged.insert(0, "fold", fold)
        detail_frames.append(merged)

        row = {
            "fold": fold,
            "n_samples": int(len(merged)),
            "gcn_accuracy": float(merged["gcn_correct"].mean()),
            "gin_accuracy": float(merged["gin_correct"].mean()),
            "gat_accuracy": float(merged["gat_correct"].mean()),
            "oracle_any_branch_accuracy": float(merged["any_branch_correct"].mean()),
            "all_three_correct_fraction": float(merged["all_branches_correct"].mean()),
        }
        for branch in BRANCH_ORDER:
            row[f"{branch}_unique_correct_count"] = int(
                merged[f"{branch}_unique_correct"].sum()
            )
            row[f"{branch}_unique_correct_fraction"] = float(
                merged[f"{branch}_unique_correct"].mean()
            )
        for a, b in (("gcn", "gin"), ("gcn", "gat"), ("gin", "gat")):
            row[f"{a}_{b}_prediction_disagreement_rate"] = float(
                merged[f"{a}_{b}_prediction_disagreement"].mean()
            )
        summary_rows.append(row)

    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(
            outdir / "single_branch_error_complementarity_all_folds.csv", index=False
        )
    if summary_rows:
        fold_summary = pd.DataFrame(summary_rows)
        fold_summary.to_csv(
            outdir / "single_branch_error_complementarity_by_fold.csv", index=False
        )
        aggregate = {
            "n_folds": int(len(fold_summary)),
        }
        for col in fold_summary.columns:
            if col in {"fold", "n_samples"}:
                continue
            vals = fold_summary[col].to_numpy(dtype=float)
            aggregate[f"{col}_mean"] = float(np.mean(vals))
            aggregate[f"{col}_SD"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        (outdir / "single_branch_error_complementarity_summary.json").write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False),
            encoding="utf-8",
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

    # Controlled initialization: create the same full reference initialization
    # for this fold, instantiate the requested variant under the same fold seed,
    # then overwrite every semantically shared tensor with the reference value.
    reference_state = make_full_reference_state(fold_seed, args.dropout)
    set_seed(fold_seed)
    model = MFMNetReviewerModel(variant=variant, dropout_rate=args.dropout)
    init_manifest = apply_controlled_initialization(model, reference_state)
    init_manifest["fold"] = int(fold_number)
    init_manifest["fold_seed"] = int(fold_seed)
    if not init_manifest["controlled_shared_initialization_verified"]:
        raise RuntimeError(
            f"Controlled initialization verification failed for {variant} fold {fold_number}"
        )
    init_dir = outdir / "initialization_manifests"
    init_dir.mkdir(parents=True, exist_ok=True)
    (init_dir / f"init_{variant}_fold{fold_number}.json").write_text(
        json.dumps(init_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    model = model.to(device)

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

    # Permanently save the validation-selected FULL-model checkpoint for external validation.
    # Save tensors on CPU so the checkpoint is portable across CPU/CUDA/XPU environments.
    if variant == "full":
        ckpt_dir = outdir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"mfmnet_full_fold{fold_number}.pt"
        state_dict_cpu = {k: v.detach().cpu() for k, v in best_state.items()}
        torch.save(
            {
                "state_dict": state_dict_cpu,
                "variant": variant,
                "fold": int(fold_number),
                "best_epoch": int(best_epoch),
                "best_val_AUROC": float(best_val_auc),
                "seed": int(fold_seed),
            },
            ckpt_path,
        )
        print(f"Saved external-validation checkpoint: {ckpt_path.resolve()}")

    yt, st, pt, it = predict(model, device, test_loader)
    m = metric_dict(yt, st, pt)

    diagnostic_summary = {}
    if variant == "full" and not args.no_diagnostics:
        diagnostic_summary = export_full_diagnostics(
            model=model,
            device=device,
            loader=test_loader,
            fold_number=fold_number,
            outdir=outdir,
        )

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
        "controlled_shared_initialization_verified": bool(
            init_manifest["controlled_shared_initialization_verified"]
        ),
        "n_state_entries_copied_from_full_reference": int(
            init_manifest["n_copied_from_full_reference"]
        ),
        **{
            f"diagnostic_{k}": v
            for k, v in diagnostic_summary.items()
            if k not in {"fold", "n_samples"}
        },
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


def paired_ablation_tests(
    fold_df: pd.DataFrame,
    metric: str,
    variants: Sequence[str],
    holm: bool,
    family: str,
) -> pd.DataFrame:
    full = fold_df[fold_df["variant"] == "full"].sort_values("fold")
    rows: List[dict] = []
    for variant in variants:
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
            "family": family,
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
        if holm:
            adjusted = holm_adjust([r["raw_p"] for r in rows])
            for r, p_adj in zip(rows, adjusted):
                r["holm_adjusted_p"] = float(p_adj)
        else:
            for r in rows:
                r["holm_adjusted_p"] = float("nan")
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


def build_publication_tables(
    fold_df: pd.DataFrame,
    primary_tests: pd.DataFrame,
    outdir: Path,
) -> None:
    pub_rows = []
    full_auc = (
        float(fold_df[fold_df.variant == "full"]["AUROC"].mean())
        if (fold_df.variant == "full").any() else float("nan")
    )
    full_pr = (
        float(fold_df[fold_df.variant == "full"]["PR_AUC"].mean())
        if (fold_df.variant == "full").any() else float("nan")
    )

    for variant, g in fold_df.groupby("variant", sort=False):
        pub_rows.append({
            "variant": variant,
            "display_name": DISPLAY.get(variant, variant),
            "analysis_role": (
                "primary_controlled_ablation"
                if variant in PRIMARY_VARIANTS else
                "supplementary_single_branch"
            ),
            "AUROC_mean": float(g["AUROC"].mean()),
            "AUROC_SD": float(g["AUROC"].std(ddof=1)),
            "PR_AUC_mean": float(g["PR_AUC"].mean()),
            "PR_AUC_SD": float(g["PR_AUC"].std(ddof=1)),
            "ACC_mean": float(g["ACC"].mean()),
            "ACC_SD": float(g["ACC"].std(ddof=1)),
            "F1_mean": float(g["F1"].mean()),
            "F1_SD": float(g["F1"].std(ddof=1)),
            "delta_AUROC_full_minus_variant": float(full_auc - g["AUROC"].mean()),
            "delta_PR_AUC_full_minus_variant": float(full_pr - g["PR_AUC"].mean()),
        })
    pub = pd.DataFrame(pub_rows)

    if not primary_tests.empty:
        for metric, prefix in (("AUROC", "AUROC"), ("PR_AUC", "PR_AUC")):
            tt = primary_tests[primary_tests.metric == metric]
            pmap = dict(zip(tt["variant"], tt["holm_adjusted_p"]))
            dmap = dict(zip(
                tt["variant"],
                tt["mean_paired_difference_full_minus_ablation"]
            ))
            lmap = dict(zip(tt["variant"], tt["ci95_lower"]))
            umap = dict(zip(tt["variant"], tt["ci95_upper"]))
            pub[f"{prefix}_holm_adjusted_p"] = pub["variant"].map(pmap)
            pub[f"{prefix}_paired_difference"] = pub["variant"].map(dmap)
            pub[f"{prefix}_difference_CI95_lower"] = pub["variant"].map(lmap)
            pub[f"{prefix}_difference_CI95_upper"] = pub["variant"].map(umap)

    pub.to_csv(outdir / "publication_table_controlled_ablation.csv", index=False)

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


def build_initialization_manifest_summary(outdir: Path) -> None:
    init_dir = outdir / "initialization_manifests"
    if not init_dir.exists():
        return
    rows = []
    for p in sorted(init_dir.glob("init_*_fold*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "file": p.name,
            "variant": d.get("variant"),
            "fold": d.get("fold"),
            "fold_seed": d.get("fold_seed"),
            "n_state_entries": d.get("n_state_entries"),
            "n_copied_from_full_reference": d.get("n_copied_from_full_reference"),
            "n_architecture_specific_not_copied": d.get("n_architecture_specific_not_copied"),
            "n_shape_mismatch_not_copied": d.get("n_shape_mismatch_not_copied"),
            "n_missing_from_reference": d.get("n_missing_from_reference"),
            "controlled_shared_initialization_verified": d.get(
                "controlled_shared_initialization_verified"
            ),
        })
    if rows:
        pd.DataFrame(rows).to_csv(
            outdir / "controlled_initialization_summary.csv", index=False
        )



def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--output", default="./result/controlled_ablation_final_20260815")
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
    ap.add_argument("--variants", nargs="*", default=VARIANTS, help="Default runs 6 primary controlled ablations plus 3 supplementary single-branch models")
    ap.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--force", action="store_true", help="Rerun selected variant/fold even if already present")
    ap.add_argument("--no-diagnostics", action="store_true", help="Skip full-model branch-similarity and CSAR-gate diagnostics")
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

    # Primary confirmatory Holm families contain only the five controlled
    # architectural interventions defined before evaluation.
    primary_ablation_variants = PRIMARY_VARIANTS[1:]
    primary_auc = paired_ablation_tests(
        fold_df, "AUROC", primary_ablation_variants,
        holm=True, family="primary_controlled_ablation"
    )
    primary_pr = paired_ablation_tests(
        fold_df, "PR_AUC", primary_ablation_variants,
        holm=True, family="primary_controlled_ablation"
    )
    primary_tests = pd.concat([primary_auc, primary_pr], ignore_index=True)
    primary_tests.to_csv(
        outdir / "controlled_ablation_primary_paired_tests.csv", index=False
    )

    # Supplementary single-branch comparisons are descriptive diagnostics and
    # are kept separate from the confirmatory Holm family.
    single_auc = paired_ablation_tests(
        fold_df, "AUROC", SUPPLEMENTARY_VARIANTS,
        holm=False, family="supplementary_single_branch"
    )
    single_pr = paired_ablation_tests(
        fold_df, "PR_AUC", SUPPLEMENTARY_VARIANTS,
        holm=False, family="supplementary_single_branch"
    )
    single_tests = pd.concat([single_auc, single_pr], ignore_index=True)
    single_tests.to_csv(
        outdir / "single_branch_descriptive_paired_tests.csv", index=False
    )

    build_publication_tables(fold_df, primary_tests, outdir)
    build_scalability_summary(outdir)
    build_initialization_manifest_summary(outdir)
    if not args.no_diagnostics:
        build_full_diagnostic_summary(outdir)
    build_single_branch_error_complementarity(outdir)

    print("\nFinished.")
    print("Primary outputs:")
    for name in [
        "ablation_fold_metrics.csv",
        "ablation_summary.csv",
        "controlled_ablation_primary_paired_tests.csv",
        "single_branch_descriptive_paired_tests.csv",
        "publication_table_controlled_ablation.csv",
        "publication_table_resources.csv",
        "publication_table_branch_csar_diagnostics.csv",
        "single_branch_error_complementarity_by_fold.csv",
        "single_branch_error_complementarity_summary.json",
        "controlled_initialization_summary.csv",
        "publication_table_scalability.csv",
        "outer_fold_assignments.csv",
        "fold_audit.csv",
        "data_audit.json",
        "environment.json",
        "run_config.json",
    ]:
        p = outdir / name
        if p.exists():
            print(" -", p)
    if fold_df["timing_anomaly"].astype(bool).any():
        bad = fold_df.loc[fold_df["timing_anomaly"].astype(bool), ["variant", "fold"]]
        print("\nWARNING: timing anomalies detected. Rerun these rows before using resource times:")
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
