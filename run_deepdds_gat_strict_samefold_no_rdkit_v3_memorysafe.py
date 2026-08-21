#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict same-fold DeepDDS-GAT baseline for the MFMNet reviewer rerun.

IMPORTANT
---------
This is a PAPER-ALIGNED DeepDDS-GAT reproduction, not a claim of byte-for-byte
reproduction of the historical GitHub code.

Fair-comparison protocol:
- EXACT same outer folds as MFMNet_reviewer_clean.py, read from:
      ./result/full_check/outer_fold_assignments.csv
- Inner validation split reconstructed with the same rule:
      train_test_split(..., test_size=0.10,
                       random_state=seed + fold_number,
                       stratify=labels[outer_train])
- Epoch/checkpoint selected ONLY by inner-validation AUROC.
- Outer test evaluated ONCE after checkpoint selection.
- No outer-test tuning.

Input:
- Existing ONeil_Drug1.pt / ONeil_Drug2.pt generated previously.
- Atomic node feature dimension = 78.
- Cell-line feature dimension = 954.
- loader.py is NOT imported, so RDKit is NOT required during this rerun.

GAT reproduction choices:
- Two GAT layers corresponding to the reported hidden widths 1024 and 512.
- The first GAT layer uses 10 attention heads and concatenates their outputs.
- The second GAT layer is a single-head output layer that maps the concatenated
  first-layer representation to 512 dimensions.
- ELU after the first GAT layer and ReLU after the second.

Memory-safe execution:
- The MODEL ARCHITECTURE is unchanged from v2.
- Training uses micro-batches plus exact sample-weighted gradient accumulation.
  The default effective batch size remains 64, while only 16 samples are held
  in each training micro-batch.
- Validation/test use a smaller batch because evaluation does not affect the
  trained parameters.
- Best checkpoints are copied to CPU rather than kept on XPU/CUDA.
- No mixed precision and no change to folds, optimizer, learning rate, dropout,
  early stopping, or model dimensions.
- Global max pooling over non-padded atom nodes.
- Cell-line MLP: 954 -> 2048 -> 512.
- Classifier: 1536 -> 1024 -> 512 -> 128 -> 2.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.data import InMemoryDataset


class PreloadedDrugDataset(InMemoryDataset):
    """
    Load the already-generated PyG processed dataset directly from
    data/processed/ONeil_Drug1.pt or ONeil_Drug2.pt.

    This deliberately avoids importing loader.py, so RDKit is not required
    during the DeepDDS baseline rerun.
    """
    def __init__(self, root: str, dataset: str):
        self.dataset = dataset
        super().__init__(root=root, transform=None, pre_transform=None)

        pt_path = Path(self.processed_paths[0])
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Missing preprocessed graph file: {pt_path}. "
                "This no-RDKit runner requires the existing processed .pt files."
            )

        try:
            loaded = torch.load(pt_path, weights_only=False)
        except TypeError:
            loaded = torch.load(pt_path)

        if not (isinstance(loaded, tuple) and len(loaded) >= 2):
            raise ValueError(
                f"Unexpected processed dataset format in {pt_path}: "
                f"{type(loaded).__name__}"
            )

        self.data, self.slices = loaded[0], loaded[1]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        pass

    def process(self):
        raise RuntimeError(
            "Preprocessed graph files are required. "
            "Do not regenerate them in this runner."
        )


N_FOLDS = 5
ATOM_DIM = 78
CELL_DIM = 954


def choose_device(requested: str) -> torch.device:
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
        raise ValueError(f"Unsupported device: {requested}")

    if xpu_ok:
        return torch.device("xpu:0")
    if cuda_ok:
        return torch.device("cuda:0")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()


def reset_peak_memory(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "reset_peak_memory_stats"):
            torch.xpu.reset_peak_memory_stats(device)
    except Exception:
        pass


def peak_memory_mb(device: torch.device) -> float:
    try:
        if device.type == "cuda":
            return float(torch.cuda.max_memory_allocated(device) / 1024**2)
        if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "max_memory_allocated"):
            return float(torch.xpu.max_memory_allocated(device) / 1024**2)
    except Exception:
        pass
    return float("nan")


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


def pr_auc_value(y_true: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(auc(recall, precision))


def metric_dict(y_true: np.ndarray, score: np.ndarray) -> dict:
    pred = (score > 0.5).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, score)),
        "PR_AUC": pr_auc_value(y_true, score),
        "ACC": float(accuracy_score(y_true, pred)),
        "BACC": float(balanced_accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "TPR": float(recall_score(y_true, pred, zero_division=0)),
        "KAPPA": float(cohen_kappa_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
    }


class DeepDDSGATPaperAligned(nn.Module):
    """
    Paper-aligned DeepDDS-GAT reproduction used for the strict same-fold rerun.

    Drug graph encoder:
      GATConv 78 -> (1024 x 10 heads), concat=True
      ELU
      dropout
      GATConv (1024 x 10) -> 512, 1 head, concat=False
      ReLU
      global max pooling over non-padded atoms

    Cell-line encoder:
      954 -> 2048 -> 512

    Classifier:
      drug1(512) + drug2(512) + cell(512) = 1536
      1536 -> 1024 -> 512 -> 128 -> 2

    The GAT weights are shared between the two drugs.

    Rationale:
    The DeepDDS paper explicitly defines multi-head GAT output by concatenating
    the head outputs. Therefore, the first layer concatenates the 10 heads.
    A single-head second layer then produces the reported 512-dimensional
    graph-node representation. This follows the standard two-layer GAT pattern
    and avoids averaging the first-layer heads before the second layer.
    """

    def __init__(self, dropout: float = 0.2, heads: int = 10):
        super().__init__()
        self.dropout = float(dropout)
        self.heads = int(heads)

        self.gat1 = GATConv(
            ATOM_DIM,
            1024,
            heads=self.heads,
            concat=True,
            dropout=self.dropout,
            add_self_loops=True,
        )
        self.gat2 = GATConv(
            1024 * self.heads,
            512,
            heads=1,
            concat=False,
            dropout=self.dropout,
            add_self_loops=True,
        )

        self.cell_mlp = nn.Sequential(
            nn.Linear(CELL_DIM, 2048),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 512 + 512, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 2),
        )

    def encode_drug(self, data):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gat2(x, data.edge_index))

        if hasattr(data, "mask"):
            valid = ~data.mask.bool()
            x_pool = global_max_pool(x[valid], data.batch[valid])
        else:
            x_pool = global_max_pool(x, data.batch)

        return x_pool

    def forward(self, left, right):
        drug1 = self.encode_drug(left)
        drug2 = self.encode_drug(right)

        cell = left.ccle_embedding
        cell = F.normalize(cell, p=2, dim=1)
        cell = self.cell_mlp(cell)

        z = torch.cat([drug1, drug2, cell], dim=1)
        return self.classifier(z)

def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_loader(dataset, indices: Sequence[int], batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)

    # InMemoryDataset.index_select keeps a lightweight index view instead of
    # regenerating molecular graphs. num_workers=0 and pin_memory=False avoid
    # extra host-memory copies on Windows/XPU.
    subset = dataset[list(map(int, indices))]
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )


def load_outer_folds(path: Path, n: int) -> list[np.ndarray]:
    df = pd.read_csv(path)
    required = {"sample_index", "outer_fold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if len(df) != n or df["sample_index"].nunique() != n:
        raise ValueError("Outer fold assignment must cover every sample exactly once.")

    folds = []
    for fold in range(1, N_FOLDS + 1):
        idx = df.loc[df["outer_fold"].astype(int) == fold, "sample_index"].to_numpy(dtype=int)
        folds.append(idx)

    covered = np.concatenate(folds)
    if len(covered) != n or len(np.unique(covered)) != n:
        raise ValueError("Outer test folds do not cover all samples exactly once.")
    return folds


def make_inner_split(labels, test_idx, fold_number, val_fraction, seed):
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
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


@torch.no_grad()
def predict(model, device, loader1, loader2):
    model.eval()
    ys, scores, indices = [], [], []

    for left, right in zip(loader1, loader2):
        left = left.to(device)
        right = right.to(device)
        logits = model(left, right)
        prob = F.softmax(logits, dim=1)[:, 1]

        ys.append(left.y.view(-1).detach().cpu().numpy().astype(int))
        scores.append(prob.detach().cpu().numpy().astype(float))

    return np.concatenate(ys), np.concatenate(scores)


def train_fold(
    fold_number,
    train_idx,
    val_idx,
    test_idx,
    dataset1,
    dataset2,
    labels,
    device,
    args,
    outdir,
):
    fold_seed = args.seed + 1000 * fold_number
    set_seed(fold_seed)

    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError(
            f"--batch-size ({args.batch_size}) must be divisible by "
            f"--micro-batch-size ({args.micro_batch_size})"
        )
    if args.micro_batch_size > args.batch_size:
        raise ValueError("--micro-batch-size cannot exceed --batch-size")

    train1 = make_loader(
        dataset1, train_idx, args.micro_batch_size, args.shuffle_train, fold_seed + 1
    )
    train2 = make_loader(
        dataset2, train_idx, args.micro_batch_size, args.shuffle_train, fold_seed + 1
    )
    val1 = make_loader(dataset1, val_idx, args.eval_batch_size, False, fold_seed + 2)
    val2 = make_loader(dataset2, val_idx, args.eval_batch_size, False, fold_seed + 2)
    test1 = make_loader(dataset1, test_idx, args.eval_batch_size, False, fold_seed + 3)
    test2 = make_loader(dataset2, test_idx, args.eval_batch_size, False, fold_seed + 3)

    model = DeepDDSGATPaperAligned(dropout=args.dropout, heads=args.heads).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
    )

    best_val_auc = -np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0
    epoch_rows = []
    epoch_seconds = []

    empty_cache(device)
    reset_peak_memory(device)
    sync(device)
    wall_start = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        sync(device)
        epoch_start = time.perf_counter()

        # Memory-safe gradient accumulation.
        # Each optimizer step still represents up to args.batch_size samples.
        # Cross-entropy is accumulated as a SUM and gradients are divided by
        # the exact number of samples before optimizer.step(), so a group of
        # four 16-sample micro-batches has the same mean-loss gradient scaling
        # as one 64-sample batch (up to floating-point operation ordering).
        optimizer.zero_grad(set_to_none=True)
        accum_samples = 0
        total_loss_sum = 0.0
        total_samples = 0

        paired_batches = zip(train1, train2)
        for micro_step, (left, right) in enumerate(paired_batches, start=1):
            left = left.to(device)
            right = right.to(device)
            y = left.y.view(-1).long()
            n_micro = int(y.numel())

            logits = model(left, right)
            loss_sum = F.cross_entropy(logits, y, reduction="sum")
            loss_sum.backward()

            batch_loss_sum = float(loss_sum.detach().cpu().item())
            total_loss_sum += batch_loss_sum
            total_samples += n_micro
            accum_samples += n_micro

            should_step = accum_samples >= args.batch_size
            is_last = micro_step == len(train1)

            if should_step or is_last:
                # Convert accumulated SUM gradients to exact per-sample MEAN.
                scale = float(accum_samples)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(scale)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_samples = 0

            # Release references to the large 10-head intermediate graph batch
            # as soon as this micro-batch is no longer needed.
            del left, right, y, logits, loss_sum

        mean_loss = float(total_loss_sum / max(total_samples, 1))
        scheduler.step(mean_loss)

        yv, sv = predict(model, device, val1, val2)
        val_auc = float(roc_auc_score(yv, sv))

        sync(device)
        sec = float(time.perf_counter() - epoch_start)
        epoch_seconds.append(sec)

        epoch_rows.append(
            {
                "fold": fold_number,
                "epoch": epoch,
                "train_loss": mean_loss,
                "val_AUROC": val_auc,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": sec,
            }
        )

        if val_auc > best_val_auc + args.min_delta:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or patience_counter >= args.early_stopping_patience:
            print(
                f"[DeepDDS-GAT fold {fold_number}] epoch={epoch} "
                f"loss={mean_loss:.5f} val_AUROC={val_auc:.5f} "
                f"best={best_val_auc:.5f}@{best_epoch} patience={patience_counter}"
            )

        if patience_counter >= args.early_stopping_patience:
            break

    sync(device)
    training_seconds = float(time.perf_counter() - wall_start)
    train_peak_mb = peak_memory_mb(device)

    if best_state is None:
        raise RuntimeError("Training ended without a valid checkpoint.")
    model.load_state_dict(best_state)

    # Outer test is evaluated only after checkpoint selection.
    yt, st = predict(model, device, test1, test2)
    if not np.array_equal(yt, labels[test_idx]):
        raise RuntimeError("Label/order mismatch between processed graphs and all_ddi.csv.")

    pred_df = pd.DataFrame(
        {
            "sample_index": test_idx.astype(int),
            "label": yt.astype(int),
            "score": st.astype(float),
            "predicted_label": (st > 0.5).astype(int),
            "fold": fold_number,
        }
    )
    pred_df.to_csv(outdir / f"deepdds_gat_predictions_fold{fold_number}.csv", index=False)

    pd.DataFrame(epoch_rows).to_csv(
        outdir / f"deepdds_gat_learning_fold{fold_number}.csv", index=False
    )

    row = {
        "fold": fold_number,
        **metric_dict(yt, st),
        "best_epoch": int(best_epoch),
        "best_val_AUROC": float(best_val_auc),
        "training_seconds": training_seconds,
        "training_minutes": training_seconds / 60.0,
        "median_epoch_seconds": float(np.median(epoch_seconds)),
        "training_peak_gpu_memory_mb": train_peak_mb,
        "trainable_parameters": count_params(model),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "seed": int(fold_seed),
    }

    print(json.dumps(row, indent=2))
    return row


def upsert(path: Path, row: dict):
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["fold"].astype(int) != int(row["fold"])]
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df.sort_values("fold").to_csv(path, index=False)


def summarize(metrics_path: Path, summary_path: Path):
    df = pd.read_csv(metrics_path)
    metrics = ["AUROC", "PR_AUC", "ACC", "BACC", "Precision", "TPR", "KAPPA", "F1"]
    rows = []

    for metric in metrics:
        x = df[metric].dropna().to_numpy(dtype=float)
        if not len(x):
            continue
        row = {
            "metric": metric,
            "n_folds": int(len(x)),
            "mean": float(np.mean(x)),
            "SD": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        }
        if len(x) > 1:
            ci = stats.t.interval(
                0.95,
                df=len(x) - 1,
                loc=float(np.mean(x)),
                scale=float(stats.sem(x)),
            )
            row["CI95_lower"] = float(ci[0])
            row["CI95_upper"] = float(ci[1])
        else:
            row["CI95_lower"] = np.nan
            row["CI95_upper"] = np.nan
        rows.append(row)

    pd.DataFrame(rows).to_csv(summary_path, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--full-check-dir", default="./result/full_check")
    p.add_argument("--output", default="./result/deepdds_gat_strict_samefold_v3_memorysafe")
    p.add_argument("--folds", nargs="*", type=int, default=[1])
    p.add_argument("--seed", type=int, default=0)

    # Paper-aligned / strict-rerun settings.
    # Effective optimizer batch size remains 64 for comparability with v2.
    p.add_argument("--batch-size", type=int, default=64)
    # Only this many samples are resident in the GAT training graph at once.
    p.add_argument("--micro-batch-size", type=int, default=16)
    # Evaluation can use a different batch size because it does not update weights.
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--heads", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--early-stopping-patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--shuffle-train", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    for fold in args.folds:
        if fold not in range(1, 6):
            raise ValueError("--folds must be in 1..5")

    device = choose_device(args.device)
    set_seed(args.seed)

    data_root = Path(args.data_root)
    full_dir = Path(args.full_check_dir)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    ddi = pd.read_csv(data_root / "raw" / "all_ddi.csv")
    labels = ddi["label"].to_numpy(dtype=np.int64)
    n = len(labels)

    dataset1 = PreloadedDrugDataset(root=str(data_root), dataset="ONeil_Drug1")
    dataset2 = PreloadedDrugDataset(root=str(data_root), dataset="ONeil_Drug2")
    if len(dataset1) != n or len(dataset2) != n:
        raise ValueError(
            f"Dataset length mismatch: raw={n}, drug1={len(dataset1)}, drug2={len(dataset2)}"
        )

    outer_path = full_dir / "outer_fold_assignments.csv"
    outer_folds = load_outer_folds(outer_path, n)

    # Record all assumptions so the paper does not overstate exact replication.
    config = vars(args).copy()
    config.update(
        {
            "model_label": "DeepDDS-GAT paper-aligned reproduction",
            "graph_loader": "direct preprocessed .pt loading; RDKit not imported",
            "exact_historical_source_available": False,
            "outer_fold_source": str(outer_path.resolve()),
            "outer_test_used_for_model_selection": False,
            "atom_feature_dim": 78,
            "cell_feature_dim": 954,
            "gat_hidden_dims": [1024, 512],
            "gat_first_layer_output_dim": int(1024 * args.heads),
            "gat_second_layer_heads": 1,
            "gat_attention_heads": int(args.heads),
            "gat_head_aggregation": "first layer concat=True with 10 heads; second layer single head concat=False",
            "gat_activations": ["ELU", "ReLU"],
            "gat_feature_dropout": float(args.dropout),
            "effective_batch_size": int(args.batch_size),
            "micro_batch_size": int(args.micro_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "gradient_accumulation": True,
            "gradient_accumulation_rule": (
                "sum cross-entropy over micro-batches; divide gradients by exact "
                "accumulated sample count before optimizer step"
            ),
            "mixed_precision": False,
            "best_checkpoint_storage": "CPU",
            "pooling": "global max pooling over non-padded atoms",
            "cell_mlp_hidden_dims": [2048, 512],
            "classifier_input_dim": 1536,
            "classifier_hidden_dims": [1024, 512, 128],
            "note": (
                "Published DeepDDS-GAT structure is followed where explicitly reported. "
                "The first GAT layer concatenates the 10 attention heads, consistent with the "
                "multi-head concatenation equation in the paper; the second layer uses one "
                "output head to produce 512 dimensions. Weight decay, strict inner-validation "
                "early stopping, and batch size are rerun-protocol choices and are recorded "
                "rather than claimed as historical DeepDDS defaults. The memory-safe runner "
                "uses micro-batches only to reduce peak memory while retaining an effective "
                "optimizer batch size of 64."
            ),
        }
    )
    (outdir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("Device:", device)
    print("Records:", n)
    print("Model: DeepDDS-GAT paper-aligned reproduction")
    print("Trainable parameters:", count_params(DeepDDSGATPaperAligned(args.dropout, args.heads)))

    metrics_path = outdir / "deepdds_gat_fold_metrics.csv"

    for fold_number in args.folds:
        test_idx = outer_folds[fold_number - 1]
        train_idx, val_idx = make_inner_split(
            labels,
            test_idx,
            fold_number,
            args.val_fraction,
            args.seed,
        )

        print(
            f"\n[DeepDDS-GAT fold {fold_number}] "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

        row = train_fold(
            fold_number,
            train_idx,
            val_idx,
            test_idx,
            dataset1,
            dataset2,
            labels,
            device,
            args,
            outdir,
        )
        upsert(metrics_path, row)

    summarize(metrics_path, outdir / "deepdds_gat_summary.csv")
    print("\nSaved to:", outdir.resolve())


if __name__ == "__main__":
    main()
