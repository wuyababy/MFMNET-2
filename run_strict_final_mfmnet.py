#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict final MFMNet ensemble rerun.

Purpose
-------
Train ONLY the original MDSyn component under a leakage-free nested protocol,
then combine its outer-test probabilities 50:50 with the already completed
strict Enhanced/full predictions from MFMNet_reviewer_clean.py.

This avoids retraining the Enhanced/full component.

Run this file from the ORIGINAL codes/ directory, next to:
    loader.py
    model.py
    data/
    result/

Expected prior results:
    ./result/full_check/pred_full_fold1.csv
    ...
    ./result/full_check/pred_full_fold5.csv

Protocol
--------
- Outer 5 folds are read from full_check/outer_fold_assignments.csv.
- The inner validation split is reconstructed exactly as in MFMNet_reviewer_clean.py:
    train_test_split(..., test_size=0.10,
                     random_state=seed + fold_number,
                     stratify=labels[outer_train])
- Model selection uses ONLY inner-validation AUROC.
- Outer test is evaluated ONCE after checkpoint selection.
- Final MFMNet score = 0.5 * strict Enhanced/full score + 0.5 * strict MDSyn score.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from loader import DrugcombDataset
from model import MDSyn


DEFAULT_SEED = 0
N_FOLDS = 5


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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)


def original_l2_penalty(model: torch.nn.Module) -> torch.Tensor:
    # Matches enhanced_main.py:
    # l2_reg += torch.norm(param, 2)
    terms = [torch.norm(p, 2) for p in model.parameters() if p.requires_grad]
    return torch.stack(terms).sum()


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


def make_loader(dataset, indices: Sequence[int], batch_size: int) -> DataLoader:
    # Original enhanced_main.py used shuffle=None for both train and test.
    subset = dataset[list(map(int, indices))]
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def predict_mdsyn(
    model: torch.nn.Module,
    device: torch.device,
    loader1: DataLoader,
    loader2: DataLoader,
    linc: torch.Tensor,
):
    model.eval()
    ys = []
    ss = []
    for data1, data2 in zip(loader1, loader2):
        data1 = data1.to(device)
        data2 = data2.to(device)
        out, _ = model(data1, data2, linc)
        prob = F.softmax(out, dim=1)[:, 1]
        ys.append(data1.y.view(-1).detach().cpu().numpy())
        ss.append(prob.detach().cpu().numpy())
    return np.concatenate(ys).astype(int), np.concatenate(ss).astype(float)


def load_outer_assignment(path: Path, n: int) -> list[np.ndarray]:
    df = pd.read_csv(path)
    required = {"sample_index", "outer_fold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(df) != n:
        raise ValueError(f"outer assignment rows={len(df)}, expected n={n}")
    if df["sample_index"].nunique() != n:
        raise ValueError("outer assignment does not contain exactly one row per sample")
    folds = []
    for fold in range(1, N_FOLDS + 1):
        idx = df.loc[df["outer_fold"].astype(int) == fold, "sample_index"].to_numpy(dtype=int)
        folds.append(idx)
    covered = np.concatenate(folds)
    if len(covered) != n or len(np.unique(covered)) != n:
        raise ValueError("outer folds do not cover every sample exactly once")
    return folds


def make_inner_split(
    labels: np.ndarray,
    test_idx: np.ndarray,
    fold_number: int,
    val_fraction: float,
    seed: int,
):
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


def train_one_fold(
    fold_number: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    drug1_data,
    drug2_data,
    linc: torch.Tensor,
    labels: np.ndarray,
    device: torch.device,
    args,
    outdir: Path,
) -> dict:
    fold_seed = args.seed + 1000 * fold_number
    set_seed(fold_seed)

    train1 = make_loader(drug1_data, train_idx, args.batch_size)
    train2 = make_loader(drug2_data, train_idx, args.batch_size)
    val1 = make_loader(drug1_data, val_idx, args.batch_size)
    val2 = make_loader(drug2_data, val_idx, args.batch_size)
    test1 = make_loader(drug1_data, test_idx, args.batch_size)
    test2 = make_loader(drug2_data, test_idx, args.batch_size)

    model = MDSyn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    # gene_vector.csv is 978 x 128 after index_col=0, header=None.
    linc_dev = linc.to(device)

    best_val_auc = -np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0
    epoch_rows = []

    sync(device)
    wall_start = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []

        for data1, data2 in zip(train1, train2):
            data1 = data1.to(device)
            data2 = data2.to(device)
            y = data1.y.view(-1).long()

            optimizer.zero_grad(set_to_none=True)
            out, _ = model(data1, data2, linc_dev)
            loss = F.cross_entropy(out, y)
            if args.l2 > 0:
                loss = loss + args.l2 * original_l2_penalty(model)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        mean_train_loss = float(np.mean(losses))
        scheduler.step(mean_train_loss)

        yv, sv = predict_mdsyn(model, device, val1, val2, linc_dev)
        val_auc = float(roc_auc_score(yv, sv))
        current_lr = float(optimizer.param_groups[0]["lr"])

        epoch_rows.append(
            {
                "fold": fold_number,
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_AUROC": val_auc,
                "learning_rate": current_lr,
            }
        )

        if val_auc > best_val_auc + args.min_delta:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0 or patience_counter >= args.early_stopping_patience:
            print(
                f"[MDSyn fold {fold_number}] epoch={epoch} "
                f"loss={mean_train_loss:.5f} val_AUROC={val_auc:.5f} "
                f"best={best_val_auc:.5f}@{best_epoch} patience={patience_counter}"
            )

        if patience_counter >= args.early_stopping_patience:
            break

    sync(device)
    training_seconds = float(time.perf_counter() - wall_start)

    if best_state is None:
        raise RuntimeError("No best MDSyn state was selected")
    model.load_state_dict(best_state)

    # Outer test is touched only here, once.
    yt, st = predict_mdsyn(model, device, test1, test2, linc_dev)
    if not np.array_equal(yt, labels[test_idx]):
        raise RuntimeError("Processed dataset labels do not align with raw all_ddi.csv labels")

    pd.DataFrame(epoch_rows).to_csv(
        outdir / f"learning_mdsyn_fold{fold_number}.csv", index=False
    )

    pred_path = outdir / f"pred_mdsyn_fold{fold_number}.csv"
    pd.DataFrame(
        {
            "sample_index": test_idx.astype(int),
            "label": yt.astype(int),
            "score": st.astype(float),
        }
    ).sort_values("sample_index").to_csv(pred_path, index=False)

    m = metric_dict(yt, st)
    row = {
        "fold": fold_number,
        **m,
        "best_epoch": int(best_epoch),
        "best_val_AUROC": float(best_val_auc),
        "training_seconds": training_seconds,
        "training_minutes": training_seconds / 60.0,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "seed": int(fold_seed),
    }
    print(json.dumps({"MDSyn": row}, indent=2))
    return row


def combine_with_full(
    fold_number: int,
    mdsyn_pred_path: Path,
    full_pred_path: Path,
    outdir: Path,
) -> dict:
    md = pd.read_csv(mdsyn_pred_path)
    full = pd.read_csv(full_pred_path)

    need = {"sample_index", "label", "score"}
    if not need.issubset(md.columns) or not need.issubset(full.columns):
        raise ValueError("Prediction files must contain sample_index, label, score")

    merged = md.merge(
        full[["sample_index", "label", "score"]],
        on="sample_index",
        how="inner",
        suffixes=("_mdsyn", "_full"),
        validate="one_to_one",
    )
    if len(merged) != len(md) or len(merged) != len(full):
        raise ValueError(f"Fold {fold_number}: MDSyn/full prediction coverage mismatch")
    if not np.array_equal(
        merged["label_mdsyn"].to_numpy(dtype=int),
        merged["label_full"].to_numpy(dtype=int),
    ):
        raise ValueError(f"Fold {fold_number}: label mismatch between MDSyn and full predictions")

    merged["score"] = 0.5 * merged["score_mdsyn"] + 0.5 * merged["score_full"]
    y = merged["label_mdsyn"].to_numpy(dtype=int)
    s = merged["score"].to_numpy(dtype=float)

    out_pred = pd.DataFrame(
        {
            "sample_index": merged["sample_index"].astype(int),
            "label": y,
            "score_mdsyn": merged["score_mdsyn"],
            "score_full": merged["score_full"],
            "score": s,
            "predicted_label": (s > 0.5).astype(int),
        }
    ).sort_values("sample_index")
    out_pred.to_csv(outdir / f"pred_strict_mfmnet_ensemble_fold{fold_number}.csv", index=False)

    row = {"fold": fold_number, **metric_dict(y, s), "n_test": int(len(y))}
    print(json.dumps({"Strict MFMNet ensemble": row}, indent=2))
    return row


def upsert(path: Path, row: dict) -> None:
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        if "fold" in old.columns:
            old = old[old["fold"].astype(int) != int(row["fold"])]
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df.sort_values("fold").to_csv(path, index=False)


def summarize(path: Path, out_path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics = ["AUROC", "PR_AUC", "ACC", "BACC", "Precision", "TPR", "KAPPA", "F1"]
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        x = df[metric].dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
        row = {
            "metric": metric,
            "n_folds": len(x),
            "mean": float(x.mean()),
            "SD": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        }
        if len(x) > 1:
            from scipy import stats
            ci = stats.t.interval(
                0.95, df=len(x)-1, loc=float(x.mean()), scale=float(stats.sem(x))
            )
            row["CI95_lower"] = float(ci[0])
            row["CI95_upper"] = float(ci[1])
        else:
            row["CI95_lower"] = np.nan
            row["CI95_upper"] = np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--full-check-dir", default="./result/full_check")
    ap.add_argument("--output", default="./result/strict_final_ensemble")
    ap.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--early-stopping-patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--lr-patience", type=int, default=10)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    for f in args.folds:
        if f not in range(1, 6):
            raise ValueError("--folds must be in 1..5")

    device = choose_device(args.device)
    set_seed(args.seed)

    full_dir = Path(args.full_check_dir)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Full-check dir:", full_dir.resolve())
    print("Output:", outdir.resolve())

    raw_ddi = pd.read_csv(Path(args.data_root) / "raw" / "all_ddi.csv")
    labels = raw_ddi["label"].to_numpy(dtype=np.int64)
    n = len(labels)

    outer_assignment = full_dir / "outer_fold_assignments.csv"
    if not outer_assignment.exists():
        raise FileNotFoundError(
            f"Missing {outer_assignment}. Use the same full_check directory produced by "
            "MFMNet_reviewer_clean.py."
        )
    outer_folds = load_outer_assignment(outer_assignment, n)

    # Original processed graph datasets.
    drug1_data = DrugcombDataset(root=args.data_root, dataset="ONeil_Drug1")
    drug2_data = DrugcombDataset(root=args.data_root, dataset="ONeil_Drug2")
    if len(drug1_data) != n or len(drug2_data) != n:
        raise ValueError(
            f"Dataset length mismatch: raw={n}, drug1={len(drug1_data)}, drug2={len(drug2_data)}"
        )

    # Exactly matches enhanced_main.py loading.
    row_lincs = pd.read_csv(
        Path(args.data_root) / "raw" / "gene_vector.csv",
        index_col=0,
        header=None,
    )
    linc = torch.tensor(row_lincs.to_numpy(), dtype=torch.float32)
    if tuple(linc.shape) != (978, 128):
        print("WARNING: gene_vector shape is", tuple(linc.shape), "expected (978, 128) from supplied code")

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    md_metrics_path = outdir / "mdsyn_fold_metrics.csv"
    ens_metrics_path = outdir / "strict_mfmnet_ensemble_fold_metrics.csv"

    for fold_number in args.folds:
        test_idx = outer_folds[fold_number - 1]
        train_idx, val_idx = make_inner_split(
            labels,
            test_idx,
            fold_number,
            args.val_fraction,
            args.seed,
        )

        full_pred = full_dir / f"pred_full_fold{fold_number}.csv"
        if not full_pred.exists():
            raise FileNotFoundError(f"Missing strict full prediction: {full_pred}")

        md_pred = outdir / f"pred_mdsyn_fold{fold_number}.csv"

        if args.force or not md_pred.exists():
            md_row = train_one_fold(
                fold_number=fold_number,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                drug1_data=drug1_data,
                drug2_data=drug2_data,
                linc=linc,
                labels=labels,
                device=device,
                args=args,
                outdir=outdir,
            )
            upsert(md_metrics_path, md_row)
        else:
            print(f"SKIP existing MDSyn fold {fold_number}: {md_pred}")

        ens_row = combine_with_full(
            fold_number,
            md_pred,
            full_pred,
            outdir,
        )
        upsert(ens_metrics_path, ens_row)

    summarize(md_metrics_path, outdir / "mdsyn_summary.csv")
    summarize(ens_metrics_path, outdir / "strict_mfmnet_ensemble_summary.csv")

    print("\nDone.")
    print("MDSyn metrics:", md_metrics_path)
    print("Strict ensemble metrics:", ens_metrics_path)


if __name__ == "__main__":
    main()
