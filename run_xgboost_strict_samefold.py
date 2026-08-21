#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict same-fold XGBoost baseline for the MFMNet reviewer rerun.

Uses EXACTLY the outer-fold assignments already produced by
MFMNet_reviewer_clean.py in:
    ./result/full_check/outer_fold_assignments.csv

Features:
    - drug 1 embedding: 768 dims
    - drug 2 embedding: 768 dims
    - cell-line profile: 954 dims
Total: 2490 dims

No outer-test sample is used for model selection.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
from xgboost import XGBClassifier


def pr_auc_value(y_true, score):
    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(auc(recall, precision))


def metrics(y_true, score):
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


def load_data(data_root: Path):
    ddi_path = data_root / "raw" / "all_ddi.csv"
    emb_path = data_root / "raw" / "all_drug_embedding.csv"

    ddi = pd.read_csv(ddi_path)
    emb = pd.read_csv(emb_path, header=None)

    if len(ddi) != len(emb):
        raise ValueError(f"Row mismatch: all_ddi={len(ddi)}, embeddings={len(emb)}")

    # Supplied dataset layout: columns from index 9 onward are the 954 cell-line features.
    cell = ddi.iloc[:, 9:].to_numpy(dtype=np.float32)
    drug = emb.to_numpy(dtype=np.float32)

    if drug.shape[1] != 1536:
        raise ValueError(f"Expected 1536 drug-embedding columns, got {drug.shape[1]}")
    if cell.shape[1] != 954:
        raise ValueError(f"Expected 954 cell-line features from all_ddi[:, 9:], got {cell.shape[1]}")

    X = np.concatenate([drug[:, :768], drug[:, 768:1536], cell], axis=1)
    y = ddi["label"].to_numpy(dtype=np.int64)

    return X, y


def load_folds(path: Path, n: int):
    df = pd.read_csv(path)
    required = {"sample_index", "outer_fold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    if len(df) != n or df["sample_index"].nunique() != n:
        raise ValueError("Outer-fold assignment does not cover each sample exactly once")

    folds = {}
    for fold in range(1, 6):
        test_idx = df.loc[df["outer_fold"].astype(int) == fold, "sample_index"].to_numpy(dtype=int)
        train_idx = np.setdiff1d(np.arange(n, dtype=int), test_idx, assume_unique=False)
        folds[fold] = (train_idx, test_idx)

    covered = np.concatenate([folds[f][1] for f in range(1, 6)])
    if len(covered) != n or len(np.unique(covered)) != n:
        raise ValueError("Five outer test folds do not cover the full dataset exactly once")

    return folds


def summarize(df: pd.DataFrame, out_path: Path):
    rows = []
    metric_names = ["AUROC", "PR_AUC", "ACC", "BACC", "Precision", "TPR", "KAPPA", "F1"]
    for metric in metric_names:
        x = df[metric].to_numpy(dtype=float)
        row = {
            "metric": metric,
            "n_folds": len(x),
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
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--full-check-dir", default="./result/full_check")
    p.add_argument("--output", default="./result/xgboost_strict_samefold")
    p.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5])

    # Fixed baseline hyperparameters; not tuned on outer test.
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    for f in args.folds:
        if f not in range(1, 6):
            raise ValueError("--folds must contain only 1..5")

    data_root = Path(args.data_root)
    full_dir = Path(args.full_check_dir)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(data_root)
    folds = load_folds(full_dir / "outer_fold_assignments.csv", len(y))

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("positive fraction:", float(np.mean(y)))

    rows = []

    metrics_path = outdir / "xgboost_fold_metrics.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
    else:
        existing = pd.DataFrame()

    for fold in args.folds:
        train_idx, test_idx = folds[fold]

        print(f"\n[XGBoost fold {fold}] train={len(train_idx)} test={len(test_idx)}")

        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.seed + 1000 * fold,
            n_jobs=args.n_jobs,
            tree_method="hist",
        )

        t0 = time.perf_counter()
        model.fit(X[train_idx], y[train_idx])
        fit_seconds = float(time.perf_counter() - t0)

        t1 = time.perf_counter()
        score = model.predict_proba(X[test_idx])[:, 1]
        inference_seconds = float(time.perf_counter() - t1)

        m = metrics(y[test_idx], score)
        row = {
            "fold": fold,
            **m,
            "fit_seconds": fit_seconds,
            "fit_minutes": fit_seconds / 60.0,
            "inference_seconds": inference_seconds,
            "inference_ms_per_sample": 1000.0 * inference_seconds / len(test_idx),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }

        print(json.dumps(row, indent=2))

        pd.DataFrame(
            {
                "sample_index": test_idx.astype(int),
                "label": y[test_idx].astype(int),
                "score": score.astype(float),
                "predicted_label": (score > 0.5).astype(int),
                "fold": fold,
            }
        ).sort_values("sample_index").to_csv(
            outdir / f"xgboost_predictions_fold{fold}.csv", index=False
        )

        if len(existing):
            existing = existing[existing["fold"].astype(int) != fold]
        existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        existing.sort_values("fold").to_csv(metrics_path, index=False)

    final_df = pd.read_csv(metrics_path)
    summarize(final_df, outdir / "xgboost_summary.csv")

    run_config = {
        "feature_dim": int(X.shape[1]),
        "drug_embedding_dim_each": 768,
        "cell_feature_dim": 954,
        "fold_source": str((full_dir / "outer_fold_assignments.csv").resolve()),
        "outer_test_used_for_tuning": False,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "tree_method": "hist",
            "seed": args.seed,
        },
    }
    (outdir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print("\n=== Current fold metrics ===")
    print(final_df.sort_values("fold").to_string(index=False))
    print("\nSaved to:", outdir.resolve())


if __name__ == "__main__":
    main()
