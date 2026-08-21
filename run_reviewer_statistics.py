#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reviewer-ready same-fold statistical analysis for MFMNet.

Reads whatever results are currently available from:
  ./result/full_check
  ./result/xgboost_strict_samefold
  ./result/deepdds_gcn_strict_samefold
  ./result/strict_final_ensemble

It is safe to run before all experiments finish:
- model summaries are produced for available folds;
- inferential paired tests are only treated as FINAL when all 5 common folds exist;
- incomplete comparisons are explicitly marked INCOMPLETE.

Outputs:
  reviewer_statistics/model_fold_metrics_combined.csv
  reviewer_statistics/model_summary_mean_sd_ci95.csv
  reviewer_statistics/paired_comparisons.csv
  reviewer_statistics/paired_comparisons_holm.csv
  reviewer_statistics/current_status.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EXPECTED_FOLDS = {1, 2, 3, 4, 5}
METRICS = ["AUROC", "PR_AUC", "ACC", "BACC", "Precision", "TPR", "KAPPA", "F1"]


def read_full(path: Path) -> pd.DataFrame:
    f = path / "ablation_fold_metrics.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "variant" in df.columns:
        df = df[df["variant"].astype(str).str.lower() == "full"].copy()
    if df.empty:
        return df
    df["Model"] = "Full backbone"
    return df


def read_named_metrics(path: Path, filename: str, model_name: str) -> pd.DataFrame:
    f = path / filename
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if df.empty:
        return df
    df["Model"] = model_name
    return df


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rename = {}
    for c in df.columns:
        low = c.lower()
        if low == "fold":
            rename[c] = "fold"
        elif low in {"auroc", "auc"}:
            rename[c] = "AUROC"
        elif low in {"pr_auc", "prauc", "aupr", "auprc"}:
            rename[c] = "PR_AUC"
        elif low == "acc":
            rename[c] = "ACC"
        elif low == "bacc":
            rename[c] = "BACC"
        elif low == "precision":
            rename[c] = "Precision"
        elif low in {"tpr", "recall"}:
            rename[c] = "TPR"
        elif low in {"kappa", "cohen_kappa"}:
            rename[c] = "KAPPA"
        elif low == "f1":
            rename[c] = "F1"
    df = df.rename(columns=rename)
    df["fold"] = df["fold"].astype(int)
    keep = ["Model", "fold"] + [m for m in METRICS if m in df.columns]
    extras = [c for c in ["best_epoch", "best_val_AUROC", "trainable_parameters",
                          "training_minutes", "training_wall_minutes",
                          "training_peak_gpu_memory_mb",
                          "inference_ms_per_sample",
                          "throughput_samples_per_second"] if c in df.columns]
    return df[keep + extras].copy()


def mean_ci95(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    if n > 1:
        sem = stats.sem(x)
        lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
        return mean, sd, float(lo), float(hi)
    return mean, sd, np.nan, np.nan


def summarize_models(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in all_df.groupby("Model", sort=False):
        folds = sorted(g["fold"].unique().tolist())
        for metric in METRICS:
            if metric not in g.columns:
                continue
            x = g[metric].dropna().to_numpy(dtype=float)
            if len(x) == 0:
                continue
            mean, sd, lo, hi = mean_ci95(x)
            rows.append({
                "Model": model,
                "metric": metric,
                "n_folds": len(x),
                "folds_present": ",".join(map(str, folds)),
                "complete_5fold": set(folds) == EXPECTED_FOLDS,
                "mean": mean,
                "SD": sd,
                "CI95_lower": lo,
                "CI95_upper": hi,
            })
    return pd.DataFrame(rows)


def paired_one(a: pd.DataFrame, b: pd.DataFrame, metric: str, name_a: str, name_b: str) -> dict:
    aa = a[["fold", metric]].dropna().rename(columns={metric: "a"})
    bb = b[["fold", metric]].dropna().rename(columns={metric: "b"})
    p = aa.merge(bb, on="fold", how="inner").sort_values("fold")
    d = p["a"].to_numpy(float) - p["b"].to_numpy(float)
    folds = p["fold"].astype(int).tolist()
    complete = set(folds) == EXPECTED_FOLDS

    out = {
        "Model_A": name_a,
        "Model_B": name_b,
        "metric": metric,
        "n_common_folds": len(d),
        "common_folds": ",".join(map(str, folds)),
        "status": "FINAL_5FOLD" if complete else f"INCOMPLETE_{len(d)}_OF_5",
        "A_mean_on_common_folds": float(p["a"].mean()) if len(p) else np.nan,
        "B_mean_on_common_folds": float(p["b"].mean()) if len(p) else np.nan,
        "mean_difference_A_minus_B": float(np.mean(d)) if len(d) else np.nan,
        "all_folds_same_direction": bool(np.all(d > 0) or np.all(d < 0)) if len(d) else False,
        "A_higher_in_n_folds": int(np.sum(d > 0)) if len(d) else 0,
        "B_higher_in_n_folds": int(np.sum(d < 0)) if len(d) else 0,
        "ties": int(np.sum(d == 0)) if len(d) else 0,
        "diff_CI95_lower": np.nan,
        "diff_CI95_upper": np.nan,
        "paired_t_stat": np.nan,
        "paired_t_p": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p": np.nan,
    }

    # Reviewer-facing inference is finalized only on all five common folds.
    if complete:
        mean, sd, lo, hi = mean_ci95(d)
        out["diff_CI95_lower"] = lo
        out["diff_CI95_upper"] = hi

        t = stats.ttest_rel(p["a"], p["b"])
        out["paired_t_stat"] = float(t.statistic)
        out["paired_t_p"] = float(t.pvalue)

        if np.allclose(d, 0):
            out["wilcoxon_stat"] = 0.0
            out["wilcoxon_p"] = 1.0
        else:
            w = stats.wilcoxon(p["a"], p["b"], alternative="two-sided", zero_method="wilcox")
            out["wilcoxon_stat"] = float(w.statistic)
            out["wilcoxon_p"] = float(w.pvalue)

    return out


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan)
    valid_idx = np.where(np.isfinite(pvals))[0]
    if len(valid_idx) == 0:
        return out

    vals = pvals[valid_idx]
    order = np.argsort(vals)
    ordered = vals[order]
    m = len(ordered)

    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for i, p in enumerate(ordered):
        adj = (m - i) * p
        running = max(running, adj)
        adjusted[i] = min(1.0, running)

    unorder = np.empty(m, dtype=float)
    unorder[order] = adjusted
    out[valid_idx] = unorder
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-dir", default="./result/full_check")
    ap.add_argument("--xgb-dir", default="./result/xgboost_strict_samefold")
    ap.add_argument("--deepdds-dir", default="./result/deepdds_gcn_strict_samefold")
    ap.add_argument("--ensemble-dir", default="./result/strict_final_ensemble")
    ap.add_argument("--output", default="./result/reviewer_statistics")
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = []

    full = harmonize(read_full(Path(args.full_dir)))
    if not full.empty:
        frames.append(full)

    xgb = harmonize(read_named_metrics(
        Path(args.xgb_dir), "xgboost_fold_metrics.csv", "XGBoost"
    ))
    if not xgb.empty:
        frames.append(xgb)

    deep = harmonize(read_named_metrics(
        Path(args.deepdds_dir), "deepdds_gcn_fold_metrics.csv",
        "DeepDDS-GCN paper-aligned"
    ))
    if not deep.empty:
        frames.append(deep)

    ens = harmonize(read_named_metrics(
        Path(args.ensemble_dir), "strict_mfmnet_ensemble_fold_metrics.csv",
        "Strict MFMNet ensemble"
    ))
    if not ens.empty:
        frames.append(ens)

    mdsyn = harmonize(read_named_metrics(
        Path(args.ensemble_dir), "mdsyn_fold_metrics.csv", "MDSyn"
    ))
    if not mdsyn.empty:
        frames.append(mdsyn)

    if not frames:
        raise FileNotFoundError("No metric files were found in the supplied result directories.")

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = all_df.sort_values(["Model", "fold"])
    all_df.to_csv(outdir / "model_fold_metrics_combined.csv", index=False)

    summary = summarize_models(all_df)
    summary.to_csv(outdir / "model_summary_mean_sd_ci95.csv", index=False)

    by_model = {m: g.copy() for m, g in all_df.groupby("Model")}
    comparisons = [
        ("Full backbone", "XGBoost"),
        ("Full backbone", "DeepDDS-GCN paper-aligned"),
        ("Strict MFMNet ensemble", "XGBoost"),
        ("Strict MFMNet ensemble", "DeepDDS-GCN paper-aligned"),
        ("Strict MFMNet ensemble", "Full backbone"),
        ("Strict MFMNet ensemble", "MDSyn"),
    ]

    rows = []
    for a_name, b_name in comparisons:
        if a_name not in by_model or b_name not in by_model:
            continue
        a = by_model[a_name]
        b = by_model[b_name]
        for metric in ["AUROC", "PR_AUC"]:
            if metric in a.columns and metric in b.columns:
                rows.append(paired_one(a, b, metric, a_name, b_name))

    paired = pd.DataFrame(rows)
    paired.to_csv(outdir / "paired_comparisons.csv", index=False)

    if not paired.empty:
        adjusted_frames = []
        for metric, g in paired.groupby("metric", sort=False):
            gg = g.copy()
            gg["paired_t_p_holm"] = holm_adjust(gg["paired_t_p"].to_numpy(float))
            gg["wilcoxon_p_holm"] = holm_adjust(gg["wilcoxon_p"].to_numpy(float))
            adjusted_frames.append(gg)
        paired_holm = pd.concat(adjusted_frames, ignore_index=True)
    else:
        paired_holm = paired.copy()

    paired_holm.to_csv(outdir / "paired_comparisons_holm.csv", index=False)

    status_lines = []
    status_lines.append("CURRENT REVIEWER STATISTICS STATUS")
    status_lines.append("=" * 44)
    for model, g in all_df.groupby("Model", sort=False):
        folds = sorted(g["fold"].astype(int).unique().tolist())
        status_lines.append(
            f"{model}: folds {folds} ({len(folds)}/5) "
            + ("COMPLETE" if set(folds) == EXPECTED_FOLDS else "INCOMPLETE")
        )

    status_lines.append("")
    status_lines.append("FINAL inferential p-values are populated only for comparisons with all 5 common folds.")
    status_lines.append("Holm-adjusted p-values are reported separately for AUROC and PR-AUC comparison families.")

    (outdir / "current_status.txt").write_text("\n".join(status_lines), encoding="utf-8")
    print("\n".join(status_lines))
    print("\nSaved to:", outdir.resolve())


if __name__ == "__main__":
    main()
