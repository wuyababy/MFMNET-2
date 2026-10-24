"""
make_fig5_model_comparison.py - fold-matched model comparison bar charts
(AUROC / PR-AUC / ACC / BACC / Precision, mean +/- SD over five folds)

Inputs (per-fold metrics)
-------------------------
  MFMNet      : results/fold_metrics/mfmnet_fold_metrics.csv (variant == 'full')
  XGBoost     : results/fold_metrics/xgboost_fold_metrics.csv
  DeepDDS-GCN : results/fold_metrics/deepdds_gcn_fold_metrics.csv
  DeepDDS-GAT : results/fold_metrics/deepdds_gat_fold_metrics.csv

Output: fig5_model_comparison.tif (300 dpi, LZW, RGB) + fig5_model_comparison.pdf (vector)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
})

METRICS = ["AUROC", "PR_AUC", "ACC", "BACC", "Precision"]
XTICKS = ["AUROC", "PR_AUC", "ACC", "BACC", "Precision"]
MODELS = [
    ("MFMNet", "results/fold_metrics/mfmnet_fold_metrics.csv", "full", "#4878D0"),
    ("XGBoost", "results/fold_metrics/xgboost_fold_metrics.csv", None, "#E5B15E"),
    ("DeepDDS-GCN", "results/fold_metrics/deepdds_gcn_fold_metrics.csv", None, "#D0607E"),
    ("DeepDDS-GAT", "results/fold_metrics/deepdds_gat_fold_metrics.csv", None, "#F39B4D"),
]


def fold_stats(path: str, variant):
    df = pd.read_csv(path)
    if variant is not None:
        df = df[df["variant"] == variant]
    means = np.array([df[m].mean() for m in METRICS])
    sds = np.array([df[m].std(ddof=1) for m in METRICS])
    return means, sds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default="figures/output/fig5")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(METRICS))
    width = 0.2

    print("Figure 5 values (mean +/- SD across 5 folds):")
    for i, (name, path, variant, color) in enumerate(MODELS):
        means, sds = fold_stats(path, variant)
        print(f"  {name:12s} " + "  ".join(f"{m}={a:.4f}±{s:.4f}" for m, a, s in zip(METRICS, means, sds)))
        ax.bar(x + (i - 1.5) * width, means, width, yerr=sds, capsize=3,
               label=name, color=color, edgecolor="black", linewidth=0.6,
               error_kw={"elinewidth": 1.0})

    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS)
    ax.set_ylabel("Performance")
    ax.set_xlabel("Metric")
    ax.set_ylim(0.75, 0.95)
    ax.legend(loc="upper right", frameon=True, ncol=1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    tif = outdir / "fig5_model_comparison.tif"
    fig.savefig(tif, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(outdir / "fig5_model_comparison.pdf")
    plt.close(fig)
    print(f"已输出: {tif} 与 {outdir/'fig5_model_comparison.pdf'}")


if __name__ == "__main__":
    main()
