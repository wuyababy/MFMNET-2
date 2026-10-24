"""
make_fig4_cell_lines.py - cell-line-level Precision and MSE comparison
(MFMNet vs. DeepDDS-GCN)

Inputs
------
  Per-sample predictions (5 folds):
    MFMNet      : results/predictions/mfmnet/pred_full_fold{1..5}.csv
    DeepDDS-GCN : results/predictions/deepdds_gcn/deepdds_gcn_predictions_fold{1..5}.csv
  Sample-to-cell-line mapping:
    results/predictions/sample_cell_map.csv (sample_index = row i of the dataset)

Metric definitions
------------------
  Precision : precision_score(label, predicted_label) per cell line (threshold 0.5)
  MSE       : mean((label - score)^2) x 10^4 per cell line

Output: fig4_cell_lines.tif (300 dpi, LZW, RGB) + fig4_cell_lines.pdf (vector)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 22,
    "xtick.labelsize": 14,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

COLOR_DEEPDDS = "#FDBF3B"
COLOR_MFMNET = "#339966"


def load_predictions(pred_dir: Path, pattern: str, cells: np.ndarray) -> pd.DataFrame:
    frames = []
    for f in range(1, 6):
        p = pd.read_csv(pred_dir / pattern.format(f=f))
        p["cell"] = cells[p["sample_index"].values]
        p["fold"] = f
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def per_cell_stats(df: pd.DataFrame):
    g = df.groupby("cell")
    prec = g.apply(lambda x: precision_score(x["label"], x["predicted_label"]), include_groups=False)
    mse = g.apply(lambda x: ((x["label"] - x["score"]) ** 2).mean() * 1e4, include_groups=False)
    return prec, mse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell-map", default="results/predictions/sample_cell_map.csv",
                    help="sample_index → 细胞系映射 CSV（列：sample_index, cell, label, outer_fold）")
    ap.add_argument("--mfmnet-pred-dir", default="results/predictions/mfmnet")
    ap.add_argument("--deepdds-pred-dir", default="results/predictions/deepdds_gcn")
    ap.add_argument("--outdir", default="figures/output/fig4")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(args.cell_map)["cell"].values
    mfm = load_predictions(Path(args.mfmnet_pred_dir), "pred_full_fold{f}.csv", cells)
    dds = load_predictions(Path(args.deepdds_pred_dir), "deepdds_gcn_predictions_fold{f}.csv", cells)

    # 逐细胞系聚合 + 折间 SD（误差棒）
    def stats_with_sd(df):
        per_fold = []
        for f in range(1, 6):
            d = df[df["fold"] == f]
            prec, mse = per_cell_stats(d)
            per_fold.append(pd.DataFrame({"precision": prec, "mse": mse}))
        s = pd.concat(per_fold).groupby(level=0)
        return s.mean(), s.std(ddof=1)

    m_mean, m_sd = stats_with_sd(mfm)   # DataFrame: index=cell, columns=[precision, mse]
    g_mean, g_sd = stats_with_sd(dds)
    mp_mean, mp_sd = m_mean["precision"], m_sd["precision"]
    gp_mean, gp_sd = g_mean["precision"], g_sd["precision"]
    mm_mean, mm_sd = m_mean["mse"], m_sd["mse"]
    gm_mean, gm_sd = g_mean["mse"], g_sd["mse"]

    cells_order = m_mean.index.tolist()
    print(f"细胞系数: {len(cells_order)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.6))
    x = np.arange(len(cells_order))
    width = 0.38

    for ax, m_mean, m_sd, g_mean, g_sd in (
        (ax1, mp_mean, mp_sd, gp_mean, gp_sd),
        (ax2, mm_mean, mm_sd, gm_mean, gm_sd),
    ):
        ax.bar(x - width / 2, g_mean.loc[cells_order], width, yerr=g_sd.loc[cells_order],
               capsize=2, label="DeepDDs", color=COLOR_DEEPDDS, error_kw={"elinewidth": 0.9})
        ax.bar(x + width / 2, m_mean.loc[cells_order], width, yerr=m_sd.loc[cells_order],
               capsize=2, label="MFMNet", color=COLOR_MFMNET, error_kw={"elinewidth": 0.9})
        ax.set_xticks(x)
        ax.set_xticklabels(cells_order, rotation=45, ha="right")
        ax.set_ylabel("Precision" if ax is ax1 else "MSE")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2, frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_ylim(0.5, 1.05)
    ax1.set_xlabel("")
    ax2.set_xlabel("Cell Line")

    for ax, tag in ((ax1, "A"), (ax2, "B")):
        ax.text(-0.07, 1.04, tag, transform=ax.transAxes, fontsize=24, fontweight="bold",
                ha="left", va="bottom")

    fig.tight_layout()
    tif = outdir / "fig4_cell_lines.tif"
    fig.savefig(tif, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(outdir / "fig4_cell_lines.pdf")
    plt.close(fig)
    print(f"已输出: {tif} 与 {outdir/'fig4_cell_lines.pdf'}")


if __name__ == "__main__":
    main()
