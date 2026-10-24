"""
make_fig3_performance.py - five-fold cross-validation performance panels
(A: prediction-score distributions, B: precision-recall curves,
 C: mean ROC curves with +/- SD band)

Inputs (per-sample predictions from the released checkpoints)
-------------------------------------------------------------
  MFMNet      : results/predictions/mfmnet/pred_full_fold{1..5}.csv
  DeepDDS-GCN : results/predictions/deepdds_gcn/deepdds_gcn_predictions_fold{1..5}.csv

Output: fig3_performance.tif (300 dpi, LZW, RGB) + fig3_performance.pdf (vector)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

C_DDS_NEG, C_DDS_POS = "#4C72B0", "#D1495B"
C_MFM_NEG, C_MFM_POS = "#55A868", "#FFAE42"


def load_predictions(pred_dir: Path, pattern: str):
    folds = []
    for f in range(1, 6):
        folds.append(pd.read_csv(pred_dir / pattern.format(f=f)))
    return folds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mfmnet-pred-dir", default="results/predictions/mfmnet")
    ap.add_argument("--deepdds-pred-dir", default="results/predictions/deepdds_gcn")
    ap.add_argument("--outdir", default="figures/output/fig3")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mfm_folds = load_predictions(Path(args.mfmnet_pred_dir), "pred_full_fold{f}.csv")
    dds_folds = load_predictions(Path(args.deepdds_pred_dir), "deepdds_gcn_predictions_fold{f}.csv")

    mfm_all = pd.concat(mfm_folds, ignore_index=True)
    dds_all = pd.concat(dds_folds, ignore_index=True)

    fig = plt.figure(figsize=(12.5, 12.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.48, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ---- A: 分数分布（密度直方图 + KDE 虚线轮廓）----
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    xs = np.linspace(0, 1, 300)
    for scores, color, label in (
        (dds_all[dds_all.label == 0].score, C_DDS_NEG, "DeepDDS - Negative"),
        (dds_all[dds_all.label == 1].score, C_DDS_POS, "DeepDDS - Positive"),
        (mfm_all[mfm_all.label == 0].score, C_MFM_NEG, "MFMNet - Negative"),
        (mfm_all[mfm_all.label == 1].score, C_MFM_POS, "MFMNet - Positive"),
    ):
        ax_a.hist(scores, bins=60, density=True, alpha=0.55, color=color, label=label)
        kde = gaussian_kde(scores)
        ax_a.plot(xs, kde(xs), "--", color=color, linewidth=1.4)
    ax_a.set_xlabel("Prediction Score")
    ax_a.set_ylabel("Density")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    ax_a.text(-0.06, 1.06, "A", transform=ax_a.transAxes, fontsize=26, fontweight="bold")

    # ---- B: PR 曲线 ----
    for d, color, name in ((dds_all, "blue", "DeepDDS"), (mfm_all, "red", "MFMNet")):
        from sklearn.metrics import average_precision_score  # noqa: PLC0415
        prec, rec, _ = precision_recall_curve(d.label, d.score)
        ap_ = average_precision_score(d.label, d.score)
        ax_b.plot(rec, prec, color=color, linewidth=1.6, label=f"{name} (PR-AUC = {ap_:.4f})")
    ax_b.set_xlabel("Recall")
    ax_b.set_ylabel("Precision")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(loc="lower left", frameon=True)
    ax_b.text(-0.14, 1.06, "B", transform=ax_b.transAxes, fontsize=26, fontweight="bold")

    # ---- C: 五折 ROC + 均值 ± SD ----
    mean_fpr = np.linspace(0, 1, 300)
    tprs, aucs = [], []
    colors = ["#7B68EE", "#E96AA0", "#5FA05A", "#E8B14C", "#9B6BB3"]
    for i, d in enumerate(mfm_folds):
        fpr, tpr, _ = roc_curve(d.label, d.score)
        auc = roc_auc_score(d.label, d.score)
        aucs.append(auc)
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs.append(interp)
        ax_c.plot(mean_fpr, interp, color=colors[i], alpha=0.7, linewidth=1.3,
                  label=f"Fold {i+1} (AUC = {auc:.3f})")
    mean_tpr = np.mean(tprs, axis=0)
    sd_tpr = np.std(tprs, axis=0)
    ax_c.fill_between(mean_fpr, np.clip(mean_tpr - sd_tpr, 0, 1), np.clip(mean_tpr + sd_tpr, 0, 1),
                      color="gray", alpha=0.25, label="±1 SD")
    ax_c.plot(mean_fpr, mean_tpr, color="black", linewidth=2.2,
              label=f"Mean ROC (AUROC = {np.mean(aucs):.3f})")
    ax_c.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2)
    ax_c.set_xlabel("False Positive Rate")
    ax_c.set_ylabel("True Positive Rate")
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(loc="lower right", frameon=True, fontsize=11)
    ax_c.text(-0.14, 1.06, "C", transform=ax_c.transAxes, fontsize=26, fontweight="bold")

    tif = outdir / "fig3_performance.tif"
    fig.savefig(tif, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(outdir / "fig3_performance.pdf")
    plt.close(fig)
    print(f"已输出: {tif} 与 {outdir/'fig3_performance.pdf'}")


if __name__ == "__main__":
    main()
