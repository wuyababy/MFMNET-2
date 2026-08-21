#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recover the original strict same-fold baseline metric files from a messy project
directory and recompute Table 3 against the final masked MFMNet full five-fold
results.

Scientific safeguards
---------------------
- Candidate baseline files are identified ONLY by model name/file-name hints and
  agreement with the already reported five-fold mean±SD in the manuscript.
- No candidate is selected based on p-value, effect size, or whether MFMNet wins.
- Paired statistics are computed only from five fold-level values aligned by fold.
- Holm correction is applied separately to the three AUROC and three PR-AUC tests.

Usage
-----
py recover_table3_stats.py --root "D:\\个人资料\\lunwen-MFMNET\\codes"

Outputs
-------
table3_recovery_candidates.csv
table3_updated_statistics.csv
table3_updated_statistics.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import stats

FULL = {
    "AUROC": np.array([
        0.9222852822244719,
        0.9279013489856864,
        0.9157930204295565,
        0.9191602465331279,
        0.9283539377289378,
    ], dtype=float),
    "PR_AUC": np.array([
        0.9094697836499168,
        0.9178191068216555,
        0.8998231917107220,
        0.9151006884257121,
        0.9213978671120615,
    ], dtype=float),
}

TARGET = {
    "DeepDDS-GCN": {
        "AUROC_mean": 0.8968, "AUROC_SD": 0.0079,
        "PR_AUC_mean": 0.8843, "PR_AUC_SD": 0.0130,
    },
    "DeepDDS-GAT": {
        "AUROC_mean": 0.8911, "AUROC_SD": 0.0142,
        "PR_AUC_mean": 0.8805, "PR_AUC_SD": 0.0154,
    },
    "XGBoost": {
        "AUROC_mean": 0.9149, "AUROC_SD": 0.0028,
        "PR_AUC_mean": 0.9054, "PR_AUC_SD": 0.0058,
    },
}

MEAN_TOL = 1.5e-4
SD_TOL = 1.5e-4


def norm_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        low = str(c).strip().lower().replace("-", "_")
        if low in {"fold", "outer_fold"}:
            rename[c] = "fold"
        elif low in {"auroc", "auc"}:
            rename[c] = "AUROC"
        elif low in {"pr_auc", "prauc", "aupr", "auprc", "pr_auc_"}:
            rename[c] = "PR_AUC"
        elif low == "model":
            rename[c] = "Model"
        elif low == "variant":
            rename[c] = "variant"
    return df.rename(columns=rename)


def model_hint(path: Path, label: str = "") -> str | None:
    s = (str(path).lower() + " " + str(label).lower())
    if "xgboost" in s or "xgb" in s:
        return "XGBoost"
    if "deepdds" in s and "gcn" in s:
        return "DeepDDS-GCN"
    if "deepdds" in s and "gat" in s:
        return "DeepDDS-GAT"
    # same5fold outputs may use short model labels
    lab = str(label).strip().lower()
    if lab in {"gcn", "deepdds-gcn", "deepdds (gcn)"}:
        return "DeepDDS-GCN"
    if lab in {"gat", "deepdds-gat", "deepdds (gat)"}:
        return "DeepDDS-GAT"
    if lab in {"xgboost", "xgb"}:
        return "XGBoost"
    return None


def summarize5(g: pd.DataFrame):
    if "fold" not in g.columns or "AUROC" not in g.columns or "PR_AUC" not in g.columns:
        return None
    z = g[["fold", "AUROC", "PR_AUC"]].copy()
    z["fold"] = pd.to_numeric(z["fold"], errors="coerce")
    z["AUROC"] = pd.to_numeric(z["AUROC"], errors="coerce")
    z["PR_AUC"] = pd.to_numeric(z["PR_AUC"], errors="coerce")
    z = z.dropna()
    z["fold"] = z["fold"].astype(int)
    z = z[z["fold"].isin([1,2,3,4,5])]
    z = z.sort_values("fold").drop_duplicates("fold", keep="last")
    if list(z["fold"]) != [1,2,3,4,5]:
        return None
    a = z["AUROC"].to_numpy(float)
    p = z["PR_AUC"].to_numpy(float)
    return z, {
        "AUROC_mean": float(a.mean()),
        "AUROC_SD": float(a.std(ddof=1)),
        "PR_AUC_mean": float(p.mean()),
        "PR_AUC_SD": float(p.std(ddof=1)),
    }


def matches_target(summary: dict, target: dict) -> bool:
    return (
        abs(summary["AUROC_mean"] - target["AUROC_mean"]) <= MEAN_TOL and
        abs(summary["AUROC_SD"] - target["AUROC_SD"]) <= SD_TOL and
        abs(summary["PR_AUC_mean"] - target["PR_AUC_mean"]) <= MEAN_TOL and
        abs(summary["PR_AUC_SD"] - target["PR_AUC_SD"]) <= SD_TOL
    )


def score(summary: dict, target: dict) -> float:
    return sum(abs(summary[k] - target[k]) for k in target)


def discover(root: Path):
    candidates = []
    for path in root.rglob("*.csv"):
        # Skip our own outputs
        if path.name in {"table3_recovery_candidates.csv", "table3_updated_statistics.csv"}:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df = norm_metric_columns(df)

        groups = []
        if "Model" in df.columns:
            for label, g in df.groupby("Model", dropna=False):
                groups.append((str(label), g))
        elif "variant" in df.columns:
            # Baseline files should generally not need this, but preserve support.
            for label, g in df.groupby("variant", dropna=False):
                groups.append((str(label), g))
        else:
            groups.append(("", df))

        for label, g in groups:
            hint = model_hint(path, label)
            if hint is None:
                continue
            result = summarize5(g)
            if result is None:
                continue
            z, sm = result
            target = TARGET[hint]
            row = {
                "model": hint,
                "path": str(path),
                "group_label": label,
                **sm,
                "matches_reported_mean_sd": matches_target(sm, target),
                "target_distance": score(sm, target),
            }
            for _, rr in z.iterrows():
                f = int(rr["fold"])
                row[f"fold{f}_AUROC"] = float(rr["AUROC"])
                row[f"fold{f}_PR_AUC"] = float(rr["PR_AUC"])
            candidates.append(row)
    return pd.DataFrame(candidates)


def pick_models(cand: pd.DataFrame):
    picked = {}
    issues = []
    for model, target in TARGET.items():
        c = cand[(cand["model"] == model) & (cand["matches_reported_mean_sd"] == True)].copy()
        if c.empty:
            issues.append(f"{model}: no five-fold candidate matches the reported mean±SD.")
            continue

        # Collapse exact duplicate fold arrays arising from copied result files.
        sig_cols = [f"fold{i}_{m}" for i in range(1,6) for m in ("AUROC","PR_AUC")]
        c["_signature"] = c[sig_cols].round(12).astype(str).agg("|".join, axis=1)
        unique = c.sort_values("target_distance").drop_duplicates("_signature")
        if len(unique) > 1:
            # If multiple distinct arrays match the rounded manuscript summaries,
            # do NOT choose based on statistical outcome.
            issues.append(
                f"{model}: {len(unique)} distinct five-fold arrays match the manuscript summaries; "
                "selection is ambiguous. See table3_recovery_candidates.csv."
            )
            continue
        picked[model] = unique.iloc[0]
    return picked, issues


def holm_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m-rank) * p[idx])
        running = max(running, val)
        out[idx] = running
    return out


def paired_row(model, metric, base):
    full = FULL[metric]
    d = full - base
    md = float(d.mean())
    lo, hi = stats.t.interval(
        0.95, len(d)-1, loc=md, scale=float(stats.sem(d))
    )
    t, p = stats.ttest_rel(full, base)
    return {
        "Comparison": f"MFMNet vs {model}",
        "Metric": metric,
        "MFMNet backbone": f"{FULL[metric].mean():.4f} ± {FULL[metric].std(ddof=1):.4f}",
        "Baseline": f"{base.mean():.4f} ± {base.std(ddof=1):.4f}",
        "Mean paired difference": md,
        "95% CI lower": float(lo),
        "95% CI upper": float(hi),
        "Paired t-test p": float(p),
        "t statistic": float(t),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project/codes directory to scan recursively")
    ap.add_argument("--output", default="./table3_recomputed")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cand = discover(root)
    cand = cand.sort_values(["model","matches_reported_mean_sd","target_distance"],
                            ascending=[True,False,True])
    cand.to_csv(outdir / "table3_recovery_candidates.csv", index=False)

    picked, issues = pick_models(cand)
    if issues:
        print("\nRECOVERY NOT YET UNIQUE:")
        for x in issues:
            print(" -", x)
        print("\nCandidate audit saved to:")
        print(outdir / "table3_recovery_candidates.csv")
        raise SystemExit(2)

    rows = []
    # Preserve manuscript row ordering.
    for metric in ["AUROC", "PR_AUC"]:
        for model in ["DeepDDS-GCN", "DeepDDS-GAT", "XGBoost"]:
            r = picked[model]
            base = np.array([float(r[f"fold{i}_{metric}"]) for i in range(1,6)])
            rows.append(paired_row(model, metric, base))

    df = pd.DataFrame(rows)

    # Holm separately within AUROC and PR-AUC families (3 comparisons each).
    df["Holm-adjusted p"] = np.nan
    for metric in ["AUROC", "PR_AUC"]:
        mask = df["Metric"] == metric
        df.loc[mask, "Holm-adjusted p"] = holm_adjust(
            df.loc[mask, "Paired t-test p"].to_numpy(float)
        )

    df.to_csv(outdir / "table3_updated_statistics.csv", index=False)

    lines = []
    lines.append("UPDATED TABLE 3 STATISTICS")
    lines.append("=" * 120)
    show = df.copy()
    show["Mean paired difference"] = show["Mean paired difference"].map(lambda x: f"{x:+.4f}")
    show["95% CI of difference"] = [
        f"{lo:.4f}–{hi:.4f}" for lo,hi in zip(show["95% CI lower"], show["95% CI upper"])
    ]
    show["Paired t-test p"] = show["Paired t-test p"].map(lambda x: f"{x:.4f}")
    show["Holm-adjusted p"] = show["Holm-adjusted p"].map(lambda x: f"{x:.4f}")
    cols = [
        "Comparison","Metric","MFMNet backbone","Baseline",
        "Mean paired difference","95% CI of difference",
        "Paired t-test p","Holm-adjusted p"
    ]
    txt = show[cols].to_string(index=False)
    lines.append(txt)
    lines.append("\nSelected source files:")
    for model, r in picked.items():
        lines.append(f"{model}: {r['path']} {('['+str(r['group_label'])+']') if r['group_label'] else ''}")

    report = "\n".join(lines)
    (outdir / "table3_updated_statistics.txt").write_text(report, encoding="utf-8")
    print(report)
    print("\nSaved:")
    print(outdir / "table3_updated_statistics.csv")
    print(outdir / "table3_updated_statistics.txt")


if __name__ == "__main__":
    main()
