#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checkpoint-matched functional ablation for MFMNet.

Scientific purpose
------------------
Quantify the functional contribution of TGP branches and CSAR while keeping the
trained FULL model checkpoint fixed. No retraining, seed search, checkpoint
selection, threshold tuning, or test-label-driven parameter adjustment is used.

This is a FUNCTIONAL (post-training knockout) ablation, not a retrained-capacity
ablation. It should be described as such in the manuscript/reviewer response.

Required
--------
- MFMNet_reviewer_clean_save_ckpt.py in the current directory
- five validation-selected FULL checkpoints:
    mfmnet_full_fold1.pt ... mfmnet_full_fold5.pt

Recommended checkpoint source
-----------------------------
Use exactly the checkpoint directory from the prior run whose FULL results you
intend to preserve. If that run had fold-1 AUROC = 0.9245515928, this script's
"full" fold-1 result should reproduce that value (within floating-point readout).

Functional variants
-------------------
full
without_gcn          : zero GCN branch output, keep trained GIN/GAT + trained CSAR
without_gin          : zero GIN branch output, keep trained GCN/GAT + trained CSAR
without_gat          : zero GAT branch output, keep trained GCN/GIN + trained CSAR
gcn_only_functional    : keep only GCN branch output through trained CSAR
gin_only_functional    : keep only GIN branch output through trained CSAR
gat_only_functional    : keep only GAT branch output through trained CSAR
no_graph               : zero the entire graph representation before classifier
without_adaptive_gate: bypass CSAR weight_net, keep the trained CSAR projection
uniform_gate         : replace learned gate by fixed 0.5 gate, keep projection
mean_fusion          : replace CSAR with equal mean of GCN/GIN/GAT outputs

No variant introduces newly trained parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats


FUNCTIONAL_ABLATION_VERSION = "2026-08-15-v3-tgp-retention-no-graph"

VARIANTS = [
    "full",
    "without_gcn",
    "without_gin",
    "without_gat",
    "gcn_only_functional",
    "gin_only_functional",
    "gat_only_functional",
    "no_graph",
    "without_adaptive_gate",
    "uniform_gate",
    "mean_fusion",
]

DISPLAY = {
    "full": "Full MFMNet",
    "without_gcn": "Functional knockout: GCN",
    "without_gin": "Functional knockout: GIN",
    "without_gat": "Functional knockout: GAT",
    "gcn_only_functional": "Functional retention: GCN only",
    "gin_only_functional": "Functional retention: GIN only",
    "gat_only_functional": "Functional retention: GAT only",
    "no_graph": "No graph representation",
    "without_adaptive_gate": "CSAR without adaptive gate",
    "uniform_gate": "CSAR with uniform gate",
    "mean_fusion": "Equal-mean graph fusion",
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-code", default="./MFMNet_reviewer_clean_save_ckpt.py")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory containing mfmnet_full_fold1.pt ... fold5.pt")
    p.add_argument("--output", default="./result/functional_ablation_checkpoint_matched")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0,
                   help="Must match the outer-fold construction used for the checkpoints")
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--folds", nargs="+", type=int, default=[1,2,3,4,5])
    return p.parse_args()


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("mfmnet_functional_base", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_full_checkpoint(base, ckpt_path: Path, device):
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
        metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    elif isinstance(payload, dict):
        state = payload
        metadata = {}
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(payload)}")

    model = base.MFMNetReviewerModel(variant="full", dropout_rate=0.3)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, metadata


def install_functional_variant(model, base, variant: str):
    """
    Replace only _graph_features. All learned parameters remain untouched.
    """
    if variant == "full":
        return

    def functional_graph_features(self, batch, batch_size: int):
        gcn = self._gcn(batch)
        gin = self._gin(batch)
        gat = self._gat(batch)

        if variant == "without_gcn":
            gcn = torch.zeros_like(gcn)
            multi = torch.cat([gcn, gin, gat], dim=1)
            fused = self.graph_fusion(multi)

        elif variant == "without_gin":
            gin = torch.zeros_like(gin)
            multi = torch.cat([gcn, gin, gat], dim=1)
            fused = self.graph_fusion(multi)

        elif variant == "without_gat":
            gat = torch.zeros_like(gat)
            multi = torch.cat([gcn, gin, gat], dim=1)
            fused = self.graph_fusion(multi)

        elif variant == "gcn_only_functional":
            gin = torch.zeros_like(gin)
            gat = torch.zeros_like(gat)
            multi = torch.cat([gcn, gin, gat], dim=1)
            fused = self.graph_fusion(multi)

        elif variant == "gin_only_functional":
            gcn = torch.zeros_like(gcn)
            gat = torch.zeros_like(gat)
            multi = torch.cat([gcn, gin, gat], dim=1)
            fused = self.graph_fusion(multi)

        elif variant == "gat_only_functional":
            gcn = torch.zeros_like(gcn)
            gin = torch.zeros_like(gin)
            multi = torch.cat([gcn, gin, gat], dim=1)
            fused = self.graph_fusion(multi)

        elif variant == "no_graph":
            # Remove the entire learned graph-level representation while leaving
            # drug-embedding and cell-line pathways untouched.
            fused = torch.zeros_like(gcn)

        elif variant == "without_adaptive_gate":
            multi = torch.cat([gcn, gin, gat], dim=1)
            # Isolate adaptive gating: keep the learned final projection but
            # bypass the learned weight_net entirely.
            fused = self.graph_fusion.projection(multi)

        elif variant == "uniform_gate":
            multi = torch.cat([gcn, gin, gat], dim=1)
            # Neutral fixed sigmoid-midpoint gate. No learned routing weights.
            fused = self.graph_fusion.projection(multi * 0.5)

        elif variant == "mean_fusion":
            fused = (gcn + gin + gat) / 3.0

        else:
            raise ValueError(variant)

        expected_nodes = batch_size * base.MAX_ATOMS
        if fused.shape[0] != expected_nodes:
            raise RuntimeError(
                f"Expected {expected_nodes} nodes "
                f"({batch_size} x {base.MAX_ATOMS}), got {fused.shape[0]}"
            )
        return fused.reshape(batch_size, base.MAX_ATOMS, base.GRAPH_DIM)

    model._graph_features = types.MethodType(functional_graph_features, model)


def predict_with_threshold(base, model, device, loader, threshold: float):
    # Reuse exact model probability path, then apply explicitly fixed threshold.
    ys, scores, indices = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1]
            ys.append(batch.label.detach().cpu().numpy())
            scores.append(prob.detach().cpu().numpy())
            indices.append(batch.index.detach().cpu().numpy())
    y = np.concatenate(ys).astype(int)
    score = np.concatenate(scores)
    idx = np.concatenate(indices).astype(int)
    pred = (score > threshold).astype(int)
    return y, score, pred, idx



def holm_adjust(raw_p_values, alpha=0.05):
    """Holm step-down multiplicity adjustment without statsmodels."""
    p = np.asarray(raw_p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted_value = (m - rank) * p[idx]
        running_max = max(running_max, adjusted_value)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]

    reject = adjusted < alpha
    return reject, adjusted


def mean_sd(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1))


def paired_stats(full_vals, ablated_vals):
    a = np.asarray(full_vals, dtype=float)
    b = np.asarray(ablated_vals, dtype=float)
    d = a - b
    n = len(d)
    md = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n-1)
    ci_lo = md - tcrit * se
    ci_hi = md + tcrit * se
    tstat, p = stats.ttest_rel(a, b)
    return {
        "mean_paired_difference_full_minus_ablation": md,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "paired_t": float(tstat),
        "raw_p": float(p),
    }


def main():
    args = parse_args()
    model_code = Path(args.model_code).resolve()
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    base = load_base(model_code)
    device = base.choose_device(args.device)
    dataset = base.DrugPairDataset(Path(args.data_root))
    outer_folds = base.make_outer_folds(len(dataset), args.seed, 5)

    rows = []
    manifest = {
        "analysis_type": "checkpoint-matched functional post-training ablation",
        "retraining_performed": False,
        "test_labels_used_for_model_or_checkpoint_selection": False,
        "seed_search_performed": False,
        "threshold_search_performed": False,
        "threshold": args.threshold,
        "model_code": str(model_code),
        "model_code_sha256": sha256_file(model_code),
        "checkpoint_dir": str(ckpt_dir),
        "device": str(device),
        "variants": VARIANTS,
        "checkpoint_files": {},
    }

    for fold in args.folds:
        ckpt = ckpt_dir / f"mfmnet_full_fold{fold}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        manifest["checkpoint_files"][str(fold)] = {
            "path": str(ckpt),
            "sha256": sha256_file(ckpt),
        }

        train_idx, val_idx, test_idx = base.make_fold_indices(
            dataset.labels, outer_folds, fold, args.val_fraction, args.seed
        )
        fold_seed = args.seed + 1000 * fold
        test_loader = base.make_loader(
            dataset, test_idx, args.batch_size, False,
            fold_seed + 3, args.num_workers
        )

        for variant in VARIANTS:
            model, metadata = load_full_checkpoint(base, ckpt, device)
            install_functional_variant(model, base, variant)

            y, score, pred, idx = predict_with_threshold(
                base, model, device, test_loader, args.threshold
            )
            metrics = base.metric_dict(y, score, pred)

            row = {
                "variant": variant,
                "display_name": DISPLAY[variant],
                "fold": fold,
                **metrics,
                "n_test": int(len(test_idx)),
                "checkpoint": str(ckpt),
                "checkpoint_sha256": manifest["checkpoint_files"][str(fold)]["sha256"],
                "checkpoint_best_epoch": metadata.get("best_epoch", ""),
                "checkpoint_best_val_AUROC": metadata.get("best_val_AUROC", ""),
                "checkpoint_seed": metadata.get("seed", ""),
                "functional_ablation": True,
                "retrained": False,
                "threshold": args.threshold,
            }
            rows.append(row)
            print(json.dumps(row, indent=2, ensure_ascii=False))

            pred_df = pd.DataFrame({
                "sample_index": idx,
                "label": y,
                "score": score,
                "predicted_label": pred,
            }).sort_values("sample_index")
            pred_df.to_csv(
                out / f"pred_functional_{variant}_fold{fold}.csv", index=False
            )

    df = pd.DataFrame(rows)
    df.to_csv(out / "functional_ablation_fold_metrics.csv", index=False)

    metric_cols = ["AUROC", "PR_AUC", "ACC", "BACC", "Precision", "TPR", "KAPPA", "F1"]
    summary_rows = []
    for variant in VARIANTS:
        sub = df[df["variant"] == variant].sort_values("fold")
        r = {"variant": variant, "display_name": DISPLAY[variant]}
        for m in metric_cols:
            mu, sd = mean_sd(sub[m].to_numpy())
            r[f"{m}_mean"] = mu
            r[f"{m}_sd"] = sd
        summary_rows.append(r)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "functional_ablation_summary.csv", index=False)

    # Paired tests: primary inference only for AUROC and PR-AUC.
    paired_rows = []
    full = df[df["variant"] == "full"].sort_values("fold")
    for metric in ["AUROC", "PR_AUC"]:
        family_rows = []
        for variant in VARIANTS:
            if variant == "full":
                continue
            ab = df[df["variant"] == variant].sort_values("fold")
            if list(full["fold"]) != list(ab["fold"]):
                raise RuntimeError("Fold mismatch")
            st = paired_stats(full[metric].to_numpy(), ab[metric].to_numpy())
            rr = {
                "comparison": f"full vs {variant}",
                "variant": variant,
                "metric": metric,
                "full_mean": float(full[metric].mean()),
                "ablation_mean": float(ab[metric].mean()),
                **st,
            }
            family_rows.append(rr)

        raw_ps = [r["raw_p"] for r in family_rows]
        reject, p_adj = holm_adjust(raw_ps)
        for r, adj, rej in zip(family_rows, p_adj, reject):
            r["holm_adjusted_p"] = float(adj)
            r["holm_reject_0_05"] = bool(rej)
            paired_rows.append(r)

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out / "functional_ablation_paired_tests.csv", index=False)

    (out / "functional_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== PAIRED TESTS ===")
    print(paired_df.to_string(index=False))
    print("\nOutput:", out.resolve())


if __name__ == "__main__":
    main()
