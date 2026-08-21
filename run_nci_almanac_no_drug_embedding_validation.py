#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NCI-ALMANAC cross-dataset validation for the pre-defined
MFMNet no_drug_embedding configuration.

Why this configuration
----------------------
The supplied 768-d auxiliary drug embeddings are record-dependent and cannot
be reconstructed consistently for independent NCI-ALMANAC records. This
script therefore evaluates the already-defined no_drug_embedding MFMNet
variant, which retains the molecular-graph branches (GCN/GIN/GATv2), CSAR
fusion, cell-line encoder, and classifier, while omitting only the
non-transferable auxiliary drug-embedding branch.

Protocol
--------
- Input external table must already have exact O'Neil drug-pair-cell overlaps
  removed and canonical duplicate pair-cell records resolved.
- Drug molecular graphs are rebuilt from the O'Neil SMILES inventory.
- Cell-line 954-d profiles are taken from O'Neil only after consistency audit.
- Drug embeddings are zero placeholders because the no_drug_embedding model
  does not use them.
- Five validation-selected checkpoints are frozen and equally averaged.
- No NCI label is used for checkpoint selection, architecture selection,
  threshold tuning, or ensemble weighting.
- Fixed classification threshold: 0.5.
- Primary discrimination metrics: AUROC and PR-AUC.
- 95% CIs: percentile cluster bootstrap resampling canonical drug pairs,
  preserving all cell-line observations within each sampled pair.

This is cross-dataset validation with shared drug/cell entities, not
unseen-drug or unseen-cell-line validation. NCI-ALMANAC ComboScore and
O'Neil Loewe are different synergy metrics; the prepared external labels use
the pre-specified harmonized rule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
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
from torch.utils.data import Dataset, DataLoader


SCRIPT_VERSION = "2026-08-13-nci-almanac-no-drug-embedding-v1"
DRUG_EMBED_DIM = 768
CELL_DIM = 954
DEFAULT_THRESHOLD = 0.5
DEFAULT_BOOTSTRAPS = 5000
DEFAULT_BOOTSTRAP_SEED = 20260813


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    p.add_argument("--data-root", default="./data")
    p.add_argument(
        "--external-csv",
        default="./result/almanac_feasibility_audit/external_classifiable_deduplicated_mean.csv",
    )
    p.add_argument(
        "--checkpoints-dir",
        default="./result/mfmnet_no_drug_embedding_external_ckpt/checkpoints",
    )
    p.add_argument(
        "--model-code",
        default="./MFMNet_external_no_drug_embedding_save_ckpt.py",
        help="Exact model-code file used to create the no_drug_embedding checkpoints",
    )
    p.add_argument("--output", default="./result/nci_almanac_no_drug_embedding_external_validation")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    p.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--feature-atol", type=float, default=1e-6)
    p.add_argument("--feature-rtol", type=float, default=1e-6)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_python_module(path: Path):
    spec = importlib.util.spec_from_file_location("mfmnet_external_model_code", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model code from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = [
        "MFMNetReviewerModel",
        "PairSample",
        "collate_pair_samples",
        "smiles_to_fixed_graph",
        "choose_device",
        "sync",
        "Chem",
    ]
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise AttributeError(f"Model code missing required objects: {missing}")
    return module


def canonical_smiles(raw: str, Chem) -> str:
    mol = Chem.MolFromSmiles(str(raw))
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {raw!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def allclose_to_reference(matrix: np.ndarray, atol: float, rtol: float) -> Tuple[bool, float]:
    if len(matrix) == 0:
        return False, float("nan")
    ref = matrix[0]
    diff = np.abs(matrix - ref[None, :])
    return bool(np.allclose(matrix, ref[None, :], atol=atol, rtol=rtol)), float(np.max(diff))


def validate_external_table(ext: pd.DataFrame):
    required = {"local_drug1", "local_drug2", "local_cell", "external_label"}
    missing = required.difference(ext.columns)
    if missing:
        raise ValueError(f"External CSV missing required columns: {sorted(missing)}")
    if ext[list(required)].isna().any().any():
        raise ValueError("External CSV contains missing mapped drug/cell/label values")

    labels = sorted(pd.unique(ext["external_label"].astype(int)).tolist())
    if labels != [0, 1]:
        raise ValueError(f"Expected external labels [0,1], got {labels}")

    if "exact_oneil_triplet_overlap" in ext.columns:
        bad = ext["exact_oneil_triplet_overlap"].astype(str).str.lower().isin(["true", "1"])
        if bad.any():
            raise ValueError(f"External CSV contains {int(bad.sum())} exact O'Neil triplet overlaps")

    if "gray_zone_0_to_10" in ext.columns:
        bad = ext["gray_zone_0_to_10"].astype(str).str.lower().isin(["true", "1"])
        if bad.any():
            raise ValueError(f"External CSV contains {int(bad.sum())} gray-zone records")

    keys = []
    for r in ext[["local_drug1", "local_drug2", "local_cell"]].itertuples(index=False):
        a, b = sorted([str(r.local_drug1), str(r.local_drug2)])
        keys.append((a, b, str(r.local_cell)))
    dup_n = int(pd.Series(keys).duplicated().sum())
    if dup_n:
        raise ValueError(
            f"External CSV contains {dup_n} duplicate canonical drug-pair-cell triplets. "
            "Resolve duplicates before inference."
        )


def build_transferable_inventory(
    ddi: pd.DataFrame,
    needed_drugs: List[str],
    needed_cells: List[str],
    Chem,
    atol: float,
    rtol: float,
):
    required = {"drug1Name", "drug2Name", "drug1", "drug2", "cell"}
    missing = required.difference(ddi.columns)
    if missing:
        raise ValueError(f"all_ddi.csv missing required columns: {sorted(missing)}")
    if ddi.shape[1] < 9 + CELL_DIM:
        raise ValueError(f"Expected at least {9 + CELL_DIM} columns in all_ddi.csv")

    cell_matrix = ddi.iloc[:, 9:9 + CELL_DIM].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(cell_matrix).all():
        raise ValueError("Non-finite cell-line features detected")

    d1_names = ddi["drug1Name"].astype(str).to_numpy()
    d2_names = ddi["drug2Name"].astype(str).to_numpy()

    drug_smiles: Dict[str, str] = {}
    drug_audit_rows = []
    failures = []

    for drug in sorted(needed_drugs):
        lidx = np.flatnonzero(d1_names == drug)
        ridx = np.flatnonzero(d2_names == drug)
        if len(lidx) + len(ridx) == 0:
            failures.append(f"Drug {drug!r} not found in O'Neil inventory")
            continue

        raw_smiles = []
        if len(lidx):
            raw_smiles.extend(ddi.iloc[lidx]["drug1"].astype(str).tolist())
        if len(ridx):
            raw_smiles.extend(ddi.iloc[ridx]["drug2"].astype(str).tolist())

        canon = [canonical_smiles(x, Chem) for x in raw_smiles]
        unique = sorted(set(canon))
        ok = len(unique) == 1
        drug_audit_rows.append({
            "drug": drug,
            "n_left_occurrences": int(len(lidx)),
            "n_right_occurrences": int(len(ridx)),
            "n_total_occurrences": int(len(raw_smiles)),
            "n_unique_canonical_smiles": int(len(unique)),
            "smiles_consistent": bool(ok),
            "canonical_smiles": unique[0] if ok else " | ".join(unique),
        })
        if not ok:
            failures.append(f"Multiple non-equivalent SMILES found for {drug!r}: {unique}")
        else:
            drug_smiles[drug] = raw_smiles[0]

    cell_inventory: Dict[str, np.ndarray] = {}
    cell_audit_rows = []

    cells = ddi["cell"].astype(str).to_numpy()
    for cell in sorted(needed_cells):
        idx = np.flatnonzero(cells == cell)
        if len(idx) == 0:
            failures.append(f"Cell line {cell!r} not found in O'Neil inventory")
            continue
        mat = cell_matrix[idx]
        ok, max_abs = allclose_to_reference(mat, atol=atol, rtol=rtol)
        cell_audit_rows.append({
            "cell": cell,
            "n_occurrences": int(len(idx)),
            "feature_consistent": bool(ok),
            "feature_max_abs_diff": float(max_abs),
        })
        if not ok:
            failures.append(
                f"Cell-line feature inconsistency for {cell!r}; max abs diff={max_abs:.8g}"
            )
        else:
            cell_inventory[cell] = mat[0].copy()

    return (
        drug_smiles,
        cell_inventory,
        pd.DataFrame(drug_audit_rows),
        pd.DataFrame(cell_audit_rows),
        failures,
    )


class ExternalDataset(Dataset):
    def __init__(self, ext, drug_smiles, cell_inventory, model_module):
        self.ext = ext.reset_index(drop=True).copy()
        self.cell_inventory = cell_inventory
        self.model_module = model_module
        self.zero_embedding = np.zeros(DRUG_EMBED_DIM, dtype=np.float32)

        drugs = sorted(
            set(self.ext["local_drug1"].astype(str))
            | set(self.ext["local_drug2"].astype(str))
        )
        print(f"Building molecular graphs for {len(drugs)} drugs ...")
        self.graph_cache = {}
        for i, drug in enumerate(drugs, start=1):
            self.graph_cache[drug] = model_module.smiles_to_fixed_graph(drug_smiles[drug])
            print(f"  graph {i}/{len(drugs)}: {drug}")

    def __len__(self):
        return len(self.ext)

    def __getitem__(self, idx):
        r = self.ext.iloc[int(idx)]
        d1 = str(r["local_drug1"])
        d2 = str(r["local_drug2"])
        cell = str(r["local_cell"])
        return self.model_module.PairSample(
            index=int(idx),
            left_graph=self.graph_cache[d1],
            right_graph=self.graph_cache[d2],
            left_embedding=torch.from_numpy(self.zero_embedding.copy()),
            right_embedding=torch.from_numpy(self.zero_embedding.copy()),
            cell=torch.from_numpy(self.cell_inventory[cell]),
            label=int(r["external_label"]),
        )


def metric_dict(y, score, threshold):
    pred = (score >= threshold).astype(int)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y, score)
    return {
        "AUROC": float(roc_auc_score(y, score)),
        "PR_AUC": float(metrics.auc(recall_curve, precision_curve)),
        "ACC": float(accuracy_score(y, pred)),
        "BACC": float(balanced_accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "TPR": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "KAPPA": float(cohen_kappa_score(y, pred)),
        "threshold": float(threshold),
    }


def canonical_pair_ids(ext: pd.DataFrame) -> np.ndarray:
    out = []
    for r in ext[["local_drug1", "local_drug2"]].itertuples(index=False):
        a, b = sorted([str(r.local_drug1), str(r.local_drug2)])
        out.append(f"{a}|{b}")
    return np.asarray(out, dtype=object)


def cluster_bootstrap_primary_metrics(
    y: np.ndarray,
    score: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    clusters = np.unique(cluster_ids)
    cluster_to_idx = {c: np.flatnonzero(cluster_ids == c) for c in clusters}

    aucs = []
    prs = []
    attempts = 0
    max_attempts = max(n_boot * 20, 1000)

    while len(aucs) < n_boot and attempts < max_attempts:
        attempts += 1
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        idx_parts = [cluster_to_idx[c] for c in sampled_clusters]
        idx = np.concatenate(idx_parts)

        yy = y[idx]
        ss = score[idx]
        if len(np.unique(yy)) < 2:
            continue

        aucs.append(float(roc_auc_score(yy, ss)))
        pc, rc, _ = metrics.precision_recall_curve(yy, ss)
        prs.append(float(metrics.auc(rc, pc)))

    if len(aucs) < n_boot:
        raise RuntimeError(
            f"Only {len(aucs)} valid cluster-bootstrap replicates obtained "
            f"after {attempts} attempts"
        )

    rows = []
    for name, vals in [("AUROC", np.asarray(aucs)), ("PR_AUC", np.asarray(prs))]:
        rows.append({
            "metric": name,
            "bootstrap_replicates": int(n_boot),
            "bootstrap_unit": "canonical drug pair",
            "bootstrap_method": "percentile cluster bootstrap",
            "CI95_lower": float(np.percentile(vals, 2.5)),
            "CI95_upper": float(np.percentile(vals, 97.5)),
            "bootstrap_mean": float(np.mean(vals)),
            "bootstrap_SD": float(np.std(vals, ddof=1)),
        })
    return pd.DataFrame(rows)


def load_checkpoint(path: Path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError(f"Expected checkpoint dict with state_dict: {path}")
    meta = {k: obj.get(k) for k in ["variant", "fold", "best_epoch", "best_val_AUROC", "seed"]}
    return obj["state_dict"], meta


def predict_one_checkpoint(model, loader, device, model_module):
    model.eval()
    scores, labels, indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1]
            scores.append(prob.detach().cpu().numpy())
            labels.append(batch.label.detach().cpu().numpy())
            indices.append(batch.index.detach().cpu().numpy())
    model_module.sync(device)
    return np.concatenate(indices), np.concatenate(labels), np.concatenate(scores)


def main():
    args = parse_args()
    if not (0 < args.threshold < 1):
        raise ValueError("--threshold must lie between 0 and 1")
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    ddi_path = data_root / "raw" / "all_ddi.csv"
    ext_path = Path(args.external_csv)
    ckpt_dir = Path(args.checkpoints_dir)
    model_code_path = Path(args.model_code)

    for p in [ddi_path, ext_path, ckpt_dir, model_code_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    print("Loading exact model code:", model_code_path)
    model_module = load_python_module(model_code_path)
    Chem = model_module.Chem

    print("Loading O'Neil transferable feature inventory ...")
    ddi = pd.read_csv(ddi_path, low_memory=False)

    print("Loading prepared NCI-ALMANAC set ...")
    ext = pd.read_csv(ext_path, low_memory=False)
    validate_external_table(ext)

    needed_drugs = sorted(
        set(ext["local_drug1"].astype(str)) | set(ext["local_drug2"].astype(str))
    )
    needed_cells = sorted(set(ext["local_cell"].astype(str)))

    print(f"External records: {len(ext)}")
    print(f"External positives: {(ext['external_label'].astype(int) == 1).sum()}")
    print(f"External negatives: {(ext['external_label'].astype(int) == 0).sum()}")
    print(f"Mapped drugs: {len(needed_drugs)}")
    print(f"Mapped cell lines: {len(needed_cells)}")

    print("\nAuditing transferable features ...")
    drug_smiles, cell_inventory, drug_audit, cell_audit, failures = build_transferable_inventory(
        ddi, needed_drugs, needed_cells, Chem, args.feature_atol, args.feature_rtol
    )
    drug_audit.to_csv(outdir / "external_drug_smiles_audit.csv", index=False)
    cell_audit.to_csv(outdir / "external_cell_feature_audit.csv", index=False)

    feature_summary = {
        "script_version": SCRIPT_VERSION,
        "variant": "no_drug_embedding",
        "n_external_records": int(len(ext)),
        "n_external_positive": int((ext["external_label"].astype(int) == 1).sum()),
        "n_external_negative": int((ext["external_label"].astype(int) == 0).sum()),
        "n_needed_drugs": int(len(needed_drugs)),
        "n_needed_cells": int(len(needed_cells)),
        "smiles_audit_passed": bool(
            len(drug_audit) == len(needed_drugs) and drug_audit["smiles_consistent"].all()
        ),
        "cell_feature_audit_passed": bool(
            len(cell_audit) == len(needed_cells) and cell_audit["feature_consistent"].all()
        ),
        "auxiliary_768d_embeddings_used": False,
        "failures": failures,
    }
    (outdir / "external_feature_audit_summary.json").write_text(
        json.dumps(feature_summary, indent=2), encoding="utf-8"
    )

    if failures:
        print("\nFEATURE AUDIT FAILED. External inference stopped.")
        for x in failures:
            print(" -", x)
        sys.exit(2)

    print("Feature audit PASSED.")

    checkpoint_paths = [
        ckpt_dir / f"mfmnet_no_drug_embedding_fold{i}.pt"
        for i in range(1, 6)
    ]
    missing = [str(p) for p in checkpoint_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    checkpoint_states = []
    manifest = []
    for fold, p in enumerate(checkpoint_paths, start=1):
        state, meta = load_checkpoint(p)
        if meta.get("variant") != "no_drug_embedding":
            raise ValueError(
                f"{p.name}: variant={meta.get('variant')!r}, expected 'no_drug_embedding'"
            )
        if meta.get("fold") is not None and int(meta["fold"]) != fold:
            raise ValueError(f"{p.name}: fold metadata mismatch")
        checkpoint_states.append((fold, state, meta))
        manifest.append({
            "fold": fold,
            "file": str(p),
            "sha256": sha256_file(p),
            "size_mb": float(p.stat().st_size / (1024 ** 2)),
            "variant": meta.get("variant"),
            "best_epoch": meta.get("best_epoch"),
            "best_val_AUROC": meta.get("best_val_AUROC"),
            "seed": meta.get("seed"),
        })
    pd.DataFrame(manifest).to_csv(outdir / "external_checkpoint_manifest.csv", index=False)

    dataset = ExternalDataset(ext, drug_smiles, cell_inventory, model_module)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=model_module.collate_pair_samples,
        drop_last=False,
    )

    device = model_module.choose_device(args.device)
    print("\nInference device:", device)

    y_expected = ext["external_label"].astype(int).to_numpy()
    fold_scores = []
    per_ckpt = []

    for fold, state, meta in checkpoint_states:
        print(f"Predicting NCI-ALMANAC with fold {fold} checkpoint ...")
        model = model_module.MFMNetReviewerModel(
            variant="no_drug_embedding", dropout_rate=0.3
        ).to(device)
        incompat = model.load_state_dict(state, strict=True)
        if getattr(incompat, "missing_keys", []) or getattr(incompat, "unexpected_keys", []):
            raise RuntimeError(f"State-dict incompatibility for fold {fold}")

        idx, y, score = predict_one_checkpoint(model, loader, device, model_module)
        order = np.argsort(idx)
        idx, y, score = idx[order], y[order], score[order]

        if not np.array_equal(idx, np.arange(len(ext))):
            raise RuntimeError(f"External order mismatch for fold {fold}")
        if not np.array_equal(y, y_expected):
            raise RuntimeError(f"Label/order mismatch for fold {fold}")

        fold_scores.append(score)
        per_ckpt.append({
            "model": f"Embedding-independent MFMNet fold {fold}",
            "fold": fold,
            **metric_dict(y, score, args.threshold),
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass

    score_matrix = np.vstack(fold_scores)
    ensemble_score = score_matrix.mean(axis=0)
    ensemble_pred = (ensemble_score >= args.threshold).astype(int)
    ensemble_metrics = metric_dict(y_expected, ensemble_score, args.threshold)

    pd.DataFrame(per_ckpt).to_csv(
        outdir / "external_per_checkpoint_metrics.csv", index=False
    )
    pd.DataFrame([{
        "model": "Embedding-independent MFMNet 5-checkpoint equal-weight ensemble",
        **ensemble_metrics,
    }]).to_csv(outdir / "external_ensemble_metrics.csv", index=False)

    cluster_ids = canonical_pair_ids(ext)
    n_clusters = int(pd.Series(cluster_ids).nunique())
    print(
        f"\nBootstrap 95% CIs using {args.bootstrap_replicates} "
        f"canonical-drug-pair cluster replicates ({n_clusters} clusters) ..."
    )
    ci_df = cluster_bootstrap_primary_metrics(
        y_expected,
        ensemble_score,
        cluster_ids,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    ci_df.to_csv(outdir / "external_primary_metric_bootstrap_ci.csv", index=False)

    pred_df = ext.copy()
    pred_df["canonical_pair_cluster"] = cluster_ids
    for fold in range(1, 6):
        pred_df[f"prob_fold{fold}"] = score_matrix[fold - 1]
    pred_df["ensemble_probability"] = ensemble_score
    pred_df["predicted_label_threshold_0_5"] = ensemble_pred
    pred_df.to_csv(outdir / "external_predictions_all.csv", index=False)

    fpr, tpr, roc_thr = metrics.roc_curve(y_expected, ensemble_score)
    pd.DataFrame({"FPR": fpr, "TPR": tpr, "threshold": roc_thr}).to_csv(
        outdir / "external_roc_curve.csv", index=False
    )

    precision, recall, pr_thr = metrics.precision_recall_curve(
        y_expected, ensemble_score
    )
    pr_df = pd.DataFrame({"recall": recall, "precision": precision})
    pr_df["threshold"] = np.nan
    if len(pr_thr):
        pr_df.loc[:len(pr_thr)-1, "threshold"] = pr_thr
    pr_df.to_csv(outdir / "external_pr_curve.csv", index=False)

    cm = metrics.confusion_matrix(y_expected, ensemble_pred, labels=[0, 1])
    pd.DataFrame(
        cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
    ).to_csv(outdir / "external_confusion_matrix.csv")

    run_info = {
        "script_version": SCRIPT_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "variant": "no_drug_embedding",
        "external_csv": str(ext_path),
        "external_csv_sha256": sha256_file(ext_path),
        "model_code": str(model_code_path),
        "model_code_sha256": sha256_file(model_code_path),
        "checkpoints_dir": str(ckpt_dir),
        "n_checkpoints": 5,
        "ensemble_weights": [0.2] * 5,
        "classification_threshold": float(args.threshold),
        "auxiliary_768d_embeddings_used": False,
        "external_tuning": False,
        "checkpoint_selection_on_external_data": False,
        "ensemble_weight_tuning_on_external_data": False,
        "threshold_tuning_on_external_data": False,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_method": "percentile cluster bootstrap",
        "bootstrap_unit": "canonical drug pair",
        "n_bootstrap_clusters": n_clusters,
    }
    (outdir / "external_run_config.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    ci_map = {
        r.metric: (r.CI95_lower, r.CI95_upper)
        for r in ci_df.itertuples(index=False)
    }
    prevalence = float((y_expected == 1).mean())

    lines = [
        "NCI-ALMANAC cross-dataset validation: embedding-independent MFMNet",
        "=================================================================",
        f"External records: {len(ext)}",
        f"Positive / negative: {int((y_expected == 1).sum())} / {int((y_expected == 0).sum())}",
        f"Positive prevalence: {prevalence:.4f}",
        f"Canonical drug-pair bootstrap clusters: {n_clusters}",
        "Model: pre-defined no_drug_embedding MFMNet configuration",
        "Ensemble: five O'Neil validation-selected checkpoints, fixed weights 0.2 each",
        f"Classification threshold: {args.threshold:.3f}",
        "Auxiliary 768-d record-dependent drug embeddings: NOT USED",
        "No NCI-ALMANAC tuning, checkpoint selection, threshold selection, or weight selection.",
        "",
        f"AUROC: {ensemble_metrics['AUROC']:.4f} "
        f"(95% pair-cluster bootstrap CI {ci_map['AUROC'][0]:.4f}-{ci_map['AUROC'][1]:.4f})",
        f"PR-AUC: {ensemble_metrics['PR_AUC']:.4f} "
        f"(95% pair-cluster bootstrap CI {ci_map['PR_AUC'][0]:.4f}-{ci_map['PR_AUC'][1]:.4f})",
        f"ACC: {ensemble_metrics['ACC']:.4f}",
        f"BACC: {ensemble_metrics['BACC']:.4f}",
        f"Precision: {ensemble_metrics['Precision']:.4f}",
        f"TPR: {ensemble_metrics['TPR']:.4f}",
        f"F1: {ensemble_metrics['F1']:.4f}",
        f"Cohen's kappa: {ensemble_metrics['KAPPA']:.4f}",
        "",
        "Interpretation:",
        "- Cross-dataset validation under shared drug/cell coverage.",
        "- Exact O'Neil drug-pair-cell triplets were removed upstream.",
        "- This is not an unseen-drug or unseen-cell-line test.",
        "- NCI-ALMANAC ComboScore and O'Neil Loewe are different synergy metrics.",
    ]
    (outdir / "external_validation_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n" + "\n".join(lines))
    print("\nOutputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
