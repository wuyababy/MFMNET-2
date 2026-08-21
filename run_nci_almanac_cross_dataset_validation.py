#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-dataset external validation of strict MFMNet checkpoints on NCI-ALMANAC.

This script is intentionally conservative and leakage-aware.

Primary external-validation protocol
------------------------------------
1. Load the already prepared NCI-ALMANAC external table produced by
   audit_nci_almanac_matrix_v2.py (default expected columns include
   local_drug1, local_drug2, local_cell, external_label).
2. Reconstruct drug and cell-line input features ONLY from the local O'Neil
   feature inventory used by MFMNet:
     - one 768-d embedding per drug,
     - one molecular graph / SMILES per drug,
     - one 954-d transcriptomic vector per cell line.
3. Before inference, audit that these static features are consistent across
   all O'Neil records and across drug1/drug2 positions. If the audit fails,
   inference stops rather than silently averaging inconsistent features.
4. Load the five validation-selected strict MFMNet checkpoints. The external
   data are not used for checkpoint selection, threshold tuning, or ensemble
   weighting.
5. Obtain a positive-class probability from each checkpoint and take their
   fixed equal-weight mean (1/5 each).
6. Use a fixed 0.5 classification threshold.
7. Report AUROC, PR-AUC, ACC, BACC, Precision, TPR, F1, and Cohen's kappa.
   AUROC and PR-AUC receive stratified percentile-bootstrap 95% CIs.

Important interpretation
------------------------
NCI-ALMANAC ComboScore and the O'Neil Loewe synergy score are different
response metrics. This evaluation should therefore be described as
cross-dataset external validation under a harmonized conservative binary
classification rule, not as validation under an identical continuous synergy
metric. The external set shares drug and cell-line entities with O'Neil but
exact O'Neil drug-pair-cell triplets have been removed by the audit script.
It is NOT an unseen-drug or unseen-cell-line evaluation.

Expected local files
--------------------
./data/raw/all_ddi.csv
./data/raw/all_drug_embedding.csv
./MFMNet_reviewer_clean_save_ckpt.py
./result/mfmnet_external_ckpt_final/checkpoints/mfmnet_full_fold1.pt ... fold5.pt
./result/almanac_feasibility_audit/external_classifiable_gt10_lt0.csv

The paths are configurable through command-line arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from dataclasses import dataclass
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


SCRIPT_VERSION = "2026-08-13-nci-almanac-cross-dataset-v1"
DRUG_EMBED_DIM = 768
CELL_DIM = 954
DEFAULT_THRESHOLD = 0.5
DEFAULT_BOOTSTRAPS = 5000
DEFAULT_BOOTSTRAP_SEED = 20260813


def load_python_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
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
    ]
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise AttributeError(f"Model code is missing required objects: {missing}")
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_smiles(raw: str, Chem) -> str:
    mol = Chem.MolFromSmiles(str(raw))
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {raw!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def allclose_to_reference(matrix: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> Tuple[bool, float]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {matrix.shape}")
    if len(matrix) == 0:
        return False, float("nan")
    diff = np.abs(matrix - ref[None, :])
    max_abs = float(np.max(diff))
    ok = bool(np.allclose(matrix, ref[None, :], atol=atol, rtol=rtol))
    return ok, max_abs


def build_feature_inventory(
    ddi: pd.DataFrame,
    emb: pd.DataFrame,
    needed_drugs: List[str],
    needed_cells: List[str],
    Chem,
    atol: float,
    rtol: float,
):
    required_cols = {"drug1Name", "drug2Name", "drug1", "drug2", "cell"}
    missing = required_cols.difference(ddi.columns)
    if missing:
        raise ValueError(f"all_ddi.csv missing required columns: {sorted(missing)}")
    if len(ddi) != len(emb):
        raise ValueError(f"Row mismatch: all_ddi={len(ddi)}, all_drug_embedding={len(emb)}")
    if emb.shape[1] != 2 * DRUG_EMBED_DIM:
        raise ValueError(f"Expected 1536 embedding columns, got {emb.shape[1]}")
    if ddi.shape[1] < 9 + CELL_DIM:
        raise ValueError(f"Expected at least {9 + CELL_DIM} all_ddi columns, got {ddi.shape[1]}")

    left_emb = emb.iloc[:, :DRUG_EMBED_DIM].to_numpy(dtype=np.float32, copy=True)
    right_emb = emb.iloc[:, DRUG_EMBED_DIM:].to_numpy(dtype=np.float32, copy=True)
    cell_features = ddi.iloc[:, 9:9 + CELL_DIM].to_numpy(dtype=np.float32, copy=True)

    if not np.isfinite(left_emb).all() or not np.isfinite(right_emb).all():
        raise ValueError("Non-finite drug embedding values detected")
    if not np.isfinite(cell_features).all():
        raise ValueError("Non-finite cell-line feature values detected")

    drug_embedding: Dict[str, np.ndarray] = {}
    drug_smiles: Dict[str, str] = {}
    drug_audit = []
    failures = []

    for drug in sorted(needed_drugs):
        lidx = np.flatnonzero(ddi["drug1Name"].astype(str).to_numpy() == str(drug))
        ridx = np.flatnonzero(ddi["drug2Name"].astype(str).to_numpy() == str(drug))
        if len(lidx) + len(ridx) == 0:
            failures.append(f"Drug {drug!r} was not found in O'Neil feature inventory")
            continue

        vectors = []
        if len(lidx):
            vectors.append(left_emb[lidx])
        if len(ridx):
            vectors.append(right_emb[ridx])
        all_vec = np.vstack(vectors)
        ref = all_vec[0].copy()
        emb_ok, max_abs_diff = allclose_to_reference(all_vec, ref, atol=atol, rtol=rtol)

        raw_smiles = []
        if len(lidx):
            raw_smiles.extend(ddi.iloc[lidx]["drug1"].astype(str).tolist())
        if len(ridx):
            raw_smiles.extend(ddi.iloc[ridx]["drug2"].astype(str).tolist())
        canonical = []
        for smi in raw_smiles:
            canonical.append(canonical_smiles(smi, Chem))
        unique_canonical = sorted(set(canonical))
        smiles_ok = len(unique_canonical) == 1

        drug_audit.append({
            "drug": str(drug),
            "n_left_occurrences": int(len(lidx)),
            "n_right_occurrences": int(len(ridx)),
            "n_total_occurrences": int(len(all_vec)),
            "embedding_consistent": bool(emb_ok),
            "embedding_max_abs_diff": max_abs_diff,
            "n_unique_canonical_smiles": int(len(unique_canonical)),
            "smiles_consistent": bool(smiles_ok),
            "canonical_smiles": unique_canonical[0] if smiles_ok else " | ".join(unique_canonical),
        })

        if not emb_ok:
            failures.append(
                f"Drug embedding inconsistency for {drug!r}; max abs diff={max_abs_diff:.8g}"
            )
        if not smiles_ok:
            failures.append(
                f"Multiple non-equivalent SMILES found for {drug!r}: {unique_canonical}"
            )

        if emb_ok and smiles_ok:
            drug_embedding[str(drug)] = ref
            # Use one raw SMILES that maps to the audited single canonical structure.
            drug_smiles[str(drug)] = raw_smiles[0]

    cell_inventory: Dict[str, np.ndarray] = {}
    cell_audit = []
    for cell in sorted(needed_cells):
        idx = np.flatnonzero(ddi["cell"].astype(str).to_numpy() == str(cell))
        if len(idx) == 0:
            failures.append(f"Cell line {cell!r} was not found in O'Neil feature inventory")
            continue
        mat = cell_features[idx]
        ref = mat[0].copy()
        ok, max_abs_diff = allclose_to_reference(mat, ref, atol=atol, rtol=rtol)
        cell_audit.append({
            "cell": str(cell),
            "n_occurrences": int(len(idx)),
            "feature_consistent": bool(ok),
            "feature_max_abs_diff": max_abs_diff,
        })
        if not ok:
            failures.append(
                f"Cell-line feature inconsistency for {cell!r}; max abs diff={max_abs_diff:.8g}"
            )
        else:
            cell_inventory[str(cell)] = ref

    return drug_embedding, drug_smiles, cell_inventory, pd.DataFrame(drug_audit), pd.DataFrame(cell_audit), failures


@dataclass
class ExternalRecord:
    index: int
    drug1: str
    drug2: str
    cell: str
    label: int


class ExternalDataset(Dataset):
    def __init__(self, ext: pd.DataFrame, drug_embedding, drug_smiles, cell_inventory, model_module):
        self.ext = ext.reset_index(drop=True).copy()
        self.drug_embedding = drug_embedding
        self.cell_inventory = cell_inventory
        self.model_module = model_module

        needed_drugs = sorted(set(self.ext["local_drug1"].astype(str)).union(
            set(self.ext["local_drug2"].astype(str))
        ))
        print(f"Building molecular graphs for {len(needed_drugs)} external-validation drugs ...")
        self.graph_cache = {}
        for i, drug in enumerate(needed_drugs, start=1):
            self.graph_cache[drug] = model_module.smiles_to_fixed_graph(drug_smiles[drug])
            print(f"  graph {i}/{len(needed_drugs)}: {drug}")

    def __len__(self):
        return len(self.ext)

    def __getitem__(self, idx):
        r = self.ext.iloc[int(idx)]
        d1 = str(r["local_drug1"])
        d2 = str(r["local_drug2"])
        cell = str(r["local_cell"])
        label = int(r["external_label"])
        return self.model_module.PairSample(
            index=int(idx),
            left_graph=self.graph_cache[d1],
            right_graph=self.graph_cache[d2],
            left_embedding=torch.from_numpy(self.drug_embedding[d1]),
            right_embedding=torch.from_numpy(self.drug_embedding[d2]),
            cell=torch.from_numpy(self.cell_inventory[cell]),
            label=label,
        )


def metric_dict(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
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


def bootstrap_primary_metrics(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int):
    """Stratified percentile bootstrap for AUROC and PR-AUC."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("External labels must contain both classes")

    aucs = np.empty(n_boot, dtype=float)
    prs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ip = rng.choice(pos, size=len(pos), replace=True)
        ineg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([ip, ineg])
        rng.shuffle(idx)
        yy = y[idx]
        ss = score[idx]
        aucs[b] = roc_auc_score(yy, ss)
        pc, rc, _ = metrics.precision_recall_curve(yy, ss)
        prs[b] = metrics.auc(rc, pc)

    rows = []
    for name, vals in [("AUROC", aucs), ("PR_AUC", prs)]:
        rows.append({
            "metric": name,
            "bootstrap_replicates": int(n_boot),
            "bootstrap_method": "stratified percentile bootstrap",
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
    if isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
        meta = {k: obj.get(k) for k in ["variant", "fold", "best_epoch", "best_val_AUROC", "seed"]}
    elif isinstance(obj, dict):
        state = obj
        meta = {}
    else:
        raise ValueError(f"Unexpected checkpoint object in {path}: {type(obj).__name__}")
    return state, meta


def predict_one_checkpoint(model, loader, device, model_module):
    model.eval()
    scores = []
    indices = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)[:, 1]
            scores.append(prob.detach().cpu().numpy())
            indices.append(batch.index.detach().cpu().numpy())
            labels.append(batch.label.detach().cpu().numpy())
    model_module.sync(device)
    return np.concatenate(indices), np.concatenate(labels), np.concatenate(scores)


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
            raise ValueError(f"External CSV still contains {int(bad.sum())} exact O'Neil triplet overlaps")
    if "gray_zone_0_to_10" in ext.columns:
        bad = ext["gray_zone_0_to_10"].astype(str).str.lower().isin(["true", "1"])
        if bad.any():
            raise ValueError(f"External CSV still contains {int(bad.sum())} gray-zone rows")

    # Audit canonical pair-cell duplicates. Do not silently drop anything.
    keys = []
    for r in ext[["local_drug1", "local_drug2", "local_cell"]].itertuples(index=False):
        pair = tuple(sorted([str(r.local_drug1), str(r.local_drug2)]))
        keys.append((pair[0], pair[1], str(r.local_cell)))
    dup_n = int(pd.Series(keys).duplicated().sum())
    if dup_n:
        raise ValueError(
            f"External CSV contains {dup_n} duplicate canonical drug-pair-cell triplets. "
            "Resolve these in the audit stage before inference."
        )


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument(
        "--external-csv",
        default="./result/almanac_feasibility_audit/external_classifiable_gt10_lt0.csv",
    )
    ap.add_argument(
        "--checkpoints-dir",
        default="./result/mfmnet_external_ckpt_final/checkpoints",
    )
    ap.add_argument(
        "--model-code",
        default="./MFMNet_reviewer_clean_save_ckpt.py",
        help="Exact model-code file used to generate the saved checkpoints",
    )
    ap.add_argument("--output", default="./result/nci_almanac_external_validation")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--feature-atol", type=float, default=1e-6)
    ap.add_argument("--feature-rtol", type=float, default=1e-6)
    args = ap.parse_args()

    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1")
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    ddi_path = data_root / "raw" / "all_ddi.csv"
    emb_path = data_root / "raw" / "all_drug_embedding.csv"
    ext_path = Path(args.external_csv)
    ckpt_dir = Path(args.checkpoints_dir)
    model_code_path = Path(args.model_code)

    for p in [ddi_path, emb_path, ext_path, ckpt_dir, model_code_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    print("Loading exact MFMNet model code:", model_code_path)
    model_module = load_python_module(model_code_path)
    Chem = model_module.Chem

    print("Loading O'Neil feature inventory ...")
    ddi = pd.read_csv(ddi_path, low_memory=False)
    emb = pd.read_csv(emb_path, low_memory=False)
    print("Loading prepared NCI-ALMANAC external set ...")
    ext = pd.read_csv(ext_path, low_memory=False)
    validate_external_table(ext)

    needed_drugs = sorted(set(ext["local_drug1"].astype(str)).union(set(ext["local_drug2"].astype(str))))
    needed_cells = sorted(set(ext["local_cell"].astype(str)))

    print(f"External records: {len(ext)}")
    print(f"External positives: {(ext['external_label'].astype(int) == 1).sum()}")
    print(f"External negatives: {(ext['external_label'].astype(int) == 0).sum()}")
    print(f"Mapped drugs used: {len(needed_drugs)}")
    print(f"Mapped cell lines used: {len(needed_cells)}")

    print("\nAuditing static feature consistency before external inference ...")
    drug_embedding, drug_smiles, cell_inventory, drug_audit, cell_audit, failures = build_feature_inventory(
        ddi=ddi,
        emb=emb,
        needed_drugs=needed_drugs,
        needed_cells=needed_cells,
        Chem=Chem,
        atol=args.feature_atol,
        rtol=args.feature_rtol,
    )
    drug_audit.to_csv(outdir / "external_drug_feature_audit.csv", index=False)
    cell_audit.to_csv(outdir / "external_cell_feature_audit.csv", index=False)

    audit_summary = {
        "script_version": SCRIPT_VERSION,
        "n_external_records": int(len(ext)),
        "n_external_positive": int((ext["external_label"].astype(int) == 1).sum()),
        "n_external_negative": int((ext["external_label"].astype(int) == 0).sum()),
        "positive_fraction": float((ext["external_label"].astype(int) == 1).mean()),
        "n_needed_drugs": int(len(needed_drugs)),
        "n_needed_cells": int(len(needed_cells)),
        "feature_atol": float(args.feature_atol),
        "feature_rtol": float(args.feature_rtol),
        "drug_feature_audit_passed": bool(len(drug_audit) == len(needed_drugs) and drug_audit["embedding_consistent"].all() and drug_audit["smiles_consistent"].all()),
        "cell_feature_audit_passed": bool(len(cell_audit) == len(needed_cells) and cell_audit["feature_consistent"].all()),
        "failures": failures,
        "interpretation_note": (
            "Cross-dataset validation with shared drug/cell entities and independent NCI-ALMANAC response labels; "
            "exact O'Neil drug-pair-cell triplets are excluded. ComboScore and O'Neil Loewe are different metrics."
        ),
    }
    (outdir / "external_feature_audit_summary.json").write_text(
        json.dumps(audit_summary, indent=2), encoding="utf-8"
    )

    if failures:
        print("\nFEATURE AUDIT FAILED. Inference has been stopped.")
        for x in failures:
            print(" -", x)
        print("See:", outdir / "external_feature_audit_summary.json")
        sys.exit(2)

    print("Feature audit PASSED.")

    checkpoint_paths = [ckpt_dir / f"mfmnet_full_fold{i}.pt" for i in range(1, 6)]
    missing_ckpt = [str(p) for p in checkpoint_paths if not p.exists()]
    if missing_ckpt:
        raise FileNotFoundError(f"Missing required checkpoints: {missing_ckpt}")

    checkpoint_rows = []
    checkpoint_states = []
    for expected_fold, p in enumerate(checkpoint_paths, start=1):
        state, meta = load_checkpoint(p)
        if meta.get("variant") not in (None, "full"):
            raise ValueError(f"Checkpoint {p} variant is {meta.get('variant')!r}, expected 'full'")
        if meta.get("fold") is not None and int(meta["fold"]) != expected_fold:
            raise ValueError(f"Checkpoint fold metadata mismatch in {p}: {meta.get('fold')} vs {expected_fold}")
        checkpoint_states.append((expected_fold, p, state, meta))
        checkpoint_rows.append({
            "fold": expected_fold,
            "file": str(p),
            "sha256": sha256_file(p),
            "size_mb": float(p.stat().st_size / (1024 ** 2)),
            "variant": meta.get("variant"),
            "best_epoch": meta.get("best_epoch"),
            "best_val_AUROC": meta.get("best_val_AUROC"),
            "seed": meta.get("seed"),
        })
    pd.DataFrame(checkpoint_rows).to_csv(outdir / "external_checkpoint_manifest.csv", index=False)

    dataset = ExternalDataset(ext, drug_embedding, drug_smiles, cell_inventory, model_module)
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
    per_checkpoint_metrics = []

    for fold, p, state, meta in checkpoint_states:
        print(f"Predicting external set with fold {fold} checkpoint ...")
        model = model_module.MFMNetReviewerModel(variant="full", dropout_rate=0.3).to(device)
        incompat = model.load_state_dict(state, strict=True)
        # With strict=True, missing/unexpected should be empty; keep the check explicit.
        if getattr(incompat, "missing_keys", []) or getattr(incompat, "unexpected_keys", []):
            raise RuntimeError(
                f"State-dict incompatibility for fold {fold}: missing={incompat.missing_keys}, "
                f"unexpected={incompat.unexpected_keys}"
            )
        idx, y, score = predict_one_checkpoint(model, loader, device, model_module)
        order = np.argsort(idx)
        idx = idx[order]
        y = y[order]
        score = score[order]
        if not np.array_equal(idx, np.arange(len(ext))):
            raise RuntimeError(f"External sample order mismatch for fold {fold}")
        if not np.array_equal(y, y_expected):
            raise RuntimeError(f"External label mismatch for fold {fold}")
        fold_scores.append(score)
        row = {"model": f"MFMNet fold {fold}", "fold": fold, **metric_dict(y, score, args.threshold)}
        per_checkpoint_metrics.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass

    score_matrix = np.vstack(fold_scores)
    ensemble_score = score_matrix.mean(axis=0)  # fixed 1/5 weights
    ensemble_pred = (ensemble_score >= args.threshold).astype(int)
    ensemble_metrics = metric_dict(y_expected, ensemble_score, args.threshold)

    per_checkpoint_df = pd.DataFrame(per_checkpoint_metrics)
    per_checkpoint_df.to_csv(outdir / "external_per_checkpoint_metrics.csv", index=False)

    final_metric_df = pd.DataFrame([{"model": "MFMNet 5-checkpoint equal-weight ensemble", **ensemble_metrics}])
    final_metric_df.to_csv(outdir / "external_ensemble_metrics.csv", index=False)

    print(f"\nBootstrapping AUROC and PR-AUC ({args.bootstrap_replicates} stratified replicates) ...")
    ci_df = bootstrap_primary_metrics(
        y_expected,
        ensemble_score,
        n_boot=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    ci_df.to_csv(outdir / "external_primary_metric_bootstrap_ci.csv", index=False)

    prediction_df = ext.copy()
    for fold in range(1, 6):
        prediction_df[f"prob_fold{fold}"] = score_matrix[fold - 1]
    prediction_df["ensemble_probability"] = ensemble_score
    prediction_df["predicted_label_threshold_0_5"] = ensemble_pred
    prediction_df.to_csv(outdir / "external_predictions_all.csv", index=False)

    fpr, tpr, roc_thr = metrics.roc_curve(y_expected, ensemble_score)
    pd.DataFrame({"FPR": fpr, "TPR": tpr, "threshold": roc_thr}).to_csv(
        outdir / "external_roc_curve.csv", index=False
    )
    precision, recall, pr_thr = metrics.precision_recall_curve(y_expected, ensemble_score)
    pr_df = pd.DataFrame({"recall": recall, "precision": precision})
    # PR thresholds are one element shorter than precision/recall.
    pr_df["threshold"] = np.nan
    if len(pr_thr):
        pr_df.loc[: len(pr_thr) - 1, "threshold"] = pr_thr
    pr_df.to_csv(outdir / "external_pr_curve.csv", index=False)

    cm = metrics.confusion_matrix(y_expected, ensemble_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(outdir / "external_confusion_matrix.csv")

    run_info = {
        "script_version": SCRIPT_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "data_root": str(data_root),
        "external_csv": str(ext_path),
        "external_csv_sha256": sha256_file(ext_path),
        "model_code": str(model_code_path),
        "model_code_sha256": sha256_file(model_code_path),
        "checkpoints_dir": str(ckpt_dir),
        "n_checkpoints": 5,
        "ensemble_weights": [0.2] * 5,
        "classification_threshold": float(args.threshold),
        "external_tuning": False,
        "checkpoint_selection_on_external_data": False,
        "ensemble_weight_tuning_on_external_data": False,
        "threshold_tuning_on_external_data": False,
        "drug_order_symmetrization": False,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_method": "stratified percentile bootstrap",
    }
    (outdir / "external_run_config.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    # Human-readable summary.
    ci_map = {r.metric: (r.CI95_lower, r.CI95_upper) for r in ci_df.itertuples(index=False)}
    summary_lines = [
        "NCI-ALMANAC cross-dataset external validation",
        "==============================================",
        f"External records: {len(ext)}",
        f"Positive / negative: {int((y_expected == 1).sum())} / {int((y_expected == 0).sum())}",
        f"Positive prevalence: {float((y_expected == 1).mean()):.4f}",
        "Model: 5 strict MFMNet checkpoints, fixed equal weights (0.2 each)",
        f"Classification threshold: {args.threshold:.3f}",
        "No external tuning / checkpoint selection / ensemble-weight selection.",
        "",
        f"AUROC: {ensemble_metrics['AUROC']:.4f} (95% bootstrap CI {ci_map['AUROC'][0]:.4f}-{ci_map['AUROC'][1]:.4f})",
        f"PR-AUC: {ensemble_metrics['PR_AUC']:.4f} (95% bootstrap CI {ci_map['PR_AUC'][0]:.4f}-{ci_map['PR_AUC'][1]:.4f})",
        f"ACC: {ensemble_metrics['ACC']:.4f}",
        f"BACC: {ensemble_metrics['BACC']:.4f}",
        f"Precision: {ensemble_metrics['Precision']:.4f}",
        f"TPR: {ensemble_metrics['TPR']:.4f}",
        f"F1: {ensemble_metrics['F1']:.4f}",
        f"Cohen's kappa: {ensemble_metrics['KAPPA']:.4f}",
        "",
        "Interpretation: shared drug/cell entities, independent NCI-ALMANAC response labels,",
        "exact O'Neil drug-pair-cell triplets removed; not an unseen-drug/unseen-cell evaluation.",
        "NCI-ALMANAC ComboScore and O'Neil Loewe are different synergy metrics.",
    ]
    (outdir / "external_validation_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(summary_lines))
    print("\nPrimary outputs:")
    for name in [
        "external_feature_audit_summary.json",
        "external_drug_feature_audit.csv",
        "external_cell_feature_audit.csv",
        "external_checkpoint_manifest.csv",
        "external_per_checkpoint_metrics.csv",
        "external_ensemble_metrics.csv",
        "external_primary_metric_bootstrap_ci.csv",
        "external_predictions_all.csv",
        "external_roc_curve.csv",
        "external_pr_curve.csv",
        "external_confusion_matrix.csv",
        "external_run_config.json",
        "external_validation_summary.txt",
    ]:
        print(" -", outdir / name)


if __name__ == "__main__":
    main()
