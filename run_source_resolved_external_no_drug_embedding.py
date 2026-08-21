#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic source-resolved external validation for the pre-defined
MFMNet no_drug_embedding configuration.

This script is intended for a frozen external CSV prepared before inference.
It never tunes checkpoints, threshold, ensemble weights, aliases, or labels
using external predictions.

Required external CSV columns:
    local_drug1, local_drug2, local_cell, external_label
Optional columns checked if present:
    exact_oneil_triplet_overlap, gray_zone_0_to_10

Primary model result:
    equal-weight average of the five validation-selected O'Neil checkpoints.
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
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader


DRUG_EMBED_DIM = 768
CELL_DIM = 954


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--external-csv", required=True)
    p.add_argument(
        "--checkpoints-dir",
        default="./result/mfmnet_no_drug_embedding_external_ckpt/checkpoints",
    )
    p.add_argument(
        "--model-code",
        default="./MFMNet_external_no_drug_embedding_save_ckpt.py",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--bootstrap-replicates", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=20260813)
    p.add_argument("--feature-atol", type=float, default=1e-6)
    p.add_argument("--feature-rtol", type=float, default=1e-6)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("mfmnet_external_model_code", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model code from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in [
        "MFMNetReviewerModel", "PairSample", "collate_pair_samples",
        "smiles_to_fixed_graph", "choose_device", "sync", "Chem"
    ]:
        if not hasattr(module, name):
            raise AttributeError(f"Model code missing {name}")
    return module


def canonical_smiles(raw: str, Chem) -> str:
    mol = Chem.MolFromSmiles(str(raw))
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {raw!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def validate_external(ext: pd.DataFrame):
    req = {"local_drug1", "local_drug2", "local_cell", "external_label"}
    missing = req.difference(ext.columns)
    if missing:
        raise ValueError(f"External CSV missing: {sorted(missing)}")
    if ext[list(req)].isna().any().any():
        raise ValueError("External CSV contains missing required values")

    labels = sorted(pd.unique(ext["external_label"].astype(int)).tolist())
    if labels != [0, 1]:
        raise ValueError(f"Both binary classes are required; got labels={labels}")

    if "exact_oneil_triplet_overlap" in ext.columns:
        bad = ext["exact_oneil_triplet_overlap"].astype(str).str.lower().isin(["true", "1"])
        if bad.any():
            raise ValueError(f"{int(bad.sum())} exact O'Neil overlaps remain")

    if "gray_zone_0_to_10" in ext.columns:
        bad = ext["gray_zone_0_to_10"].astype(str).str.lower().isin(["true", "1"])
        if bad.any():
            raise ValueError(f"{int(bad.sum())} gray-zone rows remain")

    keys = []
    for r in ext[["local_drug1", "local_drug2", "local_cell"]].itertuples(index=False):
        a, b = sorted([str(r.local_drug1), str(r.local_drug2)])
        keys.append((a, b, str(r.local_cell)))
    dup = int(pd.Series(keys).duplicated().sum())
    if dup:
        raise ValueError(f"{dup} duplicate canonical pair-cell rows remain")


def build_inventory(ddi, needed_drugs, needed_cells, Chem, atol, rtol):
    cell_matrix = ddi.iloc[:, 9:9 + CELL_DIM].to_numpy(dtype=np.float32, copy=True)
    d1 = ddi["drug1Name"].astype(str).to_numpy()
    d2 = ddi["drug2Name"].astype(str).to_numpy()

    drug_smiles = {}
    drug_audit = []
    failures = []

    for drug in sorted(needed_drugs):
        li = np.flatnonzero(d1 == drug)
        ri = np.flatnonzero(d2 == drug)
        raws = []
        if len(li):
            raws.extend(ddi.iloc[li]["drug1"].astype(str).tolist())
        if len(ri):
            raws.extend(ddi.iloc[ri]["drug2"].astype(str).tolist())
        if not raws:
            failures.append(f"Drug not found: {drug}")
            continue
        canon = sorted(set(canonical_smiles(x, Chem) for x in raws))
        ok = len(canon) == 1
        drug_audit.append({
            "drug": drug, "n_occurrences": len(raws),
            "n_unique_canonical_smiles": len(canon),
            "smiles_consistent": ok,
            "canonical_smiles": canon[0] if ok else " | ".join(canon),
        })
        if ok:
            drug_smiles[drug] = raws[0]
        else:
            failures.append(f"Inconsistent SMILES for {drug}: {canon}")

    cells_arr = ddi["cell"].astype(str).to_numpy()
    cell_inventory = {}
    cell_audit = []
    for cell in sorted(needed_cells):
        idx = np.flatnonzero(cells_arr == cell)
        if not len(idx):
            failures.append(f"Cell not found: {cell}")
            continue
        mat = cell_matrix[idx]
        ref = mat[0]
        ok = bool(np.allclose(mat, ref[None, :], atol=atol, rtol=rtol))
        max_abs = float(np.max(np.abs(mat - ref[None, :])))
        cell_audit.append({
            "cell": cell, "n_occurrences": int(len(idx)),
            "feature_consistent": ok, "feature_max_abs_diff": max_abs,
        })
        if ok:
            cell_inventory[cell] = ref.copy()
        else:
            failures.append(f"Inconsistent cell features for {cell}; max_abs={max_abs}")

    return (
        drug_smiles, cell_inventory,
        pd.DataFrame(drug_audit), pd.DataFrame(cell_audit), failures
    )


class ExternalDataset(Dataset):
    def __init__(self, ext, drug_smiles, cell_inventory, m):
        self.ext = ext.reset_index(drop=True).copy()
        self.cell_inventory = cell_inventory
        self.m = m
        self.zero = np.zeros(DRUG_EMBED_DIM, dtype=np.float32)
        drugs = sorted(
            set(self.ext["local_drug1"].astype(str))
            | set(self.ext["local_drug2"].astype(str))
        )
        self.graphs = {d: m.smiles_to_fixed_graph(drug_smiles[d]) for d in drugs}

    def __len__(self):
        return len(self.ext)

    def __getitem__(self, idx):
        r = self.ext.iloc[int(idx)]
        d1, d2, cell = str(r.local_drug1), str(r.local_drug2), str(r.local_cell)
        return self.m.PairSample(
            index=int(idx),
            left_graph=self.graphs[d1],
            right_graph=self.graphs[d2],
            left_embedding=torch.from_numpy(self.zero.copy()),
            right_embedding=torch.from_numpy(self.zero.copy()),
            cell=torch.from_numpy(self.cell_inventory[cell]),
            label=int(r.external_label),
        )


def metric_dict(y, score, threshold):
    pred = (score >= threshold).astype(int)
    pc, rc, _ = metrics.precision_recall_curve(y, score)
    return {
        "AUROC": float(roc_auc_score(y, score)),
        "PR_AUC": float(metrics.auc(rc, pc)),
        "ACC": float(accuracy_score(y, pred)),
        "BACC": float(balanced_accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "TPR": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "KAPPA": float(cohen_kappa_score(y, pred)),
    }


def cluster_ids(ext):
    ids = []
    for r in ext[["local_drug1", "local_drug2"]].itertuples(index=False):
        a, b = sorted([str(r.local_drug1), str(r.local_drug2)])
        ids.append(f"{a}|{b}")
    return np.asarray(ids, dtype=object)


def cluster_bootstrap(y, score, ids, n_boot, seed):
    rng = np.random.default_rng(seed)
    clusters = np.unique(ids)
    c2i = {c: np.flatnonzero(ids == c) for c in clusters}
    aucs, prs = [], []
    attempts = 0
    while len(aucs) < n_boot and attempts < max(1000, 30 * n_boot):
        attempts += 1
        sampled = rng.choice(clusters, len(clusters), replace=True)
        idx = np.concatenate([c2i[c] for c in sampled])
        yy, ss = y[idx], score[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(float(roc_auc_score(yy, ss)))
        pc, rc, _ = metrics.precision_recall_curve(yy, ss)
        prs.append(float(metrics.auc(rc, pc)))
    if len(aucs) < n_boot:
        raise RuntimeError(f"Only {len(aucs)} valid bootstrap replicates")
    rows = []
    for name, vals in [("AUROC", np.asarray(aucs)), ("PR_AUC", np.asarray(prs))]:
        rows.append({
            "metric": name,
            "bootstrap_replicates": n_boot,
            "bootstrap_unit": "canonical drug pair",
            "CI95_lower": float(np.percentile(vals, 2.5)),
            "CI95_upper": float(np.percentile(vals, 97.5)),
            "bootstrap_mean": float(np.mean(vals)),
            "bootstrap_SD": float(np.std(vals, ddof=1)),
        })
    return pd.DataFrame(rows)


def load_ckpt(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError(f"Invalid checkpoint: {path}")
    return obj


def predict(model, loader, device, m):
    model.eval()
    ix, yy, ss = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prob = torch.softmax(model(batch), dim=1)[:, 1]
            ix.append(batch.index.detach().cpu().numpy())
            yy.append(batch.label.detach().cpu().numpy())
            ss.append(prob.detach().cpu().numpy())
    m.sync(device)
    return np.concatenate(ix), np.concatenate(yy), np.concatenate(ss)


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    ddi_path = Path(args.data_root) / "raw" / "all_ddi.csv"
    ext_path = Path(args.external_csv)
    ckpt_dir = Path(args.checkpoints_dir)
    model_code = Path(args.model_code)

    for p in [ddi_path, ext_path, ckpt_dir, model_code]:
        if not p.exists():
            raise FileNotFoundError(p)

    m = load_module(model_code)
    ext = pd.read_csv(ext_path, low_memory=False)
    validate_external(ext)

    y = ext["external_label"].astype(int).to_numpy()
    needed_drugs = sorted(set(ext["local_drug1"].astype(str)) | set(ext["local_drug2"].astype(str)))
    needed_cells = sorted(set(ext["local_cell"].astype(str)))

    ddi = pd.read_csv(ddi_path, low_memory=False)
    drug_smiles, cell_inv, da, ca, failures = build_inventory(
        ddi, needed_drugs, needed_cells, m.Chem, args.feature_atol, args.feature_rtol
    )
    da.to_csv(outdir / "drug_smiles_audit.csv", index=False)
    ca.to_csv(outdir / "cell_feature_audit.csv", index=False)
    if failures:
        (outdir / "feature_audit_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError("Transferable-feature audit failed: " + "; ".join(failures))

    ckpts = []
    manifest = []
    for fold in range(1, 6):
        p = ckpt_dir / f"mfmnet_no_drug_embedding_fold{fold}.pt"
        if not p.exists():
            raise FileNotFoundError(p)
        obj = load_ckpt(p)
        if obj.get("variant") != "no_drug_embedding":
            raise ValueError(f"{p.name}: wrong variant {obj.get('variant')}")
        if obj.get("fold") is not None and int(obj["fold"]) != fold:
            raise ValueError(f"{p.name}: fold metadata mismatch")
        ckpts.append((fold, obj["state_dict"]))
        manifest.append({
            "fold": fold, "file": str(p), "sha256": sha256_file(p),
            "best_epoch": obj.get("best_epoch"),
            "best_val_AUROC": obj.get("best_val_AUROC"),
            "seed": obj.get("seed"),
        })
    pd.DataFrame(manifest).to_csv(outdir / "checkpoint_manifest.csv", index=False)

    ds = ExternalDataset(ext, drug_smiles, cell_inv, m)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=m.collate_pair_samples, drop_last=False
    )
    device = m.choose_device(args.device)
    print("Device:", device)

    fold_scores = []
    fold_metrics = []
    for fold, state in ckpts:
        print(f"Predicting {args.dataset_name}: fold {fold}/5")
        model = m.MFMNetReviewerModel(
            variant="no_drug_embedding", dropout_rate=0.3
        ).to(device)
        model.load_state_dict(state, strict=True)
        idx, yy, score = predict(model, loader, device, m)
        order = np.argsort(idx)
        idx, yy, score = idx[order], yy[order], score[order]
        if not np.array_equal(idx, np.arange(len(ext))):
            raise RuntimeError("Prediction order mismatch")
        if not np.array_equal(yy, y):
            raise RuntimeError("Label order mismatch")
        fold_scores.append(score)
        fold_metrics.append({
            "fold": fold,
            **metric_dict(y, score, args.threshold)
        })
        del model

    score_mat = np.vstack(fold_scores)
    ens = score_mat.mean(axis=0)
    ens_metrics = metric_dict(y, ens, args.threshold)

    pd.DataFrame(fold_metrics).to_csv(outdir / "per_checkpoint_metrics.csv", index=False)
    pd.DataFrame([{
        "dataset": args.dataset_name,
        "model": "Embedding-independent MFMNet; equal-weight five-checkpoint ensemble",
        "n": len(ext),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "positive_prevalence": float((y == 1).mean()),
        "threshold": args.threshold,
        **ens_metrics,
    }]).to_csv(outdir / "ensemble_metrics.csv", index=False)

    ids = cluster_ids(ext)
    ci = cluster_bootstrap(
        y, ens, ids, args.bootstrap_replicates, args.bootstrap_seed
    )
    ci.to_csv(outdir / "primary_metric_cluster_bootstrap_ci.csv", index=False)

    pred = ext.copy()
    pred["canonical_pair_cluster"] = ids
    for i in range(5):
        pred[f"prob_fold{i+1}"] = score_mat[i]
    pred["ensemble_probability"] = ens
    pred["predicted_label"] = (ens >= args.threshold).astype(int)
    pred.to_csv(outdir / "external_predictions.csv", index=False)

    config = {
        "dataset_name": args.dataset_name,
        "external_csv": str(ext_path),
        "external_csv_sha256": sha256_file(ext_path),
        "model_code_sha256": sha256_file(model_code),
        "variant": "no_drug_embedding",
        "auxiliary_768d_embeddings_used": False,
        "n_checkpoints": 5,
        "ensemble_weights": [0.2] * 5,
        "threshold": args.threshold,
        "external_tuning": False,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "canonical drug pair",
        "n_bootstrap_clusters": int(pd.Series(ids).nunique()),
        "torch": torch.__version__,
        "device": str(device),
        "platform": platform.platform(),
    }
    (outdir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    ci_map = {r.metric: (r.CI95_lower, r.CI95_upper) for r in ci.itertuples(index=False)}
    txt = [
        f"{args.dataset_name} external validation",
        "=" * (len(args.dataset_name) + 20),
        f"n={len(ext)}; positive={(y==1).sum()}; negative={(y==0).sum()}",
        f"positive prevalence={(y==1).mean():.4f}",
        "model=pre-defined no_drug_embedding MFMNet",
        "ensemble=five frozen O'Neil validation-selected checkpoints, 0.2 each",
        "threshold=0.5; no external tuning",
        "",
        f"AUROC={ens_metrics['AUROC']:.4f} "
        f"(95% pair-cluster bootstrap CI {ci_map['AUROC'][0]:.4f}-{ci_map['AUROC'][1]:.4f})",
        f"PR-AUC={ens_metrics['PR_AUC']:.4f} "
        f"(95% pair-cluster bootstrap CI {ci_map['PR_AUC'][0]:.4f}-{ci_map['PR_AUC'][1]:.4f})",
        f"ACC={ens_metrics['ACC']:.4f}",
        f"BACC={ens_metrics['BACC']:.4f}",
        f"Precision={ens_metrics['Precision']:.4f}",
        f"TPR={ens_metrics['TPR']:.4f}",
        f"F1={ens_metrics['F1']:.4f}",
        f"Kappa={ens_metrics['KAPPA']:.4f}",
    ]
    (outdir / "external_validation_summary.txt").write_text(
        "\n".join(txt) + "\n", encoding="utf-8"
    )
    print("\n" + "\n".join(txt))
    print("\nOutput:", outdir.resolve())


if __name__ == "__main__":
    main()
