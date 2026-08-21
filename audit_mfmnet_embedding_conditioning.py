#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose whether the 768-d inputs in all_drug_embedding.csv are stable at the
drug+cell-line level, even though they are not stable per drug or per drug pair.

This script performs NO model inference and uses NO external labels for feature
construction or selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DRUG_DIM = 768
CELL_DIM = 954


def allclose_rows(arr: np.ndarray, atol: float, rtol: float) -> tuple[bool, float]:
    if len(arr) <= 1:
        return True, 0.0
    ref = arr[0]
    maxdiff = float(np.max(np.abs(arr - ref)))
    ok = bool(np.allclose(arr, ref[None, :], atol=atol, rtol=rtol))
    return ok, maxdiff


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--external-csv", required=True)
    p.add_argument("--output", default="./result/external_embedding_conditioning_audit")
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--rtol", type=float, default=1e-5)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = Path(args.data_root) / "raw"
    ddi = pd.read_csv(raw / "all_ddi.csv", low_memory=False)
    emb = pd.read_csv(raw / "all_drug_embedding.csv", low_memory=False)
    ext = pd.read_csv(args.external_csv, low_memory=False)

    req = {"drug1Name", "drug2Name", "cell"}
    miss = req.difference(ddi.columns)
    if miss:
        raise ValueError(f"all_ddi.csv missing columns: {sorted(miss)}")
    req_ext = {"local_drug1", "local_drug2", "local_cell", "external_label"}
    miss = req_ext.difference(ext.columns)
    if miss:
        raise ValueError(f"external CSV missing columns: {sorted(miss)}")

    if len(ddi) != len(emb):
        raise ValueError(f"Row mismatch: all_ddi={len(ddi)}, embedding={len(emb)}")
    if emb.shape[1] != 2 * DRUG_DIM:
        raise ValueError(f"Expected 1536 embedding columns, got {emb.shape[1]}")

    left = emb.iloc[:, :DRUG_DIM].to_numpy(np.float32, copy=True)
    right = emb.iloc[:, DRUG_DIM:].to_numpy(np.float32, copy=True)

    d1 = ddi["drug1Name"].astype(str).to_numpy()
    d2 = ddi["drug2Name"].astype(str).to_numpy()
    cells = ddi["cell"].astype(str).to_numpy()

    # Audit same drug + same cell within each position.
    rows = []
    stable_left = {}
    stable_right = {}

    for position, names, matrix, store in [
        ("left", d1, left, stable_left),
        ("right", d2, right, stable_right),
    ]:
        tmp = pd.DataFrame({"drug": names, "cell": cells, "idx": np.arange(len(ddi))})
        for (drug, cell), g in tmp.groupby(["drug", "cell"], sort=True):
            idx = g["idx"].to_numpy(dtype=int)
            ok, maxdiff = allclose_rows(matrix[idx], args.atol, args.rtol)
            rows.append({
                "position": position,
                "drug": str(drug),
                "cell": str(cell),
                "n_records": int(len(idx)),
                "consistent": ok,
                "max_abs_diff": maxdiff,
            })
            if ok:
                store[(str(drug), str(cell))] = matrix[idx[0]].copy()

    pos_audit = pd.DataFrame(rows)
    pos_audit.to_csv(outdir / "drug_cell_position_audit.csv", index=False)

    # Audit cross-position compatibility for same drug + same cell.
    all_keys = sorted(set(stable_left).union(stable_right))
    cross_rows = []
    canonical = {}
    ambiguous = set()

    for key in all_keys:
        lv = stable_left.get(key)
        rv = stable_right.get(key)

        if lv is not None and rv is not None:
            diff = float(np.max(np.abs(lv - rv)))
            match = bool(np.allclose(lv, rv, atol=args.atol, rtol=args.rtol))
            if match:
                canonical[key] = lv
            else:
                ambiguous.add(key)
        elif lv is not None:
            diff = np.nan
            match = np.nan
            canonical[key] = lv
        else:
            diff = np.nan
            match = np.nan
            canonical[key] = rv

        cross_rows.append({
            "drug": key[0],
            "cell": key[1],
            "stable_left_available": lv is not None,
            "stable_right_available": rv is not None,
            "cross_position_comparable": lv is not None and rv is not None,
            "cross_position_match": match,
            "max_abs_diff_left_vs_right": diff,
            "usable_as_position_invariant_drug_cell_embedding": key in canonical and key not in ambiguous,
        })

    cross = pd.DataFrame(cross_rows)
    cross.to_csv(outdir / "drug_cell_cross_position_audit.csv", index=False)

    # External coverage if a position-invariant drug+cell representation exists.
    cov_rows = []
    for i, r in ext.reset_index(drop=True).iterrows():
        k1 = (str(r["local_drug1"]), str(r["local_cell"]))
        k2 = (str(r["local_drug2"]), str(r["local_cell"]))
        u1 = k1 in canonical and k1 not in ambiguous
        u2 = k2 in canonical and k2 not in ambiguous
        cov_rows.append({
            "external_index": int(i),
            "local_drug1": k1[0],
            "local_drug2": k2[0],
            "local_cell": k1[1],
            "external_label": int(r["external_label"]),
            "drug1_cell_embedding_usable": bool(u1),
            "drug2_cell_embedding_usable": bool(u2),
            "both_drug_cell_embeddings_usable": bool(u1 and u2),
        })

    coverage = pd.DataFrame(cov_rows)
    coverage.to_csv(outdir / "external_drug_cell_embedding_coverage.csv", index=False)

    repeated = pos_audit[pos_audit["n_records"] > 1]
    comp = cross[cross["cross_position_comparable"] == True]
    usable = coverage["both_drug_cell_embeddings_usable"]

    summary = {
        "n_oneil_records": int(len(ddi)),
        "left_drug_cell_groups": int((pos_audit["position"] == "left").sum()),
        "right_drug_cell_groups": int((pos_audit["position"] == "right").sum()),
        "repeated_drug_cell_position_groups": int(len(repeated)),
        "repeated_drug_cell_position_groups_consistent": int(repeated["consistent"].sum()),
        "fraction_repeated_drug_cell_position_groups_consistent": (
            float(repeated["consistent"].mean()) if len(repeated) else None
        ),
        "drug_cell_keys_comparable_across_left_right": int(len(comp)),
        "drug_cell_keys_matching_across_left_right": int(
            (comp["cross_position_match"] == True).sum()
        ),
        "fraction_cross_position_matching": (
            float((comp["cross_position_match"] == True).mean()) if len(comp) else None
        ),
        "drug_cell_keys_usable_position_invariant": int(
            cross["usable_as_position_invariant_drug_cell_embedding"].sum()
        ),
        "external_rows": int(len(ext)),
        "external_positive": int((ext["external_label"].astype(int) == 1).sum()),
        "external_negative": int((ext["external_label"].astype(int) == 0).sum()),
        "external_rows_with_both_drug_cell_embeddings_usable": int(usable.sum()),
        "external_rows_without_both_drug_cell_embeddings_usable": int((~usable).sum()),
        "external_positive_covered": int(coverage.loc[usable, "external_label"].sum()),
        "external_negative_covered": int(
            usable.sum() - coverage.loc[usable, "external_label"].sum()
        ),
        "interpretation": (
            "If drug+cell embeddings are stable across partners and match across left/right "
            "positions, the 768-d inputs may be drug-and-cell-contextual rather than static "
            "per drug. If they are not stable at this level either, the supplied embeddings "
            "are effectively record-specific for external-validation purposes."
        ),
    }

    with open(outdir / "embedding_conditioning_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nOutputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
