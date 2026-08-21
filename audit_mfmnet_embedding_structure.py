#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit whether the 768-d drug embeddings in all_drug_embedding.csv are
static per drug, static per ordered drug pair, or record-specific, and
quantify whether the prepared NCI-ALMANAC external rows are covered by
O'Neil pair-level embeddings.

This script performs NO model inference and NO tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

DRUG_DIM = 768
CELL_DIM = 954


def canon_pair(a: str, b: str) -> Tuple[str, str]:
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def max_abs_diff_to_reference(arr: np.ndarray) -> float:
    if len(arr) <= 1:
        return 0.0
    return float(np.max(np.abs(arr - arr[0])))


def allclose_rows(arr: np.ndarray, atol: float, rtol: float) -> bool:
    if len(arr) <= 1:
        return True
    return bool(np.allclose(arr, arr[0][None, :], atol=atol, rtol=rtol))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--external-csv", required=True)
    p.add_argument("--output", default="./result/external_embedding_structure_audit")
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--rtol", type=float, default=1e-5)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = Path(args.data_root) / "raw"
    ddi_path = raw / "all_ddi.csv"
    emb_path = raw / "all_drug_embedding.csv"
    ext_path = Path(args.external_csv)

    ddi = pd.read_csv(ddi_path, low_memory=False)
    emb = pd.read_csv(emb_path, low_memory=False)
    ext = pd.read_csv(ext_path, low_memory=False)

    required_ddi = {"drug1Name", "drug2Name", "cell", "drug1", "drug2"}
    missing = required_ddi.difference(ddi.columns)
    if missing:
        raise ValueError(f"all_ddi.csv missing columns: {sorted(missing)}")
    required_ext = {"local_drug1", "local_drug2", "local_cell", "external_label"}
    missing = required_ext.difference(ext.columns)
    if missing:
        raise ValueError(f"External CSV missing columns: {sorted(missing)}")

    if len(ddi) != len(emb):
        raise ValueError(f"Row mismatch: all_ddi={len(ddi)}, embeddings={len(emb)}")
    if emb.shape[1] != 2 * DRUG_DIM:
        raise ValueError(f"Expected 1536 embedding columns, got {emb.shape[1]}")
    if ddi.iloc[:, 9:9 + CELL_DIM].shape[1] != CELL_DIM:
        raise ValueError("Expected 954 cell-line features from all_ddi[:, 9:963].")

    left = emb.iloc[:, :DRUG_DIM].to_numpy(np.float32, copy=True)
    right = emb.iloc[:, DRUG_DIM:].to_numpy(np.float32, copy=True)
    cells = ddi.iloc[:, 9:9 + CELL_DIM].to_numpy(np.float32, copy=True)

    drug_rows = []
    all_drugs = sorted(set(ddi["drug1Name"].astype(str)).union(ddi["drug2Name"].astype(str)))
    d1_names = ddi["drug1Name"].astype(str).to_numpy()
    d2_names = ddi["drug2Name"].astype(str).to_numpy()
    for drug in all_drugs:
        blocks = []
        idx_l = np.flatnonzero(d1_names == drug)
        idx_r = np.flatnonzero(d2_names == drug)
        if len(idx_l):
            blocks.append(left[idx_l])
        if len(idx_r):
            blocks.append(right[idx_r])
        arr = np.concatenate(blocks, axis=0) if blocks else np.empty((0, DRUG_DIM), np.float32)
        drug_rows.append({
            "drug": drug,
            "n_occurrences": int(len(arr)),
            "n_left": int(len(idx_l)),
            "n_right": int(len(idx_r)),
            "embedding_consistent_per_drug": allclose_rows(arr, args.atol, args.rtol),
            "max_abs_diff": max_abs_diff_to_reference(arr),
        })
    drug_audit = pd.DataFrame(drug_rows)
    drug_audit.to_csv(outdir / "drug_level_embedding_audit.csv", index=False)

    ordered_rows = []
    grouped = ddi.groupby(["drug1Name", "drug2Name"], sort=True).indices
    for (d1, d2), idx in grouped.items():
        idx = np.asarray(idx, dtype=int)
        larr = left[idx]
        rarr = right[idx]
        ordered_rows.append({
            "drug1": str(d1),
            "drug2": str(d2),
            "n_records": int(len(idx)),
            "n_cells": int(ddi.iloc[idx]["cell"].astype(str).nunique()),
            "left_embedding_consistent": allclose_rows(larr, args.atol, args.rtol),
            "right_embedding_consistent": allclose_rows(rarr, args.atol, args.rtol),
            "left_max_abs_diff": max_abs_diff_to_reference(larr),
            "right_max_abs_diff": max_abs_diff_to_reference(rarr),
            "smiles1_consistent": int(ddi.iloc[idx]["drug1"].astype(str).nunique()) == 1,
            "smiles2_consistent": int(ddi.iloc[idx]["drug2"].astype(str).nunique()) == 1,
        })
    ordered_audit = pd.DataFrame(ordered_rows)
    ordered_audit["both_embeddings_consistent"] = (
        ordered_audit["left_embedding_consistent"] &
        ordered_audit["right_embedding_consistent"]
    )
    ordered_audit.to_csv(outdir / "ordered_pair_embedding_audit.csv", index=False)

    rep = {}
    for (d1, d2), idx in grouped.items():
        idx = np.asarray(idx, dtype=int)
        larr = left[idx]
        rarr = right[idx]
        if allclose_rows(larr, args.atol, args.rtol) and allclose_rows(rarr, args.atol, args.rtol):
            rep[(str(d1), str(d2))] = (larr[0], rarr[0])

    reverse_rows = []
    seen = set()
    for (d1, d2), (lv, rv) in rep.items():
        if d1 == d2 or (d2, d1) not in rep:
            continue
        key = canon_pair(d1, d2)
        if key in seen:
            continue
        seen.add(key)
        rev_l, rev_r = rep[(d2, d1)]
        reverse_rows.append({
            "drug_a": key[0],
            "drug_b": key[1],
            "forward_order": f"{d1}|{d2}",
            "reverse_order": f"{d2}|{d1}",
            "forward_left_matches_reverse_right": bool(np.allclose(lv, rev_r, atol=args.atol, rtol=args.rtol)),
            "forward_right_matches_reverse_left": bool(np.allclose(rv, rev_l, atol=args.atol, rtol=args.rtol)),
            "max_abs_diff_left_vs_reverse_right": float(np.max(np.abs(lv - rev_r))),
            "max_abs_diff_right_vs_reverse_left": float(np.max(np.abs(rv - rev_l))),
        })
    reverse_audit = pd.DataFrame(reverse_rows)
    if not reverse_audit.empty:
        reverse_audit["swap_consistent"] = (
            reverse_audit["forward_left_matches_reverse_right"] &
            reverse_audit["forward_right_matches_reverse_left"]
        )
    reverse_audit.to_csv(outdir / "reverse_order_embedding_audit.csv", index=False)

    cell_rows = []
    for cell, idx in ddi.groupby("cell", sort=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        arr = cells[idx]
        cell_rows.append({
            "cell": str(cell),
            "n_records": int(len(idx)),
            "feature_consistent": allclose_rows(arr, args.atol, args.rtol),
            "max_abs_diff": max_abs_diff_to_reference(arr),
        })
    cell_audit = pd.DataFrame(cell_rows)
    cell_audit.to_csv(outdir / "cell_feature_audit.csv", index=False)

    oneil_canon_pairs = {}
    for (d1, d2), idx in grouped.items():
        key = canon_pair(str(d1), str(d2))
        oneil_canon_pairs.setdefault(key, []).append((str(d1), str(d2)))

    coverage_rows = []
    ddi_cell = ddi["cell"].astype(str)
    for i, r in ext.reset_index(drop=True).iterrows():
        d1, d2, cell = str(r["local_drug1"]), str(r["local_drug2"]), str(r["local_cell"])
        key = canon_pair(d1, d2)
        orientations = oneil_canon_pairs.get(key, [])
        stable_orientations = [o for o in orientations if o in rep]
        exact_mask = (
            (((ddi["drug1Name"].astype(str) == d1) & (ddi["drug2Name"].astype(str) == d2)) |
             ((ddi["drug1Name"].astype(str) == d2) & (ddi["drug2Name"].astype(str) == d1))) &
            (ddi_cell == cell)
        )
        coverage_rows.append({
            "external_index": int(i),
            "local_drug1": d1,
            "local_drug2": d2,
            "local_cell": cell,
            "external_label": int(r["external_label"]),
            "canonical_pair": f"{key[0]}|{key[1]}",
            "pair_seen_in_oneil_any_cell": bool(orientations),
            "n_oneil_orientations": int(len(orientations)),
            "n_stable_ordered_orientations": int(len(stable_orientations)),
            "pair_has_stable_ordered_embedding": bool(stable_orientations),
            "exact_triplet_in_oneil": bool(exact_mask.any()),
            "available_orientations": ";".join([f"{a}|{b}" for a, b in orientations]),
            "stable_orientations": ";".join([f"{a}|{b}" for a, b in stable_orientations]),
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(outdir / "external_pair_coverage.csv", index=False)

    pair_cov = (
        coverage.groupby("canonical_pair", as_index=False)
        .agg(
            n_external_rows=("external_index", "size"),
            pair_seen_in_oneil_any_cell=("pair_seen_in_oneil_any_cell", "all"),
            pair_has_stable_ordered_embedding=("pair_has_stable_ordered_embedding", "all"),
            positive_rows=("external_label", "sum"),
        )
    )
    pair_cov["negative_rows"] = pair_cov["n_external_rows"] - pair_cov["positive_rows"]
    pair_cov.to_csv(outdir / "external_pair_coverage_by_pair.csv", index=False)

    repeated = ordered_audit[ordered_audit["n_records"] > 1]
    if reverse_audit.empty:
        n_reverse = 0
        n_reverse_swap = 0
    else:
        n_reverse = int(len(reverse_audit))
        n_reverse_swap = int(reverse_audit["swap_consistent"].sum())

    external_stable = coverage["pair_has_stable_ordered_embedding"]
    summary = {
        "n_oneil_records": int(len(ddi)),
        "n_unique_drugs": int(len(all_drugs)),
        "drug_level_static_embeddings": int(drug_audit["embedding_consistent_per_drug"].sum()),
        "drug_level_nonstatic_embeddings": int((~drug_audit["embedding_consistent_per_drug"]).sum()),
        "n_ordered_pairs": int(len(ordered_audit)),
        "n_ordered_pairs_with_stable_embeddings": int(ordered_audit["both_embeddings_consistent"].sum()),
        "n_repeated_ordered_pairs": int(len(repeated)),
        "n_repeated_ordered_pairs_with_stable_embeddings": int(repeated["both_embeddings_consistent"].sum()),
        "fraction_repeated_ordered_pairs_stable": float(repeated["both_embeddings_consistent"].mean()) if len(repeated) else None,
        "n_canonical_pairs_seen_in_both_orientations_with_stable_embeddings": n_reverse,
        "n_reverse_pairs_swap_consistent": n_reverse_swap,
        "n_cell_lines": int(len(cell_audit)),
        "n_cell_lines_with_static_954d_features": int(cell_audit["feature_consistent"].sum()),
        "external_rows": int(len(ext)),
        "external_positive": int((ext["external_label"].astype(int) == 1).sum()),
        "external_negative": int((ext["external_label"].astype(int) == 0).sum()),
        "external_rows_pair_seen_in_oneil_any_cell": int(coverage["pair_seen_in_oneil_any_cell"].sum()),
        "external_rows_with_stable_ordered_pair_embedding": int(external_stable.sum()),
        "external_rows_without_stable_ordered_pair_embedding": int((~external_stable).sum()),
        "external_positive_with_stable_pair_embedding": int(coverage.loc[external_stable, "external_label"].sum()),
        "external_negative_with_stable_pair_embedding": int(external_stable.sum() - coverage.loc[external_stable, "external_label"].sum()),
        "exact_triplet_overlap_found_in_external_file": int(coverage["exact_triplet_in_oneil"].sum()),
        "interpretation": (
            "If repeated ordered drug pairs are stable across cell lines, but individual drugs are not stable across partners, "
            "the 768-d inputs behave as ordered-pair-specific features rather than one static embedding per drug. External "
            "inference may then be possible only for external drug pairs covered by a stable O'Neil ordered-pair representation, "
            "without reusing exact drug-pair-cell triplets."
        ),
    }

    with open(outdir / "embedding_structure_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nOutputs written to:", outdir.resolve())


if __name__ == "__main__":
    main()
