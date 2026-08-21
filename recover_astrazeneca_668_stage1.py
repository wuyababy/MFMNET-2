#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1: recover the AstraZeneca 668-sample independent-test data path.

This script does NOT run MFMNet and does NOT try to force 93.3%.
It reconstructs the source-resolved ASTRAZENECA study structure from
DrugComb v1.5 and compares it with the published DeepDDS independent-test
target (668 unique drug-pair-cell-line combinations, 57 drugs, 24 cell lines).

Outputs are intended for auditing the exact filtering/mapping rule before
external inference is restored.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def norm_token(x: str) -> str:
    s = str(x).strip().upper()
    s = s.replace("–", "-").replace("—", "-").replace("_", "-")
    return re.sub(r"[^A-Z0-9]+", "", s)


def canon_pair(a: str, b: str):
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--drugcomb-summary", default="./data/external/summary_v_1_5.csv")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--study-name", default="ASTRAZENECA")
    p.add_argument("--output", default="./result/astra_668_recovery_stage1")
    p.add_argument("--chunksize", type=int, default=200000)
    return p.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.drugcomb_summary)
    ddi_path = Path(args.data_root) / "raw" / "all_ddi.csv"
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not ddi_path.exists():
        raise FileNotFoundError(ddi_path)

    header = pd.read_csv(summary_path, nrows=0).columns.tolist()
    required = {"study_name", "drug_row", "drug_col", "cell_line_name"}
    missing = required.difference(header)
    if missing:
        raise ValueError(f"DrugComb summary missing required columns: {sorted(missing)}")

    candidate_score_cols = [
        c for c in ["block_id", "synergy_loewe", "synergy_zip",
                    "synergy_bliss", "synergy_hsa", "css"]
        if c in header
    ]
    usecols = ["study_name", "drug_row", "drug_col", "cell_line_name"] + candidate_score_cols

    chunks = []
    scanned = 0
    for chunk in pd.read_csv(
        summary_path, usecols=usecols, chunksize=args.chunksize, low_memory=False
    ):
        scanned += len(chunk)
        sub = chunk[chunk["study_name"].astype(str).str.upper() == args.study_name.upper()].copy()
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        raise RuntimeError(f"No rows found for study_name={args.study_name!r}")

    az = pd.concat(chunks, ignore_index=True)
    for c in ["synergy_loewe", "synergy_zip", "synergy_bliss", "synergy_hsa", "css"]:
        if c in az.columns:
            az[c] = pd.to_numeric(az[c], errors="coerce")

    az["drug_row_norm"] = az["drug_row"].map(norm_token)
    az["drug_col_norm"] = az["drug_col"].map(norm_token)
    az["cell_norm"] = az["cell_line_name"].map(norm_token)

    pair1, pair2 = [], []
    for a, b in az[["drug_row", "drug_col"]].itertuples(index=False):
        x, y = canon_pair(a, b)
        pair1.append(x)
        pair2.append(y)
    az["canonical_drug1"] = pair1
    az["canonical_drug2"] = pair2
    az["canonical_pair_cell"] = (
        az["canonical_drug1"].astype(str) + "||"
        + az["canonical_drug2"].astype(str) + "||"
        + az["cell_line_name"].astype(str)
    )

    az.to_csv(outdir / "astrazeneca_all_summary_rows.csv", index=False)

    # Basic source structure.
    drug_names = sorted(set(az["drug_row"].astype(str)) | set(az["drug_col"].astype(str)))
    cells = sorted(set(az["cell_line_name"].astype(str)))

    pd.DataFrame({"drug_name": drug_names}).to_csv(
        outdir / "astrazeneca_unique_drugs.csv", index=False
    )
    pd.DataFrame({"cell_line_name": cells}).to_csv(
        outdir / "astrazeneca_unique_cells.csv", index=False
    )

    drug_freq = pd.concat([
        az["drug_row"].astype(str).rename("drug_name"),
        az["drug_col"].astype(str).rename("drug_name")
    ]).value_counts().rename_axis("drug_name").reset_index(name="row_occurrences")
    drug_freq.to_csv(outdir / "astrazeneca_drug_frequency.csv", index=False)

    cell_freq = az["cell_line_name"].astype(str).value_counts().rename_axis(
        "cell_line_name"
    ).reset_index(name="row_occurrences")
    cell_freq.to_csv(outdir / "astrazeneca_cell_frequency.csv", index=False)

    # Deduplicated pair-cell structures without inventing a label rule.
    paircell = (
        az.groupby(
            ["canonical_drug1", "canonical_drug2", "cell_line_name"],
            as_index=False,
            dropna=False
        )
        .agg(
            n_summary_rows=("study_name", "size"),
            n_blocks=("block_id", "nunique") if "block_id" in az.columns
            else ("study_name", "size"),
            **{
                f"{c}_mean": (c, "mean")
                for c in ["synergy_loewe", "synergy_zip", "synergy_bliss", "synergy_hsa", "css"]
                if c in az.columns
            },
            **{
                f"{c}_median": (c, "median")
                for c in ["synergy_loewe", "synergy_zip", "synergy_bliss", "synergy_hsa", "css"]
                if c in az.columns
            },
        )
    )
    paircell.to_csv(outdir / "astrazeneca_unique_pair_cell_aggregated.csv", index=False)

    # Compare names with the current O'Neil inventory. This is an audit only.
    ddi = pd.read_csv(ddi_path, low_memory=False)
    local_drugs = sorted(
        set(ddi["drug1Name"].dropna().astype(str))
        | set(ddi["drug2Name"].dropna().astype(str))
    )
    local_cells = sorted(set(ddi["cell"].dropna().astype(str)))

    local_drug_norm = {norm_token(x): x for x in local_drugs}
    local_cell_norm = {norm_token(x): x for x in local_cells}

    drug_map_rows = []
    for d in drug_names:
        n = norm_token(d)
        drug_map_rows.append({
            "external_drug": d,
            "normalized": n,
            "normalized_exact_local_drug": local_drug_norm.get(n, ""),
            "normalized_exact_match": n in local_drug_norm,
        })
    pd.DataFrame(drug_map_rows).to_csv(
        outdir / "astrazeneca_to_oneil_drug_exact_mapping.csv", index=False
    )

    cell_map_rows = []
    for c in cells:
        n = norm_token(c)
        cell_map_rows.append({
            "external_cell": c,
            "normalized": n,
            "normalized_exact_local_cell": local_cell_norm.get(n, ""),
            "normalized_exact_match": n in local_cell_norm,
        })
    pd.DataFrame(cell_map_rows).to_csv(
        outdir / "astrazeneca_to_oneil_cell_exact_mapping.csv", index=False
    )

    # Label-distribution diagnostics only; no filtering is performed here.
    label_diag = {}
    if "synergy_loewe" in az.columns:
        for level, df in [("summary_rows", az), ("unique_pair_cell_mean", paircell)]:
            col = "synergy_loewe" if level == "summary_rows" else "synergy_loewe_mean"
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                label_diag[level] = {
                    "finite": int(s.notna().sum()),
                    "gt10": int((s > 10).sum()),
                    "lt0": int((s < 0).sum()),
                    "gray_0_to_10": int(((s >= 0) & (s <= 10)).sum()),
                    "min": None if s.dropna().empty else float(s.min()),
                    "max": None if s.dropna().empty else float(s.max()),
                    "mean": None if s.dropna().empty else float(s.mean()),
                    "median": None if s.dropna().empty else float(s.median()),
                }

    summary = {
        "purpose": "recover AstraZeneca 668-sample data path before model inference",
        "model_predictions_inspected": False,
        "target_from_deepdds_paper": {
            "unique_drug_pair_cell_combinations": 668,
            "drugs": 57,
            "cell_lines": 24,
        },
        "drugcomb_file": str(summary_path),
        "drugcomb_rows_scanned": int(scanned),
        "study_name": args.study_name,
        "study_summary_rows": int(len(az)),
        "study_unique_block_ids": (
            int(az["block_id"].nunique()) if "block_id" in az.columns else None
        ),
        "study_unique_unordered_pair_cell": int(
            az["canonical_pair_cell"].nunique()
        ),
        "study_unique_drugs": int(len(drug_names)),
        "study_unique_cell_lines": int(len(cells)),
        "normalized_exact_overlap_with_current_oneil_drugs": int(
            sum(norm_token(d) in local_drug_norm for d in drug_names)
        ),
        "normalized_exact_overlap_with_current_oneil_cells": int(
            sum(norm_token(c) in local_cell_norm for c in cells)
        ),
        "label_diagnostics": label_diag,
        "important_note": (
            "No subset was forced to size 668. The next stage must recover the "
            "published/legacy filtering rule from source identifiers, supplementary "
            "drug/cell lists, or an original processed file before any MFMNet evaluation."
        ),
    }

    (outdir / "astra_668_recovery_stage1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nUpload the entire output folder:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
