#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2: reconstruct the published DeepDDS AstraZeneca 668-sample subset
using the official DeepDDS supplementary drug/cell-line lists.

Inputs
------
1) Stage-1 AstraZeneca rows extracted from DrugComb v1.5.
2) DeepDDS Supplementary Table S4 (57 AstraZeneca drugs).
3) DeepDDS Supplementary Table S5 (24 AstraZeneca cell lines).
4) Current O'Neil all_ddi.csv, used only to audit exact overlap.

This script DOES NOT run MFMNet, DOES NOT inspect predictions, and DOES NOT
force the output size to 668. If the official lists alone reproduce 668
unique unordered drug-pair-cell-line combinations, a candidate 668 CSV is
written. Otherwise, diagnostic files are written and the script stops before
model inference.
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
    p.add_argument(
        "--stage1-dir",
        default="./result/astra_668_recovery_stage1",
        help="Directory produced by recover_astrazeneca_668_stage1.py",
    )
    p.add_argument(
        "--supp-s4",
        default="./data/external/supplementary_table_s4_bbab390.xlsx",
        help="Official DeepDDS Supplementary Table S4 (AstraZeneca drugs)",
    )
    p.add_argument(
        "--supp-s5",
        default="./data/external/supplementary_table_s5_bbab390.xlsx",
        help="Official DeepDDS Supplementary Table S5 (AstraZeneca cell lines)",
    )
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output", default="./result/astra_668_recovery_stage2")
    return p.parse_args()


def read_all_excel_values(path: Path):
    book = pd.ExcelFile(path)
    rows = []
    for sheet in book.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                v = df.iat[r, c]
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if not s:
                    continue
                rows.append({
                    "sheet": sheet,
                    "row_1based": r + 1,
                    "col_1based": c + 1,
                    "raw_value": s,
                    "normalized": norm_token(s),
                })
    return pd.DataFrame(rows)


def map_supp_values(values: pd.DataFrame, external_names, entity_name: str):
    ext_norm = {}
    for x in sorted(set(map(str, external_names))):
        ext_norm.setdefault(norm_token(x), []).append(x)

    mapped = []
    for r in values.itertuples(index=False):
        hits = ext_norm.get(r.normalized, [])
        if hits:
            for h in hits:
                mapped.append({
                    "entity_type": entity_name,
                    "sheet": r.sheet,
                    "row_1based": r.row_1based,
                    "col_1based": r.col_1based,
                    "supp_raw_value": r.raw_value,
                    "supp_normalized": r.normalized,
                    "external_name": h,
                    "match_type": "normalized_exact",
                })

    m = pd.DataFrame(mapped)
    if not m.empty:
        m = m.drop_duplicates(
            ["supp_normalized", "external_name"]
        ).sort_values(["external_name", "supp_raw_value"]).reset_index(drop=True)
    return m


def write_unmatched(values, mapped_df, outpath):
    matched_norm = set(mapped_df["supp_normalized"]) if not mapped_df.empty else set()
    # Keep only text-like values that could plausibly be names; numeric table content is noise.
    x = values[
        ~values["normalized"].isin(matched_norm)
        & values["raw_value"].astype(str).str.contains(r"[A-Za-z]", regex=True)
    ].copy()
    x.drop_duplicates(["normalized", "raw_value"]).to_csv(outpath, index=False)


def main():
    args = parse_args()
    stage1 = Path(args.stage1_dir)
    s4 = Path(args.supp_s4)
    s5 = Path(args.supp_s5)
    ddi_path = Path(args.data_root) / "raw" / "all_ddi.csv"
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    az_path = stage1 / "astrazeneca_all_summary_rows.csv"
    for p in [az_path, s4, s5, ddi_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    az = pd.read_csv(az_path, low_memory=False)
    req = {"drug_row", "drug_col", "cell_line_name"}
    missing = req.difference(az.columns)
    if missing:
        raise ValueError(f"Stage-1 AstraZeneca file missing: {sorted(missing)}")

    ext_drugs = sorted(set(az["drug_row"].astype(str)) | set(az["drug_col"].astype(str)))
    ext_cells = sorted(set(az["cell_line_name"].astype(str)))

    s4_values = read_all_excel_values(s4)
    s5_values = read_all_excel_values(s5)
    s4_values.to_csv(outdir / "supp_s4_all_values.csv", index=False)
    s5_values.to_csv(outdir / "supp_s5_all_values.csv", index=False)

    drug_map = map_supp_values(s4_values, ext_drugs, "drug")
    cell_map = map_supp_values(s5_values, ext_cells, "cell")
    drug_map.to_csv(outdir / "supp_s4_to_drugcomb_drug_mapping.csv", index=False)
    cell_map.to_csv(outdir / "supp_s5_to_drugcomb_cell_mapping.csv", index=False)

    write_unmatched(s4_values, drug_map, outdir / "supp_s4_unmatched_text_values.csv")
    write_unmatched(s5_values, cell_map, outdir / "supp_s5_unmatched_text_values.csv")

    selected_drugs = set(drug_map["external_name"]) if not drug_map.empty else set()
    selected_cells = set(cell_map["external_name"]) if not cell_map.empty else set()

    # Filter solely by the official supplementary membership lists.
    sub = az[
        az["drug_row"].astype(str).isin(selected_drugs)
        & az["drug_col"].astype(str).isin(selected_drugs)
        & az["cell_line_name"].astype(str).isin(selected_cells)
    ].copy()

    if sub.empty:
        raise RuntimeError(
            "No AstraZeneca rows remained after official-list matching. "
            "Inspect the mapping/unmatched files; do not infer aliases from predictions."
        )

    # Canonical unordered pair-cell key.
    c1, c2 = [], []
    for a, b in sub[["drug_row", "drug_col"]].itertuples(index=False):
        x, y = canon_pair(a, b)
        c1.append(x); c2.append(y)
    sub["canonical_drug1"] = c1
    sub["canonical_drug2"] = c2
    sub["canonical_key"] = (
        sub["canonical_drug1"].astype(str) + "||"
        + sub["canonical_drug2"].astype(str) + "||"
        + sub["cell_line_name"].astype(str)
    )
    sub.to_csv(outdir / "astra_rows_after_official_s4_s5_membership.csv", index=False)

    # Aggregate duplicates without using labels/predictions to choose a replicate.
    agg_spec = {
        "n_summary_rows": ("study_name", "size"),
    }
    if "block_id" in sub.columns:
        agg_spec["n_blocks"] = ("block_id", "nunique")
    for c in ["synergy_loewe", "synergy_zip", "synergy_bliss", "synergy_hsa", "css"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
            agg_spec[f"{c}_mean"] = (c, "mean")
            agg_spec[f"{c}_median"] = (c, "median")

    paircell = (
        sub.groupby(
            ["canonical_drug1", "canonical_drug2", "cell_line_name"],
            as_index=False
        )
        .agg(**agg_spec)
        .sort_values(["canonical_drug1", "canonical_drug2", "cell_line_name"])
        .reset_index(drop=True)
    )

    # O'Neil overlap audit, no filtering here.
    ddi = pd.read_csv(ddi_path, low_memory=False)
    oneil_triplets = set()
    for r in ddi[["drug1Name", "drug2Name", "cell"]].itertuples(index=False):
        a, b = canon_pair(r.drug1Name, r.drug2Name)
        oneil_triplets.add((a, b, str(r.cell)))

    paircell["exact_oneil_triplet_overlap_by_name"] = [
        (r.canonical_drug1, r.canonical_drug2, str(r.cell_line_name)) in oneil_triplets
        for r in paircell[["canonical_drug1", "canonical_drug2", "cell_line_name"]].itertuples(index=False)
    ]

    paircell.to_csv(outdir / "astra_unique_pair_cell_from_s4_s5.csv", index=False)

    # Multiplicity diagnostics.
    mult = paircell["n_summary_rows"].value_counts().sort_index().rename_axis(
        "n_summary_rows_per_pair_cell"
    ).reset_index(name="n_pair_cell_keys")
    mult.to_csv(outdir / "pair_cell_multiplicity_distribution.csv", index=False)

    matched_s4_norm = int(drug_map["supp_normalized"].nunique()) if not drug_map.empty else 0
    matched_s5_norm = int(cell_map["supp_normalized"].nunique()) if not cell_map.empty else 0

    summary = {
        "purpose": "recover DeepDDS AstraZeneca 668 subset from official S4/S5 lists",
        "model_predictions_inspected": False,
        "target": {
            "pair_cell_records": 668,
            "drugs": 57,
            "cell_lines": 24,
        },
        "supp_s4_matched_unique_normalized_names": matched_s4_norm,
        "supp_s4_mapped_external_drugs": int(len(selected_drugs)),
        "supp_s5_matched_unique_normalized_names": matched_s5_norm,
        "supp_s5_mapped_external_cells": int(len(selected_cells)),
        "summary_rows_after_s4_s5_membership": int(len(sub)),
        "unique_unordered_pair_cell_after_s4_s5_membership": int(len(paircell)),
        "unique_drugs_present_in_filtered_rows": int(
            len(set(sub["drug_row"].astype(str)) | set(sub["drug_col"].astype(str)))
        ),
        "unique_cells_present_in_filtered_rows": int(sub["cell_line_name"].nunique()),
        "exact_oneil_triplet_overlaps_by_name": int(
            paircell["exact_oneil_triplet_overlap_by_name"].sum()
        ),
        "reproduces_668_exactly": bool(len(paircell) == 668),
        "important_note": (
            "The script does not select rows to force n=668. If the official S4/S5 "
            "membership lists do not yield exactly 668 unique pair-cell records, "
            "another source-defined DeepDDS filtering step or the original processed "
            "independent-test file is still required."
        ),
    }

    # Label diagnostics are descriptive only.
    if "synergy_loewe_mean" in paircell.columns:
        s = pd.to_numeric(paircell["synergy_loewe_mean"], errors="coerce")
        summary["loewe_mean_diagnostics"] = {
            "finite": int(s.notna().sum()),
            "gt10": int((s > 10).sum()),
            "lt0": int((s < 0).sum()),
            "gray_0_to_10": int(((s >= 0) & (s <= 10)).sum()),
        }

    (outdir / "astra_668_recovery_stage2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if len(paircell) == 668:
        paircell.to_csv(outdir / "astra_668_candidate_exact.csv", index=False)
        print("\nSUCCESS: official S4/S5 membership reproduced exactly 668 pair-cell rows.")
    else:
        print(
            f"\nNOT YET 668: official S4/S5 membership produced {len(paircell)} "
            "unique pair-cell rows. This is an audit result, not a failure."
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nUpload the entire output folder:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
