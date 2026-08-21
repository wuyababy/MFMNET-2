#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Matrix-aware NCI-ALMANAC feasibility audit for the official CellMiner download.

Input:
    --almanac-zip <official CellMiner ZIP>
    --data-root ./data

The official workbook is a matrix:
    8 metadata columns + 60 NCI-60 cell-line ComboScore columns.

This audit:
1. detects the real header row;
2. maps ALMANAC drugs to the local O'Neil drug inventory;
3. maps NCI-60 cells to local cell-line names;
4. melts the matrix to drug1-drug2-cell rows;
5. removes missing ComboScores;
6. removes exact O'Neil drug-pair-cell triplet overlap;
7. creates a conservative harmonized external label:
       positive: ComboScore > 10
       negative: ComboScore < 0
       gray zone: 0 <= ComboScore <= 10 (excluded)
8. writes mapping/audit files only. It does NOT run MFMNet.

Four drug-name aliases are explicitly documented:
    Fluorouracil -> 5-FU
    7-Ethyl-10-hydroxycamptothecin -> SN-38
    Mitomycin -> MITOMYCINE
    Vorinostat -> ZOLINZA
"""

from __future__ import annotations
import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def norm_drug(x):
    return re.sub(r"[^A-Z0-9]+", "", str(x).strip().upper())


def norm_cell(x):
    s = str(x).strip().upper()
    if ":" in s:
        s = s.split(":", 1)[1]
    s = s.replace("/ATCC", "")
    return re.sub(r"[^A-Z0-9]+", "", s)


def canon_pair(a, b):
    return tuple(sorted([str(a), str(b)]))


def locate_workbook(zip_path: Path, temp_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as z:
        candidates = [
            n for n in z.namelist()
            if Path(n).name.upper() == "DTP_NCI60_ALMANAC_COMBO_SCORE.XLSX"
        ]
        if not candidates:
            candidates = [
                n for n in z.namelist()
                if n.lower().endswith(".xlsx") and "almanac" in n.lower()
            ]
        if not candidates:
            raise FileNotFoundError("Could not locate the ALMANAC ComboScore workbook in the ZIP.")
        member = candidates[0]
        out = temp_dir / Path(member).name
        with z.open(member) as src, open(out, "wb") as dst:
            dst.write(src.read())
        return out


def read_official_matrix(workbook: Path):
    raw = pd.read_excel(workbook, sheet_name="all", header=None)

    header_row = None
    for i in range(min(30, len(raw))):
        vals = raw.iloc[i].astype(str).tolist()
        if vals and vals[0].strip().startswith("NSC #1"):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not detect the official ALMANAC table header row.")

    headers = raw.iloc[header_row].tolist()
    if len(headers) < 9:
        raise ValueError("Unexpected ALMANAC table width.")

    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    cell_cols = [str(x) for x in headers[8:]]

    pair = pd.DataFrame({
        "NSC_1": body.iloc[:, 0],
        "external_drug1": body.iloc[:, 1],
        "FDA_status_1": body.iloc[:, 2],
        "MOA_1": body.iloc[:, 3],
        "NSC_2": body.iloc[:, 4],
        "external_drug2": body.iloc[:, 5],
        "FDA_status_2": body.iloc[:, 6],
        "MOA_2": body.iloc[:, 7],
    })

    for j, cell in enumerate(cell_cols, start=8):
        pair[cell] = pd.to_numeric(body.iloc[:, j], errors="coerce")

    return pair, cell_cols, header_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--almanac-zip", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--output", default="./result/almanac_feasibility_audit")
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    local = pd.read_csv(data_root / "raw" / "all_ddi.csv", low_memory=False)
    emb = pd.read_csv(data_root / "raw" / "all_drug_embedding.csv", low_memory=False)

    if len(local) != len(emb):
        raise ValueError("Local all_ddi/all_drug_embedding row mismatch.")
    if emb.shape[1] != 1536:
        raise ValueError(f"Expected 1536 embedding columns, got {emb.shape[1]}.")
    if local.iloc[:, 9:].shape[1] != 954:
        raise ValueError("Expected 954 cell-line features from all_ddi[:, 9:].")

    with tempfile.TemporaryDirectory() as td:
        workbook = locate_workbook(Path(args.almanac_zip), Path(td))
        pair, cell_cols, header_row = read_official_matrix(workbook)

    local_drugs = sorted(set(local["drug1Name"]).union(set(local["drug2Name"])))
    local_drug_map = {norm_drug(x): str(x) for x in local_drugs}

    # Explicit synonym/brand mappings.
    external_to_local_normalized = {
        norm_drug("Fluorouracil"): norm_drug("5-FU"),
        norm_drug("7-Ethyl-10-hydroxycamptothecin"): norm_drug("SN-38"),
        norm_drug("Mitomycin"): norm_drug("MITOMYCINE"),
        norm_drug("Vorinostat"): norm_drug("ZOLINZA"),
    }

    def map_drug(x):
        n = norm_drug(x)
        mapped_n = external_to_local_normalized.get(n, n)
        return local_drug_map.get(mapped_n)

    pair["local_drug1"] = pair["external_drug1"].map(map_drug)
    pair["local_drug2"] = pair["external_drug2"].map(map_drug)
    pair["both_drugs_matched"] = pair["local_drug1"].notna() & pair["local_drug2"].notna()

    local_cells = {norm_cell(x): str(x) for x in local["cell"].dropna().unique()}
    external_cell_map = {norm_cell(x): x for x in cell_cols}
    overlap_norm_cells = sorted(set(local_cells).intersection(external_cell_map))
    overlap_external_cells = [external_cell_map[x] for x in overlap_norm_cells]

    matched_pair = pair[pair["both_drugs_matched"]].copy()

    long = matched_pair.melt(
        id_vars=[
            "NSC_1", "external_drug1", "NSC_2", "external_drug2",
            "local_drug1", "local_drug2"
        ],
        value_vars=overlap_external_cells,
        var_name="external_cell",
        value_name="combo_score",
    )
    long["local_cell"] = long["external_cell"].map(
        lambda x: local_cells.get(norm_cell(x))
    )
    long = long[long["combo_score"].notna()].copy()

    local_triplets = set()
    for r in local[["drug1Name", "drug2Name", "cell"]].itertuples(index=False):
        p = canon_pair(r.drug1Name, r.drug2Name)
        local_triplets.add((p[0], p[1], str(r.cell)))

    def is_overlap(r):
        p = canon_pair(r.local_drug1, r.local_drug2)
        return (p[0], p[1], str(r.local_cell)) in local_triplets

    long["exact_oneil_triplet_overlap"] = long.apply(is_overlap, axis=1)
    external = long[~long["exact_oneil_triplet_overlap"]].copy()

    external["external_label"] = np.where(
        external["combo_score"] > 10, 1,
        np.where(external["combo_score"] < 0, 0, np.nan)
    )
    external["gray_zone_0_to_10"] = external["external_label"].isna()
    classifiable = external[external["external_label"].notna()].copy()
    classifiable["external_label"] = classifiable["external_label"].astype(int)

    # Mapping table.
    external_names = sorted(set(pair["external_drug1"].dropna().astype(str)).union(
        set(pair["external_drug2"].dropna().astype(str))
    ))
    mapping_rows = []
    for name in external_names:
        mapped = map_drug(name)
        n = norm_drug(name)
        mapping_rows.append({
            "external_drug": name,
            "local_drug": mapped,
            "matched": mapped is not None,
            "mapping_type": (
                "verified_alias" if n in external_to_local_normalized
                else ("normalized_exact" if mapped is not None else "unmatched")
            ),
        })
    pd.DataFrame(mapping_rows).to_csv(outdir / "drug_mapping.csv", index=False)

    pd.DataFrame([
        {"external_cell": external_cell_map[n], "local_cell": local_cells[n]}
        for n in overlap_norm_cells
    ]).to_csv(outdir / "cell_mapping.csv", index=False)

    long.to_csv(outdir / "feature_complete_before_overlap_removal.csv", index=False)
    external.to_csv(outdir / "external_after_overlap_removal.csv", index=False)
    classifiable.to_csv(outdir / "external_classifiable_gt10_lt0.csv", index=False)

    matched_local_drugs = sorted(set(
        pd.concat([matched_pair["local_drug1"], matched_pair["local_drug2"]]).dropna()
    ))

    summary = {
        "official_header_row_0_based": int(header_row),
        "almanac_pair_rows": int(len(pair)),
        "almanac_unique_drugs": int(len(external_names)),
        "almanac_cell_columns": int(len(cell_cols)),
        "local_oneil_records": int(len(local)),
        "local_unique_drugs": int(len(local_drugs)),
        "local_unique_cells": int(local["cell"].nunique()),
        "mapped_local_drugs": int(len(matched_local_drugs)),
        "mapped_local_drug_names": matched_local_drugs,
        "mapped_local_cells": int(len(overlap_norm_cells)),
        "mapped_local_cell_names": [local_cells[x] for x in overlap_norm_cells],
        "matched_almanac_drug_pairs": int(len(matched_pair)),
        "feature_complete_nonmissing_rows_before_overlap_removal": int(len(long)),
        "exact_oneil_triplet_overlaps_removed": int(long["exact_oneil_triplet_overlap"].sum()),
        "external_rows_after_overlap_removal": int(len(external)),
        "external_positive_combo_gt10": int((external["combo_score"] > 10).sum()),
        "external_negative_combo_lt0": int((external["combo_score"] < 0).sum()),
        "external_gray_zone_0_to_10": int(external["gray_zone_0_to_10"].sum()),
        "external_classifiable_gt10_lt0": int(len(classifiable)),
        "external_classifiable_positive": int((classifiable["external_label"] == 1).sum()),
        "external_classifiable_negative": int((classifiable["external_label"] == 0).sum()),
        "note": (
            "This audit uses shared drug/cell entities but independent NCI-ALMANAC response labels. "
            "Exact O'Neil drug-pair-cell triplets are removed. ComboScore and O'Neil Loewe are "
            "different synergy metrics, so this should be described as cross-dataset validation "
            "under a harmonized conservative classification rule, not as the same assay metric."
        ),
    }

    (outdir / "almanac_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
