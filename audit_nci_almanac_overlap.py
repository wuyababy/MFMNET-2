#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NCI-ALMANAC external-validation FEASIBILITY AUDIT for MFMNet.

This script does NOT train a model and does NOT invent a binary ALMANAC label.
It only answers:
  1) Which ALMANAC drugs can be mapped to the local O'Neil drug inventory?
  2) Which ALMANAC cell lines can use the existing 954-d local cell profile?
  3) How many rows are feature-complete?
  4) How many feature-complete rows duplicate an O'Neil drug-pair-cell triplet?
  5) How many rows remain after overlap removal?

Official CellMiner provides the complete "DTP Almanac Combo Score" dataset.
Download that dataset manually, then pass it using --almanac-file.

Accepted external formats:
  - CSV / TSV / TXT
  - XLS / XLSX
  - ZIP containing one CSV/TSV/TXT/XLS/XLSX

The script tries common long-format column names automatically.
If automatic recognition fails, it writes the detected column names and exits
without guessing. You can then pass:
  --drug1-col ...
  --drug2-col ...
  --cell-col ...
  --score-col ...

Drug-name synonym guessing is deliberately NOT performed.
Optional verified aliases may be supplied with:
  --drug-alias-file verified_drug_aliases.csv
where the CSV has:
  external_name,local_name

Cell-line formatting normalization removes common NCI prefixes/punctuation
(e.g. "CO:HCT-116" -> "HCT116"), but does not invent biological aliases.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


DRUG1_ALIASES = [
    "drug1", "drug_1", "drug a", "drug_a", "compound1", "compound_1",
    "compound a", "compound_a", "drug1name", "drug1_name", "compound1_name",
    "drugname1", "agent1", "agent_1"
]
DRUG2_ALIASES = [
    "drug2", "drug_2", "drug b", "drug_b", "compound2", "compound_2",
    "compound b", "compound_b", "drug2name", "drug2_name", "compound2_name",
    "drugname2", "agent2", "agent_2"
]
CELL_ALIASES = [
    "cell", "cell line", "cell_line", "cellline", "cell name", "cell_name",
    "nci60", "nci-60", "cell_line_name"
]
SCORE_ALIASES = [
    "combo score", "combo_score", "comboscore", "synergy", "synergy score",
    "synergy_score", "score", "almanac score", "almanac_score"
]


def norm_header(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def norm_drug_name(x) -> str:
    s = "" if pd.isna(x) else str(x).strip().upper()
    # Conservative normalization only: case/spacing/punctuation.
    return re.sub(r"[^A-Z0-9]+", "", s)


def norm_cell_name(x) -> str:
    s = "" if pd.isna(x) else str(x).strip().upper()
    # Remove NCI tissue prefix such as BR:, CO:, LC:, OV:, etc.
    if ":" in s:
        s = s.split(":", 1)[1]
    # Common display suffix in NCI-60 naming.
    s = s.replace("/ATCC", "")
    return re.sub(r"[^A-Z0-9]+", "", s)


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".tsv", ".txt"}:
        # sep=None asks pandas to sniff delimiter.
        return pd.read_csv(path, sep=None, engine="python", low_memory=False)
    if suffix in {".xls", ".xlsx"}:
        sheets = pd.read_excel(path, sheet_name=None)
        # Prefer the largest sheet; preserve source-sheet label.
        name, df = max(sheets.items(), key=lambda kv: len(kv[1]))
        df = df.copy()
        df["__source_sheet__"] = name
        return df
    if suffix == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            candidates = [
                n for n in z.namelist()
                if Path(n).suffix.lower() in {".csv", ".tsv", ".txt", ".xls", ".xlsx"}
                and not n.endswith("/")
            ]
            if not candidates:
                raise ValueError("ZIP contains no supported tabular file.")
            # Prefer the largest supported member.
            member = max(candidates, key=lambda n: z.getinfo(n).file_size)
            temp = Path(path.parent) / ("__almanac_extracted_" + Path(member).name)
            with z.open(member) as src, open(temp, "wb") as dst:
                dst.write(src.read())
        try:
            return read_table(temp)
        finally:
            try:
                temp.unlink()
            except Exception:
                pass
    raise ValueError(f"Unsupported external file type: {suffix}")


def choose_column(df: pd.DataFrame, explicit: str | None, aliases: list[str]) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested column not found: {explicit}")
        return explicit

    lookup = {norm_header(c): c for c in df.columns}
    for alias in aliases:
        if norm_header(alias) in lookup:
            return lookup[norm_header(alias)]
    return None


def load_verified_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    required = {"external_name", "local_name"}
    if not required.issubset(df.columns):
        raise ValueError("Alias CSV must contain external_name,local_name")
    return {
        norm_drug_name(r.external_name): norm_drug_name(r.local_name)
        for r in df.itertuples(index=False)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--almanac-file", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--output", default="./result/almanac_feasibility_audit")
    ap.add_argument("--drug1-col")
    ap.add_argument("--drug2-col")
    ap.add_argument("--cell-col")
    ap.add_argument("--score-col")
    ap.add_argument("--drug-alias-file")
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    local = pd.read_csv(data_root / "raw" / "all_ddi.csv", low_memory=False)
    emb = pd.read_csv(data_root / "raw" / "all_drug_embedding.csv", low_memory=False)

    if len(local) != len(emb):
        raise ValueError(f"Local row mismatch: all_ddi={len(local)} embedding={len(emb)}")
    if emb.shape[1] != 1536:
        raise ValueError(f"Expected 1536 local embedding columns, got {emb.shape[1]}")
    if local.iloc[:, 9:].shape[1] != 954:
        raise ValueError("Expected 954 local cell-line feature columns from all_ddi[:,9:].")

    ext = read_table(Path(args.almanac_file))

    d1c = choose_column(ext, args.drug1_col, DRUG1_ALIASES)
    d2c = choose_column(ext, args.drug2_col, DRUG2_ALIASES)
    cc = choose_column(ext, args.cell_col, CELL_ALIASES)
    sc = choose_column(ext, args.score_col, SCORE_ALIASES)

    detected = {
        "columns": list(map(str, ext.columns)),
        "drug1_col": d1c,
        "drug2_col": d2c,
        "cell_col": cc,
        "score_col": sc,
    }
    (outdir / "detected_external_columns.json").write_text(
        json.dumps(detected, indent=2), encoding="utf-8"
    )

    if d1c is None or d2c is None or cc is None:
        raise ValueError(
            "Could not safely identify long-format drug1/drug2/cell columns. "
            "See detected_external_columns.json and rerun with explicit --drug1-col, "
            "--drug2-col and --cell-col. No mapping was guessed."
        )

    # Local drug inventory.
    drug_rows = []
    for side, name_col, smiles_col, emb_slice in [
        ("drug1", "drug1Name", "drug1", slice(0, 768)),
        ("drug2", "drug2Name", "drug2", slice(768, 1536)),
    ]:
        for idx in range(len(local)):
            drug_rows.append({
                "side": side,
                "local_name": local.iloc[idx][name_col],
                "local_smiles": local.iloc[idx][smiles_col],
                "norm_local_name": norm_drug_name(local.iloc[idx][name_col]),
                "row_index": idx,
            })
    drug_inv = pd.DataFrame(drug_rows).drop_duplicates(
        ["norm_local_name", "local_name", "local_smiles"]
    )
    drug_inv.to_csv(outdir / "local_drug_inventory.csv", index=False)

    local_drug_map = {}
    for r in drug_inv.itertuples(index=False):
        if r.norm_local_name and r.norm_local_name not in local_drug_map:
            local_drug_map[r.norm_local_name] = str(r.local_name)

    # Local cell inventory.
    local_cells = pd.DataFrame({
        "local_cell": sorted(local["cell"].dropna().astype(str).unique())
    })
    local_cells["norm_local_cell"] = local_cells["local_cell"].map(norm_cell_name)
    local_cells.to_csv(outdir / "local_cell_inventory.csv", index=False)
    local_cell_map = dict(zip(local_cells["norm_local_cell"], local_cells["local_cell"]))

    verified_aliases = load_verified_aliases(
        Path(args.drug_alias_file) if args.drug_alias_file else None
    )

    audit = ext.copy()
    audit["external_drug1"] = audit[d1c].astype(str)
    audit["external_drug2"] = audit[d2c].astype(str)
    audit["external_cell"] = audit[cc].astype(str)
    if sc is not None:
        audit["external_combo_score"] = pd.to_numeric(audit[sc], errors="coerce")

    audit["norm_drug1"] = audit["external_drug1"].map(norm_drug_name)
    audit["norm_drug2"] = audit["external_drug2"].map(norm_drug_name)
    audit["norm_cell"] = audit["external_cell"].map(norm_cell_name)

    # Apply only user-verified drug aliases.
    audit["mapped_norm_drug1"] = audit["norm_drug1"].map(
        lambda x: verified_aliases.get(x, x)
    )
    audit["mapped_norm_drug2"] = audit["norm_drug2"].map(
        lambda x: verified_aliases.get(x, x)
    )

    audit["local_drug1_name"] = audit["mapped_norm_drug1"].map(local_drug_map)
    audit["local_drug2_name"] = audit["mapped_norm_drug2"].map(local_drug_map)
    audit["local_cell_name"] = audit["norm_cell"].map(local_cell_map)

    audit["drug1_matched"] = audit["local_drug1_name"].notna()
    audit["drug2_matched"] = audit["local_drug2_name"].notna()
    audit["cell_matched"] = audit["local_cell_name"].notna()
    audit["both_drugs_matched"] = audit["drug1_matched"] & audit["drug2_matched"]
    audit["feature_complete"] = audit["both_drugs_matched"] & audit["cell_matched"]

    # Canonical local training triplets: unordered drug pair + normalized cell.
    local_triplets = set()
    for r in local.itertuples(index=False):
        a = norm_drug_name(getattr(r, "drug1Name"))
        b = norm_drug_name(getattr(r, "drug2Name"))
        c = norm_cell_name(getattr(r, "cell"))
        p = canonical_pair(a, b)
        local_triplets.add((p[0], p[1], c))

    def is_local_overlap(row):
        if not row["feature_complete"]:
            return False
        p = canonical_pair(row["mapped_norm_drug1"], row["mapped_norm_drug2"])
        return (p[0], p[1], row["norm_cell"]) in local_triplets

    audit["exact_local_triplet_overlap"] = audit.apply(is_local_overlap, axis=1)
    audit["eligible_after_overlap_removal"] = (
        audit["feature_complete"] & ~audit["exact_local_triplet_overlap"]
    )

    audit.to_csv(outdir / "almanac_row_mapping_audit.csv", index=False)

    unmatched_drugs = pd.concat([
        audit.loc[~audit["drug1_matched"], ["external_drug1"]]
             .rename(columns={"external_drug1": "external_drug"}),
        audit.loc[~audit["drug2_matched"], ["external_drug2"]]
             .rename(columns={"external_drug2": "external_drug"}),
    ], ignore_index=True)
    unmatched_drugs = (
        unmatched_drugs["external_drug"]
        .value_counts()
        .rename_axis("external_drug")
        .reset_index(name="row_occurrences")
    )
    unmatched_drugs.to_csv(outdir / "unmatched_external_drugs.csv", index=False)

    unmatched_cells = (
        audit.loc[~audit["cell_matched"], "external_cell"]
        .value_counts()
        .rename_axis("external_cell")
        .reset_index(name="row_occurrences")
    )
    unmatched_cells.to_csv(outdir / "unmatched_external_cells.csv", index=False)

    eligible = audit[audit["eligible_after_overlap_removal"]].copy()
    eligible.to_csv(outdir / "almanac_feature_complete_overlap_removed.csv", index=False)

    summary = {
        "external_rows_total": int(len(audit)),
        "external_unique_drug_names": int(pd.unique(
            pd.concat([audit["external_drug1"], audit["external_drug2"]], ignore_index=True)
        ).size),
        "external_unique_cells": int(audit["external_cell"].nunique(dropna=True)),
        "rows_both_drugs_matched": int(audit["both_drugs_matched"].sum()),
        "rows_cell_matched": int(audit["cell_matched"].sum()),
        "rows_feature_complete": int(audit["feature_complete"].sum()),
        "rows_exact_local_triplet_overlap": int(audit["exact_local_triplet_overlap"].sum()),
        "rows_eligible_after_overlap_removal": int(audit["eligible_after_overlap_removal"].sum()),
        "eligible_unique_local_drugs": int(pd.unique(pd.concat([
            eligible["local_drug1_name"], eligible["local_drug2_name"]
        ], ignore_index=True).dropna()).size),
        "eligible_unique_local_cells": int(eligible["local_cell_name"].nunique(dropna=True)),
        "binary_label_created": False,
        "note": (
            "This is a feasibility/mapping audit only. NCI-ALMANAC Combo Score is not "
            "silently converted using the O'Neil label threshold."
        ),
    }

    if "external_combo_score" in audit.columns:
        score = audit["external_combo_score"].dropna()
        summary["combo_score_n_numeric"] = int(len(score))
        if len(score):
            summary["combo_score_min"] = float(score.min())
            summary["combo_score_median"] = float(score.median())
            summary["combo_score_mean"] = float(score.mean())
            summary["combo_score_max"] = float(score.max())

    (outdir / "almanac_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nSaved to:", outdir.resolve())


if __name__ == "__main__":
    main()
