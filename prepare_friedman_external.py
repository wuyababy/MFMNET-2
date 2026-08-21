#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare the pre-selected FRIEDMAN DrugComb study for external validation.

Selection was fixed BEFORE model inference by the study-level feasibility audit:
FRIEDMAN was the only non-O'Neil/non-ALMANAC/non-AstraZeneca study with
source-resolved study_name, transferable O'Neil input coverage, exact-overlap
removal, and both Loewe classes under the pre-specified label rule.

No model predictions are read or used here.
"""

from __future__ import annotations
import argparse, json, re
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


DRUG_ALIASES = {
    norm_token("Fluorouracil"): norm_token("5-FU"),
    norm_token("5 Fluorouracil"): norm_token("5-FU"),
    norm_token("5FU"): norm_token("5-FU"),
    norm_token("7-Ethyl-10-hydroxycamptothecin"): norm_token("SN-38"),
    norm_token("SN38"): norm_token("SN-38"),
    norm_token("Mitomycin"): norm_token("MITOMYCINE"),
    norm_token("Mitomycin C"): norm_token("MITOMYCINE"),
    norm_token("Vorinostat"): norm_token("ZOLINZA"),
}

CELL_ALIASES = {
    norm_token("HCT-116"): norm_token("HCT116"),
    norm_token("NCI-H23"): norm_token("NCIH23"),
    norm_token("NCI-H460"): norm_token("NCIH460"),
    norm_token("SK-OV-3"): norm_token("SKOV3"),
    norm_token("SW-620"): norm_token("SW620"),
    norm_token("T-47D"): norm_token("T47D"),
    norm_token("UACC-62"): norm_token("UACC62"),
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--drugcomb-summary", default="./data/external/summary_v_1_5.csv")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--study-name", default="FRIEDMAN")
    p.add_argument("--output", default="./result/friedman_external_prepared")
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

    ddi = pd.read_csv(ddi_path, low_memory=False)
    drugs = sorted(set(ddi["drug1Name"].astype(str)) | set(ddi["drug2Name"].astype(str)))
    cells = sorted(set(ddi["cell"].astype(str)))

    drug_map = {norm_token(x): x for x in drugs}
    cell_map = {norm_token(x): x for x in cells}
    for a, t in DRUG_ALIASES.items():
        if t in drug_map:
            drug_map[a] = drug_map[t]
    for a, t in CELL_ALIASES.items():
        if t in cell_map:
            cell_map[a] = cell_map[t]

    oneil_triplets = set()
    for r in ddi[["drug1Name", "drug2Name", "cell"]].itertuples(index=False):
        a, b = canon_pair(r.drug1Name, r.drug2Name)
        oneil_triplets.add((a, b, str(r.cell)))

    usecols = [
        "block_id", "study_name", "drug_row", "drug_col",
        "cell_line_name", "synergy_loewe"
    ]
    retained = []
    scanned = 0
    study_rows = 0

    for chunk in pd.read_csv(
        summary_path, usecols=usecols, chunksize=args.chunksize, low_memory=False
    ):
        scanned += len(chunk)
        sub = chunk[chunk["study_name"].astype(str) == args.study_name].copy()
        if sub.empty:
            continue
        study_rows += len(sub)

        sub["synergy_loewe"] = pd.to_numeric(sub["synergy_loewe"], errors="coerce")
        sub["local_drug1"] = sub["drug_row"].astype(str).map(norm_token).map(drug_map)
        sub["local_drug2"] = sub["drug_col"].astype(str).map(norm_token).map(drug_map)
        sub["local_cell"] = sub["cell_line_name"].astype(str).map(norm_token).map(cell_map)

        sub = sub[
            sub["synergy_loewe"].notna()
            & sub["local_drug1"].notna()
            & sub["local_drug2"].notna()
            & sub["local_cell"].notna()
        ].copy()

        if not sub.empty:
            retained.append(sub)

    if not retained:
        raise RuntimeError("No transferable rows found for the selected study.")

    x = pd.concat(retained, ignore_index=True)

    c1, c2, overlap = [], [], []
    for r in x[["local_drug1", "local_drug2", "local_cell"]].itertuples(index=False):
        a, b = canon_pair(r.local_drug1, r.local_drug2)
        c1.append(a); c2.append(b)
        overlap.append((a, b, str(r.local_cell)) in oneil_triplets)

    x["canonical_drug1"] = c1
    x["canonical_drug2"] = c2
    x["exact_oneil_triplet_overlap"] = overlap

    mapped_before_overlap = len(x)
    x = x[~x["exact_oneil_triplet_overlap"]].copy()
    after_overlap = len(x)

    x["external_label"] = np.where(
        x["synergy_loewe"] > 10, 1,
        np.where(x["synergy_loewe"] < 0, 0, np.nan)
    )
    x["gray_zone_0_to_10"] = x["external_label"].isna()

    # Save all source-resolved transferable rows before gray exclusion.
    x.to_csv(outdir / "friedman_transferable_after_oneil_overlap_removal.csv", index=False)

    cls = x[x["external_label"].notna()].copy()
    cls["external_label"] = cls["external_label"].astype(int)

    key_cols = ["canonical_drug1", "canonical_drug2", "local_cell"]
    duplicate_triplets = int(cls.duplicated(key_cols, keep=False).sum())
    duplicate_groups = int(cls.groupby(key_cols).size().gt(1).sum())

    # The feasibility audit found no duplicate classifiable triplets. Stop rather
    # than inventing a post-hoc aggregation rule if this changes.
    if duplicate_groups:
        raise RuntimeError(
            f"Found {duplicate_groups} duplicate classifiable pair-cell groups "
            f"({duplicate_triplets} rows). Stop and inspect before inference."
        )

    cls = cls.sort_values(key_cols).reset_index(drop=True)
    cls.to_csv(outdir / "friedman_external_classifiable.csv", index=False)

    summary = {
        "study_name": args.study_name,
        "selection_status": "pre-selected before model inference",
        "model_predictions_used_for_selection": False,
        "drugcomb_rows_scanned": int(scanned),
        "study_rows_total": int(study_rows),
        "mapped_finite_rows_before_overlap_removal": int(mapped_before_overlap),
        "exact_oneil_triplet_overlaps_removed": int(mapped_before_overlap - after_overlap),
        "rows_after_overlap_removal": int(after_overlap),
        "positive_gt10": int((x["synergy_loewe"] > 10).sum()),
        "negative_lt0": int((x["synergy_loewe"] < 0).sum()),
        "gray_0_to_10": int(x["gray_zone_0_to_10"].sum()),
        "final_classifiable_rows": int(len(cls)),
        "final_positive": int((cls["external_label"] == 1).sum()),
        "final_negative": int((cls["external_label"] == 0).sum()),
        "positive_prevalence": float((cls["external_label"] == 1).mean()),
        "unique_drugs": int(len(set(cls["local_drug1"]) | set(cls["local_drug2"]))),
        "unique_cells": int(cls["local_cell"].nunique()),
        "duplicate_classifiable_triplet_groups": int(duplicate_groups),
        "label_rule": {
            "positive": "synergy_loewe > 10",
            "negative": "synergy_loewe < 0",
            "gray_excluded": "0 <= synergy_loewe <= 10",
        },
        "note": (
            "FRIEDMAN was selected using source identity, transferable input coverage, "
            "exact O'Neil overlap removal, and class availability before any external inference."
        ),
    }
    (outdir / "friedman_preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPrepared file:")
    print((outdir / "friedman_external_classifiable.csv").resolve())


if __name__ == "__main__":
    main()
