#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Study-level feasibility audit for an additional external drug-synergy dataset
from DrugComb v1.5 (or another DrugComb summary file exposing study_name).

Purpose
-------
This script DOES NOT run MFMNet and DOES NOT inspect model predictions.
It selects candidate external studies only on pre-inference criteria:
  - source is explicitly identifiable by study_name;
  - drug/cell entities overlap the O'Neil/DeepDDS feature inventory;
  - exact O'Neil drug-pair-cell triplets are removed;
  - Loewe labels use the fixed rule already used in the manuscript:
        synergy_loewe > 10 -> positive
        synergy_loewe < 0  -> negative
        0 <= synergy_loewe <= 10 -> gray / excluded
  - candidate studies must contain both classes after overlap removal.

This avoids choosing an external dataset based on favorable model performance.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def norm_token(x: str) -> str:
    s = str(x).strip().upper()
    s = s.replace("–", "-").replace("—", "-").replace("_", "-")
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s


def canon_pair(a: str, b: str):
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


# Conservative aliases already verified/used in the NCI work.
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
    p.add_argument(
        "--drugcomb-summary",
        default="./data/external/summary_v_1_5.csv",
        help="DrugComb summary CSV or CSV.GZ that contains study_name",
    )
    p.add_argument(
        "--data-root",
        default="./data",
        help="MFMNet data root containing raw/all_ddi.csv",
    )
    p.add_argument(
        "--output",
        default="./result/drugcomb_external_study_feasibility",
    )
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument(
        "--exclude-study-regex",
        default=r"(?i)(ONEIL|O['’]?NEIL|ALMANAC|NCI|ASTRAZENECA|DREAM)",
        help="Studies excluded from candidate ranking because they are training/current external sets",
    )
    return p.parse_args()


def build_local_inventory(ddi: pd.DataFrame):
    req = {"drug1Name", "drug2Name", "cell"}
    missing = req.difference(ddi.columns)
    if missing:
        raise ValueError(f"all_ddi.csv missing required columns: {sorted(missing)}")

    drugs = sorted(
        set(ddi["drug1Name"].dropna().astype(str))
        | set(ddi["drug2Name"].dropna().astype(str))
    )
    cells = sorted(set(ddi["cell"].dropna().astype(str)))

    drug_map = {norm_token(x): x for x in drugs}
    cell_map = {norm_token(x): x for x in cells}

    for alias_norm, target_norm in DRUG_ALIASES.items():
        if target_norm in drug_map:
            drug_map[alias_norm] = drug_map[target_norm]

    for alias_norm, target_norm in CELL_ALIASES.items():
        if target_norm in cell_map:
            cell_map[alias_norm] = cell_map[target_norm]

    local_triplets = set()
    for r in ddi[["drug1Name", "drug2Name", "cell"]].itertuples(index=False):
        a, b = canon_pair(r.drug1Name, r.drug2Name)
        local_triplets.add((a, b, str(r.cell)))

    return drugs, cells, drug_map, cell_map, local_triplets


def first_header(path: Path):
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


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

    print("Inspecting DrugComb header ...")
    columns = first_header(summary_path)
    print(columns)

    required = {
        "study_name", "drug_row", "drug_col", "cell_line_name", "synergy_loewe"
    }
    missing = required.difference(columns)
    if missing:
        raise ValueError(
            "This DrugComb file cannot support a source-resolved study audit. "
            f"Missing columns: {sorted(missing)}"
        )

    print("Loading O'Neil/DeepDDS inventory ...")
    ddi = pd.read_csv(ddi_path, low_memory=False)
    drugs, cells, drug_map, cell_map, local_triplets = build_local_inventory(ddi)
    print(f"Local drugs: {len(drugs)}")
    print(f"Local cells: {len(cells)}")
    print(f"Local canonical drug-pair-cell triplets: {len(local_triplets)}")

    # Per-study aggregated counters.
    stats = defaultdict(lambda: {
        "rows_total": 0,
        "rows_finite_loewe": 0,
        "rows_both_drugs_mapped": 0,
        "rows_cell_mapped": 0,
        "rows_all_entities_mapped": 0,
        "rows_exact_oneil_overlap": 0,
        "rows_after_overlap": 0,
        "positive_gt10": 0,
        "negative_lt0": 0,
        "gray_0_to_10": 0,
        "unique_triplets_after_overlap": set(),
        "unique_classifiable_triplets": set(),
        "unique_drugs": set(),
        "unique_cells": set(),
    })

    usecols = [
        "study_name", "drug_row", "drug_col", "cell_line_name", "synergy_loewe"
    ]
    total = 0
    chunk_i = 0

    print("Scanning DrugComb in chunks ...")
    for chunk in pd.read_csv(
        summary_path,
        usecols=usecols,
        chunksize=args.chunksize,
        low_memory=False,
    ):
        chunk_i += 1
        total += len(chunk)

        chunk["study_name"] = chunk["study_name"].fillna("").astype(str)
        chunk["synergy_loewe"] = pd.to_numeric(chunk["synergy_loewe"], errors="coerce")

        d1_norm = chunk["drug_row"].astype(str).map(norm_token)
        d2_norm = chunk["drug_col"].astype(str).map(norm_token)
        c_norm = chunk["cell_line_name"].astype(str).map(norm_token)

        chunk["local_drug1"] = d1_norm.map(drug_map)
        chunk["local_drug2"] = d2_norm.map(drug_map)
        chunk["local_cell"] = c_norm.map(cell_map)

        for study, sub in chunk.groupby("study_name", dropna=False):
            study = str(study)
            s = stats[study]
            s["rows_total"] += len(sub)

            finite = np.isfinite(sub["synergy_loewe"].to_numpy(dtype=float, na_value=np.nan))
            s["rows_finite_loewe"] += int(finite.sum())

            dmap = sub["local_drug1"].notna() & sub["local_drug2"].notna()
            cmap = sub["local_cell"].notna()
            s["rows_both_drugs_mapped"] += int(dmap.sum())
            s["rows_cell_mapped"] += int(cmap.sum())

            m = finite & dmap.to_numpy() & cmap.to_numpy()
            s["rows_all_entities_mapped"] += int(m.sum())

            if not m.any():
                continue

            x = sub.loc[m, [
                "local_drug1", "local_drug2", "local_cell", "synergy_loewe"
            ]].copy()

            for r in x.itertuples(index=False):
                a, b = canon_pair(r.local_drug1, r.local_drug2)
                triplet = (a, b, str(r.local_cell))
                if triplet in local_triplets:
                    s["rows_exact_oneil_overlap"] += 1
                    continue

                s["rows_after_overlap"] += 1
                s["unique_triplets_after_overlap"].add(triplet)
                s["unique_drugs"].update([a, b])
                s["unique_cells"].add(str(r.local_cell))

                score = float(r.synergy_loewe)
                if score > 10:
                    s["positive_gt10"] += 1
                    s["unique_classifiable_triplets"].add(triplet)
                elif score < 0:
                    s["negative_lt0"] += 1
                    s["unique_classifiable_triplets"].add(triplet)
                else:
                    s["gray_0_to_10"] += 1

        if chunk_i % 5 == 0:
            print(f"  chunks={chunk_i}, rows_scanned={total:,}, studies_seen={len(stats)}")

    rows = []
    exclude_re = re.compile(args.exclude_study_regex)
    for study, s in stats.items():
        classifiable = s["positive_gt10"] + s["negative_lt0"]
        min_class = min(s["positive_gt10"], s["negative_lt0"])
        pos_frac = (
            s["positive_gt10"] / classifiable if classifiable > 0 else np.nan
        )
        rows.append({
            "study_name": study,
            "excluded_by_name_rule": bool(exclude_re.search(study)),
            "rows_total": s["rows_total"],
            "rows_finite_loewe": s["rows_finite_loewe"],
            "rows_both_drugs_mapped": s["rows_both_drugs_mapped"],
            "rows_cell_mapped": s["rows_cell_mapped"],
            "rows_all_entities_mapped": s["rows_all_entities_mapped"],
            "rows_exact_oneil_overlap": s["rows_exact_oneil_overlap"],
            "rows_after_overlap": s["rows_after_overlap"],
            "unique_triplets_after_overlap": len(s["unique_triplets_after_overlap"]),
            "positive_gt10": s["positive_gt10"],
            "negative_lt0": s["negative_lt0"],
            "gray_0_to_10": s["gray_0_to_10"],
            "classifiable_rows": classifiable,
            "min_class_count": min_class,
            "positive_fraction_among_classifiable": pos_frac,
            "unique_classifiable_triplets": len(s["unique_classifiable_triplets"]),
            "shared_local_drugs": len(s["unique_drugs"]),
            "shared_local_cells": len(s["unique_cells"]),
            "has_both_classes": bool(
                s["positive_gt10"] > 0 and s["negative_lt0"] > 0
            ),
        })

    df = pd.DataFrame(rows)

    # Pre-inference ranking only by feasibility, never by model performance.
    # Priority: not excluded, both classes, larger minimum class, larger unique coverage.
    df["candidate_feasible"] = (
        (~df["excluded_by_name_rule"])
        & df["has_both_classes"]
        & (df["unique_classifiable_triplets"] > 0)
    )
    df = df.sort_values(
        [
            "candidate_feasible",
            "min_class_count",
            "unique_classifiable_triplets",
            "shared_local_drugs",
            "shared_local_cells",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    df.to_csv(outdir / "drugcomb_study_feasibility.csv", index=False)

    feasible = df[df["candidate_feasible"]].copy()
    feasible.to_csv(outdir / "drugcomb_candidate_studies_only.csv", index=False)

    summary = {
        "purpose": "pre-inference source/coverage/class-balance audit",
        "model_predictions_inspected": False,
        "drugcomb_file": str(summary_path),
        "rows_scanned": int(total),
        "studies_seen": int(len(df)),
        "local_drugs": int(len(drugs)),
        "local_cells": int(len(cells)),
        "excluded_study_regex": args.exclude_study_regex,
        "fixed_label_rule": {
            "positive": "synergy_loewe > 10",
            "negative": "synergy_loewe < 0",
            "gray_excluded": "0 <= synergy_loewe <= 10",
        },
        "feasible_candidate_studies": int(len(feasible)),
        "top_candidates": (
            feasible.head(15)[
                [
                    "study_name",
                    "unique_classifiable_triplets",
                    "positive_gt10",
                    "negative_lt0",
                    "min_class_count",
                    "shared_local_drugs",
                    "shared_local_cells",
                ]
            ].to_dict(orient="records")
        ),
        "selection_principle": (
            "Choose an additional external study before model inference using only "
            "explicit study identity, shared input-feature coverage, exact O'Neil overlap "
            "removal, and fixed Loewe class availability. Do not select by AUROC/PR-AUC."
        ),
    }
    (outdir / "study_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nAUDIT COMPLETE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nUpload the entire output folder:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
