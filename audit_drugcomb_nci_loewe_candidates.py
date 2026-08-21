#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit candidate NCI-ALMANAC Loewe rows inside DrugComb v1.4 summary data.

IMPORTANT
---------
summary_table_v1.4.csv does not contain an explicit study/source column.
Therefore this script DOES NOT assume that every DrugComb row with the same
drug-pair + cell-line key as NCI-ALMANAC is itself an NCI-ALMANAC experiment.

Instead it:
  1) uses the previously prepared OFFICIAL NCI-ALMANAC mapped table as a
     membership skeleton;
  2) scans DrugComb v1.4 for matching canonical drug-pair + cell-line keys;
  3) reports block-id distributions, multiplicities and coverage patterns;
  4) writes candidate rows for source-structure inspection.

It performs NO model inference, NO label tuning and NO source guessing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

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


EXPLICIT_DRUG_ALIASES = {
    norm_token("Fluorouracil"): norm_token("5-FU"),
    norm_token("5 Fluorouracil"): norm_token("5-FU"),
    norm_token("5FU"): norm_token("5-FU"),
    norm_token("7-Ethyl-10-hydroxycamptothecin"): norm_token("SN-38"),
    norm_token("SN38"): norm_token("SN-38"),
    norm_token("Mitomycin"): norm_token("MITOMYCINE"),
    norm_token("Mitomycin C"): norm_token("MITOMYCINE"),
    norm_token("Vorinostat"): norm_token("ZOLINZA"),
}

EXPLICIT_CELL_ALIASES = {
    norm_token("CO:HCT-116"): norm_token("HCT116"),
    norm_token("HCT-116"): norm_token("HCT116"),
    norm_token("CO:HT29"): norm_token("HT29"),
    norm_token("LC:NCI-H23"): norm_token("NCIH23"),
    norm_token("NCI-H23"): norm_token("NCIH23"),
    norm_token("LC:NCI-H460"): norm_token("NCIH460"),
    norm_token("NCI-H460"): norm_token("NCIH460"),
    norm_token("OV:SK-OV-3"): norm_token("SKOV3"),
    norm_token("SK-OV-3"): norm_token("SKOV3"),
    norm_token("CO:SW-620"): norm_token("SW620"),
    norm_token("SW-620"): norm_token("SW620"),
    norm_token("BR:T-47D"): norm_token("T47D"),
    norm_token("T-47D"): norm_token("T47D"),
    norm_token("ME:UACC-62"): norm_token("UACC62"),
    norm_token("UACC-62"): norm_token("UACC62"),
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--drugcomb-summary",
        default="./data/external/summary_table_v1.4.csv",
    )
    p.add_argument(
        "--nci-skeleton",
        default="./result/almanac_feasibility_audit/feature_complete_before_overlap_removal.csv",
        help=(
            "Previously prepared table derived from the official NCI-ALMANAC matrix. "
            "Must contain local_drug1, local_drug2, local_cell."
        ),
    )
    p.add_argument(
        "--external-after-overlap",
        default="./result/almanac_feasibility_audit/external_after_overlap_removal.csv",
        help="Optional previous official-NCI table after exact O'Neil triplet removal.",
    )
    p.add_argument(
        "--output",
        default="./result/drugcomb_nci_loewe_candidate_audit",
    )
    p.add_argument("--chunksize", type=int, default=200_000)
    return p.parse_args()


def make_local_maps(skeleton: pd.DataFrame):
    needed_drugs = sorted(
        set(skeleton["local_drug1"].dropna().astype(str))
        | set(skeleton["local_drug2"].dropna().astype(str))
    )
    needed_cells = sorted(set(skeleton["local_cell"].dropna().astype(str)))

    drug_norm_to_local = {norm_token(x): x for x in needed_drugs}
    cell_norm_to_local = {norm_token(x): x for x in needed_cells}

    # Alias normalized form -> canonical local name, only if target exists.
    for alias_norm, target_norm in EXPLICIT_DRUG_ALIASES.items():
        if target_norm in drug_norm_to_local:
            drug_norm_to_local[alias_norm] = drug_norm_to_local[target_norm]

    for alias_norm, target_norm in EXPLICIT_CELL_ALIASES.items():
        if target_norm in cell_norm_to_local:
            cell_norm_to_local[alias_norm] = cell_norm_to_local[target_norm]

    return needed_drugs, needed_cells, drug_norm_to_local, cell_norm_to_local


def add_canonical_key(df: pd.DataFrame):
    a = df["local_drug1"].astype(str).to_numpy()
    b = df["local_drug2"].astype(str).to_numpy()
    c = df["local_cell"].astype(str).to_numpy()
    p1 = np.where(a <= b, a, b)
    p2 = np.where(a <= b, b, a)
    df = df.copy()
    df["canonical_drug1"] = p1
    df["canonical_drug2"] = p2
    df["canonical_key"] = (
        pd.Series(p1, index=df.index).astype(str)
        + "||"
        + pd.Series(p2, index=df.index).astype(str)
        + "||"
        + pd.Series(c, index=df.index).astype(str)
    )
    return df


def contiguous_segments(block_ids: np.ndarray, gap_cutoff: int):
    ids = np.sort(np.unique(block_ids.astype(np.int64)))
    if len(ids) == 0:
        return []
    starts = [0]
    gaps = np.diff(ids)
    for i, gap in enumerate(gaps, start=1):
        if gap > gap_cutoff:
            starts.append(i)
    starts.append(len(ids))

    out = []
    for s, e in zip(starts[:-1], starts[1:]):
        seg = ids[s:e]
        out.append({
            "gap_cutoff": int(gap_cutoff),
            "segment_start_block_id": int(seg[0]),
            "segment_end_block_id": int(seg[-1]),
            "n_unique_block_ids": int(len(seg)),
            "block_span": int(seg[-1] - seg[0] + 1),
            "density_unique_blocks_per_span": float(len(seg) / (seg[-1] - seg[0] + 1)),
        })
    return out


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.drugcomb_summary)
    skeleton_path = Path(args.nci_skeleton)
    overlap_path = Path(args.external_after_overlap)

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not skeleton_path.exists():
        raise FileNotFoundError(skeleton_path)

    skeleton = pd.read_csv(skeleton_path, low_memory=False)
    req = {"local_drug1", "local_drug2", "local_cell"}
    missing = req.difference(skeleton.columns)
    if missing:
        raise ValueError(f"NCI skeleton missing columns: {sorted(missing)}")

    skeleton = skeleton.dropna(subset=list(req)).copy()
    skeleton = add_canonical_key(skeleton)
    skeleton_keys = set(skeleton["canonical_key"].astype(str))
    skeleton_unique = (
        skeleton[["canonical_key", "canonical_drug1", "canonical_drug2", "local_cell"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    needed_drugs, needed_cells, drug_map, cell_map = make_local_maps(skeleton)

    after_overlap_keys = None
    if overlap_path.exists():
        ext_after = pd.read_csv(overlap_path, low_memory=False)
        if req.issubset(ext_after.columns):
            ext_after = ext_after.dropna(subset=list(req)).copy()
            ext_after = add_canonical_key(ext_after)
            after_overlap_keys = set(ext_after["canonical_key"].astype(str))

    print(f"Official NCI skeleton rows: {len(skeleton)}")
    print(f"Official NCI unique canonical pair-cell keys: {len(skeleton_keys)}")
    print(f"Mapped local drugs in skeleton: {len(needed_drugs)}")
    print(f"Mapped local cells in skeleton: {len(needed_cells)}")
    print("\nScanning DrugComb summary in chunks ...")

    usecols = [
        "block_id",
        "drug_row",
        "drug_col",
        "cell_line_name",
        "synergy_loewe",
        "synergy_zip",
        "synergy_bliss",
        "synergy_hsa",
        "css",
    ]

    matched_chunks = []
    total_rows = 0
    finite_loewe = 0
    mapped_both_drugs = 0
    mapped_cell = 0
    chunk_no = 0

    for chunk in pd.read_csv(
        summary_path,
        usecols=lambda c: c in usecols,
        chunksize=args.chunksize,
        low_memory=False,
    ):
        chunk_no += 1
        total_rows += len(chunk)

        if "synergy_loewe" not in chunk.columns:
            raise ValueError("DrugComb summary has no synergy_loewe column")
        if not {"block_id", "drug_row", "drug_col", "cell_line_name"}.issubset(chunk.columns):
            raise ValueError("DrugComb summary missing block/drug/cell columns")

        chunk["synergy_loewe"] = pd.to_numeric(chunk["synergy_loewe"], errors="coerce")
        finite_mask = np.isfinite(chunk["synergy_loewe"].to_numpy(dtype=float, na_value=np.nan))
        finite_loewe += int(finite_mask.sum())

        d1n = chunk["drug_row"].astype(str).map(norm_token)
        d2n = chunk["drug_col"].astype(str).map(norm_token)
        cn = chunk["cell_line_name"].astype(str).map(norm_token)

        chunk["local_drug1"] = d1n.map(drug_map)
        chunk["local_drug2"] = d2n.map(drug_map)
        chunk["local_cell"] = cn.map(cell_map)

        both_drugs = chunk["local_drug1"].notna() & chunk["local_drug2"].notna()
        cell_ok = chunk["local_cell"].notna()
        mapped_both_drugs += int(both_drugs.sum())
        mapped_cell += int(cell_ok.sum())

        cand = chunk.loc[both_drugs & cell_ok & finite_mask].copy()
        if cand.empty:
            continue

        cand = add_canonical_key(cand)
        cand = cand[cand["canonical_key"].isin(skeleton_keys)].copy()
        if cand.empty:
            continue

        if after_overlap_keys is not None:
            cand["key_retained_after_exact_oneil_overlap_removal"] = cand[
                "canonical_key"
            ].isin(after_overlap_keys)

        matched_chunks.append(cand)

        if chunk_no % 5 == 0:
            print(
                f"  chunks={chunk_no}, rows_scanned={total_rows:,}, "
                f"candidate_rows_so_far={sum(len(x) for x in matched_chunks):,}"
            )

    if not matched_chunks:
        raise RuntimeError("No DrugComb rows matched the official NCI pair-cell skeleton.")

    cand = pd.concat(matched_chunks, ignore_index=True)
    cand["block_id"] = pd.to_numeric(cand["block_id"], errors="coerce")
    cand = cand.dropna(subset=["block_id"]).copy()
    cand["block_id"] = cand["block_id"].astype(np.int64)
    cand = cand.sort_values(["block_id", "canonical_key"]).reset_index(drop=True)

    cand.to_csv(outdir / "drugcomb_rows_matching_nci_skeleton.csv", index=False)

    # Pair-cell multiplicity.
    by_key = (
        cand.groupby(
            ["canonical_key", "canonical_drug1", "canonical_drug2", "local_cell"],
            as_index=False,
        )
        .agg(
            n_drugcomb_blocks=("block_id", "nunique"),
            n_rows=("block_id", "size"),
            block_id_min=("block_id", "min"),
            block_id_max=("block_id", "max"),
            loewe_mean=("synergy_loewe", "mean"),
            loewe_median=("synergy_loewe", "median"),
            loewe_min=("synergy_loewe", "min"),
            loewe_max=("synergy_loewe", "max"),
            loewe_sd=("synergy_loewe", "std"),
        )
        .sort_values(["n_drugcomb_blocks", "canonical_key"], ascending=[False, True])
    )
    by_key.to_csv(outdir / "candidate_pair_cell_multiplicity.csv", index=False)

    # DrugComb block bins.
    bid = cand["block_id"].to_numpy(dtype=np.int64)
    bin_widths = [1000, 5000, 10000, 25000, 50000]
    hist_rows = []
    for width in bin_widths:
        bins = (bid // width) * width
        tmp = cand.assign(block_bin_start=bins)
        g = (
            tmp.groupby("block_bin_start", as_index=False)
            .agg(
                candidate_rows=("block_id", "size"),
                unique_blocks=("block_id", "nunique"),
                unique_nci_keys=("canonical_key", "nunique"),
                loewe_mean=("synergy_loewe", "mean"),
                loewe_median=("synergy_loewe", "median"),
            )
        )
        g["bin_width"] = width
        g["block_bin_end"] = g["block_bin_start"] + width - 1
        hist_rows.append(g)

    hist = pd.concat(hist_rows, ignore_index=True)
    hist = hist[
        [
            "bin_width",
            "block_bin_start",
            "block_bin_end",
            "candidate_rows",
            "unique_blocks",
            "unique_nci_keys",
            "loewe_mean",
            "loewe_median",
        ]
    ]
    hist.to_csv(outdir / "candidate_block_id_histograms.csv", index=False)

    # Gap-based segment diagnostics at multiple cutoffs.
    seg_rows = []
    for cutoff in [10, 50, 100, 500, 1000, 5000, 10000]:
        seg_rows.extend(contiguous_segments(bid, cutoff))
    seg = pd.DataFrame(seg_rows)
    if not seg.empty:
        # add candidate-row and key counts inside segment
        enriched = []
        for r in seg.itertuples(index=False):
            m = (cand["block_id"] >= r.segment_start_block_id) & (
                cand["block_id"] <= r.segment_end_block_id
            )
            sub = cand.loc[m]
            d = r._asdict()
            d["candidate_rows_in_segment"] = int(len(sub))
            d["unique_nci_keys_in_segment"] = int(sub["canonical_key"].nunique())
            d["loewe_mean_in_segment"] = float(sub["synergy_loewe"].mean())
            d["loewe_median_in_segment"] = float(sub["synergy_loewe"].median())
            enriched.append(d)
        seg = pd.DataFrame(enriched)
        seg.to_csv(outdir / "candidate_block_segments.csv", index=False)

    # Distribution of block multiplicity per official NCI key.
    all_key_cov = skeleton_unique.copy()
    block_count_map = cand.groupby("canonical_key")["block_id"].nunique()
    all_key_cov["n_matching_drugcomb_blocks"] = (
        all_key_cov["canonical_key"].map(block_count_map).fillna(0).astype(int)
    )
    all_key_cov["has_matching_drugcomb_loewe"] = (
        all_key_cov["n_matching_drugcomb_blocks"] > 0
    )
    if after_overlap_keys is not None:
        all_key_cov["retained_after_exact_oneil_overlap_removal"] = all_key_cov[
            "canonical_key"
        ].isin(after_overlap_keys)
    all_key_cov.to_csv(outdir / "official_nci_key_drugcomb_coverage.csv", index=False)

    # Large jumps between successive candidate block IDs.
    unique_ids = np.sort(np.unique(bid))
    if len(unique_ids) > 1:
        gaps = np.diff(unique_ids)
        gap_df = pd.DataFrame(
            {
                "previous_block_id": unique_ids[:-1],
                "next_block_id": unique_ids[1:],
                "gap": gaps,
            }
        ).sort_values("gap", ascending=False)
        gap_df.head(500).to_csv(outdir / "largest_candidate_block_id_gaps.csv", index=False)

    summary = {
        "purpose": "source-structure audit only; no NCI source assignment was made",
        "drugcomb_summary": str(summary_path),
        "n_drugcomb_rows_scanned": int(total_rows),
        "n_drugcomb_rows_with_finite_loewe": int(finite_loewe),
        "official_nci_skeleton_rows": int(len(skeleton)),
        "official_nci_unique_pair_cell_keys": int(len(skeleton_keys)),
        "mapped_local_drugs_in_skeleton": int(len(needed_drugs)),
        "mapped_local_cells_in_skeleton": int(len(needed_cells)),
        "candidate_rows_matching_nci_pair_cell_skeleton": int(len(cand)),
        "candidate_unique_block_ids": int(cand["block_id"].nunique()),
        "candidate_unique_nci_pair_cell_keys": int(cand["canonical_key"].nunique()),
        "fraction_official_nci_keys_with_any_drugcomb_loewe_match": float(
            cand["canonical_key"].nunique() / len(skeleton_keys)
        ),
        "candidate_block_id_min": int(cand["block_id"].min()),
        "candidate_block_id_max": int(cand["block_id"].max()),
        "median_blocks_per_matched_nci_key": float(
            by_key["n_drugcomb_blocks"].median()
        ),
        "max_blocks_per_matched_nci_key": int(
            by_key["n_drugcomb_blocks"].max()
        ),
        "warning": (
            "DrugComb summary_table_v1.4.csv has no explicit study/source column. "
            "Matching an official NCI drug-pair-cell key is not sufficient by itself "
            "to prove that a matching DrugComb block originated from NCI-ALMANAC. "
            "Inspect block-id structure before constructing a Loewe external dataset."
        ),
    }

    if after_overlap_keys is not None:
        retained = all_key_cov[
            all_key_cov["retained_after_exact_oneil_overlap_removal"] == True
        ]
        summary["official_nci_keys_after_exact_oneil_overlap_removal"] = int(
            len(retained)
        )
        summary["post_overlap_keys_with_any_drugcomb_loewe_match"] = int(
            retained["has_matching_drugcomb_loewe"].sum()
        )

    (outdir / "candidate_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nAUDIT COMPLETE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nNo source was assigned and no external labels were generated.")
    print("Upload the entire output directory for inspection:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
