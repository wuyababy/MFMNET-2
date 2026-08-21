#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: resolve DeepDDS AstraZeneca S4/S5 names to DrugComb summary names
using DrugComb identifier workbooks, then reconstruct the source-defined
independent-test subset.

This stage is schema-agnostic: it scans every row of the DrugComb identifier
workbooks and links a row to a DrugComb summary name when that row contains
that summary name (or normalized equivalent). It then matches DeepDDS S4/S5
entries to the same identifier row using:
  - exact/normalized drug aliases,
  - challenge drug names,
  - canonical SMILES (when RDKit is available),
  - exact/normalized cell-line names.

No model predictions are used. The script never trims or samples rows to force
n=668. It writes an exact candidate only if source-defined mapping itself
reproduces 668 unique unordered drug-pair-cell-line combinations.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except Exception:
    Chem = None


def norm_token(x: str) -> str:
    s = str(x).strip().upper()
    s = s.replace("–", "-").replace("—", "-").replace("_", "-")
    return re.sub(r"[^A-Z0-9]+", "", s)


def canon_pair(a: str, b: str):
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def canonical_smiles(x):
    if Chem is None or x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage1-dir", default="./result/astra_668_recovery_stage1")
    p.add_argument("--supp-s4", default="./data/external/supplementary_table_s4_bbab390.xlsx")
    p.add_argument("--supp-s5", default="./data/external/supplementary_table_s5_bbab390.xlsx")
    p.add_argument("--drug-identifiers", default="./data/external/DrugComb_drug_identifiers.xlsx")
    p.add_argument("--cell-identifiers", default="./data/external/DrugComb_cell_line_identifiers.xlsx")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output", default="./result/astra_668_recovery_stage3")
    return p.parse_args()


def read_excel_sheets(path: Path):
    book = pd.ExcelFile(path)
    out = {}
    for sh in book.sheet_names:
        out[sh] = pd.read_excel(path, sheet_name=sh, header=None, dtype=object)
    return out


def iter_identifier_rows(sheets):
    for sh, df in sheets.items():
        for i in range(df.shape[0]):
            vals = []
            for j in range(df.shape[1]):
                v = df.iat[i, j]
                if pd.isna(v):
                    continue
                s = str(v).strip()
                if s:
                    vals.append((j + 1, s))
            if vals:
                yield sh, i + 1, vals


def parse_s4(path: Path):
    df = pd.read_excel(path, sheet_name=0, header=None, dtype=object)

    header_row = None
    for i in range(df.shape[0]):
        vals = [str(x).strip() if not pd.isna(x) else "" for x in df.iloc[i].tolist()]
        if any(v.lower() == "drug name" for v in vals) and any("challenge drug name" in v.lower() for v in vals):
            header_row = i
    if header_row is None:
        raise ValueError("Could not locate S4 drug table header.")

    headers = [str(x).strip() if not pd.isna(x) else f"col_{j}" for j, x in enumerate(df.iloc[header_row])]
    body = df.iloc[header_row + 1:].copy()
    body.columns = headers
    body = body.dropna(how="all").reset_index(drop=True)

    # normalize expected columns
    cols = {norm_token(c): c for c in body.columns}
    drug_col = cols.get(norm_token("Drug name"))
    chall_col = cols.get(norm_token("Challenge drug name"))
    smiles_col = cols.get(norm_token("SMILES"))
    if not drug_col or not chall_col:
        raise ValueError(f"S4 required columns not found. Columns={list(body.columns)}")

    rows = []
    for i, r in body.iterrows():
        drug = str(r[drug_col]).strip() if not pd.isna(r[drug_col]) else ""
        chall = str(r[chall_col]).strip() if not pd.isna(r[chall_col]) else ""
        smi = str(r[smiles_col]).strip() if smiles_col and not pd.isna(r[smiles_col]) else ""
        if not drug:
            continue

        aliases = set()
        aliases.add(drug)
        if chall:
            aliases.add(chall)
        for x in re.split(r"\s*,\s*", drug):
            if x.strip():
                aliases.add(x.strip())
        for x in re.split(r"\s*;\s*", drug):
            if x.strip():
                aliases.add(x.strip())

        rows.append({
            "s4_index": len(rows) + 1,
            "drug_name": drug,
            "challenge_name": chall,
            "smiles": smi,
            "canonical_smiles": canonical_smiles(smi),
            "alias_norms": sorted({norm_token(x) for x in aliases if norm_token(x)}),
        })

    return pd.DataFrame(rows)


def parse_s5(path: Path):
    df = pd.read_excel(path, sheet_name=0, header=None, dtype=object)
    header_row = None
    for i in range(df.shape[0]):
        vals = [str(x).strip() if not pd.isna(x) else "" for x in df.iloc[i].tolist()]
        if any(v.lower() == "cell line name" for v in vals) and any("az-dream" in v.lower() for v in vals):
            header_row = i
    if header_row is None:
        raise ValueError("Could not locate S5 cell-line table header.")

    headers = [str(x).strip() if not pd.isna(x) else f"col_{j}" for j, x in enumerate(df.iloc[header_row])]
    body = df.iloc[header_row + 1:].copy()
    body.columns = headers
    body = body.dropna(how="all").reset_index(drop=True)
    cols = {norm_token(c): c for c in body.columns}
    cell_col = cols.get(norm_token("Cell line name"))
    az_col = cols.get(norm_token("AZ-DREAM"))
    if not cell_col:
        raise ValueError(f"S5 cell-line column not found. Columns={list(body.columns)}")

    if az_col:
        az = pd.to_numeric(body[az_col], errors="coerce")
        body = body[(az == 1) | body[az_col].astype(str).str.strip().str.lower().isin(["1","true","yes","y","✓"])]

    rows = []
    for _, r in body.iterrows():
        cell = str(r[cell_col]).strip() if not pd.isna(r[cell_col]) else ""
        if cell:
            rows.append({
                "s5_index": len(rows) + 1,
                "cell_line_name": cell,
                "cell_norm": norm_token(cell),
            })
    return pd.DataFrame(rows)


def build_identifier_row_index(sheets, external_names, parse_smiles=False):
    ext_norm = defaultdict(list)
    for x in external_names:
        ext_norm[norm_token(x)].append(str(x))

    rows = []
    for sh, row1, vals in iter_identifier_rows(sheets):
        value_norms = {norm_token(v) for _, v in vals if norm_token(v)}
        ext_hits = sorted({
            x for n in value_norms for x in ext_norm.get(n, [])
        })

        smi_set = set()
        if parse_smiles and Chem is not None:
            for _, v in vals:
                cs = canonical_smiles(v)
                if cs:
                    smi_set.add(cs)

        if ext_hits:
            rows.append({
                "sheet": sh,
                "row_1based": row1,
                "values": vals,
                "value_norms": value_norms,
                "external_hits": ext_hits,
                "canonical_smiles_set": smi_set,
            })
    return rows


def resolve_drugs(s4, id_rows):
    out = []
    for r in s4.itertuples(index=False):
        candidates = defaultdict(set)
        evidence = defaultdict(list)

        for ir in id_rows:
            alias_match = set(r.alias_norms).intersection(ir["value_norms"])
            smiles_match = bool(r.canonical_smiles and r.canonical_smiles in ir["canonical_smiles_set"])

            if alias_match or smiles_match:
                for ext in ir["external_hits"]:
                    if alias_match:
                        candidates[ext].add("alias_or_challenge")
                        evidence[ext].append(
                            f"{ir['sheet']}!row{ir['row_1based']}:name"
                        )
                    if smiles_match:
                        candidates[ext].add("canonical_smiles")
                        evidence[ext].append(
                            f"{ir['sheet']}!row{ir['row_1based']}:smiles"
                        )

        ranked = []
        for ext, methods in candidates.items():
            score = (2 if "canonical_smiles" in methods else 0) + (1 if "alias_or_challenge" in methods else 0)
            ranked.append((score, ext, methods))
        ranked.sort(reverse=True)

        if ranked:
            best_score = ranked[0][0]
            best = [x for x in ranked if x[0] == best_score]
        else:
            best = []

        resolved = best[0][1] if len(best) == 1 else ""
        out.append({
            "s4_index": r.s4_index,
            "drug_name": r.drug_name,
            "challenge_name": r.challenge_name,
            "canonical_smiles": r.canonical_smiles or "",
            "resolved_drugcomb_name": resolved,
            "n_best_candidates": len(best),
            "best_candidates": " | ".join(x[1] for x in best),
            "best_match_methods": (
                " | ".join(sorted(best[0][2])) if len(best) == 1 else ""
            ),
            "evidence": (
                " | ".join(evidence[resolved]) if resolved else ""
            ),
        })
    return pd.DataFrame(out)


def resolve_cells(s5, id_rows):
    out = []
    for r in s5.itertuples(index=False):
        cands = defaultdict(list)
        for ir in id_rows:
            if r.cell_norm in ir["value_norms"]:
                for ext in ir["external_hits"]:
                    cands[ext].append(f"{ir['sheet']}!row{ir['row_1based']}")
        resolved = next(iter(cands)) if len(cands) == 1 else ""
        out.append({
            "s5_index": r.s5_index,
            "cell_line_name": r.cell_line_name,
            "resolved_drugcomb_name": resolved,
            "n_candidates": len(cands),
            "candidates": " | ".join(sorted(cands)),
            "evidence": " | ".join(cands.get(resolved, [])) if resolved else "",
        })
    return pd.DataFrame(out)


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    stage1 = Path(args.stage1_dir)
    az_path = stage1 / "astrazeneca_all_summary_rows.csv"
    ddi_path = Path(args.data_root) / "raw" / "all_ddi.csv"
    paths = [
        az_path, Path(args.supp_s4), Path(args.supp_s5),
        Path(args.drug_identifiers), Path(args.cell_identifiers), ddi_path
    ]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    az = pd.read_csv(az_path, low_memory=False)
    ext_drugs = sorted(set(az["drug_row"].astype(str)) | set(az["drug_col"].astype(str)))
    ext_cells = sorted(set(az["cell_line_name"].astype(str)))

    s4 = parse_s4(Path(args.supp_s4))
    s5 = parse_s5(Path(args.supp_s5))
    s4.drop(columns=["alias_norms"]).to_csv(outdir / "deepdds_s4_parsed.csv", index=False)
    s5.to_csv(outdir / "deepdds_s5_parsed.csv", index=False)

    drug_sheets = read_excel_sheets(Path(args.drug_identifiers))
    cell_sheets = read_excel_sheets(Path(args.cell_identifiers))

    drug_id_rows = build_identifier_row_index(drug_sheets, ext_drugs, parse_smiles=True)
    cell_id_rows = build_identifier_row_index(cell_sheets, ext_cells, parse_smiles=False)

    dmap = resolve_drugs(s4, drug_id_rows)
    cmap = resolve_cells(s5, cell_id_rows)
    dmap.to_csv(outdir / "deepdds_s4_to_drugcomb_resolved.csv", index=False)
    cmap.to_csv(outdir / "deepdds_s5_to_drugcomb_resolved.csv", index=False)

    resolved_drugs = set(dmap.loc[dmap["resolved_drugcomb_name"] != "", "resolved_drugcomb_name"])
    resolved_cells = set(cmap.loc[cmap["resolved_drugcomb_name"] != "", "resolved_drugcomb_name"])

    sub = az[
        az["drug_row"].astype(str).isin(resolved_drugs)
        & az["drug_col"].astype(str).isin(resolved_drugs)
        & az["cell_line_name"].astype(str).isin(resolved_cells)
    ].copy()

    c1, c2 = [], []
    for a, b in sub[["drug_row", "drug_col"]].itertuples(index=False):
        x, y = canon_pair(a, b)
        c1.append(x); c2.append(y)
    sub["canonical_drug1"] = c1
    sub["canonical_drug2"] = c2

    sub.to_csv(outdir / "astra_rows_after_identifier_resolved_s4_s5.csv", index=False)

    agg = {"n_summary_rows": ("study_name", "size")}
    if "block_id" in sub.columns:
        agg["n_blocks"] = ("block_id", "nunique")
    for c in ["synergy_loewe", "synergy_zip", "synergy_bliss", "synergy_hsa", "css"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
            agg[f"{c}_mean"] = (c, "mean")
            agg[f"{c}_median"] = (c, "median")

    paircell = (
        sub.groupby(["canonical_drug1", "canonical_drug2", "cell_line_name"], as_index=False)
        .agg(**agg)
        .sort_values(["canonical_drug1", "canonical_drug2", "cell_line_name"])
        .reset_index(drop=True)
    )

    ddi = pd.read_csv(ddi_path, low_memory=False)
    oneil = set()
    for r in ddi[["drug1Name", "drug2Name", "cell"]].itertuples(index=False):
        a, b = canon_pair(r.drug1Name, r.drug2Name)
        oneil.add((a, b, str(r.cell)))

    paircell["exact_oneil_triplet_overlap_by_name"] = [
        (r.canonical_drug1, r.canonical_drug2, str(r.cell_line_name)) in oneil
        for r in paircell[["canonical_drug1", "canonical_drug2", "cell_line_name"]].itertuples(index=False)
    ]
    paircell.to_csv(outdir / "astra_unique_pair_cell_identifier_resolved.csv", index=False)

    unresolved_drugs = dmap[dmap["resolved_drugcomb_name"] == ""]
    unresolved_cells = cmap[cmap["resolved_drugcomb_name"] == ""]
    unresolved_drugs.to_csv(outdir / "unresolved_s4_drugs.csv", index=False)
    unresolved_cells.to_csv(outdir / "unresolved_s5_cells.csv", index=False)

    summary = {
        "purpose": "resolve official DeepDDS S4/S5 membership through DrugComb identifier tables",
        "model_predictions_inspected": False,
        "rdkit_available": Chem is not None,
        "target": {"pair_cell_records": 668, "drugs": 57, "cell_lines": 24},
        "s4_rows": int(len(s4)),
        "s4_resolved_unique_rows": int((dmap["resolved_drugcomb_name"] != "").sum()),
        "s4_unresolved_rows": int((dmap["resolved_drugcomb_name"] == "").sum()),
        "s4_ambiguous_rows": int((dmap["n_best_candidates"] > 1).sum()),
        "s5_rows": int(len(s5)),
        "s5_resolved_unique_rows": int((cmap["resolved_drugcomb_name"] != "").sum()),
        "s5_unresolved_rows": int((cmap["resolved_drugcomb_name"] == "").sum()),
        "s5_ambiguous_rows": int((cmap["n_candidates"] > 1).sum()),
        "resolved_drugcomb_drug_names": int(len(resolved_drugs)),
        "resolved_drugcomb_cell_names": int(len(resolved_cells)),
        "summary_rows_after_resolved_membership": int(len(sub)),
        "unique_unordered_pair_cell": int(len(paircell)),
        "unique_drugs_present": int(len(set(sub["drug_row"].astype(str)) | set(sub["drug_col"].astype(str)))) if len(sub) else 0,
        "unique_cells_present": int(sub["cell_line_name"].nunique()) if len(sub) else 0,
        "exact_oneil_triplet_overlaps_by_name": int(paircell["exact_oneil_triplet_overlap_by_name"].sum()) if len(paircell) else 0,
        "reproduces_668_exactly": bool(len(paircell) == 668),
        "important_note": (
            "No rows were sampled or removed to force 668. If this source-resolved "
            "identifier mapping still does not reproduce 668, the next step is to inspect "
            "the official DeepDDS repository/original processed independent-test file."
        ),
    }

    (outdir / "astra_668_recovery_stage3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if len(paircell) == 668:
        paircell.to_csv(outdir / "astra_668_candidate_exact.csv", index=False)
        print("\nSUCCESS: identifier-resolved official membership produced exactly 668 records.")
    else:
        print(f"\nNOT YET 668: identifier-resolved membership produced {len(paircell)} records.")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nUpload the whole output folder:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
