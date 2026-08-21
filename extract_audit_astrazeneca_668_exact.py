#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5: extract and audit the exact 668-row AstraZeneca independent test set
from the official DeepDDS repository.

This stage DOES NOT run MFMNet and DOES NOT tune anything.
It establishes the exact source artifact, labels, drug/cell coverage, SMILES
mapping, and the DeepDDS code/data references needed to reconstruct the
external inference path.

Expected official repository artifacts:
  data/independent_set/independent_input.csv
  data/independent_set/independent_set_newcombinatioin.csv
  data/independent_set/independent_drug.csv
  data/independent_set/independent_cell.csv
  data/independent_set/independent_cell_features_954.csv

The script never samples or trims records to obtain n=668.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
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


def canon_smiles(x):
    if Chem is None or pd.isna(x):
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
    p.add_argument("--deepdds-repo", required=True,
                   help="Official DeepDDS repository ZIP or extracted directory")
    p.add_argument("--output", default="./result/astra_668_exact_source_audit")
    return p.parse_args()


def locate(root: Path, suffix: str) -> Path:
    hits = [p for p in root.rglob("*") if p.is_file()
            and str(p).replace("\\", "/").endswith(suffix)]
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {suffix}; found {len(hits)}: {hits[:10]}")
    return hits[0]


def read_text(path: Path):
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def main():
    args = parse_args()
    src = Path(args.deepdds_repo)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(src)

    tmp = None
    if src.is_file() and src.suffix.lower() == ".zip":
        tmp = tempfile.mkdtemp(prefix="deepdds_astra668_")
        root = Path(tmp)
        with zipfile.ZipFile(src) as z:
            z.extractall(root)
    elif src.is_dir():
        root = src
    else:
        raise ValueError("--deepdds-repo must be a ZIP or directory")

    try:
        input_path = locate(root, "data/independent_set/independent_input.csv")
        combo_path = locate(root, "data/independent_set/independent_set_newcombinatioin.csv")
        drug_path = locate(root, "data/independent_set/independent_drug.csv")
        cell_path = locate(root, "data/independent_set/independent_cell.csv")
        cell954_path = locate(root, "data/independent_set/independent_cell_features_954.csv")
        readme_path = locate(root, "data/independent_set/readme.txt")

        inp = pd.read_csv(input_path, low_memory=False)
        req = {"drug1", "drug2", "cell", "label"}
        missing = req.difference(inp.columns)
        if missing:
            raise ValueError(f"independent_input.csv missing {sorted(missing)}")
        if len(inp) != 668:
            raise ValueError(f"Expected 668 rows, got {len(inp)}")

        inp["drug1"] = inp["drug1"].astype(str)
        inp["drug2"] = inp["drug2"].astype(str)
        inp["cell"] = inp["cell"].astype(str)
        inp["label"] = pd.to_numeric(inp["label"], errors="raise").astype(int)

        a1, a2 = [], []
        for a, b in inp[["drug1", "drug2"]].itertuples(index=False):
            x, y = canon_pair(a, b)
            a1.append(x); a2.append(y)
        inp["canonical_drug1"] = a1
        inp["canonical_drug2"] = a2
        inp["canonical_key"] = (
            inp["canonical_drug1"] + "||" + inp["canonical_drug2"] + "||" + inp["cell"]
        )
        inp.to_csv(outdir / "astra_668_exact_independent_input.csv", index=False)

        # Basic exact-source audit.
        unique_drugs = sorted(set(inp["drug1"]) | set(inp["drug2"]))
        unique_cells = sorted(set(inp["cell"]))
        labels = inp["label"].value_counts().sort_index()

        pd.DataFrame({"drug": unique_drugs}).to_csv(
            outdir / "astra_668_exact_drugs.csv", index=False
        )
        pd.DataFrame({"cell": unique_cells}).to_csv(
            outdir / "astra_668_exact_cells.csv", index=False
        )

        # Inspect the richer source table and map each exact 668 row to source records.
        combo = pd.read_csv(combo_path, low_memory=False)
        expected_combo = {
            "Cell line name", "Compound A", "Compound B",
            "Smiles A", "Smiles B", "label"
        }
        missing_combo = expected_combo.difference(combo.columns)
        if missing_combo:
            raise ValueError(f"newcombinatioin file missing {sorted(missing_combo)}")

        for c in ["Cell line name", "Compound A", "Compound B"]:
            combo[c] = combo[c].astype(str)

        # Build orientation-insensitive index by normalized names and cell.
        combo_index = defaultdict(list)
        for i, r in combo.iterrows():
            da, db = canon_pair(norm_token(r["Compound A"]), norm_token(r["Compound B"]))
            key = (da, db, norm_token(r["Cell line name"]))
            combo_index[key].append(i)

        match_rows = []
        smiles_by_input_drug = defaultdict(set)
        matched = 0
        multi = 0

        for i, r in inp.iterrows():
            da, db = canon_pair(norm_token(r["drug1"]), norm_token(r["drug2"]))
            key = (da, db, norm_token(r["cell"]))
            ids = combo_index.get(key, [])
            if ids:
                matched += 1
            if len(ids) > 1:
                multi += 1

            source_labels = []
            source_scores = []
            source_datasets = []
            source_challenges = []

            for j in ids:
                q = combo.loc[j]
                try:
                    source_labels.append(int(float(q["label"])))
                except Exception:
                    pass
                if "Synergy score " in combo.columns:
                    try:
                        source_scores.append(float(q["Synergy score "]))
                    except Exception:
                        pass
                if "Dataset" in combo.columns:
                    source_datasets.append(str(q["Dataset"]))
                if "Challenge" in combo.columns:
                    source_challenges.append(str(q["Challenge"]))

                # Assign smiles back to the exact input drug names using normalized name.
                ca = str(q["Compound A"]); cb = str(q["Compound B"])
                sa = q["Smiles A"]; sb = q["Smiles B"]
                if norm_token(ca) == norm_token(r["drug1"]):
                    cs = canon_smiles(sa)
                    if cs: smiles_by_input_drug[str(r["drug1"])].add(cs)
                if norm_token(ca) == norm_token(r["drug2"]):
                    cs = canon_smiles(sa)
                    if cs: smiles_by_input_drug[str(r["drug2"])].add(cs)
                if norm_token(cb) == norm_token(r["drug1"]):
                    cs = canon_smiles(sb)
                    if cs: smiles_by_input_drug[str(r["drug1"])].add(cs)
                if norm_token(cb) == norm_token(r["drug2"]):
                    cs = canon_smiles(sb)
                    if cs: smiles_by_input_drug[str(r["drug2"])].add(cs)

            match_rows.append({
                "input_row": i,
                "drug1": r["drug1"],
                "drug2": r["drug2"],
                "cell": r["cell"],
                "input_label": int(r["label"]),
                "n_source_matches": len(ids),
                "source_row_indices": "|".join(map(str, ids)),
                "source_labels": "|".join(map(str, sorted(set(source_labels)))),
                "source_synergy_scores": "|".join(map(str, source_scores)),
                "source_datasets": "|".join(sorted(set(source_datasets))),
                "source_challenges": "|".join(sorted(set(source_challenges))),
                "label_agreement_all_matches": (
                    bool(source_labels) and all(x == int(r["label"]) for x in source_labels)
                ),
            })

        match_df = pd.DataFrame(match_rows)
        match_df.to_csv(outdir / "astra_668_to_rich_source_match.csv", index=False)

        drug_smiles_rows = []
        for d in unique_drugs:
            vals = sorted(smiles_by_input_drug.get(d, set()))
            drug_smiles_rows.append({
                "drug": d,
                "n_unique_canonical_smiles_from_source": len(vals),
                "canonical_smiles": " | ".join(vals),
                "smiles_consistent": len(vals) == 1,
            })
        ds = pd.DataFrame(drug_smiles_rows)
        ds.to_csv(outdir / "astra_668_drug_smiles_audit.csv", index=False)

        # Raw independent drug/cell files: preserve and audit only.
        # Try both header and no-header interpretations because repository files vary.
        raw_audits = []
        for name, p in [
            ("independent_drug", drug_path),
            ("independent_cell", cell_path),
            ("independent_cell_features_954", cell954_path),
        ]:
            for header_mode in ["infer", "none"]:
                try:
                    df = pd.read_csv(
                        p, header=0 if header_mode == "infer" else None, low_memory=False
                    )
                    raw_audits.append({
                        "file": name,
                        "header_mode": header_mode,
                        "rows": int(len(df)),
                        "cols": int(df.shape[1]),
                        "columns_preview": " | ".join(map(str, list(df.columns)[:30])),
                        "first_row_preview": " | ".join(map(str, df.iloc[0, :min(20, df.shape[1])].tolist()))
                        if len(df) else "",
                    })
                except Exception as e:
                    raw_audits.append({
                        "file": name, "header_mode": header_mode,
                        "rows": None, "cols": None,
                        "columns_preview": "", "first_row_preview": "",
                        "error": repr(e),
                    })
        pd.DataFrame(raw_audits).to_csv(
            outdir / "independent_feature_file_schema_audit.csv", index=False
        )

        # Search actual repository text/code for exact filenames and independent-set entry points.
        code_refs = []
        needles = [
            "independent_input.csv",
            "independent_allin_smiles.csv",
            "independent_cell_features_954.csv",
            "independent_set_newcombinatioin.csv",
            "independent_set_predict",
        ]
        text_ext = {".py", ".ipynb", ".md", ".txt", ".sh", ".bat", ".r"}
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in text_ext:
                continue
            try:
                text = read_text(p)
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                low = line.lower()
                for n in needles:
                    if n.lower() in low:
                        code_refs.append({
                            "path": str(p.relative_to(root)).replace("\\", "/"),
                            "line": line_no,
                            "needle": n,
                            "text": line[:1500],
                        })
        pd.DataFrame(code_refs).to_csv(
            outdir / "deepdds_independent_code_references.csv", index=False
        )

        # Copy source readme text into output as plain text for audit.
        readme_text = read_text(readme_path)
        (outdir / "deepdds_independent_readme.txt").write_text(
            readme_text, encoding="utf-8"
        )

        summary = {
            "source_artifact": str(input_path.relative_to(root)).replace("\\", "/"),
            "rows": int(len(inp)),
            "unique_drugs": int(len(unique_drugs)),
            "unique_cells": int(len(unique_cells)),
            "label_counts": {str(int(k)): int(v) for k, v in labels.items()},
            "duplicate_canonical_pair_cell_rows": int(inp["canonical_key"].duplicated().sum()),
            "rich_source_rows": int(len(combo)),
            "exact_input_rows_matched_to_rich_source_by_name_cell": int(matched),
            "input_rows_with_multiple_rich_source_matches": int(multi),
            "all_matched_source_labels_agree_with_input": bool(
                match_df.loc[match_df["n_source_matches"] > 0, "label_agreement_all_matches"].all()
            ) if matched else False,
            "drugs_with_one_consistent_smiles": int(ds["smiles_consistent"].sum()),
            "drugs_with_zero_source_smiles": int((ds["n_unique_canonical_smiles_from_source"] == 0).sum()),
            "drugs_with_multiple_source_smiles": int((ds["n_unique_canonical_smiles_from_source"] > 1).sum()),
            "rdkit_available": Chem is not None,
            "code_reference_hits": int(len(code_refs)),
            "important_note": (
                "This establishes the exact official 668-row source artifact. "
                "It does not establish the MFMNet 93.3% result. The next stage must "
                "reconstruct the MFMNet-compatible external feature path and then "
                "evaluate frozen checkpoints without tuning."
            ),
        }
        (outdir / "astra_668_exact_source_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("\nUpload the entire output folder:")
        print(outdir.resolve())

    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
