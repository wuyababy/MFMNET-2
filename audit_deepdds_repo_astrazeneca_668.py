#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4: audit the official DeepDDS repository for the exact AstraZeneca
independent-test artifact and the code path that created/evaluated it.

This script performs NO model inference and does NOT modify any data.
It accepts either:
  1) a downloaded DeepDDS repository ZIP, or
  2) an extracted DeepDDS repository directory.

It searches for:
  - files/path names containing AstraZeneca / DREAM / independent / test;
  - text/code references to 668, AstraZeneca, AZ-DREAM and external test;
  - tabular files with exactly 668 rows (or near-668 diagnostics);
  - scripts/notebooks that reference candidate data files.

The goal is to recover the original processed independent-test file and
preprocessing path rather than reconstructing 668 rows by post-hoc filtering.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


TEXT_EXT = {
    ".py", ".ipynb", ".md", ".txt", ".rst", ".json", ".yaml", ".yml",
    ".sh", ".bat", ".ps1", ".r", ".m", ".cfg", ".ini", ".toml"
}
TABLE_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
SERIAL_EXT = {".pkl", ".pickle", ".npy", ".npz"}

KEYWORDS = [
    "astrazeneca", "astra zeneca", "az-dream", "az_dream", "dream",
    "independent test", "independent_test", "independent-test",
    "external test", "external_test", "external-test", "668",
]

PATH_TERMS = [
    "astra", "dream", "independent", "external", "test", "az",
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--deepdds-repo",
        required=True,
        help="Official DeepDDS repository ZIP or extracted repository directory",
    )
    p.add_argument(
        "--output",
        default="./result/deepdds_astra_repo_audit",
    )
    p.add_argument(
        "--near-668-window",
        type=int,
        default=50,
        help="Also report tabular objects with row counts within this distance of 668",
    )
    return p.parse_args()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def read_text(path: Path):
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            pass
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def tabular_candidates(path: Path):
    """
    Yield dictionaries for each readable table/sheet/object.
    Never infers header correctness from a target row count.
    """
    ext = path.suffix.lower()

    if ext == ".csv":
        attempts = []
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(path, sep=sep, low_memory=False)
                attempts.append((sep, df))
            except Exception:
                pass
        if attempts:
            # Prefer the parsing with most columns.
            sep, df = max(attempts, key=lambda x: x[1].shape[1])
            yield {
                "object": "csv",
                "sheet_or_key": "",
                "rows": int(len(df)),
                "cols": int(df.shape[1]),
                "columns": " | ".join(map(str, df.columns[:80])),
                "preview": df.head(3).astype(str).to_json(orient="records"),
                "parse_note": f"sep={repr(sep)}",
            }

    elif ext == ".tsv":
        try:
            df = pd.read_csv(path, sep="\t", low_memory=False)
            yield {
                "object": "tsv", "sheet_or_key": "",
                "rows": int(len(df)), "cols": int(df.shape[1]),
                "columns": " | ".join(map(str, df.columns[:80])),
                "preview": df.head(3).astype(str).to_json(orient="records"),
                "parse_note": "sep='\\t'",
            }
        except Exception:
            return

    elif ext in {".xlsx", ".xls"}:
        try:
            book = pd.ExcelFile(path)
        except Exception:
            return
        for sh in book.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sh)
                yield {
                    "object": "excel", "sheet_or_key": str(sh),
                    "rows": int(len(df)), "cols": int(df.shape[1]),
                    "columns": " | ".join(map(str, df.columns[:80])),
                    "preview": df.head(3).astype(str).to_json(orient="records"),
                    "parse_note": "",
                }
            except Exception:
                pass

    elif ext == ".parquet":
        try:
            df = pd.read_parquet(path)
            yield {
                "object": "parquet", "sheet_or_key": "",
                "rows": int(len(df)), "cols": int(df.shape[1]),
                "columns": " | ".join(map(str, df.columns[:80])),
                "preview": df.head(3).astype(str).to_json(orient="records"),
                "parse_note": "",
            }
        except Exception:
            return


def serialized_candidates(path: Path):
    ext = path.suffix.lower()
    if ext in {".pkl", ".pickle"}:
        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
        except Exception:
            return
        if isinstance(obj, pd.DataFrame):
            yield {
                "object": "pickle_dataframe",
                "sheet_or_key": "",
                "rows": int(len(obj)),
                "cols": int(obj.shape[1]),
                "columns": " | ".join(map(str, obj.columns[:80])),
                "preview": obj.head(3).astype(str).to_json(orient="records"),
                "parse_note": type(obj).__name__,
            }
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, pd.DataFrame):
                    yield {
                        "object": "pickle_dict_dataframe",
                        "sheet_or_key": str(k),
                        "rows": int(len(v)),
                        "cols": int(v.shape[1]),
                        "columns": " | ".join(map(str, v.columns[:80])),
                        "preview": v.head(3).astype(str).to_json(orient="records"),
                        "parse_note": f"dict[{k!r}]",
                    }
                elif isinstance(v, np.ndarray) and v.ndim >= 1:
                    yield {
                        "object": "pickle_dict_ndarray",
                        "sheet_or_key": str(k),
                        "rows": int(v.shape[0]),
                        "cols": int(v.shape[1]) if v.ndim >= 2 else 1,
                        "columns": "",
                        "preview": repr(v[:3]),
                        "parse_note": f"shape={v.shape}",
                    }
        elif isinstance(obj, np.ndarray) and obj.ndim >= 1:
            yield {
                "object": "pickle_ndarray",
                "sheet_or_key": "",
                "rows": int(obj.shape[0]),
                "cols": int(obj.shape[1]) if obj.ndim >= 2 else 1,
                "columns": "",
                "preview": repr(obj[:3]),
                "parse_note": f"shape={obj.shape}",
            }

    elif ext == ".npy":
        try:
            arr = np.load(path, allow_pickle=True)
        except Exception:
            return
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            yield {
                "object": "npy",
                "sheet_or_key": "",
                "rows": int(arr.shape[0]),
                "cols": int(arr.shape[1]) if arr.ndim >= 2 else 1,
                "columns": "",
                "preview": repr(arr[:3]),
                "parse_note": f"shape={arr.shape}; dtype={arr.dtype}",
            }

    elif ext == ".npz":
        try:
            z = np.load(path, allow_pickle=True)
        except Exception:
            return
        for k in z.files:
            arr = z[k]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                yield {
                    "object": "npz",
                    "sheet_or_key": str(k),
                    "rows": int(arr.shape[0]),
                    "cols": int(arr.shape[1]) if arr.ndim >= 2 else 1,
                    "columns": "",
                    "preview": repr(arr[:3]),
                    "parse_note": f"shape={arr.shape}; dtype={arr.dtype}",
                }


def inspect_notebook_text(text):
    # Raw .ipynb JSON is searchable as text; this keeps cell references visible.
    return text


def main():
    args = parse_args()
    src = Path(args.deepdds_repo)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(src)

    temp_dir = None
    if src.is_file() and src.suffix.lower() == ".zip":
        temp_dir = tempfile.mkdtemp(prefix="deepdds_repo_audit_")
        root = Path(temp_dir)
        with zipfile.ZipFile(src) as z:
            z.extractall(root)
    elif src.is_dir():
        root = src
    else:
        raise ValueError("--deepdds-repo must be a ZIP or directory")

    files = [p for p in root.rglob("*") if p.is_file()]

    inventory = []
    keyword_hits = []
    table_rows = []

    for p in files:
        rel = safe_rel(p, root)
        ext = p.suffix.lower()
        size = p.stat().st_size
        path_l = rel.lower()

        inventory.append({
            "path": rel,
            "extension": ext,
            "size_bytes": int(size),
            "path_keyword_flag": any(t in path_l for t in PATH_TERMS),
        })

        # Text/code search.
        if ext in TEXT_EXT or size <= 2_000_000:
            text = read_text(p)
            if text:
                lower = text.lower()
                for kw in KEYWORDS:
                    if kw.lower() in lower:
                        # capture up to first 20 line hits for each keyword/file
                        for i, line in enumerate(text.splitlines(), start=1):
                            if kw.lower() in line.lower():
                                keyword_hits.append({
                                    "path": rel,
                                    "keyword": kw,
                                    "line": i,
                                    "text": line[:1200],
                                })
                                if sum(
                                    1 for x in keyword_hits
                                    if x["path"] == rel and x["keyword"] == kw
                                ) >= 20:
                                    break

        # Tabular/serialized structural audit.
        yielded = []
        if ext in TABLE_EXT:
            yielded = list(tabular_candidates(p))
        elif ext in SERIAL_EXT:
            yielded = list(serialized_candidates(p))

        for x in yielded:
            rows = x.get("rows")
            x.update({
                "path": rel,
                "distance_from_668": abs(int(rows) - 668) if rows is not None else None,
                "exact_668_rows": bool(rows == 668),
                "near_668_rows": bool(
                    rows is not None and abs(int(rows) - 668) <= args.near_668_window
                ),
                "path_keyword_flag": any(t in path_l for t in PATH_TERMS),
            })
            table_rows.append(x)

    inv_df = pd.DataFrame(inventory).sort_values(
        ["path_keyword_flag", "path"], ascending=[False, True]
    )
    hit_df = pd.DataFrame(keyword_hits)
    tab_df = pd.DataFrame(table_rows)

    inv_df.to_csv(outdir / "repo_file_inventory.csv", index=False)
    hit_df.to_csv(outdir / "keyword_hits.csv", index=False)

    if len(tab_df):
        tab_df = tab_df.sort_values(
            ["exact_668_rows", "near_668_rows", "path_keyword_flag", "distance_from_668"],
            ascending=[False, False, False, True],
        )
        tab_df.to_csv(outdir / "tabular_object_inventory.csv", index=False)

        likely = tab_df[
            tab_df["exact_668_rows"]
            | tab_df["near_668_rows"]
            | tab_df["path_keyword_flag"]
        ].copy()
        likely.to_csv(outdir / "likely_external_test_files.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "tabular_object_inventory.csv", index=False)
        pd.DataFrame().to_csv(outdir / "likely_external_test_files.csv", index=False)

    # Rank text/code files by relevant hits.
    if len(hit_df):
        key_rank = (
            hit_df.assign(
                high=lambda d: d["keyword"].str.lower().isin(
                    ["astrazeneca", "az-dream", "independent test",
                     "independent_test", "external test", "external_test", "668"]
                )
            )
            .groupby("path")
            .agg(
                keyword_hit_count=("keyword", "size"),
                distinct_keywords=("keyword", "nunique"),
                high_priority_hits=("high", "sum"),
            )
            .reset_index()
            .sort_values(
                ["high_priority_hits", "distinct_keywords", "keyword_hit_count"],
                ascending=False,
            )
        )
        key_rank.to_csv(outdir / "ranked_code_and_text_candidates.csv", index=False)
    else:
        key_rank = pd.DataFrame()
        key_rank.to_csv(outdir / "ranked_code_and_text_candidates.csv", index=False)

    exact_files = []
    near_files = []
    if len(tab_df):
        exact_files = tab_df[tab_df["exact_668_rows"]][
            ["path", "object", "sheet_or_key", "rows", "cols", "columns", "parse_note"]
        ].to_dict(orient="records")
        near_files = tab_df[
            tab_df["near_668_rows"] & ~tab_df["exact_668_rows"]
        ][
            ["path", "object", "sheet_or_key", "rows", "cols",
             "distance_from_668", "columns", "parse_note"]
        ].head(30).to_dict(orient="records")

    summary = {
        "purpose": "recover original DeepDDS AstraZeneca 668 artifact/code path",
        "model_predictions_inspected": False,
        "repo_source": str(src),
        "files_scanned": int(len(files)),
        "text_keyword_hits": int(len(hit_df)),
        "tabular_objects_read": int(len(tab_df)),
        "exact_668_tabular_objects": int(tab_df["exact_668_rows"].sum()) if len(tab_df) else 0,
        "exact_668_candidates": exact_files,
        "near_668_candidates": near_files,
        "top_code_text_candidates": (
            key_rank.head(20).to_dict(orient="records") if len(key_rank) else []
        ),
        "next_step_rule": (
            "Prefer an original repository data artifact or explicit preprocessing code path. "
            "Do not create a 668-row subset by choosing rows based on MFMNet predictions."
        ),
    }
    (outdir / "deepdds_astra_repo_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nUpload the entire output folder:")
    print(outdir.resolve())

    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
