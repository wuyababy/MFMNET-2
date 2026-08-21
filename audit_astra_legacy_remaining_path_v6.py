#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit the remaining provenance questions for the legacy MFMNet AstraZeneca-668 claim.

This script DOES NOT run MolFormer, DOES NOT run MFMNet, and DOES NOT use
AstraZeneca labels or prediction performance for any selection.

It answers only source/provenance questions that remain after V5:
1) What does the ORIGINAL DeepDDS code actually do for independent cell features?
2) What are the exact first-substring matches for 22RV1 and SW48 in source row order?
3) Does the original MFMNet project contain code that generates all_drug_embedding.csv,
   and if so what model/input/batch/seed/eval procedure is written there?
4) Does the old MFMNet execution path combine MDSyn and EnhancedMDSyn, and with what
   fixed rule/weight according to code?
5) Which legacy checkpoints are actually present?
6) SHA256 hashes of the critical source/data/checkpoint artifacts for traceability.

No external metric, threshold, seed, checkpoint, embedding representation, cell mapping,
or ensemble weight is chosen in this audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

TEXT_EXTS = {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".ipynb", ".sh", ".bat", ".ps1"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--codes-root", default=".", help="Original MFMNet codes directory")
    p.add_argument("--deepdds-repo", required=True, help="Official DeepDDS ZIP or extracted directory")
    p.add_argument("--output", default="./result/astra_legacy_remaining_path_audit_v6")
    return p.parse_args()


def sha256(path: Path, block=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_text(path: Path):
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def open_repo(path: str):
    p = Path(path).resolve()
    if p.is_dir():
        return p, None
    if p.is_file() and p.suffix.lower() == ".zip":
        td = tempfile.mkdtemp(prefix="deepdds_audit_v6_")
        root = Path(td)
        with zipfile.ZipFile(p) as z:
            z.extractall(root)
        return root, td
    raise ValueError("--deepdds-repo must be a directory or ZIP")


def locate_all(root: Path, suffix: str):
    norm = suffix.replace("\\", "/")
    return sorted([
        p for p in root.rglob("*")
        if p.is_file() and str(p).replace("\\", "/").endswith(norm)
    ])


def context_for_patterns(path: Path, patterns, radius=25):
    lines = read_text(path).splitlines()
    hits = []
    for i, line in enumerate(lines):
        if any(re.search(p, line, flags=re.I) for p in patterns):
            hits.append(i)
    if not hits:
        return ""
    keep = set()
    for i in hits:
        keep.update(range(max(0, i-radius), min(len(lines), i+radius+1)))
    out, prev = [], None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            out.append("\n... omitted ...\n")
        out.append(f"{i+1:05d}: {lines[i]}")
        prev = i
    return "\n".join(out)


def scan_code(root: Path):
    patterns = {
        "embedding_file": re.compile(r"all_drug_embedding|molformer_embedding1|molformer_embedding2", re.I),
        "molformer": re.compile(r"molformer|pooler_output|AutoModel\.from_pretrained|AutoTokenizer\.from_pretrained", re.I),
        "seed": re.compile(r"manual_seed|random\.seed|np\.random\.seed|set_seed|seed\s*=", re.I),
        "batch": re.compile(r"batch_size|DataLoader\s*\(", re.I),
        "model_components": re.compile(r"EnhancedMDSyn|\bMDSyn\b", re.I),
        "ensemble": re.compile(r"ensemble|0\.5\s*\*|\*\s*0\.5|/\s*2(?:\.0)?\b|mean\s*\(", re.I),
        "checkpoints": re.compile(r"Enhanced_best|MDSyn.*best|best.*\.pt|torch\.load|load_state_dict", re.I),
        "external": re.compile(r"independent|AstraZeneca|independent_input|independent_set", re.I),
    }
    rows = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXTS:
            continue
        # Skip generated/result directories except scripts/configs; they add noise.
        rel = str(p.relative_to(root)).replace("\\", "/")
        if any(part in rel.lower() for part in ["/.git/", "__pycache__"]):
            continue
        try:
            text = read_text(p)
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            cats = [name for name, pat in patterns.items() if pat.search(line)]
            if cats:
                rows.append({
                    "path": rel,
                    "line": line_no,
                    "categories": "|".join(cats),
                    "text": line[:4000],
                })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    codes_root = Path(args.codes_root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    deep_root, tmp = open_repo(args.deepdds_repo)

    try:
        # ----------------------------
        # DeepDDS source evidence
        # ----------------------------
        load_hits = locate_all(deep_root, "load_data.py")
        create_hits = locate_all(deep_root, "creat_data_DC.py")
        inp_hits = locate_all(deep_root, "data/independent_set/independent_input.csv")
        feat_hits = locate_all(deep_root, "data/independent_set/independent_cell_features_954.csv")
        meta_hits = locate_all(deep_root, "data/independent_set/independent_cell.csv")

        deep_files = {
            "load_data.py": load_hits[0] if load_hits else None,
            "creat_data_DC.py": create_hits[0] if create_hits else None,
            "independent_input.csv": inp_hits[0] if inp_hits else None,
            "independent_cell_features_954.csv": feat_hits[0] if feat_hits else None,
            "independent_cell.csv": meta_hits[0] if meta_hits else None,
        }

        for name, p in deep_files.items():
            if p and p.suffix.lower() == ".py":
                ctx = context_for_patterns(
                    p,
                    [
                        r"def\s+get_cell_feature",
                        r"cellId\s+in\s+row\[0\]",
                        r"independent_cell_features_954",
                        r"independent_input",
                        r"csv\.reader",
                    ],
                    radius=35,
                )
                (out / f"DeepDDS_{name}.relevant.txt").write_text(ctx, encoding="utf-8")

        source_rule_found = False
        load_rule_text = ""
        if deep_files["load_data.py"]:
            load_rule_text = read_text(deep_files["load_data.py"])
            source_rule_found = bool(
                re.search(r"if\s+cellId\s+in\s+row\s*\[\s*0\s*\]", load_rule_text)
            )

        # Cell lookup in exact source order. No labels are read.
        cell_rows = []
        special = {}
        if deep_files["independent_input.csv"] and deep_files["independent_cell_features_954.csv"]:
            inp = pd.read_csv(deep_files["independent_input.csv"], usecols=["cell"], low_memory=False)
            needed = sorted(inp["cell"].astype(str).unique())
            with deep_files["independent_cell_features_954.csv"].open(
                "r", newline="", encoding="utf-8-sig"
            ) as fh:
                feat_rows = list(csv.reader(fh))

            for cell in needed:
                matches = []
                for row_index, row in enumerate(feat_rows):
                    if not row:
                        continue
                    sid = str(row[0])
                    if cell in sid:
                        matches.append((row_index, sid, len(row)-1))
                selected = matches[0] if matches else None
                cell_rows.append({
                    "cell": cell,
                    "n_substring_matches": len(matches),
                    "all_matches_in_source_order": " | ".join(x[1] for x in matches),
                    "first_match_row_index": selected[0] if selected else "",
                    "first_match_identifier": selected[1] if selected else "",
                    "first_match_feature_count": selected[2] if selected else "",
                })
                if cell.upper() in {"22RV1", "SW48"}:
                    special[cell.upper()] = {
                        "n_substring_matches": len(matches),
                        "matches": [x[1] for x in matches],
                        "first_match": selected[1] if selected else None,
                        "first_match_feature_count": selected[2] if selected else None,
                    }

            pd.DataFrame(cell_rows).to_csv(out / "deepdds_cell_first_substring_audit.csv", index=False)

        # Metadata trace only.
        if deep_files["independent_cell.csv"]:
            cm = pd.read_csv(deep_files["independent_cell.csv"], low_memory=False)
            mask = pd.Series(False, index=cm.index)
            for col in cm.columns:
                s = cm[col].astype(str).str.upper()
                mask |= s.str.contains("22RV1", regex=False) | s.str.contains("924100", regex=False) | s.str.contains("SW48", regex=False)
            cm.loc[mask].to_csv(out / "deepdds_special_cell_metadata_rows.csv", index=False)

        # Official DeepDDS prediction files: row counts only, no scores/labels are inspected.
        pred_counts = []
        for p in deep_root.rglob("*.csv"):
            rel = str(p.relative_to(deep_root)).replace("\\", "/")
            low = rel.lower()
            if "independent" not in low or not any(k in low for k in ["predict", "result"]):
                continue
            try:
                d0 = pd.read_csv(p, header=None, low_memory=False)
                pred_counts.append({"path": rel, "headerless_rows": int(len(d0)), "cols": int(d0.shape[1])})
            except Exception as exc:
                pred_counts.append({"path": rel, "error": repr(exc)})
        pd.DataFrame(pred_counts).to_csv(out / "deepdds_independent_prediction_file_counts.csv", index=False)

        # ----------------------------
        # Original MFMNet source evidence
        # ----------------------------
        code_hits = scan_code(codes_root)
        code_hits.to_csv(out / "mfmnet_source_keyword_hits.csv", index=False)

        # Write concentrated contexts for the most relevant original source files.
        candidate_names = ["enhanced_main.py", "loader.py", "enhanced_model.py", "model.py"]
        for name in candidate_names:
            hits = locate_all(codes_root, name)
            for j, p in enumerate(hits[:5]):
                ctx = context_for_patterns(
                    p,
                    [
                        r"all_drug_embedding", r"molformer_embedding",
                        r"EnhancedMDSyn", r"\bMDSyn\b",
                        r"Enhanced_best", r"torch\.load", r"load_state_dict",
                        r"0\.5\s*\*", r"\*\s*0\.5", r"ensemble",
                        r"softmax", r"batch_size", r"gene_vector",
                    ],
                    radius=40,
                )
                (out / f"MFMNet_{name}_{j}.relevant.txt").write_text(ctx, encoding="utf-8")

        # Search all source files containing clear MolFormer-generation evidence and save full contexts.
        emb_candidates = []
        if len(code_hits):
            emb_mask = code_hits["categories"].str.contains("molformer|embedding_file", regex=True)
            for rel in sorted(code_hits.loc[emb_mask, "path"].unique()):
                p = codes_root / rel
                if p.exists() and p.is_file():
                    ctx = context_for_patterns(
                        p,
                        [r"molformer", r"pooler_output", r"all_drug_embedding", r"AutoModel", r"AutoTokenizer", r"batch_size", r"seed"],
                        radius=45,
                    )
                    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)
                    (out / f"embedding_candidate__{safe}.txt").write_text(ctx, encoding="utf-8")
                    emb_candidates.append(rel)

        # Checkpoint inventory without loading pickle/state dict.
        ckpt_rows = []
        for p in sorted(codes_root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
                continue
            try:
                ckpt_rows.append({
                    "path": str(p.relative_to(codes_root)).replace("\\", "/"),
                    "size_bytes": int(p.stat().st_size),
                    "sha256": sha256(p),
                })
            except Exception as exc:
                ckpt_rows.append({"path": str(p), "error": repr(exc)})
        ckpt_df = pd.DataFrame(ckpt_rows)
        ckpt_df.to_csv(out / "mfmnet_checkpoint_inventory.csv", index=False)

        # Critical hashes.
        hash_rows = []
        critical_candidates = [
            codes_root / "data/raw/all_ddi.csv",
            codes_root / "data/raw/all_drug_embedding.csv",
            codes_root / "data/raw/gene_vector.csv",
            codes_root / "enhanced_main.py",
            codes_root / "enhanced_model.py",
            codes_root / "loader.py",
            codes_root / "model.py",
            codes_root / "result/models/Enhanced_best.pt",
        ]
        for p in list(deep_files.values()) + critical_candidates:
            if p and Path(p).exists() and Path(p).is_file():
                pp = Path(p)
                try:
                    base = deep_root if str(pp).startswith(str(deep_root)) else codes_root
                    rel = str(pp.relative_to(base)).replace("\\", "/")
                except Exception:
                    rel = str(pp)
                hash_rows.append({"path": rel, "size_bytes": int(pp.stat().st_size), "sha256": sha256(pp)})
        pd.DataFrame(hash_rows).drop_duplicates(subset=["sha256", "path"]).to_csv(
            out / "critical_artifact_sha256.csv", index=False
        )

        # Conservative source-only flags. These are not performance decisions.
        ensemble_evidence_rows = 0
        if len(code_hits):
            comp = code_hits[code_hits["categories"].str.contains("model_components", regex=False)]
            ens = code_hits[code_hits["categories"].str.contains("ensemble", regex=False)]
            ensemble_evidence_rows = int(len(ens))
            files_with_both_models = sorted(set(comp["path"]))
        else:
            files_with_both_models = []

        mdsyn_ckpts = []
        enhanced_ckpts = []
        if len(ckpt_df) and "path" in ckpt_df.columns:
            for x in ckpt_df["path"].astype(str):
                xl = x.lower()
                if "enhanced" in xl:
                    enhanced_ckpts.append(x)
                elif "mdsyn" in xl or "best" in xl:
                    mdsyn_ckpts.append(x)

        summary = {
            "purpose": "source-only audit of remaining legacy AstraZeneca-668 provenance questions",
            "astra_labels_used_for_selection": False,
            "model_inference_performed": False,
            "molformer_run_performed": False,
            "deepdds_original_get_cell_feature_substring_rule_found": source_rule_found,
            "special_cell_first_substring_results": special,
            "deepdds_official_prediction_files_found": int(len(pred_counts)),
            "mfmnet_embedding_generation_candidate_files": emb_candidates,
            "mfmnet_source_files_with_MDSyn_or_EnhancedMDSyn_hits": files_with_both_models,
            "mfmnet_ensemble_keyword_or_fixed_average_hit_rows": ensemble_evidence_rows,
            "checkpoint_count": int(len(ckpt_df)),
            "enhanced_named_checkpoints": enhanced_ckpts,
            "other_mdsyn_or_best_named_checkpoints": mdsyn_ckpts,
            "safe_to_freeze_cell_lookup_from_source": bool(source_rule_found and "22RV1" in special and special["22RV1"].get("first_match")),
            "safe_to_freeze_embedding_rng_batch_path": bool(len(emb_candidates) > 0),
            "safe_to_freeze_model_composition": bool(ensemble_evidence_rows > 0 and len(files_with_both_models) > 0),
            "next_step": (
                "Use the saved original-code contexts to freeze only source-supported input generation and model composition. "
                "Do not choose among seeds, batch sizes, checkpoints, cell mappings, or ensemble weights using AstraZeneca performance."
            ),
        }
        (out / "audit_v6_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("\nAudit output:", out)

    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
