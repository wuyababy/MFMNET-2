#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V8: strict source/archive audit for the unresolved legacy AstraZeneca path.

This script performs NO model inference, NO MolFormer inference, and does not use
AstraZeneca labels or prediction performance for any selection.

It fixes two V7 false positives:
1) Only real checkpoint extensions (.pt/.pth/.ckpt) can count as checkpoints.
   CSV files whose names contain "mdsyn" can never count.
2) Recovery/audit text copies such as
   embedding_candidate__restore_legacy_*.py.txt are explicitly excluded from
   original embedding-generator evidence, both directly and inside archives.

Primary questions
-----------------
A. Is the exact legacy checkpoint expected by enhanced_main.py,
   result/models/MDSyn_best.pt, present directly or in an archive?
B. Is there an ORIGINAL (non-recovery) MolFormer generator source that actually
   loads MolFormer, uses pooler_output, targets all_drug_embedding/MolFormer
   embeddings, and writes embeddings?
C. Is the fixed MDSyn + EnhancedMDSyn equal-average rule source-confirmed?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

CKPT_EXTS = {".pt", ".pth", ".ckpt"}
TEXT_EXTS = {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".sh", ".bat", ".ps1"}

# Strong exclusions for files created during reviewer/recovery work.
RECOVERY_PATTERNS = [
    r"(^|[/\\])audit[_-]",
    r"(^|[/\\])restore[_-]",
    r"(^|[/\\])recover[_-]",
    r"(^|[/\\])extract[_-]",
    r"embedding_candidate__",
    r"legacy_remaining_path_audit",
    r"legacy_archives_audit",
    r"reviewer",
    r"same5fold",
    r"strict_final",
    r"run_nci",
    r"run_xgboost",
    r"external_no_drug_embedding",
    r"astra_668_legacy_mfmnet_restore",
]

RECOVERY_CONTENT_MARKERS = [
    "restore the legacy mfmnet astrazeneca",
    "embedding-provenance audit",
    "provenance audit",
    "astrazeneca labels used",
    "external labels are used",
    "external labels are not used",
    "deepdds 22rv1",
]

EXPECTED_MDSYN = "result/models/mdsyn_best.pt"
EXPECTED_ENHANCED = "result/models/enhanced_best.pt"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--codes-root", default=".")
    p.add_argument("--output", default="./result/astra_legacy_strict_audit_v8")
    p.add_argument("--max-text-mb", type=float, default=12.0)
    return p.parse_args()


def relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def norm_path(s: str) -> str:
    return str(s).replace("\\", "/").lstrip("./").lower()


def is_recovery_path(s: str) -> bool:
    return any(re.search(p, str(s), re.I) for p in RECOVERY_PATTERNS)


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def is_recovery_content(text: str) -> bool:
    low = text.lower()
    return any(x in low for x in RECOVERY_CONTENT_MARKERS)


def generator_signals(text: str) -> dict:
    low = text.lower()
    model_load = bool(
        re.search(r"automodel\s*\.\s*from_pretrained", text, re.I)
        or "ibm-research/molformer" in low
        or "molformer-xl" in low
    )
    tokenizer_load = bool(re.search(r"autotokenizer\s*\.\s*from_pretrained", text, re.I))
    pooler_output = "pooler_output" in low
    embedding_target = bool(
        "all_drug_embedding" in low
        or re.search(r"molformer[_ -]?embedding", text, re.I)
    )
    write_signal = bool(
        re.search(r"\.to_csv\s*\(", text, re.I)
        or re.search(r"csv\.writer|writerow|np\.save|torch\.save", text, re.I)
    )
    strict = model_load and pooler_output and embedding_target and write_signal
    return {
        "model_load": model_load,
        "tokenizer_load": tokenizer_load,
        "pooler_output": pooler_output,
        "embedding_target": embedding_target,
        "write_signal": write_signal,
        "strict_generator_evidence": strict,
    }


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_class(path: str) -> str:
    """Classification applies ONLY after extension is confirmed as checkpoint."""
    n = norm_path(path)
    base = Path(n).name
    if n.endswith(EXPECTED_MDSYN) or base == "mdsyn_best.pt":
        return "exact_expected_mdsyn_best"
    if n.endswith(EXPECTED_ENHANCED) or base == "enhanced_best.pt":
        return "exact_expected_enhanced_best"
    if "enhanced" not in base and re.search(r"(^|[_\-.])mdsyn([_\-.]|$)", base):
        return "other_named_mdsyn_checkpoint"
    return "other_checkpoint"


def ensemble_evidence(text: str) -> bool:
    return bool(
        re.search(r"model_classes\s*=\s*\[[\s\S]{0,500}\bMDSyn\b[\s\S]{0,500}\bEnhancedMDSyn\b", text, re.I)
        and re.search(r"weights\s*=\s*\[\s*1\.0\s*/\s*len\(models\)\s*\]\s*\*\s*len\(models\)", text)
        and re.search(r"ensemble_prelabels\s*=\s*\(\s*ensemble_preds\s*>\s*0\.5\s*\)", text)
    )


def main():
    args = parse_args()
    root = Path(args.codes_root).resolve()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    max_text = int(args.max_text_mb * 1024 * 1024)

    direct_ckpts = []
    direct_generators = []
    archive_ckpts = []
    archive_generators = []
    archives = []
    ensemble_sources = []

    # ------------------------ direct files ------------------------
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = relpath(p, root)
        suffix = p.suffix.lower()

        if suffix in CKPT_EXTS:
            direct_ckpts.append({
                "path": rel,
                "size_bytes": p.stat().st_size,
                "checkpoint_class": checkpoint_class(rel),
                "sha256": sha256_file(p),
            })

        if Path(rel).name.lower() == "enhanced_main.py" and p.stat().st_size <= max_text:
            try:
                text = decode(p.read_bytes())
                if ensemble_evidence(text):
                    ensemble_sources.append(rel)
            except Exception:
                pass

        if suffix in TEXT_EXTS and p.stat().st_size <= max_text:
            try:
                text = decode(p.read_bytes())
            except Exception:
                continue
            sig = generator_signals(text)
            if sig["strict_generator_evidence"]:
                rec_path = is_recovery_path(rel)
                rec_content = is_recovery_content(text)
                direct_generators.append({
                    "path": rel,
                    "recovery_path": rec_path,
                    "recovery_content": rec_content,
                    "accepted_as_original": bool(not rec_path and not rec_content),
                    **sig,
                })

        if suffix in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"}:
            archives.append(rel)

    # ------------------------ ZIP archives ------------------------
    for rel in archives:
        ap = root / rel
        if ap.suffix.lower() != ".zip":
            continue
        try:
            with zipfile.ZipFile(ap) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    member = info.filename.replace("\\", "/")
                    suffix = Path(member).suffix.lower()

                    if suffix in CKPT_EXTS:
                        archive_ckpts.append({
                            "archive": rel,
                            "member": member,
                            "size_bytes": info.file_size,
                            "checkpoint_class": checkpoint_class(member),
                        })

                    if suffix in TEXT_EXTS and info.file_size <= max_text:
                        try:
                            text = decode(z.read(info))
                        except Exception:
                            continue
                        sig = generator_signals(text)
                        if sig["strict_generator_evidence"]:
                            rec_path = is_recovery_path(member) or is_recovery_path(rel)
                            rec_content = is_recovery_content(text)
                            archive_generators.append({
                                "archive": rel,
                                "member": member,
                                "recovery_path": rec_path,
                                "recovery_content": rec_content,
                                "accepted_as_original": bool(not rec_path and not rec_content),
                                **sig,
                            })
        except Exception:
            pass

    # ------------------------ TAR-like archives ------------------------
    for rel in archives:
        ap = root / rel
        if ap.suffix.lower() == ".zip":
            continue
        try:
            with tarfile.open(ap, "r:*") as t:
                for m in t.getmembers():
                    if not m.isfile():
                        continue
                    member = m.name.replace("\\", "/")
                    suffix = Path(member).suffix.lower()

                    if suffix in CKPT_EXTS:
                        archive_ckpts.append({
                            "archive": rel,
                            "member": member,
                            "size_bytes": m.size,
                            "checkpoint_class": checkpoint_class(member),
                        })

                    if suffix in TEXT_EXTS and m.size <= max_text:
                        f = t.extractfile(m)
                        if f is None:
                            continue
                        try:
                            text = decode(f.read())
                        except Exception:
                            continue
                        sig = generator_signals(text)
                        if sig["strict_generator_evidence"]:
                            rec_path = is_recovery_path(member) or is_recovery_path(rel)
                            rec_content = is_recovery_content(text)
                            archive_generators.append({
                                "archive": rel,
                                "member": member,
                                "recovery_path": rec_path,
                                "recovery_content": rec_content,
                                "accepted_as_original": bool(not rec_path and not rec_content),
                                **sig,
                            })
        except Exception:
            pass

    # Write CSVs.
    def write_csv(path, rows, columns):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(
        out / "direct_real_checkpoints.csv",
        direct_ckpts,
        ["path", "size_bytes", "checkpoint_class", "sha256"],
    )
    write_csv(
        out / "archive_real_checkpoints.csv",
        archive_ckpts,
        ["archive", "member", "size_bytes", "checkpoint_class"],
    )
    gen_cols = [
        "path", "archive", "member", "recovery_path", "recovery_content",
        "accepted_as_original", "model_load", "tokenizer_load", "pooler_output",
        "embedding_target", "write_signal", "strict_generator_evidence",
    ]
    # normalize rows for common writer
    merged_gens = []
    for r in direct_generators:
        x = {"path": r["path"], "archive": "", "member": "", **r}
        merged_gens.append(x)
    for r in archive_generators:
        x = {"path": "", **r}
        merged_gens.append(x)
    write_csv(out / "strict_embedding_generator_candidates.csv", merged_gens, gen_cols)

    direct_exact_mdsyn = [r for r in direct_ckpts if r["checkpoint_class"] == "exact_expected_mdsyn_best"]
    archive_exact_mdsyn = [r for r in archive_ckpts if r["checkpoint_class"] == "exact_expected_mdsyn_best"]
    direct_exact_enh = [r for r in direct_ckpts if r["checkpoint_class"] == "exact_expected_enhanced_best"]
    archive_exact_enh = [r for r in archive_ckpts if r["checkpoint_class"] == "exact_expected_enhanced_best"]

    accepted_generators = [
        r for r in merged_gens
        if bool(r.get("strict_generator_evidence")) and bool(r.get("accepted_as_original"))
    ]

    summary = {
        "purpose": "strict source/archive-only audit; V8 fixes V7 false positives",
        "model_inference_performed": False,
        "molformer_inference_performed": False,
        "astrazeneca_labels_used_for_selection": False,
        "v7_false_positive_fixes": [
            "Only .pt/.pth/.ckpt files can count as checkpoints; CSV artifacts cannot count as MDSyn checkpoints.",
            "embedding_candidate__restore/audit/recovery copies are explicitly excluded from original generator evidence, including inside audit ZIPs.",
        ],
        "fixed_mdsyn_plus_enhanced_equal_average_rule_found": bool(ensemble_sources),
        "ensemble_rule_source_files": ensemble_sources,
        "exact_expected_mdsyn_checkpoint_path": "result/models/MDSyn_best.pt",
        "exact_expected_mdsyn_checkpoint_found_direct": bool(direct_exact_mdsyn),
        "exact_expected_mdsyn_checkpoint_found_in_archive": bool(archive_exact_mdsyn),
        "exact_expected_enhanced_checkpoint_found_direct": bool(direct_exact_enh),
        "exact_expected_enhanced_checkpoint_found_in_archive": bool(archive_exact_enh),
        "accepted_original_embedding_generator_candidates": [
            (r.get("path") or f"{r.get('archive')}::{r.get('member')}")
            for r in accepted_generators
        ],
        "strict_original_embedding_generator_found": bool(accepted_generators),
    }

    summary["ready_for_final_668_restore"] = bool(
        summary["fixed_mdsyn_plus_enhanced_equal_average_rule_found"]
        and (
            summary["exact_expected_mdsyn_checkpoint_found_direct"]
            or summary["exact_expected_mdsyn_checkpoint_found_in_archive"]
        )
        and (
            summary["exact_expected_enhanced_checkpoint_found_direct"]
            or summary["exact_expected_enhanced_checkpoint_found_in_archive"]
        )
        and summary["strict_original_embedding_generator_found"]
    )

    blocking = []
    if not (
        summary["exact_expected_mdsyn_checkpoint_found_direct"]
        or summary["exact_expected_mdsyn_checkpoint_found_in_archive"]
    ):
        blocking.append("Exact legacy result/models/MDSyn_best.pt was not found.")
    if not summary["strict_original_embedding_generator_found"]:
        blocking.append("No non-recovery original MolFormer embedding generator was found.")
    if not summary["fixed_mdsyn_plus_enhanced_equal_average_rule_found"]:
        blocking.append("Fixed MDSyn + EnhancedMDSyn equal-average rule was not source-confirmed.")
    summary["blocking_reasons"] = blocking

    (out / "audit_v8_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nOutput:", out.resolve())


if __name__ == "__main__":
    main()
