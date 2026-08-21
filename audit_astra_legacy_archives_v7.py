#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7 source/archive-only audit for the legacy MFMNet AstraZeneca claim.

This script performs NO model inference, NO MolFormer inference, and reads NO
AstraZeneca labels for selection. It searches the local legacy project and
archives/backups for two missing provenance artifacts:

1) an original MDSyn checkpoint needed by the fixed MDSyn + EnhancedMDSyn
   equal-weight ensemble defined in enhanced_main.py;
2) an original MolFormer embedding-generation source path that actually loads
   MolFormer, obtains pooler_output, and writes/constructs all_drug_embedding.

It also avoids the V6 false-positive rule that treated any file merely
mentioning all_drug_embedding/MolFormer as proof of an embedding generator.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

TEXT_EXTS = {
    ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".ipynb",
    ".sh", ".bat", ".ps1", ".toml", ".ini", ".cfg"
}
CKPT_EXTS = {".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle", ".model"}
ARCHIVE_SUFFIXES = (
    ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"
)

# Files created during the present recovery/reviewer workflow are still reported,
# but are not accepted as evidence of the original legacy embedding generator.
RECOVERY_NAME_PATTERNS = [
    r"(^|/)audit_", r"(^|/)restore_legacy_", r"(^|/)recover_",
    r"(^|/)extract_", r"reviewer", r"run_nci", r"run_xgboost",
    r"external_no_drug_embedding", r"same5fold"
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--codes-root", default=".")
    p.add_argument("--output", default="./result/astra_legacy_archives_audit_v7")
    p.add_argument("--max-text-mb", type=float, default=12.0,
                   help="Maximum text/archive-member size to inspect as text")
    return p.parse_args()


def norm_rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def is_recovery_path(rel: str) -> bool:
    s = rel.replace("\\", "/")
    return any(re.search(p, s, flags=re.I) for p in RECOVERY_NAME_PATTERNS)


def decode_text(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def generator_signals(text: str):
    """Strict evidence: model load + pooler output + embedding save/construction."""
    low = text.lower()
    model_load = bool(
        re.search(r"automodel\s*\.\s*from_pretrained", text, re.I)
        or re.search(r"automodel\.from_pretrained", text, re.I)
        or "molformer-xl" in low
        or "ibm-research/molformer" in low
    )
    tokenizer_load = bool(
        re.search(r"autotokenizer\s*\.\s*from_pretrained", text, re.I)
        or re.search(r"autotokenizer\.from_pretrained", text, re.I)
    )
    pooler = "pooler_output" in low
    embedding_target = bool(
        "all_drug_embedding" in low
        or re.search(r"molformer[_ -]?embedding", text, re.I)
    )
    write_signal = bool(
        re.search(r"\.to_csv\s*\(", text, re.I)
        or re.search(r"csv\.writer|writerow|np\.save|torch\.save", text, re.I)
    )
    strict = model_load and pooler and embedding_target and write_signal
    return {
        "model_load": model_load,
        "tokenizer_load": tokenizer_load,
        "pooler_output": pooler,
        "embedding_target": embedding_target,
        "write_signal": write_signal,
        "strict_generator_evidence": strict,
    }


def checkpoint_kind(name: str):
    low = name.lower().replace("\\", "/")
    base = Path(low).name
    if "enhanced" in base and "best" in base:
        return "enhanced_best"
    # Do not let EnhancedMDSyn count as MDSyn.
    if "mdsyn" in base and "enhanced" not in base:
        return "mdsyn"
    if re.search(r"(^|[_\-.])md[_\-]?syn([_\-.]|$)", base) and "enhanced" not in base:
        return "mdsyn"
    return "other"


def text_context(text: str, patterns, radius=3):
    lines = text.splitlines()
    hit_idx = []
    for i, line in enumerate(lines):
        if any(re.search(p, line, re.I) for p in patterns):
            hit_idx.append(i)
    keep = set()
    for i in hit_idx:
        keep.update(range(max(0, i-radius), min(len(lines), i+radius+1)))
    out = []
    prev = None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            out.append("... omitted ...")
        out.append(f"{i+1:05d}: {lines[i]}")
        prev = i
    return "\n".join(out)


def scan_direct(root: Path, max_text_bytes: int):
    ckpts = []
    text_rows = []
    strict_generators = []
    ensemble_contexts = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = norm_rel(p, root)
        suffix = p.suffix.lower()

        if suffix in CKPT_EXTS or checkpoint_kind(rel) != "other":
            ckpts.append({
                "location": "direct",
                "path": rel,
                "size_bytes": p.stat().st_size,
                "checkpoint_kind": checkpoint_kind(rel),
            })

        if suffix not in TEXT_EXTS or p.stat().st_size > max_text_bytes:
            continue
        try:
            text = decode_text(p.read_bytes())
        except Exception:
            continue

        sig = generator_signals(text)
        if any(sig.values()):
            row = {"location": "direct", "path": rel,
                   "recovery_or_reviewer_path": is_recovery_path(rel), **sig}
            text_rows.append(row)
            if sig["strict_generator_evidence"] and not is_recovery_path(rel):
                strict_generators.append(rel)

        if Path(rel).name.lower() == "enhanced_main.py":
            ctx = text_context(text, [
                r"model_classes", r"\bMDSyn\b", r"EnhancedMDSyn",
                r"weights\s*=\s*\[1\.0\s*/\s*len\(models\)\]",
                r"ensemble_preds", r">\s*0\.5"
            ], radius=5)
            ensemble_contexts.append((rel, ctx, text))

    return ckpts, text_rows, strict_generators, ensemble_contexts


def scan_zip(path: Path, root: Path, max_text_bytes: int):
    rows = []
    gen_rows = []
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = info.filename.replace("\\", "/")
                kind = checkpoint_kind(name)
                if Path(name).suffix.lower() in CKPT_EXTS or kind != "other":
                    rows.append({
                        "archive": norm_rel(path, root), "member": name,
                        "size_bytes": info.file_size, "checkpoint_kind": kind
                    })
                if Path(name).suffix.lower() in TEXT_EXTS and info.file_size <= max_text_bytes:
                    try:
                        text = decode_text(z.read(info))
                    except Exception:
                        continue
                    sig = generator_signals(text)
                    if sig["strict_generator_evidence"]:
                        gen_rows.append({
                            "archive": norm_rel(path, root), "member": name,
                            "recovery_or_reviewer_path": is_recovery_path(name), **sig
                        })
    except Exception as e:
        rows.append({"archive": norm_rel(path, root), "member": "<ERROR>", "error": repr(e)})
    return rows, gen_rows


def scan_tar(path: Path, root: Path, max_text_bytes: int):
    rows = []
    gen_rows = []
    try:
        with tarfile.open(path, "r:*") as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                name = m.name.replace("\\", "/")
                kind = checkpoint_kind(name)
                if Path(name).suffix.lower() in CKPT_EXTS or kind != "other":
                    rows.append({
                        "archive": norm_rel(path, root), "member": name,
                        "size_bytes": m.size, "checkpoint_kind": kind
                    })
                if Path(name).suffix.lower() in TEXT_EXTS and m.size <= max_text_bytes:
                    try:
                        fh = t.extractfile(m)
                        if fh is None:
                            continue
                        text = decode_text(fh.read())
                    except Exception:
                        continue
                    sig = generator_signals(text)
                    if sig["strict_generator_evidence"]:
                        gen_rows.append({
                            "archive": norm_rel(path, root), "member": name,
                            "recovery_or_reviewer_path": is_recovery_path(name), **sig
                        })
    except Exception as e:
        rows.append({"archive": norm_rel(path, root), "member": "<ERROR>", "error": repr(e)})
    return rows, gen_rows


def archive_type(path: Path):
    low = path.name.lower()
    if low.endswith(".zip"):
        return "zip"
    if any(low.endswith(x) for x in ARCHIVE_SUFFIXES[1:]):
        return "tar"
    return None


def main():
    args = parse_args()
    root = Path(args.codes_root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    max_text_bytes = int(args.max_text_mb * 1024 * 1024)

    direct_ckpts, direct_text, strict_direct_generators, ensemble_contexts = scan_direct(
        root, max_text_bytes
    )

    archive_ckpts = []
    archive_generators = []
    archive_inventory = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        typ = archive_type(p)
        if not typ:
            continue
        archive_inventory.append({
            "path": norm_rel(p, root), "size_bytes": p.stat().st_size, "type": typ
        })
        if typ == "zip":
            c, g = scan_zip(p, root, max_text_bytes)
        else:
            c, g = scan_tar(p, root, max_text_bytes)
        archive_ckpts.extend(c)
        archive_generators.extend(g)

    direct_ckpt_df = pd.DataFrame(direct_ckpts)
    archive_ckpt_df = pd.DataFrame(archive_ckpts)
    direct_text_df = pd.DataFrame(direct_text)
    archive_gen_df = pd.DataFrame(archive_generators)
    archive_inv_df = pd.DataFrame(archive_inventory)

    direct_ckpt_df.to_csv(out / "direct_checkpoint_candidates.csv", index=False)
    archive_ckpt_df.to_csv(out / "archive_checkpoint_candidates.csv", index=False)
    direct_text_df.to_csv(out / "direct_embedding_generator_signal_files.csv", index=False)
    archive_gen_df.to_csv(out / "archive_strict_embedding_generator_candidates.csv", index=False)
    archive_inv_df.to_csv(out / "archive_inventory.csv", index=False)

    ensemble_rule_found = False
    ensemble_files = []
    for rel, ctx, text in ensemble_contexts:
        (out / ("ensemble_context__" + re.sub(r"[^A-Za-z0-9_.-]+", "_", rel) + ".txt")).write_text(
            ctx, encoding="utf-8"
        )
        has_models = bool(re.search(r"model_classes\s*=\s*\[", text, re.I)
                          and re.search(r"\bMDSyn\b", text)
                          and re.search(r"EnhancedMDSyn", text))
        has_equal = bool(re.search(r"weights\s*=\s*\[1\.0\s*/\s*len\(models\)\]", text))
        has_threshold = bool(re.search(r"ensemble_prelabels\s*=\s*\(ensemble_preds\s*>\s*0\.5\)", text))
        if has_models and has_equal and has_threshold:
            ensemble_rule_found = True
            ensemble_files.append(rel)

    def direct_has(kind):
        return bool(len(direct_ckpt_df) and "checkpoint_kind" in direct_ckpt_df.columns
                    and (direct_ckpt_df["checkpoint_kind"] == kind).any())

    def archive_has(kind):
        return bool(len(archive_ckpt_df) and "checkpoint_kind" in archive_ckpt_df.columns
                    and (archive_ckpt_df["checkpoint_kind"] == kind).any())

    strict_archive_generators = []
    if len(archive_gen_df):
        for _, r in archive_gen_df.iterrows():
            if not bool(r.get("recovery_or_reviewer_path", False)):
                strict_archive_generators.append(f"{r['archive']}::{r['member']}")

    all_strict_generators = strict_direct_generators + strict_archive_generators

    summary = {
        "purpose": "source/archive-only legacy AstraZeneca provenance audit",
        "model_inference_performed": False,
        "molformer_inference_performed": False,
        "astrazeneca_labels_used_for_selection": False,
        "v6_embedding_flag_correction": (
            "V6 safe_to_freeze_embedding_rng_batch_path was not a valid proof; "
            "V7 requires strict original generator evidence: model load + pooler_output + "
            "embedding target + write signal, excluding recovery/reviewer scripts."
        ),
        "fixed_mdsyn_plus_enhanced_equal_average_rule_found": ensemble_rule_found,
        "ensemble_rule_source_files": ensemble_files,
        "direct_mdsyn_checkpoint_found": direct_has("mdsyn"),
        "archive_mdsyn_checkpoint_found": archive_has("mdsyn"),
        "direct_enhanced_best_checkpoint_found": direct_has("enhanced_best"),
        "archive_enhanced_best_checkpoint_found": archive_has("enhanced_best"),
        "strict_original_embedding_generator_candidates": all_strict_generators,
        "strict_original_embedding_generator_found": bool(all_strict_generators),
        "safe_to_reconstruct_fixed_legacy_ensemble": bool(
            ensemble_rule_found
            and (direct_has("mdsyn") or archive_has("mdsyn"))
            and (direct_has("enhanced_best") or archive_has("enhanced_best"))
        ),
        "safe_to_freeze_embedding_rng_batch_from_source": bool(all_strict_generators),
    }
    summary["ready_for_final_668_restore"] = bool(
        summary["safe_to_reconstruct_fixed_legacy_ensemble"]
        and summary["safe_to_freeze_embedding_rng_batch_from_source"]
    )

    reasons = []
    if not summary["strict_original_embedding_generator_found"]:
        reasons.append("No strict original MolFormer embedding-generator source was found.")
    if not (summary["direct_mdsyn_checkpoint_found"] or summary["archive_mdsyn_checkpoint_found"]):
        reasons.append("No legacy non-Enhanced MDSyn checkpoint was found.")
    if not ensemble_rule_found:
        reasons.append("The fixed MDSyn + EnhancedMDSyn equal-average rule was not fully source-confirmed.")
    summary["blocking_reasons"] = reasons
    summary["next_step"] = (
        "If ready_for_final_668_restore is false, do not rerun AstraZeneca with alternate seeds, "
        "checkpoints, mappings, weights, or thresholds. Preserve the unresolved provenance status."
    )

    (out / "audit_v7_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nOutput:", out)


if __name__ == "__main__":
    main()
