#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final deterministic controlled ablation runner for MFMNet.

Required in the same working directory:
    MFMNet_reviewer_clean_save_ckpt.py
    MFMNet_ablation_controlled_v2.py

Scientific protocol is unchanged from controlled V2/V3:
- identical outer folds and fold-level seeds
- identical training/validation/test protocol
- identical hyperparameters and early stopping
- shared-initialization control
- leave-one-branch-out TGP ablations
- fixed 0.5 classification threshold
- no result-driven seed/hyperparameter/model selection

The only behavioral setting added relative to V2 is:
    torch.use_deterministic_algorithms(True, warn_only=False)

Relative to V3, this V4 changes RECORD-KEEPING ONLY:
- records a deterministic-specific code_version
- writes a determinism_manifest.json into the output directory
It does not change model architecture, initialization, optimizer, data order,
loss, scheduler, early stopping, or evaluation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


CODE_VERSION = "2026-08-15-controlled-ablation-deterministic-v4"


def output_from_argv():
    default = "./result/ablation_controlled_deterministic_v4"
    args = sys.argv[1:]
    for i, x in enumerate(args):
        if x == "--output" and i + 1 < len(args):
            return args[i + 1]
        if x.startswith("--output="):
            return x.split("=", 1)[1]
    return default


def main():
    torch.use_deterministic_algorithms(True, warn_only=False)

    v2_path = Path("./MFMNet_ablation_controlled_v2.py")
    if not v2_path.exists():
        raise FileNotFoundError(
            "Expected ./MFMNet_ablation_controlled_v2.py in the current directory."
        )

    spec = importlib.util.spec_from_file_location(
        "mfmnet_controlled_v2_det_v4", str(v2_path.resolve())
    )
    if spec is None or spec.loader is None:
        raise ImportError(v2_path)

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Record the actual final runner identity in the base runner's environment log.
    mod.base.CODE_VERSION = CODE_VERSION

    outdir = Path(output_from_argv())
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "code_version": CODE_VERSION,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": False,
        "scientific_protocol_changed_relative_to_v3": False,
        "record_keeping_change_relative_to_v3_only": True,
        "result_driven_tuning": False,
    }
    (outdir / "determinism_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Code version:", CODE_VERSION)
    print(
        "Deterministic algorithms enabled:",
        torch.are_deterministic_algorithms_enabled(),
    )
    print("warn_only=False")

    mod.base.main()


if __name__ == "__main__":
    main()
