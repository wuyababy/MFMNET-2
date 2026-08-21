#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic controlled ablation runner for MFMNet.

Requirements in the same working directory:
    MFMNet_reviewer_clean_save_ckpt.py
    MFMNet_ablation_controlled_v2.py

This wrapper changes only the numerical determinism setting:

    torch.use_deterministic_algorithms(True, warn_only=False)

All folds, seeds, hyperparameters, model definitions, shared-initialization
controls, early stopping rules, validation-based checkpoint selection, and the
0.5 classification threshold remain those defined in
MFMNet_ablation_controlled_v2.py and the original reviewer runner.

No seed, hyperparameter, checkpoint, or ablation definition is selected based
on observed test performance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def main():
    torch.use_deterministic_algorithms(True, warn_only=False)

    v2_path = Path("./MFMNet_ablation_controlled_v2.py")
    if not v2_path.exists():
        raise FileNotFoundError(
            "Expected ./MFMNet_ablation_controlled_v2.py in the current directory."
        )

    spec = importlib.util.spec_from_file_location(
        "mfmnet_controlled_v2_det", str(v2_path.resolve())
    )
    if spec is None or spec.loader is None:
        raise ImportError(v2_path)

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    print("Deterministic algorithms enabled:",
          torch.are_deterministic_algorithms_enabled())
    print("warn_only=False")

    # V2 replaces the base runner's train_variant/model definitions during import.
    mod.base.main()


if __name__ == "__main__":
    main()
