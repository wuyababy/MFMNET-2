#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic-mode reproducibility audit for FULL MFMNet fold 1.

This script runs the exact FULL-model one-epoch audit twice, but first enables:

    torch.use_deterministic_algorithms(True, warn_only=False)

If the current XPU/PyG operator path contains an operation without a supported
deterministic implementation, PyTorch should raise an error. That error itself
is useful evidence that the accelerator path cannot currently satisfy strict
deterministic training under this configuration.

No ablation result is used for tuning or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("mfmnet_det_base", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def hash_state_dict(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode("utf-8"))
        x = v.detach().cpu().contiguous().numpy()
        h.update(str(x.dtype).encode("ascii"))
        h.update(str(tuple(x.shape)).encode("ascii"))
        h.update(x.tobytes())
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-code", default="./MFMNet_reviewer_clean_save_ckpt.py")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output", default="./result/full_deterministic_audit_v4")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--l2", type=float, default=1e-5)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def one_repeat(base, dataset, outer_folds, args, device, repeat_id):
    fold_seed = args.seed + 1000 * args.fold
    base.set_seed(fold_seed)

    train_idx, val_idx, test_idx = base.make_fold_indices(
        dataset.labels, outer_folds, args.fold, args.val_fraction, args.seed
    )

    train_loader = base.make_loader(
        dataset, train_idx, args.batch_size, False, fold_seed + 1, args.num_workers
    )
    val_loader = base.make_loader(
        dataset, val_idx, args.batch_size, False, fold_seed + 2, args.num_workers
    )

    model = base.MFMNetReviewerModel(variant="full", dropout_rate=args.dropout)
    initial_hash = hash_state_dict(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    base.empty_accelerator_cache(device)
    base.sync(device)

    model.train()
    losses = []
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = F.cross_entropy(out, batch.label)
        if args.l2 > 0:
            loss = loss + args.l2 * base.original_l2_penalty(model)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    base.sync(device)
    after_epoch_hash = hash_state_dict(model)

    yv, sv, _, _ = base.predict(model, device, val_loader)
    val_auc = float(roc_auc_score(yv, sv))

    return {
        "repeat": repeat_id,
        "fold": args.fold,
        "fold_seed": fold_seed,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "initial_state_sha256": initial_hash,
        "after_epoch1_state_sha256": after_epoch_hash,
        "epoch1_mean_train_loss": float(np.mean(losses)),
        "epoch1_val_AUROC": val_auc,
    }


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "deterministic_algorithms_requested": True,
        "warn_only": False,
        "status": None,
        "verdict": None,
    }

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)

        model_code = Path(args.model_code).resolve()
        base = load_base(model_code)
        device = base.choose_device(args.device)

        result["device"] = str(device)
        result["torch_version"] = torch.__version__
        result["deterministic_algorithms_enabled"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )

        dataset = base.DrugPairDataset(Path(args.data_root))
        outer_folds = base.make_outer_folds(len(dataset), args.seed, 5)

        rows = []
        for repeat_id in (1, 2):
            print(f"\n=== deterministic FULL fold {args.fold}: repeat {repeat_id}/2 ===")
            row = one_repeat(base, dataset, outer_folds, args, device, repeat_id)
            rows.append(row)
            print(json.dumps(row, indent=2))

        r1, r2 = rows
        result.update({
            "status": "completed",
            "initial_hash_equal": r1["initial_state_sha256"] == r2["initial_state_sha256"],
            "after_epoch1_hash_equal": r1["after_epoch1_state_sha256"] == r2["after_epoch1_state_sha256"],
            "epoch1_loss_abs_diff": abs(
                r1["epoch1_mean_train_loss"] - r2["epoch1_mean_train_loss"]
            ),
            "epoch1_val_AUROC_abs_diff": abs(
                r1["epoch1_val_AUROC"] - r2["epoch1_val_AUROC"]
            ),
            "repeat_1": r1,
            "repeat_2": r2,
        })

        if not result["initial_hash_equal"]:
            result["verdict"] = "INITIALIZATION_MISMATCH"
        elif result["after_epoch1_hash_equal"]:
            result["verdict"] = "DETERMINISTIC_ONE_EPOCH_REPRODUCIBILITY_CONFIRMED"
        else:
            result["verdict"] = "NONDETERMINISM_REMAINS_DESPITE_DETERMINISTIC_MODE"

    except Exception as e:
        result.update({
            "status": "runtime_error",
            "verdict": "DETERMINISTIC_MODE_UNSUPPORTED_OPERATION_DETECTED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        })
        print("\n=== DETERMINISTIC MODE ERROR ===")
        print(result["traceback"])

    (out / "full_deterministic_audit_v4.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== VERDICT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nOutput:", out.resolve())


if __name__ == "__main__":
    main()
