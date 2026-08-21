#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Controlled ablation runner for MFMNet reviewer revision.

This wrapper REQUIRES the original:
    MFMNet_reviewer_clean_save_ckpt.py
to be present in the same working directory.

Purpose
-------
This version does not tune any hyperparameter to favor the full model. It keeps
the original folds, seeds, optimization budget, inner-validation checkpoint
selection, threshold, data pipeline, and full-model architecture unchanged.

It adds two methodological controls:

1) Shared-initialization control
   For every fold and every ablation, an untrained reference full model is first
   reconstructed with the exact original fold seed. Every parameter/buffer with
   the same name and shape in the ablation is copied from that reference full
   initialization before training. Thus shared GCN/GIN/GAT branches, drug
   encoder, cell-line encoder, classifier, and (where shape-compatible) CSAR
   start from exactly the same initial values as the full model.

2) Leave-one-branch-out TGP ablations
   - without_gcn
   - without_gin
   - without_gat

   The missing graph branch is not instantiated, but its 128-dimensional output
   slot is replaced by zeros before concatenation. Therefore the CSAR retains
   exactly the same 384 -> 192 -> 384 -> 128 architecture as the full model.
   This isolates the marginal contribution of each TGP branch without changing
   the router dimensionality.

Primary variants
----------------
full
without_gcn
without_gin
without_gat
no_router
simple_concatenation

Interpretation rule
-------------------
Run once under the fixed protocol and report the obtained results regardless of
whether the full model ranks first. Do not alter seeds, folds, hyperparameters,
early stopping, or thresholds based on the observed ablation outcomes.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


BASE_SCRIPT = Path("./MFMNet_reviewer_clean_save_ckpt.py")
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        "Expected ./MFMNet_reviewer_clean_save_ckpt.py in the current directory. "
        "Place this controlled runner beside the original reviewer runner."
    )

spec = importlib.util.spec_from_file_location("mfmnet_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["mfmnet_base"] = base
spec.loader.exec_module(base)

OriginalFullModel = base.MFMNetReviewerModel

CONTROLLED_VARIANTS = [
    "full",
    "without_gcn",
    "without_gin",
    "without_gat",
    "no_router",
    "simple_concatenation",
]

CONTROLLED_DISPLAY = {
    "full": "Full MFMNet backbone",
    "without_gcn": "Without GCN branch",
    "without_gin": "Without GIN branch",
    "without_gat": "Without GAT branch",
    "no_router": "Without CSAR (mean fusion)",
    "simple_concatenation": "Simple concatenation",
}

# Mutate the original list object in place because parse_args() captured it as
# the default value when the base module was defined.
base.VARIANTS[:] = CONTROLLED_VARIANTS
base.DISPLAY.clear()
base.DISPLAY.update(CONTROLLED_DISPLAY)
base.CODE_VERSION = "2026-08-15-controlled-ablation-v2"


class ControlledAblationModel(base.nn.Module):
    """
    Same full backbone as the base runner, plus leave-one-branch-out variants.

    For without_gcn / without_gin / without_gat:
      - the removed branch is not instantiated;
      - a zero tensor occupies its canonical 128-d slot;
      - graph_fusion remains AdaptiveFeatureFusion(384, 128), identical in
        architecture to full.
    """

    VALID_VARIANTS = set(CONTROLLED_VARIANTS)

    def __init__(self, variant: str, dropout_rate: float = 0.3, out_channels: int = 2):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown controlled ablation variant: {variant}")

        self.variant = variant
        self.dropout_rate = dropout_rate
        self.relu = base.nn.ReLU()

        use_gcn = variant not in {"without_gcn"}
        use_gin = variant not in {"without_gin"}
        use_gat = variant not in {"without_gat"}

        if use_gcn:
            self.gcn_conv1 = base.GCNConv(base.ATOM_DIM, 512)
            self.gcn_conv2 = base.GCNConv(512, base.GRAPH_DIM)
        else:
            self.gcn_conv1 = None
            self.gcn_conv2 = None

        if use_gin:
            gin_nn1 = base.nn.Sequential(
                base.nn.Linear(base.ATOM_DIM, 512),
                base.nn.ReLU(),
                base.nn.Linear(512, 512),
            )
            gin_nn2 = base.nn.Sequential(
                base.nn.Linear(512, base.GRAPH_DIM),
                base.nn.ReLU(),
                base.nn.Linear(base.GRAPH_DIM, base.GRAPH_DIM),
            )
            self.gin_conv1 = base.GINConv(gin_nn1)
            self.gin_conv2 = base.GINConv(gin_nn2)
            self.gnorm1 = base.GraphNorm(512)
        else:
            self.gin_conv1 = None
            self.gin_conv2 = None
            self.gnorm1 = None

        if use_gat:
            self.gat_conv1 = base.GATv2Conv(
                base.ATOM_DIM, 256, heads=2, dropout=dropout_rate
            )
            self.gat_conv2 = base.GATv2Conv(
                512, base.GRAPH_DIM, heads=1, dropout=dropout_rate
            )
        else:
            self.gat_conv1 = None
            self.gat_conv2 = None

        # Full and leave-one-branch-out variants retain the exact same CSAR
        # dimensionality as the full model.
        if variant in {"full", "without_gcn", "without_gin", "without_gat"}:
            self.graph_fusion = base.AdaptiveFeatureFusion(
                base.GRAPH_DIM * 3, base.GRAPH_DIM
            )
        else:
            self.graph_fusion = None

        if variant == "simple_concatenation":
            self.simple_projection = base.nn.Linear(
                base.GRAPH_DIM * 3, base.GRAPH_DIM
            )
        else:
            self.simple_projection = None

        # These primary graph ablations retain both auxiliary modalities.
        self.use_drug_embedding = True
        self.use_cell_line = True

        self.drug_embedding_processor = base.DrugEmbeddingProcessor(
            base.DRUG_EMBED_DIM, 512, 256
        )
        self.cell_encoder = base.CellLineEncoder(base.CELL_DIM, 512, 256)

        self.classifier = base.nn.Sequential(
            base.nn.Linear(1024, 512),
            base.nn.BatchNorm1d(512),
            base.nn.ReLU(),
            base.nn.Dropout(dropout_rate),
            base.nn.Linear(512, 128),
            base.nn.BatchNorm1d(128),
            base.nn.ReLU(),
            base.nn.Dropout(dropout_rate),
            base.nn.Linear(128, 32),
            base.nn.ReLU(),
            base.nn.Dropout(dropout_rate),
            base.nn.Linear(32, out_channels),
        )

    def _gcn(self, batch):
        h = self.gcn_conv1(batch.x, batch.edge_index)
        h = self.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.gcn_conv2(h, batch.edge_index)

    def _gin(self, batch):
        h = self.gin_conv1(batch.x, batch.edge_index)
        h = self.relu(h)
        h = self.gnorm1(h, batch.batch)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.gin_conv2(h, batch.edge_index)

    def _gat(self, batch):
        h = self.gat_conv1(batch.x, batch.edge_index)
        h = self.relu(h)
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return self.gat_conv2(h, batch.edge_index)

    def _graph_features(self, batch, batch_size: int):
        v = self.variant

        # no_router and simple_concatenation always retain all three branches.
        if v in {"no_router", "simple_concatenation", "full"}:
            gcn = self._gcn(batch)
            gin = self._gin(batch)
            gat = self._gat(batch)

        elif v == "without_gcn":
            gin = self._gin(batch)
            gat = self._gat(batch)
            gcn = torch.zeros_like(gin)

        elif v == "without_gin":
            gcn = self._gcn(batch)
            gat = self._gat(batch)
            gin = torch.zeros_like(gcn)

        elif v == "without_gat":
            gcn = self._gcn(batch)
            gin = self._gin(batch)
            gat = torch.zeros_like(gcn)

        else:
            raise RuntimeError(f"Unsupported variant in _graph_features: {v}")

        if v == "no_router":
            fused = (gcn + gin + gat) / 3.0
        else:
            multi = torch.cat([gcn, gin, gat], dim=1)
            if v == "simple_concatenation":
                fused = self.simple_projection(multi)
            else:
                fused = self.graph_fusion(multi)

        expected_nodes = batch_size * base.MAX_ATOMS
        if fused.shape[0] != expected_nodes:
            raise RuntimeError(
                f"Expected {expected_nodes} nodes "
                f"({batch_size} x {base.MAX_ATOMS}), got {fused.shape[0]}"
            )
        return fused.reshape(batch_size, base.MAX_ATOMS, base.GRAPH_DIM)

    def forward(self, batch):
        batch_size = int(batch.label.shape[0])

        emb1 = self.drug_embedding_processor(batch.left_embedding)
        emb2 = self.drug_embedding_processor(batch.right_embedding)
        cell = self.cell_encoder(batch.cell)

        graph1 = self._graph_features(batch.left_graph, batch_size).mean(dim=1)
        graph2 = self._graph_features(batch.right_graph, batch_size).mean(dim=1)

        combined = torch.cat([emb1, emb2, cell, graph1, graph2], dim=1)
        return self.classifier(combined)


def copy_shared_initialization(reference: base.nn.Module, target: base.nn.Module):
    """
    Copy every same-name, same-shape state entry from the untrained full model.

    This includes trainable parameters and buffers (e.g., BatchNorm running
    statistics) when compatible.
    """
    ref_state = reference.state_dict()
    target_state = target.state_dict()

    copied = []
    target_only = []

    for key, value in target_state.items():
        if key in ref_state and tuple(ref_state[key].shape) == tuple(value.shape):
            target_state[key] = ref_state[key].detach().clone()
            copied.append(key)
        else:
            target_only.append(key)

    target.load_state_dict(target_state, strict=True)

    # Verify exact equality after loading.
    tgt_check = target.state_dict()
    max_abs_diff = 0.0
    for key in copied:
        a = ref_state[key].detach().cpu()
        b = tgt_check[key].detach().cpu()
        if a.numel():
            d = float((a - b).abs().max().item())
            max_abs_diff = max(max_abs_diff, d)

    return {
        "n_reference_state_entries": int(len(ref_state)),
        "n_target_state_entries": int(len(target_state)),
        "n_copied_same_name_same_shape": int(len(copied)),
        "copied_state_entries": copied,
        "target_specific_state_entries": target_only,
        "max_abs_diff_after_copy": float(max_abs_diff),
    }


def instantiate_controlled_model(variant: str, fold_seed: int, dropout: float):
    """
    Reconstruct the original full initialization first.

    For variant == full, the exact original full model object is returned.
    For all ablations, compatible state entries are copied from that same
    reference full initialization.

    RNG state is restored to the post-full-initialization state after constructing
    the ablation so later CPU stochastic operations are not shifted merely by
    variant-specific module construction.
    """
    base.set_seed(fold_seed)

    reference_full = OriginalFullModel(
        variant="full", dropout_rate=dropout
    )

    post_full_torch_rng = torch.get_rng_state().clone()
    post_full_python_rng = random.getstate()
    post_full_numpy_rng = np.random.get_state()

    if variant == "full":
        audit = {
            "mode": "original_full_initialization",
            "n_reference_state_entries": int(len(reference_full.state_dict())),
            "n_target_state_entries": int(len(reference_full.state_dict())),
            "n_copied_same_name_same_shape": int(len(reference_full.state_dict())),
            "copied_state_entries": list(reference_full.state_dict().keys()),
            "target_specific_state_entries": [],
            "max_abs_diff_after_copy": 0.0,
        }
        return reference_full, audit

    model = ControlledAblationModel(
        variant=variant, dropout_rate=dropout
    )
    audit = copy_shared_initialization(reference_full, model)
    audit["mode"] = "copied_from_same_fold_untrained_full_reference"

    # Restore post-full CPU-side RNG state. CUDA/XPU RNG was seeded by set_seed()
    # and neither CPU model construction nor state copying consumes accelerator RNG.
    torch.set_rng_state(post_full_torch_rng)
    random.setstate(post_full_python_rng)
    np.random.set_state(post_full_numpy_rng)

    return model, audit


def controlled_train_variant(
    variant: str,
    fold_number: int,
    dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args,
    outdir: Path,
    device: torch.device,
) -> dict:
    # Same fold-level seed as the original runner.
    fold_seed = args.seed + 1000 * fold_number
    base.set_seed(fold_seed)

    train_loader = base.make_loader(
        dataset, train_idx, args.batch_size, args.shuffle_train,
        fold_seed + 1, args.num_workers
    )
    val_loader = base.make_loader(
        dataset, val_idx, args.batch_size, False,
        fold_seed + 2, args.num_workers
    )
    test_loader = base.make_loader(
        dataset, test_idx, args.batch_size, False,
        fold_seed + 3, args.num_workers
    )

    # Rebuild the original full reference under the exact fold seed, then create
    # the requested controlled ablation.
    model, init_audit = instantiate_controlled_model(
        variant=variant,
        fold_seed=fold_seed,
        dropout=args.dropout,
    )

    init_audit.update({
        "variant": variant,
        "display_name": CONTROLLED_DISPLAY[variant],
        "fold": int(fold_number),
        "fold_seed": int(fold_seed),
        "shared_initialization_control": True,
        "full_architecture_or_hyperparameters_tuned_to_ablation_results": False,
    })
    with open(
        outdir / f"initialization_audit_{variant}_fold{fold_number}.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(init_audit, f, ensure_ascii=False, indent=2)

    print(
        f"[init audit] {variant} fold {fold_number}: "
        f"copied={init_audit['n_copied_same_name_same_shape']} "
        f"target_specific={len(init_audit['target_specific_state_entries'])} "
        f"max_abs_diff={init_audit['max_abs_diff_after_copy']:.3g}"
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    n_params = base.parameter_count(model)
    theoretical_mb = base.theoretical_weight_size_mb(model)
    best_val_auc = -np.inf
    best_epoch = 0
    best_state = None
    patience_counter = 0
    epoch_rows = []
    epoch_seconds = []

    base.empty_accelerator_cache(device)
    base.reset_peak_memory(device)
    base.sync(device)
    wall_start = time.perf_counter()

    for epoch in range(1, args.max_epochs + 1):
        base.sync(device)
        epoch_start = time.perf_counter()
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

        mean_train_loss = float(np.mean(losses))
        scheduler.step(mean_train_loss)

        yv, sv, _, _ = base.predict(model, device, val_loader)
        val_auc = float(roc_auc_score(yv, sv))

        base.sync(device)
        epoch_sec = float(time.perf_counter() - epoch_start)
        epoch_seconds.append(epoch_sec)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_rows.append({
            "variant": variant,
            "fold": fold_number,
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_AUROC": val_auc,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_sec,
        })

        if val_auc > best_val_auc + args.min_delta:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or patience_counter >= args.early_stopping_patience
        ):
            print(
                f"[{variant} fold {fold_number}] epoch={epoch} "
                f"loss={mean_train_loss:.5f} "
                f"val_AUROC={val_auc:.5f} "
                f"best={best_val_auc:.5f}@{best_epoch} "
                f"patience={patience_counter}"
            )

        if patience_counter >= args.early_stopping_patience:
            break

    base.sync(device)
    training_wall_seconds = float(time.perf_counter() - wall_start)
    training_peak_mb = base.max_memory_allocated_mb(device)

    if best_state is None:
        raise RuntimeError("Training ended without a best checkpoint")

    model.load_state_dict(best_state)

    # Save full checkpoint only, matching the original runner behavior.
    if variant == "full":
        ckpt_dir = outdir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"mfmnet_full_fold{fold_number}.pt"
        state_dict_cpu = {
            k: v.detach().cpu() for k, v in best_state.items()
        }
        torch.save(
            {
                "state_dict": state_dict_cpu,
                "variant": variant,
                "fold": int(fold_number),
                "best_epoch": int(best_epoch),
                "best_val_AUROC": float(best_val_auc),
                "seed": int(fold_seed),
                "controlled_ablation_runner": True,
            },
            ckpt_path,
        )
        print(f"Saved controlled-run full checkpoint: {ckpt_path.resolve()}")

    yt, st, pt, it = base.predict(model, device, test_loader)
    m = base.metric_dict(yt, st, pt)

    bench = base.benchmark_inference(
        model, device, test_loader,
        len(test_idx), args.inference_repeats
    )

    tmp_ckpt = outdir / f"_tmp_{variant}_fold{fold_number}.pt"
    checkpoint_mb = base.save_checkpoint_size_mb(model, tmp_ckpt)
    timing = base.training_timing_summary(
        epoch_seconds, training_wall_seconds
    )

    if variant == "full" and not args.no_scalability:
        scalability_df = base.benchmark_scalability(
            model=model,
            device=device,
            dataset=dataset,
            test_idx=test_idx,
            batch_sizes=args.scalability_batch_sizes,
            repeats=args.inference_repeats,
            seed=fold_seed + 4,
            num_workers=args.num_workers,
        )
        scalability_df.insert(0, "fold", fold_number)
        scalability_df.to_csv(
            outdir / f"scalability_full_fold{fold_number}.csv",
            index=False
        )

    pd.DataFrame({
        "sample_index": it.astype(int),
        "label": yt.astype(int),
        "score": st,
        "predicted_label": pt.astype(int),
    }).sort_values("sample_index").to_csv(
        outdir / f"pred_{variant}_fold{fold_number}.csv",
        index=False
    )

    pd.DataFrame(epoch_rows).to_csv(
        outdir / f"learning_{variant}_fold{fold_number}.csv",
        index=False
    )

    row = {
        "variant": variant,
        "display_name": CONTROLLED_DISPLAY[variant],
        "fold": fold_number,
        **m,
        "best_epoch": int(best_epoch),
        "best_val_AUROC": float(best_val_auc),
        "trainable_parameters": int(n_params),
        "theoretical_trainable_weight_size_mb": theoretical_mb,
        "checkpoint_size_mb": checkpoint_mb,
        "training_wall_seconds": training_wall_seconds,
        "training_wall_minutes": training_wall_seconds / 60.0,
        **timing,
        "training_active_minutes": timing["training_active_seconds"] / 60.0,
        "training_peak_gpu_memory_mb": training_peak_mb,
        **bench,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "seed": int(fold_seed),
        "shared_initialization_control": True,
        "initialization_reference": "same-fold untrained full model",
    }

    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row


# Replace only the model class and training function used by base.main().
base.MFMNetReviewerModel = ControlledAblationModel
base.train_variant = controlled_train_variant


if __name__ == "__main__":
    base.main()
