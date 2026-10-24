"""
dump_initial_predictions.py - generate per-sample test-fold predictions
from released checkpoints

For each outer fold N, loads checkpoints/mfmnet_full_fold{N}.pt, runs inference
on the fold's outer-test samples, and writes one CSV per fold:

    pred_full_fold{N}.csv  (columns: sample_index, label, score, predicted_label)

Conventions are identical to the training script MFMNet_reviewer_clean_save_ckpt.py:
  - fold assignment = outer_fold_assignments.csv (outer_fold == N is the test set);
  - probability     = softmax(logits)[:, 1];
  - data pipeline   = DrugPairDataset + make_loader (same processed graphs).

Usage
-----
python dump_initial_predictions.py \
    --codes-dir . --ckpt-dir checkpoints \
    --outdir results/predictions/mfmnet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--codes-dir", required=True, help="codes 根目录（含 data/ 与训练脚本）")
    ap.add_argument("--ckpt-dir", required=True, help="checkpoint 目录（含 mfmnet_full_fold{1..5}.pt）")
    ap.add_argument("--assignments", default=None,
                    help="outer_fold_assignments.csv 路径；默认 <codes>/result/reviewer_clean/ 下那份")
    ap.add_argument("--variant", default="full", help="模型变体（full / no_drug_embedding / ...）")
    ap.add_argument("--ckpt-prefix", default=None,
                    help="checkpoint 文件名前缀；默认 full→mfmnet_full_fold，其余→mfmnet_<variant>_fold")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu / cuda / xpu / auto")
    ap.add_argument("--folds", default="1,2,3,4,5", help="要补的折，如 '1' 或 '1,3,5'")
    ap.add_argument("--outdir", default="./initial_run_predictions")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.codes_dir).resolve()))
    import MFMNet_reviewer_clean_save_ckpt as M  # noqa: PLC0415

    codes = Path(args.codes_dir)
    asg_path = Path(args.assignments) if args.assignments else \
        codes / "result" / "reviewer_clean" / "outer_fold_assignments.csv"
    asg = pd.read_csv(asg_path)
    assert {"sample_index", "outer_fold", "label"} <= set(asg.columns), asg.columns.tolist()

    prefix = args.ckpt_prefix or (
        "mfmnet_full_fold" if args.variant == "full" else f"mfmnet_{args.variant}_fold")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = M.DrugPairDataset(codes / "data")
    device = M.choose_device(args.device)
    print(f"[setup] device={device}, dataset={len(dataset)} samples, assignments={asg_path.name}")

    folds = [int(x) for x in str(args.folds).split(",") if x.strip()]
    summary_rows = []
    for fold in folds:
        ckpt_path = Path(args.ckpt_dir) / f"{prefix}{fold}.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"[错误] 找不到 {ckpt_path}")

        sub = asg[asg.outer_fold == fold].sort_values("sample_index")
        test_idx = sub.sample_index.to_numpy(dtype=int)

        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        torch.manual_seed(0)
        model = M.MFMNetReviewerModel(variant=args.variant)
        model.load_state_dict(state)
        model.to(device)

        loader = M.make_loader(dataset, test_idx,
                               batch_size=args.batch_size, shuffle=False,
                               seed=fold, num_workers=args.num_workers)
        y, score, pred, idx = M.predict(model, device, loader)

        # 硬校验：样本集合与标签必须与折归属表一致
        order = np.argsort(idx)
        idx, y, score, pred = idx[order], y[order], score[order], pred[order]
        assert np.array_equal(idx, test_idx), f"fold{fold} 样本集合与归属表不一致"
        assert np.array_equal(y, sub.label.to_numpy()), f"fold{fold} 标签与归属表不一致"

        auroc = roc_auc_score(y, score)
        prauc = average_precision_score(y, score)
        out_csv = outdir / f"pred_{args.variant}_fold{fold}.csv"
        pd.DataFrame({"sample_index": idx, "label": y,
                      "score": score, "pred": pred}).to_csv(out_csv, index=False)
        print(f"[fold{fold}] AUROC={auroc:.4f}  PR-AUC={prauc:.4f}  n={len(y)}  -> {out_csv.name}")
        if isinstance(ck, dict) and "best_val_AUROC" in ck:
            print(f"          checkpoint 元数据: seed={ck.get('seed')} "
                  f"best_epoch={ck.get('best_epoch')} val_AUROC={ck.get('best_val_AUROC'):.4f}")
        summary_rows.append({"fold": fold, "AUROC": auroc, "PR-AUC": prauc, "n": len(y)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / f"summary_{args.variant}.csv", index=False)
    print("\n=== 逐折汇总 ===")
    print(summary.to_string(index=False))
    print(f"AUROC 均值±SD = {summary['AUROC'].mean():.4f} ± {summary['AUROC'].std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
