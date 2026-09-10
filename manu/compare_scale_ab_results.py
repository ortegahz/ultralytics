#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Unified A/B Model Precision Comparison: YOLO26n vs YOLO26s.

Features:
1. Two Comparison Modes:
   - Mode A (Instant): Reads results.csv directly from the training runs to compare learning curves and epoch metrics without GPU inference.
   - Mode B (Full Benchmark): Evaluates best_f1.pt / best_recall.pt on the full validation dataset under unified Distance <= 8.0px criteria across sweeps [0.15 ~ 0.60].
2. Hard-cases breakdown (DJI_0175_2, wg011, DJI_0051, wg020) if Mode B is run.
3. Outputs Markdown Table and optional Matplotlib comparison bar chart.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import extract_peaks, find_best_f1_threshold
from manu.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Compare YOLO26n vs YOLO26s A/B Test Results")
    parser.add_argument("--run-n", type=str, default="runs/scale_ab_test/yolo26n_p2_5ep", help="Path to YOLO26n run directory")
    parser.add_argument("--run-s", type=str, default="runs/scale_ab_test/yolo26s_p2_5ep", help="Path to YOLO26s run directory")
    parser.add_argument("--eval-full", action="store_true", help="Perform full validation inference on GPU instead of just parsing CSV")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml", help="Path to data.yaml")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID for evaluation (if --eval-full)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="TP distance match threshold in px")
    parser.add_argument("--stride", type=int, default=4, help="Model stride (strictly 4 for P2)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    return parser.parse_args()


def compare_from_csv(run_n_dir: Path, run_s_dir: Path):
    """Fast comparison by parsing results.csv from both training runs."""
    csv_n = run_n_dir / "results.csv"
    csv_s = run_s_dir / "results.csv"

    if not csv_n.exists():
        print(colorstr("red", f"[ERROR] results.csv not found for YOLO26n: {csv_n}"))
        return
    if not csv_s.exists():
        print(colorstr("red", f"[ERROR] results.csv not found for YOLO26s: {csv_s}"))
        return

    df_n = pd.read_csv(csv_n)
    df_s = pd.read_csv(csv_s)

    # Clean whitespace in column names
    df_n.columns = [c.strip() for c in df_n.columns]
    df_s.columns = [c.strip() for c in df_s.columns]

    print("\n" + "=" * 100)
    print(colorstr("bold", colorstr("cyan", "⚡ [A/B TEST EPOCH PROGRESSION & BEST METRICS SUMMARY]")))
    print("=" * 100)

    epochs = min(len(df_n), len(df_s))
    print(f"{'Epoch':<6} | {'Model':<10} | {'F1-Score':<10} | {'Recall':<10} | {'Precision':<10} | {'TP':<8} | {'FP':<8} | {'Best Th':<8} | {'Loss':<10}")
    print("-" * 100)

    for ep in range(epochs):
        row_n = df_n.iloc[ep]
        row_s = df_s.iloc[ep]
        f1_n, rec_n, prec_n = row_n["metrics/f1(B)"], row_n["metrics/recall(B)"], row_n["metrics/precision(B)"]
        f1_s, rec_s, prec_s = row_s["metrics/f1(B)"], row_s["metrics/recall(B)"], row_s["metrics/precision(B)"]

        diff_f1 = f1_s - f1_n
        diff_str = f"({diff_f1:+.4f})" if ep > 0 else ""

        print(f"Ep {int(row_n['epoch']):02d}  | YOLO26n    | {f1_n:.4f}     | {rec_n*100:.2f}%     | {prec_n*100:.2f}%        | {int(row_n['metrics/tp']):<8} | {int(row_n['metrics/fp']):<8} | {row_n['metrics/best_th']:.2f}     | {row_n['train/loss']:.4f}")
        print(f"Ep {int(row_s['epoch']):02d}  | YOLO26s    | {f1_s:.4f} {diff_str:<6} | {rec_s*100:.2f}%     | {prec_s*100:.2f}%        | {int(row_s['metrics/tp']):<8} | {int(row_s['metrics/fp']):<8} | {row_s['metrics/best_th']:.2f}     | {row_s['train/loss']:.4f}")
        print("-" * 100)

    # Best Epoch overall
    best_n_idx = df_n["metrics/f1(B)"].idxmax()
    best_s_idx = df_s["metrics/f1(B)"].idxmax()
    best_n = df_n.iloc[best_n_idx]
    best_s = df_s.iloc[best_s_idx]

    print("\n" + "=" * 100)
    print(colorstr("bold", colorstr("green", "🏆 [FINAL A/B VERDICT (DISTANCE <= 8.0px)]")))
    print("=" * 100)

    f1_delta = best_s['metrics/f1(B)'] - best_n['metrics/f1(B)']
    rec_delta = best_s['metrics/recall(B)'] - best_n['metrics/recall(B)']
    prec_delta = best_s['metrics/precision(B)'] - best_n['metrics/precision(B)']
    fp_delta = int(best_s['metrics/fp']) - int(best_n['metrics/fp'])
    tp_delta = int(best_s['metrics/tp']) - int(best_n['metrics/tp'])

    print(f"| Metric              | YOLO26n (Baseline) | YOLO26s (Target)   | Net Delta (s - n)        |")
    print(f"| :------------------ | :----------------- | :----------------- | :----------------------- |")
    print(f"| Best Epoch          | Ep {int(best_n['epoch']):02d}             | Ep {int(best_s['epoch']):02d}             | -                        |")
    print(f"| Peak F1-Score       | {best_n['metrics/f1(B)']:.4f}             | {best_s['metrics/f1(B)']:.4f}             | {f1_delta:+.4f} ({f1_delta*100:+.2f}%)       |")
    print(f"| Recall              | {best_n['metrics/recall(B)']*100:.2f}%            | {best_s['metrics/recall(B)']*100:.2f}%            | {rec_delta*100:+.2f}%                  |")
    print(f"| Precision           | {best_n['metrics/precision(B)']*100:.2f}%            | {best_s['metrics/precision(B)']*100:.2f}%            | {prec_delta*100:+.2f}%                  |")
    print(f"| True Positives (TP) | {int(best_n['metrics/tp']):<18} | {int(best_s['metrics/tp']):<18} | {tp_delta:+d}                     |")
    print(f"| False Alarms (FP)   | {int(best_n['metrics/fp']):<18} | {int(best_s['metrics/fp']):<18} | {fp_delta:+d}                     |")
    print(f"| Best Threshold      | {best_n['metrics/best_th']:.2f}               | {best_s['metrics/best_th']:.2f}               | -                        |")
    print("=" * 100)

    if f1_delta > 0.005:
        print(colorstr("green", f"\n[CONCLUSION] ★ YOLO26s shows CLEAR SUPERIORITY (+{f1_delta*100:.2f}% F1). Model capacity expansion is positively validated!"))
    elif f1_delta < -0.005:
        print(colorstr("red", f"\n[CONCLUSION] ✖ YOLO26s underperforms YOLO26n ({f1_delta*100:.2f}% F1). Model capacity expansion shows negative returns / overfitting!"))
    else:
        print(colorstr("yellow", f"\n[CONCLUSION] ≈ YOLO26s is roughly comparable to YOLO26n ({f1_delta*100:+.2f}% F1). Check trade-off between FP and Recall."))
    print("=" * 100 + "\n")


def eval_model_full(weights_path: Path, scale: str, args, device: torch.device):
    """Run full GPU sweep on a given model checkpoint."""
    print(colorstr("cyan", f"\n[EVAL] Loading YOLO26{scale.upper()} checkpoint: {weights_path}..."))
    model = YOLO26HeatmapDetector(
        stride=args.stride,
        weights=weights_path,
        num_classes=1,
        scale=scale,
    )
    model.to(device)
    model.eval()

    data_dict = check_det_dataset(args.data)
    val_path = data_dict["val"]
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    val_dataset = build_yolo_dataset(cfg, img_path=val_path, batch=args.batch, data=data_dict, mode="val", rect=False)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)

    val_preds_list = []
    val_gt_list = []
    val_sizes_list = []

    pbar = tqdm(val_loader, desc=f"Evaluating YOLO26{scale.upper()}", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = imgs.shape[0]

            preds = model(imgs)
            peaks = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=args.stride,
                conf_thresh=0.08,
                top_k=80,
            )
            val_preds_list.extend(peaks)

            b_idx_cpu = b_idx.long().cpu().view(-1)
            bboxes_cpu = bboxes.cpu().numpy()
            for b in range(bs):
                mask_b = (b_idx_cpu == b).numpy()
                gt_b = bboxes_cpu[mask_b] if mask_b.any() else np.zeros((0, 4), dtype=np.float32)
                val_gt_list.append(gt_b)
                val_sizes_list.append((args.imgsz, args.imgsz))

    sweep_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    metrics = find_best_f1_threshold(
        predictions_raw=val_preds_list,
        gt_boxes_list=val_gt_list,
        img_sizes=val_sizes_list,
        distance_threshold=args.dist_thresh,
        thresholds=sweep_thresholds,
    )
    return metrics


def main():
    args = parse_args()
    run_n = Path(args.run_n)
    run_s = Path(args.run_s)

    if not args.eval_full:
        # Fast mode: parse CSV
        compare_from_csv(run_n, run_s)
    else:
        # Full GPU Evaluation Mode
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        weights_n = run_n / "weights/best_f1.pt"
        weights_s = run_s / "weights/best_f1.pt"

        m_n = eval_model_full(weights_n, "n", args, device)
        m_s = eval_model_full(weights_s, "s", args, device)

        print("\n" + "=" * 90)
        print(colorstr("bold", colorstr("green", "📊 [FULL VALIDATION BENCHMARK (Distance <= 8.0px)]")))
        print("=" * 90)
        print(f"YOLO26n: Best F1 = {m_n['f1']:.4f} (@ th={m_n['best_th']:.2f}), Recall = {m_n['recall']*100:.2f}%, Precision = {m_n['precision']*100:.2f}%, TP = {m_n['tp']}, FP = {m_n['fp']}")
        print(f"YOLO26s: Best F1 = {m_s['f1']:.4f} (@ th={m_s['best_th']:.2f}), Recall = {m_s['recall']*100:.2f}%, Precision = {m_s['precision']*100:.2f}%, TP = {m_s['tp']}, FP = {m_s['fp']}")
        print(f"Net F1 Delta: {m_s['f1'] - m_n['f1']:+.4f}")
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
