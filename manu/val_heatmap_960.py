#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Zero-Training Resolution & Distance Benchmark for Heatmap Point Detector.

Features:
1. Zero training required: directly loads trial_0031 (or any checkpoint) and tests arbitrary resolution (e.g. 640 vs 960).
2. Dual-tolerance evaluation simultaneously:
   - Distance <= 4.0px (Strict CenterNet criteria)
   - Distance <= 8.0px (Official GJB / Anti-UAV Industry benchmark criteria)
3. Full confidence threshold scanning (0.10 ~ 0.50) to observe peak F1 and recall platform.
4. Optional sequence filtering (e.g. test only on hard cases like DJI_0175 or wg2022_011).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.heatmap_evaluate import extract_peaks, evaluate_point_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Heatmap Detector at 960 (or arbitrary) resolution without retraining")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Inference resolution (e.g. 640, 960, 1024)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size per forward pass")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index, e.g. 0, 1")
    parser.add_argument("--top-k", type=int, default=100, help="Max candidate peaks to extract per frame")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt_path = PROJECT_ROOT / args.weights
        if alt_path.exists():
            weights_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    print("=" * 85)
    print(colorstr("bold", "UAV Heatmap Resolution Benchmark (Zero-Retraining Inference)"))
    print("=" * 85)
    print(f"Checkpoint       : {weights_path}")
    print(f"Target Imgsz     : {args.imgsz} x {args.imgsz}")
    print(f"Batch size       : {args.batch}")
    print(f"Device           : {device}")
    print("=" * 85)

    # 1. 加载模型
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    ckpt_stride = ckpt.get("stride", 2)
    ckpt_imgsz = ckpt.get("imgsz", 640)
    print(f"Checkpoint was originally trained at imgsz={ckpt_imgsz}, stride={ckpt_stride}")

    model = YOLO26HeatmapDetector(stride=ckpt_stride, num_classes=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. 构建 DataLoader (指定新的推理分辨率)
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)
    print(f"Loaded validation set: {len(val_dataset)} images.\n")

    # 3. 批量推理收集预测与真值
    all_peaks = []
    all_gts = []
    all_sizes = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Inference @ {args.imgsz}x{args.imgsz}"):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = imgs.shape[0]

            preds = model(imgs)

            peaks = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=ckpt_stride,
                conf_thresh=0.05,  # 保留低置信度候选供阈值扫描
                top_k=args.top_k,
            )
            all_peaks.extend(peaks)

            for b in range(bs):
                mask_b = (b_idx == b)
                all_gts.append(bboxes[mask_b].cpu().numpy())
                all_sizes.append((args.imgsz, args.imgsz))

    # 4. 双准则扫描对比：Distance <= 4.0px 与 Distance <= 8.0px
    # 注意：因为 imgsz 从 640 变为了 args.imgsz，在当前缩放坐标系下：
    # 原始 640 下的 4px/8px 对应当前尺寸下的 4 * (imgsz/640) 与 8 * (imgsz/640) 物理几何等效距离
    scale_ratio = args.imgsz / 640.0
    tol_4_scaled = 4.0 * scale_ratio
    tol_8_scaled = 8.0 * scale_ratio

    print(f"\nEvaluating with Physical Angle Equivalent Tolerances:")
    print(f"  - Strict 4.0px @ 640  ==>  {tol_4_scaled:.2f}px @ {args.imgsz}")
    print(f"  - Standard 8.0px @ 640 ==> {tol_8_scaled:.2f}px @ {args.imgsz}")

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for tol_label, tol_val in [("Distance <= 4.0px (Strict)", tol_4_scaled), ("Distance <= 8.0px (GJB/Standard)", tol_8_scaled)]:
        print("\n" + "=" * 80)
        print(f"Evaluation Table: {tol_label}")
        print("=" * 80)
        print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10} | {'TP':<7} | {'FP':<7} | {'GT':<7}")
        print("-" * 80)

        best_f1 = 0.0
        best_row = None

        for th in thresholds:
            th_preds = []
            for p in all_peaks:
                keep = p["scores"] >= th
                th_preds.append({
                    "points": p["points"][keep],
                    "scores": p["scores"][keep],
                })

            m = evaluate_point_detections(
                predictions=th_preds,
                gt_boxes_list=all_gts,
                img_sizes=all_sizes,
                distance_threshold=tol_val,
            )

            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_row = (th, m)

            print(f"{th:<10.2f} | {m['recall']:<10.4f} | {m['precision']:<10.4f} | {m['f1']:<10.4f} | {m['tp']:<7} | {m['fp']:<7} | {m['total_gt']:<7}")

        print("-" * 80)
        if best_row:
            b_th, b_m = best_row
            print(colorstr("bold", colorstr("green", f">>> Best F1 for {tol_label}: {b_m['f1']:.4f} @ th={b_th:.2f} (Recall={b_m['recall']:.4f}, Precision={b_m['precision']:.4f}, TP={b_m['tp']}, FP={b_m['fp']})")))
        print("=" * 80)


if __name__ == "__main__":
    main()
