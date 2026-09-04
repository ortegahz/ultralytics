#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Evaluate standard YOLO26 model using the exact same Point-Detection criteria (Distance <= 4.0 px).
This ensures 100% fair Apple-to-Apple comparison against the Heatmap model!
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

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import evaluate_point_detections


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO26 under Point Detection Criteria")
    # 默认权重路径，后续可根据需要直接在此修改或命令行传参覆盖
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo26np2.pt",
        help="Path to YOLO26 model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="2", help="CUDA device index or cpu")
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=4.0,
        help="Distance threshold (pixels) for TP matching",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    print(colorstr("bold", f"Loading YOLO26 model from {args.weights}..."))
    model = YOLO(args.weights)

    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=8, shuffle=False)

    print(colorstr("bold", f"Running YOLO26 validation on {len(val_dataset)} images..."))

    all_points_raw = []
    val_gt_list = []
    val_sizes_list = []

    # 1. 运行推理并提取所有候选框的中心点
    for batch in tqdm(val_loader, desc="YOLO Inference"):
        imgs = batch["img"].float() / 255.0  # (B, 3, H, W) normalized to [0, 1]
        bboxes = batch["bboxes"]
        b_idx = batch["batch_idx"]
        bs = imgs.shape[0]

        # 调用 YOLO 内部 predict 接口，conf 设为极低 0.01 收集所有潜在点
        results = model.predict(imgs, conf=0.01, verbose=False, device=device)

        for b in range(bs):
            res = results[b]
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else np.zeros((0, 4))
            confs = res.boxes.conf.cpu().numpy() if len(res.boxes) > 0 else np.zeros((0,))

            if len(boxes) > 0:
                # 将边界框转换为中心点 (cx, cy)
                cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
                cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
                pts = np.stack([cx, cy], axis=1)
            else:
                pts = np.zeros((0, 2), dtype=np.float32)

            all_points_raw.append({"points": pts, "scores": confs})

            mask_b = b_idx == b
            val_gt_list.append(bboxes[mask_b].cpu().numpy())
            val_sizes_list.append((args.imgsz, args.imgsz))

    # 2. 扫描阈值计算在“距离 <= 4px”标准下的真实点指标
    thresholds = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    best_f1 = 0.0
    best_th = 0.0
    best_metrics = {}

    print("\n" + "=" * 75)
    print(f"--- YOLO26 Evaluation under Point Criteria (Distance <= {args.dist_thresh:.1f}px) ---")
    print("=" * 75)
    print(f"{'Threshold':<12} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10} | {'TP':<8} | {'FP':<8}")
    print("-" * 75)

    for th in thresholds:
        th_preds = []
        for p in all_points_raw:
            keep = p["scores"] >= th
            th_preds.append({
                "points": p["points"][keep],
                "scores": p["scores"][keep],
            })

        metrics = evaluate_point_detections(
            predictions=th_preds,
            gt_boxes_list=val_gt_list,
            img_sizes=val_sizes_list,
            distance_threshold=args.dist_thresh,
        )

        r = metrics["recall"]
        p = metrics["precision"]
        f1 = metrics["f1"]

        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = metrics

        print(f"{th:<12.2f} | {r:<10.4f} | {p:<10.4f} | {f1:<10.4f} | {metrics['tp']:<8} | {metrics['fp']:<8}")

    print("=" * 75)
    print(colorstr("bold", colorstr("green", f"\n>>> [YOLO26 Point-Baseline] Best F1: {best_f1:.4f} @ Threshold: {best_th:.2f}")))
    print(f"    Recall at Best F1:    {best_metrics['recall']:.4f}")
    print(f"    Precision at Best F1: {best_metrics['precision']:.4f}")
    print(f"    TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, GT: {best_metrics['total_gt']}\n")


if __name__ == "__main__":
    main()
