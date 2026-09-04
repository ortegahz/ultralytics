#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Scan Confidence Thresholds (0.10 ~ 0.80) to find the Peak F1-Score & Best Threshold.
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
    parser = argparse.ArgumentParser(description="Find Best F1 Threshold for Heatmap Detector")
    parser.add_argument("--data", type=str, default="/mnt/data/siping/datasets/manu/uav/data.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint (.pt)")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="2")
    parser.add_argument("--dist_thresh", type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # 1. 加载模型
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    stride = ckpt.get("stride", args.stride)
    imgsz = ckpt.get("imgsz", args.imgsz)

    model = YOLO26HeatmapDetector(stride=stride, num_classes=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. 构建验证集 DataLoader
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = imgsz
    cfg.data = args.data
    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=8, shuffle=False)

    print(colorstr("bold", f"Running inference on validation set ({len(val_dataset)} images)..."))

    # 3. 提取所有可能的候选峰值（门槛设为 0.05，后面再离线过滤）
    all_peaks_raw = []
    val_gt_list = []
    val_sizes_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = imgs.shape[0]

            preds = model(imgs)
            peaks = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
                conf_thresh=0.05,
                top_k=100,
            )
            all_peaks_raw.extend(peaks)

            for b in range(bs):
                mask_b = b_idx == b
                val_gt_list.append(bboxes[mask_b].cpu().numpy())
                val_sizes_list.append((imgsz, imgsz))

    # 4. 扫描各个阈值（0.10 到 0.80）
    thresholds = np.linspace(0.10, 0.80, 15)
    best_f1 = 0.0
    best_th = 0.0
    best_metrics = {}

    print("\n" + "=" * 70)
    print(f"{'Threshold':<12} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10} | {'TP':<8} | {'FP':<8}")
    print("-" * 70)

    for th in thresholds:
        th_preds = []
        for p in all_peaks_raw:
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

    print("=" * 70)
    print(colorstr("bold", colorstr("green", f"\n>>> Best F1-Score: {best_f1:.4f} @ Threshold: {best_th:.2f}")))
    print(f"    Recall at Best F1:    {best_metrics['recall']:.4f}")
    print(f"    Precision at Best F1: {best_metrics['precision']:.4f}")
    print(f"    TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, GT: {best_metrics['total_gt']}\n")


if __name__ == "__main__":
    main()
