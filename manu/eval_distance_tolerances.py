#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Scan distance tolerance thresholds (4.0px, 6.0px, 8.0px, 10.0px, 12.0px)
to analyze the impact of matching radius on Recall, Precision, and F1.
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
    parser = argparse.ArgumentParser(description="Evaluate Heatmap Detector across different distance tolerances")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Model weights path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--stride", type=int, default=2, help="Feature stride")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="2", help="CUDA device")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (default 0.20)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt = PROJECT_ROOT / args.weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Weights file not found: {args.weights}")

    print(colorstr("bold", f"\n>>> Loading model: {weights_path}"))
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    stride = ckpt.get("stride", args.stride)
    model = YOLO26HeatmapDetector(stride=stride, num_classes=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)

    print(colorstr("bold", f"Inferencing validation set ({len(val_dataset)} images)..."))

    all_preds_conf = []
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
                conf_thresh=0.05,  # 提取到底层低阈值候选
                top_k=80,
            )
            all_preds_conf.extend(peaks)

            for b in range(bs):
                mask_b = b_idx == b
                val_gt_list.append(bboxes[mask_b].cpu().numpy())
                val_sizes_list.append((args.imgsz, args.imgsz))

    # 过滤出指定 conf 的预测
    filtered_preds = []
    for p in all_preds_conf:
        keep = p["scores"] >= args.conf
        filtered_preds.append({
            "points": p["points"][keep],
            "scores": p["scores"][keep],
        })

    # 测试不同距离容差
    dist_thresholds = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]

    print("\n" + "=" * 80)
    print(f"ANALYSIS: Distance Tolerance vs Metrics @ Conf = {args.conf:.2f}")
    print("=" * 80)
    print(f"{'Distance Tol':<14} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10} | {'TP':<7} | {'FP':<7} | {'FN':<7}")
    print("-" * 80)

    for d_tol in dist_thresholds:
        m = evaluate_point_detections(
            predictions=filtered_preds,
            gt_boxes_list=val_gt_list,
            img_sizes=val_sizes_list,
            distance_threshold=d_tol,
        )
        total_gt = m["total_gt"]
        fn = total_gt - m["tp"]
        marker = " <-- [当前基准]" if d_tol == 4.0 else ""
        print(
            f"{d_tol:<6.1f} px{' ' * 6} | {m['recall']:<10.4f} | {m['precision']:<10.4f} | {m['f1']:<10.4f} | "
            f"{m['tp']:<7} | {m['fp']:<7} | {fn:<7}{marker}"
        )

    print("=" * 80)
    print("\n说明：")
    print("当距离容差放宽时：原先超出 4.0px 的近邻预测点不仅由 FN 变成 TP（增加 Recall），")
    print("同时由于它们不再被算作孤立误报，FP 也会同步减少（大幅改善 Precision 和 F1）！\n")


if __name__ == "__main__":
    main()
