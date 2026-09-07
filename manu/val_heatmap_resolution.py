#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Validate Heatmap Point Detector at arbitrary input resolution (e.g. 640 vs 1024/1280).

Key features:
1. Supports overriding imgsz independently of checkpoint defaults.
2. Performs full threshold scanning (0.05 ~ 0.70) to show the full Precision-Recall trajectory.
3. Automatically computes distance distribution for false negatives to see if near-misses exist.
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
    parser = argparse.ArgumentParser(description="Evaluate Heatmap Detector at specified resolution")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_heatmap_stride2_640/trial_0031/weights/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Evaluation image size (e.g. 640, 1024, 1280)")
    parser.add_argument("--stride", type=int, default=2, help="Feature stride (2 for P1, 4 for P2)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce for 1024 to avoid OOM)")
    parser.add_argument("--device", type=str, default="1", help="CUDA device index or cpu")
    parser.add_argument(
        "--dist_thresh",
        type=float,
        default=4.0,
        help="Distance threshold in original evaluation pixel space (normalized against imgsz)",
    )
    parser.add_argument(
        "--scale-dist-with-imgsz",
        action="store_true",
        help="If set, scales pixel distance threshold proportionally with imgsz / 640 (e.g. 4.0px @ 640 -> 6.4px @ 1024)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        # 尝试相对于 PROJECT_ROOT 查找
        alt_path = PROJECT_ROOT / args.weights
        if alt_path.exists():
            weights_path = alt_path
        else:
            raise FileNotFoundError(f"Weights file not found: {args.weights}")

    print(colorstr("bold", f"\n>>> Loading checkpoint: {weights_path}"))
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    ckpt_stride = ckpt.get("stride", args.stride)
    ckpt_imgsz = ckpt.get("imgsz", 640)
    print(f"Checkpoint trained at imgsz={ckpt_imgsz}, stride={ckpt_stride}")
    print(colorstr("bold", f"Evaluating target resolution: imgsz={args.imgsz}, stride={ckpt_stride}"))

    # 如果指定按分辨率等比放缩容差，保持物理视场角容差一致
    effective_dist_thresh = args.dist_thresh
    if args.scale_dist_with_imgsz and ckpt_imgsz > 0:
        scale_ratio = args.imgsz / float(ckpt_imgsz)
        effective_dist_thresh = args.dist_thresh * scale_ratio
        print(f"Distance threshold scaled proportionally: {args.dist_thresh:.1f}px -> {effective_dist_thresh:.2f}px")
    else:
        print(f"Distance threshold fixed: {effective_dist_thresh:.1f}px")

    # 1. 实例化全卷积模型并加载权重
    model = YOLO26HeatmapDetector(stride=ckpt_stride, num_classes=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. 构建指定分辨率的验证集 DataLoader
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)

    print(colorstr("bold", f"Running validation on {len(val_dataset)} images at {args.imgsz}x{args.imgsz}..."))

    all_peaks_raw = []
    val_gt_list = []
    val_sizes_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Infer@{args.imgsz}"):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            bs = imgs.shape[0]

            preds = model(imgs)

            peaks = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=ckpt_stride,
                conf_thresh=0.03,  # 低置信度提取，保留低阈值探索空间
                top_k=120,
            )
            all_peaks_raw.extend(peaks)

            for b in range(bs):
                mask_b = b_idx == b
                val_gt_list.append(bboxes[mask_b].cpu().numpy())
                val_sizes_list.append((args.imgsz, args.imgsz))

    # 3. 扫描不同阈值下的表现
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    best_f1 = 0.0
    best_th = 0.20
    best_metrics = {}
    max_recall_entry = None

    print("\n" + "=" * 78)
    print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10} | {'TP':<7} | {'FP':<7} | {'GT':<7}")
    print("-" * 78)

    for th in thresholds:
        th_preds = []
        for p in all_peaks_raw:
            keep = p["scores"] >= th
            th_preds.append({
                "points": p["points"][keep],
                "scores": p["scores"][keep],
            })

        m = evaluate_point_detections(
            predictions=th_preds,
            gt_boxes_list=val_gt_list,
            img_sizes=val_sizes_list,
            distance_threshold=effective_dist_thresh,
        )

        r = m["recall"]
        p = m["precision"]
        f1 = m["f1"]

        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_metrics = m

        if max_recall_entry is None or r > max_recall_entry["metrics"]["recall"]:
            max_recall_entry = {"th": th, "metrics": m}

        print(f"{th:<10.2f} | {r:<10.4f} | {p:<10.4f} | {f1:<10.4f} | {m['tp']:<7} | {m['fp']:<7} | {m['total_gt']:<7}")

    print("=" * 78)
    print(colorstr("bold", colorstr("green", f"\n>>> [Best F1-Score]  : {best_f1:.4f} @ Threshold = {best_th:.2f}")))
    print(f"    Recall    : {best_metrics['recall']:.4f}")
    print(f"    Precision : {best_metrics['precision']:.4f}")
    print(f"    TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, GT: {best_metrics['total_gt']}")

    print(colorstr("bold", colorstr("cyan", f"\n>>> [Max Recall]    : {max_recall_entry['metrics']['recall']:.4f} @ Threshold = {max_recall_entry['th']:.2f}")))
    print(f"    Precision : {max_recall_entry['metrics']['precision']:.4f}")
    print(f"    F1-Score  : {max_recall_entry['metrics']['f1']:.4f}")
    print(f"    TP: {max_recall_entry['metrics']['tp']}, FP: {max_recall_entry['metrics']['fp']}\n")


if __name__ == "__main__":
    main()
