#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Adaptive Tolerance Point Detection Evaluation (OKS/PCK-style).

Tolerance formula per GT target:
    dist_thresh = max(min_radius, alpha * sqrt(w^2 + h^2))

Default:
    min_radius = 4.0px (protection for tiny <= 3x3 targets)
    alpha = 0.5 (half-diagonal of bounding box, meaning detection falls within target body)
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
from manu.heatmap_evaluate import extract_peaks


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Heatmap Detector with Adaptive OKS/PCK Tolerance")
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
    parser.add_argument("--device", type=str, default="2", help="CUDA device index or cpu")
    parser.add_argument("--min-radius", type=float, default=4.0, help="Minimum matching radius in pixels")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Scale factor for bbox diagonal (0.5 = within target radius, 0.7 = slightly beyond)",
    )
    return parser.parse_args()


def evaluate_adaptive_detections(
    predictions: list[dict[str, np.ndarray]],
    gt_boxes_list: list[np.ndarray],
    img_sizes: list[tuple[int, int]],
    min_radius: float = 4.0,
    alpha: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate detections where each GT target has an adaptive distance tolerance:
        thresh_i = max(min_radius, alpha * sqrt(w_px^2 + h_px^2))
    """
    total_tp = 0
    total_fp = 0
    total_gt = 0

    for pred, gt_norm, (img_h, img_w) in zip(predictions, gt_boxes_list, img_sizes):
        pred_pts = pred["points"]  # (N, 2)
        n_pred = len(pred_pts)

        if len(gt_norm) == 0:
            total_fp += n_pred
            continue

        # Convert GT normalized [cx, cy, w, h] to pixel coords and adaptive thresholds
        gt_pts = np.zeros((len(gt_norm), 2), dtype=np.float32)
        gt_pts[:, 0] = gt_norm[:, 0] * img_w
        gt_pts[:, 1] = gt_norm[:, 1] * img_h

        w_px = gt_norm[:, 2] * img_w
        h_px = gt_norm[:, 3] * img_h
        diagonals = np.sqrt(w_px**2 + h_px**2)
        gt_tolerances = np.maximum(min_radius, alpha * diagonals)

        n_gt = len(gt_pts)
        total_gt += n_gt

        if n_pred == 0:
            continue

        # Compute pairwise euclidean distances (N_pred, M_gt)
        diff = pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        # Check against adaptive threshold for each GT: dists[p, g] <= gt_tolerances[g]
        # Sort candidate pairs by normalized distance (dist / tolerance)
        norm_dists = dists / (gt_tolerances[np.newaxis, :] + 1e-6)

        pred_indices, gt_indices = np.unravel_index(np.argsort(norm_dists, axis=None), norm_dists.shape)

        matched_gt = set()
        matched_pred = set()

        for p_idx, g_idx in zip(pred_indices, gt_indices):
            if dists[p_idx, g_idx] > gt_tolerances[g_idx]:
                break
            if p_idx not in matched_pred and g_idx not in matched_gt:
                matched_pred.add(p_idx)
                matched_gt.add(g_idx)

        tp = len(matched_gt)
        fp = n_pred - tp
        total_tp += tp
        total_fp += fp

    recall = total_tp / (total_gt + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "total_gt": int(total_gt),
    }


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

    print(colorstr("bold", f"\n>>> Loading model checkpoint: {weights_path}"))
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
                top_k=80,
            )
            all_peaks_raw.extend(peaks)

            for b in range(bs):
                mask_b = b_idx == b
                val_gt_list.append(bboxes[mask_b].cpu().numpy())
                val_sizes_list.append((args.imgsz, args.imgsz))

    # 1. 扫描不同置信度阈值 (0.10 ~ 0.60) 下的指标 (在 args.alpha, args.min_radius 下)
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    best_f1 = 0.0
    best_th = 0.20
    best_m = {}

    print("\n" + "=" * 80)
    print(
        f"ADAPTIVE EVALUATION: tol = max({args.min_radius:.1f}px, {args.alpha:.2f} * diagonal) across Conf Thresholds"
    )
    print("=" * 80)
    print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10} | {'TP':<7} | {'FP':<7} | {'GT':<7}")
    print("-" * 80)

    for th in thresholds:
        th_preds = []
        for p in all_peaks_raw:
            keep = p["scores"] >= th
            th_preds.append({
                "points": p["points"][keep],
                "scores": p["scores"][keep],
            })

        m = evaluate_adaptive_detections(
            predictions=th_preds,
            gt_boxes_list=val_gt_list,
            img_sizes=val_sizes_list,
            min_radius=args.min_radius,
            alpha=args.alpha,
        )

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
            best_m = m

        print(
            f"{th:<10.2f} | {m['recall']:<10.4f} | {m['precision']:<10.4f} | {m['f1']:<10.4f} | {m['tp']:<7} | {m['fp']:<7} | {m['total_gt']:<7}"
        )

    print("=" * 80)
    print(colorstr("bold", colorstr("green", f"\n>>> Best F1-Score: {best_f1:.4f} @ Threshold = {best_th:.2f}")))
    print(f"    Recall    : {best_m['recall']:.4f}")
    print(f"    Precision : {best_m['precision']:.4f}")
    print(f"    TP: {best_m['tp']}, FP: {best_m['fp']}, GT: {best_m['total_gt']}")

    # 2. 对比固定 4.0px vs 自适应 alpha 组合在 best_th 下的表现
    print("\n" + "=" * 80)
    print(f"COMPARISON OF CRITERIA @ Conf = {best_th:.2f}")
    print("=" * 80)
    print(f"{'Criteria Setting':<32} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
    print("-" * 80)

    # 选取 best_th 下的预测集合
    fixed_preds = []
    for p in all_peaks_raw:
        keep = p["scores"] >= best_th
        fixed_preds.append({
            "points": p["points"][keep],
            "scores": p["scores"][keep],
        })

    # 固定 4.0px
    m_fixed_4 = evaluate_adaptive_detections(
        fixed_preds, val_gt_list, val_sizes_list, min_radius=4.0, alpha=0.0
    )
    print(f"{'Fixed 4.0px (Original)':<32} | {m_fixed_4['recall']:<10.4f} | {m_fixed_4['precision']:<10.4f} | {m_fixed_4['f1']:<10.4f}")

    # 固定 8.0px
    m_fixed_8 = evaluate_adaptive_detections(
        fixed_preds, val_gt_list, val_sizes_list, min_radius=8.0, alpha=0.0
    )
    print(f"{'Fixed 8.0px (Engineering)':<32} | {m_fixed_8['recall']:<10.4f} | {m_fixed_8['precision']:<10.4f} | {m_fixed_8['f1']:<10.4f}")

    # 自适应 alpha=0.5 (落在机身外接圆内)
    m_adapt_05 = evaluate_adaptive_detections(
        fixed_preds, val_gt_list, val_sizes_list, min_radius=4.0, alpha=0.5
    )
    print(f"{'Adaptive max(4px, 0.50*diag)':<32} | {m_adapt_05['recall']:<10.4f} | {m_adapt_05['precision']:<10.4f} | {m_adapt_05['f1']:<10.4f}")

    # 自适应 alpha=0.7 (稍微放宽边缘)
    m_adapt_07 = evaluate_adaptive_detections(
        fixed_preds, val_gt_list, val_sizes_list, min_radius=4.0, alpha=0.7
    )
    print(f"{'Adaptive max(4px, 0.70*diag)':<32} | {m_adapt_07['recall']:<10.4f} | {m_adapt_07['precision']:<10.4f} | {m_adapt_07['f1']:<10.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
