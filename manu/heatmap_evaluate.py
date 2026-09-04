#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Post-processing and Evaluation utilities for Tiny/Point-like Heatmap Detection.
Extracts local peak points via 3x3 Max-Pooling (NMS-free) and computes Recall/Precision/F1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract_peaks(
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    stride: int = 4,
    conf_thresh: float = 0.20,
    top_k: int = 100,
    kernel_size: int = 3,
) -> list[dict[str, np.ndarray]]:
    """
    Extract peak points from heatmap using 3x3 Max-Pooling NMS.
    
    Args:
        heatmap: (B, 1, H, W) in [0, 1]
        offset:  (B, 2, H, W) sub-pixel offsets (dx, dy)
        stride:  downsample stride (e.g. 4 for P2)
        conf_thresh: peak confidence threshold (lower => higher recall)
        top_k:   max number of detections per image
        kernel_size: max pool kernel size (default 3)

    Returns:
        list of dict per image:
            'points': (N, 2) array of (x, y) coordinates in original image space
            'scores': (N,) array of detection confidences
    """
    pad = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
    keep = (hmax == heatmap) & (heatmap >= conf_thresh)

    B, _, H, W = heatmap.shape
    results = []

    for b in range(B):
        mask_b = keep[b, 0]  # (H, W) bool
        scores = heatmap[b, 0][mask_b]
        if scores.numel() == 0:
            results.append({
                "points": np.zeros((0, 2), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
            })
            continue

        # Get top-k scores if exceeding top_k
        if scores.numel() > top_k:
            scores, topk_inds = torch.topk(scores, top_k)
            y_coords = torch.nonzero(mask_b, as_tuple=False)[:, 0][topk_inds]
            x_coords = torch.nonzero(mask_b, as_tuple=False)[:, 1][topk_inds]
        else:
            nz = torch.nonzero(mask_b, as_tuple=False)
            y_coords = nz[:, 0]
            x_coords = nz[:, 1]

        # Extract sub-pixel offset
        dx = offset[b, 0, y_coords, x_coords]
        dy = offset[b, 1, y_coords, x_coords]

        # Map back to original image coordinates
        pred_x = (x_coords.float() + dx) * stride
        pred_y = (y_coords.float() + dy) * stride

        pts = torch.stack([pred_x, pred_y], dim=1).detach().cpu().numpy()
        scs = scores.detach().cpu().numpy()

        results.append({
            "points": pts,
            "scores": scs,
        })

    return results


def evaluate_point_detections(
    predictions: list[dict[str, np.ndarray]],
    gt_boxes_list: list[np.ndarray],
    img_sizes: list[tuple[int, int]],
    distance_threshold: float = 4.0,
) -> dict[str, float]:
    """
    Evaluate detections against ground-truth for point-like tiny targets.
    A prediction is a True Positive if its Euclidean distance to a GT center is <= distance_threshold.
    
    Args:
        predictions: list of dicts with 'points' (N, 2) in image coords
        gt_boxes_list: list of GT boxes (M, 4) in normalized [cx, cy, w, h] format
        img_sizes: list of (H_img, W_img)
        distance_threshold: pixel distance tolerance (default 4.0 pixels, fits 3x3 targets)
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

        # Convert GT normalized cx, cy to pixel coords
        gt_pts = np.zeros((len(gt_norm), 2), dtype=np.float32)
        gt_pts[:, 0] = gt_norm[:, 0] * img_w
        gt_pts[:, 1] = gt_norm[:, 1] * img_h
        n_gt = len(gt_pts)
        total_gt += n_gt

        if n_pred == 0:
            continue

        # Compute pairwise euclidean distances (n_pred, n_gt)
        diff = pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]  # (N, M, 2)
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))                  # (N, M)

        # Greedy match closest pairs
        matched_gt = set()
        matched_pred = set()

        # Sort all candidate pairs by distance
        pred_indices, gt_indices = np.unravel_index(np.argsort(dists, axis=None), dists.shape)

        for p_idx, g_idx in zip(pred_indices, gt_indices):
            if dists[p_idx, g_idx] > distance_threshold:
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


def find_best_f1_threshold(
    predictions_raw: list[dict[str, np.ndarray]],
    gt_boxes_list: list[np.ndarray],
    img_sizes: list[tuple[int, int]],
    distance_threshold: float = 4.0,
    thresholds: list[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """
    Scan confidence thresholds to find the optimal threshold that maximizes F1 score.
    Runs fast because predictions_raw is pre-extracted.
    """
    if thresholds is None:
        thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    best_f1 = -1.0
    best_th = 0.20
    best_metrics = {}

    for th in thresholds:
        th_preds = []
        for p in predictions_raw:
            keep = p["scores"] >= th
            th_preds.append({
                "points": p["points"][keep],
                "scores": p["scores"][keep],
            })

        m = evaluate_point_detections(
            predictions=th_preds,
            gt_boxes_list=gt_boxes_list,
            img_sizes=img_sizes,
            distance_threshold=distance_threshold,
        )

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
            best_metrics = m

    best_metrics["best_th"] = float(best_th)
    return best_metrics
