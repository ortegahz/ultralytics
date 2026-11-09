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
    native_scale: int = 1,
    offset_scale: float = 1.0,
    offset_bias: float = 0.0,
    offset_scale_xy: tuple[float, float] | None = None,
    offset_bias_xy: tuple[float, float] | None = None,
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
        native_scale: late_upsample 的上采样倍数 L（默认 1 = 不做处理）。
        offset_scale / offset_bias: **offset 仿射校准**（默认 1.0 / 0.0 ⇒ 逐位空操作）。
            实测（2026-11-09）：P3/hc64/relu/late 臂的 offset 头塌缩成「常数≈0.5 +
            幅度只有 1/8 的抖动」（输出 std 仅 0.066，与真值回归斜率仅 0.129），
            ⇒ 解码出的位置误差 = −6.97·tx（tx=目标在格内的偏移）⇒ σ≈2.32px
            = 8px 格距的量化下限 ⇒ T1 掉 2.4 pp（召回 72~82% vs 基线 86%）。
            正解：**在训练集上**拟合 ô = s·t_rel + b（t_rel=目标在格内的相对位置∈[0,1]），
            解码时用 ô_cal = (ô − b)/s ⇒ 拟合残差 0.052 格 ≈ **0.42px**（比基线 1.13px 还准）。
            ⇒ 校准常数随模型走（sidecar `<ckpt>.offset_calib.json`），**不改任何权重**。

            ★ 2026-11-09 修（late_upsample 结构的坐标映射缺陷，见 memory.md §10.5）：
              ``late_upsample`` 用 ``F.interpolate(..., mode="nearest")`` 把头的原生网格
              **最近邻复制** L 倍（P3: 80x80 -> 320x320, L=4；P2_fused: 160x160 -> 320x320, L=2）。
              在**复制后的 320 网格**上抽峰会有两个必然的缺陷：

              1. 一个目标的 L×L 个复制像素**分数完全相同** ⇒ 3x3 max-pool 的相等判定
                 ``hmax == heatmap`` 把它们**全部**当成峰 ⇒ 每个目标裂成 L² 个候选
                 ⇒ T1 假阳洪泛（实测 p3/L=4：每帧 51.9 个候选 vs 基线 2.5 个；
                 317,074 个假阳里 86.6% 落在真目标的 8px 内、87.9% 与某个真目标分数**完全相同**）。
              2. 峰落在哪个复制像素是**任意**的（行序优先取到的是块内左上角）
                 ⇒ 坐标系统性地偏 0~(L-1) 像素，offset 也被量化到块内
                 ⇒ 8px 点对点匹配容差被吃掉 ⇒ 召回塌陷（实测召回 72% vs 基线 86%）。

              ⇒ 正解：**在原生网格上抽峰、用 stride*L 解码**（offset 头本来就是原生分辨率的，
                 它的亚像素修正只在原生网格上才有意义）。
              ⇒ ``native_scale=1``（无 late_upsample）时**逐位空操作**
                 ⇒ 冻结基线（trial_0474 / cs_t0043）的 headline 数字不变，门禁自校准仍成立。

    Returns:
        list of dict per image:
            'points': (N, 2) array of (x, y) coordinates in original image space
            'scores': (N,) array of detection confidences
    """
    if native_scale and native_scale > 1:
        # 最近邻复制 ⇒ 原生图 = 复制图按 ::L 抽样（bit-exact）
        heatmap = heatmap[:, :, ::native_scale, ::native_scale]
        offset = offset[:, :, ::native_scale, ::native_scale]
        stride = stride * native_scale
    if offset_scale_xy is not None or offset_bias_xy is not None:
        # ★ 按轴校准（x/y 的 scale、bias 可以不同：实测 x=0.119/y=0.138）
        sx = offset_scale_xy[0] if offset_scale_xy else 1.0
        sy = offset_scale_xy[1] if offset_scale_xy else 1.0
        bx = offset_bias_xy[0] if offset_bias_xy else 0.0
        by = offset_bias_xy[1] if offset_bias_xy else 0.0
        offset = offset.clone()
        offset[:, 0] = (offset[:, 0] - bx) / sx
        offset[:, 1] = (offset[:, 1] - by) / sy
    elif offset_scale != 1.0 or offset_bias != 0.0:
        # offset: (B,2,H,W) ⇒ offset[:,0]=dx, offset[:,1]=dy
        offset = (offset - offset_bias) / offset_scale
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
