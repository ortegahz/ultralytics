#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-Centric 双区域局部 ROI 特征算子（v2 核心）。

为什么要有它（v1 的业务偏差）
------------------------------
v1 在**全画幅 48 宫格随机采点**，测的是「背景任意位置的局部纹理可跟踪性」。
吊舱业务的真实痛点却是「**特定业务目标（无人机）本身**能不能跟得住」：
天空一片纯黑时全图梯度极低，但只要目标本体与紧邻背景有足够**局部信杂比**与锐度，
跟踪器照样能稳稳锁住目标；反过来地面纹理再丰富，只要目标飞进局部热交叉区，
照样会跟丢。⇒ 必须把度量窗口**收缩到目标本体 + 紧邻背景环**。

区域定义（任务书口径）
----------------------
* **目标核心区（Target Core）**：以 ``(cx, cy)`` 为中心、``core_w = max(9, round(w))``、
  ``core_h = max(9, round(h))`` 的矩形（防止极小红外点目标采样过少）；
* **紧邻背景环（Local Surround Ring）**：以 ``(cx, cy)`` 为中心、外扩正方形
  ``ring_size = max(48, 3 * max(round(w), round(h)))``，**严格挖除**核心区外扩 2 px
  保护边界后的内部区域（防目标热扩散边缘污染背景统计）；
* 两区**交集必须严格为空**（``core_mask & ring_mask`` 恒为 0），且所有裁剪**边界安全**。

指标
----
核心区：``local_scr``（无极性局部信杂比）、``target_local_contrast``、``target_lap_var``、
``target_grad_mean``；背景环：``surround_std``、``surround_edge_density``、``surround_blockiness``；
全图对照：``global_contrast``、``global_lap_var``（复用 v1 算子口径）。

执行机：服务器 ``huangzhe@192.168.99.40:32222``（conda env ``uav``，依赖 cv2）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-5
CORE_MIN_DIM = 9                # 核心区最小边长（防极小目标采样不足）
RING_MIN_SIZE = 48              # 背景环最小边长
RING_SCALE = 3                  # 背景环边长 = max(48, 3 * max(w, h))
CORE_PROTECT_PX = 2             # 背景环挖除核心区时保留的保护边界
CANNY_LOW, CANNY_HIGH = 60, 160   # 与 v1（vq_metrics.py）同口径
BLOCK = 8                      # 块效应 block 边长（与 v1 同口径）

LOCAL_METRICS = ("local_scr", "target_local_contrast", "target_lap_var", "target_grad_mean",
                  "surround_std", "surround_edge_density", "surround_blockiness")
GLOBAL_METRICS = ("global_contrast", "global_lap_var")
ALL_METRICS = LOCAL_METRICS + GLOBAL_METRICS

#: 各指标的**先验方向**：+1 = 越大越好（画质类）、-1 = 越大越差（噪声 / 伪影类）、
#: 0 = 方向中性（不做方向判定，只看 |ρ|）。
#: 用途：相关性表判定「方向反转」——画质指标与跟住率应当**正相关**，
#: 噪声/块效应类指标与跟住率应当**负相关**。
METRIC_DIRECTION = {
    "local_scr": +1, "target_local_contrast": +1, "target_lap_var": +1, "target_grad_mean": +1,
    "global_contrast": +1, "global_lap_var": +1,
    "surround_std": -1, "surround_blockiness": -1,
    "surround_edge_density": 0,
}


@dataclass
class Regions:
    """目标核心区与背景环的像素掩码（边界安全 + 严格不重叠）。"""

    core_mask: np.ndarray
    ring_mask: np.ndarray
    core_box: tuple[int, int, int, int]      # x0, y0, x1, y1
    ring_box: tuple[int, int, int, int]
    outer_box: tuple[int, int, int, int]     # 外扩正方形（未挖空）


def _clamp_box(cx: float, cy: float, box_w: int, box_h: int,
               width: int, height: int) -> tuple[int, int, int, int]:
    """以 ``(cx, cy)`` 为中心取 ``box_w × box_h`` 的框，并**整体平移**回画面内。

    约定与 v1（``manu/tracking/lookahead/xcorr.window``）一致：**平移而不是裁剪**，
    这样掩码的坐标系与图像坐标系始终一致。
    """
    box_w = max(1, min(int(box_w), width))
    box_h = max(1, min(int(box_h), height))
    x0 = int(round(cx)) - box_w // 2
    y0 = int(round(cy)) - box_h // 2
    x1, y1 = x0 + box_w, y0 + box_h
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 = max(0, x0 - (x1 - width))
        x1 = width
    if y1 > height:
        y0 = max(0, y0 - (y1 - height))
        y1 = height
    return x0, y0, x1, y1


def build_regions(frame: np.ndarray, cx: float, cy: float, w: float, h: float) -> Regions:
    """构造目标核心区 + 背景环掩码（边界安全、``core & ring == 0``）。"""
    height, width = frame.shape[:2]
    core_w = max(CORE_MIN_DIM, int(round(float(w))))
    core_h = max(CORE_MIN_DIM, int(round(float(h))))
    cx0, cy0, cx1, cy1 = _clamp_box(cx, cy, core_w, core_h, width, height)

    ring_size = max(RING_MIN_SIZE, RING_SCALE * max(int(round(float(w))), int(round(float(h)))))
    ring_size = min(ring_size, min(width, height))
    ox0, oy0, ox1, oy1 = _clamp_box(cx, cy, ring_size, ring_size, width, height)

    core_mask = np.zeros((height, width), dtype=bool)
    core_mask[cy0:cy1, cx0:cx1] = True

    # 背景环 = 外扩正方形 − 核心区外扩 2 px（保护边界）
    px0, py0 = max(0, cx0 - CORE_PROTECT_PX), max(0, cy0 - CORE_PROTECT_PX)
    px1, py1 = min(width, cx1 + CORE_PROTECT_PX), min(height, cy1 + CORE_PROTECT_PX)
    ring_mask = np.zeros((height, width), dtype=bool)
    ring_mask[oy0:oy1, ox0:ox1] = True
    ring_mask[py0:py1, px0:px1] = False
    ring_mask &= ~core_mask          # 双保险：核心区绝不进背景环
    return Regions(core_mask=core_mask, ring_mask=ring_mask,
                   core_box=(cx0, cy0, cx1, cy1), ring_box=(ox0, oy0, ox1, oy1),
                   outer_box=(ox0, oy0, ox1, oy1))


def _p90(values: np.ndarray) -> float:
    """P90（numpy 默认线性插值口径）。"""
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values.astype(np.float64), 90))


def _blockiness_region(patch: np.ndarray, block: int = BLOCK) -> float:
    """块效应：块边界跨像素梯度均值 ÷ 块内相邻像素梯度均值（与 v1 ``vq_metrics.blockiness`` 同口径）。

    v1 的写法是「边界梯度均值 ÷ 全图相邻像素梯度均值」，其中分母几乎全部由块内像素贡献
    ⇒ 与任务书「块边界跨像素梯度均值 / 块内相邻像素梯度均值」等价。
    """
    height, width = patch.shape[:2]
    cols = np.arange(block, width, block)
    rows = np.arange(block, height, block)
    fh = cv2.absdiff(patch[:, cols], patch[:, cols - 1]) if cols.size else None
    fv = cv2.absdiff(patch[rows, :], patch[rows - 1, :]) if rows.size else None
    border_h = float(fh.mean()) if fh is not None and fh.size else 0.0
    border_v = float(fv.mean()) if fv is not None and fv.size else 0.0
    mean_h = float(cv2.absdiff(patch[:, 1:], patch[:, :-1]).mean()) + 1e-6
    mean_v = float(cv2.absdiff(patch[1:, :], patch[:-1, :]).mean()) + 1e-6
    return (border_h / mean_h + border_v / mean_v) / 2.0


def compute_roi_metrics(frame: np.ndarray, cx: float, cy: float, w: float, h: float) -> dict:
    """算一帧的局部 ROI 指标 + 全图对照指标。

    ``frame`` 必须是**原生分辨率灰度图**（不缩放全图）。
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    regions = build_regions(gray, cx, cy, w, h)
    core_vals = gray[regions.core_mask].astype(np.float64)
    ring_vals = gray[regions.ring_mask].astype(np.float64)
    if core_vals.size == 0 or ring_vals.size == 0:
        return {name: float("nan") for name in ALL_METRICS}

    mu_ring = float(ring_vals.mean())
    sigma_ring = float(ring_vals.std())
    diff = np.abs(core_vals - mu_ring)
    p90_diff = _p90(diff)
    p90_core = _p90(core_vals)

    cx0, cy0, cx1, cy1 = regions.core_box
    core_patch = gray[cy0:cy1, cx0:cx1].astype(np.float64)
    # ⚠️ 边界口径（**复算器必须同款**）：cv2 的 Sobel/Laplacian 默认 ``BORDER_DEFAULT``
    # = ``BORDER_REFLECT_101`` ⇒ 等价于 ``np.pad(mode="reflect")``（**重复**边缘像素）。
    # 另注：``cv2.Laplacian(ksize=3)`` 的有效核是**对角型** ``[[2,0,2],[0,-8,0],[2,0,2]]``
    # （``ksize=1`` 才是十字型 ``[[0,1,0],[1,-4,1],[0,1,0]]``）。
    # 若换零填充 / valid 卷积 / 十字核，小尺寸核心区（9×9 ~ 30×30）的边缘像素会被整块换掉，
    # ``target_lap_var`` 会差出 5~10 倍（实测 JPEG q10 样本 51420 vs 5164）。
    lap = cv2.Laplacian(core_patch, cv2.CV_64F, ksize=3)
    grad_x = cv2.Sobel(core_patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(core_patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    ring_edge_ratio = float((edges[regions.ring_mask] > 0).mean())

    ox0, oy0, ox1, oy1 = regions.outer_box
    outer_patch = gray[oy0:oy1, ox0:ox1]

    metrics = {
        # ---- 目标核心区
        "local_scr": p90_diff / (sigma_ring + EPS),
        "target_local_contrast": p90_diff / (mu_ring + p90_core + EPS),
        "target_lap_var": float(lap.var()),
        "target_grad_mean": float(grad_mag.mean()),
        # ---- 紧邻背景环
        "surround_std": sigma_ring,
        "surround_edge_density": ring_edge_ratio,
        "surround_blockiness": _blockiness_region(outer_patch),
        # ---- 全图对照（复用 v1 算子口径）
        "global_contrast": float((np.percentile(gray, 99) - np.percentile(gray, 1))
                                 / (np.percentile(gray, 99) + np.percentile(gray, 1) + 1e-6)),
        "global_lap_var": float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var()),
    }
    return {k: (float(v) if math.isfinite(float(v)) else float("nan")) for k, v in metrics.items()}


def regions_overlap_pixels(frame: np.ndarray, cx: float, cy: float, w: float, h: float) -> int:
    """核心区与背景环的**重叠像素数**（测试要断言恒为 0）。"""
    regions = build_regions(frame, cx, cy, w, h)
    return int(np.logical_and(regions.core_mask, regions.ring_mask).sum())
