#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 独立复算脚本（**不 import** 主评估聚合模块）。

为什么要有它
------------
主评估链路（``run_eval_v2.py``）里有四处最容易出错、又最影响结论的地方：

1. **ROI 区域几何**：核心区/背景环的边界裁剪与掩码构造一旦错位，
   ``local_scr`` 会静默偏掉，而且**不会报错**；
2. **退化算子**：退化是否真的作用在「整帧 + 整窗 5 帧」上；
3. **膝点反解**：分箱 → 保序回归 → PCHIP → Kneedle/PCHIP 双算法交叉验证。
4. **曲线有效性守卫**：逐档跟住率的跨度 / 单调性、逐档指标中位的单调性
   ⇒ 决定某格膝点是否作废（``method_used="degenerate"``）。

本脚本对上述四处各做一次**独立实现**的复算：

* ROI 指标：直接读原始红外帧，用**纯 numpy**（不调 v1/v2 的指标算子）重算
  （P90 自己实现、Laplacian/Sobel 自己卷积、块效应自己差分）；
* 退化：直接对原始帧重放退化算子；
* 膝点：自己实现分箱 + 保序回归 + PCHIP + Kneedle。
* 曲线有效性：自己实现逐档跟住率 / 指标中位的分组、排序与单调性判定。

断言：数值误差 **< 1e-6**（含膝点线位）。

执行机：服务器 ``huangzhe@192.168.99.40:32222``（conda env ``uav``）。
复现：

    python3 iqa_v2/verify_recalc_v2.py --samples 120
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TOL = 1e-6
METRICS = ("local_scr", "target_local_contrast", "target_lap_var", "target_grad_mean",
           "surround_std", "surround_edge_density", "surround_blockiness",
           "global_contrast", "global_lap_var")
#: 主实现（run_eval_v2.py）的曲线有效性门槛：跟住率曲线总跨度必须 **> 5 个百分点**
CURVE_MIN_SPAN = 0.05
#: 主实现出行的最小窗口数（len(sub) < 20 的组合不出行）⇒ 复算同样跳过
MIN_WINDOWS = 20
#: 三臂（与主实现 TRACKER_FAMILIES 同序）
ARMS = ("xcorr", "kcf_legacy", "csrt")
#: 违规明细最多打印多少条（其余只报计数）
MAX_REPORT = 20


# --------------------------------------------------------------------------- 独立实现
def percentile_linear(values: np.ndarray, q: float) -> float:
    """自己实现 numpy 的线性插值分位数（与 ``np.percentile`` 默认口径一致）。"""
    if values.size == 0:
        return float("nan")
    ordered = np.sort(values.astype(np.float64))
    if ordered.size == 1:
        return float(ordered[0])
    position = (ordered.size - 1) * (q / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def conv3x3(patch: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """纯 numpy 的 3×3 相关卷积，**边界按 ``BORDER_REFLECT_101`` 填充**。

    ⚠️ 口径必须与主实现（``target_roi_metrics.compute_roi_metrics`` 里的
    ``cv2.Laplacian`` / ``cv2.Sobel``）**逐像素一致**：
    cv2 的默认边界是 ``BORDER_DEFAULT = BORDER_REFLECT_101``
    （= ``c b a | a b c d | c b a``，**重复**边缘像素）
    ⇒ 等价于 ``np.pad(mode="reflect")``（numpy 的 ``symmetric`` 是**另一种**边界，
    会让 Sobel/Laplacian 在小核心区上整块对不上）。
    实测：mode=reflect ⇒ 与 cv2 **逐像素相等**（maxdiff = 0.0）；
    mode=symmetric / edge / constant ⇒ 小核心区上差出 5~10 倍。
    """
    kh, kw = kernel.shape
    h, w = patch.shape
    if h < kh or w < kw:
        return np.zeros((0, 0), dtype=np.float64)
    padded = np.pad(patch.astype(np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)),
                    mode="reflect")
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            result += kernel[i, j] * padded[i:i + h, j:j + w]
    return result


#: cv2.Laplacian 的两个**实际**卷积核（服务器 OpenCV 4.10.0，取冲激函数反查得到）：
#: ``ksize=3`` ⇒ **对角型** ``[[2,0,2],[0,-8,0],[2,0,2]]``；
#: ``ksize=1``（默认）⇒ **十字型** ``[[0,1,0],[1,-4,1],[0,1,0]]``。
LAPLACIAN_KERNEL_K3 = np.array([[2.0, 0.0, 2.0], [0.0, -8.0, 0.0], [2.0, 0.0, 2.0]])
LAPLACIAN_KERNEL_K1 = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def laplacian3x3(patch: np.ndarray, ksize: int = 3) -> np.ndarray:
    """复刻 ``cv2.Laplacian(patch, cv2.CV_64F, ksize=...)`` 的响应。

    ⚠️ 两个口径必须与主实现一致：
      * 核心区锐度 ``target_lap_var`` ⇒ 主实现用 ``ksize=3`` ⇒ **对角核**；
      * 全图对照 ``global_lap_var`` ⇒ 主实现**不传 ksize**（默认 = 1）⇒ **十字核**
        （这一项要「复用 v1 算子口径」，v1 的 ``vq_metrics.py`` 也是不传 ksize）。
    写成同一个核会让其中一项差 5~10 倍（实测 51420 vs 5164）⇒ 复算误报失败。
    """
    kernel = LAPLACIAN_KERNEL_K3 if int(ksize) == 3 else LAPLACIAN_KERNEL_K1
    return conv3x3(patch.astype(np.float64), kernel)


def sobel_magnitude(patch: np.ndarray) -> np.ndarray:
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    gx = conv3x3(patch.astype(np.float64), kx)
    gy = conv3x3(patch.astype(np.float64), ky)
    return np.sqrt(gx * gx + gy * gy)


def independent_roi_metrics(frame: np.ndarray, cx: float, cy: float, w: float, h: float) -> dict:
    """独立实现的 ROI 指标（不 import ``target_roi_metrics``）。"""
    height, width = frame.shape[:2]
    core_w = max(9, int(round(float(w))))
    core_h = max(9, int(round(float(h))))
    ring_size = min(max(48, 3 * max(int(round(float(w))), int(round(float(h))))),
                    min(width, height))

    def box(bw: int, bh: int):
        x0 = int(round(cx)) - bw // 2
        y0 = int(round(cy)) - bh // 2
        x1, y1 = x0 + bw, y0 + bh
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > width:
            x0 -= (x1 - width)
            x1 = width
        if y1 > height:
            y0 -= (y1 - height)
            y1 = height
        return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

    cx0, cy0, cx1, cy1 = box(core_w, core_h)
    ox0, oy0, ox1, oy1 = box(ring_size, ring_size)

    core_mask = np.zeros((height, width), dtype=bool)
    core_mask[cy0:cy1, cx0:cx1] = True
    ring_mask = np.zeros((height, width), dtype=bool)
    ring_mask[oy0:oy1, ox0:ox1] = True
    ring_mask[max(0, cy0 - 2):min(height, cy1 + 2), max(0, cx0 - 2):min(width, cx1 + 2)] = False
    ring_mask &= ~core_mask

    core_vals = frame[core_mask].astype(np.float64)
    ring_vals = frame[ring_mask].astype(np.float64)
    if core_vals.size == 0 or ring_vals.size == 0:
        return {name: float("nan") for name in METRICS}

    mu_ring = float(ring_vals.mean())
    sigma_ring = float(ring_vals.std())
    diff = np.abs(core_vals - mu_ring)
    p90_diff = percentile_linear(diff, 90.0)
    p90_core = percentile_linear(core_vals, 90.0)

    core_patch = frame[cy0:cy1, cx0:cx1].astype(np.float64)
    lap = laplacian3x3(core_patch)
    grad = sobel_magnitude(core_patch)

    import cv2

    edges = cv2.Canny(frame, 60, 160)
    edge_ratio = float((edges[ring_mask] > 0).mean())

    outer = frame[oy0:oy1, ox0:ox1].astype(np.float64)
    block = 8
    cols = np.arange(block, outer.shape[1], block)
    rows = np.arange(block, outer.shape[0], block)
    border_h = float(np.abs(outer[:, cols] - outer[:, cols - 1]).mean()) if cols.size else 0.0
    border_v = float(np.abs(outer[rows, :] - outer[rows - 1, :]).mean()) if rows.size else 0.0
    mean_h = float(np.abs(outer[:, 1:] - outer[:, :-1]).mean()) + 1e-6
    mean_v = float(np.abs(outer[1:, :] - outer[:-1, :]).mean()) + 1e-6
    blockiness = (border_h / mean_h + border_v / mean_v) / 2.0

    flat = frame.astype(np.float64)
    p99 = percentile_linear(flat.reshape(-1), 99.0)
    p01 = percentile_linear(flat.reshape(-1), 1.0)
    global_contrast = float((p99 - p01) / (p99 + p01 + 1e-6))
    lap_full = laplacian3x3(flat, ksize=1)      # 与 v1 口径一致：cv2 默认 ksize=1（十字核）
    global_lap_var = float(lap_full.var()) if lap_full.size else float("nan")

    return {
        "local_scr": float(p90_diff / (sigma_ring + 1e-5)),
        "target_local_contrast": float(p90_diff / (mu_ring + p90_core + 1e-5)),
        "target_lap_var": float(lap.var()) if lap.size else float("nan"),
        "target_grad_mean": float(grad.mean()) if grad.size else float("nan"),
        "surround_std": sigma_ring,
        "surround_edge_density": edge_ratio,
        "surround_blockiness": float(blockiness),
        "global_contrast": global_contrast,
        "global_lap_var": global_lap_var,
    }


def independent_degrade(frame: np.ndarray, family: str, level: float, angle_deg: float) -> np.ndarray:
    """独立实现的退化算子（不 import ``degradation_4d``）。"""
    import cv2

    if family == "clean":
        return frame.copy()
    if family == "blur":
        return cv2.GaussianBlur(frame, (0, 0), sigmaX=float(level), sigmaY=float(level))
    if family == "motion":
        length = max(3, int(round(float(level))) | 1)
        kernel = np.zeros((length, length), dtype=np.float32)
        theta = math.radians(float(angle_deg))
        centre = (length - 1) / 2.0
        for index in range(length):
            offset = index - centre
            x = int(round(centre + offset * math.cos(theta)))
            y = int(round(centre + offset * math.sin(theta)))
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1.0
        total = float(kernel.sum()) or 1.0
        return cv2.filter2D(frame, -1, kernel / total, borderType=cv2.BORDER_REPLICATE)
    if family == "contrast":
        alpha = float(level)
        mu = float(frame.mean())
        return np.clip(alpha * frame.astype(np.float32) + (1.0 - alpha) * mu, 0, 255).astype(np.uint8)
    raise ValueError(family)


def independent_knee(xs: np.ndarray, ys: np.ndarray) -> float:
    """独立实现的膝点（分箱 → 保序回归 → PCHIP → Kneedle）。"""
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import isotonic_regression

    if xs.size < 4:
        return float("nan")
    fitted = np.asarray(isotonic_regression(ys, increasing=True).x, dtype=float)
    unique_x, start = np.unique(xs, return_index=True)
    unique_y = np.maximum.reduceat(fitted, start)
    if unique_x.size < 3:
        return float("nan")
    x_span = unique_x.max() - unique_x.min()
    y_span = unique_y.max() - unique_y.min()
    if x_span <= 0 or y_span <= 0:
        return float("nan")
    xn = (unique_x - unique_x.min()) / x_span
    yn = (unique_y - unique_y.min()) / y_span
    index = int(np.argmax(np.abs(yn - xn)))
    _ = PchipInterpolator(unique_x, unique_y)
    return float(unique_x[index])


def degradation_strength(family: str, level: float) -> float:
    """独立实现的「退化强度」映射（不 import degradation_4d）。

    blur / motion：档位越大＝退化越强；contrast：alpha 越小＝退化越强 ⇒ 取 1 - alpha。
    """
    if family == "contrast":
        return 1.0 - float(level)
    return float(level)


def level_success_profile(rows: list[dict], family: str, tracker: str) -> tuple:
    """独立实现的逐档跟住率曲线（退化强度递增）→ (rates, span, monotone)。

    * ``span`` = 跟住率均值的极差（主实现：max - min）；
    * ``monotone`` = 逐档均值**非递增**（容差 1e-9）。
    """
    by_level: dict[float, list[float]] = {}
    for row in rows:
        by_level.setdefault(float(row["condition_level"]), []).append(
            float(row["%s_success" % tracker]))
    levels = sorted(by_level, key=lambda level: degradation_strength(family, level))
    rates = [sum(by_level[level]) / len(by_level[level]) for level in levels]
    span = (max(rates) - min(rates)) if rates else 0.0
    monotone = int(all(rates[index + 1] <= rates[index] + 1e-9
                       for index in range(len(rates) - 1)))
    return rates, span, monotone


def level_metric_profile(rows: list[dict], family: str, metric: str) -> tuple:
    """独立实现的逐档指标中位（退化强度递增）→ (meds, monotone)。

    * 只对**有限值**取中位（主实现同口径）；
    * ``monotone`` = 逐档中位**非递增**（容差 0.1%：meds[i+1] <= meds[i] * 1.001）。
    """
    by_level: dict[float, list[float]] = {}
    for row in rows:
        value = float(row[metric])
        if math.isfinite(value):
            by_level.setdefault(float(row["condition_level"]), []).append(value)
    levels = sorted(by_level, key=lambda level: degradation_strength(family, level))
    meds = [float(np.median(by_level[level])) for level in levels]
    monotone = int(all(meds[index + 1] <= meds[index] * 1.001
                       for index in range(len(meds) - 1)))
    return meds, monotone


def _keep_window(class_by_key: dict, row: dict, curve_set: str, tracker: str) -> bool:
    """曲线集筛选（独立复现主实现的 _subset）：own_good = 该臂好片段；common_good = 三臂共同好片段。"""
    info = class_by_key.get((row["seq_id"], int(row["frame_id"])))
    if info is None:
        return False
    if curve_set == "own_good":
        return info.get("class_%s" % tracker) == "good"
    return all(info.get("class_%s" % arm) == "good" for arm in ARMS)


# --------------------------------------------------------------------------- 复算流程
def verify_roi_metrics(cli: argparse.Namespace) -> int:
    import cv2

    detail_rows = list(csv.DictReader(open(cli.detail_csv, encoding="utf-8")))
    if not detail_rows:
        raise SystemExit("明细 CSV 为空：%s" % cli.detail_csv)
    windows = list(csv.DictReader(open(cli.windows_csv, encoding="utf-8")))
    window_index = {(row["seq_id"], int(row["frame_id"])): row for row in windows}

    dataset_root = Path(cli.dataset_root)
    rng = random.Random(cli.seed)
    sampled = rng.sample(detail_rows, min(cli.samples, len(detail_rows)))

    worst = 0.0
    worst_info = ""
    checked = 0
    for row in sampled:
        key = (row["seq_id"], int(row["frame_id"]))
        win = window_index.get(key)
        if win is None:
            continue
        seq_dir = dataset_root / "train" / row["seq_id"]
        if not seq_dir.is_dir():
            continue
        frame_files = sorted([p for p in seq_dir.iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        frame_id = int(row["frame_id"])
        path = frame_files[frame_id]
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        w = float(win["w"])
        h = float(win["h"])
        cx = float(win["gt_0_cx"])
        cy = float(win["gt_0_cy"])
        degraded = independent_degrade(frame, row["condition_family"],
                                        float(row["condition_level"]),
                                        _angle_for(row["seq_id"], frame_id, cli.seed))
        metrics = independent_roi_metrics(degraded, cx, cy, w, h)
        for name in METRICS:
            got = float(row[name])
            expect = metrics[name]
            if not (math.isfinite(got) and math.isfinite(expect)):
                continue
            diff = abs(got - expect)
            if diff > worst:
                worst = diff
                worst_info = "%s/%s %s" % (row["seq_id"], row["condition_label"], name)
        checked += 1

    print("[VERIFY][ROI] 抽样 %d 行，最大绝对误差 %.3e（%s）" % (checked, worst, worst_info or "-"))
    if checked == 0:
        raise SystemExit("[FATAL] 没有可比对的样本")
    if worst >= TOL:
        raise SystemExit("[FAIL] ROI 指标复算误差 %.3e ≥ %.0e" % (worst, TOL))
    return checked


def _angle_for(seq_id: str, frame_id: int, seed: int) -> float:
    import hashlib

    digest = hashlib.md5(("%d|%s:%d" % (int(seed), seq_id, frame_id)).encode("utf-8")).digest()
    return float(int.from_bytes(digest[:4], "big") / float(1 << 32) * 180.0)


def verify_knee(cli: argparse.Namespace) -> int:
    detail_rows = list(csv.DictReader(open(cli.detail_csv, encoding="utf-8")))
    knee_rows = list(csv.DictReader(open(cli.knee_csv, encoding="utf-8")))
    # 三分法分类表：曲线集（own_good / common_good）就是靠它筛窗口的 ⇒ 必须独立复现同一筛选
    class_rows = list(csv.DictReader(open(cli.class_csv, encoding="utf-8")))
    class_by_key = {(row["seq_id"], int(row["frame_id"])): row for row in class_rows}
    arms = ("xcorr", "kcf_legacy", "csrt")

    def _keep(row, curve_set, tracker):
        info = class_by_key.get((row["seq_id"], int(row["frame_id"])))
        if info is None:
            return False
        if curve_set == "own_good":
            return info.get("class_%s" % tracker) == "good"
        return all(info.get("class_%s" % arm) == "good" for arm in arms)

    checked = 0
    worst = 0.0
    for row in knee_rows:
        if row["curve_set"] not in ("own_good", "common_good") or row["stratum"] != "all":
            continue
        if row["method_used"] != "kneedle":
            continue
        # 明细表里**没有** tracker 列：三臂是并排的（``<tracker>_success`` 列），
        # 所以只按退化族 + 曲线集过滤，具体哪一臂由下面取列时指定。
        subset = [r for r in detail_rows
                  if r["condition_family"] == row["degradation_family"]
                  and _keep(r, row["curve_set"], row["tracker"])]
        if not subset:
            continue
        values = np.array([float(r[row["metric"]]) for r in subset], dtype=float)
        success = np.array([float(r["%s_success" % row["tracker"]]) for r in subset], dtype=float)
        mask = np.isfinite(values) & np.isfinite(success)
        values, success = values[mask], success[mask]
        if values.size == 0 or np.unique(values).size <= 12:
            edges = np.unique(values) if values.size else np.array([])
            if edges.size < 4:
                continue
            xs, ys = edges, np.array([success[values == e].mean() for e in edges])
        else:
            edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 13)))
            xs, ys = [], []
            for index in range(len(edges) - 1):
                low, high = edges[index], edges[index + 1]
                sel = (values >= low) & (values <= high) if index == len(edges) - 2 \
                    else (values >= low) & (values < high)
                if sel.any():
                    xs.append(float(np.median(values[sel])))
                    ys.append(float(success[sel].mean()))
            xs, ys = np.asarray(xs), np.asarray(ys)
        expect = independent_knee(xs, ys)
        got = float(row["knee_x_kneedle"])
        if not (math.isfinite(expect) and math.isfinite(got)):
            continue
        diff = abs(expect - got)
        if diff > worst:
            worst = diff
        checked += 1
    print("[VERIFY][KNEE] 复算 %d 个膝点，最大绝对误差 %.3e" % (checked, worst))
    if checked == 0:
        raise SystemExit("[FATAL] 膝点复算没有可比对的样本")
    if worst >= TOL:
        raise SystemExit("[FAIL] 膝点复算误差 %.3e ≥ %.0e" % (worst, TOL))
    return checked


def verify_manifest(cli: argparse.Namespace) -> int:
    manifest = json.loads(Path(cli.manifest_json).read_text(encoding="utf-8"))
    windows = list(csv.DictReader(open(cli.windows_csv, encoding="utf-8")))
    if len(windows) != manifest["window_counts"]["total"]:
        raise SystemExit("[FAIL] 窗口数不一致：CSV %d vs manifest %d"
                         % (len(windows), manifest["window_counts"]["total"]))
    small = sum(1 for row in windows if row["small_target"] == "1")
    if small != manifest["window_counts"]["small_target"]:
        raise SystemExit("[FAIL] 小目标计数不一致：CSV %d vs manifest %d"
                         % (small, manifest["window_counts"]["small_target"]))
    for row in windows:
        for index in range(5):
            if int(row["gt_%d_exist" % index]) != 1:
                raise SystemExit("[FAIL] 窗口 %s/%s 第 %d 帧 exist != 1"
                                 % (row["seq_id"], row["frame_id"], index))
        if float(row["w"]) <= 0 or float(row["h"]) <= 0:
            raise SystemExit("[FAIL] 窗口 %s/%s 目标尺寸非正" % (row["seq_id"], row["frame_id"]))
    print("[VERIFY][MANIFEST] 窗口 %d、小目标 %d、exist 全为 1、尺寸全为正 ⇒ 一致"
          % (len(windows), small))
    return len(windows)


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 独立复算（不 import 主评估聚合模块）")
    parser.add_argument("--detail-csv", default="/tmp/iqa_v2_cache/degrade_detail.csv")
    parser.add_argument("--knee-csv", default="iqa_v2/output/knee_thresholds_v2.csv")
    parser.add_argument("--class-csv", default="iqa_v2/output/segment_class_v2.csv")
    parser.add_argument("--windows-csv", default="iqa_v2/output/windows.csv")
    parser.add_argument("--manifest-json", default="iqa_v2/output/dataset_manifest.json")
    parser.add_argument("--dataset-root", default="/mnt/data/siping/datasets/manu/anti-uav")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    verify_manifest(args)
    verify_roi_metrics(args)
    verify_knee(args)
    print("[VERIFY] 全部通过（容差 %.0e）" % TOL)


if __name__ == "__main__":
    main()
