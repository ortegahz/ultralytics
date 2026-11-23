#!/usr/bin/env python3
"""第 3~5 步：退化视频的双路同步测量 → 物理崩溃拐点（Knee Point）→ 盲测。

四个子命令对应四份产物：

``measure``
    第 3 步的 **X 路**：对每个退化变体用 ``vq_metrics.analyze_video`` + ``vq_stats.window_values``
    算「裁黑边 + 640px 归一 + 逐 1 秒窗口」的画质指标（**完全复用现有口径**，不另写一套）
    ⇒ ``report/degrade_windows.csv``

``curve``
    第 3 步的 **Y 路**：复用 ``vq_sot_link_driver.py`` 的探针结果，用
    ``vq_sot_compare.py::load_clicks`` / ``aggregate_windows`` 重算逐 1 秒窗口的
    锁定率 UGR（**short = 5 帧 / long = 1 秒两个时程都算**），与 X 路**逐窗口对齐**
    ⇒ ``report/degrade_lock.csv`` + ``report/degrade_curve.csv``

``knee``
    第 4 步：分模态提取**物理崩溃拐点**（Kneedle 与 PCHIP 最大曲率两法交叉验证）
    ⇒ ``report/knee_thresholds.csv``

``blindtest``
    第 5 步：4 条真实坏片（未参与标定）+ 18 条好片回测，产出新旧阈值对比表
    ⇒ ``report/knee_blindtest.csv``（另出图见 ``charts``）

⚠️ **纹理地板 τ_det 的口径**：`calibrate_tau` 只在**好片**上标定（坏片/退化片绝不参与），
所以 τ_det 必须用**好片的点击**标定，再套到退化变体上；直接拿退化片标会得到空集 ⇒ UGR 全 0。

用法（服务器 192.168.99.40:32222，工作目录 /tmp/pycharm_project_a7d966e6）：
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
    python manu/video_quality/vq_degrade_knee.py measure  --variants /tmp/vq_degrade/variants.csv \\
        --out manu/video_quality/report/degrade_windows.csv --workers 8
    python manu/video_quality/vq_degrade_knee.py curve    --variants /tmp/vq_degrade/variants.csv \\
        --windows manu/video_quality/report/degrade_windows.csv \\
        --sot-dir /tmp/vq_degrade/sot_out --tau-dir /tmp/sot_out \\
        --lock-out manu/video_quality/report/degrade_lock.csv \\
        --out manu/video_quality/report/degrade_curve.csv
    python manu/video_quality/vq_degrade_knee.py knee     --curve manu/video_quality/report/degrade_curve.csv \\
        --out manu/video_quality/report/knee_thresholds.csv
    python manu/video_quality/vq_degrade_knee.py blindtest --knee manu/video_quality/report/knee_thresholds.csv \\
        --sot-windows manu/video_quality/report/sot_windows.csv \\
        --out manu/video_quality/report/knee_blindtest.csv
    python manu/video_quality/vq_degrade_knee.py charts    --knee manu/video_quality/report/knee_thresholds.csv \\
        --curve manu/video_quality/report/degrade_curve.csv \\
        --windows manu/video_quality/report/degrade_windows.csv \\
        --lock manu/video_quality/report/degrade_lock.csv \\
        --chart-dir manu/video_quality/report
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for _path in (str(PROJECT_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from vq_metrics import METRIC_DIRECTION, analyze_video  # noqa: E402
from vq_stats import window_values  # noqa: E402

# §9.5 用的 8 个候选指标（其中 5 个有区分度）
KEY_METRICS = (
    "lap_var", "edge_density", "global_contrast", "blockiness", "entropy",
    "snr_db", "noise_sigma", "dog_snr_max",
)
# 目标书 §13.2 第 4 步指定的分模态指标集
MODALITY_METRICS = {
    "ir": ("global_contrast", "edge_density", "lap_var", "blockiness", "entropy"),
    "vis": ("global_contrast", "edge_density", "lap_var"),
}
# 两个时程：short = 5 帧（§9.5.8 推荐的「上线以短时程为准」），long = 1 秒（30 帧）
HORIZONS = ("short", "long")
HORIZON_LABEL = {"short": "short(5帧)", "long": "long(1秒)"}
# 「跟得住」的判定：逐秒窗口 UGR >= 0.5（与 §9.5 完全一致）
LOCK_UGR_THRESHOLD = 0.5
# X 轴分箱数（把散点压成一条可拟合的曲线）
N_BINS = 40
# 两法（Kneedle / PCHIP 曲率）差值超过该百分比 ⇒ 曲线噪声过大，膝点只作物理参照
KNEE_AGREE_PCT = 15.0


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def fnum(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


# ---------------------------------------------------------------- phase 1: measure
def measure_one(job: tuple[dict, list[str]]) -> list[dict]:
    variant, metrics = job
    path = Path(variant["video_path"])
    meta, arrays = analyze_video(path, stride=1, max_frames=0)
    per_metric = {metric: window_values(meta, arrays, metric) for metric in metrics}
    n_windows = max((values.size for values in per_metric.values()), default=0)
    rows = []
    for index in range(int(n_windows)):
        row = {
            "variant_id": variant["variant_id"],
            "source_clip": variant["source_clip"],
            "modality": variant["modality"],
            "gradient": variant["gradient"],
            "level": variant["level"],
            "level_value": variant["level_value"],
            "window": index,
            "fps_used": "%.6f" % meta.get("fps_used", 0.0),
            "frames_analysed": meta.get("frames_analysed", 0),
            "content_size": "%dx%d" % tuple(meta["content_size"]),
        }
        for metric in metrics:
            values = per_metric[metric]
            row[metric] = round(float(values[index]), 6) if index < values.size else float("nan")
        rows.append(row)
    return rows


def phase_measure(cli: argparse.Namespace) -> None:
    variants = read_csv(Path(cli.variants))
    print("[MEASURE] %d variants" % len(variants), flush=True)
    jobs = [(variant, list(KEY_METRICS)) for variant in variants]
    rows: list[dict] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        futures = [pool.submit(measure_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if done % 20 == 0 or done == len(jobs):
                print("[MEASURE] %d/%d variants  %.0fs" % (done, len(jobs), time.perf_counter() - started), flush=True)
    rows.sort(key=lambda row: (row["source_clip"], row["gradient"], row["level"], int(row["window"])))
    fields = ["variant_id", "source_clip", "modality", "gradient", "level", "level_value", "window",
              "fps_used", "frames_analysed", "content_size"] + list(KEY_METRICS)
    write_csv(Path(cli.out), rows, fields)
    print("[MEASURE] %d window rows -> %s" % (len(rows), cli.out))


# ---------------------------------------------------------------- phase 2: curve
def load_sot_window_rows(sot_dir: Path, tau_clicks: list[dict], taus: dict[str, float]) -> dict[tuple[str, int], dict]:
    """复用 ``vq_sot_compare.py`` 的加载与聚合口径，取逐窗 UGR（center+corner 合并）。"""
    sys.path.insert(0, str(HERE))
    from vq_sot_compare import aggregate_windows, load_clicks

    clicks, _ = load_clicks([sot_dir])
    rows = aggregate_windows(clicks, taus)
    merged: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["file"], int(row["window"]))
        current = merged.get(key)
        if current is None:
            current = {
                "n_clicks": 0,
                "ugr": {h: 0.0 for h in HORIZONS},
                "lock": {h: 0.0 for h in HORIZONS},
                "tau_det": row["tau_det"],
                "modality": row["modality"],
            }
            merged[key] = current
        current["n_clicks"] += row["n_clicks"]
        for horizon in HORIZONS:
            current["ugr"][horizon] += row["xcorr_%s_ugr" % horizon] * row["n_clicks"]
            current["lock"][horizon] += row["xcorr_%s_lock" % horizon] * row["n_clicks"]
    for current in merged.values():
        total = max(current["n_clicks"], 1)
        for horizon in HORIZONS:
            current["ugr"][horizon] /= total
            current["lock"][horizon] /= total
    return merged


def phase_curve(cli: argparse.Namespace) -> None:
    sys.path.insert(0, str(HERE))
    from vq_sot_compare import calibrate_tau, load_clicks

    variants = read_csv(Path(cli.variants))
    windows = read_csv(Path(cli.windows))
    tau_clicks, _ = load_clicks([Path(cli.tau_dir)])
    taus = calibrate_tau(tau_clicks)
    print("[CURVE] tau_det from %s: %s" % (cli.tau_dir, {k: round(v, 4) for k, v in taus.items()}))
    sot = load_sot_window_rows(Path(cli.sot_dir), tau_clicks, taus)
    print("[CURVE] %d window rows, %d SOT windows" % (len(windows), len(sot)), flush=True)

    meta_by_variant = {variant["variant_id"]: variant for variant in variants}
    # SOT 探针的 summary["video"] 是**带扩展名的文件名**，variant_id 是去扩展名的 stem ⇒ 要映射
    sot_key_by_variant = {
        variant["variant_id"]: Path(variant["video_path"]).name for variant in variants
    }
    lock_rows: list[dict] = []
    missing = 0
    for row in windows:
        if row["variant_id"] not in meta_by_variant:
            continue
        entry = sot.get((sot_key_by_variant.get(row["variant_id"], ""), int(row["window"])))
        if entry is None:
            missing += 1
            continue
        lock_rows.append({
            "variant_id": row["variant_id"],
            "source_clip": row["source_clip"],
            "modality": row["modality"],
            "gradient": row["gradient"],
            "level": row["level"],
            "level_value": row["level_value"],
            "window": row["window"],
            "n_clicks": entry["n_clicks"],
            "tau_det": entry["tau_det"],
            "xcorr_short_ugr": round(entry["ugr"]["short"], 6),
            "xcorr_short_lock": round(entry["lock"]["short"], 6),
            "xcorr_long_ugr": round(entry["ugr"]["long"], 6),
            "xcorr_long_lock": round(entry["lock"]["long"], 6),
            "locked_short": 1 if entry["ugr"]["short"] >= LOCK_UGR_THRESHOLD else 0,
            "locked_long": 1 if entry["ugr"]["long"] >= LOCK_UGR_THRESHOLD else 0,
        })
    if missing:
        print("[WARN] %d quality windows had no matching SOT window (warm-up window 0 等)" % missing)
    lock_fields = ["variant_id", "source_clip", "modality", "gradient", "level", "level_value",
                   "window", "n_clicks", "tau_det", "xcorr_short_ugr", "xcorr_short_lock",
                   "locked_short", "xcorr_long_ugr", "xcorr_long_lock", "locked_long"]
    write_csv(Path(cli.lock_out), lock_rows, lock_fields)
    print("[CURVE] %d aligned window rows -> %s" % (len(lock_rows), cli.lock_out))

    quality_by_key = {(row["variant_id"], int(row["window"])): row for row in windows}
    curve_rows: list[dict] = []
    for modality, metrics in MODALITY_METRICS.items():
        pairs = [(row, quality_by_key[(row["variant_id"], int(row["window"]))])
                 for row in lock_rows if row["modality"] == modality]
        for horizon in HORIZONS:
            for metric in metrics:
                family: dict[tuple[str, str], list[tuple[float, float]]] = {}
                level_value_by_key: dict[tuple[str, str], str] = {}
                pooled: list[tuple[float, float]] = []
                for lock_row, quality_row in pairs:
                    value = fnum(quality_row[metric])
                    if not math.isfinite(value):
                        continue
                    y = float(lock_row["xcorr_%s_ugr" % horizon])
                    family.setdefault((lock_row["gradient"], lock_row["level"]), []).append((value, y))
                    level_value_by_key[(lock_row["gradient"], lock_row["level"])] = lock_row["level_value"]
                    pooled.append((value, y))
                for (gradient, level), points in sorted(family.items()):
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    curve_rows.append({
                        "scope": "family", "horizon": horizon, "modality": modality, "metric": metric,
                        "gradient": gradient, "level": level,
                        "level_value": level_value_by_key.get((gradient, level), ""),
                        "n_windows": len(points),
                        "x_median": round(float(np.median(xs)), 6),
                        "y_ugr_mean": round(float(np.mean(ys)), 6),
                        "y_lock_rate": round(float(np.mean([1.0 if v >= LOCK_UGR_THRESHOLD else 0.0 for v in ys])), 6),
                    })
                pooled.sort(key=lambda p: p[0])
                if len(pooled) >= N_BINS:
                    for index, chunk in enumerate(np.array_split(np.arange(len(pooled)), N_BINS)):
                        if chunk.size == 0:
                            continue
                        xs = [pooled[i][0] for i in chunk]
                        ys = [pooled[i][1] for i in chunk]
                        curve_rows.append({
                            "scope": "binned", "horizon": horizon, "modality": modality, "metric": metric,
                            "gradient": "__all__", "level": "bin%02d" % index, "level_value": "",
                            "n_windows": len(chunk),
                            "x_median": round(float(np.median(xs)), 6),
                            "y_ugr_mean": round(float(np.mean(ys)), 6),
                            "y_lock_rate": round(float(np.mean([1.0 if v >= LOCK_UGR_THRESHOLD else 0.0 for v in ys])), 6),
                        })
    curve_fields = ["scope", "horizon", "modality", "metric", "gradient", "level", "level_value",
                    "n_windows", "x_median", "y_ugr_mean", "y_lock_rate"]
    write_csv(Path(cli.out), curve_rows, curve_fields)
    print("[CURVE] %d curve rows -> %s" % (len(curve_rows), cli.out))


# ---------------------------------------------------------------- phase 3: knee
def _to_original(value: float, direction: int) -> float:
    """``prepare_curve`` 里对「越小越好」的指标把 X 取了负，这里换回原始量纲。"""
    return -float(value) if direction > 0 else float(value)


def curve_y_at(xs: np.ndarray, ys: np.ndarray, direction: int, query: float) -> float:
    """在**平滑后的**曲线上插值 Y（避免用带噪的分箱原始点）。"""
    from scipy.interpolate import PchipInterpolator

    curve_x, curve_y = prepare_curve(xs, ys, direction)
    if curve_x.size < 4:
        return float("nan")
    spline = PchipInterpolator(curve_x, curve_y)
    return float(spline(-query if direction > 0 else query))


def prepare_curve(xs: np.ndarray, ys: np.ndarray, direction: int) -> tuple[np.ndarray, np.ndarray]:
    """把 (X, Y) 整理成「X 越大越好、Y 单调不减」的曲线，供两种膝点算法共用。

    - ``direction = -1``（越大越好）：X 不变；
    - ``direction = +1``（越小越好，如 blockiness）：**X 取负**，关系才变成单调不减；
    - 分箱中位数本身带噪 ⇒ 先做**保序回归**（isotonic）单调化，再按 X 合并重复点。
    """
    from scipy.optimize import isotonic_regression

    order = np.argsort(xs, kind="mergesort")
    xs, ys = xs[order], ys[order]
    if direction > 0:
        # 「越小越好」的指标：把 X 取负并**同步反转** Y，否则 X/Y 配对会被打乱
        xs = -xs[::-1]
        ys = ys[::-1]
    if xs.size == 0:
        return xs, ys
    fitted = np.asarray(isotonic_regression(ys, increasing=True).x, dtype=float)
    unique_x, start = np.unique(xs, return_index=True)
    unique_y = np.maximum.reduceat(fitted, start)
    return unique_x, unique_y


def kneedle_knee(xs: np.ndarray, ys: np.ndarray, direction: int = -1) -> tuple[float, float]:
    """Kneedle：X/Y 各自归一到 [0,1]，取**到首尾连线距离最大**的点。

    曲线先按 ``prepare_curve`` 整理成「X 越大越好、Y 单调不减」，
    所以 ``|yn - xn|`` 的最大点就是真正的拐点（S 曲线的拐点必在弦的某一侧）。
    """
    curve_x, curve_y = prepare_curve(xs, ys, direction)
    x_span = curve_x.max() - curve_x.min()
    y_span = curve_y.max() - curve_y.min()
    if curve_x.size < 3 or x_span <= 0 or y_span <= 0:
        return float("nan"), float("nan")
    xn = (curve_x - curve_x.min()) / x_span
    yn = (curve_y - curve_y.min()) / y_span
    index = int(np.argmax(np.abs(yn - xn)))
    return float(_to_original(curve_x[index], direction)), float(curve_y[index])


def pchip_knee(xs: np.ndarray, ys: np.ndarray, direction: int = -1) -> tuple[float, float]:
    """PCHIP 样条二阶导极值（曲率最大处）——与 :func:`kneedle_knee` 交叉验证。"""
    from scipy.interpolate import PchipInterpolator

    curve_x, curve_y = prepare_curve(xs, ys, direction)
    if curve_x.size < 4:
        return float("nan"), float("nan")
    spline = PchipInterpolator(curve_x, curve_y)
    dense = np.linspace(curve_x[0], curve_x[-1], 1500)
    second = spline.derivative(2)(dense)
    index = int(np.argmax(np.abs(second)))
    return float(_to_original(dense[index], direction)), float(spline(dense[index]))


def curve_slope_knee(xs: np.ndarray, ys: np.ndarray, direction: int = -1) -> tuple[float, float]:
    """参考量：拟合曲线上**斜率绝对值最大**处（最陡的崩塌段）。"""
    from scipy.interpolate import PchipInterpolator

    curve_x, curve_y = prepare_curve(xs, ys, direction)
    if curve_x.size < 4:
        return float("nan"), float("nan")
    spline = PchipInterpolator(curve_x, curve_y)
    dense = np.linspace(curve_x[0], curve_x[-1], 1500)
    index = int(np.argmax(np.abs(spline.derivative(1)(dense))))
    return float(_to_original(dense[index], direction)), float(spline(dense[index]))


def curve_y50(xs: np.ndarray, ys: np.ndarray, direction: int = -1) -> float:
    """参考量：拟合曲线上锁定率**降到 0.5** 的 X（§9.5 的「跟得住」判定线）。"""
    from scipy.interpolate import PchipInterpolator

    curve_x, curve_y = prepare_curve(xs, ys, direction)
    if curve_x.size < 4:
        return float("nan")
    spline = PchipInterpolator(curve_x, curve_y)
    dense = np.linspace(curve_x[0], curve_x[-1], 1500)
    values = spline(dense)
    hit = np.nonzero(values >= LOCK_UGR_THRESHOLD)[0]
    if hit.size == 0:
        return float("nan")
    return float(_to_original(dense[hit[0]], direction))


def phase_knee(cli: argparse.Namespace) -> None:
    rows = read_csv(Path(cli.curve))
    out_rows: list[dict] = []
    for horizon in HORIZONS:
        for modality, metrics in MODALITY_METRICS.items():
            for metric in metrics:
                binned = [row for row in rows
                          if row["scope"] == "binned" and row["horizon"] == horizon
                          and row["modality"] == modality and row["metric"] == metric]
                binned.sort(key=lambda row: float(row["x_median"]))
                xs = np.array([float(row["x_median"]) for row in binned], dtype=float)
                ys = np.array([float(row["y_ugr_mean"]) for row in binned], dtype=float)
                if xs.size < 6:
                    print("[KNEE] skip %s/%s/%s (only %d points)" % (horizon, modality, metric, xs.size))
                    continue
                direction = METRIC_DIRECTION[metric]
                x_needle, y_needle = kneedle_knee(xs, ys, direction)
                x_pchip, y_pchip = pchip_knee(xs, ys, direction)
                x_steep, y_steep = curve_slope_knee(xs, ys, direction)
                x_y50 = curve_y50(xs, ys, direction)
                if not math.isfinite(x_needle):
                    x_needle = float(np.median(xs))
                    y_needle = curve_y_at(xs, ys, direction, x_needle)
                if not math.isfinite(x_pchip):
                    x_pchip, y_pchip = x_needle, y_needle
                # 两法不一致时取「更靠近好片一侧」的那个：
                #   越大越好 → 取更大的 X（阈值更高 = 更靠近好片的画质水平）
                #   越小越好 → 取更小的 X
                if direction < 0:
                    x_final = max(x_needle, x_pchip)
                    method = "kneedle" if x_needle >= x_pchip else "pchip"
                else:
                    x_final = min(x_needle, x_pchip)
                    method = "kneedle" if x_needle <= x_pchip else "pchip"
                y_final = curve_y_at(xs, ys, direction, x_final)
                span = max(float(xs.max() - xs.min()), 1e-9)
                delta_pct = abs(x_needle - x_pchip) / span * 100.0
                # 两法差 > 15% 量程 ⇒ 曲线噪声过大，膝点只作「物理参照」，不作准入线
                reliable = 1 if delta_pct <= KNEE_AGREE_PCT else 0
                family_knees: dict[str, tuple[float, float]] = {}
                for gradient in ("blur", "contrast", "crf"):
                    sub = [row for row in rows if row["scope"] == "family" and row["horizon"] == horizon
                           and row["modality"] == modality and row["metric"] == metric
                           and row["gradient"] == gradient]
                    if len(sub) < 4:
                        continue
                    sub.sort(key=lambda row: float(row["x_median"]))
                    fx = np.array([float(row["x_median"]) for row in sub], dtype=float)
                    fy = np.array([float(row["y_ugr_mean"]) for row in sub], dtype=float)
                    fn, _ = kneedle_knee(fx, fy, direction)
                    fp, _ = pchip_knee(fx, fy, direction)
                    family_knees[gradient] = (fn, fp)
                out_rows.append({
                    "horizon": horizon,
                    "modality": modality,
                    "metric": metric,
                    "direction": direction,
                    "operator": ">=" if direction < 0 else "<=",
                    "knee_x_kneedle": round(x_needle, 6),
                    "knee_y_kneedle": round(y_needle, 6),
                    "knee_x_pchip": round(x_pchip, 6),
                    "knee_y_pchip": round(y_pchip, 6),
                    "knee_x_final": round(x_final, 6),
                    "knee_y_final": round(y_final, 6),
                    "method_used": method,
                    "delta_between_methods": round(abs(x_needle - x_pchip), 6),
                    "delta_pct_of_span": round(delta_pct, 3),
                    "knee_reliable": reliable,
                    "knee_x_steep": round(x_steep, 6),
                    "knee_y_steep": round(y_steep, 6),
                    "x_at_y50": round(x_y50, 6),
                    "n_curve_points": len(binned),
                    "knee_x_blur_kneedle": round(family_knees.get("blur", (float("nan"),))[0], 6),
                    "knee_x_blur_pchip": round(family_knees.get("blur", (float("nan"), float("nan")))[1], 6),
                    "knee_x_contrast_kneedle": round(family_knees.get("contrast", (float("nan"),))[0], 6),
                    "knee_x_contrast_pchip": round(family_knees.get("contrast", (float("nan"), float("nan")))[1], 6),
                    "knee_x_crf_kneedle": round(family_knees.get("crf", (float("nan"),))[0], 6),
                    "knee_x_crf_pchip": round(family_knees.get("crf", (float("nan"), float("nan")))[1], 6),
                    "x_curve_min": round(float(xs.min()), 6),
                    "x_curve_max": round(float(xs.max()), 6),
                    "y_curve_min": round(float(ys.min()), 6),
                    "y_curve_max": round(float(ys.max()), 6),
                })
                print("[KNEE] %-5s %-4s %-16s knee=%-12.6g (%s)  needle=%-12.6g pchip=%-12.6g  "
                      "steep=%-12.6g y50=%-12.6g Y=%0.3f  delta=%.1f%% %s"
                      % (horizon, modality, metric, x_final, method, x_needle, x_pchip, x_steep, x_y50,
                         y_final, delta_pct, "" if reliable else "(双法分歧大→只作参照)"), flush=True)
    fields = ["horizon", "modality", "metric", "direction", "operator",
              "knee_x_kneedle", "knee_y_kneedle", "knee_x_pchip", "knee_y_pchip",
              "knee_x_final", "knee_y_final", "method_used", "delta_between_methods",
              "delta_pct_of_span", "knee_reliable", "knee_x_steep", "knee_y_steep", "x_at_y50",
              "n_curve_points",
              "knee_x_blur_kneedle", "knee_x_blur_pchip", "knee_x_contrast_kneedle",
              "knee_x_contrast_pchip", "knee_x_crf_kneedle", "knee_x_crf_pchip",
              "x_curve_min", "x_curve_max", "y_curve_min", "y_curve_max"]
    write_csv(Path(cli.out), out_rows, fields)
    print("[KNEE] %d thresholds -> %s" % (len(out_rows), cli.out))


# ---------------------------------------------------------------- phase 4: blind test
def merge_sot_windows(path: Path) -> list[dict]:
    """把 ``sot_windows.csv`` 的 center/corner 两行合并成一个逐窗记录。"""
    rows = read_csv(path)
    merged: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["file"], int(row["window"]))
        current = merged.get(key)
        if current is None:
            current = {
                "file": row["file"], "group": row["group"], "kind": row["kind"],
                "modality": row["modality"], "window": int(row["window"]),
                "n_clicks": 0, "metric": {},
                "ugr": {h: 0.0 for h in HORIZONS},
            }
            merged[key] = current
        clicks = int(row["n_clicks"])
        current["n_clicks"] += clicks
        for horizon in HORIZONS:
            current["ugr"][horizon] += float(row["xcorr_%s_ugr" % horizon]) * clicks
        for metric in KEY_METRICS:
            current["metric"][metric] = fnum(row[metric])
    out = []
    for current in merged.values():
        for horizon in HORIZONS:
            current["ugr"][horizon] /= max(current["n_clicks"], 1)
        out.append(current)
    return out


def passes(value: float, threshold: float, direction: int) -> bool:
    if not math.isfinite(value) or not math.isfinite(threshold):
        return False
    return value >= threshold if direction < 0 else value <= threshold


def clip_level_reject(clips: dict[str, list[bool]]) -> tuple[int, int]:
    """片级判定：某片的逐窗**过线率 < 50%** ⇒ 该片判为「不合格」。"""
    rejected = 0
    for flags in clips.values():
        if not flags:
            continue
        if sum(1 for ok in flags if ok) / len(flags) < 0.5:
            rejected += 1
    return rejected, len(clips)


def phase_blindtest(cli: argparse.Namespace) -> None:
    knees = read_csv(Path(cli.knee))
    windows = merge_sot_windows(Path(cli.sot_windows))
    good = [row for row in windows if row["group"] == "good" and row["kind"] == "source"]
    bad = [row for row in windows if row["group"] == "bad" and row["kind"] == "source"]
    print("[BLIND] good windows=%d bad windows=%d" % (len(good), len(bad)))

    lines: list[dict] = []
    for row in knees:
        lines.append({
            "line_type": "new_knee", "arm": row["horizon"], "horizon": row["horizon"],
            "modality": row["modality"], "metric": row["metric"],
            "direction": int(row["direction"]), "threshold": float(row["knee_x_final"]),
            "source": "knee_thresholds.csv",
        })
    for arm, directory in (("long", cli.long_thresholds), ("short", cli.short_thresholds)):
        directory = Path(directory)
        if not directory.exists():
            continue
        for scope_file, scope in (("sot_thresholds.csv", "all"),
                                  ("sot_thresholds_ir.csv", "ir"),
                                  ("sot_thresholds_vis.csv", "vis")):
            target = directory / scope_file
            if not target.exists():
                continue
            for row in read_csv(target):
                metric = row["metric"]
                if metric not in KEY_METRICS:
                    continue
                # 注意：``raw_threshold`` 是**带符号**的（越低越好的指标为负），
                # 绝对值口径要用 ``metric_threshold``；admit/rec/safe 三列本身就是绝对值。
                for kind, column in (("old_admission", "admit_threshold"),
                                      ("old_rec", "rec_threshold"),
                                      ("old_safe", "safe_threshold"),
                                      ("old_gap", "metric_threshold")):
                    value = fnum(row.get(column, ""))
                    if not math.isfinite(value):
                        continue
                    lines.append({
                        "line_type": kind, "arm": arm, "horizon": arm,
                        "modality": scope, "metric": metric,
                        "direction": METRIC_DIRECTION[metric], "threshold": value,
                        "source": "%s/%s" % (directory.name, scope_file),
                    })

    out_rows: list[dict] = []
    for line in lines:
        modality = line["modality"]
        scope_note = "pooled" if modality == "all" else ""
        if modality == "all":
            modality = "ir"          # 混池线跨模态禁用，这里只按本模态评估
        horizon = line["horizon"]
        good_rows = [row for row in good if row["modality"] == modality]
        bad_rows = [row for row in bad if row["modality"] == modality]
        metric = line["metric"]
        threshold = line["threshold"]
        direction = line["direction"]

        def evaluate(rows_: list[dict]) -> dict:
            above, below = [], []
            for row in rows_:
                ok = passes(row["metric"].get(metric, float("nan")), threshold, direction)
                locked = 1 if row["ugr"][horizon] >= LOCK_UGR_THRESHOLD else 0
                (above if ok else below).append(locked)
            return {
                "n_above": len(above), "n_below": len(below),
                "tpr": (sum(above) / len(above)) if above else float("nan"),
                "fpr": (sum(above) / len(above)) if above else float("nan"),
                "n_total": len(rows_),
            }

        good_eval = evaluate(good_rows)
        bad_eval = evaluate(bad_rows)
        # 片级判定用的是「**过线**标志」：过线率 < 50% ⇒ 该片判为不合格
        good_pass_flags = [passes(row["metric"].get(metric, float("nan")), threshold, direction)
                           for row in good_rows]
        bad_pass_flags = [passes(row["metric"].get(metric, float("nan")), threshold, direction)
                          for row in bad_rows]
        good_clips: dict[str, list[bool]] = {}
        for row, ok in zip(good_rows, good_pass_flags):
            good_clips.setdefault(row["file"], []).append(ok)
        bad_clips: dict[str, list[bool]] = {}
        for row, ok in zip(bad_rows, bad_pass_flags):
            bad_clips.setdefault(row["file"], []).append(ok)
        good_clip_rejected, n_good_clips = clip_level_reject(good_clips)
        bad_clip_rejected, n_bad_clips = clip_level_reject(bad_clips)
        out_rows.append({
            "line_type": line["line_type"],
            "arm": line["arm"],
            "horizon": horizon,
            "modality": modality,
            "metric": metric,
            "operator": ">=" if direction < 0 else "<=",
            "threshold": round(threshold, 6),
            "n_good_windows": good_eval["n_total"],
            "good_window_reject_pct": round(100.0 * (good_eval["n_total"] - good_eval["n_above"])
                                            / max(good_eval["n_total"], 1), 3),
            "good_above_line_n": good_eval["n_above"],
            "good_above_line_tpr": round(good_eval["tpr"], 4) if good_eval["n_above"] else "",
            "n_bad_windows": bad_eval["n_total"],
            "bad_window_reject_pct": round(100.0 * (bad_eval["n_total"] - bad_eval["n_above"])
                                           / max(bad_eval["n_total"], 1), 3),
            "bad_above_line_n": bad_eval["n_above"],
            "bad_above_line_fpr": round(bad_eval["fpr"], 4) if bad_eval["n_above"] else "",
            "n_good_clips": n_good_clips,
            "good_clip_reject_pct": round(100.0 * good_clip_rejected / max(n_good_clips, 1), 3),
            "n_bad_clips": n_bad_clips,
            "bad_clip_reject_pct": round(100.0 * bad_clip_rejected / max(n_bad_clips, 1), 3),
            "scope_note": scope_note,
            "source": line["source"],
        })
    fields = ["line_type", "arm", "horizon", "modality", "metric", "operator", "threshold",
              "n_good_windows", "good_window_reject_pct", "good_above_line_n", "good_above_line_tpr",
              "n_bad_windows", "bad_window_reject_pct", "bad_above_line_n", "bad_above_line_fpr",
              "n_good_clips", "good_clip_reject_pct", "n_bad_clips", "bad_clip_reject_pct",
              "scope_note", "source"]
    write_csv(Path(cli.out), out_rows, fields)
    print("[BLIND] %d line evaluations -> %s" % (len(out_rows), cli.out))


# ---------------------------------------------------------------- phase 5: combos
def phase_combo(cli: argparse.Namespace) -> None:
    """在**同一批盲测数据**上评估膝点线的单指标 / 组合判据，与旧线同口径对比。"""
    knees = read_csv(Path(cli.knee))
    windows = merge_sot_windows(Path(cli.sot_windows))
    good = [row for row in windows if row["group"] == "good" and row["kind"] == "source"]
    bad = [row for row in windows if row["group"] == "bad" and row["kind"] == "source"]
    lines: list[dict] = []
    for row in knees:
        lines.append({
            "line_id": "knee_%s_%s_%s" % (row["horizon"], row["modality"], row["metric"]),
            "line_type": "new_knee", "arm": row["horizon"], "horizon": row["horizon"],
            "modality": row["modality"], "metric": row["metric"],
            "direction": int(row["direction"]), "threshold": float(row["knee_x_final"]),
            "reliable": int(row.get("knee_reliable", 1)),
        })
    for arm, directory in (("long", cli.long_thresholds), ("short", cli.short_thresholds)):
        directory = Path(directory)
        if not directory.exists():
            continue
        for scope_file, scope in (("sot_thresholds.csv", "all"),
                                  ("sot_thresholds_ir.csv", "ir"),
                                  ("sot_thresholds_vis.csv", "vis")):
            target = directory / scope_file
            if not target.exists():
                continue
            for row in read_csv(target):
                metric = row["metric"]
                if metric not in KEY_METRICS:
                    continue
                for kind, column in (("old_admission", "admit_threshold"),
                                      ("old_rec", "rec_threshold"),
                                      ("old_safe", "safe_threshold")):
                    value = fnum(row.get(column, ""))
                    if not math.isfinite(value):
                        continue
                    lines.append({
                        "line_id": "old_%s_%s_%s_%s" % (kind, arm, scope, metric),
                        "line_type": kind, "arm": arm, "horizon": arm,
                        "modality": scope, "metric": metric,
                        "direction": METRIC_DIRECTION[metric], "threshold": value, "reliable": 1,
                    })

    def evaluate(rules: list[tuple[str, float]], horizon: str, modality: str) -> dict:
        good_rows = [row for row in good if row["modality"] == modality]
        bad_rows = [row for row in bad if row["modality"] == modality]

        def flags(rows: list[dict]) -> list[bool]:
            out = []
            for row in rows:
                ok = True
                for metric, threshold in rules:
                    value = row["metric"].get(metric, float("nan"))
                    ok = ok and passes(value, threshold, METRIC_DIRECTION[metric])
                out.append(ok)
            return out

        def locked(rows: list[dict]) -> list[int]:
            return [1 if row["ugr"][horizon] >= LOCK_UGR_THRESHOLD else 0 for row in rows]

        good_flags, good_lock = flags(good_rows), locked(good_rows)
        bad_flags, bad_lock = flags(bad_rows), locked(bad_rows)
        good_pass = [lock for flag, lock in zip(good_flags, good_lock) if flag]
        bad_pass = [lock for flag, lock in zip(bad_flags, bad_lock) if flag]
        good_clip_reject, n_good_clips = clip_level_reject(_group_flags(good_rows, good_flags))
        bad_clip_reject, n_bad_clips = clip_level_reject(_group_flags(bad_rows, bad_flags))
        return {
            "n_good_windows": len(good_rows),
            "good_window_reject_pct": round(100.0 * (len(good_rows) - len(good_pass)) / max(len(good_rows), 1), 3),
            "good_above_line_n": len(good_pass),
            "good_above_line_tpr": round(sum(good_pass) / max(len(good_pass), 1), 4),
            "n_bad_windows": len(bad_rows),
            "bad_window_reject_pct": round(100.0 * (len(bad_rows) - len(bad_pass)) / max(len(bad_rows), 1), 3),
            "bad_above_line_n": len(bad_pass),
            "bad_above_line_tpr": round(sum(bad_pass) / max(len(bad_pass), 1), 4),
            "n_good_clips": n_good_clips,
            "good_clip_reject_pct": round(100.0 * good_clip_reject / max(n_good_clips, 1), 3),
            "n_bad_clips": n_bad_clips,
            "bad_clip_reject_pct": round(100.0 * bad_clip_reject / max(n_bad_clips, 1), 3),
        }

    rows_out: list[dict] = []
    for horizon in HORIZONS:
        for modality in MODALITY_METRICS:
            singles = [line for line in lines
                       if line["line_type"] == "new_knee" and line["horizon"] == horizon
                       and line["modality"] == modality]
            combos: list[list[dict]] = [[line] for line in singles]
            for size in (2, 3):
                for group in itertools.combinations(singles, size):
                    combos.append(list(group))
            for group in combos:
                rules = [(line["metric"], line["threshold"]) for line in group]
                stats = evaluate(rules, horizon, modality)
                rows_out.append({
                    "line_id": " & ".join(line["line_id"] for line in group),
                    "line_type": "new_knee_combo" if len(group) > 1 else "new_knee",
                    "arm": horizon, "horizon": horizon, "modality": modality,
                    "n_metrics": len(group),
                    "rule": " AND ".join("%s %s %.6g" % (line["metric"],
                                                        ">=" if line["direction"] < 0 else "<=",
                                                        line["threshold"]) for line in group),
                    "all_reliable": min(line["reliable"] for line in group),
                    **stats,
                })
    # 旧「准入线」组合：同一 scope/arm 下 admit_exists=True 的全部指标取 AND
    for arm, directory in (("long", cli.long_thresholds), ("short", cli.short_thresholds)):
        directory = Path(directory)
        for scope in ("ir", "vis"):
            target = directory / ("sot_thresholds_%s.csv" % scope)
            if not target.exists():
                continue
            group = []
            for row in read_csv(target):
                if row.get("admit_exists", "").strip().lower() != "true":
                    continue
                value = fnum(row.get("admit_threshold", ""))
                if not math.isfinite(value):
                    continue
                group.append((row["metric"], value, METRIC_DIRECTION[row["metric"]]))
            if not group:
                continue
            stats = evaluate([(metric, value) for metric, value, _ in group], arm, scope)
            rows_out.append({
                "line_id": "old_admission_combo_%s_%s" % (arm, scope),
                "line_type": "old_admission_combo", "arm": arm, "horizon": arm, "modality": scope,
                "n_metrics": len(group),
                "rule": " AND ".join("%s %s %.6g" % (metric, ">=" if direction < 0 else "<=", value)
                                     for metric, value, direction in group),
                "all_reliable": 1, "scope_note": "", **stats,
            })

    # 旧「好片分位」组合（§9.5.8）：直接解析 sot_percentile_rules.csv 里的规则串
    for arm, directory in (("long", cli.long_thresholds), ("short", cli.short_thresholds)):
        target = Path(directory) / "sot_percentile_rules.csv"
        if not target.exists():
            continue
        for row in read_csv(target):
            percentile = row.get("good_percentile", "")
            scope = row.get("scope", "")
            if percentile != "P10" or scope not in ("ir", "vis"):
                continue
            group = []
            for metric, relation, value in re.findall(
                    r"([a-z_]+)\s*(>=|<=)\s*([0-9.eE+-]+)", row["rule"]):
                if metric not in METRIC_DIRECTION:
                    continue
                group.append((metric, float(value), 1 if relation == "<=" else -1))
            if not group:
                continue
            stats = evaluate([(metric, value) for metric, value, _ in group], arm, scope)
            rows_out.append({
                "line_id": "old_percentile_%s_%s_%s" % (arm, scope, percentile),
                "line_type": "old_percentile_combo", "arm": arm, "horizon": arm, "modality": scope,
                "n_metrics": len(group),
                "rule": row["rule"], "all_reliable": 1, "scope_note": "", **stats,
            })

    # 旧线（单指标）
    for line in lines:
        if line["line_type"] == "new_knee":
            continue
        modality = line["modality"]
        scope_note = "pooled" if modality == "all" else ""
        if modality == "all":
            modality = "ir"
        stats = evaluate([(line["metric"], line["threshold"])], line["horizon"], modality)
        rows_out.append({
            "line_id": line["line_id"], "line_type": line["line_type"], "arm": line["arm"],
            "horizon": line["horizon"], "modality": modality, "n_metrics": 1,
            "rule": "%s %s %.6g" % (line["metric"], ">=" if line["direction"] < 0 else "<=",
                                    line["threshold"]),
            "all_reliable": 1, "scope_note": scope_note, **stats,
        })
    fields = ["line_id", "line_type", "arm", "horizon", "modality", "n_metrics", "rule",
              "all_reliable", "n_good_windows", "good_window_reject_pct", "good_above_line_n",
              "good_above_line_tpr", "n_bad_windows", "bad_window_reject_pct", "bad_above_line_n",
              "bad_above_line_tpr", "n_good_clips", "good_clip_reject_pct", "n_bad_clips",
              "bad_clip_reject_pct", "scope_note"]
    write_csv(Path(cli.out), rows_out, fields)
    print("[COMBO] %d rule evaluations -> %s" % (len(rows_out), cli.out))


def _group_flags(rows: list[dict], flags: list[bool]) -> dict[str, list[bool]]:
    grouped: dict[str, list[bool]] = {}
    for row, flag in zip(rows, flags):
        grouped.setdefault(row["file"], []).append(flag)
    return grouped


def phase_charts(cli: argparse.Namespace) -> None:
    sys.path.insert(0, str(HERE))
    from vq_grid_charts import setup_fonts

    plt = setup_fonts()

    knees = read_csv(Path(cli.knee))
    curve = read_csv(Path(cli.curve))
    lock = read_csv(Path(cli.lock))
    quality = read_csv(Path(cli.windows))
    quality_by_key = {(row["variant_id"], int(row["window"])): row for row in quality}

    chart_dir = Path(cli.chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for modality, metrics in MODALITY_METRICS.items():
        for metric in metrics:
            fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6), dpi=130)
            plotted = False
            for axis, horizon in zip(axes, HORIZONS):
                row = next((r for r in knees if r["modality"] == modality
                            and r["metric"] == metric and r["horizon"] == horizon), None)
                if row is None:
                    axis.set_axis_off()
                    continue
                xs, ys = [], []
                for lock_row in lock:
                    if lock_row["modality"] != modality:
                        continue
                    quality_row = quality_by_key.get((lock_row["variant_id"], int(lock_row["window"])))
                    if quality_row is None:
                        continue
                    value = fnum(quality_row[metric])
                    if not math.isfinite(value):
                        continue
                    xs.append(value)
                    ys.append(float(lock_row["xcorr_%s_ugr" % horizon]))
                if not xs:
                    axis.set_axis_off()
                    continue
                plotted = True
                binned = sorted([r for r in curve if r["scope"] == "binned" and r["horizon"] == horizon
                                 and r["modality"] == modality and r["metric"] == metric],
                                key=lambda r: float(r["x_median"]))
                bx = np.array([float(r["x_median"]) for r in binned], dtype=float)
                by = np.array([float(r["y_ugr_mean"]) for r in binned], dtype=float)
                direction = METRIC_DIRECTION[metric]
                curve_x, curve_y = prepare_curve(bx, by, direction)
                axis.scatter(xs, ys, s=4, alpha=0.10, color="#4C78A8", label="退化变体逐 1s 窗口")
                if binned:
                    axis.plot(bx, by, color="#D62728", lw=0.9, alpha=0.55, label="分箱曲线（带噪）")
                if curve_x.size >= 4:
                    dense = np.linspace(curve_x.min(), curve_x.max(), 600)
                    from scipy.interpolate import PchipInterpolator
                    axis.plot(dense if direction < 0 else -dense,
                              PchipInterpolator(curve_x, curve_y)(dense),
                              color="#111111", lw=2.0, label="保序回归 + PCHIP 平滑")
                for key, color, label in (("knee_x_kneedle", "#2CA02C", "Kneedle 膝点"),
                                          ("knee_x_pchip", "#FF7F0E", "PCHIP 曲率膝点"),
                                          ("x_at_y50", "#9467BD", "锁定率 0.5 交点"),
                                          ("knee_x_final", "#000000", "最终采用膝点")):
                    value = fnum(row.get(key, ""))
                    if math.isfinite(value):
                        axis.axvline(value, color=color, ls="--", lw=1.3,
                                     label="%s = %.5g" % (label, value))
                axis.set_xlabel("%s（%s越好）" % (metric, "越大" if direction < 0 else "越小"))
                axis.set_ylabel("%s SOT 锁定率 UGR" % HORIZON_LABEL[horizon])
                axis.set_title("%s / %s / %s" % (modality.upper(), metric, HORIZON_LABEL[horizon]))
                axis.set_ylim(-0.03, 1.03)
                axis.grid(alpha=0.25)
                axis.legend(fontsize=7, loc="lower right")
            if not plotted:
                plt.close(fig)
                continue
            fig.suptitle("物理崩溃拐点（Knee Point）· %s / %s" % (modality.upper(), metric), fontsize=12)
            fig.tight_layout()
            target = chart_dir / ("chart_degrade_knee_%s_%s.png" % (modality, metric))
            fig.savefig(target)
            plt.close(fig)
            made.append(str(target))
    print("[CHART] %d charts -> %s" % (len(made), cli.chart_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Degradation knee-point calibration")
    sub = parser.add_subparsers(dest="phase", required=True)

    p = sub.add_parser("measure", help="X 路：退化后实测画质（逐 1s 窗口）")
    p.add_argument("--variants", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=phase_measure)

    p = sub.add_parser("curve", help="Y 路：SOT 锁定率 + 衰减曲线")
    p.add_argument("--variants", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument("--sot-dir", required=True)
    p.add_argument("--tau-dir", required=True, help="只用这里的**好片**标定纹理地板 τ_det")
    p.add_argument("--lock-out", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=phase_curve)

    p = sub.add_parser("knee", help="物理崩溃拐点")
    p.add_argument("--curve", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=phase_knee)

    p = sub.add_parser("blindtest", help="盲测：新旧阈值对比")
    p.add_argument("--knee", required=True)
    p.add_argument("--sot-windows", required=True)
    p.add_argument("--long-thresholds", default="manu/video_quality/report")
    p.add_argument("--short-thresholds", default="manu/video_quality/report/sot_short")
    p.add_argument("--out", required=True)
    p.set_defaults(func=phase_blindtest)

    p = sub.add_parser("combo", help="膝点线的单指标 / 组合判据盲测")
    p.add_argument("--knee", required=True)
    p.add_argument("--sot-windows", required=True)
    p.add_argument("--long-thresholds", default="manu/video_quality/report")
    p.add_argument("--short-thresholds", default="manu/video_quality/report/sot_short")
    p.add_argument("--out", required=True)
    p.set_defaults(func=phase_combo)

    p = sub.add_parser("charts", help="带拐点标注的衰减曲线图")
    p.add_argument("--knee", required=True)
    p.add_argument("--curve", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--chart-dir", required=True)
    p.set_defaults(func=phase_charts)

    cli = parser.parse_args()
    cli.func(cli)


if __name__ == "__main__":
    main()
