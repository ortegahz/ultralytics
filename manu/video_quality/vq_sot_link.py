#!/usr/bin/env python3
"""操作员点选 → 单目标跟踪（SOT）探针：把「画质红线」和「真实跟踪表现」挂上钩。

为什么重做（2026-11-20）
------------------------
上一版 ``vq_track_link.py`` 用**反无人机检测器**（trial_0080）在**无目标视频**上跑：
好片测出的「连续航迹」其实是**背景虚警**，坏片的「零航迹」只是糊图抑制了虚警 ——
因果论证不成立。本脚本取代它。

本脚本的物理意义（与业务场景严格对齐）
--------------------------------------
真实场景是：**操作员在画面任意位置点一下**，单目标跟踪器（SOT）锁住那个局部特征块并跟住。
于是这里：

    均匀 6×8 宫格 = 48 个候选区（与 vq_grid.py 逐格对应）
        ↓  每格里取 **Shi-Tomasi λ_min 最大**的角点 = 模拟操作员点「看得见的结构」
    仓库自带的跟踪核链式跟踪（**全程无检测器**）：
        · xcorr 臂：``manu/tracking/lookahead/xcorr.py`` 的配准核（逐帧重切模板 + 亚像素 + 唯一性诊断）
        · patch 臂：``manu/tracking/lookahead/polarity.py`` 的 PolarityTracker
                     （产品快核 LocalPatchTracker + 亮/暗极性自适应）
        · kcf   臂：``manu/tracking/lookahead/kcf_cv.py`` 的 OpenCV contrib KCF
                     （``cv2.legacy.TrackerKCF``，HOG+颜色名 + 在线模型更新），
                     开箱默认参数（含内部响应阈值 detect_thresh=0.5）
        · kcf_nt 臂：同一个 contrib KCF，但关掉内部响应阈值（detect_thresh=0），
                     失锁只由**与其它臂完全相同**的 NCC 判据决定
      三个**不同家族**的跟踪器，用来证明结论不是某一个跟踪器的伪影。
      ``--arms`` 选家族（默认 ``xcorr,patch``）；**点击点/协议对所有家族完全相同**
      （同一套网格、同一套角点、同一套 NCC 失锁与闭环判据），所以不同家族之间可直接比。
        ↓
    失锁判据：初始点击模板在**跟踪位置附近**（±8 px 搜索）的最佳 **NCC < 0.65**
    闭环判据：前向终点模板回配到起始帧，**往返误差 ≤ 3 px**（与 vq_grid 同阈值）

两个时程
--------
* ``short`` = 5 帧（与 ``vq_grid.py`` 的 gap 相同）⇒ 闭环判据与 §271 的 TGR/UGR **逐项可比**；
* ``long``  = 1 秒（业务意义上的「点一下能跟住多久」）。

闭环误差用的是**往返误差** ``|前向位移 + 回配位移|``，对相机运动不敏感：
只有「前向链走到的位置」和「回配找到的位置」对不上（外观漂移/跟丢）才会变大。

输出
----
每个 clip 一个 JSON：逐点击明细 + 逐窗口汇总 + 全片汇总。阈值相关的原始量全部落盘，
决策阈值可以在事后从原始数据重算（与 ``vq_grid.py`` 的做法一致）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from manu.tracking.async_dual_rate.fast_tracker import LocalPatchTracker  # noqa: E402
from manu.tracking.lookahead.kcf_cv import CvKcfTracker, KcfUnavailable, require_kcf  # noqa: E402
from manu.tracking.lookahead.polarity import PolarityTracker  # noqa: E402
from manu.tracking.lookahead.xcorr import match as xcorr_match  # noqa: E402
from vq_grid import grid_centers, shitiomasi_lambda_min  # noqa: E402
from vq_metrics import canonical, detect_content_box, probe_video  # noqa: E402

# --- 点击网格：与 vq_grid.py 完全一致（patch=32 / search_r=32 => 边距 50 px）--------
GRID_ROWS = 6
GRID_COLS = 8
GRID_PATCH = 32
GRID_SEARCH_R = 32

# --- 跟踪核参数 -------------------------------------------------------------------
TEMPLATE_SIZE = 24          # xcorr 默认模板边长
SEARCH_RADIUS = 12          # 逐步链式的单步搜索半径
BACK_SEARCH_RADIUS = 32      # 闭环回配的搜索半径（与 vq_grid 同量级）
LOCK_SEARCH_RADIUS = 8       # 锁定判据的 NCC 搜索半径（容忍定位抖动）
NCC_THRESH = 0.65           # 失锁/闭环的 NCC 阈值（与 vq_grid 一致）
CLOSURE_THRESH = 3.0        # 闭环往返误差阈值 px（与 vq_grid 一致）
FLAT_STD_THRESH = 3.0       # 无纹理格判据（与 vq_grid 一致）
DEFAULT_ARMS = ("xcorr", "patch")   # 跟踪器家族（--arms 可扩到 kcf）
# KCF 臂的参数覆盖表（None = 用 contrib 默认参数）：
#   kcf     —— 开箱默认：内部响应阈值 detect_thresh=0.5，响应峰低于它就中止（contrib 自带判据）
#   kcf_nt  —— 关掉内部响应阈值 => 失锁只由**统一的 NCC 判据**决定，与 xcorr/patch 完全同口径
KCF_ARM_PARAMS = {
    "kcf": None,
    "kcf_nt": {"detect_thresh": 0.0},
}

RAW_SUFFIXES = {".h264", ".hevc", ".265"}


# ---------------------------------------------------------------------------
# 解码
# ---------------------------------------------------------------------------
def open_frames_range(path: Path, width: int, height: int, start_frame: int = 0, fps: float = 0.0):
    """Decode a clip (optionally from ``start_frame``) as grayscale frames.

    ``start_frame`` uses ffmpeg's *input* seek, which decodes from the preceding
    keyframe and discards frames until the exact target - unlike OpenCV's
    ``CAP_PROP_POS_FRAMES`` on bare ``.h264`` elementary streams, which does not
    move at all (measured: ``POS_FRAMES`` stays 0 after the call).
    """
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_frame > 0:
        if fps <= 0:
            raise ValueError("start_frame > 0 requires a positive fps")
        cmd += ["-ss", f"{start_frame / fps:.6f}"]
    if path.suffix.lower() in RAW_SUFFIXES:
        cmd += ["-f", "h264" if "264" in path.suffix.lower() else "hevc"]
    cmd += ["-threads", "1", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 24)
    frame_bytes = width * height
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
    finally:
        proc.stdout.close()
        proc.wait()


# ---------------------------------------------------------------------------
# 模板 / NCC 工具
# ---------------------------------------------------------------------------
def gray_template(frame: np.ndarray, cx: float, cy: float, tsize: int):
    """Cut a raw grayscale template at ``(cx, cy)``.

    Border convention follows ``manu.tracking.lookahead.xcorr``: the window is
    **shifted** back into the picture instead of being cropped, so the template
    always keeps its full size and the coordinates stay absolute.
    """
    height, width = frame.shape
    half = tsize // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1, y1 = x0 + tsize, y0 + tsize
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    if x0 < 0 or y0 < 0 or x1 - x0 < tsize or y1 - y0 < tsize:
        return None, None
    return frame[y0:y1, x0:x1], (float(x0), float(y0))


def ncc_best(template: np.ndarray, frame: np.ndarray, cx: float, cy: float, half: int, radius: int) -> float:
    """Best NCC of ``template`` inside a ``+/- radius`` window around ``(cx, cy)``.

    The search makes the criterion tolerant to 1-3 px localisation jitter: the
    question is "is the clicked structure still present where the tracker says it
    is", not "is the tracker sub-pixel exact".
    """
    height, width = frame.shape
    x0 = int(round(cx)) - half - radius
    y0 = int(round(cy)) - half - radius
    x1 = int(round(cx)) + half + radius
    y1 = int(round(cy)) + half + radius
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return float("nan")
    roi = frame[y0:y1, x0:x1]
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return float("nan")
    response = cv2.matchTemplate(roi.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    return float(response.max())


def texture_measure(gray: np.ndarray, cx: float, cy: float, patch: int = GRID_PATCH) -> dict:
    """Texture measures of the patch the operator clicked.

    Mirrors ``vq_grid.py`` exactly (32x32 patch centred on the click):
    ``lap_var`` is the Laplacian variance used by the DET texture floor,
    ``std`` the flat-cell test, ``lam`` the Shi-Tomasi minimum eigenvalue.
    """
    half = patch // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1, y1 = x0 + patch, y0 + patch
    height, width = gray.shape
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return {"lap_var": float("nan"), "std": float("nan"), "lam": float("nan")}
    block = gray[y0:y1, x0:x1]
    return {
        "lap_var": float(cv2.Laplacian(block, cv2.CV_64F).var()),
        "std": float(np.std(block)),
        "lam": shitiomasi_lambda_min(block),
    }


def best_corner_in_cell(gray: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float] | None:
    """Shi-Tomasi corner with the largest lambda_min inside one grid cell.

    "Operator clicks something visible" -> the click lands on a corner / edge
    crossing, not on a flat patch.  Returns ``None`` when the cell holds no
    corner at all (flat cell).
    """
    x0, y0, x1, y1 = box
    patch = gray[y0:y1, x0:x1]
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    corners = cv2.goodFeaturesToTrack(
        patch, maxCorners=1, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False
    )
    if corners is None or len(corners) == 0:
        return None
    return float(x0 + corners[0][0][0]), float(y0 + corners[0][0][1])


# ---------------------------------------------------------------------------
# 两个跟踪臂
# ---------------------------------------------------------------------------
def _backward_closure(frames, click, final_pos, tsize, ncc_thresh):
    """Forward/backward round-trip error in px (NaN when the backward match fails).

    ``matched_center_in_frame0 = final_pos + backward_displacement``; the
    round-trip error is ``|final_pos + backward_displacement - click|``, i.e. how
    far the forward chain and the backward re-match disagree.  Camera motion
    cancels out (it appears in both displacements with opposite sign).
    """
    last, origin = gray_template(frames[-1], float(final_pos[0]), float(final_pos[1]), tsize)
    if last is None:
        return float("nan"), float("nan")
    anchor = np.asarray([float(click[0]), float(click[1])], dtype=np.float64)
    displacement, peak = xcorr_match(
        frames[0],
        last.astype(np.float32),
        anchor,
        anchor,
        radius=BACK_SEARCH_RADIUS,
        tsize=tsize,
        origin=origin,
    )
    if displacement is None or not np.isfinite(peak) or peak < ncc_thresh:
        return float("nan"), float(peak)
    back = np.asarray(final_pos, dtype=np.float64) + np.asarray(displacement, dtype=np.float64)
    return float(np.linalg.norm(back - anchor)), float(peak)


def _chain_xcorr(frames, click, horizon, checkpoints, tsize, radius, ncc_thresh):
    """One forward chain with the xcorr core; snapshots at every checkpoint horizon."""
    half = tsize // 2
    initial, _ = gray_template(frames[0], float(click[0]), float(click[1]), tsize)
    if initial is None:
        return {"status": "no_template"}, {}
    if float(np.std(initial)) < FLAT_STD_THRESH:
        return {"status": "flat"}, {}
    pos = np.asarray([float(click[0]), float(click[1])], dtype=np.float64)
    survived = 0
    peaks: list[float] = []
    uniq: list[float] = []
    locks: list[float] = []
    snaps: dict[int, dict] = {}
    for step in range(1, horizon + 1):
        src, origin = gray_template(frames[step - 1], float(pos[0]), float(pos[1]), tsize)
        if src is None:
            break
        diagnostics: dict = {}
        displacement, peak = xcorr_match(
            frames[step],
            src.astype(np.float32),
            pos,
            pos,
            radius=radius,
            tsize=tsize,
            origin=origin,
            diagnostics=diagnostics,
        )
        if displacement is None or not np.isfinite(peak) or peak < ncc_thresh:
            break
        pos = pos + np.asarray(displacement, dtype=np.float64)
        lock = ncc_best(initial, frames[step], float(pos[0]), float(pos[1]), half, LOCK_SEARCH_RADIUS)
        if not np.isfinite(lock) or lock < ncc_thresh:
            break
        survived = step
        peaks.append(float(peak))
        uniq.append(float(diagnostics.get("uniqueness", float("nan"))))
        locks.append(float(lock))
        if step in checkpoints:
            snaps[step] = {"pos": pos.copy(), "ncc": float(lock)}
    out = {
        "status": "ok",
        "survived": survived,
        "ncc_last": locks[-1] if locks else float("nan"),
        "ncc_min": float(min(locks)) if locks else float("nan"),
        "ncc_median": float(np.median(locks)) if locks else float("nan"),
        "peak_median": float(np.median(peaks)) if peaks else float("nan"),
        "uniq_median": float(np.median(uniq)) if uniq else float("nan"),
        "displacement": float(np.linalg.norm(pos - np.asarray(click, dtype=np.float64))),
    }
    return out, snaps


def _chain_patch(frames, click, horizon, checkpoints, tsize, ncc_thresh):
    """One forward chain with the product fast kernel (PolarityTracker, auto polarity)."""
    half = tsize // 2
    initial, _ = gray_template(frames[0], float(click[0]), float(click[1]), tsize)
    if initial is None:
        return {"status": "no_template"}, {}
    if float(np.std(initial)) < FLAT_STD_THRESH:
        return {"status": "flat"}, {}
    tracker = PolarityTracker(
        polarity="auto", win_size=32, th_k=2.0, min_energy=0.5, min_contrast=2.0, mode="centroid", bg_window=64
    )
    tracker.lock_from(frames[0], (float(click[0]), float(click[1])))
    pos = np.asarray([float(click[0]), float(click[1])], dtype=np.float64)
    survived = 0
    snrs: list[float] = []
    locks: list[float] = []
    snaps: dict[int, dict] = {}
    for step in range(1, horizon + 1):
        result = tracker.step(frames[step], pos)
        if not result.valid:
            break
        pos = np.asarray(result.xy, dtype=np.float64)
        lock = ncc_best(initial, frames[step], float(pos[0]), float(pos[1]), half, LOCK_SEARCH_RADIUS)
        if not np.isfinite(lock) or lock < ncc_thresh:
            break
        survived = step
        snrs.append(float(result.snr))
        locks.append(float(lock))
        if step in checkpoints:
            snaps[step] = {"pos": pos.copy(), "ncc": float(lock)}
    out = {
        "status": "ok",
        "survived": survived,
        "ncc_last": locks[-1] if locks else float("nan"),
        "ncc_min": float(min(locks)) if locks else float("nan"),
        "ncc_median": float(np.median(locks)) if locks else float("nan"),
        "snr_median": float(np.median(snrs)) if snrs else float("nan"),
        "displacement": float(np.linalg.norm(pos - np.asarray(click, dtype=np.float64))),
    }
    return out, snaps


def _chain_kcf(frames, click, horizon, checkpoints, tsize, ncc_thresh, params=None):
    """One forward chain with the OpenCV contrib KCF tracker (``cv2.legacy.TrackerKCF``).

    判据与 xcorr / patch 臂**完全一致**（同一段代码、同一组阈值）：
      * 失锁判据：初始模板在跟踪位置附近（±8 px 搜索）的最佳 NCC < ``ncc_thresh``；
      * 闭环判据：同一套 ``_backward_closure``（往返误差 ≤ 3 px）。

    与另外两臂的**结构性差异**（报告必须写明，不能含糊）：
      * xcorr 每帧重切模板（无记忆），KCF 用**初始模板 + 在线模型更新**（有外观记忆）；
      * xcorr 单步搜索 ±12 px，KCF 的搜索窗是模板的 3 倍（OpenCV 默认 padding=1）。
    因此不同家族的**锁定率绝对值不可直接比**，可比的是「退化曲线拐点对应的画质阈值」。
    """
    half = tsize // 2
    initial, _ = gray_template(frames[0], float(click[0]), float(click[1]), tsize)
    if initial is None:
        return {"status": "no_template"}, {}
    if float(np.std(initial)) < FLAT_STD_THRESH:
        return {"status": "flat"}, {}
    try:
        tracker = CvKcfTracker(template_size=tsize, params=params)
    except KcfUnavailable as exc:
        return {"status": "kcf_unavailable", "error": str(exc)}, {}
    if not tracker.init(frames[0], float(click[0]), float(click[1])):
        return {"status": "init_failed"}, {}
    pos = np.asarray([float(click[0]), float(click[1])], dtype=np.float64)
    survived = 0
    locks: list[float] = []
    snaps: dict[int, dict] = {}
    for step in range(1, horizon + 1):
        ok, center = tracker.update(frames[step])
        if not ok or not np.isfinite(center[0]) or not np.isfinite(center[1]):
            break
        pos = np.asarray(center, dtype=np.float64)
        lock = ncc_best(initial, frames[step], float(pos[0]), float(pos[1]), half, LOCK_SEARCH_RADIUS)
        if not np.isfinite(lock) or lock < ncc_thresh:
            break
        survived = step
        locks.append(float(lock))
        if step in checkpoints:
            snaps[step] = {"pos": pos.copy(), "ncc": float(lock)}
    out = {
        "status": "ok",
        "survived": survived,
        "ncc_last": locks[-1] if locks else float("nan"),
        "ncc_min": float(min(locks)) if locks else float("nan"),
        "ncc_median": float(np.median(locks)) if locks else float("nan"),
        # OpenCV 的 KCF 绑定不暴露响应峰值/唯一性 => 这两个量对 KCF 臂没有定义
        "peak_median": float("nan"),
        "uniq_median": float("nan"),
        "displacement": float(np.linalg.norm(pos - np.asarray(click, dtype=np.float64))),
    }
    return out, snaps


def track_click(frames, click, horizons, tsize, radius, ncc_thresh, arms=DEFAULT_ARMS):
    """Track one clicked point with every requested tracker family.

    ``horizons`` is the sorted list of checkpoint horizons (e.g. ``[5, 30]``).
    Returns ``{arm: record}``; every arm sees the *same* frames and the same click,
    only the tracking kernel differs.
    """
    horizon = max(horizons)
    checkpoints = set(horizons)

    def finish(rec, snaps):
        if rec.get("status") != "ok":
            rec = dict(rec)
            rec["horizons"] = {}
            return rec
        rec = dict(rec)
        per_horizon = {}
        for h in horizons:
            survived_h = rec["survived"] >= h
            snap = snaps.get(h)
            if snap is None:
                per_horizon[str(h)] = {
                    "survived": bool(survived_h),
                    "closure": float("nan"),
                    "back_peak": float("nan"),
                    "ncc": float("nan"),
                }
                continue
            closure, back_peak = _backward_closure(
                frames[: h + 1], click, snap["pos"], tsize, ncc_thresh
            )
            per_horizon[str(h)] = {
                "survived": bool(survived_h),
                "closure": closure,
                "back_peak": back_peak,
                "ncc": float(snap["ncc"]),
            }
        rec["horizons"] = per_horizon
        return rec

    out = {}
    for arm in arms:
        if arm == "xcorr":
            rec, snaps = _chain_xcorr(frames, click, horizon, checkpoints, tsize, radius, ncc_thresh)
        elif arm == "patch":
            rec, snaps = _chain_patch(frames, click, horizon, checkpoints, tsize, ncc_thresh)
        elif arm in KCF_ARM_PARAMS:
            rec, snaps = _chain_kcf(
                frames, click, horizon, checkpoints, tsize, ncc_thresh, KCF_ARM_PARAMS[arm]
            )
        else:
            raise ValueError(
                "unknown tracker family %r (known: xcorr, patch, %s)" % (arm, ", ".join(KCF_ARM_PARAMS))
            )
        out[arm] = finish(rec, snaps)
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _arm_summary(records, horizons, closure_thresh):
    ok = [r for r in records if r.get("status") == "ok"]
    out = {"n": len(ok)}
    for h in horizons:
        locked = 0
        survived_list = []
        closures = []
        nccs = []
        for rec in ok:
            entry = rec.get("horizons", {}).get(str(h), {})
            if not entry.get("survived"):
                continue
            survived_list.append(h)
            closure = entry.get("closure", float("nan"))
            if np.isfinite(closure):
                closures.append(closure)
                if closure <= closure_thresh:
                    locked += 1
            ncc = entry.get("ncc", float("nan"))
            if np.isfinite(ncc):
                nccs.append(ncc)
        out[f"h{h}"] = {
            "n_survived": len(survived_list),
            "survived_ratio": round(len(survived_list) / len(ok), 6) if ok else 0.0,
            "locked": locked,
            "lock_rate": round(locked / len(ok), 6) if ok else 0.0,
            "closure_median": float(np.median(closures)) if closures else float("nan"),
            "closure_p90": float(np.percentile(closures, 90)) if closures else float("nan"),
            "ncc_median": float(np.median(nccs)) if nccs else float("nan"),
        }
    return out


def run_epoch(frames, clicks, horizons, tsize, radius, ncc_thresh, closure_thresh, threads, arms=DEFAULT_ARMS):
    """Run every click of one epoch (all clicks share the same frame window)."""

    def one(click):
        return track_click(frames, click, horizons, tsize, radius, ncc_thresh, arms)

    if threads > 1 and len(clicks) > 1:
        with ThreadPoolExecutor(max_workers=min(threads, len(clicks))) as pool:
            results = list(pool.map(one, clicks))
    else:
        results = [one(click) for click in clicks]

    per_click = []
    for click, records in zip(clicks, results):
        entry = {"x": float(click[0]), "y": float(click[1])}
        for arm in arms:
            entry[arm] = records[arm]
        per_click.append(entry)
    primary = arms[0]
    flat = sum(1 for rec in per_click if rec[primary].get("status") == "flat")
    usable = [rec for rec in per_click if rec[primary].get("status") == "ok"]
    summary = {
        "n_clicks": len(per_click),
        "n_flat": flat,
        "n_usable": len(usable),
    }
    for arm in arms:
        summary[arm] = _arm_summary([rec[arm] for rec in per_click], horizons, closure_thresh)
    return summary, per_click


def analyse_clip(cli, video_path: Path) -> dict:
    """Single streaming pass over one clip; one click epoch per second."""
    started = time.perf_counter()
    info = probe_video(video_path)
    width, height = int(info["width"]), int(info["height"])
    fps = float(info["fps"]) or 25.0
    horizons = sorted({max(1, int(cli.short_frames)), max(1, int(round(fps * cli.long_seconds)))})
    horizon = max(horizons)
    ncc_thresh = float(cli.ncc_thresh)
    closure_thresh = float(cli.closure_thresh)
    arms = list(getattr(cli, "arms", DEFAULT_ARMS))

    crop = None
    buffer: list[np.ndarray] = []
    windows: list[dict] = []
    clicks_store: list[dict] = []
    frame_index = -1
    min_epoch = max(1, int(round(fps * cli.warmup_seconds)))

    for frame in open_frames_range(video_path, width, height, cli.start_frame, fps):
        frame_index += 1
        if cli.max_frames and frame_index >= cli.max_frames:
            break
        if crop is None:
            crop = detect_content_box(frame)
        top, bottom, left, right = crop
        content = frame[top: height - bottom, left: width - right]
        if content.size == 0:
            content = frame
        gray = canonical(content)

        buffer.append(gray)
        if len(buffer) > horizon + 1:
            buffer.pop(0)
        if len(buffer) < horizon + 1:
            continue
        local_start = frame_index - horizon
        if local_start < min_epoch or local_start % fps != 0:
            continue

        height_c, width_c = gray.shape
        centers = grid_centers(height_c, width_c, GRID_ROWS, GRID_COLS, GRID_PATCH, GRID_SEARCH_R)
        if centers is None:
            continue
        ys, xs = centers
        # 每格取 λ_min 最大的角点作为模拟点击点；取不到角点则退回格心
        cells: list[tuple[int, int, int, int]] = []
        for row_index, cy in enumerate(ys):
            for col_index, cx in enumerate(xs):
                half = GRID_PATCH // 2
                x0 = max(0, int(cx) - half)
                y0 = max(0, int(cy) - half)
                x1 = min(width_c, int(cx) + half)
                y1 = min(height_c, int(cy) + half)
                cells.append((x0, y0, x1, y1))
        offset = (len(windows) * cli.clicks_per_epoch) % len(cells)
        picked_index = [(offset + i) % len(cells) for i in range(min(cli.clicks_per_epoch, len(cells)))]

        # 两种点选方式轮流跑（epoch 奇偶交替），一次批跑同时给出两个口径：
        #   center —— 点格心（不挑位置，=「随便点」，与 vq_grid 的 48 格逐格对齐）
        #   corner —— 点格内 Shi-Tomasi 最优角点（=「操作员挑了个看得见的结构点」）
        mode = cli.click_mode
        if mode == "alternating":
            mode = "center" if len(windows) % 2 == 0 else "corner"

        picks = []
        for cell_index in picked_index:
            x0, y0, x1, y1 = cells[cell_index]
            center = ((x0 + x1) // 2.0, (y0 + y1) // 2.0)
            if mode == "center":
                picks.append(center)
                continue
            corner = best_corner_in_cell(gray, cells[cell_index])
            picks.append(corner if corner is not None else center)

        epoch_summary, per_click = run_epoch(
            buffer, picks, horizons, TEMPLATE_SIZE, SEARCH_RADIUS, ncc_thresh, closure_thresh, cli.threads, arms
        )
        epoch_summary["window"] = (cli.start_frame + local_start) // int(round(fps))
        epoch_summary["start_frame"] = int(cli.start_frame + local_start)
        epoch_summary["click_mode"] = mode
        windows.append(epoch_summary)
        for index, rec in enumerate(per_click):
            texture = texture_measure(gray, rec["x"], rec["y"])
            click_record = {
                "window": epoch_summary["window"],
                "cell": int(picked_index[index]),
                "click_mode": mode,
                "x": rec["x"],
                "y": rec["y"],
                "tex_lap_var": texture["lap_var"],
                "tex_std": texture["std"],
                "tex_lam": texture["lam"],
            }
            for arm in arms:
                click_record[arm] = rec[arm]
            clicks_store.append(click_record)
    summary = {
        "video": video_path.name,
        "sequence": video_path.stem,
        "group": cli.group,
        "kind": cli.kind,
        "modality": cli.modality,
        "path": str(video_path),
        "fps": round(fps, 4),
        "frames_decoded": frame_index + 1,
        "start_frame": int(cli.start_frame),
        "horizons": horizons,
        "clicks_per_epoch": int(cli.clicks_per_epoch),
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "ncc_thresh": ncc_thresh,
        "closure_thresh": closure_thresh,
        "template_size": TEMPLATE_SIZE,
        "windows": len(windows),
        "clicks": len(clicks_store),
    }
    for arm in arms:
        summary[arm] = _arm_summary([rec[arm] for rec in clicks_store], horizons, closure_thresh)
    summary["elapsed_s"] = round(time.perf_counter() - started, 1)
    return {"summary": summary, "windows": windows, "clicks": clicks_store}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click-initiated SOT probe: image quality vs tracking")
    parser.add_argument("--input", help="Single input video")
    parser.add_argument("--input-root", help="Directory of input videos")
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--group", default="")
    parser.add_argument("--kind", default="source")
    parser.add_argument("--modality", default="")
    parser.add_argument("--short-frames", type=int, default=5, help="Short horizon in frames (vq_grid gap=5)")
    parser.add_argument("--long-seconds", type=float, default=1.0, help="Long horizon in seconds")
    parser.add_argument("--clicks-per-epoch", type=int, default=16, help="Clicks per epoch (rotated over the 48 cells)")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Pilot: stop after N decoded frames")
    parser.add_argument("--threads", type=int, default=4, help="Threads used inside one epoch")
    parser.add_argument(
        "--click-mode",
        default="alternating",
        choices=["alternating", "center", "corner"],
        help="Where the simulated click lands: cell centre, best corner in the cell, or alternating",
    )
    parser.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="Comma separated tracker families (xcorr,patch,kcf); every family sees the same clicks",
    )
    parser.add_argument("--ncc-thresh", type=float, default=NCC_THRESH)
    parser.add_argument("--closure-thresh", type=float, default=CLOSURE_THRESH)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N files")
    return parser.parse_args()


def parse_arms(text: str) -> list[str]:
    """Ordered, de-duplicated tracker families."""
    arms: list[str] = []
    for part in str(text).split(","):
        name = part.strip()
        if name and name not in arms:
            arms.append(name)
    return arms


def main() -> None:
    cli = parse_args()
    arms = parse_arms(cli.arms)
    if not arms:
        raise SystemExit("--arms needs at least one tracker family")
    for arm in arms:
        if arm not in ("xcorr", "patch") and arm not in KCF_ARM_PARAMS:
            raise SystemExit(
                "unknown tracker family %r (known: xcorr, patch, %s)" % (arm, ", ".join(KCF_ARM_PARAMS))
            )
    if any(arm in KCF_ARM_PARAMS for arm in arms):
        try:
            require_kcf()
        except KcfUnavailable as exc:
            raise SystemExit("[FATAL] %s" % exc)
    cli.arms = arms
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cli.input:
        videos = [Path(cli.input)]
    else:
        videos = sorted(Path(cli.input_root).glob(cli.pattern))
    if cli.limit:
        videos = videos[: cli.limit]
    for video in videos:
        target = out_dir / f"{video.stem}.json"
        if target.exists():
            print(f"[SKIP] {target}")
            continue
        started = time.perf_counter()
        result = analyse_clip(cli, video)
        target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        summary = result["summary"]
        last = "h%d" % summary["horizons"][-1]
        rates = " ".join("%s_%s=%.4f" % (arm, last, summary[arm][last]["lock_rate"]) for arm in arms)
        print(
            "[DONE] %s frames=%d windows=%d clicks=%d %s (%.1fs)"
            % (
                video.name,
                summary["frames_decoded"],
                summary["windows"],
                summary["clicks"],
                rates,
                time.perf_counter() - started,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
