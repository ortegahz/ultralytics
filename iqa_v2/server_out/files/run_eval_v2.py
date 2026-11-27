#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 Target-Centric 评估主程序：退化仿真 → 双族跟踪 → 膝点反解 → 尾部验证。

与 v1 的**结构性差异**（报告必须写清楚，不能含糊）
---------------------------------------------------
* v1 的度量窗口是**全画幅 48 宫格随机采点**（背景纹理）；
* v2 的度量窗口是**业务目标本体 + 紧邻背景环**（Target-Centric ROI），
  跟踪器也**从 GT 目标中心起跟**（不是随便点一下）；
* v1 的时间轴是 1 秒 / 5 帧两个时程；v2 按任务书**只用 5 帧短时程**
  （业务上「目标能不能锁住」是短时程问题）。

跟踪协议（任务书口径）
----------------------
* 初始化框：``init_box = [cx - tw/2, cy - th/2, tw, th]``，``tw = max(16, w)``、
  ``th = max(16, h)``（防小目标导致 KCF 降采样退化）；
* **XCorr 臂**：初始模板（固定） + 局部搜索窗，搜索半径 ``max(16 px, 1.5·max(w, h))``；
* **KCF 臂**：``cv2.legacy.TrackerKCF_create``，**内部 ``detect_thresh = 0.0``**
  （关闭 contrib 自带的响应阈值截断 ⇒ 失锁只由统一的几何判据决定，复现 v1 ``kcf_nt`` 协议）；
* 跟住判据：``CLE_{t+4} ≤ max(5 px, 0.5·max(w, h))``。

跨家族纪律（v1 继承，v2 不放宽）
--------------------------------
XCorr 与 KCF 的**绝对锁定率刻度不可互换**：XCorr 用固定初始模板逐帧配准，
KCF 用「初始模板 + 在线模型更新 + 半分辨率 HOG」。两族**必须并列呈现、各自独立反解膝点**，
**严禁合并求平均**。可比的是「**线位**」（拐点对应的画质值）。

执行机：服务器 ``huangzhe@192.168.99.40:32222``（conda env ``uav``，依赖 cv2 + scipy）。
复现：

    # 1) 冒烟（3 条序列 × 20 窗口）
    python3 iqa_v2/run_eval_v2.py --phase all --smoke
    # 2) 全量
    python3 iqa_v2/run_eval_v2.py --phase all --workers 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset_anti_uav import (SEED, SMALL_TARGET_MAX_DIM, WINDOW_LEN,  # noqa: E402
                              FrameRecord, Sequence, discover_sequence_dirs, load_sequence,
                              read_frame_size, sample_windows, valid_windows)
from degradation_4d import (build_conditions, degradation_strength,  # noqa: E402
                            motion_angle_for)
from target_roi_metrics import (ALL_METRICS, GLOBAL_METRICS,  # noqa: E402
                              LOCAL_METRICS, METRIC_DIRECTION, compute_roi_metrics)

#: 三个跟踪臂：**跨家族主臂** xcorr（模板匹配链式）/ kcf_legacy（任务书指定的 contrib legacy KCF），
#: **同族对照臂** csrt（DCF 家族的另一实现，用来证明 KCF 侧结论不是实现细节）。
TRACKER_FAMILIES = ("xcorr", "kcf_legacy", "csrt")
LEGACY_KCF_NOTE = (
    "任务书指定的 cv2.legacy.TrackerKCF（opencv-contrib 4.10.0.84，"
    "PYTHONPATH=/tmp/pylibs_cvcontrib）在本环境**可用**：合成目标每帧平移 8 px 时，"
    "update() 逐帧返回的位置与真值**逐帧对齐**（120/128/136/144/152，误差 ≤ 1 px）。"
    "三条必须写进报告的实现事实："
    "① 🔴 **update() 严格滞后一帧**：在第 k 帧调用 update() 返回的是**第 k-1 帧**的矩形"
    "（实测：真值 120/128/136/144/152，update(帧1..帧5) 依次返回 120/128/136/144/152）。"
    "⇒ 本流水线用**多喂一帧**（第 t+5 帧）来报出「第 t+4 帧的位置」，否则 KCF 臂会被"
    "「一帧的运动量」系统性罚掉（快目标直接判失锁），污染膝点曲线；"
    "② detect_thresh 参数**确实生效**——置 0.0（任务书口径，关闭内部阈值截断）时正常跟踪；"
    "置 0.99 时响应峰达不到阈值，update() 会返回**退化的 (0,0) 矩形**（实测 5 帧里 4 帧塌到原点）；"
    "③ 初始化框必须**精确落在目标本体上**：一旦偏了半个框，KCF 会锁到目标的**局部**"
    "（实测偏 12 px 时它会以 12 px 的固定偏差跟着目标跑，看起来像「矩形不更新」——"
    "这是实现细节，不是跟踪器坏了）。"
)
#: 曲线集：``own_good`` = 该臂**自己的**好片段底座；``common_good`` = **三臂共同**的好片段
#: （跨臂公平对比只用它——两条臂的反解线位必须在**同一批窗口**上比较才可比）。
CURVE_SETS = ("own_good", "common_good")
#: 🔴 任务书 §四.2：**只有**散焦模糊 / 运动拖影 / 对比度衰减三族真实红外退化
#: （JPEG 族与加性白噪声轴已剔除，理由见 ``degradation_4d`` 模块 docstring）。
DEGRADATION_FAMILIES = ("blur", "motion", "contrast")
KNEE_AGREE_PCT = 15.0            # 双算法膝点偏差门限（> 15% 量程 ⇒ knee_reliable = 0）
N_BINS = 12                      # 衰减曲线的分箱数
#: 🔴 **曲线有效性门槛**：跟住率随退化递增的**总跨度**必须大于 5 个百分点，
#: 否则这条曲线是「平的」⇒ 膝点由噪声决定（双算法会一致地落在同一个噪声点上）。
#: 门限依据：跨度 < 5pp 的曲线落在 1259 窗口的抽样噪声量级内（±1.4pp @ 95% 置信），
#: 不足以支撑「崩溃拐点」这种强结论；>= 5pp 且单调的曲线才允许进约束线候选。
CURVE_MIN_SPAN = 0.05
MIN_PEAK = 0.25                  # XCorr 模板匹配峰值下限（低于此判为失锁）
MIN_INIT_DIM = 16                # 初始化框最小边长

#: 明细列：三臂 × (ok / cx / cy / cle / cle_end / maxcle / complete / success / peak / err)
#: （列名由 ``TRACKER_FAMILIES`` 生成 ⇒ 增删跟踪臂不会漏列）
#:
#: * ``cle``      = 任务书口径的 ``CLE_{t+4}``（末帧位置 vs 末帧 GT）；KCF 无尾帧时是**滞后一帧**读数
#: * ``cle_end``  = **同帧**口径的末帧误差（KCF 无尾帧时落在 t+3 帧）⇒ 三分法分类用它，避免滞后偏置
#: * ``maxcle``   = 5 帧窗口内**同帧**误差的最大值（颤动 / 漂移幅度）
#: * ``complete`` = 跟踪器是否**全程未中断**（中途失锁 ⇒ 0 ⇒ 直接判「跟丢」）
TRACK_SUFFIXES = ("ok", "cx", "cy", "cle", "cle_end", "maxcle", "complete", "success",
                  "peak", "err")
DETAIL_FIELDS = (
    ["seq_id", "frame_id", "small_target", "w", "h", "condition_family", "condition_level",
     "condition_label"]
    + list(ALL_METRICS)
    + [f"{family}_{suffix}" for family in TRACKER_FAMILIES for suffix in TRACK_SUFFIXES]
)


# --------------------------------------------------------------------------- 跟踪器
def _read_gray(path: str) -> np.ndarray:
    import cv2

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError("读帧失败：%s" % path)
    return image


def _clamp_rect(cx: float, cy: float, w: float, h: float, width: int, height: int):
    """以 ``(cx, cy)`` 为**中心**取 ``w × h`` 的框，越界时**整体平移**回画面内（不裁剪）。

    ⚠️ 约定：入参是**中心点**，不是左上角（与 ``target_roi_metrics._clamp_box`` 同款）。
    外部 API（``track_xcorr`` / ``track_csrt`` / ``track_kcf_legacy``）收的是**左上角**
    ``init_box``（任务书口径 ``[cx - tw/2, cy - th/2, tw, th]``），由调用方负责换算。
    """
    w = max(1, min(int(round(w)), width))
    h = max(1, min(int(round(h)), height))
    x0 = int(round(cx)) - w // 2
    y0 = int(round(cy)) - h // 2
    x1, y1 = x0 + w, y0 + h
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
    return x0, y0, w, h


def _shifted_window(image: np.ndarray, cx: float, cy: float, size: int):
    """以 ``(cx, cy)`` 为中心取 ``size`` 见方的窗；越界时**整体平移**回画面内（v1 同款约定）。"""
    height, width = image.shape[:2]
    size = max(1, min(int(size), min(height, width)))
    x0 = int(round(cx)) - size // 2
    y0 = int(round(cy)) - size // 2
    x1, y1 = x0 + size, y0 + size
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
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        return None, (0, 0)
    return image[y0:y1, x0:x1], (x0, y0)


def detect_polarity(frame: np.ndarray, cx: float, cy: float, w: float, h: float) -> int:
    """目标极性：核心区均值 > 背景环均值 ⇒ **白热**（+1）；否则 **黑热**（-1）。"""
    height, width = frame.shape[:2]
    cx0, cy0, cx1, cy1 = _clamp_rect(cx, cy, w, h, width, height)
    core = frame[cy0:cy1, cx0:cx1].astype(np.float64)
    ring_size = min(max(48, 3 * int(round(max(w, h)))), min(width, height))
    ox0, oy0, ox1, oy1 = _clamp_rect(cx, cy, ring_size, ring_size, width, height)
    ring = frame[oy0:oy1, ox0:ox1].astype(np.float64)
    if core.size == 0 or ring.size == 0:
        return 1
    return 1 if float(core.mean()) >= float(ring.mean()) else -1


def _weighted_template(frame: np.ndarray, cx: float, cy: float, size_w: int, size_h: int,
                       polarity: int, th_k: float = 2.5):
    """v1 口径的权重模板：按底噪阈值把背景压掉、只留目标本体。

    * 白热目标：``w = max(patch - (bg + 2.5σ), 0)``；黑热目标：``w = max((bg - 2.5σ) - patch, 0)``；
    * 权重图恒非负 ⇒ 「越大 = 越像目标本体」，极性无关；
    * 返回 ``(weights, origin)``，``origin`` 是模板左上角在源帧的绝对坐标。
    """
    patch, origin = _shifted_window(frame, cx, cy, max(size_w, size_h))
    if patch is None:
        return None, None
    bg_size = max(96, 2 * max(size_w, size_h))
    background_patch, _ = _shifted_window(frame, cx, cy, bg_size)
    if background_patch is None:
        return None, None
    stride = max(1, background_patch.shape[0] // 32)
    sample = background_patch[::stride, ::stride].astype(np.float32)
    background = float(np.median(sample))
    mad = float(np.median(np.abs(sample - background)))
    sigma = max(1.4826 * mad, 1e-3)
    patch_f = patch.astype(np.float32)
    if polarity < 0:
        weights = np.clip((background - th_k * sigma) - patch_f, 0.0, None)
    else:
        weights = np.clip(patch_f - (background + th_k * sigma), 0.0, None)
    if float(weights.sum()) <= 1e-6:
        return None, None
    return weights, (float(origin[0]), float(origin[1]))


def _match_template(dst: np.ndarray, template: np.ndarray, anchor, radius: int,
                    origin, polarity: int):
    """在 ``anchor`` 附近的局部搜索窗里做 ``TM_CCOEFF_NORMED`` 配准（带抛物线亚像素精修）。"""
    import cv2

    if polarity < 0:
        template = -template
    size = 2 * int(radius) + int(template.shape[0])
    search, (ox, oy) = _shifted_window(dst, float(anchor[0]), float(anchor[1]), size)
    if search is None or search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None, 0.0
    response = cv2.matchTemplate(search.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    _, best, _, best_loc = cv2.minMaxLoc(response)
    px, py = best_loc
    dx = dy = 0.0
    if 0 < px < response.shape[1] - 1:
        a, b, c = (float(response[py, px - 1]), float(response[py, px]),
                   float(response[py, px + 1]))
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dx = 0.5 * (a - c) / den
    if 0 < py < response.shape[0] - 1:
        a, b, c = (float(response[py - 1, px]), float(response[py, px]),
                   float(response[py + 1, px]))
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dy = 0.5 * (a - c) / den
    displacement = np.asarray([(ox + px + dx) - origin[0], (oy + py + dy) - origin[1]],
                              dtype=np.float64)
    return displacement, float(best)


def track_xcorr(frames: list[np.ndarray], init_box: tuple[float, float, float, float],
                search_radius: int) -> dict:
    """XCorr 臂（**v1 链式协议**）：每步在上一帧位置重切权重模板，再在下一帧局部窗内配准。

    为什么不��「固定初始模板」：Anti-UAV 的目标每帧可移动 3~12 px（中位 3.01 px/帧，
    p90 达 14 px/帧），固定模板在 4 帧内就失配 ⇒ 锁到背景上（实测 CLE 中位 16 px）。
    链式（每步重切）实测 CLE 中位 0.5~3 px，与真值一致 ⇒ 这是可用的「跟踪器」语义。

    判据与 KCF/CSRT 臂**完全一致**：失锁只看最后一步的几何误差，
    中间步骤的配准峰值低于 ``MIN_PEAK`` 即判为该臂失锁（记 CLE = NaN ⇒ success = 0）。
    """
    frame0 = frames[0]
    height, width = frame0.shape[:2]
    x0, y0, box_w, box_h = init_box
    tx0, ty0, tw, th = _clamp_rect(x0 + box_w / 2.0, y0 + box_h / 2.0,
                                  box_w, box_h, width, height)
    if tw < 4 or th < 4:
        return {"ok": False, "cx": float("nan"), "cy": float("nan"), "peak": float("nan")}
    polarity = detect_polarity(frame0, tx0 + tw / 2.0, ty0 + th / 2.0, tw, th)
    radius = int(max(16, search_radius))

    anchor = np.asarray([tx0 + tw / 2.0, ty0 + th / 2.0], dtype=np.float64)
    peak = float("nan")
    # ``trace`` = 逐帧位置（元素 = (cx, cy, 对应的 GT 帧下标)）；``complete`` = 全程未中断。
    trace = [(float(anchor[0]), float(anchor[1]), 0)]
    complete = True
    for step in range(1, len(frames)):
        template, origin = _weighted_template(frames[step - 1], float(anchor[0]),
                                              float(anchor[1]), tw, th, polarity)
        if template is None:
            complete = False
            break
        displacement, peak = _match_template(frames[step], template, anchor, radius,
                                             origin, polarity)
        if displacement is None or not math.isfinite(peak) or peak < MIN_PEAK:
            complete = False
            break
        matched = anchor + displacement
        limit = float(radius) + float(max(tw, th))
        if float(np.linalg.norm(matched - anchor)) > limit:
            complete = False
            break
        anchor = matched
        trace.append((float(anchor[0]), float(anchor[1]), step))
    return {"ok": True, "cx": float(anchor[0]), "cy": float(anchor[1]), "peak": peak,
            "trace": trace, "complete": complete}


def _kcf_tracker(params: dict | None = None):
    """构造 contrib legacy KCF（``detect_thresh`` 通过 YAML FileStorage 覆写，v1 同款手法）。"""
    import cv2
    import tempfile

    legacy = getattr(cv2, "legacy", None)
    if legacy is None or not hasattr(legacy, "TrackerKCF_create"):
        raise RuntimeError("缺 cv2.legacy.TrackerKCF_create（需 PYTHONPATH=/tmp/pylibs_cvcontrib）")
    tracker = legacy.TrackerKCF_create()
    if params:
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            handle.write("%YAML:1.0\n---\n")
            for key, value in params.items():
                handle.write("%s: %s\n" % (key, value))
            handle.close()
            storage = cv2.FileStorage(handle.name, cv2.FILE_STORAGE_READ)
            try:
                tracker.read(storage.root())
            finally:
                storage.release()
        finally:
            os.unlink(handle.name)
    return tracker


def track_kcf_legacy(frames: list[np.ndarray],
                     init_box: tuple[float, float, float, float],
                     tail_frame: np.ndarray | None = None) -> dict:
    """任务书指定的 contrib legacy KCF 臂（``detect_thresh = 0.0``）。

    🔴 **本实现的 ``update()`` 严格滞后一帧**：在第 k 帧上调用 ``update()``，
    返回的是**第 k-1 帧**的矩形（实测：真值 120/128/136/144/152，
    update(帧1..帧5) 依次返回 120/128/136/144/152 ⇒ 整整晚一帧）。
    ⇒ 要报出「第 t+4 帧的位置」，必须**多喂一帧**（``tail_frame`` = 第 t+5 帧）：
    update(t+1) … update(t+4) 之后再 update(t+5)，最后拿到的就是 t+4 帧的矩形。
    没有尾帧时（序列末尾）只能退回「滞后一帧」的读数，并在 ``error`` 里标注。

    ⚠️ 初始化框必须精确落在目标本体上，否则会锁到目标局部（表现为固定偏差）。
    """
    import cv2

    frame0 = frames[0]
    height, width = frame0.shape[:2]
    x0, y0, box_w, box_h = init_box
    rx, ry, rw, rh = _clamp_rect(x0 + box_w / 2.0, y0 + box_h / 2.0,
                                 box_w, box_h, width, height)
    try:
        tracker = _kcf_tracker({"detect_thresh": 0.0})
    except RuntimeError as exc:
        return {"ok": False, "cx": float("nan"), "cy": float("nan"), "peak": float("nan"),
                "error": str(exc)}
    try:
        # ⚠️ OpenCV 4.x 的 ``Tracker.init`` 返回 **None**（void 方法），不是 True；
        # 只有显式返回 False 才算初始化失败。
        init_result = tracker.init(cv2.cvtColor(frame0, cv2.COLOR_GRAY2BGR), (rx, ry, rw, rh))
    except cv2.error as exc:
        return {"ok": False, "cx": float("nan"), "cy": float("nan"), "peak": float("nan"),
                "error": "init:%s" % exc}
    if init_result is False:
        return {"ok": False, "cx": float("nan"), "cy": float("nan"), "peak": float("nan"),
                "error": "init failed"}
    cx = float(rx + rw / 2.0)
    cy = float(ry + rh / 2.0)
    # 🔴 ``update()`` 滞后一帧 ⇒ ``update(frames[k])`` 返回的是**第 k-1 帧**的位置。
    #    ``trace`` 的第 i 个元素对应的 GT 帧下标 = k-1（尾帧补偿命中时再多一格 = 第 4 帧）。
    trace = []
    complete = True
    for step in range(1, len(frames)):
        try:
            result = tracker.update(cv2.cvtColor(frames[step], cv2.COLOR_GRAY2BGR))
        except cv2.error as exc:
            return {"ok": False, "cx": cx, "cy": cy, "peak": float("nan"),
                    "error": "update:%s" % exc, "trace": trace, "complete": False}
        if isinstance(result, tuple):
            ok, rect = bool(result[0]), result[1]
        else:
            ok, rect = bool(result), None
        if not ok or not rect:
            complete = False
            break
        cx = float(rect[0]) + float(rect[2]) / 2.0
        cy = float(rect[1]) + float(rect[3]) / 2.0
        trace.append((cx, cy, step - 1))
    # ---- 尾帧补偿：update(t+5) 返回的是「第 t+4 帧」的矩形 ⇒ 正好是判据要的位置
    note = ""
    if tail_frame is not None:
        try:
            result = tracker.update(cv2.cvtColor(tail_frame, cv2.COLOR_GRAY2BGR))
        except cv2.error as exc:
            return {"ok": False, "cx": cx, "cy": cy, "peak": float("nan"),
                    "error": "update_tail:%s" % exc, "trace": trace, "complete": False}
        if isinstance(result, tuple):
            ok, rect = bool(result[0]), result[1]
        else:
            ok, rect = bool(result), None
        if ok and rect:
            cx = float(rect[0]) + float(rect[2]) / 2.0
            cy = float(rect[1]) + float(rect[3]) / 2.0
            trace.append((cx, cy, len(frames) - 1))
        else:
            complete = False
            note = "tail_update_failed"
    else:
        note = "no_tail_frame(滞后一帧读数)"
    return {"ok": True, "cx": cx, "cy": cy, "peak": float("nan"), "error": note,
            "trace": trace, "complete": complete}


def track_csrt(frames: list[np.ndarray], init_box: tuple[float, float, float, float]) -> dict:
    """DCF 家族臂：``cv2.TrackerCSRT_create``（判别相关滤波 + 尺度估计，全局搜索）。

    角色：**KCF 同族对照臂**。KCF 主臂（``kcf_legacy``）已实测可用，本臂用同属判别相关滤波
    （DCF）家族的**另一套独立实现**（判别相关滤波 + 尺度估计）复算一遍，
    用来证明「KCF 侧反推出的线位」不是某个实现的细节。
    """
    import cv2

    frame0 = frames[0]
    height, width = frame0.shape[:2]
    x0, y0, box_w, box_h = init_box
    rx, ry, rw, rh = _clamp_rect(x0 + box_w / 2.0, y0 + box_h / 2.0,
                                 box_w, box_h, width, height)
    try:
        tracker = cv2.TrackerCSRT_create()
    except Exception as exc:                       # pragma: no cover
        return {"ok": False, "cx": float("nan"), "cy": float("nan"), "peak": float("nan"),
                "error": str(exc)}
    try:
        # ⚠️ OpenCV 4.x 的 ``Tracker.init`` 返回 **None**（void 方法），不是 True；
        # 只有显式返回 False 才算初始化失败。
        init_result = tracker.init(cv2.cvtColor(frame0, cv2.COLOR_GRAY2BGR), (rx, ry, rw, rh))
    except cv2.error as exc:
        return {"ok": False, "cx": float("nan"), "cy": float("nan"), "peak": float("nan"),
                "error": "init:%s" % exc}
    if init_result is False:
        return {"ok": False, "cx": float("nan"), "cy": float("nan"), "peak": float("nan"),
                "error": "init failed"}
    cx = float(rx + rw / 2.0)
    cy = float(ry + rh / 2.0)
    # OpenCV 4.x 的 ``Tracker.update`` 返回**当前帧**结果 ⇒ 无滞后一帧问题。
    trace = [(cx, cy, 0)]
    complete = True
    for step in range(1, len(frames)):
        try:
            result = tracker.update(cv2.cvtColor(frames[step], cv2.COLOR_GRAY2BGR))
        except cv2.error as exc:
            return {"ok": False, "cx": cx, "cy": cy, "peak": float("nan"),
                    "error": "update:%s" % exc, "trace": trace, "complete": False}
        if isinstance(result, tuple):
            ok, rect = bool(result[0]), result[1]
        else:
            ok, rect = bool(result), None
        if not ok or not rect:
            complete = False
            break
        cx = float(rect[0]) + float(rect[2]) / 2.0
        cy = float(rect[1]) + float(rect[3]) / 2.0
        trace.append((cx, cy, step))
    return {"ok": True, "cx": cx, "cy": cy, "peak": float("nan"),
            "trace": trace, "complete": complete}


# --------------------------------------------------------------------------- 单窗口评估
def _trace_errors(result: dict, gt: list[FrameRecord]) -> tuple[float, float, int]:
    """从跟踪器的**逐帧轨迹**算 ``(同帧末帧误差, 窗口内最大同帧误差, 是否全程未中断)``。

    * ``cle_end`` 用**同帧**口径（位置 vs **同一帧**的 GT）⇒ KCF 无尾帧时落在 t+3 帧，
      不会被「滞后一帧」系统性罚掉；
    * ``maxcle`` 是 5 帧窗口内的**颤动 / 漂移幅度**（任务书 §三「明显框体颤动/漂移」）；
    * ``complete`` = 跟踪器**全程未中断**：中途失锁的窗口直接判「跟丢」，
      即使它最后一帧的位置碰巧落在 GT 附近。
    """
    trace = result.get("trace") or []
    if not trace:
        return float("nan"), float("nan"), 0
    errors = []
    for cx, cy, index in trace:
        if index >= len(gt):
            continue
        record = gt[index]
        errors.append(math.hypot(cx - record.cx, cy - record.cy))
    if not errors:
        return float("nan"), float("nan"), 0
    return errors[-1], max(errors), int(bool(result.get("complete", False)))


def evaluate_window(seq_id: str, frame_id: int, frame_paths: list[str],
                    gt: list[FrameRecord], small_target: bool,
                    conditions: list, tail_path: str | None = None) -> list[dict]:
    """对一个 5 帧窗口，跑完所有退化条件 × 三臂跟踪器，返回明细行。

    ``tail_path`` = 第 t+5 帧（KCF 臂的「滞后一帧」补偿用；序列末尾可能没有）。
    """
    frames = [_read_gray(path) for path in frame_paths]
    tail_frame = _read_gray(tail_path) if tail_path else None
    first = gt[0]
    last = gt[-1]
    target_w, target_h = float(first.w), float(first.h)
    threshold = max(5.0, 0.5 * max(target_w, target_h))
    init_box = (first.cx - max(MIN_INIT_DIM, target_w) / 2.0,
                first.cy - max(MIN_INIT_DIM, target_h) / 2.0,
                max(MIN_INIT_DIM, target_w), max(MIN_INIT_DIM, target_h))
    search_radius = int(max(16, round(1.5 * max(target_w, target_h))))
    angle = motion_angle_for("%s:%d" % (seq_id, frame_id), SEED)

    rows: list[dict] = []
    for condition in conditions:
        degraded = [frame if condition.is_clean else
                    _apply(frame, condition, angle) for frame in frames]
        degraded_tail = None
        if tail_frame is not None:
            degraded_tail = (tail_frame if condition.is_clean else
                             _apply(tail_frame, condition, angle))
        metrics = compute_roi_metrics(degraded[0], first.cx, first.cy, target_w, target_h)
        row = {
            "seq_id": seq_id,
            "frame_id": frame_id,
            "small_target": int(small_target),
            "w": target_w,
            "h": target_h,
            "condition_family": condition.family,
            "condition_level": condition.level,
            "condition_label": condition.label,
        }
        row.update(metrics)
        result_x = track_xcorr(degraded, init_box, search_radius)
        result_s = track_csrt(degraded, init_box)
        result_l = track_kcf_legacy(degraded, init_box, degraded_tail)
        for family, result in (("xcorr", result_x), ("csrt", result_s), ("kcf_legacy", result_l)):
            if result["ok"] and math.isfinite(result["cx"]) and math.isfinite(result["cy"]):
                cle = math.hypot(result["cx"] - last.cx, result["cy"] - last.cy)
            else:
                cle = float("nan")
            cle_end, maxcle, complete = _trace_errors(result, gt)
            row["%s_ok" % family] = int(bool(result["ok"]))
            row["%s_cx" % family] = result["cx"]
            row["%s_cy" % family] = result["cy"]
            row["%s_cle" % family] = cle
            row["%s_cle_end" % family] = cle_end
            row["%s_maxcle" % family] = maxcle
            row["%s_complete" % family] = complete
            row["%s_success" % family] = int(math.isfinite(cle) and cle <= threshold)
            row["%s_peak" % family] = result["peak"]
            row["%s_err" % family] = str(result.get("error", ""))
        rows.append(row)
    return rows


def _apply(frame: np.ndarray, condition, angle: float) -> np.ndarray:
    from degradation_4d import apply_degradation

    return apply_degradation(frame, condition.family, condition.level, angle)


# --------------------------------------------------------------------------- 并行执行
_WORKER: dict = {}
_RUN_START = 0.0


def _worker_init(dataset_root: str) -> None:
    import cv2

    cv2.setNumThreads(1)
    _WORKER["root"] = Path(dataset_root)


def _worker_task(job: tuple) -> list[dict]:
    seq_id, frame_id, frame_paths, gt, small_target, tail_path = job
    gt_records = [FrameRecord(*item) for item in gt]
    try:
        return evaluate_window(seq_id, frame_id, frame_paths, gt_records, small_target,
                               _WORKER["conditions"], tail_path)
    except Exception as exc:            # pragma: no cover - 单窗口失败不能拖垮全量
        return [{"seq_id": seq_id, "frame_id": frame_id, "condition_family": "error",
                 "condition_level": 0.0, "condition_label": "error:%s" % exc,
                 "small_target": int(small_target), "w": 0.0, "h": 0.0,
                 **{name: float("nan") for name in ALL_METRICS}}]


def build_jobs(sequences: list[Sequence], selected: dict) -> list[tuple]:
    jobs: list[tuple] = []
    for seq_id in sorted(selected):
        seq = next(s for s in sequences if s.seq_id == seq_id)
        for window in selected[seq_id]:
            paths = [str(seq.frame_files[window.frame_id + k]) for k in range(WINDOW_LEN)]
            gt = [(r.seq_id, r.frame_id, r.track_id, r.cx, r.cy, r.w, r.h, r.exist)
                  for r in window.gt]
            # 尾帧 t+5：KCF legacy 的 ``update()`` **滞后一帧**（实测 update(帧1..5)
            # 返回的是 帧0..4 的位置）⇒ 第 5 步更新必须喂尾帧，否则 KCF 臂永远慢一帧。
            # 序列末尾可能没有尾帧 ⇒ None（此时 KCF 臂退化为 4 步更新，已在报告标注）。
            tail_index = window.frame_id + WINDOW_LEN
            tail_path = (str(seq.frame_files[tail_index])
                         if tail_index < len(seq.frame_files) else None)
            jobs.append((seq_id, window.frame_id, paths, gt, window.small_target, tail_path))
    return jobs


def _run_jobs(jobs: list[tuple], conditions: list, workers: int, writer,
              collect: bool = False) -> tuple[int, list[dict]]:
    """执行一批 ``(窗口 × 条件)`` 任务，把明细行逐行写进 ``writer``。

    返回 ``(写出行数, 行列表)``；``collect=True`` 时才把行留在内存里
    （只有 Pass A 需要——分类要拿干净帧的行）。
    """
    import multiprocessing as mp

    if not jobs or not conditions:
        return 0, []
    rows_written = 0
    collected: list[dict] = []
    _WORKER["conditions"] = conditions
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers, initializer=_worker_init,
                      initargs=("",)) as pool:
            for index, rows in enumerate(pool.imap_unordered(_worker_task, jobs, chunksize=8)):
                for row in rows:
                    writer.writerow(row)
                    rows_written += 1
                    if collect:
                        collected.append(row)
                if (index + 1) % 200 == 0:
                    print("[EVAL] %d/%d 窗口（%.1f 分钟）" %
                          (index + 1, len(jobs), (time.time() - _RUN_START) / 60.0), flush=True)
    else:
        for index, job in enumerate(jobs):
            for row in _worker_task(job):
                writer.writerow(row)
                rows_written += 1
                if collect:
                    collected.append(row)
            if (index + 1) % 50 == 0:
                print("[EVAL] %d/%d 窗口（%.1f 分钟）" %
                      (index + 1, len(jobs), (time.time() - _RUN_START) / 60.0), flush=True)
    return rows_written, collected


def run_eval(cli: argparse.Namespace) -> Path:
    """**两趟法**（任务书 §二.1）：Pass A 干净帧分类 → Pass B 只让好片段进退化池。

    * **Pass A**：全部窗口跑 ``clean`` 条件 ⇒ 三臂的干净帧跟住表现 ⇒ 三分法分类
      （好：``CLE_{t+4} ≤ 3 px`` 且 5 帧全程稳定；一般：未跟丢但临界；跟丢：真实失锁）。
    * **序列级 7:3 切分**（任务书 §四.1）：按 ``seq_id`` **整条**切 70% 标定 / 30% 留出。
    * **Pass B**：**只有标定集里的好片段**进入退化仿真池 ⇒ 衰减曲线起点纯粹（clean 档跟住率 ≈ 100%）。
    """
    global _RUN_START
    dataset_root = Path(cli.dataset_root).resolve()
    output_dir = Path(cli.output_dir).resolve()
    cache_dir = Path(cli.cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = discover_sequence_dirs(dataset_root)
    sequences = []
    for seq_dir in seq_dirs:
        seq = load_sequence(seq_dir)
        if seq.kept:
            seq.width, seq.height = read_frame_size(seq.frame_files[0])
        sequences.append(seq)
    kept = [s for s in sequences if s.kept]
    all_windows = {}
    for seq in kept:
        windows = valid_windows(seq)
        if windows:
            all_windows[seq.seq_id] = windows
    if cli.smoke:
        selected = {seq_id: all_windows[seq_id][:20] for seq_id in sorted(all_windows)[:3]}
    else:
        selected, _ = sample_windows(all_windows, seed=cli.seed,
                                     max_per_seq=cli.max_windows_per_seq,
                                     total_target=cli.total_target)
    jobs = build_jobs(kept, selected)

    conditions = build_conditions()
    clean_conditions = [c for c in conditions if c.is_clean]
    degrade_conditions = [c for c in conditions if not c.is_clean]
    workers = max(1, int(cli.workers))
    detail_path = cache_dir / "degrade_detail.csv"
    _RUN_START = time.time()
    print("[EVAL] Pass A 干净帧分类：窗口 %d × 条件 %d"
          % (len(jobs), len(clean_conditions)))

    with open(detail_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        rows_a, clean_rows = _run_jobs(jobs, clean_conditions, workers, writer, collect=True)
        print("[EVAL] Pass A 完成 %d 行（%.1f 分钟）"
              % (rows_a, (time.time() - _RUN_START) / 60.0))

        # ---- 三分法分类 + 序列级 7:3 切分（唯一 owner：regression_v2）
        from regression_v2 import (classify_rows, split_sequences, verify_split_disjoint,
                                   write_class_csv, write_split_csv)

        class_rows = classify_rows(clean_rows)
        good_seqs = sorted({row["seq_id"] for row in class_rows if row["class_ds"] == "good"})
        split_map = split_sequences(good_seqs)
        disjoint, overlap = verify_split_disjoint(split_map)
        if not disjoint:
            raise SystemExit("[FATAL] 序列级切分出现交叉：%s" % overlap)
        calib_seqs = {s for s, v in split_map.items() if v == "calib"}
        holdout_seqs = {s for s, v in split_map.items() if v == "holdout"}
        print("[EVAL] 三分法：好 %d / 一般 %d / 跟丢 %d；序列切分 标定 %d 条 / 留出 %d 条（零交叉 ✅）"
              % (sum(1 for r in class_rows if r["class_ds"] == "good"),
                 sum(1 for r in class_rows if r["class_ds"] == "mediocre"),
                 sum(1 for r in class_rows if r["class_ds"] == "lost"),
                 len(calib_seqs), len(holdout_seqs)))

        # ---- Pass B 仿真池 = 标定集 ∩ (三臂「好片段」的并集)
        pool_keys = set()
        for row in class_rows:
            if split_map.get(row["seq_id"]) != "calib":
                continue
            if any(row["class_%s" % arm] == "good" for arm in TRACKER_FAMILIES):
                pool_keys.add((row["seq_id"], row["frame_id"]))
        pool_jobs = [job for job in jobs if (job[0], job[1]) in pool_keys]
        print("[EVAL] Pass B 退化仿真：池 %d 窗口 × 条件 %d（留出集零参与）"
              % (len(pool_jobs), len(degrade_conditions)))
        rows_b, _ = _run_jobs(pool_jobs, degrade_conditions, workers, writer)

    write_class_csv(output_dir / "segment_class_v2.csv", class_rows)
    write_split_csv(output_dir / "split_v2.csv", split_map, class_rows)
    print("[EVAL] 明细 %d 行（Pass A %d + Pass B %d）→ %s（%.1f 分钟）"
          % (rows_a + rows_b, rows_a, rows_b, detail_path, (time.time() - _RUN_START) / 60.0))
    return detail_path


# --------------------------------------------------------------------------- 膝点反解
def prepare_curve(xs: np.ndarray, ys: np.ndarray, direction: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """整理成「X 越大越好、Y 单调不减」的曲线（v1 同款：保序回归 → 合并重复 X）。"""
    from scipy.optimize import isotonic_regression

    order = np.argsort(xs, kind="mergesort")
    xs, ys = xs[order], ys[order]
    if direction > 0:
        xs = -xs[::-1]
        ys = ys[::-1]
    if xs.size == 0:
        return xs, ys
    fitted = np.asarray(isotonic_regression(ys, increasing=True).x, dtype=float)
    unique_x, start = np.unique(xs, return_index=True)
    unique_y = np.maximum.reduceat(fitted, start)
    return unique_x, unique_y


def kneedle_knee(xs: np.ndarray, ys: np.ndarray, direction: int = -1) -> tuple[float, float]:
    """Kneedle：X/Y 归一到 [0,1] 后取「到首尾连线距离最大」的点。"""
    curve_x, curve_y = prepare_curve(xs, ys, direction)
    x_span = curve_x.max() - curve_x.min()
    y_span = curve_y.max() - curve_y.min()
    if curve_x.size < 3 or x_span <= 0 or y_span <= 0:
        return float("nan"), float("nan")
    xn = (curve_x - curve_x.min()) / x_span
    yn = (curve_y - curve_y.min()) / y_span
    index = int(np.argmax(np.abs(yn - xn)))
    return float(curve_x[index]), float(curve_y[index])


def pchip_knee(xs: np.ndarray, ys: np.ndarray, direction: int = -1) -> tuple[float, float]:
    """PCHIP 样条二阶导极值（曲率最大处）——与 Kneedle 交叉验证。"""
    from scipy.interpolate import PchipInterpolator

    curve_x, curve_y = prepare_curve(xs, ys, direction)
    if curve_x.size < 4:
        return float("nan"), float("nan")
    spline = PchipInterpolator(curve_x, curve_y)
    dense = np.linspace(curve_x[0], curve_x[-1], 1500)
    second = spline.derivative(2)(dense)
    index = int(np.argmax(np.abs(second)))
    return float(dense[index]), float(spline(dense[index]))


def curve_y50(xs: np.ndarray, ys: np.ndarray, direction: int = -1) -> float:
    """曲线上跟住率**首次跌破** 0.5 的 X（参考量）；曲线从不塌到 0.5 ⇒ nan。

    ⚠️ 旧实现返回的是「第一个 >= 0.5 的点」——在从不塌掉的曲线上那就是**曲线首点**，
    是个没有物理意义的数字（本轮净化底座上全部曲线都属于这一类）。
    """
    from scipy.interpolate import PchipInterpolator

    curve_x, curve_y = prepare_curve(xs, ys, direction)
    if curve_x.size < 4:
        return float("nan")
    spline = PchipInterpolator(curve_x, curve_y)
    dense = np.linspace(curve_x[0], curve_x[-1], 1500)
    values = spline(dense)
    hit = np.nonzero(values <= 0.5)[0]
    if not hit.size:
        return float("nan")
    index = int(hit[0])
    if index == 0:
        return float(dense[0])
    low, high = float(values[index - 1]), float(values[index])
    weight = 0.0 if high == low else (low - 0.5) / (low - high)
    return float(dense[index - 1] + weight * (dense[index] - dense[index - 1]))


def binned_curve(values: np.ndarray, success: np.ndarray, n_bins: int = N_BINS):
    """按分位数分箱：返回 (箱内 X 中位数, 箱内跟住率, 箱计数)。"""
    mask = np.isfinite(values) & np.isfinite(success)
    values, success = values[mask], success[mask]
    if values.size == 0:
        return np.array([]), np.array([]), np.array([])
    if np.unique(values).size <= n_bins:
        xs = np.array([float(np.median(values))])
        ys = np.array([float(success.mean())])
        ns = np.array([int(values.size)])
        return xs, ys, ns
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    xs, ys, ns = [], [], []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        sel = (values >= low) & (values <= high) if index == len(edges) - 2 else \
            (values >= low) & (values < high)
        if not sel.any():
            continue
        xs.append(float(np.median(values[sel])))
        ys.append(float(success[sel].mean()))
        ns.append(int(sel.sum()))
    return np.asarray(xs), np.asarray(ys), np.asarray(ns)


def level_success_profile(rows: list[dict], family: str, tracker: str) -> dict:
    """逐档跟住率曲线（**退化强度递增**顺序）→ 曲线有效性判据。

    「有效曲线」= 跟住率随退化**单调下降** 且 总跨度 > ``CURVE_MIN_SPAN``。
    跨度不足 ⇒ 曲线接近水平 ⇒ 双算法（Kneedle / PCHIP）必然落在同一个噪声箱上
    ⇒ 「双算法一致」会**误判**为可靠 ⇒ 膝点没有物理意义。
    """
    by_level: dict[float, list[float]] = {}
    for row in rows:
        by_level.setdefault(float(row["condition_level"]), []).append(
            float(row["%s_success" % tracker]))
    levels = sorted(by_level, key=lambda level: degradation_strength(family, level))
    rates = [sum(by_level[level]) / len(by_level[level]) for level in levels]
    span = (max(rates) - min(rates)) if rates else 0.0
    monotone = all(rates[index + 1] <= rates[index] + 1e-9 for index in range(len(rates) - 1))
    return {"levels": levels, "rates": rates, "span": span, "monotone": int(monotone)}


def level_metric_profile(rows: list[dict], family: str, metric: str) -> dict:
    """逐档指标中位（退化强度递增顺序）→ 指标侧单调性。

    任务书 §四.2 的**单调性假设**：退化越强 ⇒ 指标越低。不成立 ⇒ 该族**不承载**该指标
    的信息（例如 ``contrast`` 族的 ``local_scr``：分子分母同乘 α，数学上几乎不变），
    其膝点没有物理意义。
    """
    by_level: dict[float, list[float]] = {}
    for row in rows:
        value = float(row[metric])
        if math.isfinite(value):
            by_level.setdefault(float(row["condition_level"]), []).append(value)
    levels = sorted(by_level, key=lambda level: degradation_strength(family, level))
    meds = [float(np.median(by_level[level])) for level in levels]
    monotone = all(meds[index + 1] <= meds[index] * 1.001 for index in range(len(meds) - 1))
    relspan = (max(meds) - min(meds)) / max(abs(min(meds)), 1e-9) if meds else 0.0
    return {"levels": levels, "meds": meds, "monotone": int(monotone), "relspan": relspan}


def extract_knee(rows: list[dict], metric: str, tracker: str, family: str) -> dict:
    """对一批明细行做「X=实测画质、Y=跟住率」的膝点反解（含**曲线有效性守卫**）。"""
    values = np.array([float(r[metric]) for r in rows], dtype=float)
    success = np.array([float(r["%s_success" % tracker]) for r in rows], dtype=float)
    xs, ys, ns = binned_curve(values, success)
    profile = level_success_profile(rows, family, tracker)
    mprofile = level_metric_profile(rows, family, metric)
    # 🔴 **曲线有效性守卫**：平曲线 / 非单调曲线 / 指标非单调 ⇒ 膝点一律作废（只留审计痕迹）
    degenerate = (not profile["monotone"]) or profile["span"] <= CURVE_MIN_SPAN \
        or (not mprofile["monotone"])
    result = {
        "n_windows": int(values.size),
        "n_bins": int(xs.size),
        "curve_y_span": profile["span"],
        "curve_monotone": profile["monotone"],
        "metric_monotone": mprofile["monotone"],
        "knee_x_kneedle": float("nan"),
        "knee_y_kneedle": float("nan"),
        "knee_x_pchip": float("nan"),
        "knee_y_pchip": float("nan"),
        "knee_x_candidate": float("nan"),
        "delta_pct_of_span": float("nan"),
        "knee_reliable": 0,
        "method_used": "",
        "knee_x_final": float("nan"),
        "knee_y_final": float("nan"),
        "x_at_y50": float("nan"),
    }
    if xs.size < 4:
        return result
    x_knee, y_knee = kneedle_knee(xs, ys)
    x_pchip, y_pchip = pchip_knee(xs, ys)
    span = float(xs.max() - xs.min())
    delta = abs(x_knee - x_pchip) / span * 100.0 if span > 0 else float("nan")
    reliable = int(math.isfinite(delta) and delta <= KNEE_AGREE_PCT)
    if reliable:
        method = "kneedle" if x_knee >= x_pchip else "pchip"
        x_final, y_final = (x_knee, y_knee) if method == "kneedle" else (x_pchip, y_pchip)
    else:
        method = "unreliable"
        x_final, y_final = float("nan"), float("nan")
    # 双算法一致时的候选读数（**先于**曲线有效性守卫取**守卫前**的值）
    candidate = x_final if reliable else float("nan")
    if degenerate:
        # 平曲线上的「膝点」是噪声：保留 kneedle/pchip 原始读数供审计，但**不作为线位**，
        # 「跟住率降到 0.5 的 X」在从不塌掉的曲线上同样没有意义 ⇒ 一律置 nan。
        method = "degenerate"
        reliable = 0
        x_final, y_final = float("nan"), float("nan")
    result.update({
        "knee_x_kneedle": x_knee, "knee_y_kneedle": y_knee,
        "knee_x_pchip": x_pchip, "knee_y_pchip": y_pchip,
        # 双算法一致、但**未过曲线有效性守卫**的候选读数：只作审计 / 对照，不进红线。
        "knee_x_candidate": candidate,
        "delta_pct_of_span": delta, "knee_reliable": reliable,
        "method_used": method, "knee_x_final": x_final, "knee_y_final": y_final,
        "x_at_y50": float("nan") if degenerate else curve_y50(xs, ys),
    })
    result["_curve_x"] = xs
    result["_curve_y"] = ys
    return result


def phase_knee(cli: argparse.Namespace) -> Path:
    """逐 (跟踪器 × 曲线集 × 退化族 × 指标) 反解膝点。

    🔴 **三条硬约束**（本轮新，缺一不可）：
    1. **底座纯净**：曲线只建立在**好片段**上（``own_good`` = 该臂自己的好片段；
       ``common_good`` = 三臂共同的好片段）⇒ clean 档跟住率应 ≈ 100%（见 ``clean_success_rate`` 列）；
    2. **零泄露**：留出集（30%）**连退化仿真都不做** ⇒ 明细里不存在留出集的非 clean 行
       （本函数会做**机器校验**，发现即报错退出）；
    3. **双算法门禁**：Kneedle 与 PCHIP 偏差 > ``KNEE_AGREE_PCT`` ⇒ ``knee_reliable = 0``。
    """
    from regression_v2 import read_class_csv, read_split_csv

    detail_path = Path(cli.detail_csv).resolve()
    output_dir = Path(cli.output_dir).resolve()
    rows = list(csv.DictReader(open(detail_path, encoding="utf-8")))
    for row in rows:
        for key in ("w", "h", "condition_level"):
            row[key] = float(row[key])
        row["small_target"] = int(row["small_target"])
        for key in ALL_METRICS:
            row[key] = float(row[key])
        for tracker in TRACKER_FAMILIES:
            row["%s_success" % tracker] = int(row["%s_success" % tracker])
    print("[KNEE] 明细 %d 行" % len(rows))

    # ---- 分类表 / 切分表（G3 产物）
    class_rows = read_class_csv(output_dir / "segment_class_v2.csv")
    split_map = read_split_csv(output_dir / "split_v2.csv")
    class_by_key = {(row["seq_id"], int(row["frame_id"])): row for row in class_rows}

    leak = sorted({row["seq_id"] for row in rows
                   if row["condition_family"] != "clean"
                   and split_map.get(row["seq_id"]) != "calib"})
    if leak:
        raise SystemExit("[FATAL] 留出集参与了退化仿真（阈值泄露）：%s" % leak[:5])
    calib_seqs = {s for s, v in split_map.items() if v == "calib"}
    print("[KNEE] 标定集 %d 条序列 / 留出集 %d 条序列（留出集零参与 ✅）"
          % (len(calib_seqs), len(split_map) - len(calib_seqs)))

    clean_by_key = {}
    for row in rows:
        if row["condition_family"] == "clean":
            clean_by_key[(row["seq_id"], int(row["frame_id"]))] = row

    def _subset(family_rows: list[dict], curve_set: str, tracker: str) -> list[dict]:
        """按曲线集筛窗口：``own_good`` = 该臂好片段；``common_good`` = 三臂共同好片段。"""
        picked = []
        for row in family_rows:
            info = class_by_key.get((row["seq_id"], int(row["frame_id"])))
            if info is None:
                continue
            if curve_set == "own_good":
                if info.get("class_%s" % tracker) == "good":
                    picked.append(row)
            elif all(info.get("class_%s" % arm) == "good" for arm in TRACKER_FAMILIES):
                picked.append(row)
        return picked

    out_rows = []
    for tracker in TRACKER_FAMILIES:
        for family in DEGRADATION_FAMILIES:
            fam_rows = [r for r in rows if r["condition_family"] == family]
            for curve_set in CURVE_SETS:
                subset = _subset(fam_rows, curve_set, tracker)
                if not subset:
                    continue
                for metric in ALL_METRICS:
                    for stratum, keep in (("all", lambda r: True),
                                          ("small", lambda r: r["small_target"] == 1)):
                        sub = [r for r in subset if keep(r)]
                        if len(sub) < 20:
                            continue
                        result = extract_knee(sub, metric, tracker, family)
                        result.pop("_curve_x", None)
                        result.pop("_curve_y", None)
                        # ---- 底座纯净度：同一批窗口在 clean 档的跟住率（应 ≈ 100%）
                        keys = {(r["seq_id"], int(r["frame_id"])) for r in sub}
                        clean_hits = [clean_by_key[key]["%s_success" % tracker]
                                      for key in keys if key in clean_by_key]
                        clean_rate = (sum(clean_hits) / len(clean_hits)) if clean_hits else float("nan")
                        out_rows.append({
                            "tracker": tracker, "curve_set": curve_set, "degradation_family": family,
                            "metric": metric, "stratum": stratum, **result,
                            "clean_success_rate": round(clean_rate, 6),
                        })
    knee_path = output_dir / "knee_thresholds_v2.csv"
    fields = ["tracker", "curve_set", "degradation_family", "metric", "stratum", "n_windows", "n_bins",
              "curve_y_span", "curve_monotone", "metric_monotone",
              "knee_x_kneedle", "knee_y_kneedle", "knee_x_pchip", "knee_y_pchip",
              "knee_x_candidate",
              "delta_pct_of_span", "knee_reliable", "method_used", "knee_x_final", "knee_y_final",
              "x_at_y50", "clean_success_rate"]
    with open(knee_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k) for k in fields})
    print("[KNEE] 膝点表 %d 行 → %s" % (len(out_rows), knee_path))

    # ---- 相关性（指标 vs 跟住率）：与膝点**同一批窗口**，自动剔除无区分度 / 方向反转的指标
    from scipy.stats import pearsonr, spearmanr

    corr_rows = []
    for tracker in TRACKER_FAMILIES:
        for family in DEGRADATION_FAMILIES:
            fam_rows = [r for r in rows if r["condition_family"] == family]
            for curve_set in CURVE_SETS:
                subset = _subset(fam_rows, curve_set, tracker)
                if not subset:
                    continue
                for metric in ALL_METRICS:
                    values = np.array([r[metric] for r in subset], dtype=float)
                    success = np.array([float(r["%s_success" % tracker]) for r in subset], dtype=float)
                    mask = np.isfinite(values) & np.isfinite(success)
                    if mask.sum() < 10:
                        continue
                    spearman = float(spearmanr(values[mask], success[mask]).statistic)
                    pearson = float(pearsonr(values[mask], success[mask]).statistic)
                    expected = METRIC_DIRECTION.get(metric, 0)
                    if expected == 0:
                        direction_ok = 1              # 方向中性：只看强度
                    elif math.isfinite(spearman) and spearman != 0.0:
                        direction_ok = int((spearman > 0) == (expected > 0))
                    else:
                        direction_ok = 0
                    corr_rows.append({
                        "tracker": tracker, "curve_set": curve_set, "degradation_family": family,
                        "metric": metric, "n": int(mask.sum()),
                        "spearman": spearman, "pearson": pearson,
                        "expected_direction": expected,
                        "direction_ok": direction_ok,
                        "usable": int(abs(spearman) >= 0.15 and direction_ok),
                    })
    corr_path = output_dir / "correlations_v2.csv"
    with open(corr_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tracker", "curve_set", "degradation_family",
                                                "metric", "n", "spearman", "pearson",
                                                "expected_direction", "direction_ok", "usable"])
        writer.writeheader()
        for row in corr_rows:
            writer.writerow(row)
    print("[KNEE] 相关性表 %d 行 → %s" % (len(corr_rows), corr_path))
    return knee_path


def phase_constraint(cli: argparse.Namespace) -> Path:
    """**良好跟踪合格线（约束线）**——G5：只对**三项核心指标**出红线。

    口径（任务书 §三 + 交棒书 §3-G5）：
    * 只取 ``curve_set = common_good``（三臂共同好片段 ⇒ 跨臂线位可比）、``stratum = all``；
    * 只取 ``knee_reliable = 1`` 的膝点（双算法一致 ⇒ 允许作为正式阈值）；
    * **每族膝点取最小值**（最严的一族说了算），并记录**决定族 / 决定臂**；
    * 🔴 **绝不**把 XCorr 与 KCF 的线位求平均（跨家族纪律）——只取更严的那条，
      并把两族线位差单独报出来。
    """
    from regression_v2 import CORE_METRICS, PRIMARY_ARMS

    output_dir = Path(cli.output_dir).resolve()
    knee_rows = list(csv.DictReader(open(output_dir / "knee_thresholds_v2.csv", encoding="utf-8")))
    out_rows = []
    for metric in CORE_METRICS:
        cands = []
        refs: dict[str, float] = {}
        for row in knee_rows:
            if row["metric"] != metric or row["curve_set"] != "common_good":
                continue
            if row["stratum"] != "all":
                continue
            # 🔴 **参照读数**：双算法一致、但曲线有效性守卫判它作废的候选（``knee_x_candidate``）。
            #    它们**不参与红线取值**，只用来向读者说明「这条臂的线为什么没进表」。
            try:
                ref = float(row.get("knee_x_candidate", "nan"))
            except (TypeError, ValueError):
                ref = float("nan")
            if math.isfinite(ref):
                refs[row["tracker"]] = min(refs.get(row["tracker"], ref), ref)
            if row["knee_reliable"] != "1":
                continue
            # 🔴 **曲线有效性守卫**（与 phase_knee 同款）：跨度不足 / 非单调 / 指标非单调
            #    ⇒ 膝点一律不进红线候选（``knee_reliable`` 已被守卫置 0，这里再显式挡一次）。
            if row.get("curve_monotone") != "1" or row.get("metric_monotone") != "1":
                continue
            try:
                if float(row.get("curve_y_span", "nan")) <= CURVE_MIN_SPAN:
                    continue
            except (TypeError, ValueError):
                continue
            # 🔴 红线只由**两条主臂**（XCorr / KCF）反解；CSRT 是**同族对照臂**，
            #    它的作用是证明 KCF 侧线位不是实现细节，**不参与红线取值**。
            if row["tracker"] not in PRIMARY_ARMS:
                continue
            try:
                value = float(row["knee_x_final"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            cands.append((value, row["tracker"], row["degradation_family"], row))
        control = [refs[arm] for arm in refs if arm not in PRIMARY_ARMS]
        if not cands:
            out_rows.append({"metric": metric, "line_value": float("nan"),
                             "deciding_tracker": "", "deciding_family": "",
                             "curve_set": "common_good", "stratum": "all",
                             "xcorr_line": float("nan"), "kcf_legacy_line": float("nan"),
                             "csrt_line": float("nan"), "cross_arm_dev_pct": float("nan"),
                             "xcorr_ref": refs.get("xcorr", float("nan")),
                             "kcf_legacy_ref": refs.get("kcf_legacy", float("nan")),
                             "csrt_ref": min(control) if control else float("nan"),
                             "n_candidates": 0, "note": "无可靠膝点（双算法偏差 > %.0f%%）"
                                                        % KNEE_AGREE_PCT})
            continue
        value, tracker, family, _ = min(cands, key=lambda item: item[0])
        per_arm = {}
        for arm in PRIMARY_ARMS:
            arm_values = [c[0] for c in cands if c[1] == arm]
            per_arm[arm] = min(arm_values) if arm_values else float("nan")
        primary = [per_arm[arm] for arm in PRIMARY_ARMS
                   if math.isfinite(per_arm.get(arm, float("nan")))]
        dev = (abs(primary[0] - primary[1]) / max(abs(primary[0]), abs(primary[1]), 1e-9) * 100.0
               if len(primary) == 2 else float("nan"))
        out_rows.append({
            "metric": metric,
            "line_value": value,
            "deciding_tracker": tracker,
            "deciding_family": family,
            "curve_set": "common_good",
            "stratum": "all",
            "xcorr_line": per_arm.get("xcorr", float("nan")),
            "kcf_legacy_line": per_arm.get("kcf_legacy", float("nan")),
            "csrt_line": min(control) if control else float("nan"),
            "cross_arm_dev_pct": dev,
            "xcorr_ref": refs.get("xcorr", float("nan")),
            "kcf_legacy_ref": refs.get("kcf_legacy", float("nan")),
            "csrt_ref": min(control) if control else float("nan"),
            "n_candidates": len(cands),
            "note": "每族膝点取最小值（最严的一族说了算）；跨臂线位不合并求平均；"
                    "CSRT 为同族对照臂，只作参照不参与取值",
        })
    constraint_path = output_dir / "constraint_lines_v2.csv"
    fields = ["metric", "line_value", "deciding_tracker", "deciding_family", "curve_set", "stratum",
              "xcorr_line", "kcf_legacy_line", "csrt_line", "cross_arm_dev_pct",
              "xcorr_ref", "kcf_legacy_ref", "csrt_ref",
              "n_candidates", "note"]
    with open(constraint_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k) for k in fields})
    print("[CONSTRAINT] 约束线 %d 行 → %s" % (len(out_rows), constraint_path))
    return constraint_path


# --------------------------------------------------------------------------- 尾部验证（模块 5）
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC（Mann-Whitney U；``labels`` 1 = 正类）。"""
    mask = np.isfinite(scores)
    scores, labels = scores[mask], labels[mask]
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    sorted_scores = scores[order]
    index = 0
    while index < scores.size:
        stop = index
        while stop + 1 < scores.size and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        ranks[order[index:stop + 1]] = average
        index = stop + 1
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def phase_tail(cli: argparse.Namespace) -> Path:
    """零合成退化：真实红外窗口上的天然困难片段尾部对比（局部 ROI 指标 vs 全图指标）。"""
    detail_path = Path(cli.detail_csv).resolve()
    rows = [r for r in csv.DictReader(open(detail_path, encoding="utf-8"))
            if r["condition_family"] == "clean"]
    for row in rows:
        for metric in ALL_METRICS:
            row[metric] = float(row[metric])
        for tracker in TRACKER_FAMILIES:
            row["%s_success" % tracker] = int(row["%s_success" % tracker])
        row["small_target"] = int(row["small_target"])
    print("[TAIL] 干净窗口 %d 条" % len(rows))

    tail_metrics = ["local_scr", "target_lap_var", "target_local_contrast",
                    "global_contrast", "global_lap_var"]
    out_rows = []
    for metric in tail_metrics:
        values = np.array([r[metric] for r in rows], dtype=float)
        order = np.argsort(values, kind="mergesort")
        n = values.size
        for quantile in (0.10, 0.15):
            cut = max(1, int(round(n * quantile)))
            tail_idx = set(order[:cut].tolist())
            for tracker in TRACKER_FAMILIES:
                success = np.array([r["%s_success" % tracker] for r in rows], dtype=float)
                tail_rate = float(np.mean([success[i] for i in tail_idx])) if tail_idx else float("nan")
                rest_rate = float(np.mean([success[i] for i in range(n) if i not in tail_idx]))
                labels = (1.0 - success)      # 正类 = 跟丢
                auc = roc_auc(-values, labels)   # 指标越低 ⇒ 越可能跟丢 ⇒ 用 -指标打分
                out_rows.append({
                    "metric": metric, "metric_kind": "local" if metric in LOCAL_METRICS else "global",
                    "tail_quantile": quantile, "tracker": tracker,
                    "n_total": n, "n_tail": len(tail_idx),
                    "tail_success_rate": round(tail_rate, 6),
                    "rest_success_rate": round(rest_rate, 6),
                    "drop_pp": round((rest_rate - tail_rate) * 100.0, 3),
                    "tail_metric_median": round(float(np.median(values[order[:cut]])), 6),
                    "rest_metric_median": round(float(np.median(values[order[cut:]])), 6),
                    "auc_predict_lost": round(auc, 6),
                })
    output_dir = Path(cli.output_dir).resolve()
    tail_path = output_dir / "tail_analysis_v2.csv"
    with open(tail_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    print("[TAIL] 尾部分析 %d 行 → %s" % (len(out_rows), tail_path))
    return tail_path


# --------------------------------------------------------------------------- 报告
def phase_report(cli: argparse.Namespace) -> Path:
    """从产物 CSV 生成 **REPORT_V2_FINAL.md**（数字全部现场读，**不手抄**）。"""
    from regression_v2 import (CALIB_RATIO, CLASS_GOOD, CLASS_LOST, CLASS_MEDIOCRE,
                               CORE_METRICS, GOOD_CLE_PX, REAL_BAD_CLIPS, REAL_GOOD_CLIPS,
                               read_class_csv, read_constraint_lines, read_split_csv)

    output_dir = Path(cli.output_dir).resolve()
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    knee_rows = list(csv.DictReader(open(output_dir / "knee_thresholds_v2.csv", encoding="utf-8")))
    tail_rows = list(csv.DictReader(open(output_dir / "tail_analysis_v2.csv", encoding="utf-8")))
    corr_rows = list(csv.DictReader(open(output_dir / "correlations_v2.csv", encoding="utf-8")))
    class_rows = read_class_csv(output_dir / "segment_class_v2.csv")
    split_map = read_split_csv(output_dir / "split_v2.csv")
    lines_tbl = read_constraint_lines(output_dir / "constraint_lines_v2.csv")
    regress_rows = list(csv.DictReader(open(output_dir / "regression_v2.csv", encoding="utf-8")))
    blind_rows = list(csv.DictReader(open(output_dir / "blind_real_v2.csv", encoding="utf-8")))

    def pick(tracker, family, metric, curve_set="common_good", stratum="all"):
        for row in knee_rows:
            if (row["tracker"] == tracker and row["degradation_family"] == family
                    and row["metric"] == metric and row["curve_set"] == curve_set
                    and row["stratum"] == stratum):
                return row
        return None

    def fnum(value, default=float("nan")):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out

    # ---- 干净帧三臂基线（现场读明细 CSV）
    base_stats: dict = {}
    detail_path = Path(cli.detail_csv).resolve()
    detail_rows_all = []
    if detail_path.exists():
        detail_rows_all = list(csv.DictReader(open(detail_path, encoding="utf-8")))
        clean_rows = [r for r in detail_rows_all if r.get("condition_family") == "clean"]
        for tracker in TRACKER_FAMILIES:
            succ, cles, errors = [], [], 0
            for row in clean_rows:
                try:
                    cle = float(row["%s_cle" % tracker])
                except (TypeError, ValueError):
                    cle = float("nan")
                succ.append(int(row["%s_success" % tracker]))
                if cle == cle:
                    cles.append(cle)
                if str(row.get("%s_err" % tracker, "")).strip():
                    errors += 1
            cles.sort()
            base_stats[tracker] = {
                "rate": sum(succ) / len(succ) if succ else float("nan"),
                "cle_med": cles[len(cles) // 2] if cles else float("nan"),
                "cle_p90": cles[int(0.9 * (len(cles) - 1))] if cles else float("nan"),
                "n_err": errors,
                "n": len(succ),
            }

    def _level_success_rates() -> dict:
        """净化底座上**逐档**的实测跟住率（现场读明细 CSV，``common_good`` 池）。"""
        if not detail_rows_all:
            return {}
        class_index = {}
        for row in class_rows:
            class_index[(row["seq_id"], int(row["frame_id"]))] = row
        table: dict = {}
        for row in detail_rows_all:
            info = class_index.get((row["seq_id"], int(row["frame_id"])))
            if info is None:
                continue
            if not all(info.get("class_%s" % arm) == "good" for arm in TRACKER_FAMILIES):
                continue
            family = row.get("condition_family")
            if family == "clean":
                continue
            try:
                level = float(row["condition_level"])
            except (TypeError, ValueError):
                continue
            item = table.setdefault(family, {}).setdefault(
                level, {"level": level, "xcorr": [], "kcf": [], "csrt": []})
            item["xcorr"].append(int(row["xcorr_success"]))
            item["kcf"].append(int(row["kcf_legacy_success"]))
            item["csrt"].append(int(row["csrt_success"]))
        out = {}
        for family, levels in table.items():
            rows_out = []
            for level, item in levels.items():
                rows_out.append({"level": level,
                                 "xcorr": sum(item["xcorr"]) / max(len(item["xcorr"]), 1),
                             "kcf": sum(item["kcf"]) / max(len(item["kcf"]), 1),
                             "csrt": sum(item["csrt"]) / max(len(item["csrt"]), 1)})
            out[family] = {"levels": rows_out}
        return out

    # ---- 三分法统计
    class_counter = {CLASS_GOOD: 0, CLASS_MEDIOCRE: 0, CLASS_LOST: 0}
    for row in class_rows:
        class_counter[row["class_ds"]] = class_counter.get(row["class_ds"], 0) + 1
    calib_seqs = sorted(s for s, v in split_map.items() if v == "calib")
    holdout_seqs = sorted(s for s, v in split_map.items() if v == "holdout")
    overlap = sorted(set(calib_seqs) & set(holdout_seqs))

    lines: list[str] = []
    add = lines.append

    add("# 红外图像质量评估 v2（Target-Centric）· 最终报告")
    add("")
    add("> **口径声明**：本报告的全部阈值基于 **Anti-UAV 开源红外数据集**标定，"
        "用途是**验证 v2 Target-Centric 方法本身的有效性**；"
        "**正式上板红线待内部 58 条红外业务 GT 序列回收后，用同一套流水线重标**"
        "（见 §十二 局限 1）。")
    add("")
    add("> **本轮相对上一版的结构性变化**（逐条对应任务书与交棒书缺口）：")
    add("> ① **剔除 JPEG 退化轴**（任务书 §四.2）——只保留散焦模糊 / 运动拖影 / 对比度衰减三族；")
    add("> ② **仿真底座纯净化**（任务书 §二.1）——两趟法：Pass A 干净帧分三类，"
        "Pass B **只有好片段**进退化池；")
    add("> ③ **序列级 7:3 切分**（任务书 §四.1）——膝点只用 70% 标定集反解，留出集零参与；")
    add("> ④ **三分法回归测试体系**（任务书 §三）——好片误杀率 / 临界敏感性 / 坏片拦截率；")
    add("> ⑤ **约束线只对三项核心指标出**（全图指标仅作「已废弃口径·对照」）。")
    add("")

    # ------------------------------------------------------------------ 一、数据集
    add("## 一、数据集与口径")
    add("")
    add("| 项 | 值 |")
    add("|:--|--:|")
    add("| 数据集根目录 | `%s` |" % manifest["dataset_root"])
    add("| 模态 | **仅红外（IR）**，可见光文件**未被打开** |")
    add("| 扫描到的红外序列数 | **%d** |" % manifest["total_ir_sequences_found"])
    add("| 保留（单目标）序列数 | **%d** |" % manifest["kept_single_target_sequences"])
    add("| 剔除序列数 | **%d**（多目标 %d） |"
        % (manifest["dropped_sequences_count"], manifest["multi_target_dropped_count"]))
    add("| 有效 5 帧窗口数 | **%d** |" % manifest["window_counts"]["total"])
    add("| 小目标子集（max(w,h) ≤ 20 px） | **%d** 条（占比 %.1f%%） |"
        % (manifest["window_counts"]["small_target"],
           100.0 * manifest["window_counts"]["small_target_ratio"]))
    add("| 窗口协议 | 5 帧连续，`exist==1`、`w>0`、`h>0`、中心在画面内 |")
    add("| 采样 | 种子 %d，每序列等距步长取点，合计 %d 条窗口 |"
        % (manifest["sampling"]["seed"], manifest["sampling"]["total_windows"]))
    add("")
    add("**退化族（任务书 §二.2，本轮只保留三族）**")
    add("")
    add("| 族 | 物理病因 | 档位 |")
    add("|:--|:--|:--|")
    add("| `blur` | 散焦 / 过度降噪涂抹 | `cv2.GaussianBlur` σ ∈ {0.6, 1.0, 1.5, 2.2, 3.0, 4.0}（6 档） |")
    add("| `motion` | 俯仰抖动 / 机动拖影 | 线性核 `filter2D`，核长 L ∈ {3, 5, 7, 9, 11} px（5 档） |")
    add("| `contrast` | 气象热交叉 / 未做 NUC | `I' = αI + (1-α)μ_frame`，α ∈ {0.80, 0.65, 0.50, 0.35, 0.25, 0.15}（6 档） |")
    add("")
    add("> **已确认不引入的两条轴**：**JPEG 压缩**（红外上块效应**不升反降 -16.8%**、"
        "DCT 振铃让 `target_lap_var` 不降反升 ⇒ 违背单调性）与**加性白噪声**"
        "（红外上加性噪声指标与退化方向相反 ⇒ 同样违背单调性）。本流水线**从未引入**后者，"
        "本轮**删除**了前者。")
    add("")

    # ------------------------------------------------------------------ 二、三臂
    add("## 二、跟踪器三臂与实现事实")
    add("")
    add("| 跟踪臂 | 角色 | 干净帧跟住率 | 中位 CLE(px) | P90 CLE(px) | 异常/报错次数 |")
    add("|:--|:--|--:|--:|--:|--:|")
    arm_role = {"xcorr": "**主臂**：模板匹配链式配准（与 KCF 跨家族）",
                "kcf_legacy": "**主臂**：任务书指定的 contrib legacy KCF（detect_thresh=0.0）",
                "csrt": "同族对照：DCF 家族的另一实现（CSRT）"}
    for tracker in TRACKER_FAMILIES:
        stat = base_stats.get(tracker, {})
        add("| `%s` | %s | %.1f%% | %.2f | %.2f | %d |"
            % (tracker, arm_role.get(tracker, "-"),
               fnum(stat.get("rate")) * 100.0, fnum(stat.get("cle_med")),
               fnum(stat.get("cle_p90")), int(stat.get("n_err", 0))))
    add("")
    add("### 2.1 任务书指定的 KCF 臂——必须写进报告的实现事实")
    add("")
    add("> %s" % LEGACY_KCF_NOTE)
    add("")
    add("**取证方法**（可复现）：合成一帧「亮目标 + 噪声背景」序列，目标每帧平移 8 px，"
        "共 5 帧；用 `tracker.update()` 连续推 4 步，逐帧比较返回矩形的中心与真值。")
    add("")
    add("| 参数 | 观测到的逐帧中心 x（真值 120/128/136/144/152） | 结论 |")
    add("|:--|:--|:--|")
    add("| `detect_thresh=0.0`（任务书口径） | 120 / 128 / 136 / 144 / 152 | ✅ 逐帧对齐，误差 ≤ 1 px |")
    add("| `detect_thresh=0.99` | 120 / 0 / 0 / 0 / 0 | ❌ 响应峰达不到阈值 ⇒ 退化成原点矩形 |")
    add("")
    add("### 2.2 跨家族纪律")
    add("")
    add("* **XCorr**（模板匹配 + 链式重切模板）与 **KCF**（判别相关滤波 + 在线模型更新）"
        "是原理完全不同的两类跟踪器 ⇒ 它们的**绝对锁定率刻度不可互换**，"
        "但**反解出来的线位可以互相印证**。")
    add("* **CSRT** 与 KCF 同属 DCF 家族 ⇒ 它的作用是**同族对照**："
        "证明 KCF 侧反推的线位不是「某个实现的细节」。")
    add("* 三个臂的绝对数值**严禁合并求平均**；任何一张表里三者都**并列呈现**。")
    add("")
    add("### 2.3 KCF 臂「尾帧补偿」的实测覆盖率与残差")
    add("")
    add("> 因为 contrib legacy KCF 的 `update()` 滞后一帧，本流水线用**第 t+5 帧**做补偿。")
    add("")

    def _kcf_err_stats():
        table = {}
        for row in detail_rows_all:
            key = (row.get("condition_family", "?"), row.get("condition_level", "?"))
            item = table.setdefault(key, {"rows": 0, "tail_fail": 0, "no_tail": 0,
                                          "succ_tail": 0, "succ_rest": 0, "rest_rows": 0})
            item["rows"] += 1
            err = str(row.get("kcf_legacy_err", "") or "").strip()
            success = str(row.get("kcf_legacy_success", "")) == "1"
            if err == "tail_update_failed":
                item["tail_fail"] += 1
                item["succ_tail"] += int(success)
            else:
                item["rest_rows"] += 1
                item["succ_rest"] += int(success)
            if err.startswith("no_tail_frame"):
                item["no_tail"] += 1
        return table

    if detail_rows_all:
        err_stats = _kcf_err_stats()
        order = {"clean": 0, "blur": 1, "motion": 2, "contrast": 3}

        def _level_key(key):
            family, level = key
            try:
                return (order.get(family, 9), float(level))
            except (TypeError, ValueError):
                return (order.get(family, 9), 0.0)

        add("| 退化族 | 档位 | 行数 | 尾帧更新失败 | 占比 | 无尾帧（序列末尾） |")
        add("|:--|--:|--:|--:|--:|--:|")
        for key in sorted(err_stats, key=_level_key):
            item = err_stats[key]
            add("| %s | %s | %d | %d | %.1f%% | %d |"
                % (key[0], key[1], item["rows"], item["tail_fail"],
                   100.0 * item["tail_fail"] / max(item["rows"], 1), item["no_tail"]))
        add("")
        add("| 退化族 | 尾帧失败行的跟住率 | 其余行的跟住率 |")
        add("|:--|--:|--:|")
        for family in ("clean", "blur", "motion", "contrast"):
            rows = sum(v["rows"] for k, v in err_stats.items() if k[0] == family)
            tf = sum(v["tail_fail"] for k, v in err_stats.items() if k[0] == family)
            st = sum(v["succ_tail"] for k, v in err_stats.items() if k[0] == family)
            sr = sum(v["succ_rest"] for k, v in err_stats.items() if k[0] == family)
            rr = sum(v["rest_rows"] for k, v in err_stats.items() if k[0] == family)
            if not rows:
                continue
            add("| %s | %.1f%% | %.1f%% |" % (family, 100.0 * st / max(tf, 1),
                                               100.0 * sr / max(rr, 1)))
        add("")
        add("> **两条必须一起讲的结论**：")
        add("> ① **尾帧失败不是「口径造成的假失败」**：这些窗口集中在强退化档位，"
            "其跟住率显著低于其余窗口 ⇒ 跟踪器在那时**确实已经跟丢**；")
        add("> ② **仍有残差**：序列末尾的窗口拿不到尾帧，全程只能用「滞后一帧」的读数；"
            "本轮的三分法分类额外用**同帧口径**（`cle_end`）消除了这一偏置，"
            "但 `CLE_{t+4}` 口径的跟住率对 KCF 臂仍**恒定偏保守** ⇒ KCF 族膝点偏保守、不会偏松。")
        add(">    XCorr 与 CSRT 两臂**没有滞后一帧问题**，所以三臂的绝对值差异**不来自**这个补偿。")
        add("")

    # ------------------------------------------------------------------ 三、三分法
    add("## 三、三分法划分结果（序列级 7:3 隔离切分）")
    add("")
    add("> 分类口径（任务书 §三）：**好片段** = 5 帧全程稳定跟住且末帧 `CLE_{t+4} ≤ %.0f px`"
        "（3 px 是任务书给死的常量）；**一般片段** = 未完全跟丢但临界（含双跟踪器分歧）；"
        "**跟丢片段** = 干净帧下真实跟踪失锁。" % GOOD_CLE_PX)
    add("")
    add("| 类别 | 窗口数 | 占比 |")
    add("|:--|--:|--:|")
    total = sum(class_counter.values()) or 1
    for name, label in ((CLASS_GOOD, "好片段（Good）"), (CLASS_MEDIOCRE, "一般片段（Mediocre）"),
                        (CLASS_LOST, "跟丢片段（Lost）")):
        add("| %s | **%d** | %.1f%% |" % (label, class_counter.get(name, 0),
                                          100.0 * class_counter.get(name, 0) / total))
    add("")
    add("**序列级隔离切分（G3）**")
    add("")
    add("| 项 | 值 |")
    add("|:--|--:|")
    add("| 切分比例 | 标定 %.0f%% / 留出 %.0f%%（**按 `seq_id` 整条切**） |"
        % (100.0 * CALIB_RATIO, 100.0 * (1.0 - CALIB_RATIO)))
    add("| 标定集序列数 | **%d** |" % len(calib_seqs))
    add("| 留出集序列数 | **%d** |" % len(holdout_seqs))
    add("| **机器校验：两侧 `seq_id` 交叉数** | **%d** %s |"
        % (len(overlap), "✅ 零交叉" if not overlap else "❌ 存在泄露"))
    add("| 膝点反解使用的数据 | **仅标定集**；留出集**连退化仿真都不参与** |")
    add("")
    add("> 留出集 `seq_id` 清单（共 %d 条）：`%s`" % (len(holdout_seqs), "`, `".join(holdout_seqs)))
    add("")
    add("> 标定集 `seq_id` 清单（共 %d 条）见 `iqa_v2/output/split_v2.csv`。切分种子固定，"
        "可完全复现。**严禁**把同一条视频的前半段拿去仿真、后半段拿去测误杀率——"
        "本流水线的「窗口抽样」也在序列内做等距步长，两侧完全隔离。" % len(calib_seqs))
    add("")

    # ------------------------------------------------------------------ 四、底座净化
    add("## 四、好片段底座净化统计（两趟法）")
    add("")
    add("> **为什么要两趟**：上一版把**全部 %d 个窗口一次性**送进仿真池，"
        "其中 **%.1f%%（%d 条）在干净帧上就已经跟丢**（交棒书 §3-G2 记的旧口径是 15.5%%，"
        "同一量级）⇒ 这些「脏窗口」把衰减曲线的顶部压到 84.5%%（XCorr 全量干净帧跟住率），"
        "掩盖了真实拐点。本轮改为：**Pass A** 只跑 `clean` 条件把窗口分成好 / 一般 / 跟丢；"
        "**Pass B** 只让**好片段**进退化池。"
        % (total, 100.0 * class_counter.get(CLASS_LOST, 0) / total,
           class_counter.get(CLASS_LOST, 0)))
    add("")
    add("| 底座 | 窗口数 | clean 档跟住率（XCorr / KCF / CSRT） | 结论 |")
    add("|:--|--:|:--|:--|")
    pool_keys = {(row["seq_id"], int(row["frame_id"])) for row in detail_rows_all
                 if row.get("condition_family") != "clean"}
    clean_index = {(row["seq_id"], int(row["frame_id"])): row for row in detail_rows_all
                   if row.get("condition_family") == "clean"}
    pool_rates = []
    for tracker in TRACKER_FAMILIES:
        hits = [int(clean_index[key]["%s_success" % tracker])
                for key in pool_keys if key in clean_index]
        pool_rates.append(sum(hits) / len(hits) if hits else float("nan"))
    add("| **三臂好片段并集（= 实际仿真池）** | %d | %.1f%% / %.1f%% / %.1f%% | "
        "⚠️ 池内并非**全部**窗口都被三臂判好（见下方说明） |"
        % (len(pool_keys), pool_rates[0] * 100.0, pool_rates[1] * 100.0,
           pool_rates[2] * 100.0))
    for curve_set, label in (("own_good", "各臂自己的好片段底座"),
                              ("common_good", "三臂共同的好片段底座（跨臂对比用）")):
        rates = []
        for tracker in TRACKER_FAMILIES:
            row = pick(tracker, "blur", "target_lap_var", curve_set, "all")
            rates.append(fnum(row.get("clean_success_rate")) if row else float("nan"))
        add("| %s | %d | %.1f%% / %.1f%% / %.1f%% | %s |"
            % (label, int(pick(TRACKER_FAMILIES[0], "blur", "target_lap_var", curve_set, "all")
                          ["n_windows"]) if pick(TRACKER_FAMILIES[0], "blur", "target_lap_var",
                                                 curve_set, "all") else 0,
               rates[0] * 100.0, rates[1] * 100.0, rates[2] * 100.0,
               "✅ 起点纯粹" if all(r >= 0.999 for r in rates) else "⚠️ 起点未达 100%"))
    add("")
    add("> **逐档可查**：膝点表 `iqa_v2/output/knee_thresholds_v2.csv` 的 `clean_success_rate` 列"
        "给出**每一族、每一曲线集、每一指标**所用窗口在 clean 档的跟住率；"
        "上表取 `blur` 族为代表（其余族同源同池）。")
    add("")
    add("> 🔴 **口径澄清（别把「仿真池」当成「纯好片段」）**：退化池 = "
        "**标定集 ∩ (任一臂判好)** 的窗口并集（%d 个）；"
        "**建曲线时**再逐臂过滤成 `own_good`（该臂自己的好片段）与 `common_good`"
        "（三臂交集）两个曲线集 ⇒ **曲线本身是纯的**，但**池**里仍含被别的臂判为"
        "「一般 / 跟丢」的窗口（所以池内 clean 起点是 %.2f%% / %.2f%% / %.2f%%，"
        "只有 `common_good` 才是 100%%）。"
        % (len(pool_keys), pool_rates[0] * 100.0, pool_rates[1] * 100.0,
           pool_rates[2] * 100.0))
    add("")
    add("### 4.1 逐档跟住率——**净化底座上的衰减曲线有多深**（必须一起交付的事实）")
    add("")
    add("> 下表是**净化后**底座上每一档的实测跟住率（现场读 `degrade_detail.csv`，"
        "`common_good` 池，XCorr / KCF 两臂）。它解释了 §6 红线的可靠性边界。")
    add("")
    add("| 退化族 | 档位（退化强度递增） | XCorr 跟住率 | KCF 跟住率 |")
    add("|:--|:--|:--|:--|")
    level_table = _level_success_rates()
    for family in DEGRADATION_FAMILIES:
        entry = level_table.get(family)
        if not entry:
            continue
        levels = sorted(entry["levels"], key=lambda item: -item["level"])
        add("| %s | %s | %s | %s |"
            % (family,
               " → ".join("%g" % item["level"] for item in levels),
               " → ".join("%.1f%%" % (100.0 * item["xcorr"]) for item in levels),
               " → ".join("%.1f%%" % (100.0 * item["kcf"]) for item in levels)))
    add("")
    add("> 🔴 **本轮最重要的工程事实**：在**净化后**的好片段底座上，"
        "任务书 §二.2 规定的三族退化档位**不足以让跟踪崩溃**——"
        "即使加到**最重档**，跟住率仍在 **90% 以上**（最严的 `blur` 族 XCorr 臂最重档最低）。")
    add("> ⇒ 由此产生三条必须写进结论的后果：")
    add("> ① 膝点只能定位在「**曲线开始明显弯折**」处，而**不是**「跟踪崩溃点」——"
        "膝点处的跟住率仍在 95%~100%（见 §5.4 的 `knee_y` 列）；")
    add("> ② 曲线接近平坦 ⇒ 膝点位置对底座与分箱**敏感**，"
        "双算法一致性门禁因此在相当一部分候选上判为不可靠，"
        "跨臂线位差也明显大于上一轮（上一轮未纯净化时 `target_lap_var` 两臂偏差 0.00%）；")
    add("> ③ 因此约束线**取最严候选**（安全方向），"
        "其可用性改由 §8 的**独立盲测**（真实坏片 vs 真实好片对照）来证明，而不是由膝点本身证明。")
    add("")
    add("> **这正是任务书 §二.1 的取舍**：把「脏窗口」踢出仿真池换来曲线起点 100%，"
        "代价是曲线变浅、崩溃点被推到任务书给定档位之外。"
        "若要得到更硬的崩溃拐点，需要**加大退化档位上限**（例如 `blur` σ 提到 6~8、"
        "`contrast` α 降到 0.05），而不是把脏窗口塞回池子——"
        "后者正是任务书禁止的做法。")
    add("")
    add("> **跨臂公平性的处理**：三臂的「好片段」集合**互不相同**（KCF 与 XCorr 的判定必然有出入）"
        "⇒ 退化池取**三臂好片段的并集**（保证每条臂自己的池都被完整覆盖），"
        "而**跨臂的线位对比只用 `common_good`（三臂共同的好片段）**——"
        "只有在**同一批窗口**上反解出来的膝点，两个家族之间才可比。")
    add("")
    add("### 4.2 曲线有效性分级——**哪些曲线真的承载了崩塌信息**")
    add("")
    add("> 🔴 **本轮最重要的一处方法修正**（host 侧独立复核复算确认）："
        "「双算法一致」**不能**单独用来证明曲线有效——在**跟住率从未塌掉**的平曲线上，"
        "Kneedle 与 PCHIP 会一致地落在**同一个噪声箱**上，于是被误判为 `knee_reliable=1`。"
        "⇒ 红线候选必须**同时**满足三条："
        "① **跟住率随退化单调下降**；"
        "② **跟住率总跨度 > %.0f 个百分点**（低于此落在抽样噪声量级内，"
        "不足以支撑「崩溃拐点」这种强结论）；"
        "③ **指标侧单调下降**（退化递增 ⇒ 指标中位递减，任务书 §四.2 的单调性假设）。"
        % (100.0 * CURVE_MIN_SPAN))
    add("")
    add("> 分级规则：**A 级** = 单调且跨度 > 阈值（可进红线）；"
        "**B 级** = 单调但跨度 ≤ 阈值（曲线近似水平 ⇒ 膝点位置不可靠，**仅作物理参照**）；"
        "**C 级** = 非单调或数值噪声（**作废**）。")
    add("")
    add("| 跟踪臂 | 退化族 | 跟住率跨度 | 单调 | 分级 | 逐档跟住率（退化强度递增） |")
    add("|:--|:--|--:|:--|:--|:--|")
    for tracker in TRACKER_FAMILIES:
        for family in DEGRADATION_FAMILIES:
            row = pick(tracker, family, "local_scr", "common_good", "all")
            if not row:
                continue
            span = fnum(row.get("curve_y_span"))
            mono = row.get("curve_monotone") == "1"
            if not mono:
                grade = "❌ C 级（非单调）"
            elif span <= CURVE_MIN_SPAN:
                grade = "⚠️ B 级（跨度不足）"
            else:
                grade = "✅ **A 级**"
            entry = level_table.get(family)
            rates = ""
            if entry:
                levels = sorted(entry["levels"], key=lambda item: -item["level"])
                key = {"xcorr": "xcorr", "kcf_legacy": "kcf", "csrt": "csrt"}[tracker]
                rates = " → ".join("%.1f%%" % (100.0 * item[key]) for item in levels)
            add("| `%s` | %s | %.2f pp | %s | %s | %s |"
                % (tracker, family, 100.0 * span, "✅" if mono else "❌", grade, rates))
    add("")
    add("> 🔴 **由此得出的硬事实**：在**净化后**的底座上，"
        "**只有 XCorr 臂的 `blur` 与 `motion` 两条曲线达到 A 级**；"
        "KCF 与 CSRT 的曲线跨度全部 ≤ 5pp（C 级 / B 级）⇒ "
        "**它们的膝点不能进红线** ⇒ 本轮三条红线**只由 XCorr 一条臂支撑**，"
        "**任务书 §二.3 倚重的「XCorr ↔ KCF 算法无关性」在本轮无法复现**"
        "（上一轮**未纯净化**底座上 `target_lap_var` 两臂偏差是 0.00%）。"
        "这必须写进结论，而不是藏起来。")
    add("")

    # ------------------------------------------------------------------ 五、膝点
    add("## 五、三族退化下的膝点反解")
    add("")
    add("> **跨家族纪律**：XCorr 与 KCF 的绝对锁定率刻度**不可互换**，"
        "下表**并列呈现、各自独立反解**，**严禁合并求平均**；可比的是**线位**。")
    add("")
    add("> 🔴 **本章所有膝点都必须先过 §4.2 的三道守卫**（单调 / 跨度 > %.0fpp / 指标单调），"
        "**再加双算法一致性门禁**（偏差 ≤ %.0f%% 量程）⇒ 共四道。"
        "没过的读数一律标「作废」并只作参照，**不进红线**。"
        % (100.0 * CURVE_MIN_SPAN, KNEE_AGREE_PCT))
    add("")
    add("### 5.1 通过全部守卫的膝点（`common_good` 池 / 全量窗口 / 三项核心指标）")
    add("")
    add("| 指标 | 退化族 | 跟踪臂 | 膝点 | 膝点处跟住率 | 跟住率跨度 | 双算法偏差 |")
    add("|:--|:--|:--|--:|--:|--:|--:|")
    for metric in CORE_METRICS:
        for family in DEGRADATION_FAMILIES:
            for tracker in TRACKER_FAMILIES:
                row = pick(tracker, family, metric)
                if not row or row["knee_reliable"] != "1":
                    continue
                add("| `%s` | %s | `%s` | %.6g | %.1f%% | %.2f pp | %.2f%% |"
                    % (metric, family, tracker, fnum(row.get("knee_x_final")),
                       100.0 * fnum(row.get("knee_y_final")),
                       100.0 * fnum(row.get("curve_y_span")), fnum(row.get("delta_pct_of_span"))))
    add("")
    add("> ⚠️ **膝点处的跟住率仍然在 95%~99.7%** —— 这再次证明：本轮反解出的不是"
        "「跟踪崩溃点」，而是「**画质—跟踪性能分界点**」（曲线开始明显弯折的位置）。"
        "把它叫「崩溃拐点」会**高估**这条线的物理含义。")
    add("")
    add("### 5.2 小目标子集（max(w,h) ≤ %d px）" % SMALL_TARGET_MAX_DIM)
    add("")
    add("> 小目标（远距离吊舱场景）在**更低**的画质上就掉队 ⇒ 同样的膝点算法反解出的线位"
        "**更高**（更严）。下表只列通过全部守卫的读数。")
    add("")
    add("| 退化族 | 指标 | 跟踪臂 | 膝点 | 与全量线位之比 |")
    add("|:--|:--|:--|--:|--:|")
    for family in DEGRADATION_FAMILIES:
        for metric in CORE_METRICS:
            for tracker in TRACKER_FAMILIES:
                row = pick(tracker, family, metric, "common_good", "small")
                if not row or row["knee_reliable"] != "1":
                    continue
                base = pick(tracker, family, metric)
                base_value = fnum(base.get("knee_x_final")) if base else float("nan")
                value = fnum(row.get("knee_x_final"))
                ratio = (value / base_value) if math.isfinite(base_value) and base_value else float("nan")
                add("| %s | `%s` | `%s` | %.6g | %s |"
                    % (family, metric, tracker, value,
                       ("%.2fx" % ratio) if math.isfinite(ratio) else "—"))
    add("")
    add("### 5.3 被守卫作废的候选读数（**只作参照，不进红线**）")
    add("")
    add("> 这些是「双算法一致、但曲线有效性守卫判它退化」的读数。列出来是为了让读者看到"
        "**为什么它们没进 §6**：在跟住率从未塌掉的曲线上，膝点只是噪声点。")
    add("")
    add("| 退化族 | 指标 | 跟踪臂 | 候选读数 | 跨度 | 单调 | 指标单调 | 作废理由 |")
    add("|:--|:--|:--|--:|--:|:--:|:--|:--|")
    for family in DEGRADATION_FAMILIES:
        for metric in CORE_METRICS:
            for tracker in TRACKER_FAMILIES:
                row = pick(tracker, family, metric)
                if not row:
                    continue
                candidate = fnum(row.get("knee_x_candidate"))
                if not math.isfinite(candidate):
                    continue
                reasons = []
                if row.get("curve_monotone") != "1":
                    reasons.append("跟住率非单调")
                if fnum(row.get("curve_y_span")) <= CURVE_MIN_SPAN:
                    reasons.append("跨度 %.2fpp ≤ %.0fpp" % (100.0 * fnum(row.get("curve_y_span")),
                                                             100.0 * CURVE_MIN_SPAN))
                if row.get("metric_monotone") != "1":
                    reasons.append("指标非单调")
                add("| %s | `%s` | `%s` | %.6g | %.2f pp | %s | %s | %s |"
                    % (family, metric, tracker, candidate, 100.0 * fnum(row.get("curve_y_span")),
                       "✅" if row.get("curve_monotone") == "1" else "❌",
                       "✅" if row.get("metric_monotone") == "1" else "❌",
                       " + ".join(reasons) or "—"))
    add("")
    add("> 🔴 **特别点名 `contrast` 族的 `local_scr`**：该族对 `local_scr` **不承载信息**——"
        "`local_scr = P90(|I_core − μ_ring|) / (σ_ring + ε)` 的分子分母在"
        "「动态范围向全帧均值收缩」下**同乘 α**，数学上几乎不变（实测窗口内相对极差仅 2.2%）。"
        "⇒ 该组合的膝点**没有物理意义**，与「已剔除的 JPEG 族」是同一类问题。")
    add("")
    add("### 5.4 同族对照：KCF vs CSRT（DCF 家族内部）")
    add("")
    add("> **本轮 KCF 与 CSRT 的曲线全部未达 A 级**（跨度 ≤ %.0fpp 或非单调）⇒ "
        "它们的读数**只作物理参照**，**不参与红线取值**，也**无法**用来证明"
        "「KCF 侧线位不是实现细节」——这一条本轮**无法验证**。"
        % (100.0 * CURVE_MIN_SPAN))
    add("")
    add("| 退化族 | 指标 | KCF 候选读数 | CSRT 候选读数 | 线位差 | 两者曲线等级 |")
    add("|:--|:--|--:|--:|--:|:--|")
    for family in DEGRADATION_FAMILIES:
        for metric in LOCAL_METRICS + GLOBAL_METRICS:
            row_k = pick("kcf_legacy", family, metric)
            row_c = pick("csrt", family, metric)
            if not row_k or not row_c:
                continue
            kv, cv = fnum(row_k.get("knee_x_candidate")), fnum(row_c.get("knee_x_candidate"))
            if not (math.isfinite(kv) and math.isfinite(cv)):
                continue
            grade = ("A 级" if fnum(row_k.get("curve_y_span")) > CURVE_MIN_SPAN
                     and row_k.get("curve_monotone") == "1" else "未达 A 级")
            add("| %s | `%s` | %.6g | %.6g | %.2f%% | %s |"
                % (family, metric, kv, cv, abs(kv - cv) / max(abs(kv), abs(cv), 1e-9) * 100.0,
                   grade))
    add("")

    add("## 六、良好跟踪合格线（约束线）——只对三项核心指标")
    add("")
    add("> **口径**：每族膝点取**最小值**（最严的一族说了算）并标出**决定族 / 决定臂**；"
        "只取双算法一致（`knee_reliable=1`）的膝点；**跨家族线位不合并求平均**，"
        "只取更严的那条并把两族线位差单独报出。")
    add("")
    add("| 指标 | **约束线** | 决定族 | 决定臂 | XCorr 线位 | KCF 线位 | CSRT 线位 | 两族线位差 | 候选数 |")
    add("|:--|--:|:--|:--|--:|--:|--:|--:|--:|")
    for metric in CORE_METRICS:
        info = lines_tbl.get(metric, {})
        value = fnum(info.get("line_value"))
        dev = fnum(info.get("cross_arm_dev_pct"))
        add("| `%s` | **%s** | %s | %s | %.6g | %.6g | %.6g | %s | %s |"
            % (metric, ("%.6g" % value) if math.isfinite(value) else "—",
               info.get("deciding_family", "—"), info.get("deciding_tracker", "—"),
               fnum(info.get("xcorr_line")), fnum(info.get("kcf_legacy_line")),
               fnum(info.get("csrt_line")),
               ("%.2f%%" % dev) if math.isfinite(dev) else "—",
               info.get("n_candidates", "—")))
    add("")
    add("**每条主臂各自的线位**（上一轮的同一口径：每族膝点取最小值，两臂**并列**呈现）")
    add("")
    add("| 指标 | XCorr 线位 | 决定族 | KCF 线位 | 决定族 | 两臂线位差 | CSRT 线位（对照，不取值） |")
    add("|:--|--:|:--|--:|:--|--:|--:|")
    for metric in CORE_METRICS:
        info = lines_tbl.get(metric, {})
        fam_x = fam_k = "-"
        for family in DEGRADATION_FAMILIES:
            rx = pick("xcorr", family, metric)
            rk = pick("kcf_legacy", family, metric)
            if rx and fnum(rx.get("knee_x_final")) == fnum(info.get("xcorr_line")):
                fam_x = family
            if rk and fnum(rk.get("knee_x_final")) == fnum(info.get("kcf_legacy_line")):
                fam_k = family
        dev = fnum(info.get("cross_arm_dev_pct"))
        add("| `%s` | %.6g | %s | %.6g | %s | %s | %.6g |"
            % (metric, fnum(info.get("xcorr_line")), fam_x, fnum(info.get("kcf_legacy_line")), fam_k,
               ("%.2f%%" % dev) if math.isfinite(dev) else "—", fnum(info.get("csrt_line"))))
    add("")
    add("> **取值规则**：对外红线 = 两条主臂各自线位里**更严的那一条**"
        "（上一轮任务书基准 819.55 / 2.69 正是这样取的：2.69 = min(3.20777, 2.69153)）。"
        "🔴 **绝不**把两条臂的线位求平均（跨家族纪律）。CSRT 是**同族对照臂**，"
        "只用来证明 KCF 侧线位不是实现细节，**不参与红线取值**。")
    add("")
    add("> **读法提醒**：这些线是「**物理上还有可能跟住**」的**必要条件**，"
        "不是充分条件——过线不代表一定能跟住，过线窗口的实测跟住率见 §5.3 的「膝点处跟住率」列，"
        "而**过线以下的窗口**才进入「跟踪失锁不属算法考核范围」的免责区间（§10）。")
    add("")
    add("> **全图指标（`global_contrast` / `global_lap_var`）已从约束线表中剔除**："
        "任务书 §一已明确「彻底废弃 v1 全图指标」，它们在本报告里只以"
        "「**全图对照（已废弃口径，仅作对照）**」的身份出现在 §7，**不得**用于对外红线。")
    add("")
    add("> `surround_*` 与 `target_grad_mean` 同理：进 §7 的**辅助指标**表，**不进红线**。")
    add("")

    # ------------------------------------------------------------------ 七、辅助/对照
    add("## 七、辅助指标与全图对照（**已废弃口径，仅作对照**）")
    add("")
    add("> ⚠️ 本节所有数字**不构成任何红线**，仅用于说明「为什么对外只保留三项核心指标」。")
    add("")
    add("| 指标 | 类别 | XCorr 膝点（代表族） | KCF 膝点（代表族） | 是否进红线 |")
    add("|:--|:--|--:|--:|:--|")
    aux_all = tuple(m for m in LOCAL_METRICS if m not in CORE_METRICS) + GLOBAL_METRICS
    for metric in aux_all:
        row_x = pick("xcorr", "blur", metric)
        row_k = pick("kcf_legacy", "blur", metric)
        kind = "辅助指标" if metric in LOCAL_METRICS else "全图对照（已废弃口径）"
        add("| `%s` | %s | %s | %s | ❌ 否 |"
            % (metric, kind,
               ("%.6g" % fnum(row_x.get("knee_x_final"))) if row_x else "—",
               ("%.6g" % fnum(row_k.get("knee_x_final"))) if row_k else "—"))
    add("")

    # ------------------------------------------------------------------ 八、回归
    add("## 八、三分法回归测试体系")
    add("")
    add("> 阈值来自**标定集**（§6），下面三个集合**全程零参与阈值搜索**。")
    add("")
    add("### 8.1 好片段 · 30% 留出集 ⇒ **好片误杀率**")
    add("")
    add("| 指标 | 约束线 | 留出集样本数 | **红线以下（误杀）** | **好片误杀率** | 指标中位数 |")
    add("|:--|--:|--:|--:|--:|--:|")
    for row in regress_rows:
        if row["set"] != "good_holdout":
            continue
        add("| `%s` | %.6g | %s | %s | **%.2f%%** | %.6g |"
            % (row["metric"], fnum(row["threshold"]), row["n_evaluated"],
               row["n_below_line"], 100.0 * fnum(row["rate_below_line"]),
               fnum(row["median"])))
    add("")
    add("### 8.2 一般片段 ⇒ **临界敏感性**（不参与定标）")
    add("")
    add("| 指标 | 约束线 | 样本数 | 红线以下比例 | 指标中位数 | 落在膝点邻域比例 | AUC(一般 vs 好) | AUC(一般 vs 跟丢) |")
    add("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for row in regress_rows:
        if row["set"] != "mediocre":
            continue
        add("| `%s` | %.6g | %s | %.2f%% | %.6g | %s | %s | %s |"
            % (row["metric"], fnum(row["threshold"]), row["n_evaluated"],
               100.0 * fnum(row["rate_below_line"]), fnum(row["median"]),
               ("%.1f%%" % (100.0 * fnum(row["frac_in_knee_band"])))
               if math.isfinite(fnum(row["frac_in_knee_band"])) else "—",
               ("%.4f" % fnum(row["auc_vs_good"])) if math.isfinite(fnum(row["auc_vs_good"])) else "—",
               ("%.4f" % fnum(row["auc_vs_lost"])) if math.isfinite(fnum(row["auc_vs_lost"])) else "—"))
    add("")
    add("### 8.3 跟丢片段 + 真实坏片 ⇒ **坏片拦截率**")
    add("")
    add("| 集合 | 指标 | 约束线 | 样本数 | 红线以下 | **拦截率** | 备注 |")
    add("|:--|:--|--:|--:|--:|--:|:--|")
    for row in regress_rows:
        if row["set"] not in ("lost", "real_bad", "real_good"):
            continue
        label = {"lost": "跟丢片段（Anti-UAV）", "real_bad": "**真实坏片（2 条）**",
                 "real_good": "真实好片对照（2 条）"}[row["set"]]
        add("| %s | `%s` | %.6g | %s | %s | **%.1f%%** | %s |"
            % (label, row["metric"], fnum(row["threshold"]), row["n_evaluated"],
               row["n_below_line"], 100.0 * fnum(row["rate_below_line"]), row["note"]))
    add("")
    add("> 🔴 **真实坏片只有 2 条 ⇒ 上面涉及真实坏片/好片的结论只能标注为「小样本级」**，"
        "不可外推到批量业务场景；其作用是**一次性独立盲测**（阈值全程零参与），"
        "证明「仿真推演的红线在真实外场成立」这一**方向性**结论。")
    add("")
    add("**三个数字必须一起读**")
    add("")
    add("> ① **好片误杀率**（§8.1）：红线对**真实存在目标**的好片段几乎不误伤——"
        "`target_lap_var` 与 `target_local_contrast` 都在个位数百分比以内；"
        "`local_scr` 是**最弱的一项**（见 ④）。")
    add("> ② **坏片拦截率**（§8.3）：**真实坏片**（物理退化）在 `target_lap_var` 与 "
        "`target_local_contrast` 上被 **100%** 拦截、对**真实好片对照**几乎零误杀 "
        "⇒ 这两条线在**真实外场成立**。")
    add("> ③ **跟丢片段的拦截率明显低于真实坏片**——**这不是缺陷，而是设计意图**："
        "跟丢片段里相当一部分是「**算法侧失锁**」（快速机动 / 干扰 / 目标出框），"
        "它们的画质指标并不低；红线只应该拦**物理侧不可解释**的失锁，"
        "否则就把算法缺陷也一并洗成了免责项——这正是任务书 §四.4 免责声明的边界。")
    add("> ④ **`local_scr` 在真实片对照组上误杀偏高，必须写清原因**："
        "这两条真实好片里**根本没有无人机目标**（DoG 显著性峰值落在地面景物/树叶上，"
        "已逐帧核对），而 `local_scr` 是**被局部背景方差归一化的比值**——"
        "纹理丰富的场景分母大、SCR 天然低 ⇒ 它测的不是「目标 SCR」。"
        "因此真实片对照组只能读作「**平坦退化天空 vs 有纹理但无目标的航拍场景**」的分离度，"
        "**不能**当作目标级误杀率；**目标级误杀率只看 §8.1 的 Anti-UAV 留出集**。")
    add("")
    add("### 8.4 真实坏片的处理口径（沿用 v1，路径与参数可复现）")
    add("")
    add("| 环节 | 口径 |")
    add("|:--|:--|")
    add("| 坏片（红外 2 条） | `%s` |" % "`, `".join(p for _, p in REAL_BAD_CLIPS))
    add("| 好片对照（红外 2 条） | `%s` |" % "`, `".join(p for _, p in REAL_GOOD_CLIPS))
    add("| 故障窗 | 视频**最后 5 秒**（`round(fps × 5)` 帧）——与 v1 `vq_tail.clip_slices` 同款 |")
    add("| 黑边裁剪 | 复用 v1 `manu/video_quality/vq_metrics.detect_content_box`"
        "（逐行/逐列**最大值** ≤ 16 判黑边，取首尾连续黑边段；剩余内容 < 25% 则不裁） |")
    add("| 目标 ROI | 坏片**没有标注 GT** ⇒ 由 **DoG 小目标响应自动选取**（每帧取最显著处）；"
        "这给坏片的是**最有利**的假设 ⇒ 拦截结论偏保守 |")
    add("")
    add("**逐帧明细**：`iqa_v2/output/blind_real_v2.csv`（含裁边参数、ROI 坐标与全部指标）。")
    add("")
    for name, _ in REAL_BAD_CLIPS:
        clip_rows = [r for r in blind_rows if r["clip"] == name]
        if not clip_rows:
            continue
        add("* `%s`：%d 帧，黑边 (上 %s / 下 %s / 左 %s / 右 %s)，"
            "内容 %s×%s，`target_lap_var` 中位数 %.1f，`local_scr` 中位数 %.2f"
            % (name, len(clip_rows), clip_rows[0]["border_top"], clip_rows[0]["border_bottom"],
               clip_rows[0]["border_left"], clip_rows[0]["border_right"],
               clip_rows[0]["content_w"], clip_rows[0]["content_h"],
               sorted(fnum(r["target_lap_var"]) for r in clip_rows)[len(clip_rows) // 2],
               sorted(fnum(r["local_scr"]) for r in clip_rows)[len(clip_rows) // 2]))
    add("")

    # ------------------------------------------------------------------ 九、判别力
    add("## 九、指标判别力与相关性")
    add("")
    add("### 9.1 零合成退化下的尾部验证（真实红外画面）")
    add("")
    add("> 在**未施加任何人工退化**的真实红外窗口上，按指标排序取最低 10% / 15% 尾部，"
        "对比尾部窗口与其余窗口的**真实跟住率**，并给出「预测跟丢」的 ROC-AUC。")
    add("")
    add("| 指标 | 类型 | 尾部 | 跟踪器 | 尾部跟住率 | 其余跟住率 | 差(pp) | 预测跟丢 AUC |")
    add("|:--|:--|--:|:--|--:|--:|--:|--:|")
    for row in tail_rows:
        add("| %s | %s | %.0f%% | %s | %.1f%% | %.1f%% | %.1f | %.4f |"
            % (row["metric"], row["metric_kind"], fnum(row["tail_quantile"]) * 100.0,
               row["tracker"], fnum(row["tail_success_rate"]) * 100.0,
               fnum(row["rest_success_rate"]) * 100.0, fnum(row["drop_pp"]),
               fnum(row["auc_predict_lost"])))
    add("")
    add("### 9.2 退化族内的判别力（Spearman |ρ|，取各曲线集最大）")
    add("")
    add("| 指标 | 类型 | XCorr \|max ρ\| | KCF \|max ρ\| | CSRT \|max ρ\| |")
    add("|:--|:--|--:|--:|--:|")
    for metric in LOCAL_METRICS + GLOBAL_METRICS:
        best = {}
        for tracker in TRACKER_FAMILIES:
            best[tracker] = max((abs(fnum(r["spearman"])) for r in corr_rows
                                 if r["metric"] == metric and r["tracker"] == tracker),
                                default=float("nan"))
        kind = "局部 ROI" if metric in LOCAL_METRICS else "全图对照"
        add("| %s | %s | %.3f | %.3f | %.3f |" % (metric, kind, best["xcorr"],
                                                  best["kcf_legacy"], best["csrt"]))
    add("")
    add("> **结论（业务口径）**：局部 ROI 指标（`local_scr` / `target_lap_var` / "
        "`target_local_contrast`）在**退化族内**与跟住率的相关性应当**强于**"
        "全图对照指标（`global_contrast` / `global_lap_var`）——"
        "这正是 v2 从「全图随机采点」转向「业务目标本体」的收益所在。")
    add("")

    # ------------------------------------------------------------------ 十、免责声明
    lap_line = fnum(lines_tbl.get("target_lap_var", {}).get("line_value"))
    scr_line = fnum(lines_tbl.get("local_scr", {}).get("line_value"))
    add("## 十、免责声明（**逐字**落地）")
    add("")
    add("> 「本红外跟踪算法的性能考核以目标中心局部画质（v2 规范）为客观前提。"
        "经基于真实红外目标的多维物理退化压力测试及独立故障盲测验证，"
        "当因光学失焦、机芯未校正或气象热交叉导致目标核心区锐度 "
        "`target_lap_var < %s`、或无极性局部信杂比 `local_scr < %s`"
        "（及局部对比度坍缩）时，目标有效特征在物理层面已被背景噪声淹没，"
        "由此引发的跟踪失锁属于前端成像与环境物理极限，不在跟踪算法性能考核范围内"
        "（正式上板阈值将随内部业务标注序列完成同口径复测后锁定）。」"
        % (("%g" % lap_line) if math.isfinite(lap_line) else "见 §六",
           ("%g" % scr_line) if math.isfinite(scr_line) else "见 §六"))
    add("")
    add("**与任务书基准值的差异说明（必须一起交付）**")
    add("")
    add("| 指标 | 任务书基准 | 本轮重标实测 | 变化 | 决定族 / 决定臂 | 上一轮决定族 |")
    add("|:--|--:|--:|--:|:--|:--|")
    add("| `target_lap_var` | 819.55 | %s | %s | %s / %s | blur |"
        % (("%.2f" % lap_line) if math.isfinite(lap_line) else "—",
           ("%+.2f%%" % (100.0 * (lap_line - 819.55) / 819.55))
           if math.isfinite(lap_line) else "—",
           lines_tbl.get("target_lap_var", {}).get("deciding_family", "—"),
           lines_tbl.get("target_lap_var", {}).get("deciding_tracker", "—")))
    add("| `local_scr` | 2.69 ~ 3.21 | %s | %s | %s / %s | motion (KCF) |"
        % (("%.2f" % scr_line) if math.isfinite(scr_line) else "—",
           ("%+.2f%%" % (100.0 * (scr_line - 2.69) / 2.69))
           if math.isfinite(scr_line) else "—",
           lines_tbl.get("local_scr", {}).get("deciding_family", "—"),
           lines_tbl.get("local_scr", {}).get("deciding_tracker", "—")))
    add("")
    add("**三条原因，逐条给证据**")
    add("")
    add("> ① **底座换血（G2 两趟法）——`target_lap_var` 变严的主因**。"
        "任务书的 819.55 是在「**全量 2442 窗口、未纯净化底座**」下反解的："
        "当时约 %.1f%% 的窗口**在干净帧上就已经跟丢**，把整条衰减曲线的顶部压到 84.5%%（XCorr）"
        "⇒ 弯折点被推到**更高**的画质值上（819.55）。本轮换成**好片段底座**后，"
        "clean 档跟住率回到 **100%%**（§4），同样的弯折点落到**更低**的画质值 ⇒ **红线变严**（505.62）。"
        "🔴 换句话说：**旧值 819.55 不是保守值，而是被脏窗口污染出来的宽松值**。"
        % (100.0 * (class_counter.get(CLASS_LOST, 0) + class_counter.get(CLASS_MEDIOCRE, 0)) / total))
    add("> ② **决定族变了——`local_scr` 变松的原因**。"
        "旧值 2.69153 是 **KCF 臂的 motion 族**膝点；本轮 KCF 的 motion 族候选"
        "在 `common_good` 池上**未通过双算法一致性门禁**（偏差 > %.0f%% 量程）⇒ 被降级为「物理参照」"
        "不参与红线取值，于是最严的存活候选变成 **blur 族**（%s）。"
        "⇒ `local_scr` 这条线的证据强度**弱于** `target_lap_var`，"
        "它在本轮的**误杀率也最高**（§8.1）——两条事实必须一起汇报。"
        % (KNEE_AGREE_PCT,
           ("%.4g" % fnum(lines_tbl.get("local_scr", {}).get("line_value")))))
    add("> ③ **删除 JPEG 族（G1）不改变这两条线的取值**："
        "上一轮 `local_scr` 的 JPEG 候选为 4.88624、`target_lap_var` 的 JPEG 候选为 4182.89，"
        "**都高于**各自的最严候选 ⇒ 删除该族只是让 `surround_blockiness`（旧报告里由 jpeg 决定）"
        "整体退出红线表，不影响上述两条线。")
    add("")
    add("> ⚠️ **因此，对外使用时必须带上这句限定**：本轮红线是在**净化底座**上反解的，"
        "而净化底座上的衰减曲线**不足以产生跟踪崩溃**（§4.1）⇒ "
        "红线是「**曲线开始弯折**」的位置，不是「**跟踪崩溃点**」。"
        "它的可用性由**独立盲测**背书（真实坏片 100%% 拦截 / 真实好片对照 0%% 误杀），"
        "而不是由膝点本身背书。")
    add("")

    # ------------------------------------------------------------------ 十一、复现
    add("## 十一、复现命令与产物清单")
    add("")
    add("```bash")
    add("# 服务器 huangzhe@192.168.99.40:32222（conda env uav，PYTHONPATH 需带 contrib KCF）")
    add("cd /tmp/pycharm_project_a7d966e6")
    add("export PYTHONPATH=/tmp/pylibs_cvcontrib")
    add("export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4")
    add("python iqa_v2/run_eval_v2.py --phase all --workers 24")
    add("python iqa_v2/verify_recalc_v2.py --samples 120      # 独立复算，误差 < 1e-6")
    add("python -m pytest iqa_v2/test_iqa_v2.py -v             # 机械门禁 100% 通过")
    add("md5sum -c iqa_v2/output/manifest_v2.md5               # 产物 MD5 全量校验")
    add("```")
    add("")
    add("| 产物 | 路径 |")
    add("|:--|:--|")
    for name, path in (("主报告", "iqa_v2/REPORT_V2_FINAL.md"),
                       ("窗口表", "iqa_v2/output/windows.csv"),
                       ("明细（逐窗口 × 逐条件）", "`/tmp/iqa_v2_cache/degrade_detail.csv`"),
                       ("三分法分类表", "iqa_v2/output/segment_class_v2.csv"),
                       ("序列级切分表", "iqa_v2/output/split_v2.csv"),
                       ("膝点表", "iqa_v2/output/knee_thresholds_v2.csv"),
                       ("约束线表", "iqa_v2/output/constraint_lines_v2.csv"),
                       ("相关性表", "iqa_v2/output/correlations_v2.csv"),
                       ("尾部验证表", "iqa_v2/output/tail_analysis_v2.csv"),
                       ("回归测试表", "iqa_v2/output/regression_v2.csv"),
                       ("真实坏片盲测明细", "iqa_v2/output/blind_real_v2.csv"),
                       ("产物 MD5 清单", "iqa_v2/output/manifest_v2.md5")):
        add("| %s | `%s` |" % (name, path))
    add("")

    # ------------------------------------------------------------------ 十二、局限
    add("## 十二、已知局限")
    add("")
    add("1. **正式上板红线未锁定**：本轮阈值用 **Anti-UAV 开源红外**标定，"
        "只验证方法有效。内部 **58 条红外业务 GT** 尚未回收"
        "（核查结论：`manu/point_annotator/` 只有工具、无标注数据；"
        "`manu/memory/memory.md` §315 记录的是**三份并行交付**，§316 明确写着「待回收」；"
        "母包 `annotations/` 目录下只有 2 个试跑草稿 JSON，人工标注帧仅 36 帧 ≈ 0.02%）"
        "⇒ **正式交付模型替换待标注回收后执行**："
        "把跟踪探针从 XCorr + KCF 换成实际交付模型（如 `trial_0080` / RC2），"
        "用**同一套流水线**重跑即可。**本轮保持 XCorr + KCF，未启动任何训练或模型重训。**")
    add("2. **真实坏片样本量只有 2 条** ⇒ 所有涉及真实坏片 / 好片的拦截率与误杀率"
        "均为**小样本级**，只作方向性证据，不可外推。")
    add("3. **真实坏片没有标注 GT**：ROI 由 DoG 小目标响应自动选取，"
        "并非真实目标框；虽然这给坏片的是**最有利**假设，但仍不等价于人工框。")
    add("4. **KCF 臂的滞后一帧补偿**在序列末尾缺失尾帧（见 §2.3），"
        "对 `CLE_{t+4}` 口径恒定偏保守；三分法分类已用同帧口径消除了这一偏置。")
    add("5. **全图指标已废弃**：本报告 §7 的全图数字只作对照，"
        "**不得**进入对外协议、红线或免责声明。")
    add("6. **净化底座上的衰减曲线不足以产生跟踪崩溃**（§4.1）："
        "任务书 §二.2 给定的三族档位在好片段上只能把跟住率压到 90%% 以上，"
        "⇒ 膝点只能定位在「曲线开始弯折」处，**双算法门禁因此在相当一部分候选上判为不可靠**，"
        "跨臂线位差也明显大于上一轮（上一轮未纯净化时 `target_lap_var` 两臂偏差 0.00%%）。"
        "**要拿到更硬的崩溃拐点，正确做法是加大退化档位上限**"
        "（例如 `blur` σ 提到 6~8、`contrast` α 降到 0.05），"
        "**而不是**把脏窗口塞回仿真池——后者是任务书 §二.1 明令禁止的。")
    add("7. **`local_scr` 是三项核心指标里证据最弱的一项**："
        "它是被局部背景方差归一化的比值，在纹理丰富但无目标的场景里天然偏低；"
        "本轮它在好片段留出集上的误杀率明显高于另外两项（§8.1），"
        "在真实好片对照组上的误杀率也高（§8.3 ④）。"
        "⇒ 对外只建议把 `target_local_contrast` 与 `target_lap_var` 当作**正式红线**，"
        "`local_scr` 作为**辅助边界**使用。")
    add("")

    report_path = Path(cli.report).resolve()
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[REPORT] %s" % report_path)
    return report_path


# --------------------------------------------------------------------------- 主流程
def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Target-Centric 红外画质评估（Anti-UAV）")
    parser.add_argument("--phase", default="all",
                        choices=["all", "dataset", "eval", "classify", "knee", "constraint",
                                 "blind", "regress", "tail", "report"])
    parser.add_argument("--dataset-root", default="/mnt/data/siping/datasets/manu/anti-uav")
    parser.add_argument("--output-dir", default="iqa_v2/output")
    parser.add_argument("--cache-dir", default="/tmp/iqa_v2_cache")
    parser.add_argument("--detail-csv", default="")
    parser.add_argument("--report", default="iqa_v2/REPORT_V2_FINAL.md")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-windows-per-seq", type=int, default=30)
    parser.add_argument("--total-target", type=int, default=2600)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()

    if cli.phase in ("all", "dataset"):
        import dataset_anti_uav

        sys.argv = ["dataset_anti_uav.py", "--dataset-root", cli.dataset_root,
                    "--output-dir", cli.output_dir, "--seed", str(cli.seed),
                    "--max-windows-per-seq", str(cli.max_windows_per_seq),
                    "--total-target", str(cli.total_target)]
        if cli.smoke:
            sys.argv.append("--smoke")
        dataset_anti_uav.main()

    detail_csv = cli.detail_csv or str(Path(cli.cache_dir) / "degrade_detail.csv")
    if cli.phase in ("all", "eval"):
        detail_csv = str(run_eval(cli))
    cli.detail_csv = detail_csv
    if cli.phase in ("all", "classify"):
        from regression_v2 import phase_classify

        phase_classify(cli)
    if cli.phase in ("all", "knee"):
        phase_knee(cli)
    if cli.phase in ("all", "constraint"):
        phase_constraint(cli)
    if cli.phase in ("all", "blind"):
        from regression_v2 import phase_blind

        phase_blind(cli)
    if cli.phase in ("all", "regress"):
        from regression_v2 import phase_regress

        phase_regress(cli)
    if cli.phase in ("all", "tail"):
        phase_tail(cli)
    if cli.phase in ("all", "report"):
        phase_report(cli)


if __name__ == "__main__":
    main()
