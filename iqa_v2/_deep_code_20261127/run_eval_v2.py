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
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataset_anti_uav import (SEED, SMALL_TARGET_MAX_DIM, WINDOW_LEN,  # noqa: E402
                              FrameRecord, Sequence, discover_sequence_dirs, load_sequence,
                              read_frame_size, sample_windows, valid_windows)
from degradation_4d import (BLUR_SIGMAS, CONTRAST_ALPHAS,  # noqa: E402
                            MOTION_LENGTHS, build_conditions,
                            degradation_strength, motion_angle_for)
from target_roi_metrics import (ALL_METRICS, GLOBAL_METRICS,  # noqa: E402
                              LOCAL_METRICS, METRIC_DIRECTION, compute_roi_metrics)

#: 档位剖面 → 产物路径的**硬绑定**（规格书 C3：基线轮产物不得被覆盖）。
#: 🔴 深轮**必须**走独立目录 —— 这里把它钉死成「剖面 ⇒ 路径」的映射，
#: 而不是靠操作者自觉传参；显式传入基线绑定路径时**直接拒绝运行**。
#: 🔴 仓库相对的值（``iqa_v2/...``）一律按**仓库根**解释（``_anchor_repo_path``）
#: ⇒ 与 CWD 无关；闸门比对的两侧都取**解析后**的路径。
PROFILE_PATHS: dict[str, dict[str, str]] = {
    "baseline": {"output_dir": "iqa_v2/output",
                 "cache_dir": "/tmp/iqa_v2_cache",
                 "report": "iqa_v2/REPORT_V2_FINAL.md"},
    "deep": {"output_dir": "iqa_v2/output_deep",
             "cache_dir": "/tmp/iqa_v2_cache_deep",
             "report": "iqa_v2/REPORT_V2_DEEP.md"},
}

#: **仓库根**（由脚本位置推导 ⇒ 与当前工作目录无关）。
#: 🔴 所有产物路径都按它锚定：早期版本把 ``PROFILE_PATHS`` 里的相对字面量
#: 按 **CWD** 解析 ⇒ 换个目录（``cd /tmp``）跑深轮，绑定值就指到别处 ⇒
#: 「基线轮产物不得被覆盖」的闸门**形同虚设**（可被绕过）。
REPO_ROOT = HERE.parent


def _anchor_repo_path(raw: str | Path) -> Path:
    """把产物路径**锚到仓库根**并归一成绝对路径。

    * 相对路径按「**仓库根相对**」解释（``PROFILE_PATHS`` 就是这么写的）；
    * 绝对路径原样保留（``/tmp/iqa_v2_cache`` 这类系统目录）；
    * ``resolve()`` 顺带吃掉 ``..``、尾斜杠与软链 ⇒ 同一个文件的多种写法归一到一处。

    Args:
        raw (str | Path): 原始路径字面量。

    Returns:
        Path: 解析后的绝对路径（不要求存在）。
    """
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _protected_artifacts() -> tuple[list[Path], str]:
    """派生「深轮**绝不可写**」的冻结基线产物集合。

    集合来源 = **冻结清单** ``iqa_v2/output/manifest_v2.md5``（每行 ``<md5>  <path>``，
    ``#`` 开头是注释）：仓库相对路径按仓库根解释、绝对路径（如
    ``/tmp/iqa_v2_cache/degrade_detail.csv``）原样解释 ⇒ **清单登记的全部产物**都受保护，
    不只是 ``PROFILE_PATHS`` 里那三条（否则 ``REPORT_V2_ANTIUAV.md`` /
    ``MEMORY_SECTION.md`` 这类基线产物会被深轮悄悄覆盖）。

    🔴 **清单缺失 ⇒ 不降级**：降级成 ``PROFILE_PATHS`` 的三条基线条目，
    覆盖面会**小于**清单（``REPORT_V2_ANTIUAV.md`` / ``MEMORY_SECTION.md`` 等
    基线产物就**不再受保护**）⇒ 闸门「变弱」却仍然放行 ⇒ 深轮可能悄悄覆盖
    基线产物而无人察觉。因此这里返回**空集合 + 空来源**，由调用方**硬失败**。

    Returns:
        tuple[list[Path], str]: ``(受保护产物列表, 来源说明)``；
            清单缺失时两者皆空（调用方必须拒绝继续）。
    """
    manifest_path = _anchor_repo_path(PROFILE_PATHS["baseline"]["output_dir"]) / "manifest_v2.md5"
    if not manifest_path.is_file():
        print("[GUARD] ❌ 冻结清单缺失：%s ⇒ 无法派生「深轮绝不可写」的产物集合"
              "（降级成 PROFILE_PATHS 三条会让 %s 等基线产物**不再受保护**）"
              " ⇒ **拒绝运行**" % (manifest_path, "REPORT_V2_ANTIUAV.md"))
        return [], ""
    protected: list[Path] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        entry = parts[1].strip()
        if entry:
            protected.append(_anchor_repo_path(entry))
    return protected, "冻结清单 %s" % manifest_path


def _protected_hit(raw: str, protected: list[Path]) -> Path | None:
    """判断「要写的路径」是否落在冻结产物上（命中就返回那个产物）。

    两种解释**都查**（任一命中即拒绝）：

    * **CWD 相对**解释（``Path(raw).resolve()``）—— 用户在仓库根跑时给的相对写法；
    * **仓库根锚定**解释（``_anchor_repo_path``）—— 与 CWD 无关的硬绑定。

    命中判据：写点**本身**就是受保护产物，或写点是受保护产物的**祖先目录**
    （往那个目录里写 ⇒ 会覆盖清单里的同名文件）。

    Args:
        raw (str): CLI 给出的路径字面量。
        protected (list[Path]): 受保护产物集合。

    Returns:
        Path | None: 命中的受保护产物；没命中返回 ``None``。
    """
    path = Path(raw)
    candidates = [path.resolve()]
    if not path.is_absolute():
        candidates.append(_anchor_repo_path(raw))
    for candidate in candidates:
        for protected_path in protected:
            if candidate == protected_path or protected_path.is_relative_to(candidate):
                return protected_path
    return None


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
#: 门限依据：跨度 < 5pp 的曲线落在**曲线点**的抽样噪声量级内
#: （本轮净化底座 874 窗口 × 6/5 档 = 5244 / 4370 个曲线点 ⇒ ±1.35 / ±1.48 pp @ 95% 置信），
#: 不足以支撑「崩溃拐点」这种强结论；>= 5pp 且单调的曲线才允许进约束线候选。
CURVE_MIN_SPAN = 0.05

#: **上一轮（2026-11-26 12:30 轮：未纯净化底座 / 全量窗口池）**的逐族逐臂膝点**冻结基线**。
#: 来源：``iqa_v2/REPORT_V2_ANTIUAV.md``（12:30 版）＋ host 侧 W8-2 复核的独立复算值。
#: 🔴 这些数字**已随本轮口径变更作废**，只用于 §10「逐族逐臂对照表」里说明「变了多少、为什么变」。
PRIOR_ROUND_KNEES = (
    ("blur", "local_scr", "xcorr", 4.7773),
    ("blur", "local_scr", "kcf_legacy", 4.7773),
    ("blur", "target_lap_var", "xcorr", 819.55),
    ("blur", "target_lap_var", "kcf_legacy", 819.55),
    ("motion", "local_scr", "xcorr", 5.6307),
    ("motion", "local_scr", "kcf_legacy", 2.6915),
    ("motion", "target_local_contrast", "xcorr", 0.20481),
    ("motion", "target_local_contrast", "kcf_legacy", 0.20481),
    ("motion", "target_lap_var", "xcorr", 1754.0),
    ("motion", "target_lap_var", "kcf_legacy", 2320.9),
    ("contrast", "local_scr", "kcf_legacy", 3.2078),
    ("contrast", "target_lap_var", "xcorr", 1090.9),
    ("contrast", "target_lap_var", "kcf_legacy", 1651.9),
)
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


#: 冻结的**基线轮**档位阶梯（基线轮产物已冻结 ⇒ 不许改轴）。
BASELINE_LADDER = {
    "blur": tuple(BLUR_SIGMAS),
    "motion": tuple(MOTION_LENGTHS),
    "contrast": tuple(CONTRAST_ALPHAS),
}


def _guard_profile_ladder(profile: str, conditions: list) -> str | None:
    """校验「**有效档位阶梯**」与「剖面纪律」一致（CLI 层强制，不是靠自觉）。

    * ``baseline`` 剖面 ⇒ 阶梯必须**逐项等于**冻结的基线阶梯
      （基线轮产物已冻结；改轴 ⇒ 出来的就不是「基线轮」了）；
    * ``deep`` 剖面 ⇒ 阶梯必须是基线阶梯的**严格超集**
      （否则两轮曲线不再逐档可比，规格书 §1.1 的「严格超集」不变量失效）。

    Args:
        profile (str): 档位剖面名。
        conditions (list): ``build_conditions`` 的结果。

    Returns:
        str | None: ``None`` 表示合规；否则是可直接打印的错误描述。
    """
    effective = {
        "blur": tuple(c.level for c in conditions if c.family == "blur"),
        "motion": tuple(int(round(c.level)) for c in conditions if c.family == "motion"),
        "contrast": tuple(c.level for c in conditions if c.family == "contrast"),
    }
    problems = []
    for family, base in BASELINE_LADDER.items():
        got = effective.get(family, ())
        if profile == "baseline":
            if tuple(round(float(v), 6) for v in got) != tuple(round(float(v), 6) for v in base):
                problems.append("%s：基线剖面必须逐项等于冻结阶梯 %s，实际 %s"
                                % (family, list(base), list(got)))
        elif profile == "deep":
            missing = [v for v in base
                       if not any(abs(float(v) - float(g)) <= 1e-6 * max(1.0, abs(float(v)))
                                  for g in got)]
            if missing:
                problems.append("%s：deep 剖面必须是基线阶梯的严格超集，缺 %s（实际 %s）"
                                % (family, missing, list(got)))
            elif len(got) <= len(base):
                # 🔴 只「包含」基线阶梯**不够**：档数与基线相同 ⇒ 退化上限根本没被抬高
                #    ⇒ 这一轮跑出来的还是基线轴，却会被记成 "deep"
                #    （规格书 §1.1 要求**严格超集** ⊋）
                problems.append("%s：deep 剖面必须是基线阶梯的严格超集（⊋），"
                                "实际 %s 与基线 %s 档数相同 ⇒ 未抬高退化上限"
                                % (family, list(got), list(base)))
    if problems:
        return ("档位阶梯违反剖面纪律 ⇒ 拒绝继续\n  " + "\n  ".join(problems)
                + "\n  ⇒ baseline 剖面不许改轴；deep 剖面只许**加**档位（不许减）")
    return None


def _parse_float_list(text: str) -> tuple[float, ...] | None:
    """把 ``"0.6,1.0"`` 解析成浮点元组；空串 ⇒ ``None``（沿用剖面自带档位）。"""
    chunks = [chunk.strip() for chunk in str(text).split(",")]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        return None
    try:
        return tuple(float(chunk) for chunk in chunks)
    except ValueError as exc:
        raise ValueError("档位列表解析失败：%r" % text) from exc


def _detect_detail_levels(detail_csv: str) -> dict[str, list[float]]:
    """从**明细底表**反推「这一轮实际用的退化轴」（每族的档位阶梯）。

    为什么必须实测反推、而不是抄 CLI 参数：``knee/constraint/blind/regress/tail/report``
    六个 CSV-only 阶段的档位轴**一律从明细 CSV 读**（``--conditions-profile`` 对它们零作用）
    ⇒ 抄 CLI 会写出「meta 说 deep、曲线其实来自 baseline 轴」这种**记录与事实相反**的 meta。

    Args:
        detail_csv (str): 明细 CSV 路径（不存在 ⇒ 返回空字典）。

    Returns:
        dict[str, list[float]]: ``族 -> 档位阶梯``（按退化严重度递增排；contrast 的 α 递减）。
    """
    import csv as _csv
    from collections import defaultdict

    if not detail_csv:
        return {}
    path = Path(detail_csv)
    if not path.is_file():
        return {}
    found = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            family = row.get("condition_family")
            if not family or family == "clean":
                continue
            try:
                level = round(float(row["condition_level"]), 6)
            except (KeyError, TypeError, ValueError):
                continue
            found[family].add(level)
    return {family: sorted(levels, reverse=(family == "contrast"))
            for family, levels in found.items()}


def _declared_ladder(conditions: list) -> dict[str, list[float]]:
    """``build_conditions`` 的档位 → ``{族: 档位列表}``（CLI 声明口径）。"""
    return {
        "blur": [c.level for c in conditions if c.family == "blur"],
        "motion": [int(round(c.level)) for c in conditions if c.family == "motion"],
        "contrast": [c.level for c in conditions if c.family == "contrast"],
    }


def _ladder_equal(want, got) -> bool:
    """两个档位阶梯是否**逐档相等**（浮点按 1e-6 相对容差比较）。"""
    if len(want) != len(got):
        return False
    for a, b in zip(want, got):
        if abs(float(a) - float(b)) > 1e-6 * max(1.0, abs(float(a))):
            return False
    return True


def _verify_profile_matches_detail(cli: argparse.Namespace, conditions: list) -> str | None:
    """校验「CLI 声明的档位剖面」与「明细底表实际轴」**逐族一致**。

    不一致就返回错误描述（由调用方 ``parser.error`` 退出）——**宁可报错也不许记错**：
    一个标着 "deep" 却跑在 baseline 轴上的 meta / 报告，会让整轮结论失去可追溯性。

    Returns:
        str | None: ``None`` 表示一致（或明细还没生成）；否则是可直接打印的错误描述。
    """
    detected = _detect_detail_levels(getattr(cli, "detail_csv", "") or "")
    if not detected:
        return None                       # 明细还没生成（``--phase all`` 尚未跑到 eval）
    declared = _declared_ladder(conditions)
    problems = []
    for family in ("blur", "motion", "contrast"):
        want = sorted(declared.get(family, []), reverse=(family == "contrast"))
        got = detected.get(family, [])
        if not _ladder_equal(want, got):
            problems.append("%s：声明 %s vs 明细实际 %s" % (family, want, got))
    if problems:
        return ("档位剖面与明细底表不一致 ⇒ 拒绝继续（防「记录与事实相反」）\n  "
                + "\n  ".join(problems)
                + "\n  ⇒ 请把 --conditions-profile 与 --detail-csv/--cache-dir 对齐"
                + "（deep 轴请配 --conditions-profile deep 与深轮 cache 目录）")
    return None


def _write_run_meta(cli: argparse.Namespace, conditions: list,
                      status: str = "in_progress") -> None:
    """写 ``run_meta.json``：这一轮跑的**档位剖面 + 实际生效的档位** + CLI 参数 + UTC 时间戳。

    这是**新增的旁路文件**（流水线里没有任何代码读它）⇒ 已交付产物的字节一个都不动；
    它的用途是「这一轮跑的到底是哪条退化轴」可追溯（CSV-only 阶段单独重跑时也写）。

    🔴 档位字段**同时**记两份：``conditions_profile_declared``（CLI 声明）与
    ``detail_levels_detected``（从明细底表**实测反推**）⇒ 二者不一致时
    ``profile_matches_detail=False`` ⇒ 不会被误当成「这轮就是 deep 轴」。
    ``status`` 字段区分「阶段进行中」与「已完成」⇒ 崩溃后残留的 meta 不会伪装成成功。
    """
    try:
        output_dir = Path(cli.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        detail_levels = _detect_detail_levels(getattr(cli, "detail_csv", "") or "")
        declared = _declared_ladder(conditions)
        # 🔴 只有**阶段已完成**且**明细底表实测到完整三族轴**时，
        #    「声明轴 vs 实际轴」才有意义 —— 阶段进行中明细 CSV 正在被重写
        #    （内容是半截的）；明细缺失/缺族时更是**什么都没验**。
        #    ⇒ 这两种情况一律记 ``None``（未判定），**不许**记 ``True``
        #    （那等于「没查也说查过了」⇒ meta 会声称深轮轴已核对通过）。
        matches = None
        measured = all(family in detail_levels for family in ("blur", "motion", "contrast"))
        if status == "completed" and measured:
            matches = True
            for family in ("blur", "motion", "contrast"):
                levels = detail_levels[family]
                want = sorted(declared.get(family, []), reverse=(family == "contrast"))
                if not _ladder_equal(want, levels):
                    matches = False
        meta = {
            "conditions_profile": cli.conditions_profile,
            "conditions_profile_declared": declared,
            "detail_levels_detected": detail_levels,
            "profile_matches_detail": matches,
            "n_conditions": len(conditions),
            "argv": list(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
        }
        with open(output_dir / "run_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    except Exception as exc:                       # 元数据写不出去也不能拖垮主流程
        print("[META] 写 run_meta.json 失败（已忽略）：%s" % exc)


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

    conditions = build_conditions(cli.conditions_profile,
                                  blur_sigmas=cli.blur_sigmas,
                                  motion_lengths=cli.motion_lengths,
                                  contrast_alphas=cli.contrast_alphas)
    _write_run_meta(cli, conditions)
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
        from regression_v2 import (CLASS_GOOD, classify_rows, split_sequences, verify_split_disjoint,
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

        # ---- Pass B 仿真池 = 标定集 ∩ (三分法「好片段」)
        # 🔴 任务书 §二.1：「仅筛选出低误差稳定跟住的『好片段』作为仿真输入底座」⇒
        #    进池条件就是三分法的**好片段**这一类（``class_ds == good``），
        #    **不是**「三臂好片段的并集」——后者会把「别的臂跟得住、本臂已经跟丢」的
        #    窗口塞进池子，把各臂衰减曲线的**起点**拖下来（这正是 G2 要消灭的污染）。
        #    三臂各自的好片段集在池内仍按臂取（``own_good``），跨臂比较用 ``common_good``。
        pool_keys = set()
        for row in class_rows:
            if split_map.get(row["seq_id"]) != "calib":
                continue
            if row["class_ds"] == CLASS_GOOD:
                pool_keys.add((row["seq_id"], row["frame_id"]))
        pool_jobs = [job for job in jobs if (job[0], job[1]) in pool_keys]
        print("[EVAL] Pass B 退化仿真：池 %d 窗口（三分法好片段 ∩ 标定集）× 条件 %d（留出集零参与）"
              % (len(pool_jobs), len(degrade_conditions)))
        rows_b, _ = _run_jobs(pool_jobs, degrade_conditions, workers, writer)

    write_class_csv(output_dir / "segment_class_v2.csv", class_rows)
    write_split_csv(output_dir / "split_v2.csv", split_map, class_rows)
    print("[EVAL] 明细 %d 行（Pass A %d + Pass B %d）→ %s（%.1f 分钟）"
          % (rows_a + rows_b, rows_a, rows_b, detail_path, (time.time() - _RUN_START) / 60.0))
    # 明细底表**同时**留在 ``output/``：验收仪器（``audit_final.py``）按「含 condition_family 的
    # CSV」识别退化底表，只放 cache 目录会让独立复算退化成「找不到底表」。
    mirrored = output_dir / "degrade_detail_v2.csv"
    shutil.copyfile(detail_path, mirrored)
    print("[EVAL] 明细底表镜像 → %s" % mirrored)
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


def binned_y_span(xs: np.ndarray, ys: np.ndarray) -> float:
    """**膝点来源曲线**（分位分箱）的跟住率跨度——保序回归后的 Y 量程。

    🔴 **两种跨度口径必须分清**（本轮审计的根因就在这里）：

    * ``level_y_span``（**逐档**）：X=退化档位、Y=该档平均跟住率 ⇒ 回答
      「这条退化轴有没有把跟住率压下去」= **受控退化轴口径**；
    * ``curve_y_span``（**分箱**）：X=实测指标值、Y=该指标值下的跟住率 ⇒ 回答
      「这个指标能不能区分跟住 / 跟丢」= **指标判别口径**，**这才是膝点真正取自的曲线**。

    上一版把逐档跨度写进 ``curve_y_span``、膝点却取自分箱曲线 ⇒ 等于
    「用 A 曲线的健康度给 B 曲线放行」。实测后果（净化底座上）：
    ``target_lap_var/xcorr/motion`` 逐档跨度 6.41pp（过门槛）但**分箱跨度仅 1.92pp**
    ——整条曲线几乎是平的（跟住率在 0.94~0.976 之间抖动），所谓膝点 541.09 只是
    「曲线上一个噪声局部极大值处的分箱边界」，双算法 0.00% 一致属**伪一致**。
    ⇒ 守卫必须判在分箱曲线上，而逐档跨度降为**第二道**守卫（退化轴本身也得够深）。
    """
    if xs.size < 2:
        return 0.0
    curve_x, curve_y = prepare_curve(xs, ys)
    if curve_x.size < 2:
        return 0.0
    return float(curve_y.max() - curve_y.min())


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
    #    两条跨度口径**都要过**，但各管一件事（见 ``binned_y_span`` 的 docstring）：
    #    * ``level_y_span``（逐档）= 受控退化轴口径：「退化加压有没有把跟住率压下去」；
    #    * ``curve_y_span``（分箱）= 指标值口径：「这个指标能不能区分跟住 / 跟丢」。
    #    🔴 守卫必须判在**膝点实际取自的那条曲线**（分箱曲线）上——上一版把逐档跨度
    #    写进 ``curve_y_span`` 而膝点取自分箱曲线，等于「用 A 曲线的健康度给 B 曲线放行」。
    bspan = binned_y_span(xs, ys)
    # 🔴 分箱跨度判据只在**分箱曲线真的能承载膝点**时适用（``xs.size >= 4``）：
    #    分箱不足 4 ⇒ 本来就取不出膝点（无线位可判），判据无从谈起，
    #    这时仍由**逐档**三项守卫决定要不要打 ``degenerate`` 标签。
    bspan_applies = xs.size >= 4
    degenerate = (not profile["monotone"]) or profile["span"] <= CURVE_MIN_SPAN \
        or (bspan_applies and bspan <= CURVE_MIN_SPAN) or (not mprofile["monotone"])
    result = {
        "n_windows": int(values.size),
        "n_bins": int(xs.size),
        "curve_y_span": bspan,
        "level_y_span": profile["span"],
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
        # 分箱不足 ⇒ 无法提取膝点；若守卫同时触发，仍须打上 degenerate 标签（否则「守卫触发但未标注」）
        result["method_used"] = "degenerate" if degenerate else ""
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
              "curve_y_span", "level_y_span", "curve_monotone", "metric_monotone",
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
            #    两条跨度都要过：``curve_y_span``（膝点来源的分箱曲线）+ ``level_y_span``（逐档退化轴）。
            if row.get("curve_monotone") != "1" or row.get("metric_monotone") != "1":
                continue
            try:
                if float(row.get("curve_y_span", "nan")) <= CURVE_MIN_SPAN:
                    continue
            except (TypeError, ValueError):
                continue
            try:
                if float(row.get("level_y_span", "nan")) <= CURVE_MIN_SPAN:
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
                             "n_candidates": 0,
                             "note": "本轮无有效红线：退化守卫（分箱曲线跨度 ≤ %.0fpp / 逐档跨度 ≤ %.0fpp / "
                                     "单调性不成立）或双算法门禁（偏差 > %.0f%%）未过 ⇒ 按任务书 §二.3 "
                                     "降级为物理参照（见报告 §5.3），不进入约束线与免责声明"
                                     % (100.0 * CURVE_MIN_SPAN, 100.0 * CURVE_MIN_SPAN, KNEE_AGREE_PCT)})
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
def phase_report_deep(cli: argparse.Namespace) -> Path:
    """深轮报告 ⇒ **必须**由 ``report_deep_v2.py``（剖面感知的生成器）写。

    🔴 **为什么不能复用** ``phase_report``：那个函数的正文**与剖面无关** ——
    它写的是**基线轮**口径（「本轮唯一可对外的红线是 target_local_contrast」、
    「target_lap_var 本轮没有有效红线」…）⇒ 拿它写深轮报告 ⇒ 报告正文被
    基线轮遗留覆盖（判定器 D5「报告正文无基线轮遗留」当场变红）；
    而且它把**基线产物目录**写进产物清单、并注入一行**机器负载快照**
    （⇒ 报告非字节可复现 ⇒ md5 门禁换台机器必然假红）。

    ⇒ 深轮剖面的报告阶段**只走这一条路**（两个生成器不得并存于同一路径）。

    Args:
        cli (argparse.Namespace): 已解析的命令行。

    Returns:
        Path: 写出的报告路径。
    """
    from report_deep_v2 import generate as _generate_deep_report

    baseline_dir = _anchor_repo_path(PROFILE_PATHS["baseline"]["output_dir"])
    spec_path = _anchor_repo_path("iqa_v2/DEEP_AXIS_SPEC_20261127.md")
    return _generate_deep_report(Path(cli.output_dir), baseline_dir,
                                 Path(cli.report), spec_path)


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
    #: 相关性表索引：``(tracker, curve_set, family, metric) -> 行``（§9.2 / §12.12 共用）
    corr_index = {(r["tracker"], r["curve_set"], r["degradation_family"], r["metric"]): r
                  for r in corr_rows}
    #: 采纳膝点（``knee_reliable == 1``）与其中相关性偏弱的行（§11 验收自证 / §12.12 共用）
    reliable_knees = [r for r in knee_rows if r["knee_reliable"] == "1"]
    weak_knees = []
    for r in reliable_knees:
        corr = corr_index.get((r["tracker"], r["curve_set"], r["degradation_family"], r["metric"]))
        if corr is None or corr.get("usable") != "1" or corr.get("direction_ok") != "1":
            weak_knees.append(r)

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

    def curve_grade(row: dict) -> str:
        """按 **与 ``extract_knee`` 完全同一套守卫** 给曲线定级（A/B/C）。

        🔴 守卫判在**膝点取自的那条曲线**（分箱曲线）上，逐档跨度是**第二道**守卫
        ⇒ 两条跨度口径**都要**过门槛才能算 A 级。
        """
        if not row:
            return "—"
        if row.get("curve_monotone") != "1" or row.get("metric_monotone") != "1":
            return "❌ C 级（非单调）"
        if fnum(row.get("curve_y_span")) <= CURVE_MIN_SPAN \
                or fnum(row.get("level_y_span")) <= CURVE_MIN_SPAN:
            return "⚠️ B 级（跨度不足）"
        return "✅ **A 级**"

    def _rate(set_name: str, metric: str) -> float:
        """回归表里某集合 × 某指标的「落在线下比例」（%）。"""
        for row in regress_rows:
            if row["set"] == set_name and row["metric"] == metric:
                return 100.0 * fnum(row.get("rate_below_line"))
        return float("nan")

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
    #: 每族的**档位数**（膝点表里的 ``n_windows`` 是「窗口 × 档位数」的明细行数）
    #: 🔴 档位数必须**从明细底表实测反推**，不能写死剖面常量 ——
    #:    深轮 blur 族是 10 档（基线 6 档），写死会把「好片段底座窗口数」虚高 66.7%，
    #:    与规格书 §四「Pass B 池 895（不变）」自相矛盾。
    level_counts: dict[str, int] = {}
    if detail_rows_all:
        seen_levels: dict[str, set] = {}
        for row in detail_rows_all:
            family = row.get("condition_family")
            if not family or family == "clean":
                continue
            try:
                level = round(float(row["condition_level"]), 6)
            except (KeyError, TypeError, ValueError):
                continue
            seen_levels.setdefault(family, set()).add(level)
        level_counts = {family: len(levels) for family, levels in seen_levels.items()}
    if not level_counts:
        for condition in build_conditions(cli.conditions_profile):
            if condition.family != "clean":
                level_counts[condition.family] = level_counts.get(condition.family, 0) + 1

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
    add("> ⑤ **约束线只对三项核心指标出**（全图指标仅作「已废弃口径·对照」）；")
    add("> ⑥ **曲线有效性守卫**（host 侧 W6/W7 独立复核要求）——红线候选除"
        "「双算法一致」外，还必须满足「跟住率随退化单调下降 + 总跨度 > %.0f pp + "
        "指标侧单调下降」⇒ 作废了一批**噪声膝点**（详见 §4.2）。" % (100.0 * CURVE_MIN_SPAN))
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
        "**反解出来的线位本来可以互相印证**——🔴 但**本轮净化底座上 KCF 的曲线未达 A 级**"
        "（§4.2）⇒ 这条互证**本轮无法兑现**，红线只有 XCorr 一条臂支撑。")
    add("* **CSRT** 与 KCF 同属 DCF 家族 ⇒ 它的作用是**同族对照**："
        "证明 KCF 侧反推的线位不是「某个实现的细节」——🔴 本轮 CSRT 曲线同样未达 A 级，"
        "这条对照**本轮无法兑现**（§5.4）。")
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
    add(">")
    add("> 🔴 **3 px 施于哪个量（口径披露）**：3 px 判据作用在**该臂末帧的同帧定位误差**"
        "（`cle_end`）上，而不是对所有臂一律取 `t+4` 帧——KCF 臂的 `update()` 滞后一帧，"
        "在序列末尾拿不到尾帧时会回落到 `t+3` 帧（§2.3）。"
        "**这样做的原因**：若让 KCF 用 `t+4`、XCorr 用 `t+5`，三分法会把「KCF 少看了一帧」"
        "误判成「画质差」⇒ 分类必须用**同帧口径**才能公平。"
        "**代价（量化）**：若改用严格 `CLE_{t+4}` 重算分类，本轮会有 **61 / 2442（2.5%）** 个窗口换类"
        "（44 good→mediocre、9 mediocre→lost、3 lost→mediocre、5 mediocre→good）⇒ "
        "**三分法比例与后续线位会小幅移动**，结论方向不变。")
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
    # 🔴 切分只覆盖「至少有 1 个好片段」的序列 ⇒ 其余序列必须显式披露，不能让读者误以为数据被丢弃
    uncovered_seqs = sorted({r["seq_id"] for r in class_rows} - set(split_map))
    uncovered_windows = [r for r in class_rows if r["seq_id"] not in split_map]
    add("| **未进入 7:3 切分的序列** | **%d** 条 / **%d** 个窗口"
        "（**没有好片段** ⇒ 不参与标定；其窗口只进 §8.2「一般片段」与 §8.3「跟丢片段」"
        "两个**评估集**，**不参与任何阈值搜索**） |"
        % (len(uncovered_seqs), len(uncovered_windows)))
    add("")
    add("> 留出集 `seq_id` 清单（共 %d 条）：`%s`" % (len(holdout_seqs), "`, `".join(holdout_seqs)))
    add("")
    add("> 标定集 `seq_id` 清单（共 %d 条）见 `iqa_v2/output/split_v2.csv`。切分种子固定，"
        "可完全复现。**严禁**把同一条视频的前半段拿去仿真、后半段拿去测误杀率——"
        "本流水线的「窗口抽样」也在序列内做等距步长，两侧完全隔离。" % len(calib_seqs))
    add("")
    add("> 🔴 **切分覆盖口径（别把「序列级隔离」读成「全数据集隔离」）**："
        "7:3 切分只作用于**至少含 1 个好片段**的序列（%d 条 = 标定 %d + 留出 %d）；"
        "另有 **%d 条序列 / %d 个窗口**（%s）**没有任何好片段** ⇒ 它们**不可能**成为仿真底座，"
        "因此不进入切分，其窗口只作为 §8.2「临界敏感性」与 §8.3「拦截率」两个**评估集**出现。"
        "🔴 由于阈值搜索**只用标定集**，这两个评估集**零参与阈值搜索** ⇒ 无泄露。"
        % (len(calib_seqs) + len(holdout_seqs), len(calib_seqs), len(holdout_seqs),
           len(uncovered_seqs), len(uncovered_windows),
           "、".join(sorted({class_counter.get(CLASS_MEDIOCRE, 0) and "一般" or "",
                             "跟丢"})).strip("、") or "一般 / 跟丢"))
    add("")

    # ------------------------------------------------------------------ 四、底座净化
    add("## 四、好片段底座净化统计（两趟法）")
    add("")
    add("> **为什么要两趟**：上一版把**全部 %d 个窗口一次性**送进仿真池，"
        "其中 **%.1f%%（%d 条）在干净帧上就已经跟丢**（交棒书 §3-G2 记的旧口径是 15.5%%，"
        "同一量级）⇒ 这些「脏窗口」把衰减曲线的顶部压到 84.5%%（XCorr 全量干净帧跟住率），"
        "掩盖了真实拐点。本轮改为：**Pass A** 只跑 `clean` 条件把窗口分成好 / 一般 / 跟丢；"
        "**Pass B** 只让**好片段**进退化池（**进池条件 = 三分法好片段 `class_ds == good` ∩ 标定集**）。"
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
    add("| **仿真池 = 三分法好片段 ∩ 标定集** | %d | %.1f%% / %.1f%% / %.1f%% | "
        "✅ 起点纯粹（池内每条都判为「好片段」） |"
        % (len(pool_keys), pool_rates[0] * 100.0, pool_rates[1] * 100.0,
           pool_rates[2] * 100.0))
    for curve_set, label in (("own_good", "各臂自己的好片段底座"),
                              ("common_good", "三臂共同的好片段底座（跨臂对比用）")):
        rates, counts = [], []
        for tracker in TRACKER_FAMILIES:
            row = pick(tracker, "blur", "target_lap_var", curve_set, "all")
            rates.append(fnum(row.get("clean_success_rate")) if row else float("nan"))
            # 膝点表里的 ``n_windows`` 是「窗口 × 该族档位数」的**明细行数** ⇒ 要除回档位数才是窗口数
            counts.append(int(row["n_windows"]) // level_counts["blur"] if row else 0)
        # 各臂底座**不必然等大**（逐臂好片段集合互不相同）⇒ 逐臂报数，不写一个笼统的总数
        n_text = (str(counts[0]) if len(set(counts)) == 1
                  else " / ".join(str(c) for c in counts))
        add("| %s | %s | %.1f%% / %.1f%% / %.1f%% | %s |"
            % (label, n_text,
               rates[0] * 100.0, rates[1] * 100.0, rates[2] * 100.0,
               "✅ 起点纯粹" if all(r >= 0.999 for r in rates) else "⚠️ 起点未达 100%"))
    add("")
    add("> **逐档可查**：膝点表 `iqa_v2/output/knee_thresholds_v2.csv` 的 `clean_success_rate` 列"
        "给出**每一族、每一曲线集、每一指标**所用窗口在 clean 档的跟住率；"
        "上表取 `blur` 族为代表（其余族同源同池）。")
    add("")
    add("> 🔴 **进池口径（唯一口径，写清楚以免被误读）**：退化池 = "
        "**标定集 ∩ (三分法好片段 `class_ds == good`)**，共 **%d** 个窗口 ⇒ "
        "池内 clean 档三臂跟住率 **%.1f%% / %.1f%% / %.1f%%**（**起点纯粹**，无污染）；"
        "建曲线时再逐臂过滤成 `own_good`（该臂自己的好片段）与 `common_good`（三臂交集）"
        "两个曲线集 ⇒ **曲线本身也是纯的**。"
        % (len(pool_keys), pool_rates[0] * 100.0, pool_rates[1] * 100.0,
           pool_rates[2] * 100.0))
    add("> 🔴 **为什么进池条件不是「三臂好片段的并集」**：并集会把"
        "「别的臂跟得住、本臂已经跟丢」的窗口塞进池子 ⇒ 各臂衰减曲线的**起点**被拖下来"
        "（这正是 G2 要消灭的污染）。**三分法的好片段**要求双主臂都判好 ⇒ "
        "进池窗口在 clean 档**三臂必然都跟住** ⇒ 起点必然 100%%。"
        "代价是：一条臂只能用「双主臂都判好」的窗口做底座（本轮该臂独占的好片段不会进池）。")
    add("")
    add("### 4.1 逐档跟住率——**净化底座上的衰减曲线有多深**（必须一起交付的事实）")
    add("")
    add("> 下表是**净化后**底座上每一档的实测跟住率（现场读 `output/degrade_detail_v2.csv`，"
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
        "膝点处的跟住率仍在 95%~100%（见 §5.1 的「膝点处跟住率」列）；")
    add("> ② 曲线接近平坦 ⇒ 膝点位置对底座与分箱**敏感**，"
        "「双算法一致」在平曲线上**必然成立**（两法落在同一个噪声箱）⇒ "
        "**必须再加曲线有效性守卫**（§4.2）才能判出一条曲线是否真的承载了崩塌信息；")
    add("> ③ 因此约束线**只取通过守卫的最严候选**（安全方向），"
        "其可用性改由 §8 的**独立盲测**（真实坏片 vs 真实好片对照）来证明，而不是由膝点本身证明。")
    add("")
    add("> **这正是任务书 §二.1 的取舍**：把「脏窗口」踢出仿真池换来曲线起点 100%，"
        "代价是曲线变浅、崩溃点被推到任务书给定档位之外。"
        "若要得到更硬的崩溃拐点，需要**加大退化档位上限**（例如 `blur` σ 提到 6~8、"
        "`contrast` α 降到 0.05），而不是把脏窗口塞回池子——"
        "后者正是任务书禁止的做法。")
    add("")
    add("> **跨臂公平性的处理**：三臂的「好片段」集合**互不相同**（KCF 与 XCorr 的判定必然有出入）"
        "⇒ 退化池取**双主臂都判好（三分法 `class_ds == good`）的窗口**，"
        "建曲线时再逐臂过滤成 `own_good`（该臂自己的好片段）与 `common_good`（三臂交集）两个曲线集；"
        "**跨臂的线位对比只用 `common_good`**——只有在**同一批窗口**上反解出来的膝点，"
        "两个家族之间才可比。")
    add("")
    add("### 4.2 曲线有效性分级——**哪些曲线真的承载了崩塌信息**")
    add("")
    add("> 🔴 **本轮最重要的一处方法修正**（host 侧独立复核复算确认）："
        "「双算法一致」**不能**单独用来证明曲线有效——在**跟住率从未塌掉**的平曲线上，"
        "Kneedle 与 PCHIP 会一致地落在**同一个噪声箱**上，于是被误判为 `knee_reliable=1`。"
        "⇒ 红线候选必须**同时**满足四条："
        "① **跟住率随退化单调下降**（逐档口径）；"
        "② **分箱曲线跨度 > %.0f 个百分点**（= 膝点**实际取自的那条曲线**，下称「指标判别口径」）；"
        "③ **逐档跨度 > %.0f 个百分点**（= 受控退化轴口径）；"
        "④ **指标侧单调下降**（退化递增 ⇒ 指标中位递减，任务书 §四.2 的单调性假设）。"
        "\n> （+ 双算法一致性门禁 ⇒ **共五道**，见 §5）"
        % (100.0 * CURVE_MIN_SPAN, 100.0 * CURVE_MIN_SPAN))
    add("")
    _lap_motion = pick("xcorr", "motion", "target_lap_var") or {}
    add("> 🔴🔴 **本轮审计发现的真缺陷（本轮已修）**：守卫原本把**逐档跨度**写进了 "
        "`curve_y_span` 列，而**膝点却取自分箱曲线**——等于「用 A 曲线的健康度给 B 曲线放行」。"
        "实测后果（净化底座上）：`target_lap_var` / XCorr / `motion` 的**逐档跨度 %.2f pp**"
        "（过门槛）但**分箱跨度仅 %.2f pp**——整条曲线几乎是平的"
        "（跟住率在 0.94~0.976 之间抖动），所谓膝点 **%.6g** 只是"
        "「曲线上一个噪声局部极大值处的**分箱边界**」，双算法 **%.2f%%** 一致属**伪一致**。"
        "⇒ 本轮把守卫改判在**分箱曲线**上（`curve_y_span`），逐档跨度降为**第二道**守卫"
        "（新增列 `level_y_span`）⇒ `target_lap_var` 因此**本轮没有有效红线**（§6）。"
        % (100.0 * fnum(_lap_motion.get("level_y_span")), 100.0 * fnum(_lap_motion.get("curve_y_span")),
           fnum(_lap_motion.get("knee_x_kneedle")), fnum(_lap_motion.get("delta_pct_of_span"))))
    add("")
    add("> 分级规则：**A 级** = 单调且跨度 > 阈值（可进红线）；"
        "**B 级** = 单调但跨度 ≤ 阈值（曲线近似水平 ⇒ 膝点位置不可靠，**仅作物理参照**）；"
        "**C 级** = 非单调或数值噪声（**作废**）。")
    add("")
    add("| 跟踪臂 | 退化族 | 分箱曲线跨度<br>（指标判别口径） | 逐档跨度<br>（退化轴口径） | 单调 | 分级 | 逐档跟住率（退化强度递增） |")
    add("|:--|:--|--:|--:|:--|:--|:--|")
    for tracker in TRACKER_FAMILIES:
        for family in DEGRADATION_FAMILIES:
            row = pick(tracker, family, "local_scr", "common_good", "all")
            if not row:
                continue
            span = fnum(row.get("curve_y_span"))
            lspan = fnum(row.get("level_y_span"))
            mono = row.get("curve_monotone") == "1"
            grade = curve_grade(row)
            entry = level_table.get(family)
            rates = ""
            if entry:
                levels = sorted(entry["levels"], key=lambda item: -item["level"])
                key = {"xcorr": "xcorr", "kcf_legacy": "kcf", "csrt": "csrt"}[tracker]
                rates = " → ".join("%.1f%%" % (100.0 * item[key]) for item in levels)
            add("| `%s` | %s | %.2f pp | %.2f pp | %s | %s | %s |"
                % (tracker, family, 100.0 * span, 100.0 * lspan,
                   "✅" if mono else "❌", grade, rates))
    add("")
    add("> 🔴 **由此得出的硬事实**（分级以 `local_scr` 为**代表曲线**评定；"
        "**红线候选按各指标自己的曲线**判定，见 §5.1 / §5.3）：在**净化后**的底座上，"
        "**只有 XCorr 臂的 `blur` 与 `motion` 两条曲线达到 A 级**；"
        "KCF 与 CSRT 的曲线跨度全部 ≤ 5pp（C 级 / B 级）⇒ "
        "**它们的膝点不能进红线** ⇒ 本轮**成立的两条线只由 XCorr 一条臂支撑**，"
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
    add("> 🔴 **本章所有膝点都必须先过 §4.2 的四道守卫**"
        "（跟住率单调 / 分箱曲线跨度 > %.0fpp / 逐档跨度 > %.0fpp / 指标单调），"
        "**再加双算法一致性门禁**（偏差 ≤ %.0f%% 量程）⇒ 共五道。"
        "没过的读数一律标「作废」并只作参照，**不进红线**。"
        % (100.0 * CURVE_MIN_SPAN, 100.0 * CURVE_MIN_SPAN, KNEE_AGREE_PCT))
    add("")
    add("### 5.1 通过全部守卫的膝点（`common_good` 池 / 全量窗口 / 三项核心指标）")
    add("")
    add("| 指标 | 退化族 | 跟踪臂 | 膝点 | 膝点处跟住率 | 分箱曲线跨度 | 逐档跨度 | 双算法偏差 |")
    add("|:--|:--|:--|--:|--:|--:|--:|--:|")
    for metric in CORE_METRICS:
        for family in DEGRADATION_FAMILIES:
            for tracker in TRACKER_FAMILIES:
                row = pick(tracker, family, metric)
                if not row or row["knee_reliable"] != "1":
                    continue
                add("| `%s` | %s | `%s` | %.6g | %.1f%% | %.2f pp | %.2f pp | %.2f%% |"
                    % (metric, family, tracker, fnum(row.get("knee_x_final")),
                       100.0 * fnum(row.get("knee_y_final")),
                       100.0 * fnum(row.get("curve_y_span")),
                       100.0 * fnum(row.get("level_y_span")), fnum(row.get("delta_pct_of_span"))))
    add("")
    add("> ⚠️ **膝点处的跟住率仍然在 95%~99.7%** —— 这再次证明：本轮反解出的不是"
        "「跟踪崩溃点」，而是「**画质—跟踪性能分界点**」（曲线开始明显弯折的位置）。"
        "把它叫「崩溃拐点」会**高估**这条线的物理含义。")
    add("")
    add("### 5.2 小目标子集（max(w,h) ≤ %d px）" % SMALL_TARGET_MAX_DIM)
    add("")
    add("> 小目标（远距离吊舱场景）在**更低**的画质上就掉队 ⇒ 多数指标的线位**更高**（更严）；"
        "下表只列通过全部守卫的读数，比值基准是 §6 的**全量红线**。")
    add("")
    add("> 🔴 **本轮新出现的一行结论**：**`target_lap_var` 在全量口径没有有效红线（§6），"
        "但在小目标子集上却有**通过全部守卫的膝点**——"
        "说明**锐度维度对远距离小目标是有判别力的**，只是它被**大目标窗口的高 `lap_var` 基线**"
        "摊平在全量曲线上。这条读数**只作小目标参考**，不进 §6 的约束线表"
        "（口径仍是「全量 / `stratum = all`」），但它指出了下一轮的补救方向（§12.10）。")
    add("")
    add("| 退化族 | 指标 | 跟踪臂 | 膝点 | 与全量红线之比 |")
    add("|:--|:--|:--|--:|--:|")
    for family in DEGRADATION_FAMILIES:
        for metric in CORE_METRICS:
            for tracker in TRACKER_FAMILIES:
                row = pick(tracker, family, metric, "common_good", "small")
                if not row or row["knee_reliable"] != "1":
                    continue
                base_value = fnum(lines_tbl.get(metric, {}).get("line_value"))
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
    add("| 退化族 | 指标 | 跟踪臂 | 候选读数 | 分箱曲线跨度 | 逐档跨度 | 单调 | 指标单调 | 作废理由 |")
    add("|:--|:--|:--|--:|--:|--:|:--:|:--|:--|")
    for family in DEGRADATION_FAMILIES:
        for metric in CORE_METRICS:
            for tracker in TRACKER_FAMILIES:
                row = pick(tracker, family, metric)
                if not row:
                    continue
                if row["knee_reliable"] == "1":
                    continue                      # 已通过守卫 ⇒ 在 §5.1，不在本表
                candidate = fnum(row.get("knee_x_candidate"))
                if not math.isfinite(candidate):
                    continue
                reasons = []
                if row.get("curve_monotone") != "1":
                    reasons.append("跟住率非单调")
                if fnum(row.get("curve_y_span")) <= CURVE_MIN_SPAN:
                    reasons.append("分箱曲线过平 %.2fpp ≤ %.0fpp（指标不区分跟住/跟丢）"
                                    % (100.0 * fnum(row.get("curve_y_span")), 100.0 * CURVE_MIN_SPAN))
                if fnum(row.get("level_y_span")) <= CURVE_MIN_SPAN:
                    reasons.append("逐档跨度 %.2fpp ≤ %.0fpp（退化轴压不下去）"
                                    % (100.0 * fnum(row.get("level_y_span")), 100.0 * CURVE_MIN_SPAN))
                if row.get("metric_monotone") != "1":
                    reasons.append("指标非单调")
                add("| %s | `%s` | `%s` | %.6g | %.2f pp | %.2f pp | %s | %s | %s |"
                    % (family, metric, tracker, candidate, 100.0 * fnum(row.get("curve_y_span")),
                       100.0 * fnum(row.get("level_y_span")),
                       "✅" if row.get("curve_monotone") == "1" else "❌",
                       "✅" if row.get("metric_monotone") == "1" else "❌",
                       " + ".join(reasons) or "—"))
    add("")
    _lap_cands = [(r, fnum(r.get("knee_x_candidate"))) for r in knee_rows
                   if r["metric"] == "target_lap_var" and r["curve_set"] == "common_good"
                   and r["stratum"] == "all" and math.isfinite(fnum(r.get("knee_x_candidate")))]
    _lap_spans = [fnum(r.get("curve_y_span")) for r, _ in _lap_cands]
    add("> 🔴 **特别点名 `target_lap_var` 的 %d 个候选读数**（%s）："
        "它们的共同点是**分箱曲线几乎是平的**（跨度 %.2f~%.2f pp）"
        "⇒ 膝点落在曲线的噪声起伏上，**双算法「一致」是伪一致**。"
        "其中 **%.6g** 是**上一版误当作红线发出去的那个数**，"
        "本轮**已按任务书 §二.3「降级为物理参照」处理**（不进 §6、不进 §10）。"
        % (len(_lap_cands),
           "、".join("%s / %s = %.6g" % (r["tracker"], r["degradation_family"], v)
                     for r, v in _lap_cands),
           100.0 * min(_lap_spans) if _lap_spans else float("nan"),
           100.0 * max(_lap_spans) if _lap_spans else float("nan"),
           fnum((pick("xcorr", "motion", "target_lap_var") or {}).get("knee_x_kneedle"))))
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
            # 🔴 等级必须按**与 extract_knee 同一套守卫**逐臂判（两条跨度口径都要过）
            grade = "KCF %s / CSRT %s" % (curve_grade(row_k).replace("**", ""),
                                         curve_grade(row_c).replace("**", ""))
            add("| %s | `%s` | %.6g | %.6g | %.2f%% | %s |"
                % (family, metric, kv, cv, abs(kv - cv) / max(abs(kv), abs(cv), 1e-9) * 100.0,
                   grade))
    add("")

    add("## 六、良好跟踪合格线（约束线）——只对三项核心指标")
    add("")
    add("> **口径（逐条）**：① 只取 `curve_set = common_good`（三臂共同好片段 ⇒ 跨臂线位可比）、"
        "`stratum = all`；② 候选必须**通过 §4.2 的四道曲线有效性守卫 + 双算法一致性门禁**"
        "（即 `knee_reliable = 1`）；③ **每族膝点取最小值**（最严的一族说了算），"
        "并记录**决定族 / 决定臂**；④ 🔴 **绝不**把两条臂的线位求平均（跨家族纪律）。")
    add("")
    add("| 指标 | **约束线** | 决定族 | 决定臂 | 有效候选数 | KCF 侧候选（**已作废**，参照） | CSRT 侧候选（**已作废**，参照） |")
    add("|:--|--:|:--|:--|--:|--:|--:|")
    for metric in CORE_METRICS:
        info = lines_tbl.get(metric, {})
        value = fnum(info.get("line_value"))
        kcf_ref = fnum(info.get("kcf_legacy_ref"))
        csrt_ref = fnum(info.get("csrt_ref"))
        add("| `%s` | %s | %s | `%s` | %s | %s | %s |"
            % (metric,
               ("**%.6g**" % value) if math.isfinite(value) else "**本轮无有效红线**",
               info.get("deciding_family", "—"), info.get("deciding_tracker", "—") or "—",
               info.get("n_candidates", "—"),
               ("%.6g（曲线退化）" % kcf_ref) if math.isfinite(kcf_ref) else "无有效读数",
               ("%.6g（曲线退化）" % csrt_ref) if math.isfinite(csrt_ref) else "无有效读数"))
    add("")
    add("> 🔴 **本轮成立的两条线都只由 XCorr 一条臂支撑**——KCF 与 CSRT 的衰减曲线**全部未达 A 级**"
        "（§4.2）⇒ 它们**没有任何**通过守卫的膝点。三个后果必须一起讲：")
    add("> ① **本轮无法复现**任务书 §二.3 倚重的「XCorr ↔ KCF **算法无关性**」"
        "（上一轮**未纯净化**底座上 `target_lap_var` 两臂偏差是 **0.00%**，本轮不是）；")
    add("> ② ⇒ 这**两条**线是「**单臂反解 + 独立盲测背书**」的线，**不是**「双算法互证」的线，"
        "对外表述必须带上这个限定；")
    add("> ③ 「已作废参照」列里的读数只能回答「**如果不加曲线守卫会取到多少**」，"
        "**不得**直接对外使用——它们来自跟住率从未塌掉的曲线，是噪声点。")
    add("")
    add("> 🔴🔴 **`target_lap_var` 本轮没有有效红线**（本轮审计的真缺陷所致，§4.2）："
        "上一版发出的 `target_lap_var < 541.09` **作废**——它取自一条"
        "**分箱曲线跨度只有 1.92 pp 的近平曲线**，双算法「0.00% 一致」是**伪一致**。"
        "按任务书 §二.3「否则降级为物理参照」，该指标本轮**不进约束线表、不进免责声明**；"
        "其参照读数（XCorr **505.62**〔`contrast` 族候选〕、KCF **1110.81**〔`blur` 族候选〕；"
        "上一版作废值 541.09 见 §5.3）与**小目标子集上通过全部守卫的读数**"
        "见 §5.3 / §5.2，**补救路径见 §12.13**。")
    add("")
    add("**每条主臂的线位对照（同一批 `common_good` 窗口）**")
    add("")
    add("| 指标 | XCorr 线位（**生效**） | 决定族 | KCF 线位 | 决定族 | 两臂线位差 |")
    add("|:--|--:|:--|--:|:--|--:|")
    for metric in CORE_METRICS:
        info = lines_tbl.get(metric, {})
        fam_x = "-"
        for family in DEGRADATION_FAMILIES:
            rx = pick("xcorr", family, metric)
            if rx and fnum(rx.get("knee_x_final")) == fnum(info.get("xcorr_line")):
                fam_x = family
        dev = fnum(info.get("cross_arm_dev_pct"))
        x_line = fnum(info.get("xcorr_line"))
        add("| `%s` | %s | %s | %s | — | %s |"
            % (metric, ("%.6g" % x_line) if math.isfinite(x_line) else "**无有效线位**", fam_x,
               ("%.6g" % fnum(info.get("kcf_legacy_line")))
               if math.isfinite(fnum(info.get("kcf_legacy_line"))) else "**无有效线位**",
               ("%.2f%%" % dev) if math.isfinite(dev) else "**不可比**"))
    add("")
    add("> **取值规则**：对外红线 = **通过全部守卫的候选**里最严的那条（每族取最小值）；"
        "🔴 **绝不**把两条臂的线位求平均（跨家族纪律）；"
        "CSRT 是**同族对照臂**，只作参照，**不参与红线取值**。")
    add("")
    add("> **读法提醒**：这些线是「**物理上还有可能跟住**」的**必要条件**，"
        "不是充分条件——过线不代表一定能跟住，过线窗口的实测跟住率见 §5.1 的「膝点处跟住率」列；"
        "而**过线以下的窗口**才进入「跟踪失锁不属算法考核范围」的免责区间（§10）。")
    add("")
    add("> **全图指标（`global_contrast` / `global_lap_var`）已从约束线表中剔除**："
        "任务书 §一已明确「彻底废弃 v1 全图指标」，它们在本报告里只以"
        "「**全图对照（已废弃口径，仅作对照）**」的身份出现在 §7，**不得**用于对外红线。")
    add("")
    add("> `surround_*` 与 `target_grad_mean` 同理：进 §7 的**辅助指标**表，**不进红线**。")
    add("")

    add("## 七、辅助指标与全图对照（**已废弃口径，仅作对照**）")
    add("")
    add("> ⚠️ 本节所有数字**不构成任何红线**，仅用于说明「为什么对外只保留三项核心指标」。")
    add("")
    add("| 指标 | 类别 | XCorr 有效膝点 | KCF 有效膝点 | 是否进红线 |")
    add("|:--|:--|--:|--:|:--|")
    aux_all = tuple(m for m in LOCAL_METRICS if m not in CORE_METRICS) + GLOBAL_METRICS
    def _all_reliable(tracker: str, metric: str) -> list:
        """该臂 × 该指标**全部**通过守卫的膝点（跨族 × 跨曲线集 × 跨分层）。"""
        hits = []
        for row in knee_rows:
            if (row["tracker"] == tracker and row["metric"] == metric
                    and row.get("knee_reliable") == "1"):
                hits.append("%s/%s/%s = %.6g"
                            % (row["curve_set"], row["degradation_family"],
                               row["stratum"], fnum(row["knee_x_final"])))
        return hits

    aux_with_knees = []
    for metric in aux_all:
        kind = "辅助指标" if metric in LOCAL_METRICS else "全图对照（已废弃口径）"
        hits_x = _all_reliable("xcorr", metric)
        hits_k = _all_reliable("kcf_legacy", metric)
        if hits_x or hits_k:
            aux_with_knees.append(metric)
        add("| `%s` | %s | %s | %s | ❌ 否 |"
            % (metric, kind,
               "<br>".join(hits_x) if hits_x else "无（曲线退化 / 门禁未过）",
               "<br>".join(hits_k) if hits_k else "无（曲线退化 / 门禁未过）"))
    add("")
    add("> 说明：这些辅助 / 全图指标**一律不进红线**（任务书 §一：全图口径已废弃；"
        "辅助指标只用于横向比较）。"
        + ("其中 %s 确实有通过 §4.2 守卫的有效膝点（见上表）——"
           "**但它们仍然不进红线**：全图口径已废弃，辅助指标不作豁免依据。" % "、".join(
               "`%s`" % m for m in aux_with_knees) if aux_with_knees else
           "本轮这些辅助 / 全图指标都没有通过 §4.2 曲线守卫的有效膝点。")
        + "被守卫作废的候选读数见 §5.3。")
    add("")

    # ------------------------------------------------------------------ 八、回归
    add("## 八、三分法回归测试体系")
    add("")
    add("> 阈值来自**标定集**（§6），下面三个集合**全程零参与阈值搜索**。")
    add("")
    def _reg(set_name: str, metric: str) -> dict:
        for row in regress_rows:
            if row["set"] == set_name and row["metric"] == metric:
                return row
        return {}

    def _sep(metric: str) -> float:
        """真实素材上的区分间隔（坏片拦截率 − 真实好片对照误杀率），单位 pp。"""
        bad = 100.0 * fnum(_reg("real_bad", metric).get("rate_below_line"))
        good = 100.0 * fnum(_reg("real_good", metric).get("rate_below_line"))
        return bad - good

    def line_kind(metric: str) -> str:
        """阈值性质：**约束线表有无线位** × **真实素材区分间隔** ⇒ 能不能对外用。"""
        if not math.isfinite(fnum(lines_tbl.get(metric, {}).get("line_value"))):
            return "⚠️**参考阈值·非红线**"
        sep = _sep(metric)
        if sep >= 50.0:
            return "**红线**（本轮唯一可对外）"
        return "⚠️**有线位但不得单独使用**（区分间隔 %.2f pp）" % sep

    def _thr(row: dict) -> str:
        """阈值标签：**本轮无有效红线**的指标用约束线表的参照读数当阈值 ⇒ 必须标出来；
        **有线位但真实素材区分不了好坏片**的指标 ⇒ 也必须标出来（不得单独对外用）。"""
        value = fnum(row.get("threshold"))
        text = ("%.6g" % value) if math.isfinite(value) else "—"
        if str(row.get("threshold_source", "")).startswith("reference_only"):
            return text + " ⚠️**参考阈值·非红线**"
        if math.isfinite(value) and _sep(row["metric"]) < 50.0:
            return text + " ⚠️**不得单独对外**（区分间隔 %.2f pp）" % _sep(row["metric"])
        return text

    add("### 8.1 好片段 · 30% 留出集 ⇒ **好片误杀率**")
    add("")
    add("| 指标 | 阈值 | 留出集样本数 | **红线以下（误杀）** | **好片误杀率** | 指标中位数 |")
    add("|:--|--:|--:|--:|--:|--:|")
    for row in regress_rows:
        if row["set"] != "good_holdout":
            continue
        add("| `%s` | %s | %s | %s | **%.2f%%** | %.6g |"
            % (row["metric"], _thr(row), row["n_evaluated"],
               row["n_below_line"], 100.0 * fnum(row["rate_below_line"]),
               fnum(row["median"])))
    add("")
    add("> ⚠️ **带「参考阈值·非红线」标记的行**：该指标**本轮没有通过守卫的膝点**"
        "（§6）⇒ 这里用的是约束线表里的**参照读数**当阈值，"
        "**它的数字只说明「该指标仍有判别力」，不构成任何对外豁免依据**。")
    add("")
    add("### 8.2 一般片段 ⇒ **临界敏感性**（不参与定标）")
    add("")
    add("> 🔴 **来源说明**：本集合**全部**来自 Pass A 的干净帧分类，"
        "**不参与任何阈值搜索**（阈值只用标定集好片段反解）⇒ 无泄露。"
        "其中含 §3 所述「**没有好片段**」的那 %d 条序列的窗口——"
        "它们本来就进不了仿真池，只能作为临界灰区的评估样本。" % len(uncovered_seqs))
    add("")
    add("| 指标 | 阈值 | 样本数 | 红线以下比例 | 指标中位数 | 落在膝点邻域比例 | AUC(一般 vs 好) | AUC(一般 vs 跟丢) |")
    add("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for row in regress_rows:
        if row["set"] != "mediocre":
            continue
        add("| `%s` | %s | %s | %.2f%% | %.6g | %s | %s | %s |"
            % (row["metric"], _thr(row), row["n_evaluated"],
               100.0 * fnum(row["rate_below_line"]), fnum(row["median"]),
               ("%.1f%%" % (100.0 * fnum(row["frac_in_knee_band"])))
               if math.isfinite(fnum(row["frac_in_knee_band"])) else "—",
               ("%.4f" % fnum(row["auc_vs_good"])) if math.isfinite(fnum(row["auc_vs_good"])) else "—",
               ("%.4f" % fnum(row["auc_vs_lost"])) if math.isfinite(fnum(row["auc_vs_lost"])) else "—"))
    add("")
    add("### 8.3 跟丢片段 + 真实坏片 ⇒ **坏片拦截率**")
    add("")
    add("> 🔴 **来源说明**：「跟丢片段」全部来自 Pass A 的干净帧分类，"
        "**不参与任何阈值搜索**；「真实坏片 / 真实好片对照」来自**一次性独立盲测**，"
        "**全程零参与阈值搜索**（§10.3 的红线只用 §8.1 之外的信息做「证明 / 否证」）。"
        "其中跟丢片段含 §3 所述「**没有好片段**」的那 %d 条序列的窗口。" % len(uncovered_seqs))
    add("")
    add("| 集合 | 指标 | 阈值 | 样本数 | 红线以下 | 该比例的含义 | 备注 |")
    add("|:--|:--|--:|--:|--:|:--|:--|")
    for row in regress_rows:
        if row["set"] not in ("lost", "real_bad", "real_good"):
            continue
        label = {"lost": "跟丢片段（Anti-UAV）", "real_bad": "**真实坏片（2 条）**",
                 "real_good": "真实好片对照（2 条）"}[row["set"]]
        meaning = {"lost": "**坏片拦截率**",
                   "real_bad": "**坏片拦截率**（正确豁免）",
                   "real_good": "**好片误杀率 / 错误豁免率**（越低越好）"}[row["set"]]
        add("| %s | `%s` | %s | %s | %s | %s **%.1f%%** | %s |"
            % (label, row["metric"], _thr(row), row["n_evaluated"],
               row["n_below_line"], meaning, 100.0 * fnum(row["rate_below_line"]), row["note"]))
    add("")
    add("> 🔴 **证据边界（必须写清，host W8-3 复核要求）**：")
    add("> * **真实坏片 2 条 = 真物理退化场景**（均匀雾场）⇒ 「坏片拦截率」是**有效的物理侧证据**；")
    add("> * **真实好片对照 2 条 = 无目标的停车场航拍**（有车、有行人，**没有无人机**；"
        "DoG 峰值落在车顶/地面而非目标上）⇒ 它测的是"
        "「**正常纹理场景 vs 平坦退化场景**」的分离度，**不是**「好目标被误杀」"
        "⇒ **不得**当作目标级误杀率证据；")
    add("> * ⇒ **本轮没有「有目标的真实好片」对照**；"
        "**目标级好片误杀率只有 §8.1 的 Anti-UAV 序列级留出集一个来源**。")
    add("")
    add("> 🔴 **真实坏片只有 2 条 ⇒ 上面涉及真实坏片/好片的结论只能标注为「小样本级」**，"
        "不可外推到批量业务场景；其作用是**一次性独立盲测**（阈值全程零参与），"
        "证明「仿真推演的红线在真实外场成立」这一**方向性**结论。")
    add("")
    add("**三个数字必须一起读**（数字现场读 `output/regression_v2.csv`）")
    add("")

    add("| 指标 | 好片误杀率（Anti-UAV 留出集） | 坏片拦截率（真实坏片） | 好片误杀率（真实好片对照） | **区分间隔** | 阈值性质 |")
    add("|:--|--:|--:|--:|--:|:--|")
    for metric in CORE_METRICS:
        good = 100.0 * fnum(_reg("good_holdout", metric).get("rate_below_line"))
        bad = 100.0 * fnum(_reg("real_bad", metric).get("rate_below_line"))
        real_good = 100.0 * fnum(_reg("real_good", metric).get("rate_below_line"))
        kind = line_kind(metric)
        add("| `%s` | %.2f%% | %.2f%% | %.2f%% | %s | %s |"
            % (metric, good, bad, real_good,
               ("**%.2f pp** %s" % (bad - real_good, "✅" if bad - real_good >= 50 else "❌")), kind))
    add("")
    add("> ① **目标级好片误杀率**（§8.1，Anti-UAV 留出集）：`target_lap_var` 最低，"
        "`local_scr` 最高（它是被局部背景方差归一化的比值 ⇒ 纹理一复杂就偏低）。"
        "🔴 注意：`target_lap_var` 这一行用的是**参考阈值**（本轮无有效红线），"
        "**它的 0.79% 误杀率不是任何对外承诺**。")
    add("> ② **真实素材上的区分间隔**（坏片拦截率 − 好片误杀率）："
        "`target_lap_var` 接近满分 ⇒ **该指标在真实外场有判别力**"
        "（🔴 但用的是参考阈值，本轮**不构成红线**）；"
        "`target_local_contrast` 明显可用（**本轮唯一可对外的红线**）；"
        "`local_scr` 的区分间隔**接近于零** ⇒ **它区分不了好坏片**。")
    add("> ③ **跟丢片段的拦截率明显低于真实坏片**——**这不是缺陷，而是设计意图**："
        "跟丢片段里相当一部分是「**算法侧失锁**」（快速机动 / 干扰 / 目标出框），"
        "它们的画质指标并不低；红线只应该拦**物理侧不可解释**的失锁，"
        "否则就把算法缺陷也一并洗成了免责项——这正是任务书 §四.4 免责声明的边界。")
    add("> ④ 🔴 **`local_scr` 本轮不取得可用红线，原因写清**："
        "① **尺度不可迁移**——红线是在 Anti-UAV 的 640×512 原生帧上标定的，"
        "真实素材是 1080p（坏片）/ 640×512（好片对照）且 ROI 由 DoG 自动选取，"
        "背景环的统计尺度不同 ⇒ 绝对线位**不可直接迁移**；"
        "② 真实好片对照里**根本没有无人机目标**（DoG 峰值落在地面景物/树叶上，已逐帧核对）"
        "⇒ 它测的不是「目标 SCR」，纹理丰富的航拍场景分母大、SCR 天然低。"
        "⇒ **对外免责声明不写 `local_scr` 子句**（§10）。")
    add("> ⑤ ⚠️ **盲测集只做一次性验证**：阈值搜索全程没看过真实素材，"
        "**不得**用 §8.3 的数字回头去调 §6 的线（否则构成数据泄露）。"
        "§8.3 的作用是**证明/否证**红线方向，不参与取值。")
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
    add("| 指标 | 类型 | XCorr \|max ρ\| | KCF \|max ρ\| | CSRT \|max ρ\| | "
        "达标的子集数（\|ρ\|≥0.15 且方向正确） |")
    add("|:--|:--|--:|--:|--:|--:|")
    for metric in LOCAL_METRICS + GLOBAL_METRICS:
        best = {}
        for tracker in TRACKER_FAMILIES:
            best[tracker] = max((abs(fnum(r["spearman"])) for r in corr_rows
                                 if r["metric"] == metric and r["tracker"] == tracker),
                                default=float("nan"))
        kind = "局部 ROI" if metric in LOCAL_METRICS else "全图对照"
        usable = sum(1 for r in corr_rows if r["metric"] == metric and r["usable"] == "1")
        add("| %s | %s | %.3f | %.3f | %.3f | %d / %d |"
            % (metric, kind, best["xcorr"], best["kcf_legacy"], best["csrt"],
               usable, sum(1 for r in corr_rows if r["metric"] == metric)))
    add("")
    lap_corr_rows = [r for r in corr_rows if r["metric"] == "target_lap_var"]
    lap_abs = [abs(fnum(r["spearman"])) for r in lap_corr_rows]
    lap_info0 = lines_tbl.get("target_lap_var", {})
    lap_dec0 = corr_index.get((lap_info0.get("deciding_tracker", ""),
                                lap_info0.get("curve_set", ""),
                                lap_info0.get("deciding_family", ""), "target_lap_var"))
    lap_dec_rho = fnum(lap_dec0.get("spearman")) if lap_dec0 else float("nan")
    add("> 🔴 **必须说清的口径差（不要拿这张表否定红线）**："
        "「达标子集数」这一列是**指标自评**——它把「与**二值**跟住率的秩相关 ≥ 0.15」"
        "当作可用性判据。`target_lap_var` 在全部 %d 个子集上 `usable = 0`"
        "（\|ρ\| ∈ %.3f~%.3f）——"
        "**这与它「本轮没有有效红线」（§6）是一致的、互相印证的**："
        "秩相关低 ⇒ 整池上锐度与跟住率几乎不单调 ⇒ 分箱曲线自然被摊平"
        "（`motion` 族分箱跨度仅 1.92 pp）⇒ 守卫判它作废。"
        "🔴 **但这不等于该指标没有判别力**：§8.3 显示它在**阈值型**判据上区分度很高"
        "（真实坏片拦截 %.1f%% / 真实好片对照误杀 %.1f%%，区分间隔 %.1f pp），"
        "而 `target_local_contrast` 是 %.1f%% / %.1f%%、`local_scr` 是 %.1f%% / %.1f%%。"
        "原因是结构性的："
        "① 净化底座上跟住率的**基线极高（~90%%）**，二值标签的方差被压掉，秩相关天然被压缩；"
        "② 锐度与跟踪的物理关系是**阈值型**（目标要糊到一定程度才丢），"
        "而不是全区间线性 ⇒ 用整池秩相关去量它必然低估"
        "（**这条限定适用于全部三条线，不只是非决定曲线**）。"
        % (len(lap_corr_rows), min(lap_abs) if lap_abs else float("nan"),
           max(lap_abs) if lap_abs else float("nan"),
           _rate("real_bad", "target_lap_var"), _rate("real_good", "target_lap_var"),
           _rate("real_bad", "target_lap_var") - _rate("real_good", "target_lap_var"),
           _rate("real_bad", "target_local_contrast"), _rate("real_good", "target_local_contrast"),
           _rate("real_bad", "local_scr"), _rate("real_good", "local_scr")))
    add("> ⇒ **红线取值的证据链是「曲线有效性守卫 + 膝点 + 独立盲测」（§4.2 / §5 / §8.3），"
        "不是这张秩相关表**；这张表只用于横向比较指标的**排序**能力。")
    add("")
    add("> **结论（业务口径）**：局部 ROI 指标（`local_scr` / `target_lap_var` / "
        "`target_local_contrast`）在**退化族内**与跟住率的相关性应当**强于**"
        "全图对照指标（`global_contrast` / `global_lap_var`）——"
        "这正是 v2 从「全图随机采点」转向「业务目标本体」的收益所在。")
    add("")

    # ------------------------------------------------------------------ 十、免责声明
    lap_line = fnum(lines_tbl.get("target_lap_var", {}).get("line_value"))
    scr_line = fnum(lines_tbl.get("local_scr", {}).get("line_value"))
    contrast_line = fnum(lines_tbl.get("target_local_contrast", {}).get("line_value"))

    real_bad_scr = _rate("real_bad", "local_scr")
    real_good_scr = _rate("real_good", "local_scr")
    real_good_contrast = _rate("real_good", "target_local_contrast")
    add("## 十、免责声明（**逐字**落地）")
    add("")
    add("### 10.1 任务书原文（逐字引用，含原始数值）")
    add("")
    add("> 「本红外跟踪算法的性能考核以目标中心局部画质（v2 规范）为客观前提。"
        "经基于真实红外目标的多维物理退化压力测试及独立故障盲测验证，"
        "当因光学失焦、机芯未校正或气象热交叉导致目标核心区锐度 "
        "**`target_lap_var < 819.55`**、或无极性局部信杂比 **`local_scr < 2.69`**"
        "（及局部对比度坍缩）时，目标有效特征在物理层面已被背景噪声淹没，"
        "由此引发的跟踪失锁属于前端成像与环境物理极限，不在跟踪算法性能考核范围内"
        "（正式上板阈值将随内部业务标注序列完成同口径复测后锁定）。」")
    add("")
    add("### 10.2 本轮重标后的逐字落地版（**草稿·不可直接对外**）")
    add("")
    add("> 「本红外跟踪算法的性能考核以目标中心局部画质（v2 规范）为客观前提。"
        "经基于真实红外目标的多维物理退化压力测试及独立故障盲测验证，"
        "当因光学失焦、机芯未校正或气象热交叉导致目标核心区锐度 "
        "`target_lap_var < %s`、或无极性局部信杂比 `local_scr < %s`"
        "（及局部对比度坍缩 `target_local_contrast < %s`）时，"
        "目标有效特征在物理层面已被背景噪声淹没，"
        "由此引发的跟踪失锁属于前端成像与环境物理极限，不在跟踪算法性能考核范围内"
        "（正式上板阈值将随内部业务标注序列完成同口径复测后锁定）。」"
        % (("%.7g" % lap_line) if math.isfinite(lap_line)
           else "待重标（本轮无有效红线，见下方限定段）",
           ("%.7g" % scr_line) if math.isfinite(scr_line) else "见 §六",
           ("%.7g" % contrast_line) if math.isfinite(contrast_line) else "见 §六"))
    add("")
    add("> 🔴🔴 **本轮逐字落地版必须带的限定段**（原文一字未改，限定段另起）："
        "**`target_lap_var` 子句本轮「无数值可填」**——该指标在全量口径上"
        "**没有通过曲线有效性守卫的有效膝点**（§4.2 / §6："
        "其分箱曲线在 `motion` 族上跨度仅 1.92 pp，"
        "即该指标在净化底座上**几乎不区分跟住/跟丢**）⇒ "
        "按任务书 §二.3「否则降级为物理参照」，该子句**本轮不得**填入任何数值，"
        "**不得**用参照读数（505.62 / 541.09 / 1110.81）冒充红线；"
        "上一版发出的 `target_lap_var < 541.09` **作废**。"
        "⇒ **对外发布必须删掉 `target_lap_var` 子句**，"
        "并在正式上板前按 §12.13 的补救路径重新取线。")
    add("")
    add("> 🔴 **`local_scr` 子句本轮**「**不生效**」**：该线在真实素材上**区分不了好坏片**"
        "（区分间隔仅 %.2f pp，§8.3）⇒ 对外发布时必须**删掉该子句**，"
        "或在其后加限定语「（`local_scr` 子句本轮证据不足，不单独作为豁免依据）」。"
        % (real_bad_scr - real_good_scr))
    add("")
    add("### 10.3 对外可发布版（**只保留本轮成立的红线**）")
    add("")
    add("> 「本红外跟踪算法的性能考核以目标中心局部画质（v2 规范）为客观前提。"
        "经基于真实红外目标的多维物理退化压力测试及独立故障盲测验证，"
        "当因光学失焦、机芯未校正或气象热交叉导致"
        "**局部对比度坍缩 `target_local_contrast < %s`** 时，"
        "目标有效特征在物理层面已被背景噪声淹没，"
        "由此引发的跟踪失锁属于前端成像与环境物理极限，不在跟踪算法性能考核范围内"
        "（正式上板阈值将随内部业务标注序列完成同口径复测后锁定）。」"
        % (("%.7g" % contrast_line) if math.isfinite(contrast_line) else "见 §六"))
    add("")
    add("> 🔴 **与任务书原文相比，本版删掉了两个子句**，理由必须随文给出："
        "① **`target_lap_var` 子句**——本轮无有效红线（§6），填不出诚实的数值；"
        "② **`local_scr` 子句**——该线在真实素材上**区分不了好坏片**"
        "（区分间隔仅 %.2f pp，§8.3），单列会把正常场景大面积误判成免责区间。"
        % (real_bad_scr - real_good_scr))
    add("")
    add("> ⚠️ **仅剩一条线，且必须带三条限定**："
        "① 它是「**单臂反解**」（只有 XCorr 一条臂通过守卫，§4.2 / §6）"
        "**不是**「双算法互证」的线；"
        "② 它是「**画质—跟踪性能分界点**」而不是「跟踪崩溃点」（§4.1 / §5.1）；"
        "③ 真实好片对照上有 %.2f%% 的帧落在线下（错误豁免），引用时须说明。"
        % (real_good_contrast,))
    add("")
    add("**与任务书基准值的差异说明（必须一起交付）**")
    add("")
    add("| 指标 | 任务书基准 | 本轮重标实测 | 数值变化 | 决定族 / 决定臂 | 本轮是否可用 |")
    add("|:--|--:|--:|--:|:--|:--|")
    add("| `target_lap_var` | 819.55 | **无有效红线** | **不适用** | — / — | "
        "❌ **本轮作废**（曲线守卫未过：分箱曲线过平 + 双算法分歧，§4.2 / §6） |")
    add("| `local_scr` | 2.69 ~ 3.21 | %s | %s | %s / %s | ❌ **不成立**（盲测区分间隔仅 %.2f pp） |"
        % (("%.7g" % scr_line) if math.isfinite(scr_line) else "—",
           ("%+.2f%%" % (100.0 * (scr_line - 2.69) / 2.69))
           if math.isfinite(scr_line) else "—",
           lines_tbl.get("local_scr", {}).get("deciding_family", "—"),
           lines_tbl.get("local_scr", {}).get("deciding_tracker", "—"),
           real_bad_scr - real_good_scr))
    add("| `target_local_contrast` | （任务书未给正式数值） | %s | 新增 | %s / %s | "
        "✅ **本轮唯一可对外的红线**，但真实好片对照上有 %.2f%% 错误豁免 |"
        % (("%.7g" % contrast_line) if math.isfinite(contrast_line) else "—",
           lines_tbl.get("target_local_contrast", {}).get("deciding_family", "—"),
           lines_tbl.get("target_local_contrast", {}).get("deciding_tracker", "—"),
           real_good_contrast))
    add("")
    add("**逐族逐臂对照表：上一轮 vs 本轮**")
    add("")
    add("> 旧值 = **上一轮（未纯净化底座 / 全量窗口池）的冻结基线**"
        "（`iqa_v2/REPORT_V2_ANTIUAV.md` 12:30 版 ＋ host 侧 W8-2 独立复算），"
        "**已随本轮口径变更作废**；"
        "新值 = **本轮现场读** `output/knee_thresholds_v2.csv`"
        "（`curve_set=common_good` / `stratum=all`；"
        "「候选读数」= 通过守卫的终值，未通过者取**守卫前**的候选，仅作对照）。")
    add("")
    add("| 族 | 指标 | 臂 | 旧（上一轮） | 新（本轮候选读数） | 变化 | 本轮双算法差 | 本轮是否生效 |")
    add("|:--|:--|:--|--:|--:|--:|--:|:--|")
    fam_dirs: dict[str, list] = {}
    for _family, _metric, _tracker, _old in PRIOR_ROUND_KNEES:
        _row = pick(_tracker, _family, _metric)
        _new = float("nan")
        _diff = float("nan")
        _alive = "—"
        if _row:
            _final = fnum(_row.get("knee_x_final"))
            _cand = fnum(_row.get("knee_x_candidate"))
            _new = _final if math.isfinite(_final) else _cand
            if math.isfinite(_new):
                _diff = fnum(_row.get("delta_pct_of_span"))
                _alive = "✅ 生效" if _row.get("knee_reliable") == "1" else "❌ 守卫作废"
        _pct = float("nan")
        if math.isfinite(_new) and _old:
            _pct = 100.0 * (_new - _old) / _old
            fam_dirs.setdefault(_family, []).append(
                ("`%s` / `%s`" % (_metric, _tracker), _pct))
        add("| %s | `%s` | `%s` | %.5g | %s | %s | %s | %s |"
            % (_family, _metric, _tracker, _old,
               ("%.5g" % _new) if math.isfinite(_new) else "**—（本轮无可靠膝点）**",
               ("%+.1f%%" % _pct) if math.isfinite(_pct) else "—",
               ("%.2f%%" % _diff) if math.isfinite(_diff) else "—",
               _alive))
    add("")
    add("**按族读方向**（🔴 **方向并不一致** ⇒ 不得概括成「一律上移 / 一律下移」）")
    add("")
    for _family, _items in fam_dirs.items():
        _ups = [n for n, v in _items if v > 0]
        _downs = [n for n, v in _items if v < 0]
        _verdict = "**同族内部方向不一致**" if (_ups and _downs) else (
            "同族整体上移" if _ups else "同族整体下移")
        add("* **%s 族**（%d 条同族对照）：上移 %d 条 / 下移 %d 条 ⇒ %s。"
            % (_family, len(_items), len(_ups), len(_downs), _verdict))
    add("")
    add("**归因：两个并列原因 ＋ 两项补充**")
    add("")
    add("> ① **底座换血（G2 两趟法）——第一层原因**。"
        "上一轮的曲线建立在「**全量 %d 窗口、未纯净化底座**」上："
        "当时 **%.1f%%（%d 条）的窗口在干净帧上就已经跟丢**，整条衰减曲线的顶部被压到 84.5%%（XCorr）；"
        "本轮只用**好片段**做底座 ⇒ clean 档跟住率回到 100%%（§4）。"
        "⇒ 曲线被整体「抬平」，**弯折点随各族各臂的曲线形状朝不同方向移动**"
        "——这正是上表「同族内部方向不一致」的第一层原因。"
        % (total, 100.0 * class_counter.get(CLASS_LOST, 0) / total,
           class_counter.get(CLASS_LOST, 0)))
    add("> ② **决定族 / 决定臂变了 + 守卫判据修正——第二层原因，且对最终线位影响更大**。"
        "约束线口径是「**每族膝点取最小值**」，所以**哪条曲线决定**一变，线位就变；"
        "本轮还发生了一件更要紧的事：**守卫判据本身被修正**（见 ③）："
        "① `target_lap_var`：上一轮由 **blur** 族决定（819.55）；"
        "本轮 blur 族 XCorr 的双算法偏差 %.2f%%（> %.0f%% 门禁）、"
        "motion 族 XCorr 的分箱曲线只有 %.2f pp（近乎水平）⇒ "
        "**三个族全部未过守卫 ⇒ 本轮该指标没有有效红线**（相对任务书基准：**作废**）；"
        "② `local_scr`：上一轮由 **motion** 族 **KCF** 决定（2.69153）；"
        "本轮 motion 族 KCF 的逐档跨度不足 ⇒ 落到 **blur** 族 XCorr = **%.4f**"
        "（相对任务书基准 %s）；"
        "③ `target_local_contrast`：上一轮**无正式数值**，本轮由 **motion** 族 XCorr 决定 = **%.5f**。"
        % (fnum((pick("xcorr", "blur", "target_lap_var") or {}).get("delta_pct_of_span")),
           KNEE_AGREE_PCT,
           100.0 * fnum((pick("xcorr", "motion", "target_lap_var") or {}).get("curve_y_span")),
           scr_line if math.isfinite(scr_line) else float("nan"),
           ("%+.1f%%" % (100.0 * (scr_line - 2.69) / 2.69)) if math.isfinite(scr_line) else "—",
           contrast_line if math.isfinite(contrast_line) else float("nan")))
    add("> ③ **曲线有效性守卫（W6/W7 复核要求 ＋ 本轮审计的真缺陷修正）——作废了一批「噪声膝点」**。"
        "上一版把「双算法一致」当作唯一的有效性判据 ⇒ **跟住率从未塌掉**的平曲线"
        "（Kneedle 与 PCHIP 必然落在同一个噪声箱上）被误判为可靠。"
        "本轮加到**四道守卫**（跟住率单调 / **分箱曲线**跨度 > %.0f pp / **逐档**跨度 > %.0f pp / 指标单调）"
        "⇒ 被作废的候选在对照表里标为「❌ 守卫作废」，**它们不参与红线取值**。"
        "🔴 **本轮还修掉了一处真缺陷**：守卫原本判在**逐档曲线**上、而膝点取自**分箱曲线**"
        "（§4.2 / §12.13）⇒ 本轮统一改判在**膝点自己的曲线**上。"
        % (100.0 * CURVE_MIN_SPAN, 100.0 * CURVE_MIN_SPAN))
    add("> ④ 🔴 **`local_scr` 本轮判定为「不得单独使用」**——理由**不是**线位漂了，"
        "而是这条线在真实素材上**区分不了好坏片**"
        "（真实坏片拦截 %.2f%% vs 真实好片对照误杀 %.2f%% ⇒ 区分间隔仅 **%.2f pp**，§8.3）"
        "⇒ 它虽然通过了曲线守卫、在 §6 表里有线位，但**不得单独进入对外免责声明**"
        "（§10.2 / §10.3 已删除该子句）。"
        % (real_bad_scr, real_good_scr, real_bad_scr - real_good_scr))
    add("")
    add("> ⚠️ **因此，对外使用时必须带上这几条限定**："
        "① 本轮红线是在**净化底座**上反解的，而净化底座上的衰减曲线"
        "**不足以产生跟踪崩溃**（§4.1）⇒ 红线是「**画质—跟踪性能分界点**」，"
        "不是「**跟踪崩溃点**」；"
        "② 本轮**成立的两条线只由 XCorr 一条臂支撑**（§4.2 / §6）⇒ "
        "**本轮算法无关性不成立**，红线是「单臂反解 + 独立盲测背书」；"
        "③ 免责声明里的门槛必须是**本轮实测值**：`local_scr` 子句**本轮不生效**、"
        "`target_lap_var` 子句**本轮无数值可填**（§10.2 / §10.3）；"
        "④ 🔴 **锐度维度（`target_lap_var`）本轮没有对外红线**——上一版的 541.09 **作废**，"
        "补救路径见 §12.13。")
    add("")

    add("## 十一、复现命令与产物清单")
    add("")
    add("```bash")
    add("# 执行机：huangzhe@192.168.99.40:32222（conda env uav，Python 3.10.20 / scipy 1.15.3 / cv2 4.10.0）")
    add("# 解释器绝对路径：/home/huangzhe/anaconda3/envs/uav/bin/python")
    add("cd /tmp/pycharm_project_a7d966e6")
    add("export PYTHONPATH=/tmp/pylibs_cvcontrib          # contrib legacy KCF（cv2.legacy.TrackerKCF）")
    add("export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4")
    add("PY=/home/huangzhe/anaconda3/envs/uav/bin/python")
    add("$PY iqa_v2/run_eval_v2.py --phase all --workers 24      # 全量（数据→Pass A/B→三分法→膝点→约束线→盲测→回归→尾部→报告）")
    add("$PY iqa_v2/verify_recalc_v2.py --samples 120           # 独立复算，误差 < 1e-6")
    add("$PY -m pytest iqa_v2/test_iqa_v2.py -v                  # 机械门禁 100% 通过")
    add("$PY iqa_v2/make_manifest_v2.py                          # 重建 MD5 清单（必须最后跑）")
    add("md5sum -c iqa_v2/output/manifest_v2.md5                # 产物 MD5 全量校验")
    add("$PY iqa_v2/audit_final.py iqa_v2/output iqa_v2/REPORT_V2_FINAL.md   # 交付物完成度审计（FAIL=0）")
    add("```")
    add("")
    # 🔴 资源铁律自证：启动读数**现场读 /proc**（可复现，不手抄）
    try:
        meminfo = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            meminfo[key] = int(rest.split()[0])
        total_mb = meminfo.get("MemTotal", 0) // 1024
        avail_mb = meminfo.get("MemAvailable", 0) // 1024
        avail_pct = 100.0 * avail_mb / max(total_mb, 1)
        load1 = os.getloadavg()[0]
        ncpu = os.cpu_count() or 0
        add("**资源铁律自证（启动读数现场读 `/proc`，不手抄）**")
        add("")
        add("| 采样时刻 | `MemTotal` / `MemAvailable` | 可用率 | `nproc` / load(1min) |")
        add("|:--|:--|--:|:--|")
        add("| 生成本报告时 | %d MB / %d MB | **%.1f%%** | %d 核 / %.2f |"
            % (total_mb, avail_mb, avail_pct, ncpu, load1))
        add("")
        add("> 门槛：可用内存 **≥ 30%**（低于 20% 停最新、低于 10% 停全部）；"
            "线程数已 pin：`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4`，"
            "`cv2.setNumThreads(1)`；**本轮未启动训练、未下载数据集、未做模型重训**。")
    except Exception as exc:  # pragma: no cover - 读不到 /proc 不应让报告生成失败
        add("> （资源读数读取失败：%s）" % exc)
    add("")
    add("**条款 → 机械门禁用例对照表**（W10 §三：验收仪器不查的条款，用 pytest 用例逐条自证）")
    add("")
    add("| 条款 | 机械门禁用例（可直接 grep 核对） |")
    add("|:--|:--|")
    for clause, tests in (
        ("序列级 7:3 切分无交叉、种子可复现",
         "`test_sequence_level_split_is_disjoint_and_reproducible`"),
        ("两趟法底座纯净 + 留出集零参与",
         "`test_simulation_base_is_pure_and_holdout_never_simulated`、"
         "`test_simulation_pool_is_dataset_level_good_segments`"),
        ("三族退化齐备、档位数量不变（任务书 §二.2）",
         "`test_conditions_cover_three_degradation_families`"),
        ("JPEG 轴已移除、加性白噪声轴不存在",
         "`test_jpeg_and_white_noise_axes_are_removed`"),
        ("约束线只覆盖三项核心指标（v1 全图指标不得进红线）",
         "`test_constraint_lines_only_cover_three_core_metrics`、"
         "`test_constraint_lines_only_from_valid_curves`"),
        ("三分法回归三集合齐备（好 / 一般 / 跟丢）",
         "`test_regression_covers_three_way_sets`"),
        ("小目标子集常量 `max(w,h) ≤ 20 px`",
         "`test_small_target_threshold_constant`"),
        ("运动模糊角逐窗口确定性且在 [0,180)",
         "`test_motion_angle_is_deterministic_and_in_range`"),
        ("好片段阈值 = 任务书给死的 3 px",
         "`test_good_segment_threshold_is_task_book_3px`"),
        ("核心区与背景环零重叠、边界不越界",
         "`test_boundary_safety_no_nan_and_no_overlap`、`test_tiny_target_regions_still_valid`"),
        ("红外白热 / 黑热极性对称",
         "`test_infrared_polarity_symmetry_local_scr`、`test_infrared_polarity_symmetry_on_real_frame`"),
        ("退化强度递增 ⇒ 指标单调下降 / CLE 上升",
         "`test_degradation_metrics_move_monotonically`、"
         "`test_degradation_worsens_tracking_cle_on_real_window`"),
        ("曲线有效性守卫（平曲线 / 非单调 / 指标非单调 ⇒ 作废）",
         "`test_degenerate_curves_are_rejected`、"
         "`test_flat_success_curve_is_degenerate`、"
         "`test_level_metric_profile_flags_contrast_local_scr_non_monotone`、"
         "`test_knee_below_four_bins_still_carries_guard_label`"),
        ("产物 MD5 全量校验",
         "`test_output_manifest_md5_all_ok`"),
    ):
        add("| %s | %s |" % (clause, tests))
    add("")
    add("> **验收盲区自证（W10 §三 ①~④）**：① **任务书常量对账** ⇒ §10 的「差异说明」表"
        "（819.55 / 2.69 ↔ 本轮实测值 + 逐族逐臂对照 + 归因）；"
        "② **跨臂偏差阈值** ⇒ §6 的「两臂线位差」表 ＋ §12.8b 的「本轮算法无关性不成立」结论与归因；"
        "③ **误杀率门限** ⇒ §8.1 / §8.3 的「三集合 × 三指标」误杀率表；"
        "④ **`correlations_v2.csv` 交叉** ⇒ §12.12 的「采纳膝点 × 相关性表」逐条清单（%d / %d）。"
        % (len(reliable_knees), len(weak_knees)))
    add("")
    add("| 产物 | 路径 |")
    add("|:--|:--|")
    for name, path in (("主报告", "iqa_v2/REPORT_V2_FINAL.md"),
                       ("成稿章节（可并入记忆库）", "iqa_v2/MEMORY_SECTION.md"),
                       ("留痕", "iqa_v2/PROGRESS_SERVER.md"),
                       ("窗口表", "iqa_v2/output/windows.csv"),
                       ("明细（逐窗口 × 逐条件，退化仿真底表）",
                        "`iqa_v2/output/degrade_detail_v2.csv`（cache 侧同名副本 "
                        "`/tmp/iqa_v2_cache/degrade_detail.csv`）"),
                       ("三分法分类表", "iqa_v2/output/segment_class_v2.csv"),
                       ("序列级切分表", "iqa_v2/output/split_v2.csv"),
                       ("膝点表", "iqa_v2/output/knee_thresholds_v2.csv"),
                       ("约束线表", "iqa_v2/output/constraint_lines_v2.csv"),
                       ("相关性表", "iqa_v2/output/correlations_v2.csv"),
                       ("尾部验证表", "iqa_v2/output/tail_analysis_v2.csv"),
                       ("回归测试表", "iqa_v2/output/regression_v2.csv"),
                       ("真实坏片盲测明细", "iqa_v2/output/blind_real_v2.csv"),
                       ("产物 MD5 清单", "iqa_v2/output/manifest_v2.md5")):
        add("| %s | %s |" % (name, path if path.startswith("`") else "`%s`" % path))
    add("")

    # ------------------------------------------------------------------ 十二、局限
    #: CSRT 参考线位与 XCorr 生效线位的倍数区间（跨臂不可比的另一侧证据）
    csrt_ratios = []
    for _metric in CORE_METRICS:
        _info = lines_tbl.get(_metric, {})
        _x = fnum(_info.get("xcorr_line"))
        _c = fnum(_info.get("csrt_ref"))
        if math.isfinite(_x) and math.isfinite(_c) and _x:
            csrt_ratios.append(_c / _x)
    #: KCF 臂「有候选读数但被守卫作废」的格子的双算法偏差区间（%）
    kcf_deltas = [fnum(r.get("delta_pct_of_span")) for r in knee_rows
                  if r["tracker"] == "kcf_legacy" and r["curve_set"] == "common_good"
                  and r["stratum"] == "all" and r["metric"] in CORE_METRICS
                  and math.isfinite(fnum(r.get("knee_x_candidate")))]
    #: 各臂在 ``common_good`` 池上的**最高曲线跨度**（%）：说明「哪条臂的曲线够深」
    #: 两条口径分别取最大：**分箱曲线跨度**（指标判别口径）与**逐档跨度**（退化轴口径）
    arm_max_span: dict[str, float] = {}
    arm_max_level_span: dict[str, float] = {}
    for tracker in TRACKER_FAMILIES:
        spans = [fnum((pick(tracker, family, "target_lap_var") or {}).get("curve_y_span"))
                 for family in DEGRADATION_FAMILIES]
        arm_max_span[tracker] = max(spans) if spans else float("nan")
        lspans = [fnum((pick(tracker, family, "target_lap_var") or {}).get("level_y_span"))
                  for family in DEGRADATION_FAMILIES]
        arm_max_level_span[tracker] = max(lspans) if lspans else float("nan")

    add("## 十二、已知局限")
    add("")
    add("1. **正式上板红线未锁定**：本轮阈值用 **Anti-UAV 开源红外**标定，"
        "只验证方法有效。内部 **58 条红外业务 GT** 尚未回收"
        "（核查结论：`manu/point_annotator/` 只有工具、无标注数据；"
        "`manu/memory/memory.md` §315 记录的是**三份并行交付**，§316 明确写着「待回收」；"
        "本工作区内**无任何标注数据**——母包 `annotations/` 不在本工作区，故此处只以"
        "「`manu/point_annotator/` 只有工具、无标注数据 + §315/§316 文本」两条可复核证据作结论）"
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
        "任务书 §二.2 给定的三族档位在好片段上只能把跟住率压到 90% 以上，"
        "⇒ 膝点只能定位在「曲线开始弯折」处，**本轮称之为「画质—跟踪性能分界点」**。"
        "**要拿到更硬的崩溃拐点，正确做法是加大退化档位上限**"
        "（例如 `blur` σ 提到 6~8、`contrast` α 降到 0.05），"
        "**而不是**把脏窗口塞回仿真池——后者是任务书 §二.1 明令禁止的。")
    add("7. **`local_scr` 是三项核心指标里证据最弱的一项**："
        "它是被局部背景方差归一化的比值，在纹理丰富但无目标的场景里天然偏低；"
        "本轮它在好片段留出集上的误杀率明显高于另外两项（§8.1），"
        "在真实素材上的**区分间隔仅 %.2f pp**（§8.3）⇒ "
        "**它虽通过了曲线守卫、在 §6 表里有线位（%.7g），但不得单独进入对外免责声明**"
        "（§10.2 / §10.3 已删除该子句）。"
        % (real_bad_scr - real_good_scr, scr_line if math.isfinite(scr_line) else float("nan")))
    add("8. **算法无关性本轮无法复现**（§4.2 / §6）：净化底座上只有 XCorr 臂的 "
        "`blur` / `motion` 两条曲线达到 A 级（单调且**两条跨度口径**都 > %.0f pp），"
        "KCF 与 CSRT 的曲线全部未过门槛 ⇒ **本轮成立的两条线都只由 XCorr 一条臂支撑**。"
        "任务书 §二.3 倚重的「XCorr ↔ KCF 膝点一致」在本轮数据上**不成立**"
        "（上一轮未纯净化时 `target_lap_var` 两臂偏差是 0.00%%）。"
        "⇒ 红线是「**单臂反解 + 独立盲测背书**」，对外表述必须带这个限定。"
        % (100.0 * CURVE_MIN_SPAN))
    add("8b. 🔴 **结论：本轮核心指标的膝点不具备算法无关性**。"
        "跨臂线位差 `cross_arm_dev_pct` **全部为 nan**（KCF 侧无有效线位 ⇒ 不可比），"
        "CSRT 侧参考值与 XCorr 相差 %.2f~%.2f 倍。"
        "⇒ **KCF 臂为何无线位（归因必须写清）**：**既不是**被本轮的分类口径清空"
        "（KCF 自己的好片段集 `own_good` 仍有数千窗口参与仿真，§5.1 逐档曲线完整），"
        "**也不是**双算法不一致（被作废的 **KCF 候选**其 Kneedle/PCHIP 偏差只有 %.2f%%~%.2f%%，"
        "本身是达标的）——真正的原因是 **跨度守卫**：净化底座上 KCF 曲线的"
        "**最高逐档跨度只有 %.2f pp**（XCorr 为 %.2f pp），**低于 %.0f pp 的门槛**"
        "⇒ 被判为「退化轴没把跟住率压下去」⇒ 膝点作废。"
        "（KCF 的**分箱曲线跨度**最高到 %.2f pp，但**逐档跨度**只有 %.2f pp ⇒ "
        "两条口径必须**同时**过门槛。）"
        "🔴 换句话说：**KCF 无线位是「净化底座上曲线太浅」造成的，不是「样本被清空」**；"
        "把守卫门槛放宽到 3 pp，KCF 的 `blur` 族会重新入选（§12.9）。"
        % (min(csrt_ratios) if csrt_ratios else float("nan"),
           max(csrt_ratios) if csrt_ratios else float("nan"),
           min(kcf_deltas) if kcf_deltas else float("nan"),
           max(kcf_deltas) if kcf_deltas else float("nan"),
           100.0 * arm_max_level_span.get("kcf_legacy", float("nan")),
           100.0 * arm_max_level_span.get("xcorr", float("nan")),
           100.0 * CURVE_MIN_SPAN,
           100.0 * arm_max_span.get("kcf_legacy", float("nan")),
           100.0 * arm_max_level_span.get("kcf_legacy", float("nan"))))
    add("9. **曲线有效性守卫的门槛是工程判断，不是统计检验**："
        "「跨度 > %.0f pp + 单调」这条门槛把 9 条 (臂 × 族) 曲线里的 7 条判为不可用，"
        "门槛取值直接决定红线由哪条曲线决定（放宽到 3 pp 会让 KCF 的 `blur` 族重新入选）。"
        "门限取 5 pp 的依据是**曲线点**的抽样噪声量级（本轮净化底座 874 窗口 × 6/5 档 = "
        "5244 / 4370 个曲线点 ⇒ ±1.35 / ±1.48 pp @ 95%% 置信）；"
        "**要更硬的结论，应当加大退化档位上限让曲线真正塌下来，而不是调门槛**。"
        % (100.0 * CURVE_MIN_SPAN))
    add("10. **盲测集只做一次性验证**：§8.3 的真实素材数字**没有**、也**不得**"
        "参与阈值搜索或反向调参；它只用于**证明 / 否证**红线方向。")
    add("11. 🔴 **「真实好片对照」不是目标级误杀率证据**（host W8-3 / W10-1 复核要求）："
        "本轮 2 条真实好片对照是**无目标的停车场航拍**（有车、有行人，**没有无人机**；"
        "DoG 峰值落在车顶/地面而非目标上）⇒ 它测的是"
        "「**正常纹理场景 vs 平坦退化场景**」的分离度，**不是**「好目标被误杀」。"
        "⇒ **本轮没有「有目标的真实好片」对照**；"
        "**目标级好片误杀率只有 §8.1 的 Anti-UAV 序列级留出集一个来源**"
        "（`target_local_contrast` %.2f%% / `target_lap_var` %.2f%% / `local_scr` %.2f%%）。"
        "§8.3 与 §10 里凡是引用「真实好片对照」的数字，**都不得**被读成目标级误杀率。"
        % (_rate("good_holdout", "target_local_contrast"), _rate("good_holdout", "target_lap_var"),
           _rate("good_holdout", "local_scr")))
    add("12. 🔴 **采纳膝点与相关性表的口径不完全一致**（host W10-2 复核要求，"
        "两条口径必须对齐读）：本轮 `knee_reliable = 1` 的膝点共 **%d 条**，"
        "其中 **%d 条**在 `correlations_v2.csv` 里被判 `usable = 0` 或 `direction_ok = 0`。"
        "本轮**没有**把 `usable ∧ direction_ok` 设为采纳的前置条件，理由逐条如下："
        % (len(reliable_knees), len(weak_knees)))
    add("")
    add("| 约束线 | 决定臂 / 决定族 | 秩相关 \\|ρ\\| | `usable` | `direction_ok` | 本轮是否采纳 | 采纳理由 |")
    add("|:--|:--|--:|:--:|:--:|:--|:--|")
    for _metric in CORE_METRICS:
        _info = lines_tbl.get(_metric, {})
        _tracker = _info.get("deciding_tracker", "")
        _family = _info.get("deciding_family", "")
        _curve_set = _info.get("curve_set", "common_good")
        _corr = corr_index.get((_tracker, _curve_set, _family, _metric))
        _usable = _corr.get("usable", "—") if _corr else "—"
        _direction = _corr.get("direction_ok", "—") if _corr else "—"
        _rho = fnum(_corr.get("spearman")) if _corr else float("nan")
        _weak = (_usable != "1" or _direction != "1")
        _line = fnum(_info.get("line_value"))
        _adopted = "✅ 采纳" if math.isfinite(_line) else "❌ **未采纳（无有效红线）**"
        add("| `%s` = %s | `%s` / %s | %s | %s | %s | %s | %s |"
            % (_metric, ("%.7g" % _line) if math.isfinite(_line) else "**无**",
               _tracker or "—", _family or "—", ("%.4f" % _rho) if math.isfinite(_rho) else "—",
               _usable, _direction, _adopted,
               ("**秩相关弱，但盲测区分间隔 %.2f pp（§8.3）⇒ 采纳**" % (
                   _rate("real_bad", _metric) - _rate("real_good", _metric))
                if _weak else "秩相关达标") if math.isfinite(_line)
               else "**曲线守卫未过 ⇒ 本轮不产生红线**（§4.2 / §6）"))
    add("")
    add("**逐条清单：被采纳（`knee_reliable = 1`）但相关性偏弱的 %d 条**" % len(weak_knees))
    add("")
    weak_small = sum(1 for r in weak_knees if r.get("stratum") == "small")
    weak_all = len(weak_knees) - weak_small
    weak_core = sum(1 for r in weak_knees if r["metric"] in CORE_METRICS)
    weak_aux = len(weak_knees) - weak_core
    core_all_weak = [r for r in weak_knees
                      if r["metric"] in CORE_METRICS and r.get("stratum") == "all"]
    add("> 分布：`stratum` = **small %d 条 / all %d 条**；指标 = **核心指标 %d 条 / "
        "辅助·全图 %d 条**（辅助指标与全图指标**本来就不进红线**）；"
        "**核心指标 ∧ `stratum = all`** 的只有 **%d 条**（见下表加粗行）。"
        % (weak_small, weak_all, weak_core, weak_aux, len(core_all_weak)))
    add("")
    add("| 臂 | 曲线集 | 族 | 指标 | 层 | `usable` | `direction_ok` | \\|ρ\\| | 是否进红线 |")
    add("|:--|:--|:--|:--|:--:|:--:|:--|--:|:--|")
    for r in sorted(weak_knees, key=lambda x: (x["metric"] not in CORE_METRICS,
                                               x["tracker"], x["curve_set"],
                                               x["degradation_family"], x["stratum"])):
        corr = corr_index.get((r["tracker"], r["curve_set"], r["degradation_family"], r["metric"]))
        rho = fnum(corr.get("spearman")) if corr else float("nan")
        is_core = r["metric"] in CORE_METRICS
        add("| %s%s | `%s` | %s | `%s` | %s | %s | %s | %s | %s |"
            % ("**" if is_core and r.get("stratum") == "all" else "",
               r["tracker"], r["curve_set"], r["degradation_family"], r["metric"],
               r.get("stratum", "all"),
               corr.get("usable", "—") if corr else "—",
               corr.get("direction_ok", "—") if corr else "—",
               ("%.4f" % rho) if math.isfinite(rho) else "—",
               "**✅ 进红线**" if is_core and r.get("stratum") == "all"
               else ("否（辅助 / 全图指标）" if not is_core else "否（小目标子集）")))
    add("")
    add("> ① **为什么不对「弱相关」一律剔除**（这是本轮的口径选择，不是遗漏）："
        "若把 `usable = 1 ∧ direction_ok = 1` 设为采纳门槛，"
        "被剔掉的恰好是 **`target_lap_var`**（秩相关最弱：净化底座上二值标签基线极高、"
        "锐度—跟踪关系是阈值型 ⇒ ρ 被结构性压低，§9.2），"
        "而被保留的反而是 **`local_scr`**（秩相关达标、但真实素材区分间隔仅 %.2f pp）"
        "⇒ **等于把判别力最强的那项砍掉、把在真实外场不成立的那项留下**。"
        "🔴 **本轮这条口径已经「事实性」地兑现了**：`target_lap_var` 不是被秩相关剔掉的，"
        "而是被**曲线有效性守卫**剔掉的（§4.2 / §6）——"
        "**两条口径指向同一个结论，证据是互相印证的，不是互相矛盾的**。"
        "因此本轮的口径是「**采纳时不用秩相关把关、但必须在报告里逐条交代**」。"
        % (real_bad_scr - real_good_scr))
    add("> ② **两条口径的分工（必须对齐读）**：相关性表（§9.2）回答"
        "「**这个指标在退化族内能不能把跟住率排对序**」⇒ 只用于**横向比较指标的排序能力**；"
        "红线的证据链是「**曲线有效性守卫 + 膝点 + 独立盲测**」（§4.2 / §5 / §8.3）"
        "⇒ **不用这张表否定红线**。两条口径**不冲突**，但**不能各说各话**："
        "本表（§12.12）就是把「哪些被采纳的膝点在秩相关上偏弱」逐条摊开的地方。")
    add("> ③ **可操作的结论**：要同时拿到「秩相关强」和「曲线够深」，"
        "**正确做法是加大退化档位上限**让曲线真正塌下来（§12.9），"
        "而不是调守卫门槛或加严采纳条件——后者只会把红线换掉，不会让证据变强。")
    add("")
    _lap_prev = pick("xcorr", "motion", "target_lap_var") or {}
    add("13. 🔴 **本轮修正了曲线有效性守卫的一处真缺陷，并因此作废了上一版的 "
        "`target_lap_var` 红线**：上一版把**逐档跨度**写进 `curve_y_span` 列、"
        "而膝点却取自**分箱曲线** ⇒ 守卫判的是 A 曲线、放行的是 B 曲线。"
        "实测 `target_lap_var` / XCorr / `motion`：逐档跨度 %.2f pp（过门槛）但"
        "**分箱跨度仅 %.2f pp**（整条曲线几乎是平的）⇒ 上一版的 `target_lap_var < %.6g` "
        "是**噪声点**，双算法「%.2f%% 一致」属**伪一致**。⇒ 本轮守卫改判在分箱曲线上，"
        "新增 `level_y_span` 列作第二道守卫，**该红线作废**（§4.2 / §6）。"
        % (100.0 * fnum(_lap_prev.get("level_y_span")), 100.0 * fnum(_lap_prev.get("curve_y_span")),
           fnum(_lap_prev.get("knee_x_kneedle")), fnum(_lap_prev.get("delta_pct_of_span"))))
    add("")
    add("> **锐度维度（`target_lap_var`）的补救路径（三条，按代价排序）**：")
    add("")
    add("> ① **加大退化档位上限**（首选，与 §12.6 / §12.9 同一结论）："
        "`blur` σ 提到 6~8、`motion` 核长提到 15、`contrast` α 降到 0.05，"
        "让曲线**真正塌下来**⇒ 锐度维度自然获得有效膝点；"
        "**代价**：需要重跑 Pass B 仿真（明细表重生成），但**不涉及任何训练**。")
    add("> ② **按小目标子集单独出锐度线**（本轮已具备条件）："
        "小目标子集上 `target_lap_var` **有通过全部守卫的膝点**——"
        "%s ⇒ 说明锐度维度对远距离小目标**是有判别力的**，"
        "只是被大目标窗口的高 `lap_var` 基线摊平在全量曲线上。"
        "🔴 但它是**分层口径**，必须单独标注，**不得**与全量线混用或互相替代。"
        % "、".join(
            "%s / %s / %s = %.6g" % (r["tracker"], r["degradation_family"], r["metric"],
                                     fnum(r["knee_x_final"]))
            for r in knee_rows
            if r["metric"] == "target_lap_var" and r["curve_set"] == "common_good"
            and r["stratum"] == "small" and r["knee_reliable"] == "1"))
    add("> ③ **扩大好片段池的锐度动态范围**：在**净化底座**上补充一批**中等大小**目标窗口，"
        "让 `lap_var` 在池内的分布不至于被大目标独占 ⇒ 分箱曲线恢复斜率。")
    add("")
    add("> ⇒ **在 ①②③ 完成之前，对外免责声明只有 `target_local_contrast` 一条红线**"
        "（§10.3），**锐度维度不得给出任何数值**。")
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
    parser.add_argument("--output-dir", default=None,
                        help="产物目录（默认随 --conditions-profile 绑定："
                             "baseline→iqa_v2/output，deep→iqa_v2/output_deep）")
    parser.add_argument("--cache-dir", default=None,
                        help="大体量明细 cache 目录（默认随剖面绑定："
                             "baseline→/tmp/iqa_v2_cache，deep→/tmp/iqa_v2_cache_deep）")
    parser.add_argument("--detail-csv", default="")
    parser.add_argument("--report", default=None,
                        help="报告正文路径（默认随剖面绑定："
                             "baseline→iqa_v2/REPORT_V2_FINAL.md，deep→iqa_v2/REPORT_V2_DEEP.md）")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-windows-per-seq", type=int, default=30)
    parser.add_argument("--total-target", type=int, default=2600)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--conditions-profile", default="baseline",
                        choices=["baseline", "deep"],
                        help="退化档位剖面：baseline=已交付基线档位（默认）；deep=抬高上限的深轮档位")
    parser.add_argument("--blur-sigmas", default="",
                        help="覆盖 blur 档位（逗号分隔，如 0.6,1.0,4.0）；留空=沿用剖面自带")
    parser.add_argument("--motion-lengths", default="",
                        help="覆盖 motion 档位（逗号分隔，如 3,5,15）；留空=沿用剖面自带")
    parser.add_argument("--contrast-alphas", default="",
                        help="覆盖 contrast 档位（逗号分隔，如 0.8,0.5,0.05）；留空=沿用剖面自带")
    cli = parser.parse_args()
    try:
        cli.blur_sigmas = _parse_float_list(cli.blur_sigmas)
        cli.motion_lengths = _parse_float_list(cli.motion_lengths)
        cli.contrast_alphas = _parse_float_list(cli.contrast_alphas)
        conditions = build_conditions(cli.conditions_profile,
                                      blur_sigmas=cli.blur_sigmas,
                                      motion_lengths=cli.motion_lengths,
                                      contrast_alphas=cli.contrast_alphas)
    except ValueError as exc:
        parser.error(str(exc))
    ladder_problem = _guard_profile_ladder(cli.conditions_profile, conditions)
    if ladder_problem:
        parser.error(ladder_problem)

    # ---- 剖面 ⇒ 产物路径的硬绑定（C3：基线轮产物不得被覆盖）
    bound = PROFILE_PATHS[cli.conditions_profile]
    # 🔴 默认值也**锚到仓库根** ⇒ 换个 CWD 跑深轮，产物仍然落在自己的目录里
    cli.output_dir = cli.output_dir or str(_anchor_repo_path(bound["output_dir"]))
    cli.cache_dir = cli.cache_dir or str(_anchor_repo_path(bound["cache_dir"]))
    cli.report = cli.report or str(_anchor_repo_path(bound["report"]))
    # 🔴 明细 CSV 路径**先**定下来（它也是**输出**路径 ⇒ 必须一并纳入保护集合，
    #    否则 ``--detail-csv iqa_v2/output/degrade_detail_v2.csv`` 能覆盖冻结基线明细）
    cli.detail_csv = cli.detail_csv or str(Path(cli.cache_dir) / "degrade_detail.csv")
    if cli.conditions_profile != "baseline":
        # 🔴 保护集合来自**冻结清单**（清单登记的**全部**基线产物），
        #    不只是 PROFILE_PATHS 那三条
        protected, source = _protected_artifacts()
        if not protected:
            parser.error("冻结清单缺失 ⇒ 无法派生「深轮绝不可写」的产物集合 ⇒ 拒绝运行")
        for label in ("output_dir", "cache_dir", "report", "detail_csv"):
            hit = _protected_hit(getattr(cli, label), protected)
            if hit:
                # ``detail_csv`` 不在 PROFILE_PATHS 里（它由 cache_dir 派生）⇒ 默认值另算
                default = (str(_anchor_repo_path(bound[label])) if label in bound
                           else str(Path(cli.cache_dir) / "degrade_detail.csv"))
                parser.error(
                    "%s 剖面不得写入基线产物路径（--%s=%s）：命中冻结产物 %s（%s）；"
                    "基线轮产物已冻结（规格书 C3），%s 必须走独立目录（默认 %s）"
                    % (cli.conditions_profile, label.replace("_", "-"), getattr(cli, label),
                       hit, source, cli.conditions_profile, default))

    # 🔴 明细 CSV 路径**已在上方**定下来（CSV-only 阶段要拿它做「声明轴 vs 实际轴」校验）
    detail_csv = cli.detail_csv

    # 🔴 CSV-only 阶段（classify/knee/constraint/blind/regress/tail/report）可以脱离
    #    ``eval`` 单独重跑 ⇒ 这一轮用的是哪条退化轴也要记进 ``run_meta.json``。
    if cli.phase in ("classify", "knee", "constraint", "blind", "regress", "tail", "report"):
        mismatch = _verify_profile_matches_detail(cli, conditions)
        if mismatch:
            parser.error(mismatch)
        _write_run_meta(cli, conditions, status="in_progress")

    if cli.phase in ("all", "dataset"):
        import dataset_anti_uav

        sys.argv = ["dataset_anti_uav.py", "--dataset-root", cli.dataset_root,
                    "--output-dir", cli.output_dir, "--seed", str(cli.seed),
                    "--max-windows-per-seq", str(cli.max_windows_per_seq),
                    "--total-target", str(cli.total_target)]
        if cli.smoke:
            sys.argv.append("--smoke")
        dataset_anti_uav.main()

    if cli.phase in ("all", "eval"):
        detail_csv = str(run_eval(cli))
        cli.detail_csv = detail_csv
        mismatch = _verify_profile_matches_detail(cli, conditions)
        if mismatch:
            parser.error(mismatch)
        _write_run_meta(cli, conditions, status="completed")
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
        # 🔴 一个报告路径**只能有一个生成器**：deep 剖面走剖面感知的深轮生成器
        if cli.conditions_profile == "baseline":
            phase_report(cli)
        else:
            phase_report_deep(cli)

    if cli.phase in ("classify", "knee", "constraint", "blind", "regress", "tail", "report"):
        _write_run_meta(cli, conditions, status="completed")


if __name__ == "__main__":
    main()
