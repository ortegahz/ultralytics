#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三分法回归验证体系（任务书 §三 + §四.1）——G3 序列级切分 + G4 回归测试。

本模块是**「好 / 一般 / 跟丢」三分法**与**序列级隔离切分**的唯一 owner：
``run_eval_v2.py`` 只在需要时惰性导入它，避免与主流程产生循环依赖。

三分法判定（任务书 §三原文口径）
--------------------------------
============ ========================================================== ====================
类别          客观标准                                                    用途
============ ========================================================== ====================
好片段        5 帧全程稳定跟住，且末帧 ``CLE_{t+4} ≤ 3 px``（任务书给死）  70% 标定 / 30% 留出
一般片段      未完全跟丢，但 ``3 px < CLE_{t+4} ≤ 容限``，或双跟踪器分歧    临界敏感性（不参与定标）
跟丢片段      原始画面下真实跟踪失锁（``CLE_{t+4} > 容限``）                盲测拦截验证集
============ ========================================================== ====================

🔴 **「好片段」的 3 px 是任务书给死的常量**（``GOOD_CLE_PX``），不得由本实现自行调整。

双臂合成规则（跨家族纪律：两臂**并列判定、不合并求平均**）
--------------------------------------------------------
* **每臂各自分类**（``class_<arm>``）：``CLE ≤ 3 px`` 且 5 帧稳定 ⇒ 好；
  ``CLE ≤ 容限`` ⇒ 一般；否则（硬失败或超容限）⇒ 跟丢。
* **数据集级三类**（``class_ds``，用两条主臂 XCorr / KCF 合成）：

  ============================== ==========================================
  条件                            数据集级类别
  ============================== ==========================================
  两臂都「好」                    好片段
  两臂都「跟丢」                  跟丢片段
  一臂好、一臂跟丢（**分歧**）      一般片段（任务书 §三「临界窗口」）
  一臂跟丢、另一臂一般             跟丢片段
  其余（两臂都一般）               一般片段
  ============================== ==========================================

序列级 7:3 切分（G3）
--------------------
按 ``seq_id`` **整条**切：70% 标定集 / 30% 留出集，切分种子固定并写入产物。
膝点反解**只用标定集**；留出集**零参与阈值搜索**（连退化仿真都不做）。

真实坏片盲测（G4）
------------------
坏组红外源片**没有标注 GT**，沿用 v1（``manu/video_quality/``）口径：
* **故障窗** = 视频**最后 5 秒**（``vq_tail.clip_slices`` 同款：``round(fps × 5)`` 帧）；
* **坏组黑边（pillarbox）检测裁边** = 复用 v1 的 ``vq_metrics.detect_content_box``
  （逐行/逐列**最大值** ≤ 16 判黑边，首尾连续黑边段，剩余内容 < 25% 则不裁），
  🔴 **不裁边会把坏片「测成」清晰**（黑边硬边缘抬高拉普拉斯能量）；
* 坏片**同样没有目标框** ⇒ ROI 由 **DoG 小目标响应自动选取**（每帧取最显著处），
  这对坏片是**最有利**的假设（若有目标必被选中）⇒ 拦截结论偏保守。
* 🔴 坏片只有 2 条 ⇒ 所有基于真实坏片的结论**只能标注为「小样本级」**。

执行机：服务器 ``huangzhe@192.168.99.40:32222``（conda env ``uav``，依赖 cv2 + scipy）。
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_eval_v2 import (ALL_METRICS, TRACKER_FAMILIES,  # noqa: E402
                         roc_auc)
from target_roi_metrics import (GLOBAL_METRICS, LOCAL_METRICS,  # noqa: E402
                                compute_roi_metrics)

#: v1 的黑边检测算子住在 ``manu/video_quality/``（5 个调用点共用）⇒ 复用，不重写。
VQ_DIR = REPO / "manu" / "video_quality"
if str(VQ_DIR) not in sys.path:
    sys.path.insert(0, str(VQ_DIR))

#: 🔴 任务书 §三 给死的「好片段」末帧 CLE 上限（px）——**不得由实现自行调整**。
GOOD_CLE_PX = 3.0
#: 序列级切分比例与种子（G3：70% 标定 / 30% 留出，固定种子 ⇒ 可复现）
CALIB_RATIO = 0.7
SPLIT_SEED = 20261126
#: 两条**跨家族主臂**（数据集级三分法只用这两条判定）
PRIMARY_ARMS = ("xcorr", "kcf_legacy")

#: 对外只保留的三项 v2 核心指标（G5：约束线表只出这三项）
CORE_METRICS = ("target_local_contrast", "target_lap_var", "local_scr")
#: **膝点邻域**的相对半宽（§8.2「一般片段的指标分布是否落在膝点邻域」的判定带 =
#: 膝点值 x [1/1.5, 1.5]）。🔴 不能用「跟住率降到 0.5 的 X」当邻域端点——本轮净化底座上
#: 所有曲线的跟住率都**从不跌破 0.5**，那个量恒为 nan。
KNEE_BAND_FACTOR = 1.5
#: 辅助指标（进「辅助指标」表，**不进红线**）
AUX_METRICS = ("target_grad_mean", "surround_std", "surround_edge_density", "surround_blockiness")

CLASS_GOOD = "good"
CLASS_MEDIOCRE = "mediocre"
CLASS_LOST = "lost"

#: 真实坏片 / 好片对照（**只做红外**；可见光 OSD 不在本任务范围）
REAL_BAD_CLIPS = (
    ("120BT-04-红外02", "/mnt/data/xxxu/纵横挂飞/120BT-04-红外02.mp4"),
    ("130CT-02红外_IRT", "/mnt/data/xxxu/纵横挂飞/130CT-02红外_IRT.mp4"),
)
REAL_GOOD_CLIPS = (
    ("VIDEO00007", "/mnt/data/xxxu/朗基天香20260903/VIDEO00007_19700101_012038IR.h264"),
    ("VIDEO00008", "/mnt/data/xxxu/朗基天香20260903/VIDEO00008_19700101_012152IR.h264"),
)
#: 故障窗长度（秒）——v1 ``vq_tail.clip_slices`` 同款口径
FAULT_SECONDS = 5.0


# --------------------------------------------------------------------------- 基础判定
def _to_float(value) -> float:
    """宽松数值转换（空串 / 非法值 ⇒ NaN）。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def tolerance_for(target_w: float, target_h: float) -> float:
    """「跟住」容限（任务书口径）：``max(5 px, 0.5 × max(w, h))``。"""
    return max(5.0, 0.5 * max(float(target_w), float(target_h)))


def classify_arm(cle_end: float, maxcle: float, complete: bool, tolerance: float,
                 success: bool = True) -> str:
    """**单臂**分类：好 / 一般 / 跟丢（任务书 §三）。

    * ``complete`` 为假 ⇒ 跟踪器**中途失锁** ⇒ 直接「跟丢」（哪怕末帧位置碰巧靠近 GT）；
    * ``cle_end ≤ 3 px``（任务书给死）**且** 5 帧内无超容限颤动 ⇒ 「好片段」；
    * ``cle_end ≤ 容限`` ⇒ 「一般片段」（颤动 / 漂移 / 临界）；
    * 否则 ⇒ 「跟丢片段」。

    🔴 ``success`` = 任务书口径的「跟住」标志（``CLE_{t+4} ≤ 容限``）：**好片段必须
    在干净帧上真的跟住了**——否则这个窗口会以 ``success=0`` 的身份进退化池，
    把衰减曲线的**起点**拖下来（这正是 G2 要消灭的污染）。
    """
    if not complete or cle_end is None or not math.isfinite(float(cle_end)):
        return CLASS_LOST                     # 跟踪器硬失败 / 中途失锁 ⇒ 真实失锁
    if success and float(cle_end) <= GOOD_CLE_PX and math.isfinite(float(maxcle)) \
            and float(maxcle) <= tolerance:
        return CLASS_GOOD
    if float(cle_end) <= tolerance:
        return CLASS_MEDIOCRE
    return CLASS_LOST


def classify_dataset(arm_classes: dict[str, str]) -> str:
    """**数据集级**三分法（两条主臂合成，见模块 docstring 的真值表）。"""
    primary = [arm_classes.get(arm, CLASS_LOST) for arm in PRIMARY_ARMS]
    if all(cls == CLASS_GOOD for cls in primary):
        return CLASS_GOOD
    if all(cls == CLASS_LOST for cls in primary):
        return CLASS_LOST
    if CLASS_GOOD in primary and CLASS_LOST in primary:
        return CLASS_MEDIOCRE                # 双跟踪器分歧 ⇒ 临界窗口
    if CLASS_LOST in primary:
        return CLASS_LOST
    return CLASS_MEDIOCRE


# --------------------------------------------------------------------------- G3 序列级切分
def split_sequences(seq_ids: list[str], ratio: float = CALIB_RATIO,
                    seed: int = SPLIT_SEED) -> dict[str, str]:
    """按 ``seq_id`` **整条**切分（70% 标定 / 30% 留出），固定种子 ⇒ 可复现。

    🔴 **同源泄露防护**：一个 ``seq_id`` 只可能落在一侧 ⇒ 不存在「同一视频前半段仿真、
    后半段测误杀率」的情况（任务书 §四.1）。
    """
    ordered = sorted(set(seq_ids))
    rng = random.Random(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    n_calib = int(round(len(shuffled) * ratio))
    n_calib = max(1, min(n_calib, len(shuffled) - 1)) if len(shuffled) > 1 else len(shuffled)
    calib = set(shuffled[:n_calib])
    return {seq_id: ("calib" if seq_id in calib else "holdout") for seq_id in ordered}


def verify_split_disjoint(split_map: dict[str, str]) -> tuple[bool, list[str]]:
    """机器校验：两侧 ``seq_id`` **零交叉**。返回 ``(是否通过, 交叉列表)``。"""
    calib = {s for s, v in split_map.items() if v == "calib"}
    holdout = {s for s, v in split_map.items() if v == "holdout"}
    return (not (calib & holdout)), sorted(calib & holdout)


# --------------------------------------------------------------------------- 明细读取
def read_clean_rows(detail_path: Path) -> list[dict]:
    """读明细 CSV 的 **clean 档**（Pass A 干净帧分类的输入）。"""
    rows = []
    with open(detail_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("condition_family") != "clean":
                continue
            rows.append(row)
    return rows


def classify_rows(clean_rows: list[dict]) -> list[dict]:
    """把干净帧明细行转成**分类表行**（每臂分类 + 数据集级分类）。"""
    out = []
    for row in clean_rows:
        target_w = float(row["w"])
        target_h = float(row["h"])
        tolerance = tolerance_for(target_w, target_h)
        arm_classes, cle_by_arm, maxcle_by_arm, complete_by_arm = {}, {}, {}, {}
        for arm in TRACKER_FAMILIES:
            cle_end = _to_float(row.get("%s_cle_end" % arm))
            maxcle = _to_float(row.get("%s_maxcle" % arm))
            complete = str(row.get("%s_complete" % arm, "")).strip() == "1"
            success = str(row.get("%s_success" % arm, "")).strip() == "1"
            arm_classes[arm] = classify_arm(cle_end, maxcle, complete, tolerance, success)
            cle_by_arm[arm] = cle_end
            maxcle_by_arm[arm] = maxcle
            complete_by_arm[arm] = int(complete)
        out.append({
            "seq_id": row["seq_id"],
            "frame_id": int(row["frame_id"]),
            "small_target": int(row.get("small_target", 0) or 0),
            "w": target_w,
            "h": target_h,
            "tolerance": tolerance,
            **{"cle_%s" % arm: cle_by_arm[arm] for arm in TRACKER_FAMILIES},
            **{"maxcle_%s" % arm: maxcle_by_arm[arm] for arm in TRACKER_FAMILIES},
            **{"complete_%s" % arm: complete_by_arm[arm] for arm in TRACKER_FAMILIES},
            **{"class_%s" % arm: arm_classes[arm] for arm in TRACKER_FAMILIES},
            "class_ds": classify_dataset(arm_classes),
        })
    return out


def write_class_csv(path: Path, class_rows: list[dict]) -> Path:
    fields = ["seq_id", "frame_id", "small_target", "w", "h", "tolerance"]
    for arm in TRACKER_FAMILIES:
        fields += ["cle_%s" % arm, "maxcle_%s" % arm, "complete_%s" % arm, "class_%s" % arm]
    fields += ["class_ds"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in class_rows:
            writer.writerow(row)
    return path


def read_class_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_split_csv(path: Path, split_map: dict[str, str], class_rows: list[dict]) -> Path:
    """写出 ``seq_id`` 切分表（含每条序列的好片段计数）。"""
    counts: dict[str, dict[str, int]] = {}
    for row in class_rows:
        bucket = counts.setdefault(row["seq_id"], {CLASS_GOOD: 0, CLASS_MEDIOCRE: 0, CLASS_LOST: 0})
        bucket[row["class_ds"]] += 1
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq_id", "split", "n_windows", "n_good", "n_mediocre", "n_lost"])
        for seq_id in sorted(split_map):
            bucket = counts.get(seq_id, {CLASS_GOOD: 0, CLASS_MEDIOCRE: 0, CLASS_LOST: 0})
            writer.writerow([seq_id, split_map[seq_id], sum(bucket.values()),
                             bucket[CLASS_GOOD], bucket[CLASS_MEDIOCRE], bucket[CLASS_LOST]])
    return path


def read_split_csv(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        return {row["seq_id"]: row["split"] for row in csv.DictReader(handle)}


# --------------------------------------------------------------------------- 约束线读取
def read_constraint_lines(path: Path) -> dict[str, dict]:
    """读 ``constraint_lines_v2.csv`` ⇒ ``{metric: {...}}``。"""
    out = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["metric"]] = row
    return out


# --------------------------------------------------------------------------- G4 真实坏片盲测
def detect_content_box(frame: np.ndarray, black_thresh: int = 16) -> tuple[int, int, int, int]:
    """黑边（pillarbox）检测——**复用 v1 的 owner** ``manu/video_quality/vq_metrics``。

    v1 结论（``manu/video_quality/report/REPORT_视频质量指标.md``）：
    坏组 IR 源片有**大面积黑边**，不裁边会把坏片「测成」清晰 ⇒ 必须先裁。
    """
    from vq_metrics import detect_content_box as _detect

    return _detect(frame, black_thresh)


def read_tail_frames(path: str, seconds: float = FAULT_SECONDS) -> tuple[list[np.ndarray], float, dict]:
    """读视频**最后 ``seconds`` 秒**的灰度帧（故障窗，v1 同款口径）。

    ``.mp4`` 可随机访问 ⇒ 直接 seek；``.h264`` 裸流**没有索引** ⇒ 顺序解码并保留环形缓冲。
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("打不开视频：%s" % path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    declared = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 🔴 裸流（``.h264`` 无容器索引 ⇒ FFmpeg 不知道时长）⇒ ``CAP_PROP_FRAME_COUNT``
    #    返回**哨兵负数** ``-230584300921369``（不是 0，``or 0`` 兜不住）⇒
    #    只认「有限正数」，其余（0 / NaN / 负数）一律当 0 ⇒ 走顺序解码兜底取真实帧数。
    total = int(declared) if math.isfinite(declared) and declared > 0 else 0
    want = max(1, int(round(fps * seconds)))
    frames: list[np.ndarray] = []
    meta = {"fps": fps, "frame_count": total, "seek": False}
    if total > want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - want)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        meta["seek"] = True
        if len(frames) < want:
            # 🔴 容器声明的总帧数**大于文件里实际可解码的帧数**（``/mnt/data`` 上这份 mp4
            #    是截断件，ffprobe 报 "partial file"）⇒ seek 到 ``total-want`` 后提前 EOF
            #    ⇒ 真实总帧数 = 起点帧号 + 实际取到的帧数（起点 = ``total-want``）。
            meta["frame_count"] = total - want + len(frames)
    if not frames:                        # 裸流 seek 失败 ⇒ 顺序解码 + 环形缓冲
        cap.release()
        cap = cv2.VideoCapture(path)
        from collections import deque

        buffer = deque(maxlen=want)
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        frames = list(buffer)
        meta["seek"] = False
        # 顺序解码走完全片 ⇒ 解码帧数**就是真实总帧数**（裸流无索引，只能这么数）。
        meta["frame_count"] = decoded
    cap.release()
    return frames, fps, meta


def detect_saliency_roi(frame: np.ndarray, reference_width: int = 640) -> tuple[float, float, float, float]:
    """**自动目标 ROI**（DoG 小目标响应峰值）——真实坏片**没有 GT** 时的唯一可行口径。

    * DoG = ``G(σ=1) - G(σ=4)``（与 v1 ``vq_metrics`` 的小目标响应同款算子）；
    * 峰值位置 ⇒ ROI 中心；峰值邻域的连通域包围盒 ⇒ ROI 尺寸；
    * 尺寸按画面宽度相对 ``reference_width`` 线性缩放（坏片 1280 px 宽 vs 好片 640 px 宽）；
    * 搜索范围**内缩 8 px**，避免 ROI 落在画面边缘。
    """
    import cv2

    height, width = frame.shape[:2]
    scale = width / float(reference_width)
    dog = (cv2.GaussianBlur(frame, (0, 0), 1.0).astype(np.float32)
           - cv2.GaussianBlur(frame, (0, 0), 4.0).astype(np.float32))
    # 🔴 ROI 搜索必须**避开画面边缘带**：坏组源片的内容底部有一条**硬边残影带**
    #    （v1 结论：黑边硬边缘会抬高拉普拉斯能量，把坏片「测成」清晰）。
    #    这里内缩 4%（≥16 px）⇒ ROI 不可能落在边缘伪影上，只能落在真实画面里。
    margin = max(16, int(round(0.04 * min(height, width))))
    masked = dog.copy()
    masked[:margin, :] = -np.inf
    masked[-margin:, :] = -np.inf
    masked[:, :margin] = -np.inf
    masked[:, -margin:] = -np.inf
    peak_y, peak_x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    peak_value = float(dog[peak_y, peak_x])

    # ---- ROI 尺寸：峰值邻域内 DoG ≥ 0.5×峰值 的连通域包围盒
    half = 24
    y0, y1 = max(0, peak_y - half), min(height, peak_y + half + 1)
    x0, x1 = max(0, peak_x - half), min(width, peak_x + half + 1)
    patch = dog[y0:y1, x0:x1]
    box_w = box_h = 0
    if peak_value > 0:
        mask = (patch >= 0.5 * peak_value).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        label = labels[peak_y - y0, peak_x - x0]
        if label > 0:
            box_w = int(stats[label, cv2.CC_STAT_WIDTH])
            box_h = int(stats[label, cv2.CC_STAT_HEIGHT])
    if box_w < 3 or box_h < 3:            # 连通域退化 ⇒ 用 Anti-UAV 目标中位尺寸缩放
        box_w, box_h = int(round(29 * scale)), int(round(21 * scale))
    roi_w = float(min(max(box_w, max(9, round(9 * scale))), max(9, round(96 * scale))))
    roi_h = float(min(max(box_h, max(9, round(9 * scale))), max(9, round(96 * scale))))
    return float(peak_x), float(peak_y), roi_w, roi_h


def evaluate_real_clip(path: str, name: str, group: str) -> list[dict]:
    """对一条真实视频的**故障窗**逐帧算 v2 局部 ROI 指标（裁边后、自动 ROI）。"""
    frames, fps, meta = read_tail_frames(path, FAULT_SECONDS)
    if not frames:
        return []
    crop = detect_content_box(frames[0])
    top, bottom, left, right = crop
    height, width = frames[0].shape[:2]
    rows = []
    for index, frame in enumerate(frames):
        content = frame[top:height - bottom, left:width - right]
        if content.size == 0:
            content = frame
        roi_x, roi_y, roi_w, roi_h = detect_saliency_roi(content)
        metrics = compute_roi_metrics(content, roi_x, roi_y, roi_w, roi_h)
        rows.append({
            "clip": name,
            "group": group,
            "frame_index": index,
            "time_s": round((len(frames) - 1 - index) / fps, 4) if fps else 0.0,
            "border_top": top, "border_bottom": bottom,
            "border_left": left, "border_right": right,
            "content_w": int(content.shape[1]), "content_h": int(content.shape[0]),
            "roi_cx": round(roi_x, 2), "roi_cy": round(roi_y, 2),
            "roi_w": round(roi_w, 2), "roi_h": round(roi_h, 2),
            "fps": round(fps, 4), "n_frames_total": meta["frame_count"],
            "n_frames_window": len(frames), "seek_ok": int(meta["seek"]),
            **{key: metrics[key] for key in ALL_METRICS},
        })
    return rows


# --------------------------------------------------------------------------- 回归统计
def _rate_below(values: np.ndarray, line: float) -> tuple[int, int, float]:
    """红线以下的样本数 / 总数 / 比例。"""
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not math.isfinite(line):
        return 0, int(finite.size), float("nan")
    below = int((finite < line).sum())
    return below, int(finite.size), below / float(finite.size)


def summarize_set(name: str, rows: list[dict], metric: str, line: float,
                  source: str, note: str = "") -> dict:
    values = np.array([float(r.get(metric, float("nan"))) for r in rows], dtype=float)
    below, total, rate = _rate_below(values, line)
    finite = values[np.isfinite(values)]
    return {
        "set": name,
        "metric": metric,
        "n_total": len(rows),
        "n_evaluated": total,
        "n_below_line": below,
        "rate_below_line": round(rate, 6) if math.isfinite(rate) else float("nan"),
        "median": round(float(np.median(finite)), 6) if finite.size else float("nan"),
        "p05": round(float(np.percentile(finite, 5)), 6) if finite.size else float("nan"),
        "p95": round(float(np.percentile(finite, 95)), 6) if finite.size else float("nan"),
        "threshold": line,
        "threshold_source": source,
        "note": note,
    }


# --------------------------------------------------------------------------- 阶段：分类 / 回归 / 盲测
def phase_classify(cli) -> Path:
    """（重）生成三分法分类表与序列级切分表——直接从明细 CSV 的 clean 档派生，不重跑跟踪。"""
    detail_path = Path(cli.detail_csv).resolve()
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_rows = classify_rows(read_clean_rows(detail_path))
    good_seqs = sorted({row["seq_id"] for row in class_rows if row["class_ds"] == CLASS_GOOD})
    split_map = split_sequences(good_seqs)
    disjoint, overlap = verify_split_disjoint(split_map)
    if not disjoint:
        raise SystemExit("[FATAL] 序列级切分出现交叉：%s" % overlap)
    write_class_csv(output_dir / "segment_class_v2.csv", class_rows)
    write_split_csv(output_dir / "split_v2.csv", split_map, class_rows)
    print("[CLASSIFY] 分类 %d 窗口 / 序列切分 标定 %d · 留出 %d（零交叉 ✅）"
          % (len(class_rows),
             sum(1 for v in split_map.values() if v == "calib"),
             sum(1 for v in split_map.values() if v == "holdout")))
    return output_dir / "segment_class_v2.csv"


def _load_clean_metrics(detail_path: Path) -> dict[tuple, dict]:
    """clean 档明细 ⇒ ``{(seq_id, frame_id): {metric: value, ...}}``。"""
    out = {}
    for row in csv.DictReader(open(detail_path, encoding="utf-8")):
        if row.get("condition_family") != "clean":
            continue
        record = {key: _to_float(row.get(key)) for key in ALL_METRICS}
        out[(row["seq_id"], int(row["frame_id"]))] = record
    return out


def phase_blind(cli) -> Path:
    """真实坏片 / 好片对照的**独立盲测**（G4）——零参与阈值搜索。

    🔴 坏片只有 2 条 ⇒ 结论只能写「**小样本级**」。
    """
    output_dir = Path(cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in REAL_BAD_CLIPS:
        rows += evaluate_real_clip(path, name, "bad")
    for name, path in REAL_GOOD_CLIPS:
        rows += evaluate_real_clip(path, name, "good")
    blind_path = output_dir / "blind_real_v2.csv"
    fields = ["clip", "group", "frame_index", "time_s", "border_top", "border_bottom",
              "border_left", "border_right", "content_w", "content_h",
              "roi_cx", "roi_cy", "roi_w", "roi_h", "fps", "n_frames_total",
              "n_frames_window", "seek_ok"] + list(ALL_METRICS)
    with open(blind_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("[BLIND] 真实坏片 %d 条 + 好片对照 %d 条，帧级明细 %d 行 → %s"
          % (len(REAL_BAD_CLIPS), len(REAL_GOOD_CLIPS), len(rows), blind_path))
    return blind_path


def phase_regress(cli) -> Path:
    """三分法回归测试汇总（G4）——好片误杀率 / 临界敏感性 / 坏片拦截率。"""
    output_dir = Path(cli.output_dir).resolve()
    detail_path = Path(cli.detail_csv).resolve()
    class_rows = read_class_csv(output_dir / "segment_class_v2.csv")
    split_map = read_split_csv(output_dir / "split_v2.csv")
    lines = read_constraint_lines(output_dir / "constraint_lines_v2.csv")
    metrics_by_key = _load_clean_metrics(detail_path)
    blind_path = output_dir / "blind_real_v2.csv"
    blind_rows = list(csv.DictReader(open(blind_path, encoding="utf-8"))) if blind_path.exists() else []

    # ---- 三个集合（好片段只看**留出集**；一般 / 跟丢用全集）
    good_holdout, mediocre, lost = [], [], []
    for row in class_rows:
        key = (row["seq_id"], int(row["frame_id"]))
        if key not in metrics_by_key:
            continue
        record = dict(metrics_by_key[key])
        record["_key"] = key
        if row["class_ds"] == CLASS_GOOD and split_map.get(row["seq_id"]) == "holdout":
            good_holdout.append(record)
        elif row["class_ds"] == CLASS_MEDIOCRE:
            mediocre.append(record)
        elif row["class_ds"] == CLASS_LOST:
            lost.append(record)

    # ---- 膝点邻域（用于「一般片段的指标分布是否落在膝点邻域」）
    #      邻域 = 膝点值的相对窗口 x [1/KNEE_BAND_FACTOR, KNEE_BAND_FACTOR]
    knee_rows = list(csv.DictReader(open(output_dir / "knee_thresholds_v2.csv", encoding="utf-8")))
    bands = {}
    for row in knee_rows:
        if row["curve_set"] != "common_good" or row["stratum"] != "all":
            continue
        try:
            knee = float(row["knee_x_final"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(knee):
            continue
        low, high = knee / KNEE_BAND_FACTOR, knee * KNEE_BAND_FACTOR
        band = bands.setdefault(row["metric"], [low, high])
        band[0] = min(band[0], low)
        band[1] = max(band[1], high)

    def _auc(pos_rows: list[dict], neg_rows: list[dict], metric: str) -> float:
        if not pos_rows or not neg_rows:
            return float("nan")
        scores = np.array([-_to_float(r.get(metric)) for r in pos_rows + neg_rows], dtype=float)
        labels = np.array([1.0] * len(pos_rows) + [0.0] * len(neg_rows), dtype=float)
        return roc_auc(scores, labels)

    out_rows = []
    for metric in CORE_METRICS:
        line = _to_float(lines.get(metric, {}).get("line_value"))
        band = bands.get(metric, [float("nan"), float("nan")])
        source = "constraint_lines_v2.csv(%s)" % lines.get(metric, {}).get("deciding_family", "?")
        # 🔴 **本轮无有效红线的指标**（守卫 / 双算法门禁未过 ⇒ 任务书 §二.3「降级为物理参照」）：
        #    回归证据**不能一起丢掉**（误杀率 / 拦截率是判别力的唯一量化来源），
        #    ⇒ 改用约束线表里的**参照读数**当阈值，并在 ``threshold_source`` 里标明
        #    「reference_only」——该列数字**不是红线**，不得对外引用为免责依据。
        if not math.isfinite(line):
            ref = _to_float(lines.get(metric, {}).get("xcorr_ref"))
            if not math.isfinite(ref):
                ref = _to_float(lines.get(metric, {}).get("csrt_ref"))
            if math.isfinite(ref):
                line = ref
                source = "reference_only(本轮无有效红线,约束线表参照读数)"
                if not (math.isfinite(band[0]) and math.isfinite(band[1])):
                    band = [ref / KNEE_BAND_FACTOR, ref * KNEE_BAND_FACTOR]

        def _band_frac(rows: list[dict]) -> float:
            values = np.array([_to_float(r.get(metric)) for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0 or not (math.isfinite(band[0]) and math.isfinite(band[1])):
                return float("nan")
            inside = ((values >= band[0]) & (values <= band[1])).sum()
            return float(inside) / float(values.size)

        # ---- 好片段 · 30% 留出集 → **好片误杀率**（红线以下 = 被误杀）
        row = summarize_set("good_holdout", good_holdout, metric, line, source,
                             "好片段·30%留出集（序列级隔离）⇒ 好片误杀率")
        row["knee_band_low"], row["knee_band_high"] = band[0], band[1]
        row["frac_in_knee_band"] = round(_band_frac(good_holdout), 6)
        row["auc_vs_lost"] = round(_auc(good_holdout, lost, metric), 6)
        row["auc_vs_good"] = float("nan")
        out_rows.append(row)

        # ---- 一般片段 → **临界敏感性**（不参与定标）
        row = summarize_set("mediocre", mediocre, metric, line, source,
                             "一般片段（临界灰区）⇒ 阈值敏感度 + 是否落在膝点邻域")
        row["knee_band_low"], row["knee_band_high"] = band[0], band[1]
        row["frac_in_knee_band"] = round(_band_frac(mediocre), 6)
        row["auc_vs_good"] = round(_auc(mediocre, good_holdout, metric), 6)
        row["auc_vs_lost"] = round(_auc(mediocre, lost, metric), 6)
        out_rows.append(row)

        # ---- 跟丢片段 → **坏片拦截率**
        row = summarize_set("lost", lost, metric, line, source,
                             "跟丢片段（干净帧真实失锁）⇒ 坏片拦截率")
        row["knee_band_low"], row["knee_band_high"] = band[0], band[1]
        row["frac_in_knee_band"] = round(_band_frac(lost), 6)
        row["auc_vs_good"] = round(_auc(lost, good_holdout, metric), 6)
        row["auc_vs_lost"] = float("nan")
        out_rows.append(row)

        # ---- 真实坏片（2 条 ⇒ **小样本级**）与真实好片对照
        bad_rows = [r for r in blind_rows if r.get("group") == "bad"]
        good_rows = [r for r in blind_rows if r.get("group") == "good"]
        row = summarize_set("real_bad", bad_rows, metric, line, source,
                             "真实坏片故障窗（最后5秒+黑边裁剪）⇒ 小样本级（2 条）")
        row["knee_band_low"], row["knee_band_high"] = band[0], band[1]
        row["frac_in_knee_band"] = float("nan")
        row["auc_vs_good"] = float("nan")
        row["auc_vs_lost"] = float("nan")
        out_rows.append(row)
        row = summarize_set("real_good", good_rows, metric, line, source,
                             "真实好片对照（同口径故障窗）⇒ 误杀检查（小样本级，2 条）")
        row["knee_band_low"], row["knee_band_high"] = band[0], band[1]
        row["frac_in_knee_band"] = float("nan")
        row["auc_vs_good"] = float("nan")
        row["auc_vs_lost"] = float("nan")
        out_rows.append(row)

    regress_path = output_dir / "regression_v2.csv"
    fields = ["set", "metric", "n_total", "n_evaluated", "n_below_line", "rate_below_line",
              "median", "p05", "p95", "threshold", "threshold_source",
              "knee_band_low", "knee_band_high", "frac_in_knee_band",
              "auc_vs_good", "auc_vs_lost", "note"]
    with open(regress_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k) for k in fields})
    print("[REGRESS] 回归汇总 %d 行 → %s（好片误杀 / 临界敏感 / 坏片拦截）"
          % (len(out_rows), regress_path))
    return regress_path
