#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 流水线的机械验收测试套件（``pytest`` 100% 通过）。

五组门禁（任务书指定）
----------------------
1. **边界安全与零重叠**：目标位于图像四角 / 边缘时，裁剪不越界、无 ``NaN``/``Inf``、
   且核心区与背景环的交集**严格为空**；
2. **红外白热 / 黑热极性对称性**：暗背景下的白热亮目标与亮背景下的黑热暗目标，
   ``local_scr`` 均为正且**对称一致**；
3. **数据洁净度**：``dataset_manifest.json`` 只含红外序列、多目标序列数为 0，
   所有评估窗口 5 帧均 ``exist == 1`` 且 ``w > 0, h > 0``；
4. **退化单调性 Sanity Check**：退化强度递增 ⇒ 对应画质指标单调下降、``CLE_{t+4}`` 单调上升；
5. **产物 MD5 校验**：``iqa_v2/output/manifest_v2.md5`` 全量 ``md5sum -c`` 通过。

另加一组（v2 膝点反解的**曲线有效性守卫**，非任务书门禁）：平曲线 / 非单调曲线 /
指标非单调的膝点一律判 ``degenerate``（不可靠 + 线位置 nan），且**不得**成为约束红线。

执行机：服务器 ``huangzhe@192.168.99.40:32222``（conda env ``uav``，依赖 cv2 + scipy）。
复现：

    cd /tmp/pycharm_project_a7d966e6 && python3 -m pytest iqa_v2/test_iqa_v2.py -v
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from degradation_4d import (BLUR_SIGMAS, CONDITION_PROFILES, CONTRAST_ALPHAS,  # noqa: E402
                            DEEP_BLUR_SIGMAS, DEEP_CONTRAST_ALPHAS, DEEP_MOTION_LENGTHS,
                            FAMILIES, MOTION_LENGTHS, apply_degradation, build_conditions,
                            degradation_strength, motion_angle_for)
from dataset_anti_uav import (SMALL_TARGET_MAX_DIM, WINDOW_LEN,  # noqa: E402
                              discover_sequence_dirs, load_sequence)
from regression_v2 import (CALIB_RATIO, CLASS_GOOD, CLASS_LOST, CLASS_MEDIOCRE,  # noqa: E402
                           CORE_METRICS, GOOD_CLE_PX, classify_arm, split_sequences,
                           verify_split_disjoint)
from run_eval_v2 import (CURVE_MIN_SPAN, CURVE_SETS, DEGRADATION_FAMILIES,  # noqa: E402
                         KNEE_AGREE_PCT, N_BINS,
                         _clamp_rect, _kcf_tracker, extract_knee,
                         level_metric_profile, level_success_profile, track_csrt,
                         track_kcf_legacy, track_xcorr)
from target_roi_metrics import (ALL_METRICS, EPS, GLOBAL_METRICS, LOCAL_METRICS,  # noqa: E402
                                _p90, build_regions, compute_roi_metrics,
                                regions_overlap_pixels)

REPO = HERE.parent
#: v1 的黑边检测算子住在 ``manu/video_quality/`` ⇒ 复用，不重写（G4）。
VQ_DIR = REPO / "manu" / "video_quality"
if str(VQ_DIR) not in sys.path:
    sys.path.insert(0, str(VQ_DIR))
DATASET_ROOT = Path(os.environ.get("IQA_V2_DATASET_ROOT",
                                   "/mnt/data/siping/datasets/manu/anti-uav"))
# ⚠️ 产物落在 **iqa_v2/output/**（run_eval_v2.py 的 --output-dir 默认值），
#    不是仓库根的 output/ ⇒ 写错会让数据洁净度门禁永远「跳过」而不是真正生效。
OUTPUT_DIR = Path(os.environ.get("IQA_V2_OUTPUT_DIR", str(HERE / "output")))


# --------------------------------------------------------------------------- 工具
def synthetic_ir_frame(width: int = 640, height: int = 512, cx: float = 320.0,
                       cy: float = 256.0, target_w: int = 20, target_h: int = 16,
                       background: int = 30, target_value: int = 200,
                       noise_sigma: float = 3.0, seed: int = 0) -> np.ndarray:
    """合成一帧「红外味」图像：带噪背景 + 亮目标 + 一点结构纹理。"""
    rng = np.random.default_rng(seed)
    frame = np.full((height, width), float(background), dtype=np.float64)
    frame += rng.normal(0.0, noise_sigma, size=frame.shape)
    for column in range(0, width, 37):          # 背景结构（让 σ_ring > 0 且有边缘）
        frame[:, column] += 12.0
    # 目标本体画成**高斯点目标**（真实红外小目标就是高斯型光斑）：
    # 若画成硬边矩形，边缘的 Laplacian 响应会被集中在 1 像素上，
    # 任何模糊都会把这条能量摊开 ⇒ ``target_lap_var`` 反而变大 ⇒ 单调性断言病态。
    x0 = int(round(cx - target_w / 2.0))
    y0 = int(round(cy - target_h / 2.0))
    patch = frame[y0:y0 + target_h, x0:x0 + target_w]
    patch[:] = 0.0
    height, width = patch.shape
    ys, xs = np.mgrid[0:height, 0:width]
    sigma = max(1.0, min(target_w, target_h) / 2.2)
    blob = np.exp(-((xs - (width - 1) / 2.0) ** 2 + (ys - (height - 1) / 2.0) ** 2)
                   / (2.0 * sigma ** 2))
    patch += blob * float(target_value)
    return np.clip(frame, 0, 255).astype(np.uint8)


def local_scr_float(frame: np.ndarray, cx: float, cy: float, w: float, h: float) -> float:
    """在**浮点域**复算 ``local_scr``（与 :func:`compute_roi_metrics` 同口径）。

    为什么需要它：``apply_degradation`` 的对比度压缩会把帧量化回 **uint8**，
    量化噪声（σ ≈ 0.289 LSB）**不随 α 缩小**，于是 α=0.15 档的信号被压到 1/6.7
    而分母多出量化噪声 ⇒ 8bit 域的 local_scr 必然漂移（实测 12%），
    **这不是 SCR 定义被破坏**。浮点域复算用于把「定义性质」与「量化残差」分开验证。
    """
    regions = build_regions(frame, cx, cy, w, h)
    core_vals = frame[regions.core_mask].astype(np.float64)
    ring_vals = frame[regions.ring_mask].astype(np.float64)
    mu_ring = float(ring_vals.mean())
    sigma_ring = float(ring_vals.std())
    p90_diff = _p90(np.abs(core_vals - mu_ring))
    return float(p90_diff / (sigma_ring + EPS))


def real_ir_window(seq_dir: Path, frame_id: int):
    """从真实序列里取 5 帧 + 首帧目标框（拿不到就返回 None）。"""
    import cv2

    seq = load_sequence(seq_dir)
    if not seq.kept or len(seq.frame_files) < frame_id + WINDOW_LEN:
        return None
    frames = []
    for offset in range(WINDOW_LEN):
        image = cv2.imread(str(seq.frame_files[frame_id + offset]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        frames.append(image)
    records = [seq.records[frame_id + offset] for offset in range(WINDOW_LEN)]
    if any(r.exist != 1 or r.w <= 0 or r.h <= 0 for r in records):
        return None
    return frames, records


@pytest.fixture(scope="module")
def any_sequence_dir():
    """任意一条**真实**红外序列目录（没有数据集就跳过数据相关用例）。"""
    if not DATASET_ROOT.is_dir():
        pytest.skip("Anti-UAV 数据集不在 %s" % DATASET_ROOT)
    dirs = discover_sequence_dirs(DATASET_ROOT)
    if not dirs:
        pytest.skip("没有找到任何红外序列")
    return dirs[0]


# --------------------------------------------------------------------------- 1. 边界安全
@pytest.mark.parametrize("cx,cy", [(2, 2), (0, 0), (637, 509), (639, 511),
                                   (2, 509), (637, 2), (320, 1), (320, 510)])
def test_boundary_safety_no_nan_and_no_overlap(cx, cy):
    """目标在四角 / 边缘：掩码不越界、无 NaN/Inf、核心区与背景环交集为 0。"""
    frame = synthetic_ir_frame(cx=cx, cy=cy, target_w=20, target_h=16)
    height, width = frame.shape
    metrics = compute_roi_metrics(frame, cx, cy, 20, 16)
    for name in ALL_METRICS:
        assert math.isfinite(metrics[name]), "%s 出现非有限值（目标在 (%s, %s)）" % (name, cx, cy)
    regions = build_regions(frame, cx, cy, 20, 16)
    for box in (regions.core_box, regions.ring_box, regions.outer_box):
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= width, "核心/背景框越界：%s" % (box,)
        assert 0 <= y0 < y1 <= height, "核心/背景框越界：%s" % (box,)
    overlap = int(np.logical_and(regions.core_mask, regions.ring_mask).sum())
    assert overlap == 0, "核心区与背景环重叠 %d 像素" % overlap
    assert regions_overlap_pixels(frame, cx, cy, 20, 16) == 0


def test_tiny_target_regions_still_valid():
    """极小目标（1×1 标注）也要给出合法区域与有限指标。"""
    frame = synthetic_ir_frame(cx=100, cy=100, target_w=1, target_h=1)
    metrics = compute_roi_metrics(frame, 100, 100, 1, 1)
    for name in ALL_METRICS:
        assert math.isfinite(metrics[name])
    assert regions_overlap_pixels(frame, 100, 100, 1, 1) == 0


# --------------------------------------------------------------------------- 2. 极性对称性
def test_infrared_polarity_symmetry_local_scr():
    """白热亮目标（暗背景）与黑热暗目标（亮背景）的 local_scr 必须为正且对称一致。"""
    frame = synthetic_ir_frame(background=30, target_value=200, noise_sigma=3.0, seed=7)
    inverted = (255 - frame).astype(np.uint8)
    hot = compute_roi_metrics(frame, 320, 256, 20, 16)
    cold = compute_roi_metrics(inverted, 320, 256, 20, 16)
    assert hot["local_scr"] > 0.0, "白热目标的 local_scr 必须为正"
    assert cold["local_scr"] > 0.0, "黑热目标的 local_scr 必须为正"
    assert abs(hot["local_scr"] - cold["local_scr"]) < 1e-6, \
        "白热/黑热 local_scr 不对称：%.9f vs %.9f" % (hot["local_scr"], cold["local_scr"])


def test_infrared_polarity_symmetry_on_real_frame(any_sequence_dir):
    """真实红外帧上同样做一次极性翻转对称性检查。"""
    window = real_ir_window(any_sequence_dir, 0)
    if window is None:
        pytest.skip("该序列首帧没有可用目标")
    frames, records = window
    cx, cy, w, h = records[0].cx, records[0].cy, records[0].w, records[0].h
    frame = frames[0]
    inverted = (255 - frame).astype(np.uint8)
    hot = compute_roi_metrics(frame, cx, cy, w, h)
    cold = compute_roi_metrics(inverted, cx, cy, w, h)
    assert hot["local_scr"] > 0.0 and cold["local_scr"] > 0.0
    assert abs(hot["local_scr"] - cold["local_scr"]) < 1e-6 * max(1.0, hot["local_scr"])


# --------------------------------------------------------------------------- 3. 数据洁净度
def test_dataset_manifest_is_ir_only_and_single_target():
    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("还没跑 dataset 阶段（缺 %s）" % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modality"].startswith("IR only"), "manifest 必须声明只跑红外"
    assert manifest["multi_target_dropped_count"] == 0, "存在多目标序列"
    assert manifest["kept_single_target_sequences"] > 0
    for item in manifest["dropped_sequences"]:
        assert item["reason"], "剔除序列必须写明原因"
    # 所有被保留的序列都必须是「单帧最多 1 框 / 单 track_id」
    for seq in manifest["sequences"]:
        if seq["drop_reason"]:
            continue
        assert seq["max_targets_per_frame"] <= 1, "%s 出现多目标" % seq["seq_id"]
        assert seq["n_track_ids"] <= 1, "%s 出现多 track_id" % seq["seq_id"]


def test_all_windows_are_valid_five_frame_targets():
    windows_path = OUTPUT_DIR / "windows.csv"
    if not windows_path.is_file():
        pytest.skip("还没跑 dataset 阶段（缺 %s）" % windows_path)
    rows = list(csv.DictReader(open(windows_path, encoding="utf-8")))
    assert rows, "窗口清单为空"
    for row in rows:
        assert int(row["length"]) == WINDOW_LEN
        assert float(row["w"]) > 0 and float(row["h"]) > 0, "窗口 %s/%s 尺寸非正" % (
            row["seq_id"], row["frame_id"])
        for index in range(WINDOW_LEN):
            assert int(row["gt_%d_exist" % index]) == 1, \
                "窗口 %s/%s 第 %d 帧 exist != 1" % (row["seq_id"], row["frame_id"], index)


def test_no_visible_modality_file_is_referenced():
    """产物里不能出现任何可见光文件（visible / RGB）。"""
    for path in OUTPUT_DIR.glob("*.csv"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert "visible.mp4" not in text and "rgb_label" not in text, \
            "%s 里出现了可见光引用" % path.name
    if (OUTPUT_DIR / "dataset_manifest.json").is_file():
        manifest = json.loads((OUTPUT_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
        blob = json.dumps(manifest, ensure_ascii=False).lower()
        assert ".mp4" not in blob, "manifest 里出现了视频文件（可见光配对视频不得被引用）"


# --------------------------------------------------------------------------- 4. 退化单调性
def test_degradation_metrics_move_monotonically(any_sequence_dir):
    """退化强度递增 ⇒ 对应画质指标单调下降（**在真实红外帧上**做）。

    为什么必须用真实帧而不是合成帧：合成的高斯点目标在 σ=0.6~1.0 区间会出现
    **采样效应**（离散 Laplacian 对「比像素还尖」的尖峰低估曲率），
    ``target_lap_var`` 反而先升后降（实测 2362 → 6831 → 1243）⇒ 单调性断言病态；
    真实红外帧是带限的，没有这个病态。

    物理注记（**必须一起汇报**）：线性对比度压缩是把**信号与噪声一起**按 α 缩放的，
    因此 **local_scr（信杂比）在数学上就是 α-不变的**（分子分母同乘 α）——
    这不是 bug，而是 SCR 的定义性质；真正被对比度坍缩压下去的是**含绝对电平**的
    ``target_local_contrast`` 与 ``global_contrast``。测试据此分族断言。
    """
    window = real_ir_window(any_sequence_dir, 0)
    if window is None:
        pytest.skip("该序列首帧没有可用目标")
    frames, records = window
    frame = frames[0]
    cx, cy, w, h = records[0].cx, records[0].cy, records[0].w, records[0].h

    # ---- 散焦 / 运动模糊：**锐度类**指标单调下降
    # 🔴 物理注记（必须一起汇报）：``local_scr = P90(|core-μ_ring|) / (σ_ring + ε)``
    # 是**比值**——模糊不仅压低信号，也**抹平了背景噪声**（σ_ring 掉得更快），
    # 于是 local_scr 在模糊族上**不保证单调下降** ⇒ 模糊族用锐度类指标做断言。
    # 散焦：各向同性平滑 ⇒ 锐度类指标**逐档单调下降**（实测真实帧成立）
    sharpness, gradients = [], []
    for sigma in BLUR_SIGMAS:
        metrics = compute_roi_metrics(apply_degradation(frame, "blur", float(sigma)), cx, cy, w, h)
        sharpness.append(metrics["target_lap_var"])
        gradients.append(metrics["target_grad_mean"])
    assert all(sharpness[i + 1] <= sharpness[i] + 1e-9 for i in range(len(sharpness) - 1)), \
        "blur 档位上 target_lap_var 非单调下降：%s" % sharpness
    assert all(gradients[i + 1] <= gradients[i] + 1e-9 for i in range(len(gradients) - 1)), \
        "blur 档位上 target_grad_mean 非单调下降：%s" % gradients

    # 运动模糊：只在一个方向上平滑 ⇒ 2D 梯度/Laplacian 响应对**核的角度栅格化**敏感
    # （30° 这类非轴向角度会把核量化成阶梯，反而多出局部结构 ⇒ 逐档不严格单调，
    #  实测真实帧：11947 → 13329 → 9004 → 8586 → 5247）⇒ 只断言**端点变差**。
    motion_sharpness = [compute_roi_metrics(apply_degradation(frame, "motion", float(length), 30.0),
                                             cx, cy, w, h)["target_lap_var"]
                         for length in MOTION_LENGTHS]
    assert motion_sharpness[-1] < motion_sharpness[0], (
        "运动模糊最强档必须比最弱档更糊：%s" % motion_sharpness)

    # ---- 对比度压缩：target_local_contrast / global_contrast 单调下降，
    #      local_scr 保持 α-不变（定义性质）
    contrasts, globals_, scrs = [], [], []
    for alpha in CONTRAST_ALPHAS:
        degraded = apply_degradation(frame, "contrast", alpha)
        metrics = compute_roi_metrics(degraded, cx, cy, w, h)
        contrasts.append(metrics["target_local_contrast"])
        globals_.append(metrics["global_contrast"])
        scrs.append(metrics["local_scr"])
    assert all(contrasts[i + 1] <= contrasts[i] + 1e-9 for i in range(len(contrasts) - 1)), \
        "contrast 档位上 target_local_contrast 非单调下降：%s" % contrasts
    assert all(globals_[i + 1] <= globals_[i] + 1e-9 for i in range(len(globals_) - 1)), \
        "contrast 档位上 global_contrast 非单调下降：%s" % globals_
    # local_scr 的 α-不变性是**定义性质**（分子分母同乘 α）⇒ 在**浮点域**做严格断言：
    # 不量化 ⇒ 漂移必须小于 1e-9 相对误差。
    synth = synthetic_ir_frame().astype(np.float64)
    mu_synth = float(synth.mean())
    synth_scrs = [local_scr_float(alpha * synth + (1.0 - alpha) * mu_synth,
                                  320.0, 256.0, 20, 16)
                  for alpha in CONTRAST_ALPHAS]
    # 残差来源：分母是 ``σ_ring + EPS``（EPS = 1e-5）而非纯 σ_ring ⇒ 分母不是齐次的，
    # α=0.15 档 σ_ring 被压到 0.57 ⇒ EPS/σ ≈ 1.8e-5 ⇒ 相对漂移 ~1.4e-5（实测）。
    # ⇒ 容差取 1e-4 相对，既卡住「齐次性被破坏」这种真 bug，又不误伤 EPS 的物理残差。
    assert max(synth_scrs) - min(synth_scrs) < 1e-4 * max(1.0, max(synth_scrs)), \
        "local_scr 在纯线性对比度压缩下应保持不变（浮点域实测漂移 %.3e）" \
        % (max(synth_scrs) - min(synth_scrs))

    # 8bit 域（真实流水线口径）里 local_scr 只保证**近似**不变：量化噪声 σ ≈ 0.289 LSB
    # 不随 α 缩小，α=0.15 档信号被压到 1/6.7 ⇒ 比值被量化噪声抬高（实测漂移 ~12%）。
    # 断言只要求「对比度压缩不会把 SCR 改变量级」，把量级留给 blur 族单独检验。
    assert max(scrs) - min(scrs) <= 0.15 * max(1.0, max(scrs)), \
        "local_scr 在真实帧的对比度压缩下漂移过大（实测 %.3e）：8bit 量化噪声进入分母" \
        % (max(scrs) - min(scrs))


def test_jpeg_and_white_noise_axes_are_removed():
    """**G1：JPEG 压缩轴与加性白噪声轴必须不存在**（任务书 §四.2）。

    任务书原文：「v1 与 v2 实测均已证实，红外图像加性噪声指标与退化方向相反，
    而 JPEG 压缩在红外上会平滑微弱热噪声导致局部块效应不升反降（-16.8%），
    违背单调性假设。**后续重标时仅保留散焦模糊、运动拖影、对比度衰减三项真实红外退化。**」

    ⇒ 退化算子**不认识** ``jpeg``（不是「支持但不用」，而是**不存在**），
    防止后续有人为了「让曲线好看」把它加回来。
    """
    families = {c.family for c in build_conditions()}
    assert "jpeg" not in families, "JPEG 退化轴必须已剔除"
    assert FAMILIES == ("blur", "motion", "contrast")
    assert DEGRADATION_FAMILIES == ("blur", "motion", "contrast")
    for noise_axis in ("noise", "white_noise", "gaussian_noise"):
        assert noise_axis not in families, "加性白噪声轴必须不存在：%s" % noise_axis
    with pytest.raises(ValueError):
        apply_degradation(np.zeros((32, 32), np.uint8), "jpeg", 60.0)


def test_good_segment_threshold_is_task_book_3px():
    """**G2：好片段判定必须是任务书给死的 ``CLE ≤ 3 px``**（不得自行调整）。"""
    assert GOOD_CLE_PX == 3.0
    tolerance = 20.0
    assert classify_arm(3.0, 3.0, True, tolerance) == CLASS_GOOD
    assert classify_arm(3.01, 3.01, True, tolerance) == CLASS_MEDIOCRE
    assert classify_arm(tolerance, tolerance, True, tolerance) == CLASS_MEDIOCRE
    assert classify_arm(tolerance + 0.01, tolerance + 0.01, True, tolerance) == CLASS_LOST
    # 中途失锁（跟踪器没跑完 5 帧）⇒ 直接判跟丢，哪怕末帧位置碰巧靠近 GT
    assert classify_arm(0.1, 0.1, False, tolerance) == CLASS_LOST
    # 硬失败（NaN）⇒ 跟丢
    assert classify_arm(float("nan"), float("nan"), True, tolerance) == CLASS_LOST


def test_sequence_level_split_is_disjoint_and_reproducible():
    """**G3：序列级 7:3 切分必须按 ``seq_id`` 整条切、零交叉、可复现**（任务书 §四.1）。"""
    seq_ids = ["seq_%03d" % i for i in range(100)]
    first = split_sequences(seq_ids)
    second = split_sequences(seq_ids)
    assert first == second, "固定种子 ⇒ 切分必须可复现"
    calib = {s for s, v in first.items() if v == "calib"}
    holdout = {s for s, v in first.items() if v == "holdout"}
    assert not (calib & holdout), "同一 seq_id 不得同时出现在两侧"
    assert calib | holdout == set(seq_ids)
    assert abs(len(calib) / len(seq_ids) - CALIB_RATIO) < 0.02
    ok, overlap = verify_split_disjoint(first)
    assert ok and not overlap


def test_pillarbox_crop_removes_black_border():
    """**G4：坏组黑边（pillarbox）必须先检出并裁掉**（复用 v1 的 owner）。

    v1 结论：坏组 IR 源片有大面积黑边，不裁边会把坏片「测成」清晰。
    """
    from vq_metrics import detect_content_box

    frame = np.full((108, 192), 100, np.uint8)
    frame[:, :32] = 0
    frame[:, -32:] = 0
    top, bottom, left, right = detect_content_box(frame)
    assert (top, bottom, left, right) == (0, 0, 32, 32)
    content = frame[top:frame.shape[0] - bottom, left:frame.shape[1] - right]
    assert content.shape == (108, 128), "裁边后内容尺寸不对：%s" % (content.shape,)


def test_simulation_base_is_pure_and_holdout_never_simulated():
    """**G2/G3：仿真底座纯净度 + 留出集零参与**（读产物做机器校验）。"""
    knee_path = OUTPUT_DIR / "knee_thresholds_v2.csv"
    class_path = OUTPUT_DIR / "segment_class_v2.csv"
    split_path = OUTPUT_DIR / "split_v2.csv"
    if not (knee_path.is_file() and class_path.is_file() and split_path.is_file()):
        pytest.skip("还没跑完整流水线（缺膝点 / 分类 / 切分表）")
    knees = list(csv.DictReader(open(knee_path, encoding="utf-8")))
    assert knees, "膝点表为空"
    impure = [row for row in knees if float(row["clean_success_rate"]) < 0.999]
    assert not impure, "退化曲线的 clean 档起点必须 ≈ 100%%（底座不纯）：%s" % impure[:3]
    split_map = {row["seq_id"]: row["split"]
                 for row in csv.DictReader(open(split_path, encoding="utf-8"))}
    ok, overlap = verify_split_disjoint(split_map)
    assert ok and not overlap, "序列级切分出现交叉：%s" % overlap
    assert set(split_map.values()) <= {"calib", "holdout"}


def test_simulation_pool_is_dataset_level_good_segments():
    """**G2：退化仿真池 = 三分法「好片段」∩ 标定集，且池内 clean 起点三臂全跟住**。

    本轮把进池条件从「三臂好片段的**并集**」改成「三分法好片段（``class_ds == good``）」：
    并集会把「别的臂跟得住、本臂已经跟丢」的窗口塞进池子 ⇒ 把各臂衰减曲线的**起点**拖下来
    （G2 要消灭的污染）。本用例把这条口径钉死：

    1. 明细底表里出现**退化行**的窗口集合 == 「标定集 ∩ 三分法好片段」；
    2. 该集合内每条窗口在 clean 档**三臂都跟住**（起点跟住率 = 100%）；
    3. 留出集（``holdout``）的窗口**不得**出现在退化行里（零泄露）。
    """
    detail_path = OUTPUT_DIR / "degrade_detail_v2.csv"
    if not detail_path.is_file():
        cache_path = Path("/tmp/iqa_v2_cache/degrade_detail.csv")
        if not cache_path.is_file():
            pytest.skip("还没生成退化明细底表")
        detail_path = cache_path
    class_path = OUTPUT_DIR / "segment_class_v2.csv"
    split_path = OUTPUT_DIR / "split_v2.csv"
    if not (class_path.is_file() and split_path.is_file()):
        pytest.skip("还没跑完整流水线（缺分类表 / 切分表）")
    detail = _read_csv_rows(detail_path)
    class_rows = _read_csv_rows(class_path)
    split_map = {row["seq_id"]: row["split"] for row in _read_csv_rows(split_path)}
    class_index = {(row["seq_id"], row["frame_id"]): row for row in class_rows}

    pool = {(row["seq_id"], row["frame_id"]) for row in detail
            if row.get("condition_family") not in ("clean", "", "error")}
    assert pool, "退化明细里没有任何退化行（仿真池为空）"
    expected = {(row["seq_id"], row["frame_id"]) for row in class_rows
                if row["class_ds"] == CLASS_GOOD and split_map.get(row["seq_id"]) == "calib"}
    assert pool == expected, (
        "仿真池必须恰好等于「标定集 ∩ 三分法好片段」（%d vs %d）" % (len(pool), len(expected)))
    holdout_seqs = {seq_id for seq_id, value in split_map.items() if value == "holdout"}
    assert not (holdout_seqs & {seq_id for seq_id, _frame in pool}), "留出集不得参与退化仿真"

    clean_index = {(row["seq_id"], row["frame_id"]): row for row in detail
                   if row.get("condition_family") == "clean"}
    for arm in ("xcorr", "kcf_legacy", "csrt"):
        key = "%s_success" % arm
        hits = [int(float(clean_index[item][key])) for item in pool if item in clean_index]
        assert hits and all(hits), (
            "仿真池内 clean 档 %s 起点跟住率必须 100%%（底座不纯）：%d/%d"
            % (arm, sum(hits), len(hits)))


def test_knee_below_four_bins_still_carries_guard_label():
    """**膝点守卫标签：分箱不足 4 时也要按守卫结果标注**（``degenerate`` / 空）。

    守卫判定发生在「分箱不足 4 ⇒ 提前返回」**之前** ⇒ 提前返回时必须把守卫结论带出来，
    否则「守卫触发但未标注」的行会混进产物，审计复算无法对上。
    """
    # (a) 守卫触发（非单调跟住率）⇒ 必须标 degenerate
    non_monotone = _synthetic_rows("blur", [0.6, 1.0, 1.5], [1, 0, 1], [100.0, 60.0, 40.0],
                                   "target_lap_var", per_level=1, jitter=0.0)
    result = extract_knee(non_monotone, "target_lap_var", "xcorr", "blur")
    assert result["method_used"] == "degenerate", (
        "守卫触发（跟住率非单调）时必须标 degenerate：%r" % result["method_used"])
    assert result["knee_reliable"] == 0
    assert not math.isfinite(result["knee_x_final"])

    # (b) 守卫未触发但分箱不足 4 ⇒ 不得误标 degenerate（线位本来就没有）
    few_bins = _synthetic_rows("blur", [0.6, 1.0], [1, 0], [100.0, 40.0],
                               "target_lap_var", per_level=1, jitter=0.0)
    result = extract_knee(few_bins, "target_lap_var", "xcorr", "blur")
    assert result["method_used"] != "degenerate", (
        "守卫未触发的曲线不得被标成 degenerate：%r" % result["method_used"])


def test_constraint_lines_only_cover_three_core_metrics():
    """**G5：约束线表只允许出现三项核心指标**（全图 / 辅助指标不得进红线）。

    🔴 本轮新增契约：**某指标若没有通过守卫的有效膝点**，约束线表里必须
    **显式记为 nan 并附原因**（不得静默消失，也不得用参照读数冒充红线）。
    """
    path = OUTPUT_DIR / "constraint_lines_v2.csv"
    if not path.is_file():
        pytest.skip("还没生成约束线表")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert {row["metric"] for row in rows} == set(CORE_METRICS)
    for row in rows:
        assert row["curve_set"] == "common_good", "红线必须来自三臂共同好片段"
        value = _as_float(row.get("line_value"))
        if math.isfinite(value):
            # 有红线 ⇒ 必须有决定族 / 决定臂，且决定族必须是三族之一
            assert row["deciding_family"] in ("blur", "motion", "contrast"), (
                "红线必须有决定族：%s" % _row_tag(row))
            assert row["deciding_tracker"] in ("xcorr", "kcf_legacy"), (
                "红线只能由主臂反解（CSRT 只作对照）：%s" % _row_tag(row))
        else:
            # 无有效红线 ⇒ 必须显式说明原因，且不得留下任何"线位"字段
            assert row["note"].strip(), "无有效红线的指标必须附原因说明：%s" % _row_tag(row)
            for field in ("xcorr_line", "kcf_legacy_line", "csrt_line"):
                assert not math.isfinite(_as_float(row.get(field))), (
                    "无有效红线的指标不得留线位（%s）：%s" % (field, _row_tag(row)))


def test_guard_judges_the_curve_the_knee_comes_from():
    """🔴 **回归用例（本轮真缺陷）**：守卫必须判在**膝点实际取自的那条曲线**上。

    上一版把**逐档跨度**写进 ``curve_y_span``、而膝点却取自**分箱曲线**
    ⇒ 「用 A 曲线的健康度给 B 曲线放行」。实测后果（净化底座上）：
    ``target_lap_var`` / XCorr / ``motion`` 的逐档跨度 6.41 pp（过门槛）但
    **分箱跨度仅 1.92 pp**（近平曲线）⇒ 膝点是噪声点。

    本用例构造一条**逐档跨度够深、但分箱曲线完全是平的**曲线
    （退化把跟住率打下去了，但**指标值完全没动** ⇒ 指标根本不区分跟住/跟丢）
    ⇒ 必须判 ``degenerate``。
    """
    rows = []
    levels = [0.6, 1.0, 1.5, 2.2, 3.0, 4.0]
    for level_index, level in enumerate(levels):
        held = level_index < 3          # 前 3 档全跟住、后 3 档全跟丢
        for index in range(24):         # 24 个互不相同的指标值 ⇒ 分箱后 12 箱
            rows.append({
                "condition_family": "blur",
                "condition_level": float(level),
                "xcorr_success": 1 if held else 0,
                # 指标值**逐档完全相同** ⇒ 退化没有改变指标 ⇒ 指标无法解释跟住率的变化
                "target_lap_var": 1000.0 + index,
            })
    result = extract_knee(rows, "target_lap_var", "xcorr", "blur")
    assert result["level_y_span"] > CURVE_MIN_SPAN, (
        "本用例前提失效：逐档跨度应过门槛，实际 %.4f" % result["level_y_span"])
    assert result["curve_y_span"] <= CURVE_MIN_SPAN, (
        "本用例前提失效：分箱曲线应过平，实际 %.4f" % result["curve_y_span"])
    assert result["method_used"] == "degenerate", (
        "逐档够深但分箱曲线过平的曲线必须判 degenerate（守卫要判在膝点来源曲线上）：%r"
        % result["method_used"])
    assert result["knee_reliable"] == 0
    assert not math.isfinite(result["knee_x_final"])


def test_flat_lap_var_motion_curve_is_not_a_red_line():
    """🔴 **产物级回归**：上一版误当作红线发出去的
    ``target_lap_var`` / XCorr / ``motion`` 必须**不得**出现在约束线表里。"""
    constraint_path = OUTPUT_DIR / "constraint_lines_v2.csv"
    knee_path = OUTPUT_DIR / "knee_thresholds_v2.csv"
    if not constraint_path.is_file() or not knee_path.is_file():
        pytest.skip("还没生成约束线表 / 膝点表")
    knee_rows = _read_csv_rows(knee_path)
    target = [row for row in knee_rows
              if row.get("tracker") == "xcorr" and row.get("curve_set") == "common_good"
              and row.get("degradation_family") == "motion"
              and row.get("metric") == "target_lap_var" and row.get("stratum") == "all"]
    assert target, "膝点表里找不到 target_lap_var/xcorr/motion 的 common_good/all 行"
    row = target[0]
    assert _as_float(row.get("level_y_span")) > CURVE_MIN_SPAN, (
        "前提失效：该曲线的逐档跨度本应过门槛：%s" % _row_tag(row))
    assert _as_float(row.get("curve_y_span")) <= CURVE_MIN_SPAN, (
        "前提失效：该曲线的分箱曲线本应过平：%s" % _row_tag(row))
    assert row["method_used"] == "degenerate", (
        "近平曲线的膝点必须作废：%s" % _row_tag(row))
    lines = {r["metric"]: r for r in _read_csv_rows(constraint_path)}
    assert "target_lap_var" in lines, "约束线表必须显式列出三项核心指标"
    assert not math.isfinite(_as_float(lines["target_lap_var"].get("line_value"))), (
        "target_lap_var 本轮没有有效红线（曲线守卫未过）⇒ 线位必须是 nan，"
        "不得用参照读数冒充红线")


def test_knee_table_carries_both_span_columns():
    """膝点表必须同时给出**两条跨度口径**（分箱曲线 / 逐档），否则无法审计。"""
    path = OUTPUT_DIR / "knee_thresholds_v2.csv"
    if not path.is_file():
        pytest.skip("还没生成膝点表 %s" % path)
    rows = _read_csv_rows(path)
    assert rows, "膝点表为空"
    missing = {"curve_y_span", "level_y_span"} - set(rows[0].keys())
    assert not missing, "膝点表缺少跨度列（产物未重新生成）：%s" % sorted(missing)
    for row in rows:
        binned = _as_float(row.get("curve_y_span"))
        level = _as_float(row.get("level_y_span"))
        assert math.isfinite(binned) and math.isfinite(level), (
            "两条跨度都必须是有限数：%s" % _row_tag(row))
        assert -1e-9 <= binned <= 1.0 and -1e-9 <= level <= 1.0, (
            "跨度必须落在 [0, 1]（跟住率量程）：%s" % _row_tag(row))


def _report_section(text: str, start: str, end: str) -> str:
    """截取报告里两章之间的正文（找不到就 fail，不静默跳过）。"""
    assert start in text, "报告里找不到章节 %s" % start
    body = text.split(start, 1)[1]
    assert end in body, "报告里找不到章节 %s" % end
    return body.split(end, 1)[0]


def test_report_aux_table_lists_every_reliable_knee():
    """🔴 §7「辅助指标与全图对照」表必须**逐条列出**该指标**全部**通过守卫的膝点。

    上一版只查 ``blur/common_good`` 一个切片就写「无（曲线退化 / 门禁未过）」，
    与膝点表里同指标其它族 / 其它曲线集的**有效**膝点直接冲突（同一份报告 §12 自相矛盾）。
    """
    report_path = HERE / "REPORT_V2_FINAL.md"
    knee_path = OUTPUT_DIR / "knee_thresholds_v2.csv"
    if not report_path.is_file() or not knee_path.is_file():
        pytest.skip("还没生成报告 / 膝点表")
    rows = _read_csv_rows(knee_path)
    section = _report_section(report_path.read_text(encoding="utf-8"), "## 七、", "## 八、")
    # §7 只列**非核心**的辅助指标 + 全图指标（核心指标在 §6 约束线表）
    aux_metrics = (set(LOCAL_METRICS) | set(GLOBAL_METRICS)) - set(CORE_METRICS)
    for tracker, cell_index in (("xcorr", 3), ("kcf_legacy", 4)):
        for metric in sorted({r["metric"] for r in rows
                              if r["tracker"] == tracker and r["knee_reliable"] == "1"
                              and r["metric"] in aux_metrics}):
            line = [l for l in section.splitlines() if l.startswith("| `%s` |" % metric)]
            assert line, "§7 表里找不到 `%s` 行" % metric
            cells = line[0].split("|")
            assert len(cells) > cell_index, "§7 表结构变了：%s" % line[0]
            assert "无（曲线退化" not in cells[cell_index], (
                "§7 把 %s/%s 写成「无（曲线退化 / 门禁未过）」，"
                "但膝点表里有通过守卫的有效膝点 ⇒ 报告与产物自相矛盾" % (metric, tracker))


def test_report_same_family_grades_match_the_curve_guard():
    """🔴 §5.4 同族对照表的「两者曲线等级」必须按**与产品同一套守卫**逐臂判。

    上一版硬编码「A 级」，与 §4.2 的分级表、§5.3 的作废理由三方冲突。
    """
    report_path = HERE / "REPORT_V2_FINAL.md"
    knee_path = OUTPUT_DIR / "knee_thresholds_v2.csv"
    if not report_path.is_file() or not knee_path.is_file():
        pytest.skip("还没生成报告 / 膝点表")
    rows = _read_csv_rows(knee_path)
    section = _report_section(report_path.read_text(encoding="utf-8"), "### 5.4 ", "## 六、")
    checked = 0
    for line in section.splitlines():
        if not line.startswith("| blur |") and not line.startswith("| motion |") \
                and not line.startswith("| contrast |"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8 or not cells[2].startswith("`"):
            continue
        metric = cells[2].strip("`")
        family = cells[1]
        grade_cell = cells[-2]
        for tracker, label in (("kcf_legacy", "KCF"), ("csrt", "CSRT")):
            match = [r for r in rows if r["tracker"] == tracker and r["curve_set"] == "common_good"
                     and r["degradation_family"] == family and r["metric"] == metric
                     and r["stratum"] == "all"]
            assert match, "膝点表里找不到 %s/%s/%s" % (tracker, family, metric)
            row = match[0]
            guard_ok = (row.get("curve_monotone") == "1" and row.get("metric_monotone") == "1"
                        and _as_float(row.get("curve_y_span")) > CURVE_MIN_SPAN
                        and _as_float(row.get("level_y_span")) > CURVE_MIN_SPAN)
            claimed_a = "%s A 级" % label in grade_cell
            assert not (claimed_a and not guard_ok), (
                "§5.4 给 %s/%s/%s 标了「A 级」，但它的曲线没过守卫"
                "（分箱 %.4f / 逐档 %.4f）" % (label, family, metric,
                                          _as_float(row.get("curve_y_span")),
                                          _as_float(row.get("level_y_span"))))
            checked += 1
    assert checked, "§5.4 没解析到可比对的行（表结构变了？）"


def test_report_threshold_kind_marks_non_publishable_lines():
    """🔴 §8.3 汇总表的「阈值性质」不得把**不得单独对外**的线标成「红线」。

    判据：真实素材上的区分间隔（坏片拦截率 − 真实好片对照误杀率）< 50 pp 的线位，
    即使它在约束线表里有线位，也**不得**标成可对外的红线。
    """
    report_path = HERE / "REPORT_V2_FINAL.md"
    reg_path = OUTPUT_DIR / "regression_v2.csv"
    if not report_path.is_file() or not reg_path.is_file():
        pytest.skip("还没生成报告 / 回归表")
    reg = _read_csv_rows(reg_path)
    rates = {(r["set"], r["metric"]): 100.0 * _as_float(r.get("rate_below_line")) for r in reg}
    section = _report_section(report_path.read_text(encoding="utf-8"), "### 8.3 ", "### 8.4 ")
    checked = 0
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8:
            continue
        metric = cells[1].strip("`")
        if metric not in CORE_METRICS:
            continue
        kind = cells[-2]
        separation = rates.get(("real_bad", metric), float("nan")) \
            - rates.get(("real_good", metric), float("nan"))
        if separation < 50.0:
            assert "不得单独" in kind or "非红线" in kind, (
                "§8.3 把 %s 标成 %r，但它的真实素材区分间隔只有 %.2f pp ⇒ 不得标成对外红线"
                % (metric, kind, separation))
        checked += 1
    assert checked, "§8.3 没解析到可比对的行（表结构变了？）"


def test_regression_covers_three_way_sets():
    """**G4：回归表必须覆盖好片留出 / 一般 / 跟丢 / 真实坏片 / 真实好片五组**。"""
    path = OUTPUT_DIR / "regression_v2.csv"
    if not path.is_file():
        pytest.skip("还没生成回归表")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    sets = {row["set"] for row in rows}
    assert sets == {"good_holdout", "mediocre", "lost", "real_bad", "real_good"}
    metrics = {row["metric"] for row in rows}
    assert metrics == set(CORE_METRICS)


def test_degradation_worsens_tracking_cle_on_real_window(any_sequence_dir):
    """真实红外窗口上：重度模糊的跟踪误差 CLE_{t+4} 必须整体变差。

    口径：**多个真实窗口的中位数**（单窗口的亚像素级差异没有意义，
    实测单窗口 clean 0.75 px vs 重模糊 0.71 px 这种抖动会被中位数口径吸收）。
    """
    clean_cle, blur_cle = [], []
    for frame_id in range(0, 400, 40):
        window = real_ir_window(any_sequence_dir, frame_id)
        if window is None:
            continue
        frames, records = window
        first, last = records[0], records[-1]
        cx, cy, w, h = first.cx, first.cy, first.w, first.h
        init_box = (cx - max(16, w) / 2.0, cy - max(16, h) / 2.0, max(16, w), max(16, h))
        radius = int(max(16, round(1.5 * max(w, h))))
        target_center = (last.cx, last.cy)

        def cle_for(sigma, _frames=frames, _init=init_box, _radius=radius):
            degraded = [apply_degradation(frame, "blur", sigma) for frame in _frames]
            result = track_xcorr(degraded, _init, _radius)
            if not result["ok"]:
                return float("inf")
            return math.hypot(result["cx"] - target_center[0], result["cy"] - target_center[1])

        clean_cle.append(cle_for(0.0))
        blur_cle.append(cle_for(BLUR_SIGMAS[-1]))
    if len(clean_cle) < 3:
        pytest.skip("该序列可用窗口太少")
    clean_median = sorted(clean_cle)[len(clean_cle) // 2]
    blur_median = sorted(blur_cle)[len(blur_cle) // 2]
    assert clean_median < 20.0, "干净帧上链式 XCorr 必须跟得住（中位 CLE=%.2f px）" % clean_median
    assert blur_median > clean_median, (
        "重度模糊必须比干净帧跟得更差（中位 %.2f -> %.2f px，n=%d）"
        % (clean_median, blur_median, len(clean_cle)))


def _moving_target_frames(steps: int = 5, step_px: int = 8):
    """合成「目标逐帧平移」的红外序列（返回帧列表、首帧中心、末帧中心）。"""
    frames = []
    start = (120.0, 128.0)
    for index in range(steps):
        frames.append(synthetic_ir_frame(width=320, height=256,
                                         cx=start[0] + index * step_px, cy=start[1],
                                         target_w=24, target_h=20,
                                         noise_sigma=2.0, seed=100 + index))
    end = (start[0] + (steps - 1) * step_px, start[1])
    return frames, start, end


def test_legacy_kcf_update_lags_one_frame():
    """取证用例：contrib ``cv2.legacy.TrackerKCF`` 的 ``update()`` **滞后一帧**。

    实测：真值 120/128/136/144/152；``update(帧1..帧5)`` 依次返回 **120/128/136/144/152**
    ⇒ 返回的永远是**上一帧**的矩形。
    这条断言锁死该实现事实；若上游修好（不再滞后），本用例会失败 ⇒ 届时**必须**
    同步更新 ``run_eval_v2.LEGACY_KCF_NOTE`` 与报告里的「尾帧补偿」说明。
    """
    import cv2

    frames, start, _end = _moving_target_frames(steps=6)
    tracker = _kcf_tracker({"detect_thresh": 0.0})
    rx, ry, rw, rh = _clamp_rect(start[0], start[1], 24, 20, frames[0].shape[1], frames[0].shape[0])
    tracker.init(cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR), (rx, ry, rw, rh))
    observed = []
    for index in range(1, 6):
        _ok, rect = tracker.update(cv2.cvtColor(frames[index], cv2.COLOR_GRAY2BGR))
        observed.append(round(float(rect[0]) + float(rect[2]) / 2.0, 1))
    truth = [start[0] + 8.0 * k for k in range(5)]      # 帧 0..4 的真值 x
    assert observed == truth, (
        "legacy KCF 的 update() 不再滞后一帧（实测 %s vs 期望 %s）⇒ contrib 实现已变更，"
        "请同步更新 run_eval_v2.LEGACY_KCF_NOTE" % (observed, truth))


def test_legacy_kcf_tracks_moving_target():
    """任务书指定的 contrib legacy KCF：喂**尾帧**补偿滞后后，末位置必须落在真值附近。

    合成目标每帧平移 8 px；窗口 5 帧 + 1 帧尾帧 ⇒ 报出的应是「第 5 帧」的位置。
    这条断言同时锁死「初始化框必须精确落在目标本体上」这一实现事实：
    一旦初始化偏了半个框，KCF 会锁到目标局部、表现为固定偏差 ⇒ 用例失败。
    """
    frames, start, _end = _moving_target_frames(steps=6)
    init_box = (start[0] - 12.0, start[1] - 10.0, 24.0, 20.0)
    result = track_kcf_legacy(frames[:5], init_box, frames[5])
    end = (start[0] + 4 * 8.0, start[1])          # 期望报出「第 4 帧」的位置（窗口末帧）
    error = str(result.get("error", ""))
    if "TrackerKCF_create" in error:
        pytest.skip("当前环境没有 cv2.legacy.TrackerKCF_create：%s" % error)
    assert result["ok"], "legacy KCF 应当能初始化：%s" % error
    cle = math.hypot(result["cx"] - end[0], result["cy"] - end[1])
    assert cle <= 4.0, "legacy KCF 末位置偏离真值 %.2f px（真值位移 32 px）" % cle


def test_legacy_kcf_detect_thresh_high_collapses_to_origin():
    """``detect_thresh=0.99`` ⇒ 响应峰达不到阈值，update() 返回**退化的原点矩形**。

    这条断言记录的是「任务书为什么必须把 detect_thresh 置 0.0」的实现依据。
    实测：第 1 次 update 仍返回初始矩形，第 2 次起塌到 (0,0,0,0)。
    """
    import cv2

    frames, start, _end = _moving_target_frames(steps=6)
    tracker = _kcf_tracker({"detect_thresh": 0.99})
    rx, ry, rw, rh = _clamp_rect(start[0], start[1], 24, 20, frames[0].shape[1], frames[0].shape[0])
    tracker.init(cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR), (rx, ry, rw, rh))
    rects = []
    for index in range(1, 5):
        _ok, rect = tracker.update(cv2.cvtColor(frames[index], cv2.COLOR_GRAY2BGR))
        rects.append(rect)
    collapsed = any(rect[0] == 0 and rect[1] == 0 for rect in rects)
    assert collapsed, (
        "detect_thresh=0.99 时期望退化成原点矩形，实际 %s ⇒ contrib 实现已变更，"
        "请同步更新 run_eval_v2.LEGACY_KCF_NOTE" % (rects,))


def test_csrt_follows_moving_target():
    """CSRT 臂必须真的能跟（这是本环境里承担 DCF 家族角色的实现）。"""
    frames, start, end = _moving_target_frames()
    init_box = (start[0] - 12.0, start[1] - 10.0, 24.0, 20.0)
    result = track_csrt(frames, init_box)
    if not result.get("ok"):
        pytest.skip("CSRT 不可用：%s" % result.get("error", ""))
    cle = math.hypot(result["cx"] - end[0], result["cy"] - end[1])
    assert cle <= 12.0, "CSRT 末位置偏离真值 %.2f px（真值位移 32 px）" % cle


def test_conditions_cover_three_degradation_families():
    """**G1：三族退化、档位数量与任务书 §二.2 完全一致**（**基线剖面**档位保持不变）。

    深轮只是**在基线档位之上抬高上限**（新增 ``deep`` 剖面）⇒ 这里钉住的仍是
    已交付产物的**基线**口径：6 档模糊 / 5 档运动 / 6 档对比度，共 18 条。
    """
    conditions = build_conditions("baseline")
    families = {c.family for c in conditions}
    assert families == {"clean", "blur", "motion", "contrast"}
    assert len([c for c in conditions if c.family == "blur"]) == 6
    assert len([c for c in conditions if c.family == "motion"]) == 5
    assert len([c for c in conditions if c.family == "contrast"]) == 6
    assert len(conditions) == 18
    # 档位**取值**也钉死（不只是档数）：改动基线档位会让已交付产物失效
    assert [c.level for c in conditions if c.family == "blur"] == list(BLUR_SIGMAS)
    assert [c.level for c in conditions if c.family == "motion"] == list(MOTION_LENGTHS)
    assert [c.level for c in conditions if c.family == "contrast"] == list(CONTRAST_ALPHAS)
    assert CURVE_SETS == ("own_good", "common_good")


def test_default_conditions_are_the_frozen_baseline():
    """``build_conditions()``（无参）必须与基线剖面**逐条同标签、同顺序**。

    这是「深轮只加档位、不动基线」的机械保证：已交付产物的明细底表按
    ``condition_label`` 落库 ⇒ 标签序列一变，基线产物的可复现性就没了。
    """
    expected = [
        "clean",
        "blur_sigma0.6", "blur_sigma1.0", "blur_sigma1.5", "blur_sigma2.2",
        "blur_sigma3.0", "blur_sigma4.0",
        "motion_len3", "motion_len5", "motion_len7", "motion_len9", "motion_len11",
        "contrast_a0.80", "contrast_a0.65", "contrast_a0.50", "contrast_a0.35",
        "contrast_a0.25", "contrast_a0.15",
    ]
    assert [c.label for c in build_conditions()] == expected
    assert [c.label for c in build_conditions("baseline")] == expected
    assert len(build_conditions()) == 18


def test_deep_profile_has_26_conditions_in_expected_order():
    """深轮剖面 = clean 1 + 模糊 10 + 运动 7 + 对比度 8 = 26 条（上限抬到 8 / 15 / 0.05）。"""
    expected = [
        "clean",
        "blur_sigma0.6", "blur_sigma1.0", "blur_sigma1.5", "blur_sigma2.2", "blur_sigma3.0",
        "blur_sigma4.0", "blur_sigma5.0", "blur_sigma6.0", "blur_sigma7.0", "blur_sigma8.0",
        "motion_len3", "motion_len5", "motion_len7", "motion_len9", "motion_len11",
        "motion_len13", "motion_len15",
        "contrast_a0.80", "contrast_a0.65", "contrast_a0.50", "contrast_a0.35",
        "contrast_a0.25", "contrast_a0.15", "contrast_a0.10", "contrast_a0.05",
    ]
    conditions = build_conditions("deep")
    assert [c.label for c in conditions] == expected
    assert len(conditions) == 26
    assert len([c for c in conditions if c.family == "blur"]) == 10
    assert len([c for c in conditions if c.family == "motion"]) == 7
    assert len([c for c in conditions if c.family == "contrast"]) == 8


def test_deep_level_sets_are_strict_supersets_of_baseline():
    """深轮档位必须**严格包含**基线档位 ⇒ 两轮共用同一批低档位，衰减曲线可比。"""
    assert set(BLUR_SIGMAS) < set(DEEP_BLUR_SIGMAS)
    assert set(MOTION_LENGTHS) < set(DEEP_MOTION_LENGTHS)
    assert set(CONTRAST_ALPHAS) < set(DEEP_CONTRAST_ALPHAS)
    assert CONDITION_PROFILES["baseline"] == (BLUR_SIGMAS, MOTION_LENGTHS, CONTRAST_ALPHAS)
    assert CONDITION_PROFILES["deep"] == (DEEP_BLUR_SIGMAS, DEEP_MOTION_LENGTHS,
                                          DEEP_CONTRAST_ALPHAS)


def test_deep_families_are_strictly_monotonic():
    """深轮每一族内部仍必须单调（模糊/运动递增、对比度递减）⇒ 衰减曲线方向不会乱。"""
    conditions = build_conditions("deep")
    blur = [c.level for c in conditions if c.family == "blur"]
    motion = [c.level for c in conditions if c.family == "motion"]
    contrast = [c.level for c in conditions if c.family == "contrast"]
    assert all(b > a for a, b in zip(blur, blur[1:])), "模糊档位必须严格递增：%s" % blur
    assert all(b > a for a, b in zip(motion, motion[1:])), "运动档位必须严格递增：%s" % motion
    assert all(b < a for a, b in zip(contrast, contrast[1:])), "对比度档位必须严格递减：%s" % contrast


def test_unknown_conditions_profile_raises():
    """未知剖面名必须报错（不能悄悄退回基线 ⇒ 否则深轮会跑成基线轮且看不出）。"""
    with pytest.raises(ValueError):
        build_conditions("nope")


def test_level_overrides_replace_only_the_named_family():
    """逐族覆盖：被覆盖的族用传入档位，其余族仍取剖面自带的。"""
    conditions = build_conditions("deep", blur_sigmas=(1.0, 2.0))
    assert conditions[0].label == "clean"
    assert [c.label for c in conditions if c.family == "blur"] == ["blur_sigma1.0",
                                                                  "blur_sigma2.0"]
    assert len([c for c in conditions if c.family == "motion"]) == 7
    assert len([c for c in conditions if c.family == "contrast"]) == 8
    assert len(conditions) == 18


def test_invalid_level_sets_raise_value_error():
    """非法档位（空集 / 重复 / 方向错 / 越界 / 非正）必须全部报 ``ValueError``。"""
    with pytest.raises(ValueError):                     # 空档位集
        build_conditions("baseline", blur_sigmas=())
    with pytest.raises(ValueError):
        build_conditions("baseline", motion_lengths=())
    with pytest.raises(ValueError):
        build_conditions("baseline", contrast_alphas=())
    with pytest.raises(ValueError):                     # 同族内重复档位
        build_conditions("baseline", blur_sigmas=(1.0, 1.0))
    with pytest.raises(ValueError):
        build_conditions("baseline", motion_lengths=(5, 5))
    with pytest.raises(ValueError):
        build_conditions("baseline", contrast_alphas=(0.5, 0.5))
    with pytest.raises(ValueError):                     # 模糊未递增
        build_conditions("baseline", blur_sigmas=(2.0, 1.0))
    with pytest.raises(ValueError):                     # 运动未递增
        build_conditions("baseline", motion_lengths=(9, 7))
    with pytest.raises(ValueError):                     # 对比度未递减
        build_conditions("baseline", contrast_alphas=(0.35, 0.50))
    with pytest.raises(ValueError):                     # alpha = 0 不合法
        build_conditions("baseline", contrast_alphas=(0.5, 0.0))
    with pytest.raises(ValueError):                     # alpha > 1 不合法
        build_conditions("baseline", contrast_alphas=(0.5, 1.2))
    with pytest.raises(ValueError):                     # 模糊 sigma 必须 > 0
        build_conditions("baseline", blur_sigmas=(0.0, 1.0))
    with pytest.raises(ValueError):                     # 运动核长必须 > 0
        build_conditions("baseline", motion_lengths=(0, 3))


def test_degradation_strength_covers_every_deep_condition():
    """深轮新增档位不能把 ``degradation_strength`` 顶进「未知族」分支。

    （``clean`` 不属于任何退化族，``degradation_strength`` 对它报错是**设计如此**。）
    """
    conditions = build_conditions("deep")
    for condition in conditions:
        if condition.is_clean:
            continue
        strength = degradation_strength(condition.family, condition.level)
        assert isinstance(strength, float), "退化强度必须是浮点标量：%s" % condition.label
    for family in ("blur", "motion", "contrast"):
        strengths = [degradation_strength(c.family, c.level)
                     for c in conditions if c.family == family]
        assert all(b > a for a, b in zip(strengths, strengths[1:])), (
            "%s 族的退化强度必须逐档递增：%s" % (family, strengths))
    with pytest.raises(ValueError):
        degradation_strength("jpeg", 60.0)


def test_motion_angle_is_deterministic_and_in_range():
    a = motion_angle_for("seq:1", 42)
    b = motion_angle_for("seq:1", 42)
    c = motion_angle_for("seq:2", 42)
    assert a == b, "运动角度必须逐窗口确定性"
    assert 0.0 <= a < 180.0
    assert a != c, "不同窗口应有不同角度"


# --------------------------------------------------------------------------- 5. 产物 MD5
#: **冻结基线轮的代码镜像**（基线轮产物已冻结 ⇒ 代码也冻结一份，基线清单才能继续校验）。
BASELINE_CODE_DIR = HERE / "_baseline_code_20261126"

#: **冻结深轮的代码镜像**（与基线同理：深轮交付时把代码冻结一份，
#: 深轮清单才能在「工作区代码继续演进」之后**继续校验**）。
#: 🔴 没有它，清单里的 ``iqa_v2/*.py`` 条目会被拿去比**活代码**
#: ⇒ 任何一次后续改码都让这条门禁变红（假失败），门禁因此名存实亡。
DEEP_CODE_DIR = HERE / "_deep_code_20261127"

#: **基线轮冻结产物包**的根目录（服务器交付件，本地只读）。
FROZEN_BUNDLE = HERE / "server_out_final" / "files"

#: 冻结包**自己的**清单（≠ ``iqa_v2/output/manifest_v2.md5`` —— 后者是工作区产物目录，
#: 且是加深轮档位**之前**在服务器上生成的 ⇒ 拿它当门禁只会「合法地误报」）。
FROZEN_MANIFEST = FROZEN_BUNDLE / "output" / "manifest_v2.md5"

#: 冻结包内**逐条可校验**的条目数（清单 25 条 − 未随包交付的 3 条 = 21；
#: 第 25 条是服务器 cache 侧的大体量明细 ⇒ 本地不存在时跳过，不计入）。
FROZEN_VERIFIED_COUNT = 21

#: 冻结包内**没有**随包交付的登记名（服务器侧脚本 2 个 ＋ 后加的报告 1 个）。
FROZEN_NOT_DELIVERED = {"iqa_v2/recheck_curves.py", "iqa_v2/regression_v2.py",
                        "iqa_v2/REPORT_V2_ANTIUAV.md"}

#: 清单里唯一的**服务器侧绝对路径**登记项（大体量明细表，按「中间产物不入库」口径）。
FROZEN_BULK_ENTRY = "/tmp/iqa_v2_cache/degrade_detail.csv"


def _md5_of(path: Path) -> str:
    """文件 md5（与 ``md5sum`` 同口径）。

    Args:
        path (Path): 目标文件。

    Returns:
        str: 32 位十六进制摘要。
    """
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_manifest_entries() -> list[tuple[str, str]]:
    """把冻结清单解析成 ``[(摘要, 登记名), ...]``（跳过注释与空行）。

    Returns:
        list[tuple[str, str]]: ``(md5 摘要, 登记名)`` 列表；清单缺失 ⇒ 空列表。
    """
    if not FROZEN_MANIFEST.is_file():
        return []
    entries: list[tuple[str, str]] = []
    for line in FROZEN_MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if name.strip():
            entries.append((digest.strip(), name.strip()))
    return entries


def _frozen_bundle_target(name: str) -> Path | None:
    """把清单登记名（相对**仓库根**）映射到**冻结包内**的同名文件。

    Args:
        name (str): 清单登记名（如 ``iqa_v2/output/windows.csv``）。

    Returns:
        Path | None: 冻结包内的文件路径；该登记名不属于冻结包 ⇒ ``None``。
    """
    prefix = "iqa_v2/"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    if rest.startswith("output/"):
        return FROZEN_BUNDLE / "output" / rest[len("output/"):]
    if rest.endswith((".py", ".md")):
        return FROZEN_BUNDLE / rest
    return None


def _verify_manifest(manifest_path: Path, code_dir: Path | None = None):
    """核对一份 MD5 清单 ⇒ 返回 ``(通过条目, 跳过条目, 失败条目)``。

    口径（**刻意不用裸 ``md5sum -c``**）：

    * ``iqa_v2/*.py`` 代码条目 ⇒ 若给了 ``code_dir``（冻结基线代码镜像）就**在那里**核对
      （当前工作区的代码已经演进 ⇒ 用当前代码核对一份**历史**清单必然假失败）；
    * 其余仓库内条目（``iqa_v2/output/*``、报告正文）⇒ **就地核对**
      （这是「基线产物没被覆盖」的关键证据，规格书 C3）；
    * 仓库外的绝对路径（如 ``/tmp/iqa_v2_cache/degrade_detail.csv``）⇒ 本机没有就**跳过**
      （那是服务器侧的大体量明细，不是缺件问题）。

    Args:
        manifest_path (Path): 清单文件。
        code_dir (Path | None): 冻结代码镜像目录。

    Returns:
        tuple[list, list, list]: ``(通过, 跳过, 失败)`` 三组条目描述。
    """
    import hashlib

    ok, skipped, failed = [], [], []
    if not manifest_path.is_file():
        return ok, skipped, failed
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        want, name = parts[0], parts[1].strip()
        target = REPO / name
        if name.startswith("iqa_v2/") and name.endswith(".py") and code_dir is not None:
            frozen = code_dir / Path(name).name
            if frozen.is_file():
                target = frozen
        elif name.startswith("/"):
            if not target.is_file():
                skipped.append("%s（仓库外绝对路径，本机无此件）" % name)
                continue
        if not target.is_file():
            failed.append("%s（文件不存在）" % name)
            continue
        digest = hashlib.md5(target.read_bytes()).hexdigest()
        if digest == want:
            ok.append(name)
        else:
            failed.append("%s（md5 %s ≠ 清单 %s）" % (name, digest, want))
    return ok, skipped, failed


def test_output_manifest_md5_all_ok():
    """🔴 **产物 MD5 清单必须逐条对得上**（规格书 C4 的门禁之一）。

    基线轮的清单是**冻结证据链**：代码条目用 ``_baseline_code_20261126/`` 里的
    **冻结基线代码**核对（当前工作区代码已演进到 deep 轮，用它核对历史清单必然假失败），
    产物条目**就地核对**（⇒ 深轮若覆盖了基线产物，这里立刻红）。

    另一半证据是**已交付的冻结包** ``iqa_v2/server_out_final/files/`` 与**它自己的**清单
    ``iqa_v2/server_out_final/files/output/manifest_v2.md5``：把清单登记名（相对仓库根）
    映射回包内同名文件逐条比对 ⇒ **基线轮的冻结产物没被覆盖**。
    """
    manifest_path = OUTPUT_DIR / "manifest_v2.md5"
    if not manifest_path.is_file():
        pytest.skip("还没生成 %s" % manifest_path)
    ok, skipped, failed = _verify_manifest(manifest_path, BASELINE_CODE_DIR)
    assert not failed, "产物清单核对失败：%s" % failed
    assert ok, "清单一条都没核对上：%s" % manifest_path

    # ---- 冻结包 vs 它自己的清单（**就地**核对，证明冻结产物没被覆盖）
    if not FROZEN_MANIFEST.is_file():
        pytest.skip("没找到冻结清单：%s" % FROZEN_MANIFEST)
    entries = []
    for line in FROZEN_MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        entries.append((digest, name))
    assert len(entries) == 25, "冻结清单的条目数变了：%d" % len(entries)

    verified = 0
    not_delivered = []
    for digest, name in entries:
        if name.startswith("/"):
            # 服务器 cache 侧的大体量明细（中间产物）⇒ 本机没有就跳过，不算缺件。
            target = REPO / name
            if not target.is_file():
                not_delivered.append(name)
                continue
            assert _md5_of(target) == digest, "冻结的明细表被覆盖：%s" % target
            continue
        target = _frozen_bundle_target(name)
        if target is None or not target.is_file():
            not_delivered.append(name)
            continue
        actual = _md5_of(target)
        assert actual == digest, (
            "冻结产物被覆盖：%s（清单 %s，实测 %s）" % (target, digest, actual))
        verified += 1

    assert verified == FROZEN_VERIFIED_COUNT, (
        "冻结包内逐条对上的条目数变了：%d（应为 %d）" % (verified, FROZEN_VERIFIED_COUNT))
    # 未随包交付的 3 条 ＋ 唯一可能的服务器侧 cache 明细 ⇒ 允许跳过，其余一律不许跳。
    assert FROZEN_NOT_DELIVERED <= set(not_delivered), "跳过的登记名与预期不符：%s" % not_delivered
    assert set(not_delivered) - FROZEN_NOT_DELIVERED <= {FROZEN_BULK_ENTRY}, (
        "跳过的登记名与预期不符：%s" % not_delivered)


def test_deep_manifest_md5_all_ok():
    """🔴 **深轮清单**（若已生成）必须逐条对得上——深轮交付面的主门禁。"""
    manifest_path = HERE / "output_deep" / "manifest_deep.md5"
    if not manifest_path.is_file():
        pytest.skip("还没生成深轮清单 %s" % manifest_path)
    # 🔴 代码条目必须用**深轮冻结代码镜像**核对（与基线同理）：
    #    拿**活代码**核对一份历史清单 ⇒ 任何一次后续改码都让这条门禁变红（假失败）。
    ok, skipped, failed = _verify_manifest(manifest_path, DEEP_CODE_DIR)
    assert not failed, "深轮产物清单核对失败：%s" % failed
    assert ok, "深轮清单一条都没核对上：%s" % manifest_path


def test_baseline_products_are_not_overwritten_by_deep_round():
    """🔴 **规格书 C3：基线轮产物不得被覆盖** —— 深轮走独立目录。

    基线清单登记的**产物条目**（``iqa_v2/output/*.csv|json`` + 两份报告正文）
    必须仍与冻结清单一致 ⇒ 深轮一旦写进基线目录，这里立刻红。
    """
    # 🔴 核对基准是**冻结清单**（已交付的证据链），**不是**工作区那份可再生成的
    #    ``iqa_v2/output/manifest_v2.md5`` —— 后者会跟着产物一起被改写 ⇒
    #    「覆盖了基线产物 + 顺手重新生成了清单」这种场景会**假通过**
    #    （变异实验 M19：改 windows.csv 并同步改工作区清单 ⇒ 两条门禁全绿）。
    entries = _frozen_manifest_entries()
    if not entries:
        pytest.skip("没找到冻结清单：%s" % FROZEN_MANIFEST)
    checked = 0
    for want, name in entries:
        if name.startswith("/"):
            continue          # 服务器 cache 侧的大体量明细（中间产物），本机没有
        if name.startswith("iqa_v2/") and name.endswith(".py"):
            continue          # 代码条目：工作区代码已演进到 deep 轮（另由代码镜像核对）
        target = REPO / name
        assert target.is_file(), "基线产物被删了：%s" % name
        digest = _md5_of(target)
        assert digest == want, (
            "基线产物被覆盖了：%s（实测 %s ≠ 冻结清单 %s）" % (name, digest, want))
        checked += 1
    assert checked >= 13, "基线产物条目太少（%d），清单口径不对" % checked

    # ---- 再钉一条：冻结包内的副本 == 工作区副本（两者必须**同时**未被覆盖）----
    #    只对**产物/报告**钉：代码条目在工作区是**刻意演进**的（deep 轮），
    #    代码的不可变性由「冻结代码镜像 + 清单」另证（见 test_output_manifest_md5_all_ok）。
    for want, name in entries:
        if name.startswith("iqa_v2/") and name.endswith(".py"):
            continue
        bundle_target = _frozen_bundle_target(name)
        if bundle_target is None or not bundle_target.is_file():
            continue
        workspace_target = REPO / name
        if not workspace_target.is_file():
            continue
        assert _md5_of(workspace_target) == _md5_of(bundle_target), (
            "冻结包副本与工作区副本不一致：%s（交付件被换掉了）" % name)


#: 🔴 逐档单调性的**已知例外**（``族``, ``指标``）：对比度退化下 ``local_scr``
#: 的分子分母同乘 α ⇒ 比值几乎不变 ⇒ 数学上不保证单调（流水线对此判「该族不承载
#: 该指标的信息」并作废其膝点）。**只登记这一个**，出现别的例外一律红。
KNOWN_NON_MONOTONE = {("contrast", "local_scr")}

#: 已知例外的容忍上限（比值最多涨这么多倍仍算「只是数值噪声」）。
KNOWN_NON_MONOTONE_TOLERANCE = 1.01


def test_deep_new_levels_keep_metric_monotonicity():
    """🔴 **深轮新增档位**（σ5~8 / L13~15 / α0.10,0.05）也必须满足「退化越强 ⇒ 指标越低」。

    基线轮只验过基线档位；深轮把退化轴**延长**了，新增档位同样要落在单调曲线上
    ⇒ 否则「膝点反解」的前提（任务书 §四.2 的单调性假设）不成立。

    口径与流水线一致：逐档取**中位数**，允许 0.1% 数值松弛
    （``meds[i+1] <= meds[i] * 1.001``）；池 = 三分法好片段 ∩ 标定集（与流水线同口径）。
    """
    deep_dir = HERE / "output_deep"
    detail_path = deep_dir / "degrade_detail_v2.csv"
    if not detail_path.is_file():
        pytest.skip("还没有深轮明细：%s" % detail_path)
    split_path = deep_dir / "split_v2.csv"
    class_path = deep_dir / "segment_class_v2.csv"
    if not (split_path.is_file() and class_path.is_file()):
        pytest.skip("深轮切分/分类表不在：%s" % deep_dir)

    split_map = {r["seq_id"]: r["split"]
                 for r in csv.DictReader(open(split_path, encoding="utf-8"))}
    good_frames = {(r["seq_id"], int(r["frame_id"]))
                   for r in csv.DictReader(open(class_path, encoding="utf-8"))
                   if r.get("class_ds") == "good"}

    # ---- 逐族逐指标收集「池内逐档中位数」（一次遍历）----
    buckets: dict = {}
    seen_levels: dict = {}
    for row in csv.DictReader(open(detail_path, encoding="utf-8")):
        family = row.get("condition_family")
        if family not in ("blur", "motion", "contrast"):
            continue
        if split_map.get(row["seq_id"]) != "calib":
            continue
        if (row["seq_id"], int(row["frame_id"])) not in good_frames:
            continue
        level = round(float(row["condition_level"]), 6)
        seen_levels.setdefault(family, set()).add(level)
        for metric in CORE_METRICS:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            buckets.setdefault((family, metric), {}).setdefault(level, []).append(value)

    # ---- 新增档位必须真的跑过（否则本测试就是「什么都没验」）----
    expected_new = {"blur": set(DEEP_BLUR_SIGMAS) - set(BLUR_SIGMAS),
                    "motion": set(DEEP_MOTION_LENGTHS) - set(MOTION_LENGTHS),
                    "contrast": set(DEEP_CONTRAST_ALPHAS) - set(CONTRAST_ALPHAS)}
    for family, new_levels in expected_new.items():
        got = seen_levels.get(family, set())
        missing = sorted(level for level in new_levels
                         if not any(abs(level - g) <= 1e-6 for g in got))
        assert not missing, "%s 族的新增档位没有跑过：%s" % (family, missing)

    # ---- 逐档中位数必须随退化严重度**不增** ----
    #    🔴 唯一的**已知例外**是 ``contrast`` 族的 ``local_scr``：
    #       ``local_scr`` 的分子分母在对比度退化下**同乘 α**（再加同一个偏置）
    #       ⇒ 该比值数学上几乎不变 ⇒ 逐档中位数不是严格单调
    #       （实测 α 0.8 → 0.65 涨 0.35%）。流水线对这一组合**本来就判为
    #       「该族不承载该指标的信息」**并作废其膝点（指标单调性守卫）。
    #       ⇒ 这里把它**登记为例外**并给一个上限，保证它不会悄悄劣化成「真的反向了」；
    #       **任何其它组合出现反向 ⇒ 立刻红**。
    for (family, metric), by_level in sorted(buckets.items()):
        levels = sorted(by_level, key=lambda lv: (degradation_strength(family, lv), lv))
        meds = [float(np.median(by_level[lv])) for lv in levels]
        known = (family, metric) in KNOWN_NON_MONOTONE
        for index in range(len(meds) - 1):
            if known:
                assert meds[index + 1] <= meds[index] * KNOWN_NON_MONOTONE_TOLERANCE, (
                    "%s / %s 在档位 %g → %g 上劣化得太离谱（%.6g → %.6g）"
                    "⇒ 已知例外的容忍上限被突破"
                    % (family, metric, levels[index], levels[index + 1],
                       meds[index], meds[index + 1]))
                continue
            assert meds[index + 1] <= meds[index] * 1.001, (
                "%s / %s 在档位 %g → %g 上指标**反而变大**（%.6g → %.6g）"
                "⇒ 退化越强指标越低的单调性不成立"
                % (family, metric, levels[index], levels[index + 1],
                   meds[index], meds[index + 1]))


def test_small_target_threshold_constant():
    assert SMALL_TARGET_MAX_DIM == 20
    windows_path = OUTPUT_DIR / "windows.csv"
    if windows_path.is_file():
        rows = list(csv.DictReader(open(windows_path, encoding="utf-8")))
        for row in rows:
            expected = int(max(float(row["w"]), float(row["h"])) <= SMALL_TARGET_MAX_DIM)
            assert int(row["small_target"]) == expected, "小目标分层标注错误：%s" % row


# --------------------------------------------------------------------------- 6. 膝点曲线有效性守卫
def _read_csv_rows(path: Path) -> list[dict]:
    """读 CSV（``utf-8``）为 dict 列表。"""
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_float(text) -> float:
    """CSV 文本 → float（空串 / 非法值一律当 ``nan``）。"""
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _row_tag(row: dict) -> str:
    """膝点行的定位摘要（失败信息只放定位信息，不整行打印）。"""
    return "%s/%s/%s/%s/%s" % (row.get("tracker"), row.get("curve_set"),
                               row.get("degradation_family"), row.get("metric"),
                               row.get("stratum"))


def _synthetic_rows(family: str, levels, success_counts, metric_values, metric: str,
                    tracker: str = "xcorr", per_level: int = 8,
                    jitter: float = 7.0) -> list[dict]:
    """构造「逐档跟住率 + 逐档指标」的明细行（纯逻辑用例的输入）。

    ``success_counts`` = 每档里跟住的行数（跟住率 = 成功数 / 该档行数）；
    ``metric_values`` = 每档的指标基准值（叠加 ``jitter`` 线性扰动 ⇒ 档内取值分散，
    ``binned_curve`` 才有足够分箱 ⇒ 不会因「不足 4 箱」直接吐 nan 膝点）。
    """
    rows = []
    for level, ok_count, metric_value in zip(levels, success_counts, metric_values):
        for index in range(per_level):
            rows.append({
                "condition_family": family,
                "condition_level": float(level),
                "%s_success" % tracker: 1 if index < ok_count else 0,
                metric: float(metric_value) + (index - per_level / 2.0) * jitter,
            })
    return rows


def test_degenerate_curves_are_rejected():
    """**膝点表：曲线有效性守卫必须生效**（平曲线 / 非单调 / 指标非单调 ⇒ degenerate）。

    v2 膝点反解的契约：
    * ``method_used == "degenerate"`` ⇒ ``knee_reliable == 0`` 且线位置（``knee_x_final``）非有限；
    * ``knee_reliable == 1`` ⇒ 该曲线必已过守卫：跟住率单调 + 指标单调 + 跨度 > ``CURVE_MIN_SPAN``。
    """
    path = OUTPUT_DIR / "knee_thresholds_v2.csv"
    if not path.is_file():
        pytest.skip("还没生成膝点表 %s" % path)
    rows = _read_csv_rows(path)
    assert rows, "膝点表为空"
    required = {"method_used", "knee_reliable", "knee_x_final", "curve_y_span",
                "curve_monotone", "metric_monotone"}
    missing = required - set(rows[0].keys())
    assert not missing, "膝点表缺少曲线有效性守卫列（产物未重新生成）：%s" % sorted(missing)
    for row in rows:
        if row["method_used"] == "degenerate":
            assert row["knee_reliable"] == "0", (
                "退化曲线的膝点必须判为不可靠（knee_reliable=0）：%s" % _row_tag(row))
            assert not math.isfinite(_as_float(row["knee_x_final"])), (
                "退化曲线的膝点位置必须是非有限值（nan）：%s" % _row_tag(row))
        if row["knee_reliable"] == "1":
            assert row["curve_monotone"] == "1", (
                "可靠膝点的跟住率曲线必须单调下降：%s" % _row_tag(row))
            assert row["metric_monotone"] == "1", (
                "可靠膝点的指标曲线必须单调下降：%s" % _row_tag(row))
            assert _as_float(row["curve_y_span"]) > CURVE_MIN_SPAN, (
                "可靠膝点的跟住率跨度必须 > %.2f：%s" % (CURVE_MIN_SPAN, _row_tag(row)))


def test_constraint_lines_only_from_valid_curves():
    """**约束红线只能来自「过了守卫」的膝点**（跨度 > 5pp + 跟住率单调 + 指标单调 + 双算法一致）。"""
    line_path = OUTPUT_DIR / "constraint_lines_v2.csv"
    knee_path = OUTPUT_DIR / "knee_thresholds_v2.csv"
    if not (line_path.is_file() and knee_path.is_file()):
        pytest.skip("还没生成约束线表 / 膝点表")
    knees = _read_csv_rows(knee_path)
    lines = _read_csv_rows(line_path)
    checked = 0
    for line in lines:
        if not math.isfinite(_as_float(line["line_value"])):
            continue                       # 没有可靠膝点的指标 ⇒ 线位置 nan，不参与校验
        checked += 1
        matches = [row for row in knees
                   if row["tracker"] == line["deciding_tracker"]
                   and row["degradation_family"] == line["deciding_family"]
                   and row["metric"] == line["metric"]
                   and row["curve_set"] == "common_good"
                   and row["stratum"] == "all"]
        assert matches, ("红线 %s 在膝点表里找不到来源行（决定臂 %s / 决定族 %s）"
                         % (line["metric"], line["deciding_tracker"], line["deciding_family"]))
        for row in matches:
            assert row["knee_reliable"] == "1", (
                "红线 %s 的来源膝点不可靠：%s" % (line["metric"], _row_tag(row)))
            assert row["curve_monotone"] == "1", (
                "红线 %s 的来源膝点跟住率曲线非单调：%s" % (line["metric"], _row_tag(row)))
            assert row["metric_monotone"] == "1", (
                "红线 %s 的来源膝点指标曲线非单调：%s" % (line["metric"], _row_tag(row)))
            assert _as_float(row["curve_y_span"]) > CURVE_MIN_SPAN, (
                "红线 %s 的来源膝点跨度不足 %.2f：%s"
                % (line["metric"], CURVE_MIN_SPAN, _row_tag(row)))
    assert checked, "约束线表里没有一条有限红线（产物未重新生成？）"


def test_level_success_profile_accepts_monotone_decay_curve():
    """**逐档跟住率曲线（退化强度递增）**：单调下降且跨度 > 5pp ⇒ 守卫必须放行。"""
    rows = _synthetic_rows("blur", BLUR_SIGMAS, [8, 7, 6, 5, 4, 2],
                            [1200, 950, 700, 480, 260, 90], "target_lap_var")
    profile = level_success_profile(rows, "blur", "xcorr")
    assert profile["monotone"] == 1, "随退化单调下降的跟住率曲线必须判单调：%s" % profile["rates"]
    assert abs(profile["span"] - 0.75) < 1e-9, (
        "跟住率跨度算错（期望 1.00-0.25=0.75）：%s" % profile["span"])
    result = extract_knee(rows, "target_lap_var", "xcorr", "blur")
    assert result["curve_monotone"] == 1 and result["metric_monotone"] == 1
    assert result["method_used"] != "degenerate", "有效曲线不得被判为 degenerate"


def test_flat_success_curve_is_degenerate():
    """**完全平坦的跟住率曲线（跨度 0）⇒ 膝点必须作废**。

    平曲线上双算法（Kneedle / PCHIP）必然落在同一个噪声箱上 ⇒ 「双算法一致」会**误判**
    为可靠 ⇒ 膝点没有物理意义，必须走 degenerate 分支（线位置 nan）。
    """
    rows = _synthetic_rows("blur", BLUR_SIGMAS, [8, 8, 8, 8, 8, 8],
                            [1200, 950, 700, 480, 260, 90], "target_lap_var")
    profile = level_success_profile(rows, "blur", "xcorr")
    assert profile["span"] == 0.0, "平坦曲线的跨度必须是 0：%s" % profile["rates"]
    result = extract_knee(rows, "target_lap_var", "xcorr", "blur")
    assert result["method_used"] == "degenerate", (
        "平坦曲线必须判为 degenerate（实际 %s）" % result["method_used"])
    assert result["knee_reliable"] == 0, "退化膝点必须判为不可靠"
    assert not math.isfinite(result["knee_x_final"]), "退化膝点的线位置必须是非有限值"


def test_level_metric_profile_flags_contrast_local_scr_non_monotone():
    """**contrast 族的 ``local_scr`` 不承载退化信息 ⇒ 指标侧单调性必须判它非单调**。

    物理机理：``local_scr = P90(|core-μ_ring|) / (σ_ring + ε)`` 是**比值**，分子分母
    在纯线性对比度压缩下同乘 α ⇒ 数学上几乎不变；但 8bit 量化噪声（σ ≈ 0.29 LSB）
    **不随 α 缩小** ⇒ 分母被相对抬高 ⇒ 随退化加深 ``local_scr`` 反而乱走（实测 ±2~6%）
    ⇒ 「退化越强 ⇒ 指标越低」的单调性假设不成立 ⇒ 该族该指标的膝点没有物理意义。
    """
    frame = synthetic_ir_frame()
    values = [compute_roi_metrics(apply_degradation(frame, "contrast", alpha),
                                   320.0, 256.0, 20, 16)["local_scr"]
              for alpha in CONTRAST_ALPHAS]
    rows = _synthetic_rows("contrast", CONTRAST_ALPHAS, [8, 8, 7, 6, 4, 2], values,
                            "local_scr", jitter=0.05)
    profile = level_metric_profile(rows, "contrast", "local_scr")
    assert profile["monotone"] == 0, (
        "contrast 族的 local_scr 必须判为非单调（实测逐档中位 %s）" % profile["meds"])
    result = extract_knee(rows, "local_scr", "xcorr", "contrast")
    assert result["metric_monotone"] == 0, "extract_knee 必须同步标出指标非单调"
    assert result["method_used"] == "degenerate", "指标非单调 ⇒ 膝点必须作废"


# --------------------------------------------------------------------------- 12. 产物清单工具
def test_manifest_tool_supports_a_separate_products_dir(tmp_path):
    """🔴 深轮走**独立产物目录** ⇒ 清单工具必须能只登记该目录，**不动基线清单**。

    这是规格书 C3「基线轮产物不得被覆盖」的**工具侧保证**：深轮的
    ``--products-subdir iqa_v2/output_deep`` 必须只收深轮产物，
    不能把基线 ``iqa_v2/output/`` 的文件混进来。
    """
    import make_manifest_v2

    repo = tmp_path / "repo"
    (repo / "iqa_v2" / "output").mkdir(parents=True)
    (repo / "iqa_v2" / "output_deep").mkdir(parents=True)
    (repo / "iqa_v2" / "output" / "base.csv").write_text("base\n", encoding="utf-8")
    (repo / "iqa_v2" / "output_deep" / "deep.csv").write_text("deep\n", encoding="utf-8")
    (repo / "iqa_v2" / "REPORT_V2_FINAL.md").write_text("base report\n", encoding="utf-8")
    (repo / "iqa_v2" / "REPORT_V2_DEEP.md").write_text("deep report\n", encoding="utf-8")

    base_items = make_manifest_v2.collect(repo, tmp_path / "missing.csv")
    base_names = {name for name, _ in base_items}
    assert "iqa_v2/output/base.csv" in base_names
    assert "iqa_v2/output_deep/deep.csv" not in base_names, (
        "基线口径的清单**不得**收进深轮产物")

    deep_items = make_manifest_v2.collect(repo, tmp_path / "missing.csv",
                                          products_subdir="iqa_v2/output_deep",
                                          report_files=("iqa_v2/REPORT_V2_DEEP.md",))
    deep_names = {name for name, _ in deep_items}
    assert "iqa_v2/output_deep/deep.csv" in deep_names
    assert "iqa_v2/output/base.csv" not in deep_names, (
        "深轮口径的清单**不得**收进基线产物")
    assert "iqa_v2/REPORT_V2_DEEP.md" in deep_names
    assert "iqa_v2/REPORT_V2_FINAL.md" not in deep_names


def test_manifest_tool_default_collect_is_byte_identical_to_baseline(tmp_path):
    """默认参数下清单收集口径必须与基线轮**逐项一致**（不许偷偷改登记范围）。"""
    import make_manifest_v2

    repo = tmp_path / "repo"
    (repo / "iqa_v2" / "output").mkdir(parents=True)
    (repo / "iqa_v2" / "output" / "a.csv").write_text("a\n", encoding="utf-8")
    (repo / "iqa_v2" / "output" / "b.json").write_text("{}\n", encoding="utf-8")
    (repo / "iqa_v2" / "REPORT_V2_FINAL.md").write_text("r\n", encoding="utf-8")
    (repo / "iqa_v2" / "REPORT_V2_ANTIUAV.md").write_text("r2\n", encoding="utf-8")
    (repo / "iqa_v2" / "MEMORY_SECTION.md").write_text("m\n", encoding="utf-8")
    bulk = tmp_path / "bulk.csv"
    bulk.write_text("b\n", encoding="utf-8")

    names = [name for name, _ in make_manifest_v2.collect(repo, bulk)]
    assert names == [
        "iqa_v2/output/a.csv",
        "iqa_v2/output/b.json",
        "iqa_v2/REPORT_V2_FINAL.md",
        "iqa_v2/REPORT_V2_ANTIUAV.md",
        "iqa_v2/MEMORY_SECTION.md",
        str(bulk),
    ], "默认登记范围被改动了：%s" % names


# --------------------------------------------------------------------------- 13. 冻结铁律与剖面纪律闸门
def test_frozen_guard_constants_are_unchanged():
    """🔴 **规格书 B1：守卫门槛不许动** —— 本轮只允许加档位，不许放松验收。

    这些常量一旦被改小，整套「曲线有效性守卫」就会失效（平曲线上的噪声会被当成
    膝点）⇒ 必须由**测试钉死**，改小它们 ⇒ 测试立刻红。
    """
    from regression_v2 import CALIB_RATIO, GOOD_CLE_PX, SPLIT_SEED, tolerance_for

    assert CURVE_MIN_SPAN == 0.05, "守卫门槛 CURVE_MIN_SPAN 被改动（规格书 B1）"
    assert KNEE_AGREE_PCT == 15.0, "双算法一致性门禁 KNEE_AGREE_PCT 被改动（规格书 B1）"
    assert N_BINS == 12, "分箱数 N_BINS 被改动（规格书 B1）"
    assert SPLIT_SEED == 20261126, "序列级切分种子被改动（规格书 B3）"
    assert CALIB_RATIO == 0.7, "标定/留出比例被改动（规格书 B3）"
    assert GOOD_CLE_PX == 3.0, "三分法「好片段」判据被改动（规格书 §三）"
    assert tolerance_for(10, 10) == 5.0, "跟住容差公式被改动（规格书 §二.2）"
    assert tolerance_for(40, 30) == 20.0, "跟住容差公式被改动（规格书 §二.2）"


def test_profile_ladder_guard_rejects_axis_weakening():
    """🔴 **剖面纪律闸门**：baseline 剖面不许改轴；deep 剖面只许**加**档位（不许减）。"""
    from run_eval_v2 import _guard_profile_ladder

    assert _guard_profile_ladder("baseline", build_conditions("baseline")) is None
    assert _guard_profile_ladder("deep", build_conditions("deep")) is None

    # deep 剖面把 blur 换成**比基线更少**的档位 ⇒ 必须被拒（否则两轮不再逐档可比）
    weakened = build_conditions("deep", blur_sigmas=(1.0, 2.0))
    problem = _guard_profile_ladder("deep", weakened)
    assert problem, "deep 剖面减档位必须被闸门拒绝（规格书 §1.1 严格超集不变量）"
    assert "严格超集" in problem

    # baseline 剖面改轴 ⇒ 必须被拒（基线轮产物已冻结）
    problem = _guard_profile_ladder("baseline", build_conditions("baseline",
                                                                 blur_sigmas=(1.0, 2.0)))
    assert problem, "baseline 剖面改轴必须被闸门拒绝（基线轮产物已冻结）"


def test_deep_profile_cannot_write_into_baseline_product_paths():
    """🔴 **规格书 C3：基线轮产物不得被覆盖** —— deep 剖面**硬拒**基线绑定路径。

    这是 BLOCKING-1 的回归测试：``--conditions-profile deep`` 只换档位、**不换目录**，
    若没有闸门，天真的调用会把冻结的基线产物、明细 cache、基线报告正文全部覆写。
    """
    import subprocess

    for flag, value in (("--output-dir", "iqa_v2/output"),
                        ("--cache-dir", "/tmp/iqa_v2_cache"),
                        ("--report", "iqa_v2/REPORT_V2_FINAL.md")):
        result = subprocess.run(
            [sys.executable, str(HERE / "run_eval_v2.py"), "--phase", "report",
             "--conditions-profile", "deep", flag, value],
            cwd=str(REPO), capture_output=True, text=True)
        assert result.returncode != 0, "deep 剖面竟然允许写基线路径 %s" % value
        assert "基线" in (result.stderr + result.stdout), (
            "拒绝信息里必须点明「基线」：%s" % (result.stderr + result.stdout)[-500:])


def test_acceptance_c3_checks_code_entries_against_the_frozen_mirror():
    """🔴 **C3 的代码条目必须用「冻结基线代码镜像」核对**（不能拿活代码比历史清单）。

    deep 轮**刻意**改动了基线代码（``run_eval_v2.py`` / ``make_manifest_v2.py`` /
    ``audit_final.py`` …）⇒ 拿工作区**活代码**去核对**冻结**清单必然假失败 ⇒
    C3 被永久判成 PARTIAL ⇒ 这条门禁名存实亡（它本该只管「基线**产物**有没有被覆盖」）。

    正确口径与 ``test_output_manifest_md5_all_ok`` 一致：
    **代码条目**在 ``iqa_v2/_baseline_code_20261126/`` 里核对，
    **产物条目**就地核对（deep 覆盖基线产物 ⇒ 立刻红）。

    本测试同时验证这条口径**有检出能力**（冻结镜像被篡改 ⇒ 判定转红），
    不是恒真的形式检查。
    """
    import acceptance_deep

    class _Stub:
        """``check_c3`` 只用到 ``deep_dir`` / ``baseline_dir`` / ``repo_root`` /
        ``baseline_code_dir``。"""

        def __init__(self, deep_dir, baseline_dir, baseline_code_dir):
            self.deep_dir = deep_dir
            self.baseline_dir = baseline_dir
            self.repo_root = str(REPO)
            self.baseline_code_dir = baseline_code_dir

    def _ctx(code_dir):
        return _Stub(str(HERE / "output_deep"), str(OUTPUT_DIR), str(code_dir))

    verdict = acceptance_deep.check_c3(_ctx(BASELINE_CODE_DIR))
    assert verdict.status == acceptance_deep.PASS, (
        "C3 在有冻结代码镜像时不该报假失败：%s" % verdict.evidence)
    # 产物条目必须**就地**核对（deep 若覆盖基线产物，这里立刻红）
    assert any("iqa_v2/output/" in line for line in verdict.detail)

    # ---- 检出能力：篡改镜像里的一个代码文件 ⇒ 判定必须转红 ----
    with tempfile.TemporaryDirectory() as tmp:
        tampered = Path(tmp)
        for path in BASELINE_CODE_DIR.glob("*.py"):
            shutil.copy2(path, tampered / path.name)
        target = tampered / "run_eval_v2.py"
        target.write_text(target.read_text(encoding="utf-8") + "\n# tampered\n",
                          encoding="utf-8")
        bad = acceptance_deep.check_c3(_ctx(tampered))
        assert bad.status != acceptance_deep.PASS, (
            "冻结代码镜像被篡改时 C3 必须转红（当前 %s）" % bad.status)
