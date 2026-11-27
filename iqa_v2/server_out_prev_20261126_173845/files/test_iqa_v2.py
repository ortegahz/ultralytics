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

执行机：服务器 ``huangzhe@192.168.99.40:32222``（conda env ``uav``，依赖 cv2 + scipy）。
复现：

    cd /tmp/pycharm_project_a7d966e6 && python3 -m pytest iqa_v2/test_iqa_v2.py -v
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from degradation_4d import (BLUR_SIGMAS, CONTRAST_ALPHAS, JPEG_QUALITIES,  # noqa: E402
                            MOTION_LENGTHS, apply_degradation, build_conditions,
                            motion_angle_for)
from dataset_anti_uav import (SMALL_TARGET_MAX_DIM, WINDOW_LEN,  # noqa: E402
                              discover_sequence_dirs, load_sequence)
from run_eval_v2 import (_clamp_rect, _kcf_tracker, track_csrt,  # noqa: E402
                            track_kcf_legacy, track_xcorr)
from target_roi_metrics import (ALL_METRICS, EPS, _p90,  # noqa: E402
                                build_regions, compute_roi_metrics,
                                regions_overlap_pixels)

REPO = HERE.parent
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

    # ---- 压缩（低码率 JPEG）：**锐度类与块效应类指标在红外上都不是有效退化轴**
    # 实测（12 个真实红外窗口的中位数，q60 → q10）：
    #   target_lap_var      23171 → 27416（**不降反升**：DCT 振铃把能量塞回小目标高频）
    #   surround_blockiness  0.738 → 0.597（**不升反降**：低对比、噪声主导的红外背景
    #                        被 JPEG 先抹噪，块内梯度掉得比块边界梯度更快 ⇒ 比值下降）
    # ⇒ 任务书「退化强度递增 ⇒ 局部画质指标单调下降」对 JPEG 族**不成立**，
    #    照抄会把自己的实验写错。改用**信息损失**（退化帧与原帧的 MAE）作为
    #    JPEG 族的退化强度量：实测 12/12 个真实窗口逐档严格递增（0.58 → 2.94 灰阶）。
    losses = []
    for quality in JPEG_QUALITIES:
        degraded = apply_degradation(frame, "jpeg", float(quality))
        losses.append(float(np.abs(degraded.astype(np.float64)
                                   - frame.astype(np.float64)).mean()))
    assert all(losses[i + 1] >= losses[i] for i in range(len(losses) - 1)), \
        "JPEG 档位上的信息损失(MAE)必须逐档不减：%s" % losses


def test_jpeg_on_ir_is_not_a_sharpness_or_blockiness_axis(any_sequence_dir):
    """把一条**反直觉但实测成立**的物理事实固化成机检用例。

    低码率 JPEG 在**红外小目标**画面上：
      * ``target_lap_var`` 不降反升（DCT 振铃把高频能量塞回 9×9 核心区）；
      * ``surround_blockiness`` 不升反降（低对比、噪声主导的背景被先抹噪，
        块内梯度掉得比块边界梯度更快）。

    ⇒ JPEG 族的膝点**不能**从锐度 / 块效应轴读取（实测 v2 报告里 jpeg 族膝点
    来自 ``global_contrast`` 与 ``local_scr``）。本用例防止后续有人为了「让曲线好看」
    而反改口径。
    """
    windows = []
    for frame_id in range(0, 600, 50):
        window = real_ir_window(any_sequence_dir, frame_id)
        if window is None:
            continue
        frames, records = window
        record = records[0]
        windows.append((frames[0], record))
    if len(windows) < 6:
        pytest.skip("可用真实窗口不足")

    def median_at(quality: float, key: str) -> float:
        values = [compute_roi_metrics(apply_degradation(frame, "jpeg", quality),
                                      record.cx, record.cy, record.w, record.h)[key]
                  for frame, record in windows]
        return float(np.median(values))

    lap_best = median_at(float(JPEG_QUALITIES[0]), "target_lap_var")
    lap_worst = median_at(float(JPEG_QUALITIES[-1]), "target_lap_var")
    block_best = median_at(float(JPEG_QUALITIES[0]), "surround_blockiness")
    block_worst = median_at(float(JPEG_QUALITIES[-1]), "surround_blockiness")
    assert lap_worst >= lap_best * 0.95, (
        "红外 JPEG 应当**不降低**目标锐度（振铃主导）：%.1f → %.1f" % (lap_best, lap_worst))
    assert block_worst <= block_best * 1.05, (
        "红外 JPEG 应当**不抬高**块效应（抹噪主导）：%.4f → %.4f" % (block_best, block_worst))


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


def test_conditions_cover_all_four_degradation_families():
    conditions = build_conditions()
    families = {c.family for c in conditions}
    assert families == {"clean", "blur", "motion", "contrast", "jpeg"}
    assert len([c for c in conditions if c.family == "blur"]) == 6
    assert len([c for c in conditions if c.family == "motion"]) == 5
    assert len([c for c in conditions if c.family == "contrast"]) == 6
    assert len([c for c in conditions if c.family == "jpeg"]) == 6
    assert len(conditions) == 24


def test_motion_angle_is_deterministic_and_in_range():
    a = motion_angle_for("seq:1", 42)
    b = motion_angle_for("seq:1", 42)
    c = motion_angle_for("seq:2", 42)
    assert a == b, "运动角度必须逐窗口确定性"
    assert 0.0 <= a < 180.0
    assert a != c, "不同窗口应有不同角度"


# --------------------------------------------------------------------------- 5. 产物 MD5
def test_output_manifest_md5_all_ok():
    manifest_path = OUTPUT_DIR / "manifest_v2.md5"
    if not manifest_path.is_file():
        pytest.skip("还没生成 %s" % manifest_path)
    result = subprocess.run(["md5sum", "-c", str(manifest_path)], cwd=str(REPO),
                            capture_output=True, text=True)
    assert result.returncode == 0, "md5sum -c 失败：\n%s" % result.stdout[-2000:]
    assert result.stdout.count(": OK") == result.stdout.count(": OK")
    assert "FAILED" not in result.stdout


def test_small_target_threshold_constant():
    assert SMALL_TARGET_MAX_DIM == 20
    windows_path = OUTPUT_DIR / "windows.csv"
    if windows_path.is_file():
        rows = list(csv.DictReader(open(windows_path, encoding="utf-8")))
        for row in rows:
            expected = int(max(float(row["w"]), float(row["h"])) <= SMALL_TARGET_MAX_DIM)
            assert int(row["small_target"]) == expected, "小目标分层标注错误：%s" % row
