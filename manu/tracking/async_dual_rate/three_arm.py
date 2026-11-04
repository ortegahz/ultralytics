#!/usr/bin/env python3
"""三臂对照验收表（臂 1 纯检测 / 臂 2 纯前向外推 / 臂 3 退回重推演）。

**这是用户拍板的那张验收表的生产者。** 三个臂的定义（不再改动）：

======================== ==============================================
臂                       机制
======================== ==============================================
臂 1 纯检测基准          不跟踪。第 t 帧直接甩第 t−Δ 帧算出来的旧检测框
                         （det_hz=25、Δ=8 ⇒ 每帧都有输出，静态硬吃 320 ms）
臂 2 检跟纯前向外推      快核每帧前推；迟到检测**只对比、不回溯、不修正**，
                         且**永不**用检测重新锚定位置
臂 3 异步退回重推演      快核 + 帧环形缓存；迟到检测作绝对锚点，退回第 t−Δ 帧
                         逐帧重推 Δ−1 步，由主流程走完第 t 帧，覆盖当前状态；
                         **不加任何人为平滑衰减**
======================== ==============================================

**时序口径**（不能用常规单帧 mAP/F1）：

* 物理时刻 ``t`` 的输出 ``(x̂_t, ŷ_t)`` 直接对账 ``GT_t``；
* 后台检测器**只能**给 ``Det_{t−Δ}``（Δ=8 ≈ 320 ms @25 Hz）。

**四项核心指标**（与验收契约一一对应）：

1. CLE：p50 / p95 / p99 / max（合格线 p50 ≤ 1.0 px、p95 ≤ 3.0 px）；
2. Hit Rate @T（T=5 px）+ **Wilson 95% CI**；另给冻结 inbox 口径；
3. 脱靶帧数 / 断裂次数 / 自持周期；
4. 抖动：一阶差分与二阶差分 ``Δv = ‖(p_t−p_{t−1}) − (p_{t−1}−p_{t−2})‖``，
   **按稳态 / 状态机切换 / 硬拉回分开归因**（不拆会被冷启动跳变吃掉）。

用法::

    cd /media/manu/1TB-Volume/workspace/ultralytics
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate uav
    PYTHONPATH=. python -m manu.tracking.async_dual_rate.three_arm \\
        --sequence VIDEO00007_19700101_001324 \\
        --out-dir runs/tracking_async
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .arms import (
    ArmSpec,
    JpegFrameSource,
    SlowStream,
    arm1_detector_only_sparse,
    run_lockstep,
)
from .servo_metrics import mcnemar_exact, percentiles, servo_metrics, wilson_interval
from .tracker import TrackerConfig

__all__ = ["run_three_arm", "build_acceptance_table", "main"]


# --------------------------------------------------------------------------
# 默认路径（只读挂载的实验数据）
# --------------------------------------------------------------------------

DEFAULT_CACHE_ROOT = "/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"
DEFAULT_FRAMES_ROOT = "/home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
DEFAULT_LABELS_ROOT = "/home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"


# ==============================================================================
# 辅助统计
# ==============================================================================


def _gt_centres(ground_truth: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {
        int(frame): np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[0, :2]
        for frame, boxes in ground_truth.items()
    }


def _anchor_ceiling(
    stream: SlowStream, centres: dict[int, np.ndarray], delta: int
) -> dict:
    """臂 3 的精度**天花板**由两件事决定，量化出来：

    1. **锚点误差**：检测框在它自己那一帧的定位误差 ``‖Det_f − GT_f‖``。
       退回重推演消不掉它——重推只是把"累积漂移"清零，锚点本身的偏差还在。
    2. **Δ 帧内的目标位移** ``‖GT_{f+Δ} − GT_f‖``**：即使锚点完美、
       重推演零误差，输出也会因为目标在这 Δ 帧里真的动了而偏离 ``GT_t``。

    这两条合起来给出臂 3 的**理论下限**（Ground-Truth Parity 的真实含义）。
    """
    anchor_errors: list[float] = []
    for frame_index in sorted(centres):
        detection = stream.best(frame_index)
        if detection is None:
            continue
        anchor_errors.append(
            float(np.linalg.norm(np.asarray(detection.xy) - centres[frame_index]))
        )

    displacement: list[float] = []
    for frame_index in sorted(centres):
        later = centres.get(frame_index + delta)
        if later is None:
            continue
        displacement.append(float(np.linalg.norm(later - centres[frame_index])))

    return {
        "delta_frames": int(delta),
        "anchor_error_px": percentiles(anchor_errors, points=(50, 95, 99)),
        "gt_displacement_over_delta_px": percentiles(displacement, points=(50, 95, 99)),
        "note": (
            "臂 3 的误差下界 ≈ 锚点误差 ⊕ Δ 帧内目标位移；"
            "重推演消掉的是'累积漂移'，消不掉这两项。"
        ),
    }


def _kernel_ceiling(source, centres: dict[int, np.ndarray], delta: int) -> dict:
    """**臂 3 的理论精度天花板**（与本 harness 的实现无关，只取决于快核本身）。

    做法：把**真值**当作第 f 帧的锚点，走 1 步、Δ−1 步、Δ 步，看误差多大。

    * ``single_step_error_px``：快核的单步局部误差——这是快核能力的直接度量；
    * ``delta_step_error_px``：从完美锚点连续重推 Δ 步的累积误差——**这就是
      "退回重推演"能做到的最好结果**（实际运行还要叠加锚点自身的定位误差）。

    没有这两个数，就分不清"臂 3 效果不好"是**实现问题**还是**内核能力上限**。
    """
    from .fast_tracker import LocalPatchTracker

    kernel = LocalPatchTracker(
        win_size=32, th_k=3.0, min_energy=1.0, min_contrast=5.0, mode="centroid", bg_window=64
    )
    frames = sorted(centres)
    single: list[float] = []
    single_valid_only: list[float] = []
    cumulative: list[float] = []
    single_valid = 0
    for index in range(len(frames) - delta):
        start = frames[index]
        image = source.read(start)
        if image is None:
            continue
        result = kernel.step(image, centres[start])
        error = float(np.linalg.norm(result.xy - centres[frames[index + 1]]))
        single.append(error)
        single_valid += int(result.valid)
        if result.valid:
            single_valid_only.append(error)

        position = centres[start].copy()
        ok = True
        for step in range(1, delta):
            image = source.read(frames[index + step])
            if image is None:
                ok = False
                break
            result = kernel.step(image, position)
            if not result.valid:
                ok = False
                break
            position = result.xy
        if ok:
            cumulative.append(float(np.linalg.norm(position - centres[frames[index + delta]])))

    return {
        "delta_frames": int(delta),
        "single_step_error_px": percentiles(single, points=(50, 90, 95, 99)),
        "single_step_error_px_valid_only": percentiles(
            single_valid_only, points=(50, 90, 95, 99)
        ),
        "delta_step_error_px": percentiles(cumulative, points=(50, 90, 95, 99)),
        "single_step_valid_rate": (single_valid / len(single)) if single else float("nan"),
        "single_step_samples": len(single),
        "delta_step_samples": len(cumulative),
        "note": (
            "delta_step_error_px 是从**完美锚点**重推 Δ 步的误差——"
            "臂 3 的理论天花板；实际运行还要叠加锚点（检测器）自身的定位误差。"
            "⚠️ 三个数各有**不同的样本集合**，不可直接相减："
            "single_step_error_px 含 invalid 样本（invalid 时快核位置不动，"
            "此时误差其实是目标真实位移而非内核误差）⇒ 想看纯内核能力请用 "
            "single_step_error_px_valid_only；delta_step_error_px 只含**走通 Δ 步**的样本。"
        ),
    }


def _trackability_precondition(centres: dict[int, np.ndarray], delta: int, roi_half: float) -> dict:
    """**可跟性前提判据**：快核只有 32×32 视野，半窗 ``roi_half`` 像素。

    若目标在 Δ 帧内的位移超过 ``2 × roi_half``，快核**在原理上**就追不上，
    退回重推演也无从谈起（每一步都会把目标甩出 ROI，重推只会弃权）。

    这条判据用来回答："臂 3 表现不好，是实现不行，还是这条序列根本不在
    文档假设的'转台微动'前提里？"
    """
    frames = sorted(centres)
    per_frame: list[float] = []
    over_delta: list[float] = []
    for index in range(len(frames) - 1):
        step = float(np.linalg.norm(centres[frames[index + 1]] - centres[frames[index]]))
        per_frame.append(step)
        if index + delta < len(frames):
            over_delta.append(
                float(np.linalg.norm(centres[frames[index + delta]] - centres[frames[index]]))
            )
    stats = percentiles(per_frame, points=(50, 90, 95, 99))
    delta_stats = percentiles(over_delta, points=(50, 90, 95, 99))
    return {
        "roi_half_width_px": float(roi_half),
        "per_frame_displacement_px": stats,
        "over_delta_displacement_px": delta_stats,
        "frames_per_frame_over_roi": sum(1 for v in per_frame if v > roi_half) / len(per_frame)
        if per_frame
        else 0.0,
        "delta_exceeds_roi_span": bool(delta_stats["p50"] > 2 * roi_half),
        "note": (
            "单帧位移超过 ROI 半窗即追不上；Δ 帧位移超过 ROI 全跨即"
            "'退回重推演'在原理上失效（每步都会脱靶 → 弃权）。"
        ),
    }


def _arm_diagnostics(
    outcome, centres: dict[int, np.ndarray], tolerance: float = 5.0
) -> dict:
    """臂级诊断：输出覆盖率、首次失锁帧、纯前向的漂移增长率。"""
    positions = outcome.positions
    ordered = sorted(positions)
    target_frames = sorted(centres)

    # **覆盖率分子必须与分母同属一个帧集合**：
    # 分母是"目标存在（有 GT）的帧"，分子就必须是"**这些帧上**有输出的帧"。
    # 早期版本用"全部有输出的帧"当分子（含无 GT 帧），导致覆盖率被高估
    # 2.7~3.4 个百分点，且**各臂高估量不同** ⇒ 跨臂比较失真。
    frames_with_output_on_target = sum(1 for frame_index in target_frames if frame_index in positions)

    first_missing = None
    for frame_index in target_frames:
        if frame_index not in positions:
            first_missing = frame_index
            break

    errors: list[float] = []
    for frame_index in ordered:
        centre = centres.get(frame_index)
        if centre is None:
            continue
        errors.append(float(np.linalg.norm(positions[frame_index] - centre)))

    # 漂移增长率：前半段与后半段的误差中位数之差
    half = len(errors) // 2
    early = float(np.median(errors[:half])) if half else float("nan")
    late = float(np.median(errors[half:])) if half else float("nan")

    # **存活窗口命中率**：只在自己还有输出的帧上算。
    # 臂 2「纯前向外推」一旦失锁就再也拿不回目标（不允许回头纠偏），
    # 整段序列的命中率会被"大面积无输出"压成接近 0，看不出它在**还活着**
    # 的时候到底跟得准不准——这两个数必须一起报。
    hits = sum(1 for value in errors if value <= tolerance)
    alive_low, alive_high = wilson_interval(hits, len(errors))

    # **这条臂真正用掉的锚点的定位误差**。
    # 与 `_anchor_ceiling` 的区别：天花板用的是"最高分检测"，而臂 3 实际是
    # 按"离历史位置最近"挑锚点（见 tracker._associate）——两者选的框可能不同。
    # 这里直接量**实际消费掉的**那个框在自己的源帧上的定位误差，
    # 才是"锚点误差叠加到这条臂上"的真值。
    anchor_errors: list[float] = []
    for record in outcome.records:
        source = int(record.get("det_source", -1))
        if source < 0 or source not in centres:
            continue
        det_x = float(record.get("det_x", float("nan")))
        det_y = float(record.get("det_y", float("nan")))
        if math.isnan(det_x) or math.isnan(det_y):
            continue
        anchor_errors.append(
            float(np.linalg.norm(np.asarray([det_x, det_y]) - centres[source]))
        )

    return {
        "frames_with_output": len(ordered),
        "frames_target_present": len(target_frames),
        "frames_with_output_on_target_frames": frames_with_output_on_target,
        "coverage": (frames_with_output_on_target / len(target_frames)) if target_frames else 0.0,
        "coverage_note": (
            "分子 = **有 GT 的帧上**有输出的帧数；分母 = 有 GT 的帧数。"
            "两者同属一个帧集合，可跨臂直接比。"
        ),
        "first_frame_without_output": first_missing,
        "error_p50_early_half": early,
        "error_p50_late_half": late,
        "drift_growth_px": (late - early) if not math.isnan(early) and not math.isnan(late) else float("nan"),
        "hit_rate_on_output_frames": (hits / len(errors)) if errors else 0.0,
        "hit_rate_on_output_frames_wilson95": [alive_low, alive_high],
        "actual_anchor_error_px": percentiles(anchor_errors, points=(50, 95, 99)),
        "actual_anchor_samples": len(anchor_errors),
        "tolerance_px": float(tolerance),
    }


def _replay_cost(outcome) -> dict:
    """臂 3 的退回重推演开销：多少帧触发了重推、单次重推多少毫秒。"""
    applied = []
    costs = []
    steps = []
    reasons: dict[str, int] = {}
    for record in outcome.records:
        if "replay_applied" not in record:
            continue
        if bool(record.get("replay_applied")):
            applied.append(record["frame_index"])
            costs.append(float(record.get("replay_cost_ms", float("nan"))))
            steps.append(int(record.get("replay_steps", 0)))
        reason = str(record.get("replay_reason", ""))
        if reason:
            reasons[reason.split(":")[0]] = reasons.get(reason.split(":")[0], 0) + 1
    return {
        "replay_frames": len(applied),
        "replay_cost_ms": percentiles(costs, points=(50, 95, 99)) if costs else {},
        "replay_steps": percentiles(steps, points=(50, 95, 99)) if steps else {},
        "replay_reason_counts": reasons,
    }


def _step_latency(outcome) -> dict:
    """端到端伺服延时：``tracker.step()`` 的逐帧墙钟（含臂 3 的重推演开销）。"""
    values = [
        float(record["step_ms"])
        for record in outcome.records
        if "step_ms" in record and not math.isnan(float(record["step_ms"]))
    ]
    stats = percentiles(values, points=(50, 95, 99))
    stats["within_1ms_p99"] = bool(stats.get("p99", float("inf")) <= 1.0)
    stats["within_1ms_p50"] = bool(stats.get("p50", float("inf")) <= 1.0)
    return stats


def _hit_flags(outcome, centres: dict[int, np.ndarray], tolerance: float) -> dict[int, bool]:
    """逐帧命中标记；没有输出的帧算未命中（与 HitRate@T 的口径一致）。"""
    flags: dict[int, bool] = {}
    for frame_index in sorted(centres):
        position = outcome.positions.get(frame_index)
        if position is None:
            flags[frame_index] = False
            continue
        flags[frame_index] = bool(
            np.linalg.norm(position - centres[frame_index]) <= tolerance
        )
    return flags


def _paired_comparison(
    name_a: str,
    outcome_a,
    name_b: str,
    outcome_b,
    centres: dict[int, np.ndarray],
    tolerance: float,
) -> dict:
    """两条臂在同一批 GT 帧上的配对命中比较（McNemar 精确检验）。

    分母 = 全部目标存在的帧（与验收表命中率列同口径），某臂没输出的帧记为
    未命中。这样净差多少个百分点与逐帧方向上是否真的不同用的是同一个分母，
    不会像两套分母并列那样被人误读。
    """
    flags_a = _hit_flags(outcome_a, centres, tolerance)
    flags_b = _hit_flags(outcome_b, centres, tolerance)
    frames = sorted(centres)
    b = sum(1 for f in frames if flags_a[f] and not flags_b[f])
    c = sum(1 for f in frames if flags_b[f] and not flags_a[f])
    hits_a = sum(1 for f in frames if flags_a[f])
    hits_b = sum(1 for f in frames if flags_b[f])
    total = len(frames)
    result = mcnemar_exact(b, c)
    result["arm_a"] = name_a
    result["arm_b"] = name_b
    result["n_frames"] = total
    result["hits_a"] = hits_a
    result["hits_b"] = hits_b
    result["rate_a"] = (hits_a / total) if total else 0.0
    result["rate_b"] = (hits_b / total) if total else 0.0
    result["rate_diff_pp"] = (((hits_a - hits_b) / total) * 100.0) if total else 0.0
    result["net_hits_diff"] = hits_a - hits_b
    result["denominator"] = "全部目标存在（有 GT）的帧；某臂无输出记为未命中"

    # ---- 第二套分母：两臂**都有输出**的帧 ----
    # 臂 1「同锚点密度」变体每 2 帧才输出一次，上面的全序列口径会把它
    # "没输出的那一半帧"全记成未命中 ⇒ 命中率被**覆盖率**而非**精度**压低。
    # 在共同输出帧上比，才是"同一信息量下的精度"对照（对应表里的存活窗口列）。
    common = [
        f
        for f in frames
        if outcome_a.positions.get(f) is not None and outcome_b.positions.get(f) is not None
    ]
    cb = sum(1 for f in common if flags_a[f] and not flags_b[f])
    cc = sum(1 for f in common if flags_b[f] and not flags_a[f])
    chits_a = sum(1 for f in common if flags_a[f])
    chits_b = sum(1 for f in common if flags_b[f])
    common_result = mcnemar_exact(cb, cc)
    common_result["arm_a"] = name_a
    common_result["arm_b"] = name_b
    common_result["n_frames"] = len(common)
    common_result["hits_a"] = chits_a
    common_result["hits_b"] = chits_b
    common_result["rate_a"] = (chits_a / len(common)) if common else 0.0
    common_result["rate_b"] = (chits_b / len(common)) if common else 0.0
    common_result["rate_diff_pp"] = (
        ((chits_a - chits_b) / len(common)) * 100.0 if common else 0.0
    )
    common_result["net_hits_diff"] = chits_a - chits_b
    common_result["denominator"] = "两臂都有输出且目标存在的帧（存活窗口口径）"
    result["on_common_output_frames"] = common_result
    return result


# ==============================================================================
# 主流程
# ==============================================================================


def _arm_specs(det_hz: float, latency: int) -> list[ArmSpec]:
    """三个对照臂 + 两个消融臂（定位收益来自哪个开关）。"""
    return [
        # 臂 1：纯检测基准（每帧都跑检测，输出的是 Δ 帧前的旧框）
        ArmSpec(
            name="arm1_detector_only",
            kind="detector_only",
            det_hz=25.0,
            latency_frames=latency,
        ),
        # 公平性对照臂：臂 1 的**同锚点密度**变体（det_hz 与臂 2/3 相同）。
        # 臂 1 用 det_hz=25（每帧一个锚点），信息量是臂 2/3 的 2 倍；
        # 这条臂把"锚点密度"这个混淆变量隔离出来，让"退回重推演有没有用"
        # 在同信息量下重新对照。它不是用户拍板的三条臂之一。
        arm1_detector_only_sparse(det_hz=det_hz, latency_frames=latency),
        # 臂 2：纯前向外推（只对比、不修正、永不回溯）
        ArmSpec(
            name="arm2_forward_only",
            det_hz=det_hz,
            latency_frames=latency,
            fusion_mode="none",
            detection_reinit=False,
        ),
        # 臂 3：退回重推演（精度理论上限）
        ArmSpec(
            name="arm3_rollback_replay",
            det_hz=det_hz,
            latency_frames=latency,
            fusion_mode="replay",
        ),
        # 消融：臂 2 但**允许**检测重新引导（否则臂 2 一旦失锁就整段报废，
        # 数字无法与其它臂同口径比较）
        ArmSpec(
            name="ablation_forward_only_reinit",
            det_hz=det_hz,
            latency_frames=latency,
            fusion_mode="none",
            detection_reinit=True,
        ),
        # 消融：臂 3 + 弃权时硬拉回兜底（量化"退回重推演"剩余的空间在哪）
        ArmSpec(
            name="ablation_replay_fallback",
            det_hz=det_hz,
            latency_frames=latency,
            fusion_mode="replay",
            overrides=(("replay_fallback", "hard"),),
        ),
        # 消融：文档原版的残差平滑融合（不回溯）
        ArmSpec(
            name="ablation_fused_alpha",
            det_hz=det_hz,
            latency_frames=latency,
            fusion_mode="alpha",
        ),
        # 消融：残差硬拉回（无平滑）
        ArmSpec(
            name="ablation_fused_hard",
            det_hz=det_hz,
            latency_frames=latency,
            fusion_mode="alpha",
            overrides=(("smooth_when_small", False),),
        ),
    ]


def run_three_arm(
    sequence: str,
    cache_root: str = DEFAULT_CACHE_ROOT,
    frames_root: str = DEFAULT_FRAMES_ROOT,
    labels_root: str = DEFAULT_LABELS_ROOT,
    det_hz: float = 12.5,
    latency: int = 8,
    camera_hz: float = 25.0,
    total_frames: int = 0,
    score_threshold: float = 0.22,
    dedup_radius: float = 768.0,
    tolerance_px: float = 5.0,
    lock_radius: float = 16.0,
    out_dir: Path | None = None,
    base_config: TrackerConfig | None = None,
    velocity_source: str = "none",
) -> dict:
    """跑三臂 + 消融臂，出验收表。"""
    from manu.evaluation.eval_multiscale_cache_th import load_ground_truth

    base = base_config or TrackerConfig()
    base.latency_frames = latency
    base.camera_hz = camera_hz
    base.velocity_source = velocity_source

    stream = SlowStream(
        cache_root,
        sequence,
        dedup_radius=dedup_radius,
        score_threshold=score_threshold,
    )
    frames_total = int(total_frames or stream.frames)
    source = JpegFrameSource(Path(frames_root) / sequence)
    specs = _arm_specs(det_hz, latency)

    started = time.perf_counter()
    outcomes = run_lockstep(
        source,
        stream,
        specs,
        base_config=base,
        total_frames=frames_total,
        camera_hz=camera_hz,
    )
    elapsed = time.perf_counter() - started

    ground_truth = load_ground_truth(Path(labels_root), sequence, stream.width, stream.height)
    if frames_total < stream.frames:
        # 只跑了一部分帧时，GT 必须**同步裁剪**——否则命中率的分母是整段序列、
        # 分子只有跑过的那几帧，统计口径直接失真。
        ground_truth = {
            frame: boxes for frame, boxes in ground_truth.items() if frame < frames_total
        }
    centres = _gt_centres(ground_truth)

    arms: dict[str, dict] = {}
    for spec in specs:
        outcome = outcomes[spec.name]
        metrics = servo_metrics(
            outcome.records,
            ground_truth,
            tolerance=tolerance_px,
            lock_radius=lock_radius,
        ).to_dict()
        arms[spec.name] = {
            "spec": {
                "kind": spec.kind,
                "det_hz": spec.det_hz,
                "latency_frames": spec.latency_frames,
                "fusion_mode": spec.fusion_mode,
                "detection_reinit": spec.detection_reinit,
            },
            "servo_metrics": metrics,
            "diagnostics": _arm_diagnostics(outcome, centres, tolerance=tolerance_px),
            "time_to_servo": _step_latency(outcome),
            "replay": _replay_cost(outcome),
        }

    payload = {
        "sequence": sequence,
        "detector": "RC2 / trial_0080 / epoch3 (ms_cache)",
        "frames": frames_total,
        "camera_hz": camera_hz,
        "det_hz": det_hz,
        "latency_frames": latency,
        "latency_ms": latency * (1000.0 / camera_hz),
        "tolerance_px": tolerance_px,
        "lock_radius_px": lock_radius,
        "elapsed_seconds": elapsed,
        "anchor_ceiling": _anchor_ceiling(stream, centres, latency),
        "kernel_ceiling": _kernel_ceiling(source, centres, latency),
        "trackability": _trackability_precondition(centres, latency, base.win_size / 2.0),
        "arms": arms,
    }

    # ---- 配对检验（McNemar 精确检验）：同一批 GT 帧上的两两对比 ----
    # 规格验收表里"命中率互不可区分"这句话必须由**配对检验**支撑，
    # 而不是"两条独立样本的 Wilson CI 是否重叠"（那是用错了检验）。
    paired_pairs = (
        ("arm1_detector_only", "arm3_rollback_replay"),
        ("arm1_detector_only_sparse", "arm3_rollback_replay"),
        ("arm1_detector_only", "ablation_fused_alpha"),
        ("arm1_detector_only_sparse", "ablation_fused_alpha"),
        ("arm3_rollback_replay", "ablation_fused_alpha"),
    )
    paired_tests = []
    for name_a, name_b in paired_pairs:
        if name_a in outcomes and name_b in outcomes:
            paired_tests.append(
                _paired_comparison(
                    name_a,
                    outcomes[name_a],
                    name_b,
                    outcomes[name_b],
                    centres,
                    tolerance_px,
                )
            )
    payload["paired_tests"] = paired_tests
    payload["paired_tests_note"] = (
        "同一批 GT 帧上的配对命中比较（McNemar 精确二项检验，双侧）。"
        "b = A 命中 B 未命中，c = B 命中 A 未命中。p<0.05 表示两臂逐帧结果"
        "存在统计显著差异；p>=0.05 只能说'本样本量下分辨不出'，"
        "**不等于**证明了两臂等价。Wilson CI 的独立性假设在连续帧上不成立，"
        "因此 CI 重叠只能当保守旁证。"
    )
    payload["acceptance_table"] = build_acceptance_table(payload)

    if out_dir is not None:
        out_dir = Path(out_dir) / sequence
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "three_arm_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_evidence(out_dir, outcomes, ground_truth, lock_radius)
        (out_dir / "three_arm_table.md").write_text(
            render_markdown_table(payload), encoding="utf-8"
        )
        payload["artifacts"] = str(out_dir)
    return payload


def _write_evidence(out_dir: Path, outcomes, ground_truth, lock_radius: float) -> None:
    """逐帧证据 TSV（每条结论都能落到具体帧）。"""
    from .evaluate import write_evidence

    for name, outcome in outcomes.items():
        write_evidence(
            out_dir / f"evidence_{name}.tsv",
            outcome.records,
            ground_truth,
            lock_radius=lock_radius,
        )


# ==============================================================================
# 验收表
# ==============================================================================

_ARM_LABELS = {
    "arm1_detector_only": "臂 1 纯检测基准",
    "arm1_detector_only_sparse": "（对照）臂 1 同锚点密度（det_hz=12.5）",
    "arm2_forward_only": "臂 2 检跟纯前向外推",
    "arm3_rollback_replay": "臂 3 异步退回重推演",
    "ablation_forward_only_reinit": "（消融）纯前向 + 允许重引导",
    "ablation_replay_fallback": "（消融）臂 3 + 弃权硬拉回兜底",
    "ablation_fused_alpha": "（消融）残差平滑融合",
    "ablation_fused_hard": "（消融）残差硬拉回",
}


def build_acceptance_table(payload: dict) -> list[dict]:
    """按用户给的表头产出对照表。

    表头：评估臂 | 命中率 (Hit Rate) | CLE p50 | 动态抖动 (Max Jitter) | 端到端延时
    """
    rows: list[dict] = []
    for name, label in _ARM_LABELS.items():
        arm = payload["arms"].get(name)
        if arm is None:
            continue
        servo = arm["servo_metrics"]
        cle = servo["cle"]
        hit = servo["hit_rate"]
        jitter = servo["jitter"]
        latency = arm["time_to_servo"]
        rows.append(
            {
                "arm": label,
                "arm_key": name,
                "hit_rate": hit["rate"],
                "hit_rate_wilson95": hit["wilson95"],
                "inbox_hit_rate": servo["inbox_hit_rate"]["rate"],
                "cle_p50": cle["p50"],
                "cle_p95": cle["p95"],
                "cle_p99": cle["p99"],
                "cle_max": cle["max"],
                "max_jitter_acceleration": jitter["acceleration"]["max"],
                "max_jitter_acceleration_steady": jitter["acceleration_steady"]["max"],
                "median_jitter_acceleration_steady": jitter["acceleration_steady"]["p50"],
                "jitter_acceleration_steady_p95": jitter["acceleration_steady"]["p95"],
                "jitter_acceleration_p95": jitter["acceleration"]["p95"],
                "step_jump_at_correction_p95": jitter["step_jump"]["at_correction_any"]["p95"],
                "step_jump_at_correction_max": jitter["step_jump"]["at_correction_any"]["max"],
                "step_jump_at_correction_n": jitter["step_jump"]["at_correction_any"]["n"],
                "step_jump_over2px_at_correction": jitter["step_jump"]["over_threshold_counts"][
                    "correction_any"
                ],
                "step_jump_criterion_2px_pass_p95": jitter["step_jump"]["criterion_2px_pass_p95"],
                "step_jump_smooth_p95": jitter["step_jump"]["at_correction_smooth"]["p95"],
                "step_jump_smooth_max": jitter["step_jump"]["at_correction_smooth"]["max"],
                "step_jump_smooth_n": jitter["step_jump"]["at_correction_smooth"]["n"],
                "step_jump_smooth_over2px": jitter["step_jump"]["over_threshold_counts"]["smooth"],
                "lost_frame_rate": servo["fragmentation"]["lost_frame_rate"],
                "lost_frames": servo["fragmentation"]["lost_frames"],
                "end_to_end_latency_p50_ms": latency["p50"],
                "end_to_end_latency_p99_ms": latency["p99"],
                "hit_rate_on_output_frames": arm["diagnostics"]["hit_rate_on_output_frames"],
                "hit_rate_on_output_frames_wilson95": arm["diagnostics"][
                    "hit_rate_on_output_frames_wilson95"
                ],
                "coverage": arm["diagnostics"]["coverage"],
                "frames_with_output": arm["diagnostics"]["frames_with_output"],
                "frames_target_present": arm["diagnostics"]["frames_target_present"],
                "fragmentations": servo["fragmentation"]["fragmentations"],
                "lock_streak_max": servo["fragmentation"]["lock_streak_max"],
                "lost_streak_max": servo["fragmentation"]["lost_streak_max"],
            }
        )
    return rows


def _fmt_px(value: float, count: int) -> str:
    """像素值渲染；样本数为 0 时显示 "—"（而不是 nan）。"""
    if not count or value is None or math.isnan(value):
        return "—"
    return f"{value:.2f} px"


def render_markdown_table(payload: dict) -> str:
    """把验收表渲染成 markdown（直接贴进汇报材料）。"""
    ceiling = payload["anchor_ceiling"]
    lines: list[str] = []
    lines.append(f"# 三臂对照验收表 — {payload['sequence']}")
    lines.append("")
    lines.append(
        f"检测器：{payload['detector']}；相机 {payload['camera_hz']} Hz；"
        f"慢流 {payload['det_hz']} Hz；滞后 Δ={payload['latency_frames']} 帧"
        f"（{payload['latency_ms']:.0f} ms）；总帧数 {payload['frames']}"
    )
    lines.append("")
    lines.append("## 对照矩阵")
    lines.append("")
    lines.append(
        "| 评估臂 | 命中率 HitRate@5px | Wilson 95% CI | 存活窗口命中率 | "
        "CLE p50 | CLE p95 | 抖动 p95(steady) | StepJump@smooth p95 | "
        "端到端延时 p50/p99 | 输出覆盖率 | 脱靶率 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["acceptance_table"]:
        low, high = row["hit_rate_wilson95"]
        lines.append(
            f"| {row['arm']} | {row['hit_rate'] * 100:.2f}% | "
            f"[{low * 100:.2f}%, {high * 100:.2f}%] | "
            f"{row['hit_rate_on_output_frames'] * 100:.2f}% | "
            f"{row['cle_p50']:.2f} px | {row['cle_p95']:.2f} px | "
            f"{row['jitter_acceleration_steady_p95']:.2f} px | "
            f"{_fmt_px(row['step_jump_smooth_p95'], row['step_jump_smooth_n'])} | "
            f"{row['end_to_end_latency_p50_ms']:.3f} / {row['end_to_end_latency_p99_ms']:.3f} ms | "
            f"{row['coverage'] * 100:.1f}% | "
            f"{row['lost_frame_rate'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append(
        "> **分母脚注（各列分母不同，不可混用）**："
        "「命中率」分母 = 全部**目标存在**的帧（无输出的帧计入分母且不算命中）；"
        "「存活窗口命中率」分母 = **有输出 ∧ 目标存在**的帧（只看这条臂还活着时跟得准不准）；"
        "「输出覆盖率」分母 = 目标存在的帧，分子 = **这些帧上**有输出的帧；"
        "「脱靶率」分母 = 目标存在的帧，分子 = 未锁定（无输出 或 离 GT > 16 px）的帧；"
        "「StepJump@校正帧」只统计**检测异步到达并施加了校正**的那些帧。"
    )
    lines.append("")
    lines.append(
        "> **抖动口径**：「抖动 p95(steady)」= 二阶差分 ‖Δv‖ 的 p95，"
        "且**排除**状态机切换帧与故意重定位帧（硬拉回 / 退回重推演）——"
        "这几个全局 max 值会被状态机切换吃掉，不适合跨臂比较；"
        "「StepJump」= 一阶差分 ‖p_t − p_{t−1}‖ 在校正帧上的 p95（规格判据，阈值 2 px）。"
    )
    lines.append("")
    lines.append("## 含全局最大值的抖动诊断（不建议跨臂比较）")
    lines.append("")
    lines.append(
        "| 评估臂 | Max Jitter(Δv) 全局 max | steady 桶 max | 校正帧 StepJump max | "
        "校正帧 > 2 px 帧数 | 校正帧数 | smooth 桶 StepJump max | smooth 桶 > 2 px |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["acceptance_table"]:
        lines.append(
            f"| {row['arm']} | {row['max_jitter_acceleration']:.2f} px | "
            f"{row['max_jitter_acceleration_steady']:.2f} px | "
            f"{_fmt_px(row['step_jump_at_correction_max'], row['step_jump_at_correction_n'])} | "
            f"{row['step_jump_over2px_at_correction']} | "
            f"{row['step_jump_at_correction_n']} | "
            f"{_fmt_px(row['step_jump_smooth_max'], row['step_jump_smooth_n'])} | "
            f"{row['step_jump_smooth_over2px']} |"
        )
    lines.append("")
    paired = payload.get("paired_tests") or []
    if paired:
        lines.append("## 配对检验（同一批 GT 帧，McNemar 精确二项检验）")
        lines.append("")
        lines.append(
            "| 对比 | 分母 | A 命中 | B 命中 | 净差 (pp) | b (A独赢) | c (B独赢) | z | p (双侧) | 显著? |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for test in paired:
            lines.append(
                f"| {test['arm_a']} vs {test['arm_b']} | 全 GT 帧 | "
                f"{test['hits_a']} | {test['hits_b']} | "
                f"{test['rate_diff_pp']:+.3f} | {test['b']} | {test['c']} | "
                f"{test['z']:.2f} | {test['p_value']:.3g} | "
                f"{'是' if test['significant_5pct'] else '否'} |"
            )
            common = test.get("on_common_output_frames") or {}
            if common:
                lines.append(
                    f"| {test['arm_a']} vs {test['arm_b']} | 共同输出帧 | "
                    f"{common['hits_a']} | {common['hits_b']} | "
                    f"{common['rate_diff_pp']:+.3f} | {common['b']} | {common['c']} | "
                    f"{common['z']:.2f} | {common['p_value']:.3g} | "
                    f"{'是' if common['significant_5pct'] else '否'} |"
                )
        lines.append("")
        lines.append(f"> {payload.get('paired_tests_note', '')}")
        lines.append("")
    lines.append("## 臂 3 的精度天花板（锚点误差上限）")
    lines.append("")
    anchor = ceiling["anchor_error_px"]
    disp = ceiling["gt_displacement_over_delta_px"]
    lines.append(
        f"- 检测框在**自己那一帧**的定位误差：p50={anchor['p50']:.2f} / "
        f"p95={anchor['p95']:.2f} / p99={anchor['p99']:.2f} px"
    )
    lines.append(
        f"- 目标在 Δ={ceiling['delta_frames']} 帧内的真实位移："
        f"p50={disp['p50']:.2f} / p95={disp['p95']:.2f} / p99={disp['p99']:.2f} px"
    )
    lines.append(f"- {ceiling['note']}")
    lines.append("")

    kernel = payload["kernel_ceiling"]
    single = kernel["single_step_error_px"]
    cumulative = kernel["delta_step_error_px"]
    lines.append("## 臂 3 的理论天花板（快核能力，与本 harness 实现无关）")
    lines.append("")
    lines.append(
        f"- 快核**单步**局部误差（从真值锚点走一步）：p50={single['p50']:.2f} / "
        f"p90={single['p90']:.2f} / p95={single['p95']:.2f} / p99={single['p99']:.2f} px"
    )
    lines.append(
        f"- 从**完美锚点**连续重推 Δ={kernel['delta_frames']} 步的累积误差："
        f"p50={cumulative['p50']:.2f} / p90={cumulative['p90']:.2f} / "
        f"p95={cumulative['p95']:.2f} / p99={cumulative['p99']:.2f} px"
    )
    lines.append(
        f"- 单步有效率 {kernel['single_step_valid_rate']:.3f}；"
        f"能走完 Δ 步的样本 {kernel['delta_step_samples']} 帧"
    )
    lines.append(f"- {kernel['note']}")
    lines.append("")

    track = payload["trackability"]
    per_frame = track["per_frame_displacement_px"]
    over_delta = track["over_delta_displacement_px"]
    lines.append("## 可跟性前提判据（快核视野 = 32×32）")
    lines.append("")
    lines.append(
        f"- 目标**单帧**位移：p50={per_frame['p50']:.2f} / p90={per_frame['p90']:.2f} / "
        f"p95={per_frame['p95']:.2f} / p99={per_frame['p99']:.2f} px"
    )
    lines.append(
        f"- 目标在 Δ={payload['latency_frames']} 帧内的位移：p50={over_delta['p50']:.2f} / "
        f"p90={over_delta['p90']:.2f} / p95={over_delta['p95']:.2f} / p99={over_delta['p99']:.2f} px"
    )
    lines.append(
        f"- 单帧位移超过 ROI 半窗（{track['roi_half_width_px']:.0f} px）的帧占比："
        f"{track['frames_per_frame_over_roi'] * 100:.2f}%"
    )
    lines.append(
        f"- Δ 帧位移超过 ROI 全跨（{2 * track['roi_half_width_px']:.0f} px）："
        f"{'是 —— 退回重推演在原理上失效' if track['delta_exceeds_roi_span'] else '否'}"
    )
    lines.append(f"- {track['note']}")
    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# CLI
# ==============================================================================


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="三臂对照验收表")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--frames-root", default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--labels-root", default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--det-hz", type=float, default=12.5)
    parser.add_argument("--latency", type=int, default=8)
    parser.add_argument("--camera-hz", type=float, default=25.0)
    parser.add_argument("--total-frames", type=int, default=0)
    parser.add_argument("--tolerance-px", type=float, default=5.0)
    parser.add_argument("--lock-radius-px", type=float, default=16.0)
    parser.add_argument(
        "--velocity-source",
        default="none",
        choices=("history", "detection", "none"),
        help=(
            "硬拉回投影的速度源（默认 none=不投影，即文档 §三 伪代码语义；"
            "history=快跟踪器历史速度 / detection=检测流速度，"
            "两者真实序列实测均为负收益，见 README「速度源对照」段）"
        ),
    )
    parser.add_argument("--out-dir", default="runs/tracking_async")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    payload = run_three_arm(
        sequence=args.sequence,
        cache_root=args.cache_root,
        frames_root=args.frames_root,
        labels_root=args.labels_root,
        det_hz=args.det_hz,
        latency=args.latency,
        camera_hz=args.camera_hz,
        total_frames=args.total_frames,
        tolerance_px=args.tolerance_px,
        lock_radius=args.lock_radius_px,
        out_dir=None if args.no_write else Path(args.out_dir),
        velocity_source=args.velocity_source,
    )

    print(render_markdown_table(payload))
    print(
        f"[env] python={platform.python_version()} numpy={np.__version__} "
        f"git={_git_commit()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
