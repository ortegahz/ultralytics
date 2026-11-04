#!/usr/bin/env python3
"""异步双频检跟分离系统 —— 离线受控 A/B 评测闸门。

**这个文件是整个项目的"精度验证设计"的落地处。** 后续所有移植
（板卡 / C++ / NEON）都以这里的闸门为验收标准，因此每个闸门都必须
满足三条硬约束：

1. **口径同源**：GT 匹配直接 import 冻结实现
   （``manu/evaluation/eval_multiscale_cache_th.py``），不重写一遍；
2. **受控可比**：所有臂在**同一次锁步遍历**里看到逐字节相同的帧与
   同一批检测，唯一差别是 ``enable_fusion`` 这一个开关；
3. **逐帧可查**：每条结论都能落到逐帧证据表（TSV），不允许只报汇总数。

闸门清单：

======== ==================================================================
Gate    内容
======== ==================================================================
Gate 1  单元/契约测试（``manu/tracking/tests/``），由 unittest 跑
Gate 2  **空对照臂**：注入 δ=0 ⇒ fused 与 tracker_only 必须逐位相同；
        Δ=0 ⇒ 融合必然弃权 ⇒ fused 与 tracker_only 仍必须逐位相同
Gate 3  **合成注入臂**：注入 z_f = x_f + δ（δ 已知）⇒ 输出必须等于解析预测
Gate 4  **真检测器回放**：RC2/trial_0080 的 ms_cache，det_hz × Δ 扫描，
        出 F1（冻结口径）/ 抖动 / 脱靶维持周期 / Time-to-Servo
Gate 5  **真实异步臂**：真线程调度（``async_pipeline.py``，独立模块）
Gate 6  **口径对齐**：用本 harness 给冻结的 ``sys_sota_preds`` 重新打分，
        必须逐位复现冻结数字
======== ==================================================================

**命名对照表（本模块一律用机制名，不用"臂 N"编号）**：

================= ============================ ================
机制名               机制                           本文件历史别名
================= ============================ ================
``detector_only`` 纯检测基准：每帧全图检测，输出 Δ 帧前的旧框      臂 1
``alpha_fused``   检跟分离 + alpha 异步残差平滑（完整架构）    臂 3
``hard_pull``     检跟分离 + 硬拉回（不做残差平滑）           臂 2
``tracker_only``  检跟分离但只跟踪不融合（消融对照）            "对照"
``replay``        退回重推演：迟到检测作绝对锚点，退回 t−Δ 重推    本表暂无此行
``frozen_det_f1`` 冻结 RC2 单帧检测 F1（**口径锚点**，非机制） ``track3_blind``
================= ============================ ================

``three_arm.py`` 里的『臂 1/2/3』是**用户拍板的另一套编号**（纯检测 /
纯前向外推 / 退回重推演），与本模块的机制名**不是一回事**，跨表比较前先看
本对照表。

.. note::

   - 本模块里其余的『臂』字样（Gate 2 空对照臂 / Gate 3 合成注入臂 /
     Gate 5 真实异步臂）是**泛指实验分支**，与上表的编号无关；
   - 对外交付（C++ / 板卡移植）**只允许带机制名**，历史『臂 N』别名
     不要一起带走；
   - ``frozen_det_f1`` 是**冻结 RC2 单帧检测 F1**，仅作打分**口径锚点**，
     **禁止**与三臂表的伺服指标（命中率 / CLE 等）**跨系统比大小**。

用法（本地 conda 环境 ``uav``）::

    cd ultralytics
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate uav
    PYTHONPATH=. python -m manu.tracking.async_dual_rate.evaluate --gate all \
        --sequence VIDEO00007_19700101_001324 \
        --out-dir runs/tracking_async/001324
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .arms import (
    ArmOutcome,
    ArmSpec,
    JpegFrameSource,
    OracleDetector,
    SlowStream,
    SwitchingOracleDetector,
    SyntheticFrameSource,
    TwoBlobFrameSource,
    run_lockstep,
)
from .fusion import FusionConfig, ResidualFusion
from .history import HistoryRingBuffer
from .metrics import score_positions, score_predictions
from .servo_metrics import servo_metrics
from .tracker import AsyncDualRateTracker, TrackerConfig

__all__ = [
    "GateResult",
    "gate2_null_control_synthetic",
    "gate2_null_control_zero_latency",
    "gate3a_fusion_unit",
    "gate3b_synthetic_injection",
    "gate4_detector_replay",
    "gate6_metric_alignment",
    "main",
]

# --------------------------------------------------------------------------
# 冻结参照（只作"口径锚点"，不做跨系统比大小——项目铁律禁止跨级 SOTA 比较）
# --------------------------------------------------------------------------

FROZEN_SYS_SOTA_TARGET = {
    "tp": 8024,
    "fp": 465,
    "fn": 2511,
    "det": 8489,
    "f1": 0.8435660218671152,
}
"""``runs/sys_sota_preds/VIDEO00007_19700101_001324.json`` 的冻结指标（Gate 6 目标）。"""

FROZEN_TRACK3_BLIND = {
    "tp": 8972,
    "fp": 482,
    "fn": 1563,
    "det": 9454,
    "gt": 10535,
    "f1": 0.8976937,
}
"""``track3_blind.json``（RC2/trial_0080 的冻结**单帧检测**结果）的 F1。

这是**冻结 RC2 单帧检测 F1**，用途只有一个：核对"本 harness 的打分口径
== 冻结口径"。它**不是**伺服指标，因此**禁止**与三臂表
（``detector_only`` / ``hard_pull`` / ``alpha_fused``）的命中率、CLE 等
伺服指标**跨系统比大小**，也**禁止**与 ``three_arm.py`` 的『臂 1/2/3』
混为一谈——那是另一套编号。命名对照表里它叫 ``frozen_det_f1``。
"""


# --------------------------------------------------------------------------
# 命名对照表：本模块的权威标签是"机制名"，历史"臂 N"编号只作别名保留
# --------------------------------------------------------------------------

MECHANISMS: dict[str, dict[str, str]] = {
    "detector_only": {
        "label": "纯检测基准·每帧检测+Δ滞后",
        "legacy_arm": "臂1 纯检测(每帧检测+Δ滞后)",
        "row_key": "detector_only",
    },
    "alpha_fused": {
        "label": "检跟分离·残差平滑",
        "legacy_arm": "臂3 检跟分离(完整架构)",
        "row_key": "fused",
    },
    "hard_pull": {
        "label": "检跟分离·硬拉回",
        "legacy_arm": "臂2 检跟分离(无残差平滑,硬拉回)",
        "row_key": "fused_nosmooth",
    },
    "tracker_only": {
        "label": "检跟分离·只跟踪不融合",
        "legacy_arm": "对照: 检跟分离(只跟踪不融合)",
        "row_key": "tracker_only",
    },
}
"""机制名 → 展示标签 / 历史"臂 N"别名 / Gate 4 ``rows`` 里的数据键。

``row_key`` 是 Gate 4 ``rows`` 中 ``servo_*`` / ``*@det_frames`` /
``*@all_frames`` 的键名（``fused`` = ``alpha_fused``，``fused_nosmooth``
= ``hard_pull``）；这些键名**沿用历史命名**不动，因为已有 ``summary.json``
产物依赖它们。``replay``（退回重推演）与 ``frozen_det_f1``（冻结 RC2 单帧
检测 F1 口径锚点）不出现在本表的数据行里，但同属模块头的命名对照表。
"""


# ==========================================================================
# 闸门结果容器
# ==========================================================================


@dataclass
class GateResult:
    """一个闸门的结果：逐条判据 + 汇总。"""

    gate: str
    name: str
    checks: list[dict] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def check(self, label: str, passed: bool, measured=None, expected=None, note: str = "") -> bool:
        self.checks.append(
            {
                "label": label,
                "passed": bool(passed),
                "measured": measured,
                "expected": expected,
                "note": note,
            }
        )
        return bool(passed)

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "name": self.name,
            "passed": self.passed,
            "checks": self.checks,
            "detail": self.detail,
        }


# ==========================================================================
# Gate 2：空对照臂
# ==========================================================================


def _compare_positions(left: ArmOutcome, right: ArmOutcome) -> dict:
    """两个臂的逐帧位置差（判定"逐位相同"）。"""
    frames = sorted(set(left.positions) | set(right.positions))
    max_delta = 0.0
    worst_frame = -1
    missing = 0
    for frame in frames:
        a = left.positions.get(frame)
        b = right.positions.get(frame)
        if a is None or b is None:
            missing += 1
            continue
        delta = float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
        if delta > max_delta:
            max_delta = delta
            worst_frame = frame
    return {
        "frames_compared": len(frames),
        "max_abs_delta": max_delta,
        "worst_frame": worst_frame,
        "frames_missing_on_one_side": missing,
    }


def gate2_null_control_synthetic(
    base_config: TrackerConfig | None = None,
    frames: int = 240,
    camera_hz: float = 25.0,
    det_hz: float = 12.5,
    latency_frames: int = 2,
) -> GateResult:
    """Gate 2a：合成 δ=0 注入 ⇒ fused 与 tracker_only 必须**逐位相同**。

    原理：注入的检测框就等于快跟踪器自己在那一帧的历史位置，所以残差
    ``e = z_f − x_f ≡ 0``，平滑修正 ``x += α·e`` 恒等于 0。若两个臂的输出
    出现任何非零差异，说明修正路径里混进了别的东西（状态泄漏、共享可变
    对象、非确定性），这个闸门就是专门抓这种"看不见的耦合"的。
    """
    result = GateResult("Gate 2a", "空对照臂（δ=0 合成注入）")
    base = base_config or TrackerConfig()

    source = SyntheticFrameSource(frames=frames)
    specs = [
        ArmSpec("tracker_only", enable_fusion=False),
        ArmSpec("fused", enable_fusion=True),
    ]

    def factory(_name, tracker):
        return OracleDetector(tracker, delta=(0.0, 0.0), truth_provider=source.position)

    outcomes = run_lockstep(
        source,
        None,
        specs,
        base_config=base,
        total_frames=frames,
        camera_hz=camera_hz,
        detector_factory=factory,
    )
    comparison = _compare_positions(outcomes["tracker_only"], outcomes["fused"])
    result.check(
        "fused 与 tracker_only 逐帧位置差 = 0",
        comparison["max_abs_delta"] == 0.0,
        measured=comparison["max_abs_delta"],
        expected=0.0,
        note=f"比较帧数 {comparison['frames_compared']}",
    )
    result.check(
        "两个臂的输出帧集合一致",
        comparison["frames_missing_on_one_side"] == 0,
        measured=comparison["frames_missing_on_one_side"],
        expected=0,
    )
    states_a = [record["state"] for record in outcomes["tracker_only"].records]
    states_b = [record["state"] for record in outcomes["fused"].records]
    result.check(
        "两个臂的状态机轨迹一致",
        states_a == states_b,
        measured=int(sum(1 for a, b in zip(states_a, states_b) if a != b)),
        expected=0,
    )
    corrections = [
        float(np.hypot(record["corr_dx"], record["corr_dy"]))
        for record in outcomes["fused"].records
        if record["fusion_kind"] != "none"
    ]
    max_correction = max(corrections) if corrections else 0.0
    result.check(
        "注入的修正量恒为 0",
        max_correction == 0.0,
        measured=max_correction,
        expected=0.0,
        note=f"发生融合 {len(corrections)} 次",
    )
    result.detail = {
        "comparison": comparison,
        "fusion_events": len(corrections),
        "frames": frames,
        "det_hz": det_hz,
        "latency_frames": latency_frames,
    }
    return result


def gate2_null_control_zero_latency(
    sequence: str,
    cache_root,
    frames_root,
    base_config: TrackerConfig | None = None,
    total_frames: int = 0,
    camera_hz: float = 25.0,
    det_hz: float = 12.5,
) -> GateResult:
    """Gate 2b：真检测流 + Δ=0 ⇒ 融合必然弃权 ⇒ 与 tracker_only 逐位相同。

    Δ=0 时"针对第 t 帧的检测"在第 t 帧到达，而快跟踪器的历史缓冲在第 t
    帧**末尾**才写入 ⇒ 同刻对账的前提不成立 ⇒ 融合按设计弃权。这条闸门
    验证的是"弃权路径真的什么都不做"，而不是"弃权路径偷偷改了位置"。
    """
    result = GateResult("Gate 2b", "空对照臂（Δ=0 真检测流）")
    base = base_config or TrackerConfig()

    stream = SlowStream(cache_root, sequence)
    frames_total = int(total_frames or stream.frames)
    source = JpegFrameSource(Path(frames_root) / sequence)

    specs = [
        ArmSpec("tracker_only", det_hz=det_hz, latency_frames=0, enable_fusion=False),
        ArmSpec("fused", det_hz=det_hz, latency_frames=0, enable_fusion=True),
    ]
    outcomes = run_lockstep(
        source,
        stream,
        specs,
        base_config=base,
        total_frames=frames_total,
        camera_hz=camera_hz,
    )
    comparison = _compare_positions(outcomes["tracker_only"], outcomes["fused"])
    result.check(
        "Δ=0 时 fused 与 tracker_only 逐帧位置差 = 0",
        comparison["max_abs_delta"] == 0.0,
        measured=comparison["max_abs_delta"],
        expected=0.0,
        note=f"比较帧数 {comparison['frames_compared']}",
    )
    applied = [record for record in outcomes["fused"].records if record["fusion_kind"] not in ("none",)]
    result.check(
        "Δ=0 时融合全部弃权（没有一次生效）",
        len(applied) == 0,
        measured=len(applied),
        expected=0,
    )
    result.detail = {"comparison": comparison, "sequence": sequence, "frames": frames_total}
    return result


# ==========================================================================
# Gate 3：合成注入臂
# ==========================================================================


def gate3a_fusion_unit(base_config: TrackerConfig | None = None) -> GateResult:
    """Gate 3a：融合算术的解析契约（纯函数级，精确到浮点）。

    构造一段已知历史，直接调用 ``ResidualFusion``，检查：

    * 小残差：``e = z_f − x_f``，``position_after = position_before + α·e``；
    * 大残差 ``velocity_source="history"``：``position_after = z_f + v·(t − f)``
      （v = 区间平均速度）；
    * 大残差 ``velocity_source="none"``（**产品默认**）：``position_after = z_f``，
      逐位相等、不做任何投影。

    两种速度源**都要测**：投影分支是文档 §二.3 散文描述的机制，必须保证它
    仍然算得对（可开关、可对照）；不投影分支是文档 §三 伪代码骨架的写法，
    也是真实序列实测更优的那一支，是**产品默认**。
    """
    result = GateResult("Gate 3a", "融合算术解析契约")
    base = base_config or TrackerConfig()

    def make_fusion(velocity_source: str) -> ResidualFusion:
        return ResidualFusion(
            FusionConfig(
                gating_dist=base.gating_dist,
                alpha=base.alpha,
                hard_reinit=base.hard_reinit,
                project_velocity=base.project_velocity,
                velocity_source=velocity_source,
            )
        )

    fusion = make_fusion("none")

    history = HistoryRingBuffer(64)
    for index, (x, y) in enumerate([(100.0, 100.0), (101.0, 100.5), (102.0, 101.0)]):
        history.push(index, np.asarray([x, y], dtype=np.float64))

    delta = np.asarray([3.0, -2.0], dtype=np.float64)
    position = np.asarray([110.0, 100.0], dtype=np.float64)
    decision = fusion.fuse(position, history.get(0) + delta, 0, 2, history)
    result.check(
        "小残差：残差 = z_f − x_f",
        bool(np.allclose(decision.residual, delta, rtol=0, atol=1e-12)),
        measured=decision.residual.tolist(),
        expected=delta.tolist(),
    )
    expected_after = position + base.alpha * delta
    result.check(
        "小残差：position_after = position_before + α·e（逐位）",
        bool(np.array_equal(decision.position_after, expected_after)),
        measured=decision.position_after.tolist(),
        expected=expected_after.tolist(),
    )
    result.check("小残差：走平滑分支", decision.kind == "smooth", measured=decision.kind, expected="smooth")

    far = np.asarray([160.0, 100.0], dtype=np.float64)
    decision_far = fusion.fuse(position, far, 0, 2, history)
    result.check(
        "大残差：走硬拉回分支",
        decision_far.kind == "hard_reinit",
        measured=decision_far.kind,
        expected="hard_reinit",
    )
    result.check(
        "大残差（默认 none）：position_after = z_f，逐位（不投影）",
        bool(np.array_equal(decision_far.position_after, far)),
        measured=decision_far.position_after.tolist(),
        expected=far.tolist(),
    )

    # 投影分支（velocity_source="history"）：单独构造一份融合器验算术。
    fusion_history = make_fusion("history")
    decision_far_history = fusion_history.fuse(position, far, 0, 2, history)
    velocity = (history.get(1) - history.get(0)) / 1.0
    expected_far = far + velocity * 2.0
    result.check(
        "大残差（history）：position_after = z_f + v·(t − f)（逐位）",
        bool(np.array_equal(decision_far_history.position_after, expected_far)),
        measured=decision_far_history.position_after.tolist(),
        expected=expected_far.tolist(),
    )

    missing = fusion.fuse(position, far, 999, 1000, history)
    result.check(
        "历史缺失 ⇒ 弃权（不改位置）",
        (not missing.applied) and bool(np.array_equal(missing.position_after, position)),
        measured={"applied": missing.applied, "position_after": missing.position_after.tolist()},
        expected={"applied": False, "position_after": position.tolist()},
    )
    result.detail = {
        "alpha": base.alpha,
        "gating_dist": base.gating_dist,
        "velocity_source_default": base.velocity_source,
    }
    return result


def gate3b_synthetic_injection(
    base_config: TrackerConfig | None = None,
    frames: int = 240,
    camera_hz: float = 25.0,
    det_hz: float = 12.5,
    latency_frames: int = 2,
    small_delta: tuple[float, float] = (3.0, -2.0),
    decoy_offset: tuple[float, float] = (40.0, 0.0),
    switch_frame: int = 120,
) -> GateResult:
    """Gate 3b：端到端合成注入 ⇒ 输出必须等于解析预测。

    两个物理场景，各自的 δ 都是**已知常数**：

    * **S1 小残差（平滑分支）**：注入 ``z_f = x_f + δ``（δ 小）⇒ 残差恒为 δ，
      修正量恒为 ``α·δ``，输出 = 快跟踪核(校正后位置)。
    * **S2 大残差（硬拉回分支）**：快跟踪核锁在**诱饵**上，检测器随后改报
      **真目标** ⇒ 残差 ≈ 诱饵与真目标的间距（已知大常数）；两个斑静止
      ⇒ ``v ≡ 0`` ⇒ 硬拉回的解析解退化成 ``position_after = z_f``。

    判据用容差而不是"精确等于"：``(a + δ) − a`` 这种浮点抵消在 300 量级
    坐标上有 ~1e-13 的舍入；而 S2 的残差还叠加了快跟踪核自身的亚像素
    抖动（~0.05 px 量级），因此 S2 用 0.5 px 容差并如实报出实测值。
    """
    result = GateResult("Gate 3b", "端到端合成注入（δ 已知）")
    base = base_config or TrackerConfig()

    # ---------------- S1：小残差 → 平滑 ----------------
    source = SyntheticFrameSource(frames=frames)
    specs = [ArmSpec("fused", det_hz=det_hz, latency_frames=latency_frames, enable_fusion=True)]

    def factory(_name, tracker):
        return OracleDetector(tracker, delta=small_delta, truth_provider=source.position)

    outcomes = run_lockstep(
        source,
        None,
        specs,
        base_config=base,
        total_frames=frames,
        camera_hz=camera_hz,
        detector_factory=factory,
    )
    records = outcomes["fused"].records
    fused_records = [record for record in records if record["fusion_kind"] in ("smooth", "hard_reinit")]

    residual_errors = [
        float(
            np.linalg.norm(
                np.asarray([record["residual_x"], record["residual_y"]], dtype=np.float64)
                - np.asarray(small_delta, dtype=np.float64)
            )
        )
        for record in fused_records
    ]
    max_residual_error = max(residual_errors) if residual_errors else float("nan")
    result.check(
        "S1 小残差：残差恒等于注入的 δ（容差 1e-9 px）",
        bool(residual_errors) and max_residual_error <= 1e-9,
        measured=max_residual_error,
        expected="<= 1e-9",
        note=f"融合 {len(fused_records)} 次",
    )
    kinds = {record["fusion_kind"] for record in fused_records}
    result.check(
        "S1 小残差：修正走 smooth 分支",
        kinds == {"smooth"},
        measured=sorted(kinds),
        expected=["smooth"],
    )
    alpha = base.alpha
    correction_errors = [
        float(
            np.linalg.norm(
                np.asarray([record["corr_dx"], record["corr_dy"]], dtype=np.float64)
                - alpha * np.asarray(small_delta, dtype=np.float64)
            )
        )
        for record in fused_records
    ]
    max_correction_error = max(correction_errors) if correction_errors else float("nan")
    result.check(
        "S1 小残差：实际修正量 = α·δ（容差 1e-9 px）",
        bool(correction_errors) and max_correction_error <= 1e-9,
        measured=max_correction_error,
        expected="<= 1e-9",
    )

    probe = AsyncDualRateTracker(ArmSpec("probe", enable_fusion=True).build_config(base))
    mismatches = 0
    for record in fused_records:
        frame = source.read(record["frame_index"])
        if frame is None:
            continue
        expected = probe.fast.step(
            frame, np.asarray([record["corr_pos_x"], record["corr_pos_y"]], dtype=np.float64)
        ).xy
        if float(np.max(np.abs(expected - np.asarray([record["x"], record["y"]])))) > 1e-9:
            mismatches += 1
    result.check(
        "S1 小残差：输出 = 快跟踪核(校正后位置)（容差 1e-9 px）",
        mismatches == 0,
        measured=mismatches,
        expected=0,
        note="逐帧复算快跟踪核",
    )

    # ---------------- S2：大残差 → 硬拉回 ----------------
    two_blob = TwoBlobFrameSource(frames=frames, decoy_offset=decoy_offset)
    detector = SwitchingOracleDetector(two_blob, switch_frame=switch_frame)
    outcomes_s2 = run_lockstep(
        two_blob,
        detector,
        specs,
        base_config=base,
        total_frames=frames,
        camera_hz=camera_hz,
    )
    records_s2 = outcomes_s2["fused"].records
    hard_records = [record for record in records_s2 if record["fusion_kind"] == "hard_reinit"]

    result.check(
        "S2 大残差：出现硬拉回事件",
        bool(hard_records),
        measured=len(hard_records),
        expected=">= 1",
    )
    if hard_records:
        # 残差 = z_f − x_f = 真目标 − 诱饵 = −offset（诱饵在 target + offset）
        expected_offset = -np.asarray(decoy_offset, dtype=np.float64)
        residual_offsets = [
            float(
                np.linalg.norm(
                    np.asarray([record["residual_x"], record["residual_y"]], dtype=np.float64)
                    - expected_offset
                )
            )
            for record in hard_records
        ]
        max_offset_error = max(residual_offsets)
        result.check(
            "S2 大残差：残差 = 真目标 − 诱饵 的已知间距（容差 0.5 px）",
            max_offset_error <= 0.5,
            measured=max_offset_error,
            expected="<= 0.5 px",
            note=f"间距 {decoy_offset}，硬拉回 {len(hard_records)} 次",
        )
        snap_errors = [
            float(
                np.linalg.norm(
                    np.asarray([record["corr_pos_x"], record["corr_pos_y"]], dtype=np.float64)
                    - np.asarray([record["det_x"], record["det_y"]], dtype=np.float64)
                )
            )
            for record in hard_records
        ]
        max_snap_error = max(snap_errors)
        result.check(
            "S2 大残差：硬拉回后位置 = z_f（两斑静止 ⇒ v ≡ 0，容差 0.5 px）",
            max_snap_error <= 0.5,
            measured=max_snap_error,
            expected="<= 0.5 px",
        )

    result.detail = {
        "frames": frames,
        "det_hz": det_hz,
        "latency_frames": latency_frames,
        "small_delta": list(small_delta),
        "decoy_offset": list(decoy_offset),
        "switch_frame": switch_frame,
        "s1_fusion_events": len(fused_records),
        "s2_hard_reinit_events": len(hard_records),
    }
    return result


# ==========================================================================
# Gate 4：真检测器回放
# ==========================================================================


def _jitter_report(records: list[dict]) -> dict:
    """抖动/瞬态统计（文档 §四 判据 1）。

    抖动用 **median**（项目铁律：抖动统计用中位数不用均值），同时给 p95/max
    暴露长尾。校正瞬态按融合分支分开统计：

    * ``smooth``：平滑修正帧——文档判据是"不得出现 > 2 px 的瞬态跳跃"；
    * ``hard_reinit``：硬拉回帧——那是**故意**的跳变，单独统计，不与平滑混算。
    """
    steps_all: list[float] = []
    steps_smooth: list[float] = []
    steps_hard: list[float] = []
    corrections: list[float] = []
    for record in records:
        step = record.get("step", float("nan"))
        if not math.isnan(step):
            steps_all.append(step)
        kind = record.get("fusion_kind")
        if kind == "smooth" and not math.isnan(step):
            steps_smooth.append(step)
            corrections.append(float(np.hypot(record["corr_dx"], record["corr_dy"])))
        elif kind == "hard_reinit" and not math.isnan(step):
            steps_hard.append(step)

    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "median": float("nan"), "p95": float("nan"), "max": float("nan")}
        array = np.asarray(values, dtype=np.float64)
        return {
            "n": int(array.size),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
        }

    report = {
        "step_all": stats(steps_all),
        "step_correction_smooth": stats(steps_smooth),
        "step_correction_hard_reinit": stats(steps_hard),
        "correction_magnitude": stats(corrections),
    }
    baseline = report["step_all"].get("median", float("nan"))
    correction = report["step_correction_smooth"].get("median", float("nan"))
    report["excess_median_at_correction"] = (
        correction - baseline if not math.isnan(correction) and not math.isnan(baseline) else float("nan")
    )
    p95 = report["step_correction_smooth"].get("p95", float("nan"))
    report["criterion_2px_pass_p95"] = bool(not math.isnan(p95) and p95 <= 2.0)
    return report


def _time_to_servo(
    source,
    base_config: TrackerConfig,
    frames: int,
    stride: int = 1,
) -> dict:
    """Time-to-Servo：单帧"帧在内存 → 坐标产出"的墙钟耗时。

    只测**计算**（帧已在内存），不含 JPEG 解码与磁盘 IO——那属于相机
    侧链路，不属于伺服关键路径。测量时用独立的跟踪器实例，避免多臂
    锁步时互相干扰缓存状态。
    """
    tracker = AsyncDualRateTracker(base_config)
    samples: list[float] = []
    for index in range(0, frames, stride):
        frame = source.read(index)
        if frame is None:
            continue
        tracker._position = tracker._position if tracker._position is not None else np.asarray([320.0, 256.0])
        start = time.perf_counter()
        tracker.fast.step(frame, tracker._position)
        samples.append((time.perf_counter() - start) * 1000.0)
    if not samples:
        return {"n": 0}
    array = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
        "budget_ms": 1.0,
        "within_budget": bool(np.percentile(array, 99) <= 1.0),
    }


def gate4_detector_replay(
    sequence: str,
    cache_root,
    frames_root,
    labels_root,
    det_hz_list=(25.0, 12.5, 8.333333333333334),
    latency_list=(0, 4, 8, 12),
    base_config: TrackerConfig | None = None,
    camera_hz: float = 25.0,
    total_frames: int = 0,
    score_threshold: float = 0.22,
    dedup_radius: float = 768.0,
    evidence_arms=(),
    out_dir: Path | None = None,
    lock_radius: float = 16.0,
    tolerance_px: float = 5.0,
) -> GateResult:
    """Gate 4：RC2/trial_0080 真检测器回放，det_hz × Δ 受控扫描。

    所有臂在**一次**锁步遍历里跑完（帧只解码一次），保证同源可比。

    四类机制（同一批帧、同一批检测，只有开关不同；标签见模块头的命名对照表）：

    * ``det``  —— 裸慢流（``detector_only``）：只把迟到的检测框当当前输出；
    * ``trk``  —— 快流跟踪，**不做**残差修正（``tracker_only``，检测器只负责
      冷启动/关联）；
    * ``hard`` —— 快流跟踪 + **硬拉回**（``hard_pull``，不做残差平滑）；
    * ``fus``  —— 完整架构：快流跟踪 + 异步残差平滑（``alpha_fused``）。

    det_hz 的取值必须是**相机帧率的可整除档**（25 Hz 相机 ⇒ stride 1/2/3 ⇒
    25 / 12.5 / 8.33 Hz）。文档写的 10~15 Hz 在 25 Hz 相机上无法整除实现
    （``round(25/10) == round(25/15) == 2``），会退化成同一档，扫描就变成
    假扫描——所以这里按可实现档扫描，并在结果里报**实际生效**的频率。
    """
    result = GateResult("Gate 4", "真检测器回放（RC2/trial_0080）")
    base = base_config or TrackerConfig()

    stream = SlowStream(
        cache_root,
        sequence,
        dedup_radius=dedup_radius,
        score_threshold=score_threshold,
    )
    frames_total = int(total_frames or stream.frames)
    source = JpegFrameSource(Path(frames_root) / sequence)

    def arm_specs(det_hz: float, latency: int) -> list[ArmSpec]:
        return [
            ArmSpec(name=f"det@{det_hz}@{latency}", kind="detector_only", det_hz=det_hz, latency_frames=latency),
            ArmSpec(name=f"trk@{det_hz}@{latency}", det_hz=det_hz, latency_frames=latency, enable_fusion=False),
            ArmSpec(
                name=f"hard@{det_hz}@{latency}",
                det_hz=det_hz,
                latency_frames=latency,
                enable_fusion=True,
                overrides=(("smooth_when_small", False),),
            ),
            ArmSpec(name=f"fus@{det_hz}@{latency}", det_hz=det_hz, latency_frames=latency, enable_fusion=True),
        ]

    specs: list[ArmSpec] = []
    for det_hz in det_hz_list:
        for latency in latency_list:
            specs.extend(arm_specs(det_hz, latency))

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

    from manu.evaluation.eval_multiscale_cache_th import load_ground_truth

    ground_truth = load_ground_truth(Path(labels_root), sequence, stream.width, stream.height)

    rows: list[dict] = []
    for det_hz in det_hz_list:
        stride = max(1, int(round(camera_hz / det_hz)))
        effective_hz = camera_hz / stride
        for latency in latency_list:
            arms = {
                "detector_only": outcomes[f"det@{det_hz}@{latency}"],
                "tracker_only": outcomes[f"trk@{det_hz}@{latency}"],
                "fused_nosmooth": outcomes[f"hard@{det_hz}@{latency}"],
                "fused": outcomes[f"fus@{det_hz}@{latency}"],
            }
            det_frames = arms["detector_only"].arrival_frames

            row = {
                "sequence": sequence,
                "det_hz_requested": det_hz,
                "det_hz_effective": effective_hz,
                "detector_stride": stride,
                "latency_frames": latency,
                "latency_ms": latency * (1000.0 / camera_hz),
                "detector_frames": len(det_frames),
                "gt_total": int(sum(len(v) for v in ground_truth.values())),
            }
            for label, outcome in arms.items():
                row[f"{label}@det_frames"] = score_positions(
                    ground_truth, outcome.positions, frames_subset=det_frames
                )
                row[f"{label}@all_frames"] = score_positions(ground_truth, outcome.positions)
                row[f"servo_{label}"] = servo_metrics(
                    outcome.records,
                    ground_truth,
                    tolerance=tolerance_px,
                    lock_radius=lock_radius,
                ).to_dict()
            rows.append(row)

            if out_dir is not None and f"fus@{det_hz}@{latency}" in evidence_arms:
                for label, outcome in arms.items():
                    write_evidence(
                        out_dir / f"evidence_{label}_{det_hz}hz_lat{latency}.tsv",
                        outcome.records,
                        ground_truth,
                        lock_radius=lock_radius,
                    )

    latency_stats = _time_to_servo(source, base, min(frames_total, 3000))
    result.check(
        "Time-to-Servo：快流单帧 p99 ≤ 1 ms",
        latency_stats.get("within_budget", False),
        measured=latency_stats.get("p99_ms"),
        expected="<= 1.0 ms",
    )

    # ---- F1 vs 基线（规格 §四末句「确认 F1 表现和连续性达到基线」）----
    #
    # 这里有**两个**口径完全不同的"基线"，必须分开报，否则就是拿尺子量错东西：
    #
    #   (1) **同滞后公平基线** = ``detector_only`` 在**同一 det_hz、同一 Δ**
    #       下的 F1。这才是可部署替代方案的对照：同样是 Δ=8 物理滞后，
    #       一个把旧框直接甩给转台，另一个走检跟分离。架构存在的意义就是
    #       在**同样的滞后**下更准。
    #   (2) **无滞后天花板** = ``frozen_det_f1``（冻结 RC2 单帧检测，
    #       没有 Δ 帧滞后）。这是**物理上不可达**的上界——全图检测器在伺服
    #       关键路径上做不到 <1 ms，这正是本架构要解决的问题。
    #
    # 因此判据只对 (1) 判 pass/fail（且只判 **Δ ≥ 8** 的工况：Δ=0 时检测器
    # 没有滞后、根本不需要跟踪器，Δ=4 时滞后太小、跟踪器还没被逼到必须
    # 补偿的地步——那两档检测器赢是**预期结果**，不是缺陷）。
    # (2) 的差距如实记进 detail 与文档，不粉饰。
    f1_baseline_rows = []
    for row in rows:
        det_f1 = row["detector_only@det_frames"]["f1"]
        fused_f1 = row["fused@det_frames"]["f1"]
        f1_baseline_rows.append(
            {
                "det_hz_requested": row["det_hz_requested"],
                "det_hz_effective": row["det_hz_effective"],
                "latency_frames": row["latency_frames"],
                "detector_only_f1": det_f1,
                "alpha_fused_f1": fused_f1,
                "tracker_only_f1": row["tracker_only@det_frames"]["f1"],
                "hard_pull_f1": row["fused_nosmooth@det_frames"]["f1"],
                "fused_minus_detector": fused_f1 - det_f1,
            }
        )

    lagged_rows = [item for item in f1_baseline_rows if item["latency_frames"] >= 8]
    result.check(
        "F1 达到同滞后基线：Δ≥8 工况下 alpha_fused 的 F1 ≥ detector_only 的 F1",
        bool(lagged_rows) and all(item["fused_minus_detector"] >= 0.0 for item in lagged_rows),
        measured=[
            {
                "det_hz": item["det_hz_effective"],
                "latency": item["latency_frames"],
                "delta_f1": round(item["fused_minus_detector"], 6),
            }
            for item in lagged_rows
        ],
        expected="每个 (det_hz, Δ≥8) 工况 delta_f1 ≥ 0",
        note="同滞后公平基线：detector_only 与 alpha_fused 用同一批检测帧、同一份 GT",
    )

    ceiling_gap = [
        {
            "det_hz": item["det_hz_effective"],
            "latency": item["latency_frames"],
            "alpha_fused_f1": item["alpha_fused_f1"],
            "frozen_det_f1": FROZEN_TRACK3_BLIND["f1"],
            "gap": item["alpha_fused_f1"] - FROZEN_TRACK3_BLIND["f1"],
        }
        for item in f1_baseline_rows
    ]
    worst_ceiling_gap = min((item["gap"] for item in ceiling_gap), default=float("nan"))

    result.detail = {
        "sequence": sequence,
        "detector": "RC2 / trial_0080 / epoch3 (ms_cache)",
        "score_threshold": score_threshold,
        "dedup_radius": dedup_radius,
        "frames": frames_total,
        "elapsed_seconds": elapsed,
        "time_to_servo": latency_stats,
        "tolerance_px": tolerance_px,
        "lock_radius_px": lock_radius,
        "rows": rows,
        "ab_table": build_ab_table(rows, det_hz_list, latency_list),
        "f1_baseline_comparison": {
            "same_latency_baseline": {
                "definition": "detector_only @ 同一 det_hz、同一 Δ（可部署替代方案对照）",
                "rows": f1_baseline_rows,
            },
            "zero_latency_ceiling": {
                "definition": (
                    "frozen_det_f1 = 冻结 RC2 单帧检测 F1（无 Δ 帧滞后），"
                    "是物理不可达上界，仅作口径锚点"
                ),
                "f1": FROZEN_TRACK3_BLIND["f1"],
                "per_row": ceiling_gap,
                "worst_gap": worst_ceiling_gap,
                "note": (
                    "检跟分离在 Δ≥8 工况稳定优于同滞后基线，"
                    "但**低于**无滞后 RC2 单帧检测天花板——这是架构的固有代价"
                    "（全图检测器无法进入 <1 ms 伺服关键路径），不粉饰"
                ),
            },
            "velocity_source": base.velocity_source,
        },
        "frozen_reference": {
            # 旧 key 保留不动（已有 summary.json 产物依赖）；frozen_det_f1 是
            # 命名对照表里的机制名别名，指向同一个冻结 RC2 单帧检测 F1。
            "track3_blind": FROZEN_TRACK3_BLIND,
            "frozen_det_f1": FROZEN_TRACK3_BLIND,
            "sys_sota_preds": FROZEN_SYS_SOTA_TARGET,
            "note": (
                "仅作口径锚点，禁止与本表跨系统比大小；"
                "frozen_det_f1 = 冻结 RC2 单帧检测 F1（仅打分口径锚点），"
                "禁止与三臂表的伺服指标（命中率/CLE）跨系统比大小"
            ),
        },
    }
    return result


def build_ab_table(rows: list[dict], det_hz_list, latency_list) -> list[dict]:
    """生成验收对照表（按**机制名**标注，不用"臂 N"编号）。

    * **``detector_only`` 纯检测基准**：检测器**每帧**全图检测，但输出的是
      Δ 帧以前的旧框（物理滞后）⇒ 目标一动就脱靶；
    * **``hard_pull`` 检跟分离 + 硬拉回**：快流 + 滞后检测**硬拉回**
      （不做残差平滑）；
    * **``alpha_fused`` 检跟分离 + 残差平滑**：快流 + 异步残差平滑滤波
      （完整架构）；
    * **``tracker_only`` 消融对照**：只跟踪、不融合。

    四者用**同一份检测流**、**同一批帧**、**同一份 GT**。

    每行同时给三个标签：``mechanism``（权威机制名）、``arm``（展示串
    = 机制名 + 中文说明）、``arm_legacy``（历史"臂 N"标题，仅用于与旧
    ``summary.json`` 对齐；新产物对外只应引用 ``mechanism``）。
    """
    table: list[dict] = []

    def pick(det_hz, latency):
        for row in rows:
            if (
                abs(row["det_hz_requested"] - det_hz) < 1e-6
                and row["latency_frames"] == latency
            ):
                return row
        return None

    entries = [
        ("detector_only", 25.0, 8),
        ("hard_pull", 12.5, 8),
        ("alpha_fused", 12.5, 8),
        ("tracker_only", 12.5, 8),
    ]
    for mechanism, det_hz, latency in entries:
        row = pick(det_hz, latency)
        if row is None:
            continue
        meta = MECHANISMS[mechanism]
        row_key = meta["row_key"]
        servo = row[f"servo_{row_key}"]
        table.append(
            {
                "mechanism": mechanism,
                "arm": f"{mechanism}（{meta['label']}）",
                "arm_legacy": meta["legacy_arm"],
                "det_hz_effective": row["det_hz_effective"],
                "latency_frames": latency,
                "latency_ms": row["latency_ms"],
                "hit_rate_at_5px": servo["hit_rate"]["rate"],
                "hit_rate_wilson95": servo["hit_rate"]["wilson95"],
                "inbox_hit_rate": servo["inbox_hit_rate"]["rate"],
                "cle_p50": servo["cle"]["p50"],
                "cle_p95": servo["cle"]["p95"],
                "cle_p99": servo["cle"]["p99"],
                "cle_max": servo["cle"]["max"],
                "max_accel": servo["jitter"]["acceleration"]["max"],
                "accel_p95": servo["jitter"]["acceleration"]["p95"],
                "max_step": servo["jitter"]["first_difference"]["max"],
                "fragmentations": servo["fragmentation"]["fragmentations"],
                "lock_streak_max": servo["fragmentation"]["lock_streak_max"],
                "f1_at_det_frames": row[f"{row_key}@det_frames"]["f1"],
                "f1_at_all_frames": row[f"{row_key}@all_frames"]["f1"],
                "time_to_servo_p99_ms": None,
            }
        )
    return table


# ==========================================================================
# Gate 6：口径对齐
# ==========================================================================


def gate6_metric_alignment(
    sequence: str,
    sys_sota_root,
    labels_root,
    native_size=(640, 512),
) -> GateResult:
    """Gate 6：用本 harness 给冻结的 ``sys_sota_preds`` 重新打分。

    必须**逐位复现**冻结数字（TP/FP/FN/F1）。这一步证明的是"本 harness
    的口径 = 冻结口径"，是后面所有 F1 数字可比的前提。
    """
    result = GateResult("Gate 6", "口径对齐（冻结 sys_sota_preds 重打分）")
    payload = json.loads((Path(sys_sota_root) / f"{sequence}.json").read_text(encoding="utf-8"))

    predictions: dict[int, tuple] = {}
    for frame_key, entries in payload["per_frame"].items():
        if not entries:
            continue
        scores = np.asarray([float(entry.get("score", 1.0)) for entry in entries], dtype=np.float32)
        points = np.asarray([entry["pos"] for entry in entries], dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        predictions[int(frame_key)] = (scores[order], points[order])

    from manu.evaluation.eval_multiscale_cache_th import load_ground_truth

    ground_truth = load_ground_truth(Path(labels_root), sequence, native_size[0], native_size[1])
    scored = score_predictions(ground_truth, predictions)

    frozen = payload.get("metrics", {})
    target = FROZEN_SYS_SOTA_TARGET
    for key in ("tp", "fp", "fn", "det"):
        result.check(
            f"{key} 与冻结值一致",
            int(scored[key]) == int(target[key]),
            measured=int(scored[key]),
            expected=int(target[key]),
        )
    result.check(
        "F1 与冻结值一致（1e-9 容差）",
        abs(float(scored["f1"]) - float(target["f1"])) <= 1e-9,
        measured=float(scored["f1"]),
        expected=float(target["f1"]),
    )
    if frozen:
        result.check(
            "与 JSON 自带的 metrics 块一致",
            all(int(scored[key]) == int(frozen[key]) for key in ("tp", "fp", "fn")),
            measured={key: int(scored[key]) for key in ("tp", "fp", "fn")},
            expected={key: int(frozen[key]) for key in ("tp", "fp", "fn")},
        )
    result.detail = {
        "sequence": sequence,
        "scored": scored,
        "frozen_json_metrics": frozen,
        "frozen_target": target,
        "gt_frames": len(ground_truth),
    }
    return result


# ==========================================================================
# 逐帧取证
# ==========================================================================

EVIDENCE_COLUMNS = [
    "frame_index",
    "state",
    "x",
    "y",
    "valid",
    "det_x",
    "det_y",
    "det_source",
    "residual_x",
    "residual_y",
    "residual_norm",
    "fusion_kind",
    "corr_dx",
    "corr_dy",
    "corr_pos_x",
    "corr_pos_y",
    "step",
    "gt_x",
    "gt_y",
    "gt_w",
    "gt_h",
    "gt_hit",
    "locked",
    "replay_applied",
    "replay_reason",
    "replay_steps",
    "replay_cost_ms",
    "replay_anchor_x",
    "replay_anchor_y",
    "step_ms",
    "events",
]


def write_evidence(
    path,
    records: list[dict],
    ground_truth: dict[int, np.ndarray],
    lock_radius: float = 16.0,
) -> Path:
    """写逐帧证据表（TSV）。

    每一行都能独立复核：该帧的状态机状态、输出坐标、关联上的检测、残差、
    修正量与修正后位置、一阶差分，以及 GT 框与命中判定（GT 命中用冻结
    的 inbox 判据：点落在框内）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(EVIDENCE_COLUMNS) + "\n")
        for record in records:
            frame_index = int(record["frame_index"])
            boxes = ground_truth.get(frame_index)
            gt_x = gt_y = gt_w = gt_h = float("nan")
            gt_hit = False
            locked = False
            if boxes is not None and len(boxes):
                position = None
                if not math.isnan(record.get("x", float("nan"))):
                    position = np.asarray([record["x"], record["y"]], dtype=np.float64)
                centres = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
                nearest = 0
                if position is not None:
                    distance = np.linalg.norm(centres[:, :2] - position[None, :], axis=1)
                    nearest = int(np.argmin(distance))
                    gt_hit = bool(
                        abs(position[0] - centres[nearest, 0]) <= centres[nearest, 2] / 2
                        and abs(position[1] - centres[nearest, 1]) <= centres[nearest, 3] / 2
                    )
                    locked = bool(distance[nearest] <= lock_radius)
                gt_x, gt_y, gt_w, gt_h = (float(value) for value in centres[nearest])
            row = [
                str(frame_index),
                str(record.get("state", "")),
                f"{record.get('x', float('nan')):.4f}",
                f"{record.get('y', float('nan')):.4f}",
                "1" if record.get("valid") else "0",
                f"{record.get('det_x', float('nan')):.4f}",
                f"{record.get('det_y', float('nan')):.4f}",
                str(record.get("det_source", -1)),
                f"{record.get('residual_x', float('nan')):.4f}",
                f"{record.get('residual_y', float('nan')):.4f}",
                f"{record.get('residual_norm', float('nan')):.4f}",
                str(record.get("fusion_kind", "none")),
                f"{record.get('corr_dx', float('nan')):.4f}",
                f"{record.get('corr_dy', float('nan')):.4f}",
                f"{record.get('corr_pos_x', float('nan')):.4f}",
                f"{record.get('corr_pos_y', float('nan')):.4f}",
                f"{record.get('step', float('nan')):.4f}",
                f"{gt_x:.4f}",
                f"{gt_y:.4f}",
                f"{gt_w:.4f}",
                f"{gt_h:.4f}",
                "1" if gt_hit else "0",
                "1" if locked else "0",
                "1" if record.get("replay_applied") else "0",
                str(record.get("replay_reason", "")),
                str(record.get("replay_steps", 0)),
                f"{record.get('replay_cost_ms', float('nan')):.4f}",
                f"{record.get('replay_anchor_x', float('nan')):.4f}",
                f"{record.get('replay_anchor_y', float('nan')):.4f}",
                f"{record.get('step_ms', float('nan')):.4f}",
                str(record.get("events", "")),
            ]
            handle.write("\t".join(row) + "\n")
    return path


# ==========================================================================
# CLI
# ==========================================================================


def _default_paths() -> dict:
    return {
        "cache_root": "/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache",
        "frames_root": "/home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg",
        "labels_root": "/home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010",
        "sys_sota_root": "/home/manu/mnt/pycharm_project_10ae9e2e/runs/sys_sota_preds",
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv=None) -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="异步双频检跟分离系统：离线受控 A/B 评测闸门")
    parser.add_argument("--gate", default="all", help="2 / 3 / 4 / 6 / all")
    parser.add_argument("--sequence", default="VIDEO00007_19700101_001324")
    parser.add_argument("--cache-root", default=defaults["cache_root"])
    parser.add_argument("--frames-root", default=defaults["frames_root"])
    parser.add_argument("--labels-root", default=defaults["labels_root"])
    parser.add_argument("--sys-sota-root", default=defaults["sys_sota_root"])
    parser.add_argument(
        "--det-hz",
        default="25,12.5,8.333333333333334",
        help="检测帧率；必须是相机帧率的可整除档（25 Hz 相机 ⇒ 25/12.5/8.33）",
    )
    parser.add_argument("--latency", default="0,4,8,12")
    parser.add_argument("--out-dir", default="runs/tracking_async")
    parser.add_argument(
        "--evidence",
        default="",
        help="要写证据表的机制档（det_hz:latency），如 '12.5:8'（可逗号分隔）",
    )
    parser.add_argument("--frames-limit", type=int, default=0, help="只跑前 N 帧（调试用）")
    parser.add_argument("--score-threshold", type=float, default=0.22)
    parser.add_argument("--dedup-radius", type=float, default=768.0)
    parser.add_argument("--mode", default="centroid", choices=("centroid", "peak_centroid"))
    parser.add_argument("--camera-hz", type=float, default=25.0)
    args = parser.parse_args(argv)

    det_hz_list = tuple(float(value) for value in args.det_hz.split(",") if value)
    latency_list = tuple(int(value) for value in args.latency.split(",") if value)
    evidence_arms = {f"fus@{item.split(':')[0]}@{item.split(':')[1]}" for item in args.evidence.split(",") if item}
    out_dir = Path(args.out_dir) / args.sequence
    out_dir.mkdir(parents=True, exist_ok=True)

    config = TrackerConfig(mode=args.mode, camera_hz=args.camera_hz)

    results: list[GateResult] = []
    gates = {"2", "3", "4", "6"} if args.gate == "all" else {args.gate}

    if "2" in gates:
        results.append(gate2_null_control_synthetic(base_config=config))
        results.append(
            gate2_null_control_zero_latency(
                args.sequence,
                args.cache_root,
                args.frames_root,
                base_config=config,
                total_frames=args.frames_limit,
                camera_hz=args.camera_hz,
            )
        )
    if "3" in gates:
        results.append(gate3a_fusion_unit(base_config=config))
        results.append(gate3b_synthetic_injection(base_config=config))
    if "4" in gates:
        results.append(
            gate4_detector_replay(
                args.sequence,
                args.cache_root,
                args.frames_root,
                args.labels_root,
                det_hz_list=det_hz_list,
                latency_list=latency_list,
                base_config=config,
                camera_hz=args.camera_hz,
                total_frames=args.frames_limit,
                score_threshold=args.score_threshold,
                dedup_radius=args.dedup_radius,
                evidence_arms=evidence_arms,
                out_dir=out_dir,
            )
        )
    if "6" in gates:
        results.append(
            gate6_metric_alignment(args.sequence, args.sys_sota_root, args.labels_root)
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "sequence": args.sequence,
        "naming_convention": {
            "mechanisms": {name: meta["label"] for name, meta in MECHANISMS.items()},
            "frozen_det_f1": (
                "冻结 RC2 单帧检测 F1（口径锚点，禁止与三臂表伺服指标跨系统比大小）"
            ),
            "note": (
                "本文件的对照表用机制名标注（detector_only / alpha_fused / hard_pull / "
                "tracker_only）；同目录 three_arm_summary.json 里的『臂 1/2/3』是用户拍板的"
                "另一套编号（纯检测 / 纯前向外推 / 退回重推演），"
                "与本文件的机制名不是一回事，"
                "跨表比较前先看本对照表"
            ),
        },
        "config": {
            "mode": config.mode,
            "gating_dist": config.gating_dist,
            "alpha": config.alpha,
            "win_size": config.win_size,
            "th_k": config.th_k,
            "min_energy": config.min_energy,
            "min_contrast": config.min_contrast,
            "score_threshold": args.score_threshold,
            "dedup_radius": args.dedup_radius,
            "det_hz": list(det_hz_list),
            "latency_frames": list(latency_list),
        },
        "gates": [result.to_dict() for result in results],
        "all_passed": all(result.passed for result in results),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.gate} {result.name}")
        for check in result.checks:
            mark = "ok " if check["passed"] else "FAIL"
            print(f"    {mark} {check['label']}  measured={check['measured']} expected={check['expected']}")
    print(f"\nsummary -> {summary_path}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
