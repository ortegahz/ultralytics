"""评测臂（arm）：同源帧、同源检测，只切换"残差融合是否生效"这一个开关。

**受控 A/B 的铁律在这里落地**：所有臂在**同一次锁步遍历**里推进，因此

* 看到的是**逐字节相同**的帧（同一份 JPEG 解码结果，同一个数组对象）；
* 看到的是**同一批检测**（同一个慢流对象、同一个到达时刻）；
* 唯一的差别是配置里的 ``enable_fusion``（以及 detector_only 臂压根没有跟踪器）。

任何其它差别（阈值、初始化、关联规则）都会让 A/B 失去归因能力，所以这里
刻意把臂的差异面收敛到最小。

**三个对照臂（验收表的主干）**：

===================== ====================================================
臂                    机制
===================== ====================================================
臂 1 纯检测基准       不跟踪。第 t 帧直接甩第 t−Δ 帧算出来的旧检测框
                      （det_hz=25，Δ=8 ⇒ 每帧都有输出，静态硬吃 320 ms）
臂 2 纯前向外推       快核每帧前推；迟到检测**只对比、不回溯、不修正**，
                      且**永不**用检测重新锚定位置
臂 3 退回重推演       快核 + 帧环形缓存；迟到检测作绝对锚点，退回第 t−Δ 帧
                      逐帧重推 Δ−1 步，由主流程走完第 t 帧，覆盖当前状态
===================== ====================================================

另有若干**消融臂**（``alpha`` / ``hard`` / ``no_smooth`` / ``tracker_only``）
用来定位"收益到底来自哪个开关"。

此外有一条**公平性对照臂** ``arm1_detector_only_sparse``：机制与臂 1
**完全相同**，只把 ``det_hz`` 从 25 降到 12.5（锚点密度对齐到臂 2 / 臂 3），
用来隔离"臂 1 锚点更密"这个混淆变量。它**不是**验收表的三条臂之一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np

from .detector import Detection, DetectorScheduler
from .tracker import AsyncDualRateTracker, TrackerConfig, TrackerState

__all__ = [
    "SlowStream",
    "JpegFrameSource",
    "SyntheticFrameSource",
    "OracleDetector",
    "TwoBlobFrameSource",
    "SwitchingOracleDetector",
    "ArmSpec",
    "ArmOutcome",
    "run_lockstep",
    "DETECTOR_ONLY",
    "arm1_detector_only",
    "arm1_detector_only_sparse",
    "arm2_forward_only",
    "arm3_rollback_replay",
    "canonical_arms",
]

DETECTOR_ONLY = "detector_only"


# ==============================================================================
# 慢流：冻结候选构造 → Detection
# ==============================================================================


class SlowStream:
    """慢流检测器输出。

    候选构造**直接复用冻结实现**（``load_predictions`` + ``merge_scales``）：
    三尺度（tag 0/1/2）跨尺度按分数降序去重（``dedup_radius``），刻意排除
    烘焙融合行（tag 255，因为它的地板 0.22 会清空弱脉冲挽救池）。

    这里**不**复用冻结流水线的天空抑制 / 连续性滤波 / 双向平滑那些层：
    本 harness 的 A/B 只关心"融合开关"这一个变量，把旧流水线的层叠进来会
    让归因失效。代价是本 harness 的 detector_only 臂精度低于冻结 SOTA 基线，
    这一点在结果表里必须写清楚，不允许跨系统直接比大小。

    Parameters
    ----------
    cache_root:
        ``ms_cache`` 目录（含 ``<SEQ>.pkl``）。
    sequence:
        序列名。
    dedup_radius:
        跨尺度去重半径（native px）。冻结基线用 768（≈ 每帧只剩 1 个候选）。
    score_threshold:
        慢流分数门限。冻结单帧基线用 0.22。
    """

    def __init__(
        self,
        cache_root,
        sequence: str,
        dedup_radius: float = 768.0,
        score_threshold: float = 0.22,
        tags: Sequence[int] = (0, 1, 2),
    ) -> None:
        from manu.evaluation.eval_multiscale_cache_th import (  # 延迟 import：避免循环依赖
            load_predictions,
            merge_scales,
        )

        root = Path(cache_root)
        meta, by_tag = load_predictions(root, sequence, list(tags), 0.0)
        candidates = merge_scales(by_tag, dedup_radius)

        self.meta = meta
        self.sequence = sequence
        self.frames = int(meta["frames"])
        self.width = int(meta["native_size"][0])
        self.height = int(meta["native_size"][1])
        self.score_threshold = float(score_threshold)
        self._by_frame: dict[int, list[Detection]] = {}
        for frame_index, (scores, points) in candidates.items():
            found: list[Detection] = []
            for score, point in zip(scores.tolist(), points.tolist()):
                if score < self.score_threshold:
                    continue
                found.append(
                    Detection(
                        frame_index=int(frame_index),
                        xy=np.asarray(point, dtype=np.float64),
                        score=float(score),
                    )
                )
            if found:
                self._by_frame[int(frame_index)] = found

    def detections(self, frame_index: int) -> list[Detection]:
        """该帧的全部检测（按分数降序）。"""
        return list(self._by_frame.get(int(frame_index), ()))

    def best(self, frame_index: int) -> Detection | None:
        found = self.detections(frame_index)
        return found[0] if found else None

    def covered_frames(self) -> list[int]:
        return sorted(self._by_frame)


# ==============================================================================
# 帧源
# ==============================================================================


class JpegFrameSource:
    """从目录读灰度帧（``frame_%06d.jpg``）。"""

    def __init__(self, directory, pattern: str = "frame_%06d.jpg", max_cached: int = 16) -> None:
        self.directory = Path(directory)
        self.pattern = pattern
        self.max_cached = int(max_cached)
        """帧缓存上限。整段序列解码后是 3.9 GB，必须限量；锁步遍历里每帧
        只读一次，缓存只对"复测同一段"有用。"""
        self._cache: dict[int, np.ndarray] = {}

    def path(self, frame_index: int) -> Path:
        return self.directory / (self.pattern % int(frame_index))

    def read(self, frame_index: int) -> np.ndarray | None:
        cached = self._cache.get(int(frame_index))
        if cached is not None:
            return cached
        import cv2

        image = cv2.imread(str(self.path(frame_index)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        if self.max_cached > 0:
            if len(self._cache) >= self.max_cached:
                self._cache.clear()
            self._cache[int(frame_index)] = image
        return image


class SyntheticFrameSource:
    """合成帧源：一个在画面里匀速平移的高斯亮斑（红外弱小目标的理想化替身）。

    用于 Gate 2/3 的合成注入臂：不碰真实数据 ⇒ 判据可以写成解析式。
    """

    def __init__(
        self,
        frames: int = 240,
        size: tuple[int, int] = (512, 640),
        start: tuple[float, float] = (320.0, 256.0),
        velocity: tuple[float, float] = (0.6, 0.2),
        amplitude: float = 120.0,
        background: float = 30.0,
        sigma: float = 2.0,
    ) -> None:
        self.frames = int(frames)
        self.height, self.width = size
        self.start = start
        self.velocity = velocity
        self.amplitude = float(amplitude)
        self.background = float(background)
        self.sigma = float(sigma)
        self._yy, self._xx = np.mgrid[0 : self.height, 0 : self.width]
        self._cache: dict[int, np.ndarray] = {}

    def position(self, frame_index: int) -> tuple[float, float]:
        return (
            self.start[0] + self.velocity[0] * frame_index,
            self.start[1] + self.velocity[1] * frame_index,
        )

    def read(self, frame_index: int) -> np.ndarray | None:
        cached = self._cache.get(int(frame_index))
        if cached is not None:
            return cached
        cx, cy = self.position(frame_index)
        blob = self.amplitude * np.exp(
            -(((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / (2 * self.sigma**2))
        )
        image = np.clip(self.background + blob, 0, 255).astype(np.uint8)
        self._cache[int(frame_index)] = image
        return image


class OracleDetector:
    """合成注入臂用的"完美检测器"：直接读跟踪器自己的历史位置。

    ``detections(f)`` 返回 ``历史位置[f] + δ``。因为它读的就是**同一个臂**
    的历史缓冲，所以残差 ``e = z_f − x_f`` 恒等于 ``δ``（精确到浮点误差），
    融合的输出因此有解析解可比对。

    这不是"作弊"：它模拟的是"检测器完美复现了快跟踪器当时看到的目标位置
    再叠加一个已知偏差 δ"，用来把融合算术本身钉死。真实检测器的回放放在
    Gate 4。
    """

    def __init__(
        self,
        tracker: AsyncDualRateTracker,
        delta: tuple[float, float] = (0.0, 0.0),
        score: float = 0.9,
        truth_provider=None,
    ) -> None:
        self.tracker = tracker
        self.delta = np.asarray(delta, dtype=np.float64)
        self.score = float(score)
        self.frames = 0
        self.truth_provider = truth_provider
        """冷启动兜底：历史里还没有该帧时，返回"真值位置 + δ"。

        文档 §二.4 规定冷启动由检测器完成，所以合成注入臂必须先让检测器
        用真值把跟踪器拉进 TRACK 态，之后才切换到"历史位置 + δ"。
        """

    def detections(self, frame_index: int) -> list[Detection]:
        position = self.tracker.history.get(int(frame_index))
        if position is None:
            if self.truth_provider is None:
                return []
            fallback = self.truth_provider(int(frame_index))
            if fallback is None:
                return []
            position = np.asarray(fallback, dtype=np.float64)
        return [
            Detection(
                frame_index=int(frame_index),
                xy=position + self.delta,
                score=self.score,
            )
        ]


class TwoBlobFrameSource:
    """两个静止亮斑：真目标 ``T`` 与诱饵 ``D = T + offset``。

    用来构造文档 §二.3 "情况 B（大残差/失配）" 的物理场景：快跟踪核锁在
    诱饵上，检测器随后报出真目标 ⇒ 残差很大 ⇒ 走硬拉回。两个斑都是静止
    的，所以速度 ``v ≡ 0``，硬拉回的解析解退化成 ``position_after = z_f``。
    """

    def __init__(
        self,
        frames: int = 240,
        size: tuple[int, int] = (512, 640),
        target: tuple[float, float] = (300.0, 256.0),
        decoy_offset: tuple[float, float] = (40.0, 0.0),
        amplitude: float = 120.0,
        background: float = 30.0,
        sigma: float = 2.0,
    ) -> None:
        self.frames = int(frames)
        self.height, self.width = size
        self.target = target
        self.decoy_offset = decoy_offset
        self.amplitude = float(amplitude)
        self.background = float(background)
        self.sigma = float(sigma)
        self._yy, self._xx = np.mgrid[0 : self.height, 0 : self.width]
        self._cache: dict[int, np.ndarray] = {}

    def position(self, frame_index: int) -> np.ndarray:
        """真目标位置。"""
        return np.asarray(self.target, dtype=np.float64)

    def decoy_position(self, frame_index: int) -> np.ndarray:
        """诱饵位置。"""
        return np.asarray(self.target, dtype=np.float64) + np.asarray(self.decoy_offset, dtype=np.float64)

    def read(self, frame_index: int) -> np.ndarray | None:
        cached = self._cache.get(int(frame_index))
        if cached is not None:
            return cached
        image = np.full((self.height, self.width), self.background, dtype=np.float32)
        for centre in (self.position(frame_index), self.decoy_position(frame_index)):
            blob = self.amplitude * np.exp(
                -(((self._xx - centre[0]) ** 2 + (self._yy - centre[1]) ** 2) / (2 * self.sigma**2))
            )
            image = np.maximum(image, blob)
        result = np.clip(image, 0, 255).astype(np.uint8)
        self._cache[int(frame_index)] = result
        return result


class SwitchingOracleDetector:
    """先报诱饵、到 ``switch_frame`` 之后改报真目标的"检测器"。

    模拟检测器重新锁定正确目标的那一瞬间：残差从"跟踪器锁在诱饵上"变成
    "真目标 − 诱饵"，是个**已知的大常数** ⇒ 硬拉回分支有解析解可比对。
    """

    def __init__(
        self,
        source,
        switch_frame: int,
        score: float = 0.9,
    ) -> None:
        self.source = source
        self.switch_frame = int(switch_frame)
        self.score = float(score)
        self.frames = 0

    def detections(self, frame_index: int) -> list[Detection]:
        index = int(frame_index)
        if index < self.switch_frame:
            position = self.source.decoy_position(index)
        else:
            position = self.source.position(index)
        return [Detection(frame_index=index, xy=position, score=self.score)]


# ==============================================================================
# 锁步遍历
# ==============================================================================


@dataclass(frozen=True)
class ArmSpec:
    """一个臂的完整定义。

    Attributes
    ----------
    name:
        结果表里的键名。
    kind:
        ``"tracker"``（有跟踪器）或 ``"detector_only"``（裸慢流）。
    det_hz:
        该臂的检测器帧率。
    latency_frames:
        该臂的检测滞后 Δ（帧）。
    enable_fusion:
        是否施加残差修正。受控 A/B 的**唯一**开关。
    overrides:
        额外覆盖的 ``TrackerConfig`` 字段。
    """

    name: str
    kind: str = "tracker"
    det_hz: float = 12.5
    latency_frames: int = 8
    enable_fusion: bool = True
    fusion_mode: str = "alpha"
    """定位修正机制：``"alpha"`` / ``"replay"`` / ``"none"``（见 ``TrackerConfig``）。"""

    detection_reinit: bool = True
    """是否允许检测流重新锚定位置（臂 2 设为 False）。"""

    overrides: tuple[tuple[str, object], ...] = ()

    def build_config(self, base: TrackerConfig) -> TrackerConfig:
        config = TrackerConfig(**{**base.__dict__})
        config.enable_fusion = self.enable_fusion
        config.fusion_mode = self.fusion_mode
        config.detection_reinit = self.detection_reinit
        for key, value in self.overrides:
            setattr(config, key, value)
        return config


def arm1_detector_only() -> ArmSpec:
    """臂 1：纯检测基准臂（Baseline Arm / 现状基准）。

    不做任何跟踪。第 t 帧输出的就是第 t−Δ 帧算出来的旧框（静态硬吃
    320 ms 物理滞后），用来量化"脱离跟踪辅助、硬扛滞后"的真实脱靶损失。

    ``det_hz`` 取 25（= 相机帧率）：基准臂代表"每帧都跑一次全图检测"的现状。
    """
    return ArmSpec(
        name="arm1_detector_only",
        kind=DETECTOR_ONLY,
        det_hz=25.0,
        latency_frames=8,
    )


def arm1_detector_only_sparse(det_hz: float = 12.5, latency_frames: int = 8) -> ArmSpec:
    """臂 1 的**同锚点密度**变体（公平性对照臂）。

    机制与 :func:`arm1_detector_only` **完全相同**——纯检测、不跟踪，
    第 t 帧直接甩第 t−Δ 帧算出来的旧检测框；**唯一差别是 ``det_hz=12.5``**
    （stride=2，每 2 帧才有一个锚点）。返回值结构（``ArmSpec`` / 字段 /
    产出记录）与之完全一致，接入方无需任何特殊处理。

    **为什么需要这个变体**：臂 1 用 ``det_hz=25``（stride=1，每帧一个锚点），
    而臂 2 / 臂 3 用 ``det_hz=12.5``（stride=2，每 2 帧一个锚点）⇒
    **臂 1 的检测信息量是另外两条臂的 2 倍**。三条臂因此不在同一信息量下，
    "臂 3 追平臂 1"这类结论会被"臂 1 锚点更密"这个混淆变量污染。
    本变体把臂 1 的锚点密度降到与臂 2 / 臂 3 相同，让"退回重推演到底
    有没有用"在**同信息量**下重新对照。

    ⚠️ 它**不是**用户拍板的三条臂之一，**不进验收表**，只是公平性对照臂：
    唯一职责是把"锚点密度"这个混淆变量单独隔离出来。
    """
    return ArmSpec(
        name="arm1_detector_only_sparse",
        kind=DETECTOR_ONLY,
        det_hz=float(det_hz),
        latency_frames=int(latency_frames),
    )


def arm2_forward_only() -> ArmSpec:
    """臂 2：检跟纯前向外推臂（Forward-Only / 无回溯）。

    快核每帧基于上一帧坐标前推一步并输出。迟到检测**只做对比**（残差照样
    算、照样进逐帧证据），但**不回溯、不修正**；``detection_reinit=False``
    保证检测流**永远不能**把位置搬回某个旧检测点——"完全不回头纠偏"的极限
    工况，用来量快核自身的抗累积漂移能力与自持周期。
    """
    return ArmSpec(
        name="arm2_forward_only",
        kind="tracker",
        det_hz=12.5,
        latency_frames=8,
        fusion_mode="none",
        detection_reinit=False,
    )


def arm3_rollback_replay() -> ArmSpec:
    """臂 3：异步退回重推演臂（Rollback & Replay / 现阶段精度理论上限）。

    迟到检测框当作第 t−Δ 帧的**绝对锚点**，从帧环形缓存里调出历史图像切片，
    逐帧重推 Δ−1 步到第 t−1 帧，由主流程用当前帧正常走完第 t 帧，用重算
    出来的**无漂移**坐标覆盖当前状态并输出。**不加任何人为平滑衰减**。
    """
    return ArmSpec(
        name="arm3_rollback_replay",
        kind="tracker",
        det_hz=12.5,
        latency_frames=8,
        fusion_mode="replay",
    )


def canonical_arms() -> list[ArmSpec]:
    """验收表用的三个对照臂（顺序 = 表格顺序）。"""
    return [arm1_detector_only(), arm2_forward_only(), arm3_rollback_replay()]


@dataclass
class ArmOutcome:
    """一个臂跑完之后的全部产物。"""

    name: str
    positions: dict[int, np.ndarray] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    arrival_frames: list[int] = field(default_factory=list)
    """慢流检测"到达"的帧号（= 源帧号 + Δ）。同一调度参数下所有臂共享。"""


def run_lockstep(
    source,
    stream,
    specs: Sequence[ArmSpec],
    base_config: TrackerConfig | None = None,
    total_frames: int = 0,
    camera_hz: float = 25.0,
    detector_factory: Callable[[str, AsyncDualRateTracker], object] | None = None,
) -> dict[str, ArmOutcome]:
    """一次遍历，所有臂同帧推进。

    Parameters
    ----------
    source:
        帧源（``read(frame_index) -> HxW uint8 | None``）。
    stream:
        慢流检测器（``detections(frame_index) -> list[Detection]``）。
    specs:
        臂定义列表。不同 ``det_hz`` / ``latency_frames`` 的臂可以在**同一次
        遍历**里一起跑——帧只解码一次，实验成本降一个数量级。
    base_config:
        所有臂共享的基准配置（各臂再叠加自己的 overrides）。
    detector_factory:
        可选，``(arm_name, tracker) -> detector``。合成注入臂用它把
        "完美检测器"绑到该臂自己的历史缓冲上；为 None 时用 ``stream``。
    """
    base = base_config or TrackerConfig()
    trackers: dict[str, AsyncDualRateTracker] = {}
    schedulers: dict[str, DetectorScheduler] = {}
    for spec in specs:
        if spec.kind == DETECTOR_ONLY:
            detector = stream
        else:
            tracker = AsyncDualRateTracker(spec.build_config(base))
            trackers[spec.name] = tracker
            detector = detector_factory(spec.name, tracker) if detector_factory else stream
        schedulers[spec.name] = DetectorScheduler(detector, camera_hz, spec.det_hz, spec.latency_frames)

    outcomes = {spec.name: ArmOutcome(spec.name) for spec in specs}
    previous: dict[str, np.ndarray | None] = {spec.name: None for spec in specs}

    for index in range(int(total_frames)):
        frame = source.read(index)
        if frame is None:
            continue
        for spec in specs:
            outcome = outcomes[spec.name]
            arrivals = schedulers[spec.name].poll(index)
            if arrivals:
                outcome.arrival_frames.append(index)

            if spec.kind == DETECTOR_ONLY:
                position = None
                score = float("nan")
                source_frame = -1
                if arrivals:
                    source_frame, found = arrivals[0]
                    if found:
                        best = max(found, key=lambda det: det.score)
                        position = best.xy.copy()
                        score = float(best.score)
                if position is not None:
                    outcome.positions[index] = position
                outcome.records.append(
                    {
                        "frame_index": index,
                        "state": "DETECT",
                        "x": float(position[0]) if position is not None else float("nan"),
                        "y": float(position[1]) if position is not None else float("nan"),
                        "valid": position is not None,
                        "det_x": float(position[0]) if position is not None else float("nan"),
                        "det_y": float(position[1]) if position is not None else float("nan"),
                        "det_score": score,
                        "det_source": source_frame,
                        "residual_x": float("nan"),
                        "residual_y": float("nan"),
                        "residual_norm": float("nan"),
                        "fusion_kind": "none",
                        "corr_dx": 0.0,
                        "corr_dy": 0.0,
                        "corr_pos_x": float("nan"),
                        "corr_pos_y": float("nan"),
                        "step": float("nan"),
                        "step_ms": 0.0,
                        "replay_applied": False,
                        "replay_reason": "",
                        "replay_steps": 0,
                        "replay_cost_ms": float("nan"),
                        "replay_anchor_x": float("nan"),
                        "replay_anchor_y": float("nan"),
                        "events": "",
                    }
                )
                previous[spec.name] = position
                continue

            tracker = trackers[spec.name]
            step_started = perf_counter()
            out = tracker.step(index, frame, arrivals)
            step_ms = (perf_counter() - step_started) * 1000.0
            fusion = out.fusion
            detection = out.detection
            position = out.xy

            correction = np.zeros(2, dtype=np.float64)
            if fusion is not None and fusion.applied:
                correction = np.asarray(fusion.position_after, dtype=np.float64) - np.asarray(
                    fusion.position_before, dtype=np.float64
                )
            step = float("nan")
            if position is not None and previous[spec.name] is not None:
                step = float(np.linalg.norm(np.asarray(position) - np.asarray(previous[spec.name])))
            if position is not None:
                outcome.positions[index] = np.asarray(position, dtype=np.float64)
            outcome.records.append(
                {
                    "frame_index": index,
                    "state": out.state.value if isinstance(out.state, TrackerState) else str(out.state),
                    "x": float(position[0]) if position is not None else float("nan"),
                    "y": float(position[1]) if position is not None else float("nan"),
                    "valid": bool(out.valid),
                    "det_x": float(detection[0]) if detection is not None else float("nan"),
                    "det_y": float(detection[1]) if detection is not None else float("nan"),
                    "det_score": float("nan"),
                    "det_source": arrivals[0][0] if arrivals else -1,
                    "residual_x": float(fusion.residual[0]) if fusion is not None else float("nan"),
                    "residual_y": float(fusion.residual[1]) if fusion is not None else float("nan"),
                    "residual_norm": float(fusion.residual_norm) if fusion is not None else float("nan"),
                    "fusion_kind": fusion.kind if fusion is not None else "none",
                    "corr_dx": float(correction[0]),
                    "corr_dy": float(correction[1]),
                    "corr_pos_x": float(fusion.position_after[0]) if fusion is not None else float("nan"),
                    "corr_pos_y": float(fusion.position_after[1]) if fusion is not None else float("nan"),
                    "step": step,
                    "step_ms": step_ms,
                    "replay_applied": bool(out.replay.applied) if out.replay is not None else False,
                    "replay_reason": out.replay.reason if out.replay is not None else "",
                    "replay_steps": int(out.replay.steps) if out.replay is not None else 0,
                    "replay_cost_ms": (
                        float(out.replay.cost_ms) if out.replay is not None else float("nan")
                    ),
                    "replay_anchor_x": (
                        float(out.replay.anchor_xy[0]) if out.replay is not None else float("nan")
                    ),
                    "replay_anchor_y": (
                        float(out.replay.anchor_xy[1]) if out.replay is not None else float("nan")
                    ),
                    "events": "|".join(out.events),
                }
            )
            previous[spec.name] = None if position is None else np.asarray(position, dtype=np.float64)

    return outcomes
