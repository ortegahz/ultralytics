"""三态有限状态机：把快流（每帧）与慢流（低频 + 滞后）编排成一个系统。

文档 §二.4：

* ``SEARCH``：跟踪器待机，帧直接丢给检测器；检测器**连续 K 次**在稳定空域给出
  高置信度目标后进入 ``TRACK``，用最新框初始化跟踪器。
* ``TRACK``：快跟踪器每帧输出坐标（低延迟送出）；检测器在后台异步核对，
  确认吻合就刷新置信度、修正漂移。
* ``LOST``：跟踪器局部能量消失（SNR 骤降）或检测器连续多帧未在目标区检出时，
  允许外推维持若干帧；仍未恢复就退回 ``SEARCH`` 重新拉网。

每个物理帧的推进顺序（``fuse_first=True``，文档伪代码的顺序）：

1. **慢流回溯校正**：处理本帧到达的滞后检测（对账时刻 = 检测的源帧号）；
2. **快流前向跟踪**：在（可能被修正过的）位置上做一次局部提取。

先融合再跟踪的好处：修正量会改变本帧局部搜索窗的中心，快跟踪器在**已经
校正过的**位置上收敛 ⇒ 伺服看到的位置不跳变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .fast_tracker import LocalPatchTracker, TrackerResult
from .fusion import FusionConfig, FusionDecision, ResidualFusion
from .history import HistoryRingBuffer
from .replay import FrameRingBuffer, ReplayResult, replay_forward

__all__ = ["TrackerState", "TrackerConfig", "TrackerOutput", "AsyncDualRateTracker"]


class TrackerState(str, Enum):
    """三态。"""

    SEARCH = "SEARCH"
    TRACK = "TRACK"
    LOST = "LOST"


@dataclass
class TrackerConfig:
    """系统全部可调参数（默认值 = 文档给的那一套）。"""

    # -- 快流 --
    win_size: int = 32
    th_k: float = 3.0
    min_energy: float = 1.0
    min_contrast: float = 5.0
    bg_window: int = 64
    """底噪估计窗边长。质心窗仍是 32（文档判据"32×32 视野"不变）。"""
    mode: str = "centroid"
    peak_window: int = 9

    # -- 融合 --
    gating_dist: float = 12.0
    alpha: float = 0.4
    hard_reinit: bool = True
    project_velocity: bool = True
    """硬拉回时是否把 Δ 帧前的检测位置按速度投影到当前时刻。"""

    velocity_source: str = "none"
    """硬拉回 / 初始化投影的速度源（``"none"`` / ``"history"`` / ``"detection"``）。

    默认 ``"none"`` = **不投影**，即文档 §三 伪代码骨架的语义
    （``self.curr_pos = np.array(det_xy)``）。语义与实测依据见
    :class:`~manu.tracking.async_dual_rate.fusion.FusionConfig`。
    """
    smooth_when_small: bool = True
    """小残差是否走平滑。关掉 = "无残差平滑"消融臂（任何残差都硬拉回）。"""
    enable_fusion: bool = True
    """总开关。关掉 = "只跟踪不融合"臂（受控 A/B 的另一半）。

    关掉时仍然**计算**残差并记进逐帧证据，只是不把修正作用到位置上——
    于是两个臂的差异严格只来自"修正是否生效"这一个开关。
    """

    fusion_mode: str = "alpha"
    """定位修正的**机制**（三个对照臂的主开关）。

    ``"alpha"``
        文档 §二.3 的异步延迟残差融合：小残差 ``x ← x + α·e`` 平滑注入，
        大残差硬拉回。**不回溯历史帧**。
    ``"replay"``
        臂 3「退回重推演」：迟到的检测框当作第 ``t−Δ`` 帧的**绝对锚点**，
        从缓存图像里逐帧重推 Δ−1 步到第 ``t−1`` 帧，覆盖当前状态后由主流程
        正常走完第 ``t`` 帧。**清零累积漂移**，不加任何平滑衰减。
    ``"none"``
        臂 2「纯前向外推」：迟到的检测框**只做对比**（残差照样算、照样记进
        逐帧证据），但**绝不修改位置**，也绝不回头纠偏。
    """

    replay_max_dist: float = 96.0
    """臂 3 的锚点可信门限（像素）。

    退回重推演把迟到检测框当成**真值锚点**用，所以必须先确认它确实是
    同一个目标：锚点与"历史轨迹在第 f 帧的位置"距离超过这个门限，就说明
    要么检测是假阳性、要么轨迹已经漂到别处——此时弃权（保留前向状态），
    不要拿一个可疑锚点去重推整条轨迹。

    门限取得比 :attr:`assoc_max_dist` 宽，是因为 Δ=8 帧的名义位移本身
    就有几十像素（实测 p90 ≈ 46 px），门限太窄会毙掉合法的锚点。
    """

    replay_require_valid: bool = True
    """重推途中快核若判定"信号消失"，是否整体弃权（默认 True）。"""

    replay_fallback: str = "none"
    """重推演**弃权**时的兜底处置。

    ``"none"``（规范默认）
        保留原状态机的前向位置（带累积漂移）。这是用户定义的臂 3 原始口径，
        **不动**。
    ``"hard"``（消融）
        退化成"锚点 + 检测流速度外推"（等价于残差融合的硬拉回分支）。

    实测依据：龙泉山 001324 序列上**尝试重推演的帧里有 20.0%（824/4,117）弃权**，
    弃权帧上臂 3 的 CLE p50 = 27.46 px，而纯检测臂只有 14.41 px —— 弃权时
    "保留漂移的前向状态"是纯粹的亏损。``"hard"`` 兜底就是把这部分亏损拿回来。
    """

    detection_reinit: bool = True
    """是否允许**检测流**重新锚定跟踪器位置。

    ``True``（臂 1/臂 3）：SEARCH 态可以引导、LOST 态可以重捕。
    ``False``（臂 2「纯前向外推」）：检测流**只能**在最开始引导一次，
    之后**永远不许**把位置搬回某个旧检测点——这是"完全不回头纠偏"的
    极限工况，用来量快核自身的抗漂移能力。
    """

    frame_cache_capacity: int = 0
    """帧环形缓冲深度；0 = 自动取 ``latency_frames + 2``。"""

    # -- 时钟 --
    camera_hz: float = 25.0
    det_hz: float = 12.5
    latency_frames: int = 8

    # -- 状态机 --
    init_consec: int = 2
    """SEARCH → TRACK 需要的连续检测确认次数（文档说"如 2 帧"）。"""

    init_min_score: float = 0.25
    """参与初始化的检测分数下限。"""

    init_max_disp: float = 12.0
    """连续检测位置之间的最大允许位移（"稳定空域"）。"""

    init_max_frame_gap: int = 4
    """SEARCH → TRACK 的 K 次确认之间，源帧号最大允许的间隔（相机帧）。

    规格说检测器要"连续 K 帧"确认，连续必须是**到达时刻连续**，不能是
    "攒够 K 个样本就算"。慢流 det_hz=12.5 Hz 对应 stride=2，det_hz=25 Hz
    对应 stride=1；本门限取 4 帧对这两种配置都成立，而检测器停摆超过
    4 个相机帧时，之前攒下的样本作废、重新累计。
    """

    track_to_lost_frames: int = 2
    """TRACK → LOST 的判据：快跟踪核连续多少帧"能量消失"（SNR 骤降）。

    取 2 而不是 1，是为了容忍单帧噪声/遮挡造成的假失效；连续两帧都失效
    才认为快跟踪核真的脱靶了。
    """

    det_miss_budget: int = 3
    """TRACK → LOST 的另一个判据：检测器连续多少次"在别处检出"（或在目标区
    完全没检出）。计数只在**有检测到达但没能关联上**时递增，检测器本身
    低频（12.5 Hz vs 25 Hz）不计入。
    """

    max_lost_frames: int = 5
    """LOST 态最多**对外输出**多少帧外推位置，超时退回 SEARCH（文档说 3~5 帧）。

    注意这是"输出帧数"而不是"内部计数"：转入 LOST 的那一帧算第 1 帧，
    之后每帧外推一次；第 ``max_lost_frames + 1`` 帧不再输出位置（退回 SEARCH）。
    """

    lost_capture_dist: float = 48.0
    """LOST 态的捕获门限：检测框离外推位置多远才算重新捕获到目标。"""

    assoc_max_dist: float = 64.0
    """TRACK 态的关联门限：太远的检测直接忽略（避免被远处杂波拉走）。"""

    history_capacity: int = 64
    """历史环形缓冲深度，必须 > latency_frames。"""


@dataclass
class TrackerOutput:
    """一帧的完整输出（逐帧取证的原料）。"""

    frame_index: int
    state: TrackerState
    xy: np.ndarray | None
    valid: bool
    """位置是否可信（TRACK 态且快跟踪器能量足够 ⇒ True）。"""

    fusion: FusionDecision | None = None
    tracker: TrackerResult | None = None
    events: tuple[str, ...] = ()
    detection: np.ndarray | None = None
    """本帧到达的检测框中心（若有关联上的检测）。"""

    replay: ReplayResult | None = None
    """臂 3 退回重推演的完整记录（其它模式恒为 None）。"""


@dataclass
class _DetectionSample:
    frame_index: int
    xy: np.ndarray
    score: float


@dataclass
class AsyncDualRateTracker:
    """异步双频检跟分离系统（Python 参考实现）。"""

    config: TrackerConfig = field(default_factory=TrackerConfig)

    def __post_init__(self) -> None:
        cfg = self.config
        if cfg.fusion_mode not in ("alpha", "replay", "none"):
            raise ValueError(
                f"unknown fusion_mode: {cfg.fusion_mode!r} "
                "(expected one of 'alpha' / 'replay' / 'none')"
            )
        if cfg.history_capacity <= cfg.latency_frames:
            raise ValueError(
                "history_capacity must exceed latency_frames "
                f"({cfg.history_capacity} <= {cfg.latency_frames})"
            )
        self.fast = LocalPatchTracker(
            win_size=cfg.win_size,
            th_k=cfg.th_k,
            min_energy=cfg.min_energy,
            min_contrast=cfg.min_contrast,
            mode=cfg.mode,
            peak_window=cfg.peak_window,
            bg_window=cfg.bg_window,
        )
        self.fusion = ResidualFusion(
            FusionConfig(
                gating_dist=cfg.gating_dist,
                alpha=cfg.alpha,
                hard_reinit=cfg.hard_reinit,
                project_velocity=cfg.project_velocity,
                velocity_source=cfg.velocity_source,
                smooth_when_small=cfg.smooth_when_small,
            )
        )
        self.history = HistoryRingBuffer(cfg.history_capacity)
        self.frame_cache = FrameRingBuffer(
            cfg.frame_cache_capacity or (cfg.latency_frames + 2)
        )

        self.state = TrackerState.SEARCH
        self._position: np.ndarray | None = None
        self._lost_count = 0
        self._weak_count = 0
        """快跟踪核连续失效帧数（TRACK → LOST 的判据之一）。"""
        self._det_miss_count = 0
        """检测器连续"在别处检出"次数（TRACK → LOST 的判据之二）。"""
        self._init_samples: list[_DetectionSample] = []
        self._det_history: list[_DetectionSample] = []
        """最近若干次关联上的检测，用于估计速度（重捕投影用）。"""

        self._last_detection: np.ndarray | None = None
        self._bootstrapped: bool = False
        """是否已经用检测流引导过一次。

        臂 2「纯前向外推」靠它实现"引导一次、之后永不回头"：一旦置位，
        后续 SEARCH 态不再接受检测引导（``_reset_search`` **不清**这个标志，
        否则"失锁—重引导"会成为一条隐形的纠偏通道）。
        """

    # -- 主接口 -------------------------------------------------------------

    def step(self, frame_index: int, frame: np.ndarray, arrivals) -> TrackerOutput:
        """推进一个物理帧。

        Parameters
        ----------
        frame_index:
            物理帧号。
        frame:
            当前相机的原始灰度帧（快流输入）。
        arrivals:
            本帧到达的慢流检测，形式 ``list[(source_frame_index, [Detection, ...])]``。
        """
        events: list[str] = []
        fusion_decision: FusionDecision | None = None
        tracker_result: TrackerResult | None = None
        detection_xy: np.ndarray | None = None
        replay_result: ReplayResult | None = None

        # 帧进缓存：臂 3「退回重推演」要回放历史图像切片。
        # 存引用不拷贝（锁步遍历里同一帧被所有臂共享）。
        self.frame_cache.push(frame_index, frame)

        # ---- 1. 慢流回溯校正 -------------------------------------------
        for source_frame, detections in arrivals:
            chosen = self._associate(detections, source_frame)
            if chosen is None:
                # 检测结果到了，但没有一个落在目标区 ⇒ 记一次"检测器漏检"。
                # 只在 TRACK 态计数：SEARCH 态本来就没有"目标区"可言。
                if self.state is TrackerState.TRACK:
                    self._det_miss_count += 1
                continue
            if self.state is TrackerState.TRACK:
                self._det_miss_count = 0
            detection_xy = chosen.xy.copy()
            self._last_detection = detection_xy
            self._det_history.append(
                _DetectionSample(source_frame, detection_xy.copy(), float(chosen.score))
            )
            if len(self._det_history) > 8:
                self._det_history.pop(0)

            if self.state is TrackerState.SEARCH:
                if not (self.config.detection_reinit or not self._bootstrapped):
                    # 臂 2「纯前向外推」：引导过一次之后，检测流再也不许
                    # 把位置搬回某个旧检测点。
                    events.append("init:blocked")
                elif self._accumulate_init(source_frame, chosen):
                    self._position = self._project_to_now(chosen, source_frame, frame_index)
                    self.state = TrackerState.TRACK
                    self._bootstrapped = True
                    events.append("SEARCH->TRACK")
                    self._lost_count = 0
                    self._weak_count = 0
                    self._det_miss_count = 0
            elif self.state is TrackerState.TRACK:
                assert self._position is not None
                if self.config.fusion_mode == "replay":
                    replay_result = self._run_replay(chosen, source_frame, frame_index)
                    fusion_decision = self._replay_decision(
                        replay_result, chosen, source_frame
                    )
                    if replay_result.applied and self.config.enable_fusion:
                        events.append("replay:ok")
                    elif replay_result.applied:
                        events.append("replay:disabled")
                    else:
                        events.append(f"replay:abstain:{replay_result.reason}")
                        if self.config.replay_fallback == "hard":
                            fallback = self.fusion.fuse(
                                self._position,
                                chosen.xy,
                                source_frame,
                                frame_index,
                                self.history,
                                fallback_velocity=self._detection_velocity(),
                            )
                            if fallback.applied:
                                self._position = fallback.position_after.copy()
                                fusion_decision = fallback
                                events.append("replay:fallback:hard_reinit")
                else:
                    # 历史对账不可用（同刻位置取不回来）时，融合自己会弃权——
                    # 判据统一放在 ``ResidualFusion`` 里，避免两处规则不一致。
                    fusion_decision = self.fusion.fuse(
                        self._position,
                        chosen.xy,
                        source_frame,
                        frame_index,
                        self.history,
                        fallback_velocity=self._detection_velocity(),
                    )
                    if self.config.fusion_mode == "none":
                        # 臂 2「纯前向外推」：只对比，不修正。
                        events.append("fusion:compare_only")
                    elif fusion_decision.applied and self.config.enable_fusion:
                        self._position = fusion_decision.position_after.copy()
                        events.append(f"fusion:{fusion_decision.kind}")
                    elif fusion_decision.applied:
                        events.append("fusion:disabled")
                    else:
                        events.append("fusion:abstain")
            else:  # LOST
                if self.config.fusion_mode == "replay":
                    # 臂 3 在 LOST 态也用退回重推演重捕：这是它把目标"拉回来"
                    # 的唯一手段（不靠把旧检测点直接搬到当前位置）。
                    replay_result = self._run_replay(chosen, source_frame, frame_index)
                    fusion_decision = self._replay_decision(
                        replay_result, chosen, source_frame
                    )
                    if replay_result.applied and self.config.enable_fusion:
                        self.state = TrackerState.TRACK
                        events.append("LOST->TRACK:replay")
                        self._lost_count = 0
                        self._weak_count = 0
                        self._det_miss_count = 0
                    else:
                        events.append(f"replay:abstain:{replay_result.reason}")
                        if self.config.replay_fallback == "hard":
                            fallback = self.fusion.fuse(
                                self._position,
                                chosen.xy,
                                source_frame,
                                frame_index,
                                self.history,
                                fallback_velocity=self._detection_velocity(),
                            )
                            if fallback.applied:
                                self._position = fallback.position_after.copy()
                                fusion_decision = fallback
                                events.append("replay:fallback:hard_reinit")
                elif self.config.detection_reinit and self._capture(chosen, source_frame):
                    self._position = self._project_to_now(chosen, source_frame, frame_index)
                    self.state = TrackerState.TRACK
                    events.append("LOST->TRACK")
                    self._lost_count = 0
                    self._weak_count = 0
                    self._det_miss_count = 0

        # ---- 2. 快流前向跟踪 -------------------------------------------
        if self.state is TrackerState.SEARCH:
            return TrackerOutput(
                frame_index=frame_index,
                state=self.state,
                xy=None,
                valid=False,
                fusion=fusion_decision,
                tracker=None,
                events=tuple(events),
                detection=detection_xy,
                replay=replay_result,
            )

        assert self._position is not None
        tracker_result = self.fast.step(frame, self._position)
        self._position = tracker_result.xy.copy()

        if self.state is TrackerState.TRACK:
            if tracker_result.valid:
                self._weak_count = 0
                self._lost_count = 0
            else:
                self._weak_count += 1
                if self._weak_count >= self.config.track_to_lost_frames:
                    self.state = TrackerState.LOST
                    events.append("TRACK->LOST:energy")
                    self._lost_count = 0
            if self.state is TrackerState.TRACK and self._det_miss_count >= self.config.det_miss_budget:
                self.state = TrackerState.LOST
                events.append("TRACK->LOST:det_miss")
                self._lost_count = 0
        else:  # LOST：仍然跑快跟踪器（可能自愈），但状态要等检测器确认
            self._lost_count += 1

        if self.state is TrackerState.LOST:
            if not tracker_result.valid:
                # 文档 §二.4：LOST 态"允许外推维持若干帧"。快跟踪核没能量时
                # 位置保持不动，这里用历史速度把它推出去，避免输出完全冻结。
                velocity = self.history.latest_velocity()
                if velocity is not None:
                    self._position = self._position + velocity
            if self._lost_count >= self.config.max_lost_frames:
                self.state = TrackerState.SEARCH
                events.append("LOST->SEARCH")
                self._reset_search()

        if self._position is None:
            # LOST→SEARCH 之后没有位置可写（``_reset_search`` 把位置清了）。
            return TrackerOutput(
                frame_index=frame_index,
                state=self.state,
                xy=None,
                valid=False,
                fusion=fusion_decision,
                tracker=tracker_result,
                events=tuple(events),
                detection=detection_xy,
                replay=replay_result,
            )

        self.history.push(frame_index, self._position)

        return TrackerOutput(
            frame_index=frame_index,
            state=self.state,
            xy=self._position.copy(),
            valid=(self.state is TrackerState.TRACK and tracker_result.valid),
            fusion=fusion_decision,
            tracker=tracker_result,
            events=tuple(events),
            detection=detection_xy,
            replay=replay_result,
        )

    # -- 关联 ---------------------------------------------------------------

    def _run_replay(self, chosen, source_frame: int, current_frame: int) -> ReplayResult:
        """臂 3：退回重推演。

        迟到的检测框被当作第 ``source_frame`` 帧的**绝对锚点**，
        从缓存图像里逐帧重推到 ``current_frame - 1``，然后由主流程用当前帧
        正常走完最后一步。

        重推成功后：
        * 当前状态（位置）被重算结果覆盖；
        * 历史缓冲里 ``[source_frame, current_frame-1]`` 的位置一并回写，
          否则后续帧的"同刻对账"残差还会拿旧的前向轨迹做基准。
        """
        cfg = self.config
        anchor = np.asarray(chosen.xy, dtype=np.float64).reshape(2).copy()

        # 锚点必须可验证：拿不到第 source_frame 帧的历史位置，就没法确认
        # 这个检测框确实是"同一个目标"。此时**弃权**——不能拿一个无法验证的
        # 检测框当绝对锚点去重推整条轨迹。
        #
        # （锚点的"距离可信门限"由 :meth:`_associate` 统一把关：replay 模式下
        # 关联门限取 ``replay_max_dist``，超限的检测根本不会被选中。这里
        # 不再重复设第二道门限，避免两处规则不一致。）
        if not self.history.has(source_frame):
            return ReplayResult(
                applied=False,
                kind="replay_abstain",
                reason="no_history_for_anchor",
                anchor_frame=int(source_frame),
                anchor_xy=anchor,
                target_frame=int(current_frame),
                steps=0,
            )

        result = replay_forward(
            self.fast,
            self.frame_cache,
            anchor,
            source_frame,
            current_frame,
            require_valid=cfg.replay_require_valid,
        )
        if not result.applied:
            return result

        position = result.positions.get(int(current_frame) - 1)
        if position is None:
            return ReplayResult(
                applied=False,
                kind="replay_abstain",
                reason="no_position_at_target_minus_1",
                anchor_frame=int(source_frame),
                anchor_xy=anchor,
                target_frame=int(current_frame),
                steps=result.steps,
                positions=result.positions,
                cost_ms=result.cost_ms,
            )

        self._position = np.asarray(position, dtype=np.float64).copy()

        # 回写历史：锚点帧 + 重推出来的每一帧。
        for frame_index, xy in result.positions.items():
            self.history.overwrite(int(frame_index), xy)
        return result

    def _replay_decision(
        self, result: ReplayResult, chosen, source_frame: int
    ) -> FusionDecision:
        """把一次退回重推演包装成 :class:`FusionDecision`（逐帧取证的统一格式）。

        这里的残差 ``e = z_{t−Δ} − x_{t−Δ}`` 仍然是"同刻对账"的量，含义是
        **重推之前、快核在第 f 帧累积了多大的漂移**——这正是退回重推演要
        消掉的东西。
        """
        before = self._position.copy() if self._position is not None else np.zeros(2)
        after = before.copy()
        historical = self.history.get(source_frame)
        if historical is not None:
            residual = np.asarray(chosen.xy, dtype=np.float64) - historical
            residual_norm = float(np.linalg.norm(residual))
        else:
            residual = np.full(2, np.nan)
            residual_norm = float("nan")
        return FusionDecision(
            applied=result.applied,
            kind="replay" if result.applied else "replay_abstain",
            residual=residual,
            residual_norm=residual_norm,
            position_before=before,
            position_after=after,
        )

    def _associate(self, detections, source_frame: int):
        """从一批检测里挑一个与历史位置一致的。

        策略（TRACK 态）：取**残差最小**（离历史位置最近）且在关联门限内的检测；
        没有就返回 None（本轮不做任何修正）。
        SEARCH 态没有历史位置可用，退化为"分数最高"。

        """
        if not detections:
            return None
        if self.state is TrackerState.SEARCH:
            return max(detections, key=lambda det: det.score)

        historical = self.history.get(source_frame)
        if historical is None:
            # 取不到同刻历史 ⇒ 关联无法用残差判据，只能退化成分数最高；
            # 此时残差融合会因缺少对账依据而自行弃权（不修正位置）。
            return max(detections, key=lambda det: det.score)

        if self.state is TrackerState.LOST:
            limit = self.config.lost_capture_dist
        elif self.config.fusion_mode == "replay":
            # 臂 3 的锚点门限更宽：Δ=8 帧的名义位移本身就有几十像素
            #（实测 p90 ≈ 46 px），用窄门限会把合法锚点也毙掉。
            limit = self.config.replay_max_dist
        else:
            limit = self.config.assoc_max_dist
        best = None
        best_distance = float("inf")
        for det in detections:
            distance = float(np.linalg.norm(det.xy - historical))
            if distance < best_distance:
                best_distance = distance
                best = det
        if best is None or best_distance > limit:
            return None
        return best

    # -- 初始化 -------------------------------------------------------------

    def _accumulate_init(self, source_frame: int, detection) -> bool:
        """累计连续的高置信度检测；满足 K 次且位置稳定 ⇒ 确认目标。

        规格要求"连续 K 帧"，所以除了"K 次 + 位置稳定"，还必须保证这 K 次
        是**连续到达**的：源帧号严格递增，且相邻两次的源帧间隔不超过
        ``init_max_frame_gap``。检测器停摆太久时，之前攒下的样本作废。
        """
        cfg = self.config
        if detection.score < cfg.init_min_score:
            self._init_samples.clear()
            return False

        if self._init_samples:
            gap = int(source_frame) - int(self._init_samples[-1].frame_index)
            if gap <= 0 or gap > cfg.init_max_frame_gap:
                # 不是连续到达（检测器停摆或乱序）⇒ 重新开始累计。
                self._init_samples.clear()

        self._init_samples.append(
            _DetectionSample(source_frame, np.asarray(detection.xy, dtype=np.float64), float(detection.score))
        )
        if len(self._init_samples) > cfg.init_consec:
            self._init_samples.pop(0)
        if len(self._init_samples) < cfg.init_consec:
            return False

        newest = self._init_samples[-1].xy
        for sample in self._init_samples:
            if float(np.linalg.norm(sample.xy - newest)) > cfg.init_max_disp:
                return False
        return True

    def _project_to_now(self, detection, source_frame: int, current_frame: int) -> np.ndarray:
        """把"Δ 帧前"的检测位置投影到当前时刻。

        速度用**最近几次检测**估计（而不是快跟踪器的历史——此刻还没有跟踪历史）。

        ⚠️ ``velocity_source == "none"`` 时**不投影**：文档 §三 的伪代码骨架在
        SEARCH 初始化分支写的是 ``self.curr_pos = det_xy``（直接落在检测框），
        §二.4 的散文没有提投影。真实序列实测：不投影的命中更高（见 README
        的速度源对照段）——把 8 帧前的检测位置外推 8 帧，误差会被放大。
        """
        cfg = self.config
        position = np.asarray(detection.xy, dtype=np.float64).copy()
        if not cfg.project_velocity or cfg.velocity_source == "none":
            return position
        if len(self._det_history) < 2:
            return position

        first = self._det_history[0]
        last = self._det_history[-1]
        span = last.frame_index - first.frame_index
        if span <= 0:
            return position
        velocity = (last.xy - first.xy) / float(span)
        return position + velocity * (current_frame - source_frame)

    def _detection_velocity(self) -> np.ndarray | None:
        """用最近几次关联上的检测估计速度（像素/帧）。

        这是"硬拉回到当前时刻"的速度兜底源：快跟踪核一旦在 LOST 里冻住，
        它的历史速度就是 0，而检测流仍在以真实速度报位置。
        """
        if len(self._det_history) < 2:
            return None
        first = self._det_history[0]
        last = self._det_history[-1]
        span = last.frame_index - first.frame_index
        if span <= 0:
            return None
        return (last.xy - first.xy) / float(span)

    def _capture(self, detection, source_frame: int) -> bool:
        """LOST 态：检测框是否落回捕获门限内（相对当前外推位置）。"""
        if self._position is None:
            return False
        distance = float(np.linalg.norm(np.asarray(detection.xy) - self._position))
        return distance <= self.config.lost_capture_dist

    def _reset_search(self) -> None:
        self._init_samples.clear()
        self._det_history.clear()
        self._position = None
        self._lost_count = 0
        self._weak_count = 0
        self._det_miss_count = 0
        self.history.clear()
