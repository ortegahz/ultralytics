"""臂 3「退回重推演」的契约测试。

这套用例锁的是**机制**，不是数字：

* 重推演的结果必须只依赖"锚点 + 锚点之后的图像"，**不能**依赖锚点之前的历史
  —— 这正是"退回"的定义；
* 臂 2「纯前向外推」必须**绝不**被迟到检测改动位置（受控 A/B 的零差异闸门）；
* 退回重推演必须能把**被杂波抓走**的目标重新拉回来，而纯前向不能。

时序约定与 ``test_fsm`` 一致：相机 25 Hz、检测 12.5 Hz（每 2 帧检一次）、
Δ = 2。"针对第 f 帧的检测"在第 f+Δ 帧到达。
"""

from __future__ import annotations

import unittest

import numpy as np

from manu.tracking.async_dual_rate.arms import ArmSpec, SyntheticFrameSource, run_lockstep
from manu.tracking.async_dual_rate.detector import Detection
from manu.tracking.async_dual_rate.fast_tracker import TrackerResult
from manu.tracking.async_dual_rate.replay import FrameRingBuffer, replay_forward
from manu.tracking.async_dual_rate.tracker import (
    AsyncDualRateTracker,
    TrackerConfig,
    TrackerState,
)


# ==============================================================================
# 工具
# ==============================================================================


class _TruthDetector:
    """报**真值**位置的检测器（合成臂专用）。

    注意不能用 :class:`~.arms.OracleDetector`：它读的是跟踪器自己的历史，
    而臂 3 会把历史回写成重推结果 —— 那样锚点就变成了"重推后的漂移位置"，
    测不出"退回"到底有没有发生。
    """

    def __init__(self, source, score: float = 0.9) -> None:
        self.source = source
        self.score = float(score)
        self.frames = getattr(source, "frames", 0)

    def detections(self, frame_index: int) -> list[Detection]:
        position = self.source.position(int(frame_index))
        return [
            Detection(
                frame_index=int(frame_index),
                xy=np.asarray(position, dtype=np.float64),
                score=self.score,
            )
        ]


class _FrozenFastKernel:
    """"冻住"的快核：永远返回上一次的位置，什么都不纠正。

    真实系统里快核只有 32×32 的视野——一旦位置误差超过半窗（约 16 px），
    它就**再也拉不回来**，只能等一个外部锚点。这正是臂 3 存在的理由：
    退回重推演把状态**整体替换**成锚点，而不是指望快核自己爬回来。
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def step(self, frame, center_xy) -> TrackerResult:
        self.calls += 1
        return TrackerResult(
            xy=np.asarray(center_xy, dtype=np.float64).copy(),
            valid=True,
            energy=0.0,
            snr=0.0,
            peak_xy=None,
        )


def make_tracker(**overrides) -> AsyncDualRateTracker:
    config = TrackerConfig(
        latency_frames=2,
        init_consec=2,
        track_to_lost_frames=2,
        max_lost_frames=3,
        history_capacity=32,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return AsyncDualRateTracker(config)


def truth_detection(source, frame_index: int) -> Detection:
    return Detection(
        frame_index=int(frame_index),
        xy=np.asarray(source.position(int(frame_index)), dtype=np.float64),
        score=0.9,
    )


def bootstrap(tracker, source, frames=(0, 2)):
    """用两次真值检测把跟踪器推进 TRACK 态。"""
    for frame_index in frames:
        tracker.step(frame_index, source.read(frame_index), [(frame_index, [truth_detection(source, frame_index)])])


# ==============================================================================
# A. 帧环形缓冲
# ==============================================================================


class FrameRingBufferTest(unittest.TestCase):
    def test_stores_and_reads_by_absolute_frame_index(self):
        cache = FrameRingBuffer(capacity=4)
        for index in range(3):
            cache.push(index, np.full((2, 2), index, dtype=np.uint8))
        for index in range(3):
            self.assertIsNotNone(cache.get(index))
            self.assertEqual(int(cache.get(index)[0, 0]), index)

    def test_missing_frame_returns_none(self):
        cache = FrameRingBuffer(capacity=4)
        cache.push(0, np.zeros((2, 2), dtype=np.uint8))
        self.assertIsNone(cache.get(1))
        self.assertFalse(cache.has(1))

    def test_oldest_frame_is_overwritten_when_capacity_exceeded(self):
        cache = FrameRingBuffer(capacity=3)
        for index in range(5):
            cache.push(index, np.full((2, 2), index, dtype=np.uint8))
        self.assertIsNone(cache.get(0))
        self.assertIsNone(cache.get(1))
        self.assertIsNotNone(cache.get(4))

    def test_absent_frame_number_cannot_alias_a_present_one(self):
        """绝对帧号索引：第 0 帧被覆盖后，同槽位上的第 3 帧不能读成旧的第 0 帧。"""
        cache = FrameRingBuffer(capacity=3)
        cache.push(0, np.full((2, 2), 7, dtype=np.uint8))
        cache.push(1, np.full((2, 2), 8, dtype=np.uint8))
        cache.push(2, np.full((2, 2), 9, dtype=np.uint8))
        cache.push(3, np.full((2, 2), 10, dtype=np.uint8))
        self.assertIsNone(cache.get(0))
        self.assertEqual(int(cache.get(3)[0, 0]), 10)


# ==============================================================================
# B. 重推演内核
# ==============================================================================


class ReplayForwardTest(unittest.TestCase):
    def test_replay_reproduces_the_true_trajectory_from_the_anchor(self):
        source = SyntheticFrameSource(frames=40, velocity=(0.6, 0.2))
        tracker = make_tracker()
        cache = FrameRingBuffer(capacity=16)
        for index in range(20):
            cache.push(index, source.read(index))

        anchor_frame = 10
        target_frame = 18
        anchor = np.asarray(source.position(anchor_frame), dtype=np.float64)
        result = replay_forward(tracker.fast, cache, anchor, anchor_frame, target_frame)

        self.assertTrue(result.applied, result.reason)
        self.assertEqual(result.steps, target_frame - anchor_frame - 1)
        for frame_index, position in result.positions.items():
            truth = np.asarray(source.position(frame_index), dtype=np.float64)
            self.assertLess(
                float(np.linalg.norm(position - truth)),
                0.5,
                f"frame {frame_index}: 重推位置 {position} 偏离真值 {truth}",
            )

    def test_missing_frame_abstains(self):
        tracker = make_tracker()
        cache = FrameRingBuffer(capacity=8)
        cache.push(0, np.zeros((8, 8), dtype=np.uint8))
        result = replay_forward(tracker.fast, cache, np.array([10.0, 10.0]), 0, 4)
        self.assertFalse(result.applied)
        self.assertTrue(result.reason.startswith("missing_frame"))

    def test_lost_target_abstains_when_require_valid(self):
        tracker = make_tracker()
        cache = FrameRingBuffer(capacity=8)
        for index in range(6):
            cache.push(index, np.zeros((64, 64), dtype=np.uint8))
        result = replay_forward(
            tracker.fast, cache, np.array([5.0, 5.0]), 0, 4, require_valid=True
        )
        self.assertFalse(result.applied)
        self.assertTrue(result.reason.startswith("lost_target"))

    def test_empty_span_abstains(self):
        tracker = make_tracker()
        cache = FrameRingBuffer(capacity=8)
        result = replay_forward(tracker.fast, cache, np.array([10.0, 10.0]), 5, 5)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "empty_span")


# ==============================================================================
# C. 状态机接入
# ==============================================================================


class TrackerReplayModeTest(unittest.TestCase):
    def test_replay_output_equals_hand_computed_replay(self):
        """状态机输出必须与"手工重算"逐位一致（算术契约）。"""
        source = SyntheticFrameSource(frames=40, velocity=(0.6, 0.2))
        tracker = make_tracker(fusion_mode="replay")
        anchor_frame = 12
        arrival_frame = anchor_frame + tracker.config.latency_frames

        bootstrap(tracker, source)
        for index in range(3, arrival_frame):
            tracker.step(index, source.read(index), [])

        out = tracker.step(
            arrival_frame,
            source.read(arrival_frame),
            [(anchor_frame, [truth_detection(source, anchor_frame)])],
        )
        self.assertIsNotNone(out.replay)
        self.assertTrue(out.replay.applied, out.replay.reason)
        self.assertEqual(out.replay.anchor_frame, anchor_frame)

        cache = FrameRingBuffer(capacity=16)
        for index in range(anchor_frame, arrival_frame + 1):
            cache.push(index, source.read(index))
        manual = replay_forward(
            tracker.fast,
            cache,
            np.asarray(source.position(anchor_frame), dtype=np.float64),
            anchor_frame,
            arrival_frame,
        )
        self.assertTrue(manual.applied)
        expected = tracker.fast.step(
            source.read(arrival_frame), manual.positions[arrival_frame - 1]
        )
        self.assertLess(float(np.linalg.norm(out.xy - expected.xy)), 1e-12)

    def test_replay_rewrites_history(self):
        """重推结果必须回写历史，否则后续"同刻对账"还会拿旧轨迹做基准。"""
        source = SyntheticFrameSource(frames=40, velocity=(0.6, 0.2))
        tracker = make_tracker(fusion_mode="replay")
        anchor_frame = 12
        arrival_frame = anchor_frame + tracker.config.latency_frames

        bootstrap(tracker, source)
        for index in range(3, arrival_frame):
            tracker.step(index, source.read(index), [])
        tracker.step(
            arrival_frame,
            source.read(arrival_frame),
            [(anchor_frame, [truth_detection(source, anchor_frame)])],
        )

        truth = np.asarray(source.position(anchor_frame), dtype=np.float64)
        recorded = tracker.history.get(anchor_frame)
        self.assertIsNotNone(recorded)
        self.assertLess(float(np.linalg.norm(recorded - truth)), 0.5)

    def test_anchor_without_history_abstains(self):
        """拿不到历史位置就没法验证锚点 ⇒ 弃权，不能拿不可验证的框去重推。"""
        source = SyntheticFrameSource(frames=40, velocity=(0.6, 0.2))
        tracker = make_tracker(fusion_mode="replay")
        bootstrap(tracker, source)
        tracker.history.clear()
        out = tracker.step(
            10,
            source.read(10),
            [(8, [truth_detection(source, 8)])],
        )
        self.assertIsNotNone(out.replay)
        self.assertFalse(out.replay.applied)
        self.assertEqual(out.replay.reason, "no_history_for_anchor")

    def test_far_detection_is_not_associated_in_replay_mode(self):
        """离历史位置太远的检测在 replay 模式下不被关联 ⇒ 位置不动。

        受控 A/B：两个完全相同的跟踪器，唯一差别是其中一个收到一个
        故意偏到 (10, 10) 的旧检测。两者输出必须逐位相同。
        """
        source = SyntheticFrameSource(frames=40, velocity=(0.6, 0.2))

        def build():
            tracker = make_tracker(fusion_mode="replay", replay_max_dist=8.0)
            bootstrap(tracker, source)
            for index in range(3, 11):
                tracker.step(index, source.read(index), [])
            return tracker

        without = build()
        with_far = build()

        far = Detection(frame_index=9, xy=np.array([10.0, 10.0]), score=0.9)
        out_a = without.step(11, source.read(11), [])
        out_b = with_far.step(11, source.read(11), [(9, [far])])

        self.assertIsNone(out_b.replay)
        self.assertLess(
            float(np.linalg.norm(out_a.xy - out_b.xy)),
            1e-12,
            "远检测不该被关联，更不该改变位置",
        )

    def test_forward_only_mode_never_moves_position(self):
        """臂 2：迟到检测只做对比，位置一个字节都不许变。

        受控 A/B：跑两个**完全相同**的跟踪器，唯一差别是其中一个收到了
        一个故意偏 30 px 的旧检测。两者输出必须逐位相同（max|Δ| = 0）。
        """
        source = SyntheticFrameSource(frames=40, velocity=(0.6, 0.2))

        def build():
            tracker = make_tracker(fusion_mode="none")
            bootstrap(tracker, source)
            for index in range(3, 13):
                tracker.step(index, source.read(index), [])
            return tracker

        without = build()
        with_detection = build()

        drifted = np.asarray(source.position(11), dtype=np.float64) + np.array([30.0, 0.0])
        out_a = without.step(13, source.read(13), [])
        out_b = with_detection.step(
            13, source.read(13), [(11, [Detection(11, drifted, 0.9)])]
        )
        self.assertIsNotNone(out_a.xy)
        self.assertIsNotNone(out_b.xy)
        self.assertLess(
            float(np.linalg.norm(out_a.xy - out_b.xy)),
            1e-12,
            "臂 2 的位置被迟到检测改动了 —— 违反'只对比不修正'的契约",
        )
        self.assertIn("fusion:compare_only", out_b.events)


# ==============================================================================
# D. 「退回」语义：输出不能依赖锚点之前的历史
# ==============================================================================


class RollbackSemanticsTest(unittest.TestCase):
    """臂 3 的定义性性质：**输出只由锚点 + 锚点之后的图像决定**。

    构造两个只在"锚点之前的历史"上不同的跟踪器（历史被人为污染），
    锚点到达后两者的输出必须**逐位相同**。

    同样的污染在 ``alpha``（残差融合）模式下会让输出不同 —— 那正好反证
    这条测试确实能区分"退回"与"不回溯"。
    """

    def _build(self, mode, source, anchor_frame):
        tracker = make_tracker(fusion_mode=mode, replay_max_dist=200.0)
        # 用"冻住"的快核：否则快核会自校正到真值，把残差融合/重推演的效果
        # 全部抹平（这本身也是一条重要事实——融合只在快核已经跑偏时才有意义）。
        tracker.fast = _FrozenFastKernel(tracker.fast)
        bootstrap(tracker, source)
        # 必须**逐帧**推进到"锚点帧 + Δ − 1"：锚点帧 f 的历史位置既是
        # "锚点可验证"的前提，也是 alpha 模式残差对账的基准，两条用例都要用。
        stop = anchor_frame + tracker.config.latency_frames
        for index in range(3, stop):
            tracker.step(index, source.read(index), [])
        return tracker

    def _pollute_history(self, tracker, offset=(7.0, -5.0)):
        for frame_index in tracker.history.frames().tolist():
            position = tracker.history.get(int(frame_index))
            if position is not None:
                tracker.history.overwrite(int(frame_index), position + np.asarray(offset))

    def _run_to_anchor(self, tracker, source, anchor_frame):
        arrival_frame = anchor_frame + tracker.config.latency_frames
        out = tracker.step(
            arrival_frame,
            source.read(arrival_frame),
            [(anchor_frame, [truth_detection(source, anchor_frame)])],
        )
        return out

    def test_replay_output_is_independent_of_earlier_history(self):
        source = SyntheticFrameSource(frames=60, velocity=(0.6, 0.2))
        anchor_frame = 20

        clean = self._build("replay", source, anchor_frame)
        dirty = self._build("replay", source, anchor_frame)
        self._pollute_history(dirty)

        out_clean = self._run_to_anchor(clean, source, anchor_frame)
        out_dirty = self._run_to_anchor(dirty, source, anchor_frame)

        self.assertTrue(out_clean.replay.applied, out_clean.replay.reason)
        self.assertTrue(out_dirty.replay.applied, out_dirty.replay.reason)
        self.assertLess(float(np.linalg.norm(out_clean.xy - out_dirty.xy)), 1e-12)

    def test_alpha_mode_does_roll_back(self):
        """对照组：alpha（残差融合）**不回溯**，历史被污染 ⇒ 输出不同。"""
        source = SyntheticFrameSource(frames=60, velocity=(0.6, 0.2))
        anchor_frame = 20

        clean = self._build("alpha", source, anchor_frame)
        dirty = self._build("alpha", source, anchor_frame)
        self._pollute_history(dirty)

        out_clean = self._run_to_anchor(clean, source, anchor_frame)
        out_dirty = self._run_to_anchor(dirty, source, anchor_frame)

        self.assertGreater(
            float(np.linalg.norm(out_clean.xy - out_dirty.xy)),
            1e-6,
            "alpha 模式对历史敏感是预期的；若此处相同说明残差根本没生效",
        )


# ==============================================================================
# E. 三臂受控 A/B（合成）
# ==============================================================================


class ThreeArmSyntheticTest(unittest.TestCase):
    """臂 2 vs 臂 3 的受控对照（合成）。

    场景用"冻住的快核"：真实系统里快核只有 32×32 视野，一旦误差超过半窗
    就再也拉不回来（这在龙泉山实测序列上是**主要**的脱靶机制）。这里把它
    显式建模，好处是判据可以写成解析式：

    * 臂 2（纯前向）：位置永远停在引导点 ⇒ 误差随目标运动**单调发散**；
    * 臂 3（退回重推演）：每 Δ 帧被锚点整体替换 ⇒ 误差被压回"Δ 帧的位移量"。
    """

    def _run(self, source, frames=240, frozen=True):
        stream = _TruthDetector(source)
        specs = [
            ArmSpec(
                name="forward_only",
                kind="tracker",
                det_hz=12.5,
                latency_frames=8,
                fusion_mode="none",
                detection_reinit=False,
            ),
            ArmSpec(
                name="rollback_replay",
                kind="tracker",
                det_hz=12.5,
                latency_frames=8,
                fusion_mode="replay",
            ),
        ]
        base = TrackerConfig(
            latency_frames=8,
            init_consec=2,
            track_to_lost_frames=2,
            max_lost_frames=3,
            history_capacity=64,
            # "安静锁在杂波"是实测序列里最常见的脱靶形态：跟踪器**还活着**
            # （状态机还在 TRACK），但位置早就错到 ROI 之外、永远拉不回来。
            # 这里把"检测器连续漏检 ⇒ 判 LOST"的预算放到极大，
            # 好让两个臂都能一直跑完，差异只来自"锚点有没有把状态复位"。
            det_miss_budget=10**9,
        )

        def factory(_name, tracker):
            if frozen:
                tracker.fast = _FrozenFastKernel(tracker.fast)
            return stream

        return run_lockstep(
            source,
            stream,
            specs,
            base_config=base,
            total_frames=frames,
            detector_factory=factory,
        )

    def test_replay_resets_a_corrupted_state_to_the_anchor(self):
        """快核卡住不动时：臂 3 靠锚点整体复位，臂 2 只能一直错下去。"""
        source = SyntheticFrameSource(frames=240, velocity=(0.6, 0.2))
        outcomes = self._run(source)

        probe = 140
        truth = np.asarray(source.position(probe), dtype=np.float64)

        replay_position = outcomes["rollback_replay"].positions.get(probe)
        self.assertIsNotNone(replay_position, "臂 3 应当有输出")
        replay_error = float(np.linalg.norm(replay_position - truth))

        forward_position = outcomes["forward_only"].positions.get(probe)
        self.assertIsNotNone(forward_position, "臂 2 也应当有输出（只是位置错）")
        forward_error = float(np.linalg.norm(forward_position - truth))

        self.assertLess(
            replay_error,
            forward_error,
            f"退回重推演没有复位: forward {forward_error:.2f} px, replay {replay_error:.2f} px",
        )
        # 臂 3 的误差应当被"Δ 帧内的目标位移"量级约束住（0.6 px/帧 × 8 ≈ 5 px）
        self.assertLess(replay_error, 12.0)

    def test_forward_only_error_grows_without_bound(self):
        """纯前向的误差必须随时间**单调发散**——这正是要修的病。"""
        source = SyntheticFrameSource(frames=240, velocity=(0.6, 0.2))
        outcomes = self._run(source)
        forward = outcomes["forward_only"].positions

        def error_at(frame_index):
            position = forward.get(frame_index)
            if position is None:
                return None
            truth = np.asarray(source.position(frame_index), dtype=np.float64)
            return float(np.linalg.norm(position - truth))

        early = error_at(30)
        late = error_at(200)
        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        self.assertGreater(late, early + 50.0)

    def test_the_two_arms_are_not_the_same_thing(self):
        """两个臂的输出必须不同——否则 A/B 归因失效。"""
        source = SyntheticFrameSource(frames=240, velocity=(0.6, 0.2))
        outcomes = self._run(source)
        forward = outcomes["forward_only"].positions
        replay = outcomes["rollback_replay"].positions
        shared = sorted(set(forward) & set(replay))
        self.assertGreater(len(shared), 50)
        differences = [
            float(np.linalg.norm(forward[index] - replay[index])) for index in shared
        ]
        self.assertGreater(max(differences), 1.0)

    def test_arms_agree_when_nothing_goes_wrong(self):
        """干净场景（快核无偏、无杂波）下两个臂应当基本一致——空对照。"""
        source = SyntheticFrameSource(frames=240, velocity=(0.6, 0.2))
        outcomes = self._run(source, frozen=False)
        forward = outcomes["forward_only"].positions
        replay = outcomes["rollback_replay"].positions
        shared = sorted(set(forward) & set(replay))
        self.assertGreater(len(shared), 50)
        differences = [
            float(np.linalg.norm(forward[index] - replay[index])) for index in shared
        ]
        self.assertLess(max(differences), 1.0)


if __name__ == "__main__":
    unittest.main()
