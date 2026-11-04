"""三态状态机的契约测试。

**时序约定**（与真实系统一致，务必按这个来写用例）：
检测器按 ``det_hz`` 抽样，结果滞后 ``latency_frames`` 帧才到达。因此在
第 ``t`` 帧到达的检测，其**源帧号**是 ``f = t − Δ``；而快跟踪器在第 ``f``
帧末尾把位置写进历史环形缓冲，所以第 ``t`` 帧做残差对账时
``history.get(f)`` 一定拿得到（Δ ≥ 1）。

用例里统一取：相机 25 Hz、检测 12.5 Hz（每 2 帧检一次）、Δ = 2。
于是"针对第 f 帧的检测"在第 f+2 帧到达，例如：

    f0: 到达 det(0)         -> SEARCH（记 1 次确认）
    f2: 到达 det(2)         -> SEARCH→TRACK
    f4: 到达 det(2)（Δ=2）  -> TRACK 态残差融合（history 里有第 2 帧）

注意第 4 帧这一行：**源帧号是 2，不是 4**。用 Δ=0 写用例会让
"同刻对账"的前提不成立，残差融合按设计必须弃权——那是对的行为，不是 bug。
"""

from __future__ import annotations

import unittest

import numpy as np

from manu.tracking.async_dual_rate.detector import Detection
from manu.tracking.async_dual_rate.tracker import (
    AsyncDualRateTracker,
    TrackerConfig,
    TrackerState,
)


def make_frame(center=(320.0, 256.0), amplitude=120.0, background=30.0, size=(512, 640)):
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]
    blob = amplitude * np.exp(
        -(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * 2.0**2))
    )
    return np.clip(background + blob, 0, 255).astype(np.uint8)


def detection(frame_index, xy, score=0.9):
    return Detection(frame_index=frame_index, xy=np.asarray(xy, dtype=np.float64), score=score)


class FsmContractTest(unittest.TestCase):
    def make_tracker(self, **overrides):
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

    def bootstrap_to_track(self, tracker, frame):
        """把跟踪器推到 TRACK 态：f0 记一次确认，f2 二次确认进入 TRACK。"""
        tracker.step(0, frame, [(0, [detection(0, (320.0, 256.0))])])
        out = tracker.step(2, frame, [(2, [detection(2, (320.0, 256.0))])])
        self.assertIs(out.state, TrackerState.TRACK)
        return out

    # -- SEARCH ---------------------------------------------------------

    def test_search_until_enough_confirmations(self):
        tracker = self.make_tracker()
        frame = make_frame()
        out = tracker.step(0, frame, [(0, [detection(0, (320.0, 256.0))])])
        self.assertIs(out.state, TrackerState.SEARCH)
        out = tracker.step(2, frame, [(2, [detection(2, (320.0, 256.0))])])
        self.assertIs(out.state, TrackerState.TRACK)
        self.assertIn("SEARCH->TRACK", out.events)

    def test_low_score_breaks_the_confirmation_streak(self):
        tracker = self.make_tracker(init_min_score=0.25)
        frame = make_frame()
        tracker.step(0, frame, [(0, [detection(0, (320.0, 256.0), score=0.9)])])
        tracker.step(2, frame, [(2, [detection(2, (320.0, 256.0), score=0.1)])])
        out = tracker.step(4, frame, [(4, [detection(4, (320.0, 256.0), score=0.9)])])
        self.assertIs(out.state, TrackerState.SEARCH, "低分检测应打断连续确认")

    def test_unstable_detections_do_not_initialise(self):
        """位置跳得太远 ⇒ 不算"稳定空域"，不初始化。"""
        tracker = self.make_tracker(init_max_disp=12.0)
        frame = make_frame()
        tracker.step(0, frame, [(0, [detection(0, (320.0, 256.0))])])
        out = tracker.step(2, frame, [(2, [detection(2, (400.0, 256.0))])])
        self.assertIs(out.state, TrackerState.SEARCH)

    # -- TRACK → LOST ---------------------------------------------------

    def test_weak_signal_drives_track_to_lost(self):
        """快跟踪核连续 ``track_to_lost_frames`` 帧失效 ⇒ TRACK→LOST。"""
        tracker = self.make_tracker(track_to_lost_frames=2, max_lost_frames=3)
        good = make_frame()
        blank = np.full((512, 640), 35, dtype=np.uint8)
        self.bootstrap_to_track(tracker, good)

        out = tracker.step(3, blank, [])
        self.assertIs(out.state, TrackerState.TRACK, "单帧失效应被容忍")
        out = tracker.step(4, blank, [])
        self.assertIs(out.state, TrackerState.LOST, "连续两帧失效 ⇒ LOST")
        self.assertIn("TRACK->LOST:energy", out.events)

    def test_lost_returns_to_search_after_budget(self):
        """LOST 态外推预算耗尽 ⇒ LOST→SEARCH。

        契约（对齐规格"允许外推维持 3~5 帧"）：LOST 态**对外输出位置**的帧数
        必须恰好等于 ``max_lost_frames``。旧实现用 ``_lost_count > budget``，
        实际会多输出 1 帧（预算 2 却输出 3 帧）——已修为 ``>=``。
        """
        tracker = self.make_tracker(track_to_lost_frames=2, max_lost_frames=2)
        good = make_frame()
        blank = np.full((512, 640), 35, dtype=np.uint8)
        self.bootstrap_to_track(tracker, good)

        states = {}
        outputs = {}
        for index in range(3, 14):
            out = tracker.step(index, blank, [])
            states[index] = (out.state, out.events)
            outputs[index] = out.xy
        lost_frame = next(i for i, (s, _) in states.items() if s is TrackerState.LOST)
        search_frame = next(i for i, (s, _) in states.items() if s is TrackerState.SEARCH)
        self.assertEqual(lost_frame, 4, "连续两帧失效 ⇒ 第 4 帧进入 LOST")
        self.assertEqual(search_frame, 6, "外推预算 2 帧 ⇒ 第 6 帧退回 SEARCH")
        self.assertIn("LOST->SEARCH", states[search_frame][1])
        lost_outputs = [
            index
            for index, (state, _) in states.items()
            if state is TrackerState.LOST and outputs[index] is not None
        ]
        self.assertEqual(
            len(lost_outputs),
            2,
            f"LOST 态对外输出的帧数必须恰好等于预算 2，实际 {lost_outputs}",
        )

    def test_lost_extrapolates_with_velocity(self):
        """LOST 态没有能量时按历史速度外推，而不是把位置冻死。

        帧号必须**连续**（0..6 逐帧推进）：历史速度是相邻两帧的位置差，
        跳帧时速度按设计置零（"断帧时速度未知"），外推自然也退化为 0。
        """
        tracker = self.make_tracker(track_to_lost_frames=1, max_lost_frames=8)

        def frame_at(x):
            return make_frame(center=(x, 256.0))

        tracker.step(0, frame_at(320.0), [(0, [detection(0, (320.0, 256.0))])])
        tracker.step(1, frame_at(325.0), [])
        tracker.step(2, frame_at(330.0), [(2, [detection(2, (330.0, 256.0))])])
        tracker.step(3, frame_at(335.0), [])
        tracker.step(4, frame_at(340.0), [])
        tracker.step(5, frame_at(345.0), [])
        before = tracker._position.copy()
        out = tracker.step(6, np.full((512, 640), 35, dtype=np.uint8), [])
        self.assertIs(out.state, TrackerState.LOST)
        self.assertGreater(float(out.xy[0] - before[0]), 0.0, "LOST 态应继续外推")

    # -- LOST → TRACK ----------------------------------------------------

    def test_lost_recaptures_when_detector_agrees(self):
        tracker = self.make_tracker(track_to_lost_frames=2, max_lost_frames=6)
        good = make_frame()
        blank = np.full((512, 640), 35, dtype=np.uint8)
        self.bootstrap_to_track(tracker, good)
        for index in range(3, 6):
            tracker.step(index, blank, [])
        self.assertIs(tracker.state, TrackerState.LOST)
        # 检测器在目标附近再次命中 ⇒ 重捕
        out = tracker.step(6, good, [(6, [detection(6, (320.0, 256.0))])])
        self.assertIs(out.state, TrackerState.TRACK)
        self.assertIn("LOST->TRACK", out.events)

    # -- 残差融合 --------------------------------------------------------

    def test_fusion_is_applied_in_track_state(self):
        """TRACK 态下迟到的检测必须走残差融合（并留下记录）。

        第 4 帧到达的是**源帧 2** 的检测（Δ = 2），历史缓冲里有第 2 帧的
        位置，所以残差 = z_2 − x_2 是有意义的。
        """
        tracker = self.make_tracker(gating_dist=12.0, alpha=0.5, latency_frames=2)
        frame = make_frame()
        self.bootstrap_to_track(tracker, frame)
        # 检测框比第 2 帧的历史位置偏 6 px（残差 6 < 12）⇒ 走平滑
        out = tracker.step(4, frame, [(2, [detection(2, (326.0, 256.0))])])
        self.assertIsNotNone(out.fusion)
        self.assertEqual(out.fusion.kind, "smooth")
        self.assertTrue(out.fusion.applied)
        self.assertIn("fusion:smooth", out.events)
        # 修正量必须严格等于 α·残差
        expected = out.fusion.position_before + 0.5 * out.fusion.residual
        np.testing.assert_allclose(out.fusion.position_after, expected, rtol=0, atol=1e-12)
        # 残差应接近 6 px（检测偏 6，历史位置 ≈ blob 中心）
        self.assertAlmostEqual(float(out.fusion.residual_norm), 6.0, delta=0.5)

    def test_fusion_abstains_without_same_instant_history(self):
        """历史对账不可用（Δ=0，源帧就是当前帧）⇒ 必须弃权，不做修正。"""
        tracker = self.make_tracker(gating_dist=12.0, alpha=0.5)
        frame = make_frame()
        self.bootstrap_to_track(tracker, frame)
        position_before = tracker._position.copy()
        out = tracker.step(4, frame, [(4, [detection(4, (400.0, 256.0))])])
        self.assertIsNotNone(out.fusion)
        self.assertEqual(out.fusion.kind, "none")
        self.assertFalse(out.fusion.applied)
        self.assertIn("fusion:abstain", out.events)
        np.testing.assert_allclose(out.fusion.position_after, position_before, rtol=0, atol=1e-12)

    def test_large_residual_triggers_hard_reinit(self):
        """残差超过门限 ⇒ 走速度投影的硬拉回。"""
        tracker = self.make_tracker(gating_dist=12.0, alpha=0.5)
        frame = make_frame()
        self.bootstrap_to_track(tracker, frame)
        # 残差 30 px：超过融合门限 12，但仍在同一目标关联门限 64 以内
        out = tracker.step(4, frame, [(2, [detection(2, (350.0, 256.0))])])
        self.assertEqual(out.fusion.kind, "hard_reinit")
        self.assertIn("fusion:hard_reinit", out.events)

    # -- 输出 ------------------------------------------------------------

    def test_output_position_is_absolute_pixels(self):
        tracker = self.make_tracker()
        frame = make_frame(center=(400.0, 300.0))
        tracker.step(0, frame, [(0, [detection(0, (400.0, 300.0))])])
        out = tracker.step(2, frame, [(2, [detection(2, (400.0, 300.0))])])
        self.assertIsNotNone(out.xy)
        self.assertAlmostEqual(out.xy[0], 400.0, delta=0.5)
        self.assertAlmostEqual(out.xy[1], 300.0, delta=0.5)

    def test_search_state_has_no_position(self):
        tracker = self.make_tracker()
        frame = make_frame()
        out = tracker.step(0, frame, [])
        self.assertIsNone(out.xy)
        self.assertFalse(out.valid)


if __name__ == "__main__":
    unittest.main()
