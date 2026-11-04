"""快速前向推演外挂（Look-ahead Plugin）的契约测试。

这套用例锁的是**机制**，不是数据集上的数字：

* ``off`` 臂必须与输入**逐位**一致（外挂不得改动任何历史估计——硬验收项）；
* 匀速合成轨迹上 kinematic 臂应当**解析命中**真值（误差 < 1e-9）；
* 三档速度估计（kf / diff1 / diffN）在同一条轨迹上必须自洽；
* replay 臂要能跟住合成亮斑，目标丢失时按 ``replay_fallback`` 兜底；
* ``servo_metrics`` 在人工 (report, truth) 上给出可手算的指标；
* **能变红**的守卫：``frame_source`` 读到"未来帧"就抛异常，证明 replay 不读未来帧；
* rolling 档位：一条链跨物理帧持续推进，锁住跟踪精度、帧间位移的平滑性、
  离群步被拒、观测失效后的 hold/重捕获，以及 ``stats`` 的口径。

时序约定与契约一致：管线时间轴 = record index；滞后 Δ 以 record 为单位；
物理帧 ``t`` 的上报其源帧是 ``f = t − Δ``。
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pytest

from manu.tracking.lookahead.extrapolator import (
    FrameSource,
    LookaheadConfig,
    _apply_output_slew_limit,
    build_report_stream,
    estimate_velocity,
    servo_metrics,
)


# ==============================================================================
# 工具：合成轨迹 / 合成亮斑帧 / 帧源
# ==============================================================================


def _uniform_dets(frames: int, start=(100.0, 200.0), velocity=(2.5, -1.5), track_id: int = 1):
    """匀速合成轨迹的 D(f)：位置严格落在线上，速度恒定（解析可算）。"""
    dets = []
    for index in range(frames):
        x = start[0] + velocity[0] * index
        y = start[1] + velocity[1] * index
        dets.append([{"pos": (x, y), "score": 0.9, "track_id": track_id, "infilled": False}])
    return dets


def _uniform_position(index: int, start=(40.0, 40.0), velocity=(2.0, 1.0)):
    return (start[0] + velocity[0] * index, start[1] + velocity[1] * index)


def _turning_position(index: int, start=(40.0, 40.0)):
    """先快后慢的折线轨迹。

    **为什么不用匀速**：匀速下"重推成功"与"运动学兜底"会落在同一个点上，
    兜底分支就测不出区别了；折线轨迹让两者差出 4 px，断言才有鉴别力。
    """
    x = 3.0 * index if index <= 6 else 18.0 + (index - 6)
    return (start[0] + x, start[1])


def _render_blob(cx, cy, seed, height=128, width=160, amplitude=120.0, background=30.0, sigma=2.0):
    """渲染一帧高斯亮斑（红外弱小目标的理想化替身）。"""
    yy, xx = np.mgrid[0:height, 0:width]
    blob = amplitude * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2)))
    image = background + blob
    rng = np.random.default_rng(seed)
    image = image + rng.uniform(0.0, 6.0, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _render_blank(seed, height=128, width=160, background=30.0):
    """没有目标的帧（只剩底噪）——replay 经过这里必须判定丢目标。"""
    rng = np.random.default_rng(seed)
    image = background + rng.uniform(0.0, 6.0, size=(height, width))
    return np.clip(image, 0, 255).astype(np.uint8)


class _DictFrameSource:
    """最简帧源：按 record index 取灰度整帧，取不到返回 None。"""

    def __init__(self, images: dict[int, np.ndarray]) -> None:
        self.images = images

    def get(self, frame_index: int) -> np.ndarray | None:
        return self.images.get(int(frame_index))


class _GuardedFrameSource:
    """读到超过 ``limit`` 的帧就抛异常——用来证明 replay 不会读未来帧。"""

    def __init__(self, images: dict[int, np.ndarray], limit: int) -> None:
        self.images = images
        self.limit = int(limit)
        self.requested: list[int] = []

    def get(self, frame_index: int) -> np.ndarray | None:
        index = int(frame_index)
        self.requested.append(index)
        if index > self.limit:
            raise AssertionError(f"replay 读到了未来帧 {index} > {self.limit}")
        return self.images.get(index)


# ==============================================================================
# 1. off 臂：与输入逐位一致
# ==============================================================================


class LookaheadOffModeTest(unittest.TestCase):
    """契约 §4 验收项 1：``mode="off"`` 与输入逐位一致（坐标 / score / track_id 全等）。"""

    def test_off_reproduces_the_input_stream_bit_by_bit(self):
        frames = 12
        dets = _uniform_dets(frames)
        report, stats = build_report_stream(dets, frames, LookaheadConfig(mode="off"))
        self.assertEqual(len(report), frames)
        for index in range(frames):
            self.assertEqual(len(report[index]), len(dets[index]))
            for got, want in zip(report[index], dets[index]):
                self.assertEqual(got["pos"], want["pos"])
                self.assertEqual(got["score"], want["score"])
                self.assertEqual(got["track_id"], want["track_id"])
                self.assertEqual(got["infilled"], want["infilled"])
        self.assertEqual(stats["mode"], "off")

    def test_off_does_not_introduce_the_lag(self):
        """``off`` 不加滞后：``report[t]`` 就是 ``D(t)``，不是 ``D(t−Δ)``。"""
        frames = 10
        dets = _uniform_dets(frames)
        report, _ = build_report_stream(dets, frames, LookaheadConfig(mode="off", delta=4))
        for index in range(frames):
            self.assertEqual(report[index][0]["pos"], dets[index][0]["pos"])


class LookaheadStaleModeTest(unittest.TestCase):
    """stale 臂：老逻辑（硬吃 Δ 帧滞后），尺子 2 的基线。"""

    def test_stale_copies_the_source_frame_verbatim(self):
        frames, delta = 10, 4
        dets = _uniform_dets(frames)
        report, stats = build_report_stream(dets, frames, LookaheadConfig(mode="stale", delta=delta))
        for index in range(delta):
            self.assertEqual(report[index], [])
        for index in range(delta, frames):
            self.assertEqual(report[index][0]["pos"], dets[index - delta][0]["pos"])
            self.assertEqual(report[index][0]["track_id"], dets[index - delta][0]["track_id"])
            self.assertEqual(report[index][0]["score"], dets[index - delta][0]["score"])
        self.assertEqual(stats["no_source_frames"], delta)


# ==============================================================================
# 2. kinematic 臂：匀速轨迹上解析命中真值
# ==============================================================================


class LookaheadKinematicTest(unittest.TestCase):
    """契约 §4 验收项 2：匀速合成轨迹上 kinematic 臂外推后位置落在真值上（误差 < 1e-9）。"""

    def test_kinematic_lands_on_the_truth(self):
        frames, delta = 20, 4
        velocity = (2.5, -1.5)
        dets = _uniform_dets(frames, velocity=velocity)
        cfg = LookaheadConfig(delta=delta, mode="kinematic", velocity_source="diff1")
        report, stats = build_report_stream(dets, frames, cfg)

        for index in range(delta):
            # t < Δ 没有源帧 ⇒ 该帧不输出
            self.assertEqual(report[index], [])
        for index in range(delta + 1, frames):
            truth = np.asarray((100.0 + velocity[0] * index, 200.0 + velocity[1] * index))
            got = np.asarray(report[index][0]["pos"], dtype=np.float64)
            self.assertLess(float(np.linalg.norm(got - truth)), 1e-9)
        # 源帧 f=0 没有前一帧 ⇒ 速度缺失（按 0 外推），只有这一帧
        self.assertEqual(stats["missing_velocity"], 1)
        self.assertEqual(stats["mode"], "kinematic")

    def test_stationary_target_is_not_counted_as_missing_velocity(self):
        """静止目标（KF 速度 = 0）是**合法速度**，不该被算成"缺速度"。"""
        frames, delta = 8, 3
        dets = _uniform_dets(frames, velocity=(0.0, 0.0))
        kf_states = {}
        for index in range(frames):
            position = np.asarray([100.0, 200.0], dtype=np.float64)
            kf_states[index] = {1: (position, np.zeros(2, dtype=np.float64))}
        cfg = LookaheadConfig(delta=delta, mode="kinematic", velocity_source="kf")
        report, stats = build_report_stream(dets, frames, cfg, kf_states=kf_states)
        self.assertEqual(stats["missing_velocity"], 0)
        for index in range(delta, frames):
            np.testing.assert_allclose(report[index][0]["pos"], (100.0, 200.0), atol=1e-12)


# ==============================================================================
# 3. 三档速度估计
# ==============================================================================


class EstimateVelocityTest(unittest.TestCase):
    """契约 §4 验收项 3：kf / diff1 / diffN 在匀速合成轨迹上给出一致速度。"""

    def test_three_sources_agree_on_uniform_motion(self):
        frames = 20
        velocity = np.asarray([2.5, -1.5], dtype=np.float64)
        dets = _uniform_dets(frames)
        kf_states = {}
        for index in range(frames):
            position = np.asarray([100.0 + 2.5 * index, 200.0 - 1.5 * index], dtype=np.float64)
            kf_states[index] = {1: (position, velocity.copy())}

        frame = 10
        v_kf = estimate_velocity(dets, frame, 1, source="kf", kf_states=kf_states)
        v_diff1 = estimate_velocity(dets, frame, 1, source="diff1")
        v_diffn = estimate_velocity(dets, frame, 1, source="diffN", window=3)

        for value in (v_kf, v_diff1, v_diffn):
            self.assertEqual(value.dtype, np.float64)
            self.assertEqual(value.shape, (2,))
            np.testing.assert_allclose(value, velocity, atol=1e-9)

    def test_missing_track_or_history_gives_zero_velocity(self):
        dets = _uniform_dets(5)
        zeros = np.zeros(2)
        # 第一帧没有上一帧
        np.testing.assert_array_equal(estimate_velocity(dets, 0, 1, source="diff1"), zeros)
        # 未知 track
        np.testing.assert_array_equal(estimate_velocity(dets, 3, 99, source="diff1"), zeros)
        # 没有 KF 状态
        np.testing.assert_array_equal(estimate_velocity(dets, 3, 1, source="kf"), zeros)
        # diffN 只凑得出一个点
        np.testing.assert_array_equal(estimate_velocity(dets, 0, 1, source="diffN", window=3), zeros)


# ==============================================================================
# 4. replay 臂：跟住亮斑 / 丢目标时兜底
# ==============================================================================


class LookaheadReplayTest(unittest.TestCase):
    """契约 §4 验收项 4：合成亮斑序列上 replay 能跟住；丢目标时按 fallback 兜底。"""

    def test_replay_follows_the_blob(self):
        frames, delta = 12, 3
        images = {}
        dets = []
        for index in range(frames):
            cx, cy = _uniform_position(index)
            images[index] = _render_blob(cx, cy, seed=1000 + index)
            dets.append([{"pos": (cx, cy), "score": 0.9, "track_id": 1, "infilled": False}])

        cfg = LookaheadConfig(delta=delta, mode="replay", velocity_source="diff1")
        report, stats = build_report_stream(
            dets, frames, cfg, frame_source=_DictFrameSource(images)
        )
        self.assertGreater(stats["replay_steps"], 0)
        for index in range(delta, frames):
            self.assertEqual(len(report[index]), 1)
            truth = np.asarray(_uniform_position(index))
            got = np.asarray(report[index][0]["pos"], dtype=np.float64)
            self.assertLess(float(np.linalg.norm(got - truth)), 1.5)
            # 重推必须真的把位置往前推了（不能只是搬运源帧坐标）
            stale = np.asarray(dets[index - delta][0]["pos"], dtype=np.float64)
            self.assertGreater(float(np.linalg.norm(got - stale)), 3.0)

    def _sequence_with_a_gap(self, frames: int, blank: int):
        images = {}
        dets = []
        for index in range(frames):
            cx, cy = _turning_position(index)
            if index == blank:
                images[index] = _render_blank(seed=2000 + index)
            else:
                images[index] = _render_blob(cx, cy, seed=2000 + index)
            dets.append([{"pos": (cx, cy), "score": 0.9, "track_id": 1, "infilled": False}])
        return dets, images

    def test_replay_falls_back_when_the_target_is_lost(self):
        """第 8 帧没有目标 ⇒ 经过它的重推整体弃权 ⇒ 按 fallback 兜底。"""
        frames, delta, blank = 12, 3, 8
        dets, images = self._sequence_with_a_gap(frames, blank)
        source = _DictFrameSource(images)

        # 兜底 = stale：原样搬运源帧坐标（物理帧 8 的源帧是 5）
        cfg = LookaheadConfig(
            delta=delta, mode="replay", velocity_source="diff1", replay_fallback="stale"
        )
        report, stats = build_report_stream(dets, frames, cfg, frame_source=source)
        self.assertGreater(stats["fallback_used"], 0)
        self.assertEqual(report[blank][0]["pos"], dets[blank - delta][0]["pos"])

        # 兜底 = kinematic：源帧坐标 + Δ·v
        cfg = LookaheadConfig(
            delta=delta, mode="replay", velocity_source="diff1", replay_fallback="kinematic"
        )
        report, stats = build_report_stream(dets, frames, cfg, frame_source=source)
        self.assertGreater(stats["fallback_used"], 0)
        expected = np.asarray(dets[blank - delta][0]["pos"], dtype=np.float64) + delta * np.asarray([3.0, 0.0])
        np.testing.assert_allclose(report[blank][0]["pos"], expected, atol=1e-9)
        # 运动学兜底与真值差 4 px ⇒ 这条断言确实能区分"兜底"与"重推成功"
        truth = np.asarray(_turning_position(blank))
        self.assertGreater(float(np.linalg.norm(expected - truth)), 1.0)

        # 兜底 = abstain：该帧直接不输出
        cfg = LookaheadConfig(
            delta=delta, mode="replay", velocity_source="diff1", replay_fallback="abstain"
        )
        report, stats = build_report_stream(dets, frames, cfg, frame_source=source)
        self.assertEqual(report[blank], [])
        self.assertGreater(stats["replay_abstained"], 0)


# ==============================================================================
# 5. servo_metrics：人工 (report, truth) 上的可手算指标
# ==============================================================================


class ServoMetricsTest(unittest.TestCase):
    """契约 §4 验收项 5：``servo_metrics`` 在人工构造的 (report, truth) 上给出预期值。"""

    def _case(self):
        truth = {}
        for index in range(5):
            centre_x = 100.0 + 10.0 * index
            truth[index] = np.asarray([[centre_x, 200.0, 20.0, 20.0]], dtype=np.float64)
        report = [
            [{"pos": (100.0, 200.0), "score": 0.9, "track_id": 1, "infilled": False}],
            [{"pos": (113.0, 204.0), "score": 0.9, "track_id": 1, "infilled": False}],
            [{"pos": (130.0, 200.0), "score": 0.9, "track_id": 1, "infilled": False}],
            [],
            [{"pos": (240.0, 200.0), "score": 0.9, "track_id": 1, "infilled": False}],
        ]
        return report, truth

    def test_expected_hit_rates_and_coverage(self):
        report, truth = self._case()
        metrics = servo_metrics(report, truth, 5)
        self.assertEqual(metrics["frames_with_report"], 4)
        self.assertAlmostEqual(metrics["coverage"], 0.8)
        self.assertEqual(metrics["gt_frames"], 5)
        # 帧 0/1/2 的上报点落在 20×20 的 GT 框内；帧 4 的点差 100 px ⇒ 不算
        self.assertEqual(metrics["inbox_hits"], 3)
        self.assertAlmostEqual(metrics["hit_rate_inbox"], 0.6)
        # 帧 0（误差 0）与帧 1（误差恰好 5.0）命中；帧 2（误差 10）不命中
        self.assertAlmostEqual(metrics["hit_rate_5px"], 0.4)
        self.assertEqual(metrics["lost_frames"], 1)

    def test_expected_cle_distribution(self):
        report, truth = self._case()
        metrics = servo_metrics(report, truth, 5)
        # CLE 样本 = [0, 5, 10, 100]
        self.assertAlmostEqual(metrics["cle_p50"], 7.5)
        self.assertAlmostEqual(metrics["cle_p90"], 73.0)
        self.assertAlmostEqual(metrics["cle_p95"], 86.5)
        self.assertAlmostEqual(metrics["cle_p99"], 97.3)
        self.assertAlmostEqual(metrics["cle_max"], 100.0)
        self.assertAlmostEqual(metrics["cle_mean"], 28.75)

    def test_expected_jitter_and_fragmentation(self):
        report, truth = self._case()
        metrics = servo_metrics(report, truth, 5)
        # 上报序列 (100,200) → (113,204) → (130,200) → (240,200)
        # 二阶差分：‖(17,−4)−(13,4)‖ = √80、‖(110,0)−(17,−4)‖ = √8665
        self.assertAlmostEqual(metrics["max_jitter"], math.sqrt(8665.0), places=6)
        self.assertGreater(metrics["p99_jitter"], 0.0)
        self.assertLessEqual(metrics["p99_jitter"], metrics["max_jitter"])
        # track 1 在帧 3 缺检、帧 4 又出现 ⇒ 「有 → 无 → 有」一次
        self.assertEqual(metrics["fragmentation"], 1)

    def test_empty_report_gives_zeroed_metrics(self):
        _, truth = self._case()
        metrics = servo_metrics([[] for _ in range(5)], truth, 5)
        self.assertEqual(metrics["frames_with_report"], 0)
        self.assertEqual(metrics["gt_frames"], 5)
        self.assertEqual(metrics["lost_frames"], 5)
        self.assertAlmostEqual(metrics["cle_max"], 0.0)
        self.assertAlmostEqual(metrics["max_jitter"], 0.0)


# ==============================================================================
# 6. 能变红的守卫：replay 不得读未来帧
# ==============================================================================


class LookaheadNoFutureFrameTest(unittest.TestCase):
    """契约 §4 验收项 6：``frame_source`` 读到 ``> t`` 的帧就抛异常 ⇒ replay 不读未来帧。

    守卫的构造方式：把每次运行的 ``num_frames`` 设成 ``t+1``（本次运行的**最后一个**
    物理帧就是 ``t``），帧源只允许读到 ``t``。于是任何"读到 ``t+1``"的实现
    （也就是真去读未来帧）都会当场炸掉，测试变红。
    """

    def test_replay_never_reads_a_future_frame(self):
        frames, delta = 12, 3
        images = {}
        dets = []
        for index in range(frames):
            cx, cy = _uniform_position(index)
            images[index] = _render_blob(cx, cy, seed=3000 + index)
            dets.append([{"pos": (cx, cy), "score": 0.9, "track_id": 1, "infilled": False}])

        cfg = LookaheadConfig(delta=delta, mode="replay", velocity_source="diff1")
        for target in range(delta, frames):
            source = _GuardedFrameSource(images, limit=target)
            report, _ = build_report_stream(dets, target + 1, cfg, frame_source=source)
            self.assertLessEqual(max(source.requested), target)
            # 物理帧 target 的重推必须真的读了 target 本身（否则守卫形同虚设）
            self.assertIn(target, source.requested)
            self.assertEqual(len(report[target]), 1)


if __name__ == "__main__":
    unittest.main()


# ==============================================================================
# 9. 首步离群门（2026-11-02 修复的"死代码"bug 的回归守卫）
# ==============================================================================


class _FakeObservation:
    """快核观测的最小替身：只要 ``valid`` 与 ``xy``。"""

    def __init__(self, xy, valid: bool = True) -> None:
        self.xy = np.asarray(xy, dtype=np.float64)
        self.valid = bool(valid)


class _JumpFirstTracker:
    """假快核：**第一次** ``step`` 返回一个远离 ROI 中心的点（模拟 21 px 假跳变）。

    之后每步正常走 ``(2, 0)``。用来复现"重推首步离群 ⇒ 后面几步把它放大"的机理。
    """

    def __init__(self, jump_px: float = 60.0, **kwargs) -> None:
        self.jump_px = float(jump_px)
        self.calls = 0

    def step(self, image, roi_center):
        self.calls += 1
        center = np.asarray(roi_center, dtype=np.float64).reshape(2)
        if self.calls == 1:
            return _FakeObservation(center + np.array([self.jump_px, 0.0]))
        return _FakeObservation(center + np.array([2.0, 0.0]))


class FirstStepOutlierGateTest(unittest.TestCase):
    """首步离群门**必须真的会触发**。

    历史 bug：判定写成 ``not accepted_steps``，而 ``accepted_steps.append(...)`` 在它
    **之前**执行 ⇒ 条件恒为 False ⇒ 门从来没生效过（8 序列 ``replay_step_rejected``
    恒为 0）。这两个用例把"门会触发"和"门关掉就不触发"都钉住。

    直接调 ``_replay_detections``（而不是整条流）——因为流里**所有物理帧共用同一个
    快核实例**，没法把"第一次 step"精确对齐到"某一条链的第一步"。
    """

    def _run(self, rel_tol: float):
        from manu.tracking.lookahead.extrapolator import _replay_detections

        frames = 6
        dets_by_frame = [
            [{"pos": _uniform_position(index), "score": 0.9, "track_id": 1, "infilled": False}]
            for index in range(frames)
        ]
        images = {index: _render_blank(seed=5000 + index) for index in range(frames)}
        cfg = LookaheadConfig(
            delta=3,
            mode="replay",
            velocity_source="diff1",
            replay_max_hold=8,
            replay_step_rel_tol_px=rel_tol,
        )
        outcome = _replay_detections(
            dets_by_frame[1],
            1,
            4,
            cfg,
            None,
            _JumpFirstTracker(),
            _DictFrameSource(images),
            dets_by_frame,
        )
        return outcome, np.asarray(_uniform_position(4))

    def test_gate_rejects_the_outlier_first_step(self):
        outcome, truth = self._run(rel_tol=15.0)
        self.assertGreaterEqual(outcome["step_rejected"], 1)
        # 离群首步被丢弃 ⇒ 末位上报仍在真值附近（< 5 px）
        got = np.asarray(outcome["entries"][0]["pos"], dtype=np.float64)
        self.assertLess(float(np.linalg.norm(got - truth)), 5.0)

    def test_gate_off_keeps_the_outlier_step(self):
        outcome, truth = self._run(rel_tol=0.0)
        self.assertEqual(outcome["step_rejected"], 0)
        # 门关掉 ⇒ 60 px 假跳变被当成实测步长一路带下去（> 30 px）
        got = np.asarray(outcome["entries"][0]["pos"], dtype=np.float64)
        self.assertGreater(float(np.linalg.norm(got - truth)), 30.0)


# ==============================================================================
# 10. 连续性兜底 holdc：转台不能抽搐
# ==============================================================================


class ContinuityFallbackTest(unittest.TestCase):
    """``holdc`` 兜底 = **上一帧上报位置 + 参考速度×1**（输出级状态，不回写管线）。

    锁两条性质：
    ① 兜底帧的上报位置 = 上一帧同 track 的上报位置 + 源帧处的参考速度；
    ② 单帧跳变被限制在 |v| 量级（``holdv`` 的"剩余步数×步长"会跳到上百 px）。
    """

    def _sequence_with_a_gap(self, frames: int = 12, blank: int = 8, delta: int = 3):
        images = {}
        dets = []
        for index in range(frames):
            cx, cy = _uniform_position(index)
            if index == blank:
                images[index] = _render_blank(seed=4000 + index)
            else:
                images[index] = _render_blob(cx, cy, seed=4000 + index)
            dets.append([{"pos": (cx, cy), "score": 0.9, "track_id": 1, "infilled": False}])
        return dets, images, delta, blank

    def test_holdc_continues_from_the_previous_report(self):
        dets, images, delta, blank = self._sequence_with_a_gap()
        cfg = LookaheadConfig(
            delta=delta,
            mode="replay",
            velocity_source="diff1",
            replay_max_hold=0,
            replay_fallback="holdc",
        )
        report, stats = build_report_stream(
            dets, len(dets), cfg, frame_source=_DictFrameSource(images)
        )
        self.assertGreater(stats["fallback_used"], 0)

        source = blank - delta
        velocity = np.asarray(dets[source][0]["pos"], dtype=np.float64) - np.asarray(
            dets[source - 1][0]["pos"], dtype=np.float64
        )
        previous = np.asarray(report[blank - 1][0]["pos"], dtype=np.float64)
        got = np.asarray(report[blank][0]["pos"], dtype=np.float64)
        np.testing.assert_allclose(got, previous + velocity, atol=1e-9)

        # 单帧跳变被钉在 |v| 量级（伺服不抽搐）
        self.assertLessEqual(float(np.linalg.norm(got - previous)), 1.5 * float(np.linalg.norm(velocity)))

    def test_holdc_and_holdv_use_different_fallback_paths(self):
        dets, images, delta, blank = self._sequence_with_a_gap()
        positions = {}
        for mode in ("holdv", "holdc"):
            cfg = LookaheadConfig(
                delta=delta,
                mode="replay",
                velocity_source="diff1",
                replay_max_hold=0,
                replay_fallback=mode,
            )
            report, _ = build_report_stream(
                dets, len(dets), cfg, frame_source=_DictFrameSource(images)
            )
            positions[mode] = np.asarray(report[blank][0]["pos"], dtype=np.float64)
        self.assertGreater(float(np.linalg.norm(positions["holdv"] - positions["holdc"])), 1e-6)


# ==============================================================================
# 11. rolling 档位（一条链跨物理帧持续推进，每物理帧只走一步）
# ==============================================================================

_ROLLING_HEIGHT = 200
_ROLLING_WIDTH = 400
_ROLLING_FRAMES = 40
_ROLLING_START = (80.0, 120.0)
_ROLLING_VELOCITY = (2.5, -1.5)


def _rolling_truth(index: int) -> tuple[float, float]:
    """rolling 用例的统一真值：``truth(i) = START + VELOCITY · i``（40 帧内目标不出画）。"""
    return (
        _ROLLING_START[0] + _ROLLING_VELOCITY[0] * index,
        _ROLLING_START[1] + _ROLLING_VELOCITY[1] * index,
    )


def _rolling_render(index: int, extra=None, blank: bool = False, target: bool = True):
    """渲染一帧灰度图：高斯亮斑（红外弱小目标的替身）+ 均匀噪声。

    Args:
        index (int): 帧号，同时决定噪声种子 ``default_rng(1000 + index)``。
        extra (tuple | None): 额外叠加的杂波亮斑中心 ``(x, y)``。
        blank (bool): True = 纯底噪帧（画面里一个亮斑都没有）。
        target (bool): False = 不画目标（画面里只剩 ``extra`` 杂波）。
    """
    yy, xx = np.mgrid[0:_ROLLING_HEIGHT, 0:_ROLLING_WIDTH]
    image = np.full((_ROLLING_HEIGHT, _ROLLING_WIDTH), 30.0)
    if not blank:
        if target:
            cx, cy = _rolling_truth(index)
            image = image + 120.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 2.0**2)))
        if extra is not None:
            ecx, ecy = extra
            image = image + 120.0 * np.exp(-(((xx - ecx) ** 2 + (yy - ecy) ** 2) / (2.0 * 2.0**2)))
    rng = np.random.default_rng(1000 + index)
    image = image + rng.uniform(0.0, 6.0, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


class _RollingFrameSource:
    """只读帧源：只暴露 ``get(i)``，取不到那一帧返回 None。"""

    def __init__(self, images: dict) -> None:
        self.images = images

    def get(self, frame_index: int):
        return self.images.get(int(frame_index))


class TestRollingReplay:
    """rolling 档位（一条链跨物理帧持续推进）的契约测试。

    合成数据：H=200 / W=400 的画布上，目标从 ``(80, 120)`` 以 ``(2.5, -1.5)``/帧匀速移动；
    每帧渲染一个高斯亮斑 + 均匀噪声，管线每帧给一条落在真值上的检测。
    基准配置：``delta=6 / mode="replay" / replay_mode="rolling" / velocity_source="diff1" /
    replay_fallback="holdc" / replay_max_hold=8``。

    锁的性质：跟踪精度、帧间位移的平滑性、离群步被拒、观测失效后的 hold/重捕获，
    以及 ``stats`` 的口径。
    """

    def _sequence(self, blanks=(), clutter=None, clutter_only=(), frames: int = _ROLLING_FRAMES):
        """合成 ``(dets_by_frame, images)``。

        Args:
            blanks (set): 渲染成纯底噪的帧号（观测失效场景）。
            clutter (dict): ``{帧号: (x, y)}``，额外叠加的杂波亮斑。
            clutter_only (set): 这些帧**不画目标**，只剩杂波（目标暂时不可见）。
            frames (int): 序列长度。
        """
        clutter = clutter or {}
        images, dets = {}, []
        for index in range(frames):
            images[index] = _rolling_render(
                index,
                extra=clutter.get(index),
                blank=index in blanks,
                target=index not in clutter_only,
            )
            dets.append(
                [{"pos": _rolling_truth(index), "score": 0.9, "track_id": 1, "infilled": False}]
            )
        return dets, images

    def _config(self, **overrides) -> LookaheadConfig:
        """rolling 档位的基准配置。"""
        options = dict(
            delta=6,
            mode="replay",
            replay_mode="rolling",
            velocity_source="diff1",
            replay_fallback="holdc",
            replay_max_hold=8,
        )
        options.update(overrides)
        return LookaheadConfig(**options)

    def _run(self, dets, images, **overrides):
        """跑一条上报流，返回 ``(report, stats)``。"""
        config = self._config(**overrides)
        return build_report_stream(
            dets, len(dets), config, frame_source=_RollingFrameSource(images)
        )

    @staticmethod
    def _positions(report, lo: int = 0):
        """按物理帧取上报坐标（没有上报的帧不出现）。"""
        return {
            index: np.asarray(report[index][0]["pos"], dtype=np.float64)
            for index in range(lo, len(report))
            if report[index]
        }

    @classmethod
    def _max_jitter(cls, report, lo: int = 6) -> float:
        """帧间抖动 = 相邻两帧位移向量之差的最大模。"""
        points = list(cls._positions(report, lo).values())
        steps = [points[k + 1] - points[k] for k in range(len(points) - 1)]
        return max(float(np.linalg.norm(steps[k + 1] - steps[k])) for k in range(len(steps) - 1))

    # -- 用例 1：rolling 跟踪精度 -------------------------------------------

    def test_rolling_tracks_the_target_within_one_pixel(self):
        """合成匀速轨迹上，rolling 的上报坐标与真值的中心距误差 ≤ 1 px。"""
        dets, images = self._sequence()
        report, stats = self._run(dets, images)
        positions = self._positions(report, 6)
        assert len(positions) == _ROLLING_FRAMES - 6, f"有效上报帧数 {len(positions)} != {_ROLLING_FRAMES - 6}"
        for index, got in positions.items():
            truth = np.asarray(_rolling_truth(index), dtype=np.float64)
            error = float(np.linalg.norm(got - truth))
            assert error <= 1.0, (
                f"第 {index} 帧中心距误差 {error:.4f} px > 1.0 px"
                f"（上报 {got.tolist()}，真值 {truth.tolist()}）"
            )

    def test_rolling_outputs_nothing_before_the_source_frame_exists(self):
        """``t < Δ`` 没有源帧 ⇒ 该帧不输出（上报窗口与其它臂一致）。"""
        dets, images = self._sequence()
        report, stats = self._run(dets, images)
        for index in range(6):
            assert report[index] == [], f"第 {index} 帧没有源帧却输出了 {report[index]}"
        assert stats["no_source_frames"] == 6, f"no_source_frames={stats['no_source_frames']} != 6"

    # -- 用例 2：帧间位移 ≈ 真实速度 -----------------------------------------

    def test_rolling_frame_to_frame_step_matches_the_true_speed(self):
        """相邻两帧的上报位移 ≈ 真值速度 ``hypot(2.5, 1.5)``，且帧间抖动 ≤ 0.5 px。"""
        dets, images = self._sequence()
        report, _ = self._run(dets, images)
        points = list(self._positions(report, 6).values())
        steps = [points[k + 1] - points[k] for k in range(len(points) - 1)]
        expected = math.hypot(*_ROLLING_VELOCITY)
        for k, step in enumerate(steps):
            norm = float(np.linalg.norm(step))
            assert norm == pytest.approx(expected, abs=0.5), (
                f"第 {6 + k} → {7 + k} 帧的位移模 {norm:.4f} px 偏离真速 {expected:.4f} px 超过 0.5 px"
            )
        jitter = self._max_jitter(report)
        assert jitter <= 0.5, f"帧间最大抖动 {jitter:.4f} px > 0.5 px"

    # -- 用例 3：与 perframe 的对照 -------------------------------------------

    def test_rolling_jitter_stays_inside_the_servo_budget(self):
        """rolling 的帧间抖动必须留在伺服平滑预算内，且与 perframe 不差出数量级。

        **当前实现的真实行为（2026-11-03 实测，干净合成序列）**：
        rolling 的最大帧间抖动 **0.370 px 大于** perframe 的 **0.085 px**，
        与 "rolling 比 perframe 更平滑" 的预期相反。机理：perframe 每 Δ 帧从**精确的
        管线锚点**重新起链，质心偏置不累积；rolling 的锚点残差修正是 "滞后 Δ 帧的
        偏置差"，帧间抖动因此更大（rolling 的设计收益在**源帧失锁**场景——perframe
        整段作废，干净序列上体现不出来）。因此这里锁的是 "两臂同量级 + rolling 满足
        伺服平滑预算"，不锁方向（方向会随实现演进）。
        """
        dets, images = self._sequence()
        rolling_report, _ = self._run(dets, images)
        perframe_report, _ = self._run(dets, images, replay_mode="perframe")
        rolling_jitter = self._max_jitter(rolling_report)
        perframe_jitter = self._max_jitter(perframe_report)
        assert rolling_jitter <= 0.5, f"rolling 最大帧间抖动 {rolling_jitter:.4f} px > 0.5 px"
        assert perframe_jitter <= 0.5, f"perframe 最大帧间抖动 {perframe_jitter:.4f} px > 0.5 px"
        assert rolling_jitter <= 10.0 * perframe_jitter, (
            f"rolling 抖动 {rolling_jitter:.4f} px 与 perframe 抖动 {perframe_jitter:.4f} px "
            f"相差超过一个数量级（rolling 抖得离谱）"
        )

    # -- 用例 4：离群步被拒（防 runaway） -------------------------------------

    def test_far_outlier_blob_never_drags_the_report(self):
        """远处（+60 px）的强亮斑不能把上报坐标拽过去。

        **当前实现的真实行为**：快核 ROI 只有 32 px（中心 ±16 px），60 px 外的亮斑
        **根本进不了快核**，所以离群步门（``rolling_max_step_px=12``）根本没被触发
        （``rolling_step_rejected=0``）——挡住 runaway 的是 "快核看不见它"，不是门。
        """
        blob = (_rolling_truth(20)[0] + 60.0, _rolling_truth(20)[1])
        dets, images = self._sequence(clutter={20: blob})
        report, stats = self._run(dets, images)
        blob_position = np.asarray(blob, dtype=np.float64)
        for index, got in self._positions(report, 20).items():
            truth = np.asarray(_rolling_truth(index), dtype=np.float64)
            error = float(np.linalg.norm(got - truth))
            assert error <= 3.0, (
                f"第 {index} 帧上报 {got.tolist()} 离真值 {error:.4f} px > 3.0 px"
                f"（rolling_step_rejected={stats['rolling_step_rejected']}）"
            )
            distance = float(np.linalg.norm(got - blob_position))
            assert distance >= 20.0, (
                f"第 {index} 帧上报 {got.tolist()} 跳到了杂波亮斑上（距离 {distance:.2f} px）"
            )
        assert stats["rolling_step_rejected"] == 0, (
            f"rolling_step_rejected={stats['rolling_step_rejected']} != 0"
            f"（32 px 的快核 ROI 看不到 60 px 外的亮斑，门不应被触发）"
        )

    def test_outlier_step_inside_the_kernel_roi_is_rejected(self):
        """ROI 内的离群步被门挡下（走 hold），上报坐标不跟着杂波跑。

        构造：第 20 帧目标不可见（画面里只剩杂波），杂波落在链位置右侧 14 px —— 这是
        32 px 的快核 ROI 能看见、且单步位移（≈13 px）超过 ``rolling_max_step_px=12``
        的唯一可行距离 ⇒ 门必须把这一步作废，链改走 hold。
        """
        clutter = (_rolling_truth(19)[0] + 14.0, _rolling_truth(20)[1])
        dets, images = self._sequence(clutter={20: clutter}, clutter_only={20})
        report, stats = self._run(dets, images)
        assert stats["rolling_step_rejected"] >= 1, (
            f"rolling_step_rejected={stats['rolling_step_rejected']} < 1"
            f"（杂波在 ROI 内、单步位移 {stats['replay_max_step_disp']:.2f} px，离群步没有被拒）"
        )
        for index, got in self._positions(report, 20).items():
            truth = np.asarray(_rolling_truth(index), dtype=np.float64)
            error = float(np.linalg.norm(got - truth))
            assert error <= 3.0, (
                f"第 {index} 帧上报 {got.tolist()} 离真值 {error:.4f} px > 3.0 px"
                f"（链跟到杂波上去了，杂波在 {list(clutter)}）"
            )

    # -- 用例 5：观测失效 ⇒ hold / 链死亡后的重捕获 ---------------------------

    def test_rolling_holds_through_a_short_observation_gap(self):
        """连续 4 帧没有观测（画面只剩底噪）⇒ 链走 hold-over，不发散。

        **当前实现的真实行为**：空洞长度 4 ≤ ``replay_max_hold=8`` ⇒ 链**没有死**
        （``rolling_dead_steps=0``、``rolling_reacquired=0``），只是按 hold-over 继续
        推进（``replay_held_steps=4``）。
        """
        dets, images = self._sequence(blanks=set(range(25, 29)))
        report, stats = self._run(dets, images)
        assert stats["replay_held_steps"] >= 1, f"replay_held_steps={stats['replay_held_steps']} < 1"
        assert stats["rolling_dead_steps"] == 0, (
            f"rolling_dead_steps={stats['rolling_dead_steps']} != 0"
            f"（空洞 4 帧 ≤ max_hold=8，链不该死）"
        )
        for index, got in self._positions(report, 24).items():
            truth = np.asarray(_rolling_truth(index), dtype=np.float64)
            error = float(np.linalg.norm(got - truth))
            assert error <= 3.0, f"第 {index} 帧上报 {got.tolist()} 离真值 {error:.4f} px > 3.0 px"

    def test_rolling_chain_dies_and_is_reacquired_after_a_long_gap(self):
        """连续 11 帧没有观测（超过 ``max_hold``）⇒ 链死亡并被锚点重捕获，且不发散。"""
        dets, images = self._sequence(blanks=set(range(25, 36)))
        report, stats = self._run(dets, images)
        assert stats["rolling_dead_steps"] >= 1 or stats["rolling_reacquired"] >= 1, (
            f"rolling_dead_steps={stats['rolling_dead_steps']}, "
            f"rolling_reacquired={stats['rolling_reacquired']}（链既没死也没被重捕获）"
        )
        positions = self._positions(report, 6)
        assert len(positions) == _ROLLING_FRAMES - 6, (
            f"失效期间 rolling 整段作废：有效上报帧数 {len(positions)} != {_ROLLING_FRAMES - 6}"
        )
        for index, got in positions.items():
            truth = np.asarray(_rolling_truth(index), dtype=np.float64)
            error = float(np.linalg.norm(got - truth))
            assert error <= 20.0, (
                f"第 {index} 帧上报 {got.tolist()} 离真值 {error:.4f} px > 20 px（链发散了）"
            )

    # -- 用例 6：与 perframe 的行为一致性（回归护栏） -------------------------

    def test_rolling_stats_follow_the_replay_contract(self):
        """``stats`` 的口径：mode / delta / 步数，以及 applied 与 abstained 互补。"""
        dets, images = self._sequence()
        report, stats = self._run(dets, images)
        assert stats["mode"] == "replay", f"mode={stats['mode']!r} != 'replay'"
        assert stats["delta"] == 6, f"delta={stats['delta']} != 6"
        assert stats["replay_steps"] > 0, f"replay_steps={stats['replay_steps']} <= 0"
        assert stats["applied"] + stats["abstained"] == _ROLLING_FRAMES, (
            f"applied({stats['applied']}) + abstained({stats['abstained']}) != {_ROLLING_FRAMES}"
        )
        assert stats["applied"] == _ROLLING_FRAMES - 6, (
            f"applied={stats['applied']} != {_ROLLING_FRAMES - 6}"
        )


class OutputSlewLimitTest(unittest.TestCase):
    """输出级限幅（伺服 slew limiter）的契约测试。

    锁的是**机制**，不是数据集上的数字：

    * ``output_max_step_px = 0``（默认）⇒ 输出**逐位**不变（既有行为完全可复现）；
    * 限幅开启 ⇒ 相邻两帧的输出位移 ≤ 上限 ⇒ 伺服抖动（二阶差分）≤ 2×上限；
    * track 消失若干帧后再出现 ⇒ 视为重新捕获，**原样放行**（不被"追不上"）；
    * 限幅**只作用于 replay 臂**：stale / kinematic 对照基线臂不受影响。
    """

    @staticmethod
    def _det(pos, track_id=1):
        return {
            "pos": (float(pos[0]), float(pos[1])),
            "score": 0.9,
            "track_id": track_id,
            "infilled": False,
        }

    @staticmethod
    def _steps(report):
        out = []
        for index in range(1, len(report)):
            if not report[index] or not report[index - 1]:
                continue
            out.append(
                float(
                    np.linalg.norm(
                        np.asarray(report[index][0]["pos"], dtype=float)
                        - np.asarray(report[index - 1][0]["pos"], dtype=float)
                    )
                )
            )
        return out

    def test_limit_off_keeps_the_output_verbatim(self):
        report = [[self._det((0.0, 0.0))], [self._det((190.0, 0.0))]]
        stats = {}
        _apply_output_slew_limit(report, LookaheadConfig(output_max_step_px=0.0), stats)
        self.assertEqual(report[1][0]["pos"], (190.0, 0.0))
        self.assertEqual(int(stats.get("output_step_limited", 0)), 0)

    def test_limit_caps_the_step_and_bounds_the_servo_jitter(self):
        # 孤立单帧飞跳 190 px（实测 014053 frame 427 的机理）
        report = [
            [self._det((0.0, 0.0))],
            [self._det((190.0, 0.0))],
            [self._det((195.0, 0.0))],
        ]
        stats = {}
        _apply_output_slew_limit(report, LookaheadConfig(output_max_step_px=30.0), stats)
        self.assertAlmostEqual(report[1][0]["pos"][0], 30.0, places=6)
        steps = self._steps(report)
        self.assertLessEqual(max(steps), 30.0 + 1e-9)
        # 二阶差分（转台看到的抖动）被限幅钉在 2×上限以内
        jitter = abs(steps[1] - steps[0])
        self.assertLessEqual(jitter, 60.0 + 1e-9)
        # 第 1 帧（0→190）与第 2 帧（30→195）都被限幅 ⇒ 限了 2 帧
        self.assertEqual(int(stats.get("output_step_limited", 0)), 2)

    def test_reacquisition_after_a_gap_is_also_slew_limited(self):
        # track 在第 2 帧缺席，第 3 帧重新出现且位置已经挪了 150 px：
        # 伺服在目标丢失期间是"保持"而不是"瞬移" ⇒ 重新捕获也要按速率限幅追过去，
        # 否则跨缺口的重新捕获会原样放行，直接把二阶差分顶到 100 px 以上。
        report = [
            [self._det((0.0, 0.0))],
            [],
            [self._det((150.0, 0.0))],
        ]
        stats = {}
        _apply_output_slew_limit(report, LookaheadConfig(output_max_step_px=30.0), stats)
        self.assertAlmostEqual(report[2][0]["pos"][0], 30.0, places=6)
        self.assertEqual(int(stats.get("output_step_limited", 0)), 1)

    def test_first_appearance_has_no_reference_and_passes_through(self):
        # 没有任何历史参考（track 首次出现）⇒ 原样放行
        report = [[self._det((150.0, 0.0))]]
        stats = {}
        _apply_output_slew_limit(report, LookaheadConfig(output_max_step_px=30.0), stats)
        self.assertEqual(report[0][0]["pos"], (150.0, 0.0))
        self.assertEqual(int(stats.get("output_step_limited", 0)), 0)

    def test_baseline_arms_are_not_limited(self):
        # stale 臂是"老逻辑硬吃 Δ 帧滞后"的对照基线：限幅不得作用到它
        # stale 臂：report[t] = dets[t-Δ]（Δ=1）
        dets = [
            [self._det((100.0, 100.0))],
            [self._det((100.0, 100.0))],
            [self._det((280.0, 100.0))],
            [self._det((280.0, 100.0))],
        ]
        config = LookaheadConfig(delta=1, mode="stale", output_max_step_px=30.0)
        report, _ = build_report_stream(dets, len(dets), config)
        # report[3] 来自 dets[2] = (280, 100)：若限幅误作用到基线臂，它会被削到 130
        self.assertEqual(report[3][0]["pos"], (280.0, 100.0))


class AbstentionOnUnconfidentMatchTest(unittest.TestCase):
    """「不自信就不上报」：模板峰值低 **且** 链跑得远 ⇒ 该帧不上报。

    机理：模板配准的相关峰是这次重推唯一的自洽性证据。峰值低有两种成因——
    目标机动太大（模板跟不上，但快核跟得住，位置其实是对的）或快核跟丢了（位置是
    错的）。用**链位移**区分：跟丢时链从锚点跑出去很远。默认关闭（``0``），关着
    时行为与历史逐位一致。
    """

    def _sequence(self, frames=10, delta=3, vanish=9, speed=10.0):
        """目标高速移动，并在最后一帧消失：模板配不上 ⇒ 峰值≈0。"""
        images = {}
        dets = []
        for index in range(frames):
            cx, cy = 40.0 + speed * index, 60.0
            if index == vanish:
                images[index] = _render_blank(seed=3000 + index)
            else:
                images[index] = _render_blob(cx, cy, seed=3000 + index)
            dets.append([{"pos": (cx, cy), "score": 0.9, "track_id": 1, "infilled": False}])
        return dets, images

    def _config(self, **overrides):
        base = dict(
            delta=3,
            mode="replay",
            velocity_source="diff1",
            replay_anchor="delta",
            replay_xcorr_radius=24,
            replay_xcorr_template=16,
            replay_xcorr_min_peak=0.25,
            replay_max_hold=8,
            replay_fallback="holdv",
        )
        base.update(overrides)
        return LookaheadConfig(**base)

    def test_abstention_is_off_by_default(self):
        """默认（峰门 0）⇒ 行为与历史一致：该帧照常上报。"""
        dets, images = self._sequence()
        report, stats = build_report_stream(dets, 10, self._config(), frame_source=_DictFrameSource(images))
        self.assertEqual(len(report[9]), 1)
        self.assertEqual(stats.get("replay_xcorr_abstained", 0), 0)

    def test_low_peak_with_a_far_chain_abstains_the_frame(self):
        """峰值低 + 链跑得远 ⇒ 不上报（宁可沉默，也不要给转台一个错坐标）。"""
        dets, images = self._sequence()
        cfg = self._config(replay_abstain_peak_below=0.5, replay_abstain_disp_px=5.0)
        report, stats = build_report_stream(dets, 10, cfg, frame_source=_DictFrameSource(images))
        self.assertEqual(report[9], [])
        self.assertGreaterEqual(stats.get("replay_xcorr_abstained", 0), 1)
        # 弃权是**局部**的：其它帧照常上报
        self.assertEqual(len(report[8]), 1)

    def test_low_peak_with_a_stationary_chain_keeps_the_frame(self):
        """峰值低但链几乎没跑动（目标在原地）⇒ 位置其实是对的 ⇒ 不该弃权。"""
        frames, delta = 10, 3
        images = {}
        dets = []
        for index in range(frames):
            images[index] = _render_blob(40.0, 60.0, seed=4000 + index)
            dets.append([{"pos": (40.0, 60.0), "score": 0.9, "track_id": 1, "infilled": False}])
        cfg = self._config(replay_abstain_peak_below=0.5, replay_abstain_disp_px=5.0)
        report, stats = build_report_stream(dets, frames, cfg, frame_source=_DictFrameSource(images))
        self.assertEqual(len(report[frames - 1]), 1)
        self.assertEqual(stats.get("replay_xcorr_abstained", 0), 0)
