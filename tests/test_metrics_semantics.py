"""新口径指标的契约测试。

这一组用例锁的是**语义**，不是数字：

* ``mcnemar_exact``：配对检验的数值必须与手算精确二项一致，且在
  discordant 上千时**不能溢出**（``2**n`` 会 OverflowError）；
* ``step_jump``：规格判据 1「检测异步到达校正那一帧的一阶位移 > 2 px」必须
  被**真的度量**，且 smooth / hard_reinit / replay / transition / steady
  五种来源必须**互斥且穷尽**（漏掉 replay 会污染稳态统计）；
* ``lost_frame_rate``：脱靶率 = 目标存在但未锁定的帧占比；
* 覆盖率分子必须与分母同属一个帧集合。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from manu.tracking.async_dual_rate.arms import arm1_detector_only, arm1_detector_only_sparse
from manu.tracking.async_dual_rate.servo_metrics import (
    ATTRIBUTION_BUCKETS,
    mcnemar_exact,
    servo_metrics,
)


def _record(frame_index, x, y, **kwargs):
    record = {
        "frame_index": frame_index,
        "x": float(x),
        "y": float(y),
        "state": "TRACK",
        "fusion_kind": "none",
        "det_source": -1,
        "events": "",
    }
    record.update(kwargs)
    return record


class McNemarTest(unittest.TestCase):
    def test_matches_hand_computed_exact_binomial(self):
        """b=10, c=3：discordant=13，双侧 p = 2 * (1+13+78+286) / 2^13。"""
        result = mcnemar_exact(10, 3)
        expected = 2.0 * (1 + 13 + 78 + 286) / (2**13)
        self.assertAlmostEqual(result["p_value"], expected, places=12)
        self.assertAlmostEqual(result["z"], (10 - 3) / math.sqrt(13), places=12)
        self.assertFalse(result["significant_5pct"])

    def test_identical_arms_give_no_signal(self):
        result = mcnemar_exact(0, 0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertFalse(result["significant_5pct"])
        self.assertEqual(result["discordant"], 0)

    def test_large_discordant_does_not_overflow(self):
        """discordant 上千时 2**n 会 OverflowError——必须用对数空间求和。"""
        result = mcnemar_exact(1050, 950)
        self.assertEqual(result["discordant"], 2000)
        self.assertTrue(math.isfinite(result["p_value"]))
        self.assertTrue(result["significant_5pct"])

    def test_extreme_split_is_significant(self):
        result = mcnemar_exact(200, 20)
        self.assertTrue(result["significant_5pct"])
        self.assertLess(result["p_value"], 1e-10)


class StepJumpAttributionTest(unittest.TestCase):
    """规格判据 1：一阶差分 Step Jump，必须归因到异步校正帧。"""

    def _metrics(self, records):
        ground_truth = {index: np.asarray([[100.0, 100.0, 10.0, 10.0]]) for index in range(200)}
        return servo_metrics(records, ground_truth, tolerance=5.0, lock_radius=16.0).to_dict()

    def test_buckets_are_mutually_exclusive_and_exhaustive(self):
        self.assertEqual(
            set(ATTRIBUTION_BUCKETS),
            {"transition", "replay", "replay_abstain", "hard_reinit", "smooth", "steady"},
        )

    def test_correction_frames_are_measured_and_attributed(self):
        # 平稳段（每帧 +1 px）；第 10 帧是 smooth 校正帧，跳 30 px。
        records = []
        position = 100.0
        for index in range(40):
            if index == 10:
                position += 30.0
                records.append(
                    _record(index, position, 100.0, fusion_kind="smooth", events="")
                )
            else:
                position += 1.0
                records.append(_record(index, position, 100.0))
        metrics = self._metrics(records)
        jump = metrics["jitter"]["step_jump"]

        self.assertEqual(jump["threshold_px"], 2.0)
        self.assertGreaterEqual(jump["at_correction_smooth"]["n"], 1)
        # smooth 那一帧的一阶位移是 30 px（> 2 px 门限）
        self.assertGreater(jump["at_correction_smooth"]["max"], 2.0)
        self.assertGreaterEqual(jump["over_threshold_counts"]["smooth"], 1)
        self.assertFalse(jump["criterion_2px_pass_max"])

    def test_replay_is_not_counted_as_steady(self):
        """漏掉 replay 桶 ⇒ 臂 3 最大的有意跳变会污染稳态统计。"""
        records = []
        position = 100.0
        for index in range(40):
            if index == 10:
                position += 40.0
                records.append(
                    _record(index, position, 100.0, fusion_kind="replay", events="")
                )
            else:
                position += 1.0
                records.append(_record(index, position, 100.0))
        metrics = self._metrics(records)
        jumps = metrics["jitter"]["jump_attribution"]
        self.assertGreaterEqual(jumps["replay"], 1)
        self.assertEqual(jumps["steady"], 0)


class LostRateTest(unittest.TestCase):
    def test_lost_frame_rate_counts_unlocked_target_frames(self):
        ground_truth = {index: np.asarray([[100.0, 100.0, 10.0, 10.0]]) for index in range(10)}
        records = []
        for index in range(10):
            if index < 6:
                records.append(_record(index, 100.0, 100.0))
            elif index < 8:
                records.append(_record(index, 140.0, 100.0))  # 离 GT 40 px > lock_radius
            # index 8、9 完全无输出
        metrics = servo_metrics(records, ground_truth, tolerance=5.0, lock_radius=16.0).to_dict()
        fragmentation = metrics["fragmentation"]
        self.assertEqual(fragmentation["target_frames"], 10)
        self.assertEqual(fragmentation["lost_frames"], 4)
        self.assertAlmostEqual(fragmentation["lost_frame_rate"], 0.4)
        self.assertEqual(fragmentation["no_output_frames"], 2)


class SparseBaselineArmTest(unittest.TestCase):
    def test_sparse_variant_matches_arm1_except_anchor_density(self):
        dense = arm1_detector_only()
        sparse = arm1_detector_only_sparse()
        self.assertEqual(sparse.kind, dense.kind)
        self.assertEqual(sparse.latency_frames, dense.latency_frames)
        self.assertEqual(dense.det_hz, 25.0)
        self.assertEqual(sparse.det_hz, 12.5)
        self.assertNotEqual(sparse.name, dense.name)

    def test_sparse_variant_accepts_cli_parameters(self):
        sparse = arm1_detector_only_sparse(det_hz=8.33, latency_frames=4)
        self.assertEqual(sparse.det_hz, 8.33)
        self.assertEqual(sparse.latency_frames, 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
