"""条件化对比模块的契约测试（口径守卫）。

这些测试锁住的是**统计口径**，防止有人改公式时悄悄换掉分母/分子：

- 位置为 NaN 的帧不参与评测（与 ``servo_metrics`` 一致）；
- 只在**目标存在**的帧上算 CLE（多框 GT 取**第一个**框的中心）；
- 命中率的分母是"参与评测的帧数"，不是序列总帧数；
- 配对比较必须严格同帧，且平局单独计数；
- 帧集切分只看臂 3 自己的 ``fusion_kind``，弃权原因只取前缀。

⚠️ 这些用例必须**能区分不同口径**——若把 ``_cle_map`` 的遍历源从 GT 换成位置，
或把「取第一个框」换成「取最后一个框」，测试必须变红。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from manu.tracking.async_dual_rate.conditional import (
    _cle_map,
    _paired,
    _positions,
    _split_frame_sets,
    _stats,
)


class PositionsTest(unittest.TestCase):
    def test_nan_positions_are_dropped(self):
        records = [
            {"frame_index": 0, "x": 10.0, "y": 20.0},
            {"frame_index": 1, "x": float("nan"), "y": 20.0},
            {"frame_index": 2, "x": 30.0, "y": float("nan")},
        ]
        positions = _positions(records)
        self.assertEqual(sorted(positions), [0])
        np.testing.assert_allclose(positions[0], [10.0, 20.0])

    def test_missing_keys_are_treated_as_nan(self):
        records = [{"frame_index": 7}, {"frame_index": 8, "x": 1.0, "y": 2.0}]
        positions = _positions(records)
        self.assertEqual(sorted(positions), [8])


class CleMapTest(unittest.TestCase):
    def setUp(self):
        # 帧 0：有位置 + 有 GT
        # 帧 1：有位置 + 无 GT  ⇒ 必须被排除（目标不存在，不该评）
        # 帧 2：无位置 + 有 GT  ⇒ 必须被排除（没输出，评不了）
        self.positions = {
            0: np.asarray([10.0, 20.0]),
            1: np.asarray([30.0, 40.0]),
        }
        self.ground_truth = {
            0: np.asarray([[10.0, 20.0, 4.0, 4.0]]),
            2: np.asarray([[10.0, 20.0, 4.0, 4.0]]),
        }

    def test_only_frames_with_both_position_and_ground_truth_are_evaluated(self):
        """帧 1（有位置无 GT）和帧 2（有 GT 无位置）都必须被排除。

        如果把遍历源从 ``ground_truth`` 换成 ``positions``，帧 1 会被算进来 ⇒ 本用例变红。
        """
        cle = _cle_map(self.positions, self.ground_truth)
        self.assertEqual(sorted(cle), [0])
        self.assertAlmostEqual(cle[0], 0.0)

    def test_distance_is_to_ground_truth_centre(self):
        positions = {0: np.asarray([13.0, 24.0])}
        cle = _cle_map(positions, self.ground_truth)
        self.assertAlmostEqual(cle[0], 5.0)

    def test_multi_box_ground_truth_uses_the_first_box(self):
        """多框 GT 取**第一个**框的中心（与 ``servo_metrics._gt_centre`` 一致）。

        如果改成取最后一个框，本用例必须变红。
        """
        positions = {0: np.asarray([100.0, 100.0])}
        ground_truth = {
            0: np.asarray(
                [
                    [10.0, 20.0, 4.0, 4.0],
                    [200.0, 200.0, 8.0, 8.0],
                ]
            )
        }
        cle = _cle_map(positions, ground_truth)
        expected = float(
            np.linalg.norm(np.asarray([100.0, 100.0]) - np.asarray([10.0, 20.0]))
        )
        self.assertAlmostEqual(cle[0], expected)


class StatsTest(unittest.TestCase):
    def test_empty(self):
        """空帧集也必须返回**完整键集**（下游 render_markdown / verify_docs 依赖它）。"""
        stats = _stats([])
        self.assertEqual(stats["n"], 0)
        for key in ("cle_p50", "cle_p90", "cle_p95", "hit_rate_pct"):
            self.assertTrue(math.isnan(stats[key]), f"{key} 应当是 NaN")

    def test_percentiles_and_hit_rate(self):
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]
        stats = _stats(values, tolerance=5.0)
        self.assertEqual(stats["n"], 8)
        self.assertAlmostEqual(stats["cle_p50"], 3.5)
        # numpy 线性插值：(8-1)*0.95 = 6.65 ⇒ 6 + 0.65*(100-6) = 67.1
        self.assertAlmostEqual(stats["cle_p95"], 67.1)
        # 5.0 及以内算命中 => 6/8
        self.assertAlmostEqual(stats["hit_rate_pct"], 75.0)

    def test_tolerance_boundary_is_inclusive(self):
        stats = _stats([5.0, 5.0001], tolerance=5.0)
        self.assertAlmostEqual(stats["hit_rate_pct"], 50.0)


class PairedTest(unittest.TestCase):
    def test_counts_better_worse_tie(self):
        base = {0: 1.0, 1: 5.0, 2: 3.0}
        other = {0: 2.0, 1: 4.0, 2: 3.0}
        paired = _paired(base, other, {0, 1, 2})
        self.assertEqual(paired["paired_frames"], 3)
        self.assertEqual(paired["better"], 1)   # 帧 0：1 < 2
        self.assertEqual(paired["worse"], 1)    # 帧 1：5 > 4
        self.assertEqual(paired["tie"], 1)      # 帧 2：完全相等
        self.assertAlmostEqual(paired["median_diff_px"], 0.0)

    def test_frames_missing_in_either_arm_are_not_paired(self):
        paired = _paired({0: 1.0}, {1: 2.0}, {0, 1})
        self.assertEqual(paired["paired_frames"], 0)
        self.assertEqual(paired["better"], 0)
        self.assertEqual(paired["worse"], 0)


class SplitFrameSetsTest(unittest.TestCase):
    """帧集切分是 §6 全部数字的唯一来源，必须有守卫。"""

    def setUp(self):
        self.records = [
            {"frame_index": 0, "fusion_kind": "replay", "replay_reason": "ok"},
            {"frame_index": 1, "fusion_kind": "replay", "replay_reason": "ok"},
            {"frame_index": 2, "fusion_kind": "replay_abstain", "replay_reason": "lost_target:1055"},
            {"frame_index": 3, "fusion_kind": "replay_abstain", "replay_reason": "lost_target:1055"},
            {"frame_index": 4, "fusion_kind": "replay_abstain", "replay_reason": "no_history_for_anchor"},
            {"frame_index": 5, "fusion_kind": "none", "replay_reason": ""},
        ]

    def test_splits_on_fusion_kind(self):
        success, abstain, reasons = _split_frame_sets(self.records)
        self.assertEqual(success, {0, 1})
        self.assertEqual(abstain, {2, 3, 4})
        self.assertNotIn(5, success)
        self.assertNotIn(5, abstain)

    def test_abstain_reasons_take_the_prefix_only(self):
        """``lost_target:1055`` 只按 ``lost_target`` 计数（帧号不进原因名）。"""
        _, _, reasons = _split_frame_sets(self.records)
        self.assertEqual(reasons, {"lost_target": 2, "no_history_for_anchor": 1})

    def test_empty_records(self):
        success, abstain, reasons = _split_frame_sets([])
        self.assertEqual(success, set())
        self.assertEqual(abstain, set())
        self.assertEqual(reasons, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
