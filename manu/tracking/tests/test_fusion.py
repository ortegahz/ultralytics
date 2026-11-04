"""异步延迟残差融合的契约测试（含"空对照臂"：残差为零 ⇒ 位级不变）。"""

from __future__ import annotations

import unittest

import numpy as np

from manu.tracking.async_dual_rate.fusion import FusionConfig, ResidualFusion
from manu.tracking.async_dual_rate.history import HistoryRingBuffer


class ResidualFusionTest(unittest.TestCase):
    def setUp(self):
        self.history = HistoryRingBuffer(capacity=64)
        # 帧 0..20：目标以每帧 (1, 0) 匀速运动，位置精确已知
        for index in range(21):
            self.history.push(index, (100.0 + index, 200.0))

    def test_abstains_when_history_lacks_the_source_frame(self):
        """拿不到同刻历史 ⇒ 弃权，绝不拿错时刻的差值猛推伺服。"""
        fusion = ResidualFusion(FusionConfig(gating_dist=12.0, alpha=0.4))
        decision = fusion.fuse(
            position=(150.0, 200.0),
            detection_xy=(160.0, 200.0),
            detection_frame=999,           # 不在缓冲里
            current_frame=1000,
            history=self.history,
        )
        self.assertFalse(decision.applied)
        self.assertEqual(decision.kind, "none")
        np.testing.assert_allclose(decision.position_after, [150.0, 200.0])

    def test_null_control_zero_residual_is_bit_identical(self):
        """空对照臂：检测框与历史位置完全一致 ⇒ 输出必须逐位不变。"""
        fusion = ResidualFusion(FusionConfig(gating_dist=12.0, alpha=0.4))
        source_frame = 10
        historical = self.history.get(source_frame)
        position = np.array([120.0, 200.0])
        decision = fusion.fuse(
            position=position,
            detection_xy=historical,
            detection_frame=source_frame,
            current_frame=18,
            history=self.history,
        )
        self.assertTrue(decision.applied)
        self.assertEqual(decision.kind, "smooth")
        np.testing.assert_array_equal(decision.position_after, position)

    def test_small_residual_is_smoothed_by_alpha(self):
        """小残差：x ← x + α·e。"""
        alpha = 0.4
        fusion = ResidualFusion(FusionConfig(gating_dist=12.0, alpha=alpha))
        source_frame = 10
        historical = self.history.get(source_frame)
        offset = np.array([3.0, -2.0])           # 残差模长 3.6 < 12
        position = np.array([120.0, 200.0])
        decision = fusion.fuse(
            position=position,
            detection_xy=historical + offset,
            detection_frame=source_frame,
            current_frame=18,
            history=self.history,
        )
        self.assertEqual(decision.kind, "smooth")
        np.testing.assert_allclose(decision.position_after, position + alpha * offset, atol=1e-12)

    def test_large_residual_hard_reinits_with_velocity_projection(self):
        """大残差：硬拉回，且把 Δ 帧前的检测位置按历史速度投影到当前时刻。

        必须**显式**指定 ``velocity_source="history"``：默认是 ``"none"``（不投影，
        直接落在 ``z_f``），不写就退化成"落在 z_f"而不是"按历史速度投影"。
        """
        fusion = ResidualFusion(
            FusionConfig(gating_dist=12.0, alpha=0.4, hard_reinit=True, velocity_source="history")
        )
        source_frame = 10
        current_frame = 18
        historical = self.history.get(source_frame)
        far = np.array([400.0, 200.0])            # 残差 ~300 ≫ 12
        decision = fusion.fuse(
            position=(120.0, 200.0),
            detection_xy=far,
            detection_frame=source_frame,
            current_frame=current_frame,
            history=self.history,
        )
        self.assertEqual(decision.kind, "hard_reinit")
        # 速度 = (x_{17} - x_{10}) / 7 = (1, 0)
        expected = far + np.array([1.0, 0.0]) * (current_frame - source_frame)
        np.testing.assert_allclose(decision.position_after, expected, atol=1e-9)

    def test_zero_history_velocity_is_not_replaced_by_detection_fallback(self):
        """历史速度为 0（目标静止）是**合法速度**，不得被检测流兜底速度顶掉。

        早期实现是"历史速度算出来是 0 就换用检测流速度"，在跨目标切换的场景里
        会把一个正确的 0 换成脏速度（−2.857 px/帧），乘 Δ 后变成假位移。
        """
        static = HistoryRingBuffer(capacity=64)
        for index in range(21):
            static.push(index, (100.0, 200.0))     # 位置完全相同 ⇒ 历史速度恒为 0
        fusion = ResidualFusion(
            FusionConfig(gating_dist=12.0, alpha=0.4, hard_reinit=True, velocity_source="history")
        )
        source_frame = 10
        current_frame = 18
        far = np.array([400.0, 200.0])            # 残差 ~300 ≫ 12
        dirty = np.array([-2.857, 0.0])           # 跨目标切换算出的脏速度
        decision = fusion.fuse(
            position=(120.0, 200.0),
            detection_xy=far,
            detection_frame=source_frame,
            current_frame=current_frame,
            history=static,
            fallback_velocity=dirty,
        )
        self.assertEqual(decision.kind, "hard_reinit")
        # 历史速度取得到（不是 0 才算取不到）⇒ 不得用兜底速度 ⇒ 直接落在 z_f
        np.testing.assert_array_equal(decision.position_after, far)
        # 没有兜底速度时结果同样落在 z_f：两条路径必须一致
        decision_without_fallback = fusion.fuse(
            position=(120.0, 200.0),
            detection_xy=far,
            detection_frame=source_frame,
            current_frame=current_frame,
            history=static,
            fallback_velocity=None,
        )
        self.assertEqual(decision_without_fallback.kind, "hard_reinit")
        np.testing.assert_array_equal(decision_without_fallback.position_after, far)

    def test_default_velocity_source_is_none_and_lands_on_z_f(self):
        """默认 ``velocity_source="none"``：硬拉回**不投影**，直接落在 ``z_f``。

        即便历史速度非 0（目标在动）也不加 ``v × Δ``——乘 Δ 会被放大成假位移。
        """
        fusion = ResidualFusion(FusionConfig(gating_dist=12.0, alpha=0.4, hard_reinit=True))
        source_frame = 10
        current_frame = 18
        far = np.array([400.0, 200.0])            # 残差 ~300 ≫ 12
        decision = fusion.fuse(
            position=(120.0, 200.0),
            detection_xy=far,
            detection_frame=source_frame,
            current_frame=current_frame,
            history=self.history,                 # 历史速度 = (1, 0) ≠ 0
        )
        self.assertEqual(decision.kind, "hard_reinit")
        np.testing.assert_array_equal(decision.position_after, far)

    def test_hard_reinit_can_be_disabled(self):
        """消融臂：关掉硬拉回后，大残差也只做平滑。"""
        fusion = ResidualFusion(FusionConfig(gating_dist=12.0, alpha=0.25, hard_reinit=False))
        source_frame = 10
        historical = self.history.get(source_frame)
        offset = np.array([50.0, 0.0])
        position = np.array([120.0, 200.0])
        decision = fusion.fuse(
            position=position,
            detection_xy=historical + offset,
            detection_frame=source_frame,
            current_frame=18,
            history=self.history,
        )
        self.assertEqual(decision.kind, "smooth")
        np.testing.assert_allclose(decision.position_after, position + 0.25 * offset, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
