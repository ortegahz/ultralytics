"""目标极性（亮/暗）的回归防线（2026-11-03 真实 bug 的教训）。

**Bug 本体**：快核与 xcorr 模板都写死了「比背景亮」。龙泉山实测里目标**逐帧会变极性**
（``014453`` 整条是暗的；``015000`` 只有前 175 帧是暗的）。亮假设遇上暗目标 ⇒
权重恒为 0、xcorr 模板只剩噪点 ⇒ **配准峰锁到画面里最亮的云边**
（实测 ``014453`` 上 FAR 0.3608、hit@5px 36.04%）。

**更隐蔽的一半**：`cv2.TM_CCOEFF_NORMED` 在真匹配处的**符号取决于目标极性** ——
暗目标的权重图在目标处为正、目标帧灰度在那里为低，零均值化后两者**反号** ⇒
真匹配处的相关值是**负的**。若仍取极大，就会锁到最亮处。修法是**暗极性时把模板整体取反**。

**不变量**：
1. 暗目标 + 暗极性 ⇒ 配准位移 ≈ 0、相关峰 > 0.8；
2. 暗目标 + 亮极性 ⇒ **不得**配到旁边的亮云（这是 bug 的具体表现）；
3. ``PolarityTracker`` 在亮假设下与 ``LocalPatchTracker`` **逐位一致**（默认路径不得被扰动）；
4. ``estimate_polarity`` 对亮斑给 ``+1``、对暗斑给 ``-1``；
5. ``dark_ratio`` 置信门：亮斑在任何倍率下都判**亮**（否则会把亮目标误判成暗）。
"""
from __future__ import annotations

import unittest

import numpy as np

from manu.tracking.async_dual_rate.fast_tracker import LocalPatchTracker
from manu.tracking.lookahead import xcorr
from manu.tracking.lookahead.polarity import PolarityTracker, estimate_polarity

SIZE = 160
TARGET = np.asarray([100.0, 100.0], dtype=np.float64)
CLOUD = np.asarray([60.0, 60.0], dtype=np.float64)


def _scene(dark: bool) -> np.ndarray:
    """背景 100 + 目标（亮 170 / 暗 30）+ 干扰亮云 220 + 固定噪声。"""
    img = np.full((SIZE, SIZE), 100.0, dtype=np.float32)
    rng = np.random.default_rng(20261103)
    img += rng.normal(0.0, 1.5, img.shape).astype(np.float32)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    blob = ((xx - TARGET[0]) ** 2 + (yy - TARGET[1]) ** 2) <= 36
    img[blob] = 30.0 if dark else 170.0
    cloud = ((xx - CLOUD[0]) ** 2 + (yy - CLOUD[1]) ** 2) <= 64
    img[cloud] = 220.0
    return img


class TestPolarityXcorr(unittest.TestCase):
    def test_dark_target_matches_with_dark_polarity(self):
        img = _scene(dark=True)
        template, origin = xcorr.make_template(
            img, TARGET, tsize=32, bg_size=96, th_k=2.0, polarity=-1
        )
        self.assertIsNotNone(template)
        displacement, peak = xcorr.match(
            img, template, TARGET, TARGET, radius=64, tsize=32, origin=origin, polarity=-1
        )
        self.assertIsNotNone(displacement)
        self.assertLess(float(np.linalg.norm(displacement)), 0.5, f"位移 {displacement} 过大")
        self.assertGreater(peak, 0.8, f"暗目标真匹配的峰值只有 {peak}")

    def test_wrong_polarity_never_finds_the_target(self):
        """极性猜错 ⇒ 模板几乎为空 ⇒ 配准峰落到**别的**东西上（实测 014453 偏 70~100 px）。

        四格对称性（同一张合成图）：

        ==========  =========  ==================  ==============
        目标        极性        |位移 - 目标|         相关峰
        ==========  =========  ==================  ==============
        暗          暗 ✓        ≈ 0 px              ≈ 1.0
        暗          亮 ✗        ≫ 20 px             < 0.5
        亮          亮 ✓        ≈ 0 px              ≈ 1.0
        亮          暗 ✗        ≫ 20 px             < 0.5
        ==========  =========  ==================  ==============
        """
        for dark in (True, False):
            img = _scene(dark=dark)
            for polarity in (1, -1):
                correct = (polarity < 0) == dark
                template, origin = xcorr.make_template(
                    img, TARGET, tsize=32, bg_size=96, th_k=2.0, polarity=polarity
                )
                self.assertIsNotNone(template)
                displacement, peak = xcorr.match(
                    img, template, TARGET, TARGET, radius=64, tsize=32,
                    origin=origin, polarity=polarity,
                )
                self.assertIsNotNone(displacement)
                away = float(np.linalg.norm(displacement))
                label = f"目标={'暗' if dark else '亮'} 极性={polarity:+d}"
                if correct:
                    self.assertLess(away, 0.5, f"{label}：正确极性必须配到目标上，实得 {away:.2f} px")
                    self.assertGreater(peak, 0.8, f"{label}：正确极性的真匹配峰值只有 {peak}")
                else:
                    self.assertGreater(away, 20.0, f"{label}：错误极性竟配到了目标附近（{away:.2f} px）")
                    self.assertLess(peak, 0.5, f"{label}：错误极性的峰值不该有 {peak}")

    def test_bright_target_unchanged_by_polarity_argument(self):
        img = _scene(dark=False)
        a, origin_a = xcorr.make_template(img, TARGET, tsize=32, bg_size=96, th_k=2.0)
        b, origin_b = xcorr.make_template(
            img, TARGET, tsize=32, bg_size=96, th_k=2.0, polarity=1
        )
        np.testing.assert_allclose(a, b)
        self.assertEqual(origin_a, origin_b)
        disp_a, peak_a = xcorr.match(img, a, TARGET, TARGET, radius=64, tsize=32, origin=origin_a)
        disp_b, peak_b = xcorr.match(
            img, b, TARGET, TARGET, radius=64, tsize=32, origin=origin_b, polarity=1
        )
        np.testing.assert_allclose(disp_a, disp_b)
        self.assertAlmostEqual(peak_a, peak_b, places=12)


class TestPolarityTracker(unittest.TestCase):
    def test_bright_mode_is_identical_to_the_plain_fast_tracker(self):
        kwargs = dict(win_size=32, th_k=2.5, min_energy=1.0, min_contrast=5.0, bg_window=64)
        plain = LocalPatchTracker(**kwargs)
        wrapped = PolarityTracker(polarity="bright", **kwargs)
        img = _scene(dark=False)
        base = plain.step(img, TARGET)
        got = wrapped.step(img, TARGET)
        np.testing.assert_array_equal(base.xy, got.xy)
        self.assertEqual(base.valid, got.valid)
        self.assertEqual(base.energy, got.energy)
        self.assertEqual(base.snr, got.snr)
        self.assertEqual(base.peak_xy, got.peak_xy)

    def test_estimate_polarity_distinguishes_bright_and_dark(self):
        tracker = PolarityTracker(
            polarity="auto", win_size=52, th_k=2.5, min_energy=50.0,
            min_contrast=5.0, bg_window=104, dark_ratio=10.0,
        )
        self.assertEqual(estimate_polarity(_scene(dark=False), TARGET, tracker), 1)
        self.assertEqual(estimate_polarity(_scene(dark=True), TARGET, tracker), -1)

    def test_dark_ratio_gate_keeps_bright_targets_bright(self):
        """置信门必须挡住「亮目标上偶发误判成暗」—— 实测 001810 上 68 条误判链。"""
        tracker = PolarityTracker(
            polarity="auto", win_size=52, th_k=2.5, min_energy=50.0,
            min_contrast=5.0, bg_window=104, dark_ratio=10.0,
        )
        self.assertEqual(tracker.lock_from(_scene(dark=False), TARGET), 1)
        self.assertEqual(tracker.lock_from(_scene(dark=True), TARGET), -1)

    def test_dark_tracker_finds_the_dark_target(self):
        tracker = PolarityTracker(
            polarity="dark", win_size=52, th_k=2.5, min_energy=50.0,
            min_contrast=5.0, bg_window=104,
        )
        result = tracker.step(_scene(dark=True), TARGET)
        self.assertTrue(result.valid, "暗极性下必须看得见暗目标")
        self.assertLess(float(np.linalg.norm(result.xy - TARGET)), 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
