"""快流内核的契约测试。

这些断言是"Python 版作为 ref"的地基：本地质心必须**解析地**正确，
而不是"看起来差不多"。
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from manu.tracking.async_dual_rate.fast_tracker import LocalPatchTracker


def make_blob(size=(512, 640), center=(320.0, 256.0), sigma=2.0, amplitude=120.0, background=30.0):
    """造一张带单个高斯亮斑的灰度图（目标形状接近红外弱小目标）。"""
    height, width = size
    yy, xx = np.mgrid[0:height, 0:width]
    blob = amplitude * np.exp(
        -(((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2.0 * sigma * sigma))
    )
    frame = background + blob
    return np.clip(frame, 0, 255).astype(np.uint8)


class FastTrackerContractTest(unittest.TestCase):
    def test_recovers_subpixel_center(self):
        """质心必须落在亚像素真值上（容差 0.05 px）。"""
        for true_center in [(320.0, 256.0), (200.25, 300.75), (411.5, 128.25)]:
            frame = make_blob(center=true_center)
            tracker = LocalPatchTracker(win_size=32, th_k=3.0)
            result = tracker.step(frame, true_center)
            self.assertTrue(result.valid, f"signal lost at {true_center}")
            self.assertAlmostEqual(result.xy[0], true_center[0], delta=0.05)
            self.assertAlmostEqual(result.xy[1], true_center[1], delta=0.05)

    def test_tracks_a_moving_blob(self):
        """目标平移后仍应跟住（窗口覆盖范围内）。"""
        tracker = LocalPatchTracker(win_size=32, th_k=3.0)
        position = np.array([320.0, 256.0])
        for step in range(1, 6):
            true_center = (320.0 + step * 2.0, 256.0 + step * 1.0)
            frame = make_blob(center=true_center)
            result = tracker.step(frame, position)
            self.assertTrue(result.valid)
            self.assertAlmostEqual(result.xy[0], true_center[0], delta=0.2)
            self.assertAlmostEqual(result.xy[1], true_center[1], delta=0.2)
            position = result.xy

    def test_uniform_patch_is_weak(self):
        """没有目标 ⇒ 必须报 weak（交给外推逻辑），而不是编一个位置。"""
        frame = np.full((512, 640), 40, dtype=np.uint8)
        tracker = LocalPatchTracker(win_size=32, th_k=3.0)
        result = tracker.step(frame, (320.0, 256.0))
        self.assertFalse(result.valid)
        self.assertEqual(result.xy[0], 320.0)
        self.assertEqual(result.xy[1], 256.0)

    def test_position_kept_when_target_leaves_frame(self):
        """目标跑出画面 ⇒ 报 weak，位置保持最后一次的。"""
        # 目标在画面左下角，搜索位置在右上角 ⇒ ROI 里没有目标
        frame = make_blob(center=(5.0, 5.0))
        tracker = LocalPatchTracker(win_size=32, th_k=3.0)
        result = tracker.step(frame, (600.0, 480.0))
        self.assertFalse(result.valid)
        self.assertEqual(result.xy[0], 600.0)
        self.assertEqual(result.xy[1], 480.0)

    def test_border_roi_does_not_bias_centroid(self):
        """目标贴边时 ROI 平移（不是裁剪），质心不应被坐标系带偏。"""
        frame = make_blob(center=(1.5, 1.5), sigma=2.0)
        tracker = LocalPatchTracker(win_size=32, th_k=3.0)
        result = tracker.step(frame, (1.5, 1.5))
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.xy[0], 1.5, delta=0.6)
        self.assertAlmostEqual(result.xy[1], 1.5, delta=0.6)

    def test_peak_centroid_rejects_distant_clutter(self):
        """ROI 里有远处杂波时，peak_centroid 不应被拖走（整块质心会）。"""
        frame = make_blob(center=(320.0, 256.0), amplitude=120.0)
        frame = frame.astype(np.float32)
        yy, xx = np.mgrid[0:512, 0:640]
        clutter = 100.0 * np.exp(-(((xx - 344.0) ** 2 + (yy - 240.0) ** 2) / (2 * 1.5**2)))
        frame = np.clip(frame + clutter, 0, 255).astype(np.uint8)

        centroid = LocalPatchTracker(win_size=32, th_k=3.0, mode="centroid").step(frame, (320.0, 256.0))
        peak = LocalPatchTracker(win_size=32, th_k=3.0, mode="peak_centroid").step(frame, (320.0, 256.0))

        error_peak = float(np.linalg.norm(peak.xy - np.array([320.0, 256.0])))
        error_centroid = float(np.linalg.norm(centroid.xy - np.array([320.0, 256.0])))
        self.assertLess(error_peak, 2.0, f"peak_centroid error {error_peak}")
        self.assertLess(error_peak, error_centroid, "peak_centroid should beat the whole-patch centroid")

    def test_budget_is_under_one_millisecond(self):
        """文档判据 3：快流单帧必须 < 1 ms。"""
        frame = make_blob(center=(320.0, 256.0))
        tracker = LocalPatchTracker(win_size=32, th_k=3.0)
        tracker.step(frame, (320.0, 256.0))  # 预热（首次调用含 ramp 缓存建立）

        start = time.perf_counter()
        iterations = 2000
        for _ in range(iterations):
            tracker.step(frame, (320.0, 256.0))
        elapsed_ms = (time.perf_counter() - start) / iterations * 1000.0
        self.assertLess(elapsed_ms, 1.0, f"fast tracker costs {elapsed_ms:.4f} ms/frame")


if __name__ == "__main__":
    unittest.main()
