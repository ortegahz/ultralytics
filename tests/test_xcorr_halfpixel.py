"""半像素约定不变量的回归防线（2026-11-03 真实 bug 的教训）。

**Bug 本体**：`xcorr.match()` 曾把「模板的离散中心」当成「源帧 anchor 的物理点」上报。
模板从 ``round(cx) - tsize//2`` 切出（tsize 偶数）⇒ 离散中心对应源帧 ``round(cx) - 0.5``；
单次匹配系统性少 0.5 px（x、y 各），**8 步链式累计 −4 px** ⇒ chain 档位 hit@5px 只有 13.6%
（修复后 83%+）。direct 档位只差一次，看不出问题。

**不变量**：同一张图里把目标平移 ``(dx, dy)`` 后，``make_template`` + ``match`` 报出的位移
必须 ≈ ``(dx, dy)``；链式 8 步的累积位移必须 ≈ ``8 * step``，与 tsize 奇偶无关。
"""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from manu.tracking.lookahead import xcorr

WIDTH = HEIGHT = 256
ANCHOR = np.asarray([128.0, 128.0], dtype=np.float64)


def _base_blob(radius: float = 6.0, peak: float = 200.0, background: float = 30.0) -> np.ndarray:
    """背景 + 高斯亮斑 + 固定随机噪声（噪声必须随目标一起平移，否则相关面会退化）。"""
    ys, xs = np.ogrid[:HEIGHT, :WIDTH]
    blob = np.exp(-(((xs - ANCHOR[0]) ** 2 + (ys - ANCHOR[1]) ** 2) / (2.0 * radius ** 2)))
    img = background + peak * blob
    rng = np.random.default_rng(20261103)
    img = img + rng.normal(0.0, 1.5, img.shape)
    return img.astype(np.float32)


def _translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    return cv2.warpAffine(
        img, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


class XcorrHalfPixelInvariantTest(unittest.TestCase):
    """``match`` 必须上报「anchor 物理点的位移」，而不是「模板离散中心的落点」。"""

    def test_single_match_reports_true_displacement(self):
        for dx, dy in ((7.0, -4.0), (-6.0, 5.0), (3.0, 2.0)):
            with self.subTest(dx=dx, dy=dy):
                src = _base_blob()
                dst = _translate(src, dx, dy)
                template, origin = xcorr.make_template(src, ANCHOR, tsize=24, bg_size=96, th_k=2.5)
                self.assertIsNotNone(template, "合成图上必须能切出非空模板")
                self.assertIsNotNone(origin)
                displacement, peak = xcorr.match(
                    dst, template, ANCHOR, ANCHOR + np.asarray([dx, dy]),
                    radius=16, tsize=24, origin=origin,
                )
                self.assertIsNotNone(displacement)
                self.assertGreater(peak, 0.5)
                self.assertAlmostEqual(displacement[0], dx, delta=0.05)
                self.assertAlmostEqual(displacement[1], dy, delta=0.05)

    def test_eight_step_chain_does_not_accumulate_half_pixel_drift(self):
        """8 步链式的累积位移必须 ≈ 8 * step；每步少 0.5 px 会累计成 −4 px。"""
        src = _base_blob()
        step = np.asarray([1.5, -0.75], dtype=np.float64)
        frames = [_translate(src, step[0] * k, step[1] * k) for k in range(9)]

        position = ANCHOR.copy()
        for k in range(1, 9):
            matched, peak = xcorr.chain_step(
                frames[k], frames[k - 1], position, position + step,
                radius=12, tsize=24, bg_size=96, th_k=2.5, min_peak=0.25,
            )
            self.assertIsNotNone(matched, f"第 {k} 步必须配准成功")
            position = np.asarray(matched, dtype=np.float64)

        truth = ANCHOR + step * 8.0
        drift = float(np.linalg.norm(position - truth))
        self.assertLess(drift, 0.2, f"8 步链式累积漂移 {drift:.3f} px 过大（半像素 bug 会是 ~4 px）")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
