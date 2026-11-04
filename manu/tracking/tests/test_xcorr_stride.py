"""分段配准（stride 档位）的回归防线。

**为什么要有这一档**：``direct`` 的一次跨 Δ 帧大匹配里，模板与目标的外观已漂移
Δ 帧 ⇒ 相关峰不准（离线大比试实测位移 p95 ~48.9 px、hit@5px 91.2%）；分段后每段
只隔 1~2 帧、模板每段刷新 ⇒ 峰很锐（实测 p95 ~2.75 px、hit@5px 99.2%）。

**不变量**：同一张图里把目标平移 ``(dx, dy)`` 后，分段配准恢复出的位移必须
≈ ``(dx, dy)``，且与段长无关（``stride=8`` 一次跨 8 帧、``stride=1`` 逐帧都要成立）。
"""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from manu.tracking.lookahead import extrapolator
from manu.tracking.lookahead.extrapolator import LookaheadConfig, _xcorr_stride_refine

WIDTH = HEIGHT = 128
ANCHOR = np.asarray([64.0, 64.0], dtype=np.float64)
DISPLACEMENT = np.asarray([7.0, -4.0], dtype=np.float64)
DELTA = 8


class _FrameSource:
    """最小帧源：按帧号返回对应的合成帧。"""

    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self._frames = frames

    def get(self, frame: int) -> np.ndarray | None:
        return self._frames.get(int(frame))


def _base_blob(radius: float = 3.0, peak: float = 200.0, background: float = 30.0) -> np.ndarray:
    """背景 + 高斯亮斑 + 固定随机噪声（噪声必须随目标一起平移，否则相关面会退化）。"""
    ys, xs = np.ogrid[:HEIGHT, :WIDTH]
    blob = np.exp(-(((xs - ANCHOR[0]) ** 2 + (ys - ANCHOR[1]) ** 2) / (2.0 * radius ** 2)))
    img = background + peak * blob
    rng = np.random.default_rng(20261104)
    img = img + rng.normal(0.0, 1.5, img.shape)
    return img.astype(np.float32)


def _translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    return cv2.warpAffine(
        img, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _frame_source() -> _FrameSource:
    """第 k 帧的目标平移 ``k/Δ · 位移`` ⇒ 整段位移恰好是 ``DISPLACEMENT``。"""
    base = _base_blob()
    frames = {
        k: _translate(base, DISPLACEMENT[0] * k / DELTA, DISPLACEMENT[1] * k / DELTA)
        for k in range(DELTA + 1)
    }
    return _FrameSource(frames)


def _cfg(stride: int) -> LookaheadConfig:
    """只开 xcorr 档位的最小配置（其余保持默认）。"""
    return LookaheadConfig(
        replay_xcorr_radius=16.0,
        replay_xcorr_template=24,
        replay_xcorr_th_k=2.5,
        replay_xcorr_min_peak=0.25,
        replay_xcorr_bg=96,
        replay_xcorr_stride=int(stride),
    )


class XcorrStrideTest(unittest.TestCase):
    """分段配准必须恢复出真实位移，且与段长无关。"""

    def test_recovers_displacement_for_any_stride(self):
        truth = ANCHOR + DISPLACEMENT
        for stride in (DELTA, 1, 2):
            with self.subTest(stride=stride):
                result: dict = {}
                refined = _xcorr_stride_refine(
                    _frame_source(),
                    0,
                    DELTA,
                    ANCHOR,
                    truth,
                    _cfg(stride),
                    result,
                )
                self.assertIsNotNone(refined, f"stride={stride} 必须配准成功")
                refined = np.asarray(refined, dtype=np.float64).reshape(2)
                error = float(np.linalg.norm(refined - truth))
                self.assertLess(error, 0.5, f"stride={stride} 位移误差 {error:.3f} px 过大")
                self.assertEqual(int(result.get("xcorr_applied", 0)), 1)
                self.assertEqual(int(result.get("xcorr_stride_hold", 0)), 0)

    def test_first_segment_failure_returns_none(self):
        """第一段就配不上 ⇒ 没有可 hold 的位移 ⇒ 整个精修作废（调用方走既有兜底）。"""
        # 把源帧换成空图 ⇒ 切不出模板 ⇒ 第一段失败。
        empty = _FrameSource({k: np.zeros((HEIGHT, WIDTH), dtype=np.float32) for k in range(DELTA + 1)})
        result: dict = {}
        refined = _xcorr_stride_refine(
            empty,
            0,
            DELTA,
            ANCHOR,
            ANCHOR + DISPLACEMENT,
            _cfg(DELTA),
            result,
        )
        self.assertIsNone(refined)
        self.assertEqual(int(result.get("xcorr_applied", 0)), 0)
        self.assertGreaterEqual(int(result.get("xcorr_stride_hold", 0)), 1)

    def test_config_default_keeps_stride_off(self):
        """默认必须全关（0 = 关），否则历史档位的行为会被改掉。"""
        self.assertEqual(int(LookaheadConfig().replay_xcorr_stride), 0)

    def test_stride_mode_passes_mode_validation(self):
        """``--replay-xcorr-mode stride`` 必须过取值校验；非法取值仍要报错。"""
        cfg = LookaheadConfig(replay_xcorr_mode="stride", replay_xcorr_stride=2)
        outcome = extrapolator._replay_detections([], 0, DELTA, cfg, {}, None, None, [])
        self.assertEqual(outcome["entries"], [])

        with self.assertRaises(ValueError) as ctx:
            extrapolator._replay_detections(
                [], 0, DELTA, LookaheadConfig(replay_xcorr_mode="bogus"), {}, None, None, []
            )
        self.assertIn("unknown replay_xcorr_mode", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
