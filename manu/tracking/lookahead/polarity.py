"""目标极性自适应：亮目标 / 暗目标。

红外场景下目标相对背景的极性**不是恒定的**。龙泉山实测（2026-11-03，用 GT
中心 16×16 窗与腐蚀背景环比中值）：``VIDEO00032_19700101_014453`` 的对比度是
**−12.91**、``VIDEO00033_19700101_015000`` 是 **−22.56**（两者 100% 的帧都
「更暗」）；其余 6 条序列是 +12.46 ~ +76.81（0% 更暗）。也就是说，**8 序列的
官方 gate 集里就有 1 条是暗目标**（015000）。

快核 :class:`~manu.tracking.async_dual_rate.fast_tracker.LocalPatchTracker` 与
xcorr 模板都**写死了「比背景亮」**：

* 亮目标：门限 ``bg + kσ``、权重 ``max(patch - 门限, 0)``、峰取 ``argmax``；
* 暗目标：门限 ``bg - kσ``、权重 ``max(门限 - patch, 0)``、峰取 ``argmin``。

暗目标上继续用亮假设 ⇒ 权重**几乎恒为 0** ⇒ 质心退化成窗心、xcorr 模板里只剩
几颗噪点 ⇒ 配准峰锁到画面里**最亮的云边**（合成图验证：位移 (−22.9, +20.0)；
实测 014453 上 replay 偏到 70~100 px 外）。

本模块提供两件事：

1. :meth:`PolarityTracker.lock_from` —— 在**锚点帧**（位置已知 = 检测框位置，
   误差 ~2 px）现场判极性，并**锁定整条链**（ROI 一旦漂走，局部极性判断会翻，
   锁定可避免级联跑飞）；
2. :class:`PolarityTracker` —— 快核的极性镜像实现。

**默认行为逐位不变**：``--replay-polarity bright`` 时走的是原来的
:class:`LocalPatchTracker`（不是本类），代码路径一行未改。
"""
from __future__ import annotations

import numpy as np

from manu.tracking.async_dual_rate.fast_tracker import LocalPatchTracker, TrackerResult, _ramps

__all__ = ["estimate_polarity", "PolarityTracker"]


def _core_block(patch: np.ndarray) -> np.ndarray:
    """取 ROI 正中央的一块（奇数边长），用于判极性。

    目标在锚点帧就位于 ROI 正中（锚点 = 检测框中心，误差 ~2 px），所以中央块
    里必然压着目标本体；而 ROI 四周是背景。判极性只需要"中央块 vs 背景"的对比。
    """
    height, width = patch.shape
    size = max(5, (min(height, width) // 4) | 1)
    size = min(size, min(height, width) | 1)
    top = (height - size) // 2
    left = (width - size) // 2
    return patch[top : top + size, left : left + size]


def polarity_masses(frame: np.ndarray, xy, tracker: LocalPatchTracker) -> tuple[float, float]:
    """返回 ``(亮向越界能量, 暗向越界能量)``（均按面积归一化）。

    判据：在 ROI 正中央的块上，比较**亮向**与**暗向**的"越界能量"：

    * ``bright_mass = mean(max(patch - (bg + kσ), 0))`` —— 有多少像素明显比背景亮；
    * ``dark_mass  = mean(max((bg - kσ) - patch, 0))`` —— 有多少像素明显比背景暗。

    亮目标 ⇒ ``bright_mass`` 远大于 ``dark_mass``；暗目标 ⇒ 反之。
    """
    cx, cy = float(xy[0]), float(xy[1])
    patch, _ = tracker._window(frame, cx, cy, tracker.win_size)
    if patch is None:
        return 0.0, 0.0
    background_info = tracker._background(frame, cx, cy)
    if background_info is None:
        return 0.0, 0.0
    background, sigma = background_info
    core = _core_block(patch.astype(np.float32))
    margin = float(tracker.th_k) * sigma
    bright_mass = float(np.maximum(core - (background + margin), 0.0).sum()) / core.size
    dark_mass = float(np.maximum((background - margin) - core, 0.0).sum()) / core.size
    return bright_mass, dark_mass


def estimate_polarity(frame: np.ndarray, xy, tracker: LocalPatchTracker) -> int:
    """返回 ``+1``（目标比背景亮）或 ``-1``（目标比背景暗）。

    两者相等（平坦区 / 目标不可见）时**返回 +1**（历史行为），因为此时极性选谁都
    不影响结果，保持旧行为最安全。
    """
    bright_mass, dark_mass = polarity_masses(frame, xy, tracker)
    return 1 if bright_mass >= dark_mass else -1


class PolarityTracker(LocalPatchTracker):
    """按目标极性镜像快核实现的局部质心核。

    Parameters
    ----------
    polarity:
        ``"bright"``（默认，等价于原 :class:`LocalPatchTracker`）、``"dark"``、
        ``"auto"``。``"auto"`` 在 :meth:`lock_from` 时现场判极性。

    亮假设下 :meth:`step` 直接调用父类实现（**同一份代码、逐字节一致**）；
    暗假设下是它的镜像：门限 ``bg - kσ``、权重 ``max(门限 - patch, 0)``、
    峰取 ``argmin``、对比度取 ``bg - 峰``。
    """

    def __init__(self, polarity: str = "bright", dark_ratio: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        mode = str(polarity).strip().lower()
        if mode not in ("bright", "dark", "auto"):
            raise ValueError(f"unknown polarity: {polarity!r}")
        self.polarity_mode = mode
        self.polarity = -1 if mode == "dark" else 1
        """当前生效的极性（``+1`` 亮 / ``-1`` 暗）。"""
        self.dark_ratio = float(dark_ratio)
        """``auto`` 模式下判暗所需的**置信倍数**：只有 ``暗向能量 > dark_ratio ×
        亮向能量`` 才判暗。默认 1.0 = 「暗向严格大于亮向」（旧行为）。实测
        真暗目标的倍数是 90~840、真亮目标 ≤ 0.08 ⇒ 提高它可以让判定**稳定**。"""
        self.last_ratio = 0.0
        """最近一次判定的 ``暗向/亮向`` 能量倍数（诊断用）。"""

    # -- 极性判定 -----------------------------------------------------------

    def lock_from(self, frame: np.ndarray, xy) -> int:
        """在锚点帧判极性并锁定；返回锁定后的极性。

        ``"auto"`` 之外的两个模式直接返回既定极性（不消耗额外计算）。
        """
        if self.polarity_mode == "auto":
            bright_mass, dark_mass = polarity_masses(frame, xy, self)
            self.last_ratio = float(dark_mass / bright_mass) if bright_mass > 0.0 else float("inf")
            self.polarity = -1 if dark_mass > self.dark_ratio * bright_mass else 1
        return int(self.polarity)

    # -- 主接口 -------------------------------------------------------------

    def step(self, frame: np.ndarray, center_xy) -> TrackerResult:
        """与父类同签名；亮假设下**原样**调用父类。"""
        if self.polarity > 0:
            return super().step(frame, center_xy)
        return self._step_dark(frame, center_xy)

    def _step_dark(self, frame: np.ndarray, center_xy) -> TrackerResult:
        """暗目标版快核：与 :meth:`LocalPatchTracker.step` 逐行镜像。"""
        cx, cy = float(center_xy[0]), float(center_xy[1])

        patch, (x1, y1) = self._window(frame, cx, cy, self.win_size)
        if patch is None:
            return TrackerResult(
                xy=np.asarray([cx, cy], dtype=np.float64),
                valid=False,
                energy=0.0,
                snr=0.0,
                peak_xy=None,
            )
        patch = patch.astype(np.float32)

        background_info = self._background(frame, cx, cy)
        if background_info is None:
            return TrackerResult(
                xy=np.asarray([cx, cy], dtype=np.float64),
                valid=False,
                energy=0.0,
                snr=0.0,
                peak_xy=None,
            )
        background, sigma = background_info
        threshold = background - self.th_k * sigma
        weights = np.maximum(threshold - patch, 0.0)
        energy = float(weights.sum())

        if energy <= self.min_energy:
            return TrackerResult(
                xy=np.asarray([cx, cy], dtype=np.float64),
                valid=False,
                energy=energy,
                snr=0.0,
                peak_xy=None,
            )

        peak_flat = int(np.argmin(patch))
        py, px = divmod(peak_flat, patch.shape[1])
        peak_value = float(patch[py, px])
        snr = (background - peak_value) / sigma
        contrast = background - peak_value
        if contrast < self.min_contrast:
            return TrackerResult(
                xy=np.asarray([cx, cy], dtype=np.float64),
                valid=False,
                energy=energy,
                snr=snr,
                peak_xy=(x1 + px + 0.5, y1 + py + 0.5),
            )

        if self.mode == "peak_centroid":
            hw = self.peak_window // 2
            sx1 = max(0, px - hw)
            sy1 = max(0, py - hw)
            sx2 = min(patch.shape[1], px + hw + 1)
            sy2 = min(patch.shape[0], py + hw + 1)
            sub = weights[sy1:sy2, sx1:sx2]
            sub_energy = float(sub.sum())
            if sub_energy <= 0.0:
                sub = weights
                sub_energy = energy
                sx1, sy1 = 0, 0
            xramp, yramp = _ramps(sub.shape[0], sub.shape[1])
            local_x = float(sub.sum(axis=0) @ xramp) / sub_energy
            local_y = float(sub.sum(axis=1) @ yramp) / sub_energy
            out_x = x1 + sx1 + local_x
            out_y = y1 + sy1 + local_y
        else:
            xramp, yramp = _ramps(patch.shape[0], patch.shape[1])
            out_x = x1 + float(weights.sum(axis=0) @ xramp) / energy
            out_y = y1 + float(weights.sum(axis=1) @ yramp) / energy

        return TrackerResult(
            xy=np.asarray([out_x, out_y], dtype=np.float64),
            valid=True,
            energy=energy,
            snr=snr,
            peak_xy=(x1 + px + 0.5, y1 + py + 0.5),
        )
