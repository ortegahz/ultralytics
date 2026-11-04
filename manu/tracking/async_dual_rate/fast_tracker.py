"""快流（Fast Tracker）：局部质心/峰值提取核。

这是异步双频系统的**前台关键路径**：相机每来一帧就必须跑一次，预算 **< 1 ms**，
所以这里只允许 NumPy 切片与极小的点积，不允许任何网络前向。

算法（按《异步双频检跟分离系统》文档 §二.1）：
  1. ROI 裁剪：在上一帧位置周围裁 ``win_size x win_size`` 的局部窗口；
  2. 局部背景抑制：用**更大的邻域窗**（``bg_window``）的中值估计低频底噪，
     用 MAD 估计噪声尺度；
  3. 亚像素质心提取：``x = sum(i * (I - th)+) / sum((I - th)+)``。

**为什么底噪要在"更大的窗"里估**（这是实测打脸文档原版的地方）：
文档写的是"在 32×32 ROI 内算中值与 MAD"。但在龙泉山实测序列上，
目标是**成簇的亮点**（本体 + 运动拖尾），在 32×32 ROI 内占比可达 60~80%
⇒ ROI 的中值被目标自身抬高、MAD 被目标自身撑大 ⇒ 门限 ``中值+3σ`` 直接
高过 ROI 内的峰值 ⇒ 快跟踪核把"正盯着目标"误判成"没有信号"，于是每
14 帧就掉一次（实测 TRACK→LOST 830 次 / 11810 帧）。

于是把**底噪估计与质心提取解耦**：质心仍在 32×32 的小窗里提（文档判据
"快核能否撑住 32×32 视野"不变），底噪改在 64×64 邻域窗里估——目标在
那个窗里只占 ~20%，中值/MAD 重新回到"底噪"的语义。


坐标约定：``(x, y)``，``x`` = 列（column），``y`` = 行（row），与项目里
GT 标签 ``cx cy w h``（像素、原点左上）的口径一致。

两种提取模式（用于受控 A/B，见 ``manu/tracking/async_dual_rate/evaluate.py``）：
  * ``"centroid"``      —— 文档原版：整块 ROI 的阈值化加权质心；
  * ``"peak_centroid"`` —— 在 ROI 内先取局部极大，再在极大点邻域内求质心。
                          红外弱小目标场景下 ROI 里常混有杂波，整块质心会被拖走，
                          故保留这个变体做同源受控对照，而不是直接替换文档原版。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TrackerResult", "LocalPatchTracker"]


# ROI 坐标斜坡（x 方向 / y 方向）按窗口尺寸缓存：同一序列里窗口尺寸几乎不变，
# 每次调用重新 np.indices 会把 0.008 ms 的内核开销抬高一倍以上。
_RAMP_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _ramps(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (x 斜坡, y 斜坡)，按 (h, w) 缓存。"""
    key = height * 4096 + width
    cached = _RAMP_CACHE.get(key)
    if cached is None:
        cached = (
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        _RAMP_CACHE[key] = cached
    return cached


@dataclass(frozen=True)
class TrackerResult:
    """快跟踪核的一次输出。"""

    xy: np.ndarray
    """亚像素目标位置 ``(x, y)``；即使 ``valid`` 为 False 也带一个（外推用的）位置。"""

    valid: bool
    """False = ROI 内能量低于门限（疑似脱靶 / SNR 骤降）。"""

    energy: float
    """阈值化后的前景能量和 ``sum((I - th)+)``。"""

    snr: float
    """``(ROI 峰值 - 背景) / 噪声尺度``。"""

    peak_xy: tuple[float, float] | None
    """ROI 内局部极大点的亚像素位置；``peak_centroid`` 模式下即质心锚点。"""


class LocalPatchTracker:
    """局部质心/峰值提取核。

    Parameters
    ----------
    win_size:
        ROI 边长（像素）。文档给的是 32。
    th_k:
        门限 = ``背景中值 + th_k * 噪声尺度``。越大越保守（只保留强峰）。
    min_energy:
        低于此能量判为"信号消失"，``valid=False``。
    min_contrast:
        局部极大点相对 ROI 背景的**最小灰度差**。能量门限只看"前景像素总量"，
        在接近均匀的 ROI 里会被少数噪点撑起来；再加一道绝对对比度门限，
        要求"确实有一个比背景亮一截的峰"，才认账。
    bg_window:
        底噪估计窗边长（像素）。质心提取窗仍是 ``win_size``，只是底噪改在
        更大的邻域里估（原因见模块文档）。设成 ``win_size`` 即退回文档原版。
    mode:
        ``"centroid"``（文档原版）或 ``"peak_centroid"``（邻域质心变体）。
    peak_window:
        ``peak_centroid`` 模式下围绕局部极大点取质心的邻域边长。
    """

    def __init__(
        self,
        win_size: int = 32,
        th_k: float = 3.0,
        min_energy: float = 1.0,
        min_contrast: float = 5.0,
        mode: str = "centroid",
        peak_window: int = 9,
        bg_window: int = 64,
    ) -> None:
        if mode not in ("centroid", "peak_centroid"):
            raise ValueError(f"unknown mode: {mode!r}")
        if win_size < 4 or win_size % 2 != 0:
            raise ValueError("win_size must be an even integer >= 4")
        self.win_size = int(win_size)
        self.th_k = float(th_k)
        self.min_energy = float(min_energy)
        self.min_contrast = float(min_contrast)
        self.mode = mode
        self.peak_window = int(peak_window)
        self.bg_window = max(int(bg_window), int(win_size))
        """底噪估计窗。必须 ≥ 提取窗，否则退回"在目标自己的窗口里估底噪"。"""
        self.bg_stride = max(1, self.bg_window // 32)
        """底噪窗的抽稀步长：64→2，只取 1/4 像素，中值/MAD 精度绰绰有余，
        但背景估计的开销不变（保护 1 ms 预算）。"""


    # -- 内部 ---------------------------------------------------------------

    def _window(self, frame: np.ndarray, cx: float, cy: float, size: int):
        """取以 ``(cx, cy)`` 为中心、边长 ``size`` 的窗。

        越界时把窗口**整体平移**回画面内，而不是裁剪——裁剪会让质心偏置
        （分母和分子的坐标系不一致），平移只改变覆盖范围，坐标仍是绝对的。

        Returns
        -------
        tuple[np.ndarray | None, tuple[int, int]]
            ``(窗口像素, 左上角坐标)``；完全出画时返回 ``(None, (0, 0))``。
        """
        height, width = frame.shape
        half = size // 2
        x1 = int(round(cx)) - half
        y1 = int(round(cy)) - half
        x2 = x1 + size
        y2 = y1 + size

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > width:
            x1 -= x2 - width
            x2 = width
        if y2 > height:
            y1 -= y2 - height
            y2 = height
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            return None, (0, 0)
        return frame[y1:y2, x1:x2], (x1, y1)

    def _background(self, frame: np.ndarray, cx: float, cy: float):
        """在**底噪窗**里估 (背景中值, 噪声尺度 sigma)。

        噪声尺度用 MAD 估计（``1.4826 * MAD`` 是正态分布下 sigma 的一致估计），
        比 std 更抗强目标污染。

        关键点：底噪窗比质心窗大（默认 64 vs 32）。龙泉山实测目标在 32×32
        里能占 60~80%，在 ROI 内估底噪会被目标自己抬高；放到大窗里，
        目标只占 ~20%，中值/MAD 才真的代表"底噪"。
        """
        patch, origin = self._window(frame, cx, cy, self.bg_window)
        if patch is None:
            return None
        sample = patch[:: self.bg_stride, :: self.bg_stride].astype(np.float32)
        background = float(np.median(sample))
        mad = float(np.median(np.abs(sample - background)))
        sigma = 1.4826 * mad
        # 完全均匀的窗（sigma == 0）会让门限退化成背景本身，
        # 于是任何 1 灰阶的抖动都变成"前景"。给一个与 8-bit 量化匹配的绝对下限。
        return background, max(sigma, 1e-3)

    # -- 主接口 -------------------------------------------------------------

    def step(self, frame: np.ndarray, center_xy) -> TrackerResult:
        """在 ``center_xy`` 周围做一次局部提取。

        Parameters
        ----------
        frame:
            2-D ``uint8`` 灰度图（相机原始帧）。
        center_xy:
            上一帧/外推得到的中心位置 ``(x, y)``。
        """
        cx, cy = float(center_xy[0]), float(center_xy[1])

        patch, (x1, y1) = self._window(frame, cx, cy, self.win_size)
        if patch is None:
            # 目标完全跑出画面
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
        threshold = background + self.th_k * sigma
        weights = np.maximum(patch - threshold, 0.0)
        energy = float(weights.sum())

        if energy <= self.min_energy:
            # 信号消失：位置保持不变交给外推逻辑，不在这里编造运动。
            return TrackerResult(
                xy=np.asarray([cx, cy], dtype=np.float64),
                valid=False,
                energy=energy,
                snr=0.0,
                peak_xy=None,
            )

        peak_flat = int(np.argmax(patch))
        py, px = divmod(peak_flat, patch.shape[1])
        peak_value = float(patch[py, px])
        snr = (peak_value - background) / sigma
        contrast = peak_value - background
        if contrast < self.min_contrast:
            # ROI 里没有比背景明显更亮的峰：这是平坦区/噪声，不是目标。
            return TrackerResult(
                xy=np.asarray([cx, cy], dtype=np.float64),
                valid=False,
                energy=energy,
                snr=snr,
                peak_xy=(x1 + px + 0.5, y1 + py + 0.5),
            )

        if self.mode == "peak_centroid":
            # 只在局部极大点周围的小邻域内求质心，避开远处杂波。
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
            # sub 是 (h, w)，必须先沿轴求和降成向量再做点积：
            # 直接 `sub @ xramp` 会得到长度 h 的向量而不是标量。
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
