"""慢流（Slow Detector）：检测器输出的回放与异步调度。

本原型**不重跑检测网络**，而是回放 RC2（``trial_0080/weights/epoch3.pt``）
已经算好的多尺度热图峰值缓存 ``ms_cache/<SEQ>.pkl``。这样做的好处：

* 检测输出是**真实的**（同一份权重、同一套阈值几何），不是伪造的框；
* 离线可复现、零 GPU、秒级迭代；
* 后续把 ``ReplayDetector`` 换成真检测器（服务器上带 GPU 跑），上层
  ``AsyncDualRateTracker`` 一行都不用改——这正是"检测器只作为后台真理源"
  的架构要求。

缓存结构（``ms_cache/*.pkl``）::

    meta:          {sequence, frames, native_size, imgsz, stride, scales, ...}
    frame_index:   (N,) int32        每帧的物理帧号
    offset:        (N+1,) int64      CSR 式：第 i 帧占 points[offset[i]:offset[i+1]]
    points:        (M,2) float32     峰值像素坐标（native_pixel 空间）
    sizes:         (M,2) float32     框尺寸 (w, h)
    scores:        (M,) float16      峰值分数
    sums/area/tag: (M,)              连通域统计量

异步调度：检测器按 ``det_hz`` 采样帧（相机 25 Hz、检测 12.5 Hz ⇒ 每 2 帧一次），
推理耗时折算成 ``latency_frames``（Δ）的滞后，于是"针对第 f 帧的结果"
在第 ``f + Δ`` 帧才到达。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Detection", "ReplayDetector", "DetectorScheduler"]


@dataclass(frozen=True)
class Detection:
    """一次检测输出。"""

    frame_index: int
    """该检测对应的**源帧号** f（不是到达时刻）。"""

    xy: np.ndarray
    """峰值像素坐标 ``(x, y)``。"""

    score: float
    wh: tuple[float, float] = (0.0, 0.0)
    """框尺寸 ``(w, h)``（像素）。"""


class ReplayDetector:
    """从 ``ms_cache/*.pkl`` 回放检测器的逐帧输出。"""

    def __init__(
        self,
        cache_path,
        min_score: float = 0.0,
        tags: tuple[int, ...] | None = None,
    ) -> None:
        path = Path(cache_path)
        with path.open("rb") as handle:
            cache = pickle.load(handle)

        self.meta = cache["meta"]
        self.frames = int(self.meta["frames"])
        self.width = int(self.meta["native_size"][0])
        self.height = int(self.meta["native_size"][1])

        frame_index = np.asarray(cache["frame_index"], dtype=np.int64)
        offset = np.asarray(cache["offset"], dtype=np.int64)
        points = np.asarray(cache["points"], dtype=np.float32)
        sizes = np.asarray(cache["sizes"], dtype=np.float32)
        scores = np.asarray(cache["scores"], dtype=np.float32)
        tags_array = np.asarray(cache["tag"], dtype=np.int32)

        keep = scores >= min_score
        if tags is not None:
            keep &= np.isin(tags_array, np.asarray(tags, dtype=np.int32))
        self._keep = keep
        self._points = points
        self._sizes = sizes
        self._scores = scores
        self._tags = tags_array
        self._frame_index = frame_index
        self._offset = offset

        # frame_index -> 在 points 里的行号区间（假设 frame_index 连续 0..N-1）
        self._by_frame: dict[int, tuple[int, int]] = {}
        for position, physical in enumerate(frame_index):
            self._by_frame[int(physical)] = (
                int(offset[position]),
                int(offset[position + 1]),
            )

    def detections(self, frame_index: int) -> list[Detection]:
        """返回该帧的全部检测，按分数从高到低。"""
        span = self._by_frame.get(int(frame_index))
        if span is None:
            return []
        start, stop = span
        if stop <= start:
            return []
        window = slice(start, stop)
        scores = self._scores[window]
        order = np.argsort(-scores, kind="stable")
        out: list[Detection] = []
        for row in order:
            if not self._keep[start + row]:
                continue
            out.append(
                Detection(
                    frame_index=int(frame_index),
                    xy=self._points[start + row].astype(np.float64).copy(),
                    score=float(self._scores[start + row]),
                    wh=(
                        float(self._sizes[start + row, 0]),
                        float(self._sizes[start + row, 1]),
                    ),
                )
            )
        return out

    def best(self, frame_index: int) -> Detection | None:
        """该帧分数最高的检测。"""
        found = self.detections(frame_index)
        return found[0] if found else None


class DetectorScheduler:
    """把慢流检测器按"低频 + 固定滞后"调度成异步事件流。

    Parameters
    ----------
    detector:
        任意提供 ``detections(frame_index) -> list[Detection]`` 的对象
        （``ReplayDetector`` 或真检测器）。
    camera_hz:
        相机帧率（快流时钟）。
    det_hz:
        检测器帧率（慢流时钟）。``det_hz >= camera_hz`` 时退化为每帧都检测。
    latency_frames:
        Δ：检测结果滞后多少帧才到达。
    """

    def __init__(
        self,
        detector,
        camera_hz: float = 25.0,
        det_hz: float = 12.5,
        latency_frames: int = 8,
    ) -> None:
        if det_hz <= 0 or camera_hz <= 0:
            raise ValueError("rates must be positive")
        self.detector = detector
        self.camera_hz = float(camera_hz)
        self.det_hz = float(det_hz)
        self.latency_frames = int(latency_frames)
        self.stride = max(1, int(round(self.camera_hz / self.det_hz)))
        self._issued: dict[int, list[Detection]] = {}

    def detector_frames(self) -> list[int]:
        """慢流实际会推理的帧号（算术级数）。"""
        return list(range(0, self._frames_total(), self.stride))

    def _frames_total(self) -> int:
        frames = getattr(self.detector, "frames", None)
        return int(frames) if frames else 0

    def poll(self, frame_index: int) -> list[tuple[int, list[Detection]]]:
        """在第 ``frame_index`` 帧，取出**此刻才到达**的检测结果。

        Returns
        -------
        list[(source_frame_index, detections)]
            每个元素是一批同源检测（通常 1 个框；多目标序列会有多个）。
        """
        arrivals: list[tuple[int, list[Detection]]] = []
        source = int(frame_index) - self.latency_frames
        if source < 0:
            return arrivals
        if source % self.stride != 0:
            return arrivals
        if source not in self._issued:
            self._issued[source] = self.detector.detections(source)
        found = self._issued[source]
        if found:
            arrivals.append((source, found))
        return arrivals
