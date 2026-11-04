"""历史环形缓冲（History Ring Buffer）。

慢流检测器有 Δ 帧（约 300 ms）的处理延迟，所以**不能**拿"现在的跟踪位置"去和
"Δ 帧前的检测框"比较——必须能取回 **同一时刻** 的历史状态做对账。

缓冲按**绝对帧号**索引（不是列表移位），容量固定，满了覆盖最老的。
这与项目铁律"环形缓冲按绝对帧号 mod depth 索引，禁列表移位"一致。

除了位置，还存每个样本的**速度**（由相邻两帧位置差得到），因为融合的
"硬拉回"分支需要用历史速度把 Δ 帧前的检测位置投影回当前时刻。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["HistoryRingBuffer"]


class HistoryRingBuffer:
    """定容历史池：``frame_index -> (xy, velocity)``。

    Parameters
    ----------
    capacity:
        保留的历史帧数。必须 **> Δ**（延迟帧数），否则取不到对账所需的时刻。
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self.capacity = int(capacity)
        self._frames = np.full(self.capacity, -1, dtype=np.int64)
        self._xy = np.zeros((self.capacity, 2), dtype=np.float64)
        self._vel = np.zeros((self.capacity, 2), dtype=np.float64)
        self._time = np.full(self.capacity, np.nan, dtype=np.float64)
        self._count = 0
        self._cursor = 0
        self._last_index: int | None = None
        self._last_xy = np.zeros(2, dtype=np.float64)

    def __len__(self) -> int:
        return self._count

    def push(self, frame_index: int, xy, timestamp: float | None = None) -> None:
        """写入一帧的位置，并顺手更新该帧的速度（相对**上一帧**）。

        速度是"这一帧相对上一帧的位移"，所以必须拿**最近一次 push 的位置**做差，
        而不是拿环里的上一个槽位——后者在缓冲未满时是空的、满了之后又会被覆盖。
        """
        slot = self._cursor
        position = np.asarray(xy, dtype=np.float64).reshape(2)

        self._frames[slot] = int(frame_index)
        self._xy[slot] = position
        if timestamp is None:
            # 没给时间戳时用帧号当时间代理（定帧率下等价），但**显式标 NaN**
            # 以便调用方分辨"真时间戳"与"帧号代理"。
            self._time[slot] = float("nan")
        else:
            self._time[slot] = float(timestamp)

        if self._last_index is not None and self._last_index == int(frame_index) - 1:
            self._vel[slot] = position - self._last_xy
        else:
            # 断帧或刚启动：速度未知（置零），外推时就是"保持不动"。
            self._vel[slot] = 0.0

        self._last_index = int(frame_index)
        self._last_xy = position.copy()

        self._cursor = (slot + 1) % self.capacity
        if self._count < self.capacity:
            self._count += 1

    def get(self, frame_index: int) -> np.ndarray | None:
        """取某帧的位置；不在缓冲里（或已被覆盖）返回 ``None``。"""
        slot = self._find(frame_index)
        return None if slot < 0 else self._xy[slot].copy()

    def get_velocity(self, frame_index: int) -> np.ndarray | None:
        """取某帧的速度；不在缓冲里返回 ``None``。"""
        slot = self._find(frame_index)
        return None if slot < 0 else self._vel[slot].copy()

    def get_timestamp(self, frame_index: int) -> float | None:
        """取某帧的时间戳（秒）。

        规格写的是 ``RingBuffer[t] = (x̂_t, ŷ_t, timestamp_t)``。本实现以
        **绝对帧号**为主键（帧号就是定帧率下的时间代理），时间戳是**可选**的
        附加量：调用方显式传入才有值，否则是 NaN。
        """
        slot = self._find(frame_index)
        if slot < 0:
            return None
        value = float(self._time[slot])
        return None if math.isnan(value) else value
    def overwrite(self, frame_index: int, xy) -> bool:
        """覆写某帧已记录的位置（臂 3 退回重推演后回写历史）。

        退回重推演算出一条**无漂移**的新轨迹，必须把它写回历史缓冲，
        否则后续帧的"同刻对账"残差仍然会拿旧的前向轨迹做基准，
        累积漂移会被算成"快核误差"。

        连带更新两处派生量：

        * 该帧的**速度**（= 覆写后的位置 − 前一帧位置）；
        * 若覆写的是缓冲里**最新**的一帧，同步更新 ``_last_index/_last_xy``，
          否则下一次 :meth:`push` 会拿"覆写前的旧位置"去算速度。

        Returns
        -------
        bool
            是否真的覆写了（帧不在缓冲里时返回 False）。
        """
        slot = self._find(frame_index)
        if slot < 0:
            return False
        position = np.asarray(xy, dtype=np.float64).reshape(2)
        self._xy[slot] = position

        previous = self.get(int(frame_index) - 1)
        if previous is not None:
            self._vel[slot] = position - previous

        if self._last_index == int(frame_index):
            self._last_xy = position.copy()
        return True

    def latest_velocity(self) -> np.ndarray | None:
        """取缓冲里**最新一帧**的速度（不要求帧号连续）。

        外推用这个而不是 ``get_velocity(t-1)``：丢帧/跳帧时后者会取不到，
        外推就退化成"位置冻结"，而文档要求 LOST 态"允许外推维持若干帧"。
        """
        if self._count == 0 or self._last_index is None:
            return None
        slot = self._find(self._last_index)
        return None if slot < 0 else self._vel[slot].copy()
    def has(self, frame_index: int) -> bool:
        return self._find(frame_index) >= 0

    def frames(self) -> np.ndarray:
        """缓冲里现存的帧号（升序）。"""
        live = self._frames[: self._count]
        return np.sort(live[live >= 0])

    def oldest(self) -> int | None:
        live = self._frames[: self._count]
        live = live[live >= 0]
        return int(live.min()) if len(live) else None

    def newest(self) -> int | None:
        live = self._frames[: self._count]
        live = live[live >= 0]
        return int(live.max()) if len(live) else None

    def clear(self) -> None:
        self._frames[:] = -1
        self._xy[:] = 0.0
        self._vel[:] = 0.0
        self._time[:] = np.nan
        self._count = 0
        self._cursor = 0
        self._last_index: int | None = None
        self._last_xy = np.zeros(2, dtype=np.float64)

    # -- 内部 ---------------------------------------------------------------

    def _find(self, frame_index: int) -> int:
        target = int(frame_index)
        for offset in range(self._count):
            slot = (self._cursor - 1 - offset) % self.capacity
            if int(self._frames[slot]) == target:
                return slot
        return -1
