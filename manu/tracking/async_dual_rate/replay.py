"""臂 3：退回重推演（Rollback & Replay）内核。

**为什么需要它**：在"异步检跟分离"里，后台检测器给回来的永远是 ``Δ`` 帧以前
的旧框。臂 2（纯前向外推）拿这个旧框只做对比、不做修正，于是**快核自身的
累积漂移**没有任何东西能压住——Δ 帧之后误差只增不减，伺服坐标越走越偏。

臂 3 的做法是**把历史推倒重来**：

1. 第 ``t−Δ`` 帧的检测框到达时，把它当作**第 t−Δ 帧的绝对锚点**（trust anchor）；
2. 从缓存里调出历史图像切片，从锚点出发**逐帧重推** ``Δ−1`` 步，
   一直到第 ``t−1`` 帧；
3. 再由主流程用当前帧正常走一步，落到第 ``t`` 帧；
4. 用重算出来的**无漂移轨迹**覆盖当前状态与历史缓冲。

于是"从锚点到输出"这条路径上的累积漂移被**整体清零**，重推演的结果只受
两个误差源约束：

* **锚点自身误差**：检测框在第 ``f`` 帧的定位误差（这是天花板的上限，
  重推演消不掉——它消的是**累积漂移**，不是**锚点偏差**）；
* **重推演期间的局部跟踪误差**：每一步快核在 32×32 ROI 内的提取误差。

本模块只负责"重推"这一件事：帧缓存 + 逐帧重推。状态机与历史回写在
:mod:`tracker` 里做。

**内存账**：640×512 的 uint8 灰度帧 = 320 KB/帧，容量取 ``Δ+2`` 帧 ⇒
约 3.3 MB。这是 Python 参考实现的代价；板卡移植时换成 ROI 切片即可
（只存 32×32 而不是整帧，降到 10 KB 量级）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

__all__ = ["FrameRingBuffer", "ReplayResult", "replay_forward"]


class FrameRingBuffer:
    """按**绝对帧号**索引的灰度帧环形缓冲。

    退回重推演要回放历史图像，所以必须能按绝对帧号取回"第 f 帧那张图"。
    容量必须 **> Δ**：要重放 ``[f+1, t−1]``（Δ−1 帧）再算当前帧。

    与 :class:`~.history.HistoryRingBuffer` 一样按绝对帧号 ``mod depth``
    索引，禁列表移位（项目铁律）。
    """

    def __init__(self, capacity: int = 16) -> None:
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self.capacity = int(capacity)
        self._slots: list[np.ndarray | None] = [None] * self.capacity
        self._frames = np.full(self.capacity, -1, dtype=np.int64)
        self._cursor = 0

    def __len__(self) -> int:
        return int((self._frames >= 0).sum())

    def push(self, frame_index: int, frame: np.ndarray) -> None:
        """写入一帧图像（**存引用，不拷贝**）。

        同一帧在锁步遍历里被所有臂共享，所以不拷贝几乎不占额外内存。
        """
        slot = self._cursor
        self._slots[slot] = frame
        self._frames[slot] = int(frame_index)
        self._cursor = (slot + 1) % self.capacity

    def get(self, frame_index: int) -> np.ndarray | None:
        """取某帧的图像；不在缓冲里返回 ``None``。"""
        slot = self._find(frame_index)
        return None if slot < 0 else self._slots[slot]

    def has(self, frame_index: int) -> bool:
        return self._find(frame_index) >= 0

    def frames(self) -> np.ndarray:
        """缓冲里现存的帧号（升序）。"""
        live = self._frames[self._frames >= 0]
        return np.sort(live)

    def clear(self) -> None:
        self._slots = [None] * self.capacity
        self._frames[:] = -1
        self._cursor = 0

    # -- 内部 ---------------------------------------------------------------

    def _find(self, frame_index: int) -> int:
        target = int(frame_index)
        for offset in range(self.capacity):
            slot = (self._cursor - 1 - offset) % self.capacity
            if int(self._frames[slot]) == target:
                return slot
        return -1


@dataclass(frozen=True)
class ReplayResult:
    """一次退回重推演的结果（逐帧取证的原料）。"""

    applied: bool
    """是否真的重推成功（False = 弃权，保留原状态）。"""

    kind: str
    """``"replay"`` / ``"replay_abstain"``。"""

    reason: str
    """弃权原因（``"ok"`` / ``"missing_frame:<n>"`` / ``"lost_target:<n>"`` ...）。"""

    anchor_frame: int
    anchor_xy: np.ndarray
    """绝对锚点（= 迟到检测框的中心，落在第 ``anchor_frame`` 帧）。"""

    target_frame: int
    """重推到的目标帧（**不含**，该帧由主流程正常走一步）。"""

    steps: int
    """实际重推的步数。"""

    positions: dict[int, np.ndarray] = field(default_factory=dict)
    """重推出来的逐帧位置（含锚点帧）。"""

    cost_ms: float = 0.0
    """重推本身的耗时（毫秒）。"""


def replay_forward(
    fast_tracker,
    cache: FrameRingBuffer,
    anchor_xy,
    anchor_frame: int,
    target_frame: int,
    require_valid: bool = True,
) -> ReplayResult:
    """从绝对锚点出发逐帧重推到 ``target_frame``（不含）。

    Parameters
    ----------
    fast_tracker:
        快跟踪核（:class:`~.fast_tracker.LocalPatchTracker`）。必须是**无状态**
        的——重推会用一个实例连续跑多帧。
    cache:
        帧环形缓冲。
    anchor_xy:
        绝对锚点坐标（``anchor_frame`` 时刻的真值位置）。
    anchor_frame:
        锚点所在的帧号 ``f``。
    target_frame:
        重推到的帧号 ``t``（**不含**）。区间 ``[f+1, t)`` 共 ``t−f−1`` 步。
    require_valid:
        重推途中任一步快核判定"信号消失"就整体弃权。

        默认 True：锚点本身可能是个假阳性/错目标，硬推只会把轨迹带进
        虚空；宁可保留前向状态（带漂移）也不要一个错得更离谱的答案。
    """
    anchor = np.asarray(anchor_xy, dtype=np.float64).reshape(2).copy()
    start_time = perf_counter()

    if target_frame <= anchor_frame:
        return ReplayResult(
            applied=False,
            kind="replay_abstain",
            reason="empty_span",
            anchor_frame=int(anchor_frame),
            anchor_xy=anchor,
            target_frame=int(target_frame),
            steps=0,
            cost_ms=(perf_counter() - start_time) * 1000.0,
        )

    position = anchor
    positions: dict[int, np.ndarray] = {int(anchor_frame): anchor.copy()}

    for frame_index in range(int(anchor_frame) + 1, int(target_frame)):
        image = cache.get(frame_index)
        if image is None:
            return ReplayResult(
                applied=False,
                kind="replay_abstain",
                reason=f"missing_frame:{frame_index}",
                anchor_frame=int(anchor_frame),
                anchor_xy=anchor,
                target_frame=int(target_frame),
                steps=len(positions) - 1,
                positions=positions,
                cost_ms=(perf_counter() - start_time) * 1000.0,
            )
        result = fast_tracker.step(image, position)
        if require_valid and not result.valid:
            return ReplayResult(
                applied=False,
                kind="replay_abstain",
                reason=f"lost_target:{frame_index}",
                anchor_frame=int(anchor_frame),
                anchor_xy=anchor,
                target_frame=int(target_frame),
                steps=len(positions) - 1,
                positions=positions,
                cost_ms=(perf_counter() - start_time) * 1000.0,
            )
        position = np.asarray(result.xy, dtype=np.float64).copy()
        positions[int(frame_index)] = position.copy()

    return ReplayResult(
        applied=True,
        kind="replay",
        reason="ok",
        anchor_frame=int(anchor_frame),
        anchor_xy=anchor,
        target_frame=int(target_frame),
        steps=len(positions) - 1,
        positions=positions,
        cost_ms=(perf_counter() - start_time) * 1000.0,
    )
