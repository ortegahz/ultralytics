"""异步延迟残差融合（Delayed Gating & Correction）。

慢流检测器在 ``t-Δ`` 帧上推理、于 ``t`` 时刻才把结果交回来。要判断"快跟踪器
这 Δ 帧里有没有跟丢"，必须拿**同一时刻**的两���坐标做差：

    e = z_{t-Δ} − x_{t-Δ}

其中 ``z`` 是检测框中心，``x`` 是快跟踪器在那一帧的位置（从历史环形缓冲取回）。

两种处置（文档 §二.3）：

* **情况 A（小残差，‖e‖ ≤ 门限）**：快跟踪器没跟丢，只是有累积漂移。
  把残差平滑地注入当前帧：``x ← x + α·e``（α ≈ 0.3~0.5），
  避免伺服指令瞬时突变。
* **情况 B（大残差，‖e‖ > 门限）**：快跟踪器被地杂波拉偏脱靶了。
  放弃当前跟踪状态，用检测框作基准、**叠上历史运动速度 v × Δ** 强行投影重置。

本模块只做"纯函数"式的残差判定与位置更新，不含状态机；状态机在 ``tracker.py``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .history import HistoryRingBuffer

__all__ = ["FusionConfig", "FusionDecision", "ResidualFusion"]


@dataclass(frozen=True)
class FusionConfig:
    """融合参数。"""

    gating_dist: float = 12.0
    """残差门限（像素）。‖e‖ 超过它就判为失配，走硬拉回。"""

    alpha: float = 0.4
    """小残差情况下的平滑系数（文档给 0.3~0.5）。"""

    hard_reinit: bool = True
    """大残差时是否硬拉回。关掉可以做成"只平滑不重置"的消融臂。"""

    project_velocity: bool = True
    """硬拉回时是否用历史速度把 Δ 帧前的检测位置投影到当前时刻。"""

    velocity_source: str = "none"
    """硬拉回投影用的**速度源**。

    * ``"none"``（**默认**）：**不投影**，直接落在 ``z_f``。这正是文档 §三
      伪代码骨架的写法（``self.curr_pos = np.array(det_xy)``）。
    * ``"history"``：用**快跟踪器自己的历史速度**，对应文档 §二.3 情况 B 的
      "叠加上历史运动速度 v × Δ"。**真实序列实测为负收益**（见下）。
    * ``"detection"``：用**检测流速度**（最近若干次关联上的检测）。

    ⚠️ **文档自身在这两点上不一致**：§二.3 的散文说要乘 ``v × Δ``，§三 的
    Python 骨架却是直接 ``= det_xy``。本实现把两种语义都做成可切换，
    默认取 ``"none"``——**真实序列实测**（龙泉山 001324，三臂 + 消融，
    同一份 ``evidence_ablation_fused_alpha.tsv`` 逐帧量化）：
    ``none`` 的消融 alpha 臂命中率 **46.64%**，``history`` 只有 **45.20%**。
    原因见 ``manu/tracking/README.md`` 的「速度源对照」段：硬拉回触发的前提
    就是"快跟踪器已锁在杂波上"，此刻它的历史速度**正是杂波的速度**，乘 Δ=8
    后被放大成 p50 25.27 / p90 78.11 / max 408.90 px 的假位移，把刚拉回
    的目标又甩出 ROI——``history`` 档实测 **65.0%** 的硬拉回落在 ROI 半窗
    （16 px）之外，硬拉回后第 1 帧的 inbox 命中率只有 **75.0%**
    （168/224）；``none`` 档投影位移恒为 **0**、**0.0%** 落在窗外，
    第 1 帧 inbox 命中率 **84.0%**（200/238）。

    ⚠️ 无论选哪个源，都**不得**因为"速度算出来是 0"就换成另一个源——
    0 是**合法速度**（目标静止）。早期实现是"历史速度为 0 就用检测流速度
    兜底"，在 Gate 3b 的双斑场景里把一个正确的 0 换成了**跨目标切换**算出
    的脏速度（−2.857 px/帧），乘 Δ=2 后变成 5.71 px 的假位移。
    检测流速度只应在**历史速度取不到**（缓冲里没有可用帧）时作为兜底。
    """

    smooth_when_small: bool = True
    """小残差时是否走平滑（``x += α·e``）。

    关掉 = 消融臂"检跟分离但**没有**残差平滑"：任何残差都走硬拉回。
    用来暴露"硬拉回会把转台指令打成抽搐"这件事——它与完整架构之间只差
    这一个开关，是受控 A/B 的第三个臂。
    """


@dataclass(frozen=True)
class FusionDecision:
    """一次融合处置的完整记录（用于逐帧取证）。"""

    applied: bool
    """是否真的做了位置修改。"""

    kind: str
    """``"none"`` / ``"smooth"`` / ``"hard_reinit"``。"""

    residual: np.ndarray
    """历史残差 ``e = z_{t-Δ} − x_{t-Δ}``；无历史对账时为 NaN。"""

    residual_norm: float
    """‖e‖；无历史对账时为 NaN。"""

    position_before: np.ndarray
    position_after: np.ndarray
    """处置前后的当前帧位置。"""


class ResidualFusion:
    """把"迟到的检测框"折算成对"当前快跟踪位置"的修正。"""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(
        self,
        position,
        detection_xy,
        detection_frame: int,
        current_frame: int,
        history: HistoryRingBuffer,
        fallback_velocity=None,
    ) -> FusionDecision:
        """执行一次残差融合。

        Parameters
        ----------
        position:
            当前（上一帧末的）快跟踪位置 ``(x, y)``。
        detection_xy:
            迟到的检测框中心 ``(x, y)``。
        detection_frame:
            检测框对应的**源帧号** ``f``（= current_frame − Δ）。
        current_frame:
            当前物理帧号 ``t``。
        history:
            历史环形缓冲，必须含 ``f`` 时刻的快跟踪位置。
        fallback_velocity:
            备用速度（像素/帧）。**仅当历史速度取不到时**才用它（缓冲里没有
            可用帧）——此时**检测流自己的速度**比"没有历史"更接近真值。

            ⚠️ **不得因为历史速度算出来是 0 就换用这个兜底值**：0 是**合法
            速度**（目标静止）。早期实现是"历史速度为 0 就用检测流速度兜底"，
            在 Gate 3b 的双斑场景里把一个正确的 0 换成了**跨目标切换**算出的
            脏速度（−2.857 px/帧），乘 Δ=2 后变成 5.71 px 的假位移。
            真实场景里硬拉回触发的前提就是"快跟踪核已锁在杂波上"，此刻它的
            历史速度**正是杂波的速度**，用它投影只会把刚拉回的目标再甩出去。
        """
        before = np.asarray(position, dtype=np.float64).reshape(2).copy()
        detection = np.asarray(detection_xy, dtype=np.float64).reshape(2)

        historical = history.get(detection_frame)
        if historical is None:
            # 拿不到同一时刻的历史状态 ⇒ 无法做"同刻同帧对账"，
            # 残差无意义。这次**不做任何修正**（宁可不动，也不要拿错时刻的差值
            # 去猛推伺服）。文档要求的"历史对账"前提不成立时必须弃权。
            return FusionDecision(
                applied=False,
                kind="none",
                residual=np.full(2, np.nan),
                residual_norm=float("nan"),
                position_before=before,
                position_after=before,
            )

        residual = detection - historical
        norm = float(np.linalg.norm(residual))
        cfg = self.config

        if norm <= cfg.gating_dist and cfg.smooth_when_small:
            after = before + cfg.alpha * residual
            return FusionDecision(
                applied=True,
                kind="smooth",
                residual=residual,
                residual_norm=norm,
                position_before=before,
                position_after=after,
            )

        if not cfg.hard_reinit:
            # 消融臂：门限外也只做平滑（甚至可以完全不动）。
            after = before + cfg.alpha * residual
            return FusionDecision(
                applied=True,
                kind="smooth",
                residual=residual,
                residual_norm=norm,
                position_before=before,
                position_after=after,
            )

        # 硬拉回：z_f 是 Δ 帧前的位置，要投影到当前时刻 t。
        after = detection.copy()
        if cfg.project_velocity and cfg.velocity_source != "none":
            velocity = self._resolve_velocity(
                history, detection_frame, current_frame, fallback_velocity
            )
            span = max(current_frame - detection_frame, 0)
            after = after + velocity * span
        return FusionDecision(
            applied=True,
            kind="hard_reinit",
            residual=residual,
            residual_norm=norm,
            position_before=before,
            position_after=after,
        )

    # -- 内部 ---------------------------------------------------------------

    def _resolve_velocity(
        self,
        history: HistoryRingBuffer,
        detection_frame: int,
        current_frame: int,
        fallback_velocity,
    ) -> np.ndarray:
        """硬拉回投影用的速度（像素/帧）。

        ⚠️ 关键：**不因"速度为 0"而替换速度源**。0 是合法速度（目标静止）；
        替换成检测流速度会引入**跨目标切换**的脏速度（见 ``velocity_source``）。
        """
        cfg = self.config
        history_velocity = self._window_velocity(history, detection_frame, current_frame)
        if cfg.velocity_source == "detection" and fallback_velocity is not None:
            return np.asarray(fallback_velocity, dtype=np.float64).reshape(2)
        if cfg.velocity_source == "history":
            if fallback_velocity is not None and not self._history_velocity_available(
                history, detection_frame, current_frame
            ):
                # 历史速度**取不到**（缓冲里没有可用帧）才用检测流兜底。
                return np.asarray(fallback_velocity, dtype=np.float64).reshape(2)
            return history_velocity
        return history_velocity

    @staticmethod
    def _history_velocity_available(
        history: HistoryRingBuffer, detection_frame: int, current_frame: int
    ) -> bool:
        """历史速度是否**取得到**（而不是"是否为 0"）。"""
        if history.get(detection_frame) is None:
            return False
        if history.get(current_frame - 1) is not None:
            return (current_frame - 1) - detection_frame > 0
        return history.get_velocity(detection_frame) is not None

    @staticmethod
    def _window_velocity(
        history: HistoryRingBuffer, detection_frame: int, current_frame: int
    ) -> np.ndarray:
        """估计 [detection_frame, current_frame) 区间上的平均速度（像素/帧）。

        优先用**区间两端**的历史位置做差（平均速度，最稳）；
        两端拿不到就退化成"检测帧处的瞬时速度"，再退化成 0（保持不动）。
        """
        start = history.get(detection_frame)
        end = history.get(current_frame - 1)
        if start is not None and end is not None:
            span = (current_frame - 1) - detection_frame
            if span > 0:
                return (end - start) / float(span)
        instantaneous = history.get_velocity(detection_frame)
        if instantaneous is not None:
            return instantaneous
        return np.zeros(2, dtype=np.float64)
