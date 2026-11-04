"""快速前向推演外挂（Look-ahead Plugin）。

把「上报给转台的坐标」从滞后的 ``t−Δ`` 帧推到物理帧 ``t``，作为成熟跟踪管线的
**输出末端外挂**：不改任何历史估计，只产出新的上报流。

本包当前提供外推核心（:mod:`manu.tracking.lookahead.extrapolator`）：
配置、帧源协议、速度估计、上报流构造、伺服物理指标。
"""

from manu.tracking.lookahead.extrapolator import (
    FrameSource,
    LookaheadConfig,
    build_report_stream,
    estimate_velocity,
    servo_metrics,
)

__all__ = [
    "FrameSource",
    "LookaheadConfig",
    "build_report_stream",
    "estimate_velocity",
    "servo_metrics",
]
