"""异步双频检跟分离系统（Async Dual-Rate Detection-Guided Tracking Pipeline）—— Python 参考实现。

把"高算力、高延迟的全局检测"移出伺服控制关键路径，由轻量局部跟踪器每帧
低延迟输出，检测器只作为后台低频真理校正源。

模块地图：

* :mod:`fast_tracker` —— 快流：局部质心/峰值提取核（< 1 ms，纯 NumPy）
* :mod:`history`      —— 历史环形缓冲（按绝对帧号索引，供"同刻同帧对账"）
* :mod:`fusion`       —— 异步延迟残差融合（小残差平滑 / 大残差硬拉回）
* :mod:`tracker`      —— 三态状态机（SEARCH / TRACK / LOST）
* :mod:`detector`     —— 慢流：RC2 检测器输出回放 + 低频/滞后调度
* :mod:`metrics`      —— 冻结口径**适配层**（**import 冻结实现**；不提供时序指标）
* :mod:`servo_metrics` —— 时序对齐的"伺服有效性"度量（CLE / HitRate+Wilson CI / 断裂 / 抖动归因）
* :mod:`replay`       —— **臂 3「退回重推演」内核**：帧环形缓存 + 逐帧重推
* :mod:`arms`         —— 三个对照臂的定义 + **同源锁步遍历**（受控 A/B 的载体）
* :mod:`evaluate`     —— 离线评测与验证闸门（Gate 1~4 / 6）
* :mod:`three_arm`    —— **三臂验收表生产器**（含三个天花板的量化）
* :mod:`async_pipeline` —— Gate 5 真实异步臂（真线程 + 队列）
"""

from .detector import Detection, DetectorScheduler, ReplayDetector
from .fast_tracker import LocalPatchTracker, TrackerResult
from .fusion import FusionConfig, FusionDecision, ResidualFusion
from .history import HistoryRingBuffer
from .replay import FrameRingBuffer, ReplayResult, replay_forward
from .tracker import AsyncDualRateTracker, TrackerConfig, TrackerOutput, TrackerState

__all__ = [
    "AsyncDualRateTracker",
    "Detection",
    "DetectorScheduler",
    "FrameRingBuffer",
    "FusionConfig",
    "FusionDecision",
    "HistoryRingBuffer",
    "LocalPatchTracker",
    "ReplayDetector",
    "ReplayResult",
    "ResidualFusion",
    "TrackerConfig",
    "TrackerOutput",
    "TrackerResult",
    "TrackerState",
    "replay_forward",
]
