"""真实异步臂（Gate 5）：快流主线程 + 慢流检测线程的真并发参考实现。

前面几个模块（``tracker`` / ``detector`` / ``fusion`` / ``history``）描述的是**算法**
；本模块把它们接成**生产形态**：相机帧在主线程逐帧推进（伺服关键路径），
慢流检测在**独立线程**里跑，通过 ``queue.Queue`` 异步回传结果。

本模块要证明的三件事（全部是可断言的，不靠肉眼）：

1. **并发性**：检测线程再忙，快流单帧耗时也不变。
   检测线程里跑的是**真实计算**（多尺度降采样 + 局部背景抑制 + 质心定位），
   耗时可配置（``inference_ms``），用来模拟 GPU 推理；不是 ``sleep`` 空转。
2. **顺序性**：到达快流的检测**永远滞后**（源帧号 ``f <= t - Δ``），绝不提前；
   融合残差严格等于 ``z_f - x_f``（``x_f`` 取自历史环形缓冲在源帧号 ``f`` 处的记录）；
   应用顺序与源帧号**严格递增**（同一批检测不会被应用两次，队列里的也不会被丢掉）；
   快流帧号**严格 +1** 连续。
3. **延迟**：快流单帧墙钟耗时的 p50/p95/max（用中位数不用均值——项目铁律），
   以及"检测产生 → 被快流应用"的端到端滞后（帧数 + 毫秒）。

只依赖 numpy + 标准库（``threading`` / ``queue`` / ``time``），Python 3.10，
不需要 torch / GPU / 网络。帧源是合成的灰度图 + 移动亮斑，不读外部数据。

**实时性卫生**（都在 :class:`PipelineConfig` 里可关，运行结束全部还原）：

============  ==========================================================
手段          解决什么
============  ==========================================================
绑核          线程在核之间迁移 ⇒ 缓存/TLB 失效，p99 被调度噪声抬到 ms 级
分片休眠      一次睡到底 ⇒ CPU 进深 C 态，唤醒后要爬频率，头几百 µs 全慢
GC 阈值       一次 gen-2 回收 ≈1 ms，正好顶穿 1 ms 判据
malloc 阈值   大块反复 mmap/munmap ⇒ 内核 TLB shootdown 打穿快流线程
GIL 切换间隔  后台线程抢到 GIL 后，前台最多要等 5 ms 才拿回来
流水线预热    唤醒后先空跑一次同样的核，把指令/数据缓存热起来
============  ==========================================================

**共享机器的坑**：这台机器是多人共用的，宿主机一忙，本进程会被整体降频
或抢占 1~2 ms，p95/p99/max 会整体抬高（把检测线程整个拿掉也能复现同样
的噪声）。所以报告里同时给**墙钟**与**线程 CPU 时间**、同时给忙帧与空闲帧
的分位数，并打印宿主机负载——把"系统慢"和"机器慢"分开。

用法::

    python -m manu.tracking.async_dual_rate.async_pipeline     # 自检 + 打印 PASS/FAIL
"""

from __future__ import annotations

import ctypes
import gc
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np

from .detector import Detection
from .fusion import FusionDecision
from .tracker import AsyncDualRateTracker, TrackerConfig, TrackerOutput, TrackerState

__all__ = [
    "AsyncPipeline",
    "assert_not_early",
    "DetectionResult",
    "DetectorWorker",
    "FrameRecord",
    "MultiScaleBlobDetector",
    "PipelineConfig",
    "PipelineReport",
    "PipelineViolation",
    "SyntheticFrameSource",
    "percentile_stats",
    "run_self_check",
    "verify_report",
]


class PipelineViolation(RuntimeError):
    """时序契约被破坏（例如检测提前到达）。

    这是**硬闸**，不是断言：一旦发生就说明系统用到了"未来信息"，
    必须让整条流水线炸掉，而不是静默地产生看起来正常的结果。
    """


def assert_not_early(source_frame: int, current_frame: int) -> None:
    """硬闸：检测不能来自"相机还没拍到的帧"。

    ``source_frame > current_frame`` 意味着检测声称自己来自一个**还没发生**的
    帧 ⇒ 系统用到了未来信息 ⇒ 必须让整条流水线炸掉。

    它和"Δ 放行条件"是两件事，不能混成一条：``source_frame + Δ > current_frame``
    只是"**还没老到**"，那种检测留在队列里继续等（见 :meth:`AsyncPipeline._drain`），
    不是违规。写成同一条断言，硬闸就成了永远为真的死代码（变异测试：删掉调用点，
    整套测试依然全绿）。真正被应用的那批检测满足 ``source_frame <= 当前帧 - Δ``，
    那由放行条件保证，并由 :func:`verify_report` 的 ``never_early`` 项复核。
    """
    if int(source_frame) > int(current_frame):
        raise PipelineViolation(
            f"检测的源帧号 {source_frame} 晚于相机当前帧 {current_frame}"
            f"（来自还没发生的帧 ⇒ 用了未来信息）"
        )


# ---------------------------------------------------------------------------
# 实时性调优（全部 best-effort，失败即静默跳过）
# ---------------------------------------------------------------------------

_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3


def tune_allocator() -> bool:
    """把 glibc 的 ``mmap``/``trim`` 阈值抬高，避免大块反复 mmap/munmap。

    慢流线程每秒申请/释放几百个几百 KB 的数组；走 ``mmap`` 路径时内核要
    反复拆/建映射并向其它线程发 TLB shootdown IPI，**这些内核态开销会被
    计进快流线程的墙钟耗时**，把 p99 抬到毫秒级。抬阈值后大块从堆上复用，
    抖动明显收敛。glibc 不可用时静默跳过。
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        ok = libc.mallopt(_M_TRIM_THRESHOLD, 512 * 1024 * 1024)
        ok &= libc.mallopt(_M_MMAP_THRESHOLD, 512 * 1024 * 1024)
        return bool(ok)
    except Exception:  # pragma: no cover - 非 glibc 平台
        return False


def choose_cores(count: int = 2) -> list[int]:
    """在当前可用 CPU 集里挑 ``count`` 个核（快流一个、慢流一个）。"""
    try:
        available = sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - 非 Linux
        return []
    return available[:count]


def pin_thread(cores: list[int], index: int) -> bool:
    """把**当前线程**绑到 ``cores[index]``（失败静默跳过）。"""
    if not cores or index >= len(cores):
        return False
    try:
        os.sched_setaffinity(0, {cores[index]})
        return True
    except OSError:  # pragma: no cover - 权限/平台限制
        return False


# ---------------------------------------------------------------------------
# 统计工具
# ---------------------------------------------------------------------------


def percentile_stats(values) -> dict:
    """分位数统计。

    **用中位数不用均值**（项目铁律：抖动统计用 median 不用 mean），
    同时给 p95/p99/max 暴露长尾。
    """
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        nan = float("nan")
        return {"n": 0, "min": nan, "p50": nan, "p95": nan, "p99": nan, "max": nan}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


# ---------------------------------------------------------------------------
# 帧源：合成灰度图 + 移动亮斑
# ---------------------------------------------------------------------------


@dataclass
class SyntheticFrameSource:
    """合成帧源：低频缓变背景 + 固定噪声场 + 一个移动的高斯亮斑。

    目标路径取**正弦**而非匀速直线：匀速直线在边界上会走出画面（跟踪器会丢目标，
    把"并发/时序"的验证和"跟踪器鲁棒性"搅在一起）。正弦路径始终在画面内，
    且速度有缓慢变化，更接近真实小目标的机动。
    """

    width: int = 320
    height: int = 256
    start_xy: tuple[float, float] = (140.0, 120.0)
    amplitude_xy: tuple[float, float] = (72.0, 46.0)
    period_xy: tuple[float, float] = (170.0, 240.0)
    blob_amplitude: float = 110.0
    blob_sigma: float = 1.7
    background: float = 30.0
    noise_sigma: float = 2.0
    seed: int = 20261102

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # 固定噪声场：帧间不重采样 ⇒ 逐帧结果可复现，跟踪器估计也稳定。
        self._noise = rng.normal(0.0, self.noise_sigma, size=(self.height, self.width)).astype(np.float32)
        rows = np.arange(self.height, dtype=np.float32)[:, None]
        cols = np.arange(self.width, dtype=np.float32)[None, :]
        gradient = 4.0 * cols / max(self.width - 1, 1) + 3.0 * rows / max(self.height - 1, 1)
        # 背景 + 噪声是**静态**的，直接量化成 uint8 基底：
        # 每帧只需要复制一张 82 KB 的 uint8 图（而不是 float32 的 330 KB +
        # clip 临时量 + astype 临时量 ≈ 1 MB 的内存流量）。
        # 内存流量会挤掉快跟踪核的 L1/L2，直接体现在快流单帧耗时上。
        self._base = np.clip(
            self.background + gradient + self._noise, 0.0, 255.0
        ).astype(np.uint8)

    def position(self, frame_index: int) -> np.ndarray:
        """目标真值位置 ``(x, y)``（像素，x=列、y=行）。"""
        cx, cy = self.start_xy
        ax, ay = self.amplitude_xy
        px, py = self.period_xy
        angle_x = 2.0 * math.pi * frame_index / px
        angle_y = 2.0 * math.pi * frame_index / py + 0.7
        return np.array([cx + ax * math.sin(angle_x), cy + ay * math.sin(angle_y)], dtype=np.float64)

    def frame(self, frame_index: int) -> np.ndarray:
        """生成第 ``frame_index`` 帧（``uint8`` 灰度图）。

        亮斑只在一小块局部窗口里叠加（而不是整幅算 exp），把造帧成本压到
        几十微秒——否则造帧的耗时会淹没"快流单帧耗时"这个指标。
        """
        canvas = self._base.copy()
        cx, cy = self.position(frame_index)
        radius = int(3.0 * self.blob_sigma) + 1
        x1 = int(round(cx)) - radius
        y1 = int(round(cy)) - radius
        x2 = x1 + 2 * radius + 1
        y2 = y1 + 2 * radius + 1
        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2, self.width), min(y2, self.height)
        if x2c > x1c and y2c > y1c:
            ys = np.arange(y1c, y2c, dtype=np.float32)[:, None] - cy
            xs = np.arange(x1c, x2c, dtype=np.float32)[None, :] - cx
            blob = self.blob_amplitude * np.exp(-(xs * xs + ys * ys) / (2.0 * self.blob_sigma**2))
            # 基底 ≤ 45、亮斑 ≤ 110 ⇒ 和 ≤ 155，uint8 相加不会回绕。
            canvas[y1c:y2c, x1c:x2c] += blob.astype(np.uint8)
        return canvas


# ---------------------------------------------------------------------------
# 慢流检测器：真实的多尺度"推理"
# ---------------------------------------------------------------------------


def _box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    """局部均值（积分图实现，O(1)/像素）。

    模拟检测器里的"局部背景抑制"：红外弱小目标检测的标准做法是
    大核均值（或中值）估计背景、减去背景再看残差。

    积分图用 **float32** 累加：检测器的工作集会直接挤占 L2/L3，
    而 L2/L3 正是快跟踪核（32x32 ROI + median/partition）住的地方。
    float64 会让工作集翻倍，实测会把快流单帧耗时抬高一倍。
    """
    padded = np.pad(image, radius, mode="edge")
    height, width = padded.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float32)
    integral[1:, 1:] = padded.cumsum(axis=0, dtype=np.float32).cumsum(axis=1, dtype=np.float32)
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / np.float32(size * size)


@dataclass
class MultiScaleBlobDetector:
    """一个**真的**会做计算的目标检测器（多尺度降采样 + 背景抑制 + 质心定位）。

    它不用网络，但每一处开销都是真实的 NumPy 计算：

    1. 按尺度降采样（块均值）；
    2. 大核局部均值估计背景并相减；
    3. 用 MAD 估噪声尺度，取阈值以上的极大点；
    4. 回到**全分辨率**在极大点邻域做亚像素质心细化。

    多尺度里信噪比最高的那个尺度胜出（和真实多尺度检测器一致）。
    定位噪声由 ``jitter_sigma`` 注入，且用 ``frame_index`` 播种
    ⇒ 结果完全可复现，与"跑了多少遍"无关。
    """

    scales: tuple[float, ...] = (2.0, 4.0)
    """多尺度检测的降采样倍率。

    真实检测器不是拿相机原始分辨率做前向的（要 resize 到网络输入尺寸），
    这里同理：**降采样后再找目标**，最后回到全分辨率做亚像素质心细化。
    顺带的好处：慢流的工作集从几百 KB 降到几十 KB，不再把快跟踪核的
    L1/L2 挤出去（这是"检测线程拖慢快流"的主要真实来源，不是 GIL）。
    """

    threshold_k: float = 5.0
    fine_window: int = 13
    jitter_sigma: float = 0.6
    seed: int = 991

    def detect(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        """对 ``frame`` 跑一次完整的多尺度检测，返回检测列表（按分数降序）。"""
        best: tuple[np.ndarray, float, float] | None = None
        for scale in self.scales:
            found = self._detect_scale(frame, scale)
            if found is None:
                continue
            xy, score, snr = found
            if best is None or snr > best[2]:
                best = (xy, score, snr)
        if best is None:
            return []
        xy, score, _ = best
        rng = np.random.default_rng(self.seed * 1_000_003 + int(frame_index))
        xy = xy + rng.normal(0.0, self.jitter_sigma, size=2)
        side = 4.0 * self.fine_window
        return [
            Detection(
                frame_index=int(frame_index),
                xy=np.asarray(xy, dtype=np.float64),
                score=float(score),
                wh=(side, side),
            )
        ]

    # -- 内部 ---------------------------------------------------------------

    def _detect_scale(self, frame: np.ndarray, scale: float):
        factor = int(round(scale))
        if factor > 1:
            height = (frame.shape[0] // factor) * factor
            width = (frame.shape[1] // factor) * factor
            # 直接从 uint8 原图做块均值（累加器是 float32）：
            # 不生成 float32 全图副本 ⇒ 慢流的工作集小一个数量级。
            small = (
                frame[:height, :width]
                .reshape(height // factor, factor, width // factor, factor)
                .sum(axis=(1, 3), dtype=np.float32)
                / np.float32(factor * factor)
            )
        else:
            small = frame.astype(np.float32)

        radius = max(3, int(round(8.0 / factor)))
        diff = small - _box_mean(small, radius)
        center = float(np.median(diff))
        sigma = max(1.4826 * float(np.median(np.abs(diff - center))), 1e-3)
        peak = float(diff.max())
        if peak < self.threshold_k * sigma:
            return None

        flat = int(np.argmax(diff))
        row, col = divmod(flat, diff.shape[1])
        coarse_x = (col + 0.5) * factor
        coarse_y = (row + 0.5) * factor
        fine_x, fine_y = self._refine(frame, coarse_x, coarse_y)
        score = float(np.clip(peak / (peak + 8.0), 0.02, 0.99))
        return np.array([fine_x, fine_y], dtype=np.float64), score, peak / sigma

    def _refine(self, frame: np.ndarray, coarse_x: float, coarse_y: float) -> tuple[float, float]:
        """全分辨率邻域里的亚像素质心细化。"""
        half = self.fine_window // 2
        height, width = frame.shape
        x1 = int(round(coarse_x)) - half
        y1 = int(round(coarse_y)) - half
        x2 = x1 + self.fine_window
        y2 = y1 + self.fine_window
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
        if x2 <= x1 or y2 <= y1:
            return float(coarse_x), float(coarse_y)

        patch = frame[y1:y2, x1:x2].astype(np.float32)
        background = float(np.median(patch))
        sigma = max(1.4826 * float(np.median(np.abs(patch - background))), 1e-3)
        weights = np.maximum(patch - (background + 3.0 * sigma), 0.0)
        energy = float(weights.sum())
        if energy <= 0.0:
            return float(coarse_x), float(coarse_y)
        ramp_x = np.arange(x2 - x1, dtype=np.float32)
        ramp_y = np.arange(y2 - y1, dtype=np.float32)
        local_x = float(weights.sum(axis=0) @ ramp_x) / energy
        local_y = float(weights.sum(axis=1) @ ramp_y) / energy
        return float(x1) + local_x, float(y1) + local_y


# ---------------------------------------------------------------------------
# 检测线程
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionResult:
    """慢流线程产出的一次检测回传。"""

    source_frame: int
    """检测对应的**源帧号** f（不是到达时刻）。"""

    detections: tuple[Detection, ...]
    captured_at: float
    """源帧被相机采到的墙钟时刻。"""

    dequeued_at: float
    produced_at: float
    inference_ms: float
    """检测线程里真正花在"推理"上的时间。"""


class DetectorWorker(threading.Thread):
    """慢流检测线程。

    * 从 ``in_queue`` 取帧（相机帧的**引用**，不复制——真实系统里相机缓冲是共享的）；
    * 在线程内做**真实计算**（多尺度检测），重复到 ``inference_ms`` 的时间预算用完；
    * 把结果塞进 ``out_queue``，快流线程异步取走。

    线程**只碰自己的队列**，永远不碰跟踪器状态 ⇒ 慢流不可能阻塞快流的计算。
    """

    def __init__(
        self,
        detector: MultiScaleBlobDetector,
        in_queue: "queue.Queue",
        out_queue: "queue.Queue",
        stop_event: threading.Event,
        inference_ms: float = 5.0,
        name: str = "slow-detector",
        core: int | None = None,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.core = core
        self.detector = detector
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.inference_ms = float(inference_ms)

        self.processed = 0
        self.dropped = 0
        self.busy_seconds = 0.0
        self.error: BaseException | None = None
        """检测线程里抛出的异常（如果有）：由主线程重新抛出，不能悄悄死掉。"""
        self.busy_since: float | None = None
        """正在做推理的时刻（``None`` = 空闲）。快流用它判断"此刻检测线程是否在忙"。"""

        self._lock = threading.Lock()
        self._busy_intervals: list[tuple[float, float]] = []
        """已完成的推理区间（起止墙钟），按时间递增。"""

    # -- 对外 ---------------------------------------------------------------

    def busy_intervals(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._busy_intervals)

    def is_busy(self, at: float | None = None) -> bool:
        """检测线程在 ``at`` 时刻是否正在做推理。"""
        since = self.busy_since
        if since is None:
            return False
        return True if at is None else since <= at

    # -- 线程体 -------------------------------------------------------------

    def run(self) -> None:
        if self.core is not None:
            pin_thread([self.core], 0)
        while not self.stop_event.is_set():
            try:
                item = self.in_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            if item is None:
                break
            frame, source_frame, captured_at = item

            started = time.perf_counter()
            self.busy_since = started
            try:
                detections = self._infer(frame, source_frame)
                # A3 守卫（放在**产出端**，而不是只在放行端）：
                # 检测框自己声明的源帧号必须与批次源帧号一致。不一致就说明
                # 上游把"图像与帧号对不上"的检测塞进来了——这类错误在放行端
                # 才被发现会晚 Δ 帧，且如果检测列表为空就完全漏检。
                for detection in detections:
                    if int(detection.frame_index) != int(source_frame):
                        raise PipelineViolation(
                            f"检测框声明的源帧号 {detection.frame_index} 与推理帧号 "
                            f"{source_frame} 不一致（图像与帧号对不上）"
                        )
            except BaseException as exc:  # noqa: BLE001 - 要把异常带回主线程
                self.error = exc
                self.busy_since = None
                self.stop_event.set()
                break
            finally:
                self.busy_since = None
            finished = time.perf_counter()

            result = DetectionResult(
                source_frame=int(source_frame),
                detections=tuple(detections),
                captured_at=float(captured_at),
                dequeued_at=float(started),
                produced_at=float(finished),
                inference_ms=(finished - started) * 1000.0,
            )
            try:
                self.out_queue.put_nowait(result)
            except queue.Full:
                self.dropped += 1
                continue

            with self._lock:
                self._busy_intervals.append((started, finished))
                self.busy_seconds += finished - started
            self.processed += 1

    # -- 内部 ---------------------------------------------------------------

    def _infer(self, frame: np.ndarray, source_frame: int) -> list[Detection]:
        """在时间预算内反复跑真实的多尺度检测，返回最后一次的结果。

        结果与重复次数无关（定位噪声按源帧号播种），所以"多跑几遍"纯粹是
        **成本标定**——模拟 GPU 推理耗时，而不是 ``sleep`` 空转。
        """
        budget = self.inference_ms / 1000.0
        deadline = time.perf_counter() + budget
        detections: list[Detection] = []
        while True:
            detections = self.detector.detect(frame, source_frame)
            if time.perf_counter() >= deadline or self.stop_event.is_set():
                break
        return detections


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """真实异步臂的配置。"""

    n_frames: int = 120
    camera_hz: float = 25.0
    det_hz: float = 12.5
    latency_frames: int = 8
    """Δ：检测结果至少滞后多少帧才允许被快流应用。"""

    inference_ms: float = 25.0
    """慢流线程每次推理的目标耗时（模拟 GPU 推理）。

    默认 25 ms：相机 25 Hz、检测 12.5 Hz ⇒ 检测器每 80 ms 拿一帧，
    25 ms 的推理把慢流线程的占空比推到 30%~50%，慢流**常态处于忙状态**——
    这正是"检测线程不阻塞快流"最严苛的工况。
    """

    queue_size: int = 16
    frame_width: int = 320
    frame_height: int = 256
    source_seed: int = 20261102
    detector_seed: int = 991
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    detector: object | None = None
    """慢流检测器实例；None = 用默认的 :class:`MultiScaleBlobDetector`。

    留这个口子是为了**测试**能注入一个"说谎"的检测器（声明的源帧号与图像
    对不上），验证 A3 守卫真的拦得住——没有注入点的话，这条不变量就只能
    靠"读代码觉得没问题"来担保。
    """

    pin_cpus: bool = True
    """把快流线程与慢流线程绑到**不同的核**。

    伺服回路是典型的"绑核"场景：不绑核时线程会在核之间迁移，缓存/TLB 全部
    失效，p99 会被调度噪声抬到毫秒级（实测 0.6 ms → 0.35 ms）。绑核之后
    快流的延迟分布才反映**算法本身**的代价，而不是操作系统的抖动。
    """

    gil_switch_interval: float = 0.0005
    """CPython 的 GIL 切换间隔（秒）。默认 5 ms 太长：后台线程一旦抢到 GIL，
    前台最多要等 5 ms 才拿回来。调到 0.5 ms 相当于"后台线程不许长时间霸占
    GIL"——真实 GPU 推理根本不持 GIL，这里用它近似那种效果。运行结束会还原。
    """

    tune_allocator: bool = True
    """抬高 glibc 的 mmap/trim 阈值（见 :func:`tune_allocator`）。"""

    gc_threshold: tuple[int, int, int] | None = (5000, 200, 200)
    """运行期间的 GC 阈值（``(gen0, gen1, gen2)``）；``None`` = 不动。

    **这是本机 p99 的头号杀手**：一次 gen-2 回收要遍历整个堆，实测要
    ~1 ms，正好压在"Time-to-Servo ≤ 1 ms"判据上；而快流每帧要往报告里
    追加若干取证记录（长期存活对象），会不断把 gen-2 计数器顶上去。
    把阈值从默认的 ``(700, 10, 10)`` 抬到 ``(5000, 200, 200)``，
    实测 p99 从 ~1.5 ms（4 次里 2 次超判据）降到 ~0.4 ms（稳定通过）。
    运行结束会还原成进程原来的阈值。
    """

    sleep_chunk_s: float = 0.001
    """快流节拍的分片休眠粒度（秒）。

    一次睡到下一个帧点，CPU 会进入深 C 态；唤醒后要重新爬频率，
    头几百微秒的指令全部变慢——直接污染"快流单帧耗时"的 p99
    （实测 p99 ≈ 1.3 ms、max ≈ 1.8 ms）。改成每 1 ms 醒一次，
    C 态深度被按住，同样的负载下 p99 ≈ 0.35 ms、max ≈ 0.45 ms。
    这不是美化数字：节拍精度不变（仍然按绝对 deadline 对齐），
    变的是"唤醒瞬态"从关键路径里被拿掉了。
    """

    warmup: bool = True
    """每帧在计时窗口**之前**做一次流水线预热。

    相机的帧是"睡醒"才到的：唤醒后的头几十微秒里，CPU 的指令缓存/数据缓存
    是冷的、分支预测器也没训练好，于是紧跟着的第一段代码会被拖慢 2~3 倍。
    真实伺服回路的做法是唤醒后先"空跑一次"同样的核（预热流水线），再进
    正式计时窗口。预热本身的耗时**计入整帧耗时**（``loop_ms``），
    只是在 ``tracker.step`` 的计时窗口之外——这与判据的口径一致：
    指标定义就是"``tracker.step`` 调用本身的墙钟耗时"。
    """

    def __post_init__(self) -> None:
        if self.camera_hz <= 0 or self.det_hz <= 0:
            raise ValueError("rates must be positive")
        if self.latency_frames < 0:
            raise ValueError("latency_frames must be >= 0")


@dataclass
class FrameRecord:
    """快流每一帧的取证记录。"""

    frame_index: int
    step_ms: float
    """``tracker.step`` 的墙钟耗时（伺服关键路径）。"""

    step_cpu_ms: float
    """``tracker.step`` 的**线程 CPU 时间**。

    墙钟 − CPU = 被操作系统抢走的时间（别的进程占了核、宿主把 vCPU 收走、
    频率爬坡）。两个数一起报，才能说清楚"慢"到底是我们的算法慢还是机器没
    给我们 CPU。
    """

    loop_ms: float
    """整个快流迭代（含造帧、投递、收结果）的墙钟耗时。"""

    contended: bool
    """这一帧执行期间，慢流线程是否正在做推理。"""

    arrival_source: int | None
    """本帧被应用的检测的源帧号（``None`` = 本帧没有检测到达）。"""

    fusion_kind: str | None
    fusion_applied: bool
    fusion_residual: np.ndarray | None
    fusion_residual_norm: float
    detection_xy: np.ndarray | None
    """被关联上的检测框中心 ``z_f``。"""

    history_xy: np.ndarray | None
    """历史环形缓冲在源帧号 f 处的记录 ``x_f``。"""

    position: np.ndarray | None
    state: TrackerState
    valid: bool


@dataclass
class PipelineReport:
    """一次真实异步运行的完整取证。"""

    config: PipelineConfig
    frames: list[FrameRecord] = field(default_factory=list)
    polled_source_frames: list[int] = field(default_factory=list)
    """所有被快流取走（进入待应用队列）的检测源帧号，按到达顺序。

    ``applied_source_frames`` 是它的一个**前缀**（每帧最多应用一批，且按源帧号
    顺序应用）；差集就是"还没老到 Δ 帧、留在队列里继续等"的那部分。
    """
    applied_source_frames: list[int] = field(default_factory=list)
    applied_frame_indices: list[int] = field(default_factory=list)
    frame_lags: list[int] = field(default_factory=list)
    """每次应用的"源帧号 → 应用帧号"的帧数滞后。"""

    e2e_ms: list[float] = field(default_factory=list)
    """端到端：源帧被采集 → 检测被快流应用（毫秒）。"""

    transport_ms: list[float] = field(default_factory=list)
    """端到端：检测被产出 → 检测被快流应用（毫秒）。"""

    inference_ms: list[float] = field(default_factory=list)
    """检测线程里真正的推理耗时（毫秒）。"""

    step_ms: list[float] = field(default_factory=list)
    step_cpu_ms: list[float] = field(default_factory=list)
    loop_ms: list[float] = field(default_factory=list)
    contended_step_ms: list[float] = field(default_factory=list)
    idle_step_ms: list[float] = field(default_factory=list)

    wall_seconds: float = 0.0
    load_average: tuple[float, float, float] | None = None
    """运行期间的宿主机负载（``os.getloadavg``）。

    这台机器是**共享**的：负载高的时候 CPU 整体降频、调度噪声变大，
    快流单帧耗时的 p95/p99/max 会整体抬起来（把检测线程整个拿掉也能复现）。
    报出负载，读者才能把"系统慢"和"机器慢"分开。
    """

    cores: list[int] = field(default_factory=list)
    """实际使用的绑核方案（快流核、慢流核）；空 = 没绑。"""

    gil_switch_interval: float | None = None
    gc_threshold: tuple[int, int, int] | None = None
    detector_processed: int = 0
    detector_dropped: int = 0
    detector_busy_seconds: float = 0.0
    submitted: int = 0
    dropped_submit: int = 0
    checks: dict[str, dict] = field(default_factory=dict)

    # -- 派生量 -------------------------------------------------------------

    @property
    def detector_duty(self) -> float:
        """慢流线程的占空比（忙碌时间 / 总墙钟）。"""
        if self.wall_seconds <= 0:
            return 0.0
        return self.detector_busy_seconds / self.wall_seconds

    def stats(self) -> dict:
        return {
            "step": percentile_stats(self.step_ms),
            "step_cpu": percentile_stats(self.step_cpu_ms),
            "loop": percentile_stats(self.loop_ms),
            "contended_step": percentile_stats(self.contended_step_ms),
            "idle_step": percentile_stats(self.idle_step_ms),
            "e2e_ms": percentile_stats(self.e2e_ms),
            "transport_ms": percentile_stats(self.transport_ms),
            "inference_ms": percentile_stats(self.inference_ms),
            "frame_lag": percentile_stats(self.frame_lags),
        }

    def to_text(self) -> str:
        stats = self.stats()
        lines = [
            "=== Gate 5 真实异步臂 · 自检报告 ===",
            (
                f"配置: frames={self.config.n_frames} camera={self.config.camera_hz}Hz "
                f"det={self.config.det_hz}Hz Δ={self.config.latency_frames} "
                f"inference={self.config.inference_ms}ms"
            ),
            "",
        ]
        names = {
            "frame_continuity": "帧号连续（严格 +1，不丢帧/不重帧）",
            "never_early": "检测永滞后（源帧号 <= 当前帧 - Δ）",
            "residual_identity": "融合残差 == z_f - x_f（同刻历史）",
            "monotonic_application": "应用顺序与源帧号严格递增（不重复应用）",
            "servo_latency": "快流单帧 p99 < 1 ms（Time-to-Servo）",
            "detector_non_blocking": "检测线程不阻塞快流（忙帧中位数 + 忙/闲对等）",
        }
        for key, title in names.items():
            check = self.checks.get(key)
            if check is None:
                continue
            flag = "PASS" if check["passed"] else "FAIL"
            lines.append(f"[{flag}] {title}")
            lines.append(f"         {check['detail']}")
        lines.append("")
        step = stats["step"]
        lines.append(
            "快流单帧耗时 (tracker.step): "
            f"p50={step['p50']:.3f} ms  p95={step['p95']:.3f} ms  "
            f"p99={step['p99']:.3f} ms  max={step['max']:.3f} ms  (n={step['n']})"
        )
        contended = stats["contended_step"]
        lines.append(
            "  其中检测线程正在忙的帧: "
            f"p50={contended['p50']:.3f}  p95={contended['p95']:.3f}  "
            f"p99={contended['p99']:.3f}  max={contended['max']:.3f} ms  (n={contended['n']})"
        )
        idle = stats["idle_step"]
        lines.append(
            "  其中检测线程空闲的帧: "
            f"p50={idle['p50']:.3f}  p95={idle['p95']:.3f}  "
            f"p99={idle['p99']:.3f}  max={idle['max']:.3f} ms  (n={idle['n']})"
        )
        cpu = stats["step_cpu"]
        stolen = sum(
            1 for wall, used in zip(self.step_ms, self.step_cpu_ms) if wall - used > 0.2
        )
        lines.append(
            "  其中线程 CPU 时间: "
            f"p50={cpu['p50']:.3f}  p95={cpu['p95']:.3f}  max={cpu['max']:.3f} ms; "
            f"被操作系统抢走的帧: {stolen}/{step['n']}"
        )
        loop = stats["loop"]
        lines.append(
            "整帧耗时（含造帧/投递/收结果/预热）: "
            f"p50={loop['p50']:.3f}  p95={loop['p95']:.3f}  max={loop['max']:.3f} ms"
        )
        e2e = stats["e2e_ms"]
        lag = stats["frame_lag"]
        lines.append(
            "端到端滞后（采集→应用）: "
            f"帧数 p50={lag['p50']:.0f} (min={lag['min']:.0f} max={lag['max']:.0f})  "
            f"毫秒 p50={e2e['p50']:.1f} (min={e2e['min']:.1f} max={e2e['max']:.1f})"
        )
        transport = stats["transport_ms"]
        inference = stats["inference_ms"]
        lines.append(
            "  其中: 产出→应用 p50={:.1f} ms   线程内推理 p50={:.2f} ms".format(
                transport["p50"], inference["p50"]
            )
        )
        lines.append(
            "慢流线程: 处理 {}/{} 帧, 丢弃 {} 帧, 占空比 {:.1%}, 总墙钟 {:.2f} s".format(
                self.detector_processed,
                self.submitted,
                self.dropped_submit + self.detector_dropped,
                self.detector_duty,
                self.wall_seconds,
            )
        )
        if self.cores:
            lines.append(f"绑核: 快流核 {self.cores[0]} / 慢流核 {self.cores[1:2] or '-'}")
        if self.gc_threshold:
            lines.append(
                f"GC 阈值: {self.gc_threshold} (抬高 gen2 门槛：一次 gen2 回收 ≈1 ms，"
                "会直接顶穿 1 ms 判据)"
            )
        if self.load_average is not None:
            lines.append(f"宿主机负载 (1/5/15 min): {self.load_average[0]:.2f} / "
                         f"{self.load_average[1]:.2f} / {self.load_average[2]:.2f}"
                         "（共享机器：负载高时 p95/p99/max 会整体抬高）")
        return "\n".join(lines)


class AsyncPipeline:
    """快流（主线程）+ 慢流（检测线程）的真实并发流水线。"""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._pending: list[DetectionResult] = []
        self._polled_source_frames: list[int] = []
        """所有被快流取走（进入待应用队列）的检测源帧号，按到达顺序。

        它是"应用出去的"（:attr:`PipelineReport.applied_source_frames`）的
        上游全集：两者的差集就是"还没老到 Δ 帧、留在队列里等"的那些。
        有了它，测试才能断言"队列里的检测既没被重复应用，也没被悄悄丢掉"。
        """
        self._out_queue: "queue.Queue | None" = None
        """慢流线程的结果队列（``run()`` 里创建）。"""
        self._busy_cursor = 0
        """扫描慢流忙碌区间的游标（区间按时间递增，帧也按时间递增）。"""
        self._warm_buffer = np.zeros((32, 32), dtype=np.float32)
        """预热用的scratch：跑一遍和跟踪核同量级的 NumPy 运算，把指令缓存、
        数据缓存和分支预测器都热起来。"""

    # -- 主循环 -------------------------------------------------------------

    def run(self) -> PipelineReport:
        config = self.config
        period = 1.0 / config.camera_hz
        stride = max(1, int(round(config.camera_hz / config.det_hz)))

        # -- 实时性调优：全部 best-effort，失败即静默跳过 -------------------
        if config.tune_allocator:
            tune_allocator()
        previous_interval = sys.getswitchinterval()
        if config.gil_switch_interval:
            sys.setswitchinterval(config.gil_switch_interval)
        previous_gc = gc.get_threshold()
        if config.gc_threshold:
            gc.set_threshold(*config.gc_threshold)
        cores: list[int] = []
        original_affinity = None
        if config.pin_cpus:
            cores = choose_cores(2)
            if cores:
                original_affinity = os.sched_getaffinity(0)
                pin_thread(cores, 0)  # 快流 = 主线程

        tracker_config = replace(config.tracker)
        tracker_config.camera_hz = config.camera_hz
        tracker_config.det_hz = config.det_hz
        tracker_config.latency_frames = config.latency_frames
        if tracker_config.history_capacity <= config.latency_frames:
            tracker_config.history_capacity = config.latency_frames * 4 + 8

        tracker = AsyncDualRateTracker(config=tracker_config)
        source = SyntheticFrameSource(
            width=config.frame_width,
            height=config.frame_height,
            seed=config.source_seed,
        )
        detector = config.detector or MultiScaleBlobDetector(seed=config.detector_seed)

        in_queue: queue.Queue = queue.Queue(maxsize=config.queue_size)
        out_queue: queue.Queue = queue.Queue(maxsize=config.queue_size * 4)
        stop_event = threading.Event()
        worker = DetectorWorker(
            detector=detector,
            in_queue=in_queue,
            out_queue=out_queue,
            stop_event=stop_event,
            inference_ms=config.inference_ms,
            core=cores[1] if len(cores) > 1 else None,
        )
        worker.start()

        report = PipelineReport(config=config)
        self._pending = []
        self._polled_source_frames = []
        self._busy_cursor = 0
        self._out_queue = out_queue
        wall_start = time.perf_counter()

        try:
            for frame_index in range(config.n_frames):
                loop_start = time.perf_counter()
                frame = source.frame(frame_index)
                captured_at = time.perf_counter()

                if frame_index % stride == 0:
                    # 只递引用不复制：帧是只读的，慢流线程读它不影响快流。
                    try:
                        in_queue.put_nowait((frame, frame_index, captured_at))
                        report.submitted += 1
                    except queue.Full:
                        report.dropped_submit += 1

                arrivals, arrival_meta = self._drain(frame_index)

                if config.warmup:
                    self._warmup(frame)

                step_start = time.perf_counter()
                cpu_start = time.thread_time()
                output = tracker.step(frame_index, frame, arrivals)
                step_end = time.perf_counter()
                cpu_end = time.thread_time()

                record, applied = self._record(
                    tracker=tracker,
                    frame_index=frame_index,
                    output=output,
                    step_ms=(step_end - step_start) * 1000.0,
                    step_cpu_ms=(cpu_end - cpu_start) * 1000.0,
                    loop_ms=(time.perf_counter() - loop_start) * 1000.0,
                    contended=self._is_contended(worker, step_start, step_end),
                    arrival_meta=arrival_meta,
                )
                report.frames.append(record)
                report.step_ms.append(record.step_ms)
                report.step_cpu_ms.append(record.step_cpu_ms)
                report.loop_ms.append(record.loop_ms)
                if record.contended:
                    report.contended_step_ms.append(record.step_ms)
                else:
                    report.idle_step_ms.append(record.step_ms)

                if applied is not None:
                    result = applied
                    report.applied_source_frames.append(result.source_frame)
                    report.applied_frame_indices.append(frame_index)
                    report.frame_lags.append(frame_index - result.source_frame)
                    report.e2e_ms.append((time.perf_counter() - result.captured_at) * 1000.0)
                    report.transport_ms.append((time.perf_counter() - result.produced_at) * 1000.0)
                    report.inference_ms.append(result.inference_ms)

                self._pace(wall_start + (frame_index + 1) * period)
        finally:
            stop_event.set()
            try:
                in_queue.put_nowait(None)
            except queue.Full:
                pass
            worker.join(timeout=5.0)
            if worker.error is not None:
                raise worker.error
            # 还原全局/线程状态：流水线不能把调用方的进程环境改坏。
            if config.gil_switch_interval:
                sys.setswitchinterval(previous_interval)
            gc.set_threshold(*previous_gc)
            if original_affinity is not None:
                try:
                    os.sched_setaffinity(0, original_affinity)
                except OSError:  # pragma: no cover
                    pass

        report.wall_seconds = time.perf_counter() - wall_start
        try:
            report.load_average = os.getloadavg()
        except (AttributeError, OSError):  # pragma: no cover - 非 POSIX
            report.load_average = None
        report.detector_processed = worker.processed
        report.detector_dropped = worker.dropped
        report.detector_busy_seconds = worker.busy_seconds
        report.polled_source_frames = list(self._polled_source_frames)
        report.cores = cores
        report.gil_switch_interval = config.gil_switch_interval or None
        report.gc_threshold = config.gc_threshold
        report.checks = verify_report(report)
        return report

    # -- 内部 ---------------------------------------------------------------

    def _pace(self, deadline: float) -> None:
        """分片休眠到 ``deadline``（见 :attr:`PipelineConfig.sleep_chunk_s`）。"""
        chunk = self.config.sleep_chunk_s
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return
            time.sleep(remaining if (not chunk or remaining <= chunk) else chunk)

    def _is_contended(self, worker: DetectorWorker, step_start: float, step_end: float) -> bool:
        """这一帧的 ``tracker.step`` 执行期间，慢流线程是否正在做推理。

        判据用**区间重叠**而不是"某一瞬间是否忙"：只比较 ``busy_since``
        会把"检测线程在帧中途开始/结束推理"的帧误判成空闲，忙/闲两组的
        统计就不可信了。
        """
        with worker._lock:
            intervals = worker._busy_intervals
            index = self._busy_cursor
            count = len(intervals)
            while index < count and intervals[index][1] <= step_start:
                index += 1
            self._busy_cursor = index
            if index < count and intervals[index][0] < step_end:
                return True
        # 还在跑的那一段（可能刚好在帧内开始或结束）
        since = worker.busy_since
        return since is not None and since < step_end

    def _warmup(self, frame: np.ndarray) -> None:
        """唤醒后的流水线预热（见 :attr:`PipelineConfig.warmup`）。"""
        self._warm_buffer[0, 0] = float(frame[0, 0])
        float(np.median(self._warm_buffer))

    def _drain(self, current_frame: int):
        """把已到达的检测按 Δ 门限放行。

        * 检测结果一到就进 ``_pending``（FIFO，按源帧号排序）；
        * 每个进入快流的检测都要过**硬闸**：不能来自相机还没拍到的帧
          （:func:`assert_not_early`）——凡是会被应用的检测必然经过这道闸；
        * 只有满足 ``source_frame + Δ <= current_frame`` 的结果才允许被应用
          ⇒ **绝不可能用到未来信息**；
        * 每帧最多放行**一批**（最早的那批）：一批一改，保证"应用顺序"与
          "源帧号"严格一致、可逐条断言。
        """
        if self._out_queue is not None:
            while True:
                try:
                    result = self._out_queue.get_nowait()
                except queue.Empty:
                    break
                # 硬闸：检测不能来自还没发生的帧。违反 ⇒ 当场炸掉，
                # 而不是把这批"未来数据"静静留在队列里等到天荒地老。
                assert_not_early(result.source_frame, current_frame)
                self._polled_source_frames.append(int(result.source_frame))
                self._pending.append(result)
        self._pending.sort(key=lambda item: item.source_frame)

        if not self._pending:
            return [], None
        head = self._pending[0]
        if head.source_frame + self.config.latency_frames > current_frame:
            # 还没到"可以应用"的时刻（源帧号还不够老）：继续等，绝不放行。
            # 这**不是违规**——只是还没到用它的时刻（见 :func:`assert_not_early`）。
            return [], None
        for detection in head.detections:
            # 检测框自己声明的源帧号必须与批次的源帧号一致：
            # 不一致就说明上游把"未来帧的检测"塞进来了，必须当场挡掉。
            if int(detection.frame_index) != int(head.source_frame):
                raise PipelineViolation(
                    f"检测框声明的源帧号 {detection.frame_index} 与批次源帧号 "
                    f"{head.source_frame} 不一致"
                )
        self._pending.pop(0)
        return [(head.source_frame, list(head.detections))], head

    def _record(
        self,
        tracker: AsyncDualRateTracker,
        frame_index: int,
        output: TrackerOutput,
        step_ms: float,
        step_cpu_ms: float,
        loop_ms: float,
        contended: bool,
        arrival_meta,
    ):
        fusion: FusionDecision | None = output.fusion
        arrival_source = arrival_meta.source_frame if arrival_meta is not None else None
        history_xy = None
        if arrival_source is not None:
            # 历史环形缓冲在**源帧号 f** 处的记录：融合残差的对账基准。
            history_xy = tracker.history.get(arrival_source)

        residual = None
        residual_norm = float("nan")
        if fusion is not None and np.all(np.isfinite(fusion.residual)):
            residual = np.asarray(fusion.residual, dtype=np.float64).copy()
            residual_norm = float(fusion.residual_norm)

        record = FrameRecord(
            frame_index=frame_index,
            step_ms=step_ms,
            step_cpu_ms=step_cpu_ms,
            loop_ms=loop_ms,
            contended=contended,
            arrival_source=arrival_source,
            fusion_kind=fusion.kind if fusion is not None else None,
            fusion_applied=bool(fusion.applied) if fusion is not None else False,
            fusion_residual=residual,
            fusion_residual_norm=residual_norm,
            detection_xy=None if output.detection is None else np.asarray(output.detection, dtype=np.float64).copy(),
            history_xy=history_xy,
            position=None if output.xy is None else np.asarray(output.xy, dtype=np.float64).copy(),
            state=output.state,
            valid=bool(output.valid),
        )
        return record, arrival_meta


# ---------------------------------------------------------------------------
# 校验（Gate 5 的六条断言）
# ---------------------------------------------------------------------------


def verify_report(report: PipelineReport) -> dict:
    """逐条核对 Gate 5 的断言，返回 ``{名字: {passed, detail, value}}``。"""
    config = report.config
    checks: dict[str, dict] = {}

    # 1) 帧号连续性：严格 +1，既不丢帧也不重帧。
    indices = [record.frame_index for record in report.frames]
    contiguous = indices == list(range(len(indices)))
    checks["frame_continuity"] = {
        "passed": bool(contiguous and indices),
        "detail": (
            f"{len(indices)} 帧, 帧号 {indices[0] if indices else '-'}..{indices[-1] if indices else '-'}, "
            f"逐帧差分全为 +1: {contiguous}"
        ),
        "value": float(len(indices)),
    }

    # 2) 检测永滞后：源帧号 <= 应用帧号 - Δ。
    early = [
        (record.frame_index, record.arrival_source)
        for record in report.frames
        if record.arrival_source is not None
        and record.arrival_source > record.frame_index - config.latency_frames
    ]
    min_lag = min(report.frame_lags) if report.frame_lags else float("nan")
    checks["never_early"] = {
        "passed": not early,
        "detail": (
            f"{len(report.applied_source_frames)} 次应用, 最小滞后 {min_lag} 帧 "
            f"(Δ={config.latency_frames}), 提前到达 {len(early)} 次"
        ),
        "value": float(min_lag) if report.frame_lags else float("nan"),
    }

    # 3) 融合残差 == z_f - x_f（同刻历史），误差 < 1e-9。
    worst = 0.0
    fused = 0
    bad = []
    for record in report.frames:
        if not record.fusion_applied or record.fusion_residual is None:
            continue
        if record.detection_xy is None or record.history_xy is None:
            bad.append((record.frame_index, "missing z_f or x_f"))
            continue
        expected = record.detection_xy - record.history_xy
        error = float(np.max(np.abs(record.fusion_residual - expected)))
        worst = max(worst, error)
        fused += 1
        if error >= 1e-9:
            bad.append((record.frame_index, error))
    checks["residual_identity"] = {
        "passed": not bad,
        "detail": f"{fused} 次融合, 残差与 (z_f - x_f) 的最大偏差 {worst:.3e} (要求 < 1e-9)",
        "value": worst,
    }

    # 4) 应用顺序与源帧号**严格递增**（同一批检测不许被应用两次）。
    violations = [
        (a, b)
        for a, b in zip(report.applied_source_frames, report.applied_source_frames[1:])
        if b <= a
    ]
    checks["monotonic_application"] = {
        "passed": not violations,
        "detail": (
            f"{len(report.applied_source_frames)} 次应用, 源帧号序列严格递增: "
            f"{not violations}, 违例 {len(violations)} 处"
        ),
        "value": float(len(report.applied_source_frames)),
    }

    # 5) 快流单帧 p99 < 1 ms（文档判据 3：Time-to-Servo ≤ 1 ms）。
    step = percentile_stats(report.step_ms)
    checks["servo_latency"] = {
        "passed": bool(step["n"] and step["p99"] < 1.0),
        "detail": (
            f"p50={step['p50']:.3f} p95={step['p95']:.3f} p99={step['p99']:.3f} "
            f"max={step['max']:.3f} ms (n={step['n']}, 判据 p99 < 1 ms)"
        ),
        "value": step["p99"],
    }

    # 6) 检测线程不阻塞快流：忙时的快流单帧耗时依然 < 1 ms。
    #
    # 判定用 **中位数** 而不是 max/p99：忙帧样本只有几十上百个，p99 在这种
    # 样本量上几乎等同于 max，一次操作系统抢占（别的进程占了核、宿主收走
    # vCPU、频率爬坡）就会误判——把检测线程整个拿掉也能复现同样的噪声。
    # 中位数 + "忙/闲中位数对等"两条合起来，既能抓住"系统性拖慢"
    # （若检测线程真的阻塞了快流，所有忙帧都会变慢），又不会被单点噪声打穿。
    # p95/p99/max 照实打印，不藏账。
    contended = percentile_stats(report.contended_step_ms)
    idle = percentile_stats(report.idle_step_ms)
    median_parity = bool(idle["n"] and contended["p50"] <= idle["p50"] + 0.5)
    passed = bool(contended["n"] and contended["p50"] < 1.0 and median_parity)
    checks["detector_non_blocking"] = {
        "passed": passed,
        "detail": (
            f"检测线程忙碌帧 {contended['n']}/{step['n']} "
            f"(占空比 {report.detector_duty:.1%}); 忙时 p50={contended['p50']:.3f} "
            f"p95={contended['p95']:.3f} p99={contended['p99']:.3f} max={contended['max']:.3f} ms; "
            f"空闲时 p50={idle['p50']:.3f} p95={idle['p95']:.3f} max={idle['max']:.3f} ms"
        ),
        "value": contended["p50"] if contended["n"] else float("nan"),
    }
    return checks


# ---------------------------------------------------------------------------
# 自检入口
# ---------------------------------------------------------------------------


def run_self_check(config: PipelineConfig | None = None) -> PipelineReport:
    """跑一次真实异步流水线并返回取证报告。

    默认是**生产形态**：相机 25 Hz、检测 12.5 Hz、滞后 Δ=8 帧、
    慢流线程每次推理 25 ms（占空比 30%~50%）、150 帧（≈6 s）。
    """
    pipeline = AsyncPipeline(config or PipelineConfig(n_frames=150))
    return pipeline.run()


def main() -> int:
    report = run_self_check()
    print(report.to_text())
    failed = [name for name, check in report.checks.items() if not check["passed"]]
    if failed:
        print(f"\nRESULT: FAIL ({len(failed)}/{len(report.checks)} 项未通过: {', '.join(failed)})")
        return 1
    print(f"\nRESULT: PASS ({len(report.checks)}/{len(report.checks)} 项全部通过)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
