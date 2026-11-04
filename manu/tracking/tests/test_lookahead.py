"""look-ahead 守卫：第 t 帧的输出**不得**依赖任何 frame > t 的数据。

这是异步检跟分离系统最容易写出、也最致命的一类 bug：后台检测器滞后 Δ 帧
返回，如果实现里不小心用了"未来"的图像或"未来"的检测框，本序列的精度数字
就会凭空变好，**C++ 移植后按同样语义实现却得不到同样的数**。

本模块用**两条互相独立的守卫**把这个不变量钉死：

1. **功能守卫（变异式）**：把 frame > T 的图像全部换成噪声，重跑；
   第 <= T 帧的输出必须**逐位不变**。
2. **结构守卫（接口式）**：给重推演喂一个"读到越界帧就抛异常"的帧缓存，
   证明重推区间真的是 [f+1, t) 而不是 [f+1, t] 或更长。

两条守卫都必须是**能变红**的：任何一处引入未来信息，它们立刻失败。
"""

from __future__ import annotations

import unittest

import numpy as np

from manu.tracking.async_dual_rate.arms import ArmSpec, SyntheticFrameSource, run_lockstep
from manu.tracking.async_dual_rate.detector import Detection
from manu.tracking.async_dual_rate.fast_tracker import LocalPatchTracker
from manu.tracking.async_dual_rate.replay import FrameRingBuffer, replay_forward
from manu.tracking.async_dual_rate.tracker import TrackerConfig


class _TruthDetector:
    """报真值位置的检测器（合成臂专用）。

    它只按**源帧号**给框，因此天然不含未来信息——look-ahead 只可能来自
    **图像侧**或**重推区间**。
    """

    def __init__(self, source, score: float = 0.9) -> None:
        self.source = source
        self.score = float(score)
        self.frames = getattr(source, "frames", 0)

    def detections(self, frame_index: int) -> list[Detection]:
        position = self.source.position(int(frame_index))
        return [
            Detection(
                frame_index=int(frame_index),
                xy=np.asarray(position, dtype=np.float64),
                score=self.score,
            )
        ]


class _FutureCorruptingSource:
    """把 frame_index > cutoff 的图像换成**确定性噪声**的帧源。

    噪声是确定性的（固定种子），所以两次运行之间唯一的差别就是
    "未来帧的内容变了"。若实现里存在 look-ahead，第 <= cutoff 帧的输出
    就会随之改变。
    """

    def __init__(self, inner, cutoff: int) -> None:
        self.inner = inner
        self.cutoff = int(cutoff)
        self.frames = getattr(inner, "frames", 0)
        self.reads: list[int] = []

    def position(self, frame_index: int) -> tuple[float, float]:
        return self.inner.position(frame_index)

    def read(self, frame_index: int) -> np.ndarray | None:
        self.reads.append(int(frame_index))
        if int(frame_index) <= self.cutoff:
            return self.inner.read(frame_index)
        rng = np.random.default_rng(1000 + int(frame_index))
        height, width = self.inner.height, self.inner.width
        return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


def _specs(latency: int) -> list[ArmSpec]:
    """三条主臂（alpha / replay / none）+ 纯检测基准，全部走同一次遍历。"""
    return [
        ArmSpec(
            name="arm1_detector_only",
            kind="detector_only",
            det_hz=25.0,
            latency_frames=latency,
        ),
        ArmSpec(
            name="arm2_forward_only",
            det_hz=12.5,
            latency_frames=latency,
            fusion_mode="none",
            detection_reinit=False,
        ),
        ArmSpec(
            name="arm3_rollback_replay",
            det_hz=12.5,
            latency_frames=latency,
            fusion_mode="replay",
        ),
        ArmSpec(
            name="ablation_fused_alpha",
            det_hz=12.5,
            latency_frames=latency,
            fusion_mode="alpha",
        ),
    ]


class LookAheadGuardTest(unittest.TestCase):
    """功能守卫：未来帧内容变化 => 过去帧输出必须逐位不变。"""

    def _run(self, source, latency: int = 4, total_frames: int = 120):
        base = TrackerConfig()
        base.latency_frames = latency
        base.camera_hz = 25.0
        return run_lockstep(
            source,
            _TruthDetector(source),
            _specs(latency),
            base_config=base,
            total_frames=total_frames,
            camera_hz=25.0,
        )

    def test_outputs_up_to_cutoff_are_bit_identical(self):
        frames = 120
        cutoff = 60
        plain = SyntheticFrameSource(frames=frames, velocity=(0.8, 0.3))
        reference = self._run(plain, total_frames=frames)

        corrupting = _FutureCorruptingSource(
            SyntheticFrameSource(frames=frames, velocity=(0.8, 0.3)), cutoff
        )
        mutated = self._run(corrupting, total_frames=frames)

        self.assertTrue(corrupting.reads, "帧源一次都没被读过，测试没跑起来")
        self.assertTrue(
            any(frame > cutoff for frame in corrupting.reads),
            "未来帧没有被读取，变异无效",
        )

        for name, reference_outcome in reference.items():
            mutated_outcome = mutated[name]
            for frame_index, position in reference_outcome.positions.items():
                if frame_index > cutoff:
                    continue
                self.assertIn(
                    frame_index, mutated_outcome.positions, f"{name} 第 {frame_index} 帧输出丢了"
                )
                self.assertTrue(
                    np.array_equal(position, mutated_outcome.positions[frame_index]),
                    f"{name} 第 {frame_index} 帧的输出依赖了未来帧（look-ahead）",
                )

    def test_future_corruption_actually_changes_the_future(self):
        """自证：变异确实改掉了未来帧的内容（否则上面的守卫是空转）。"""
        frames = 60
        cutoff = 30
        plain = SyntheticFrameSource(frames=frames, velocity=(0.8, 0.3))
        corrupting = _FutureCorruptingSource(
            SyntheticFrameSource(frames=frames, velocity=(0.8, 0.3)), cutoff
        )
        future_plain = plain.read(cutoff + 1)
        future_corrupt = corrupting.read(cutoff + 1)
        self.assertFalse(
            np.array_equal(future_plain, future_corrupt),
            "变异没有真正改掉未来帧，守卫会空转",
        )


class ReplayWindowGuardTest(unittest.TestCase):
    """结构守卫：重推演只允许读 [f+1, t) 区间内的帧。"""

    def test_replay_never_reads_the_target_frame_or_later(self):
        class _StrictCache(FrameRingBuffer):
            """记录所有被取过的帧号；取越界帧直接抛异常。"""

            def __init__(self, capacity: int = 16, allowed_until: int = 0) -> None:
                super().__init__(capacity)
                self.allowed_until = int(allowed_until)
                self.requested: list[int] = []

            def get(self, frame_index: int):
                self.requested.append(int(frame_index))
                if int(frame_index) > self.allowed_until:
                    raise AssertionError(
                        f"重推演请求了未来帧 {frame_index}（允许上界 {self.allowed_until}）"
                    )
                return super().get(frame_index)

        frames = 40
        source = SyntheticFrameSource(frames=frames, velocity=(0.8, 0.3))
        target_frame = 20
        anchor_frame = 12
        cache = _StrictCache(capacity=16, allowed_until=target_frame - 1)
        for index in range(anchor_frame, target_frame):
            cache.push(index, source.read(index))

        kernel = LocalPatchTracker(
            win_size=32, th_k=3.0, min_energy=1.0, min_contrast=5.0, mode="centroid", bg_window=64
        )
        anchor_xy = np.asarray(source.position(anchor_frame), dtype=np.float64)
        result = replay_forward(
            kernel,
            cache,
            anchor_xy,
            anchor_frame=anchor_frame,
            target_frame=target_frame,
        )
        self.assertTrue(result.applied, f"重推演应当成功，实际 reason={result.reason!r}")
        self.assertTrue(cache.requested, "重推演一次都没读帧，测试没跑起来")
        self.assertTrue(
            all(frame < target_frame for frame in cache.requested),
            f"重推演读到了 target_frame 或之后：{cache.requested}",
        )
        self.assertEqual(result.steps, target_frame - anchor_frame - 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
