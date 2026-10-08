#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Streaming FIFO three-channel feature engine -- Golden Reference for the RK3588 port.

Contract (locked to what Trial 0474 was trained on):

    Ch0 = I_t
    Ch1 = |I_t - W(I_{t-2})|                (lag 2 exactly; never 1, never 4)
    Ch2 = (I_t - B_t)^+,  B_t = median{W(I_{t-2k})}, k = 1..21

    output = np.ndarray, shape (3, H, W), dtype uint8, channel order [Ch0, Ch1, Ch2]

This is a pure online / causal engine: it keeps only a fixed FIFO state and never looks at a
future frame. It reproduces, bit for bit, the offline ``build_nogmc_median_dataset.py
--mode tree --anchor-step 10`` output, which is the feature set Trial 0474 is currently
evaluated against. Bit-equality is obtained by construction, not by re-derivation:

* the GMC estimator, the 3x3 similarity compose and the anchor grid are **imported** from the
  offline builders, so both paths execute the same code;
* the only thing this module re-implements is the *state management* (ring buffers), which is
  exactly what the streaming refactor is meant to change.

FIFO state
    frames : the last ``max_lag + 1`` **consecutive** frames, as a circular buffer indexed by
            absolute frame index modulo depth (43 slots = 14.1 MiB at 640x512).
    steps  : stride-step transforms H_{j -> j+s}, keyed by absolute index j and pruned to max_lag;
            about 20 live entries. Keyed by absolute index rather than by lag because one fit is
            reused by several consecutive frames.

Why the frame ring must hold every consecutive frame, not 22 strided slots
    A 22-slot stride-2 ring looks sufficient because the history lags are 2,4,...,42, but **every
    frame is itself the current frame of its own push**. Frame f is needed at lag 2j by frames
    f+2j, f+4j, ..., i.e. it must survive up to ``max_lag`` pushes while odd-indexed frames are
    still being pushed. So all frames must be retained and the depth is ``max_lag + 1``.
    A second trap lives in the same place: shifting the ring by one slot per push indexes it by
    *one-frame* lag while lookups stride by ``stride_step``, which silently fetches every history
    frame at half the intended lag. The ring is therefore indexed by absolute frame index, never
    by lag and never by shifting.

Determinism
    ``cv2.setNumThreads(1)`` is mandatory and is set in ``__init__``. The offline builder pins it too
    (``build_nogmc_median_dataset.py``: every worker calls it before touching frames). Under multiple
    OpenCV threads, ``estimateAffinePartial2D``'s RANSAC and Lucas-Kanade are not bit-reproducible:
    an anchor fit can flip between two equally plausible solutions, which shifts a whole warped edge
    and shows up as max|d| in the hundreds. The C++ port must be single-thread-deterministic for the
    same reason.

Anchors
    ``anchor_grid(s, anchor_step, max_lag)`` for the frozen lags 2..42 with anchor_step=10 gives
    {2, 12, 22, 32, 42}. Anchors are lag-based, so frame ``t`` directly fits against
    ``t-2, t-12, t-22, t-32, t-42``. The non-anchor lags are reached by
    ``R_L = p_{t-L} . p_{t-L+s} . ... . p_{t-a-s} . R_a`` with ``a`` the largest anchor <= L, so
    the composition depth is ``(L - a) / s <= anchor_step / s - 1`` (= 4 here). Composition runs
    from the anchor outward, i.e. decreasing lag_abs, because the factor nearest R_a is applied
    first. Note this is NOT a periodic "every N frames" schedule: the anchor set is a property of
    the lag grid, not of the frame index.

Cold start
    Lags that predate the first frame resolve to a **replica of frame 0**, and the corresponding
    GMC fits are still executed against that replica -- exactly what the offline builder does via
    ``read(i) = max(0, min(i, n-1))``. Output shape is therefore ``(3, H, W)`` from the very first
    frame onward.

Usage (server):
    PYTHONPATH=. python manu/pipeline/streaming_feature_pipeline.py          # built-in smoke test
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterator, Protocol

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_nogmc_median_dataset import _compose, anchor_grid  # noqa: E402
from manu.data.build_sample_median_dataset import (  # noqa: E402
    IMAGE_SUFFIXES,
    FastGMCEstimator,
    find_sequence_folder,
    natural_key,
)

IDENTITY = np.eye(2, 3, dtype=np.float32)


def fit_similarity(estimator: FastGMCEstimator, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Identical to the offline builder's ``fit_matrix``: estimate, then guard non-finite output."""
    m = estimator.compute_affine(prev, curr)
    if m is None or not np.isfinite(m).all():
        return IDENTITY.copy()
    return m.astype(np.float32)


class FrameSource(Protocol):
    """Any iterable of grayscale frames."""

    def __iter__(self) -> Iterator[np.ndarray]: ...


class SequenceFrameSource:
    """Raw-frame directory whose ordering is identical to the offline builder's.

    The offline datasets are keyed by ``<sequence>__<frame number>.jpg`` and their builders
    resolve each name to a list index through ``idx_map``. Reproducing that ordering here is what
    makes a streaming run comparable frame-for-frame with the frozen feature set.
    """

    def __init__(self, raw_root: str | Path, sequence: str, limit: int = 0) -> None:
        seq_dir = find_sequence_folder(Path(raw_root), sequence, {})
        if seq_dir is None:
            raise FileNotFoundError(f"sequence folder not found for {sequence!r} under {raw_root}")
        frames = [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        frames.sort(key=natural_key)
        if limit > 0:
            frames = frames[:limit]
        if not frames:
            raise FileNotFoundError(f"no frames in {seq_dir}")
        self.sequence = sequence
        self.directory = seq_dir
        self.paths = frames

    def __iter__(self) -> Iterator[np.ndarray]:
        for p in self.paths:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"failed to read {p}")
            yield img


class VideoFrameSource:
    """Sequential video reader.

    NOTE: ``VideoCapture`` yields BGR, so BGR->GRAY conversion here is **not** bit-comparable to a
    directory source that reads original grayscale files. Use a directory source whenever the run
    has to be reconciled against a frozen dataset.
    """

    def __init__(self, path: str | Path, limit: int = 0) -> None:
        self.path = Path(path)
        self.limit = limit
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"cannot open video {self.path}")
        cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"cannot open video {self.path}")
        try:
            emitted = 0
            while self.limit <= 0 or emitted < self.limit:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                emitted += 1
        finally:
            cap.release()


class OnlineFeaturePipeline:
    """Causal streaming producer of ``[Ch0, Ch1, Ch2]`` for Trial 0474."""

    def __init__(
        self,
        window: int = 21,
        stride_step: int = 2,
        anchor_step: int = 10,
        downscale: int = 2,
        expected_shape: tuple[int, int] | None = None,  # (H, W), numpy order
        dump_intermediate: bool = False,
        dump_dir: str | Path | None = None,
    ) -> None:
        self.window = window
        self.stride_step = stride_step
        self.max_lag = window * stride_step
        self.anchors = anchor_grid(stride_step, anchor_step, self.max_lag)
        self._anchor_set = set(self.anchors)
        self.lags = list(range(stride_step, self.max_lag + 1, stride_step))
        self._anchor_below = {
            lag: max(a for a in self.anchors if a <= lag) for lag in self.lags if lag not in self._anchor_set
        }
        cv2.setNumThreads(1)
        self.estimator = FastGMCEstimator(downscale=downscale)
        self.ring_depth = self.max_lag + 1
        self._ring: list[np.ndarray | None] = [None] * self.ring_depth
        self._steps: dict[int, np.ndarray] = {}
        self._first: np.ndarray | None = None
        self._shape = expected_shape
        self._t = -1
        self.last_mats: dict[int, np.ndarray] | None = None
        self.dump_intermediate = dump_intermediate
        self.dump_dir = Path(dump_dir) if dump_dir else None
        if self.dump_intermediate and self.dump_dir is None:
            raise ValueError("dump_intermediate=True requires dump_dir")
        if self.dump_intermediate:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            self._dump_meta()

    # ------------------------------------------------------------------ state introspection
    def state_bytes(self) -> int:
        """Bytes held by the live FIFO state, for the embedded memory budget."""
        held = [f for f in self._ring if f is not None]
        return sum(f.nbytes for f in held) + sum(m.nbytes for m in self._steps.values())

    def reset(self) -> None:
        self._ring = [None] * self.ring_depth
        self._steps.clear()
        self._first = None
        self._t = -1

    # ------------------------------------------------------------------ frame access
    def _as_gray(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype != np.uint8:
            raise TypeError(f"frames must be uint8 grayscale, got {arr.dtype}")
        if arr.ndim != 2:
            raise ValueError(f"frames must be 2-D, got shape {arr.shape}")
        if self._shape is None:
            self._shape = arr.shape
        elif arr.shape != self._shape:
            raise ValueError(f"frame size changed: got (H,W)={arr.shape}, expected (H,W)={self._shape}")
        return arr

    def _frame_at_lag(self, lag: int) -> np.ndarray:
        """Frame at ``lag`` behind the current one; cold-start slots resolve to a frame-0 replica.

        The ring is indexed by **absolute frame index modulo depth**, never by lag. Shifting a list
        by one slot per push would index by *one-frame* lag while lag lookups are strides of
        ``stride_step``, which silently fetches every history frame at half the intended lag.
        """
        idx = self._t - lag
        if idx <= 0:
            return self._first
        frame = self._ring[idx % self.ring_depth]
        return frame if frame is not None else self._first

    def _step(self, abs_index: int) -> np.ndarray:
        """Stride-step transform ``H_{abs_index -> abs_index + stride_step}``, cached by absolute index."""
        m = self._steps.get(abs_index)
        if m is None:
            m = fit_similarity(
                self.estimator,
                self._frame_at_lag(self._t - abs_index),
                self._frame_at_lag(self._t - abs_index - self.stride_step),
            )
            self._steps[abs_index] = m
        return m

    # ------------------------------------------------------------------ transforms
    def _transforms(self, curr: np.ndarray) -> dict[int, np.ndarray]:
        mats: dict[int, np.ndarray] = {}
        for lag in self.anchors:
            mats[lag] = fit_similarity(self.estimator, self._frame_at_lag(lag), curr)
        for lag in self.lags:
            if lag in self._anchor_set:
                continue
            below = self._anchor_below[lag]
            acc = mats[below]
            for lag_abs in range(below + self.stride_step, lag + 1, self.stride_step):
                acc = _compose(self._step(self._t - lag_abs), acc)
            mats[lag] = acc
        return mats

    # ------------------------------------------------------------------ main entry
    def push(self, frame: np.ndarray) -> np.ndarray:
        """Consume one grayscale frame, return ``(3, H, W)`` uint8 ``[Ch0, Ch1, Ch2]``."""
        curr = self._as_gray(frame)
        if self._first is None:
            self._first = curr
        self._t += 1
        self._ring[self._t % self.ring_depth] = curr

        mats = self._transforms(curr)
        self.last_mats = mats
        ch1 = cv2.absdiff(curr, self.estimator.warp(self._frame_at_lag(self.stride_step), mats[self.stride_step]))
        history = [self.estimator.warp(self._frame_at_lag(lag), mats[lag]) for lag in self.lags]
        median_bg = np.median(np.stack(history, axis=0), axis=0).astype(np.float32)
        ch2 = np.clip(curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
        out = np.stack([curr, ch1, ch2], axis=0)

        for j in [j for j in self._steps if j < self._t - self.max_lag]:
            self._steps.pop(j, None)

        if self.dump_intermediate:
            self._dump(self._t, out)
        return out

    def run(self, source: FrameSource) -> Iterator[np.ndarray]:
        for frame in source:
            yield self.push(frame)

    # ------------------------------------------------------------------ dump
    def _dump_meta(self) -> None:
        meta = {
            "window": self.window,
            "stride_step": self.stride_step,
            "max_lag": self.max_lag,
            "anchor_step_lags": self.anchors,
            "ring_depth": self.ring_depth,
            "channel_order": ["Ch0 I_t", "Ch1 |I_t - W(I_t-2)|", "Ch2 (I_t - B_t)^+"],
            "note": "Ch1 uses lag 2 exactly. 21 history lags -> median index 10 of 0..20.",
        }
        (self.dump_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _dump(self, index: int, out: np.ndarray) -> None:
        if index == 0:
            per_frame = 3 * out.shape[1] * out.shape[2]
            print(f"[DUMP] {self.dump_dir} : {per_frame / 1048576:.2f} MiB/frame, "
                  f"1000 frames = {1000 * per_frame / 1073741824:.3f} GiB (npy, uncompressed)")
        for c in range(3):
            np.save(self.dump_dir / f"{index:06d}_ch{c}.npy", out[c])


def run_verification_test() -> None:
    """Self-contained smoke test: shape contract, cold start, channel semantics, dtype."""
    rng = np.random.default_rng(20261008)
    h, w = 512, 640
    base = (rng.normal(96, 6, size=(h, w))).clip(0, 255).astype(np.uint8)

    pipe = OnlineFeaturePipeline(window=21, stride_step=2, anchor_step=10, downscale=2)
    print(f"ring_depth={pipe.ring_depth} lags={pipe.lags[0]}..{pipe.lags[-1]} anchors={pipe.anchors}")

    t1 = base.copy()
    t1[256, 320] = 220
    t2 = base.copy()
    t2[256, 321] = 220

    f0 = pipe.push(t1)
    f1 = pipe.push(t2)
    print(f"shape={f0.shape} dtype={f0.dtype} (expected (3, {h}, {w}) uint8)")
    assert f0.shape == (3, h, w) and f0.dtype == np.uint8
    assert np.array_equal(f0[0], t1), "Ch0 must be the untouched current frame"
    assert np.array_equal(f1[0], t2)
    print(f"Ch0 == I_t : exact")
    print(f"Ch1 non-zero pixels = {int((f1[1] > 0).sum())} (moving point must light it up)")
    print(f"Ch2 non-zero pixels = {int((f1[2] > 0).sum())}")
    assert (f1[1] > 0).any(), "Ch1 must respond to the moving point"
    assert (f1[2] > 0).any(), "Ch2 must respond to the moving point"
    print(f"state_bytes after 2 frames = {pipe.state_bytes()} "
          f"({pipe.state_bytes() / 1048576:.2f} MiB, full ring = "
          f"{pipe.ring_depth * h * w / 1048576:.2f} MiB)")
    print("SMOKE OK")


if __name__ == "__main__":
    run_verification_test()