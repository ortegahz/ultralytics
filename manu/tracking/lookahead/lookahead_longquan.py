#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Longquanshan adapter for the Anti-UAV system-SOTA post-processing pipeline.

本文件是 ``manu/evaluation/system_sota_longquan.py`` 的拷贝，外挂「快速前向推演」
（look-ahead extrapolator）：在成熟管线的输出末端，把上报给转台的坐标从「t−Δ 帧的旧位置」
推到「当前物理帧 t」。原文件不得修改；本拷贝只做契约允许的改动，其中 ``--lookahead off``
必须与原脚本**逐位一致**（外挂是纯后处理，绝不改写历史估计）。

The system pipeline (``manu/evaluation/eval_bidirectional_track_fusion.py``) is reused untouched. This module only

* adapts the Longquanshan multi-scale heatmap cache (``runs/ms_cache_nop0/<SEQ>.pkl``) into the record format the
  pipeline expects, in native 640x512 pixels (no letterbox round trip),
* adds the two continuity layers the business domain needs (consecutive-frame confirmation on the output side,
  temporal persistence on the input side), and
* scores the result with the business-domain matching rule (greedy score-descending, pure point-in-bbox) taken
  from ``manu/evaluation/eval_multiscale_cache_th.py`` so the numbers are directly comparable with the frozen
  single-frame baseline of F1 87.070%.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    COUNT_KEYS,
    count_matches,
    load_ground_truth,
    load_predictions,
    merge_scales,
    metrics,
)
from manu.tracking.lookahead.extrapolator import (  # noqa: E402
    LookaheadConfig,
    build_report_stream,
    servo_metrics,
)
from manu.tracking.lookahead.track_fusion import (  # noqa: E402
    evaluate_sequence_bidirectional,
    natural_sort_key,
)

IMG_W = 640
IMG_H = 512
SCALE_TAGS = (0, 1, 2)  # native 640 / 160 / 80 rows; the baked fused row (255) is deliberately excluded

DEFAULT_PARAMS = {
    # thresholds and sky partition
    "th_base": 0.22,
    "th_salvage": 0.06,
    "th_ground": 0.35,
    "sky_ratio": 0.60,
    # tracker
    "min_hits": 3,
    "max_age": 3,
    "match_dist": 12.0,
    "max_match_dist": 18.0,
    "instant_conf": 0.25,
    "min_disp": 2.5,
    "min_track_score": 0.08,
    # smoother
    "stitch_gap": 4,
    "infill_gap": 3,
    "min_hits_infill": 5,
    "min_rigid_disp": 2.0,
    "max_rigid_var": 0.50,
    "min_hits_prune": 8,
    # cross-scale de-duplication (native px)
    "dedup_radius": 32.0,
    # Phase-A / P3 extensions, all OFF in the Anti-UAV golden default
    "th_deep": 0.0,
    "min_hits_deep": 3,
    "min_rigid_disp_sky": -1.0,
    "hover_vel": 0.0,
    "hover_infill_gap": 15,
    "coast_frames": 0,
    "min_hits_hover": 8,
    "hover_sky_only": 1,
    "stitch_long_gap": 0,
    "stitch_vel_diff": 4.0,
    # continuity layers
    "confirm_consec": 3,
    "confirm_gap_tol": 0,
    "confirm_trim_prefix": 1,
    "persist_min_hits": 0,
    "persist_window": 0,
    "persist_tol": 4.0,
}

# Frozen single-frame baseline, runs/logs/eval_nop0_inbox_ext2.json (tag=all, merge_radius=768, inbox, th=0.22).
GATE_EXPECTED = {
    "tp": 110788,
    "fp": 10435,
    "fn": 22469,
    "gt": 133257,
    "recall": 0.8313859684669473,
    "precision": 0.9139189757719245,
    "f1": 0.8707010374096196,
    "far_per_frame": 0.06315706651011058,
}

_META_CACHE: dict[tuple[str, str], dict] = {}


# ==============================================================================
# 1. Cache adapters
# ==============================================================================


def load_cache_meta(cache_root, sequence: str) -> dict:
    """Return the cache meta of one sequence (memoised; the cache directory is read-only)."""
    key = (str(cache_root), sequence)
    if key not in _META_CACHE:
        with open(Path(cache_root) / f"{sequence}.pkl", "rb") as handle:
            _META_CACHE[key] = pickle.load(handle)["meta"]
    return _META_CACHE[key]


def build_candidates(cache_root, sequence: str, dedup_radius: float, floor: float = 0.0) -> dict[int, tuple]:
    """Union of the three per-scale cache rows, greedily de-duplicated across scales by score.

    The baked fused row (tag 255) is excluded on purpose: its floor is 0.22, which would empty the weak-pulse
    salvage pool the system pipeline relies on. ``dedup_radius`` must be > 0 because ``merge_scales`` overwrites
    scales instead of merging them when the radius is not positive.
    """
    if dedup_radius <= 0:
        raise ValueError("dedup_radius must be > 0; merge_scales overwrites scales for radius <= 0")
    meta, by_tag = load_predictions(cache_root, sequence, list(SCALE_TAGS), floor)
    _META_CACHE[(str(cache_root), sequence)] = meta
    return merge_scales(by_tag, dedup_radius)


def apply_persistence_filter(
    candidates: dict[int, tuple],
    persist_min_hits: int = 0,
    persist_window: int = 0,
    persist_tol: float = 4.0,
) -> dict[int, tuple]:
    """Keep only candidates that recur over time, i.e. that are not a one-frame flash.

    A candidate of frame ``f`` survives when at least ``persist_min_hits`` detections fall within
    ``persist_tol`` pixels of it inside the window of the ``persist_window`` frames ending at ``f``
    (frames ``f - persist_window + 1 .. f``). The candidate's own detection always counts, so
    ``persist_min_hits <= 1`` keeps everything. ``persist_min_hits == 0`` or ``persist_window == 0``
    disables the filter. Score-descending order of the surviving candidates is preserved.
    """
    if not persist_min_hits or not persist_window:
        return candidates

    frames = np.array(sorted(candidates), dtype=np.int64)
    counts = np.array([len(candidates[int(frame)][0]) for frame in frames], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    points = np.concatenate([candidates[int(frame)][1] for frame in frames]).astype(np.float32)

    filtered: dict[int, tuple] = {}
    left = 0
    for position, frame in enumerate(frames):
        while frames[left] < frame - persist_window + 1:
            left += 1
        start, end = offsets[position], offsets[position + 1]
        current = points[start:end]
        pool = points[offsets[left] : end]
        distances = np.linalg.norm(current[:, None, :] - pool[None, :, :], axis=2)
        keep = (distances <= persist_tol).sum(axis=1) >= persist_min_hits
        if keep.any():
            scores = candidates[int(frame)][0]
            filtered[int(frame)] = (scores[keep], current[keep])
    return filtered


def build_records(
    cache_root,
    labels_root,
    sequence: str,
    dedup_radius: float,
    floor: float = 0.0,
    persist_min_hits: int = 0,
    persist_window: int = 0,
    persist_tol: float = 4.0,
) -> list[dict]:
    """Build the per-frame records the system pipeline consumes for one sequence.

    Every record carries the native-pixel candidate pool, the ground-truth centres and the ground-truth boxes
    (cx, cy, w, h). The frame set is the union of candidate frames and ground-truth frames so that a frame
    without any detection still contributes to the false-negative denominator.
    """
    candidates = build_candidates(cache_root, sequence, dedup_radius, floor)
    candidates = apply_persistence_filter(candidates, persist_min_hits, persist_window, persist_tol)

    meta = load_cache_meta(cache_root, sequence)
    width, height = meta["native_size"]
    truth = load_ground_truth(labels_root, sequence, width, height)
    return make_records(candidates, truth, sequence)


def make_records(candidates: dict[int, tuple], truth: dict[int, np.ndarray], sequence: str) -> list[dict]:
    """Assemble the per-frame records from a candidate pool and the ground truth of one sequence.

    The frame set is the union of candidate frames and ground-truth frames so that a frame without any
    detection still contributes to the false-negative denominator.
    """
    records = []
    for frame_index in sorted(set(candidates) | set(truth)):
        found = candidates.get(frame_index)
        if found is None:
            points = np.zeros((0, 2), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            scores, points = found
            scores = np.asarray(scores, dtype=np.float32)
            points = np.asarray(points, dtype=np.float32)
        boxes = truth.get(frame_index)
        if boxes is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        records.append(
            {
                "im_name": f"{sequence}__{frame_index:06d}.jpg",
                "frame_index": int(frame_index),
                "pred_points": points,
                "pred_scores": scores,
                "gt_pts": boxes[:, :2].copy(),
                "gt_bboxes": boxes.copy(),
            }
        )
    return records


# ==============================================================================
# 2. System pipeline and continuity layers
# ==============================================================================


def build_tracker_config(params: dict, img_h: int) -> dict:
    """Tracker configuration; the key names mirror eval_bidirectional_track_fusion.main()."""
    return {
        "max_age": int(params["max_age"]),
        "min_hits": int(params["min_hits"]),
        "match_dist": float(params["match_dist"]),
        "max_match_dist": float(params["max_match_dist"]),
        "min_track_score": float(params["min_track_score"]),
        "instant_conf": float(params["instant_conf"]),
        "min_displacement": float(params["min_disp"]),
        "sky_ratio": float(params["sky_ratio"]),
        "img_h": int(img_h),
        "th_deep_salvage": float(params["th_deep"]),
        "min_hits_deep_salvage": int(params["min_hits_deep"]),
    }


def build_smoother_config(params: dict, img_h: int) -> dict:
    """Smoother configuration; the key names mirror eval_bidirectional_track_fusion.main()."""
    return {
        "stitch_max_gap": int(params["stitch_gap"]),
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": int(params["min_hits_infill"]),
        "max_infill_gap": int(params["infill_gap"]),
        "min_track_hits": int(params["min_hits"]),
        "min_track_score": float(params["min_track_score"]),
        "instant_conf": float(params["instant_conf"]),
        "min_rigid_displacement": float(params["min_rigid_disp"]),
        "max_rigid_variance": float(params["max_rigid_var"]),
        "min_hits_for_prune": int(params["min_hits_prune"]),
        "stitch_long_gap": int(params["stitch_long_gap"]),
        "stitch_max_vel_diff": float(params["stitch_vel_diff"]),
        "hover_vel_thresh": float(params["hover_vel"]),
        "hover_infill_gap": int(params["hover_infill_gap"]),
        "coast_max_frames": int(params["coast_frames"]),
        "min_hits_hover": int(params["min_hits_hover"]),
        "hover_sky_only": bool(params["hover_sky_only"]),
        # -1.0 means "off / legacy"; the pipeline only accepts None or a non-negative threshold here
        "min_rigid_disp_sky": (
            None if float(params["min_rigid_disp_sky"]) < 0 else float(params["min_rigid_disp_sky"])
        ),
    }


def run_system_sota(
    records: list[dict],
    params: dict,
    img_h: int = IMG_H,
    capture_kf_states: bool = False,
) -> tuple[list[list[dict]], dict]:
    """Run the Anti-UAV system pipeline on the Longquanshan records and apply the consecutive-frame gate.

    Returns the (filtered) per-frame detections and the raw system report. The system report holds the
    Hungarian reference metrics of the *unfiltered* pipeline output, the business-domain metrics are computed
    from the returned detections.
    """
    res = evaluate_sequence_bidirectional(
        records=records,
        dist_thresh=8.0,
        th_base=float(params["th_base"]),
        th_salvage=float(params["th_salvage"]),
        th_ground=float(params["th_ground"]),
        sky_ratio=float(params["sky_ratio"]),
        img_h=int(img_h),
        tracker_config=build_tracker_config(params, img_h),
        smoother_config=build_smoother_config(params, img_h),
        match_mode="bbox",
        capture_kf_states=capture_kf_states,
    )
    dets = apply_consecutive_confirmation(
        res["bidi_frame_dets"],
        confirm_consec=int(params.get("confirm_consec", 3)),
        confirm_gap_tol=int(params.get("confirm_gap_tol", 0)),
        confirm_trim_prefix=int(params.get("confirm_trim_prefix", 1)),
    )
    return dets, res


def apply_consecutive_confirmation(
    bidi_frame_dets: list[list[dict]],
    confirm_consec: int = 3,
    confirm_gap_tol: int = 0,
    confirm_trim_prefix: int = 1,
) -> list[list[dict]]:
    """Drop tracks that never reach ``confirm_consec`` consecutive observations, and trim their prefix.

    ``infilled=False`` marks a real observation, ``infilled=True`` a synthesised frame. Per track the real
    observation frames are collected and split into runs; a frame belongs to the same run as its predecessor
    when the frame distance is at most ``1 + confirm_gap_tol``. A track whose longest run is shorter than
    ``confirm_consec`` is a one-frame flash and is removed from the output entirely, synthetic frames included.
    For a surviving track the start frame of the first run that reaches ``confirm_consec`` is the confirmation
    frame; with ``confirm_trim_prefix`` enabled every detection before it is dropped, so a target only shows up
    once it has been seen continuously. The frame list keeps its length and every frame is re-sorted by
    descending score, because the business-domain matcher assigns detections greedily in array order.
    """
    confirm_consec = max(1, int(confirm_consec))
    confirm_gap_tol = max(0, int(confirm_gap_tol))
    max_gap = 1 + confirm_gap_tol

    observed: dict[int, list[int]] = {}
    for frame, dets in enumerate(bidi_frame_dets):
        for det in dets:
            if not det.get("infilled", False):
                observed.setdefault(int(det["track_id"]), []).append(frame)

    confirmed_at: dict[int, int] = {}
    for track_id, frames in observed.items():
        frames = sorted(set(frames))
        runs: list[tuple[int, int]] = []
        run_start, run_length, previous = frames[0], 1, frames[0]
        for frame in frames[1:]:
            if frame - previous <= max_gap:
                run_length += 1
            else:
                runs.append((run_length, run_start))
                run_start, run_length = frame, 1
            previous = frame
        runs.append((run_length, run_start))
        if max(length for length, _ in runs) < confirm_consec:
            continue
        confirmed_at[track_id] = next(start for length, start in runs if length >= confirm_consec)

    filtered: list[list[dict]] = []
    for frame, dets in enumerate(bidi_frame_dets):
        kept = []
        for det in dets:
            track_id = int(det["track_id"])
            if track_id not in confirmed_at:
                continue
            if confirm_trim_prefix and frame < confirmed_at[track_id]:
                continue
            kept.append(det)
        kept.sort(key=lambda det: -float(det["score"]))
        filtered.append(kept)
    return filtered


# ==============================================================================
# 3. Business-domain evaluation
# ==============================================================================


def evaluate_biz(records: list[dict], bidi_frame_dets: list[list[dict]], num_frames: int | None = None) -> dict:
    """Score the final per-frame detections with the business-domain rule.

    Greedy score-descending assignment, pure point-in-bbox matching (the tolerance is ignored), and a ground
    truth absorbs at most one detection. The detections are re-sorted by descending score because the matcher
    walks the arrays in order.
    """
    ordered = sorted(records, key=lambda record: natural_sort_key(record["im_name"]))
    if len(bidi_frame_dets) != len(ordered):
        raise ValueError(f"detection/record mismatch: {len(bidi_frame_dets)} vs {len(ordered)}")

    frames = []
    for record, dets in zip(ordered, bidi_frame_dets):
        boxes = np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4)
        if len(dets):
            ranked = sorted(dets, key=lambda det: -float(det["score"]))
            points = np.asarray([det["pos"] for det in ranked], dtype=np.float32).reshape(-1, 2)
            scores = np.asarray([float(det["score"]) for det in ranked], dtype=np.float32)
        else:
            points, scores = None, None
        frames.append((boxes if len(boxes) else None, points, scores))

    counts = count_matches(frames, 0.0, 8.0, "inbox")
    return metrics(counts, num_frames if num_frames is not None else len(ordered))


def single_frame_dets(records: list[dict], threshold: float) -> list[list[dict]]:
    """The raw candidate pool above ``threshold``, shaped like the tracker's per-frame output.

    This is the single-frame arm: no tracker, no smoother, no continuity gate.
    """
    ordered = sorted(records, key=lambda record: natural_sort_key(record["im_name"]))
    out = []
    for record in ordered:
        scores = np.asarray(record["pred_scores"], dtype=np.float32)
        points = np.asarray(record["pred_points"], dtype=np.float32)
        keep = np.nonzero(scores >= threshold)[0]
        out.append(
            [
                {
                    "pos": (float(points[index][0]), float(points[index][1])),
                    "score": float(scores[index]),
                    "track_id": -1,
                    "infilled": False,
                }
                for index in keep
            ]
        )
    return out


def aggregate_metrics(per_sequence: list[dict], num_frames: int) -> dict:
    """Aggregate per-sequence counts and recompute the metrics once (no per-sequence F1 averaging)."""
    counts = {key: int(sum(row[key] for row in per_sequence)) for key in COUNT_KEYS}
    return metrics(counts, num_frames)


# ==============================================================================
# 3.1 Look-ahead 外挂：四臂对照（尺子 1 / 尺子 2）
# ==============================================================================

# 对照臂 ↔ 外推档位：original 是原脚本自己的输出（无滞后，精度天花板），
# 其余三臂分别对应契约 §1 的 stale / kinematic / replay。
ARM_MODES = {"original": "off", "stale": "stale", "kinematic": "kinematic", "replay": "replay"}


class CachedFrameSource:
    """从**本地 memmap 帧缓存**取灰度整帧（索引 = 绝对帧号）。

    龙泉山帧目录挂在 sshfs 上，实测读+解码 11.3 ms/帧，而同样的帧放本地盘只要
    2.8 ms/帧、memmap 后 ~0.1 ms/帧。评测里每条臂、每组参数都要把两万九千帧
    过一遍，走网络光 I/O 就是每趟 4 分钟；本地缓存把这个开销一次性摊掉。
    缓存由 ``manu/tracking/lookahead/build_frame_cache.py`` 生成。
    """

    def __init__(self, npy_path, frame_indices: list[int], delta: int, extra_window: int = 0):
        self.path = Path(npy_path)
        self.frame_indices = list(frame_indices)
        self.capacity = max(2, int(delta) + 2 + int(extra_window))
        self.reads = 0
        self._array = np.load(self.path, mmap_mode="r")
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame_index: int) -> np.ndarray | None:
        """取 record 下标对应的灰度帧；越界返回 ``None``。"""
        key = int(frame_index)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if key < 0 or key >= len(self.frame_indices):
            return None
        absolute = int(self.frame_indices[key])
        if absolute < 0 or absolute >= self._array.shape[0]:
            return None
        image = np.array(self._array[absolute], dtype=np.uint8)
        self.reads += 1
        self._cache[key] = image
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return image


def build_frame_source(frames_root, frames_cache, sequence, frame_indices, delta, extra_window=0):
    """优先用本地 memmap 缓存，退回 sshfs 上的 JPEG 目录，都没有则返回 ``None``。

    ``extra_window`` = 回退链的最长回退帧数（``--replay-rollback-max-frames``）：
    回退后源帧变成 ``f' = t − Δ − step_back``，重推链要读的区间变成 ``[f', t]``
    （最长 Δ+1+extra_window 帧）⇒ 缓存窗口必须**含回退**，否则每帧会被重复读盘
    （实测交付档 CFG 下 `frame_reads` 比帧数多 7.8%~16.5%）。
    """
    if frames_cache:
        npy_path = Path(frames_cache) / f"{sequence}.npy"
        if npy_path.exists():
            return CachedFrameSource(npy_path, frame_indices, delta, extra_window)
    if frames_root:
        directory = Path(frames_root) / sequence
        if directory.is_dir():
            return SequenceFrameSource(directory, frame_indices, delta, extra_window)
    return None


class SequenceFrameSource:
    """按**管线时间轴**（record 下标）取灰度整帧。

    record 下标与视频里的绝对帧号不是一回事（绝对帧号存在 ``records[i]["frame_index"]``），
    所以这里先把 record 下标映射回绝对帧号再取图。缓存留最近 ``delta + 2 + 回退窗口`` 张：
    replay 的重推区间是 ``f+1..f+Δ``，**回退后**是 ``f−rollback+1..f+Δ``
    ⇒ 窗口含回退时每个源帧**至多读一次**（``frame_reads`` 与帧数相等）。
    """

    def __init__(self, frames_root, frame_indices: list[int], delta: int, extra_window: int = 0):
        self.root = Path(frames_root)
        self.frame_indices = list(frame_indices)
        self.capacity = max(2, int(delta) + 2 + int(extra_window))
        self.reads = 0
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, frame_index: int) -> np.ndarray | None:
        """取 record 下标对应的灰度帧；缺文件返回 ``None``。"""
        key = int(frame_index)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if key < 0 or key >= len(self.frame_indices):
            return None
        path = self.root / f"frame_{self.frame_indices[key]:06d}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            # 部分 OpenCV 构建的 imread 处理不了含中文的路径，用 imdecode 兜底（帧根目录里有「龙泉山」）。
            try:
                buffer = np.fromfile(path, dtype=np.uint8)
            except OSError:
                return None
            if not buffer.size:
                return None
            image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None
        self.reads += 1
        self._cache[key] = image
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return image


def build_lookahead_config(args, mode: str) -> LookaheadConfig:
    """把 CLI 参数折成外挂配置（各臂之间只差一个 ``mode``）。"""
    return LookaheadConfig(
        delta=int(args.delta),
        mode=mode,
        velocity_source=str(args.velocity_source),
        velocity_window=int(args.velocity_window),
        position_source=str(args.position_source),
        win_size=int(args.fast_win_size),
        th_k=float(args.fast_th_k),
        min_energy=float(args.fast_min_energy),
        min_contrast=float(args.fast_min_contrast),
        fast_mode=str(args.fast_mode),
        bg_window=int(args.fast_bg_window),
        peak_window=int(args.fast_peak_window),
        replay_anchor=str(args.replay_anchor),
        replay_kf_r=float(args.fast_kf_r),
        require_valid=bool(args.replay_require_valid),
        replay_fallback=str(args.replay_fallback),
        replay_step_gate_px=float(args.replay_step_gate_px),
        replay_max_hold=int(args.replay_max_hold),
        replay_anchor_check_px=float(args.replay_anchor_check_px),
        replay_pred_radius_px=float(args.replay_pred_radius_px),
        replay_roi_mode=str(args.replay_roi_mode),
        replay_roi_selfpred_window=int(args.replay_roi_selfpred_window),
        replay_roi_selfpred_max_px=float(args.replay_roi_selfpred_max_px),
        replay_xcorr_radius=float(args.replay_xcorr_radius),
        replay_xcorr_template=int(args.replay_xcorr_template),
        replay_xcorr_th_k=float(args.replay_xcorr_th_k),
        replay_xcorr_min_peak=float(args.replay_xcorr_min_peak),
        replay_xcorr_bg=int(args.replay_xcorr_bg),
        replay_xcorr_core_frac=float(args.replay_xcorr_core_frac),
        replay_xcorr_uniq_gate=float(args.replay_xcorr_uniq_gate),
        replay_abstain_peak_below=float(args.replay_abstain_peak_below),
        replay_rollback_max_frames=int(args.replay_rollback_max_frames),
        replay_rollback_min_peak=float(args.replay_rollback_min_peak),
        replay_abstain_disp_px=float(args.replay_abstain_disp_px),
        replay_fov_margin_px=float(args.replay_fov_margin_px),
        replay_fov_frame_size=(IMG_W, IMG_H),
        replay_xcorr_mode=str(args.replay_xcorr_mode),
        replay_xcorr_chain_radius=float(args.replay_xcorr_chain_radius),
        replay_xcorr_stride=int(args.replay_xcorr_stride),
        replay_polarity=str(args.replay_polarity),
        replay_polarity_ratio=float(args.replay_polarity_ratio),
        replay_consistency_gate_px=float(args.replay_consistency_gate_px),
        replay_step_rel_tol_px=float(args.replay_step_rel_tol_px),
        replay_hold_use_ref=bool(args.replay_hold_use_ref),
        replay_outlier_tol_px=float(args.replay_outlier_tol_px),
        replay_hold_window=int(args.replay_hold_window),
        replay_hold_outlier_tol_px=float(args.replay_hold_outlier_tol_px),
        replay_hold_robust=bool(args.replay_hold_robust),
        replay_obs_shrink_alpha=float(args.replay_obs_shrink_alpha),
        replay_freeze_speed_px=float(args.replay_freeze_speed_px),
        replay_disp_shrink_alpha=float(args.replay_disp_shrink_alpha),
        replay_disp_merge_sigma_px=float(args.replay_disp_merge_sigma_px),
        replay_disp_gain=float(args.replay_disp_gain),
        replay_mode=str(args.replay_mode),
        rolling_recenter=bool(int(args.rolling_recenter)),
        rolling_hold_mode=str(args.rolling_hold_mode),
        rolling_max_step_px=float(args.rolling_max_step_px),
        rolling_max_offset_px=float(args.rolling_max_offset_px),
        rolling_reacquire_px=float(args.rolling_reacquire_px),
        rolling_vel_alpha=float(args.rolling_vel_alpha),
        refine_gate_px=float(args.refine_gate_px),
        output_max_step_px=float(args.output_max_step_px),
    )


def same_detections(left: list[dict], right: list[dict]) -> bool:
    """两个检测列表是否逐位相同（坐标 / 分数 / track_id 全等）。"""
    if len(left) != len(right):
        return False
    for one, other in zip(left, right):
        if int(one["track_id"]) != int(other["track_id"]):
            return False
        if float(one["score"]) != float(other["score"]):
            return False
        if (float(one["pos"][0]), float(one["pos"][1])) != (float(other["pos"][0]), float(other["pos"][1])):
            return False
    return True


def evaluate_arms(
    records: list[dict],
    dets: list[list[dict]],
    frames: int,
    arms: list[str],
    configs: dict[str, LookaheadConfig],
    delta: int,
    kf_states: dict | None,
    frame_source: SequenceFrameSource | None,
) -> tuple[dict[str, dict], dict[str, list[list[dict]]]]:
    """跑完对照臂，返回 ``(每臂指标, 每臂上报流)``。

    尺子 1（历史帧等价性）：源帧 f 的历史估计 D(f) 对 GT_f 的评测。外挂是纯后处理，
    绝不回写历史估计，所以这一口径在**所有臂**上必须逐位相等（ΔF1 = 0.000000）——
    这正是尺子 1 要钉死的不变量。
    尺子 2（最新帧伺服有效性）：上报坐标对**当前帧** GT_t 的评测，外加伺服物理量
    （CLE / 命中率 / 抖动 / 脱靶）。
    """
    ordered = sorted(records, key=lambda record: natural_sort_key(record["im_name"]))
    truth = {
        index: np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4)
        for index, record in enumerate(ordered)
    }

    # 尺子 1 的共同源帧窗口 [0, N-Δ)：外挂滞后 Δ，源帧 f 只有 f < N-Δ 才有对应的物理帧。
    source_limit = max(0, len(ordered) - int(delta))

    results: dict[str, dict] = {}
    reports: dict[str, list[list[dict]]] = {}
    for arm in arms:
        if arm == "original":
            # original 臂 = 原脚本自己的输出，一个字节都不搬运：这是「关掉开关 ⇒ 逐位一致」的锚。
            report = dets
            stats = {
                "mode": "off",
                "delta": 0,
                "applied": sum(len(frame) for frame in dets),
                "abstained": 0,
                "fallback_used": 0,
                "missing_velocity": 0,
                "replay_steps": 0,
                "replay_failed": 0,
                "replay_step_jump": 0,
                "replay_max_step_disp": 0.0,
            }
        else:
            report, stats = build_report_stream(
                dets,
                len(ordered),
                configs[arm],
                kf_states=kf_states,
                frame_source=frame_source,
            )
        reports[arm] = report

        # 该臂在物理帧 t 上报的坐标是否**原样搬运**了源帧 f = t-Δ 的历史估计：
        # stale 臂必须是 True（老逻辑），kinematic / replay 必须是 False（它们改了坐标）。
        verbatim = None
        if arm != "original" and source_limit:
            verbatim = all(same_detections(report[f + delta], dets[f]) for f in range(source_limit))

        results[arm] = {
            "current_frame": evaluate_biz(ordered, report, num_frames=frames),
            "source_frame": evaluate_biz(ordered[:source_limit], dets[:source_limit], num_frames=frames),
            "servo": servo_metrics(report, truth, len(ordered), return_samples=True),
            "stream_stats": stats,
            "source_verbatim": verbatim,
            "source_window": source_limit,
            # 上报流的长度是 record 数（servo_metrics 的分母），聚合时要用它而不是视频帧数。
            "report_frames": len(ordered),
            # 从磁盘读了多少张灰度帧：replay 臂的「每帧只读一次」就是靠这个数看出来的。
            "frame_reads": frame_source.reads if frame_source is not None else 0,
        }
    return results, reports


def aggregate_servo(
    per_sequence: dict[str, dict],
    report_frames: int,
    pooled_cle: list[float] | None = None,
) -> dict:
    """把逐序列的伺服指标汇总成总体。

    计数类（帧数 / 命中数 / 脱靶帧数 / 碎片数）可以精确合并；命中率和覆盖率也是**全帧池化**。
    分位数类分两种口径，报告里必须分清：
    * ``cle_p50/p90/p95/p99/max/mean`` = **逐序列最差值**（``worst_over_sequences``）——
      旧口径，跨序列不可合并（一条烂序列会把全局中位数拉成几百像素，看着像全线崩盘）；
    * ``cle_pooled_*`` = **全帧池化分位数**（把所有序列的逐帧 CLE 样本拼起来算）——
      CLE 是「本帧上报点到最近 GT 中心」的距离，没有跨帧差分，跨序列拼接是安全的；
    * 抖动（``max_jitter``/``p99_jitter``）**只能取逐序列最差值**：它是二阶差分，
      把不同序列拼在一起会在序列边界造出假抖动。
    """
    rows = list(per_sequence.values())
    if not rows:
        return {}
    total_frames = sum(int(row.get("frames_with_report", 0)) for row in rows)
    total_gt = sum(int(row.get("gt_frames", 0)) for row in rows)
    total_inbox = sum(int(row.get("inbox_hits", 0)) for row in rows)
    total_lost = sum(int(row.get("lost_frames", 0)) for row in rows)
    total_frag = sum(int(row.get("fragmentation", 0)) for row in rows)
    weighted_5px = sum(float(row.get("hit_rate_5px", 0.0)) * int(row.get("gt_frames", 0)) for row in rows)
    summary = {
        "frames_with_report": total_frames,
        "gt_frames": total_gt,
        "inbox_hits": total_inbox,
        "lost_frames": total_lost,
        "fragmentation": total_frag,
        "coverage": total_frames / report_frames if report_frames else 0.0,
        "hit_rate_inbox": total_inbox / total_gt if total_gt else 0.0,
        "hit_rate_5px": weighted_5px / total_gt if total_gt else 0.0,
    }
    for key in (
        "cle_p50",
        "cle_p90",
        "cle_p95",
        "cle_p99",
        "cle_max",
        "cle_mean",
        "max_jitter",
        "p99_jitter",
    ):
        values = [float(row.get(key, 0.0)) for row in rows]
        summary[key] = max(values) if values else 0.0
    summary["percentile_aggregation"] = "worst_over_sequences"
    summary["cle_pooled_aggregation"] = "pooled_over_all_frames"
    summary["cle_pooled_samples"] = 0
    if pooled_cle:
        pooled = np.asarray(pooled_cle, dtype=np.float64)
        summary["cle_pooled_p50"] = float(np.percentile(pooled, 50))
        summary["cle_pooled_p90"] = float(np.percentile(pooled, 90))
        summary["cle_pooled_p95"] = float(np.percentile(pooled, 95))
        summary["cle_pooled_p99"] = float(np.percentile(pooled, 99))
        summary["cle_pooled_max"] = float(pooled.max())
        summary["cle_pooled_mean"] = float(pooled.mean())
        summary["cle_pooled_samples"] = int(pooled.size)
    return summary


def render_arm_table(arm_summary: dict[str, dict], delta: int, main_arm: str) -> str:
    """把四臂汇总渲染成 markdown 对照表。"""
    lines = [
        f"# 快速前向推演四臂对照（Δ = {delta} record，主臂 = {main_arm}）",
        "",
        "| 臂 | 档位 | 尺子1 F1 | 尺子2 F1 | Recall | Prec | FAR/帧 | CLE50(池化) | CLE95(池化) "
        "| CLE95(最差序列) | hit@5px | inbox | 抖动p99 | 碎片 | 覆盖率 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, summary in arm_summary.items():
        source = summary["source_frame"]["overall"]
        current = summary["current_frame"]["overall"]
        servo = summary["servo"]["overall"]
        lines.append(
            f"| {arm} | {summary['mode']} | {source['f1']:.6f} | {current['f1']:.6f} | "
            f"{current['recall'] * 100:.2f}% | {current['precision'] * 100:.2f}% | {current['far_per_frame']:.4f} | "
            f"{float(servo.get('cle_pooled_p50', 0.0)):.2f} | "
            f"{float(servo.get('cle_pooled_p95', 0.0)):.2f} | "
            f"{float(servo.get('cle_p95', 0.0)):.2f} | "
            f"{float(servo.get('hit_rate_5px', 0.0)) * 100:.2f}% | "
            f"{float(servo.get('hit_rate_inbox', 0.0)) * 100:.2f}% | "
            f"{float(servo.get('p99_jitter', 0.0)):.2f} | {int(servo.get('fragmentation', 0))} | "
            f"{float(servo.get('coverage', 0.0)):.4f} |"
        )
    lines += [
        "",
        "> 尺子1 = 源帧口径（源帧 f 的历史估计 D(f) 对 GT_f），各臂共享同一源帧窗口，ΔF1 必须为 0；",
        "> 尺子2 = 上报坐标对当前帧 GT_t 的口径。`CLE50/95(池化)` = 全帧池化分位数（**伺服可接受性看这个**）；",
        "> `CLE95(最差序列)` = 逐序列分位数取最差（**尾部风险看这个**）；抖动是二阶差分，跨序列拼接会造假抖动，"
        "故只给最差序列口径；",
        "> 计数类（inbox 命中 / 碎片 / 覆盖率 / hit@5px）按序列精确合并。",
        "",
    ]
    return "\n".join(lines)


# ==============================================================================
# 4. Correctness gate
# ==============================================================================


def run_correctness_gate(
    cache_root,
    labels_root,
    sequences: list[str],
    radius: float = 768.0,
    threshold: float = 0.22,
) -> dict:
    """Reproduce the frozen single-frame baseline through this adapter.

    Radius 768 px plus threshold 0.22 collapses the candidate pool to roughly one detection per frame, which is
    exactly the configuration behind runs/logs/eval_nop0_inbox_ext2.json. The adapter must reproduce those
    numbers bit for bit before any system post-processing is trusted.
    """
    per_sequence = []
    num_frames = 0
    for sequence in sequences:
        records = build_records(cache_root, labels_root, sequence, radius, 0.0)
        frames = load_cache_meta(cache_root, sequence)["frames"]
        dets = single_frame_dets(records, threshold)
        per_sequence.append(evaluate_biz(records, dets, num_frames=frames))
        num_frames += frames
    overall = aggregate_metrics(per_sequence, num_frames)
    passed = (
        overall["tp"] == GATE_EXPECTED["tp"]
        and overall["fp"] == GATE_EXPECTED["fp"]
        and overall["fn"] == GATE_EXPECTED["fn"]
        and overall["gt"] == GATE_EXPECTED["gt"]
        and abs(overall["f1"] - GATE_EXPECTED["f1"]) < 1e-12
        and abs(overall["far_per_frame"] - GATE_EXPECTED["far_per_frame"]) < 1e-12
    )
    return {"passed": passed, "expected": GATE_EXPECTED, "actual": overall, "num_frames": num_frames}


# ==============================================================================
# 5. Output contract
# ==============================================================================


def frame_payload(
    records: list[dict],
    bidi_frame_dets: list[list[dict]],
    num_frames: int | None = None,
) -> dict:
    """Per-frame detections keyed by the string form of the frame number (an empty frame stays an empty list).

    ``num_frames`` is the length of the sequence; every frame of the sequence gets a key, the frames without a
    detection get an empty list, so a renderer can look up any frame index of the source video.
    """
    ordered = sorted(records, key=lambda record: natural_sort_key(record["im_name"]))
    detected: dict[int, list] = {}
    for record, dets in zip(ordered, bidi_frame_dets):
        detected[int(record["frame_index"])] = [
            {
                "pos": [float(det["pos"][0]), float(det["pos"][1])],
                "score": float(det["score"]),
                "track_id": int(det["track_id"]),
                "infilled": bool(det.get("infilled", False)),
            }
            for det in dets
        ]
    if num_frames is None:
        return {str(frame): detected.get(frame, []) for frame in sorted(detected)}
    return {str(frame): detected.get(frame, []) for frame in range(int(num_frames))}


def write_sequence_predictions(
    out_dir,
    sequence: str,
    config: dict,
    row: dict,
    payload: dict,
) -> Path:
    """Write one <sequence>.json prediction file following the shared contract."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sequence}.json"
    path.write_text(
        json.dumps({"config": config, "metrics": row, "per_frame": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_summary(out_dir, params: dict, overall: dict, per_sequence: dict) -> Path:
    """Write the <out-dir>/summary.json roll-up."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(
        json.dumps({"params": params, "overall": overall, "per_sequence": per_sequence}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def resolve_sequences(cache_root, sequence: str, all_seqs: bool, sequences_csv: str = "") -> list[str]:
    """Either the requested sequence(s) or every cached sequence, sorted.

    ``--seqs`` takes a comma separated subset and keeps the caller's order, so a named pool
    (golden pool, blind hold-out) is evaluated as one aggregate instead of 7 separate runs.
    A missing cache is an error: a typo must not silently shrink the pool.
    """
    if all_seqs:
        sequences = sorted(path.stem for path in Path(cache_root).glob("*.pkl"))
        if not sequences:
            raise FileNotFoundError(f"No .pkl caches under {cache_root}")
        return sequences
    requested = [name.strip() for name in sequences_csv.split(",") if name.strip()]
    if requested:
        missing = [name for name in requested if not (Path(cache_root) / f"{name}.pkl").exists()]
        if missing:
            raise FileNotFoundError(f"missing cache .pkl under {cache_root}: {missing}")
        return requested
    if not sequence:
        raise ValueError("either --seq, --seqs or --all-seqs is required")
    return [sequence]


# ==============================================================================
# 6. CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anti-UAV system-SOTA pipeline on the Longquanshan cache")
    parser.add_argument("--cache-root", required=True, help="Directory of <sequence>.pkl heatmap caches")
    parser.add_argument("--labels-root", required=True, help="Directory of {sequence}__{index:06d}.txt labels")
    parser.add_argument("--seq", default="", help="Single sequence name")
    parser.add_argument("--seqs", default="", help="Comma separated sequence subset, evaluated in the given order")
    parser.add_argument("--all-seqs", action="store_true", help="Run every cached sequence")
    parser.add_argument("--dedup-radius", type=float, default=None, help="Cross-scale de-duplication radius (px)")
    parser.add_argument("--params-json", default="", help="JSON file with the full parameter set")
    parser.add_argument("--out-json", default="", help="Optional run report path")
    parser.add_argument("--out-preds-dir", default="", help="Optional directory for the per-sequence predictions")
    parser.add_argument("--no-pipeline", action="store_true", help="Single-frame arm only, no system post-processing")
    parser.add_argument("--self-check", action="store_true", help="Run the adapter correctness gate and exit")
    parser.add_argument("--expect-f1", type=float, default=None, help="Self-calibration gate on the overall F1")
    parser.add_argument("--expect-tp", type=int, default=None, help="Self-calibration gate on the overall TP")
    parser.add_argument("--expect-fp", type=int, default=None, help="Self-calibration gate on the overall FP")
    # ------------------------- 快速前向推演外挂 -------------------------
    parser.add_argument(
        "--lookahead",
        default="off",
        choices=["off", "stale", "kinematic", "replay"],
        help="快推演开关：off=原脚本输出（逐位一致），stale=硬滞后，kinematic=运动学外推，replay=视觉重推",
    )
    parser.add_argument("--delta", type=int, default=8, help="检测滞后 Δ（record 数，默认 8 ≈ 320ms@25Hz）")
    parser.add_argument(
        "--velocity-source",
        default="kf",
        choices=["kf", "diff1", "diffN"],
        help="速度来源：kf=KF 速度分量，diff1=相邻两帧差分，diffN=最近 N 帧最小二乘",
    )
    parser.add_argument("--velocity-window", type=int, default=3, help="diffN 的拟合窗长度（record 数）")
    parser.add_argument(
        "--position-source",
        default="output",
        choices=["output", "kf"],
        help="外推起点位置来源：output=管线输出位置，kf=KF 位置",
    )
    parser.add_argument(
        "--frames-cache",
        default="",
        help="本地 memmap 帧缓存目录（build_frame_cache.py 的产物）；优先于 --frames-root",
    )
    parser.add_argument(
        "--frames-root",
        default="/home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg",
        help="replay 臂的灰度帧根目录（<frames-root>/<sequence>/frame_%%06d.jpg）",
    )
    parser.add_argument(
        "--replay-require-valid",
        dest="replay_require_valid",
        action="store_true",
        default=True,
        help="replay 中途丢目标即整体弃权（默认开）",
    )
    parser.add_argument(
        "--no-replay-require-valid",
        dest="replay_require_valid",
        action="store_false",
        help="replay 中途丢目标仍继续（默认关）",
    )
    parser.add_argument(
        "--replay-fallback",
        default="kinematic",
        choices=["kinematic", "stale", "abstain", "hold", "holdv", "holdc"],
        help="replay 弃权时的兜底档位：hold=用重推途中最后一个有效位置；"
        "holdv=再按最后实测的单步位移补齐剩余步数；"
        "holdc=连续性兜底：从上一帧实际上报位置沿参考速度推进一步（单帧跳变最小）",
    )
    parser.add_argument(
        "--replay-max-hold",
        type=int,
        default=0,
        help="连续观测缺失时允许 hold 几步（0=严格，任一步失效即整段作废；>0 按最后实测位移继续外推）",
    )
    parser.add_argument(
        "--replay-anchor-check-px",
        type=float,
        default=0.0,
        help="delta 档位：源帧快核位置与管线锚点的最大容差（px，0=不校验）。"
        "超容差说明快核在源帧就锁错了目标 ⇒ 整段作废走兜底",
    )
    parser.add_argument(
        "--arms",
        default="original,stale,kinematic,replay",
        help="逗号分隔的对照臂（replay 臂缺帧则跳过并注明）",
    )
    parser.add_argument(
        "--replay-step-gate-px",
        type=float,
        default=0.0,
        help="重推逐帧位移门（px/步）：单步跳太远 = 快核抓住杂波 ⇒ 该步观测作废、"
        "按最后实测位移 hold 一步继续（不弃权整帧）；0 = 不设门",
    )
    parser.add_argument(
        "--replay-anchor",
        default="kernel",
        choices=["kernel", "delta", "pred", "kf"],
        help="replay 上报坐标怎么合成：kernel=快核绝对位置（原始口径）；"
        "delta=管线锚点+快核位移（消掉快核的绝对质心偏置）；"
        "pred=同 delta 的融合，但 ROI 中心跟卡尔曼预测走（消除质心自反馈跑飞）；"
        "kf=逐帧递推卡尔曼状态+质心观测（文档档位 B 原文口径）",
    )
    parser.add_argument(
        "--replay-mode",
        default="perframe",
        choices=["perframe", "rolling", "refine"],
        help="replay 档位的实现方式：perframe=每物理帧从源帧 t-Δ 重新起链、重跑 Δ 步"
        "（默认，保持现状可复现）；rolling=一条链跨物理帧持续推进、每帧只走一步；"
        "refine=绝对位置永远锚在管线上、快核只做一次单步局部修正（零漂移、低抖动、不拒帧）",
    )
    parser.add_argument(
        "--rolling-recenter",
        type=int,
        default=1,
        choices=[0, 1],
        help="rolling：锚点到达时是否把快核 ROI 中心重锚到校正后的绝对位置"
        "（默认 1：防止链的绝对质心偏置随时间单调累积）",
    )
    parser.add_argument(
        "--rolling-hold-mode",
        default="vel",
        choices=["last_step", "prev_report", "vel"],
        help="rolling：链观测失效时的推进依据：last_step=最后一次实测单步位移；"
        "prev_report=上一次上报的帧间位移；"
        "vel=链的平滑速度（已接受步长的指数滑动平均，最稳，推荐）",
    )
    parser.add_argument(
        "--replay-obs-shrink-alpha",
        type=float,
        default=1.0,
        help="观测层 shrinkage：把灰度质心往『锚点+匀速预测』收缩的权重（1.0=不收缩）",
    )
    parser.add_argument(
        "--replay-disp-merge-sigma-px",
        type=float,
        default=0.0,
        help="位移自适应融合：w=1/(1+(d/sigma)^2)，d=实测位移与参考位移之差（0=不融合）",
    )
    parser.add_argument(
        "--replay-disp-gain",
        type=float,
        default=1.0,
        help="位移增益：对最终位移做整体放大，补偿模板配准的向心偏置（1.0=关闭）",
    )
    parser.add_argument(
        "--replay-disp-shrink-alpha",
        type=float,
        default=1.0,
        help="位移层 shrinkage：最终位移向『v·Δ』收缩的权重（1.0=不收缩）",
    )
    parser.add_argument(
        "--replay-freeze-speed-px",
        type=float,
        default=0.0,
        help="低速冻结：|参考速度| < 该阈值(px/帧) ⇒ 重推位移锁死为 0（0=不冻结）",
    )
    parser.add_argument(
        "--output-max-step-px",
        type=float,
        default=0.0,
        help="输出级限幅（伺服 slew limiter）：每帧上报位移的上限（px），0=不限幅。"
        "转台看到的是帧间位移，重推链偶发抓住杂波会造成孤立单帧飞跳（实测 190 px）"
        "⇒ 二阶差分 331 px 的抽搐；限幅只约束输出轨迹本身，不依赖运动学参考",
    )
    parser.add_argument(
        "--refine-gate-px",
        type=float,
        default=24.0,
        help="refine：快核单步修正的最大位移（px），超门 ⇒ 只用运动学预测（绝对锚点不变）",
    )
    parser.add_argument(
        "--rolling-vel-alpha",
        type=float,
        default=0.5,
        help="rolling：链平滑速度的指数滑动平均系数（0<α≤1，越大越跟手、越大越抖）",
    )
    parser.add_argument(
        "--rolling-max-step-px",
        type=float,
        default=12.0,
        help="rolling：单步位移门（px/步）：一步跳太远 = 快核抓住杂波 ⇒ 该步作废走 hold。"
        "缺这道门链会指数发散（实测单步位移能到 1e143 px）",
    )
    parser.add_argument(
        "--rolling-max-offset-px",
        type=float,
        default=40.0,
        help="rolling：锚点残差上限（px）：残差过大 = 链已跟丢 ⇒ 不采纳该锚点",
    )
    parser.add_argument(
        "--rolling-reacquire-px",
        type=float,
        default=30.0,
        help="rolling：链位置相对『锚点+Δ·v』的允许偏离（px）：超出 ⇒ 投影回边界并清零步长",
    )
    parser.add_argument(
        "--replay-consistency-gate-px",
        type=float,
        default=0.0,
        help="一致性门（px）：重推结果与『锚点+Δ·v』偏离超过它 ⇒ 判跑飞，改用运动学预测（默认关）",
    )
    parser.add_argument(
        "--replay-pred-radius-px",
        type=float,
        default=12.0,
        help="pred 档位：ROI 中心允许偏离匀速预测的最大半径（px，0=纯预测不回看）",
    )
    parser.add_argument(
        "--replay-roi-mode",
        type=str,
        default="chain",
        choices=("chain", "selfpred"),
        help="重推链 ROI 中心口径：chain（默认，窗心=链上一帧质心）| selfpred（窗心=链上一帧质心+链自身速度前推）",
    )
    parser.add_argument(
        "--replay-xcorr-radius",
        type=float,
        default=0.0,
        help="模板匹配（xcorr）位移精修的目标帧搜索半径（px）；0=关闭。原理：同一份源帧模板跨两端配准，质心偏置精确抵消",
    )
    parser.add_argument("--replay-xcorr-template", type=int, default=24, help="xcorr 模板边长（px）")
    parser.add_argument("--replay-xcorr-th-k", type=float, default=2.5, help="xcorr 模板二值化的底噪倍数")
    parser.add_argument("--replay-xcorr-min-peak", type=float, default=0.25, help="xcorr 相关峰值的最低要求")
    parser.add_argument("--replay-xcorr-bg", type=int, default=96, help="xcorr 底噪估计窗边长（px）")
    parser.add_argument(
        "--replay-xcorr-core-frac",
        type=float,
        default=0.0,
        help="xcorr 模板核心化：只保留最亮的前景像素比例（0=关；实测可剔除运动拖尾对相关峰的干扰）",
    )
    parser.add_argument(
        "--replay-xcorr-uniq-gate",
        type=float,
        default=0.0,
        help="xcorr 相关峰唯一性比门：次峰/最佳峰 > 阈值 ⇒ 判配准歧义，以锚点为中心重配一次，仅当重配更唯一且更自信才采纳（0=关）",
    )
    parser.add_argument(
        "--replay-xcorr-mode",
        type=str,
        default="direct",
        choices=["direct", "chain", "chain_direct", "stride"],
        help="xcorr 用法：direct=一次跨Δ帧大匹配；chain=逐步配准（模板每步刷新，质心偏置不累积）；chain_direct=逐步配准+末尾再做大匹配；stride=分段配准（每N帧配一次、模板每段刷新）",
    )
    parser.add_argument(
        "--replay-xcorr-chain-radius",
        type=float,
        default=0.0,
        help="chain 档位每步的搜索半径（px）；0=复用 --replay-xcorr-radius",
    )
    parser.add_argument(
        "--replay-xcorr-stride",
        type=int,
        default=0,
        help="分段配准的段长（帧）；0=关闭（仅在 --replay-xcorr-mode stride 下生效）",
    )
    parser.add_argument(
        "--replay-abstain-peak-below",
        type=float,
        default=0.0,
        help=(
            "模板配准峰值低于此值时**不上报**（不自信就别给坐标）；"
            "0 = 关（默认，历史行为，逐位不变）"
        ),
    )
    parser.add_argument(
        "--replay-abstain-disp-px",
        type=float,
        default=0.0,
        help="配合 --replay-abstain-peak-below：链位移（px）超过此值才算「跟丢」；0 = 不看位移",
    )
    parser.add_argument(
        "--replay-fov-margin-px",
        type=float,
        default=0.0,
        help=(
            "出框门：上报坐标必须落在画面内且距边界 ≥ 此值（px），否则该条上报丢弃；"
            "0 = 关（默认，历史行为，逐位不变）"
        ),
    )
    parser.add_argument(
        "--replay-rollback-max-frames",
        type=int,
        default=0,
        help=(
            "源帧（t−Δ）检测器漏检时，最多往回退多少帧去找锚点（多走几步重推）；"
            "0 = 不回退（默认，历史行为，逐位不变）"
        ),
    )
    parser.add_argument(
        "--replay-rollback-min-peak",
        type=float,
        default=0.0,
        help=(
            "回退链的相关峰值置信门：峰值低于此值该帧不上报（防「幽灵坐标」）；"
            "0 = 不设门（默认）"
        ),
    )
    parser.add_argument(
        "--replay-polarity",
        type=str,
        default="bright",
        choices=["bright", "dark", "auto"],
        help=(
            "目标极性（亮/暗）假设：bright=比背景亮（默认，历史行为）；"
            "dark=强制比背景暗；auto=在锚点帧现场判极性并锁定整条链"
            "（实测 014453 是暗目标：亮假设下 xcorr 模板为空、锁到 70~100 px 外的云边；"
            "015000 只有开头 ~150 帧是暗的、其后转亮 ⇒ 必须逐链判、不能整段判）"
        ),
    )
    parser.add_argument(
        "--replay-polarity-ratio",
        type=float,
        default=10.0,
        help=(
            "auto 模式下判暗的置信倍数：只有「暗向能量 > 倍数 × 亮向能量」才判暗。"
            "默认 10.0（实测最优：亮目标的链条几乎全被挡回亮假设、暗目标链条 99.9%% 保留）；"
            "只在 --replay-polarity auto 下生效"
        ),
    )
    parser.add_argument(
        "--replay-roi-selfpred-window",
        type=int,
        default=3,
        help="selfpred：取最近 N 步已接受步的中位数作链速度（0=用全部已接受步）",
    )
    parser.add_argument(
        "--replay-roi-selfpred-max-px",
        type=float,
        default=0.0,
        help="selfpred：ROI 前推量上限（px，0=不限）",
    )
    parser.add_argument(
        "--replay-step-rel-tol-px",
        type=float,
        default=0.0,
        help="首步离群门（px）：重推第一步位移与参考速度之差超过它 ⇒ 该步不入链（0=不设门）。"
        "实测：门真正生效后 F1 94.32%%→92.58%%（净亏 1.74pp，p95 35.7→63.4）⇒ 默认关",
    )
    parser.add_argument(
        "--replay-outlier-tol-px",
        type=float,
        default=0.0,
        help="批量稳健位移（px）：单步位移偏离步长中位数超过它 ⇒ 该步换成中位数再求和（0=不处理）",
    )
    parser.add_argument(
        "--replay-hold-window",
        type=int,
        default=3,
        help="hold 推进速度的参考窗：取最近 N 步已接受步（0=全窗口）",
    )
    parser.add_argument(
        "--replay-hold-outlier-tol-px",
        type=float,
        default=10.0,
        help="末步与参考中位数差超过它 ⇒ 末步判为离群，改用中位数推进",
    )
    parser.add_argument(
        "--replay-hold-robust",
        dest="replay_hold_robust",
        action="store_true",
        default=False,
        help="hold 用稳健参考推进（默认关：末步口径在 8 序列上 F1 最优）",
    )
    parser.add_argument(
        "--no-replay-hold-robust",
        dest="replay_hold_robust",
        action="store_false",
        help="hold 用末步口径推进",
    )
    parser.add_argument(
        "--replay-hold-use-ref",
        dest="replay_hold_use_ref",
        action="store_true",
        default=True,
        help="hold 时按稳健参考速度推进（默认开，避免把单步离群跳变放大成幻影位移）",
    )
    parser.add_argument(
        "--no-replay-hold-use-ref",
        dest="replay_hold_use_ref",
        action="store_false",
        help="hold 时按最后一次实测步长推进（旧口径）",
    )
    parser.add_argument(
        "--fast-kf-r",
        type=float,
        default=2.0,
        help="kf 档位的观测噪声标准差（px）：质心观测的可信度，越大越信 KF 自己的预测",
    )
    parser.add_argument(
        "--fast-mode",
        default="centroid",
        choices=["centroid", "peak_centroid"],
        help="快核提取模式：centroid=整块 ROI 加权质心；peak_centroid=先取局部极大再在邻域求质心",
    )
    parser.add_argument("--fast-win-size", type=int, default=32, help="快核 ROI 边长（px）")
    parser.add_argument("--fast-th-k", type=float, default=3.0, help="快核底噪倍数（门限=背景中值+th_k*sigma）")
    parser.add_argument("--fast-bg-window", type=int, default=64, help="快核底噪估计窗边长（px，必须 ≥ win_size）")
    parser.add_argument("--fast-peak-window", type=int, default=9, help="peak_centroid 模式下极大点邻域边长（px）")
    parser.add_argument("--fast-min-energy", type=float, default=1.0, help="快核最小前景能量")
    parser.add_argument("--fast-min-contrast", type=float, default=5.0, help="快核最小峰值对比度")
    parser.add_argument("--out-table", default="", help="可选的 markdown 对照表输出路径")
    return parser.parse_args()


def load_params(params_json: str) -> dict:
    """Start from the Anti-UAV golden default and override with the parameter file."""
    params = dict(DEFAULT_PARAMS)
    if params_json:
        payload = json.loads(Path(params_json).read_text(encoding="utf-8"))
        override = payload.get("params", payload)
        for key, value in override.items():
            params[key] = value
    return params


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    started = time.perf_counter()

    sequences = resolve_sequences(cache_root, args.seq, args.all_seqs, args.seqs)
    params = load_params(args.params_json)
    if args.dedup_radius is not None:
        params["dedup_radius"] = float(args.dedup_radius)

    if args.self_check:
        gate = run_correctness_gate(cache_root, labels_root, sequences)
        print("[GATE] adapter vs frozen single-frame baseline (radius=768px, th=0.22, inbox)")
        for key in ("tp", "fp", "fn", "gt"):
            print(f"  {key:>3}: expected {gate['expected'][key]:>8d} | actual {gate['actual'][key]:>8d}")
        for key in ("recall", "precision", "f1", "far_per_frame"):
            print(
                f"  {key:>13}: expected {gate['expected'][key]:.12f} | actual {gate['actual'][key]:.12f}"
            )
        if gate["passed"]:
            print("[GATE PASS]")
        else:
            print("[GATE FAIL]")
            raise SystemExit(1)
        return

    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARM_MODES]
    if unknown:
        raise SystemExit(f"未知的对照臂 {unknown}；可选 {sorted(ARM_MODES)}")
    if not arms:
        raise SystemExit("--arms 至少要有一个臂")
    # off = 原脚本自己的输出（original 臂）；其余档位直接取同名臂。
    main_arm = "original" if args.lookahead == "off" else args.lookahead
    configs = {arm: build_lookahead_config(args, ARM_MODES[arm]) for arm in arms}
    # 只有非 original 臂才需要 KF 状态；纯 original 时不捕获，省一份拷贝。
    need_kf_states = any(arm != "original" for arm in arms)

    per_sequence_rows = {}
    per_sequence_metrics = {}
    arm_rows: dict[str, dict[str, dict]] = {arm: {} for arm in arms}
    num_frames = 0
    for sequence in sequences:
        records = build_records(
            cache_root,
            labels_root,
            sequence,
            float(params["dedup_radius"]),
            0.0,
            int(params["persist_min_hits"]),
            int(params["persist_window"]),
            float(params["persist_tol"]),
        )
        frames = load_cache_meta(cache_root, sequence)["frames"]
        if args.no_pipeline:
            dets = single_frame_dets(records, float(params["th_base"]))
            system_res = None
        else:
            dets, system_res = run_system_sota(
                records, params, img_h=IMG_H, capture_kf_states=need_kf_states
            )

        # ---- 四臂对照（尺子 1 / 尺子 2）----
        frame_source = None
        if "replay" in arms:
            frame_source = build_frame_source(
                args.frames_root,
                args.frames_cache,
                sequence,
                [int(record["frame_index"]) for record in records],
                int(args.delta),
                int(args.replay_rollback_max_frames),
            )
            if frame_source is None:
                print(
                    f"[ARM] {sequence} 缺帧（cache={args.frames_cache} root={args.frames_root}），"
                    "replay 臂跳过",
                    flush=True,
                )
        sequence_arms = [arm for arm in arms if arm != "replay" or frame_source is not None]
        if main_arm not in sequence_arms:
            raise SystemExit(f"[ARM] 主臂 {main_arm} 在序列 {sequence} 上不可用（缺帧）")

        arm_result, arm_reports = evaluate_arms(
            records,
            dets,
            frames,
            sequence_arms,
            configs,
            int(args.delta),
            system_res.get("kf_states") if system_res else None,
            frame_source,
        )
        for arm, payload in arm_result.items():
            arm_rows[arm][sequence] = payload
        main_report = arm_reports[main_arm]

        row = evaluate_biz(records, main_report, num_frames=frames)
        per_sequence_metrics[sequence] = row
        per_sequence_rows[sequence] = row
        num_frames += frames

        if args.out_preds_dir:
            config = {
                "sequence": sequence,
                "cache_root": str(cache_root),
                "labels_root": str(labels_root),
                "params": params,
                "dedup_radius": float(params["dedup_radius"]),
                "img_w": IMG_W,
                "img_h": IMG_H,
                "match_mode": "inbox",
                "tol": 8.0,
                "source": "system_sota" if main_arm == "original" else f"lookahead_{args.lookahead}",
            }
            write_sequence_predictions(
                args.out_preds_dir, sequence, config, row, frame_payload(records, main_report, frames)
            )
        print(
            f"[SEQ] {sequence} frames={frames} GT={row['gt']} TP={row['tp']} FP={row['fp']} FN={row['fn']} "
            f"R={row['recall'] * 100:.2f}% P={row['precision'] * 100:.2f}% F1={row['f1'] * 100:.3f}% "
            f"FAR={row['far_per_frame']:.4f}",
            flush=True,
        )

    overall = aggregate_metrics(list(per_sequence_metrics.values()), num_frames)
    print(
        f"\n[OVERALL] frames={num_frames} GT={overall['gt']} TP={overall['tp']} FP={overall['fp']} "
        f"FN={overall['fn']} (absent={overall['fn_absent']} compete={overall['fn_compete']}) "
        f"R={overall['recall'] * 100:.2f}% P={overall['precision'] * 100:.2f}% F1={overall['f1'] * 100:.3f}% "
        f"FAR={overall['far_per_frame']:.4f}",
        flush=True,
    )

    # ---- 汇总四臂 ----
    arm_summary: dict[str, dict] = {}
    for arm in arms:
        rows = arm_rows[arm]
        if not rows:
            continue
        source_rows = {sequence: payload["source_frame"] for sequence, payload in rows.items()}
        current_rows = {sequence: payload["current_frame"] for sequence, payload in rows.items()}
        servo_rows = {sequence: payload["servo"] for sequence, payload in rows.items()}
        # 汇总时把逐帧 CLE 样本抽出来做池化分位数；样本本身不进 JSON（会撑爆产物）。
        pooled_cle: list[float] = []
        for payload in rows.values():
            pooled_cle.extend(payload["servo"].pop("cle_values", []))
        stats_rows = {sequence: payload["stream_stats"] for sequence, payload in rows.items()}
        arm_summary[arm] = {
            "mode": ARM_MODES[arm],
            "delta": 0 if arm == "original" else int(args.delta),
            "source_frame": {
                "window": int(next(iter(rows.values()))["source_window"]),
                "overall": aggregate_metrics(list(source_rows.values()), num_frames),
                "per_sequence": source_rows,
            },
            "current_frame": {
                "overall": aggregate_metrics(list(current_rows.values()), num_frames),
                "per_sequence": current_rows,
            },
            "servo": {
                "overall": aggregate_servo(
                    servo_rows,
                    sum(int(row["report_frames"]) for row in rows.values()),
                    pooled_cle=pooled_cle,
                ),
                "per_sequence": servo_rows,
            },
            "stream_stats": {
                "overall": {
                    "applied": sum(int(row.get("applied", 0)) for row in stats_rows.values()),
                    "abstained": sum(int(row.get("abstained", 0)) for row in stats_rows.values()),
                    "fallback_used": sum(int(row.get("fallback_used", 0)) for row in stats_rows.values()),
                    "missing_velocity": sum(int(row.get("missing_velocity", 0)) for row in stats_rows.values()),
                    "replay_steps": sum(int(row.get("replay_steps", 0)) for row in stats_rows.values()),
                    "replay_failed": sum(int(row.get("replay_failed", 0)) for row in stats_rows.values()),
                    "replay_step_jump": sum(int(row.get("replay_step_jump", 0)) for row in stats_rows.values()),
                    "replay_max_step_disp": max(
                        (float(row.get("replay_max_step_disp", 0.0)) for row in stats_rows.values()),
                        default=0.0,
                    ),
                    "replay_xcorr_ambiguous": sum(
                        int(row.get("replay_xcorr_ambiguous", 0)) for row in stats_rows.values()
                    ),
                    "replay_xcorr_uniq_sum": sum(
                        float(row.get("replay_xcorr_uniq_sum", 0.0)) for row in stats_rows.values()
                    ),
                    "replay_xcorr_uniq_n": sum(
                        int(row.get("replay_xcorr_uniq_n", 0)) for row in stats_rows.values()
                    ),
                    "fov_gated": sum(int(row.get("fov_gated", 0)) for row in stats_rows.values()),
                },
                "per_sequence": stats_rows,
            },
            "source_verbatim": {
                sequence: payload["source_verbatim"] for sequence, payload in rows.items()
            },
            # 从磁盘读了多少张灰度帧：replay 臂的「每帧只读一次」靠这个数验收。
            "frame_reads": {
                sequence: payload["frame_reads"] for sequence, payload in rows.items()
            },
        }

    # ---- 尺子 1 的硬门禁：各臂的源帧口径必须与 original 臂逐位相等 ----
    if "original" in arm_summary:
        original_f1 = arm_summary["original"]["source_frame"]["overall"]["f1"]
        for arm, summary in arm_summary.items():
            delta_f1 = summary["source_frame"]["overall"]["f1"] - original_f1
            summary["delta_f1_vs_original"] = delta_f1
            if delta_f1 != 0.0:
                raise SystemExit(
                    f"[GATE] 尺子1 失败：{arm} 臂的源帧口径 F1 与 original 臂不相等（ΔF1={delta_f1:.9f}）"
                )
        print("[GATE] 尺子1 各臂源帧口径 ΔF1 = 0.000000", flush=True)

    for arm, summary in arm_summary.items():
        current = summary["current_frame"]["overall"]
        servo = summary["servo"]["overall"]
        print(
            f"[ARM] {arm:<10} 尺子1 F1={summary['source_frame']['overall']['f1']:.6f} | "
            f"尺子2 F1={current['f1']:.6f} R={current['recall'] * 100:.2f}% "
            f"P={current['precision'] * 100:.2f}% FAR={current['far_per_frame']:.4f} | "
            f"CLE p95={float(servo.get('cle_p95', 0.0)):.2f} "
            f"hit@5px={float(servo.get('hit_rate_5px', 0.0)) * 100:.2f}%",
            flush=True,
        )

    if args.expect_f1 is not None:
        f1_ok = abs(overall["f1"] - args.expect_f1) < 5e-6
        tp_ok = args.expect_tp is None or overall["tp"] == args.expect_tp
        fp_ok = args.expect_fp is None or overall["fp"] == args.expect_fp
        passed = f1_ok and tp_ok and fp_ok
        print(
            f"[GATE] 期望 F1={args.expect_f1:.6f} TP={args.expect_tp} FP={args.expect_fp} | "
            f"实测 F1={overall['f1']:.6f} TP={overall['tp']} FP={overall['fp']} -> "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        if not passed:
            print("[GATE] 未复现冻结 baseline，后续对比不可信 —— 中止")
            raise SystemExit(2)

    if args.out_preds_dir:
        write_summary(args.out_preds_dir, params, overall, per_sequence_metrics)
        print(f"[SUCCESS] predictions -> {Path(args.out_preds_dir).resolve()}")
    if args.out_table:
        table_path = Path(args.out_table)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(render_arm_table(arm_summary, int(args.delta), main_arm), encoding="utf-8")
        print(f"[SUCCESS] table -> {table_path.resolve()}")
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "cache_root": str(cache_root),
                        "labels_root": str(labels_root),
                        "params": params,
                        "img_w": IMG_W,
                        "img_h": IMG_H,
                        "match_mode": "inbox",
                        "tol": 8.0,
                        "pipeline": not args.no_pipeline,
                    },
                    "overall": overall,
                    "per_sequence": per_sequence_metrics,
                    "arms": arm_summary,
                    "lookahead": {
                        "mode": args.lookahead,
                        "main_arm": main_arm,
                        "delta": int(args.delta),
                        "velocity_source": args.velocity_source,
                        "velocity_window": int(args.velocity_window),
                        "position_source": args.position_source,
                        "replay_require_valid": bool(args.replay_require_valid),
                        "replay_fallback": args.replay_fallback,
                        "replay_step_gate_px": float(args.replay_step_gate_px),
                        # ↓ 2026-11-03：把**实际生效的快核/重推旋钮**写进产物。
                        # 起因：旧产物只记 12 个字段，导致"这份 F1 是哪套参数跑出来的"
                        # 无法从 JSON 反查（win48 / win52 / win54 的产物数字完全一样，
                        # 只能靠文件名猜）。现在每份产物自带完整旋钮快照。
                        "fast_mode": str(args.fast_mode),
                        "fast_win_size": int(args.fast_win_size),
                        "fast_bg_window": int(args.fast_bg_window),
                        "fast_th_k": float(args.fast_th_k),
                        "fast_min_energy": float(args.fast_min_energy),
                        "fast_min_contrast": float(args.fast_min_contrast),
                        "replay_anchor": str(args.replay_anchor),
                        "replay_anchor_check_px": float(args.replay_anchor_check_px),
                        "replay_mode": str(args.replay_mode),
                        "replay_max_hold": int(args.replay_max_hold),
                        "replay_hold_use_ref": bool(args.replay_hold_use_ref),
                        "replay_hold_robust": bool(args.replay_hold_robust),
                        "replay_hold_window": int(args.replay_hold_window),
                        "replay_hold_outlier_tol_px": float(args.replay_hold_outlier_tol_px),
                        "replay_step_rel_tol_px": float(args.replay_step_rel_tol_px),
                        "replay_outlier_tol_px": float(args.replay_outlier_tol_px),
                        "replay_consistency_gate_px": float(args.replay_consistency_gate_px),
                        "replay_obs_shrink_alpha": float(args.replay_obs_shrink_alpha),
                        "replay_freeze_speed_px": float(args.replay_freeze_speed_px),
                        "replay_disp_shrink_alpha": float(args.replay_disp_shrink_alpha),
                        "replay_disp_merge_sigma_px": float(args.replay_disp_merge_sigma_px),
                        "replay_disp_gain": float(args.replay_disp_gain),
                        "replay_roi_mode": str(args.replay_roi_mode),
                        "replay_roi_selfpred_window": int(args.replay_roi_selfpred_window),
                        "replay_roi_selfpred_max_px": float(args.replay_roi_selfpred_max_px),
                        "replay_xcorr_radius": float(args.replay_xcorr_radius),
                        "replay_xcorr_template": int(args.replay_xcorr_template),
                        "replay_xcorr_th_k": float(args.replay_xcorr_th_k),
                        "replay_xcorr_min_peak": float(args.replay_xcorr_min_peak),
                        "replay_xcorr_bg": int(args.replay_xcorr_bg),
                        "replay_xcorr_core_frac": float(args.replay_xcorr_core_frac),
                        "replay_xcorr_uniq_gate": float(args.replay_xcorr_uniq_gate),
                        "replay_xcorr_mode": str(args.replay_xcorr_mode),
                        "replay_xcorr_chain_radius": float(args.replay_xcorr_chain_radius),
                        "replay_xcorr_stride": int(args.replay_xcorr_stride),
                        "replay_rollback_max_frames": int(args.replay_rollback_max_frames),
                        "replay_rollback_min_peak": float(args.replay_rollback_min_peak),
                        # ↓ 2026-11-04：补记**此前漏记**的弃权/回退旋钮 —— 缺了它们
                        # 就无法从产物 JSON 反查「这份 F1 是哪套参数跑出来的」
                        # （本轮 p3knobs 变体的 diff 只看得到 rollback_min_peak）。
                        "replay_abstain_peak_below": float(args.replay_abstain_peak_below),
                        "replay_abstain_disp_px": float(args.replay_abstain_disp_px),
                        "replay_fov_margin_px": float(args.replay_fov_margin_px),
                        "replay_polarity": str(args.replay_polarity),
                        "replay_polarity_ratio": float(args.replay_polarity_ratio),
                        "output_max_step_px": float(args.output_max_step_px),
                        "arms": arms,
                        "frames_root": args.frames_root,
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[SUCCESS] report -> {path.resolve()}")
    print(f"[SUCCESS] elapsed {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
