#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Longquanshan adapter for the Anti-UAV system-SOTA post-processing pipeline.

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
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_bidirectional_track_fusion import (  # noqa: E402
    evaluate_sequence_bidirectional,
    natural_sort_key,
)
from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    COUNT_KEYS,
    count_matches,
    load_ground_truth,
    load_predictions,
    merge_scales,
    metrics,
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


def run_system_sota(records: list[dict], params: dict, img_h: int = IMG_H) -> tuple[list[list[dict]], dict]:
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


def resolve_sequences(cache_root, sequence: str, all_seqs: bool) -> list[str]:
    """Either the requested sequence or every cached sequence, sorted."""
    if all_seqs:
        sequences = sorted(path.stem for path in Path(cache_root).glob("*.pkl"))
        if not sequences:
            raise FileNotFoundError(f"No .pkl caches under {cache_root}")
        return sequences
    if not sequence:
        raise ValueError("either --seq or --all-seqs is required")
    return [sequence]


# ==============================================================================
# 6. CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anti-UAV system-SOTA pipeline on the Longquanshan cache")
    parser.add_argument("--cache-root", required=True, help="Directory of <sequence>.pkl heatmap caches")
    parser.add_argument("--labels-root", required=True, help="Directory of {sequence}__{index:06d}.txt labels")
    parser.add_argument("--seq", default="", help="Single sequence name")
    parser.add_argument("--all-seqs", action="store_true", help="Run every cached sequence")
    parser.add_argument("--dedup-radius", type=float, default=None, help="Cross-scale de-duplication radius (px)")
    parser.add_argument("--params-json", default="", help="JSON file with the full parameter set")
    parser.add_argument("--out-json", default="", help="Optional run report path")
    parser.add_argument("--out-preds-dir", default="", help="Optional directory for the per-sequence predictions")
    parser.add_argument("--no-pipeline", action="store_true", help="Single-frame arm only, no system post-processing")
    parser.add_argument("--self-check", action="store_true", help="Run the adapter correctness gate and exit")
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

    sequences = resolve_sequences(cache_root, args.seq, args.all_seqs)
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

    per_sequence_rows = {}
    per_sequence_metrics = {}
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
            dets, system_res = run_system_sota(records, params, img_h=IMG_H)
        row = evaluate_biz(records, dets, num_frames=frames)
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
                "source": "system_sota",
            }
            write_sequence_predictions(
                args.out_preds_dir, sequence, config, row, frame_payload(records, dets, frames)
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

    if args.out_preds_dir:
        write_summary(args.out_preds_dir, params, overall, per_sequence_metrics)
        print(f"[SUCCESS] predictions -> {Path(args.out_preds_dir).resolve()}")
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
