#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render 龙泉山-style **tracked** OSD review videos for a domain without ground truth.

This is the 「纵横AI数据采集」 tracked sibling of ``render_rc2_center_osd_video.py``. The
single-frame renderer draws one centre marker per detection. This one runs the **龙泉山 business-domain
spatio-temporal trajectory post-processing** on top of the same multi-scale detections, and draws the
*final system detections* that survive it:

    per-scale heatmap rows  ->  cross-scale merge (dedup_radius)
                            ->  OnlineAdaptiveTracker  (bidirectional, sky/ground tiered thresholds)
                            ->  BidirectionalTemporalSmoother  (stitch / infill / rigid pruning / hover)
                            ->  consecutive-frame confirmation  (confirm_consec)
                            ->  one red circle per surviving track observation, labelled T<id>:<score>

The tracker consumes **candidate points only**. Ground truth is used by the Longquan pipeline purely to
score the result (and to keep frames that have no detection in the frame set), so this script builds the
record set from the candidate frames **plus the cache's full frame count**. That keeps the frame index a
real time axis -- without the padding, every empty frame would be dropped, the tracker would see a
compressed index, and ``max_age`` / ``min_hits`` / ``confirm_consec`` would all mean something else.

Two things are deliberately **not** copied from 龙泉山:
  * the feature semantics stay 纵横AI's own (official per-lag GMC fit, computed in memory from the mp4),
    so the detections are the same ones the single-frame OSD showed;
  * the thresholds are re-measured on this domain instead of being copied from the 龙泉山 delivery file,
    because a threshold tuned on 龙泉山's feature statistics does not transfer.

Usage
-----
    python manu/videos/render_rc2_tracked_osd_video.py \
        --input-root /mnt/data/siping/datasets/manu/纵横AI数据采集_mp4 \
        --output-dir  /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_tracked_th020 \
        --threshold 0.20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_full_median_dataset import FastGMCEstimator  # noqa: E402
from manu.data.extract_video_frames import safe_name  # noqa: E402
from manu.data.multiscale_heatmap_cache import (  # noqa: E402
    compute_features,
    detect_components,
    heatmap_to_native,
    run_heatmaps,
    scale_inputs,
)
from manu.data.preannotate_hm_bbox import load_hm_model, resolve_weights  # noqa: E402
from manu.evaluation.system_sota_longquan import (  # noqa: E402
    apply_consecutive_confirmation,
    build_smoother_config,
    build_tracker_config,
)
from manu.evaluation.eval_bidirectional_track_fusion import evaluate_sequence_bidirectional  # noqa: E402
from manu.videos.render_pred_gt_osd_video import PAD, ROW_HEADER, zoom_panel  # noqa: E402

COLOR_DET = (0, 0, 255)
COLOR_IDLE = (120, 120, 120)

# The trajectory parameters are read from the 龙泉山 RC2 delivery working point
# (system_sota_longquan_trial0080_delivery_v3_params.json) rather than duplicated here, so they cannot
# drift from the file the delivery arm actually used. The *detection* thresholds are NOT taken from
# there: a threshold measured on 龙泉山's feature statistics does not transfer to this domain, and this
# domain has no ground truth to re-measure against, so they are set explicitly on the command line.
LONGQUAN_PARAMS_JSON = PROJECT_ROOT / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"

REQUIRED_TRAJECTORY_PARAMS = {
    "max_age", "min_hits", "match_dist", "max_match_dist", "min_track_score", "instant_conf",
    "min_disp", "sky_ratio", "th_deep", "min_hits_deep", "stitch_gap", "stitch_long_gap",
    "stitch_vel_diff", "infill_gap", "min_hits_infill", "min_rigid_disp", "max_rigid_var",
    "min_rigid_disp_sky", "min_hits_prune", "hover_vel", "hover_infill_gap", "coast_frames",
    "min_hits_hover", "hover_sky_only", "confirm_consec", "confirm_gap_tol", "confirm_trim_prefix",
}


def load_longquan_params() -> dict:
    """Load the 龙泉山 delivery trajectory parameters (detection thresholds are overridden by the caller)."""
    data = json.loads(LONGQUAN_PARAMS_JSON.read_text(encoding="utf-8"))
    params = data.get("params", data)
    missing = sorted(REQUIRED_TRAJECTORY_PARAMS - set(params))
    if missing:
        raise SystemExit(f"{LONGQUAN_PARAMS_JSON.name} is missing trajectory parameters: {missing}")
    return dict(params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tracked (龙泉山-style) OSD review videos")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Single input video")
    source.add_argument("--input-root", help="Directory of input videos to render as a batch")
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--shard", default="", help="Render only shard INDEX/COUNT of the sorted video list")
    parser.add_argument("--output", default="", help="Output MP4; required with --input")
    parser.add_argument("--output-dir", default="", help="Output directory used with --input-root")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--hm-weights", default="runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--scales", default="160,80")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Detection threshold of the merged rows (the 龙泉山 track arm calls this th_base)",
    )
    parser.add_argument("--th-ground", type=float, default=0.20, help="Ground-region threshold")
    parser.add_argument("--th-salvage", type=float, default=0.06, help="Weak-pulse salvage threshold")
    parser.add_argument("--region-threshold", type=float, default=0.06)
    parser.add_argument("--merge-radius", type=float, default=768.0)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--crop-left", type=int, default=320)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=1280)
    parser.add_argument("--crop-height", type=int, default=1024)
    parser.add_argument("--native-width", type=int, default=640)
    parser.add_argument("--native-height", type=int, default=512)
    parser.add_argument("--canvas", default="1920x1080")
    parser.add_argument("--zoom", type=int, default=360)
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--out-json", default="", help="Optional per-video detection manifest")
    parser.add_argument(
        "--out-preds-dir",
        default="",
        help="Optional directory for the per-sequence track predictions (龙泉山 preds_track layout)",
    )
    parser.add_argument(
        "--out-cache-dir",
        default="",
        help="Optional directory for the per-sequence heatmap cache pkl (龙泉山 cache layout)",
    )
    parser.add_argument(
        "--out-frames-dir",
        default="",
        help=(
            "Optional directory for the per-frame detection records (one <sequence>.jsonl per sequence), "
            "in the same format the single-frame renderer writes, so the sequence partition and the "
            "annotation seed can be built from either tier with the same tooling"
        ),
    )
    args = parser.parse_args()

    args.scales = sorted({int(item) for item in args.scales.split(",") if item.strip()}, reverse=True)
    if not args.scales or min(args.scales) <= 0:
        parser.error("--scales must list positive downsample targets")
    if args.min_area < 1 or args.max_detections < 1:
        parser.error("--min-area and --max-detections must be positive")
    if args.merge_radius <= 0:
        parser.error("--merge-radius must be > 0")
    if args.median_window < 1 or args.median_window % 2 == 0:
        parser.error("--median-window must be a positive odd number")
    if args.temporal_stride < 1:
        parser.error("--temporal-stride must be positive")
    try:
        canvas_width, canvas_height = (int(value) for value in args.canvas.lower().split("x"))
    except ValueError:
        parser.error("--canvas must look like WIDTHxHEIGHT")
    if canvas_width < 640 or canvas_height < 480:
        parser.error("--canvas must be at least 640x480")
    args.canvas = (canvas_width, canvas_height)
    if args.zoom and (args.crop_size <= 0 or args.crop_size % 2):
        parser.error("--crop-size must be a positive even number when the zoom panel is enabled")
    args.shard_index, args.shard_count = None, None
    if args.shard:
        try:
            args.shard_index, args.shard_count = (int(item) for item in args.shard.split("/"))
        except ValueError:
            parser.error("--shard must look like INDEX/COUNT")
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            parser.error(f"--shard index must satisfy 0 <= INDEX < COUNT, got {args.shard}")
    if args.input and not args.output:
        parser.error("--input requires --output")
    if args.input_root and not args.output_dir:
        parser.error("--input-root requires --output-dir")
    return args


def collect_jobs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.input:
        source = Path(args.input)
        if not source.is_file():
            raise FileNotFoundError(source)
        return [(source, Path(args.output))]

    root = Path(args.input_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    videos = sorted(root.glob(args.pattern))
    if not videos:
        raise FileNotFoundError(f"No videos matching '{args.pattern}' under {root}")
    if args.shard_count:
        videos = [path for index, path in enumerate(videos) if index % args.shard_count == args.shard_index]
    output_dir = Path(args.output_dir)
    jobs = []
    taken: dict[str, Path] = {}
    for video in videos:
        name = f"{safe_name(video)}_osd.mp4"
        if name in taken:
            raise RuntimeError(f"Output name collision: {taken[name].name} and {video.name} both map to {name}")
        taken[name] = video
        output = output_dir / name
        if args.skip_existing and output.exists():
            if args.out_cache_dir and (Path(args.out_cache_dir) / f"{video.stem}.pkl").exists():
                continue
            output.unlink(missing_ok=True)
        jobs.append((video, output))
    return jobs


def restore_native(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Cut the recorded letterbox back to the native frame the frozen model expects."""
    height, width = gray.shape[:2]
    right, bottom = args.crop_left + args.crop_width, args.crop_top + args.crop_height
    if args.crop_left < 0 or args.crop_top < 0 or right > width or bottom > height:
        raise ValueError(f"crop box {(args.crop_left, args.crop_top, right, bottom)} exceeds frame {width}x{height}")
    crop = gray[args.crop_top : bottom, args.crop_left : right]
    return cv2.resize(crop, (args.native_width, args.native_height), interpolation=cv2.INTER_AREA)


def collect_candidates(
    model, device: torch.device, video_path: Path, args: argparse.Namespace
) -> tuple[dict[int, tuple], int]:
    """Stream the video and collect the per-scale rows the 龙泉山 tracker consumes.

    Returns the per-frame candidate pool in the same shape ``build_candidates`` produces: one
    ``(scores, points)`` pair per frame, already de-duplicated across the scales. The frame set is the
    full time axis, so a frame with no detection is still present as an empty entry.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    gmc = FastGMCEstimator(downscale=2)
    history: list[np.ndarray] = []
    history_window = args.median_window * args.temporal_stride + 2
    scale_tags = list(range(len(args.scales) + 1))
    per_tag: dict[int, dict[int, tuple]] = {tag: {} for tag in scale_tags}
    frame_number = -1
    total_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            total_frames = max(total_frames, frame_number + 1)
            gray = restore_native(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args)
            features, history = compute_features([gray], history, args, gmc)
            history = history[-history_window:]
            canvases, mappings = scale_inputs(
                features[0], args.scales, args.imgsz, args.native_height, args.native_width
            )
            heatmaps = run_heatmaps(model, np.stack(canvases), device, args.batch_size)
            for tag in scale_tags:
                heatmap = heatmap_to_native(
                    heatmaps[tag], *mappings[tag], args.imgsz, args.native_width, args.native_height
                )
                entries = detect_components(
                    heatmap, args.region_threshold, 1, args.min_area, args.max_detections
                )
                if entries:
                    scores = np.asarray([entry[5] for entry in entries], dtype=np.float32)
                    points = np.asarray([(entry[0], entry[1]) for entry in entries], dtype=np.float32)
                    per_tag[tag][frame_number] = (scores, points)
    finally:
        capture.release()

    candidates = merge_across_scales(per_tag, scale_tags, args.merge_radius, args.threshold)
    # Pad to the full time axis: a frame with no detection must stay in the record set, otherwise the
    # tracker's frame index stops being a time axis and max_age / confirm_consec change meaning.
    for frame_index in range(total_frames):
        candidates.setdefault(frame_index, (np.zeros(0, np.float32), np.zeros((0, 2), np.float32)))
    return candidates, total_frames


def merge_across_scales(
    per_tag: dict[int, dict[int, tuple]], scale_tags: list[int], radius: float, threshold: float
) -> dict[int, tuple]:
    """Union the per-scale rows of a frame, keeping the highest score inside each radius.

    Same greedy rule as ``merge_scales`` in the evaluation pipeline: the rows are pooled, sorted by
    descending score, and a row is dropped when another kept row lies within ``radius`` pixels.
    """
    merged: dict[int, tuple] = {}
    radius_squared = radius * radius
    frames = set().union(*(set(per_tag[tag]) for tag in scale_tags)) if scale_tags else set()
    for frame in frames:
        pooled = []
        for tag in scale_tags:
            if frame not in per_tag[tag]:
                continue
            scores, points = per_tag[tag][frame]
            pooled.extend((float(score), points[i].tolist()) for i, score in enumerate(scores))
        pooled.sort(key=lambda item: -item[0])
        kept, kept_xy = [], []
        for score, point in pooled:
            if score < threshold:
                continue
            x, y = point
            if radius > 0 and any((x - px) ** 2 + (y - py) ** 2 <= radius_squared for px, py in kept_xy):
                continue
            kept.append((score, point))
            kept_xy.append((x, y))
        merged[frame] = (
            np.asarray([item[0] for item in kept], np.float32),
            np.asarray([item[1] for item in kept], np.float32).reshape(-1, 2),
        )
    return merged


def build_records(candidates: dict[int, tuple], sequence: str, frames: int) -> list[dict]:
    """Assemble the tracker's input records. No ground truth: this domain has none.

    The frame set is the full time axis (``frames`` entries), so ``f_idx`` inside the tracker is the
    real frame number and the temporal parameters keep their meaning.
    """
    records = []
    for frame_index in range(frames):
        found = candidates.get(frame_index)
        if found is None:
            points = np.zeros((0, 2), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            scores, points = found
            scores = np.asarray(scores, dtype=np.float32)
            points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        records.append(
            {
                "im_name": f"{sequence}__{frame_index:06d}.jpg",
                "frame_index": int(frame_index),
                "pred_points": points,
                "pred_scores": scores,
                "gt_pts": np.zeros((0, 2), dtype=np.float32),
                "gt_bboxes": np.zeros((0, 4), dtype=np.float32),
            }
        )
    return records


def run_tracks(records: list[dict], args: argparse.Namespace) -> list[list[dict]]:
    """Run the 龙泉山 spatio-temporal trajectory post-processing.

    The tracker is fed the candidate points only. The thresholds are the ones measured on this
    domain; the trajectory parameters are the 龙泉山 ones.
    """
    params = load_longquan_params()
    params["th_base"] = args.threshold
    params["th_ground"] = args.th_ground
    params["th_salvage"] = args.th_salvage
    img_h = args.native_height

    result = evaluate_sequence_bidirectional(
        records=records,
        dist_thresh=8.0,
        th_base=float(params["th_base"]),
        th_salvage=float(params["th_salvage"]),
        th_ground=float(params["th_ground"]),
        sky_ratio=float(params["sky_ratio"]),
        img_h=img_h,
        tracker_config=build_tracker_config(params, img_h),
        smoother_config=build_smoother_config(params, img_h),
        match_mode="bbox",
    )
    dets = apply_consecutive_confirmation(
        result["bidi_frame_dets"],
        confirm_consec=int(params["confirm_consec"]),
        confirm_gap_tol=int(params["confirm_gap_tol"]),
        confirm_trim_prefix=int(params["confirm_trim_prefix"]),
    )
    return dets


def write_cache_pkl(path: Path, sequence: str, candidates: dict[int, tuple], args, frames: int) -> None:
    """Write the per-sequence heatmap cache in the 龙泉山 layout.

    Only the merged candidate rows are stored (the track arm reads the per-scale rows, but the
    de-duplication already happened, so storing the merged pool keeps the file honest about what the
    tracker actually saw). The meta records the same identity fields as the 龙泉山 cache, so the two
    cannot be silently mixed.
    """
    frame_indices, points, sizes, scores = [], [], [], []
    for frame_index in sorted(candidates):
        found_scores, found_points = candidates[frame_index]
        if not len(found_scores):
            continue
        frame_indices.append(frame_index)
        for score, point in zip(found_scores, found_points):
            points.append((float(point[0]), float(point[1])))
            sizes.append((0.0, 0.0))
            scores.append(float(score))
    counts = [0] * len(frame_indices)
    cache = {
        "meta": {
            "sequence": sequence,
            "frames": frames,
            "native_size": [args.native_width, args.native_height],
            "imgsz": args.imgsz,
            "stride": 2,
            "scales": [args.native_width] + args.scales,
            "fused_tag": 255,
            "coordinate_space": "native_pixel",
            "source": "raw_frames",
            "region_threshold": args.region_threshold,
            "main_threshold": args.threshold,
            "fusion_dilate": 0,
            "min_area": args.min_area,
            "features_root": "",
            "hm_weights": str(args.hm_weights),
            "use_p0_highway": False,
            "adapter_weights": "",
            "adapter_mid_channels": 0,
            "adapter_guide_channel": -1,
            "adapter_single": False,
            "note": "merged candidate rows of the tracked OSD renderer (single pass, no separate cache build)",
        },
        "frame_index": np.asarray(frame_indices, dtype=np.int32),
        "offset": np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))]).astype(np.int64),
        "points": np.asarray(points, dtype=np.float32).reshape(-1, 2),
        "sizes": np.asarray(sizes, dtype=np.float32).reshape(-1, 2),
        "scores": np.asarray(scores, dtype=np.float16),
        "sums": np.zeros(len(scores), dtype=np.float32),
        "area": np.zeros(len(scores), dtype=np.int32),
        "tag": np.zeros(len(scores), dtype=np.uint8),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))


def render_video(
    model,
    device: torch.device,
    video_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict:
    """Render one video: detect, run the trajectory post-processing, then draw the surviving tracks."""
    started = time.perf_counter()
    candidates, frames = collect_candidates(model, device, video_path, args)
    records = build_records(candidates, video_path.stem, frames)
    dets = run_tracks(records, args)

    if args.out_cache_dir:
        write_cache_pkl(Path(args.out_cache_dir) / f"{video_path.stem}.pkl", video_path.stem, candidates, args, frames)
    if args.out_preds_dir:
        preds_dir = Path(args.out_preds_dir)
        preds_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sequence": video_path.stem,
            "per_frame": {
                str(frame_index): [
                    {
                        "pos": [float(det["pos"][0]), float(det["pos"][1])],
                        "score": float(det["score"]),
                        "track_id": int(det["track_id"]),
                        "infilled": bool(det.get("infilled", False)),
                    }
                    for det in frame_dets
                ]
                for frame_index, frame_dets in enumerate(dets)
                if frame_dets
            },
        }
        (preds_dir / f"{video_path.stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.out_frames_dir:
        # The same per-frame record format as the single-frame renderer, but carrying the *tracked*
        # detections (centre + score), so the same partition / prelabel tooling serves either tier.
        frames_dir = Path(args.out_frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        with (frames_dir / f"{video_path.stem}.jsonl").open("w", encoding="utf-8") as handle:
            for frame_index, frame_dets in enumerate(dets):
                record = {
                    "sequence": video_path.stem,
                    "video": video_path.name,
                    "frame": frame_index,
                    "detections": [
                        {
                            "cx": round(float(det["pos"][0]), 2),
                            "cy": round(float(det["pos"][1]), 2),
                            "score": round(float(det["score"]), 4),
                            "track_id": int(det["track_id"]),
                            "infilled": bool(det.get("infilled", False)),
                        }
                        for det in frame_dets
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Second pass over the video to draw: the trajectory post-processing needs the whole sequence
    # before any frame can be drawn, so the pixels are read again for the OSD panel.
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    output_fps = args.fps if args.fps > 0 else source_fps
    zoom = min(args.zoom, args.canvas[1] - ROW_HEADER - 2 * PAD) if args.zoom else 0
    frame_area_w = args.canvas[0] - 2 * PAD - (zoom + PAD if zoom else 0)
    frame_area_h = args.canvas[1] - ROW_HEADER - 2 * PAD
    factor = min(frame_area_w / args.native_width, frame_area_h / args.native_height)
    panel_size = (int(round(args.native_width * factor)), int(round(args.native_height * factor)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, tuple(args.canvas))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open output video: {output_path}")

    rendered = hit_frames = detections = 0
    frame_number = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_number >= len(dets):
                break
            gray = restore_native(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args)
            panel = cv2.cvtColor(cv2.resize(gray, panel_size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
            frame_dets = dets[frame_number]
            count = len(frame_dets)
            label_slots: list[tuple[int, int]] = []
            for det in frame_dets:
                x = int(round(float(det["pos"][0]) * factor))
                y = int(round(float(det["pos"][1]) * factor))
                cv2.circle(panel, (x, y), max(2, int(round(3 * factor))), COLOR_DET, -1)
                cv2.circle(panel, (x, y), max(4, int(round(7 * factor))), COLOR_DET, 1)
                tag = "Infill" if det.get("infilled", False) else f"{float(det['score']):.2f}"
                offset = 14
                while any(
                    abs(slot_x - x) < 80 and abs(slot_y - (y + offset)) < 12 for slot_x, slot_y in label_slots
                ):
                    offset += 13
                label_slots.append((x, y + offset))
                cv2.putText(
                    panel,
                    f"T{int(det['track_id'])}:{tag}",
                    (x + 6, y + offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COLOR_DET,
                    1,
                    cv2.LINE_AA,
                )
            detections += count

            canvas = np.zeros((args.canvas[1], args.canvas[0], 3), np.uint8)
            panel_x, panel_y = PAD, ROW_HEADER + PAD
            canvas[panel_y : panel_y + panel.shape[0], panel_x : panel_x + panel.shape[1]] = panel
            if zoom:
                if count:
                    centre = (float(frame_dets[0]["pos"][0]), float(frame_dets[0]["pos"][1]))
                    color = COLOR_DET
                else:
                    centre, color = (args.native_width / 2, args.native_height / 2), COLOR_IDLE
                patch = zoom_panel(gray, centre, args.crop_size, zoom, color)
                zoom_x, zoom_y = args.canvas[0] - PAD - zoom, ROW_HEADER + PAD
                canvas[zoom_y : zoom_y + zoom, zoom_x : zoom_x + zoom] = patch
                cv2.putText(
                    canvas,
                    f"ZOOM {args.crop_size}px",
                    (zoom_x, zoom_y + zoom + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            header = (
                f"{video_path.stem}  frame={frame_number:06d}  th={args.threshold:.2f} "
                f"r={args.merge_radius:.0f}px  |  RC2(trial_0080) TRACKED DET {count:3d}  (no GT)"
            )
            cv2.putText(canvas, header, (PAD, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            legend = (
                f"RED = final system detection after spatio-temporal tracking | "
                f"RC2 trial_0080 | multi-scale {args.native_width}/{'/'.join(str(s) for s in args.scales)} | NO GT"
            )
            cv2.putText(
                canvas,
                legend,
                (PAD, args.canvas[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
            rendered += 1
            hit_frames += bool(count)
    finally:
        capture.release()
        writer.release()

    summary = {
        "video": video_path.name,
        "sequence": video_path.stem,
        "output": str(output_path.resolve()),
        "frames": frames,
        "rendered_frames": rendered,
        "frames_with_tracks": hit_frames,
        "frames_with_tracks_ratio": round(hit_frames / max(rendered, 1), 4),
        "detections": detections,
        "det_per_rendered_frame": round(detections / max(rendered, 1), 4),
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(
        f"[DONE] {video_path.name}: frames={frames} | tracked frames {hit_frames}/{rendered} "
        f"({hit_frames / max(rendered, 1) * 100:.1f}%) | detections={detections} "
        f"| {summary['seconds']:.1f}s -> {output_path.name}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(max(1, args.cpu_threads))
    jobs = collect_jobs(args)
    if not jobs:
        print("[SUCCESS] every selected video already has an output; nothing to render", flush=True)
        return

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, stride = load_hm_model(resolve_weights(args.hm_weights), device, use_p0_highway=False)
    scale_names = [str(args.native_width)] + [str(value) for value in args.scales]
    longquan_params = load_longquan_params()

    print(
        f"[INFO] videos={len(jobs)} | model={args.hm_weights} | p0_highway=off | stride={stride} | device={device}\n"
        f"[INFO] scales={scale_names} (0 = native letterbox) | th={args.threshold} | r={args.merge_radius:.0f}px | "
        f"ground th={args.th_ground} | salvage th={args.th_salvage} | confirm_consec="
        f"{longquan_params['confirm_consec']} | no ground truth on this domain",
        flush=True,
    )

    started = time.perf_counter()
    summaries = []
    for index, (video, output) in enumerate(jobs, 1):
        print(f"\n[VIDEO {index}/{len(jobs)}] {video}", flush=True)
        summaries.append(render_video(model, device, video, output, args))

    total_frames = sum(item["rendered_frames"] for item in summaries)
    total_hits = sum(item["frames_with_tracks"] for item in summaries)
    total_dets = sum(item["detections"] for item in summaries)
    print(
        f"\n[SUCCESS] videos={len(summaries)} | frames={total_frames} | tracked_frames={total_hits} "
        f"({total_hits / max(total_frames, 1) * 100:.1f}%) | detections={total_dets} "
        f"({total_dets / max(total_frames, 1):.3f}/frame) | elapsed {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "model": "trial_0080 (RC2)",
                        "hm_weights": args.hm_weights,
                        "use_p0_highway": False,
                        "scales": scale_names,
                        "threshold": args.threshold,
                        "th_ground": args.th_ground,
                        "th_salvage": args.th_salvage,
                        "merge_radius": args.merge_radius,
                        "region_threshold": args.region_threshold,
                        "native_size": [args.native_width, args.native_height],
                        "crop": [args.crop_left, args.crop_top, args.crop_width, args.crop_height],
                        "median_window": args.median_window,
                        "temporal_stride": args.temporal_stride,
                        "tracking": "龙泉山 spatio-temporal trajectory (bidirectional tracker + smoother + confirm_consec)",
                        "confirm_consec": longquan_params["confirm_consec"],
                        "sky_ratio": longquan_params["sky_ratio"],
                        "marker": "one red circle per final system detection, T<id>:<score>",
                        "ground_truth": "none on this domain; the tracker consumes candidate points only",
                    },
                    "videos": summaries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[SUCCESS] JSON -> {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
