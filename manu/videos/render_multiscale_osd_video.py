#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render multi-scale SOTA OSD review videos straight from raw MP4, with no ground truth.

Same visual language as ``render_pred_gt_osd_video.py`` (single 1920x1080 panel, full frame on the left,
local zoom on the right, header bar, legend) but driven by the video itself instead of a detection cache,
so it also covers a domain that has no labels yet:

    - the recorded letterbox is cropped back to native 640x512 before anything else, matching the
      龙泉山 contract (``--crop-*``/``--native-*``), so GMC, the 21-frame median and every scale see the
      same pixels Trial 0474 was trained on;
    - each frame runs the frozen Trial 0474 heatmap detector at the native letterbox plus every
      ``--scales`` downsample, the heatmaps are mapped back onto the native grid, and the rows above
      ``--threshold`` are cross-scale deduplicated by ``--merge-radius`` -- the exact pipeline of
      ``multiscale_heatmap_cache.py`` and the operating point of the business-domain evaluation;
    - boxes are drawn in one colour because there is nothing to score them against. ``--show-scales``
      overlays the undilated per-scale rows on top, which is what separates "this scale alone missed it"
      from "every scale missed it".

``--start`` gates rendering only: frames before it are still fed through GMC and the median window, so a
rendered frame never sits on a cold history. That is the same trap that made two sequences look identical
for their first 22 frames in the tree-GMC A/B.

No frame is written to disk and no cache is produced; a 3150-frame video costs one streaming pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from manu.videos.render_pred_gt_osd_video import (  # noqa: E402
    PAD,
    ROW_HEADER,
    SCALE_COLORS,
    draw_box,
    merge_scales_with_sizes,
    zoom_panel,
)

COLOR_DET = (0, 220, 0)
COLOR_IDLE = (120, 120, 120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render multi-scale Trial 0474 OSD review videos from raw MP4")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Single input video")
    source.add_argument("--input-root", help="Directory of input videos to render as a batch")
    parser.add_argument("--pattern", default="*.mp4", help="Glob used with --input-root")
    parser.add_argument("--shard", default="", help="Render only shard INDEX/COUNT of the sorted video list")
    parser.add_argument("--output", default="", help="Output MP4; required with --input")
    parser.add_argument("--output-dir", default="", help="Output directory used with --input-root")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos whose output already exists")
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--scales", default="160,80", help="Downsample targets besides the native letterbox input")
    parser.add_argument("--threshold", type=float, default=0.30, help="Score threshold of the merged rows")
    parser.add_argument("--region-threshold", type=float, default=0.06, help="Threshold of the per-scale rows")
    parser.add_argument("--merge-radius", type=float, default=64.0, help="Cross-scale dedup radius in native pixels")
    parser.add_argument("--min-area", type=int, default=4, help="Minimum connected-component pixel count")
    parser.add_argument("--max-detections", type=int, default=100, help="Detections kept per frame per scale")
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--crop-left", type=int, default=320)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=1280)
    parser.add_argument("--crop-height", type=int, default=1024)
    parser.add_argument("--native-width", type=int, default=640)
    parser.add_argument("--native-height", type=int, default=512)
    parser.add_argument("--canvas", default="1920x1080", help="Output canvas, for example 1920x1080")
    parser.add_argument("--zoom", type=int, default=360, help="Edge of the zoom panel; 0 disables it")
    parser.add_argument("--crop-size", type=int, default=48, help="Zoom crop edge in native pixels")
    parser.add_argument("--show-scales", default="", help="Comma-separated scale indices to overlay, for example 0,1,2")
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS; 0 uses the input FPS")
    parser.add_argument(
        "--start", type=int, default=0, help="First rendered frame; earlier frames still warm the history"
    )
    parser.add_argument("--end", type=int, default=0, help="Inclusive last rendered frame; 0 means the last frame")
    parser.add_argument("--limit", type=int, default=0, help="Render at most this many frames; 0 means every frame")
    parser.add_argument("--batch-size", type=int, default=8, help="GPU batch size, auto-halved on CUDA OOM")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--out-json", default="", help="Optional per-frame detection JSON")
    args = parser.parse_args()

    args.scales = sorted({int(item) for item in args.scales.split(",") if item.strip()}, reverse=True)
    if not args.scales or min(args.scales) <= 0:
        parser.error("--scales must list positive downsample targets")
    if args.min_area < 1 or args.max_detections < 1:
        parser.error("--min-area and --max-detections must be positive")
    if args.merge_radius < 0:
        parser.error("--merge-radius must be non-negative")
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
    args.show_scales = [int(item) for item in args.show_scales.split(",") if item.strip()]
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
    """Resolve the video-to-output pairs of this run, refusing silent name collisions."""
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
            continue
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


def rows_above_threshold(heatmap: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Per-scale rows above --threshold as an (N, 7) array of cx, cy, w, h, area, peak, sum."""
    entries = detect_components(heatmap, args.region_threshold, 1, args.min_area, args.max_detections)
    kept = [entry for entry in entries if entry[5] >= args.threshold]
    return np.asarray(kept, dtype=np.float32).reshape(-1, 7)


def render_video(
    model,
    device: torch.device,
    video_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    scale_tags: list[int],
    scale_names: list[str],
) -> dict:
    """Stream one video through the multi-scale detector and write the OSD review video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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

    gmc = FastGMCEstimator(downscale=2)
    history: list[np.ndarray] = []
    history_window = args.median_window * args.temporal_stride + 2
    processed, rendered, hit_frames, detections = 0, 0, 0, 0
    detected: list[int] = []
    frame_number = -1
    started = time.perf_counter()
    progress = tqdm(total=declared or None, desc=video_path.stem, unit="frame", dynamic_ncols=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            gray = restore_native(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args)
            features, history = compute_features([gray], history, args, gmc)
            history = history[-history_window:]
            processed += 1
            progress.update(1)

            if frame_number < args.start:
                continue
            if args.end and frame_number > args.end:
                break
            if args.limit and rendered >= args.limit:
                break

            canvases, mappings = scale_inputs(
                features[0], args.scales, args.imgsz, args.native_height, args.native_width
            )
            heatmaps = run_heatmaps(model, np.stack(canvases), device, args.batch_size)
            per_tag = {tag: {} for tag in scale_tags}
            for tag in scale_tags:
                heatmap = heatmap_to_native(
                    heatmaps[tag], *mappings[tag], args.imgsz, args.native_width, args.native_height
                )
                rows = rows_above_threshold(heatmap, args)
                if len(rows):
                    per_tag[tag][frame_number] = (rows[:, 5].copy(), rows[:, :2].copy(), rows[:, 2:4].copy())
            merged = merge_scales_with_sizes(per_tag, scale_tags, args.merge_radius)
            final = merged.get(frame_number)

            panel = cv2.cvtColor(cv2.resize(gray, panel_size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
            count = 0
            if final is not None:
                scores, points, sizes = final
                count = len(scores)
                for position in range(count):
                    draw_box(
                        panel,
                        points[position][0] * factor,
                        points[position][1] * factor,
                        (sizes[position][0] * factor) or 2.0,
                        (sizes[position][1] * factor) or 2.0,
                        COLOR_DET,
                        f"{scores[position]:.2f}",
                    )
                for tag in args.show_scales:
                    found = per_tag[tag].get(frame_number)
                    if found is None:
                        continue
                    tag_scores, tag_points, tag_sizes = found
                    for position in range(len(tag_scores)):
                        draw_box(
                            panel,
                            tag_points[position][0] * factor,
                            tag_points[position][1] * factor,
                            (tag_sizes[position][0] * factor) or 2.0,
                            (tag_sizes[position][1] * factor) or 2.0,
                            SCALE_COLORS[tag % len(SCALE_COLORS)],
                            f"s{scale_names[tag]} {tag_scores[position]:.2f}",
                            thickness=1,
                        )

            canvas = np.zeros((args.canvas[1], args.canvas[0], 3), np.uint8)
            panel_x, panel_y = PAD, ROW_HEADER + PAD
            canvas[panel_y : panel_y + panel.shape[0], panel_x : panel_x + panel.shape[1]] = panel
            if zoom:
                if count:
                    centre, color = (float(final[1][0][0]), float(final[1][0][1])), COLOR_DET
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
                f"r={args.merge_radius:.0f}px  |  MULTI-SCALE DET {count:3d}  (no GT)"
            )
            cv2.putText(canvas, header, (PAD, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            legend = f"GREEN = merged detection ({'/'.join(scale_names)}) | NO GT ON THIS DOMAIN"
            for tag in args.show_scales:
                legend += f" | s{scale_names[tag]}={SCALE_COLORS[tag % len(SCALE_COLORS)]}"
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
            detections += count
            if count:
                detected.append(frame_number)
            elapsed = max(time.perf_counter() - started, 1e-6)
            progress.set_postfix(det=count, proc_fps=f"{processed / elapsed:.1f}", out_fps=f"{rendered / elapsed:.1f}")
    finally:
        progress.close()
        capture.release()
        writer.release()

    if not rendered:
        raise RuntimeError(f"{video_path.name}: no frame rendered (start={args.start} is past the last frame)")
    summary = {
        "video": video_path.name,
        "output": str(output_path.resolve()),
        "declared_frames": declared,
        "processed_frames": processed,
        "rendered_frames": rendered,
        "frames_with_detections": hit_frames,
        "detections": detections,
        "det_per_rendered_frame": round(detections / rendered, 4),
        "first_detected_frame": detected[0] if detected else -1,
        "last_detected_frame": detected[-1] if detected else -1,
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(
        f"[DONE] {video_path.name}: rendered {rendered}/{declared} | frames_with_det={hit_frames} "
        f"({hit_frames / rendered * 100:.1f}%) | {summary['seconds']:.1f}s -> {output_path.name}",
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
    model, stride = load_hm_model(resolve_weights(args.hm_weights), device)
    scale_tags = list(range(len(args.scales) + 1))
    scale_names = [str(args.native_width)] + [str(value) for value in args.scales]
    unknown = [tag for tag in args.show_scales if tag not in scale_tags]
    if unknown:
        raise ValueError(f"--show-scales references absent scales {unknown}; available tags are {scale_tags}")

    print(
        f"[INFO] videos={len(jobs)} | scales={scale_names} (0 = native letterbox) | stride={stride} | device={device}\n"
        f"[INFO] merged rows = per-scale components >= {args.threshold} (per-scale gate "
        f"{args.region_threshold}) deduplicated within {args.merge_radius}px | native "
        f"{args.native_width}x{args.native_height} | crop ({args.crop_left}, {args.crop_top}, "
        f"{args.crop_width}, {args.crop_height}) | no ground truth on this domain",
        flush=True,
    )

    started = time.perf_counter()
    summaries = []
    for index, (video, output) in enumerate(jobs, 1):
        print(f"\n[VIDEO {index}/{len(jobs)}] {video}", flush=True)
        summaries.append(render_video(model, device, video, output, args, scale_tags, scale_names))

    total_frames = sum(item["rendered_frames"] for item in summaries)
    total_hits = sum(item["frames_with_detections"] for item in summaries)
    print(
        f"\n[SUCCESS] videos={len(summaries)} | rendered_frames={total_frames} | frames_with_det={total_hits} "
        f"({total_hits / total_frames * 100:.1f}%) | elapsed {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "scales": scale_names,
                        "threshold": args.threshold,
                        "region_threshold": args.region_threshold,
                        "merge_radius": args.merge_radius,
                        "min_area": args.min_area,
                        "native_size": [args.native_width, args.native_height],
                        "crop": [args.crop_left, args.crop_top, args.crop_width, args.crop_height],
                        "median_window": args.median_window,
                        "temporal_stride": args.temporal_stride,
                        "hm_weights": args.hm_weights,
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
