#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
🔴🔴🔴 【已作废 · 禁止运行】DEPRECATED — DO NOT RUN. 逻辑不成立，整版数据作废。

**作废原因（2026-11-20 用户纠偏）**：本脚本拿**反无人机检测器**去跑**没有无人机**的测试视频。
- 好片测出的「连续航迹」其实是**背景虚警**（检测器把楼角/路灯/树梢当成疑似弱小目标，
  地物纹理清晰 ⇒ 运动补偿后被时空关联当成连续轨迹）⇒ 测到的不是「目标可跟踪性」，是**虚警率**；
- 坏片「零航迹」只是因为**糊图把背景锐利边缘抹平、连虚警都触发不了**（0.167%）
  ⇒ 「零航迹」**不能**证明「画质差 ⇒ 跟不住」；
- 真实业务交互是**操作员在画面任意位置点一下 → SOT 锁住局部特征块**，
  「检测器报不报候选」与「用户点选后能不能跟住」是两码事。

⇒ **本脚本及其全部产物（好片有航迹 / 坏片零航迹）不得写进任何正式报告。**

**正解已落地（用这两套，不要用本脚本）**：
- 第八节「48 宫格模拟点击 + 双向闭环」：``manu/video_quality/vq_grid*.py``
- 第九节「点选式 SOT 实测 + 画质合格线」：``manu/video_quality/vq_sot_link.py`` /
  ``vq_sot_link_driver.py`` / ``vq_sot_compare.py`` / ``vq_sot_verify.py``
  （汇报件 ``manu/video_quality/report/REPORT_视频质量指标.md`` 第九节；
  主档 ``manu/memory/memory.md`` §284；专题档 §八）

本文件仅作历史留档，保留以便追溯当时的口径与代码。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_full_median_dataset import FastGMCEstimator  # noqa: E402
from manu.data.multiscale_heatmap_cache import (  # noqa: E402
    compute_features,
    detect_components,
    heatmap_to_native,
    run_heatmaps,
    scale_inputs,
)
from manu.data.preannotate_hm_bbox import load_hm_model, resolve_weights  # noqa: E402
from manu.videos.render_rc2_tracked_osd_video import (  # noqa: E402
    LONGQUAN_PARAMS_JSON,
    build_records,
    load_longquan_params,
    merge_across_scales,
    restore_native,
    run_tracks,
)

# The delivered working point.  The trajectory parameters (confirm_consec, gap handling, ...)
# come from the v3 delivery parameter file; the detection thresholds are overridable so the
# quality linkage can be checked for threshold sensitivity instead of trusting one point.
DELIVERY_PARAMS = LONGQUAN_PARAMS_JSON


def probe_geometry(video_path: Path) -> dict:
    """Read the frame geometry with ffprobe; fall back to OpenCV when ffprobe is unavailable."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", str(video_path),
    ]
    try:
        raw = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True).stdout
        stream = json.loads(raw)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        numerator, _, denominator = str(stream.get("r_frame_rate", "0/1")).partition("/")
        fps = float(numerator) / float(denominator or 1) if float(numerator or 0) else 0.0
    except Exception:  # noqa: BLE001 - ffprobe missing or odd stream: fall back to OpenCV
        capture = cv2.VideoCapture(str(video_path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
    return {"width": width, "height": height, "fps": fps}


def build_crop(geometry: dict, mode: str, native_width: int) -> dict:
    """Resolve the crop box and the native (model-facing) size for one clip.

    ``center1280`` is the convention this repo already uses for 1920-wide sources
    (``render_rc2_tracked_osd_video.py`` defaults): take the central 1280x1024 and hand the
    model a 640x512 frame, which is the geometry trial_0080 was trained on.  ``full640`` keeps
    the whole picture and only rescales -- it throws away no content, but the target gets
    smaller, so it is reported as a separate arm rather than silently mixed in.  ``native`` is
    the identity crop for clips that already are 640x512.
    """
    width, height = geometry["width"], geometry["height"]
    if mode == "native" or (width, height) == (native_width, 512):
        crop_width, crop_height = width, height
        crop_left, crop_top = 0, 0
    elif mode == "center1280":
        crop_width = min(1280, width)
        crop_height = min(1024, height)
        crop_left = (width - crop_width) // 2
        crop_top = (height - crop_height) // 2
    elif mode == "full640":
        crop_width, crop_height = width, height
        crop_left, crop_top = 0, 0
    else:
        raise ValueError(f"unknown crop mode: {mode}")
    native_height = max(1, round(crop_height * native_width / crop_width))
    return {
        "crop_left": int(crop_left),
        "crop_top": int(crop_top),
        "crop_width": int(crop_width),
        "crop_height": int(crop_height),
        "native_width": int(native_width),
        "native_height": int(native_height),
    }


def make_args_namespace(params: dict, crop: dict, cli: argparse.Namespace) -> argparse.Namespace:
    """Assemble the argparse namespace the shared pipeline helpers expect."""
    namespace = argparse.Namespace()
    namespace.crop_left = crop["crop_left"]
    namespace.crop_top = crop["crop_top"]
    namespace.crop_width = crop["crop_width"]
    namespace.crop_height = crop["crop_height"]
    namespace.native_width = crop["native_width"]
    namespace.native_height = crop["native_height"]
    namespace.imgsz = cli.imgsz
    namespace.scales = [int(value) for value in str(cli.scales).split(",") if value]
    namespace.batch_size = cli.batch_size
    namespace.median_window = params["median_window"]
    namespace.temporal_stride = params["temporal_stride"]
    namespace.region_threshold = params["region_threshold"]
    namespace.min_area = params["min_area"]
    namespace.max_detections = params["max_detections"]
    namespace.merge_radius = params["merge_radius"]
    namespace.threshold = params["threshold"]
    namespace.th_ground = params["th_ground"]
    namespace.th_salvage = params["th_salvage"]
    return namespace


def default_params(cli: argparse.Namespace) -> dict:
    """Detection parameters: the delivered v3 point unless the caller overrides one field."""
    data = json.loads(DELIVERY_PARAMS.read_text(encoding="utf-8"))
    params = dict(data.get("params", data))
    params.update(
        {
            "median_window": cli.median_window,
            "temporal_stride": cli.temporal_stride,
            "region_threshold": cli.region_threshold,
            "min_area": cli.min_area,
            "max_detections": cli.max_detections,
            "merge_radius": cli.merge_radius,
            "threshold": cli.threshold if cli.threshold is not None else params["th_base"],
            "th_ground": cli.th_ground if cli.th_ground is not None else params["th_ground"],
            "th_salvage": cli.th_salvage if cli.th_salvage is not None else params["th_salvage"],
        }
    )
    return params


def summarise_tracks(per_frame_tracks: np.ndarray, tracks: list[list[dict]], frames: int, window: int, fps: float) -> dict:
    """GT-free tracking health for one threshold arm."""
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    observations: dict[int, int] = {}
    for index, dets in enumerate(tracks):
        for det in dets:
            track_id = int(det["track_id"])
            if track_id not in first_seen:
                first_seen[track_id] = index
            last_seen[track_id] = index
            observations[track_id] = observations.get(track_id, 0) + 1
    lengths = np.array(
        [last_seen[track_id] - first_seen[track_id] + 1 for track_id in sorted(first_seen)], dtype=np.int32
    )
    observation_counts = np.array(
        [observations[track_id] for track_id in sorted(first_seen)], dtype=np.int32
    )
    frames_with_tracks = int((per_frame_tracks > 0).sum())
    covered = int(per_frame_tracks[per_frame_tracks > 0].sum())
    long_threshold = max(1, int(round(fps)))
    windows = []
    n_windows = int(np.ceil(frames / window)) if frames else 0
    for index in range(n_windows):
        start = index * window
        stop = min(frames, start + window)
        if stop <= start:
            continue
        windows.append(
            [
                int(index),
                int(stop - start),
                int((per_frame_tracks[start:stop] > 0).sum()),
                int(per_frame_tracks[start:stop].sum()),
            ]
        )
    return {
        "per_frame_tracks": per_frame_tracks.tolist(),
        "frames_with_tracks": frames_with_tracks,
        "frames_with_tracks_ratio": round(frames_with_tracks / max(frames, 1), 6),
        "track_cover_ratio": round(covered / max(frames, 1), 6),
        "tracks_total": int(len(first_seen)),
        "track_observations_total": int(observation_counts.sum()),
        "median_track_len": int(np.median(lengths)) if len(lengths) else 0,
        "mean_track_len": round(float(lengths.mean()), 4) if len(lengths) else 0.0,
        "max_track_len": int(lengths.max()) if len(lengths) else 0,
        "long_track_ratio": round(float((lengths >= long_threshold).mean()), 6) if len(lengths) else 0.0,
        "long_track_threshold_frames": long_threshold,
        "windows": windows,
        "track_lengths": lengths.tolist(),
        "track_observations": observation_counts.tolist(),
    }


def collect_candidates_capped(
    model,
    device: torch.device,
    video_path: Path,
    namespace: argparse.Namespace,
    start_frame: int,
    max_frames: int,
) -> tuple[dict[int, tuple], int]:
    """Same pipeline as ``render_rc2_tracked_osd_video.collect_candidates`` plus a frame cap.

    The cap exists for pilots and for the segmented runs on the long 纵横挂飞 clips: the
    per-scale rows, the cross-scale merge and the time-axis padding are byte-for-byte the same
    code path, so a capped run and a full run agree on every frame they share.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    gmc = FastGMCEstimator(downscale=2)
    history: list[np.ndarray] = []
    history_window = namespace.median_window * namespace.temporal_stride + 2
    scale_tags = list(range(len(namespace.scales) + 1))
    per_tag: dict[int, dict[int, tuple]] = {tag: {} for tag in scale_tags}
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    frame_number = start_frame - 1
    total_frames = start_frame
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            total_frames = max(total_frames, frame_number + 1)
            gray = restore_native(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), namespace)
            features, history = compute_features([gray], history, namespace, gmc)
            history = history[-history_window:]
            canvases, mappings = scale_inputs(
                features[0], namespace.scales, namespace.imgsz, namespace.native_height, namespace.native_width
            )
            heatmaps = run_heatmaps(model, np.stack(canvases), device, namespace.batch_size)
            for tag in scale_tags:
                heatmap = heatmap_to_native(
                    heatmaps[tag], *mappings[tag], namespace.imgsz, namespace.native_width, namespace.native_height
                )
                entries = detect_components(
                    heatmap, namespace.region_threshold, 1, namespace.min_area, namespace.max_detections
                )
                if entries:
                    scores = np.asarray([entry[5] for entry in entries], dtype=np.float32)
                    points = np.asarray([(entry[0], entry[1]) for entry in entries], dtype=np.float32)
                    per_tag[tag][frame_number] = (scores, points)
            if max_frames and (total_frames - start_frame) >= max_frames:
                break
    finally:
        capture.release()

    candidates = merge_across_scales(
        per_tag, scale_tags, namespace.merge_radius, namespace.threshold
    )
    for frame_index in range(total_frames):
        candidates.setdefault(frame_index, (np.zeros(0, np.float32), np.zeros((0, 2), np.float32)))
    return candidates, total_frames


def analyse_clip(
    model, device: torch.device, video_path: Path, cli: argparse.Namespace, params: dict
) -> dict:
    """Run detector + tracker on one clip and return the GT-free tracking health record."""
    started = time.perf_counter()
    geometry = probe_geometry(video_path)
    crop = build_crop(geometry, cli.crop_mode, cli.native_width)
    namespace = make_args_namespace(params, crop, cli)

    candidates, frames = collect_candidates_capped(
        model, device, video_path, namespace, cli.start_frame, cli.max_frames
    )
    records = build_records(candidates, video_path.stem, frames)

    # The expensive half (GMC features + detector) runs once; the tracker thresholds are cheap to
    # sweep, so every arm reuses the SAME candidate pool.  The arms differ only in the detection
    # thresholds handed to the tracker, which is exactly the axis that could make the good/bad gap
    # look like a threshold artefact instead of a quality effect.
    per_frame_candidates = np.array(
        [len(candidates.get(index, (np.zeros(0), np.zeros((0, 2))))[0]) for index in range(frames)],
        dtype=np.int32,
    )

    fps = geometry["fps"] if geometry["fps"] > 0 else 25.0
    # The quality report's 1-second windows use the METRIC stride (1), not the GMC stride (2), so
    # the window index lines up with report/windows.csv row-for-row.
    window = max(1, int(round(fps * cli.align_stride)))

    arms: dict[str, dict] = {}
    for arm in cli.threshold_arms:
        arm_namespace = argparse.Namespace(**vars(namespace))
        arm_namespace.threshold = float(arm)
        arm_namespace.th_ground = float(
            cli.th_ground if cli.th_ground is not None else params["th_ground"]
        )
        arm_namespace.th_salvage = float(
            cli.th_salvage if cli.th_salvage is not None else params["th_salvage"]
        )
        tracks = run_tracks(records, arm_namespace)
        per_frame_tracks = np.array([len(dets) for dets in tracks], dtype=np.int32)
        arms[str(arm)] = summarise_tracks(per_frame_tracks, tracks, frames, window, fps)

    primary = arms[str(cli.threshold_arms[0])]
    frames_with_candidates = int((per_frame_candidates > 0).sum())
    summary = {
        "video": video_path.name,
        "sequence": video_path.stem,
        "group": cli.group,
        "kind": cli.kind,
        "modality": cli.modality,
        "geometry": geometry,
        "crop": crop,
        "frames": int(frames),
        "start_frame": int(cli.start_frame),
        "fps": round(fps, 4),
        "window_frames": window,
        "frames_with_candidates": frames_with_candidates,
        "frames_with_candidates_ratio": round(frames_with_candidates / max(frames, 1), 6),
        "candidates_total": int(per_frame_candidates.sum()),
        "det_per_frame": round(float(per_frame_candidates.sum()) / max(frames, 1), 6),
        "primary_arm": str(cli.threshold_arms[0]),
        "arms": {
            arm: {
                key: value
                for key, value in payload.items()
                if key
                not in ("windows", "track_lengths", "track_observations", "per_frame_tracks")
            }
            for arm, payload in arms.items()
        },
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    return {
        "summary": summary,
        "per_frame_candidates": per_frame_candidates.tolist(),
        "per_frame_tracks": primary["per_frame_tracks"],
        "arms": arms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link image quality to real tracker outcome on raw clips")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Single input video")
    source.add_argument("--input-root", help="Directory of input videos")
    parser.add_argument("--pattern", default="*", help="Glob applied to --input-root")
    parser.add_argument("--out-dir", required=True, help="Directory for the per-clip JSON records")
    parser.add_argument("--hm-weights", default="/mnt/data/siping/backup/mix_ratio_search_20261022/weights/trial_0080_epoch3.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--native-width", type=int, default=640)
    parser.add_argument("--scales", default="160,80")
    parser.add_argument("--crop-mode", default="center1280", choices=["native", "center1280", "full640"])
    parser.add_argument("--threshold", type=float, default=None, help="Override the delivered th_base")
    parser.add_argument("--th-ground", type=float, default=None)
    parser.add_argument("--th-salvage", type=float, default=None)
    parser.add_argument("--region-threshold", type=float, default=0.06)
    parser.add_argument("--merge-radius", type=float, default=768.0)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--group", default="", help="good / bad (recorded in the output)")
    parser.add_argument("--kind", default="source", help="source / osd")
    parser.add_argument("--modality", default="", help="ir / vis")
    parser.add_argument("--max-frames", type=int, default=0, help="Pilot: stop after N frames (0 = all)")
    parser.add_argument("--start-frame", type=int, default=0, help="Segment start (global frame index)")
    parser.add_argument("--align-stride", type=int, default=1, help="Quality window stride (1 = 1 second)")
    parser.add_argument(
        "--threshold-arms",
        default="0.20",
        help="Comma list of tracker detection thresholds; the candidate pool is computed once",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def collect_jobs(cli: argparse.Namespace) -> list[Path]:
    if cli.input:
        return [Path(cli.input)]
    root = Path(cli.input_root)
    return sorted(path for path in root.glob(cli.pattern) if path.is_file())


def main() -> None:
    cli = parse_args()
    cli.threshold_arms = [float(value) for value in str(cli.threshold_arms).split(",") if value]
    cv2.setNumThreads(max(1, cli.cpu_threads))
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = collect_jobs(cli)
    if not jobs:
        print("[ERROR] no input videos matched", flush=True)
        sys.exit(1)

    params = default_params(cli)
    device = torch.device(f"cuda:{cli.device}" if torch.cuda.is_available() and cli.device != "cpu" else "cpu")
    model, stride = load_hm_model(resolve_weights(cli.hm_weights), device, use_p0_highway=False)
    print(
        f"[INFO] videos={len(jobs)} | weights={cli.hm_weights} | stride={stride} | device={device} | "
        f"crop={cli.crop_mode} | th_base={params['threshold']} | th_ground={params['th_ground']} | "
        f"th_salvage={params['th_salvage']} | no ground truth on this domain",
        flush=True,
    )

    summaries = []
    for index, video in enumerate(jobs, 1):
        suffix = f"_s{cli.start_frame:07d}" if cli.start_frame else ""
        target = out_dir / f"{video.stem}{suffix}.json"
        if cli.skip_existing and target.exists():
            print(f"[SKIP] {video.name} (exists)", flush=True)
            continue
        print(f"\n[VIDEO {index}/{len(jobs)}] {video}", flush=True)
        record = analyse_clip(model, device, video, cli, params)
        target.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        summary = record["summary"]
        summaries.append(summary)
        print(
            f"[DONE] {video.name}: frames={summary['frames']} | "
            f"tracked {summary['frames_with_tracks']}/{summary['frames']} "
            f"({summary['frames_with_tracks_ratio'] * 100:.1f}%) | "
            f"tracks={summary['tracks_total']} | median_track_len={summary['median_track_len']} | "
            f"{summary['elapsed_s']:.1f}s",
            flush=True,
        )

    manifest = {
        "config": {
            "weights": cli.hm_weights,
            "crop_mode": cli.crop_mode,
            "imgsz": cli.imgsz,
            "scales": cli.scales,
            "native_width": cli.native_width,
            "threshold": params["threshold"],
            "th_ground": params["th_ground"],
            "th_salvage": params["th_salvage"],
            "merge_radius": params["merge_radius"],
            "median_window": params["median_window"],
            "temporal_stride": params["temporal_stride"],
            "params_file": str(DELIVERY_PARAMS),
            "detector": "trial_0080 (RC2) delivery heatmap, multi-scale",
            "trajectory": "龙泉山 bidirectional tracker + smoother + consecutive confirmation",
            "ground_truth": "none on this domain; every tracking number is GT-free",
        },
        "videos": summaries,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[SUCCESS] manifest -> {(out_dir / 'run_manifest.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()
