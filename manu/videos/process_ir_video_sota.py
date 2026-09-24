#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render a paper-style diagnostic video from an arbitrary infrared video."""

from __future__ import annotations

import argparse
from collections import deque
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

from manu.data.build_full_median_dataset import FastGMCEstimator
from manu.videos.generate_sota_paper_video import PointTracker, create_paper_insets, letterbox_bgr
from manu.evaluation.heatmap_evaluate import extract_peaks
from manu.models.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Render paper-style SOTA diagnostic video from an infrared video")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2, help="Frame spacing for GMC temporal history")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS; 0 uses input FPS")
    parser.add_argument("--max-age", type=int, default=4)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--match-dist", type=float, default=12.0)
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument("--restore-native", action="store_true", help="Crop 2x letterbox and restore native 640x512 before temporal processing")
    parser.add_argument("--crop-left", type=int, default=320)
    parser.add_argument("--crop-top", type=int, default=28)
    parser.add_argument("--crop-width", type=int, default=1280)
    parser.add_argument("--crop-height", type=int, default=1024)
    parser.add_argument("--native-width", type=int, default=640)
    parser.add_argument("--native-height", type=int, default=512)
    return parser.parse_args()


def restore_native_frame(frame: np.ndarray, args) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = args.crop_left
    y1 = args.crop_top
    x2 = x1 + args.crop_width
    y2 = y1 + args.crop_height
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"Native crop {(x1, y1, x2, y2)} exceeds frame size {(width, height)}")
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (args.native_width, args.native_height), interpolation=cv2.INTER_AREA)


def load_model(weights_path: Path, device: torch.device, stride_fallback: int):
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    stride = int(ckpt.get("stride", stride_fallback))
    p0_kwargs = ckpt.get(
        "p0_kwargs",
        {
            "use_spatial_gate": True,
            "stem_type": "standard_dw",
            "downsample_mode": "pixel_unshuffle",
            "gate_input_mode": "diff_only",
            "gate_mid_channels": 16,
            "gate_depth": 2,
            "fusion_mode": "scalar_gate",
        },
    )
    model = YOLO26HeatmapDetector(
        stride=stride,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=True,
        p0_highway_kwargs=p0_kwargs,
    )
    own_state = model.state_dict()
    matched = 0
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_key in own_state and own_state[clean_key].shape == value.shape:
            own_state[clean_key].copy_(value)
            matched += 1
    model.to(device).eval()
    print(f"[INFO] Loaded {matched} tensors from {weights_path}")
    print(f"[INFO] Architecture: {p0_kwargs}")
    return model, stride


def make_input(
    frames: deque[np.ndarray],
    imgsz: int,
    median_window: int,
    temporal_stride: int,
    gmc: FastGMCEstimator,
) -> tuple[np.ndarray, np.ndarray]:
    current = frames[-1]
    lag_frame = frames[max(0, len(frames) - 3)]
    aligned_lag = gmc.warp(lag_frame, gmc.compute_affine(lag_frame, current))
    motion = cv2.absdiff(current, aligned_lag)

    history = list(frames)[::-1][::temporal_stride][:median_window]
    aligned_history = []
    for history_frame in history:
        transform = gmc.compute_affine(history_frame, current)
        aligned_history.append(gmc.warp(history_frame, transform))
    background = np.median(np.stack(aligned_history, axis=0), axis=0).astype(np.uint8)
    residual = np.clip(current.astype(np.int16) - background.astype(np.int16), 0, 255).astype(np.uint8)

    feature = np.stack([residual, motion, current], axis=2)
    feature = letterbox_bgr(feature, imgsz)
    raw = cv2.cvtColor(letterbox_bgr(current, imgsz), cv2.COLOR_GRAY2BGR)
    return feature, raw


def render_panel(
    raw_panel: np.ndarray,
    heatmap: np.ndarray,
    tracks: list[dict],
    points: np.ndarray,
    scores: np.ndarray,
    frame_idx: int,
    total_frames: int,
    input_fps: float,
    processing_fps: float,
    imgsz: int,
) -> np.ndarray:
    left = raw_panel.copy()
    strongest = points[int(np.argmax(scores))] if len(points) else np.array([imgsz / 2, imgsz / 2])
    if tracks:
        strongest = np.asarray(tracks[0]["pos"], dtype=np.float32)

    for track in tracks:
        x, y = (int(round(v)) for v in track["pos"])
        color = (0, 165, 255) if track["is_coasting"] else (255, 255, 0)
        history = track.get("history", [])
        for index in range(1, len(history)):
            p1 = tuple(int(round(v)) for v in history[index - 1])
            p2 = tuple(int(round(v)) for v in history[index])
            cv2.line(left, p1, p2, color, 1)
        cv2.rectangle(left, (x - 6, y - 6), (x + 6, y + 6), color, 1)
        label = "COAST" if track["is_coasting"] else f"{track['score']:.2f}"
        cv2.putText(left, f"TRK-{track['id']} [{label}]", (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    for point, score in zip(points, scores):
        x, y = (int(round(v)) for v in point)
        cv2.drawMarker(left, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(left, f"{score:.2f}", (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    inset_raw, inset_enh = create_paper_insets(left, (float(strongest[0]), float(strongest[1])), crop_size=36, box_w=140, box_h=140)
    left[imgsz - 150 : imgsz - 10, 10:150] = inset_raw
    left[imgsz - 150 : imgsz - 10, imgsz - 150 : imgsz - 10] = inset_enh

    heatmap_display = np.clip(heatmap * 255.0 * 2.5, 0, 255).astype(np.uint8)
    right = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_MAGMA)
    for point in points:
        x, y = (int(round(v)) for v in point)
        cv2.drawMarker(right, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 7, 1)
    for track in tracks:
        x, y = (int(round(v)) for v in track["pos"])
        cv2.circle(right, (x, y), 5, (0, 255, 255), 1)

    elapsed_text = f"F:{frame_idx:06d}/{total_frames:06d} | Det:{len(points):02d} | Trk:{len(tracks):02d}"
    perf_text = f"Input:{input_fps:.1f} FPS | Proc:{processing_fps:.1f} FPS"
    for panel, title, subtitle in (
        (left, "INFRARED SOTA | TRIAL 0474", elapsed_text),
        (right, "HEATMAP SURFACE | [I, MOTION, MEDIAN]", perf_text),
    ):
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (imgsz, 36), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.78, panel, 0.22, 0, panel)
        cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
        text_width = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(panel, subtitle, (imgsz - text_width - 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    combined = np.hstack([left, right])
    cv2.line(combined, (imgsz, 0), (imgsz, imgsz), (80, 80, 80), 2)
    return combined


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    weights_path = Path(args.weights)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not weights_path.exists():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(args.weights)
    if args.median_window < 1 or args.median_window % 2 == 0:
        raise ValueError("--median-window must be a positive odd number")
    if args.temporal_stride < 1:
        raise ValueError("--temporal-stride must be positive")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, stride = load_model(weights_path, device, 2)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = args.fps if args.fps > 0 else source_fps
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (args.imgsz * 2, args.imgsz))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {output_path}")

    frames: deque[np.ndarray] = deque(maxlen=1 + (args.median_window - 1) * args.temporal_stride)
    gmc = FastGMCEstimator(downscale=2)
    tracker = None if args.no_tracking else PointTracker(args.max_age, args.min_hits, args.match_dist, True, args.conf)
    read_frames = 0
    processed = 0
    started = time.perf_counter()
    last_frame = None

    progress = tqdm(total=total_frames or None, desc="Processing video", unit="frame", dynamic_ncols=True)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if read_frames % args.frame_stride:
            read_frames += 1
            progress.update(1)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args.restore_native:
            gray = restore_native_frame(gray, args)
        frames.append(gray)
        feature, raw_panel = make_input(frames, args.imgsz, args.median_window, args.temporal_stride, gmc)
        tensor = torch.from_numpy(feature.transpose(2, 0, 1)).unsqueeze(0).to(device).float() / 255.0
        with torch.no_grad():
            prediction = model(tensor)
            peaks = extract_peaks(prediction["heatmap"], prediction["offset"], stride=stride, conf_thresh=args.conf, top_k=args.top_k)[0]
        points = peaks["points"]
        scores = peaks["scores"]
        tracks = tracker.update(points, scores) if tracker is not None else []
        heatmap = cv2.resize(prediction["heatmap"][0, 0].cpu().numpy(), (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        processed += 1
        read_frames += 1
        elapsed = time.perf_counter() - started
        last_frame = render_panel(raw_panel, heatmap, tracks, points, scores, read_frames, total_frames, source_fps, processed / max(elapsed, 1e-6), args.imgsz)
        writer.write(last_frame)
        progress.update(1)
        progress.set_postfix(process_fps=f"{processed / max(elapsed, 1e-6):.1f}", detections=len(points), tracks=len(tracks))

    progress.close()
    capture.release()
    writer.release()
    if last_frame is None:
        raise RuntimeError("Input video contains no readable frames")
    print(f"[SUCCESS] Saved: {output_path.resolve()}")
    print(f"[INFO] Frames: {processed} | Input FPS: {source_fps:.2f} | Output FPS: {output_fps:.2f}")


if __name__ == "__main__":
    main()
