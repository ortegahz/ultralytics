#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run batched Trial 0474 inference on preprocessed video features and render OSD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.generate_sota_paper_video import PointTracker, create_paper_insets, letterbox_bgr
from manu.heatmap_evaluate import extract_peaks
from manu.heatmap_model import YOLO26HeatmapDetector


def load_model(weights: Path, device: torch.device, stride: int):
    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint.get("state_dict", checkpoint)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    p0_kwargs = checkpoint.get("p0_kwargs", {"use_spatial_gate": True, "stem_type": "standard_dw", "downsample_mode": "pixel_unshuffle", "gate_input_mode": "diff_only", "gate_mid_channels": 16, "gate_depth": 2, "fusion_mode": "scalar_gate"})
    model = YOLO26HeatmapDetector(stride=checkpoint.get("stride", stride), num_classes=1, temporal_mode="standard", use_p0_highway=True, p0_highway_kwargs=p0_kwargs)
    own_state = model.state_dict()
    matched = 0
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_key in own_state and own_state[clean_key].shape == value.shape:
            own_state[clean_key].copy_(value)
            matched += 1
    model.to(device).eval()
    print(f"[INFO] Loaded {matched} tensors from {weights}")
    return model, int(checkpoint.get("stride", stride))


def render(left: np.ndarray, heatmap: np.ndarray, points: np.ndarray, scores: np.ndarray, tracks: list[dict], index: int, total: int, imgsz: int) -> np.ndarray:
    left = left.copy()
    focus = points[int(np.argmax(scores))] if len(points) else np.array([imgsz / 2, imgsz / 2])
    for track in tracks:
        focus = np.asarray(track["pos"])
        x, y = (int(round(v)) for v in track["pos"])
        color = (0, 165, 255) if track["is_coasting"] else (255, 255, 0)
        history = track.get("history", [])
        for i in range(1, len(history)):
            cv2.line(left, tuple(np.int32(np.round(history[i - 1]))), tuple(np.int32(np.round(history[i]))), color, 1)
        cv2.rectangle(left, (x - 6, y - 6), (x + 6, y + 6), color, 1)
        cv2.putText(left, f"TRK-{track['id']}", (x + 7, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    for point, score in zip(points, scores):
        x, y = (int(round(v)) for v in point)
        cv2.drawMarker(left, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.putText(left, f"{score:.2f}", (x + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    raw_inset, enhanced_inset = create_paper_insets(left, (float(focus[0]), float(focus[1])), crop_size=36, box_w=140, box_h=140)
    left[imgsz - 150 : imgsz - 10, 10:150] = raw_inset
    left[imgsz - 150 : imgsz - 10, imgsz - 150 : imgsz - 10] = enhanced_inset
    right = cv2.applyColorMap(np.clip(heatmap * 255.0 * 2.5, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    for point in points:
        cv2.drawMarker(right, tuple(np.int32(np.round(point))), (0, 255, 255), cv2.MARKER_CROSS, 7, 1)
    for track in tracks:
        cv2.circle(right, tuple(np.int32(np.round(track["pos"]))), 5, (0, 255, 255), 1)
    for panel, title, subtitle in ((left, "INFRARED SOTA | TRIAL 0474", f"F:{index:06d}/{total:06d} | Det:{len(points)} | Trk:{len(tracks)}"), (right, "HEATMAP SURFACE | GMC + MEDIAN", "Batched inference")):
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (imgsz, 36), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.78, panel, 0.22, 0, panel)
        cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
        width = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.putText(panel, subtitle, (imgsz - width - 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    combined = np.hstack([left, right])
    cv2.line(combined, (imgsz, 0), (imgsz, imgsz), (80, 80, 80), 2)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Batch infer preprocessed video features and render paper OSD")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-age", type=int, default=4)
    parser.add_argument("--match-dist", type=float, default=12.0)
    args = parser.parse_args()
    features_dir, frames_dir = Path(args.features_dir), Path(args.frames_dir)
    feature_paths = sorted(features_dir.glob("feature_*.npy"))
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if len(feature_paths) != len(frame_paths) or not feature_paths:
        raise ValueError("Feature and frame counts must match and be non-zero")
    metadata = json.loads((frames_dir / "metadata.json").read_text(encoding="utf-8"))
    source_fps = float(metadata.get("fps", 25.0))
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, stride = load_model(Path(args.weights), device, 2)
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    writer_path = Path(args.output)
    writer_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (args.imgsz * 2, args.imgsz))
    tracker = PointTracker(args.max_age, args.min_hits, args.match_dist, True, args.conf)
    all_peaks = []
    all_heatmaps = []
    with torch.no_grad():
        for start in tqdm(range(0, len(feature_paths), args.batch), desc="Batched inference", unit="batch", dynamic_ncols=True):
            batch_paths = feature_paths[start : start + args.batch]
            arrays = [letterbox_bgr(np.load(path, allow_pickle=False), args.imgsz).transpose(2, 0, 1) for path in batch_paths]
            batch = torch.from_numpy(np.stack(arrays)).to(device).float() / 255.0
            prediction = model(batch)
            all_peaks.extend(extract_peaks(prediction["heatmap"], prediction["offset"], stride=stride, conf_thresh=args.conf, top_k=args.top_k))
            all_heatmaps.extend(prediction["heatmap"][:, 0].cpu().numpy())
    for index, (frame_path, peaks) in enumerate(tqdm(list(zip(frame_paths, all_peaks)), desc="Rendering OSD", unit="frame", dynamic_ncols=True)):
        raw_gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        raw = cv2.cvtColor(letterbox_bgr(raw_gray, args.imgsz), cv2.COLOR_GRAY2BGR)
        tracks = tracker.update(peaks["points"], peaks["scores"])
        heatmap = cv2.resize(all_heatmaps[index], (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        writer.write(render(raw, heatmap, peaks["points"], peaks["scores"], tracks, index + 1, len(frame_paths), args.imgsz))
    writer.release()
    print(f"[SUCCESS] Saved: {writer_path.resolve()}")


if __name__ == "__main__":
    main()
