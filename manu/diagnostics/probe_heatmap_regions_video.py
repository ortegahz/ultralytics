#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render Trial 0474 heatmap regions from saved three-channel feature images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.models.heatmap_model import YOLO26HeatmapDetector
from manu.training.train_trial0474_wh import WhHead


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def letterbox(image: np.ndarray, size: int):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, image.shape[2]), 114, dtype=np.uint8) if image.ndim == 3 else np.full((size, size), 114, dtype=np.uint8)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def downsample_pad(feature: np.ndarray, target: int, out_size: int):
    height, width = feature.shape[:2]
    scale = target / max(width, height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(feature, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    canvas[:, :, 0] = int(resized[:, :, 0].mean())
    pad_x = (out_size - new_width) // 2
    pad_y = (out_size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def resolve_checkpoint(path: str) -> Path:
    candidates = [Path(path), REPO_ROOT / path, Path("/home/manu/mnt/pycharm_project_10ae9e2e") / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def load_model(weights: Path, device: torch.device, default_stride: int):
    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint.get("state_dict", checkpoint)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    p0_kwargs = checkpoint.get(
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
        stride=checkpoint.get("stride", default_stride),
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
    print(f"[INFO] Loaded {weights} ({matched}/{len(own_state)} tensors)")
    return model, checkpoint.get("stride", default_stride), checkpoint.get("imgsz", 640)


def load_wh_head(path: Path, model: YOLO26HeatmapDetector, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("wh_head", checkpoint.get("state_dict", checkpoint))
    wh_head = WhHead(model.head.feat_conv[0].conv.in_channels)
    wh_head.load_state_dict(state_dict)
    wh_head.to(device).eval()
    print(f"[INFO] Loaded Wh head: {path}")
    return wh_head


def draw_wh_boxes(panel: np.ndarray, heatmap: torch.Tensor, offset: torch.Tensor, wh_map: torch.Tensor | None, threshold: float, stride: int, scale: float, pad_x: int, pad_y: int, top_k: int = 100):
    if wh_map is None:
        return
    pooled = F.max_pool2d(heatmap, 3, 1, 1)
    keep = (heatmap == pooled) & (heatmap >= threshold)
    points = torch.nonzero(keep[0, 0], as_tuple=False)
    if len(points) == 0:
        return
    scores = heatmap[0, 0, points[:, 0], points[:, 1]]
    if len(scores) > top_k:
        scores, selected = torch.topk(scores, top_k)
        points = points[selected]
    y, x = points[:, 0], points[:, 1]
    centers_x = (x.float() + offset[0, 0, y, x]) * stride
    centers_y = (y.float() + offset[0, 1, y, x]) * stride
    widths = torch.exp(wh_map[0, 0, y, x]) / max(scale, 1e-9)
    heights = torch.exp(wh_map[0, 1, y, x]) / max(scale, 1e-9)
    native_x = (centers_x - pad_x) / max(scale, 1e-9)
    native_y = (centers_y - pad_y) / max(scale, 1e-9)
    for cx, cy, width, height, score in zip(native_x.cpu().numpy(), native_y.cpu().numpy(), widths.cpu().numpy(), heights.cpu().numpy(), scores.cpu().numpy()):
        x1, y1 = round(cx - width / 2), round(cy - height / 2)
        x2, y2 = round(cx + width / 2), round(cy + height / 2)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(panel, f"{width:.0f}x{height:.0f} {score:.2f}", (x1, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)


def draw_panel(
    panel: np.ndarray,
    heatmap: np.ndarray,
    threshold: float,
    min_area: int,
    color: tuple[int, int, int],
    rows: list[dict],
    frame_index: int,
    panel_name: str,
):
    mask = (heatmap >= threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scale_x = panel.shape[1] / heatmap.shape[1]
    scale_y = panel.shape[0] / heatmap.shape[0]
    entries = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < min_area:
            continue
        values = heatmap[labels == label]
        entries.append(
            {
                "frame": frame_index,
                "panel": panel_name,
                "threshold": threshold,
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area": int(area),
                "max": float(values.max()),
                "sum": float(values.sum()),
                "mean": float(values.mean()),
                "cx": float(centroids[label][0]),
                "cy": float(centroids[label][1]),
            }
        )
    entries.sort(key=lambda item: item["sum"], reverse=True)
    for rank, entry in enumerate(entries[:8], start=1):
        x = round(entry["x"] * scale_x)
        y = round(entry["y"] * scale_y)
        width = max(1, round(entry["width"] * scale_x))
        height = max(1, round(entry["height"] * scale_y))
        cv2.rectangle(panel, (x, y), (x + width - 1, y + height - 1), color, 1)
        text = f"{rank} A{entry['area']} M{entry['max']:.2f} S{entry['sum']:.2f} u{entry['mean']:.2f}"
        cv2.putText(panel, text, (x, max(14, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        rows.append(entry)
    cv2.putText(panel, f"{panel_name} >= {threshold:.2f} regions={len(entries)}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="Render Trial 0474 heatmap regions from saved GMC+median feature images")
    parser.add_argument("--features-dir", required=True, help="Directory containing saved three-channel feature JPGs")
    parser.add_argument("--pattern", default="*.jpg", help="Feature filename glob, for example VIDEO00005_19700101_002959__frame_*.jpg")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--wh-weights", default="", help="Optional Wh checkpoint; omit to keep legacy heatmap-only mode")
    parser.add_argument("--wh-threshold", type=float, default=0.22)
    parser.add_argument("--wh-top-k", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--region-threshold", type=float, default=0.06)
    parser.add_argument("--main-threshold", type=float, default=0.22)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--direct-downsample", type=int, default=0, choices=[0, 160, 320], help="Downsample the full 3-channel feature and channel-aware pad to imgsz")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(features_dir.glob(args.pattern), key=natural_key)
    if args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise FileNotFoundError(f"No saved feature images found in {features_dir}")
    first_feature = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_feature is None or first_feature.ndim != 3 or first_feature.shape[2] != 3:
        raise ValueError("The input directory must contain three-channel saved feature images")
    height, width = first_feature.shape[:2]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, stride, imgsz = load_model(resolve_checkpoint(args.weights), device, args.stride)
    wh_head = load_wh_head(resolve_checkpoint(args.wh_weights), model, device) if args.wh_weights else None
    output_path = output_dir / "heatmap_regions_diagnostic.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width * 3, height)
    )
    csv_path = output_dir / "heatmap_regions.csv"
    csv_fields = ["frame", "panel", "threshold", "x", "y", "width", "height", "area", "max", "sum", "mean", "cx", "cy"]
    rows = []
    print(f"[INFO] Reading saved features directly: {features_dir}")
    if args.direct_downsample:
        print(f"[INFO] Direct downsample enabled: {args.direct_downsample}px feature with channel-aware padding to {imgsz}px")
    else:
        print(f"[INFO] Direct downsample disabled; feature shape {width}x{height}x3; GMC and median recomputation disabled")
    try:
        for index, path in enumerate(frame_paths):
            feature_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if feature_bgr is None or feature_bgr.shape[:2] != (height, width) or feature_bgr.shape[2] != 3:
                raise RuntimeError(f"Invalid feature image: {path}")
            if args.direct_downsample:
                model_feature, scale, pad_x, pad_y = downsample_pad(feature_bgr, args.direct_downsample, imgsz)
            else:
                model_feature, scale, pad_x, pad_y = letterbox(feature_bgr, imgsz)
            model_image = model_feature.transpose(2, 0, 1)[::-1].copy()
            tensor = torch.from_numpy(model_image).unsqueeze(0).to(device).float() / 255.0
            with torch.no_grad():
                prediction = model(tensor)
                wh_map = wh_head(model.extract_features(tensor)) if wh_head is not None else None
            heatmap = prediction["heatmap"][0, 0].detach().cpu().numpy()
            heatmap_full = cv2.resize(heatmap, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            new_width, new_height = round(width * scale), round(height * scale)
            heatmap_full = heatmap_full[pad_y : pad_y + new_height, pad_x : pad_x + new_width]
            heatmap_full = cv2.resize(heatmap_full, (width, height), interpolation=cv2.INTER_LINEAR)
            original = cv2.cvtColor(feature_bgr[:, :, 0], cv2.COLOR_GRAY2BGR)
            heat_color_abs = cv2.applyColorMap(
                np.clip(heatmap_full * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            h_min, h_max = float(heatmap_full.min()), float(heatmap_full.max())
            if h_max - h_min > 1e-6:
                heatmap_norm = (heatmap_full - h_min) / (h_max - h_min)
            else:
                heatmap_norm = np.zeros_like(heatmap_full)
            heat_color_dyn = cv2.applyColorMap(
                np.clip(heatmap_norm * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            region_panel = original.copy()
            heat_panel = cv2.addWeighted(heat_color_dyn, 0.70, original, 0.30, 0)
            main_panel = cv2.addWeighted(heat_color_abs, 0.70, original, 0.30, 0)
            draw_panel(region_panel, heatmap_full, args.region_threshold, args.min_area, (0, 255, 255), rows, index, "region")
            draw_panel(main_panel, heatmap_full, args.main_threshold, args.min_area, (0, 0, 255), rows, index, "absolute")
            if wh_head is not None:
                draw_wh_boxes(region_panel, prediction["heatmap"], prediction["offset"], wh_map, args.wh_threshold, stride, scale, pad_x, pad_y, args.wh_top_k)
                draw_wh_boxes(main_panel, prediction["heatmap"], prediction["offset"], wh_map, args.wh_threshold, stride, scale, pad_x, pad_y, args.wh_top_k)
                cv2.putText(region_panel, "GREEN: Wh estimated boxes", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(main_panel, "ABSOLUTE [0,1]", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                heat_panel,
                f"DYN min={h_min:.4f} max={h_max:.4f} abs_sum={heatmap_full.sum():.1f}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
            )
            canvas = np.hstack([region_panel, heat_panel, main_panel])
            cv2.putText(canvas, f"frame={index} {path.name}", (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
            if index % 50 == 0:
                print(f"[INFO] {index + 1}/{len(frame_paths)} {path.name} heatmap_max={heatmap_full.max():.4f} heatmap_sum={heatmap_full.sum():.2f}")
    finally:
        writer.release()
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=csv_fields)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(f"[SUCCESS] Video: {output_path}")
    print(f"[SUCCESS] CSV: {csv_path}")


if __name__ == "__main__":
    main()
