#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pre-annotate one sequence with HM center candidates and YOLO26 BBox sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from manu.build_full_median_dataset import FastGMCEstimator
from manu.heatmap_evaluate import extract_peaks
from manu.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-annotate a sequence: HM center + YOLO26 BBox size")
    parser.add_argument("--frames-dir", required=True, help="Directory of restored native frames (640x512)")
    parser.add_argument("--output-root", required=True, help="YOLO dataset root to create")
    parser.add_argument("--seq", default="", help="Sequence name, defaults to the frames directory name")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="YOLO split folder to write")
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--bbox-weights", default="runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--hm-conf", type=float, default=0.06, help="HM peak harvest threshold")
    parser.add_argument("--hm-top-k", type=int, default=50)
    parser.add_argument("--bbox-conf", type=float, default=0.06, help="YOLO26 BBox confidence threshold")
    parser.add_argument("--bbox-iou", type=float, default=0.70)
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--coarse-min", type=float, default=6.0, help="Minimum HM coarse box size in native px")
    parser.add_argument("--coarse-max", type=float, default=40.0, help="Maximum HM coarse box size in native px")
    parser.add_argument("--include-bbox-only", action="store_true", help="Also emit YOLO boxes with no HM point as class 2")
    parser.add_argument("--no-copy", action="store_true", help="Do not copy images into the dataset")
    parser.add_argument("--limit", type=int, default=0, help="Optional frame limit for a quick smoke test")
    return parser.parse_args()


def resolve_weights(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        path = PROJECT_ROOT / value
    if not path.exists():
        raise FileNotFoundError(value)
    return path


def load_hm_model(weights: Path, device: torch.device, stride_fallback: int = 2):
    ckpt = torch.load(weights, map_location="cpu")
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
    print(f"[INFO] HM weights loaded: {matched} tensors from {weights}")
    return model, stride


def letterbox_params(width: int, height: int, size: int) -> tuple[float, int, int, int, int]:
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    pad_w, pad_h = size - new_w, size - new_h
    left, top = pad_w // 2, pad_h // 2
    return scale, left, top, new_w, new_h


def letterbox_gray(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - new_w, size - new_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114)


def to_native(point: np.ndarray, scale: float, left: int, top: int) -> np.ndarray:
    return np.array([(point[0] - left) / scale, (point[1] - top) / scale], dtype=np.float32)


def build_hm_features(gray_frames: list[np.ndarray], imgsz: int, window: int, stride: int, gmc: FastGMCEstimator) -> np.ndarray:
    features = np.zeros((len(gray_frames), 3, imgsz, imgsz), dtype=np.uint8)
    for index, current in enumerate(tqdm(gray_frames, desc="HM features (GMC+median)", unit="frame", dynamic_ncols=True)):
        lag_index = max(0, index - stride)
        lag_frame = gray_frames[lag_index]
        motion = cv2.absdiff(current, gmc.warp(lag_frame, gmc.compute_affine(lag_frame, current)))
        history = []
        for step in range(1, window + 1):
            history_index = max(0, index - step * stride)
            history_frame = gray_frames[history_index]
            history.append(gmc.warp(history_frame, gmc.compute_affine(history_frame, current)))
        background = np.median(np.stack(history, axis=0), axis=0).astype(np.float32)
        residual = np.clip(current.astype(np.float32) - background, 0, 255).astype(np.uint8)
        merged = np.stack([residual, motion, current], axis=2)
        features[index] = letterbox_gray(merged, imgsz).transpose(2, 0, 1)
    return features


def build_bbox_inputs(gray_frames: list[np.ndarray]) -> list[np.ndarray]:
    inputs = []
    for index, current in enumerate(gray_frames):
        previous = gray_frames[max(0, index - 1)]
        previous2 = gray_frames[max(0, index - 2)]
        inputs.append(np.stack([current, cv2.absdiff(current, previous), cv2.absdiff(current, previous2)], axis=2))
    return inputs


def run_hm(model, features: np.ndarray, device: torch.device, batch: int, conf: float, top_k: int, stride: int):
    all_peaks, all_heatmaps = [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(features), batch), desc="HM inference", unit="batch", dynamic_ncols=True):
            chunk = torch.from_numpy(features[start : start + batch]).to(device).float() / 255.0
            prediction = model(chunk)
            all_peaks.extend(
                extract_peaks(prediction["heatmap"], prediction["offset"], stride=stride, conf_thresh=conf, top_k=top_k)
            )
            all_heatmaps.extend(prediction["heatmap"][:, 0].cpu().numpy())
    return all_peaks, all_heatmaps


def run_bbox(model, inputs: list[np.ndarray], device: str, conf: float, iou: float, batch: int, imgsz: int):
    results = []
    for start in tqdm(range(0, len(inputs), batch), desc="YOLO26 BBox inference", unit="batch", dynamic_ncols=True):
        chunk = inputs[start : start + batch]
        predictions = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        for prediction in predictions:
            if prediction.boxes is not None and len(prediction.boxes) > 0:
                boxes = prediction.boxes.xyxy.cpu().numpy().astype(np.float32)
                confs = prediction.boxes.conf.cpu().numpy().astype(np.float32)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                confs = np.zeros((0,), dtype=np.float32)
            results.append((boxes, confs))
    return results


def estimate_coarse_size(
    heatmap: np.ndarray,
    peak_lb: np.ndarray,
    conf: float,
    scale: float,
    stride: int,
    coarse_min: float,
    coarse_max: float,
) -> tuple[float, float]:
    height, width = heatmap.shape[:2]
    radius = 20
    cx = int(round(peak_lb[0] / stride))
    cy = int(round(peak_lb[1] / stride))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    window = heatmap[y0:y1, x0:x1]
    if window.size == 0:
        return coarse_min, coarse_min
    threshold = max(conf, 0.5 * float(window.max()))
    ys, xs = np.nonzero(window >= threshold)
    if len(xs) < 2:
        return coarse_min, coarse_min
    width_native = (xs.max() - xs.min() + 1) * stride / scale
    height_native = (ys.max() - ys.min() + 1) * stride / scale
    return (
        float(np.clip(width_native, coarse_min, coarse_max)),
        float(np.clip(height_native, coarse_min, coarse_max)),
    )


def merge_frame_detections(
    points_lb: np.ndarray,
    scores_lb: np.ndarray,
    heatmap: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    native_width: int,
    native_height: int,
    scale: float,
    left: int,
    top: int,
    stride: int,
    hm_conf: float,
    coarse_min: float,
    coarse_max: float,
    include_bbox_only: bool,
) -> tuple[list[str], dict, dict[int, int]]:
    lines = []
    frame_record = {"detections": []}
    class_counts = {0: 0, 1: 0, 2: 0}
    used_boxes = set()
    for point_lb, score in zip(points_lb, scores_lb):
        point = to_native(point_lb, scale, left, top)
        candidates = [
            (float(confs[b]), b)
            for b in range(len(boxes))
            if boxes[b][0] <= point[0] <= boxes[b][2] and boxes[b][1] <= point[1] <= boxes[b][3]
        ]
        if candidates:
            best_conf, best_box = max(candidates, key=lambda item: item[0])
            x1, y1, x2, y2 = boxes[best_box]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            width, height = x2 - x1, y2 - y1
            class_id = 0
            used_boxes.add(best_box)
            source = "hm_bbox"
            source_conf = best_conf
        else:
            width, height = estimate_coarse_size(heatmap, point_lb, hm_conf, scale, stride, coarse_min, coarse_max)
            cx, cy = point[0], point[1]
            class_id = 1
            source = "hm_coarse"
            source_conf = float(score)
        lines.append(f"{class_id} {cx / native_width:.6f} {cy / native_height:.6f} {width / native_width:.6f} {height / native_height:.6f}")
        class_counts[class_id] += 1
        frame_record["detections"].append({"source": source, "class": class_id, "score": round(source_conf, 4), "center": [round(float(cx), 2), round(float(cy), 2)], "size": [round(float(width), 2), round(float(height), 2)]})

    if include_bbox_only:
        for b in range(len(boxes)):
            if b in used_boxes:
                continue
            x1, y1, x2, y2 = boxes[b]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            width, height = x2 - x1, y2 - y1
            lines.append(f"2 {cx / native_width:.6f} {cy / native_height:.6f} {width / native_width:.6f} {height / native_height:.6f}")
            class_counts[2] += 1
            frame_record["detections"].append({"source": "bbox_only", "class": 2, "score": round(float(confs[b]), 4), "center": [round(float(cx), 2), round(float(cy), 2)], "size": [round(float(width), 2), round(float(height), 2)]})
    return lines, frame_record, class_counts


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    output_root = Path(args.output_root)
    if not frames_dir.is_dir():
        raise NotADirectoryError(frames_dir)
    seq_name = args.seq or frames_dir.name
    frame_paths = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))
    if args.limit:
        frame_paths = frame_paths[: args.limit]
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    images_dir = output_root / "images" / args.split
    labels_dir = output_root / "labels" / args.split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    gray_frames = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in frame_paths]
    if any(frame is None for frame in gray_frames):
        raise RuntimeError("At least one frame cannot be decoded")
    native_height, native_width = gray_frames[0].shape[:2]
    scale, left, top, _, _ = letterbox_params(native_width, native_height, args.imgsz)
    print(f"[INFO] Sequence: {seq_name} | frames: {len(frame_paths)} | native: {native_width}x{native_height} | letterbox scale={scale}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    hm_model, stride = load_hm_model(resolve_weights(args.hm_weights), device)
    bbox_model = YOLO(str(resolve_weights(args.bbox_weights)))

    started = time.perf_counter()
    features = build_hm_features(gray_frames, args.imgsz, args.median_window, args.temporal_stride, FastGMCEstimator(downscale=2))
    hm_peaks, hm_heatmaps = run_hm(hm_model, features, device, 16, args.hm_conf, args.hm_top_k, stride)
    bbox_inputs = build_bbox_inputs(gray_frames)
    bbox_results = run_bbox(bbox_model, bbox_inputs, args.device, args.bbox_conf, args.bbox_iou, 16, args.imgsz)
    print(f"[INFO] Inference finished in {time.perf_counter() - started:.1f}s")

    provenance = []
    class_counts = {0: 0, 1: 0, 2: 0}
    for index, frame_path in enumerate(tqdm(frame_paths, desc="Merging labels", unit="frame", dynamic_ncols=True)):
        peaks = hm_peaks[index]
        heatmap = hm_heatmaps[index]
        boxes, confs = bbox_results[index]

        if len(peaks["points"]) > 0:
            order = np.argsort(-peaks["scores"])
            points_lb = peaks["points"][order]
            scores_lb = peaks["scores"][order]
        else:
            points_lb = np.zeros((0, 2), dtype=np.float32)
            scores_lb = np.zeros((0,), dtype=np.float32)

        lines, frame_record, counts = merge_frame_detections(
            points_lb,
            scores_lb,
            heatmap,
            boxes,
            confs,
            native_width,
            native_height,
            scale,
            left,
            top,
            stride,
            args.hm_conf,
            args.coarse_min,
            args.coarse_max,
            args.include_bbox_only,
        )
        frame_record["frame"] = frame_path.name
        for class_id, value in counts.items():
            class_counts[class_id] += value

        target_stem = f"{seq_name}__{index:06d}"
        (labels_dir / f"{target_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not args.no_copy:
            shutil.copy(frame_path, images_dir / f"{target_stem}{frame_path.suffix}")
        frame_record["label"] = f"{target_stem}.txt"
        provenance.append(frame_record)

    names = {0: "uav_bbox", 1: "uav_hm_coarse"}
    if args.include_bbox_only:
        names[2] = "uav_bbox_only"
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        f"path: {output_root.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n"
        + "\n".join(f"  {key}: {value}" for key, value in names.items())
        + "\n",
        encoding="utf-8",
    )
    (output_root / f"provenance_{seq_name}.json").write_text(
        json.dumps({"sequence": seq_name, "frames": provenance}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[SUCCESS] images -> {images_dir}")
    print(f"[SUCCESS] labels -> {labels_dir}")
    print(f"[SUCCESS] data.yaml -> {data_yaml}")
    print(f"[INFO] class 0 (hm_bbox): {class_counts[0]} | class 1 (hm_coarse): {class_counts[1]} | class 2 (bbox_only): {class_counts[2]}")


if __name__ == "__main__":
    main()
