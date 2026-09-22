#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Generate HM+BBox pseudo-labels for every restored sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from manu.preannotate_hm_bbox import (
    letterbox_gray,
    letterbox_params,
    load_hm_model,
    merge_frame_detections,
    resolve_weights,
)


def is_cuda_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def infer_hm(model, features: np.ndarray, device: torch.device, stride: int, conf: float, top_k: int, batch_size: int):
    peaks, heatmaps = [], []
    start = 0
    size = max(1, batch_size)
    while start < len(features):
        count = min(size, len(features) - start)
        try:
            with torch.no_grad():
                tensor = torch.from_numpy(features[start : start + count]).to(device).float() / 255.0
                prediction = model(tensor)
                peaks.extend(extract_peaks(prediction["heatmap"], prediction["offset"], stride=stride, conf_thresh=conf, top_k=top_k))
                heatmaps.extend(prediction["heatmap"][:, 0].cpu().numpy())
            start += count
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            torch.cuda.empty_cache()
            if size == 1:
                raise
            size = max(1, count // 2)
            print(f"[WARN] CUDA OOM during HM inference, reducing batch size to {size}", flush=True)
    return peaks, heatmaps


def infer_bbox(model, inputs: list[np.ndarray], device: str, conf: float, iou: float, imgsz: int, batch_size: int):
    results = []
    start = 0
    size = max(1, batch_size)
    while start < len(inputs):
        count = min(size, len(inputs) - start)
        try:
            predictions = model.predict(inputs[start : start + count], imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
            for prediction in predictions:
                if prediction.boxes is not None and len(prediction.boxes) > 0:
                    boxes = prediction.boxes.xyxy.cpu().numpy().astype(np.float32)
                    confs = prediction.boxes.conf.cpu().numpy().astype(np.float32)
                else:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    confs = np.zeros((0,), dtype=np.float32)
                results.append((boxes, confs))
            start += count
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            torch.cuda.empty_cache()
            if size == 1:
                raise
            size = max(1, count // 2)
            print(f"[WARN] CUDA OOM during BBox inference, reducing batch size to {size}", flush=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-annotate all restored video sequences")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--bbox-weights", default="runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--hm-conf", type=float, default=0.06)
    parser.add_argument("--hm-top-k", type=int, default=50)
    parser.add_argument("--bbox-conf", type=float, default=0.06)
    parser.add_argument("--bbox-iou", type=float, default=0.70)
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--coarse-min", type=float, default=6.0)
    parser.add_argument("--coarse-max", type=float, default=40.0)
    parser.add_argument("--include-bbox-only", action="store_true")
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing images/labels trees under output-root")
    parser.add_argument("--sequences", default="", help="Comma-separated sequence directory names")
    parser.add_argument("--limit-per-sequence", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=64, help="Frames held and inferred per chunk")
    parser.add_argument("--worker", action="store_true", help="Single-sequence mode: skip global data.yaml/manifest writes")
    parser.add_argument("--cpu-threads", type=int, default=1, help="OpenCV thread count per process (parallel mode)")
    parser.add_argument("--batch-size", type=int, default=8, help="GPU batch size, auto-halved on CUDA OOM")
    parser.add_argument("--yolo-batch-size", type=int, default=8, help="YOLO batch size, auto-halved on CUDA OOM")
    return parser.parse_args()


def main():
    args = parse_args()
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    if not frames_root.is_dir():
        raise NotADirectoryError(frames_root)
    if output_root.resolve() == frames_root.resolve() or frames_root.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("Refusing to overwrite frames-root with output-root")

    sequence_filter = {item.strip() for item in args.sequences.split(",") if item.strip()}
    sequence_dirs = [path for path in sorted(frames_root.iterdir()) if path.is_dir() and (not sequence_filter or path.name in sequence_filter)]
    if not sequence_dirs:
        raise FileNotFoundError(f"No sequence directories under {frames_root}")

    if args.worker:
        if args.overwrite:
            raise ValueError("--overwrite is not allowed in --worker mode")
        if len(sequence_dirs) != 1:
            raise ValueError("--worker mode expects exactly one sequence")
    cv2.setNumThreads(max(1, args.cpu_threads))

    images_dir = output_root / "images" / args.split
    labels_dir = output_root / "labels" / args.split
    if args.overwrite:
        print(f"[WARN] Removing existing {output_root / 'images'} and {output_root / 'labels'}", flush=True)
        shutil.rmtree(output_root / "images", ignore_errors=True)
        shutil.rmtree(output_root / "labels", ignore_errors=True)
        for stale_file in [output_root / "data.yaml", output_root / "manifest.json", *output_root.glob("provenance_*.json")]:
            stale_file.unlink(missing_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    hm_model, hm_stride = load_hm_model(resolve_weights(args.hm_weights), device)
    bbox_model = YOLO(str(resolve_weights(args.bbox_weights)))
    print(f"[INFO] Sequences: {len(sequence_dirs)} | split: {args.split} | device: {device}", flush=True)

    gmc = FastGMCEstimator(downscale=2)
    history_window = args.median_window * args.temporal_stride + 2
    global_counts = {0: 0, 1: 0, 2: 0}
    sequence_summaries = []
    total_started = time.perf_counter()
    for sequence_index, frames_dir in enumerate(sequence_dirs, 1):
        frame_paths = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))
        if args.limit_per_sequence:
            frame_paths = frame_paths[: args.limit_per_sequence]
        if not frame_paths:
            print(f"[WARN] {frames_dir.name}: no frames, skipped", flush=True)
            continue
        print(f"\n[SEQUENCE {sequence_index}/{len(sequence_dirs)}] {frames_dir.name} | frames={len(frame_paths)}", flush=True)

        history: list[np.ndarray] = []
        sequence_counts = {0: 0, 1: 0, 2: 0}
        provenance = []
        for start in tqdm(range(0, len(frame_paths), args.chunk_size), desc=f"Processing {frames_dir.name}", unit="chunk", dynamic_ncols=True):
            chunk_paths = frame_paths[start : start + args.chunk_size]
            chunk_grays = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in chunk_paths]
            if any(frame is None for frame in chunk_grays):
                raise RuntimeError(f"Unreadable frame in {frames_dir}")
            if start == 0:
                native_height, native_width = chunk_grays[0].shape[:2]
                if (native_width, native_height) != (640, 512):
                    raise ValueError(f"{frames_dir.name}: expected 640x512, got {native_width}x{native_height}")
                scale, left, top, _, _ = letterbox_params(native_width, native_height, args.imgsz)

            features = np.zeros((len(chunk_grays), 3, args.imgsz, args.imgsz), dtype=np.uint8)
            bbox_inputs = []
            rolling = list(history)
            for index, current in enumerate(chunk_grays):
                lag_frame = rolling[max(0, len(rolling) - args.temporal_stride)] if rolling else current
                motion = cv2.absdiff(current, gmc.warp(lag_frame, gmc.compute_affine(lag_frame, current)))
                aligned_history = []
                for step in range(1, args.median_window + 1):
                    history_index = max(0, len(rolling) - step * args.temporal_stride)
                    history_frame = rolling[history_index] if rolling else current
                    aligned_history.append(gmc.warp(history_frame, gmc.compute_affine(history_frame, current)))
                background = np.median(np.stack(aligned_history, axis=0), axis=0).astype(np.float32)
                residual = np.clip(current.astype(np.float32) - background, 0, 255).astype(np.uint8)
                features[index] = letterbox_gray(np.stack([residual, motion, current], axis=2), args.imgsz).transpose(2, 0, 1)

                previous = rolling[-1] if len(rolling) >= 1 else current
                previous2 = rolling[-2] if len(rolling) >= 2 else previous
                bbox_inputs.append(np.stack([current, cv2.absdiff(current, previous), cv2.absdiff(current, previous2)], axis=2))

                rolling.append(current)
                if len(rolling) > history_window:
                    rolling.pop(0)
            history = rolling

            hm_peaks, hm_heatmaps = infer_hm(
                hm_model, features, device, hm_stride, args.hm_conf, args.hm_top_k, args.batch_size
            )
            bbox_results = infer_bbox(
                bbox_model, bbox_inputs, args.device, args.bbox_conf, args.bbox_iou, args.imgsz, args.yolo_batch_size
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

            for index, frame_path in enumerate(chunk_paths):
                peaks = hm_peaks[index]
                order = np.argsort(-peaks["scores"]) if len(peaks["scores"]) else np.zeros((0,), dtype=np.int64)
                points_lb = peaks["points"][order] if len(order) else np.zeros((0, 2), dtype=np.float32)
                scores_lb = peaks["scores"][order] if len(order) else np.zeros((0,), dtype=np.float32)
                boxes, confs = bbox_results[index]
                lines, frame_record, counts = merge_frame_detections(
                    points_lb, scores_lb, hm_heatmaps[index], boxes, confs,
                    native_width, native_height, scale, left, top, hm_stride,
                    args.hm_conf, args.coarse_min, args.coarse_max, args.include_bbox_only,
                )
                global_index = start + index
                target_stem = f"{frames_dir.name}__{global_index:06d}"
                (labels_dir / f"{target_stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                if not args.no_copy:
                    shutil.copy2(frame_path, images_dir / f"{target_stem}{frame_path.suffix}")
                frame_record["frame"] = frame_path.name
                frame_record["label"] = f"{target_stem}.txt"
                provenance.append(frame_record)
                for class_id in sequence_counts:
                    sequence_counts[class_id] += counts[class_id]
                    global_counts[class_id] += counts[class_id]

        (output_root / f"provenance_{frames_dir.name}.json").write_text(
            json.dumps({"sequence": frames_dir.name, "frames": provenance}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {"sequence": frames_dir.name, "frames": len(frame_paths), "counts": sequence_counts}
        sequence_summaries.append(summary)
        summaries_dir = output_root / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        (summaries_dir / f"{frames_dir.name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] {frames_dir.name}: c0={sequence_counts[0]} c1={sequence_counts[1]} c2={sequence_counts[2]}", flush=True)

    if args.worker:
        print(f"[WORKER DONE] {sequence_summaries[0]['sequence']} -> {images_dir.parent.parent.resolve()}", flush=True)
        return

    names = {0: "uav_bbox", 1: "uav_hm_coarse"}
    if args.include_bbox_only:
        names[2] = "uav_bbox_only"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data.yaml").write_text(
        f"path: {output_root.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n"
        + "\n".join(f"  {key}: {value}" for key, value in names.items())
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "frames_root": str(frames_root.resolve()),
        "output_root": str(output_root.resolve()),
        "split": args.split,
        "sequence_count": len(sequence_summaries),
        "sequences": sequence_summaries,
        "global_counts": global_counts,
        "elapsed_seconds": time.perf_counter() - total_started,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Total sequences={len(sequence_summaries)} | c0={global_counts[0]} c1={global_counts[1]} c2={global_counts[2]}", flush=True)
    print(f"[SUCCESS] Output: {output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
