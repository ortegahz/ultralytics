#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run the YOLO26 BBox expert on an H.264 video with diagnostic OSD."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Infer YOLO26 BBox model on an H.264 video")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--mode", choices=["diff", "raw"], default="diff")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--fps", type=float, default=0.0)
    return parser.parse_args()


def make_input(history: deque[np.ndarray], mode: str) -> np.ndarray:
    current = history[-1]
    previous = history[-2]
    previous2 = history[-3]
    if mode == "raw":
        return np.stack([current, previous, previous2], axis=2)
    return np.stack([current, cv2.absdiff(current, previous), cv2.absdiff(current, previous2)], axis=2)


def draw_frame(frame: np.ndarray, result, index: int, total: int, mode: str, conf: float, processing_fps: float) -> np.ndarray:
    canvas = frame.copy()
    boxes = result.boxes
    count = len(boxes)
    for box, score in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = np.round(box).astype(int)
        color = (255, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"YOLO26 {score:.2f} [{x2 - x1}x{y2 - y1}]"
        cv2.putText(canvas, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, ((x1 + x2) // 2, (y1 + y2) // 2), color, cv2.MARKER_CROSS, 8, 1)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    left = f"YOLO26 BBOX | mode={mode} | F:{index:06d}/{total:06d}"
    right = f"Det:{count} | conf>={conf:.2f} | FPS:{processing_fps:.1f}"
    cv2.putText(canvas, left, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    width = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
    cv2.putText(canvas, right, (canvas.shape[1] - width - 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    input_path = Path(args.input)
    weights_path = Path(args.weights)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not weights_path.exists():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(args.weights)

    model = YOLO(str(weights_path))
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = args.fps if args.fps > 0 else source_fps
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    history: deque[np.ndarray] = deque(maxlen=3)
    started = time.perf_counter()
    processed = 0
    progress = tqdm(total=total or None, desc="YOLO26 H.264 inference", unit="frame", dynamic_ncols=True)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        history.append(gray)
        if len(history) < 3:
            progress.update(1)
            continue
        tensor_input = make_input(history, args.mode)
        result = model.predict(tensor_input, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open output video: {output_path}")
        processed += 1
        elapsed = time.perf_counter() - started
        rendered = draw_frame(frame, result, processed, total, args.mode, args.conf, processed / max(elapsed, 1e-6))
        writer.write(rendered)
        progress.update(1)
        progress.set_postfix(det=len(result.boxes), process_fps=f"{processed / max(elapsed, 1e-6):.1f}")

    progress.close()
    capture.release()
    if writer is not None:
        writer.release()
    if processed == 0:
        raise RuntimeError("The video did not contain three decodable frames")
    print(f"[SUCCESS] Saved: {output_path.resolve()}")
    print(f"[INFO] Mode: {args.mode} | Processed: {processed} | Input FPS: {source_fps:.2f} | Output FPS: {output_fps:.2f}")


if __name__ == "__main__":
    main()
