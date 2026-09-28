#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render a local dynamic review video for sparse YOLO labels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"__(\d+)$", path.stem)
    return (int(match.group(1)) if match else -1, path.stem)


def read_label(path: Path) -> tuple[int, float, float, float, float] | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) >= 5:
            return tuple([int(float(values[0])), *map(float, values[1:5])])
    return None


def collect_frames(root: Path) -> list[tuple[Path, Path]]:
    frames = []
    for part in ("part1", "part2"):
        image_dir = root / part / "images"
        label_dir = root / part / "labels"
        for image_path in image_dir.glob("*.jpg"):
            frames.append((image_path, label_dir / f"{image_path.stem}.txt"))
    return sorted(frames, key=lambda item: natural_key(item[0]))


def local_contrast(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1.0, 99.0))
    if high <= low + 1e-6:
        low, high = float(image.min()), float(image.max())
    if high <= low + 1e-6:
        return np.zeros_like(image)
    return np.clip((image.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def crop_with_padding(image: np.ndarray, cx: float, cy: float, crop_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    half = crop_size // 2
    center_x, center_y = round(cx), round(cy)
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REPLICATE)
    return padded[center_y : center_y + crop_size, center_x : center_x + crop_size]


def draw_crosshair(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = center
    gap = 4
    length = 12
    cv2.line(image, (x - length, y), (x - gap, y), color, 1, cv2.LINE_AA)
    cv2.line(image, (x + gap, y), (x + length, y), color, 1, cv2.LINE_AA)
    cv2.line(image, (x, y - length), (x, y - gap), color, 1, cv2.LINE_AA)
    cv2.line(image, (x, y + gap), (x, y + length), color, 1, cv2.LINE_AA)
    cv2.rectangle(image, (x - 1, y - 1), (x + 1, y + 1), color, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render local enlarged YOLO-label review video")
    parser.add_argument("--label-root", required=True, help="Directory containing part1/ and part2/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--zoom", type=int, default=16)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0, help="Inclusive frame number; 0 means the last frame")
    parser.add_argument("--skip-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.crop_size <= 0 or args.crop_size % 2:
        raise ValueError("--crop-size must be a positive even number")
    if args.zoom <= 0:
        raise ValueError("--zoom must be positive")

    root = Path(args.label_root)
    frames = collect_frames(root)
    if not frames:
        raise FileNotFoundError(f"No JPG frames found under {root}/part1 or {root}/part2")

    labels = [read_label(label_path) for _, label_path in frames]
    first_labeled = next((label for label in labels if label is not None), None)
    if first_labeled is None:
        raise ValueError("No non-empty YOLO labels found")

    selected = []
    end = args.end if args.end > 0 else 10**18
    for index, item in enumerate(frames):
        frame_number = natural_key(item[0])[0]
        if args.start <= frame_number <= end and (not args.skip_empty or labels[index] is not None):
            selected.append(index)
    if not selected:
        raise ValueError("No frames selected")

    first = cv2.imread(str(frames[selected[0]][0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Cannot read image: {frames[selected[0]][0]}")
    height, width = first.shape[:2]
    crop_output_size = args.crop_size * args.zoom
    overview_width = 512
    overview_height = round(height * overview_width / width)
    panel_width = max(overview_width, crop_output_size)
    output_width = panel_width * 2
    output_height = max(overview_height, crop_output_size) + 64

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (output_width, output_height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {output_path}")

    last_center = (first_labeled[1] * width, first_labeled[2] * height)
    for output_index, frame_index in enumerate(selected):
        image_path, _ = frames[frame_index]
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (height, width):
            raise RuntimeError(f"Invalid image or resolution mismatch: {image_path}")
        label = labels[frame_index]
        if label is not None:
            center = (label[1] * width, label[2] * height)
            last_center = center
            box_width = label[3] * width
            box_height = label[4] * height
            label_text = f"class={label[0]} box={box_width:.1f}x{box_height:.1f}px"
            status = "LABELED"
        else:
            center = last_center
            box_width = box_height = 0.0
            label_text = "no label; crop follows last labeled center"
            status = "EMPTY"

        crop = crop_with_padding(local_contrast(gray), center[0], center[1], args.crop_size)
        crop = cv2.resize(crop, (crop_output_size, crop_output_size), interpolation=cv2.INTER_NEAREST)
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        draw_crosshair(crop, (crop_output_size // 2, crop_output_size // 2), (0, 255, 255))

        overview = cv2.resize(gray, (overview_width, overview_height), interpolation=cv2.INTER_AREA)
        overview = cv2.cvtColor(overview, cv2.COLOR_GRAY2BGR)
        scale_x = overview_width / width
        scale_y = overview_height / height
        cx, cy = round(center[0] * scale_x), round(center[1] * scale_y)
        draw_crosshair(overview, (cx, cy), (0, 255, 255))
        if label is not None:
            x1 = round((center[0] - box_width / 2) * scale_x)
            y1 = round((center[1] - box_height / 2) * scale_y)
            x2 = round((center[0] + box_width / 2) * scale_x)
            y2 = round((center[1] + box_height / 2) * scale_y)
            cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 255, 0), 1)

        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        canvas[:overview_height, :overview_width] = overview
        crop_x = panel_width
        crop_y = max(0, (max(overview_height, crop_output_size) - crop_output_size) // 2)
        canvas[crop_y : crop_y + crop_output_size, crop_x : crop_x + crop_output_size] = crop
        cv2.putText(canvas, "OVERVIEW", (10, output_height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "LOCAL 32x32 | nearest + contrast stretch", (crop_x + 10, output_height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        frame_number = natural_key(image_path)[0]
        info = f"{status} | frame={frame_number:04d} | {output_index + 1}/{len(selected)} | center=({center[0]:.1f},{center[1]:.1f}) | {label_text}"
        cv2.rectangle(canvas, (0, 0), (output_width, 30), (20, 20, 20), -1)
        cv2.putText(canvas, info, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255) if label is None else (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    print(f"[SUCCESS] Saved: {output_path.resolve()} | frames={len(selected)} | size={output_width}x{output_height}")


if __name__ == "__main__":
    main()
