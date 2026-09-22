#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Parse YOLO labels and render a review video for one sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CLASS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}
CLASS_NAMES = {0: "uav_bbox", 1: "uav_hm_coarse", 2: "uav_bbox_only"}


def parse_args():
    parser = argparse.ArgumentParser(description="Render a YOLO label review video")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--seq", default="", help="Sequence prefix filter")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scale", type=float, default=2.0, help="Render upscale factor")
    parser.add_argument("--show-centers", action="store_true")
    return parser.parse_args()


def read_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append((int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return rows


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    images_dir = root / "images" / args.split
    labels_dir = root / "labels" / args.split
    if not images_dir.is_dir():
        raise NotADirectoryError(images_dir)

    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.seq:
        images = [path for path in images if path.stem.startswith(args.seq)]
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images for sequence '{args.seq}' in {images_dir}")

    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    out_w, out_h = int(round(width * args.scale)), int(round(height * args.scale))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {output_path}")

    for index, image_path in enumerate(tqdm(images, desc="Rendering labels", unit="frame", dynamic_ncols=True)):
        image = cv2.imread(str(image_path))
        if image is None:
            image = np.zeros((height, width, 3), dtype=np.uint8)
        canvas = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        labels = read_labels(labels_dir / f"{image_path.stem}.txt")

        counts: dict[int, int] = {}
        for class_id, cx, cy, box_w, box_h in labels:
            counts[class_id] = counts.get(class_id, 0) + 1
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            x1 = int(round((cx - box_w / 2.0) * width * args.scale))
            y1 = int(round((cy - box_h / 2.0) * height * args.scale))
            x2 = int(round((cx + box_w / 2.0) * width * args.scale))
            y2 = int(round((cy + box_h / 2.0) * height * args.scale))
            x1, x2 = sorted((max(0, x1), min(out_w - 1, x2)))
            y1, y2 = sorted((max(0, y1), min(out_h - 1, y2)))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES.get(class_id, class_id)} {box_w * width:.0f}x{box_h * height:.0f}"
            cv2.putText(canvas, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if args.show_centers:
                cv2.drawMarker(
                    canvas,
                    (int(round(cx * width * args.scale)), int(round(cy * height * args.scale))),
                    color,
                    cv2.MARKER_CROSS,
                    8,
                    1,
                )

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (out_w, 36), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)
        left = f"YOLO LABELS | {image_path.stem} | F:{index:06d}/{len(images):06d} | {width}x{height}"
        right = " | ".join(f"c{key}:{value}" for key, value in sorted(counts.items())) or "no labels"
        cv2.putText(canvas, left, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        text_width = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        cv2.putText(canvas, right, (out_w - text_width - 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    print(f"[SUCCESS] Saved: {output_path.resolve()} | frames: {len(images)} | size: {out_w}x{out_h}")


if __name__ == "__main__":
    main()
