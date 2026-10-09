#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render a ground-truth OSD review video for one sequence from a flat Final_Labels directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
CLASS_COLORS = ((0, 255, 0), (0, 165, 255), (0, 255, 255), (0, 0, 255))
HEADER_HEIGHT = 36
PADDING = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a GT OSD review video from Final_Labels")
    parser.add_argument("--labels-root", required=True, help="Flat directory of {sequence}__{index:06d}.txt")
    parser.add_argument("--frames-root", required=True, help="Root holding <sequence>/frame_XXXXXX.jpg")
    parser.add_argument("--seq", default="", help="Sequence name; omit together with --output when using --list")
    parser.add_argument("--list", action="store_true", help="Print every sequence with label counts, then exit")
    parser.add_argument("--output", default="", help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start", type=int, default=0, help="Inclusive first frame index")
    parser.add_argument("--end", type=int, default=0, help="Inclusive last frame index; 0 means the last frame")
    parser.add_argument("--skip-empty", action="store_true", help="Drop frames that carry no label")
    parser.add_argument("--scale", type=float, default=2.0, help="Full-frame upscale factor")
    parser.add_argument("--inset", type=int, default=512, help="Edge of the local zoom panel in pixels; 0 disables it")
    parser.add_argument("--crop-size", type=int, default=48, help="Local zoom crop edge in native pixels")
    args = parser.parse_args()
    if args.list:
        return args
    missing = [name for name, value in (("--seq", args.seq), ("--output", args.output)) if not value]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    if args.scale <= 0 or args.inset < 0 or args.crop_size <= 0 or args.crop_size % 2:
        parser.error("--scale must be positive, --inset non-negative, --crop-size a positive even number")
    return args


def index_labels(labels_root: Path) -> dict[str, list[int]]:
    """Group every {sequence}__{index:06d}.txt under labels_root by sequence, sorted by frame index."""
    grouped: dict[str, list[int]] = {}
    for path in labels_root.iterdir():
        if path.suffix.lower() != ".txt":
            continue
        sequence, separator, digits = path.stem.rpartition("__")
        if not separator or not digits.isdigit():
            continue
        grouped.setdefault(sequence, []).append(int(digits))
    for indices in grouped.values():
        indices.sort()
    return dict(sorted(grouped.items()))


def list_frames(sequence_dir: Path) -> dict[int, Path]:
    """Map trailing frame number to path, so labels never rely on filename guessing or lexicographic order."""
    frames: dict[int, Path] = {}
    for path in sequence_dir.iterdir():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digits = path.stem.rpartition("_")[2]
        if digits.isdigit():
            frames[int(digits)] = path
    return dict(sorted(frames.items()))


def class_names(labels_root: Path) -> dict[int, str]:
    path = labels_root / "classes.txt"
    if not path.is_file():
        return {}
    return {
        index: name.strip() for index, name in enumerate(path.read_text(encoding="utf-8").splitlines()) if name.strip()
    }


def read_labels(path: Path) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Parse one YOLO label file, dropping exact duplicate rows (Final_Labels contains 22 of them)."""
    rows: list[tuple[int, float, float, float, float]] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    duplicates = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        row = (int(float(values[0])), *(float(value) for value in values[1:5]))
        if row in seen:
            duplicates += 1
            continue
        seen.add(row)
        rows.append(row)
    return rows, duplicates


def local_contrast(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, (1.0, 99.0))
    if high <= low + 1e-6:
        low, high = float(gray.min()), float(gray.max())
    if high <= low + 1e-6:
        return np.zeros_like(gray)
    return np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def crop_with_padding(gray: np.ndarray, cx: float, cy: float, crop_size: int) -> np.ndarray:
    half = crop_size // 2
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_REPLICATE)
    return padded[round(cy) : round(cy) + crop_size, round(cx) : round(cx) + crop_size]


def draw_box(
    canvas: np.ndarray, box: tuple[float, float, float, float], color: tuple[int, int, int], thickness: int
) -> None:
    x1, y1, x2, y2 = box
    height, width = canvas.shape[:2]
    cv2.rectangle(
        canvas,
        (max(0, min(width - 1, round(x1))), max(0, min(height - 1, round(y1)))),
        (max(0, min(width - 1, round(x2))), max(0, min(height - 1, round(y2)))),
        color,
        thickness,
    )


def main() -> None:
    args = parse_args()
    labels_root = Path(args.labels_root)
    if not labels_root.is_dir():
        raise NotADirectoryError(labels_root)

    grouped = index_labels(labels_root)
    if not grouped:
        raise FileNotFoundError(f"No {{sequence}}__{{index:06d}}.txt labels found in {labels_root}")

    if args.list:
        print(f"{'sequence':34s} {'labels':>8s} {'non-empty':>10s} {'first':>7s} {'last':>7s}")
        for sequence, indices in grouped.items():
            labeled = sum(1 for index in indices if (labels_root / f"{sequence}__{index:06d}.txt").stat().st_size > 0)
            print(f"{sequence:34s} {len(indices):8d} {labeled:10d} {indices[0]:7d} {indices[-1]:7d}")
        print(f"[SUCCESS] {len(grouped)} sequences | labels: {labels_root}")
        return

    if args.seq not in grouped:
        raise KeyError(f"Sequence '{args.seq}' not found in {labels_root}; run with --list to see all {len(grouped)}")

    label_indices = grouped[args.seq]
    frames = list_frames(Path(args.frames_root) / args.seq)
    if not frames:
        raise FileNotFoundError(f"No frames found in {Path(args.frames_root) / args.seq}")
    if len(frames) != len(label_indices):
        print(
            f"[WARN] {args.seq}: {len(label_indices)} labels but {len(frames)} frames; "
            "only the selected range is checked below"
        )

    selected = [
        index
        for index in label_indices
        if args.start <= index <= (args.end if args.end > 0 else 10**18)
        and (not args.skip_empty or (labels_root / f"{args.seq}__{index:06d}.txt").stat().st_size > 0)
    ]
    if not selected:
        raise ValueError("No frames selected; relax --start/--end or drop --skip-empty")
    missing = [index for index in selected if index not in frames]
    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(selected)} selected labels have no frame, first={missing[0]} last={missing[-1]}; "
            "the label index does not address the frame directory"
        )

    first = cv2.imread(str(frames[selected[0]]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError(f"Cannot decode frame {frames[selected[0]]}")
    height, width = first.shape[:2]
    out_w, out_h = int(round(width * args.scale)), int(round(height * args.scale))
    inset = args.inset if args.inset else 0
    content_h = max(out_h, inset + 2 * PADDING) if inset else out_h
    canvas_w = out_w + (inset + PADDING if inset else 0)

    names = class_names(labels_root)
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"--output must end with .mp4, got '{output_path.name}'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (canvas_w, content_h + HEADER_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {output_path}")

    boxes_total = 0
    labeled_total = 0
    duplicates_total = 0
    for position, index in enumerate(
        tqdm(selected, desc="Rendering GT OSD", unit="frame", dynamic_ncols=True), start=1
    ):
        gray = cv2.imread(str(frames[index]), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (height, width):
            raise RuntimeError(f"Undecodable frame or resolution mismatch at index {index}: {frames[index]}")
        labels, duplicates = read_labels(labels_root / f"{args.seq}__{index:06d}.txt")
        boxes_total += len(labels)
        duplicates_total += duplicates
        labeled_total += bool(labels)

        overview = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        for class_id, cx, cy, box_w, box_h in labels:
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            half_w, half_h = box_w * width * args.scale / 2, box_h * height * args.scale / 2
            center = (cx * width * args.scale, cy * height * args.scale)
            draw_box(
                overview, (center[0] - half_w, center[1] - half_h, center[0] + half_w, center[1] + half_h), color, 2
            )
            cv2.drawMarker(overview, (round(center[0]), round(center[1])), color, cv2.MARKER_CROSS, 12, 1)
            caption = f"{names.get(class_id, class_id)} {box_w * width:.0f}x{box_h * height:.0f}"
            cv2.putText(
                overview,
                caption,
                (max(0, round(center[0] - half_w)), max(14, round(center[1] - half_h) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        canvas = np.zeros((content_h + HEADER_HEIGHT, canvas_w, 3), dtype=np.uint8)
        canvas[HEADER_HEIGHT : HEADER_HEIGHT + out_h, :out_w] = overview

        if inset and labels:
            _, cx, cy, _, _ = labels[0]
            patch = cv2.resize(
                crop_with_padding(local_contrast(gray), cx * width, cy * height, args.crop_size),
                (inset, inset),
                interpolation=cv2.INTER_NEAREST,
            )
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(patch, (inset // 2, inset // 2), CLASS_COLORS[0], cv2.MARKER_CROSS, 24, 1)
            x0 = out_w + PADDING
            y0 = HEADER_HEIGHT + PADDING
            canvas[y0 : y0 + inset, x0 : x0 + inset] = patch
            cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + inset, y0 + inset), CLASS_COLORS[0], 1)
            cv2.putText(
                canvas,
                f"LOCAL {args.crop_size}x{args.crop_size} nearest+stretch",
                (x0, y0 + inset + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                CLASS_COLORS[0],
                1,
                cv2.LINE_AA,
            )
        elif inset:
            cv2.putText(
                canvas,
                "LOCAL | no label on this frame",
                (out_w + PADDING, HEADER_HEIGHT + PADDING + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (128, 128, 128),
                1,
                cv2.LINE_AA,
            )

        status = f"boxes={len(labels)}" if labels else "no label"
        header = f"GT OSD | {args.seq} | frame={index:06d} | {position}/{len(selected)} | {width}x{height} | {status}"
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas_w, HEADER_HEIGHT), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)
        cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    print(
        f"[SUCCESS] Saved: {output_path.resolve()} | frames={len(selected)} | boxes={boxes_total} | "
        f"labeled_frames={labeled_total} | duplicate_rows_dropped={duplicates_total} | "
        f"size={canvas_w}x{content_h + HEADER_HEIGHT} | native={width}x{height}"
    )


if __name__ == "__main__":
    main()
