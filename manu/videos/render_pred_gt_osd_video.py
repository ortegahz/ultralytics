#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render a single-panel pred-vs-GT review video straight from the detection cache.

The panel draws the merged multi-scale detections on top of the raw grayscale frame, with every box coloured
by its matching outcome:

    green   TP   merged prediction inside a ground-truth box
    red     FP   merged prediction with no ground truth
    white   GT   ground truth that the merged row hit
    yellow  FN   ground truth with no prediction

``--show-scales`` additionally overlays the raw per-scale detections in their own colours, which is how a
reviewer tells "this scale alone missed it" apart from "every scale missed it".

The zoom panel follows the ground truth when there is one, otherwise the strongest prediction, which is what
makes "the annotator missed this frame" visible on a sequence whose global working point is just too strict.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
COLOR_TP = (0, 220, 0)
COLOR_FP = (0, 0, 255)
COLOR_GT = (255, 255, 255)
COLOR_FN = (0, 220, 255)
SCALE_COLORS = ((255, 160, 0), (255, 0, 255), (0, 255, 255))
ROW_HEADER = 30
PAD = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a multi-scale pred-vs-GT review video from the cache")
    parser.add_argument("--cache-root", required=True, help="Directory of <sequence>.pkl")
    parser.add_argument("--labels-root", required=True, help="Flat Final_Labels directory")
    parser.add_argument("--frames-root", required=True, help="Root of <sequence>/frame_XXXXXX.jpg")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Score threshold for predictions")
    parser.add_argument("--merge-radius", type=float, default=64.0, help="Cross-scale dedup radius in native pixels")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0, help="Inclusive last frame index; 0 means the last frame")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--canvas", default="1920x1080", help="Output canvas, for example 1920x1080")
    parser.add_argument("--zoom", type=int, default=360, help="Edge of the zoom panel; 0 disables it")
    parser.add_argument("--crop-size", type=int, default=48, help="Zoom crop edge in native pixels")
    parser.add_argument(
        "--show-scales",
        default="",
        help="Comma-separated scale indices to overlay alongside the merged boxes, for example 0,1,2",
    )
    parser.add_argument("--out-json", default="", help="Optional per-frame match JSON")
    args = parser.parse_args()
    if not args.output.lower().endswith(".mp4"):
        parser.error("--output must end with .mp4")
    try:
        canvas_width, canvas_height = (int(value) for value in args.canvas.lower().split("x"))
    except ValueError:
        parser.error("--canvas must look like WIDTHxHEIGHT")
    if canvas_width < 640 or canvas_height < 480:
        parser.error("--canvas must be at least 640x480")
    args.canvas = (canvas_width, canvas_height)
    if args.zoom and (args.crop_size <= 0 or args.crop_size % 2):
        parser.error("--crop-size must be a positive even number when the zoom panel is enabled")
    return args


def collect_rows(cache: dict, threshold: float):
    """Split every cached row above the threshold by tag, sorted by descending score per frame.

    The cache also holds a baked fused row (tag == meta["fused_tag"], 255 by default); it is dropped here
    because the merged row is recomputed from the per-scale rows with the requested merge radius.
    """
    fused_tag = cache["meta"].get("fused_tag", 255)
    offset, frame_index = cache["offset"], cache["frame_index"]
    keep = cache["scores"].astype(np.float32) >= threshold
    per_tag: dict[int, dict[int, tuple]] = {}
    for position, number in enumerate(frame_index):
        start, end = offset[position], offset[position + 1]
        if start == end:
            continue
        tags = cache["tag"][start:end]
        for tag in np.unique(tags):
            tag = int(tag)
            if tag == fused_tag:
                continue
            mask = (tags == tag) & keep[start:end]
            if not mask.any():
                continue
            index = np.nonzero(mask)[0]
            order = index[np.argsort(-cache["scores"][start:end][index].astype(np.float32), kind="stable")]
            per_tag.setdefault(tag, {})[int(number)] = (
                cache["scores"][start:end][order].astype(np.float32),
                cache["points"][start:end][order],
                cache["sizes"][start:end][order],
            )
    return per_tag


def merge_scales_with_sizes(per_tag: dict[int, dict[int, tuple]], scale_tags: list[int], radius: float) -> dict:
    """Union the per-scale rows, keeping the highest score inside each radius and carrying its size along."""
    merged: dict[int, tuple] = {}
    radius_squared = radius * radius
    frames = set().union(*(set(per_tag[tag]) for tag in scale_tags)) if scale_tags else set()
    for frame in frames:
        pooled = []
        for tag in scale_tags:
            if frame not in per_tag[tag]:
                continue
            scores, points, sizes = per_tag[tag][frame]
            pooled.extend((float(score), points[i].tolist(), sizes[i].tolist()) for i, score in enumerate(scores))
        pooled.sort(key=lambda item: -item[0])
        kept, kept_xy = [], []
        for score, point, size in pooled:
            x, y = point
            if radius > 0 and any((x - px) ** 2 + (y - py) ** 2 <= radius_squared for px, py in kept_xy):
                continue
            kept.append((score, point, size))
            kept_xy.append((x, y))
        if kept:
            merged[frame] = (
                np.asarray([item[0] for item in kept], np.float32),
                np.asarray([item[1] for item in kept], np.float32),
                np.asarray([item[2] for item in kept], np.float32),
            )
    return merged


def load_ground_truth(
    labels_root: Path, sequence: str, width: int, height: int, frames: set[int]
) -> dict[int, np.ndarray]:
    """Read Final_Labels boxes as (N,4) cx,cy,w,h in native pixels for the requested frames only."""
    truth: dict[int, np.ndarray] = {}
    prefix = f"{sequence}__"
    scale = np.float32([width, height, width, height])
    with os.scandir(labels_root) as entries:
        for entry in entries:
            name = entry.name
            if not name.startswith(prefix) or not name.endswith(".txt"):
                continue
            digits = name[len(prefix) : -4]
            if not digits.isdigit() or int(digits) not in frames or entry.stat().st_size == 0:
                continue
            rows, seen = [], set()
            for line in Path(entry.path).read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) < 5:
                    continue
                row = tuple(float(value) for value in values[1:5])
                if row in seen:
                    continue
                seen.add(row)
                rows.append(row)
            if rows:
                truth[int(digits)] = np.asarray(rows, np.float32) * scale
    return truth


def match_frame(boxes: np.ndarray | None, points: np.ndarray, scores: np.ndarray):
    """Greedy nearest-ground-truth assignment; returns per-prediction hit flags and unmatched GT indices."""
    if boxes is None or not len(boxes):
        return np.zeros(len(points), bool), np.arange(len(boxes)) if boxes is not None else np.zeros(0, int)
    taken = np.zeros(len(boxes), bool)
    hits = np.zeros(len(points), bool)
    for position in range(len(points)):
        free = np.nonzero(~taken)[0]
        if not len(free):
            break
        distance = ((boxes[free, :2] - points[position]) ** 2).sum(axis=1)
        nearest = int(np.argmin(distance))
        box = free[nearest]
        if (
            abs(points[position][0] - boxes[box][0]) <= boxes[box][2] / 2
            and abs(points[position][1] - boxes[box][1]) <= boxes[box][3] / 2
        ):
            taken[box] = True
            hits[position] = True
    return hits, np.nonzero(~taken)[0]


def draw_box(panel, cx, cy, box_w, box_h, color, label, thickness: int = 2):
    height, width = panel.shape[:2]
    x1 = max(0, min(width - 1, round(cx - box_w / 2)))
    y1 = max(0, min(height - 1, round(cy - box_h / 2)))
    x2 = max(0, min(width - 1, round(cx + box_w / 2)))
    y2 = max(0, min(height - 1, round(cy + box_h / 2)))
    cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness)
    if thickness >= 2:
        cv2.drawMarker(panel, (round(cx), round(cy)), color, cv2.MARKER_CROSS, 10, 1)
    cv2.putText(panel, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def local_contrast(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, (1.0, 99.0))
    if high <= low + 1e-6:
        low, high = float(gray.min()), float(gray.max())
    if high <= low + 1e-6:
        return np.zeros_like(gray)
    return np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def zoom_panel(gray: np.ndarray, center: tuple[float, float], crop_size: int, edge: int, color) -> np.ndarray:
    half = crop_size // 2
    padded = cv2.copyMakeBorder(local_contrast(gray), half, half, half, half, cv2.BORDER_REPLICATE)
    patch = padded[round(center[1]) : round(center[1]) + crop_size, round(center[0]) : round(center[0]) + crop_size]
    patch = cv2.resize(patch, (edge, edge), interpolation=cv2.INTER_NEAREST)
    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(patch, (edge // 2, edge // 2), color, cv2.MARKER_CROSS, 20, 1)
    cv2.rectangle(patch, (0, 0), (edge - 1, edge - 1), color, 1)
    return patch


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache_root) / f"{args.seq}.pkl"
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    cache = pickle.loads(cache_path.read_bytes())
    meta = cache["meta"]
    width, height = meta["native_size"]
    scale_tags = list(range(len(meta["scales"])))
    labels_root, frames_root = Path(args.labels_root), Path(args.frames_root)

    sequence_dir = frames_root / args.seq
    available = {
        int(path.stem.rpartition("_")[2]): path
        for path in sequence_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and path.stem.rpartition("_")[2].isdigit()
    }
    if not available:
        raise FileNotFoundError(f"No frames under {sequence_dir}")

    wanted = sorted(number for number in available if args.start <= number <= (args.end or 10**18))
    if args.limit:
        wanted = wanted[: args.limit]
    if not wanted:
        raise ValueError("No frames selected")

    per_tag = collect_rows(cache, args.threshold)
    present = [tag for tag in scale_tags if tag in per_tag]
    merged = merge_scales_with_sizes(per_tag, present, args.merge_radius)
    show_scales = [int(item) for item in args.show_scales.split(",") if item.strip()]
    unknown = [tag for tag in show_scales if tag not in per_tag]
    if unknown:
        raise ValueError(f"--show-scales references tags absent from the cache: {unknown}")

    truth = load_ground_truth(labels_root, args.seq, width, height, set(wanted))

    zoom = min(args.zoom, args.canvas[1] - ROW_HEADER - 2 * PAD) if args.zoom else 0
    frame_area_w = args.canvas[0] - 2 * PAD - (zoom + PAD if zoom else 0)
    frame_area_h = args.canvas[1] - ROW_HEADER - 2 * PAD
    factor = min(frame_area_w / width, frame_area_h / height)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, tuple(args.canvas))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open {output}")

    scale_tags = [tag for tag in range(len(meta["scales"])) if tag in per_tag]
    scale_labels = {tag: str(meta["scales"][tag]) for tag in scale_tags}
    print(
        f"[INFO] {args.seq} | frames={len(wanted)} | th={args.threshold} radius={args.merge_radius} "
        f"| canvas={args.canvas[0]}x{args.canvas[1]} frame_scale={factor:.3f} zoom={zoom} "
        f"| scales={show_scales}",
        flush=True,
    )

    totals = {"tp": 0, "fp": 0, "fn": 0}
    records = []
    started = time.perf_counter()
    try:
        for number in tqdm(wanted, desc=f"{args.seq}", unit="frame", dynamic_ncols=True):
            gray = cv2.imread(str(available[number]), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape[:2] != (height, width):
                raise RuntimeError(f"Undecodable or mismatched frame {number}")
            boxes = truth.get(number)
            panel = cv2.cvtColor(
                cv2.resize(
                    gray,
                    (int(round(width * factor)), int(round(height * factor))),
                    interpolation=cv2.INTER_LINEAR,
                ),
                cv2.COLOR_GRAY2BGR,
            )

            final = merged.get(number)
            if final is not None and len(final[0]):
                scores, points, sizes = final
                hits, missed = match_frame(boxes, points, scores)
                for position in range(len(points)):
                    draw_box(
                        panel,
                        points[position][0] * factor,
                        points[position][1] * factor,
                        (sizes[position][0] * factor) or 2.0,
                        (sizes[position][1] * factor) or 2.0,
                        COLOR_TP if hits[position] else COLOR_FP,
                        f"{scores[position]:.2f}",
                    )
                matched_boxes = set(range(len(boxes))) - set(missed.tolist()) if boxes is not None else set()
                if boxes is not None:
                    for index in range(len(boxes)):
                        draw_box(
                            panel,
                            boxes[index][0] * factor,
                            boxes[index][1] * factor,
                            boxes[index][2] * factor,
                            boxes[index][3] * factor,
                            COLOR_GT if index in matched_boxes else COLOR_FN,
                            "GT" if index in matched_boxes else "FN",
                        )
                entry = {"tp": int(hits.sum()), "fp": int((~hits).sum()), "fn": int(len(missed))}
                totals["tp"] += entry["tp"]
                totals["fp"] += entry["fp"]
                totals["fn"] += entry["fn"]
            else:
                matched_boxes = set()
                if boxes is not None:
                    for index in range(len(boxes)):
                        draw_box(
                            panel,
                            boxes[index][0] * factor,
                            boxes[index][1] * factor,
                            boxes[index][2] * factor,
                            boxes[index][3] * factor,
                            COLOR_FN,
                            "FN",
                        )
                entry = {"tp": 0, "fp": 0, "fn": int(0 if boxes is None else len(boxes))}
                totals["fn"] += entry["fn"]

            for tag in show_scales:
                found = per_tag[tag].get(number)
                if found is None or not len(found[0]):
                    continue
                tag_scores, tag_points, tag_sizes = found
                tag_hits, _ = match_frame(boxes, tag_points, tag_scores)
                for position in range(len(tag_points)):
                    draw_box(
                        panel,
                        tag_points[position][0] * factor,
                        tag_points[position][1] * factor,
                        (tag_sizes[position][0] * factor) or 2.0,
                        (tag_sizes[position][1] * factor) or 2.0,
                        SCALE_COLORS[tag % len(SCALE_COLORS)],
                        f"s{scale_labels[tag]} {tag_scores[position]:.2f}",
                        thickness=1,
                    )

            canvas = np.zeros((args.canvas[1], args.canvas[0], 3), np.uint8)
            panel_x, panel_y = PAD, ROW_HEADER + PAD
            canvas[panel_y : panel_y + panel.shape[0], panel_x : panel_x + panel.shape[1]] = panel
            if zoom:
                if boxes is not None and len(boxes):
                    centre = (float(boxes[0][0]), float(boxes[0][1]))
                    color = COLOR_GT
                elif final is not None and len(final[0]):
                    centre = (float(final[1][0][0]), float(final[1][0][1]))
                    color = COLOR_FP
                else:
                    centre, color = (width / 2, height / 2), (120, 120, 120)
                patch = zoom_panel(gray, centre, args.crop_size, zoom, color)
                zoom_x = args.canvas[0] - PAD - zoom
                zoom_y = ROW_HEADER + PAD
                canvas[zoom_y : zoom_y + zoom, zoom_x : zoom_x + zoom] = patch
                cv2.putText(
                    canvas,
                    f"ZOOM {args.crop_size}px",
                    (zoom_x, zoom_y + zoom + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            stats = entry
            header = (
                f"{args.seq}  frame={number:06d}  th={args.threshold:.2f} r={args.merge_radius:.0f}px  |  "
                f"MERGED  TP {stats['tp']:3d}  FP {stats['fp']:3d}  FN {stats['fn']:3d}"
            )
            cv2.putText(canvas, header, (PAD, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            legend = "GREEN TP | RED FP | WHITE GT(hit) | YELLOW FN"
            for tag in show_scales:
                legend += f" | s{scale_labels[tag]}={SCALE_COLORS[tag % len(SCALE_COLORS)]}"
            cv2.putText(
                canvas,
                legend,
                (PAD, args.canvas[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            records.append({"frame": number, **stats})
            writer.write(canvas)
    finally:
        writer.release()

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(
        f"[DONE] {args.seq} | merged totals TP={totals['tp']} FP={totals['fp']} FN={totals['fn']} | "
        f"Recall={recall * 100:.2f}% Precision={precision * 100:.2f}% F1={f1 * 100:.2f}%",
        flush=True,
    )
    print(f"[SUCCESS] {output.resolve()} | elapsed {time.perf_counter() - started:.1f}s", flush=True)

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "sequence": args.seq,
                        "cache": str(cache_path),
                        "threshold": args.threshold,
                        "merge_radius": args.merge_radius,
                        "frames": [wanted[0], wanted[-1]],
                    },
                    "merged_totals": totals,
                    "merged_metrics": {"recall": recall, "precision": precision, "f1": f1},
                    "per_frame": records,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[SUCCESS] JSON -> {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
