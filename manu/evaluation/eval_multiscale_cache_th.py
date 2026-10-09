#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Threshold sweep and per-sequence report for the multi-scale heatmap cache against Final_Labels ground truth."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import random
import time

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def iter_frames(cache: dict):
    """Walk one cached sequence frame by frame; mirrors multiscale_heatmap_cache.iter_frames.

    Defined here rather than imported so that threshold search never has to import torch.
    """
    offset = cache["offset"]
    for position, frame_index in enumerate(cache["frame_index"]):
        start, end = offset[position], offset[position + 1]
        yield (
            int(frame_index),
            cache["points"][start:end],
            cache["scores"][start:end],
            cache["sizes"][start:end],
            cache["sums"][start:end],
            cache["area"][start:end],
            cache["tag"][start:end],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the score threshold of the multi-scale cache against Final_Labels"
    )
    parser.add_argument("--cache-root", required=True, help="Directory of <sequence>.pkl written by the cache script")
    parser.add_argument(
        "--labels-root", required=True, help="Flat Final_Labels directory of {sequence}__{index:06d}.txt"
    )
    parser.add_argument("--tag", default="255", help="255 = fused row, 'all' = every row, or one scale index")
    parser.add_argument(
        "--match-mode",
        default="dist",
        choices=["dist", "inbox"],
        help="dist = centres within --tol pixels; inbox = predicted centre inside the ground-truth box",
    )
    parser.add_argument("--tol", type=float, default=8.0, help="Matching tolerance in native pixels, dist mode only")
    parser.add_argument("--th-min", type=float, default=0.0)
    parser.add_argument("--th-max", type=float, default=0.90)
    parser.add_argument("--th-step", type=float, default=0.01)
    parser.add_argument("--merge-radius", type=float, default=8.0, help="Cross-scale dedup radius in native pixels")
    parser.add_argument(
        "--sweep-merge-radius",
        default="",
        help="Comma-separated radii to search instead of a single --merge-radius",
    )
    parser.add_argument("--top-disagree", type=int, default=8, help="Rows printed in the disagreement ranking")
    parser.add_argument("--out-json", default="", help="Optional JSON report path")
    return parser.parse_args()


def merge_scales(per_scale: dict[int, tuple], radius: float) -> dict[int, tuple]:
    """Union several scale rows per frame, keeping the highest score inside each merge radius."""
    if radius <= 0:
        merged = {}
        for items in per_scale.values():
            merged.update(items)
        return merged
    radius_squared = radius * radius
    merged = {}
    for frame_index in set().union(*(set(items) for items in per_scale.values())):
        pooled = []
        for items in per_scale.values():
            if frame_index in items:
                scores, points = items[frame_index]
                pooled.extend(zip(scores.tolist(), points.tolist()))
        pooled.sort(key=lambda item: -item[0])
        kept_scores, kept_points, kept_xy = [], [], []
        for score, (x, y) in pooled:
            if any((x - px) ** 2 + (y - py) ** 2 <= radius_squared for px, py in kept_xy):
                continue
            kept_scores.append(score)
            kept_points.append((x, y))
            kept_xy.append((x, y))
        if kept_scores:
            merged[frame_index] = (
                np.asarray(kept_scores, dtype=np.float32),
                np.asarray(kept_points, dtype=np.float32),
            )
    return merged


def load_ground_truth(labels_root: Path, sequence: str, width: int, height: int) -> dict[int, np.ndarray]:
    """Read {sequence}__{index:06d}.txt into per-frame centres, dropping exact duplicate rows."""
    prefix = f"{sequence}__"
    truth: dict[int, np.ndarray] = {}
    with os.scandir(labels_root) as entries:
        for entry in entries:
            name = entry.name
            if not name.startswith(prefix) or not name.endswith(".txt"):
                continue
            digits = name[len(prefix) : -4]
            if not digits.isdigit() or entry.stat().st_size == 0:
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
                # cx, cy, w, h in native pixels; w/h are needed by the point-in-bbox matching rule
                truth[int(digits)] = np.asarray(rows, dtype=np.float32) * np.float32([width, height, width, height])
    return truth


def load_predictions(
    cache_root: Path, sequence: str, tags: list[int], floor: float
) -> tuple[dict, dict[int, dict[int, tuple]]]:
    """Return cache meta plus, per requested tag, the per-frame (scores descending, centres)."""
    cache = pickle.loads((cache_root / f"{sequence}.pkl").read_bytes())
    by_tag: dict[int, dict[int, tuple]] = {tag: {} for tag in tags}
    for frame_index, points, scores, _, _, _, row_tag in iter_frames(cache):
        rows = np.isin(row_tag, tags)
        if not rows.any():
            continue
        selected = np.nonzero(rows & (scores.astype(np.float32) >= floor))[0]
        if not len(selected):
            continue
        order = selected[np.argsort(-scores[selected], kind="stable")]
        for tag in np.unique(row_tag[order]):
            subset = order[row_tag[order] == tag]
            by_tag[int(tag)][int(frame_index)] = (
                scores[subset].astype(np.float32),
                points[subset],
            )
    return cache["meta"], by_tag


def build_frames(truth: dict[int, np.ndarray], predictions: dict[int, tuple]) -> list[tuple]:
    """One entry per frame over the union of ground truth and predictions, so GT-only frames cannot be skipped."""
    merged = []
    for frame_index in set(truth) | set(predictions):
        centres = truth.get(frame_index)
        found = predictions.get(frame_index)
        merged.append((centres, found[1] if found else None, found[0] if found else None))
    return merged


def _hits(box: np.ndarray, point: np.ndarray, mode: str, tol_squared: float) -> bool:
    """Does one prediction satisfy the matching rule against one ground-truth box?"""
    if mode == "inbox":
        return abs(point[0] - box[0]) <= box[2] / 2 and abs(point[1] - box[1]) <= box[3] / 2
    distance = (box[0] - point[0]) ** 2 + (box[1] - point[1]) ** 2
    return distance <= tol_squared


COUNT_KEYS = ("tp", "fp", "fn", "fn_absent", "fn_compete", "det")


def count_matches(frames: list[tuple], threshold: float, tol: float, mode: str) -> dict:
    """Greedy score-descending matching against the closest still-unmatched ground truth.

    ``dist`` accepts a hit when the centres are within ``tol`` pixels; ``inbox`` accepts a hit when the
    predicted centre falls inside the ground-truth box. A ground truth absorbs at most one detection.

    False negatives are split into ``fn_absent`` (nothing was ever detected near the box) and ``fn_compete``
    (a detection did qualify but a better-scoring one consumed it). That split is what tells a useful merge
    radius from an over-merged one, because a growing ``fn_compete`` means real targets are being swallowed.
    """
    counts = dict.fromkeys(COUNT_KEYS, 0)
    tol_squared = tol * tol
    for boxes, points, scores in frames:
        detected = 0 if scores is None else int((scores >= threshold).sum())
        truth_count = 0 if boxes is None else len(boxes)
        if detected == 0:
            counts["fn_absent"] += truth_count
            continue
        points = points[scores >= threshold]
        counts["det"] += detected
        taken = np.zeros(truth_count, dtype=bool)
        matched = 0
        for position in range(detected):
            if not truth_count:
                break
            free = np.nonzero(~taken)[0]
            if not len(free):
                break
            distance = ((boxes[free, :2] - points[position]) ** 2).sum(axis=1)
            nearest = int(np.argmin(distance))
            box = free[nearest]
            if _hits(boxes[box], points[position], mode, tol_squared):
                taken[box] = True
                matched += 1
        counts["tp"] += matched
        counts["fp"] += detected - matched
        for box in range(truth_count):
            if taken[box]:
                continue
            if any(_hits(boxes[box], points[position], mode, tol_squared) for position in range(detected)):
                counts["fn_compete"] += 1
            else:
                counts["fn_absent"] += 1
    counts["fn"] = counts["fn_absent"] + counts["fn_compete"]
    return counts


def metrics(counts: dict, frames: int) -> dict:
    true_positives, false_positives = counts["tp"], counts["fp"]
    false_negatives = counts["fn"]
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "fn_absent": counts["fn_absent"],
        "fn_compete": counts["fn_compete"],
        "det": counts["det"],
        "gt": true_positives + false_negatives,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "far_per_frame": false_positives / frames if frames else 0.0,
    }


def resolve_sources(by_tag: dict[int, dict[int, tuple]], mode: str, fused_tag: int, scale_tags: list[int]):
    """Pick which cached rows feed the merge: the baked fused row, one scale, or every scale."""
    if mode == "255" and fused_tag in by_tag:
        return [by_tag[fused_tag]]
    if mode != "all":
        return [by_tag[int(mode)]]
    return [by_tag[tag] for tag in scale_tags if tag in by_tag]


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    if not cache_root.is_dir():
        raise NotADirectoryError(cache_root)
    if not labels_root.is_dir():
        raise NotADirectoryError(labels_root)

    thresholds = np.round(np.arange(args.th_min, args.th_max + 1e-9, args.th_step), 6)
    radii = (
        sorted({float(item) for item in args.sweep_merge_radius.split(",") if item.strip()})
        if args.sweep_merge_radius
        else [args.merge_radius]
    )
    sequences = sorted(path.stem for path in cache_root.glob("*.pkl"))
    if not sequences:
        raise FileNotFoundError(f"No .pkl caches under {cache_root}")

    meta_sample = None
    loaded, started = {}, time.perf_counter()
    for position, sequence in enumerate(sequences, 1):
        cache = pickle.loads((cache_root / f"{sequence}.pkl").read_bytes())
        meta = cache["meta"]
        meta_sample = meta
        tags = sorted(int(value) for value in np.unique(cache["tag"]))
        selected = tags if args.tag == "all" else [int(args.tag)] if int(args.tag) in tags else tags
        _, by_tag = load_predictions(cache_root, sequence, selected, args.th_min)
        truth = load_ground_truth(labels_root, sequence, meta["native_size"][0], meta["native_size"][1])
        loaded[sequence] = (by_tag, truth, meta)
        if position <= 3 or position == len(sequences):
            counts = {tag: sum(len(item[0]) for item in by_tag[tag].values()) for tag in selected}
            print(
                f"[{position:2d}/{len(sequences)}] {sequence} | frames={meta['frames']} | "
                f"gt_boxes={sum(len(v) for v in truth.values())} | rows={counts}",
                flush=True,
            )

    fused_tag = meta_sample.get("fused_tag", 255)
    scale_tags = [tag for tag in (range(len(meta_sample["scales"]))) if tag in loaded[sequences[0]][0]]

    mode_note = f"tol={args.tol:.1f}px" if args.match_mode == "dist" else "tol ignored"
    print(
        f"\n[Sweep] tags={args.tag} mode={args.match_mode} ({mode_note}) | "
        f"merge radii={radii} | thresholds={len(thresholds)}",
        flush=True,
    )
    print(
        f"{'radius':>7} {'th':>6} {'TP':>7} {'FP':>7} {'FN':>7} {'Recall':>9} {'Prec':>9} {'F1':>9} "
        f"{'FAR/frame':>10} {'FN-nodet':>9} {'FN-compete':>11}"
    )
    sweep = []
    for radius in radii:
        merged_cache = {}
        for sequence, (by_tag, truth, meta) in loaded.items():
            frames = merge_scales(dict(enumerate(resolve_sources(by_tag, args.tag, fused_tag, scale_tags))), radius)
            merged_cache[sequence] = (build_frames(truth, frames), meta)

        for threshold in thresholds:
            totals = dict.fromkeys(COUNT_KEYS, 0)
            frame_total = 0
            for frames, meta in merged_cache.values():
                for key, value in count_matches(frames, float(threshold), args.tol, args.match_mode).items():
                    totals[key] += value
                frame_total += meta["frames"]
            row = metrics(totals, frame_total)
            row["th"] = float(threshold)
            row["merge_radius"] = float(radius)
            sweep.append(row)
        best_here = max((row for row in sweep if row["merge_radius"] == radius), key=lambda r: r["f1"])
        print(
            f"{radius:7.1f} {best_here['th']:6.2f} {best_here['tp']:7d} {best_here['fp']:7d} {best_here['fn']:7d} "
            f"{best_here['recall'] * 100:8.2f}% {best_here['precision'] * 100:8.2f}% {best_here['f1'] * 100:8.3f}% "
            f"{best_here['far_per_frame']:10.4f} {best_here['fn_absent']:9d} {best_here['fn_compete']:11d}",
            flush=True,
        )

    best = max(sweep, key=lambda row: (row["f1"], row["precision"]))
    print(
        f"\n[BEST GLOBAL] merge_radius={best['merge_radius']:.1f}px th={best['th']:.2f} | F1={best['f1'] * 100:.3f}% | "
        f"Recall={best['recall'] * 100:.2f}% | Precision={best['precision'] * 100:.2f}% | "
        f"TP={best['tp']} FP={best['fp']} FN={best['fn']} (no-detection={best['fn_absent']} "
        f"stolen={best['fn_compete']}) | FAR={best['far_per_frame']:.4f}/frame",
        flush=True,
    )
    print(
        "[NOTE] FN splits into fn_absent (nothing detected anywhere near the box; a frontend limit) and "
        "fn_compete\n       (a detection qualified but a higher-scoring one consumed it; a merge or duplicate "
        "limit)."
    )

    merged_cache = {}
    for sequence, (by_tag, truth, meta) in loaded.items():
        frames = merge_scales(
            dict(enumerate(resolve_sources(by_tag, args.tag, fused_tag, scale_tags))), best["merge_radius"]
        )
        merged_cache[sequence] = (build_frames(truth, frames), meta)

    per_sequence = []
    for sequence, (frames, meta) in merged_cache.items():
        counts = count_matches(frames, best["th"], args.tol, args.match_mode)
        row = metrics(counts, meta["frames"])
        row["sequence"] = sequence
        row["frames"] = meta["frames"]
        row["gt_frames"] = sum(1 for boxes, _, _ in frames if boxes is not None)
        curve = [
            metrics(count_matches(frames, float(t), args.tol, args.match_mode), meta["frames"]) for t in thresholds
        ]
        best_local = max(curve, key=lambda item: item["f1"])
        row["best_seq_th"] = float(thresholds[curve.index(best_local)])
        row["best_seq_f1"] = best_local["f1"]
        row["errors"] = counts["fp"] + counts["fn"]
        per_sequence.append(row)

    per_sequence.sort(key=lambda item: -item["errors"])
    print(
        f"\n[PER-SEQUENCE @ radius={best['merge_radius']:.1f}px th={best['th']:.2f}] sorted by disagreement (FP+FN) descending"
    )
    print(
        f"{'sequence':30s} {'frames':>7} {'GT':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'FN-abs':>7} {'FN-cmp':>7} "
        f"{'Recall':>8} {'Prec':>8} {'F1':>8} {'dF1':>7} {'FP+FN':>7} {'seqBestTh':>9} {'seqBestF1':>9}"
    )
    for row in per_sequence:
        print(
            f"{row['sequence']:30s} {row['frames']:7d} {row['gt']:6d} {row['tp']:6d} {row['fp']:6d} {row['fn']:6d} "
            f"{row['fn_absent']:7d} {row['fn_compete']:7d} "
            f"{row['recall'] * 100:7.2f}% {row['precision'] * 100:7.2f}% {row['f1'] * 100:7.3f}% "
            f"{(row['f1'] - best['f1']) * 100:+6.2f} {row['errors']:7d} {row['best_seq_th']:9.2f} "
            f"{row['best_seq_f1'] * 100:8.3f}%",
            flush=True,
        )

    print(
        f"\n[TOP DISAGREEMENTS] the {args.top_disagree} sequences with the most FP+FN at radius={best['merge_radius']:.1f}px th={best['th']:.2f}"
    )
    for row in per_sequence[: args.top_disagree]:
        print(
            f"  {row['sequence']:30s} FP+FN={row['errors']:6d} (FP={row['fp']:6d} FN={row['fn']:6d}) | "
            f"F1={row['f1'] * 100:6.2f}% vs global {best['f1'] * 100:.2f}% ({(row['f1'] - best['f1']) * 100:+.2f}) | "
            f"its own best th={row['best_seq_th']:.2f} -> F1 {row['best_seq_f1'] * 100:.2f}%",
            flush=True,
        )

    print("\n[GT BOX SIZE] 2% random sample of every sequence's ground-truth files")
    short_edges = []
    for sequence in sequences:
        width, height = loaded[sequence][2]["native_size"]
        with os.scandir(labels_root) as entries:
            for entry in entries:
                name = entry.name
                if not name.startswith(f"{sequence}__") or not name.endswith(".txt"):
                    continue
                if random.random() > 0.02 or entry.stat().st_size == 0:
                    continue
                for line in Path(entry.path).read_text(encoding="utf-8").splitlines():
                    values = line.split()
                    if len(values) >= 5:
                        short_edges.append(min(float(values[3]) * width, float(values[4]) * height))
    if short_edges:
        edges = np.asarray(short_edges, dtype=np.float32)
        print(
            f"  n={len(edges)} | shortest edge p10={np.percentile(edges, 10):.0f}px "
            f"p50={np.median(edges):.0f}px p90={np.percentile(edges, 90):.0f}px"
        )
        if args.match_mode == "dist":
            print(
                f"  {args.tol:.0f}px tolerance = {args.tol / np.median(edges) * 100:.0f}% of the median shortest edge, "
                f"{args.tol / np.percentile(edges, 10) * 100:.0f}% of the p10 shortest edge"
            )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "config": {
                        "cache_root": str(cache_root),
                        "labels_root": str(labels_root),
                        "tag": args.tag,
                        "merge_radius": best["merge_radius"],
                        "tol_px": args.tol,
                        "match_mode": args.match_mode,
                        "th_min": args.th_min,
                        "th_max": args.th_max,
                        "th_step": args.th_step,
                    },
                    "best_global": best,
                    "sweep": sweep,
                    "per_sequence": per_sequence,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n[SUCCESS] JSON -> {Path(args.out_json).resolve()}", flush=True)
    print(f"[SUCCESS] elapsed {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
