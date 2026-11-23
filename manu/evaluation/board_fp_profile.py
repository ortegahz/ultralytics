#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Profile the FALSE POSITIVES of a fused TSV (board side) or a server reference (server side).

The board-vs-server comparison shows that the board's per-frame FAR is above the red line
while the server's is below it on the same sequence.  The tracker is bit-exact on both sides
(see ``board_crossfeed_verify.py``), so the extra false positives must come from the candidate
pool.  This script splits every detection into a true positive and a false positive with the
**same matcher** the scoring uses (``inbox``: the detection centre falls inside the ground
truth box, greedy score-descending), and then answers the two delivery questions:

  1. **Where do the false positives live?** -- their score distribution and their distance to
     the closest ground truth (a false positive sitting 2 px from a box is a localisation
     problem; one sitting 200 px away is genuine clutter).
  2. **Would a higher detection threshold buy the red line back?** -- for a grid of thresholds
     it reports tp/fp/F1/FAR, so the cheapest threshold that satisfies ``FAR <= --far-limit``
     and the recall it costs is a measured number, not a guess.

Usage
-----
    python manu/evaluation/board_fp_profile.py \
        --board-tsv /mnt/data/siping/runs/trial0080_board24_20261120/tsv/VIDEO00007_19700101_001324_fused.tsv \
        --server-preds /mnt/data/siping/runs/trial0080_onnx/eval/shortlist/preds_cc2_md10/VIDEO00007_19700101_001324.json \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --sequence VIDEO00007_19700101_001324 \
        --out /tmp/crossfeed/fp_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = str(Path(__file__).resolve().parents[2])
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.board_far_parity_check import parse_board_tsv, parse_server_preds  # noqa: E402
from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402


def describe(values: list[float]) -> dict:
    """Compact description of a numeric sample."""
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _match_frame(boxes: np.ndarray | None, ranked: list) -> tuple[list, list]:
    """Greedy score-descending matching of one frame; returns the (tp, fp) detection lists."""
    if boxes is None or not len(boxes):
        return [], list(ranked)
    taken = np.zeros(len(boxes), dtype=bool)
    true_positives: list = []
    false_positives: list = []
    for det in ranked:
        point = np.asarray(det[:2], dtype=np.float64)
        distance = float(np.sqrt(((boxes[:, :2] - point) ** 2).sum(axis=1)).min())
        free = np.nonzero(~taken)[0]
        hit = False
        if len(free):
            nearest = int(free[int(np.argmin(((boxes[free, :2] - point) ** 2).sum(axis=1)))])
            box = boxes[nearest]
            if abs(point[0] - box[0]) <= box[2] / 2 and abs(point[1] - box[1]) <= box[3] / 2:
                taken[nearest] = True
                hit = True
        if hit:
            true_positives.append((det[2], distance))
        else:
            false_positives.append((det[2], distance))
    return true_positives, false_positives


def split_detections(
    dets: dict[int, list], truth: dict[int, np.ndarray]
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split the detections into true and false positives.

    Returns two lists of ``(score, distance_to_the_closest_ground_truth_centre)``.
    """
    true_positives: list[tuple[float, float]] = []
    false_positives: list[tuple[float, float]] = []
    for frame, frame_dets in dets.items():
        ranked = sorted(frame_dets, key=lambda item: -item[2])
        tp, fp = _match_frame(truth.get(frame), ranked)
        true_positives.extend(tp)
        false_positives.extend(fp)
    return true_positives, false_positives


def sweep(
    dets: dict[int, list], truth: dict[int, np.ndarray], frames: int, thresholds: list[float]
) -> list[dict]:
    """tp/fp/F1/FAR for every threshold of the sweep."""
    rows = []
    for threshold in thresholds:
        tp = fp = 0
        for frame, frame_dets in dets.items():
            boxes = truth.get(frame)
            ranked = sorted(
                (det for det in frame_dets if det[2] >= threshold), key=lambda item: -item[2]
            )
            if not ranked:
                continue
            matched, unmatched = _match_frame(boxes, ranked)
            tp += len(matched)
            fp += len(unmatched)
        gt = int(sum(len(value) for value in truth.values()))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt if gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "gt": gt,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "far_per_frame": fp / frames if frames else 0.0,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-tsv", type=Path, default=None)
    parser.add_argument("--server-preds", type=Path, default=None)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=0, help="the frame count of the scoring window")
    parser.add_argument("--far-limit", type=float, default=0.03)
    parser.add_argument(
        "--threshold-grid", default="0.48,0.52,0.56,0.60,0.65,0.70,0.75,0.80,0.90"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    truth = load_ground_truth(args.labels_root, args.sequence, args.width, args.height)
    thresholds = [float(value) for value in args.threshold_grid.split(",")]
    sides = {}
    if args.board_tsv:
        sides["board"] = parse_board_tsv(args.board_tsv)
    if args.server_preds:
        sides["server"] = parse_server_preds(args.server_preds)

    payload = {"sequence": args.sequence, "far_limit": args.far_limit, "sides": {}}
    for name, dets in sides.items():
        frames = args.frames or (max(dets) + 1 if dets else 0)
        true_positives, false_positives = split_detections(dets, truth)
        fp_scores = [item[0] for item in false_positives]
        fp_dist = [item[1] for item in false_positives if np.isfinite(item[1])]
        tp_scores = [item[0] for item in true_positives]
        rows = sweep(dets, truth, frames, thresholds)
        affordable = [row for row in rows if row["far_per_frame"] <= args.far_limit]
        side = {
            "frames": frames,
            "detections": len(true_positives) + len(false_positives),
            "tp": len(true_positives),
            "fp": len(false_positives),
            "fp_score": describe(fp_scores),
            "fp_distance_to_nearest_gt": describe(fp_dist),
            "fp_within_10px_of_a_gt": int(sum(1 for value in fp_dist if value <= 10.0)),
            "tp_score": describe(tp_scores),
            "threshold_sweep": rows,
            "cheapest_threshold_within_far_limit": affordable[0] if affordable else None,
        }
        payload["sides"][name] = side
        print(
            f"[{name}] det={side['detections']} tp={side['tp']} fp={side['fp']} "
            f"fp_score(med)={side['fp_score'].get('median', float('nan')):.4f} "
            f"fp_dist(med)={side['fp_distance_to_nearest_gt'].get('median', float('nan')):.1f} "
            f"fp<=10px_of_gt={side['fp_within_10px_of_a_gt']} "
            f"cheapest_tau_within_far={side['cheapest_threshold_within_far_limit']}"
        )

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"[OUT] {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
