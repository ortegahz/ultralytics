"""Per-frame false-positive attribution for the system-SOTA predictions on the Longquan domain.

The business-domain rule is greedy score-descending assignment with pure point-in-bbox matching, and one
ground truth absorbs at most one detection.  This script replays that rule frame by frame on an already
generated ``runs/sys_sota_preds/{sequence}.json`` so that every false positive can be named:

* ``DUP``   - the detection point falls inside a ground-truth box that a better-scoring detection already
  consumed, i.e. the target is covered twice (the "split into several small boxes" failure the reviewers see);
* ``BG``    - the detection point falls inside no ground-truth box at all, i.e. a genuine background
  false positive;
* plus the track id and the ``infilled`` flag, so it is visible whether the box came from the
  high-confidence interpolation stage.

Usage:
    python manu/diagnostics/fp_attribution_sys_sota.py \
        --preds-dir runs/sys_sota_preds \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --seq VIDEO00009_19700101_001531
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402


def load_truth(labels_root: Path, sequence: str, width: int, height: int) -> dict[int, np.ndarray]:
    """Read the ground truth exactly the way the evaluator does (cx, cy, w, h in native pixels)."""
    return load_ground_truth(labels_root, sequence, width, height)


def hits(box: np.ndarray, point: np.ndarray) -> bool:
    """Pure point-in-bbox rule: the detection centre must fall inside the ground-truth box."""
    half_w = box[2] * 0.5
    half_h = box[3] * 0.5
    return bool(abs(point[0] - box[0]) <= half_w and abs(point[1] - box[1]) <= half_h)


def attribute_frame(boxes: np.ndarray, dets: list[dict]) -> tuple[list[dict], int, int]:
    """Replay the greedy assignment and label every detection as TP / FP(DUP) / FP(BG)."""
    ranked = sorted(dets, key=lambda det: -float(det["score"]))
    truth_count = len(boxes)
    taken = np.zeros(truth_count, dtype=bool)
    verdicts: list[dict] = []
    matched = 0
    for det in ranked:
        point = np.asarray(det["pos"], dtype=np.float32)
        verdict = "FP_BG"
        if truth_count:
            free = np.nonzero(~taken)[0]
            if len(free):
                distance = ((boxes[free, :2] - point) ** 2).sum(axis=1)
                nearest = int(free[int(np.argmin(distance))])
                if hits(boxes[nearest], point):
                    taken[nearest] = True
                    matched += 1
                    verdict = "TP"
                elif any(hits(boxes[i], point) for i in range(truth_count)):
                    # inside some box, but every box was already consumed -> a duplicate on a covered target
                    verdict = "FP_DUP"
        verdicts.append(
            {
                "verdict": verdict,
                "pos": [float(point[0]), float(point[1])],
                "score": float(det["score"]),
                "track_id": det.get("track_id"),
                "infilled": bool(det.get("infilled", False)),
            }
        )
    return verdicts, matched, truth_count


def attribute_sequence(payload: dict, truth: dict[int, np.ndarray]) -> dict:
    """Attribute every detection of one sequence to TP / FP(DUP) / FP(BG)."""
    per_frame = payload["per_frame"]
    total_tp = total_dup = total_bg = 0
    infill_fp = 0
    infill_dup = 0
    fp_frames: list[dict] = []
    for key in sorted(per_frame, key=int):
        frame_index = int(key)
        dets = per_frame[key]
        boxes = truth.get(frame_index)
        boxes = (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if boxes is not None
            else np.zeros((0, 4), np.float32)
        )
        verdicts, matched, truth_count = attribute_frame(boxes, dets)
        fps = [v for v in verdicts if v["verdict"].startswith("FP")]
        total_tp += matched
        total_dup += sum(1 for v in fps if v["verdict"] == "FP_DUP")
        total_bg += sum(1 for v in fps if v["verdict"] == "FP_BG")
        infill_fp += sum(1 for v in fps if v["infilled"])
        infill_dup += sum(1 for v in fps if v["infilled"] and v["verdict"] == "FP_DUP")
        if fps:
            fp_frames.append(
                {
                    "frame": frame_index,
                    "n_gt": truth_count,
                    "n_det": len(dets),
                    "n_tp": matched,
                    "gt_centres": [[float(b[0]), float(b[1])] for b in boxes],
                    "fp": fps,
                }
            )
    return {
        "frames": len(per_frame),
        "dets": sum(len(v) for v in per_frame.values()),
        "tp": total_tp,
        "dup": total_dup,
        "bg": total_bg,
        "infill_fp": infill_fp,
        "infill_dup": infill_dup,
        "fp_frames": fp_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", default="runs/sys_sota_preds")
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--seq", default="", help="single sequence; empty means every prediction file in the dir")
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=512)
    parser.add_argument("--show-frames", type=int, default=60, help="how many FP frames to print in detail")
    args = parser.parse_args()

    preds_dir = Path(args.preds_dir)
    labels_root = Path(args.labels_root)

    if args.seq:
        targets = [args.seq]
    else:
        targets = sorted(p.stem for p in preds_dir.glob("*.json") if p.stem != "summary")

    rows = []
    for sequence in targets:
        payload = json.loads((preds_dir / f"{sequence}.json").read_text(encoding="utf-8"))
        truth = load_truth(labels_root, sequence, args.img_w, args.img_h)
        report = attribute_sequence(payload, truth)
        rows.append((sequence, report))

    print("[逐帧 FP 归因] 口径:greedy + point-in-bbox,一个 GT 只吸收一个检出")
    print(f"{'序列':>32} {'帧':>6} {'检出':>6} {'TP':>6} {'FP_DUP':>7} {'FP_BG':>6} {'其中插补':>8} {'FP帧':>5}")
    agg = {"frames": 0, "dets": 0, "tp": 0, "dup": 0, "bg": 0, "infill_fp": 0, "nfpframes": 0}
    for sequence, r in rows:
        agg["frames"] += r["frames"]
        agg["dets"] += r["dets"]
        agg["tp"] += r["tp"]
        agg["dup"] += r["dup"]
        agg["bg"] += r["bg"]
        agg["infill_fp"] += r["infill_fp"]
        agg["nfpframes"] += len(r["fp_frames"])
        print(
            f"{sequence:>32} {r['frames']:>6} {r['dets']:>6} {r['tp']:>6} {r['dup']:>7} {r['bg']:>6} "
            f"{r['infill_fp']:>8} {len(r['fp_frames']):>5}"
        )
    print(
        f"{'合计':>32} {agg['frames']:>6} {agg['dets']:>6} {agg['tp']:>6} {agg['dup']:>7} {agg['bg']:>6} "
        f"{agg['infill_fp']:>8} {agg['nfpframes']:>5}"
    )
    print(
        f"\n  FP_DUP(重复框,目标被框两次) = {agg['dup']}  ({agg['dup'] / max(agg['dup'] + agg['bg'], 1) * 100:.1f}% of FP)"
    )
    print(f"  FP_BG(纯背景误检)             = {agg['bg']}  ({agg['bg'] / max(agg['dup'] + agg['bg'], 1) * 100:.1f}% of FP)")

    if args.seq:
        sequence, r = rows[0]
        print(f"\n  [{sequence}] 有 FP 的帧明细(最多 {args.show_frames} 帧):")
        for item in r["fp_frames"][: args.show_frames]:
            for fp in item["fp"]:
                print(
                    f"    frame={item['frame']:>6} n_gt={item['n_gt']} n_det={item['n_det']} "
                    f"{fp['verdict']:6s} score={fp['score']:.3f} track={fp['track_id']} "
                    f"infilled={int(fp['infilled'])} pos=({fp['pos'][0]:.1f},{fp['pos'][1]:.1f}) "
                    f"gt={[[round(c, 1) for c in g] for g in item['gt_centres']]}"
                )



if __name__ == "__main__":
    main()
