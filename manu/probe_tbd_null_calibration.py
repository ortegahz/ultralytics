#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Stage-1a: TBD False-Track Calibration on PURE BACKGROUND (null calibration).

Purpose
-------
The Oracle probe proved the target's temporal energy is integrable when the trajectory is
known. The open question is the SEARCH: without knowing the trajectory, how many spurious
trajectories does a velocity-gated search invent on data that contains NO target at all?

This script runs a lightweight track-before-detect (DP chaining over matched-filter
response peaks) on clean background and counts false tracks. Every track it returns here
is, by construction, a false alarm.

Method
------
1. Per frame: matched-filter response Z (DoG + CFAR), same detector as the earlier probes.
2. Candidate peaks: 3x3 local maxima with Z >= kmin (low threshold; temporal integration,
   not the single-frame threshold, does the discrimination).
3. Velocity-gated dynamic programming over candidates:
       score[i] = Z[i] + max(0, max over predecessors j in frame t-1 with |p_i-p_j| <= vmax of score[j])
   The target's score grows LINEARLY with track length L (signal ~ 3.3 L), while a random
   noise path grows as sqrt(L), so longer tracks separate better.
4. Greedy extraction with neighbourhood suppression; report tracks with length >= min_length
   and normalised score = score / sqrt(L) above a threshold.

Feasibility read-out
--------------------
The Oracle measured the target's per-frame matched-filter Z at ~3.3, so a true track of
length L should reach normalised score ~3.3*sqrt(L):
    L=9  -> ~9.9 | L=16 -> ~13.2 | L=25 -> ~16.5 | L=49 -> ~23.1
If the null (background) normalised scores sit well below those values, the search has
enough margin and TBD is viable. If background already produces tracks at those levels,
the search is not separable and Stage 1 stops here.

Coordinate contract
-------------------
Native coordinates throughout (matched filter is computed on the native raw channel).

Usage on server:
    python manu/probe_tbd_null_calibration.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --seq wg2022_ir_020_split_05 \
        --output-csv runs/badcase_analysis/tbd_null_calibration.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(p: Path):
    s = p.stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_args():
    p = argparse.ArgumentParser(description="TBD false-track calibration on pure background")
    p.add_argument("--data-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    p.add_argument("--seq", type=str, default="wg2022_ir_020_split_05", help="Sequence to chain (default: GT=0 negative seq)")
    p.add_argument("--sigma-center", type=float, default=0.6)
    p.add_argument("--sigma-surround", type=float, default=2.0)
    p.add_argument("--sigma-noise", type=float, default=6.0)
    p.add_argument("--kmin", type=float, default=3.0, help="Candidate peak threshold on Z")
    p.add_argument("--vmax", type=float, default=3.0, help="Max per-frame displacement (px)")
    p.add_argument("--margin", type=int, default=12)
    p.add_argument("--min-length", type=int, default=5, help="Minimum track length (frames)")
    p.add_argument("--suppress-radius", type=float, default=8.0, help="Neighbourhood suppression radius")
    p.add_argument("--score-thresholds", type=str, default="4,5,6,8,10,12,15,20", help="Normalised score thresholds (score/sqrt(L))")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--output-csv", type=str, default="runs/badcase_analysis/tbd_null_calibration.csv")
    return p.parse_args()


def matched_response(gray: np.ndarray, sc: float, ss: float) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), sc) - cv2.GaussianBlur(gray, (0, 0), ss)


def cfar_zscore(response: np.ndarray, sn: float) -> np.ndarray:
    mean = cv2.GaussianBlur(response, (0, 0), sn)
    mean_sq = cv2.GaussianBlur(response * response, (0, 0), sn)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return (response - mean) / (np.sqrt(var) + 1e-6)


def extract_candidates(z: np.ndarray, kmin: float, margin: int) -> np.ndarray:
    h, w = z.shape
    if h <= 2 * margin or w <= 2 * margin:
        return np.zeros((0, 3), dtype=np.float32)
    core = z[margin : h - margin, margin : w - margin]
    dil = cv2.dilate(core, np.ones((3, 3), np.uint8))
    mask = (core >= dil - 1e-6) & (core >= kmin)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(
        [xs.astype(np.float32) + margin, ys.astype(np.float32) + margin, core[ys, xs].astype(np.float32)],
        axis=1,
    )


def chain_candidates(cands: List[np.ndarray], vmax: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Velocity-gated DP. Returns per-frame cumulative scores and backpointers."""
    scores: List[np.ndarray] = []
    backptr: List[np.ndarray] = []
    for t, c in enumerate(cands):
        n = c.shape[0]
        if n == 0:
            scores.append(np.zeros((0,), dtype=np.float32))
            backptr.append(np.zeros((0,), dtype=np.int32))
            continue
        if t == 0 or cands[t - 1].shape[0] == 0:
            scores.append(c[:, 2].copy())
            backptr.append(np.full((n,), -1, dtype=np.int32))
            continue
        prev = cands[t - 1]
        prev_scores = scores[t - 1]
        d = np.linalg.norm(c[:, None, :2] - prev[None, :, :2], axis=2)
        gated = np.where(d <= vmax, prev_scores[None, :], -np.inf)
        best = gated.max(axis=1)
        best_idx = gated.argmax(axis=1)
        add = np.maximum(best, 0.0)
        score = c[:, 2] + add
        valid = np.isfinite(best) & (add > 0)
        scores.append(score.astype(np.float32))
        backptr.append(np.where(valid, best_idx, -1).astype(np.int32))
    return scores, backptr


def extract_tracks(cands: List[np.ndarray], scores: List[np.ndarray], backptr: List[np.ndarray],
                   min_length: int, suppress_radius: float) -> List[Dict]:
    entries = []
    for t, s in enumerate(scores):
        for i in range(s.shape[0]):
            entries.append((float(s[i]), t, i))
    entries.sort(key=lambda e: -e[0])
    active = [np.ones(s.shape[0], dtype=bool) for s in scores]
    tracks: List[Dict] = []

    for best_val, best_t, best_i in entries:
        if best_val <= 0 or not active[best_t][best_i]:
            continue

        path = []
        t, i = best_t, best_i
        while t >= 0 and i >= 0 and active[t][i]:
            path.append((t, i, float(cands[t][i, 0]), float(cands[t][i, 1]), float(cands[t][i, 2])))
            active[t][i] = False
            # neighbourhood suppression in this frame
            if cands[t].shape[0] > 1:
                d = np.linalg.norm(cands[t][:, :2] - cands[t][i, :2], axis=1)
                active[t][d <= suppress_radius] = False
            j = int(backptr[t][i]) if backptr[t].shape[0] > 0 else -1
            t, i = t - 1, j
        path.reverse()
        if len(path) >= min_length:
            total = float(sum(p[4] for p in path))
            tracks.append({
                "length": len(path),
                "score": total,
                "norm_score": total / np.sqrt(len(path)),
                "start_frame": path[0][0],
                "end_frame": path[-1][0],
            })
    return tracks


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    img_dir = data_root / "images" / "val"
    if not img_dir.exists():
        img_dir = Path("/home/manu/mnt/datasets/manu/uav_gmc_median/images/val")

    files = sorted(
        [p for p in img_dir.glob(f"{args.seq}__*") if p.suffix.lower() in (".jpg", ".png")],
        key=natural_sort_key,
    )
    if args.max_frames:
        files = files[: args.max_frames]
    if not files:
        print(colorstr("red", f"[ERROR] No frames found for {args.seq} in {img_dir}"))
        sys.exit(1)

    print(f"[INFO] Sequence: {args.seq} | frames: {len(files)} | kmin={args.kmin} vmax={args.vmax}")

    cands: List[np.ndarray] = []
    for p in tqdm(files, desc="Detector", file=sys.stdout, leave=False):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            cands.append(np.zeros((0, 3), dtype=np.float32))
            continue
        gray = (img[:, :, 0] if img.ndim == 3 else img).astype(np.float32)
        z = cfar_zscore(matched_response(gray, args.sigma_center, args.sigma_surround), args.sigma_noise)
        cands.append(extract_candidates(z, args.kmin, args.margin))

    counts = np.array([c.shape[0] for c in cands], dtype=np.float32)
    print(f"[INFO] Candidates/frame: mean={counts.mean():.1f} median={np.median(counts):.0f} max={counts.max():.0f} total={int(counts.sum())}")

    scores, backptr = chain_candidates(cands, args.vmax)
    tracks = extract_tracks(cands, scores, backptr, args.min_length, args.suppress_radius)

    thresholds = [float(v) for v in args.score_thresholds.split(",") if v.strip()]
    lengths = np.array([t["length"] for t in tracks], dtype=np.float32) if tracks else np.zeros((0,))
    norms = np.array([t["norm_score"] for t in tracks], dtype=np.float32) if tracks else np.zeros((0,))

    print("\n" + "=" * 120)
    print(colorstr("bold", colorstr("cyan", f"STAGE-1a NULL CALIBRATION — {args.seq} (NO TARGET, every track is a false alarm)")))
    print("=" * 120)
    print(f"• Frames processed            : {len(files)}")
    print(f"• Tracks found (len>={args.min_length})     : {len(tracks)}")
    if len(tracks):
        print(f"• Track length  : min={lengths.min():.0f} median={np.median(lengths):.0f} max={lengths.max():.0f}")
        print(f"• Norm score    : median={np.median(norms):.2f} max={norms.max():.2f}")
    print(f"\n{'NormScore>=':<12} | {'#false tracks':<14} | {'per 1000 frames':<16}")
    print("-" * 60)
    rows = []
    for th in thresholds:
        n = int(np.sum(norms >= th)) if len(norms) else 0
        per1k = n / len(files) * 1000.0
        print(f"{th:<12.1f} | {n:<14d} | {per1k:<16.2f}")
        rows.append({"seq": args.seq, "norm_score_threshold": th, "n_false_tracks": n, "per_1000_frames": per1k})

    print("-" * 60)
    print("Reference: a TRUE target reaches normalised score ~ 3.3*sqrt(L) (Oracle per-frame Z ~ 3.3):")
    for L in (5, 9, 16, 25, 49, 75):
        print(f"   L={L:<3} -> expected true-track norm score ~ {3.3 * np.sqrt(L):.1f}")
    print("Feasibility: false-track counts must stay near zero below those true-track levels.")
    print("If background already yields tracks at ~10-16, the search is NOT separable and Stage 1 stops.")
    print("=" * 120 + "\n")

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[INFO] Wrote: {out}")


if __name__ == "__main__":
    main()
