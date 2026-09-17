#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Stage-1a (corrected): TBD detectability by RANDOM SMOOTH TRAJECTORY sampling.

Why this replaces the threshold-then-chain calibration
-----------------------------------------------------
The previous null calibration thresholded candidates at Z>=3.0 and then chained them, which
makes every chain's average Z ~3.3 (the truncation mean) - exactly the target's level, so it
could not separate by construction. This probe integrates the RAW (unthresholded) response,
where background integrates toward 0 while a coherent target stays at its per-frame level.

Question
--------
On pure background, if we sample many smooth, physically plausible trajectories, how high can
their integrated mean response get? If random trajectories routinely reach the target's level
(~3.3), a trajectory search cannot separate the target. If they never do, the search has margin.

Method
------
1. Per frame: matched-filter response Z (DoG + CFAR). Raw values, no thresholding.
2. Sample M smooth trajectories of length L:
       start position uniform, initial velocity uniform in [-vmax, vmax],
       velocity evolves with bounded random acceleration and stays within [-vmax, vmax].
   This matches the real target's slow motion (measured median speed ~1.0 px/frame).
3. Bilinearly sample Z at each sub-pixel trajectory point; trajectory score = mean of Z over L.
4. Build the null distribution of scores and compare with the Oracle reference level
   (the target's own trajectory mean Z, measured at ~3.3 by the Oracle TBD probe).

Read-out
--------
- null max / high quantiles << target level  ->  search separable, TBD viable.
- null routinely exceeds target level        ->  search not separable, Stage 1 stops.
The count of random trajectories whose score reaches the target level is the headline number.

Coordinate contract
-------------------
Native coordinates; matched filter on the native raw channel; bilinear sub-pixel sampling.

Usage on server:
    python manu/probe_tbd_random_trajectory.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --seq wg2022_ir_020_split_05 \
        --output-csv runs/badcase_analysis/tbd_random_trajectory.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List

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
    p = argparse.ArgumentParser(description="TBD detectability via random smooth trajectory sampling")
    p.add_argument("--data-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    p.add_argument("--seq", type=str, default="wg2022_ir_020_split_05", help="Background sequence (GT=0)")
    p.add_argument("--sigma-center", type=float, default=0.6)
    p.add_argument("--sigma-surround", type=float, default=2.0)
    p.add_argument("--sigma-noise", type=float, default=6.0)
    p.add_argument("--windows", type=str, default="5,9,15,25,45,75")
    p.add_argument("--n-traj", type=int, default=100000, help="Random trajectories sampled per window length")
    p.add_argument("--max-entries", type=int, default=20000000,
                   help="Cap on n_traj*L to bound RAM; effective M = min(n_traj, max_entries/L)")
    p.add_argument("--vmax", type=float, default=3.0, help="Max speed (px/frame)")
    p.add_argument("--amax", type=float, default=0.1, help="Max acceleration (px/frame^2)")
    p.add_argument("--margin", type=int, default=14)
    p.add_argument("--target-level", type=float, default=3.3,
                   help="Oracle reference: the target's own trajectory-mean Z (~3.3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", type=str, default="runs/badcase_analysis/tbd_random_trajectory.csv")
    return p.parse_args()


def matched_response(gray: np.ndarray, sc: float, ss: float) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), sc) - cv2.GaussianBlur(gray, (0, 0), ss)


def cfar_zscore(response: np.ndarray, sn: float) -> np.ndarray:
    mean = cv2.GaussianBlur(response, (0, 0), sn)
    mean_sq = cv2.GaussianBlur(response * response, (0, 0), sn)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return (response - mean) / (np.sqrt(var) + 1e-6)


def sample_bilinear(z: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = z.shape
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 2)
    dx = xs - x0
    dy = ys - y0
    return (
        z[y0, x0] * (1 - dx) * (1 - dy)
        + z[y0, x0 + 1] * dx * (1 - dy)
        + z[y0 + 1, x0] * (1 - dx) * dy
        + z[y0 + 1, x0 + 1] * dx * dy
    ).astype(np.float32)


def generate_trajectories(n: int, length: int, n_frames: int, w: int, h: int, args, rng) -> tuple:
    starts = rng.integers(0, max(1, n_frames - length + 1), size=n)
    xs = rng.uniform(args.margin, w - args.margin, size=n).astype(np.float32)
    ys = rng.uniform(args.margin, h - args.margin, size=n).astype(np.float32)
    vx = rng.uniform(-args.vmax, args.vmax, size=n).astype(np.float32)
    vy = rng.uniform(-args.vmax, args.vmax, size=n).astype(np.float32)

    pos = np.empty((n, length, 2), dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    cx, cy = xs.copy(), ys.copy()
    for t in range(length):
        pos[:, t, 0] = cx
        pos[:, t, 1] = cy
        valid &= (cx >= args.margin) & (cx < w - args.margin) & (cy >= args.margin) & (cy < h - args.margin)
        if t < length - 1:
            vx = np.clip(vx + rng.uniform(-args.amax, args.amax, size=n).astype(np.float32), -args.vmax, args.vmax)
            vy = np.clip(vy + rng.uniform(-args.amax, args.amax, size=n).astype(np.float32), -args.vmax, args.vmax)
            cx = cx + vx
            cy = cy + vy
    return starts, pos, valid


def score_trajectories(files: List[Path], starts: np.ndarray, pos: np.ndarray, valid: np.ndarray, args) -> np.ndarray:
    n, length, _ = pos.shape
    frame_of = starts[:, None] + np.arange(length)[None, :]
    flat_frame = frame_of.ravel()
    flat_traj = np.repeat(np.arange(n), length)
    flat_t = np.tile(np.arange(length), n)
    flat_x = pos[:, :, 0].ravel()
    flat_y = pos[:, :, 1].ravel()
    flat_valid = valid[flat_traj]

    order = np.argsort(flat_frame, kind="stable")
    flat_frame_sorted = flat_frame[order]
    samples = np.full(flat_frame.shape[0], np.nan, dtype=np.float32)

    boundaries = np.searchsorted(flat_frame_sorted, np.arange(len(files) + 1))
    for f in tqdm(range(len(files)), desc="Scoring", file=sys.stdout, leave=False):
        lo, hi = boundaries[f], boundaries[f + 1]
        if hi <= lo:
            continue
        idx = order[lo:hi]
        keep = flat_valid[idx]
        if not keep.any():
            continue
        idx = idx[keep]
        img = cv2.imread(str(files[f]), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        gray = (img[:, :, 0] if img.ndim == 3 else img).astype(np.float32)
        z = cfar_zscore(matched_response(gray, args.sigma_center, args.sigma_surround), args.sigma_noise)
        samples[idx] = sample_bilinear(z, flat_x[idx], flat_y[idx])

    samples = samples.reshape(n, length)
    valid_counts = np.sum(np.isfinite(samples), axis=1)
    sums = np.nansum(samples, axis=1)
    scores = np.where(valid_counts > 0, sums / np.maximum(valid_counts, 1), np.nan).astype(np.float32)
    finite = valid & np.isfinite(scores)
    return scores, finite


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
    if not files:
        print(colorstr("red", f"[ERROR] No frames for {args.seq} in {img_dir}"))
        sys.exit(1)

    first = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    h, w = first.shape[:2]
    windows = [int(v) for v in args.windows.split(",") if v.strip()]
    rng = np.random.default_rng(args.seed)

    print(f"[INFO] Background sequence: {args.seq} | frames={len(files)} | size={w}x{h}")
    print(f"[INFO] vmax={args.vmax}px/f, amax={args.amax}px/f^2, M<= {args.n_traj} (RAM-capped by max_entries/L)")
    print(f"[INFO] Oracle reference (target trajectory mean Z) = {args.target_level}")

    rows = []
    for length in windows:
        m = int(min(args.n_traj, max(1000, args.max_entries // max(1, length))))
        print(f"[INFO] L={length}: sampling {m:,} random trajectories")
        starts, pos, valid = generate_trajectories(m, length, len(files), w, h, args, rng)
        scores, finite = score_trajectories(files, starts, pos, valid, args)
        s = scores[finite]
        if s.size == 0:
            continue
        hit = int(np.sum(s >= args.target_level))
        rows.append({
            "seq": args.seq,
            "L": length,
            "n_sampled": int(scores.size),
            "n_valid": int(s.size),
            "valid_frac": float(finite.mean()) if finite.size else 0.0,
            "null_mean": float(np.mean(s)),
            "null_std": float(np.std(s)),
            "null_max": float(np.max(s)),
            "q999": float(np.percentile(s, 99.9)),
            "q9999": float(np.percentile(s, 99.99)),
            "q99999": float(np.percentile(s, 99.999)),
            "n_above_target": hit,
            "frac_above_target": hit / s.size,
            "theory_std": 0.8416 / np.sqrt(length),
        })

    print("\n" + "=" * 132)
    print(colorstr("bold", colorstr("cyan", f"STAGE-1a (CORRECTED): RANDOM SMOOTH TRAJECTORY NULL — {args.seq} (background only)")))
    print("=" * 132)
    print(
        f"{'L':<4} | {'n_valid':<8} | {'valid%':<7} | {'null_mean':<10} | {'null_std':<9} | {'theory_std':<10} | "
        f"{'null_max':<9} | {'q99.9':<8} | {'q99.99':<8} | {'#>=target':<10} | {'frac':<10}"
    )
    print("-" * 132)
    for r in rows:
        print(
            f"{r['L']:<4} | {r['n_valid']:<8} | {r['valid_frac'] * 100:>6.1f}% | {r['null_mean']:>10.4f} | "
            f"{r['null_std']:>9.4f} | {r['theory_std']:>10.4f} | "
            f"{r['null_max']:>9.3f} | {r['q999']:>8.3f} | {r['q9999']:>8.3f} | {r['n_above_target']:>10} | {r['frac_above_target']:>10.2e}"
        )
    print("=" * 132)
    print("null_mean should be ~0 (raw integration, no threshold bias). null_std should track theory_std = 0.84/sqrt(L).")
    print(f"target_level = {args.target_level} (the target's own trajectory mean Z measured by the Oracle probe).")
    print("VERDICT: if null_max << target_level, a trajectory search is separable and TBD is viable.")
    print("         if null_max >= target_level, random trajectories already match the target and Stage 1 stops.")
    print()

    if args.output_csv and rows:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"[INFO] Wrote: {out}")


if __name__ == "__main__":
    main()
