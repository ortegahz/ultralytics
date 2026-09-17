#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Oracle Track-Before-Detect (TBD) Feasibility Probe.

Question
--------
A human can identify this target only after ~2-3 seconds because they integrate evidence
along the motion trajectory. Does the same temporal information exist for an algorithm?
This probe measures, with NO training and NO model, whether integrating a matched-filter
response ALONG THE KNOWN GROUND-TRUTH TRAJECTORY (sub-pixel) increases the target's SNR.

This is an ORACLE upper bound: the trajectory is taken from GT, so it isolates the
information question ("is the temporal energy coherent and integrable?") from the search
question ("can a trajectory be found without knowing it?").

Method
------
1. Per GT frame: compute the matched-filter response map Z (DoG + CFAR), same as
   probe_matched_filter.py.
2. Sample Z at the sub-pixel GT position (bilinear) -> signal sample s_i.
3. Sample Z at K random background positions (>12px from GT) -> noise samples n_i,k.
4. For each window length N, integrate over N consecutive GT frames (never crossing an
   annotation gap):
       signal_N = mean_i s_i           (coherent: signal survives averaging)
       noise_N  = mean_i n_i,k         (incoherent: std shrinks as sigma/sqrt(N))
       SNR_N    = mean(signal_N) / std(noise_N)   (d-prime)
5. If the target is temporally coherent, SNR_N grows ~sqrt(N). If it flickers or jitters,
   SNR_N plateaus near the single-frame value.

Decision rule
-------------
    SNR_N rises ~sqrt(N) and reaches >6 sigma   -> temporal energy is recoverable -> TBD viable
    SNR_N plateaus near the N=1 value           -> target temporally incoherent -> TBD dead

Coordinate contract
-------------------
Native coordinates everywhere: GT native pixel = (box[0]*W, box[1]*H), sampled with bilinear
interpolation to preserve sub-pixel position. No letterbox, no cache, no mixing of spaces.

Usage on server:
    python manu/probe_oracle_tbd.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --seq wg2022_ir_020_split_03 \
        --output-csv runs/badcase_analysis/oracle_tbd.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr

DEFAULT_CONTROLS = ["wg2022_ir_011_split_03", "DJI_0175_2"]


def natural_sort_key(p: Path):
    s = p.stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_args():
    p = argparse.ArgumentParser(description="Oracle TBD feasibility probe (trajectory-aligned integration)")
    p.add_argument("--data-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    p.add_argument("--seq", type=str, default="wg2022_ir_020_split_03")
    p.add_argument("--control-seqs", type=str, default=",".join(DEFAULT_CONTROLS))
    p.add_argument("--sigma-center", type=float, default=0.6)
    p.add_argument("--sigma-surround", type=float, default=2.0)
    p.add_argument("--sigma-noise", type=float, default=6.0)
    p.add_argument("--windows", type=str, default="1,3,5,9,15,21,45,75")
    p.add_argument("--noise-samples", type=int, default=200, help="Random background samples per frame")
    p.add_argument("--exclude-radius", type=float, default=12.0, help="Min distance of noise samples from GT")
    p.add_argument("--margin", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", type=str, default="runs/badcase_analysis/oracle_tbd.csv")
    return p.parse_args()


def read_first_box(lbl_path: Path):
    if not lbl_path.exists():
        return None
    with lbl_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                return tuple(float(v) for v in parts[1:5])
    return None


def matched_response(gray: np.ndarray, sigma_c: float, sigma_s: float) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), sigma_c) - cv2.GaussianBlur(gray, (0, 0), sigma_s)


def cfar_zscore(response: np.ndarray, sigma_n: float) -> np.ndarray:
    mean = cv2.GaussianBlur(response, (0, 0), sigma_n)
    mean_sq = cv2.GaussianBlur(response * response, (0, 0), sigma_n)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return (response - mean) / (np.sqrt(var) + 1e-6)


def bilinear(z: np.ndarray, x: float, y: float) -> float:
    h, w = z.shape
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 < 0 or y0 < 0 or x0 + 1 >= w or y0 + 1 >= h:
        return 0.0
    dx, dy = x - x0, y - y0
    z00 = z[y0, x0]
    z01 = z[y0, x0 + 1]
    z10 = z[y0 + 1, x0]
    z11 = z[y0 + 1, x0 + 1]
    return float(
        z00 * (1 - dx) * (1 - dy) + z01 * dx * (1 - dy) + z10 * (1 - dx) * dy + z11 * dx * dy
    )


def build_samples(img_dir: Path, lbl_dir: Path, seq: str, args) -> Tuple[List[Tuple[int, float, np.ndarray]], List[int]]:
    files = sorted(
        [p for p in img_dir.glob(f"{seq}__*") if p.suffix.lower() in (".jpg", ".png")],
        key=natural_sort_key,
    )
    rng = np.random.default_rng(args.seed)
    records: List[Tuple[int, float, np.ndarray]] = []
    gap_total = 0
    prev_idx = None

    for idx, p in enumerate(tqdm(files, desc=f"{seq[:26]:<26}", file=sys.stdout, leave=False)):
        box = read_first_box(lbl_dir / f"{p.stem}.txt")
        if box is None:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        gray = (img[:, :, 0] if img.ndim == 3 else img).astype(np.float32)
        h, w = gray.shape[:2]
        x, y = box[0] * w, box[1] * h
        if not (args.margin <= x < w - args.margin and args.margin <= y < h - args.margin):
            continue

        z = cfar_zscore(matched_response(gray, args.sigma_center, args.sigma_surround), args.sigma_noise)
        z_gt = bilinear(z, x, y)

        noise = np.empty(args.noise_samples, dtype=np.float32)
        k = 0
        while k < args.noise_samples:
            nx = rng.uniform(args.margin, w - args.margin)
            ny = rng.uniform(args.margin, h - args.margin)
            if (nx - x) ** 2 + (ny - y) ** 2 < args.exclude_radius ** 2:
                continue
            noise[k] = bilinear(z, nx, ny)
            k += 1

        records.append((idx, z_gt, noise))
        if prev_idx is not None and idx - prev_idx > 1:
            gap_total += 1
        prev_idx = idx

    return records, [gap_total]


def make_segments(records):
    segments = []
    current = []
    prev_idx = None
    for rec in records:
        if prev_idx is not None and rec[0] - prev_idx > 1:
            if current:
                segments.append(current)
            current = []
        current.append(rec)
        prev_idx = rec[0]
    if current:
        segments.append(current)
    return segments


def analyze_sequence(records, windows: List[int]):
    segments = make_segments(records)
    rows = []
    for N in windows:
        signals, noises = [], []
        for seg in segments:
            if len(seg) < N:
                continue
            for start in range(0, len(seg) - N + 1):
                chunk = seg[start : start + N]
                signals.append(np.mean([c[1] for c in chunk]))
                noises.append(np.mean(np.stack([c[2] for c in chunk]), axis=0))
        if not signals or not noises:
            continue
        signals = np.asarray(signals, dtype=np.float32)
        noise_means = np.concatenate(noises).astype(np.float32)
        noise_std = float(np.std(noise_means)) + 1e-6
        snr = float(np.mean(signals)) / noise_std
        threshold = float(np.mean(noise_means) + 3.0 * noise_std)
        rows.append({
            "N": N,
            "n_windows": len(signals),
            "signal_mean": float(np.mean(signals)),
            "noise_std": noise_std,
            "snr": snr,
            "pd_3sigma": float(np.mean(signals >= threshold) * 100),
        })
    return segments, rows


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    img_dir = data_root / "images" / "val"
    lbl_dir = data_root / "labels" / "val"
    if not img_dir.exists():
        alt = Path("/home/manu/mnt/datasets/manu/uav_gmc_median")
        img_dir = alt / "images" / "val"
        lbl_dir = alt / "labels" / "val"

    windows = [int(v) for v in args.windows.split(",") if v.strip()]
    seqs = [args.seq] + [s.strip() for s in args.control_seqs.split(",") if s.strip()]

    all_rows = []
    summary = []
    for seq in seqs:
        records, gaps = build_samples(img_dir, lbl_dir, seq, args)
        if not records:
            print(colorstr("yellow", f"[WARN] No GT samples for {seq}"))
            continue
        segments, rows = analyze_sequence(records, windows)
        lengths = [len(s) for s in segments]
        summary.append((seq, len(records), len(segments), lengths))
        for r in rows:
            r["seq"] = seq
            all_rows.append(r)

    if args.output_csv and all_rows:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[INFO] Wrote: {out}")

    for seq, n, nseg, lengths in summary:
        print(f"[INFO] {seq}: {n} GT frames, {nseg} track segment(s), lengths={lengths[:8]}")

    print("\n" + "=" * 120)
    print(colorstr("bold", colorstr("cyan", "ORACLE TBD: SNR vs INTEGRATION LENGTH (trajectory-aligned, sub-pixel)")))
    print("=" * 120)
    print(f"{'Sequence':<28} | {'N':<4} | {'windows':<8} | {'signal_mean':<12} | {'noise_std':<10} | {'SNR(d-prime)':<13} | {'gain':<7} | {'sqrt(N)':<8} | {'PD@3sig':<8}")
    print("-" * 120)
    for seq in seqs:
        rows = [r for r in all_rows if r["seq"] == seq]
        if not rows:
            continue
        base = rows[0]["snr"] if rows and rows[0]["N"] == 1 else None
        for r in rows:
            gain = (r["snr"] / base) if base else float("nan")
            print(
                f"{seq[:28]:<28} | {r['N']:<4} | {r['n_windows']:<8} | {r['signal_mean']:>12.3f} | "
                f"{r['noise_std']:>10.4f} | {r['snr']:>13.2f} | {gain:>7.2f} | {np.sqrt(r['N']):>8.2f} | {r['pd_3sigma']:>7.1f}%"
            )
        print("-" * 120)

    print("signal_mean = mean matched-filter Z at GT | noise_std = std of window-mean background Z")
    print("SNR = signal_mean / noise_std (d-prime).  gain = SNR(N)/SNR(1).  Coherent target -> gain tracks sqrt(N).")
    print("If gain stays ~1.0 the target is temporally incoherent and even an oracle trajectory cannot integrate it.")
    print()


if __name__ == "__main__":
    main()
