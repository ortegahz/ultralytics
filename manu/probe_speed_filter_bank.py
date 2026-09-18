#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Zero-Training Speed-Filter-Bank Probe (pure CPU, no model, no training, no GPU).

Two evaluation modes:

  snr : sample max_v A_v at GT points (signal) and random background points (noise).
        Reports SNR(W), gain vs W=1, sqrt(W), velocity selectivity and PD@FPR (per-sample).

  roc : build the FULL max_v A_v map, extract 3x3 local-max candidates, and sweep a threshold
        to get a real operating-point curve: PD (per-GT, 8px native) vs FP/frame.
        This is the only mode that answers "can this feature raise recall at the system's
        operational false-alarm rate (~0.045 FP/frame)?".

Speed-filter-bank definition
----------------------------
    A_v(x, y) = sum_{i=0..W-1} Z_{t-i}(x - i*vx, y - i*vy)
where Z is the matched-filter + CFAR normalized response (DoG then local z-score) and Z_{t-i}
is warped into frame t coordinates with a GMC affine estimated DIRECTLY on raw grayscale
(no pairwise composition, to avoid drift accumulation).

No training, no network weights, no GPU. Output is a single small CSV.

Usage on server (single hard case, full ROC at operational FAR):
    python manu/probe_speed_filter_bank.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --seqs wg2022_ir_020_split_03 --mode roc \
        --roc-windows 1,3,5,9 --speeds 0.5,1,2,3,4 --directions 8 \
        --frame-stride 2 --target-far 0.045 \
        --output-csv runs/badcase_analysis/speed_filter_hc1_roc.csv
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import re
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr

from manu.build_s2_median_dataset import FastGMCEstimator

DEFAULT_SEQS = "wg2022_ir_020_split_03,wg2022_ir_011_split_03,DJI_0051_2,02_6321_0274-2773"
DEFAULT_BG_SEQS = "wg2022_ir_020_split_05"


def natural_sort_key(p):
    s = p.stem if isinstance(p, Path) else str(p)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_args():
    p = argparse.ArgumentParser(description="Zero-training speed-filter-bank probe (pure CPU)")
    p.add_argument("--data-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    p.add_argument("--seqs", type=str, default=DEFAULT_SEQS, help="Comma-separated target sequences")
    p.add_argument("--background-seqs", type=str, default=DEFAULT_BG_SEQS, help="Comma-separated GT-free sequences")
    p.add_argument("--all-val", action="store_true", help="Ignore --seqs and sweep every sequence in labels/val")
    p.add_argument("--mode", type=str, default="both", choices=["snr", "roc", "both"])
    p.add_argument("--windows", type=str, default="1,3,5,9,15,21", help="SNR integration window lengths")
    p.add_argument("--roc-windows", type=str, default="1,3,5,9", help="ROC integration window lengths")
    p.add_argument("--speeds", type=str, default="0.5,1,2,3,4", help="Velocity magnitudes (px/frame)")
    p.add_argument("--directions", type=int, default=8, help="Number of velocity directions per speed")
    p.add_argument("--gmc-mode", type=str, default="direct", choices=["direct", "none"], help="GMC estimation mode")
    p.add_argument("--sigma-center", type=float, default=0.6)
    p.add_argument("--sigma-surround", type=float, default=2.0)
    p.add_argument("--sigma-noise", type=float, default=6.0)
    p.add_argument("--background-samples", type=int, default=64, help="Random background samples per frame (snr mode)")
    p.add_argument("--exclude-radius", type=float, default=12.0, help="Min distance of background samples from GT")
    p.add_argument("--margin", type=int, default=12, help="Border margin excluded from sampling/detection")
    p.add_argument("--frame-stride", type=int, default=1, help="Evaluate every Nth frame (history always full)")
    p.add_argument("--max-frames", type=int, default=0, help="Cap frames per sequence (0 = all)")
    p.add_argument("--hit-radius", type=float, default=8.0, help="GT match radius in native pixels")
    p.add_argument("--roc-topk", type=int, default=200, help="Max candidates kept per frame in roc mode")
    p.add_argument("--target-far", type=float, default=0.045, help="Operational false alarms per frame")
    p.add_argument("--fprs", type=str, default="90,99,99.9,99.99", help="Background FPR percentiles for PD@FPR")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-csv", type=str, default="runs/badcase_analysis/speed_filter.csv")
    return p.parse_args()


def read_gt_boxes(lbl_path: Path):
    if not lbl_path.exists():
        return []
    boxes = []
    with lbl_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                boxes.append(tuple(float(v) for v in parts[1:5]))
    return boxes


def matched_response(gray: np.ndarray, sigma_c: float, sigma_s: float) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), sigma_c) - cv2.GaussianBlur(gray, (0, 0), sigma_s)


def cfar_zscore(response: np.ndarray, sigma_n: float) -> np.ndarray:
    mean = cv2.GaussianBlur(response, (0, 0), sigma_n)
    mean_sq = cv2.GaussianBlur(response * response, (0, 0), sigma_n)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return (response - mean) / (np.sqrt(var) + 1e-6)


def build_velocity_bank(speeds, directions):
    vx, vy = [0.0], [0.0]
    for s in speeds:
        for d in range(directions):
            ang = 2.0 * np.pi * d / directions
            vx.append(float(s * np.cos(ang)))
            vy.append(float(s * np.sin(ang)))
    return np.asarray(vx, dtype=np.float32), np.asarray(vy, dtype=np.float32)


def bilinear_vec(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = z.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    valid = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    dx = (x - x0c).astype(np.float32)
    dy = (y - y0c).astype(np.float32)
    out = (
        z[y0c, x0c] * (1 - dx) * (1 - dy)
        + z[y0c, x0c + 1] * dx * (1 - dy)
        + z[y0c + 1, x0c] * (1 - dx) * dy
        + z[y0c + 1, x0c + 1] * dx * dy
    )
    return np.where(valid, out, 0.0).astype(np.float32)


def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        return img[:, :, 0]
    return img


def local_maxima_topk(z: np.ndarray, margin: int, top_k: int):
    h, w = z.shape
    if h <= 2 * margin or w <= 2 * margin:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
    core = z[margin : h - margin, margin : w - margin]
    dil = cv2.dilate(core, np.ones((3, 3), np.uint8))
    mask = core >= dil - 1e-6
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
    scores = core[ys, xs].astype(np.float32)
    if top_k and ys.size > top_k:
        idx = np.argpartition(scores, -top_k)[-top_k:]
        ys, xs, scores = ys[idx], xs[idx], scores[idx]
    coords = np.stack([xs.astype(np.float32) + margin, ys.astype(np.float32) + margin], axis=1)
    return coords, scores


def sample_background(rng, h, w, gt_xy, n, margin, exclude_r):
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 60:
        x = rng.uniform(margin, w - margin)
        y = rng.uniform(margin, h - margin)
        tries += 1
        if gt_xy.size:
            d = np.sqrt(((gt_xy - np.array([x, y], dtype=np.float32)) ** 2).sum(axis=1))
            if float(d.min()) < exclude_r:
                continue
        pts.append((x, y))
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def list_sequence_files(img_dir: Path, seq: str, max_frames: int):
    files = sorted(
        [p for p in img_dir.glob(f"{seq}__*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")],
        key=natural_sort_key,
    )
    if max_frames and len(files) > max_frames:
        files = files[:max_frames]
    return files


def warped_history(z_buffer, raw_buffer, wmax, args, estimator):
    """Return [Z_{t}, Z_{t-1}^w, ..., Z_{t-wmax+1}^w] aligned to frame t coordinates."""
    wz = []
    for i in range(wmax):
        if i == 0 or args.gmc_mode == "none":
            wz.append(z_buffer[-1 - i])
            continue
        h_mat = estimator.compute_affine(raw_buffer[-1 - i], raw_buffer[-1])
        h, w = z_buffer[-1 - i].shape
        wz.append(cv2.warpAffine(z_buffer[-1 - i], h_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    return wz


# --------------------------------------------------------------------------------------------------
# SNR mode
# --------------------------------------------------------------------------------------------------
def evaluate_sequence_snr(seq, img_dir, lbl_dir, args, estimator, vx, vy, windows, fprs):
    files = list_sequence_files(img_dir, seq, args.max_frames)
    if len(files) < 2:
        return None
    wmax = max(windows)
    buffer = deque(maxlen=wmax)
    raw_buffer = deque(maxlen=wmax)
    rng = np.random.default_rng(args.seed + zlib.crc32(seq.encode()) % 10_000)

    signals = {w: [] for w in windows}
    noises = {w: [] for w in windows}
    select = {w: [] for w in windows}
    n_eval = 0
    for idx, path in enumerate(tqdm(files, desc=f"snr {seq[:26]:<26}", file=sys.stdout, leave=False)):
        gray = load_gray(path)
        if gray is None:
            continue
        raw_buffer.append(gray)
        z = cfar_zscore(matched_response(gray.astype(np.float32), args.sigma_center, args.sigma_surround), args.sigma_noise)
        buffer.append(z)
        h, w = gray.shape

        if idx >= wmax - 1 and len(buffer) >= wmax and (idx % args.frame_stride == 0):
            boxes = read_gt_boxes(lbl_dir / f"{path.stem}.txt")
            gt_xy = np.asarray([(b[0] * w, b[1] * h) for b in boxes], dtype=np.float32).reshape(-1, 2)
            if gt_xy.size:
                gt_xy = gt_xy[
                    (gt_xy[:, 0] >= args.margin)
                    & (gt_xy[:, 0] < w - args.margin)
                    & (gt_xy[:, 1] >= args.margin)
                    & (gt_xy[:, 1] < h - args.margin)
                ]
            bg_xy = sample_background(rng, h, w, gt_xy, args.background_samples, args.margin, args.exclude_radius)
            px = np.concatenate([gt_xy[:, 0], bg_xy[:, 0]]) if gt_xy.size else bg_xy[:, 0]
            py = np.concatenate([gt_xy[:, 1], bg_xy[:, 1]]) if gt_xy.size else bg_xy[:, 1]
            if px.size == 0:
                continue
            is_sig = np.zeros(px.shape[0], dtype=bool)
            is_sig[: gt_xy.shape[0]] = True

            wz = warped_history(buffer, raw_buffer, wmax, args, estimator)
            n_pts, n_v = px.shape[0], vx.shape[0]
            samp = np.empty((wmax, n_pts, n_v), dtype=np.float32)
            for i in range(wmax):
                samp[i] = bilinear_vec(wz[i], px[:, None] - i * vx[None, :], py[:, None] - i * vy[None, :])
            a = np.cumsum(samp, axis=0) / np.arange(1, wmax + 1, dtype=np.float32)[:, None, None]
            amax = a.max(axis=2)

            for wlen in windows:
                vals = amax[wlen - 1]
                if is_sig.any():
                    signals[wlen].append(vals[is_sig])
                    a_sig = a[wlen - 1][is_sig]
                    select[wlen].append(a_sig.max(axis=1) / (np.median(a_sig, axis=1) + 1e-6))
                if (~is_sig).any():
                    noises[wlen].append(vals[~is_sig])
            n_eval += 1

    if n_eval == 0:
        return None
    rows = []
    for wlen in windows:
        sig = np.concatenate(signals[wlen]) if signals[wlen] else np.zeros(0, dtype=np.float32)
        noi = np.concatenate(noises[wlen]) if noises[wlen] else np.zeros(0, dtype=np.float32)
        if sig.size == 0 or noi.size == 0:
            continue
        sel = np.concatenate(select[wlen]) if select[wlen] else np.zeros(0, dtype=np.float32)
        row = {
            "seq": seq,
            "W": wlen,
            "sqrt_W": float(np.sqrt(wlen)),
            "n_signal": int(sig.size),
            "n_noise": int(noi.size),
            "signal_mean": float(sig.mean()),
            "noise_mean": float(noi.mean()),
            "noise_std": float(noi.std()),
            "snr": float(sig.mean()) / (float(noi.std()) + 1e-6),
            "selectivity_med": float(np.median(sel)) if sel.size else 0.0,
        }
        for q in fprs:
            thr = float(np.percentile(noi, q))
            row[f"PD@FPR{q}"] = float(np.mean(sig >= thr) * 100.0)
        rows.append(row)
    if rows:
        base = rows[0]["snr"]
        for row in rows:
            row["gain"] = row["snr"] / (base + 1e-6)
    return {"seq": seq, "n_eval_frames": n_eval, "rows": rows}


# --------------------------------------------------------------------------------------------------
# ROC mode (full-map max_v A_v, real PD vs FP/frame)
# --------------------------------------------------------------------------------------------------
def evaluate_sequence_roc(seq, img_dir, lbl_dir, args, estimator, vx, vy, roc_windows):
    files = list_sequence_files(img_dir, seq, args.max_frames)
    if len(files) < 2:
        return None
    wmax = max(roc_windows)
    z_buffer = deque(maxlen=wmax)
    raw_buffer = deque(maxlen=wmax)
    n_v = vx.shape[0]
    stats = {w: {"gt_best": [], "fp": [], "n_frames": 0, "n_gt": 0} for w in roc_windows}
    n_eval = 0

    for idx, path in enumerate(tqdm(files, desc=f"roc {seq[:26]:<26}", file=sys.stdout, leave=False)):
        gray = load_gray(path)
        if gray is None:
            continue
        raw_buffer.append(gray)
        z = cfar_zscore(matched_response(gray.astype(np.float32), args.sigma_center, args.sigma_surround), args.sigma_noise)
        z_buffer.append(z)
        h, w = gray.shape

        if idx >= wmax - 1 and len(z_buffer) >= wmax and (idx % args.frame_stride == 0):
            wz = warped_history(z_buffer, raw_buffer, wmax, args, estimator)
            boxes = read_gt_boxes(lbl_dir / f"{path.stem}.txt")
            gt_xy = np.asarray([(b[0] * w, b[1] * h) for b in boxes], dtype=np.float32).reshape(-1, 2)
            if gt_xy.size:
                gt_xy = gt_xy[
                    (gt_xy[:, 0] >= args.margin)
                    & (gt_xy[:, 0] < w - args.margin)
                    & (gt_xy[:, 1] >= args.margin)
                    & (gt_xy[:, 1] < h - args.margin)
                ]

            acc = [np.zeros((h, w), np.float32) for _ in range(n_v)]
            for i in range(wmax):
                zi = wz[i]
                ii = float(i)
                for vi in range(n_v):
                    tx, ty = ii * vx[vi], ii * vy[vi]
                    if tx == 0.0 and ty == 0.0:
                        acc[vi] += zi
                    else:
                        m = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
                        acc[vi] += cv2.warpAffine(zi, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                if (i + 1) in stats:
                    z_a = acc[0].copy()
                    for vi in range(1, n_v):
                        np.maximum(z_a, acc[vi], out=z_a)
                    cand_xy, cand_score = local_maxima_topk(z_a, args.margin, args.roc_topk)
                    if gt_xy.shape[0] and cand_xy.shape[0]:
                        d = np.linalg.norm(cand_xy[:, None, :] - gt_xy[None, :, :], axis=2)
                        near = d.min(axis=1) <= args.hit_radius
                        for g in range(gt_xy.shape[0]):
                            sel = d[:, g] <= args.hit_radius
                            stats[i + 1]["gt_best"].append(float(cand_score[sel].max()) if sel.any() else -1e9)
                    elif gt_xy.shape[0]:
                        near = np.zeros(cand_xy.shape[0], dtype=bool)
                        stats[i + 1]["gt_best"].extend([-1e9] * gt_xy.shape[0])
                    else:
                        near = np.zeros(cand_xy.shape[0], dtype=bool)
                    if cand_score.size:
                        stats[i + 1]["fp"].append(cand_score[~near])
                    stats[i + 1]["n_frames"] += 1
                    stats[i + 1]["n_gt"] += gt_xy.shape[0]
                    n_eval += 1

    if n_eval == 0:
        return None

    out = {"seq": seq, "rows": [], "roc": []}
    for wlen in roc_windows:
        s = stats[wlen]
        if not s["n_frames"]:
            continue
        gt_best = np.asarray(s["gt_best"], dtype=np.float32) if s["gt_best"] else np.zeros(0, np.float32)
        fp = np.concatenate(s["fp"]) if s["fp"] else np.zeros(0, np.float32)
        nf = s["n_frames"]
        frac = min(1.0, max(0.0, args.target_far * nf / max(1, fp.size)))
        k_star = float(np.percentile(fp, 100.0 * (1.0 - frac))) if fp.size else 1e9
        pd = float(np.mean(gt_best >= k_star) * 100.0) if gt_best.size else 0.0
        actual_fp = float(np.sum(fp >= k_star) / nf) if nf else 0.0
        out["rows"].append(
            {
                "seq": seq,
                "mode": "roc",
                "W": wlen,
                "n_frames": nf,
                "n_gt": int(s["n_gt"]),
                "n_candidates": int(fp.size + gt_best.size),
                "target_far": args.target_far,
                "PD_at_FAR": pd,
                "FP_per_frame_at_FAR": actual_fp,
                "threshold_at_FAR": k_star,
            }
        )
        if fp.size and gt_best.size:
            qs = np.unique(np.percentile(fp, [0, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 99.9, 99.95, 99.99]))
            for k in qs:
                out["roc"].append(
                    {
                        "seq": seq,
                        "W": wlen,
                        "threshold": float(k),
                        "PD": float(np.mean(gt_best >= k) * 100.0),
                        "FP_per_frame": float(np.sum(fp >= k) / nf),
                    }
                )
    return out


def print_snr_table(results, fprs):
    print("\n" + "=" * 118)
    print(colorstr("bold", colorstr("cyan", "SPEED FILTER BANK -- SNR (per-sample)")))
    print("=" * 118)
    print(
        f"{'Sequence':<30} | {'W':<3} | {'signal_mean':<11} | {'noise_std':<9} | {'SNR':<8} | {'gain':<6} | "
        f"{'sqrt(W)':<7} | {'select.':<8} | " + " | ".join(f"{'PD@' + str(q) + '%':<9}" for q in fprs)
    )
    print("-" * 118)
    for res in results:
        for row in res["rows"]:
            pd_cols = " | ".join(f"{row['PD@FPR' + str(q)]:>8.1f}" for q in fprs)
            print(
                f"{row['seq'][:30]:<30} | {row['W']:<3} | {row['signal_mean']:>11.3f} | {row['noise_std']:>9.4f} | "
                f"{row['snr']:>8.2f} | {row['gain']:>6.2f} | {row['sqrt_W']:>7.2f} | {row['selectivity_med']:>8.2f} | {pd_cols}"
            )
        print("-" * 118)


def print_roc_table(results, target_far):
    print("\n" + "=" * 118)
    print(colorstr("bold", colorstr("magenta", f"SPEED FILTER BANK -- FULL-MAP ROC (PD vs FP/frame, target FAR={target_far})")))
    print("=" * 118)
    print(f"{'Sequence':<28} | {'W':<3} | {'frames':<7} | {'GT':<6} | {'thr@FAR':<9} | {'PD@FAR':<8} | {'actual FP/f':<11}")
    print("-" * 118)
    for res in results:
        for row in res["rows"]:
            print(
                f"{row['seq'][:28]:<28} | {row['W']:<3} | {row['n_frames']:<7} | {row['n_gt']:<6} | "
                f"{row['threshold_at_FAR']:>9.3f} | {row['PD_at_FAR']:>7.1f}% | {row['FP_per_frame_at_FAR']:>11.3f}"
            )
    print("-" * 118)
    print(colorstr("bold", "PD vs FP/frame curve (per window):"))
    for res in results:
        for w in sorted({r["W"] for r in res["roc"]}):
            rows = sorted([r for r in res["roc"] if r["W"] == w], key=lambda r: -r["FP_per_frame"])
            curve = "  ".join(f"FP{r['FP_per_frame']:.3f}->PD{r['PD']:.0f}%" for r in rows)
            print(f"  {res['seq'][:28]:<28} W={w:<3} {curve}")
    print("Interpretation: compare PD at the SAME FP/frame across W. A real gain must survive at FP/frame <= target-FAR.")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    img_dir = data_root / "images" / "val"
    lbl_dir = data_root / "labels" / "val"
    if not img_dir.exists():
        alt = Path("/home/manu/mnt/datasets/manu/uav_gmc_median")
        img_dir = alt / "images" / "val"
        lbl_dir = alt / "labels" / "val"
    if not img_dir.exists():
        print(colorstr("red", f"[ERROR] Validation images not found: {img_dir}"))
        sys.exit(1)

    windows = [int(v) for v in args.windows.split(",") if v.strip()]
    roc_windows = sorted({int(v) for v in args.roc_windows.split(",") if v.strip()})
    speeds = [float(v) for v in args.speeds.split(",") if v.strip()]
    fprs = [float(v) for v in args.fprs.split(",") if v.strip()]

    if args.all_val:
        seqs = sorted({p.stem.split("__")[0] for p in img_dir.glob("*__*") if p.suffix.lower() in (".jpg", ".png")})
        seqs = [s for s in seqs if any(read_gt_boxes(p) for p in list(lbl_dir.glob(f"{s}__*.txt"))[:5])]
    else:
        seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
        if args.mode in ("snr", "both"):
            seqs += [s.strip() for s in args.background_seqs.split(",") if s.strip()]

    vx, vy = build_velocity_bank(speeds, args.directions)
    estimator = FastGMCEstimator(downscale=2) if args.gmc_mode == "direct" else None

    print("=" * 118)
    print(colorstr("bold", "SPEED-FILTER-BANK PROBE (zero training, zero model, pure CPU)"))
    print(f"Data root     : {data_root}")
    print(f"Sequences     : {len(seqs)} | mode={args.mode} | gmc={args.gmc_mode}")
    print(f"SNR windows   : {windows} | ROC windows: {roc_windows}")
    print(f"Velocity bank : {vx.shape[0]} hypotheses ({args.directions} dirs x speeds {speeds} + hover)")
    print(f"Matched filter: DoG({args.sigma_center},{args.sigma_surround}) + CFAR({args.sigma_noise})")
    print("=" * 118)

    all_rows = []
    if args.mode in ("snr", "both"):
        results = []
        for seq in seqs:
            res = evaluate_sequence_snr(seq, img_dir, lbl_dir, args, estimator, vx, vy, windows, fprs)
            if res is None or not res["rows"]:
                print(colorstr("yellow", f"[WARN] SNR skipped {seq}: no GT samples in evaluated range"))
                continue
            results.append(res)
            all_rows.extend(res["rows"])
            print(f"[SNR] {seq:<30} {res['rows'][0]['snr']:.2f} -> {res['rows'][-1]['snr']:.2f}")
        if results:
            print_snr_table(results, fprs)

    if args.mode in ("roc", "both"):
        roc_results = []
        for seq in seqs:
            if args.mode == "roc" and seq in [s.strip() for s in args.background_seqs.split(",") if s.strip()]:
                continue
            res = evaluate_sequence_roc(seq, img_dir, lbl_dir, args, estimator, vx, vy, roc_windows)
            if res is None or not res["rows"]:
                print(colorstr("yellow", f"[WARN] ROC skipped {seq}: no frames"))
                continue
            roc_results.append(res)
            all_rows.extend(res["rows"])
        if roc_results:
            print_roc_table(roc_results, args.target_far)

    if args.output_csv and all_rows:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for row in all_rows for k in row.keys()})
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[INFO] Wrote: {out}")


if __name__ == "__main__":
    main()
