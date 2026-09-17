#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Deep Forensic Pathology Probe for a single UAV sequence (default: wg2022_ir_020_split_03).

Coordinate contract (CRITICAL)
------------------------------
The dataset holds NATIVE images of mixed resolution (512x512 or 640x512), never 640x640.
YOLO labels are normalized against the NATIVE image, so native pixel coords are
`(box[0]*W, box[1]*H)`. The inference cache stores `gt_pts` and `pred_points` in the
640x640 LETTERBOXED model-input space. The two spaces differ by an aspect-preserving
scale plus padding, so they must never be mixed. This probe:
  - samples image channels at NATIVE coordinates,
  - measures model response distance against the cache `gt_pts` (letterboxed), pairing
    boxes by index within each frame.

Polarity
--------
Targets may be bright (IR) or dark (visible light). `max` over the target patch only
finds bright targets, so a dark target yields a near-zero statistic and would be
misranked as noise. This probe reports both signed and absolute (polarity-agnostic)
contrast / MAD-Z.

Investigates
------------
1. Target geometry, scale distribution and trajectory kinematics.
2. Three-channel input signal at the true target centroid (raw / GMC diff / median).
3. Heatmap activation: what did the network actually predict around GT?
4. Temporal continuity, annotation structure and an oracle temporal integration bound.

Usage on server:
    python manu/probe_wg020_03_deep.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --output-csv runs/badcase_analysis/wg020_03_signal_profile.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import re
import sys
from typing import Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Forensic Pathology Probe")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to dataset root containing images/val and labels/val",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 inference cache (letterboxed 640x640 coordinate space)",
    )
    parser.add_argument("--seq", type=str, default="wg2022_ir_020_split_03")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="runs/badcase_analysis/wg020_03_signal_profile.csv",
        help="Output path for per-GT-frame measurements",
    )
    return parser.parse_args()


def patch_metrics(channel: np.ndarray, cx: int, cy: int) -> Dict[str, float]:
    """Robust target/background statistics at native (cx, cy). Handles bright and dark targets."""
    h, w = channel.shape[:2]
    empty = {
        "peak": 0.0, "trough": 0.0, "mean5": 0.0, "bg_mean": 0.0, "bg_std": 0.0,
        "bg_median": 0.0, "signed": 0.0, "mad_z": 0.0, "abs_mad_z": 0.0, "contrast": 0.0,
    }
    if not (10 <= cx < w - 10 and 10 <= cy < h - 10):
        return empty
    target = channel[cy - 2 : cy + 3, cx - 2 : cx + 3].astype(np.float32)
    outer = channel[cy - 10 : cy + 11, cx - 10 : cx + 11].astype(np.float32)
    mask = np.ones((21, 21), dtype=bool)
    mask[5:16, 5:16] = False
    background = outer[mask]
    bg_median = float(np.median(background))
    bg_mad = float(np.median(np.abs(background - bg_median)))
    bg_mean = float(np.mean(background))
    bg_std = float(np.std(background))
    peak = float(np.max(target))
    trough = float(np.min(target))
    robust_sigma = 1.4826 * bg_mad + 1e-4
    signed = peak - bg_median if (peak - bg_median) >= (bg_median - trough) else trough - bg_median
    return {
        "peak": peak,
        "trough": trough,
        "mean5": float(np.mean(target)),
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "bg_median": bg_median,
        "signed": signed,
        "mad_z": signed / robust_sigma,
        "abs_mad_z": abs(signed) / robust_sigma,
        "contrast": signed / (bg_std + 1e-4),
    }


def channel_change(channel: np.ndarray, cx: int, cy: int, radius: int = 10) -> float:
    h, w = channel.shape[:2]
    if not (radius <= cx < w - radius and radius <= cy < h - radius):
        return 0.0
    patch = channel[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1].astype(np.float32)
    return float(np.std(patch))


def percentile_summary(values) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def longest_runs(flags: List[bool]) -> List[int]:
    runs, current = [], 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return sorted(runs, reverse=True)


def load_labels(lbl_path: Path):
    boxes = []
    if lbl_path.exists():
        with lbl_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    boxes.append([float(v) for v in parts[1:5]])
    return boxes


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    val_img_dir = data_root / "images" / "val"
    val_lbl_dir = data_root / "labels" / "val"

    if not val_img_dir.exists():
        alt_root = Path("/home/manu/mnt/datasets/manu/uav_gmc_median")
        if (alt_root / "images" / "val").exists():
            val_img_dir = alt_root / "images" / "val"
            val_lbl_dir = alt_root / "labels" / "val"
        else:
            print(colorstr("red", f"[ERROR] Validation images not found at: {val_img_dir}"))
            sys.exit(1)

    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        for cand in [
            PROJECT_ROOT / cache_path,
            Path("/tmp/pycharm_project_10ae9e2e") / cache_path,
            Path("/home/manu/mnt/pycharm_project_10ae9e2e") / cache_path,
        ]:
            if cand.exists():
                cache_path = cand
                break

    cache_records: Dict[str, dict] = {}
    if cache_path.exists():
        print(f"[INFO] Loading inference cache from: {cache_path}")
        with open(cache_path, "rb") as f:
            for r in pickle.load(f):
                if args.seq in r["im_name"]:
                    cache_records[Path(r["im_name"]).name] = r
    else:
        print(colorstr("yellow", f"[WARN] Cache file not found: {cache_path}. Proceeding without cache."))

    img_files = sorted(
        [p for p in val_img_dir.glob(f"*{args.seq}*") if p.suffix.lower() in [".jpg", ".png"]],
        key=natural_sort_key,
    )
    print("\n" + "=" * 120)
    print(f"🔬 DEEP FORENSIC PATHOLOGY REPORT FOR SEQUENCE: {colorstr('bold', colorstr('magenta', args.seq))}")
    print(f"Total Video Frames Found: {len(img_files)}")
    print("=" * 120)

    rows: List[dict] = []
    patches: List[np.ndarray] = []
    resolution_seen: Dict[str, int] = {}
    cache_mismatch = 0

    for frame_idx, p in enumerate(img_files):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        resolution_seen[f"{w}x{h}"] = resolution_seen.get(f"{w}x{h}", 0) + 1

        boxes = load_labels(val_lbl_dir / f"{p.stem}.txt")
        if not boxes:
            continue

        if img.ndim == 3 and img.shape[2] >= 3:
            channels = (img[:, :, 0], img[:, :, 1], img[:, :, 2])
        else:
            channels = (img, np.zeros_like(img), np.zeros_like(img))

        rec = cache_records.get(p.name)
        gt_pts_lb = rec["gt_pts"] if rec is not None else None
        pred_pts = rec["pred_points"] if rec is not None else None
        pred_scs = rec["pred_scores"] if rec is not None else None
        pair_ok = rec is not None and gt_pts_lb is not None and len(gt_pts_lb) == len(boxes)
        if rec is not None and not pair_ok:
            cache_mismatch += 1

        for bi, box in enumerate(boxes):
            cx = int(round(box[0] * w))
            cy = int(round(box[1] * h))
            raw, diff, median = [patch_metrics(ch, cx, cy) for ch in channels]
            cloud_std = channel_change(channels[0], cx, cy)
            patch = None
            if 10 <= cx < w - 10 and 10 <= cy < h - 10:
                patch = channels[0][cy - 10 : cy + 11, cx - 10 : cx + 11].astype(np.float32)

            pred_score, pred_dist = 0.0, 999.0
            if pair_ok and pred_pts is not None and len(pred_pts):
                gx, gy = gt_pts_lb[bi]
                distances = np.linalg.norm(pred_pts - np.array([gx, gy], dtype=np.float32), axis=1)
                nearest = int(np.argmin(distances))
                pred_dist = float(distances[nearest])
                pred_score = float(pred_scs[nearest])

            rows.append({
                "frame_index": frame_idx,
                "file": p.name,
                "native_w": w,
                "native_h": h,
                "box_index": bi,
                "cx": cx,
                "cy": cy,
                "box_w": box[2] * w,
                "box_h": box[3] * h,
                "raw_peak": raw["peak"],
                "raw_trough": raw["trough"],
                "raw_bg_mean": raw["bg_mean"],
                "raw_bg_std": raw["bg_std"],
                "raw_signed": raw["signed"],
                "raw_contrast": raw["contrast"],
                "raw_mad_z": raw["mad_z"],
                "raw_abs_mad_z": raw["abs_mad_z"],
                "diff_peak": diff["peak"],
                "diff_signed": diff["signed"],
                "diff_mad_z": diff["mad_z"],
                "median_peak": median["peak"],
                "median_signed": median["signed"],
                "median_mad_z": median["mad_z"],
                "cloud_local_std": cloud_std,
                "prediction_score": pred_score,
                "prediction_distance": pred_dist,
                "detected": int(pred_dist <= 8.0 and pred_score >= 0.22),
                "weak_candidate": int(pred_dist <= 8.0 and pred_score >= 0.06),
            })
            patches.append(patch)

    if not rows:
        print(colorstr("red", "[ERROR] No ground truth boxes found for this sequence."))
        sys.exit(1)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"[INFO] Native resolutions seen: {resolution_seen}")
    print(f"[INFO] Ground truth boxes profiled: {n} | cache/label pairing mismatches: {cache_mismatch}")
    print(f"[INFO] Wrote per-frame signal profile: {output_csv}")

    print("\n" + colorstr("bold", colorstr("cyan", "📊 0. ROBUST SIGNAL / BACKGROUND DIAGNOSTICS (NATIVE COORDS):")))
    print("-" * 120)
    print(f"• Detected (dist<=8px, score>=0.22): {sum(r['detected'] for r in rows)} ({100 * np.mean([r['detected'] for r in rows]):.1f}%)")
    print(f"• Weak candidate (dist<=8px, score>=0.06): {sum(r['weak_candidate'] for r in rows)} ({100 * np.mean([r['weak_candidate'] for r in rows]):.1f}%)")
    print("• Metric                         Mean     Median      P10       P25       P75       P90")
    for key in ("raw_peak", "raw_bg_mean", "raw_bg_std", "raw_signed", "raw_contrast", "raw_mad_z", "raw_abs_mad_z", "diff_peak", "diff_signed", "median_peak", "median_signed", "cloud_local_std"):
        s = percentile_summary([r[key] for r in rows])
        print(f"  {key:<29} {s['mean']:>7.2f} {s['median']:>9.2f} {s['p10']:>9.2f} {s['p25']:>9.2f} {s['p75']:>9.2f} {s['p90']:>9.2f}")
    frac_dark = float(np.mean([r["raw_signed"] < 0 for r in rows]) * 100)
    print(f"• Dark-target frames (signed raw < 0): {frac_dark:.1f}% (polarity check; max-based metrics fail on dark targets)")
    print("• raw_abs_mad_z is polarity-agnostic and is the fair detectability statistic.")

    weak_flags = [r["weak_candidate"] for r in rows]
    visible_flags = [r["raw_abs_mad_z"] >= 3.0 for r in rows]
    frame_ids = sorted({r["frame_index"] for r in rows})
    gaps = [frame_ids[i] - frame_ids[i - 1] for i in range(1, len(frame_ids))]
    print("\n" + colorstr("bold", colorstr("cyan", "📊 0b. TEMPORAL CONTINUITY & ANNOTATION STRUCTURE:")))
    print("-" * 120)
    print(f"• GT-annotated span        : Frame #{frame_ids[0]} ~ #{frame_ids[-1]}")
    print(f"• Consecutive GT frames     : {sum(1 for g in gaps if g == 1)} / {len(gaps)} intervals")
    print(f"• Annotation gap            : max={max(gaps) if gaps else 0}, mean={np.mean(gaps) if gaps else 0:.2f}")
    print(f"• Longest weak-candidate run near GT : {longest_runs(weak_flags)[:5]}")
    print(f"• Longest raw-visible run (|MAD-Z|>=3): {longest_runs(visible_flags)[:5]}")

    valid_patches = [p for p in patches if p is not None]
    print("\n" + colorstr("bold", colorstr("cyan", "📊 0c. ORACLE TEMPORAL INTEGRATION UPPER BOUND (GT-ALIGNED WINDOW AVERAGING):")))
    print("-" * 120)
    print(f"  {'Window N':<8} | {'Mean |contrast|':>15} | {'Median |contrast|':>17} | {'Mean |MAD-Z|':>12} | {'>=2.0':>10} | {'>=3.0':>10}")
    for window in (1, 3, 5, 9, 21):
        if len(valid_patches) < window:
            continue
        contrasts, mad_zs = [], []
        for start in range(0, len(valid_patches) - window + 1):
            stacked = np.mean(np.stack(valid_patches[start : start + window], axis=0), axis=0)
            center = stacked[8:13, 8:13]
            ring = np.concatenate([stacked[0:4, :].ravel(), stacked[17:21, :].ravel(), stacked[4:17, 0:4].ravel(), stacked[4:17, 17:21].ravel()])
            ring_median = float(np.median(ring))
            ring_mad = float(np.median(np.abs(ring - ring_median)))
            peak = float(np.max(center))
            trough = float(np.min(center))
            signal = peak - ring_median if (peak - ring_median) >= (ring_median - trough) else trough - ring_median
            contrasts.append(abs(signal) / (float(np.std(ring)) + 1e-4))
            mad_zs.append(abs(signal) / (1.4826 * ring_mad + 1e-4))
        contrasts = np.asarray(contrasts)
        mad_zs = np.asarray(mad_zs)
        print(f"  {window:<8} | {np.mean(contrasts):>15.2f} | {np.median(contrasts):>17.2f} | {np.mean(mad_zs):>12.2f} | {int(np.sum(contrasts >= 2.0)):>4}/{len(contrasts):<5} | {int(np.sum(contrasts >= 3.0)):>4}/{len(contrasts):<5}")
    print("• GT-aligned patches are already motion-centered, so N-frame averaging IS motion-compensated integration.")
    print("• Contrast rising with N means temporal energy is recoverable (Phase-B justified). Flat/dropping means the")
    print("  target is not a temporally persistent point even under perfect alignment.")

    first_per_frame = {}
    for r in rows:
        first_per_frame.setdefault(r["frame_index"], r)
    ordered = [first_per_frame[k] for k in sorted(first_per_frame)]
    w_px = np.array([r["box_w"] for r in ordered], dtype=np.float32)
    h_px = np.array([r["box_h"] for r in ordered], dtype=np.float32)
    cx_px = np.array([r["cx"] for r in ordered], dtype=np.float32)
    cy_px = np.array([r["cy"] for r in ordered], dtype=np.float32)
    positions = np.stack([cx_px, cy_px], axis=1)
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    print("\n" + colorstr("bold", colorstr("cyan", "📐 1. TARGET SCALE & KINEMATIC GEOMETRY PROFILE (NATIVE COORDS):")))
    print("-" * 80)
    print(f"• Bounding Box Width  (px)  : Min={w_px.min():.2f}, Mean={w_px.mean():.2f}, Median={np.median(w_px):.2f}, Max={w_px.max():.2f}")
    print(f"• Bounding Box Height (px)  : Min={h_px.min():.2f}, Mean={h_px.mean():.2f}, Median={np.median(h_px):.2f}, Max={h_px.max():.2f}")
    print(f"• Target Centroid X Span    : Min={cx_px.min():.1f} ~ Max={cx_px.max():.1f}")
    print(f"• Target Centroid Y Span    : Min={cy_px.min():.1f} ~ Max={cy_px.max():.1f}")
    if len(step):
        print(f"• Inter-frame Velocity (px) : Min={step.min():.2f}, Mean={step.mean():.2f}, Median={np.median(step):.2f}, Max={step.max():.2f}")
        print(f"• Frames moving > 1.0 px/f  : {int(np.sum(step > 1.0))} ({np.mean(step > 1.0) * 100:.1f}%)")
        print(f"• Frames moving < 0.3 px/f  : {int(np.sum(step < 0.3))} ({np.mean(step < 0.3) * 100:.1f}%)")
    print(f"• Net Start-to-End Displacement: {np.linalg.norm(positions[-1] - positions[0]):.2f}px")
    print(f"• Cumulative Path Length       : {step.sum():.1f}px")
    print("• NOTE: the earlier 'stop-and-go' / '829px odometer' conclusions came from a 640x640 coordinate bug and are void.")

    print("\n" + colorstr("bold", colorstr("magenta", "📡 2. THREE-CHANNEL SIGNAL AT GT (SAMPLED):")))
    print("-" * 120)
    sample_step = max(1, len(ordered) // 15)
    sample = ordered[::sample_step]
    if ordered[-1] not in sample:
        sample.append(ordered[-1])
    print(f"{'file':<30} | {'native':<9} | {'GT (cx,cy)':<12} | {'Ch0':<7} | {'bg':<14} | {'Ch1':<6} | {'Ch2':<6} | {'score@dist':<14} | status")
    print("-" * 120)
    for r in sample:
        if r["prediction_distance"] <= 8.0 and r["prediction_score"] >= 0.22:
            status = colorstr("green", "DETECTED")
        elif r["prediction_distance"] <= 8.0 and r["prediction_score"] >= 0.06:
            status = colorstr("yellow", "WEAK PEAK")
        else:
            status = colorstr("red", "BLIND")
        print(
            f"{r['file'][:29]:<30} | {r['native_w']}x{r['native_h']:<5} | "
            f"({r['cx']:>3d},{r['cy']:>3d})  | {r['raw_peak']:>6.1f} | "
            f"{r['raw_bg_mean']:.1f}±{r['raw_bg_std']:<8.1f} | {r['diff_peak']:>5.1f} | {r['median_peak']:>5.1f} | "
            f"{r['prediction_score']:>5.2f}@{r['prediction_distance']:>5.1f}px | {status}"
        )

    print("\n" + colorstr("bold", colorstr("cyan", "📊 3. DIAGNOSTIC HYPOTHESIS:")))
    print("-" * 100)
    mean_raw = np.mean([r["raw_peak"] for r in rows])
    mean_diff = np.mean([r["diff_peak"] for r in rows])
    mean_med = np.mean([r["median_peak"] for r in rows])
    mean_abs_madz = np.mean([r["raw_abs_mad_z"] for r in rows])
    print(f"• Mean Ch0 raw peak={mean_raw:.2f} | Ch1 diff={mean_diff:.2f} | Ch2 median={mean_med:.2f} | mean |raw MAD-Z|={mean_abs_madz:.2f}")
    if mean_diff < 3.0 and mean_med < 3.0:
        print(colorstr("bold", colorstr("red", "🚨 TEMPORAL FEATURE COLLAPSE: raw signal exists but diff/median channels are near zero.")))
    elif mean_abs_madz < 3.0:
        print(colorstr("bold", colorstr("red", "🚨 SUB-DETECTION RAW SIGNAL: target peak does not clear background noise.")))
    else:
        print(colorstr("bold", colorstr("yellow", "⚡ INPUT/RAW SIGNAL IS HEALTHY: the failure is in the network response or in annotation alignment, not in the input energy.")))
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
