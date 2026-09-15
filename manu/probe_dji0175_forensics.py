#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Forensic Time-Slice Diagnostic Probe for DJI_0175_2.
Diagnoses:
1. Ground Truth trajectory and continuity (448 frames).
2. Per-frame maximum prediction score and detection positions around GT.
3. Distribution of missed targets (FN) across confidence bins ([0.0, 0.06), [0.06, 0.22), [0.22, 1.0]).
4. Temporal gap lengths (consecutive missed frames due to attitude roll / transient darkening).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.eval_bidirectional_track_fusion import natural_sort_key


def parse_args():
    parser = argparse.ArgumentParser(description="Probe DJI_0175_2 Ground Truth & Confidence Dynamics")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 inference cache",
    )
    parser.add_argument("--seq", type=str, default="DJI_0175_2")
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    return parser.parse_args()


def main():
    args = parse_args()
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

    if not cache_path.exists():
        print(colorstr("red", f"[ERROR] Cache not found: {args.cache_file}"))
        sys.exit(1)

    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    seq_records = [r for r in records if args.seq in r["im_name"]]
    seq_records.sort(key=lambda r: natural_sort_key(r["im_name"]))

    total_frames = len(seq_records)
    print(colorstr("bold", f"\n>>> FORENSIC PROBE FOR {args.seq} ({total_frames} FRAMES) <<<"))

    gt_counts = 0
    scores_at_gt = []
    max_scores_in_frame = []
    miss_ranges = []
    cur_miss_start = None
    cur_miss_len = 0

    conf_bins = {
        ">=0.22 (Direct Seed)": 0,
        "0.15~0.22 (Strong Salvage)": 0,
        "0.06~0.15 (Weak Salvage)": 0,
        "0.03~0.06 (Ultra-Weak Pulse)": 0,
        "<0.03 (Completely Dark / Missed)": 0,
    }

    for f_idx, r in enumerate(seq_records):
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        pred_pts = np.asarray(r["pred_points"], dtype=np.float32)
        pred_scs = np.asarray(r["pred_scores"], dtype=np.float32)

        if len(gt_pts) == 0:
            continue

        gt_counts += 1
        gx, gy = gt_pts[0]

        # Check nearest prediction to GT
        best_sc = 0.0
        best_dist = 999.0
        if len(pred_pts) > 0:
            dists = np.sqrt(np.sum((pred_pts - gt_pts[0]) ** 2, axis=1))
            min_idx = np.argmin(dists)
            best_dist = dists[min_idx]
            if best_dist <= args.dist_thresh:
                best_sc = float(pred_scs[min_idx])

        scores_at_gt.append((f_idx, best_sc, best_dist))
        max_scores_in_frame.append(float(np.max(pred_scs)) if len(pred_scs) > 0 else 0.0)

        # Categorize
        if best_sc >= 0.22:
            conf_bins[">=0.22 (Direct Seed)"] += 1
        elif best_sc >= 0.15:
            conf_bins["0.15~0.22 (Strong Salvage)"] += 1
        elif best_sc >= 0.06:
            conf_bins["0.06~0.15 (Weak Salvage)"] += 1
        elif best_sc >= 0.03:
            conf_bins["0.03~0.06 (Ultra-Weak Pulse)"] += 1
        else:
            conf_bins["<0.03 (Completely Dark / Missed)"] += 1

        # Track consecutive missing frames (best_sc < 0.22)
        if best_sc < 0.22:
            if cur_miss_start is None:
                cur_miss_start = f_idx
                cur_miss_len = 1
            else:
                cur_miss_len += 1
        else:
            if cur_miss_start is not None:
                miss_ranges.append((cur_miss_start, cur_miss_start + cur_miss_len - 1, cur_miss_len))
                cur_miss_start = None
                cur_miss_len = 0

    if cur_miss_start is not None:
        miss_ranges.append((cur_miss_start, cur_miss_start + cur_miss_len - 1, cur_miss_len))

    print("\n" + "=" * 70)
    print("📊 TARGET PEAK ACTIVATION BREAKDOWN AT GT POSITION (Tolerance <= 8.0px):")
    print("=" * 70)
    for bin_name, count in conf_bins.items():
        ratio = count / total_frames * 100
        print(f"  {bin_name:<36} : {count:>4} frames ({ratio:5.1f}%)")

    print("\n" + "=" * 70)
    print("🔍 TRANSIENT DARKENING GAP DISTRIBUTION (Consecutive frames where score < 0.22):")
    print("=" * 70)
    gap_lens = [g[2] for g in miss_ranges]
    print(f"Total Darkening / Miss Episodes : {len(miss_ranges)}")
    if gap_lens:
        print(f"Max Consecutive Miss Gap        : {max(gap_lens)} frames")
        print(f"Mean Consecutive Miss Gap       : {np.mean(gap_lens):.1f} frames")
        print(f"Median Miss Gap                 : {np.median(gap_lens):.1f} frames")
        print("\nTop-10 Longest Darkening Gaps:")
        for start_f, end_f, length in sorted(miss_ranges, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  • Frames {start_f:03d} -> {end_f:03d} (Gap Length: {length:2d} frames)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
