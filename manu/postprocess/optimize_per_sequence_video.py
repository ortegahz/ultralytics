#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Per-Sequence Hyperparameter Search and Best Diagnostic OSD Video Generator.

Features:
1. Fast local grid search over tracking and post-processing thresholds on cached predictions:
   - Search space focused on:
     * th_base: [0.12, 0.16, 0.20, 0.22, 0.25, 0.28] (seed threshold)
     * th_salvage: [0.03, 0.05, 0.06, 0.08] (kinematic salvage threshold)
     * th_ground: [0.28, 0.32, 0.35, 0.40] (ground clutter threshold)
     * min_hits_infill: [3, 4, 5, 6] (track confirmation threshold for infill)
     * min_rigid_disp: [1.5, 2.0, 2.5] (rigid static pruner displacement)
   - Evaluates in memory using pure numpy/scipy operations.
   - Typically finishes search for a single sequence (~500-1500 frames) in < 15-30 seconds!
2. Renders the diagnostic OSD video using the PER-SEQUENCE OPTIMAL parameters (or optionally baseline).
3. At the end, prints a structured comparison table showing:
   - Baseline (Global System SOTA) metrics: TP, FP, Recall, Precision, F1
   - Optimized (Per-Sequence Best) metrics: TP, FP, Recall, Precision, F1, and Best Parameters
   - Delta gains (ΔRecall, ΔPrecision, ΔF1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import colorstr
from manu.evaluation.eval_bidirectional_track_fusion import (
    evaluate_sequence_bidirectional,
    extract_seq_name,
    natural_sort_key,
)
from manu.videos.generate_sys_sota_osd import build_image_lookup, render_osd_frame


TARGET_HARD_CASES = [
    "wg2022_ir_020_split_03",
    "DJI_0051_2",
    "wg2022_ir_011_split_03",
    "02_6321_0274-2773",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Per-Sequence Parameter Optimizer & Best OSD Video Generator")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 cache file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to data.yaml (or dataset root) to locate original images and labels",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="wg2022_ir_020_split_03",
        help="Target sequence name, or 'hard4' for the 4 hard cases, or 'all' for all 24 sequences",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/per_seq_best_videos",
        help="Directory to save output MP4 video(s)",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="GJB pixel tolerance (default: 8.0px)")
    parser.add_argument("--img-h", type=int, default=640)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--no-video", action="store_true", help="Only run optimization & report comparisons without rendering video")
    parser.add_argument("--quick-search", action="store_true", help="Use a more compact search grid (runs in < 10 seconds per sequence)")
    return parser.parse_args()


def get_search_grid(quick: bool = False):
    """
    Define a compact, high-impact hyperparameter grid.
    Focused on thresholds that directly govern Recall vs False Alarms.
    """
    if quick:
        return [
            # th_base, th_salvage, th_ground, min_hits_infill, min_rigid_disp
            # 1. Global SOTA baseline
            (0.22, 0.06, 0.35, 5, 2.0),
            # 2. High sensitivity / Weak target search
            (0.12, 0.04, 0.30, 4, 2.0),
            (0.14, 0.05, 0.32, 5, 2.0),
            (0.16, 0.05, 0.32, 4, 2.0),
            (0.18, 0.05, 0.35, 4, 2.0),
            (0.18, 0.06, 0.35, 5, 2.0),
            # 3. High precision / Clutter suppression search
            (0.24, 0.08, 0.38, 5, 2.0),
            (0.26, 0.08, 0.40, 6, 2.5),
            (0.28, 0.08, 0.40, 6, 2.5),
            # 4. Infill variations
            (0.20, 0.05, 0.35, 3, 2.0),
            (0.22, 0.06, 0.35, 3, 2.0),
            (0.22, 0.06, 0.35, 6, 2.0),
            # 5. Static pruner variations
            (0.22, 0.06, 0.35, 5, 1.5),
            (0.22, 0.06, 0.35, 5, 2.5),
        ]

    # Full search grid: ~72 combinations (typically takes ~10-15 seconds per sequence)
    grid = []
    th_bases = [0.12, 0.16, 0.20, 0.22, 0.25, 0.28]
    salvages = [0.04, 0.06, 0.08]
    grounds = [0.30, 0.35, 0.40]
    infills = [3, 5]
    prunes = [2.0]

    for tb in th_bases:
        for ts in salvages:
            if ts >= tb:
                continue
            for tg in grounds:
                if tg < tb:
                    continue
                for inf in infills:
                    for pr in prunes:
                        grid.append((tb, ts, tg, inf, pr))

    # Always ensure baseline is in the search list
    baseline_param = (0.22, 0.06, 0.35, 5, 2.0)
    if baseline_param not in grid:
        grid.insert(0, baseline_param)

    # Extra aggressive pruning configs for high-FP sequences
    grid.append((0.24, 0.08, 0.38, 6, 2.5))
    grid.append((0.26, 0.08, 0.40, 6, 2.5))
    grid.append((0.14, 0.04, 0.32, 4, 1.5))
    grid.append((0.16, 0.04, 0.32, 4, 2.0))
    return grid


def optimize_sequence_params(
    seq_records: List[Dict],
    dist_thresh: float = 8.0,
    img_h: int = 640,
    quick_search: bool = False,
) -> Tuple[Dict, Dict, Dict]:
    """
    Run fast grid search over sequence records.
    Returns:
        (baseline_metrics, best_metrics, best_params_dict)
    """
    grid = get_search_grid(quick=quick_search)

    # Evaluate Global SOTA baseline first
    baseline_param = (0.22, 0.06, 0.35, 5, 2.0)
    b_tracker_cfg = {
        "max_age": 3,
        "min_hits": 3,
        "match_dist": 12.0,
        "max_match_dist": 18.0,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_displacement": 2.5,
        "sky_ratio": 0.60,
        "img_h": img_h,
    }
    b_smoother_cfg = {
        "stitch_max_gap": 4,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": 5,
        "max_infill_gap": 3,
        "min_track_hits": 3,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_rigid_displacement": 2.0,
        "max_rigid_variance": 0.5,
        "min_hits_for_prune": 8,
    }

    base_res = evaluate_sequence_bidirectional(
        records=seq_records,
        dist_thresh=dist_thresh,
        th_base=0.22,
        th_salvage=0.06,
        th_ground=0.35,
        sky_ratio=0.60,
        img_h=img_h,
        tracker_config=b_tracker_cfg,
        smoother_config=b_smoother_cfg,
    )
    baseline_metrics = base_res["bidirectional"]

    best_f1 = baseline_metrics["f1"]
    best_metrics = dict(baseline_metrics)
    best_params = {
        "th_base": 0.22,
        "th_salvage": 0.06,
        "th_ground": 0.35,
        "min_hits_infill": 5,
        "min_rigid_disp": 2.0,
    }
    best_eval_res = base_res

    # Grid search
    for tb, ts, tg, inf, pr in grid:
        cur_tracker_cfg = dict(b_tracker_cfg)
        cur_smoother_cfg = dict(b_smoother_cfg)
        cur_smoother_cfg["min_hits_for_infill"] = inf
        cur_smoother_cfg["min_rigid_displacement"] = pr

        res = evaluate_sequence_bidirectional(
            records=seq_records,
            dist_thresh=dist_thresh,
            th_base=tb,
            th_salvage=ts,
            th_ground=tg,
            sky_ratio=0.60,
            img_h=img_h,
            tracker_config=cur_tracker_cfg,
            smoother_config=cur_smoother_cfg,
        )
        cur_metrics = res["bidirectional"]

        # Selection criteria: Higher F1 score; if tied, prefer higher Recall
        if (cur_metrics["f1"] > best_f1 + 1e-4) or (
            abs(cur_metrics["f1"] - best_f1) <= 1e-4 and cur_metrics["recall"] > best_metrics["recall"]
        ):
            best_f1 = cur_metrics["f1"]
            best_metrics = dict(cur_metrics)
            best_params = {
                "th_base": tb,
                "th_salvage": ts,
                "th_ground": tg,
                "min_hits_infill": inf,
                "min_rigid_disp": pr,
            }
            best_eval_res = res

    return baseline_metrics, best_metrics, best_params, best_eval_res


def render_best_video(
    seq_name: str,
    eval_res: Dict,
    data_path: Path,
    out_dir: Path,
    best_params: Dict,
    dist_thresh: float = 8.0,
    img_h: int = 640,
    img_w: int = 640,
    fps: float = 25.0,
):
    """Render OSD video using the best evaluated parameters."""
    bidi_frame_dets = eval_res["bidi_frame_dets"]
    records_sorted = eval_res["records_sorted"]

    # Image lookup
    img_lookup = {}
    if data_path.is_file() and data_path.suffix in (".yaml", ".yml"):
        data_dict = check_det_dataset(str(data_path))
        val_source = data_dict.get("val")
        if val_source:
            img_lookup = build_image_lookup(val_source)

    if not img_lookup:
        for cand in [
            Path("/mnt/data/siping/datasets/manu/uav_gmc_median/images/val"),
            Path("/home/manu/mnt/datasets/manu/uav_gmc_median/images/val"),
            Path("/mnt/data/siping/datasets/manu/uav/images/val"),
        ]:
            if cand.exists():
                img_lookup = build_image_lookup(cand)
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / f"{seq_name}_best_optimized_osd.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    cum_tp, cum_fp, cum_gt = 0, 0, 0
    total_frames = len(records_sorted)

    param_tag = f"b={best_params['th_base']:.2f}/s={best_params['th_salvage']:.2f}/g={best_params['th_ground']:.2f}/inf={best_params['min_hits_infill']}"

    for f_idx, r in enumerate(tqdm(records_sorted, desc=f"Video: {seq_name}")):
        im_name = r["im_name"]
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        gt_bboxes = np.asarray(r.get("gt_bboxes", np.zeros((len(gt_pts), 4))), dtype=np.float32)
        gt_sizes = gt_bboxes[:, 2:4] if len(gt_bboxes) == len(gt_pts) else None
        sys_dets = bidi_frame_dets[f_idx]

        frame = r.get("img_hwc")
        if frame is None:
            p = img_lookup.get(im_name)
            if p is None or not p.exists():
                for ext in [".jpg", ".png", ".jpeg"]:
                    alt = img_lookup.get(Path(im_name).stem + ext)
                    if alt and alt.exists():
                        p = alt
                        break
            if p and p.exists():
                frame = cv2.imread(str(p))

        if frame is None:
            frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        if frame.shape[0] != img_h or frame.shape[1] != img_w:
            frame = cv2.resize(frame, (img_w, img_h))

        osd_frame, tp, fp, fn = render_osd_frame(
            frame=frame,
            gt_pts=gt_pts,
            sys_dets=sys_dets,
            dist_thresh=dist_thresh,
            seq_name=f"{seq_name} [BEST: {param_tag}]",
            frame_idx=f_idx,
            total_frames=total_frames,
            cum_tp=cum_tp,
            cum_fp=cum_fp,
            cum_gt=cum_gt,
            gt_sizes=gt_sizes,
        )

        cum_tp += tp
        cum_fp += fp
        cum_gt += len(gt_pts)

        if writer is None:
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (img_w, img_h))

        writer.write(osd_frame)

    if writer is not None:
        writer.release()
    print(colorstr("green", f"  [SAVED VIDEO] -> {out_video_path.resolve()}"))


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
        print(colorstr("red", f"[ERROR] Cache file not found: {args.cache_file}"))
        sys.exit(1)

    print(colorstr("bold", f"\n>>> Loading cached inferences from: {cache_path}"))
    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    label_dir = data_path.parent / "labels" / "val" if data_path.suffix in (".yaml", ".yml") else data_path / "labels" / "val"
    if not label_dir.exists():
        label_dir = Path("/mnt/data/siping/datasets/manu/uav_gmc_median/labels/val")

    # Enrich gt_bboxes for Point-in-BBox evaluation
    for record in records:
        if "gt_bboxes" not in record:
            boxes = []
            label_path = label_dir / f"{Path(record['im_name']).stem}.txt"
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        _, cx, cy, width, height = map(float, parts[:5])
                        boxes.append([cx * 640.0, cy * 640.0, width * 640.0, height * 640.0])
            record["gt_bboxes"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    # Group records by sequence
    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    # Determine which sequences to process
    if args.seq == "all":
        seqs_to_process = sorted(seq_records.keys(), key=natural_sort_key)
    elif args.seq == "hard4":
        seqs_to_process = [s for s in TARGET_HARD_CASES if s in seq_records]
    else:
        seqs_to_process = [args.seq] if args.seq in seq_records else []

    if not seqs_to_process:
        print(colorstr("red", f"[ERROR] No matching sequence found for '{args.seq}'!"))
        sys.exit(1)

    print(f"[INFO] Optimizing parameters for {len(seqs_to_process)} sequence(s)...\n")

    summary_rows = []
    tot_base = {"gt": 0, "tp": 0, "fp": 0}
    tot_best = {"gt": 0, "tp": 0, "fp": 0}

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    for s_idx, seq_name in enumerate(seqs_to_process, 1):
        recs = seq_records[seq_name]
        t0 = time.time()
        b_metrics, best_m, best_params, best_eval_res = optimize_sequence_params(
            seq_records=recs,
            dist_thresh=args.dist_thresh,
            img_h=args.img_h,
            quick_search=args.quick_search,
        )
        t_elapsed = time.time() - t0

        tot_base["gt"] += int(b_metrics["gt"])
        tot_base["tp"] += int(b_metrics["tp"])
        tot_base["fp"] += int(b_metrics["fp"])

        tot_best["gt"] += int(best_m["gt"])
        tot_best["tp"] += int(best_m["tp"])
        tot_best["fp"] += int(best_m["fp"])

        d_f1 = best_m["f1"] - b_metrics["f1"]
        d_rec = best_m["recall"] - b_metrics["recall"]
        d_tp = int(best_m["tp"]) - int(b_metrics["tp"])
        d_fp = int(best_m["fp"]) - int(b_metrics["fp"])

        p_str = f"b={best_params['th_base']:.2f}, s={best_params['th_salvage']:.2f}, g={best_params['th_ground']:.2f}, inf={best_params['min_hits_infill']}"

        print(
            f"[{s_idx}/{len(seqs_to_process)}] {seq_name:<26} ({t_elapsed:.1f}s) | "
            f"Baseline: F1={b_metrics['f1']:.4f} (R={b_metrics['recall']:.1f}%, P={b_metrics['precision']:.1f}%) -> "
            f"Best: F1={best_m['f1']:.4f} (R={best_m['recall']:.1f}%, P={best_m['precision']:.1f}%) | "
            f"ΔF1={d_f1:+6.4f} (TP:{d_tp:+d}, FP:{d_fp:+d}) | BestParams: [{p_str}]"
        )

        summary_rows.append({
            "seq": seq_name,
            "gt": int(b_metrics["gt"]),
            "base_tp": int(b_metrics["tp"]),
            "base_fp": int(b_metrics["fp"]),
            "base_rec": b_metrics["recall"],
            "base_prec": b_metrics["precision"],
            "base_f1": b_metrics["f1"],
            "best_tp": int(best_m["tp"]),
            "best_fp": int(best_m["fp"]),
            "best_rec": best_m["recall"],
            "best_prec": best_m["precision"],
            "best_f1": best_m["f1"],
            "d_f1": d_f1,
            "d_rec": d_rec,
            "d_tp": d_tp,
            "d_fp": d_fp,
            "params": p_str,
        })

        if not args.no_video:
            render_best_video(
                seq_name=seq_name,
                eval_res=best_eval_res,
                data_path=data_path,
                out_dir=out_dir,
                best_params=best_params,
                dist_thresh=args.dist_thresh,
                img_h=args.img_h,
                img_w=args.img_w,
                fps=args.fps,
            )

    # Final Overall Comparison Table
    print("\n" + "=" * 135)
    print(colorstr("bold", "PER-SEQUENCE PARAMETER SEARCH VS GLOBAL BASELINE SUMMARY REPORT"))
    print("=" * 135)
    header = (
        f"{'Sequence Name':<26} | {'GT':<5} | "
        f"{'Base(TP/FP)':<12} | {'Base F1':<8} | "
        f"{'Best(TP/FP)':<12} | {'Best F1':<8} | "
        f"{'Gain (ΔF1, ΔTP, ΔFP)':<22} | {'Optimal Config (th_base, salvage, ground, infill)'}"
    )
    print(header)
    print("-" * 135)

    for row in summary_rows:
        base_tf = f"{row['base_tp']}/{row['base_fp']}"
        best_tf = f"{row['best_tp']}/{row['best_fp']}"
        gain_str = f"{row['d_f1']:+6.4f} (TP:{row['d_tp']:+d}, FP:{row['d_fp']:+d})"

        line = (
            f"{row['seq']:<26} | {row['gt']:<5} | "
            f"{base_tf:<12} | {row['base_f1']:>7.4f} | "
            f"{best_tf:<12} | {row['best_f1']:>7.4f} | "
            f"{gain_str:<22} | {row['params']}"
        )
        if row["d_f1"] > 0.005:
            print(colorstr("green", colorstr("bold", line)))
        else:
            print(line)

    print("=" * 135)

    # Compute overall macro/micro metrics across all processed sequences
    tot_base_rec = tot_base["tp"] / max(1, tot_base["gt"]) * 100.0
    tot_base_prec = tot_base["tp"] / max(1, tot_base["tp"] + tot_base["fp"]) * 100.0
    tot_base_f1 = 2 * tot_base_rec * tot_base_prec / max(1e-6, tot_base_rec + tot_base_prec)

    tot_best_rec = tot_best["tp"] / max(1, tot_best["gt"]) * 100.0
    tot_best_prec = tot_best["tp"] / max(1, tot_best["tp"] + tot_best["fp"]) * 100.0
    tot_best_f1 = 2 * tot_best_rec * tot_best_prec / max(1e-6, tot_best_rec + tot_best_prec)

    overall_df1 = tot_best_f1 - tot_base_f1
    overall_dtp = tot_best["tp"] - tot_base["tp"]
    overall_dfp = tot_best["fp"] - tot_base["fp"]

    print(
        f"TOTAL PROCESSED ({len(seqs_to_process)} Seqs) | GT: {tot_base['gt']}\n"
        f"  GLOBAL BASELINE SOTA: TP={tot_base['tp']}, FP={tot_base['fp']}, Recall={tot_base_rec:.2f}%, Prec={tot_base_prec:.2f}%, F1={tot_base_f1:.4f}\n"
        f"  PER-SEQ BEST SOTA   : TP={tot_best['tp']}, FP={tot_best['fp']}, Recall={tot_best_rec:.2f}%, Prec={tot_best_prec:.2f}%, F1={tot_best_f1:.4f}\n"
        f"  NET GAIN            : ΔTP = {overall_dtp:+d}, ΔFP = {overall_dfp:+d}, ΔRecall = {tot_best_rec - tot_base_rec:+.2f}%, ΔF1 = {overall_df1:+.4f}"
    )
    print("=" * 135 + "\n")


if __name__ == "__main__":
    main()
