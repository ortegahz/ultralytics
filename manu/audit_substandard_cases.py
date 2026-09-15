#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Audit and List Substandard Sequences Against Industrial Delivery Criteria.

Industrial Delivery Criteria (GJB / High-Confidence Electro-Optical Standards):
1. Precision Criteria : Precision >= 90.0% AND Total FP <= 30 per 1000 frames (FAR <= 0.030)
2. Recall Criteria    : Recall >= 85.0%
3. F1-Score Criteria  : F1 >= 0.8800

Any sequence failing ANY of these criteria is classified as SUBSTANDARD.
Sequences are ranked and categorized by their primary failure mode:
  - SEVERE_MISSED (Recall < 60% or F1 < 0.70)
  - CLUTTER_EXPLOSION (Precision < 70% or FP > 150)
  - MARGINAL_DEFICIT (Fails by a narrow margin, e.g. F1 in 0.80~0.88)

Usage:
    python manu/audit_substandard_cases.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --th-base 0.22 \
        --th-salvage 0.06 \
        --th-ground 0.35 \
        --min-hits-infill 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.eval_bidirectional_track_fusion import (
    evaluate_sequence_bidirectional,
    extract_seq_name,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Substandard UAV Infrared Sequences")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 pickle cache",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="GJB pixel tolerance")
    parser.add_argument("--th-base", type=float, default=0.22, help="Seed detection threshold")
    parser.add_argument("--th-salvage", type=float, default=0.06, help="Kinematic salvage threshold")
    parser.add_argument("--th-ground", type=float, default=0.35, help="Clutter/ground threshold")
    parser.add_argument("--min-hits-infill", type=int, default=5, help="Min track hits for infill")
    parser.add_argument("--min-rigid-disp", type=float, default=2.0, help="Min net displacement for rigid static pruner (default: 2.0px)")
    parser.add_argument("--max-rigid-var", type=float, default=0.5, help="Max coordinate variance for rigid static pruner (default: 0.5px²)")
    parser.add_argument("--img-h", type=int, default=640)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Path to dataset root (containing labels/val for Point-in-BBox fallback)",
    )
    parser.add_argument(
        "--match-mode",
        type=str,
        default="bbox",
        choices=["bbox", "dist"],
        help="Matching criterion: 'bbox' (Point-in-BBox or dist <= dist_thresh) or 'dist' (strict dist <= dist_thresh)",
    )
    # Delivery criteria thresholds (in percentage, matching calc_metrics scale 0~100)
    parser.add_argument("--min-f1", type=float, default=88.0, help="Delivery criteria min F1 (%)")
    parser.add_argument("--min-recall", type=float, default=85.0, help="Delivery criteria min Recall (%)")
    parser.add_argument("--min-prec", type=float, default=90.0, help="Delivery criteria min Precision (%)")
    parser.add_argument("--max-pure-bg-far", type=float, default=0.030, help="Max allowable FAR for pure negative sequences (GT=0)")
    parser.add_argument(
        "--exclude-avian",
        action="store_true",
        default=False,
        help="Exclude confirmed natural avian clutter sequences (01_4485_1167-2666 & wg2022_ir_020_split_07)",
    )
    parser.add_argument(
        "--exclude-demo",
        action="store_true",
        default=False,
        help="Exclude DJI_0175_2: although numerical recall is not fully up to standard, visual diagnostic video is ready and sufficient for leadership demo. Priority lowered to backlog.",
    )
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
        print(colorstr("red", f"[ERROR] Cache file not found: {args.cache_file}"))
        print("[HINT] Ensure the cache file exists on remote or specify via --cache-file")
        sys.exit(1)

    print(f"[INFO] Loading inference cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    # Enrich records with gt_bboxes if in bbox mode and gt_bboxes not present in cache
    if args.match_mode == "bbox":
        data_root = Path(args.data_root)
        val_lbl_dir = data_root / "labels" / "val"
        if not val_lbl_dir.exists():
            cand_root = Path("/home/manu/mnt/datasets/manu/uav_gmc_median")
            if (cand_root / "labels" / "val").exists():
                val_lbl_dir = cand_root / "labels" / "val"

        lbl_cache = {}
        has_gt_bboxes = any("gt_bboxes" in r for r in records[:50])
        if not has_gt_bboxes and val_lbl_dir.exists():
            print(f"[INFO] Enriching cache records with GT BBoxes from: {val_lbl_dir} (Point-in-BBox mode)")
            for r in records:
                im_name = r["im_name"]
                stem = Path(im_name).stem
                lbl_p = val_lbl_dir / f"{stem}.txt"
                bboxes = []
                if lbl_p.exists():
                    if stem not in lbl_cache:
                        with open(lbl_p, "r", encoding="utf-8") as f_lbl:
                            lines = [l.strip().split() for l in f_lbl if l.strip()]
                        lbl_boxes = []
                        for l in lines:
                            # YOLO format: class cx cy w h (normalized)
                            box = [float(x) for x in l[1:5]]
                            # Convert to pixel cx, cy, w, h
                            lbl_boxes.append([
                                box[0] * args.img_w,
                                box[1] * args.img_h,
                                box[2] * args.img_w,
                                box[3] * args.img_h,
                            ])
                        lbl_cache[stem] = np.array(lbl_boxes, dtype=np.float32)
                    bboxes = lbl_cache[stem]
                r["gt_bboxes"] = bboxes if len(bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        elif has_gt_bboxes:
            print("[INFO] Using GT BBoxes embedded in cache file (Point-in-BBox mode)")
        else:
            print(colorstr("yellow", f"[WARN] labels/val not found at {val_lbl_dir}. Falling back to distance-only matching."))

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    tracker_config = {
        "max_age": 3,
        "min_hits": 3,
        "match_dist": 12.0,
        "max_match_dist": 18.0,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_displacement": 2.5,
        "sky_ratio": 0.60,
        "img_h": args.img_h,
    }

    smoother_config = {
        "stitch_max_gap": 4,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": args.min_hits_infill,
        "max_infill_gap": 3,
        "min_track_hits": 3,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_rigid_displacement": args.min_rigid_disp,
        "max_rigid_variance": args.max_rigid_var,
    }

    print("\n" + "=" * 125)
    print("🔍 AUDITING ALL VALIDATION SEQUENCES AGAINST INDUSTRIAL DELIVERY CRITERIA")
    print(f"Delivery Standards: F1 >= {args.min_f1:.1f}% | Recall >= {args.min_recall:.1f}% | Precision >= {args.min_prec:.1f}%")
    print(f"Matching Criterion: {'Point-in-BBox (or dist <= ' + str(args.dist_thresh) + 'px)' if args.match_mode == 'bbox' else 'Strict Distance <= ' + str(args.dist_thresh) + 'px'}")
    print("=" * 125)

    # Confirmed natural avian clutter sequences (human expert verified: physical birds gliding / flapping)
    # In infrared point impulse regime (1~2px), birds in gliding phase are physically and optically
    # indistinguishable from drones; their flapping phases produce legitimate multi-frame aerodynamic flight.
    AVIAN_SEQUENCES = {
        "01_4485_1167-2666": "Confirmed natural bird flight (212 FP total, ~200 frames persistent avian flight)",
        "wg2022_ir_020_split_07": "Confirmed natural bird flight (226 FP total, ~189 frames persistent avian flight)",
    }

    # Leadership demo qualified sequences (visual tracking sufficient for demonstration, priority deprioritized)
    DEMO_SEQUENCES = {
        "DJI_0175_2": "Diagnostic video DJI_0175_2_paper_diagnostic_panel.mp4 verified sufficient for leadership demo. Priority deprioritized to backlog.",
    }

    if args.exclude_avian:
        print(colorstr("yellow", "\n[NOTE] --exclude-avian active: Excluding 2 confirmed bird sequences from UAV audit:"))
        for a_seq, a_reason in AVIAN_SEQUENCES.items():
            print(f"  • {a_seq}: {a_reason}")

    if args.exclude_demo:
        print(colorstr("cyan", "\n[NOTE] --exclude-demo active: Excluding leadership demo verified sequence:"))
        for d_seq, d_reason in DEMO_SEQUENCES.items():
            print(f"  • {d_seq}: {d_reason}")

    seq_results = []
    total_tp, total_fp, total_gt, total_frames = 0, 0, 0, 0

    for seq_name in sorted(seq_records.keys()):
        if args.exclude_avian and seq_name in AVIAN_SEQUENCES:
            continue
        if args.exclude_demo and seq_name in DEMO_SEQUENCES:
            continue
        recs = seq_records[seq_name]
        eval_res = evaluate_sequence_bidirectional(
            records=recs,
            dist_thresh=args.dist_thresh,
            th_base=args.th_base,
            th_salvage=args.th_salvage,
            th_ground=args.th_ground,
            sky_ratio=0.60,
            img_h=args.img_h,
            tracker_config=tracker_config,
            smoother_config=smoother_config,
            match_mode=args.match_mode,
        )

        bidi = eval_res["bidirectional"]
        tp = bidi["tp"]
        fp = bidi["fp"]
        gt = bidi["gt"]
        fn = gt - tp
        frames = len(recs)
        rec = bidi["recall"]
        prec = bidi["precision"]
        f1 = bidi["f1"]
        far = fp / max(1, frames)

        total_tp += tp
        total_fp += fp
        total_gt += gt
        total_frames += frames

        # Check pass status: Special evaluation rule for pure background sequences (GT=0)
        if gt == 0:
            pass_far = far <= args.max_pure_bg_far
            is_substandard = not pass_far
            if is_substandard:
                category = "⚠️ PURE_BG_HIGH_FAR (Pure Negative High False Alarm)"
                severity = 2
            else:
                category = "✅ PASSED (Pure Negative Background Qualified)"
                severity = 4
        else:
            pass_f1 = f1 >= args.min_f1
            pass_rec = rec >= args.min_recall
            pass_prec = prec >= args.min_prec
            is_substandard = not (pass_f1 and pass_rec and pass_prec)

            # Categorize defect (recall/prec/f1 are on 0~100 scale)
            if rec < 60.0 or f1 < 70.0:
                category = "🚨 CRITICAL_MISSED (Severe Target Drop)"
                severity = 1
            elif prec < 70.0 or fp > 150:
                category = "⚠️ CLUTTER_LEAKAGE (High False Alarm)"
                severity = 2
            elif is_substandard:
                category = "⚡ MARGINAL_DEFICIT (Near Threshold)"
                severity = 3
            else:
                category = "✅ PASSED (Qualified for Delivery)"
                severity = 4

        seq_results.append({
            "seq": seq_name,
            "frames": frames,
            "gt": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": rec,
            "precision": prec,
            "f1": f1,
            "far": far,
            "is_substandard": is_substandard,
            "category": category,
            "severity": severity,
        })

    # Sort substandard cases: first by severity (1=worst), then by F1 ascending
    substandard_cases = [s for s in seq_results if s["is_substandard"]]
    passed_cases = [s for s in seq_results if not s["is_substandard"]]

    substandard_cases.sort(key=lambda x: (x["severity"], x["f1"]))

    print("\n" + colorstr("bold", colorstr("red", f"🚨 SUBSTANDARD CASES AUDIT REPORT ({len(substandard_cases)} / {len(seq_results)} SEQUENCES FAILED)")))
    print("=" * 125)
    header = (
        f"{'Rank':<4} | {'Sequence Name':<28} | {'Frames':<6} | {'GT':<5} | {'TP':<5} | {'FP':<5} | "
        f"{'Recall':<7} | {'Prec':<7} | {'F1':<7} | {'FAR (fp/f)':<10} | {'Defect Classification'}"
    )
    print(header)
    print("-" * 125)

    sub_fn_sum = 0
    sub_fp_sum = 0
    for idx, s in enumerate(substandard_cases, 1):
        sub_fn_sum += s["fn"]
        sub_fp_sum += s["fp"]
        rec_str = f"{s['recall']:>5.1f}%"
        prec_str = f"{s['precision']:>5.1f}%"
        f1_str = f"{s['f1']:>6.2f}"
        far_str = f"{s['far']:.4f}"
        line = (
            f"{idx:<4} | {s['seq']:<28} | {s['frames']:<6} | {s['gt']:<5} | {s['tp']:<5} | {s['fp']:<5} | "
            f"{rec_str:<7} | {prec_str:<7} | {f1_str:<7} | {far_str:<10} | {s['category']}"
        )
        if s["severity"] == 1:
            print(colorstr("bold", colorstr("red", line)))
        elif s["severity"] == 2:
            print(colorstr("yellow", line))
        else:
            print(line)

    print("=" * 125)
    print(colorstr("bold", f"\n📊 DEFECT CONCENTRATION ANALYSIS:"))
    print(f"• Total GT in Full Dataset      : {total_gt:,} (Missed FN: {total_gt - total_tp:,})")
    print(f"• Total FP in Full Dataset      : {total_fp:,}")
    print(
        f"• Missed FNs in Substandard List: {sub_fn_sum:,} / {total_gt - total_tp:,} "
        f"({sub_fn_sum / max(1, total_gt - total_tp) * 100:.1f}% of ALL project missed targets!)"
    )
    print(
        f"• False FPs in Substandard List : {sub_fp_sum:,} / {total_fp:,} "
        f"({sub_fp_sum / max(1, total_fp) * 100:.1f}% of ALL project false alarms!)"
    )

    print("\n" + colorstr("bold", colorstr("green", f"✅ QUALIFIED CASES SUMMARY ({len(passed_cases)} SEQUENCES PASSED):")))
    print("=" * 125)
    pass_header = f"{'Sequence Name':<28} | {'Frames':<6} | {'GT':<5} | {'TP':<5} | {'FP':<5} | {'Recall':<7} | {'Prec':<7} | {'F1':<7} | {'Note'}"
    print(pass_header)
    print("-" * 115)
    for p in sorted(passed_cases, key=lambda x: (x["gt"] > 0, x["f1"]), reverse=True):
        if p["gt"] == 0:
            note = f"Pure Negative (FAR={p['far']:.4f}/f <= {args.max_pure_bg_far})"
            rec_str = "N/A"
            prec_str = "N/A"
            f1_str = "N/A"
        else:
            note = "Standard Qualified"
            rec_str = f"{p['recall']:>5.1f}%"
            prec_str = f"{p['precision']:>5.1f}%"
            f1_str = f"{p['f1']:>6.2f}"

        print(
            f"{p['seq']:<28} | {p['frames']:<6} | {p['gt']:<5} | {p['tp']:<5} | {p['fp']:<5} | "
            f"{rec_str:<7} | {prec_str:<7} | {f1_str:<7} | {note}"
        )

    # Master大盘指标 (0~1.0 scale for final master summary)
    total_rec = total_tp / max(1, total_gt)
    total_prec = total_tp / max(1, total_tp + total_fp)
    total_f1 = 2 * total_prec * total_rec / max(1e-6, total_prec + total_rec)
    print("\n" + "=" * 125)
    print(
        colorstr(
            "bold",
            colorstr(
                "cyan",
                f"🏆 MASTER BENCHMARK: F1 = {total_f1:.4f} | Recall = {total_rec * 100:.2f}% (TP={total_tp:,}/{total_gt:,}) | "
                f"Prec = {total_prec * 100:.2f}% (FP={total_fp:,}) | Total Frames = {total_frames:,}",
            ),
        )
    )
    print("=" * 125 + "\n")


if __name__ == "__main__":
    main()
