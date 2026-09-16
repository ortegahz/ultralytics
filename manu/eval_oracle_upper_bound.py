#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Oracle Upper Bound Probe & Error Decomposition Tool for UAV Tiny Object Detection.

Answers the fundamental question: "Have we reached the information-theoretic limit?"
by decomposing total system error across three pipeline stages:
  1. Frontend candidate coverage (is the GT target even present in the top-k/conf>=0.02 pool?)
  2. Oracle discrimination & association ceiling (what if a perfect post-processor existed?)
     - Oracle-A: Physics-optimistic upper bound (all non-GT discarded, including birds)
     - Oracle-B: Engineering-realistic upper bound (known 389 bird FPs forced retained)
  3. Score stratification & noise competition (how many true targets live at 0.02~0.06, 0.06~0.22,
     and how many noise candidates compete with them at each band?)
  4. Per-sequence gap audit: Quantify the exact headroom remaining for each Hard Case.

Zero model weights needed. Zero inference required. Operates directly on the existing .pkl cache.
Supports both Strict Distance (<= 8.0px) and Point-in-BBox criteria.

Usage on Server:
    python manu/eval_oracle_upper_bound.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --dist-thresh 8.0 \
        --match-mode bbox
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(path_or_str: str | Path):
    s = Path(path_or_str).stem
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_seq_name(im_name: str) -> str:
    stem = Path(im_name).stem
    if "___" in stem:
        return stem.split("___")[0]
    if "__" in stem:
        return stem.split("__")[0]
    match = re.search(r"^(.*?)(?:[_-]+)?\d{3,}$", stem)
    return match.group(1).rstrip("_-") if match else stem


def point_in_bbox(pt: np.ndarray, bbox: np.ndarray) -> bool:
    """
    Check if a point pt [x, y] is inside bbox [cx, cy, w, h] (in pixel coordinates).
    """
    cx, cy, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    x_min, x_max = cx - w / 2.0, cx + w / 2.0
    y_min, y_max = cy - h / 2.0, cy + h / 2.0
    return bool((x_min <= pt[0] <= x_max) and (y_min <= pt[1] <= y_max))


def is_candidate_matching_gt(
    cand_pt: np.ndarray,
    gt_pt: np.ndarray,
    gt_bbox: np.ndarray | None,
    dist_thresh: float,
    match_mode: str,
) -> Tuple[bool, float]:
    """
    Determine if candidate matches GT under specified criterion.
    Returns (is_match, spatial_distance).
    """
    dist = float(np.linalg.norm(cand_pt - gt_pt))
    if match_mode == "bbox" and gt_bbox is not None and len(gt_bbox) == 4 and gt_bbox[2] > 0 and gt_bbox[3] > 0:
        if point_in_bbox(cand_pt, gt_bbox) or dist <= dist_thresh:
            return True, dist
    return dist <= dist_thresh, dist


def calc_metrics(tp: int, fp: int, gt: int) -> Dict[str, float]:
    recall = (tp / max(1, gt)) * 100.0
    prec = (tp / max(1, tp + fp)) * 100.0
    f1 = 2 * (prec * recall) / max(1e-6, (prec + recall)) / 100.0
    return {"tp": tp, "fp": fp, "gt": gt, "recall": recall, "precision": prec, "f1": f1}


# Confirmed bird sequences from hard_cases.md (contribute ~389 known indistinguishable FPs)
BIRD_SEQUENCES = {
    "01_4485_1167-2666": 200,  # flight trajectory frames
    "wg2022_ir_020_split_07": 189,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Oracle Upper Bound Probe & Error Decomposition")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 inference cache (.pkl)",
    )
    parser.add_argument(
        "--dist-thresh",
        type=float,
        default=8.0,
        help="Distance tolerance threshold in pixels (default: 8.0)",
    )
    parser.add_argument(
        "--match-mode",
        type=str,
        default="bbox",
        choices=["bbox", "dist"],
        help="Evaluation criterion: 'bbox' (Point-in-BBox ∪ Dist<=8px) or 'dist' (Strict Dist<=8px)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="",
        help="Comma-separated sequence filter (empty for all)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median",
        help="Dataset root containing labels/val (used to enrich gt_bboxes for bbox mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 110)
    print("   UAV Tiny Object Detection: Oracle Upper Bound & Error Decomposition Probe")
    print(f"   Cache File   : {args.cache_file}")
    print(f"   Dist Thresh  : {args.dist_thresh:.1f} px")
    print(f"   Match Mode   : {args.match_mode.upper()} ({'Point-in-BBox ∪ Dist<=8px' if args.match_mode == 'bbox' else 'Strict Dist<=8px'})")
    print("=" * 110)

    cache_path = Path(args.cache_file)
    if not cache_path.exists():
        alt = PROJECT_ROOT / args.cache_file
        if alt.exists():
            cache_path = alt
        else:
            raise FileNotFoundError(f"Cache file not found: {args.cache_file}")

    print(f"[INFO] Loading inference cache from {cache_path}...")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"[INFO] Loaded {len(records)} frame records.")

    # Enrich gt_bboxes for Point-in-BBox mode when the cache does not embed them
    if args.match_mode == "bbox":
        has_gt_bboxes = any("gt_bboxes" in r for r in records[:50])
        if not has_gt_bboxes:
            lbl_dir = Path(args.data_root) / "labels" / "val"
            if not lbl_dir.exists():
                cand_root = Path("/home/manu/mnt/datasets/manu/uav_gmc_median")
                if (cand_root / "labels" / "val").exists():
                    lbl_dir = cand_root / "labels" / "val"
            if lbl_dir.exists():
                print(f"[INFO] Enriching records with GT BBoxes from: {lbl_dir} (Point-in-BBox mode)")
                for r in records:
                    stem = Path(r["im_name"]).stem
                    lbl_p = lbl_dir / f"{stem}.txt"
                    boxes = []
                    if lbl_p.exists():
                        with open(lbl_p, "r", encoding="utf-8") as f_lbl:
                            for line in f_lbl:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    b = [float(x) for x in parts[1:5]]
                                    boxes.append([b[0] * 640.0, b[1] * 640.0, b[2] * 640.0, b[3] * 640.0])
                    r["gt_bboxes"] = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            else:
                print(colorstr("yellow", f"[WARN] labels/val not found at {lbl_dir}. bbox mode degrades to distance-only matching!"))

    # Group by sequence
    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    # Sort frames naturally inside each sequence
    for seq in seq_records:
        seq_records[seq] = sorted(seq_records[seq], key=lambda r: natural_sort_key(r["im_name"]))

    filter_seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]

    # Global accumulation structures
    total_frames = 0
    total_gt = 0
    total_gt_covered = 0  # GT that has AT LEAST ONE candidate within tolerance
    total_gt_lost_frontend = 0  # GT with ZERO candidates within tolerance (frontend dead zone)
    total_candidates_all = 0
    topk_saturated_frames = 0  # Frames where top_k=100 was hit

    # Score stratification of GT-covered candidates: highest-score candidate matching each GT
    # Bands: <0.06 (deep noise), 0.06~0.22 (salvage zone), 0.22~0.25 (near-thresh), >=0.25 (strong)
    gt_best_score_bands = {
        "deep_weak_lt_006": 0,
        "salvage_006_to_022": 0,
        "near_thresh_022_to_025": 0,
        "strong_ge_025": 0,
    }

    # Noise competition: total non-GT candidates per score band across the whole dataset
    noise_score_bands = {
        "deep_weak_lt_006": 0,
        "salvage_006_to_022": 0,
        "near_thresh_022_to_025": 0,
        "strong_ge_025": 0,
    }

    per_seq_stats = {}

    all_keys = sorted(seq_records.keys())
    for seq_name in all_keys:
        if filter_seqs and not any(f in seq_name for f in filter_seqs):
            continue

        recs = seq_records[seq_name]
        seq_gt = 0
        seq_covered = 0
        seq_lost = 0
        seq_total_cand = 0
        seq_matched_cand_count = 0

        # Score distribution for this sequence
        seq_gt_scores = []

        for r in recs:
            total_frames += 1
            gt_pts = np.array(r["gt_pts"], dtype=np.float32) if len(r["gt_pts"]) > 0 else np.zeros((0, 2), dtype=np.float32)
            gt_boxes = np.array(r["gt_bboxes"], dtype=np.float32) if "gt_bboxes" in r and len(r["gt_bboxes"]) > 0 else np.zeros((0, 4), dtype=np.float32)
            pred_pts = np.array(r["pred_points"], dtype=np.float32) if len(r["pred_points"]) > 0 else np.zeros((0, 2), dtype=np.float32)
            pred_scs = np.array(r["pred_scores"], dtype=np.float32) if len(r["pred_scores"]) > 0 else np.zeros((0,), dtype=np.float32)

            num_gt = len(gt_pts)
            num_cands = len(pred_pts)
            seq_gt += num_gt
            total_gt += num_gt
            seq_total_cand += num_cands
            total_candidates_all += num_cands

            if num_cands >= 100:
                topk_saturated_frames += 1

            matched_cand_indices = set()

            # For each GT, find the BEST (highest confidence) candidate within tolerance
            for g_idx in range(num_gt):
                g_pt = gt_pts[g_idx]
                g_box = gt_boxes[g_idx] if g_idx < len(gt_boxes) else None

                best_match_idx = -1
                best_match_score = -1.0
                best_match_dist = 9999.0

                for c_idx in range(num_cands):
                    c_pt = pred_pts[c_idx]
                    c_score = float(pred_scs[c_idx])
                    is_match, d = is_candidate_matching_gt(c_pt, g_pt, g_box, args.dist_thresh, args.match_mode)
                    if is_match:
                        # Prioritize higher score, then closer distance
                        if c_score > best_match_score:
                            best_match_score = c_score
                            best_match_dist = d
                            best_match_idx = c_idx

                if best_match_idx >= 0:
                    seq_covered += 1
                    total_gt_covered += 1
                    matched_cand_indices.add(best_match_idx)
                    seq_gt_scores.append(best_match_score)

                    # Stratification
                    if best_match_score < 0.06:
                        gt_best_score_bands["deep_weak_lt_006"] += 1
                    elif best_match_score < 0.22:
                        gt_best_score_bands["salvage_006_to_022"] += 1
                    elif best_match_score < 0.25:
                        gt_best_score_bands["near_thresh_022_to_025"] += 1
                    else:
                        gt_best_score_bands["strong_ge_025"] += 1
                else:
                    seq_lost += 1
                    total_gt_lost_frontend += 1

            seq_matched_cand_count += len(matched_cand_indices)

            # Noise candidates (all candidates not matching any GT)
            for c_idx in range(num_cands):
                if c_idx not in matched_cand_indices:
                    sc = float(pred_scs[c_idx])
                    if sc < 0.06:
                        noise_score_bands["deep_weak_lt_006"] += 1
                    elif sc < 0.22:
                        noise_score_bands["salvage_006_to_022"] += 1
                    elif sc < 0.25:
                        noise_score_bands["near_thresh_022_to_025"] += 1
                    else:
                        noise_score_bands["strong_ge_025"] += 1

        coverage_rate = (seq_covered / max(1, seq_gt)) * 100.0 if seq_gt > 0 else 100.0
        per_seq_stats[seq_name] = {
            "gt": seq_gt,
            "covered": seq_covered,
            "lost": seq_lost,
            "coverage_rate": coverage_rate,
            "frames": len(recs),
            "total_cand": seq_total_cand,
            "gt_scores": seq_gt_scores,
        }

    # ==============================================================================
    # PART 1: Overall Oracle Ceilings
    # ==============================================================================
    print("\n" + "=" * 110)
    print(colorstr("bold", "SECTION 1: THEORETICAL ORACLE CEILINGS (Full Dataset)"))
    print("=" * 110)

    # Oracle-A: Perfect tracker discards ALL noise -> TP = total_gt_covered, FP = 0
    # To be conservative, let FP = 0 for pure upper bound (mathematical ceiling)
    oracle_a = calc_metrics(tp=total_gt_covered, fp=0, gt=total_gt)

    # Oracle-B: Engineering-realistic upper bound
    # Includes 389 confirmed bird flight frames that are physically indistinguishable from drones
    confirmed_birds_fp = 389
    oracle_b = calc_metrics(tp=total_gt_covered, fp=confirmed_birds_fp, gt=total_gt)

    # SOTA Reference Baseline (Trial 0474 + Smoothing + Pruning)
    # Mode-dependent reference numbers from memory:
    # BBox mode: TP=22,654, FP=1,434, F1=0.9209, Rec=90.22%, Prec=94.05%
    # Dist mode: TP=22,616, FP=1,472, F1=0.9194, Rec=90.06%, Prec=93.89%
    sota_tp = 22654 if args.match_mode == "bbox" else 22616
    sota_fp = 1434 if args.match_mode == "bbox" else 1472
    sota_ref = calc_metrics(tp=sota_tp, fp=sota_fp, gt=total_gt)

    print(f"Total Validation Frames : {total_frames:,}")
    print(f"Total Ground Truth (GT) : {total_gt:,}")
    print(f"Top-k=100 Saturated Frames: {topk_saturated_frames:,} ({topk_saturated_frames / max(1, total_frames) * 100:.2f}%)")
    print("-" * 110)
    print(f"{'System / Bound':<38} | {'TP':<7} | {'FP':<7} | {'FN':<7} | {'Recall':<8} | {'Prec':<8} | {'F1-Score':<8}")
    print("-" * 110)
    print(f"{'Current SOTA Baseline (Trial 0474)':<38} | {sota_ref['tp']:<7} | {sota_ref['fp']:<7} | {total_gt - sota_ref['tp']:<7} | {sota_ref['recall']:>6.2f}% | {sota_ref['precision']:>6.2f}% | {sota_ref['f1']:>6.4f}")
    label_b = "Oracle-B (Realistic Upper Bound, Birds retained)"
    label_a = "Oracle-A (Pure Mathematical Ceiling, FP=0)"
    f1_b_str = f"{oracle_b['f1']:>6.4f}"
    f1_a_str = f"{oracle_a['f1']:>6.4f}"
    print(colorstr("cyan", f"{label_b:<38} | {oracle_b['tp']:<7} | {oracle_b['fp']:<7} | {oracle_b['gt'] - oracle_b['tp']:<7} | {oracle_b['recall']:>6.2f}% | {oracle_b['precision']:>6.2f}% | {f1_b_str}"))
    print(colorstr("green", f"{label_a:<38} | {oracle_a['tp']:<7} | {oracle_a['fp']:<7} | {oracle_a['gt'] - oracle_a['tp']:<7} | {oracle_a['recall']:>6.2f}% | {oracle_a['precision']:>6.2f}% | {f1_a_str}"))
    print("=" * 110)

    delta_tp_possible = total_gt_covered - sota_ref["tp"]
    delta_f1_b = oracle_b["f1"] - sota_ref["f1"]
    print(f"[VERDICT] Theoretical Headroom Remaining in Current Frontend Pipeline:")
    print(f"  • Maximum Recoverable Targets via Post-Processing (ΔTP): +{delta_tp_possible:,} frames (from {sota_ref['tp']:,} -> {total_gt_covered:,})")
    print(f"  • Hard Frontend Dead Zone (Zero candidate @ conf>=0.02)  : {total_gt_lost_frontend:,} frames ({total_gt_lost_frontend / max(1, total_gt) * 100:.2f}% of all GT)")
    print(f"  • Maximum Realistic F1 Ceiling (Oracle-B)               : {oracle_b['f1']:.4f} (ΔF1 = +{delta_f1_b:.4f})")
    print(f"  • Maximum FP Reduction Potential (Algorithmic Noise)    : {sota_ref['fp'] - confirmed_birds_fp:,} FPs addressable")

    # ==============================================================================
    # PART 2: Score Stratification & Noise Competition (P1 Deep Salvage Payoff)
    # ==============================================================================
    print("\n" + "=" * 110)
    print(colorstr("bold", "SECTION 2: SCORE STRATIFICATION & NOISE COMPETITION (P1 Feasibility)"))
    print("=" * 110)
    print(f"{'Score Band':<25} | {'GT Covered (TP Pool)':<22} | {'Noise Candidates (FP Pool)':<26} | {'Signal-to-Noise Ratio (SNR)'}")
    print("-" * 110)

    bands = [
        ("Strong (>= 0.25)", "strong_ge_025"),
        ("Near-Thresh (0.22 ~ 0.25)", "near_thresh_022_to_025"),
        ("Salvage Zone (0.06 ~ 0.22)", "salvage_006_to_022"),
        ("Deep Weak (< 0.06)", "deep_weak_lt_006"),
    ]

    for band_label, key in bands:
        gt_cnt = gt_best_score_bands[key]
        noise_cnt = noise_score_bands[key]
        gt_pct = (gt_cnt / max(1, total_gt_covered)) * 100.0
        snr = gt_cnt / max(1, noise_cnt)
        print(f"{band_label:<25} | {gt_cnt:>6,} ({gt_pct:>5.1f}%)         | {noise_cnt:>8,}                   | {snr:>7.4f} (1 : {max(1, noise_cnt) / max(1, gt_cnt):.1f})")
    print("=" * 110)
    print(f"[INSIGHT]")
    print(f"  • Salvage Zone (0.06~0.22) harbors {gt_best_score_bands['salvage_006_to_022']:,} valid targets.")
    print(f"  • Deep Weak (<0.06) harbors {gt_best_score_bands['deep_weak_lt_006']:,} valid targets, but competes with {noise_score_bands['deep_weak_lt_006']:,} noise peaks.")
    print(f"    -> Deep salvage (th=0.035~0.05) MUST remain gated to confirmed active tracks to avoid clutter explosion.")

    # ==============================================================================
    # PART 3: Per-Sequence Breakdown & Hard Case Audit
    # ==============================================================================
    print("\n" + "=" * 110)
    print(colorstr("bold", "SECTION 3: PER-SEQUENCE FRONTEND COVERAGE & HARD CASE AUDIT"))
    print("=" * 110)
    print(f"{'Sequence Name':<28} | {'GT':<6} | {'Covered':<8} | {'Lost (Dead)':<11} | {'Coverage':<9} | {'Category / Diagnosis'}")
    print("-" * 110)

    hard_cases_summary = {}

    for seq_name in all_keys:
        if filter_seqs and not any(f in seq_name for f in filter_seqs):
            continue
        st = per_seq_stats[seq_name]
        cov_pct = st["coverage_rate"]

        # Tag category
        cat = "Nominal"
        if "wg2022_ir_020_split_03" in seq_name:
            cat = colorstr("red", "Hard Case 1 (Stop-Go & 312f Gap)")
            hard_cases_summary["HC1"] = st
        elif "DJI_0051_2" in seq_name:
            cat = colorstr("red", "Hard Case 2 (Parallax Clutter)")
            hard_cases_summary["HC2"] = st
        elif "wg2022_ir_011_split_03" in seq_name:
            cat = colorstr("yellow", "Hard Case 3 (Cold Sky Weak Pulses)")
            hard_cases_summary["HC3"] = st
        elif "DJI_0175_2" in seq_name:
            cat = colorstr("cyan", "Hard Case 4 (Attitude Darkening, Demo-OK)")
            hard_cases_summary["HC4"] = st
        elif "02_6321_0274-2773" in seq_name:
            cat = colorstr("red", "Hard Case 5 (Hover + Big Body Edge)")
            hard_cases_summary["HC5"] = st
        elif "01_4485" in seq_name or "wg2022_ir_020_split_07" in seq_name:
            cat = colorstr("magenta", "Confirmed Bird Clutter")
        elif cov_pct < 85.0:
            cat = colorstr("yellow", "Low Frontend Coverage (<85%)")

        cov_str = f"{cov_pct:>6.2f}%"
        if cov_pct < 80.0:
            cov_str = colorstr("red", cov_str)
        elif cov_pct < 90.0:
            cov_str = colorstr("yellow", cov_str)
        else:
            cov_str = colorstr("green", cov_str)

        print(f"{seq_name:<28} | {st['gt']:<6} | {st['covered']:<8} | {st['lost']:<11} | {cov_str:<18} | {cat}")

    print("=" * 110)

    # ==============================================================================
    # PART 4: Summary & Clear Recommendations
    # ==============================================================================
    print("\n" + "=" * 110)
    print(colorstr("bold", "SECTION 4: STRATEGIC DECISION MATRIX"))
    print("=" * 110)

    if oracle_b["f1"] >= 0.9500:
        conclusion = colorstr("green", f"STRONG ROOM FOR POST-PROCESSING BREAKTHROUGH (Oracle-B F1 = {oracle_b['f1']:.4f})")
        action = "Proceed aggressively with P1 (HC3 deep salvage) -> P2 (HC1 long stitch) -> P3 (HC5 dual expert)."
    elif oracle_b["f1"] >= 0.9300:
        conclusion = colorstr("yellow", f"MODERATE ROOM FOR POST-PROCESSING (Oracle-B F1 = {oracle_b['f1']:.4f})")
        action = "Post-processing can gain +0.010~0.020 F1. Focus exclusively on the 4 remaining Hard Cases."
    else:
        conclusion = colorstr("red", f"POST-PROCESSING CEILING IS NEARLY SATURATED (Oracle-B F1 = {oracle_b['f1']:.4f})")
        action = "Post-processing alone cannot move the needle. Must innovate on frontend alignment (P4 Parallax GMC)."

    print(f"Conclusion : {conclusion}")
    print(f"Next Action: {action}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
