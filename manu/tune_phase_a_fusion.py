#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Phase-A Coordinate-Ascent Tuner: Deep Salvage + Elastic Stitch + Hover-Lock + Dual-Expert.

Strategy: lock the golden SOTA core (th_base=0.22 / th_salvage=0.06 / th_ground=0.35 /
stitch_gap=4 / infill=3 / min_hits_infill=5 / pruner disp=2.0 / var=0.5 / hits=8) and
coordinate-ascent ONLY the new Phase-A dimensions, stage by stage:

  Stage 0: Legacy reference (Phase-A all OFF) -> must reproduce F1=0.9209 (bbox mode)
  Stage 1: Deep Salvage   (th_deep x min_hits_deep)
  Stage 2: Elastic Stitch (stitch_long_gap x vel_diff)
  Stage 3: Hover-Lock     (hover_vel x hover_infill_gap x coast_frames)
  Stage 4: Sky Pruner     (min_rigid_disp_sky)
  Stage 5: Dual Expert    (size_gate x bbox_conf, requires --bbox-cache)

Each stage keeps the best value of the previous stage (greedy). Final combined config is
re-evaluated and reported alongside the legacy reference.

Usage on Server:
    python manu/tune_phase_a_fusion.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        [--bbox-cache runs/gmc_eval/uav_median_bbox_trial0028_cache.pkl] \
        [--sequences wg011,wg020,DJI_0051,02_6321]   # fast hard-case iteration first
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr
from manu.eval_bidirectional_track_fusion import (
    evaluate_sequence_bidirectional,
    extract_seq_name,
)

# Golden SOTA core parameters (Trial 0474 + Smoothing + Rigid Pruning, bbox mode)
GOLDEN = {
    "th_base": 0.22,
    "th_salvage": 0.06,
    "th_ground": 0.35,
    "min_hits": 3,
    "max_age": 3,
    "match_dist": 12.0,
    "max_match_dist": 18.0,
    "instant_conf": 0.25,
    "min_disp": 2.5,
    "sky_ratio": 0.60,
    "stitch_gap": 4,
    "infill_gap": 3,
    "min_hits_infill": 5,
    "min_rigid_disp": 2.0,
    "max_rigid_var": 0.50,
    "min_hits_prune": 8,
    "dist_thresh": 8.0,
    "match_mode": "bbox",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase-A Coordinate-Ascent Tuner")
    parser.add_argument("--cache-file", type=str, default="runs/gmc_eval/uav_median_trial0474_cache.pkl")
    parser.add_argument("--bbox-cache", type=str, default="", help="Optional bbox expert cache (.pkl)")
    parser.add_argument("--sequences", type=str, default="", help="Comma-separated sequence filter for fast iteration")
    parser.add_argument("--stages", type=str, default="1,2,3,4,5", help="Which stages to run (default: all)")
    return parser.parse_args()


def resolve_cache(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        for cand in [PROJECT_ROOT / path, Path("/tmp/pycharm_project_10ae9e2e") / path]:
            if cand.exists():
                return cand
    return path


def make_configs(pa: Dict) -> Tuple[Dict, Dict]:
    tracker_config = {
        "max_age": GOLDEN["max_age"],
        "min_hits": GOLDEN["min_hits"],
        "match_dist": GOLDEN["match_dist"],
        "max_match_dist": GOLDEN["max_match_dist"],
        "min_track_score": 0.08,
        "instant_conf": GOLDEN["instant_conf"],
        "min_displacement": GOLDEN["min_disp"],
        "sky_ratio": GOLDEN["sky_ratio"],
        "img_h": 640,
        "th_deep_salvage": pa["th_deep"],
        "min_hits_deep_salvage": pa["min_hits_deep"],
    }
    smoother_config = {
        "stitch_max_gap": GOLDEN["stitch_gap"],
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": GOLDEN["min_hits_infill"],
        "max_infill_gap": GOLDEN["infill_gap"],
        "min_track_hits": GOLDEN["min_hits"],
        "min_track_score": 0.08,
        "instant_conf": GOLDEN["instant_conf"],
        "min_rigid_displacement": GOLDEN["min_rigid_disp"],
        "max_rigid_variance": GOLDEN["max_rigid_var"],
        "min_hits_for_prune": GOLDEN["min_hits_prune"],
        "stitch_long_gap": pa["stitch_long_gap"],
        "stitch_max_vel_diff": pa["stitch_vel_diff"],
        "hover_vel_thresh": pa["hover_vel"],
        "hover_infill_gap": pa["hover_infill_gap"],
        "coast_max_frames": pa["coast_frames"],
        "min_hits_hover": pa["min_hits_hover"],
        "hover_sky_only": pa["hover_sky_only"],
        "min_rigid_disp_sky": pa["min_rigid_disp_sky"],
    }
    return tracker_config, smoother_config


def run_eval(
    seq_records: Dict[str, List[Dict]],
    seq_filter: List[str],
    pa: Dict,
    bbox_records_by_name: Optional[Dict[str, Dict]],
    size_gate: float,
    bbox_conf: float,
) -> Dict[str, float]:
    tracker_config, smoother_config = make_configs(pa)
    tot = {"tp": 0, "fp": 0, "gt": 0}
    for s_name, recs in seq_records.items():
        if seq_filter and not any(f in s_name for f in seq_filter):
            continue
        res = evaluate_sequence_bidirectional(
            records=recs,
            dist_thresh=GOLDEN["dist_thresh"],
            th_base=GOLDEN["th_base"],
            th_salvage=GOLDEN["th_salvage"],
            th_ground=GOLDEN["th_ground"],
            sky_ratio=GOLDEN["sky_ratio"],
            img_h=640,
            tracker_config=tracker_config,
            smoother_config=smoother_config,
            match_mode=GOLDEN["match_mode"],
            bbox_records_by_name=bbox_records_by_name if pa["dual_expert"] else None,
            size_gate=size_gate,
            bbox_conf_min=bbox_conf,
            bbox_dedup_radius=20.0,
        )
        m = res["bidirectional"]
        tot["tp"] += int(m["tp"])
        tot["fp"] += int(m["fp"])
        tot["gt"] += int(m["gt"])
    rec = (tot["tp"] / max(1, tot["gt"])) * 100.0
    prec = (tot["tp"] / max(1, tot["tp"] + tot["fp"])) * 100.0
    f1 = 2 * rec * prec / max(1e-6, rec + prec) / 100.0
    return {"tp": tot["tp"], "fp": tot["fp"], "gt": tot["gt"], "recall": rec, "precision": prec, "f1": f1}


def fmt_pa(pa: Dict) -> str:
    return (
        f"deep={pa['th_deep']:.3f}/{pa['min_hits_deep']} stitchL={pa['stitch_long_gap']}/{pa['stitch_vel_diff']:.1f} "
        f"hover={pa['hover_vel']:.2f}/{pa['hover_infill_gap']}/{pa['coast_frames']}/sky{pa['hover_sky_only']} "
        f"skyPrune={pa['min_rigid_disp_sky']} dual={pa['dual_expert']}"
    )


def main():
    args = parse_args()
    cache_path = resolve_cache(args.cache_file)
    if not cache_path.exists():
        print(colorstr("red", f"[ERROR] Cache not found: {args.cache_file}"))
        sys.exit(1)

    print(colorstr("bold", colorstr("green", f"\n>>> Loading heatmap cache from: {cache_path}")))
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} image predictions.")

    bbox_records_by_name = None
    if args.bbox_cache:
        bbox_path = resolve_cache(args.bbox_cache)
        if bbox_path.exists():
            with open(bbox_path, "rb") as f:
                bbox_records = pickle.load(f)
            bbox_records_by_name = {Path(r["im_name"]).stem: r for r in bbox_records}
            print(colorstr("cyan", f"[INFO] Loaded Bbox expert cache: {len(bbox_records)} frames"))
        else:
            print(colorstr("red", f"[WARN] Bbox cache not found: {args.bbox_cache} (Stage 5 skipped)"))

    seq_records: Dict[str, List[Dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    seq_filter = [s.strip() for s in args.sequences.split(",") if s.strip()]
    stages = {int(s) for s in args.stages.split(",") if s.strip()}

    # Current Phase-A state (starts all OFF = legacy)
    pa = {
        "th_deep": 0.0,
        "min_hits_deep": 3,
        "stitch_long_gap": 0,
        "stitch_vel_diff": 4.0,
        "hover_vel": 0.0,
        "hover_infill_gap": 15,
        "coast_frames": 0,
        "min_hits_hover": 8,
        "hover_sky_only": 1,
        "min_rigid_disp_sky": None,
        "dual_expert": False,
    }

    print("\n" + "=" * 118)
    print(colorstr("bold", "STAGE 0: LEGACY REFERENCE (Phase-A all OFF, golden SOTA core)"))
    print("=" * 118)
    t0 = time.time()
    base_m = run_eval(seq_records, seq_filter, pa, None, 40.0, 0.40)
    print(
        f"   Legacy: F1={base_m['f1']:.4f} | Recall={base_m['recall']:.2f}% (TP {base_m['tp']}) | "
        f"Prec={base_m['precision']:.2f}% (FP {base_m['fp']}) | {time.time() - t0:.1f}s"
    )

    best_f1 = base_m["f1"]

    def sweep(stage_name: str, candidates: List[Dict], use_bbox: bool = False):
        nonlocal best_f1, pa
        print("\n" + "=" * 118)
        print(colorstr("bold", f"{stage_name} (current best F1={best_f1:.4f})"))
        print("=" * 118)
        t_s = time.time()
        for i, cand in enumerate(candidates, 1):
            trial_pa = dict(pa)
            trial_pa.update(cand)
            m = run_eval(
                seq_records, seq_filter, trial_pa,
                bbox_records_by_name if use_bbox else None,
                cand.get("size_gate", 40.0), cand.get("bbox_conf", 0.40),
            )
            mark = ""
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                pa = trial_pa
                mark = colorstr("bold", colorstr("green", "  <-- NEW BEST"))
            print(
                f"   [{i:02d}/{len(candidates):02d}] {fmt_pa(trial_pa)} | F1={m['f1']:.4f} | "
                f"Rec={m['recall']:.2f}% (TP {m['tp']}) | Prec={m['precision']:.2f}% (FP {m['fp']}){mark}"
            )
        print(f"   Stage done in {time.time() - t_s:.1f}s | Best F1 now: {best_f1:.4f}")

    # Stage 1: Deep Salvage
    if 1 in stages:
        cands = [
            {"th_deep": d, "min_hits_deep": h}
            for d in [0.03, 0.035, 0.04, 0.05]
            for h in [2, 3, 4]
        ]
        sweep("STAGE 1: TRACK-GATED DEEP SALVAGE (sky, mature tracks only)", cands)

    # Stage 2: Elastic Long Stitching
    if 2 in stages:
        cands = [
            {"stitch_long_gap": g, "stitch_vel_diff": v}
            for g in [8, 12, 16]
            for v in [3.0, 5.0]
        ]
        sweep("STAGE 2: ELASTIC LONG STITCHING (velocity-coherent extended gaps)", cands)

    # Stage 3: Hover-Lock (infill + coast), sky-only vs any-region
    if 3 in stages:
        cands = [
            {"hover_vel": hv, "hover_infill_gap": g, "coast_frames": c, "hover_sky_only": so}
            for hv in [0.6, 1.0]
            for g in [10, 15]
            for c in [10, 20]
            for so in [1, 0]
        ]
        sweep("STAGE 3: HOVER-LOCK COASTING (quasi-static extended infill + trailing coast)", cands)

    # Stage 4: Sky-aware rigid pruner
    if 4 in stages:
        cands = [
            {"min_rigid_disp_sky": 1.0},
            {"min_rigid_disp_sky": 0.5},
            {"min_rigid_disp_sky": 0.0},
        ]
        sweep("STAGE 4: SKY-AWARE RIGID PRUNER (exempt real sky hover targets from bad-pixel purge)", cands)

    # Stage 5: Dual Expert
    if 5 in stages and bbox_records_by_name is not None:
        cands = [
            {"dual_expert": True, "size_gate": sg, "bbox_conf": bc}
            for sg in [30.0, 40.0, 60.0]
            for bc in [0.30, 0.45]
        ]
        sweep("STAGE 5: SIZE-GATED DUAL-EXPERT (large-body Bbox fusion)", cands, use_bbox=True)
    elif 5 in stages:
        print(colorstr("yellow", "\n[SKIP] Stage 5 skipped (no --bbox-cache provided)."))

    # Final combined evaluation
    print("\n" + "=" * 118)
    print(colorstr("bold", colorstr("magenta", "FINAL COMBINED PHASE-A CONFIGURATION")))
    print("=" * 118)
    final_m = run_eval(
        seq_records, seq_filter, pa,
        bbox_records_by_name if pa["dual_expert"] else None,
        pa.get("size_gate", 40.0), pa.get("bbox_conf", 0.40),
    )
    print(f"   Config : {fmt_pa(pa)}")
    print(
        f"   Legacy : F1={base_m['f1']:.4f} | Recall={base_m['recall']:.2f}% (TP {base_m['tp']}) | Prec={base_m['precision']:.2f}% (FP {base_m['fp']})"
    )
    print(
        f"   PhaseA : F1={final_m['f1']:.4f} | Recall={final_m['recall']:.2f}% (TP {final_m['tp']}) | Prec={final_m['precision']:.2f}% (FP {final_m['fp']})"
    )
    delta = final_m["f1"] - base_m["f1"]
    verdict = colorstr("bold", colorstr("green", f"ΔF1 = {delta:+.4f}")) if delta > 0 else colorstr("bold", colorstr("red", f"ΔF1 = {delta:+.4f}"))
    print(f"   Verdict: {verdict}  (TP {final_m['tp'] - base_m['tp']:+d}, FP {final_m['fp'] - base_m['fp']:+d})")
    print("=" * 118 + "\n")


if __name__ == "__main__":
    main()
