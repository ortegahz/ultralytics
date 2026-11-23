#!/usr/bin/env python3
"""Re-measure every shortlist candidate on the DETERMINISTIC cadence sampler.

Why: the shortlist jump-F1 column in the delivery docs was measured BEFORE the
`hash()` -> `zlib.crc32` sampler fix, i.e. under a per-process randomised cadence
sample.  This script loads the cache ONCE and re-measures, for every candidate:
  * jump arm, physical time base (the board's real cadence)  -> _jump_mean_f1
  * jump arm, dense time base (what today's SDK does)        -> dense mean F1
  * per-frame arm on the same 4000-frame truncated caliber   -> truncated F1/FAR
so the delivered row and the shortlist table share one reproducible sampler.
"""
import argparse
import json
import time
from pathlib import Path

import manu.evaluation.rt_fps_jitter_search as S

SHORTLIST_DIR = Path("/mnt/data/siping/runs/trial0080_onnx/eval/shortlist")
OUT = Path("/mnt/data/siping/runs/trial0080_onnx/search")
BASE = Path("/mnt/data/siping/runs/npu20ms/nano_search/repo/manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json")
DELIV = Path("/mnt/data/siping/runs/npu20ms/nano_search/repo/manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="/mnt/data/siping/runs/trial0080_onnx/ms_cache")
    parser.add_argument("--labels-root", default="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=4000)
    parser.add_argument("--scenarios", default="s1,s2,s3,s4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out-json", default=str(OUT / "shortlist_deterministic.json"))
    args = parser.parse_args()
    # prepare() reads these two directly from the namespace.
    args.all_seqs = True
    args.seqs = ""

    scenarios = [s for s in args.scenarios.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    started = time.time()
    sequences = S.prepare(args)
    print(f"[setup] {len(sequences)} sequences prepared in {time.time() - started:.1f}s", flush=True)

    import multiprocessing as mp

    pool = mp.get_context("fork").Pool(args.workers)

    candidates: dict[str, dict] = {}
    candidates["v3_base"] = json.loads(BASE.read_text(encoding="utf-8"))["params"]
    candidates["delivered_cc2_md10"] = json.loads(DELIV.read_text(encoding="utf-8"))["params"]
    for path in sorted(SHORTLIST_DIR.glob("params_*.json")):
        candidates[path.stem.replace("params_", "")] = json.loads(path.read_text(encoding="utf-8"))["params"]

    results: dict[str, dict] = {}
    for name, params in candidates.items():
        t0 = time.time()
        phys = S.evaluate_params(params, sequences, scenarios, seeds, pool)
        pf = S.evaluate_params(params, sequences, ["perframe"], [seeds[0]], pool)
        dense = None
        if name in ("v3_base", "delivered_cc2_md10"):
            dense = S.evaluate_params(params, sequences, scenarios, seeds, pool, dense_time_base=True)
        row = {
            "knobs": {k: params.get(k) for k in
                      ("confirm_consec", "confirm_gap_tol", "max_age_phys", "r_slope",
                       "alpha_decay", "match_dist", "min_hits")},
            "jump_physical_mean_f1": phys["summary"]["_jump_mean_f1"],
            "jump_physical_mean_far": phys["summary"]["_jump_mean_far"],
            "jump_physical_worst_f1": phys["summary"]["_jump_worst_f1"],
            "jump_physical_per_scenario": {
                s: {"f1": phys["summary"][s]["f1_delivered_agg"],
                    "far": phys["summary"][s]["far_delivered_agg"]} for s in scenarios},
            "perframe_truncated_f1_agg": pf["summary"]["perframe"]["f1_delivered_agg"],
            "perframe_truncated_far_agg": pf["summary"]["perframe"]["far_delivered_agg"],
            "perframe_truncated_f1_mean": pf["summary"]["perframe"]["f1_delivered"],
            "perframe_truncated_far_mean": pf["summary"]["perframe"]["far_delivered"],
        }
        if dense is not None:
            row["jump_dense_mean_f1"] = dense["summary"]["_jump_mean_f1"]
            row["jump_dense_per_scenario"] = {
                s: {"f1": dense["summary"][s]["f1_delivered_agg"],
                    "far": dense["summary"][s]["far_delivered_agg"]} for s in scenarios}
        results[name] = row
        print(f"[{name}] jump_physical_mean_f1={row['jump_physical_mean_f1']:.6f} "
              f"perframe_trunc_F1={row['perframe_truncated_f1_agg']:.6f} "
              f"perframe_trunc_FAR={row['perframe_truncated_far_agg']:.6f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    payload = {
        "meta": {
            "sampler": "zlib.crc32 deterministic (post-fix)",
            "cache_root": args.cache_root,
            "max_frames": args.max_frames,
            "scenarios": scenarios,
            "seeds": seeds,
            "elapsed_s": time.time() - started,
            "note": "all numbers here come from ONE process, one cache load, one sampler",
        },
        "results": results,
    }
    Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out_json}  ({time.time() - started:.1f}s total)")


if __name__ == "__main__":
    main()
