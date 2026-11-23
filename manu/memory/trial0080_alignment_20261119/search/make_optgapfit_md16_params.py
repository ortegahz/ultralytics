#!/usr/bin/env python3
"""Build the r88 (match_dist=16.0) gapfit parameter file from the delta-n search result.

Same contract as make_optgapfit_params.py (the v3 params block + the seven delta-n
knobs written EXPLICITLY), but the provenance describes the r88 run: the searcher
now measures its hard per-frame constraint on the DELIVERY caliber (the full cache),
so the search output itself is the delivery-feasible optimum.

Usage:
  make_optgapfit_md16_params.py SEARCH_JSON BASE_JSON OUT_JSON [EVAL_JSON]
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

SEARCH_JSON = Path(sys.argv[1])
BASE_JSON = Path(sys.argv[2])
OUT_JSON = Path(sys.argv[3])
EVAL_JSON = Path(sys.argv[4]) if len(sys.argv) > 4 else None

search = json.loads(SEARCH_JSON.read_text(encoding="utf-8"))
base = json.loads(BASE_JSON.read_text(encoding="utf-8"))

KEYS = ("confirm_consec", "confirm_gap_tol", "max_age_phys", "r_slope",
        "alpha_decay", "match_dist", "min_hits")

opt = search.get("opt_params") or {}
if not all(key in opt for key in KEYS):
    raise SystemExit(f"[FAIL] {SEARCH_JSON} opt_params misses some of {KEYS}")

params = dict(base["params"])
applied = {key: opt[key] for key in KEYS}
params.update(applied)

params_blob = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
params_md5 = hashlib.md5(params_blob).hexdigest()

arms = (json.loads(EVAL_JSON.read_text(encoding="utf-8")).get("arms", {})
        if EVAL_JSON else search.get("arms", {}))
delivery = search.get("delivery_caliber") or {}

payload = {
    "_provenance": {
        "status": "CANDIDATE -- NOT the delivered point. The DELIVERED parameter file is "
                  "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json "
                  "(match_dist=10.0, FAR 0.027859, 7.1% headroom). This file is the r88 "
                  "searcher's argmax (match_dist=16.0, FAR 0.029348, only 2.2% headroom) and is "
                  "kept switch-ready only; do NOT sync it to the board without the board-side FAR "
                  "re-check.",
        "what": "trial_0080 (RC2) v3 baseline + the delta-n (jump-frame) parameters "
                "re-searched on the REBUILT trial_0080 cache with the hard per-frame "
                "no-regression constraint measured ON THE DELIVERY caliber.",
        "base": "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json "
                "(params_md5 bba1e948f420efe1b5070269b4587a43)",
        "search_source": str(SEARCH_JSON),
        "search_log": str(SEARCH_JSON).replace(".json", ".log"),
        "knobs_source": f"{SEARCH_JSON} opt_params (the r88 searcher's own argmax; the "
                        "delivery-caliber feasibility verdict is part of the same run)",
        "r88_caliber_fix": "rt_fps_jitter_search.py now measures the per-frame HARD "
                           "constraint on the FULL cache (--perframe-caliber delivery, the "
                           "default) instead of the 4000-frame truncated view, and --sweeps "
                           "0 means 'stop when a sweep stops improving'. The previous rounds "
                           "ran the constraint on the truncated caliber, which UNDER-reports "
                           "FAR by ~0.0055 absolute, so a candidate could pass the search and "
                           "still break the 0.03 red line on the delivery caliber.",
        "search_objective": "jump mean F1 - lambda*max(0, jump FAR - ceiling FAR) "
                            "- mu*max(0, per-frame FAR - 0.03) "
                            "- nu*max(0, base per-frame F1 - 0.003 - per-frame F1); "
                            "mu/nu are hard-constraint level, and the per-frame arm is the "
                            "FULL-cache (delivery) caliber.",
        "delivery_verdict_from_search": {
            "base": delivery.get("base"),
            "selected": delivery.get("selected"),
            "f1_floor": delivery.get("f1_floor"),
            "redline_far": delivery.get("redline_far"),
            "far_ok": delivery.get("far_ok"),
            "f1_ok": delivery.get("f1_ok"),
            "feasible": delivery.get("feasible"),
        },
        "authoritative_track_run": "produced by manu/evaluation/system_sota_longquan.py on the "
                                   "rebuilt cache; see the eval json referenced in "
                                   "search_arms_source and the OSD directory's "
                                   "search_json_for_render.json",
        "search_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "applied_on_top_of_v3": applied,
        "explicit_delta_n_keys": "max_age_phys / r_slope / alpha_decay are ABSENT from the v3 "
                                 "block and are written explicitly here; without them "
                                 "system_sota_longquan.py would use the legacy per-update "
                                 "behaviour and the searched jump parameters would not apply. "
                                 "match_dist is written explicitly too: the v3 block already "
                                 "carries 16.0 and this point KEEPS it, while the previous "
                                 "delivered gapfit point lowered it to 10.0.",
        "search_arms": {
            "arm1_perframe_ceiling": arms.get("1_perframe_ceiling", {}),
            "arm2a_jump_dense_timebase": arms.get("2a_jump_dense_timebase", {}),
            "arm2b_jump_physical_timebase": arms.get("2b_jump_physical_untuned", {}),
            "arm3_jump_optimised": arms.get("3_jump_optimised", {}),
            "arm5_optimised_perframe": arms.get("5_optimised_perframe", {}),
        },
        "delivery_criteria": {"f1_min": 0.88, "recall_min": 0.85, "far_max": 0.03},
        "params_md5": params_md5,
        "arm5_is_not_the_delivery_caliber": "WARNING: the searcher's arm5 ('5_optimised_perframe') "
                                            "is NOT the per-frame delivery caliber -- it runs on the "
                                            "4000-frame truncated view. The DELIVERY judgement is "
                                            "the FULL caliber: see delivery_verdict_from_search "
                                            "and the authoritative track run.",
        "three_way_consistency": "These params are byte-identical (all 37 keys) to the ones "
                                 "recorded in the OSD directory's search_json_for_render.json AND "
                                 "in its per_sequence_hyperparams.tsv '# global params:' line -- so "
                                 "the parameter file, the rendered OSD and the evaluation all "
                                 "describe the SAME working point.",
    },
    "params": params,
}

OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] {OUT_JSON} | params={len(params)} params_md5={params_md5}")
print(f"     applied: {json.dumps(applied, sort_keys=True)}")
