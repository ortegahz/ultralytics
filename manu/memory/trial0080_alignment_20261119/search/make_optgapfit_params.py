#!/usr/bin/env python3
"""Build the v3-optgapfit parameter file from the delta-n search result.

The v3 parameter block does NOT carry max_age_phys / r_slope / alpha_decay, so the
delta-n knobs must be written explicitly -- otherwise system_sota_longquan.py falls
back to .get(...) defaults and the searched jump parameters would silently not apply.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

SEARCH_JSON = Path(sys.argv[1])
BASE_JSON = Path(sys.argv[2])
OUT_JSON = Path(sys.argv[3])
# Optional 4th argument: a JSON file holding the SEVEN delta-n knobs of the winner.
# The search json's own "opt_params" is the searcher's feasibility fallback and may
# be the (useless) v3 base -- the winner is selected against the FULL-caliber
# delivery judgement, so the knobs are passed in explicitly.
KNOBS_JSON = Path(sys.argv[4]) if len(sys.argv) > 4 else None
# Optional 5th argument: the three-arm report produced WITH the delivered
# parameters.  The search json's own "arms" belong to the search's candidate
# (which may be the feasibility fallback), so they must not be quoted here.
EVAL_JSON = Path(sys.argv[5]) if len(sys.argv) > 5 else None

search = json.loads(SEARCH_JSON.read_text(encoding="utf-8"))
base = json.loads(BASE_JSON.read_text(encoding="utf-8"))

KEYS = ("confirm_consec", "confirm_gap_tol", "max_age_phys", "r_slope",
        "alpha_decay", "match_dist", "min_hits")

if KNOBS_JSON:
    knobs_doc = json.loads(KNOBS_JSON.read_text(encoding="utf-8"))
    opt = knobs_doc.get("_knobs") or knobs_doc
    if not all(k in opt for k in KEYS):
        raise SystemExit(f"[FAIL] {KNOBS_JSON} misses some of {KEYS}")
    source = f"{KNOBS_JSON} (winner selected on the FULL-caliber delivery judgement)"
else:
    opt = search.get("opt_params")
    source = f"{SEARCH_JSON} opt_params"
    if not opt:
        raise SystemExit(f"[FAIL] {SEARCH_JSON} has no opt_params")

params = dict(base["params"])
applied = {}
for key in KEYS:
    if key in opt:
        params[key] = opt[key]
        applied[key] = opt[key]

# The parameter block is what the delivery actually runs: its md5 is the fingerprint.
params_blob = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
params_md5 = hashlib.md5(params_blob).hexdigest()

arms = (json.loads(EVAL_JSON.read_text(encoding="utf-8")).get("arms", {})
        if EVAL_JSON else search.get("arms", {}))
summary = {
    "arm1_perframe_ceiling": arms.get("1_perframe_ceiling", {}),
    "arm2a_jump_dense_timebase": arms.get("2a_jump_dense_timebase", {}),
    "arm2b_jump_physical_timebase": arms.get("2b_jump_physical_untuned", {}),
    "arm3_jump_optimised": arms.get("3_jump_optimised", {}),
    "arm5_optimised_perframe": arms.get("5_optimised_perframe", {}),
}

payload = {
    "_provenance": {
        "what": "trial_0080 (RC2) v3 baseline + the delta-n (jump-frame) parameters "
                "re-searched on the REBUILT trial_0080 cache, with a hard "
                "per-frame no-regression constraint.",
        "base": "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json "
                "(params_md5 bba1e948f420efe1b5070269b4587a43)",
        "search_source": str(SEARCH_JSON),
        "knobs_source": source,
        "two_stage_selection": "The searcher maximises the JUMP arm on the 4000-frame TRUNCATED "
                               "caliber; that caliber systematically UNDER-reports FAR by about "
                               "+0.005..0.006 absolute, so the DELIVERY judgement (FAR <= 0.03 on "
                               "the FULL 165,223-frame caliber, per-frame F1 regression reported) "
                               "is applied afterwards and is what actually selects these knobs. "
                               "The searcher's own objective-best (max_age_phys=10 / match_dist=16.0, "
                               "identical in run 2 and run 3) scores FAR 0.0352 on the full caliber "
                               "and was therefore REJECTED. See the selection evidence file.",
        "selection_evidence": "/mnt/data/siping/runs/trial0080_onnx/search/final_selection.json",
        "search_objective": "jump mean F1 - lambda*max(0, jump FAR - ceiling FAR) "
                            "- mu*max(0, per-frame FAR - 0.03) "
                            "- nu*max(0, base per-frame F1 - 0.003 - per-frame F1); "
                            "the DELIVERED set is then the best FEASIBLE candidate "
                            "(per-frame FAR <= 0.03 AND per-frame F1 >= base - 0.003), "
                            "cross-checked on the FULL-caliber delivery judgement "
                            "(FAR <= 0.03, per-frame F1 regression reported).",
        "search_arms_source": str(EVAL_JSON) if EVAL_JSON else str(SEARCH_JSON),
        "search_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "applied_on_top_of_v3": applied,
        "explicit_delta_n_keys": "max_age_phys / r_slope / alpha_decay are ABSENT from the v3 "
                                 "block and are written explicitly here; without them "
                                 "system_sota_longquan.py would use the legacy per-update "
                                 "behaviour and the searched jump parameters would not apply.",
        "search_arms": summary,
        "delivery_criteria": {"f1_min": 0.88, "recall_min": 0.85, "far_max": 0.03},
    },
    "params": params,
}
payload["_provenance"]["params_md5"] = params_md5

OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] {OUT_JSON}")
print(f"[OK] params_md5 = {params_md5}")
print(f"[OK] applied = {json.dumps(applied, sort_keys=True)}")
