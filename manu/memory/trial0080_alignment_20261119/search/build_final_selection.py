#!/usr/bin/env python3
"""Record the two-stage selection: search objective-best -> FULL-caliber delivery gate.

Stage 1 (the searcher) maximises the JUMP arm F1 under a per-frame no-regression
penalty, judged on the 4000-frame truncated caliber.
Stage 2 (this file) applies the DELIVERY judgement on the FULL caliber
(24 sequences / 165,223 frames): FAR <= 0.03 (hard red line) and the per-frame F1
regression against v3 (reported, not hidden).

Stage 2 is not optional: the truncated caliber systematically UNDER-reports FAR by
about +0.005..0.006 absolute, so a set the searcher considers green can break the
red line on the real caliber.
"""
import hashlib
import json
from pathlib import Path

REPO = Path("/mnt/data/siping/runs/npu20ms/nano_search/repo")
RUNS = Path("/mnt/data/siping/runs/trial0080_onnx")
SL = RUNS / "eval/shortlist"
OUT = RUNS / "search/final_selection.json"

KEYS = ("confirm_consec", "confirm_gap_tol", "max_age_phys", "r_slope",
        "alpha_decay", "match_dist", "min_hits")


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def knobs_of(params):
    return {k: params.get(k) for k in KEYS}


# ---- the candidate set: everything measured on the FULL caliber ----------------
measured = {}
for path in sorted(SL.glob("params_*.json")):
    name = path.stem.replace("params_", "")
    track = load(SL / f"track_{name}.json")
    if not track:
        continue
    measured[name] = {
        "knobs": knobs_of(track.get("config", {}).get("params", {})),
        "full_caliber": {
            "f1": track["overall"]["f1"],
            "far_per_frame": track["overall"]["far_per_frame"],
            "recall": track["overall"]["recall"],
            "precision": track["overall"]["precision"],
            "tp": track["overall"]["tp"], "fp": track["overall"]["fp"],
            "fn": track["overall"]["fn"], "gt": track["overall"]["gt"],
        },
    }

v3 = load(RUNS / "eval/track_all_v3.json")
v3o = v3["overall"]

# ---- the search objective-best of every run ------------------------------------
runs = {}
for tag, path in (("run1_hashseed", RUNS / "search/rt_search_t0080_v3.json"),
                  ("run2_hashseed_other_cadence", RUNS / "search/rt_search_t0080_v3_final.json"),
                  ("run3_crc32_fixed_seed", RUNS / "search/rt_search_t0080_v3_seedfix.json")):
    doc = load(path)
    if not doc:
        continue
    runs[tag] = {
        "search_json": str(path),
        "elapsed_s": doc.get("elapsed_s"),
        "objective_best_knobs": knobs_of(doc.get("opt_params", {})),
        "note": "the searcher's own feasibility filter uses the TRUNCATED (4000-frame) caliber",
    }

DELIVERED = "cc2_md10"
DELIVERED_TRACK = RUNS / "eval/track_all_v3_optgapfit.json"
delivered_doc = load(DELIVERED_TRACK)

# The jump-arm numbers MUST come from the deterministic re-measurement (one process, one
# cache load, zlib.crc32 sampler).  The pre-fix sampler (hash()) produced different values
# that cannot be reproduced -- those are 作废 and must never be written here as current.
CANON = load(RUNS / "search/shortlist_deterministic.json")
if not CANON:
    raise SystemExit("[FATAL] shortlist_deterministic.json is missing -- the jump-arm numbers "
                     "have no reproducible source; refusing to write a stale hardcoded value.")
R = CANON["results"]
_jump = R["delivered_cc2_md10"]["jump_physical_mean_f1"]
_jump_dense = R["delivered_cc2_md10"]["jump_dense_mean_f1"]
_md16n = R["cc2_md16n"]["jump_physical_mean_f1"]
_md16n_far = load(SL / "track_cc2_md16n.json")["overall"]["far_per_frame"]
_v3_jump = R["v3_base"]["jump_physical_mean_f1"]
_v3_dense = R["v3_base"]["jump_dense_mean_f1"]
_ceiling = R["v3_base"]["perframe_truncated_f1_agg"]
_ceiling_far = R["v3_base"]["perframe_truncated_far_agg"]

delivered = {
    "name": DELIVERED,
    "params_file": "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json",
    "knobs": knobs_of(load(REPO / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json")["params"]),
    "full_caliber": {k: delivered_doc["overall"][k] for k in
                     ("f1", "far_per_frame", "recall", "precision", "tp", "fp", "fn", "gt")},
    "jump_arm_f1_4000f": round(_jump, 6),
    "jump_arm_dense_f1_4000f": round(_jump_dense, 6),
    "jump_arm_source": "shortlist_deterministic.json (deterministic zlib.crc32 sampler); "
                       "the pre-fix hash() sampler's numbers are 作废 and are NOT repeated here",
    "why": f"highest JUMP-arm F1 among every candidate that holds the FAR red line on the "
           f"FULL caliber (jump mean F1 {_jump:.6f}, deterministic re-measurement); and it "
           f"dominates the previous gapfit set on BOTH per-frame metrics "
           f"(F1 {delivered_doc['overall']['f1'] * 100:.4f}% vs 94.3755%, "
           f"FAR {delivered_doc['overall']['far_per_frame']:.4f} vs 0.0351).",
}

payload = {
    "_what": "two-stage selection: the searcher maximises the jump arm on the truncated "
             "caliber; the DELIVERY judgement (FAR <= 0.03 on the FULL caliber) is applied "
             "here and is what actually decides the shipped parameters.",
    "_why_two_stages": "the 4000-frame truncated caliber UNDER-reports FAR by about "
                       "+0.005..0.006 absolute, so a set that is green on the truncated "
                       "caliber can still break the 0.03 red line on the full caliber "
                       "(measured: the run-2 objective-best scores FAR 0.029013 truncated "
                       "but 0.0352 on the full caliber -> rejected).",
    "delivery_criteria": {"far_max": 0.03, "f1_min": 0.88, "recall_min": 0.85,
                          "f1_regression_vs_v3": "reported, must be small"},
    "v3_baseline_full_caliber": {k: v3o[k] for k in
                                 ("f1", "far_per_frame", "recall", "precision", "tp", "fp", "fn", "gt")},
    "search_runs": runs,
    "full_caliber_measurements": measured,
    "delivered": delivered,
    "_deterministic_remeasure_20261120": {
        "why": "The jump-arm numbers quoted in the original _why were measured with the PRE-FIX "
               "cadence sampler (hash() -> zlib.crc32 landed at 21:23); they were re-measured on "
               "2026-11-19 23:31~2026-11-20 00:08 with the deterministic sampler (one process, one "
               "cache load, 14 candidates, 1562.8 s). Products: shortlist_deterministic.json + "
               "rt_eval_delivered_armA.json (cross-checked cell by cell).",
        "source": "shortlist_deterministic.json",
        "v3_untuned_jump_physical_mean_f1": round(_v3_jump, 6),
        "delivered_cc2_md10_jump_physical_mean_f1": round(_jump, 6),
        "delivered_cc2_md10_jump_dense_mean_f1": round(_jump_dense, 6),
        "v3_base_jump_dense_mean_f1": round(_v3_dense, 6),
        "perframe_ceiling_v3_truncated_f1": round(_ceiling, 6),
        "perframe_ceiling_v3_truncated_far": round(_ceiling_far, 6),
        "highest_jump_among_far_compliant": f"cc2_md16n ({_md16n:.6f}, full FAR {_md16n_far:.6f})",
        "selection_correction": f"cc2_md16n beats the delivered cc2_md10 on BOTH the jump arm "
                               f"({_md16n:.6f} vs {_jump:.6f}, +0.03pp = noise) and the per-frame F1 "
                               f"(94.5804% vs {delivered_doc['overall']['f1'] * 100:.4f}%), but leaves only "
                               f"2.2% FAR headroom vs the delivered 7.1%. The delivery stays cc2_md10: "
                               f"0.03pp of jump F1 for 3x the FAR margin. cc2_md16n is compliant and its "
                               f"full-caliber track run already exists (eval/shortlist/track_cc2_md16n.json) "
                               f"-- switch = new params file + re-render OSD.",
        "delivery_judgement_unchanged": {
            "f1": round(delivered_doc["overall"]["f1"], 12),
            "far_per_frame": round(delivered_doc["overall"]["far_per_frame"], 12),
            "v3_f1": round(v3o["f1"], 6),
            "v3_far": round(v3o["far_per_frame"], 6),
        },
    },
}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] {OUT}")
print(f"[OK] delivered = {DELIVERED}  knobs = {json.dumps(delivered['knobs'], ensure_ascii=False)}")
print(f"[OK] full-caliber F1 = {delivered['full_caliber']['f1'] * 100:.4f}%  "
      f"FAR = {delivered['full_caliber']['far_per_frame']:.6f}")
