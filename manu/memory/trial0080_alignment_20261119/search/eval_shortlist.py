#!/usr/bin/env python3
"""Build the shortlist parameter files (v3 base + delta-n overrides)."""
import json
from pathlib import Path

REPO = Path("/mnt/data/siping/runs/npu20ms/nano_search/repo")
BASE = REPO / "manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"
OUT = Path("/mnt/data/siping/runs/trial0080_onnx/eval/shortlist")
OUT.mkdir(parents=True, exist_ok=True)

base = json.loads(BASE.read_text(encoding="utf-8"))

# name -> the seven delta-n knobs
SHORTLIST = {
    "cc2_md10":  dict(confirm_consec=2, confirm_gap_tol=8, max_age_phys=None, r_slope=5.0, alpha_decay=1.0, match_dist=10.0, min_hits=3),
    "cc3_md10":  dict(confirm_consec=3, confirm_gap_tol=8, max_age_phys=None, r_slope=5.0, alpha_decay=1.0, match_dist=10.0, min_hits=3),
    "cc4_md10":  dict(confirm_consec=4, confirm_gap_tol=8, max_age_phys=None, r_slope=5.0, alpha_decay=1.0, match_dist=10.0, min_hits=3),
    "cc5_md10":  dict(confirm_consec=5, confirm_gap_tol=8, max_age_phys=None, r_slope=5.0, alpha_decay=1.0, match_dist=10.0, min_hits=3),
    "cc6_md10":  dict(confirm_consec=6, confirm_gap_tol=8, max_age_phys=None, r_slope=5.0, alpha_decay=1.0, match_dist=10.0, min_hits=3),
    "cc7_md10":  dict(confirm_consec=7, confirm_gap_tol=8, max_age_phys=None, r_slope=5.0, alpha_decay=1.0, match_dist=10.0, min_hits=3),
    "cc3_md16":  dict(confirm_consec=3, confirm_gap_tol=8, max_age_phys=None, r_slope=5.0, alpha_decay=1.0, match_dist=16.0, min_hits=3),
    "old_gapfit": dict(confirm_consec=2, confirm_gap_tol=6, max_age_phys=10, r_slope=3.5, alpha_decay=1.0, match_dist=16.0, min_hits=3),
}

for name, knobs in SHORTLIST.items():
    params = dict(base["params"])
    params.update(knobs)
    payload = {
        "_shortlist": name,
        "_note": "scratch candidate for the full-caliber selection; "
                 "max_age_phys/r_slope/alpha_decay are written EXPLICITLY "
                 "(the v3 block has no such keys -> system_sota_longquan.py "
                 "would otherwise fall back to None/0.0/1.0)",
        "_knobs": knobs,
        "params": params,
    }
    p = OUT / f"params_{name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {p}")
print(f"[OK] {len(SHORTLIST)} candidates")
