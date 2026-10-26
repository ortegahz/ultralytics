#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Audit the provenance of a multi-scale heatmap cache directory.

Why
---
A cache row is a *pure function* of (weights, feature set, build parameters).  Two rows built
from different feature semantics look identical on disk unless the build stamps its identity
into ``meta``:

* ``source`` only says ``"features"`` vs ``"raw_frames"`` -- the FPV chained-affine feature set
  and the official per-lag GMC fit both write ``"features"``;
* ``hm_weights`` / ``use_p0_highway`` identify the model, but not the features.

Mixing the two silently turns a full-set metric into a weighted average of two distributions
and makes the difference unattributable.  This tool exists to make that failure visible
*before* a number is published.

What it checks
--------------
For every ``<sequence>.pkl`` under a cache root it reads the ``meta`` block and reports the
identity fields.  It then flags:

1. ``mixed_source``     -- the sequences do not agree on the feature source;
2. ``missing_features_root`` -- the cache predates the ``features_root`` stamp, so its feature
   set cannot be proven from the file itself (it must be proven externally);
3. ``mixed_weights`` / ``mixed_params`` -- the sequences do not agree on the model or on the
   build parameters (scales, thresholds, dilution, ...).

Exit code 0 = every sequence agrees and the feature source is provable, 1 = at least one
finding, 2 = the cache root is unusable.

Usage
-----
    python manu/data/audit_cache_provenance.py --cache-root runs/optuna_mix_ratio/trials/trial_0080/lq_eval/ms_cache
    python manu/data/audit_cache_provenance.py --cache-root runs/ms_cache_nop0 --allow-missing-features-root

Execution machine: any machine that can read the cache directory (no GPU, no model).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# The fields that must agree across every sequence of one cache root.
PARAM_FIELDS = (
    "scales",
    "imgsz",
    "stride",
    "region_threshold",
    "main_threshold",
    "fusion_dilate",
    "min_area",
    "hm_weights",
    "use_p0_highway",
    "adapter_weights",
    "adapter_guide_channel",
    "adapter_single",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the provenance of a heatmap cache directory")
    parser.add_argument("--cache-root", required=True, help="Directory of <sequence>.pkl caches")
    parser.add_argument("--allow-missing-features-root", action="store_true",
                        help="Treat a cache without the features_root stamp as acceptable "
                             "(use for caches built before the stamp existed)")
    parser.add_argument("--json", default="", help="Optional path for the machine readable report")
    return parser.parse_args()


def read_meta(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle).get("meta", {})


def main() -> int:
    args = parse_args()
    root = Path(args.cache_root)
    if not root.is_dir():
        print(f"[FAIL] cache root not found: {root}")
        return 2

    files = sorted(root.glob("*.pkl"))
    if not files:
        print(f"[FAIL] no <sequence>.pkl under {root}")
        return 2

    rows = []
    for path in files:
        meta = read_meta(path)
        rows.append({
            "sequence": path.stem,
            "source": meta.get("source", "<missing>"),
            "features_root": meta.get("features_root", ""),
            "frames": meta.get("frames"),
            "native_size": meta.get("native_size"),
            **{field: meta.get(field) for field in PARAM_FIELDS},
        })

    findings = []

    def flag(code: str, detail: str) -> None:
        findings.append({"code": code, "detail": detail})

    # 1) feature source must agree across sequences
    sources = sorted({str(row["source"]) for row in rows})
    if len(sources) > 1:
        flag("mixed_source", f"sequences disagree on source: {sources}")
    if sources and sources[0] == "features":
        roots = sorted({str(row["features_root"]) for row in rows})
        if any(not r for r in roots):
            missing = [r["sequence"] for r in rows if not str(r["features_root"])]
            if args.allow_missing_features_root:
                print(f"[WARN] {len(missing)} sequence(s) have no features_root stamp "
                      f"(allowed by --allow-missing-features-root)")
            else:
                flag("missing_features_root",
                     f"{len(missing)} sequence(s) have no features_root stamp, e.g. {missing[:3]}")
        elif len(roots) > 1:
            flag("mixed_features_root", f"sequences disagree on features_root: {roots}")

    # 2) model identity and build parameters must agree across sequences
    for field in PARAM_FIELDS:
        values = sorted({repr(row[field]) for row in rows})
        if len(values) > 1:
            flag(f"mixed_{field}", f"sequences disagree on {field}: {values}")

    # 3) frame count must be positive
    empty = [row["sequence"] for row in rows if not row["frames"]]
    if empty:
        flag("empty_cache", f"{len(empty)} sequence(s) have zero frames: {empty[:3]}")

    # ---- report ----
    print(f"=== cache provenance audit: {root} ===")
    print(f"sequences: {len(rows)}")
    src = sources[0] if len(sources) == 1 else "/".join(sources)
    print(f"source: {src}")
    if src == "features":
        roots = sorted({str(row["features_root"]) for row in rows})
        print(f"features_root: {'/'.join(r if r else '<missing>' for r in roots)}")
    print(f"hm_weights: {rows[0]['hm_weights']}  |  use_p0_highway: {rows[0]['use_p0_highway']}")
    print(f"scales: {rows[0]['scales']}  |  frames: {sum(r['frames'] or 0 for r in rows)}")
    print()
    print(f"{'sequence':<34} {'frames':>7}  {'source':<10} features_root")
    for row in rows:
        print(f"{row['sequence']:<34} {str(row['frames']):>7}  {row['source']:<10} "
              f"{row['features_root'] or '<missing>'}")

    print()
    if findings:
        print(f"[FAIL] {len(findings)} finding(s):")
        for item in findings:
            print(f"  - {item['code']}: {item['detail']}")
    else:
        print("[PASS] all sequences agree on model, build parameters and feature source")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        path.write_text(json.dumps({
            "cache_root": str(root),
            "sequences": rows,
            "findings": findings,
            "ok": not findings,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report -> {path.resolve()}")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
