#!/usr/bin/env python3
"""Compare a re-scored run against the 2026-10-22 official record, sequence by sequence.

The official record (osd_sys_sota_trial0080_{th045,v2,v3}_20261022) stores its final working point
in search_json_for_render.json -> best_global, with per-sequence metrics for all 24 sequences.
This script lines the freshly computed metrics up against those rows and prints the per-sequence
delta plus the aggregate delta, so "does my number match the record" is answered per sequence
instead of only in the total.
"""
import argparse
import json
from pathlib import Path


def load_official(path: Path) -> tuple[dict, dict]:
    doc = json.loads(path.read_text())
    best = doc["best_global"]
    return best.get("per_sequence", {}), best.get("metrics", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--official-json", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument(
        "--max-df1-pp",
        type=float,
        default=None,
        help="Gate: fail when any sequence's |delta F1| exceeds this many percentage points",
    )
    ap.add_argument(
        "--require-gt-identical",
        action="store_true",
        help="Gate: fail when any sequence's GT count differs from the official record",
    )
    ap.add_argument(
        "--min-f1",
        type=float,
        default=None,
        help="Gate: fail when any sequence's own F1 (percent) falls below this floor",
    )
    args = ap.parse_args()

    got = json.loads(Path(args.eval_json).read_text())
    mine = got.get("per_sequence", {})
    off, off_metrics = load_official(Path(args.official_json))

    rows, ranked, missing = [], [], []
    for seq in sorted(set(mine) | set(off)):
        m, o = mine.get(seq), off.get(seq)
        if not m or not o:
            missing.append({"sequence": seq, "missing_in": "mine" if not m else "official"})
            continue
        df1 = m["f1"] * 100 - o["f1"] * 100
        drow = {
            "sequence": seq,
            "f1_mine": m["f1"] * 100, "f1_official": o["f1"] * 100, "df1_pp": round(df1, 6),
            "recall_mine": m["recall"] * 100, "recall_official": o["recall"] * 100,
            "precision_mine": m["precision"] * 100, "precision_official": o["precision"] * 100,
            "far_mine": m["far_per_frame"], "far_official": o["far_per_frame"],
            "gt_mine": m["gt"], "gt_official": o["gt"],
            "tp_mine": m["tp"], "tp_official": o["tp"],
            "fp_mine": m["fp"], "fp_official": o["fp"],
            "fn_mine": m["fn"], "fn_official": o["fn"],
        }
        rows.append(drow)
        ranked.append((abs(df1), seq, drow))

    ranked.sort(reverse=True)
    matched = len(rows)
    gt_match = sum(1 for r in rows if "gt_mine" in r and r["gt_mine"] == r["gt_official"])
    exact = sum(1 for r in rows if "df1_pp" in r and abs(r["df1_pp"]) < 1e-9)
    print(f"[CMP] sequences: mine={len(mine)} official={len(off)} compared={matched} "
          f"missing={len(missing)}"
          + (f"  ({missing[0]['sequence']} ...)" if missing else ""))
    print(f"[CMP] GT identical {gt_match}/{matched} | F1 exactly identical {exact}/{matched}")
    print("[CMP] top-8 |df1| sequences:")
    for _, seq, r in ranked[:8]:
        print(f"   {seq:32s} mine={r['f1_mine']:8.3f}%  official={r['f1_official']:8.3f}%  "
              f"df1={r['df1_pp']:+8.4f}pp  tp {r['tp_mine']}/{r['tp_official']} "
              f"fp {r['fp_mine']}/{r['fp_official']} fn {r['fn_mine']}/{r['fn_official']}")
    ov = got.get("overall", {})
    print(f"[CMP] OVERALL mine    : f1={ov.get('f1')} recall={ov.get('recall')} "
          f"precision={ov.get('precision')} tp={ov.get('tp')} fp={ov.get('fp')} "
          f"fn={ov.get('fn')} gt={ov.get('gt')} far={ov.get('far_per_frame')}")
    print(f"[CMP] OVERALL official: f1={off_metrics.get('f1')} recall={off_metrics.get('recall')} "
          f"precision={off_metrics.get('precision')} tp={off_metrics.get('tp')} "
          f"fp={off_metrics.get('fp')} fn={off_metrics.get('fn')} gt={off_metrics.get('gt')} "
          f"far={off_metrics.get('far_per_frame')}")
    for key, mine_key in (("f1", "f1"), ("recall", "recall"), ("precision", "precision")):
        if ov.get(mine_key) is not None and off_metrics.get(key) is not None:
            print(f"[CMP] d{mine_key:10s} = {(ov[mine_key] - off_metrics[key]) * 100:+.6f} pp")
    if ov.get("fp") is not None and off_metrics.get("fp") is not None:
        print(f"[CMP] dfp        = {ov['fp'] - off_metrics['fp']:+d}   "
              f"dfn = {ov.get('fn', 0) - off_metrics.get('fn', 0):+d}   "
              f"dgt = {ov.get('gt', 0) - off_metrics.get('gt', 0):+d}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(
            {"eval_json": args.eval_json, "official_json": args.official_json,
             "overall_mine": ov, "overall_official": off_metrics, "per_sequence": rows,
             "missing": missing},
            ensure_ascii=False, indent=1))
        print(f"[CMP] -> {args.out_json}")
    violations = []
    if args.require_gt_identical:
        bad_gt = [r for r in rows if r.get("gt_mine") != r.get("gt_official")]
        if bad_gt:
            violations.append(f"GT 不一致 {len(bad_gt)} 条：{[r['sequence'] for r in bad_gt]}")
    if args.max_df1_pp is not None:
        bad_df1 = [r for r in rows if abs(r.get("df1_pp", 0.0)) > args.max_df1_pp]
        if bad_df1:
            violations.append(
                f"|dF1| > {args.max_df1_pp}pp 共 {len(bad_df1)} 条："
                f"{[(r['sequence'], r['df1_pp']) for r in bad_df1]}"
            )
    if args.min_f1 is not None:
        bad_f1 = [r for r in rows if r.get("f1_mine", 100.0) < args.min_f1]
        if bad_f1:
            violations.append(
                f"F1 < {args.min_f1}% 共 {len(bad_f1)} 条："
                f"{[(r['sequence'], round(r['f1_mine'], 4)) for r in bad_f1]}"
            )
    if violations:
        print("[CMP GATE FAIL] " + " | ".join(violations))
        return 1
    if args.max_df1_pp is not None or args.require_gt_identical or args.min_f1 is not None:
        print("[CMP GATE PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
