#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Per-sequence + overall TP/FP/FN summary for the Trial-0474 CPU reference.

Reads a `detections.json` written by hm_reference.py and reports every ratio
with the absolute numbers behind it and a 95% interval, per manu/memory/rules.md
("任何以 X/Y = Z% 形式汇报的...必须同时给出分子分母的绝对数 + Wilson 置信区间").

Reporting rules enforced here:
  * precision / recall are true proportions, so they carry a Wilson score
    interval (never a normal approximation -- it goes out of [0,1] on the small
    per-sequence denominators this dataset produces).
  * F1 is NOT a proportion, so Wilson does not apply to it. It is bounded by
    mapping the Wilson endpoints of P and R through F1 (monotone increasing in
    both), which is conservative -- it ignores the P/R correlation. Labelled
    `rect` below so it is never mistaken for a Wilson interval.
  * every sequence is flagged `low-n` when n_gt < 100, i.e. below the two
    orders of magnitude the same rule requires before a stratified conclusion
    may be drawn from it.

Sequences are sorted by ASCENDING recall, so the worst ones are visible first.

Usage:
    python hm_seq_stats.py /path/to/detections.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 95% two-sided normal quantile
Z = 1.959963984540054
# rules.md: a stratified proportion needs a denominator of this order before
# conclusions may be drawn from it.
MIN_DENOM_FOR_CLAIM = 100


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval for k successes out of n. Returns (lo, hi).

    n == 0 yields (0.0, 0.0); callers must report the denominator as n/a.
    """
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = Z * Z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = Z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def f1_of(p: float, r: float) -> float:
    return 0.0 if (p + r) <= 0 else 2.0 * p * r / (p + r)


def ratio_cell(k: int, n: int, fixed: int = 4) -> str:
    """`k/n=p.xxx [lo--hi]` -- the absolute pair is never dropped."""
    if n <= 0:
        return f"{k}/0=n/a   [n/a----n/a]"
    lo, hi = wilson(k, n)
    return f"{k}/{n}={k / n:.{fixed}f} [{lo:.4f}--{hi:.4f}]"


def accumulate(recs: list[dict]) -> dict[str, dict]:
    """Roll per-frame tp/fp/fn/gt up into one bucket per sequence (+ 'ALL')."""
    agg: dict[str, dict] = {}

    def bucket(seq: str) -> dict:
        return agg.setdefault(seq, {"n_frames": 0, "tp": 0, "fp": 0, "fn": 0})

    for rec in recs:
        seq = rec.get("seq", "<no-seq>")
        n_gt = len(rec.get("gt_canvas", []))
        for key in (seq, "ALL"):
            b = bucket(key)
            b["n_frames"] += 1
            b["tp"] += int(rec.get("tp", 0))
            b["fp"] += int(rec.get("fp", 0))
            b["fn"] += int(rec.get("fn", 0))
            b["n_gt"] = b.get("n_gt", 0) + n_gt
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("detections", help="detections.json written by hm_reference.py")
    ap.add_argument("--sort", default="recall", choices=["recall", "f1", "seq"],
                    help="ascending sort key (default: recall, worst first)")
    args = ap.parse_args()

    path = Path(args.detections)
    if not path.is_file():
        raise SystemExit(f"[FATAL] no such file: {path}")
    recs = json.loads(path.read_text())
    if not recs:
        raise SystemExit(f"[FATAL] {path} holds no records")
    missing = [k for k in ("tp", "fp", "fn", "gt_canvas") if k not in recs[0]]
    if missing:
        raise SystemExit(f"[FATAL] {path} lacks required keys: {missing}")

    agg = accumulate(recs)
    seqs = [s for s in agg if s != "ALL"]

    def key(s: str):
        b = agg[s]
        if args.sort == "seq":
            return (s,)
        # zero-GT sequences cannot produce a recall -> push them to the bottom
        if b.get("n_gt", 0) <= 0:
            return (1, 0.0)
        p = b["tp"] / max(1, b["tp"] + b["fp"])
        r = b["tp"] / b["n_gt"]
        return (0, r if args.sort == "recall" else f1_of(p, r))

    seqs.sort(key=key)

    print(f"\n=== per-sequence TP/FP/FN  ({path}) ===")
    print(f"sort: ascending {args.sort} (worst first) | "
          f"'low-n' = n_gt < {MIN_DENOM_FOR_CLAIM}, too small for a stratified claim")

    hdr = (f"{'sequence':<26}{'nfr':>4}{'n_gt':>6}{'tp':>5}{'fp':>5}{'fn':>5}  "
           f"{'precision (tp/(tp+fp))':<30}{'recall (tp/n_gt)':<30}{'F1 (rect)':<24} flag")
    print(hdr)
    print("-" * len(hdr))

    for s in seqs:
        b = agg[s]
        n_gt = b.get("n_gt", 0)
        tp, fp, fn = b["tp"], b["fp"], b["fn"]
        p_lo, p_hi = wilson(tp, tp + fp)
        r_lo, r_hi = wilson(tp, n_gt)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / n_gt if n_gt else 0.0
        f_lo, f_hi = f1_of(p_lo, r_lo), f1_of(p_hi, r_hi)
        flag = "low-n" if 0 < n_gt < MIN_DENOM_FOR_CLAIM else ("no-gt" if n_gt == 0 else "")
        print(f"{s:<26}{b['n_frames']:>4}{n_gt:>6}{tp:>5}{fp:>5}{fn:>5}  "
              f"{ratio_cell(tp, tp + fp):<30}{ratio_cell(tp, n_gt):<30}"
              f"{f1_of(p, r):.4f} [{f_lo:.4f}--{f_hi:.4f}]{'':<12}{flag}")

    b = agg["ALL"]
    n_gt, tp, fp, fn = b.get("n_gt", 0), b["tp"], b["fp"], b["fn"]
    p_lo, p_hi = wilson(tp, tp + fp)
    r_lo, r_hi = wilson(tp, n_gt)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / n_gt if n_gt else 0.0
    print("=" * len(hdr))
    print(f"{'OVERALL':<26}{b['n_frames']:>4}{n_gt:>6}{tp:>5}{fp:>5}{fn:>5}  "
          f"{ratio_cell(tp, tp + fp):<30}{ratio_cell(tp, n_gt):<30}"
          f"{f1_of(p, r):.4f} [{f1_of(p_lo, r_lo):.4f}--{f1_of(p_hi, r_hi):.4f}]")

    # self-consistency: the frame-level tally must close
    rec_gt = sum(len(r.get("gt_canvas", [])) for r in recs)
    rec_tp = sum(int(r.get("tp", 0)) for r in recs)
    rec_fp = sum(int(r.get("fp", 0)) for r in recs)
    rec_fn = sum(int(r.get("fn", 0)) for r in recs)
    ok = (rec_gt, rec_tp, rec_fp, rec_fn) == (n_gt, tp, fp, fn)
    print(f"\n[check] frames={b['n_frames']} (json {len(recs)}) | "
          f"gt={rec_gt} tp={rec_tp} fp={rec_fp} fn={rec_fn} | "
          f"tp+fn==gt: {rec_tp + rec_fn == rec_gt} | {rec_tp + rec_fp == sum(int(r.get('n_det', 0)) for r in recs)}"
          f" (tp+fp==sum n_det) | aggregate {'' if ok else 'MISMATCH'}")
    print("[check] F1 CI is the Wilson-endpoint rectangle over (P, R) -- a "
          "conservative bound, not a Wilson interval; P and R CIs above are Wilson.")
    print(f"[check] denominator for a stratified claim is n_gt per sequence; "
          f"only {sum(1 for s in seqs if agg[s].get('n_gt', 0) >= MIN_DENOM_FOR_CLAIM)}"
          f"/{len(seqs)} sequences reach n_gt >= {MIN_DENOM_FOR_CLAIM}.")


if __name__ == "__main__":
    main()
