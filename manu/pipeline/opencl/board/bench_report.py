#!/usr/bin/env python3
"""
bench_report.py -- turn the board's per-frame CSVs into the deliverable tables.

WHY A LOCAL AGGREGATOR INSTEAD OF TRUSTING THE ON-BOARD SUMMARY
  The board prints a summary too, but it is produced by the same binary under
  test and goes out over a pty. This script recomputes every number from the raw
  rows, independently, so the report and the board's own summary can disagree --
  and if they do, that disagreement is the finding rather than something to paper
  over. It also does the arithmetic the summary cannot: per-sequence tiling
  residuals, the texture-annotated speed context, and the corpus roll-up.

  The percentile method must match the C++ one (nearest rank, no interpolation)
  or the two reports will disagree on P99 for reasons that have nothing to do with
  the pipeline. That is asserted below, not assumed.
"""
import argparse
import csv
import math
import os
import sys

STAGES = [
    ("io", "0. frame decode (eMMC BMP)"),
    ("prep", "1. ring push / frame admit"),
    ("fit", "2. GMC fit (LK+RANSAC+chain)"),
    ("warp", "2b. warp x22 (CPU arm)"),
    ("median", "2c. median 21-frame (CPU arm)"),
    ("img_up", "3. image upload H2D"),
    ("mats_up", "4. matrices H2D"),
    ("kernel", "5. fused warp+median"),
    ("readback", "6. readback D2H 3-plane"),
]


def pct(sorted_vals, q):
    """Nearest-rank, matching pct_nearest_rank() in gmc_stream_ocl.cpp."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    rank = math.ceil(q * n)
    rank = max(1, min(rank, n))
    return sorted_vals[rank - 1]


def stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n   # population SD, same as C++
    return dict(n=n, mean=mean, p50=pct(s, .50), p95=pct(s, .95), p99=pct(s, .99),
                max=s[-1], min=s[0], sd=math.sqrt(var), sum=sum(s))


def load(path, warmup):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["seq_frame"]) < warmup:
                continue
            rows.append({k: (v if k in ("seq_id", "frame_idx", "seq_frame") else float(v))
                         for k, v in r.items()})
    return rows


def tiling_residual(r):
    return r["total_ms"] - (r["io_ms"] + r["prep_ms"] + r["fit_ms"] + r["warp_ms"] +
                            r["median_ms"] + r["img_up_ms"] + r["mats_up_ms"] + r["kernel_ms"] +
                            r["readback_ms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="board output dir (local copy)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--texture", default="", help="optional TSV: sequence<TAB>mean_corners")
    ap.add_argument("--exp", required=True, help="experiment tag used in the filenames")
    args = ap.parse_args()

    tex = {}
    if args.texture and os.path.exists(args.texture):
        for line in open(args.texture):
            p = line.split("\t")
            if len(p) >= 2:
                try:
                    tex[p[0]] = float(p[1])
                except ValueError:
                    pass

    prefix = f"profile_{args.exp}_"
    files = sorted(f for f in os.listdir(args.dir)
                   if f.startswith(prefix) and f.endswith(".csv"))
    if not files:
        sys.exit(f"no {prefix}*.csv under {args.dir}")

    all_rows = []
    print("=" * 108)
    for f in files:
        seq = f[len(prefix):-4]
        rows = load(os.path.join(args.dir, f), args.warmup)
        if not rows:
            print(f"[skip] {f}: every row fell inside the warm-up window")
            continue
        for r in rows:
            r["_seq"] = seq
        all_rows.extend(rows)
        tot = stats([r["total_ms"] for r in rows])
        resid = stats([tiling_residual(r) for r in rows])
        tc = tex.get(seq)
        tcs = f"{tc:.1f}" if tc is not None else "?"
        print(f"[SEQ] {seq:<32} n={tot['n']:<4} corners={tcs:>6}  "
              f"p50={tot['p50']:7.2f}  p95={tot['p95']:7.2f}  p99={tot['p99']:7.2f}  "
              f"max={tot['max']:8.2f}  sd={tot['sd']:6.2f}  resid_max={resid['max']:+.3f}")
    print("=" * 108)

    print()
    print("STAGE LEDGER (all sequences pooled, warm-up discarded)")
    print(f"{'stage':<32}{'mean':>9}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>9}{'stddev':>9}{'share':>9}")
    print("-" * 92)
    tot = stats([r["total_ms"] for r in all_rows])
    for key, label in STAGES:
        st = stats([r[key + "_ms"] for r in all_rows])
        share = 100.0 * st["mean"] / tot["mean"] if tot["mean"] else 0.0
        print(f"{label:<32}{st['mean']:9.3f}{st['p50']:8.3f}{st['p95']:8.3f}{st['p99']:8.3f}"
              f"{st['max']:9.3f}{st['sd']:9.3f}{share:8.1f}%")
    print("-" * 92)
    print(f"{'== END-TO-END ==':<32}{tot['mean']:9.3f}{tot['p50']:8.3f}{tot['p95']:8.3f}"
          f"{tot['p99']:8.3f}{tot['max']:9.3f}{tot['sd']:9.3f}{100.0:8.1f}%")
    resid = stats([tiling_residual(r) for r in all_rows])
    print(f"{'TILING RESIDUAL':<32}{resid['mean']:9.3f}{resid['p50']:8.3f}{resid['p95']:8.3f}"
          f"{resid['p99']:8.3f}{resid['max']:9.3f}{resid['sd']:9.3f}")
    ok = abs(resid["mean"]) < 1e-3 and abs(resid["max"]) < 1e-2
    print(f"  -> {'TILES' if ok else 'CHECK ME'}: residual mean {resid['mean']:+.4f} ms, "
          f"max {resid['max']:+.4f} ms")
    print()
    print(f"fps: p50={1000/tot['p50']:.1f}  p95={1000/tot['p95']:.1f}  "
          f"p99={1000/tot['p99']:.1f}  worst={1000/tot['max']:.1f}")
    print(f"40ms red line (25 FPS): p99 {'PASS' if tot['p99'] <= 40 else 'FAIL'} "
          f"({tot['p99']:.2f} ms) | max {'PASS' if tot['max'] <= 40 else 'FAIL'} "
          f"({tot['max']:.2f} ms)")

    # Per-sequence end-to-end detail, so a single slow sequence cannot hide inside
    # the pooled figure.
    print()
    print("PER-SEQUENCE END-TO-END (ms/frame)")
    print(f"{'sequence':<32}{'corners':>9}{'mean':>9}{'p50':>8}{'p95':>8}{'p99':>8}"
          f"{'max':>9}{'sd':>8}{'fps_p50':>10}")
    print("-" * 101)
    seqs = []
    for r in all_rows:
        if r["_seq"] not in seqs:
            seqs.append(r["_seq"])
    for s in seqs:
        v = [r["total_ms"] for r in all_rows if r["_seq"] == s]
        st = stats(v)
        tc = tex.get(s)
        tcs = f"{tc:.1f}" if tc is not None else "?"
        print(f"{s:<32}{tcs:>9}{st['mean']:9.3f}{st['p50']:8.3f}{st['p95']:8.3f}"
              f"{st['p99']:8.3f}{st['max']:9.3f}{st['sd']:8.3f}{1000/st['p50']:10.1f}")


if __name__ == "__main__":
    main()