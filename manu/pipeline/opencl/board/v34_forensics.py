#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v34_forensics.py -- why does VIDEO00034 still spike, WITH pre-allocation active?

The delivery target was: VIDEO00034 max 53.50 ms -> below 25 ms, by removing the
runtime device reallocation. The board campaign ran with --prealloc-pad 1024 and
devrealloc did drop to 1 on every sequence -- the handle did exactly what it was
built to do. And the spike did NOT go away: max is 43.16 ms.

So the counterfactual in the report ("53.50 -> ~13.0 ms, because the spike was
the reallocation") is falsified on the board. This script finds what the spike
actually is, by ranking the worst frames and decomposing their stages rather
than asserting a cause.

Every number is read from the raw per-frame CSV. Percentiles are nearest-rank.
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["io_ms", "prep_ms", "fit_ms", "img_up_ms", "mats_up_ms",
          "kernel_ms", "readback_ms"]
WARMUP = 100
RED = 40.0


def nearest_rank(v, q):
    s = sorted(v)
    return s[max(1, math.ceil(q / 100.0 * len(s))) - 1]


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".bench_out", "GD")
    arm = os.path.basename(d).rstrip("0123456789")
    path = None
    for f in os.listdir(d):
        if f.endswith(".csv") and "VIDEO00034" in f:
            path = os.path.join(d, f)
            break
    if not path:
        print("no VIDEO00034 CSV under", d)
        return 1
    rows = load(path)
    steady = [r for r in rows if int(r["frame_idx"]) >= WARMUP]
    print(f"arm {arm}: {os.path.basename(path)}  rows={len(rows)}  "
          f"steady={len(steady)}  warmup={WARMUP}")
    print()

    tot = [float(r["total_ms"]) for r in steady]
    print("total_ms  p50=%.2f p95=%.2f p99=%.2f max=%.2f   over %.0f ms: %d/%d"
          % (nearest_rank(tot, 50), nearest_rank(tot, 95), nearest_rank(tot, 99),
             max(tot), RED, sum(1 for t in tot if t > RED), len(tot)))
    print()
    print("stage            p50      p95      p99      max   share@max-frame")
    print("-" * 74)
    worst = max(steady, key=lambda r: float(r["total_ms"]))
    wtot = float(worst["total_ms"])
    for s in STAGES:
        v = [float(r[s]) for r in steady]
        print(f"{s:<12}{nearest_rank(v,50):>8.2f}{nearest_rank(v,95):>8.2f}"
              f"{nearest_rank(v,99):>8.2f}{max(v):>8.2f}"
              f"{100.0*float(worst[s])/wtot:>13.1f}%")
    resid = [float(r["resid_ms"]) for r in steady]
    print(f"{'resid_ms':<12}{nearest_rank(resid,50):>8.2f}{nearest_rank(resid,95):>8.2f}"
          f"{nearest_rank(resid,99):>8.2f}{max(resid):>8.2f}")
    print()
    print("worst 12 frames (steady window):")
    print(f"  {'frame':>6}{'total':>9}{'fit':>8}{'img_up':>9}{'kernel':>8}"
          f"{'io':>7}{'dominant stage':>18}")
    print("  " + "-" * 66)
    for r in sorted(steady, key=lambda r: -float(r["total_ms"]))[:12]:
        vals = {s: float(r[s]) for s in STAGES}
        dom = max(vals, key=vals.get)
        print(f"  {r['frame_idx']:>6}{float(r['total_ms']):>9.2f}"
              f"{vals['fit_ms']:>8.2f}{vals['img_up_ms']:>9.2f}"
              f"{vals['kernel_ms']:>8.2f}{vals['io_ms']:>7.2f}"
              f"{dom.replace('_ms',''):>18}")
    print()

    # The decisive question: is the spike a one-off, or a recurring mode?
    up = [float(r["img_up_ms"]) for r in steady]
    print("img_up_ms distribution -- is this bimodal (a few events) or a tail?")
    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1000)]
    for lo, hi in bands:
        c = sum(1 for x in up if lo <= x < hi)
        print(f"  [{lo:>4},{hi:>4}) ms : {c:>4} frames  ({100.0*c/len(up):>5.1f}%)")
    print()
    over = [r for r in steady if float(r["total_ms"]) > RED]
    print(f"frames over {RED:.0f} ms: {len(over)}")
    for r in over:
        print(f"  frame {r['frame_idx']:>4}: total={float(r['total_ms']):>6.2f}  "
              f"fit={float(r['fit_ms']):>6.2f}  img_up={float(r['img_up_ms']):>6.2f}  "
              f"kernel={float(r['kernel_ms']):>6.2f}  io={float(r['io_ms']):>5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())