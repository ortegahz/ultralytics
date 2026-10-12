#!/usr/bin/env python3
"""breach_census.py -- how many frames actually cross the 40 ms line, and where.

p99 answers "how bad is the tail"; it does NOT answer "how often are we late".
Those are different questions and on this corpus they disagree in a way that
matters for a 25 FPS delivery: nearest-rank p99 puts VIDEO00034's single 53.50 ms
frame below the p99 line (p99 19.52 -> PASS), while VIDEO00036 sits just over it.
Counting the frames directly shows the opposite emphasis -- VIDEO00034 breaches
ONCE, VIDEO00036 breaches TWENTY TIMES in steady state. One is a spike that
pre-allocation removes; the other is a sustained ~10% late rate that needs the
corner budget. Reporting only p99/max would have hidden that split.
"""
import csv
import glob
import math
import os
import sys

RED_LINE = 40.0
WARMUP = 100


def pct(values, q):
    s = sorted(values)
    return s[max(1, math.ceil(q / 100.0 * len(s))) - 1]


d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".bench_out", "FIX7")

print(f"source: {d}   red line: {RED_LINE:.0f} ms   warmup: {WARMUP} frames")
print()
print(f"{'sequence':<30}{'frames':>7}{'steady':>8}{'breach all':>12}{'breach steady':>15}"
      f"{'rate':>8}{'p99':>8}{'max':>8}")
print("-" * 96)
t_all = t_st = 0
for path in sorted(glob.glob(os.path.join(d, "profile_*.csv"))):
    rows = list(csv.DictReader(open(path)))
    seq = os.path.basename(path).replace(".csv", "")
    if seq.startswith("profile_"):
        seq = seq.split("_", 2)[2]
    steady = rows[WARMUP:]
    v_all = [float(r["total_ms"]) for r in rows]
    v = [float(r["total_ms"]) for r in steady]
    b_all = sum(1 for x in v_all if x > RED_LINE)
    b_st = sum(1 for x in v if x > RED_LINE)
    t_all += b_all
    t_st += b_st
    rate = 100.0 * b_st / len(v) if v else 0.0
    print(f"{seq:<30}{len(v_all):>7}{len(v):>8}{b_all:>12}{b_st:>15}"
          f"{rate:>7.1f}%{pct(v, 99):>8.2f}{max(v):>8.2f}")
print("-" * 96)
print(f"{'TOTAL':<30}{'':>7}{'':>8}{t_all:>12}{t_st:>15}")
print()
print("  Frames breaching during warm-up but not in steady state are transient")
print("  start-up cost (Mali JIT, context build), not a steady-state property.")
