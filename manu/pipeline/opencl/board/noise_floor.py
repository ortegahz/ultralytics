#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noise_floor.py -- run-to-run spread of the x86 timing ledger.

Why this exists: arm A of the first timing run started while the 1-minute load
average was still ~5.5 from the parallel census/accuracy work, so it measured a
busier machine than arm B did. A single A-vs-B comparison therefore cannot
separate "the handle changed the cost" from "the second arm ran on a quieter
box". Running both arms a second time under one settled load gives a same-arm
spread to compare the A-vs-B delta against.

Reading the result:
  - |A_run1 - A_run2|  = NOISE. Same config, same binary, same input.
  - |A - B|           = EFFECT. Only interpretable if it clears NOISE.
Anything smaller than NOISE is reported as "not distinguishable from noise",
not as an effect.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("run1", os.path.join(HERE, ".bench_out", "REGEN")),
        ("run2", os.path.join(HERE, ".bench_out", "REGEN_REP"))]
SEQ_ORDER = ["VIDEO00034", "VIDEO00028", "VIDEO00032", "VIDEO00002", "VIDEO00036"]
WARMUP = 100


def nearest_rank(vals, q):
    n = len(vals)
    if n == 0:
        return None
    k = max(1, min(n, int(math.ceil(q / 100.0 * n))))
    return vals[k - 1]


def load(root, arm, seq):
    d = os.path.join(root, arm)
    if not os.path.isdir(d):
        return None
    hit = None
    for f in os.listdir(d):
        # names are profile_<arm>_<SEQ>_<stamp>.csv, so the sequence id is the
        # THIRD underscore field. Splitting on field 0 matches the literal
        # "profile" and finds nothing -- which this script reported as NO DATA
        # rather than as an error.
        parts = f.split("_")
        if f.endswith(".csv") and len(parts) > 2 and parts[2] == seq:
            hit = os.path.join(d, f)
            break
    if not hit:
        return None
    with open(hit, errors="replace") as fh:
        hdr = fh.readline().strip().split(",")
        try:
            ti, fi = hdr.index("total_ms"), hdr.index("frame_idx")
        except ValueError:
            return None
        vals = []
        for line in fh:
            c = line.strip().split(",")
            if len(c) <= max(ti, fi):
                continue
            try:
                if int(c[fi]) < WARMUP:
                    continue
                vals.append(float(c[ti]))
            except ValueError:
                continue
    if not vals:
        return None
    vals.sort()
    return {"n": len(vals), "p50": nearest_rank(vals, 50),
            "p99": nearest_rank(vals, 99), "max": vals[-1]}


def main():
    print("=" * 104)
    print("X86 RUN-TO-RUN NOISE FLOOR  (nearest-rank percentiles, post-warmup frames)")
    print("=" * 104)
    print(f"{'seq':<12} {'arm':<4} {'run':<5} {'n':>4} {'p50':>8} {'p99':>8} {'max':>8}")
    print("-" * 104)
    data = {}
    for tag, root in RUNS:
        for arm in ("A", "B"):
            for seq in SEQ_ORDER:
                r = load(root, arm, seq)
                if not r:
                    print(f"{seq:<12} {arm:<4} {tag:<5} {'NO DATA':>4}")
                    continue
                data[(seq, arm, tag)] = r
                print(f"{seq:<12} {arm:<4} {tag:<5} {r['n']:>4} "
                      f"{r['p50']:>8.2f} {r['p99']:>8.2f} {r['max']:>8.2f}")

    print()
    print("-" * 104)
    print("NOISE  = same arm, run1 vs run2   |  EFFECT = arm A vs arm B (run2, quieter box)")
    print("-" * 104)
    print(f"{'seq':<12} {'metric':<6} {'noise|A|':>9} {'noise|B|':>9} "
          f"{'effect A-B':>11} {'verdict':>28}")
    print("-" * 104)
    for seq in SEQ_ORDER:
        for metric in ("p50", "p99"):
            a1 = data.get((seq, "A", "run1"), {}).get(metric)
            a2 = data.get((seq, "A", "run2"), {}).get(metric)
            b1 = data.get((seq, "B", "run1"), {}).get(metric)
            b2 = data.get((seq, "B", "run2"), {}).get(metric)
            if None in (a1, a2, b1, b2):
                continue
            nA, nB = abs(a1 - a2), abs(b1 - b2)
            eff = abs(a2 - b2)
            noise = max(nA, nB)
            if noise == 0:
                verdict = "effect, zero noise"
            elif eff >= noise:
                verdict = "EFFECT clears noise"
            else:
                verdict = "not distinguishable"
            print(f"{seq:<12} {metric:<6} {nA:>9.2f} {nB:>9.2f} {eff:>11.2f} {verdict:>28}")
    print()
    print("NOTE: these are PoCL-on-CPU numbers (see GMC_HANDLE_DELIVERY.md section 0).")
    print("      They bound whether a handle CHANGED the cost, nothing about the board.")
    print("=" * 104)


if __name__ == "__main__":
    sys.exit(main())