#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
board_noise_floor.py -- three things, all from raw board CSVs, no hand math.

  1. RUN-TO-RUN NOISE FLOOR.  Arm D and arm H are the SAME configuration on the
     SAME binary, run twice (H re-runs D's config to prove --pin-pad defaults off,
     which also hands us the repeat). |D - H| is therefore noise, and any A-vs-B
     delta smaller than it is not an effect.

  2. WHAT --pin-pad BOUGHT.  Arm D vs arm F, same binary, one flag apart.

  3. BASELINE -> DELIVERY.  The gate-version baseline from report 3.2 against
     arm F, which is the configuration being handed over.

The sequence id is taken from os.path.basename and the THIRD underscore field.
Splitting the full path is wrong here -- ".bench_out" itself contains an
underscore, so path.split('_')[2] yields the arm letter, not the sequence, and
every lookup silently misses. That mistake was made twice in this project;
hence a helper instead of an inline split.
"""
import csv
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP = 100

ORDER = ["VIDEO00034", "VIDEO00028", "VIDEO00032", "VIDEO00002", "VIDEO00036"]

# gate-version baseline, report section 3.2 (p50, p99, max)
BASE = {
    "VIDEO00034": (10.67, 19.52, 53.50),
    "VIDEO00028": (13.22, 17.97, 18.28),
    "VIDEO00032": (19.84, 30.37, 31.60),
    "VIDEO00002": (15.90, 16.62, 16.82),
    "VIDEO00036": (33.37, 42.88, 43.09),
}


def seq_of(path):
    parts = os.path.basename(path).split("_")
    return parts[2] if len(parts) > 2 else path


def load(d, arm):
    out = {}
    for path in glob.glob(os.path.join(d, f"profile_{arm}_*.csv")):
        rows = list(csv.DictReader(open(path)))[WARMUP:]
        if not rows:
            continue
        v = sorted(float(r["total_ms"]) for r in rows)

        def nr(q):
            return v[max(1, math.ceil(q / 100.0 * len(v))) - 1]

        out[seq_of(path)] = {"p50": nr(50), "p99": nr(99), "max": v[-1],
                             "n": len(v), "over40": sum(1 for x in v if x > 40.0)}
    return out


def main():
    D = load(os.path.join(HERE, ".bench_out", "GD"), "D")
    H = load(os.path.join(HERE, ".bench_out", "GH"), "H")
    F = load(os.path.join(HERE, ".bench_out", "GF"), "F")
    for tag, s in (("D", D), ("H", H), ("F", F)):
        if not s:
            print(f"arm {tag}: NO DATA")
            return 1

    print("=" * 84)
    print("1. BOARD RUN-TO-RUN NOISE FLOOR  (arm D vs arm H: same config, same binary, two runs)")
    print("=" * 84)
    print(f"{'seq':<12}{'metric':<6}{'D':>9}{'H':>9}{'noise':>9}{'F(pin)':>9}"
          f"{'F-D':>9}{'  verdict':>22}")
    print("-" * 84)
    for s in ORDER:
        for m in ("p50", "p99", "max"):
            d, h, f = D[s][m], H[s][m], F[s][m]
            noise = abs(d - h)
            eff = f - d
            if noise == 0:
                v = "noise 0"
            elif abs(eff) >= noise:
                v = "EFFECT clears noise"
            else:
                v = "within noise"
            print(f"{s:<12}{m:<6}{d:>9.2f}{h:>9.2f}{noise:>9.2f}{f:>9.2f}"
                  f"{eff:>+9.2f}{v:>22}")
    print()
    print("D vs H is the control. Read the D->F column against the noise column:")
    print("a delta smaller than noise is not an effect and is not claimed as one.")
    print()

    print("=" * 84)
    print("2. BREACH COUNT (steady-state frames over 40 ms, 300 per sequence)")
    print("=" * 84)
    print(f"{'seq':<12}{'arm D':>12}{'arm H (repeat)':>18}{'arm F (pin-pad)':>18}")
    print("-" * 84)
    td = th = tf = 0
    for s in ORDER:
        td += D[s]["over40"]; th += H[s]["over40"]; tf += F[s]["over40"]
        print(f"{s:<12}{D[s]['over40']:>12}{H[s]['over40']:>18}{F[s]['over40']:>18}")
    print("-" * 84)
    print(f"{'TOTAL / 1500':<12}{td:>12}{th:>18}{tf:>18}")
    print()

    print("=" * 84)
    print("3. BASELINE (gate version) -> DELIVERY (arm F)")
    print("=" * 84)
    print(f"{'seq':<12}{'base p50':>9}{'base p99':>9}{'base max':>10} | "
          f"{'F p50':>8}{'F p99':>8}{'F max':>8} | {'max delta':>11}")
    print("-" * 84)
    for s in ORDER:
        b = BASE[s]
        f = F[s]
        print(f"{s:<12}{b[0]:>9.2f}{b[1]:>9.2f}{b[2]:>10.2f} | "
              f"{f['p50']:>8.2f}{f['p99']:>8.2f}{f['max']:>8.2f} | {f['max']-b[2]:>+11.2f}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())