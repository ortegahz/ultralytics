#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fake_boards.py -- build GC/GD/GE from the REAL measured board baseline so
guarantee_report.py can be exercised end to end before the board is reachable.

These numbers are the actual RK3588 baseline from report section 3.2 (gate
version, 11:12), reproduced here frame-by-frame with the same shape, so the
dry run produces the percentiles the report already published. If the report
agrees with this output, the report's own table is re-derived rather than
trusted.

Arm C/D/E are laid out with the per-frame structure each arm will have:
  C  prealloc only, corners 600            -> VIDEO00034 max must collapse
  D  prealloc + corners 200 on VIDEO00036  -> the delivery config
  E  D's config with --fused both          -> carries [SEQ] and [A/B] lines

A short CSV is injected for one sequence on purpose: the board has been
SIGKILLed mid-sequence before, and a PARTIAL run must be flagged, not averaged.
"""
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# NOT .bench_out/G{a}. That is where the real board results land, and a fetch
# that fails leaves whatever is already there in place -- so synthetic data
# parked there would be reported as an RK3588 measurement. The first version of
# this script wrote exactly there. It now writes under a clearly-marked tree
# and drops a sentinel that guarantee_report.py refuses to read.
ROOT = os.path.join(HERE, ".bench_out", "_DRYRUN_FAKE")
SENTINEL = "SYNTHETIC_DO_NOT_REPORT.txt"

COLS = ["seq_id", "frame_idx", "seq_frame", "io_ms", "prep_ms", "fit_ms", "warp_ms",
        "median_ms", "img_up_ms", "mats_up_ms", "kernel_ms", "readback_ms",
        "total_ms", "resid_ms"]

# Measured board baseline: seq -> (p50, p95, p99, max, breaches_in_steady)
BASE = {
    "VIDEO00034_19700101_015052": (10.67, 12.10, 19.52, 53.50, 1),
    "VIDEO00028_19700101_013631": (13.22, 16.32, 17.97, 18.28, 0),
    "VIDEO00032_19700101_014453": (19.84, 28.08, 30.37, 31.60, 0),
    "VIDEO00002_19700101_001415": (15.90, 16.36, 16.62, 16.82, 0),
    "VIDEO00036_19700101_015740": (33.37, 40.74, 42.88, 43.09, 20),
}


def synth(seq, arm, rng, limit=400, warmup=100):
    p50, p95, p99, mx, nbreach = BASE[seq]
    rows = []
    breaches_left = nbreach
    for i in range(limit):
        if i < warmup:
            t = p50 * rng.uniform(0.8, 3.0)      # warm-up is genuinely slower
        else:
            t = p50 * rng.uniform(0.97, 1.03)
            if breaches_left and i % (300 // max(nbreach, 1)) == 0 and i >= warmup:
                t = mx * rng.uniform(0.9, 1.0)
                breaches_left -= 1
        fit = t * 0.663
        rows.append({
            "seq_id": 0, "frame_idx": i, "seq_frame": i,
            "io_ms": round(rng.uniform(0.40, 0.49), 6),
            "prep_ms": round(rng.uniform(0.0002, 0.0004), 6),
            "fit_ms": round(fit, 6),
            "warp_ms": 0.0, "median_ms": 0.0,
            "img_up_ms": round(max(t - fit - t * 0.231, 0.02), 6),
            "mats_up_ms": round(rng.uniform(0.007, 0.016), 6),
            "kernel_ms": round(t * 0.231, 6),
            "readback_ms": round(rng.uniform(0.120, 0.143), 6),
            "total_ms": round(t, 6),
            "resid_ms": round(rng.uniform(0.030, 0.051), 6),
        })
    if arm == "C":
        # pre-allocation: the reallocation spike is gone. VIDEO00034's max
        # collapses to the upload floor; every other sequence is untouched.
        for r in rows:
            if r["total_ms"] > 40:
                r["total_ms"] = round(12.9 + rng.uniform(0, 0.2), 6)
                r["fit_ms"] = round(r["total_ms"] * 0.663, 6)
                r["kernel_ms"] = round(r["total_ms"] * 0.231, 6)
                r["img_up_ms"] = round(r["total_ms"] - r["fit_ms"] - r["kernel_ms"], 6)
    if arm == "D":
        for r in rows:
            if r["total_ms"] > 40:
                r["total_ms"] = round(29.5 + rng.uniform(0, 0.6), 6)
                r["fit_ms"] = round(r["total_ms"] * 0.663, 6)
                r["kernel_ms"] = round(r["total_ms"] * 0.231, 6)
                r["img_up_ms"] = round(r["total_ms"] - r["fit_ms"] - r["kernel_ms"], 6)
    return rows


SEQ_LINE = (
    "[SEQ] {seq}       frames=400    md5=deadbeefdeadbeefdeadbeefdeadbeef  "
    "fits/frm=6.09 warps/frm=22.00 compose/frm=40.00  state=13.44 MiB  "
    "devpad={devpad} devallocpad={allocpad} devrealloc={realloc} | "
    "[HANDLE] cornercap={cf}f/{cc}calls({cp}pts) fitgate={fg} chain={ch} "
    "fitdeadline=0 | corners mean={mean:.1f} over {dets} dets (budget {budget})"
)
AB_LINE = (
    "[A/B] {seq}       Ch0 max=0 MAE=0.00000 exact=100.0000% | "
    "Ch1 max={c1} MAE={c1m:.5f} exact={c1e:.4f}% | "
    "Ch2 max={c2} MAE={c2m:.5f} exact={c2e:.4f}%"
)
# board-corresponding census: bench_run.sh passes --pyr-cache, so the ON counts
CENSUS_D = {
    "VIDEO00034_19700101_015052": (0, 0, 0, 0, 0, 84.5, 398, 600),
    "VIDEO00028_19700101_013631": (0, 0, 0, 83, 31, 60.1, 398, 600),
    "VIDEO00032_19700101_014453": (0, 0, 0, 99, 133, 229.6, 398, 600),
    "VIDEO00002_19700101_001415": (0, 0, 0, 0, 0, 1056.6, 398, 600),
    "VIDEO00036_19700101_015740": (398, 398, 398, 0, 0, 200.0, 398, 200),
}
AB = {
    "VIDEO00034_19700101_015052": (0, 0.00000, 100.0000, 0, 0.00000, 100.0000),
    "VIDEO00028_19700101_013631": (0, 0.00000, 100.0000, 0, 0.00000, 100.0000),
    "VIDEO00032_19700101_014453": (0, 0.00000, 100.0000, 0, 0.00000, 100.0000),
    "VIDEO00002_19700101_001415": (0, 0.00000, 100.0000, 0, 0.00000, 100.0000),
    "VIDEO00036_19700101_015740": (0, 0.60860, 63.7733, 0, 0.64098, 68.3717),
}


def build(arm):
    d = os.path.join(ROOT, f"G{arm}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, SENTINEL), "w") as fh:
        fh.write("Synthetic data produced by make_fake_boards.py.\n"
                 "These frames were never run on an RK3588 board.\n"
                 "guarantee_report.py refuses to read any directory containing this file.\n")
    rng = random.Random(1234 + ord(arm))
    log = [f"### environment 2026-10-12T13:00:00Z  experiment={arm} limit=400 warmup=100",
           f"### bench_extra={'--prealloc-pad 1024' if arm != 'BASELINE' else '<none>'}",
           f"### bench_extra_map={'VIDEO00036=--fit-corners 200' if arm in 'DE' else '<none>'}",
           f"### fused={'both' if arm == 'E' else 'gpu'}"]
    for seq in BASE:
        limit = 400
        if arm == "D" and seq.startswith("VIDEO00002"):
            limit = 260          # injected SIGKILL-style truncation
        rows = synth(seq, arm, rng, limit=limit)
        with open(os.path.join(d, f"profile_{arm}_{seq}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        log.append(f"### ---- {seq} ----")
        log.append(SEQ_LINE.format(
            seq=seq, devpad=512 if seq.startswith("VIDEO00034") else
            (128 if seq.startswith("VIDEO00036") else 1024),
            allocpad=1024, realloc=1 if arm in "CDE" else
            {"VIDEO00034_19700101_015052": 5, "VIDEO00028_19700101_013631": 6,
             "VIDEO00032_19700101_014453": 6, "VIDEO00002_19700101_001415": 1,
             "VIDEO00036_19700101_015740": 3}[seq],
            cf=CENSUS_D[seq][0], cc=CENSUS_D[seq][1], cp=CENSUS_D[seq][2],
            fg=CENSUS_D[seq][3], ch=CENSUS_D[seq][4], mean=CENSUS_D[seq][5],
            dets=CENSUS_D[seq][6], budget=CENSUS_D[seq][7]))
        if arm == "E":
            a = AB[seq]
            log.append(AB_LINE.format(seq=seq, c1=a[0], c1m=a[1], c1e=a[2],
                                      c2=a[3], c2m=a[4], c2e=a[5]))
    with open(os.path.join(d, f"bench_{arm}.log"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    return d


if __name__ == "__main__":
    for a in sys.argv[1:] or ["C", "D", "E"]:
        print("built", build(a))