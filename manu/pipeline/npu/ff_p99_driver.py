#!/usr/bin/env python3
"""The p99 driver: which stage explains the slow frames?

For each run, split the per-frame GPU total into its stages and report the
mean and the p99 of each stage, plus the mean/p99 of the stages restricted to
the frames that are over 40 ms (the tail) and over the 40 ms budget.
"""
import sys
sys.path.insert(0, '/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu')
from ff_summarize import load, stats, pctl

STAGES = [("resize", "gpu_resize"), ("gFTT", "gpu_gftt"), ("select", "gpu_select"),
          ("pyr", "gpu_pyr"), ("upload", "gpu_upload"), ("LK batch", "gpu_lk_batch"),
          ("LK 1anchor", "gpu_lk_1anchor"), ("LK seq", "gpu_lk_seq"),
          ("RANSAC", "gpu_ransac")]

for path in sys.argv[1:]:
    hdr, rows = load(path)
    tot = [float(r["gpu_total"]) for r in rows]
    s = stats(tot)
    print("=" * 104)
    print(f"{path}   n={len(rows)}  mean={s['mean']:.2f} p50={s['p50']:.2f} "
          f"p95={s['p95']:.2f} p99={s['p99']:.2f} max={s['max']:.2f} fps={s['fps']:.2f} "
          f">40ms={s['over40']:.1f}%")
    tail = [r for r in rows if float(r["gpu_total"]) > 40.0]
    head = [r for r in rows if float(r["gpu_total"]) <= 40.0]
    print(f"  {'stage':12s} {'all mean':>9s} {'all p99':>9s} {'slow mean':>10s} {'slow p99':>9s} "
          f"{'fast mean':>10s}")
    for name, c in STAGES:
        if c not in hdr:
            continue
        am = sum(float(r[c]) for r in rows) / len(rows)
        ap = pctl([float(r[c]) for r in rows], .99)
        sm = (sum(float(r[c]) for r in tail) / len(tail)) if tail else 0.0
        sp = (pctl([float(r[c]) for r in tail], .99)) if tail else 0.0
        fm = (sum(float(r[c]) for r in head) / len(head)) if head else 0.0
        print(f"  {name:12s} {am:9.3f} {ap:9.3f} {sm:10.3f} {sp:9.3f} {fm:10.3f}")
    # the CPU side
    for c in ("cpu_total", "cpu_lk", "cpu_ransac", "cpu_corners"):
        if c in hdr:
            v = [float(r[c]) for r in rows]
            print(f"  CPU {c:12s} mean {sum(v)/len(v):8.3f}  p99 {pctl(v,.99):8.3f}  "
                  f"max {max(v):8.3f}")
