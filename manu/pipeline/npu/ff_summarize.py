#!/usr/bin/env python3
"""Aggregate the GMC full-frame probe per-frame TSV into the speed tables.

Usage: ff_summarize.py TSV [TSV ...]

Each TSV is one full-frame probe run (one window, one corner budget).  The
columns are the per-frame wall time and the per-stage breakdown, so the
percentiles and the ">40 ms" fraction are computed from the SAME per-frame
numbers that the probe wrote.
"""
import sys


def load(path):
    hdr = None
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if hdr is None:
                hdr = parts
                continue
            rows.append(dict(zip(hdr, parts)))
    return hdr, rows


def pctl(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    if p <= 0:
        return s[0]
    if p >= 1:
        return s[-1]
    i = p * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def stats(v):
    if not v:
        return None
    over = sum(1 for x in v if x > 40.0)
    return dict(
        n=len(v), mean=sum(v) / len(v), p50=pctl(v, .50), p95=pctl(v, .95),
        p99=pctl(v, .99), max=max(v), min=min(v),
        over40=100.0 * over / len(v), fps=1000.0 / (sum(v) / len(v)))


def fmt(s):
    if s is None:
        return "   -   "
    return (f"{s['mean']:7.3f} {s['p50']:7.3f} {s['p95']:7.3f} {s['p99']:7.3f} "
            f"{s['max']:8.3f} {s['fps']:6.2f} {s['over40']:6.2f}")


def main():
    for path in sys.argv[1:]:
        hdr, rows = load(path)
        print("=" * 110)
        print(f"FILE: {path}   ({len(rows)} frames)")
        for arm, pre in (("CPU", "cpu_"), ("GPU", "gpu_")):
            tot = [float(r[pre + "total"]) for r in rows]
            s = stats(tot)
            print(f"  {arm} total  mean {s['mean']:8.3f}  p50 {s['p50']:8.3f}  p95 {s['p95']:8.3f}  "
                  f"p99 {s['p99']:8.3f}  max {s['max']:8.3f}  fps {s['fps']:6.2f}  >40ms {s['over40']:6.2f}%")
        # the GPU breakdown
        cols = [("resize", "gpu_resize"), ("gFTT", "gpu_gftt"), ("select", "gpu_select"),
                ("pyr", "gpu_pyr"), ("upload", "gpu_upload"), ("LK batch", "gpu_lk_batch"),
                ("LK 1anchor", "gpu_lk_1anchor"), ("LK seq", "gpu_lk_seq"),
                ("RANSAC", "gpu_ransac")]
        print("  GPU breakdown (mean ms):  " + "  ".join(
            f"{n}={sum(float(r[c]) for r in rows)/len(rows):.3f}" for n, c in cols))
        cc = [float(r["ncorners_gpu"]) for r in rows]
        print(f"  corners: mean {sum(cc)/len(cc):.1f}  capped {sum(1 for x in cc if x >= 599.5)}/{len(cc)}")
        for c in ("cpu_lk", "cpu_ransac", "cpu_corners", "cpu_pyr"):
            if c in hdr:
                v = [float(r[c]) for r in rows]
                print(f"  CPU {c:12s} mean {sum(v)/len(v):8.3f}  p99 {pctl(v,.99):8.3f}")


if __name__ == "__main__":
    main()
