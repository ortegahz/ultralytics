#!/usr/bin/env python3
"""v34_prealloc_counterfactual.py -- what VIDEO00034 looks like once the device
reallocations are gone, derived by REMOVING MEASURED EVENTS rather than by
transferring a timing ratio from another machine.

The ratio projection used elsewhere in this report (x86 ratio -> board absolute)
is weak, because the x86 and A76 do not share a memory system. The pre-allocation
claim is stronger than that: the five reallocation events are individually
identified in the trace, and pre-allocation removes exactly those events and
nothing else. So the counterfactual here subtracts the measured upload excess on
the frames where it occurred.

It is still a counterfactual, not a measurement. What it rests on is one causal
claim -- that the upload spikes ARE the reallocations. The evidence for that:
  * devrealloc=5 over 400 frames, matching the 4 steady-state frames with
    img_up > 10 ms plus the one during warm-up;
  * those frames have img_up up to 41.56 ms against a p50 of 1.078 ms;
  * every other frame sits at 0.30-1.10 ms.
Two bounding cases are printed rather than one number, because "how much of the
excess belongs to the reallocation" is a judgement: the tight case subtracts only
the part above p95, the loose case subtracts everything above p50.
"""
import csv
import math
import os

CSV = ("/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board/"
       ".bench_out/FIX7/profile_A_VIDEO00034_19700101_015052.csv")
RED_LINE = 40.0


def pct(values, q):
    s = sorted(values)
    return s[max(1, math.ceil(q / 100.0 * len(s))) - 1]


rows = list(csv.DictReader(open(CSV)))
steady = rows[100:]
tot = [float(r["total_ms"]) for r in steady]
up = [float(r["img_up_ms"]) for r in steady]

base_p50, base_p95 = pct(tot, 50), pct(tot, 95)
up_p50, up_p95 = pct(up, 50), pct(up, 95)

print(f"VIDEO00034, {len(steady)} steady frames (warmup 100 dropped)")
print(f"  measured total : p50 {base_p50:6.2f}  p95 {pct(tot,95):6.2f}  "
      f"p99 {pct(tot,99):6.2f}  max {max(tot):6.2f} ms")
print(f"  measured img_up: p50 {up_p50:6.3f}  p95 {up_p95:6.3f}  max {max(up):6.3f} ms")
print(f"  frames over {RED_LINE:.0f} ms: "
      f"{sum(1 for t in tot if t > RED_LINE)}")
print()

for label, floor in (("tight  (subtract excess above p95)", up_p95),
                     ("loose  (subtract everything above p50)", up_p50)):
    adj = [t - max(0.0, u - floor) for t, u in zip(tot, up)]
    over = sum(1 for a in adj if a > RED_LINE)
    print(f"  {label}")
    print(f"    p50 {pct(adj,50):6.2f}  p95 {pct(adj,95):6.2f}  p99 {pct(adj,99):6.2f}  "
          f"max {max(adj):6.2f} ms   frames>{RED_LINE:.0f}ms: {over}")
print()
print("  Both cases remove only img_up excess on the frames that have it; every other")
print("  stage is left exactly as measured. A residual breach would therefore have to")
print("  come from fit or kernel, which the trace shows flat at 5-7 ms and ~4 ms.")
