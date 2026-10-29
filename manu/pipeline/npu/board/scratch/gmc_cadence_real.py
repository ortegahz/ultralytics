#!/usr/bin/env python3
"""
gmc_cadence_real.py -- reconstruct the GMC's REAL burst structure from the
board's own frame-id sequence, and test the "load imbalance" hypothesis for
the 1567 frames CANON21 lost in Buffer A.

WHY THIS IS THE RIGHT QUESTION
-------------------------------
CANON21 (board, decoupled, Buffer A depth 3) reported:
    features_dropped = 0      -> the GMC->Stage1 queue (feat_q) never overflowed
    bufA_dropped     = 1567   -> EVERY lost group was lost INSIDE Buffer A
    stage2_wait_ms   = 254322 -> the NPU stage was STARVED, waiting for work
    npu_util_pct     = 72.569 -> the NPU was not the bottleneck

And the per-core hardware load:
    core0 mean 83.45%   core1 mean 41.28%   core2 mean 0.00%

If the two NPU services were equally loaded, their two cores would show
NEARLY IDENTICAL occupancy, because both run the same model at the same
121.9 ms per frame.  core0 being 2x core1 means the two services are NOT
equally loaded: one service is saturated and the other is starving.

So the loss mechanism is probably NOT "a 3-slot ring cannot hold a burst".
It is probably:  the two stages are paired 1:1, but they share ONE feat_q,
so whichever service's ring happens to empty first wins the next feature.
The faster service's ring empties first -> it wins more features -> it
saturates -> its ring overflows -> IT drops the oldest frames, while the
slower service's ring stays empty and its NPU waits.

If that is true, then:
  * a deeper Buffer A helps only a LITTLE (it hides the overflow, it does
    not fix the imbalance), and
  * the REAL fix is to balance the two services.

This script reconstructs the GMC's cadence from the measured frame ids and
estimates how much of the loss the imbalance explains.
"""
import sys
from collections import deque

SEQ_MS = 40.0        # the board's --pace-ms 40 => 25 fps camera
WALL_S = 472.763     # CANON21 total_wall


def load_frame_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ids.append(int(parts[1]))
    return ids


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "canon21_feature_md5.txt"
    ids = load_frame_ids(path)
    n = len(ids)
    print(f"groups from {path}: {n}")

    # ---- 1. the GMC's cadence in TIME -----------------------------------
    # A group's frame id is the physical camera frame it was computed from.
    # At 25 fps the frame id maps to arrival time id*40 ms.
    times = [i * SEQ_MS / 1000.0 for i in ids]
    gaps = [times[i] - times[i - 1] for i in range(1, n)]
    gaps_sorted = sorted(gaps)

    def pct(v, p):
        return v[min(len(v) - 1, int(len(v) * p))]

    print()
    print("=== GMC group inter-arrival time (from the measured frame ids) ===")
    print(f"  mean {sum(gaps)/len(gaps)*1000:.1f} ms   p50 {pct(gaps_sorted,0.50)*1000:.1f} ms")
    print(f"  p95 {pct(gaps_sorted,0.95)*1000:.1f} ms   p99 {pct(gaps_sorted,0.99)*1000:.1f} ms")
    print(f"  max {max(gaps)*1000:.1f} ms")

    # ---- 2. the instantaneous rate, in 1 s bins --------------------------
    bins = {}
    for t in times:
        bins[int(t)] = bins.get(int(t), 0) + 1
    rates = sorted(bins.values())
    over = sum(1 for r in rates if r > 16)
    print()
    print("=== GMC instantaneous rate (1 s bins) ===")
    print(f"  mean {n/WALL_S:.2f} groups/s   p50 {pct(rates,0.50)}   "
          f"p90 {pct(rates,0.90)}   p99 {pct(rates,0.99)}   MAX {max(rates)}")
    print(f"  seconds with > 16 groups/s: {over} / {len(rates)}")

    # the worst burst: the longest run of seconds above the Stage-2 capacity
    STAGE2_FPS = 2 * 1000.0 / (121.9186 + 0.9)
    cap_per_sec = int(STAGE2_FPS)
    print(f"  (Stage-2 capacity = {STAGE2_FPS:.2f} fps -> {cap_per_sec}/s)")

    # how much backlog does a burst build up?
    backlog = 0
    max_backlog = 0
    backlog_curve = []
    for s in range(int(WALL_S) + 1):
        backlog += bins.get(s, 0) - cap_per_sec
        if backlog < 0:
            backlog = 0
        backlog_curve.append(backlog)
        max_backlog = max(max_backlog, backlog)
    print(f"  the worst backlog the GMC can build against a {cap_per_sec}/s "
          f"consumer: {max_backlog} groups")
    print(f"  -> a ring of D=3 CANNOT hold it; D>=16 can; D>=32 is safe margin")

    # ---- 3. the load-imbalance hypothesis --------------------------------
    # The board's per-core load says the two services split the work ~2:1.
    core0, core1 = 83.45, 41.28
    total = core0 + core1
    s0 = core0 / total
    print()
    print("=== the load-imbalance hypothesis ===")
    print(f"  core0 {core0}% vs core1 {core1}%  -> service0 gets "
          f"{s0*100:.1f}% of the frames")
    # If service 0 is saturated and service 1 is idle, the LOSS is bounded
    # by how much MORE service 0 was offered than it could take.
    delivered = 5628
    lost = 1567
    groups = 7195
    print(f"  measured: {groups} groups -> {delivered} delivered, {lost} lost "
          f"in Buffer A")
    print(f"  if the split were {s0*100:.0f}/{100-s0*100:.0f}, service0 would be "
          f"offered ~{groups*s0:.0f} groups")
    per_service_cap = STAGE2_FPS / 2 * WALL_S
    print(f"  one service can take {per_service_cap:.0f} frames in {WALL_S:.0f} s "
          f"({STAGE2_FPS/2:.2f} fps)")
    offered0 = groups * s0
    if offered0 > per_service_cap:
        print(f"  => service0 was offered {offered0:.0f} > its capacity "
              f"{per_service_cap:.0f}: it MUST drop "
              f"{offered0-per_service_cap:.0f} frames")
        print(f"     and the measured loss is {lost} -> the imbalance explains "
              f"{(offered0-per_service_cap)/lost*100:.0f}% of it")
    else:
        print(f"  => service0 was NOT over its capacity ({offered0:.0f} <= "
              f"{per_service_cap:.0f}); the imbalance alone does not explain "
              f"the loss")

    # ---- 4. what the per-service split implies for the fps ceiling --------
    print()
    print("=== the fps ceiling under the imbalance ===")
    # a saturated service0 caps at STAGE2_FPS/2, service1 adds whatever it gets
    print(f"  balanced split : min({groups}, {STAGE2_FPS:.2f} fps * {WALL_S:.0f} s) "
          f"= {min(groups, STAGE2_FPS*WALL_S):.0f} frames "
          f"= {min(groups, STAGE2_FPS*WALL_S)/WALL_S:.2f} fps")
    print(f"  measured CANON21: {delivered/WALL_S:.3f} fps")
    print(f"  => the gap is what a deeper (and BALANCED) ring could recover.")


if __name__ == "__main__":
    main()
