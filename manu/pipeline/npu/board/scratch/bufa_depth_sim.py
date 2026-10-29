#!/usr/bin/env python3
"""
bufa_depth_sim.py -- OFFLINE simulation of the Buffer A depth's effect.

WHY THIS EXISTS
---------------
The board arm CANON22 asked for --buffer-a-depth 16 and silently ran depth 3
(the clamp table's ceiling was 8, and an out-of-range request falls back to
the reference default).  Before spending another board run on depths 16/32/64,
we simulate what each depth SHOULD deliver, so the arm matrix is aimed at the
interesting depths instead of a blind sweep.

THE MODEL
---------
The GMC is the only producer; it emits groups at a rate that is NOT constant:
the anchor fit has a long tail (p50 41.6 ms, p95 130 ms, p99 184 ms, max
246 ms), and when a fit is slow the GMC catches up afterwards with a burst.

  * producer:  GMC groups, produced when a fit completes
  * consumer:  Stage 2, at most 2 services x (1 / 121.9 ms) = 16.4 fps
  * Buffer A:  a bounded ring of D slots per service; on overflow the OLDEST
               READY slot is reclaimed (that frame is lost) -- the "drop
               oldest" policy, and the consumer NEVER blocks the producer.

The GMC's group cadence is reconstructed from the board's own accounting:
  CANON21: gmc_groups_done = 7195 over total_wall = 472.76 s
           -> mean 15.22 groups/s;  gmc_fit p50/p95/p99/max as above.

We replay the measured GMC group arrival times on a 1 ms grid and let two
Stage-2 services drain them, each with its own D-slot ring.

OUTPUT
------
For each depth: delivered frames, lost frames, effective fps, and the
resulting e2e latency (queueing delay).
"""
import sys
from dataclasses import dataclass, field


@dataclass
class Sim:
    # --- the measured CANON21 baseline (board, 11810 frames, pace 40 ms) -----
    gmc_groups: int = 7195
    wall_s: float = 472.763
    # Stage 2's per-frame cost: rknn_run 121.9186 ms + set_io_mem + outputs_get
    stage2_ms: float = 121.9186 + 0.9
    n_services: int = 2
    # The GMC's fit-cost distribution (board timing, CANON21)
    fit_p50: float = 41.618
    fit_p95: float = 130.222
    fit_p99: float = 184.003
    fit_max: float = 246.018
    # the camera's own ceiling: the GMC can never see more than 25 groups/s
    max_rate: float = 25.0

    def gmc_arrivals(self, burstiness: float):
        """Replay the GMC's group arrivals on a 1 ms grid.

        The GMC's cadence is the fit cost plus the catch-up burst that follows
        a slow fit.  We model it as: each group costs a fit time drawn from the
        measured p50/p95/p99/max, and a fit slower than p50 is followed by a
        catch-up burst of `burstiness` extra groups arriving back to back.
        """
        # Deterministic, reproducible draw from the measured percentiles.
        # We walk the percentiles so the tail is represented exactly.
        ladder = [
            (0.50, self.fit_p50), (0.90, self.fit_p95),
            (0.99, self.fit_p99), (1.00, self.fit_max),
        ]
        out = []
        t = 0.0
        i = 0
        n = self.gmc_groups
        while len(out) < n:
            # pick the percentile bucket by a deterministic sawtooth so the
            # distribution's shape (mostly p50, a long tail) is reproduced
            frac = (i % 1000) / 1000.0
            cost = self.fit_p50
            for thr, val in ladder:
                if frac <= thr:
                    cost = val
                    break
            i += 1
            t += cost / 1000.0
            out.append(t)
            # a slow fit leaves frames unprocessed -> the GMC catches up
            if cost > self.fit_p95 and burstiness > 0:
                for _ in range(burstiness):
                    out.append(t)
                    if len(out) >= n:
                        break
        return out[:n]

    def run(self, depth: int, burstiness: int = 3):
        """Two independent Stage-2 services, each with its own D-slot ring.

        The GMC's groups are handed to the service that is READY first, which
        is what the 1:1 pairing does in practice (both Stage-1 threads pull
        from the same feat_q).
        """
        arrivals = self.gmc_arrivals(burstiness)
        if not arrivals:
            return None
        total = self.wall_s

        # Each service: a ring of `depth` slots, and a "free at" timestamp.
        svc_free = [0.0, 0.0]
        svc_ring = [[] for _ in range(self.n_services)]   # pending arrival times
        delivered = 0
        lost = 0
        latencies = []
        idx = 0
        t = 0.0
        # Event loop over a 1 ms grid.
        step = 0.001
        while t <= total and idx < len(arrivals):
            # services whose previous frame just finished release a slot
            for s in range(self.n_services):
                if svc_free[s] <= t:
                    svc_free[s] = t
            # deliver all arrivals that have happened
            while idx < len(arrivals) and arrivals[idx] <= t:
                at = arrivals[idx]
                idx += 1
                # choose the service with the shortest queue (they are peers)
                s = min(range(self.n_services),
                        key=lambda k: (len(svc_ring[k]), svc_free[k]))
                ring = svc_ring[s]
                if len(ring) >= depth:
                    # DROP OLDEST: the oldest READY frame is reclaimed
                    ring.pop(0)
                    lost += 1
                ring.append(at)
            # run the services
            for s in range(self.n_services):
                if svc_ring[s] and svc_free[s] <= t:
                    at = svc_ring[s].pop(0)
                    latencies.append((t - at) * 1000.0)
                    delivered += 1
                    svc_free[s] = t + self.stage2_ms / 1000.0
            t += step

        # drain whatever is left at the end of the wall clock
        for s in range(self.n_services):
            while svc_ring[s]:
                at = svc_ring[s].pop(0)
                if svc_free[s] <= total + self.stage2_ms / 1000.0:
                    latencies.append((max(total, svc_free[s]) - at) * 1000.0)
                    delivered += 1
                    svc_free[s] += self.stage2_ms / 1000.0
                else:
                    lost += 1
        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        return {
            "depth": depth,
            "delivered": delivered,
            "lost": lost,
            "fps": delivered / total,
            "lat_p50": p50,
            "lat_p95": p95,
        }


def main():
    sim = Sim()
    print("=" * 78)
    print("OFFLINE Buffer A depth simulation (CANON21's measured GMC cadence)")
    print("=" * 78)
    print(f"GMC supply      : {sim.gmc_groups} groups / {sim.wall_s:.1f} s "
          f"= {sim.gmc_groups / sim.wall_s:.2f} groups/s")
    print(f"Stage 2 capacity: {sim.n_services} services x "
          f"{sim.stage2_ms:.1f} ms = {sim.n_services * 1000 / sim.stage2_ms:.2f} fps")
    print()
    print(f"{'burst':>6} {'D':>4} {'delivered':>10} {'lost':>7} {'fps':>8} "
          f"{'lat_p50':>9} {'lat_p95':>9}")
    print("-" * 78)
    for burstiness in (0, 2, 4, 8):
        for depth in (3, 8, 16, 32, 64):
            r = sim.run(depth, burstiness)
            if r is None:
                continue
            print(f"{burstiness:>6} {r['depth']:>4} {r['delivered']:>10} "
                  f"{r['lost']:>7} {r['fps']:>8.3f} {r['lat_p50']:>9.1f} "
                  f"{r['lat_p95']:>9.1f}")
        print("-" * 78)

    # The board's ACTUAL CANON21 numbers, for calibration:
    print()
    print("BOARD CANON21 (D=3, measured): delivered 5628, lost 1567, "
          "fps 11.904, lat_p50 315.6 ms")
    print("=> compare the sim's D=3 row against this; the sim is a MODEL, not")
    print("   a measurement, and only its SHAPE (how depth trades frames for")
    print("   latency) is used to aim the arms.")


if __name__ == "__main__":
    main()
