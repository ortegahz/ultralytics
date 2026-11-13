"""Seed-noise audit: within a recipe group, how much does T1/T3 move across seeds?

Why: the six-criteria funnel says T3 is the final killer (2/12). Two very different
remedies:
  (A) more seeds of near-miss cells  -> only works if seed sigma >> gap
  (B) new recipes                   -> only needed if seed sigma << gap
This script measures (A) vs (B) directly from disk gate reports + ledger.

Read-only. CPU only. No GPU, no ledger writes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
GATE_DIR = f"{ROOT}/gate"
LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
NAME_RE = re.compile(r"^NANO_(\d+)_ep(\d+)_official$")

BASE = {
    "nano": 0.708648393194707,
    "nano4": 0.6472117202268431,
    "f1": 0.8897117275427806,
    "t1": 0.88808649190182,
    "t3": 0.9484893511639424,
    "t2_fp": 5,
}


def parse_report(path):
    try:
        rep = json.load(open(path))
    except Exception:
        return None
    tr = rep.get("tracks") or {}
    t1 = ((tr.get("track1_uav") or {}).get("finetune_peak") or {})
    t2 = tr.get("track2_golden") or {}
    t3 = tr.get("track3_blind") or {}
    if not t1.get("f1") or not t3.get("f1"):
        return None
    return {
        "t1": float(t1.get("f1", 0)), "t1_fp": int(t1.get("fp", 0)),
        "t2": float(t2.get("f1", 0)), "t2_fp": int(t2.get("fp", 0)),
        "t3": float(t3.get("f1", 0)), "t3_fp": int(t3.get("fp", 0)),
    }


def collect(ledger_path, gate_dir):
    led = json.load(open(ledger_path))
    arms = led.get("arms") or led
    rows = []
    for d in sorted(glob.glob(f"{gate_dir}/NANO_*_official")):
        m = NAME_RE.match(os.path.basename(d))
        if not m:
            continue
        trial, ep = int(m.group(1)), int(m.group(2))
        rp = os.path.join(d, "gate_report.json")
        if not os.path.isfile(rp):
            continue
        g = parse_report(rp)
        if not g:
            continue
        st = arms.get(str(trial)) or {}
        nb = (st.get("nanobench") or {}).get(str(ep))
        if not nb:
            continue
        p = st.get("params") or {}
        rows.append({
            "trial": trial, "ep": ep, "seed": st.get("seed"),
            "nano": nb.get("nano"), "nano4": nb.get("nano4"), "f1": nb.get("f1"),
            "t1": g["t1"], "t2_fp": g["t2_fp"], "t3": g["t3"],
            "sw": p.get("sw"), "swl": p.get("swl"), "lrf": p.get("lr_floor"),
            "mix": p.get("mix"), "thr": p.get("thr"), "ms": p.get("maxside"),
            "sched": p.get("sched"), "init": (p.get("init") or "").split("/")[-1],
        })
    return rows


def rkey(r):
    return (r["sw"], r["swl"], r["lrf"], r["mix"], r["thr"], r["ms"], r["sched"], r["init"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--gate-dir", default=GATE_DIR)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    rows = collect(a.ledger, a.gate_dir)
    if not rows:
        print("no paired readings")
        return
    print(f"paired (arm, epoch) rows with full readings: {len(rows)}")

    # --- 1. per-metric seed noise within recipe groups (groups with >=2 arms) ---
    groups = defaultdict(list)
    for r in rows:
        groups[rkey(r)].append(r)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"recipe groups: {len(groups)}  (with >=2 arms: {len(multi)})")

    stats = {}
    for metric in ("nano", "nano4", "f1", "t1", "t3"):
        sigmas = []
        for k, v in multi.items():
            vals = [r[metric] for r in v if r[metric] is not None]
            if len(vals) >= 2:
                sigmas.append(statistics.pstdev(vals))
        if sigmas:
            med = statistics.median(sigmas)
            stats[metric] = med
            print(f"  seed sigma {metric:<6} median within-group pstdev = {med:.6f} "
                  f"(n_groups={len(sigmas)}, max={max(sigmas):.6f})")
    t3sig = stats.get("t3", 0.0)
    t1sig = stats.get("t1", 0.0)

    # --- 2. T3 pass rate as a function of mean T3 of the group (calibration) ---
    print("\n[T3 calibration] group mean T3 -> individual T3 pass rate")
    buckets = []
    for k, v in groups.items():
        vals = [r["t3"] for r in v]
        if not vals:
            continue
        m = statistics.mean(vals)
        npass = sum(1 for x in vals if x >= BASE["t3"])
        buckets.append((m, npass, len(vals)))
    buckets.sort()
    for lo, hi in [(0.9480, 0.948489), (0.948489, 0.9489), (0.9489, 0.9495),
                   (0.9495, 0.9505), (0.9505, 10.0)]:
        sel = [b for b in buckets if lo <= b[0] < hi]
        if not sel:
            continue
        tot = sum(b[2] for b in sel)
        ps = sum(b[1] for b in sel)
        print(f"  meanT3 in [{lo:.6f},{hi:.6f}): groups={len(sel):3d} arms={tot:3d} "
              f"T3-pass={ps:3d} rate={ps / tot * 100:.1f}%")

    # --- 3. how close are the near-miss groups? (is seed enough?) ---
    print("\n[near-miss groups] mean T3 within 1e-3 of the line, with >=1 arm")
    near = []
    for k, v in groups.items():
        vals = [r["t3"] for r in v]
        if not vals:
            continue
        m = statistics.mean(vals)
        if BASE["t3"] - 1e-3 <= m <= BASE["t3"] + 1e-3:
            near.append((m, len(v), sum(1 for x in vals if x >= BASE["t3"]), k, v))
    near.sort()
    for m, n, np_, k, v in near[: a.top]:
        tags = " ".join(f"t3={r['t3']:.6f}{'P' if r['t3'] >= BASE['t3'] else 'x'}" for r in v)
        print(f"  meanT3={m:.6f} n={n} pass={np_} sw={k[0]} swl={k[1]} lrf={k[2]} "
              f"mix={k[3]} thr={k[4]} ms={k[5]} | {tags}")
    print(f"  (total near-miss groups: {len(near)})")

    # --- 4. decisive verdict ---
    gaps = [BASE["t3"] - r["t3"] for r in rows if r["t3"] < BASE["t3"]]
    gaps.sort()
    if gaps:
        med_gap = statistics.median(gaps)
        print(f"\n[verdict] median T3 gap of failing arms = {med_gap:.6f}; "
              f"seed sigma(t3) = {t3sig:.6f} (ratio {med_gap / max(t3sig, 1e-9):.2f})")
        print("  => seed-retry is the right lever" if med_gap <= 1.5 * t3sig
              else "  => recipe change is the right lever (seed noise too small)")
    print(f"[verdict] seed sigma(t1) = {t1sig:.6f}")


if __name__ == "__main__":
    main()
