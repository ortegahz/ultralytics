"""Per-recipe-cell joint yield audit (read-only, CPU only).

For every recipe cell (sw/swl/lrf/mix/thr/ms/sched/init) collect all (arm, epoch) rows
with full readings, then report per-criterion pass counts and the joint 6/6 rate.
Ranking guide for the remaining GPU budget: cells whose *joint* rate is highest AND
whose seeds are still few are where new arms must go.

Also prints the current champion recipe and the seed-scan target, because the seed scan
is the single biggest arm group and it must point at the best cell.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

from seed_noise_audit import BASE, collect, rkey  # reuse readers

LEDGER_PATH = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
GATE_DIR = "/mnt/data/siping/runs/npu20ms_offcal/gate"

CRIT = ["nano", "nano4", "f1", "t1", "t2fp", "t3"]


def passes(r):
    return {
        "nano": r["nano"] is not None and r["nano"] >= BASE["nano"],
        "nano4": r["nano4"] is not None and r["nano4"] >= BASE["nano4"],
        "f1": r["f1"] is not None and r["f1"] >= BASE["f1"],
        "t1": r["t1"] >= BASE["t1"],
        "t2fp": r["t2_fp"] <= BASE["t2_fp"],
        "t3": r["t3"] >= BASE["t3"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    led = json.load(open("/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"))
    rounds = led.get("rounds") or []
    if rounds:
        last = rounds[-1]
        print(f"[ledger] last round r{last.get('round')} ts={last.get('ts')} "
              f"arms={len(last.get('arms') or [])} beat={last.get('beat')} "
              f"champion_before={(last.get('champion_before') or '')[:90]}")
    print(f"[ledger] replication_pending = {len((led.get('search_state') or {}).get('replication_pending') or [])}")

    rows = collect(LEDGER_PATH, GATE_DIR)
    led2 = json.load(open(LEDGER_PATH))
    arms2 = led2.get("arms") or led2
    synth = {int(t) for t, st in arms2.items() if (st.get("params") or {}).get("synth")
             or st.get("synth") or (st.get("params") or {}).get("init") is None}
    seen_pair = set()
    dedup = []
    n_drop = 0
    for r in sorted(rows, key=lambda x: (x["trial"], x["ep"])):
        pair = (rkey(r), r["seed"])
        if pair in seen_pair:
            n_drop += 1
            continue
        seen_pair.add(pair)
        dedup.append(r)
    print(f"rows={len(rows)}  dedup(arm,epoch)={len(dedup)}  dropped duplicate (recipe,seed) rows={n_drop}")
    print(f"synth arms among rows: {sum(1 for r in dedup if r['trial'] in synth)}")
    rows = dedup
    groups = defaultdict(list)
    for r in rows:
        groups[rkey(r)].append(r)

    scored = []
    for k, v in groups.items():
        fl = [passes(r) for r in v]
        n = len(v)
        cnt = {c: sum(1 for f in fl if f[c]) for c in CRIT}
        joint = sum(1 for f in fl if all(f.values()))
        mean = {c: statistics.mean([r[c] for r in v if r[c] is not None])
                for c in ("nano", "nano4", "f1", "t1", "t3")}
        scored.append({
            "k": k, "n": n, "cnt": cnt, "joint": joint, "mean": mean, "v": v,
        })

    scored.sort(key=lambda s: (-s["joint"], -s["cnt"]["t3"], -s["mean"]["nano"]))
    print("\n===== cells ranked by joint 6/6 hits =====")
    print(f"{'joint':>5} {'n':>3} {'nano':>4} {'nano4':>5} {'f1':>3} {'t1':>3} {'t2':>3} {'t3':>3} "
          f"| {'m_nano':>8} {'m_nano4':>8} {'m_f1':>8} {'m_t1':>8} {'m_t3':>8} | cell")
    for s in scored[: a.top]:
        k = s["k"]
        c = s["cnt"]
        m = s["mean"]
        print(f"{s['joint']:>5} {s['n']:>3} {c['nano']:>4} {c['nano4']:>5} {c['f1']:>3} "
              f"{c['t1']:>3} {c['t2fp']:>3} {c['t3']:>3} | {m['nano']:8.6f} {m['nano4']:8.6f} "
              f"{m['f1']:8.6f} {m['t1']:8.6f} {m['t3']:8.6f} | sw={k[0]} swl={k[1]} lrf={k[2]} "
              f"mix={k[3]} thr={k[4]} ms={k[5]} sched={k[6]} init={k[7]}")

    # --- cells that are "one criterion away" at the group-mean level (systematic) ---
    print("\n===== cells whose group MEAN passes >=3 criteria (systematic near-miss) =====")
    print("  (a cell whose mean clears a line but individual seeds still lose it is a "
          "seed-lottery cell; whose mean fails it is a recipe-lift cell)")
    for s in scored:
        m = s["mean"]
        ok = []
        if m["nano"] >= BASE["nano"]:
            ok.append("nano")
        if m["nano4"] >= BASE["nano4"]:
            ok.append("nano4")
        if m["f1"] >= BASE["f1"]:
            ok.append("f1")
        if m["t1"] >= BASE["t1"]:
            ok.append("t1")
        if m["t3"] >= BASE["t3"]:
            ok.append("t3")
        if len(ok) >= 3:
            k = s["k"]
            miss = [c for c in ("nano", "nano4", "f1", "t1", "t3")
                    if m[c] < BASE[c]]
            gaps = " ".join(f"{c}{m[c]-BASE[c]:+.6f}" for c in miss)
            print(f"  sw={k[0]} swl={k[1]} lrf={k[2]} mix={k[3]} thr={k[4]} ms={k[5]} "
                  f"n={s['n']} joint={s['joint']} | pass@mean={ok} gaps={gaps or 'none'}")

    # --- how many arms of each cell are left (tried seeds) ---
    print("\n===== arms already spent per top cell (trial ids) =====")
    for s in scored[:12]:
        trials = sorted({r["trial"] for r in s["v"]})
        eps = sorted({r["ep"] for r in s["v"]})
        print(f"  sw={s['k'][0]} swl={s['k'][1]} lrf={s['k'][2]} mix={s['k'][3]} thr={s['k'][4]} "
              f"ms={s['k'][5]} sched={s['k'][6]} init={s['k'][7]} -> trials={trials} eps={eps}")


if __name__ == "__main__":
    main()
