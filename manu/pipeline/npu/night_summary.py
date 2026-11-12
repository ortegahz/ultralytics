#!/usr/bin/env python
"""一夜搜索结果速览（明早一条命令出结论）。

用法：
    python manu/pipeline/npu/night_summary.py [--ledger PATH] [--top 12]

输出四块：
  ① 总量（臂数 / 轮数 / 训练臂 / 有官方读数的臂）
  ② **六项全过的臂**（nano / nano4 / F1 / T1 / T2FP / T3 相对 Baseline_v2.0=trial_9863）
  ③ **最接近的候选**（按「过了几项」+ nano 排序，附逐 epoch 读数）
  ④ 配方组统计（n / 均值 nano / 最佳 nano / 三项全过 epoch 数 / 官方 T1 过线数）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
from manu.pipeline.npu import nano_search as ns  # noqa: E402

REQUIRED = ("nano", "nano4", "f1", "t1", "t2fp", "t3")


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def base_of(led: dict) -> dict:
    base = dict(ns.BASE_V2)
    a = (led.get("arms") or {}).get(str(ns.BASE_V2["trial"]), {})
    pk = a.get("picked") or {}
    of = a.get("official") or {}
    for k in ("nano", "nano4", "f1"):
        if pk.get(k) is not None:
            base[k] = pk[k]
    for k in ("t1", "t2_fp", "t3"):
        if of.get(k) is not None:
            base[k] = of[k]
    return base


def checks(st: dict, base: dict) -> dict:
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    out = {}
    if pk.get("nano") is not None:
        out["nano"] = pk["nano"] >= base["nano"]
    if pk.get("nano4") is not None:
        out["nano4"] = pk["nano4"] >= base["nano4"]
    if pk.get("f1") is not None:
        out["f1"] = pk["f1"] >= base["f1"]
    if of.get("t1") is not None:
        out["t1"] = of["t1"] >= base["t1"]
    if of.get("t2_fp") is not None:
        out["t2fp"] = of["t2_fp"] <= base["t2_fp"]
    if of.get("t3") is not None:
        out["t3"] = of["t3"] >= base["t3"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=ns.LEDGER)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    led = json.load(open(args.ledger))
    base = base_of(led)
    arms = led.get("arms") or {}
    rounds = led.get("rounds") or []

    trained = [t for t, s in arms.items() if s.get("trained")]
    synth = [t for t, s in arms.items() if s.get("synth")]
    gated = [t for t, s in arms.items() if (s.get("official") or {}).get("t1") is not None]
    print("=== ① 总量 ===")
    print(f"  台账臂 {len(arms)}（训练 {len(trained)} / 合成 {len(synth)}）｜完成轮 {len(rounds)}"
          f"｜有官方 T1 读数 {len(gated)}")
    print(f"  基线 Baseline_v2.0(trial_9863)：nano {base['nano']:.6f} nano4 {base['nano4']:.6f} "
          f"F1 {base['f1']:.6f} T1 {base['t1']:.6f} T2FP≤{base['t2_fp']} T3 {base['t3']:.6f}")
    if rounds:
        last = rounds[-1]
        print(f"  最后一轮：round={last.get('round')} 臂数={len(last.get('arms') or [])} "
              f"时刻={last.get('ts')}")

    scored = []
    for t, st in arms.items():
        c = checks(st, base)
        if not c:
            continue
        scored.append((sum(1 for v in c.values() if v), len(c), t, st, c))
    scored.sort(key=lambda x: (-x[0], -x[1], -(f((x[3].get("picked") or {}).get("nano")) or 0)))

    win = [s for s in scored if len(s[4]) == 6 and all(s[4].values())]
    print(f"\n=== ② 六项全过的臂（{len(win)}）===")
    if not win:
        print("  （无）⇒ 今晚没有直接可升轨的臂")
    for n, m, t, st, c in win:
        pk = st.get("picked") or {}
        of = st.get("official") or {}
        print(f"  ✔ trial {t} ep={st.get('picked_epoch')} rule={st.get('select_rule')} "
              f"nano={pk.get('nano'):.6f} nano4={pk.get('nano4'):.6f} F1={pk.get('f1'):.6f} "
              f"T1={of.get('t1'):.6f} T2FP={of.get('t2_fp')} T3={of.get('t3'):.6f} seed={st.get('seed')}")
        print(f"      desc: {st.get('desc')}")

    print(f"\n=== ③ 最接近的候选（前 {args.top}）===")
    print("  trial  过/测  nano       nano4     F1        T1        T2FP   T3        seed")
    for n, m, t, st, c in scored[: args.top]:
        pk = st.get("picked") or {}
        of = st.get("official") or {}

        def s(k, v, fmt="%.6f"):
            x = f(v)
            return (fmt % x) if x is not None else "   —   "
        print(f"  {t:>6} {n}/{m}   {s('nano', pk.get('nano'))} {s('nano4', pk.get('nano4'))} "
              f"{s('f1', pk.get('f1'))} {s('t1', of.get('t1'))} "
              f"{str(of.get('t2_fp') if of.get('t2_fp') is not None else '—'):>5} "
              f"{s('t3', of.get('t3'))} {st.get('seed')}")
    if scored:
        t = scored[0][2]
        st = arms[str(t)]
        print(f"\n  ── trial {t} 的逐 epoch 读数 ──")
        for ep in sorted((st.get("nanobench") or {}).keys(), key=lambda x: int(x)):
            mm = (st.get("nanobench") or {}).get(ep) or {}
            print(f"     ep{ep}: nano={mm.get('nano')} nano4={mm.get('nano4')} F1={mm.get('f1')}")

    print("\n=== ④ 配方组统计（n≥2，按均值 nano 降序）===")
    try:
        from manu.pipeline.npu import adaptive_runner as ar  # noqa: PLC0415
        groups = ar.recipe_stats(led)
    except Exception as e:  # noqa: BLE001
        print(f"  （配方统计不可用：{e}）")
        groups = {}
    print("  n   均值nano   最佳nano   三项全过epoch  T1过线  配方")
    for g in sorted(groups.values(), key=lambda x: -(x.get("mean_nano") or 0)):
        if g.get("n", 0) < 2:
            continue
        all3 = 0
        t1ok = 0
        for t in g.get("trials", []):
            st = arms.get(str(t)) or {}
            traj = st.get("nanobench") or {}
            eps = [1 for m in traj.values()
                   if m and (f(m.get("nano")) or -1) >= base["nano"]
                   and (f(m.get("nano4")) or -1) >= base["nano4"]
                   and (f(m.get("f1")) or -1) >= base["f1"]]
            all3 += len(eps)
            of = st.get("official") or {}
            if of.get("t1") is not None and of["t1"] >= base["t1"]:
                t1ok += 1
        print(f"  {g['n']:<3} {g.get('mean_nano', 0):.6f}   {g.get('max_nano', 0):.6f}   "
              f"{all3:<12} {t1ok:<7} {g.get('key', '')[:96]}")


if __name__ == "__main__":
    main()
