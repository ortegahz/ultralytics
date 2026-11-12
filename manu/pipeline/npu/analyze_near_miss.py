#!/usr/bin/env python
"""近失候选归因分析：谁差在哪一档、差多少、差在哪个参数上。

用法：
    python manu/pipeline/npu/analyze_near_miss.py [--ledger PATH] [--top 25] [--cohort]

输出：
  ① 六项判据（相对 Baseline_v2.0 = trial_9863）与基线精确线
  ② **同配方多种子队列**：同一配方不同 seed 的六项读数（看种子噪声有多大）
  ③ 全体有官方读数的臂里，**每一项的失败率**（哪一项最常卡人）
  ④ 近失候选表：过了几项 + 逐项差值 + 配方参数（用于设计下一波定向突变）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
from manu.pipeline.npu import nano_search as ns  # noqa: E402

LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
KEYS = ("nano", "nano4", "f1", "t1", "t2fp", "t3")


def load(path: str) -> dict:
    return json.load(open(path))


def metrics(st: dict) -> dict:
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    out = {"nano": pk.get("nano"), "nano4": pk.get("nano4"), "f1": pk.get("f1"),
           "t1": of.get("t1"), "t2fp": of.get("t2_fp"), "t3": of.get("t3")}
    return out


def base_line(led: dict) -> dict:
    """**用流水线自己的精确阈值**（台账里 9863 的实测读数，不���截断常量）。"""
    n, n4, f1 = ns.base_pick_thresholds()
    t1, fp, t3 = ns.base_official_thresholds(led)
    return {"nano": n, "nano4": n4, "f1": f1, "t1": t1, "t2fp": fp, "t3": t3}


def chk(m: dict, b: dict) -> dict:
    def ge(k):
        return None if m.get(k) is None else float(m[k]) >= float(b[k])
    def le(k):
        return None if m.get(k) is None else float(m[k]) <= float(b[k])
    return {"nano": ge("nano"), "nano4": ge("nano4"), "f1": ge("f1"),
            "t1": ge("t1"), "t2fp": le("t2fp"), "t3": ge("t3")}


def fmt(x, n=6):
    return "—" if x is None else f"{float(x):.{n}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--cohort", action="store_true", help="只打印同配方多种子队列")
    args = ap.parse_args()

    led = load(args.ledger)
    arms = led.get("arms") or {}
    b = base_line(led)
    print("=== ① 基线精确线（Baseline_v2.0 = trial_9863）===")
    print("   " + "  ".join(f"{k}={fmt(b[k])}" for k in KEYS) + "   (t2fp 为上限 ≤)")

    rows = []
    for t, st in arms.items():
        m = metrics(st)
        if not any(v is not None for v in m.values()):
            continue
        c = chk(m, b)
        known = [k for k in KEYS if c[k] is not None]
        npass = sum(1 for k in known if c[k])
        rows.append((t, st, m, c, npass, len(known)))

    if args.cohort:
        # 同配方队列：按 recipe key 分组，取 seed 最多的几组
        from collections import defaultdict
        groups = defaultdict(list)
        for t, st, m, c, npass, nk in rows:
            key = st.get("recipe_key") or st.get("recipe") or json.dumps(
                {k: (st.get("params") or {}).get(k) for k in
                 ("sw", "swl", "lr_floor", "warmup", "mix", "lr", "fb", "ow", "wd",
                  "mosaic", "scale", "translate", "clip", "thr", "epochs", "sched", "mr", "maxside")},
                sort_keys=True)
            groups[key].append((t, st, m, c, npass, nk))
        for key, g in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:4]:
            print(f"\n=== ② 同配方队列（n={len(g)}）===")
            print(f"   配方: {key[:160]}")
            print("   trial     nano      nano4     F1        T1        T2FP  T3        过/测")
            for t, st, m, c, npass, nk in sorted(g, key=lambda x: -x[4]):
                flags = "".join("✔" if c[k] else ("✘" if c[k] is False else "?") for k in KEYS)
                print(f"   {t:>6} {fmt(m['nano'])} {fmt(m['nano4'])} {fmt(m['f1'])} "
                      f"{fmt(m['t1'])} {fmt(m['t2fp'],0):>4} {fmt(m['t3'])}  {flags} {npass}/{nk}")
        return

    # ③ 每一项的失败率（限定在「NanoBench 三项全过」的臂上 = 真正的竞争者）
    comp = [r for r in rows if r[3]["nano"] and r[3]["nano4"] and r[3]["f1"]]
    print(f"\n=== ③ 竞争者（nano/nano4/F1 三项全过）共 {len(comp)} 臂：各项失败率 ===")
    for k in ("t1", "t2fp", "t3"):
        f_ = sum(1 for r in comp if r[3][k] is False)
        print(f"   {k}: 失败 {f_}/{len(comp)} = {100.0*f_/max(1,len(comp)):.1f}%")

    # ④ 近失候选表
    scored = sorted(rows, key=lambda r: (-r[4], -(r[1].get("picked") or {}).get("nano") or 0))
    print(f"\n=== ④ 近失候选 top {args.top}（过/测 降序，nano 降序）===")
    print("   trial  过/测  nano      nano4     F1        T1        T2FP   T3        | 差值(nano/T1/T2FP/T3) | 配方")
    for t, st, m, c, npass, nk in scored[:args.top]:
        p = st.get("params") or {}
        d = []
        for k, better_higher in (("nano", 1), ("t1", 1), ("t2fp", -1), ("t3", 1)):
            if m.get(k) is None or b.get(k) is None:
                d.append("—")
            else:
                diff = float(m[k]) - float(b[k])
                d.append(f"{diff:+.4f}" if better_higher == 1 else f"{diff:+.1f}")
        recipe = (f"sw{p.get('sw')}|swl{p.get('swl')}|lrfl{p.get('lr_floor')}|"
                  f"wu{p.get('warmup')}|{p.get('mix')}|lr{p.get('lr')}|fb{p.get('fb')}|"
                  f"ow{p.get('ow')}|thr{p.get('thr')}|ep{p.get('epochs')}/{p.get('sched')}|"
                  f"mr{p.get('mr')}|wd{p.get('wd')}")
        print(f"   {t:>6} {npass}/{nk} {fmt(m['nano'])} {fmt(m['nano4'])} {fmt(m['f1'])} "
              f"{fmt(m['t1'])} {fmt(m['t2fp'],0):>4} {fmt(m['t3'])} | {' '.join(d)} | {recipe}")
        print(f"          seed={st.get('seed')} desc={str(st.get('desc'))[:70]}")


if __name__ == "__main__":
    main()
