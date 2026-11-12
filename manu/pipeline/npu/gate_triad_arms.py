#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对「NanoBench 三项全平基线」的臂补跑官方三轨（GMC）门禁。

═══ 为什么需要这个脚本（2026-11-12 09:00 实测发现）═══
夜间调度器的门禁窗口是「nano 前 8 名」⇒ **nano 高但 F1 塌掉的臂把窗口占满**
⇒ 真正「nano/nano4/F1 三项全平基线」的臂反被标记 `nano_below_gate_window` **从未跑官方三轨**
⇒ 六项判据永远缺 T1/T2/T3 ⇒ **无法升轨**（实测 4 个臂：12300 / 12263 / 12252 / 12053）。

本脚本：**不按 nano 排序，按「三项全平」筛选** ⇒ 只要三项全平就一定补跑官方三轨。
幂等：官方三轨有缓存（gate_report.json 存在则复用）。
用法（服务器）：
    setsid nohup python manu/pipeline/npu/gate_triad_arms.py [--gpus 0,1,2,3] [--topk 0] \
        > /mnt/data/siping/runs/npu20ms/nano_search/triad_gate.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_search as ns  # noqa: E402


def triad_pass(a: dict, base: dict) -> bool:
    p = a.get("picked") or {}
    return ((p.get("nano") or 0) >= base["nano"]
            and (p.get("nano4") or 0) >= base["nano4"]
            and (p.get("f1") or 0) >= base["f1"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--topk", type=int, default=0, help="额外再按 nano 补几个（0=不补）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.replace(" ", "").split(",") if x.strip()]

    led = ns.load_ledger()
    base = led.get("baseline_v2") or {}
    arms_all = led.get("arms") or {}
    tri = []
    for k, a in arms_all.items():
        if triad_pass(a, base):
            tri.append((int(k), a))
    tri.sort(key=lambda x: -((x[1].get("picked") or {}).get("nano") or 0))
    if args.topk:
        extra = sorted(
            ((int(k), a) for k, a in arms_all.items()
             if int(k) not in {t for t, _ in tri}),
            key=lambda x: -((x[1].get("picked") or {}).get("nano") or 0))[:args.topk]
        tri += extra
    print(f"[triad-gate] 基线 nano={base.get('nano'):.6f} nano4={base.get('nano4'):.6f} "
          f"F1={base.get('f1'):.6f}｜三项全平臂 {len(tri)} 个")
    for t, a in tri:
        p = a.get("picked") or {}
        print(f"   trial {t} ep{a.get('picked_epoch')} nano={p.get('nano'):.6f} "
              f"nano4={p.get('nano4'):.6f} F1={p.get('f1'):.6f}")
    if args.dry_run:
        return

    items, gpu_of = [], {}
    for i, (t, a) in enumerate(tri):
        items.append((t, int(a.get("seed") or 0), a.get("params") or {}, a.get("desc") or ""))
        gpu_of[t] = gpus[i % len(gpus)]
    ns.phase_gate(items, gpu_of)

    led = ns.load_ledger()
    out = {"ts": time.strftime("%F %T"), "base": {k: base.get(k) for k in
           ("trial", "nano", "nano4", "f1", "t1", "t2", "t3", "t3_fp")}, "results": []}
    for t, _a in tri:
        st = (led.get("arms") or {}).get(str(t)) or {}
        p = st.get("picked") or {}
        o = st.get("official") or {}
        row = {"trial": t, "epoch": st.get("picked_epoch"),
               "nano": p.get("nano"), "nano4": p.get("nano4"), "f1": p.get("f1"),
               "t1": o.get("t1"), "t2": o.get("t2"), "t2_fp": o.get("t2_fp"),
               "t3": o.get("t3"), "t3_fp": o.get("t3_fp"),
               "verdict": o.get("verdict"), "error": o.get("error")}
        ok = (row["nano"] >= base["nano"] and row["nano4"] >= base["nano4"]
              and row["f1"] >= base["f1"] and (row["t1"] or 0) >= base["t1"]
              and (row["t2"] or 0) >= base["t2"] and (row["t3"] or 0) >= base["t3"]
              and (row["t3_fp"] if row["t3_fp"] is not None else 10 ** 9) <= base["t3_fp"])
        row["six_ok"] = bool(ok)
        out["results"].append(row)
        print(f"[triad-gate] trial {t} ep{row['epoch']} nano={row['nano']} nano4={row['nano4']} "
              f"F1={row['f1']} | T1={row['t1']} T2={row['t2']}(fp {row['t2_fp']}) "
              f"T3={row['t3']}(fp {row['t3_fp']}) ⇒ 六项全平={row['six_ok']}")
    path = f"{ns.WORK}/triad_gate_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[triad-gate] 报告已落盘：{path}")


if __name__ == "__main__":
    main()
