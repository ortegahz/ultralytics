#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补回被并发写覆盖丢失的 R6 轮记录，并按**新规则**重算待复现队列（幂等）。

═══ 事故背景（2026-11-12 09:19~09:23 实测）═══
`gate_triad_arms.py` 09:10:58 读台账 → 09:19:36 写回 ⇒ 它手上的台账是**旧的**
（R6 轮记录 09:11:19 才写）⇒ **整个 R6 轮记录被覆盖丢失**（`rounds` 少一轮）
⇒ 守护的续跑轮号会算错（从 R6 重跑 = 3 h 白烧）。
⇒ 已修 `nano_search.save_ledger`（写盘前重读磁盘并合并 + flock 跨进程互斥）。

本脚本：
  ① 幂等地补回 R6 轮记录（字段取自 adaptive.log / night_log.md）；
  ② 按 runner 同规则重算 `search_state.replication_pending`，
     并**剔除 T2 死区配方**（swl<1.0 ⇒ 官方 T2 必塌 ⇒ 复现只是白烧）；
  ③ 校验 R6 的 8 个门禁臂官方读数仍在台账里。

用法（服务器）：python manu/pipeline/npu/restore_r6_record.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_search as ns  # noqa: E402
from adaptive_runner import base_ref, recipe_stats, recipe_t2_dead  # noqa: E402

R6_ARMS = list(range(12294, 12326))
R6_GATED = [12290, 12291, 12292, 12293, 12302, 12305, 12307, 12325]


def main() -> None:
    led = ns.load_ledger()
    base = base_ref(led)
    groups = recipe_stats(led)

    # ① R6 轮记录
    rounds = {int(r.get("round", 0)): r for r in (led.get("rounds") or []) if isinstance(r, dict)}
    if 6 not in rounds:
        rounds[6] = {
            "round": 6, "ts": "2026-11-12 09:11:19",
            "champion_before": None,
            "arms": R6_ARMS,
            "beat": [],
            "gated": R6_GATED,
            "board_measured": [],
            "replication_pending": [],
            "best_nano": 0.755198,
            "restored": "2026-11-12 09:23 由并发写覆盖事故后重建（字段取自 adaptive.log）",
        }
        print("[restore] 已补回 R6 轮记录")
    else:
        print("[restore] R6 轮记录已存在 ⇒ 跳过")
    led["rounds"] = [rounds[k] for k in sorted(rounds)]

    # ② 待复现队列（runner 同规则 + 剔除 T2 死区）
    pending, dead = [], []
    for g in groups.values():
        if g["n"] >= 2:
            continue
        if g["max_nano"] < base["nano"]:
            continue
        if recipe_t2_dead(led, g, base):
            dead.append(g["key"])
            continue
        pending.append(g["key"])
    pending = pending[:8]
    led.setdefault("search_state", {})["replication_pending"] = pending
    for r in led["rounds"]:
        if int(r.get("round", 0)) == 6:
            r["replication_pending"] = pending
    ns.save_ledger(led)

    print(f"[restore] 待复现（{len(pending)}）：")
    for rk in pending:
        print("   ", rk.split("|")[0][:70])
    print(f"[restore] 剔除 T2 死区（{len(dead)}）：")
    for rk in dead:
        print("   ", rk.split("|")[0][:70])

    # ③ 校验 R6 门禁臂读数
    miss = []
    for t in R6_GATED:
        of = ((led.get("arms") or {}).get(str(t)) or {}).get("official") or {}
        if of.get("t1") is None:
            miss.append(t)
    print(f"[restore] R6 门禁臂读数缺失：{miss or '无'}")
    print(f"[restore] rounds = {[r.get('round') for r in led['rounds']]}")


if __name__ == "__main__":
    main()
