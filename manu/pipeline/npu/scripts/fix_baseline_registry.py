#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性修复：**基线名通胀**（2026-11-18 09:12）

事实：昨夜 auto_promote_watch 把同一个 trial_1117002 反复升轨 82 次
      ⇒ registry 里 89 条（Baseline_v2.4 ~ Baseline_v2.86 全是同一份权重）
      ⇒ baseline_v2.name = "Baseline_v2.86"，名不副实。

处置：
  1. registry 里**同 trial 的重复条目**只保留一条（最早的那条 id），
     状态置 ACTIVE / OFFICIAL BASELINE；其余折叠成一条 note（aliases 列表）。
  2. baseline_v2.name 复位成 "Baseline_v2.4"（该名对应的就是 trial_1117002 这份权重）。
  3. 其余历史条目（9863 / trial_0031 / trial_0080）原样保留。
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
W = "/mnt/data/siping/runs/npu20ms/nano_search"
LEDGER = os.path.join(W, "nano_search_ledger.json")
sys.path.insert(0, REPO)
os.chdir(REPO)
from manu.pipeline.npu import nano_search as NS  # noqa: E402


def main() -> None:
    led = NS.load_ledger()
    reg = list(led.get("baseline_registry") or [])
    retired = set(led.get("registry_retired_ids") or [])
    base = led.get("baseline_v2") or {}
    cur_trial = int(base.get("trial") or 0)
    seen: dict[int, dict] = {}
    others = []
    aliases: dict[int, list[str]] = {}
    for r in reg:
        t = int(r.get("trial") or 0)
        if t and t in seen:
            aliases.setdefault(t, []).append(str(r.get("id")))
            retired.add(str(r.get("id")))
            continue
        if t:
            seen[t] = r
        else:
            others.append(r)
    new_reg = []
    for t, r in seen.items():
        r = dict(r)
        if t == cur_trial:
            r["status"] = "ACTIVE / OFFICIAL BASELINE"
            r["id"] = "Baseline_v2.4"
            r["note"] = ("当前唯一官方基线（trial_1117002@5 权重 = soup(640001@5,1100296@1) w=0.56）；"
                         "2026-11-18 凌晨守望曾把同一权重重复升轨 82 次（v2.5~v2.86）"
                         "⇒ 已折叠为别名：" + ",".join(aliases.get(t, [])[:8]) +
                         ("…" if len(aliases.get(t, [])) > 8 else ""))
        elif r.get("status", "").startswith("ACTIVE"):
            r["status"] = "SUPERSEDED / DEPRECATED BASELINE"
        new_reg.append(r)
    led["baseline_registry"] = new_reg + others
    led["registry_retired_ids"] = sorted(retired)
    if base:
        base["name"] = "Baseline_v2.4"
        led["baseline_v2"] = base
    NS.save_ledger(led)
    print(f"[fix] registry {len(reg)} ⇒ {len(new_reg) + len(others)} 条；"
          f"baseline name ⇒ {base.get('name')}（trial {cur_trial}）；"
          f"别名折叠 {sum(len(v) for v in aliases.values())} 条")
    for r in (led["baseline_registry"])[:8]:
        print("   ", r.get("id"), r.get("trial"), r.get("status"))


if __name__ == "__main__":
    main()
