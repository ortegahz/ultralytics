#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""台账回填：把**磁盘 gate_report.json** 里的官方三轨读数补进台账的 `official` 段，
并给老臂补 `nanobench[ep]`（老格式臂只有 `picked`）。

🔴 **为什么必须做**（2026-11-18 16:55）：
  · `six_criteria_audit.collect()` 的入参是 **`arms[tid]["nanobench"][str(ep)]`**；
    早期台账只写了 `picked`、没写 `nanobench` ⇒ **这些臂在审计/Track-B 报告里根本不存在**
    （例：trial_1104642 = 全盘 F1 最高 .8908641 / T1 最高 .8910094 / 板卡 19.666 ✔ 的臂，
      因为缺 `nanobench` 而从未进入任何榜单 ⇒ 它的 T3 缺口（−1.1e-4）也没人去补）。
  · 同理 `official` 缺失 ⇒ 官方三轨读数丢失 ⇒ 无法判 Track-B。
⇒ 本脚本只**补数据**，不改任何已有数值；幂等，可反复跑。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e")
sys.path.insert(0, "/tmp/pycharm_project_10ae9e2e/manu/pipeline/npu")
import nano_search as ns  # noqa: E402
from manu.pipeline.npu import six_criteria_audit as A  # noqa: E402

GATE_DIR = "/mnt/data/siping/runs/npu20ms_offcal/gate"
NAME_RE = re.compile(r"^NANO_(\d+)_ep([0-9.]+)_official$")


def parse_report(path: str):
    try:
        rep = json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    tr = rep.get("tracks") or {}
    t1 = (tr.get("track1_uav") or {}).get("finetune_peak") or {}
    t2 = tr.get("track2_golden") or {}
    t3 = tr.get("track3_blind") or {}
    if not t1.get("f1") or not t3.get("f1"):
        return None
    return {
        "t1": float(t1.get("f1", 0)), "t1_fp": int(t1.get("fp", 0)),
        "t2": float(t2.get("f1", 0)), "t2_fp": int(t2.get("fp", 0)),
        "t3": float(t3.get("f1", 0)), "t3_fp": int(t3.get("fp", 0)),
        "verdict": rep.get("verdict"), "veto": rep.get("veto_failures", []),
    }


def main() -> None:
    led = ns.load_ledger()
    arms = led.get("arms") or {}
    # sid → gate 报告
    rep_map = {}
    for d in sorted(glob.glob(f"{GATE_DIR}/NANO_*_official")):
        m = NAME_RE.match(os.path.basename(d))
        if not m:
            continue
        rp = os.path.join(d, "gate_report.json")
        if not os.path.isfile(rp):
            continue
        g = parse_report(rp)
        if g:
            rep_map[(int(m.group(1)), m.group(2))] = g
    n_off = n_nb = 0
    touched = []
    for tid, st in arms.items():
        pk = st.get("picked") or {}
        ep = str(st.get("picked_epoch") or 5)
        of = st.get("official") or {}
        if not of.get("t1") or not of.get("t3"):
            g = rep_map.get((int(tid), ep))
            if g:
                st["official"] = dict(g)
                n_off += 1
                touched.append((tid, "official"))
        nb = st.get("nanobench")
        if (not nb or ep not in nb) and pk.get("nano") is not None:
            st["nanobench"] = {ep: dict(pk)}
            n_nb += 1
            touched.append((tid, "nanobench"))
    # ── 🔴🔴 2026-11-19 14:30 **重算 `official.six_ok`**（升轨后必须刷新）────────
    #   现场：`official.six_ok` 是**打门禁那一刻**按**当时**的基线算的；基线一升轨
    #   （Baseline_v2.10 → v3.0）这些标记就全部**过期** ⇒
    #     · `promote_candidate` 的种子关用 `st["six_ok"]` 计数 ⇒ 计数失真；
    #     · `board_t4_watch` / `night_soup_batch` 的「可升轨家族」加权用 `official.six_ok`
    #       ⇒ **已经不能升轨的臂**被当成升轨预备 ⇒ 板卡/门禁机时继续烧在死路上。
    #   ⇒ 每轮用 `six_criteria_audit.judge()`（口径 = `live_base()` = 台账当前基线，
    #     T2 轴 = `max(基线 t2_fp, 5)`）重算并回写。
    n_six = 0
    _B = A.live_base(led)
    for tid, st in arms.items():
        of = st.get("official") or {}
        pk = st.get("picked") or {}
        if not of or pk.get("nano") is None or of.get("t1") is None or of.get("t3") is None:
            continue
        row = {
            "nano": pk.get("nano"), "nano4": pk.get("nano4"), "f1": pk.get("f1"),
            "t1": of.get("t1"), "t2_fp": of.get("t2_fp"), "t3": of.get("t3"),
        }
        if any(v is None for v in row.values()):
            continue
        npass, _miss = A.judge(row, _B)
        ok = bool(npass == 6)
        if of.get("six_ok") != ok:
            of["six_ok"] = ok
            st["official"] = of
            arms[tid] = st
            n_six += 1
    ns.save_ledger(led)
    print("[backfill] 官方三轨补写 %d 个臂｜nanobench 补写 %d 个臂｜six_ok 重算 %d 个臂｜总臂数 %d"
          % (n_off, n_nb, n_six, len(arms)))
    if touched:
        print("[backfill] 样例：", touched[:12])


if __name__ == "__main__":
    main()
