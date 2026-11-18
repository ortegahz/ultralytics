#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抛光**级联**的锚点选择器（2026-11-13 新增）。

═══ 为什么需要它 ═══
``trial_999821``（= ``12679@3`` ＋ L2 锚定抛光 lr0.2/λ1e-3/fb1/ep1）打穿六项 + 板卡 T4，
⇒ **抛光轴是当前唯一有效轴**。但「第二批」用的是**固定锚点表**（8 个臂 × 12 个配置），
一旦这批跑完，后面就没有新组合可跑了。

而抛光轴的**天然特性**是：抛光产物本身也是 5/6 或 6/6 臂 ⇒ 它又是一个新锚点
⇒ 天然支持**深度 2+ 的级联抛光**（抛光 → 再抛光），且每一层都只花 1 epoch。

本脚本从台账里按「离六项全过最近」排序，滑窗取 K 个锚点输出给
``polish_nearmiss.py``（跨批次 warm_key 去重 ⇒ 只会跑**新组合**）。

排序口径与调度器 ``adaptive_runner._near_miss_score`` **完全一致**（同一套 BASE + 同一套归一化），
⇒ 级联锚点 = 调度器 warm 锚点，两条通道看到的「最近臂」不会打架。

用法：
    python manu/pipeline/npu/polish_cascade_anchors.py --top-k 8 --offset 0
    # ⇒ 输出 "12679:3,999821:1,..."（trial:epoch，逗号分隔）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = "/tmp/pycharm_project_10ae9e2e"
if os.path.isdir(REPO):
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

from manu.pipeline.npu import nano_search as ns  # noqa: E402
from manu.pipeline.npu.six_criteria_audit import BASE  # noqa: E402

ROOT = ns.ROOT


def score(st: dict) -> tuple[int, float]:
    """该臂过了几项 + 归一化亏损（0 = 全过）；缺项不计入亏损。"""
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    vals = {
        "nano": pk.get("nano"), "nano4": pk.get("nano4"), "f1": pk.get("f1"),
        "t1": of.get("t1"), "t2fp": of.get("t2_fp"), "t3": of.get("t3"),
    }
    checks = {
        "nano": vals["nano"] is not None and vals["nano"] >= BASE["nano"],
        "nano4": vals["nano4"] is not None and vals["nano4"] >= BASE["nano4"],
        "f1": vals["f1"] is not None and vals["f1"] >= BASE["f1"],
        "t1": vals["t1"] is not None and vals["t1"] >= BASE["t1"],
        "t2fp": vals["t2fp"] is not None and vals["t2fp"] <= BASE["t2_fp"],
        "t3": vals["t3"] is not None and vals["t3"] >= BASE["t3"],
    }
    npass = sum(1 for v in checks.values() if v)
    deficit = 0.0
    for k, ok in checks.items():
        if ok:
            continue
        v = vals.get(k)
        if v is None:
            continue                       # 缺项不算亏损
        if k == "t2fp":
            d = (float(v) - float(BASE["t2_fp"])) / 5.0
        else:
            bv = float(BASE[k]) or 1.0
            d = (bv - float(v)) / bv
        deficit += max(0.0, d)
    return npass, deficit


def main() -> None:
    ap = argparse.ArgumentParser(description="抛光级联锚点选择器")
    ap.add_argument("--top-k", type=int, default=8, help="本轮取几个锚点")
    ap.add_argument("--offset", type=int, default=0, help="滑窗偏移（每轮 +K ⇒ 换一批锚点）")
    ap.add_argument("--ledger", default=ns.LEDGER if hasattr(ns, "LEDGER") else
                    "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json")
    ap.add_argument("--min-npass", type=int, default=4,
                    help="至少过几项才配当锚（低于此值太远，抛光不划算）")
    ap.add_argument("--print-detail", action="store_true")
    args = ap.parse_args()

    with open(args.ledger, encoding="utf-8") as f:
        led = json.load(f)
    arms = led.get("arms") or {}
    rows = []
    for tid, st in arms.items():
        try:
            ti = int(tid)
        except (TypeError, ValueError):
            continue
        pk = st.get("picked") or {}
        of = st.get("official") or {}
        if pk.get("nano") is None or of.get("t1") is None or of.get("t3") is None:
            continue
        ep = st.get("picked_epoch") or pk.get("epoch")
        if not ep:
            continue
        ck = os.path.join(ROOT, "trials", f"trial_{ti}", "weights", f"epoch{int(ep)}.pt")
        if not os.path.isfile(ck):
            continue
        npass, deficit = score(st)
        if npass < args.min_npass:
            continue
        rows.append((npass, deficit, ti, int(ep), pk.get("nano")))
    # 排序：过项多优先 → 亏损小优先 → nano 高优先 → 臂号大优先（新臂优先）
    rows.sort(key=lambda r: (-r[0], r[1], -(r[4] or 0.0), -r[2]))
    if not rows:
        print("")
        return
    k = max(1, int(args.top_k))
    off = int(args.offset) % max(1, len(rows))
    win = []
    for i in range(k):
        win.append(rows[(off + i) % len(rows)])
    print(",".join(f"{t}:{e}" for (_n, _d, t, e, _nano) in win))
    if args.print_detail:
        for (n, d, t, e, nano) in win:
            print(f"  # {t}@{e}  npass={n}/6 deficit={d:.6f} nano={nano}", file=sys.stderr)


if __name__ == "__main__":
    main()
