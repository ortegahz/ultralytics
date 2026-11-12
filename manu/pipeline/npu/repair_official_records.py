"""台账 `official` 字段修复器 —— 以**磁盘上的权威 gate_report.json** 为准回填。

🔴 为什么需要（2026-11-12 12:50 实测，不是推演）
--------------------------------------------------------------------------
台账 `nano_search_ledger.json` 里 9 个臂的 `official` 字段是
`{'stale': 'epoch_changed X->Y'}` ⇒ `t1` / `t2_fp` / `t3` **全为空**。

后果链条（三处都会被它坑到）：
1. `core_checks()` 对缺项是**跳过** ⇒ 这些臂在六项漏斗里**既不算过也不算不过**
   ⇒ 台账看不出「12053 只差 T1 一项」。
2. `near_miss_cells()` 里有 `if official.t1 is None: continue`
   ⇒ **最接近基线的臂被排除在近失复现队列之外**。
   实测：12053（冠军配方 + lr_floor 0.1）的 ep4 是**全台账唯一的「NanoBench 三项全过 + T2/T3 过、
   只差 T1 0.000984」**（ep4: nano 0.709594 / nano4 0.656664 / f1 0.892338 / T1 0.887102 /
   T2 fp 4 / T3 0.949250）⇒ **它的配方格本来是今晚最该复现的近失格，却因为台账陈旧而被漏掉**。
3. `beat` / `promotion_candidates()` 用同一份数据 ⇒ 升轨清单漏项。

修法：**磁盘上的 gate_report.json 才是权威**（它是门禁脚本直接写出的产物）。
本模块扫描 `{ROOT}/gate/NANO_<trial>_ep<epoch>_official/gate_report.json`，
对**当前 picked epoch** 有报告、但台账里缺 `t1`/`t3`（或标了 stale）的臂，
把该 epoch 的官方读数**回填**进台账。

⚠️ **只修 picked epoch**：别的 epoch 的读数与 picked epoch 不是同一个 checkpoint，
混填会造成「用 A epoch 的官方分给 B epoch」⇒ 反而更危险（这正是 stale 的由来）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
GATE_DIR = f"{ROOT}/gate"
LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
NAME_RE = re.compile(r"^NANO_(\d+)_ep(\d+)_official$")


def parse_report(path: str) -> dict | None:
    """解析 gate_report.json ⇒ 与 `nano_search.run_official_gate` 同一口径的 dict。"""
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
        "out_dir": os.path.dirname(path),
        "verdict": rep.get("verdict"),
        "t1": float(t1.get("f1", 0)), "t1_fp": int(t1.get("fp", 0)),
        "t2": float(t2.get("f1", 0)), "t2_fp": int(t2.get("fp", 0)),
        "t3": float(t3.get("f1", 0)), "t3_fp": int(t3.get("fp", 0)),
        "veto_failures": rep.get("veto_failures", []),
    }


def scan_gate_reports(gate_dir: str = GATE_DIR) -> dict:
    """⇒ {(trial, epoch): 官方读数 dict}"""
    out: dict = {}
    for d in sorted(glob.glob(f"{gate_dir}/NANO_*_official")):
        m = NAME_RE.match(os.path.basename(d))
        if not m:
            continue
        rp = os.path.join(d, "gate_report.json")
        if not os.path.isfile(rp):
            continue
        g = parse_report(rp)
        if g:
            out[(int(m.group(1)), int(m.group(2)))] = g
    return out


def repair_ledger(led: dict, gate_dir: str = GATE_DIR) -> list[dict]:
    """就地修 `led`（不落盘）。返回修复明细列表。"""
    reports = scan_gate_reports(gate_dir)
    fixed = []
    for t, st in (led.get("arms") or {}).items():
        pk = st.get("picked") or {}
        ep = pk.get("epoch")
        if ep is None:
            continue
        of = st.get("official") or {}
        have = isinstance(of, dict) and of.get("t1") is not None and of.get("t3") is not None
        if have:
            continue
        g = reports.get((int(t), int(ep)))
        if not g:
            continue
        st["official"] = g
        st["official_repaired"] = True
        fixed.append({"trial": int(t), "epoch": int(ep), "old": of, "new": g})
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser(description="台账 official 字段修复（以磁盘 gate_report 为准）")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--gate-dir", default=GATE_DIR)
    ap.add_argument("--apply", action="store_true", help="真正写回台账（默认只 dry-run）")
    args = ap.parse_args()

    led = json.load(open(args.ledger))
    fixed = repair_ledger(led, args.gate_dir)
    if not fixed:
        print("无需修复（所有 picked epoch 的官方读数都已齐全）")
        return
    for f in fixed:
        g = f["new"]
        print(f"[{f['trial']}] ep{f['epoch']}: {f['old']} ⇒ "
              f"t1={g['t1']:.6f} t2_fp={g['t2_fp']} t3={g['t3']:.6f} verdict={g['verdict']}")
    if args.apply:
        tmp = args.ledger + ".repair.tmp"
        with open(tmp, "w") as fh:
            json.dump(led, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, args.ledger)
        print(f"已写回 {args.ledger}（{len(fixed)} 个臂）")
    else:
        print(f"dry-run：{len(fixed)} 个臂可修复（加 --apply 才写回）")


if __name__ == "__main__":
    sys.exit(main())
