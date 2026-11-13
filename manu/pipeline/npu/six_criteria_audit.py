"""六项判据全盘复算审计（只读）。

🟢 2026-11-12 15:30 新增（§152.1）：回答「有没有更好的模型」的唯一权威口径。

为什么需要单独一个脚本（而不是信台账）：
  ① 台账 `official` 字段可能是 `{'stale': 'epoch_changed 5->4'}` ⇒ T1/T2/T3 缺失
     ⇒ `core_checks()` 对缺项是**跳过** ⇒ 六项漏斗里该臂「既不算过也不算不过」。
  ② 台账只对 **picked epoch** 存官方读数；但磁盘上 `gate/NANO_<trial>_ep<epoch>_official/`
     里可能躺着**别的 epoch** 的门禁报告 ⇒ 单看台账会漏掉候选。
  ⇒ 本脚本以**磁盘 gate_report.json 为权威**，把台账里所有 (臂, epoch) 的
     NanoBench 读数（nano / nano4 / f1）与官方三轨读数（t1 / t2 虚警 / t3）配对，
     逐个算六项，给出**全盘六项全过清单**与**最接近候选榜**。

口径（与 Baseline_v2.0 = trial_9863 ep5 完全一致，`>=` 判过）：
    nano  ≥ 0.708648 ｜ nano4 ≥ 0.647212 ｜ F1 ≥ 0.889712
    T1    ≥ 0.888086 ｜ T2 虚警 ≤ 5       ｜ T3 ≥ 0.948489
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

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
CRIT = ["nano", "nano4", "f1", "t1", "t2fp", "t3"]


def parse_report(path: str) -> dict | None:
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
        "verdict": rep.get("verdict"), "veto": rep.get("veto_failures", []),
    }


def collect(ledger_path: str = LEDGER, gate_dir: str = GATE_DIR) -> list[dict]:
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
            "t1": g["t1"], "t2_fp": g["t2_fp"], "t3": g["t3"], "veto": g["veto"],
            "sw": p.get("sw"), "swl": p.get("swl"), "lrf": p.get("lr_floor"),
            "mix": p.get("mix"), "thr": p.get("thr"), "ms": p.get("maxside"),
            "init": (p.get("init") or "").split("/")[-1],
            "picked_ep": (st.get("picked") or {}).get("epoch"),
        })
    return rows


def judge(r: dict) -> tuple[int, list[str]]:
    flags = [
        (r["nano"] is not None and r["nano"] >= BASE["nano"]),
        (r["nano4"] is not None and r["nano4"] >= BASE["nano4"]),
        (r["f1"] is not None and r["f1"] >= BASE["f1"]),
        (r["t1"] >= BASE["t1"]),
        (r["t2_fp"] <= BASE["t2_fp"]),
        (r["t3"] >= BASE["t3"]),
    ]
    miss = [n for n, ok in zip(CRIT, flags) if not ok]
    return 6 - len(miss), miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--gate-dir", default=GATE_DIR)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    rows = collect(a.ledger, a.gate_dir)
    if not rows:
        print("（没有可配对的 (臂, epoch) 读数）")
        return
    for r in rows:
        r["npass"], r["miss"] = judge(r)
    rows.sort(key=lambda r: (-r["npass"], -(r["nano"] or 0.0)))

    full = [r for r in rows if r["npass"] == 6]
    print(f"===== 六项判据全盘复算审计（{len(rows)} 个 (臂,epoch) 有完整读数）=====")
    print(f"基线 = Baseline_v2.0 / trial_9863 ep5；判过口径 `>=`")
    print(f"六项全过：**{len(full)}** 个")
    for r in full:
        tag = "  ← 基线本体" if r["trial"] == 9863 else "  🔴 **新候选（可升轨）**"
        print(f"  ✔ trial={r['trial']} ep={r['ep']} seed={r['seed']} "
              f"nano={r['nano']:.6f} nano4={r['nano4']:.6f} f1={r['f1']:.6f} "
              f"t1={r['t1']:.6f} t2fp={r['t2_fp']} t3={r['t3']:.6f}{tag}")
    print()
    print(f"--- 最接近候选榜 TOP {a.top}（按过项数，同项按 nano 排）---")
    for r in rows[: a.top]:
        gaps = []
        if r["nano"] < BASE["nano"]:
            gaps.append(f"nano{r['nano'] - BASE['nano']:+.6f}")
        if r["nano4"] < BASE["nano4"]:
            gaps.append(f"nano4{r['nano4'] - BASE['nano4']:+.6f}")
        if r["f1"] < BASE["f1"]:
            gaps.append(f"f1{r['f1'] - BASE['f1']:+.6f}")
        if r["t1"] < BASE["t1"]:
            gaps.append(f"t1{r['t1'] - BASE['t1']:+.6f}")
        if r["t2_fp"] > BASE["t2_fp"]:
            gaps.append(f"t2fp+{r['t2_fp'] - BASE['t2_fp']}")
        if r["t3"] < BASE["t3"]:
            gaps.append(f"t3{r['t3'] - BASE['t3']:+.6f}")
        star = "★" if r["ep"] != r["picked_ep"] else " "
        print(f" {r['npass']}/6{star} t={r['trial']:<6} ep={r['ep']:<3} "
              f"sw={r['sw']} swl={r['swl']} lrf={r['lrf']} mix={r['mix']} "
              f"init={r['init']} | " + (" ".join(gaps) if gaps else "全过"))
    print("（★ = 该 epoch 不是台账 picked epoch ⇒ 只看台账会漏掉它）")


if __name__ == "__main__":
    main()
