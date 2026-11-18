
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把**所有**合成汤的 sidecar 补登记进台账（2026-11-13 18:50 新增）。

🟢 为什么需要它：
  `family_soup.py` 的 `register_synth_arms` 期望 entry 里**扁平**的 nano/nano4/f1 字段，
  但调用方只给了嵌套的 `nanobench` ⇒ 登记出来的汤臂 **指标全为 None**
  ⇒ `six_criteria_audit.collect()` 看不到它们 ⇒ **汤臂永远进不了升轨通道**。
  本脚本直接扫 69800~69999 的 sidecar，按正确口径补登记（并覆盖写坏掉的那批）。

用法：
    python manu/pipeline/npu/soup_register_all.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
if os.path.isdir(REPO):
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))
os.chdir(REPO)

LEDGER = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
SPACES = (69800, 69900, 98900)
SID_CARS = ("family_soup_synth.json", "basin_soup_synth.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def main() -> None:
    from manu.pipeline.npu import nano_search as ns  # noqa: PLC0415
    led = ns.load_ledger()
    arms = led.get("arms") or {}
    n_ok = n_skip = 0
    # 🟢🟢🟢 2026-11-16 12:15：**扫全盘 sidecar**（不再用固定 5×100 的 id 窗口）
    #   背景：汤臂 id 段已经扩到 30 万/40 万（5 条并行通道 + 每轮换种子/换段）
    #   ⇒ 固定窗口**登记不到新汤** ⇒ 台账里没有这些臂 ⇒
    #     `six_criteria_audit.collect()` 配不上 nanobench 读数 ⇒
    #     **候选永远进不了升轨通道**（等于白跑门禁）。
    sids = []
    for _d in glob.glob(os.path.join(ROOT, "trials", "trial_*", "weights")):
        _m = re.search(r"trial_(\d+)$", os.path.dirname(_d))
        if _m:
            sids.append(int(_m.group(1)))
    sids = sorted(set(sids))
    log(f"[扫描] trial 目录 {len(sids)} 个（含 sidecar 的会逐个检查）")
    for sid in sids:
        if True:
            d = os.path.join(ROOT, "trials", f"trial_{sid}", "weights")
            sc = None
            for name in SID_CARS:
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    try:
                        with open(p, encoding="utf-8") as fh:
                            sc = json.load(fh)
                    except Exception:  # noqa: BLE001
                        sc = None
                    if sc:
                        break
            if not sc:
                continue
            nb = sc.get("nanobench") or {}
            gate = sc.get("gate") or {}
            mem = sc.get("members") or []
            if not mem or nb.get("nano") is None:
                n_skip += 1
                continue
            if gate.get("error") or gate.get("t1") is None or gate.get("t3") is None:
                n_skip += 1
                continue
            ck = os.path.join(d, "epoch5.pt")
            if not os.path.isfile(ck):
                n_skip += 1
                continue
            src = (arms.get(str(mem[0][0])) or {}).get("params") or {}
            p = dict(src)
            p["init"] = ck
            p["synth"] = 1
            full = {"nano": nb.get("nano"), "nano4": nb.get("nano4"), "f1": nb.get("f1"),
                    "micro": nb.get("micro"), "small": nb.get("small"), "regular": nb.get("regular"),
                    "f1_4": nb.get("f1_4")}
            full.update({"epoch": 5, "epoch_key": "5", "ckpt": "epoch5.pt", "cache": sc.get("cache")})
            fam = str(sc.get("family") or sc.get("tag") or "")
            # 🟢 2026-11-13 18:00 家族键归一：**按「池」而不是按成员数 k**
            #   "bs<t>:eq<k>" → "bs<t>:eq"（同一盆地 + 同一成员池 + 等权 = 同一个方法；
            #   k 是该方法的超参，不该把不同 k 拆成不同家族，否则 rep 家族永远凑不满 n≥4）
            m_eq = re.match(r"^bs(\d+):eq\d+$", fam)
            if m_eq:
                fam = "bs%s:eq" % m_eq.group(1)
            elif not fam.startswith("bs"):
                # 🟢 2026-11-16 09:10：家族键的「盆地 id」取 **sidecar 里的 basin（=参考臂）**，
                #   兜底才用首位成员的 trial
                #   ⇒ 成员池增长（汤里再套汤）时**家族键不会漂移**，n≥4 计数才连续。
                fam = "bs%s:eq" % (sc.get("basin") or mem[0][0])
            else:
                _b = sc.get("basin")
                if _b:
                    fam = re.sub(r"^bs\d+:", "bs%s:" % int(_b), fam)
            _keep = {}
            _old = arms.get(str(sid)) or {}
            for _k in ("board_ms", "board_ms_epochs", "board_pass_t4", "board_limit_ms",
                       "board_measured_at", "is_baseline_v2", "status", "baseline_name",
                       "beat_to_pass_ref", "six_ok_manual"):
                if _k in _old:
                    _keep[_k] = _old[_k]
            arms[str(sid)] = dict({
                "trial": sid, "synth": True, "source_trial": mem[0][0],
                "soup_members": [list(x) for x in mem], "soup_weights": sc.get("weights") or [],
                "soup_family": fam,
                "params": p, "seed": None,
                "nanobench": {"5": full}, "picked": dict(full), "picked_epoch": 5,
                "official": {k: gate.get(k) for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp", "verdict")},
                "six_ok": bool(gate.get("six_ok")),
                "registered_by": "soup_register_all",
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, **_keep)
            log(f"[登记] synth={sid} family={fam} 成员={mem} nano={full['nano']:.6f} "
                f"nano4={full['nano4']:.6f} F1={full['f1']:.6f} T1={gate.get('t1')} "
                f"T2fp={gate.get('t2_fp')} T3={gate.get('t3')} six_ok={gate.get('six_ok')}")
            n_ok += 1
    if n_ok:
        led["arms"] = arms
        ns.save_ledger(led)
    log(f"[完成] 登记 {n_ok} 个｜跳过 {n_skip} 个")


if __name__ == "__main__":
    main()