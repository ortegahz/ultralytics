#!/usr/bin/env python3
"""夜间 8 个种子臂的**官方三轨门禁**补测（T1 + T2 + T3 + verdict）。

背景（2026-11-10 08:55）：
  夜间阶段2 只跑了 T1 快筛（`gate_sweep.sh t1only`），**8 个种子臂的 T2 虚警从来没被测过**。
  而 SILU9222（该族 T1 最高的臂）官方门禁 T2 FP=69 ⇒ 超出限值（基线 26，限 +10 ⇒ ≤36）⇒ FAIL。
  ⇒ 单看 T1 选型会挑出不可交付的臂；**必须用官方三轨门禁重新选型**。

本脚本：对 trial_9301~9308 依次（4 卡并行）跑 `gate_sweep.sh full`，
      解析 T1/T2/T3 的 f1 与 fp，最后写 overnight_gate_sweep.json，并给出
      「T1 ≥ trial_0031 且 T2 FP ≤ 36 且 T3 不劣于 trial_0031」的**可交付候选**。

只做评测，不训练、不碰板卡。
"""

import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

RUNS = "/mnt/data/siping/runs/npu20ms"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
REPO = "/tmp/pycharm_project_10ae9e2e"
OUT = f"{RUNS}/overnight_gate_sweep.json"
LOG = f"{RUNS}/overnight_gate_sweep.log"

T31 = {"t1": 0.885997, "t2": 0.997531, "t2_fp": 10, "t3": 0.948330, "t3_fp": 110}
T2_FP_LIMIT = 36  # 基线 26 + 10

FULL_RE = re.compile(
    r"trial_(\S+) FULL T1=(\S+) tp=(\S+) fp=(\S+) \| T2=(\S+) fp=(\S+) \| T3=(\S+) fp=(\S+) \| verdict=(\S+)"
)


def tail(path, n=400):
    try:
        return open(path, errors="ignore").read()[-n:]
    except OSError:
        return ""


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=5400):
    return subprocess.run(
        cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=timeout
    )


def one(args):
    trial, gpu = args
    w = f"{ROOT}/trials/trial_{trial}/weights/epoch8.pt"
    if not os.path.isfile(w):
        log(f"[{trial}] 缺权重 {w} ⇒ 跳过")
        return None
    outd = f"{ROOT}/gate/{trial}"
    log(f"[{trial}] 官方门禁开跑 gpu={gpu} out={outd}")
    r = sh(
        f"cd {REPO} && export PYTHONPATH=. && "
        f"/home/huangzhe/anaconda3/envs/uav/bin/python -u manu/training/auto_epoch_gate_check.py "
        f"--weights {w} --out-dir {outd} --device {gpu} --infer-gpu {gpu} --gpus 1 "
        f"> {outd}_gate.log 2>&1",
        timeout=7200,
    )
    rp = f"{outd}/gate_report.json"
    if not os.path.isfile(rp):
        log(f"[{trial}] 无报告 rc={r.returncode}: {tail(outd + '_gate.log')}")
        return {"trial": trial, "error": tail(outd + "_gate.log"), "rc": r.returncode}
    rep = json.load(open(rp))
    tr = rep.get("tracks", {}) or {}
    t1 = (tr.get("track1_uav") or {}).get("finetune_peak") or {}
    t2 = tr.get("track2_golden") or {}
    t3 = tr.get("track3_blind") or {}
    d = {
        "trial": trial,
        "t1": float(t1.get("f1", 0)),
        "t1_tp": int(t1.get("tp", 0)),
        "t1_fp": int(t1.get("fp", 0)),
        "t2": float(t2.get("f1", 0)),
        "t2_fp": int(t2.get("fp", 0)),
        "t3": float(t3.get("f1", 0)),
        "t3_fp": int(t3.get("fp", 0)),
        "verdict": rep.get("verdict"),
        "out": outd,
    }
    d["t1_ge_t31"] = d["t1"] >= T31["t1"]
    d["t2_fp_ok"] = d["t2_fp"] <= T2_FP_LIMIT
    d["t3_ge_t31"] = d["t3"] >= T31["t3"]
    d["deliverable"] = d["t1_ge_t31"] and d["t2_fp_ok"] and d["t3_ge_t31"]
    log(f"[{trial}] => {d}")
    return d


def main():
    trials = [9301, 9302, 9303, 9304, 9305, 9306, 9307, 9308]
    gpus = [0, 1, 2, 3]
    res = []
    log(f"=== 8 臂官方三轨门禁开始（判据 T1≥{T31['t1']} 且 T2 FP≤{T2_FP_LIMIT} 且 T3≥{T31['t3']}）===")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(one, [(t, gpus[i % 4]) for i, t in enumerate(trials)]):
            if r:
                res.append(r)
    res.sort(key=lambda x: x.get("t1", 0), reverse=True)
    good = [r for r in res if r.get("deliverable")]
    payload = {
        "criterion": {
            "t1_ge": T31["t1"],
            "t2_fp_le": T2_FP_LIMIT,
            "t3_ge": T31["t3"],
            "reference": "trial_0031 官方门禁",
        },
        "results": res,
        "deliverable_candidates": good,
        "best_by_t1": res[0] if res else None,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log(f"=== 完成：可交付候选 {len(good)} 个 {[g['trial'] for g in good]} ===")
    for r in res:
        log(
            f"  trial_{r['trial']}: T1={r['t1']:.6f} T2={r['t2']:.6f}(fp={r['t2_fp']}) "
            f"T3={r['t3']:.6f}(fp={r['t3_fp']}) verdict={r['verdict']} "
            f"{'★可交付' if r.get('deliverable') else ''}"
        )
    log("=== GATE SWEEP DONE ===")


if __name__ == "__main__":
    main()
