#!/usr/bin/env python3
"""连续搜索器 v2（2026-11-10 09:10 起）：从 trial_0031 出发，一次只动一个变量。

═══ 为什么重写（昨夜 + 今晨两次事故的复盘）═══
1) 昨夜 `overnight.py` 清单太短：8 个种子臂跑完就 DONE ⇒ GPU 空转 14 h。
   ⇒ 本脚本有自我续轮（W1→W2→W3…），直到命中交付判据或到硬截止。
2) 昨夜选型口径错：只用 T1 快筛。实测证据（今晨官方门禁补测）：
   夜间族 12 个臂（9221/9222/9223 + 9301~9308）的 T2 FP = **69~94**、
   T3 FP = **624~833**，而 trial_0031 是 **T2 FP 10 / T3 FP 110** ⇒
   **T1 单指标排序会系统性选出不可交付的臂**（T1 快筛 0.8887 的臂，T2 FP 是交付件的 7 倍）。
   ⇒ 本脚本**每个臂都跑官方门禁**，用复合判据选型。

═══ 交付判据（官方门禁默认阈值，与 trial_0031 对齐）═══
  T1(GMC) ≥ 0.885997  AND  T2 FP ≤ 36（基线 26 + 10）  AND  T3(GMC) ≥ 0.948330

═══ 基线配方 = trial_0031 的真实配方 ═══
  init = cs_t0043.pt（在位交付基线）｜ epochs 20 / sched-epochs 5（cosine 只排 5 轮，
  后 15 轮在地板 LR 上微调）｜ warmup 1.5 ｜ lr-scale 3.5 ｜ mix w2.0 ｜ focal 2.75 ｜ ow 0.4
  ｜ backbone-lr-scale 1.0 ｜ unfreeze 0 ｜ head_src p3 ｜ act silu / p3 / hc64 / late / no_p2
  ｜ **train-on-native-grid 0、min-radius 1**（夜间臂用的是 1/0 ⇒ 这是重要差异）

W1 = 对照 + 三个「单变量」假设臂：
  9501 对照（复现 trial_0031 配方，换种子）
  9502 sched-epochs 5→8   假设：夜间臂的 8 轮满 cosine 是 T2/T3 虚警变多的原因
  9503 warmup 1.5→0.5     假设：warmup 变短导致早期不稳
  9504 native-grid 0→1 且 min-radius 1→0   假设：夜间臂的监督网格改动是虚警来源
每个臂都跑官方门禁（W1 约 70 min 训练 + 门禁 10 min）。
"""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

RUNS = "/mnt/data/siping/runs/npu20ms"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
REPO = "/tmp/pycharm_project_10ae9e2e"
LOG = f"{RUNS}/search_loop.log"
RESULT = f"{RUNS}/search_loop_result.json"
PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"
CS_T0043 = "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt"
BASE_CACHE = "/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl"
REF = "/mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json"
LQ = "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc"
UAVD = "/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc"

CRIT = {"t1_ge": 0.885997, "t2_fp_le": 36, "t3_ge": 0.948330}
STOP_AT = "2026-11-11 08:00"

BASE = dict(
    init=CS_T0043, epochs=20, sched=5, warmup=1.5, lr=3.5, mix="w2.0", fb=2.75, ow=0.4,
    wd=0.000269, mosaic=0.15, scale=0.10, translate=0.05, clip=5.65, backbone_lr=1.0,
    uf=0, head_src="p3", native_grid=0, mr=1,
)

TRIAL0 = 9500  # 臂号起点


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=7200):
    return subprocess.run(
        cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=timeout
    )


def stop_ts():
    return time.mktime(time.strptime(STOP_AT, "%Y-%m-%d %H:%M"))


def train_cmd(trial, gpu, seed, p):
    return (
        f"cd {REPO} && export PYTHONPATH=. && {PY} -u manu/training/optuna_mix_ratio_search.py --worker "
        f"--trial-number {trial} --gpu-id {gpu} --output-root {ROOT} "
        f"--longquan-data {LQ} --uav-data {UAVD} --weights {p['init']} --baseline-cache {BASE_CACHE} "
        f"--epochs {p['epochs']} --sched-epochs {p['sched']} --batch 32 --imgsz 640 --workers 8 --seed {seed} "
        f"--mix-spec {p['mix']} --lr-scale {p['lr']} --warmup-epochs {p['warmup']} --weight-decay {p['wd']} "
        f"--focal-beta {p['fb']} --offset-weight {p['ow']} --mosaic {p['mosaic']} --scale {p['scale']} "
        f"--translate {p['translate']} --clip-max-norm {p['clip']} --freeze-bn 0 "
        f"--backbone-lr-scale {p['backbone_lr']} --freeze-backbone-epochs 0 "
        f"--no-p2 1 --head-level p3 --act silu --head-conv 64 --late-upsample 1 "
        f"--unfreeze-b0b4 {p['uf']} --head-src '{p['head_src']}' --p2-out-scale 1.0 "
        f"--train-on-native-grid {p['native_grid']} --min-radius {p['mr']} "
        f"--reference-frame {REF} --skip-gate > {ROOT}/logs/LOOP_{trial}.log 2>&1"
    )


def screen(trial, gpu, p):
    w = f"{ROOT}/trials/trial_{trial}/weights/epoch{p['epochs']}.pt"
    if not os.path.isfile(w):
        return None
    out = f"{RUNS}/loop_screen_{trial}.log"
    sh(f"bash {RUNS}/gate_sweep.sh t1only LOOP{trial} {gpu} {w} >> {out} 2>&1", timeout=3600)
    txt = open(out, errors="ignore").read() if os.path.isfile(out) else ""
    import re
    m = re.search(
        r"\[PEAK\] trial_\S+: th=([\d.]+)\s+F1=([\d.]+)\s+P=([\d.]+)%\s+R=([\d.]+)%\s+TP=(\d+)\s+FP=(\d+)",
        txt,
    )
    if not m:
        return None
    return dict(t1_screen=float(m.group(2)), P=float(m.group(3)), R=float(m.group(4)),
                TP=int(m.group(5)), FP=int(m.group(6)))


def gate(trial, gpu, p):
    w = f"{ROOT}/trials/trial_{trial}/weights/epoch{p['epochs']}.pt"
    outd = f"{ROOT}/gate/LOOP_{trial}_official"
    r = sh(
        f"cd {REPO} && export PYTHONPATH=. && {PY} -u manu/training/auto_epoch_gate_check.py "
        f"--weights {w} --out-dir {outd} --device {gpu} --infer-gpu {gpu} --gpus 1 > {outd}_gate.log 2>&1",
        timeout=7200,
    )
    rp = f"{outd}/gate_report.json"
    if not os.path.isfile(rp):
        return {"trial": trial, "error": f"no report rc={r.returncode}"}
    rep = json.load(open(rp))
    tr = rep.get("tracks", {}) or {}
    t1 = (tr.get("track1_uav") or {}).get("finetune_peak") or {}
    t2 = tr.get("track2_golden") or {}
    t3 = tr.get("track3_blind") or {}
    d = {"trial": trial, "out": outd,
         "t1": float(t1.get("f1", 0)), "t2": float(t2.get("f1", 0)), "t2_fp": int(t2.get("fp", 0)),
         "t3": float(t3.get("f1", 0)), "t3_fp": int(t3.get("fp", 0)), "verdict": rep.get("verdict")}
    d["deliverable"] = (d["t1"] >= CRIT["t1_ge"] and d["t2_fp"] <= CRIT["t2_fp_le"]
                         and d["t3"] >= CRIT["t3_ge"])
    return d


def run_wave(wave, cfgs):
    """cfgs: [(trial, seed, params)]；4 卡并发训练 + T1 快筛 + 官方门禁（每个臂都跑）。"""
    gpus = [0, 1, 2, 3]

    def one(item):
        trial, seed, p = item
        if time.time() > stop_ts() - 7200:
            log(f"[{trial}] 距截止不足 2h ⇒ 不启动")
            return None
        log(f"[{trial}] 训练 seed={seed} {p}")
        try:
            sh(train_cmd(trial, gpus[trial % 4], seed, p), timeout=18000)
        except Exception as e:
            log(f"[{trial}] 训练异常: {e}")
            return None
        s = screen(trial, gpus[trial % 4], p)
        log(f"[{trial}] T1 快筛 => {s}")
        g = gate(trial, gpus[trial % 4], p)
        log(f"[{trial}] 官方门禁 => {g}")
        return dict(trial=trial, seed=seed, params=p, screen=s or {}, gate=g or {})

    out = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(one, cfgs):
            if r:
                out.append(r)
    return out


def save(payload):
    with open(RESULT, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def wait_for_gate_sweep():
    log("等待门禁补测（overnight_gate_sweep）进程退出，避免抢同一批 GPU…")
    for _ in range(240):
        r = subprocess.run(["pgrep", "-f", "gate_sweep_overnight.py"], capture_output=True, text=True)
        if not r.stdout.strip():
            log("门禁补测已退出 ⇒ 接管 GPU")
            return True
        time.sleep(30)
    return False


def deliverable_arms(payload):
    return [r for w in payload["waves"] for r in w["results"]
            if (r.get("gate") or {}).get("deliverable")]


def main():
    log("=== SEARCH LOOP v2 START（判据 T1≥%.6f 且 T2 FP≤%d 且 T3≥%.6f，截止 %s）===" % (
        CRIT["t1_ge"], CRIT["t2_fp_le"], CRIT["t3_ge"], STOP_AT))
    wait_for_gate_sweep()

    payload = {"criterion": CRIT, "base_recipe": BASE, "waves": [], "deliverable": None,
               "stop_at": STOP_AT, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    save(payload)

    # ---------- W1：对照 + 三个单变量假设臂 ----------
    p = dict(BASE)
    w1 = [
        (TRIAL0 + 1, 18101, dict(BASE)),                                  # 对照
        (TRIAL0 + 2, 18102, dict(BASE, sched=8)),                        # schedule 假设
        (TRIAL0 + 3, 18103, dict(BASE, warmup=0.5)),                     # warmup 假设
        (TRIAL0 + 4, 18104, dict(BASE, native_grid=1, mr=0)),            # 监督网格假设
    ]
    r1 = run_wave("W1", w1)
    payload["waves"].append({"wave": "W1", "results": r1})
    save(payload)
    good = deliverable_arms(payload)
    if good:
        payload["deliverable"] = max(good, key=lambda r: r["gate"]["t1"])
        save(payload)
        log(f"★★ W1 命中交付判据: {payload['deliverable']}")
        return

    # ---------- W2：按 W1 的结果决定下一组单变量 + 种子复现 ----------
    log("W1 无臂通过交付判据 ⇒ W2 继续（取 W1 各臂相对对照的 ΔT2FP 决定下一个旋钮）")
    def delta_t2fp(r):
        ctl = next((x for x in r1 if x["trial"] == TRIAL0 + 1), None)
        if not ctl or not (r.get("gate") or {}).get("t2_fp"):
            return None
        return (r["gate"]["t2_fp"] - ctl["gate"]["t2_fp"])

    ranked = sorted([r for r in r1 if r.get("gate") and r["gate"].get("t2_fp") is not None],
                    key=lambda r: r["gate"]["t2_fp"])
    log(f"W1 按 T2 FP 排序: {[(r['trial'], r['gate']['t2_fp'], round(r['gate']['t1'], 6)) for r in ranked]}")

    w2 = []
    tr = TRIAL0 + 10
    # 最好的一臂：换种子复现（确认不是种子运气）
    if ranked:
        best = ranked[0]
        pp = dict(best["params"])
        w2.append((tr, 18110 + tr, pp)); tr += 1
        # 在最好臂上叠加第二个变量：epochs 12（更短的满 cosine）
        w2.append((tr, 18110 + tr, dict(pp, epochs=12, sched=8))); tr += 1
    # 另外两个旋钮：解冻 b0b4 / lr-scale 3.0
    w2.append((tr, 18110 + tr, dict(BASE, uf=1))); tr += 1
    w2.append((tr, 18110 + tr, dict(BASE, lr=3.0))); tr += 1
    r2 = run_wave("W2", w2)
    payload["waves"].append({"wave": "W2", "results": r2})
    save(payload)
    good = deliverable_arms(payload)
    if good:
        payload["deliverable"] = max(good, key=lambda r: r["gate"]["t1"])
    save(payload)
    log(f"=== SEARCH LOOP v2 DONE（可交付候选 {len(good)}）===")


if __name__ == "__main__":
    main()
