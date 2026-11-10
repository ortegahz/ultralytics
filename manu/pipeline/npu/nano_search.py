#!/usr/bin/env python3
"""NanoBench 定向搜索器 v3（2026-11-10）：以 `trial_0031` 为基准，单变量搜索「小目标召回更高」的新模型。

═══ 目标 ═══
交付件 `trial_0031`（P3/hc64/silu/late/no_p2，int8 19.63 ms）在 NanoBench-T1（no-GMC，8px 主档）
上是 **F1 0.887122 / nano 0.6666 / micro 0.8370 / small 0.8654 / regular 0.9445**。
本脚本的目标是**整体超过它**：NanoBench 四个桶 + 官方三轨 + 板卡 ≤20 ms。

═══ 为什么不直接用 `search_loop.py` ═══
1. `search_loop.py` 只用 T1 快筛排雷 + 官方门禁选型，**没有 NanoBench 分层** ⇒ 选不出「小目标更好」的臂；
2. 它的 W1（9501~9504）在 Ep03 被路线二抢占 GPU 而中断，**从未门禁** ⇒ 那四个格至今是空白；
3. 它没有「按 nano 下限选 epoch」的规则（late_upsample 的 nano 随训练下滑 ⇒ 纯 F1 选臂会挑到最差 epoch）。

═══ 本脚本的流水线（每臂都跑全链）═══
  ① 训练（4 卡并发，`--skip-gate`）
  ② no-GMC 缓存（T1 val 31613 帧 / conf 0.02 / top-k 100 / --ablate-p0）
  ③ NanoBench 分层（8px 主档 + 4px 严格档）
  ④ 按 **nano 下限** 选 epoch（先取 nano ≥ 下限的集合取 F1 最高者，否则取 nano 最高者）
  ⑤ 官方三轨门禁（GMC 语义，**必须带 --baseline-cache**）
  ⑥ 导 ONNX → int8 rknn（act 跟随 ckpt 自描述 ⇒ 不会导出「silu 权重 + relu 图」的死模型）
  ⑦ **一次**板卡会话测完全部候选臂（测前清残留进程、归档 arms/、控制臂必带）
  ⑧ 联合门禁 `dual_constraint_gate.py`（速度 × nano）

═══ 铁律 ═══
* 幂等可续跑（每个环节都先查产物在不在）；硬截止 `--stop-at`；JSON 台账逐臂落盘。
* 评估与训练**不共卡**（评测用同一张卡的空闲时段串行执行）。
* GMC 与 no-GMC **严禁混用**：②③ 用 no-GMC（NanoBench），⑤ 用 GMC（官方门禁），分列报告。
* 只改本地仓库、同步服务器靠 `cp -av` + md5 双侧比对；服务器代码只读。

用法（服务器）：
    python manu/pipeline/npu/nano_search.py --wave W1
    python manu/pipeline/npu/nano_search.py --wave W1 --phase eval      # 只补跑评测/门禁
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = "/tmp/pycharm_project_10ae9e2e"
PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
RUNS = "/mnt/data/siping/runs/npu20ms"
WORK = f"{RUNS}/nano_search"
LOG = f"{WORK}/nano_search.log"
LEDGER = f"{WORK}/nano_search_ledger.json"
DATA_YAML = "/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml"
BASE_CACHE = "/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl"
CS_T0043 = "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt"
REF = "/mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json"
LQ = "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc"
UAVD = "/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc"
BOARD = "root@192.168.0.64"
BOARD_KEY = "-i /home/huangzhe/.ssh/id_board -o ConnectTimeout=10 -o StrictHostKeyChecking=no"

# ---- 基线配方 = trial_0031 的真实配方（memory §110.3，逐项对齐）----
BASE = dict(
    init=CS_T0043, epochs=20, sched=5, warmup=1.5, lr=3.5, mix="w2.0", fb=2.75, ow=0.4,
    wd=0.000269, mosaic=0.15, scale=0.10, translate=0.05, clip=5.65, backbone_lr=1.0,
    uf=0, head_src="p3", native_grid=0, mr=1,
    sw=1.0, swl=1.0, thr=16.0, maxside=4.0, lr_floor=0.0,
    hf=0, hfsrc="b1", hfch=1,
)

# ---- 交付判据（与交付件 trial_0031 对齐；NanoBench nano 下限 = 交付件自身水平）----
# t2_fp_le  = 官方门禁红线（F1 ≥ 0.985 且 FP ≤ 36）
# t2_fp_doc = 任务书「T2 虚警红线 FP > 20 一票否决」（更严 ⇒ 两条都要过）
# budget_ms = 铁律「板端 ≤ 20 ms」（任务书「+1.0 ms 即熔断」比铁律松 ⇒ 取更严的）
DELIVER = dict(nano_min=0.6666, budget_ms=20.0, t1_ge=0.885997, t2_fp_le=36, t2_fp_doc=20,
                t3_ge=0.948330)

# ---- 臂表（全部「trial_0031 配方 + 单变量」）----
W1 = [
    (9801, 19101, dict(BASE),
     "control：复现 trial_0031 配方（换种子 19101）⇒ 给出 nano/T2-FP 的种子噪声基线"),
    (9802, 19102, dict(BASE, sw=1.5),
     "方向三① 下限：微小目标(≤4px 或 ≤16px²) 的 focal 正样本权重 1.5"),
    (9803, 19103, dict(BASE, sw=3.0),
     "方向三① 上限：微小目标的 focal 正样本权重 3.0（与 9802 构成剂量-响应对）"),
    (9804, 19104, dict(BASE, native_grid=1, mr=0),
     "监督落到头的原生网格(80×80) + 单格点目标 ⇒ 直击 §88 的根因（监督分辨率≠头原生分辨率⇒峰被稀释）"),
]

# ============================================================================
# 整晚波次表（2026-11-10 17:50 排定，依据 W1 实测读数）
# ----------------------------------------------------------------------------
# W1 实测（NanoBench no-GMC 8px 主档；对照 = 9801 同配方换种子）：
#   臂        ep5 nano   ep10 nano  ep15 nano  ep20 nano   best F1     nano@4px
#   9801 对照  0.6673     0.6659     0.6668     0.6661      0.889960    0.578~0.590
#   9802 sw1.5 0.6843     0.6786     0.6699     0.6645      0.891083    0.606~0.615
#   9803 sw3.0 0.6919     0.6933     0.6902     0.6895      0.892343    0.621~0.631
#   9804 原生网格 0.6668  0.6751     0.6749     —           0.892028    0.6465~0.6609
# ⇒ ① **size-weight 剂量-响应单调为正**，且 sw=3.0 **全 epoch 稳定**（sw1.5 会随训练衰减）⇒ 上探剂量。
# ⇒ ② **原生网格把 nano@4px 从 0.578 拉到 0.661（+8.3 pp）** ⇒ §88 根因（监督分辨率≠头原生网格）成立。
# ⇒ ③ 判据基准用**对照臂 9801**（同配方换种子），不是 0031 本体。
# ============================================================================
W2 = [
    (9811, 19111, dict(BASE, hf=1, hfsrc="b1", hfch=1),
     "方向一：跨层高频残差注入（b1 源，1 窄通道）⇒ 浅层高频能量加性注入头网格"),
    (9812, 19112, dict(BASE, lr_floor=0.10),
     "方向三②：学习率保底 = 峰值的 10%（禁止退火到接近 0 ⇒ 阻止卷积核退化为低通平滑）"),
    (9813, 19113, dict(BASE, sched=8),
     "方向三③：周期退火网格 8 ep（0031=5 是下界；§124.2 长周期高峰 LR 是 nano 头号杀手）"),
    (9814, 19114, dict(BASE, sw=4.0),
     "方向三① 剂量上探：微小目标 focal 权重 4.0（sw1.5 会衰减、sw3.0 稳定 ⇒ 往高处走）"),
]
W3 = [
    (9821, 19121, dict(BASE, sw=6.0), "方向三① 剂量上探：focal 权重 6.0（找剂量峰值）"),
    (9822, 19122, dict(BASE, sw=8.0), "方向三① 剂量上探：focal 权重 8.0（找剂量峰值）"),
    (9823, 19123, dict(BASE, sw=3.0, native_grid=1, mr=0),
     "★ 组合：原生网格(定位) × sw3.0(灵敏度) ⇒ 两条独立机理叠加"),
    (9824, 19124, dict(BASE, sw=4.0, native_grid=1, mr=0),
     "★ 组合：原生网格 × sw4.0"),
]
W4 = [
    (9831, 19131, dict(BASE, hf=1, sw=3.0), "方向一 × 剂量：hf_skip + sw3.0"),
    (9832, 19132, dict(BASE, hf=1, sw=3.0, native_grid=1, mr=0), "★ 三合一：hf_skip × 原生网格 × sw3.0"),
    (9833, 19133, dict(BASE, sched=6), "方向三③ 周期网格 6 ep"),
    (9834, 19134, dict(BASE, sched=10), "方向三③ 周期网格 10 ep"),
]
W5 = [
    (9841, 19141, dict(BASE, sw=6.0, native_grid=1, mr=0), "组合：原生网格 × sw6.0"),
    (9842, 19142, dict(BASE, epochs=12), "短周期：只训 12 ep（0031 的 nano 峰值在 ep5~10 ⇒ 同样效果省 40% 训练时间）"),
    (9843, 19143, dict(BASE, sw=3.0, lr_floor=0.10), "组合：sw3.0 × LR 保底 10%"),
    (9844, 19144, dict(BASE, sched=12), "方向三③ 周期网格 12 ep（网格上界）"),
]
W6 = [
    (9851, 19151, dict(BASE, sw=8.0, native_grid=1, mr=0), "组合：原生网格 × sw8.0"),
    (9852, 19152, dict(BASE, sw=3.0, epochs=12), "短周期 × sw3.0（12 ep）"),
    (9853, 19153, dict(BASE, hf=1, hfsrc="b0"), "方向一另一取数点：b0（/2 骨干，搬运量翻倍 ⇒ 看 nano 增益是否更大）"),
    (9854, 19154, dict(BASE, sw=3.0, swl=0.5), "★ 大目标降权（regular 权重 0.5）+ sw3.0 ⇒ 削弱『大目标梯度霸凌』"),
]
W7 = [
    (9861, 19161, dict(BASE, sw=3.0, swl=0.5, native_grid=1, mr=0), "组合：原生网格 × sw3.0 × 大目标降权"),
    (9862, 19162, dict(BASE, hf=1, hfch=2), "方向一：双窄通道（hf_ch=2）"),
    (9863, 19163, dict(BASE, sw=5.0), "剂量补点 5.0"),
    (9864, 19164, dict(BASE, sw=3.0, native_grid=1, mr=0, epochs=12), "短周期 × 原生网格 × sw3.0"),
]
W8 = [
    (9871, 19171, dict(BASE, sw=3.0, warmup=0.5), "warmup 1.5 → 0.5（更快进入高 LR 段）"),
    (9872, 19172, dict(BASE, sw=3.0, lr=4.0), "lr_scale 3.5 → 4.0（mix 轴的主导因子，记忆 §0 corr=+0.824）"),
    (9873, 19173, dict(BASE, sw=3.0, mix="w2.5"), "混样 w2.0 → w2.5（甜点区上界；w8.0 已被否证）"),
    (9874, 19174, dict(BASE, sw=3.0, epochs=12, native_grid=1, mr=0), "短周期 × 原生网格 × sw3.0（复现 9864 的规格，换种子）"),
]
W9 = [
    (9881, 19181, dict(BASE, sw=3.0), "★ 种子复现：sw3.0 换种子（验证增益是否可复现，非噪声）"),
    (9882, 19182, dict(BASE, sw=4.0, native_grid=1, mr=0, epochs=12), "组合短周期版：原生网格 × sw4.0 × 12 ep"),
    (9883, 19183, dict(BASE, hf=1, sw=4.0), "方向一 × 剂量上探：hf_skip + sw4.0"),
    (9884, 19184, dict(BASE, sw=3.0, swl=0.5, epochs=12), "短周期 × sw3.0 × 大目标降权"),
]

WAVES = {"W1": W1, "W2": W2, "W3": W3, "W4": W4, "W5": W5, "W6": W6, "W7": W7, "W8": W8, "W9": W9}


def log(m: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    os.makedirs(WORK, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(cmd: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, executable="/bin/bash",
                          capture_output=True, text=True, timeout=timeout)


def train_cmd(trial: int, gpu: int, seed: int, p: dict) -> str:
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
        f"--size-weight-small {p['sw']} --size-weight-large {p['swl']} "
        f"--size-thr-px {p['thr']} --size-max-side-px {p['maxside']} "
        f"--lr-floor-ratio {p['lr_floor']} "
        f"--hf-skip {p['hf']} --hf-skip-src {p['hfsrc']} --hf-skip-channels {p['hfch']} "
        f"--reference-frame {REF} --skip-gate > {ROOT}/logs/NANO_{trial}.log 2>&1"
    )


# ============================================================================
# 台账
# ============================================================================
def load_ledger() -> dict:
    if os.path.isfile(LEDGER):
        try:
            return json.load(open(LEDGER))
        except Exception as e:  # noqa: BLE001
            log(f"[台账] 读取失败（{e}）⇒ 重建空台账")
    return {"arms": {}, "waves": [], "created": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_ledger(led: dict) -> None:
    os.makedirs(WORK, exist_ok=True)
    tmp = LEDGER + ".tmp"
    json.dump(led, open(tmp, "w"), ensure_ascii=False, indent=2)
    os.replace(tmp, LEDGER)


def arm_state(led: dict, trial: int) -> dict:
    return led["arms"].setdefault(str(trial), {"trial": trial})


# ============================================================================
# ① 训练
# ============================================================================
def phase_train(arms: list, stop_ts: float) -> None:
    led = load_ledger()
    gpus = [0, 1, 2, 3]

    def one(item):
        trial, seed, p, desc = item
        st = arm_state(led, trial)
        st.update(seed=seed, desc=desc, params={k: v for k, v in p.items()})
        last = f"{ROOT}/trials/trial_{trial}/weights/epoch{p['epochs']}.pt"
        if os.path.isfile(last):
            log(f"[{trial}] 末 epoch 权重已存在 ⇒ 跳过训练")
            st["trained"] = True
            return
        if time.time() > stop_ts:
            log(f"[{trial}] 距硬截止不足 ⇒ 不启动")
            st["trained"] = False
            return
        gpu = gpus[trial % 4]
        log(f"[{trial}] GPU{gpu} 训练 seed={seed} | {desc}")
        try:
            sh(train_cmd(trial, gpu, seed, p), timeout=36000)
        except Exception as e:  # noqa: BLE001
            log(f"[{trial}] 训练异常：{e}")
        ok = os.path.isfile(last)
        st["trained"] = ok
        log(f"[{trial}] 训练结束 ⇒ {'OK' if ok else 'FAIL（查日志）'}")
        save_ledger(led)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, arms))
    save_ledger(led)


# ============================================================================
# ②③ 缓存 + NanoBench
# ============================================================================
def phase_nanobench(arms: list, epochs: list[int], gpu_of: dict) -> None:
    sys.path.insert(0, REPO)
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415

    led = load_ledger()

    def one(item):
        trial, seed, p, desc = item
        st = arm_state(led, trial)
        traj = st.setdefault("nanobench", {})
        for ep in epochs:
            ck = f"{ROOT}/trials/trial_{trial}/weights/epoch{ep}.pt"
            if not os.path.isfile(ck):
                continue
            key = str(ep)
            if key in traj:
                continue
            pkl = f"{RUNS}/tiny_ref/nano_{trial}_ep{ep}.pkl"
            if not RA.build_cache(ck, pkl, gpu_of.get(trial, 0)):
                traj[key] = {"error": "cache_fail"}
                continue
            res = RA.nanobench(pkl, f"n{trial}_ep{ep}")
            if not res:
                traj[key] = {"error": "nanobench_fail"}
                continue
            p8 = res["peak_tol8"]
            p4 = res["peak_tol4"]
            rb, rb4 = p8["recall_by_bucket"], p4["recall_by_bucket"]
            traj[key] = {
                "f1": p8["f1"], "th": p8["th"], "fp": p8["fp"], "fn": p8["fn"],
                "nano": rb["nano"], "micro": rb["micro"], "small": rb["small"],
                "regular": rb["regular"],
                "nano4": rb4["nano"], "micro4": rb4["micro"], "f1_4": p4["f1"],
                "cache": pkl,
            }
            log(f"[{trial}] ep{ep} F1={p8['f1']:.6f} nano={rb['nano']:.4f} "
                f"nano@4px={rb4['nano']:.4f} micro={rb['micro']:.4f} regular={rb['regular']:.4f}")
            save_ledger(led)
        save_ledger(led)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, arms))


def pick_epoch(traj: dict, nano_min: float) -> tuple[dict | None, str]:
    """选 epoch：先取「nano ≥ 下限」的集合取 F1 最高者；否则取 nano 最高者。"""
    cands = [dict(v, epoch=int(k)) for k, v in traj.items() if "f1" in v]
    if not cands:
        return None, "none"
    ok = [c for c in cands if c["nano"] >= nano_min]
    if ok:
        return max(ok, key=lambda c: c["f1"]), "f1_within_nano_floor"
    return max(cands, key=lambda c: c["nano"]), "max_nano_nofloor"


# ============================================================================
# ⑤ 官方三轨门禁（GMC 语义）
# ============================================================================
def run_official_gate(trial: int, epoch: int, gpu: int) -> dict:
    w = f"{ROOT}/trials/trial_{trial}/weights/epoch{epoch}.pt"
    outd = f"{ROOT}/gate/NANO_{trial}_ep{epoch}_official"
    rp = f"{outd}/gate_report.json"
    if os.path.isfile(rp):
        log(f"[{trial}] 门禁报告已存在 ⇒ 复用 {rp}")
    else:
        os.makedirs(outd, exist_ok=True)
        cmd = (f"cd {REPO} && export PYTHONPATH=. && {PY} -u manu/training/auto_epoch_gate_check.py "
               f"--weights {w} --out-dir {outd} --device {gpu} --infer-gpu {gpu} --gpus 1 "
               f"--baseline-cache {BASE_CACHE} > {outd}_gate.log 2>&1")
        r = sh(cmd, timeout=10800)
        if not os.path.isfile(rp):
            log(f"[{trial}] 门禁未产出报告 rc={r.returncode}（查 {outd}_gate.log）")
            return {"error": f"no_report rc={r.returncode}"}
    rep = json.load(open(rp))
    tr = rep.get("tracks", {}) or {}
    t1 = (tr.get("track1_uav") or {}).get("finetune_peak") or {}
    t2 = tr.get("track2_golden") or {}
    t3 = tr.get("track3_blind") or {}
    d = {
        "out_dir": outd, "verdict": rep.get("verdict"),
        "t1": float(t1.get("f1", 0)), "t1_fp": int(t1.get("fp", 0)),
        "t2": float(t2.get("f1", 0)), "t2_fp": int(t2.get("fp", 0)),
        "t3": float(t3.get("f1", 0)), "t3_fp": int(t3.get("fp", 0)),
        "veto_failures": rep.get("veto_failures", []),
    }
    d["official_ok"] = (d["t1"] >= DELIVER["t1_ge"] and d["t2_fp"] <= DELIVER["t2_fp_le"]
                         and d["t3"] >= DELIVER["t3_ge"])
    # 任务书级「交付通过」= 官方三轨 + 任务书更严的 T2 虚警红线
    d["deliver_ok"] = (d["official_ok"] and d["t2_fp"] <= DELIVER["t2_fp_doc"])
    return d


def phase_gate(arms: list, gpu_of: dict) -> None:
    led = load_ledger()

    def one(item):
        trial, seed, p, desc = item
        st = arm_state(led, trial)
        traj = st.get("nanobench", {})
        best, how = pick_epoch(traj, DELIVER["nano_min"])
        if not best:
            log(f"[{trial}] 无有效 NanoBench 读数 ⇒ 跳过门禁")
            return
        st["picked_epoch"] = best["epoch"]
        st["select_rule"] = how
        st["picked"] = best
        log(f"[{trial}] 选 epoch={best['epoch']}（{how}）nano={best['nano']:.4f} F1={best['f1']:.6f}")
        g = run_official_gate(trial, best["epoch"], gpu_of.get(trial, 0))
        st["official"] = g
        if "error" not in g:
            log(f"[{trial}] 官方门禁：T1={g['t1']:.6f} T2={g['t2']:.6f}(FP {g['t2_fp']}) "
                f"T3={g['t3']:.6f}(FP {g['t3_fp']}) verdict={g['verdict']} ⇒ "
                f"{'PASS' if g['official_ok'] else 'FAIL'}")
        else:
            log(f"[{trial}] 官方门禁失败：{g.get('error')}")
        save_ledger(led)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, arms))
    save_ledger(led)


# ============================================================================
# ⑥⑦ 导出 + 一次板卡会话
# ============================================================================
def clean_board_residue() -> None:
    """测速前清掉板卡上的残留 NPU 进程（3 个上下文越 P=2 上限 ⇒ IOMMU page fault + 掉线）。"""
    cmd = (f"ssh {BOARD_KEY} {BOARD} "
           "\"for p in e2e_par2_demo e2e_par_demo e2e_demo rknn_heatmap_smoke; do pkill -9 -f \\$p 2>/dev/null; done; "
           "sleep 1; ps -eo pid,etime,args | grep -E '[r]knn_|[e]2e_' || echo '[CLEAN] 无残留进程'\"")
    r = sh(cmd, timeout=120)
    log(f"[board] 残留清理 ⇒ {(r.stdout or '').strip()[-200:]}")


def phase_board(arms: list) -> None:
    sys.path.insert(0, REPO)
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415

    led = load_ledger()
    board_arms = []
    for trial, seed, p, desc in arms:
        st = arm_state(led, trial)
        best = st.get("picked")
        if not best:
            continue
        name = f"n{trial}_ep{best['epoch']}"
        ck = f"{ROOT}/trials/trial_{trial}/weights/epoch{best['epoch']}.pt"
        rknn, stride = RA.export_rknn(ck, name)
        if rknn:
            board_arms.append((name, rknn, stride))
            st["rknn"] = rknn
            st["rknn_stride"] = stride
        save_ledger(led)
    if not board_arms:
        log("[board] 无候选臂 ⇒ 跳过")
        return
    clean_board_residue()
    ms_map = RA.board_measure(board_arms)
    for trial, seed, p, desc in arms:
        st = arm_state(led, trial)
        best = st.get("picked")
        if not best:
            continue
        name = f"n{trial}_ep{best['epoch']}"
        ms = ms_map.get(name)
        st["board_ms"] = ms
        st["speed_ok"] = (ms is not None and ms <= DELIVER["budget_ms"])
        log(f"[{trial}] 板卡 int8 = {ms} ms ⇒ {'PASS' if st['speed_ok'] else 'FAIL/UNKNOWN'}")
    save_ledger(led)


# ============================================================================
# ⑧ 联合门禁
# ============================================================================
def phase_dual(arms: list) -> None:
    led = load_ledger()
    specs = []
    for trial, seed, p, desc in arms:
        st = arm_state(led, trial)
        best = st.get("picked")
        if not best or not best.get("cache"):
            continue
        ms = st.get("board_ms")
        spec = f"n{trial}_ep{best['epoch']}={best['cache']}"
        if ms is not None:
            spec += f"={ms}"
        specs.append(spec)
    if not specs:
        log("[dual] 无候选 ⇒ 跳过")
        return
    out = f"{WORK}/dual_gate.json"
    cmd = (f"cd {REPO} && export PYTHONPATH=. && {PY} manu/evaluation/dual_constraint_gate.py "
           + " ".join(f"--candidate '{s}'" for s in specs)
           + f" --baseline {BASE_CACHE} --budget-ms {DELIVER['budget_ms']} "
           f"--nano-min {DELIVER['nano_min']} --out {out}")
    r = sh(cmd, timeout=7200)
    if os.path.isfile(out):
        rep = json.load(open(out))
        rows = rep.get("rows", []) or []
        for trial, seed, p, desc in arms:
            st = arm_state(led, trial)
            key = f"n{trial}_ep{st['picked']['epoch']}"
            cand = next((c for c in rows if c.get("name") == key), None)
            if cand:
                st["dual"] = cand
                st["dual_pass"] = cand.get("verdict") == "PASS"
                log(f"[{trial}] 联合门禁 ⇒ {cand.get('verdict')} "
                    f"(speed={cand.get('speed_verdict')} recall={cand.get('recall_verdict')} "
                    f"nano={cand.get('nano')} ms={cand.get('board_ms')})")
            save_ledger(led)
    else:
        log(f"[dual] 未产出报告 rc={r.returncode}：{(r.stdout or '')[-400:]}{(r.stderr or '')[-400:]}")
    save_ledger(led)


# ============================================================================
def summary(arms: list) -> None:
    led = load_ledger()
    print("\n" + "=" * 118)
    print("%-8s %-9s %-9s %-9s %-9s %-9s %-10s %-9s %-8s %-8s %s" % (
        "trial", "ep", "nano", "micro", "small", "regular", "F1", "nano@4px",
        "ms", "T2FP", "官方门禁"))
    print("-" * 118)
    for trial, seed, p, desc in arms:
        st = led["arms"].get(str(trial), {})
        b = st.get("picked")
        if not b:
            print(f"{trial:<8} (无读数)")
            continue
        g = st.get("official", {})
        ms = st.get("board_ms")
        print("%-8s %-9s %-9.4f %-9.4f %-9.4f %-9.4f %-10.6f %-9.4f %-8s %-8s %s" % (
            trial, b["epoch"], b["nano"], b["micro"], b["small"], b["regular"], b["f1"],
            b.get("nano4", 0), f"{ms:.3f}" if ms else "-",
            g.get("t2_fp", "-"), g.get("verdict", "-")))
    print("=" * 118)
    print("交付件 trial_0031 对照：nano 0.6666 / micro 0.8370 / small 0.8654 / regular 0.9445 "
          "/ F1 0.887122 / int8 19.629 ms / 官方 T1 0.885997、T2 FP 10、T3 0.948330")


def main() -> None:
    ap = argparse.ArgumentParser(description="NanoBench 定向搜索器 v3（单变量，基线 = trial_0031）")
    ap.add_argument("--wave", default="W1", choices=sorted(WAVES))
    ap.add_argument("--phase", default="all",
                    choices=["all", "train", "nanobench", "gate", "board", "dual"])
    ap.add_argument("--epochs", default="5,10,15,20", help="NanoBench 候选 epoch")
    ap.add_argument("--stop-at", default="2026-11-11 08:00", help="硬截止")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    os.makedirs(f"{RUNS}/tiny_ref", exist_ok=True)
    arms = WAVES[args.wave]
    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    stop_ts = time.mktime(time.strptime(args.stop_at, "%Y-%m-%d %H:%M"))
    gpu_of = {t: i % 4 for i, (t, _s, _p, _d) in enumerate(arms)}

    led = load_ledger()
    if args.wave not in led["waves"]:
        led["waves"].append(args.wave)
    led["base_recipe"] = BASE
    led["deliver_criteria"] = DELIVER
    save_ledger(led)

    log(f"=== NANO SEARCH v3 START | wave={args.wave} phase={args.phase} | 截止 {args.stop_at} ===")
    for trial, seed, p, desc in arms:
        log(f"  · {trial} seed={seed} | {desc}")

    if args.phase in ("all", "train"):
        phase_train(arms, stop_ts)
    if args.phase in ("all", "nanobench"):
        phase_nanobench(arms, epochs, gpu_of)
    if args.phase in ("all", "gate"):
        phase_gate(arms, gpu_of)
    if args.phase in ("all", "board"):
        phase_board(arms)
    if args.phase in ("all", "dual"):
        phase_dual(arms)
    summary(arms)
    log("=== NANO SEARCH v3 DONE ===")


if __name__ == "__main__":
    main()
