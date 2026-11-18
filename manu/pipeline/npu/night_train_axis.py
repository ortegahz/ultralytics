#!/usr/bin/env python3
"""夜间训练轴（新成员生产）—— 2026-11-16 §195 之后的主力搜索轴。

为什么必须换轴（实测结论，不是猜测）
--------------------------------------
· 策展汤轴（纯权重汤）**已被证伪**：对台账里 311 个「有实测指标 + 官方两轨 + ckpt 在盘」的臂
  做加权 max-min-margin LP ⇒ 最优 min-margin = **-0.07σ**（<0 ⇒ 池内不存在能同时压过
  nano/nano4/f1 的汤）；穷举 C(52,5)=15762 组均等权重 ⇒ 过线组合 **0** 组。
· 贴线候选全部落在基线 ±1σ 内（`pareto_pool_lp.py` 输出）⇒ 基线本身的六个数已经顶到
  「nano↑ ⇒ nano4↓」这条 Pareto 前沿上 ⇒ **只能靠新训练的臂把均值抬高**。
· 关键结构事实：**汤指标 ≈ 成员加权均值** ⇒ 汤的 F1 上限 = 成员 F1 上限 = 基线 F1 ⇒
  纯汤轴在数学上不可能同时超过基线三轴。这条是本轴换线的理论依据。

两条子通道（= 2 个「通道」，符合铁律「通道并发 ≤3」）
--------------------------------------------------
· **通道 P（抛光微调轴）**：从冠军族锚点（640001/610000/98902/600004/69901 的 epoch5.pt）
  热启动，2 epoch、warmup 0.3、cosine 周期 2、lr_scale ∈ {0.2,0.5,1.0,2.0}、
  lr_floor ∈ {0.1,0.2,0.3}、可选 anchor_l2（L2 锚定）、sw ∈ {4.0,4.1,4.2}。
  ⇒ 每个 run 出 ep1/ep2 两个候选，成本 ≈ 48 min/臂（≈ 24 min/候选）。
· **通道 T（新盆地训练轴）**：从 cs_t0043 全量 5 epoch，sw ∈ {4.2,4.4,4.6,4.8}、
  thr ∈ {12,13}、maxside ∈ {3.0,3.25}、cp_prob ∈ {0,0.1}（羽化贴图）、ema_decay ∈ {0,0.999}。
  ⇒ 每个 run 出 ep3/ep5 两个候选，成本 ≈ 103 min/臂。

管线（复用 nano_search 的既有实现，保证与冠军族口径逐位一致）
------------------------------------------------------------
train_cmd → cache（RA.build_cache，GPU）→ NanoBench（进程池，纯 CPU）
→ 写台账 picked → **三轴全过才跑官方门禁**（run_official_gate）→ 写台账 official
→ 板卡 T4 / 升轨交给已在跑的 board_t4_watch + auto_promote_watch（不重复造轮子）。

资源铁律
--------
· 每进程 OMP/MKL/OPENBLAS/NUMEXPR/NUMBA 全钉 4；训练 nice 10。
· 默认 6 个训练臂并发（2 通道 × 3 臂），错峰 60 s 启动 ⇒ 内存 ≈ 6×11 GB ≈ 53%，
  CPU ≈ 6×4.1 核 ≈ 25/80 核 ⇒ 远低于 90% 红线；每臂前后打印 free/uptime/nvidia-smi 水位。
· 评测进程池 NANO_BENCH_WORKERS=2（铁律：进程池 worker ≤ 4）。

用法（服务器）
------------
  python -u manu/pipeline/npu/night_train_axis.py --arms 6 --deadline "2026-11-17 09:00" \
      --gpus 0,1,2,3 --stagger 60
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

WORK = "/mnt/data/siping/runs/npu20ms/nano_search"
LEDGER = f"{WORK}/nano_search_ledger.json"
REPO = "/tmp/pycharm_project_10ae9e2e"
ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
RUNS = "/mnt/data/siping/runs/npu20ms"
PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"
CS_T0043 = "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt"
# 冠军族锚点（trial, epoch）——「六项贴线」的最强几个（见 pareto_pool_lp.py 贴线候选）
BASE_ANCHORS = [(98905, 5), (69901, 5), (600004, 5), (98902, 5), (640001, 5),
                (620002, 5), (610000, 5), (620004, 5), (98900, 5), (98901, 5)]
# 🟢 round-2 新增「近失臂锚点」：昨夜抛光轴产出的高位臂，**直接当新起点再抛光**
#   依据（昨夜台账实测）：
#   · 1100275@1 = 唯一三轴全过臂，nano .720936 / nano4 .657136 / f1 .890828，
#     但官方门禁 t2_fp=3（红线 ≤1）、t3=.9485210（基线 .9485250，差 4e-6）⇒ 只差「T2 虚警」
#   · 1100148@1 = 昨夜 nano4 最高 .671078（基线 .6554820），配方 sw4.0（虚警安全区）、无 L2
#   · 1100287@1 = 边际 -0.006σ（f1 差 5e-6）；1100338@1 = f1 最高 .891080
#   ⇒ 「近失臂再抛光」是本轮**性价比最高的动作**：起点只差 0.06~0.28σ
R2_ANCHORS = [(1100275, 1), (1100148, 1), (1100287, 1), (1100338, 1),
              (1100296, 1), (1100143, 1), (1100262, 2), (1100250, 2)]
ANCHORS = BASE_ANCHORS + R2_ANCHORS
# 抛光轴 trial 号段（与历史段位错开，避免覆盖旧臂）
# 🔴 臂号段登记表（§196）：三段**互不重叠**，且都避开历史已训段位
#   历史：round-1 P=1100100~1100339、T=1100200~1100223（⚠️ 历史上 T 与 P 曾撞段）
#   round-2：T=1100900~1100915、S=1100800~1100839、P=1101000~1101329
#   规则：**任何臂号只能用一次**；新配方必须开新段（见 plan() 的唯一性断言 + 配方注册表）
POLISH_BASE = 1102000   # 抛光轴臂号段（round-2 全新段；此前 1101300 段因配方改版作废）
FULL_BASE = 1104300     # 新盆地轴臂号段（cp 维度改版后另开新段，避开注册表里的旧配方）
SEED_BASE = 1100800     # 种子扫描轴臂号段（round-2 全新段）
REPEAT_BASE = 1103000     # 阶段2「复现关」臂号段（同配方换种子重跑 ⇒ 家族 n≥4）
CP_BASE = 1104200         # 阶段3-A「cp 强度轴」臂号段
ADAPT_BASE = 1104000      # 阶段3「自适应局部搜索」臂号段
SEED2_BASE = 1104100
CP2_BASE = 1104400       # 阶段3-B「细粒度 cp 补扫」臂号段（cp 强度轴实测降级后的新段）
LOTTERY_BASE = 1105200
LOTTERY_REGISTRY = f"{WORK}/lottery_arm_registry.json"      # 阶段4「中签扫描」臂号段（哈希定位 ⇒ 重启后臂号稳定）      # 阶段3「追加种子扫描」臂号段
RECIPE_REGISTRY = f"{WORK}/arm_recipe_registry.json"   # 臂号 → 配方描述（防「同号换配方」）

ENV_PIN = ("export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "
           "NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4")


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def watermark(tag: str) -> str:
    """采样一次资源水位（铁律：开跑前必查）。"""
    try:
        free = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=20).stdout
        avail = [l for l in free.splitlines() if l.startswith("Mem:")]
        up = subprocess.run(["uptime"], capture_output=True, text=True, timeout=20).stdout.strip()
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        thr = subprocess.run(["bash", "-lc", "ps -eLf|wc -l"], capture_output=True, text=True, timeout=20).stdout.strip()
        line = (f"[水位 {tag}] {avail[0].strip() if avail else 'mem?'} | {up} | "
                f"threads={thr} | gpu: {gpu.replace(chr(10), ' / ')}")
    except Exception as exc:  # noqa: BLE001
        line = f"[水位 {tag}] 采样失败 {exc}"
    log(line)
    return line


MEM_START_MIN_GB = 42.0     # ≈ 总内存 33%：低于此值**不启新臂**（§194 铁律：<30% 禁启新任务）
CACHE_CONCURRENCY = 2       # 评测建缓存并发上限（单作业 ~7 GB ⇒ 并发越高峰值越危险）
#   2026-11-17 09:05 实测：5 臂 + 3 路建缓存 ⇒ MemAvailable 掉到 30.8GB（24% 总内存）
#   ⇒ 触到 30% 铁线；降到 2 路后峰值 ≈ 41GB（32%）⇒ 并发评测不再拖垮内存。
CACHE_SEM = threading.Semaphore(CACHE_CONCURRENCY)


def mem_available_gb() -> float:
    """读 MemAvailable（GB）。"""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return 999.0


def build_params(base: dict, **over) -> dict:
    p = dict(base)
    p.update(over)
    return p


def base_params(init: str, epochs: int, sched: int, warmup: float) -> dict:
    return dict(
        init=init, epochs=epochs, sched=sched, warmup=warmup, lr=3.5, mix="w2.0",
        fb=2.75, ow=0.4, wd=0.000269, mosaic=0.15, scale=0.10, translate=0.05,
        clip=5.65, backbone_lr=1.0, uf=0, head_src="p3", native_grid=0, mr=1,
        sw=4.0, swl=1.0, thr=12.0, maxside=3.0, lr_floor=0.0,
        hf=0, hfsrc="b1", hfch=1, swb=1.0, swlb=1.0,
    )


def plan(args) -> list[dict]:
    """生成确定性臂序列（P/T 交错，先做 T 的大臂，保证早班前有长臂成果）。"""
    rng = random.Random(args.seed)
    specs: list[dict] = []

    # ---- 通道 T：新盆地全量 5 epoch ----
    t_grid = []
    # 🟢 round-2 修正：sw 从 4.2~4.8 收到 **4.0/4.2**
    #   依据：昨夜唯一三轴全过臂用 sw4.2 ⇒ 官方门禁 t2_fp=3（红线 ≤1）
    #   ⇒ 高 sw 的臂基本注定被 T2 红线否掉 ⇒ 只留 4.0（贴线）与 4.2（已知会炸的对照）
    for sw in (4.0, 4.2):
        for thr in (12.0, 13.0):
            for ms in (3.0, 3.25):
                for cp in (0.0, 0.1, 0.2):
                    t_grid.append((sw, thr, ms, cp))
    # ⚠️ 臂号必须**稳定**：先按确定性顺序编号，再打乱执行顺序
    #   （否则重启后同号可能对应不同配方 ⇒ 幂等跳过训练会把旧权重复用成新配方 ⇒ 假读数）
    # 🟢 排序键把 **cp 放最前** ⇒ 截断时先砍 thr/ms 维度，而不是砍 cp 强度维度
    t_grid.sort(key=lambda x: (x[3], x[0], x[1], x[2]))
    for i, (sw, thr, ms, cp) in enumerate(t_grid):
        if i >= args.full_arms:
            break
        p = base_params(CS_T0043, 5, 5, 1.5)
        p.update(sw=sw, thr=thr, maxside=ms, lr_floor=0.1, cp_prob=cp,
                 cp_max_px=12.0, cp_per_img=2, cp_feather_px=2.0)
        if i % 4 == 3:
            p["ema_decay"] = 0.999
        specs.append({"trial": FULL_BASE + i, "seed": 41000 + i, "params": p,
                      "epochs": [3, 5], "channel": "T", "prio": 3,
                      "desc": f"[TX] sw{sw}_thr{thr}_ms{ms}_cp{cp} ema{p.get('ema_decay', 0)}"})

    # ---- 通道 P：抛光 2 epoch（热启动冠军锚点）----
    # 🟢 round-2 修正（§196）：**按锚点均衡采样**，不再「排序后取前 N」
    #   旧写法：p_grid.sort() 后取前 polish_arms ⇒ 排序键 = 锚点号升序
    #          ⇒ 只覆盖编号最小的 1~2 个锚点（600004/610000），
    #          而**昨夜唯一三轴全过的臂恰恰来自 anchor98905** ⇒ 旧采样把最有希望的锚点剪掉了。
    #   新写法：每个锚点先做 HOT 热组合（lr/lrf/l2 贴昨夜赢家，sw 贴 4.0/4.1，cp 补 ≤8px 监督），
    #            剩余预算按**跨锚点均匀 stride** 抽样 ⇒ 10 个锚点全覆盖。
    HOT_BASE = {  # 基线锚点的热组合（lr/lrf/l2 贴昨夜唯一三轴全过臂）
        (0.30, 0.30, 0.001, 4.0, 0.00), (0.30, 0.30, 0.001, 4.0, 0.05),
        (0.30, 0.30, 0.001, 4.0, 0.10), (0.30, 0.30, 0.001, 4.1, 0.00),
        (0.30, 0.30, 0.001, 4.1, 0.05), (0.30, 0.30, 0.001, 4.1, 0.10),
        (0.25, 0.30, 0.001, 4.0, 0.05), (0.35, 0.30, 0.001, 4.0, 0.05),
        (0.30, 0.20, 0.001, 4.0, 0.05), (0.30, 0.20, 0.002, 4.0, 0.05),
        (0.25, 0.20, 0.002, 4.0, 0.00), (0.35, 0.20, 0.002, 4.1, 0.05),
    }
    HOT_R2 = {  # 近失臂的热组合：**lr 再小一档**（起点已抛光过，别把它踢出盆地）+ 强 L2 + sw≤4.1
        (0.05, 0.10, 0.001, 4.0, 0.00), (0.05, 0.20, 0.001, 4.0, 0.05),
        (0.10, 0.10, 0.001, 4.0, 0.00), (0.10, 0.20, 0.001, 4.0, 0.05),
        (0.10, 0.30, 0.001, 4.1, 0.00), (0.15, 0.10, 0.001, 4.0, 0.00),
        (0.15, 0.20, 0.001, 4.0, 0.05), (0.15, 0.20, 0.002, 4.1, 0.05),
        (0.20, 0.10, 0.001, 4.0, 0.00), (0.20, 0.20, 0.001, 4.0, 0.05),
        (0.20, 0.30, 0.002, 4.1, 0.00), (0.25, 0.20, 0.001, 4.0, 0.05),
        (0.25, 0.30, 0.002, 4.1, 0.05), (0.30, 0.30, 0.001, 4.0, 0.00),
    }
    per_anchor: dict[tuple, tuple] = {}
    for anchor in ANCHORS:
        hot_set = HOT_BASE if anchor in BASE_ANCHORS else HOT_R2
        combos = []
        #   ① lr 收窄到 0.15~0.4（夜里唯一三轴全过的臂 = lr0.3；lr≥1.0 会掉 nano 1.4e-2）
        #   ② **sw 收到 ≤4.1**：昨夜唯一三轴全过的臂用 sw4.2 ⇒ T2_fp 3（红线 ≤1）
        #      ⇒ 判「sw 抬小目标权重 ⇒ 长泉金池虚警涨」，抛光必须贴住锚点的 sw4.0
        #   ③ 加 anchor_l2 0.002/0.003（更强的 L2 锚定 ⇒ 更贴盆地 ⇒ 更不容易破 T2/T3）
        #   ④ 新增 cp_prob（微小目标 Copy-Paste 贴图）：**不用抬 sw 也能补 ≤8px 正样本监督**
        #      ⇒ 目标「nano↑ 而 T2_fp 不涨」
        lr_grid = ((0.15, 0.2, 0.25, 0.3, 0.35, 0.4) if anchor in BASE_ANCHORS
                   else (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4))
        for lr in lr_grid:
            for lrf in (0.1, 0.2, 0.3):
                for l2 in (0.001, 0.002, 0.003):
                    for sw in (4.0, 4.1):
                        for cp in (0.0, 0.05, 0.1):
                            combos.append((anchor, lr, lrf, l2, sw, cp))
        # 少量大步（≈5%）：当作「从锚点开新盆地」的补充探索
        for lr in (1.0, 2.0):
            for lrf in (0.1, 0.2):
                combos.append((anchor, lr, lrf, 0.0, 4.0, 0.0))
        hot = [c for c in combos if (c[1], c[2], c[3], c[4], c[5]) in hot_set]
        rest = [c for c in combos if (c[1], c[2], c[3], c[4], c[5]) not in hot_set]
        per_anchor[anchor] = (hot, rest)
    # 预算分层：基线锚点 16 臂（12 热 + 4 stride）；近失臂锚点拿走其余（价值最高）
    base_each = 16
    n_base, n_r2 = len(BASE_ANCHORS), len(R2_ANCHORS)
    budget = max(args.polish_arms, base_each * n_base)
    r2_total = max(n_r2, budget - base_each * n_base)
    r2_each = max(1, r2_total // n_r2)
    leftover = max(0, budget - base_each * n_base - r2_each * n_r2)
    alloc: dict[tuple, int] = {a: base_each for a in BASE_ANCHORS}
    for i, a in enumerate(R2_ANCHORS):
        alloc[a] = r2_each + (1 if i < leftover else 0)
    p_grid: list[tuple] = []
    p_prio: list[int] = []
    # 🟢 编号顺序：**近失臂锚点排在前面** ⇒ 臂号小 ⇒ 优先级序里先跑（价值最高）
    for anchor in R2_ANCHORS + BASE_ANCHORS:
        hot, rest = per_anchor[anchor]
        take = alloc[anchor]
        sel = list(hot[:take])
        p_grid.extend(sel)
        p_prio.extend([0] * len(sel))          # prio 0：热组合（最先跑）
        extra = take - len(sel)
        if extra > 0:
            stride = len(rest) / float(extra)
            for k in range(extra):
                p_grid.append(rest[int(k * stride)])
                p_prio.append(2 if anchor in R2_ANCHORS else 4)
    for i, (anchor, lr, lrf, l2, sw, cp) in enumerate(p_grid):
        at, ae = anchor
        init = f"{ROOT}/trials/trial_{at}/weights/epoch{ae}.pt"
        p = base_params(init, 2, 2, 0.3)
        p.update(lr=lr, lr_floor=lrf, sw=sw, thr=12.0, maxside=3.0)
        if l2 > 0:
            p["anchor_l2"] = l2
            p["anchor_ckpt"] = init
        if cp > 0:
            p.update(cp_prob=cp, cp_max_px=12.0, cp_per_img=2, cp_feather_px=2.0)
        tag = "PX2" if anchor in R2_ANCHORS else "PX"
        specs.append({"trial": POLISH_BASE + i, "seed": 42000 + i, "params": p,
                      "epochs": [1, 2], "channel": "P", "prio": p_prio[i],
                      "desc": (f"[{tag}] anchor{at}@{ae} lr{lr} lrf{lrf} l2{l2} sw{sw} cp{cp}")})
    # ---- 通道 S：冠军配方**种子扫描**（cs_t0043 → sw4.0/thr12/ms3/lrf0.1, 5 ep）----
    #   依据：实验日志 §7「冠军种子臂命中率高 6 倍」（1/23 vs 0.7%）⇒ 这是历史最高产轴。
    for i in range(args.seed_arms):
        p = base_params(CS_T0043, 5, 5, 1.5)
        p.update(sw=4.0, thr=12.0, maxside=3.0, lr_floor=0.1)
        specs.append({"trial": SEED_BASE + i, "seed": 50000 + i * 7, "params": p,
                      "epochs": [3, 5], "channel": "S", "prio": 1,
                      "desc": f"[SX] seedscan_sw4.0_thr12ms3_lrf0.1 s{50000 + i * 7}"})

    # ---- 交错：P/T/S 交替，保证三条通道都在跑 ----
    out, ip, it, isd = [], 0, 0, 0
    tp = [s for s in specs if s["channel"] == "P"]
    tt = [s for s in specs if s["channel"] == "T"]
    sd = [s for s in specs if s["channel"] == "S"]
    while ip < len(tp) or it < len(tt) or isd < len(sd):
        if isd < len(sd):
            out.append(sd[isd]); isd += 1
        if it < len(tt):
            out.append(tt[it]); it += 1
        if ip < len(tp):
            out.append(tp[ip]); ip += 1
        if ip < len(tp):
            out.append(tp[ip]); ip += 1
    # 🔴 臂号唯一性断言：三段（1100800/1100900/1101000+）不得互相重叠
    ids = [s["trial"] for s in specs]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise RuntimeError(f"臂号冲突！重复段：{dup[:20]}")
    # 🔴 配方注册表：同一臂号若已绑定**不同配方** ⇒ 宁可不跑，也不得复用旧权重目录
    #   （否则「权重已存在 ⇒ 跳过训练」会把上一配方的权重当成本臂读数 ⇒ 假读数）
    try:
        with open(RECIPE_REGISTRY, "r", encoding="utf-8") as fh:
            reg = json.load(fh)
    except Exception:  # noqa: BLE001
        reg = {}
    kept, dropped = [], []
    for sp in specs:
        key = str(sp["trial"])
        old_desc = reg.get(key)
        if old_desc is not None and old_desc != sp["desc"]:
            dropped.append((key, old_desc, sp["desc"]))
            continue
        if old_desc is None:
            # 注册表里没有 ⇒ 可能是**历史遗留臂号**（round-1 的 1100100~1100339 等）
            #   若该臂号目录里已有权重 ⇒ 一律不复用（宁可少跑一个臂，也不要假读数）
            tdir = f"{ROOT}/trials/trial_{sp['trial']}"
            try:
                has_ck = os.path.isdir(f"{tdir}/weights") and any(
                    n.endswith(".pt") for n in os.listdir(f"{tdir}/weights"))
            except Exception:  # noqa: BLE001
                has_ck = False
            if has_ck:
                dropped.append((key, "历史遗留权重目录", sp["desc"]))
                continue
        kept.append(sp)
    for key, old_desc, new_desc in dropped:
        log(f"[臂号冲突] {key} 已绑定旧配方「{old_desc}」≠ 新配方「{new_desc}」⇒ 跳过该臂")
    # ---- 跳过**已经跑出读数**的臂（重启不重复烧算力；否则重启会把昨夜+今晨的成果重跑一遍）----
    try:
        with open(LEDGER, "r", encoding="utf-8") as fh:
            _led = json.load(fh)
    except Exception:  # noqa: BLE001
        _led = {}
    _arms = _led.get("arms") or {}
    _keep2 = []
    for sp in kept:
        _st = _arms.get(str(sp["trial"])) or {}
        if (_st.get("picked") or {}).get("nano") is not None:
            dropped.append((str(sp["trial"]), "已有读数", sp["desc"]))
            continue
        _keep2.append(sp)
    if len(_keep2) != len(kept):
        log(f"[幂等] 已有读数的臂 {len(kept) - len(_keep2)} 个 ⇒ 跳过（不重复烧算力）")
    specs = _keep2
    for sp in specs:
        reg[str(sp["trial"])] = sp["desc"]
    try:
        with open(RECIPE_REGISTRY, "w", encoding="utf-8") as fh:
            json.dump(reg, fh, ensure_ascii=False, indent=1)
    except Exception as exc:  # noqa: BLE001
        log(f"配方注册表写失败：{exc}")
    # （过滤后）重建交错队列：S → T → P → P
    out = []
    tp = [s for s in specs if s["channel"] == "P"]
    tt = [s for s in specs if s["channel"] == "T"]
    sd = [s for s in specs if s["channel"] == "S"]
    ip = it = isd = 0
    while ip < len(tp) or it < len(tt) or isd < len(sd):
        if isd < len(sd):
            out.append(sd[isd]); isd += 1
        if it < len(tt):
            out.append(tt[it]); it += 1
        if ip < len(tp):
            out.append(tp[ip]); ip += 1
        if ip < len(tp):
            out.append(tp[ip]); ip += 1
    # 🟢 执行顺序 = **优先级序**（不再随机打乱）
    #   prio 0=P 热组合（近失臂/基线锚点）→ 1=S 种子扫描 → 2=近失臂 stride → 3=T 新盆地 → 4=基线 stride
    #   理由：deadline 截断时**先砍低价值臂**，而不是随机砍掉一半好臂。
    out.sort(key=lambda sp: (sp.get("prio", 9), sp["trial"]))
    return out


def build_repeat_specs(args, ns_mod) -> list[dict]:
    """阶段2：**复现关** —— 把本轮最好的 K 个配方**换种子**再跑 3 次。

    为什么必须有这一步（§196）：
    · `auto_promote_watch` 的升轨判据是「**同配方族六项全过臂 n≥4** + 板卡 T4 ≤ 19.989 ms」；
    · 而抛光轴每个臂的配方（lr/lrf/l2/sw/cp/锚点）都不同 ⇒ 家族内天然只有 n=1
      ⇒ **即使跑出六项全过的臂，也永远凑不齐 n≥4 ⇒ 夜里不会自动升轨**；
    · 家族键（`family_key.arm_family_key`）只取 init + BASE_KEYS（**不含 seed、不含 anchor_l2**）
      ⇒ 同配方换种子重跑会**自动落进同一个家族**，n 变成 4 ⇒ 升轨关可达。
    ⇒ 本阶段挑「边际最好且已跑过官方门禁」的 K 个配方，各换 3 个种子重跑。

    返回的臂仍走同一条管线（train→cache→NanoBench→官方门禁→台账）。
    """
    from manu.pipeline.npu import family_key as FK  # 局部导入 ⇒ 无循环依赖
    led = ns_mod.load_ledger()
    arms = led.get("arms") or {}
    cands = []
    for key, st in arms.items():
        try:
            t = int(key)
        except Exception:  # noqa: BLE001
            continue
        if not (POLISH_BASE <= t < POLISH_BASE + 1000):
            continue
        picked = st.get("picked") or {}
        if not picked or "nano" not in picked:
            continue
        marg = min((picked.get("nano", 0) - ns_mod.BASE_V2["nano"]) / 0.0042,
                   (picked.get("nano4", 0) - ns_mod.BASE_V2["nano4"]) / 0.0060,
                   (picked.get("f1", 0) - ns_mod.BASE_V2["f1"]) / 0.0008)
        if marg < -0.35:      # 太远 ⇒ 复现也没意义
            continue
        try:
            fk = FK.arm_family_key(st)
        except Exception:  # noqa: BLE001
            fk = str(t)
        cands.append((marg, t, st, fk))
    cands.sort(key=lambda x: -x[0])
    out: list[dict] = []
    idx = 0
    seen_fk = set()
    for marg, t, st, fk in cands:
        if len(seen_fk) >= args.repeat_configs:
            break
        if fk in seen_fk:
            continue
        seen_fk.add(fk)
        params = dict(st.get("params") or {})
        if not params.get("init"):
            continue
        seed0 = int(st.get("seed") or 0)
        log(f"[阶段2] 复现配方 {fk}（原臂 {t}，边际 {marg:+.3f}σ）⇒ 换种子重跑 {args.repeat_reps} 次")
        for rep in range(2, 2 + args.repeat_reps):
            out.append({"trial": REPEAT_BASE + idx, "seed": seed0 + 100000 * rep,
                        "params": dict(params), "epochs": [1, 2], "channel": "P", "prio": 0,
                        "desc": f"{st.get('desc', '')}|rep{rep}"})
            idx += 1
    return out


def build_adapt_specs(args, ns_mod) -> list[dict]:
    """阶段3：**自适应局部搜索**（把今夜的最好臂当新锚点，在其邻域再采样）+ 追加种子扫描。

    为什么需要阶段3（§196）：
    · 阶段1（优先级网格）+ 阶段2（复现关）预计 02:30 收工 ⇒ 到 09:00 之间 **≈5 h × 5 臂 = 25 GPU·h 空转**；
      用户要求「占满算力」，空转就是浪费。
    · 阶段3 用**已得的实测结果**驱动搜索：拿今夜边际最好的 K 个配方当**新锚点**，
      在其邻域（lr×lrf×l2×sw×cp）重新采样 ⇒ 这是「近失臂再抛光」策略的**迭代版**
      （昨夜 1100275 正是这样从 98905 抛光出来的）。
    · 另加一批**追加种子扫描**（同冠军配方换新种子）⇒ 同家族 n 增长，
      有机会**原生**满足升轨关的 n≥4（不必等阶段2 的复现臂）。
    · 去重：与台账已有臂的「配方键」比对，已跑过的组合不重跑。
    """
    from manu.pipeline.npu import family_key as FK  # noqa: PLC0415
    led = ns_mod.load_ledger()
    arms = led.get("arms") or {}

    def cfg_key(p: dict) -> tuple:
        return (str(p.get("init")), round(float(p.get("lr") or 0), 6),
                round(float(p.get("lr_floor") or 0), 6), round(float(p.get("sw") or 0), 6),
                round(float(p.get("cp_prob") or 0), 6), round(float(p.get("anchor_l2") or 0), 6))

    done_keys = set()
    for st in arms.values():
        p = st.get("params") or {}
        if p.get("init"):
            done_keys.add(cfg_key(p))

    # ---- 候选：本夜所有有读数的臂 ----
    cands = []
    for key, st in arms.items():
        try:
            t = int(key)
        except Exception:  # noqa: BLE001
            continue
        if not (POLISH_BASE <= t < POLISH_BASE + 1000 or REPEAT_BASE <= t < REPEAT_BASE + 100):
            continue
        picked = st.get("picked") or {}
        if not picked or "nano" not in picked:
            continue
        marg = min((picked.get("nano", 0) - ns_mod.BASE_V2["nano"]) / 0.0042,
                   (picked.get("nano4", 0) - ns_mod.BASE_V2["nano4"]) / 0.0060,
                   (picked.get("f1", 0) - ns_mod.BASE_V2["f1"]) / 0.0008)
        if marg < -0.6:
            continue
        cands.append((marg, t, st, picked))
    cands.sort(key=lambda x: -x[0])

    out: list[dict] = []
    idx = 0
    seen_fk = set()
    for marg, t, st, picked in cands:
        if len(seen_fk) >= args.adapt_configs:
            break
        p0 = dict(st.get("params") or {})
        if not p0.get("init"):
            continue
        try:
            fk = FK.arm_family_key(st)
        except Exception:  # noqa: BLE001
            fk = str(t)
        if fk in seen_fk:
            continue
        seen_fk.add(fk)
        # 新起点 = 该臂的最好 epoch 权重
        ep = picked.get("epoch") or 1
        new_init = f"{ROOT}/trials/trial_{t}/weights/epoch{ep}.pt"
        if not os.path.isfile(new_init):
            log(f"[阶段3] 锚点 {t}@ep{ep} 权重不在盘 ⇒ 跳过")
            continue
        base_lr = float(p0.get("lr") or 0.2)
        base_lrf = float(p0.get("lr_floor") or 0.2)
        lrs = sorted({max(0.03, min(0.6, round(base_lr * f, 6)))
                      for f in (0.5, 0.75, 1.0, 1.3, 1.8)} | {0.05, 0.1, 0.15, 0.2, 0.3})
        l2s = (0.001, 0.002, 0.003)
        combos = []
        for lr in lrs:
            for lrf in (0.1, 0.2, 0.3):
                for l2 in l2s:
                    for sw in (4.0, 4.1):
                        for cp in (0.0, 0.05, 0.1, 0.2, 0.35):
                            combos.append((lr, lrf, l2, sw, cp))
        # 先做「热邻域」：lr 贴近原配方，sw 贴 4.0/4.1，cp 全扫
        hot = [c for c in combos if abs(c[0] - base_lr) < 1e-9 and c[1] == base_lrf]
        hot += [c for c in combos if abs(c[0] - base_lr) < 1e-9 and c[3] == 4.0]
        rest = [c for c in combos if c not in hot]
        take = args.adapt_per_config
        sel, picked_keys = [], set()
        for c in hot:
            if len(sel) >= take:
                break
            key = (new_init, c[0], c[1], c[3], c[4], c[2])
            if key in done_keys or key in picked_keys:
                continue
            picked_keys.add(key)
            sel.append(c)
        if len(sel) < take and rest:
            stride = len(rest) / float(max(1, take - len(sel)))
            for k in range(take - len(sel)):
                c = rest[int(k * stride) % len(rest)]
                key = (new_init, c[0], c[1], c[3], c[4], c[2])
                if key in done_keys or key in picked_keys:
                    continue
                picked_keys.add(key)
                sel.append(c)
        log(f"[阶段3] 锚点 {t}@ep{ep}（边际 {marg:+.3f}σ）⇒ 邻域 {len(sel)} 臂"
            f"（lr≈{base_lr} lrf{base_lrf} sw4.0/4.1 cp 0/0.05/0.1）")
        for lr, lrf, l2, sw, cp in sel:
            p = dict(p0)
            p.update(init=new_init, lr=lr, lr_floor=lrf, sw=sw)
            p["anchor_l2"] = l2
            p["anchor_ckpt"] = new_init
            if cp > 0:
                p.update(cp_prob=cp, cp_max_px=12.0, cp_per_img=2, cp_feather_px=2.0)
            else:
                p.pop("cp_prob", None)
                p.pop("cp_max_px", None)
            out.append({"trial": ADAPT_BASE + idx, "seed": 700000 + idx * 13, "params": p,
                        "epochs": [1, 2], "channel": "A", "prio": 0,
                        "desc": (f"[AX] anchor{t}@ep{ep} lr{lr} lrf{lrf} l2{l2} sw{sw} cp{cp}")})
            idx += 1

    # ---- 追加种子扫描：同冠军配方换新种子（同家族 n 增长）----
    used_seeds = set()
    for st in arms.values():
        sd = st.get("seed")
        if sd is not None:
            used_seeds.add(int(sd))
    sd_idx = 0
    for i in range(args.seed2_arms):
        seed = 600000 + i * 11
        while seed in used_seeds:
            seed += 1
        used_seeds.add(seed)
        p = base_params(CS_T0043, 5, 5, 1.5)
        p.update(sw=4.0, thr=12.0, maxside=3.0, lr_floor=0.1)
        out.append({"trial": SEED2_BASE + sd_idx, "seed": seed, "params": p,
                    "epochs": [3, 5], "channel": "S2", "prio": 1,
                    "desc": f"[SX2] seedscan_sw4.0_thr12ms3_lrf0.1 s{seed}"})
        sd_idx += 1
    return out


def build_cp_axis_specs(args, ns_mod) -> list[dict]:
    """阶段3-A：**cp（Copy-Paste）强度轴** —— 今夜实测逼出来的新方向。

    实测依据（2026-11-17 09:20 首批读数）：
    · 昨夜唯一三轴全过臂 1100275@1 = **sw4.2** ⇒ nano .7209 ✓，但官方门禁 **t2_fp=3（红线 ≤1）**
    · 今夜 sw4.0 抛光（1102001@2）⇒ nano .7129 ✗（掉 0.8e-2），但 nano4 **.6675**（基线 .6555，富余 +2σ）
    ⇒ **sw 沿 Pareto 前沿横移**：sw↑ ⇒ nano↑ 但 T2 虚警涨；sw↓ ⇒ nano↓。
    ⇒ 「用 sw 抬 nano」这条路在 T2 红线处被堵死。

    正交解 = **cp（Copy-Paste）**：把真实 ≤12px 小目标**贴进训练图**，
    等于**补回被 Mosaic+仿射 抹掉的 ≤8px 正样本监督**（用户方案 §5），
    它**不放大全局小目标权重** ⇒ 理论上「nano↑ 而 T2 虚警不涨」。
    且实测 cp0.05（1102001）**确实生效**（日志 `[CP] ... prob=0.05`）但强度太弱、不足以抬 nano。
    ⇒ 本轴用 cp ∈ {0.15, 0.2, 0.3} × sw ∈ {4.0, 4.1} 在**最强 sw4.0 锚点**上扫强度，
      找出「nano↑ 且 T2_fp≤1」的甜点。
    """
    out: list[dict] = []
    # 🟢 2026-11-17 10:20 重排锚点（按**实测盆地质量**排序，不是随手排的）：
    #   640001@5 ⇒ t1=.8904942（**全场最高**，比基线还高 3e-7）nano=.719991✓ nano4=.655718✓（只差 f1 3.7e-5）
    #   610000@5 ⇒ **就是基线本体**（六项全等于基线）⇒ 抛光它 = 「在不破坏六项的前提下抬 nano/nano4/f1」
    #   1100275@1 ⇒ nano=.720936✓ nano4=.657136✓ f1=.890828✓ t1=.890758✓（**只差 T2_fp=3**）⇒ 最高三轴
    #   98905@5  ⇒ t2_fp=1 ✓ t3=.948572✓（**T2/T3 最干净**）⇒ 适合当「cp 补监督」的干净底座
    #   ⇒ cp 是**唯一正交于 sw** 的杠杆（补 ≤8px 正样本监督，不放大全局小目标权重）
    #     ⇒ 「nano↑ 而 T2_fp 不涨」的甜点只能在这里找 ⇒ 故把它提到**阶段1 最前面**跑。
    anchors = [(640001, 5), (610000, 5), (1100275, 1), (98905, 5),
               (69901, 5), (98902, 5)][: max(1, args.cp_anchors)]
    idx = 0
    for at, ae in anchors:
        init = f"{ROOT}/trials/trial_{at}/weights/epoch{ae}.pt"
        if not os.path.isfile(init):
            log(f"[阶段3-A] 锚点 {at}@{ae} 权重不在盘 ⇒ 跳过")
            continue
        for sw in (4.0, 4.1):
            for cp in (0.15, 0.2, 0.3):
                for lr in (0.15, 0.3):
                    p = base_params(init, 2, 2, 0.3)
                    p.update(lr=lr, lr_floor=0.3, sw=sw, thr=12.0, maxside=3.0,
                             anchor_l2=0.001, anchor_ckpt=init,
                             cp_prob=cp, cp_max_px=12.0, cp_per_img=2, cp_feather_px=2.0)
                    out.append({"trial": CP_BASE + idx, "seed": 800000 + idx * 17, "params": p,
                                "epochs": [1, 2], "channel": "CP", "prio": 0,
                                "desc": f"[CX] anchor{at}@{ae} sw{sw} cp{cp} lr{lr}"})
                    idx += 1
    # 🟢🟢 2026-11-17 10:25 **截断 cp 强度轴**：实测 cp0.15~0.3 会**砸穿 F1**（见下）⇒ 强 cp 方向证伪
    #   实测（anchor640001，sw4.0）：1104200 cp0.15 lr0.15 ep1 F1 .887904（-2.5e-3）ep2 .883614；
    #   1104203/1104204 cp0.2/0.3 ⇒ F1 .875~.885（**最低 -1.7e-2 ≈ -21σ**）；
    #   ⇒ 代价（F1 -15σ 级）远大于收益（nano +4σ）⇒ **cp 强度必须大幅调小**（0.02~0.08），且要控贴图伪影。
    #   ⚠️ 只保留**前 8 个臂**（已跑的 1104200~1104206 + 1）⇒ 其余臂号（1104208+）作废不再复用
    #   （臂号↔配方必须一一对应且稳定，否则「权重已存在 ⇒ 跳过训练」会拿旧权重当新配方 ⇒ 假读数）。
    out = out[:8]

    # ---- 阶段3-B：**细粒度 cp 补扫**（10:20 实测逼出来）----
    #   实测 1104200（anchor640001, sw4.0, **cp0.15**, lr0.15）ep1：
    #   **nano .728025（+8.3e-3 ≈ +2σ）** / nano4 .663516（+1.35σ）/ **F1 .887904（-2.5e-3 ✗ 跌破基线）**
    #   ⇒ **cp 确实能把 nano 顶到 sw 够不到的高度**（这是 sw 做不到的）⇒ cp 方向被证实；
    #   但 cp0.15 + lr0.15 代价是 **F1 跌破基线**（很可能是贴图硬边界的假峰 ⇒ 用户方案 §5 的 T2 风险同源）。
    #   ⇒ 补扫：**更小 cp / 更小 lr / 1 epoch（少走一步）/ 强 L2（护 F1）/ 少贴多羽化（贴图伪影控制）**
    fine_anchors = [(640001, 5), (610000, 5), (98905, 5), (1100275, 1)]
    idx2 = 0
    for at, ae in fine_anchors:
        init = f"{ROOT}/trials/trial_{at}/weights/epoch{ae}.pt"
        if not os.path.isfile(init):
            log(f"[阶段3-B] 锚点 {at}@{ae} 权重不在盘 ⇒ 跳过")
            continue
        # B1：**小 cp**（0.02/0.05/0.08）+ 小 lr + **1 epoch**（少走一步 ⇒ F1 少受损）
        for cp in (0.02, 0.05, 0.08):
            for lr in (0.05, 0.10):
                p = base_params(init, 1, 1, 0.3)
                p.update(lr=lr, lr_floor=0.3, sw=4.0, thr=12.0, maxside=3.0,
                         anchor_l2=0.001, anchor_ckpt=init,
                         cp_prob=cp, cp_max_px=12.0, cp_per_img=2, cp_feather_px=2.0)
                out.append({"trial": CP2_BASE + idx2, "seed": 810000 + idx2 * 17, "params": p,
                            "epochs": [1], "channel": "CP", "prio": -1,
                            "desc": f"[CX2] anchor{at}@{ae} sw4.0 cp{cp} lr{lr} ep1"})
                idx2 += 1
        # B2：**贴图伪影控制**（每图少贴一张 + 羽化更大 + 强 L2 护 F1）
        for cp in (0.05, 0.08):
            for lr in (0.05, 0.10):
                p = base_params(init, 2, 2, 0.3)
                p.update(lr=lr, lr_floor=0.3, sw=4.0, thr=12.0, maxside=3.0,
                         anchor_l2=0.002, anchor_ckpt=init,
                         cp_prob=cp, cp_max_px=12.0, cp_per_img=1, cp_feather_px=3.0)
                out.append({"trial": CP2_BASE + idx2, "seed": 820000 + idx2 * 17, "params": p,
                            "epochs": [1, 2], "channel": "CP", "prio": -1,
                            "desc": f"[CX3] anchor{at}@{ae} sw4.0 cp{cp} lr{lr} l2.002 perimg1"})
                idx2 += 1
        # B3：**只贴更小的目标**（cp_max_px 12→8）⇒ 少教「云边大块=目标」⇒ 护 F1/T2
        for cp in (0.05, 0.10):
            p = base_params(init, 1, 1, 0.3)
            p.update(lr=0.10, lr_floor=0.3, sw=4.0, thr=12.0, maxside=3.0,
                     anchor_l2=0.001, anchor_ckpt=init,
                     cp_prob=cp, cp_max_px=8.0, cp_per_img=2, cp_feather_px=2.0)
            out.append({"trial": CP2_BASE + idx2, "seed": 830000 + idx2 * 17, "params": p,
                        "epochs": [1], "channel": "CP", "prio": -1,
                        "desc": f"[CX4] anchor{at}@{ae} sw4.0 cp{cp} maxpx8 lr0.10 ep1"})
            idx2 += 1
    if out:
        log(f"[阶段3-A] cp 强度轴 {len(out)} 臂（锚点 {[a[0] for a in anchors]}）")
    return out



LOTTERY_REASONS = "同一配方换种子 ⇒ 收割 T1/T2_fp 的种子抖动 ⇒ 把家族 n 顶满"


def build_lottery_specs(args, ns_mod, seen_desc: set) -> list[dict]:
    """阶段4：**中签扫描**（同一配方换种子 × N）⇒ 家族 n≥4 + 收割 T1/T2 抖动。

    为什么必须有这一阶段（§196 追加6，11-17 上午实测）：
    · 阶段1~3 预计 19:00 收工 ⇒ 到 09:00 还有 ≈14 h × 5 臂 ≈ 70 GPU·h 空转（违背「占满算力」）；
    · 实测：同一「三轴全过」配方在**不同种子**下 t1 抖动 ±1.5e-4、t2_fp 抖动 ±1
      （1100275 t1=.890758 ✓ vs 1100296 t1=.890044 ✗）⇒ 这是**系统性的种子彩票**，不是噪声；
    · 而升轨关要求「同家族六项全过 n≥4」⇒ **一张彩票不够，必须连买 4 张**；
    · 家族键（`family_key.arm_family_key`）**不含 seed** ⇒ 同配方换种子自动落进同一家族 ⇒ n 直接涨。

    臂号用 **crc32(family_key|rep) 取模** ⇒ 同一配方同一 rep 永远落在同一臂号
    ⇒ **重启后臂号稳定**（否则「权重已存在 ⇒ 跳过训练」会拿旧权重当新配方读数 ⇒ 假读数）。
    """
    import zlib  # noqa: PLC0415
    from manu.pipeline.npu import family_key as FK  # 局部导入 ⇒ 无循环依赖

    led = ns_mod.load_ledger()
    arms = led.get("arms") or {}
    fam_count: dict[str, int] = {}
    for st in arms.values():
        try:
            fk = FK.arm_family_key(st)
        except Exception:  # noqa: BLE001
            continue
        fam_count[fk] = fam_count.get(fk, 0) + 1

    cands = []
    for key, st in arms.items():
        try:
            t = int(key)
        except Exception:  # noqa: BLE001
            continue
        if st.get("synth"):
            continue                      # 汤臂不进训练轴
        picked = st.get("picked") or {}
        if not picked or picked.get("nano") is None:
            continue
        params = st.get("params") or {}
        if not params.get("init"):
            continue
        marg = min((picked.get("nano", 0) - ns_mod.BASE_V2["nano"]) / 0.0042,
                   (picked.get("nano4", 0) - ns_mod.BASE_V2["nano4"]) / 0.0060,
                   (picked.get("f1", 0) - ns_mod.BASE_V2["f1"]) / 0.0008)
        if marg < args.lottery_min_margin:
            continue
        try:
            fk = FK.arm_family_key(st)
        except Exception:  # noqa: BLE001
            continue
        cands.append((marg, t, st, fk))
    cands.sort(key=lambda x: (-x[0], x[1]))

    out: list[dict] = []
    used_fk: set[str] = set()
    used_slot: set[int] = set()
    # 🔴 中签臂有自己的**臂号↔配方注册表**（不写进主注册表）：
    #   否则重启后同臂号可能对应**另一个配方** ⇒ 「权重已存在 ⇒ 跳过训练」会拿旧权重当新配方 ⇒ 假读数。
    lot_reg: dict[str, str] = {}
    try:
        with open(LOTTERY_REGISTRY, "r", encoding="utf-8") as fh:
            lot_reg = json.load(fh)
    except Exception:  # noqa: BLE001
        lot_reg = {}
    for marg, t, st, fk in cands:
        if len(out) >= args.lottery_arms:
            break
        if fk in used_fk:
            continue
        used_fk.add(fk)
        n_exist = fam_count.get(fk, 0)
        if n_exist >= args.lottery_target_n:
            continue                      # 该家族已经够大 ⇒ 把算力让给别的家族
        params = dict(st.get("params") or {})
        ep = int(params.get("epochs") or 2)
        epochs = [1, 2] if ep <= 2 else ([3, 5] if ep >= 5 else [1, 2])
        seed0 = int(st.get("seed") or 0)
        for rep in range(1, args.lottery_target_n - n_exist + 1):
            desc = f"{st.get('desc', '')}|lot{fk}|{rep}"
            if desc in seen_desc:
                continue
            slot = zlib.crc32(f"{fk}|{rep}".encode("utf-8")) % 700
            if slot in used_slot:
                # 🔴 哈希撞槽 ⇒ **宁可不跑**（两个不同配方落到同一臂号 ⇒ 权重互相覆盖 ⇒ 假读数）
                log(f"[阶段4] 撞槽跳过：family={fk[:40]}… rep={rep} slot={slot}")
                continue
            used_slot.add(slot)
            trial = LOTTERY_BASE + slot
            _old = lot_reg.get(str(trial))
            if _old is not None and _old != desc:
                log(f"[阶段4] 撞配方跳过：trial={trial} 已绑定「{_old[:60]}」≠ 新配方")
                continue
            if _old is None:
                lot_reg[str(trial)] = desc
            out.append({"trial": trial, "seed": seed0 + 200000 + 137 * rep + (t % 97),
                        "params": dict(params), "epochs": epochs, "channel": "LOT",
                        "prio": 0, "desc": desc, "family_key": fk})
            seen_desc.add(desc)
    if out:
        try:
            with open(LOTTERY_REGISTRY, "w", encoding="utf-8") as fh:
                json.dump(lot_reg, fh, ensure_ascii=False, indent=1)
        except Exception as exc:  # noqa: BLE001
            log(f"[阶段4] 注册表写失败：{exc}")
        log(f"[阶段4] 中签扫描 {len(out)} 臂（家族 {len(used_fk)} 个；{LOTTERY_REASONS}）")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=int, default=6)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--stagger", type=int, default=60)
    ap.add_argument("--deadline", default="2026-11-17 09:00")
    ap.add_argument("--polish-arms", type=int, default=330)
    ap.add_argument("--full-arms", type=int, default=24)
    ap.add_argument("--seed-arms", type=int, default=40)
    ap.add_argument("--repeat-configs", type=int, default=3,
                    help="阶段2：复现几个最好的配方（每个再换种子跑 repeat-reps 次）")
    ap.add_argument("--cp-anchors", type=int, default=4,
                    help="阶段3-A：cp 强度轴用几个最强锚点")
    ap.add_argument("--adapt-configs", type=int, default=2,
                    help="阶段3：拿几个最好的臂当新锚点做局部搜索")
    ap.add_argument("--adapt-per-config", type=int, default=24,
                    help="阶段3：每个锚点采样多少个邻域配方")
    ap.add_argument("--seed2-arms", type=int, default=12,
                    help="阶段3：追加多少个冠军配方种子扫描臂（同家族 n 增长）")
    ap.add_argument("--repeat-reps", type=int, default=3,
                    help="阶段2：每个配方换几个种子（家族 n = 1 + reps，≥4 才能过升轨关）")
    ap.add_argument("--lottery-arms", type=int, default=40,
                    help="阶段4：每轮中签扫描最多排多少臂")
    ap.add_argument("--lottery-target-n", type=int, default=6,
                    help="阶段4：每个家族的目标成员数（n≥4 才够升轨关）")
    ap.add_argument("--lottery-min-margin", type=float, default=-0.35,
                    help="阶段4：只有边际 ≥ 该值的配方才值得买彩票")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--report", default=f"{WORK}/night_train_axis_report.json")
    args = ap.parse_args()

    os.environ["NANO_BENCH_WORKERS"] = "2"
    sys.path.insert(0, REPO)
    from manu.pipeline.npu import nano_search as NS  # noqa: PLC0415
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    deadline = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
    limit = {"t": deadline}      # 软截止：阶段3 会收紧，保证最后一臂在晨报前跑完
    specs = plan(args)
    # 🟢🟢 2026-11-17 10:25：**cp 强度轴提到队首**（不是阶段3）。
    #   依据（11-17 上午实测）：sw 是唯一在动 nano 的旋钮，但 sw↑ ⇒ T2 虚警涨（1100275 t2_fp=3）
    #   ⇒ 「用 sw 抬 nano」这条路在 T2 红线处被堵死；**cp（Copy-Paste）与 sw 正交**
    #   （补回被 Mosaic+仿射抹掉的 ≤8px 正样本监督，不放大全局小目标权重）
    #   ⇒ 「nano↑ 而 T2_fp 不涨」的甜点**只可能出现在 cp 轴上** ⇒ 先跑它，别等阶段3。
    # 🟢🟢 2026-11-17 11:00：**中签扫描提到队首**（prio -2）。
    #   依据（全盘门禁复算，320 个有门禁读数的臂）：**唯一 5/6 候选 1105000 只差 T2_fp（3 vs ≤1）**，
    #   且**只有基线自己**同时满足「t2_fp≤1 且 t1,t3 过线」⇒ **T2 虚警是唯一的卡点，且是逐臂的彩票**。
    #   ⇒ 最优算力分配 = **对「三轴全过」的配方反复换种子买彩票**（家族键不含 seed ⇒ n 直接涨）
    #   ⇒ 每个家族连买 4~6 张 ⇒ 命中「同家族六项全过 n≥4」⇒ 自动升轨。
    lot_specs = []
    try:
        _seen0: set[str] = set()
        with open(RECIPE_REGISTRY, "r", encoding="utf-8") as fh:
            _seen0 = set(str(v) for v in json.load(fh).values())
        lot_specs = build_lottery_specs(args, NS, _seen0)
    except Exception as exc:  # noqa: BLE001
        log(f"中签扫描构造失败（不影响其它阶段）：{exc}")
        lot_specs = []
    for sp in lot_specs:
        sp["prio"] = -2                      # 最高优先级（比 cp 轴还先）
    cp_specs = []
    try:
        cp_specs = build_cp_axis_specs(args, NS)
    except Exception as exc:  # noqa: BLE001
        log(f"cp 强度轴构造失败（不影响其它阶段）：{exc}")
        cp_specs = []
    try:
        with open(LEDGER, "r", encoding="utf-8") as fh:
            _led0 = json.load(fh)
    except Exception:  # noqa: BLE001
        _led0 = {}
    _a0 = _led0.get("arms") or {}
    _cp_keep = []
    for sp in cp_specs:
        if ((_a0.get(str(sp["trial"])) or {}).get("picked") or {}).get("nano") is not None:
            continue                        # 已有读数 ⇒ 不重复烧算力
        _cp_keep.append(sp)
    if len(_cp_keep) != len(cp_specs):
        log(f"[幂等] cp 轴已有读数的臂 {len(cp_specs) - len(_cp_keep)} 个 ⇒ 跳过")
    cp_specs = _cp_keep
    for sp in cp_specs:
        sp["prio"] = -1
    specs = lot_specs + cp_specs + specs
    log(f"臂计划 {len(specs)} 条（LOT={len(lot_specs)} / CP={len(cp_specs)} / "
        f"P={sum(1 for s in specs if s['channel']=='P')} / "
        f"T={sum(1 for s in specs if s['channel']=='T')}）｜并发 {args.arms}｜GPU {gpus}｜"
        f"截止 {args.deadline}")
    watermark("启动前")

    q = list(specs)
    qlock = threading.Lock()
    stop = threading.Event()
    done: list[dict] = []
    dlock = threading.Lock()
    fail_streak = [0]   # 连续训练失败计数（熔断用）

    def next_spec():
        with qlock:
            if not q:
                return None
            return q.pop(0)

    def worker(idx: int, gpu: int):
        # 错峰启动（铁律：多通道必须错峰 ≥30s）
        time.sleep(args.stagger * idx)
        while not stop.is_set():
            if time.time() > limit["t"]:
                log(f"[臂 {idx}] 已过截止时间（{time.strftime('%F %T', time.localtime(limit['t']))}）⇒ 收工")
                return
            # 🔴 铁律闸门：内存不足**不启新臂**（宁可空转等待，也不越 90% 红线）
            _m = mem_available_gb()
            if _m < MEM_START_MIN_GB:
                log(f"[臂 {idx}] MemAvailable={_m:.1f}GB < {MEM_START_MIN_GB}GB ⇒ 不启新臂，等待 120s")
                time.sleep(120)
                continue
            spec = next_spec()
            if spec is None:
                return
            trial = spec["trial"]
            p = spec["params"]
            epochs = spec["epochs"]
            log(f"[臂 {idx} GPU{gpu}] ▶ trial {trial} {spec['desc']} "
                f"epochs={p['epochs']} lr={p['lr']} init={os.path.basename(os.path.dirname(os.path.dirname(p['init'])))}")
            watermark(f"训练前 t{trial}")
            # 幂等续跑：末 epoch 的权重已存在 ⇒ 跳过训练直接评测（重启不重复烧算力）
            _last_ep = max(epochs)
            _ck_last = f"{ROOT}/trials/trial_{trial}/weights/epoch{_last_ep}.pt"
            if os.path.isfile(_ck_last):
                log(f"[{trial}] 权重已存在（epoch{_last_ep}.pt）⇒ 跳过训练，直接评测")
                r = None
            else:
                cmd = NS.train_cmd(trial, gpu, spec["seed"], p)
                # 🔴 nice 必须带目标命令（`nice -n 10` 单独用会 rc=125）⇒ 插到 python 之前
                cmd = cmd.replace("export PYTHONPATH=. &&",
                                  f"export PYTHONPATH=. && {ENV_PIN} &&")
                cmd = cmd.replace(PY + " -u manu/training/",
                                  "nice -n 10 " + PY + " -u manu/training/", 1)
                try:
                    r = NS.sh(cmd, timeout=6 * 3600)
                except Exception as exc:  # noqa: BLE001
                    log(f"[{trial}] 训练异常：{exc}")
                    continue
                if r.returncode != 0:
                    # 读训练日志判失败类型：已知 bug（copy-paste 广播）不计入熔断、且重排一次
                    _logf = f"{ROOT}/logs/NANO_{trial}.log"
                    _txt = ""
                    try:
                        with open(_logf, "r", encoding="utf-8", errors="ignore") as fh:
                            _txt = fh.read()[-4000:]
                    except Exception:  # noqa: BLE001
                        pass
                    known = ("small_copy_paste" in _txt) or ("could not be broadcast" in _txt)
                    log(f"[{trial}] 训练失败 rc={r.returncode}（known_bug={known}）")
                    if known and spec.get("retried") != 1:
                        spec["retried"] = 1
                        with qlock:
                            q.insert(0, spec)  # 修好 bug 后**插队重排**（不等队尾）
                        continue
                    if r.returncode == -9:
                        log(f"[{trial}] 被 SIGKILL（多半是看门狗按内存刹车）⇒ 不计熔断")
                        with qlock:
                            q.insert(0, spec)      # 放回队首，等内存回来再跑
                        continue
                    with dlock:
                        fail_streak[0] += 1
                        streak = fail_streak[0]
                    if streak >= 8:
                        log(f"[熔断] 连续 {streak} 次训练失败 ⇒ 整个训练轴停机（避免空转）")
                        stop.set()
                        return
                    continue
            with dlock:
                fail_streak[0] = 0
            # ---- 评测：每个 epoch 建缓存 + NanoBench ----
            led = NS.load_ledger()
            st = NS.arm_state(led, trial)
            st["params"] = p
            st["seed"] = spec["seed"]
            st["desc"] = spec["desc"]
            st["channel"] = spec["channel"]
            st["trained"] = True
            traj = st.setdefault("nanobench", {})
            for ep in epochs:
                ck = f"{ROOT}/trials/trial_{trial}/weights/epoch{ep}.pt"
                if not os.path.isfile(ck):
                    continue
                pkl = f"{RUNS}/tiny_ref/nano_{trial}_ep{ep}.pkl"
                with CACHE_SEM:                      # 建缓存并发闸（单作业 ~7 GB）
                    _ok = RA.build_cache(ck, pkl, gpu, extra="--no-peak-attrs")
                if not _ok:
                    traj[str(ep)] = {"error": "cache_fail"}
                    continue
                try:
                    res = NS._nb_pool().submit(NS._nb_worker, (pkl, f"n{trial}_ep{ep}")).result()
                except Exception as exc:  # noqa: BLE001
                    log(f"[{trial}] ep{ep} 评测异常 {exc} ⇒ 线程内直算")
                    res = NS._nb_inline(pkl, f"n{trial}_ep{ep}")
                if not res:
                    traj[str(ep)] = {"error": "nanobench_fail"}
                    continue
                traj[str(ep)] = {
                    "f1": res["f1"], "th": res["th"], "fp": res["fp"], "fn": res["fn"],
                    "nano": res["nano"], "micro": res["micro"], "small": res["small"],
                    "regular": res["regular"], "nano4": res["nano4"], "micro4": res["micro4"],
                    "f1_4": res["f1_4"], "cache": pkl,
                }
                log(f"[{trial}] ep{ep} F1={res['f1']:.6f} nano={res['nano']:.6f} "
                    f"nano4={res['nano4']:.6f} micro={res['micro']:.4f} regular={res['regular']:.4f}")
            # ---- 选 epoch ----
            b_nano, b_nano4, b_f1 = NS.base_pick_thresholds()
            best, how = NS.pick_epoch(traj, b_nano, b_nano4, b_f1)
            if best:
                st["picked"] = best
                st["picked_epoch"] = best["epoch"]
                st["select_rule"] = how
                st["beat_to_pass_ref"] = "Baseline_v2.2(trial_610000)"
            NS.save_ledger(led)
            # ---- 三轴全过才跑官方门禁（贵）----
            triad = bool(best and best.get("nano", 0) > NS.BASE_V2["nano"]
                         and best.get("nano4", 0) > NS.BASE_V2["nano4"]
                         and best.get("f1", 0) > NS.BASE_V2["f1"])
            # 🟢 round-2：**近失臂也跑官方门禁**（min-margin ≥ -0.3σ）
            #   目的：昨夜唯一三轴全过的臂 T2_fp=3 / T3 差 4e-6 ⇒ 必须搞清「哪些配方保 T2_fp≤1」
            #   ⇒ 只看三轴是不够的，得把官方两轨一起量出来，才能指导下一轮 sw/lr/cp 的取舍。
            _marg = 1e9
            if best:
                _marg = min((best.get("nano", 0) - NS.BASE_V2["nano"]) / 0.0042,
                            (best.get("nano4", 0) - NS.BASE_V2["nano4"]) / 0.0060,
                            (best.get("f1", 0) - NS.BASE_V2["f1"]) / 0.0008)
            near = _marg >= -0.3
            rec = {"trial": trial, "gpu": gpu, "desc": spec["desc"], "triad": triad,
                   "nano": (best or {}).get("nano"), "nano4": (best or {}).get("nano4"),
                   "f1": (best or {}).get("f1"), "ep": (best or {}).get("epoch")}
            if (triad or near) and best:
                log(f"[{trial}] {'★ 三轴全过' if triad else '近失臂'}（边际 {_marg:+.3f}σ）⇒ 跑官方门禁（GPU{gpu}）")
                try:
                    off = NS.run_official_gate(trial, best["epoch"], gpu)
                except Exception as exc:  # noqa: BLE001
                    off = {"error": str(exc)}
                led = NS.load_ledger()
                st = NS.arm_state(led, trial)
                st["official"] = off
                # 🔴 six_ok 必须用**严格六项**（与 six_criteria_audit.judge 同口径）：
                #   nano/nano4/f1 **严格大于**基线 + t1 ≥ 基线 + t2_fp ≤ 基线(1) + t3 ≥ 基线
                #   （昨夜误用门禁自带的 official_ok：t2_fp≤36 ⇒ 把 t2_fp=3 的臂误标 six_ok）
                st["six_ok"] = bool(
                    off.get("t1") is not None and off.get("t3") is not None
                    and best.get("nano", 0) > NS.BASE_V2["nano"]
                    and best.get("nano4", 0) > NS.BASE_V2["nano4"]
                    and best.get("f1", 0) > NS.BASE_V2["f1"]
                    and off.get("t1", 0) >= NS.BASE_V2["t1"]
                    and off.get("t2_fp", 99) <= NS.BASE_V2["t2_fp"]
                    and off.get("t3", 0) >= NS.BASE_V2["t3"])
                rec["official"] = off
                NS.save_ledger(led)
                log(f"[{trial}] 门禁：{json.dumps(off, ensure_ascii=False)[:400]}")
            with dlock:
                done.append(rec)
            watermark(f"臂后 t{trial}")
            # ---- 报告落盘（随时可查）----
            try:
                with open(args.report, "w", encoding="utf-8") as fh:
                    json.dump({"ts": time.strftime("%F %T"), "done": done,
                               "remaining": len(q), "deadline": args.deadline},
                              fh, ensure_ascii=False, indent=1)
            except Exception as exc:  # noqa: BLE001
                log(f"报告写失败：{exc}")

    def spawn_workers() -> None:
        threads = []
        for i in range(args.arms):
            gpu = gpus[i % len(gpus)]
            t = threading.Thread(target=worker, args=(i, gpu), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    spawn_workers()          # ---- 阶段 1：主计划（优先级序） ----
    # ---- 阶段 2：复现关（同配方换种子重跑 ⇒ 家族 n≥4 ⇒ 夜里能自动升轨） ----
    if time.time() < deadline - 2400:
        try:
            s2 = build_repeat_specs(args, NS)
        except Exception as exc:  # noqa: BLE001
            log(f"阶段2 构造失败（不影响阶段1成果）：{exc}")
            s2 = []
        if s2:
            log(f"阶段2 复现计划 {len(s2)} 条")
            with qlock:
                q.extend(s2)
            spawn_workers()
        else:
            log("阶段2 无可复现配方（边际最好的臂太远）⇒ 跳过")
    else:
        log("距截止不足 40 分钟 ⇒ 跳过阶段2 复现")

    # ---- 阶段 3：自适应局部搜索 + 追加种子扫描（填满剩余算力窗口） ----
    if time.time() < limit["t"] - 1800:
        s3 = []          # cp 轴已在阶段1 队首跑过 ⇒ 这里不再重复
        try:
            s3 += build_adapt_specs(args, NS)
        except Exception as exc:  # noqa: BLE001
            log(f"阶段3 构造失败（不影响前两阶段成果）：{exc}")
        if s3:
            log(f"阶段3 自适应计划 {len(s3)} 条（CP 强度轴 + A 局部搜索 + S2 追加种子）")
            limit["t"] = deadline - 1800      # 最后一臂在 08:30 前收手 ⇒ 晨报 09:10 能收录
            with qlock:
                q.extend(s3)
            spawn_workers()
        else:
            log("阶段3 无自适应配方 ⇒ 跳过")

    # ---- 阶段4：**中签扫描循环**（动态补队列，把剩余算力窗口填满）----
    seen_desc: set[str] = set()
    try:
        with open(RECIPE_REGISTRY, "r", encoding="utf-8") as fh:
            _reg = json.load(fh)
        seen_desc = set(str(v) for v in _reg.values())
    except Exception:  # noqa: BLE001
        pass
    while time.time() < limit["t"] - 900:
        try:
            s4 = build_lottery_specs(args, NS, seen_desc)
        except Exception as exc:  # noqa: BLE001
            log(f"阶段4 构造失败（不影响前面成果）：{exc}")
            s4 = []
        if not s4:
            log("阶段4 无新中签配方 ⇒ 收工")
            break
        log(f"阶段4 中签计划 {len(s4)} 条")
        with qlock:
            q.extend(s4)
        spawn_workers()
        time.sleep(60)
    else:
        log("距截止不足 30 分钟 ⇒ 跳过阶段3")
    watermark("收工")
    log(f"=== 夜间训练轴 DONE：完成 {len(done)} 臂 ===")


if __name__ == "__main__":
    main()
