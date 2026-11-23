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
# 🔴🔴🔴 2026-11-20 16:00 **V3 锚点：当前基线本身 + 它的 4 个组成成员**（周末新增，最高优先级）
#   依据（§268 §18 实测，见 manu/memory/memory.md）：
#     · 汤的权重空间已**证伪饱和**：1,154 个有完整六项读数的臂里，
#       「其余五项 ≥ 基线 且 dT1 ≥ 0 且 dT3 > 0」的臂 = **0 个**（空集）。
#     · 292 个**训练成员**里只有 19 个 dT1 > 0，**全部 dNano < 0**
#       ⇒ 现有成员的凸包**不含**任何支配基线的点 ⇒ **冠军只能来自新训练的成员**。
#     · 唯一能在**不牺牲 nano/nano4** 的前提下把 T1/T3 往上推的动作 =
#       **从基线本身热启动的 2-epoch 微调**（起点六项 = 基线 ⇒ 任何净改善都是严格更优）。
#   1990055@5 = **当前基线 Baseline_v3.1 的权重汤**（起点六项与基线逐位相同）
#   1608003@5 / 1612001@5 / 1530006@5 / 1623003@5 = 基线的 4 个组成成员
#     （各自 dT1 全 < 0，但 dNano4/dT3 各有强项 ⇒ 抛光后可能补上缺口）
#   1543009@5 = 弱支配 1543008 的成员（dNano4 +1.42e-3 / dF1 +1.31e-4 / dT3 +1.97e-4）
V3_ANCHORS = [(1990055, 5), (1608003, 5), (1612001, 5),
              (1530006, 5), (1623003, 5), (1543009, 5)]
ANCHORS = V3_ANCHORS + R2_ANCHORS + BASE_ANCHORS
# 抛光轴 trial 号段（与历史段位错开，避免覆盖旧臂）
# 🔴 臂号段登记表（§196）：三段**互不重叠**，且都避开历史已训段位
#   历史：round-1 P=1100100~1100339、T=1100200~1100223（⚠️ 历史上 T 与 P 曾撞段）
#   round-2：T=1100900~1100915、S=1100800~1100839、P=1101000~1101329
#   规则：**任何臂号只能用一次**；新配方必须开新段（见 plan() 的唯一性断言 + 配方注册表）
POLISH_BASE = 2100000   # 抛光轴臂号段（round-2 全新段；此前 1101300 段因配方改版作废）
FULL_BASE = 2101000     # 新盆地轴臂号段（cp 维度改版后另开新段，避开注册表里的旧配方）
SEED_BASE = 2102000     # 种子扫描轴臂号段（round-2 全新段）
REPEAT_BASE = 2103000     # 阶段2「复现关」臂号段（同配方换种子重跑 ⇒ 家族 n≥4）
CP_BASE = 2104000         # 阶段3-A「cp 强度轴」臂号段
ADAPT_BASE = 2104100      # 阶段3「自适应局部搜索」臂号段
SEED2_BASE = 2104200
CP2_BASE = 2104300       # 阶段3-B「细粒度 cp 补扫」臂号段（cp 强度轴实测降级后的新段）
LOTTERY_BASE = 2105000
# 🔴🔴🔴 2026-11-20 16:35 **PT1_BASE：T1 供体再抛光轴的臂号段**（§270 新增，round-3 全新段）
#   · 依据：全台账 32 个 dT1>0 的成员**全部**出自 5 epoch 的「BASE_A2 代」；
#     当前基线的叶子全是 2 epoch 抛光代 ⇒ **从基线热启动的抛光没有 T1 升幅空间**；
#   · ⇒ 要拿到「dT1>0 且 nano 不掉」的成员，只能**从 T1 富余的 BASE_A2 成员热启动 + cp 补 nano**。
#   · 与既有段位不冲突（2105000 是 lottery，2106000 起为新段；台账该段为空）。
PT1_BASE = 2106000
# 🔴🔴🔴 2026-11-20 17:05 **L5_BASE：T1 供体「二代长训」轴的臂号段**（§282 新增，周末最后一枚新方向）
#   · 依据（§281 全台账实测）：全台账 **dT1>0** 的成员**全部**出自 **5 epoch** 的 BASE_A2 代，
#     而 PT1 轴只做 **2 epoch 轻抛** ⇒ **够不到**那个真正产出 T1 余量的训练尺度；
#   · ⇒ 本轴 = 「从 T1 供体热启动 + 5 epoch 同配方长训 + cp 补 ≤8px 监督 + L2 锚定保 T1」；
#   · 与既有段位不冲突（注册表最大 sid = 2102039 ⇒ 2107000 段整段为空）⇒ 不碰任何历史绑定。
L5_BASE = 2107000
# 🔴🔴🔴 2026-11-20 17:25 **L5B_BASE：L5 的「lr/cp 补网格」轴臂号段**（§282 追加）
#   · 动机：L5 只测了 lr {1.5, 3.5} × cp {0.05, 0.15} 四个角 ⇒ **中间的 lr 尺度（0.75 / 2.5）
#     与 cp 中点（0.10）完全没测**；
#     而「cp 0.15 ⇒ nano +8.3e-3（+35 阶）」的单点实测 ⇒ cp 0.10 的插值约 **+23 阶**
#     ⇒ **恰好等于 14963@4 的 nano 缺口 −22 阶** ⇒ 这是最有希望「一步补平」的点；
#   · ⚠ **臂号段全新（2108000+）** ⇒ 不动 2107000（L5）也不动 2100000（P）⇒ **零映射风险**；
#   · ⚠ 本机 18:00 关机 ⇒ **周一之前无法根据 L5 结果临时加轴** ⇒ 必须**现在就预先排好**补网格。
L5B_BASE = 2108000
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


MEM_START_MIN_GB = 26.0     # 低于此值**不启新臂**（§194 铁律：<30% 禁启新任务）
#   🔴🔴🔴 2026-11-20 16:55 **周末重排：42 → 26 GB**（§280）
#   背景：① 汤（权重平均）空间已实测封死 ⇒ 周末算力全部让给**训练轴**；
#        ② 服务器上同时有**外部**视频质量作业（vq_track_link_driver --workers 32），
#           RSS 合计 ~140 GB（大量 swap）⇒ MemAvailable 被压到 5~10 GB；
#        ③ 原 42 GB 门槛 ⇒ 训练轴**完全停摆**（连一臂都起不来）⇒ 周末零产出。
#   取值依据：单臂实测 PSS ~10.8~11.1 GB ⇒ 26 GB 门槛 ⇒ 起臂后仍留 ~15 GB 余量；
#   并发臂数由 --arms 控制（周末 3 臂），**闸门只负责「不越红线」，不再卡死管线**。
#   ⚠ 配套放宽：nano_watchdog.sh 的 <25 GB 杀门禁 / <12 GB 全停 ⇒ 25/6 GB。
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
    HOT_V3 = {  # 🔴 V3 锚点（基线/基线成员）的热组合
        #   **元组 = (lr_scale, lr_floor_ratio, anchor_l2, sw, cp_prob)**
        #   ⚠️ `lr` 是**倍率**不是绝对 lr（`nano_search.py:336` → `--lr-scale`，
        #      实测 G4 峰值 = 3e-5 × lr_scale）：lr=0.3 ⇒ head lr 9e-6。
        #   ⚠️ **lr_floor 在小倍率下是空操作**（实测 lr_scale=0.05 时 G2/G3 退化成常数 lr、
        #      G4 被 1e-6 绝对地板接管；lr_scale ≤ 0.25 时 G4 的 lrf 也基本不起作用）
        #      ⇒ **绝不为 lrf 单独花臂**，V3 的 lrf 只在 lr ≥ 0.25 时才真正起作用。
        #   ★ **第一个组合就是「基线自己的配方」**：实测基线 v3.1 的成员 1608003 的
        #     `finetune` 块显示它本身就是一次 **2-epoch / lr_scale=0.3 / anchor_l2=0.001 /
        #     cp=0.05 / sw=4.0** 的抛光 ⇒ 把同一套配方作用在**基线汤**上是最可能直接获胜的一枪。
        (0.30, 0.30, 0.001, 4.0, 0.05),
        (0.30, 0.20, 0.001, 4.0, 0.05), (0.30, 0.30, 0.001, 4.0, 0.00),
        (0.30, 0.30, 0.002, 4.0, 0.05), (0.30, 0.30, 0.000, 4.0, 0.05),
        (0.20, 0.20, 0.001, 4.0, 0.05), (0.20, 0.20, 0.002, 4.0, 0.00),
        (0.40, 0.30, 0.001, 4.0, 0.05), (0.40, 0.30, 0.002, 4.0, 0.00),
        (0.15, 0.10, 0.001, 4.0, 0.05), (0.15, 0.10, 0.003, 4.0, 0.00),
        (0.25, 0.20, 0.001, 4.1, 0.05), (0.35, 0.30, 0.001, 4.0, 0.10),
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
        hot_set = (HOT_V3 if anchor in V3_ANCHORS
                   else (HOT_BASE if anchor in BASE_ANCHORS else HOT_R2))
        combos = []
        #   ① lr 收窄到 0.15~0.4（夜里唯一三轴全过的臂 = lr0.3；lr≥1.0 会掉 nano 1.4e-2）
        #   ② **sw 收到 ≤4.1**：昨夜唯一三轴全过的臂用 sw4.2 ⇒ T2_fp 3（红线 ≤1）
        #      ⇒ 判「sw 抬小目标权重 ⇒ 长泉金池虚警涨」，抛光必须贴住锚点的 sw4.0
        #   ③ 加 anchor_l2 0.002/0.003（更强的 L2 锚定 ⇒ 更贴盆地 ⇒ 更不容易破 T2/T3）
        #   ④ 新增 cp_prob（微小目标 Copy-Paste 贴图）：**不用抬 sw 也能补 ≤8px 正样本监督**
        #      ⇒ 目标「nano↑ 而 T2_fp 不涨」
        lr_grid = ((0.15, 0.2, 0.25, 0.3, 0.35, 0.4) if anchor in V3_ANCHORS
                   else ((0.15, 0.2, 0.25, 0.3, 0.35, 0.4) if anchor in BASE_ANCHORS
                         else (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)))
        #   ⚠️ V3 的 l2 额外允许 **0**（= 不加锚定 ⇒ 纯探索）；其余组保持原样
        l2_grid = ((0.0, 0.001, 0.002, 0.003) if anchor in V3_ANCHORS
                   else (0.001, 0.002, 0.003))
        for lr in lr_grid:
            for lrf in (0.1, 0.2, 0.3):
                for l2 in l2_grid:
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
    # 🔴🔴 2026-11-20 16:00 **预算重分配（§268 §18）**：V3 锚点拿走 60%
    #   旧逻辑：基线锚点 16 臂/锚点 + 近失臂拿走其余 ⇒ **v2.x 时代的老锚点**（600004/610000/
    #   98902…，实测 dNano ≈ −4.7e-3 ⇒ 离当前基线差 2~3 个 nano 阶）拿走了最大一份预算。
    #   新逻辑：**V3（基线本身 + 基线成员）60% / R2（近失臂）25% / BASE（v2.x 老锚点）15%**。
    budget = max(int(args.polish_arms), 1)
    n_v3, n_r2, n_base = len(V3_ANCHORS), len(R2_ANCHORS), len(BASE_ANCHORS)
    v3_total = max(n_v3, int(budget * 0.60))
    r2_total = max(n_r2, int(budget * 0.25))
    base_total = max(n_base, budget - v3_total - r2_total)
    alloc: dict[tuple, int] = {}
    for _grp, _total in ((V3_ANCHORS, v3_total), (R2_ANCHORS, r2_total),
                          (BASE_ANCHORS, base_total)):
        _each = max(1, _total // len(_grp))
        _left = _total - _each * len(_grp)
        for _i, _a in enumerate(_grp):
            alloc[_a] = _each + (1 if _i < _left else 0)
    p_grid: list[tuple] = []
    p_prio: list[int] = []
    # 🟢 编号顺序：**近失臂锚点排在前面** ⇒ 臂号小 ⇒ 优先级序里先跑（价值最高）
    for anchor in V3_ANCHORS + R2_ANCHORS + BASE_ANCHORS:
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
                p_prio.append(1 if anchor in V3_ANCHORS
                              else (2 if anchor in R2_ANCHORS else 4))
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
        tag = ("PV3" if anchor in V3_ANCHORS else ("PX2" if anchor in R2_ANCHORS else "PX"))
        specs.append({"trial": POLISH_BASE + i, "seed": 42000 + i, "params": p,
                      "epochs": [1, 2], "channel": "P", "prio": p_prio[i],
                      # 🔴 `source_anchor` 是**升轨种子关**的必需字段：
                      #   `promote_candidate.polish_family_stats` 用它统计
                      #   「同一套抛光配方作用在 ≥2 个不同锚点」⇒ 种子关才能过。
                      "source_anchor": f"{at}@{ae}",
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


def build_pt1_specs(args, ns_mod) -> list[dict]:
    """阶段1-B：**T1 供体再抛光轴（PT1）** —— 2026-11-20 §270 新增（周末最后一枚新方向）。

    🔴 **为什么必须有这条轴**（§270 全台账实测，不是推理）：
      · 全台账 **32 个 dT1>0** 的成员**无一例外**出自 **5 epoch / lr3.5 / warmup1.5 / swb1.0**
        的「BASE_A2 代」（14963 / 14523 / 15510 / 12543 / 12703 / 9803 …）；
      · 而**当前基线的 4 个叶子全是 2 epoch 抛光代**（1100275@1 / 1100296@1 / 640001@5 / 1614000@5）
        ⇒ **从基线热启动的 2-epoch 抛光在原理上没有 T1 升幅空间**（抛光代 T1 恒在 .8907~.8909）；
      · ⇒ 想要「dT1>0 且 nano 不掉」，唯一有实测理据的动作 =
        **从 T1 富余的 BASE_A2 成员热启动，用 cp（Copy-Paste 补 ≤8px 正样本监督）把 nano 补回来**。
        cp 是**已实测的正交杠杆**：1104200（cp0.15）⇒ nano **+8.3e-3**，而 sw 抬不到这个高度；
        但 **cp ≥ 0.15 会砸穿 F1**（实测 cp0.2/0.3 ⇒ F1 −1.7e-2 ≈ −21σ）⇒ 这里只取 **0.05 / 0.10**。

    🔴 **闭环设计**：本轴产出的臂只要 dT1>0，就会被 §270 新建的 `gen_t1donor_recipes.py`
       **自动**选为供体（该生成器每次调用都重读台账）⇒
       「**训练产 T1 供体 → 基线骨架微剂量混合 → 六项候选**」形成完整闭环。
       若不产 T1 供体，则基线骨架的 dT1 恒 = 0，微剂量轴只能在 dT1 = 0 的平面上打转。

    ⚠ **臂号段独立（PT1_BASE）**：不动 P/T/S 的号段 ⇒ `arm_recipe_registry.json` 里
       既有 742 条绑定**全部不受影响**（否则同号换配方 ⇒ 计划期直接拒绝跑）。
    """
    out: list[dict] = []
    # 锚点 = §270 的 T1 余量普查 Top 供体（按「T1 效率 = dT1 / max(|dNano|,|dNano4|)」排序）
    #   14963@4 ★最佳（dT1 +3.95e-4 / dNano −5.2e-3 / **dNano4 +9.45e-4** / dF1 +3.58e-3）
    #   15510@4（dT1 +7.27e-4 / **dT3 +2.59e-3**）· 12543@3（+7.81e-4）
    #   12703@5（+7.29e-4 / dF1 +4.25e-3）· 1610001@5（+7.51e-5 / dNano 只差 −4.73e-4）
    #   14523@4（+1.01e-3）· 14241@2（+9.52e-4）
    anchors = [(14963, 4), (15510, 4), (12543, 3), (12703, 5), (1610001, 5),
               (14523, 4), (14241, 2)]
    idx = 0
    _k = 0                                   # 只在真正产臂时推进（保持历史 seed 口径不变）
    for at, ae in anchors:
        init = f"{ROOT}/trials/trial_{at}/weights/epoch{ae}.pt"
        _have = os.path.isfile(init)
        if not _have:
            log(f"[阶段1-B] T1 供体锚点 {at}@{ae} 权重不在盘 ⇒ 跳过（臂号位仍保留 ⇒ 不平移 sid↔配方映射）")
        # lr 是**倍率**（`nano_search.py` → `--lr-scale`）⇒ 小倍率下 lr_floor 是空操作
        #   （实测 lr=0.05 ⇒ G2/G3/G4 = 1.5e-7/7.5e-7/1.5e-6 ⇒ lr_floor 根本轮不到生效）
        for lr in (0.10, 0.20, 0.30):
            for cp in (0.05, 0.10):
                sid = PT1_BASE + idx
                idx += 1
                if not _have:
                    continue
                p = base_params(init, 2, 2, 0.3)
                p.update(lr=lr, lr_floor=0.3, sw=4.0, thr=12.0, maxside=3.0,
                         anchor_l2=0.001, anchor_ckpt=init, swb=1.0,
                         cp_prob=cp, cp_max_px=12.0, cp_per_img=2, cp_feather_px=2.0)
                out.append({"trial": sid, "seed": 840000 + _k * 17, "params": p,
                            "epochs": [1, 2], "channel": "PT1", "prio": 0,
                            # 🔴 `source_anchor` 是升轨「抛光家族种子关」的必需字段
                            "source_anchor": f"{at}@{ae}",
                            "desc": f"[PT1] t1donor{at}@{ae} lr{lr} cp{cp} sw4.0 thr12ms3 l2.001"})
                _k += 1
    budget = max(int(getattr(args, "pt1_arms", 30)), 0)
    if budget and len(out) > budget:
        out = out[:budget]
    if out:
        log(f"[阶段1-B] T1 供体再抛光轴 {len(out)} 臂"
            f"（锚点 {[a[0] for a in anchors]}，臂号 {PT1_BASE}~{PT1_BASE + len(out) - 1}）")
    return out


def _l5_specs_generic(base: int, anchors, lrs, cps, tag: str, seed0: int) -> list[dict]:
    """L5 / L5B 共用的臂构造器（**臂号位与 seed 位分离** ⇒ 锚点缺失不位移映射）。

    🔴 为什么要拆成两个轴（L5 / L5B）而不是一个更大网格：
      · L5 = **四个角**（lr{1.5,3.5} × cp{0.05,0.15}）⇒ 先探「这个方向到底成不成立」；
      · L5B = **中间的插值点**（lr{0.75,2.5} × cp{0.05,0.10,0.15}）⇒ 若 L5 四角全灭，
        插值点几乎不可能独活；但若 L5 成立，插值点**才有机会落在甜点上**
        （尤其 **cp 0.10 ≈ nano +23 阶 ≈ 14963@4 的 −22 阶缺口**，一步补平）。
      · ⚠ 本机 18:00 关机 ⇒ 周一之前**无人能根据 L5 结果临时加轴** ⇒ 补网格必须**现在就排好**。
    """
    out: list[dict] = []
    idx = 0
    _k = 0                                   # 只在真正产臂时推进（seed 口径与历史一致）
    for at, ae in anchors:
        init = f"{ROOT}/trials/trial_{at}/weights/epoch{ae}.pt"
        _have = os.path.isfile(init)
        if not _have:
            log(f"[{tag}] T1 供体锚点 {at}@{ae} 权重不在盘 ⇒ 跳过"
                f"（臂号位仍保留 ⇒ **不平移 sid↔配方映射**，避免旧权被记成新配方）")
        for lr in lrs:
            for cp in cps:
                sid = base + idx
                idx += 1
                if not _have:
                    continue
                p = base_params(init, 5, 5, 1.5)
                p.update(lr=lr, lr_floor=0.1, sw=4.0, swb=1.0, thr=12.0, maxside=3.0,
                         anchor_l2=0.001, anchor_ckpt=init,
                         cp_prob=cp, cp_max_px=12.0, cp_per_img=2, cp_feather_px=2.0)
                out.append({"trial": sid, "seed": seed0 + _k * 17, "params": p,
                            "epochs": [3, 4, 5], "channel": tag, "prio": 0,
                            # 🔴 `source_anchor` 是升轨「抛光家族种子关」的必需字段
                            "source_anchor": f"{at}@{ae}",
                            "desc": f"[{tag}] t1donor{at}@{ae} lr{lr} cp{cp} sw4.0 swb1.0 thr12ms3 l2.001"})
                _k += 1
    return out


def build_l5_specs(args, ns_mod) -> list[dict]:
    """阶段1-C：**T1 供体「二代长训」轴（L5）** —— 2026-11-20 §282 新增（周末最后一枚新方向）。

    🔴 **为什么 PT1 之后还要再开一条轴**（§281 全台账实测，不是推理）：
      · PT1 走的是 `base_params(init, 2, 2, 0.3)` = **2 epoch 轻抛**；
      · 而全台账里 **dT1>0 的成员无一例外出自 5 epoch 的 BASE_A2 代**
        （14963@4 / 15510@4 / 12543@3 / 12703@5 / 14523@4 / 14241@2）
        ⇒ **PT1 在训练尺度上够不到那个真正产生 T1 余量的族**；
      · ⇒ 本轴把两件事绑在一起：
          **长训（5 ep）拿 T1 余量** + **cp 补 ≤8px 正样本监督拿 nano**，
          再用 **anchor_l2 + anchor_ckpt** 把模型**钉在供体附近**（保 T1 不漂）；
      · ⇒ 「nano↑ 而 T1 不掉」这个目标，第一次有了**实测理据**的配方。

    🔴 **锚点 = 7 个实测 T1 供体**（2026-11-20 17:20 全台账复扫后定稿，详见函数体注释）：
      · 14963@4 ★**最接近冠军**：T1 +21 阶 / F1 +16 阶当量 / t2fp 4 ≤ 5 ✓，
        只差 **nano −22 阶** 与 **T3 −6 阶**（基线 = Baseline_v3.1 / trial_1990055）；
      · 1610001@5 ★**近基线供体**（SB3 汤）：nano 只差 **−2 阶** ⇒ cp 补 nano 成功率最高；
      · 15510@4（T3 +139 阶最强）· 12703@5（F1 +19 阶）· 12543@3（T1 +42 阶但 t2fp=10 超线）
        · 14523@4 · 14241@2（**台账无 official 读数 ⇒ T1 未验证**，排最后）；
      · 排序 = **执行顺序**（`next_spec()` 是 `q.pop(0)`）⇒ 最有希望的 14963@4 先跑。

    ⚠ **臂号段独立（L5_BASE）**：不动 P/T/S/CP/PT1/LOT 的号段
       ⇒ `arm_recipe_registry.json` 里既有 1404 条绑定**全部不受影响**
       （否则同号换配方 ⇒ 计划期直接拒绝跑）。
    """
    # 🔴 锚点顺序 = **执行顺序**（`next_spec()` = `q.pop(0)`）⇒ 按「离六项全过的距离」排序
    #   14963@4 ★最优（只差 nano −22 阶 + T3 −6 阶；T1 +21 / F1 +16 / nano4 +4 阶富余）
    #   1610001@5 ★近基线供体（SB3 汤）：**nano 只差 −2 阶**、F1 +7 阶、T1 +4 阶、t2fp 3
    #        ⇒ 它是「**nano 缺口最小**」的供体 ⇒ cp 补 nano 的**成功率最高**（cp0.15 实测 nano +35 阶）
    #   15510@4（T1 +39 / T3 +139 阶，但 nano −72）· 12703@5（T1 +39 / F1 +19，nano −67）
    #   12543@3（T1 +42 最高，但 t2fp=10 超红线）· 14523@4 · 14241@2（无 official 读数）
    anchors = [(14963, 4), (1610001, 5), (15510, 4), (12703, 5), (12543, 3),
               (14523, 4), (14241, 2)]
    # 🔴 `lr` 是**倍率**（`nano_search.py` → `--lr-scale`）
    #   · 1.5 = 温和续训（保 T1 的主档）
    #   · 3.5 = 复刻历史 BASE_A2 全步长（拿 T1 余量的激进档）
    # cp 是**已实测的正交杠杆**（补 ≤8px 正样本监督，不放大全局小目标权重）
    #   · 0.05 = 温和补（历史供体配方无 cp，此档尽量不破坏原盆地）
    #   · 0.15 = 强补（实测 cp0.15 ⇒ nano **+8.3e-3**；cp≥0.2 会砸穿 F1 ⇒ 不取）
    out = _l5_specs_generic(L5_BASE, anchors, (1.5, 3.5), (0.05, 0.15), "L5", 850000)
    budget = max(int(getattr(args, "l5_arms", 28)), 0)
    if budget and len(out) > budget:
        out = out[:budget]
    if out:
        log(f"[阶段1-C] T1 供体二代长训轴 {len(out)} 臂"
            f"（锚点 {[a[0] for a in anchors]}，臂号 {L5_BASE}~{L5_BASE + len(out) - 1}）")
    return out


def build_l5b_specs(args, ns_mod) -> list[dict]:
    """阶段1-D：**L5 的 lr/cp 补网格（L5B）** —— 2026-11-20 17:25 §282 追加。

    🔴 **为什么必须有这条轴**（不是保险，是算出来的）：
      · L5 只覆盖网格的**四个角**（lr{1.5,3.5} × cp{0.05,0.15}）；
      · 而 **cp 的 nano 增益只有单点实测**（cp0.15 ⇒ nano **+8.3e-3 = +35 阶**）
        ⇒ cp 0.10 的插值约 **+23 阶** ⇒ **恰好等于 14963@4 的 nano 缺口 −22 阶**
        ⇒ 「一步补平 nano」的那一枪**就在两个已测点中间**，不测就永远打不中；
      · 同理 lr 尺度：PT1 是 0.1~0.3（2 ep）、L5 是 1.5/3.5（5 ep）⇒ **0.75 / 2.5 两个尺度从未测过**。

    ⚠ **臂号段全新（L5B_BASE = 2108000）** ⇒ 不动 L5(2107000) / P(2100000) ⇒ **零映射风险**。
    ⚠ 本机 18:00 关机 ⇒ 周一前无法按 L5 结果临时加轴 ⇒ **必须现在预先排好**。
    """
    anchors = [(14963, 4), (1610001, 5), (15510, 4), (12703, 5), (12543, 3),
               (14523, 4), (14241, 2)]
    out = _l5_specs_generic(L5B_BASE, anchors, (0.75, 2.5), (0.05, 0.10, 0.15), "L5B", 860000)
    budget = max(int(getattr(args, "l5b_arms", 42)), 0)
    if budget and len(out) > budget:
        out = out[:budget]
    if out:
        log(f"[阶段1-D] L5 补网格轴（L5B）{len(out)} 臂"
            f"（lr{0.75,2.5} × cp{0.05,0.10,0.15}，臂号 {L5B_BASE}~{L5B_BASE + len(out) - 1}）")
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
    # 🔴 2026-11-20 16:05：默认预算 330 → 600
    #   依据（服务器 `night_train_axis.log` **实测**：P 臂「▶ 启动 → 臂后水位」中位 **9.2 min**
    #   （p10 7.8 / p90 12.7），docstring 里的 48 min 是陈旧值）：
    #     4 并发 × 65 h ÷ 9.2 min ≈ **1,700 臂**的吞吐上限
    #   ⇒ 600 臂是一个「够跑满周末、又不把臂号段撑爆」的预算。
    #   ⚠️ `plan()` 在**计划期**就会把臂号写进 `arm_recipe_registry.json`
    #      ⇒ 预算越大，被占用的臂号越多（2100000+ 段位 100% 空闲，可放心用）。
    ap.add_argument("--polish-arms", type=int, default=600)
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
    ap.add_argument("--pt1-arms", type=int, default=30,
                    help="阶段1-B：T1 供体再抛光轴臂数（§270；0 = 不跑）")
    ap.add_argument("--l5-arms", type=int, default=28,
                    help="阶段1-C：T1 供体二代长训轴臂数（§282；0 = 不跑）")
    ap.add_argument("--l5b-arms", type=int, default=42,
                    help="阶段1-D：L5 的 lr/cp 补网格轴臂数（§282；0 = 不跑）")
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
    pt1_specs = []
    try:
        pt1_specs = build_pt1_specs(args, NS)
    except Exception as exc:  # noqa: BLE001
        log(f"T1 供体再抛光轴构造失败（不影响其它阶段）：{exc}")
        pt1_specs = []
    l5_specs = []
    try:
        l5_specs = build_l5_specs(args, NS)
    except Exception as exc:  # noqa: BLE001
        log(f"T1 供体二代长训轴构造失败（不影响其它阶段）：{exc}")
        l5_specs = []
    l5b_specs = []
    try:
        l5b_specs = build_l5b_specs(args, NS)
    except Exception as exc:  # noqa: BLE001
        log(f"L5 补网格轴构造失败（不影响其它阶段）：{exc}")
        l5b_specs = []
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
    # 🔴 幂等（PT1 / L5 共用）：**已有 picked 且已有官方 T1 读数**的臂 ⇒ 不重复烧算力。
    #   ⚠ 只看 `picked` 不够：picked 在 epoch 评测后立刻落盘，而 `official`（T1 读数）
    #     是门禁跑完才写 ⇒ 若按 picked 过滤，会把「已训练但门禁被中断」的臂**永久丢掉 T1 读数**
    #     ⇒ 链断（供体池永远拿不到 T1 供体）。故要求 **picked 与 official.t1 同时存在**。
    def _done_with_t1(sp) -> bool:
        _st = _a0.get(str(sp["trial"])) or {}
        return bool(((_st.get("picked") or {}).get("nano") is not None)
                    and ((_st.get("official") or {}).get("t1") is not None))

    _pt1_keep = [sp for sp in pt1_specs if not _done_with_t1(sp)]
    if len(_pt1_keep) != len(pt1_specs):
        log(f"[幂等] PT1 轴已有 T1 读数的臂 {len(pt1_specs) - len(_pt1_keep)} 个 ⇒ 跳过")
    pt1_specs = _pt1_keep
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
    # 🔴 幂等：L5 臂若已有读数（重启续跑）⇒ 不重复烧算力
    _l5_keep = [sp for sp in l5_specs if not _done_with_t1(sp)]
    if len(_l5_keep) != len(l5_specs):
        log(f"[幂等] L5 轴已有读数的臂 {len(l5_specs) - len(_l5_keep)} 个 ⇒ 跳过")
    l5_specs = _l5_keep
    _l5b_keep = [sp for sp in l5b_specs if not _done_with_t1(sp)]
    if len(_l5b_keep) != len(l5b_specs):
        log(f"[幂等] L5B 轴已有 T1 读数的臂 {len(l5b_specs) - len(_l5b_keep)} 个 ⇒ 跳过")
    l5b_specs = _l5b_keep
    # ⚠ 列表顺序 = 执行顺序（`next_spec()` = `q.pop(0)`）
    #   ⇒ **L5 / L5B 紧跟 PT1 跑**（PT1 是 2 epoch 轻抛，L5/L5B 是 5 epoch 长训 ⇒ 尺度不同，互不替代）
    specs = lot_specs + pt1_specs + l5_specs + l5b_specs + cp_specs + specs
    log(f"臂计划 {len(specs)} 条（LOT={len(lot_specs)} / PT1={len(pt1_specs)} / L5={len(l5_specs)} / "
        f"L5B={len(l5b_specs)} / CP={len(cp_specs)} / "
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
            # 🔴🔴 2026-11-20 16:00 **抛光臂必须打 `polish` + `source_anchor` 标记**
            #   否则 `promote_candidate.check_arm` 的种子关分支永远进不去
            #   （`recipe_stats()` 只收 `init == cs_t0043.pt` 的臂 ⇒ 抛光臂不在普通配方组里）
            #   ⇒ 训练出来的臂**永远升不了轨**（与板卡是否空闲无关）。
            if spec["channel"] in ("P", "PT1", "L5", "L5B"):
                st["polish"] = 1
                st["source_anchor"] = spec.get("source_anchor")
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
                st["beat_to_pass_ref"] = "Baseline_v3.1(trial_1990055)"
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
            # 🔴🔴🔴 2026-11-20 §270/§282：PT1 / L5 臂（T1 供体再抛光 / 二代长训）**无条件跑官方门禁**
            #   理由：它们的起点（BASE_A2 代成员）nano 就比基线低 **22 阶** ⇒ 永远进不了
            #   triad / near(margin≥−0.3σ) 关 ⇒ 而它们的**唯一价值就是 T1 余量**
            #   ⇒ 不打官方门禁就永远量不出 T1 ⇒ 也就永远进不了 §270 新轴的供体池 ⇒ 链断。
            _force_gate = (spec.get("channel") in ("PT1", "L5", "L5B"))
            if (triad or near or _force_gate) and best:
                _why = ("★ 三轴全过" if triad else
                        ("近失臂" if near else "强制门禁（唯一价值在 T1 余量）"))
                log(f"[{trial}] {_why}（边际 {_marg:+.3f}σ）⇒ 跑官方门禁（GPU{gpu}）")
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
