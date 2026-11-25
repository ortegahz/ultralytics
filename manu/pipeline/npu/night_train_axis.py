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
# 🔴🔴🔴 2026-11-23 **PT2_BASE：「nano 富集端出发抛光」轴的臂号段**（全新段，零撞号）
#   · 依据（台账实测 2026-11-23 12:0x）：**全库 2,297 个已门禁臂里
#     「六项全过 ∧ 至少一轴严格优」= 0 臂**（57 个六项全过臂**全是纯打平**）⇒ 常规微扫没有出路；
#   · 已有的 PT1 / L5 / L5B 三条轴**全部从「T1 富 / nano 贫」的供体出发抛光**
#     （init = 15510@4 / 14963@4 一类，dNano ≈ −11…−36 阶）
#     ⇒ 抛光只会把 nano 越推越低 ⇒ **结构上补不上 nano 那一轴**；
#   · ⇒ 唯一剩下的方向：**从「nano 富集端」出发，反向抛光**（规格见 build_pt2_specs）。
#   ⚠ **臂号段零撞号验证**（扫 `nano_search_ledger.json` 的 arms 键，13,939 个数字臂号）：
#       · 2110000~2110020 **已被占用**（21 个臂）⇒ 2110000 段**不可用**；
#       · 连续空闲块 = **2110021~2119999**（9,979 个 id，零占用）；
#       · ⇒ PT2 段取 **2115000~2115119**（120 臂 = 24 run × 5 epoch），
#         上下各留 4,880 个空位 ⇒ 不碰 P(2100000+)/LOT(2105000+)/PT1(2106000+)/
#         L5(2107000+)/L5B(2108000+) ⇒ **零映射风险**；
#       · `arm_recipe_registry.json`（1,404 条绑定，最大 sid=2102039）在 2115000 段**零命中**。
PT2_BASE = 2115000
PT2_EP_PER_RUN = 5      # 每 run 5 个 epoch ⇒ 5 个臂（**逐 epoch 登记**）
# 🔴🔴🔴 2026-11-23 **PT3_BASE：「T1 富供体 × PX 族配方」局部微扫轴**的臂号段（全新段，零撞号）
#   · 依据（台账快照实测，数字权威，勿自行推算）：
#     · 全库 2358 个可读臂里「六项全过 ∧ 至少一轴严格优」= **0 臂** ⇒ 汤轴（权重汤）已被对偶证书证伪；
#     · 唯一还有希望的是**训练轴**：库里存在「只差一轴」的训练臂（2106006 只差 nano 一轴：dT1 +68.2 阶）；
#     · **[PX] 族配方是全库唯一「nano 与 T1 同涨」的旋钮**：
#         1100275（init=98905@5, sw4.2, lr0.3, lr_floor0.3, ep2）⇒ Δnano **+19.5 阶** 且 ΔT1 **+66.3 阶**；
#         1100287（同 init, lr0.5）⇒ Δnano +18.0 阶 / ΔT1 −10.0 阶；
#       ⇒ **nano:T1 兑换比 ≈ 1:0.6**（其它配方都是 1:8 ~ 1:200 ⇒ 那是「拿 T1 换 nano」的坏交易）；
#     · ⇒ **PT3 = 把这个「PX 族配方」施加到「T1 富供体」上** ⇒ 用供体已有的 T1 余量去买 nano。
#   ⚠ **臂号段零撞号验证**（扫 live 台账 + `.snap_ledger.json` 的 arms 键 + `arm_recipe_registry.json`，只读）：
#       · 2116000~2119999 **零占用**；自 2116000 起连续空闲 **4000** 个 id（下一占用段 = 2120000）；
#       · ⇒ PT3 段取 **2116000~2119999**（**上限 4000 臂**）。
#       · 段位扫描在 `build_pt3_specs` 里**每次现算**（`_pt3_alloc_base`）⇒ 被别人占了就自动后移。
#   ⚠ **满格需求（2026-11-23 增量）**：配方网格扩成「主网格 72 + 中间带 PT3M 36」= **108 run/供体**
#       ⇒ 20 供体 × 108 run × 2 臂 = **4320 臂**（2116000~2120319）；
#       但本段**只有 4000 个连续空闲 id**（2116000~2119999）⇒ **4320 装不下**
#       ⇒ 满格臂数按**实际段位下调为 4000 臂（2000 run）**，**绝不越界到 2120000+（他人段位）**。
PT3_BASE = 2116000
PT3_EP_PER_RUN = 2       # 每 run 2 epoch ⇒ 2 个臂（**逐 epoch 各成一臂**）
PT3_SEG_MIN_FREE = 4320  # 段位扫描要求：起点起至少连续空闲 4320 个臂号（= 满格需求；装不下则按实际段位下调）
# 🔴 **他人段位下界**：服务器实测（2026-11-23）`2120000~2120022` 已被他人占用、`2130000+` / `2140000+` 同
#   ⇒ `2120000+` 属**他人段位** ⇒ PT3 **绝不越界**到那里（哪怕 2120023+ 暂时空闲）。
PT3_RESERVED_FLOOR = 2120000

# ── PT3 供体表（init 权重）—— **台账实测，数字权威** ──
#   阶 = (供体读数 − 基线 Baseline_v3.1) / 该轴量子步长；量子见下方 PT3_*_STEP。
#   列：(trial, epoch, 优先级, dNano阶, dNano4阶, dF1阶, dT1阶, dT3阶, 官方t2fp, 备注)
#   ⚠ 供体**不在盘 ⇒ 跳过并打日志，但臂号位保留**（不平移 sid↔配方映射 ⇒ 重启后臂号稳定）。
#
# 🟢🟢 2026-11-23 **按「总缺口 gap」重排**（主代理实测口径）：
#   `gap = Σ max(0, −d)  over {nano, nano4, f1, t1, t3}`（t2fp 单独看，不计入 gap）
#   ⇒ gap 越小 = 离六项全过越近 = **越先跑**（下表已按主代理的「新序」列重排）。
#   优先级标签按 gap 分档（口径写在 `_pt3_donor_gap` 旁）：
#     P1 = gap ≤ 100 ｜ P2 = 100 < gap ≤ 200 ｜ P3 = gap > 200。
#   ⚠ **实测读数一律未改**，只改排列顺序与 prio 标签。
#   ⚠ **1100148@1 的落位说明**（主代理「新序」列里它写成 12，与 2106022@1 重复 ⇒ 该列缺 17）：
#     取「新序恰好是 1..20 的一个排列」这一自洽读法 ⇒ **1100148@1 = 第 17 位**
#     （12543@3 之后、14525@5 之前）。若主代理本意是与 2106022@1 相邻（第 13 位），
#     只需把下面这一行挪一格即可；**两种读法下前 10 个供体完全一致**
#     ⇒ 本轮可执行段（2000 run / 4000 臂）**完全不受影响**。
PT3_DONORS = [
    # (trial, epoch, 优先级, dNano阶, dNano4阶, dF1阶, dT1阶, dT3阶, 官方t2fp, 备注)
    (2106006, 2, "P1", -24.5, +17.0, +62.3, +68.2, +24.5, 4, "★只差 nano 一轴（T1 +68.2/nano4 +17/T3 +24.5）"),
    (14963, 4, "P1", -11.0, +4.0, +193.2, +86.8, -2.2, 4, "★nano 缺口最小(−11 阶)+T1 +86.8 阶 ⇒ A2 吸引子"),
    (15510, 4, "P1", -36.0, -8.0, +24.6, +159.8, +49.8, 3, "★T1 最强(+159.8 阶)且 T3 +49.8 阶 ⇒ 典型 T1 富供体"),
    (2106018, 2, "P1", -28.0, -18.0, +245.8, -3.0, +13.2, 1, "F1 最强 +245.8 阶、t2fp 最低 1；T1 贴线 −3 阶"),
    (1100275, 1, "P1", -7.0, -16.0, +7.8, -36.4, -7.0, 3, "PX 族本体(init=98905@5, sw4.2, lr0.3, lrf0.3)"),
    (14299, 3, "P1", -24.0, -9.0, +165.3, +37.8, -46.3, 3, "T3 −46.3 阶最差 ⇒ 只作参照"),
    (14760, 3, "P2", -14.0, -39.0, +133.4, +12.7, -49.5, 2, "F1 +133.4 阶；T3 −49.5 阶最差之一"),
    (12703, 5, "P2", -33.5, -58.0, +229.1, +160.2, -13.5, 4, "T1 +160.2 阶；nano4 −58 / T3 −13.5 阶"),
    (14971, 4, "P2", -33.0, -53.0, +163.1, +16.5, -19.6, 4, "F1 +163.1 阶；T1 +16.5 阶"),
    (1100287, 1, "P2", -8.5, -13.0, -15.7, -112.6, -8.4, 3, "PX 族 lr0.5 变体 ⇒ T1 −112.6 阶（兑换比掉档）"),
    (12702, 5, "P2", -33.0, -76.0, +218.0, +79.2, -8.1, 2, "F1 +218 阶 / T1 +79.2 阶；nano4 −76 阶"),
    (2106022, 1, "P3", -0.5, -7.0, +226.8, -200.8, +12.8, 2, "F1 最强(+226.8 阶)但 T1 深亏 −200.8 阶"),
    (14201, 5, "P2", -36.5, -91.0, +33.4, +44.7, +24.3, 4, "T3 +24.3 阶 / T1 +44.7 阶；nano4 −91 阶"),
    (12526, 4, "P2", -22.5, -78.0, +77.4, +45.6, -37.6, 5, "F1 +77.4 / T1 +45.6；nano4 −78 / T3 −37.6 阶"),
    (9915, 5, "P2", -41.5, -116.0, +148.0, +86.2, -16.5, 2, "T1 +86.2 阶 / t2fp 2；nano4 −116 阶"),
    (12543, 3, "P1", -26.5, -59.0, +123.1, +171.7, -8.7, 10, "T1 +171.7 阶但 t2fp=10 超红线(>5) ⇒ 只作参照"),
    (1100148, 1, "P2", -6.5, +43.0, -18.4, -98.4, +0.9, 5, "nano4 +43 阶最强；但 F1 −18.4 / T1 −98.4 阶"),
    (14525, 5, "P3", -38.5, -138.0, +19.1, +19.1, -31.9, 5, "nano4 −138 阶最差 ⇒ 只作参照"),
    (2106023, 1, "P3", +0.5, +33.0, +123.4, -996.5, +16.2, 5, "nano4 富余 +33 阶但 T1 亏 −996.5 阶"),
    (2106029, 1, "P3", +9.0, +65.0, -151.7, -1807.7, +0.9, 3, "nano/nano4 富集但 T1 亏 −1807.7 阶、F1 −151.7 阶"),
]
PT3_DONOR_BY_KEY = {(d[0], d[1]): d for d in PT3_DONORS}

# ── PT3 配方网格（**任务书给定口径，逐字执行**）──
#   sw ∈ {4.0,4.1,4.2,4.3} × cp_prob ∈ {0.0,0.05,0.10} × lr ∈ {0.05,0.10,0.20} × lr_floor ∈ {0.2,0.3}
#   ⇒ 每供体 4×3×3×2 = **72 个 run**；每 run 2 epoch ⇒ **2 臂/run**。
#   展开顺序 = **字典序**（sw 升序 → cp 升序 → lr 升序 → lrf 升序）⇒ 列表顺序 = 执行顺序。
PT3_SW_GRID = (4.0, 4.1, 4.2, 4.3)
PT3_CP_GRID = (0.0, 0.05, 0.10)
PT3_LR_GRID = (0.05, 0.10, 0.20)
PT3_LRF_GRID = (0.2, 0.3)
PT3_RECIPES = [(sw, cp, lr, lrf)
               for sw in PT3_SW_GRID for cp in PT3_CP_GRID
               for lr in PT3_LR_GRID for lrf in PT3_LRF_GRID]

# 🟢🟢 2026-11-23 **PT3M「中间带」配方族**（36 条/供体）
#   依据（主代理实测的配方响应面）：全库里 nano/T1 的兑换完全由 **swl（小目标梯度占比）
#   + thr（小目标判定阈值）** 决定：
#     · `sw4.0 / swl1.0 / thr12`（A2 族）⇒ **T1 富 / nano 贫**（14963：dT1 +86.8 / dNano −11.0）；
#     · `sw5.0 / swl0.1~0.3 / thr16~18`（R2/R4/R5 族）⇒ **nano 富 / T1 死**（12126：dNano +23.0 / dT1 −1738）；
#     · **中间带（swl 0.5~0.7、thr 13~14）全库零样本** ⇒ 若存在甜点，只可能在那里。
#   ⇒ 本族在**每个供体**上都跑一遍（与主网格同底子：2 epoch / sched2 / warmup0.3 / l2=0.001 /
#     mix w2.0 / fb2.75 / ow0.4 / mosaic0.15 / scale0.1 / swb1.0 / maxside3.0 / cp0.05 / lrf0.3）。
#   排列：sw↑ → swl↑ → thr↑ → lr↑（与主网格同样的字典序口径 ⇒ 臂号稳定可复现）。
PT3M_SW_GRID = (4.0, 4.2, 4.5)
PT3M_SWL_GRID = (0.5, 0.7)
PT3M_THR_GRID = (12.0, 13.0, 14.0)
PT3M_LR_GRID = (0.1, 0.3)
PT3M_LRF = 0.3
PT3M_CP = 0.05
PT3M_RECIPES = [(sw, swl, thr, lr)
                for sw in PT3M_SW_GRID for swl in PT3M_SWL_GRID
                for thr in PT3M_THR_GRID for lr in PT3M_LR_GRID]
# 每个供体的配方序列 = **72 条主网格 + 36 条中间带** = 108 run ⇒ 216 臂/供体
PT3_RUNS_PER_DONOR = len(PT3_RECIPES) + len(PT3M_RECIPES)      # = 108
PT3_ARMS_PER_DONOR = PT3_RUNS_PER_DONOR * PT3_EP_PER_RUN      # = 216

# ── 量化步长（1 量子；台账实测）与基线读数（Baseline_v3.1 = trial 1990055@5）──
PT3_NANO_STEP = 2.3629e-4   # nano 台阶（🔴 2026-11-23 修正：原 4.73e-4 大了 2 倍。
                              #   台账实测：7339 个 nano 读数与基线的差值**全部**是 2.3629489603e-4 的整数倍
                              #   ⇒ 原值会让「近过线」判据松 2 倍 ⇒ 白跑门禁、且门禁内存尖峰翻倍）
PT3_NANO4_STEP = 2.363e-4    # nano4 台阶
PT3_F1_STEP = 1.8536e-5      # f1 台阶
PT3_T1_STEP = 4.5518e-6      # T1 台阶
PT3_T3_STEP = 5.206e-5       # T3 台阶
PT3_BASE_NANO = 0.7242438563327032
PT3_BASE_NANO4 = 0.6609168241965974
PT3_BASE_F1 = 0.8906832039622916
PT3_BASE_T1 = 0.8909238824339264
PT3_BASE_T2FP = 3
PT3_BASE_T3 = 0.9488855869242199

# ── 门禁短路阈值（**PT3 专用**，与 PT2 的「200 阶 T1 近线」不同，勿照抄）──
PT3_NEAR_QUANTA = 1.0        # (A) 四轴近过线：nano/nano4/f1/t3 各容差 1 量子
PT3_T1_NEAR_QUANTA = 5.0     # (B) T1 近线：容差 5 量子
PT3_TOL = 1e-12              # 所有基线比较的数值容差（语义一律 `>=`）
# 「读不到某 epoch 的 T1/T3」时的**保守兜底**（绝对值；2-epoch 轻抛 + L2 锚定 ⇒ 假定小幅回吐）
PT3_T1_FALLBACK_DROP = 2.3e-4   # ≈ 50 个 T1 量子
PT3_T3_FALLBACK_DROP = 5.2e-4   # ≈ 10 个 T3 量子

# ── PX 族先验（单点实测；**全库唯一「nano 与 T1 同涨」的旋钮**）──
PT3_PX_TRIAL = 1100275
PT3_PX_LR = 0.3
PT3_PX_SW = 4.2
PT3_PX_NANO_STEPS = 19.5   # Δnano（相对其 init 98905@5）
PT3_PX_T1_STEPS = 66.3     # ΔT1（同一单点）
PT3_PX_RATIO = 0.6         # nano:T1 兑换比 ≈ 1:0.6（其它配方 1:8 ~ 1:200）
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


# ─────────────────────────────────────────────────────────────────────────
# 🟢🟢🟢 2026-11-23 **PT2 规格：「nano 富集端」出发做 A2 族抛光**（台账实测，数字权威）
# ─────────────────────────────────────────────────────────────────────────
# init 集 = 台账里 6 个**「nano 富集」臂**（用它们**自己的 epoch 权重**热启动）：
#   臂            dNano阶  dNano4阶  dF1阶    dT1阶    dT3阶   t2fp  说明
#   12126@4       +23.0   +94.0    −165.7   −1738   −309    0    ★首选（R2 种子扫描 5/6 冠军）
#   12130@3       +12.5   −82.0    −595.0   −3210   −458    0    R2 突变 swl0.2
#   12131@4       +70.9   +78.0    −699.6   −5776   −1104   0    R2 突变 swl0.1
#   12183@5       +65.4   +76.0    −707.3   −5751   −1101   0    合成臂 swa_late
#   12287@3       +67.9   +98.0    −340.2   −6773   −1833   0    R5 突变 uf1
#   12249@3       +56.0   +154.0   −475.0   −4981   −1297   0    R4 突变 mixw2.2
#   （阶 = 基线相对；nano 阶用**实测台阶 4.73e-4**，nano4 2.363e-4，f1 1.8536e-5，
#     T1 4.5518e-6，T3 5.206e-5）
# A2 吸引子（终点）：14963@4 = dNano −11.0 / dNano4 +4.0 / dF1 +193.2 / **dT1 +86.8** /
#   dT3 −2.2 / t2fp 4  ⇒ **全台账 dT1>0 的训练臂里 nano 最高的一个**
PT2_INITS = [(12126, 4), (12130, 3), (12131, 4), (12183, 5), (12287, 3), (12249, 3)]
# 配方网格：lr × cp_prob × swl（A2 底子：sw4.0 / thr12ms3 / maxside3.0 / mix w2.0 /
#   fb2.75 / ow0.4 / mosaic0.15 / scale0.1 / sched5 / warmup1.5 / l2=0.001）
#   ⚠ 完整网格 12 格 × 6 init = 72 run = 360 臂 ≫ 120 臂预算 ⇒ 每个 init 取 **4 个角**，
#     但**覆盖全部 6 个网格值**（lr{3.5,1.5} / cp{0.00,0.05,0.15} / swl{1.0,0.5}）：
#     ① (3.5, 0.05, 0.5) ★最高先验：全速爬 T1 + 温和 cp 补 nano + 弱大目标权重保 nano
#     ② (3.5, 0.00, 1.0) 纯 A2 对照（量轨迹本身：nano 掉多快、T1 爬多快）
#     ③ (1.5, 0.05, 0.5) 半速 + 温和 cp（若 ① 的 nano 掉太快，这是第二次机会）
#     ④ (3.5, 0.15, 0.5) 全速 + 强 cp（nano 硬补；代价是 f1）
PT2_RECIPES = [(3.5, 0.05, 0.5), (3.5, 0.00, 1.0), (1.5, 0.05, 0.5), (3.5, 0.15, 0.5)]
PT2_T1_STEP = 4.5518e-6      # T1 量化台阶（实测）
PT2_NANO_STEP = 4.73e-4      # nano 实测台阶（基线相对）
PT2_NANO4_STEP = 2.363e-4    # nano4 台阶
PT2_F1_STEP = 1.8536e-5      # f1 台阶
PT2_T1_NEAR_STEPS = 200      # 「T1 只差 ≤ 200 阶」⇒ 值得花门禁机时（短路阈值）
PT2_ATTRACTOR = (14963, 4)   # A2 吸引子锚（dT1 +86.8 阶 / dNano −11.0 阶 / t2fp 4）
# 各 init 的**实测 dT1 缺口**（阶；台账 official.t1 实测）⇒ **台账读不到时的保守兜底**
#   （否则预测起点会退化成「已在 T1 线上」⇒ 5 个 epoch 全部门禁 ⇒ 机时翻 5 倍）
PT2_INIT_T1_STEPS = {12126: -1738.0, 12130: -3210.0, 12131: -5776.0,
                      12183: -5751.0, 12287: -6773.0, 12249: -4981.0}


def _pt2_t1_of(ns_mod, trial: int):
    """台账里某臂的**官方 T1**（读不到 ⇒ None）。"""
    try:
        led = ns_mod.load_ledger()
    except Exception:  # noqa: BLE001
        return None
    st = (led.get("arms") or {}).get(str(trial)) or {}
    t1 = (st.get("official") or {}).get("t1")
    return float(t1) if t1 is not None else None


def _link_ep_ckpt(run_trial: int, arm_id: int, epoch: int) -> None:
    """把 run 的真实 epoch 权重**软链**到 epoch 臂自己的 trial 目录。

    🔴 为什么必须做：升轨/板卡工具按 `trials/trial_<臂号>/weights/epoch<ep>.pt` 解析 ckpt
      （`promote_candidate.py:502` / `auto_promote_watch.py:177`）
      ⇒ 「逐 epoch 登记」的臂（臂号 ≠ 训练 trial 号）必须有**自己的可解析路径**，
      否则升轨预检会判「ckpt 不存在」⇒ 逐 epoch 捞到的穿越点白白浪费。

    🟢🟢 2026-11-23：**泛化**为 PT2 / PT3 共用（两条轴都是「逐 epoch 各成一臂」⇒ 同一套软链逻辑）。
    """
    if int(arm_id) == int(run_trial):
        return                                    # ep1 的臂号 = 训练 trial 号 ⇒ 真实权重已在盘
    src = f"{ROOT}/trials/trial_{run_trial}/weights/epoch{epoch}.pt"
    dst_dir = f"{ROOT}/trials/trial_{arm_id}/weights"
    dst = f"{dst_dir}/epoch{epoch}.pt"
    if os.path.islink(dst) or os.path.isfile(dst):
        return
    if not os.path.isfile(src):
        return
    try:
        os.makedirs(dst_dir, exist_ok=True)
        os.symlink(os.path.relpath(src, dst_dir), dst)
    except OSError:
        pass


# 🔴 兼容别名：旧调用点（PT2）保留；PT3 走同一实现 ⇒ 行为逐字不变
_pt2_link_ckpt = _link_ep_ckpt


def _pt2_handle(spec, traj, gpu, ns_mod) -> list[dict]:
    """PT2 run：**逐 epoch 登记为臂** + **逐 epoch 官方门禁**（带短路）。

    🔴 **短路规则**（门禁 ~30~90 s/臂 ⇒ 必须先筛后门禁）：
      只有满足下面任一条的 epoch 才跑官方门禁：
        ① **三轴全过**：nano ≥ 基线 且 nano4 ≥ 基线 且 f1 ≥ 基线（门禁口径 = `>=`，不是 `>`）；
        ② **T1 近线**：该 epoch 的 T1 缺口 ≤ 200 阶（4.5518e-6/阶）。
           · 门禁**前**没有 T1 ⇒ 用**轨迹预测**：
               T1(k) ≈ T1(init) + (T1(吸引子) − T1(init))·(k−1)/(n_epoch−1)
             （T1(init) = 台账里 init 臂的 official.t1；T1(吸引子) = 14963@4 的 official.t1）
           · 某个 epoch 真的门禁过之后 ⇒ 改用**实测 T1** 外推后续 epoch
             （A2 轨迹的 T1 单调上升 ⇒ 越走越近线 ⇒ 后续 epoch 更值得门禁）；
        ③ **兜底**：若整条轨迹一个 epoch 都不满足 ①/② ⇒ 门禁「三轴余量最大」的那个 epoch
           （否则整条轨迹失读 ⇒ 拿不到任何 T1/T3 读数 ⇒ 轴白跑）。
    """
    run_trial = int(spec["trial"])
    arm_ids = [int(x) for x in (spec.get("pt2_ep_arms") or [])]
    epochs = [int(e) for e in (spec.get("epochs") or [])]
    p = spec["params"]
    if not arm_ids:
        return []
    b_nano, b_nano4, b_f1 = ns_mod.base_pick_thresholds()
    base_t1 = float(ns_mod.BASE_V2["t1"])
    n_ep = max(len(epochs), 1)

    t1_init = _pt2_t1_of(ns_mod, int(spec.get("pt2_init_trial") or 0))
    t1_attr = _pt2_t1_of(ns_mod, PT2_ATTRACTOR[0])
    if t1_init is None:
        # 🔴 台账读不到（并发写导致 JSON 撕裂）⇒ 用**实测的 init 缺口**兜底（保守）
        t1_init = base_t1 + PT2_INIT_T1_STEPS.get(
            int(spec.get("pt2_init_trial") or 0), -3000.0) * PT2_T1_STEP
    if t1_attr is None:
        t1_attr = base_t1 + 86.8 * PT2_T1_STEP

    def _pred_t1(idx: int, measured: float | None = None) -> float:
        """第 idx 个 epoch（1..n_ep）的 T1 预测（绝对值）。"""
        if measured is not None:
            rest = n_ep - idx
            if rest <= 0:
                return measured
            return measured + (t1_attr - measured) / rest
        if n_ep <= 1:
            return t1_init
        return t1_init + (t1_attr - t1_init) * (idx - 1) / (n_ep - 1)

    # ── ① 逐 epoch 登记（先全部登记 ⇒ 轨迹读数不丢）──
    led = ns_mod.load_ledger()
    for i, ep in enumerate(epochs, start=1):
        m = traj.get(str(ep))
        if not m or m.get("error"):
            continue
        arm_id = arm_ids[i - 1] if i - 1 < len(arm_ids) else arm_ids[-1]
        _link_ep_ckpt(run_trial, arm_id, ep)
        st = ns_mod.arm_state(led, arm_id)
        st["trial"] = arm_id
        st["params"] = p
        st["seed"] = spec.get("seed")
        st["desc"] = f"{spec.get('desc')} ep{ep}"
        st["channel"] = "PT2"
        st["trained"] = True
        st["polish"] = 1
        st["source_anchor"] = spec.get("source_anchor")
        st["pt2_run_trial"] = run_trial
        st["pt2_epoch"] = ep
        st["pt2_init"] = p.get("init")
        st["pt2_recipe"] = spec.get("pt2_recipe")
        st["pt2_traj"] = {str(k): dict(v) for k, v in traj.items() if isinstance(v, dict)}
        st["nanobench"] = {str(ep): dict(m)}
        picked = dict(m)
        picked.update({"epoch": ep, "epoch_key": str(ep), "ckpt": f"epoch{ep}.pt"})
        st["picked"] = picked
        st["picked_epoch"] = ep
        st["select_rule"] = "pt2_per_epoch"
        st["beat_to_pass_ref"] = "Baseline_v3.1(trial_1990055)"
    ns_mod.save_ledger(led)

    # ── ② 短路判定（三轴全过 / T1 近线 / 兜底）──
    cand = []
    for i, ep in enumerate(epochs, start=1):
        m = traj.get(str(ep))
        if not m or m.get("error"):
            continue
        three_ok = (m.get("nano", 0) >= b_nano and m.get("nano4", 0) >= b_nano4
                    and m.get("f1", 0) >= b_f1)
        gap = (base_t1 - _pred_t1(i)) / PT2_T1_STEP
        cand.append({"i": i, "ep": ep, "m": m, "three_ok": three_ok, "gap": gap})
    if not cand:
        return []
    force_set = set()
    if not any(c["three_ok"] or c["gap"] <= PT2_T1_NEAR_STEPS for c in cand):
        def _marg(c):
            return min((c["m"].get("nano", 0) - b_nano) / PT2_NANO_STEP,
                       (c["m"].get("nano4", 0) - b_nano4) / PT2_NANO4_STEP,
                       (c["m"].get("f1", 0) - b_f1) / PT2_F1_STEP)
        pick = max(cand, key=_marg)
        force_set = {pick["ep"]}
        log(f"[PT2 {run_trial}] 三轴全过/T1 近线都没有 ⇒ 兜底门禁 ep{pick['ep']}（三轴余量最大）")

    # ── ③ 逐 epoch 门禁（实测 T1 会反过来修正后续 epoch 的预测）──
    recs: list[dict] = []
    measured_t1 = None
    for c in cand:
        i, ep = c["i"], c["ep"]
        gap = c["gap"]
        if measured_t1 is not None:
            gap = (base_t1 - _pred_t1(i, measured=measured_t1)) / PT2_T1_STEP
        if not (c["three_ok"] or gap <= PT2_T1_NEAR_STEPS or ep in force_set):
            continue
        arm_id = arm_ids[i - 1] if i - 1 < len(arm_ids) else arm_ids[-1]
        _why = ("★ 三轴全过" if c["three_ok"]
                else (f"T1 近线（≤{PT2_T1_NEAR_STEPS} 阶）" if gap <= PT2_T1_NEAR_STEPS
                      else "兜底（三轴余量最大）"))
        log(f"[PT2 {run_trial}] ep{ep} ⇒ 官方门禁（{_why}；T1 缺口≈{gap:+.0f} 阶）")
        try:
            off = ns_mod.run_official_gate(run_trial, ep, gpu)
        except Exception as exc:  # noqa: BLE001
            off = {"error": str(exc)}
        led = ns_mod.load_ledger()
        st = ns_mod.arm_state(led, arm_id)
        st["official"] = off
        m = c["m"]
        st["six_ok"] = bool(
            off.get("t1") is not None and off.get("t3") is not None
            and m.get("nano", 0) >= b_nano
            and m.get("nano4", 0) >= b_nano4
            and m.get("f1", 0) >= b_f1
            and off.get("t1", 0) >= base_t1
            and off.get("t2_fp", 99) <= max(int(ns_mod.BASE_V2.get("t2_fp") or 0), 5)
            and off.get("t3", 0) >= float(ns_mod.BASE_V2["t3"]))
        ns_mod.save_ledger(led)
        if off.get("t1") is not None:
            measured_t1 = float(off["t1"])
        log(f"[PT2 {run_trial}] ep{ep} 门禁 = {json.dumps(off, ensure_ascii=False)[:300]}")
        recs.append({"trial": arm_id, "gpu": gpu, "channel": "PT2",
                     "run_trial": run_trial, "ep": ep,
                     "desc": st.get("desc"), "triad": c["three_ok"],
                     "nano": m.get("nano"), "nano4": m.get("nano4"), "f1": m.get("f1"),
                     "t1": off.get("t1"), "t2_fp": off.get("t2_fp"), "t3": off.get("t3"),
                     "six_ok": st.get("six_ok"), "official": off})
    log(f"[PT2 {run_trial}] 门禁完成：{len(recs)}/{len(cand)} 个 epoch 过了短路")
    return recs


def build_pt2_specs(args, ns_mod) -> list[dict]:
    """阶段1-E：**「nano 富集端出发」的 A2 抛光轴（PT2）** —— 2026-11-23 新增（全库唯一突破口）。

    🔴 **为什么必须有这条轴**（台账实测，不是推理）：
      · 全库 2,297 个已门禁臂里「六项全过 ∧ 至少一轴严格优」= **0 臂**
        （57 个六项全过臂**全是纯打平**）⇒ 常规微扫已经榨干；
      · 已有的 PT1 / L5 / L5B 三条轴**全部从「T1 富 / nano 贫」的供体出发抛光**
        （init = 15510@4 / 14963@4 一类，dNano ≈ −11…−36 阶）
        ⇒ 抛光只会把 nano 越推越低 ⇒ **结构上补不上 nano 那一轴**
        （实测：2106006 dNano −11 阶 / 14963 dNano −11 阶 —— 都只差 nano 一轴）；
      · 台账里另有一小撮 **「nano 富集」臂**（本轴 init 集）：dNano +23…+71 阶、t2_fp = 0，
        代价是 dT1 −1738…−6773 阶、dT3 −309…−1833 阶；
      · ⇒ **把方向反过来**：从 nano 富集端出发做 **A2 族抛光**
        （sw4.0 / thr12ms3 / maxside3.0 / mix w2.0 / fb2.75 / ow0.4 / mosaic0.15 /
          scale0.1 / sched5 / warmup1.5 / l2=0.001），
        让轨迹**从 nano 富集端一路走到 A2 吸引子**（14963@4：dT1 +86.8 阶 / dNano −11 阶 /
        dF1 +193 阶 / t2fp 4）⇒ **中途必然经过 (nano ≥ 0, T1 > 0) 的窗口**；
      · ⚠ **必须逐 epoch 采样 + 逐 epoch 门禁**（现在只 gate `best` epoch ⇒ 会整段错过穿越点）。

    ⚠ **臂号段独立（PT2_BASE = 2115000）**：2110000~2110020 已被占用 ⇒ 2115000 段零占用。
    ⚠ **每 run = 5 个 epoch = 5 个臂**（臂号连续）⇒ 120 臂预算 = 24 run。
    """
    out: list[dict] = []
    idx = 0
    _k = 0                                   # 只在真正产臂时推进（seed 口径与历史一致）
    for at, ae in PT2_INITS:
        init = f"{ROOT}/trials/trial_{at}/weights/epoch{ae}.pt"
        _have = os.path.isfile(init)
        if not _have:
            log(f"[PT2] nano 富集锚点 {at}@{ae} 权重不在盘 ⇒ 跳过"
                f"（臂号位仍保留 ⇒ **不平移 sid↔配方映射**）")
        for (lr, cp, swl) in PT2_RECIPES:
            sid = PT2_BASE + idx * PT2_EP_PER_RUN
            arm_ids = [sid + i for i in range(PT2_EP_PER_RUN)]
            idx += 1
            if not _have:
                continue
            p = base_params(init, 5, 5, 1.5)
            p.update(lr=lr, lr_floor=0.1, sw=4.0, swb=1.0, swl=swl, thr=12.0,
                     maxside=3.0, anchor_l2=0.001, anchor_ckpt=init,
                     cp_prob=cp, cp_max_px=12.0, cp_per_img=2, cp_feather_px=2.0)
            out.append({"trial": sid, "seed": 870000 + _k * 17, "params": p,
                        "epochs": [1, 2, 3, 4, 5], "channel": "PT2", "prio": 0,
                        # 🔴 `source_anchor` 是升轨「抛光家族种子关」的必需字段
                        "source_anchor": f"{at}@{ae}",
                        "pt2_ep_arms": arm_ids,
                        "pt2_init_trial": at, "pt2_init_epoch": ae,
                        "pt2_recipe": f"lr{lr}_cp{cp}_swl{swl}",
                        "desc": f"[PT2] nano{at}@{ae} lr{lr} cp{cp} swl{swl} "
                                f"sw4.0 swb1.0 thr12ms3 l2.001"})
            _k += 1
    budget = max(int(getattr(args, "pt2_arms", 120)), 0)
    if budget and len(out) * PT2_EP_PER_RUN > budget:
        out = out[:budget // PT2_EP_PER_RUN]
    if out:
        log(f"[阶段1-E] nano 富集抛光轴 {len(out)} run = {len(out) * PT2_EP_PER_RUN} 臂"
            f"（init {[a[0] for a in PT2_INITS]}，臂号 {PT2_BASE}~"
            f"{PT2_BASE + len(out) * PT2_EP_PER_RUN - 1}）")
    return out


# ─────────────────────────────────────────────────────────────────────────
# 🟢🟢🟢 2026-11-23 **PT3 规格：「T1 富供体 × PX 族配方」局部微扫轴**
# ─────────────────────────────────────────────────────────────────────────
# 🔴 为什么是这一枪（台账实测，数字权威）：
#   · 汤轴（权重汤）已被对偶证书证伪：全库 2358 个可读臂里「六项全过 ∧ 至少一轴严格优」= **0 臂**；
#   · 训练轴里已有「**只差一轴**」的近失臂（2106006：dNano −24.5 阶 / dNano4 +17 / dF1 +62.3 /
#     **dT1 +68.2** / dT3 +24.5 / t2fp 4 ⇒ **只差 nano 一轴**）⇒ 缺口是**结构性的**：
#     常规抛光只会把 nano 越推越低（PT1/L5/L5B 三轴全部如此）⇒ 补不上；
#   · 唯一的结构性例外 = **[PX] 族配方**：全库**唯一**「nano 与 T1 同涨」的旋钮
#     （1100275：init=98905@5, sw4.2, lr0.3, lrf0.3, ep2 ⇒ Δnano **+19.5 阶** ∧ ΔT1 **+66.3 阶**）
#     ⇒ **nano:T1 兑换比 ≈ 1:0.6**（其它配方是 1:8 ~ 1:200 ⇒ 拿 T1 换 nano 的坏交易）；
#   · ⇒ **PT3 = 把 PX 族配方施加到「T1 富供体」上** ⇒ 用供体已有的 T1 余量**买** nano。
#     本轴的成功判据 = 六项门禁（nano/nano4/f1/t1/t3 ≥ 基线 ∧ t2_fp ≤ max(基线 t2_fp, 5)=5）。
#
# ⚠ 本块**只读**台账（`.snap_ledger.json` / live 台账）与配方注册表；**不写**这两个文件
#   （注册表的写入走 `_pt3_bind_registry`，且只在真正开跑时执行 ⇒ `--plan-only` 零副作用）。
LOCAL_MNT_ROOT = "/home/manu/mnt"   # 本地 sshfs 挂载根（服务器 /mnt/data ↔ 本地 /home/manu/mnt/data）


def _local_mirror(path: str) -> str:
    """服务器绝对路径 → 本地挂载镜像路径（/mnt/x → /home/manu/mnt/x；/tmp/x 同理）。

    🔴 本地 sshfs 挂载把服务器的 `/mnt/data` 挂在 `/home/manu/mnt/data`、
       `/tmp/pycharm_project_*` 挂在 `/home/manu/mnt/pycharm_project_*`
       ⇒ 规则 = **去掉第一段目录名 + 挂到 LOCAL_MNT_ROOT 下**。
    """
    for prefix in ("/mnt/", "/tmp/"):
        if path.startswith(prefix):
            return LOCAL_MNT_ROOT + path[len(prefix) - 1:]
    return path


def _first_existing(*paths: str) -> str | None:
    """返回第一个**真实存在**的路径（服务器路径优先，其次本地镜像）。"""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _work_path(name: str) -> str:
    """WORK 目录下的文件（服务器路径优先，本地镜像兜底 ⇒ 本地也能做只读自检）。"""
    return (_first_existing(f"{WORK}/{name}", f"{_local_mirror(WORK)}/{name}")
            or f"{WORK}/{name}")


def _pt3_ckpt_path(trial: int, epoch: int) -> str:
    """供体权重路径（**服务器口径** ⇒ 训练用）。"""
    return f"{ROOT}/trials/trial_{trial}/weights/epoch{epoch}.pt"


def _pt3_ckpt_on_disk(trial: int, epoch: int) -> bool:
    """供体权重是否**实测在盘**（服务器路径或本地镜像任一命中即算在盘）。"""
    p = _pt3_ckpt_path(trial, epoch)
    return os.path.isfile(p) or os.path.isfile(_local_mirror(p))


_PT3_ARMS_CACHE: dict | None = None


def _pt3_ledger_arms() -> dict:
    """台账 arms 快照（**只读**）：live 台账优先，缺项用 `.snap_ledger.json` 补。"""
    global _PT3_ARMS_CACHE
    if _PT3_ARMS_CACHE is not None:
        return _PT3_ARMS_CACHE
    arms: dict = {}
    for name in ("nano_search_ledger.json", ".snap_ledger.json"):
        try:
            with open(_work_path(name), "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        for k, v in (data.get("arms") or {}).items():
            arms.setdefault(str(k), v)      # live 台账先入 ⇒ snap 只补缺项
    _PT3_ARMS_CACHE = arms
    return arms


def _pt3_registry() -> dict:
    """配方注册表（臂号 → 配方描述；**只读**）。"""
    try:
        with open(_work_path("arm_recipe_registry.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _pt3_own_arm_ids() -> set[int]:
    """**本轴自己**登记/执行的臂号（注册表配方前缀 "[PT3]"）⇒ 不算「他人占用」。

    🔴🔴 2026-11-23 13:2x 实测踩坑（重启即自锁死）：
      · PT3 首跑把 2116000~2119999 **全部**写进 `arm_recipe_registry.json`（配方前缀 "[PT3] "）；
      · 重启时 `_pt3_used_arm_ids()` 把这 4000 个 id 全判成「已占用」
        ⇒ `_pt3_alloc_base()` 扫不到任何连续空闲块 ⇒ 臂计划 PT3=0 ⇒ **整条 PT3 轴永久失效**。
      · 修法：把**本轴自己**的条目排除出「占用集」；段位 [PT3_BASE, PT3_RESERVED_FLOOR) 本就是 PT3 专属段。
    """
    own: set[int] = set()
    for k, v in _pt3_registry().items():
        # ⚠ 前缀是 "[PT3]" **和** "[PT3M]"（中间带族）两种 ⇒ 用 startswith("[PT3") 兜住
        if not str(v).startswith("[PT3"):
            continue
        try:
            own.add(int(str(k)))
        except Exception:  # noqa: BLE001
            pass
    return own


def _pt3_used_arm_ids() -> set[int]:
    """台账 + 配方注册表里**他人已占用**的臂号（数字臂号；排除本轴自己的条目）。"""
    own = _pt3_own_arm_ids()
    used: set[int] = set()
    for k in _pt3_ledger_arms().keys():
        try:
            sid = int(str(k))
        except Exception:  # noqa: BLE001
            continue
        if sid in own:
            continue
        used.add(sid)
    for k in _pt3_registry().keys():
        try:
            sid = int(str(k))
        except Exception:  # noqa: BLE001
            continue
        if sid in own:
            continue
        used.add(sid)
    return used


def _pt3_alloc_base(need: int) -> tuple[int, str, int]:
    """扫描台账 + 注册表 ⇒ 找起点 ≥ PT3_BASE 的**第一个连续空闲**臂号段。

    🔴 **绝不越界到他人段位**（硬规则）：
      · 允许区间 = **[PT3_BASE, PT3_RESERVED_FLOOR)** = `2116000~2119999`
        （服务器实测：`2120000~2120022` 已被他人占用、`2130000+` / `2140000+` 同
        ⇒ `2120000+` 属**他人段位** ⇒ 本轴**不碰 2120023+ 等后续空段**）；
      · 区间内优先取**第一块容量 ≥ need 的连续空闲段**；没有 ⇒ 取**第一块**空闲段
        并**按其真实容量下调**（调用方打日志说明实际可用臂数）。
      · 满格需求 = 20 供体 × 108 run × 2 臂 = **4320 臂**；本段只有 **4000** 个空闲 id
        ⇒ **4320 装不下 ⇒ 满格按实际段位下调为 4000 臂（2000 run）**。

    返回 (起点, 说明, 该段容量=可用臂数)。
    """
    used = _pt3_used_arm_ids()
    need = max(int(need), PT3_SEG_MIN_FREE)
    scan_limit = 200000       # 扫描上限（防「台账读不到 ⇒ used 为空」时的死循环）
    ceiling = min(PT3_RESERVED_FLOOR, PT3_BASE + scan_limit)
    blocks: list[tuple[int, int]] = []
    start = None
    for i in range(PT3_BASE, ceiling):
        if i in used:
            if start is not None:
                blocks.append((start, i - 1))
                start = None
        elif start is None:
            start = i
    if start is not None:
        blocks.append((start, ceiling - 1))
    if not blocks:
        return PT3_BASE, (f"⚠ 自 {PT3_BASE} 起至 {ceiling - 1} **无任何连续空闲臂号** ⇒ PT3 段不可用"), 0
    # 优先「第一块容量 ≥ need 的空闲段」；都没有 ⇒ 取第一块并按其容量下调
    pick = next((b for b in blocks if (b[1] - b[0] + 1) >= need), blocks[0])
    seg_start, seg_end = pick
    free = seg_end - seg_start + 1
    next_used = min((u for u in used if u > seg_end), default=None)
    if free >= need:
        note = (f"起点 {seg_start}（连续空闲段 {seg_start}~{seg_end}，容量 {free} 个 id ≥ 需求 {need}）"
                + (f"；下一占用段 = {next_used}（他人段位，**不越界**）" if next_used is not None else ""))
        return seg_start, note, free
    # 容量不足 ⇒ 按实际段位下调（**不跨到他人段位**）
    note = (f"起点 {seg_start}（连续空闲段 {seg_start}~{seg_end}，容量 **{free} 个 id** < 需求 {need} ⇒ "
            f"**满格按实际段位下调**为 {free} 臂（{free // PT3_EP_PER_RUN} run）")
    if next_used is not None:
        note += f"；下一占用段 = {next_used}（他人段位，**不越界**）"
    note += f"；允许区间上限 = {ceiling - 1}（`PT3_RESERVED_FLOOR` 之后属他人段位）"
    return seg_start, note, free


def _pt3_base_vals(ns_mod=None) -> dict:
    """基线读数（优先取 `nano_search.BASE_V2` ⇒ 与台账逐位一致；否则用内置常量）。"""
    b = {"nano": PT3_BASE_NANO, "nano4": PT3_BASE_NANO4, "f1": PT3_BASE_F1,
         "t1": PT3_BASE_T1, "t2_fp": PT3_BASE_T2FP, "t3": PT3_BASE_T3}
    bv = (getattr(ns_mod, "BASE_V2", None) or {}) if ns_mod is not None else {}
    for k in ("nano", "nano4", "f1", "t1", "t3"):
        if bv.get(k) is not None:
            b[k] = float(bv[k])
    if bv.get("t2_fp") is not None:
        b["t2_fp"] = int(bv["t2_fp"])
    return b


def _pt3_donor_reading(donor: tuple) -> dict:
    """供体的实测读数（台账优先；读不到 ⇒ 用阶数表换算成绝对值）。"""
    t, ep, prio, dn, dn4, df1, dt1, dt3, t2fp, note = donor
    out = {"trial": t, "epoch": ep, "prio": prio, "note": note,
           "d_nano": dn, "d_nano4": dn4, "d_f1": df1, "d_t1": dt1, "d_t3": dt3,
           "t2fp": t2fp, "nano": None, "nano4": None, "f1": None,
           "t1": None, "t3": None, "src": "阶数表(内置)"}
    st = _pt3_ledger_arms().get(str(t)) or {}
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    if pk.get("nano") is not None:
        out["nano"] = float(pk["nano"])
        out["nano4"] = float(pk.get("nano4") or 0.0)
        out["f1"] = float(pk.get("f1") or 0.0)
        out["src"] = "台账 picked"
    if of.get("t1") is not None:
        out["t1"] = float(of["t1"])
        out["t3"] = float(of.get("t3") or 0.0)
        out["t2fp"] = int(of.get("t2_fp") or t2fp)
        out["src"] = "台账 picked+official"
    # 官方 T1/T3 读不到 ⇒ 用阶数表换算（保守但可用）
    if out["t1"] is None:
        out["t1"] = PT3_BASE_T1 + dt1 * PT3_T1_STEP
    if out["t3"] is None:
        out["t3"] = PT3_BASE_T3 + dt3 * PT3_T3_STEP
    return out


def _pt3_donor_gap(donor: tuple) -> float:
    """供体的**总缺口 gap** = Σ max(0, −d)，over {nano, nano4, f1, t1, t3}（t2fp 不计入）。

    · gap 越小 = 离「六项全过」越近 = **越先跑**（2026-11-23 增量：供体表按 gap 升序重排）；
    · 优先级分档：**P1 = gap ≤ 100 ｜ P2 = 100 < gap ≤ 200 ｜ P3 = gap > 200**。
    """
    return float(sum(max(0.0, -float(x)) for x in donor[3:8]))


def _pt3_donor_order_label(donor: tuple) -> str:
    """供体的展示标签：`trial@epoch`（日志用）。"""
    return f"{donor[0]}@{donor[1]}"


def _pt3_recipe_plan() -> list[dict]:
    """**单个供体的完整配方序列** = 72 条主网格 + 36 条 PT3M 中间带 = 108 run。

    🔴 **列表顺序 = 执行顺序 = 臂号顺序**（供体 i 占用
       `PT3_BASE + i*216 ~ PT3_BASE + i*216 + 215`）：
      · 前 72 条 = `PT3_RECIPES`（sw↑ → cp↑ → lr↑ → lrf↑ 的**字典序**）；
      · 后 36 条 = `PT3M_RECIPES`（sw↑ → swl↑ → thr↑ → lr↑ 的字典序）。
    """
    out: list[dict] = []
    for (sw, cp, lr, lrf) in PT3_RECIPES:
        out.append({"band": "M0", "sw": sw, "cp": cp, "lr": lr, "lrf": lrf,
                    "swl": 1.0, "thr": 12.0})
    for (sw, swl, thr, lr) in PT3M_RECIPES:
        out.append({"band": "M1", "sw": sw, "cp": PT3M_CP, "lr": lr, "lrf": PT3M_LRF,
                    "swl": swl, "thr": thr})
    return out


def _pt3_lin_extrap(points: dict, idx: int):
    """线性插值/外推（points: {x: y}；x = epoch 序号，**0 = 供体起点**）。"""
    if not points:
        return None
    if idx in points:
        return float(points[idx])
    xs = sorted(points)
    if len(xs) == 1:
        return float(points[xs[0]])
    lo = [x for x in xs if x < idx]
    hi = [x for x in xs if x > idx]
    if lo and hi:
        x1, x2 = lo[-1], hi[0]
    elif lo:
        x1, x2 = lo[-2], lo[-1]
    else:
        x1, x2 = hi[0], hi[1]
    y1, y2 = float(points[x1]), float(points[x2])
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (idx - x1) / (x2 - x1)


def build_pt3_specs(args, ns_mod=None) -> list[dict]:
    """阶段1-F：**PT3「T1 富供体 × PX 族配方」局部微扫轴** —— 2026-11-23 新增。

    🔴 **臂号映射（必须稳定 ⇒ 重启后臂号↔配方一一对应）**：
      · `臂号 = 段起点 + run_index*2 + (epoch-1)`（**逐 epoch 连续**，每 run 2 臂）；
      · `run_index` 遍历 **所有供体 × 全部 108 个配方**（72 主网格 + 36 中间带；含不在盘的供体 ⇒ 臂号位保留、
        **不平移映射**）⇒ 供体缺失不会让后面的臂号漂移；
      · 供体 i 占用 `段起点 + i*216 ~ 段起点 + i*216 + 215`（108 run × 2 臂）
        ⇒ **中间带的臂号排在同一供体的主网格臂号之后**（不另开段）；
      · 段起点 = `_pt3_alloc_base()` 现算（默认 PT3_BASE；被占 ⇒ 向后找连续空闲段）。
    🔴 **预算**：`--pt3-arms` 单位是**臂**（不是 run）⇒ `run 数 = pt3_arms // 2`；
      **列表顺序 = 执行顺序**（供体按 gap 升序；同供体内 72 主网格（sw↑ → cp↑ → lr↑ → lrf↑）
      → 36 中间带（sw↑ → swl↑ → thr↑ → lr↑））。
    🔴 **段位容量是硬上限**：`2116000~2119999` 只有 **4000** 个连续空闲 id ⇒ 满格 4320 臂
      **装不下** ⇒ 实际按 **4000 臂（2000 run）** 下调，**绝不越界到 2120000+（他人段位）**。
    """
    budget = max(int(getattr(args, "pt3_arms", 0) or 0), 0)
    if budget <= 0:
        log(f"[阶段1-F] PT3 轴关闭（--pt3-arms=0）⇒ 不排任何臂")
        return []
    n_runs_total = len(PT3_DONORS) * PT3_RUNS_PER_DONOR
    n_arms_total = n_runs_total * PT3_EP_PER_RUN
    base, seg_note, capacity = _pt3_alloc_base(n_arms_total)
    log(f"[阶段1-F] PT3 臂号段：{seg_note}｜理论满格 {n_runs_total} run = {n_arms_total} 臂")
    # 🔴 段位容量是**硬上限**（不越界到他人段位）；预算 `--pt3-arms` 再在其下截断
    max_runs = capacity // PT3_EP_PER_RUN
    if max_runs < n_runs_total:
        log(f"[阶段1-F] ⚠ 段位容量 {capacity} 臂 < 理论满格 {n_arms_total} 臂 ⇒ "
            f"**满格按实际段位下调为 {capacity} 臂（{max_runs} run）**；"
            f"最后一个供体可能被截断（臂号仍连续、不重叠）")
    budget_runs = budget // PT3_EP_PER_RUN
    if budget_runs < max_runs:
        max_runs = budget_runs
    out: list[dict] = []
    run_index = 0
    _k = 0                                   # 只在真正产臂时推进（seed 位与臂号位分离）
    for donor in PT3_DONORS:
        t, ep = donor[0], donor[1]
        have = _pt3_ckpt_on_disk(t, ep)
        if not have:
            log(f"[PT3] 供体 {t}@{ep} 权重**不在盘** ⇒ 跳过"
                f"（臂号位保留 ⇒ **不平移 sid↔配方映射**）")
        for rc in _pt3_recipe_plan():
            sid = base + run_index * PT3_EP_PER_RUN
            run_index += 1
            if not have:
                continue
            if len(out) >= max_runs:
                break                       # 段位容量 / 预算用尽 ⇒ 停止（臂号仍连续）
            sw, cp, lr, lrf = rc["sw"], rc["cp"], rc["lr"], rc["lrf"]
            init = _pt3_ckpt_path(t, ep)
            p = base_params(init, 2, 2, 0.3)
            p.update(lr=lr, lr_floor=lrf, sw=sw, swl=rc["swl"], swb=1.0,
                     thr=rc["thr"], maxside=3.0, mix="w2.0", fb=2.75, ow=0.4,
                     mosaic=0.15, scale=0.10, anchor_l2=0.001, anchor_ckpt=init)
            if float(cp) > 0.0:
                # 🔴 cp=0 时**不加** cp flag（与现有 L5/PT1 口径一致：`train_cmd` 只在 cp>0 时加）
                p.update(cp_prob=cp, cp_max_px=12.0,
                         cp_per_img=2, cp_feather_px=2.0)
            if rc["band"] == "M1":
                desc = (f"[PT3M] t1donor{t}@{ep} sw{sw} swl{rc['swl']} thr{rc['thr']} cp{cp} "
                        f"lr{lr} lrf{lrf} swb1.0 maxside3.0 l2.001")
                recipe = f"M1_sw{sw}_swl{rc['swl']}_thr{rc['thr']}_cp{cp}_lr{lr}_lrf{lrf}"
            else:
                desc = (f"[PT3] t1donor{t}@{ep} sw{sw} cp{cp} lr{lr} lrf{lrf} "
                        f"swl{rc['swl']} swb1.0 thr{rc['thr']}ms3 l2.001")
                recipe = f"sw{sw}_cp{cp}_lr{lr}_lrf{lrf}"
            out.append({"trial": sid, "seed": 880000 + _k * 17, "params": p,
                        "epochs": [1, 2], "channel": "PT3", "prio": 0,
                        # 🔴 `source_anchor` 是升轨「抛光家族种子关」的必需字段
                        "source_anchor": f"{t}@{ep}",
                        "pt3_ep_arms": [sid, sid + 1],
                        "pt3_init_trial": t, "pt3_init_epoch": ep,
                        "pt3_band": rc["band"],
                        "pt3_recipe": recipe,
                        "desc": desc})
            _k += 1
        if len(out) >= max_runs:
            break
    if out:
        used_donors: list[str] = []
        for sp in out:
            lb = _pt3_donor_order_label((sp["pt3_init_trial"], sp["pt3_init_epoch"]))
            if lb not in used_donors:
                used_donors.append(lb)
        log(f"[阶段1-F] PT3 轴 {len(out)} run = {len(out) * PT3_EP_PER_RUN} 臂"
            f"（供体序 {used_donors}，"
            f"臂号 {out[0]['trial']}~{out[-1]['pt3_ep_arms'][-1]}）")
    return out


def _pt3_bind_registry(specs: list[dict]) -> list[dict]:
    """把 PT3 的臂号↔配方绑定写进配方注册表（**沿用既有唯一性机制**）。

    · 注册表里同号不同配方 ⇒ **宁可不跑**（否则「权重已存在 ⇒ 跳过训练」会把旧权重当新配方 ⇒ 假读数）；
    · 注册表里没有、但该臂号目录里已有权重 ⇒ 同样跳过（历史遗留臂号）。
    """
    reg = _pt3_registry()
    kept, dropped = [], []
    for sp in specs:
        desc = sp["desc"]
        bad = None
        for arm_id in (sp.get("pt3_ep_arms") or [sp["trial"]]):
            key = str(arm_id)
            old = reg.get(key)
            if old is not None and old != desc:
                bad = (key, old)
                break
            if old is None:
                tdir = f"{ROOT}/trials/trial_{arm_id}"
                try:
                    has_ck = os.path.isdir(f"{tdir}/weights") and any(
                        n.endswith(".pt") for n in os.listdir(f"{tdir}/weights"))
                except Exception:  # noqa: BLE001
                    has_ck = False
                if has_ck:
                    bad = (key, "历史遗留权重目录")
                    break
        if bad:
            dropped.append((sp["trial"], bad[0], bad[1]))
            continue
        kept.append(sp)
    for tr, key, old in dropped:
        log(f"[PT3 臂号冲突] run {tr} 的臂 {key} 已绑定「{old}」⇒ 跳过该 run")
    for sp in kept:
        for arm_id in (sp.get("pt3_ep_arms") or [sp["trial"]]):
            reg[str(arm_id)] = sp["desc"]
    try:
        with open(_work_path("arm_recipe_registry.json"), "w", encoding="utf-8") as fh:
            json.dump(reg, fh, ensure_ascii=False, indent=1)
    except Exception as exc:  # noqa: BLE001
        log(f"[PT3] 配方注册表写失败：{exc}")
    return kept


def _pt3_handle(spec, traj, gpu, ns_mod) -> list[dict]:
    """PT3 run：**逐 epoch 登记为臂** + **逐 epoch 官方门禁**（PT3 专用短路规则）。

    🔴 **短路规则**（门禁很贵 ⇒ 先筛后门禁；**与 PT2 的阈值不同**）：
      (A) **四轴近过线**：nano ≥ 基线−1 量子 ∧ nano4 ≥ 基线−1 量子 ∧ f1 ≥ 基线−1 量子
                           ∧ t3 ≥ 基线−1 量子；
      (B) **T1 近线**：该 epoch 的 t1 ≥ 基线−5 量子；
      (C) **兜底**：整条 run 没有任何 epoch 满足 (A)/(B) ⇒ 对「**归一化缺口最小**」的 epoch
          跑一次门禁（归一化缺口 = 四轴缺口各自除以自己的量子步长后求和）。
      · 比较一律用 **`>=`** 语义（基线比较带容差 `PT3_TOL`），**不用**严格 `>`；
      · `six_ok` 的 t2_fp 上限 = **max(基线 t2_fp, 5)**。

    🔴 **门禁前读不到 t1/t3 怎么办**（绝不崩）：
      ① 该 epoch 自己已门禁过 ⇒ 用实测值；
      ② 同 run 其它 epoch 的实测值（+ 供体起点 epoch0）⇒ **线性外推**；
      ③ 仍读不到 ⇒ **供体臂的实测 T1 + 保守兜底值**（`PT3_T1_FALLBACK_DROP`）。
    """
    run_trial = int(spec["trial"])
    arm_ids = [int(x) for x in (spec.get("pt3_ep_arms") or [])]
    epochs = [int(e) for e in (spec.get("epochs") or [])]
    p = spec["params"]
    if not arm_ids:
        return []
    base = _pt3_base_vals(ns_mod)
    b_nano, b_nano4, b_f1 = base["nano"], base["nano4"], base["f1"]
    base_t1, base_t3 = base["t1"], base["t3"]
    t2fp_max = max(int(base["t2_fp"]), 5)
    donor = PT3_DONOR_BY_KEY.get((int(spec.get("pt3_init_trial") or 0),
                                  int(spec.get("pt3_init_epoch") or 0)))
    donor_t1 = donor_t3 = None
    if donor is not None:
        _rd = _pt3_donor_reading(donor)
        donor_t1 = _rd.get("t1")
        donor_t3 = _rd.get("t3")

    def _pred_t1(idx: int, measured: dict):
        """第 idx 个 epoch（1..n_ep）的 T1 预测（绝对值；链：实测 → 线性外推 → 供体+保守兜底）。"""
        if idx in measured:
            return float(measured[idx])
        pts: dict = {}
        if donor_t1 is not None:
            pts[0] = float(donor_t1)          # 供体 = 这条 run 的 epoch0
        for k, v in measured.items():
            if k != idx:
                pts[int(k)] = float(v)
        if len(pts) >= 2:
            v = _pt3_lin_extrap(pts, idx)
            if v is not None:
                return v
        if donor_t1 is not None:
            return float(donor_t1) - PT3_T1_FALLBACK_DROP
        return base_t1 - PT3_T1_FALLBACK_DROP

    def _pred_t3(idx: int, measured: dict):
        """T3 预测（与 T1 同链；门禁短路时 t3 是哨兵 −1.0 ⇒ 不入 points）。"""
        if idx in measured:
            return float(measured[idx])
        pts: dict = {}
        if donor_t3 is not None:
            pts[0] = float(donor_t3)
        for k, v in measured.items():
            if k != idx:
                pts[int(k)] = float(v)
        if len(pts) >= 2:
            v = _pt3_lin_extrap(pts, idx)
            if v is not None:
                return v
        if donor_t3 is not None:
            return float(donor_t3) - PT3_T3_FALLBACK_DROP
        return base_t3 - PT3_T3_FALLBACK_DROP

    def _rule_a(m: dict, t3_pred) -> bool:
        """(A) 四轴近过线（nano / nano4 / f1 / t3，各容差 1 量子）。"""
        if t3_pred is None:
            return False
        return bool(
            float(m.get("nano", 0)) >= b_nano - PT3_NEAR_QUANTA * PT3_NANO_STEP - PT3_TOL
            and float(m.get("nano4", 0)) >= b_nano4 - PT3_NEAR_QUANTA * PT3_NANO4_STEP - PT3_TOL
            and float(m.get("f1", 0)) >= b_f1 - PT3_NEAR_QUANTA * PT3_F1_STEP - PT3_TOL
            and float(t3_pred) >= base_t3 - PT3_NEAR_QUANTA * PT3_T3_STEP - PT3_TOL)

    def _rule_b(t1_pred) -> bool:
        """(B) T1 近线（容差 5 量子）。"""
        if t1_pred is None:
            return False
        return bool(float(t1_pred) >= base_t1 - PT3_T1_NEAR_QUANTA * PT3_T1_STEP - PT3_TOL)

    def _norm_gap(m: dict, t3_pred) -> float:
        """归一化缺口 = 四轴缺口各除以自己的量子步长后求和（越小越接近过线）。"""
        g = 0.0
        for key, bv, step in (("nano", b_nano, PT3_NANO_STEP),
                              ("nano4", b_nano4, PT3_NANO4_STEP),
                              ("f1", b_f1, PT3_F1_STEP)):
            v = m.get(key)
            g += ((bv - float(v)) / step) if v is not None else 1e6
        g += ((base_t3 - float(t3_pred)) / PT3_T3_STEP) if t3_pred is not None else 1e6
        return g

    # ── ① 逐 epoch 登记（先全部登记 ⇒ 轨迹读数不丢）──
    led = ns_mod.load_ledger()
    for i, ep in enumerate(epochs, start=1):
        m = traj.get(str(ep))
        if not m or m.get("error"):
            continue
        arm_id = arm_ids[i - 1] if i - 1 < len(arm_ids) else arm_ids[-1]
        _link_ep_ckpt(run_trial, arm_id, ep)
        st = ns_mod.arm_state(led, arm_id)
        st["trial"] = arm_id
        st["params"] = p
        st["seed"] = spec.get("seed")
        st["desc"] = f"{spec.get('desc')} ep{ep}"
        st["channel"] = "PT3"
        st["trained"] = True
        st["polish"] = 1
        st["source_anchor"] = spec.get("source_anchor")
        st["pt3_run_trial"] = run_trial
        st["pt3_epoch"] = ep
        st["pt3_init"] = p.get("init")
        st["pt3_recipe"] = spec.get("pt3_recipe")
        st["pt3_traj"] = {str(k): dict(v) for k, v in traj.items() if isinstance(v, dict)}
        st["nanobench"] = {str(ep): dict(m)}
        picked = dict(m)
        picked.update({"epoch": ep, "epoch_key": str(ep), "ckpt": f"epoch{ep}.pt"})
        st["picked"] = picked
        st["picked_epoch"] = ep
        st["select_rule"] = "pt3_per_epoch"
        st["beat_to_pass_ref"] = "Baseline_v3.1(trial_1990055)"
        # 🔴 家族键（与 `family_key.polish_family_key` 的兜底口径逐字一致 ⇒ 派生值相同）
        st["family_key"] = ("polish:lr=%s|l2=%s|fb=%s|ep=%s"
                            % (p.get("lr"), p.get("anchor_l2"),
                               p.get("freeze_bn", 0), p.get("epochs")))
    ns_mod.save_ledger(led)

    # ── ② 短路判定（(A) 四轴近过线 / (B) T1 近线 / (C) 兜底）──
    cand = []
    for i, ep in enumerate(epochs, start=1):
        m = traj.get(str(ep))
        if not m or m.get("error"):
            continue
        t1_pred = _pred_t1(i, {})
        t3_pred = _pred_t3(i, {})
        cand.append({"i": i, "ep": ep, "m": m, "t1": t1_pred, "t3": t3_pred,
                     "a": _rule_a(m, t3_pred), "b": _rule_b(t1_pred),
                     "gap": _norm_gap(m, t3_pred)})
    if not cand:
        return []
    force_set = set()
    if not any(c["a"] or c["b"] for c in cand):
        pick = min(cand, key=lambda c: c["gap"])
        force_set = {pick["ep"]}
        log(f"[PT3 {run_trial}] 四轴近线/T1 近线都没有 ⇒ 兜底门禁 ep{pick['ep']}"
            f"（归一化缺口最小 {pick['gap']:.1f}）")

    # ── ③ 逐 epoch 门禁（实测 T1/T3 会反过来修正后续 epoch 的预测）──
    recs: list[dict] = []
    measured_t1: dict = {}
    measured_t3: dict = {}
    for c in cand:
        i, ep = c["i"], c["ep"]
        t1_pred = _pred_t1(i, measured_t1)
        t3_pred = _pred_t3(i, measured_t3)
        a_ok = _rule_a(c["m"], t3_pred)
        b_ok = _rule_b(t1_pred)
        if not (a_ok or b_ok or ep in force_set):
            continue
        arm_id = arm_ids[i - 1] if i - 1 < len(arm_ids) else arm_ids[-1]
        _why = ("★(A) 四轴近过线" if a_ok
                else ("(B) T1 近线" if b_ok else "(C) 兜底（归一化缺口最小）"))
        log(f"[PT3 {run_trial}] ep{ep} ⇒ 官方门禁（{_why}；"
            f"T1 预测={t1_pred:.7f}（缺口 {(base_t1 - t1_pred) / PT3_T1_STEP:+.0f} 阶）"
            f" / T3 预测={t3_pred:.7f}）")
        try:
            off = ns_mod.run_official_gate(run_trial, ep, gpu)
        except Exception as exc:  # noqa: BLE001
            off = {"error": str(exc)}
        led = ns_mod.load_ledger()
        st = ns_mod.arm_state(led, arm_id)
        st["official"] = off
        m = c["m"]
        st["six_ok"] = bool(
            off.get("t1") is not None and off.get("t3") is not None
            and float(m.get("nano", 0)) >= b_nano
            and float(m.get("nano4", 0)) >= b_nano4
            and float(m.get("f1", 0)) >= b_f1
            and float(off.get("t1", 0)) >= base_t1
            and int(off.get("t2_fp", 99)) <= t2fp_max
            and float(off.get("t3", 0)) >= base_t3)
        ns_mod.save_ledger(led)
        # 🔴 门禁的 T1 短路会让 T2/T3 不评（哨兵 −1.0）⇒ 只把**正数**读数写进 points
        if off.get("t1") is not None and float(off["t1"]) > 0:
            measured_t1[i] = float(off["t1"])
        if off.get("t3") is not None and float(off["t3"]) > 0:
            measured_t3[i] = float(off["t3"])
        log(f"[PT3 {run_trial}] ep{ep} 门禁 = {json.dumps(off, ensure_ascii=False)[:300]}")
        recs.append({"trial": arm_id, "gpu": gpu, "channel": "PT3",
                     "run_trial": run_trial, "ep": ep,
                     "desc": st.get("desc"), "triad": bool(a_ok or b_ok),
                     "nano": m.get("nano"), "nano4": m.get("nano4"), "f1": m.get("f1"),
                     "t1": off.get("t1"), "t2_fp": off.get("t2_fp"), "t3": off.get("t3"),
                     "six_ok": st.get("six_ok"), "official": off})
    log(f"[PT3 {run_trial}] 门禁完成：{len(recs)}/{len(cand)} 个 epoch 过了短路")
    return recs


def _pt3_print_plan(specs: list[dict], args) -> None:
    """`--plan-only`：打印**完整臂计划**（臂号 / 供体 / 配方 / 预测说明）后直接退出。

    🔴 本函数**只读**：不训练、不跑门禁、不写台账、不写配方注册表。
    """
    print("", flush=True)
    print("=" * 108, flush=True)
    print("PT3 计划干跑（--plan-only）：「T1 富供体 × PX 族配方」局部微扫轴", flush=True)
    print("=" * 108, flush=True)
    used = _pt3_used_arm_ids()
    # ---- 供体表（在盘实测）----
    donors_on, donors_off = [], []
    for d in PT3_DONORS:
        (donors_on if _pt3_ckpt_on_disk(d[0], d[1]) else donors_off).append(d)
    print(f"[供体表] 共 {len(PT3_DONORS)} 个供体：在盘 {len(donors_on)} 个 / 不在盘 {len(donors_off)} 个",
          flush=True)
    print(f"{'供体(trial@ep)':>18} {'prio':>4} {'在盘':>5} {'dNano':>8} {'dNano4':>8} {'dF1':>9} "
          f"{'dT1':>9} {'dT3':>8} {'t2fp':>5} {'gap':>8}  备注", flush=True)
    for d in PT3_DONORS:
        on = "是" if _pt3_ckpt_on_disk(d[0], d[1]) else "**否**"
        print(f"{str(d[0]) + '@' + str(d[1]):>18} {d[2]:>4} {on:>5} {d[3]:>+8.1f} {d[4]:>+8.1f} "
              f"{d[5]:>+9.1f} {d[6]:>+9.1f} {d[7]:>+8.1f} {d[8]:>5} {_pt3_donor_gap(d):>8.1f}  {d[9]}",
              flush=True)
    if donors_off:
        print(f"[供体表] ⚠ 不在盘的供体（**跳过但保留臂号位**）："
              f"{[(d[0], d[1]) for d in donors_off]}", flush=True)
    print(f"[供体序] gap 升序（gap = Σ max(0,−d)，over nano/nano4/f1/t1/t3；t2fp 不计入）："
          f"{[_pt3_donor_order_label(d) for d in PT3_DONORS]}", flush=True)
    # ---- 配方网格 ----
    print(f"[配方网格] sw{PT3_SW_GRID} × cp{PT3_CP_GRID} × lr{PT3_LR_GRID} × lrf{PT3_LRF_GRID} "
          f"= {len(PT3_RECIPES)} run/供体（M0 主网格，字典序展开）；每 run {PT3_EP_PER_RUN} epoch ⇒ "
          f"{PT3_EP_PER_RUN} 臂/run", flush=True)
    print(f"[配方网格 M1 中间带] sw{PT3M_SW_GRID} × swl{PT3M_SWL_GRID} × thr{PT3M_THR_GRID} × "
          f"lr{PT3M_LR_GRID} = {len(PT3M_RECIPES)} run/供体（swl↑ → thr↑ → lr↑ 字典序）",
          flush=True)
    print(f"[固定项] M0：swl=1.0 swb=1.0 thr=12.0 maxside=3.0 mix=w2.0 fb=2.75 ow=0.4 "
          f"mosaic=0.15 scale=0.1 anchor_l2=0.001 epochs=2 sched=2 warmup=0.3", flush=True)
    print(f"[固定项] M1：swb=1.0 maxside=3.0 mix=w2.0 fb=2.75 ow=0.4 mosaic=0.15 scale=0.1 "
          f"anchor_l2=0.001 epochs=2 sched=2 warmup=0.3 lrf={PT3M_LRF} cp={PT3M_CP}", flush=True)
    print(f"[臂布局] 每供体 {PT3_RUNS_PER_DONOR} run（72 M0 + 36 M1）= {PT3_ARMS_PER_DONOR} 臂；"
          f"供体 i 占用 `段起点 + {PT3_ARMS_PER_DONOR}*i ~ +{PT3_ARMS_PER_DONOR}*i + "
          f"{PT3_ARMS_PER_DONOR - 1}`（**中间带排在本供体主网格之后，不另开段**）", flush=True)
    # ---- 臂计划 ----
    arm_ids = []
    for sp in specs:
        arm_ids.extend(int(x) for x in (sp.get("pt3_ep_arms") or [sp["trial"]]))
    print(f"[臂计划] run {len(specs)} 个 / 臂 {len(arm_ids)} 个；"
          f"臂号 {arm_ids[0] if arm_ids else '-'}~{arm_ids[-1] if arm_ids else '-'}", flush=True)
    for sp in specs:
        rd = _pt3_donor_reading(PT3_DONOR_BY_KEY[(sp["pt3_init_trial"], sp["pt3_init_epoch"])])
        # 配方四元组直接从 params 取（pt3_recipe = swX_cpY_lrZ_lrfW，仅作日志用）
        _sw = sp["params"]["sw"]
        _cp = float(sp["params"].get("cp_prob") or 0.0)
        _lr = float(sp["params"]["lr"])
        _lrf = float(sp["params"]["lr_floor"])
        _swl = sp["params"].get("swl")
        _thr = sp["params"].get("thr")
        _band = sp.get("pt3_band") or "M0"
        gap_nano = max(0.0, -float(rd["d_nano"]))
        est_nano = PT3_PX_NANO_STEPS * (_lr / PT3_PX_LR)      # 仅按 lr 线性缩放 PX 单点（下界）
        est_t1 = PT3_PX_T1_STEPS * (_lr / PT3_PX_LR)
        t1_after = float(rd["d_t1"]) + est_t1
        near_b = (rd["t1"] - PT3_T1_FALLBACK_DROP
                  >= PT3_BASE_T1 - PT3_T1_NEAR_QUANTA * PT3_T1_STEP - PT3_TOL)
        print(f"  臂 {sp['pt3_ep_arms'][0]}（ep1）/ {sp['pt3_ep_arms'][1]}（ep2）"
              f" | 供体 {sp['pt3_init_trial']}@{sp['pt3_init_epoch']}"
              f"（{rd['prio']}, dNano {rd['d_nano']:+.1f} 阶 / dT1 {rd['d_t1']:+.1f} 阶）"
              f" | [{_band}] 配方 sw{_sw} swl{_swl} thr{_thr} cp{_cp} lr{_lr} lrf{_lrf}"
              f" | 预测：nano 缺口 {gap_nano:.1f} 阶；PX 先验(lr 线性缩放) "
              f"Δnano {est_nano:+.1f} 阶 / ΔT1 {est_t1:+.1f} 阶 ⇒ T1 落点 {t1_after:+.1f} 阶"
              f"；(B)T1 近线={'是' if near_b else '否'}"
              f"（Δnano/ΔT1 未建模 sw·cp 增益 ⇒ 打印值为下界）", flush=True)
    # ---- 自检 ----
    print("", flush=True)
    print("[自检]", flush=True)
    dup = sorted({i for i in arm_ids if arm_ids.count(i) > 1})
    print(f"  · 臂数 = {len(arm_ids)}｜run 数 = {len(specs)}｜本计划用到的供体数 = "
          f"{len({(sp['pt3_init_trial'], sp['pt3_init_epoch']) for sp in specs})}"
          f"｜供体表总数 = {len(PT3_DONORS)}（在盘 {len(donors_on)} / 不在盘 {len(donors_off)}）",
          flush=True)
    print(f"  · 臂号零重复：{'✅ 通过（0 重复）' if not dup else '❌ 失败 ' + str(dup[:20])}", flush=True)
    conflict = sorted(set(arm_ids) & used)
    print(f"  · 与台账/注册表零冲突：{'✅ 通过（0 冲突）' if not conflict else '❌ 失败 ' + str(conflict[:20])}",
          flush=True)
    # ---- 每个供体的臂号连续性（供体 i ⇒ i*216 起的连续段）+ 配方族分布 ----
    per_donor: dict[tuple, list[int]] = {}
    for sp in specs:
        per_donor.setdefault((sp["pt3_init_trial"], sp["pt3_init_epoch"]), []).extend(
            int(x) for x in (sp.get("pt3_ep_arms") or [sp["trial"]]))
    bad_cont = []
    for (t, ep), ids in per_donor.items():
        ids_sorted = sorted(ids)
        if ids_sorted != list(range(ids_sorted[0], ids_sorted[0] + len(ids_sorted))):
            bad_cont.append(f"{t}@{ep}")
    band_cnt: dict[str, int] = {}
    for sp in specs:
        b = sp.get("pt3_band") or "M0"
        band_cnt[b] = band_cnt.get(b, 0) + 1
    first_key = list(per_donor)[0] if per_donor else None
    print(f"  · 每供体臂号连续且不重叠：{'✅ 通过' if not bad_cont else '❌ 失败 ' + str(bad_cont)}"
          f"（供体数 {len(per_donor)}；首个 {first_key or '-'} ⇒ 臂号 "
          f"{min(per_donor[first_key]) if first_key else '-'}~"
          f"{max(per_donor[first_key]) if first_key else '-'}）", flush=True)
    print(f"  · 配方族分布：M0 主网格 {band_cnt.get('M0', 0)} run / M1 中间带 "
          f"{band_cnt.get('M1', 0)} run（每供体 72 + 36）", flush=True)
    _full_arms = len(PT3_DONORS) * PT3_RUNS_PER_DONOR * PT3_EP_PER_RUN
    base_now, seg_note_now, cap_now = _pt3_alloc_base(_full_arms)
    print(f"  · 臂号段：起点 {base_now}｜可用连续空闲 {cap_now} 臂｜满格需求 {_full_arms} 臂 ⇒ "
          f"{'装得下' if cap_now >= _full_arms else '**装不下 ⇒ 按实际段位下调**'}"
          f"（{seg_note_now}）", flush=True)
    print(f"  · 本计划供体执行序："
          f"{[_pt3_donor_order_label((sp['pt3_init_trial'], sp['pt3_init_epoch'])) for sp in specs[:3]]}"
          f"…（共 {len(per_donor)} 个供体进入本计划）", flush=True)
    print(f"  · 台账已占用臂号数 = {len(used)}｜注册表绑定数 = {len(_pt3_registry())}", flush=True)
    print("  · 本次干跑**未训练、未门禁、未写任何文件**。", flush=True)
    print("=" * 108, flush=True)


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
    ap.add_argument("--pt2-arms", type=int, default=120,
                    help="阶段1-E：nano 富集端 A2 抛光轴（PT2）臂数"
                         "（每 run 5 epoch ⇒ 每臂 1 epoch；0 = 不跑）")
    ap.add_argument("--pt3-arms", type=int, default=0,
                    help="阶段1-F：PT3「T1 富供体 × PX 族配方」局部微扫轴臂数"
                         "（单位=**臂**；每 run 2 epoch ⇒ 2 臂/run；"
                         "每供体 108 run（72 主网格 + 36 中间带）= 216 臂；"
                         "满格 20 供体 = 4320 臂，受臂号段容量限制会自动下调；0 = 不跑）")
    ap.add_argument("--plan-only", action="store_true",
                    help="只打印完整臂计划（臂号/供体/配方/预测说明）后**直接退出**："
                         "不训练、不跑门禁、不写台账、不写配方注册表（零副作用自检）")
    ap.add_argument("--lottery-arms", type=int, default=40,
                    help="阶段4：每轮中签扫描最多排多少臂")
    ap.add_argument("--lottery-target-n", type=int, default=6,
                    help="阶段4：每个家族的目标成员数（n≥4 才够升轨关）")
    ap.add_argument("--lottery-min-margin", type=float, default=-0.35,
                    help="阶段4：只有边际 ≥ 该值的配方才值得买彩票")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--report", default=f"{WORK}/night_train_axis_report.json")
    args = ap.parse_args()

    # 🟢🟢 2026-11-23：**`--plan-only` 只读模式**（上线前自检用）
    #   · 打印完整臂计划（臂号 / 供体 / 配方 / 预测说明）后**直接退出**；
    #   · **不训练、不跑门禁、不写台账、不写配方注册表** ⇒ 零副作用、零 GPU 作业；
    #   · 放在训练栈 import **之前** ⇒ 本机没有 `/tmp/pycharm_project_*` 也能跑通。
    if args.plan_only:
        try:
            _pt3_specs = build_pt3_specs(args, None)
        except Exception as exc:  # noqa: BLE001
            log(f"PT3 计划构造失败：{exc}")
            _pt3_specs = []
        _pt3_print_plan(_pt3_specs, args)
        log("=== --plan-only：计划已打印 ⇒ 直接退出（未训练、未门禁、未写任何文件）===")
        return

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
    pt2_specs = []
    try:
        pt2_specs = build_pt2_specs(args, NS)
    except Exception as exc:  # noqa: BLE001
        log(f"nano 富集抛光轴构造失败（不影响其它阶段）：{exc}")
        pt2_specs = []
    # 🟢🟢 2026-11-23：**PT3 段「T1 富供体 × PX 族配方」局部微扫轴**
    #   · 先绑配方注册表（沿用既有唯一性机制：同号不同配方 ⇒ 宁可不跑）⇒ 再做幂等过滤。
    pt3_specs = []
    try:
        pt3_specs = _pt3_bind_registry(build_pt3_specs(args, NS))
    except Exception as exc:  # noqa: BLE001
        log(f"PT3 轴构造失败（不影响其它阶段）：{exc}")
        pt3_specs = []
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

    # 🔴 幂等（PT3）：run 的 **ep1 臂**已有 picked.nano **且** official.t1 ⇒ 整条 run 跳过；
    #   ⚠ 只有 picked、official.t1 缺失的 run **不跳过** ⇒ 要把 T1 读数补上（否则供体池永远缺 T1）。
    _pt3_keep = []
    _pt3_skip_done = 0
    _pt3_keep_no_t1 = 0
    for sp in pt3_specs:
        _st = _a0.get(str(sp["trial"])) or {}
        _pk = (_st.get("picked") or {}).get("nano")
        _t1 = (_st.get("official") or {}).get("t1")
        if _pk is not None and _t1 is not None:
            _pt3_skip_done += 1
            continue
        if _pk is not None:
            _pt3_keep_no_t1 += 1
        _pt3_keep.append(sp)
    if _pt3_skip_done or _pt3_keep_no_t1:
        log(f"[幂等] PT3 轴：已完成(读数+T1)跳过 {_pt3_skip_done} 个 run；"
            f"有读数但缺官方 T1 ⇒ 保留补测 {_pt3_keep_no_t1} 个 run")
    pt3_specs = _pt3_keep
    # 🔴 幂等：PT2 run 若已登记过 epoch 读数 ⇒ 不重复烧算力（重启续跑）
    _pt2_keep = []
    for sp in pt2_specs:
        _st = _a0.get(str(sp["trial"])) or {}
        if (_st.get("picked") or {}).get("nano") is not None:
            continue                        # 已有读数 ⇒ 不重复烧算力
        _pt2_keep.append(sp)
    if len(_pt2_keep) != len(pt2_specs):
        log(f"[幂等] PT2 轴已有读数的 run {len(pt2_specs) - len(_pt2_keep)} 个 ⇒ 跳过")
    pt2_specs = _pt2_keep
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
    # 🟢🟢 2026-11-23：**PT2（nano 富集端抛光）插到 PT1/L5/L5B 之前**
    #   理由：PT1/L5/L5B 三轴都从「T1 富 / nano 贫」的供体出发 ⇒ 结构上补不上 nano 那一轴；
    #   PT2 是**全库唯一**还有结构性希望的通道（从 nano 富集端反向抛光 ⇒ 轨迹穿过可行区）。
    # 🟢🟢 2026-11-23：**PT3 插到 PT2 之前**（PT3 的先验更强 ⇒ [PX] 族是全库唯一 nano&T1 同涨的旋钮）
    specs = lot_specs + pt3_specs + pt2_specs + pt1_specs + l5_specs + l5b_specs + cp_specs + specs
    log(f"臂计划 {len(specs)} 条（LOT={len(lot_specs)} / "
        f"PT3={len(pt3_specs)}（{len(pt3_specs) * PT3_EP_PER_RUN} 臂）/ "
        f"PT2={len(pt2_specs)}（{len(pt2_specs) * PT2_EP_PER_RUN} 臂）/ "
        f"PT1={len(pt1_specs)} / L5={len(l5_specs)} / "
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
            if spec["channel"] in ("P", "PT1", "L5", "L5B", "PT2", "PT3"):
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
            # 🔴🔴 2026-11-23：PT2 = **逐 epoch 各自成臂** ⇒ 不能用 `best` 过滤
            #   （`best` 会整段错过「轨迹穿越可行区」的那个 epoch ⇒ 本轴的全部意义就没了）
            _is_pt2 = (spec.get("channel") == "PT2")
            # 🟢🟢 2026-11-23：PT3 同理 = **逐 epoch 各自成臂**（臂号 = run 首臂号 + ep−1）
            _is_pt3 = (spec.get("channel") == "PT3")
            if best and not (_is_pt2 or _is_pt3):
                st["picked"] = best
                st["picked_epoch"] = best["epoch"]
                st["select_rule"] = how
                st["beat_to_pass_ref"] = "Baseline_v3.1(trial_1990055)"
            NS.save_ledger(led)
            # ---- 三轴全过才跑官方门禁（贵）----
            # 🔴🔴 2026-11-23 真 bug 修复：triad 判据 `>` ⇒ `>=`
            #   权威口径（`six_criteria_audit.judge`）是 **nano/nano4/f1 ≥ 基线**；
            #   老代码用严格 `>` ⇒ **恰好等于基线**的候选臂被误杀（漏门禁 ⇒ 永远量不出 T1）。
            triad = bool(best and best.get("nano", 0) >= NS.BASE_V2["nano"]
                         and best.get("nano4", 0) >= NS.BASE_V2["nano4"]
                         and best.get("f1", 0) >= NS.BASE_V2["f1"])
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
            if _is_pt2:
                # 🔴🔴🔴 2026-11-23：**PT2 逐 epoch 登记 + 逐 epoch 官方门禁**（带短路）
                #   · 每个 epoch 单独成臂（臂号 = run 首臂号 + ep−1）⇒ 轨迹上每个采样点都是候选臂；
                #   · 门禁**不按 best 过滤** ⇒ 不会整段错过穿越点；
                #   · 门禁很贵（30~90 s/臂）⇒ 先过「三轴全过 / T1 近线 ≤200 阶」短路
                #     （详见 `_pt2_handle` 的注释块）。
                _pt2_recs = _pt2_handle(spec, traj, gpu, NS)
                with dlock:
                    done.extend(_pt2_recs)
            elif _is_pt3:
                # 🔴🔴🔴 2026-11-23：**PT3 逐 epoch 登记 + 逐 epoch 官方门禁**（PT3 专用短路）
                #   · 每个 epoch 单独成臂（臂号 = PT3_BASE + run_index*2 + ep−1）；
                #   · 门禁**不按 best 过滤** ⇒ 不会整段错过穿越点；
                #   · 短路规则 = (A) 四轴近过线 / (B) T1 近线(5 量子) / (C) 兜底（归一化缺口最小）
                #     （**与 PT2 的阈值不同**，详见 `_pt3_handle` 的注释块）。
                _pt3_recs = _pt3_handle(spec, traj, gpu, NS)
                with dlock:
                    done.extend(_pt3_recs)
            elif (triad or near or _force_gate) and best:
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
                # 🔴 six_ok 必须用**六项判据**（与 six_criteria_audit.judge 同口径）：
                #   nano/nano4/f1 **≥**基线 + t1 ≥ 基线 + t2_fp ≤ max(基线,5) + t3 ≥ 基线
                #   （昨夜误用门禁自带的 official_ok：t2_fp≤36 ⇒ 把 t2_fp=3 的臂误标 six_ok）
                # 🔴🔴 2026-11-23：`>` ⇒ `>=`（与 `six_criteria_audit.judge` 逐字对齐）
                #   + T2 上限改用权威口径 `max(基线 t2_fp, 5)`（老代码 ≤基线 t2_fp=3 ⇒ 比权威更严
                #   ⇒ t2_fp=4 的合法六项全过臂会被误标 five_ok）。
                st["six_ok"] = bool(
                    off.get("t1") is not None and off.get("t3") is not None
                    and best.get("nano", 0) >= NS.BASE_V2["nano"]
                    and best.get("nano4", 0) >= NS.BASE_V2["nano4"]
                    and best.get("f1", 0) >= NS.BASE_V2["f1"]
                    and off.get("t1", 0) >= NS.BASE_V2["t1"]
                    and off.get("t2_fp", 99) <= max(int(NS.BASE_V2.get("t2_fp") or 0), 5)
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
