#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自适应闭环搜索器 v3（2026-11-11 15:20 接管）—— 目标：击败 Baseline_v2.0 = trial_9863

═══ v2 → v3 的四个结构性改动（全部来自 W10 / W11 / SWA 实测，不是拍脑袋）═══

① **训练臂一律从 base init（cs_t0043.pt）起训**，不再从冠军 ckpt 起训。
   证据：W11（9920~9923）与 12009~12019 / 12024~12034 全部是「以 9863 ep5 为 init 再训练」，
        nano 0.6498~0.6779，比基线低 3~6 pp ⇒ **二次退火伤 nano，该路径已证伪**；
        而 v2 的 R1 训练臂与之同型 ⇒ 继续跑等于把一晚上的算力扔进死路。

② **「epochs=20」改「epochs=5」：算力 ÷4，结论逐位不变。**
   证据（代码级，`optuna_mix_ratio_search.py`）：
     - L1190 起 LR 曲线是**全局步号**的函数（`progress = min(1, (step-warmup)/(sched_total-warmup))`
       ⇒ cosine 周期走完后恒在地板，**没有 warm restart**）；`--sched-epochs` 与 `--epochs` 解耦；
     - 采样器种子 = `seed + trial_number`、增广 RNG = `seed`，**都与 `--epochs` 无关**；
     - `total_steps`（L1111）只用于打印。
   ⇒ epochs=20 的 ep1~ep5 与 epochs=5 的 ep1~ep5 **逐位等价**；epochs 6~20 只是 LR 地板上空转，
     而 ep5 权重早在进 ep6 前就存好了 ⇒ 所有新臂统一 `--epochs 5`（cosine 周期仍 5），
     单臂 ~14 min（原来 ~56 min）⇒ **同样的显卡时间能多跑 3 倍的臂**。

③ **按「配方」而不是按「单臂」判优**（防种子运气污染搜索方向）。
   证据：同配方只换种子，sw1.0 差 0.07 pp、sw4.0 差 1.04 pp、**sw5.0 差 1.53 pp**
        （9863 0.708648 vs 9912 0.693289）⇒ 9863 的 0.7086 **含显著运气**。
   ⇒ 新增 recipe_key / recipe_stats：同配方多 seed 求均值（**稳健线**）；
     单臂最乐观读数只作**严格线**；冠军配方按稳健线选，晋级按严格线 + 种子复现双关。

④ **官方三轨门禁只跑在「有可能打平」的臂上**（nano ≥ 基线 − 2 pp，或当轮 nano 前 3），
   其余臂只记 NanoBench ⇒ 省下的 GPU 时间全部换成训练轮次。

═══ 产物 ═══
  - 台账 `nano_search_ledger.json`（顶层 `rounds` / `search_state`）
  - `final_report.md` / `beat_to_pass_report.md`
  - **每轮追加** `manu/memory/adaptive_night_20261111.md`（本夜结果的可读台账）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nano_search as ns  # noqa: E402
from ckpt_synth import average_checkpoints  # noqa: E402

REPO = ns.REPO
TRIAL_BASE = 12000          # 自适应波次 trial 号段（与 W1~W11 不冲突）
SYNTH_EPOCH = 5             # 合成臂统一用的 epoch 文件名（评测按 epoch 号索引）
EPOCHS_EVAL = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
BASE_INIT = ns.CS_T0043     # 所有训练臂的唯一起点（= trial_0031 / 9863 的 base init）
NIGHT_MEM = os.path.join(REPO, "manu/memory/adaptive_night_20261111.md")
NIGHT_LOG = f"{ns.WORK}/night_log.md"

# 配方指纹：只含「决定训练动力学」的参数；不含 seed / init / epochs
RECIPE_KEYS = ("sw", "swl", "thr", "maxside", "lr", "lr_floor", "warmup", "mix", "fb", "ow",
               "wd", "mosaic", "scale", "translate", "clip", "sched", "uf", "hf", "hfsrc",
               "hfch", "native_grid", "mr", "backbone_lr", "head_src")
CHAMP_KEYS = RECIPE_KEYS
GATE_NANO_SLACK = 0.02      # 门禁窗口：nano ≥ 基线 − 2 pp 才值得跑官方三轨
SEED_BASE = 21000           # 种子扫描的种子基数

# ── 时间预算（2026-11-11 16:10 按 R1 实测校准）──────────────────────────────
#   训练：4 卡并行，一波 4 臂 ≈ 16 min（单臂 ~14.5 min ⇒ 摊到墙钟 4 min/臂）
#   评测：NanoBench（ep3/4/5 三档缓存）≈ 1.3 min/臂（4 卡并行）
#         + 门禁 / dual / 板卡 / 报告 ≈ 25 min（与臂数基本无关的固定开销）
ROUND_FIXED_MIN = 62.0      # 门禁 + dual + 板卡 + 报告（2026-11-13 13:50：25 → 45）
#                              2026-11-13 11:55：45 → 62（探索名额 6→8 + SWA 12 + triad
#                              ⇒ 官方三轨 ~21 臂 ⇒ ~63 min ⇒ 固定开销必须跟上，否则末轮超时）
#                              （门禁名额增到 14~16 臂 ⇒ 官方三轨 ~12 min/臂、4 并发 ⇒ ~45 min）
# 🔴 2026-11-12 12:40 实测重标定：4 臂并发一波 4 臂 ≈ 13 min、8 臂并发一波 8 臂 ≈ 24.5 min
#   ⇒ **臂/分钟 ≈ 0.27，两种并发同速**（GPU 早已饱和）⇒ 单臂摊销 ≈ 3.4 min
TRAIN_MIN_PER_ARM = 3.4
EVAL_MIN_PER_ARM = 1.3      # NanoBench 三档缓存
MIN_ROUND_MIN = 60.0        # 剩余时间低于此值 ⇒ 收工，不再开新轮
GATE_RESERVE_MIN = 58.0     # 末轮官方三轨预留（~12 min/臂 × 4 并发 ⇒ ~45 min；11:55：45→58）
FULL_ROUND_MIN = 375.0      # 满载一轮的**训练窗口**需求（96 臂 × 3.725 + SWA 评测 13 ≈ 371；
                              # 旧值 260 是 61 臂口径、360 是 91 臂口径，均已过期）
SAFETY_MARGIN_MIN = 8.0     # 末轮时间自洽表余量只有 0.1 min ⇒ 留 8 min 保险（门禁超时也不冲线）
POST_SWA_ARMS_EST = 16      # 训练后 SWA 臂（8 源 × 等权/LAWA 两变体）⇒ 只占评测时间，预算里要预留
                           # 🟢 2026-11-13 21:50：12 → 16（SWA 零训练、单位算力命中率最高的一类）

# ══════════════════════════════════════════════════════════════════════════
# 突变格队列（按先验概率排序；每个格 = 冠军配方 + 一处改动，起点都是 base init）
#   依据行给出该格的历史证据；无证据的写明「未测」
# ══════════════════════════════════════════════════════════════════════════
MUTATIONS: list[tuple[str, dict, str]] = [
    # ══════════════════════════════════════════════════════════════════════
    # 🔴 2026-11-12 09:20 重排：**T2 安全优先**（官方金池是死线，NanoBench 看不见）
    #   实测（23 个有官方三轨读数的臂，swl 单变量扫描）：
    #     swl=1.0 → T2 0.9940~0.9987 ｜ swl=0.5 → T2 0.9892~0.9978
    #     swl=0.35 → 0.9571 ｜ swl=0.3 → 0.9516~0.9625 ｜ swl=0.2 → 0.9229~0.9307
    #     swl=0.1  → 0.7954~0.8512（**死区**）
    #   ⇒ **大目标降权（swl<1.0）单调毁掉官方 T2 金池**，而 NanoBench（no-GMC）
    #     完全看不见这个塌陷 ⇒ 只按 nano 选冠军 ⇒ 整夜算力全喂进死区。
    #   ⇒ **本队列不再含任何 swl<1.0 的格**；swl 固定 = 1.0（= 基线配方）。
    # ══════════════════════════════════════════════════════════════════════
    # ── 第 1 梯队：**lr_floor 阶梯**（唯一「四项同时上升」的旋钮）────────────
    #   12053（sw5.0 × swl1.0 × lr_floor0.10 @ep4）：nano/nano4/F1/micro **同时**上升
    #   （+0.09 / +0.95 / +0.26 / +0.70 pp）⇒ 是唯一「拿 nano 而不丢 F1」的旋钮。
    ("lrfloor0.10", dict(lr_floor=0.10),
     "LR 保底 10%：**唯一四项全上的旋钮**（12053）⇒ 在基线格（sw5.0×swl1.0）上复现并找峰"),
    ("lrfloor0.05", dict(lr_floor=0.05), "LR 保底 5%（0.10 的下一档，验证保底越大越稳）"),
    ("lrfloor0.075", dict(lr_floor=0.075), "LR 保底 7.5%（0.05 与 0.10 之间细探）"),
    ("lrfloor0.15", dict(lr_floor=0.15), "LR 保底 15%（保底越高 ⇒ 卷积核退化越少 ⇒ 4px 档可能再涨）"),
    ("lrfloor0.20", dict(lr_floor=0.20), "LR 保底 20%"),
    ("lrfloor0.30", dict(lr_floor=0.30), "LR 保底 30%（激进上探）"),
    # ── 第 2 梯队：**sw 剂量阶梯**（实测 sw 1.0~10.0 全都 T2≈0.996 ⇒ 安全轴）────
    ("sw6.0", dict(sw=6.0), "小目标剂量 5.0→6.0（9914 sw6.0 nano 0.7006 / T2 0.9957 安全）"),
    ("sw8.0", dict(sw=8.0), "小目标剂量 5.0→8.0（9822 sw8.0 nano@4px 0.6706 历史最高 / T2 0.9966）"),
    ("sw10.0", dict(sw=10.0), "小目标剂量 5.0→10.0（9915 sw10.0 nano 0.7046 / T2 0.9965）"),
    ("sw4.5", dict(sw=4.5), "剂量细探 sw=4.5（sw4.0=0.7004 与 sw5.0=0.7086 之间）"),
    ("sw5.25", dict(sw=5.25), "剂量细探 sw=5.25"),
    ("sw6.5", dict(sw=6.5), "剂量细探 sw=6.5（sw6.0=0.7006 / sw7.0=0.6883 之间）"),
    ("sw12.0", dict(sw=12.0), "剂量上探 sw=12.0"),
    # ── 第 3 梯队：**小目标定义档位**（thr/maxside；nano@4px 严格档的直接旋钮）──
    ("thr18ms5", dict(thr=18.0, maxside=5.0), "中宽小目标定义（thr16/maxside4 的下一档）"),
    ("thr32ms8", dict(thr=32.0, maxside=8.0), "更宽小目标定义（直击 nano@4px 严格档）"),
    ("thr12ms3", dict(thr=12.0, maxside=3.0), "更严小目标定义（4px 档压力测试）"),
    ("thr25ms6", dict(thr=25.0, maxside=6.0), "宽小目标定义"),
    ("thr40ms10", dict(thr=40.0, maxside=10.0), "极宽小目标定义"),
    # ── 第 4 梯队：**优化器/损失/数据侧**（都不动大目标权重 ⇒ T2 安全）──────
    ("ow0.5", dict(ow=0.5), "offset 权重 0.4→0.5（定位更准 ⇒ 4px 档受益）"),
    ("ow0.3", dict(ow=0.3), "offset 权重 0.4→0.3"),
    ("mr2", dict(mr=2), "min_radius 1→2（监督峰更宽 ⇒ F1）"),
    ("fb3.0", dict(fb=3.0), "focal beta 2.75→3.0"),
    ("fb2.5", dict(fb=2.5), "focal beta 2.75→2.5"),
    ("mixw2.2", dict(mix="w2.2"), "混样 w2.0→w2.2（细探）"),
    ("mixw2.5", dict(mix="w2.5"), "混样 w2.0→w2.5（9873 sw3.0 上 ≈持平）"),
    ("mixw1.8", dict(mix="w1.8"), "混样 w2.0→w1.8"),
    ("warmup1.0", dict(warmup=1.0), "warmup 1.5→1.0"),
    ("warmup2.5", dict(warmup=2.5), "warmup 1.5→2.5"),
    ("lr3.0", dict(lr=3.0), "lr_scale 3.5→3.0"),
    ("lr4.0", dict(lr=4.0), "lr_scale 3.5→4.0"),
    ("lr4.5", dict(lr=4.5), "lr_scale 3.5→4.5"),
    ("mosaic0.30", dict(mosaic=0.30), "mosaic 0.15→0.30（小目标尺度多样性）"),
    ("scale0.20", dict(scale=0.20), "scale 抖动 0.10→0.20"),
    ("translate0.10", dict(translate=0.10), "translate 抖动 0.05→0.10"),
    ("wd5e4", dict(wd=0.0005), "weight decay ×1.86"),
    ("wd1.34e4", dict(wd=0.000134), "weight decay ×0.5"),
    ("bblr0.5", dict(backbone_lr=0.5), "backbone lr ×0.5（骨干少动 ⇒ 头更专注文本域）"),
    ("clip3.0", dict(clip=3.0), "clip_max_norm 5.65→3.0（梯度更稳）"),
    ("uf1", dict(uf=1), "解冻 b0~b4（更激进的域适配）"),
    # ── 第 5 梯队：**双变量组合**（都建立在上面已验证安全的单变量格之上）────
    ("lrfloor0.10_thr12ms3", dict(lr_floor=0.10, thr=12.0, maxside=3.0),
     "LR 保底 10% × 更严小目标定义（12057 单靠 thr12ms3 已 nano4/F1 全过）"),
    ("lrfloor0.10_ow0.5", dict(lr_floor=0.10, ow=0.5), "LR 保底 10% × offset 0.5"),
    ("lrfloor0.10_sw6.0", dict(lr_floor=0.10, sw=6.0), "LR 保底 10% × 小目标剂量 6.0"),
    ("lrfloor0.05_thr12ms3", dict(lr_floor=0.05, thr=12.0, maxside=3.0),
     "LR 保底 5% × 更严小目标定义"),
    ("lrfloor0.10_mixw2.2", dict(lr_floor=0.10, mix="w2.2"), "LR 保底 10% × 混样 w2.2"),
    ("lrfloor0.15_ow0.5", dict(lr_floor=0.15, ow=0.5), "LR 保底 15% × offset 0.5"),
    ("sw6.0_thr25ms6", dict(sw=6.0, thr=25.0, maxside=6.0), "sw6.0 × 宽小目标定义"),
    ("sw8.0_thr18ms5", dict(sw=8.0, thr=18.0, maxside=5.0), "sw8.0 × 中宽小目标定义"),
    ("ow0.5_thr12ms3", dict(ow=0.5, thr=12.0, maxside=3.0), "offset 0.5 × 更严小目标定义"),
    ("fb3.0_thr12ms3", dict(fb=3.0, thr=12.0, maxside=3.0), "focal 3.0 × 更严小目标定义"),
    ("mr2_lrfloor0.10", dict(mr=2, lr_floor=0.10), "min_radius 2 × LR 保底 10%"),
    ("sw8.0_lrfloor0.10", dict(sw=8.0, lr_floor=0.10), "sw8.0 × LR 保底 10%"),
]


# 🟢 2026-11-12 11:20 新增：**优先格（priority cells）** —— 每格每轮 2 个 seed
#   动机（来自 2026-11-12 上午的台账重算）：
#     ① **单臂读数分不清「配方好」还是「种子好」** ⇒ 每格每轮固定 2 个 seed，
#        攒到 n≥4 后按「均值 vs 基线配方均值」淘汰（稳健判优）。
#     ② 优先格 = **离基线只差一档** 的格（不是随机新格）：
#        - thr12ms3(12057) nano 0.707703，**离严格线 0.708648 只差 0.000945**，
#          且 nano4 0.651701 / F1 0.890647 **都已过** ⇒ 加 LR 保底补 nano 即可越线；
#        - lrfloor0.10(12053) nano 0.709594 过线、官方 T1 0.887102 **只差 0.000984**
#          ⇒ 需要「nano 峰右移一格」把 T1 一起带上去 ⇒ 拉长 cosine 周期（sched 6/7）；
#        - sw8.0(9822) T1 0.889051 已过线、nano 只差 0.003545 ⇒ 加 LR 保底；
#        - swl<1.0 **确认死区**（实测 swl0.1→T1≤0.848/T3≤0.854；swl0.2~0.35→T1≤0.884、T3≤0.9356）
#          ⇒ 全部优先格固定 swl=1.0。
PRIORITY_PATCHES: list[tuple[str, dict, str]] = [
    ("lrf0.10_thr12ms3", dict(lr_floor=0.10, thr=12.0, maxside=3.0),
     "LR 保底 10% × 更严小目标定义 thr12/ms3（12057 单靠 thr12ms3 nano 只差 0.000945）"),
    ("thr12ms3", dict(thr=12.0, maxside=3.0),
     "更严小目标定义 thr12/ms3（12057 复现：nano 0.707703 距严格线 0.000945）"),
    ("sched6_lrf0.10", dict(sched=6, lr_floor=0.10),
     "cosine 周期 5→6 + LR 保底 10%（nano 峰右移一格 ⇒ 与官方 T1 过线点重合）"),
    # 🟢 2026-11-12 15:30 新增：**sw=4.0 家族**（台账全量复算 87 个有官方读数的臂后，
    #   找到的**唯一「T1 + nano4 双过线」家族**；此前只在 epochs=20 上试过 3 次，epochs=5 从未跑过）
    #   证据（sw4.0 × swl1.0 × w2.0 × thr16 × lrf0 × init=cs_t0043）：
    #     · 9814(ep20) nano 0.702032 / **nano4 0.647684 ✔** / f1 0.892624 ✔ / **T1 0.890275 ✔** / T3 0.947311
    #     · 9913(ep10) nano 0.696834 / **nano4 0.657372 ✔** / f1 0.890973 ✔ / **T1 0.888642 ✔** / T3 0.946350
    #   ⇒ **2/2 臂的 nano4 与 T1 都过线**（对照：冠军家族 sw5.0 的 nano4 过线仅 3/23、T1 过线 ~1/3）
    #   ⇒ 缺口只剩 **nano（差 ~0.008~0.012）** 与 **T3（差 ~0.002）**
    #   ⇒ 与「thr12ms3」这个 nano 加料格相乘 ⇒ sw4.0_thr12ms3
    ("sw4.0", dict(sw=4.0),
     "小目标剂量 5.0→4.0（**唯一 T1+nano4 双过线家族**：9814/9913 两臂 nano4 0.6477/0.6574、T1 0.8903/0.8886 全过线；"
     "缺口只剩 nano −0.008 与 T3 −0.002）"),
    ("sw4.0_thr12ms3", dict(sw=4.0, thr=12.0, maxside=3.0),
     "sw4.0 × 更严小目标定义 thr12/ms3（**把 sw4 的 T1/nano4 强项 与 12057 的 nano4 0.651701 强项相乘**："
     "12057 单靠 thr12ms3 就 nano4 过线、nano 只差 0.000945 ⇒ 两格互补，从未试过）"),
    ("sw8.0_lrf0.10", dict(sw=8.0, lr_floor=0.10),
     "小目标剂量 8.0 + LR 保底 10%（9822 sw8 官方 T1 已过线，nano 只差 0.003545）"),
    ("lrf0.10", dict(lr_floor=0.10),
     "LR 保底 10%（12053 复现：nano 过线、官方 T1 只差 0.000984）"),
    # 🟢 2026-11-12 12:45 新增：**mix 轴是今晚证据最强的方向**
    #   实测：mix=w2.0（基线）nano/T1 过线但 T3 −0.0017；mix=w1.8 T3 ✔ T2 ✔ 但 nano −0.0033；
    #        mix=w2.5 nano ✔ T1 ✔ T2 ✔ 但 T3 −0.0045、nano4 −0.019
    #   ⇒ **w1.9 正好落在 w1.8（官方强）与 w2.0（nano 强）之间** ⇒ 最可能同时压住两端的点
    ("mixw1.9", dict(mix="w1.9"),
     "混样 w2.0→w1.9（w1.8 官方强 / w2.0 nano 强 ⇒ 中间点最可能两侧同过）"),
    # 🟢 2026-11-12 13:10 新增：**mixw1.9 × lrf0.10**（两个「已验证强项」的组合，从未试过）
    #   依据（台账全量复算）：
    #     · 12347（mix w1.8）：F1 ✔ / 虚警4 ✔ / **T3 0.949390 ✔** / T1 0.887942（差 0.00014）
    #       / nano 0.705340（差 0.0033）/ nano4 0.643667（差 0.0035）⇒ **4/6 近失**
    #     · 12053（lrf0.10）：**nano ✔ / nano4 +0.0095 ✔ / F1 ✔ / 虚警4 ✔ / T3 0.949250 ✔**
    #       / T1 0.887102（差 0.00098）⇒ **5/6，全台账最接近的一个**
    #   ⇒ w1.9 负责把 **T3/T1** 拉回线内，lrf0.10 负责把 **nano/nano4** 拉过线
    #     ⇒ 二者相乘是台账里唯一「两侧强项互补」的未试组合
    ("lrf0.10_mixw1.9", dict(lr_floor=0.10, mix="w1.9"),
     "LR 保底 10% × 混样 w1.9（12347 的 T3 强项 × 12053 的 nano 强项 ⇒ 两侧互补）"),
    # 🟢🟢 2026-11-13 09:40 新增：**T3 前沿格族**（thr12ms3 × lrf0.1 × sw 邻域）
    #   证据（11-13 08:52 六项全盘复算，112 个有完整读数的 (臂,epoch)）：
    #     · 12519 (sw5.0 × thr12ms3 × lrf0.1 × w2.0, ep3)：nano 0.715974 ✔ / nano4 0.657372 ✔
    #       / f1 0.890534 ✔ / T1 0.890586 ✔ / 虚警 2 ✔ / **T3 0.948374（只差 0.000115）** ⇒ 5/6
    #     · 12679 (sw4.0 × thr12ms3 × lrf0.1 × w2.0, ep3)：nano 0.718809 ✔ / nano4 0.655718 ✔
    #       / f1 0.890329 ✔ / T1 0.890550 ✔ / 虚警 1 ✔ / **T3 0.948332（只差 0.000157）** ⇒ 5/6
    #     · 12694 / 12520 / 12635 同族同款，T3 差 0.0008~0.0018
    #   ⇒ **绑定项只剩 T3，且只差 1e-4 量级** ⇒ 这是台账上离六项全过最近的家族。
    #   ⇒ 下面 8 格把算力**钉死**在这个族上（每格 2 seed/轮）：
    #       sw ∈ {4.0, 4.5, 5.0, 5.5} × thr12ms3 × lrf0.1 × w2.0
    #       + thr14（12 与 16 之间的插值）× sw ∈ {4.0, 5.0}
    #       + lrf0.12（0.10 与 0.20 之间的插值）
    ("T3F_sw5.0", dict(sw=5.0, thr=12.0, maxside=3.0, lr_floor=0.10, mix="w2.0"),
     "T3 前沿：sw5.0 × thr12ms3 × lrf0.1（12519 的配方，T3 只差 0.000115）"),
    ("T3F_sw4.0", dict(sw=4.0, thr=12.0, maxside=3.0, lr_floor=0.10, mix="w2.0"),
     "T3 前沿：sw4.0 × thr12ms3 × lrf0.1（12679 的配方，T3 只差 0.000157）"),
    ("T3F_sw4.5", dict(sw=4.5, thr=12.0, maxside=3.0, lr_floor=0.10, mix="w2.0"),
     "T3 前沿：sw4.5（4.0 与 5.0 之间插值）× thr12ms3 × lrf0.1"),

    ("T3F_sw5.0_thr14", dict(sw=5.0, thr=14.0, maxside=3.0, lr_floor=0.10, mix="w2.0"),
     "T3 前沿：sw5.0 × thr14ms3（thr12 与 thr16 之间的插值）× lrf0.1"),
    ("T3F_sw4.0_thr14", dict(sw=4.0, thr=14.0, maxside=3.0, lr_floor=0.10, mix="w2.0"),
     "T3 前沿：sw4.0 × thr14ms3（thr12 与 thr16 之间的插值）× lrf0.1"),


    # 🟢🟢 2026-11-13 09:05 新增：**mix 细粒度格**（拿「nano 富余」换「T3 缺口」）
    #   依据（前沿臂的六项余量）：12519 nano +0.0073 / nano4 +0.0102 / f1 +0.0008 / T1 +0.0025
    #        / 虚警 −3 / **T3 −0.000115**；12679 nano +0.0102 / f1 +0.0006 / T1 +0.0025 / T3 −0.000157
    #   ⇒ **nano 有 ~0.7~1.0 pp 富余、T3 只差 1e-4** ⇒ 该拿 nano 富余去换 T3。
    #   mix 口径 1：P(LQ)=1/(1+r)，P(UAV)=r/(1+r)
    #     w2.00 → P(UAV)=0.6667（当前）｜ w1.95 → 0.6618 ｜ w1.90 → 0.6552 ｜ w1.80 → 0.6429
    #   实测方向（台账）：mix 越小 T3 越好、nano 略差（w1.8：T3 0.949390 ✔ / nano −0.0033）
    #   ⇒ 取 **w1.95**（nano 只掉 ~0.0008，够不到 0.0073 富余；T3 期望 +0.0003~0.0006）
    ("T3F_sw5.0_mixw1.95", dict(sw=5.0, thr=12.0, maxside=3.0, lr_floor=0.10, mix="w1.95"),
     "T3 前沿：sw5.0 × thr12ms3 × lrf0.1 × mix1.95（拿 nano 富余换 T3 缺口）"),
    ("T3F_sw4.0_mixw1.95", dict(sw=4.0, thr=12.0, maxside=3.0, lr_floor=0.10, mix="w1.95"),
     "T3 前沿：sw4.0 × thr12ms3 × lrf0.1 × mix1.95（拿 nano 富余换 T3 缺口）"),
    # 🟢🟢 2026-11-13 10:45 新增：**thr/ms 插值格**（thr12 与 thr16 之间的真实空白）
    #   依据（11-13 09:24 全盘审计，沿 thr 轴 lrf0.1 × sw5.0 × w2.0 的两个近失臂）：
    #     · thr12/ms3（12519 ep3）：nano 0.715974 ✔ / nano4 0.657372 ✔ / f1 0.890534 ✔
    #       / T1 0.890586 ✔ / 虚警 2 ✔ / **T3 0.948374 ✘（只差 0.000115）**
    #     · thr16/ms4（12053 ep4）：nano 0.709594 ✔ / nano4 0.656664 ✔ / f1 0.892338 ✔
    #       / **T1 0.887102 ✘（只差 0.000985）** / T3 0.949250 ✔
    #   ⇒ 两个近失臂**夹住同一个缺口方向**（thr 越大 T3 越好、T1 越差）
    #   ⇒ **thr14** 正好在中间：T3 有望越线、T1 有望仍在线上 ⇒ 六项可能同时过。
    #   ⚠️ maxside 必须沿 **ms = thr/4** 这条线走（台账实测：thr12→3 / thr16→4 / thr18→5 /
    #     thr20→5 / thr25→6 / thr32→8 / thr40→10）⇒ 现有的 thr14×**ms3** 反而是离线的点，
    #     真正的插值点是 **thr14 × ms3.5**（11-13 前从未跑过）。
    ("T3I_sw5.0_thr14ms3.5", dict(sw=5.0, thr=14.0, maxside=3.5, lr_floor=0.10, mix="w2.0"),
     "T3 插值：sw5.0 × thr14 × ms3.5（ms=thr/4 线上 thr12 与 thr16 的中点 ⇒ 夹住 12519/12053 的缺口）"),
    ("T3I_sw4.0_thr14ms3.5", dict(sw=4.0, thr=14.0, maxside=3.5, lr_floor=0.10, mix="w2.0"),
     "T3 插值：sw4.0 × thr14 × ms3.5（12679 的 sw 侧 + 12519/12053 的 thr 侧中点）"),
    ("T3I_sw5.0_thr13ms3.25", dict(sw=5.0, thr=13.0, maxside=3.25, lr_floor=0.10, mix="w2.0"),
     "T3 插值：sw5.0 × thr13 × ms3.25（thr12→thr14 之间的细粒度点）"),
    ("T3I_sw5.0_thr15ms3.75", dict(sw=5.0, thr=15.0, maxside=3.75, lr_floor=0.10, mix="w2.0"),
     "T3 插值：sw5.0 × thr15 × ms3.75（thr14→thr16 之间的细粒度点）"),
    # 🟢🟢 2026-11-13 17:50：**双插值格**（sw 轴 × thr 轴的交点）
    #   依据（台账 545 个有读数的臂全盘复算）：
    #     · 12 个「三项全平」臂里，**T3 过线的 2 个全部来自 thr16ms4 族**（9863 / 12053）；
    #       thr12ms3 族 6 个**全灭 T3**，但最接近的只差 1.15e-4（12519）。
    #     · 冠军族（sw4.0×thr12ms3×lrf0.1，n=90）产出 3 个三项全平臂（12679/12694/12709），
    #       **T1 全过、F1/nano 全过，只差 T3 1.6e-4~7.8e-4** ⇒ 只差 T3 一个旋钮。
    #     · 已知能把 T3 顶上去的两个旋钮：**thr 16→12 会掉 T3、sw 5→4 会掉 nano**；
    #       而 **lrf 0→0.1 让 T3 过线**（12053：T3 0.949250 ✔ / 只差 T1 0.000984）。
    #   ⇒ 缺的是「**thr12 的 T1/F1 强度 + thr16 的 T3 强度**」的中间点：
    #     thr14ms3.5 是 thr 轴的插值点，sw4.0/sw5.0 两端已有，**sw4.5（sw 轴插值）从未采样**。
    ("T3I_sw4.5_thr14ms3.5", dict(sw=4.5, thr=14.0, maxside=3.5, lr_floor=0.10, mix="w2.0"),
     "**双插值交点**：sw4.5（sw 轴插值）× thr14ms3.5（thr 轴插值）× lrf0.1 —— "
     "既有 thr12 族的 T1/F1 强项、又有 thr16 族的 T3 强项，且 sw4.5 正是 nano↔T3 交换曲线的中点（此前从未采样）"),
    ("T3I_sw4.25_thr14ms3.5", dict(sw=4.25, thr=14.0, maxside=3.5, lr_floor=0.10, mix="w2.0"),
     "双插值交点的细网格：sw4.25 × thr14ms3.5（sw4.0 与 sw4.5 之间，把 nano↔T3 交换曲线切得更细）"),
]

# 优先格淘汰线：n>=PRIORITY_MIN_N 后，若均值 nano 低于基线配方均值 PRIORITY_MEAN_SLACK ⇒ 停喂
PRIORITY_MIN_N = 4
PRIORITY_MEAN_SLACK = 0.002
# 优先格每轮臂数上限（格数 × 2 seed）⇒ 时间预算里先给它留位
#   2026-11-13 09:40：T3 前沿族加进来 ⇒ 上限 18 → 26（去重后实际喂 ~9~12 格）
PRIORITY_ARMS_EST = 48      # 优先格预留臂数（2026-11-13 12:50：34 → 48）
#                              （新增 5 个「基线派生绝对格」共 12 臂 + thr14 插值格加种子 2 臂）
# 合成臂（SWA）每轮数量上限（只占评测时间，不占训练时间）｜2026-11-12 13:15
SWA_ARMS_EST = 50    # 合成臂预估（2026-11-13 13:50：42 → 50；含 LAWA 加权变体）
#                              （源臂 3 → 6 个 × 7 窗口 = 42 ⇒ SWA 是**零训练**且命中率最高的一类臂）
SWA_SRC_MIN = 8      # 每轮 SWA 源臂数下限（CLI 的 --swa-top-n 只作下限，不再当上限用）
                     # 🟢 2026-11-13 21:50：6 → 8（多 2 个源臂 ⇒ 多 4 个零训练候选臂）

# 🟢🟢 2026-11-13 12:40 新增：**基线派生绝对格**（以 Baseline_v2.0 的配方为底，**不再受「冠军漂移」影响**）
#   依据（2026-11-13 12:20 台账全量复算 947 臂，按 (sw,thr,ms,lrf) 家族统计）：
#     · T3（官方盲测）随 sw **下降**变好：sw1 54% 过线(7/13) / sw3 20%(3/15) / sw4 0%(0/7) / sw5 2%(4/194)
#     · nano（微小目标召回）随 sw **上升**变好：sw1 0% / sw3 0% / sw4 16%(4/25) / sw5 56%(108/194)
#     · F1 方向相反：sw4 76%(19/25) 过线 / sw5 23%(44/194)
#     ⇒ **sw 是 nano ↔ (F1 / T1 / T3) 的主交换轴**，而 4.0~5.0 之间是**未采样的插值区**
#     ⇒ 且「sw5 × thr16ms4 × lrf0.1」（= 12053 的配方，全台账唯一「T3 ✔ nano ✔ nano4 ✔ F1 ✔ 虚警 ✔
#       只差 T1 0.000985」的 5/6 臂）**从未跑过 sw4 / sw4.5 版本** ⇒ 交叉出两个互补强项
#   ⇒ 这 5 格每轮固定喂 2~3 个 seed（零探索成本、命中率最高的未采样邻域）
BASE_DERIVED_CELLS: list[tuple[str, dict, str, int]] = [
    # ══════════════════════════════════════════════════════════════════════
    # 🔴🟢🟢 2026-11-13 11:20 **按「格的期望产出率 E」重排 + 加权**（全台账 947 臂复算）
    #   E[六项全过/臂] = p_triad × p_T1 × p_虚警 × p_T3（n = 有官方读数的臂数）：
    #     格                                n    p_triad p_t1  p_fp  p_t3   E
    #     A2 = sw4.0×thr12ms3×lrf0.1      13    0.23   0.69  0.62  0.31   **0.030**
    #     A1 = sw5.0×thr12ms3×lrf0.1       8    0.38   0.50  0.62  0.25   **0.029**
    #     B1 = sw5.0×thr16ms4×lrf0.0      52    0.06   0.19  0.87  0.04    0.0004
    #     C  = sw4.0×thr16ms4×lrf0.0       6    0.00   1.00  1.00  0.00    0.000
    #   ⇒ **A1/A2 的单臂期望产出率是 B1 的 30~70 倍**；而旧表几乎全押 thr16ms4 家族
    #     （B1 已投 52 臂 / 52 条官方读数 ⇒ 只出 1 个 6/6 = 9863 本体）⇒ **算力错配**。
    #   ⇒ A1/A2 的「三项全平臂」= 12679/12694/12709/12635/12519/12520
    #     **六项只差 T3 1.2e-4 ~ 2.7e-3（σ_t3 = 8.6e-4 ⇒ 1~3σ 内）** ⇒ 换 seed 就可能全过。
    #   ⇒ **表序 = 末轮预算收紧时的砍序**（整格丢规则）⇒ 高产格必须排在最前。
    # ══════════════════════════════════════════════════════════════════════
    ("BASE_A2_sw4.0_thr12ms3_lrf0.1", dict(sw=4.0, thr=12, maxside=3.0, lr_floor=0.10),
     "**全台账最高产格 A2**（sw4.0×thr12ms3×lrf0.1）：E=0.030 = B1 的 70 倍；"
     "12679/12694/12709/12635 四个 5/6 臂全在此格，只差 T3 1.6e-4~7.8e-4 ⇒ 5 seed",
     5),
    ("BASE_A1_sw5.0_thr12ms3_lrf0.1", dict(sw=5.0, thr=12, maxside=3.0, lr_floor=0.10),
     "**次高产格 A1**（sw5.0×thr12ms3×lrf0.1）：E=0.029；p_triad=0.38 最高；"
     "12519/12520 两个 5/6 臂在此格（只差 T3 1.15e-4 / 1.84e-3）⇒ 5 seed",
     5),
    ("BASE_A2i_sw4.25_thr12ms3_lrf0.1", dict(sw=4.25, thr=12, maxside=3.0, lr_floor=0.10),
     "A2 族的 sw 插值（4.0 与 4.5 之间）：把 nano↔T3 交换曲线切细", 4),
    ("BASE_Ai_sw4.5_thr12ms3_lrf0.1", dict(sw=4.5, thr=12, maxside=3.0, lr_floor=0.10),
     "**A 族交换曲线中点**（sw4.5 = A1 与 A2 正中间：nano 侧更强 / T3 侧稍弱）", 4),
    ("BASE_A1i_sw4.75_thr12ms3_lrf0.1", dict(sw=4.75, thr=12, maxside=3.0, lr_floor=0.10),
     "A1 族的 sw 插值（4.5 与 5.0 之间）", 3),
    ("BASE_A1hi_sw5.25_thr12ms3_lrf0.1", dict(sw=5.25, thr=12, maxside=3.0, lr_floor=0.10),
     "A1 族向 B1 方向的外插（sw5.25）：nano 侧更强、T3 侧更弱的未知点", 2),
    ("BASE_sw5.0_lrf0.1", dict(lr_floor=0.10),
     "B2 = 12053 的配方（thr16ms4 × lrf0.1）：T3 0.949250 ✔、虚警 4 ✔，只差 T1 0.000985 ≈ 0.74σ"
     "⇒ lrf0→0.1 是**已被验证**能把 T3 顶过线的旋钮 ⇒ 4 seed",
     4),
    ("BASE_sw4.0_lrf0.1", dict(sw=4.0, lr_floor=0.10),
     "C × lrf0.1：把 12053 的 T3 强项搬到 sw4.0 上（C 与 B2 的乘积格，从未采样）", 3),
    ("BASE_sw4.25_lrf0.1", dict(sw=4.25, lr_floor=0.10),
     "C 族的 lrf0.1 细网格（sw4.0 与 sw4.5 之间）", 3),
    ("BASE_sw4.5_lrf0.1", dict(sw=4.5, lr_floor=0.10),
     "C 族的 lrf0.1 插值（sw4.5 = nano↔T3 交换曲线中点）", 3),
    ("BASE_sw4.75_lrf0.1", dict(sw=4.75, lr_floor=0.10),
     "C 族的 lrf0.1 细网格（sw4.5 与 sw5.0 之间）", 2),
    ("BASE_sw5.0", dict(),
     "**基线配方本体**（B1 = 9863 的配方，thr16ms4 × lrf0.0）：全台账唯一产出过 6/6 的格"
     "⇒ 但实测 E=0.0004（n=52 ⇒ 只出 1 个 6/6）⇒ **降配到 3 seed**（算力让给 A 家族）",
     3),
    ("BASE_sw4.0", dict(sw=4.0),
     "C = 基线只改 sw4.0：12702/12703 两个 SWA 臂都是 5/6（nano/F1/T1/虚警全过，"
     "只差 nano4 0.0043 与 T3 2.4e-5）⇒ 但正式臂实测 p_triad=0 ⇒ 先用 2 seed 验证",
     2),
    ("BASE_sw4.25", dict(sw=4.25),
     "sw 轴细网格（sw4.0 与 sw4.5 之间）：把 nano↔T3 交换曲线切细（C 族插值）", 2),
    ("BASE_sw4.5", dict(sw=4.5),
     "sw 轴插值（4.0 与 5.0 之间，C 族的邻格）", 2),
    ("BASE_sw4.75", dict(sw=4.75),
     "sw 轴插值（4.5 与 5.0 之间，nano 侧最靠近 B1 的点）", 2),
]

# 🟢 2026-11-13 12:45：**thr14 插值格加种子**（thr12 的 nano 强 + thr16 的 T3 强 ⇒ 真正插值点）
#   thr12ms3 家族：6 个三项全平臂 T3 全部不过线（−1.2e-4 ~ −2.7e-3）
#   thr16ms4 家族：T3 过 2/3 ⇒ thr14（ms3.5）是最有意义的中间点 ⇒ 每格 2 → 3 seed
PRIORITY_SEED_OVERRIDE: dict[str, int] = {
    "T3I_sw4.0_thr14ms3.5": 3,
    "T3I_sw5.0_thr14ms3.5": 3,
    "T3I_sw4.5_thr14ms3.5": 3,   # 🟢 2026-11-13 17:50 新增：双插值交点（从未采样 ⇒ 3 seed）
    "T3I_sw4.25_thr14ms3.5": 2,
    "T3I_sw5.0_thr13ms3.25": 2,
    "T3I_sw5.0_thr15ms3.75": 2,
}

# ── 算力再分配（2026-11-12 12:40，R7 全量读数实测得出）────────────────────
#   实测（R7 全体 24 臂）：
#     · **种子扫描**（冠军配方换 seed）：6 臂里 **2 臂是 4/6 近失**（12333 / 12335）⇒ 33%
#     · **突变**（单参数改动）：18 臂里 **2 臂是 4/6 近失**（12346 mixw2.5 / 12347 mixw1.8）⇒ 11%
#     · 全台账 469 臂里 NanoBench 三项全过的只有 6 个（其中 3 个还是 swl<1.0 死区）
#   ⇒ **近失格是稀缺资源**：把算力从「泛突变」挪到「多种子复现近失格」是最优分配。
#   ⇒ 种子扫描臂数提到 12（胜率 ≈ 突变臂的 3 倍），突变臂数压到 8（保留探索但不让它吃掉主力）。
# 🟢 2026-11-12 13:10 再分配（台账 469 臂全量复算后的结论，见 §149.18）：
#   **总臂数受 6 并发限制 ⇒ 一夜下来总臂数基本恒定（≈210~220）**
#   ⇒ 唯一可控的量是「算力配比」，不是臂数。
#   分层命中率实测（官方三轨读数回填后）：
#     · 基配方族（sw5/swl1/thr16/lrf0/w2.0）：nano 三项全过 2/32 臂（6%），
#       T1 过 8/26（31%）、**T3 过 3/26（11.5%，最紧的一项）**、虚警≤5 过 23/26（88%）
#       ⇒ 单臂六项全过概率 ≈ 1%（**台账里 9863 / 12333 / 12335 / 12327 全部出自这一族**）
#     · swl<1.0 一律 T1/T3 塌陷（死区，已 3 轮证实）⇒ 不投
#     · 突变臂（单参数改动）近失率 11% ≈ 种子臂 33% 的 1/3 ⇒ 压到 5 臂保底
#   ⇒ 种子格 14→20（最高产家族，多下注），突变格 8→5（保底探索），新增 1 个互补配方格。
SEED_SCAN_TARGET = 24        # 每轮种子扫描臂数（多格分配，见 seed_scan_targets）｜11-13 11:40 由 18 提回 24
#                              11-13 09:45 曾由 24 压到 18（当时种子扫描**只打冠军 A2 一格**）
#                              ⇒ 11-13 11:40 改为**按产出率分给 4 个高产格**（A1/A2/B1/B2）后，
#                                 边际收益判据反转：种子臂的三项全平产出率 ≈ 6.5%（A2 3.9%、A1 8.3%、
#                                 B1 9.1%、B2 4.8%），而突变臂 ≈ 1% ⇒ 6 臂种子 > 6 臂突变
#                                 ⇒ 提回 24（实测分配 B1 9 / A1 7 / B2 4 / A2 4，含**基线组 B1 保底**）
# 🟢 2026-11-12 15:30 由 20 提到 24：台账全量复算给出的**命中率分层**（470 臂）
#   · 冠军家族（sw5.0×swl1.0×lrf0×w2.0×thr16，init=cs_t0043，epochs=5）23 臂：
#     nano 过线 1/23、nano4 过线 3/23、f1 过线 11/23、**三项全过 1/23（4.3%）**
#   · swl=1.0 的**非**冠军家族 287 臂：三项全过仅 2 臂（0.7%）
#   ⇒ 冠军种子臂的「三项全过」概率 ≈ 非冠军臂的 **6 倍** ⇒ 种子扫描是最划算的算力投放
#   ⇒ 同时把探索性优先格控制在 9 格 × 2 seed（18 臂）不扩量 ⇒ 总量不涨、命中率涨
MUT_CAP = 3                  # 每轮突变臂数上限（探索保底）｜2026-11-13 17:55 由 5 压到 3
#                              理由：实测突变臂的三项全平产出率 ≈1%（种子臂 ≈6.5%），
#                              而夜间算力要压到「双插值交点 + 近失前沿」上 ⇒ 突变让位 2 臂
# 近失前沿复现：每轮自适应挑「只差一档」的配方格，补没试过的 seed
NEAR_MISS_CELLS = 5          # 每轮最多复现几个近失格｜2026-11-13 09:45 由 4 提到 5
NEAR_MISS_SEEDS = 4          # 每个格每轮补几个 seed｜2026-11-12 13:15 由 2 提到 3
#                              2026-11-13 09:45 由 3 提到 4（逐臂 T2 口径解锁后，近失格＝前沿格）
NEAR_MISS_MIN_PASS = 3       # 至少过 3/6 项才算「近失」
NEAR_MISS_MAX_DEFIIT = 0.04  # 过 4/6 时的归一化亏损上限
NEAR_MISS_MAX_DEFIIT_3 = 0.012  # 只过 3/6 时必须**非常接近**（否则不算近失）


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 时间预算：按剩余时间决定本轮跑多少臂（保证最后一轮能在硬截止前跑完）
# ---------------------------------------------------------------------------
def fixed_cost(remain_min: float) -> float:
    """一轮的**固定开销**（门禁 + dual + 板卡 + 报告），**随剩余时间收缩**。

    🟢 2026-11-13 14:00：门禁名额增到 14~16 臂 ⇒ 官方三轨 ~12 min/臂、4 并发 ⇒ ~45 min。
    ⇒ 固定开销不再是常数：剩 62 min 时不可能真的花 45 min 做门禁（会把末轮撑爆）
    ⇒ 改成 `min(ROUND_FIXED_MIN, max(20, remain*0.5))`：**时间紧 ⇒ 门禁只跑能跑完的臂**。
    """
    return float(min(ROUND_FIXED_MIN, max(20.0, remain_min * 0.5)))


def plan_minutes(n_arms: int, swa_arms: int | None = None, fixed: float | None = None) -> float:
    """跑 n_arms 臂预计占用的墙钟时间（分钟）。

    🔴 2026-11-12 12:40 重标定（8 臂并发实测）：一波 8 臂 ≈ **24.5 min**，
       4 臂并发时一波 4 臂 ≈ 13 min ⇒ **摊销同为 ≈3.2 min/臂** ⇒
       **GPU 早在 4 臂/卡时就已经饱和**（util 45%→90% 但臂/分钟没变）⇒
       不能再用「waves × 16 min」那个按 4 并发算的公式（会把预算高估 2 倍 ⇒ 最后一轮被砍）。
    """
    conc = max(1, int(getattr(ns, "TRAIN_WORKERS", [4])[0]))
    per_arm_train = 3.4 if conc >= 4 else 5.2
    # 🟢 2026-11-12 13:15：合成臂（SWA）**不占训练时间**，只占评测时间
    #   ⇒ 预算里要单列，否则会高估（合成臂每轮 ~4~6 个）。
    swa = SWA_ARMS_EST if swa_arms is None else int(swa_arms)
    fx = ROUND_FIXED_MIN if fixed is None else float(fixed)
    return (n_arms * (per_arm_train + EVAL_MIN_PER_ARM)
            + swa * EVAL_MIN_PER_ARM + fx)


def swa_reserve(remain_min: float) -> int:
    """SWA（合成臂）评测预留臂数：**随剩余时间收缩**。

    🔴 2026-11-13 11:00 修（selftest 抓到的真坑）：
       SWA 臂是**零训练**的（只花评测时间），但原来的固定预留 SWA_ARMS_EST 会随
       「优先格扩容」一起变大 ⇒ 末轮（remain≈65 min）**固定开销就超了预算**
       ⇒ selftest 的「缩臂后仍超预算」断言挂掉
       ⇒ 而 `shrink_plan` 在 main() 的 **try 之外** ⇒ 断言异常会**直接终止调度器**
         ⇒ 搜索会在截止前提前停（最坏情况）。
       ⇒ 固定预留必须**按剩余时间等比例收缩**：时间紧时**先砍 SWA**（它零训练、
         边际成本最低），优先格（离基线只差一档的核心资产）留到最后。
    """
    usable = max(0.0, remain_min - fixed_cost(remain_min))
    return int(max(0, min(SWA_ARMS_EST, (usable * 0.35) // EVAL_MIN_PER_ARM)))


def prio_reserve(remain_min: float, swa_arms: int) -> int:
    """优先格预留臂数：随剩余时间收缩，但**优先于**种子/突变被砍。"""
    usable = max(0.0, remain_min - fixed_cost(remain_min) - swa_arms * EVAL_MIN_PER_ARM)
    per_arm = TRAIN_MIN_PER_ARM + EVAL_MIN_PER_ARM
    return int(max(0, min(PRIORITY_ARMS_EST, (usable * 0.55) // per_arm)))


def shrink_plan(remain_min: float, n_seed: int, n_mut: int) -> tuple[int, int]:
    """按剩余时间缩臂：先砍突变格，再砍种子格（种子格保证稳健线 ⇒ 至少留 2 个）。

    ⚠️ 2026-11-12：优先格（PRIORITY_ARMS_EST 臂）**先占预算**，剩下的才分给种子格/突变格
    ⇒ 缩臂时不会把「离基线只差一档」的优先格挤出最后一轮。
    """
    per_arm = TRAIN_MIN_PER_ARM + EVAL_MIN_PER_ARM
    # 🟢 2026-11-13 10:15：合成臂（SWA）**只占评测时间**，但也占墙钟 ⇒ 缩臂时必须先扣掉
    #   （否则最后一轮会算不下 ⇒ selftest 的「缩臂后仍超预算」断言会挂）
    swa = swa_reserve(remain_min)
    prio = prio_reserve(remain_min, swa)
    avail = remain_min - fixed_cost(remain_min) - swa * EVAL_MIN_PER_ARM
    fit = int(max(4.0, avail / per_arm)) - prio
    fit = max(4, (fit // 4) * 4)
    # 🟢 2026-11-12：种子扫描臂的胜率 ≈ 突变的 3 倍 ⇒ 缩臂时**优先保种子**（原来 fit//3 → 现在 fit//2）
    seed_fit = max(2, min(SEED_SCAN_TARGET, fit // 2))
    mut_fit = max(1, min(MUT_CAP, fit - seed_fit))
    return seed_fit, mut_fit


def round_arm_budget(remain_min: float) -> int | None:
    """本轮**训练臂硬上限**（None = 时间充裕，不限）。

    🔴 2026-11-13 16:00：**末轮超预算的真修法**。
       旧逻辑 `shrink_plan()` 只缩 n_seed/n_mut，但 `build_train_arms()` 里两者都被
       `max(n_seed, SEED_SCAN_TARGET)` / `min(n_mut, MUT_CAP)` **覆盖** ⇒ 缩臂从未生效
       ⇒ 末轮照样铺 ~95 臂（实测一轮 59~61 臂 ≈ 4h15m）⇒ 最后一轮必然冲过 09:00
       ⇒ 官方三轨读数来不及出 ⇒ 夜里最好的候选白丢。
       ⇒ 这里直接给**训练臂总数**一个硬上限，砍的顺序按「价值」排：
         基线派生格（从未采样的 sw 插值区）> T3 前沿格 > 待复现配方 > 种子扫描
         > 近失格 > 其余优先格 > 突变格。
    """
    # 训练窗口 = 剩余时间 − 门禁预留 − 安全余量
    train_window = remain_min - GATE_RESERVE_MIN - SAFETY_MARGIN_MIN
    if train_window >= FULL_ROUND_MIN:
        return None                      # 时间够跑满一轮（91 臂）⇒ 不限臂
    # 合成臂（SWA，零训练）+ 训练后 SWA 臂：只占评测时间 ⇒ 先从窗口里扣掉
    swa = swa_reserve(remain_min)
    per_arm = TRAIN_MIN_PER_ARM + EVAL_MIN_PER_ARM / 4.0   # 评测 4 卡并行 ⇒ 摊薄
    usable = train_window - (swa + POST_SWA_ARMS_EST) * EVAL_MIN_PER_ARM / 4.0
    return int(max(4, usable // per_arm))


# ---------------------------------------------------------------------------
# 基线与单臂判据（沿用 v2，数值口径不变）
# ---------------------------------------------------------------------------
def _cell_of(a: dict) -> tuple:
    """臂的「配方格」指纹：sw（纳米侧权重）× thr/maxside（尺寸阈值）× lr_floor（lr 下限）。"""
    p = a.get("params") or {}
    return (p.get("sw"), p.get("thr"), p.get("maxside"), p.get("lr_floor"))


def cell_yield_table(led: dict, base: dict) -> dict:
    """按格统计**期望产出率** E[六项全过/臂] = p_triad × p_T1 × p_虚警 × p_T3。

    🔴 2026-11-13 11:55（台账 947 臂 / 128 个有官方读数的臂，全盘复算）：
      · A2 = sw4.0×thr12ms3×lrf0.1  E = **0.030**（n=13，4 个 5/6 臂）
      · A1 = sw5.0×thr12ms3×lrf0.1  E = **0.029**（n=8，2 个 5/6 臂）
      · B1 = sw5.0×thr16ms4×lrf0.0  E = 0.0004（n=52 ⇒ 只出 1 个 6/6 = 9863 本体）
      · C  = sw4.0×thr16ms4×lrf0.0  E = 0.000（n=6，p_triad = 0）
    ⇒ **官方三轨是最贵的一步（~12 min/臂，4 并发）⇒ 名额必须按 E 分配，不能按裸 nano**：
      旧逻辑「nano 前 6」会把名额几乎全喂给 B1 家族（E=0.0004）⇒ 每轮 6 个名额
      期望产出 ≈ 0.002 个六项全过臂；按 E 排则 ≈ 0.18 ⇒ **约 70 倍**。
    ⇒ 层级回退（精确格 n≥3 → 同 thr/maxside/lrf 家族 → 同 lrf → 全局）：
      新格（如 sw4.25×thr12ms3×lrf0.1）也能继承 A1/A2 的高 E，不会被当成未知而漏掉。
    """
    stat: dict = {}
    for _t, a in (led.get("arms") or {}).items():
        pk = a.get("picked") or {}
        of = a.get("official") or {}
        if not of or "skipped" in of or "stale" in of:
            continue
        if pk.get("nano") is None or of.get("t3") is None:
            continue
        d = stat.setdefault(_cell_of(a), {"n": 0, "triad": 0, "t1": 0, "fp": 0, "t3": 0})
        d["n"] += 1
        if ((pk.get("nano") or 0) >= (base.get("nano") or 0)
                and (pk.get("nano4") or 0) >= (base.get("nano4") or 0)
                and (pk.get("f1") or 0) >= (base.get("f1") or 0)):
            d["triad"] += 1
        if of.get("t1") is not None and of["t1"] >= (base.get("t1") or 0):
            d["t1"] += 1
        if of.get("t2_fp") is not None and of["t2_fp"] <= (base.get("t2_fp") or 1e9):
            d["fp"] += 1
        if of.get("t3") is not None and of["t3"] >= (base.get("t3") or 0):
            d["t3"] += 1

    def _y(d: dict) -> float:
        n = max(1, d["n"])
        return (d["triad"] / n) * (d["t1"] / n) * (d["fp"] / n) * (d["t3"] / n)

    def _pool_sum(pool: list) -> dict:
        return {"n": sum(x["n"] for x in pool), "triad": sum(x["triad"] for x in pool),
                "t1": sum(x["t1"] for x in pool), "fp": sum(x["fp"] for x in pool),
                "t3": sum(x["t3"] for x in pool)}

    fam: dict = {}
    lrf_fam: dict = {}
    for k, d in stat.items():
        fam.setdefault((k[1], k[2], k[3]), []).append(d)
        lrf_fam.setdefault(k[3], []).append(d)
    glob = _y(_pool_sum(list(stat.values()))) if stat else 0.0
    out: dict = {}
    for k, d in stat.items():
        if d["n"] >= 3:
            out[k] = _y(d)
            continue
        pool = fam.get((k[1], k[2], k[3])) or lrf_fam.get(k[3]) or []
        if pool:
            out[k] = 0.5 * _y(d) + 0.5 * _y(_pool_sum(pool))
        else:
            out[k] = glob
    return out


def base_ref(led: dict) -> dict:
    """取 Baseline_v2.0 的**精确读数**（常量是 6 位截断，会让基线臂自己判负）。"""
    base = dict(ns.BASE_V2)
    a98 = (led.get("arms") or {}).get(str(ns.BASE_V2["trial"]), {})
    pk = a98.get("picked") or {}
    of = a98.get("official") or {}
    for k in ("nano", "micro", "small", "regular", "f1", "nano4"):
        if pk.get(k) is not None:
            base[k] = pk[k]
    for k in ("t1", "t1_fp", "t2", "t2_fp", "t3", "t3_fp"):
        if of.get(k) is not None:
            base[k] = of[k]
    return base


def ckpt_path(trial: int, epoch: int) -> str:
    return f"{ns.ROOT}/trials/trial_{trial}/weights/epoch{epoch}.pt"


def core_checks(st: dict, base: dict) -> dict:
    """Beat-to-Pass 六项（ms 项另算，见 ns.phase_report）。"""
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    out = {}
    if pk.get("nano") is not None:
        out["nano"] = pk["nano"] >= base["nano"]
    if pk.get("nano4") is not None:
        out["nano4"] = pk["nano4"] >= base["nano4"]
    if pk.get("f1") is not None:
        out["f1"] = pk["f1"] >= base["f1"]
    if of.get("t1") is not None:
        out["t1"] = of["t1"] >= base["t1"]
    if of.get("t2_fp") is not None:
        out["t2fp"] = of["t2_fp"] <= base["t2_fp"]
    if of.get("t3") is not None:
        out["t3"] = of["t3"] >= base["t3"]
    return out


REQUIRED_CHECKS = ("nano", "nano4", "f1", "t1", "t2fp", "t3")


def full_checks(st: dict, base: dict) -> dict | None:
    """六项必须**齐全且全 True** 才算「打平基线」。

    🔴 2026-11-11 15:35 修：`core_checks` 对缺项是**跳过**（如官方三轨被门禁窗口跳过），
    于是 `all(chk.values())` 会把「只过了 nano/nano4/F1、根本没测过 T1/T2FP/T3」的臂
    误判成打平基线 ⇒ 假 beat + 白跑板卡。⇒ 这里强制六项齐全。
    """
    chk = core_checks(st, base)
    if any(k not in chk for k in REQUIRED_CHECKS):
        return None
    return chk


def champion_score(st: dict, base: dict) -> tuple:
    """单臂分数 = nano 主导 + 达标项奖励 − 违约项惩罚（仅用于排序/板卡靶子）。"""
    pk = st.get("picked") or {}
    if pk.get("nano") is None:
        return None, {}
    chk = core_checks(st, base)
    ok = sum(1 for v in chk.values() if v)
    # 缺项（官方三轨没测）按违约计 ⇒ 非门禁臂不会被高排序
    bad = len(REQUIRED_CHECKS) - ok
    dn = pk.get("nano", 0.0) - base["nano"]
    d4 = (pk.get("nano4") or 0.0) - base["nano4"]
    df = (pk.get("f1") or 0.0) - base["f1"]
    score = dn + 0.15 * d4 + 0.15 * df + 0.004 * ok - 0.010 * bad
    return score, chk


def pick_champion(led: dict, base: dict) -> dict | None:
    """选当前**单臂**最强者：必须有 picked 读数 + ckpt 文件真实存在。"""
    best = None
    for k, st in (led.get("arms") or {}).items():
        pk = st.get("picked") or {}
        if not pk or pk.get("epoch") is None:
            continue
        path = ckpt_path(int(k), int(pk["epoch"]))
        if not os.path.isfile(path):
            continue
        score, chk = champion_score(st, base)
        if score is None:
            continue
        cand = {"trial": int(k), "epoch": int(pk["epoch"]), "ckpt": path, "score": score,
                "checks": chk, "nano": pk.get("nano"), "nano4": pk.get("nano4"),
                "f1": pk.get("f1"), "micro": pk.get("micro"),
                "params": st.get("params") or {}, "desc": st.get("desc", "")}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


# ---------------------------------------------------------------------------
# ③ 配方级统计与冠军配方（v3 新增）
# ---------------------------------------------------------------------------
def recipe_key(p: dict) -> str:
    if not p:
        return ""
    return "|".join(f"{k}={p.get(k)}" for k in RECIPE_KEYS)


def used_recipe_seeds(led: dict) -> set:
    """台账里已用过的 **(配方, seed)** 对（**只算真训练臂**）。

    🟢🟢 2026-11-13 21:15：修「同一个 (配方, seed) 被反复重训」的巨大浪费。
      实测（最近 118 个臂）：**49 对 (配方, seed) 重复** ⇒ ≈41% 的 GPU 时间花在训练
      **完全相同**的模型上（同 init + 同 seed + 同配方 ⇒ 确定性同结果）。
      根因：某轮被截断后重跑该轮剩余臂时，`seed0 = SEED_BASE + round_idx*97` 仍按
      **旧轮号**算 ⇒ 与上一轮同配方同种子撞车（R13 里 12780/12849、12786/12855…）。
      ⇒ 修法：建臂时若 (配方, seed) 已存在，**顺延 seed** 直到未用过的组合。
    """
    out: set = set()
    for _t, _st in (led.get("arms") or {}).items():
        _p = _st.get("params") or {}
        _sd = _st.get("seed")
        if _sd is None:
            continue
        if _st.get("synth") or _p.get("synth"):
            continue          # SWA 合成臂：权重来源不同，不是重复训练
        out.add((recipe_key(_p), int(_sd)))
    return out


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def recipe_stats(led: dict) -> dict:
    """把「从 base init 起训」的臂按配方分组，给出每组的 seed 数与均值。

    ⚠️ 只收 `init == cs_t0043.pt` 的臂：SWA/Model-Soup 臂的 init 是 epoch*.pt，
       它们不在同一 loss 盆地（实测跨模型 soup 会崩），混进来会污染统计。
    """
    groups: dict[str, dict] = {}
    for k, st in (led.get("arms") or {}).items():
        p = st.get("params") or {}
        if p.get("init") != BASE_INIT or p.get("synth"):
            continue
        pk = st.get("picked") or {}
        if pk.get("nano") is None:
            continue
        gk = recipe_key(p)
        g = groups.setdefault(gk, {"key": gk, "n": 0, "nano": [], "nano4": [], "f1": [],
                                   "micro": [], "trials": [], "params": p})
        g["n"] += 1
        g["nano"].append(pk["nano"])
        g["nano4"].append(pk.get("nano4") or 0.0)
        g["f1"].append(pk.get("f1") or 0.0)
        g["micro"].append(pk.get("micro") or 0.0)
        g["trials"].append(int(k))
    for g in groups.values():
        g["mean_nano"] = _mean(g["nano"])
        g["mean_nano4"] = _mean(g["nano4"])
        g["mean_f1"] = _mean(g["f1"])
        g["mean_micro"] = _mean(g["micro"])
        g["max_nano"] = max(g["nano"])
        g["min_nano"] = min(g["nano"])
    return groups


def base_group(led: dict, groups: dict) -> dict | None:
    """基线配方组 = trial_9863 的配方组（9863 + 9912 两 seed ⇒ 稳健线 0.70097）。"""
    a98 = (led.get("arms") or {}).get(str(ns.BASE_V2["trial"]), {})
    k = recipe_key(a98.get("params") or {})
    return groups.get(k)


def robust_score(g: dict, bg: dict | None) -> float:
    """稳健分数 = 配方均值相对基线配方均值的加权差（nano 主导 + nano4/F1 修正）。"""
    if not bg:
        return 0.0
    return ((g["mean_nano"] - bg["mean_nano"])
            + 0.15 * (g["mean_nano4"] - bg["mean_nano4"])
            + 0.15 * (g["mean_f1"] - bg["mean_f1"]))


def best_arm_of(led: dict, g: dict, base: dict | None = None) -> dict | None:
    """组内 nano 最高的臂（且 ckpt 存在）。

    🟢 2026-11-12：若给了 base，则**优先选「三项全平（nano/nano4/F1 ≥ 基线）」的臂**。
      冠军臂会被用来做 SWA 合成与板卡测速 ⇒ 选一个 F1 塌掉的臂做 SWA 会把好配方带偏
      （实测 swl0.2 组里 nano 最高的是 12229：nano 0.7413 但 F1 0.8853；
       而 12300：nano 0.7214 / F1 0.8904 三项全平 ⇒ 更该当冠军臂）。
    """
    best = None
    best_triad = None
    for t in g["trials"]:
        st = (led.get("arms") or {}).get(str(t)) or {}
        pk = st.get("picked") or {}
        if pk.get("nano") is None or pk.get("epoch") is None:
            continue
        path = ckpt_path(t, int(pk["epoch"]))
        if not os.path.isfile(path):
            continue
        cand = {"trial": t, "epoch": int(pk["epoch"]), "ckpt": path,
                "nano": pk["nano"], "nano4": pk.get("nano4"), "f1": pk.get("f1"),
                "micro": pk.get("micro"), "params": st.get("params") or {},
                "desc": st.get("desc", "")}
        if best is None or cand["nano"] > best["nano"]:
            best = cand
        if base is not None:
            if ((pk.get("nano") or 0) >= base.get("nano", 0.0)
                    and (pk.get("nano4") or 0) >= base.get("nano4", 0.0)
                    and (pk.get("f1") or 0) >= base.get("f1", 0.0)):
                if best_triad is None or cand["nano"] > best_triad["nano"]:
                    best_triad = cand
    return best_triad or best


# ────────────────────────────────────────────────────────────────────────────
# 🔴 T2 安全闸门（2026-11-12 09:20 实测发现，见 memory §149.6）
#   实测证据（23 个有官方三轨读数的臂，swl 单变量扫描，sw∈{5,6,8}）：
#     swl=1.0 → T2 0.9940~0.9987 ｜ swl=0.5 → T2 0.9892~0.9978
#     swl=0.35 → 0.9571 ｜ swl=0.3 → 0.9516~0.9625 ｜ swl=0.2 → 0.9229~0.9307
#     swl=0.1  → 0.7954~0.8512（**死区**）
#   ⇒ **大目标降权（swl<1.0）单调毁掉官方 T2 金池**（基线 T2 = 0.997440）；
#     而 NanoBench（no-GMC）**完全看不见**这个塌陷（no-GMC 上 nano 反而涨）。
#   ⇒ 只按 nano 选冠军 ⇒ 整夜算力全喂进「nano 高但官方必败」的死区
#     （昨夜 375 臂 + 今晨 triad 4 臂全部如此 ⇒ 0 个能升轨）。
#   ⇒ **冠军配方必须官方 T2 已验证不塌**；未验证的格**不准**当冠军
#     ⇒ 搜索中心自动回落到基线配方（sw5.0 × swl1.0）⇒ 突变落在 T2 安全区。
# ────────────────────────────────────────────────────────────────────────────
T2_SAFE_SLACK = 0.002      # T2 容差：基线 − 0.002 内视为未塌


def group_t2(led: dict, g: dict) -> float | None:
    """组内所有臂的官方 T2 最大值（None = 该组从未跑过官方三轨 ⇒ 未验证）。"""
    vals = []
    for t in g.get("trials") or []:
        of = ((led.get("arms") or {}).get(str(t)) or {}).get("official") or {}
        if of.get("t2") is not None:
            vals.append(float(of["t2"]))
    return max(vals) if vals else None


def t2_safe(led: dict, g: dict, base: dict) -> bool:
    """该配方组是否**官方 T2 安全**（已验证且未塌）。"""
    t2 = group_t2(led, g)
    if t2 is None:
        return False
    return t2 >= (base.get("t2") or 0.0) - T2_SAFE_SLACK


def recipe_t2_dead(led: dict, g: dict, base: dict) -> bool:
    """该配方是否已被证伪「官方 T2 塌陷」⇒ **不值得再花算力复现**。

    判据两条（任一命中即判死）：
      ① `swl < 1.0`（大目标降权）—— 实测单调塌陷，属**结构性**死区；
      ② 已有官方读数且 T2 < 基线 − 容差。
    ⚠️ 与 `t2_safe` 的区别：`t2_safe` 要求「**已验证安全**」（可作冠军）；
       这里判的是「**已证伪**」（不可复现）—— 未验证的格**不算死**。
    """
    p = g.get("params") or {}
    if float(p.get("swl") or 1.0) < 1.0:
        return True
    t2 = group_t2(led, g)
    if t2 is not None and t2 < (base.get("t2") or 0.0) - T2_SAFE_SLACK:
        return True
    return False


def swl_safe_group(g: dict) -> bool:
    """🟢 2026-11-12：**冠军升格硬闸门** swl >= 1.0。

    实测（§150.2，全量 403 臂重算）：swl<1.0 的格官方 T1 ≤ 0.888、T3 ≤ 0.936
    ⇒ 官方三轨必败 ⇒ 绝不能当搜索中心（否则整夜突变格都继承 swl<1.0 ⇒ 算力全喂死区）。
    ⇒ 「观测安全」（t2_safe）与「结构安全」（swl 硬闸门）不是一回事，这里补结构这一层。
    """
    swl = (g.get("params") or {}).get("swl")
    if swl is None:
        return True
    try:
        return float(swl) >= 1.0
    except (TypeError, ValueError):
        return True


def pick_champion_recipe(led: dict, base: dict) -> dict:
    """选**冠军配方**：只认 n≥2 且稳健分数 > 0 的组；否则回落到基线配方（9863）。

    ⇒ 单臂运气臂不会篡改搜索方向（v2 的老问题）。
    """
    groups = recipe_stats(led)
    bg = base_group(led, groups)
    adopted = None
    for g in groups.values():
        if bg and g["key"] == bg["key"]:
            continue
        if g["n"] < 2:
            continue
        # 🟢 2026-11-12：swl<1.0 的格**硬性**不得升格为冠军（见 swl_safe_group）
        if not swl_safe_group(g):
            continue
        # 🔴 主指标必须**真的更好**（同配方多 seed 的均值 nano）才准升格。
        #   实测反例：sw4.0 组（9814+9913）均值 nano 0.6986 < 基线 0.7010，
        #   但 nano4(+1.68 pp)/F1(+0.31 pp) 更好 ⇒ 只看加权分会被误升格，
        #   搜索方向就整体滑到 sw4.0 上（那是**均值更差**的配方）。
        if bg and g["mean_nano"] <= bg["mean_nano"]:
            continue
        if bg and g["mean_nano4"] < bg["mean_nano4"] - 0.002:
            continue
        if bg and g["mean_f1"] < bg["mean_f1"] - 0.001:
            continue
        # 🔴 2026-11-12 09:22：冠军格的**最佳单臂必须达到基线严格线**
        #   （实测 sw4.0 组均值 0.6986 > 基线配方均值 0.6966，但**最佳单臂 0.7004 < 严格线 0.708648**）
        #   ⇒ 拿它当搜索中心 ⇒ 突变全在「够不到基线」的格上 ⇒ 白烧算力。
        if g.get("max_nano", 0.0) < (base.get("nano") or 0.0) - 0.001:
            continue
        # 🔴 T2 安全闸门（2026-11-12 09:20）：该格必须**官方 T2 已验证不塌**
        #   （swl<1.0 的格 T2 单调塌到 0.80~0.96 ⇒ 官方必败 ⇒ 绝不能当搜索中心）。
        if not t2_safe(led, g, base):
            continue
        s = robust_score(g, bg)
        if s <= 0:
            continue
        if adopted is None or s > adopted[0]:
            adopted = (s, g)
    # 🟢 「单臂最佳格」直通通道（2026-11-11 17:45 新增）
    #   问题：基线配方组的种子越多，其**均值**越被拉低（9863 的 0.7086 含运气；
    #   同配方 10 个 seed 的均值 ≈ 0.69）⇒ 任何 n=2 的格只要均值高一点就会「误升格」，
    #   搜索中心被拖到**均值更差**的配方上（实测：sw4.0 组 0.6986 > sw5.0 组新均值 ⇒ 被选中）。
    #   修正：当某格的**单臂最佳读数**比基线配方的最佳读数高 **≥0.5 pp**，
    #   且**同一臂有 ≥2 个 epoch 都过基线线**（臂内一致 ⇒ 不是挑 epoch 的运气），
    #   则允许它当搜索中心（搜索中心要跟着**最强的格**走，而不是被均值稀释机制拖走）。
    bg_max = bg["max_nano"] if bg else base.get("nano", 0.0)
    best_cell = None
    for g in groups.values():
        if bg and g["key"] == bg["key"]:
            continue
        # 🟢 2026-11-12：直通通道同样受 swl 硬闸门约束
        if not swl_safe_group(g):
            continue
        for t in g["trials"]:
            st = (led.get("arms") or {}).get(str(t)) or {}
            traj = st.get("nanobench") or {}
            vals = [v.get("nano") for v in traj.values() if v.get("nano") is not None]
            if len(vals) < 2:
                continue
            over = [v for v in vals if v >= bg_max]
            if not (len(over) >= 2 and max(vals) >= bg_max + 0.005):
                continue
            # 🟢 2026-11-12 修（实测）：直通通道若只看 nano，搜索中心会滑进
            #   「nano 高但 F1 塌掉」的**死区**（实测 swl0.1 组：nano 0.7578 / nano4 0.6793，
            #   但 F1 0.8777、官方 T2 0.830 ⇒ 官方三轨必败 ⇒ 整夜算力白烧）。
            #   ⇒ 直通通道必须**兼容 F1/nano4 底线**：臂的 picked epoch 三项全过基线，
            #     或该臂有 ≥2 个 epoch 三项全过基线。
            pk = st.get("picked") or {}
            triad_ok = ((pk.get("nano") or 0) >= base.get("nano", 0.0)
                        and (pk.get("nano4") or 0) >= base.get("nano4", 0.0)
                        and (pk.get("f1") or 0) >= base.get("f1", 0.0))
            full_epochs = sum(1 for v in traj.values()
                              if (v.get("nano") or 0) >= base.get("nano", 0.0)
                              and (v.get("nano4") or 0) >= base.get("nano4", 0.0)
                              and (v.get("f1") or 0) >= base.get("f1", 0.0))
            if not (triad_ok or full_epochs >= 2):
                continue
            # 🔴 同上：直通通道也必须 T2 安全（否则等于开了一条通往死区的捷径）
            if not t2_safe(led, g, base):
                continue
            if best_cell is None or max(vals) > best_cell[0]:
                best_cell = (max(vals), g)
            break
    if best_cell and (adopted is None or best_cell[0] > (adopted[1].get("max_nano") or 0.0)):
        adopted = (robust_score(best_cell[1], bg), best_cell[1])

    champ_g = adopted[1] if adopted else bg
    arm = best_arm_of(led, champ_g, base) if champ_g else None
    if arm is None:
        arm = pick_champion(led, base)
    params = dict(ns.BASE)
    if champ_g:
        for k in CHAMP_KEYS:
            if k in (champ_g["params"] or {}):
                params[k] = champ_g["params"][k]
    return {"group": champ_g, "base_group": bg, "arm": arm, "params": params,
            "adopted": adopted is not None,
            "robust_score": adopted[0] if adopted else 0.0}


def promotion_candidates(led: dict, base: dict) -> list[dict]:
    """列出**可以升轨**的候选配方（明早只需跑一次 promote_candidate.py 即可）。

    判据（三关全过才算数）：
      ① 同一配方 **n ≥ 2 个 seed**（单臂打平可能只是种子运气，§140.2）；
      ② 组内**至少一个臂六项全平基线**（nano / nano4 / F1 / T1 / T2FP / T3）；
      ③ 组内均值 nano **不低于**基线配方均值（不许拿均值更差的配方去升格）。
    """
    groups = recipe_stats(led)
    bg = base_group(led, groups)
    out = []
    for g in groups.values():
        if g["n"] < 2:
            continue
        # 🟢 2026-11-12：swl<1.0 的配方不得进升轨候选名单（官方 T1/T3 必败）
        if not swl_safe_group(g):
            continue
        # 🟢 2026-11-12 09:27：基线配方组**不再整组排除** ——
        #   排除的只是「基线臂本身（9863）」；同配方的**新臂**（不同 seed / 不同 epoch）
        #   只要六项全平基线，就是合法的新基线候选（该配方本身已 n=8 复现 ⇒ ①自动满足）。
        #   ⚠️ 原逻辑把整组排除 ⇒ 种子扫描臂（每轮 6 个基线配方臂）**永远无法升轨** ⇒ 纯浪费。
        is_base_group = bool(bg and g["key"] == bg["key"])
        winners = []
        for t in g["trials"]:
            if is_base_group and int(t) == int(ns.BASE_V2["trial"]):
                continue          # 基线臂本身不参与升轨
            st = (led.get("arms") or {}).get(str(t)) or {}
            chk = full_checks(st, base)
            if chk and all(chk.values()):
                winners.append(t)
        if not winners:
            continue
        # 🟢🟢🟢 2026-11-13 22:40：**均值护栏对「已产出六项全过臂」的格让路**
        #   背景（实测）：护栏「组均值 nano ≥ 基线配方组均值」会把**离六项最近的两格**直接挡死：
        #     · C = sw4.0×thr16ms4×lrf0（12702/12703 两个 **5/6** 臂，T3 只差 2.4e-5）均值 nano 0.692344
        #       < 基线组均值 0.694066 ⇒ **即使它出六项全过臂也会被护栏否掉**
        #     · B2 = sw5.0×thr16ms4×lrf0.1（12053 的 5/6 配方）均值 0.689159 ⇒ 同样被挡
        #   根因：均值是**历史臂的均值**（含大量中庸臂），而候选臂本身六项全过 ⇒ 均值代理与直接证据矛盾。
        #   ⇒ 新规则：**先看有没有六项全过臂**；有 ⇒ 均值护栏让路（并在条目里标注命中数）；
        #     没有 ⇒ 均值护栏照旧（防止用均值更差的配方去升格）。
        #   ⚠️ 六项判据本身**一点没放松**；放松的只是「均值代理否决」。
        #   ⇒ 组均值 nano 这条护栏**只对「没有六项全过臂」的格生效**（上面的 `if not winners: continue`
        #   已经保证了这一点）⇒ 此处不再重复判断。
        if bg and g["mean_nano4"] < bg["mean_nano4"] - 0.002:
            continue
        if bg and g["mean_f1"] < bg["mean_f1"] - 0.001:
            continue
        out.append({"recipe": g["key"], "n": g["n"], "mean_nano": g["mean_nano"],
                    "mean_nano4": g["mean_nano4"], "mean_f1": g["mean_f1"],
                    "mean_micro": g["mean_micro"], "trials": g["trials"],
                    "beat_trials": winners, "n_winners": len(winners),
                    "single_hit": len(winners) == 1,
                    "params": g["params"]})
    out.sort(key=lambda r: -r["mean_nano"])
    return out


# ---------------------------------------------------------------------------
# ① 合成臂（零训练）：只出 SWA（跨模型 soup 实测会崩，默认关闭）
# ---------------------------------------------------------------------------
def _lawa_weights(n: int, lam: float = 0.5) -> list[float]:
    """LAWA 式权重：w_i = lam^(n-1-i)（靠后 epoch 权重更大）。

    ⚠️ λ=0.75 实测**太温和**：等权与加权平均的首个浮点张量 L2 差只有 4.95e-07
    ⇒ 评测读数几乎逐位相同 ⇒ 白占评测算力。⇒ 改 **λ=0.5**
    （2 元素 [0.5,1.0] ⇒ 末 epoch 占 67%；5 元素 ⇒ 末 epoch 占 51%），
    与等权平均是**实质不同**的权重空间点。
    """
    n = max(1, int(n))
    return [round(lam ** (n - 1 - i), 6) for i in range(n)]


def _swa_key(mem, weights=None) -> tuple:
    """SWA 臂的去重键：**成员集合 + 权重签名**（同成员不同权重 ⇒ 不同模型，不能互相去重）。"""
    ws = None if not weights else tuple(round(float(w), 6) for w in weights)
    return (tuple(sorted(str(x) for x in mem)), ws)


def _swa_windows(cdir: str, win: int) -> list[tuple[str, list[int], str]]:
    """按**该臂真实存在的 epoch 文件**给出 4 个**互不相同**的 SWA 窗口。

    🔴 2026-11-13 09:35 修（R12 实测）：原窗口是固定区间
    （neigh=win-2~win+2 / late=win~win+4 / wide=win-2~win+6 / early=win-2~win+1），
    而训练只有 5 个 epoch ⇒ neigh / wide / early **全部塌成同一组 {2,3,4,5}**
    ⇒ 同一份平均权重被重复评测 3 次，还白占 3/8 个官方门禁名额
    （12702 / 12704 / 12705 六项读数逐位相同 ⇒ 实为同一个模型）。
    ⇒ 改为按**可用 epoch 列表**切出 4 个**互异**窗口（late=末 2 / neigh=末 3 / early=首 3 / wide=全部）。
    """
    avail: list[int] = []
    try:
        for fn in os.listdir(cdir):
            if fn.startswith("epoch") and fn.endswith(".pt"):
                try:
                    avail.append(int(fn[5:-3]))
                except ValueError:
                    continue
    except OSError:
        avail = []
    avail = sorted(set(avail))
    if len(avail) < 2:
        return []
    if len(avail) == 2:
        return [("swa_late", list(avail), "champion arm 全部 epoch（只有 2 个）", None),
                ("swa_late_w", list(avail), "champion arm 全部 epoch（**加权**：靠后 epoch 权重更大）",
                 _lawa_weights(len(avail)))]
    if len(avail) == 3:
        return [("swa_late", avail[-2:], "champion arm 末 2 个 epoch（实测最有效）", None),
                ("swa_early", avail[:2], "champion arm 首 2 个 epoch 早窗", None),
                ("swa_wide", list(avail), "champion arm 全部 3 个 epoch", None),
                ("swa_late_w", avail[-2:], "末 2 个 epoch（**加权** LAWA）", _lawa_weights(2)),
                ("swa_wide_w", list(avail), "全部 3 个 epoch（**加权** LAWA）", _lawa_weights(3))]
    # 🟢 2026-11-13 10:40：**相邻二元窗**（前沿臂的最佳 epoch 常在中间，如 12519 ep3）
    #   依据（11-13 09:24 全盘审计）：前沿臂 12519（ep3）/ 12679（ep3）**只差 T3 一线**
    #   （−1.15e-4 / −1.57e-4），而 §153.12 实测 SWA 能把 T3 顶到**只差 2.4e-5**（12705）
    #   ⇒ SWA 是**零训练**拿下六项的最好杠杆。
    #   三元/四元窗会把「最佳 epoch + 邻居」的平均稀释 ⇒ 补上**相邻二元窗**
    #   （{2,3} / {3,4}）：只把最佳 epoch 和它的一个邻居平均 ⇒ 保留 T3 优势。
    #   ⚠️ 长训练臂（epochs=20 ⇒ 20 个 epoch 文件）不能把「全部 epoch」平均进来
    #   ⇒ 只围绕 picked epoch `win` 取 4 个相邻 epoch 当核心集。
    if len(avail) <= 6:
        core = list(avail)
    else:
        i = avail.index(win) if win in avail else len(avail) - 1
        core = sorted(set(avail[max(0, i - 1): max(0, i - 1) + 4]))
    cands = [("swa_late", core[-2:], "末 2 个 epoch（实测最有效）", None),
             ("swa_neigh", core[-3:], "末 3 个 epoch", None),
             ("swa_early", core[:3], "首 3 个 epoch 早窗", None),
             ("swa_wide", list(core), "全部核心 epoch 宽窗", None)]
    for i in range(len(core) - 1):
        cands.append(("swa_p%d" % i, [core[i], core[i + 1]],
                      "相邻二元窗 epoch%d+%d（保留最佳 epoch 的 T3 优势）" % (core[i], core[i + 1]), None))
    # 🟢🟢 2026-11-13 13:40 新增：**LAWA 式加权 SWA**（等权平均的加权变体，零训练成本）
    #   依据（§153.16.2）：等权 SWA 已经能把 F1/T1/虚警顶到很高（12702 F1 0.8947 / T1 0.8913 / 虚警 2），
    #   只差 nano4 与 T3；而 **后期 epoch 的权重更高**（LAWA）能保留「末段权重更贴基线分布」的偏置，
    #   同时仍保留平均带来的方差削减 ⇒ 在 nano/nano4/F1/T3 四项之间多探几个点。
    #   权重：w_i = 0.75^(n-1-i)（靠后 epoch 权重更大，等比 λ=0.75，温和偏置）。
    cands.append(("swa_late_w", core[-2:], "末 2 个 epoch（**加权** LAWA λ=0.5）", _lawa_weights(2)))
    cands.append(("swa_neigh_w", core[-3:], "末 3 个 epoch（**加权** LAWA λ=0.5）", _lawa_weights(3)))
    cands.append(("swa_wide_w", list(core), "全部核心 epoch（**加权** LAWA λ=0.5）", _lawa_weights(len(core))))
    out, seen = [], set()
    for name, eps, note, wts in cands:
        key = (tuple(eps), None if not wts else tuple(round(float(w), 6) for w in wts))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, list(eps), note, wts))
    return out


def frontier_swa_sources(led: dict, base: dict, k: int = 3) -> list[dict]:
    """挑「离六项全过最近」的**历史臂**当 SWA 源（**零训练成本**）。

    🟢 2026-11-13 09:50 动机（当日审计 + SWA 实测）：
      · 台账上离六项全过最近的臂是 12519 / 12679 / 12694 / 12520（均 5/6，只差 T3 1.2e-4~1.8e-3）
      · SWA（同臂相邻 epoch 等权平均）实测能把 T3 顶到**只差 2.4e-5**（12705 = 12639 的 SWA）
      ⇒ 对这些**前沿臂**做 SWA（4 个互异窗口）有机会**不训练**就造出六项全过的候选
        （前沿臂的 nano4 有 +0.008~+0.010 富余，正好抵消 SWA 对 nano4 的损耗）。
    硬约束：① 必须有**完整官方读数**（没测过 T1/T3 的不算）
             ② swl >= 1.0（swl<1.0 是 T1/T3 死区）
             ③ 排序：过项数多 → 归一化亏损小。
    """
    out: list[dict] = []
    for t, st in (led.get("arms") or {}).items():
        if st.get("synth"):
            continue
        of = st.get("official") or {}
        pk = st.get("picked") or {}
        if of.get("t1") is None or pk.get("nano") is None:
            continue
        p = st.get("params") or {}
        swl = p.get("swl")
        swl = 1.0 if swl is None else float(swl)
        if swl < 1.0 - 1e-9:
            continue
        ep = pk.get("epoch")
        ck = f"{ns.ROOT}/trials/trial_{t}/weights/epoch{ep}.pt"
        if ep is None or not os.path.isfile(ck):
            continue
        # ⚠️ 基线臂本体（9863）不做 SWA：它不是「新配方」，
        #   且它的 SWA 会白占官方门禁名额（真赢要靠新配方赢）。
        if int(t) == int(ns.BASE_V2["trial"]):
            continue
        npass, deficit = _near_miss_score(st, base)
        if npass < 4:
            continue
        out.append({"trial": int(t), "epoch": int(ep), "ckpt": ck, "params": p,
                    "npass": npass, "deficit": deficit})
    out.sort(key=lambda s: (-s["npass"], s["deficit"]))
    return out[:k]


def build_synth_arms(led: dict, champ: dict, base: dict, round_idx: int,
                     next_trial: list[int], cross_soup: bool = False,
                     extra_sources=None) -> list[tuple]:
    """生成合成臂并把权重写到 trials/trial_X/weights/epoch5.pt。

    🔴 2026-11-11 14:46 实测：**跨配方/跨种子的 Model Soup 在本任务上会崩**
       （soup_multiseed nano 0.2422 · soup_dim_top2 0.3046 · soup_same_sw 0.4627 · soup_top3 0.6545，
        全部远低于基线 0.7086）⇒ 原因是这些权重**不在同一 loss 盆地**，平均互相抵消。
    🟢 **同一臂相邻 epoch 的 SWA 成立**（swa_late nano 0.708885 **已略超基线**）。
    ⇒ 故默认 **只出 SWA**；`cross_soup=True` 才额外生成跨模型 soup。
    """
    arms = []
    # 🟢 2026-11-12 13:10：**去重** —— 同一 (源臂, 窗口) 的 SWA 已经做过就别再做一遍
    #   实测：冠军臂不变时，每轮都会把**同一组权重**重新平均 + 重新评测一遍
    #   ⇒ 纯浪费评测算力（每轮 ~4 臂 × 1.3 min）。
    _done = set()
    for _t, _st in (led.get("arms") or {}).items():
        _mem = _st.get("synth_members") or []
        if _mem:
            _done.add(_swa_key(_mem, _st.get("synth_weights")))
    # 🟢 2026-11-13 09:50：源臂 = **冠军臂 + 前沿臂**（near-miss 家族）
    #   ⇒ 每轮多出 3×4 = 12 个**零训练**的 SWA 候选，专门冲「只差 T3 一线」的前沿臂。
    sources: list[tuple[str, dict]] = []
    if champ:
        sources.append(("champ", champ))
    for _s in (extra_sources or []):
        sources.append((f"f{_s['trial']}", _s))
    if not sources:
        return arms
    for tag, src in sources:
        cdir = os.path.dirname(src["ckpt"])
        win = src["epoch"]
        sp = dict(src.get("params") or {})
        for name, eps, note, wts in _swa_windows(cdir, win):
            mem = [f"{cdir}/epoch{e}.pt" for e in eps]
            mem = [m for m in mem if os.path.isfile(m)]
            if len(mem) < 2:
                continue
            if _swa_key(mem, wts) in _done:
                log(f"[synth] {tag}_{name} 跳过（该窗口已存在，避免重复评测）")
                continue
            trial = next_trial[0]
            next_trial[0] += 1
            out = f"{ns.ROOT}/trials/trial_{trial}/weights/epoch{SYNTH_EPOCH}.pt"
            try:
                average_checkpoints(mem, out, epoch=SYNTH_EPOCH, weights=wts,
                                    note=f"SWA {tag}_{name}: {note}")
            except Exception as e:  # noqa: BLE001
                log(f"[synth] {tag}_{name} 放弃：{e}")
                continue
            # 🟢 2026-11-13 09:35：**本轮内**也去重（跨轮的 _done 只覆盖历史窗口）
            #   实测（R12）：训练只有 5 个 epoch ⇒ 三个窗口（neigh/wide/early）都塌成 epoch{2,3,4,5}
            #   ⇒ 同一份平均权重被**重复评测 3 次**，还占了 3/8 个官方门禁名额（12702/12704/12705 读数逐位相同）。
            _done.add(_swa_key(mem, wts))
            st = ns.arm_state(led, trial)
            st.update(trial=trial, desc=f"★ 合成臂 {tag}_{name}：{note}", trained=True, synth=True,
                      synth_members=mem, synth_note=note, synth_weights=list(wts) if wts else None,
                      params={"init": mem[0], "synth": f"{tag}_{name}", "epochs": SYNTH_EPOCH,
                              "sched": SYNTH_EPOCH,
                              "sw": sp.get("sw", 5.0), "mix": sp.get("mix", "w2.0"),
                              "warmup": sp.get("warmup", 1.5), "lr": sp.get("lr", 3.5),
                              "fb": sp.get("fb", 2.75), "ow": sp.get("ow", 0.4),
                              "wd": sp.get("wd", 0.000269), "mosaic": sp.get("mosaic", 0.15),
                              "scale": sp.get("scale", 0.1), "translate": sp.get("translate", 0.05),
                              "clip": sp.get("clip", 5.65), "backbone_lr": sp.get("backbone_lr", 1.0),
                              "uf": sp.get("uf", 0), "head_src": sp.get("head_src", "p3"),
                              # 🟢 2026-11-12 12:45：swl 之前**硬编码 1.0** ⇒ 死亡区冠军的 SWA 产物
                              #   被错标成「swl 安全」⇒ 现在跟随源臂真实 swl（口径诚实）
                              "native_grid": sp.get("native_grid", 0), "mr": sp.get("mr", 1),
                              "swl": sp.get("swl", 1.0), "thr": sp.get("thr", 16.0),
                              "maxside": sp.get("maxside", 4.0),
                              "lr_floor": sp.get("lr_floor", 0.0), "hf": sp.get("hf", 0),
                              "hfsrc": sp.get("hfsrc", "b1"), "hfch": sp.get("hfch", 1)})
            log(f"[synth] {tag}_{name} ⇒ trial {trial}（成员 {len(mem)} 个）")
            arms.append((trial, 20000 + trial, st["params"], st["desc"]))
    ns.save_ledger(led)
    return arms


# ---------------------------------------------------------------------------
# ② 训练臂：种子扫描 + 突变格（**全部从 base init 起训**）
# ---------------------------------------------------------------------------
def clean_recipe(params: dict) -> dict:
    p = dict(ns.BASE)
    for k in CHAMP_KEYS:
        if k in (params or {}):
            p[k] = params[k]
    p["init"] = BASE_INIT
    return p


def tried_cells(led: dict) -> set:
    """已试过的配方键（**只有跑出读数的才算试过** ⇒ 被中断的臂不占格）。"""
    tried = set()
    for _k, st in (led.get("arms") or {}).items():
        p = st.get("params") or {}
        if p.get("init") == BASE_INIT and not p.get("synth") and (st.get("picked") or st.get("nanobench")):
            tried.add(recipe_key(p))
    return tried


def generate_mutations(led: dict, champ_params: dict, n: int) -> list[tuple]:
    """固定队列用完后的**系统网格**扩展（保证整晚都有新格可试）。

    网格顺序按先验排：sw 先在 5.0 附近细探 ⇒ 小目标定义逐档 ⇒ 再换 sw；
    再是 lr×LR 保底、warmup×focal、offset×混样三层网格。
    全部跳过台账里已跑出读数的配方键。

    🔴 2026-11-12 09:20：**网格里不再有 swl<1.0 的格**（实测 swl<1.0 ⇒ 官方 T2
    单调塌到 0.80~0.96 ⇒ 官方必败 ⇒ 整格作废）⇒ swl 固定为 1.0。
    """
    tried = tried_cells(led)
    out: list[tuple] = []

    def _take(patch: dict, name: str, why: str) -> bool:
        if len(out) >= n:
            return False
        probe = dict(champ_params)
        probe.update(patch)
        rk = recipe_key(probe)
        if rk in tried:
            return True
        tried.add(rk)
        out.append((name, patch, why))
        return True

    # 网格以**冠军配方**为中心（2026-11-11 18:15）：原来固定从 sw5.0 / swl0.5 起 ⇒
    #   冠军变成 sw5.0×swl0.3 之后，系统网格会绕着 swl0.5 那条老路走 ⇒ 偏靶。
    c_sw = champ_params.get("sw")
    c_swl = champ_params.get("swl")
    sws = [c_sw] + [x for x in [5.0, 4.5, 5.5, 4.0, 6.0, 7.0, 3.5, 8.0, 10.0, 12.0] if x != c_sw]
    # 🔴 swl 固定 1.0：swl<1.0 的格官方 T2 必塌（见 docstring）
    swls = [1.0]
    thrdefs = [(16.0, 4.0), (18.0, 5.0), (25.0, 6.0), (32.0, 8.0), (20.0, 5.0), (12.0, 3.0)]
    for sw in sws:
        for swl in swls:
            for thr, ms in thrdefs:
                if not _take({"sw": sw, "swl": swl, "thr": thr, "maxside": ms},
                             f"grid_sw{sw}_swl{swl}_thr{int(thr)}",
                             f"系统网格：sw={sw} × swl={swl} × 小目标定义 thr{int(thr)}/maxside{int(ms)}"):
                    return out
    for lr in [3.0, 4.0, 2.8, 3.2, 4.2]:
        for lrf in [0.05, 0.1, 0.2]:
            if not _take({"lr": lr, "lr_floor": lrf}, f"grid_lr{lr}_lrf{lrf}",
                         f"系统网格：lr_scale={lr} × LR 保底 {lrf:.0%}"):
                return out
    for wu in [1.0, 2.0, 1.2, 2.5]:
        for fb in [3.0, 2.5]:
            if not _take({"warmup": wu, "fb": fb}, f"grid_wu{wu}_fb{fb}",
                         f"系统网格：warmup={wu} × focal beta={fb}"):
                return out
    for ow in [0.5, 0.3, 0.6]:
        for mx in ["w2.2", "w2.5", "w1.8"]:
            if not _take({"ow": ow, "mix": mx}, f"grid_ow{ow}_{mx}",
                         f"系统网格：offset 权重={ow} × 混样 {mx}"):
                return out
    return out


def next_mutations(led: dict, champ_params: dict, n: int) -> list[tuple]:
    """从突变格队列里取 n 个**没试过**的格（固定队列用完 ⇒ 转系统网格生成器）。"""
    tried = tried_cells(led)
    out = []
    for name, patch, why in MUTATIONS:
        if len(out) >= n:
            break
        probe = dict(champ_params)
        probe.update(patch)
        rk = recipe_key(probe)
        if rk in tried:
            continue
        tried.add(rk)
        out.append((name, patch, why))
    if len(out) < n:
        # 固定队列（43 格）用完 ⇒ 用系统网格继续，保证整晚都有新格可试
        out += generate_mutations(led, champ_params, n - len(out))
    return out


def priority_cells(led: dict, base_params: dict) -> list[tuple[str, dict, str]]:
    """本轮要喂的**优先格**（已过淘汰线的）。

    淘汰线：某格攒到 PRIORITY_MIN_N 个 seed 后，若其均值 nano 仍低于基线配方均值
    PRIORITY_MEAN_SLACK 以上 ⇒ 该格判死，不再每轮喂 2 个 seed（把算力让给好格）。

    🟢 2026-11-13 09:35 加**去重**：同一配方格只喂一次。
       原因：优先格是「补丁」叠加在冠军配方上 ⇒ 冠军一旦变成 thr12ms3 家族，
       「thr12ms3 / sw4.0 / sw4.0_thr12ms3 / lrfloor0.10_thr12ms3」四个补丁
       会**塌成同一个配方**（R12 实测每轮浪费 8 臂，喂的还是冠军自己）。
       ⇒ 去重后省下的算力全部转投 T3 前沿格。
    """
    groups = recipe_stats(led) or {}
    bg = base_group(led, groups)
    bmean = (bg or {}).get("mean_nano") or 0.0
    out = []
    seen = {recipe_key(base_params)}     # 冠军配方本身由「种子扫描」覆盖，不重复喂
    for name, patch, why in PRIORITY_PATCHES:
        p = dict(base_params)
        p.update(patch)
        p["epochs"] = max(5, int(p.get("sched") or 5))
        k = recipe_key(p)
        if k in seen:
            continue
        seen.add(k)
        grp = groups.get(k)
        if (grp and grp.get("n", 0) >= PRIORITY_MIN_N
                and (grp.get("mean_nano") or 0.0) < bmean - PRIORITY_MEAN_SLACK):
            continue
        out.append((name, p, why))
    return out


# ---------------------------------------------------------------------------
# ②' 近失前沿复现（2026-11-12 12:40 新增）
#   实测依据见 SEED_SCAN_TARGET 处：种子扫描臂的近失率 33% vs 突变臂 11%
#   ⇒ 每轮把「只差一档」的配方格再补几个**没试过的**seed（近失格 = 稀缺资源）。
#   排序：① 过项数多 ② 归一化亏损小 ③ 组内 nano 均值高
# ---------------------------------------------------------------------------
def base_derived_cells(led: dict, groups: dict, base_params: dict,
                       exclude_keys: set | None = None) -> list[tuple[str, dict, str, int]]:
    """**基线派生绝对格**：以 Baseline_v2.0（trial_9863）自己的配方为底，只改 1~2 个参数。

    🟢🟢 2026-11-13 12:40 新增。与 `priority_cells()` 的本质区别：
      · `priority_cells()` = **冠军配方 + 补丁** ⇒ 冠军一变，补丁的含义就变（漂移）；
      · 这里 = **基线配方 + 补丁** ⇒ 语义稳定，永远以「唯一产出过 6/6 的配方」为底。
    证据见 BASE_DERIVED_CELLS 注释（sw 是 nano ↔ (F1/T1/T3) 的主交换轴，
    而 4.0~5.0 的插值区 + sw5×thr16ms4×lrf0.1 的交叉格**从未采样**）。

    返回：[(name, params, why, n_seeds), ...]
    """
    bg = base_group(led, groups)
    bparams = clean_recipe((bg or {}).get("params") or base_params)
    bmean = (bg or {}).get("mean_nano") or 0.0
    bmean4 = (bg or {}).get("mean_nano4") or 0.0
    bmeanf1 = (bg or {}).get("mean_f1") or 0.0
    seen = set(exclude_keys or ())
    out = []
    for name, patch, why, ns in BASE_DERIVED_CELLS:
        p = dict(bparams)
        p.update(patch)
        p["epochs"] = max(5, int(p.get("sched") or 5))
        k = recipe_key(p)
        if k in seen:
            continue
        grp = groups.get(k)
        # 🔴 2026-11-13 17:15：**bdc 家族不能用「均值 nano 低于基线」判死**。
        #   审计发现（reviewer）：基线组（sw5.0）mean_nano=0.693124 ⇒ 死线 0.691124，
        #   而 `BASE_sw4.5_lrf0.0` 的 1 臂均值已经 0.673677 ⇒ 按旧线，**2~3 轮后
        #   sw4.0~4.75 插值格会被全部判死** ⇒ 正好废掉本策略的核心区。
        #   根因：插值格的**设计意图就是拿 nano 换 F1/T1/T3** ⇒ 用 nano 淘汰它 = 用旧偏好的尺子量新东西。
        #   ⇒ 改成「**补偿项也弱才算死**」：nano4 与 F1 同时明显低于基线才判死（且要 n ≥ 8）。
        if (grp and grp.get("n", 0) >= PRIORITY_MIN_N * 2
                and (grp.get("mean_nano4") or 0.0) < bmean4 - 0.003
                and (grp.get("mean_f1") or 0.0) < bmeanf1 - 0.001):
            continue
        seen.add(k)
        out.append((name, p, why, int(ns)))
    return out


def _near_miss_score(st: dict, base: dict) -> tuple[int, float]:
    """该臂过了几项 + 归一化亏损（0 = 全过）；缺项（没测）不计入亏损。"""
    pk = st.get("picked") or {}
    of = st.get("official") or {}
    chk = core_checks(st, base)
    npass = sum(1 for v in chk.values() if v)
    deficit = 0.0
    for k, ok in chk.items():
        if ok:
            continue
        if k == "t2fp":
            d = (float(of.get("t2_fp") or 0) - float(base.get("t2_fp") or 0)) / 5.0
        else:
            bv = float(base.get(k) or 0.0) or 1.0
            mv = float(pk.get(k) if k in ("nano", "nano4", "f1") else (of.get(k) or 0))
            d = (bv - mv) / bv
        deficit += max(0.0, d)
    return npass, deficit


def near_miss_t2_ok(led: dict, g: dict, base: dict, best_trial=None) -> bool:
    """近失格的 T2 口径：**按交付判据**（T2 虚警 ≤ 基线 + 2），而不是 T2 F1 的 0.002 硬容差。

    理由（实测）：12346（mix w2.5）T2 F1 = 0.995290，只差基线 0.00015 就被 `recipe_t2_dead`
    判死 ⇒ 被近失队列排除；但它的 **T2 虚警只有 1 个**（基线 5 个）⇒ 六项判据里它是**过的**
    ⇒ 这类格恰恰是今晚最该复现的「只差一档」。
    ⇒ 这里用宽松口径（T2 F1 掉 0.01 以上 / 虚警超基线 2 个以上 才算塌），与交付判据一致。

    🔴 2026-11-13 09:30 修（组级聚合误杀前沿格，**今晚最重要的一个 bug**）：
       原实现按**整组**判定健康（组内 T2 F1 最小值 / 组内虚警**最大**值）⇒
       组里**任意一个运气差的臂**（虚警 9~15 个）就把整格判死 ⇒
       全台账最好的两个格被整整一夜排除在近失队列之外：
         · 12519 组（sw5.0 × thr12ms3 × lrf0.1 × w2.0，n=36）单臂虚警 **2**、T3 只差 **1.2e-4**
         · 12679 组（sw4.0 × thr12ms3 × lrf0.1 × w2.0，n=76）单臂虚警 **1**、T3 只差 **1.7e-4**
       但这两组的组内虚警最大值分别是 15 / 9（> 基线+2=7）⇒ 被组级口径判死。
       ⇒ 改为**按该格最佳臂（best_trial）逐臂判定**，与六项交付判据（t2fp ≤ 基线）同口径；
         最佳臂没有官方读数时才回退到组级聚合。
    """
    of = ((led.get("arms") or {}).get(str(best_trial)) or {}).get("official") or {}
    t2 = of.get("t2")
    fp = of.get("t2_fp")
    if t2 is None or fp is None:
        t2 = group_t2(led, g)
        fp = None
        for t in (g.get("trials") or []):
            of2 = ((led.get("arms") or {}).get(str(t)) or {}).get("official") or {}
            if of2.get("t2_fp") is not None:
                fp = of2["t2_fp"] if fp is None else max(fp, of2["t2_fp"])
    if t2 is not None and t2 < (base.get("t2") or 0.0) - 0.01:
        return False
    if fp is not None and fp > (base.get("t2_fp") or 0) + 2:
        return False
    return True


def near_miss_cells(led: dict, base: dict, exclude_keys=()) -> list[tuple]:
    """挑出「只差一档」的配方格（自适应），供本轮补 seed。

    过滤：① 不是基线配方组（那组由种子扫描覆盖）② 必须有官方三轨读数（才算竞争者）
          ③ swl >= 1.0（硬闸门）④ 未被证伪 T2 塌陷 ⑤ 不在本轮其它队列里（避免重复臂）
    阈值：过 4/6 ⇒ 亏损 ≤ 0.04；只过 3/6 ⇒ 亏损必须 ≤ 0.012（否则离全过太远，不值得补 seed）
    """
    groups = recipe_stats(led) or {}
    bg = base_group(led, groups)
    bkey = (bg or {}).get("key")
    excl = set(exclude_keys or ())
    out = []
    for g in groups.values():
        if g["key"] == bkey or g["n"] < 1 or g["key"] in excl:
            continue
        best = None
        for t in g["trials"]:
            st = (led.get("arms") or {}).get(str(t)) or {}
            if (st.get("official") or {}).get("t1") is None:
                continue
            npass, deficit = _near_miss_score(st, base)
            if best is None or (npass, -deficit) > (best[0], -best[1]):
                best = (npass, deficit, int(t))
        if best is None or best[0] < NEAR_MISS_MIN_PASS:
            continue
        lim = NEAR_MISS_MAX_DEFIIT if best[0] >= 4 else NEAR_MISS_MAX_DEFIIT_3
        if best[1] > lim:
            continue
        if not swl_safe_group(g):
            continue
        if not near_miss_t2_ok(led, g, base, best_trial=best[2]):
            continue
        out.append((g, best))
    out.sort(key=lambda x: (-x[1][0], x[1][1], -(x[0].get("mean_nano") or 0.0)))
    return out[:NEAR_MISS_CELLS]


def _group_t3_stats(led: dict, g: dict, base: dict) -> tuple[int, int]:
    """该配方组里「三项全平」的臂中，有多少过了官方 T3 线（返回 (k, n_tri)）。

    🟢 2026-11-13 13:20：给 `seed_scan_targets()` 的**六项联合权重**提供 T3 因子。
    只统计**有官方读数**的臂（三项全平的臂才可能有官方读数 ⇒ n_tri 很小 ⇒ 需平滑）。
    """
    b_nano = base.get("nano") or 0.0
    b_n4 = base.get("nano4") or 0.0
    b_f1 = base.get("f1") or 0.0
    b_t3 = base.get("t3") or 0.0
    k = 0
    n = 0
    for t in g.get("trials") or []:
        st = (led.get("arms") or {}).get(str(t)) or {}
        pk = st.get("picked") or {}
        of = st.get("official") or {}
        if pk.get("nano") is None or of.get("t3") is None:
            continue
        if not (pk["nano"] >= b_nano and (pk.get("nano4") or 0) >= b_n4
                and (pk.get("f1") or 0) >= b_f1):
            continue
        n += 1
        if of["t3"] >= b_t3:
            k += 1
    return k, n


def cell_stats(led: dict, base: dict) -> dict:
    """**格级**（sw × thr × maxside × lr_floor）统计：三项全平率 + 三项全平臂的 T3 过线率。

    🔴 2026-11-13 11:35：**为什么不能只看「完整配方组」**。
      `recipe_stats()` 的分组键是**完整配方**（含 mix/lr/fb/ow/mosaic/… 共 24 个参数）
      ⇒ 同一个「有意义的格」会被拆成好几个组，n3/n 被严重稀释：
        · 完整组口径：A2 格 n=76 只出 3 个三项全平 ⇒ 3.9%；A1 n=36 ⇒ 8.3%；
          B1 n=22 ⇒ 9.1%  ⇒ 排序把 B1 排第一
        · **格级口径（只按 sw/thr/ms/lrf 聚合，抹掉噪声参数）**：
          A2 官方读数 13 臂 ⇒ p_triad **0.23**；A1 8 臂 ⇒ **0.38**；
          B1 52 臂 ⇒ **0.06**；C 6 臂 ⇒ 0.00
      ⇒ **同一批臂，只因分组粒度不同，结论就完全相反** ⇒ 种子扫描的权重必须用格级口径。
    """
    b_nano = base.get("nano") or 0.0
    b_n4 = base.get("nano4") or 0.0
    b_f1 = base.get("f1") or 0.0
    b_t3 = base.get("t3") or 0.0
    out: dict = {}
    for _t, a in (led.get("arms") or {}).items():
        pk = a.get("picked") or {}
        if pk.get("nano") is None:
            continue
        k = _cell_of(a)
        d = out.setdefault(k, {"n": 0, "triad": 0, "t3": 0, "ntri": 0})
        d["n"] += 1
        if (pk["nano"] >= b_nano and (pk.get("nano4") or 0) >= b_n4
                and (pk.get("f1") or 0) >= b_f1):
            d["triad"] += 1
            of = a.get("official") or {}
            if of.get("t3") is not None:
                d["ntri"] += 1
                if of["t3"] >= b_t3:
                    d["t3"] += 1
    return out


def _cell_weight(led: dict, base: dict, params: dict) -> tuple[float, str]:
    """格级期望产出率权重（用于种子扫描的臂数分配）。

    E[六项全过/臂] ≈ p_triad(格) × P(官方三轨全过 | 三项全平, 格)
      · p_triad：格级实测（抹掉 mix/lr 等噪声参数 ⇒ 样本利用率高）
      · P(官方|三项全平)：格级 n_tri 只有 1~3 ⇒ **必须向全局均值收缩**（否则
        1 个样本的 B2 会拿到 0.75 的虚高权重）⇒ 平滑：p = (k + 2·p_pool) / (n_tri + 2)
    """
    cs = cell_stats(led, base)
    if not cs:
        return 1e-6, "no-cell-stats"
    # 全局 T3 过线率（三项全平臂口径）⇒ 收缩目标
    _tn = sum(d["ntri"] for d in cs.values())
    _tk = sum(d["t3"] for d in cs.values())
    p_pool = (_tk + 0.5) / (_tn + 1.0) if _tn else 0.4
    k = _cell_of({"params": params})
    d = cs.get(k)
    if not d or d["n"] < 3:
        return 1e-6, "thin-cell"
    p_triad = d["triad"] / d["n"]
    if d["ntri"]:
        p_t3 = (d["t3"] + 2.0 * p_pool) / (d["ntri"] + 2.0)
    else:
        p_t3 = p_pool
    return max(1e-6, p_triad * (0.25 + 0.75 * p_t3)), f"cell p_triad={p_triad:.2f} p_t3={p_t3:.2f}"


def seed_scan_targets(led: dict, base: dict, champ_params: dict,
                       k: int = 4, n_total: int | None = None) -> list[tuple]:
    """种子扫描的**多格分配**：把种子臂按「三项全平产出率」分给多个配方格。

    2026-11-13 11:40（台账**严格 recipe_key** 分组，全盘 947 臂复算）：
      「三项全平」（nano & nano4 & f1 三项同时过线）是**六项全过的必要条件**
      （官方 T1/T3 只可能在 NanoBench 三项过线的臂上同时过线），
      而全台账只有 **4 个 swl=1.0 的配方组**产出过三项全平臂：

        · A1  sw5.0×thr12ms3×lrf0.1×w2.0   n=36 → 3 个 ⇒ 8.3%
        · A2  sw4.0×thr12ms3×lrf0.1×w2.0   n=76 → 3 个 ⇒ 3.9%（= 当前冠军）
        · B1  sw5.0×thr16ms4×lrf0.0×w2.0   n=22 → 2 个（含**唯一的 6/6 = 9863**）⇒ 9.1%
        · B2  sw5.0×thr16ms4×lrf0.1×w2.0   n=21 → 1 个 ⇒ 4.8%

      其余 **140+ 个臂只出 3 个** ⇒ 产出率约 1%。
      ⇒ 旧口径「18 个种子臂全打冠军 A2」= 把算力压在**产出率最低**的那一格：
        期望三项全平数 = 18×3.9% = 0.70；
        按产出率分给 4 个高产格 = 18×(8.3+3.9+9.1+4.8)/4% = 1.17（**+67%**）。
      ⇒ 另：**B1 是基线组**，被 `near_miss_cells()` 以「那组由种子扫描覆盖」为由**排除**，
        而种子扫描打的其实是**冠军**（A2）⇒ **B1 一条专属臂都没有**，
        而它恰恰是**唯一产出过 6/6 的格** ⇒ 这里显式保底。

    返回：[(params, n_arms, recipe_key, 产出率), ...]
    """
    n_total = int(n_total or SEED_SCAN_TARGET)
    groups = recipe_stats(led) or {}
    b_nano = base.get("nano") or 0.0
    b_n4 = base.get("nano4") or 0.0
    b_f1 = base.get("f1") or 0.0
    cand = []
    for g in groups.values():
        if g["n"] < 8:
            continue
        p = g.get("params") or {}
        if (p.get("swl") or 1.0) < 1.0:      # swl<1.0 是官方 T1/T3 死区 ⇒ 不投
            continue
        n3 = sum(1 for a, b, c in zip(g["nano"], g["nano4"], g["f1"])
                 if a >= b_nano and b >= b_n4 and c >= b_f1)
        if n3 <= 0:
            continue
        # 🟢🟢 2026-11-13 13:20：**权重改成「六项联合估计」** = 三项全平率 × P(T3 | 三项全平)
        #   依据（§153.16 家族级统计）：六项判据的**真实瓶颈是 T3**，而「三项全平」只是必要条件：
        #     · A1(sw5×thr12ms3) 三项全平臂 **T3 全灭**（0/3）
        #     · A2(sw4×thr12ms3) 三项全平臂 **T3 全灭**（0/3）
        #     · B1(sw5×thr16ms4) 三项全平臂 T3 过 1/3（**含唯一的 6/6 = 9863**）
        #     · B2(sw5×thr16ms4×lrf0.1) 三项全平臂 T3 过 1/1（12053，T3 0.949250）
        #   ⇒ 旧权重把 10/24 个种子臂压在「T3 全灭」的 A1 上 ⇒ **改用联合估计**把算力挪向 thr16 家族。
        #   平滑用 Laplace（n_tri 很小，不能用裸比例）：p = (k + 0.5) / (n_tri + 1)
        # 🔴🟢 2026-11-13 11:35：**权重改用格级口径**（旧口径用完整配方组 ⇒ n3/n 被噪声
        #   参数稀释 ⇒ 把 B1 排第一、把真实最高产的 A1/A2 排后面）
        _w, _why = _cell_weight(led, base, p)
        cand.append((_w, g["n"], n3, g))
    cand.sort(key=lambda x: (-x[0], -x[1]))
    picked = list(cand[:k])
    have = {g["key"] for _r, _n, _c, g in picked}
    a98 = (led.get("arms") or {}).get(str(ns.BASE_V2["trial"]), {})
    champ_key = recipe_key(champ_params)
    for key in (recipe_key(a98.get("params") or {}), champ_key):
        if key in groups and key not in have:
            picked.append((0.0, groups[key]["n"], 0, groups[key]))
            have.add(key)
    if not picked:
        return [(dict(champ_params), n_total, "champ", 0.0)]
    rates = [max(r, 1e-6) for r, _n, _c, _g in picked]
    tot = sum(rates)
    alloc = []
    for r, _n, _c, g in picked:
        ni = max(2, int(round(n_total * r / tot)))
        alloc.append([clean_recipe(g["params"]), ni, g["key"], r])
    used = sum(a[1] for a in alloc)
    while used > n_total and len(alloc) > 1:
        j = min(range(len(alloc)), key=lambda i: alloc[i][1])
        if alloc[j][1] <= 2:
            break
        alloc[j][1] -= 1
        used -= 1
    if used < n_total:
        j = max(range(len(alloc)), key=lambda i: rates[i])
        alloc[j][1] += (n_total - used)
    for idx, a in enumerate(alloc):
        if a[2] == champ_key and a[1] < 4:
            need = 4 - a[1]
            a[1] += need
            j = min((i for i in range(len(alloc)) if i != idx), key=lambda i: alloc[i][1])
            alloc[j][1] = max(2, alloc[j][1] - need)
    return [(a[0], a[1], a[2], a[3]) for a in alloc]


def build_train_arms(led: dict, champ: dict, round_idx: int, next_trial: list[int],
                     n_seed: int = 6, n_mut: int = 18,
                     arm_budget: int | None = None) -> list[tuple]:
    """一轮的训练臂：
      ① 种子扫描 n_seed 臂（冠军配方、不同 seed）⇒ 建立该配方的 seed 分布（稳健线）
      ② 突变格 n_mut 臂（冠军配方 + 一处改动）⇒ 探索
      ③ 待复现配方 replication_pending（上一轮单臂打平过、但还没第二个 seed 的配方）
    **所有臂都从 base init 起训**（二次退火路径已证伪，见文件头证据①）。

    🟢 2026-11-13 16:10：新增 `arm_budget`（末轮训练臂硬上限）。
       保留优先级（先留后砍）：
         ① 基线派生格（BASE_*，**从未采样**的 sw 4.0~4.75 插值区）
         ② T3 前沿格（优先格里 T3F_/T3I_*：直接攻击唯一卡点 T3）
         ③ 待复现配方（上一轮单臂打平过 ⇒ 把「运气臂」变成「真配方」）
         ④ 种子扫描（实测三项全平产出率最高 ≈6.5%/臂）
         ⑤ 近失格复现 ⑥ 其余优先格 ⑦ 突变格（产出率最低 ≈1%）
       硬组（格级）放不下就**整格丢**，软组（种子/突变）可切单臂。
       arm_budget=None ⇒ 行为与原来**完全一致**（臂序也不变）。
    """
    base_params = clean_recipe(champ["params"])
    groups: list[tuple[str, list[tuple]]] = []   # (价值标签, 该组的臂)
    _seq = 0                                     # 单调臂序号（seed 偏移必须逐臂唯一）
    seed0 = SEED_BASE + round_idx * 97
    # 🟢🟢 2026-11-13 21:15：**全局 (配方, seed) 去重**（见 used_recipe_seeds 的实测）
    _used_pairs = used_recipe_seeds(led)

    _round_seeds: set = set()      # 本轮已发出的 seed（保持「全局唯一」旧不变式）

    def _fresh_seed(_p: dict, _seed: int) -> int:
        """返回一个**未用过**的 seed。

        两条约束同时满足：
          ① **本轮内 seed 全局唯一**（旧不变式：种子偏移必须逐臂唯一）
          ② **(配方, seed) 组合未在台账里出现过** ⇒ 不会重训出**完全相同**的模型
             （同 init + 同 seed + 同配方 ⇒ 确定性同结果 ⇒ 纯浪费）
        """
        _k = recipe_key(_p)
        _s = int(_seed)
        while (_k, _s) in _used_pairs or _s in _round_seeds:
            _s += 1
        _used_pairs.add((_k, _s))
        _round_seeds.add(_s)
        return _s

    # ③ 待复现配方（优先级最高：能把「运气臂」变成「真配方」）
    _gs_all = recipe_stats(led) or {}
    # 🟢 2026-11-13 21:45：**待复现配方按「离六项最近」排序**（旧代码按台账插入序
    #   ⇒ 复现的往往是离六项很远的配方，8 臂白烧）。
    #   排序键 = (过项数降序, 归一化亏损升序)，复用 `_near_miss_score` 的同一口径。
    _pend_all = (led.get("search_state") or {}).get("replication_pending") or []
    _base_now = base_ref(led)

    def _pend_rank(_rk: str) -> tuple:
        _g = _gs_all.get(_rk)
        if not _g:
            return (0, 9.9)
        _best_n, _best_d = 0, 9.9
        for _t in (_g.get("trials") or []):
            _st = (led.get("arms") or {}).get(str(_t)) or {}
            _n, _d = _near_miss_score(_st, _base_now)
            if (_n, -_d) > (_best_n, -_best_d):
                _best_n, _best_d = _n, _d
        return (-_best_n, _best_d)

    pend = sorted(_pend_all, key=_pend_rank)[:4]
    # 🟢 2026-11-12 12:50：近失队列要把「本轮已经喂过的配方」排除（同一配方不重复喂）
    #   ⚠️ 只排除**真的产出了臂**的待复现配方（否则被 recipe_t2_dead 跳过的配方会误伤近失队列）
    _pcells = priority_cells(led, base_params)
    _emitted_keys: set = set()
    for rk in pend:
        grp = _gs_all.get(rk)
        if not grp:
            continue
        # 🔴 T2 死区配方不复现（见 recipe_t2_dead）
        if recipe_t2_dead(led, grp, base_ref(led)):
            continue
        p = clean_recipe(grp["params"])
        _g = []
        for k in range(2):
            trial = next_trial[0]
            next_trial[0] += 1
            _pp = dict(p, epochs=max(5, int(p.get("sched") or 5)))
            _g.append((trial, _fresh_seed(_pp, seed0 + 500 + _seq), _pp,
                       f"[R{round_idx}] ★★ 复现 {rk.split('|')[0][:40]} seed{k+1}（上一轮单臂打平基线）"))
            _seq += 1
            _emitted_keys.add(rk)
        if _g:
            groups.append(("pend", _g))

    # ⓿ 优先格（2026-11-12 新增）：每格每轮 2 个 seed
    #   ① 配方效应 vs 种子噪声必须分开量（单臂读数分不清）
    #   ② 攒到 PRIORITY_MIN_N 个 seed 后按「均值 vs 基线配方均值」淘汰差格
    for _name, _p, _why in _pcells:
        _ns = int(PRIORITY_SEED_OVERRIDE.get(_name, 2))
        _g = []
        for _k in range(_ns):
            _t = next_trial[0]
            next_trial[0] += 1
            _pp = dict(_p)
            _g.append((_t, _fresh_seed(_pp, seed0 + 200 + _seq), _pp,
                       f"[R{round_idx}] ★★ 优先格 {_name} seed{_k+1}：{_why}"))
            _seq += 1
            _emitted_keys.add(recipe_key(_p))
        if _g:
            _tag = "prio_t3" if _name.startswith(("T3F", "T3I")) else "prio"
            groups.append((_tag, _g))

    # ⓿⓿ **基线派生绝对格**（2026-11-13 12:40 新增）
    #   以 Baseline_v2.0（trial_9863）的**自身配方**为底，只改 1~2 个参数 ⇒
    #   **不受「冠军配方漂移」影响**（优先格是冠军 + 补丁，冠军一变补丁含义就变）。
    #   这 5 格是台账里**先验最高、却从未采样**的邻域（sw 插值 + lrf0.1 交叉）。
    for _name, _p, _why, _ns in base_derived_cells(led, _gs_all, base_params,
                                                   exclude_keys=_emitted_keys):
        _g = []
        for _i in range(int(_ns)):
            _t = next_trial[0]
            next_trial[0] += 1
            _pp = dict(_p)
            _g.append((_t, _fresh_seed(_pp, seed0 + 300 + _seq), _pp,
                       f"[R{round_idx}] ★★ 基线派生格 {_name} seed{_i+1}：{_why}"))
            _seq += 1
        if _g:
            _emitted_keys.add(recipe_key(_p))
            groups.append(("bdc", _g))

    # ②' 近失前沿复现（2026-11-12 12:40 新增）：给「只差一档」的格补没试过的 seed
    #   实测依据：种子扫描臂近失率 33% vs 突变臂 11% ⇒ 近失格是稀缺资源，值得专门补 seed
    for g, best in near_miss_cells(led, base_ref(led), exclude_keys=_emitted_keys):
        p = clean_recipe(g["params"])
        used_seeds = set()
        for t in g["trials"]:
            _s = (led.get("arms") or {}).get(str(t), {}).get("seed")
            if _s is not None:
                used_seeds.add(int(_s))
        _g = []
        for k in range(NEAR_MISS_SEEDS):
            trial = next_trial[0]
            next_trial[0] += 1
            _pp = dict(p, epochs=max(5, int(p.get("sched") or 5)))
            sd = _fresh_seed(_pp, seed0 + 700 + _seq)
            used_seeds.add(sd)
            _g.append((trial, sd, _pp,
                       f"[R{round_idx}] ★★ 近失复现 {g['key'].split('|')[0][:36]} seed{k+1}"
                       f"（该格最佳臂过 {best[0]}/6 项，亏损 {best[1]:.4f}）"))
            _seq += 1
        if _g:
            groups.append(("nm", _g))

    # ① 种子扫描（🟢 2026-11-12 12:40：提到 SEED_SCAN_TARGET=12，近失率是突变的 3 倍）
    # 🟢🟢 2026-11-13 11:45：**多格分配**（不再把 18 个种子臂全压在冠军那一格）
    _targets = seed_scan_targets(led, base_ref(led), base_params, k=4,
                                  n_total=max(n_seed, SEED_SCAN_TARGET))
    _si = 0
    for _p, _n, _key, _rate in _targets:
        # ⚠️ 每个种子格单独成组 ⇒ 预算切臂时**整格切**，不会留下「1 臂碎片」
        #   （单臂种子扫描给不出种子分布 ⇒ 统计上无意义）
        _g = []
        for i in range(_n):
            trial = next_trial[0]
            next_trial[0] += 1
            p = dict(_p, epochs=5)
            _g.append((trial, _fresh_seed(p, seed0 + _si), p,
                       f"[R{round_idx}] ★ 种子扫描 {i+1}/{_n}"
                       f"（{_key.split('|')[0][:28]}｜三项全平产出率 {_rate:.1%}）"))
            _si += 1
            _seq += 1        # ⚠️ 必须计入：旧代码的种子偏移计数器是「已发臂总数」
        if _g:
            groups.append(("seed", _g))

    # ② 突变格（🟢 2026-11-12 12:40：封顶 MUT_CAP ⇒ 主力算力让给种子/近失格）
    _g = []
    for name, patch, why in next_mutations(led, base_params, min(n_mut, MUT_CAP)):
        trial = next_trial[0]
        next_trial[0] += 1
        p = dict(base_params)
        p.update(patch)
        p["epochs"] = max(5, int(p.get("sched") or 5))
        _g.append((trial, _fresh_seed(p, seed0 + 100 + _seq), p,
                   f"[R{round_idx}] ★ 突变 {name}：{why}"))
        _seq += 1
    if _g:
        groups.append(("mut", _g))

    if arm_budget is None:
        return [a for _tag, g in groups for a in g]      # 原序 ⇒ 行为完全不变

    # ── 末轮硬上限：按价值顺序砍 ──────────────────────────────────────────
    # 🟢🟢 2026-11-13 21:20：**价值顺序重排 + 每组硬上限**
    #   依据（`cell_yield_audit.py` 全盘去重复算 + `seed_noise_audit.py`）：
    #     · bdc（基线派生族：B1 基线本体 / C=sw4.0×thr16ms4 / sw 插值）= 最高产
    #       （含全台账唯一 6/6 的格，与最近的 5/6 格）
    #     · nm（近失格复现）= T3 缺口 1.2e-4~7.8e-4 **小于 seed σ_t3=8.6e-4**
    #       ⇒ 靠种子噪声就能翻过去 ⇒ 单臂产出率最高的一类
    #     · seed（种子扫描，多格按联合产出率分配）
    #     · prio_t3（T3 前沿/插值格）= 探索性 ⇒ 封顶 18 臂
    #     · pend（上一轮打平待复现）/ mut（突变）/ prio = 保底
    #   ⚠️ 旧顺序把 prio_t3 排在 nm/seed 之前 ⇒ 预算一收紧，**近失格复现整组被丢**
    order = ("bdc", "nm", "seed", "prio_t3", "pend", "mut", "prio")
    caps = {"bdc": None, "nm": None, "seed": None,
            "prio_t3": 18, "pend": 4, "mut": 3, "prio": 4}
    budget = int(arm_budget)
    keep: list[tuple] = []
    for tag in order:
        if budget <= 0:
            break
        cap = caps.get(tag)
        spent = 0
        for _t, g in [x for x in groups if x[0] == tag]:
            if budget <= 0:
                break
            if cap is not None and spent + len(g) > cap:
                continue        # 硬上限：**整格丢**（不留碎片）
            if len(g) <= budget:
                keep.extend(g)
                budget -= len(g)
                spent += len(g)
            elif tag in ("seed", "mut") and budget >= 2:
                take = budget if cap is None else min(budget, max(0, cap - spent))
                if take >= 2:
                    keep.extend(g[:take])   # 软组：可切单臂，但**至少留 2 臂**（碎片无统计意义）
                    budget -= take
                    spent += take
            # 硬组（格级）/ 软组不足 2 臂：放不下就整格丢
    #   复盘要能看清「留了什么、丢了什么」⇒ 分别打印**保留**与**原始**的分组规模
    from collections import Counter  # noqa: PLC0415
    _kept_tags = Counter()
    for _a in keep:
        _d = _a[3]
        _kept_tags["bdc" if "基线派生格" in _d else
                   "prio_t3" if "优先格 T3" in _d else
                   "prio" if "优先格" in _d else
                   "pend" if "复现" in _d else
                   "nm" if "近失" in _d else
                   "seed" if "种子扫描" in _d else "mut"] += 1
    # ── 收尾补齐：整格丢规则会让预算剩 1~2 臂用不掉 ⇒ 从**被丢掉的格**里再捞一组
    #   （≥2 臂，避免 1 臂碎片；仍受每组硬上限约束）
    _kept_ids = {a[0] for a in keep}
    _spent_by_tag: dict[str, int] = {}
    for _tag in order:
        _spent_by_tag[_tag] = sum(len(g) for _tt, g in groups
                                  if _tt == _tag and g and g[0][0] in _kept_ids)
    if budget >= 2:
        for _tag in order:
            if budget < 2:
                break
            _cap = caps.get(_tag)
            for _tt, g in [x for x in groups if x[0] == _tag]:
                if budget < 2:
                    break
                if not g or g[0][0] in _kept_ids:
                    continue
                _take = min(len(g), budget)
                if _cap is not None:
                    _take = min(_take, max(0, _cap - _spent_by_tag.get(_tag, 0)))
                if _take >= 2:
                    keep.extend(g[:_take])
                    budget -= _take
                    _spent_by_tag[_tag] = _spent_by_tag.get(_tag, 0) + _take
                    break
    dropped = sum(len(g) for _t, g in groups) - len(keep)
    log(f"[arm_budget] 上限 {arm_budget} 臂 ⇒ 保留 {len(keep)} 臂（砍掉 {dropped} 臂）"
        f"｜保留构成：{dict(_kept_tags)}"
        f"｜原始构成：" + ", ".join(f"{t}={sum(len(g) for _tt, g in groups if _tt == t)}" for t in order))
    return keep


def build_posttrain_swa(led: dict, train_trials: list, next_trial: list,
                        top_n: int = 6) -> list[tuple]:
    """🟢 训练后自动 SWA：对本轮训练臂里 nano 最好的几个，取其**相邻 epoch** 等权平均。

    依据：2026-11-11 14:46 实测 —— 跨模型 soup 会崩，但**同臂相邻 epoch 的 SWA 成立**
    （swa_late nano 0.708885 已略超基线）⇒ 把「训练 → SWA」变成**标准配方**。
    """
    cands = []
    for t in train_trials:
        st = (led.get("arms") or {}).get(str(t)) or {}
        tr = st.get("nanobench") or {}
        vals = [v for v in tr.values() if "nano" in v]
        if not vals:
            continue
        # 🟢 2026-11-12 13:10：**只 SWA「swl 安全」的源臂**
        #   实测（全台账 36 个 SWA 臂）：swl<1.0 源臂的 SWA 产物**官方三轨全塌**
        #   （12131 源：nano 0.7552 但 T1 0.8647 / T3 0.8916 ⇒ 绝无升轨可能），
        #   而 swl>=1.0 源臂的 SWA 至少官方线都在（12327：只差 F1 0.000242 / T3 0.001173）。
        #   ⇒ swl<1.0 的源臂直接跳过，别浪费评测算力（每臂 ~1.3 min）。
        _swl = float(((st.get("params") or {}).get("swl") if
                      (st.get("params") or {}).get("swl") is not None else 1.0) or 1.0)
        if _swl < 1.0 - 1e-9:
            continue
        best = max(vals, key=lambda v: v["nano"])
        # 🟢🟢 2026-11-13 13:05：源臂排序改成**三项联动打分**（nano 主导 + nano4/F1 过线加分）
        #   依据（全台账 48 个 SWA 臂）：SWA 臂是**离六项最近的一类**——
        #     12702 (SWA e2..e5)：nano ✔ nano4 ✗(−0.0043) F1 ✔ T1 ✔ T3 ✗(−0.000024) 虚警 2 ✔
        #     12327 (SWA e5..e9)：nano ✔ nano4 ✔ F1 ✗(−0.000242) T1 ✔ T3 ✗(−0.001173) 虚警 5 ✔
        #     12579 (SWA e4,e5)：F1 ✔ T1 ✔ T3 ✔ 虚警 4 ✔（只差 nano/nano4）
        #   ⇒ 而**纯训练臂**绝大多数只过 1~3 项 ⇒ 零训练的 SWA 臂「单位算力命中率」远高于训练臂
        #   ⇒ 源臂不能只看 nano：还要偏好「nano4/F1 也在线」的臂（它们才有机会一次过六项）
        _bn, _bn4, _bf1 = ns.BASE_V2["nano"], ns.BASE_V2["nano4"], ns.BASE_V2["f1"]
        _score = (best["nano"]
                  + 0.30 * max(0.0, (best.get("nano4") or 0.0) - _bn4)
                  + 0.30 * max(0.0, (best.get("f1") or 0.0) - _bf1))
        cands.append((_score, best["nano"], t, best.get("epoch")))
    cands.sort(reverse=True)
    arms = []
    for _score, nano, t, ep in cands[:top_n]:
        if ep is None:
            continue
        cdir = f"{ns.ROOT}/trials/trial_{t}/weights"
        mem = [f"{cdir}/epoch{e}.pt" for e in range(max(1, int(ep) - 1), int(ep) + 2)]
        mem = [m for m in mem if os.path.isfile(m)]
        if len(mem) < 2:
            continue
        # 🟢🟢 2026-11-13 13:45：每个源臂出**两个** SWA：**等权** + **LAWA 加权**（λ=0.75）
        #   （加权变体零训练成本；§153.16.2 证明 SWA 臂是离六项最近的一类 ⇒ 多探权重空间的点）
        for _wtag, _wts in (("eq", None), ("lawa", _lawa_weights(len(mem)))):
            trial = next_trial[0]
            next_trial[0] += 1
            out = f"{ns.ROOT}/trials/trial_{trial}/weights/epoch{SYNTH_EPOCH}.pt"
            try:
                average_checkpoints(mem, out, epoch=SYNTH_EPOCH, weights=_wts,
                                    note=f"SWA(after-train,{_wtag}) of trial {t} ep{ep}")
            except Exception as e:  # noqa: BLE001
                log(f"[post-SWA] trial {t} 放弃：{e}")
                continue
            src = (led.get("arms") or {}).get(str(t)) or {}
            params = dict(src.get("params") or {})
            params.update(init=mem[0], synth=f"post_swa_of_{t}_{_wtag}",
                          epochs=SYNTH_EPOCH, sched=SYNTH_EPOCH)
            st2 = ns.arm_state(led, trial)
            st2.update(trial=trial,
                       desc=(f"★ 训练后 SWA({_wtag})：trial {t} ep{ep} 的相邻 epoch "
                             f"{'等权' if _wts is None else 'LAWA 加权'}平均"),
                       trained=True, synth=True, synth_members=mem, synth_note="post-train SWA",
                       synth_weights=list(_wts) if _wts else None,
                       params=params)
            log(f"[post-SWA] trial {t} ep{ep} → trial {trial}（{len(mem)} 成员，{_wtag}）")
            arms.append((trial, 30000 + trial, params, st2["desc"]))
    ns.save_ledger(led)
    return arms


# ---------------------------------------------------------------------------
# 结果落盘：每轮追加一段可读台账
# ---------------------------------------------------------------------------
def append_night_log(led: dict, base: dict, rec: dict, champ: dict) -> None:
    groups = recipe_stats(led)
    bg = champ.get("base_group")
    cg = champ.get("group")
    lines = []
    lines.append(f"\n## [{rec['ts']}] 第 {rec['round']} 轮完成\n")
    lines.append(f"- 冠军配方：`{(cg or {}).get('key', '?')}`"
                 f"（n={(cg or {}).get('n', 0)}，均值 nano={(cg or {}).get('mean_nano', 0):.6f}，"
                 f"最佳 {(cg or {}).get('max_nano', 0):.6f}）"
                 f"{'★已升格为搜索基线' if champ.get('adopted') else '（沿用基线配方）'}")
    if bg:
        lines.append(f"- 基线配方稳健线：n={bg['n']} 均值 nano **{bg['mean_nano']:.6f}**"
                     f"（严格线 = 单臂 {bg['max_nano']:.6f}，trial {bg['trials']}）")
    lines.append(f"- 本轮 {len(rec['arms'])} 臂；六项全平基线的臂：**{rec['beat'] or '无'}**")
    lines.append(f"- 待复现配方：{rec.get('replication_pending') or '无'}\n")
    pc = promotion_candidates(led, base)
    if pc:
        lines.append("- 可升轨候选 (n>=2 seed + 六项全平 + 均值不劣于基线配方):")
        for c in pc[:5]:
            lines.append(f"  - `{c['recipe'][:90]}` | n={c['n']} mean_nano={c['mean_nano']:.6f} | beat_arms={c['beat_trials']}")
        lines.append("")
    rows = []
    for t in rec["arms"]:
        st = (led.get("arms") or {}).get(str(t)) or {}
        pk = st.get("picked") or {}
        of = st.get("official") or {}
        if pk.get("nano") is None:
            continue
        rows.append((pk["nano"], t, st.get("desc", ""), pk.get("epoch"), pk.get("nano"),
                     pk.get("nano4"), pk.get("f1"), pk.get("micro"), of.get("t1"),
                     of.get("t2_fp"), of.get("t3"), of.get("verdict")))
    rows.sort(reverse=True)
    if rows:
        lines.append("| trial | 变更 | ep | nano | nano@4px | F1 | micro | T1 | T2FP | T3 | 门禁 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows[:14]:
            lines.append(f"| {r[1]} | {r[2][:40]} | {r[3]} | {r[4]:.6f} | "
                         f"{(r[5] or 0):.6f} | {(r[6] or 0):.6f} | {(r[7] or 0):.4f} | "
                         f"{r[8] if r[8] is not None else '-'} | "
                         f"{r[9] if r[9] is not None else '-'} | "
                         f"{r[10] if r[10] is not None else '-'} | {r[11] or '-'} |")
        lines.append("")
    text = "\n".join(lines)
    for path in (NIGHT_LOG, NIGHT_MEM):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                if not os.path.getsize(path):
                    f.write("# 自适应闭环夜间台账（2026-11-11 夜 → 2026-11-12 09:00）\n"
                            "> 目标：击败 Baseline_v2.0 = trial_9863（nano 0.708648 严格线 / 0.70097 稳健线）\n")
                f.write(text)
        except Exception as e:  # noqa: BLE001
            log(f"[night-log] 写入 {path} 失败：{e}")
    log(f"[night-log] 已追加第 {rec['round']} 轮结果 → {NIGHT_LOG} / {NIGHT_MEM}")


# ---------------------------------------------------------------------------
# 崩溃恢复（2026-11-11 16:52 的台账竞态事故之后新增）
# ---------------------------------------------------------------------------
def find_orphan_arms(led: dict) -> list:
    """找出「**训练完成但没有 picked**」的孤儿臂（崩溃遗留）。

    2026-11-11 16:52 实例：save_ledger 的临时文件竞态让 phase_nanobench 抛异常 ⇒
    R1 的 24 个臂训完却只有 5 个拿到 NanoBench 读数、**0 个有 picked**
    ⇒ 六项判据永远缺项、门禁/板卡/报告全没跑、整轮记录也丢了。
    ⇒ 下一轮把它们**并进来**（phase_nanobench 幂等 ⇒ 只补缺口），流程照常走完。
    """
    out = []
    for k, st in (led.get("arms") or {}).items():
        if st.get("synth") or st.get("official"):
            continue          # 已经跑过官方三轨（或被门禁窗口跳过）⇒ 不重复
        prm = st.get("params") or {}
        if prm.get("synth"):
            continue
        if not (st.get("picked") or st.get("nanobench")):
            continue          # 还没拿到任何读数 ⇒ 不是「待收尾」，是真没跑
        ep = int(prm.get("epochs") or 5)
        if not os.path.isfile(ckpt_path(int(k), ep)):
            continue          # 没有末 epoch 权重 ⇒ 没训完，不收
        out.append((int(k), int(st.get("seed") or 0), prm, st.get("desc", "")))
    out.sort()
    return out


def repick_all_arms(led: dict) -> int:
    """全局重挑 epoch（幂等、CPU-only、不占 GPU）。

    🟢 2026-11-11 18:05：运行中的调度器是 **17:40:53 启动的旧代码**，而
    `nano_search.py` 的新选 epoch 规则（六项合取）在 **17:45:54** 才落盘
    ⇒ 旧规则会把「nano4 差 0.0005」的 epoch 选上（实测 12047：ep5 被选，
      而它的 ep4 nano4=0.6569 其实是过的）⇒ **白丢候选**。
    ⇒ 每轮开头对**所有**有 NanoBench 读数的臂按**当前规则**重挑一次。

    ⚠️ 若某臂的 picked_epoch 变了 ⇒ 旧 epoch 的官方三轨 / dual / 板卡读数**必须作废**
      （否则会拿「另一个 epoch 的官方分数」去和基线比 ⇒ 假「六项全平」）。
    """
    ids = sorted(int(k) for k in (led.get("arms") or {}))
    if not ids:
        return 0
    old_ep = {}
    for t in ids:
        st = (led.get("arms") or {}).get(str(t)) or {}
        old_ep[t] = st.get("picked_epoch")
    ns.phase_pick([(t, 0, {}, "") for t in ids])
    led = ns.load_ledger()
    changed = []
    for t in ids:
        st = (led.get("arms") or {}).get(str(t)) or {}
        new_ep = st.get("picked_epoch")
        if old_ep.get(t) is not None and new_ep is not None and old_ep[t] != new_ep:
            changed.append((t, old_ep[t], new_ep))
            for f in ("dual", "dual_pass", "speed_ok", "board_ms", "rknn"):
                st.pop(f, None)
            st["official"] = {"stale": f"epoch_changed {old_ep[t]}->{new_ep}"}
    ns.save_ledger(led)
    log(f"[re-pick] 全局重挑 epoch：{len(ids)} 臂有读数，{len(changed)} 臂 epoch 变化 "
        f"{changed[:12]}（变化的臂已作废旧 epoch 的官方/dual/板卡读数）")
    return len(changed)


def safe_phase(name: str, fn, *args, **kw):
    """一个阶段炸了不许掀掉整轮（只记日志，本轮其余部分照走）。"""
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        log(f"[{name}] 阶段异常（本轮其余部分照走）：{e}")
        import traceback
        log(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# 一轮
# ---------------------------------------------------------------------------
def run_round(round_idx: int, deadline_ts: float, gpus: list, next_trial: list,
              cross_soup: bool = False, swa_top_n: int = 6, n_seed: int = 6,
              n_mut: int = 18, arm_budget: int | None = None) -> dict:
    led = ns.load_ledger()
    base = base_ref(led)

    # ── 崩溃恢复（先做！）──────────────────────────────────────────────
    #   把「训完但没 picked」的孤儿臂的 NanoBench 读数补齐（幂等），
    #   这样**冠军配方的计算能用到它们**（否则 R1 的突变格全白跑）。
    orphans = find_orphan_arms(led)
    if orphans:
        log(f"[R{round_idx}] 崩溃恢复：{len(orphans)} 个孤儿臂 {[t for t, _s, _p, _d in orphans]}")
        gpu_of_o = {t: gpus[i % len(gpus)] for i, (t, _s, _p, _d) in enumerate(orphans)}
        safe_phase("orphan-nanobench", ns.phase_nanobench, orphans, EPOCHS_EVAL, gpu_of_o)
        safe_phase("orphan-pick", ns.phase_pick, orphans)
        led = ns.load_ledger()
        base = base_ref(led)

    # 全局重挑 epoch（当前规则）：旧规则挑出的 picked 可能在本轮才被写进台账 ⇒ 统一重算，
    # 否则「nano4 差 0.0005」的 epoch 会被选上 ⇒ 白丢候选（见 repick_all_arms 文档）。
    try:
        repick_all_arms(led)
        led = ns.load_ledger()
        base = base_ref(led)
    except Exception as e:  # noqa: BLE001
        log(f"[re-pick] 失败（不影响本轮其余流程）：{e}")

    # 🟢 2026-11-12 12:55：台账 `official` 字段修复（以磁盘 gate_report 为准）
    #   实测：9 个臂的 official 是 {'stale': 'epoch_changed X->Y'} ⇒ t1/t2_fp/t3 全空
    #   ⇒ ① 六项漏斗里它们既不算过也不算不过；② near_miss_cells 里 `official.t1 is None`
    #     的臂会被**直接跳过** ⇒ 最接近基线的臂（12053：NanoBench 三项全过 + T2/T3 过，
    #     只差 T1 0.000984）的配方格被排除在近失复现队列之外（白丢最强候选）。
    #   ⇒ 每轮开头先按磁盘上的权威 gate_report 回填（只修 picked epoch，不跨 epoch 混填）。
    try:
        from manu.pipeline.npu.repair_official_records import repair_ledger  # noqa: PLC0415
        _fixed = repair_ledger(led)
        if _fixed:
            ns.save_ledger(led)
            log(f"[R{round_idx}] 台账 official 回填 {len(_fixed)} 个臂："
                f"{[(f['trial'], f['epoch']) for f in _fixed]}")
            base = base_ref(led)
    except Exception as e:  # noqa: BLE001
        log(f"[repair-official] 失败（不影响本轮其余流程）：{e}")

    champ = pick_champion_recipe(led, base)
    cg = champ.get("group") or {}
    bg = champ.get("base_group") or {}
    log(f"[R{round_idx}] 冠军配方 n={cg.get('n', 0)} 均值nano={cg.get('mean_nano', 0):.6f} "
        f"最佳={cg.get('max_nano', 0):.6f} | 基线配方 n={bg.get('n', 0)} "
        f"均值nano={bg.get('mean_nano', 0):.6f} | 已升格={champ.get('adopted')}")

    _fsrc = frontier_swa_sources(led, base, k=2)
    if _fsrc:
        log(f"[R{round_idx}] 前沿 SWA 源：{[(s['trial'], s['npass'], round(s['deficit'], 6)) for s in _fsrc]}")
    synth = build_synth_arms(led, champ.get("arm"), base, round_idx, next_trial,
                              cross_soup=cross_soup, extra_sources=_fsrc)
    trains = build_train_arms(led, champ, round_idx, next_trial, n_seed=n_seed, n_mut=n_mut,
                              arm_budget=arm_budget)
    all_arms = synth + trains
    for t, s, p, d in all_arms:
        ns.arm_state(led, t).update(trial=t, seed=s, desc=d, params=p)
    ns.save_ledger(led)

    # 崩溃恢复：把「训完但没 picked」的孤儿臂并进本轮（nanobench 幂等 ⇒ 只补缺口）
    orphans = find_orphan_arms(led)
    if orphans:
        all_arms = orphans + all_arms
        log(f"[R{round_idx}] 崩溃恢复：并入 {len(orphans)} 个孤儿臂 {[t for t, _s, _p, _d in orphans]}")

    # 训练必须留出评测时间：硬截止前 45 min 停训（已训完的臂照常进评测与台账）
    stop_ts = min(deadline_ts - 45 * 60, time.time() + 5 * 3600)
    log(f"[R{round_idx}] 训练 {len(trains)} 臂（合成 {len(synth)} 臂免训练）")
    ns.phase_train(trains, stop_ts, gpus)

    # 训练后自动 SWA（把「训练 → SWA」变成标准配方）
    led = ns.load_ledger()
    post = build_posttrain_swa(led, [t for t, _s, _p, _d in trains], next_trial,
                               top_n=max(int(swa_top_n), SWA_SRC_MIN))
    if post:
        all_arms = all_arms + post
        for t, s_, p_, d_ in post:
            ns.arm_state(led, t).update(trial=t, seed=s_, desc=d_, params=p_)
        ns.save_ledger(led)
        log(f"[R{round_idx}] 训练后 SWA 追加 {len(post)} 臂 ⇒ 本轮共 {len(all_arms)} 臂")

    led = ns.load_ledger()
    gpu_of = {t: gpus[i % len(gpus)] for i, (t, _s, _p, _d) in enumerate(all_arms)}
    safe_phase("nanobench", ns.phase_nanobench, all_arms, EPOCHS_EVAL, gpu_of)
    safe_phase("pick", ns.phase_pick, all_arms)   # 先给所有臂定 picked（不进榜的臂也要留读数）

    # ④ 门禁窗口：只对「有可能打平基线」的臂跑官方三轨
    led = ns.load_ledger()
    cand = []
    for t, _s, _p, _d in all_arms:
        st = (led.get("arms") or {}).get(str(t)) or {}
        pk = st.get("picked") or {}
        if pk.get("nano") is not None and pk["nano"] >= base["nano"] - GATE_NANO_SLACK:
            cand.append(t)
    cand.sort(key=lambda t: -((led["arms"][str(t)].get("picked") or {}).get("nano") or 0))
    # 🟢 2026-11-12 修（实测）：门禁窗口只取「nano 前 8」⇒ nano 高但 F1 塌掉的臂把窗口占满，
    #   真正「nano/nano4/F1 三项全平基线」的臂反被标 `nano_below_gate_window` **从未跑官方三轨**
    #   ⇒ 六项判据永远缺 T1/T2/T3 ⇒ **无法升轨**（实测 4 臂：12300/12263/12252/12053）。
    #   ⇒ 三项全平的臂**必须**无条件进官方三轨（它们是唯一有资格升轨的臂）。
    triad = []
    for t, _s, _p, _d in all_arms:
        st = (led.get("arms") or {}).get(str(t)) or {}
        pk = st.get("picked") or {}
        if ((pk.get("nano") or 0) >= base["nano"] and (pk.get("nano4") or 0) >= base["nano4"]
                and (pk.get("f1") or 0) >= base["f1"]):
            triad.append(t)
    # 🟢 2026-11-13 10:05 门禁名额**平衡**（官方三轨是最贵的一步，R12 实测 ~12 min/臂）：
    #   新增 8 个零训练 SWA 候选后，若仍按「纯 nano 前 8」，SWA 臂（nano 与源臂相当、
    #   常常还略高）会把**新鲜前沿臂**挤出窗口 ⇒ 探索面被挤掉。
    #   ⇒ 窗口拆成两半：非 SWA 前 4 + SWA 前 4；再叠加「三项全平」的臂
    #     （其中 SWA 部分同样封顶 4 ⇒ 官方三轨的总量有上界）。
    _is_synth = {t: bool(((led.get("arms") or {}).get(str(t)) or {}).get("synth")) for t in cand + triad}
    # 🔴 2026-11-13 11:35：**探索名额不能与 triad 臂重复**（否则 8 个名额只换来 5 个新臂：
    #   实测 E 排序榜首 12679/12694/12709 都是 triad 臂 ⇒ 已被无条件门禁 ⇒ 浪费 3 个名额）。
    _triad_set = set(triad)
    _cand_train = [t for t in cand if not _is_synth.get(t) and t not in _triad_set]
    _cand_synth = [t for t in cand if _is_synth.get(t)]
    _triad_train = [t for t in triad if not _is_synth.get(t)]
    _triad_synth = [t for t in triad if _is_synth.get(t)]
    #   SWA 臂合计**封顶 4 个**官方门禁名额（优先给「三项全平」的 SWA ⇒ 最有希望过六项），
    #   非 SWA 保留 6 个 nano 名额 + 全部三项全平的臂 ⇒ 新鲜前沿臂不会被挤掉。
    # 🟢 2026-11-13 10:50：SWA 门禁名额 4 → 6
    #   理由：SWA 臂是**零训练**的，且实测能把 T3 顶到只差 2.4e-5（12705）
    #   ⇒ 它是「唯一有资格升轨却几乎不占训练时间」的一类臂
    #   ⇒ 名额从 4 提到 6，让更多前沿 SWA 候选拿到官方三轨读数（每臂 ~12 min）。
    #   🟢 2026-11-13 13:50：SWA 门禁名额 6 → **8**（SWA 臂是离六项最近的一类，且**零训练**成本）
    #   🟢 2026-11-13 21:50：SWA 门禁名额 8 → **12**（SWA 臂零训练，且实测最接近六项：
    #   12702/12705 把 T3 顶到只差 2.4e-5；每多 1 个名额 ≈ 12 min/4 卡 ≈ 3 min 墙钟）
    _synth_pick = (_triad_synth + [t for t in _cand_synth if t not in set(_triad_synth)])[:12]
    # 🟢 2026-11-13 21:25：**门禁窗口按 nano 降序**（官方三轨 ~12 min/臂 ⇒ 若被 deadline
    #   截断，必须先门禁**最好的**臂；旧顺序是台账插入序 ⇒ 可能先门禁掉最差的）
    def _nano_of(_t: int) -> float:
        return (((led.get("arms") or {}).get(str(_t)) or {}).get("picked") or {}).get("nano") or 0.0

    # 🔴🟢 2026-11-13 11:55：**探索名额按「格的期望产出率 E」排序**，不再按裸 nano。
    #   理由：官方三轨 ~12 min/臂（最贵的一步），而「nano 前 N」几乎必然把名额喂给
    #   B1 家族（sw5×thr16ms4×lrf0，E=0.0004）⇒ 名额浪费；A1/A2（thr12ms3×lrf0.1）
    #   E≈0.030（70 倍）却常被 nano 排序挤掉 ⇒ **今夜最该门禁的臂反而没被测**。
    #   ⚠️ 三项全平（triad）的臂仍然**全部**进官方三轨（唯一有资格升轨的一类）⇒ 判据没放松。
    try:
        _ytab = cell_yield_table(led, base)
    except Exception as _e:  # noqa: BLE001
        _ytab = {}
        log(f"[R{round_idx}] ⚠️ 期望产出率表计算失败（{_e}）⇒ 退回按 nano 排序")
    def _yield_of(_t: int) -> float:
        _st = (led.get("arms") or {}).get(str(_t)) or {}
        return _ytab.get(_cell_of(_st), 0.0)

    _cand_train = sorted(_cand_train, key=lambda t: (-_yield_of(t), -_nano_of(t)))
    _triad_train = sorted(_triad_train, key=_nano_of, reverse=True)
    _synth_pick = sorted(_synth_pick, key=_nano_of, reverse=True)
    #   探索名额 6 → 8（官方三轨 4 并发 ⇒ 每多 1 个名额 ≈ 3 min 墙钟；E=0.03 的臂很便宜）
    _top = (_cand_train[:8] + _synth_pick + _triad_train)
    gate_set = set(_top)
    # 🟢 三项全平的臂**排在最前**：官方三轨很贵，若被 deadline 截断，
    #   必须是「唯一有资格升轨的臂」先跑完（ThreadPool 按列表顺序派发）。
    gate_arms = ([a for a in all_arms if a[0] in set(_triad_train)]
                 + [a for a in all_arms if a[0] in set(_synth_pick)]
                 + [a for a in all_arms if a[0] in set(_cand_train[:8]) and a[0] not in set(triad)])
    for t, _s, _p, _d in all_arms:
        if t not in gate_set:
            st = ns.arm_state(led, t)
            st["official"] = {"skipped": "nano_below_gate_window"}
            st["gate_skipped"] = True
    ns.save_ledger(led)
    log(f"[R{round_idx}] 门禁窗口：{len(gate_arms)}/{len(all_arms)} 臂进官方三轨"
        f"（期望产出率前 8 + SWA 12 + 三项全平 {len(triad)} 臂：{triad}）")
    safe_phase("gate", ns.phase_gate, gate_arms, gpu_of)
    safe_phase("dual", ns.phase_dual, all_arms)

    led = ns.load_ledger()
    base = base_ref(led)
    beat = []
    for t, _s, _p, _d in all_arms:
        st = (led.get("arms") or {}).get(str(t)) or {}
        chk = full_checks(st, base)
        if chk and all(chk.values()):
            beat.append(t)

    # 板卡测速：只对「打平基线的臂」+ 当轮最强臂
    champ2 = pick_champion(led, base)
    board_targets = [a for a in all_arms if a[0] in set(beat)]
    if champ2 and champ2["trial"] not in set(beat):
        # 只有「六项齐全」的单臂才值得占板卡（没测过官方三轨的不测）
        cst = (led.get("arms") or {}).get(str(champ2["trial"])) or {}
        if full_checks(cst, base):
            board_targets += [a for a in all_arms if a[0] == champ2["trial"]]
    if board_targets:
        try:
            log(f"[R{round_idx}] 板卡测速：{[a[0] for a in board_targets]}")
            ns.phase_board(board_targets)
        except Exception as e:  # noqa: BLE001
            log(f"[R{round_idx}] 板卡测速失败（不影响本轮结论）：{e}")

    # 待复现配方：本轮单臂打平了基线、但该配方还没有 ≥2 个 seed
    groups = recipe_stats(led)
    pending = list((led.get("search_state") or {}).get("replication_pending") or [])
    for g in groups.values():
        if g["n"] >= 2:
            continue
        if g["max_nano"] >= base["nano"] and g["key"] not in pending:
            pending.append(g["key"])
    # 🔴 2026-11-12 09:25：过滤掉「官方 T2 已证伪」的待复现配方
    #   （swl0.2 / swl0.35 这类 ⇒ 复现它们只是再喂一遍 T2 死区）
    _by_key = {g["key"]: g for g in groups.values()}
    _dead = [rk for rk in pending
             if (_by_key.get(rk) is not None and recipe_t2_dead(led, _by_key[rk], base))]
    if _dead:
        log(f"[R{round_idx}] 待复现队列剔除 {len(_dead)} 个 T2 死区配方："
            f"{[rk.split('|')[0][:40] for rk in _dead]}")
    pending = [rk for rk in pending if rk not in set(_dead)]
    pending = pending[:8]

    rec = {"round": round_idx, "ts": time.strftime("%F %T"),
           "champion_before": (champ.get("group") or {}).get("key"),
           "arms": [t for t, _s, _p, _d in all_arms], "beat": beat,
           "gated": [a[0] for a in gate_arms],
           "board_measured": [a[0] for a in board_targets],
           "replication_pending": pending,
           "best_nano": max(((led["arms"].get(str(t), {}).get("picked") or {}).get("nano") or 0)
                            for t, _s, _p, _d in all_arms) if all_arms else 0}
    led.setdefault("rounds", []).append(rec)
    led.setdefault("search_state", {})["replication_pending"] = pending
    led["search_state"]["prefix_equiv"] = True      # epochs 20 ≡ 5（见文件头证据②）
    ns.save_ledger(led)
    safe_phase("report", ns.phase_report)
    try:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "beat_report.py")], check=False, timeout=900)
    except Exception as e:  # noqa: BLE001
        log(f"[beat_report] 失败：{e}")
    champ3 = pick_champion_recipe(led, base)
    append_night_log(led, base, rec, champ3)
    try:
        pc = promotion_candidates(led, base)
        with open(f"{ns.WORK}/promotion_ready.json", "w", encoding="utf-8") as f:
            json.dump({"ts": time.strftime("%F %T"), "round": round_idx,
                       "base_trial": base.get("trial"), "base_nano": base.get("nano"),
                       "candidates": pc}, f, ensure_ascii=False, indent=2)
        if pc:
            brief = [(c["recipe"][:40], c["n"], round(c["mean_nano"], 6), c["beat_trials"]) for c in pc]
            log(f"[R{round_idx}] 可升轨候选: {brief}")
    except Exception as e:
        log(f"[promotion_ready] 写失败: {e}")
    log(f"[R{round_idx}] 完成：打平基线的臂 = {beat or '无'}；"
        f"最佳 nano = {rec['best_nano']:.6f}（基线严格线 {base['nano']:.6f}）")
    return rec


def selftest() -> None:
    """CPU-only 自检：配方指纹 / 配方统计 / 冠军配方 / 合成臂写盘回读（不碰台账、不占 GPU）。"""
    import torch  # noqa: PLC0415
    led = ns.load_ledger()
    base = base_ref(led)
    groups = recipe_stats(led)
    assert groups, "配方统计为空"
    bg = base_group(led, groups)
    assert bg and bg["n"] >= 1, "找不到基线配方组"
    log(f"[selftest] 基线配方 n={bg['n']} 均值 nano={bg['mean_nano']:.6f} "
        f"trial={bg['trials']}")
    champ = pick_champion_recipe(led, base)
    assert champ["params"].get("init") == BASE_INIT, "冠军配方的 init 不是 base init"
    log(f"[selftest] 冠军配方 n={(champ.get('group') or {}).get('n')} "
        f"已升格={champ.get('adopted')} sw={champ['params'].get('sw')}")
    arms = build_train_arms(led, champ, 999, [999999], n_seed=2, n_mut=3)
    # ⚠️ 2026-11-12：训练臂数 = 待复现配方（每个 2 臂，最多 4 个）+ 种子扫描 + 突变格
    #   （台账新增 `search_state.replication_pending` 后不再恒等于 n_seed+n_mut）
    _pend = ((led.get("search_state") or {}).get("replication_pending") or [])[:4]
    _gs = recipe_stats(led)
    # ⚠️ 2026-11-12 09:25：T2 死区配方**不再复现** ⇒ 期望臂数要扣掉它们
    # 🟢 2026-11-12 12:55：期望臂数 = 待复现×2 + 种子扫描(≥SEED_SCAN_TARGET)
    #   + 突变(min(n_mut, MUT_CAP)) + 优先格×2 + **近失格×2**
    _pc = priority_cells(led, clean_recipe(champ["params"]))
    _pend_ok = [rk for rk in _pend if _gs.get(rk) and not recipe_t2_dead(led, _gs[rk], base)]
    # 🟢 2026-11-13 12:55：新增「基线派生绝对格」+ 优先格每格种子数可配 ⇒ 期望臂数要跟着算
    _bdc = base_derived_cells(led, _gs, clean_recipe(champ["params"]),
                              exclude_keys={recipe_key(pp) for _n, pp, _w in _pc} | set(_pend_ok))
    _nm = near_miss_cells(led, base,
                          exclude_keys=({recipe_key(pp) for _n, pp, _w in _pc} | set(_pend_ok)
                                        | {recipe_key(pp) for _n, pp, _w, _ns in _bdc}))
    _expect = (2 * len(_pend_ok)
               + max(2, SEED_SCAN_TARGET)
               + min(3, MUT_CAP)
               + sum(int(PRIORITY_SEED_OVERRIDE.get(_n, 2)) for _n, _pp, _w in _pc)
               + sum(int(_ns) for _n, _pp, _w, _ns in _bdc)
               + NEAR_MISS_SEEDS * len(_nm))
    assert len(arms) == _expect, f"训练臂数不对：{len(arms)} != {_expect}"
    for t, s, p, d in arms:
        assert p["init"] == BASE_INIT, f"臂 {t} 的 init 不是 base init"
        assert p["epochs"] == max(5, int(p["sched"])), f"臂 {t} 的 epochs/sched 不自洽"
    log(f"[selftest] 训练臂样例：{[(a[0], a[2].get('sw'), a[2].get('swl'), a[2].get('epochs')) for a in arms]}")
    # 🟢 2026-11-13 16:25：末轮硬上限回归
    assert round_arm_budget(600.0) is None, "时间充裕时不该限臂"
    for _rm in (240.0, 144.0, 90.0):
        _b = round_arm_budget(_rm)
        assert isinstance(_b, int) and 4 <= _b <= 120, f"末轮臂数上限异常：remain={_rm} → {_b}"
    _ab = build_train_arms(led, champ, 999, [999999], n_seed=2, n_mut=3, arm_budget=10)
    assert len(_ab) <= 10, f"末轮臂数超上限：{len(_ab)}"
    #   硬组整格丢 ⇒ 最多剩 1 臂用不掉（补齐也要求 ≥2 臂 ⇒ 不留 1 臂碎片）
    assert 9 <= len(_ab) <= 10, f"末轮臂数异常：{len(_ab)}"
    #   末轮必须**先留基线派生格**（从未采样的 sw 插值区）⇒ 前几臂的 desc 应含「基线派生格」
    _hit = sum(1 for a in _ab if "基线派生格" in a[3])
    assert _hit >= 5, f"末轮优先格顺序不对：基线派生格只占 {_hit} 臂"
    #   同一优先格内的多个 seed 必须互不相同（否则「配方效应 vs 种子噪声」分不开）
    _cells: dict[str, set] = {}
    for a in arms:
        _m = a[3].split(" seed")
        if len(_m) < 2 or "优先格 " not in a[3]:
            continue
        _cells.setdefault(_m[0], set()).add(a[1])
    for _c, _s in _cells.items():
        assert len(_s) == len([1 for a in arms if a[3].split(" seed")[0] == _c]), \
            f"优先格 {_c} 的 seed 重复"
    #   轮内 **seed 全局唯一**（各组 seed 偏移段互斥：种子 0~ / 突变 100+ / 优先 200+
    #   / 基线派生 300+ / 待复现 500+ / 近失 700+，且段内用单调计数器 _seq）
    assert len({a[1] for a in arms}) == len(arms), "训练臂 seed 有重复"
    log(f"[selftest] 末轮上限 OK（budget=10 ⇒ {len(_ab)} 臂，其中基线派生格 {_hit} 臂）")
    gens = generate_mutations(led, champ["params"], 5)
    assert gens, "系统网格生成器没有产出新格"
    log(f"[selftest] 系统网格生成器样例：{[g[0] for g in gens]}")
    # 🔴 回归（2026-11-12 09:20）：突变队列 / 系统网格**不得**含 swl<1.0 的格
    #   （实测 swl<1.0 ⇒ 官方 T2 单调塌到 0.80~0.96 ⇒ 官方必败 ⇒ 整格作废）。
    for _name, _patch, _why in MUTATIONS:
        assert ( _patch.get("swl") is None or _patch["swl"] >= 1.0), \
            f"突变队列含 T2 死区格 swl={_patch['swl']}（{_name}）"
    for _name, _patch, _why in gens:
        assert ( _patch.get("swl") is None or _patch["swl"] >= 1.0), \
            f"系统网格含 T2 死区格 swl={_patch['swl']}（{_name}）"
    log("[selftest] 回归：突变队列/系统网格无 swl<1.0 死区格 ✔")
    # 🔴 回归：若冠军配方被升格，它**必须**官方 T2 已验证不塌
    if champ.get("adopted") and champ.get("group"):
        assert t2_safe(led, champ["group"], base), "冠军配方未过 T2 安全闸门"
    log("[selftest] 回归：冠军配方过 T2 安全闸门 ✔")
    # 回归：缺官方三轨的臂**不得**被判成「打平基线」
    fake = {"picked": {"nano": 0.9, "nano4": 0.9, "f1": 0.9}, "official": {}}
    assert full_checks(fake, base) is None, "缺项臂被误判为打平基线"
    fake2 = {"picked": {"nano": 0.9, "nano4": 0.9, "f1": 0.9},
             "official": {"t1": 0.9, "t2_fp": 0, "t3": 0.9}}
    assert full_checks(fake2, base) is not None, "六项齐全臂被漏判"
    log("[selftest] 回归：缺项臂不算 beat ✔")
    # 合成臂写盘回读（用两个随机张量模拟同构 ckpt）
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "a.pt")
        b = os.path.join(td, "b.pt")
        torch.save({"model": {"w": torch.zeros(3, 3)},
                    "no_p2": 1, "head_level": "p3", "act": "silu", "head_conv": 64,
                    "late_upsample": 1, "stride": 2, "epoch": 1, "step": 1}, a)
        torch.save({"model": {"w": torch.ones(3, 3)},
                    "no_p2": 1, "head_level": "p3", "act": "silu", "head_conv": 64,
                    "late_upsample": 1, "stride": 2, "epoch": 2, "step": 2}, b)
        out = os.path.join(td, "out.pt")
        average_checkpoints([a, b], out, epoch=5, note="selftest")
        got = torch.load(out, map_location="cpu", weights_only=False)
        assert float(got["model"]["w"].flatten()[0]) == 0.5, "SWA 平均值不对"
        assert got["epoch"] == 5 and got.get("synth"), "合成臂的溯源字段缺失"
    for remain in (600.0, 240.0, 120.0, 65.0):
        sd, mt = shrink_plan(remain, 6, 18)
        need = plan_minutes(sd + mt, swa_arms=swa_reserve(remain), fixed=fixed_cost(remain))
        assert sd >= 2, f"seed arms collapsed: {sd}"
        assert need <= remain + 1.0, f"shrunk plan still overruns: remain={remain} need={need}"
    log(f"[selftest] time budget ok (6+18 arms needs {plan_minutes(24):.0f} min)")
    pc = promotion_candidates(led, base)
    log(f"[selftest] promotion candidates: {len(pc)}")
    log("[selftest] PASS")


# ── 续跑轮号自愈（2026-11-11 18:10，host 关机前加固）─────────────────────
#   背景：守护进程只能从台账顶层 `rounds` 取「已完成轮」，而轮记录是**轮末**才写
#   ⇒ 轮中途崩溃时台账里查不到该轮 ⇒ 守护会把续跑轮号算成 1（整轮重跑，浪费 ~2.5 h，
#   还会把重复配方混进 recipe_stats 污染判优）。
#   这里让 runner 自己再兜一层底：
#     已完成轮 = 台账 rounds 的最大轮号
#     最近开跑轮 = round_state.json 的 last_started_round（runner 每轮开头原子写）
#     续跑轮号 = max(--start-round, 已完成+1, 最近开跑轮)
#   ⇒ 中途崩溃重启 ⇒ 续跑「被打断的那一轮」，其孤儿臂由 find_orphan_arms 并入补评。
def _round_state_file() -> str:
    return f"{ns.WORK}/round_state.json"


def write_round_state(last_started: int) -> None:
    try:
        path = _round_state_file()
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"last_started_round": int(last_started),
                       "last_completed_round": completed_round(ns.load_ledger()),
                       "ts": time.strftime("%F %T")}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        log(f"[round_state] 写失败：{e}")


def read_round_state() -> dict:
    try:
        with open(_round_state_file(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def completed_round(led: dict) -> int:
    rs = []
    for rec in (led.get("rounds") or []):
        try:
            rs.append(int(rec.get("round", 0)))
        except Exception:
            pass
    return max(rs) if rs else 0


def resume_round(led: dict, requested: int) -> int:
    comp = completed_round(led)
    try:
        started = int(read_round_state().get("last_started_round", 0))
    except Exception:
        started = 0
    auto = max(comp + 1, started, 1)
    if requested < auto:
        log(f"[resume] --start-round {requested} < 自算续跑轮 {auto}"
            f"（台账已完成={comp} 最近开跑={started}）⇒ 从 R{auto} 续跑（不重跑已完成轮）")
        return auto
    return requested


def main() -> None:
    ap = argparse.ArgumentParser(description="自适应搜索调度器 v3（目标：击败 Baseline_v2.0 / trial_9863）")
    ap.add_argument("--deadline", default="2026-11-12 09:00", help="硬截止")
    ap.add_argument("--rounds", type=int, default=99, help="最多轮数")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--trial-base", type=int, default=TRIAL_BASE)
    ap.add_argument("--cross-soup", action="store_true",
                    help="额外生成跨模型 Model Soup（2026-11-11 实测会崩 ⇒ 默认关闭，仅留 SWA）")
    ap.add_argument("--swa-top-n", type=int, default=6, help="训练后自动 SWA 覆盖的臂数")
    ap.add_argument("--n-seed", type=int, default=6, help="每轮种子扫描臂数")
    ap.add_argument("--n-mut", type=int, default=18, help="每轮突变格臂数")
    ap.add_argument("--start-round", type=int, default=1, help="起始轮号（重启续跑用）")
    ap.add_argument("--selftest", action="store_true", help="CPU-only 自检（不碰台账、不占 GPU）")
    args = ap.parse_args()

    deadline_ts = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M"))
    gpus = [int(x) for x in args.gpus.replace(" ", "").split(",") if x.strip()]
    led0 = ns.load_ledger()
    used = [int(k) for k in (led0.get("arms") or {})]
    start_trial = max([args.trial_base - 1] + used) + 1
    next_trial = [start_trial]
    log(f"trial 号从 {start_trial} 起（台账已用 {len(used)} 臂，最大 {max(used) if used else '-'}）")
    args.start_round = resume_round(led0, args.start_round)

    if args.selftest:
        selftest()
        return

    log(f"=== ADAPTIVE v3 RUNNER START | deadline={args.deadline} | gpus={gpus} | "
        f"每轮 {args.n_seed} 种子 + {args.n_mut} 突变 ===")
    for r in range(args.start_round, args.start_round + args.rounds):
        # 剩余时间不够跑一整轮 ⇒ 按剩余时间缩小臂数，而不是硬开一轮把时间花光
        remain_min = (deadline_ts - time.time()) / 60.0
        n_seed, n_mut = max(args.n_seed, SEED_SCAN_TARGET), min(args.n_mut, MUT_CAP)
        # 🟢 2026-11-13 11:05：估计与缩臂用**同一套收缩预留**（否则末轮算不下 ⇒ 调度器提前停）
        _swa_res = swa_reserve(remain_min)
        _prio_res = prio_reserve(remain_min, _swa_res)
        est = plan_minutes(n_seed + n_mut + _prio_res + NEAR_MISS_CELLS * NEAR_MISS_SEEDS,
                           swa_arms=_swa_res, fixed=fixed_cost(remain_min))
        if remain_min < est + 10:
            n_seed, n_mut = shrink_plan(remain_min, n_seed, n_mut)
            log(f"[R{r}] 剩余 {remain_min:.0f} min ⇒ 缩臂到 {n_seed} 种子 + {n_mut} 突变")
        if remain_min < MIN_ROUND_MIN:
            log(f"距硬截止不足 {MIN_ROUND_MIN:.0f} min ⇒ 收工（不再开新轮）")
            break
        write_round_state(r)
        # 🟢 2026-11-13 16:20：末轮**训练臂硬上限**（旧 shrink_plan 只缩 n_seed/n_mut，
        #   而 build_train_arms 里两者被 max()/min() 覆盖 ⇒ 缩臂从未生效 ⇒ 末轮必冲过 09:00）
        _budget = round_arm_budget(remain_min)
        if _budget is not None:
            log(f"[R{r}] 剩余 {remain_min:.0f} min ⇒ 末轮训练臂硬上限 {_budget}（按价值顺序砍）")
        try:
            run_round(r, deadline_ts, gpus, next_trial, cross_soup=args.cross_soup,
                      swa_top_n=args.swa_top_n, n_seed=n_seed, n_mut=n_mut,
                      arm_budget=_budget)
        except Exception as e:  # noqa: BLE001
            log(f"[R{r}] 异常：{e}（继续下一轮）")
            import traceback
            log(traceback.format_exc())
        write_round_state(r)
        if time.time() > deadline_ts:
            log("超过截止 ⇒ 退出")
            break
    log("=== ADAPTIVE v3 RUNNER DONE ===")


if __name__ == "__main__":
    main()

