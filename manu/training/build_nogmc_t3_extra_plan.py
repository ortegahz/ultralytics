#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成「补投臂」清单（2026-11-03 T3 前沿轮中期自适应用）。

用法
====
```bash
# 看有哪些候选
python manu/training/build_nogmc_t3_extra_plan.py --list

# 选若干条生成补投计划
python manu/training/build_nogmc_t3_extra_plan.py --pick T3_W2_LR30_15EP_S4,T3_W2_LR40_8EP \
    --out manu/training/nogmc_t3_extra_plan.json

# 然后（不重启调度器、不动正在跑的臂）执行
python manu/training/enqueue_extra_arms.py \
    --out-root /mnt/data/siping/runs/nogmc_t3 --study-name mix_ratio_search_nogmc_t3 \
    --plan manu/training/nogmc_t3_plan.json --plan-extra manu/training/nogmc_t3_extra_plan.json
```

三条纪律
========
1. 补投的臂**排在队尾**（trial number 更大 ⇒ FIFO 靠后）⇒ 想让它们**优先**跑，
   必须同时**剪掉**等量的低价值 WAITING 臂（`--prune-last N` 或 `--prune-roles`），
   否则总臂数会超过调度器的 `--n-trials` 上限（30）⇒ **新臂永远不会被执行**。
2. 剪掉的臂是**事前判断为低价值**的那些 ⇒ 剪错不可逆（Optuna 4.9 没有 `delete_trial`；
   标 PRUNED 后指纹仍在去重表里 ⇒ 重新入队同参同种子会被跳过 ⇒ 想再跑它必须换 seed）。
3. 形状超参**一律沿用 focus study 的同一组值** ⇒ 补投臂与原 30 条臂**受控可比**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SHAPE = {
    "warmup_epochs": 1.5,
    "weight_decay": 0.000269,
    "focal_beta": 2.75,
    "offset_weight": 0.4,
    "mosaic": 0.15,
    "scale": 0.1,
    "translate": 0.05,
    "clip_max_norm": 5.65,
    "freeze_bn": False,
    "backbone_lr_scale": 1.0,
    "freeze_backbone_epochs": 0,
}

# 候选：role → (mix, lr, epochs, seed, init_x, 说明)
CANDIDATES = {
    # ---- A 类：给「T3 最高家族」补种子（跨种子 soup 的燃料）----
    "T3_W2_LR30_15EP_S4": ("w2.0", 3.0, 15, 340, "",
                           "★w2.0·lr3.0·15ep 第 4 种子（ep15 家族 soup）"),
    "T3_W3_LR30_15EP_S4": ("w3.0", 3.0, 15, 341, "",
                           "★w3.0·lr3.0·15ep 第 4 种子（最可能的严格 PASS 家族）"),
    "T3_W2_LR35_15EP_S2": ("w2.0", 3.5, 15, 342, "",
                           "w2.0·lr3.5·15ep 第 2 种子"),
    "T3_W3_LR2882_15EP_S2": ("w3.0", 2.8821, 15, 343, "",
                             "cs_t0008 家族（严格 PASS 家族）加长训练的第 2 种子"),
    # ---- B 类：把 lr 天花板再往上顶（ep8 ⇒ 便宜，92 min）----
    "T3_W2_LR40_8EP": ("w2.0", 4.0, 8, 344, "",
                       "★lr 首次到 4.0（+2.3 pp T3 预期）⇒ ep8 便宜探顶（lr 汇率的 1.0 跨度实测）"),
    "T3_W2_LR40_15EP": ("w2.0", 4.0, 15, 346, "",
                        "★★T3 冲 RC2 的关键一条：ep15 段 lr 3.0→3.5 的受控对是"
                        "「ΔT3 +0.258 pp、ΔT1 **+0.101 pp**」⇒ lr 的 T1 代价在 ep15 上已消失 ⇒ "
                        "lr 推到 4.0 期望 T3 0.897~0.900、T1 不掉"),
    "T3_W3_LR40_8EP": ("w3.0", 4.0, 8, 345, "",
                       "w3.0 侧 lr=4.0（T1 更好）"),
    "T3_W2_LR35_8EP_S2": ("w2.0", 3.5, 8, 346, "",
                           "w2.0·lr3.5·8ep 第 2 种子"),
    # ---- C 类：soup 家族补成员（ep8 ⇒ 便宜）----
    "INS_T0033_S7": ("w3.0", 3.0, 8, 347, "",
                     "trial_0033 家族补种子（4 种子 soup）"),
    "INS_T0027_S6": ("w2.0", 3.0, 8, 348, "",
                     "T3 最高家族（w2.0·lr3.0·8ep）补种子（4 种子 soup）"),
    "INS_T0008_S6": ("w3.0", 2.8821, 8, 349, "",
                     "cs_t0008 家族补种子（6 种子 soup）"),
    # ---- D 类：init 换 soup_strict3（起点 T1 更高）----
    "T3_W2_LR30_15EP_SOUPSRC_S2": ("w2.0", 3.0, 15, 350, "soup_strict3",
                                   "soup_strict3 起手 + 15ep"),
    # ---- F 类：soup 初始化（2026-11-03 15:30 新发现 ⇒ 第一优先级）----
    # 实测：同配方只差 init（rc2 → soup_strict3）在 ep15 上是**双赢**：
    #   w2.0·lr3.0·15ep：T3 0.891676 → **0.897759**（+0.608 pp）、T1 0.890445 → 0.891252（+0.081 pp）
    #   w3.0·lr3.0·15ep：T3 0.885724 → **0.893239**（+0.752 pp）、T1 0.892730 → 0.891356（−0.137 pp）
    # ⇒ 汇率 7.5 / 5.5 pp T3 per 1 pp T1 ⇒ **优于前沿斜率 4.4 ⇒ 它把前沿向外推**
    # ⇒ 而 w2.0·lr3.5·15ep（rc2 init）T3 已 0.894255 ⇒ 再叠 soup init 有望摸到 **0.900**
    "T3_W2_LR35_15EP_SOUPSRC": ("w2.0", 3.5, 15, 360, "soup_strict3",
                                 "★★★T3 冲 0.90：w2.0·lr3.5·15ep 再叠 soup 初始化"
                                 "（rc2 init 版已 0.894255，soup init 版按 +0.6 pp 期望 ≈0.900）"),
    "T3_W2_LR30_15EP_SOUPSRC_S2": ("w2.0", 3.0, 15, 361, "soup_strict3",
                                    "★★★当前最强配置（w2.0·lr3.0·15ep·soup，T3 0.897759）的**第 2 个种子**"
                                    " ⇒ 凑 2 种子跨种子 soup（soup 的 T1 增益是唯一零训练成本手段）"),
    "T3_W3_LR30_15EP_SOUPSRC_S2": ("w3.0", 3.0, 15, 362, "soup_strict3",
                                    "★★★w3.0·lr3.0·15ep·soup（T3 0.893239、T2 FP 32 最低）的第 2 个种子"
                                    " ⇒ 它的 2 种子 soup 有望同时拿到 T3 与 T1 红线"),
    "T3_W2_LR35_8EP_SOUPSRC": ("w2.0", 3.5, 8, 363, "soup_strict3",
                                "★★便宜探针：lr3.5 + soup 初始化在 ep8 上的读数"
                                "（ep8 便宜 ⇒ 92 min；兼作「lr3.5×soup 初始化」是否稳定的体检）"),

    # ---- G 类：ep20 × soup 初始化（2026-11-03 17:10 新发现 ⇒ 第一优先级）----
    # 实测：**T1 在 ep15 是凹点、ep20 回升**（同配方 w2.0·lr3.0）：
    #   ep8  → T1 0.893555 / T3 0.880511
    #   ep15 → T1 0.890445 / T3 0.891676   ← T1 凹点
    #   ep20 → T1 0.893670 / T3 0.890249   ← **T1 回升 +0.323 pp，只掉 T3 0.143 pp**
    # ⇒ ep20 是前沿上另一个更靠右的点；再叠 soup 初始化（ep15 上 ΔT3 +0.6 pp）⇒ 双目标候选
    "T3_W2_LR30_20EP_SOUPSRC": ("w2.0", 3.0, 20, 370, "soup_strict3",
                                 "★★★★★双目标最强候选：w2.0·lr3.0·**20ep**·soup 初始化"
                                 "（ep20 的 T1 回升 + soup 的 T3 加成，期望 T1≈0.895、T3≈0.895）"),
    "T3_W3_LR2882_8EP_SOUPSRC": ("w3.0", 2.8821, 8, 371, "soup_strict3",
                                  "★★★★★双目标候选（便宜版）：T1 最好的家族"
                                  "（w3.0·lr2.8821·8ep，cs_t0008 的家族）叠 soup 初始化 ⇒ ep8 只要 93 min"),
    "T3_W3_LR2882_20EP_SOUPSRC": ("w3.0", 2.8821, 20, 372, "soup_strict3",
                                   "★★★★T1 最好的家族在 ep20 + soup 初始化 ⇒ 期望 T1 最高"),
    "T3_W2_LR40_15EP_SOUPSRC": ("w2.0", 4.0, 15, 373, "soup_strict3",
                                 "★★★★堆叠版 T3 上限：lr4.0 × ep15 × soup 初始化"),
    "T3_W3_LR35_15EP_SOUPSRC": ("w3.0", 3.5, 15, 374, "soup_strict3",
                                 "★★★ep15 段「兼顾型」最好的一条：w3.0·lr3.5·15ep 的 T1 是 ep15 里最高"
                                 "（0.892856）再叠 soup 初始化"),
    "T3_W2_LR30_20EP_S2": ("w2.0", 3.0, 20, 375, "",
                            "★★w2.0·lr3.0·20ep 的第 2 种子 ⇒ 2 种子 soup（ep20 家族的 T1 回升可复现性）"),
    "T3_W2_LR40_8EP_SOUPSRC": ("w2.0", 4.0, 8, 376, "soup_strict3",
                                "★★便宜探针：lr4.0 × soup 初始化在 ep8 上的读数（兼作稳定性体检）"),

    # ---- I 类（2026-11-03 18:15 新增）：**补回判断文档点名的那条路径** ----
    # 判断文档（用户给的）核心建议：「跨种子 Soup 是吃掉 76 个 TP 的终极利器 ——
    #   把这个手法原样复制到 trial_0033 配方（w3.0 · lr3.0 · 8ep · rc2）上，
    #   只需要平移 +0.033 pp，就能顺理成章把 T1 强行拔过 0.895897 的红线。」
    # ⚠️ 落地障碍：`nogmc_t3_families.json` 的 `t0033` 家族引用了 t3 study 的
    #    `INS_T0033_S4` / `INS_T0033_S5` 两个成员，而它们在 **17:10 第三次剪枝**里
    #    被剪掉了 ⇒ 家族只剩 focus 的 trial_0033 **1 个粒子 ⇒ 构不成 soup**。
    # ⇒ 补跑一条**同配方同权重空间**（rc2-init，**不是** soup-init）的 w3.0·lr3.0·8ep，
    #    role 沿用 `INS_T0033_S5`（families.json 已引用该 role ⇒ 变体门禁零改动即生效）。
    #    ep8 只要 ~93 min ⇒ 零 GPU 成本的「跨种子 soup」得以真正落地。
    "INS_T0033_S5": ("w3.0", 3.0, 8, 382, "",
                     "★★★★★补回判断文档点名的 t0033 家族第 2 粒子"
                     "（w3.0·lr3.0·8ep·rc2-init，与 focus 的 trial_0033 同配方同权重空间）"
                     " ⇒ 让 t0033 家族能真正构建跨种子 soup（吃 76 个 TP 的路径）"),

    # ---- J 类（2026-11-04 09:00 新增）：把 `t0033` 家族补到 4 粒子 ----
    # 2026-11-04 09:01 实测：`cs_t0033`（focus `trial_0033` + 昨晚 `t45` 的 2 粒子跨种子平均）
    #   T1 = **0.894861**（红线 0.8958967，**只差 0.001036 pp ≈ 2.5 个 TP**）、T2 0.998679/FP 40 ✅、T3 0.872633 ✅
    # ⇒ 而同配方 4 粒子的 `cs_t0008` 的 T1 = **0.895901 已过线**
    # ⇒ **「同配方粒子越多、soup 的 T1 越高」**（2 粒子 0.894861 → 4 粒子 0.895901）
    # ⇒ 补 2 条同配方同权重空间（**rc2-init**）的种子 ⇒ t0033 家族变 4 粒子 ⇒ 有望跨过红线
    # ⇒ 这就是判断文档点名的「补跑 2 个同配方种子做跨种子平均」的**完整落地**。
    "INS_T0033_S6": ("w3.0", 3.0, 8, 383, "",
                     "★★★★★补 t0033 家族第 3 粒子（w3.0·lr3.0·8ep·rc2-init，与 focus trial_0033 同空间）"
                     " ⇒ 目标：cs_t0033 从 2 粒子到 3~4 粒子 ⇒ T1 从 0.894861 推向红线 0.895897"),
    "INS_T0033_S7": ("w3.0", 3.0, 8, 384, "",
                     "★★★★★补 t0033 家族第 4 粒子（同上）⇒ 凑齐 4 粒子（对齐 cs_t0008 的 4 粒子 T1=0.895901）"),

    # ---- E 类：ep 阶梯补点 ----
    "T3_W2_LR30_20EP_S2": ("w2.0", 3.0, 20, 351, "",
                           "w2.0·lr3.0·20ep 第 2 种子（ep20 家族）"),

    # ---- H 类（2026-11-03 18:15 新增）：**把三条实测汇率叠在一起** ----
    # 受控对（全部来自本轮 nogmc_t3 已完成的 11 条臂，单变量对照）：
    #   ① lr 3.0→3.5（w2.0·ep15·rc2-init）：t0000 → t0002 ⇒ ΔT1 **+0.101 pp**、ΔT3 **+0.258 pp**
    #   ② ep 15→20（w2.0·lr3.0·rc2-init）：t0000 → t0004 ⇒ ΔT1 **+0.323 pp**、ΔT3 **−0.143 pp**
    #   ③ init rc2→soup_strict3（w2.0·lr3.0·ep15）：t0000 → t0005 ⇒ ΔT1 **+0.081 pp**、ΔT3 **+0.608 pp**
    # 三重重叠外推（可加性近似）：T1 ≈ 0.890445+0.001+0.0032+0.0008 = **0.8955**（红线 0.8958967）
    #                              T3 ≈ 0.891676+0.0026−0.0014+0.0061 = **0.8989**（> RC2 0.897694）
    # ⇒ 这是本轮**唯一**同时「T1 贴线 + T3 超 RC2」的候选 ⇒ 最高优先级补投
    "T3_W2_LR35_20EP_SOUPSRC": ("w2.0", 3.5, 20, 380, "soup_strict3",
                                "★★★★★**双目标最强候选**：w2.0·lr3.5·**20ep**·soup_strict3"
                                "（lr3.5 与 ep20 与 soup 三条实测汇率叠加 ⇒ 期望 T1≈0.8955 贴红线、"
                                "T3≈0.8989 超 RC2）"),
    # 给 t32（w2.0·lr3.5·15ep·soup，当前期望 T3 最高的那条）补**第 2 个种子**
    # ⇒ 铁律：决策单元 = 配置（同超参多种子中位）⇒ 单种子臂进不了正式交接判定
    "T3_W2_LR35_15EP_SOUPSRC_S2": ("w2.0", 3.5, 15, 381, "soup_strict3",
                                   "★★★★w2.0·lr3.5·15ep·soup_strict3 的**第 2 个种子**"
                                   " ⇒ 让 t32 这条「期望 T3 最高」的臂凑齐 2 种子中位（铁律：决策单元=配置）"),

    # ---- K 类（2026-11-04 11:40 新增）：把 **T3 冠军家族**补到 3 种子 ----
    # 2026-11-04 11:20 实测：`trial_0043`（w2.0·lr3.5·20ep·soup_strict3 起手）
    #   T3 = **0.901305**（破 0.90，比带 GMC 的 RC2 0.897694 还高 **+0.361 pp**）、T2 0.998829/FP 35 ✅
    #   ⇒ 但它是**单种子**（n=1）⇒ 按铁律「决策单元 = 配置（同超参多种子中位）」，
    #      「T3 超 RC2」这个结论目前只有**一次测量**支撑 ⇒ 补 2 个同配方同权重空间的种子。
    #   预期：同配方种子噪声实测只有 **0.02~0.19 pp**（T3）⇒ 3 种子中位应稳在 0.895~0.901。
    #   家族 `t0043` 已写进 `nogmc_t3_families.json` ⇒ 训完自动能构建 3 粒子跨种子 soup。
    "INS_T0043_S1": ("w2.0", 3.5, 20, 385, "soup_strict3",
                     "★★★★★**T3 冠军家族第 2 种子**（w2.0·lr3.5·20ep·soup_strict3 起手，"
                     "与 trial_0043 同配方同权重空间）⇒ 把「T3=0.901305 超 RC2 +0.361 pp」"
                     "从单次测量升级成 3 种子中位（铁律：决策单元=配置）"),
    "INS_T0043_S2": ("w2.0", 3.5, 20, 386, "soup_strict3",
                     "★★★★★**T3 冠军家族第 3 种子**（同上）⇒ 凑齐 3 种子 ⇒ 可构建 "
                     "cs_t0043（3 粒子跨种子 soup）"),

    # ---- L 类（2026-11-04 15:55 新增）：**clip_max_norm 6.5 的受控实验** ----
    # 2026-11-04 15:50 实测（前沿报告的逐旋钮汇率）：
    #   `clip_max_norm` 是**唯一「T1 正向」的旋钮**（ΔT1 **+0.0977 pp** / ΔT3 −0.1347 pp，n=1），
    #   而那个 n=1 受控对是 `focust0030`(clip 4.05) vs `weekendt0077`(clip 4.054)
    #   ⇒ **只差 0.004 的 clip 却报出 +0.098 pp T1** ⇒ 极可能是**种子噪声冒充旋钮效应**（未受控）。
    #   ⇒ 而**若它是真效应**：`t0033 配方`（w3.0·lr3.0·8ep·rc2-init，clip 5.65）的中位 T1 ≈ 0.89566
    #      +0.098 pp ⇒ **0.8965 ⇒ 过 T1 红线**，而 T3 0.872645 −0.135 pp ⇒ **0.8713 ⇒ 仍 ≥ 0.87**
    #      ⇒ ★ **这将是「T1 过线 + T3 ≥ 0.87 同一配置」的第一个可行解**
    #      （而 §3.9.10 刚证明：按现有 177 条记录，这条路是**互斥**的 ⇒ 必须实测）。
    #   ⇒ 本组用 **4 个种子**（clip 6.5）对 **4 个种子的 clip 5.65 基线**（focus t0033 + t45/t47/t48）
    #      ⇒ n=4 vs n=4 的**受控**比较 ⇒ 旋钮效应与种子噪声可分离。
    #   ⇒ 配方除 clip 外与 t0033 家族**逐键一致**（同 rc2-init 权重空间）。
    "INS_T0033_C1": ("w3.0", 3.0, 8, 387, "",
                     "★★★**clip_max_norm 6.5 受控实验第 1 种子**（w3.0·lr3.0·8ep·rc2-init，"
                     "其余超参与 t0033 家族逐键一致）⇒ 验证「clip 抬高 T1 +0.098 pp」是"
                     "**真旋钮效应还是种子噪声** ⇒ 若为真 ⇒ T1 过线 + T3≥0.87 的兼得产物"),
    "INS_T0033_C2": ("w3.0", 3.0, 8, 388, "",
                     "★★★clip_max_norm 6.5 受控实验第 2 种子"),
    "INS_T0033_C3": ("w3.0", 3.0, 8, 389, "",
                     "★★★clip_max_norm 6.5 受控实验第 3 种子"),
    "INS_T0033_C4": ("w3.0", 3.0, 8, 390, "",
                     "★★★clip_max_norm 6.5 受控实验第 4 种子（n=4 对 n=4 受控比较）"),
}

# role → clip_max_norm 覆盖（默认取 SHAPE 的 5.65）
CLIP_OVERRIDE = {
    "INS_T0033_C1": 6.5,
    "INS_T0033_C2": 6.5,
    "INS_T0033_C3": 6.5,
    "INS_T0033_C4": 6.5,
}


def build(picks):
    out = []
    for role in picks:
        mix, lr, ep, seed, init_x, note = CANDIDATES[role]
        p = dict(SHAPE)
        p["mix_spec"] = mix
        p["lr_scale"] = lr
        p["epochs"] = ep
        p["init_from"] = "rc2"
        if role in CLIP_OVERRIDE:
            p["clip_max_norm"] = CLIP_OVERRIDE[role]
        if init_x:
            p["init_from_x"] = init_x
        out.append({"role": role, "seed": seed, "note": note, "params": p})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pick", default="", help="逗号分隔的 role")
    ap.add_argument("--out", default="manu/training/nogmc_t3_extra_plan.json")
    args = ap.parse_args()

    if args.list or not args.pick:
        print("候选补投臂：")
        for role, (mix, lr, ep, seed, init_x, note) in CANDIDATES.items():
            print("  %-28s mix=%-8s lr=%-7s ep=%-3s seed=%s init_x=%-14s %s"
                  % (role, mix, lr, ep, seed, init_x or "-", note))
        return 0

    picks = [x.strip() for x in args.pick.split(",") if x.strip()]
    miss = [x for x in picks if x not in CANDIDATES]
    if miss:
        print("未知 role：%s" % miss)
        return 1
    items = build(picks)
    Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("已写 %s：%d 条" % (args.out, len(items)))
    for it in items:
        p = it["params"]
        print("  %-28s mix=%-8s lr=%-7s ep=%-3s seed=%s init_x=%s"
              % (it["role"], p["mix_spec"], p["lr_scale"], p["epochs"], it["seed"],
                 p.get("init_from_x", "-")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
