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
ROUND_FIXED_MIN = 25.0      # 门禁 + dual + 板卡 + 报告
TRAIN_MIN_PER_ARM = 4.0     # 4 卡并行后的墙钟摊销
EVAL_MIN_PER_ARM = 1.3      # NanoBench 三档缓存
MIN_ROUND_MIN = 60.0        # 剩余时间低于此值 ⇒ 收工，不再开新轮

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
    ("sched7_lrf0.10", dict(sched=7, lr_floor=0.10),
     "cosine 周期 5→7 + LR 保底 10%（nano 峰右移两格）"),
    ("sw8.0_lrf0.10", dict(sw=8.0, lr_floor=0.10),
     "小目标剂量 8.0 + LR 保底 10%（9822 sw8 官方 T1 已过线，nano 只差 0.003545）"),
    ("lrf0.10", dict(lr_floor=0.10),
     "LR 保底 10%（12053 复现：nano 过线、官方 T1 只差 0.000984）"),
]

# 优先格淘汰线：n>=PRIORITY_MIN_N 后，若均值 nano 低于基线配方均值 PRIORITY_MEAN_SLACK ⇒ 停喂
PRIORITY_MIN_N = 4
PRIORITY_MEAN_SLACK = 0.002
# 优先格每轮臂数上限（8 格 × 2 seed）⇒ 时间预算里先给它留位
PRIORITY_ARMS_EST = 12


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 时间预算：按剩余时间决定本轮跑多少臂（保证最后一轮能在硬截止前跑完）
# ---------------------------------------------------------------------------
def plan_minutes(n_arms: int) -> float:
    """跑 n_arms 臂预计占用的墙钟时间（分钟）。"""
    waves = (n_arms + 3) // 4
    return waves * 16.0 + n_arms * EVAL_MIN_PER_ARM + ROUND_FIXED_MIN


def shrink_plan(remain_min: float, n_seed: int, n_mut: int) -> tuple[int, int]:
    """按剩余时间缩臂：先砍突变格，再砍种子格（种子格保证稳健线 ⇒ 至少留 2 个）。

    ⚠️ 2026-11-12：优先格（PRIORITY_ARMS_EST 臂）**先占预算**，剩下的才分给种子格/突变格
    ⇒ 缩臂时不会把「离基线只差一档」的优先格挤出最后一轮。
    """
    per_arm = TRAIN_MIN_PER_ARM + EVAL_MIN_PER_ARM
    fit = int(max(4.0, (remain_min - ROUND_FIXED_MIN) / per_arm)) - PRIORITY_ARMS_EST
    fit = max(4, (fit // 4) * 4)
    seed_fit = max(2, min(n_seed, fit // 3))
    mut_fit = max(1, fit - seed_fit)
    return seed_fit, mut_fit


# ---------------------------------------------------------------------------
# 基线与单臂判据（沿用 v2，数值口径不变）
# ---------------------------------------------------------------------------
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
        if bg and g["mean_nano"] <= bg["mean_nano"] and not is_base_group:
            continue
        if bg and g["mean_nano4"] < bg["mean_nano4"] - 0.002:
            continue
        if bg and g["mean_f1"] < bg["mean_f1"] - 0.001:
            continue
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
        out.append({"recipe": g["key"], "n": g["n"], "mean_nano": g["mean_nano"],
                    "mean_nano4": g["mean_nano4"], "mean_f1": g["mean_f1"],
                    "mean_micro": g["mean_micro"], "trials": g["trials"],
                    "beat_trials": winners, "params": g["params"]})
    out.sort(key=lambda r: -r["mean_nano"])
    return out


# ---------------------------------------------------------------------------
# ① 合成臂（零训练）：只出 SWA（跨模型 soup 实测会崩，默认关闭）
# ---------------------------------------------------------------------------
def build_synth_arms(led: dict, champ: dict, base: dict, round_idx: int,
                     next_trial: list[int], cross_soup: bool = False) -> list[tuple]:
    """生成合成臂并把权重写到 trials/trial_X/weights/epoch5.pt。

    🔴 2026-11-11 14:46 实测：**跨配方/跨种子的 Model Soup 在本任务上会崩**
       （soup_multiseed nano 0.2422 · soup_dim_top2 0.3046 · soup_same_sw 0.4627 · soup_top3 0.6545，
        全部远低于基线 0.7086）⇒ 原因是这些权重**不在同一 loss 盆地**，平均互相抵消。
    🟢 **同一臂相邻 epoch 的 SWA 成立**（swa_late nano 0.708885 **已略超基线**）。
    ⇒ 故默认 **只出 SWA**；`cross_soup=True` 才额外生成跨模型 soup。
    """
    arms = []
    if not champ:
        return arms
    cdir = os.path.dirname(champ["ckpt"])
    win = champ["epoch"]
    windows = [
        ("swa_neigh", list(range(max(1, win - 2), win + 3)), "champion 臂 epoch{~2}~{~+2}"),
        ("swa_late", list(range(win, win + 5)), "champion 臂 epoch{~0}~{~+4}（实测最有效）"),
        ("swa_wide", list(range(max(1, win - 2), win + 7)), "champion 臂 epoch{~2}~{~+6} 宽窗"),
        ("swa_early", list(range(max(1, win - 2), win + 2)), "champion 臂 epoch{~2}~{~+1} 早窗"),
    ]
    for name, eps, note in windows:
        mem = [f"{cdir}/epoch{e}.pt" for e in eps]
        mem = [m for m in mem if os.path.isfile(m)]
        if len(mem) < 2:
            continue
        trial = next_trial[0]
        next_trial[0] += 1
        out = f"{ns.ROOT}/trials/trial_{trial}/weights/epoch{SYNTH_EPOCH}.pt"
        try:
            average_checkpoints(mem, out, epoch=SYNTH_EPOCH, note=f"SWA {name}: {note}")
        except Exception as e:  # noqa: BLE001
            log(f"[synth] {name} 放弃：{e}")
            continue
        st = ns.arm_state(led, trial)
        st.update(trial=trial, desc=f"★ 合成臂 {name}：{note}", trained=True, synth=True,
                  synth_members=mem, synth_note=note,
                  params={"init": mem[0], "synth": name, "epochs": SYNTH_EPOCH,
                          "sched": SYNTH_EPOCH, "sw": (champ.get("params") or {}).get("sw"),
                          "mix": "w2.0", "warmup": 1.5, "lr": 3.5, "fb": 2.75, "ow": 0.4,
                          "wd": 0.000269, "mosaic": 0.15, "scale": 0.1, "translate": 0.05,
                          "clip": 5.65, "backbone_lr": 1.0, "uf": 0, "head_src": "p3",
                          "native_grid": 0, "mr": 1, "swl": 1.0, "thr": 16.0,
                          "maxside": 4.0, "lr_floor": 0.0, "hf": 0, "hfsrc": "b1", "hfch": 1})
        log(f"[synth] {name} ⇒ trial {trial}（成员 {len(mem)} 个）")
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
    """
    groups = recipe_stats(led) or {}
    bg = base_group(led, groups)
    bmean = (bg or {}).get("mean_nano") or 0.0
    out = []
    for name, patch, why in PRIORITY_PATCHES:
        p = dict(base_params)
        p.update(patch)
        p["epochs"] = max(5, int(p.get("sched") or 5))
        grp = groups.get(recipe_key(p))
        if (grp and grp.get("n", 0) >= PRIORITY_MIN_N
                and (grp.get("mean_nano") or 0.0) < bmean - PRIORITY_MEAN_SLACK):
            continue
        out.append((name, p, why))
    return out


def build_train_arms(led: dict, champ: dict, round_idx: int, next_trial: list[int],
                     n_seed: int = 6, n_mut: int = 18) -> list[tuple]:
    """一轮的训练臂：
      ① 种子扫描 n_seed 臂（冠军配方、不同 seed）⇒ 建立该配方的 seed 分布（稳健线）
      ② 突变格 n_mut 臂（冠军配方 + 一处改动）⇒ 探索
      ③ 待复现配方 replication_pending（上一轮单臂打平过、但还没第二个 seed 的配方）
    **所有臂都从 base init 起训**（二次退火路径已证伪，见文件头证据①）。
    """
    base_params = clean_recipe(champ["params"])
    arms = []
    seed0 = SEED_BASE + round_idx * 97

    # ③ 待复现配方（优先级最高：能把「运气臂」变成「真配方」）
    pend = ((led.get("search_state") or {}).get("replication_pending") or [])[:4]
    _gs_all = recipe_stats(led) or {}
    for rk in pend:
        grp = _gs_all.get(rk)
        if not grp:
            continue
        # 🔴 T2 死区配方不复现（见 recipe_t2_dead）
        if recipe_t2_dead(led, grp, base_ref(led)):
            continue
        p = clean_recipe(grp["params"])
        for k in range(2):
            trial = next_trial[0]
            next_trial[0] += 1
            arms.append((trial, seed0 + 500 + len(arms),
                         dict(p, epochs=max(5, int(p.get("sched") or 5))),
                         f"[R{round_idx}] ★★ 复现 {rk.split('|')[0][:40]} seed{k+1}（上一轮单臂打平基线）"))

    # ⓿ 优先格（2026-11-12 新增）：每格每轮 2 个 seed
    #   ① 配方效应 vs 种子噪声必须分开量（单臂读数分不清）
    #   ② 攒到 PRIORITY_MIN_N 个 seed 后按「均值 vs 基线配方均值」淘汰差格
    for _name, _p, _why in priority_cells(led, base_params):
        for _k in range(2):
            _t = next_trial[0]
            next_trial[0] += 1
            arms.append((_t, seed0 + 200 + len(arms), dict(_p),
                         f"[R{round_idx}] ★★ 优先格 {_name} seed{_k+1}：{_why}"))

    # ① 种子扫描
    for i in range(n_seed):
        trial = next_trial[0]
        next_trial[0] += 1
        p = dict(base_params, epochs=5)
        arms.append((trial, seed0 + i, p,
                     f"[R{round_idx}] ★ 种子扫描 {i+1}/{n_seed}（冠军配方复现：sw={base_params.get('sw')}）"))

    # ② 突变格
    for name, patch, why in next_mutations(led, base_params, n_mut):
        trial = next_trial[0]
        next_trial[0] += 1
        p = dict(base_params)
        p.update(patch)
        p["epochs"] = max(5, int(p.get("sched") or 5))
        arms.append((trial, seed0 + 100 + len(arms), p, f"[R{round_idx}] ★ 突变 {name}：{why}"))
    return arms


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
        best = max(vals, key=lambda v: v["nano"])
        cands.append((best["nano"], t, best.get("epoch")))
    cands.sort(reverse=True)
    arms = []
    for nano, t, ep in cands[:top_n]:
        if ep is None:
            continue
        cdir = f"{ns.ROOT}/trials/trial_{t}/weights"
        mem = [f"{cdir}/epoch{e}.pt" for e in range(max(1, int(ep) - 1), int(ep) + 2)]
        mem = [m for m in mem if os.path.isfile(m)]
        if len(mem) < 2:
            continue
        trial = next_trial[0]
        next_trial[0] += 1
        out = f"{ns.ROOT}/trials/trial_{trial}/weights/epoch{SYNTH_EPOCH}.pt"
        try:
            average_checkpoints(mem, out, epoch=SYNTH_EPOCH,
                                note=f"SWA(after-train) of trial {t} ep{ep}")
        except Exception as e:  # noqa: BLE001
            log(f"[post-SWA] trial {t} 放弃：{e}")
            continue
        src = (led.get("arms") or {}).get(str(t)) or {}
        params = dict(src.get("params") or {})
        params.update(init=mem[0], synth=f"post_swa_of_{t}", epochs=SYNTH_EPOCH, sched=SYNTH_EPOCH)
        st2 = ns.arm_state(led, trial)
        st2.update(trial=trial, desc=f"★ 训练后 SWA：trial {t} ep{ep} 的相邻 epoch 等权平均",
                   trained=True, synth=True, synth_members=mem, synth_note="post-train SWA",
                   params=params)
        log(f"[post-SWA] trial {t} ep{ep} → trial {trial}（{len(mem)} 成员）")
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
              n_mut: int = 18) -> dict:
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

    champ = pick_champion_recipe(led, base)
    cg = champ.get("group") or {}
    bg = champ.get("base_group") or {}
    log(f"[R{round_idx}] 冠军配方 n={cg.get('n', 0)} 均值nano={cg.get('mean_nano', 0):.6f} "
        f"最佳={cg.get('max_nano', 0):.6f} | 基线配方 n={bg.get('n', 0)} "
        f"均值nano={bg.get('mean_nano', 0):.6f} | 已升格={champ.get('adopted')}")

    synth = build_synth_arms(led, champ.get("arm"), base, round_idx, next_trial, cross_soup=cross_soup)
    trains = build_train_arms(led, champ, round_idx, next_trial, n_seed=n_seed, n_mut=n_mut)
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
    post = build_posttrain_swa(led, [t for t, _s, _p, _d in trains], next_trial, top_n=swa_top_n)
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
    gate_set = set(cand[:8]) | set(triad)
    # 🟢 三项全平的臂**排在最前**：官方三轨很贵，若被 deadline 截断，
    #   必须是「唯一有资格升轨的臂」先跑完（ThreadPool 按列表顺序派发）。
    gate_arms = ([a for a in all_arms if a[0] in set(triad)]
                 + [a for a in all_arms if a[0] in set(cand[:8]) and a[0] not in set(triad)])
    for t, _s, _p, _d in all_arms:
        if t not in gate_set:
            st = ns.arm_state(led, t)
            st["official"] = {"skipped": "nano_below_gate_window"}
            st["gate_skipped"] = True
    ns.save_ledger(led)
    log(f"[R{round_idx}] 门禁窗口：{len(gate_arms)}/{len(all_arms)} 臂进官方三轨"
        f"（nano 前 8 + 三项全平 {len(triad)} 臂：{triad}）")
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
    _expect = (2 * sum(1 for rk in _pend
                       if _gs.get(rk) and not recipe_t2_dead(led, _gs[rk], base))
               + 2 + 3
               + 2 * len(priority_cells(led, clean_recipe(champ["params"]))))
    assert len(arms) == _expect, f"训练臂数不对：{len(arms)} != {_expect}"
    for t, s, p, d in arms:
        assert p["init"] == BASE_INIT, f"臂 {t} 的 init 不是 base init"
        assert p["epochs"] == max(5, int(p["sched"])), f"臂 {t} 的 epochs/sched 不自洽"
    log(f"[selftest] 训练臂样例：{[(a[0], a[2].get('sw'), a[2].get('swl'), a[2].get('epochs')) for a in arms]}")
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
        need = plan_minutes(sd + mt)
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
        n_seed, n_mut = args.n_seed, args.n_mut
        est = plan_minutes(n_seed + n_mut + PRIORITY_ARMS_EST)
        if remain_min < est + 10:
            n_seed, n_mut = shrink_plan(remain_min, n_seed, n_mut)
            log(f"[R{r}] 剩余 {remain_min:.0f} min ⇒ 缩臂到 {n_seed} 种子 + {n_mut} 突变")
        if remain_min < MIN_ROUND_MIN:
            log(f"距硬截止不足 {MIN_ROUND_MIN:.0f} min ⇒ 收工（不再开新轮）")
            break
        write_round_state(r)
        try:
            run_round(r, deadline_ts, gpus, next_trial, cross_soup=args.cross_soup,
                      swa_top_n=args.swa_top_n, n_seed=n_seed, n_mut=n_mut)
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

