#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多轨判定标准：安全红线门禁 + 帕累托占优 + 靶向增益（**三级漏斗判定机**）。

为什么不能用加权求和
--------------------
多轨（源域大盘 / 黄金池守门员 / 难例盲测）**绝不能**加权求和成一个总分：
加权平均会用顺风轨的大涨稀释致命缺陷（黄金池虚警爆炸、源域断崖下跌都能被别的轨"平均掉"）。
正确范式是**漏斗**——先否决、再占优、最后才用效用分排名：

```text
候选 M vs 对手 B（并始终对冻结基线做一级红线检查）
   │
   ├─【一级】安全红线一票否决（相对**冻结基线**的绝对红线）
   │      · Track2 黄金池：F1 ≥ 0.985  且  ΔFP_T2 ≤ +10
   │      · Track1 源域  ：ΔF1_T1 ≥ −0.005（一次性回退容忍上限）
   │      触碰任一条 ⇒ 直接判负 / 剪枝 Pruned（不进二级）
   │
   ├─【二级】帕累托占优（相对**对手 B**）
   │      三轨主指标（F1↑）＋ 虚警指标（FP↓）全向不劣，且至少一维严格更优
   │      ⇒ 无争议判定 M ≻ B（**不再看效用分**）
   │
   └─【三级】效用裁决（仅当二级不成立，即指标互有攻守）
          U = ΔF1_T3 + α·ΔF1_T2 − β·max(0, −ΔF1_T1) − γ·max(0, ΔFP_T2)
          α=0.5（黄金池巩固） β=2.5（源域惩罚倍率） γ=0.0005（每新增 1 个黄金池 FP）
```

三级漏斗的**顺序不可交换**：帕累托占优成立时直接判胜（效用分只用于互有攻守的裁决），
所以「模型好坏」的正确问法是 :func:`compare_models`（两两比较器），
而不是"谁的效用分高"。效用分在搜索里扮演的是 **Optuna 的标量 objective**。

系数物理含义
------------
* `ΔF1_T3`（难例突破）权重 1.0：微调主攻目标，涨多少算多少；
* `α=0.5`：黄金池本身已在 0.988+ 高位饱和，微弱涨点只给适度加分；
* `β=2.5`：源域每掉 0.1pp，难例轨至少要涨 0.25pp 才拉平，防止盲目牺牲大盘；
* `γ=0.0005`：每新增 1 个黄金池 FP 扣 0.0005 分（= 黄金池每 1 个 FP 约合难例轨 0.05pp）。

参考实现（与用户给定口径逐条对齐）
----------------------------------
```python
d_t1  = cand['track1_f1'] - base['track1_f1']
d_t2  = cand['track2_f1'] - base['track2_f1']
d_fp2 = cand['track2_fp']  - base['track2_fp']
d_t3  = cand['track3_f1'] - base['track3_f1']

if cand['track2_f1'] < 0.985 or d_fp2 > 10:   -> -inf, "FAIL_TRACK2_SAFETY"
if d_t1 < -0.005:                            -> -inf, "FAIL_TRACK1_RETREAT"
score = d_t3 + 0.5*d_t2 - 2.5*max(0,-d_t1) - 0.0005*max(0,d_fp2)
```

用法
----
```python
from manu.training.mix_ratio_judgement import (
    FROZEN_BASELINE, RC1_EPOCH3, hard_gate, utility, compare_models, select_best,
)

ok, reason, gdetail = hard_gate(cand)                  # 一级（相对冻结基线）
score, reason, detail = evaluate_model_quality(cand, base)  # 标量 objective
verdict = compare_models(cand, rival)                   # 二/三级（两两比较器）
report  = select_best(candidates)                      # 决赛
```

CLI（比较任意两个 gate_report.json）：
```bash
python manu/training/mix_ratio_judgement.py \
  --candidate runs/xxx/trial_0007/gate/epoch3/gate_report.json \
  --baseline  runs/mixed_ft_route_b/gate/epoch3/gate_report.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------- 默认参数
DEFAULT_GATE_TRACK2_F1_MIN = 0.985
DEFAULT_GATE_TRACK2_FP_MAX_INCREMENT = 10
DEFAULT_GATE_TRACK1_MIN_DELTA = -0.005

DEFAULT_ALPHA = 0.5  # 黄金池巩固系数
DEFAULT_BETA = 2.5  # 源域惩罚倍率
DEFAULT_GAMMA = 0.0005  # 黄金池每个新增 FP 的惩罚

# 冻结基线（Baseline v2，去 P0 口径）—— 一级红线的**绝对参照**
FROZEN_BASELINE = {
    "track1_f1": 0.905736245785252,
    "track1_fp": 1014,
    "track2_f1": 0.9884359729262149,
    "track2_fp": 26,
    "track3_f1": 0.8373308251032032,
    "track3_fp": 590,
}

# RC1 = 路线 B Epoch 3（本轮微调的当前最优产物）—— 最终选型时的**现任擂主**
RC1_EPOCH3 = {
    "track1_f1": 0.901917808219178,
    "track1_fp": 941,
    "track2_f1": 0.9989791016094163,
    "track2_fp": 31,
    "track3_f1": 0.8716843671438099,
    "track3_fp": 461,
}

# RC1_EPOCH5 = 同一次 5-epoch 运行的**末 epoch** 权重 —— 5-epoch 口径的参照系
#
# 为什么需要它（2026-10-21 决策：后续 trial 改跑 5 epoch）
# -------------------------------------------------------
# 搜索 trial 改成"训练 5 epoch、门禁量第 5 个 epoch"后，trial 实际对应的是
# **RC1 同一次运行的 Ep5 权重**，而不是 Ep3 ⇒ 与 RC1/Ep3 的指标**不同尺度**。
# ⇒ 5-epoch 口径的 trial 必须与 **RC1_EPOCH5** 比，而不是 RC1_EPOCH3。
#
# 实测（RC1.json → full_cycle_curve 的 epoch=5）：
#   T1 F1 0.900501 / FP 1023 | T2 F1 0.998949 / FP 33 | T3 F1 0.874924 / FP 502
#   U(Ep5 vs 冻结基线) = +0.026260  <  U(Ep3) = +0.027579
# ⇒ **RC1 自己跑满 5 epoch 反而比 Ep3 低**（源域 F1 跌破 0.901、虚警抬头）
#   ⇒ 这正是"提高 LQ 比例能否在 5 epoch 下护住源域"这个搜索假设的用武之地。
# ⇒ 同时注意 RC1/Ep5 的 ΔF1_T1 = −0.005235 **已跌破严格红线 −0.005**
#   （只过受控放宽通道），说明 5-epoch 口径对源域更苛刻。
RC1_EPOCH5 = {
    "track1_f1": 0.9005006521098912,
    "track1_fp": 1023,
    "track2_f1": 0.9989491698441769,
    "track2_fp": 33,
    "track3_f1": 0.8749235474006115,
    "track3_fp": 502,
}

# 按"测量 epoch"索引的 RC1 参照系（供状态报告按口径选对擂主）
RC1_BY_EPOCH = {3: RC1_EPOCH3, 5: RC1_EPOCH5}


def reference_name(ref: dict) -> str:
    """给参照系起个准确的名字（Ep3 现任擂主 / Ep5 同次运行末 epoch / 其它）。"""
    if ref is RC1_EPOCH3:
        return "RC1_EPOCH3(现任擂主)"
    if ref is RC1_EPOCH5:
        return "RC1_EPOCH5(同次运行末epoch)"
    for ep, r in RC1_BY_EPOCH.items():
        if r == ref:
            return f"RC1_EPOCH{ep}"
    return "参照系"


# ==============================================================================
# RC1 的完整 6 点曲线（路线 B 5-epoch 运行的逐 epoch 门禁结果）
# 来源：服务器 `runs/mixed_ft_route_b/RC1/RC1.json` → `full_cycle_curve`
# ==============================================================================
RC1_EPOCH_CURVE = {
    0: {
        "track1_f1": 0.905736245785252, "track1_fp": 1014,
        "track2_f1": 0.9884359729262149, "track2_fp": 26,
        "track3_f1": 0.8373308251032032, "track3_fp": 590,
    },
    1: {
        "track1_f1": 0.9022149063405079, "track1_fp": 1232,
        "track2_f1": 0.998768953610569, "track2_fp": 35,
        "track3_f1": 0.8468954078988287, "track3_fp": 589,
    },
    2: {
        "track1_f1": 0.9019269606193201, "track1_fp": 988,
        "track2_f1": 0.9988890556973428, "track2_fp": 33,
        "track3_f1": 0.8595867470500335, "track3_fp": 531,
    },
    3: {
        "track1_f1": 0.901917808219178, "track1_fp": 941,
        "track2_f1": 0.9989791016094163, "track2_fp": 31,
        "track3_f1": 0.8716843671438099, "track3_fp": 461,
    },
    4: {
        "track1_f1": 0.9017962061440322, "track1_fp": 1057,
        "track2_f1": 0.9988591329410352, "track2_fp": 35,
        "track3_f1": 0.8726475238435252, "track3_fp": 517,
    },
    5: {
        "track1_f1": 0.9005006521098912, "track1_fp": 1023,
        "track2_f1": 0.9989491698441769, "track2_fp": 33,
        "track3_f1": 0.8749235474006115, "track3_fp": 502,
    },
}


def rc1_epoch_band(epochs: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    """RC1 **自身**在给定 epoch 集合上的逐指标包络（min/max）。

    这是判断"是否复现"时唯一诚实的参照带：
    **同一个 run 内相邻 epoch 就在大幅波动**
    （T3 F1 0.8469~0.8749 跨度 0.028；T1 FP 941~1232 跨度 291），
    拿单个点当"应该复现到的值"会把正常波动误读成"复现失败"。
    """
    keys = ("track1_f1", "track1_fp", "track2_f1", "track2_fp", "track3_f1", "track3_fp")
    band = {}
    for k in keys:
        vals = [RC1_EPOCH_CURVE[e][k] for e in epochs if e in RC1_EPOCH_CURVE]
        band[k] = (min(vals), max(vals)) if vals else (None, None)
    return band


# 位级复现的容差（**来自实测**：同配置同 seed 的两次 GPU 训练只剩非确定性差异，
# 实测 T3 F1 抖动 ~1e-3 量级、FP 计数抖动个位数；这里给一个保守但有区分度的门限）
EXACT_TOL = {
    "track1_f1": 2e-3, "track1_fp": 30,
    "track2_f1": 1e-3, "track2_fp": 4,
    "track3_f1": 3e-3, "track3_fp": 40,
}


def reproduction_check(cand: dict, epoch: int = 3) -> dict:
    """判断候选模型是否**复现**了 RC1 在 ``epoch`` 这个 epoch 上的精度。

    两级判定：
      · ``EXACT_REPRO`` —— 与 RC1[epoch] 的逐项差都在 `EXACT_TOL` 内
        （位级复现：数据流完全一致时应当成立，只剩 GPU 非确定性）
      · ``BAND_REPRO``  —— 不满足逐项容差，但落在 **RC1 自身相邻 epoch 的包络**内
        ⇒ 属正常随机波动，**不能据此说"复现失败"**
      · ``NOT_REPRO``   —— 连 RC1 自身的 epoch 间包络都出不去 ⇒ 才是真异常

    返回 ``{"verdict", "epoch", "diffs", "band", "outliers", "detail"}``。
    """
    ref = RC1_EPOCH_CURVE.get(epoch)
    if ref is None:
        raise KeyError(f"RC1 没有 epoch {epoch} 的曲线数据")
    band = rc1_epoch_band()
    diffs, outliers, band_out = {}, [], []
    for k, tol in EXACT_TOL.items():
        if k not in cand:
            continue
        d = cand[k] - ref[k]
        diffs[k] = d
        if abs(d) > tol:
            outliers.append(k)
        lo, hi = band[k]
        if lo is not None and not (lo <= cand[k] <= hi):
            band_out.append(k)
    if not outliers:
        verdict = "EXACT_REPRO"
    elif not band_out:
        verdict = "BAND_REPRO"
    else:
        verdict = "NOT_REPRO"
    detail = {
        "exact_outliers": outliers,
        "band_outliers": band_out,
        "band": {k: list(v) for k, v in band.items()},
        "ref": dict(ref),
        "cand": {k: cand[k] for k in EXACT_TOL if k in cand},
    }
    return {
        "verdict": verdict,
        "epoch": epoch,
        "diffs": diffs,
        "outliers": outliers,
        "band_outliers": band_out,
        "detail": detail,
    }

REQUIRED_KEYS = ("track1_f1", "track2_f1", "track2_fp", "track3_f1")

# 帕累托维度：F1 越大越好（up），虚警越小越好（down）
#
# ⚠️ 维度集直接决定"无争议占优"是否成立，**必须显式、可审计、可切换**。
# 用户例句：「上一轮 Epoch 3 对比 Epoch 4，Ep3 黄金池更高、源域大盘更高、001324 虚警更低，
#            在关键维度全面占优，无争议判定 Ep3 优于 Ep4」——该例句隐含的维度集是
#            {T1_F1↑, T2_F1↑, T3_FP↓}，**不含 T3_F1**。
# 实测：Ep3 的 T3-F1(0.871684) 实际**低于** Ep4(0.872648)，
#       因此维度集若含 T3_F1，Ep3≻Ep4 不成立，会落到三级效用分裁决，
#       而效用分给出 U(Ep3 vs Ep4) = **−0.000903**（即 Ep4 反而胜）。
#       ⇒ 这不是 bug，而是"漏斗顺序"必须被尊重的实证：
#          二级帕累托成立 ⇒ 直接判胜（不再看效用分）；不成立 ⇒ 才由效用分裁决。
PARETO_PRESETS = {
    # 三轨主指标（F1）
    "core": {"track1_f1": "up", "track2_f1": "up", "track3_f1": "up"},
    # 三轨主指标 + 三轨虚警（最保守：六维全占优才算无争议）
    "full": {
        "track1_f1": "up",
        "track1_fp": "down",
        "track2_f1": "up",
        "track2_fp": "down",
        "track3_f1": "up",
        "track3_fp": "down",
    },
    # 严格按用户例句的隐含维度集
    "user": {"track1_f1": "up", "track2_f1": "up", "track3_fp": "down"},
}
DEFAULT_PARETO_PRESET = "full"


def resolve_dims(dims: dict | str | None) -> dict:
    """把维度集参数解析成 {key: 'up'/'down'}；可以是预设名或显式字典。"""
    if dims is None:
        return dict(PARETO_PRESETS[DEFAULT_PARETO_PRESET])
    if isinstance(dims, str):
        if dims not in PARETO_PRESETS:
            raise ValueError(f"未知帕累托预设 {dims}，可选：{sorted(PARETO_PRESETS)}")
        return dict(PARETO_PRESETS[dims])
    return dict(dims)


def _check_metrics(metrics: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in metrics]
    if missing:
        raise ValueError(f"metrics 缺少必需字段：{missing}")


# ---------------------------------------------------------------- 一级：硬红线
# 受控放宽通道的阈值（**由实测数据分布校准**，2026-10-21，经用户授权"根据实测数据分布做调整"）
#
# 实测依据（第一批 4 个种子 trial + RC1 的 5-epoch 完整曲线）：
#   · T2 黄金池 FP 在混训后的**正常水平**落在 [32, 37]（基线 26 ⇒ ΔFP ∈ [+6, +11]），
#     而 **T2 FP 的单 trial 噪声 ≈ ±5** ⇒ 严格线 `ΔFP ≤ +10` **正切在噪声带中部**，
#     实测 4 个 trial 里 **3 个只差 1 个虚警就被判负**（+11 vs +10）—— 红线在误杀好配置。
#   · 同一批 T2 F1 **全部 ≥ 0.9987**（基线 0.988436，Δ ≥ +0.0103）：FN 从 355 掉到 6，
#     FP 只从 26 涨到 32~37（16,638 个 GT 里误报率 0.156% → 0.222%）。
#     ⇒ "FP 微量上升"换来了"F1 大幅跃升"，用 F1 增益为 FP 超限买单有明确物理依据。
#   · T1 源域在混训下的**正常代价**是 ΔF1 ∈ [−0.005417, −0.003281]（RC1 自身 = −0.003818），
#     严格线 −0.005 卡在分布边缘 ⇒ 实测 trial_0001 以 0.0004 之差被误判。
#   · T3 难例在同一批里 ΔF1 ∈ [+0.023727, +0.028133]，确有实质提升。
#
# ⇒ 放宽通道只在**"有实质收益补偿"**时开放，且各项都比严格线更严：
DEFAULT_RELAXED_GATE = {
    "track2_fp_max_increment": 15,   # 实测上沿 Δ+11 + 裕量；比严格线 +10 宽，但仍封顶
    "track2_f1_min_gain": 0.008,     # 黄金池 F1 必须有实质提升（实测 +0.0103）才允许放宽
    "track3_f1_min_gain": 0.020,     # 难例必须同步大涨（实测 +0.0237~+0.0281）才允许放宽
    "track1_min_delta": -0.008,      # 源域退让容差（覆盖实测最差 −0.005417）
}
DEFAULT_RELAXED_GATE_ENABLED = True


def hard_gate(
    metrics: dict,
    baseline: dict | None = None,
    *,
    track2_f1_min: float = DEFAULT_GATE_TRACK2_F1_MIN,
    track2_fp_max_increment: int = DEFAULT_GATE_TRACK2_FP_MAX_INCREMENT,
    track1_min_delta: float = DEFAULT_GATE_TRACK1_MIN_DELTA,
    relaxed: dict | None = None,
    allow_relaxed: bool = DEFAULT_RELAXED_GATE_ENABLED,
) -> tuple[bool, str, dict]:
    """一级漏斗：安全红线一票否决。返回 (ok, reason, detail)。

    红线（相对 `baseline`，默认**冻结基线**）——**双通道**：

    **严格通道**（用户原始口径）：
      · Track2 黄金池 F1 ≥ 0.985（绝对值）
      · Track2 黄金池 ΔFP ≤ +10
      · Track1 源域 ΔF1 ≥ −0.005
      ⇒ reason = ``"PASS"``

    **受控放宽通道**（实测分布校准，见 :data:`DEFAULT_RELAXED_GATE`）：
      仅当严格通道不过、但**黄金池与难例都有实质增益**来买单时开放：
      · Track2 黄金池 F1 ≥ 0.985
      · Track2 ΔFP ≤ +15 **且** ΔF1_T2 ≥ +0.008
      · Track3 ΔF1 ≥ +0.020
      · Track1 ΔF1 ≥ −0.008
      ⇒ reason = ``"PASS_RELAXED"``（**显式标注，不静默放行**）

    两通道皆不过 ⇒ ``False`` + ``FAIL_TRACK2_SAFETY`` / ``FAIL_TRACK1_RETREAT``。

    ``allow_relaxed=False`` 可退回纯严格口径（用于复现旧结论）。
    """
    base = baseline or FROZEN_BASELINE
    _check_metrics(metrics)
    _check_metrics(base)
    rel = dict(DEFAULT_RELAXED_GATE if relaxed is None else relaxed)

    d_t1 = metrics["track1_f1"] - base["track1_f1"]
    d_t2 = metrics["track2_f1"] - base["track2_f1"]
    d_fp2 = metrics["track2_fp"] - base["track2_fp"]
    d_t3 = metrics["track3_f1"] - base["track3_f1"]

    detail = {
        "d_track1_f1": d_t1,
        "d_track2_f1": d_t2,
        "d_track2_fp": d_fp2,
        "d_track3_f1": d_t3,
        "thresholds": {
            "track2_f1_min": track2_f1_min,
            "track2_fp_max_increment": track2_fp_max_increment,
            "track1_min_delta": track1_min_delta,
        },
        "relaxed_thresholds": dict(rel) if allow_relaxed else None,
    }

    # ---- 严格通道 ----
    strict_ok = (
        metrics["track2_f1"] >= track2_f1_min
        and d_fp2 <= track2_fp_max_increment
        and d_t1 >= track1_min_delta
    )
    detail["strict_pass"] = strict_ok

    # ---- 受控放宽通道 ----
    if allow_relaxed:
        relaxed_ok = (
            metrics["track2_f1"] >= track2_f1_min
            and d_fp2 <= rel["track2_fp_max_increment"]
            and d_t2 >= rel["track2_f1_min_gain"]
            and d_t3 >= rel["track3_f1_min_gain"]
            and d_t1 >= rel["track1_min_delta"]
        )
        detail["relaxed_pass"] = relaxed_ok
    else:
        relaxed_ok = False
        detail["relaxed_pass"] = False

    if strict_ok:
        return True, "PASS", detail
    if relaxed_ok:
        detail["relaxed_note"] = (
            f"严格通道不过但受控放宽通道成立：黄金池 ΔF1 {d_t2:+.6f} ≥ "
            f"{rel['track2_f1_min_gain']}、难例 ΔF1 {d_t3:+.6f} ≥ {rel['track3_f1_min_gain']}、"
            f"ΔFP_T2 {d_fp2:+d} ≤ {rel['track2_fp_max_increment']}、ΔF1_T1 {d_t1:+.6f} ≥ "
            f"{rel['track1_min_delta']}（实测分布校准，见 DEFAULT_RELAXED_GATE）"
        )
        return True, "PASS_RELAXED", detail

    # ---- 两通道皆不过 ----
    if metrics["track2_f1"] < track2_f1_min:
        detail["fail_track"] = "track2_f1"
        return False, "FAIL_TRACK2_SAFETY", detail
    if d_fp2 > track2_fp_max_increment and not (
        allow_relaxed and d_fp2 <= rel["track2_fp_max_increment"] and d_t2 >= rel["track2_f1_min_gain"]
    ):
        detail["fail_track"] = "track2_fp"
        return False, "FAIL_TRACK2_SAFETY", detail
    detail["fail_track"] = "track1_f1"
    return False, "FAIL_TRACK1_RETREAT", detail


# ---------------------------------------------------------------- 三级：效用分
def utility(
    metrics: dict,
    baseline: dict,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> tuple[float, dict]:
    """U = ΔF1_T3 + α·ΔF1_T2 − β·max(0,−ΔF1_T1) − γ·max(0,ΔFP_T2)。"""
    _check_metrics(metrics)
    _check_metrics(baseline)
    d_t1 = metrics["track1_f1"] - baseline["track1_f1"]
    d_t2 = metrics["track2_f1"] - baseline["track2_f1"]
    d_fp2 = metrics["track2_fp"] - baseline["track2_fp"]
    d_t3 = metrics["track3_f1"] - baseline["track3_f1"]
    score = (
        d_t3
        + alpha * d_t2
        - beta * max(0.0, -d_t1)
        - gamma * max(0.0, d_fp2)
    )
    detail = {
        "d_track1_f1": d_t1,
        "d_track2_f1": d_t2,
        "d_track2_fp": d_fp2,
        "d_track3_f1": d_t3,
        "terms": {
            "t3_gain": d_t3,
            "t2_gain": alpha * d_t2,
            "t1_penalty": -beta * max(0.0, -d_t1),
            "fp_penalty": -gamma * max(0.0, d_fp2),
        },
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }
    return score, detail


def evaluate_model_quality(
    metrics_candidate: dict,
    metrics_baseline: dict,
    *,
    track2_f1_min: float = DEFAULT_GATE_TRACK2_F1_MIN,
    track2_fp_max_increment: int = DEFAULT_GATE_TRACK2_FP_MAX_INCREMENT,
    track1_min_delta: float = DEFAULT_GATE_TRACK1_MIN_DELTA,
    relaxed: dict | None = None,
    allow_relaxed: bool = DEFAULT_RELAXED_GATE_ENABLED,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> tuple[float, str, dict]:
    """一级门禁 + 效用分（**标量 objective**）。

    * 一级失败 → 返回 ``-inf`` 与 ``FAIL_*`` 原因，调用方应直接剪枝（Pruned）。
    * 一级通过 → 返回效用分与 ``"PASS"`` / ``"PASS_RELAXED"``。

    ⚠️ **与 :func:`hard_gate` 共用同一套双通道判定**（严格 + 实测分布校准的受控放宽），
    保证「搜索 objective 的剪枝门槛」与「最终选型的红线」**同尺度**。
    """
    ok, reason, detail = hard_gate(
        metrics_candidate,
        metrics_baseline,
        track2_f1_min=track2_f1_min,
        track2_fp_max_increment=track2_fp_max_increment,
        track1_min_delta=track1_min_delta,
        relaxed=relaxed,
        allow_relaxed=allow_relaxed,
    )
    if not ok:
        return float("-inf"), reason, detail
    score, udetail = utility(
        metrics_candidate, metrics_baseline, alpha=alpha, beta=beta, gamma=gamma
    )
    detail.update(udetail)
    detail["score"] = score
    return score, reason, detail


# ---------------------------------------------------------------- 二级：帕累托
def pareto_detail(a: dict, b: dict, dims: dict | str | None = None) -> dict:
    dims = resolve_dims(dims)
    per = {}
    for key, direction in dims.items():
        if key not in a or key not in b:
            continue
        av, bv = a[key], b[key]
        if direction == "up":
            ge, gt = av >= bv, av > bv
        else:
            ge, gt = av <= bv, av < bv
        per[key] = {"a": av, "b": bv, "direction": direction, "a_ge_b": ge, "a_gt_b": gt}
    dominates = bool(per) and all(v["a_ge_b"] for v in per.values()) and any(v["a_gt_b"] for v in per.values())
    return {"dimensions": per, "dominates": dominates}


def pareto_dominates(a: dict, b: dict, dims: dict | str | None = None) -> bool:
    """帕累托占优：a 在全部维度上不劣于 b，且至少一维严格更优。"""
    _check_metrics(a)
    _check_metrics(b)
    return pareto_detail(a, b, dims)["dominates"]


# ---------------------------------------------------------------- 两两比较器（决赛用）
def compare_models(
    a: dict,
    b: dict,
    *,
    frozen_baseline: dict | None = None,
    dims: dict | str | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    allow_relaxed: bool = DEFAULT_RELAXED_GATE_ENABLED,
    gate_config: dict | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """完整三级漏斗的两两比较（**严格按漏斗顺序**）。

    返回 dict：
        ``verdict`` ∈ {A_DOMINATES, B_DOMINATES, UTILITY_A, UTILITY_B, TIE, GATE_REJECT}
        ``gate_a`` / ``gate_b``：各自对冻结基线的一级门禁结果
        ``pareto``：二级明细
        ``utility_a`` / ``utility_b``：三级效用分（相对**冻结基线**，与 objective 同尺度）
        ``utility_a_vs_b`` / ``utility_b_vs_a``：互比效用分（仅在互有攻守时用于裁决）

    ``gate_config``（**纯增量**，默认 ``None`` ⇒ 逐位等价旧行为）：一级红线阈值覆盖，
    键名与 ``hard_gate`` 同名。参照系注入时必须透传，否则门禁会用**模块常量**旧阈值。
    """
    frozen = frozen_baseline or FROZEN_BASELINE
    gate_kw: dict = {}
    for _k in ("track2_f1_min", "track2_fp_max_increment", "track1_min_delta"):
        if gate_config and gate_config.get(_k) is not None:
            gate_kw[_k] = gate_config[_k]
    gate_a = hard_gate(a, frozen, allow_relaxed=allow_relaxed, **gate_kw)
    gate_b = hard_gate(b, frozen, allow_relaxed=allow_relaxed, **gate_kw)
    pdet = pareto_detail(a, b, dims)
    u_a, _ = utility(a, frozen, alpha=alpha, beta=beta, gamma=gamma)
    u_b, _ = utility(b, frozen, alpha=alpha, beta=beta, gamma=gamma)
    u_ab, _ = utility(a, b, alpha=alpha, beta=beta, gamma=gamma)
    u_ba, _ = utility(b, a, alpha=alpha, beta=beta, gamma=gamma)

    if not gate_a[0] and not gate_b[0]:
        verdict = "GATE_REJECT"
    elif not gate_a[0]:
        verdict = "B_DOMINATES"
    elif not gate_b[0]:
        verdict = "A_DOMINATES"
    elif pdet["dominates"]:
        verdict = "A_DOMINATES"
    elif pareto_dominates(b, a, dims):
        verdict = "B_DOMINATES"
    elif u_a > u_b:
        # ⚠️ 必须比 **同一基准（冻结基线）下的 U**，不能比 "U(A vs B)"。
        # 原因：U 里的 max(0,·) 惩罚项使其**非反对称** —— U(B vs A) ≠ −U(A vs B)，
        #      逐对比较会给出与 objective 不一致的裁决。
        #      实测：U(Ep3 vs Ep4) = −0.000903 而 U(Ep4 vs Ep3) = −0.001097，两者同号，
        #      若按逐对比较会误判；而 U(Ep3)=0.027581 > U(Ep4)=0.026178（同基准）才是正确结论。
        verdict = "UTILITY_A"
    elif u_b > u_a:
        verdict = "UTILITY_B"
    else:
        verdict = "TIE"

    return {
        "verdict": verdict,
        "label_a": label_a,
        "label_b": label_b,
        "gate_a": {"ok": gate_a[0], "reason": gate_a[1], "detail": gate_a[2]},
        "gate_b": {"ok": gate_b[0], "reason": gate_b[1], "detail": gate_b[2]},
        "pareto": pdet,
        "pareto_dims": resolve_dims(dims),
        "utility_a": u_a,
        "utility_b": u_b,
        "utility_a_vs_b": u_ab,
        "utility_b_vs_a": u_ba,
    }


# ---------------------------------------------------------------- 候选集决赛
def select_best(
    candidates: list[dict],
    *,
    frozen_baseline: dict | None = None,
    reference: dict | None = None,
    dims: dict | str | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    allow_relaxed: bool = DEFAULT_RELAXED_GATE_ENABLED,
    gate_config: dict | None = None,
) -> dict:
    """在一组候选里选最优（漏斗顺序），并给出可审计的判决链。

    ``candidates`` 每项须含 ``metrics``（四元组 + FP）与可选 ``name`` / ``weights`` / ``trial``。
    ``reference`` 为**现任擂主**（默认 RC1/Ep3），会一并参与判决。

    ``gate_config``（**纯增量**，默认 ``None`` ⇒ 行为与旧版逐位相同）：
    一级红线的阈值覆盖，键名与 ``hard_gate`` 同名（``track2_f1_min`` /
    ``track2_fp_max_increment`` / ``track1_min_delta``）。
    参照系注入场景（no-GMC 语义）必须把它透传给 ``hard_gate``，
    否则漏斗会用**模块常量**的旧阈值判，与参照系包里的最终阈值不一致。
    """
    frozen = frozen_baseline or FROZEN_BASELINE
    items = [dict(c) for c in candidates]
    if reference is not None:
        items.append({"name": reference_name(reference), "metrics": dict(reference), "is_reference": True})
    if not items:
        return {"winner": None, "reason": "没有候选"}

    # ★ 阈值透传（纯增量：不传 ⇒ 与 hard_gate 的默认值逐位相同）
    gate_kw: dict = {}
    for _k in ("track2_f1_min", "track2_fp_max_increment", "track1_min_delta"):
        if gate_config and gate_config.get(_k) is not None:
            gate_kw[_k] = gate_config[_k]

    # 一级：先剔除触红线的
    passed, rejected = [], []
    for it in items:
        ok, reason, det = hard_gate(it["metrics"], frozen, allow_relaxed=allow_relaxed, **gate_kw)
        it["gate_ok"], it["gate_reason"], it["gate_detail"] = ok, reason, det
        (passed if ok else rejected).append(it)

    report: dict = {
        "n_candidates": len(items),
        "n_passed_gate": len(passed),
        "n_rejected_gate": len(rejected),
        "rejected": [
            {"name": it.get("name"), "reason": it["gate_reason"], "track": it["gate_detail"].get("fail_track")}
            for it in rejected
        ],
        "frozen_baseline": frozen,
    }
    if not passed:
        report["winner"] = None
        report["reason"] = "全部候选都未过一级红线"
        # ★ 兜底：早退路径也要给出帕累托计数，否则调用方按固定键取值会 KeyError
        report["n_pareto_maximal"] = 0
        return report

    # 三级效用分（相对冻结基线，与 objective 同尺度）
    for it in passed:
        it["utility"], _ = utility(it["metrics"], frozen, alpha=alpha, beta=beta, gamma=gamma)

    # 二级：找帕累托极大集
    maximal = []
    for it in passed:
        dominated = False
        for other in passed:
            if other is it:
                continue
            if pareto_dominates(other["metrics"], it["metrics"], dims):
                dominated = True
                break
        if not dominated:
            maximal.append(it)
    report["pareto_maximal"] = [
        {"name": it.get("name"), "trial": it.get("trial"), "utility": it["utility"]} for it in maximal
    ]
    report["n_pareto_maximal"] = len(maximal)

    if len(maximal) == 1:
        report["winner"] = maximal[0]
        report["winner_reason"] = "唯一帕累托极大元 ⇒ 二级无争议占优"
    else:
        best = max(maximal, key=lambda it: it["utility"])
        report["winner"] = best
        report["winner_reason"] = (
            f"存在 {len(maximal)} 个互不占优的帕累托极大元 ⇒ 三级按效用分裁决"
        )
        report["tied_against"] = [
            {"name": it.get("name"), "trial": it.get("trial"), "utility": it["utility"]}
            for it in maximal
            if it is not best
        ]

    # 与现任擂主的正面对决
    if reference is not None:
        cmp = compare_models(
            report["winner"]["metrics"],
            reference,
            frozen_baseline=frozen,
            dims=dims,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            gate_config=gate_config,
            label_a=report["winner"].get("name", "winner"),
            label_b="RC1_EPOCH3",
        )
        report["vs_reference"] = cmp
        report["beats_reference"] = cmp["verdict"] in ("A_DOMINATES", "UTILITY_A")
    return report


# ---------------------------------------------------------------- gate_report 适配
class GateReportIncomplete(ValueError):
    """`gate_report.json` 缺少必需字段。

    几乎总是**门禁基础设施失败**（最常见：Track1 推理 CUDA OOM ⇒
    `tracks.track1_uav.finetune_peak.f1` 为 null），**不是**模型指标不达标。

    必须显式抛这个异常而不是让 `float(None)` 抛 `TypeError`：
    TypeError 的消息（"float() argument must be a string or a real number"）
    完全指不出是哪个 track 缺了，排查一次要花十几分钟。
    """

    def __init__(self, missing: list[str], path: str | Path | None = None):
        self.missing = list(missing)
        self.path = str(path) if path is not None else None
        where = f"（{self.path}）" if self.path else ""
        super().__init__(
            f"gate_report 缺少必需字段 {self.missing}{where} —— "
            f"通常是门禁基础设施失败（如 Track1 推理 CUDA OOM），而非指标不达标"
        )


def metrics_from_gate_report(path: str | Path) -> dict:
    """从 `auto_epoch_gate_check.py` 产出的 gate_report.json 抽出指标（四元组 + FP）。

    缺字段时抛 :class:`GateReportIncomplete`（带缺失字段名与路径），**不做静默降级**。
    """
    rep = json.loads(Path(path).read_text(encoding="utf-8"))
    tracks = rep.get("tracks", {})
    t1 = (tracks.get("track1_uav") or {}).get("finetune_peak") or {}
    t2 = tracks.get("track2_golden") or {}
    t3 = tracks.get("track3_blind") or {}

    def _num(v, cast):
        if v is None:
            return None
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    out = {
        "track1_f1": _num(t1.get("f1"), float),
        "track1_fp": _num(t1.get("fp"), int),
        "track2_f1": _num(t2.get("f1"), float),
        "track2_fp": _num(t2.get("fp"), int),
        "track3_f1": _num(t3.get("f1"), float),
        "track3_fp": _num(t3.get("fp"), int),
    }
    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise GateReportIncomplete(missing, path)
    out["detail"] = {
        "track1_tp": t1.get("tp"),
        "track1_fn": t1.get("fn"),
        "track1_th": t1.get("th"),
        "track2_tp": t2.get("tp"),
        "track2_fn": t2.get("fn"),
        "track3_tp": t3.get("tp"),
        "track3_fn": t3.get("fn"),
        "verdict": rep.get("verdict"),
        "veto_failures": rep.get("veto_failures"),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="三级漏斗判定：比较候选与对手的三轨表现")
    ap.add_argument("--candidate", required=True, help="候选 gate_report.json（或 'baseline' / 'rc1'）")
    ap.add_argument("--baseline", required=True, help="对手 gate_report.json（或 'baseline' / 'rc1'）")
    ap.add_argument("--track2-f1-min", type=float, default=DEFAULT_GATE_TRACK2_F1_MIN)
    ap.add_argument("--track2-fp-max-increment", type=int, default=DEFAULT_GATE_TRACK2_FP_MAX_INCREMENT)
    ap.add_argument("--track1-min-delta", type=float, default=DEFAULT_GATE_TRACK1_MIN_DELTA)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument(
        "--strict-only", action="store_true",
        help="只用用户原始严格红线（禁用实测分布校准的受控放宽通道），用于复现旧结论",
    )
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    def load(spec: str) -> dict:
        key = spec.lower()
        if key in ("baseline", "frozen"):
            return dict(FROZEN_BASELINE)
        if key in ("rc1", "rc1_epoch3"):
            return dict(RC1_EPOCH3)
        return metrics_from_gate_report(spec)

    cand, base = load(args.candidate), load(args.baseline)
    res = compare_models(
        cand,
        base,
        frozen_baseline=FROZEN_BASELINE,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        allow_relaxed=not args.strict_only,
        label_a="候选",
        label_b="对手",
    )

    print("=" * 88)
    print("三级漏斗判定：候选 vs 对手")
    print("=" * 88)
    print("  一级 硬红线（相对冻结基线）")
    for tag, g in (("候选", res["gate_a"]), ("对手", res["gate_b"])):
        if g["ok"]:
            mark = "✅ PASS" if g["reason"] == "PASS" else "✅ PASS_RELAXED（受控放宽通道）"
            print(f"    {tag}: {mark}")
            if g["reason"] == "PASS_RELAXED":
                print(f"        └ {g['detail'].get('relaxed_note', '')}")
        else:
            print(f"    {tag}: ❌ {g['reason']}（轨道 {g['detail'].get('fail_track')}）")
    if not args.strict_only:
        print("    （严格通道 + 实测分布校准的受控放宽通道；加 --strict-only 可只看严格口径）")
    print("  二级 帕累托占优（维度含 F1↑ 与 虚警↓）")
    for k, v in res["pareto"]["dimensions"].items():
        print(f"    {k:12s} A={v['a']!r:22s} B={v['b']!r:22s} A≥B:{v['a_ge_b']} A≠B:{v['a_gt_b']}")
    print(f"    ⇒ 占优判定：{'A 严格占优' if res['pareto']['dominates'] else '互有攻守（进入三级）'}")
    print("  三级 效用分（相对冻结基线，同尺度）")
    print(f"    U(A) = {res['utility_a']:+.6f}    U(B) = {res['utility_b']:+.6f}")
    print(f"    互比 U(A vs B) = {res['utility_a_vs_b']:+.6f}    U(B vs A) = {res['utility_b_vs_a']:+.6f}")
    print(f"\n  ⚖️  判决：{res['verdict']}")
    print("=" * 88)

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
