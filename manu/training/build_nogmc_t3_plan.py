#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 **no-GMC T3 前沿** 新 study 的确定性臂计划（2026-11-03）。

为什么是新 study
================
Optuna 4.9 的分布兼容检查是 **study 级**：
  · `CategoricalDistribution` 换 choices ⇒ 抛
    "CategoricalDistribution does not support dynamic value space"
    ⇒ **改 `epochs` 的取值集合必须换 study**；
  · `FloatDistribution` 的 low/high **不参与**兼容检查（只查类型与 log 标志），
    所以 `lr_scale` 区间本身可扩 —— 但既然 `epochs` 已被 categorical 锁死，
    统一开新 study 更干净（两套 study 的语义也不会互相污染）。

本轮扩的三处空间
================
| 维度 | 旧（focus study） | 新（t3 study） | 依据 |
|---|---|---|---|
| `lr_scale` | [0.8, 3.0] | **[0.8, 4.0]** | 20/20 受控对里 lr↑ ⇒ T3↑ **单调**；最优 T3 全挤在 2.75~3.0 的天花板 |
| `epochs`  | {6,8,10,15} | **{6,8,10,15,20,25}** | 4/4 受控对里 ep↑ ⇒ T3↑ +0.77~+2.58 pp，而 T1 只掉 0.03~0.38 pp；ep=15 也是天花板 |
| `mix_spec`| {w2.0,w3.0,w6.0,w8.0,natural} | **{natural,w2.0,w2.5,w3.0,w6.0}** | mix 比↑ ⇒ T3↓（13/14 受控对）；砍掉 w8.0，补进未测过的 w2.5 |

实测依据（全部来自 focus + weekend 两个 study 的 119 条 COMPLETE 记录）
=====================================================================
1. **lr_scale 是唯一「20/20 全对都涨 T3」的旋钮**，且当前上限 3.0 正是最优 T3 的位置：
   w2.0·lr2.5→2.75→3.0 的 T3 = 0.873693 / 0.874692 / 0.880004·0.881018（两个种子）。
   近端汇率 ≈ **+0.023 T3 / 1.0 lr**（2.75→3.0 段：+0.0058 / 0.25 lr）。
2. **epochs 是被忽略的 T3 杠杆**：同配方只改 epochs 的受控对里
   ep8→15 给出 ΔT3 = **+2.40 / +2.58 pp**（lr1.0·w3.0）、
   ep8→10 给出 **+1.43 pp**（lr2.882·w6.0）、ep10→15 给出 **+1.36 / +0.77 pp**；
   同期 ΔT1 只有 −0.21 / −0.12 / −0.38 / +0.03 pp ⇒ **性价比优于继续加 lr**。
   ⚠️ 但这些受控对的 lr 都是 1.0~2.882、mix 都是 w3.0/w6.0 ⇒
   **「w2.0/w3.0 + lr2.88~3.0 + 长训练」这一角完全没测过** ⇒ 本计划主打这一角。
3. **mix 比是纯 T3 供体**：比值↑ ⇒ T3↓（w2.0→w3.0：ΔT3 −0.74/−0.84 pp，ΔT1 +0.17/+0.23 pp）。
4. **init_from=rc2 比 base474 高 +3.88 pp T3**（仅 −0.10 pp T1）⇒ 主力用 rc2。
5. **跨种子 soup 只给 +0.01~0.07 pp T1 / +0.11 pp T3**（实测 3 族）⇒ 它是「保险」不是「主力」；
   但它是**唯一零训练成本**的 T1 手段 ⇒ 值得为最强家族补种子。
6. **T1–T3 强负相关（严格通道 Pearson −0.721）**，前沿斜率 ≈ **每丢 1 pp T1 换 7 pp T3**
   ⇒ 「T3 达 0.89 且 T1 过 0.895897 线」在结构上极难同取 ⇒ 本轮必须**两条线并跑**：
   （a）T3 冲线线（w2.0/长训练/高 lr），（b）严格 PASS 保险线（补种子 → 跨种子 soup）。

臂的编排规则
============
* **入队顺序 = 执行顺序**（Optuna `ask()` 优先弹 WAITING 队列，按 trial number 升序）
  ⇒ 计划里**按价值从高到低**排，价值最高的最先跑。
* 但**长臂必须排在够早的位置**：硬截止 `hard_cut` 会把「预计跑不完」的臂标 PRUNED
  ⇒ 长臂若排在后面，会占着 GPU 槽位却跑不完 ⇒ 所以 15/20/25 ep 的臂集中在前段，
  8/10 ep 的臂放后段（它们可以在收尾窗口里从容收尾）。
* 每条臂都**只改前沿线上的维度**（mix / lr / epochs / init），
  其余形状超参沿用 focus study 的同一组值（clip 5.65 / focal 2.75 / offset 0.4 /
  mosaic 0.15 / scale 0.1 / translate 0.05 / warmup 1.5 / wd 2.6906e-4 /
  bblr 1.0 / fbb 0 / freeze_bn False）⇒ **受控对比**，任何差异都只可能来自被改的那一维。

用法
====
```bash
python manu/training/build_nogmc_t3_plan.py \
  --out manu/training/nogmc_t3_plan.json \
  --start "2026-11-03 11:00" --gpus 4
```
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# 形状超参：沿用 focus study 全部臂的同一组值（受控对比的前提）
# ---------------------------------------------------------------------------
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

# 参照配方（focus study 实测，用来写 note / 预测）
REF = {
    "w2.0_lr3.0_8ep": dict(mix="w2.0", lr=3.0, ep=8, t1=0.893555, t3=0.880511),
    "w3.0_lr3.0_8ep": dict(mix="w3.0", lr=3.0, ep=8, t1=0.895565, t3=0.872645),
    "w3.0_lr2.8821_8ep": dict(mix="w3.0", lr=2.8821, ep=8, t1=0.895677, t3=0.867272),
    "w2.0_lr2.75_8ep": dict(mix="w2.0", lr=2.75, ep=8, t1=0.894888, t3=0.874692),
}

# 每 epoch 训练耗时（分钟，实测中位数）+ 末尾门禁 ~19.5 min
PER_EPOCH_MIN = 9.15
GATE_MIN = 19.5


def arm_minutes(epochs: int) -> float:
    return round(PER_EPOCH_MIN * epochs + GATE_MIN, 1)


# ---------------------------------------------------------------------------
# 臂清单：role / mix / lr / epochs / init / seed / 价值档 / 说明
# ---------------------------------------------------------------------------
# init_x: 可选，指定则走 `init_from_x`（新参数名）承载 soup 初始化
ARMS: list[dict] = [
    # ===== A 段：T3 上攻（用户主目标：T3 → 0.89）=======================
    dict(role="T3_W2_LR30_15EP", mix="w2.0", lr=3.0, ep=15, seed=300, tier="A",
         note="★主攻：当前 T3 最高配方（w2.0·lr3.0·8ep，T3 0.8805/2 种子）"
              "只加 epochs 15（ep8→15 受控对 ΔT3 +1.4~+2.6 pp）⇒ 预测 T3 0.895~0.906、T1 0.890~0.893"),
    dict(role="T3_W3_LR30_15EP", mix="w3.0", lr=3.0, ep=15, seed=301, tier="A",
         note="★主攻：trial_0033 配方（w3.0·lr3.0·8ep，T1 0.895565 只差 0.0332 pp）加 epochs 15"
              "⇒ 预测 T3 0.887~0.899、T1 0.892~0.895 —— 最有可能同取 T3 过 0.89 与 T1 过线的一条"),
    dict(role="T3_W2_LR35_15EP", mix="w2.0", lr=3.5, ep=15, seed=302, tier="A",
         note="★T3 上限探测：lr 首次越过 3.0 天花板（近端汇率 +0.023 T3/1.0 lr）"
              "⇒ 预测 T3 0.906~0.917、T1 0.888~0.891"),
    dict(role="T3_W3_LR2882_15EP", mix="w3.0", lr=2.8821, ep=15, seed=303, tier="A",
         note="★严格 PASS 家族（cs_t0008：4 种子 soup 已 T1 0.8959005 严格过线）加 epochs 15"
              "⇒ 预测 T3 0.882~0.893、T1 0.893~0.895 ⇒ 若 T3 也能过 0.87 就是双目标"),
    dict(role="T3_W2_LR30_20EP", mix="w2.0", lr=3.0, ep=20, seed=304, tier="A",
         note="★T3 上攻：epochs 再往上推一档（ep25 的前置读数）⇒ 预测 T3 0.900~0.910"),
    dict(role="T3_W2_LR30_15EP_SOUPSRC", mix="w2.0", lr=3.0, ep=15, seed=305, tier="A",
         init_x="soup_strict3",
         note="初始化换 soup_strict3（自身 T1 0.896846，比 rc2 起点更高）"
              "⇒ 同样 15 ep 下 T1 可能比 rc2 版高 0.1~0.25 pp"),
    dict(role="T3_W3_LR30_15EP_SOUPSRC", mix="w3.0", lr=3.0, ep=15, seed=306, tier="A",
         init_x="soup_strict3", note="同上，但走 w3.0（T1 更好的一侧）"),
    dict(role="T3_W3_LR35_15EP", mix="w3.0", lr=3.5, ep=15, seed=307, tier="A",
         note="T3 上限探测（w3.0 侧）：预测 T3 0.897~0.909"),
    # ===== B 段：严格 PASS 保险 + 跨种子 soup 家族 ======================
    dict(role="INS_W2_LR30_15EP_S2", mix="w2.0", lr=3.0, ep=15, seed=308, tier="B",
         note="★T3_W2_LR30_15EP 的同配方第二种子 ⇒ 可做 2 种子 soup（零成本 T1 增益）"),
    dict(role="INS_W3_LR30_15EP_S2", mix="w3.0", lr=3.0, ep=15, seed=309, tier="B",
         note="★T3_W3_LR30_15EP 的同配方第二种子 ⇒ 2 种子 soup"),
    dict(role="T3_W2_LR325_15EP", mix="w2.0", lr=3.25, ep=15, seed=310, tier="B",
         note="lr 阶梯补点（3.0 / 3.25 / 3.5）⇒ 给出 ep15 段落的 lr→T3 汇率曲线"),
    dict(role="T3_W3_LR325_15EP", mix="w3.0", lr=3.25, ep=15, seed=311, tier="B",
         note="lr 阶梯补点（w3.0 侧）"),
    dict(role="T3_W2_LR30_25EP", mix="w2.0", lr=3.0, ep=25, seed=312, tier="B",
         note="★epochs 上限探测（25 ep，cosine 周期同步拉长）⇒ 若 ep 杠杆不衰减，T3 可到 0.90+"),
    dict(role="T3_W3_LR30_25EP", mix="w3.0", lr=3.0, ep=25, seed=313, tier="B",
         note="★epochs 上限探测（w3.0 侧，T1 更好）"),
    dict(role="T3_W3_LR2882_20EP", mix="w3.0", lr=2.8821, ep=20, seed=314, tier="B",
         note="严格 PASS 家族的长训练档（cs_t0008 同族）⇒ 预测 T3 0.885~0.896、T1 0.892~0.895"),
    dict(role="T3_W3_LR30_20EP", mix="w3.0", lr=3.0, ep=20, seed=326, tier="D",
         note="★T3 冲 0.89 的最好一条：w3.0 侧 T1 比 w2.0 好 ~0.2 pp，"
              "ep20 再叠 ep8→15 的 +1.4~+2.6 pp ⇒ 预测 T3 0.890~0.902、T1 0.891~0.894"),
    dict(role="INS_T0033_S4", mix="w3.0", lr=3.0, ep=8, seed=315, tier="B",
         note="★trial_0033 家族（唯一「T1 只差 0.0332 pp = 76 个 TP」的那条）补种子"
              "⇒ 凑 4 种子跨种子 soup（cs_t0033），实测 soup 有 +0.01~0.07 pp T1 增益"),
    dict(role="INS_T0033_S5", mix="w3.0", lr=3.0, ep=8, seed=316, tier="B",
         note="★trial_0033 家族第 5 种子"),
    dict(role="INS_T0008_S4", mix="w3.0", lr=2.8821, ep=8, seed=317, tier="B",
         note="cs_t0008 家族补种子（该族 4 种子 soup 已 T1 0.8959005 严格 PASS、T3 0.868572）"
              "⇒ 6 种子 soup 有望把 T3 从 0.8686 推过 0.87"),
    dict(role="INS_T0008_S5", mix="w3.0", lr=2.8821, ep=8, seed=318, tier="B",
         note="cs_t0008 家族补种子（第 6 个成员）"),
    dict(role="INS_T0027_S3", mix="w2.0", lr=3.0, ep=8, seed=319, tier="B",
         note="T3 最高家族（w2.0·lr3.0）补种子 ⇒ 4 种子 soup（T3 0.8805 的家族）"),
    dict(role="INS_T0027_S4", mix="w2.0", lr=3.0, ep=8, seed=320, tier="B",
         note="T3 最高家族第 4 个成员"),
    dict(role="INS_T0035_SOUPSRC_S2", mix="w3.0", lr=2.8821, ep=8, seed=321, tier="B",
         init_x="soup_strict3",
         note="soup 初始化家族补种子（foc t0035 用 soup_strict3 起手拿到 T3 0.878782）"),
    # ===== C 段：前沿补点（epoch 阶梯 / 补齐 mix 阶梯）=================
    dict(role="T3_NAT_LR30_8EP", mix="natural", lr=3.0, ep=8, seed=324, tier="C",
         note="natural 混合比在 lr3.0 上的读数（weekend 实测 natural·lr2.907·ep15 拿到 T3 0.879907）"
              "⇒ 判定 natural 与 w2.0 谁更 T3 友好"),
    dict(role="T3_W25_LR30_8EP", mix="w2.5", lr=3.0, ep=8, seed=325, tier="C",
         note="mix 阶梯补点（w2.5 未测过）⇒ 补全 w2.0/w2.5/w3.0 的 T3 排序"),
    # ===== D 段：追加的四条（算力有余 ⇒ 把「最可能命中」的方向加厚）========
    dict(role="INS_T0033_S6", mix="w3.0", lr=3.0, ep=8, seed=327, tier="D",
         note="★trial_0033 家族第 6 种子 ⇒ 4 种子跨种子 soup（唯一「T1 只差 76 个 TP」那族）"),
    dict(role="INS_T0027_S5", mix="w2.0", lr=3.0, ep=8, seed=328, tier="D",
         note="T3 最高家族第 5 个成员 ⇒ 5 种子 soup（T3 0.8805 的家族）"),
    dict(role="INS_W2_LR30_15EP_S3", mix="w2.0", lr=3.0, ep=15, seed=329, tier="D",
         note="T3_W2_LR30_15EP 家族第 3 种子 ⇒ 3 种子 soup（若 ep15 是本轮最优解，"
              "它的 soup 是唯一可能同时拿下 T3 与 T1 红线的零成本手段"),
    # ===== E 段：价值最低、放在最后（被 hard_cut 砍掉也不心疼）=============
    dict(role="T3_W2_LR30_10EP", mix="w2.0", lr=3.0, ep=10, seed=322, tier="E",
         note="epoch 阶梯补点（8/10/15/20/25）⇒ 在 T3 最优点给出 ep→T3 曲线"),
    dict(role="T3_W3_LR30_10EP", mix="w3.0", lr=3.0, ep=10, seed=323, tier="E",
         note="epoch 阶梯补点（w3.0 侧）"),
]


def build_items() -> list[dict]:
    items = []
    for a in ARMS:
        params = dict(SHAPE)
        params["mix_spec"] = a["mix"]
        params["lr_scale"] = a["lr"]
        params["epochs"] = a["ep"]
        params["init_from"] = "rc2"
        if a.get("init_x"):
            params["init_from_x"] = a["init_x"]
        items.append({
            "role": a["role"],
            "seed": a["seed"],
            "note": f"[tier {a['tier']}] {a['note']}",
            "params": params,
        })
    return items


def simulate(items: list[dict], start: datetime, gpus: int) -> list[dict]:
    """4 卡 FIFO 排期模拟：返回每条臂的 (start, end, gpu)。"""
    free_at = [start for _ in range(gpus)]
    rows = []
    for it in items:
        ep = int(it["params"]["epochs"])
        dur = timedelta(minutes=arm_minutes(ep))
        gi = free_at.index(min(free_at))
        s = free_at[gi]
        e = s + dur
        free_at[gi] = e
        rows.append({"role": it["role"], "gpu": gi, "start": s, "end": e,
                     "minutes": arm_minutes(ep), "epochs": ep})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manu/training/nogmc_t3_plan.json")
    ap.add_argument("--start", default="", help="调度器启动时刻（YYYY-MM-DD HH:MM），用于排期模拟")
    ap.add_argument("--gpus", type=int, default=4)
    args = ap.parse_args()

    items = build_items()
    out = Path(args.out)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    total = sum(arm_minutes(int(i["params"]["epochs"])) for i in items)
    print(f"[PLAN] 写入 {out}：{len(items)} 条臂，合计 {total:.0f} GPU·分钟 "
          f"（{total / 60:.1f} GPU·小时）")
    print(f"[PLAN] 4 卡 FIFO 理论完成时刻 ≈ 启动后 {total / 60 / args.gpus:.1f} 小时")

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
        rows = simulate(items, start, args.gpus)
        print(f"\n[PLAN] 排期（启动 {args.start}）：")
        for r in rows:
            print(f"   {r['role']:<26} ep{r['epochs']:<3} {r['minutes']:>6.1f}min  "
                  f"GPU{r['gpu']}  {r['start']:%m-%d %H:%M} → {r['end']:%m-%d %H:%M}")
        print(f"[PLAN] 最后一条臂结束：{rows[-1]['end']:%Y-%m-%d %H:%M}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
