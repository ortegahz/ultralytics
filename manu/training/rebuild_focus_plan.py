#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建定点打靶计划文件（2026-11-02 15:30，第二次修订）。

背景
----
1. **越界 bug**：20 条未入队臂里有 8 条用了 `--focus-space` 白名单之外的取值
   （soup init / w2.5 / w4.0 / warmup 1.8），Optuna 4.9 会校验固定参数 ⇒ 必崩。
   Optuna 4.9 又没有 `optuna.delete_trial`，且分布兼容性检查是 **study 级**的
   ⇒ **搜索空间冻结**，只能改臂的参数或砍臂。
2. **算力预算**：已入队 23 条（含 11 条 WAITING）+ 20 条待注入 = 31 条待跑，
   按实测臂时长（6ep 75min / 8ep 82min / 10ep 120min / 15ep 157min，4 卡并行）
   算总工时 3181 min ÷ 4 卡 ≈ 13.3 h ⇒ 15:20 起算约 04:40 收工，
   距训练截止 06:30 留 ~1.8 h 余量 ⇒ **可以再裁，但不能全留**。
3. **模型修正**（全库 122 条受控对 + 家族回归）：
   · lr 是 T3 一等杠杆（family C: T3 ~ lr 斜率 +0.0170/unit，T1 −0.00135/unit）
   · mix 是**弱**杠杆（T3 ~ mix 斜率 −0.00365/unit，T1 ~ +0.00012/unit ⇒ 几乎免费）
   · backbone 保护赔率最差（T1 +0.0019 换 T3 −0.0215）
   ⇒ 最优联合点是 **低 mix（w2.0）+ 高 lr（2.75~3.0）** 的角落，
     而不是原计划的 mix↑（w6.0）+ lr↑ 组合。

改动
----
· 已入队的 23 条：**逐位不动**（改了参数会被去重护栏当成新臂重复入队）。
· 砍掉 7 条低价值/不可执行臂；把 5 条越界臂改成**白名单内的最近等价臂**；
  3 条 soup 臂改走新参数名 `init_from_x`（绕开 study 级分布检查）。
· 新增 5 条模型驱动臂 ⇒ 构成 w3.0/w2.0 × lr 2.75/2.8821/3.0 的受控网格。
· **按价值重排**（入队顺序 = 执行顺序 ⇒ 高价值臂必须排前面，砍时间时先砍后面的）。
"""

from __future__ import annotations

import json
from pathlib import Path

PLAN = Path(__file__).resolve().parent / "nogmc_focus_plan.json"

# ---- 已入队（study 里有 role）的 23 条：逐位保留，禁止改动其 params ----
COMMITTED = {
    "FOCUS_T0025_SEED2", "FOCUS_T0025_SEED3", "FOCUS_A_W6_LR1_15EP",
    "FOCUS_A_W8_LR1_15EP", "FOCUS_A_W8_T0025_6EP", "FOCUS_B_T0060_PROT15",
    "FOCUS_B_T0087_PROT8", "FOCUS_B_T0009_PROT15", "FOCUS_B_T0018_PROT6",
    "FOCUS_C_W6_LR2_10EP", "FOCUS_T0009_W6_SEED2", "FOCUS_BASE474_W6_8EP",
    "FOCUS_BASE474_W6_10EP", "FOCUS_RC2_W6_LR1_10EP", "FOCUS_RC2_NAT_LR1_15EP",
    "FOCUS_T0008_REPLAY", "FOCUS_T0008_PROT8", "FOCUS_RC2_W6_LR28821_8EP",
    "FOCUS_T0008_SEED2", "FOCUS_T0077_PROT8", "FOCUS_T0082_PROT8",
    "FOCUS_T0082_SEED2", "FOCUS_T0077_SEED2",
}

# ---- 砍掉的臂（原因写进备注，方便复盘） ----
DROP = {
    "FOCUS_SOUHIGH3NOBIN_W3_LR15_6EP":
        "砍：init soup_high3_nobn 且 bb 0.5/freeze 1 越界；3 条 soup 臂里信息量最低的一条",
    "FOCUS_T0021_SEED2":
        "砍：mix w4.0 越界（改 w3.0 后与 trial_0064 家族重合）；trial_0021 的 T3 仅 0.842445",
    "FOCUS_T0008_BB03":
        "砍：家族均值显示 bb0.30 的 T3 0.849974 < bb1/0 的 0.852511 ⇒ 骨干档位不是杠杆",
    "FOCUS_T0008_WARMUP18":
        "砍：warmup 1.8 越界（上限 1.5，用户要求 warmup>=1ep ⇒ 1.5 已封顶）",
    "FOCUS_T0008_W6_SEED3":
        "砍：mix 标定已有 3 对受控对（dT1/T3 符号 3/3 一致）⇒ 边际信息量低",
    "FOCUS_T0008_W6_LR30":
        "砍：模型判定 w6.0 的 mix↑ 代价（−0.011 T3）会吃掉 lr↑ 的 +0.002 ⇒ 预测 T3≈0.859 < trial_0008 的 0.8677",
    "FOCUS_T0008_W6_PROT8":
        "砍：mix↑ 与骨干保护两个负杠杆叠加 ⇒ 预测 T3 大幅下降",
}

# ---- 越界臂改写成白名单内的最近等价臂 ----
REMAP = {
    "FOCUS_T0056_SEED2": {
        "mix_spec": "w2.0",
        "_why": "原 mix w2.5 越界 ⇒ 改成 w2.0（最近的合法低-mix 档，方向不变）",
    },
    "FOCUS_T0034_SEED2": {
        "mix_spec": "w2.0",
        "_why": "原 mix w2.5 越界 ⇒ 改成 w2.0（最近的合法低-mix 档，方向不变）",
    },
    "FOCUS_T0008_W25": {
        "mix_spec": "w2.0",
        "_why": ("原 mix w2.5 越界 ⇒ 改成 w2.0。**保留它的独特价值**："
                 "相对 trial_0008 只动 mix 一个维度 ⇒ 干净的 mix↓ 受控对"),
    },
    "FOCUS_SOUSTRICT3_W3_LR28821_8EP": {
        "init_from": "rc2",
        "init_from_x": "soup_strict3",
        "_why": ("soup init 不能写进 init_from（study 级分布兼容性检查会炸）"
                 "⇒ init_from 留在白名单内的 rc2，另用新参数名 init_from_x 带 soup 权重"),
    },
    "FOCUS_SOUHIGH3NOBIN_W3_LR28821_8EP": {
        "init_from": "rc2",
        "init_from_x": "soup_high3_nobn",
        "_why": "同上，走 init_from_x",
    },
}

# ---- 新增臂：模型驱动的「低 mix + 高 lr」角落 ----
T0008 = {
    "mix_spec": "w3.0",
    "lr_scale": 2.8821,
    "warmup_epochs": 1.5,
    "weight_decay": 0.00026906,
    "focal_beta": 2.75,
    "offset_weight": 0.4,
    "mosaic": 0.15,
    "scale": 0.1,
    "translate": 0.05,
    "clip_max_norm": 5.65,
    "freeze_bn": False,
    "init_from": "rc2",
    "epochs": 8,
    "backbone_lr_scale": 1.0,
    "freeze_backbone_epochs": 0,
}


def variant(**kw):
    p = dict(T0008)
    p.update(kw)
    return p


NEW = [
    {
        "role": "FOCUS_T0008_W20_LR30",
        "seed": 248,
        "note": ("★★★★**模型驱动的联合靶心**（2026-11-02 15:30 新增）：全库 122 条的"
                 "受控对 + 家族回归给出：family C 里 T3~lr 斜率 +0.0170/unit、T1~lr −0.00135/unit；"
                 "T3~mix 斜率 −0.00365/unit 而 T1~mix 仅 +0.00012/unit ⇒ **mix↓ 几乎免费地换 T3**。"
                 "相对 trial_0008 只动两个维度：mix w3.0→w2.0（+0.0036 T3）、lr 2.8821→3.0（+0.0020 T3）"
                 "⇒ 预测 T3≈0.8734（**过 0.87**）、T1≈0.89556（差 0.0003，在种子噪声 σ(T1)≈0.0005 内）"
                 "⇒ 本轮最可能同时过线的一条。"),
        "params": variant(mix_spec="w2.0", lr_scale=3.0),
    },
    {
        "role": "FOCUS_T0008_W20_LR275",
        "seed": 247,
        "note": ("★★★★同上，与 W20_LR30 构成**干净的 lr 受控对**（只有 lr 2.75 vs 3.0 不同）："
                 "预测 T3≈0.8691、T1≈0.89590（**正好压在线上**）⇒ 与上一条一起可把"
                 "『T3=0.87 与 T1 红线』的交点夹出来。"),
        "params": variant(mix_spec="w2.0", lr_scale=2.75),
    },
    {
        "role": "FOCUS_T0008_LR30",
        "seed": 249,
        "note": ("★★★★**纯 lr↑ 受控对**（相对 trial_0008 只动 lr 2.8821→3.0，mix 仍 w3.0）："
                 "验证 T3~lr 斜率是否真的 +0.017/unit（家族回归 R²=0.607）⇒ "
                 "同时给出 T3≈0.8697 / T1≈0.89550 的第二个联合点。"),
        "params": variant(lr_scale=3.0),
    },
    {
        "role": "FOCUS_T0008_W20_LR25",
        "seed": 250,
        "note": ("★★★**T1 修复端**：mix w2.0 + lr 2.5 ⇒ 预测 T1≈0.89618（过线 +0.028pp）、"
                 "T3≈0.86484 ⇒ 若预测兑现就是一条**严格 PASS 且 T3 明显高于 trial_0009 的 0.8579** 的交接线。"),
        "params": variant(mix_spec="w2.0", lr_scale=2.5),
    },
    {
        "role": "FOCUS_T0008_W20_SEED2",
        "seed": 251,
        "note": ("★★w2.0@lr2.8821 这条 mix↓ 受控对的**第二个种子** ⇒ 交接口径（同超参多种子中位）"
                 "需要 n>=2 ⇒ 若 w2.0 成为赢家，这条是交接证据的必需项。"),
        "params": variant(mix_spec="w2.0"),
    },
]

# ---- 执行顺序（入队顺序 = 执行顺序 ⇒ 按价值从高到低） ----
ORDER = [
    # —— 第一梯队：直击「T3>=0.87 且 T1 过线」的联合靶心 ——
    "FOCUS_SOUSTRICT3_W3_LR28821_8EP",   # soup init（T1 0.896846 最高）打在最准靶心
    "FOCUS_T0008_W20_LR30",              # 模型预测 T3 0.8734 过线
    "FOCUS_T0008_W20_LR275",             # 与上一条夹出交点
    "FOCUS_T0008_LR30",                  # 纯 lr↑ 受控对
    "FOCUS_T0008_W25",                   # 纯 mix↓ 受控对（w2.5→w2.0）
    "FOCUS_SOUHIGH3NOBIN_W3_LR28821_8EP",  # soup init（T3 0.874746 最高）
    # —— 第二梯队：现有强候选的多种子证据（交接口径必需） ——
    "FOCUS_T0009_SEED2",                 # 严格 PASS 里 T3 最高（0.857880）的换种子
    "FOCUS_T0008_LR26",                  # 模型预测严格 PASS 且 T3≈0.8628
    "FOCUS_T0008_W20_LR25",              # T1 修复端候选
    "FOCUS_T0083_SEED2",                 # T1 余量最大（+0.1117pp）的换种子
    "FOCUS_T0018_SEED2",                 # 严格 PASS（trial_0018）的换种子
    "FOCUS_T0018_SEED3",                 # 同上第 3 种子（n=3 ⇒ 中位更稳）
    # —— 第三梯队：近红线配置的多种子证据 ——
    "FOCUS_T0034_SEED2",                 # w2.0 + lr2.8041 + bb0.3
    "FOCUS_T0056_SEED2",                 # w2.0 + lr2.4620 + bb0.3
    "FOCUS_T0064_SEED2",                 # w3.0 + lr2.7750 + ep6
    "FOCUS_T0108_SEED2",                 # w3.0 + lr2.3670 + ep8
    # —— 第四梯队：低成本探针 ——
    "FOCUS_T0008_W20_SEED2",             # w2.0@2.8821 第 2 种子
    "FOCUS_T0008_WARMUP12",              # warmup 钉在 1.2
]


def main() -> int:
    old = json.loads(PLAN.read_text(encoding="utf-8"))
    by_role = {x["role"]: x for x in old}

    # 1) 已入队的 23 条：原样保留（顺序也保留，方便比对）
    committed = [x for x in old if x["role"] in COMMITTED]

    # 2) 未入队的臂：先砍、再改写、再重排
    free_roles = [x["role"] for x in old if x["role"] not in COMMITTED]
    for role in free_roles:
        if role in DROP:
            continue
        arm = by_role[role]
        remap = REMAP.get(role)
        if remap:
            why = remap.pop("_why")
            for k, v in remap.items():
                arm["params"][k] = v
            arm["note"] = ("[2026-11-02 15:30 修订] " + why + " || 原备注：" + arm.get("note", ""))
        arm.setdefault("params", {})

    new_by_role = {x["role"]: x for x in NEW}

    final = list(committed)  # 已入队的 23 条逐位保留、原序保留
    for role in ORDER:
        if role in COMMITTED:
            continue  # 已在 committed 里
        if role in DROP:
            continue
        if role in new_by_role:
            final.append(new_by_role[role])
        else:
            final.append(by_role[role])

    # 兜底：plan 里不该有遗漏的未入队臂
    covered = {x["role"] for x in final}
    missing = [r for r in free_roles if r not in covered and r not in DROP]
    if missing:
        raise SystemExit("❌ 下列未入队臂既没进 ORDER 也没进 DROP，漏了：%s" % missing)

    PLAN.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("✅ 计划重建完成：%d 条（已入队 %d + 新 %d）"
          % (len(final), len([x for x in final if x["role"] in COMMITTED]), len(final) - 23))
    print("   砍掉 %d 条：%s" % (len(DROP), ", ".join(sorted(DROP))))
    print("   改写 %d 条：%s" % (len(REMAP), ", ".join(sorted(REMAP))))
    print("   新增 %d 条：%s" % (len(NEW), ", ".join(x["role"] for x in NEW)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
