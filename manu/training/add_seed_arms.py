#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给定点打靶计划**补 4 条"补种子"臂**并按新证据重排（2026-11-02 15:40）。

为什么补种子而不是补配置
------------------------
2026-11-02 15:35 用精确参数重扫受控对后拿到两条硬数字：

1. **杠杆排序**（每换 1 单位 T3 要付的 T1）：lr_scale **0.047** < epochs 0.088
   < backbone（负向）< mix（负向）；init_from 是类别维度（rc2→base474：T1 +0.0003~0.001，T3 −0.016~−0.039）。
   ⇒ **T3 只能靠 lr / epochs / init 买**，mix 与骨干保护是**反向**的。
2. **种子噪声**：同配方换种子的 |ΔT1| 中位 **0.002464**（最大 0.003104）、
   |ΔT3| 中位 **0.002995**（最大 0.005394）。

而离两个目标最近的几个配置，T1 缺口只有 **0.000146 ~ 0.001234**：

| 配置 | T3 | T1 缺口 | 与种子噪声之比 |
|:--|--:|--:|--:|
| trial_0082 配方 | 0.865458 | 0.000146 | 0.06 σ |
| trial_0008 配方 | 0.867711 | 0.000236 | 0.10 σ |
| trial_0077 配方 | 0.871475 | 0.001234 | 0.50 σ |

⇒ **"过不过 T1 红线"基本由种子运气决定，而不是配置本身** ⇒
**交接口径（同超参多种子中位）是唯一能给出可信结论的办法**，
而 n=2 的中位仍然很脆 ⇒ **给这三个最近点各补一个种子（n=3）比再开新配置更值钱**。

另外给模型预测的联合靶点 `w2.0 + lr3.0`（预测 T3 0.8734 过线、T1 差 0.0003）补一个种子，
否则它即使跑出好结果也只有 n=1 ⇒ 交接口径同样不成立。

重排原则
--------
**入队顺序 = 执行顺序**（`study.ask()` 按 trial_id FIFO）⇒ 超时被砍的只能是排在后面的低价值臂。
⇒ 4 条补种子臂插到**最前面**（紧跟模型靶点），保证它们一定跑到。
"""

from __future__ import annotations

import json
from pathlib import Path

PLAN = Path(__file__).resolve().parent / "nogmc_focus_plan.json"

# ---- 4 条补种子臂的完整配方（全部从"同配方"复制，只换 seed）----
T0008 = {
    "mix_spec": "w3.0", "lr_scale": 2.8821, "warmup_epochs": 1.5,
    "weight_decay": 0.00026906, "focal_beta": 2.75, "offset_weight": 0.4,
    "mosaic": 0.15, "scale": 0.1, "translate": 0.05, "clip_max_norm": 5.65,
    "freeze_bn": False, "init_from": "rc2", "epochs": 8,
    "backbone_lr_scale": 1.0, "freeze_backbone_epochs": 0,
}
W20_LR30 = dict(T0008, mix_spec="w2.0", lr_scale=3.0)
T0077 = {
    "mix_spec": "w3.0", "lr_scale": 2.7675790243131346, "warmup_epochs": 1.4480592521676685,
    "weight_decay": 3.132e-05, "focal_beta": 2.5, "offset_weight": 0.55,
    "mosaic": 0.15, "scale": 0.2, "translate": 0.05, "clip_max_norm": 4.05,
    "freeze_bn": False, "init_from": "rc2", "epochs": 8,
    "backbone_lr_scale": 1.0, "freeze_backbone_epochs": 0,
}
T0082 = {
    "mix_spec": "w3.0", "lr_scale": 2.422, "warmup_epochs": 1.359,
    "weight_decay": 2.059e-05, "focal_beta": 2.5, "offset_weight": 0.6,
    "mosaic": 0.15, "scale": 0.15, "translate": 0.05, "clip_max_norm": 5.09,
    "freeze_bn": False, "init_from": "rc2", "epochs": 8,
    "backbone_lr_scale": 1.0, "freeze_backbone_epochs": 0,
}

NEW = [
    {
        "role": "FOCUS_T0008_W20_LR30_SEED2",
        "seed": 252,
        "note": ("★★★★★**模型靶点的第 2 个种子**（2026-11-02 15:40 新增）："
                 "w2.0+lr3.0 是模型预测的联合靶点（T3≈0.8734 过 0.87、T1≈0.89556 差 0.0003）；"
                 "**种子噪声 |ΔT1| 中位 0.0025 ≫ 该缺口 0.0003** ⇒ 单种子结果无法判定，"
                 "必须 n≥2 才能进交接口径 ⇒ 这条是它的必需配套。"),
        "params": W20_LR30,
    },
    {
        "role": "FOCUS_T0008_SEED3",
        "seed": 253,
        "note": ("★★★★★**离双目标最近点的第 3 个种子**（trial_0008 配方 w3.0·lr2.8821·8ep·rc2）："
                 "T3 0.867711 / T1 缺口仅 0.000236（0.10 σ）⇒ n=2 的中位仍很脆 ⇒ 补到 n=3。"),
        "params": T0008,
    },
    {
        "role": "FOCUS_T0077_SEED3",
        "seed": 254,
        "note": ("★★★★★**唯一 T3 已过 0.87 的配方的第 3 个种子**（trial_0077 配方 w3.0·lr2.7676·8ep·rc2）："
                 "T3 0.871475（过线）/ T1 缺口 0.001234（0.50 σ）⇒ 它的 T1 是**种子运气问题**，"
                 "补到 n=3 才能给出可信中位。"),
        "params": T0077,
    },
    {
        "role": "FOCUS_T0082_SEED3",
        "seed": 255,
        "note": ("★★★★**T1 缺口最小的配方的第 3 个种子**（trial_0082 配方 w3.0·lr2.422·8ep·rc2）："
                 "T1 缺口 0.000146（0.06 σ，最小）/ T3 0.865458 ⇒ 一旦某个种子过线就是最稳的交接线。"),
        "params": T0082,
    },
]

# ---- 重排：已入队的 23 条不动；未入队的按价值重排（补种子臂插最前）----
ORDER = [
    # —— 第一梯队：模型靶点 + 它的配套种子（必须先跑完，否则 n 不足）——
    "FOCUS_T0008_W20_LR30",
    "FOCUS_T0008_W20_LR30_SEED2",
    "FOCUS_T0008_SEED3",
    "FOCUS_T0077_SEED3",
    "FOCUS_T0082_SEED3",
    # —— 第二梯队：另两个模型靶点 + 干净受控对 ——
    "FOCUS_T0008_W20_LR275",
    "FOCUS_T0008_LR30",
    "FOCUS_T0008_W25",
    # —— 第三梯队：soup init（唯一还没试的机制）——
    "FOCUS_SOUSTRICT3_W3_LR28821_8EP",
    "FOCUS_SOUHIGH3NOBIN_W3_LR28821_8EP",
    # —— 第四梯队：其余模型点 / 多种子证据 ——
    "FOCUS_T0008_W20_LR25",
    "FOCUS_T0009_SEED2",
    "FOCUS_T0008_LR26",
    "FOCUS_T0083_SEED2",
    "FOCUS_T0018_SEED2",
    "FOCUS_T0018_SEED3",
    "FOCUS_T0034_SEED2",
    "FOCUS_T0056_SEED2",
    "FOCUS_T0064_SEED2",
    "FOCUS_T0108_SEED2",
    "FOCUS_T0008_W20_SEED2",
    "FOCUS_T0008_WARMUP12",
]


def main() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    by_role = {x["role"]: x for x in plan}
    committed = [x for x in plan if x["role"] not in by_role or True]  # 保序，见下

    # 已入队的 23 条（study 里有 role 的）——逐位不动
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
    committed = [x for x in plan if x["role"] in COMMITTED]

    new_by_role = {x["role"]: x for x in NEW}
    final = list(committed)
    seen = set(COMMITTED)
    for role in ORDER:
        if role in seen:
            continue
        if role in new_by_role:
            final.append(new_by_role[role])
        elif role in by_role:
            final.append(by_role[role])
        else:
            raise SystemExit("❌ ORDER 里的角色不存在：%s" % role)
        seen.add(role)

    # 兜底：plan 里不该有遗漏的未入队臂
    missing = [x["role"] for x in plan if x["role"] not in seen]
    if missing:
        raise SystemExit("❌ 下列臂既没进 ORDER 也没在已入队集合里，漏了：%s" % missing)

    PLAN.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("✅ 计划更新完成：%d 条（已入队 %d + 未入队 %d）" % (len(final), len(committed), len(final) - len(committed)))
    print("   新增补种子臂 %d 条：%s" % (len(NEW), ", ".join(x["role"] for x in NEW)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
