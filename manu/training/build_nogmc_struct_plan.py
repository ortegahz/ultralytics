#!/usr/bin/env python
"""构建 no-GMC **结构搜索**计划：cs_t0043 → 删 neck 的 P2 融合级 + all ReLU。

背景（用户目标）
----------------
> 在「候选：cs_t0043（3 粒子同配方 Model Soup）」这个模型的基础上，删除 p2 分支以及
> wh 预测分支，直接用 cs_t0043 这个模型初始化，并使用 all relu，继续搜索最优模型，
> 期望得到一个速度大幅提升、精度微量下降的模型。

落地后的目标结构（YOLO26n 原生拓扑 + heatmap 头 + 全 ReLU）：

    输入 640x640x3
      骨干 b0..b10   标准 YOLO26 骨干（/2 /4 /8 /16 /32 五级全保留）
      neck          标准 YOLO26 PANet，只在 P3/P4/P5 上做  ← 删掉 neck 的 P2 融合级
      head          HeatmapHead 输出 heatmap(+offset)，**不接任何 bbox/wh 回归**
      激活          全部 ReLU
      初始化        cs_t0043（逐名匹配继承，未匹配的层重新初始化）

相对 cs_t0043 **删掉的模块**（这就是"删 P2 分支"+"删 bbox 预测分支"的落地）：
  · neck 的 P2 融合级：``c19`` / ``up3`` / ``down1`` / ``c22`` / ``up_p2`` / ``fuse_p2``
  · P1(320x320) 头分支：``up_p1`` / ``fuse_p1``
  · bbox/wh 预测分支：本 checkpoint **本来就没有** ⇒ 保持没有（不做 --drop-offset）

⇒ 头的输出分辨率层级成为可配维度 ``--head-level {p3,p2}``：
  · ``p3`` = 标准 YOLO26n 的 Detect 层级：head 吃 **P3 特征 80x80**（速度上限）
  · ``p2`` = 保住小目标分辨率：head 吃**骨干 /4 级的 160x160 特征**（精度底线）
  两者**必须都跑**，让数据裁决。

搜索空间与预算
--------------
预算 = 4 卡 × 14 h ≈ **20 臂**（单臂中位 158.1 min）。

  Stage 1（8 臂）结构扫描：{p3,p2} × {late-up on/off} × {head-conv 64/32}
          超参**固定**（lr_scale=3.0 / epochs=15 / mix=w2.5）⇒ 结构效应可受控归因。
  Stage 2（8 臂）超参精修：Stage-1 最好的 2 个结构 × lr_scale{2.5,3.5} × mix{w2.0,w2.5}
  Stage 3（2 臂）epoch 扫描：全盘最佳配置的 epochs ∈ {8,20}
  Stage 4（2 臂）种子复现：全盘最佳配置 seed ∈ {1,2}

硬约束（上一轮已证伪，**不再花预算**）：
  · ``mix_spec`` 已知是**负杠杆**（−0.8073 T3 / +0.0436 T1）⇒ 只测 w2.0 / w2.5 两档
  · ``clip_max_norm`` 有效应 ≤ +0.03 pp ⇒ 不测
  · α 插值族已知无解 ⇒ 不测
  · ``init_from`` 固定 cs_t0043（用户指定）

汇率红线
--------
上一轮 no-GMC 结论：``T3_FRONTIER`` 局部前沿斜率 = **每丢 1 pp T1 换 5.8 pp T3**
⇒ **新旋钮的汇率必须优于 5.8 pp T3 / 1 pp T1** 才算把前沿向外推。
本线的结构维度是**速度**换精度，汇率由 Stage-1 的受控对照单独估出。

用法::

    python manu/training/build_nogmc_struct_plan.py --out runs/nogmc_struct/plan.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# ---------------------------------------------------------------- 固定锚点
DEFAULT_INIT = "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt"
UAV_DATA = "/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml"
LQ_DATA = "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/data.yaml"

#: cs_t0043 在 **no-GMC 特征语义** 上的三轨基准（主档给定，判据锚点）
ANCHOR = {
    "t1_f1": 0.890766573,
    "t2_f1": 0.998859,
    "t2_fp": 34,
    "t3_f1": 0.903960496,
    "source": "cs_t0043 (3 粒子同配方 Model Soup) on no-GMC features",
}

#: 同板同日实测的 NPU 速度基线（cs_t0043 同源 base 臂）
NPU_BASELINE_MS = 55.5
NPU_TARGET_RANGE_MS = [24.0, 35.0]

#: 上一轮 no-GMC study 实测的「每 epoch 臂耗时（分钟）」中位数
EPOCH_MIN_EST = {8: 92.2, 15: 154.8, 20: 200.0}

#: 结构 → 训练耗时倍率（相对同 epoch 的基准臂；粗估，仅用于 hard_cut 前的时间闸门）
STRUCT_TIME_MULT = {
    ("p3", False): 0.75,
    ("p3", True): 0.70,
    ("p2", False): 1.35,
    ("p2", True): 1.25,
}
HEAD_CONV_TIME_MULT = {32: 0.90, 64: 1.00}

#: 本线判据（**不是** T1 红线）
CRITERIA = {
    "anchor": ANCHOR,
    "t3_drop_max_pp": 1.0,          # T3 相对 cs_t0043 的下降上限（本线主判据）
    "t1_drop_report_pp": None,      # T1 相对 cs_t0043 的下降：记录在案，不做硬闸
    "npu_baseline_ms": NPU_BASELINE_MS,
    "npu_target_range_ms": NPU_TARGET_RANGE_MS,
    "frontier_exchange_rate": 5.8,  # pp T3 / 1 pp T1 —— 新旋钮必须优于它
}


def _arm_id(stage: str, head_level: str, late: bool, head_conv: int, lr: float,
            epochs: int, mix: str, seed: int) -> str:
    lu = "lu1" if late else "lu0"
    return f"{stage}_{head_level}_{lu}_h{head_conv}_lr{lr:g}_ep{epochs}_{mix}_s{seed}"


def _est_min(head_level: str, late: bool, head_conv: int, epochs: int) -> float:
    base = EPOCH_MIN_EST.get(epochs, 200.0)
    mult = STRUCT_TIME_MULT.get((head_level, late), 1.0) * HEAD_CONV_TIME_MULT.get(head_conv, 1.0)
    return round(base * mult, 1)


def _mk(stage: str, head_level: str, late: bool, head_conv: int, lr: float,
        epochs: int, mix: str, seed: int = 0) -> dict:
    return {
        "arm_id": _arm_id(stage, head_level, late, head_conv, lr, epochs, mix, seed),
        "stage": stage,
        # ---- 结构维度 ----
        "head_level": head_level,       # p3 | p2
        "late_upsample": bool(late),    # 头内延迟上采样 on/off
        "head_conv": int(head_conv),    # 64 | 32
        "act": "relu",                  # 用户要求 all relu
        "no_p2": True,                  # 删 neck 的 P2 融合级
        "drop_offset": False,           # 本 checkpoint 无 wh/bbox 分支 ⇒ 保持没有
        # ---- 超参维度 ----
        "lr_scale": float(lr),
        "epochs": int(epochs),
        "mix_spec": mix,
        "seed": int(seed),
        # ---- 固定项 ----
        "init_from": DEFAULT_INIT,
        "est_min": _est_min(head_level, late, head_conv, epochs),
    }


#: 公开别名（dispatcher 用它展开动态阶段）
make_arm = _mk


def build_plan() -> dict:
    stages: list[dict] = []

    # ---------------- Stage 1：结构扫描（8 臂，超参固定） ----------------
    s1 = []
    for head_level in ("p3", "p2"):
        for late in (False, True):
            for head_conv in (64, 32):
                s1.append(_mk("s1", head_level, late, head_conv, 3.0, 15, "w2.5"))
    stages.append({
        "name": "s1_structure",
        "desc": "结构扫描：{p3,p2} × {late-up on/off} × {head-conv 64/32}，超参固定",
        "arms": s1,
    })

    # ---------------- Stage 2：超参精修（8 臂，结构由 Stage-1 选出） ----------------
    stages.append({
        "name": "s2_hyper",
        "desc": "取 Stage-1 最好的 2 个结构 × lr_scale{2.5,3.5} × mix{w2.0,w2.5}，epochs=15",
        "resolve": {
            "pick_structures": 2,
            "from_stage": "s1_structure",
            "lr_scale": [2.5, 3.5],
            "mix_spec": ["w2.0", "w2.5"],
            "epochs": 15,
        },
    })

    # ---------------- Stage 3：epoch 扫描（2 臂） ----------------
    stages.append({
        "name": "s3_epochs",
        "desc": "全盘最佳配置的 epochs ∈ {8,20}（epochs 是最强超参杠杆：+1.1847 T3 / −0.1698 T1）",
        "resolve": {
            "pick_best": 1,
            "from_stage": "auto",
            "epochs": [8, 20],
        },
    })

    # ---------------- Stage 4：种子复现（2 臂） ----------------
    stages.append({
        "name": "s4_seed",
        "desc": "全盘最佳配置的 seed ∈ {1,2}（受控复现 ⇒ 支撑汇率估计）",
        "resolve": {
            "pick_best": 1,
            "from_stage": "auto",
            "epochs": "keep",
            "seed": [1, 2],
        },
    })

    total_fixed = sum(len(s.get("arms", [])) for s in stages)
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "line": "nogmc_struct",
        "description": "cs_t0043 → 删 neck P2 融合级 + all ReLU；head_level{p3,p2} 结构搜索",
        "init_from": DEFAULT_INIT,
        "datasets": {"uav": UAV_DATA, "longquan": LQ_DATA},
        "criteria": CRITERIA,
        "epoch_min_est": EPOCH_MIN_EST,
        "stages": stages,
        "planned_fixed_arms": total_fixed,
        "planned_total_arms": total_fixed + 8 + 2 + 2,   # 8 (s2) + 2 (s3) + 2 (s4)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 no-GMC 结构搜索计划")
    ap.add_argument("--out", default="runs/nogmc_struct/plan.json")
    args = ap.parse_args()

    plan = build_plan()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[PLAN] 写入 {out}")
    print(f"[PLAN] init_from = {plan['init_from']}")
    print(f"[PLAN] 判据锚点 cs_t0043：T1={ANCHOR['t1_f1']:.6f} T2={ANCHOR['t2_f1']:.6f} "
          f"(FP={ANCHOR['t2_fp']}) T3={ANCHOR['t3_f1']:.6f}")
    print(f"[PLAN] 本线判据：T3 相对 cs_t0043 下降 ≤ {CRITERIA['t3_drop_max_pp']} pp；"
          f"T1 下降记录在案；NPU 基线 {NPU_BASELINE_MS} ms（目标 {NPU_TARGET_RANGE_MS} ms）")
    print(f"[PLAN] 汇率红线：新旋钮须优于 {CRITERIA['frontier_exchange_rate']} pp T3 / 1 pp T1")
    print("-" * 100)
    for st in plan["stages"]:
        if "arms" in st:
            print(f"[PLAN] Stage {st['name']:<16} {len(st['arms'])} 臂 | {st['desc']}")
            for a in st["arms"]:
                print(f"          - {a['arm_id']:<44} est={a['est_min']:>6.1f} min")
        else:
            print(f"[PLAN] Stage {st['name']:<16} 动态 | {st['desc']}")
    print("-" * 100)
    total_fixed = sum(len(s.get("arms", [])) for s in plan["stages"])
    print(f"[PLAN] 合计计划臂数 = {plan['planned_total_arms']}（固定 {total_fixed} + 动态 12）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
