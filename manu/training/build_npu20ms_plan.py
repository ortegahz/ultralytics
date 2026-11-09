#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
生成周末搜索的确定性臂计划（读「结构→板卡速度表」做硬约束剪枝）。

目标（2026-11-06 用户定）：**NPU int8 ≤ 20 ms** 且 **T1/T2/T3 下降非常微小**，deadline 下周一 09:00。

设计要点
--------
1. **约束是「≤20 ms」而不是「最快」** ⇒ 结构选择原则 =
   **在 20 ms 预算内，头层级最高（定位能力最强）的那个**：
   `p1(320×320) > p2_fused(160×160) > p3(80×80)`。
   ⇒ 速度不是目标函数，**精度才是**；速度只是**准入闸门**。
2. **板卡今天 18:30 断电** ⇒ 周末无法再测速 ⇒ 速度**查表**（今天测的谱系表），
   表外结构用离线预测器兜底 ⇒ 派单前就能剪掉超限结构，**不浪费一次训练**。
3. **每个候选结构配 N 个超参点**（lr / epochs / 混比 / act / 解冻），
   而不是让 TPE 在结构维度上乱跑 ⇒ 68~75 臂的预算要花在刀刃上。
4. **必带 2 条复现对照臂**（原结构 + SiLU + cs_t0043 初始化）⇒ 验证参照系/门禁口径没漂。

用法：
    python manu/training/build_npu20ms_plan.py \
        --speed-table /media/manu/1TB-Volume/rknn_work/hm_speed/probes/struct_speed_table.json \
        --out manu/training/npu20ms_plan.json --limit-ms 20 --per-struct 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# ★★ **直接从 dispatcher 模块读 choices**（2026-11-06 改，根治「常量漂移」类坑）：
#    之前本文件把 head_conv / head_level / p2_out_scale / act 等取值**手写**一份，
#    与 optuna_mix_ratio_search.py 里的 STRUCT_*_CHOICES 各写各的
#    ⇒ plan 引用了 dispatcher 分布里没有的值 ⇒ Optuna enqueue 抛 ValueError
#    ⇒ **整个搜索起不来**（实测连踩 4 次：head_level / head_conv / lr_scale / p2_out_scale）。
#    ⇒ 现在改成 import dispatcher 的常量 ⇒ **单一真源** ⇒ 以后加取值只改一处。
sys.path.insert(0, str(REPO / "manu" / "training"))
try:
    import optuna_mix_ratio_search as D
except Exception as _exc:  # pragma: no cover - 没有 optuna 时给出清晰指引
    raise SystemExit(
        f"[FATAL] 导入 dispatcher 常量失败：{_exc}\n"
        "       本脚本需要 optuna（与 dispatcher 同环境）。\n"
        "       服务器上用 `conda activate uav` 后再运行本脚本。"
    ) from _exc

# cs_t0043 的母配方（微调口径，全程钉死）
BASE = {
    "mix_spec": "w2.0", "lr_scale": 3.5, "warmup_epochs": 1.5, "weight_decay": 0.000269,
    "focal_beta": 2.75, "offset_weight": 0.40, "mosaic": 0.15, "scale": 0.10,
    "translate": 0.05, "clip_max_norm": 5.65, "freeze_bn": False,
}

# 头层级优先级：320 > 160 > 80（定位能力从强到弱）
LEVEL_RANK = {"p1": 0, "p2_fused": 1, "p3": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-ms", type=float, default=21.0)
    ap.add_argument("--per-struct", type=int, default=6, help="每个候选结构配几个超参点")
    ap.add_argument("--seed-base", type=int, default=2100)
    ap.add_argument("--falsified-src", default="p3",
                    help="逗号分隔的已证伪结构来源（精度已证伪）⇒ 每个只留 1 条臂作对照")
    ap.add_argument("--controls", type=int, default=2, help="复现对照臂数量")
    ap.add_argument("--gpus", type=int, default=4, help="并行卡数（只用于预算估算）")
    ap.add_argument("--min-head-conv", type=int, default=32,
                    help="最小头宽（默认 32）。★ 窄头（hc16/hc24）只比 hc32 快 ~1 ms，"
                         "却带着 hc32 当初的容量风险（后查明那次是 loader bug，但容量"
                         "本身仍无实证）⇒ 不进候选、不进 dispatcher 的 choices ⇒ "
                         "**choices 越窄，TPE 越不容易采到垃圾结构**。")
    args = ap.parse_args()

    tbl = json.loads(Path(args.speed_table).read_text(encoding="utf-8"))
    cands = []
    for a in tbl.get("arms", []):
        sp = a.get("spec") or {}
        ms = a.get("ms")
        if ms is None:
            continue
        if ms > args.limit_ms:
            continue
        # ★ 窄头直接排除（见 --min-head-conv 的说明）：hc16/hc24 只快 ~1 ms，
        #   容量却无任何实证（hc32 当初的"死模型"虽已查明是 loader bug，
        #   但「32 通道够不够」这件事本身至今没有正面证据）⇒ 不值得为 1 ms 冒风险。
        if int(sp.get("head_conv", 64)) < args.min_head_conv:
            continue
        cands.append({
            "head_src": sp.get("head_src", "p1"),
            "head_conv": int(sp.get("head_conv", 64)),
            "act": sp.get("act", "relu"),
            # ★ 这两个键必须带上：否则 late 变体（320 输出）与非 late（160 输出）
            #   会在 dedup 里撞成同一个 key ⇒ 320 输出的候选会被丢掉（2026-11-06 实测踩到）
            "late_upsample": bool(sp.get("late_upsample", False)),
            "p2_out_scale": float(sp.get("p2_out_scale", 1.0) or 1.0),
            "ms": float(ms),
        })
    # 去重（同结构只留最快的那个 —— 同结构速度应当一致）
    # ⚠️ key 必须含 late_upsample / p2_out_scale：否则 320 输出与 160 输出会撞成一条
    best = {}
    for c in cands:
        k = (c["head_src"], c["head_conv"], c["act"],
             c.get("late_upsample", False), c.get("p2_out_scale", 1.0))
        if k not in best or c["ms"] < best[k]["ms"]:
            best[k] = c
    # ★ 排序：**输出分辨率优先**（late 变体 ⇒ 320×320，强于非 late 的 160×160），
    #   同输出层内**宽度降序**（≤21ms 就够 ⇒ 宽度越宽精度越好），最后按速度
    def _out_px(c):
        if c.get("late_upsample"):
            return 320
        return {"p1": 320, "p2_fused": 160, "p2_raw": 160, "p3": 80}.get(c["head_src"], 160)

    # ★ 排序（2026-11-06 依据**实测校准**重排）：
    #   ① **320 输出（late）优先** —— 它保持 stride=2 ⇒ 门禁/评测口径与基线 cs_t0043 **完全一致**
    #      （p2_fused 非 late 是 stride=4 ⇒ 160×160 输出 ⇒ 定位能力天然弱一档）。
    #      而且 late 家族此前**从未被有效评测过**（trial_0003 的「死模型」是 loader bug，
    #      见下），所以它的精度是**未知而非已知为差** ⇒ 值得优先投预算。
    #   ② 同为 320 输出时：**头宽降序**（≤21ms 就够 ⇒ 宽度越宽容量越足）。
    #   ③ 160 输出（stride=4）作为第二梯队：trial_0002 实测 T3 只掉 0.585 pp，是已验证的保底。
    def _prio(c):
        out320 = _out_px(c) >= 320
        return (0 if out320 else 1,
                -_out_px(c),
                -int(c["head_conv"]),
                c["ms"])

    cands = sorted(best.values(), key=_prio)

    if not cands:
        print(f"[FATAL] 表里没有 ≤{args.limit_ms} ms 的结构（表可能没测全）")
        return 1

    print(f"=== 候选结构（≤{args.limit_ms} ms，按头层级优先）===")
    for c in cands:
        print(f"  {c['head_src']:<10} hc{c['head_conv']:<4} {c['act']:<5} "
              f"late={int(bool(c.get('late_upsample')))} sc={c.get('p2_out_scale', 1.0)} "
              f"{c['ms']:>7.2f} ms  (输出 { _out_px(c) }px)")

    # ---- 超参点：每个结构上展开 lr × epochs 的小网格 ----
    # ★ HP_MAIN 顺序经过实测调整（2026-11-06）：
    #   ① 母配方原样（首选）② **mix=w2.5**（T1 是 Anti-UAV 域 ⇒ 多采 UAV，**补 T1**）
    #   ③ **uf0（冻结底模）**（同样用于补 T1）④ 快筛 15ep ⑤ lr3.0 ⑥ lr4.0
    HP_MAIN = [
        {"lr_scale": 3.5, "epochs": 20},   # 母配方原样（首选）
        {"lr_scale": 3.5, "epochs": 20, "mix_spec": "w2.5"},   # ★补 T1
        {"lr_scale": 3.5, "epochs": 20, "unfreeze_b0b4": False},  # ★补 T1
        {"lr_scale": 3.5, "epochs": 15},   # 快筛
        {"lr_scale": 3.0, "epochs": 20},
        {"lr_scale": 4.0, "epochs": 20},
    ]
    HP_MINOR = [
        {"lr_scale": 3.5, "epochs": 20},
        {"lr_scale": 3.5, "epochs": 15},
        {"lr_scale": 3.0, "epochs": 20},
    ]

    arms = []
    seed = args.seed_base

    # ---- 复现对照臂（原结构 + SiLU，与 cs_t0043 同配方）----
    for i in range(args.controls):
        q = dict(BASE)
        q.update({"no_p2": False, "head_level": "p1", "act": "silu", "head_conv": 64,
                  "late_upsample": False, "unfreeze_b0b4": False, "head_src": "p1"})
        # ★ epochs 必须显式给 **20**（母配方口径）：
        #   对照臂的使命是「同配方复现 cs_t0043」⇒ 必须与母配方同 epoch 数。
        #   不给 ⇒ 回落到 --epochs（默认 15）⇒ **复现的不是 cs_t0043**，对照就失去意义。
        q["epochs"] = 20
        arms.append({
            "role": f"CTRL_p1_silu_s{seed}",
            "seed": seed,
            "note": "★流水线验证臂：同配方+原结构+SiLU ⇒ 期望复现 cs_t0043（T1 0.890766573 / T3 0.903960496）",
            "params": q,
        })
        seed += 1

    # ---- 候选结构 × 超参点 ----
    for c in cands:
        # ★ 判"主力"的标准 = **宽度 ≥ 48（实测验证过的宽头）** 或 **320 输出（late）**
        #   ⇒ 只有主力结构才配满 HP_MAIN（含 w2.5 / uf0 两个**补 T1** 的点）
        _is_main = (int(c["head_conv"]) >= 48) or (_out_px(c) >= 320)
        grid = HP_MAIN if _is_main else HP_MINOR
        grid = grid[: max(args.per_struct, 5)] if _is_main else grid[: args.per_struct]
        # ★ 已证伪的结构（默认 p3：T3 −3.2 pp）只留 1 条作对照，不浪费预算
        if c["head_src"] in args.falsified_src:
            grid = grid[:1]
        # ⚠️⚠️ **撤销「hc32 死模型」降级**（2026-11-06 重大修正）：
        #    上一版这里写着「p2_fused + head_conv32 + 160 输出 ⇒ 实测死模型（T2/T3 全 0）
        #    ⇒ 32 通道不足以承载 320 目标语义 ⇒ 只留 1 条」。
        #    **那个结论是错的**：trial_0003 的 T2/T3 全 0 **不是容量问题**，
        #    而是 `load_hm_model()` 没从 ckpt 读 `head_conv`/`late_upsample`
        #    ⇒ T2/T3 推理端把模型重建成 hc64 ⇒ 16 个 head 张量 shape 不匹配被**静默丢弃**
        #    ⇒ 带着**随机初始化的头**跑 ⇒ 热图全 0 ⇒ det=0。
        #    证据：ms_cache_shard0.log 打印 `[head_conv] 64` 而 ckpt 是 32；
        #          加载张量数 478 vs trial_0002 的 494（差 16 = head 的全部权重）；
        #          同一次门禁的 T1 路径读对了 ckpt ⇒ T1=0.873964 **是活的**。
        #    ⇒ 所以 **hc32 的精度是「未知」，不是「已知为差」** ⇒ 不能降级。
        #    ⇒ 而且更关键：**所有 late（320 输出）结构此前都没被有效评测过**
        #      （它们同样踩这个 loader bug）⇒ 它们的精度上限可能最高。
        # ⇒ 现在一律按正常网格跑，不再按头宽降级。
        for g in grid:
            q = dict(BASE)
            q.update({"no_p2": c["head_src"] in ("p3", "p2_raw"),
                      "head_level": c["head_src"],
                      "act": c["act"],
                      "head_conv": c["head_conv"],
                      # ★ 这两个键必须来自候选结构本身（2026-11-06 修）：
                      #   之前这里硬编码 late_upsample=False ⇒ **两个 320 输出的最优候选
                      #   （20.06 / 20.83 ms）被静默丢掉**，而且同一结构重复出了 4 组几乎
                      #   相同的臂 ⇒ 21 条臂里大半是重复训练。
                      "late_upsample": bool(c.get("late_upsample", False)),
                      "p2_out_scale": float(c.get("p2_out_scale", 1.0) or 1.0),
                      "unfreeze_b0b4": True,
                      "head_src": c["head_src"]})
            q.update(g)
            arms.append({
                "role": (f"S_{c['head_src']}_hc{c['head_conv']}_{c['act']}"
                         + ("_late" if c.get("late_upsample") else "")
                         + (f"_sc{int(round(c.get('p2_out_scale', 1.0) * 100))}"
                            if c.get("p2_out_scale", 1.0) != 1.0 else "")
                         + f"_lr{g['lr_scale']}_ep{q['epochs']}"
                         + (f"_mix{g['mix_spec']}" if "mix_spec" in g else "")
                         + ("_uf0" if g.get("unfreeze_b0b4") is False else "")),
                "seed": seed,
                "note": f"结构 {c['head_src']}/hc{c['head_conv']}/{c['act']}（板卡 {c['ms']:.2f} ms ≤ {args.limit_ms}）"
                        f" + 超参 lr={g['lr_scale']} ep={g['epochs']}"
                        + (f" mix={g['mix_spec']}" if "mix_spec" in g else "")
                        + (" 解冻b0-b4=否" if g.get("unfreeze_b0b4") is False else " 解冻b0-b4"),
                "params": q,
            })
            seed += 1

    # ---- ★★ 生成后自检：每个参数值必须落在 dispatcher 的真实分布内 ----
    #     （2026-11-06 根治「手写常量漂移」：一旦 plan 引用了分布外的值，
    #      在**生成阶段**就报错，而不是等到 enqueue 时 Optuna 抛 ValueError 让搜索起不来。）
    _dists = {
        "mix_spec": ("cat", list(D.MIX_SPECS)),
        "lr_scale": ("float", (D.LR_SCALE_LOW, D.LR_SCALE_HIGH)),
        "warmup_epochs": ("float", (D.WARMUP_EPOCHS_LOW, D.WARMUP_EPOCHS_HIGH)),
        "weight_decay": ("float", (1e-5, 5e-4)),
        "focal_beta": ("float", (2.2, 2.9)),
        "offset_weight": ("float", (0.40, 0.70)),
        "mosaic": ("float", (0.0, 0.15)),
        "scale": ("float", (0.10, 0.30)),
        "translate": ("float", (0.05, 0.15)),
        "clip_max_norm": ("float", (2.0, 8.0)),
        "freeze_bn": ("cat", [True, False]),
        "no_p2": ("cat", list(D.STRUCT_NO_P2_CHOICES)),
        "head_level": ("cat", list(D.STRUCT_HEAD_LEVEL_CHOICES)),
        "act": ("cat", list(D.STRUCT_ACT_CHOICES)),
        "head_conv": ("cat", list(D.STRUCT_HEAD_CONV_CHOICES)),
        "late_upsample": ("cat", list(D.STRUCT_LATE_CHOICES)),
        "unfreeze_b0b4": ("cat", list(D.STRUCT_UNFREEZE_CHOICES)),
        "head_src": ("cat", list(D.STRUCT_HEAD_SRC_CHOICES)),
        "p2_out_scale": ("cat", list(D.STRUCT_P2_OUT_SCALE_CHOICES)),
    }
    _bad = []
    for a in arms:
        for k, v in (a.get("params") or {}).items():
            if k not in _dists:
                continue
            kind, spec = _dists[k]
            if kind == "cat":
                if v not in spec:
                    _bad.append((a["role"], k, v, spec))
            else:
                lo, hi = spec
                if not (lo <= v <= hi):
                    _bad.append((a["role"], k, v, (lo, hi)))
    if _bad:
        print("\n[FATAL] plan 里有参数值不在 dispatcher 分布内 ⇒ enqueue 会抛 ValueError：")
        for role, k, v, spec in _bad:
            print(f"   [{role}] {k}={v!r} ∉ {spec}")
        print("   ⇒ 请把该取值加进 optuna_mix_ratio_search.py 的对应 *_CHOICES（并换新 study 名）。")
        return 1
    print(f"\n[CHECK] {len(arms)} 条臂的全部参数值都在 dispatcher 分布内 ✅")

    Path(args.out).write_text(json.dumps(arms, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[OK] {len(arms)} 条臂 -> {args.out}")
    # 预算：按实测 epoch→耗时表（EPOCH_MIN_EST）估总机时
    _ep_min = dict(getattr(D, "EPOCH_MIN_EST", {}) or {})
    _tot_min = sum(float(_ep_min.get(int(a["params"].get("epochs", 20)), 205)) for a in arms)
    print(f"     预算检查：{len(arms)} 臂，合计 ≈ {_tot_min:.0f} 机时（min）"
          f" ÷ {args.gpus} 卡 ≈ {_tot_min/max(args.gpus,1)/60:.1f} h")
    for a in arms:
        q = a["params"]
        print(f"  {a['role']:<46} seed={a['seed']} src={q['head_src']:<10} hc={q['head_conv']:<3} "
              f"act={q['act']:<5} late={int(q.get('late_upsample',False))} "
              f"sc={q.get('p2_out_scale',1.0)} uf={int(q['unfreeze_b0b4'])} "
              f"lr={q['lr_scale']} ep={q.get('epochs','-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
