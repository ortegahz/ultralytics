#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# ⚠️⚠️ **已废弃（2026-11-06 18:30）** —— 请改用 **`build_round2_plan_v2.py`** ⚠️⚠️
#
# 为什么废弃：
#   本脚本生成的家族是 `hc32/hc24 + p2_fused`（82 臂版），但实测证明：
#   · `p2_out_scale=0.25`（sc025 家族）**4/4 全灭**（neck 收窄 4 倍 ⇒ 训不动）；
#   · `hc24` 不在 Optuna 的 head_conv choices 里 ⇒ 用它生成计划会直接报
#     `ValueError: '24' not in (64, 48, 40, 32)`；
#   · 主线已改为 **`p3/hc64/relu/late`（13.131 ms）** 与 `p2f/hc32/relu/late/sc1.0`（20.829 ms），
#     由 `build_round2_plan_v2.py` 生成（6 诊断臂 + F_MAIN 48 + F_ALT 30，交错排布 + role 防撞名）。
#   · 另：`silu 权重喂 relu` 会让头大量神经元死亡 ⇒ loader 已加规则「act 变了就重置头」
#     （见 `manu/data/preannotate_hm_bbox.py::merge_channels` 附近）。
#
# ⇒ **保留本文件仅为历史留档，不要用它生成计划。**
# ==============================================================================
# -*- coding: utf-8 -*-
"""生成「第二轮」臂计划（把 GPU 从现在一直喂到 deadline）—— 2026-11-06。

为什么不用 `build_seed_arms.py`
---------------------------
`build_seed_arms.py` 的 TARGETS 写死 3 个结构、每结构 2 个种子（共 6 臂）⇒
一轮只够跑 ~5 h ⇒ **周六白天 GPU 会空转**。本脚本一次生成 ~68 臂，
覆盖「4 个 320 输出家族（凑 soup）+ 可 informative 消融 + 快结构单臂取证」，
让调度器的队列**永远不空**（队列耗尽 = GPU 空转 = 浪费服务器）。

臂的排布（★ 顺序即优先级：FIFO 入队，先排的先跑）
--------------------------------------------
1. `informative`（20 臂）：非 320 输出结构（p2f hc32/hc40/hc48/sc075、p3）
   的 uf=True / uf=False 单臂 ⇒ 回答「更快的结构能不能守住 ΔT1 ≥ −0.5 pp」。
   这些是**单臂证据**，不参与 soup ⇒ 放最前面，保证它们一定跑完。
2. `bulk`（48 臂）：4 个 320 输出家族各 12 个种子 ⇒ n≈13 的 soup。
   ★ **每个家族先排 3 个**（保证任何截断下每个家族都 ≥3 个成员），
   再排剩下 9 个。

家族（结构 → 板卡实测速度，来自 `struct_speed_table.json`）
--------------------------------------------------
  L4 = p2_fused hc24 late sc0.5  20.015 ms（第二快，且第一轮**没跑过**）
  L1 = p2_fused hc32 late sc0.25 19.827 ms（最快）
  L2 = p2_fused hc32 late sc0.5  20.057 ms
  L3 = p2_fused hc32 late sc1.0  20.829 ms（最慢，余量最小）

用法（服务器，conda uav）：
    python manu/training/build_round2_plan.py \
        --speed-table manu/pipeline/npu/struct_speed_table.json \
        --out /mnt/data/siping/runs/npu20ms_seed/seed_arms.json \
        --seed-base 3000 --seeds-per-struct 12 --limit-ms 21.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "manu" / "training"))
try:
    import optuna_mix_ratio_search as D  # noqa: F401
except Exception as _exc:  # pragma: no cover
    raise SystemExit(f"[FATAL] 导入 dispatcher 常量失败：{_exc}") from _exc

# 母配方（与 cs_t0043 同配方；与第一轮 / build_seed_arms.py 完全一致）
BASE = {
    "mix_spec": "w2.0", "lr_scale": 3.5, "warmup_epochs": 1.5, "weight_decay": 0.000269,
    "focal_beta": 2.75, "offset_weight": 0.40, "mosaic": 0.15, "scale": 0.10,
    "translate": 0.05, "clip_max_norm": 5.65, "freeze_bn": False,
}

# (tag, head_conv, late_upsample, p2_out_scale, 说明)
LATE_FAMILIES = [
    ("L4", 24, True, 0.5, "20.02ms+320输出+neck收窄0.5"),
    ("L1", 32, True, 0.25, "19.83ms+320输出+最快"),
    ("L2", 32, True, 0.5, "20.06ms+320输出+neck收窄0.5"),
    ("L3", 32, True, 1.0, "20.83ms+320输出+neck不收窄"),
]

# 非 320 输出结构（head_src, head_conv, head_level, no_p2, p2_out_scale, act, 说明）
FAST_STRUCTS = [
    ("p2f_hc32", "p2_fused", 32, "p2_fused", False, 1.0, "relu", "17.97ms"),
    ("p2f_hc32_sc075", "p2_fused", 32, "p2_fused", False, 0.75, "relu", "18.44ms"),
    ("p2f_hc40", "p2_fused", 40, "p2_fused", False, 1.0, "relu", "20.61ms"),
    ("p2f_hc48", "p2_fused", 48, "p2_fused", False, 1.0, "relu", "20.60ms"),
    ("p3_hc32", "p3", 32, "p3", True, 1.0, "relu", "9.38ms"),
]


def _late_params(hc: int, late: bool, sc: float, *, epochs: int = 20,
                 unfreeze: bool = True, **over) -> dict:
    q = dict(BASE)
    q.update({
        "no_p2": False, "head_level": "p2_fused", "act": "relu", "head_conv": hc,
        "late_upsample": late, "p2_out_scale": float(sc),
        # ⚠️ uf=True 才训得动（uf=False 在 p2_fused/late 新头上 loss 停在 1.44~1.63 ⇒ 训不动）
        "unfreeze_b0b4": bool(unfreeze), "head_src": "p2_fused", "epochs": int(epochs),
    })
    q.update(over)
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-base", type=int, default=3000)
    ap.add_argument("--seeds-per-struct", type=int, default=12)
    ap.add_argument("--limit-ms", type=float, default=21.0)
    args = ap.parse_args()

    tbl = json.loads(Path(args.speed_table).read_text(encoding="utf-8"))
    # 结构 → 实测 ms（ms_source=measured 优先）
    speed: dict[tuple, float] = {}
    for a in tbl.get("arms", []):
        ms = a.get("ms")
        if ms is None or ms > args.limit_ms:
            continue
        sp = a.get("spec") or {}
        if not bool(sp.get("ablate_p0", True)):
            continue
        k = (sp.get("head_src"), int(sp.get("head_conv", 64)), sp.get("act", "relu"),
             bool(sp.get("late_upsample", False)),
             round(float(sp.get("p2_out_scale", 1.0) or 1.0), 3))
        prev = speed.get(k)
        if prev is None or (a.get("ms_source") != "measured" and prev[1] != "measured"):
            speed[k] = (float(ms), a.get("ms_source", "measured"))

    arms: list[dict] = []
    seed = args.seed_base

    # ---------- 1) informative：快结构单臂（uf=True / uf=False）----------
    for tag, hsrc, hc, hlevel, no_p2, sc, act, note in FAST_STRUCTS:
        for unfreeze in (True, False):
            q = dict(BASE)
            q.update({
                "no_p2": no_p2, "head_level": hlevel, "act": act, "head_conv": hc,
                "late_upsample": False, "p2_out_scale": float(sc),
                "unfreeze_b0b4": unfreeze, "head_src": hsrc, "epochs": 20,
            })
            hit = speed.get((hsrc, hc, act, False, round(float(sc), 3)))
            ms_txt = f"{hit[0]:.2f}ms" if hit else "表外"
            arms.append({
                "role": f"INFO_{tag}_uf{int(unfreeze)}_s{seed}",
                "seed": seed,
                "note": (f"★快结构单臂取证 {note} | uf={unfreeze} | {ms_txt}"
                         f" | 判『非320输出能否守住 ΔT1≥-0.5pp』"),
                "params": q,
            })
            seed += 1

    # ---------- 2) 320 输出家族：每家族先 3 个（保底 n≥3），再交错补齐 ----------
    # ★ 交错排布的原因：dispatcher 是 **FIFO** 取臂 ⇒ 若按家族连排，
    #   队尾那个家族会在截断时**一个都没跑到**（n=0 ⇒ 家族消失）。
    #   ⇒ 交错后，任何截断点下每个家族都至少有 3 个成员（n≥3 ⇒ 能出 soup）。
    fam_specs = []
    for tag, hc, late, sc, note in LATE_FAMILIES:
        k = ("p2_fused", hc, "relu", late, round(float(sc), 3))
        hit = speed.get(k)
        if hit is None:
            print(f"[SKIP] {tag} 结构 {k} 不在 ≤{args.limit_ms} ms 的速度表里 ⇒ 不补种子")
            continue
        fam_specs.append((tag, hc, late, sc, note, hit))

    per = max(3, args.seeds_per_struct)
    for i in range(per):
        for tag, hc, late, sc, note, hit in fam_specs:
            q = _late_params(hc, late, sc)
            arms.append({
                "role": f"SEED_{tag}_hc{hc}_late{int(late)}_sc{int(round(sc*100))}_s{seed}",
                "seed": seed,
                "note": (f"★320输出家族 {tag} 种子 | 板卡 {hit[0]:.2f} ms（{hit[1]}）"
                         f" | {note} | 同配方不同种子 ⇒ 凑 soup"),
                "params": q,
            })
            seed += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(arms, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 自检：角色唯一 / 家族分布 / 结构是否都在速度表里 ----
    roles = [a["role"] for a in arms]
    assert len(roles) == len(set(roles)), "role 有重复！"
    seeds = [a["seed"] for a in arms]
    assert len(seeds) == len(set(seeds)), "seed 有重复！"
    from collections import Counter
    fam = Counter(a["role"].split("_")[1] for a in arms if a["role"].startswith("SEED_"))
    print(f"[OK] 写出 {len(arms)} 臂 -> {out}")
    print(f"     家族分布：{dict(fam)}")
    print(f"     informative：{sum(1 for a in arms if a['role'].startswith('INFO_'))} 臂")
    print(f"     seed 区间：{min(seeds)} ~ {max(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
