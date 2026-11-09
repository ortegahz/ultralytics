#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动前预检：plan 里每个参数值是否落在 dispatcher 对应 suggest_* 的分布内。

⚠️ **为什么必须有它**（2026-11-06 实测连踩三次）：
Optuna 的 `enqueue_trial` 会校验「固定参数值是否在已有分布里」，
不在 ⇒ 直接 `ValueError: 'p1' not in ('p3','p2')` ⇒ **整个搜索起不来**。
而这个错只在**真正入队时**才暴露 ⇒ 每修一个就要重试一次启动。

本脚本把「plan 的取值集合」与「dispatcher 的 choices/范围」逐项对照，
**在启动前**把不一致全部列出来 ⇒ 一次改对。

用法：
    python manu/training/check_plan_distributions.py --plan manu/training/npu20ms_plan.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "manu" / "training"))

import optuna_mix_ratio_search as D  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    args = ap.parse_args()
    arms = json.loads(Path(args.plan).read_text(encoding="utf-8"))

    # name -> (choices 或 (low, high) 或 step)
    dists = {
        "mix_spec": ("cat", D.MIX_SPECS),
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
        "no_p2": ("cat", D.STRUCT_NO_P2_CHOICES),
        "head_level": ("cat", D.STRUCT_HEAD_LEVEL_CHOICES),
        "act": ("cat", D.STRUCT_ACT_CHOICES),
        "head_conv": ("cat", D.STRUCT_HEAD_CONV_CHOICES),
        "late_upsample": ("cat", D.STRUCT_LATE_CHOICES),
        "unfreeze_b0b4": ("cat", D.STRUCT_UNFREEZE_CHOICES),
        "head_src": ("cat", D.STRUCT_HEAD_SRC_CHOICES),
        "p2_out_scale": ("cat", D.STRUCT_P2_OUT_SCALE_CHOICES),
    }
    used = {}
    for a in arms:
        for k, v in (a.get("params") or {}).items():
            used.setdefault(k, set()).add(v)

    bad = []
    for k, vals in sorted(used.items()):
        if k not in dists:
            print(f"  ~ {k}: plan 用 {sorted(vals, key=str)}（无分布约束，随 fixed_params 走）")
            continue
        kind, spec = dists[k]
        if kind == "cat":
            illegal = [v for v in vals if v not in spec]
            ok = "OK " if not illegal else "BAD"
            print(f"  {ok} {k}: plan {sorted(vals, key=str)} | 分布 {spec}")
            if illegal:
                bad.append((k, illegal, spec))
        else:
            lo, hi = spec
            illegal = [v for v in vals if not (lo <= v <= hi)]
            ok = "OK " if not illegal else "BAD"
            print(f"  {ok} {k}: plan {sorted(vals)} | 范围 [{lo}, {hi}]")
            if illegal:
                bad.append((k, illegal, (lo, hi)))

    if bad:
        print("\n[FATAL] 以下参数值不在分布内 ⇒ enqueue 会抛 ValueError ⇒ 搜索起不来：")
        for k, illegal, spec in bad:
            print(f"   {k}: 非法值 {illegal}，分布 {spec}")
        return 1
    print("\n[OK] plan 的全部参数值都在分布内 ⇒ 可以安全入队")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
