#!/usr/bin/env python3
"""从 no-GMC 搜索汇总 CSV 生成**复现计划**（JSON）。

用途
----
周末实验收尾时，把「配置级多种子榜」前 N 个配置**各补到 3 个种子**。
交接口径必须是**同超参多种子中位**，不是单 trial 最高 U（见 memory §12.6.2 / 专题档 §15.29）。

判定规则（与汇总器 §4.3.1 同一条思路，但此处只负责**挑配置**，不做最终判定）：
  1. 按 15 维调参逐位相同分组（同配置 = 同一条训练配方）；
  2. 组内 U 取中位（只有 U 非空的 trial 参与）；
  3. 取 U 中位最高的 N 个配置；
  4. 每个配置补齐到 --seeds 个种子（已有种子不重复投）。

用法：
  python manu/training/plan_replication.py --out-root <OUT> --top 3 --seeds 3 \
      --out-json <OUT>/replication_plan.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

CFG_COLS = ("mix_spec", "lr_scale", "warmup_epochs", "weight_decay", "focal_beta",
            "offset_weight", "mosaic", "scale", "translate", "clip_max_norm", "freeze_bn",
            "init_from", "epochs", "backbone_lr_scale", "freeze_backbone_epochs")
# 需要原样回填到入队参数里的列（其余从 CSV 读出的都是字符串 ⇒ 必须转回数值）
FLOAT_COLS = {"lr_scale", "warmup_epochs", "weight_decay", "focal_beta", "offset_weight",
              "mosaic", "scale", "translate", "clip_max_norm", "backbone_lr_scale"}
INT_COLS = {"epochs", "freeze_backbone_epochs"}


def _num(v: str, field: str):
    s = (v or "").strip()
    if s == "":
        return None
    if field in INT_COLS:
        try:
            return int(float(s))
        except Exception:
            return None
    if field in FLOAT_COLS:
        try:
            return float(s)
        except Exception:
            return None
    if field == "freeze_bn":
        return s.strip().lower() in ("true", "1", "yes")
    return s


def _cfg_key(row: dict) -> tuple:
    out = []
    for c in CFG_COLS:
        v = row.get(c, "")
        s = str(v).strip()
        if c in ("mix_spec", "init_from"):
            out.append(s)
        elif c in INT_COLS:
            try:
                out.append(int(float(s)))
            except Exception:
                out.append(s)
        elif c == "freeze_bn":
            out.append(s.lower() in ("true", "1", "yes"))
        else:
            try:
                out.append(round(float(s), 6))
            except Exception:
                out.append(s)
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 no-GMC 复现计划（多种子）")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--top", type=int, default=3, help="取 U 中位最高的几个配置")
    ap.add_argument("--seeds", type=int, default=3, help="每个配置补齐到几个种子")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--seed-base", type=int, default=900, help="新种子的起始编号")
    args = ap.parse_args()

    csv_path = Path(args.out_root) / "optuna_summary.csv"
    if not csv_path.is_file():
        print(f"[FAIL] 缺 {csv_path}")
        Path(args.out_json).write_text("[]\n", encoding="utf-8")
        return 1

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if (r.get("status") or "").strip().upper() not in ("SUCCESS",):
            continue  # 只用成功过线的 trial 参与排名
        # ⚠️ **2026-10-30 修正：必须按「严格口径」挑复现对象**。
        #    memory 铁律：「**严格通道为上位门槛；放宽通道仅作诊断标记**」。
        #    本脚本原先只看 status==SUCCESS ⇒ **把 PASS_RELAXED 也当成可上线**，
        #    于是 2026-11-01 那轮复现挑中了两个「放宽口径」配置，
        #    而真正严格过线的配置全是 n=1 ⇒ 复现跑错了对象（该复现的没复现）。
        #    ⇒ 复现候选必须 **verdict 严格等于 PASS**。
        if (r.get("gate_verdict") or "").strip() != "PASS":
            continue
        groups.setdefault(_cfg_key(r), []).append(r)

    def u_of(r: dict):
        try:
            return float(r.get("utility") or "")
        except Exception:
            return None

    scored = []
    for key, rs in groups.items():
        us = [u for u in (u_of(r) for r in rs) if u is not None]
        if not us:
            continue
        scored.append((statistics.median(us), key, rs))
    scored.sort(key=lambda x: -x[0])

    plan = []
    next_seed = args.seed_base
    for rank, (umed, key, rs) in enumerate(scored[: args.top], start=1):
        have = len(rs)
        need = max(0, args.seeds - have)
        if need == 0:
            print(f"[SKIP] 配置 #{rank} 已有 {have} 个种子 ≥ {args.seeds}")
            continue
        first = rs[0]
        params = {}
        for c in CFG_COLS:
            params[c] = _num(first.get(c, ""), c)
        for k in ("epochs", "freeze_backbone_epochs", "backbone_lr_scale"):
            if params.get(k) is None:
                params.pop(k, None)
        for _ in range(need):
            plan.append({
                "role": f"REPRO_R{rank}_SEED{next_seed}",
                "seed": next_seed,
                "note": (f"复现：配置 U 中位 {umed:+.6f}（已有 {have} 种子，补到 {args.seeds}）"
                         f" init={params.get('init_from')} ep={params.get('epochs')} "
                         f"lr={params.get('lr_scale')}"),
                "params": params,
            })
            next_seed += 1
        print(f"[PLAN] 配置 #{rank}: U中位 {umed:+.6f} | 已有 {have} 种子 | 补投 {need} 条")

    Path(args.out_json).write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] 写出 {args.out_json}（{len(plan)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
