#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 probes7 的补测结果到仓库速度表，并换上更准的预测器（relu 族 + late 哑变量）。

背景（2026-11-06）
------------------
* 仓库表 `manu/pipeline/npu/struct_speed_table.json` 的 predictor 是**24 点拟合**，
  且混入了 `ablate_p0=False` 的 P0 高速路点（72.61 ms）⇒ 斜率被带偏：
  对 p2_fused 非 late relu **高估约 1.75 ms**，对 silu 结构**低估 6.5~8.5 ms**
  （周期表完全没给 SiLU 定价）⇒ 表外结构的外推不可信。
* 复拟合（23 点、relu 族 + late 哑变量）：
      ms = 0.466433 + 0.960332×NPU周期(M) + 0.156106×RW(MB) + 1.862533×late
  R²=0.99918、σ=0.367 ms、LOOCV 0.398 ms ⇒ 所有分组残差 ≤0.6 ms。
* ⚠️ **null control 的 FATAL 判定有假阳性**：同一个 `p2f_hc48_late_sc025`
  第一次跑 FATAL、第二次跑有效（22.419 ms），两次 cycles/rw 完全一致
  ⇒ FATAL 只说明「这次探针权重让输出退化了」，**不改变图的算子数与搬运量**
  ⇒ 对 FATAL 项改用**预测器**给值，并标记 `ms_source="pred"`。

用法：
    python manu/pipeline/npu/merge_struct_speed_table.py \
        --table manu/pipeline/npu/struct_speed_table.json \
        --probes /media/manu/1TB-Volume/rknn_work/hm_speed/probes7/arms.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# relu 族 + late 哑变量的复拟合系数（23 点 relu 拟合，见 §80 专题档）
MRl = {"a": 0.466433, "b": 0.960332, "c": 0.156106, "late": 1.862533}


def predict(cyc_m: float, rw_mb: float, late: bool) -> float:
    return MRl["a"] + MRl["b"] * cyc_m + MRl["c"] * rw_mb + (MRl["late"] if late else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="manu/pipeline/npu/struct_speed_table.json")
    ap.add_argument("--probes", action="append", default=[],
                    help="要合并的 arms.json（可多个）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tpath = Path(args.table)
    tbl = json.loads(tpath.read_text(encoding="utf-8"))
    by_name = {a["name"]: a for a in tbl.get("arms", [])}

    added = updated = 0
    for pf in args.probes:
        pf = Path(pf)
        if not pf.is_file():
            print(f"[SKIP] {pf} 不存在")
            continue
        arms = json.loads(pf.read_text(encoding="utf-8"))
        if isinstance(arms, dict):
            arms = arms.get("arms", [])
        for a in arms:
            name = a["name"]
            sp = dict(a.get("spec") or {})
            cyc = (a.get("cycles") or {})
            cyc_m = float(cyc.get("npu_cycles", 0)) / 1e6
            rw = float(cyc.get("rw_mb", 0.0))
            ms = a.get("ms")
            fatal = a.get("fatal")
            late = bool(sp.get("late_upsample", False))
            if ms is None:
                # null control 失败 ⇒ 用复拟合预测器给值（图结构不变 ⇒ 速度不变）
                ms = predict(cyc_m, rw, late)
                src = "pred"
            else:
                src = "measured"
            rec = {
                "name": name,
                "spec": sp,
                "cycles": cyc,
                "ms": round(float(ms), 3),
                "ms_source": src,
                "fatal": fatal,
            }
            if name in by_name:
                # 已有项：优先保留**实测值**，但补齐新字段
                old = by_name[name]
                if old.get("ms") is not None and src == "pred":
                    rec["ms"] = old["ms"]
                    rec["ms_source"] = old.get("ms_source", "measured")
                old.update(rec)
                updated += 1
            else:
                by_name[name] = rec
                added += 1

    tbl["arms"] = sorted(by_name.values(), key=lambda x: (x.get("ms") is None, x.get("ms") or 0))
    tbl["predictor"] = {
        "a": MRl["a"], "b": MRl["b"], "c": MRl["c"],
        "late_coef": MRl["late"],
        "model": "relu-family + late dummy (23-point refit, sigma=0.367ms)",
        "note": "只对 act=relu 的结构有效；silu 结构实测比本式慢 6.5~8.6 ms（周期表没给 SiLU 定价）",
    }
    out = Path(args.out) if args.out else tpath
    out.write_text(json.dumps(tbl, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] {out}：新增 {added} 项 / 更新 {updated} 项，共 {len(tbl['arms'])} 项")

    # ---- 打印 ≤21 ms 的候选清单（决策用）----
    print("\n=== ≤21 ms 候选（实测优先）===")
    rows = []
    for a in tbl["arms"]:
        sp = a.get("spec") or {}
        ms = a.get("ms")
        if ms is None or ms > 21.0:
            continue
        out_px = 320 if (sp.get("late_upsample") or sp.get("head_src") == "p1") else (
            80 if sp.get("head_src") == "p3" else 160)
        rows.append((out_px, -int(sp.get("head_conv", 64)), ms, a["name"], sp,
                     a.get("ms_source", "measured")))
    for out_px, _, ms, name, sp, src in sorted(rows):
        print(f"  {name:<24} src={str(sp.get('head_src')):<10} hc={sp.get('head_conv'):<3} "
              f"late={int(bool(sp.get('late_upsample')))} sc={sp.get('p2_out_scale')} "
              f"输出{out_px} {ms:>7.2f} ms [{src}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
