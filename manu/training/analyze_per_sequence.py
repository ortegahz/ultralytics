#!/usr/bin/env python
"""路线 B 微调效果的**逐序列机理归因**。

要回答的核心问题（方案核心假说）：
  「利用真实地表大图抑制城市高频假目标，激活地表暗刚体特征」——到底发生了吗？

做法：对比 Epoch-0（基线）与各 epoch 的 **逐序列** F1 / TP / FP / FN，
按序列在**训练集**里的角色（暗刚体攻坚 / 负样本挖掘 / 白热空域基准 / 盲测难例）
分组，看增益落在哪一类上。

依据：``manu/data/longquan_ft_split.json`` 里各序列的 role/note 字段。

用法::

    python manu/training/analyze_per_sequence.py \\
        --run-dir runs/mixed_ft_route_b --epochs 0 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 序列角色（来源：manu/data/longquan_ft_split.json，val 组的 role/note）
SEQ_ROLE = {
    "VIDEO00009_19700101_001531": ("黄金池 1/7 · 标准空域白热基准", "① 标准空域白热基准"),
    "VIDEO00029_19700101_014053": ("黄金池 2/7", "② 黄金池其余 6 条"),
    "VIDEO00033_19700101_015000": ("黄金池 3/7 · 池内最弱一条", "② 黄金池其余 6 条"),
    "VIDEO00034_19700101_015052": ("黄金池 4/7", "② 黄金池其余 6 条"),
    "VIDEO00006_19700101_000703": ("黄金池 5/7", "② 黄金池其余 6 条"),
    "VIDEO00008_19700101_001307": ("黄金池 6/7", "② 黄金池其余 6 条"),
    "VIDEO00007_19700101_000818": ("黄金池 7/7", "② 黄金池其余 6 条"),
    "VIDEO00007_19700101_001324": ("独立难例盲测 · 全序列绝对留出", "③ 盲测 001324 · 贴地弱目标/真实缺口"),
}

# 训练集序列角色（决定模型「学到了什么」）
TRAIN_ROLE = {
    "VIDEO00008_19700101_002001": "暗刚体攻坚（机身变黑）+ 地表高频负",
    "VIDEO00005_19700101_002959": "核心攻坚（近距离大目标）",
    "VIDEO00011_19700101_001810": "核心攻坚（中值吸收漏检）",
    "VIDEO00028_19700101_013631": "核心攻坚（中心跳变分裂）",
    "VIDEO00030_19700101_014144": "负样本挖掘（持续虚警）",
    "VIDEO00031_19700101_014227": "负样本挖掘（持续虚警）",
    "VIDEO00002_19700101_001415": "负样本挖掘（贴地弱目标/地表噪点）",
    "VIDEO00009_19700101_003002": "负样本挖掘（真实缺口最严重）",
    "VIDEO00010_19700101_001550": "运动补偿（快速位移/云层误检）",
}


def load(run_dir: Path, epoch: int) -> dict:
    g = run_dir / "gate" / f"epoch{epoch}"
    out = {}
    for tag, fn in (("track2", "track2_golden.json"), ("track3", "track3_blind.json")):
        p = g / fn
        if p.is_file():
            out[tag] = json.loads(p.read_text(encoding="utf-8"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="逐序列机理归因")
    ap.add_argument("--run-dir", default="runs/mixed_ft_route_b")
    ap.add_argument("--epochs", default="0,1,2,3,4,5")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    epochs = [int(x) for x in args.epochs.split(",")]
    data = {e: load(run_dir, e) for e in epochs}
    base = data.get(epochs[0], {})

    rows = []
    for tag in ("track2", "track3"):
        if tag not in base:
            continue
        for seq, b in base[tag]["per_sequence"].items():
            role, kind = SEQ_ROLE.get(seq, ("", ""))
            row = {
                "track": tag,
                "seq": seq,
                "role": role,
                "kind": kind,
                "gt": b.get("gt"),
                "base_f1": b.get("f1"),
                "base_tp": b.get("tp"),
                "base_fp": b.get("fp"),
                "base_fn": b.get("fn"),
                "epochs": {},
            }
            for e in epochs[1:]:
                d = data.get(e, {}).get(tag, {}).get("per_sequence", {}).get(seq)
                if d is None:
                    continue
                row["epochs"][e] = {
                    "f1": d.get("f1"),
                    "tp": d.get("tp"),
                    "fp": d.get("fp"),
                    "fn": d.get("fn"),
                    "df1": (d.get("f1") or 0) - (b.get("f1") or 0),
                    "dfp": (d.get("fp") or 0) - (b.get("fp") or 0),
                    "dfn": (d.get("fn") or 0) - (b.get("fn") or 0),
                }
            rows.append(row)

    print("=" * 118)
    print("  逐序列机理归因（基线 = Epoch-0；Δ 为相对基线变化）")
    print("=" * 118)
    for tag, title in (("track2", "Track2 黄金池 7 条"), ("track3", "Track3 盲测 001324")):
        sub = [r for r in rows if r["track"] == tag]
        if not sub:
            continue
        print(f"\n【{title}】")
        head = "  " + f"{'序列':<28}{'角色':<34}" + "".join(f"{'Ep'+str(e)+' ΔF1':>11}{'ΔFP':>7}{'ΔFN':>7}" for e in epochs[1:])
        print(head)
        for r in sub:
            line = f"  {r['seq']:<28}{r['role'][:32]:<34}"
            for e in epochs[1:]:
                v = r["epochs"].get(e)
                if v is None:
                    line += f"{'—':>11}{'—':>7}{'—':>7}"
                else:
                    line += f"{v['df1']:>+11.6f}{v['dfp']:>+7d}{v['dfn']:>+7d}"
            print(line)

    # 分组汇总：空域白热 vs 贴地/难例
    print("\n" + "=" * 118)
    print("  分组汇总（各组 F1 变化均值，ΔFN = 漏检收缩为正）")
    print("=" * 118)
    print(f"  {'分组':<24}" + "".join(f"{'Ep'+str(e)+' ΔF1':>12}{'ΔFN':>9}{'ΔFP':>9}" for e in epochs[1:]))
    groups = {}
    for r in rows:
        groups.setdefault(r["kind"], []).append(r)
    for kind, rs in groups.items():
        line = f"  {kind:<24}"
        for e in epochs[1:]:
            vals = [r["epochs"][e] for r in rs if e in r["epochs"]]
            if not vals:
                line += f"{'—':>12}{'—':>9}{'—':>9}"
                continue
            line += (
                f"{sum(v['df1'] for v in vals)/len(vals):>+12.6f}"
                f"{sum(v['dfn'] for v in vals)/len(vals):>+9.1f}"
                f"{sum(v['dfp'] for v in vals)/len(vals):>+9.1f}"
            )
        print(line)

    print("\n  训练集序列角色（模型学到了什么）：")
    for s, r in TRAIN_ROLE.items():
        print(f"    {s:<28}{r}")

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
