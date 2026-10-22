#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交叉校验 train 拟合度诊断的全部数字，并重算三个诊断信号的判定。

为什么需要这个脚本
------------------
`train_fit_diagnostic.py` 产出 `train_fit_diagnostic.json`（汇总）与两份 `train_eval.log`（逐序列原始行）。
汇总表是**二次加工**的结果，验收不能只看汇总。本脚本用**两个独立来源**互校：

  源 A = `train_fit_diagnostic.json` 的 `summary` 块
  源 B = `train_eval.log` 里的 `[SEQ] ...` 原始行 + `[SUCCESS]` 汇总

逐条比对 F1/TP/FP/FN/Recall/Precision，任何一条不一致即 `exit 1`；
一致后再**重算**三个诊断信号的判定（欠拟合 / 过拟合 / 顽固误检）与泛化率，
确保结论不是从表格里"看"出来的，而是从原始行算出来的。

用法（执行机：服务器，根目录 `/tmp/pycharm_project_10ae9e2e`）：

```bash
python manu/training/verify_train_fit_report.py \
  --report runs/train_fit_diag_s5/train_fit_diagnostic.json \
  --base-log runs/train_fit_diag_s5/baseline/train_eval.log \
  --ft-log runs/train_fit_diag_s5/ft/train_eval.log
```
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

SEQ_RE = re.compile(
    r"\[SEQ\]\s+(?P<seq>\S+)\s+frames=(?P<frames>\d+)\s+GT=(?P<gt>\d+)\s+"
    r"TP=(?P<tp>\d+)\s+FP=(?P<fp>\d+)\s+FN=(?P<fn>\d+)\s+"
    r"R=(?P<r>[\d.]+)%\s+P=(?P<p>[\d.]+)%\s+F1=(?P<f1>[\d.]+)%"
)


def parse_log(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = SEQ_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        out[d["seq"]] = {
            "frames": int(d["frames"]),
            "gt": int(d["gt"]),
            "tp": int(d["tp"]),
            "fp": int(d["fp"]),
            "fn": int(d["fn"]),
            "recall": float(d["r"]) / 100.0,
            "precision": float(d["p"]) / 100.0,
            "f1": float(d["f1"]) / 100.0,
        }
    if not out:
        raise SystemExit(f"[FATAL] 日志里没有 [SEQ] 行：{path}")
    return out


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    ap = argparse.ArgumentParser(description="交叉校验 train 拟合度诊断数字 + 重算诊断判定")
    ap.add_argument("--report", required=True, help="train_fit_diagnostic.json")
    ap.add_argument("--base-log", required=True, help="baseline 的 train_eval.log")
    ap.add_argument("--ft-log", required=True, help="微调权重的 train_eval.log")
    ap.add_argument("--blind-f1-base", type=float, default=0.8373308251032032, help="盲测 001324 基线 F1")
    ap.add_argument("--blind-f1-ft", type=float, default=0.8749235474006115, help="盲测 001324 微调 F1")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    summary = report["summary"]
    base_log = parse_log(Path(args.base_log))
    ft_log = parse_log(Path(args.ft_log))

    errors: list[str] = []

    # ---------- 1. 汇总块 vs 日志逐序列 ----------
    table = {r["sequence"]: r for r in summary["per_sequence_table"]}
    if set(table) != set(base_log) or set(table) != set(ft_log):
        errors.append(f"序列集合不一致：report={sorted(table)} base={sorted(base_log)} ft={sorted(ft_log)}")
    for seq, row in table.items():
        for tag, log in (("base", base_log), ("ft", ft_log)):
            src = log.get(seq)
            if src is None:
                errors.append(f"{seq}: {tag} 日志缺该序列")
                continue
            for key in ("tp", "fp", "fn"):
                if row[f"{tag}_{key}"] != src[key]:
                    errors.append(f"{seq}.{tag}.{key}: report={row[f'{tag}_{key}']} log={src[key]}")
            # gt 在报告里是共享列（两臂同一套真值）
            if row.get("gt") != src["gt"]:
                errors.append(f"{seq}.{tag}.gt: report={row.get('gt')} log={src['gt']}")
            # 报告存全精度；日志里的 R/P/F1 是**打印值**（R/P 两位小数百分比、F1 三位），
            # 所以用日志的整数 TP/FP/FN **重算**后做精确比对，打印值只做宽松比对。
            rec = src["tp"] / (src["tp"] + src["fn"]) if src["tp"] + src["fn"] else 0.0
            pre = src["tp"] / (src["tp"] + src["fp"]) if src["tp"] + src["fp"] else 0.0
            f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
            for key, recomputed in (("f1", f1), ("recall", rec), ("precision", pre)):
                if not close(float(row[f"{tag}_{key}"]), recomputed, 1e-9):
                    errors.append(
                        f"{seq}.{tag}.{key}: report={row[f'{tag}_{key}']} recomputed_from_ints={recomputed}"
                    )
                if not close(float(row[f"{tag}_{key}"]), src[key], 1e-4):
                    errors.append(
                        f"{seq}.{tag}.{key}: report={row[f'{tag}_{key}']} printed_log={src[key]} (打印精度差)"
                    )

    # ---------- 2. 汇总块的整体值 vs 日志重算 ----------
    def agg(log: dict[str, dict]) -> dict:
        tp = sum(v["tp"] for v in log.values())
        fp = sum(v["fp"] for v in log.values())
        fn = sum(v["fn"] for v in log.values())
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "recall": rec, "precision": pre, "f1": f1}

    b_agg, f_agg = agg(base_log), agg(ft_log)
    for tag, log in (("base", base_log), ("ft", ft_log)):
        a = b_agg if tag == "base" else f_agg
        r = summary["overall"][tag]
        for key in ("tp", "fp", "fn"):
            if r[key] != a[key]:
                errors.append(f"overall.{tag}.{key}: report={r[key]} recomputed={a[key]}")
        for key in ("f1", "recall", "precision"):
            if not close(float(r[key]), a[key], 1e-9):
                errors.append(f"overall.{tag}.{key}: report={r[key]} recomputed={a[key]}")
    # delta 恒为 ft − base
    for key in ("tp", "fp", "fn"):
        if summary["overall"]["delta"][key] != f_agg[key] - b_agg[key]:
            errors.append(
                f"overall.delta.{key}: report={summary['overall']['delta'][key]} "
                f"recomputed={f_agg[key] - b_agg[key]}"
            )
    for key in ("f1", "recall", "precision"):
        if not close(float(summary["overall"]["delta"][key]), f_agg[key] - b_agg[key], 1e-9):
            errors.append(
                f"overall.delta.{key}: report={summary['overall']['delta'][key]} "
                f"recomputed={f_agg[key] - b_agg[key]}"
            )

    # ---------- 3. 重算三个诊断信号 ----------
    b_all, f_all = b_agg, f_agg
    print("=" * 78)
    print("[1] 训练集整体")
    print(f"    底模  F1={b_all['f1']:.6f} R={b_all['recall']*100:.2f}% P={b_all['precision']*100:.2f}% "
          f"TP={b_all['tp']} FP={b_all['fp']} FN={b_all['fn']}")
    print(f"    Ep3   F1={f_all['f1']:.6f} R={f_all['recall']*100:.2f}% P={f_all['precision']*100:.2f}% "
          f"TP={f_all['tp']} FP={f_all['fp']} FN={f_all['fn']}")

    print("\n[2] 信号① 欠拟合检查")
    rec_ft = f_all["recall"]
    print(f"    train Recall {b_all['recall']*100:.2f}% -> {rec_ft*100:.2f}%  (健康带 92%~96%)"
          f" => {'健康' if 0.92 <= rec_ft <= 0.96 else ('偏高(容量有余)' if rec_ft > 0.96 else '欠拟合')}")
    for seq, tag in (("VIDEO00008_19700101_002001", "002001 下落变暗段"),
                     ("VIDEO00009_19700101_003002", "003002 贴地弱目标")):
        if seq not in base_log or seq not in ft_log:
            print(f"    {seq}: 日志缺失，跳过")
            continue
        bb, ff = base_log[seq], ft_log[seq]
        cut = (ff["fn"] - bb["fn"]) / bb["fn"] if bb["fn"] else 0.0
        print(f"    {tag}: FN {bb['fn']} -> {ff['fn']} ({cut:+.1%}) | R {bb['recall']*100:.2f}% -> {ff['recall']*100:.2f}% "
              f"| F1 {bb['f1']*100:.3f}% -> {ff['f1']*100:.3f}%")

    print("\n[3] 信号② 过拟合检查")
    print(f"    train F1 {f_all['f1']*100:.3f}%  (危险带 ≥99.5% / 健康带 95%~97%) => "
          f"{'接近过拟合' if f_all['f1'] >= 0.995 else ('健康，泛化余量充足' if 0.95 <= f_all['f1'] <= 0.97 else '偏高但未到危险带')}")
    print(f"    train FP {f_all['fp']}（是否趋近 0：{'是' if f_all['fp'] <= 20 else '否'}）")
    zero_fp = [s for s, v in ft_log.items() if v["fp"] == 0]
    print(f"    Ep3 上 FP 压到 0 的序列：{len(zero_fp)}/{len(ft_log)} 条 {sorted(x[-6:] for x in zero_fp)}")

    print("\n[4] 信号③ 顽固误检定位（train 内 ΔFP）")
    ups = []
    for seq in sorted(base_log):
        d = ft_log[seq]["fp"] - base_log[seq]["fp"]
        flag = "  <== 反弹" if d > 0 else ""
        if d > 0:
            ups.append((seq, d))
        print(f"    {seq[-6:]}: FP {base_log[seq]['fp']:3d} -> {ft_log[seq]['fp']:3d}  Δ={d:+d}{flag}")
    print(f"    => train 内误报上升的序列：{[(s[-6:], d) for s, d in ups] or '无'}")

    print("\n[5] 拟合度 -> 泛化度 转化率")
    dt = f_all["f1"] - b_all["f1"]
    db = args.blind_f1_ft - args.blind_f1_base
    print(f"    ΔF1  train {dt:+.6f} / blind {db:+.6f}  => 泛化率 {db/dt:.1%}" if dt else "    ΔF1=0，无法计算")
    drt = f_all["recall"] - b_all["recall"]
    print(f"    ΔRecall train {drt:+.6f}（盲测 ΔRecall 需另取轨道日志，本脚本不臆算）")

    print("\n" + "=" * 78)
    if errors:
        print(f"[FAIL] 交叉校验发现 {len(errors)} 处不一致：")
        for e in errors[:20]:
            print("   -", e)
        return 1
    print("[PASS] 汇总表与两份原始日志逐字段一致；三个诊断信号已由原始行重算")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
