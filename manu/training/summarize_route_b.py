#!/usr/bin/env python
"""汇总路线 B 混训的**三轨效果对比表**（Epoch-0 基线 + 各 epoch）。

数据源：
  runs/mixed_ft_route_b/gate/gate_history.jsonl   —— 每个 epoch 的门禁报告
  runs/mixed_ft_route_b/logs/train_log.csv         —— 训练 loss / lr 曲线

输出：Markdown 对比表 + CSV。

用法::

    python manu/training/summarize_route_b.py \\
        --run-dir runs/mixed_ft_route_b --out-md runs/mixed_ft_route_b/route_b_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_history(run_dir: Path) -> list[dict]:
    p = run_dir / "gate" / "gate_history.jsonl"
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def train_curve(run_dir: Path) -> list[dict]:
    p = run_dir / "logs" / "train_log.csv"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def epoch_avg_loss(curve: list[dict]) -> dict[int, dict]:
    agg: dict[int, dict] = {}
    acc: dict[int, dict] = {}
    for r in curve:
        try:
            ep = int(r["epoch"])
            it = int(r["iter"])
            last = int(r["iters_per_epoch"]) if "iters_per_epoch" in r else None
        except (ValueError, KeyError):
            continue
        a = acc.setdefault(ep, {"n": 0, "loss": 0.0, "hm": 0.0, "off": 0.0, "lq": 0, "uav": 0})
        a["n"] += 1
        a["loss"] += float(r["loss_total"])
        a["hm"] += float(r["loss_hm"])
        a["off"] += float(r["loss_offset"])
        a["lq"] += int(r.get("batch_lq", 0) or 0)
        a["uav"] += int(r.get("batch_uav", 0) or 0)
    for ep, a in acc.items():
        n = max(a["n"], 1)
        agg[ep] = {
            "loss": a["loss"] / n,
            "hm": a["hm"] / n,
            "off": a["off"] / n,
            "mix_lq": a["lq"] / n,
            "mix_uav": a["uav"] / n,
        }
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description="路线 B 三轨对比表")
    ap.add_argument("--run-dir", default="runs/mixed_ft_route_b")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    history = load_history(run_dir)
    curve = train_curve(run_dir)
    losses = epoch_avg_loss(curve)

    rows = []
    for rec in history:
        ep = rec.get("epoch")
        tracks = rec.get("tracks", {})
        t1 = tracks.get("track1_uav", {})
        t2 = tracks.get("track2_golden", {})
        t3 = tracks.get("track3_blind", {})
        f1_arm = t1.get("finetune_peak", {})
        f1_base = t1.get("baseline_peak", {})
        row = {
            "epoch": ep,
            "verdict": rec.get("verdict"),
            "t1_f1": f1_arm.get("f1"),
            "t1_tp": f1_arm.get("tp"),
            "t1_fp": f1_arm.get("fp"),
            "t1_base_f1": f1_base.get("f1"),
            "t2_f1": t2.get("f1"),
            "t2_tp": t2.get("tp"),
            "t2_fp": t2.get("fp"),
            "t2_fn": t2.get("fn"),
            "t3_f1": t3.get("f1"),
            "t3_tp": t3.get("tp"),
            "t3_fp": t3.get("fp"),
            "t3_fn": t3.get("fn"),
            "loss": (losses.get(ep, {}) or {}).get("loss"),
            "hm": (losses.get(ep, {}) or {}).get("hm"),
            "off": (losses.get(ep, {}) or {}).get("off"),
            "mix_lq": (losses.get(ep, {}) or {}).get("mix_lq"),
            "mix_uav": (losses.get(ep, {}) or {}).get("mix_uav"),
        }
        rows.append(row)

    def f(v, nd=6):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    def d(v, nd=6):
        if v is None:
            return "—"
        return f"{v:+.{nd}f}"

    md: list[str] = []
    md.append("# 路线 B 全帧对等混训 —— 三轨效果对比表\n")
    md.append("基线（Epoch-0 = 原始 trial_0474 权重，去 P0 口径）\n")
    md.append("| 轨道 | F1 | TP | FP | FN | 判据 |")
    md.append("|:---|--:|--:|--:|--:|:---|")
    md.append("| Track1 Anti-UAV 官方大盘 | 0.905736 | 21,624 | 1,014 | — | ≥ 基线 |")
    md.append("| Track2 龙泉山黄金池 7 条 | 0.988436 | 16,283 | 26 | 355 | F1≥0.985 且 FP≤36（一票否决）|")
    md.append("| Track3 盲测 001324 | 0.837331 | 8,012 | 590 | 2,523 | 观察轨 |")
    md.append("")
    md.append("## 各 Epoch 结果\n")
    md.append(
        "| Epoch | 门禁 | Track1 F1 | ΔF1 | TP | FP | Track2 F1 | ΔF1 | TP | FP | FN | "
        "Track3 F1 | ΔF1 | TP | FP | FN | train loss (hm/off) | batch 混样 LQ/UAV |"
    )
    md.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|--:|")
    for r in rows:
        md.append(
            f"| {r['epoch']} | {r['verdict']} | {f(r['t1_f1'])} | {d((r['t1_f1'] or 0) - (r['t1_base_f1'] or 0))} | "
            f"{f(r['t1_tp'])} | {f(r['t1_fp'])} | {f(r['t2_f1'])} | {d((r['t2_f1'] or 0) - 0.9884359729262149)} | "
            f"{f(r['t2_tp'])} | {f(r['t2_fp'])} | {f(r['t2_fn'])} | "
            f"{f(r['t3_f1'])} | {d((r['t3_f1'] or 0) - 0.8373308251032032)} | "
            f"{f(r['t3_tp'])} | {f(r['t3_fp'])} | {f(r['t3_fn'])} | "
            f"{f(r['loss'], 4)} ({f(r['hm'], 4)}/{f(r['off'], 4)}) | "
            f"{f(r['mix_lq'], 2)}/{f(r['mix_uav'], 2)} |"
        )
    md.append("")
    md.append(
        "> 注：Track1 口径 = 8px 点对点（tol 生效）；Track2/3 口径 = point-in-bbox（tol 不生效）。\n"
        "> 三轨 F1 **不可横向比较**，只做轨道内纵向对比（基线 vs 各 epoch）。\n"
        "> 训练 loss 为每 50 iters 打点日志的 epoch 内均值（**非** epoch 级验证 loss）。"
    )

    text = "\n".join(md)
    print(text)
    if args.out_md:
        p = Path(args.out_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"\nMarkdown 已写入: {p}")
    if args.out_csv:
        p = Path(args.out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            with open(p, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"CSV 已写入: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
