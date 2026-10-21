#!/usr/bin/env python
"""对比龙泉山单帧 baseline 的两套特征语义（FPV 仿射连乘 vs official 直接拟合）。

输入是 ``system_sota_longquan.py`` 产出的 JSON 报告（--no-pipeline 单帧臂）。

用法::

    python manu/data/compare_lq_baselines.py \\
        --fpv runs/logs/ft_golden_pool_single.json runs/logs/ft_blind_001324_single.json \\
        --official runs/logs/ft_official_golden_pool_single.json runs/logs/ft_official_blind_001324_single.json \\
        --out-md runs/logs/lq_baseline_semantics_compare.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PCT = 100.0


def load(paths: list[str]) -> tuple[dict, dict]:
    """合并多份报告 -> (per_sequence, overall_by_file)。"""
    per_seq: dict[str, dict] = {}
    overall: list[tuple[str, dict]] = []
    for raw in paths:
        data = json.loads(Path(raw).read_text())
        for seq, row in (data.get("per_sequence") or {}).items():
            per_seq[seq] = row
        overall.append((Path(raw).stem, data.get("overall", {})))
    return per_seq, overall


def fmt_row(name: str, a: dict, b: dict) -> str:
    def f1(x):
        return x.get("f1", 0.0) * PCT

    def delta(x, y):
        # F1 已是百分数，差值不要再乘一次
        return (y.get("f1", 0.0) - x.get("f1", 0.0)) * PCT

    return (
        f"| {name} | {a.get('tp')} | {a.get('fp')} | {a.get('fn')} | "
        f"{a.get('recall', 0) * PCT:.3f} | {a.get('precision', 0) * PCT:.3f} | {f1(a):.3f} | "
        f"{a.get('far_per_frame', 0):.6f} | {b.get('tp')} | {b.get('fp')} | {b.get('fn')} | "
        f"{b.get('recall', 0) * PCT:.3f} | {b.get('precision', 0) * PCT:.3f} | {f1(b):.3f} | "
        f"{b.get('far_per_frame', 0):.6f} | {delta(a, b):+.3f} |"
    )


HEADER = (
    "| 序列 | FPV TP | FP | FN | R% | P% | F1% | FAR | OFC TP | FP | FN | R% | P% | F1% | FAR | ΔF1(pp) |\n"
    "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="对比两套特征语义下的龙泉山单帧 baseline")
    parser.add_argument("--fpv", nargs="+", required=True, help="FPV 语义的评测报告（可多份）")
    parser.add_argument("--official", nargs="+", required=True, help="official 语义的评测报告")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    fpv_seq, fpv_overall = load(args.fpv)
    ofc_seq, ofc_overall = load(args.official)

    lines = ["# 龙泉山单帧 Baseline：FPV 仿射连乘 vs official 直接拟合", ""]
    lines.append("口径：`system_sota_longquan.py --no-pipeline`（纯单帧裸跑，不带后处理航迹融合）")
    lines.append("")
    lines.append(HEADER)
    for seq in sorted(fpv_seq):
        if seq not in ofc_seq:
            continue
        lines.append(fmt_row(seq, fpv_seq[seq], ofc_seq[seq]))
    # 合计行：按传入文件的**顺序**配对（goldens 在前、blind 在后）
    for (fpv_tag, fpv_ov), (ofc_tag, ofc_ov) in zip(fpv_overall, ofc_overall):
        if not fpv_ov or not ofc_ov:
            continue
        lines.append(fmt_row(f"**合计 {fpv_tag}**", fpv_ov, ofc_ov))

    data = {
        "fpv": {seq: {k: v for k, v in row.items()} for seq, row in fpv_seq.items()},
        "official": {seq: {k: v for k, v in row.items()} for seq, row in ofc_seq.items()},
        "fpv_overall": {tag: ov for tag, ov in fpv_overall},
        "official_overall": {tag: ov for tag, ov in ofc_overall},
    }
    text = "\n".join(lines) + "\n"
    print(text)

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Markdown 已写入: {out}")
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"JSON 已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
