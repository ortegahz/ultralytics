#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render a Δn (jump-frame) search/eval report as markdown.

The json written by ``rt_fps_jitter_search.py --out-json`` is the single source of truth:
every number in the markdown is copied from that json, so the two can never drift apart.

Usage::

    python manu/evaluation/render_search_report_md.py <report.json> [report.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_report_md(report: dict) -> str:
    """Render a search/eval report dict as markdown."""
    lines: list[str] = []
    add = lines.append
    meta = report.get("meta", {})
    add("# trial_0080 · Δn（跳帧）搜索报告")
    add("")
    add("> 本文件由 `--out-md` 从**同名 json** 自动生成；**数字以 json 为准**。")
    add("> ⚠️ 本文件是**某一次跑批的原始记录**，**跳帧数字不是交付口径**：若该批次产于采样器修复前的 "
        "`hash()` 采样器、或只是 objective-best 阶段，其跳帧值**已作废**。"
        "交付口径见 `artifacts/shortlist_deterministic.json` 与 `artifacts/rt_eval_delivered_armA.json`。")
    add("")
    add("## 1. 运行元信息")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add(f"| cache_root | `{meta.get('cache_root', '')}` |")
    add(f"| labels_root | `{meta.get('labels_root', '')}` |")
    add(f"| 序列数 | {len(meta.get('sequences', []))} |")
    add(f"| 情景 | {', '.join(str(s) for s in meta.get('scenarios', []))} |")
    add(f"| seeds | {', '.join(str(s) for s in meta.get('seeds', []))} |")
    add(f"| max_frames | {meta.get('max_frames')} |")
    add(f"| 耗时 | {report.get('elapsed_s')} s |")
    add("")
    add("## 2. 各臂结果")
    add("")
    add("| 臂 | 口径 | mean F1 | worst F1 | mean FAR |")
    add("| --- | --- | --- | --- | --- |")
    for name, arm in report.get("arms", {}).items():
        if "perframe" in arm:
            per = arm["perframe"]
            add(f"| {name} | 逐帧（截断口径） | {per.get('f1_delivered_agg', 0.0):.6f} | - | "
                f"{per.get('far_delivered_agg', 0.0):.6f} |")
        elif "_jump_mean_f1" in arm:
            add(f"| {name} | 跳帧 | {arm['_jump_mean_f1']:.6f} | {arm['_jump_worst_f1']:.6f} | "
                f"{arm['_jump_mean_far']:.6f} |")
        else:
            add(f"| {name} | 逐帧 | {arm.get('f1', 0.0):.6f} | - | {arm.get('far_per_frame', 0.0):.6f} |")
    add("")
    add("## 3. 跳帧臂逐情景（截断口径）")
    add("")
    add("| 臂 | 情景 | F1 | FAR | Recall | Precision |")
    add("| --- | --- | --- | --- | --- | --- |")
    for name, arm in report.get("arms", {}).items():
        for scenario in meta.get("scenarios", []):
            row = arm.get(scenario)
            if not isinstance(row, dict):
                continue
            add(f"| {name} | {scenario} | {row.get('f1_delivered_agg', 0.0):.6f} | "
                f"{row.get('far_delivered_agg', 0.0):.6f} | {row.get('recall_delivered_agg', 0.0):.6f} | "
                f"{row.get('precision_delivered_agg', 0.0):.6f} |")
    add("")
    add("## 4. 逐帧天花板（逐情景）")
    add("")
    add("| 情景 | F1 | FAR | Recall | Precision |")
    add("| --- | --- | --- | --- | --- |")
    for scenario, row in report.get("ceiling", {}).items():
        if not isinstance(row, dict):
            continue
        add(f"| {scenario} | {row.get('f1', 0.0):.6f} | {row.get('far_per_frame', 0.0):.6f} | "
            f"{row.get('recall', 0.0):.6f} | {row.get('precision', 0.0):.6f} |")
    add("")
    add("## 5. 注入节奏（cadence）")
    add("")
    add("| 情景 | fps | mean Δn | updates | Δn 直方图 |")
    add("| --- | --- | --- | --- | --- |")
    for scenario, row in report.get("cadence", {}).items():
        if not isinstance(row, dict):
            continue
        hist = row.get("delta_n_hist", {})
        hist_txt = " ".join(f"{k}:{v}" for k, v in sorted(hist.items(), key=lambda kv: int(kv[0])))
        add(f"| {scenario} | {row.get('fps')} | {row.get('mean_delta_n')} | {row.get('updates')} | {hist_txt} |")
    add("")
    add("## 6. 候选参数（opt_params）")
    add("")
    add("```json")
    add(json.dumps(report.get("opt_params", {}), indent=2, ensure_ascii=False))
    add("```")
    add("")
    history = report.get("search_history")
    if history:
        add(f"## 7. 搜索轨迹：{len(history)} 个候选（objective 全量落在 json 的 `search_history` 字段里）")
        add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_json", help="the json written by rt_fps_jitter_search.py --out-json")
    parser.add_argument("report_md", nargs="?", default="", help="output markdown (default: alongside the json)")
    args = parser.parse_args()
    source = Path(args.report_json)
    target = Path(args.report_md) if args.report_md else source.with_suffix(".md")
    target.write_text(render_report_md(json.loads(source.read_text(encoding="utf-8"))), encoding="utf-8")
    print(f"[out] {target}")


if __name__ == "__main__":
    main()
