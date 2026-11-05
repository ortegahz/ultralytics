#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""「出框门」（FOV margin gate）受控 A/B 报告生成器。

所有数字**从产物 JSON 读出，不重算**；缺失的输入对应小节直接跳过并留说明。

    python manu/tracking/lookahead/make_fov_gate_report.py \
        --root runs/lookahead/fovgate --ref-root runs/lookahead/fi_nogmc \
        --title "P3 新机理 · 出框门（FOV margin gate）受控 A/B" \
        --out-md runs/lookahead/fovgate/report.md --out-json runs/lookahead/fovgate/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ["original", "stale", "kinematic", "replay"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="出框门报告生成器")
    p.add_argument("--root", default="runs/lookahead/fovgate")
    p.add_argument("--ref-root", default="runs/lookahead/fi_nogmc")
    p.add_argument("--title", default="P3 新机理 · 出框门（FOV margin gate）受控 A/B")
    p.add_argument("--out-md", default="runs/lookahead/fovgate/report.md")
    p.add_argument("--out-json", default="runs/lookahead/fovgate/report.json")
    return p.parse_args()


def load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def arm_row(doc: dict, arm: str) -> dict:
    a = doc["arms"][arm]
    cf = a["current_frame"]["overall"]
    sv = a["servo"]["overall"]
    src = (a.get("source_frame") or {}).get("overall", {})
    return {
        "f1": cf["f1"] * 100,
        "recall": cf["recall"] * 100,
        "precision": cf["precision"] * 100,
        "far": cf["far_per_frame"],
        "tp": cf["tp"],
        "fp": cf["fp"],
        "fn": cf["fn"],
        "hit5": sv.get("hit_rate_5px", 0.0) * 100,
        "cle50": sv.get("cle_pooled_p50"),
        "cle95": sv.get("cle_pooled_p95"),
        "coverage": sv.get("coverage"),
        "src_f1": src.get("f1"),
        "gated": (a.get("stream_stats") or {}).get("fov_gated"),
    }


GATE_FIELDS = (
    ("f1", "current_frame.overall.f1"),
    ("recall", "current_frame.overall.recall"),
    ("precision", "current_frame.overall.precision"),
    ("far_per_frame", "current_frame.overall.far_per_frame"),
    ("tp", "current_frame.overall.tp"),
    ("fp", "current_frame.overall.fp"),
    ("fn", "current_frame.overall.fn"),
    ("det", "current_frame.overall.det"),
    ("gt", "current_frame.overall.gt"),
)


def gate_compare(new_doc: dict, ref_doc: dict) -> list[tuple[str, str, object, object]]:
    """四臂逐位比对（严格相等，无容差）⇒ 扫描器是否引入额外变量。"""
    diffs = []
    for arm in ARMS:
        na, ra = new_doc["arms"][arm], ref_doc["arms"][arm]
        ncf, rcf = na["current_frame"]["overall"], ra["current_frame"]["overall"]
        nsv, rsv = na["servo"]["overall"], ra["servo"]["overall"]
        nsrc = (na.get("source_frame") or {}).get("overall", {})
        rsrc = (ra.get("source_frame") or {}).get("overall", {})
        for key, label in GATE_FIELDS:
            if ncf.get(key) != rcf.get(key):
                diffs.append((arm, label, ncf.get(key), rcf.get(key)))
        for key in ("hit_rate_5px", "cle_pooled_p50", "cle_pooled_p95", "cle_pooled_p99",
                    "cle_pooled_max", "cle_pooled_samples"):
            if nsv.get(key) != rsv.get(key):
                diffs.append((arm, f"servo.overall.{key}", nsv.get(key), rsv.get(key)))
        for key in ("f1", "far_per_frame"):
            if nsrc.get(key) != rsrc.get(key):
                diffs.append((arm, f"source_frame.overall.{key}", nsrc.get(key), rsrc.get(key)))
    return diffs


def fmt(value, digits=4, dash="—"):
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return dash
    return f"{number:.{digits}f}"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    md: list[str] = []
    gates: list[dict] = []
    summary: dict = {"root": str(root), "sets": {}}

    md.append(f"# {args.title}")
    md.append("")
    md.append(
        "> 生成脚本：`manu/tracking/lookahead/make_fov_gate_report.py` —— **所有数字从产物 JSON 读出，不重算**\n"
        "> 扫描器：`manu/tracking/lookahead/server_overnight/run_fov_gate_scan.sh`"
        "（CFG 单一真值 = `run_delivery.sh` 的 `CFG=`；唯一变量 = `--replay-fov-margin-px`）\n"
        "> 新旋钮**默认 0 = 关闭 ⇒ 四臂逐位不变**（计算门禁见下方「门禁」小节）"
    )
    md.append("")

    md.append("## 0. 口径声明")
    md.append("")
    md.append("| # | 纪律 | 说明 |")
    md.append("|:--|:--|:--|")
    md.append("| 1 | **交付基线 = `stale` 臂** | `original` 臂 = 零延迟天花板（oracle 上界），**不是**基线 |")
    md.append("| 2 | **总体 CLE 用 `cle_pooled_*`** | 逐序 `cle_p*` 是逐序列最差值，口径不同 |")
    md.append("| 3 | **唯一变量 = 出框门阈值 m** | 其余旋钮一字不改；门只作用于 `replay` 臂（stale/kinematic 是对照基线） |")
    md.append(
        "| 4 | **8 序 ≠ 24 序** | 8 序只覆盖 24 序官方口径一小部分 GT（精确覆盖率每次重算，见 "
        "`runs/lookahead/fi_nogmc/report.md` §0 纪律 6）⇒ 只能做机理筛选/A/B，**不能给交付档排序** |"
    )
    md.append("| 5 | **交付三门** | F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030 |")
    md.append("")

    for set_name in sorted(p.name for p in root.glob("runs/*") if p.is_dir()):
        model_dir = root / "runs" / set_name
        deltas = sorted(int(p.name[1:]) for p in model_dir.glob("d*") if p.is_dir())
        for delta in deltas:
            d_dir = model_dir / f"d{delta}"
            tags = sorted(
                (p for p in d_dir.glob("fov__m*") if (p / "arms.json").is_file()),
                key=lambda p: int(p.name[len("fov__m") :]),
            )
            if not tags:
                continue
            md.append(f"## {set_name} · Δ={delta}")
            md.append("")
            md.append("| 出框门 m (px) | F1% | ΔF1(pp) | Recall% | FAR/帧 | ΔFAR | FP | TP | hit@5px% | Δhit5(pp) | CLE p95(池) | coverage | 交付三门 |")
            md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
            base = None
            rows = []
            m0_doc = None
            for tag in tags:
                doc = load(tag / "arms.json")
                if not doc:
                    continue
                row = arm_row(doc, "replay")
                row["m"] = int(tag.name.split("m")[-1])
                row["tag"] = tag.name
                rows.append(row)
                if row["m"] == 0:
                    base = row
                    m0_doc = doc
            for row in rows:
                ok = row["f1"] >= 88.0 and row["recall"] >= 85.0 and row["far"] <= 0.030
                df1 = f"{row['f1'] - base['f1']:+.4f}" if base else "—"
                dfar = f"{row['far'] - base['far']:+.6f}" if base else "—"
                dhit = f"{row['hit5'] - base['hit5']:+.4f}" if base else "—"
                gate_cell = "**PASS**" if ok else "**FAIL**"
                if ok:
                    gate_cell += f"（FAR 余量 {0.030 - row['far']:.6f}）"
                md.append(
                    f"| {row['m']} | {fmt(row['f1'])} | {df1} | {fmt(row['recall'])} | {fmt(row['far'], 6)} | {dfar} | "
                    f"{row['fp']} | {row['tp']} | {fmt(row['hit5'])} | {dhit} | "
                    f"{fmt(row['cle95'], 3)} | {fmt(row['coverage'], 4)} | {gate_cell} |"
                )
            md.append("")
            ref_path = Path(args.ref_root) / set_name / f"d{delta}" / "arms.json"
            ref_doc = load(ref_path) if m0_doc else None
            if m0_doc and ref_doc:
                diffs = gate_compare(m0_doc, ref_doc)
                gates.append({"set": set_name, "delta": delta, "ref": str(ref_path),
                              "n_diff": len(diffs),
                              "diffs": [{"arm": a, "field": f, "new": n, "ref": r}
                                        for a, f, n, r in diffs]})
            else:
                gates.append({"set": set_name, "delta": delta,
                              "ref": str(ref_path), "n_diff": None,
                              "diffs": [], "note": "m=0 产物或参考产物缺失"})
            summary["sets"][f"{set_name}_d{delta}"] = rows

    md.append("## 门禁：m=0 与参考产物逐位一致（出框门默认 0 = 关闭 ⇒ 四臂逐位不变）")
    md.append("")
    md.append("> 出框门 `--replay-fov-margin-px` 默认 **0 = 关闭** ⇒ **m=0 的四臂必须与参考产物逐位相同**"
              "（严格相等，无容差）。参考产物 = `<ref-root>/<模型包>/d<Δ>/arms.json`。")
    md.append("")
    md.append("| 模型包 | Δ | 参考产物 | 不一致处数 | 判定 |")
    md.append("|:--|--:|:--|--:|:--|")
    for g in gates:
        if g["n_diff"] is None:
            md.append(f"| {g['set']} | {g['delta']} | `{g['ref']}` | — | ⚠️ {g.get('note', '产物缺失')} |")
        else:
            ok = g["n_diff"] == 0
            md.append(f"| {g['set']} | {g['delta']} | `{g['ref']}` | {g['n_diff']} | "
                      + ("**PASS**" if ok else "**FAIL**") + " |")
    md.append("")
    for g in gates:
        if g["n_diff"]:
            md.append(f"**{g['set']} Δ={g['delta']} 的不一致明细：**")
            md.append("")
            md.append("| 臂 | 字段 | m=0 产物 | 参考产物 |")
            md.append("|:--|:--|--:|--:|")
            for d in g["diffs"]:
                md.append(f"| {d['arm']} | `{d['field']}` | {d['new']!r} | {d['ref']!r} |")
            md.append("")
    summary["gates"] = gates

    md.append("## 机理与出处")
    md.append("")
    md.append(
        "- 新旋钮 `--replay-fov-margin-px`（默认 0 = 关 = 逐位不变），实现于 "
        "`manu/tracking/lookahead/extrapolator.py::_apply_fov_margin_gate`，"
        "在 `build_report_stream` 的 `mode==\"replay\"` 分支里、**输出限幅之前**调用"
        "（限幅会把画外锚点写进 `last_pos` ⇒ 门若跑在限幅之后就删不掉那些被拖回画内的误报）。"
    )
    md.append(
        "- 机理：转台只能瞄画面里的东西 ⇒ 上报坐标落在画面外、或距边界不足 m px 的那一条上报必然是误报。"
    )
    md.append(
        "- **FAR 增量的帧级构成**：8 序 cs_t0043 Δ=3 的帧级归因（位移型 / 幽灵型两类占比）见专题档 "
        "`manu/memory/lookahead_extrapolator_20261102.md` §37.6。"
        "⚠️ **该归因未留可复算脚本 ⇒ 本报告不复述其百分比**，避免不可复算的数字被当成实测值引用。"
    )
    md.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[SUCCESS] -> {out_md}")
    print(f"[SUCCESS] -> {args.out_json}")


if __name__ == "__main__":
    main()
