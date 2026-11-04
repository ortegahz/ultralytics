#!/usr/bin/env python3
"""从 arms_*.json 产物生成变体对照表（Markdown + JSON）：相对冠军基线列出差异旋钮与尺子2 指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARMS_DIR = ROOT / "runs/lookahead/final"
DEFAULT_OUT_MD = DEFAULT_ARMS_DIR / "variant_table.md"
DEFAULT_OUT_JSON = DEFAULT_ARMS_DIR / "variant_table.json"
DEFAULT_BASELINE_TAG = "WTH15"

# 冠军产物在 final/ 下有两个副本：arms_<tag>.json 与 arms_final_<tag>.json（旋钮与指标一致，仅 arms 列表不同）；
# 优先取 final_ 副本（README §7 登记的定稿主产物）。
BASELINE_TEMPLATES = ("arms_final_{tag}.json", "arms_{tag}.json")

# 运行形态字段：arms=本次评测的臂列表，mode/main_arm=运行模式，frames_root=数据根目录；与算法调参无关，不参与 diff。
NON_KNOB_KEYS = ("arms", "main_arm", "mode", "frames_root")

REPLAY = ("arms", "replay")

# 尺子2 指标在产物里的位置：F1/FAR 在 current_frame（当前物理帧口径），hit@5px/CLE 在 servo。
METRIC_PATHS = {
    "f1": ("current_frame", "overall", "f1"),
    "far": ("current_frame", "overall", "far_per_frame"),
    "hit5": ("servo", "overall", "hit_rate_5px"),
    "cle_p50": ("servo", "overall", "cle_p50"),
    "cle_p95": ("servo", "overall", "cle_p95"),
}
PERCENT_METRICS = ("f1", "hit5")
FORMATS = {
    "f1": "{:.4f}%",
    "hit5": "{:.2f}%",
    "far": "{:.4f}",
    "cle_p50": "{:.2f}px",
    "cle_p95": "{:.2f}px",
}


def _dig(doc, *keys):
    """按路径取值，任一层缺失返回 None。"""
    cur = doc
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _fmt_value(value):
    """旋钮取值的紧凑写法（None → —，bool → True/False，float → 去掉尾零）。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fmt_value(v) for v in value) + "]"
    return str(value)


def _rel(path):
    """仓库内路径用相对路径表示，仓库外用绝对路径。"""
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _collect_metrics(doc):
    """取尺子2 指标；返回 (metrics, 不可用原因)。"""
    replay = _dig(doc, *REPLAY)
    if not isinstance(replay, dict):
        mode = _dig(doc, "lookahead", "mode")
        arm = _dig(doc, "lookahead", "main_arm")
        detail = f"（mode={mode}, main_arm={arm}）" if mode or arm else ""
        return None, f"产物无 arms.replay 臂{detail}"
    metrics = {}
    for name, path in METRIC_PATHS.items():
        value = _dig(replay, *path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None, f"缺指标 {name}（arms.replay.{'.'.join(path)}）"
        metrics[name] = float(value) * (100.0 if name in PERCENT_METRICS else 1.0)
    return metrics, None


def _sequence_names(doc):
    """当前物理帧口径的序列集合（无则 None）。"""
    per_sequence = _dig(doc, *REPLAY, "current_frame", "per_sequence")
    return sorted(per_sequence) if isinstance(per_sequence, dict) else None


def _knob_diff(lookahead, base_knobs, base_order):
    """对比产物 `lookahead` 与基线旋钮。

    返回 (差异字典, 基线有而本产物未记录的键)。差异字典只含**两方都有记录**且取值不同的旋钮；
    基线有而本产物没有记录的旋钮不可比，单列出来（不混进差异，避免谎称"本产物取值不同"）。
    """
    variant = {k: v for k, v in (lookahead or {}).items() if k not in NON_KNOB_KEYS}
    diff = {k: v for k, v in variant.items() if k in base_knobs and base_knobs[k] != v}
    missing = sorted(set(base_knobs) - set(variant))
    ordered = {k: diff[k] for k in base_order if k in diff}
    ordered.update({k: v for k, v in sorted(diff.items()) if k not in ordered})
    return ordered, missing


def _axis(doc, baseline_delta: int) -> str:
    """实验轴分类：**延迟轴**（Δ≠基线）与**旋钮轴**不可混排。

    延迟轴产物（Δ 扫描）改的是「检测滞后」这个自变量，不是旋钮；把它按 hit@5px
    混进旋钮表，会让「Δ=2 的 87.35%」压过冠军（84.67%），看起来像「旋钮调优赢过冠军」
    —— 那是两个不同的轴，必须分开排。
    """
    delta = (doc.get("lookahead") or {}).get("delta")
    if delta is not None and int(delta) != int(baseline_delta):
        return "delay"
    return "knob"


def _knob_cell(row, base_knobs):
    """差异旋钮列的 Markdown 文本：`key=本产物值 → 基线值`。"""
    parts = []
    for key, value in row["knob_diff"].items():
        champion = base_knobs.get(key)
        parts.append(f"{key}={_fmt_value(value)} → {_fmt_value(champion) if key in base_knobs else '(基线无此键)'}")
    if row["knob_missing"]:
        parts.append(f"(未记 {len(row['knob_missing'])} 键)")
    return "; ".join(parts) if parts else "(与冠军相同)"


def _load_baseline(arms_dir, tag):
    """按 tag 定位基线产物；同 tag 的两个副本取先存在且可用的那个。"""
    candidates = [arms_dir / tpl.format(tag=tag) for tpl in BASELINE_TEMPLATES]
    for path in candidates:
        if path.is_file():
            try:
                doc = _load(path)
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"基线产物读不出来：{path}（{exc}）")
            metrics, reason = _collect_metrics(doc)
            if metrics is not None:
                return path, doc
    raise SystemExit(f"目录 {arms_dir} 里找不到可用的基线产物（tag={tag}，试过：{[c.name for c in candidates]}）")


def main():
    parser = argparse.ArgumentParser(description="从 arms_*.json 产物生成变体对照表（相对冠军基线）")
    parser.add_argument("--arms-dir", default=str(DEFAULT_ARMS_DIR), help="产物目录（扫描 arms_*.json）")
    parser.add_argument(
        "--baseline-tag",
        default=DEFAULT_BASELINE_TAG,
        help="冠军基线 tag（产物 arms_<tag>.json / arms_final_<tag>.json）",
    )
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD), help="Markdown 输出路径")
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON), help="JSON 输出路径")
    args = parser.parse_args()

    arms_dir = Path(args.arms_dir).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()

    files = sorted(arms_dir.glob("arms_*.json"))
    if not files:
        raise SystemExit(f"目录里没有 arms_*.json：{arms_dir}")

    baseline_path, baseline_doc = _load_baseline(arms_dir, args.baseline_tag)
    baseline_metrics, reason = _collect_metrics(baseline_doc)
    if baseline_metrics is None:
        raise SystemExit(f"基线产物不可用：{baseline_path.name}（{reason}）")
    baseline_knobs = {k: v for k, v in (baseline_doc.get("lookahead") or {}).items() if k not in NON_KNOB_KEYS}
    baseline_order = [k for k in (baseline_doc.get("lookahead") or {}) if k not in NON_KNOB_KEYS]
    baseline_seqs = _sequence_names(baseline_doc) or []
    baseline_delta = int((baseline_doc.get("lookahead") or {}).get("delta", 8))

    rows, excluded = [], []
    for path in files:
        name = path.name
        tag = name[len("arms_") : -len(".json")]
        try:
            doc = _load(path)
        except (OSError, json.JSONDecodeError) as exc:
            excluded.append({"file": name, "tag": tag, "reason": f"JSON 读不出来：{exc}"})
            continue
        metrics, reason = _collect_metrics(doc)
        if metrics is None:
            excluded.append({"file": name, "tag": tag, "reason": reason})
            continue
        seqs = _sequence_names(doc)
        if seqs is None:
            excluded.append({"file": name, "tag": tag, "reason": "缺 arms.replay.current_frame.per_sequence"})
            continue
        if seqs != baseline_seqs:
            excluded.append(
                {"file": name, "tag": tag, "reason": f"序列口径不一致：{len(seqs)} 序列 ≠ 基线 {len(baseline_seqs)} 序列"}
            )
            continue
        diff, missing = _knob_diff(doc.get("lookahead"), baseline_knobs, baseline_order)
        recorded = [k for k in (doc.get("lookahead") or {}) if k not in NON_KNOB_KEYS]
        rows.append(
            {
                "tag": tag,
                "source": name,
                "knob_diff": diff,
                "knob_missing": missing,
                "recorded_knobs": len(recorded),
                "axis": _axis(doc, baseline_delta),
                "delta": (doc.get("lookahead") or {}).get("delta"),
                **metrics,
            }
        )

    knob_rows = [r for r in rows if r["axis"] == "knob"]
    other_rows = [r for r in rows if r["axis"] != "knob"]
    knob_rows.sort(key=lambda r: (-r["hit5"], -r["f1"], r["tag"]))
    other_rows.sort(key=lambda r: (r["tag"], -r["hit5"]))
    assert len(rows) + len(excluded) == len(files), "行数与扫描到的产物数对不上（聚合有 bug）"

    comparable = sum(1 for row in rows if not row["knob_missing"])
    copies = [
        path.name
        for path in (arms_dir / tpl.format(tag=args.baseline_tag) for tpl in BASELINE_TEMPLATES)
        if path.is_file() and path != baseline_path
    ]

    # Markdown
    header = [
        f"# Look-ahead 变体对照表（相对冠军基线 {args.baseline_tag}）",
        "",
        f"> 产物目录：`{_rel(arms_dir)}/arms_*.json`（`lookahead_longquan.py` 产物）；"
        f"**所有数字直接取自产物 JSON，未重算**。",
        f"> F1 / FAR 取 `arms.replay.current_frame.overall`（尺子2 当前物理帧口径），"
        f"hit@5px / CLE 取 `arms.replay.servo.overall`。",
        f"> 「差异旋钮」= 该产物 `lookahead` 中**两方都有记录**且取值不同的旋钮，格式 `key=本产物值 → 基线值`；"
        f"基线有而本产物未记录的旋钮不可比，用 `(未记 N 键)` 标出（清单见 `variant_table.json`）。",
        f"> 运行形态字段（`{'`/`'.join(NON_KNOB_KEYS)}`）不参与 diff。产物里取值为 null 的旋钮记作 `—`。",
        "",
        f"- 基线：`{args.baseline_tag}`（产物 `{baseline_path.name}`"
        + (f"；同 tag 另有副本 `{'`、`'.join(copies)}`，旋钮与指标一致" if copies else "")
        + "）",
        f"- 扫描 **{len(files)}** 个 `arms_*.json` → **可用 {len(rows)}**"
        f"（旋钮轴 **{len(knob_rows)}** 行 = §1 本表行数；延迟轴 **{len(other_rows)}** 行 = §2，不参与旋钮排序）"
        f"、**不可用 {len(excluded)}**（见 §3，未静默丢弃）",
        f"- 口径：{len(baseline_seqs)} 序列（取自基线产物；本表只收录序列集合与基线完全一致的产物）",
        f"- 旋钮覆盖：**{comparable}** 行与冠军逐键可比（{len(baseline_knobs)} 个旋钮全记录）；"
        f"**{len(rows) - comparable}** 行有「基线有而该产物未记录」的旋钮"
        f"（最多 {max((len(r['knob_missing']) for r in rows), default=0)} 键）"
        f"——老一代产物的 `lookahead` 实际只记了 **{min((r['recorded_knobs'] for r in rows), default=0)} 个旋钮**"
        f"（基线 {len(baseline_knobs)} 个），未记 fast_*/replay_xcorr_* 等字段，"
        f"与冠军无法逐键比对，变体身份只能靠 tag 名推断",
        f"- **实验轴**：Δ≠{baseline_delta} 的产物归入**延迟轴**（§2）——它们改的是「检测滞后」这个自变量，"
        f"不是旋钮；混进旋钮表会让「Δ=2 的高命中率」压过冠军，看着像旋钮调优赢了 ⇒ 两个轴必须分开",
        "- 排序：hit@5px 降序 → F1 降序 → tag 升序（同名次按 tag 排，结果可重复）",
        "",
        f"## 1. 变体总表 · 旋钮轴（{len(knob_rows)} 行）",
        "",
        f"| {' | '.join(('tag', f'差异旋钮(相对 {args.baseline_tag})', 'F1%', 'hit@5px%', 'FAR/帧', 'CLE p50', 'CLE p95'))} |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in knob_rows:
        cells = [
            row["tag"],
            _knob_cell(row, baseline_knobs),
            FORMATS["f1"].format(row["f1"]),
            FORMATS["hit5"].format(row["hit5"]),
            FORMATS["far"].format(row["far"]),
            FORMATS["cle_p50"].format(row["cle_p50"]),
            FORMATS["cle_p95"].format(row["cle_p95"]),
        ]
        header.append("| " + " | ".join(cells) + " |")

    header += [
        "",
        f"## 2. 其他实验轴 · 延迟轴（{len(other_rows)} 行，**不参与 §1 的旋钮排序**）",
        "",
        f"| {' | '.join(('tag', 'Δ', f'差异旋钮(相对 {args.baseline_tag})', 'F1%', 'hit@5px%', 'FAR/帧', 'CLE p50', 'CLE p95'))} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in other_rows:
        cells = [
            row["tag"],
            str(row.get("delta", "—")),
            _knob_cell(row, baseline_knobs),
            FORMATS["f1"].format(row["f1"]),
            FORMATS["hit5"].format(row["hit5"]),
            FORMATS["far"].format(row["far"]),
            FORMATS["cle_p50"].format(row["cle_p50"]),
            FORMATS["cle_p95"].format(row["cle_p95"]),
        ]
        header.append("| " + " | ".join(cells) + " |")
    header += [
        "",
        f"## 3. 不可用产物（{len(excluded)} 个）",
        "",
        "| 文件 | 原因 |",
        "|---|---|",
    ]
    for item in excluded:
        header.append(f"| `{item['file']}` | {item['reason']} |")
    header += [
        "",
        f"> 合计：扫描 {len(files)} = 可用 {len(rows)}（旋钮轴 {len(knob_rows)} + 延迟轴 {len(other_rows)}）"
        f" + 不可用 {len(excluded)}。",
        "",
    ]
    md_text = "\n".join(header) + "\n"

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text, encoding="utf-8")

    payload = {
        "baseline_tag": args.baseline_tag,
        "baseline_source_file": baseline_path.name,
        "baseline_knobs": baseline_knobs,
        "metric_sources": {name: "arms.replay." + ".".join(path) for name, path in METRIC_PATHS.items()},
        "units": {"f1": "%", "hit5": "%", "far": "false-alarms/frame", "cle_p50": "px", "cle_p95": "px"},
        "row_count": len(rows),
        "scanned_file_count": len(files),
        "source_files": [row["source"] for row in rows],
        "excluded_files": excluded,
        "rows": [
            {
                "tag": row["tag"],
                "source": row["source"],
                "knob_diff": row["knob_diff"],
                "knob_missing": row["knob_missing"],
                "f1": row["f1"],
                "hit5": row["hit5"],
                "far": row["far"],
                "cle_p50": row["cle_p50"],
                "cle_p95": row["cle_p95"],
            }
            for row in rows
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"扫描 {len(files)} 个 arms_*.json → 可用 {len(rows)} 行 / 不可用 {len(excluded)} 个")
    print(f"基线 {args.baseline_tag} ← {baseline_path.name}（hit@5px {baseline_metrics['hit5']:.2f}%）")
    print(f"写出 {_rel(out_md)} 与 {_rel(out_json)}")


if __name__ == "__main__":
    main()
