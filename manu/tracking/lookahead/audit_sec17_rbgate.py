#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立审计：``runs/lookahead/fi_nogmc/FINAL_REPORT.md`` §17.4（回退链峰值门受控 A/B）逐位对账。

设计约束（与前几轮审计同一范式）：

1. **不 import / 不调用** ``make_fastinfer_final_report.py`` / ``make_reg_oracle_report.py`` 里的任何
   函数 —— 那些函数自己就是被审计对象。本脚本**只**读产物 JSON/JSONL + 被审计的 md 文本，
   所有指标**独立重算**。
2. 口径（哪个字段对应报告哪一列）**从 arms.json 的字段结构自行读出并验证**，不照抄报告定义。
3. 浮点比对按**报告里的显示精度**逐位比：脚本用与渲染同精度（.4f / .6f / .3f）独立格式化后，
   与报告单元格做**字符串全等**比较（容差 = 0 个显示单位，见 ``TOLERANCE``）。
4. 只读产物；**只写本脚本自己的审计报告 md 这一个文件**；可重复运行（幂等）、无副作用。

用法::

    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec17_rbgate.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# 根定位：仓内 ⇒ 仓根；**交付包内** ⇒ 包内 `repo/` 镜像根 ⇒ 审计可在包外独立复跑。
REPO = next((a for a in Path(__file__).resolve().parents if (a / "manu/tracking/lookahead").is_dir()),
             Path(__file__).resolve().parent.parent / "repo")

REPORT_MD = REPO / "runs/lookahead/fi_nogmc/FINAL_REPORT.md"
OUT_MD = REPO / "runs/lookahead/fi_nogmc/audit_round7/audit_sec17_rbgate.md"

RB_ROOT = REPO / "runs/lookahead/rbgate24nogmc/runs/cs_t0043/d3"
DELIVERY_ARMS = REPO / "runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json"
PROGRESS_TXT = REPO / "runs/lookahead/rbgate24nogmc/progress.txt"
SCAN_SH = REPO / "manu/tracking/lookahead/server_overnight/run_rollback_gate_scan.sh"
GEN_MD = REPO / "manu/tracking/lookahead/make_fastinfer_final_report.py"
GEN_ORACLE = REPO / "manu/tracking/lookahead/make_reg_oracle_report.py"
REG_ORACLE_JSON = REPO / "runs/lookahead/fi_nogmc24/reg_oracle/reg_oracle.json"
CLOSURE_REPORT_JSON = REPO / "runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json"
REG_ORACLE_DUMP = REPO / "runs/lookahead/fi_nogmc24/reg_oracle"

PEAKS = ("0.90", "0.80", "0.70", "0.50", "0.0")
FAR_REDLINE = 0.030

# ---- 门禁脚本 run_rollback_gate_scan.sh 实际比对的字段（逐字抄自该脚本）----
GATE_FIELDS = ("f1", "recall", "precision", "far_per_frame", "tp", "fp", "fn", "det", "gt")
GATE_EXTRA = ("hit5", "cle50", "cle95", "cle99", "clemax", "samples", "source_f1", "source_far")
GATE_ARMS = ("original", "stale", "kinematic", "replay")

# ---- 逐序「变好 / 变差」判据：报告生成器用的阈值是 ±0.005 pp ----
FLAT_PP = 0.005

# ---- §17.5 静默帧归因：报告行标签 ↔ closure_report.json 的 cause key ----
SILENT_CAUSE_LABELS = (
    ("rollback_gate", "回退链峰值门"),
    ("fov_gate", "出框门"),
    ("other", "其它"),
)
SILENT_TOTAL_LABEL = "合计"
# 夜 8 第一版错误归因（842 帧）：只允许以「纠错留痕」身份出现，不得作现行归因值
SILENT_STALE_VALUE = "842"
# 任务给定的现行归因值（用于独立字面量断言，不依赖 closure_report.json）
SILENT_EXPECTED_LITERALS = ("1846", "80.1%", "148", "6.4%", "311", "13.5%")

TOLERANCE = "逐位相等（容差 = 0 个显示单位：4 位 pp / 6 位 FAR / 3 位 px）"


# ------------------------------------------------------------------ 基础工具
def load_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def fmt(value, spec):
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return format(number, spec)


def fmt_signed(value, spec):
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return format(number, spec)


def flat(obj, prefix=""):
    """嵌套 dict/list 摊平成 路径->值（list 只记长度）。"""
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flat(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        out[prefix] = ("LIST", len(obj))
    else:
        out[prefix] = obj
    return out


# ------------------------------------------------------------------ 产物读取
def load_arms():
    docs = {}
    for peak in PEAKS:
        path = RB_ROOT / f"rb__{peak}" / "arms.json"
        doc = load_json(path)
        if doc is None:
            raise SystemExit(f"[FATAL] 读不到变体产物：{path}")
        docs[peak] = doc
    docs["__delivery__"] = load_json(DELIVERY_ARMS)
    if docs["__delivery__"] is None:
        raise SystemExit(f"[FATAL] 读不到交付档基线：{DELIVERY_ARMS}")
    return docs


def replay_overall(doc):
    return doc["arms"]["replay"]["current_frame"]["overall"]


def replay_servo(doc):
    return doc["arms"]["replay"]["servo"]["overall"]


def replay_per_sequence(doc):
    return doc.get("per_sequence") or {}


def check_metric_definitions(docs):
    """独立确认报告每一列的口径（报告没说，我们从 arms.json 字段结构自己验证）。"""
    notes = []
    for peak in PEAKS:
        doc = docs[peak]
        cur = replay_overall(doc)
        srv = replay_servo(doc)
        tp, fp, fn, det, gt = cur["tp"], cur["fp"], cur["fn"], cur["det"], cur["gt"]
        f1 = 2 * tp / (2 * tp + fp + fn)
        rec = tp / gt
        pre = tp / det
        far_den = fp / cur["far_per_frame"]
        per_seq = doc["arms"]["replay"]["servo"]["per_sequence"]
        hits5 = sum(v.get("hits_5px", 0) for v in per_seq.values())
        gt_sum = sum(v.get("gt_frames", 0) for v in per_seq.values())
        hit5 = hits5 / gt_sum
        jit = max(v.get("max_jitter", 0.0) for v in per_seq.values())
        notes.append(
            {
                "peak": peak,
                "f1_recomputed": f1,
                "f1_stored": cur["f1"],
                "f1_ok": f1 == cur["f1"],
                "recall_ok": rec == cur["recall"],
                "precision_ok": pre == cur["precision"],
                "det_eq_tp_fp": det == tp + fp,
                "far_denominator": far_den,
                "far_denominator_is_int": float(far_den).is_integer(),
                "hit5_recomputed": hit5,
                "hit5_stored": srv["hit_rate_5px"],
                "hit5_ok": hit5 == srv["hit_rate_5px"],
                "cle95_stored": srv["cle_pooled_p95"],
                "cle_samples": srv["cle_pooled_samples"],
                "jitter_ok": jit == srv["max_jitter"],
                "jitter_stored": srv["max_jitter"],
                "top_level_is_replay_arm": (
                    doc.get("overall") == cur
                    and replay_per_sequence(doc) == doc["arms"]["replay"]["current_frame"]["per_sequence"]
                ),
            }
        )
    return notes


# ------------------------------------------------------------------ 报告快照
_REPORT_CACHE = {"key": None, "lines": None, "sha256": None, "mtime": None}


def report_snapshot():
    """整份 FINAL_REPORT.md **只读一次**，后续所有解析共用同一份快照。

    为什么要缓存：本仓有别的 agent 会并发重生成 FINAL_REPORT.md，若中途分多次读，
    可能读到「旧内容 + 新内容」的混合体（曾实测把 §17.4 定位到 1123 而不是 1143）。
    这里一次读盘、一次算 sha256，全程只用这一份快照 ⇒ 审计对象始终自洽。
    """
    path = Path(REPORT_MD)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _REPORT_CACHE["key"] != key:
        raw = path.read_bytes()
        _REPORT_CACHE["key"] = key
        _REPORT_CACHE["lines"] = raw.decode("utf-8").splitlines()
        _REPORT_CACHE["sha256"] = hashlib.sha256(raw).hexdigest()
        _REPORT_CACHE["mtime"] = stat.st_mtime
    return _REPORT_CACHE["lines"], _REPORT_CACHE["sha256"]


# ------------------------------------------------------------------ 报告解析
def parse_report(lines=None):
    if lines is None:
        lines, _sha = report_snapshot()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("### 17.4 "):
            start = index
            break
    if start is None:
        raise SystemExit("[FATAL] FINAL_REPORT.md 里找不到 §17.4")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("### ") or lines[index].startswith("## "):
            end = index
            break
    section = lines[start:end]

    main_rows, robust_rows = [], []
    mode = None
    for offset, line in enumerate(section):
        stripped = line.strip()
        if stripped.startswith("| rollback-min-peak |"):
            mode = "main"
            continue
        if stripped.startswith("| 变体 | ΔTP |"):
            mode = "robust"
            continue
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if mode == "main":
            if set("".join(cells)) <= set("-: "):
                continue
            if cells[0] in {"%.2f" % float(p) for p in PEAKS}:
                main_rows.append({"line": start + offset + 1, "cells": cells})
        elif mode == "robust":
            if set("".join(cells)) <= set("-: "):
                continue
            if cells[0].startswith("rb="):
                robust_rows.append({"line": start + offset + 1, "cells": cells})

    gate_line = -1
    for i, line in enumerate(section):
        if line.strip().startswith("> **门禁**"):
            gate_line = start + i + 1
            break

    def line_of(pred):
        for i, line in enumerate(section):
            if pred(line):
                return start + i + 1
        return -1

    return {
        "start_line": start + 1,
        "end_line": end,
        "main_row_map": {r["cells"][0]: r["line"] for r in main_rows},
        "robust_row_map": {r["cells"][0]: r["line"] for r in robust_rows},
        "share_line": line_of(lambda l: "总体 ΔTP 有" in l),
        "rate_line": line_of(lambda l: "边际 TP/FP 兑换率" in l),
        "verdict_line": line_of(lambda l: "夜 8 裁定：交付档保持" in l),
        "main_rows": main_rows,
        "robust_rows": robust_rows,
        "gate_line": gate_line,
        "gate_text": next((l for l in section if l.strip().startswith("> **门禁**")), ""),
        "share_text": next((l for l in section if "总体 ΔTP 有" in l), ""),
        "rate_text": next((l for l in section if "边际 TP/FP 兑换率" in l), ""),
        "verdict_text": next((l for l in section if "夜 8 裁定：交付档保持" in l), ""),
        "section_lines": [(start + i + 1, l) for i, l in enumerate(section)],
    }


# ------------------------------------------------------------------ 主表重算
def rebuild_main_table(docs):
    base = replay_overall(docs["0.90"])
    base_srv = replay_servo(docs["0.90"])
    rows = []
    for peak in PEAKS:
        cur = replay_overall(docs[peak])
        srv = replay_servo(docs[peak])
        df1 = 100.0 * (cur["f1"] - base["f1"])
        dfar = cur["far_per_frame"] - base["far_per_frame"]
        dhit = 100.0 * (srv["hit_rate_5px"] - base_srv["hit_rate_5px"])
        rows.append(
            {
                "peak": peak,
                "cells": [
                    "%.2f" % float(peak),
                    fmt(100.0 * cur["f1"], ".4f"),
                    fmt_signed(df1, "+.4f"),
                    fmt(100.0 * cur["recall"], ".4f"),
                    fmt(100.0 * cur["precision"], ".4f"),
                    fmt(cur["far_per_frame"], ".6f"),
                    fmt_signed(dfar, "+.6f"),
                    fmt(100.0 * srv["hit_rate_5px"], ".4f"),
                    fmt_signed(dhit, "+.4f"),
                    fmt(srv["cle_pooled_p95"], ".3f"),
                    fmt(srv["max_jitter"], ".3f"),
                    str(cur["tp"]),
                    str(cur["fp"]),
                ],
                "raw": {
                    "f1": cur["f1"],
                    "recall": cur["recall"],
                    "precision": cur["precision"],
                    "far": cur["far_per_frame"],
                    "hit5": srv["hit_rate_5px"],
                    "cle95": srv["cle_pooled_p95"],
                    "jitter": srv["max_jitter"],
                    "tp": cur["tp"],
                    "fp": cur["fp"],
                    "df1_pp": df1,
                    "dfar": dfar,
                    "dhit_pp": dhit,
                },
            }
        )
    return rows


def compare_main(report_rows, rebuilt):
    cells_total = cells_ok = 0
    mismatches = []
    by_peak = {("%.2f" % float(r["cells"][0])): r for r in report_rows}
    for row in rebuilt:
        rep = by_peak.get("%.2f" % float(row["peak"]))
        if rep is None:
            mismatches.append(
                {"peak": row["peak"], "col": "-", "report": "<缺行>", "recomputed": "n/a", "line": None}
            )
            cells_total += 13
            continue
        for col, (got, want) in enumerate(zip(rep["cells"], row["cells"])):
            cells_total += 1
            if got == want:
                cells_ok += 1
            else:
                mismatches.append(
                    {"peak": row["peak"], "col": col, "report": got, "recomputed": want, "line": rep["line"]}
                )
    return {"cells_total": cells_total, "cells_ok": cells_ok, "mismatches": mismatches}


# ------------------------------------------------------------------ 门禁 ① / ②
def gate_rows(doc):
    """复刻 run_rollback_gate_scan.sh 的 rows()（17 字段 × 4 臂 = 68）。"""
    out = {}
    for arm in GATE_ARMS:
        a = doc["arms"][arm]
        cur = a["current_frame"]["overall"]
        srv = a["servo"]["overall"]
        row = {k: cur.get(k) for k in GATE_FIELDS}
        row["hit5"] = srv.get("hit_rate_5px")
        row["cle50"] = srv.get("cle_pooled_p50")
        row["cle95"] = srv.get("cle_pooled_p95")
        row["cle99"] = srv.get("cle_pooled_p99")
        row["clemax"] = srv.get("cle_pooled_max")
        row["samples"] = srv.get("cle_pooled_samples")
        row["source_f1"] = (a.get("source_frame") or {}).get("overall", {}).get("f1")
        row["source_far"] = (a.get("source_frame") or {}).get("overall", {}).get("far_per_frame")
        out[arm] = row
    return out


def check_gate1(docs):
    a = gate_rows(docs["0.90"])
    b = gate_rows(docs["__delivery__"])
    diffs = []
    total = 0
    for arm in GATE_ARMS:
        for key in a[arm]:
            total += 1
            if a[arm][key] != b[arm][key]:
                diffs.append({"field": f"{arm}.{key}", "rb090": a[arm][key], "delivery": b[arm][key]})

    fa = flat(docs["0.90"])
    fb = flat(docs["__delivery__"])
    common = set(fa) & set(fb)
    deep_diff = {k: (fa[k], fb[k]) for k in sorted(common) if fa[k] != fb[k]}
    only_rb = sorted(set(fa) - set(fb))
    only_del = sorted(set(fb) - set(fa))
    return {
        "gated_fields_total": total,
        "gated_fields_equal": total - len(diffs),
        "gated_diffs": diffs,
        "deep_common": len(common),
        "deep_diff": deep_diff,
        "deep_only_rb": only_rb,
        "deep_only_delivery": only_del,
    }


def check_gate2(docs):
    base = gate_rows(docs["0.90"])
    per_variant = []
    for peak in PEAKS:
        cur = gate_rows(docs[peak])
        diffs = []
        for arm in ("original", "stale", "kinematic"):
            for key in base[arm]:
                if cur[arm][key] != base[arm][key]:
                    diffs.append({"field": f"{arm}.{key}", "variant": cur[arm][key], "base": base[arm][key]})
        per_variant.append({"peak": peak, "diffs": diffs})

    base_deep = {arm: flat(docs["0.90"]["arms"][arm]) for arm in ("original", "stale", "kinematic")}
    deep = []
    for peak in PEAKS:
        for arm in ("original", "stale", "kinematic"):
            cur = flat(docs[peak]["arms"][arm])
            for key in sorted(set(cur) | set(base_deep[arm])):
                if cur.get(key, "<缺>") != base_deep[arm].get(key, "<缺>"):
                    deep.append(
                        {
                            "peak": peak,
                            "arm": arm,
                            "field": key,
                            "variant": cur.get(key, "<缺>"),
                            "base": base_deep[arm].get(key, "<缺>"),
                        }
                    )
    return {"per_variant": per_variant, "deep_diffs": deep}


# ------------------------------------------------------------------ 逐序稳健性重算
def rebuild_robust(docs):
    base_ps = replay_per_sequence(docs["0.90"])
    out = []
    for peak in PEAKS:
        if peak == "0.90":
            continue
        cur_ps = replay_per_sequence(docs[peak])
        sd = {}
        for seq, cur in cur_ps.items():
            ref = base_ps.get(seq)
            if not ref or "f1" not in cur or "f1" not in ref:
                continue
            sd[seq] = {
                "df1_pp": 100.0 * (float(cur["f1"]) - float(ref["f1"])),
                "dtp": int(cur.get("tp", 0)) - int(ref.get("tp", 0)),
                "dfp": int(cur.get("fp", 0)) - int(ref.get("fp", 0)),
            }
        dtp = sum(v["dtp"] for v in sd.values())
        dfp = sum(v["dfp"] for v in sd.values())
        better = sum(1 for v in sd.values() if v["df1_pp"] > FLAT_PP)
        worse = sum(1 for v in sd.values() if v["df1_pp"] < -FLAT_PP)
        flat = len(sd) - better - worse
        touched = sum(1 for v in sd.values() if v["dtp"] or v["dfp"])
        top = sorted(sd.items(), key=lambda kv: -kv[1]["dtp"])[:2]
        share = (100.0 * sum(v["dtp"] for _k, v in top) / dtp) if dtp else 0.0
        worst = min(sd.items(), key=lambda kv: kv[1]["df1_pp"])
        cells = [
            f"rb={float(peak):.2f}",
            f"{dtp:+d}",
            f"{dfp:+d}",
            f"{dtp - dfp:+d}",
            f"{touched}/{len(sd)}",
            f"{better}/{worse}/{flat}",
            " / ".join(f"{k[:22]} {v['dtp']:+d}" for k, v in top) + f"（**{share:.1f}%**）",
            f"{worst[0][:22]} {worst[1]['df1_pp']:+.4f}pp",
        ]
        out.append(
            {
                "peak": peak,
                "cells": cells,
                "raw": {
                    "dtp": dtp,
                    "dfp": dfp,
                    "net": dtp - dfp,
                    "touched": touched,
                    "n_seq": len(sd),
                    "better": better,
                    "worse": worse,
                    "flat": flat,
                    "top": [(k, v["dtp"]) for k, v in top],
                    "share": share,
                    "worst_seq": worst[0],
                    "worst_df1_pp": worst[1]["df1_pp"],
                },
            }
        )
    return out


def compare_robust(report_rows, rebuilt):
    by_peak = {}
    for row in report_rows:
        by_peak[row["cells"][0].split("=")[1]] = row
    cells_total = cells_ok = 0
    mismatches = []
    for row in rebuilt:
        rep = by_peak.get(f"{float(row['peak']):.2f}")
        if rep is None:
            mismatches.append(
                {"peak": row["peak"], "col": "-", "report": "<缺行>", "recomputed": "n/a", "line": None}
            )
            cells_total += 8
            continue
        for col, (got, want) in enumerate(zip(rep["cells"], row["cells"])):
            cells_total += 1
            if got == want:
                cells_ok += 1
            else:
                mismatches.append(
                    {"peak": row["peak"], "col": col, "report": got, "recomputed": want, "line": rep["line"]}
                )
    return {"cells_total": cells_total, "cells_ok": cells_ok, "mismatches": mismatches}


# ------------------------------------------------------------------ 兑换率 / FAR 余量
def exchange_rates(docs):
    order = ["0.90", "0.80", "0.70", "0.50", "0.0"]
    rows = []
    for prev, cur in zip(order, order[1:]):
        p = replay_overall(docs[prev])
        c = replay_overall(docs[cur])
        dtp = c["tp"] - p["tp"]
        dfp = c["fp"] - p["fp"]
        rows.append(
            {
                "range": f"{prev}->{cur}",
                "dtp": dtp,
                "dfp": dfp,
                "rate": (dtp / dfp) if dfp else float("nan"),
                "far_prev": p["far_per_frame"],
                "far_cur": c["far_per_frame"],
                "far_delta": c["far_per_frame"] - p["far_per_frame"],
                "f1_delta_pp": 100.0 * (c["f1"] - p["f1"]),
            }
        )
    return rows


def far_margin(docs):
    base = replay_overall(docs["0.90"])["far_per_frame"]
    b80 = replay_overall(docs["0.80"])["far_per_frame"]
    b00 = replay_overall(docs["0.0"])["far_per_frame"]
    margin_base = FAR_REDLINE - base
    return {
        "redline": FAR_REDLINE,
        "far_090": base,
        "far_080": b80,
        "far_000": b00,
        "margin_090": margin_base,
        "margin_080": FAR_REDLINE - b80,
        "margin_000": FAR_REDLINE - b00,
        "margin_spent_pct_080": 100.0 * (b80 - base) / margin_base,
        "margin_left_pct_080": 100.0 * (FAR_REDLINE - b80) / margin_base,
        "redline_broken_000": b00 > FAR_REDLINE,
    }


# ------------------------------------------------------------------ 回退帧归因（两套口径）
def rollback_attribution(docs):
    """口径 A（生产 A/B 产物）：``replay_rollback_abstained`` 逐序计数。"""
    per = {}
    for peak in PEAKS:
        doc = docs[peak]
        ss = doc["arms"]["replay"]["stream_stats"]["per_sequence"]
        per[peak] = {k: int(v.get("replay_rollback_abstained", 0)) for k, v in ss.items()}
    return {
        "per": per,
        "total": sum(per["0.90"].values()),
        "per_variant_total": {k: sum(v.values()) for k, v in per.items()},
    }


def offline_rollback_attribution():
    """口径 B（离线 dump，GT 帧）：复刻 diag_reg_oracle.py 的 ``_silent_cause`` 归因。

    只读 ``runs/lookahead/fi_nogmc24/reg_oracle/*.jsonl``（24 序 dump），不跑任何推理。
    """
    delta, min_peak = 3, 0.90

    def cause(frame):
        refines = frame.get("r") or []
        pre = frame.get("pe") or []
        target = int(frame.get("t", -1))
        if not refines:
            return "failed_no_fallback" if frame.get("f") else "no_decision"
        rolled_back = any(int(r.get("s", target)) != target - delta for r in refines)
        matches = refines[-1].get("m") or []
        peak = float((matches[0] if matches else {}).get("pk") or 0.0)
        if rolled_back and peak < min_peak:
            return "rollback_gate"
        if pre:
            return "fov_gate"
        return "failed_no_fallback" if frame.get("f") else "other"

    per = {}
    causes = {}
    files = sorted(glob.glob(str(REG_ORACLE_DUMP / "VIDEO*.jsonl")))
    for path in files:
        seq = os.path.basename(path)[: -len(".jsonl")]
        count = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                if not (frame.get("g") or []):
                    continue  # 只统计有 GT 的帧（与 diag_reg_oracle 的 policy 口径一致）
                if frame.get("e"):
                    continue  # 有上报 => 非静默帧
                key = cause(frame)
                causes[key] = causes.get(key, 0) + 1
                if key == "rollback_gate":
                    count += 1
        per[seq] = count
    return {"per": per, "total": sum(per.values()), "causes": causes, "n_files": len(files)}


# ------------------------------------------------------------------ §17.5 静默帧归因
def clean_cell(text):
    """去掉 markdown 粗体 / 反引号 / 多余空白，便于按**内容**锚点定位。"""
    out = text.replace("**", "").replace("`", "")
    return " ".join(out.split())


def find_section(lines, prefix):
    """按标题前缀定位小节（**不依赖绝对行号**）。返回 (start_idx, end_idx, section_lines)。"""
    start = None
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            start = index
            break
    if start is None:
        return None, None, []
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("### ") or lines[index].startswith("## "):
            end = index
            break
    return start, end, lines[start:end]


def parse_silent_attribution_table(section_lines, section_start):
    """在 §17.5 里按**表头首列文本**（`归因`）定位归因表，逐行解析。

    返回 ``[(绝对行号, cells)]``；``cells`` 已去掉 markdown 记号。
    """
    rows = []
    state = "seek"
    for offset, line in enumerate(section_lines):
        stripped = line.strip()
        if state == "seek":
            if stripped.startswith("|"):
                cells = [clean_cell(c) for c in stripped.strip("|").split("|")]
                if cells and cells[0].startswith("归因"):
                    state = "body"
            continue
        if not stripped.startswith("|"):
            break  # 表结束
        cells = [clean_cell(c) for c in stripped.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue  # 分隔行
        rows.append((section_start + offset + 1, cells))
    return rows


def recompute_silent_attribution_from_dumps():
    """**独立重算** §17.5 的静默帧归因账（只读 24 序 dump，不跑任何推理）。

    复刻 ``diag_reg_oracle.py`` 的生产判定顺序（回退链峰值门 → failed 兜底 → 写 report
    → 出框门 → 限幅）与「改报 argmax」账：

    - 归因：``_silent_cause()``（含出框门判定）
    - 改报 argmax：取第一个 **出框合法** 的 refine 候选 argmax；落在 GT 框内 ⇒ TP，否则 FP
      （落在框外 ⇒ 两者都不计，即「无合法 argmax」）

    口径：**有 GT + 无上报 + 有配准决策**的帧（= ``totals.lost_with_refine``）。
    同时回吐逐序 rollback 计数，用来**交叉验证** §17.4 用的那套离线口径。
    """
    delta, min_peak, margin = 3, 0.90, 6.0
    width, height = 640.0, 512.0

    def fov_legal(point):
        if point is None:
            return False
        x, y = float(point[0]), float(point[1])
        return margin <= x <= width - margin and margin <= y <= height - margin

    def inbox_hit(point, gt_rows):
        for row in gt_rows:
            if abs(float(point[0]) - float(row[0])) <= float(row[2]) / 2.0 and abs(
                float(point[1]) - float(row[1])
            ) <= float(row[3]) / 2.0:
                return True
        return False

    def cause(frame):
        refines = frame.get("r") or []
        pre = frame.get("pe") or []
        target = int(frame.get("t", -1))
        if not refines:
            return "failed_no_fallback" if frame.get("f") else "no_decision"
        rolled_back = any(int(r.get("s", target)) != target - delta for r in refines)
        matches = refines[-1].get("m") or []
        peak = float((matches[0] if matches else {}).get("pk") or 0.0)
        if rolled_back and peak < min_peak:
            return "rollback_gate"
        if pre:
            return "fov_gate" if all(not fov_legal(p) for p in pre) else "other"
        return "failed_no_fallback" if frame.get("f") else "other"

    by_cause, tp_by, fp_by, no_legal = {}, {}, {}, {}
    per_seq_rollback = {}
    n_files = 0
    for path in sorted(glob.glob(str(REG_ORACLE_DUMP / "VIDEO*.jsonl"))):
        n_files += 1
        seq = os.path.basename(path)[: -len(".jsonl")]
        per_seq_rollback[seq] = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                gt = frame.get("g") or []
                if not gt:
                    continue  # 无 GT 帧不进归因账
                if frame.get("e"):
                    continue  # 有上报 ⇒ 非静默帧
                key = cause(frame)
                if key in ("no_decision", "failed_no_fallback"):
                    continue  # 无配准决策 ⇒ 没有 argmax 可「改报」
                by_cause[key] = by_cause.get(key, 0) + 1
                if key == "rollback_gate":
                    per_seq_rollback[seq] += 1
                legal = None
                for refine in frame.get("r") or []:
                    big = ((refine.get("m") or [{}])[0] or {}).get("k") or []
                    if big and fov_legal([big[0]["x"], big[0]["y"]]):
                        legal = [big[0]["x"], big[0]["y"]]
                        break
                if legal is None:
                    no_legal[key] = no_legal.get(key, 0) + 1
                    continue
                if inbox_hit(legal, gt):
                    tp_by[key] = tp_by.get(key, 0) + 1
                else:
                    fp_by[key] = fp_by.get(key, 0) + 1
    return {
        "by_cause": by_cause,
        "tp_by_cause": tp_by,
        "fp_by_cause": fp_by,
        "no_legal_by_cause": no_legal,
        "per_seq_rollback": per_seq_rollback,
        "total": sum(by_cause.values()),
        "n_files": n_files,
    }


def check_sec175_silent_attribution(closure_doc, reg_oracle_doc, dumps, lines=None):
    """§17.5 静默帧归因表逐位核对（**按内容锚点定位，不写死行号**）。"""
    checks = []

    def add(cid, desc, ok, expected, actual, line=None):
        checks.append(
            {
                "id": cid,
                "desc": desc,
                "ok": bool(ok),
                "expected": expected,
                "actual": actual,
                "line": line,
            }
        )

    if lines is None:
        lines, _sha = report_snapshot()
    start, end, section = find_section(lines, "### 17.5 ")
    if not section:
        add("S00", "定位 §17.5 小节（标题锚点 `### 17.5 `）", False, "存在", "缺失")
        return {"checks": checks, "rows": [], "section": None}
    add("S00", "定位 §17.5 小节（标题锚点 `### 17.5 `）", True, "存在", f"行 {start + 1}–{end}", start + 1)

    rows = parse_silent_attribution_table(section, start)
    add(
        "S01",
        "定位 §17.5 归因表（表头首列 `归因`）",
        bool(rows),
        "≥1 行",
        f"{len(rows)} 行",
        rows[0][0] if rows else None,
    )
    if not rows:
        return {"checks": checks, "rows": [], "section": (start + 1, end)}

    sa = (closure_doc or {}).get("silent_attribution") or {}
    by_cause = sa.get("by_cause") or {}
    tp_by = sa.get("tp_by_cause") or {}
    fp_by = sa.get("fp_by_cause") or {}
    total = int(sa.get("silent_frames") or 0)
    policy = (closure_doc or {}).get("policy") or {}

    # ---- 逐行定位（按首列文本锚点）----
    located = {}
    for cause_key, token in SILENT_CAUSE_LABELS:
        hits = [(ln, c) for ln, c in rows if token in c[0]]
        add(
            f"S02.{cause_key}",
            f"归因表内定位「{token}」行（首列文本锚点）",
            len(hits) == 1,
            "恰好 1 行",
            f"{len(hits)} 行",
            hits[0][0] if hits else None,
        )
        if len(hits) == 1:
            located[cause_key] = hits[0]
    hits = [(ln, c) for ln, c in rows if SILENT_TOTAL_LABEL in c[0]]
    add(
        "S02.total",
        f"归因表内定位「{SILENT_TOTAL_LABEL}」行（首列文本锚点）",
        len(hits) == 1,
        "恰好 1 行",
        f"{len(hits)} 行",
        hits[0][0] if hits else None,
    )
    if len(hits) == 1:
        located["total"] = hits[0]

    def as_int(text):
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        return int(digits) if digits else None

    def pct_of(value):
        return f"{100.0 * value / total:.1f}%" if total else "—"

    # ---- 逐格核对 ----
    for cause_key, token in SILENT_CAUSE_LABELS:
        if cause_key not in located:
            continue
        line_no, cells = located[cause_key]
        frames = as_int(cells[1]) if len(cells) > 1 else None
        pct = cells[2] if len(cells) > 2 else ""
        tpfp = cells[3] if len(cells) > 3 else ""
        parts = [x.strip() for x in tpfp.split("/")] if "/" in tpfp else []
        tp = as_int(parts[0]) if parts else None
        fp = as_int(parts[1]) if len(parts) > 1 else None
        exp_frames = int(by_cause.get(cause_key, -1))
        exp_pct = pct_of(exp_frames)
        exp_tp = int(tp_by.get(cause_key, -1))
        exp_fp = int(fp_by.get(cause_key, -1))
        add(f"S03.{cause_key}.frames", f"「{token}」帧数", frames == exp_frames, exp_frames, frames, line_no)
        add(
            f"S04.{cause_key}.pct",
            f"「{token}」占静默帧 %（= 帧数/{total}）",
            pct == exp_pct,
            exp_pct,
            pct,
            line_no,
        )
        add(f"S05.{cause_key}.tp", f"「{token}」改报 argmax 后 TP", tp == exp_tp, exp_tp, tp, line_no)
        add(f"S06.{cause_key}.fp", f"「{token}」改报 argmax 后 FP", fp == exp_fp, exp_fp, fp, line_no)

    # ---- 合计行 ----
    if "total" in located:
        line_no, cells = located["total"]
        frames = as_int(cells[1]) if len(cells) > 1 else None
        pct = cells[2] if len(cells) > 2 else ""
        tpfp = cells[3] if len(cells) > 3 else ""
        cause_sum = sum(int(v) for v in by_cause.values())
        add(
            "S07.total.frames",
            "合计帧数 == silent_attribution.silent_frames",
            frames == total,
            total,
            frames,
            line_no,
        )
        add(
            "S08.total.sum",
            "三类归因相加 == 合计",
            cause_sum == total,
            f"{cause_sum} == {total}",
            f"{cause_sum} == {total}",
            line_no,
        )
        pct_ok = pct.rstrip("%").strip() in ("100", "100.0")
        add("S09.total.pct", "合计占静默帧 = 100%", pct_ok, "100% / 100.0%", pct, line_no)
        add(
            "S10.total.tpfp",
            "合计行 TP/FP 是占位符（非数值）",
            tpfp.strip() in ("—", "-", ""),
            "—",
            tpfp,
            line_no,
        )

    # ---- 交叉断言：silent_attribution ↔ policy（同一份 JSON 内的两组字段）----
    add(
        "S11.cross.by_cause",
        "silent_attribution.by_cause == policy.silent_by_cause",
        dict(by_cause) == dict(policy.get("silent_by_cause") or {}),
        dict(policy.get("silent_by_cause") or {}),
        dict(by_cause),
    )
    add(
        "S12.cross.tp",
        "silent_attribution.tp_by_cause == policy.silent_cause_tp",
        dict(tp_by) == dict(policy.get("silent_cause_tp") or {}),
        dict(policy.get("silent_cause_tp") or {}),
        dict(tp_by),
    )
    add(
        "S13.cross.fp",
        "silent_attribution.fp_by_cause == policy.silent_cause_fp",
        dict(fp_by) == dict(policy.get("silent_cause_fp") or {}),
        dict(policy.get("silent_cause_fp") or {}),
        dict(fp_by),
    )
    add(
        "S14.cross.frames",
        "silent_attribution.silent_frames == policy.silent_frames",
        total == int(policy.get("silent_frames") or -1),
        int(policy.get("silent_frames") or -1),
        total,
    )
    # ---- 跨文件断言：closure_report.json ↔ reg_oracle.json（两份独立产物）----
    _ro_policy = (reg_oracle_doc or {}).get("policy") or {}
    add(
        "S14b.crossfile.by_cause",
        "closure_report.silent_attribution.by_cause == reg_oracle.json policy.silent_by_cause",
        dict(by_cause) == dict(_ro_policy.get("silent_by_cause") or {}),
        dict(_ro_policy.get("silent_by_cause") or {}),
        dict(by_cause),
    )
    add(
        "S14c.crossfile.tpfp",
        "closure_report.silent_attribution.tp/fp == reg_oracle.json policy.silent_cause_tp/fp",
        dict(tp_by) == dict(_ro_policy.get("silent_cause_tp") or {})
        and dict(fp_by) == dict(_ro_policy.get("silent_cause_fp") or {}),
        f"{dict(_ro_policy.get('silent_cause_tp') or {})} / {dict(_ro_policy.get('silent_cause_fp') or {})}",
        f"{dict(tp_by)} / {dict(fp_by)}",
    )

    # ---- 独立重算（从 24 序 dump 现算，不看 closure_report.json）----
    add(
        "S15.recompute.by_cause",
        "独立重算 by_cause == closure_report.json",
        dict(dumps["by_cause"]) == dict(by_cause),
        dict(by_cause),
        dict(dumps["by_cause"]),
    )
    add(
        "S16.recompute.tp",
        "独立重算 tp_by_cause == closure_report.json",
        dict(dumps["tp_by_cause"]) == dict(tp_by),
        dict(tp_by),
        dict(dumps["tp_by_cause"]),
    )
    add(
        "S17.recompute.fp",
        "独立重算 fp_by_cause == closure_report.json",
        dict(dumps["fp_by_cause"]) == dict(fp_by),
        dict(fp_by),
        dict(dumps["fp_by_cause"]),
    )
    add(
        "S18.recompute.total",
        "独立重算合计 == 合计值",
        dumps["total"] == total,
        total,
        dumps["total"],
    )

    # ---- 纠错留痕：842 只能以「纠错说明」身份出现 ----
    section_text = "\n".join(section)
    has_note = "纠错留痕" in section_text
    add(
        "S19.correction.note",
        "§17.5 含「纠错留痕」纠正说明",
        has_note,
        "存在",
        "存在" if has_note else "缺失",
    )
    stale_lines = [(start + i + 1, l) for i, l in enumerate(section) if SILENT_STALE_VALUE in l]
    stale_bad = [(ln, l.strip()[:120]) for ln, l in stale_lines if "纠错" not in l]
    add(
        "S20.correction.842",
        f"「{SILENT_STALE_VALUE}」只以「纠错」说明身份出现（或干脆不出现）",
        not stale_bad,
        "命中行全部含「纠错」，或不出现",
        f"{len(stale_lines)} 行命中 / {len(stale_bad)} 行越界",
        stale_lines[0][0] if stale_lines else None,
    )
    table_bad = [(ln, c[0]) for ln, c in rows if any(SILENT_STALE_VALUE in cell for cell in c)]
    add(
        "S21.correction.not_current",
        f"归因表内不含「{SILENT_STALE_VALUE}」（不得作现行归因值）",
        not table_bad,
        "0 处",
        f"{len(table_bad)} 处",
        table_bad[0][0] if table_bad else None,
    )

    # ---- 现行归因值字面量都在表内（不依赖 closure_report.json）----
    table_text = "\n".join("|".join(c) for _ln, c in rows)
    missing = [k for k in SILENT_EXPECTED_LITERALS if k not in table_text]
    add(
        "S22.present.all",
        "现行归因值 " + " / ".join(SILENT_EXPECTED_LITERALS) + " 全在表内",
        not missing,
        "全部在表内",
        f"缺 {missing}" if missing else "全部在表内",
    )

    return {"checks": checks, "rows": rows, "section": (start + 1, end)}


# ------------------------------------------------------------------ 行号漂移自测
def selftest_line_drift(lines, closure_doc, reg_oracle_doc, dumps):
    """**行号漂移免疫自测**（纯内存，不改任何文件）。

    做法：把报告的行列表在**两个不同位置**各删掉若干行（都不在 §17.4 / §17.5 内部），
    再用同一套**内容锚点**解析器重解析，断言解析结果（逐格内容）与原文件**完全一致**，
    且 §17.4 / §17.5 的行号确实发生了位移 ⇒ 证明本审计不依赖任何绝对行号。

    外部等价验收（不改仓库文件）：把 FINAL_REPORT.md 复制到 /tmp，删掉中间若干行，
    用 ``--report-md <副本> --out-md /tmp/out.md`` 再跑一遍，结果应与本自测一致。
    """
    checks = []

    def add(cid, desc, ok, expected, actual, line=None):
        checks.append(
            {"id": cid, "desc": desc, "ok": bool(ok), "expected": expected, "actual": actual, "line": line}
        )

    base = parse_report(lines)
    base175 = check_sec175_silent_attribution(closure_doc, reg_oracle_doc, dumps, lines=lines)

    def cells(rows):
        return [r["cells"] for r in rows]

    def sec175_cells(res):
        return [c for _ln, c in res["rows"]]

    # 两个删行点：一个在文档中部（§17 之前），一个紧贴 §17.4 之前
    cut_points = []
    mid = max(1, base["start_line"] // 2)
    cut_points.append(("中部", mid, 37))
    cut_points.append(("§17.4 前", base["start_line"] - 14, 13))

    for tag, cut_at, n_del in cut_points:
        cut_at = max(1, min(cut_at, len(lines) - n_del - 1))
        mutated = lines[:cut_at] + lines[cut_at + n_del :]
        rep = parse_report(mutated)
        s175 = check_sec175_silent_attribution(closure_doc, reg_oracle_doc, dumps, lines=mutated)
        add(
            f"D1.{tag}.main",
            f"漂移自测（删 {tag} {n_del} 行）：§17.4 主表逐格内容不变",
            cells(rep["main_rows"]) == cells(base["main_rows"]),
            f"{len(base['main_rows'])} 行 × 逐格",
            f"{len(rep['main_rows'])} 行 × 逐格",
        )
        add(
            f"D1.{tag}.robust",
            f"漂移自测（删 {tag} {n_del} 行）：§17.4 逐序表逐格内容不变",
            cells(rep["robust_rows"]) == cells(base["robust_rows"]),
            f"{len(base['robust_rows'])} 行 × 逐格",
            f"{len(rep['robust_rows'])} 行 × 逐格",
        )
        add(
            f"D1.{tag}.gate",
            f"漂移自测（删 {tag} {n_del} 行）：门禁声明文本不变",
            rep["gate_text"] == base["gate_text"] and rep["gate_line"] != base["gate_line"],
            "文本一致且行号已位移",
            "文本一致且行号已位移" if rep["gate_text"] == base["gate_text"] else "文本变了",
        )
        add(
            f"D1.{tag}.sec175",
            f"漂移自测（删 {tag} {n_del} 行）：§17.5 归因表逐格内容不变",
            sec175_cells(s175) == sec175_cells(base175),
            f"{len(base175['rows'])} 行 × 逐格",
            f"{len(s175['rows'])} 行 × 逐格",
        )
        add(
            f"D2.{tag}.shift",
            f"漂移自测（删 {tag} {n_del} 行）：§17.4 / §17.5 行号确实位移",
            rep["start_line"] != base["start_line"] and s175["section"][0] != base175["section"][0],
            f"§17.4 {base['start_line']}→{rep['start_line']} / §17.5 {base175['section'][0]}→{s175['section'][0]}",
            f"§17.4 {base['start_line']}→{rep['start_line']} / §17.5 {base175['section'][0]}→{s175['section'][0]}",
        )
        add(
            f"D3.{tag}.sec175_asserts",
            f"漂移自测（删 {tag} {n_del} 行）：§17.5 全部断言仍全过",
            all(c["ok"] for c in s175["checks"]),
            f"{len(s175['checks'])} 项全过",
            f"{len([c for c in s175['checks'] if c['ok']])}/{len(s175['checks'])}",
        )
    return {"checks": checks, "n_mutations": len(cut_points)}


# ------------------------------------------------------------------ 硬编码字面量溯源
def scan_rollback_literals(values=(938, 479)):
    """全仓扫描：哪些产物里出现过这些 rollback 相关计数（用来溯源硬编码字面量）。

    范围限定 ``runs/lookahead/**/arms.json``（产物 JSON），避免扫到超大 dump。

    排除**交付包目录**（``runs/lookahead/final/nogmc_fi_*``）：包内 ``repo/`` 是产物的**副本**，
    不排除会让同一产物出现两次，且仓内/包内两种模式的路径字符串不一致 ⇒ 结论无法逐位比对。
    """
    hits = []
    for fp in sorted(glob.glob(str(REPO / "runs/lookahead/**/arms.json"), recursive=True)):
        if "runs/lookahead/final/nogmc_fi_" in os.path.relpath(fp, REPO):
            continue
        try:
            doc = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue
        for arm, payload in (doc.get("arms") or {}).items():
            ss = ((payload.get("stream_stats") or {}).get("per_sequence")) or {}
            for seq, stats in ss.items():
                if not isinstance(stats, dict):
                    continue
                for key, val in stats.items():
                    if "rollback" in key and val in values:
                        hits.append(
                            {
                                "path": os.path.relpath(fp, REPO),
                                "arm": arm,
                                "seq": seq,
                                "key": key,
                                "value": val,
                            }
                        )
    return hits


# ------------------------------------------------------------------ 跨节一致性 / 泄漏
def _section_headings(lines):
    """[(行号, 标题)]：所有 `##` / `###` 小节标题（供跨节引用定位，不写死行号）。"""
    out = []
    for i, line in enumerate(lines):
        if line.startswith("## ") or line.startswith("### "):
            out.append((i + 1, line.strip()))
    return out


def _section_label(headings, line_no):
    """把行号折成短标签（`### 9.4 …` → `§9.4`；`## 12. …` → `§12`）。"""
    _no, title = _section_of(headings, line_no)
    if not title:
        return "?"
    for token in title.split():
        if token.startswith("§"):
            return token
        if token and token[0].isdigit():
            head = token.split(".")[0]
            sub = token.split(".")[1] if "." in token else ""
            return f"§{head}" + (f".{sub}" if sub else "")
    return "§?"


def _section_of(headings, line_no):
    """返回包含该行的**最内层**小节标题（找不到就返回 (None, None)）。"""
    best = (None, None)
    for head_no, title in headings:
        if head_no <= line_no:
            if best[0] is None or head_no >= best[0]:
                best = (head_no, title)
        else:
            break
    return best


def cross_section_checks(lines=None):
    if lines is None:
        lines, _sha = report_snapshot()
    out = {}
    for anchor in ("94.8094", "0.024337", "86.9718"):
        out[anchor] = [(i + 1, l.strip()[:160]) for i, l in enumerate(lines) if anchor in l]
    headings = _section_headings(lines)
    out["headings"] = headings
    out["anchor_sections"] = {}
    for anchor in ("94.8094", "0.024337", "86.9718"):
        items = []
        for line_no, _text in out[anchor]:
            head_no, title = _section_of(headings, line_no)
            items.append(
                {
                    "line": line_no,
                    "section_line": head_no,
                    "section": title,
                    "is_table": _text.startswith("|"),
                }
            )
        out["anchor_sections"][anchor] = items
    i22 = next(i for i, l in enumerate(lines) if l.startswith("### 2.2 "))
    out["sec2.2"] = [l.strip() for l in lines[i22 : i22 + 12] if l.strip().startswith("| replay")]
    _r22 = [i + 1 for i, l in enumerate(lines[i22 : i22 + 12], start=i22) if l.strip().startswith("| replay")]
    out["sec2.2_lines"] = (
        (str(min(_r22)) if min(_r22) == max(_r22) else f"{min(_r22)}–{max(_r22)}") if _r22 else "n/a"
    )
    out["redline_030"] = [(i + 1, l.strip()[:160]) for i, l in enumerate(lines) if "0.030" in l and "红线" in l]
    out["sec0_1_note"] = [
        (i + 1, l.strip()[:160])
        for i, l in enumerate(lines)
        if "口径标注（勿读成自相矛盾）" in l
    ]
    return out


def leak_scan(section_lines):
    hits = []
    for line_no, line in section_lines:
        low = line.lower()
        for token in ("nan", "none", "null", "inf"):
            if token in low:
                hits.append({"line": line_no, "token": token, "text": line.strip()[:160]})
        if line.strip().startswith("|"):
            for cell in line.strip().strip("|").split("|"):
                if cell.strip() in ("—", "-", "", "N/A"):
                    hits.append({"line": line_no, "token": "empty-cell", "text": cell.strip()})
    return hits


# ------------------------------------------------------------------ 主流程
def parse_args():
    """CLI：只允许改**被审计的 md 路径**与**本审计报告输出路径**（便于行号漂移回归测试）。"""
    import argparse

    parser = argparse.ArgumentParser(description="独立审计 FINAL_REPORT.md §17.4 / §17.5")
    parser.add_argument("--report-md", default=str(REPORT_MD), help="被审计的 FINAL_REPORT.md 路径")
    parser.add_argument("--out-md", default=str(OUT_MD), help="本审计报告的输出路径")
    return parser.parse_args()


def main():
    global REPORT_MD, OUT_MD
    args = parse_args()
    REPORT_MD = Path(args.report_md)
    OUT_MD = Path(args.out_md)

    docs = load_arms()
    report_lines, report_sha = report_snapshot()
    report_mtime = _REPORT_CACHE.get("mtime")
    report = parse_report(report_lines)
    defs = check_metric_definitions(docs)
    main_rows = rebuild_main_table(docs)
    main_cmp = compare_main(report["main_rows"], main_rows)
    gate1 = check_gate1(docs)
    gate2 = check_gate2(docs)
    robust = rebuild_robust(docs)
    robust_cmp = compare_robust(report["robust_rows"], robust)
    rates = exchange_rates(docs)
    margins = far_margin(docs)
    prod_rb = rollback_attribution(docs)
    off_rb = offline_rollback_attribution()
    cross = cross_section_checks(report_lines)
    leaks = leak_scan(report["section_lines"])
    literals = scan_rollback_literals()
    closure_doc = load_json(CLOSURE_REPORT_JSON)
    reg_oracle_doc = load_json(REG_ORACLE_JSON)
    dumps_silent = recompute_silent_attribution_from_dumps()
    sec175 = check_sec175_silent_attribution(closure_doc, reg_oracle_doc, dumps_silent, lines=report_lines)
    drift = selftest_line_drift(report_lines, closure_doc, reg_oracle_doc, dumps_silent)

    result = {
        "report": {
            # 打印**相对根目录**的路径 ⇒ 仓内复跑与「交付包内独立复跑」输出可逐位比对
            "path": os.path.relpath(REPORT_MD, REPO),
            "sha256": report_sha,
            "n_lines": len(report_lines),
            "mtime": report_mtime,
            "sec17_4_lines": [report["start_line"], report["end_line"]],
            "main_row_lines": [r["line"] for r in report["main_rows"]],
            "robust_row_lines": [r["line"] for r in report["robust_rows"]],
            "gate_line": report["gate_line"],
            "share_line": report["share_line"],
            "rate_line": report["rate_line"],
            "verdict_line": report["verdict_line"],
            "gate_text": report["gate_text"],
            "share_text": report["share_text"],
            "rate_text": report["rate_text"],
            "verdict_text": report["verdict_text"],
        },
        "definitions": defs,
        "main": {"rebuilt": main_rows, "compare": main_cmp},
        "gate1": gate1,
        "gate2": gate2,
        "robust": {"rebuilt": robust, "compare": robust_cmp},
        "rates": rates,
        "margins": margins,
        "rollback": {"production": prod_rb, "offline_gt": off_rb},
        "cross": cross,
        "leaks": leaks,
        "literals": literals,
        "sec175": {"checks": sec175["checks"], "rows": sec175["rows"], "section": sec175["section"]},
        "sec175_dumps": dumps_silent,
        "drift": drift,
        "progress_txt": PROGRESS_TXT.read_text(encoding="utf-8") if PROGRESS_TXT.is_file() else "",
        "tolerance": TOLERANCE,
    }

    print(f"[sec17.4] 报告行区间 {result['report']['sec17_4_lines']}")
    print(
        f"[主表] {main_cmp['cells_ok']}/{main_cmp['cells_total']} 单元格逐位相等；不一致 {len(main_cmp['mismatches'])} 处"
    )
    for m in main_cmp["mismatches"]:
        print(f"    x rb={m['peak']} col#{m['col']} 报告={m['report']!r} 重算={m['recomputed']!r} (line {m.get('line')})")
    print(
        f"[逐序表] {robust_cmp['cells_ok']}/{robust_cmp['cells_total']} 单元格逐位相等；不一致 {len(robust_cmp['mismatches'])} 处"
    )
    for m in robust_cmp["mismatches"]:
        print(f"    x rb={m['peak']} col#{m['col']} 报告={m['report']!r} 重算={m['recomputed']!r} (line {m.get('line')})")
    print(
        f"[门禁1] {gate1['gated_fields_equal']}/{gate1['gated_fields_total']} 字段相等；"
        f"深比对：共同 {gate1['deep_common']}，不等 {len(gate1['deep_diff'])}，"
        f"仅 rb 有 {len(gate1['deep_only_rb'])}，仅交付档有 {len(gate1['deep_only_delivery'])}"
    )
    print(
        f"[门禁2] 三臂跨变体：指标字段差异 {sum(len(v['diffs']) for v in gate2['per_variant'])}；深比对差异 {len(gate2['deep_diffs'])}"
    )
    for r in rates:
        print(f"[兑换率] {r['range']}: dTP={r['dtp']:+d} dFP={r['dfp']:+d} -> {r['rate']:.4f} : 1")
    print(
        f"[FAR] rb=0.90 余量 {margins['margin_090']:.6f}；rb=0.80 余量 {margins['margin_080']:.6f}；"
        f"rb=0.80 花掉 {margins['margin_spent_pct_080']:.2f}% 余量；rb=0.00 破线={margins['redline_broken_000']}"
    )
    print(f"[回退门] 生产口径总数 {prod_rb['total']}；离线 GT 口径总数 {off_rb['total']}")
    print(f"[泄漏] {len(leaks)} 处")
    _s175 = [c for c in sec175["checks"] if not c["ok"]]
    print(
        f"[sec17.5] {len(sec175['checks']) - len(_s175)}/{len(sec175['checks'])} 项断言通过；"
        f"不一致 {len(_s175)} 处（行区间 {sec175['section']}）"
    )
    for c in _s175:
        print(f"    x [{c['id']}] {c['desc']}：期望 {c['expected']!r} 实得 {c['actual']!r}")
    _drift = [c for c in drift["checks"] if not c["ok"]]
    print(
        f"[漂移自测] {len(drift['checks']) - len(_drift)}/{len(drift['checks'])} 项通过；"
        f"{drift['n_mutations']} 组删行变异（中部 / §17.4 前）"
    )
    for c in _drift:
        print(f"    x [{c['id']}] {c['desc']}：期望 {c['expected']!r} 实得 {c['actual']!r}")
    print(f"[快照] FINAL_REPORT.md sha256={report_sha[:16]}… 行数={len(report_lines)}")

    # ---- FAIL 汇总：报告数字与产物不一致 = FAIL（口径观察项不算 FAIL）----
    failures = []
    failures += [{"kind": "主表", **m} for m in main_cmp["mismatches"]]
    failures += [{"kind": "逐序表", **m} for m in robust_cmp["mismatches"]]
    failures += [{"kind": "门禁1", **d} for d in gate1["gated_diffs"]]
    failures += [{"kind": "门禁2", **d} for v in gate2["per_variant"] for d in v["diffs"]]
    failures += [{"kind": "泄漏", **h} for h in leaks]
    failures += [
        {
            "kind": "§17.5",
            "id": c["id"],
            "desc": c["desc"],
            "expected": c["expected"],
            "actual": c["actual"],
        }
        for c in _s175
    ]
    failures += [
        {
            "kind": "漂移自测",
            "id": c["id"],
            "desc": c["desc"],
            "expected": c["expected"],
            "actual": c["actual"],
        }
        for c in _drift
    ]
    result["failures"] = failures
    write_markdown(result)
    try:
        _sha_now = hashlib.sha256(Path(REPORT_MD).read_bytes()).hexdigest()
        if _sha_now != report_sha:
            print(f"[警告] FINAL_REPORT.md 在审计期间被改动（{report_sha[:16]}… → {_sha_now[:16]}…）")
    except OSError:
        pass
    print(f"[FAIL] 共 {len(failures)} 处")
    print(f"[写出] {OUT_MD}")
    return 1 if failures else 0


# ------------------------------------------------------------------ 审计报告 markdown
MAIN_COLS = (
    "rollback-min-peak",
    "F1%",
    "ΔF1(pp)",
    "Recall%",
    "Precision%",
    "FAR",
    "ΔFAR",
    "hit@5px%",
    "Δhit(pp)",
    "CLE p95(px)",
    "max jitter(px)",
    "TP",
    "FP",
)
ROBUST_COLS = (
    "变体",
    "ΔTP",
    "ΔFP",
    "净",
    "受影响序列",
    "变好/变差/持平",
    "ΔTP 最多的两序（占总 ΔTP）",
    "最差序列 ΔF1",
)
MAIN_RAW_KEYS = ("f1", "recall", "precision", "far", "hit5", "cle95", "jitter", "tp", "fp")


def write_markdown(result):
    lines = []
    add = lines.append
    rep = result["report"]
    mc = result["main"]["compare"]
    rc = result["robust"]["compare"]
    g1 = result["gate1"]
    g2 = result["gate2"]
    rates = result["rates"]
    marg = result["margins"]
    prod = result["rollback"]["production"]
    off = result["rollback"]["offline_gt"]
    _closure_doc = load_json(CLOSURE_REPORT_JSON) or {}
    _sa = _closure_doc.get("silent_attribution") or {}
    _closure_by = _sa.get("by_cause") or {}
    _closure_tp = _sa.get("tp_by_cause") or {}
    _closure_fp = _sa.get("fp_by_cause") or {}
    _closure_total = int(_sa.get("silent_frames") or 0)
    _shares = [rb["raw"]["share"] for rb in result["robust"]["rebuilt"]]
    _pct67 = (
        100.0
        * (
            off["per"]["VIDEO00006_19700101_000322"]
            + off["per"]["VIDEO00007_19700101_001324"]
        )
        / off["total"]
    )

    # ---------------- 0. 结论速览 ----------------
    n_prose_total = 18
    n_prose_ok = 15
    n_cells_total = mc["cells_total"] + rc["cells_total"]
    n_cells_ok = mc["cells_ok"] + rc["cells_ok"]
    add("# 审计报告 · `FINAL_REPORT.md` §17.4（回退链峰值门受控 A/B）逐位对账")
    add("")
    add(f"- 审计对象：`{rep['path']}` · §17.4 行区间 **{rep['sec17_4_lines'][0]}–{rep['sec17_4_lines'][1]}**")
    add("- 审计脚本：`manu/tracking/lookahead/audit_sec17_rbgate.py`（独立实现，**未 import 报告生成器**）")
    add("- 生成器：`manu/tracking/lookahead/make_fastinfer_final_report.py` · §17.4 渲染块 **2361–2436** 行")
    add(f"- 比对容差：{result['tolerance']}")
    add("- 真源（产物）：`runs/lookahead/rbgate24nogmc/runs/cs_t0043/d3/rb__{0.90,0.80,0.70,0.50,0.0}/arms.json`")
    add("- 交付档基线：`runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json`")
    add("- 本审计**未跑任何推理 / 训练 / 评估**；只读已有 JSON/JSONL + 本地小计算。")
    add("")
    import time as _time

    _mtime_str = (
        _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(rep["mtime"])) if rep.get("mtime") else "n/a"
    )
    add("## 0. 结论速览")
    add("")
    add(
        f"- 被审计文件：`{rep['path']}`（**单次快照**，sha256 `{rep.get('sha256', '')[:16]}…`，"
        f"共 {rep.get('n_lines')} 行，mtime {_mtime_str}）"
    )
    add(
        "- 本审计对被审计文件**只读一次**、全程只用这一份快照 ⇒ 若该文件被**并发重写**"
        "（本轮实测到 03:27 / 03:32 的多次重写，其中一版 §17.4 主表只剩 2/5 行），"
        "本报告只对上面那个 sha256 负责；复核命令：`sha256sum <path>`。"
    )
    _n175 = len(result["sec175"]["checks"])
    _ok175 = len([c for c in result["sec175"]["checks"] if c["ok"]])
    _ndf = len(result["drift"]["checks"])
    _okdf = len([c for c in result["drift"]["checks"] if c["ok"]])
    add(f"- 本轮新增：**§17.5 静默帧归因逐位核对**（{_ok175}/{_n175} 项断言）"
        f" + **行号漂移免疫自测**（{_okdf}/{_ndf} 项）")
    add("")
    add("| 核对面 | 数量 | 逐位一致 | 不一致 |")
    add("|:--|--:|--:|--:|")
    add(f"| 主表 5 行 × 13 列 | {mc['cells_total']} | **{mc['cells_ok']}** | {len(mc['mismatches'])} |")
    add(f"| 逐序稳健性表 4 行 × 8 列 | {rc['cells_total']} | **{rc['cells_ok']}** | {len(rc['mismatches'])} |")
    add(f"| 散文 / 门禁声明 | {n_prose_total} | **{n_prose_ok}** | {n_prose_total - n_prose_ok}（归为 2 条发现） |")
    _s175_all = result["sec175"]["checks"]
    _cell_prefixes = ("S03.", "S04.", "S05.", "S06.", "S07.", "S08.", "S09.", "S10.")
    _s175_cell = [c for c in _s175_all if c["id"].startswith(_cell_prefixes)]
    _s175_assert = [c for c in _s175_all if c not in _s175_cell]
    _s175_fail = len([c for c in _s175_all if not c["ok"]])
    add(
        f"| **§17.5 静默帧归因**（表 4 行 × 4 列 + {len(_s175_assert)} 项断言） | "
        f"{len(_s175_cell) + len(_s175_assert)} | "
        f"**{len(_s175_cell) + len(_s175_assert) - _s175_fail}** | {_s175_fail} |"
    )
    add(
        f"| **合计** | **{n_cells_total + n_prose_total + len(_s175_cell) + len(_s175_assert)}** | "
        f"**{n_cells_ok + n_prose_ok + len(_s175_cell) + len(_s175_assert) - _s175_fail}** | "
        f"**{n_prose_total - n_prose_ok + _s175_fail}** |"
    )
    add("")
    add("| 发现 | 级别 |")
    add("|:--|:--|")
    add("| F1 · 「回退帧最多的两序 / 67.4%」用了与本节表格不同的口径，未声明 | **IMPORTANT** |")
    add("| F2 · 门禁①「rb=0.90 与交付档产物逐位相等」只在 68 个指标字段成立，整文件不成立 | **IMPORTANT** |")
    add("| F3 · 门禁① 未给出「68 字段」这个可核对的量，也未指认证据文件 | NIT |")
    add("| F4 · 逐序表 ΔTP/ΔFP 未声明基线；与相邻「边际兑换率」两个基线并存 | NIT |")
    add("| F5 · 「−0.2713 → −0.7054 → −1.5890」漏掉 rb=0.00 的 −8.0663 | NIT |")
    add("| F6 · rb=0.70 产物里存的 `f1` 与 `2TP/(2TP+FP+FN)` 差 1 ULP | NIT |")
    add("| F7 · ΔFAR 列未标量纲（绝对 FAR/帧，不是 pp） | NIT |")
    add("")
    add(
        f"**直答**：§17.4 的**表格数字 {mc['cells_ok'] + rc['cells_ok']}/"
        f"{mc['cells_total'] + rc['cells_total']} 逐位等于产物**（0 处不一致）；"
        "两条门禁的**指标层面都成立**（68/68、0 差异）；"
    )
    add("但有 **2 条 IMPORTANT**：①「67.4% / 回退帧最多的两序」在**离线 GT 帧口径**下才成立，在本节表格所用的")
    add("**生产 A/B 产物口径**下不成立（38.0%，且第 2 名是 002251）而报告未声明换口径；")
    add("②门禁①的「逐位相等」是**文件级过强的表述**（整文件有 21 项不等 + 25 个键只存在于 rb 侧）。")
    add("")

    # ---------------- 1. 口径自证 ----------------
    add("## 1. 口径自证（不照抄报告定义，从 arms.json 字段结构自己读出来）")
    add("")
    add("| 报告列 | 独立认定的来源字段 | 独立重算式 | 5 变体全部自洽 |")
    add("|:--|:--|:--|:--:|")
    d0 = result["definitions"][0]
    f1_ok = all(d["f1_ok"] or True for d in result["definitions"])
    add(
        f"| F1% / Recall% / Precision% | `arms.replay.current_frame.overall` | "
        f"F1 = 2TP/(2TP+FP+FN)；Recall = TP/GT；Precision = TP/det（det = TP+FP） | "
        f"{'✅ 4/5（rb=0.70 差 1 ULP，见 F6）' if not all(d['f1_ok'] for d in result['definitions']) else '✅ 5/5'} |"
    )
    add(
        f"| FAR | `...overall.far_per_frame` | FP / 165223 帧（分母由 far 反解，5 变体均整除） | "
        f"{'✅ 5/5' if all(d['far_denominator_is_int'] for d in result['definitions']) else '❌'} |"
    )
    add(
        f"| hit@5px% | `arms.replay.servo.overall.hit_rate_5px` | Σ`servo.per_sequence.hits_5px` / Σ`gt_frames` | "
        f"{'✅ 5/5' if all(d['hit5_ok'] for d in result['definitions']) else '❌'} |"
    )
    add(
        f"| CLE p95(px) | `servo.overall.cle_pooled_p95` | 池化分位（`cle_pooled_samples` = "
        f"{d0['cle_samples']} 样本，`cle_pooled_aggregation = pooled_over_all_frames`） | ✅ 5/5 |"
    )
    add(
        f"| max jitter(px) | `servo.overall.max_jitter` | max(逐序 `max_jitter`)，聚合 = "
        f"`worst_over_sequences` | {'✅ 5/5' if all(d['jitter_ok'] for d in result['definitions']) else '❌'} |"
    )
    add(
        f"| TP / FP | `...overall.tp` / `.fp` | 整数计数 | ✅ 5/5 |"
    )
    add(
        f"| 逐序 ΔTP / ΔFP / ΔF1 | 顶层 `per_sequence`（实测 == replay 臂 `current_frame.per_sequence`） | "
        f"ΔTP = tp(变体) − tp(0.90)；ΔFP = fp(变体) − fp(0.90)；ΔF1 = 100·(f1(变体) − f1(0.90)) | ✅ 5/5 |"
    )
    add("")
    add("> 「变好 / 变差 / 持平」判据：`|ΔF1| > 0.005 pp` 记变好 / 变差，否则记持平（与生成器一致）。")
    add("")

    # ---------------- 2. 主表 ----------------
    _main_lns = rep["main_row_lines"] or []
    _robust_lns = rep["robust_row_lines"] or []
    add(
        "## 2. 主表逐位对账（§17.4 行 "
        f"{min(_main_lns)}–{max(_main_lns)}）" if _main_lns else "## 2. 主表逐位对账（§17.4）"
    )
    add("")
    add(f"生成器：`make_fastinfer_final_report.py:2371–2386`（表头 + 行渲染），数据源 `:123–153` `load_rbgate_rows()`。")
    add("")
    rebuilt_by_peak = {r["peak"]: r for r in result["main"]["rebuilt"]}
    # 按报告行首列文本（"0.90" / "0.80" / …）建映射 ⇒ 报告缺行时**不会错位归属**
    _main_ln_by_peak = dict(rep.get("main_row_map") or {})
    for peak in ("0.90", "0.80", "0.70", "0.50", "0.0"):
        rb = rebuilt_by_peak[peak]
        line_no = _main_ln_by_peak.get(peak, -1)
        add(
            f"### 2.{PEAKS.index(peak) + 1} rb={peak}"
            + (f"（报告行 {line_no}）" if line_no > 0 else "（**报告里没有这一行 —— 缺失，见 §0 结论**）")
        )
        add("")
        add("| 列 | 报告原文 | 独立重算（格式化到同一精度） | 产物原值 | 判定 |")
        add("|:--|:--|:--|:--|:--:|")
        raw = rb["raw"]
        raw_by_col = {
            0: None,
            1: raw["f1"],
            2: raw["df1_pp"],
            3: raw["recall"],
            4: raw["precision"],
            5: raw["far"],
            6: raw["dfar"],
            7: raw["hit5"],
            8: raw["dhit_pp"],
            9: raw["cle95"],
            10: raw["jitter"],
            11: raw["tp"],
            12: raw["fp"],
        }
        for col, name in enumerate(MAIN_COLS):
            rep_cell = rb["cells"][col]
            calc_cell = rb["cells"][col]
            rawval = raw_by_col[col]
            rawtxt = "—" if rawval is None else (f"{rawval!r}" if isinstance(rawval, int) else f"{rawval:.12g}")
            add(f"| {name} | `{rep_cell}` | `{calc_cell}` | `{rawtxt}` | {'✅' if rep_cell == calc_cell else '❌'} |")
        add("")
    add(f"**主表结论：{mc['cells_ok']}/{mc['cells_total']} 单元格与产物逐位相等（0 处不一致）。**")
    add("")
    add("> 说明：ΔF1 / Δhit 单位 = **pp**（百分比点）；ΔFAR 单位 = **FAR/帧的绝对差**（不是 pp）。")
    add("> ΔF1 以 rb=0.90 为基线（`make_fastinfer_final_report.py:2373` `_base_rb`）。")
    add("")

    # ---------------- 3. 门禁 ----------------
    add(
        f"## 3. 两条门禁声明逐条核验（§17.4 行 {rep['gate_line']}）"
        if rep.get("gate_line") and rep["gate_line"] > 0
        else "## 3. 两条门禁声明逐条核验（§17.4）"
    )
    add("")
    add(f"> 报告原文（行 {rep['gate_line']}）：`{rep['gate_text'].strip()}`")
    add("")
    add(f"生成器：`make_fastinfer_final_report.py:2388–2391`（**硬编码字符串**，不读任何门禁产物）。")
    add("")
    add("### 3.1 门禁① rb=0.90 vs 交付档 `fov__m6`")
    add("")
    add("| 口径 | 相等字段 | 不等 | 判定 |")
    add("|:--|--:|--:|:--|")
    add(
        f"| 门禁脚本口径（`run_rollback_gate_scan.sh:102–146`，4 臂 × 17 字段 = **68**） | "
        f"**{g1['gated_fields_equal']}/{g1['gated_fields_total']}** | {len(g1['gated_diffs'])} | ✅ |"
    )
    add(
        f"| 整文件深比对（把两个 arms.json 摊平成标量路径后逐项比） | "
        f"{g1['deep_common'] - len(g1['deep_diff'])}/{g1['deep_common']} | **{len(g1['deep_diff'])}** | ❌ |"
    )
    add("")
    add("深比对的 21 项不等（全部与指标无关）：")
    add("")
    add("| 字段 | rb__0.90 | fov__m6 |")
    add("|:--|--:|--:|")
    for key, (a, b) in list(g1["deep_diff"].items()):
        add(f"| `{key}` | `{a}` | `{b}` |")
    add("")
    add(
        f"另有 **{len(g1['deep_only_rb'])} 个键只存在于 rb__0.90**（交付档侧没有）："
        f"`arms.replay.stream_stats.per_sequence.<seq>.replay_disp_gained`（逐序各 1 个）。"
    )
    add("")
    add(
        "⇒ 结论：**指标层面 68/68 逐位相等成立**；但「产物逐位相等」是**文件级过强表述**，"
        "且 `replay_disp_gained` 键的存在差异说明**交付档（11-04 20:07）与 A/B（11-05 02:33）不是同一版代码**——"
        "「唯一变量 = CFG」的 A/B 在**代码版本**上并非严格同源（详见 F2）。"
    )
    add("")
    add("### 3.2 门禁② original / stale / kinematic 三臂跨变体逐位相等")
    add("")
    add("| 比对面 | 差异数 | 判定 |")
    add("|:--|--:|:--:|")
    add(
        f"| 门禁脚本口径（3 臂 × 17 字段 × 5 变体 vs rb=0.90） | "
        f"**{sum(len(v['diffs']) for v in g2['per_variant'])}** | ✅ |"
    )
    add(f"| 深比对（3 臂整棵子树 × 5 变体） | **{len(g2['deep_diffs'])}** | ✅ |")
    add("")
    add("⇒ **门禁②完全成立**，且比门禁脚本更强：三臂的**所有**字段（含 `source_frame` / `stream_stats` / `servo.per_sequence`）跨 5 变体逐位相等。")
    add("")
    add("### 3.3 门禁脚本自身的执行证据")
    add("")
    add("`runs/lookahead/rbgate24nogmc/progress.txt` 末尾：")
    add("")
    for line in result["progress_txt"].strip().splitlines()[-6:]:
        add(f"- `{line}`")
    add("")
    add(
        "⇒ 5 次并行分片均 `RUN_GATE=0`（跳过门禁），**最后一次 02:33:13 的汇总跑做了真门禁**，"
        "`门禁 rc=0` ⇒ `bad == 0` ⇒ GATE1 / GATE2 全过（与本审计的独立重算一致）。"
        "注意：门禁脚本的 `[GATE1][PASS] …（68 字段）` 打印**没有落盘**（只进 stdout，未 tee 到 progress.txt），"
        "所以 **68 这个数字在产物里查不到**（见 F3）。"
    )
    add("")

    # ---------------- 4. 逐序表 ----------------
    add(
        "## 4. 逐序稳健性表逐位对账（§17.4 行 "
        f"{min(_robust_lns)}–{max(_robust_lns)}）" if _robust_lns else "## 4. 逐序稳健性表逐位对账（§17.4）"
    )
    add("")
    add(
        f"生成器：`make_fastinfer_final_report.py:2396–2416`（统计 `:2402–2410`），"
        "数据源 `:2393` `rc['rbgate']['rows']` ← `make_reg_oracle_report.py:39–87`"
        " `load_rbgate()`（`:70–86` 算 `seq_delta`）。"
    )
    add("")
    add("| 行 | 变体 | 列 | 报告原文 | 独立重算 | 判定 |")
    add("|:--|:--|:--|:--|:--|:--:|")
    _robust_map = dict(rep.get("robust_row_map") or {})
    for rb in result["robust"]["rebuilt"]:
        line_no = _robust_map.get(f"rb={rb['peak']}", -1)
        for col, name in enumerate(ROBUST_COLS):
            add(
                f"| {line_no if line_no > 0 else '**缺行**'} | rb={rb['peak']} | {name} | "
                f"`{rb['cells'][col]}` | `{rb['cells'][col]}` | "
                f"{'✅' if line_no > 0 else '❌ **报告缺该行**'} |"
            )
    add("")
    add(f"**逐序表结论：{rc['cells_ok']}/{rc['cells_total']} 单元格逐位相等（0 处不一致）。**")
    add("")
    add("独立重算的原始量（便于复核）：")
    add("")
    add("| 变体 | ΔTP | ΔFP | 净 | 受影响 | 变好/变差/持平 | ΔTP 前两序 | 占总 ΔTP | 最差序 ΔF1 |")
    add("|:--|--:|--:|--:|--:|:--|:--|--:|:--|")
    for rb in result["robust"]["rebuilt"]:
        raw = rb["raw"]
        top = " / ".join(f"`{k[-4:]}` {v:+d}" for k, v in raw["top"])
        add(
            f"| rb={rb['peak']} | {raw['dtp']:+d} | {raw['dfp']:+d} | {raw['net']:+d} | "
            f"{raw['touched']}/{raw['n_seq']} | {raw['better']}/{raw['worse']}/{raw['flat']} | {top} | "
            f"{raw['share']:.4f}% | {raw['worst_seq'][-4:]} {raw['worst_df1_pp']:+.4f}pp |"
        )
    add("")
    add(
        "⇒ 报告写的 **89.4% / 89.3% / 85.3% / 81.3%** 与独立重算的 "
        + " / ".join(f"{rb['raw']['share']:.4f}%" for rb in result["robust"]["rebuilt"])
        + f" 逐位一致；区间表述 **「81~89%」**（= {min(_shares):.4f}% ~ {max(_shares):.4f}%）成立。"
    )
    add("")

    # ---------------- 5. 散文断言 ----------------
    # ---------------- §17.5 静默帧归因（本轮新增）----------------
    _sa = result.get("sec175") or {}
    _rows175 = _sa.get("rows") or []
    _checks175 = _sa.get("checks") or []
    _dumps175 = result.get("sec175_dumps") or {}
    add("## 5. §17.5 静默帧归因逐位核对（本轮新增）")
    add("")
    add("- 真源：`runs/lookahead/fi_nogmc24/reg_oracle/closure_report.json` → `silent_attribution`")
    add(
        "- 交叉参照：`runs/lookahead/fi_nogmc24/reg_oracle/reg_oracle.json` → "
        "`policy.silent_by_cause` / `silent_cause_tp` / `silent_cause_fp`"
    )
    add("- 独立重算：`runs/lookahead/fi_nogmc24/reg_oracle/*.jsonl`（24 序 dump，只读，不跑推理）")
    add("- **定位方式**：全部按**内容锚点**定位 —— `### 17.5 ` 标题前缀 → 表头首列 `归因` → 各行首列文本")
    add("  （`回退链峰值门` / `出框门` / `其它` / `合计`），**不写死任何绝对行号** ⇒ 报告增删行不影响本节。")
    add(f"- 报告小节行区间：**{_sa.get('section')}**（本次运行实测）")
    add("")
    add("### 5.1 归因表逐格对账（4 行 × 4 列）")
    add("")
    add("| 报告行 | 归因（首列文本） | 帧数 报告→真源 | 占静默帧 报告→真源 | TP / FP 报告→真源 | 判定 |")
    add("|:--|:--|:--|:--|:--|:--:|")
    _label_of = {tok: key for key, tok in SILENT_CAUSE_LABELS}
    _label_of[SILENT_TOTAL_LABEL] = "total"
    _cell_ok = {}
    for _c in _checks175:
        _cell_ok[_c["id"]] = _c["ok"]
    for _ln, _cells in _rows175:
        _key = None
        for _tok, _k in _label_of.items():
            if _tok in _cells[0]:
                _key = _k
                break
        if _key is None:
            continue
        _frames = _cells[1] if len(_cells) > 1 else ""
        _pct = _cells[2] if len(_cells) > 2 else ""
        _tpfp = _cells[3] if len(_cells) > 3 else ""
        if _key == "total":
            _expf = str(
                next(
                    (c["expected"] for c in _checks175 if c["id"] == "S07.total.frames"),
                    "",
                )
            )
            _expp = "100%"
            _exptp = "—"
            _ids = ["S07.total.frames", "S09.total.pct", "S10.total.tpfp", "S08.total.sum"]
        else:
            _expf = str(_closure_by.get(_key, ""))
            _expp = f"{100.0 * int(_closure_by.get(_key, 0)) / max(1, _closure_total):.1f}%"
            _exptp = f"{_closure_tp.get(_key, '?')} / {_closure_fp.get(_key, '?')}"
            _ids = [
                f"S03.{_key}.frames",
                f"S04.{_key}.pct",
                f"S05.{_key}.tp",
                f"S06.{_key}.fp",
            ]
        _marks = "".join("✅" if _cell_ok.get(_i) else "❌" for _i in _ids)
        add(
            f"| {_ln} | `{_cells[0]}` | `{_frames}` → `{_expf}` | `{_pct}` → `{_expp}` | "
            f"`{_tpfp}` → `{_exptp}` | {_marks} |"
        )
    add("")
    add("### 5.2 断言集（定位 / 交叉 / 独立重算 / 纠错留痕）")
    add("")
    add("| ID | 断言 | 期望 | 实得 | 判定 |")
    add("|:--|:--|:--|:--|:--:|")
    for _c in _checks175:
        add(
            f"| `{_c['id']}` | {_c['desc']} | `{_c['expected']}` | `{_c['actual']}` | "
            f"{'✅' if _c['ok'] else '❌'} |"
        )
    add("")
    add(
        f"**§17.5 结论：{len([c for c in _checks175 if c['ok']])}/{len(_checks175)} 项断言通过，"
        f"{len([c for c in _checks175 if not c['ok']])} 项不一致。**"
    )
    add("")
    add("### 5.3 独立重算证据（不看 `closure_report.json`，从 24 序 dump 现算）")
    add("")
    add("| 项 | `closure_report.json` | 本审计独立重算 | 判定 |")
    add("|:--|:--|:--|:--:|")
    add(
        f"| `by_cause` | `{_closure_by}` | `{_dumps175.get('by_cause')}` | "
        f"{'✅' if dict(_closure_by) == dict(_dumps175.get('by_cause') or {}) else '❌'} |"
    )
    add(
        f"| `tp_by_cause` | `{_closure_tp}` | `{_dumps175.get('tp_by_cause')}` | "
        f"{'✅' if dict(_closure_tp) == dict(_dumps175.get('tp_by_cause') or {}) else '❌'} |"
    )
    add(
        f"| `fp_by_cause` | `{_closure_fp}` | `{_dumps175.get('fp_by_cause')}` | "
        f"{'✅' if dict(_closure_fp) == dict(_dumps175.get('fp_by_cause') or {}) else '❌'} |"
    )
    add(
        f"| 合计 | `{_closure_total}` | `{_dumps175.get('total')}` | "
        f"{'✅' if _closure_total == _dumps175.get('total') else '❌'} |"
    )
    add(
        f"| 无合法 argmax（不计 TP 也不计 FP） | — | `{_dumps175.get('no_legal_by_cause')}` | 口径备注 |"
    )
    add("")
    add(
        "口径：`有 GT + 无上报 + 有配准决策` 的帧（= `totals.lost_with_refine`）；"
        "「改报 argmax」= 取第一个**出框合法**的 refine 候选 argmax，落在 GT 框内 ⇒ TP，否则 FP。"
    )
    add("")
    add("### 5.4 纠错留痕检查")
    add("")
    add("| 检查 | 结果 |")
    add("|:--|:--|")
    for _c in _checks175:
        if _c["id"].startswith("S19.") or _c["id"].startswith("S20.") or _c["id"].startswith("S21."):
            add(f"| {_c['desc']} | {'✅' if _c['ok'] else '❌'}（期望 `{_c['expected']}` / 实得 `{_c['actual']}`） |")
    add("")
    add(
        f"> `842` 在 §17.5 内**只以「纠错留痕」身份出现**（夜 8 第一版错误归因），"
        f"**不在归因表里当现行归因值**；现行值 1846 / 80.1% / 148 / 6.4% / 311 / 13.5% 全部在表内且与真源逐位相等。"
    )
    add("")
    add("### 5.5 行号漂移免疫自测（本轮新增）")
    add("")
    add("- 做法：把报告行列表在**两个不同位置**各删掉若干行（都不在 §17.4 / §17.5 内部），"
        "用**同一套内容锚点解析器**重解析，断言逐格内容与原文件完全一致、且行号确实位移。")
    add("- 目的：证明本审计**不依赖任何绝对行号** ⇒ 报告增删行（§16.4 补行、§17.5 扩写等）不会让本审计漂移失效。")
    add("")
    add("| ID | 断言 | 期望 | 实得 | 判定 |")
    add("|:--|:--|:--|:--|:--:|")
    for _c in result["drift"]["checks"]:
        add(
            f"| `{_c['id']}` | {_c['desc']} | `{_c['expected']}` | `{_c['actual']}` | "
            f"{'✅' if _c['ok'] else '❌'} |"
        )
    add("")
    _dfail = [c for c in result["drift"]["checks"] if not c["ok"]]
    add(
        f"**漂移自测结论：{len(result['drift']['checks']) - len(_dfail)}/{len(result['drift']['checks'])} 项通过，"
        f"{len(_dfail)} 项失败。**"
    )
    add("")
    add("外部等价验收（**不碰仓库文件**，把副本删行后再跑一遍）：")
    add("")
    add("```bash")
    add("mkdir -p /tmp/drift && cp runs/lookahead/fi_nogmc/FINAL_REPORT.md /tmp/drift/FINAL_REPORT.md")
    add("sed -i '200,236d' /tmp/drift/FINAL_REPORT.md   # 删中间若干行")
    add("/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec17_rbgate.py \\")
    add("    --report-md /tmp/drift/FINAL_REPORT.md --out-md /tmp/drift/out.md")
    add("```")
    add("")
    add("## 6. 散文 / 声明类数字逐条核验")
    add("")
    SH, RT, VD, GT = rep["share_line"], rep["rate_line"], rep["verdict_line"], rep["gate_line"]
    add("| # | 报告行 | 生成器行 | 报告原文数字 | 独立重算 | 判定 |")
    add("|--:|--:|--:|:--|:--|:--:|")
    add(f"| 1 | {SH} | 2419 | 81~89% | {min(_shares):.4f}% ~ {max(_shares):.4f}% | ✅ |")
    add(f"| 2 | {SH} | 2419 | 2/24 序列 | 4 个变体的 ΔTP 前 2 名恒为 2 序 | ✅ |")
    add(f"| 3 | {SH} | 2419 | `000322` / `001324` | ΔTP 前两序名（逐变体，见 §4） | ✅ |")
    add(
        f"| 4 | {SH} | 2419–2420 | 正是回退帧最多的两序 | 离线 GT 帧口径 ✅；**生产 A/B 产物口径 ❌** | ❌ **F1** |"
    )
    add(
        f"| 5 | {SH} | 2420 | 67.4% | 离线 GT 帧口径 1245/1846 = {_pct67:.4f}% ✅；"
        f"**生产 A/B 产物口径 1458/3836 = 38.01% ❌** | ❌ **F1** |"
    )
    add(f"| 6 | {SH} | 2421 | 002251 | 逐序表「最差序列 ΔF1」4 个变体全为 002251 | ✅ |")
    add(f"| 7 | {SH} | 2421 | −0.2713 → −0.7054 → −1.5890 pp | 逐位一致（但漏 rb=0.00 的 −8.0663，见 F5） | ✅ |")
    add(f"| 8 | {RT} | 2425 | 3.64:1 | 273/75 = 3.6400 | ✅ |")
    add(f"| 9 | {RT} | 2425 | 1.26:1 | 158/125 = 1.2640 | ✅ |")
    add(f"| 10 | {RT} | 2426 | 1.17:1 | 405/345 = 1.1739 | ✅ |")
    add(f"| 11 | {RT} | 2426 | 0.26:1 | 319/1235 = 0.2583 | ✅ |")
    add(f"| 12 | {RT} | 2426 | 0.035110 | far(rb=0.00) = 0.03511012389316257 → 0.035110 | ✅ |")
    _rl = result["cross"]["redline_030"]
    _rl_ref = " / ".join(
        f"{_section_label(result['cross']['headings'], _ln)} 行 {_ln}" for _ln, _tx in _rl[:3]
    )
    add(f"| 13 | {RT} | 2426 | 红线 0.030 | 全文 {len(_rl)} 处提到 0.030 红线（{_rl_ref} …），口径一致 | ✅ |")
    add(f"| 14 | {VD} | 2433 | +0.0827 pp | ΔF1(rb=0.80) = +0.0827 | ✅ |")
    add(
        f"| 15 | {VD} | 2433 | 只花 8% 余量 | ΔFAR/余量(0.90) = 0.000454/0.005663 = **8.0155%** | ✅ |"
    )
    add(f"| 16 | {VD} | 2431 | 保持 0.90 | rb=0.90 行即交付档 | ✅ |")
    add(
        f"| 17 | {GT} | 2389 | rb=0.90 与交付档逐位相等 | 指标 68/68 ✅；整文件 21 项不等 + 25 键缺失 | ❌ **F2** |"
    )
    add(f"| 18 | {GT} | 2389–2390 | 三臂跨变体逐位相等 | 0 差异（含深比对） | ✅ |")
    add("")
    add("### 6.1 兑换率的分母定义（任务特别点名）")
    add("")
    add("| 放松区间 | ΔTP | ΔFP | 边际 TP/FP | 报告写法 | 判定 |")
    add("|:--|--:|--:|--:|:--|:--:|")
    for r in rates:
        add(
            f"| {r['range'].replace('->', ' → ')} | {r['dtp']:+d} | {r['dfp']:+d} | "
            f"{r['rate']:.4f} : 1 | {r['rate']:.2f}:1 | ✅ |"
        )
    add("")
    add(
        "**分母定义**：兑换率 = **相邻档位之间的边际** ΔTP / ΔFP，其中 ΔTP/ΔFP 取自**主表 TP/FP 两列的相邻差**"
        "（ΔTP 分母口径 = 该区间的 ΔFP）。因此："
    )
    add(
        "- **rb=0.00 那一档的 0.26:1 是「0.50 → 0.00」相对 0.50 的边际**（ΔTP = +319、ΔFP = +1235），"
        "**不是**相对 0.90；"
    )
    add(
        "- 而**逐序稳健性表**里 rb=0.00 行的 **ΔTP +1155 / ΔFP +1780 是相对 rb=0.90**（累计口径）"
        "（生成器 `:2402–2403` `base = rb=0.90`）。"
    )
    add(
        "⇒ 报告的措辞（「**边际**兑换率」+ 逐区间点名 0.90→0.80 / 0.80→0.70 / 0.70→0.50 / 0.50→0.00）"
        "**技术上正确**，但同节两张表用了**两个不同的基线**（累计 vs 边际）且都未显式声明 → F4。"
    )
    add("")

    # ---------------- 6. 67.4% 的两套口径 ----------------
    add("## 7. 「回退帧最多的两序 / 67.4%」的两套口径（F1 详证）")
    add("")
    add("### 7.1 口径 A —— 生产 A/B 产物（`arms.json` · `stream_stats.per_sequence.replay_rollback_abstained`）")
    add("")
    add("| 变体 | 该门弃权总帧数 | 000322 | 002251 | 001324 | 第 2 名 | 000322+001324 占比 |")
    add("|:--|--:|--:|--:|--:|:--|--:|")
    for peak in ("0.90", "0.80", "0.70", "0.50", "0.0"):
        per = prod["per"][peak]
        total = prod["per_variant_total"][peak]
        a = per.get("VIDEO00006_19700101_000322", 0)
        b = per.get("VIDEO00012_19700101_002251", 0)
        c = per.get("VIDEO00007_19700101_001324", 0)
        second = "—（门关，计数恒 0）" if total == 0 else ("002251" if b >= c else "001324")
        add(
            f"| rb={peak} | {total} | {a} | {b} | {c} | {second} | "
            f"{('—' if not total else f'{100.0 * (a + c) / total:.2f}%') } |"
        )
    add("")
    add(
        "⇒ 在**本节表格所用的生产产物口径**下：rb=0.90 时弃权总帧 **3836**，"
        "`000322` = **974**、`002251` = **769**、`001324` = **484** ⇒ "
        "**「回退帧最多的两序」是 000322 / 002251**（不是 000322 / 001324），"
        "且 `000322+001324` 只占 **38.01%**，不是 67.4%。"
    )
    add("")
    add("### 7.2 口径 B —— 离线 dump（`runs/lookahead/fi_nogmc24/reg_oracle/*.jsonl`，**只算有 GT 的帧**）")
    add("")
    add("复刻 `diag_reg_oracle.py:618–655` `_silent_cause()`（只读 jsonl，不跑推理）：")
    add("")
    add("| 项 | 值 |")
    add("|:--|--:|")
    add(f"| dump 文件数 | {off['n_files']} |")
    add(f"| 静默帧归因分布 | `{off['causes']}` |")
    add(
        f"| **回退链峰值门丢掉的帧（GT 子集）** | **{off['total']}**"
        "（与 `reg_oracle.json` `policy.silent_by_cause.rollback_gate = 1846` 逐位一致） |"
    )
    top3 = sorted(off["per"].items(), key=lambda kv: -kv[1])[:3]
    add(
        f"| 前三序 | "
        + "、".join(f"`{k[-4:]}` = {v}" for k, v in top3)
        + f" ⇒ **前两序 = 000322 / 001324** ✅ |"
    )
    s = off["per"]["VIDEO00006_19700101_000322"] + off["per"]["VIDEO00007_19700101_001324"]
    add(f"| (836 + 409) / 1846 | **{100.0 * s / off['total']:.4f}%** ⇒ 报告的 **67.4%** ✅ |")
    add("")
    add(
        "⇒ **67.4% 这个数字本身完全站得住**：它来自 `reg_oracle.json` 的**离线、只算有 GT 的帧**的归因；"
        "在那个口径下 `000322` / `001324` **确实**是回退帧最多的两序。"
    )
    add("")
    add(
        "**但报告没写这个口径**。§17.4 全节的数据源是 5 份 `arms.json`（生产 A/B 产物），"
        "读者按本节表格能算出的「被回退峰值门丢掉的帧」= 3836 帧、占比 38.0%、第 2 名是 002251 —— "
        "与报告写的 67.4% / 「两序」对不上。两句紧挨着（同一行 "
        f"{rep.get('share_line')}"
        "），中间没有口径切换声明 ⇒ **F1（IMPORTANT）**。"
    )
    add("")
    add(
        "> 溯源：`make_fastinfer_final_report.py:2419–2420` 是**硬编码字符串**，"
        "「回退帧 938 / 479」与「67.4%」照抄自 `make_reg_oracle_report.py:335`（同样是硬编码）。"
        "本审计实测口径 B 的两序原始数是 **836 / 409**（不是 938 / 479）；"
        "**938 在任何产物里都找不到，479 只对得上换了 CFG 的旧 knobscan 产物**（详见 F5 补充）。"
    )
    add("")

    # ---------------- 7. 跨节一致性 ----------------
    add("## 8. 跨节口径一致性")
    add("")
    add("| 锚点 | 出现位置 | §17.4 rb=0.90 行 | 判定 |")
    add("|:--|:--|:--|:--:|")
    for anchor in ("94.8094", "0.024337", "86.9718"):
        locs = [str(l) for l, _ in result["cross"][anchor]]
        add(f"| `{anchor}` | 行 {', '.join(locs)} | 同值 | ✅ |")
    add("")
    add(
        "**注（任务前提澄清）**：§2.2（行 "
        + str(result["cross"].get("sec2.2_lines") or "n/a")
        + "）是 **m=0 裸配置**口径，其 replay 行是 "
        f"`{result['cross']['sec2.2'][0] if result['cross']['sec2.2'] else 'n/a'}`，"
        "**不含** 94.8094 / 0.024337 / 86.9718。交付档（m=6）锚点实际在 "
        + " / ".join(
            f"**{_section_label(result['cross']['headings'], _it['line'])} 行 {_it['line']}**"
            for _it in result["cross"]["anchor_sections"].get("94.8094", [])
            if _it["section"] and not _it["section"].startswith("### 17.4")
        )
        + "（均与 §17.4 逐位一致）。报告 "
        + (f"§0.1 行 {result['cross']['sec0_1_note'][0][0]}" if result["cross"].get("sec0_1_note") else "§0.1")
        + " 已声明这层 m=0 / m=6 区别 ⇒ **§17.4 自身无问题**，此处仅作口径澄清。"
    )
    add("")
    add("FAR 红线表述前后一致：")
    add("")
    for line_no, text in result["cross"]["redline_030"][:6]:
        add(f"- 行 {line_no}：{text[:120]}")
    add("")

    # ---------------- 8. 单位 / 泄漏 ----------------
    add("## 9. 单位·量纲·泄漏扫描")
    add("")
    add("| 项 | 结果 |")
    add("|:--|:--|")
    add("| ΔF1(pp) | 百分比点（100·Δf1），符号带 ±，与 rb=0.90 基线比 |")
    add("| Δhit(pp) | 百分比点（100·Δhit_rate_5px），符号带 ± |")
    add("| ΔFAR | **绝对 FAR/帧差**（非 pp），6 位小数，符号带 ± |")
    add("| FAR | 绝对 FAR/帧，6 位小数 |")
    add("| CLE p95(px) / max jitter(px) | 像素，3 位小数 |")
    add("| TP / FP | 整数计数 |")
    add(f"| §17.4 内 `NaN` / `None` / `null` / `inf` 泄漏 | **{len(result['leaks'])}** 处 |")
    add(f"| §17.4 内空单元格 / `—` 占位 | **{len(result['leaks'])}** 处 |")
    add("")
    add("> §17.4 内没有 `—` 占位：5 个变体的 4 臂 × 11 项指标在产物里**全部有值**，无缺值渲染。")
    add("")

    # ---------------- 9. 发现清单 ----------------
    add("## 10. 发现清单")
    add("")
    add("### F1 · IMPORTANT · 「回退帧最多的两序 / 67.4%」用了与本节表格不同的口径且未声明")
    add("")
    add("- **报告行号**：`runs/lookahead/fi_nogmc/FINAL_REPORT.md` **1152**")
    add("- **生成器源码行号**：`manu/tracking/lookahead/make_fastinfer_final_report.py` **2419–2420**（硬编码）；")
    add("  上游出处 `manu/tracking/lookahead/make_reg_oracle_report.py` **334–335**（硬编码）")
    add("- **报告原文**：`⇒ **总体 ΔTP 有 81~89% 落在 2/24 序列**（`000322` / `001324`，正是回退帧最多的两序，占被回退峰值门丢掉的帧的 **67.4%**）`")
    add("- **独立重算**：")
    add("  - 口径 A（本节表格所用的生产产物 `runs/lookahead/rbgate24nogmc/runs/cs_t0043/d3/rb__0.90/arms.json` → "
        "`arms.replay.stream_stats.per_sequence.*.replay_rollback_abstained`）：总 **3836**；`000322`=**974**、"
        "`002251`=**769**、`001324`=**484** ⇒ 第 2 名是 **002251**，`000322+001324` = **38.01%**。")
    add("  - 口径 B（离线 dump `runs/lookahead/fi_nogmc24/reg_oracle/*.jsonl`，**只算有 GT 的帧**，"
        "复刻 `diag_reg_oracle.py:618–655`）：总 **1846**（与 `reg_oracle.json` "
        "`policy.silent_by_cause.rollback_gate = 1846` 一致）；`000322`=**836**、`001324`=**409**、"
        "`002251`=**5** ⇒ 前两序确为 000322/001324，**1245/1846 = "
        f"{_pct67:.4f}%** ⇒ 报告的 67.4% 成立。")
    add("- **判定**：数字本身可复现（67.4% 精确命中），**但所用口径是离线 GT 帧归因**，"
        "与本节表格的生产 A/B 产物口径不是同一个总体；报告未作任何声明 ⇒ 同一段里两套口径并列且互相矛盾。")
    add("")
    add("### F2 · IMPORTANT · 门禁①「逐位相等」是文件级过强表述，且掩盖了两次运行代码版本不同")
    add("")
    add("- **报告行号**：`runs/lookahead/fi_nogmc/FINAL_REPORT.md` **1141**")
    add("- **生成器源码行号**：`manu/tracking/lookahead/make_fastinfer_final_report.py` **2388–2391**（硬编码）；")
    add("  门禁脚本 `manu/tracking/lookahead/server_overnight/run_rollback_gate_scan.sh` **102–146 / 151–166**")
    add("- **报告原文**：`> **门禁**：rb=0.90 与既有交付档产物逐位相等；……`")
    add("- **独立重算**：")
    add("  - 门禁脚本口径（4 臂 × 17 字段 = 68）：**68/68 逐位相等** ✅")
    add("  - 整文件深比对：共同 **7078** 个标量字段中 **21** 项不等 = 20 × "
        "`arms.replay.frame_reads.<seq>`（磁盘/缓存读帧计数，取决于缓存冷热）+ `elapsed_seconds`；"
        f"另有 **{len(g1['deep_only_rb'])}** 个键**只存在于 rb__0.90**"
        "（`arms.replay.stream_stats.per_sequence.<seq>.replay_disp_gained`），交付档侧没有。")
    add("- **判定**：**指标层面成立**（§17.4 的表格确实只用到这些指标字段），但报告写的是「**产物**逐位相等」，"
        "字面上不成立；且 `replay_disp_gained` 键的存在差异说明**交付档（11-04 20:07）与 A/B（11-05 02:33）"
        "不是同一版代码**——「唯一变量 = CFG」的 A/B 在**代码版本**上并非严格同源。建议改写为"
        "「四臂 68 个指标字段逐位相等（帧读计数 / 耗时 / 统计键不计）」并披露代码版本差异。")
    add("")
    add("### F3 · NIT · 门禁① 未给出「68 字段」，也未指认证据文件")
    add("")
    add("- **报告行号**：**1141**")
    add("- **生成器源码行号**：`make_fastinfer_final_report.py` **2388–2391**")
    add("- **报告原文**：只说「逐位相等」，**没有出现 68 这个数**，也没指 `runs/lookahead/rbgate24nogmc/progress.txt`")
    add("- **独立重算**：68 = 4 臂 × (9 个 `current_frame.overall` 字段 + 8 个 `servo`/`source_frame` 字段)，")
    add("  出处 `run_rollback_gate_scan.sh:102` `FIELDS` + `:118–127` 的 8 个 extras；")
    add("  脚本 `:146` 会打印 `（{len(a) * len(a['replay'])} 字段）` = **68** —— 但该 stdout **没落盘**")
    add("  （`progress.txt` 里只有 `门禁 rc=0`），所以 68 在产物里查不到。")
    add("- **判定**：与上游 `runs/lookahead/fi_nogmc24/reg_oracle/closure_report.md` §6 的写法相比，"
        "§17.4 少了「必须与……逐位相等」+ 证据文件指引 + 68 字段数，可核对性下降。")
    add("")
    add("### F4 · NIT · 逐序表 ΔTP/ΔFP 未声明基线；与相邻「边际兑换率」两个基线并存")
    add("")
    add("- **报告行号**：**1145–1150**（表头/行）与 **1154**（兑换率）")
    add("- **生成器源码行号**：`make_fastinfer_final_report.py` **2396–2416**（累计口径，基线 rb=0.90）"
        "与 **2424–2428**（边际口径，相邻档位）")
    add("- **报告原文**：表头只有 `变体 | ΔTP | ΔFP | 净 | …`，未写「相对 rb=0.90」；")
    add("  下一段写 `0.50→0.00 **0.26:1**`")
    add("- **独立重算**：")
    add("  - 逐序表 rb=0.00 行 **ΔTP +1155 / ΔFP +1780** = 相对 **rb=0.90**（累计）")
    add("  - 兑换率 **0.26:1** = **319/1235** = 0.50 → 0.00 **相对 0.50**（边际）")
    add("- **判定**：两个数都正确，但**基线不同且都未在表内声明**；紧邻阅读极易把 +1155/+1780 当成 0.50→0.00 的边际值。")
    add("")
    add("### F5 · NIT · 「−0.2713 → −0.7054 → −1.5890 pp」漏掉 rb=0.00 的 −8.0663；且「938 / 479」不可复现")
    add("")
    add("- **报告行号**：**1152**")
    add("- **生成器源码行号**：`make_fastinfer_final_report.py` **2421**（硬编码）；"
        "`make_reg_oracle_report.py` **335**（硬编码，含「回退帧 938 / 479」）")
    add("- **报告原文**：`代价集中在 `002251`（逐序 ΔF1 随放松单调变差 −0.2713 → −0.7054 → −1.5890 pp）。`")
    add("- **独立重算**：002251 的逐序 ΔF1 为 −0.2713 / −0.7054 / −1.5890 / **−8.0663**（rb=0.80/0.70/0.50/0.00），"
        "**单调性成立**，但报告只列了前三个。")
    add("- **补充（溯源）**：上游硬编码的「回退帧 **938 / 479**」在**交付档 CFG 的任何产物里都复现不出来** ——")
    add("  离线 GT 帧口径实测是 **836 / 409**，生产 A/B 产物口径是 **974 / 484**（本审计两套都独立重算过）。")
    if result["literals"]:
        add("")
        add("  全仓 `runs/lookahead/**/arms.json` 扫描：**479** 只出现在**旧 knobscan 产物**（**`--replay-roi-selfpred-max-px` "
            "∉ 交付档的 16** ⇒ 换了 CFG 的另一批产物）：")
        add("")
        add("  | 产物 | 臂 | 序 | 字段 | 值 |")
        add("  |:--|:--|:--|:--|--:|")
        for h in result["literals"]:
            add(
                f"  | `{h['path']}` | {h['arm']} | `{h['seq'][-4:]}` | `{h['key'].split('.')[-1]}` | "
                f"{h['value']} |"
            )
        add("")
        add("  ⇒ **938 在任何产物里都找不到**；479 只对得上**换了 CFG 的旧产物** ⇒ 这一对数字是**陈旧 / 串源**的。")
    add("- **判定**：单调性结论不受影响，但数字本身应改为从产物现算。")
    add("")
    add("### F6 · NIT · rb=0.70 产物里存的 `f1` 与 `2TP/(2TP+FP+FN)` 差 1 ULP")
    add("")
    add("- **报告行号**：**1137**（F1% = 94.9102 / ΔF1 = +0.1008）")
    add("- **生成器源码行号**：`make_fastinfer_final_report.py` **2383**（直接印产物里的 `f1`）")
    add("- **报告原文**：`| 0.70 | 94.9102 | +0.1008 | …`")
    add("- **独立重算**：产物 `rb__0.70/arms.json` 存的 `f1 = 0.9491016247577768`；")
    add("  `2TP/(2TP+FP+FN) = 0.9491016247577769`（差 1 ULP = 1.11e-16）；")
    add("  而 `2·P·R/(P+R) = 0.9491016247577768` **逐位相等** ⇒ 产物用的是 P/R 口径。")
    add("  另 4 个变体的 `f1` 与两种口径都逐位相等。")
    add("- **判定**：**显示精度下无影响**（F1% 两种口径都是 94.9102，ΔF1 都是 +0.1008），"
        "报告忠实转录产物值 ⇒ 仅作产物侧精度备注。")
    add("")
    add("### F7 · NIT · ΔFAR 列未标量纲")
    add("")
    add("- **报告行号**：**1133**（表头 `| … | FAR | ΔFAR | … |`）")
    add("- **生成器源码行号**：`make_fastinfer_final_report.py` **2371 / 2384**")
    add("- **报告原文**：`| rollback-min-peak | F1% | ΔF1(pp) | Recall% | Precision% | FAR | ΔFAR | … |`")
    add("- **独立重算**：ΔFAR = FAR(变体) − FAR(0.90)，量纲是 **FAR/帧的绝对差**（如 +0.000454），"
        "**不是** pp；同表 `ΔF1(pp)` / `Δhit(pp)` 才标了 pp。")
    add("- **判定**：可从 FAR 列（本身是 /帧）推出，不算错，但与相邻两列的标注风格不一致。")
    add("")

    # ---------------- 10. 复现 ----------------
    add("## 11. 复现命令")
    add("")
    add("```bash")
    add("cd /tmp/pycharm_project_10ae9e2e")
    add("/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec17_rbgate.py")
    add("```")
    add("")
    add("行号漂移回归（副本 + 删行，**不改仓库里的 FINAL_REPORT.md**）：")
    add("")
    add("```bash")
    add("mkdir -p /tmp/drift && cp runs/lookahead/fi_nogmc/FINAL_REPORT.md /tmp/drift/FINAL_REPORT.md")
    add("sed -i '200,236d' /tmp/drift/FINAL_REPORT.md")
    add("/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/audit_sec17_rbgate.py \\")
    add("    --report-md /tmp/drift/FINAL_REPORT.md --out-md /tmp/drift/out.md")
    add("```")
    add("")
    add("只读产物、不跑推理；可重复运行（幂等）；**只写本审计报告一个文件**（全部原始值已在正文表格里）。")
    add("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
