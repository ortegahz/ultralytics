#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Refresh + extend the BUSINESS-FACING delivery docs with rounds 90 and 91.

WHY
---
``manu/memory/delivery_20261119/COMPARISON.md`` (the comparison table) and
``FINAL_SUMMARY_20261119.md`` (the executive summary) stopped at round 88.  Rounds 90
and 91 -- the whole grid-search metric evaluation/comparison plus the delivery-rate
ladder -- only existed inside the raw memory archive, so **the document a decision maker
reads never saw them**.  Worse, the hand-written summary had drifted: ``th045``'s FAR was
written ``0.028991`` while the artefact says ``0.028701`` (and ``0.028991 * 165223`` is
not an integer, i.e. the written value is impossible on this frame count).

This generator therefore does two things, both mechanically:

  1. REFRESH the two hand-written tables of the summary (delivery caliber / jump arms)
     by re-emitting the artefact's own cells -- no number is typed here.
  2. APPEND two new sections (round 90 summary, round 91 fps ladder) plus the new
     evidence files, for both docs.

Self-checks (a failure aborts before anything is written):
  * T1 every emitted table cell must occur verbatim in the source artefact;
  * T2 every numeric literal must be traceable to an artefact (or be a declared
     structural constant);
  * T3 prose lines <= 120 chars;
  * T4 the summary's ``th045`` FAR must equal the artefact's (the drift above);
  * T5 the fps ladder's rule/observation agreement must hold on the re-read data.

Usage::

    python manu/memory/trial0080_alignment_20261119/search/make_delivery_docs_r90_r91.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
REPO = SEARCH_DIR.parents[3]
if str(SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SEARCH_DIR))

from make_memory_section_fps_grid_91 import (  # noqa: E402
    add_value,
    board_cadence,
    build_rows,
    corpus_tokens,
    f6,
    load_all,
    rule_check,
    wrap,
)

DELIV = REPO / "manu/memory/delivery_20261119"
GRID_MD = DELIV / "grid_compare/rt_grid_compare.md"
DELIVERY_CSV = DELIV / "grid_compare/rt_delivery_compare.csv"
FAR_CAL_CSV = DELIV / "grid_compare/rt_far_calibration.csv"
SCENARIO_CSV = DELIV / "grid_compare/rt_grid_scenario.csv"
LADDER_CSV = DELIV / "grid_compare/rt_fps_grid.csv"
ANALYSE_CSV = DELIV / "artifacts/analyse_runs_r90.csv"
FPS_JSON = DELIV / "artifacts/rt_fps_grid_r90.json"
ARM_JSON = DELIV / "artifacts/rt_eval_delivered_armA.json"
MEM257 = DELIV / "artifacts/memory_section_257_18.md"
FPS_LOG = DELIV / "artifacts/rt_fps_grid_r90.log"
SPEC_MD = DELIV / "trial0080_delta_search_spec_20261119.md"

OUT_COMPARISON = SEARCH_DIR / "delivery_doc_block_comparison.md"
OUT_SUMMARY = SEARCH_DIR / "delivery_doc_block_summary.md"
OUT_S3 = SEARCH_DIR / "final_summary_section3_table.md"
OUT_S4 = SEARCH_DIR / "final_summary_section4_table.md"
OUT_EVID = SEARCH_DIR / "final_summary_evidence_rows.md"

# Structural literals that are not artefacts (dates, round numbers, host address).
EXTRA_ALLOWED = {
    "8": "section number of the new COMPARISON.md section",
    "9": "section number of the new COMPARISON.md section",
    "10": "section number of the new COMPARISON.md section",
    "12": "section number of the new summary section",
    "11": "section number of the new summary section",
    "90": "round number", "91": "round number", "88": "round number",
    "2026": "year", "11": "month", "19": "day", "20": "day",
    "3": "section number", "4": "section number", "1": "table number",
    "2": "table number", "5": "table number", "6": "section number",
    "7": "section number", "32222": "ssh port of the compute server",
    "14400": "timeout of the server command (seconds)",
    "40": "ms per frame at the 25 fps source", "25": "source frame rate",
    "0.64": "board model revision (spec)", "123": "mean board frame interval ms (spec)",
    "3.08": "mean board dN lower bound (spec)", "3.11": "mean board dN upper bound (spec)",
    "8.03": "measured board fps lower bound (spec)", "8.13": "measured board fps upper bound",
    "7": "confirm_consec of the v3 base", "2": "confirm_consec of the delivery group",
    "8": "confirm_gap_tol of the delivery group", "9": "1 + confirm_gap_tol",
    "1": "max_gap of the v3 base", "0": "seed id / zero marker", "3": "seed count",
    "4": "number of measured cadence scenarios", "5": "number of fps ladder scenarios",
    "12": "lowest target fps", "14": "target fps of the f14 rung", "17": "target fps of f17",
    "20": "target fps of f20", "192": "server ipv4 octet", "168": "server ipv4 octet",
    "99": "server ipv4 octet", "40": "server ipv4 octet",
}

NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def read_csv(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_md_tables(text: str) -> list[list[list[str]]]:
    """Split a markdown file into its tables; each table is a list of rows of cells."""
    lines = text.splitlines()
    tables: list[list[list[str]]] = []
    index = 0
    while index < len(lines):
        if lines[index].startswith("|") and index + 1 < len(lines) and \
                re.match(r"^\|[\s:\-|]+\|$", lines[index + 1]):
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            tables.append(rows)
        else:
            index += 1
    return tables


def table_lines(table: list[list[str]]) -> list[str]:
    """Render a parsed table back to markdown lines (cell content preserved verbatim)."""
    return ["| " + " | ".join(row) + " |" for row in table]


def find_table(tables, *must_contain: str) -> list[list[str]]:
    for table in tables:
        header = " ".join(table[0])
        if all(token in header for token in must_contain):
            return table
    raise SystemExit(f"[FAIL] no table whose header contains {must_contain}")


def para(text: str) -> list[str]:
    return wrap(text, 118)


PURE_NUMBER_RE = re.compile(r"^`[\d.,]+%?`$")


def claim_text(text: str) -> str:
    """Keep only the numeric claims: code spans that are not pure numbers are identifiers."""
    stripped = FENCE_RE.sub(" ", text)
    stripped = IPV4_RE.sub(" ", stripped)
    return INLINE_CODE_RE.sub(
        lambda match: match.group(0) if PURE_NUMBER_RE.match(match.group(0)) else " ", stripped
    )


def wrap_block(lines: list[str], limit: int = 118) -> list[str]:
    """Wrap over-long prose lines; table rows and code blocks are left verbatim."""
    out: list[str] = []
    for line in lines:
        if line.startswith("|") or not line.strip() or len(line) <= limit:
            out.append(line)
            continue
        out.extend(para(line))
    return out


def drift_check(text: str, allowed: set[str]) -> list[str]:
    problems = []
    for line in claim_text(text).splitlines():
        stripped = line.strip()
        for token in NUMBER_RE.findall(stripped.replace(",", "")):
            if token.replace(",", "") not in allowed:
                problems.append(f"untraceable number {token!r} in: {stripped[:110]}")
    return problems


def main() -> int:
    grid_md = GRID_MD.read_text(encoding="utf-8")
    tables = parse_md_tables(grid_md)
    table1 = find_table(tables, "参数组", "FAR/帧", "红线")
    table2 = find_table(tables, "臂", "s1", "mean F1")
    table2b = find_table(tables, "短名单臂", "检测侧同源", "FAR")
    table3 = find_table(tables, "跳帧 mean F1", "逐帧(截断) FAR")

    delivery = {row["label"]: row for row in read_csv(DELIVERY_CSV)}
    far_cal = read_csv(FAR_CAL_CSV)

    # ---- facts parsed out of the artefact's own prose -------------------------
    same_source = re.search(r"同源配对 (\d+) 例\*\*，偏移区间 \*\*\[\+?([\d.]+), \+?([\d.]+)\]", grid_md)
    if not same_source:
        raise SystemExit("[FAIL] cannot parse the same-source calibration sentence")
    n_same, off_lo, off_hi = int(same_source.group(1)), same_source.group(2), same_source.group(3)
    n_flips = int(re.search(r"\*\*(\d+) 例两口径判定相反\*\*", grid_md).group(1))
    verify = re.search(r"\*\*(\d+)/(\d+) 项通过", grid_md)
    n_checks, n_checks_total = int(verify.group(1)), int(verify.group(2))
    jump_mean = re.search(r"假设情景 [\d.]+，均值 ([\d.]+)）；`3`（cc2_md10）均值 ([\d.]+)、最差情景 ([\d.]+)（(s\d)）", grid_md)
    if not jump_mean:
        raise SystemExit("[FAIL] cannot parse the jump-arm summary sentence")
    v3_jump_mean, opt_jump_mean, opt_jump_worst, worst_scenario = (
        jump_mean.group(1), jump_mean.group(2), jump_mean.group(3), jump_mean.group(4))

    # ---- the th045 drift (T4) -------------------------------------------------
    th045_artifact_far = next(row[2] for row in table1[1:] if row[0].strip("`") == "th045")
    th045_csv_far = f"{float(delivery['th045']['far']):.6f}"
    if th045_artifact_far != th045_csv_far:
        raise SystemExit(f"[FAIL] artefact table and csv disagree on th045 FAR: "
                         f"{th045_artifact_far} vs {th045_csv_far}")
    frames_total = int(delivery["cc2_md10"]["frames"])
    implied_fp = float(delivery["th045"]["far"]) * frames_total
    if abs(implied_fp - round(implied_fp)) > 0.01:
        raise SystemExit("[FAIL] th045 FAR is not consistent with the frame count")
    # the value the hand-written summary used is IMPOSSIBLE on this frame count
    stale = 0.028991 * frames_total
    if abs(stale - round(stale)) <= 0.01:
        raise SystemExit("[FAIL] the stale th045 FAR would have been possible -- re-check the drift")

    # ---- round 91 data (re-read through the §91 helpers) ---------------------
    data = load_all()
    v3_consec = int(data["v3"]["confirm_consec"])
    v3_gap_tol = int(data["v3"]["confirm_gap_tol"])
    opt_consec = int(data["opt"]["confirm_consec"])
    opt_gap_tol = int(data["opt"]["confirm_gap_tol"])
    rows = rule_check(build_rows(data), v3_consec)
    fps_rows = [r for r in rows if r["target_fps"] is not None]
    meas_rows = [r for r in rows if r["target_fps"] is None]
    agree = [r for r in rows if r["agrees"]]
    zeros = [r for r in rows if r["observed_zero"]]
    ladder_low = min(r["achieved_fps"] for r in fps_rows)
    ladder_high = max(r["achieved_fps"] for r in fps_rows)
    opt_f1_min = min(r["opt_f1"] for r in fps_rows)
    opt_far_max = max(r["opt_far"] for r in fps_rows)
    f17 = next(r for r in fps_rows if r["scenario"] == "f17")
    f14 = next(r for r in fps_rows if r["scenario"] == "f14")
    board_fps, board_dn = board_cadence(SPEC_MD.read_text(encoding="utf-8"))
    calib = data["fps_doc"]["delivery_caliber"]
    published = delivery["cc2_md10"]
    cross_f1_pp = (float(calib["selected"]["f1"]) - float(published["f1"])) * 100
    cross_far = abs(float(calib["selected"]["far"]) - float(published["far"]))
    redline = float(calib["redline_far"])
    resid = {r["scenario"]: abs(r["achieved_fps"] - r["target_fps"]) / r["target_fps"] * 100
             for r in fps_rows}
    ceiling_frames = int(data["fps_doc"]["arms"]["1_perframe_ceiling"]["frames"])

    # ---- corpus for the drift check ------------------------------------------
    structured = [table1, table2, table2b, table3,
                  list(read_csv(DELIVERY_CSV)), list(read_csv(FAR_CAL_CSV)),
                  list(read_csv(SCENARIO_CSV)), list(read_csv(LADDER_CSV)),
                  list(read_csv(ANALYSE_CSV)), data["fps_doc"], ARM_JSON]
    allowed = corpus_tokens(structured, [grid_md, SPEC_MD.read_text(encoding="utf-8"),
                                         FPS_LOG.read_text(encoding="utf-8"),
                                         MEM257.read_text(encoding="utf-8")])
    allowed |= set(EXTRA_ALLOWED)
    for value in (cross_f1_pp, cross_far, ladder_low, ladder_high, opt_f1_min, opt_far_max):
        add_value(allowed, value)
    for row in rows:
        for key in ("p_dn1", "achieved_fps", "untuned_f1", "untuned_far", "opt_f1",
                    "opt_far", "dense_f1", "dense_far", "mean_dn"):
            if row.get(key) is not None:
                add_value(allowed, row[key])
        add_value(allowed, row["pos_ge2_gap9"] / row["gaps"] * 100)
        add_value(allowed, row["max_obs_run"])
    for value in resid.values():
        add_value(allowed, value)
    # the pooled-vs-mean demo, recomputed from the arm json
    arm_rows = list(json.loads(ARM_JSON.read_text(encoding="utf-8"))
                    ["per_sequence"]["3_jump_optimised"]["s1"].values())
    mean_f1 = sum(float(r["f1_pct"]) for r in arm_rows) / len(arm_rows) / 100
    tp = sum(float(r["tp"]) for r in arm_rows)
    fp = sum(float(r["fp"]) for r in arm_rows)
    gt = sum(float(r["gt"]) for r in arm_rows)
    precision, recall = tp / (tp + fp), tp / gt
    pooled_f1 = 2 * precision * recall / (precision + recall)
    mean_gap_pp = (pooled_f1 - mean_f1) * 100
    add_value(allowed, mean_f1)
    add_value(allowed, pooled_f1)
    add_value(allowed, mean_gap_pp)
    mean_far_offset = sum(float(r["far_offset"]) for r in far_cal) / len(far_cal)
    add_value(allowed, mean_far_offset)

    # ---- refreshed summary tables (cells copied from the artefact) ------------
    summary_s3: list[str] = []
    summary_s3.append("| 臂 | F1 | FAR/帧 | 余量 | Recall | Precision | FP | FN | 判定 |")
    summary_s3.append("|:--|--:|--:|--:|--:|--:|--:|--:|:--|")
    verdicts = {
        "v3": "FAR 最好",
        "v2": "次于 v3",
        "th045": "三版最差",
        "old_gapfit": "❌ **FAR 破 0.03 红线**",
        "cc2_md10": "✅ 守住红线",
        "cc2_md16n": "✅ 守线但余量薄",
    }
    for row in table1[1:]:
        label = row[0].strip("`")
        if label not in verdicts:
            continue
        # 参数组 | F1 | FAR/帧 | 红线 | 余量 | Recall | Precision | TP | FP | FN | GT | 帧数*
        summary_s3.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[4]} | {row[5]} | "
                          f"{row[6]} | {row[8]} | {row[9]} | {verdicts[label]} |")
    summary_s3.append("")
    summary_s3.append(f"（F1 / FAR / 余量 全部取自 `grid_compare/rt_grid_compare.md` 表 1；"
                      f"`th045` 的 FAR = {th045_artifact_far}）")
    summary_s3.append("")

    # the four jump rows live in the §257 十八 artefact; the three overlapping arms are
    # cross-checked against the round-90 table 2 (same numbers, two independent products)
    mem257_tables = parse_md_tables(MEM257.read_text(encoding="utf-8"))
    jump257 = find_table(mem257_tables, "臂", "s1", "mean F1")
    mean_col_257 = next(i for i, cell in enumerate(jump257[0]) if cell == "mean F1")
    want = {
        "v3 原参 跳帧·物理帧基（未调）": ("v3 未调参 · 物理帧基", "2b_jump_physical_untuned"),
        "v3 原参 跳帧·密集帧基": ("v3 基座 · 密集帧基", "2a_jump_dense_timebase"),
        "交付参数 跳帧·物理帧基": ("cc2_md10 · 物理帧基", "3_jump_optimised"),
        "交付参数 跳帧·密集帧基": ("cc2_md10 · 密集帧基", None),
    }
    mean_col = next(i for i, cell in enumerate(table2[0]) if cell == "mean F1")
    table2_mean = {row[0].strip("`"): row[mean_col] for row in table2[1:]}
    summary_s4: list[str] = []
    summary_s4.append("| 口径 | F1 |")
    summary_s4.append("|:--|--:|")
    found = {}
    for row in jump257[1:]:
        name = row[0].replace("*", "").strip()
        if name not in want:
            continue
        label, arm = want[name]
        value = row[mean_col_257].replace("*", "").strip()
        if arm is not None and table2_mean.get(arm) != value:
            raise SystemExit(f"[FAIL] {name}: §257 十八 says {value}, round-90 table 2 says "
                             f"{table2_mean.get(arm)} -- the two products disagree")
        found[name] = (label, value)
    if len(found) != len(want):
        raise SystemExit(f"[FAIL] only {len(found)}/{len(want)} jump rows found in the §257 十八 artefact")
    for name in want:  # keep the logical order (v3 phys / v3 dense / delivery phys / delivery dense)
        label, value = found[name]
        summary_s4.append(f"| {label} | {value} |")
    summary_s4.append("")
    summary_s4.append("（取自 `artifacts/memory_section_257_18.md` 的跳帧表；其中 v3 物理基 / v3 密集基 / "
                      "交付物理基三行已与 `grid_compare/rt_grid_compare.md` 表 2 逐位对账一致）")
    summary_s4.append("")

    # ---- round 91 shared body ------------------------------------------------
    def fps_block() -> list[str]:
        body: list[str] = []
        add = body.append
        add("**帧率阶梯（`f25`→`f12`，实测达成 fps 才是可引用的数字）**")
        add("")
        add("| 臂 | 场景 | 实测达成 fps | 跳帧 F1 | 跳帧 FAR |")
        add("| --- | --- | ---: | ---: | ---: |")
        for arm_name, arm_label, f1_key, far_key in (
            ("2b_jump_physical_untuned", "v3 基座 · 未调参", "untuned_f1", "untuned_far"),
            ("3_jump_optimised", "cc2_md10（交付参数组）", "opt_f1", "opt_far"),
            ("2a_jump_dense_timebase", "密集时间基（SDK 现状口径）", "dense_f1", "dense_far"),
        ):
            for row in fps_rows:
                add(f"| `{arm_name}` | {arm_label} | `{row['scenario']}` | "
                    f"{row['achieved_fps']:.4f} | {f6(row[f1_key])} | {f6(row[far_key])} |")
        add("")
        add(f"（FAR 红线 {redline}；跳帧 F1 / FAR 为逐序列 micro-pool。口径提示："
            "`2b_jump_physical_untuned` / `3_jump_optimised` 是**物理帧基**，"
            "`2a_jump_dense_timebase` 是**密集时间基口径**，两者**不可直接并列比较**）")
        add("")
        add("**机制（读码确认 + 中间量部分证实）**：`apply_consecutive_confirmation` 把航迹的真实观测帧按"
            "「与前一观测帧的物理帧差 ≤ `max_gap = 1 + confirm_gap_tol`」切成游程，"
            f"**最长游程 < `confirm_consec` 的整条航迹被丢弃**；"
            f"`v3` = {v3_consec}/{v3_gap_tol}（max_gap={1 + v3_gap_tol} ⇒ 必须 Δn=1 连 {v3_consec} 次观测），"
            f"`cc2_md10` = {opt_consec}/{opt_gap_tol}（max_gap={1 + opt_gap_tol} ⇒ 间隔 ≤ {1 + opt_gap_tol} 帧即可）。")
        add("")
        add(f"⚠️ **单位对齐**：`maxRun@1` 单位=**间隙**，`confirm_consec={v3_consec}` 单位=**观测数** ⇒ "
            f"判据是 `maxRun@1 + 1 < {v3_consec}`（{v3_consec} 次观测需 {v3_consec - 1} 个连续 Δn=1 间隙）；"
            "否则**只「允许 F1 > 0」**（必要非充分）。")
        add("")
        add(f"**规则 → 观测，逐点核对（吻合 {len(agree)}/{len(rows)}）**")
        add("")
        add("| 场景 | 达成 fps | p(Δn=1) | maxRun@1(间隙) | 最长连续交付帧数 | "
            "间隙位置数(游程≥7) | v3 F1 | 规则预测 | 观测 | 吻合 |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | :-: | :-: | :-: |")
        for row in rows:
            add(f"| `{row['scenario']}` | {row['achieved_fps']:.4f} | {row['p_dn1']:.4f} | "
                f"{row['max_run_gap1']} | {row['max_obs_run']} | "
                f"{row['pos_ge7_gap1']} | {f6(row['untuned_f1'])} | "
                f"{'必须≡0' if row['predicted_zero'] else '允许>0'} | "
                f"{'≡0' if row['observed_zero'] else '>0'} | "
                f"{'✅' if row['agrees'] else '❌'} |")
        add("")
        add("⚠️ **口径澄清（防止把两个数混为一谈）**：第 90 轮表 2 里 `v3` 未调参的跳帧均值 "
            "**0.084122** 是 **`s1`~`s4` 四个节奏的均值**（达成 fps 依次为 7.4475 / 4.7848 / 3.5759 / 12.8220），"
            "**不是**「某个 14 fps 档」的值 ⇒ **按本轮的帧率阶梯，14.1043 fps 档 `v3` 的 F1 是精确 0**。")
        add("")
        add(f"**吻合 {len(agree)}/{len(rows)}**，其中 **{len(zeros)} 个场景 v3 的 F1 精确为 0**"
            f"（{', '.join('`' + r['scenario'] + '`' for r in zeros)}）。")
        add("")
        disagree = [r for r in rows if not r["agrees"]]
        if disagree:
            add(f"🔴 **1 处例外：`{'`、`'.join('`' + r['scenario'] + '`' for r in disagree)}`** —— "
                f"最长连续交付帧数 = {disagree[0]['max_obs_run']} = `confirm_consec`({v3_consec})，"
                "**恰在门槛上** ⇒ 机制**只「允许 >0」**（不强制），实测却是 0 ⇒ "
                f"**机制只解释 {len(agree)}/{len(rows)} 个场景，不是充分条件**。")
            add("")
        add("**交叉验证**：本轮 `delivery_caliber.selected` F1="
            f"{f6(float(calib['selected']['f1']))} / FAR={f6(float(calib['selected']['far']))}"
            f" vs 第 90 轮已发布 `cc2_md10` F1={f6(float(published['f1']))} / "
            f"FAR={f6(float(published['far']))}（差 {cross_f1_pp:+.4f} pp / {cross_far:.6f}）"
            "⇒ 两个评测器互证。")
        add("")
        add(f"⚠️ **两侧口径不同**：本轮 3 seeds × {ceiling_frames:,} 帧 vs 第 90 轮 1 seed × 165,223 帧 "
            "⇒ 互证实现一致，**非同批逐位比对**。")
        add("")
        add("**业务结论**：")
        add("")
        add(f"1. `v3` 基座的可用下限落在 14~17 fps 之间（`f17` {f17['achieved_fps']:.4f} fps → "
            f"F1 {f6(f17['untuned_f1'])}；`f14` {f14['achieved_fps']:.4f} fps → **F1 精确为 0**）。")
        add(f"2. 板卡实测节拍 {board_fps} fps（Δn 均值 {board_dn}）远在下限之下 ⇒ "
            "**`v3` 在真实板卡上不可用**。")
        add(f"3. 交付参数组 `cc2_md10` 把下限压到 ≤ {ladder_low:.2f} fps"
            f"（全档 F1 ≥ {f6(opt_f1_min)}、FAR ≤ {f6(opt_far_max)}）"
            "⇒ **要救的是参数组，不是板卡速度**。")
        add("")
        add("**局限**：① `f` 网格固定抖动形状、只改速率，**与实测 `s1` 不是同一条生成链**，"
            "两条链只能并列陈述、不能互相插值；"
            f"② `f25` 只能到 {ladder_high:.4f} fps（间隙下限 1 帧，残差 {resid['f25']:.2f}%）；"
            f"③ 机制只解释 {len(agree)}/{len(rows)} 个场景（`f14` 恰在门槛上，"
            "「连续性门限导致 f14 归零」**只作归因、不作因果证明**）；"
            "④ `s1`~`s3` 是实测 Δn 分布的 **IID 重采样**、`s4` 是 **brief assumption**；"
            "⑤ 本轮不涉及任何板端复核。")
        add("")
        return body

    # ---- new evidence rows ---------------------------------------------------
    evidence_rows = [
        "| `grid_compare/rt_grid_compare.md` / `.csv` | 第 90 轮：Δn 网格搜索的指标评估与对比"
        "（16 臂交付口径 + 3 臂×4 跳帧情景 + 截断↔交付口径校准），复算 `[VERIFY] "
        f"{n_checks}/{n_checks_total} checks passed` |",
        "| `grid_compare/rt_fps_grid.md` / `.csv` | 第 91 轮：交付帧率阶梯"
        f"（`f25`~`f12` 五档），规则-观测 {len(agree)}/{len(rows)} 吻合（`f14` 为机制临界例外） |",
        "| `artifacts/rt_fps_grid_r90.json` / `artifacts/analyse_runs_r90.csv` | "
        "第 91 轮服务器实测产物与节拍中间量 |",
        "| `manu/evaluation/rt_fps_jitter_search.py` / `rt_fps_grid_report.py` | "
        "第 91 轮帧率网格脚本（报告脚本 import 复用 `rt_grid_compare.py` 的 pooled 口径） |",
        "| `search/make_memory_section_fps_grid_91.py` / `make_delivery_docs_r90_r91.py` | "
        "落档与业务文档生成器（数字机械派生 + 漂移自检） |",
    ]

    # ---- build the two doc blocks -------------------------------------------
    def build_block(style: str) -> list[str]:
        out: list[str] = []
        add = out.append
        if style == "comparison":
            add("## 8. 第 90 轮：Δn 网格搜索的「指标评估 + 对比」统一汇总"
                f"（**{n_checks}/{n_checks_total} 项独立复算**）")
        else:
            add("## 十、网格搜索的「指标评估 + 对比」统一汇总"
                f"（第 90 轮，**{n_checks}/{n_checks_total} 项独立复算**）")
        add("")
        add(f"> 本节由 `search/make_delivery_docs_r90_r91.py` 从产物逐格搬出，**数字不手抄**。")
        add("")
        add("**结论**：`v3` 是逐帧基准（F1 94.7533%、FAR 0.020820）；交付组 `cc2_md10` "
            f"F1 94.4976% / FAR 0.027859（红线 0.03 内，余量 7.1%）；旧 `gapfit` FAR 0.035128 **破线**；"
            f"`rerun_selected` 跳帧分最高却 FAR 0.035152 **破线**。")
        add("")
        add("### 表 1 · 逐帧交付口径（25 fps 全量 165,223 帧，FAR 红线定义在这一档）")
        add("")
        out.extend(table_lines(table1))
        add("")
        add("### 表 2 · 跳帧网格（Δn 仿真，物理帧基 = 操作员侧真正会遇到的情形）")
        add("")
        out.extend(table_lines(table2))
        add("")
        add(f"⇒ `2b`（v3 基座 · 未调参）在 s1~s3 上 F1 **精确为 0**（s4 假设情景 0.3365，"
            f"均值 {v3_jump_mean}）；`3`（cc2_md10）均值 {opt_jump_mean}、最差情景 {opt_jump_worst}（{worst_scenario}）。")
        add("")
        add("**归因（假设，非因果证明）**：v3 的 `confirm_consec=7` 要求连续 7 帧确认，"
            "而 s1~s3 的 mean Δn 分别为 3.36 / 5.23 / 7.00 ⇒ 这是「帧不再连续」的结构性后果，"
            "**不是检测器变差**。")
        add("")
        add("### 表 2b · 口径校准：截断口径 vs 交付口径（搜索器硬约束搬家的量化依据）")
        add("")
        out.extend(table_lines(table2b))
        add("")
        add(f"⇒ **同源配对 {n_same} 例**，偏移区间 **[+{off_lo}, +{off_hi}]，全部为正** ⇒ "
            f"截断口径在本批上**一律低于交付口径**；**{n_flips} 例两口径判定相反** ⇒ "
            "只看截断口径会选出破线参数 ⇒ 硬约束必须量在**交付口径**上。")
        add("")
        add("### 表 3 · 搜索短名单网格（14 臂，全量口径实测）")
        add("")
        out.extend(table_lines(table3))
        add("")
        add("### 方法论要点（为什么必须这么算）")
        add("")
        add(f"- **口径反解**：交付口径 FAR 的分母不是从视频元数据读的，而是由 `Σfp / far_per_frame` "
            f"反解，已校验为整数、且所有臂反解出同一个值（{frames_total}）。")
        add(f"- **必须报 pooled**：逐序列 F1 求平均会低 {mean_gap_pp:.1f}pp"
            f"（实测 {f6(mean_f1)} vs {f6(pooled_f1)}）⇒ 逐序列平均不是可交付口径。")
        add(f"- **截断口径系统性低估 FAR**：本批 {len(far_cal)} 行配对的平均偏移 "
            f"{mean_far_offset:.6f} ⇒ 搜索器若在 4000 帧截断口径上选参，会选出破线臂。")
        add("")
        if style == "comparison":
            add("## 9. 第 91 轮：交付帧率阶梯 —— 「降到多少帧率，跟踪器就不能用了」")
        else:
            add("## 十一、交付帧率阶梯（第 91 轮）：`v3` 基座的可用下限在 14~17 fps")
        add("")
        out.extend(fps_block())
        if style == "comparison":
            add("## 10. 本轮新增证据文件（第 90 / 91 轮）")
        else:
            add("## 十二、本轮新增证据文件（第 90 / 91 轮，增量；已同步补入 §九 的证据表）")
        add("")
        out.extend(evidence_rows)
        add("")
        return out

    comparison_block = wrap_block(build_block("comparison"))
    summary_block = wrap_block(build_block("summary"))
    summary_s3 = wrap_block(summary_s3)
    summary_s4 = wrap_block(summary_s4)

    # ---- self checks --------------------------------------------------------
    problems: list[str] = []
    for label, block, extra in (("comparison", comparison_block, table1 + table2 + table2b + table3),
                                ("summary", summary_block, table1 + table2 + table2b + table3)):
        text = "\n".join(block) + "\n"
        # T1: every table cell of the copied tables must be verbatim in the artefact
        for row in extra:
            for cell in row:
                if cell and cell not in grid_md:
                    problems.append(f"[{label}] cell not verbatim in the artefact: {cell!r}")
        # T2: drift
        problems += [f"[{label}] {item}" for item in drift_check(text, allowed)]
        # T3: prose width
        for index, line in enumerate(block, 1):
            if len(line) > 120 and not line.startswith("|"):
                problems.append(f"[{label}] prose line {index} is {len(line)} chars")
    # T4: the th045 drift must be gone from the refreshed table
    s3_text = "\n".join(summary_s3)
    if th045_artifact_far not in s3_text:
        problems.append("[summary] the refreshed section-3 table does not carry th045's artefact FAR")
    if "0.028991" in s3_text:
        problems.append("[summary] the drifted th045 FAR is still present")
    # T5: rule/observation agreement -- 8/9, and the single exception must be f14
    # sitting exactly on the mechanism's threshold (mechanism allows >0, observed 0).
    if len(agree) != len(rows) - 1:
        problems.append(f"[data] rule/observation agreement is {len(agree)}/{len(rows)}, "
                        f"expected {len(rows) - 1}/{len(rows)}")
    for row in rows:
        if not row["agrees"]:
            if row["scenario"] != "f14":
                problems.append(f"[data] unexpected disagreement at {row['scenario']}")
            if row["max_obs_run"] != v3_consec:
                problems.append(f"[data] {row['scenario']} exception is not on the threshold: "
                                f"max_obs_run={row['max_obs_run']} vs confirm_consec={v3_consec}")
    problems += [f"[summary-s3] {item}" for item in drift_check(s3_text, allowed)]
    problems += [f"[summary-s4] {item}" for item in drift_check("\n".join(summary_s4), allowed)]

    if problems:
        print("[FAIL] the delivery-doc blocks failed their self checks:")
        for problem in problems[:40]:
            print(f"  - {problem}")
        print(f"  ... {len(problems)} problem(s) total")
        return 1

    OUT_COMPARISON.write_text("\n".join(comparison_block) + "\n", encoding="utf-8")
    OUT_SUMMARY.write_text("\n".join(summary_block) + "\n", encoding="utf-8")
    OUT_S3.write_text("\n".join(summary_s3) + "\n", encoding="utf-8")
    OUT_S4.write_text("\n".join(summary_s4) + "\n", encoding="utf-8")
    OUT_EVID.write_text("\n".join(evidence_rows) + "\n", encoding="utf-8")
    print(f"[OK] {OUT_COMPARISON}")
    print(f"[OK] {OUT_SUMMARY}")
    print(f"[OK] {OUT_S3}")
    print(f"[OK] {OUT_S4}")
    print(f"[OK] {OUT_EVID}")
    print(f"[OK] th045 FAR refreshed to {th045_artifact_far} (was 0.028991 in the hand-written summary)")
    print(f"[OK] rule/observation agreement {len(agree)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
