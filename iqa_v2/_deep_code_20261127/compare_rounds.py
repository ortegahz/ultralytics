#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两轮 ``constraint_lines_v2.csv`` 的红线对比与 B5 判定工具。

为什么必须有它
--------------
``iqa_v2/DEEP_AXIS_SPEC_20261127.md`` §二.2 判据 **B5** 规定：新红线**不得**在
「比基线更松」的方向上被无条件采纳——若 deep 轮线位**比基线更松**（数值更大），
报告必须写明「口径变化导致线位移动」，并**取两者更严者**作为对外线。

这条判据此前**只靠写报告时手工执行**，没有任何工具支撑 ⇒ 一旦漏做，deep 轮就会
在对外免责声明里悄悄放宽红线。本工具把 B5 变成可复算的机器结论：逐键比对两轮
线位，给出 ``verdict`` / ``final_line`` / ``final_source``，并点名哪些键**必须**在
报告里写明。

红线约定（决定「更严 / 更松」的方向）
------------------------------------
对外免责句式统一为「``quality_metric < line_value`` ⇒ 该档位**可豁免**」——
见 ``iqa_v2/server_out_final/files/REPORT_V2_FINAL.md`` 第 450 行（任务书原文
``target_lap_var < 819.55`` / ``local_scr < 2.69``）、第 454 行（本轮重标
``local_scr < 5.828548`` / ``target_local_contrast < 0.1891664``）、第 462 行
（对外可发布版 ``target_local_contrast < 0.1891664``）——三条核心指标写法一致 ⇒
``line_value`` **越大** ⇒ 可豁免区间越宽 ⇒ 越**松** ⇒ 「取更严者」== **取小者**
（``min``）。该推导固化为常量 :data:`STRICTER_IS_SMALLER`，便于审计。

用法（仓库根执行）::

    python3 iqa_v2/compare_rounds.py --baseline iqa_v2/output/constraint_lines_v2.csv \\
        --deep iqa_v2/output_deep/constraint_lines_v2.csv [--strict] [--eps 1e-9]

输出**全部机器可解析**：每条结论行都是 ``KEY: value`` 形式（``verdict: DEEP_LOOSER`` /
``final_source: baseline`` / ``notes_required: 1`` …），中文只作旁注（``report_note:``
行），不破坏该格式；末尾另附一块人工可读的 markdown 表格（报告作者直接粘贴）。

退出码：0 = 正常；1 = 输入不合法；2 = ``--strict`` 且存在**更松**的线位。

执行机：任何有 CPython 3.8+ 的机器（纯标准库，不需要 cv2 / numpy / 数据集）。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

#: 判定两条线位是否相等的**相对**容差（可由 ``--eps`` 覆盖）。
DEFAULT_EPS: float = 1e-9

#: 「取更严者」的取法常量：``True`` ⇒ 数值**更小**者更严 ⇒ 取 ``min(baseline, deep)``。
#: 推导依据（可审计）：红线约定为「``quality_metric < line_value`` ⇒ 该档位**可豁免**」
#: —— ``iqa_v2/server_out_final/files/REPORT_V2_FINAL.md`` 第 450 行（任务书原文
#: ``target_lap_var < 819.55`` / ``local_scr < 2.69``）、第 454 行（本轮重标
#: ``local_scr < 5.828548`` / ``target_local_contrast < 0.1891664``）、第 462 行
#: （对外可发布版 ``target_local_contrast < 0.1891664``），三条核心指标写法一致 ⇒
#: ``line_value`` 越大 ⇒ 可豁免区间越宽 ⇒ 越**松** ⇒ 更严者 = 数值**更小**者。
STRICTER_IS_SMALLER: bool = True

#: 行的匹配键：``(metric, curve_set, stratum)`` 唯一标识一条红线。
KEY_COLS: tuple[str, ...] = ("metric", "curve_set", "stratum")

#: 必备列（缺任一列 ⇒ 输入不合法 ⇒ 非 0 退出）。
REQUIRED_COLS: tuple[str, ...] = (
    "metric",
    "line_value",
    "deciding_tracker",
    "deciding_family",
    "curve_set",
    "stratum",
)

#: 判定结果：两轮都无有效线位。
VERDICT_BOTH_NONE: str = "BOTH_NONE"
#: 判定结果：基线无线位、deep 引入新线位。
VERDICT_DEEP_NEW: str = "DEEP_NEW"
#: 判定结果：deep 本轮**无**有效线位（对外线不得因此放宽）。
VERDICT_DEEP_NONE: str = "DEEP_NONE"
#: 判定结果：deep 线位**更严**。
VERDICT_DEEP_STRICTER: str = "DEEP_STRICTER"
#: 判定结果：deep 线位**更松**（B5 命中 ⇒ 必须取更严者并写明口径变化）。
VERDICT_DEEP_LOOSER: str = "DEEP_LOOSER"
#: 判定结果：两轮线位一致。
VERDICT_SAME: str = "SAME"

#: 必须由报告写明的判定（其余判定不额外欠账）。
NOTE_VERDICTS: frozenset[str] = frozenset({VERDICT_DEEP_LOOSER, VERDICT_DEEP_NONE})

#: 视作「缺失」的单元格取值。
_MISSING_TOKENS: frozenset[str] = frozenset({"", "nan", "none", "null"})


class CompareRoundsError(Exception):
    """输入不满足规范（文件缺失 / 缺必备列 / 键重复 / 线位非法）⇒ 无法给出可信结论。"""


@dataclass(frozen=True)
class RoundRow:
    """一轮里某个键对应的红线行。"""

    line_value: float
    deciding_tracker: str
    deciding_family: str


@dataclass(frozen=True)
class Comparison:
    """某个键在两轮之间的红线对比结论。"""

    metric: str
    curve_set: str
    stratum: str
    baseline_line: float
    deep_line: float
    verdict: str
    final_line: float
    final_source: str
    deciding_tracker_baseline: str
    deciding_tracker_deep: str
    deciding_tracker_changed: bool
    deciding_family_baseline: str
    deciding_family_deep: str
    deciding_family_changed: bool
    note_needed: bool


def normalize_cell(raw: str | None) -> str:
    """归一化单元格文本：去空白；``""`` / ``nan`` / ``none`` 视作「缺失」。

    Args:
        raw (str | None): 单元格原始文本。

    Returns:
        str: 归一化后的文本（缺失时为空串）。
    """
    text = (raw or "").strip()
    return "" if text.lower() in _MISSING_TOKENS else text


def parse_line_value(raw: str | None) -> float:
    """把 ``line_value`` 单元格解析成浮点数。

    Args:
        raw (str | None): 单元格原始文本。

    Returns:
        float: 线位数值；空 / ``nan`` ⇒ ``nan``（本轮无有效红线）。

    Raises:
        CompareRoundsError: 不是合法浮点数。
    """
    text = (raw or "").strip()
    if text.lower() in _MISSING_TOKENS:
        return math.nan
    try:
        return float(text)
    except ValueError as exc:
        raise CompareRoundsError("line_value 不是合法浮点数：%r" % text) from exc


def read_round(path: Path) -> dict[tuple[str, str, str], RoundRow]:
    """读一轮的 ``constraint_lines_v2.csv``。

    Args:
        path (Path): CSV 路径。

    Returns:
        dict[tuple[str, str, str], RoundRow]: ``{(metric, curve_set, stratum): 行}``。

    Raises:
        CompareRoundsError: 文件不存在、缺必备列、键重复或线位非法。
    """
    if not path.is_file():
        raise CompareRoundsError("文件不存在：%s" % path)
    table: dict[tuple[str, str, str], RoundRow] = {}
    # ``utf-8-sig`` ⇒ 容忍 Excel / Windows 导出的 **BOM**（否则首列名会带 BOM）。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        missing = [col for col in REQUIRED_COLS if col not in header]
        if missing:
            raise CompareRoundsError(
                "缺少必备列 %s（文件 %s 的表头为：%s）"
                % ("、".join(missing), path, "、".join(header))
            )
        for row_index, row in enumerate(reader, start=2):
            key = tuple((row.get(col) or "").strip() for col in KEY_COLS)
            if not any(key):
                continue
            if key in table:
                raise CompareRoundsError(
                    "键重复：%s（第 %d 行）" % ("/".join(key), row_index)
                )
            table[key] = RoundRow(
                line_value=parse_line_value(row.get("line_value")),
                deciding_tracker=normalize_cell(row.get("deciding_tracker")),
                deciding_family=normalize_cell(row.get("deciding_family")),
            )
    return table


def stricter_line(baseline: float, deep: float) -> float:
    """取两条线位中**更严**者。

    方向由 :data:`STRICTER_IS_SMALLER` 固定：``line_value`` 越大 ⇒ 可豁免区间越宽
    ⇒ 越松 ⇒ 更严者 = 数值更小者（``min``）。

    Args:
        baseline (float): 基线轮线位。
        deep (float): deep 轮线位。

    Returns:
        float: 更严的那条线位。
    """
    if STRICTER_IS_SMALLER:
        return min(baseline, deep)
    return max(baseline, deep)


def compare_lines(baseline: float, deep: float, eps: float) -> str:
    """判定两轮线位的对比结论（DEEP_AXIS_SPEC §二.2 判据 B5）。

    Args:
        baseline (float): 基线轮线位（``nan`` = 无有效红线）。
        deep (float): deep 轮线位（``nan`` = 无有效红线）。
        eps (float): 判定「相等」的相对容差。

    Returns:
        str: 判定结果（``BOTH_NONE`` / ``DEEP_NEW`` / ``DEEP_NONE`` /
        ``DEEP_STRICTER`` / ``DEEP_LOOSER`` / ``SAME``）。
    """
    has_baseline = not math.isnan(baseline)
    has_deep = not math.isnan(deep)
    if not has_baseline and not has_deep:
        return VERDICT_BOTH_NONE
    if not has_baseline:
        return VERDICT_DEEP_NEW
    if not has_deep:
        return VERDICT_DEEP_NONE
    if math.isclose(deep, baseline, rel_tol=eps, abs_tol=0.0):
        return VERDICT_SAME
    if deep < baseline:
        return VERDICT_DEEP_STRICTER
    return VERDICT_DEEP_LOOSER


def build_report_note(item: Comparison) -> str:
    """拼出该键在报告里必须写明的中文旁注。

    Args:
        item (Comparison): 对比结论。

    Returns:
        str: 中文旁注文本（无需写明时为空串）。
    """
    parts: list[str] = []
    if item.verdict == VERDICT_DEEP_LOOSER:
        parts.append(
            "口径变化导致线位移动：deep 线位（%s）比基线（%s）更松 ⇒ 对外取更严者（%s）"
            % (
                format_table_number(item.deep_line),
                format_table_number(item.baseline_line),
                format_table_number(item.final_line),
            )
        )
    elif item.verdict == VERDICT_DEEP_NONE:
        parts.append(
            "deep 轮本轮无有效红线（line_value 为空/nan）⇒ 对外线不得放宽，沿用基线线位（%s）"
            % format_table_number(item.final_line)
        )
    if item.deciding_tracker_changed or item.deciding_family_changed:
        parts.append(
            "决定口径发生变化：决定臂 %s → %s、决定族 %s → %s ⇒ 报告须披露"
            % (
                item.deciding_tracker_baseline or "(无)",
                item.deciding_tracker_deep or "(无)",
                item.deciding_family_baseline or "(无)",
                item.deciding_family_deep or "(无)",
            )
        )
    return "；".join(parts)


def compare_rounds(
    baseline_rows: dict[tuple[str, str, str], RoundRow],
    deep_rows: dict[tuple[str, str, str], RoundRow],
    eps: float = DEFAULT_EPS,
) -> list[Comparison]:
    """逐键比对两轮红线。

    Args:
        baseline_rows (dict): 基线轮的行表。
        deep_rows (dict): deep 轮的行表。
        eps (float): 判定「相等」的相对容差。

    Returns:
        list[Comparison]: 按键排序的对比结论列表。
    """
    empty = RoundRow(line_value=math.nan, deciding_tracker="", deciding_family="")
    results: list[Comparison] = []
    for key in sorted(set(baseline_rows) | set(deep_rows)):
        base_row = baseline_rows.get(key, empty)
        deep_row = deep_rows.get(key, empty)
        verdict = compare_lines(base_row.line_value, deep_row.line_value, eps)
        if verdict == VERDICT_BOTH_NONE:
            final_line, final_source = math.nan, "none"
        elif verdict == VERDICT_DEEP_NEW:
            final_line, final_source = deep_row.line_value, "deep"
        elif verdict == VERDICT_DEEP_NONE:
            final_line, final_source = base_row.line_value, "baseline"
        elif verdict == VERDICT_DEEP_STRICTER:
            final_line = stricter_line(base_row.line_value, deep_row.line_value)
            final_source = "deep"
        elif verdict == VERDICT_DEEP_LOOSER:
            final_line = stricter_line(base_row.line_value, deep_row.line_value)
            final_source = "baseline"
        else:
            final_line, final_source = base_row.line_value, "baseline"
        tracker_changed = base_row.deciding_tracker.lower() != deep_row.deciding_tracker.lower()
        family_changed = base_row.deciding_family.lower() != deep_row.deciding_family.lower()
        note_needed = verdict in NOTE_VERDICTS or tracker_changed or family_changed
        results.append(
            Comparison(
                metric=key[0],
                curve_set=key[1],
                stratum=key[2],
                baseline_line=base_row.line_value,
                deep_line=deep_row.line_value,
                verdict=verdict,
                final_line=final_line,
                final_source=final_source,
                deciding_tracker_baseline=base_row.deciding_tracker,
                deciding_tracker_deep=deep_row.deciding_tracker,
                deciding_tracker_changed=tracker_changed,
                deciding_family_baseline=base_row.deciding_family,
                deciding_family_deep=deep_row.deciding_family,
                deciding_family_changed=family_changed,
                note_needed=note_needed,
            )
        )
    return results


def format_number(value: float) -> str:
    """机器可解析的浮点写法（``nan`` 原样；其余用最短可回环表示）。

    Args:
        value (float): 数值。

    Returns:
        str: 输出串。
    """
    if math.isnan(value):
        return "nan"
    return repr(value)


def format_table_number(value: float) -> str:
    """markdown 表格里的浮点写法（保留 8 位有效数字，便于人读）。

    Args:
        value (float): 数值。

    Returns:
        str: 输出串。
    """
    if math.isnan(value):
        return "nan"
    return "%.8g" % value


def summarize(comparisons: list[Comparison]) -> dict[str, int]:
    """统计各判定的计数。

    Args:
        comparisons (list[Comparison]): 对比结论列表。

    Returns:
        dict[str, int]: 计数字典。
    """
    counts = {
        "keys": len(comparisons),
        "looser_count": 0,
        "stricter_count": 0,
        "new_count": 0,
        "none_count": 0,
        "same_count": 0,
        "both_none_count": 0,
        "notes_required": 0,
    }
    for item in comparisons:
        if item.verdict == VERDICT_DEEP_LOOSER:
            counts["looser_count"] += 1
        elif item.verdict == VERDICT_DEEP_STRICTER:
            counts["stricter_count"] += 1
        elif item.verdict == VERDICT_DEEP_NEW:
            counts["new_count"] += 1
        elif item.verdict == VERDICT_DEEP_NONE:
            counts["none_count"] += 1
        elif item.verdict == VERDICT_SAME:
            counts["same_count"] += 1
        else:
            counts["both_none_count"] += 1
        if item.note_needed:
            counts["notes_required"] += 1
    return counts


def print_report(comparisons: list[Comparison]) -> dict[str, int]:
    """打印机器可解析结论 + 人工可读表格。

    Args:
        comparisons (list[Comparison]): 对比结论列表。

    Returns:
        dict[str, int]: 计数字典。
    """
    counts = summarize(comparisons)

    print("=== machine-readable ===")
    for index, item in enumerate(comparisons, start=1):
        print("key_index: %d" % index)
        print("key: %s" % "|".join((item.metric, item.curve_set, item.stratum)))
        print("metric: %s" % item.metric)
        print("curve_set: %s" % item.curve_set)
        print("stratum: %s" % item.stratum)
        print("baseline_line: %s" % format_number(item.baseline_line))
        print("deep_line: %s" % format_number(item.deep_line))
        print("verdict: %s" % item.verdict)
        print("final_line: %s" % format_number(item.final_line))
        print("final_source: %s" % item.final_source)
        print("deciding_tracker_baseline: %s" % item.deciding_tracker_baseline)
        print("deciding_tracker_deep: %s" % item.deciding_tracker_deep)
        print("deciding_tracker_changed: %d" % int(item.deciding_tracker_changed))
        print("deciding_family_baseline: %s" % item.deciding_family_baseline)
        print("deciding_family_deep: %s" % item.deciding_family_deep)
        print("deciding_family_changed: %d" % int(item.deciding_family_changed))
        print("note_needed: %d" % int(item.note_needed))
        note = build_report_note(item)
        if note:
            print("report_note: %s" % note)

    print("")
    print("=== summary ===")
    print("keys: %d" % counts["keys"])
    print("looser_count: %d" % counts["looser_count"])
    print("stricter_count: %d" % counts["stricter_count"])
    print("new_count: %d" % counts["new_count"])
    print("none_count: %d" % counts["none_count"])
    print("same_count: %d" % counts["same_count"])
    print("both_none_count: %d" % counts["both_none_count"])
    print("notes_required: %d" % counts["notes_required"])
    print("verdict: %s" % ("NEEDS_ACTION" if counts["notes_required"] else "OK"))

    print("")
    print("=== markdown table ===")
    print("| metric | curve_set | stratum | baseline | deep | verdict | final_line |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for item in comparisons:
        print(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                item.metric,
                item.curve_set,
                item.stratum,
                format_table_number(item.baseline_line),
                format_table_number(item.deep_line),
                item.verdict,
                format_table_number(item.final_line),
            )
        )
    return counts


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        argparse.ArgumentParser: 解析器。
    """
    parser = argparse.ArgumentParser(
        description="两轮 constraint_lines_v2.csv 的红线对比与 B5 判定（更松 ⇒ 取更严者）")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="基线轮 constraint_lines_v2.csv 路径")
    parser.add_argument("--deep", type=Path, required=True,
                        help="deep 轮 constraint_lines_v2.csv 路径")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                        help="判定两条线位相等的相对容差（默认 %g）" % DEFAULT_EPS)
    parser.add_argument("--strict", action="store_true",
                        help="存在更松的线位时以非 0 退出（报告仍欠「取更严者 + 写明口径变化」）")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。

    Args:
        argv (list[str] | None): 命令行参数（``None`` ⇒ 取 ``sys.argv[1:]``）。

    Returns:
        int: 退出码（0 = 正常；1 = 输入不合法；2 = ``--strict`` 且存在更松的线位）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eps < 0:
        print("error: --eps 不能为负", file=sys.stderr)
        return 1
    try:
        baseline_rows = read_round(args.baseline)
        deep_rows = read_round(args.deep)
    except CompareRoundsError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    counts = print_report(compare_rounds(baseline_rows, deep_rows, args.eps))
    if args.strict and counts["looser_count"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
