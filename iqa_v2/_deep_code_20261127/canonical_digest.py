#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""明细 CSV 的**顺序无关**规范摘要（canonical digest）——复现性证明工具。

为什么必须有它
--------------
``iqa_v2/run_eval_v2.py`` 用多进程池汇总 ``(窗口 × 条件)`` 明细行，收集端是
``pool.imap_unordered(_worker_task, jobs, chunksize=8)`` ⇒ **同一份输入重跑一次，
``degrade_detail_v2.csv`` 的行顺序会变**（内容不变）。

所以对明细 CSV 直接做 ``md5sum`` **不是**有效的复现性证明：它会把「行序抖动」
误判成「数据变化」（上一轮台账据此给出的 md5 校验项必然假失败，且证明不了任何
东西）。本工具改用「先规范化（键唯一 + 规范行串 + 排序）再摘要」的口径：

* 同一份内容、任意行序 ⇒ 摘要必然相同 ⇒ **有效的可复现性证明**；
* 摘要不同 ⇒ 说明内容真的变了 ⇒ 给出**可诊断**的逐列差异清单（哪一行、哪一列、
  改了多少个值），而不是只丢一句「md5 不一样」。

规范化口径（必须与台账/报告里的口径一致，勿随意改动）
--------------------------------------------------
1. 关键列默认 ``seq_id,frame_id,condition_family,condition_level``（一个
   ``(窗口, 条件)`` 组合只出现一次 ⇒ 关键列唯一）；
2. 规范行串 = 该行**按表头顺序**取到的字段值，用 ``\\x1f``（单元分隔符）连接，
   **逐字使用**：不做任何归一化、不做浮点重格式化 ⇒ 数值真的变了摘要就会变；
3. 摘要 = **表头**（按表头顺序用 ``\x1f`` 连接）+ 所有规范行串**排序后**逐行
   （行尾补 ``\n``）拼起来的 ``md5`` ⇒ **只改列名**摘要也会变。

用法（仓库根执行）::

    python3 iqa_v2/canonical_digest.py digest  <csv> [--key-cols a,b,c]
    python3 iqa_v2/canonical_digest.py compare <csvA> <csvB> [--key-cols a,b,c] [--max-report N]
    python3 iqa_v2/canonical_digest.py verify  <sidecar.txt> [--key-cols a,b,c]

``verify`` 核对**旁挂规范摘要清单**（形状同 md5sum 清单：每行 ``<摘要>  <路径>``；空行与
``#`` 开头的行是注释、跳过）⇒ 逐条重算并比对 ⇒ 输出 ``entries:`` /
``entry_<i>_path:`` / ``entry_<i>_want:`` / ``entry_<i>_got:`` /
``entry_<i>_verdict: OK|MISMATCH|MISSING|ERROR`` / ``ok_count:`` / ``failed_count:`` /
``verdict: OK|FAILED``；退出码 0 = 全部一致（任何一条对不上/缺件/算不出摘要即 1）。

门禁接线：``iqa_v2/make_manifest_v2.py`` 把大体量明细的**顺序无关摘要**登记在旁挂清单里
（而不是登记裸 md5 ⇒ 重跑必假失败），门禁第二步就是本工具的 ``verify``。

输出**全部机器可解析**：每条结论行都是 ``KEY: value`` 形式
（``verdict: IDENTICAL`` / ``rows_a: 17657`` / ``digest_a: md5:...`` /
``keys_only_a: 0`` / ``common_differing: 0`` …），中文注解只作旁注、不破坏该格式。

执行机：任何有 CPython 3.8+ 的机器（纯标准库，不需要 cv2 / numpy / 数据集）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import NamedTuple

#: 关键列默认值：``(窗口, 条件)`` 组合在明细表里只出现一次 ⇒ 关键列唯一。
DEFAULT_KEY_COLS: tuple[str, ...] = ("seq_id", "frame_id", "condition_family", "condition_level")

#: 规范行串的字段分隔符（ASCII 单元分隔符 ⇒ 不会与 CSV 内容里的逗号/引号打架）。
FIELD_SEP: str = "\x1f"

#: 摘要计算时每行末尾补的行分隔符。
ROW_SEP: str = "\n"

#: ``compare`` 默认最多列出多少条差异样例。
DEFAULT_MAX_REPORT: int = 20

#: DictReader 对「短行」缺失字段的填充值——用哨兵对象避免和真实的空字符串混淆。
_MISSING: object = object()


class CanonicalDigestError(Exception):
    """输入不满足规范（文件缺失 / 缺关键列 / 关键列重复 / 行残缺）⇒ 无法给出可信摘要。"""


class SidecarEntry(NamedTuple):
    """旁挂规范摘要清单里的一条登记。

    Attributes:
        want (str): 期望的规范摘要（``md5:<hex>``）。
        target (str): 明细 CSV 的路径（原样保留清单里写的字符串）。
        error (str): 该行的格式错误（空串 = 行格式正常）。
    """

    want: str
    target: str
    error: str = ""


def parse_key_cols(raw: str) -> list[str]:
    """把 ``--key-cols a,b,c`` 解析成列名列表（去空白、去重、保序）。

    Args:
        raw (str): 逗号分隔的列名串。

    Returns:
        list[str]: 关键列名列表。

    Raises:
        CanonicalDigestError: 一个列名都没给。
    """
    cols: list[str] = []
    for piece in raw.split(","):
        name = piece.strip()
        if name and name not in cols:
            cols.append(name)
    if not cols:
        raise CanonicalDigestError("--key-cols 不能为空（至少给一个关键列）")
    return cols


def normalize_digest(value: str) -> str:
    """把摘要串规范成 ``md5:<hex>``（裸 hex 也吃得下）。

    Args:
        value (str): 清单里写的摘要串。

    Returns:
        str: 规范后的摘要（``md5:<hex>``）。
    """
    text = value.strip().lower()
    if text.startswith("md5:"):
        text = text[4:]
    return "md5:" + text


def parse_sidecar(path: Path) -> list[SidecarEntry]:
    """读旁挂规范摘要清单 ⇒ 逐条登记。

    清单形状与 md5sum 清单一致：每行 ``<摘要>  <路径>``；空行与 ``#`` 开头的行是注释、跳过。

    Args:
        path (Path): 旁挂清单路径。

    Returns:
        list[SidecarEntry]: 逐条登记（行格式错的也保留，原因写在 ``error`` 里）。

    Raises:
        CanonicalDigestError: 清单文件不存在。
    """
    if not path.is_file():
        raise CanonicalDigestError("旁挂清单不存在：%s" % path)
    entries: list[SidecarEntry] = []
    # ``utf-8-sig`` ⇒ 容忍 Windows 编辑器写出的 **BOM**（否则首行的摘要会被 BOM 污染）。
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            entries.append(SidecarEntry("", line,
                                       "第 %d 行格式不对（应为 <摘要>  <路径>）" % line_no))
            continue
        entries.append(SidecarEntry(normalize_digest(parts[0]), parts[1].strip()))
    return entries


def read_detail(path: Path, key_cols: list[str]) -> tuple[list[str], list[dict]]:
    """读明细 CSV ⇒ ``(表头, 行 dict 列表)``。

    Args:
        path (Path): CSV 路径。
        key_cols (list[str]): 关键列名列表。

    Returns:
        tuple[list[str], list[dict]]: 表头列表与行列表（行是 DictReader 的 dict）。

    Raises:
        CanonicalDigestError: 文件不存在、缺关键列、或出现残缺行（字段数与表头不符）。
    """
    if not path.is_file():
        raise CanonicalDigestError("文件不存在：%s" % path)
    # ``utf-8-sig`` ⇒ 容忍 Excel / Windows 导出的 **BOM**（否则表头首列名会带 BOM
    # ⇒ 误报「缺少关键列 seq_id」）。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, restval=_MISSING)
        header = list(reader.fieldnames or [])
        missing = [col for col in key_cols if col not in header]
        if missing:
            raise CanonicalDigestError(
                "缺少关键列 %s（文件 %s 的表头为：%s）"
                % ("、".join(missing), path, "、".join(header) if header else "<空表头>"))
        rows: list[dict] = []
        for row in reader:
            line_no = reader.line_num
            if None in row:
                raise CanonicalDigestError(
                    "第 %d 行字段数多于表头 %d 列（文件 %s）：明细 CSV 必须是矩形"
                    % (line_no, len(header), path))
            if any(value is _MISSING for value in row.values()):
                raise CanonicalDigestError(
                    "第 %d 行字段数少于表头 %d 列（文件 %s）：明细 CSV 必须是矩形"
                    % (line_no, len(header), path))
            rows.append(row)
    return header, rows


def canonical_row(header: list[str], row: dict) -> str:
    """一行 ⇒ 规范行串：按**表头顺序**取值，用 ``\\x1f`` 连接（逐字使用，不做任何归一化）。

    Args:
        header (list[str]): 表头列名列表。
        row (dict): 行字典。

    Returns:
        str: 该行的规范行串。
    """
    return FIELD_SEP.join(row[col] for col in header)


def compute_digest(header: list[str], rows: list[dict]) -> str:
    """规范摘要 = ``md5``(排序后的规范行串，每行以 ``\\n`` 结尾)。

    Args:
        header (list[str]): 表头列名列表。
        rows (list[dict]): 行列表。

    Returns:
        str: ``md5:<hex>`` 形式的摘要。
    """
    lines = sorted(canonical_row(header, row) for row in rows)
    digest = hashlib.md5()
    # 🔴 **表头也进摘要** ⇒ 只改列名（值不变）时摘要**必须变**；
    #    否则「只换列名」这种数据变化会被摘要漏掉。
    digest.update(FIELD_SEP.join(header).encode("utf-8"))
    digest.update(ROW_SEP.encode("utf-8"))
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(ROW_SEP.encode("utf-8"))
    return "md5:" + digest.hexdigest()


def format_key(key: tuple, key_cols: list[str]) -> str:
    """把关键列取值元组格式化成 ``列=值|列=值`` 形式（供人读）。

    Args:
        key (tuple): 关键列取值元组。
        key_cols (list[str]): 关键列名列表。

    Returns:
        str: 可读的键串。
    """
    return "|".join("%s=%s" % (col, value) for col, value in zip(key_cols, key))


def load_table(path: Path, key_cols: list[str]) -> dict:
    """读表 + 校验关键列唯一 + 算摘要 ⇒ 供 ``digest`` / ``compare`` 共用的结果包。

    Args:
        path (Path): CSV 路径。
        key_cols (list[str]): 关键列名列表。

    Returns:
        dict: 含 ``path`` / ``header`` / ``rows`` / ``by_key`` / ``digest`` 的字典。

    Raises:
        CanonicalDigestError: 输入不合法（含关键列重复）。
    """
    header, rows = read_detail(path, key_cols)
    by_key: dict[tuple, dict] = {}
    duplicates: list[tuple] = []
    for row in rows:
        key = tuple(row[col] for col in key_cols)
        if key in by_key:
            duplicates.append(key)
        else:
            by_key[key] = row
    if duplicates:
        raise CanonicalDigestError(
            "关键列 (%s) 存在重复取值：%s（本次共 %d 行重复）——明细表要求 (窗口, 条件) 唯一"
            % ("、".join(key_cols), format_key(duplicates[0], key_cols), len(duplicates)))
    return {"path": path, "header": header, "rows": rows, "by_key": by_key,
            "digest": compute_digest(header, rows)}


def cmd_digest(path: Path, key_cols: list[str]) -> int:
    """``digest`` 子命令：打印单个 CSV 的规范摘要 + 行数 + 列数。

    Args:
        path (Path): CSV 路径。
        key_cols (list[str]): 关键列名列表。

    Returns:
        int: 退出码（0 = 成功）。
    """
    table = load_table(path, key_cols)
    print("digest: %s" % table["digest"])
    print("rows: %d" % len(table["rows"]))
    print("columns: %d" % len(table["header"]))
    print("key_cols: %s" % ",".join(key_cols))
    return 0


def cmd_compare(path_a: Path, path_b: Path, key_cols: list[str], max_report: int) -> int:
    """``compare`` 子命令：比较两个 CSV，给出机器可解析的结论 + 可诊断的差异清单。

    Args:
        path_a (Path): CSV A 路径。
        path_b (Path): CSV B 路径。
        key_cols (list[str]): 关键列名列表。
        max_report (int): 每类差异最多列出的样例条数。

    Returns:
        int: 退出码（0 = 完全一致；1 = 表头/内容不一致）。
    """
    table_a = load_table(path_a, key_cols)
    table_b = load_table(path_b, key_cols)
    header_a, header_b = table_a["header"], table_b["header"]

    # ---- 表头必须逐列一致（顺序也算）⇒ 否则列对齐关系不可信，差异清单无意义
    if header_a != header_b:
        only_a = [col for col in header_a if col not in header_b]
        only_b = [col for col in header_b if col not in header_a]
        first_diff = next((i for i, (x, y) in enumerate(zip(header_a, header_b)) if x != y), None)
        print("verdict: HEADER_MISMATCH")
        print("columns_a: %d" % len(header_a))
        print("columns_b: %d" % len(header_b))
        print("header_only_a: %s" % ",".join(only_a))
        print("header_only_b: %s" % ",".join(only_b))
        print("header_order_differs: %d" % int(not only_a and not only_b))
        print("header_first_diff_pos: %s" % ("" if first_diff is None else first_diff))
        print("error: 表头不一致——A 独有列 [%s]、B 独有列 [%s]（列集合相同则为顺序不同）"
              % (",".join(only_a), ",".join(only_b)), file=sys.stderr)
        return 1

    # ---- 行数、摘要
    print("rows_a: %d" % len(table_a["rows"]))
    print("rows_b: %d" % len(table_b["rows"]))
    print("columns_a: %d" % len(header_a))
    print("columns_b: %d" % len(header_b))
    print("digest_a: %s" % table_a["digest"])
    print("digest_b: %s" % table_b["digest"])

    if table_a["digest"] == table_b["digest"]:
        print("verdict: IDENTICAL")
        print("keys_only_a: 0")
        print("keys_only_b: 0")
        print("common_differing: 0")
        return 0

    # ---- 摘要不同 ⇒ 内容真的变了 ⇒ 拆到键级别做可诊断的差异
    map_a, map_b = table_a["by_key"], table_b["by_key"]
    keys_a, keys_b = set(map_a), set(map_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = keys_a & keys_b

    differing: list[tuple] = []
    col_counts: dict[str, int] = {}
    for key in sorted(common):
        row_a, row_b = map_a[key], map_b[key]
        diff_cols = [col for col in header_a if row_a[col] != row_b[col]]
        if diff_cols:
            differing.append((key, diff_cols))
            for col in diff_cols:
                col_counts[col] = col_counts.get(col, 0) + 1

    print("verdict: DIFFERENT")
    print("keys_only_a: %d" % len(only_a))
    print("keys_only_b: %d" % len(only_b))
    print("common_differing: %d" % len(differing))
    # 逐列差异计数（表头一致 ⇒ 对 A/B 两侧对称；先看列 ⇒ 回归可归因到具体列）
    for col in header_a:
        count = col_counts.get(col, 0)
        if count:
            print("col_diff_%s: %d" % (col, count))

    for index, key in enumerate(only_a[:max_report], start=1):
        print("keys_only_a_example_%d: %s" % (index, format_key(key, key_cols)))
    for index, key in enumerate(only_b[:max_report], start=1):
        print("keys_only_b_example_%d: %s" % (index, format_key(key, key_cols)))
    for index, (key, diff_cols) in enumerate(differing[:max_report], start=1):
        row_a, row_b = map_a[key], map_b[key]
        print("diff_example_%d_key: %s" % (index, format_key(key, key_cols)))
        print("diff_example_%d_columns: %s" % (index, ",".join(diff_cols)))
        print("diff_example_%d_value_a: %s"
              % (index, ",".join("%s=%s" % (col, row_a[col]) for col in diff_cols)))
        print("diff_example_%d_value_b: %s"
              % (index, ",".join("%s=%s" % (col, row_b[col]) for col in diff_cols)))
    return 1


def cmd_verify(sidecar: Path, key_cols: list[str]) -> int:
    """``verify`` 子命令：按旁挂清单逐条核对明细 CSV 的**顺序无关**规范摘要。

    Args:
        sidecar (Path): 旁挂规范摘要清单路径。
        key_cols (list[str]): 关键列名列表。

    Returns:
        int: 退出码（0 = 全部一致；1 = 有条目对不上/缺件/算不出摘要）。
    """
    entries = parse_sidecar(sidecar)
    print("entries: %d" % len(entries))
    ok_count = 0
    failed_count = 0
    for index, entry in enumerate(entries, start=1):
        print("entry_%d_path: %s" % (index, entry.target))
        print("entry_%d_want: %s" % (index, entry.want))
        if entry.error:
            print("entry_%d_got: " % index)
            print("entry_%d_verdict: ERROR" % index)
            print("entry_%d_error: %s" % (index, entry.error))
            failed_count += 1
            continue
        target = Path(entry.target)
        if not target.is_file():
            print("entry_%d_got: " % index)
            print("entry_%d_verdict: MISSING" % index)
            failed_count += 1
            continue
        try:
            got = load_table(target, key_cols)["digest"]
        except CanonicalDigestError as exc:
            print("entry_%d_got: " % index)
            print("entry_%d_verdict: ERROR" % index)
            print("entry_%d_error: %s" % (index, exc))
            failed_count += 1
            continue
        print("entry_%d_got: %s" % (index, got))
        if got == entry.want:
            print("entry_%d_verdict: OK" % index)
            ok_count += 1
        else:
            print("entry_%d_verdict: MISMATCH" % index)
            failed_count += 1
    print("ok_count: %d" % ok_count)
    print("failed_count: %d" % failed_count)
    print("verdict: %s" % ("OK" if not failed_count else "FAILED"))
    return 0 if not failed_count else 1


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        argparse.ArgumentParser: 解析器。
    """
    parser = argparse.ArgumentParser(
        description="明细 CSV 的顺序无关规范摘要（与行序无关的复现性证明）")
    sub = parser.add_subparsers(dest="command", required=True)

    digest = sub.add_parser("digest", help="打印单个 CSV 的规范摘要")
    digest.add_argument("csv", type=Path, help="明细 CSV 路径")
    digest.add_argument("--key-cols", default=",".join(DEFAULT_KEY_COLS),
                        help="关键列（逗号分隔，默认 %s）" % ",".join(DEFAULT_KEY_COLS))

    compare = sub.add_parser("compare", help="比较两个 CSV")
    compare.add_argument("csv_a", type=Path, help="CSV A 路径")
    compare.add_argument("csv_b", type=Path, help="CSV B 路径")
    compare.add_argument("--key-cols", default=",".join(DEFAULT_KEY_COLS),
                         help="关键列（逗号分隔，默认 %s）" % ",".join(DEFAULT_KEY_COLS))
    compare.add_argument("--max-report", type=int, default=DEFAULT_MAX_REPORT,
                         help="每类差异最多列出的样例条数（默认 %d）" % DEFAULT_MAX_REPORT)

    verify = sub.add_parser("verify", help="按旁挂规范摘要清单逐条核对明细 CSV")
    verify.add_argument("sidecar", type=Path, help="旁挂规范摘要清单路径")
    verify.add_argument("--key-cols", default=",".join(DEFAULT_KEY_COLS),
                        help="关键列（逗号分隔，默认 %s）" % ",".join(DEFAULT_KEY_COLS))
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。

    Args:
        argv (list[str] | None): 命令行参数（``None`` ⇒ 取 ``sys.argv[1:]``）。

    Returns:
        int: 退出码（0 = 成功 / 完全一致 / 旁挂清单逐条一致；1 = 输入不合法、
            两文件不一致或清单核对失败）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        key_cols = parse_key_cols(args.key_cols)
        if args.command == "digest":
            return cmd_digest(args.csv, key_cols)
        if args.command == "verify":
            return cmd_verify(args.sidecar, key_cols)
        return cmd_compare(args.csv_a, args.csv_b, key_cols, args.max_report)
    except CanonicalDigestError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
