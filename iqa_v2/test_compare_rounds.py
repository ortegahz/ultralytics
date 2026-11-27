#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``compare_rounds.py`` 的验收测试（``pytest`` 全绿）。

被测对象：``iqa_v2/compare_rounds.py`` —— 两轮 ``constraint_lines_v2.csv`` 的红线
对比与 **B5** 判定（``DEEP_AXIS_SPEC_20261127.md`` §二.2 判据 B5）。

为什么要有这组测试
------------------
B5「新红线不得在更松的方向上被无条件采纳」此前**只靠人工写报告时执行** ⇒ 没有
任何工具支撑。本组测试把这条口径钉死：

1. 同一份 CSV 自比 ⇒ 逐键 ``SAME``、``verdict: OK``、退出码 0；
2. deep 线位**更松** ⇒ ``DEEP_LOOSER`` + ``final_source: baseline``（取更严者）+
   ``note_needed: 1``，且 ``--strict`` 必须非 0 退出；
3. deep 线位**更严** ⇒ ``DEEP_STRICTER`` + ``final_source: deep``；
4. deep **无线位** ⇒ ``DEEP_NONE``（对外线不得放宽，沿用基线）；基线无线位而 deep
   有 ⇒ ``DEEP_NEW``；两边都没有 ⇒ ``BOTH_NONE``；
5. 相对容差内的小差异算 ``SAME``；
6. 决定臂 / 决定族变了 ⇒ ``deciding_*_changed: 1`` 且报告欠一条说明；
7. 文件缺失 / 缺必备列 ⇒ 非 0 退出并说明原因。

复现::

    cd <仓库根> && python3 -m pytest iqa_v2/test_compare_rounds.py -v
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compare_rounds import main  # noqa: E402

#: 仓库自带基线轮的产物（用于「自比」这条真实路径）。
REPO_BASELINE: Path = HERE / "output" / "constraint_lines_v2.csv"

#: ``constraint_lines_v2.csv`` 的真实表头（测试用同一套列 ⇒ 贴近真实产物）。
HEADER: list[str] = [
    "metric",
    "line_value",
    "deciding_tracker",
    "deciding_family",
    "curve_set",
    "stratum",
    "xcorr_line",
    "kcf_legacy_line",
    "csrt_line",
    "cross_arm_dev_pct",
    "xcorr_ref",
    "kcf_legacy_ref",
    "csrt_ref",
    "n_candidates",
    "note",
]


def make_row(
    metric: str,
    line_value: str,
    tracker: str = "xcorr",
    family: str = "motion",
    curve_set: str = "common_good",
    stratum: str = "all",
) -> dict[str, str]:
    """造一行红线记录。

    Args:
        metric (str): 指标名。
        line_value (str): 线位（空串 / ``nan`` = 本轮无有效红线）。
        tracker (str): 决定臂。
        family (str): 决定族。
        curve_set (str): 曲线集合名。
        stratum (str): 分层名。

    Returns:
        dict[str, str]: 行记录。
    """
    return {
        "metric": metric,
        "line_value": line_value,
        "deciding_tracker": tracker,
        "deciding_family": family,
        "curve_set": curve_set,
        "stratum": stratum,
    }


def write_round(path: Path, rows: list[dict[str, str]]) -> Path:
    """写一份 ``constraint_lines_v2.csv``。

    Args:
        path (Path): 目标路径。
        rows (list[dict[str, str]]): 行记录。

    Returns:
        Path: 目标路径。
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in HEADER})
    return path


def parse_blocks(stdout: str) -> list[dict[str, str]]:
    """把机器可解析区块按 ``key_index`` 拆成逐键的结论字典。

    Args:
        stdout (str): CLI 的标准输出。

    Returns:
        list[dict[str, str]]: 逐键结论字典列表。
    """
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    section = ""
    for line in stdout.splitlines():
        if line.startswith("===") and line.endswith("==="):
            if section == "machine-readable" and current:
                blocks.append(current)
                current = {}
            section = line.strip("= ")
            continue
        if section != "machine-readable" or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key == "key_index" and current:
            blocks.append(current)
            current = {}
        current[key] = value
    if current:
        blocks.append(current)
    return blocks


def parse_summary(stdout: str) -> dict[str, str]:
    """解析 ``=== summary ===`` 区块。

    Args:
        stdout (str): CLI 的标准输出。

    Returns:
        dict[str, str]: 键值对。
    """
    fields: dict[str, str] = {}
    in_summary = False
    for line in stdout.splitlines():
        if line.startswith("===") and line.endswith("==="):
            in_summary = line.strip("= ") == "summary"
            continue
        if in_summary and ": " in line:
            key, value = line.split(": ", 1)
            fields[key] = value
    return fields


def run_cli(capsys: pytest.CaptureFixture, *args: str) -> tuple[int, str]:
    """跑一次 CLI 并回收退出码与标准输出。

    Args:
        capsys (pytest.CaptureFixture): pytest 的输出捕获夹具。
        *args (str): 命令行参数。

    Returns:
        tuple[int, str]: ``(退出码, 标准输出)``。
    """
    code = main(list(args))
    return code, capsys.readouterr().out


def test_identical_rounds_are_same(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """同一份 CSV 自比 ⇒ 逐键 ``SAME``、无欠账、退出码 0。"""
    csv_path = write_round(
        tmp_path / "constraint_lines_v2.csv",
        [
            make_row("target_local_contrast", "0.18916642395250532"),
            make_row("local_scr", "5.828548284248887", family="blur"),
            make_row("target_lap_var", "nan", tracker="", family=""),
        ],
    )
    code, out = run_cli(
        capsys, "--baseline", str(csv_path), "--deep", str(csv_path))
    blocks = parse_blocks(out)
    summary = parse_summary(out)
    verdicts = {item["metric"]: item["verdict"] for item in blocks}
    assert code == 0
    assert verdicts == {
        "target_local_contrast": "SAME",
        "local_scr": "SAME",
        "target_lap_var": "BOTH_NONE",
    }
    assert all(item["note_needed"] == "0" for item in blocks)
    assert summary["keys"] == "3"
    assert summary["same_count"] == "2"
    assert summary["both_none_count"] == "1"
    assert summary["looser_count"] == "0"
    assert summary["notes_required"] == "0"
    assert summary["verdict"] == "OK"


def test_repo_baseline_against_itself_is_same(capsys: pytest.CaptureFixture) -> None:
    """仓库自带的基线轮产物自比 ⇒ 全 ``SAME``（真实路径，不造数）。"""
    code, out = run_cli(
        capsys, "--baseline", str(REPO_BASELINE), "--deep", str(REPO_BASELINE))
    blocks = parse_blocks(out)
    summary = parse_summary(out)
    assert code == 0
    assert blocks
    assert all(item["verdict"] in {"SAME", "BOTH_NONE"} for item in blocks)
    assert summary["looser_count"] == "0"
    assert summary["notes_required"] == "0"
    assert summary["verdict"] == "OK"


def test_deeper_line_takes_stricter_baseline(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """deep 线位**更松** ⇒ 取更严者（基线线位）并欠一条说明。"""
    baseline = write_round(
        tmp_path / "baseline.csv", [make_row("target_local_contrast", "0.18916642395250532")])
    deep = write_round(
        tmp_path / "deep.csv", [make_row("target_local_contrast", "0.21200000000000002")])
    code, out = run_cli(capsys, "--baseline", str(baseline), "--deep", str(deep))
    block = parse_blocks(out)[0]
    summary = parse_summary(out)
    assert code == 0
    assert block["verdict"] == "DEEP_LOOSER"
    assert float(block["final_line"]) == pytest.approx(0.18916642395250532)
    assert block["final_source"] == "baseline"
    assert block["note_needed"] == "1"
    assert "口径变化导致线位移动" in block["report_note"]
    assert summary["looser_count"] == "1"
    assert summary["notes_required"] == "1"
    assert summary["verdict"] == "NEEDS_ACTION"


def test_strict_flag_fails_on_looser(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``--strict`` 命中「更松」线位 ⇒ 非 0 退出。"""
    baseline = write_round(
        tmp_path / "baseline.csv", [make_row("local_scr", "5.828548284248887", family="blur")])
    deep = write_round(
        tmp_path / "deep.csv", [make_row("local_scr", "6.5", family="blur")])
    code, out = run_cli(
        capsys, "--baseline", str(baseline), "--deep", str(deep), "--strict")
    assert code != 0
    assert parse_summary(out)["looser_count"] == "1"


def test_strict_flag_passes_when_not_looser(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """``--strict`` 且没有更松线位 ⇒ 退出码 0。"""
    baseline = write_round(
        tmp_path / "baseline.csv", [make_row("local_scr", "5.828548284248887", family="blur")])
    deep = write_round(
        tmp_path / "deep.csv", [make_row("local_scr", "4.5", family="blur")])
    code, out = run_cli(
        capsys, "--baseline", str(baseline), "--deep", str(deep), "--strict")
    assert code == 0
    assert parse_summary(out)["looser_count"] == "0"


def test_stricter_line_takes_deep(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """deep 线位**更严** ⇒ 直接采用 deep 线位。"""
    baseline = write_round(
        tmp_path / "baseline.csv", [make_row("local_scr", "5.828548284248887", family="blur")])
    deep = write_round(
        tmp_path / "deep.csv", [make_row("local_scr", "4.5", family="blur")])
    code, out = run_cli(capsys, "--baseline", str(baseline), "--deep", str(deep))
    block = parse_blocks(out)[0]
    summary = parse_summary(out)
    assert code == 0
    assert block["verdict"] == "DEEP_STRICTER"
    assert float(block["final_line"]) == pytest.approx(4.5)
    assert block["final_source"] == "deep"
    assert block["note_needed"] == "0"
    assert summary["stricter_count"] == "1"


def test_deep_none_keeps_baseline(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """deep 本轮无有效红线 ⇒ 对外线不得放宽，沿用基线线位并欠一条说明。"""
    baseline = write_round(
        tmp_path / "baseline.csv", [make_row("target_lap_var", "819.55")])
    deep = write_round(
        tmp_path / "deep.csv", [make_row("target_lap_var", "nan", tracker="", family="")])
    code, out = run_cli(capsys, "--baseline", str(baseline), "--deep", str(deep))
    block = parse_blocks(out)[0]
    summary = parse_summary(out)
    assert code == 0
    assert block["verdict"] == "DEEP_NONE"
    assert float(block["final_line"]) == pytest.approx(819.55)
    assert block["final_source"] == "baseline"
    assert block["note_needed"] == "1"
    assert summary["none_count"] == "1"
    assert summary["notes_required"] == "1"


def test_deep_new_line(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """基线无线位、deep 引入新线位 ⇒ 采用 deep 线位。"""
    baseline = write_round(
        tmp_path / "baseline.csv", [make_row("target_lap_var", "nan", tracker="", family="")])
    deep = write_round(tmp_path / "deep.csv", [make_row("target_lap_var", "541.09")])
    code, out = run_cli(capsys, "--baseline", str(baseline), "--deep", str(deep))
    block = parse_blocks(out)[0]
    summary = parse_summary(out)
    assert code == 0
    assert block["verdict"] == "DEEP_NEW"
    assert float(block["final_line"]) == pytest.approx(541.09)
    assert block["final_source"] == "deep"
    assert summary["new_count"] == "1"


def test_both_none(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """两轮都无有效红线 ⇒ ``BOTH_NONE``、最终无线位。"""
    baseline = write_round(
        tmp_path / "baseline.csv", [make_row("target_lap_var", "nan", tracker="", family="")])
    deep = write_round(
        tmp_path / "deep.csv", [make_row("target_lap_var", "", tracker="", family="")])
    code, out = run_cli(capsys, "--baseline", str(baseline), "--deep", str(deep))
    block = parse_blocks(out)[0]
    assert code == 0
    assert block["verdict"] == "BOTH_NONE"
    assert block["final_line"] == "nan"
    assert block["final_source"] == "none"


def test_within_eps_is_same(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """相对容差内的小差异 ⇒ 判 ``SAME``（不虚报口径变化）。"""
    baseline = write_round(
        tmp_path / "baseline.csv", [make_row("local_scr", "5.828548284248887", family="blur")])
    deep = write_round(
        tmp_path / "deep.csv", [make_row("local_scr", "5.828548284248892", family="blur")])
    code, out = run_cli(capsys, "--baseline", str(baseline), "--deep", str(deep))
    block = parse_blocks(out)[0]
    assert code == 0
    assert block["verdict"] == "SAME"


def test_deciding_arm_change_is_disclosed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """决定臂 / 决定族变了 ⇒ 属「口径变化」，必须披露。"""
    baseline = write_round(
        tmp_path / "baseline.csv",
        [make_row("target_local_contrast", "0.189", tracker="xcorr", family="motion")])
    deep = write_round(
        tmp_path / "deep.csv",
        [make_row("target_local_contrast", "0.189", tracker="kcf_legacy", family="blur")])
    code, out = run_cli(capsys, "--baseline", str(baseline), "--deep", str(deep))
    block = parse_blocks(out)[0]
    summary = parse_summary(out)
    assert code == 0
    assert block["verdict"] == "SAME"
    assert block["deciding_tracker_baseline"] == "xcorr"
    assert block["deciding_tracker_deep"] == "kcf_legacy"
    assert block["deciding_tracker_changed"] == "1"
    assert block["deciding_family_changed"] == "1"
    assert block["note_needed"] == "1"
    assert summary["notes_required"] == "1"
    assert summary["verdict"] == "NEEDS_ACTION"


def test_key_missing_in_deep_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """deep 文件里**没有该键** ⇒ 视作 deep 无有效红线（对外线不得放宽）。"""
    baseline = write_round(
        tmp_path / "baseline.csv",
        [
            make_row("target_local_contrast", "0.189"),
            make_row("local_scr", "5.828", family="blur"),
        ])
    deep = write_round(tmp_path / "deep.csv", [make_row("local_scr", "5.828", family="blur")])
    code, out = run_cli(capsys, "--baseline", str(baseline), "--deep", str(deep))
    blocks = {item["metric"]: item for item in parse_blocks(out)}
    assert code == 0
    assert blocks["target_local_contrast"]["verdict"] == "DEEP_NONE"
    assert blocks["target_local_contrast"]["final_source"] == "baseline"
    assert blocks["local_scr"]["verdict"] == "SAME"
    assert parse_summary(out)["keys"] == "2"


def test_missing_file_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """输入文件不存在 ⇒ 非 0 退出并说明原因。"""
    baseline = write_round(tmp_path / "baseline.csv", [make_row("local_scr", "5.8", family="blur")])
    code = main(["--baseline", str(baseline), "--deep", str(tmp_path / "nope.csv")])
    assert code == 1
    assert "文件不存在" in capsys.readouterr().err


def test_missing_required_column_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """缺必备列 ⇒ 非 0 退出并点名缺哪一列。"""
    bad = tmp_path / "bad.csv"
    with bad.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "curve_set", "stratum"])
        writer.writerow(["local_scr", "common_good", "all"])
    baseline = write_round(tmp_path / "baseline.csv", [make_row("local_scr", "5.8", family="blur")])
    code = main(["--baseline", str(baseline), "--deep", str(bad)])
    assert code == 1
    err = capsys.readouterr().err
    assert "line_value" in err
    assert "deciding_tracker" in err


def test_duplicate_key_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """同一键出现两行 ⇒ 非 0 退出（歧义数据不可给结论）。"""
    dup = tmp_path / "dup.csv"
    with dup.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerow(["local_scr", "5.8", "xcorr", "blur", "common_good", "all"])
        writer.writerow(["local_scr", "5.9", "xcorr", "blur", "common_good", "all"])
    baseline = write_round(tmp_path / "baseline.csv", [make_row("local_scr", "5.8", family="blur")])
    code = main(["--baseline", str(baseline), "--deep", str(dup)])
    assert code == 1
    assert "键重复" in capsys.readouterr().err


def test_markdown_table_is_printed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """人工可读的 markdown 表格单独成块，供报告直接粘贴。"""
    baseline = write_round(
        tmp_path / "baseline.csv", [make_row("target_local_contrast", "0.18916642395250532")])
    deep = write_round(
        tmp_path / "deep.csv", [make_row("target_local_contrast", "0.212")])
    _code, out = run_cli(capsys, "--baseline", str(baseline), "--deep", str(deep))
    assert "=== markdown table ===" in out
    table = out.split("=== markdown table ===", 1)[1].strip().splitlines()
    assert table[0].startswith(
        "| metric | curve_set | stratum | baseline | deep | verdict | final_line |")
    assert table[2].startswith("| target_local_contrast | common_good | all |")
    assert "DEEP_LOOSER" in table[2]
