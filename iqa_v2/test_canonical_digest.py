#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``canonical_digest.py`` 的验收测试（``pytest`` 全绿）。

被测对象：``iqa_v2/canonical_digest.py`` —— 明细 CSV 的**顺序无关**规范摘要。
（以及 ``iqa_v2/make_manifest_v2.py`` —— 清单工具：明细只登记顺序无关摘要。）

为什么要有这组测试
------------------
``run_eval_v2.py`` 用 ``pool.imap_unordered(...)`` 收集明细行 ⇒ 同一份输入重跑一次，
CSV 的**行顺序会变**。因此「对 CSV 直接做 md5sum」不是有效的复现性证明（会把行序
抖动误判成数据变化）。本组测试把这条口径钉死：

1. 行序不同 ⇒ 摘要相同 ⇒ ``compare`` 判 ``IDENTICAL`` 且退出码 0；
2. 真的改了一个值 ⇒ 退出码非 0、``common_differing`` 恰好 1、并点名改动的列；
3. A 少一行 / 多一行 ⇒ 分别报 ``keys_only_a: 1`` / ``keys_only_b: 1``；
4. 表头不同 / 关键列重复 / 缺关键列 ⇒ 一律非 0 退出并点名相关列。
5. ``verify``（旁挂规范摘要清单）⇒ 行序不同仍判 ``OK``、内容被改判 ``MISMATCH``、
   缺件判 ``MISSING``，且退出码一一对应；
6. 清单工具 ⇒ 大体量明细**只**进旁挂规范摘要清单、不再作为裸 md5 行进清单
   （``--bulk-mode raw`` 仍能复现历史形状）。

复现::

    cd <仓库根> && python3 -m pytest iqa_v2/test_canonical_digest.py -v
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from canonical_digest import main  # noqa: E402

#: 一个小而真实的明细表片段（列名取自 ``degrade_detail_v2.csv`` 的前几列 + 两个指标列）。
HEADER: list[str] = ["seq_id", "frame_id", "condition_family", "condition_level",
                     "local_scr", "xcorr_cle"]

ROWS: list[list[str]] = [
    ["S1", "10", "clean", "0.0", "15.263246789670164", "6.5809822592002085"],
    ["S1", "10", "blur", "2.0", "9.5", "8.25"],
    ["S2", "20", "clean", "0.0", "12.0", "4.5"],
]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    """写一个小 CSV（带表头）。

    Args:
        path (Path): 目标路径。
        header (list[str]): 表头列名。
        rows (list[list[str]]): 数据行。

    Returns:
        Path: 目标路径。
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def parse_fields(stdout: str) -> dict[str, str]:
    """把 ``KEY: value`` 输出解析成字典。

    Args:
        stdout (str): CLI 的标准输出。

    Returns:
        dict[str, str]: 键值对。
    """
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            fields.setdefault(key, value)
    return fields


def run_cli(args: list[str], capsys: pytest.CaptureFixture) -> tuple[int, str, str]:
    """跑一次 CLI（进程内），返回 ``(退出码, stdout, stderr)``。

    Args:
        args (list[str]): 命令行参数。
        capsys (pytest.CaptureFixture): pytest 的输出捕获夹具。

    Returns:
        tuple[int, str, str]: 退出码、标准输出、标准错误。
    """
    code = main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_shuffled_rows_give_identical_digest(tmp_path: Path, capsys: pytest.CaptureFixture):
    """**行序不同 ⇒ 摘要相同**（这正是 imap_unordered 场景下的复现性口径）。"""
    path_a = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    path_b = write_csv(tmp_path / "b.csv", HEADER, list(reversed(ROWS)))

    code_a, out_a, _ = run_cli(["digest", str(path_a)], capsys)
    code_b, out_b, _ = run_cli(["digest", str(path_b)], capsys)
    assert code_a == 0 and code_b == 0, "digest 必须成功退出"
    digest_a = parse_fields(out_a)["digest"]
    digest_b = parse_fields(out_b)["digest"]
    assert digest_a == digest_b, "行序不同但内容相同 ⇒ 摘要必须相同（%s vs %s）" % (digest_a, digest_b)
    assert digest_a.startswith("md5:"), "摘要必须带 md5: 前缀"

    code, out, _ = run_cli(["compare", str(path_a), str(path_b)], capsys)
    fields = parse_fields(out)
    assert code == 0, "内容一致 ⇒ 退出码必须是 0（实际 %d）" % code
    assert fields["verdict"] == "IDENTICAL", "结论必须是 IDENTICAL（实际 %s）" % fields["verdict"]
    assert fields["rows_a"] == str(len(ROWS)), "rows_a 必须是行数"
    assert fields["common_differing"] == "0", "一致时 common_differing 必须是 0"


def test_shuffled_rows_identical_through_cli(tmp_path: Path):
    """端到端：真实子进程调用，退出码必须真的传到 shell（0 = 一致）。"""
    path_a = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    path_b = write_csv(tmp_path / "b.csv", HEADER, list(reversed(ROWS)))
    result = subprocess.run(
        [sys.executable, str(HERE / "canonical_digest.py"), "compare",
         str(path_a), str(path_b)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, "子进程退出码必须是 0（实际 %d）\n%s" % (
        result.returncode, result.stderr)
    assert "verdict: IDENTICAL" in result.stdout, "标准输出必须含 verdict: IDENTICAL"


def test_changed_value_is_attributed_to_its_column(tmp_path: Path, capsys: pytest.CaptureFixture):
    """**改了一个值** ⇒ 非 0 退出、恰好 1 行公共差异、并且点名是哪个列变了。"""
    changed = [list(row) for row in ROWS]
    changed[1][4] = "9.75"          # 只改 S1/blur 那一行的 local_scr
    path_a = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    path_b = write_csv(tmp_path / "b.csv", HEADER, changed)

    code, out, _ = run_cli(["compare", str(path_a), str(path_b)], capsys)
    fields = parse_fields(out)
    assert code != 0, "内容不同 ⇒ 退出码必须非 0"
    assert fields["verdict"] == "DIFFERENT", "结论必须是 DIFFERENT"
    assert fields["common_differing"] == "1", "恰好 1 行公共差异（实际 %s）" % fields["common_differing"]
    assert fields["keys_only_a"] == "0" and fields["keys_only_b"] == "0", "行数没变，键集合不该有差"
    assert fields["col_diff_local_scr"] == "1", "必须把差异归因到 local_scr 列"
    assert "local_scr" in fields["diff_example_1_columns"], "差异样例必须点名改动的列"
    assert "xcorr_cle" not in fields.get("diff_example_1_columns", ""), "未改动的列不得被点名"
    assert "diff_example_1_value_a" in fields and "diff_example_1_value_b" in fields, \
        "必须同时给出 A/B 两侧的取值"


def test_missing_row_reported_as_keys_only_a(tmp_path: Path, capsys: pytest.CaptureFixture):
    """B 少一行 ⇒ ``keys_only_a: 1``（该键只存在于 A）。"""
    path_a = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    path_b = write_csv(tmp_path / "b.csv", HEADER, ROWS[:-1])

    code, out, _ = run_cli(["compare", str(path_a), str(path_b)], capsys)
    fields = parse_fields(out)
    assert code != 0, "行数不同 ⇒ 退出码必须非 0"
    assert fields["keys_only_a"] == "1", "必须报 keys_only_a: 1（实际 %s）" % fields["keys_only_a"]
    assert fields["keys_only_b"] == "0", "B 没有多出来的行"
    assert fields["common_differing"] == "0", "公共键的值没变"
    assert fields["rows_a"] == "3" and fields["rows_b"] == "2", "必须报两侧行数"


def test_extra_row_reported_as_keys_only_b(tmp_path: Path, capsys: pytest.CaptureFixture):
    """B 多一行 ⇒ ``keys_only_b: 1``（该键只存在于 B）。"""
    path_a = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    extra = ["S2", "30", "clean", "0.0", "11.0", "3.0"]
    path_b = write_csv(tmp_path / "b.csv", HEADER, ROWS + [extra])

    code, out, _ = run_cli(["compare", str(path_a), str(path_b)], capsys)
    fields = parse_fields(out)
    assert code != 0, "行数不同 ⇒ 退出码必须非 0"
    assert fields["keys_only_b"] == "1", "必须报 keys_only_b: 1（实际 %s）" % fields["keys_only_b"]
    assert fields["keys_only_a"] == "0", "A 没有多出来的行"


def test_different_header_names_the_differing_columns(tmp_path: Path,
                                                       capsys: pytest.CaptureFixture):
    """表头不同 ⇒ 非 0 退出，且消息里要点名不同的列。"""
    header_b = list(HEADER[:-1]) + ["xcorr_maxcle"]
    path_a = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    path_b = write_csv(tmp_path / "b.csv", header_b, [row[:-1] + ["1.0"] for row in ROWS])

    code, out, err = run_cli(["compare", str(path_a), str(path_b)], capsys)
    fields = parse_fields(out)
    assert code != 0, "表头不同 ⇒ 退出码必须非 0"
    assert fields["verdict"] == "HEADER_MISMATCH", "结论必须是 HEADER_MISMATCH"
    assert fields["header_only_a"] == "xcorr_cle", "必须点名 A 独有的列"
    assert fields["header_only_b"] == "xcorr_maxcle", "必须点名 B 独有的列"
    assert "xcorr_cle" in err and "xcorr_maxcle" in err, "错误消息必须点名不同的列：%s" % err


def test_duplicate_key_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture):
    """关键列重复 ⇒ 非 0 退出，且消息里要点名重复的那个键。"""
    rows = ROWS + [["S1", "10", "clean", "0.0", "15.263246789670164", "6.5809822592002085"]]
    path_a = write_csv(tmp_path / "a.csv", HEADER, rows)
    path_b = write_csv(tmp_path / "b.csv", HEADER, ROWS)

    code, _, err = run_cli(["compare", str(path_a), str(path_b)], capsys)
    assert code != 0, "关键列重复 ⇒ 退出码必须非 0"
    assert "S1" in err and "重复" in err, "错误消息必须点名重复的键：%s" % err

    code_digest, _, err_digest = run_cli(["digest", str(path_a)], capsys)
    assert code_digest != 0, "digest 同样必须拒绝重复键"
    assert "S1" in err_digest, "digest 的错误消息也要点名重复的键"


def test_missing_key_column_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture):
    """缺关键列 ⇒ 非 0 退出，且消息里要点名缺的那一列。"""
    header_b = [col for col in HEADER if col != "condition_level"]
    rows_b = [[value for name, value in zip(HEADER, row) if name != "condition_level"]
              for row in ROWS]
    path_a = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    path_b = write_csv(tmp_path / "b.csv", header_b, rows_b)

    code, _, err = run_cli(["compare", str(path_a), str(path_b)], capsys)
    assert code != 0, "缺关键列 ⇒ 退出码必须非 0"
    assert "condition_level" in err, "错误消息必须点名缺失的列：%s" % err

    code_digest, _, err_digest = run_cli(["digest", str(path_b)], capsys)
    assert code_digest != 0, "digest 同样必须拒绝缺关键列"
    assert "condition_level" in err_digest, "digest 的错误消息也要点名缺失的列"


def test_digest_prints_rows_and_digest_lines(tmp_path: Path, capsys: pytest.CaptureFixture):
    """``digest`` 必须打印机器可解析的 ``rows:`` / ``digest:`` / ``columns:`` 行。"""
    path = write_csv(tmp_path / "a.csv", HEADER, ROWS)

    code, out, _ = run_cli(["digest", str(path)], capsys)
    fields = parse_fields(out)
    assert code == 0, "digest 必须成功退出"
    assert fields["rows"] == str(len(ROWS)), "必须打印 rows: %d" % len(ROWS)
    assert fields["columns"] == str(len(HEADER)), "必须打印 columns: %d" % len(HEADER)
    assert fields["digest"].startswith("md5:"), "必须打印 md5:<hex> 形式的摘要"
    assert fields["key_cols"] == "seq_id,frame_id,condition_family,condition_level", \
        "必须打印所用的关键列"


# --------------------------------------------------------------------------- verify 子命令
def write_sidecar(path: Path, entries: list[tuple[str, Path]], header: list[str] | None = None) -> Path:
    """写一份旁挂规范摘要清单（与 md5sum 清单同形：``<摘要>  <路径>``）。

    Args:
        path (Path): 目标路径。
        entries (list[tuple[str, Path]]): ``(摘要, 明细 CSV 路径)`` 列表。
        header (list[str] | None): 可选的注释表头。

    Returns:
        Path: 目标路径。
    """
    lines = list(header or [])
    for want, target in entries:
        lines.append("%s  %s" % (want, target))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def digest_of(path: Path, capsys: pytest.CaptureFixture) -> str:
    """跑一次 ``digest`` 拿规范摘要。

    Args:
        path (Path): 明细 CSV 路径。
        capsys (pytest.CaptureFixture): pytest 的输出捕获夹具。

    Returns:
        str: ``md5:<hex>`` 形式的规范摘要。
    """
    code, out, _ = run_cli(["digest", str(path)], capsys)
    assert code == 0, "digest 必须成功退出"
    return parse_fields(out)["digest"]


def test_verify_accepts_a_shuffled_csv(tmp_path: Path, capsys: pytest.CaptureFixture):
    """🔴 **行序不同 ⇒ verify 仍判 OK**（门禁真正成立的那条性质）。

    明细行由 ``imap_unordered`` 收集 ⇒ 重跑一次行序就变；门禁若因行序抖动报红，
    这套顺序无关摘要就白做了。
    """
    path_a = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    path_b = write_csv(tmp_path / "b.csv", HEADER, list(reversed(ROWS)))
    digest = digest_of(path_a, capsys)
    sidecar = write_sidecar(tmp_path / "sidecar.txt", [(digest, path_b)],
                            header=["# 旁挂规范摘要清单", ""])

    code, out, _ = run_cli(["verify", str(sidecar)], capsys)
    fields = parse_fields(out)
    assert code == 0, "行序不同但内容一致 ⇒ 退出码必须是 0（实际 %d）" % code
    assert fields["verdict"] == "OK", "结论必须是 OK（实际 %s）" % fields["verdict"]
    assert fields["entries"] == "1", "必须报 entries: 1（实际 %s）" % fields["entries"]
    assert fields["ok_count"] == "1", "必须报 ok_count: 1（实际 %s）" % fields["ok_count"]
    assert fields["failed_count"] == "0", "必须报 failed_count: 0"
    assert fields["entry_1_verdict"] == "OK", "单条结论必须是 OK（实际 %s）" % fields["entry_1_verdict"]
    assert fields["entry_1_want"] == digest, "必须报出期望摘要"
    assert fields["entry_1_got"] == digest, "必须报出实际摘要"


def test_verify_reports_a_tampered_csv(tmp_path: Path, capsys: pytest.CaptureFixture):
    """🔴 **内容真被改了 ⇒ verify 必须判 FAILED**（顺序无关 ≠ 内容无关）。"""
    path = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    digest = digest_of(path, capsys)
    tampered = [list(row) for row in ROWS]
    tampered[1][4] = "9.75"          # 只改 S1/blur 那一行的 local_scr
    path_t = write_csv(tmp_path / "tampered.csv", HEADER, tampered)
    sidecar = write_sidecar(tmp_path / "sidecar.txt", [(digest, path_t)])

    code, out, _ = run_cli(["verify", str(sidecar)], capsys)
    fields = parse_fields(out)
    assert code != 0, "内容被改 ⇒ 退出码必须非 0"
    assert fields["verdict"] == "FAILED", "结论必须是 FAILED（实际 %s）" % fields["verdict"]
    assert fields["entry_1_verdict"] == "MISMATCH", \
        "单条结论必须是 MISMATCH（实际 %s）" % fields["entry_1_verdict"]
    assert fields["ok_count"] == "0", "必须报 ok_count: 0"
    assert fields["failed_count"] == "1", "必须报 failed_count: 1"
    assert fields["entry_1_want"] == digest, "必须报出期望摘要"
    assert fields["entry_1_got"] != digest, "必须报出实际摘要（≠ 期望）"


def test_verify_reports_a_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture):
    """清单里列的文件不存在 ⇒ MISSING + 非 0 退出（缺件不能被当成通过）。"""
    path = write_csv(tmp_path / "a.csv", HEADER, ROWS)
    digest = digest_of(path, capsys)
    sidecar = write_sidecar(tmp_path / "sidecar.txt", [(digest, tmp_path / "missing.csv")])

    code, out, _ = run_cli(["verify", str(sidecar)], capsys)
    fields = parse_fields(out)
    assert code != 0, "缺件 ⇒ 退出码必须非 0"
    assert fields["verdict"] == "FAILED", "结论必须是 FAILED（实际 %s）" % fields["verdict"]
    assert fields["entry_1_verdict"] == "MISSING", \
        "单条结论必须是 MISSING（实际 %s）" % fields["entry_1_verdict"]
    assert fields["failed_count"] == "1", "必须报 failed_count: 1"
    assert fields["ok_count"] == "0", "缺件不得计入通过数"


# --------------------------------------------------------------------------- 清单工具接线
def _make_fake_repo(root: Path) -> Path:
    """造一个临时仓库（``iqa_v2/*.py`` + ``iqa_v2/output/*.csv``）。

    Args:
        root (Path): 临时目录。

    Returns:
        Path: 仓库根。
    """
    repo = root / "repo"
    (repo / "iqa_v2" / "output").mkdir(parents=True)
    (repo / "iqa_v2" / "dummy.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "iqa_v2" / "output" / "base.csv").write_text("a\n", encoding="utf-8")
    return repo


def _run_make_manifest(repo: Path, bulk: Path, out: Path,
                       extra: list[str] | None = None) -> subprocess.CompletedProcess:
    """跑一次 ``make_manifest_v2.py``（子进程 ⇒ 真实 CLI 行为）。

    Args:
        repo (Path): 仓库根。
        bulk (Path): 大体量明细 CSV。
        out (Path): 清单输出路径。
        extra (list[str] | None): 追加的命令行参数。

    Returns:
        subprocess.CompletedProcess: 子进程结果。
    """
    args = [sys.executable, str(HERE / "make_manifest_v2.py"),
            "--repo-root", str(repo), "--bulk-detail", str(bulk), "--out", str(out)]
    return subprocess.run(args + list(extra or []), capture_output=True, text=True, check=False)


def _checkable_lines(manifest: Path) -> list[str]:
    """清单里**可校验**的行（跳过空行与 ``#`` 注释行）。

    Args:
        manifest (Path): 清单路径。

    Returns:
        list[str]: 可校验行。
    """
    return [line for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def test_make_manifest_registers_bulk_only_in_the_canonical_sidecar(tmp_path: Path,
                                                                    capsys: pytest.CaptureFixture):
    """🔴 大体量明细**不得**再作为裸 md5 行进清单 ⇒ 它只进旁挂规范摘要清单。

    这就是 review 发现 IMPORTANT-6 的修复点：明细行序由 ``imap_unordered`` 决定 ⇒
    裸 md5 每次重跑都变 ⇒ ``md5sum -c`` 必然假失败。
    """
    repo = _make_fake_repo(tmp_path)
    bulk = write_csv(tmp_path / "bulk.csv", HEADER, ROWS)
    out = tmp_path / "manifest.md5"

    result = _run_make_manifest(repo, bulk, out)
    assert result.returncode == 0, "清单生成必须成功：%s" % result.stderr

    checkable = _checkable_lines(out)
    assert not any(str(bulk) in line for line in checkable), \
        "大体量明细不得作为可校验的 md5 行出现在清单里：%s" % checkable

    sidecar = tmp_path / "manifest.bulk.canonical.txt"
    assert sidecar.is_file(), "必须生成旁挂规范摘要清单：%s" % sidecar
    sidecar_lines = _checkable_lines(sidecar)
    assert sidecar_lines == ["%s  %s" % (digest_of(bulk, capsys), bulk.resolve())], \
        "旁挂清单必须登记一条「<规范摘要>  <绝对路径>」：%s" % sidecar_lines

    code, stdout, _ = run_cli(["verify", str(sidecar)], capsys)
    fields = parse_fields(stdout)
    assert code == 0, "旁挂清单必须自洽（实际 %d）" % code
    assert fields["verdict"] == "OK", "旁挂清单校验结论必须是 OK（实际 %s）" % fields["verdict"]
    assert fields["entries"] == "1", "旁挂清单必须只有 1 条（实际 %s）" % fields["entries"]


def test_make_manifest_raw_mode_keeps_the_legacy_bare_md5_line(tmp_path: Path):
    """``--bulk-mode raw`` 必须复现**历史形状**（明细 = 清单里的裸 md5 行、无旁挂清单）。

    冻结基线清单 ``iqa_v2/output/manifest_v2.md5`` 就是 raw 模式生成的 ⇒ 该模式必须留着。
    """
    repo = _make_fake_repo(tmp_path)
    bulk = write_csv(tmp_path / "bulk.csv", HEADER, ROWS)
    out = tmp_path / "manifest.md5"

    result = _run_make_manifest(repo, bulk, out, extra=["--bulk-mode", "raw"])
    assert result.returncode == 0, "清单生成必须成功：%s" % result.stderr

    want = hashlib.md5(bulk.read_bytes()).hexdigest()
    checkable = _checkable_lines(out)
    assert "%s  %s" % (want, bulk) in checkable, \
        "raw 模式必须在清单里登记裸 md5 行：%s" % checkable
    assert not (tmp_path / "manifest.bulk.canonical.txt").exists(), "raw 模式不得生成旁挂清单"


def test_collect_excludes_run_meta_json_but_keeps_other_json_products(tmp_path: Path):
    """🔴 ``run_meta.json`` 是**运行留痕**（含 UTC 时间戳）⇒ **必须不进清单**。

    一进清单，每次重生成清单都因时间戳不同而变 ⇒ 清单失去字节可复现性。
    排除口径必须**窄**：普通 ``*.json`` 产物照收（否则会误伤真产物）。
    """
    import make_manifest_v2

    repo = _make_fake_repo(tmp_path)
    products = repo / "iqa_v2" / "output"
    (products / "run_meta.json").write_text('{"finished_at": "2026-11-27T00:00:00Z"}\n',
                                            encoding="utf-8")
    (products / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")

    assert "run_meta.json" in make_manifest_v2.EXCLUDED_PRODUCT_FILES, \
        "排除名单必须显式登记 run_meta.json"

    names = [name for name, _ in make_manifest_v2.collect(repo, tmp_path / "missing.csv")]
    assert "iqa_v2/output/run_meta.json" not in names, \
        "run_meta.json（含时间戳）不得进清单：%s" % names
    assert "iqa_v2/output/dataset_manifest.json" in names, \
        "普通 *.json 产物必须照收（排除口径必须窄）：%s" % names
    assert "iqa_v2/output/base.csv" in names, "普通 *.csv 产物必须照收：%s" % names
