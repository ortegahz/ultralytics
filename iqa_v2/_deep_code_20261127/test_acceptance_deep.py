#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deep 轮验收判定器 ``iqa_v2/acceptance_deep.py`` 的**回归测试**。

为什么要有这个文件
------------------
一次独立审计在判定器里找出 12 个缺陷——绝大多数是「**判据根本无法失败**」的
假通过（例如：A1 只要**任意**臂跌破 50% 就通过，而 csrt 只是同族对照臂；
C1 只查判据编号出现过没有，不查有没有给结论；D1/D2/D4/D6 遇到解析不出的
单元格就当无事发生 ⇒ ``abs(x - nan) > tol`` 恒为假）。

本文件把每个缺陷都固化成**变异探针**：把 ``iqa_v2/output_deep/*.csv`` 与
``iqa_v2/REPORT_V2_DEEP.md`` 拷进 ``tmp_path`` 后就地改坏，再跑一遍判定器，
断言该判据**必须变红**（修复前这些探针全是假通过）。

判定器用 ``subprocess`` 按**真实门禁的口径**调用；所有变异都落在 ``tmp_path`` 里，
仓库里的真实产物一个字都不动。

    python3 -m pytest iqa_v2/test_acceptance_deep.py -q
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEEP_DIR = HERE / "output_deep"
BASELINE_DIR = HERE / "output"
REPORT = HERE / "REPORT_V2_DEEP.md"
ACCEPTANCE = HERE / "REPORT_V2_DEEP_ACCEPTANCE.md"
#: 判定器读的门禁结果 JSON —— 用**两侧通用**的最终汇总文件。
#: 🔴 原来指向 ``_gates_deep_host_20261127.json``（**只有 host 有**）⇒ 服务器侧
#:    判定器拿不到门禁结果 ⇒ C4 判不通过 ⇒ ``test_real_artifacts_keep_the_22_verdicts``
#:    在服务器侧**假失败**（同一份代码、同一份产物，两侧结论却不一致）。
GATES = HERE / "_gates_deep_final_20261127.json"
if not GATES.is_file():
    GATES = HERE / "_gates_deep_host_20261127.json"
CHECKER = HERE / "acceptance_deep.py"

#: 判定器输出里的判据行（`A1       PASS     ……`）
VERDICT_RE = re.compile(r"^(A\d|B\d|C\d|D\d)\s+(\S+)\s")
#: 判据表的一行（`| **A1** | …… |`）
CRITERION_ROW_RE = re.compile(r"^\|\s*\*\*[A-D]\d\*\*\s*\|")

#: 真实产物下的判定集（验收记录 §一）——**任何修复都不得改动它**
EXPECTED_REAL = {
    "A1": "FAIL", "A2": "PARTIAL", "A3": "PASS", "A4": "FAIL",
    "B1": "PASS", "B2": "PASS", "B3": "PASS", "B4": "PASS", "B5": "PASS",
    "B6": "PASS",
    "C1": "PASS", "C2": "PASS", "C3": "PASS", "C4": "PASS",
    "D1": "PASS", "D2": "PASS", "D3": "PASS", "D4": "PASS",
    "D5": "PASS", "D6": "PASS", "D7": "PASS", "D8": "PASS",
}


# --------------------------------------------------------------------------
# 工具：跑判定器 / 造变异副本
# --------------------------------------------------------------------------
def run_checker(deep_dir, report, cwd=None):
    """按真实门禁的口径跑一遍判定器，返回 (判据 → (状态, 整行), stdout)。"""
    cmd = [sys.executable, str(CHECKER),
           "--deep-dir", str(deep_dir),
           "--baseline-dir", str(BASELINE_DIR),
           "--code-dir", str(HERE),
           "--report", str(report),
           "--acceptance", str(ACCEPTANCE),
           "--gates-json", str(GATES)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(cwd) if cwd is not None else None)
    assert proc.returncode == 0, "判定器异常退出 rc=%s：\n%s" % (proc.returncode, proc.stderr)
    verdicts = {}
    for line in proc.stdout.splitlines():
        match = VERDICT_RE.match(line)
        if match:
            verdicts[match.group(1)] = (match.group(2), line)
    return verdicts, proc.stdout


def copy_deep(tmp_path, name="deep"):
    """把 deep 产物整目录拷进 tmp_path（判定器只读它）。"""
    dst = tmp_path / name
    shutil.copytree(DEEP_DIR, dst)
    return dst


def patched_report(tmp_path, replacements, name="report.md"):
    """按 (原文, 改后) 改写报告副本；原文必须存在，否则测试直接失败。"""
    text = REPORT.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in text, "报告里找不到待改写的内容：%r" % old[:70]
        text = text.replace(old, new)
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def appended_report(tmp_path, extra, name="report.md", base=None):
    """在报告末尾追加一段（模拟「补写了说明」）。"""
    text = (REPORT.read_text(encoding="utf-8") if base is None else base) + extra
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def edit_csv(path, mutate):
    """就地改写 CSV（mutate 收到行列表，返回新行列表）；列序保持不变。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    rows = mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def status(verdicts, ident):
    """取某条判据的状态。"""
    assert ident in verdicts, "判定器没有输出 %s" % ident
    return verdicts[ident][0]


def evidence(verdicts, ident):
    """取某条判据的证据行。"""
    assert ident in verdicts, "判定器没有输出 %s" % ident
    return verdicts[ident][1]


# --------------------------------------------------------------------------
# 0) 回归总闸：真实产物的判定集不得改变
# --------------------------------------------------------------------------
def test_real_artifacts_keep_the_22_verdicts(tmp_path):
    """真实产物 ⇒ PASS 19 / PARTIAL 1 / FAIL 2（A1、A4 两条 FAIL）。"""
    verdicts, out = run_checker(DEEP_DIR, REPORT, cwd=tmp_path)
    assert len(verdicts) == 22, "判定器应输出 22 条判据，实际 %d 条" % len(verdicts)
    assert {ident: value[0] for ident, value in verdicts.items()} == EXPECTED_REAL
    assert "合计：PASS 19  PARTIAL 1  FAIL 2（共 22 条）" in out


# --------------------------------------------------------------------------
# 1) C1：判据编号必须落在**带结论**的表格行里
# --------------------------------------------------------------------------
def test_c1_needs_a_conclusion_not_just_the_id(tmp_path):
    """判据行整行删掉、只剩一句「判据编号清单」⇒ C1 必须不通过。

    编号**仍然出现在正文里**（旧判据只看「出现过没有」⇒ 会假通过），
    但没有任何一行同时给出「编号 + 结论」⇒ 新判据必须判不通过。
    """
    text = REPORT.read_text(encoding="utf-8")
    stripped = "\n".join(line for line in text.splitlines()
                         if not CRITERION_ROW_RE.match(line.strip()))
    stripped += "\n\n## 八、判据总表（判据编号清单：A1 A2 A3 A4 B1 B2 B3 B4 B5）\n"
    report = tmp_path / "report.md"
    report.write_text(stripped, encoding="utf-8")
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "C1") == "FAIL"
    assert "A1" in evidence(verdicts, "C1")


# --------------------------------------------------------------------------
# 2) D1 / D2 / D4 / D6：单元格解析不出来 ⇒ 必须记问题（不能静默通过）
# --------------------------------------------------------------------------
def test_d1_rejects_unparsable_baseline_cell(tmp_path):
    """§4.1 基线轮那一列写 N/A ⇒ D1 必须不通过。"""
    report = patched_report(tmp_path, [(
        "| `blur` | 98.7% → 97.5% → 95.4% → 92.0% → 91.6% → 91.0% |",
        "| `blur` | 98.7% → 97.5% → 95.4% → 92.0% → 91.6% → N/A |")])
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "D1") == "FAIL"
    assert "无法解析为数值" in evidence(verdicts, "D1")


def test_d2_rejects_unparsable_span_cell(tmp_path):
    """§5.1 分箱跨度写 N/A ⇒ D2 必须不通过。"""
    report = patched_report(tmp_path, [(
        "| `kcf_legacy` | `blur` | 5.03 pp | **10.97 pp** | 3.66 pp | **8.47 pp** | A |",
        "| `kcf_legacy` | `blur` | 5.03 pp | **N/A** | 3.66 pp | **8.47 pp** | A |")])
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "D2") == "FAIL"
    assert "无法解析为数值" in evidence(verdicts, "D2")


def test_d4_rejects_unparsable_blind_cell(tmp_path):
    """§7.3 坏片拦截率写 N/A ⇒ D4 必须不通过。"""
    report = patched_report(tmp_path, [(
        "| `local_scr` | **72.84%** | **100.00%** | 98.00% | **2.00 pp** |",
        "| `local_scr` | **72.84%** | **N/A** | 98.00% | **2.00 pp** |")])
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "D4") == "FAIL"
    assert "无法解析为数值" in evidence(verdicts, "D4")


def test_d6_rejects_unparsable_heaviest_cell(tmp_path):
    """§4.1 最重档跟住率写 N/A ⇒ D6 必须不通过（且核对条数不得悄悄变少）。"""
    report = patched_report(tmp_path, [(
        "| `blur` | 91.0% | **87.9%** | 96.0% | **91.2%** | 93.1% |",
        "| `blur` | 91.0% | **N/A** | 96.0% | **91.2%** | 93.1% |")])
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "D6") == "FAIL"
    assert "无法解析为数值" in evidence(verdicts, "D6")


# --------------------------------------------------------------------------
# 3) D7：现场真值缺失时不能算「核过了」
# --------------------------------------------------------------------------
def test_d7_missing_ground_truth_is_not_counted_as_checked(tmp_path):
    """target_lap_var 本轮无线位 ⇒ A3 真值算不出来 ⇒ D7 必须不通过。"""
    deep = copy_deep(tmp_path)

    def mutate(rows):
        for row in rows:
            if row["metric"] == "target_lap_var":
                row["line_value"] = "nan"
                row["deciding_tracker"] = ""
                row["deciding_family"] = ""
        return rows

    edit_csv(deep / "constraint_lines_v2.csv", mutate)
    verdicts, _ = run_checker(deep, REPORT, cwd=tmp_path)
    assert status(verdicts, "A3") == "FAIL"
    assert status(verdicts, "D7") == "FAIL"
    assert "全部一致" not in evidence(verdicts, "D7")


# --------------------------------------------------------------------------
# 4) D7：A2 的核对必须是**双向**的（没有族达 A 时不能写「仅 blur 达 A 级」）
# --------------------------------------------------------------------------
def test_d7_a2_cell_must_match_the_recomputed_a_families(tmp_path):
    """把 KCF 臂的膝点全部压平（A 级归零）⇒ A2 单元格的说法就成了假话 ⇒ D7 必红。"""
    deep = copy_deep(tmp_path)

    def mutate(rows):
        for row in rows:
            if row["tracker"] == "kcf_legacy":
                row["curve_y_span"] = "0.01"
                row["level_y_span"] = "0.01"
                row["curve_monotone"] = "0"
        return rows

    edit_csv(deep / "knee_thresholds_v2.csv", mutate)
    verdicts, _ = run_checker(deep, REPORT, cwd=tmp_path)
    assert status(verdicts, "D7") == "FAIL"
    assert "达 A 级" in evidence(verdicts, "D7")


# --------------------------------------------------------------------------
# 5) A1：只看**主臂**（csrt 是同族对照臂，不参与判据取值）
# --------------------------------------------------------------------------
def test_a1_ignores_the_csrt_reference_arm(tmp_path):
    """把最重档的 csrt 跟住率打成 0 ⇒ A1 仍必须不通过（主臂没塌）。"""
    deep = copy_deep(tmp_path)
    heaviest = {"blur": "8.0", "motion": "15.0", "contrast": "0.05"}

    def mutate(rows):
        for row in rows:
            level = heaviest.get(row["condition_family"])
            if level is not None and row["condition_level"] == level:
                row["csrt_success"] = "0"
        return rows

    edit_csv(deep / "degrade_detail_v2.csv", mutate)
    verdicts, _ = run_checker(deep, REPORT, cwd=tmp_path)
    assert status(verdicts, "A1") == "FAIL"
    assert "/csrt" not in evidence(verdicts, "A1")


# --------------------------------------------------------------------------
# 6) C2：解析不出数值 ⇒ 记问题，**绝不**崩掉整轮判定
# --------------------------------------------------------------------------
def test_c2_unparsable_threshold_does_not_crash_the_run(tmp_path):
    """对外版把 `<` 写成 `≤` ⇒ 判定器不得崩，仍要输出 22 条判据。"""
    report = patched_report(tmp_path, [("`target_lap_var < 220.6151`",
                                        "`target_lap_var ≤ 220.6151`")])
    verdicts, out = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert len(verdicts) == 22
    assert status(verdicts, "C2") == "FAIL"
    assert "无可解析数值" in evidence(verdicts, "C2")
    assert "合计" in out


def test_a_raising_criterion_fails_alone_and_keeps_the_other_21(tmp_path):
    """一条判据内部抛异常 ⇒ 只有那一条判不通过，其余 21 条照常给出。"""
    report = patched_report(tmp_path, [(
        ("| `target_lap_var` | **220.6151** | blur | `xcorr` | 2 | 220.6151 | 359.8505 | "
         "141.7701 | 38.69% |"),
        "| `target_lap_var` | **220.6151** |")])
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert len(verdicts) == 22
    assert status(verdicts, "D3") == "FAIL"
    assert "异常" in evidence(verdicts, "D3")
    assert status(verdicts, "D1") == "PASS"


# --------------------------------------------------------------------------
# 7) founded_metric：缺盲测回归行 ⇒ 判**不成立**（不得因缺数据而放行）
# --------------------------------------------------------------------------
def test_local_scr_is_not_founded_without_blind_regression_rows(tmp_path):
    """删掉 local_scr 的盲测回归行 ⇒ local_scr 不得被判「成立」⇒ C2 不得要求对外收录。"""
    deep = copy_deep(tmp_path)

    def drop_rows(rows):
        return [row for row in rows
                if not (row["metric"] == "local_scr"
                        and row["set"] in ("real_bad", "real_good"))]

    edit_csv(deep / "regression_v2.csv", drop_rows)

    def make_founded(rows):
        """把 local_scr 的决定曲线改成一条**过守卫**的曲线（否则「成立」判不到回归门禁）。"""
        for row in rows:
            if row["metric"] == "local_scr":
                row["line_value"] = "8.029360923693313"
                row["deciding_tracker"] = "xcorr"
                row["deciding_family"] = "blur"
                row["curve_set"] = "common_good"
                row["stratum"] = "all"
        return rows

    edit_csv(deep / "constraint_lines_v2.csv", make_founded)

    def reliable(rows):
        for row in rows:
            if (row["tracker"] == "xcorr" and row["curve_set"] == "common_good"
                    and row["stratum"] == "all" and row["degradation_family"] == "blur"
                    and row["metric"] == "local_scr"):
                row["knee_reliable"] = "1"
                row["delta_pct_of_span"] = "0.02"
        return rows

    edit_csv(deep / "knee_thresholds_v2.csv", reliable)
    verdicts, _ = run_checker(deep, REPORT, cwd=tmp_path)
    assert status(verdicts, "C2") == "PASS"
    assert "未收录" not in evidence(verdicts, "C2")


# --------------------------------------------------------------------------
# 8) B2 / B4：复算必须用**约束线行自己声明**的曲线池
# --------------------------------------------------------------------------
def test_b2_b4_use_the_curve_pool_declared_by_the_constraint_line(tmp_path):
    """约束线表声明 own_good/small ⇒ A3 / B2 / B4 都必须在**那个池**上复核。"""
    deep = copy_deep(tmp_path)

    def mutate(rows):
        for row in rows:
            if row["metric"] in ("target_local_contrast", "target_lap_var"):
                row["curve_set"] = "own_good"
                row["stratum"] = "small"
        return rows

    edit_csv(deep / "constraint_lines_v2.csv", mutate)
    verdicts, _ = run_checker(deep, REPORT, cwd=tmp_path)
    assert status(verdicts, "A3") == "FAIL"
    assert status(verdicts, "B2") == "FAIL"
    assert status(verdicts, "B4") == "FAIL"


# --------------------------------------------------------------------------
# 9) B5 / C2：对外版必须取**两轮更严者**，deep 更松时要写明「口径变化导致线位移动」
# --------------------------------------------------------------------------
def test_b5_accepts_the_stricter_line_but_demands_the_wording(tmp_path):
    """deep 线位比基线更松 ⇒ 对外版取基线更严者可以，但必须写明线位移动。"""
    deep = copy_deep(tmp_path)

    def mutate(rows):
        for row in rows:
            if row["metric"] == "target_local_contrast":
                row["line_value"] = "0.25"       # 比基线轮 0.18916642 更松
        return rows

    edit_csv(deep / "constraint_lines_v2.csv", mutate)
    report = patched_report(tmp_path, [("`target_local_contrast < 0.1005622`",
                                        "`target_local_contrast < 0.1891664`")])
    verdicts, _ = run_checker(deep, report, cwd=tmp_path)
    assert status(verdicts, "C2") == "PASS"
    assert status(verdicts, "B5") == "FAIL"
    assert "未取更严者" not in evidence(verdicts, "B5")
    assert "口径变化导致线位移动" in evidence(verdicts, "B5")


def test_b5_passes_once_the_wording_is_stated(tmp_path):
    """补上「口径变化导致线位移动」⇒ B5 的补救口径就可达 ⇒ B5 必须通过。"""
    deep = copy_deep(tmp_path)

    def mutate(rows):
        for row in rows:
            if row["metric"] == "target_local_contrast":
                row["line_value"] = "0.25"
        return rows

    edit_csv(deep / "constraint_lines_v2.csv", mutate)
    text = REPORT.read_text(encoding="utf-8").replace(
        "`target_local_contrast < 0.1005622`", "`target_local_contrast < 0.1891664`")
    report = appended_report(
        tmp_path,
        "\n口径变化导致线位移动：本轮对外版取基线轮更严的线位 0.1891664。\n",
        name="report2.md", base=text)
    verdicts, _ = run_checker(deep, report, cwd=tmp_path)
    assert status(verdicts, "B5") == "PASS"


# --------------------------------------------------------------------------
# 10) D8：引用外部规格书的 §X.Y 不是「悬空交叉引用」
# --------------------------------------------------------------------------
def test_d8_exempts_references_to_the_external_spec(tmp_path):
    """报告里写「档位表见 DEEP_AXIS_SPEC_20261127.md §1.1」⇒ D8 必须通过。"""
    report = appended_report(tmp_path, "\n档位表见 DEEP_AXIS_SPEC_20261127.md §1.1（a-priori）。\n")
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "D8") == "PASS"
    assert "悬空交叉引用" not in evidence(verdicts, "D8")


# --------------------------------------------------------------------------
# 11) D1 / D2：报告的每一行都要对上 CSV（§5.2 逐指标分级 + §4.1 基线轮列）
# --------------------------------------------------------------------------
def test_d2_checks_every_metrics_own_curve_grading(tmp_path):
    """§5.2 里 target_lap_var/kcf_legacy/blur 的分箱跨度写错 ⇒ D2 必须不通过。"""
    report = patched_report(tmp_path, [(
        "| `target_lap_var` | `kcf_legacy` | `blur` | 14.54 pp | 8.47 pp | 3.37% | ✅ **过** |",
        "| `target_lap_var` | `kcf_legacy` | `blur` | 14.55 pp | 8.47 pp | 3.37% | ✅ **过** |")])
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "D2") == "FAIL"


def test_d1_checks_the_baseline_round_column(tmp_path):
    """§4.1 基线轮那一列写错 ⇒ D1 必须不通过。"""
    report = patched_report(tmp_path, [(
        "| `blur` | 98.7% → 97.5% → 95.4% → 92.0% → 91.6% → 91.0% |",
        "| `blur` | 98.7% → 97.5% → 95.4% → 92.0% → 91.6% → 71.0% |")])
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "D1") == "FAIL"


# --------------------------------------------------------------------------
# 12) D5：遗留措辞/数字必须按**本轮实测**判定（不再误伤正确表述）
# --------------------------------------------------------------------------
def test_d5_accepts_the_no_red_line_wording_when_a3_really_failed(tmp_path):
    """A3 确实没拿到红线时，「本轮没有有效红线」是**正确**表述 ⇒ D5 必须通过。"""
    deep = copy_deep(tmp_path)

    def mutate(rows):
        for row in rows:
            if row["metric"] == "target_lap_var":
                row["line_value"] = "nan"
                row["deciding_tracker"] = ""
                row["deciding_family"] = ""
        return rows

    edit_csv(deep / "constraint_lines_v2.csv", mutate)
    report = appended_report(tmp_path, "\n本轮 `target_lap_var` 本轮没有有效红线。\n")
    verdicts, _ = run_checker(deep, report, cwd=tmp_path)
    assert status(verdicts, "A3") == "FAIL"
    assert status(verdicts, "D5") == "PASS"


def test_d5_accepts_the_cross_arm_clarification(tmp_path):
    """「双算法偏差 ≤ 15%，与跨臂偏差无关」是**正确**澄清 ⇒ D5 必须通过。"""
    report = appended_report(tmp_path, "\n双算法（Kneedle / PCHIP）偏差 ≤ 15%，与跨臂偏差无关。\n")
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "D5") == "PASS"


def test_d5_catches_a_single_baseline_round_number(tmp_path):
    """单个基线轮参照读数（505.624）也是遗留 ⇒ D5 必须不通过。"""
    report = appended_report(tmp_path, "\n基线轮参照读数 505.624。\n")
    verdicts, _ = run_checker(DEEP_DIR, report, cwd=tmp_path)
    assert status(verdicts, "D5") == "FAIL"
