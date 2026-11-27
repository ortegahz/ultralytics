#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``recheck_curves.py`` 的验收测试（``pytest`` 全绿）。

被测对象：``iqa_v2/recheck_curves.py`` —— 退化曲线**有效性复算器**（host 侧独立复算）。

为什么要有这组测试
------------------
复算器原来把退化档位阶梯**硬编码**成基线轮的 6/5/6，并按**字符串**去匹配明细里的
``condition_level``。深化轴（deep profile：blur 10 档 / motion 7 档 / contrast 8 档）一旦落地，
硬编码阶梯会把新增的 8 个档位**静默丢掉** —— 不报错、输出看着还挺正常，于是：

* 曲线跨度只按「浅半截轴」算（deep 轴真正的跨度根本没进账）；
* 只在 σ6/σ8、L13/L15、α0.10/α0.05 上出现的**非单调**永远检测不到；
* A/B/C 有效性等级与「有资格进红线」的候选集全是错的。

本组测试把这条口径钉死：

1. 档位阶梯**从明细底表反推**（deep 档位一个不落），不依赖任何硬编码常量；
2. 档位值一律经 ``float`` 归一（明细写的是 ``str(float)``：``0.8`` 不是 ``0.80``、``0.1`` 不是 ``0.10``）；
3. 严重度方向与 ``iqa_v2/verify_recalc_v2.py`` 的 ``degradation_strength`` **同口径**
   （contrast ⇒ ``1 - alpha``，方向与 blur/motion 相反）；
4. 只在 deep 档位上出现的非单调**必须**被抓到（旧版只看基线档 ⇒ 漏判成单调）；
5. 明细里有、工具覆盖不到的档位/退化族**必须**被大声报出，不许静默丢；
6. **回归护栏**：喂基线轮的明细 CSV，跨度 / 单调性 / A-B-C 等级与旧版**逐位相同**；
7. 膝点表路径**可配**（``--knee-csv`` / ``KNEE_CSV``）—— 深轮的膝点表在 ``output_deep/``，
   不配就会把 deep 明细和基线膝点表配成一对；
8. 审计器（``iqa_v2/audit_final.py``）侧：档位阶梯**自动识别**（基线 6/5/6 ⇒ 深轮 10/7/8）、
   已知口径交叉校验、**最重档跟住率**按最重档那一档算（不是全档均值）、
   明细底表缺失 ⇒ **显式 FAIL**（不许静默通过）。

复现::

    cd <仓库根> && python3 -m pytest iqa_v2/test_recheck_curves.py -v
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_final as audit  # noqa: E402
import recheck_curves as rc  # noqa: E402

#: 仓库自带基线轮的明细底表（回归护栏用；本测试**只读**，不碰 ``iqa_v2/output/``）。
BASELINE_DETAIL: Path = HERE / "output" / "degrade_detail_v2.csv"

#: 基线轮的档位阶梯（旧版硬编码在 ``recheck_curves.LEVEL_ORDER`` 里的那一套）。
BASELINE_LADDER: dict[str, list[float]] = {
    "blur": [0.6, 1.0, 1.5, 2.2, 3.0, 4.0],
    "motion": [3.0, 5.0, 7.0, 9.0, 11.0],
    "contrast": [0.8, 0.65, 0.5, 0.35, 0.25, 0.15],
}

#: 深化轴（deep profile）的档位阶梯 —— 基线的**严格超集**。
DEEP_LADDER: dict[str, list[float]] = {
    "blur": [0.6, 1.0, 1.5, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "motion": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
    "contrast": [0.8, 0.65, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05],
}

#: 测试明细的表头（与真实产物同形的关键列即可）。
HEADER: list[str] = [
    "seq_id",
    "frame_id",
    "condition_family",
    "condition_level",
    "local_scr",
    "target_local_contrast",
    "target_lap_var",
    "xcorr_success",
    "kcf_legacy_success",
    "csrt_success",
]


def make_row(family: str, level: str, success: float = 1.0, **metrics: float) -> dict[str, str]:
    """造一行明细。

    Args:
        family (str): 退化族名（``clean`` / ``blur`` / ``motion`` / ``contrast``）。
        level (str): 档位（**原样写入**，用于复现 ``"0.10"`` 与 ``"0.1"`` 这类写法差异）。
        success (float): 三臂的跟住率。
        **metrics (float): 三个核心指标的取值。

    Returns:
        dict[str, str]: 明细行。
    """
    return {
        "seq_id": "seq",
        "frame_id": "0",
        "condition_family": family,
        "condition_level": level,
        "local_scr": repr(metrics.get("local_scr", 10.0)),
        "target_local_contrast": repr(metrics.get("target_local_contrast", 0.5)),
        "target_lap_var": repr(metrics.get("target_lap_var", 1000.0)),
        "xcorr_success": repr(success),
        "kcf_legacy_success": repr(success),
        "csrt_success": repr(success),
    }


def make_rows(ladder: dict[str, list[float]], per_level: int = 2,
              bump: dict[str, float] | None = None) -> list[dict[str, str]]:
    """造一份明细行（``condition_level`` 按 ``str(float)`` 写，与真实产物同形）。

    Args:
        ladder (dict): ``{族: [档位, ...]}``，档位按**退化严重度递增**排列。
        per_level (int): 每档造几行。
        bump (dict | None): ``{族: 档位}`` —— 该档的跟住率**反弹**，制造「只在该档出现」的非单调。

    Returns:
        list[dict[str, str]]: 明细行（含 clean 行）。
    """
    bump = bump or {}
    rows = [make_row("clean", "0.0", 1.0)]
    for family, levels in ladder.items():
        for position, level in enumerate(levels):
            rate = 0.99 - 0.08 * position
            if bump.get(family) == level:
                rate = min(0.99, rate + 0.30)
            scale = 1.0 - 0.1 * position
            if bump.get(family) == level:
                scale *= 1.6
            for _ in range(per_level):
                rows.append(make_row(family, str(float(level)), rate,
                                     local_scr=10.0 * scale,
                                     target_local_contrast=0.5 * scale,
                                     target_lap_var=1000.0 * scale))
    return rows


def write_detail(path: Path, rows: list[dict[str, str]]) -> Path:
    """把明细行写成 CSV。

    Args:
        path (Path): 目标路径。
        rows (list): 明细行。

    Returns:
        Path: 目标路径。
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in HEADER})
    return path


# ---------------------------------------------------------------- 阶梯反推
def test_ladder_is_derived_from_detail_csv() -> None:
    """档位阶梯从明细底表反推 ⇒ deep 档位一个不落（不靠硬编码常量）。"""
    ladder, notes = rc.derive_ladder(make_rows(DEEP_LADDER))
    assert notes == []
    assert ladder == DEEP_LADDER


def test_contrast_order_is_by_severity_not_by_alpha() -> None:
    """contrast 的严重度方向与另外两族相反（alpha 越小越重）。"""
    assert rc.degradation_strength("contrast", 0.05) > rc.degradation_strength("contrast", 0.8)
    assert rc.degradation_strength("blur", 8.0) > rc.degradation_strength("blur", 0.6)
    assert rc.degradation_strength("motion", 15.0) > rc.degradation_strength("motion", 3.0)


def test_severity_mapping_matches_verify_recalc_v2() -> None:
    """严重度映射与 ``verify_recalc_v2.py`` 的同名函数**同口径**（别各写各的）。"""
    sys.path.insert(0, str(HERE))
    from verify_recalc_v2 import degradation_strength as reference  # noqa: PLC0415

    for family, levels in DEEP_LADDER.items():
        for level in levels:
            assert rc.degradation_strength(family, level) == reference(family, level)


def test_level_values_are_normalized_through_float() -> None:
    """档位按 ``float`` 归一：``"0.10"`` 与 ``"0.1"`` 是同一档（不许靠字符串相等判档）。"""
    rows = [make_row("clean", "0.0"),
            make_row("contrast", "0.8"),
            make_row("contrast", "0.10"),
            make_row("contrast", "0.1")]
    ladder, notes = rc.derive_ladder(rows)
    assert not [note for note in notes if "contrast" in note]
    assert ladder["contrast"] == [0.8, 0.1]
    assert rc.curve_validity(rows)[("xcorr", "contrast")]["levels"] == [0.8, 0.1]


def test_unknown_family_is_reported_loudly() -> None:
    """工具覆盖不到的退化族必须被报出来（不许静默丢）。"""
    rows = make_rows({"blur": [0.6, 1.0]}) + [make_row("noise", "0.05")]
    ladder, notes = rc.derive_ladder(rows)
    assert ladder["blur"] == [0.6, 1.0]
    assert any("noise" in note for note in notes)


def test_unparsable_level_is_reported_loudly() -> None:
    """档位解析不出来的行必须被报出来（不许静默丢）。"""
    rows = make_rows({"blur": [0.6, 1.0]}) + [make_row("blur", "abc")]
    ladder, notes = rc.derive_ladder(rows)
    assert ladder["blur"] == [0.6, 1.0]
    assert any("abc" in note for note in notes)


def test_empty_family_falls_back_loudly() -> None:
    """某族一行退化都没有 ⇒ 回退到基线硬编码阶梯，且**大声告警**（不许静默替换）。"""
    ladder, notes = rc.derive_ladder([make_row("clean", "0.0")])
    assert ladder["blur"] == BASELINE_LADDER["blur"]
    assert ladder["motion"] == BASELINE_LADDER["motion"]
    assert ladder["contrast"] == BASELINE_LADDER["contrast"]
    assert len([note for note in notes if "回退" in note]) == 3


def test_level_without_usable_success_is_reported() -> None:
    """明细里有该档、但算不出可用跟住率 ⇒ 报出来（不许静默丢档）。"""
    rows = make_rows({"blur": [0.6, 1.0, 1.5]})
    for row in rows:
        if row["condition_family"] == "blur" and row["condition_level"] == "1.5":
            row["xcorr_success"] = "nan"
    notes: list[str] = []
    validity = rc.curve_validity(rows, None, notes)
    assert validity[("xcorr", "blur")]["levels"] == [0.6, 1.0]
    assert any("1.5" in note for note in notes)


# ---------------------------------------------------------------- deep 覆盖
def test_deep_levels_are_covered_not_dropped() -> None:
    """deep 档位一个不落：曲线的档位数 = 数据里的档位数。"""
    validity = rc.curve_validity(make_rows(DEEP_LADDER))
    for family in ("blur", "motion", "contrast"):
        assert validity[("xcorr", family)]["levels"] == DEEP_LADDER[family]
        assert len(validity[("xcorr", family)]["rates"]) == len(DEEP_LADDER[family])


def test_nonmonotonicity_only_at_deep_level_is_detected() -> None:
    """只在最深档出现的反弹**必须**被抓到（旧版只看基线档 ⇒ 漏判成单调）。"""
    rows = make_rows(DEEP_LADDER, bump={"contrast": 0.05})
    validity = rc.curve_validity(rows)
    contrast = validity[("xcorr", "contrast")]
    assert contrast["mono"] is False
    assert contrast["grade"].startswith("C-DEGENERATE(not-monotone)")

    # 去掉最深档（= 只剩基线档）⇒ 单调 ⇒ 旧版正是在这里漏判的
    shallow_rows = [row for row in rows
                    if row["condition_family"] != "contrast"
                    or float(row["condition_level"]) != 0.05]
    assert rc.curve_validity(shallow_rows)[("xcorr", "contrast")]["mono"] is True


def test_metric_monotonicity_covers_deep_levels() -> None:
    """指标侧单调性也必须覆盖 deep 档位（否则 σ6/σ8 的指标反弹永远发现不了）。"""
    rows = make_rows(DEEP_LADDER, bump={"contrast": 0.05})
    monotonicity = rc.metric_monotonicity(rows)
    for family in ("blur", "motion", "contrast"):
        _mono, _relspan, meds = monotonicity[(family, "local_scr")]
        assert len(meds) == len(DEEP_LADDER[family])
    mono, _relspan, _meds = monotonicity[("contrast", "local_scr")]
    assert mono is False


def test_deep_span_covers_the_hard_half() -> None:
    """跨度的分母/分子必须包含 deep 档（否则跨度只反映浅半截轴）。"""
    rows = make_rows(DEEP_LADDER)
    validity = rc.curve_validity(rows)
    deep_span = validity[("xcorr", "blur")]["span"]
    shallow_rows = [row for row in rows
                    if row["condition_family"] != "blur"
                    or float(row["condition_level"]) <= 4.0]
    shallow_span = rc.curve_validity(shallow_rows)[("xcorr", "blur")]["span"]
    assert deep_span > shallow_span


# ---------------------------------------------------------------- 回归护栏
def test_baseline_ladder_is_unchanged_by_derivation() -> None:
    """喂基线轮的明细 ⇒ 反推出的阶梯与旧版硬编码阶梯逐位相同。"""
    detail = rc.load_detail(str(BASELINE_DETAIL))
    ladder, notes = rc.derive_ladder(detail)
    assert notes == []
    assert ladder == BASELINE_LADDER


def test_baseline_cli_numbers_do_not_move(tmp_path: Path) -> None:
    """**回归护栏**：跑基线轮的明细 ⇒ 跨度/单调性/A-B-C 等级与旧版逐位相同。"""
    detail_path = tmp_path / "degrade_detail_v2.csv"
    detail_path.write_bytes(BASELINE_DETAIL.read_bytes())
    environ = dict(os.environ)
    environ["DEGRADE_DETAIL"] = str(detail_path)
    result = subprocess.run(
        [sys.executable, str(HERE / "recheck_curves.py"), str(tmp_path)],
        capture_output=True, text=True, env=environ, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout

    # §2 跟住率曲线有效性（跨度 / 单调性 / 等级）—— 与旧版逐位相同
    assert "blur      span=0.0793 mono=True  A-USABLE" in out
    assert "motion    span=0.0615 mono=True  A-USABLE" in out
    assert "blur      span=0.0380 mono=True  B-WEAK(span<=0.05)" in out
    assert "contrast  span=0.0112 mono=False C-DEGENERATE(not-monotone)" in out
    assert "contrast  span=0.0212 mono=True  B-WEAK(span<=0.05)" in out
    assert "blur      span=0.0145 mono=False C-DEGENERATE(not-monotone)" in out
    # §5 结论：A 级组合集合不变
    assert "达到 A 级(单调且跨度>0.05)的组合: [('xcorr', 'blur'), ('xcorr', 'motion')]" in out
    # §0 新增的阶梯反推块也必须与基线阶梯一致
    assert "blur      n=6   0.6 1.0 1.5 2.2 3.0 4.0" in out
    assert "motion    n=5   3.0 5.0 7.0 9.0 11.0" in out
    assert "contrast  n=6   0.8 0.65 0.5 0.35 0.25 0.15" in out


def test_baseline_grade_matrix_is_stable() -> None:
    """**回归护栏**：基线轮 9 个(臂,族)组合的等级逐个比对。"""
    detail = rc.load_detail(str(BASELINE_DETAIL))
    validity = rc.curve_validity(detail)
    expected = {
        ("xcorr", "blur"): ("A-USABLE", 0.0793, True),
        ("xcorr", "motion"): ("A-USABLE", 0.0615, True),
        ("xcorr", "contrast"): ("C-DEGENERATE(not-monotone)", 0.0112, False),
        ("kcf_legacy", "blur"): ("B-WEAK(span<=0.05)", 0.0380, True),
        ("kcf_legacy", "motion"): ("C-DEGENERATE(not-monotone)", 0.0123, False),
        ("kcf_legacy", "contrast"): ("B-WEAK(span<=0.05)", 0.0212, True),
        ("csrt", "blur"): ("C-DEGENERATE(not-monotone)", 0.0145, False),
        ("csrt", "motion"): ("C-DEGENERATE(not-monotone)", 0.0022, False),
        ("csrt", "contrast"): ("C-DEGENERATE(not-monotone)", 0.0034, False),
    }
    for key, (grade, span, mono) in expected.items():
        assert validity[key]["grade"] == grade
        assert round(validity[key]["span"], 4) == span
        assert validity[key]["mono"] is mono


@pytest.mark.parametrize("family", ["blur", "motion", "contrast"])
def test_baseline_metric_monotonicity_is_stable(family: str) -> None:
    """**回归护栏**：基线轮的指标侧单调性判定不变（contrast/local_scr 仍 NOT-MONO）。"""
    detail = rc.load_detail(str(BASELINE_DETAIL))
    monotonicity = rc.metric_monotonicity(detail)
    expected = {("blur", "local_scr"): True, ("blur", "target_local_contrast"): True,
                ("blur", "target_lap_var"): True, ("motion", "local_scr"): True,
                ("motion", "target_local_contrast"): True,
                ("motion", "target_lap_var"): True,
                ("contrast", "local_scr"): False,
                ("contrast", "target_local_contrast"): True,
                ("contrast", "target_lap_var"): True}
    for metric in rc.CORE_METRICS:
        mono, _relspan, meds = monotonicity[(family, metric)]
        assert mono is expected[(family, metric)]
        assert len(meds) == len(BASELINE_LADDER[family])


# ---------------------------------------------------------------- 膝点表路径可配
def candidate_rows(stdout: str) -> list[str]:
    """从 §4 区块里抽出「有效候选」的行（去掉表头与分隔线）。

    Args:
        stdout (str): CLI 的标准输出。

    Returns:
        list[str]: 候选行。
    """
    section = stdout.split("§4")[-1].split("§5")[0]
    return [line for line in section.splitlines()
            if line.split() and line.split()[0] in ("xcorr", "kcf_legacy", "csrt")]


def test_parse_args_splits_positional_and_flags() -> None:
    """``--knee-csv`` 的值**不能**被当成位置参数（否则 root 会被带偏）。"""
    positional, options = rc.parse_args(["iqa_v2", "--knee-csv", "out_deep/knee.csv", "--blind"])
    assert positional == ["iqa_v2"]
    assert options["--knee-csv"] == "out_deep/knee.csv"
    assert options["--blind"] == ""
    positional, options = rc.parse_args(["--knee-csv=out_deep/knee.csv", "iqa_v2"])
    assert positional == ["iqa_v2"]
    assert options["--knee-csv"] == "out_deep/knee.csv"


def test_knee_csv_flag_overrides_default(tmp_path: Path) -> None:
    """``--knee-csv`` 能把膝点表指到别处（深轮在 ``output_deep/``）。"""
    detail_path = tmp_path / "degrade_detail_v2.csv"
    detail_path.write_bytes(BASELINE_DETAIL.read_bytes())
    knee_path = tmp_path / "output_deep" / "knee_thresholds_v2.csv"
    knee_path.parent.mkdir(parents=True, exist_ok=True)
    knee_path.write_text("metric,curve_set,stratum,tracker,degradation_family,"
                         "knee_x_final,knee_y_final,delta_pct_of_span,knee_reliable\n"
                         "local_scr,common_good,all,xcorr,blur,1.25,0.5,12.5,1\n",
                         encoding="utf-8")

    import os  # noqa: PLC0415

    environ = dict(os.environ)
    environ["DEGRADE_DETAIL"] = str(detail_path)
    result = subprocess.run(
        [sys.executable, str(HERE / "recheck_curves.py"), str(tmp_path),
         "--knee-csv", str(knee_path)],
        capture_output=True, text=True, env=environ, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    # 指定的膝点表生效 ⇒ §4 里只有它那一条候选
    rows = candidate_rows(result.stdout)
    assert len(rows) == 1
    assert "xcorr" in rows[0] and "common_good" in rows[0] and "blur" in rows[0]


def test_knee_csv_env_var_is_honoured(tmp_path: Path) -> None:
    """环境变量 ``KNEE_CSV`` 与命令行等价。"""
    detail_path = tmp_path / "degrade_detail_v2.csv"
    detail_path.write_bytes(BASELINE_DETAIL.read_bytes())
    knee_path = tmp_path / "knee.csv"
    knee_path.write_text("metric,curve_set,stratum,tracker,degradation_family,"
                         "knee_x_final,knee_y_final,delta_pct_of_span,knee_reliable\n"
                         "local_scr,common_good,all,xcorr,blur,1.25,0.5,12.5,1\n",
                         encoding="utf-8")

    import os  # noqa: PLC0415

    environ = dict(os.environ)
    environ["DEGRADE_DETAIL"] = str(detail_path)
    environ["KNEE_CSV"] = str(knee_path)
    result = subprocess.run(
        [sys.executable, str(HERE / "recheck_curves.py"), str(tmp_path)],
        capture_output=True, text=True, env=environ, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(candidate_rows(result.stdout)) == 1


def test_missing_knee_table_is_reported_loudly(tmp_path: Path) -> None:
    """膝点表找不到/是空 ⇒ 必须**大声报出来**（§4 悄悄空着 = 假绿）。"""
    detail_path = tmp_path / "degrade_detail_v2.csv"
    detail_path.write_bytes(BASELINE_DETAIL.read_bytes())

    import os  # noqa: PLC0415

    environ = dict(os.environ)
    environ["DEGRADE_DETAIL"] = str(detail_path)
    environ["KNEE_CSV"] = str(tmp_path / "nope.csv")
    result = subprocess.run(
        [sys.executable, str(HERE / "recheck_curves.py"), str(tmp_path)],
        capture_output=True, text=True, env=environ, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "复算告警" in result.stdout
    assert "膝点表为空或不存在" in result.stdout


# ---------------------------------------------------------------- 审计器侧口径
def run_audit(root: Path, report_path: Path) -> tuple[int, list[tuple[str, str, str]]]:
    """跑一次审计器并回收退出码与逐条结论。

    Args:
        root (Path): 产物根目录。
        report_path (Path): 报告 md 的路径。

    Returns:
        tuple: ``(退出码, [(状态, 编号, 结论), ...])``。
    """
    audit.RES.clear()
    code = audit.main(str(root), str(report_path))
    return code, list(audit.RES)


def statuses(records: list[tuple[str, str, str]], code: str) -> list[tuple[str, str]]:
    """按编号筛出审计结论。

    Args:
        records (list): ``run_audit`` 的结论列表。
        code (str): 编号（如 ``"R11"``）。

    Returns:
        list: ``[(状态, 结论文本), ...]``。
    """
    return [(status, text) for status, item_code, text in records if item_code == code]


def test_audit_detects_deep_ladder_from_detail() -> None:
    """审计器从明细底表**自动识别**深轮的 10/7/8 阶梯（不写死）。"""
    severity = audit.detect_severity(make_rows(DEEP_LADDER))
    assert severity["blur"] == DEEP_LADDER["blur"]
    assert severity["motion"] == DEEP_LADDER["motion"]
    assert severity["contrast"] == DEEP_LADDER["contrast"]


def test_audit_known_ladders_are_recognised_and_alive() -> None:
    """``SEVERITY`` / ``DEEP_SEVERITY`` 都**在用**（交叉校验，不是死代码）。"""
    assert audit.match_known_ladder(audit.detect_severity(make_rows(BASELINE_LADDER))) == "基线轮"
    assert audit.match_known_ladder(audit.detect_severity(make_rows(DEEP_LADDER))) == "深轮(deep)"
    assert audit.KNOWN_LADDERS["基线轮"] is audit.SEVERITY
    assert audit.KNOWN_LADDERS["深轮(deep)"] is audit.DEEP_SEVERITY


def test_audit_and_recheck_agree_on_the_ladder() -> None:
    """两个仪器对「实际跑过的阶梯」必须**口径一致**。"""
    rows = make_rows(DEEP_LADDER)
    ladder, _notes = rc.derive_ladder(rows)
    severity = audit.detect_severity(rows)
    assert severity == ladder


def test_audit_reports_heaviest_level_rate_not_the_mean(tmp_path: Path) -> None:
    """R11 报的必须是**最重档那一档**的跟住率（全档均值另起名，不能冒充）。"""
    rows = [make_row("clean", "0.0", 1.0)]
    for level, rate in (("0.6", 0.9), ("1.0", 0.8)):
        for _ in range(2):
            rows.append(make_row("blur", level, rate))
    write_detail(tmp_path / "degrade_detail_v2.csv", rows)
    report = tmp_path / "REPORT.md"
    report.write_text("本轮曲线过浅，膝点是弯折点而非崩溃点。", encoding="utf-8")

    _code, records = run_audit(tmp_path, report)
    evidence = statuses(records, "R11")
    assert evidence and evidence[0][0] == "PASS"
    assert "最重档 1.0 跟住率 0.800" in evidence[0][1]
    assert "全档平均跟住率 0.850" in evidence[0][1]


def test_audit_missing_detail_is_a_loud_fail(tmp_path: Path) -> None:
    """明细底表缺失 ⇒ **显式 FAIL**（不是「跳过 ⇒ 看起来全绿」）。"""
    report = tmp_path / "REPORT.md"
    report.write_text("本轮曲线过浅，膝点是弯折点而非崩溃点。", encoding="utf-8")
    code, records = run_audit(tmp_path, report)
    assert code == 1
    for code_name in ("R1", "R1d", "R2", "R2b", "R11"):
        found = statuses(records, code_name)
        assert found, "明细缺失时 %s 必须有结论" % code_name
        assert found[0][0] == "FAIL", "%s 必须判 FAIL（实际 %s）" % (code_name, found[0][0])


def test_audit_detects_deep_only_non_monotonicity(tmp_path: Path) -> None:
    """只在最深档（α0.05）出现的非单调 ⇒ 审计器必须判 FAIL（写死基线阶梯 ⇒ 漏检）。"""
    rows = make_rows(DEEP_LADDER, bump={"contrast": 0.05})
    write_detail(tmp_path / "degrade_detail_v2.csv", rows)
    report = tmp_path / "REPORT.md"
    report.write_text("本轮曲线过浅，膝点是弯折点而非崩溃点。", encoding="utf-8")

    _code, records = run_audit(tmp_path, report)
    mono = statuses(records, "R2b")
    assert mono and mono[0][0] == "FAIL"
    assert "contrast/local_scr" in mono[0][1]
    ladder = statuses(records, "R0a")
    assert ladder and "8.0" in ladder[0][1] and "0.05" in ladder[0][1]
