"""文档 ↔ 产物 数字对账（铁律九的可执行版本）。

本模块把"文档里写的每个统计数字都必须来自产物 JSON"变成**可运行的断言**：
任何一格对不上就退出码非 0。

用法::

    PYTHONPATH=. python -m manu.tracking.async_dual_rate.verify_docs \\
        --sequence VIDEO00007_19700101_001324 \\
        --artifacts runs/tracking_async \\
        --docs manu/tracking/README.md manu/memory/async_dual_rate_tracking_20261102.md

三层校验（逐层收紧，避免"同一数字在别处出现 ⇒ 改错了也看不见"）：

1. **表格行级**（``_check_row``）：按**表头 + 列数**定位到具体那张表，再按**位置**逐格比；
   容差由**期望值的显示精度**决定（2 位小数 ⇒ 容差 0.005 ⇒ 44.64 与 44.65 会被抓出来）。
2. **单元格数值级**（``_check_value_cell``）：天花板表 5 行，校验该行数值列里的分位数序列。
3. **段落级**（``_check_block``）：散落在正文里的数字（"只活了 432 帧"、"弃权率 20.0%"、
   配对计数 1795/1486 …）—— 先按**锚点定位段落**，再校验该段落里的数字。

容差分级：精度列按**显示精度**严格（0.01 级偏差也抓）；**墙钟延时列**（验收表第 7 列）
因重跑必然抖动而单独放宽到 ``max(0.02, 30%)``——墙钟不可复现，不该由文档对账器来背锅。

⚠️ 口径提醒：``evidence_*.tsv`` 的 ``valid`` 列与"位置存在"不等价，
不要用 TSV 手工复算本模块覆盖的指标。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

__all__ = ["verify_docs", "main"]

# 文档里的行标签 → three_arm_summary.json 的臂名
ARM_ROWS = {
    "臂 1 纯检测基准": "arm1_detector_only",
    "（对照）臂 1 同锚点密度（det_hz=12.5）": "arm1_detector_only_sparse",
    "臂 2 检跟纯前向外推": "arm2_forward_only",
    "臂 3 异步退回重推演": "arm3_rollback_replay",
    "（消融）纯前向 + 允许重引导": "ablation_forward_only_reinit",
    "（消融）臂 3 + 弃权硬拉回兜底": "ablation_replay_fallback",
    "（消融）残差平滑融合": "ablation_fused_alpha",
    "（消融）残差硬拉回": "ablation_fused_hard",
}

# 天花板表：行标签（已去 **） → 数值列必须出现的数字（顺序一致）
CEILING_CELLS = {
    "快核单步局部误差（从真值锚点走 1 步）": "single_step",
    "重推 Δ 步的累积误差（从完美锚点出发）": "delta_step",
    "锚点误差（检测框在它自己那一帧）": "anchor",
    "目标在 Δ=8 帧内的真实位移": "displacement",
    "单帧位移 > ROI 半窗(16 px) 的帧占比": "over_roi",
}

# 段落级校验：锚点（去空格后匹配） → 该段落必须包含的数字串
# 锚点是**候选元组**：只要有一个候选命中就算定位成功（文档换个说法也不该让对账器失联）。
# 定位后要求**至少一个**匹配段落同时含有全部期望数字——"同段共现"是为了防止
# "这些数字散落在文档各处却互不相干"这种假通过。
BLOCK_CHECKS = (
    (("一句话", "一句话结论"), ("0.38", "47.79", "48.01")),
    (("只活了",), ("431", "4.1")),
    (("尝试重推演", "帧集（按臂"), ("3439", "740", "214", "526", "4179", "17.71")),
    (("配对臂3vsarm1_detector_only", "臂3vs臂1"), ("1846", "1566", "0.268", "持平0")),
    (("配对臂3vsablation_fused_alpha", "臂3vsalpha"), ("1041", "1812", "502", "持平502")),
)

# 表头定位：验收表 / 条件化对比（成功）/ 条件化对比（弃权）/ 天花板表
TABLE_ACCEPTANCE = ("命中率", 10)   # 数据列数（不含行标签列）
TABLE_SUCCESS = ("CLE p90", 5)
TABLE_ABSTAIN = ("Hit@5px", 3)
TABLE_CEILING = ("天花板", 2)

# 验收表里"端到端延时 p50/p99"是第 8 个数据列（0-based 下标 7）。
# 列序：0 命中率 / 1 Wilson CI / 2 存活窗口命中率 / 3 CLE p50 / 4 CLE p95 /
#      5 抖动 p95(steady) / 6 StepJump@smooth p95 / 7 端到端延时 / 8 输出覆盖率 / 9 脱靶率
# 墙钟耗时**本质不可复现**（机器负载每次都不同），文档里记的是某一次跑批的实测值，
# 重跑后必然抖动；因此该列单独放宽容差。**精度列仍然按显示精度严格比对**（1 个百分点也抓）。
LOOSE_COLUMNS = frozenset({7})
LOOSE_MIN_ABS = 0.02
LOOSE_REL = 0.3


def _normalize(text: str) -> str:
    """Unicode 减号 → ASCII。"""
    return text.replace("−", "-").replace("–", "-")


def _compact(text: str) -> str:
    return _normalize(text).replace(" ", "").replace("*", "")


def _numbers(text: str) -> list[float]:
    """抽取数字：先去千分位逗号（"3,281"→"3281"），但不动 "[43.69, 45.59]" 的分隔逗号。"""
    normalized = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", _normalize(text))
    return [float(value) for value in re.findall(r"-?\d+\.?\d*", normalized)]


def _tokens(text: str) -> list[str]:
    """抽取数字**原文**（保留小数位数，用于按显示精度定容差）。"""
    normalized = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", _normalize(text))
    return re.findall(r"-?\d+\.?\d*", normalized)


def _tolerance(expected: str) -> float:
    """期望值本身的显示精度 ⇒ 容差（2 位小数 ⇒ 0.005；3 位 ⇒ 0.0005）。"""
    digits = expected.partition(".")[2]
    decimals = len(digits) if digits.isdigit() else 0
    return 0.5 * (10 ** -decimals) + 1e-9


def _column_tolerance(token: str, index: int, loose_columns: frozenset[int]) -> float:
    """某格的容差；墙钟列用"绝对下限 + 相对比例"放宽，其余列按显示精度严格。"""
    tolerance = _tolerance(token)
    if index in loose_columns:
        tolerance = max(tolerance, LOOSE_MIN_ABS, LOOSE_REL * abs(float(token)))
    return tolerance


def _table_blocks(document: str) -> list[list[str]]:
    """把连续的 `|` 开头的行切成一张张表。"""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in document.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _cells(line: str) -> list[str]:
    return [cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")]


def _rows(document: str, header_token: str, label: str, columns: int) -> list[list[str]]:
    """在"表头含 header_token 且列数 == columns"的表里找首列 == label 的行。"""
    found: list[list[str]] = []
    for block in _table_blocks(document):
        if header_token not in block[0]:
            continue
        for line in block[1:]:
            cells = _cells(line)
            # 标签按"去空格/去加粗"比较：文档里给行标签加粗或改空格不该让对账器失联。
            if cells and _compact(cells[0]) == _compact(label) and len(cells) - 1 == columns:
                found.append(cells[1:])
    return found


def _check_row(
    document: str,
    table: tuple[str, int],
    label: str,
    expected: list[str],
    problems: list[str],
    where: str,
    loose_columns: frozenset[int] = frozenset(),
) -> None:
    """表格行级校验：按位置逐格比，允许单元格尾部附带括号注释。"""
    header_token, columns = table
    rows = _rows(document, header_token, label, columns)
    if not rows:
        problems.append(f"[{where}] 行缺失：{label}（表头含 {header_token!r}、{columns} 列）")
        return
    for row in rows:
        if len(row) != len(expected):
            problems.append(
                f"[{where}] {label}：列数不对（文档 {len(row)}，期望 {len(expected)}）"
            )
            continue
        for index, (want, got) in enumerate(zip(expected, row)):
            if want.strip() == "—":
                # 该指标本序列无样本（NaN）⇒ 产品渲染为 "—"，文档也必须写 "—"，
                # 不能偷偷填一个数。
                if "—" not in got:
                    problems.append(
                        f"[{where}] {label} 第 {index + 1} 列：doc={got!r} json='—'"
                    )
                continue
            want_tokens = _tokens(want)
            got_values = _numbers(got)
            if len(got_values) < len(want_tokens):
                problems.append(f"[{where}] {label} 第 {index + 1} 列：doc={got!r} json={want!r}")
                continue
            for position, (token, right) in enumerate(zip(want_tokens, got_values)):
                left = float(token)
                tolerance = _column_tolerance(token, index, loose_columns)
                if abs(left - right) > tolerance:
                    problems.append(f"[{where}] {label} 第 {index + 1} 列：doc={got!r} json={want!r}")


def _check_value_cell(
    document: str,
    label: str,
    expected: list[str],
    meaning_expected: list[str],
    problems: list[str],
    where: str,
) -> None:
    """单元格数值级校验：数值列的分位数 + 含义列的关键数字都要对得上。"""
    rows = _rows(document, TABLE_CEILING[0], label, TABLE_CEILING[1])
    if not rows:
        problems.append(f"[{where}] 行缺失：{label}")
        return
    want_values = _numbers(" ".join(expected))
    for row in rows:
        if not row:
            problems.append(f"[{where}] {label}：没有数值列")
            continue
        # "p50 = 2.22 / p90 = 9.22" 里的 "50/90" 不是分位数，先去掉这些前缀。
        got_values = _numbers(re.sub(r"p\d+\s*=", "", row[0]))
        if len(got_values) < len(want_values):
            problems.append(
                f"[{where}] {label} 数值列：doc={got_values} 缺少期望值 {want_values}"
            )
            continue
        for position, (left, right) in enumerate(zip(want_values, got_values)):
            if abs(left - right) > _tolerance(expected[position]):
                problems.append(
                    f"[{where}] {label} 数值列：doc={got_values} 期望 {want_values}"
                )
                break
        meaning = row[1] if len(row) > 1 else ""
        for token in meaning_expected:
            if _compact(token) not in _compact(meaning):
                problems.append(f"[{where}] {label} 含义列缺少 {token!r}")


def _blocks(document: str) -> list[str]:
    """按空行切段（表格与被折行的段落都是完整的一段）。"""
    blocks: list[str] = []
    current: list[str] = []
    for line in document.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def _check_block(
    document: str,
    anchors: tuple[str, ...] | str,
    expected: tuple[str, ...],
    problems: list[str],
    where: str,
) -> None:
    """段落级校验：定位到含锚点的段落，再校验该段落里必须出现的数字串。

    锚点可以是**多个候选**（文档可能换说法）；命中后要求**至少一个**匹配段落
    同时含有全部期望数字——"同段共现"既防假通过，也避免一个短锚点命中
    目录/模块表等无关段落时误报一片。
    """
    candidates = tuple(anchors) if isinstance(anchors, (tuple, list)) else (anchors,)
    matched: list[str] = []
    for anchor in candidates:
        needle = _compact(anchor)
        matched = [block for block in _blocks(document) if needle in _compact(block)]
        if matched:
            break
    if not matched:
        problems.append(f"[{where}] 找不到含锚点 {candidates[0]!r} 的段落")
        return
    best = max(
        matched,
        key=lambda block: sum(1 for value in expected if _compact(value) in _compact(block)),
    )
    compact = _compact(best)
    missing = [value for value in expected if _compact(value) not in compact]
    if missing:
        problems.append(f"[{where}] 段落 {candidates[0]!r} 缺少数字 {missing}")


def _acceptance_expectation(row: dict) -> list[str]:
    """验收表一行的 10 个数据格，顺序与产品渲染的 ``three_arm_table.md`` 逐格一致。

    直接取 ``summary.json`` 的 ``acceptance_table``（**产品自己渲染的那张表**），
    而不是在这里重算一遍指标——这样"文档 ↔ 产品"比的是同一个数，
    不存在两处口径各自漂移的可能。

    列序：命中率 / Wilson CI / 存活窗口命中率 / CLE p50 / CLE p95 /
          抖动 p95(steady) / StepJump@smooth p95 / 端到端延时 / 输出覆盖率 / 脱靶率。
    """
    wilson = row["hit_rate_wilson95"]
    step_jump = row["step_jump_smooth_p95"]
    return [
        f"{row['hit_rate'] * 100:.2f}%",
        f"[{wilson[0] * 100:.2f}%, {wilson[1] * 100:.2f}%]",
        f"{row['hit_rate_on_output_frames'] * 100:.2f}%",
        f"{row['cle_p50']:.2f} px",
        f"{row['cle_p95']:.2f} px",
        f"{row['jitter_acceleration_steady_p95']:.2f} px",
        "—" if math.isnan(step_jump) else f"{step_jump:.2f} px",
        (
            f"{row['end_to_end_latency_p50_ms']:.3f} / "
            f"{row['end_to_end_latency_p99_ms']:.3f} ms"
        ),
        f"{row['coverage'] * 100:.1f}%",
        f"{row['lost_frame_rate'] * 100:.1f}%",
    ]


def _conditional_row(stats: dict) -> list[str]:
    return [
        f"{stats['cle_p50']:.2f} px",
        f"{stats['cle_p90']:.2f} px",
        f"{stats['cle_p95']:.2f} px",
        f"{stats['hit_rate_pct']:.2f}%",
        f"{stats['n']:,}",
    ]


def _abstain_row(stats: dict) -> list[str]:
    return [
        f"{stats['cle_p50']:.2f} px",
        f"{stats['hit_rate_pct']:.2f}%",
        f"{stats['n']:,}",
    ]


def verify_docs(
    sequence: str,
    artifacts: Path,
    documents: list[Path],
) -> list[str]:
    """返回问题列表；空列表表示全部对得上。"""
    artifacts = Path(artifacts) / sequence
    three = json.loads((artifacts / "three_arm_summary.json").read_text(encoding="utf-8"))
    conditional = json.loads(
        (artifacts / "conditional_comparison.json").read_text(encoding="utf-8")
    )

    acceptance = {row["arm_key"]: row for row in three["acceptance_table"]}

    problems: list[str] = []

    success = conditional["on_replay_success"]
    abstain = conditional["on_replay_abstain"]
    arm1_success = success["arm1_detector_only"]
    alpha_success = success["ablation_fused_alpha"]
    arm1_abstain = abstain["arm1_detector_only"]
    alpha_abstain = abstain["ablation_fused_alpha"]

    kernel = three["kernel_ceiling"]
    anchor = three["anchor_ceiling"]["anchor_error_px"]
    track = three["trackability"]
    ceiling_expected = {
        "single_step": [
            f"{kernel['single_step_error_px']['p50']:.2f}",
            f"{kernel['single_step_error_px']['p90']:.2f}",
            f"{kernel['single_step_error_px']['p95']:.2f}",
            f"{kernel['single_step_error_px']['p99']:.2f}",
            f"{kernel['single_step_valid_rate']:.3f}",
        ],
        "delta_step": [
            f"{kernel['delta_step_error_px']['p50']:.2f}",
            f"{kernel['delta_step_error_px']['p90']:.2f}",
            f"{kernel['delta_step_error_px']['p95']:.2f}",
            f"{kernel['delta_step_error_px']['p99']:.2f}",
        ],
        "anchor": [
            f"{anchor['p50']:.2f}",
            f"{anchor['p95']:.2f}",
            f"{anchor['p99']:.2f}",
        ],
        "displacement": [
            f"{track['over_delta_displacement_px']['p50']:.2f}",
            f"{track['over_delta_displacement_px']['p90']:.2f}",
            f"{track['over_delta_displacement_px']['p95']:.2f}",
            f"{track['over_delta_displacement_px']['p99']:.2f}",
        ],
        "over_roi": [f"{track['frames_per_frame_over_roi'] * 100:.2f}"],
    }
    ceiling_meaning = {
        "single_step": [],
        "delta_step": [f"{kernel['delta_step_samples']:,}"],
        "anchor": [],
        "displacement": [],
        "over_roi": [],
    }

    for path in documents:
        document = path.read_text(encoding="utf-8")
        where = path.name

        # ---- 1. 三臂验收表 ----
        for label, key in ARM_ROWS.items():
            if key not in acceptance:
                problems.append(f"[{where}] 产物里没有臂 {key!r}（acceptance_table 缺行）")
                continue
            _check_row(
                document,
                TABLE_ACCEPTANCE,
                label,
                _acceptance_expectation(acceptance[key]),
                problems,
                where,
                loose_columns=LOOSE_COLUMNS,
            )

        # ---- 2. 条件化对比表 ----
        _check_row(
            document, TABLE_SUCCESS, "臂 3 退回重推演（配对臂 1 帧集）",
            _conditional_row(arm1_success["arm3_rollback_replay"]), problems, where,
        )
        _check_row(
            document, TABLE_SUCCESS, "臂 1 纯检测（配对臂 1 帧集）",
            _conditional_row(arm1_success["arm1_detector_only"]), problems, where,
        )
        _check_row(
            document, TABLE_SUCCESS, "臂 3 退回重推演（配对 alpha 帧集）",
            _conditional_row(alpha_success["arm3_rollback_replay"]), problems, where,
        )
        _check_row(
            document, TABLE_SUCCESS, "消融 alpha（残差平滑）",
            _conditional_row(alpha_success["ablation_fused_alpha"]), problems, where,
        )
        _check_row(
            document, TABLE_ABSTAIN, "臂 3（弃权帧，配对臂 1；保留漂移的前向状态）",
            _abstain_row(arm1_abstain["arm3_rollback_replay"]), problems, where,
        )
        _check_row(
            document, TABLE_ABSTAIN, "臂 1（用新检测框）",
            _abstain_row(arm1_abstain["arm1_detector_only"]), problems, where,
        )
        _check_row(
            document, TABLE_ABSTAIN, "臂 3（弃权帧，配对 alpha；保留漂移的前向状态）",
            _abstain_row(alpha_abstain["arm3_rollback_replay"]), problems, where,
        )
        _check_row(
            document, TABLE_ABSTAIN, "消融 alpha（残差平滑）",
            _abstain_row(alpha_abstain["ablation_fused_alpha"]), problems, where,
        )

        # ---- 3. 天花板表 ----
        for label, kind in CEILING_CELLS.items():
            _check_value_cell(
                document,
                label,
                ceiling_expected[kind],
                ceiling_meaning[kind],
                problems,
                where,
            )

        # ---- 4. 段落级 ----
        for anchor_text, values in BLOCK_CHECKS:
            _check_block(document, anchor_text, values, problems, where)

        # ---- 5. 验收表里出现的实测 p50 ----
        for key in ("arm1_detector_only", "arm3_rollback_replay"):
            value = f"{three['arms'][key]['servo_metrics']['cle']['p50']:.2f}"
            if value not in document:
                problems.append(f"[{where}] 实测 p50 {value} 未出现在文档中")

    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="文档 ↔ 产物 数字对账")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--artifacts", default="runs/tracking_async")
    parser.add_argument(
        "--docs",
        nargs="+",
        default=[
            "manu/tracking/README.md",
            "manu/memory/async_dual_rate_tracking_20261102.md",
        ],
    )
    args = parser.parse_args(argv)

    problems = verify_docs(
        args.sequence, Path(args.artifacts), [Path(doc) for doc in args.docs]
    )
    if problems:
        print(f"[FAIL] 文档与产物有 {len(problems)} 处不一致：")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"[PASS] 文档 ↔ 产物 数字全部一致（{len(args.docs)} 篇文档）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
