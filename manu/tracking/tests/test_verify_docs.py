"""文档 ↔ 产物 对账器的契约测试。

对账器是"文档里的数字必须来自产物"的最后一道闸门——它自己必须**能变红**，
而且**不能对无关内容误报**。

下面的用例**按解析出的表格定位单元格**再变异，而不是硬编码 README 里的
字符串：文档排版（加粗、换行、列序说明）一变，硬编码就会假挂；
定位式变异只依赖"这张表存在且列数符合契约"，改动排版不会误伤，
改错数字则一定被抓。
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from manu.tracking.async_dual_rate.verify_docs import (
    TABLE_ACCEPTANCE,
    TABLE_CEILING,
    _cells,
    _table_blocks,
    verify_docs,
)

ARTIFACTS = Path("runs/tracking_async")
SEQUENCE = "VIDEO00007_19700101_001324"
README = Path("manu/tracking/README.md")
MEMORY = Path("manu/memory/async_dual_rate_tracking_20261102.md")

# 验收表的列序（0-based，含行标签列）
COL_LABEL = 0
COL_HIT_RATE = 1
COL_CLE_P50 = 4
COL_LATENCY = 8


def _artifacts_available() -> bool:
    return (ARTIFACTS / SEQUENCE / "three_arm_summary.json").exists()


def _find_row_line(document: str, header_token: str, columns: int, label_prefix: str) -> str:
    """定位一行原始文本：表头含 header_token、列数 == columns、行标签以 label_prefix 开头。"""
    for block in _table_blocks(document):
        if header_token not in block[0]:
            continue
        for line in block[1:]:
            cells = _cells(line)
            if cells and cells[0].startswith(label_prefix) and len(cells) - 1 == columns:
                return line
    raise AssertionError(
        f"文档里找不到行 {label_prefix!r}（表头含 {header_token!r}、{columns} 列）"
    )


def _replace_cell(line: str, index: int, new_text: str) -> str:
    """把一行的第 index 格替换成 new_text（保留表格结构，去掉加粗标记）。"""
    cells = line.strip().strip("|").split("|")
    cells[index] = f" {new_text} "
    return "| " + " | ".join(cell.strip() for cell in cells) + " |"


def _bump_number(cell_text: str, delta: float) -> str:
    """把单元格里的第一个数按 delta 平移，保持原有显示精度。"""
    match = re.search(r"-?\d+\.\d+", cell_text)
    if match is None:
        raise AssertionError(f"单元格里没有小数可改：{cell_text!r}")
    decimals = len(match.group(0).partition(".")[2])
    new_value = float(match.group(0)) + delta
    return cell_text.replace(match.group(0), f"{new_value:.{decimals}f}", 1)


@unittest.skipUnless(_artifacts_available(), "产物未落盘（需先跑三臂主流程）")
class VerifyDocsTest(unittest.TestCase):
    def _run(self, text: str) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.md"
            path.write_text(text, encoding="utf-8")
            return verify_docs(SEQUENCE, ARTIFACTS, [path])

    def _readme(self) -> str:
        return README.read_text(encoding="utf-8")

    def test_real_documents_pass(self):
        problems = verify_docs(SEQUENCE, ARTIFACTS, [README, MEMORY])
        self.assertEqual(problems, [], f"文档与产物不一致：{problems}")

    def test_acceptance_table_has_the_contracted_shape(self):
        """验收表必须是 11 列（臂 / 命中率 / CI / 存活窗口 / p50 / p95 / 抖动 /
        StepJump / 延时 / 覆盖率 / 脱靶率）——少一列说明文档没跟上指标口径。"""
        document = self._readme()
        header_token, columns = TABLE_ACCEPTANCE
        rows = [
            line
            for block in _table_blocks(document)
            if header_token in block[0]
            for line in block[1:]
            if len(_cells(line)) - 1 == columns
        ]
        self.assertGreaterEqual(len(rows), 8, f"验收表行数不足：{len(rows)}")

    def test_detects_a_corrupted_table_number(self):
        """验收表里的数字改错 ⇒ 必须报错。"""
        original = self._readme()
        line = _find_row_line(original, *TABLE_ACCEPTANCE, "臂 1 纯检测基准")
        broken = original.replace(line, _replace_cell(line, COL_HIT_RATE, "99.99%"), 1)
        self.assertNotEqual(broken, original)
        self.assertTrue(self._run(broken))

    def test_detects_a_one_cent_drift(self):
        """**0.01 级**的偏差也必须被抓到（显示精度之内不算错）。"""
        original = self._readme()
        line = _find_row_line(original, *TABLE_ACCEPTANCE, "臂 1 纯检测基准")
        cell = _cells(line)[COL_HIT_RATE]
        broken = original.replace(
            line, _replace_cell(line, COL_HIT_RATE, _bump_number(cell, 0.01)), 1
        )
        self.assertNotEqual(broken, original)
        problems = self._run(broken)
        self.assertTrue(problems)
        self.assertIn("第 1 列", problems[0])

    def test_detects_a_number_that_also_appears_elsewhere(self):
        """同一个数字在文档别处也出现时，改错表格里那一处**照样要被抓到**。

        （这正是"纯字符串存在性校验"的系统性弱点。）
        """
        original = self._readme()
        self.assertGreaterEqual(original.count("3.83"), 2, "前提变了：3.83 只出现一次")
        line = _find_row_line(original, *TABLE_CEILING, "单帧位移 > ROI 半窗")
        broken = original.replace(line, _replace_cell(line, 1, "4.83%"), 1)
        self.assertNotEqual(broken, original)
        self.assertTrue(self._run(broken))

    def test_detects_a_deleted_number(self):
        """把受管数字整段删掉 ⇒ 必须报错。"""
        original = self._readme()
        broken = original.replace("341.28", "", 1)
        self.assertNotEqual(broken, original)
        self.assertTrue(self._run(broken))

    def test_detects_a_deleted_table_row(self):
        """整行删掉 ⇒ 必须报'行缺失'。"""
        original = self._readme()
        line = _find_row_line(original, *TABLE_ACCEPTANCE, "臂 2 检跟纯前向外推")
        broken = original.replace(line + "\n", "", 1)
        self.assertNotEqual(broken, original)
        problems = self._run(broken)
        self.assertTrue(problems)
        self.assertIn("行缺失", problems[0])

    def test_unrelated_content_does_not_trigger_false_positives(self):
        """追加无关段落 / 无关表格 / 无关数字 ⇒ 不能误报。"""
        original = self._readme()
        extra = (
            "\n## 附录\n\n随手记的数：123.456 / 789 / 0.001\n\n"
            "| 无关 | 表 |\n| --- | --- |\n| 1.23 | 4.56 |\n"
        )
        self.assertEqual(self._run(original + extra), [])

    def test_latency_column_tolerates_wallclock_jitter(self):
        """墙钟延时会随机器负载抖动 ⇒ 延时列的小幅变化**不应**报错。"""
        original = self._readme()
        line = _find_row_line(original, *TABLE_ACCEPTANCE, "臂 3 异步退回重推演")
        cell = _cells(line)[COL_LATENCY]
        # p50 与 p99 各抖一点：墙钟不可复现，必须在放宽容差内。
        numbers = re.findall(r"\d+\.\d+", cell)
        self.assertEqual(len(numbers), 2, f"延时格格式变了：{cell!r}")
        jittered = (
            cell.replace(numbers[0], f"{float(numbers[0]) + 0.002:.3f}", 1)
            .replace(numbers[1], f"{float(numbers[1]) + 0.058:.3f}", 1)
        )
        broken = original.replace(line, _replace_cell(line, COL_LATENCY, jittered), 1)
        self.assertNotEqual(broken, original)
        self.assertEqual(self._run(broken), [])

    def test_latency_looseness_does_not_leak_into_precision_columns(self):
        """延时列放宽后，精度列仍然要严格（0.01 级偏差照样抓）。"""
        original = self._readme()
        line = _find_row_line(original, *TABLE_ACCEPTANCE, "臂 3 异步退回重推演")
        cell = _cells(line)[COL_CLE_P50]
        broken = original.replace(
            line, _replace_cell(line, COL_CLE_P50, _bump_number(cell, 0.01)), 1
        )
        self.assertNotEqual(broken, original)
        problems = self._run(broken)
        self.assertTrue(problems)
        # 对账器按**数据列**编号（不含行标签列）：CLE p50 是第 4 个数据列。
        self.assertIn("第 4 列", problems[0])


class NumbersHelperTest(unittest.TestCase):
    def test_extracts_multiple_numbers(self):
        from manu.tracking.async_dual_rate.verify_docs import _numbers

        self.assertEqual(_numbers("[43.69, 45.59]"), [43.69, 45.59])
        self.assertEqual(_numbers("0.159 / 1.140 ms"), [0.159, 1.140])
        self.assertEqual(_numbers("3,281"), [3281.0])

    def test_handles_unicode_minus(self):
        from manu.tracking.async_dual_rate.verify_docs import _numbers

        self.assertEqual(_numbers("中位差 **−0.353 px**"), [-0.353])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
