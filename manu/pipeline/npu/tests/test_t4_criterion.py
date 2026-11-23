# -*- coding: utf-8 -*-
"""板卡 T4 判据（**绝对口径 OR 同会话相对基线口径**）的契约测试。

被测对象
--------
* ``manu.pipeline.npu.t4_criterion`` —— T4 判据的**唯一实现**（纯函数、零第三方依赖）
* ``auto_promote_watch.board_pass_map`` —— 消费者①（升轨守望）
* ``promote_candidate.check_arm`` —— 消费者②（升轨预检）

为什么要有这组测试（2026-11-20 现场事故，memory §266）
---------------------------------------------------
板卡读数**热漂移 ±0.5 ms**，而交付红线 19.989 ms 正落在漂移带中央
⇒ **同一张计算图**（基线家族克隆臂）现在测出 19.94~21.46 ms
⇒ 纯绝对口径把它们**全部误判 FAIL** ⇒ **升轨链最后一环断链**。
修法 = 引入「同会话相对基线」口径；**但历史上它被绝对口径刷掉过两次**：

* 2026-11-19 15:27 —— `sync_measured_to_ledger()` 用纯绝对口径把 116 个臂的
  `board_pass_t4=True` 覆盖回 False；
* 2026-11-19 15:30 —— `auto_promote_watch.board_pass_map()` 写死 `ms <= LIMIT_MS`
  ⇒ 相对口径的结果整个丢掉。

⇒ 这组测试锁死三件事：
  **T1** 判定函数本身在「热漂移 + 同图」时判 PASS、在「真更慢」时判 FAIL；
  **T2** 参照缺失时**退回绝对口径**（绝不误放行）；
  **T3** 两个消费者**只信台账里已算好的 `pass_t4`**（老记录才退回绝对口径）。

跑法（仓库根目录）::

    PYTHONPATH=. python -m unittest manu.pipeline.npu.tests.test_t4_criterion -v
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest

_NPU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_NPU_DIR)))
for _p in (_REPO_ROOT, _NPU_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from manu.pipeline.npu import t4_criterion as T4  # noqa: E402


class TestCriterion(unittest.TestCase):
    """T4 判据本体（纯函数）。"""

    def test_same_graph_under_drift_passes(self):
        """热漂移下**同一张计算图**的臂：绝对口径 FAIL、相对口径 PASS ⇒ 判 PASS。"""
        # 现场读数：同会话基线臂 20.067 ms、候选臂（同图）20.050 ms
        d = T4.decide(ms_median=20.050, diff_median=round(20.050 - 20.067, 4), tol=T4.diff_tol(19.629))
        self.assertFalse(d["ok_abs"], "绝对口径应当 FAIL（20.05 > 19.989）")
        self.assertTrue(d["ok_rel"], "相对口径应当 PASS（diff −0.017 ≤ 容差）")
        self.assertTrue(d["ok"], "综合判定必须 PASS（否则热漂移会误杀同图臂 ⇒ 升轨断链）")

    def test_really_slower_arm_fails(self):
        """**真更慢**的臂（比同会话基线慢 0.7 ms）⇒ 必须 FAIL。"""
        d = T4.decide(ms_median=20.80, diff_median=0.70, tol=T4.diff_tol(19.629))
        self.assertFalse(d["ok_abs"])
        self.assertFalse(d["ok_rel"], "diff 0.70 > 容差 0.36 ⇒ 相对口径必须 FAIL")
        self.assertFalse(d["ok"], "真更慢的臂绝不能被放行")

    def test_missing_ref_falls_back_to_absolute(self):
        """同会话**没有基线臂**（ref 缺失）⇒ 只剩绝对口径 ⇒ 不误放行、不崩。"""
        d = T4.decide(ms_median=20.05, diff_median=None, tol=T4.diff_tol(19.629))
        self.assertFalse(d["ok_rel"])
        self.assertFalse(d["ok"], "参照缺失时不得凭相对口径放行")
        d2 = T4.decide(ms_median=19.50, diff_median=None, tol=T4.diff_tol(19.629))
        self.assertTrue(d2["ok"], "参照缺失时仍应认绝对口径（19.50 ≤ 19.989）")

    def test_none_ms_never_passes(self):
        self.assertFalse(T4.decide(None, None, 0.36)["ok"])
        self.assertFalse(T4.decide(None, -0.5, 0.36)["ok_abs"])
        self.assertTrue(T4.decide(None, -0.5, 0.36)["ok_rel"],
                        "有参照差值但没读到绝对值时，相对口径仍可用")
        self.assertFalse(T4.decide(None, -0.5, 0.36)["ok"],
                         "🔴 没有绝对读数 ⇒ **不判定**（不能凭差值放行）")

    def test_boundary_is_inclusive(self):
        tol = T4.diff_tol(19.629)
        self.assertTrue(T4.decide(20.5, tol, tol)["ok_rel"], "diff == 容差 ⇒ PASS（≤）")
        self.assertFalse(T4.decide(20.5, tol + 0.001, tol)["ok_rel"])

    def test_diff_tol_from_clean_base_ref(self):
        self.assertAlmostEqual(T4.diff_tol(19.629), round(T4.LIMIT_MS - 19.629, 6))
        self.assertAlmostEqual(T4.diff_tol(None), round(T4.LIMIT_MS - T4.BASE_REF_DEFAULT, 6))
        self.assertAlmostEqual(T4.diff_tol(19.682), round(T4.LIMIT_MS - 19.682, 6))
        self.assertEqual(T4.diff_tol(30.0), T4.DIFF_TOL_MIN, "容差必须有下限（防异常参照值）")

    def test_abs_ok(self):
        self.assertTrue(T4.abs_ok(19.989))
        self.assertFalse(T4.abs_ok(19.99))
        self.assertFalse(T4.abs_ok(None))

    def test_session_ref_ms(self):
        summary = {"arm_n2020035_ep5_int8": 20.067, "arm_n1700001_ep5_int8": 19.9}
        self.assertAlmostEqual(T4.session_ref_ms(summary, 2020035, 5), 20.067)
        self.assertIsNone(T4.session_ref_ms(summary, 999, 5), "基线臂不在板上 ⇒ None")
        self.assertIsNone(T4.session_ref_ms(summary, None, 5))

    def test_pick_ms(self):
        summary = {"arm_n1700001_ep5_int8": 19.9, "arm_n1700001_ep3_int8": 18.2}
        self.assertAlmostEqual(T4.pick_ms(summary, 1700001, 5), 19.9)
        self.assertIsNone(T4.pick_ms(summary, 1700001, 4))
        self.assertIsNone(T4.pick_ms(summary, 1700001, 9),
                          "同前缀多臂 ⇒ 歧义 ⇒ 不得瞎猜")
        self.assertAlmostEqual(T4.pick_ms({"arm_n1700002_ep5_int8": 18.7}, 1700002, 5), 18.7)

    def test_pool_median(self):
        self.assertAlmostEqual(T4.pool_median([3.0, 1.0, 2.0]), 2.0)
        self.assertAlmostEqual(T4.pool_median([4.0, 1.0, 3.0, 2.0]), 2.5)
        with self.assertRaises(ValueError):
            T4.pool_median([])

    def test_merge_session_builds_diff_pool(self):
        pools: dict = {}
        touched = T4.merge_session(
            pools,
            {"arm_n1700001_ep5_int8": 20.05, "arm_n2020035_ep5_int8": 20.067,
             "control_base_int8": 57.9, "not_an_arm": 1.0},
            ref_ms=20.067,
        )
        self.assertEqual(sorted(touched), ["1700001_ep5", "2020035_ep5"])
        self.assertNotIn("control_base_int8", touched, "控制臂/杂项键不得进样本池")
        self.assertAlmostEqual(pools["1700001_ep5|d"][0], round(20.05 - 20.067, 4))
        self.assertAlmostEqual(pools["2020035_ep5|d"][0], 0.0)

    def test_merge_session_without_ref_has_no_diff_pool(self):
        pools: dict = {}
        T4.merge_session(pools, {"arm_n1700001_ep5_int8": 20.05}, ref_ms=None)
        self.assertNotIn("1700001_ep5|d", pools, "无参照 ⇒ 不得造差值池")

    def test_stored_pass_prefers_recorded_verdict(self):
        self.assertTrue(T4.stored_pass({"ms": 21.4, "pass_t4": True}),
                        "台账已判 PASS（相对口径）⇒ 不得用绝对口径推翻")
        self.assertFalse(T4.stored_pass({"ms": 19.5, "pass_t4": False}))
        self.assertTrue(T4.stored_pass({"ms": 19.5}), "老记录无 pass_t4 ⇒ 退回绝对口径")
        self.assertFalse(T4.stored_pass({"ms": 20.5}), "老记录无 pass_t4 ⇒ 绝对口径 FAIL")


class TestConsumers(unittest.TestCase):
    """两个消费者：必须**只信**台账里已算好的判定，不得自己重算绝对口径。"""

    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        # `auto_promote_watch` 在 import 时会 chdir 到 /tmp/pycharm_project_10ae9e2e
        os.makedirs("/tmp/pycharm_project_10ae9e2e", exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def _base(self) -> dict:
        return {"nano": 0.72, "nano4": 0.66, "f1": 0.89, "t1": 0.89,
                "t2_fp": 3, "t3": 0.948}

    def _ledger(self, arm_state: dict) -> dict:
        return {"arms": {"1700001": arm_state}}

    def test_auto_promote_honours_recorded_pass(self):
        apw = importlib.import_module("manu.pipeline.npu.auto_promote_watch")
        with tempfile.TemporaryDirectory() as td:
            measured = os.path.join(td, "measured.json")
            with open(measured, "w", encoding="utf-8") as fh:
                json.dump({
                    "1700001_ep5": {"ms": 21.4, "pass_t4": True, "pass_t4_rel": True},
                    "1700002_ep5": {"ms": 19.5},
                }, fh)
            old = apw.MEASURED
            apw.MEASURED = measured
            try:
                out = apw.board_pass_map()
            finally:
                apw.MEASURED = old
        self.assertTrue(out[(1700001, 5)][1],
                        "同会话相对口径已判 PASS ⇒ 升轨守望必须认（否则升轨链断链）")
        self.assertTrue(out[(1700002, 5)][1], "老记录无 pass_t4 ⇒ 退回绝对口径（19.5 ≤ 19.989）")

    def test_auto_promote_still_fails_slow_arm(self):
        apw = importlib.import_module("manu.pipeline.npu.auto_promote_watch")
        with tempfile.TemporaryDirectory() as td:
            measured = os.path.join(td, "measured.json")
            with open(measured, "w", encoding="utf-8") as fh:
                json.dump({"1700003_ep5": {"ms": 21.4, "pass_t4": False}}, fh)
            old = apw.MEASURED
            apw.MEASURED = measured
            try:
                out = apw.board_pass_map()
            finally:
                apw.MEASURED = old
        self.assertFalse(out[(1700003, 5)][1], "台账判 FAIL ⇒ 升轨守望必须 FAIL")

    def test_promote_candidate_honours_board_pass_t4(self):
        pc = importlib.import_module("manu.pipeline.npu.promote_candidate")
        st = {"picked": {"nano": 0.73, "nano4": 0.67, "f1": 0.90, "epoch": 5},
              "official": {"t1": 0.90, "t2_fp": 3, "t3": 0.95},
              "board_ms": 21.4, "board_pass_t4": True,
              "board_ms_ref": 20.067, "board_diff_ms": -0.017}
        res = pc.check_arm(self._ledger(st), self._base(), 1700001)
        self.assertTrue(res["checks"].get("ms"),
                        "台账 board_pass_t4=True ⇒ 预检必须认（热漂移下的同图臂）")
        self.assertEqual(res["board_ms_ref"], 20.067)
        self.assertEqual(res["board_diff_ms"], -0.017)

    def test_promote_candidate_rejects_failed_board(self):
        pc = importlib.import_module("manu.pipeline.npu.promote_candidate")
        st = {"picked": {"nano": 0.73, "nano4": 0.67, "f1": 0.90, "epoch": 5},
              "official": {"t1": 0.90, "t2_fp": 3, "t3": 0.95},
              "board_ms": 21.4, "board_pass_t4": False}
        res = pc.check_arm(self._ledger(st), self._base(), 1700001)
        self.assertFalse(res["checks"].get("ms"), "台账 board_pass_t4=False ⇒ 预检必须拒")

    def test_promote_candidate_rejects_stale_reference(self):
        """🔴 R3：板端判定打的是**旧基线** ⇒ 判定过期 ⇒ 保守判否（等重测）。"""
        pc = importlib.import_module("manu.pipeline.npu.promote_candidate")
        st = {"picked": {"nano": 0.73, "nano4": 0.67, "f1": 0.90, "epoch": 5},
              "official": {"t1": 0.90, "t2_fp": 3, "t3": 0.95},
              "board_ms": 21.4, "board_pass_t4": True,
              "board_ref_trial": "2020035", "board_ref_ep": "5"}
        led = self._ledger(st)
        led["baseline_v2"] = {"trial": 1990055, "epoch": 5, "ms_new_board": 19.682}
        res = pc.check_arm(led, self._base(), 1700001)
        self.assertFalse(res["checks"].get("ms"),
                         "参照基线已换人 ⇒ 旧判定过期 ⇒ 必须判否（不能直接放行）")
        self.assertTrue(res.get("board_stale"))
        self.assertTrue(any("旧基线" in w for w in res["warnings"]))

    def test_promote_candidate_accepts_current_reference(self):
        """参照基线与当前基线一致 ⇒ 认台账判定。"""
        pc = importlib.import_module("manu.pipeline.npu.promote_candidate")
        st = {"picked": {"nano": 0.73, "nano4": 0.67, "f1": 0.90, "epoch": 5},
              "official": {"t1": 0.90, "t2_fp": 3, "t3": 0.95},
              "board_ms": 21.4, "board_pass_t4": True,
              "board_ref_trial": "1990055", "board_ref_ep": "5"}
        led = self._ledger(st)
        led["baseline_v2"] = {"trial": 1990055, "epoch": 5, "ms_new_board": 19.682}
        res = pc.check_arm(led, self._base(), 1700001)
        self.assertTrue(res["checks"].get("ms"))
        self.assertFalse(res.get("board_stale"))

    def test_promote_candidate_legacy_arm_uses_absolute_redline(self):
        """🔴 R5：老臂（只有 `board_ms`）⇒ 用**绝对红线**，不用历史 t31（23.528 ms × 1.02）。"""
        pc = importlib.import_module("manu.pipeline.npu.promote_candidate")
        st = {"picked": {"nano": 0.73, "nano4": 0.67, "f1": 0.90, "epoch": 5},
              "official": {"t1": 0.90, "t2_fp": 3, "t3": 0.95},
              "board_ms": 21.4}
        led = self._ledger(st)
        led["board_sessions"] = [{"session2": {"arms": {"t31_silu_int8": 23.528}}}]
        res = pc.check_arm(led, self._base(), 1700001)
        self.assertFalse(res["checks"].get("ms"),
                         "21.4 ms 的老臂不得因为「别的板上的 t31=23.528」而被判 PASS")
        self.assertEqual(res.get("board_verdict_source"), "legacy-absolute")

    def test_promote_candidate_without_board_reading_warns(self):
        pc = importlib.import_module("manu.pipeline.npu.promote_candidate")
        st = {"picked": {"nano": 0.73, "nano4": 0.67, "f1": 0.90, "epoch": 5},
              "official": {"t1": 0.90, "t2_fp": 3, "t3": 0.95},
              "board_ms": None, "board_pass_t4": True}
        res = pc.check_arm(self._ledger(st), self._base(), 1700001)
        self.assertFalse(res["checks"].get("ms"), "无读数却判 PASS ⇒ 必须保守拒绝")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestProducer(unittest.TestCase):
    """**生产者**（`board_t4_watch`）：历史上「相对口径被绝对口径刷掉」的事故点。

    2026-11-19 15:27 现场事故：`sync_measured_to_ledger()` 用**纯绝对口径**把 116 个臂的
    `board_pass_t4=True` 覆盖回 False ⇒ 升轨链断在最后一环。
    ⇒ 这组用例专门锁死「**综合判定不被绝对口径刷掉**」。
    """

    @classmethod
    def setUpClass(cls):
        cls._cwd = os.getcwd()
        os.makedirs("/tmp/pycharm_project_10ae9e2e", exist_ok=True)
        cls.btw = importlib.import_module("manu.pipeline.npu.board_t4_watch")

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.btw.OUT = self.td.name
        self.btw.DONE = os.path.join(self.td.name, "measured.json")
        self.btw.SAMPLES = os.path.join(self.td.name, "t4_samples.json")
        self._saved: list = []
        self._led = {"arms": {"1700001": {"picked": {"epoch": 5}}}}
        self.btw.ns.load_ledger = lambda: self._led          # type: ignore[assignment]
        self.btw.ns.save_ledger = lambda led: self._saved.append(led)  # type: ignore[assignment]
        self.addCleanup(setattr, self.btw, "ns", self.btw.ns)

    def test_sync_does_not_clobber_relative_pass(self):
        """🔴 事故回归：`measured.json` 里已判 PASS 的臂，sync 后**必须仍是 PASS**。"""
        with open(self.btw.DONE, "w", encoding="utf-8") as fh:
            json.dump({"1700001_ep5": {"ms": 21.4, "pass_t4": True,
                                       "pass_t4_abs": False, "pass_t4_rel": True}}, fh)
        self.btw.sync_measured_to_ledger()
        self.assertTrue(self._saved, "应当写回台账")
        st = self._saved[-1]["arms"]["1700001"]
        self.assertTrue(st.get("board_pass_t4"),
                        "sync 不得用绝对口径刷掉相对口径的 PASS（2026-11-19 15:27 事故）")

    def test_sync_still_marks_slow_arm_failed(self):
        with open(self.btw.DONE, "w", encoding="utf-8") as fh:
            json.dump({"1700001_ep5": {"ms": 21.4, "pass_t4": False}}, fh)
        self.btw.sync_measured_to_ledger()
        st = self._saved[-1]["arms"]["1700001"]
        self.assertFalse(st.get("board_pass_t4"), "台账判 FAIL ⇒ sync 必须保持 FAIL")

    def test_sync_legacy_record_falls_back_to_absolute(self):
        """老记录没有 `pass_t4` ⇒ 退回绝对口径（19.5 ms ⇒ PASS）。"""
        with open(self.btw.DONE, "w", encoding="utf-8") as fh:
            json.dump({"1700001_ep5": {"ms": 19.5}}, fh)
        self.btw.sync_measured_to_ledger()
        st = self._saved[-1]["arms"]["1700001"]
        self.assertTrue(st.get("board_pass_t4"))

    def test_write_back_records_reference_identity(self):
        """台账必须记下**参照基线身份**（换基线后要能识别旧判定过期）。"""
        self.btw.write_back_ledger_batch(
            [(1700001, "5", 20.05, True, 20.067, -0.017, "1990055_ep5")])
        st = self._saved[-1]["arms"]["1700001"]
        self.assertEqual(st.get("board_ref_trial"), "1990055")
        self.assertEqual(st.get("board_ref_ep"), "5")
        self.assertTrue(st.get("board_pass_t4"))
        self.assertAlmostEqual(float(st.get("board_ms_ref")), 20.067)

    def test_base_ref_rejects_polluted_baseline_reading(self):
        """🔴 R1：**基线读数超红线**（说明它是靠相对口径过的漂移读数）⇒ 不得当干净参照。

        否则容差 = 19.989 − 20.05 ⇒ 夹到 0.05 ms ⇒ 相对口径退化成"抛硬币"。
        """
        self._led["baseline_v2"] = {"trial": 1990055, "epoch": 5, "ms_new_board": 20.05}
        self.assertAlmostEqual(self.btw.base_ref_ms(), T4.BASE_REF_DEFAULT)
        self.assertAlmostEqual(self.btw.diff_tol(), round(T4.LIMIT_MS - T4.BASE_REF_DEFAULT, 6))

    def test_base_ref_accepts_clean_baseline_reading(self):
        self._led["baseline_v2"] = {"trial": 1990055, "epoch": 5, "ms_new_board": 19.682}
        self.assertAlmostEqual(self.btw.base_ref_ms(), 19.682)
        self.assertAlmostEqual(self.btw.diff_tol(), round(T4.LIMIT_MS - 19.682, 6))

    def test_base_ref_prefers_explicit_clean_ref(self):
        self._led["baseline_v2"] = {"trial": 1990055, "epoch": 5,
                                    "ms_new_board": 20.05, "ms_clean_ref": 19.629}
        self.assertAlmostEqual(self.btw.base_ref_ms(), 19.629)

    def test_current_ref_id(self):
        self._led["baseline_v2"] = {"trial": 1990055, "epoch": 5}
        self.assertEqual(self.btw.current_ref_id(), "1990055_ep5")
        self._led.pop("baseline_v2")
        self.assertEqual(self.btw.current_ref_id(), "")


class TestRefScopedPools(unittest.TestCase):
    """🔴 R2：差值池必须**带参照身份**（换基线后不得混池）。"""

    def test_diff_pools_are_scoped_by_reference(self):
        pools: dict = {}
        T4.merge_session(pools, {"arm_n1700001_ep5_int8": 20.05}, 20.067, "1990055_ep5")
        T4.merge_session(pools, {"arm_n1700001_ep5_int8": 19.80}, 19.85, "2020035_ep5")
        self.assertIn("1700001_ep5|d@1990055_ep5", pools)
        self.assertIn("1700001_ep5|d@2020035_ep5", pools)
        self.assertAlmostEqual(pools["1700001_ep5|d@1990055_ep5"][0], -0.017)
        self.assertAlmostEqual(pools["1700001_ep5|d@2020035_ep5"][0], -0.05)
        self.assertNotIn("1700001_ep5|d", pools, "旧格式键不得再被写（否则会与新池混淆）")


class TestTrackB(unittest.TestCase):
    """🔴 R4：Track-B 升轨链也必须认「同会话相对」口径（它是第三个消费者）。"""

    def test_qualifies_accepts_relative_pass(self):
        tb = importlib.import_module("manu.pipeline.npu.trackb_core")
        base = {"nano": 0.72, "nano4": 0.66, "f1": 0.89, "t1": 0.89,
                "t2_fp": 3, "t3": 0.948}
        row = {"nano": 0.73, "nano4": 0.67, "f1": 0.90, "t1": 0.90,
               "t2_fp": 3, "t3": 0.95, "board_ms": 21.4, "board_pass": True}
        ok, miss, win = tb.qualifies(row, base, require_board=True)
        self.assertTrue(ok, "台账已判 PASS（相对口径）⇒ Track-B 不得再按绝对口径误杀")
        self.assertTrue(win)

    def test_qualifies_rejects_failed_board(self):
        tb = importlib.import_module("manu.pipeline.npu.trackb_core")
        base = {"nano": 0.72, "nano4": 0.66, "f1": 0.89, "t1": 0.89,
                "t2_fp": 3, "t3": 0.948}
        row = {"nano": 0.73, "nano4": 0.67, "f1": 0.90, "t1": 0.90,
               "t2_fp": 3, "t3": 0.95, "board_ms": 21.4, "board_pass": False}
        ok, _miss, _win = tb.qualifies(row, base, require_board=True)
        self.assertFalse(ok, "台账判 FAIL ⇒ 必须拒绝升轨")

    def test_qualifies_legacy_absolute(self):
        tb = importlib.import_module("manu.pipeline.npu.trackb_core")
        base = {"nano": 0.72, "nano4": 0.66, "f1": 0.89, "t1": 0.89,
                "t2_fp": 3, "t3": 0.948}
        row = {"nano": 0.73, "nano4": 0.67, "f1": 0.90, "t1": 0.90,
               "t2_fp": 3, "t3": 0.95, "board_ms": 19.5}
        ok, _miss, _win = tb.qualifies(row, base, require_board=True)
        self.assertTrue(ok, "老臂（无 board_pass）⇒ 退回绝对口径（19.5 ≤ 19.989）")


class TestScriptModeImports(unittest.TestCase):
    """🔴 部署形态回归：这些模块在服务器上**是以脚本方式**跑的
    （`python -u manu/pipeline/npu/xxx.py`）⇒ `sys.path[0]` 是**脚本目录**、
    **仓库根不在 sys.path**（cwd 不会自动加入）⇒ `from manu.pipeline.npu import ...`
    会 **ModuleNotFoundError: No module named 'manu'**。

    ⚠️ 本轮真的踩过：`beat_report.py` 加了 `from manu.pipeline.npu import t4_criterion`
    却没把仓库根加进 `sys.path` ⇒ **本地全绿、服务器一跑就炸**。
    ⇒ 用例锁死「每个消费判据的模块，在脚本形态下都能 import」。
    """

    MODULES = ["board_t4_watch.py", "auto_promote_watch.py", "promote_candidate.py",
               "board_t4_rejudge.py", "beat_report.py", "trackb_core.py"]

    def test_modules_import_in_script_mode(self):
        cwd = os.getcwd()
        script_dir = _NPU_DIR
        saved_path = list(sys.path)
        try:
            for mod in self.MODULES:
                with self.subTest(module=mod):
                    # 脚本形态：sys.path[0] = 脚本目录；其余路径去掉（只剩 stdlib/site-packages）
                    sys.path = [script_dir] + [p for p in saved_path
                                               if "site-packages" in p or "python3" in p]
                    os.chdir(_REPO_ROOT)
                    src = open(os.path.join(script_dir, mod), encoding="utf-8").read()
                    for marker in ("def main(", "if __name__ ==", "def cmd_promote(", "def run_once("):
                        i = src.find(marker)
                        if i > 0:
                            src = src[:i]
                            break
                    ns_ = {"__file__": os.path.join(script_dir, mod),
                           "__name__": "probe_" + mod.replace(".", "_")}
                    # ⚠️ 必须先**清掉已缓存的模块**，否则 `from manu.pipeline.npu import ...`
                    #   会直接命中 sys.modules ⇒ 路径问题被掩盖 ⇒ 变异测试测不出来。
                    popped = {}
                    for key in list(sys.modules):
                        base = key.rsplit(".", 1)[-1]
                        if (key == "manu" or key.startswith("manu.")
                                or base in ("nano_search", "adaptive_runner", "ckpt_synth",
                                            "family_key", "six_criteria_audit",
                                            "route2_acceptance", "board_speed_search",
                                            "t4_criterion", "trackb_core")):
                            popped[key] = sys.modules.pop(key)
                    try:
                        exec(compile(src, mod, "exec"), ns_)   # noqa: S102（故意执行 import 段）
                    finally:
                        sys.modules.update(popped)
        finally:
            sys.path = saved_path
            os.chdir(cwd)
