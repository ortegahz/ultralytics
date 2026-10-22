#!/usr/bin/env python
# ==============================================================================
# Optuna 混合比例搜索 —— **调度器离线自测**
#
# 为什么需要这个自测
# --------------------
# `optuna_mix_ratio_search.py` 的 dispatcher 要在服务器上**无人值守跑 20 小时**。
# 一旦它挂在某行上（例如 argparse 属性名写错），整个 20 小时会静默空跑，
# 早上才发现就来不及了。而真实 trial 要 45 分钟才跑完一个，迭代太慢。
#
# 本脚本用**假 worker**（monkeypatch subprocess.Popen）把整条调度链路
# 在几秒内跑完，逐项断言：
#   1. 4 卡并行派发，GPU 分配不重复、不漏
#   2. 种子 trial 真的生效（头 4 个 trial = natural / w2.0 / w3.0 / w4.0）
#   3. 收割分类正确：SUCCESS / PRUNED（一级红线失败）/ CRASHED（无结果文件）
#   4. study.db 里 trial state 与分类一致（PRUNED 不参与 TPE 采样）
#   5. optuna_summary.csv 逐行正确
#   6. workers.json / status.json / dispatcher.pid 产物齐全
#   7. best_value 只取自 COMPLETE（PRUNED 不算）
#   8. 陈旧 RUNNING trial 在重启时被自愈为 FAIL
#
# 用法：
#   python manu/training/test_optuna_dispatcher.py
# 退出码 0 = 全部通过；非 0 = 有断言失败。
# ==============================================================================
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

import manu.training.optuna_mix_ratio_search as mod  # noqa: E402


# ==============================================================================
# 假 worker
# ==============================================================================
# 每个 trial_number 对应一种结局：
#   "success" → tier1_ok=True，带 utility ⇒ study.tell(score) ⇒ COMPLETE
#   "pruned"  → tier1_ok=False ⇒ study.tell(PRUNED)
#   "infra"   → 有 trial_result.json 但门禁报告不完整（infrastructure_failure=True）
#                ⇒ 必须记 **FAIL**（不是 PRUNED！两者语义不同：PRUNED=触了一级红线，
#                FAIL=根本没测成，不该进 TPE 的"已探索"统计）
#   "crash"   → 不写 trial_result.json ⇒ 收割时判 CRASHED ⇒ FAIL
SCENARIO = {
    0: "success",
    1: "pruned",
    2: "infra",
    3: "success",
    4: "crash",
}
FAKE_UTILITY = {0: 0.031234, 3: 0.052101}


class FakePopen:
    """模拟 subprocess.Popen：立即写结果文件并"退出"。"""

    _counter = 0
    _launched = []  # 记录每次派发的 (trial_number, gpu_id)

    def __init__(self, cmd, stdout=None, stderr=None, cwd=None, **kwargs):
        self.cmd = cmd
        self.pid = 900000 + FakePopen._counter
        FakePopen._counter += 1
        self._rc = 0

        args = cmd[cmd.index("--trial-number") + 1:]
        trial_number = int(args[0])
        gpu_id = int(args[args.index("--gpu-id") + 1])
        out_root = Path(args[args.index("--output-root") + 1])
        FakePopen._launched.append((trial_number, gpu_id))

        kind = SCENARIO.get(trial_number, "success")
        if kind == "crash":
            return  # 不写结果文件 ⇒ 模拟崩溃（worker 进程被杀）

        trial_dir = out_root / "trials" / f"trial_{trial_number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        ok = kind == "success"
        if kind == "infra":
            # 模拟"门禁基础设施失败"（如 Track1 CUDA OOM ⇒ 报告缺字段）
            res = {
                "trial": trial_number,
                "trial_name": f"trial_{trial_number:04d}",
                "mix_spec": "w4.0",
                "measured_ratio": 4.106,
                "gate_rc": 1,
                "gate_verdict": "GATE_REPORT_INCOMPLETE",
                "tier1_ok": False,
                "tier1_reason": "GATE_REPORT_INCOMPLETE",
                "infrastructure_failure": True,
                "gate_error": "GateReportIncomplete: gate_report 缺少必需字段 ['track1_f1']",
                "metrics": None,
                "utility": None,
            }
            (trial_dir / "trial_result.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return
        res = {
            "trial": trial_number,
            "trial_name": f"trial_{trial_number:04d}",
            "mix_spec": "w4.0",
            "measured_ratio": 4.106,
            "gate_rc": 0 if ok else 1,
            "gate_verdict": "UTILITY_A" if ok else "FAIL_TRACK1_RETREAT",
            "tier1_ok": ok,
            "tier1_reason": "PASS" if ok else "FAIL_TRACK1_RETREAT",
            "utility": FAKE_UTILITY.get(trial_number) if ok else None,
            "utility_vs_rc1": 0.003 if ok else -0.01,
            "metrics": {
                "track1_f1": 0.906, "track1_fp": 1000,
                "track2_f1": 0.989, "track2_fp": 20,
                "track3_f1": 0.84, "track3_fp": 580,
            },
        }
        (trial_dir / "trial_result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return self._rc

    def kill(self):
        self._rc = -9

    def terminate(self):
        self._rc = -15


# ==============================================================================
# 断言工具
# ==============================================================================
FAILURES: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}  {detail}")
        FAILURES.append(f"{name} {detail}")


# ==============================================================================
# 主测试
# ==============================================================================
def run_dispatcher(out_root: Path, n_trials: int = 4, max_hours: float = 0.05,
                   study_name: str = "selftest") -> None:
    """跑一次 dispatcher（假 worker），返回 study。"""
    argv = sys.argv
    sys.argv = [
        "optuna_mix_ratio_search.py",
        "--output-root", str(out_root),
        "--study-name", study_name,
        "--gpus", "0,1,2,3",
        "--n-trials", str(n_trials),
        "--epochs", "3",
        "--max-hours", str(max_hours),
    ]
    try:
        mod.main()
    finally:
        sys.argv = argv


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dispatcher_selftest_"))
    out_root = tmp / "runs" / "optuna_mix_ratio"
    original_popen = mod.subprocess.Popen
    FakePopen._launched = []
    mod.subprocess.Popen = FakePopen  # type: ignore[assignment]
    try:
        print("=" * 90)
        print("调度器离线自测（假 worker）")
        print("=" * 90)
        run_dispatcher(out_root)
    finally:
        mod.subprocess.Popen = original_popen  # type: ignore[assignment]

    db = out_root / "study.db"
    study = optuna.load_study(
        study_name="selftest",
        storage=f"sqlite:///{db.resolve()}",
    )
    trials = study.get_trials(deepcopy=False)

    print("\n--- 1. 派发与 GPU 分配 ---")
    check("派发了 4 个 trial", len(FakePopen._launched) == 4, f"实际 {len(FakePopen._launched)}")
    gpus = sorted(g for _, g in FakePopen._launched)
    check("GPU 0/1/2/3 各一个（不重复不漏）", gpus == [0, 1, 2, 3], f"实际 {gpus}")

    print("\n--- 2. 种子 trial 生效 ---")
    # enqueue 顺序：natural / w2.0 / w3.0 / w4.0 / w6.0，前 4 个应按此顺序出
    specs = [t.params.get("mix_spec") for t in trials[:4]]
    check("头 4 个 trial = natural/w2.0/w3.0/w4.0",
          specs == ["natural", "w2.0", "w3.0", "w4.0"], f"实际 {specs}")
    # 第 5 个种子 w6.0 还在 enqueue 队列里 → 在 study.trials 中表现为 **WAITING** 状态
    waiting = [t.number for t in trials if t.state == optuna.trial.TrialState.WAITING]
    check("第 5 个种子 w6.0 仍在队列（WAITING，尚未被 ask）",
          len(waiting) == 1 and waiting[0] == 4, f"实际 WAITING={waiting}")

    print("\n--- 3. 收割分类与 study state ---")
    by_num = {t.number: t for t in trials}
    check("trial_0000 → COMPLETE（SUCCESS）",
          by_num[0].state == optuna.trial.TrialState.COMPLETE, str(by_num[0].state))
    check("trial_0001 → PRUNED（一级红线失败）",
          by_num[1].state == optuna.trial.TrialState.PRUNED, str(by_num[1].state))
    check("trial_0002 → FAIL（门禁基础设施失败，**不是** PRUNED）",
          by_num[2].state == optuna.trial.TrialState.FAIL, str(by_num[2].state))
    check("trial_0002 → 不是 PRUNED（基础设施失败与触红线必须区分）",
          by_num[2].state != optuna.trial.TrialState.PRUNED, str(by_num[2].state))
    check("trial_0003 → COMPLETE（SUCCESS）",
          by_num[3].state == optuna.trial.TrialState.COMPLETE, str(by_num[3].state))

    print("\n--- 4. best_value 只取自 COMPLETE ---")
    best = study.best_value
    check(f"best_value = {best}（应为两个 SUCCESS 中较大者 0.052101）",
          abs(best - 0.052101) < 1e-9, f"实际 {best}")
    check("best_trial = 3", study.best_trial.number == 3, f"实际 {study.best_trial.number}")

    print("\n--- 5. 汇总 CSV ---")
    csv_path = out_root / "optuna_summary.csv"
    check("optuna_summary.csv 已生成", csv_path.is_file())
    if csv_path.is_file():
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        check(f"CSV 行数 = 1 表头 + 4 数据行（实际 {len(lines)}）", len(lines) == 5, f"实际 {len(lines)}")
        check("含 SUCCESS 记录", any(",SUCCESS," in ln for ln in lines), str(lines))
        check("含 PRUNED 记录", any(",PRUNED," in ln for ln in lines), str(lines))
        check("含 CRASHED 记录", any(",CRASHED," in ln for ln in lines), str(lines))

    print("\n--- 6. 状态产物 ---")
    for name in ("dispatcher.pid", "workers.json", "status.json"):
        p = out_root / name
        check(f"{name} 已生成", p.is_file())
    if (out_root / "status.json").is_file():
        st = json.loads((out_root / "status.json").read_text(encoding="utf-8"))
        check("status.json.done = 4", st.get("done") == 4, f"实际 {st.get('done')}")
        check("status.json.best_value 非空", st.get("best_value") is not None, str(st.get("best_value")))
    if (out_root / "dispatcher.pid").is_file():
        pid_txt = (out_root / "dispatcher.pid").read_text(encoding="utf-8").strip()
        check("dispatcher.pid 是数字", pid_txt.isdigit(), f"实际 {pid_txt!r}")

    print("\n--- 7. 陈旧 RUNNING trial 自愈（断点续跑核心路径）---")
    # 真实场景：进程被断电/杀死后，db 里留下 RUNNING trial。
    # 用 ask() 造一个**不 tell** 的 trial —— 它天然处于 RUNNING 状态。
    # ⚠️ 绝不能手动 set_trial_state_values(t.number, ...)：
    #    Optuna RDB 的 trial_id 是 1 起自增主键，Trial.number 是 0 起业务编号，两者差 1；
    #    set_trial_state_values 收的是前者，传 number 会改错行并抛 UpdateFinishedTrialError
    #    ⇒ 断点续跑时 dispatcher 直接崩溃（该 bug 已由本自测定位并修复）。
    st2 = optuna.load_study(study_name="selftest", storage=f"sqlite:///{db.resolve()}")
    stale = st2.ask()          # 保持 RUNNING，不 tell
    running_before = [
        t.number for t in st2.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.RUNNING
    ]
    check("注入后确实存在 RUNNING trial", len(running_before) >= 1, f"实际 {running_before}")

    # 再跑一次 dispatcher（复用同一个 study.db）⇒ 启动时应自愈
    FakePopen._launched = []
    original_popen = mod.subprocess.Popen
    mod.subprocess.Popen = FakePopen  # type: ignore[assignment]
    try:
        run_dispatcher(out_root, n_trials=5, study_name="selftest")
    finally:
        mod.subprocess.Popen = original_popen  # type: ignore[assignment]

    st3 = optuna.load_study(study_name="selftest", storage=f"sqlite:///{db.resolve()}")
    stale_state = [t.state.name for t in st3.get_trials(deepcopy=False) if t.number == stale.number]
    check("陈旧 RUNNING trial 已被自愈（不再是 RUNNING）",
          bool(stale_state) and stale_state[0] != "RUNNING",
          f"实际 {stale_state}")

    print("\n--- 8. 断点续跑：已完成 trial 不被重跑 ---")
    # 第一次跑的编号集合 vs 第二次派发的编号集合 ⇒ 必须无交集
    first_round = set(range(4))  # trial_0000 ~ trial_0003
    second_round = {n for n, _ in FakePopen._launched}
    overlap = first_round & second_round
    check(f"第二次运行没有重跑第一次已完成的 trial（本次派发 {sorted(second_round)}）",
          not overlap, f"重跑了 {sorted(overlap)}")
    check("第二次派发的编号在第一次之后（编号单调递增）",
          all(n >= 4 for n in second_round), f"实际 {sorted(second_round)}")

    print("\n" + "=" * 90)
    shutil.rmtree(tmp, ignore_errors=True)
    if FAILURES:
        print(f"❌ 自测失败：{len(FAILURES)}/{CHECKS} 项不通过")
        for f in FAILURES:
            print(f"   - {f}")
        return 1
    print(f"✅ 调度器自测全部通过：{CHECKS}/{CHECKS}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
