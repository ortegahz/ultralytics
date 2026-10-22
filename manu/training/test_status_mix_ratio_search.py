#!/usr/bin/env python
# ==============================================================================
# 混合比例搜索 —— **状态报告离线自测**
#
# 为什么需要这个自测
# --------------------
# `status_mix_ratio_search.py` 是**明早上班第一件事**要用的工具（出 Top-N、出分组最优、
# 出与 RC1 的正面对决结论）。但它读取的是 `optuna_summary.csv`，
# 而真实 CSV 要等第一个 trial 跑完 47 分钟才会出现。
# 一旦它的"有数据"分支有 bug（列名对不上、select_best 调用签名不对、决赛段崩），
# 明早第一件事就出不来结论。
#
# 本脚本造一份**格式与真实产物完全一致**的假 CSV + status.json，
# 逐项断言状态报告的每个段落都能正确产出。
#
# 用法：
#   python manu/training/test_status_mix_ratio_search.py
# 退出码 0 = 全部通过。
# ==============================================================================
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_SCRIPT = REPO_ROOT / "manu" / "training" / "status_mix_ratio_search.py"

# 与 optuna_mix_ratio_search.py 的 write_summary_row 表头逐字一致
HEADER = (
    "trial,status,mix_spec,mix_mode,target_ratio,measured_ratio,gate_verdict,"
    "t1_f1,t1_fp,t2_f1,t2_fp,t3_f1,t3_fp,utility,utility_vs_rc1,"
    "lr_scale,warmup_epochs,weight_decay,focal_beta,offset_weight,mosaic,scale,translate,"
    "clip_max_norm,freeze_bn,duration_min"
)

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


def make_row(trial, status, mix, target, measured, t1f1, t1fp, t2f1, t2fp, t3f1, t3fp,
             utility, u_rc1, dur=47.0):
    def _num(v):
        # 真实产物里，被硬杀的 trial（CRASHED）指标列是**空字符串**，不是 0
        return "" if v == "" else (f"{v:.6f}" if isinstance(v, float) else str(v))

    return [trial, status, mix, "weighted", f"{target:.3f}", f"{measured:.3f}", "UTILITY_A",
            _num(t1f1), _num(t1fp), _num(t2f1), _num(t2fp), _num(t3f1), _num(t3fp),
            utility, u_rc1,
            "1.0000", "0.500", "0.00010000", "2.70", "0.55", "0.05", "0.20", "0.10",
            "5.00", "False", f"{dur:.2f}"]


def build_fixture(root: Path) -> None:
    """造一份与真实格式一致的假产物。

    ⚠️ utility 列必须填**真实重算值**（相对冻结基线），否则 CSV 与判定逻辑不自洽，
    会把"数据不自洽"误读成"判定有 bug"。这里三个候选的真实 U 为：
      · candidate_better  U = +0.036950  （T3 大幅超越 ⇒ 应**打败** RC1）
      · candidate_worse   U = +0.028100  （与 RC1 接近）
      · pruned / crashed  U 为空
    RC1/Ep3 的 U = +0.027579（真实值）。
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "dispatcher.log").write_text("[LAUNCH] trial_0000 → GPU 0\n", encoding="utf-8")

    # ★ 每个 trial 的训练日志（**[SCHED] 行 = 调度口径、[GATE] 行 = 测量 epoch** 的 ground truth）
    #   新调度（cosine 周期 5 ep）
    NEW_SCHED = ("[SCHED] 训练 3 ep（5337 步）| warmup 890 步（0.5 ep）| "
                 "cosine 周期 5 ep（8895 步） → 1e-006  ← 与 RC1 同口径\n")
    #   旧调度（--sched-epochs 特性之前，日志里没有"cosine 周期"字样 ⇒ 周期=3 ep）
    OLD_SCHED = "[SCHED] warmup 890 步（0.5 ep）| cosine → 1e-06\n"
    #   门禁行：[GATE] 三轨门禁（mix / epochK）
    def gate_line(ep: int) -> str:
        return f"[GATE] 三轨门禁（w4.0 / epoch{ep}）\n"

    #   trial_0001 → 新调度 + 量 Ep3  ⇒ RC1-Ep3口径
    (root / "logs" / "trial_0001.log").write_text(NEW_SCHED + gate_line(3), encoding="utf-8")
    #   trial_0003 / trial_0005 → 新调度 + 量 Ep5 ⇒ RC1-Ep5口径（5-epoch 协议的主力组）
    (root / "logs" / "trial_0003.log").write_text(NEW_SCHED + gate_line(5), encoding="utf-8")
    (root / "logs" / "trial_0005.log").write_text(NEW_SCHED + gate_line(5), encoding="utf-8")
    #   trial_0002 → 旧调度 ⇒ 不可比
    (root / "logs" / "trial_0002.log").write_text(OLD_SCHED + gate_line(3), encoding="utf-8")

    rows = [
        # ① 真正优于 RC1：T1 不退(0.9060>0.9057)、T2 略升、T3 大幅超越(0.8740)
        #    U = (0.8740-0.8373308) + 0.5*(0.9890-0.9884360) - 0 - 0 = 0.036950
        make_row("trial_0003", "SUCCESS", "w4.0", 4.0, 4.106, 0.906000, 1000,
                 0.989000, 26, 0.874000, 540, "0.036950", "0.009371"),
        # ② 与 RC1 接近但更差：T3 略好、T2 差 ⇒ U 低于 RC1
        #    U = (0.8390-0.8373308) + 0.5*(0.9889-0.9884360) - 2.5*(0.9057362-0.9035) = 0.010024
        # ⚠️ utility 列故意写成一个**错误的**值（0.010024）：验证报告用的是**重算**的 U
        #    （重算真值 = -0.003689），而不是 CSV 里写了什么就算什么
        make_row("trial_0001", "SUCCESS", "w2.0", 2.0, 2.030, 0.903500, 960,
                 0.988900, 24, 0.839000, 570, "0.010024", "-0.017555"),
        # ③ 被红线剪枝（utility 为空）
        make_row("trial_0002", "PRUNED", "w3.0", 3.0, 3.020, 0.899000, 930,
                 0.989000, 23, 0.835000, 575, "", ""),
        # ④ 崩溃
        make_row("trial_0004", "CRASHED", "w6.0", 6.0, 0.0, "", "", "", "", "", "", "", ""),
        # ⑤ **只过放宽通道**（严格口径不过）：ΔFP_T2=+12(>10) / ΔF1_T1=-0.006036(<-0.005)
        #    但放宽通道要求 ΔF1_T2=+0.009064(≥+0.008)、ΔF1_T3=+0.021169(≥+0.020) 均满足
        #    ⇒ 严格 FAIL / 受控放宽 PASS_RELAXED
        make_row("trial_0005", "SUCCESS", "w5.0", 5.0, 5.010, 0.899700, 1000,
                 0.997500, 38, 0.858500, 570, "0.004611", "-0.022968"),
    ]
    with (root / "optuna_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER.split(","))
        for r in rows:
            w.writerow(r)

    (root / "status.json").write_text(json.dumps({
        "updated": __import__("time").time(),
        "elapsed_h": 1.57,
        "done": 4,
        "running": [{"trial": "trial_0005", "gpu": 0, "pid": 12345, "mix": "w5.0"}],
        "free_gpus": [1, 2, 3],
        "best_value": 0.03695,
        "best_trial": 3,
    }), encoding="utf-8")


def run_status(root: Path, extra: list[str] | None = None) -> tuple[int, str]:
    cmd = [sys.executable, str(STATUS_SCRIPT), "--out-root", str(root),
           "--tail-log", "0", "--out-json", str(root / "status_report.json")]
    if extra:
        cmd += extra
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="status_selftest_"))
    root = tmp / "runs" / "optuna_mix_ratio"
    build_fixture(root)

    print("=" * 90)
    print("状态报告离线自测")
    print("=" * 90)
    rc, out = run_status(root)
    print(out[-4000:])

    print("\n--- 断言 ---")
    check("退出码 0", rc == 0, f"实际 {rc}")
    check("[2] 进度计数正确（共 5 / 过线 3 / 剪枝 1 / 异常 1）",
          "共 5 个 trial | 过线 3 | 剪枝 1 | 异常 1" in out, "未找到预期进度行")
    check("[4] Top-N 含 trial_0003（U 最大者）",
          "trial_0003" in out and "trial_0003" in out.split("[4]")[1].split("[5]")[0],
          "Top-N 段未含 trial_0003")
    top_seg = out.split("[4]")[1].split("[5]")[0] if "[4]" in out else ""
    check("[4] Top-N 按 U 降序（trial_0003 在 trial_0001 之前）",
          top_seg.find("trial_0003") < top_seg.find("trial_0001"), "顺序不对")
    # ★ Top-N **不隐藏**被红线剪枝的 trial：它的 U 是真实存在的，只是没过一级红线。
    #   （与 [5] 分组表口径一致 —— 藏起来会让明早的选型表凭空少掉大半配置）
    check("[4] Top-N 含被剪枝的 trial_0002（不隐藏数据）",
          "trial_0002" in top_seg, "被剪枝项从 Top-N 消失了（信息被藏）")
    check("[4] Top-N 里 trial_0002 标注 ⛔触红线",
          "trial_0002" in top_seg and "⛔触红线" in top_seg.split("trial_0002")[-1].split("\n")[0],
          "trial_0002 行缺 ⛔触红线 标记")
    check("[4] Top-N 不含崩溃的 trial_0004（无指标）", "trial_0004" not in top_seg, "崩溃项泄漏到 Top-N")
    # ★ 双通道区分：只过放宽通道的 trial_0005 标 🟡过线(放宽)，严格过线的标 ✅过线
    check("[4] Top-N 里 trial_0005 标注 🟡过线(放宽)",
          "trial_0005" in top_seg and "🟡过线(放宽)" in top_seg.split("trial_0005")[-1].split("\n")[0],
          "trial_0005 行缺 🟡过线(放宽) 标记")
    check("[4] Top-N 里 trial_0003 标注 ✅过线（严格通道）",
          "trial_0003" in top_seg and "✅过线" in top_seg.split("trial_0003")[-1].split("\n")[0],
          "trial_0003 行缺 ✅过线 标记")
    check("[4] 表中含【冻结基线】参照行", "【冻结基线】" in top_seg, "缺参照系行")
    check("[4] 表中含【RC1=Ep3】参照行", "【RC1=Ep3】" in top_seg, "缺参照系行")
    check("[4] 参照系 U 值正确（RC1=+0.027579）", "+0.027579" in top_seg, "RC1 的 U 不对")
    check("[5] 按 mix_spec 分组出现 w4.0 与 w2.0", "w4.0" in out and "w2.0" in out, "分组缺失")
    # ★ 分组表必须**包含被红线剪枝的档位**（否则 3/4 的配置在明早的选型表里直接消失）
    seg5 = out.split("[5]")[1].split("[6]")[0] if "[5]" in out else ""
    check("[5] 分组表包含被剪枝的 trial_0002 (w3.0, PRUNED)", "trial_0002" in seg5, f"分组段: {seg5[:300]}")
    check("[5] 分组表排除无指标的 trial_0004 (CRASHED)", "trial_0004" not in seg5, f"分组段: {seg5[:300]}")
    # ★ 调度口径分栏：主表 = 与 RC1 同 cosine 周期（U 可直接比）；附表 = 旧口径（不可比）
    main_seg = seg5.split("旧调度口径")[0] if "旧调度口径" in seg5 else seg5
    old_seg = seg5.split("旧调度口径")[1] if "旧调度口径" in seg5 else ""
    check("[5] 主表只含与 RC1 同调度口径的 trial（trial_0002 属旧口径）",
          "trial_0002" not in main_seg, f"主表段: {main_seg[:400]}")
    check("[5] 附表含旧调度口径的 trial_0002", "trial_0002" in old_seg, f"附表段: {old_seg[:300]}")
    # ★ 二维口径：cosine 周期 × **测量 epoch** ⇒ 5-epoch trial 的擂主必须是 RC1/Ep5
    check("[5] 出现 RC1-Ep3口径 分组", "RC1-Ep3口径" in seg5, f"分组段: {seg5[:400]}")
    check("[5] 出现 RC1-Ep5口径 分组", "RC1-Ep5口径" in seg5, f"分组段: {seg5[:400]}")
    check("[5] Ep5 分组用 RC1/Ep5 作擂主（U=+0.026260）", "+0.026260" in seg5, f"分组段: {seg5[:400]}")
    check("[5] Ep3 分组用 RC1/Ep3 作擂主（U=+0.027579）", "+0.027579" in seg5, f"分组段: {seg5[:400]}")
    check("[5] 旧口径分组明说无参照系", "无同口径参照系" in seg5, f"分组段: {seg5[:400]}")
    # ★ 统计量必须是**中位数**（读结果铁律：单 trial 不能否定一个比例）
    check("[5] 分组表给 U 中位数", "U中位数=" in seg5, f"分组段: {seg5[:300]}")
    # ★ 分组表的 U 必须**重算**（CSV 里 PRUNED 行的 utility 列是 "None"，直接用会得 0.0 误导值）
    #    trial_0002 (w3.0, PRUNED) 的重算 U = -0.018889
    check("[5] 分组表 U 用重算值（PRUNED 行非 0/None）",
          "-0.018889" in seg5, f"分组段: {seg5[:400]}")
    check("[5] 分组表标注触红线/过线", "⛔触红线" in seg5 and "✅过线" in seg5, f"分组段: {seg5[:400]}")
    check("[5] 分组表区分受控放宽（w5.0 trial_0005 标 🟡过线(放宽)）",
          "🟡过线(放宽)" in seg5, f"分组段: {seg5[:400]}")
    check("[5] 分组表用重算 U（trial_0001 的重算真值 -0.003689）",
          "-0.003689" in seg5, f"分组段: {seg5[:400]}")
    check("[5] 分组表不采信 CSV 里被写错的 utility 列（0.010024）",
          "+0.010024" not in seg5 and " 0.010024" not in seg5, f"分组段: {seg5[:400]}")
    check("[5] 给出 T2 噪声警示", "噪声" in seg5, "缺噪声提示")
    check("[6] 决赛段产出结论", "[6]" in out and ("最优" in out or "尚无过线" in out), "无决赛段")
    # ★ 最关键的一条：真正优于 RC1 的候选必须被判为最优且"打败现任擂主"
    seg6 = out.split("[6]")[1].split("[7]")[0] if "[6]" in out else ""
    # 决赛按**口径分组**，每个分组各有自己的擂主 ⇒ 可能有多个 🏆 段
    check("[6] 决赛最优 = trial_0003（U 最高者）",
          any("trial_0003" in blk.split("理由")[0] for blk in seg6.split("🏆")[1:]),
          f"决赛段: {seg6[:400]}")
    check("[6] 结论为『✅ 打败 RC1/Ep3』（现行擂主）",
          "✅ 打败 RC1/Ep3" in seg6, f"决赛段: {seg6[:400]}")
    check("[6] Ep5 组额外报 vs 现任擂主 RC1/Ep3",
          "vs 现任擂主 RC1/Ep3" in seg6, f"决赛段: {seg6[:400]}")
    check("[6] vs RC1 判决为 UTILITY_A",
          "UTILITY_A" in seg6, f"决赛段: {seg6[:400]}")
    # ★ 决赛候选池默认**按调度口径过滤**（旧口径 trial 的 U 系统性偏低，混进来会污染结论）
    check("[6] 默认按口径排除旧调度 trial（提示已排除）",
          "已按口径排除" in seg6 and "trial_0002" in seg6, f"决赛段: {seg6[:400]}")
    check("[6] 决赛候选池不含旧口径 trial_0002",
          "trial_0002" not in seg6.split("已按口径排除")[0], f"决赛段: {seg6[:400]}")
    check("[7] 平均耗时段出现（47 分钟）", "[7]" in out and "47.0" in out, "无耗时段")
    check("[JSON] 已写出 status_report.json", (root / "status_report.json").is_file())

    if (root / "status_report.json").is_file():
        rep = json.loads((root / "status_report.json").read_text(encoding="utf-8"))
        check("JSON.total_trials = 5", rep.get("total_trials") == 5, str(rep.get("total_trials")))
        check("JSON.success = 3", rep.get("success") == 3, str(rep.get("success")))
        check("JSON.pruned = 1", rep.get("pruned") == 1, str(rep.get("pruned")))
        check("JSON.crashed = 1", rep.get("crashed") == 1, str(rep.get("crashed")))
        check("JSON.group_stats 含 w4.0", "w4.0" in (rep.get("group_stats") or {}),
              str(list((rep.get("group_stats") or {}).keys())))
        check("JSON.group_stats 含 U 中位数", "u_median" in (rep.get("group_stats") or {}).get("w4.0", {}),
              str((rep.get("group_stats") or {}).get("w4.0")))
        check("JSON.calibers 标出旧口径 trial_0002",
              (rep.get("calibers") or {}).get("trial_0002", {}).get("comparable") is False,
              str((rep.get("calibers") or {}).get("trial_0002")))

    # ---- --all-calibers：放开口径过滤，旧口径 trial 回到候选池 ----
    print("\n--- 放开口径过滤（--all-calibers）---")
    rc4, out4 = run_status(root, extra=["--all-calibers"])
    seg6b = out4.split("[6]")[1].split("[7]")[0] if "[6]" in out4 else ""
    check("all-calibers 退出码 0", rc4 == 0, f"实际 {rc4}")
    check("all-calibers 下 trial_0002 回到候选池",
          "trial_0002" in seg6b and "已按口径排除" not in seg6b, f"决赛段: {seg6b[:400]}")
    check("all-calibers 标注了放开口径", "全部口径" in seg6b, f"决赛段: {seg6b[:300]}")

    # ---- --strict-only 回退：只过放宽通道的 trial_0005 必须退回 ⛔触红线 ----
    print("\n--- 严格口径回退（--strict-only）---")
    rc3, out3 = run_status(root, extra=["--strict-only"])
    top3 = out3.split("[4]")[1].split("[5]")[0] if "[4]" in out3 else ""
    check("strict-only 退出码 0", rc3 == 0, f"实际 {rc3}")
    check("strict-only 下 trial_0005 不再标放宽",
          "trial_0005" in top3 and "🟡过线(放宽)" not in top3.split("trial_0005")[-1].split("\n")[0],
          "strict-only 仍标了放宽")
    check("strict-only 下 trial_0005 标 ⛔触红线",
          "trial_0005" in top3 and "⛔触红线" in top3.split("trial_0005")[-1].split("\n")[0],
          "strict-only 未标触红线")
    check("strict-only 给出严格口径提示", "严格口径" in out3, "缺严格口径提示")

    # ---- 空目录（还没任何 trial）不应崩 ----
    print("\n--- 空状态（搜索刚启动）---")
    empty = tmp / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    rc2, out2 = run_status(empty)
    check("空状态退出码 0（不崩）", rc2 == 0, f"实际 {rc2}")
    check("空状态提示未启动", "未启动" in out2 or "尚无过线" in out2, out2[:300])

    print("\n" + "=" * 90)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    if FAILURES:
        print(f"❌ 自测失败：{len(FAILURES)}/{CHECKS} 项不通过")
        for f in FAILURES:
            print(f"   - {f}")
        return 1
    print(f"✅ 状态报告自测全部通过：{CHECKS}/{CHECKS}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
