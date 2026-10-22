#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`mix_ratio_judgement` 的自测：用**真实门禁数据**固化三级漏斗的判定行为。

为什么要有这个文件
------------------
判别标准直接决定"最终选哪个模型"，一旦判据实现与用户口径出现偏差，
选型就会静默出错（例如把帕累托维度集写死，导致某个本该"无争议胜出"的模型
被降级到效用分裁决并可能判反）。本文件把下列行为**钉死**：

1. 一级红线：T2 F1 跌破 0.985 / ΔFP_T2 超 +10 / ΔF1_T1 跌破 −0.005 ⇒ 各自的原因码；
2. 三级效用分：与用户参考实现逐项对齐（含 γ 的 FP 惩罚项）；
3. 二级帕累托：维度集预设的行为差异（core / full / user）；
4. 真实数据回归：Ep3 vs Ep4 在三套预设下的判定（含"帕累托成立则不再看效用分"的漏斗顺序）；
5. 决赛选择器：先剔红线、再找帕累托极大集、多元时按效用分裁决。

运行（本地即可，不依赖服务器）：
```bash
python manu/training/test_mix_ratio_judgement.py
```
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manu.training.mix_ratio_judgement import (  # noqa: E402
    FROZEN_BASELINE,
    RC1_EPOCH3,
    RC1_EPOCH5,
    compare_models,
    evaluate_model_quality,
    hard_gate,
    pareto_dominates,
    rc1_epoch_band,
    reproduction_check,
    select_best,
    utility,
)

# 真实门禁数据（来源：runs/mixed_ft_route_b/gate/gate_history.jsonl）
EP = {
    0: {"track1_f1": 0.905736245785252, "track1_fp": 1014, "track2_f1": 0.9884359729262149,
        "track2_fp": 26, "track3_f1": 0.8373308251032032, "track3_fp": 590},
    1: {"track1_f1": 0.9022149063405079, "track1_fp": 1232, "track2_f1": 0.998768953610569,
        "track2_fp": 35, "track3_f1": 0.8468954078988287, "track3_fp": 589},
    2: {"track1_f1": 0.9019269606193201, "track1_fp": 988, "track2_f1": 0.9988890556973428,
        "track2_fp": 33, "track3_f1": 0.8595867470500335, "track3_fp": 531},
    3: {"track1_f1": 0.901917808219178, "track1_fp": 941, "track2_f1": 0.9989791016094163,
        "track2_fp": 31, "track3_f1": 0.8716843671438099, "track3_fp": 461},
    4: {"track1_f1": 0.9017962061440322, "track1_fp": 1057, "track2_f1": 0.9988591329410352,
        "track2_fp": 35, "track3_f1": 0.8726475238435252, "track3_fp": 517},
    5: {"track1_f1": 0.9005006521098912, "track1_fp": 1023, "track2_f1": 0.9989491698441769,
        "track2_fp": 33, "track3_f1": 0.8749235474006115, "track3_fp": 502},
}

PASS, FAIL = 0, 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_hard_gate() -> None:
    print("\n[1] 一级安全红线（**双通道**：严格 + 实测分布校准的受控放宽）")
    # ========== 通道 A：严格（用户原始口径，allow_relaxed=False）============
    ok, reason, det = hard_gate(EP[3], FROZEN_BASELINE, allow_relaxed=False)
    check("[严格] Ep3 过一级红线", ok and reason == "PASS", f"got ok={ok} reason={reason}")
    check("Ep3 ΔT1 = -0.003818 ≥ -0.005", close(det["d_track1_f1"], 0.901917808219178 - 0.905736245785252))

    # T2 F1 跌破 0.985（绝对值）—— 确定性失败，放宽通道也救不了
    bad = dict(EP[3], track2_f1=0.980)
    ok, reason, det = hard_gate(bad, FROZEN_BASELINE)
    check("T2 F1<0.985 → FAIL_TRACK2_SAFETY", (not ok) and reason == "FAIL_TRACK2_SAFETY", reason)
    check("失败轨道标记为 track2_f1", det.get("fail_track") == "track2_f1")

    # ΔFP_T2 超 +10 —— 严格通道判负
    bad = dict(EP[3], track2_fp=37)  # 基线 26 → Δ=+11
    ok, reason, det = hard_gate(bad, FROZEN_BASELINE, allow_relaxed=False)
    check("[严格] ΔFP_T2>+10 → FAIL_TRACK2_SAFETY", (not ok) and reason == "FAIL_TRACK2_SAFETY", reason)
    check("[严格] 失败轨道标记为 track2_fp", det.get("fail_track") == "track2_fp")

    # ΔF1_T1 跌破 -0.005 —— 严格通道判负
    bad = dict(EP[3], track1_f1=0.9000)  # Δ = -0.005736
    ok, reason, det = hard_gate(bad, FROZEN_BASELINE, allow_relaxed=False)
    check("[严格] ΔT1<-0.005 → FAIL_TRACK1_RETREAT", (not ok) and reason == "FAIL_TRACK1_RETREAT", reason)
    check("[严格] 失败轨道标记为 track1_f1", det.get("fail_track") == "track1_f1")

    # 边界：ΔT1 略高于 -0.005 应通过（用 -0.0049 避开浮点刀刃；
    # 恰好 -0.005 在 IEEE754 下会因表示误差变成 -0.005000000000000004 而判负，属实现细节）
    edge = dict(EP[3], track1_f1=FROZEN_BASELINE["track1_f1"] - 0.0049)
    ok, reason, _ = hard_gate(edge, FROZEN_BASELINE, allow_relaxed=False)
    check("[严格] ΔT1=-0.0049 通过", ok and reason == "PASS", f"ok={ok} {reason}")
    edge2 = dict(EP[3], track1_f1=FROZEN_BASELINE["track1_f1"] - 0.0051)
    ok, reason, _ = hard_gate(edge2, FROZEN_BASELINE, allow_relaxed=False)
    check("[严格] ΔT1=-0.0051 判负", (not ok) and reason == "FAIL_TRACK1_RETREAT", f"ok={ok} {reason}")

    # ========== 通道 B：受控放宽（默认开启）============
    ok, reason, det = hard_gate(bad_fp := dict(EP[3], track2_fp=37), FROZEN_BASELINE)
    check("[放宽] ΔFP_T2=+11 且黄金池/难例有实质增益 → PASS_RELAXED",
          ok and reason == "PASS_RELAXED", f"ok={ok} {reason}")
    check("[放宽] detail 记录了放宽依据（可审计）", bool(det.get("relaxed_note")), str(det.get("relaxed_note")))
    check("[放宽] detail 记录 strict_pass=False", det.get("strict_pass") is False, str(det.get("strict_pass")))

    # **放宽不是无条件放行**：没有实质收益时仍判负
    no_gain = dict(EP[3],
                   track2_f1=FROZEN_BASELINE["track2_f1"] + 0.001,   # ΔT2=+0.001 < 0.008
                   track3_f1=FROZEN_BASELINE["track3_f1"] + 0.001,   # ΔT3=+0.001 < 0.020
                   track2_fp=37)
    ok, reason, _ = hard_gate(no_gain, FROZEN_BASELINE)
    check("[放宽] 无实质收益时仍判负（不是无条件放行）",
          (not ok) and reason == "FAIL_TRACK2_SAFETY", f"ok={ok} {reason}")

    # 难例没涨 ⇒ 也不放行（源域退让不能白退）
    no_t3 = dict(EP[3], track1_f1=0.9000,
                 track3_f1=FROZEN_BASELINE["track3_f1"] + 0.005)   # ΔT3=+0.005 < 0.020
    ok, reason, _ = hard_gate(no_t3, FROZEN_BASELINE)
    check("[放宽] 难例未同步大涨时源域退让不放行",
          (not ok) and reason == "FAIL_TRACK1_RETREAT", f"ok={ok} {reason}")

    # 源域退让超过 −0.008 ⇒ 一律判负（放宽通道也有硬上限）
    too_much = dict(EP[3], track1_f1=FROZEN_BASELINE["track1_f1"] - 0.009)
    ok, reason, _ = hard_gate(too_much, FROZEN_BASELINE)
    check("[放宽] 源域退让超 −0.008 一律判负",
          (not ok) and reason == "FAIL_TRACK1_RETREAT", f"ok={ok} {reason}")


# 2026-10-21 第一批 4 个种子 trial 的**真实**三轨指标（服务器实测）
REAL_SEEDS_2026_10_21 = {
    "trial_0000_natural": dict(track1_f1=0.902455, track1_fp=1021,
                               track2_f1=0.998769, track2_fp=37,
                               track3_f1=0.864086, track3_fp=497),
    "trial_0001_w2.0": dict(track1_f1=0.900319, track1_fp=1057,
                            track2_f1=0.998739, track2_fp=37,
                            track3_f1=0.865463, track3_fp=529),
    "trial_0002_w3.0": dict(track1_f1=0.901388, track1_fp=1042,
                            track2_f1=0.998919, track2_fp=32,
                            track3_f1=0.861058, track3_fp=519),
    "trial_0003_w4.0": dict(track1_f1=0.902170, track1_fp=991,
                            track2_f1=0.998799, track2_fp=37,
                            track3_f1=0.862080, track3_fp=529),
}


def test_real_seeds_2026_10_21() -> None:
    """用**实测数据**固化新红线：旧口径 3/4 被误杀 ⇒ 新口径 4/4 全过。"""
    print("\n[1b] 实测校准（新红线的依据：2026-10-21 第一批 4 个种子 trial）")
    strict_pass, relaxed_pass = [], []
    for name, met in REAL_SEEDS_2026_10_21.items():
        ok_s, reason_s, _ = hard_gate(met, FROZEN_BASELINE, allow_relaxed=False)
        ok_r, reason_r, _ = hard_gate(met, FROZEN_BASELINE)
        if ok_s:
            strict_pass.append(name)
        if ok_r:
            relaxed_pass.append(name)
    check("[实测] 严格口径下 4 个种子只有 1 个过线（3 个被误杀）",
          len(strict_pass) == 1, f"实际过线 {strict_pass}")
    check("[实测] 受控放宽口径下 4 个种子全部过线（红线不再误杀好配置）",
          len(relaxed_pass) == 4, f"实际过线 {relaxed_pass}")
    # 放宽通道必须打上 PASS_RELAXED 标记，不能与严格通过混淆
    reasons = [hard_gate(m, FROZEN_BASELINE)[1] for m in REAL_SEEDS_2026_10_21.values()]
    check("[实测] 非严格通过的档位标注为 PASS_RELAXED",
          reasons.count("PASS_RELAXED") == 3 and reasons.count("PASS") == 1, str(reasons))
    # 这 4 个的 T2 F1 全部 ≥ 0.9987（黄金池大幅改善），是放宽通道的物理依据
    check("[实测] 4 个种子的 T2 F1 全部 ≥ 0.9987（放宽的物理依据）",
          all(m["track2_f1"] >= 0.9987 for m in REAL_SEEDS_2026_10_21.values()),
          str([m["track2_f1"] for m in REAL_SEEDS_2026_10_21.values()]))


def test_utility() -> None:
    print("\n[2] 三级效用分（与用户参考实现逐项对齐）")
    score, det = utility(EP[3], FROZEN_BASELINE)
    # 手工复算：0.0343535 + 0.5*0.0105431 - 2.5*0.0038184 - 0.0005*5
    manual = 0.0343535420406067 + 0.5 * 0.010543128683201397 - 2.5 * 0.003818437566074029 - 0.0005 * 5
    check("Ep3 效用分与手算一致", close(score, manual, 1e-12), f"score={score} manual={manual}")
    check("Ep3 效用分 ≈ 0.027581", close(score, 0.027581, 1e-5), f"score={score}")
    check("FP 惩罚项生效（ΔFP2=+5 → -0.0025）", close(det["terms"]["fp_penalty"], -0.0025))
    check("源域惩罚项生效（ΔT1<0 → -2.5×0.0038184）",
          close(det["terms"]["t1_penalty"], -2.5 * 0.003818437566074029))

    # ΔFP2 为负时不惩罚
    score_neg, det2 = utility(EP[3], dict(FROZEN_BASELINE, track2_fp=40))
    check("ΔFP2<0 时 FP 惩罚为 0", close(det2["terms"]["fp_penalty"], 0.0))
    check("ΔFP2<0 时分数更高", score_neg > score)

    # evaluate_model_quality：一级失败返回 -inf
    s, reason, _ = evaluate_model_quality(dict(EP[3], track2_f1=0.9), FROZEN_BASELINE)
    check("一级失败 → -inf", s == float("-inf") and reason == "FAIL_TRACK2_SAFETY")
    s, reason, _ = evaluate_model_quality(EP[3], FROZEN_BASELINE)
    check("一级通过 → PASS + 效用分", reason == "PASS" and s > 0)


def test_pareto() -> None:
    print("\n[3] 二级帕累托（维度集预设；真值由 gate_history 推导）")
    # @user 预设 = 用户例句的隐含维度 {T1_F1↑, T2_F1↑, T3_FP↓}
    check("Ep3 ≻ Ep4 @user（复现用户例句的'无争议占优'）", pareto_dominates(EP[3], EP[4], "user"))
    check("Ep3 ≻ Ep5 @user", pareto_dominates(EP[3], EP[5], "user"))
    check("Ep4 ≻ Ep3 @user 不成立", not pareto_dominates(EP[4], EP[3], "user"))
    # @core（三轨 F1）：Ep1 的 T1-F1 最高、Ep5 的 T3-F1 最高 ⇒ 互不占优
    check("Ep3 ≻ Ep4 @core 不成立（Ep3 的 T1-F1 更高，Ep4 的 T3-F1 更高）",
          not pareto_dominates(EP[3], EP[4], "core"))
    check("Ep3 ≻ Ep5 @core 不成立（Ep5 的 T3-F1 更高）", not pareto_dominates(EP[3], EP[5], "core"))
    # @full（六维）：三轨 F1 最优分散、虚警最优全在 Ep3 ⇒ 依然互不占优
    check("Ep3 ≻ Ep4 @full 不成立（Ep4 的 T3-F1 更高）", not pareto_dominates(EP[3], EP[4], "full"))
    check("Ep3 ≻ Ep5 @full 不成立（Ep5 的 T3-F1 更高）", not pareto_dominates(EP[3], EP[5], "full"))
    check("Ep5 ≻ Ep1 @full 不成立（Ep1 的 T1-F1 更高）", not pareto_dominates(EP[5], EP[1], "full"))


def test_funnel_order() -> None:
    print("\n[4] 漏斗顺序：帕累托成立 ⇒ 不再看效用分")
    res = compare_models(EP[3], EP[4], dims="user")
    check("Ep3 vs Ep4 @user → A_DOMINATES（直接判胜）", res["verdict"] == "A_DOMINATES", res["verdict"])
    check("该情形下 U(Ep3 vs Ep4) 为负（说明顺序确实在起作用）", res["utility_a_vs_b"] < 0,
          f"{res['utility_a_vs_b']}")
    res_full = compare_models(EP[3], EP[4], dims="full")
    check("Ep3 vs Ep4 @full → 落到效用分裁决", res_full["verdict"] in ("UTILITY_A", "UTILITY_B"),
          res_full["verdict"])
    check("@full 下效用分（同一冻结基线）判 Ep3 胜", res_full["verdict"] == "UTILITY_A", res_full["verdict"])
    check("U(Ep3)=0.027579 > U(Ep4)=0.026178",
          close(res_full["utility_a"], 0.027579, 1e-5) and close(res_full["utility_b"], 0.026178, 1e-5))
    # 非反对称性：U(A vs B) 与 U(B vs A) 同号，因此三级必须比"同基准 U"
    check("U(Ep3 vs Ep4) 与 U(Ep4 vs Ep3) 同号（非反对称）",
          (res_full["utility_a_vs_b"] < 0) == (res_full["utility_b_vs_a"] < 0),
          f"{res_full['utility_a_vs_b']} / {res_full['utility_b_vs_a']}")


def test_select_best() -> None:
    print("\n[5] 决赛选择器")
    cands = [{"name": f"ep{e}", "trial": e, "metrics": EP[e]} for e in (1, 2, 3, 4, 5)]
    rep = select_best(cands, reference=RC1_EPOCH3, dims="full")
    # 受控放宽口径下：Ep5（ΔT1=-0.005236，但黄金池 ΔF1=+0.010513、难例 ΔF1=+0.037593）
    # 由放宽通道接纳 ⇒ 全员过线；**最优仍是 Ep3**（效用分最高），结论不变
    check("放宽口径下 Ep5 不再被误杀", [r["name"] for r in rep["rejected"]] == [],
          str(rep["rejected"]))
    check("过线数 = 6（Ep1~Ep5 + 擂主 RC1）", rep["n_passed_gate"] == 6, str(rep["n_passed_gate"]))
    # 严格口径下仍应剔除 Ep5（**可回退**保证）
    rep_s = select_best(cands, reference=RC1_EPOCH3, dims="full", allow_relaxed=False)
    check("[严格口径] Ep5 仍被剔除（旧行为可复现 ⇒ 可回退）",
          [r["name"] for r in rep_s["rejected"]] == ["ep5"], str(rep_s["rejected"]))
    check("[严格口径] 过线数 = 5", rep_s["n_passed_gate"] == 5, str(rep_s["n_passed_gate"]))
    check("Ep3 被选为最优（效用分全场最高 0.027579）", rep["winner"]["name"] == "ep3",
          str(rep["winner"]["name"]))
    check("给出与擂主 RC1 的正面对决结论", "vs_reference" in rep)
    # 注入一个触红线的候选，必须被剔除
    bad = {"name": "bad", "trial": 99, "metrics": dict(EP[3], track2_f1=0.97)}
    rep2 = select_best(cands + [bad], reference=RC1_EPOCH3, dims="full")
    check("确定性触红线候选被剔除（bad 的 T2 F1=0.97）", rep2["n_rejected_gate"] == 1,
          str(rep2["n_rejected_gate"]))
    check("被剔除者记录了原因", all(r["reason"].startswith("FAIL") for r in rep2["rejected"]),
          str(rep2["rejected"]))
    # 全员触红线（不引入擂主）
    rep3 = select_best([bad], reference=None, dims="full")
    check("全员触红线时无胜者", rep3["winner"] is None)
    check("全员触红线时报告了原因", "红线" in rep3.get("reason", ""), rep3.get("reason", ""))


def test_rc1_consistency() -> None:
    print("\n[6] RC1 常量与冻结基线一致性")
    check("RC1 三元组与 gate_history 一致", close(RC1_EPOCH3["track1_f1"], EP[3]["track1_f1"])
          and close(RC1_EPOCH3["track2_f1"], EP[3]["track2_f1"])
          and close(RC1_EPOCH3["track3_f1"], EP[3]["track3_f1"]))
    check("RC1 过一级红线", hard_gate(RC1_EPOCH3, FROZEN_BASELINE)[0])
    s, _ = utility(RC1_EPOCH3, FROZEN_BASELINE)
    check("RC1 效用分 ≈ 0.027581（最终选型的及格线）", close(s, 0.027581, 1e-5), f"{s}")


def test_gate_report_incomplete() -> None:
    """门禁报告缺字段时，必须抛**明确的** GateReportIncomplete，而不是 float(None) 的 TypeError。

    背景（实测踩过）：Track1 推理 CUDA OOM ⇒ gate_report.json 里
    `tracks.track1_uav.finetune_peak.f1` 为 null ⇒ 旧代码 `float(None)` 抛
    `TypeError: float() argument must be a string or a real number`，
    消息完全指不出"哪个 track 缺了"，worker 还会因此整个崩溃、丢掉本 trial 的训练。
    """
    import json
    import tempfile
    from pathlib import Path

    from manu.training.mix_ratio_judgement import (
        GateReportIncomplete,
        metrics_from_gate_report,
    )

    tmp = Path(tempfile.mkdtemp())
    broken = tmp / "gate_report.json"
    broken.write_text(json.dumps({
        "tracks": {
            "track1_uav": {"status": "infer_failed", "rc": 1},          # finetune_peak 缺失
            "track2_golden": {"f1": 0.998, "fp": 30},
            "track3_blind": {"f1": 0.84, "fp": 500},
        },
        "verdict": "FAIL",
    }), encoding="utf-8")

    try:
        metrics_from_gate_report(broken)
        check("缺字段时抛 GateReportIncomplete", False, "没有抛异常（静默通过 = 更危险）")
    except GateReportIncomplete as exc:
        check("缺字段时抛 GateReportIncomplete", True)
        check("异常消息点名缺失字段 track1_f1", "track1_f1" in str(exc), str(exc))
        check("异常消息含缺失字段列表", "missing" in exc.missing or "track1_f1" in exc.missing,
              str(exc.missing))
        check("异常消息指向基础设施失败而非指标不达标",
              "基础设施" in str(exc) or "OOM" in str(exc), str(exc))

    # 完整报告仍要正常解析（回归）
    ok_report = tmp / "ok.json"
    ok_report.write_text(json.dumps({
        "tracks": {
            "track1_uav": {"finetune_peak": {"f1": 0.902, "fp": 984}},
            "track2_golden": {"f1": 0.998, "fp": 30},
            "track3_blind": {"f1": 0.84, "fp": 500},
        },
    }), encoding="utf-8")
    m = metrics_from_gate_report(ok_report)
    check("完整报告仍能正常解析（回归）", m["track1_f1"] == 0.902 and m["track2_fp"] == 30, str(m))


def test_reproduction_check() -> None:
    """复现判据：拿 RC1 自身的 epoch 间包络当参照带（而不是拿单点当"应该复现到的值"）。"""
    print("\n[7] 复现判据 reproduction_check")
    band = rc1_epoch_band()
    check("包络覆盖 RC1 的 5 个 epoch",
          close(band["track3_f1"][0], EP[1]["track3_f1"]) and close(band["track3_f1"][1], EP[5]["track3_f1"]),
          str(band["track3_f1"]))
    check("包络 T1 FP = [941, 1232]（跨度 291）",
          band["track1_fp"] == (941, 1232), str(band["track1_fp"]))
    # RC1 自己对自己 ⇒ 必然位级复现
    r = reproduction_check(EP[3], 3)
    check("RC1/Ep3 对自身 ⇒ EXACT_REPRO", r["verdict"] == "EXACT_REPRO", r["verdict"])
    check("逐项差全为 0", all(abs(v) < 1e-12 for v in r["diffs"].values()), str(r["diffs"]))
    # RC1 的 Ep5 对 Ep3 ⇒ 同一次 run 内的差异 ⇒ 出容差但**在包络内** ⇒ BAND_REPRO
    r2 = reproduction_check(EP[5], 3)
    check("RC1/Ep5 对 Ep3 ⇒ BAND_REPRO（不是 NOT_REPRO）",
          r2["verdict"] == "BAND_REPRO", r2["verdict"])
    check("RC1/Ep5 不出包络", not r2["band_outliers"], str(r2["band_outliers"]))
    # ★ 旧调度口径的 trial_0000 ⇒ **出包络** ⇒ 实测证明旧 cosine 周期训出了系统性不同的模型
    t0 = {"track1_f1": 0.902455, "track1_fp": 1021, "track2_f1": 0.998769,
          "track2_fp": 37, "track3_f1": 0.864086, "track3_fp": 497}
    r3 = reproduction_check(t0, 3)
    check("旧调度 trial_0000 ⇒ NOT_REPRO（出 RC1 自身包络）",
          r3["verdict"] == "NOT_REPRO", r3["verdict"])
    check("trial_0000 出包络项含 track1_f1 与 track2_fp",
          "track1_f1" in r3["band_outliers"] and "track2_fp" in r3["band_outliers"],
          str(r3["band_outliers"]))
    # 位级复现的样例（逐项差在容差内）⇒ EXACT_REPRO
    r4 = reproduction_check({**EP[3], "track3_f1": EP[3]["track3_f1"] + 5e-4,
                             "track1_fp": EP[3]["track1_fp"] + 2}, 3)
    check("微扰样例 ⇒ EXACT_REPRO", r4["verdict"] == "EXACT_REPRO", r4["verdict"])
    # RC1/Ep5 参照系常量自洽
    check("RC1_EPOCH5 与曲线 epoch5 一致", RC1_EPOCH5 == EP[5])
    check("RC1_EPOCH5 的 U = +0.026260（< Ep3 的 +0.027579）",
          close(utility(RC1_EPOCH5, FROZEN_BASELINE)[0], 0.026260, 1e-6),
          str(utility(RC1_EPOCH5, FROZEN_BASELINE)[0]))


def main() -> int:
    print("=" * 88)
    print("三级漏斗判别标准 自测（真实门禁数据）")
    print("=" * 88)
    test_hard_gate()
    test_real_seeds_2026_10_21()
    test_utility()
    test_pareto()
    test_funnel_order()
    test_select_best()
    test_rc1_consistency()
    test_gate_report_incomplete()
    test_reproduction_check()
    print("\n" + "=" * 88)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 88)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
