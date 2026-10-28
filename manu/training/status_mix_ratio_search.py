#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混合比例超参搜索的**状态查询**（任何时候、任何机器都能查，只要能 SSH 到服务器）。

用法（服务器）：
```bash
python manu/training/status_mix_ratio_search.py
python manu/training/status_mix_ratio_search.py --top 15
python manu/training/status_mix_ratio_search.py --out-json runs/optuna_mix_ratio/status_report.json
```

输出内容：
1. 调度器存活状态（PID / 心跳 / 是否还活着）
2. 进度：完成 / 剪枝 / 失败 / 在跑
3. GPU 实时占用
4. Top-N 试验表（按效用分 U 排序，含三轨指标与一级门禁）
5. **按 mix_spec 分组的最优**（直接回答"哪种混合比例最好"）
6. 与现任擂主 RC1(Ep3) 的正面对决（`select_best` 漏斗判定）
7. 预计剩余时间
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.training.mix_ratio_judgement import (  # noqa: E402
    DEFAULT_GATE_TRACK1_MIN_DELTA,
    DEFAULT_GATE_TRACK2_F1_MIN,
    DEFAULT_GATE_TRACK2_FP_MAX_INCREMENT,
    FROZEN_BASELINE,
    RC1_BY_EPOCH,
    RC1_EPOCH3,
    compare_models,
    hard_gate,
    reference_name,
    select_best,
    utility,
)

OUT_ROOT_DEFAULT = "runs/optuna_mix_ratio"
STUDY_NAME_DEFAULT = "mix_ratio_search"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def gpu_status() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        lines = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                lines.append(f"GPU{parts[0]}: {parts[1]}/{parts[2]} MiB, util {parts[3]}%")
        return " | ".join(lines) if lines else "(nvidia-smi 不可用)"
    except Exception:
        return "(nvidia-smi 不可用)"


def read_summary(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ==============================================================================
# 口径判定：某个 trial 的**调度口径**（cosine 周期）与**测量 epoch** ⇒ 决定它能与谁比
# ==============================================================================
# 为什么必须判口径（实测教训）
# ------------------------------
# RC1/Ep3 经历的是 **5 epoch 全长的 cosine 曲线**（Ep3 末 lr = 1.8264e-06 = 峰值 60.9%）；
# 而修复前的搜索按 **3 epoch 自己的 cosine**（Ep3 末 lr 已退到 1.0000e-06 = 峰值 33.3%，偏差 1.83×）
# ⇒ 那些 trial 的末 epoch 权重**系统性比 RC1 学得少**，与 mix_spec 的真实效应混在一起。
#
# 2026-10-21 决策：后续 trial 改跑 **5 epoch** ⇒ 门禁量的是 **Ep5**，
# 而 **RC1/Ep5**（同一次 5-epoch 运行的末 epoch）与 **RC1/Ep3**（其第 3 个 epoch）**指标不同**
# ⇒ trial 必须与**同一个测量 epoch 的 RC1** 比，否则是跨尺度比较。
#
# 判定方式：直接读 trial 自己的训练日志（ground truth）
#   · `cosine 周期 N ep`  → sched_epochs（新格式）
#   · `[GATE] 三轨门禁（... / epochK）` → gate_epoch（被测量的 epoch）
#   · 旧格式（无"cosine 周期"字样）→ sched_epochs = 3
RC1_COSINE_EPOCHS = 5

# 口径分组：key = (sched_epochs, gate_epoch) → (分组名, RC1 参照系 or None)
def caliber_group(sched: int | None, gate_ep: int | None) -> tuple[str, dict | None]:
    """把 (cosine 周期, 测量 epoch) 映射到「口径分组名 + 该组对应的 RC1 参照系」。"""
    if sched is None or gate_ep is None:
        return "未知口径", None
    if sched != RC1_COSINE_EPOCHS:
        return "旧调度口径", None
    ref = RC1_BY_EPOCH.get(gate_ep)
    if ref is None:
        return f"RC1-Ep{gate_ep}口径(无参照)", None
    return f"RC1-Ep{gate_ep}口径", ref


def trial_caliber(out_root: Path, trial_name: str) -> dict:
    """返回该 trial 的口径信息：``sched_epochs`` / ``gate_epoch`` / ``comparable`` / ``group`` / ``note``。"""
    log = out_root / "logs" / f"{trial_name}.log"
    base = {"sched_epochs": None, "gate_epoch": None, "comparable": False,
            "group": "未知口径", "note": "无日志", "ref": None}
    if not log.is_file():
        return base
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except Exception:
        base["note"] = "日志读取失败"
        return base
    sched_line = ""
    gate_line = ""
    for line in text.splitlines():
        if line.startswith("[SCHED]") and not sched_line:
            sched_line = line
        if line.startswith("[GATE] 三轨门禁") and not gate_line:
            gate_line = line
    if not sched_line:
        base["note"] = "无[SCHED]行"
        return base
    if "cosine 周期" in sched_line:
        try:
            sched = int(sched_line.split("cosine 周期")[1].split("ep")[0].strip())
        except Exception:
            sched = None
    else:
        # 旧格式（--sched-epochs 特性之前）：cosine 周期 = 训练 epoch 数 = 3
        sched = 3
    gate_ep = None
    if gate_line:
        # 兼容两种格式：
        #   旧："[GATE] 三轨门禁（w3.5 / epoch3）"
        #   新："[GATE] 三轨门禁（w3.5 / epoch3（末 epoch））"
        m = re.search(r"epoch(\d+)", gate_line)
        if m:
            try:
                gate_ep = int(m.group(1))
            except Exception:
                gate_ep = None
    group, ref = caliber_group(sched, gate_ep)
    if sched == RC1_COSINE_EPOCHS and ref is not None:
        note = f"与 RC1/Ep{gate_ep} 同口径"
    elif sched == RC1_COSINE_EPOCHS:
        note = f"cosine {sched}ep 但测量 epoch 未知"
    else:
        note = f"旧口径(cosine周期{sched}ep,末epoch lr仅RC1的55%)"
    return {"sched_epochs": sched, "gate_epoch": gate_ep, "comparable": ref is not None,
            "group": group, "note": note, "ref": ref}


def median(xs: list[float]) -> float:
    """中位数（奇偶都取中间两个的平均）。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description="混合比例超参搜索状态查询")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--study-name", default=STUDY_NAME_DEFAULT)
    ap.add_argument(
        "--reference-frame",
        default="",
        help="参照系包 JSON（换特征语义如 no-GMC 时**必须传**；"
             "内含 frozen_baseline / rc1 / gate 阈值）。不传 = 沿用 GMC 语义的模块常量（旧行为不变）",
    )
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tail-log", type=int, default=15)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--strict-only", action="store_true",
                    help="只认严格一级红线（禁用受控放宽通道），用于复现旧结论")
    ap.add_argument("--all-calibers", action="store_true",
                    help="决赛候选池不按调度口径过滤（默认只取 cosine 周期 5ep、与 RC1 可比的 trial）")
    args = ap.parse_args()

    # ---- 参照系注入（不传 --reference-frame ⇒ 与模块常量逐位相同，旧行为不变）----
    base = FROZEN_BASELINE
    rc1_ref = RC1_EPOCH3
    gate_cfg = {
        "track2_f1_min": DEFAULT_GATE_TRACK2_F1_MIN,
        "track2_fp_max_increment": DEFAULT_GATE_TRACK2_FP_MAX_INCREMENT,
        "track1_min_delta": DEFAULT_GATE_TRACK1_MIN_DELTA,
    }
    if args.reference_frame:
        _rf = Path(args.reference_frame)
        if not _rf.is_file():
            print(f"[FAIL] 找不到参照系包：{_rf}")
            return 3
        _ref = json.loads(_rf.read_text(encoding="utf-8"))
        base = {**FROZEN_BASELINE, **(_ref.get("frozen_baseline") or {})}
        rc1_ref = {**RC1_EPOCH3, **(_ref.get("rc1") or {})}
        gate_cfg = {**gate_cfg, **(_ref.get("gate") or {})}
        print(f"[REF] 已注入参照系：{_rf}")
        print(f"[REF] 门禁阈值：{gate_cfg}")

    allow_relaxed = not args.strict_only

    out_root = Path(args.out_root)
    print("=" * 100)
    print(f"混合比例超参搜索 · 状态报告   {time.strftime('%F %T')}")
    print("=" * 100)

    # ---- 1. 调度器存活 ----
    pid_file = out_root / "dispatcher.pid"
    pid = None
    alive = False
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            alive = pid_alive(pid)
        except Exception:
            pid = None
    status_path = out_root / "status.json"
    hb = {}
    if status_path.is_file():
        try:
            hb = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            hb = {}
    hb_age = (time.time() - hb.get("updated", 0)) if hb.get("updated") else None
    print("[1] 调度器")
    if pid is None:
        print("    未启动（无 PID 文件）")
    elif alive:
        print(f"    ✅ 运行中  PID={pid}   心跳 {hb_age:.0f}s 前" if hb_age else f"    ✅ 运行中  PID={pid}")
    else:
        print(f"    ❌ 已退出  PID={pid}（可能被杀/断电；重跑 launch 会自动从断点继续）")
    if hb.get("running"):
        print("    在跑的 trial：")
        for r in hb["running"]:
            print(f"      {r.get('trial')}  GPU{r.get('gpu')}  PID{r.get('pid')}  mix={r.get('mix')}")
    if hb.get("elapsed_h") is not None:
        print(f"    已运行 {hb['elapsed_h']:.2f} 小时")

    # ---- 2. 进度 ----
    rows = read_summary(out_root / "optuna_summary.csv")
    total = len(rows)
    success = [r for r in rows if r["status"] == "SUCCESS"]
    pruned = [r for r in rows if r["status"] == "PRUNED"]
    crashed = [r for r in rows if r["status"] not in ("SUCCESS", "PRUNED")]
    print(f"\n[2] 进度：共 {total} 个 trial | 过线 {len(success)} | 剪枝 {len(pruned)} | 异常 {len(crashed)}")
    if hb.get("best_value") is not None:
        _u_champ = utility(rc1_ref, base)[0]
        _champ_name = "natural 对照" if args.reference_frame else "RC1/Ep3"
        # ★ GMC 侧保持旧文案逐字一致（不带正号）；no-GMC 侧用带符号写法更清楚
        _u_txt = f"{_u_champ:+.6f}" if args.reference_frame else f"{_u_champ:.6f}"
        print(f"    当前最佳 U = {hb['best_value']:+.6f}（trial #{hb.get('best_trial')}）"
              f"   及格线 {_champ_name} = {_u_txt}")

    # ---- 3. GPU ----
    print(f"\n[3] GPU：{gpu_status()}")

    # ---- 4. Top-N ----
    def f(row, key, default=0.0):
        try:
            return float(row.get(key, default) or default)
        except Exception:
            return default

    def i(row, key, default=0):
        try:
            return int(float(row.get(key, default) or default))
        except Exception:
            return default

    def metrics_of(row) -> dict | None:
        """从 CSV 行还原 metrics（缺失则 None）。"""
        if not row.get("t1_f1") or not row.get("t2_f1") or not row.get("t3_f1"):
            return None
        return {
            "track1_f1": f(row, "t1_f1"), "track1_fp": i(row, "t1_fp"),
            "track2_f1": f(row, "t2_f1"), "track2_fp": i(row, "t2_fp"),
            "track3_f1": f(row, "t3_f1"), "track3_fp": i(row, "t3_fp"),
        }

    def u_of(row) -> float | None:
        """**重算**效用分（相对冻结基线），不信任 CSV 的 utility 列。

        理由（实测）：PRUNED 的 trial 其 CSV utility 列是 "None"（worker 只在过线时写数），
        直接用会得到 0.0 的误导值；而手填/污染的列同样不可信。
        判决口径与 `mix_ratio_judgement` 完全一致 —— 同一把尺子。
        """
        m = metrics_of(row)
        if m is None:
            return None
        return utility(m, base)[0]

    def gate_verdict(row) -> tuple[bool, str]:
        """按一级红线重判该行（CSV 的 status 列只作参考）。

        返回 ``(是否过线, 理由)``；理由可能是 ``PASS`` / ``PASS_RELAXED`` / 失败原因。
        双通道口径见 `mix_ratio_judgement.DEFAULT_RELAXED_GATE`。
        """
        m = metrics_of(row)
        if m is None:
            return False, "NO_METRICS"
        ok, reason, _ = hard_gate(
            m, base,
            track2_f1_min=gate_cfg["track2_f1_min"],
            track2_fp_max_increment=gate_cfg["track2_fp_max_increment"],
            track1_min_delta=gate_cfg["track1_min_delta"],
            allow_relaxed=allow_relaxed,
        )
        return ok, reason

    def gate_ok(row) -> bool:
        return gate_verdict(row)[0]

    def gate_mark(row) -> str:
        ok, reason = gate_verdict(row)
        if not ok:
            return "⛔触红线"
        if reason == "PASS_RELAXED":
            return "🟡过线(放宽)"
        return "✅过线"

    ranked = sorted(
        [r for r in rows if metrics_of(r) is not None],
        key=lambda r: (u_of(r) if u_of(r) is not None else -99), reverse=True,
    )
    # 调度口径（cosine 周期）—— 决定该 trial 的 U 是否可与 RC1 直接比
    cal_of = {r["trial"]: trial_caliber(out_root, r["trial"]) for r in rows}
    print(f"\n[4] Top-{args.top}（按效用分 U，相对冻结基线；U 一律重算，不信 CSV 列）")
    print("    " + "-" * 96)
    print(f"    {'trial':<14}{'mix':<9}{'实测1:x':<9}{'T1 F1':<10}{'T2 F1':<10}{'T3 F1':<10}"
          f"{'T1FP':<7}{'T2FP':<7}{'T3FP':<7}{'U':<11}{'U-RC1':<11}")
    # 参照系两行（**相对冻结基线**的 U；U-RC1 列 = U − U(RC1)）
    u_rc1 = utility(rc1_ref, base)[0]
    print(f"    {'【冻结基线】':<14}{'—':<9}{'—':<9}"
          f"{base['track1_f1']:<10.6f}{base['track2_f1']:<10.6f}"
          f"{base['track3_f1']:<10.6f}"
          f"{base['track1_fp']:<7}{base['track2_fp']:<7}"
          f"{base['track3_fp']:<7}{0.0:<+11.6f}{-u_rc1:<+11.6f}")
    _champ_label = "【现任擂主】" if args.reference_frame else "【RC1=Ep3】"
    print(f"    {_champ_label:<14}{'natural':<9}{'3.09':<9}"
          f"{rc1_ref['track1_f1']:<10.6f}{rc1_ref['track2_f1']:<10.6f}"
          f"{rc1_ref['track3_f1']:<10.6f}"
          f"{rc1_ref['track1_fp']:<7}{rc1_ref['track2_fp']:<7}{rc1_ref['track3_fp']:<7}"
          f"{u_rc1:<+11.6f}{0.0:<+11.6f}")
    for r in ranked[: args.top]:
        u = u_of(r)
        cal = "✅口径" if cal_of[r["trial"]]["comparable"] else "⚠️旧口径"
        print(f"    {r['trial']:<14}{r.get('mix_spec',''):<9}{f(r,'measured_ratio'):<9.2f}"
              f"{f(r,'t1_f1'):<10.6f}{f(r,'t2_f1'):<10.6f}{f(r,'t3_f1'):<10.6f}"
              f"{i(r,'t1_fp'):<7}{i(r,'t2_fp'):<7}{i(r,'t3_fp'):<7}"
              f"{(u if u is not None else 0.0):<+11.6f}"
              f"{(u - u_rc1) if u is not None else 0.0:<+11.6f}  {gate_mark(r)}  {cal}")
    print("    " + "-" * 96)

    # ---- 5. 按 mix_spec 分组（**纳入全部有指标的 trial**，含被红线剪枝的）----
    # ★ **按调度口径分栏**：
    #    「主表」= cosine 周期 5 ep（与 RC1 同口径）⇒ U 可与 RC1 直接比
    #    「附表」= 旧口径（3 ep 周期，末 epoch lr 仅 RC1 的 55%）⇒ **不可与 RC1 比**，只作链路健康证据
    # ★ **统计量用中位数**（读结果铁律：单 trial 的 U 不能否定一个比例），同时给出最好的一条供参考。
    print("\n[5] 按混合比例分组（同档取**中位数**；主表=与 RC1 同调度口径）")
    print("    " + "-" * 110)

    groups: dict[str, list[dict]] = {}
    for r in rows:
        if metrics_of(r) is None:
            continue
        groups.setdefault(r.get("mix_spec", "?"), []).append(r)

    def print_group_table(items: list[dict], title: str, note: str, ref_u: float | None) -> None:
        print(f"  ── {title} ── {note}")
        if not items:
            print("      （无）")
            return
        for m in sorted(items):
            rs = items[m]
            us = [u_of(r) for r in rs if u_of(r) is not None]
            if not us:
                continue
            u_med = median(us)
            best = max(rs, key=lambda r: (u_of(r) if u_of(r) is not None else -99))
            u_best = u_of(best)
            ub = u_best if u_best is not None else 0.0
            bench = "" if ref_u is None else f" (中位-擂主={u_med-ref_u:+.6f})"
            print(f"    {m:<9} n={len(rs):<3} 目标 1:{f(best,'target_ratio'):<6.2f} 实测 1:{f(best,'measured_ratio'):<6.2f}"
                  f"  U中位数={u_med:+.6f}{bench}"
                  f"  U最好={ub:+.6f}" + ("" if ref_u is None else f" (最好-擂主={ub-ref_u:+.6f})"))
            print(f"      最好条：[{best['trial']}] T1={f(best,'t1_f1'):.6f} T2={f(best,'t2_f1'):.6f} "
                  f"T3={f(best,'t3_f1'):.6f} T1FP={i(best,'t1_fp')} T2FP={i(best,'t2_fp')} T3FP={i(best,'t3_fp')}"
                  f"  {gate_mark(best)}")
        print("    " + "-" * 110)

    # ★ 按**口径分组**（cosine 周期 × 测量 epoch）—— 每组用自己的 RC1 参照系
    by_cal: dict[str, dict] = {}
    for m, rs in groups.items():
        for r in rs:
            g = cal_of[r["trial"]]["group"]
            slot = by_cal.setdefault(g, {"items": {}, "ref": cal_of[r["trial"]]["ref"], "trials": []})
            slot["items"].setdefault(m, []).append(r)
            slot["trials"].append(r)

    # ★ 参照系注入时，no-GMC 语义下的「现任擂主」是 **natural 对照**（参照系包里的 rc1），
    #   而不是 GMC 的 RC1_EPOCH3 ⇒ 口径组的擂主一律换成注入的 rc1，
    #   否则会拿 GMC 的擂主跟 no-GMC 的候选对局（跨语义，不可比）。
    champion = rc1_ref
    if args.reference_frame:
        for _g, _slot in by_cal.items():
            if _slot["ref"] is not None:
                _slot["ref"] = champion

    # 排序：有参照系的组排前面（按测量 epoch），其余在后
    def group_sort_key(name: str):
        ep = 0
        for k in (3, 5):
            if f"Ep{k}" in name:
                ep = k
        return (0 if by_cal[name]["ref"] is not None else 1, ep, name)

    for gname in sorted(by_cal, key=group_sort_key):
        slot = by_cal[gname]
        ref = slot["ref"]
        ref_u = utility(ref, base)[0] if ref is not None else None
        if ref is not None:
            note = f"⇒ 擂主 = 该口径的 RC1（U={ref_u:+.6f}）"
        else:
            note = "⇒ 无同口径参照系，仅作链路健康证据"
        print_group_table(slot["items"], gname, note, ref_u)
        print()
    if allow_relaxed:
        print(f"    ⚠️ 一级红线为**双通道**：严格口径 ΔFP_T2 ≤ {gate_cfg['track2_fp_max_increment']:+d}；受控放宽口径 ΔFP_T2 ≤ +15 且要求")
        print("       T2 F1 ≥ +0.008、T3 F1 ≥ +0.020 同时有实质增益（🟡过线(放宽) 标记）。")
    else:
        print(f"    ⚠️ 当前为**严格口径**（--strict-only）：ΔFP_T2 ≤ {gate_cfg['track2_fp_max_increment']:+d}。")
    if args.reference_frame and allow_relaxed:
        print("    ⚠️ 参照系注入下：**受控放宽通道的 ΔFP_T2 上限仍是 GMC 默认 +15**（未随参照系重标定），")
        print(f"       而本语义正常带为 [+43,+51] ⇒ 放宽通道实际不可达；**以严格口径 ΔFP_T2 ≤ {gate_cfg['track2_fp_max_increment']:+d} 为准**。")
    print("    ⚠️ T2 虚警的单 trial 噪声约 ±5 ⇒ **红线切在噪声带中部**；")
    print("       同一 mix_spec 多个 trial 一律看**中位数**，别用单点下结论。")

    # ---- 6. 与 RC1 的正面对决（**按口径分组**，每组用自己那个测量 epoch 的 RC1 擂主）----
    # 候选池 = **有指标**的 trial（不按 CSV 的 status 过滤），
    #   因为 CSV 的 status 是 worker 当时那套红线的产物；这里用**当前口径重新判**，
    #   才能让"严格口径剪枝、但放宽口径过线"的 trial 回到候选池（可审计）。
    # ★ **默认只取有同口径参照系的组**：
    #   · 旧调度口径（cosine 3ep）⇒ U 系统性偏低，不可与 RC1 同尺度比
    #   · 5-epoch 口径的 trial 量的是 Ep5 ⇒ 擂主必须是 **RC1/Ep5**，不是 RC1/Ep3
    #   `--all-calibers` 可放开（仅供回溯对照）。
    def mk_cand(r):
        return {
            "name": r["trial"], "trial": r["trial"],
            "metrics": {
                "track1_f1": f(r, "t1_f1"), "track1_fp": i(r, "t1_fp"),
                "track2_f1": f(r, "t2_f1"), "track2_fp": i(r, "t2_fp"),
                "track3_f1": f(r, "t3_f1"), "track3_fp": i(r, "t3_fp"),
            },
        }

    duel_groups = [g for g in sorted(by_cal, key=group_sort_key)
                   if args.all_calibers or by_cal[g]["ref"] is not None]
    skipped_caliber = sorted({r["trial"] for g in by_cal
                               if not args.all_calibers and by_cal[g]["ref"] is None
                               for r in by_cal[g]["trials"]})
    if duel_groups:
        print("\n[6] 决赛（漏斗判定：过线 → 帕累托占优 → 效用分裁决）；**按口径分组，各组各自擂主**")
        for gname in duel_groups:
            slot = by_cal[gname]
            ref = slot["ref"]
            cands = [mk_cand(r) for r in slot["trials"]]
            if not cands:
                continue
            scope = "全部口径（--all-calibers）" if args.all_calibers else gname
            if ref is None:
                # 该口径**没有**同尺度的 RC1 参照系（如旧调度口径）⇒ 只列候选，不做对局
                print(f"\n    ▸ 分组【{scope}】 无同口径 RC1 参照系 ⇒ **不参与对局**"
                      f"（候选 {len(cands)} 个：{[c['name'] for c in cands]}）")
                continue
            rep = select_best(cands, reference=ref, dims="full", allow_relaxed=allow_relaxed,
                              frozen_baseline=base, gate_config=gate_cfg)
            print(f"\n    ▸ 分组【{scope}】 擂主={reference_name(ref)}"
                  f"（U={utility(ref, base)[0]:+.6f}）")
            print(f"      候选 {rep['n_candidates']} 个，过一级红线 {rep['n_passed_gate']} 个，"
                  f"帕累托极大元 {rep.get('n_pareto_maximal', 0)} 个")
            if rep.get("rejected"):
                print(f"      被红线剔除：{[(r['name'], r['reason']) for r in rep['rejected']]}")
            if rep.get("winner"):
                w = rep["winner"]
                print(f"      🏆 最优：{w['name']}（{w.get('params', {}).get('mix_spec', '')}）"
                      f"  U={w.get('utility'):+.6f}")
                print(f"         理由：{rep.get('winner_reason')}")
                vs = rep.get("vs_reference", {})
                print(f"         vs 擂主：{vs.get('verdict')}   U={vs.get('utility_a', 0):+.6f} "
                      f"vs {vs.get('utility_b', 0):+.6f}")
                print(f"         ⇒ {'✅ 打败本组擂主' if rep.get('beats_reference') else '❌ 未打败本组擂主'}")
            # ★ 用户的目标是「比 RC1/Ep3 更好」⇒ 额外对**现任擂主 Ep3** 也报一次
            if ref is not champion:
                rep3 = select_best(cands, reference=champion, dims="full", allow_relaxed=allow_relaxed,
                                   frozen_baseline=base, gate_config=gate_cfg)
                vs3 = rep3.get("vs_reference", {})
                # ★ 文案**按参照系**分支：GMC 侧逐字保留旧文案（断言按字面匹配），
                #   no-GMC 侧换成「现任擂主(natural 对照)」措辞，避免跨语义误读。
                if args.reference_frame:
                    _cn, _beat, _lose = "natural 对照", "✅ 打败现任擂主", "❌ 未打败现任擂主"
                else:
                    _cn, _beat, _lose = "RC1/Ep3", "✅ 打败 RC1/Ep3", "❌ 未打败 RC1/Ep3"
                print(f"         vs 现任擂主 {_cn}（U={u_rc1:+.6f}）：{vs3.get('verdict')}   "
                      f"U={vs3.get('utility_a', 0):+.6f} vs {vs3.get('utility_b', 0):+.6f}"
                      f"  ⇒ {_beat if rep3.get('beats_reference') else _lose}")
        if skipped_caliber and not args.all_calibers:
            print(f"\n    （已按口径排除 {len(skipped_caliber)} 个非同口径 trial：{skipped_caliber}）")
    else:
        if args.all_calibers:
            print("\n[6] 决赛：尚无有指标的候选 trial")
        else:
            print("\n[6] 决赛：尚无**有同口径 RC1 参照系**的候选 trial")
            print("    （旧调度口径的 trial 其 U 系统性偏低；5-epoch trial 量的是 Ep5，擂主须为 RC1/Ep5）")

    # ---- 7. 预计剩余 ----
    if success and total:
        avg_min = sum(f(r, "duration_min") for r in rows if f(r, "duration_min")) / max(
            1, len([r for r in rows if f(r, "duration_min")])
        )
        print(f"\n[7] 平均单 trial 耗时 {avg_min:.1f} 分钟"
              f"（4 卡并行 ⇒ 约 {avg_min/4:.1f} 分钟/trial）")

    # ---- 8. 日志尾部 ----
    log = out_root / "logs" / "dispatcher.log"
    if log.is_file() and args.tail_log > 0:
        print(f"\n[8] 调度器日志尾部（{log}）")
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-args.tail_log:]:
                print("    " + line)
        except Exception as exc:
            print(f"    读取失败：{exc}")

    if args.out_json:
        # 分组统计（按 mix_spec）：可比口径与旧口径分开，各给 n / U中位数 / U最好
        group_stats = {}
        for m, rs in groups.items():
            us = [u_of(r) for r in rs if u_of(r) is not None]
            if not us:
                continue
            group_stats[m] = {
                "n": len(rs),
                "n_comparable": sum(1 for r in rs if cal_of[r["trial"]]["comparable"]),
                "u_median": median(us),
                "u_best": max(us),
                "u_median_comparable": median(
                    [u_of(r) for r in rs if cal_of[r["trial"]]["comparable"] and u_of(r) is not None]
                ) if any(cal_of[r["trial"]]["comparable"] for r in rs) else None,
                "trials": [r["trial"] for r in rs],
            }
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "generated": time.time(),
                    "dispatcher_pid": pid,
                    "dispatcher_alive": alive,
                    "heartbeat_age_s": hb_age,
                    "total_trials": total,
                    "success": len(success),
                    "pruned": len(pruned),
                    "crashed": len(crashed),
                    "u_rc1": u_rc1,
                    "calibers": cal_of,
                    "group_stats": group_stats,
                    "ranked": ranked[: args.top],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n[JSON] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
