#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
no-GMC 结构线的**验收合并表**：把板卡 NPU 实测速度 + 三轨门禁指标 + cs_t0043 基线
并成一张表，直接回答「速度大幅提升 + 精度微量下降」是否达成。

判据（用户 2026-11-05 定）
--------------------------
* 速度：NPU `rknn_run` 必须**显著低于** cs_t0043 同源基线（**同板同日实测**，不是历史值）
* 精度：ΔT3 ≤ 1.0 pp（vs cs_t0043），T1/T2 的下降幅度一并列出
* 两条口径都报：**peak**（门禁默认）与**带均**（阈值平均，更接近部署现实）

用法（宿主或服务器均可）：
    python manu/training/nogmc_struct_acceptance.py \
        --speed-results /home/manu/mnt/data/siping/runs/nogmc_struct2/npu_speed/hm_speed_results.txt \
        --gate-summary  /home/manu/mnt/data/siping/runs/nogmc_struct2/gate_rerun_summary.json \
        --root          /home/manu/mnt/data/siping/runs/nogmc_struct2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# cs_t0043 的**实测**三轨（no-GMC 语义，trial_0043/0049/0050 三粒子同配方 Model Soup）
CS_T0043 = {
    "t1_f1": 0.890766573,
    "t1_fp": 1210,
    "t2_f1": 0.998859,
    "t2_fp": 34,
    "t3_f1": 0.903960496,
    "t3_fp": 341,
}

# 臂 -> 结构标签（来自 nogmc_struct_plan.json）
ARM_STRUCT = {
    "trial_0000": ("CONTROL 复现", "原结构 + SiLU（= cs_t0043 同配方）"),
    "trial_0001": ("CONTROL 复现", "原结构 + SiLU（= cs_t0043 同配方）"),
    "trial_0002": ("STRUCT p3", "head@P3(80×80) + all-relu + 解冻b0-b4 + hc64"),
    "trial_0003": ("STRUCT p3", "head@P3(80×80) + all-relu + 解冻b0-b4 + hc32"),
    "trial_0004": ("STRUCT p3", "head@P3(80×80) + all-relu + 冻结b0-b4 + hc64"),
    "trial_0005": ("STRUCT p3", "head@P3(80×80) + all-relu + 冻结b0-b4 + hc32"),
    "trial_0006": ("STRUCT p2", "head@P2(160×160 骨干/4) + all-relu + 解冻 + hc64"),
    "trial_0007": ("STRUCT p2", "head@P2(160×160 骨干/4) + all-relu + 冻结 + hc32"),
    "trial_0008": ("STRUCT p2", "head@P2 + all-relu + 解冻 + hc64"),
    "trial_0009": ("STRUCT p2", "head@P2 + all-relu + 解冻 + hc32"),
    "trial_0010": ("STRUCT p3 lr3.0", "head@P3 + all-relu + 解冻 + hc64，lr 3.0"),
    "trial_0011": ("STRUCT p3 lr4.0", "head@P3 + all-relu + 解冻 + hc64，lr 4.0"),
}


def parse_speed(path: Path) -> dict[str, float]:
    """从 hm_speed_results.txt 解析 model -> rknn_run ms。"""
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    txt = path.read_text(encoding="utf-8", errors="ignore")
    for line in txt.splitlines():
        # 形如:  trial_0002_epoch20          23.585      -31.9%      1.87x
        m = re.match(r"\s*(\S+)\s+([0-9]+\.[0-9]+)\s+", line)
        if m and "rknn_run_ms" not in line:
            out[m.group(1)] = float(m.group(2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-results", required=True)
    ap.add_argument("--gate-summary", default="")
    ap.add_argument("--root", required=True)
    ap.add_argument("--t3-drop-max", type=float, default=1.0, help="T3 允许下降上限（pp）")
    args = ap.parse_args()

    root = Path(args.root)
    speed = parse_speed(Path(args.speed_results))
    # 归一到 trial 号
    speed_by_trial = {}
    for name, ms in speed.items():
        for t in ARM_STRUCT:
            if name.startswith(t):
                speed_by_trial[t] = ms
                break

    gates: dict[str, dict] = {}
    if args.gate_summary:
        gp = Path(args.gate_summary)
        if gp.is_file():
            for row in json.loads(gp.read_text(encoding="utf-8")):
                gates[row["trial"]] = row.get("report") or {}
    # 兜底：直接读各 trial 的 gate_report.json
    for t in ARM_STRUCT:
        if t in gates:
            continue
        for p in sorted((root / "trials" / t).glob("gate*/epoch*/gate_report.json")):
            try:
                gates[t] = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:
                pass

    base_speed = speed_by_trial.get("trial_0000") or speed_by_trial.get("trial_0001")

    print("=" * 150)
    print(f"{'arm':<11} {'结构':<14} {'NPU ms':>8} {'vs基线':>9} {'加速':>7} | "
          f"{'T1':>9} {'ΔT1(pp)':>8} {'T2':>9} {'T3':>9} {'ΔT3(pp)':>9} {'T3FP':>6}  判定")
    print("-" * 150)
    for t, (tag, desc) in ARM_STRUCT.items():
        ms = speed_by_trial.get(t)
        sp = base_speed / ms if (ms and base_speed) else None
        vs = (ms / base_speed - 1) * 100 if (ms and base_speed) else None
        g = gates.get(t, {})
        def gv(k):
            v = g.get(k)
            return f"{v:.6f}" if isinstance(v, (int, float)) else "-"
        d1 = d3 = None
        if isinstance(g.get("t1_f1"), (int, float)):
            d1 = (g["t1_f1"] - CS_T0043["t1_f1"]) * 100
        if isinstance(g.get("t3_f1"), (int, float)):
            d3 = (g["t3_f1"] - CS_T0043["t3_f1"]) * 100
        verdict = "-"
        if d3 is not None and ms is not None and base_speed:
            ok_acc = d3 >= -args.t3_drop_max
            ok_spd = ms < base_speed * 0.75
            verdict = ("✅ 达标" if (ok_acc and ok_spd)
                       else ("⚠️ 精度达标/速度不足" if ok_acc else
                             ("⚠️ 速度达标/精度掉太多" if ok_spd else "❌ 双不达标")))
        print(f"{t:<11} {tag:<14} {ms if ms else '-':>8} "
              f"{(f'{vs:+.1f}%' if vs is not None else '-'):>9} "
              f"{(f'{sp:.2f}x' if sp else '-'):>7} | "
              f"{gv('t1_f1'):>9} {(f'{d1:+.3f}' if d1 is not None else '-'):>8} "
              f"{gv('t2_f1'):>9} {gv('t3_f1'):>9} {(f'{d3:+.3f}' if d3 is not None else '-'):>9} "
              f"{gv('t3_fp'):>6}  {verdict}")
    print("=" * 150)
    print(f"\n基线：cs_t0043（3 粒子同配方 Model Soup）")
    print(f"  三轨：T1 {CS_T0043['t1_f1']:.6f} (FP {CS_T0043['t1_fp']}) | "
          f"T2 {CS_T0043['t2_f1']:.6f} (FP {CS_T0043['t2_fp']}) | "
          f"T3 {CS_T0043['t3_f1']:.6f} (FP {CS_T0043['t3_fp']})")
    print(f"  NPU：{base_speed if base_speed else '?'} ms（**同板同日实测**，非历史值）")
    print(f"\n判据：ΔT3 ≥ −{args.t3_drop_max} pp 且 NPU ms < 基线×0.75（即提速 > 25%）")
    for t, (tag, desc) in ARM_STRUCT.items():
        print(f"  {t}: {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
