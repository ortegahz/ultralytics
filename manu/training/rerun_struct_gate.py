#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
补跑 no-GMC 结构线的三轨门禁（不重训）。

为什么需要它
------------
2026-11-05 夜里的结构线搜索里，**p3 结构的 6 条臂训练完全正常（权重完好）**，
但门禁在重建模型时炸了：

    [STRUCT] ckpt 自描述: no_p2=True head_level=p3 act=relu head_conv=64
    AssertionError: Only stride 4 (P2) or stride 2 (P1) is supported, got 8.

根因是 ``heatmap_model.py`` 的 stride 断言没放行 8（P3 头）⇒
缓存生成失败 ⇒ 门禁报告缺字段 ⇒ 被记成 ``GATE_REPORT_INCOMPLETE``（基础设施失败，
不是指标不达标）⇒ dispatcher 把它当成 CRASHED。断言已修（md5 0ff7c875…），
**权重一个字没动**，所以只要补跑门禁就能拿到真实指标。

本脚本做的事
------------
读参照系 JSON（no-GMC 语义）里的 ``frozen_baseline`` / ``gate`` / ``gate_base``，
**按 dispatcher 的同一口径**拼出门禁命令（绝对阈值语义），逐条臂跑，
把结果汇总成一张表。**基线数字全部从 JSON 读，不手抄**。

用法（服务器，conda `uav`）：
    python manu/training/rerun_struct_gate.py \
        --root /mnt/data/siping/runs/nogmc_struct2 \
        --reference-frame /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
        --baseline-cache /mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl \
        --trials trial_0002 trial_0003 trial_0004 trial_0005 trial_0010 trial_0011
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPT = "manu/training/auto_epoch_gate_check.py"


def build_cmd(ref: dict, weights: Path, out_dir: Path, gpu: int, args) -> list[str]:
    """按 dispatcher 的同一口径拼门禁命令（绝对阈值语义）。"""
    fb = ref["frozen_baseline"]
    gate = ref["gate"]
    gb = ref.get("gate_base") or {}

    track2_fp_abs_max = fb["track2_fp"] + gate["track2_fp_max_increment"]
    track1_f1_abs_min = fb["track1_f1"] + gate["track1_min_delta"]

    cmd = [
        sys.executable, GATE_SCRIPT,
        "--weights", str(weights),
        "--out-dir", str(out_dir),
        "--gpus", "1",
        "--device", str(gpu),
        "--infer-gpu", str(gpu),
        "--cache-gpus", str(gpu),
        "--baseline-cache", str(args.baseline_cache),
        "--min-track2-f1", str(gate["track2_f1_min"]),
        "--max-track2-fp", str(track2_fp_abs_max),
        "--min-track1-f1", str(track1_f1_abs_min),
        "--uav-data", str(args.uav_data),
        "--ft-root", str(args.ft_root),
    ]
    for key, flag in (
        ("track1_f1", "--base-track1-f1"), ("track1_tp", "--base-track1-tp"), ("track1_fp", "--base-track1-fp"),
        ("track2_f1", "--base-track2-f1"), ("track2_tp", "--base-track2-tp"),
        ("track2_fp", "--base-track2-fp"), ("track2_fn", "--base-track2-fn"),
        ("track3_f1", "--base-track3-f1"), ("track3_tp", "--base-track3-tp"),
        ("track3_fp", "--base-track3-fp"), ("track3_fn", "--base-track3-fn"),
    ):
        if key in gb:
            cmd += [flag, str(gb[key])]
    return cmd


def read_report(out_dir: Path) -> dict:
    p = out_dir / "gate_report.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="搜索输出根（ trials/ 在它下面）")
    ap.add_argument("--reference-frame", required=True)
    ap.add_argument("--baseline-cache", required=True)
    ap.add_argument("--uav-data", default="/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml")
    ap.add_argument("--ft-root", default="/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc")
    ap.add_argument("--trials", nargs="*", default=[])
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-tag", default="gate_rerun")
    args = ap.parse_args()

    ref = json.loads(Path(args.reference_frame).read_text(encoding="utf-8"))
    root = Path(args.root)
    trials = args.trials or sorted(p.name for p in (root / "trials").iterdir() if p.is_dir())

    rows = []
    for name in trials:
        ckpt = root / "trials" / name / "weights" / f"epoch{args.epoch}.pt"
        if not ckpt.is_file():
            print(f"[SKIP] {name}: 权重不存在 {ckpt}", flush=True)
            continue
        out_dir = root / "trials" / name / args.out_tag / f"epoch{args.epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(ref, ckpt, out_dir, args.gpu, args)
        print(f"\n[GATE] {name} → {out_dir}", flush=True)
        print(f"$ {' '.join(cmd)}\n", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
        rep = read_report(out_dir)
        rows.append((name, rc, round(time.time() - t0, 1), rep))
        print(f"[GATE] {name} rc={rc} 用时 {rows[-1][2]}s", flush=True)

    # ---- 汇总表 ----
    print("\n" + "=" * 120)
    print(f"{'arm':<12} {'rc':>3} {'sec':>7} {'T1 F1':>10} {'T1 FP':>7} {'T2 F1':>10} {'T2 FP':>7} {'T3 F1':>10} {'T3 FP':>7}  verdict")
    print("-" * 120)
    for name, rc, sec, rep in rows:
        def g(k):
            v = rep.get(k)
            return f"{v:.6f}" if isinstance(v, float) else (str(v) if v is not None else "-")
        print(f"{name:<12} {rc:>3} {sec:>7} {g('track1_f1'):>10} {g('track1_fp'):>7} "
              f"{g('track2_f1'):>10} {g('track2_fp'):>7} {g('track3_f1'):>10} {g('track3_fp'):>7}  {rep.get('verdict', '-')}")
    print("=" * 120)
    out_json = root / f"{args.out_tag}_summary.json"
    out_json.write_text(json.dumps(
        [{"trial": n, "rc": rc, "seconds": s, "report": r} for n, rc, s, r in rows],
        indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] 汇总 -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
