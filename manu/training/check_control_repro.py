#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对若干 trial 做**复现判决**（回答"相同配置到底复现了 RC1 没有"）。

用法（服务器）：
```bash
python manu/training/check_control_repro.py                      # 默认查全部对照 trial
python manu/training/check_control_repro.py trial_0012 trial_0014
python manu/training/check_control_repro.py --epoch 5            # 按别的测量 epoch 判
```

判决（`mix_ratio_judgement.reproduction_check`）：
  · `EXACT_REPRO` —— 与 RC1[epoch] 逐项差都在容差内 ⇒ 位级复现
  · `BAND_REPRO`  —— 不满足逐项容差，但落在 **RC1 自身 epoch 间包络**内 ⇒ 正常波动
  · `NOT_REPRO`   —— 连包络都出不去 ⇒ 真异常
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.training.mix_ratio_judgement import (  # noqa: E402
    FROZEN_BASELINE,
    RC1_EPOCH3,
    reproduction_check,
    utility,
)

OUT_ROOT_DEFAULT = "runs/optuna_mix_ratio"
# 默认关注的对照 trial（含用户关心的位级复现 #14）
DEFAULT_TRIALS = ("trial_0012", "trial_0013", "trial_0014", "trial_0015")


def load_rows(out_root: Path) -> dict:
    p = out_root / "optuna_summary.csv"
    if not p.is_file():
        print(f"[FATAL] 找不到 {p}")
        return {}
    with p.open("r", encoding="utf-8", newline="") as fh:
        return {r["trial"]: r for r in csv.DictReader(fh)}


def metrics_of(r: dict) -> dict:
    return {
        "track1_f1": float(r["t1_f1"]), "track1_fp": int(float(r["t1_fp"])),
        "track2_f1": float(r["t2_f1"]), "track2_fp": int(float(r["t2_fp"])),
        "track3_f1": float(r["t3_f1"]), "track3_fp": int(float(r["t3_fp"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="对照 trial 的复现判决")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    ap.add_argument("--trials", nargs="*", default=list(DEFAULT_TRIALS))
    ap.add_argument("--epoch", type=int, default=3, help="对照的测量 epoch（默认 3 = RC1/Ep3）")
    ap.add_argument("--all", action="store_true", help="对 summary 里所有有指标的 trial 都判一遍")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    rows = load_rows(out_root)
    if not rows:
        return 1

    targets = list(args.trials)
    if args.all:
        targets = [t for t, r in rows.items() if r.get("t1_f1")]

    u_rc1 = utility(RC1_EPOCH3, FROZEN_BASELINE)[0]
    print("=" * 98)
    print(f"复现判决（对照 RC1/Ep{args.epoch}；参照带 = RC1 自身 epoch1~5 包络）")
    print(f"RC1/Ep{args.epoch} 的 U = {u_rc1:+.6f}（及格线）")
    print("=" * 98)

    n_exact = n_band = n_not = 0
    for t in targets:
        r = rows.get(t)
        if r is None:
            print(f"{t}: 尚未结算")
            continue
        if not r.get("t1_f1"):
            print(f"{t}: 无指标（status={r.get('status')}）")
            continue
        m = metrics_of(r)
        rep = reproduction_check(m, args.epoch)
        u = utility(m, FROZEN_BASELINE)[0]
        v = rep["verdict"]
        n_exact += v == "EXACT_REPRO"
        n_band += v == "BAND_REPRO"
        n_not += v == "NOT_REPRO"
        print(f"\n{t}  mix={r.get('mix_spec')}  实测1:{r.get('measured_ratio')}  "
              f"status={r.get('status')}  gate={r.get('gate_verdict')}")
        print(f"   T1={m['track1_f1']:.6f}/{m['track1_fp']}  T2={m['track2_f1']:.6f}/{m['track2_fp']}  "
              f"T3={m['track3_f1']:.6f}/{m['track3_fp']}")
        print(f"   U={u:+.6f}（比 RC1/Ep{args.epoch} {u - u_rc1:+.6f}）")
        print(f"   >>> 判决：{v}   超逐项容差={rep['outliers']}   出RC1包络={rep['band_outliers']}")

    print("\n" + "=" * 98)
    print(f"汇总：EXACT_REPRO {n_exact} 个 / BAND_REPRO {n_band} 个 / NOT_REPRO {n_not} 个")
    print("=" * 98)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
