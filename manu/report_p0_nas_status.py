#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Comprehensive Status & Monday Morning Report Generator for P0 NAS 4096 Study.

Usage:
    python manu/report_p0_nas_status.py --output-root runs/optuna_p0_nas
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def parse_args():
    parser = argparse.ArgumentParser(description="Report P0 NAS Progress & Top-10 Architectures")
    parser.add_argument("--output-root", type=str, default="runs/optuna_p0_nas")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_root)
    if not out_dir.is_absolute():
        for cand in [PROJECT_ROOT / out_dir, Path("/tmp/pycharm_project_10ae9e2e") / out_dir]:
            if cand.exists():
                out_dir = cand
                break

    summary_csv = out_dir / "optuna_summary.csv"
    if not summary_csv.exists():
        print(colorstr("red", f"[ERROR] Summary CSV not found: {summary_csv}"))
        sys.exit(1)

    with open(summary_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(colorstr("yellow", "[WARN] Summary CSV is empty."))
        return

    total_trials = len(rows)
    success_trials = [r for r in rows if r["status"] == "SUCCESS"]
    crashed_trials = [r for r in rows if r["status"] == "CRASHED"]

    print("=" * 115)
    print(colorstr("bold", colorstr("green", f"   P0 NAS Study Status Report: {out_dir.name}")))
    print(f"   Total Trials Executed: {total_trials} | Success: {len(success_trials)} | Crashed/Failed: {len(crashed_trials)}")
    print("=" * 115)

    if not success_trials:
        print(colorstr("yellow", "[WARN] No successful trials recorded yet."))
        return

    # Sort by F1 descending
    success_trials.sort(key=lambda r: float(r["f1"]), reverse=True)
    top_trials = success_trials[: args.top_k]

    print(f"\n{colorstr('bold', colorstr('cyan', f'>>> TOP {len(top_trials)} BEST MICRO-ARCHITECTURES & CONFIGURATIONS:'))}")
    print("-" * 115)
    print(
        f"{'Rank':<5} | {'Trial':<10} | {'F1':<8} | {'Recall':<8} | {'Prec':<8} | {'Th':<5} | "
        f"{'Stem Type':<16} | {'Downsample':<15} | {'Gate Mode':<15} | {'Gate Ch':<7} | {'Depth':<5} | {'Fusion':<12}"
    )
    print("-" * 115)

    for rank, r in enumerate(top_trials, 1):
        f1 = float(r["f1"])
        rec = float(r["recall"]) * 100.0 if float(r["recall"]) <= 1.0 else float(r["recall"])
        prec = float(r["precision"]) * 100.0 if float(r["precision"]) <= 1.0 else float(r["precision"])
        th = float(r["best_th"])

        print(
            f"{rank:<5} | {r['trial']:<10} | {f1:<8.4f} | {rec:<7.2f}% | {prec:<7.2f}% | {th:<5.2f} | "
            f"{r['stem_type']:<16} | {r['downsample_mode']:<15} | {r['gate_input_mode']:<15} | {r['gate_mid_channels']:<7} | "
            f"{r['gate_depth']:<5} | {r['fusion_mode']:<12}"
        )

    best = top_trials[0]
    best_ckpt = out_dir / best["trial"] / "weights" / "best.pt"
    print("-" * 115)
    print(colorstr("bold", colorstr("magenta", f"\n★ ABSOLUTE CHAMPION TRIAL: {best['trial']}")))
    print(f"  F1-Score  : {float(best['f1']):.4f} (Recall: {float(best['recall'])*100:.2f}%, Precision: {float(best['precision'])*100:.2f}%)")
    print(f"  Best Th   : {float(best['best_th']):.2f} (TP: {best['tp']}, FP: {best['fp']}, GT: {best['gt']})")
    print(f"  Checkpoint: {best_ckpt}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
