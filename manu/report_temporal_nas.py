#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Quick Status Reporter & Leaderboard Extractor for Distributed Spatio-Temporal NAS Study.

Usage:
    python manu/report_temporal_nas.py \
        --output-root runs/optuna_temporal_nas \
        --top-k 15
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
    parser = argparse.ArgumentParser(description="Report Spatio-Temporal NAS Leaderboard")
    parser.add_argument("--output-root", type=str, default="runs/optuna_temporal_nas")
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_root)
    csv_path = out_dir / "optuna_summary.csv"

    if not csv_path.exists():
        print(colorstr("red", f"[ERROR] Summary CSV not found: {csv_path}"))
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[WARN] Summary CSV is currently empty.")
        sys.exit(0)

    success_rows = [r for r in rows if r["status"] == "SUCCESS"]
    crashed_rows = [r for r in rows if r["status"] != "SUCCESS"]

    print("=" * 128)
    print(colorstr("bold", colorstr("cyan", f"📊 Spatio-Temporal NAS Progress Report ({out_dir.name})")))
    print(f"Total Trials Launched: {len(rows)} | Succeeded: {len(success_rows)} | Crashed/Failed: {len(crashed_rows)}")
    print("=" * 128)

    if not success_rows:
        print("[INFO] No successful trials completed yet.")
        sys.exit(0)

    # Sort by F1 descending
    sorted_rows = sorted(success_rows, key=lambda r: float(r["f1"]), reverse=True)
    top_k = sorted_rows[: args.top_k]

    header = f"{'Rank':<5} | {'Trial':<11} | {'F1':<7} | {'Recall':<8} | {'Prec':<8} | {'Th':<5} | {'Kernel Mode':<15} | {'Gate Mode':<17} | {'K':<2} | {'S':<2} | {'PosW':<5}"
    print(header)
    print("-" * 128)

    for i, r in enumerate(top_k, 1):
        f1 = float(r["f1"])
        rec = float(r["recall"]) * 100.0 if float(r["recall"]) <= 1.0 else float(r["recall"])
        prec = float(r["precision"]) * 100.0 if float(r["precision"]) <= 1.0 else float(r["precision"])
        th = float(r["best_th"])
        k_mode = r["kernel_mode"]
        g_mode = r["gate_mode"]
        seq = r["seq_len"]
        stride = r["stride"]
        pos_w = r.get("pos_weight", "1.0")

        line = f"{i:<5} | {r['trial']:<11} | {f1:.4f}  | {rec:>6.2f}%  | {prec:>6.2f}%  | {th:.2f}  | {k_mode:<15} | {g_mode:<17} | {seq:<2} | {stride:<2} | {pos_w:<5}"
        if i == 1:
            print(colorstr("bold", colorstr("green", line)))
        elif f1 >= 0.9064:
            print(colorstr("bold", colorstr("yellow", line)))
        else:
            print(line)

    print("=" * 128)
    best = sorted_rows[0]
    best_f1 = float(best["f1"])
    delta_sota = best_f1 - 0.9064

    best_dir = out_dir / best["trial"] / "weights" / "best.pt"
    print(colorstr("bold", colorstr("magenta", f"★ CURRENT LEADER: {best['trial']} (F1 = {best_f1:.4f}, ΔSOTA = {delta_sota:+.4f})")))
    print(f"  Checkpoint Path: {best_dir}")
    print(f"  Architecture   : kernel={best['kernel_mode']}, mid_ch={best['mid_channels']}, gate_mode={best['gate_mode']}, K={best['seq_len']}, stride={best['stride']}")
    print(f"  Hyperparameters: lr0={best['lr0']}, beta={best['focal_beta']}, offset_w={best['offset_weight']}")
    print("=" * 128 + "\n")


if __name__ == "__main__":
    main()
