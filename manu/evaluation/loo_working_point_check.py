#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Leave-one-sequence-out check of a delivered working point.

Why
---
A delivery working point is picked by scanning candidates on the *same* sequences it is later
reported on.  That makes one question mandatory before signing it off:

    *Is the chosen point overfitted to particular sequences, or would it have been chosen
    without them?*

The check is the classical leave-one-sequence-out (LOO) protocol applied to the candidate
grid:

    for every sequence s:
        1. re-rank all candidates using ONLY the other 23 sequences;
        2. take the winner c*(s);
        3. look at sequence s under c*(s) -- does the point still deliver there?

If c*(s) is (almost) always the same point, the delivered working point is what an
independent sequence would have selected too, and the choice is not sequence-specific.
If c*(s) is stable but the held-out sequence regresses under it, the point is fragile for
that sequence and that has to be stated in the delivery notes.

The grid report carries per-sequence counts, so every aggregate is recomputed from counts
(``tp``/``fp``/``fn``/``gt``) plus the per-sequence frame counts; nothing is re-inferred and
no model is touched.  The frame counts come from the cache meta, the same source the probe
used, so the recomputed aggregates reproduce the grid's own ``overall`` numbers exactly
(self-check, see ``--self-check``).

Usage
-----
    python manu/evaluation/loo_working_point_check.py \
        --grid-json runs/optuna_mix_ratio/trials/trial_0080/lq_eval/robustness_grid.json \
        --cache-root runs/optuna_mix_ratio/trials/trial_0080/lq_eval/ms_cache \
        --delivered "th_base=0.42,confirm_consec=7,th_ground=0.26"

    # point the check at a different grid / delivered point
    python manu/evaluation/loo_working_point_check.py ... --delivered base

Execution machine: any machine that can read the grid report and the cache meta (CPU only).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

TARGET_F1 = 0.88
TARGET_RECALL = 0.85
TARGET_FAR = 0.030


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-sequence-out check of a delivery working point")
    parser.add_argument("--grid-json", required=True, help="robustness_grid.json produced by the probe")
    parser.add_argument("--cache-root", required=True, help="Cache root the grid was produced from (for frame counts)")
    parser.add_argument("--delivered", required=True,
                        help="Grid label of the delivered point, e.g. "
                             "'th_base=0.42,confirm_consec=7,th_ground=0.26' or 'base'")
    parser.add_argument("--out-json", default="", help="Optional path for the machine readable report")
    return parser.parse_args()


def sequence_frames(cache_root: Path, sequence: str) -> int:
    with open(cache_root / f"{sequence}.pkl", "rb") as handle:
        return int(pickle.load(handle)["meta"]["frames"])


def aggregate(rows: dict, frames: int) -> dict:
    """Recompute the delivery metrics from raw counts (the same way the probe does)."""
    values = rows.values() if isinstance(rows, dict) else rows
    tp = sum(r["tp"] for r in values)
    fp = sum(r["fp"] for r in values)
    fn = sum(r["fn"] for r in values)
    det = tp + fp
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / det if det else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "det": det,
        "recall": recall, "precision": precision, "f1": f1,
        "far_per_frame": fp / frames if frames else 0.0,
        "frames": frames,
    }


def passes(metrics: dict) -> bool:
    return (metrics["f1"] >= TARGET_F1 and metrics["recall"] >= TARGET_RECALL
            and metrics["far_per_frame"] <= TARGET_FAR)


def rank_key(metrics: dict) -> tuple:
    """The same objective the project uses: criteria first, then F1, then the smaller FAR."""
    return (not passes(metrics), -metrics["f1"], metrics["far_per_frame"])


def main() -> int:
    args = parse_args()
    grid_path = Path(args.grid_json)
    cache_root = Path(args.cache_root)
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    per_sequence = grid["per_sequence"]
    labels = list(per_sequence)
    sequences = list(per_sequence[labels[0]])

    delivered = args.delivered
    if delivered not in per_sequence:
        raise SystemExit(f"delivered label {delivered!r} not in the grid; available: {labels[:5]} ...")

    frames = {seq: sequence_frames(cache_root, seq) for seq in sequences}
    total_frames = sum(frames.values())

    # ---- self-check: the recomputed aggregate must reproduce the grid's own overall ----
    mismatches = []
    for label in labels:
        recomputed = aggregate({s: per_sequence[label][s] for s in sequences}, total_frames)
        for key in ("tp", "fp", "fn"):
            if recomputed[key] != grid["overall"][label][key]:
                mismatches.append((label, key, recomputed[key], grid["overall"][label][key]))
        for key in ("f1", "recall", "far_per_frame"):
            if abs(recomputed[key] - grid["overall"][label][key]) > 1e-9:
                mismatches.append((label, key, recomputed[key], grid["overall"][label][key]))

    delivered_metrics = aggregate({s: per_sequence[delivered][s] for s in sequences}, total_frames)

    # ---- leave-one-sequence-out ----
    loo_rows = []
    for held_out in sequences:
        others = [s for s in sequences if s != held_out]
        frames_others = sum(frames[s] for s in others)
        ranked = sorted(
            labels,
            key=lambda label: rank_key(aggregate({s: per_sequence[label][s] for s in others}, frames_others)),
        )
        best = ranked[0]
        best_others = aggregate({s: per_sequence[best][s] for s in others}, frames_others)
        # the held-out sequence, evaluated at the point the other 23 would have chosen
        held_row = per_sequence[best][held_out]
        held_f1 = held_row["f1"]
        # ... and at the point actually delivered
        delivered_held = per_sequence[delivered][held_out]
        loo_rows.append({
            "held_out": held_out,
            "loo_selected": best,
            "loo_selected_is_delivered": best == delivered,
            "loo_f1_on_others": best_others["f1"],
            "loo_passes_on_others": passes(best_others),
            "held_f1_at_loo_point": held_f1,
            "held_f1_at_delivered": delivered_held["f1"],
            "held_f1_delta": held_f1 - delivered_held["f1"],
            "held_passes_at_loo_point": held_f1 >= TARGET_F1,
            "held_passes_at_delivered": delivered_held["f1"] >= TARGET_F1,
        })

    same = sum(1 for r in loo_rows if r["loo_selected_is_delivered"])
    loo_pass = sum(1 for r in loo_rows if r["held_passes_at_loo_point"])
    delivered_pass = sum(1 for r in loo_rows if r["held_passes_at_delivered"])
    regressions = [r for r in loo_rows
                   if r["held_f1_at_loo_point"] < r["held_f1_at_delivered"] - 1e-9]

    report = {
        "grid_json": str(grid_path),
        "cache_root": str(cache_root),
        "delivered": delivered,
        "delivered_metrics": delivered_metrics,
        "num_sequences": len(sequences),
        "total_frames": total_frames,
        "self_check_mismatches": mismatches,
        "loo": {
            "same_point_as_delivered": same,
            "num_sequences": len(loo_rows),
            "held_pass_at_loo_point": loo_pass,
            "held_pass_at_delivered": delivered_pass,
            "regressions": regressions,
            "rows": loo_rows,
        },
    }

    print("=" * 104)
    print(f"LOO working-point check  |  grid={grid_path.name}  delivered={delivered!r}")
    print("=" * 104)
    print(f"candidates: {len(labels)}   sequences: {len(sequences)}   frames: {total_frames}")
    print(f"delivered metrics: F1={delivered_metrics['f1'] * 100:.4f}%  "
          f"R={delivered_metrics['recall'] * 100:.3f}%  "
          f"P={delivered_metrics['precision'] * 100:.3f}%  "
          f"TP={delivered_metrics['tp']} FP={delivered_metrics['fp']} "
          f"FAR={delivered_metrics['far_per_frame']:.6f}")
    print(f"self-check (recomputed == grid overall): "
          f"{'PASS' if not mismatches else 'FAIL ' + str(mismatches[:3])}")
    print()
    print(f"LOO 选出的工作点与交付点相同: {same}/{len(loo_rows)}")
    print(f"留出序列在 LOO 点上仍达 F1>=88: {loo_pass}/{len(loo_rows)}")
    print(f"留出序列在交付点上达 F1>=88:    {delivered_pass}/{len(loo_rows)}")
    print()
    print(f"{'held-out sequence':<32} {'LOO picked':<44} {'F1@LOO':>8} {'F1@deliv':>9} {'Δ':>8} {'pass@LOO':>9}")
    print("-" * 118)
    for row in loo_rows:
        flag = "" if row["held_f1_delta"] >= -1e-9 else "  <== 回归"
        mark = "" if row["loo_selected_is_delivered"] else "  (≠交付点)"
        print(f"{row['held_out']:<32} {row['loo_selected']:<44} "
              f"{row['held_f1_at_loo_point'] * 100:>8.2f} {row['held_f1_at_delivered'] * 100:>9.2f} "
              f"{row['held_f1_delta'] * 100:>+8.2f} {'OK' if row['held_passes_at_loo_point'] else 'FAIL':>9}"
              f"{mark}{flag}")

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report -> {path.resolve()}")

    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
