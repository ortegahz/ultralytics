#!/usr/bin/env python3
"""Re-rank a completed system-SOTA search sweep under the delivery constraint.

The search itself optimises a generic objective (meet-all-criteria, then F1, then lower FAR).
This script re-selects the winner from the *recorded* sweep under a stricter, explicitly
delivery-driven rule, at zero extra evaluation cost:

    hard filter : F1 >= frozen single-frame baseline F1   (never ship worse than what we have)
                  and Recall >= 85.0
    objective   : minimise FAR per frame (the binding delivery criterion, 0.030)
                  then maximise F1
                  then maximise Recall

It also reports the best config under the original objective, and the best config that meets
all three delivery criteria, so the selection can be audited.

Usage:
    python manu/postprocess/rank_sweep_delivery.py \
        --search-json runs/logs/sys_sota_longquan_search.json \
        --out-params runs/logs/sys_sota_longquan_best_params.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINE_F1 = 0.870701037410      # frozen single-frame baseline, 24 sequences, inbox
BASELINE_FAR = 0.063157066510
DELIVERY_F1 = 88.0
DELIVERY_RECALL = 85.0
DELIVERY_FAR = 0.030


def normalise(entry: dict) -> dict:
    """Flatten a sweep record into a common shape (metrics may be flat or nested)."""
    m = entry.get("metrics") or {}
    if not m:
        m = entry
    frames = entry.get("frames") or m.get("frames") or 0
    tp = m.get("tp", 0)
    fp = m.get("fp", 0)
    fn = m.get("fn", 0)
    gt = m.get("gt", 0)
    recall = m.get("recall")
    precision = m.get("precision")
    f1 = m.get("f1")
    if recall is None and (tp + fn):
        recall = tp / (tp + fn)
    if precision is None and (tp + fp):
        precision = tp / (tp + fp)
    if f1 is None and (recall or 0) and (precision or 0):
        f1 = 2 * recall * precision / (recall + precision)
    far = m.get("far_per_frame")
    if far is None and frames:
        far = fp / frames
    return {
        "tag": entry.get("tag", ""),
        "params": entry.get("params", {}),
        "tp": tp, "fp": fp, "fn": fn, "gt": gt, "frames": frames,
        "recall": recall or 0.0, "precision": precision or 0.0, "f1": f1 or 0.0,
        "far_per_frame": far or 0.0,
    }


def fmt(row: dict) -> str:
    return ("F1=%7.3f%%  R=%6.2f%%  P=%6.2f%%  FAR=%.4f  tp=%-7s fp=%-7s fn=%-7s"
            % (row["f1"] * 100, row["recall"] * 100, row["precision"] * 100,
               row["far_per_frame"], row["tp"], row["fp"], row["fn"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-json", required=True)
    ap.add_argument("--out-params", default="")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    data = json.loads(Path(args.search_json).read_text())
    rows = [normalise(e) for e in data.get("sweep", []) if e.get("params")]
    print(f"[RANK] sweep 记录 {len(rows)} 条 | 搜索用时 {data.get('elapsed_seconds', 0):.0f}s | "
          f"budget_exhausted={data.get('budget_exhausted')} rounds_completed={data.get('rounds_completed')}")

    if not rows:
        raise SystemExit("[RANK] sweep 为空,无法重排")

    # --- A. hard filter: never worse than the frozen baseline, and recall clears 85 ---
    passing = [r for r in rows if r["f1"] >= BASELINE_F1 and r["recall"] * 100 >= DELIVERY_RECALL]
    # --- B. original objective (what the search optimised) ---
    by_original = sorted(rows, key=lambda r: (r["f1"], -r["far_per_frame"]), reverse=True)
    # --- C. meets all three delivery criteria ---
    by_delivery = [r for r in rows
                   if r["f1"] * 100 >= DELIVERY_F1
                   and r["recall"] * 100 >= DELIVERY_RECALL
                   and r["far_per_frame"] <= DELIVERY_FAR]

    print(f"\n[A] 硬门槛(F1 >= {BASELINE_F1*100:.3f}% 且 R >= {DELIVERY_RECALL}%):通过 {len(passing)}/{len(rows)}")
    print(f"[B] 三条交付判据全达标(F1>={DELIVERY_F1} R>={DELIVERY_RECALL} FAR<={DELIVERY_FAR}):{len(by_delivery)} 条")
    if by_delivery:
        best_delivery = sorted(by_delivery, key=lambda r: (r["f1"], -r["far_per_frame"]), reverse=True)[0]
        print("    三判据全达标者中最好的一条: " + fmt(best_delivery))

    if not passing:
        print("\n!! 没有任何配置达到硬门槛 —— 系统级后处理在当前候选池口径下不优于冻结基线。")
        print("   仍按原目标给出最优者供参考:")
        print("   " + fmt(by_original[0]))
        chosen = by_original[0]
    else:
        chosen = sorted(passing, key=lambda r: (r["far_per_frame"], -r["f1"], -r["recall"]))[0]
        print("\n[A 组] 按「FAR 最小 → F1 最大」排序,前 %d 条:" % args.top)
        for r in sorted(passing, key=lambda r: (r["far_per_frame"], -r["f1"]))[:args.top]:
            print("   " + fmt(r) + "  | " + r["tag"])

    print("\n>>> 选中配置: " + fmt(chosen))
    for k, v in chosen["params"].items():
        print(f"      {k:24s} {v}")

    if args.out_params:
        out = Path(args.out_params)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(chosen["params"], indent=2, ensure_ascii=False))
        print(f"\n[OUT] 参数已写入 {out}")


if __name__ == "__main__":
    main()
