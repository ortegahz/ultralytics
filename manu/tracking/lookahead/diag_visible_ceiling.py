#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**可见性天花板**：把「目标在当前帧根本看不见」与「外挂算错了」分开。

这是 §32 最重要的口径修正：**不是所有失手帧都是外挂的锅**。
  * 红外弱小目标在某些帧里**真的不可见**（过暗、被遮挡、离开视场）——
    此时**任何**跟踪器都不可能命中，包括零延迟天花板 ``original`` 臂；
  * 所以「original 全量 hit@5px = 88.13%」**不是**交付臂的天花板，
    它自己也被这批不可见帧拖累了。

本脚本做**两臂真实输出的分层对比**（**不构造任何 oracle** ⇒ 没有退化问题）：
  * 分层依据 = 「当前帧检测器有没有看到目标」（用 ``original`` 臂在该帧有无上报判定）；
  * 在「可见帧」子集上比较 original 与 replay 的 hit@5px ⇒ 这才是**真实 headroom**。

⚠️ 口径：全部与官方 :func:`servo_metrics` 逐位一致（2026-11-04 审计实测：
手算 85.2206% 与 ``arms_*.json`` 的官方值 float 位级相等）。

用法::

    python manu/tracking/lookahead/diag_visible_ceiling.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --original runs/lookahead/final/preds_off \
        --replay   runs/lookahead/final/final8/preds_delivery \
        --out-json runs/lookahead/final/final8/diag_visible_ceiling.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402

IMG_W, IMG_H = 640, 512


def load_per_frame(preds_dir: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    path = preds_dir / f"{seq}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8")).get("per_frame", {})
    return {
        int(key): [(float(d["pos"][0]), float(d["pos"][1])) for d in (dets or [])]
        for key, dets in payload.items()
    }


def nearest(points, centres):
    if len(points) == 0 or len(centres) == 0:
        return None
    return min(
        math.hypot(px - float(cx), py - float(cy))
        for px, py in points
        for cx, cy in centres
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--original", required=True, help="零延迟天花板 preds 目录（current-frame 管线输出）")
    ap.add_argument("--replay", required=True, help="外挂交付档 preds 目录")
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    orig_dir, rep_dir = Path(args.original), Path(args.replay)
    seqs = sorted(p.stem for p in rep_dir.glob("*.json") if p.stem != "summary")
    delta, tol = int(args.delta), float(args.hit_tol_px)

    totals = {"gt": 0, "visible": 0, "orig_hit": 0, "orig_hit_visible": 0,
              "rep_hit": 0, "rep_hit_visible": 0, "source_visible": 0}
    per_seq: dict[str, dict] = {}

    for seq in seqs:
        truth = load_ground_truth(labels_root, seq, IMG_W, IMG_H)
        orig = load_per_frame(orig_dir, seq)
        rep = load_per_frame(rep_dir, seq)
        row = {"gt": 0, "visible": 0, "orig_hit": 0, "orig_hit_visible": 0,
               "rep_hit": 0, "rep_hit_visible": 0, "source_visible": 0}
        for frame, boxes in truth.items():
            if boxes.size == 0:
                continue
            centres = boxes[:, :2].astype(np.float64)
            row["gt"] += 1
            totals["gt"] += 1

            current_visible = len(orig.get(frame, [])) > 0
            source_visible = len(orig.get(frame - delta, [])) > 0
            if current_visible:
                row["visible"] += 1
                totals["visible"] += 1
            if source_visible:
                row["source_visible"] += 1
                totals["source_visible"] += 1

            original_distance = nearest(orig.get(frame, []), centres)
            replay_distance = nearest(rep.get(frame, []), centres)
            if original_distance is not None and original_distance <= tol:
                row["orig_hit"] += 1
                totals["orig_hit"] += 1
                if current_visible:
                    row["orig_hit_visible"] += 1
                    totals["orig_hit_visible"] += 1
            if replay_distance is not None and replay_distance <= tol:
                row["rep_hit"] += 1
                totals["rep_hit"] += 1
                if current_visible:
                    row["rep_hit_visible"] += 1
                    totals["rep_hit_visible"] += 1
        per_seq[seq] = row

    visible = max(totals["visible"], 1)
    result = {
        "label": args.label,
        "delta": delta,
        "hit_tol_px": tol,
        "gt_frames": totals["gt"],
        "visible_frames": totals["visible"],
        "invisible_frames": totals["gt"] - totals["visible"],
        "source_visible_frames": totals["source_visible"],
        "original_hit_rate_5px": totals["orig_hit"] / totals["gt"] * 100.0,
        "replay_hit_rate_5px": totals["rep_hit"] / totals["gt"] * 100.0,
        "original_hit_rate_5px_visible": totals["orig_hit_visible"] / visible * 100.0,
        "replay_hit_rate_5px_visible": totals["rep_hit_visible"] / visible * 100.0,
        "headroom_visible_pp": (totals["orig_hit_visible"] - totals["rep_hit_visible"]) / visible * 100.0,
        "counters": totals,
        "per_sequence": per_seq,
    }
    print(f"=== 可见性天花板 {args.label} ===")
    print(f"GT 帧 = {totals['gt']}，当前帧可见 = {totals['visible']} "
          f"({totals['visible'] / max(totals['gt'], 1) * 100:.2f}%)，"
          f"不可见 = {totals['gt'] - totals['visible']}")
    print(f"  original 全量 hit@5px  = {result['original_hit_rate_5px']:.4f}%   ← 不是天花板（自己也踩不可见帧）")
    print(f"  replay   全量 hit@5px  = {result['replay_hit_rate_5px']:.4f}%")
    print(f"  original 可见帧内 hit  = {result['original_hit_rate_5px_visible']:.4f}%   ← 真实天花板")
    print(f"  replay   可见帧内 hit  = {result['replay_hit_rate_5px_visible']:.4f}%")
    print(f"  ⇒ 可见帧内的真实 headroom = {result['headroom_visible_pp']:+.4f} pp")
    for seq, row in per_seq.items():
        vis = max(row["visible"], 1)
        print(
            f"  {seq}: gt={row['gt']} 可见={row['visible']} "
            f"orig可见内={row['orig_hit_visible'] / vis * 100:6.2f}% "
            f"rep可见内={row['rep_hit_visible'] / vis * 100:6.2f}%"
        )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SUCCESS] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
