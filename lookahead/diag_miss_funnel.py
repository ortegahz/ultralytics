#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**失手漏斗**：把「GT 帧 → 有没有上报 → 位置对不对」逐层拆开，定位最大损失层。

每层回答：
  L1 ``no_source``      ：源帧 t−Δ 上**管线没有检测** ⇒ 重推根本无从下手；
  L2 ``no_gt_at_src``   ：源帧上**真值本身没有目标**（目标刚出现/刚消失）；
  L3 ``no_output``      ：能重推，但管线**主动不上报**（弃权 / 回退链峰值门 / 重推失败）；
  L4 ``wrong_pos``      ：上报了，但离 GT 中心 > 5 px（按 CLE 分桶）。

口径与 ``diag_gap_attrib.py`` 完全一致（hit@5px = 上报点到最近 GT 中心 ≤ 5 px）。

用法::

    python manu/tracking/lookahead/diag_miss_funnel.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --original runs/lookahead/final/preds_off \
        --replay   runs/lookahead/final/final8/preds_delivery \
        --delta 8 --out-json runs/lookahead/final/final8/diag_miss_funnel.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

IMG_W, IMG_H = 640, 512


def load_gt(labels_root: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    for f in sorted(labels_root.glob(f"{seq}__*.txt")):
        idx = int(f.stem.split("__")[-1])
        pts = []
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                pts.append((float(parts[1]) * IMG_W, float(parts[2]) * IMG_H))
        if pts:
            out[idx] = pts
    return out


def load_per_frame(preds_dir: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    path = preds_dir / f"{seq}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8")).get("per_frame", {})
    return {
        int(k): [(float(d["pos"][0]), float(d["pos"][1])) for d in (dets or [])]
        for k, dets in payload.items()
    }


def nearest(points, targets):
    if not points:
        return None
    return min(math.hypot(px - gx, py - gy) for px, py in points for gx, gy in targets)


def bucket(value: float | None) -> str:
    if value is None:
        return "no_output"
    if value <= 5.0:
        return "hit<=5"
    if value <= 10.0:
        return "(5,10]"
    if value <= 20.0:
        return "(10,20]"
    if value <= 50.0:
        return "(20,50]"
    return ">50"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--original", required=True, help="零延迟管线输出 preds 目录（判源帧有没有检测）")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    orig_dir, rep_dir = Path(args.original), Path(args.replay)
    delta = int(args.delta)
    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()] or sorted(
        p.stem for p in rep_dir.glob("*.json") if p.stem != "summary"
    )

    per_seq: dict[str, dict] = {}
    grand = Counter()
    for seq in seqs:
        gt = load_gt(labels_root, seq)
        orig = load_per_frame(orig_dir, seq)
        rep = load_per_frame(rep_dir, seq)
        row: Counter = Counter()
        for t, gts in gt.items():
            row["gt"] += 1
            src = t - delta
            if src < 0:
                row["no_source"] += 1
                continue
            if not orig.get(src):
                row["no_src_det"] += 1
                continue
            if not gt.get(src):
                row["no_gt_at_src"] += 1
                continue
            err = nearest(rep.get(t, []), gts)
            row[bucket(err)] += 1
            if err is None:
                row["no_output"] += 1
            elif err > args.hit_tol_px:
                row["wrong_pos"] += 1
            else:
                row["hit"] += 1
        per_seq[seq] = dict(row)
        grand.update(row)

    total = max(1, grand["gt"])
    print(f"=== 失手漏斗 {args.label}（Δ={delta}）===")
    print(f"{'层':22s} {'帧数':>9s} {'占比':>8s}")
    for key, name in (
        ("gt", "GT 帧（目标存在）"),
        ("no_source", "L1 源帧不存在"),
        ("no_src_det", "L2 源帧无检测"),
        ("no_gt_at_src", "L3 源帧无真值目标"),
        ("no_output", "L4 能重推但未上报"),
        ("wrong_pos", "L5 上报了但位置错"),
        ("hit", "命中"),
    ):
        print(f"{name:22s} {grand.get(key, 0):9d} {grand.get(key, 0) / total * 100:7.2f}%")
    print("--- 位置误差分桶（仅 L5 上报了但位置错的帧）---")
    for key in ("(5,10]", "(10,20]", "(20,50]", ">50"):
        print(f"  {key:>10s} {grand.get(key, 0):8d}")
    print("--- 逐序：命中 / GT ---")
    for seq, row in per_seq.items():
        g = max(1, row.get("gt", 0))
        print(
            f"  {seq:34s} hit {row.get('hit', 0) / g * 100:6.2f}%  "
            f"未上报 {row.get('no_output', 0):5d}  源帧无检测 {row.get('no_src_det', 0):5d}  "
            f"源帧无真值 {row.get('no_gt_at_src', 0):5d}"
        )
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"label": args.label, "overall": dict(grand), "per_sequence": per_seq},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"[SAVED] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
