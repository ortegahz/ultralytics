#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**剩余缺口的帧级分桶**：把「外挂为什么还差 2.58 pp」拆成可行动的桶。

口径（与 `diag_oracle_bound.py` 完全一致）：
  * GT：`<labels-root>/<SEQ>__<frame>.txt`，`class cx cy w h`（归一化 0~1）；
  * 命中：上报点到**最近 GT 中心**的欧氏距离 ≤ `--hit-tol-px`（默认 5.0）；
  * `original` 臂 = 零延迟天花板（当前帧检测）；`replay` 臂 = 外挂交付档；
  * **锚点** = `original` 臂在 **t−Δ** 帧的输出（外挂的链原点就是它）。

分桶口径：
  ① 「好锚点」= 源帧 t−Δ 的检测落在**源帧 GT** 的 5 px 内（= 锚点 oracle 的定义）；
  ② 在好锚点的帧里，把「外挂失手」按 **CLE** 分桶（5-10 / 10-20 / 20-40 / >40 / 不上报）；
  ③ 同时统计**目标 8 帧真位移**（GT(t) − GT(t−Δ)）—— 用来判断「一次大匹配够不够得着」。

用法：
    python manu/tracking/lookahead/diag_gap_buckets.py \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --original runs/lookahead/ref/preds_original_v3 \
        --replay  runs/lookahead/final/final8/preds_delivery \
        --delta 8 --out-json runs/lookahead/final/final8/diag_gap_buckets.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

IMG_W, IMG_H = 640, 512
BUCKETS = ("no_report", "5-10", "10-20", "20-40", ">40")


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
    p = preds_dir / f"{seq}.json"
    if not p.is_file():
        return {}
    pf = json.loads(p.read_text(encoding="utf-8")).get("per_frame", {})
    return {int(k): [(d["pos"][0], d["pos"][1]) for d in (dets or [])] for k, dets in pf.items()}


def near(a, g):
    return min(math.hypot(px - gx, py - gy) for px, py in a for gx, gy in g) if a else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--radius-px", type=float, default=32.0, help="xcorr 一次大匹配的搜索半径（用于对照）")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    orig_dir, rep_dir = Path(args.original), Path(args.replay)
    seqs = sorted(p.stem for p in rep_dir.glob("*.json") if p.stem != "summary")
    d, tol = args.delta, args.hit_tol_px

    per_seq = {}
    tot = {"gt": 0, "good_anchor": 0, "good_hit": 0, "good_miss": 0}
    agg_buckets = {b: 0 for b in BUCKETS}
    agg_disp_hit, agg_disp_miss = [], []

    for seq in seqs:
        gt = load_gt(labels_root, seq)
        o = load_per_frame(orig_dir, seq)
        r = load_per_frame(rep_dir, seq)
        row = {"gt": 0, "good_anchor": 0, "good_hit": 0, "good_miss": 0, "buckets": {b: 0 for b in BUCKETS},
               "disp_hit": [], "disp_miss": [], "anchor_err": [], "chain_err": []}
        for t, gts in gt.items():
            tot["gt"] += 1
            row["gt"] += 1
            src = o.get(t - d, [])
            if not src:
                continue
            se = near(src, gt.get(t - d, []) or gts)
            if se is None or se > tol:
                continue
            # 好锚点
            row["good_anchor"] += 1
            tot["good_anchor"] += 1
            row["anchor_err"].append(se)
            rd = near(r.get(t, []), gts)
            gtd = None
            if gt.get(t - d):
                gtd = math.hypot(gts[0][0] - gt[t - d][0][0], gts[0][1] - gt[t - d][0][1])
            if rd is not None and rd <= tol:
                row["good_hit"] += 1
                tot["good_hit"] += 1
                if gtd is not None:
                    row["disp_hit"].append(gtd)
                continue
            row["good_miss"] += 1
            tot["good_miss"] += 1
            if gtd is not None:
                row["disp_miss"].append(gtd)
            if rd is None:
                b = "no_report"
            elif rd <= 10:
                b = "5-10"
            elif rd <= 20:
                b = "10-20"
            elif rd <= 40:
                b = "20-40"
            else:
                b = ">40"
            row["buckets"][b] += 1
            agg_buckets[b] += 1
            if rd is not None:
                row["chain_err"].append(rd)
        per_seq[seq] = row
        agg_disp_hit += row["disp_hit"]
        agg_disp_miss += row["disp_miss"]

    def pct(v, p):
        return statistics.median(v) if not v else sorted(v)[int(len(v) * p / 100)]

    def stat(v):
        return {
            "n": len(v),
            "p50": round(statistics.median(v), 3) if v else None,
            "p90": round(pct(v, 90), 3) if v else None,
            "max": round(max(v), 3) if v else None,
            "over_radius_ratio": round(sum(1 for x in v if x > args.radius_px) / len(v), 6) if v else None,
        }

    res = {
        "label": args.label,
        "delta": d,
        "hit_tol_px": tol,
        "radius_px": args.radius_px,
        "overall": {
            "gt_frames": tot["gt"],
            "good_anchor_frames": tot["good_anchor"],
            "good_anchor_hit": tot["good_hit"],
            "good_anchor_miss": tot["good_miss"],
            "good_anchor_hit_rate": tot["good_hit"] / tot["good_anchor"] if tot["good_anchor"] else None,
            "miss_buckets": agg_buckets,
            "true_disp_hit": stat(agg_disp_hit),
            "true_disp_miss": stat(agg_disp_miss),
        },
        "per_sequence": per_seq,
    }
    o = res["overall"]
    print(f"=== 剩余缺口分桶 {args.label} ===")
    print(f"GT 帧数                    = {o['gt_frames']}")
    print(f"好锚点帧（源帧检测在 5px 内）  = {o['good_anchor_frames']}")
    print(f"  其中外挂命中              = {o['good_anchor_hit']} ({o['good_anchor_hit_rate']*100:.2f}%)")
    print(f"  其中外挂失手              = {o['good_anchor_miss']}")
    print(f"失手 CLE 分桶              = {o['miss_buckets']}")
    print(f"真位移(8帧) 命中帧 p50/p90  = {o['true_disp_hit']['p50']} / {o['true_disp_hit']['p90']}"
          f"  (>半径 {args.radius_px:g} 的比例 {o['true_disp_hit']['over_radius_ratio']})")
    print(f"真位移(8帧) 失手帧 p50/p90  = {o['true_disp_miss']['p50']} / {o['true_disp_miss']['p90']}"
          f"  (>半径 {args.radius_px:g} 的比例 {o['true_disp_miss']['over_radius_ratio']})")
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[SAVED] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
