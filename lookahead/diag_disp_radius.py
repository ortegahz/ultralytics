#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**失手帧的位移-半径结构**（§32 的下一刀：位移估计到底在哪儿断掉）。

§31/§32 的结论链：
  * 锚点本身几乎完美（误差 p50 = 1.50 px / p99 = 4.54 px）⇒ 不需要「推翻锚点」；
  * 剩余误差全在**位移估计**（残差 p50 = 5.26 px / p90 = 42.08 px）；
  * chain × ROI 前推上限 0/8/16/24/48 **完全打平** ⇒ 「ROI 前推卡死」这个假设被证伪。

本脚本回答**下一个**问题：失手帧的**真实位移**（|GT(t) − GT(t−Δ)|）与**搜索半径**的关系。
  * 若失手帧集中在「真实位移 > 搜索半径」⇒ 该换的是**搜索策略**（多级/金字塔搜索）；
  * 若真实位移远小于半径却仍失手 ⇒ 该换的是**匹配判据**（歧义/外观漂移）。

⚠️ 口径（**必须与官方一致**，否则数字不可信）：
  * GT 用官方 :func:`manu.evaluation.eval_multiscale_cache_th.load_ground_truth`
    （原生像素 (cx,cy,w,h)，丢重复行，空文件跳过）；
  * 上报点用 preds 的 ``per_frame``（key = **视频帧号**，与官方 ``frame_payload`` 一致）；
  * 命中判据 = 官方 :func:`servo_metrics` 的口径：取**离最近 GT 中心最近的那个上报点**，
    距离 ≤ 5 px 记命中；分母 = 有 GT 的帧数。
  （2026-11-04 核对：手算 hit@5px 与 ``arms_*.json`` 的官方值**逐位一致**。）

用法::

    python manu/tracking/lookahead/diag_disp_radius.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --original runs/lookahead/final/preds_off \
        --replay   runs/lookahead/final/final8/preds_delivery \
        --delta 8 --radius 32 --out-json runs/lookahead/final/final8/diag_disp_radius.json
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
    """preds 的 ``per_frame``（key = 视频帧号）→ {帧号: [(x, y), ...]}。"""
    path = preds_dir / f"{seq}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8")).get("per_frame", {})
    return {
        int(key): [(float(d["pos"][0]), float(d["pos"][1])) for d in (dets or [])]
        for key, dets in payload.items()
    }


def nearest(points, centres):
    """离最近 GT 中心最近的那个上报点的距离（官方 ``servo_metrics`` 口径）。"""
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
    ap.add_argument("--original", required=True, help="零延迟天花板 preds 目录（t 帧输出即锚点 A）")
    ap.add_argument("--replay", required=True, help="外挂交付档 preds 目录")
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--radius", type=float, default=32.0, help="对照用的搜索半径（px）")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    orig_dir, rep_dir = Path(args.original), Path(args.replay)
    seqs = sorted(p.stem for p in rep_dir.glob("*.json") if p.stem != "summary")
    delta, tol, radius = int(args.delta), float(args.hit_tol_px), float(args.radius)

    # 失手帧按「真实位移是否超出搜索半径」交叉分桶
    beyond = {"within": 0, "beyond": 0}
    disp_miss: list[float] = []
    disp_hit: list[float] = []
    seq_rows: dict[str, dict] = {}
    totals = {"gt": 0, "report": 0, "hit": 0, "miss_within": 0, "miss_beyond": 0}
    # 位移分桶（全帧）
    disp_buckets: dict[str, int] = {}

    def bucket(value: float) -> str:
        for hi, name in ((5.0, "0-5"), (10.0, "5-10"), (20.0, "10-20"), (40.0, "20-40"), (80.0, "40-80")):
            if value <= hi:
                return name
        return ">80"

    for seq in seqs:
        truth = {
            index: boxes[:, :2].astype(np.float64)
            for index, boxes in load_ground_truth(labels_root, seq, IMG_W, IMG_H).items()
        }
        orig = load_per_frame(orig_dir, seq)
        rep = load_per_frame(rep_dir, seq)
        row = {"gt": 0, "report": 0, "hit": 0, "miss_within": 0, "miss_beyond": 0}
        for t, centres in truth.items():
            totals["gt"] += 1
            row["gt"] += 1
            src_centres = truth.get(t - delta)
            disp = None
            if src_centres is not None:
                # 真实位移：GT(t) 与 GT(t−Δ) 各自最近中心之差（用最近的那对）
                disp = float(
                    np.linalg.norm(centres.mean(axis=0) - src_centres.mean(axis=0))
                )
                disp_buckets[bucket(disp)] = disp_buckets.get(bucket(disp), 0) + 1
            report = rep.get(t, [])
            if report:
                totals["report"] += 1
                row["report"] += 1
            distance = nearest(report, centres)
            hit = distance is not None and distance <= tol
            if hit:
                totals["hit"] += 1
                row["hit"] += 1
                if disp is not None:
                    disp_hit.append(disp)
                continue
            if disp is None:
                continue
            disp_miss.append(disp)
            if disp > radius:
                beyond["beyond"] += 1
                totals["miss_beyond"] += 1
                row["miss_beyond"] += 1
            else:
                beyond["within"] += 1
                totals["miss_within"] += 1
                row["miss_within"] += 1
        seq_rows[seq] = row

    def pct(values, q):
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    miss_total = len(disp_miss)
    result = {
        "label": args.label,
        "delta": delta,
        "hit_tol_px": tol,
        "search_radius_px": radius,
        "gt_frames": totals["gt"],
        "report_frames": totals["report"],
        "hit_frames": totals["hit"],
        "hit_rate_5px": totals["hit"] / totals["gt"] if totals["gt"] else 0.0,
        "miss_frames": miss_total,
        "miss_within_radius": beyond["within"],
        "miss_beyond_radius": beyond["beyond"],
        "miss_beyond_radius_pct": beyond["beyond"] / miss_total * 100.0 if miss_total else 0.0,
        "disp_p50_hit": pct(disp_hit, 50),
        "disp_p90_hit": pct(disp_hit, 90),
        "disp_p50_miss": pct(disp_miss, 50),
        "disp_p90_miss": pct(disp_miss, 90),
        "disp_buckets_all_frames": disp_buckets,
        "per_sequence": seq_rows,
    }
    print(f"=== 位移-半径结构分解 {args.label} ===")
    print(
        f"GT={totals['gt']} 有上报={totals['report']} 命中={totals['hit']} "
        f"hit@5px={result['hit_rate_5px'] * 100:.4f}%"
    )
    print(
        f"失手帧={miss_total}：真实位移 **≤ {radius:g}px** 的 {beyond['within']} 帧 "
        f"({beyond['within'] / miss_total * 100 if miss_total else 0:.2f}%) / "
        f"**> {radius:g}px** 的 {beyond['beyond']} 帧 "
        f"({result['miss_beyond_radius_pct']:.2f}%)"
    )
    print(
        f"命中帧真实位移：p50={result['disp_p50_hit']:.2f} p90={result['disp_p90_hit']:.2f} | "
        f"失手帧：p50={result['disp_p50_miss']:.2f} p90={result['disp_p90_miss']:.2f}"
    )
    print(f"全帧真实位移分桶：{disp_buckets}")
    for seq, row in seq_rows.items():
        print(f"  {seq}: gt={row['gt']} hit={row['hit']} miss_within={row['miss_within']} miss_beyond={row['miss_beyond']}")
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SUCCESS] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
