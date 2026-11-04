#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**把剩余缺口拆成「锚点误差」与「位移估计误差」两个可独立兑现的 oracle**。

动机（§32 新证据）：
  * 锚点误差 p50 = 1.50 px / p99 = 4.54 px ⇒ 锚点**几乎完美**；
  * 位移残差 p50 = 5.26 px / p90 = 42.08 px ⇒ 误差**几乎全在位移估计**；
  * 但 §32 又证伪了「位移超出搜索半径」假设（74.48% 的失手帧真实位移 ≤ 32 px）⇒
    位移估计错在**匹配本身不准**，不是**搜不到**。

本脚本用**两个 oracle**把天花板量化，从而决定该修哪一个：

  * ``oracle_disp``：上报 = A(f) + (GT(t) − GT(f))
    ⇒ 误差 = |A(f) − GT(f)| = **纯锚点误差**。
    ⚠️ 这是「**纯平移分解**」，**不是**交付臂的真实天花板：交付臂是**逐帧重锁**
    （`_replay_detections` 每步重新配准），锚点偏了快核能重新拉回目标 ⇒
    真实天花板比这个数高。引用时必须写「纯平移分解」，不能写「上限」。
  * ``oracle_anchor``：上报 = GT(f) + D̂(f→t)，D̂ = 交付档**实测位移**
    （= report(t) − report(t−Δ)，即臂内部真正用的那个位移量）
    ⇒ 误差 = **纯位移估计误差**。
    ⚠️ D̂ **不能**取「report(t) − GT(f)」：那会代数恒等于 report(t) ⇒ 退化成
    replay 本身（同义反复）。这是 2026-11-04 审计实测抓到的退化。
  * 交付臂的**真实**天花板见 ``diag_visible_ceiling.py``（按「目标当前帧是否可见」
    分层比较两臂的真实输出，不构造 oracle ⇒ 无此问题）。

口径：**全部与官方 :func:`servo_metrics` 逐位一致**（2026-11-04 核对：
手算 hit@5px 与 ``arms_*.json`` 的官方值 diff = 0.0000）。

用法::

    python manu/tracking/lookahead/diag_oracle_split.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --original runs/lookahead/final/preds_off \
        --replay   runs/lookahead/final/final8/preds_delivery \
        --delta 8 --out-json runs/lookahead/final/final8/diag_oracle_split.json
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


def centres_of(boxes):
    """GT 框数组 → 中心点数组 (N, 2)（原生像素）。"""
    return np.asarray(boxes, dtype=np.float64)[:, :2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--original", required=True, help="零延迟天花板 preds 目录（t 帧输出即锚点 A）")
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

    counters = {"gt": 0, "replay_hit": 0, "oracle_disp_hit": 0, "oracle_anchor_hit": 0}
    per_seq: dict[str, dict] = {}

    for seq in seqs:
        truth = load_ground_truth(labels_root, seq, IMG_W, IMG_H)
        orig = load_per_frame(orig_dir, seq)
        rep = load_per_frame(rep_dir, seq)
        row = {"gt": 0, "replay_hit": 0, "oracle_disp_hit": 0, "oracle_anchor_hit": 0}
        for t, boxes in truth.items():
            if boxes.size == 0:
                continue
            centres = centres_of(boxes)
            counters["gt"] += 1
            row["gt"] += 1

            replay_pts = rep.get(t, [])
            if nearest(replay_pts, centres) is not None and nearest(replay_pts, centres) <= tol:
                counters["replay_hit"] += 1
                row["replay_hit"] += 1

            # ---- oracle_disp：锚点取源帧实测，位移取真值 ----
            src = orig.get(t - delta, [])
            src_truth = truth.get(t - delta)
            if src and src_truth is not None and src_truth.size:
                src_centres = centres_of(src_truth)
                # 源帧锚点取「离源帧 GT 最近的那个上报点」；位移取「对应 GT 中心之差」
                distances = np.linalg.norm(np.asarray(src, dtype=np.float64)[:, None, :] - src_centres[None], axis=2)
                best = np.unravel_index(int(np.argmin(distances)), distances.shape)
                anchor = np.asarray(src[best[0]], dtype=np.float64)
                disp = centres[int(best[1])] - src_centres[int(best[1])]
                cand = [(anchor + disp)]
                if nearest(cand, centres) is not None and nearest(cand, centres) <= tol:
                    counters["oracle_disp_hit"] += 1
                    row["oracle_disp_hit"] += 1

            # ---- oracle_anchor：锚点取真值，位移取交付档**实测位移** ----
            # ⚠️ 位移必须取「臂内部实测的位移」= report(t) − A(f)，**不能**取
            #    「report(t) − GT(f)」：后者代数上恒等于 report(t) ⇒ oracle 退化成
            #    replay 本身（同义反复）。审计代理 2026-11-04 实测确认了这个退化。
            # 这里 A(f) = 交付档在源帧 t−Δ 的**上报位置**（= 管线锚点 D(f) 经
            # 外挂修正后的坐标）；位移 = report(t) − report(t−Δ)。
            if replay_pts and src_truth is not None and src_truth.size:
                src_centres = centres_of(src_truth)
                src_report = rep.get(t - delta, [])
                if src_report:
                    src_report_pts = np.asarray(src_report, dtype=np.float64)
                    anchor_distances = np.linalg.norm(
                        src_report_pts[:, None, :] - src_centres[None], axis=2
                    )
                    anchor_index = int(np.argmin(anchor_distances.min(axis=1)))
                    displacement = np.asarray(replay_pts[0], dtype=np.float64) - src_report_pts[anchor_index]
                    cand = [src_centres[anchor_index] + displacement]
                    distance = nearest(cand, centres)
                    if distance is not None and distance <= tol:
                        counters["oracle_anchor_hit"] += 1
                        row["oracle_anchor_hit"] += 1
        per_seq[seq] = row

    total = max(counters["gt"], 1)
    result = {
        "label": args.label,
        "delta": delta,
        "hit_tol_px": tol,
        "gt_frames": counters["gt"],
        "replay_hit_rate_5px": counters["replay_hit"] / total * 100.0,
        "oracle_disp_hit_rate_5px": counters["oracle_disp_hit"] / total * 100.0,
        "oracle_anchor_hit_rate_5px": counters["oracle_anchor_hit"] / total * 100.0,
        "counters": counters,
        "per_sequence": per_seq,
    }
    print(f"=== oracle 拆分 {args.label} ===")
    print(f"GT 帧 = {counters['gt']}")
    print(f"  交付档（实测）        hit@5px = {result['replay_hit_rate_5px']:.4f}%")
    print(f"  oracle_disp（位移完美）hit@5px = {result['oracle_disp_hit_rate_5px']:.4f}%   ← 上限由锚点误差决定")
    print(f"  oracle_anchor（锚点完美）hit@5px = {result['oracle_anchor_hit_rate_5px']:.4f}%   ← 上限由位移估计决定")
    for seq, row in per_seq.items():
        gt = max(row["gt"], 1)
        print(
            f"  {seq}: replay={row['replay_hit'] / gt * 100:6.2f}% "
            f"oracle_disp={row['oracle_disp_hit'] / gt * 100:6.2f}% "
            f"oracle_anchor={row['oracle_anchor_hit'] / gt * 100:6.2f}%  (gt={row['gt']})"
        )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SUCCESS] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
