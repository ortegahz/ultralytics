#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**锚点继承误差 vs 位移估计残差** 的逐帧分解（§31 的下一刀）。

背景（§31 诊断）：
  外挂上报坐标 = ``A(f) + D_pattern(f→t)``（A = 源帧 t−Δ 的管线锚点）。
  * ``D_pattern`` = 模板匹配的**图形位移**（目标本体运动）；
  * 输出与真值的误差 = ``|e(f) + (D_pattern − D_true)|``，
    其中 ``e(f) = A(f) − GT(f)`` 是**锚点自身的误差**（被原样继承）。
  ⇒ 「位移估计残差」与「锚点继承误差」必须分开算，否则会误判该修哪一个。

本脚本逐帧算三件事（全部从既有产物读，不跑模型）：
  1. ``e(f)`` = |A(f) − GT(f)|：锚点误差（用 `original` 臂在 f 帧的输出当 A）；
  2. ``o(t)`` = |报告(t) − GT(t)|：外挂的输出误差；
  3. ``d_true`` = |GT(t) − GT(f)|：8 帧真实位移。

然后把「锚点好（e(f) ≤ 5 px）却失手」��帧按 e(f) / o(t) 分桶，
回答**决定性问题**：
  * 若 o(t) ≈ e(f)  ⇒ 误差是**锚点继承**的 ⇒ 该修「锚点重定位」；
  * 若 o(t) ≫ e(f)  ⇒ 误差是**位移估计残差** ⇒ 该修「位移估计」。

用法::

    python manu/tracking/lookahead/diag_anchor_residual.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --original runs/lookahead/final/preds_off \
        --replay   runs/lookahead/final/preds_delivery \
        --delta 8 --out-json runs/lookahead/final/final8/diag_anchor_residual.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

IMG_W, IMG_H = 640, 512


def load_gt(labels_root: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    for f in sorted(labels_root.glob(f"{seq}__*.txt")):
        idx = int(f.stem.split("__")[-1])
        pts = []
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            pts.append((float(parts[1]) * IMG_W, float(parts[2]) * IMG_H))
        if pts:
            out[idx] = pts
    return out


def load_per_frame(preds_dir: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    p = preds_dir / f"{seq}.json"
    if not p.is_file():
        return {}
    pf = json.loads(p.read_text(encoding="utf-8")).get("per_frame", {})
    return {
        int(k): [(d["pos"][0], d["pos"][1]) for d in (dets or [])]
        for k, dets in pf.items()
    }


def nearest(pts, gts):
    if not pts:
        return None
    return min(math.hypot(px - gx, py - gy) for px, py in pts for gx, gy in gts)


def bucket(v: float) -> str:
    for hi, name in ((5.0, "<=5"), (10.0, "5-10"), (20.0, "10-20"), (40.0, "20-40")):
        if v <= hi:
            return name
    return ">40"


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
    d, tol = args.delta, args.hit_tol_px

    # 汇总桶：好锚点(e<=tol) 的帧，按 (e 分桶, o 分桶) 交叉
    cross: dict[str, dict[str, int]] = {}
    n_gt = n_report = n_hit = 0
    good = 0
    good_hit = 0
    resid_hist = []          # 好锚点且失手帧的 o - e
    anchor_hist = []         # 好锚点帧的 e
    seq_rows = {}
    agg = {
        "e_le_5": 0, "e_5_10": 0, "e_10_20": 0, "e_gt_20": 0,
        "miss_e_le_2": 0, "miss_e_2_5": 0, "miss_e_5_10": 0, "miss_e_gt_10": 0,
        "miss_o_le_5": 0, "miss_o_5_10": 0, "miss_o_gt_10": 0,
    }

    for seq in seqs:
        gt = load_gt(labels_root, seq)
        o_map = load_per_frame(orig_dir, seq)
        r_map = load_per_frame(rep_dir, seq)
        row = dict(gt=0, report=0, hit=0, good=0, good_hit=0, miss=0)
        for t, gts in gt.items():
            n_gt += 1
            row["gt"] += 1
            src = o_map.get(t - d, [])
            a_err = nearest(src, gt.get(t - d, []) or gts) if src else None
            rd = nearest(r_map.get(t, []), gts)
            if rd is not None:
                n_report += 1
                row["report"] += 1
            hit = rd is not None and rd <= tol
            if hit:
                n_hit += 1
                row["hit"] += 1
            if a_err is None or a_err > tol:
                continue
            good += 1
            row["good"] += 1
            anchor_hist.append(a_err)
            if hit:
                good_hit += 1
                row["good_hit"] += 1
                continue
            row["miss"] += 1
            eb, ob = bucket(a_err), bucket(rd if rd is not None else 999.0)
            cross.setdefault(eb, {}).setdefault(ob, 0)
            cross[eb][ob] += 1
            resid_hist.append((rd if rd is not None else 999.0) - a_err)
            if a_err <= 2.0:
                agg["miss_e_le_2"] += 1
            elif a_err <= 5.0:
                agg["miss_e_2_5"] += 1
            elif a_err <= 10.0:
                agg["miss_e_5_10"] += 1
            else:
                agg["miss_e_gt_10"] += 1
            if rd is None:
                agg["miss_o_le_5"] += 0
                agg["miss_o_5_10"] += 0
                agg["miss_o_gt_10"] += 1
            elif rd <= 5.0:
                agg["miss_o_le_5"] += 1
            elif rd <= 10.0:
                agg["miss_o_5_10"] += 1
            else:
                agg["miss_o_gt_10"] += 1
        seq_rows[seq] = row

    def pct(vals, q):
        if not vals:
            return 0.0
        v = sorted(vals)
        k = min(len(v) - 1, max(0, int(round(q / 100.0 * (len(v) - 1)))))
        return v[k]

    res = {
        "label": args.label,
        "delta": d,
        "hit_tol_px": tol,
        "gt_frames": n_gt,
        "report_frames": n_report,
        "hit_frames": n_hit,
        "good_anchor_frames": good,
        "good_anchor_hit": good_hit,
        "good_anchor_miss": good - good_hit,
        "anchor_err_p50": pct(anchor_hist, 50),
        "anchor_err_p90": pct(anchor_hist, 90),
        "anchor_err_p99": pct(anchor_hist, 99),
        "residual_p50": pct(resid_hist, 50),
        "residual_p90": pct(resid_hist, 90),
        "cross_bucket": cross,
        "miss_attribution": agg,
        "per_sequence": seq_rows,
    }
    print(f"=== 锚点继承误差 vs 位移残差 分解 {args.label} ===")
    print(f"GT 帧 = {n_gt} / 有上报 = {n_report} / 命中 = {n_hit}")
    print(f"好锚点(e<={tol:g}px)帧 = {good}，其中命中 = {good_hit}，失手 = {good - good_hit}")
    print(f"好锚点的锚点误差：p50 = {res['anchor_err_p50']:.2f} px, p90 = {res['anchor_err_p90']:.2f}, p99 = {res['anchor_err_p99']:.2f}")
    print(f"★ 失手帧的 (输出误差 − 锚点误差) 残差：p50 = {res['residual_p50']:.2f} px, p90 = {res['residual_p90']:.2f}")
    print("--- 失手帧按「锚点误差 e」分桶（e 桶 -> 输出误差 o 桶的帧数）---")
    for eb in ("<=5", "5-10", "10-20", ">40"):
        if eb in cross:
            print(f"  e {eb:>6}: {cross[eb]}")
    print(f"--- 失手归因 ---")
    print(f"  锚点误差 e<=2 px : {agg['miss_e_le_2']}   （误差来自位移残差 ⇒ 该修位移估计）")
    print(f"  锚点误差 e 2-5 px: {agg['miss_e_2_5']}")
    print(f"  锚点误差 e 5-10  : {agg['miss_e_5_10']}   （误差来自锚点继承 ⇒ 该修锚点重定位）")
    print(f"  锚点误差 e>10 px : {agg['miss_e_gt_10']}")
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[SAVED] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
