#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**锚点 oracle 天花板**：外挂（快速前向推演）在信息论上最多能做到多少 hit@5px。

机理（§28 的核心结论）：
  外挂上报坐标 = `A(f) + 位移估计(f→t)`，其中 `A(f)` 是**源帧 t−Δ 的管线输出**
  （= 检测器原始点，管线不做 KF 平滑，实测 98.7% 与 ms_cache 逐位相同）。
  位移估计残差 δ 的实测分布极小（p50 0.55~1.74 px、p90 1.31~3.13 px），
  所以外挂的误差 ≈ **源帧锚点误差 e(f)**，而不是重推链自身的漂移。
  ⇒ 若位移估计完美（δ=0），外挂的 hit@5px 上限 = 「源帧锚点本身就落在 GT 5 px 内」
    的比例 = **锚点 oracle**。
  ⇒ 超过锚点 oracle 就必须**重新定位**（用图像证据推翻管线锚点），而不是继续调重推。

口径与 `extrapolator.servo_metrics` 一致：
  * GT：`<labels-root>/<SEQ>__<frame>.txt`，`class cx cy w h`（**归一化** 0~1）；
  * 命中：上报点到**最近 GT 中心**的欧氏距离 ≤ `--hit-tol-px`（默认 5.0）；
  * `per_frame` 的键是**视频帧号**。

用法：
    python manu/tracking/lookahead/diag_oracle_bound.py \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --original runs/lookahead/ref/preds_original_v3 \
        --replay  runs/lookahead/final/final8/preds_delivery \
        --delta 8 --out-json runs/lookahead/final/final8/diag_oracle_bound.json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--original", required=True, help="原逻辑（零延迟天花板）preds 目录")
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

    tot = 0
    orig_hit = 0
    rep_hit = 0
    no_src = 0
    oracle = 0
    oracle_and_hit = 0
    per_seq = {}

    for seq in seqs:
        gt = load_gt(labels_root, seq)
        o = load_per_frame(orig_dir, seq)
        r = load_per_frame(rep_dir, seq)
        row = dict(gt_frames=0, orig_hit=0, replay_hit=0, no_source=0, anchor_oracle=0)
        for t, gts in gt.items():
            tot += 1
            row["gt_frames"] += 1
            od = nearest(o.get(t, []), gts)
            rd = nearest(r.get(t, []), gts)
            oh = od is not None and od <= tol
            rh = rd is not None and rd <= tol
            orig_hit += oh
            rep_hit += rh
            row["orig_hit"] += oh
            row["replay_hit"] += rh
            src = o.get(t - d, [])
            if not src:
                no_src += 1
                row["no_source"] += 1
                continue
            se = nearest(src, gt.get(t - d, []) or gts)
            if se is not None and se <= tol:
                oracle += 1
                oracle_and_hit += rh
                row["anchor_oracle"] += 1
        per_seq[seq] = row

    res = {
        "label": args.label,
        "delta": d,
        "hit_tol_px": tol,
        "gt_frames": tot,
        "original_hit_rate": orig_hit / tot,
        "replay_hit_rate": rep_hit / tot,
        "source_frame_no_detection_rate": no_src / tot,
        "anchor_oracle_hit_rate": oracle / tot,
        "replay_within_oracle": oracle_and_hit / tot,
        "gap_to_oracle_pp": (oracle - rep_hit) / tot * 100.0,
        "gap_to_ceiling_pp": (orig_hit - rep_hit) / tot * 100.0,
        "per_sequence": per_seq,
    }
    print(f"=== 锚点 oracle 天花板（Δ={d}, hit@{tol:g}px） {args.label} ===")
    print(f"GT 帧                = {tot}")
    print(f"original（零延迟天花板） = {orig_hit/tot*100:.3f}%")
    print(f"replay（外挂交付档）     = {rep_hit/tot*100:.3f}%")
    print(f"源帧(t−Δ)无任何输出      = {no_src/tot*100:.3f}%  ({no_src} 帧)")
    print(f"★ 锚点 oracle（δ=0 上界）= {oracle/tot*100:.3f}%  ({oracle} 帧)")
    print(f"   其中外挂已经命中       = {oracle_and_hit/tot*100:.3f}%")
    print(f"⇒ 外挂与 oracle 的差距    = {res['gap_to_oracle_pp']:+.4f} pp（{(oracle-rep_hit)} 帧）")
    print(f"⇒ 外挂与天花板的差距      = {res['gap_to_ceiling_pp']:+.4f} pp")
    print(f"⇒ 源帧无输出造成的天花板   = {100.0 - no_src/tot*100:.3f}%")
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[SAVED] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
