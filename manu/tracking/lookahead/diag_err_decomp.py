#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**误差分解**：把外挂重推的误差拆成「源帧锚点误差 e(f)」与「重推自身残差」。

机理假设（待验证）：xcorr 档位上报的绝对位置 = `anchor + 模板位移`，
其中模板是从**源帧锚点 anchor 附近**切出来的 ⇒ 上报坐标继承了 anchor 的定位误差 e(f)。
若 e(f) 大（管线自己在源帧就偏了），模板切到的是背景，配准必然歪 ⇒ 该帧必失。

本脚本量化：
1. 外挂丢帧中，源帧锚点误差 e(f) 的分布（e(f) ≫ 5px 的占比）；
2. 外挂丢帧里，源帧**有检测**的占比（有 ⇒ 仍有救；无 ⇒ 检测器/管线上限，插件无责）；
3. 两臂都丢的帧里，源帧有检测的比例（这部分是**潜在可回收**的上限）。

用法：
    python manu/tracking/lookahead/diag_err_decomp.py \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --original runs/lookahead/ref/preds_original_v3 \
        --replay  runs/lookahead/final/final8/preds_delivery \
        --delta 8 --out-json runs/lookahead/final/final8/diag_err_decomp.json
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
    out: dict[int, list[tuple[float, float]]] = {}
    for k, dets in pf.items():
        out[int(k)] = [(d["pos"][0], d["pos"][1]) for d in (dets or [])]
    return out


def nearest(pts, gts):
    if not pts:
        return None
    return min(math.hypot(px - gx, py - gy) for px, py in pts for gx, gy in gts)


def bucket(v: float | None) -> str:
    if v is None:
        return "no_coord"
    if v <= 5:
        return "<=5"
    if v <= 10:
        return "(5,10]"
    if v <= 20:
        return "(10,20]"
    if v <= 50:
        return "(20,50]"
    if v <= 100:
        return "(50,100]"
    return ">100"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    orig_dir, rep_dir = Path(args.original), Path(args.replay)
    seqs = sorted(p.stem for p in rep_dir.glob("*.json") if p.stem != "summary")
    d = args.delta
    tol = args.hit_tol_px

    loss_src_err = Counter()   # 外挂丢帧：源帧锚点误差 e(f) 分桶
    loss_src_err_ok = Counter()  # 外挂丢帧：源帧锚点误差是否 <=tol
    loss_src_hasdet = Counter()  # 外挂丢帧：源帧有无检测
    both_src_hasdet = Counter()  # 两臂都丢：源帧有无检测
    both_src_err = Counter()
    loss_total = 0
    both_total = 0
    seq_rows = {}
    gt_total = 0

    for seq in seqs:
        gt = load_gt(labels_root, seq)
        o = load_per_frame(orig_dir, seq)
        r = load_per_frame(rep_dir, seq)
        row = Counter()
        for t, gts in gt.items():
            gt_total += 1
            f = t - d
            od = nearest(o.get(t, []), gts)
            rd = nearest(r.get(t, []), gts)
            oh = od is not None and od <= tol
            rh = rd is not None and rd <= tol
            src_pts = o.get(f, [])
            src_has = bool(src_pts)
            src_err = nearest(src_pts, gt.get(f, [])) if src_pts else None
            if oh and not rh:
                loss_total += 1
                row["loss"] += 1
                loss_src_err[bucket(src_err)] += 1
                loss_src_hasdet["has" if src_has else "none"] += 1
                loss_src_err_ok["e<=tol" if (src_err is not None and src_err <= tol) else "e>tol"] += 1
            elif not oh and not rh:
                both_total += 1
                row["both"] += 1
                both_src_hasdet["has" if src_has else "none"] += 1
                both_src_err[bucket(src_err)] += 1
        seq_rows[seq] = dict(row)

    res = {
        "label": args.label,
        "delta": d,
        "hit_tol_px": tol,
        "gt_frames": gt_total,
        "plugin_loss_frames": loss_total,
        "both_miss_frames": both_total,
        "loss_src_anchor_err_bucket": dict(loss_src_err),
        "loss_src_anchor_err_ok": dict(loss_src_err_ok),
        "loss_src_has_detection": dict(loss_src_hasdet),
        "both_miss_src_has_detection": dict(both_src_hasdet),
        "both_miss_src_anchor_err_bucket": dict(both_src_err),
        "per_sequence": seq_rows,
    }
    print(f"=== 外挂误差分解（delta={d}, hit@{tol:g}px） {args.label} ===")
    print(f"GT 帧总数 {gt_total}；外挂独丢 {loss_total}；两臂都丢 {both_total}")
    print(f"[外挂丢帧] 源帧锚点误差 e(f) 分桶: {dict(loss_src_err)}")
    print(f"[外挂丢帧] e(f) 是否已越界    : {dict(loss_src_err_ok)}")
    print(f"[外挂丢帧] 源帧有无检测        : {dict(loss_src_hasdet)}")
    print(f"[两臂都丢] 源帧有无检测        : {dict(both_src_hasdet)}")
    print(f"[两臂都丢] 源帧锚点误差分桶     : {dict(both_src_err)}")
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[SAVED] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
