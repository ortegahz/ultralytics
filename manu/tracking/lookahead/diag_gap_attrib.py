#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**插件精度损失的逐帧归属**：把「原逻辑（零延迟）」与「快速前向推演外挂」的每一帧对账。

回答三个问题（这正是「跟原来的代码对比计算精度损失」要落到帧级的样子）：

1. **净差距**：外挂比原逻辑**多丢**了多少帧（= hit@5px 的净 pp 差）；
2. **损失的构成**：外挂丢的帧里，外挂侧的 CLE 落在哪个区间
   （(5,10] 说明**只差一点点**，是可修的近靶；>100 是彻底跑飞的 walk-off）；
3. **不可修的部分**：两臂**都**丢的帧 —— 那是检测器/管线本身的能力上限，**插件无责**。

口径与 `extrapolator.servo_metrics` 完全一致（不发明新口径）：
- GT：`<labels-root>/<SEQ>__<frame>.txt`，每行 `class cx cy w h`（**归一化** 0~1），空文件 = 显式无目标；
- 命中：上报点到**最近 GT 中心**的欧氏距离 ≤ `hit_tol_px`（默认 5.0）；
- 两臂的 `per_frame` 以**视频帧号**为键。

用法：
    python manu/tracking/lookahead/diag_gap_attrib.py \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --original runs/lookahead/ref/preds_original_v3 \
        --replay  runs/lookahead/final/final8/preds_delivery \
        --out-json runs/lookahead/final/final8/diag_gap_attrib.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

IMG_W, IMG_H = 640, 512


def load_gt(labels_root: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    """帧号 → [(cx, cy), ...]（像素坐标，已乘 img_w/img_h）。"""
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


def hit_and_cle(pts: list[tuple[float, float]], gts: list[tuple[float, float]], tol: float):
    """返回 (是否命中, 最近 GT 的 CLE)；无上报点 ⇒ (False, None)。"""
    if not pts:
        return False, None
    best = min(math.hypot(px - gx, py - gy) for px, py in pts for gx, gy in gts)
    return best <= tol, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--original", required=True, help="原逻辑（零延迟天花板）preds 目录")
    ap.add_argument("--replay", required=True, help="外挂交付档 preds 目录")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--label", default="", help="写进产物的一行标签")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    orig_dir, rep_dir = Path(args.original), Path(args.replay)
    seqs = sorted(p.stem for p in rep_dir.glob("*.json") if p.stem != "summary")

    grand = Counter()
    per_seq = {}
    KEYS = ("both_hit", "orig_hit_replay_miss", "orig_miss_replay_hit", "both_miss")
    gap_cle: list[float] = []
    gap_no_coord = 0
    for seq in seqs:
        gt = load_gt(labels_root, seq)
        o = load_per_frame(orig_dir, seq)
        r = load_per_frame(rep_dir, seq)
        c = Counter()
        for fidx, gts in gt.items():
            oh, _ = hit_and_cle(o.get(fidx, []), gts, args.hit_tol_px)
            rh, rc = hit_and_cle(r.get(fidx, []), gts, args.hit_tol_px)
            if oh and rh:
                c["both_hit"] += 1
            elif oh and not rh:
                c["orig_hit_replay_miss"] += 1
                if rc is None:
                    gap_no_coord += 1
                else:
                    gap_cle.append(rc)
            elif rh and not oh:
                c["orig_miss_replay_hit"] += 1
            else:
                c["both_miss"] += 1
        row = {k: int(c.get(k, 0)) for k in KEYS}
        row["gt_frames"] = sum(row[k] for k in KEYS)
        row["replay_hit_rate"] = (row["both_hit"] + row["orig_miss_replay_hit"]) / max(1, row["gt_frames"])
        row["orig_hit_rate"] = (row["both_hit"] + row["orig_hit_replay_miss"]) / max(1, row["gt_frames"])
        per_seq[seq] = row
        grand.update({k: row[k] for k in KEYS})

    tot = max(1, sum(r["gt_frames"] for r in per_seq.values()))
    buckets = Counter()
    for v in gap_cle:
        if v <= 10:
            buckets["(5,10]"] += 1
        elif v <= 20:
            buckets["(10,20]"] += 1
        elif v <= 50:
            buckets["(20,50]"] += 1
        elif v <= 100:
            buckets["(50,100]"] += 1
        else:
            buckets[">100"] += 1

    gap = grand.get("orig_miss_replay_hit", 0) - grand.get("orig_hit_replay_miss", 0)
    result = {
        "label": args.label,
        "hit_tol_px": args.hit_tol_px,
        "gt_frames": tot,
        "counts": {
            **{k: int(grand.get(k, 0)) for k in KEYS},
        },
        "replay_hit_rate": (grand.get("both_hit", 0) + grand.get("orig_miss_replay_hit", 0)) / tot,
        "orig_hit_rate": (grand.get("both_hit", 0) + grand.get("orig_hit_replay_miss", 0)) / tot,
        "net_gap_frames": gap,
        "net_gap_pp": gap / tot * 100.0,
        "loss_cle_buckets": dict(buckets),
        "loss_frames_without_coord": gap_no_coord,
        "per_sequence": per_seq,
    }

    print(f"=== 插件精度损失逐帧归属（hit@<={args.hit_tol_px:g}px） {args.label} ===")
    print(f"{'序列':<26} {'GT帧':>7} {'双命中':>8} {'原中/外挂失':>11} {'原失/外挂中':>11} {'双失':>7} {'外挂':>8} {'原逻辑':>8}")
    for seq, c in per_seq.items():
        print(f"{seq:<26} {c['gt_frames']:>7} {c['both_hit']:>8} {c['orig_hit_replay_miss']:>11} "
              f"{c['orig_miss_replay_hit']:>11} {c['both_miss']:>7} "
              f"{c['replay_hit_rate']*100:>7.2f}% {c['orig_hit_rate']*100:>7.2f}%")
    print(f"\n合计 GT 帧 {tot}")
    for k in KEYS:
        print(f"   {k:<22} {grand.get(k,0):>7}  ({grand.get(k,0)/tot*100:.2f}%)")
    print(f"   ★ 净差距 = {gap} 帧 = {gap/tot*100:+.4f} pp")
    print(f"   损失帧的 CLE 分桶：{dict(buckets)}  + 无坐标 {gap_no_coord} 帧")
    print(f"   ★ 不可修（两臂都丢，检测器/管线上限）= {grand.get('both_miss',0)} 帧 ({grand.get('both_miss',0)/tot*100:.2f}%)")
    print(f"   ★ 可修（仅外挂丢）= {grand.get('orig_hit_replay_miss',0)} 帧 ({grand.get('orig_hit_replay_miss',0)/tot*100:.2f}%)")

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[SAVED] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
