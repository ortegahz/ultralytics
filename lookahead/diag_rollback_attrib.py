#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**回退机制（源帧无检测 ⇒ 再往回退重推）的帧级收益/代价归属**。

回答三个问题：

1. 回退多上报了多少帧？这些帧里**目标在不在**（GT 有没有）？
2. 目标在 ⇒ 真赚（TP 增益）；目标不在 ⇒ 纯误报（FAR 代价，转台会跟到鬼影）；
3. 两臂都上报的帧里，回退有没有把命中改成脱靶（抖动/跑飞）？

口径与 `extrapolator.servo_metrics` 一致：命中 = 上报点到**最近 GT 中心** ≤ `--hit-tol-px`；
GT = `<labels-root>/<SEQ>__<frame>.txt`（平铺，归一化 0~1，空文件 = 显式无目标）。

用法：
    python manu/tracking/lookahead/diag_rollback_attrib.py \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --base  runs/lookahead/final/final8/preds_delivery \
        --other runs/lookahead/final/rb8/preds_rb8_mp0.50 \
        --out-json runs/lookahead/final/final8/diag_rollback_attrib.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
import sys
from pathlib import Path

import numpy as np

IMG_W, IMG_H = 640, 512

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from manu.evaluation.eval_multiscale_cache_th import count_matches, metrics  # noqa: E402


def load_gt(labels_root: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    """帧号 → [(cx, cy), ...]（像素坐标）。"""
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
    return {int(k): [(d["pos"][0], d["pos"][1]) for d in (dets or [])] for k, dets in pf.items()}


def load_per_frame_full(preds_dir: Path, seq: str) -> dict[int, list[dict]]:
    """帧号 → 上报点（含 score），用于业务域 inbox 判据。"""
    p = preds_dir / f"{seq}.json"
    if not p.is_file():
        return {}
    pf = json.loads(p.read_text(encoding="utf-8")).get("per_frame", {})
    return {int(k): list(dets or []) for k, dets in pf.items()}


def load_gt_boxes(labels_root: Path, seq: str) -> dict[int, np.ndarray]:
    """帧号 → (N,4) 的 (cx, cy, w, h) 像素框。"""
    out: dict[int, np.ndarray] = {}
    for f in sorted(labels_root.glob(f"{seq}__*.txt")):
        idx = int(f.stem.split("__")[-1])
        boxes = []
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            boxes.append(
                [
                    float(parts[1]) * IMG_W,
                    float(parts[2]) * IMG_H,
                    float(parts[3]) * IMG_W,
                    float(parts[4]) * IMG_H,
                ]
            )
        if boxes:
            out[idx] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return out


def inbox_metrics(per_frame: dict[int, list[dict]], gt_boxes: dict[int, np.ndarray], num_frames: int) -> dict:
    """业务域 inbox 判据下的 TP/FP/FN/FAR（与 `lookahead_longquan.evaluate_biz` 同一套计数）。"""
    frames = []
    for frame in range(num_frames):
        boxes = gt_boxes.get(frame)
        dets = per_frame.get(frame, [])
        if dets:
            ranked = sorted(dets, key=lambda det: -float(det.get("score", 1.0)))
            points = np.asarray([det["pos"] for det in ranked], dtype=np.float32).reshape(-1, 2)
            scores = np.asarray([float(det.get("score", 1.0)) for det in ranked], dtype=np.float32)
        else:
            points, scores = None, None
        frames.append((boxes if boxes is not None and len(boxes) else None, points, scores))
    return metrics(count_matches(frames, 0.0, 8.0, "inbox"), num_frames)


def inbox_summary(counts: Counter) -> dict:
    """把累计计数汇总成 F1 / FAR（FAR 分母 = 视频总帧数）。"""
    tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
    frames = int(counts["frames"])
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "det": int(counts["det"]),
        "frames": frames,
        "f1": (2 * tp / denominator) if denominator else 0.0,
        "far_per_frame": (fp / frames) if frames else 0.0,
    }


def hit_and_cle(pts, gts, tol):
    if not pts or not gts:
        return False, None
    best = min(math.hypot(px - gx, py - gy) for px, py in pts for gx, gy in gts)
    return best <= tol, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--base", required=True, help="对照档（冠军）preds 目录")
    ap.add_argument("--other", required=True, help="被测档（回退）preds 目录")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    base_dir, other_dir = Path(args.base), Path(args.other)
    sequences = sorted(p.stem for p in base_dir.glob("*.json") if (other_dir / f"{p.stem}.json").is_file())
    if not sequences:
        raise SystemExit("没有可对比的序列")

    agg = {
        "frames_with_gt": 0,
        "base_only_report": 0,
        "other_only_report": 0,
        "other_only_with_gt": 0,
        "other_only_without_gt": 0,
        "other_only_hit": 0,
        "other_only_miss": 0,
        "base_only_hit": 0,
        "base_only_miss": 0,
        "both_hit_to_hit": 0,
        "both_hit_to_miss": 0,
        "both_miss_to_hit": 0,
        "both_miss_to_miss": 0,
    }
    biz = {"base": Counter(), "other": Counter()}
    per_seq = {}
    for seq in sequences:
        gt = load_gt(labels_root, seq)
        base = load_per_frame(base_dir, seq)
        other = load_per_frame(other_dir, seq)
        gt_boxes = load_gt_boxes(labels_root, seq)
        base_full = load_per_frame_full(base_dir, seq)
        other_full = load_per_frame_full(other_dir, seq)
        num_frames = max(
            max(base_full or {0: []}), max(other_full or {0: []}), max(gt_boxes or {0: None})
        ) + 1
        for name, full in (("base", base_full), ("other", other_full)):
            row = inbox_metrics(full, gt_boxes, num_frames)
            biz[name].update({key: int(row[key]) for key in ("tp", "fp", "fn", "det")})
            biz[name]["frames"] += int(num_frames)
        row = {k: 0 for k in agg}
        row["frames_with_gt"] = len(gt)
        for frame in sorted(set(base) | set(other)):
            b_pts, o_pts = base.get(frame, []), other.get(frame, [])
            gts = gt.get(frame, [])
            b_hit, _ = hit_and_cle(b_pts, gts, args.hit_tol_px)
            o_hit, _ = hit_and_cle(o_pts, gts, args.hit_tol_px)
            if b_pts and not o_pts:
                row["base_only_report"] += 1
                row["base_only_hit" if b_hit else "base_only_miss"] += 1
            elif o_pts and not b_pts:
                row["other_only_report"] += 1
                if gts:
                    row["other_only_with_gt"] += 1
                    row["other_only_hit" if o_hit else "other_only_miss"] += 1
                else:
                    row["other_only_without_gt"] += 1
            elif b_pts and o_pts:
                if b_hit and o_hit:
                    row["both_hit_to_hit"] += 1
                elif b_hit and not o_hit:
                    row["both_hit_to_miss"] += 1
                elif o_hit:
                    row["both_miss_to_hit"] += 1
                else:
                    row["both_miss_to_miss"] += 1
        for key, value in row.items():
            agg[key] += value
        per_seq[seq] = row

    payload = {
        "label": args.label,
        "base": str(base_dir),
        "other": str(other_dir),
        "hit_tol_px": args.hit_tol_px,
        "sequences": sequences,
        "overall": agg,
        "inbox": {name: inbox_summary(counts) for name, counts in biz.items()},
        "per_sequence": per_seq,
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WRITE] {out}")
    print(f"[LABEL] {args.label}")
    for key, value in agg.items():
        print(f"  {key:<26} {value}")
    for name, row in payload["inbox"].items():
        print(
            f"  [inbox/{name}] tp={row['tp']} fp={row['fp']} fn={row['fn']} "
            f"F1={row['f1'] * 100:.4f}% FAR={row['far_per_frame']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
