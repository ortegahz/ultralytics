#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挑一段最能说明「老逻辑 vs 外挂」差距的 OSD 窗口（三臂）。

三臂：stale（老逻辑，真吃 Δ=8 滞后）/ replay（外挂交付档）/ original（零延迟天花板）。
判据：``mean(cle[stale]) - mean(cle[replay])`` 最大 ⇒ 老臂飞得最远、外挂最稳的那一段。
辅助输出：同窗口下 replay 与 original 的差距（外挂离天花板还有多远）。

用法：
    python pick_osd_window_3arm.py --labels-root <GT> --seq <SEQ> \
        --stale <dir> --replay <dir> --original <dir> --out-json <out.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from manu.videos.render_pred_gt_osd_video import load_ground_truth  # noqa: E402


def load_arm(preds_dir: str, seq: str) -> dict[int, list[tuple[float, float, float]]]:
    data = json.loads((Path(preds_dir) / f"{seq}.json").read_text(encoding="utf-8"))
    per_frame: dict[int, list[tuple[float, float, float]]] = {}
    for key, rows in (data.get("per_frame") or {}).items():
        dets = []
        for row in rows or []:
            pos = row.get("pos")
            if pos is None:
                continue
            dets.append((float(pos[0]), float(pos[1]), float(row.get("score", 0.0))))
        per_frame[int(key)] = dets
    return per_frame


def centre_errors(per_frame, truth) -> dict[int, float]:
    out: dict[int, float] = {}
    for frame, dets in per_frame.items():
        boxes = truth.get(frame)
        if boxes is None or not len(boxes) or not dets:
            continue
        centres = np.asarray(boxes, np.float64)[:, :2]
        best = None
        for x, y, _s in dets:
            d = float(np.sqrt(((centres - np.asarray([x, y])) ** 2).sum(axis=1)).min())
            best = d if best is None else min(best, d)
        out[frame] = best
    return out


def runs(frames: list[int]) -> list[list[int]]:
    out, cur = [], []
    for f in frames:
        if cur and f != cur[-1] + 1:
            out.append(cur)
            cur = []
        cur.append(f)
    if cur:
        out.append(cur)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--stale", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    arms = {"stale": args.stale, "replay": args.replay, "original": args.original}
    loaded = {k: load_arm(v, args.seq) for k, v in arms.items()}
    frame_set = set.intersection(*[set(v) for v in loaded.values()])
    truth = load_ground_truth(Path(args.labels_root), args.seq, args.width, args.height, frame_set)
    errs = {k: centre_errors(v, truth) for k, v in loaded.items()}
    frames = sorted(set.intersection(*[set(v) for v in errs.values()]))

    best = None
    for run in runs(frames):
        if len(run) < args.window:
            continue
        step = max(1, args.window // 10)
        for i in range(0, len(run) - args.window + 1, step):
            win = run[i:i + args.window]
            sep = float(np.mean([errs["stale"][f] for f in win]) -
                        np.mean([errs["replay"][f] for f in win]))
            if best is None or sep > best[0]:
                best = (sep, win[0], win[-1])
    if best is None:
        raise SystemExit("no window found")
    sep, start, end = best

    stats = {}
    for name, e in errs.items():
        vals = np.asarray([e[f] for f in range(start, end + 1) if f in e], dtype=float)
        stats[name] = dict(
            cle_mean=float(vals.mean()), cle_p50=float(np.median(vals)),
            cle_p90=float(np.percentile(vals, 90)), cle_max=float(vals.max()),
            hit_5px=float((vals <= 5.0).mean()), n=int(vals.size),
        )
    doc = dict(
        seq=args.seq, start=int(start), end=int(end), limit=int(end - start + 1),
        criterion="stale_vs_replay", separation_score=sep,
        replay_gap_to_ceiling=float(stats["original"]["cle_mean"] - stats["replay"]["cle_mean"]),
        per_arm=stats,
    )
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(doc, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
