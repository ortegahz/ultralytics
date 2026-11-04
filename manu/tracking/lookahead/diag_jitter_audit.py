#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**伺服抖动口径审计**：把 `servo_metrics` 的 `max_jitter` 拆成「真实跳变」与「口径伪影」。

为什么需要这个审计：`servo_metrics` 的抖动是**二阶差分**，而
① 轨迹点**只在「有 GT 的帧」累计**（`box_array is None → continue` 在轨迹累计之前）；
② 帧号用的是 **records 列表下标**，不是真实视频帧号 —— 而 records 列表的
`frame_index` 可能有**空洞**（缓存里没这一帧 ⇒ records 里就没有这一条）。
于是「列表下标相邻」≠「视频帧相邻」：目标在空洞期间真的移动了几十像素，
指标却把它当成**一帧**的速度差 ⇒ 造出几百像素的**假抖动**。

本脚本按**真实帧号**重算两件事：
* ``max_jitter_true``：二阶差分时把速度按真实帧间隔归一（单位 px/帧），
  并在**空洞处断开**（不跨空洞做二阶差分）⇒ 只反映真实的速度突变；
* ``max_step_px``：**逐帧单步位移**的最大值（物理上转台真正看到的量）。

用法：
    python manu/tracking/lookahead/diag_jitter_audit.py \
        --arms-json runs/lookahead/final/rb8/rb8.json \
        --preds-dir runs/lookahead/final/rb8/preds_rb8 \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --out-json runs/lookahead/final/rb8/diag_jitter_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from manu.tracking.lookahead.lookahead_longquan import build_records  # noqa: E402

IMG_W, IMG_H = 640, 512


def load_gt_frames(labels_root: Path, seq: str) -> set[int]:
    """有 GT 的真实帧号集合。"""
    out = set()
    for f in sorted(labels_root.glob(f"{seq}__*.txt")):
        idx = int(f.stem.split("__")[-1])
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                out.add(idx)
                break
    return out


def audit_sequence(seq: str, arms_json: Path, preds_dir: Path, labels_root: Path, cache_root: Path) -> dict:
    records = build_records(cache_root, labels_root, seq, 0.06, 0.0, 3, 5, 0.3)
    records = sorted(records, key=lambda record: int(record["frame_index"]))
    gt_frames = {
        int(record["frame_index"])
        for record in records
        if np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4).size
    }
    payload = json.loads((preds_dir / f"{seq}.json").read_text(encoding="utf-8"))
    per_frame = payload.get("per_frame", {})

    # 轨迹点：按 **真实帧号** 累计，且只取有 GT 的帧（与 servo_metrics 的口径一致）
    tracks: dict[int, dict[int, np.ndarray]] = {}
    for record in records:
        frame = int(record["frame_index"])
        if frame not in gt_frames:
            continue
        for det in per_frame.get(str(frame), []):
            tracks.setdefault(int(det.get("track_id", -1)), {})[frame] = np.asarray(
                det["pos"], dtype=np.float64
            )

    jitter_raw = []       # 不归一、不断开（= 指标口径）
    jitter_true = []      # 归一 + 空洞断开（= 真实速度突变）
    max_step = 0.0        # 真实**单帧**位移（相邻视频帧）的最大值
    max_step_spanning = 0.0
    gaps = 0
    for per_track in tracks.values():
        order = sorted(per_track)
        for i in range(1, len(order)):
            gap = order[i] - order[i - 1]
            step = float(np.linalg.norm(per_track[order[i]] - per_track[order[i - 1]]))
            if gap == 1:
                max_step = max(max_step, step)
            else:
                max_step_spanning = max(max_step_spanning, step)
        for i in range(2, len(order)):
            f0, f1, f2 = order[i - 2], order[i - 1], order[i]
            v1 = per_track[f1] - per_track[f0]
            v2 = per_track[f2] - per_track[f1]
            dt1 = max(1, f1 - f0)
            dt2 = max(1, f2 - f1)
            jitter_raw.append(float(np.linalg.norm(v2 - v1)))
            if f1 - f0 > 1 or f2 - f1 > 1:
                gaps += 1
                continue  # 跨空洞 ⇒ 不做二阶差分（真实帧号不连续）
            jitter_true.append(float(np.linalg.norm(v2 / dt2 - v1 / dt1)))

    json_row = {}
    data = json.loads(arms_json.read_text(encoding="utf-8"))
    arms = data.get("arms", data)
    for arm, arm_payload in arms.items():
        servo = arm_payload.get("servo", {})
        row = servo.get("per_sequence", {}).get(seq)
        if row:
            json_row[arm] = {
                "max_jitter": float(row.get("max_jitter", 0.0)),
                "p99_jitter": float(row.get("p99_jitter", 0.0)),
                "jitter_samples": int(row.get("jitter_samples", 0)),
                "tracks": int(row.get("tracks", 0)),
            }

    return {
        "sequence": seq,
        "records": len(records),
        "gt_frames": len(gt_frames),
        "frame_index_holes": (max(int(r["frame_index"]) for r in records) + 1 - len(records))
        if records
        else 0,
        "json": json_row,
        "audit": {
            "max_jitter_metric": max(jitter_raw) if jitter_raw else 0.0,
            "max_jitter_true": max(jitter_true) if jitter_true else 0.0,
            "p99_jitter_true": float(np.percentile(jitter_true, 99)) if jitter_true else 0.0,
            "max_step_px": max_step,
            "max_step_spanning_px": max_step_spanning,
            "gap_samples": gaps,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-json", required=True)
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    arms_json = Path(args.arms_json)
    preds_dir = Path(args.preds_dir)
    labels_root = Path(args.labels_root)
    cache_root = Path(args.cache_root)
    # preds 目录里还有 write_summary 写的 summary.json，它不是序列。
    sequences = sorted(path.stem for path in preds_dir.glob("*.json") if path.stem != "summary")
    rows = []
    for seq in sequences:
        row = audit_sequence(seq, arms_json, preds_dir, labels_root, cache_root)
        rows.append(row)
        replay = row["json"].get("replay", {})
        audit = row["audit"]
        print(
            f"  {seq:<34} metric_max_jitter={replay.get('max_jitter', 0.0):8.2f} "
            f"true_max={audit['max_jitter_true']:7.2f} step1f={audit['max_step_px']:6.2f} "
            f"step_span={audit['max_step_spanning_px']:7.2f} "
            f"holes={row['frame_index_holes']:5d} gap_samples={audit['gap_samples']:5d}"
        )
    payload = {
        "label": args.label,
        "arms_json": str(arms_json),
        "preds_dir": str(preds_dir),
        "per_sequence": rows,
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WRITE] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
