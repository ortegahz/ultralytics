#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
🔴🔴🔴 【已作废 · 禁止运行】DEPRECATED — DO NOT RUN. 逻辑不成立，整版数据作废。

**作废原因（2026-11-20 用户纠偏）**：本脚本拿**反无人机检测器**去跑**没有无人机**的测试视频。
- 好片测出的「连续航迹」其实是**背景虚警**（检测器把楼角/路灯/树梢当成疑似弱小目标，
  地物纹理清晰 ⇒ 运动补偿后被时空关联当成连续轨迹）⇒ 测到的不是「目标可跟踪性」，是**虚警率**；
- 坏片「零航迹」只是因为**糊图把背景锐利边缘抹平、连虚警都触发不了**（0.167%）
  ⇒ 「零航迹」**不能**证明「画质差 ⇒ 跟不住」；
- 真实业务交互是**操作员在画面任意位置点一下 → SOT 锁住局部特征块**，
  「检测器报不报候选」与「用户点选后能不能跟住」是两码事。

⇒ **本脚本及其全部产物（好片有航迹 / 坏片零航迹）不得写进任何正式报告。**

**正解已落地（用这两套，不要用本脚本）**：
- 第八节「48 宫格模拟点击 + 双向闭环」：``manu/video_quality/vq_grid*.py``
- 第九节「点选式 SOT 实测 + 画质合格线」：``manu/video_quality/vq_sot_link.py`` /
  ``vq_sot_link_driver.py`` / ``vq_sot_compare.py`` / ``vq_sot_verify.py``
  （汇报件 ``manu/video_quality/report/REPORT_视频质量指标.md`` 第九节；
  主档 ``manu/memory/memory.md`` §284；专题档 §八）

本文件仅作历史留档，保留以便追溯当时的口径与代码。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

# Quality metrics carried over from the report, with their "higher is better" sign.
QUALITY_METRICS = {
    "lap_var": +1,
    "edge_density": +1,
    "global_contrast": +1,
    "entropy": +1,
    "blockiness": -1,
    "illum_jump": -1,
    "stripe_idx_row": -1,
}
# Grid (spatial trackability) metrics, higher is better.
GRID_METRICS = ("ugr", "ugr_raw", "tgr_raw", "tgr_dog", "det_ratio", "tgr_dog_best_row", "tgr_dog_best_block")
# Tracking outcomes, higher is better.
TRACK_METRICS = ("track_frame_ratio", "cand_frame_ratio", "track_cover_ratio", "track_count_norm", "long_track_ratio")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def auc_good_higher(good: list[float], bad: list[float]) -> float:
    """P(good > bad) with ties counted as one half, computed by the rank-sum identity."""
    if not good or not bad:
        return float("nan")
    combined = sorted([(value, 0) for value in good] + [(value, 1) for value in bad])
    ranks: dict[int, float] = {}
    index = 0
    while index < len(combined):
        stop = index
        while stop + 1 < len(combined) and combined[stop + 1][0] == combined[index][0]:
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average_rank
        index = stop + 1
    rank_sum = sum(ranks[position] for position, (_, label) in enumerate(combined) if label == 0)
    n_good, n_bad = len(good), len(bad)
    return (rank_sum - n_good * (n_good + 1) / 2.0) / (n_good * n_bad)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation without a scipy dependency (average ranks for ties)."""
    if len(xs) < 3 or len(ys) < 3:
        return float("nan")

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        ranks = [0.0] * len(values)
        position = 0
        while position < len(order):
            stop = position
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[position]]:
                stop += 1
            average = (position + stop) / 2.0 + 1.0
            for index in range(position, stop + 1):
                ranks[order[index]] = average
            position = stop + 1
        return ranks

    rank_x, rank_y = rank(xs), rank(ys)
    mean_x, mean_y = statistics.fmean(rank_x), statistics.fmean(rank_y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    denominator = (
        sum((a - mean_x) ** 2 for a in rank_x) ** 0.5 * sum((b - mean_y) ** 2 for b in rank_y) ** 0.5
    )
    return numerator / denominator if denominator else float("nan")


def load_segments(track_dir: Path, warmup: int) -> tuple[dict[str, dict], list[dict]]:
    """Aggregate the per-segment records into one record per clip, per threshold arm."""
    per_clip: dict[str, dict] = {}
    segments: list[dict] = []
    for path in sorted(track_dir.glob("*.json")):
        if path.name == "run_manifest.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        summary = record["summary"]
        clip = summary["sequence"]
        start = int(summary["start_frame"])
        frames = int(summary["frames"])
        window = int(summary["window_frames"])

        entry = per_clip.setdefault(
            clip,
            {
                "video": summary["video"],
                "group": summary["group"],
                "kind": summary["kind"],
                "modality": summary["modality"],
                "fps": summary["fps"],
                "frames_total": 0,
                "segments": 0,
                "arms": {},
            },
        )
        entry["frames_total"] += frames
        entry["segments"] += 1

        for arm, payload in record["arms"].items():
            arm_entry = entry["arms"].setdefault(
                arm, {"frames": 0, "frames_with_tracks": 0, "frames_with_candidates": 0,
                      "tracks_total": 0, "observations": 0, "long_tracks": 0,
                      "windows": {}, "track_lengths": []}
            )
            per_frame = payload.get("per_frame_tracks")
            candidates = record.get("per_frame_candidates") or []
            if not per_frame:
                # Older records may not carry the per-frame array; fall back to the window rows.
                for row in payload.get("windows", []):
                    arm_entry["windows"][row[0]] = row
                arm_entry["frames"] += frames
                arm_entry["frames_with_tracks"] += payload["frames_with_tracks"]
                arm_entry["frames_with_candidates"] += payload.get("frames_with_candidates", 0)
                arm_entry["tracks_total"] += payload["tracks_total"]
                arm_entry["observations"] += payload.get("track_observations_total", 0)
                arm_entry["track_lengths"].extend(payload.get("track_lengths", []))
                continue

            # The first frames of a segment are GMC/tracker warm-up: the median window needs 21
            # frames of history and the confirmation needs 7 consecutive observations.  Those frames
            # are dropped so a segment boundary cannot be mistaken for a detection failure.
            usable = max(0, frames - warmup)
            arm_entry["frames"] += usable
            arm_entry["frames_with_tracks"] += sum(1 for value in per_frame[warmup:] if value > 0)
            if candidates:
                arm_entry["frames_with_candidates"] += sum(1 for value in candidates[warmup:] if value > 0)
            arm_entry["tracks_total"] += payload["tracks_total"]
            arm_entry["observations"] += payload.get("track_observations_total", 0)
            arm_entry["track_lengths"].extend(payload.get("track_lengths", []))

            # Re-window the per-frame counts on the GLOBAL window grid.
            for local in range(0, frames, window):
                stop = min(frames, local + window)
                first = max(local, warmup)
                if stop <= first:
                    continue
                global_index = (start + local) // window
                chunk = per_frame[first:stop]
                arm_entry["windows"][global_index] = [
                    global_index,
                    stop - first,
                    sum(1 for value in chunk if value > 0),
                    int(sum(chunk)),
                ]

        segments.append({"clip": clip, "start": start, "frames": frames, "file": path.name})

    for entry in per_clip.values():
        for arm_entry in entry["arms"].values():
            arm_entry["track_frame_ratio"] = (
                arm_entry["frames_with_tracks"] / arm_entry["frames"] if arm_entry["frames"] else 0.0
            )
            arm_entry["cand_frame_ratio"] = (
                arm_entry["frames_with_candidates"] / arm_entry["frames"] if arm_entry["frames"] else 0.0
            )
            arm_entry["track_cover_ratio"] = (
                arm_entry["observations"] / arm_entry["frames"] if arm_entry["frames"] else 0.0
            )
            lengths = arm_entry["track_lengths"]
            arm_entry["median_track_len"] = statistics.median(lengths) if lengths else 0
            arm_entry["max_track_len"] = max(lengths) if lengths else 0
            arm_entry["n_tracks"] = len(lengths)
    return per_clip, segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Join image quality with the real tracker outcome")
    parser.add_argument("--track-dir", required=True)
    parser.add_argument("--grid-summary", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--arm", default="0.2", help="Threshold arm to report (default 0.2)")
    parser.add_argument("--warmup", type=int, default=30, help="Warm-up frames dropped per segment")
    cli = parser.parse_args()

    report_dir = Path(cli.report_dir)
    per_clip, segments = load_segments(Path(cli.track_dir), cli.warmup)
    grid_rows = {row["file"]: row for row in read_csv(Path(cli.grid_summary))}
    window_rows = read_csv(Path(cli.windows))

    # ---------------- per-clip table ----------------
    clip_rows: list[dict[str, Any]] = []
    for clip, entry in sorted(per_clip.items(), key=lambda item: (item[1]["group"], item[0])):
        arm = entry["arms"].get(cli.arm) or next(iter(entry["arms"].values()))
        video = entry["video"]
        grid = grid_rows.get(video, {})
        row: dict[str, Any] = {
            "file": video,
            "clip": clip,
            "group": entry["group"],
            "kind": entry["kind"],
            "modality": entry["modality"],
            "segments": entry["segments"],
            "frames_usable": arm["frames"],
            "frames_with_tracks": arm["frames_with_tracks"],
            "frames_with_candidates": arm["frames_with_candidates"],
            "track_frame_ratio": round(arm["track_frame_ratio"], 6),
            "cand_frame_ratio": round(arm["cand_frame_ratio"], 6),
            "track_cover_ratio": round(arm["track_cover_ratio"], 6),
            "n_tracks": arm["n_tracks"],
            "median_track_len": arm["median_track_len"],
            "max_track_len": arm["max_track_len"],
            "track_observations": arm["observations"],
        }
        for name in QUALITY_METRICS:
            row[name] = grid.get(name, "")
        for name in GRID_METRICS:
            row[name] = grid.get(name, "")
        row["tier"] = grid.get("tier", "")
        clip_rows.append(row)

    clip_fields = list(clip_rows[0].keys()) if clip_rows else []
    write_csv(report_dir / "track_summary.csv", clip_rows, clip_fields)

    # ---------------- per-window table ----------------
    windows_by_clip: dict[str, list[dict[str, str]]] = {}
    for row in window_rows:
        windows_by_clip.setdefault(row["file"], []).append(row)

    window_out: list[dict[str, Any]] = []
    for clip, entry in sorted(per_clip.items()):
        arm = entry["arms"].get(cli.arm) or next(iter(entry["arms"].values()))
        video = entry["video"]
        for index, payload in sorted(arm["windows"].items()):
            quality = None
            for row in windows_by_clip.get(video, []):
                if int(row["window"]) == index:
                    quality = row
                    break
            if quality is None:
                continue
            record: dict[str, Any] = {
                "file": video,
                "group": entry["group"],
                "modality": entry["modality"],
                "window": index,
                "frames": payload[1],
                "frames_with_track": payload[2],
                "n_track": payload[3],
                "tracked": int(payload[2] > 0),
            }
            for name in QUALITY_METRICS:
                record[name] = quality.get(name, "")
            window_out.append(record)
    if window_out:
        write_csv(report_dir / "track_windows.csv", window_out, list(window_out[0].keys()))

    # ---------------- discrimination ----------------
    good = [row for row in clip_rows if row["group"] == "good"]
    bad = [row for row in clip_rows if row["group"] == "bad"]
    discrimination: list[dict[str, Any]] = []
    predictors = list(QUALITY_METRICS) + list(GRID_METRICS)
    for predictor in predictors:
        for outcome in TRACK_METRICS:
            good_x = [float(row[predictor]) for row in good if row[predictor] not in ("", None)]
            good_y = [float(row[outcome]) for row in good if row[predictor] not in ("", None)]
            bad_x = [float(row[predictor]) for row in bad if row[predictor] not in ("", None)]
            bad_y = [float(row[outcome]) for row in bad if row[predictor] not in ("", None)]
            if not good_x or not bad_x:
                continue
            discrimination.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "auc_good_higher": round(auc_good_higher(good_x, bad_x), 6),
                    "auc_good_vs_track": round(
                        auc_good_higher(
                            [float(row[outcome]) for row in good],
                            [float(row[outcome]) for row in bad],
                        ),
                        6,
                    ),
                    "spearman": round(spearman(good_x + bad_x, good_y + bad_y), 6),
                    "good_median": round(statistics.median(good_x), 6),
                    "bad_median": round(statistics.median(bad_x), 6),
                    "good_min": min(good_x),
                    "good_max": max(good_x),
                    "bad_min": min(bad_x),
                    "bad_max": max(bad_x),
                    "n_good": len(good_x),
                    "n_bad": len(bad_x),
                }
            )
    if discrimination:
        write_csv(
            report_dir / "track_discrimination.csv",
            discrimination,
            list(discrimination[0].keys()),
        )

    # ---------------- actionable floors ----------------
    thresholds: list[dict[str, Any]] = []
    for predictor in predictors:
        values_good = [float(row[predictor]) for row in good if row[predictor] not in ("", None)]
        values_bad = [float(row[predictor]) for row in bad if row[predictor] not in ("", None)]
        if not values_good or not values_bad:
            continue
        sign = QUALITY_METRICS.get(predictor, +1)
        safe_line = min(values_good) if sign > 0 else max(values_good)
        red_line = max(values_bad) if sign > 0 else min(values_bad)
        thresholds.append(
            {
                "predictor": predictor,
                "direction": "higher_is_better" if sign > 0 else "lower_is_better",
                "safe_line": round(safe_line, 6),
                "red_line": round(red_line, 6),
                "margin": round(abs(safe_line - red_line), 6),
                "zero_overlap": bool(
                    (sign > 0 and min(values_good) > max(values_bad))
                    or (sign < 0 and max(values_good) < min(values_bad))
                ),
                "good_min": min(values_good),
                "good_max": max(values_good),
                "bad_min": min(values_bad),
                "bad_max": max(values_bad),
                "good_median": round(statistics.median(values_good), 6),
                "bad_median": round(statistics.median(values_bad), 6),
            }
        )
    if thresholds:
        write_csv(report_dir / "track_thresholds.csv", thresholds, list(thresholds[0].keys()))

    # ---------------- console summary ----------------
    print(f"[SUMMARY] clips={len(clip_rows)} good={len(good)} bad={len(bad)} segments={len(segments)} arm={cli.arm}")
    print(f"[SUMMARY] window rows={len(window_out)}")
    for row in clip_rows:
        print(
            f"  {row['group']:4} {row['file'][:34]:36} frames={row['frames_usable']:6} "
            f"cand_ratio={row['cand_frame_ratio']:.4f} track_ratio={row['track_frame_ratio']:.4f} "
            f"tracks={row['n_tracks']:4} med_len={row['median_track_len']:4} "
            f"ugr={row['ugr']} lap_var={row['lap_var']}"
        )
    print("\n[THRESHOLDS]")
    for row in sorted(thresholds, key=lambda item: -item["margin"]):
        print(
            f"  {row['predictor']:20} {row['direction']:18} safe>={row['safe_line']:<12} "
            f"red<={row['red_line']:<12} zero_overlap={row['zero_overlap']}"
        )


if __name__ == "__main__":
    main()
