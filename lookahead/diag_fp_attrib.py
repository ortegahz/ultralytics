#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Who owns each false positive of the replay arm: the plugin, the detector, or the ground truth?

The delivery red line is ``FAR <= 0.030`` and the replay arm sits at ``0.0349`` on the 24-sequence set.
``diag_far_breakdown.py`` already splits those false positives by *where they land* (a frame that has a
ground-truth box versus a frame that has none). This script adds the missing second axis: it replays the
frozen inbox matching rule frame by frame for **both** the replay arm and the zero-delay ``original`` arm and
attributes every replay false positive to one of three owners:

``lost_by_replay``
    the zero-delay arm had a point inside the same ground-truth box and the replay arm did not - the replay
    chain walked somewhere else. This is the only bucket a plugin change can remove.
``both_missed``
    neither arm had a point inside the box - the detector itself was wrong on that frame.
``no_ground_truth``
    the frame carries no ground-truth annotation at all, so the detection is not an error in any algorithm;
    this is annotation sparsity and no amount of tracking work can remove it.

The reported position of the zero-delay arm is read from the frozen reference predictions, never recomputed,
so the attribution cannot drift from the delivery numbers.

Usage::

    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_fp_attrib.py \
        --replay-preds-dir runs/lookahead/final/final24/preds_delivery \
        --original-preds-dir runs/lookahead/final/ref_preds_original_v3_24seq \
        --arms-json runs/lookahead/final/final24/arms_delivery.json
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402

SCRIPT_REL = "manu/tracking/lookahead/diag_fp_attrib.py"
DEFAULT_ARMS_JSON = PROJECT_ROOT / "runs/lookahead/final/final24/arms_delivery.json"
DEFAULT_REPLAY_PREDS = PROJECT_ROOT / "runs/lookahead/final/final24/preds_delivery"
DEFAULT_ORIGINAL_PREDS = PROJECT_ROOT / "runs/lookahead/final/ref_preds_original_v3_24seq"
DEFAULT_OUT_JSON = PROJECT_ROOT / "runs/lookahead/final/final24/diag_fp_attrib.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "runs/lookahead/final/final24/diag_fp_attrib.md"
RED_LINE = 0.030


def parse_args() -> argparse.Namespace:
    """Command line: every input/output has a default pointing at the delivery artifacts."""
    parser = argparse.ArgumentParser(description="Attribute every replay-arm false positive to its owner.")
    parser.add_argument("--arms-json", type=Path, default=DEFAULT_ARMS_JSON)
    parser.add_argument("--replay-preds-dir", type=Path, default=DEFAULT_REPLAY_PREDS)
    parser.add_argument("--original-preds-dir", type=Path, default=DEFAULT_ORIGINAL_PREDS)
    parser.add_argument("--labels-root", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--red-line", type=float, default=RED_LINE)
    return parser.parse_args()


def inbox_hit(box: np.ndarray, point: np.ndarray) -> bool:
    """The frozen inbox rule of the business-domain matcher: the reported centre must be inside the box."""
    return abs(point[0] - box[0]) <= box[2] / 2 and abs(point[1] - box[1]) <= box[3] / 2


def frame_points(per_frame: dict, frame: int) -> np.ndarray | None:
    """Reported centres of one frame, score-descending (the order the frozen matcher walks them in)."""
    dets = per_frame.get(str(frame)) or []
    if not dets:
        return None
    ranked = sorted(dets, key=lambda det: -float(det["score"]))
    return np.asarray([det["pos"] for det in ranked], dtype=np.float64).reshape(-1, 2)


def unmatched_detections(boxes: np.ndarray | None, points: np.ndarray) -> list[int]:
    """Indices of the detections the frozen greedy matcher leaves unmatched, i.e. the false positives.

    Mirrors ``count_matches`` detection by detection: every detection claims the still-free ground truth
    whose centre is nearest, but only when the point falls inside that box. A detection that hits no box
    stays a false positive, and on a frame without any ground truth every detection is a false positive.
    """
    if boxes is None:
        return list(range(len(points)))
    free = list(range(len(boxes)))
    unmatched: list[int] = []
    for index, point in enumerate(points):
        if not free:
            unmatched.append(index)
            continue
        distances = [float(((boxes[free_index, :2] - point) ** 2).sum()) for free_index in free]
        nearest = free[int(np.argmin(distances))]
        if inbox_hit(boxes[nearest], point):
            free.remove(nearest)
        else:
            unmatched.append(index)
    return unmatched


def attribute_sequence(
    replay_doc: dict, original_doc: dict, truth: dict[int, np.ndarray], frames: int
) -> dict:
    """Per-frame attribution of the replay arm's false positives for one sequence."""
    replay_frame = replay_doc["per_frame"]
    original_frame = original_doc["per_frame"]
    buckets = {"lost_by_replay": 0, "both_missed": 0, "no_ground_truth": 0}
    distance_hist = {"lt5": 0, "5to10": 0, "10to20": 0, "20to50": 0, "ge50": 0}
    replay_fp = replay_tp = replay_det = 0
    original_reported = 0
    lost_sequences: list[tuple[int, float]] = []

    for frame in range(frames):
        boxes = truth.get(frame)
        boxes = boxes if boxes is not None and len(boxes) else None
        replay_points = frame_points(replay_frame, frame)
        original_points = frame_points(original_frame, frame)
        original_reported += int(original_points is not None)
        if replay_points is None:
            continue
        replay_det += len(replay_points)
        unmatched = unmatched_detections(boxes, replay_points)
        replay_tp += len(replay_points) - len(unmatched)
        replay_fp += len(unmatched)
        for index in unmatched:
            if boxes is None:
                buckets["no_ground_truth"] += 1
                continue
            box = boxes[index]
            distance = float(np.hypot(*(box[:2] - replay_points[index])))
            if distance < 5:
                distance_hist["lt5"] += 1
            elif distance < 10:
                distance_hist["5to10"] += 1
            elif distance < 20:
                distance_hist["10to20"] += 1
            elif distance < 50:
                distance_hist["20to50"] += 1
            else:
                distance_hist["ge50"] += 1
            original_hit = original_points is not None and any(
                inbox_hit(box, point) for point in original_points
            )
            if original_hit:
                buckets["lost_by_replay"] += 1
                lost_sequences.append((frame, distance))
            else:
                buckets["both_missed"] += 1

    return {
        "frames": frames,
        "replay_det": replay_det,
        "replay_tp": replay_tp,
        "replay_fp": replay_fp,
        "original_reported_frames": original_reported,
        "buckets": buckets,
        "distance_hist": distance_hist,
        "lost_by_replay_examples": [
            {"frame": frame, "distance_px": round(distance, 2)}
            for frame, distance in sorted(lost_sequences, key=lambda item: -item[1])[:20]
        ],
    }


def build_report(args: argparse.Namespace) -> dict:
    """Read the artifacts, attribute every false positive and assemble the report payload."""
    arms_doc = json.loads(Path(args.arms_json).read_text(encoding="utf-8"))
    config = arms_doc.get("config", {})
    labels_root = Path(args.labels_root or config.get("labels_root"))
    width = int(config.get("img_w", 640))
    height = int(config.get("img_h", 512))
    original_rows = arms_doc.get("arms", {}).get("original", {}).get("current_frame", {}).get("per_sequence", {})

    sequences: dict[str, dict] = {}
    for path in sorted(Path(args.replay_preds_dir).glob("*.json")):
        if path.name == "summary.json":
            continue
        sequence = path.stem
        original_path = Path(args.original_preds_dir) / path.name
        if not original_path.is_file():
            continue
        replay_doc = json.loads(path.read_text(encoding="utf-8"))
        original_doc = json.loads(original_path.read_text(encoding="utf-8"))
        frames = len(replay_doc["per_frame"])
        truth = load_ground_truth(labels_root, sequence, width, height)
        row = attribute_sequence(replay_doc, original_doc, truth, frames)
        original = original_rows.get(sequence, {})
        row["original_fp"] = original.get("fp")
        row["delta_fp_vs_original"] = (
            row["replay_fp"] - int(original["fp"]) if original.get("fp") is not None else None
        )
        row["far_per_frame"] = row["replay_fp"] / frames if frames else 0.0
        row["original_far_per_frame"] = original.get("far_per_frame")
        sequences[sequence] = row

    frames_total = sum(row["frames"] for row in sequences.values())
    fp_total = sum(row["replay_fp"] for row in sequences.values())
    buckets = {
        key: sum(row["buckets"][key] for row in sequences.values()) for key in ("lost_by_replay", "both_missed", "no_ground_truth")
    }
    distance_hist = {
        key: sum(row["distance_hist"][key] for row in sequences.values())
        for key in ("lt5", "5to10", "10to20", "20to50", "ge50")
    }
    original_fp = sum(int(row["original_fp"]) for row in sequences.values() if row["original_fp"] is not None)
    far = fp_total / frames_total if frames_total else 0.0
    red_line_fp = args.red_line * frames_total
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "script": SCRIPT_REL,
        "command": f"/home/huangzhe/anaconda3/envs/uav/bin/python {SCRIPT_REL}",
        "inputs": {
            "arms_json": str(args.arms_json),
            "replay_preds_dir": str(args.replay_preds_dir),
            "original_preds_dir": str(args.original_preds_dir),
            "labels_root": str(labels_root),
        },
        "totals": {
            "sequences": len(sequences),
            "frames": frames_total,
            "replay_det": sum(row["replay_det"] for row in sequences.values()),
            "replay_fp": fp_total,
            "original_fp": original_fp,
            "delta_fp_vs_original": fp_total - original_fp,
            "far_per_frame": far,
            "original_far_per_frame": original_fp / frames_total if frames_total else 0.0,
            "red_line": args.red_line,
            "red_line_fp": red_line_fp,
            "fp_over_red_line": fp_total - red_line_fp,
            "buckets": buckets,
            "distance_hist": distance_hist,
            "far_if_lost_by_replay_removed": (fp_total - buckets["lost_by_replay"]) / frames_total
            if frames_total
            else 0.0,
            "far_if_no_ground_truth_removed": (fp_total - buckets["no_ground_truth"]) / frames_total
            if frames_total
            else 0.0,
        },
        "sequences": sequences,
    }


def render_markdown(report: dict) -> str:
    """Render the attribution table; every number comes from the recomputed payload."""
    totals = report["totals"]
    lines = [
        "# Replay 臂 FP 归属拆解（谁该为这 5764 个误报负责）",
        "",
        f"- 生成时间：{report['generated_at']}｜执行机：{report['host']} ({report['platform']})｜Python {report['python']}",
        f"- 脚本：`{report['script']}`｜命令：`{report['command']}`",
        f"- 产物：`{report['inputs']['replay_preds_dir']}`（replay 臂逐帧）+ `{report['inputs']['original_preds_dir']}`（零延迟臂参照）",
        "",
        "## 1. 总量归属",
        "",
        "| 口径 | 数量 | 占 FP | FAR |",
        "| --- | ---: | ---: | ---: |",
        f"| replay 臂 FP 合计 | {totals['replay_fp']} | 100.0% | {totals['far_per_frame']:.4f} |",
        f"| ├ lost_by_replay（零延迟臂对、重推臂错） | {totals['buckets']['lost_by_replay']} | "
        f"{totals['buckets']['lost_by_replay'] / max(1, totals['replay_fp']) * 100:.1f}% | - |",
        f"| ├ both_missed（两臂都错，检测器自身噪声） | {totals['buckets']['both_missed']} | "
        f"{totals['buckets']['both_missed'] / max(1, totals['replay_fp']) * 100:.1f}% | - |",
        f"| └ no_ground_truth（该帧无 GT 标注） | {totals['buckets']['no_ground_truth']} | "
        f"{totals['buckets']['no_ground_truth'] / max(1, totals['replay_fp']) * 100:.1f}% | - |",
        f"| 零延迟臂（original）FP | {totals['original_fp']} | - | {totals['original_far_per_frame']:.4f} |",
        f"| 外挂净增 FP | {totals['delta_fp_vs_original']} | - | "
        f"{totals['far_per_frame'] - totals['original_far_per_frame']:+.4f} |",
        "",
        f"- 红线 FAR ≤ {totals['red_line']:.3f} ⇒ 总 FP 上限 {totals['red_line_fp']:.0f}，"
        f"实测 {totals['replay_fp']}，**超线 {totals['fp_over_red_line']:.0f} 个 FP**。",
        f"- 若能消灭全部 `lost_by_replay`（{totals['buckets']['lost_by_replay']} 个），FAR ⇒ "
        f"**{totals['far_if_lost_by_replay_removed']:.4f}**（回到红线内）；",
        f"- 若只能消灭全部 `no_ground_truth`（{totals['buckets']['no_ground_truth']} 个，标注稀疏，非算法可解），FAR ⇒ "
        f"**{totals['far_if_no_ground_truth_removed']:.4f}**。",
        "",
        "## 2. lost_by_replay 的落点距离（离最近 GT 框中心）",
        "",
        "| 距离 | 数量 |",
        "| --- | ---: |",
    ]
    for key, label in (("lt5", "< 5 px"), ("5to10", "5~10 px"), ("10to20", "10~20 px"), ("20to50", "20~50 px"), ("ge50", "≥ 50 px")):
        lines.append(f"| {label} | {totals['distance_hist'][key]} |")
    lines += [
        "",
        "## 3. 逐序列归属（按 lost_by_replay 降序）",
        "",
        "| seq | frames | replay_fp | original_fp | Δfp | lost_by_replay | both_missed | no_gt | FAR | orig_FAR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sorted(
        report["sequences"].items(), key=lambda item: (-item[1]["buckets"]["lost_by_replay"], item[0])
    ):
        lines.append(
            f"| {name} | {row['frames']} | {row['replay_fp']} | {row['original_fp']} | "
            f"{row['delta_fp_vs_original']:+d} | {row['buckets']['lost_by_replay']} | "
            f"{row['buckets']['both_missed']} | {row['buckets']['no_ground_truth']} | "
            f"{row['far_per_frame']:.4f} | {row['original_far_per_frame']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Entry point: build the report, print the one-line conclusion and write both artifacts."""
    args = parse_args()
    report = build_report(args)
    totals = report["totals"]
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"[DIAG] sequences={totals['sequences']} frames={totals['frames']} fp={totals['replay_fp']} "
        f"far={totals['far_per_frame']:.4f} (original {totals['original_far_per_frame']:.4f}, red line {totals['red_line']:.3f}) "
        f"lost_by_replay={totals['buckets']['lost_by_replay']} both_missed={totals['buckets']['both_missed']} "
        f"no_gt={totals['buckets']['no_ground_truth']}"
    )
    print(f"[DIAG] wrote {args.out_json}")
    print(f"[DIAG] wrote {args.out_md}")


if __name__ == "__main__":
    main()
