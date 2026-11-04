#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FAR breakdown of the 24-sequence replay arm: where does the false-alarm rate come from?

Reads only three things (never writes to them):

* ``runs/lookahead/final/arms_final_WTH15_24seq.json`` — the four-arm roll-up (``original`` is the anchor),
* ``runs/lookahead/final/preds_final_WTH15_24seq/<SEQ>.json`` — the replay arm per-frame output,
* ``<labels_root>/<SEQ>__<frame>.txt`` — the flat business-domain ground truth.

The business-domain rule is *not* re-implemented: the ground truth is read with the delivery's own
``load_ground_truth`` and the matching runs through the frozen ``count_matches`` / ``metrics`` of
``manu.evaluation.eval_multiscale_cache_th`` (greedy score-descending, point-in-box, one ground truth
absorbs at most one detection). The per-frame pass only re-walks the same rule frame by frame so the false
positives can be split into "the frame had a target but the reported point was wrong" and "the frame had no
target at all"; its per-frame totals are cross-checked against the frozen matcher.

Usage::

    /home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_far_breakdown.py
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

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    count_matches,
    load_ground_truth,
    metrics,
)

SCRIPT_REL = "manu/tracking/lookahead/diag_far_breakdown.py"
DEFAULT_ARMS_JSON = PROJECT_ROOT / "runs/lookahead/final/arms_final_WTH15_24seq.json"
DEFAULT_PREDS_DIR = PROJECT_ROOT / "runs/lookahead/final/preds_final_WTH15_24seq"
DEFAULT_OUT_JSON = PROJECT_ROOT / "runs/lookahead/final/diag_far_breakdown.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "runs/lookahead/final/diag_far_breakdown.md"
RED_LINE = 0.030  # delivery red line: FAR <= 0.030


def parse_args() -> argparse.Namespace:
    """Command line: every input/output has a default pointing at the delivery artifacts."""
    parser = argparse.ArgumentParser(description="Split the replay-arm FAR into its per-sequence causes.")
    parser.add_argument("--arms-json", type=Path, default=DEFAULT_ARMS_JSON)
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--labels-root", type=Path, default=None, help="defaults to the labels_root of the arms json")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--red-line", type=float, default=RED_LINE)
    return parser.parse_args()


def inbox_hit(box: np.ndarray, point: np.ndarray) -> bool:
    """The frozen inbox rule: the reported centre must fall inside the ground-truth box."""
    return abs(point[0] - box[0]) <= box[2] / 2 and abs(point[1] - box[1]) <= box[3] / 2


def match_frame(boxes: np.ndarray | None, points: np.ndarray | None, scores: np.ndarray | None) -> dict:
    """One frame of the frozen business-domain rule, kept per frame so the false positives can be attributed.

    Mirrors ``count_matches`` frame by frame: the detections are walked in score-descending order and each one
    claims the still-free ground truth whose centre is nearest, but only if the point falls inside that box.
    """
    detected = 0 if scores is None else int((scores >= 0.0).sum())
    truth_count = 0 if boxes is None else len(boxes)
    if detected == 0:
        return {"tp": 0, "fp": 0, "fn": truth_count, "fn_absent": truth_count, "fn_compete": 0, "det": 0}
    keep = points[scores >= 0.0]
    taken = np.zeros(truth_count, dtype=bool)
    matched = 0
    for position in range(detected):
        if not truth_count:
            break
        free = np.nonzero(~taken)[0]
        if not len(free):
            break
        distance = ((boxes[free, :2] - keep[position]) ** 2).sum(axis=1)
        box = int(free[int(np.argmin(distance))])
        if inbox_hit(boxes[box], keep[position]):
            taken[box] = True
            matched += 1
    fn_compete = 0
    for box in range(truth_count):
        if taken[box]:
            continue
        if any(inbox_hit(boxes[box], keep[position]) for position in range(detected)):
            fn_compete += 1
    fn_absent = truth_count - matched - fn_compete
    return {
        "tp": matched,
        "fp": detected - matched,
        "fn": fn_absent + fn_compete,
        "fn_absent": fn_absent,
        "fn_compete": fn_compete,
        "det": detected,
    }


def load_sequence_truth(labels_root: Path, sequence: str, width: int, height: int) -> dict[int, np.ndarray]:
    """Ground truth of one sequence in native pixels, read with the delivery's own loader."""
    return load_ground_truth(labels_root, sequence, width, height)


def sequence_frames(per_frame: dict, truth: dict[int, np.ndarray], frames: int) -> list[tuple]:
    """The per-frame tuples the frozen matcher consumes: (boxes|None, points|None, scores|None)."""
    out = []
    for frame in range(frames):
        dets = per_frame.get(str(frame), [])
        boxes = truth.get(frame)
        if len(dets):
            ranked = sorted(dets, key=lambda det: -float(det["score"]))
            points = np.asarray([det["pos"] for det in ranked], dtype=np.float32).reshape(-1, 2)
            scores = np.asarray([float(det["score"]) for det in ranked], dtype=np.float32)
        else:
            points = None
            scores = None
        out.append((boxes if boxes is not None and len(boxes) else None, points, scores))
    return out


def cache_frames(cache_root: Path, sequence: str) -> int | None:
    """Frame count of the heatmap cache, i.e. the denominator the delivery used for FAR."""
    path = Path(cache_root) / f"{sequence}.pkl"
    if not path.is_file():
        return None
    try:
        import pickle

        with path.open("rb") as handle:
            meta = pickle.load(handle)["meta"]
        return int(meta["frames"])
    except Exception:  # the cache is only a cross-check, never a dependency
        return None


def analyse_sequence(doc: dict, truth: dict[int, np.ndarray], frames: int) -> dict:
    """Per-frame pass of one sequence: counts, false-positive attribution and the shipped/recomputed delta."""
    per_frame = doc["per_frame"]
    tuples = sequence_frames(per_frame, truth, frames)

    reference = metrics(count_matches(tuples, 0.0, 8.0, "inbox"), frames)

    tp = fp = fn = fn_absent = fn_compete = det = 0
    fp_on_gt = fp_on_no_gt = 0
    n_reported = n_empty = n_gt_frames = n_no_gt_frames = 0
    for boxes, points, scores in tuples:
        row = match_frame(boxes, points, scores)
        tp += row["tp"]
        fp += row["fp"]
        fn += row["fn"]
        fn_absent += row["fn_absent"]
        fn_compete += row["fn_compete"]
        det += row["det"]
        has_gt = boxes is not None
        n_gt_frames += int(has_gt)
        n_no_gt_frames += int(not has_gt)
        if points is not None:
            n_reported += 1
        else:
            n_empty += 1
        if row["fp"]:
            if has_gt:
                fp_on_gt += row["fp"]
            else:
                fp_on_no_gt += row["fp"]

    shipped = doc.get("metrics", {})
    return {
        "frames": frames,
        "n_reported": n_reported,
        "n_empty": n_empty,
        "n_gt_frames": n_gt_frames,
        "n_no_gt_frames": n_no_gt_frames,
        "gt_boxes": int(reference["gt"]),
        "det": int(det),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "fn_absent": int(fn_absent),
        "fn_compete": int(fn_compete),
        "fp_on_gt_frames": int(fp_on_gt),
        "fp_on_no_gt_frames": int(fp_on_no_gt),
        "far_per_frame": fp / frames if frames else 0.0,
        "far_if_drop_no_gt_report": fp_on_gt / frames if frames else 0.0,
        "shipped_metrics": {
            "tp": shipped.get("tp"),
            "fp": shipped.get("fp"),
            "fn": shipped.get("fn"),
            "det": shipped.get("det"),
            "gt": shipped.get("gt"),
            "far_per_frame": shipped.get("far_per_frame"),
        },
        "recompute_diff": {
            "tp": int(tp) - int(shipped.get("tp", 0)),
            "fp": int(fp) - int(shipped.get("fp", 0)),
            "fn": int(fn) - int(shipped.get("fn", 0)),
        },
        "reference_diff": {
            "tp": int(tp) - int(reference["tp"]),
            "fp": int(fp) - int(reference["fp"]),
            "fn": int(fn) - int(reference["fn"]),
        },
    }


def build_report(args: argparse.Namespace) -> dict:
    """Read the artifacts, recompute the business-domain metrics and assemble the report payload."""
    arms_doc = json.loads(Path(args.arms_json).read_text(encoding="utf-8"))
    config = arms_doc.get("config", {})
    labels_root = Path(args.labels_root or config.get("labels_root"))
    cache_root = Path(config.get("cache_root", ""))
    width = int(config.get("img_w", 640))
    height = int(config.get("img_h", 512))
    arms = arms_doc.get("arms", {})
    original_rows = arms.get("original", {}).get("current_frame", {}).get("per_sequence", {})

    sequences = {}
    for path in sorted(Path(args.preds_dir).glob("*.json")):
        if path.name == "summary.json":
            continue
        sequence = path.stem
        doc = json.loads(path.read_text(encoding="utf-8"))
        per_frame = doc["per_frame"]
        keys = [int(key) for key in per_frame]
        frames = len(per_frame)
        truth = load_sequence_truth(labels_root, sequence, width, height)
        row = analyse_sequence(doc, truth, frames)
        row["frames_source"] = "preds per_frame key coverage (= the delivery num_frames denominator)"
        row["frames_max_key_plus_one"] = max(keys) + 1 if keys else 0
        row["frames_from_cache_meta"] = cache_frames(cache_root, sequence) if str(cache_root) else None
        original = original_rows.get(sequence, {})
        row["original"] = {
            "tp": original.get("tp"),
            "fp": original.get("fp"),
            "fn": original.get("fn"),
            "far_per_frame": original.get("far_per_frame"),
        }
        row["delta_fp_vs_original"] = row["fp"] - int(original["fp"]) if original.get("fp") is not None else None
        row["delta_far_vs_original"] = (
            row["far_per_frame"] - float(original["far_per_frame"])
            if original.get("far_per_frame") is not None
            else None
        )
        row["fp_over_red_line"] = max(0.0, row["fp"] - args.red_line * row["frames"])
        sequences[sequence] = row

    def total(key: str) -> int:
        return int(sum(row[key] for row in sequences.values()))

    frames_total = total("frames")
    fp_total = total("fp")
    fp_on_gt = total("fp_on_gt_frames")
    fp_on_no_gt = total("fp_on_no_gt_frames")
    original_totals = {
        key: int(sum(int(row["original"][key]) for row in sequences.values() if row["original"].get(key) is not None))
        for key in ("tp", "fp", "fn")
    }
    original_far = original_totals["fp"] / frames_total if frames_total else 0.0
    far = fp_total / frames_total if frames_total else 0.0
    extra_fp = fp_total - original_totals["fp"]

    ranked = sorted(sequences.items(), key=lambda item: (-item[1]["fp"], item[0]))
    over_line_sequences = [(name, row) for name, row in ranked if row["fp_over_red_line"] > 0]
    over_line_fp = sum(row["fp_over_red_line"] for _, row in over_line_sequences)
    attributable = sorted(
        ((name, row) for name, row in sequences.items() if row["delta_fp_vs_original"] is not None),
        key=lambda item: (-item[1]["delta_fp_vs_original"], item[0]),
    )

    totals = {
        "sequences": len(sequences),
        "frames": frames_total,
        "n_reported": total("n_reported"),
        "n_empty": total("n_empty"),
        "n_gt_frames": total("n_gt_frames"),
        "n_no_gt_frames": total("n_no_gt_frames"),
        "gt_boxes": total("gt_boxes"),
        "det": total("det"),
        "tp": total("tp"),
        "fp": fp_total,
        "fn": total("fn"),
        "fn_absent": total("fn_absent"),
        "fn_compete": total("fn_compete"),
        "fp_on_gt_frames": fp_on_gt,
        "fp_on_no_gt_frames": fp_on_no_gt,
        "fp_on_gt_share": fp_on_gt / fp_total if fp_total else 0.0,
        "fp_on_no_gt_share": fp_on_no_gt / fp_total if fp_total else 0.0,
        "far_per_frame": far,
        "far_if_drop_no_gt_report": fp_on_gt / frames_total if frames_total else 0.0,
        "recompute_diff": {
            "tp": total("tp") - int(sum(row["shipped_metrics"]["tp"] for row in sequences.values())),
            "fp": total("fp") - int(sum(row["shipped_metrics"]["fp"] for row in sequences.values())),
            "fn": total("fn") - int(sum(row["shipped_metrics"]["fn"] for row in sequences.values())),
        },
        "original": {
            "tp": original_totals["tp"],
            "fp": original_totals["fp"],
            "fn": original_totals["fn"],
            "far_per_frame": original_far,
        },
        "delta_far_vs_original": far - original_far,
        "fp_delta_vs_original": extra_fp,
        "red_line": args.red_line,
        "over_line_far": far - args.red_line,
        "over_line_fp": fp_total - args.red_line * frames_total,
        "over_line_fp_from_sequences": over_line_fp,
        "top_contributors": [
            {
                "sequence": name,
                "fp": row["fp"],
                "frames": row["frames"],
                "far": row["far_per_frame"],
                "fp_over_red_line": row["fp_over_red_line"],
                "share_of_total_fp": row["fp"] / fp_total if fp_total else 0.0,
                "share_of_over_line_fp": row["fp_over_red_line"] / over_line_fp if over_line_fp else 0.0,
                "fp_on_gt_frames": row["fp_on_gt_frames"],
                "fp_on_no_gt_frames": row["fp_on_no_gt_frames"],
            }
            for name, row in over_line_sequences
        ],
        "top_replay_attributable": [
            {
                "sequence": name,
                "fp": row["fp"],
                "fp_original": row["original"]["fp"],
                "delta_fp": row["delta_fp_vs_original"],
                "share_of_extra_fp": row["delta_fp_vs_original"] / extra_fp if extra_fp else 0.0,
                "far": row["far_per_frame"],
                "original_far": row["original"]["far_per_frame"],
                "delta_far": row["delta_far_vs_original"],
                "fp_on_gt_frames": row["fp_on_gt_frames"],
                "fp_on_no_gt_frames": row["fp_on_no_gt_frames"],
            }
            for name, row in attributable[:5]
            if row["delta_fp_vs_original"] > 0
        ],
    }

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": f"{socket.gethostname()} ({platform.platform()})",
        "python": sys.version.split()[0],
        "script": SCRIPT_REL,
        "command": f"/home/huangzhe/anaconda3/envs/uav/bin/python {SCRIPT_REL}",
        "inputs": {
            "arms_json": str(args.arms_json),
            "preds_dir": str(args.preds_dir),
            "labels_root": str(labels_root),
            "cache_root": str(cache_root),
        },
        "method": {
            "arm": "replay (the main arm of the 24-sequence delivery)",
            "match_rule": (
                "business-domain inbox: the reported point hits a ground-truth box when "
                "|px-cx|<=w/2 and |py-cy|<=h/2; matching is greedy in score-descending order, each detection "
                "claims the still-free box with the nearest centre, one ground truth absorbs at most one "
                "detection, unmatched detections are false positives"
            ),
            "matcher": (
                "manu.evaluation.eval_multiscale_cache_th.count_matches(threshold=0.0, tol=8.0, mode='inbox') "
                "+ metrics(); the per-frame pass re-walks the same rule and is cross-checked against it "
                "(reference_diff in every sequence)"
            ),
            "ground_truth": (
                f"flat files <SEQ>__<frame>.txt under {labels_root}; every line 'class cx cy w h' with cx/cy/w/h "
                f"normalised to [0,1], scaled to native pixels with img_w={width} / img_h={height}; the class "
                "column is ignored and exact duplicate rows are dropped (the delivery's own load_ground_truth)"
            ),
            "frames_denominator": (
                "frames = number of keys in the replay arm's per_frame payload (= range(num_frames) written by "
                "lookahead_longquan.frame_payload), which is exactly the denominator the delivery used for "
                "FAR; cross-checked against the heatmap cache meta 'frames' when the cache is readable"
            ),
            "fp_split": (
                "fp_on_gt_frames = false positives on frames that carry at least one ground-truth box (the "
                "reported position was wrong); fp_on_no_gt_frames = false positives on frames without any "
                "ground-truth box (a pure no-target false alarm)"
            ),
            "far_if_drop_no_gt_report": (
                "upper-bound reasoning only: FAR if every no-ground-truth frame stayed silent, i.e. "
                "fp_on_gt_frames / frames"
            ),
            "original_arm": "arms['original']['current_frame']['per_sequence'] of the delivery roll-up",
            "red_line": args.red_line,
        },
        "sequences": sequences,
        "totals": totals,
    }


def render_markdown(report: dict) -> str:
    """The human-readable twin of the JSON: one table plus the reading of it."""
    totals = report["totals"]
    lines = [
        "# Replay 臂 FAR 拆解（24 序列交付）",
        "",
        f"- 生成时间：{report['generated_at']}｜执行机：{report['host']}｜Python {report['python']}",
        f"- 脚本：`{report['script']}`｜命令：`{report['command']}`",
        f"- 产物：`{report['inputs']['preds_dir']}`（replay 臂逐帧）+ `{report['inputs']['arms_json']}`（四臂汇总）",
        "- 口径：inbox 贪心匹配（score 降序），frames = per_frame 键覆盖 = 交付 num_frames",
        "",
        f"**汇总**：frames={totals['frames']}｜TP={totals['tp']}｜FP={totals['fp']}｜FN={totals['fn']}｜"
        f"FAR={totals['far_per_frame']:.4f}（original 臂 {totals['original']['far_per_frame']:.4f}，"
        f"差值 {totals['delta_far_vs_original']:+.4f}）｜红线 {totals['red_line']:.3f}，"
        f"超线 {totals['over_line_far']:+.4f}（约 {totals['over_line_fp']:.0f} 个 FP）",
        "",
        "| seq | frames | n_reported | tp | fp | fp_on_gt | fp_on_no_gt | far | original_far | 差值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(report["sequences"].items(), key=lambda item: (-item[1]["fp"], item[0]))
    for name, row in ranked:
        original_far = row["original"].get("far_per_frame")
        original_text = f"{original_far:.4f}" if original_far is not None else "n/a"
        delta = row["delta_far_vs_original"]
        delta_text = f"{delta:+.4f}" if delta is not None else "n/a"
        lines.append(
            f"| {name} | {row['frames']} | {row['n_reported']} | {row['tp']} | {row['fp']} | "
            f"{row['fp_on_gt_frames']} | {row['fp_on_no_gt_frames']} | {row['far_per_frame']:.4f} | "
            f"{original_text} | {delta_text} |"
        )

    lines += [
        "",
        "## replay 归因：相对 original 臂多出的 FP",
        "",
        "| seq | fp_replay | fp_original | delta_fp | 占比 | far | original_far |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in totals["top_replay_attributable"]:
        lines.append(
            f"| {item['sequence']} | {item['fp']} | {item['fp_original']} | {item['delta_fp']} | "
            f"{item['share_of_extra_fp'] * 100:.1f}% | {item['far']:.4f} | {item['original_far']:.4f} |"
        )

    top = totals["top_contributors"]
    lines += ["", "## 结论", ""]
    if top:
        names = "、".join(item["sequence"] for item in top[:3])
        top3_fp = sum(item["fp"] for item in top[:3])
        top3_over = sum(item["fp_over_red_line"] for item in top[:3])
        over_line_sum = totals["over_line_fp_from_sequences"]
        lines.append(
            f"1. FAR 超线（红线 {totals['red_line']:.3f}，实测 {totals['far_per_frame']:.4f}）由 "
            f"{len(top)} 个超线序列贡献：{names} 合计 FP={top3_fp:.0f}（占全部 FP 的 "
            f"{top3_fp / totals['fp'] * 100:.1f}%），折合超线部分 {top3_over:.0f} 个 FP（占各序列超线部分之和 "
            f"{over_line_sum:.0f} 的 {top3_over / over_line_sum * 100:.1f}%；"
            f"整体口径的超线量为 {totals['over_line_fp']:.0f} 个 FP，差额来自低于红线的序列留出的余量）。"
        )
    else:
        lines.append(f"1. 没有单序列的 FAR 超过红线 {totals['red_line']:.3f}。")
    lines.append(
        f"2. FP 落点结构：落在有 GT 帧 {totals['fp_on_gt_frames']} 个（{totals['fp_on_gt_share'] * 100:.1f}%），"
        f"落在无 GT 帧 {totals['fp_on_no_gt_frames']} 个（{totals['fp_on_no_gt_share'] * 100:.1f}%）。"
    )
    if top:
        gt_heavy = [item for item in top if item["fp_on_gt_frames"] >= item["fp_on_no_gt_frames"]]
        lines.append(
            f"3. 超线序列里 {len(gt_heavy)}/{len(top)} 个以「有 GT 帧定位错」为主"
            f"（{', '.join(item['sequence'] for item in gt_heavy) or '无'}），属于位置错误；"
            "其余以「无 GT 帧误报」为主，属于无目标帧噪声。"
        )
    lines.append(
        f"4. 上界推演：若无 GT 帧一律不上报，FAR = {totals['far_if_drop_no_gt_report']:.4f}"
        f"（{'仍超线' if totals['far_if_drop_no_gt_report'] > totals['red_line'] else '可回到红线内'}）；"
        "该推演只作参考，不作为结论。"
    )
    worst = totals["top_replay_attributable"][0] if totals["top_replay_attributable"] else None
    if worst:
        lines.append(
            f"5. replay 归因：replay 臂比 original 臂多出 {totals['fp_delta_vs_original']} 个 FP"
            f"（FAR {totals['far_per_frame']:.4f} vs {totals['original']['far_per_frame']:.4f}，"
            f"{totals['delta_far_vs_original']:+.4f}），其中 {worst['sequence']} 一个序列就占 "
            f"{worst['share_of_extra_fp'] * 100:.1f}%（{worst['delta_fp']} 个 FP，"
            f"其中 {worst['fp_on_gt_frames']}/{worst['fp']} 落在有 GT 帧）；"
            "反之 VIDEO00012 的超线在 original 臂同样存在，属于管线既有问题而非 replay 引入。"
        )
    lines += ["", "## 复算对账", ""]
    diff = totals["recompute_diff"]
    lines.append(
        f"自算与交付 metrics 的差值：tp={diff['tp']}，fp={diff['fp']}，fn={diff['fn']}"
        f"（{'完全一致' if not any(diff.values()) else '存在不一致，见 JSON'}）。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Write the JSON and the Markdown twin of the same numbers."""
    args = parse_args()
    report = build_report(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    totals = report["totals"]
    print(
        f"[DIAG] sequences={totals['sequences']} frames={totals['frames']} tp={totals['tp']} fp={totals['fp']} "
        f"fn={totals['fn']} far={totals['far_per_frame']:.4f} (red line {totals['red_line']:.3f}) "
        f"fp_on_gt={totals['fp_on_gt_frames']} fp_on_no_gt={totals['fp_on_no_gt_frames']} "
        f"recompute_diff={totals['recompute_diff']}"
    )
    print(f"[DIAG] wrote {out_json}")
    print(f"[DIAG] wrote {out_md}")


if __name__ == "__main__":
    main()
