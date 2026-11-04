"""离线策略仿真：用已落盘的逐帧产物评估「上报策略」，不重跑管线。

为什么需要它
------------
``lookahead_longquan.py`` 每跑一个变体要几分钟（每臂每帧都要过一遍图像与快核），
但**上报层策略**（限幅 / 截断 / 弃权 / 回落）只作用在「锚点 + 重推结果」两个已算好的
坐标上。这类策略可以**纯产物回放**地评估：读入原臂与重推臂的逐帧上报点，按候选策略
合成新的上报流，再用**交付同款**的匹配口径（``count_matches`` + inbox 判据）重算
F1 / P / R / FAR / hit@5px。

本脚本的匹配与计数**直接 import 交付代码**（``manu.evaluation.eval_multiscale_cache_th``），
不自己发明判据，因此结果与 ``lookahead_longquan.py`` 的产物**逐位一致**（脚本启动时
用 ``--ship-arms`` 做自检：不一致直接退出非零）。

用法::

    python manu/tracking/lookahead/sim_policy_offline.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --original-preds runs/lookahead/final/ref_preds_original_v3_24seq \
        --replay-preds  runs/lookahead/final/preds_final_WTH15_24seq \
        --ship-arms     runs/lookahead/final/arms_final_WTH15_24seq.json \
        --out-json runs/lookahead/final/sim_policy_offline.json \
        --out-md   runs/lookahead/final/sim_policy_offline.md
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import count_matches, load_ground_truth  # noqa: E402
from manu.tracking.lookahead.lookahead_longquan import build_candidates  # noqa: E402

POLICY_NAMES = ("replay", "stale", "kinematic", "original", "clamp", "abstain", "cap_stale", "cap_kinematic")


def _load_reports(directory: Path, sequence: str) -> dict[int, list[dict]]:
    """读一个序列的逐帧上报（``per_frame`` → ``{帧号: [det, ...]}``，空 list 也保留）。"""
    path = directory / f"{sequence}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    per_frame = document["per_frame"]
    return {int(key): list(value) for key, value in per_frame.items()}


def _load_predictions(directory: Path, sequence: str) -> dict[int, list[tuple[np.ndarray, float]]]:
    """每个帧号上的全部上报点，按得分降序：``[(pos, score), ...]``（空帧 = 空 list）。"""
    reports = _load_reports(directory, sequence)
    out: dict[int, list[tuple[np.ndarray, float]]] = {}
    for frame, detections in reports.items():
        ranked = sorted(detections, key=lambda det: -float(det.get("score", 0.0)))
        out[frame] = [
            (np.asarray(det["pos"], dtype=np.float64).reshape(2), float(det.get("score", 0.0)))
            for det in ranked
        ]
    return out


def _frame_index_map(cache_root: Path, sequence: str, truth: dict, dedup_radius, floor: float) -> list[int]:
    """报告索引 → 真实帧号 的映射。

    ``lookahead_longquan.py`` 的帧集是 ``sorted(候选帧 ∪ 有 GT 的帧)``，报告流按这个
    顺序编号；**有跳帧的序列上报告索引 ≠ 真实帧号**（实测 014144 差 1 帧 ⇒ stale 臂
    多算 1 个 FP）。Δ 也是**报告索引**空间的偏移，所以这里必须复刻同一套编号。
    """
    candidates = build_candidates(cache_root, sequence, dedup_radius, floor)
    frame_set = sorted(set(candidates) | set(truth))
    return frame_set


def _build_cache(
    labels_root: Path,
    original_dir: Path,
    replay_dir: Path,
    width: int,
    height: int,
    cache_root: Path | None = None,
    dedup_radius=None,
    floor: float = 0.22,
) -> dict:
    sequences = sorted(
        path.stem for path in original_dir.glob("*.json") if path.stem != "summary"
    )
    cache = {}
    for sequence in sequences:
        truth = load_ground_truth(labels_root, sequence, width, height)
        # 记录序号 → 真实帧号（records 的帧集 = 候选帧 ∪ 有 GT 的帧）。
        # 流水线的 Δ=8 走的是**记录序号**空间，不是帧号空间 ⇒ 仿真必须同一口径。
        if cache_root is None:
            frame_map = sorted(truth)
        else:
            frame_map = _frame_index_map(cache_root, sequence, truth, dedup_radius, floor)
        cache[sequence] = {
            "truth": truth,
            "original": _load_predictions(original_dir, sequence),
            "replay": _load_predictions(replay_dir, sequence),
            "frame_map": frame_map,
        }
    return cache


def _velocity(points: dict[int, list[tuple[np.ndarray, float]]], frame: int) -> np.ndarray | None:
    """相邻两帧最高分上报点的差分（``None`` 表示缺数据）。"""
    current = points.get(frame)
    previous = points.get(frame - 1)
    if not current or not previous:
        return None
    return current[0][0] - previous[0][0]


def _policy_detections(name, parameter, frame, original, replay, delta):
    """按策略名合成该帧的上报点列表；返回空 list 表示弃权。

    ``replay`` / ``original`` 直接用产物；``stale`` 搬运源帧的全部上报点（与交付
    ``mode="stale"`` 同口径）；``kinematic`` 给每个点加上 ``Δ·v``；``clamp`` /
    ``abstain`` / ``cap_*`` 在「锚点 + 重推」两坐标上做变换。
    """
    replay_dets = replay.get(frame, [])
    anchor_dets = original.get(frame - delta, [])
    if name == "replay":
        return list(replay_dets)
    if name == "original":
        return list(original.get(frame, []))
    if name == "stale":
        return list(anchor_dets)
    if name == "kinematic":
        step = _velocity(original, frame - delta)
        if step is None:
            return list(anchor_dets)
        return [(pos + delta * step, score) for pos, score in anchor_dets]
    if not replay_dets:
        return []
    if not anchor_dets:
        return list(replay_dets)
    anchor = anchor_dets[0][0]
    best = replay_dets[0][0]
    gap = best - anchor
    distance = float(np.linalg.norm(gap))
    if name == "abstain":
        if distance > parameter:
            return []
        return list(replay_dets)
    step = _velocity(original, frame - delta)
    if name == "cap_stale":
        fallback = anchor
    elif name == "cap_kinematic":
        fallback = anchor if step is None else anchor + delta * step
    elif name == "clamp":
        scale = parameter / distance if distance > parameter else 1.0
        return [(anchor + gap * scale, score) for pos, score in replay_dets]
    else:
        raise ValueError(f"unknown policy {name!r}")
    if distance <= parameter:
        return list(replay_dets)
    return [(fallback, score) for pos, score in replay_dets]


def evaluate(cache: dict, name: str, parameter: float = 0.0, delta: int = 8) -> dict:
    """按交付口径（inbox 贪心匹配）重算整段指标。"""
    totals = dict.fromkeys(("tp", "fp", "fn", "fn_absent", "fn_compete", "det"), 0)
    gt_total = 0
    frames_total = 0
    hit5 = 0
    gt_with_report = 0
    cle_samples: list[float] = []
    per_sequence = []
    for sequence, item in cache.items():
        truth = item["truth"]
        original = item["original"]
        replay = item["replay"]
        frames = []
        frame_map = item["frame_map"]
        # 记录序号 i 的上报：payload 以**真实帧号**为键 ⇒ 取 frame_map[i] 对应的 payload 项。
        by_record = {
            index: original.get(frame_map[index], []) for index in range(len(frame_map))
        }
        replay_by_record = {
            index: replay.get(frame_map[index], []) for index in range(len(frame_map))
        }
        for index in range(len(frame_map)):
            detections = _policy_detections(
                name, parameter, index, by_record, replay_by_record, delta
            )
            frame = frame_map[index]
            boxes = truth.get(frame)
            box_array = (
                np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if boxes is not None else None
            )
            if not detections:
                frames.append((box_array, None, None))
                continue
            points = np.asarray([pos for pos, _ in detections], dtype=np.float32)
            scores = np.asarray([score for _, score in detections], dtype=np.float32)
            frames.append((box_array, points, scores))
        counts = count_matches(frames, 0.0, 8.0, "inbox")
        for key in totals:
            totals[key] += int(counts[key])
        gt_total += sum(1 for frame in frame_map if frame in truth)
        frames_total += len(original)
        sequence_hit5 = 0
        sequence_gt = 0
        for index, frame in enumerate(frame_map):
            if frame not in truth:
                continue
            sequence_gt += 1
            detections = _policy_detections(
                name, parameter, index, by_record, replay_by_record, delta
            )
            if not detections:
                continue
            boxes = np.asarray(truth[frame], dtype=np.float32).reshape(-1, 4)
            best = min(
                min(float(np.hypot(pos[0] - box[0], pos[1] - box[1])) for box in boxes)
                for pos, _ in detections
            )
            if best <= 5.0:
                sequence_hit5 += 1
                cle_samples.append(best)
        gt_with_report += sequence_gt
        hit5 += sequence_hit5
        per_sequence.append(
            {
                "sequence": sequence,
                "frames": len(original),
                "tp": int(counts["tp"]),
                "fp": int(counts["fp"]),
                "fn": int(counts["fn"]),
                "hit_rate_5px": sequence_hit5 / sequence_gt if sequence_gt else 0.0,
            }
        )
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "policy": name,
        "parameter": parameter,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "far_per_frame": totals["fp"] / frames_total if frames_total else 0.0,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "gt": gt_total,
        "frames": frames_total,
        "hit_rate_5px": hit5 / gt_with_report if gt_with_report else 0.0,
        "cle_p50": float(np.percentile(cle_samples, 50)) if cle_samples else 0.0,
        "per_sequence": per_sequence,
    }


def _check_parity(cache: dict, ship_arms: Path, delta: int) -> list[str]:
    """用交付产物自检：仿真重算的 original / stale / replay 必须与产物逐位一致。"""
    document = json.loads(ship_arms.read_text(encoding="utf-8"))
    arms = document["arms"]
    problems = []
    for name, arm in (("original", "original"), ("stale", "stale"), ("replay", "replay")):
        shipped = arms[arm]["current_frame"]["overall"]
        recomputed = evaluate(cache, name, delta=delta)
        for key in ("f1", "precision", "recall", "far_per_frame", "tp", "fp", "fn"):
            if abs(float(shipped[key]) - float(recomputed[key])) > 1e-9:
                problems.append(
                    f"{name}.{key}: shipped={shipped[key]} recomputed={recomputed[key]}"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--original-preds", required=True, help="同帧检测臂的逐帧产物目录")
    parser.add_argument("--replay-preds", required=True, help="重推臂的逐帧产物目录")
    parser.add_argument("--ship-arms", default="", help="交付产物 JSON；给了就做逐位一致性自检")
    parser.add_argument("--delta", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--cache", default="", help="产物缓存（pickle）；给了就复用，不重读 GT")
    parser.add_argument("--cache-root", default="", help="热图缓存根目录；给了就精确复刻报告索引→真实帧号映射")
    parser.add_argument("--dedup-radius", type=float, default=768.0, help="与交付参数档一致（system_sota_longquan_trial0080_delivery_v3_params.json）")
    parser.add_argument("--floor", type=float, default=0.0, help="候选池阈值（与交付一致）")
    parser.add_argument("--parameters", default="", help="要扫的参数表：name:value1,value2;...")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    started = time.time()
    labels_root = Path(args.labels_root)
    original_dir = Path(args.original_preds)
    replay_dir = Path(args.replay_preds)

    cache = None
    if args.cache and Path(args.cache).exists():
        with Path(args.cache).open("rb") as handle:
            cache = pickle.load(handle)
    if cache is None:
        cache = _build_cache(
            labels_root,
            original_dir,
            replay_dir,
            args.width,
            args.height,
            Path(args.cache_root) if args.cache_root else None,
            args.dedup_radius,
            args.floor,
        )
        if args.cache:
            with Path(args.cache).open("wb") as handle:
                pickle.dump(cache, handle)
    print(f"[SIM] sequences={len(cache)} loaded in {time.time() - started:.1f}s", flush=True)

    parity = []
    if args.ship_arms:
        parity = _check_parity(cache, Path(args.ship_arms), args.delta)
        if parity:
            print("[SIM][FAIL] 与交付产物不一致：")
            for item in parity:
                print(f"    {item}")
            return 1
        print("[SIM][PASS] original / stale / replay 三臂与交付产物逐位一致", flush=True)

    rows = []
    for name in ("replay", "stale", "kinematic", "original"):
        rows.append(evaluate(cache, name, delta=args.delta))
    if args.parameters:
        for chunk in args.parameters.split(";"):
            if not chunk.strip():
                continue
            name, _, values = chunk.partition(":")
            name = name.strip()
            for value in values.split(","):
                rows.append(evaluate(cache, name, float(value), delta=args.delta))

    lines = [
        "# 离线策略仿真（上报层）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%dT%H:%M:%S%z')}｜执行机：{os.uname().nodename}",
        f"- 脚本：`manu/tracking/lookahead/sim_policy_offline.py`｜Δ={args.delta}",
        f"- 口径：交付同款 inbox 贪心匹配（`count_matches`）；hit@5px 分母 = 有 GT 且在帧集内的帧",
        "",
        "| 策略 | 参数 | F1 | P | R | FAR/帧 | hit@5px | CLE p50 | 三闸 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:--|",
    ]
    for row in rows:
        passed = row["f1"] >= 0.88 and row["recall"] >= 0.85 and row["far_per_frame"] <= 0.030
        lines.append(
            f"| {row['policy']} | {row['parameter']:.0f} | {row['f1'] * 100:.2f}% | "
            f"{row['precision'] * 100:.2f}% | {row['recall'] * 100:.2f}% | {row['far_per_frame']:.4f} | "
            f"{row['hit_rate_5px'] * 100:.2f}% | {row['cle_p50']:.2f}px | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
    markdown = "\n".join(lines) + "\n"
    print(markdown)

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "host": os.uname().nodename,
                    "delta": args.delta,
                    "parity_problems": parity,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[SIM] wrote {args.out_json}")
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
        print(f"[SIM] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
