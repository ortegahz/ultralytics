#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Random-FPS (delta-n) simulation + hyper-parameter search for the spatio-temporal fusion stage.

WHAT THIS DOES
--------------
The board pushes frames to the SDK at the speed the NPU can keep up with, not at the
camera's 25 fps.  Measured on the RK3588 board (``cad_on.tsv``): 8.03 updates/s, mean
interval 124.6 ms, and the *source frame step* between two consecutive updates is

    dN = 1: 2.2%  2: 19.0%  3: 51.6%  4: 21.7%  5: 4.4%  6: 1.1%  7: 0.1%   (mean 3.11)

i.e. one tracker update covers ~3.1 camera frames.  The fusion engine used to be blind
to that: it saw a *dense* update counter, so its Kalman step was 1 frame, its track age
counted updates and its association gate never widened.

This script simulates that cadence on the frozen detection substrate, then searches the
delta-n aware parameters.  Three arms are produced for the report:

  (1) per-frame ceiling   -- dN == 1, the frozen parameters
  (2) jumped, untuned     -- the sampled cadence, the frozen parameters
  (3) jumped, optimised   -- the sampled cadence, the searched parameters

The simulation is a *frame-sampling* simulation: the detector only "delivers" the
sampled camera frames, the tracker only steps on them, and every undelivered frame has
no candidate at all (that is what really happens on the board: the NPU never ran on it).

CADENCE SCENARIOS
-----------------
``s1``  the measured cadence as-is (board, 1 NPU worker, ~8.0 fps)
``s2``  the measured cadence slowed 1.5x with 25 % log-normal jitter (other business
        sharing the SoC) -- ~5.4 fps, wider spread
``s3``  the measured cadence slowed 2.0x with 35 % jitter (heavy load) -- ~4.0 fps
``s4``  the assumption from the simulation brief (2 NPU workers, ~13.9 fps):
        dN = 1:35 % 2:50 % 3:15 %

Every scenario additionally injects 8 % adjacent-frame swaps (the brief's out-of-order
rule); the engine's ``fid <= last_fid`` guard drops the late frame, exactly like on the
board.

Usage::

    python manu/evaluation/rt_fps_jitter_search.py \
        --cache-root /mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_params.json \
        --seqs VIDEO00007_19700101_000818,... \
        --phase search --workers 4 --out-json /tmp/rt_search.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_bidirectional_track_fusion import (  # noqa: E402
    evaluate_sequence_bidirectional,
    natural_sort_key,
)
from manu.evaluation.eval_multiscale_cache_th import count_matches, metrics  # noqa: E402
from manu.evaluation.system_sota_longquan import (  # noqa: E402
    IMG_H,
    apply_consecutive_confirmation,
    build_records,
    build_smoother_config,
    build_tracker_config,
    load_cache_meta,
)

MEASURED_DELTA_N_JSON = Path(__file__).resolve().parent / "rt_fps_jitter_measured_delta_n.json"

# ---------------------------------------------------------------------------
# The scenarios: how the detector's update cadence is simulated.
#   slowdown -- the measured cadence is multiplied by it (other business on the SoC)
#   sigma    -- log-normal jitter added on top (scheduling noise)
#   kind     -- "measured" (bootstrap the measured dN sequence) or "doc" (the brief)
# ---------------------------------------------------------------------------
SCENARIOS: dict[str, dict] = {
    "s1": {"kind": "measured", "slowdown": 1.0, "sigma": 0.0,
           "label": "measured 1x (board idle, ~8.0 fps)"},
    "s2": {"kind": "measured", "slowdown": 1.5, "sigma": 0.25,
           "label": "measured 1.5x + jitter (shared SoC, ~5.4 fps)"},
    "s3": {"kind": "measured", "slowdown": 2.0, "sigma": 0.35,
           "label": "measured 2x + jitter (heavy load, ~4.0 fps)"},
    "s4": {"kind": "doc", "label": "brief assumption (2 workers, ~13.9 fps)"},
}
DOC_DELTA_N_WEIGHTS = ((1, 0.35), (2, 0.50), (3, 0.15))
SWAP_PROB = 0.08

# ---------------------------------------------------------------------------
# The search space.  Every entry maps a *params* key to the values tried.
# `max_age_phys: None` means "off" -> the legacy per-update age (max_age).
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, list] = {
    "min_hits": [2, 3],
    "max_age_phys": [None, 10, 12, 15, 18],
    "match_dist": [4.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    "r_slope": [0.0, 2.0, 3.5, 5.0],
    "alpha_decay": [1.0, 0.75, 0.82, 0.88, 0.93],
    "confirm_gap_tol": [0, 2, 3, 4, 5, 6],
    "confirm_consec": [2, 3, 4, 6],
}

# The keys that are handed to the tracker (not to the continuity gate).
TRACKER_KEYS = ("min_hits", "max_age_phys", "match_dist", "r_slope", "alpha_decay")

_RECORDS: dict[str, list[dict]] = {}
_MEASURED: list[int] = []
_FRAMES: dict[str, int] = {}


# ---------------------------------------------------------------------------
# 1. Cadence sampling
# ---------------------------------------------------------------------------
def load_measured_delta_n(path: Path = MEASURED_DELTA_N_JSON) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        return list(json.load(handle)["delta_n_sequence"])


def sample_delivered_frames(
    total_frames: int, scenario: str, seed: int, measured: list[int]
) -> list[int]:
    """Return the camera frame ids the detector delivers, in *arrival* order.

    The arrival order may contain an adjacent swap (8 %); the caller feeds the frames
    to the engine in ascending order, and the engine's ``fid <= last_fid`` guard drops
    the late one, which is why the returned list is filtered here as well.
    """
    spec = SCENARIOS[scenario]
    rng = random.Random((hash(scenario) & 0xFFFF) * 1_000_003 + seed)

    frames: list[int] = []
    current = 0
    while current < total_frames:
        frames.append(current)
        if spec["kind"] == "doc":
            step = rng.choices([w[0] for w in DOC_DELTA_N_WEIGHTS],
                               [w[1] for w in DOC_DELTA_N_WEIGHTS])[0]
        else:
            step = rng.choice(measured)
            sigma = float(spec.get("sigma", 0.0))
            if sigma > 0.0:
                step = step * math.exp(rng.gauss(0.0, sigma))
            step = max(1, int(round(step * float(spec.get("slowdown", 1.0)))))
        current += step

    # 8 % adjacent swap -> a local out-of-order delivery.
    for index in range(len(frames) - 1):
        if rng.random() < SWAP_PROB:
            frames[index], frames[index + 1] = frames[index + 1], frames[index]

    # The engine's guard: a frame that arrives at or before the last accepted one is
    # dropped (it was computed but arrived late).
    accepted: list[int] = []
    last = -1
    for frame in frames:
        if frame > last:
            accepted.append(frame)
            last = frame
    return accepted


def cadence_stats(frames: list[int]) -> dict:
    steps = [frames[i] - frames[i - 1] for i in range(1, len(frames))]
    if not steps:
        return {"updates": len(frames), "mean_delta_n": 0.0, "delta_n_hist": {}}
    counter = Counter(steps)
    return {
        "updates": len(frames),
        "mean_delta_n": round(sum(steps) / len(steps), 4),
        "delta_n_hist": {str(k): counter[k] for k in sorted(counter)},
        "fps": round(len(frames) / (frames[-1] / 25.0), 4) if frames[-1] else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. One arm evaluation (one sequence, one scenario, one seed, one parameter set)
# ---------------------------------------------------------------------------
def run_pipeline(
    sequence: str,
    params: dict,
    delivered: list[int] | None,
    dense_time_base: bool = False,
) -> dict:
    """Run the system pipeline on one sequence and return its per-frame detections.

    ``delivered`` is ``None`` for the per-frame ceiling arm (every camera frame is
    delivered).  ``dense_time_base`` selects the engine's time base:

    * ``False`` (physical): the engine sees the camera frame NUMBER, so a skipped frame
      is a real time gap -- this is what the delta-n aware engine must survive.
    * ``True`` (dense): the engine sees a dense update counter, i.e. exactly what the
      shipped SDK does today (it cannot see the physical gaps at all).
    """
    records = _RECORDS[sequence]
    delivered_set = set(delivered) if delivered is not None else None

    if delivered_set is None:
        run_records = records
    elif dense_time_base:
        run_records = [r for r in records if r["frame_index"] in delivered_set]
    else:
        run_records = []
        for record in records:
            if record["frame_index"] in delivered_set:
                run_records.append(record)
            else:
                blank = dict(record)
                blank["pred_points"] = np.zeros((0, 2), dtype=np.float32)
                blank["pred_scores"] = np.zeros((0,), dtype=np.float32)
                run_records.append(blank)

    tracker_config = build_tracker_config(params, IMG_H)
    tracker_config["max_age_phys"] = params.get("max_age_phys", None)
    tracker_config["r_slope"] = float(params.get("r_slope", 0.0))
    tracker_config["alpha_decay"] = float(params.get("alpha_decay", 1.0))
    smoother_config = build_smoother_config(params, IMG_H)

    res = evaluate_sequence_bidirectional(
        records=run_records,
        dist_thresh=8.0,
        th_base=float(params["th_base"]),
        th_salvage=float(params["th_salvage"]),
        th_ground=float(params["th_ground"]),
        sky_ratio=float(params["sky_ratio"]),
        img_h=IMG_H,
        tracker_config=tracker_config,
        smoother_config=smoother_config,
        match_mode="bbox",
        use_physical_frame_index=not dense_time_base,
        delivered_frames=None if dense_time_base else delivered_set,
    )

    dets = apply_consecutive_confirmation(
        res["bidi_frame_dets"],
        confirm_consec=int(params.get("confirm_consec", 3)),
        confirm_gap_tol=int(params.get("confirm_gap_tol", 0)),
        confirm_trim_prefix=int(params.get("confirm_trim_prefix", 1)),
    )

    ordered = sorted(run_records, key=lambda r: natural_sort_key(r["im_name"]))
    use_physical = not dense_time_base

    # The detections, placed back on the PHYSICAL frame axis.
    # In the physical mode the engine's output axis IS the camera frame number; in the
    # dense mode it is the (compacted) record position.
    det_by_frame: dict[int, tuple] = {}
    for position, record in enumerate(ordered):
        frame_id = int(record["frame_index"])
        index = frame_id if use_physical else position
        frame_dets = dets[index] if index < len(dets) else []
        if len(frame_dets):
            ranked = sorted(frame_dets, key=lambda det: -float(det["score"]))
            points = np.asarray([det["pos"] for det in ranked], dtype=np.float32).reshape(-1, 2)
            scores = np.asarray([float(det["score"]) for det in ranked], dtype=np.float32)
        else:
            points, scores = None, None
        det_by_frame[frame_id] = (points, scores)

    # The ground truth of EVERY camera frame (the frames the detector never delivered
    # still have targets -- they are the false negatives of a starved detector).
    gt_by_frame: dict[int, np.ndarray] = {}
    for record in records:
        boxes = np.asarray(record["gt_bboxes"], dtype=np.float32).reshape(-1, 4)
        gt_by_frame[int(record["frame_index"])] = boxes if len(boxes) else None

    return {
        "sequence": sequence,
        "det_by_frame": det_by_frame,
        "gt_by_frame": gt_by_frame,
        "total_frames": max(gt_by_frame) + 1 if gt_by_frame else 0,
    }


def score_run(run: dict, delivered: list[int] | None) -> dict:
    """Score one pipeline run, both over all camera frames and over the delivered ones."""
    det_by_frame = run["det_by_frame"]
    gt_by_frame = run["gt_by_frame"]
    total_frames = run["total_frames"]
    delivered_set = set(delivered) if delivered is not None else None

    all_frames: list[tuple] = []
    for frame_id in range(total_frames):
        boxes = gt_by_frame.get(frame_id)
        points, scores = det_by_frame.get(frame_id, (None, None))
        all_frames.append((boxes, points, scores))

    delivered_ids = sorted(
        frame_id for frame_id in gt_by_frame
        if delivered_set is None or frame_id in delivered_set
    )
    delivered_frames = [
        (gt_by_frame.get(frame_id), *det_by_frame.get(frame_id, (None, None)))
        for frame_id in delivered_ids
    ]

    m_all = metrics(count_matches(all_frames, 0.0, 8.0, "inbox"), len(all_frames))
    # The FAR denominator is the number of camera frames the detector actually delivered,
    # not the number of frames that carry a ground truth.
    far_frames = len(delivered_set) if delivered_set is not None else len(all_frames)
    m_del = metrics(count_matches(delivered_frames, 0.0, 8.0, "inbox"), far_frames)
    return {
        "sequence": run["sequence"],
        "f1_all": m_all["f1"],
        "f1_delivered": m_del["f1"],
        "recall_all": m_all["recall"],
        "precision_all": m_all["precision"],
        "far_all": m_all["far_per_frame"],
        "recall_delivered": m_del["recall"],
        "precision_delivered": m_del["precision"],
        "far_delivered": m_del["far_per_frame"],
        "tp_all": m_all["tp"],
        "fp_all": m_all["fp"],
        "fn_all": m_all["fn"],
        "gt_all": m_all["gt"],
        "tp_delivered": m_del["tp"],
        "fp_delivered": m_del["fp"],
        "fn_delivered": m_del["fn"],
        "gt_delivered": m_del["gt"],
        "delivered_frames": far_frames,
        "total_frames": total_frames,
    }


def _worker(task: tuple) -> dict:
    sequence, scenario, seed, params, delivered, dense = task
    run = run_pipeline(sequence, params, delivered, dense)
    row = score_run(run, delivered)
    row["scenario"] = scenario
    row["seed"] = seed
    row["dense_time_base"] = dense
    return row


# ---------------------------------------------------------------------------
# 3. The parameter evaluation (all sequences x scenarios x seeds)
# ---------------------------------------------------------------------------
def evaluate_params(
    params: dict,
    sequences: list[str],
    scenarios: list[str],
    seeds: list[int],
    pool=None,
    dense_time_base: bool = False,
) -> dict:
    """Return the aggregated metrics of one parameter set."""
    tasks = []
    for scenario in scenarios:
        for seed in seeds:
            for sequence in sequences:
                delivered = None
                if scenario != "perframe":
                    delivered = sample_delivered_frames(
                        _FRAMES[sequence], scenario, seed, _MEASURED
                    )
                tasks.append((sequence, scenario, seed, params, delivered, dense_time_base))

    if pool is not None:
        results = pool.map(_worker, tasks, chunksize=1)
    else:
        results = [_worker(task) for task in tasks]

    grouped: dict[str, list[dict]] = {}
    for row in results:
        grouped.setdefault(row["scenario"], []).append(row)

    summary: dict[str, dict] = {}
    for scenario, rows in grouped.items():
        bucket = {
            key: [row[key] for row in rows]
            for key in ("f1_delivered", "f1_all", "far_delivered", "far_all",
                        "recall_delivered", "precision_delivered")
        }
        entry = {key: (sum(values) / len(values) if values else 0.0)
                 for key, values in bucket.items()}
        entry["n_samples"] = len(rows)
        # The frame-weighted (aggregate) view over the DELIVERED frames -- the same
        # convention as the frozen "OVERALL" row of per_sequence_hyperparams.tsv.
        entry["f1_delivered_agg"] = _aggregate_f1(rows)
        entry["far_delivered_agg"] = _aggregate_far(rows)
        entry["recall_delivered_agg"] = _aggregate_recall(rows)
        entry["precision_delivered_agg"] = _aggregate_precision(rows)
        summary[scenario] = entry

    jump_scenarios = [s for s in scenarios if s != "perframe"]
    if jump_scenarios:
        summary["_jump_mean_f1"] = sum(summary[s]["f1_delivered_agg"] for s in jump_scenarios) / len(jump_scenarios)
        summary["_jump_worst_f1"] = min(summary[s]["f1_delivered_agg"] for s in jump_scenarios)
        summary["_jump_mean_far"] = sum(summary[s]["far_delivered_agg"] for s in jump_scenarios) / len(jump_scenarios)
    else:
        summary["_jump_mean_f1"] = 0.0
        summary["_jump_worst_f1"] = 0.0
        summary["_jump_mean_far"] = 0.0
    return {"summary": summary, "rows": results}


def _per_sequence_rows(rows: list[dict]) -> dict[str, dict]:
    """Group the per-(scenario, seed, sequence) rows by sequence.

    The per-sequence F1 uses the SAME convention as the frozen
    ``per_sequence_hyperparams.tsv`` of osd_sys_sota_trial0080_th045_20261022:
    precision = tp/(tp+fp), recall = tp/(tp+fn), F1 = 2PR/(P+R).  That is what
    makes the numbers comparable with the ``*_F1-<value>.mp4`` file names.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)
    table: dict[str, dict] = {}
    for sequence, items in grouped.items():
        tp = sum(i["tp_delivered"] for i in items)
        fp = sum(i["fp_delivered"] for i in items)
        fn = sum(i["fn_delivered"] for i in items)
        gt = sum(i["gt_delivered"] for i in items)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        table[sequence] = {
            "f1_pct": 100.0 * f1,
            "recall_pct": 100.0 * recall,
            "precision_pct": 100.0 * precision,
            "far": sum(i["far_delivered"] for i in items) / len(items),
            "tp": tp, "fp": fp, "fn": fn, "gt": gt,
            "delivered_frames": sum(i["delivered_frames"] for i in items) / len(items),
            "seeds": len(items),
        }
    return table


def _aggregate_counts(rows: list[dict]) -> tuple[int, int, int, int]:
    tp = fp = fn = frames = 0
    for row in rows:
        tp += row["tp_delivered"]
        fp += row["fp_delivered"]
        fn += row["fn_delivered"]
        frames += row["delivered_frames"]
    return tp, fp, fn, frames


def _aggregate_f1(rows: list[dict]) -> float:
    """The frame-weighted F1 of the delivered frames (pooling tp/fp/fn over the rows)."""
    tp, fp, fn, _frames = _aggregate_counts(rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _aggregate_far(rows: list[dict]) -> float:
    _tp, fp, _fn, frames = _aggregate_counts(rows)
    return fp / frames if frames else 0.0


def _aggregate_recall(rows: list[dict]) -> float:
    tp, _fp, fn, _frames = _aggregate_counts(rows)
    return tp / (tp + fn) if (tp + fn) else 0.0


def _aggregate_precision(rows: list[dict]) -> float:
    tp, fp, _fn, _frames = _aggregate_counts(rows)
    return tp / (tp + fp) if (tp + fp) else 0.0


# ---------------------------------------------------------------------------
# 4. Search
# ---------------------------------------------------------------------------
def objective(result: dict, ceiling_far: float) -> float:
    """The search objective.

    The frame-weighted F1 over the jump scenarios, with a hard penalty when the
    false-alarm rate grows above the per-frame ceiling's (the delivery red line).
    """
    summary = result["summary"]
    score = summary["_jump_mean_f1"]
    far_penalty = max(0.0, summary["_jump_mean_far"] - ceiling_far) * 2.0
    return score - far_penalty


def coordinate_search(
    base_params: dict,
    sequences: list[str],
    scenarios: list[str],
    seeds: list[int],
    space: dict[str, list],
    pool,
    ceiling_far: float,
    sweeps: int = 2,
    log=print,
) -> tuple[dict, list[dict]]:
    """Coordinate descent over the search space."""
    current = dict(base_params)
    best = objective(evaluate_params(current, sequences, scenarios, seeds, pool), ceiling_far)
    history = [{"params": dict(current), "objective": best, "stage": "start"}]
    log(f"[search] start objective={best:.6f}")

    for sweep in range(sweeps):
        improved = False
        for key, values in space.items():
            base_value = current.get(key)
            for value in values:
                if value == base_value:
                    continue
                candidate = dict(current)
                candidate[key] = value
                result = evaluate_params(candidate, sequences, scenarios, seeds, pool)
                score = objective(result, ceiling_far)
                history.append({
                    "params": dict(candidate),
                    "objective": score,
                    "stage": f"sweep{sweep}:{key}={value}",
                    "summary": {k: v for k, v in result["summary"].items() if not k.startswith("_")},
                })
                if score > best + 1e-9:
                    best = score
                    current = candidate
                    improved = True
                    log(f"[search] sweep{sweep} {key}={value} -> objective={score:.6f} (better)")
        if not improved:
            log(f"[search] sweep{sweep} converged")
            break
    return current, history


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-FPS delta-n simulation + hyper-parameter search")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--params-json", required=True)
    parser.add_argument("--seqs", default="")
    parser.add_argument("--all-seqs", action="store_true")
    parser.add_argument("--phase", default="search", choices=["search", "eval"])
    parser.add_argument("--scenarios", default="s1,s2,s3,s4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--space-json", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--params-out", default="")
    parser.add_argument("--opt-params-json", default="")
    return parser.parse_args()


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        blob = json.load(handle)
    params = dict(blob.get("params", blob))
    return params


def prepare(args: argparse.Namespace) -> list[str]:
    global _MEASURED
    _MEASURED = load_measured_delta_n()

    if args.all_seqs:
        sequences = sorted(
            p.stem for p in Path(args.cache_root).glob("VIDEO*.pkl")
        )
    elif args.seqs:
        sequences = [s for s in args.seqs.split(",") if s]
    else:
        raise SystemExit("[ERROR] give --seqs or --all-seqs")

    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    for sequence in sequences:
        meta = load_cache_meta(cache_root, sequence)
        frames = int(meta["frames"])
        records = build_records(cache_root, labels_root, sequence, 768.0, 0.0)
        if args.max_frames:
            records = [r for r in records if r["frame_index"] < args.max_frames]
            frames = min(frames, args.max_frames)
        ids = [r["frame_index"] for r in records]
        if any(b <= a for a, b in zip(ids, ids[1:])):
            raise SystemExit(f"[ERROR] {sequence}: frame ids are not strictly increasing")
        _RECORDS[sequence] = records
        _FRAMES[sequence] = frames
    return sequences


def main() -> None:
    args = parse_args()
    base_params = load_params(args.params_json)
    for key, value in SEARCH_SPACE.items():
        base_params.setdefault(key, value[0])
    base_params.setdefault("max_age_phys", None)
    base_params.setdefault("r_slope", 0.0)
    base_params.setdefault("alpha_decay", 1.0)

    sequences = prepare(args)
    scenarios = [s for s in args.scenarios.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    print(f"[setup] {len(sequences)} sequences | scenarios={scenarios} | seeds={seeds}")

    started = time.time()

    # ------------------------------------------------------------------
    # The per-frame ceiling: every camera frame delivered, the frozen params.
    # It is the anchor of every delta reported below: it is scored on the SAME
    # delivered frame subset, so the delta isolates the tracker, not the frame set.
    # ------------------------------------------------------------------
    print("[setup] running the per-frame ceiling ...")
    ceiling_runs = {seq: run_pipeline(seq, base_params, None, False) for seq in sequences}
    ceiling_all = [score_run(run, None) for run in ceiling_runs.values()]
    ceiling_scores: dict[tuple[str, int, str], dict] = {}
    for scenario in scenarios:
        for seed in seeds:
            for sequence in sequences:
                delivered = sample_delivered_frames(
                    _FRAMES[sequence], scenario, seed, _MEASURED
                )
                ceiling_scores[(scenario, seed, sequence)] = score_run(
                    ceiling_runs[sequence], delivered
                )

    def ceiling_aggregate(scenario: str) -> dict:
        rows = [ceiling_scores[(scenario, seed, seq)] for seed in seeds for seq in sequences]
        tp, fp, fn, frames = _aggregate_counts(rows)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"f1": f1, "precision": precision, "recall": recall,
                "far_per_frame": fp / frames if frames else 0.0,
                "tp": tp, "fp": fp, "fn": fn, "frames": frames}

    pool = None
    if args.workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("fork").Pool(args.workers)

    report: dict = {
        "meta": {
            "cache_root": args.cache_root,
            "labels_root": args.labels_root,
            "sequences": sequences,
            "scenarios": scenarios,
            "seeds": seeds,
            "max_frames": args.max_frames,
            "base_params": base_params,
            "scenarios_spec": SCENARIOS,
        },
        "arms": {},
        "ceiling": {},
    }

    # ---- arm (1): the per-frame ceiling ------------------------------------
    tp = sum(r["tp_all"] for r in ceiling_all)
    fp = sum(r["fp_all"] for r in ceiling_all)
    fn = sum(r["fn_all"] for r in ceiling_all)
    frames = sum(r["total_frames"] for r in ceiling_all)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    report["arms"]["1_perframe_ceiling"] = {
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "precision": precision,
        "recall": recall,
        "far_per_frame": fp / frames if frames else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "frames": frames,
    }
    for scenario in scenarios:
        report["ceiling"][scenario] = ceiling_aggregate(scenario)
    print(f"[arm1] per-frame ceiling: F1={report['arms']['1_perframe_ceiling']['f1']:.6f} "
          f"FAR={report['arms']['1_perframe_ceiling']['far_per_frame']:.6f}")
    for scenario in scenarios:
        row = report["ceiling"][scenario]
        print(f"[arm1]   {scenario} (on the delivered subset): F1={row['f1']:.6f} "
              f"FAR={row['far_per_frame']:.6f}")

    # ---- arm (2a): jumped, DENSE time base == what the shipped SDK does today ----
    dense = evaluate_params(base_params, sequences, scenarios, seeds, pool, dense_time_base=True)
    report["arms"]["2a_jump_dense_timebase"] = dense["summary"]
    for scenario in scenarios:
        row = dense["summary"][scenario]
        print(f"[arm2a] {scenario} (dense time base): F1={row['f1_delivered_agg']:.6f} "
              f"FAR={row['far_delivered_agg']:.6f}")

    # ---- arm (2b): jumped, PHYSICAL time base, frozen parameters --------------
    untuned = evaluate_params(base_params, sequences, scenarios, seeds, pool)
    report["arms"]["2b_jump_physical_untuned"] = untuned["summary"]
    report.setdefault("per_sequence", {})
    report["per_sequence"]["2a_jump_dense_timebase"] = {
        scenario: _per_sequence_rows([r for r in dense["rows"] if r["scenario"] == scenario])
        for scenario in scenarios
    }
    report["per_sequence"]["2b_jump_physical_untuned"] = {
        scenario: _per_sequence_rows([r for r in untuned["rows"] if r["scenario"] == scenario])
        for scenario in scenarios
    }
    for scenario in scenarios:
        row = untuned["summary"][scenario]
        print(f"[arm2b] {scenario} (physical time base, untuned): F1={row['f1_delivered_agg']:.6f} "
              f"FAR={row['far_delivered_agg']:.6f}")

    # ---- arm (3): jumped, PHYSICAL time base, optimised ---------------------
    if args.phase == "search":
        space = dict(SEARCH_SPACE)
        if args.space_json:
            with open(args.space_json, "r", encoding="utf-8") as handle:
                space.update(json.load(handle))
        ceiling_far = sum(report["ceiling"][s]["far_per_frame"] for s in scenarios) / len(scenarios)
        best_params, history = coordinate_search(
            base_params, sequences, scenarios, seeds, space, pool, ceiling_far,
            sweeps=args.sweeps,
        )
        report["search_history"] = history
        report["opt_params"] = best_params
        print(f"[search] best params: {json.dumps(best_params, sort_keys=True)}")
    else:
        best_params = dict(base_params)
        if args.opt_params_json:
            best_params.update(load_params(args.opt_params_json))
        report["opt_params"] = best_params

    optimised = evaluate_params(best_params, sequences, scenarios, seeds, pool)
    report["arms"]["3_jump_optimised"] = optimised["summary"]
    report["per_sequence"]["3_jump_optimised"] = {
        scenario: _per_sequence_rows([r for r in optimised["rows"] if r["scenario"] == scenario])
        for scenario in scenarios
    }
    for scenario in scenarios:
        row = optimised["summary"][scenario]
        ceiling_row = report["ceiling"][scenario]
        print(f"[arm3] {scenario}: F1={row['f1_delivered_agg']:.6f} "
              f"(ceiling {ceiling_row['f1']:.6f}, dF1={row['f1_delivered_agg'] - ceiling_row['f1']:+.6f}) "
              f"FAR={row['far_delivered_agg']:.6f} (ceiling {ceiling_row['far_per_frame']:.6f})")

    # ---- arm (5): the optimised parameters at the full rate (regression check) ----
    optimised_full = evaluate_params(best_params, sequences, ["perframe"], seeds, pool)
    report["arms"]["5_optimised_perframe"] = optimised_full["summary"]
    report["per_sequence"]["5_optimised_perframe"] = {
        "perframe": _per_sequence_rows(optimised_full["rows"]),
    }
    report["per_sequence"]["1_perframe_ceiling"] = _per_sequence_rows(ceiling_all)
    frozen_full = report["arms"]["1_perframe_ceiling"]
    opt_full = optimised_full["summary"]["perframe"]
    print(f"[arm5] optimised params, per-frame: F1={opt_full['f1_delivered_agg']:.6f} "
          f"(frozen {frozen_full['f1']:.6f}, dF1={opt_full['f1_delivered_agg'] - frozen_full['f1']:+.6f}) "
          f"FAR={opt_full['far_delivered_agg']:.6f} (frozen {frozen_full['far_per_frame']:.6f})")

    # ---- the cadence that was actually simulated ----------------------------
    report["cadence"] = {
        scenario: cadence_stats(
            sample_delivered_frames(_FRAMES[sequences[0]], scenario, seeds[0], _MEASURED)
        )
        for scenario in scenarios
    }
    report["elapsed_s"] = round(time.time() - started, 1)

    if pool is not None:
        pool.close()
        pool.join()

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"[out] {args.out_json}")
    if args.params_out:
        with open(args.params_out, "w", encoding="utf-8") as handle:
            json.dump({"params": best_params}, handle, indent=2, ensure_ascii=False)
        print(f"[out] {args.params_out}")


if __name__ == "__main__":
    main()
