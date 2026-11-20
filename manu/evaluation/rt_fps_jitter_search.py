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

TARGET-FPS SCENARIOS (``f*``)
-----------------------------
``s1``~``s4`` answer "how bad is the cadence we measured".  They do NOT answer
"at which delivery rate does the tracker break", because their gaps are all *slower*
than the measured board cadence (mean dN 3.36 / 5.23 / 7.00 / 1.95) while the
per-frame arm (dN == 1) is the only fast reference.  The whole interesting band
between them -- delivery rates between 25 fps (dN 1.0) and the measured 8.03 fps
(dN 3.11) -- was never sampled.

``f25`` / ``f20`` / ``f17`` / ``f14`` / ``f12`` fill that band.  They are NOT a
re-parameterisation of the slowdown model: their long-run mean gap is pinned to
``CAMERA_FPS / target_fps`` camera frames (25/20/17/14/12 fps => mean dN
1.00 / 1.25 / 1.47 / 1.79 / 2.08), while the *shape* of the jitter is the measured
one (every measured gap is normalised by the measured mean and re-scaled to the
target mean).  A fractional accumulator carries the rounding remainder, so the
long-run mean gap is the target mean and not a rounded approximation of it.
The achieved mean dN and the achieved fps are reported per scenario in
``cadence`` -- quote those, never the target.

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
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.render_search_report_md import render_report_md  # noqa: E402
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
# The camera is a fixed 25 fps source: every gap below is counted in camera frames.
CAMERA_FPS = 25.0
# The delivery-rate band between the per-frame arm (25 fps) and the measured board
# cadence (8.03 fps).  See the module docstring: this band was never sampled.
for _target in (25.0, 20.0, 17.0, 14.0, 12.0):
    SCENARIOS[f"f{int(_target)}"] = {
        "kind": "target_fps",
        "target_fps": _target,
        "label": f"target {_target:g} fps (mean dN {CAMERA_FPS / _target:.2f})",
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
_RECORDS_FULL: dict[str, list[dict]] = {}
_MEASURED: list[int] = []
_FRAMES: dict[str, int] = {}
# Which caliber the PER-FRAME arm (the delivery caliber: every camera frame of every
# sequence) is measured on.  "delivery" = the full cache -- the red line is defined
# there, so the constraint that guards it must be measured there too.  "truncated" =
# the `--max-frames` search substrate; it is a cheap screen, but it is NOT the
# delivery caliber and a parameter set can pass it while breaking the red line.
_PERFRAME_CALIBER: str = "delivery"


# ---------------------------------------------------------------------------
# 1. Cadence sampling
# ---------------------------------------------------------------------------
def load_measured_delta_n(path: Path = MEASURED_DELTA_N_JSON) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        return list(json.load(handle)["delta_n_sequence"])


def _target_fps_gaps(measured: list[int], target_fps: float, rng: random.Random) -> list[int]:
    """Integer gaps whose long-run mean is ``CAMERA_FPS / target_fps`` camera frames.

    The measured cadence supplies the *shape*: every measured gap is divided by the
    measured mean and re-scaled to the target mean, so the scheduler jitter keeps its
    dispersion while the rate moves to the target.  A fractional accumulator carries
    the rounding remainder (the invariant is ``debt = earned - consumed``), so the
    long-run mean gap is the target mean and not a rounded approximation of it.

    The seed picks a rotation of the measured sequence, so the seeds are different
    jitter realisations of the same rate.
    """
    mean_measured = sum(measured) / len(measured)
    # The 8 % adjacent swap makes the engine DROP the late frame, so only
    # (1 - SWAP_PROB) of the sampled frames reach the tracker: the rate the
    # tracker sees is (1 - SWAP_PROB) x the sampled rate.  The sampled gap is
    # therefore narrowed by that factor, so the ACHIEVED rate is the requested
    # one.  (The achieved rate is measured and reported per scenario anyway --
    # quote the measurement, not the target.)
    target_gap = (CAMERA_FPS / target_fps) * (1.0 - SWAP_PROB)
    offset = rng.randrange(len(measured))
    gaps: list[int] = []
    debt = 0.0
    for index in range(len(measured)):
        shape = measured[(index + offset) % len(measured)] / mean_measured
        debt += target_gap * shape
        gap = math.floor(debt)
        if gap < 1:
            gap = 1
        debt -= gap
        gaps.append(int(gap))
    return gaps


def sample_delivered_frames(
    total_frames: int, scenario: str, seed: int, measured: list[int]
) -> list[int]:
    """Return the camera frame ids the detector delivers, in *arrival* order.

    The arrival order may contain an adjacent swap (8 %); the caller feeds the frames
    to the engine in ascending order, and the engine's ``fid <= last_fid`` guard drops
    the late one, which is why the returned list is filtered here as well.
    """
    spec = SCENARIOS[scenario]
    # zlib.crc32, NOT hash(): CPython randomises str hashing per process
    # (PYTHONHASHSEED), so hash(scenario) made the cadence sample -- and therefore
    # the whole search -- irreproducible between runs.
    rng = random.Random(zlib.crc32(scenario.encode("utf-8")) * 1_000_003 + seed)

    gaps = (_target_fps_gaps(measured, float(spec["target_fps"]), rng)
            if spec["kind"] == "target_fps" else None)

    frames: list[int] = []
    current = 0
    index = 0
    while current < total_frames:
        frames.append(current)
        if gaps is not None:
            step = gaps[index % len(gaps)]
        elif spec["kind"] == "doc":
            step = rng.choices([w[0] for w in DOC_DELTA_N_WEIGHTS],
                               [w[1] for w in DOC_DELTA_N_WEIGHTS])[0]
        else:
            step = rng.choice(measured)
            sigma = float(spec.get("sigma", 0.0))
            if sigma > 0.0:
                step = step * math.exp(rng.gauss(0.0, sigma))
            step = max(1, int(round(step * float(spec.get("slowdown", 1.0)))))
        current += step
        index += 1

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
    records: list[dict] | None = None,
    caliber: str | None = None,
) -> dict:
    """Run the system pipeline on one sequence and return its per-frame detections.

    ``delivered`` is ``None`` for the per-frame ceiling arm (every camera frame is
    delivered).  ``dense_time_base`` selects the engine's time base:

    * ``False`` (physical): the engine sees the camera frame NUMBER, so a skipped frame
      is a real time gap -- this is what the delta-n aware engine must survive.
    * ``True`` (dense): the engine sees a dense update counter, i.e. exactly what the
      shipped SDK does today (it cannot see the physical gaps at all).
    """
    if records is None:
        cal = caliber or _PERFRAME_CALIBER
        # delivered is None  <=>  the per-frame arm.  It is the delivery caliber.
        # The jump arms keep the truncated universe (--max-frames): that is the
        # search substrate the cadence sampler draws from.
        records = (
            _RECORDS_FULL[sequence]
            if delivered is None and cal == "delivery"
            else _RECORDS[sequence]
        )
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
    sequence, scenario, seed, params, delivered, dense, caliber = task
    run = run_pipeline(sequence, params, delivered, dense, caliber=caliber)
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
    perframe_caliber: str | None = None,
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
                tasks.append(
                    (sequence, scenario, seed, params, delivered, dense_time_base,
                     perframe_caliber)
                )

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
def objective(
    result: dict,
    ceiling_far: float,
    perframe: dict | None = None,
    base_perframe_f1: float | None = None,
    far_redline: float = 0.03,
    perframe_f1_tol: float = 0.003,
    lambda_far: float = 2.0,
    mu_far: float = 25.0,
    nu_f1: float = 25.0,
) -> float:
    """The search objective.

    Three terms, all in the frame-weighted F1 unit:

    * the jump-arm F1 (what the board actually sees),
    * a penalty when the jump-arm FAR grows above the frozen per-frame ceiling's,
    * a HARD penalty when the PER-FRAME arm (delta-n == 1, the 25 fps delivery
      caliber) breaks the delivery red line or regresses against the frozen
      per-frame F1.

    The third term is what the first search missed: optimising the jump arm alone
    happily selects a parameter set whose per-frame FAR blows the red line (the
    "gapfit" set: FAR 0.0351 > 0.03, F1 -0.38 pp against v3).  The penalties are
    scaled so that one red-line breach outweighs any plausible F1 gain.
    """
    summary = result["summary"]
    score = summary["_jump_mean_f1"]
    far_penalty = max(0.0, summary["_jump_mean_far"] - ceiling_far) * lambda_far
    score -= far_penalty
    if perframe is not None:
        pf = perframe["summary"]["perframe"]
        score -= max(0.0, pf["far_delivered_agg"] - far_redline) * mu_far
        if base_perframe_f1 is not None:
            score -= max(
                0.0, (base_perframe_f1 - perframe_f1_tol) - pf["f1_delivered_agg"]
            ) * nu_f1
    return score


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
    constraints: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Coordinate descent over the search space.

    Every candidate is scored on the jump scenarios AND on the per-frame arm, so
    that the objective can reject a parameter set which wins the jump arm but
    breaks the 25 fps delivery caliber.
    """
    constraints = dict(constraints or {})

    def evaluate_candidate(candidate: dict) -> tuple[float, dict, dict | None]:
        result = evaluate_params(candidate, sequences, scenarios, seeds, pool)
        perframe = evaluate_params(
            candidate, sequences, ["perframe"], [seeds[0]], pool
        )
        score = objective(result, ceiling_far, perframe, **constraints)
        return score, result, perframe

    current = dict(base_params)
    best, _, perframe_start = evaluate_candidate(current)
    history = [{
        "params": dict(current),
        "objective": best,
        "stage": "start",
        "perframe": {
            "f1": perframe_start["summary"]["perframe"]["f1_delivered_agg"],
            "far": perframe_start["summary"]["perframe"]["far_delivered_agg"],
        },
    }]
    log(f"[search] start objective={best:.6f}")

    # `sweeps == 0` means "until convergence".  A fixed budget cuts the descent
    # short whenever the last sweep still improves -- and a coordinate descent that
    # is cut short never revisits the corners it had not reached yet (the
    # max_age_phys x match_dist corner that the delivered set lives in was exactly
    # such an unreached corner).  A hard cap is still allowed, but it warns.
    max_sweeps = sweeps if sweeps > 0 else 50
    for sweep in range(max_sweeps):
        improved = False
        for key, values in space.items():
            base_value = current.get(key)
            for value in values:
                if value == base_value:
                    continue
                candidate = dict(current)
                candidate[key] = value
                score, result, perframe = evaluate_candidate(candidate)
                entry = {
                    "params": dict(candidate),
                    "objective": score,
                    "stage": f"sweep{sweep}:{key}={value}",
                    "summary": {k: v for k, v in result["summary"].items() if not k.startswith("_")},
                }
                if perframe is not None:
                    entry["perframe"] = {
                        "f1": perframe["summary"]["perframe"]["f1_delivered_agg"],
                        "far": perframe["summary"]["perframe"]["far_delivered_agg"],
                    }
                history.append(entry)
                if score > best + 1e-9:
                    best = score
                    current = candidate
                    improved = True
                    log(f"[search] sweep{sweep} {key}={value} -> objective={score:.6f} (better)")
        if not improved:
            log(f"[search] sweep{sweep} converged")
            break
    else:
        log(f"[search] WARNING: the sweep budget ({max_sweeps}) ran out while the last "
            "sweep still improved -- the descent was CUT SHORT and the result is a LOCAL "
            "optimum. Re-run with --sweeps 0 (sweep until convergence).")

    # The objective already punishes a red-line breach, but a punished candidate can
    # still win if its jump-arm gain is big enough.  The DELIVERED parameter set is
    # the best FEASIBLE one: per-frame FAR <= red line AND per-frame F1 not below
    # the frozen base (minus the tolerance).
    redline = constraints.get("far_redline", 0.03)
    tol = constraints.get("perframe_f1_tol", 0.001)
    base_f1 = constraints.get("base_perframe_f1")
    feasible = [
        entry for entry in history
        if entry.get("perframe")
        and entry["perframe"]["far"] <= redline
        and (base_f1 is None or entry["perframe"]["f1"] >= base_f1 - tol)
    ]
    if feasible:
        best_feasible = max(feasible, key=lambda entry: entry["objective"])
        if best_feasible["params"] != current:
            log(f"[search] objective-best violates the per-frame constraints -> "
                f"switch to the best feasible candidate "
                f"(objective={best_feasible['objective']:.6f}, "
                f"perframe F1={best_feasible['perframe']['f1']:.6f} "
                f"FAR={best_feasible['perframe']['far']:.6f})")
        current = dict(best_feasible["params"])
    else:
        log("[search] WARNING: no feasible candidate -- the base parameters are returned")
    log(f"[search] selected params: {json.dumps(current, sort_keys=True)}")
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
    parser.add_argument("--sweeps", type=int, default=0,
                        help="Coordinate-descent sweeps. 0 (default) = keep sweeping until "
                             "convergence; N > 0 = hard cap (the run warns loudly when the "
                             "cap cuts the descent short while the last sweep still improved)")
    parser.add_argument("--perframe-caliber", choices=["delivery", "truncated"],
                        default="delivery",
                        help="Caliber of the per-frame arm that carries the hard constraint: "
                             "'delivery' (default) = the full cache, i.e. the caliber the "
                             "red line is defined on; 'truncated' = the --max-frames screen")
    parser.add_argument("--space-json", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--params-out", default="")
    parser.add_argument("--opt-params-json", default="")
    # --- the delivery-constraint knobs of the objective -------------------------
    parser.add_argument("--far-redline", type=float, default=0.03,
                        help="Absolute FAR red line the per-frame arm must respect")
    # How much per-frame F1 the delivered set may give up against the frozen base.
    # 0.001 was far too tight: relaxing confirm_gap_tol (which is what makes the
    # jump arm work at all) costs ~0.15-0.28 pp of per-frame F1, so a 0.001
    # allowance rejected EVERY jump-capable candidate and the search fell back to
    # the v3 base -- whose jump arm scores 0.0841 (all tracks killed; the reproducible seedfix run).  0.003
    # admits the jump-capable candidates while still rejecting the degenerate
    # ones; the FAR red line stays the hard filter.
    parser.add_argument("--perframe-f1-tol", type=float, default=0.003,
                        help="How much per-frame F1 regression below the base is tolerated")
    parser.add_argument("--lambda-far", type=float, default=2.0,
                        help="Weight of the jump-arm FAR penalty")
    parser.add_argument("--mu-far", type=float, default=25.0,
                        help="Weight of the per-frame FAR red-line penalty (hard)")
    parser.add_argument("--nu-f1", type=float, default=25.0,
                        help="Weight of the per-frame F1 no-regression penalty (hard)")
    return parser.parse_args()


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        blob = json.load(handle)
    params = dict(blob.get("params", blob))
    return params


def prepare(args: argparse.Namespace) -> list[str]:
    global _MEASURED, _PERFRAME_CALIBER
    _MEASURED = load_measured_delta_n()
    _PERFRAME_CALIBER = args.perframe_caliber

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
        # The full cache is the DELIVERY caliber (the red line is defined on it);
        # the truncated view below is the search substrate for the jump arms.
        _RECORDS_FULL[sequence] = records
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
    # The ceiling stays on the truncated search substrate: it is the anchor of the
    # jump-arm deltas, whose frame universe is the truncated one.  The DELIVERY
    # caliber numbers live in report["delivery_caliber"] below.
    ceiling_runs = {
        seq: run_pipeline(seq, base_params, None, False, records=_RECORDS[seq])
        for seq in sequences
    }
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
        # The frozen per-frame F1 of the BASE parameters: the no-regression anchor.
        # Same caliber as the candidate's per-frame arm (both pooled over the same
        # frames), so the constraint compares like with like.
        base_perframe = evaluate_params(
            base_params, sequences, ["perframe"], [seeds[0]], pool
        )
        base_perframe_f1 = base_perframe["summary"]["perframe"]["f1_delivered_agg"]
        base_perframe_far = base_perframe["summary"]["perframe"]["far_delivered_agg"]
        print(f"[setup] base per-frame arm: F1={base_perframe_f1:.6f} "
              f"FAR={base_perframe_far:.6f} (red line FAR<={args.far_redline:.4f})")
        constraints = {
            "base_perframe_f1": base_perframe_f1,
            "far_redline": args.far_redline,
            "perframe_f1_tol": args.perframe_f1_tol,
            "lambda_far": args.lambda_far,
            "mu_far": args.mu_far,
            "nu_f1": args.nu_f1,
        }
        best_params, history = coordinate_search(
            base_params, sequences, scenarios, seeds, space, pool, ceiling_far,
            sweeps=args.sweeps, constraints=constraints,
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
    optimised_full = evaluate_params(
        best_params, sequences, ["perframe"], seeds, pool, perframe_caliber="truncated"
    )
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

    # ---- the DELIVERY caliber: that is where the red line is defined --------
    # The hard per-frame constraint is measured on this caliber (--perframe-caliber).
    # It is the only caliber on which "FAR <= 0.03" and "F1 not below the v3 base"
    # are the delivery criterion, so the selected parameters are re-measured here
    # and the verdict goes into the report -- a search that ends on a parameter set
    # which breaks the red line must not be able to call itself a success.
    delivery_base = evaluate_params(
        base_params, sequences, ["perframe"], [seeds[0]], pool, perframe_caliber="delivery"
    )
    delivery_opt = evaluate_params(
        best_params, sequences, ["perframe"], [seeds[0]], pool, perframe_caliber="delivery"
    )
    _db = delivery_base["summary"]["perframe"]
    _do = delivery_opt["summary"]["perframe"]
    _floor = _db["f1_delivered_agg"] - args.perframe_f1_tol
    _far_ok = _do["far_delivered_agg"] <= args.far_redline
    _f1_ok = _do["f1_delivered_agg"] >= _floor
    report["delivery_caliber"] = {
        "note": "per-frame arm on the FULL cache (every camera frame of every sequence) "
                "-- the caliber the FAR red line is defined on",
        "redline_far": args.far_redline,
        "perframe_f1_tol": args.perframe_f1_tol,
        "base": {"f1": _db["f1_delivered_agg"], "far": _db["far_delivered_agg"]},
        "selected": {"f1": _do["f1_delivered_agg"], "far": _do["far_delivered_agg"]},
        "f1_floor": _floor,
        "far_ok": _far_ok,
        "f1_ok": _f1_ok,
        "feasible": _far_ok and _f1_ok,
    }
    print(f"[delivery] base    : F1={_db['f1_delivered_agg']:.6f} "
          f"FAR={_db['far_delivered_agg']:.6f}")
    print(f"[delivery] selected: F1={_do['f1_delivered_agg']:.6f} "
          f"FAR={_do['far_delivered_agg']:.6f} "
          f"(red line FAR<={args.far_redline:.4f}, F1 floor>={_floor:.6f}) "
          f"=> {'FEASIBLE' if _far_ok and _f1_ok else 'INFEASIBLE'}")

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
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as handle:
            handle.write(render_report_md(report))
        print(f"[out] {args.out_md}")
    if args.params_out:
        with open(args.params_out, "w", encoding="utf-8") as handle:
            json.dump({"params": best_params}, handle, indent=2, ensure_ascii=False)
        print(f"[out] {args.params_out}")


if __name__ == "__main__":
    main()
