#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Global hyper-parameter search for the Longquanshan system-SOTA pipeline.

The Anti-UAV system pipeline is ported to the Longquanshan business domain by
``manu/evaluation/system_sota_longquan.py``. This script searches the single global parameter set that maximises
the business-domain score over the whole cache, never a per-sequence parameter set.

Search method: coordinate descent over seven groups, two rounds. Inside a group every parameter is swept on its
own with the rest of the parameters frozen, and the group is left at its best combination. The objective is
ranked by ``(meets the delivery criteria, F1, -FAR)`` so that hitting F1 >= 88 %, Recall >= 85 % and
FAR <= 0.030 per frame always outranks a marginally better F1 that misses one of them.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.system_sota_longquan import (  # noqa: E402
    DEFAULT_PARAMS,
    IMG_H,
    aggregate_metrics,
    apply_persistence_filter,
    build_candidates,
    evaluate_biz,
    frame_payload,
    load_cache_meta,
    make_records,
    run_system_sota,
    single_frame_dets,
    write_sequence_predictions,
    write_summary,
)
from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402

# ==============================================================================
# 1. Search space
# ==============================================================================

SEARCH_SPACE = {
    # G1 thresholds
    # The th_base / th_ground grids were extended on 2026-10-22 (trial_0080 delivery work, §55.8/§55.9 of
    # manu/memory/archive/06_领域迁移复评_§五十四~§五十五.md).  The previous grids stopped at th_base=0.40
    # and had no th_ground=0.26 at all, so a blind run could never reach the delivered working point
    # (th_base=0.42 / th_ground=0.26 / confirm_consec=7) -- and the coordinate-descent step *skips a whole
    # group* when the current value is not in the candidate list ("if best_params[key] not in values: continue").
    # All previous values are kept, so the frozen trial_0474 workflow is unaffected.
    "th_base": [0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.40, 0.42, 0.45, 0.48, 0.52],
    "th_salvage": [0.03, 0.06, 0.09, 0.12],
    "th_ground": [0.18, 0.22, 0.26, 0.28, 0.35, 0.42, 0.50],
    "sky_ratio": [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
    # G2 tracking
    "min_hits": [2, 3, 4],
    "max_age": [3, 5, 8],
    "match_dist": [6.0, 9.0, 12.0, 16.0, 20.0],
    "max_match_dist": [12.0, 15.0, 18.0, 24.0],
    "instant_conf": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35],
    "min_disp": [0.0, 0.5, 1.0, 2.0, 2.5, 4.0],
    "min_track_score": [0.04, 0.08, 0.12, 0.16],
    # G3 smoothing
    "stitch_gap": [2, 4, 6, 8],
    "infill_gap": [1, 2, 3, 5],
    "min_hits_infill": [3, 5, 8],
    # G4 pruning
    "min_rigid_disp": [0.0, 0.5, 1.0, 2.0, 3.0],
    "max_rigid_var": [0.25, 0.5, 1.0, 1e9],
    "min_hits_prune": [4, 6, 8, 12],
    # G5 input
    # The radius has to cover the whole range: the small radii keep a rich candidate pool, the large radii
    # collapse it to the highest scoring candidate per frame, which is the frozen single-frame baseline.
    "dedup_radius": [8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0],
    # G6 Phase-A / P3 extensions (all OFF in the Anti-UAV golden default)
    "min_rigid_disp_sky": [-1.0, 0.0, 1.0, 2.0, 999.0],
    "hover_vel": [0.0, 0.8, 2.0, 4.0],
    "hover_infill_gap": [15, 30],
    "coast_frames": [0, 10, 15, 30],
    "min_hits_hover": [8, 12],
    "hover_sky_only": [0, 1],
    "stitch_long_gap": [0, 8, 12, 20],
    "stitch_vel_diff": [4.0, 8.0],
    "th_deep": [0.0, 0.035, 0.05],
    "min_hits_deep": [3, 4],
    # G7 continuity
    # confirm_consec gained 7 and 8 on 2026-10-22: the 54-candidate grid of §55.8 showed it is a real
    # lever (5/6/7 -> 7/18, 11/18, 13/18 of the candidates pass the delivery triple), and the delivered
    # working point uses 7, which the old grid could not even reach (the group is skipped when the
    # current value is missing from the list).
    "confirm_consec": [1, 2, 3, 4, 5, 6, 7, 8],
    "confirm_gap_tol": [0, 1, 2],
    "confirm_trim_prefix": [0, 1],
    "persist_min_hits": [0, 2, 3, 4],
    "persist_window": [0, 5, 9, 15],
    "persist_tol": [0.0, 4.0, 8.0, 12.0],
}

# The groups are searched in order of value density, so that a tight wall-clock budget drops the least
# valuable work first: the thresholds are the global lever, the continuity gate is the requirement the user
# named explicitly, the de-duplication radius decides the shape of the candidate pool, the rigid pruner is the
# risk in the hover domain, the tracker follows and the smoother (net gain on Anti-UAV) comes last. The Phase-A
# extensions are the most speculative and therefore run last of all.
GROUPS = [
    ("G1_thresholds", ["th_base", "th_salvage", "th_ground", "sky_ratio"]),
    (
        "G7_continuity",
        [
            "confirm_consec",
            "confirm_gap_tol",
            "confirm_trim_prefix",
            "persist_min_hits",
            "persist_window",
            "persist_tol",
        ],
    ),
    ("G5_input", ["dedup_radius"]),
    ("G4_pruning", ["min_rigid_disp", "max_rigid_var", "min_hits_prune"]),
    (
        "G2_tracking",
        ["min_hits", "max_age", "match_dist", "max_match_dist", "instant_conf", "min_disp", "min_track_score"],
    ),
    ("G3_smoothing", ["stitch_gap", "infill_gap", "min_hits_infill"]),
]

# The Phase-A extensions are not searched in this round, but they stay selectable with --groups G6_extensions.
EXTRA_GROUPS = [
    (
        "G6_extensions",
        [
            "min_rigid_disp_sky",
            "hover_vel",
            "hover_infill_gap",
            "coast_frames",
            "min_hits_hover",
            "hover_sky_only",
            "stitch_long_gap",
            "stitch_vel_diff",
            "th_deep",
            "min_hits_deep",
        ],
    ),
]

# Criteria for the business-domain delivery, checked before the F1 ranking.
TARGET_F1 = 0.88
TARGET_RECALL = 0.85
TARGET_FAR = 0.030

# Representative subset for the coarse stage: four failure modes plus two good sequences.
SUBSET = [
    "VIDEO00009_19700101_003002",  # complex urban background, hover recall collapse
    "VIDEO00008_19700101_002001",  # complex urban background, hover recall collapse
    "VIDEO00032_19700101_014453",  # black hot
    "VIDEO00004_19700101_002438",  # uncalibrated / fog
    "VIDEO00010_19700101_001550",  # tree background false positives
    "VIDEO00005_19700101_002959",  # tree background false positives
    "VIDEO00034_19700101_015052",  # good sequence, full marks
    "VIDEO00009_19700101_001531",  # good sequence, sky
]

# The manual review marked ten sequences as out of domain (black hot, infrared not calibrated or fog, and one
# out-of-domain viewpoint). They stay in the search objective because the frozen baseline was measured over all
# 24 sequences, but they are also reported on their own so the in-domain picture is visible.
IN_DOMAIN_SUFFIXES = (
    "001415",
    "000703",
    "001810",
    "000818",
    "001307",
    "002001",
    "001531",
    "003002",
    "001550",
    "013631",
    "014053",
    "014144",
    "015052",
    "002959",
)
OUT_OF_DOMAIN_SUFFIXES = (
    "002111",
    "002438",
    "000322",
    "001324",
    "002251",
    "014227",
    "014453",
    "015000",
    "015138",
    "015740",
)

# Wall-clock seconds kept aside for the ablation, the final evaluation and the prediction files.
TAIL_RESERVE_SECONDS = 420.0

PROBE_SEQUENCES = [
    "VIDEO00006_19700101_000322",  # 17017 frames
    "VIDEO00012_19700101_002251",  # 15565 frames
]


# ==============================================================================
# 2. Candidate store
# ==============================================================================


class CandidateStore:
    """Cache the candidate pools of every sequence for every de-duplication radius.

    The cache pickle of a sequence is read once, the three per-scale rows are merged for every radius once,
    and the ground truth is read once. The per-frame records are rebuilt only when the radius or the
    persistence settings change.
    """

    def __init__(self, cache_root, labels_root, sequences: list[str], radii: list[float]):
        self.cache_root = Path(cache_root)
        self.labels_root = Path(labels_root)
        self.sequences = list(sequences)
        self.radii = sorted({float(radius) for radius in radii})
        self._candidates: dict[str, dict[float, dict]] = {}
        self._truth: dict[str, dict] = {}
        self._records: dict[str, list] = {}
        self._record_key: tuple | None = None

    def _load(self, sequence: str) -> None:
        if sequence in self._candidates:
            return
        by_radius = {}
        for radius in self.radii:
            by_radius[radius] = build_candidates(self.cache_root, sequence, radius, 0.0)
        self._candidates[sequence] = by_radius
        meta = load_cache_meta(self.cache_root, sequence)
        width, height = meta["native_size"]
        self._truth[sequence] = load_ground_truth(self.labels_root, sequence, width, height)

    def candidates(self, sequence: str, radius: float) -> dict:
        self._load(sequence)
        return self._candidates[sequence][float(radius)]

    def warm(self, sequences: list[str], radius: float, persist_key: tuple) -> None:
        """Load every sequence, every radius and the records of ``radius`` before the workers are forked.

        The fork workers then inherit the whole store copy-on-write, so no cache data is ever pickled and no
        worker re-reads a cache pickle.
        """
        for sequence in sequences:
            self._load(sequence)
            self.records(sequence, radius, persist_key)

    def frames(self, sequence: str) -> int:
        return int(load_cache_meta(self.cache_root, sequence)["frames"])

    def records(self, sequence: str, radius: float, persist_key: tuple) -> list[dict]:
        """Per-frame records of one sequence, rebuilt only when the radius or the persistence settings change."""
        key = (float(radius), persist_key)
        if self._record_key != key:
            self._records = {}
            self._record_key = key
        if sequence not in self._records:
            candidates = self.candidates(sequence, radius)
            candidates = apply_persistence_filter(candidates, *persist_key)
            self._records[sequence] = make_records(candidates, self._truth[sequence], sequence)
        return self._records[sequence]


def persist_key(params: dict) -> tuple:
    """The three parameters that drive the input-side persistence filter."""
    return (
        int(params["persist_min_hits"]),
        int(params["persist_window"]),
        float(params["persist_tol"]),
    )


# ==============================================================================
# 3. Evaluation
# ==============================================================================


def rank_key(row: dict) -> tuple:
    """Delivery criteria first, then F1, then the lower FAR."""
    meets = row["f1"] >= TARGET_F1 and row["recall"] >= TARGET_RECALL and row["far_per_frame"] <= TARGET_FAR
    return (1 if meets else 0, row["f1"], -row["far_per_frame"])


def meets_criteria(row: dict) -> bool:
    """Does the row satisfy F1 >= 88 %, Recall >= 85 % and FAR <= 0.030 per frame?"""
    return row["f1"] >= TARGET_F1 and row["recall"] >= TARGET_RECALL and row["far_per_frame"] <= TARGET_FAR


# The candidate store lives in a module global so that the fork workers inherit it copy-on-write; nothing of it
# is ever pickled. It is assigned before the worker pool is created.
_STORE: CandidateStore | None = None


def _evaluate_candidate(task: tuple) -> dict:
    """Evaluate one parameter candidate over a sequence set.

    Module level on purpose: the worker pool pickles the function by name. The store is inherited through the
    fork context, so a worker never rebuilds it and never re-reads a cache pickle.
    """
    params, sequences, tag, capture = task
    store = _STORE
    if store is None:
        raise RuntimeError("the candidate store is not initialised")
    started = time.perf_counter()
    per_sequence = []
    per_sequence_rows = {}
    detections = {}
    reports = {}
    num_frames = 0
    key = persist_key(params)
    for sequence in sequences:
        records = store.records(sequence, float(params["dedup_radius"]), key)
        dets, res = run_system_sota(records, params, img_h=IMG_H)
        frames = store.frames(sequence)
        per_sequence_rows[sequence] = evaluate_biz(records, dets, num_frames=frames)
        per_sequence.append(per_sequence_rows[sequence])
        num_frames += frames
        if capture:
            detections[sequence] = dets
            reports[sequence] = res
    overall = aggregate_metrics(per_sequence, num_frames)
    result = {
        "tag": tag,
        "params": dict(params),
        "metrics": overall,
        "per_sequence": per_sequence_rows,
        "num_frames": num_frames,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if capture:
        result["detections"] = detections
        result["reports"] = reports
    return result


class Evaluator:
    """Runs the system pipeline over a sequence set and aggregates the business-domain counts.

    The candidates of a group are evaluated as one batch: sequentially with ``--jobs 1`` and, when a fork pool
    is available, one worker per candidate value. The counts are accumulated as integers, so the parallel and
    the sequential path produce bit-identical metrics.
    """

    def __init__(self, store: CandidateStore, pool=None, max_evals: int = 0):
        self.store = store
        self.pool = pool
        self.sweep: list[dict] = []
        self.evaluations = 0
        self.max_evals = max_evals

    def exhausted(self) -> bool:
        """Has the evaluation budget been used up?"""
        return bool(self.max_evals) and self.evaluations >= self.max_evals

    def run_batch(
        self,
        tasks: list[tuple],
        sequences: list[str],
        capture: bool = False,
        truncate: bool = True,
    ) -> list[dict]:
        """Evaluate a batch of parameter candidates and record every evaluation in the sweep log.

        Inside the search a batch never exceeds the remaining evaluation budget, so --max-evals truncates the
        batch instead of running it as a whole; the truncation is the same for the sequential and for the
        parallel path. The control, the ablation and the final evaluation pass ``truncate=False`` so that the
        report is always complete, even when the search ran out of budget.
        """
        if truncate and self.max_evals:
            room = max(0, self.max_evals - self.evaluations)
            tasks = tasks[:room]
        if not tasks:
            return []
        payload = [(params, sequences, tag, capture) for params, tag in tasks]
        if self.pool is None:
            results = [_evaluate_candidate(item) for item in payload]
        else:
            results = self.pool.map(_evaluate_candidate, payload)
        for result in results:
            self._record(result)
        return results

    def run_local(self, tasks: list[tuple], sequences: list[str], capture: bool = False) -> list[dict]:
        """Evaluate a batch inside this process, so the per-frame detections can be written out afterwards."""
        payload = [(params, sequences, tag, capture) for params, tag in tasks]
        if not payload:
            return []
        results = [_evaluate_candidate(item) for item in payload]
        for result in results:
            self._record(result)
        return results

    def _record(self, result: dict) -> None:
        overall = result["metrics"]
        self.evaluations += 1
        entry = {
            "tag": result["tag"],
            "params": result["params"],
            "sequences": len(result["per_sequence"]),
            "frames": result["num_frames"],
            **{key: overall[key] for key in ("tp", "fp", "fn", "fn_absent", "fn_compete", "det", "gt")},
            "recall": overall["recall"],
            "precision": overall["precision"],
            "f1": overall["f1"],
            "far_per_frame": overall["far_per_frame"],
            "meets_criteria": meets_criteria(overall),
            "elapsed_seconds": result["elapsed_seconds"],
        }
        self.sweep.append(entry)
        print(
            f"[EVAL {self.evaluations:04d}] {result['tag']} TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
            f"R={overall['recall'] * 100:.2f}% P={overall['precision'] * 100:.2f}% "
            f"F1={overall['f1'] * 100:.3f}% FAR={overall['far_per_frame']:.4f} "
            f"meets={'Y' if meets_criteria(overall) else 'N'} {result['elapsed_seconds']:.1f}s",
            flush=True,
        )


def single_frame_control(store: "CandidateStore", sequences: list[str], params: dict) -> tuple[dict, dict]:
    """The raw candidate pool at the same radius and threshold, without any system post-processing.

    This separates the two effects of the final parameter set: the shape of the candidate pool comes from the
    de-duplication radius, the temporal gain comes from the tracker, the smoother and the continuity gate.
    """
    rows = {}
    per_sequence = []
    num_frames = 0
    for sequence in sequences:
        records = store.records(sequence, float(params["dedup_radius"]), persist_key(params))
        dets = single_frame_dets(records, float(params["th_base"]))
        frames = store.frames(sequence)
        rows[sequence] = evaluate_biz(records, dets, num_frames=frames)
        per_sequence.append(rows[sequence])
        num_frames += frames
    return aggregate_metrics(per_sequence, num_frames), rows


# ==============================================================================
# 4. Coordinate descent
# ==============================================================================


class Budget:
    """Wall-clock budget of the whole run.

    The deadline is measured with ``time.monotonic`` so that it is a real wall-clock constraint, also while the
    candidate sweep runs in parallel worker processes.
    """

    def __init__(self, seconds: float, reserve: float = 0.0):
        self.seconds = float(seconds)
        self.reserve = float(reserve)
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        """Wall-clock seconds left before the hard deadline."""
        return self.seconds - self.elapsed()

    def search_remaining(self) -> float:
        """Wall-clock seconds left for the search, the tail reserve excluded."""
        return self.remaining() - self.reserve

    def expired(self) -> bool:
        """Has the search deadline been reached? The tail reserve is still available for the final steps."""
        return self.search_remaining() <= 0.0


def coordinate_descent(
    evaluator: Evaluator,
    params: dict,
    sequences: list[str],
    groups: list[tuple[str, list[str]]],
    rounds: int,
    tag_prefix: str,
    budget: "Budget | None" = None,
) -> tuple[dict, dict, int, dict, list[float]]:
    """Sweep group by group, parameter by parameter, and keep the best combination found.

    Inside a group the candidates of every parameter are evaluated against the same baseline snapshot, in one
    parallel batch, and the results are then applied greedily in parameter order: the value of a parameter is
    only taken when it improves the running best, and the running best is the baseline of the next parameter.
    Evaluating the whole group at once is what keeps the wall-clock budget: the parameters of a group do not
    depend on each other until their results are applied, and a single evaluation of the full cache costs about
    40 s, so a per-parameter batch would burn the whole budget on the first round.

    Returns the best parameters, the best metrics, the number of completed rounds, the per-parameter improvement
    of the last round and the wall-clock seconds of every round.
    """
    best_params = dict(params)
    results = evaluator.run_batch([(best_params, f"{tag_prefix}/start")], sequences, truncate=False)
    best_row = results[0]["metrics"]
    rounds_completed = 0
    improvements: dict[str, tuple] = {}
    round_seconds: list[float] = []

    for round_index in range(rounds):
        round_start = time.monotonic()
        for group_name, keys in groups:
            if evaluator.exhausted() or (budget is not None and budget.expired()):
                break
            group_start = time.monotonic()
            group_start_f1 = best_row["f1"]
            tasks = []
            for key in keys:
                for value in SEARCH_SPACE[key]:
                    if value == best_params[key]:
                        continue
                    trial = dict(best_params)
                    trial[key] = value
                    tasks.append((trial, key, value, f"{tag_prefix}/r{round_index + 1}/{group_name}/{key}={value}"))
            if not tasks:
                continue
            results = evaluator.run_batch([(task[0], task[3]) for task in tasks], sequences)
            grouped: dict[str, list] = {}
            for task, result in zip(tasks, results):
                grouped.setdefault(task[1], []).append(result)
            for key in keys:
                previous_value, previous_f1 = best_params[key], best_row["f1"]
                for result in grouped.get(key, []):
                    if rank_key(result["metrics"]) > rank_key(best_row):
                        best_params, best_row = result["params"], result["metrics"]
                if best_params[key] != previous_value:
                    improvements[key] = (previous_value, best_params[key], previous_f1, best_row["f1"])
                    print(
                        f"[ROUND {round_index + 1}] {group_name:16s} {key}: {previous_value}->{best_params[key]} "
                        f"(F1 {previous_f1 * 100:.3f}%->{best_row['f1'] * 100:.3f}%)",
                        flush=True,
                    )
            print(
                f"[ROUND {round_index + 1}] {group_name:16s} done | F1={best_row['f1'] * 100:.3f}% "
                f"R={best_row['recall'] * 100:.2f}% P={best_row['precision'] * 100:.2f}% "
                f"FAR={best_row['far_per_frame']:.4f} "
                f"(dF1={(best_row['f1'] - group_start_f1) * 100:+.3f}pp) | "
                f"组内用时 {time.monotonic() - group_start:.0f}s",
                flush=True,
            )
        rounds_completed = round_index + 1
        round_seconds.append(time.monotonic() - round_start)
        remaining = budget.search_remaining() if budget is not None else float("inf")
        print(
            f"[ROUND {round_index + 1}] 完成,最优 F1={best_row['f1'] * 100:.3f}%,"
            f"累计用时 {budget.elapsed():.0f}s,预算剩余 {remaining:.0f}s",
            flush=True,
        )
        if evaluator.exhausted() or (budget is not None and budget.expired()):
            reason = "评估次数预算" if evaluator.exhausted() else "墙钟预算"
            print(f"[ROUND] {reason}耗尽,停止坐标下降", flush=True)
            break
    return best_params, best_row, rounds_completed, improvements, round_seconds


def refine_around(
    evaluator: Evaluator,
    params: dict,
    sequences: list[str],
    depth: int,
    tag_prefix: str,
    budget: "Budget | None" = None,
    order: list[str] | None = None,
) -> tuple[dict, dict]:
    """Single-parameter refinement around the incumbent, one grid step at a time.

    ``order`` restricts the refinement to a subset of the parameters, for example the ones that improved the
    most in the previous round, so that a tight budget is spent where it pays off. Like the group sweep, the
    neighbours of all selected parameters are evaluated in one parallel batch and then applied in order.
    """
    best_params = dict(params)
    results = evaluator.run_batch([(best_params, f"{tag_prefix}/start")], sequences, truncate=False)
    best_row = results[0]["metrics"]
    tasks = []
    for key in order if order is not None else list(SEARCH_SPACE):
        if evaluator.exhausted() or (budget is not None and budget.expired()):
            break
        values = SEARCH_SPACE[key]
        if best_params[key] not in values:
            continue
        index = values.index(best_params[key])
        neighbours = [values[j] for j in range(index - depth, index + depth + 1) if 0 <= j < len(values)]
        for value in neighbours:
            if value == best_params[key]:
                continue
            trial = dict(best_params)
            trial[key] = value
            tasks.append((trial, key, value, f"{tag_prefix}/{key}={value}"))
    if tasks:
        results = evaluator.run_batch([(task[0], task[3]) for task in tasks], sequences)
        grouped: dict[str, list] = {}
        for task, result in zip(tasks, results):
            grouped.setdefault(task[1], []).append(result)
        for key in [task[1] for task in tasks]:
            for result in grouped.get(key, []):
                if rank_key(result["metrics"]) > rank_key(best_row):
                    best_params, best_row = result["params"], result["metrics"]
            print(
                f"[REFINE] {key:20s} -> {best_params[key]} | F1={best_row['f1'] * 100:.3f}% "
                f"FAR={best_row['far_per_frame']:.4f}",
                flush=True,
            )
    return best_params, best_row


# ==============================================================================
# 5. CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Global hyper-parameter search for the Longquanshan system SOTA")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--out-json", default="runs/logs/sys_sota_longquan_search.json")
    parser.add_argument("--out-preds-dir", default="runs/sys_sota_preds")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--time-budget-hours", type=float, default=0.42, help="Wall-clock budget for the whole run")
    parser.add_argument("--refine-depth", type=int, default=1, help="Grid steps around the Stage 1 optimum")
    parser.add_argument("--measure-only", action="store_true", help="Only measure the pipeline speed")
    parser.add_argument("--groups", default="", help="Comma-separated group names to run (default: all)")
    parser.add_argument("--max-evals", type=int, default=0, help="Stop the search after N evaluations (0 = no limit)")
    parser.add_argument("--jobs", type=int, default=24, help="Worker processes for the candidate sweep (fork)")
    parser.add_argument("--radii", default="", help="Comma-separated de-duplication radii to preload (default: all)")
    parser.add_argument(
        "--base-params",
        default="",
        help="JSON file with parameter overrides applied on top of the defaults (the coordinate descent starts there)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    # The budget covers the whole run; the reserve keeps room for the ablation, the final evaluation and the
    # prediction files, so the script always finishes inside the budget and still writes its results.
    budget = Budget(args.time_budget_hours * 3600, reserve=TAIL_RESERVE_SECONDS)
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    sequences = sorted(path.stem for path in cache_root.glob("*.pkl"))
    if not sequences:
        raise FileNotFoundError(f"No .pkl caches under {cache_root}")
    radii = SEARCH_SPACE["dedup_radius"]
    if args.radii:
        radii = [float(item) for item in args.radii.split(",") if item.strip()]
        # Only the preloaded radii can be evaluated, so the G5 sweep is restricted to them as well.
        # Without this the sweep would ask for a radius that was never loaded and raise KeyError.
        SEARCH_SPACE["dedup_radius"] = [r for r in SEARCH_SPACE["dedup_radius"] if float(r) in radii]

    store = CandidateStore(cache_root, labels_root, sequences, radii)
    global _STORE
    _STORE = store
    params = dict(DEFAULT_PARAMS)
    if args.base_params:
        overrides = json.loads(Path(args.base_params).read_text(encoding="utf-8"))
        # A delivery parameter file stores the actual values under "params" (with a "_provenance"
        # block beside them); a flat file is accepted as well.  Without this unwrapping the
        # overrides would be applied under the keys "_provenance"/"params" and the run would
        # *silently* start from DEFAULT_PARAMS instead of the given working point.
        if isinstance(overrides, dict) and isinstance(overrides.get("params"), dict):
            overrides = overrides["params"]
        params.update(overrides)
        print(f"[BASE] 基准参数覆盖 {len(overrides)} 项: {overrides}", flush=True)

    groups = GROUPS if not args.groups else groups_all()
    if args.groups:
        wanted = {name.strip() for name in args.groups.split(",") if name.strip()}
        # an explicit list keeps the order the user gave
        groups = [group for group in groups_all() if group[0] in wanted]
        groups.sort(key=lambda group: list(wanted).index(group[0]))
        if not groups:
            raise ValueError(f"no group matches {args.groups}; known: {[g[0] for g in groups_all()]}")

    total_frames = sum(store.frames(sequence) for sequence in sequences)
    print(f"[INFO] {len(sequences)} sequences, {total_frames} frames, radii {radii}", flush=True)

    # Load every cache and every radius up front so the fork workers inherit the data copy-on-write.
    print(f"[WARM] loading {len(sequences)} caches for {len(radii)} radii", flush=True)
    store.warm(sequences, float(params["dedup_radius"]), persist_key(params))
    print("[WARM] done", flush=True)

    pool = None
    if args.jobs > 1:
        context = mp.get_context("fork")
        pool = context.Pool(processes=args.jobs)
        print(f"[POOL] fork pool with {args.jobs} workers", flush=True)
    evaluator = Evaluator(store, pool=pool, max_evals=args.max_evals)

    # ---- runtime measurement -------------------------------------------------
    print("\n[MEASURE] pipeline speed on two large sequences", flush=True)
    measure_rows = []
    for sequence in PROBE_SEQUENCES:
        if sequence not in sequences:
            continue
        key = persist_key(params)
        records = store.records(sequence, float(params["dedup_radius"]), key)
        clock = time.perf_counter()
        run_system_sota(records, params, img_h=IMG_H)
        elapsed = time.perf_counter() - clock
        frames = store.frames(sequence)
        measure_rows.append({"sequence": sequence, "frames": frames, "seconds": elapsed})
        print(
            f"[MEASURE] {sequence} frames={frames} pipeline={elapsed:.1f}s "
            f"({elapsed / frames * 1000:.3f} ms/frame)",
            flush=True,
        )
    seconds_per_frame = sum(row["seconds"] for row in measure_rows) / sum(row["frames"] for row in measure_rows)
    estimated_full = seconds_per_frame * total_frames
    evals_per_round = sum(len(SEARCH_SPACE[key]) - 1 for _, keys in groups for key in keys)
    estimated_total = estimated_full * (evals_per_round * args.rounds + 2) / max(1, args.jobs)
    print(
        f"[MEASURE] {seconds_per_frame * 1000:.3f} ms/frame | one full 24-sequence evaluation "
        f"~{estimated_full:.0f}s | {evals_per_round} evals/round -> {args.rounds} rounds | jobs={args.jobs} "
        f"~{estimated_total / 3600:.2f}h",
        flush=True,
    )
    if args.measure_only:
        return

    # ---- stage plan ----------------------------------------------------------
    if estimated_total <= budget.seconds:
        plan = "single_stage"
        stage1_sequences = sequences
    else:
        plan = "two_stage"
        stage1_sequences = [sequence for sequence in SUBSET if sequence in sequences]
    subset_frames = sum(store.frames(sequence) for sequence in stage1_sequences)
    print(
        f"[PLAN] {plan} | Stage 1 on {len(stage1_sequences)} sequences ({subset_frames} frames) | "
        f"budget {args.time_budget_hours * 60:.0f}min",
        flush=True,
    )
    if plan == "two_stage":
        for sequence in stage1_sequences:
            print(f"[PLAN]   subset {sequence} frames={store.frames(sequence)}", flush=True)

    # ---- legacy control ------------------------------------------------------
    print("\n[CONTROL] Anti-UAV golden default (all Phase-A extensions OFF) on all sequences", flush=True)
    legacy_params = dict(DEFAULT_PARAMS)
    # The control must run on a radius that was actually preloaded, and it is far more informative when it
    # sits on the same radius as the search start: it then isolates "what the other parameters buy" instead of
    # mixing in a radius change.
    legacy_params["dedup_radius"] = float(params["dedup_radius"])
    legacy_results = evaluator.run_batch([(legacy_params, "control/legacy")], sequences, truncate=False)
    legacy_row = legacy_results[0]["metrics"]
    legacy_per_sequence = legacy_results[0]["per_sequence"]

    # ---- stage 1: coordinate descent ------------------------------------------
    print("\n[STAGE 1] coordinate descent", flush=True)
    best_params, best_row, rounds_completed, improvements, round_seconds = coordinate_descent(
        evaluator, params, stage1_sequences, groups, args.rounds, "stage1", budget
    )

    # ---- stage 2: a full second round only when the budget still allows it -----
    print("\n[STAGE 2] second round / targeted refinement", flush=True)
    round_cost = (sum(round_seconds) / len(round_seconds)) if round_seconds else float("inf")
    if rounds_completed < args.rounds and budget.search_remaining() >= round_cost * 0.9:
        print(
            f"[STAGE 2] 预算允许再跑一整轮 (剩余 {budget.search_remaining():.0f}s >= 一轮 {round_cost:.0f}s)",
            flush=True,
        )
        best_params, best_row, rounds_completed, improvements, round_seconds = coordinate_descent(
            evaluator, best_params, stage1_sequences, groups, 1, "stage1b", budget
        )
    else:
        # Only refine the parameters that improved the most, and only as many as the budget allows.
        ranked = sorted(improvements.items(), key=lambda item: -(item[1][3] - item[1][2]))
        order = [key for key, _ in ranked]
        print(f"[STAGE 2] 预算不足以再跑一整轮,改为细化改进最大的参数: {order}", flush=True)
        best_params, best_row = refine_around(
            evaluator, best_params, sequences, args.refine_depth, "stage2", budget, order
        )

    # The budget state of the search itself, captured before the tail consumes the reserve.
    search_budget_exhausted = bool(budget.expired())

    # ---- continuity ablation -------------------------------------------------
    ablation_configs = [
        ("no_consecutive_gate", {"confirm_consec": 1, "confirm_trim_prefix": 1}),
        ("consec3_keep_prefix", {"confirm_consec": 3, "confirm_trim_prefix": 0}),
        ("consec3_trim_prefix", {"confirm_consec": 3, "confirm_trim_prefix": 1}),
    ]
    ablation = {}
    tasks = []
    for name, override in ablation_configs:
        trial = dict(best_params)
        trial.update(override)
        tasks.append((trial, f"ablation/{name}"))
    for result in evaluator.run_batch(tasks, sequences, truncate=False):
        name = result["tag"].split("/")[-1]
        ablation[name] = {
            "override": next(override for label, override in ablation_configs if label == name),
            "metrics": result["metrics"],
            "per_sequence": result["per_sequence"],
        }

    # ---- final evaluation of the winner on all sequences ----------------------
    print("\n[FINAL] best global parameters on all sequences", flush=True)
    final_results = evaluator.run_local([(best_params, "final/best_global")], sequences, capture=True)
    final_row = final_results[0]["metrics"]
    final_per_sequence = final_results[0]["per_sequence"]
    final_detections = final_results[0]["detections"]
    reference = {
        sequence: {
            "baseline": res["baseline"],
            "online_fusion": res["online_fusion"],
            "bidirectional": res["bidirectional"],
        }
        for sequence, res in final_results[0]["reports"].items()
    }

    # ---- in-domain view and single-frame control ------------------------------
    in_domain = [sequence for sequence in sequences if sequence[-6:] in IN_DOMAIN_SUFFIXES]
    in_domain_frames = sum(store.frames(sequence) for sequence in in_domain)
    metrics_in_domain = aggregate_metrics(
        [final_per_sequence[sequence] for sequence in in_domain], in_domain_frames
    )
    print(
        f"[IN-DOMAIN] {len(in_domain)}/{len(sequences)} sequences, {in_domain_frames} frames | "
        f"TP={metrics_in_domain['tp']} FP={metrics_in_domain['fp']} FN={metrics_in_domain['fn']} "
        f"R={metrics_in_domain['recall'] * 100:.2f}% P={metrics_in_domain['precision'] * 100:.2f}% "
        f"F1={metrics_in_domain['f1'] * 100:.3f}% FAR={metrics_in_domain['far_per_frame']:.4f}",
        flush=True,
    )
    control_row, control_per_sequence = single_frame_control(store, sequences, best_params)
    print(
        f"[CONTROL single-frame @r={best_params['dedup_radius']:.0f},th={best_params['th_base']:.2f}] "
        f"TP={control_row['tp']} FP={control_row['fp']} FN={control_row['fn']} "
        f"R={control_row['recall'] * 100:.2f}% P={control_row['precision'] * 100:.2f}% "
        f"F1={control_row['f1'] * 100:.3f}% FAR={control_row['far_per_frame']:.4f}",
        flush=True,
    )

    # ---- predictions ---------------------------------------------------------
    out_preds_dir = Path(args.out_preds_dir)
    for sequence in sequences:
        config = {
            "sequence": sequence,
            "cache_root": str(cache_root),
            "labels_root": str(labels_root),
            "params": best_params,
            "dedup_radius": float(best_params["dedup_radius"]),
            "img_w": 640,
            "img_h": IMG_H,
            "match_mode": "inbox",
            "tol": 8.0,
            "source": "system_sota",
        }
        write_sequence_predictions(
            out_preds_dir,
            sequence,
            config,
            final_per_sequence[sequence],
            frame_payload(
                store.records(sequence, float(best_params["dedup_radius"]), persist_key(best_params)),
                final_detections[sequence],
                store.frames(sequence),
            ),
        )
    write_summary(out_preds_dir, best_params, final_row, final_per_sequence)
    print(f"[SUCCESS] predictions -> {out_preds_dir.resolve()}", flush=True)

    # ---- report --------------------------------------------------------------
    report = {
        "config": {
            "cache_root": str(cache_root),
            "labels_root": str(labels_root),
            "sequences": sequences,
            "num_sequences": len(sequences),
            "total_frames": total_frames,
            "rounds": args.rounds,
            "search_space": SEARCH_SPACE,
            "groups": {name: keys for name, keys in groups},
            "target": {"f1": TARGET_F1, "recall": TARGET_RECALL, "far_per_frame": TARGET_FAR},
            "seconds_per_frame": seconds_per_frame,
            "estimated_full_evaluation_seconds": estimated_full,
            "measure_rows": measure_rows,
        },
        "stages": {
            "plan": plan,
            "stage1_sequences": stage1_sequences,
            "stage1_frames": subset_frames,
            "stage2_sequences": sequences,
            "refine_depth": args.refine_depth,
            "evaluations": evaluator.evaluations,
            "jobs": args.jobs,
            "rounds_completed": rounds_completed,
            "rounds_requested": args.rounds,
            "budget_exhausted": bool(budget.expired()),
            "budget_seconds": budget.seconds,
            "budget_elapsed_seconds": budget.elapsed(),
            "tail_reserve_seconds": TAIL_RESERVE_SECONDS,
            "round_seconds": round_seconds,
            "improvements": {key: list(value) for key, value in improvements.items()},
        },
        "legacy_control": {"params": legacy_params, "metrics": legacy_row, "per_sequence": legacy_per_sequence},
        "single_frame_control": {
            "radius": float(best_params["dedup_radius"]),
            "th_base": float(best_params["th_base"]),
            "metrics": control_row,
            "per_sequence": control_per_sequence,
        },
        "metrics_in_domain_14": {
            "sequences": in_domain,
            "out_of_domain": [sequence for sequence in sequences if sequence not in in_domain],
            "frames": in_domain_frames,
            "metrics": metrics_in_domain,
            "per_sequence": {sequence: final_per_sequence[sequence] for sequence in in_domain},
        },
        "best_global": {
            "params": best_params,
            "metrics": final_row,
            "per_sequence": final_per_sequence,
            "meets_criteria": meets_criteria(final_row),
            "single_frame_control": {
                "radius": float(best_params["dedup_radius"]),
                "th_base": float(best_params["th_base"]),
                "metrics": control_row,
            },
            "metrics_in_domain_14": metrics_in_domain,
        },
        "ablation": ablation,
        "reference_metrics": reference,
        "sweep": evaluator.sweep,
        "elapsed_seconds": time.perf_counter() - started,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUCCESS] report -> {out_json.resolve()}", flush=True)

    print("\n[BEST GLOBAL]")
    for key in sorted(best_params):
        print(f"  {key:22s} = {best_params[key]}")
    print(
        f"  TP={final_row['tp']} FP={final_row['fp']} FN={final_row['fn']} GT={final_row['gt']} "
        f"R={final_row['recall'] * 100:.2f}% P={final_row['precision'] * 100:.2f}% "
        f"F1={final_row['f1'] * 100:.3f}% FAR={final_row['far_per_frame']:.4f} "
        f"meets={'YES' if meets_criteria(final_row) else 'NO'}"
    )
    if pool is not None:
        pool.close()
        pool.join()
    print(f"[SUCCESS] elapsed {time.perf_counter() - started:.1f}s")


def groups_all() -> list[tuple[str, list[str]]]:
    """The search groups in their fixed order; G6 is available but not searched by default."""
    return GROUPS + EXTRA_GROUPS


if __name__ == "__main__":
    main()
