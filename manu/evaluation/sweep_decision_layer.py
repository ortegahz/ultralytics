#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CPU-only decision-layer sweep over already-landed multi-scale heatmap caches.

What this script is
-------------------
The three-track gate freezes the decision layer in two files:
``manu/evaluation/system_sota_longquan_delivery_params.json`` (``th_base`` / ``dedup_radius``) and
``manu/evaluation/eval_multiscale_cache_th.py`` (the matching rule). Everything
downstream of the heatmap cache is pure post-processing, so a decision-layer knob can be swept on the **cached**
candidates without touching a GPU, re-running inference, or rebuilding a cache.

Two sweeps are implemented:

* **K1 -- (th_base x dedup_radius) grid.** The classical global knobs. Sweeping them is only meaningful on the
  single-frame arm (``--no-pipeline``), because the system pipeline re-derives its own thresholds from the candidate
  pool. The grid is *out-of-sample honest* for T3: the delivered threshold was frozen in the delivery JSON and never
  tuned on the blind sequence.
* **K2/K3 -- discriminative candidate gate.** A per-candidate predicate on the attributes the cache already stores
  but the current decision layer throws away: the connected-component ``area`` (pixel count), the integral energy
  ``sums`` (sum of the heatmap values inside the component) and the scale source ``tag`` (0 = native 640, 1 = 160,
  2 = 80). The gate is meant to separate "weak but real" candidates from noise, which a global threshold cannot do
  once the candidate pool has collapsed to one detection per frame.

Faithfulness
------------
With the gate disabled this script reproduces the frozen gate chain **bit for bit**:

* the candidate pooling order, the greedy cross-scale de-duplication and the score-descending greedy match are the
  same operations as ``manu/evaluation/eval_multiscale_cache_th.py`` (``merge_scales``) and
  ``manu/evaluation/system_sota_longquan.py`` (``build_candidates`` / ``single_frame_dets`` / ``evaluate_biz``);
* the ground truth, the matching rule (``point-in-bbox``, ``tol=8``) and the metric aggregation
  (``aggregate_metrics``: the per-sequence counts are summed and the metrics are recomputed once) are **imported**,
  not re-implemented;
* the ``fn_absent`` / ``fn_compete`` split is preserved, because it is the only way to tell a frontend limit
  ("the target never surfaced") from a decision-layer limit ("a better candidate stole the box").

Pass ``--check-json TRACK=PATH`` to compare the computed baseline against a frozen ``track*.json`` gate report.

Hard constraints honoured by this script
----------------------------------------
* **No GPU, no inference, no cache rebuild.** Only ``pickle`` reads of already-landed ``ms_cache`` files.
* **No change to the existing chain.** This file is additive; ``auto_epoch_gate_check.py`` /
  ``system_sota_longquan.py`` / ``eval_multiscale_cache_th.py`` keep their current behaviour and their defaults.
* **The frozen reference frame is never written.** This script only reads caches and labels and writes into
  ``--out-dir``.

Usage
-----
::

    python manu/evaluation/sweep_decision_layer.py \\
        --arm nogmc_t0049=/mnt/data/siping/runs/nogmc_t3/trials/trial_0049/gate/epoch20/ms_cache \\
        --track t3=VIDEO00007_19700101_001324 \\
        --track t2=VIDEO00009_19700101_001531,VIDEO00029_19700101_014053,VIDEO00033_19700101_015000,\\
VIDEO00034_19700101_015052,VIDEO00006_19700101_000703,VIDEO00008_19700101_001307,VIDEO00007_19700101_000818 \\
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \\
        --out-dir /mnt/data/siping/runs/nogmc_t3/decision_sweep/run1 \\
        --stage all --until "2026-11-05 07:00"
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    count_matches,
    load_ground_truth,
    metrics,
)
from manu.evaluation.system_sota_longquan import SCALE_TAGS, aggregate_metrics  # noqa: E402

# The T2 hard gate as applied by the current gate chain (track2_golden: F1 >= 0.985 and FP <= 112).
DEFAULT_GATE_F1_MIN = 0.985
DEFAULT_GATE_FP_MAX = 112

# The literal "1.21 TP/FP" bar quoted by the study: it is (FP_base + GT_base) / TP_base, i.e. how many false
# positives one true positive may buy before F1 stops improving.
STRICT_BAR = 1.2065


def breakeven_tp_per_fp(tp: int, fp: int, gt: int) -> float:
    """Return the exact break-even ratio ``dTP/dFP`` above which F1 increases.

    ``F1 = 2*TP/(TP+FP+GT)`` with a fixed ground truth, so ``dF1 > 0`` exactly when
    ``dTP/dFP > TP/(FP+GT)``.
    """
    denominator = fp + gt
    return tp / denominator if denominator else float("inf")


# --------------------------------------------------------------------------------------------------
# Configuration objects
# --------------------------------------------------------------------------------------------------


@dataclass
class Arm:
    """One cached arm: a name plus the directory holding ``<sequence>.pkl`` heatmap caches."""

    name: str
    cache_root: Path


@dataclass
class Track:
    """One evaluation track: a name plus the sequences that are pooled into a single metric."""

    name: str
    sequences: list[str]


@dataclass
class Gate:
    """The discriminative candidate gate.

    ``area_min``   absolute connected-component area floor in native pixels.
    ``sum_mode``   ``none`` (no energy floor), ``abs`` (absolute ``sums`` floor) or ``tagq`` (per-tag quantile).
    ``sum_value``  the floor itself: an absolute ``sums`` value in ``abs`` mode, a quantile level in ``tagq`` mode.
    ``tag_allow``  the scale sources that survive the gate (0 = native 640, 1 = 160, 2 = 80).
    ``thresholds`` the resolved per-tag ``sums`` thresholds, filled in by :func:`build_sum_quantiles`.
    """

    area_min: int = 0
    sum_mode: str = "none"
    sum_value: float = 0.0
    tag_allow: tuple[int, ...] = SCALE_TAGS
    thresholds: dict[int, float] = field(default_factory=dict)

    def key(self) -> tuple:
        """A hashable identity of this gate configuration."""
        return (self.area_min, self.sum_mode, self.sum_value, tuple(self.tag_allow))

    def describe(self) -> str:
        """A short human-readable label used in the console log."""
        if self.sum_mode == "none":
            energy = "-"
        elif self.sum_mode == "abs":
            energy = f"sums>={self.sum_value:g}"
        else:
            energy = f"sums>=q{self.sum_value:g}"
        return f"area>={self.area_min},{energy},tags={list(self.tag_allow)}"

    def mask(self, pool: "SequencePool") -> np.ndarray:
        """Return the boolean mask of the candidates that survive this gate."""
        mask = pool.area >= self.area_min
        if self.sum_mode == "abs":
            mask &= pool.sums >= self.sum_value
        elif self.sum_mode == "tagq":
            floors = np.array(
                [self.thresholds.get(int(tag), -np.inf) for tag in pool.tag],
                dtype=np.float32,
            )
            mask &= pool.sums >= floors
        mask &= np.isin(pool.tag, np.asarray(self.tag_allow, dtype=np.int16))
        return mask


# --------------------------------------------------------------------------------------------------
# Cache reading
# --------------------------------------------------------------------------------------------------


class SequencePool:
    """The cached candidates of one sequence, flattened into per-candidate attribute arrays.

    The cross-scale pooling order is pre-computed once, exactly the way
    ``manu/evaluation/eval_multiscale_cache_th.py::load_predictions`` orders the rows before
    ``merge_scales`` pools them: tag-major (0, 1, 2), each tag stable-sorted by descending score, then a final
    stable sort by descending score over the concatenation. A Python ``list.sort`` is stable, so the result is
    identical to the upstream pooling.
    """

    def __init__(self, cache_root: Path, sequence: str) -> None:
        cache = pickle.loads((cache_root / f"{sequence}.pkl").read_bytes())
        self.sequence = sequence
        self.meta = cache["meta"]
        self.frames = int(self.meta["frames"])
        self.width, self.height = (int(value) for value in self.meta["native_size"])

        self.score = cache["scores"].astype(np.float32)
        self.point = cache["points"].astype(np.float32)
        self.area = cache["area"].astype(np.int32)
        self.sums = cache["sums"].astype(np.float32)
        self.tag = cache["tag"].astype(np.int16)

        self.frame_ids: list[int] = []
        self.pooled: list[np.ndarray] = []
        offset = cache["offset"]
        frame_index = cache["frame_index"]
        scale_tags = np.asarray(SCALE_TAGS, dtype=np.int16)
        for position in range(len(frame_index)):
            start, end = int(offset[position]), int(offset[position + 1])
            self.frame_ids.append(int(frame_index[position]))
            if end <= start:
                self.pooled.append(np.empty(0, dtype=np.int64))
                continue
            rows = np.arange(start, end, dtype=np.int64)
            rows = rows[np.isin(self.tag[rows], scale_tags)]
            if not len(rows):
                self.pooled.append(np.empty(0, dtype=np.int64))
                continue
            parts = []
            for tag in SCALE_TAGS:
                subset = rows[self.tag[rows] == tag]
                if len(subset):
                    parts.append(subset[np.argsort(-self.score[subset], kind="stable")])
            pooled = np.concatenate(parts)
            self.pooled.append(pooled[np.argsort(-self.score[pooled], kind="stable")])

        # Python lists for the hot merge loop (numpy scalar indexing is ~10x slower here).
        self.xs = self.point[:, 0].tolist()
        self.ys = self.point[:, 1].tolist()
        self.scores_list = self.score.tolist()

    def merged(self, position: int, radius: float, keep: np.ndarray | None = None) -> list[int]:
        """Greedy cross-scale de-duplication of one frame.

        The candidates are visited in descending score; a candidate is dropped when it lies within ``radius`` of
        an already kept candidate. This is the same rule as ``merge_scales``.
        """
        kept: list[int] = []
        kept_xy: list[tuple[float, float]] = []
        radius_squared = radius * radius
        xs, ys = self.xs, self.ys
        for row in self.pooled[position]:
            if keep is not None and not keep[row]:
                continue
            x, y = xs[row], ys[row]
            blocked = False
            for px, py in kept_xy:
                if (x - px) ** 2 + (y - py) ** 2 <= radius_squared:
                    blocked = True
                    break
            if blocked:
                continue
            kept.append(row)
            kept_xy.append((x, y))
        return kept


def build_sum_quantiles(
    cache_root: Path,
    levels: list[float],
    exclude_sequences: set[str] | None = None,
) -> dict[int, dict[float, float]]:
    """Per-tag ``sums`` quantile table of one arm.

    The ``sums`` of the three scale rows live on different scales (the native row integrates the fine grid, the
    160 row and the 80 row integrate their own coarse grids), so a single absolute energy floor is not
    comparable across tags. The floor is therefore expressed as a **per-tag quantile level** and resolved
    against the whole cache root of the arm, i.e. the same table is applied to every track of that arm (the gate
    is applied as a group to the tested arm and to the reference arm).

    The population is every cached candidate of that tag over every sequence of the cache root; it is reported
    in the output JSON so that the threshold can be reproduced. ``exclude_sequences`` removes sequences from
    that population, which is how the blind hold-out is kept out of its own quantile (out-of-sample measurement).
    """
    excluded = exclude_sequences or set()
    chunks: dict[int, list[np.ndarray]] = {tag: [] for tag in SCALE_TAGS}
    for path in sorted(cache_root.glob("*.pkl")):
        if path.stem in excluded:
            continue
        cache = pickle.loads(path.read_bytes())
        tag = cache["tag"]
        sums = cache["sums"].astype(np.float32)
        for scale in SCALE_TAGS:
            selected = tag == scale
            if selected.any():
                chunks[scale].append(sums[selected])
    table: dict[int, dict[float, float]] = {}
    for scale, pieces in chunks.items():
        if not pieces:
            continue
        values = np.concatenate(pieces)
        table[scale] = {level: float(np.percentile(values, level * 100.0)) for level in levels}
    return table


# --------------------------------------------------------------------------------------------------
# Evaluation (mirrors system_sota_longquan.build_records / single_frame_dets / evaluate_biz)
# --------------------------------------------------------------------------------------------------


def merged_view(pool: SequencePool, radius: float, keep: np.ndarray | None = None) -> dict[int, list[int]]:
    """The merged candidate pool of one sequence, keyed by frame index."""
    view: dict[int, list[int]] = {}
    for position, frame_id in enumerate(pool.frame_ids):
        kept = pool.merged(position, radius, keep)
        if kept:
            view[frame_id] = kept
    return view


def score_view(
    pool: SequencePool,
    view: dict[int, list[int]],
    truth: dict[int, np.ndarray],
    threshold: float,
    keep: np.ndarray | None = None,
) -> dict:
    """Score one sequence with the business-domain rule (point-in-bbox, greedy score-descending).

    ``threshold`` plays the role of ``th_base``; the matching itself always runs at 0.0 because the threshold was
    already applied when the detections were selected, exactly like ``evaluate_biz`` does.
    """
    frames: list[tuple] = []
    for frame_id in sorted(set(truth) | set(view)):
        rows = view.get(frame_id, [])
        if keep is not None:
            rows = [row for row in rows if keep[row]]
        boxes = truth.get(frame_id)
        boxes = None if boxes is None or not len(boxes) else np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if not rows:
            frames.append((boxes, None, None))
            continue
        scores = np.asarray([pool.scores_list[row] for row in rows], dtype=np.float32)
        points = np.asarray([[pool.xs[row], pool.ys[row]] for row in rows], dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        scores, points = scores[order], points[order]
        selected = np.nonzero(scores >= threshold)[0]
        frames.append(
            (
                boxes,
                points[selected] if len(selected) else None,
                scores[selected] if len(selected) else None,
            )
        )
    return count_matches(frames, 0.0, 8.0, "inbox")


# --------------------------------------------------------------------------------------------------
# Sweep machinery
# --------------------------------------------------------------------------------------------------


def parse_floats(text: str) -> list[float]:
    """Parse a comma-separated float list."""
    return [float(item) for item in text.replace(" ", "").split(",") if item]


def parse_ints(text: str) -> list[int]:
    """Parse a comma-separated int list."""
    return [int(item) for item in text.replace(" ", "").split(",") if item]


@dataclass
class ArmContext:
    """Everything one arm needs: the cached pools, the ground truth and the per-tag energy quantiles."""

    arm: Arm
    tracks: dict[str, Track]
    pools: dict[str, dict[str, SequencePool]]
    truths: dict[str, dict[str, dict[int, np.ndarray]]]
    sum_quantiles: dict[int, dict[float, float]]
    quantile_scope: str = "arm"
    quantile_excluded: list[str] = field(default_factory=list)
    baselines: dict[str, dict] = field(default_factory=dict)
    baseline_gate: dict[str, bool] = field(default_factory=dict)
    view_cache: dict = field(default_factory=dict)

    def views(self, track_name: str, radius: float, gate: Gate, gate_stage: str) -> dict[str, dict[int, list[int]]]:
        """The merged candidate pool of every sequence of one track.

        The ``post`` gate stage filters the *merged* candidates, so its merged view does not depend on the gate
        and is cached. The ``pre`` gate stage filters the candidates *before* the merge, so its view depends on
        the gate and is rebuilt (and discarded) for every gate configuration.
        """
        if gate_stage == "post":
            key = ("post", track_name, radius)
            cached = self.view_cache.get(key)
            if cached is None:
                cached = {
                    sequence: merged_view(pool, radius) for sequence, pool in self.pools[track_name].items()
                }
                self.view_cache[key] = cached
            return cached
        return {
            sequence: merged_view(pool, radius, gate.mask(pool))
            for sequence, pool in self.pools[track_name].items()
        }


def load_arm_context(
    arm: Arm,
    tracks: list[Track],
    labels_root: Path,
    truth_cache: dict[str, dict[int, np.ndarray]],
    quantile_scope: str = "arm",
    primary_track: str = "",
    quantile_levels: list[float] | None = None,
) -> ArmContext:
    """Load the caches, the ground truth and the energy quantiles of one arm.

    ``quantile_scope`` selects the population of the per-tag energy quantiles:
    ``arm`` uses every cached sequence of the arm, ``exclude-primary`` drops the sequences of the primary
    track so that the blind hold-out is not part of its own energy threshold.

    ``quantile_levels`` are the quantile levels to materialise.  They default to the historical hard-coded
    set; the caller must pass the levels it actually sweeps, otherwise a requested level silently maps to a
    threshold of 0.0 (i.e. "no energy floor").
    """
    pools: dict[str, dict[str, SequencePool]] = {}
    truths: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for track in tracks:
        track_pools: dict[str, SequencePool] = {}
        track_truths: dict[str, dict[int, np.ndarray]] = {}
        for sequence in track.sequences:
            path = arm.cache_root / f"{sequence}.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"missing cache: {path}")
            pool = SequencePool(arm.cache_root, sequence)
            track_pools[sequence] = pool
            key = f"{labels_root}::{sequence}"
            if key not in truth_cache:
                truth_cache[key] = load_ground_truth(labels_root, sequence, pool.width, pool.height)
            track_truths[sequence] = truth_cache[key]
        pools[track.name] = track_pools
        truths[track.name] = track_truths
    excluded: set[str] = set()
    if quantile_scope == "exclude-primary":
        primary = next((track for track in tracks if track.name == primary_track), None)
        if primary is not None:
            excluded = set(primary.sequences)
    # NOTE: the quantile levels must be the ones the caller actually asked for.  A hard-coded
    # list silently degraded every requested level outside it (e.g. q0.4 / q0.5) to a
    # threshold of 0.0, i.e. "no energy floor at all" -- a quiet grid that does not do
    # what it says on the tin.
    levels = quantile_levels if quantile_levels else [0.0, 0.05, 0.1, 0.2, 0.3]
    quantiles = build_sum_quantiles(arm.cache_root, levels, excluded)
    return ArmContext(
        arm=arm,
        tracks={track.name: track for track in tracks},
        pools=pools,
        truths=truths,
        sum_quantiles=quantiles,
        quantile_scope=quantile_scope,
        quantile_excluded=sorted(excluded),
    )


def track_metrics(
    context: ArmContext,
    track_name: str,
    views: dict[str, dict[int, list[int]]],
    threshold: float,
    keeps: dict[str, np.ndarray] | None = None,
) -> dict:
    """Score one track and aggregate it exactly like ``system_sota_longquan.main`` does."""
    per_sequence: list[dict] = []
    num_frames = 0
    for sequence, pool in context.pools[track_name].items():
        keep = keeps[sequence] if keeps is not None else None
        counts = score_view(pool, views[sequence], context.truths[track_name][sequence], threshold, keep)
        per_sequence.append(metrics(counts, pool.frames))
        num_frames += pool.frames
    return aggregate_metrics(per_sequence, num_frames)


def compute_baseline(
    context: ArmContext,
    radius: float,
    threshold: float,
    gated_tracks: set[str],
    gate_f1_min: float,
    gate_fp_max: int,
) -> None:
    """Compute the no-gate baseline of every track (the reference point of all deltas)."""
    for name in context.tracks:
        views = context.views(name, radius, Gate(), "post")
        row = track_metrics(context, name, views, threshold)
        context.baselines[name] = row
        context.baseline_gate[name] = (
            bool(row["f1"] >= gate_f1_min and row["fp"] <= gate_fp_max) if name in gated_tracks else None
        )


def sweep_rows(
    context: ArmContext,
    stage: str,
    radius: float,
    gate: Gate,
    thresholds: list[float],
    primary: str,
    gated_tracks: set[str],
    gate_f1_min: float,
    gate_fp_max: int,
    gate_stage: str,
) -> list[dict]:
    """Evaluate one (arm, stage, radius, gate) configuration over every threshold and every track."""
    rows: list[dict] = []
    masks = {
        name: {sequence: gate.mask(pool) for sequence, pool in context.pools[name].items()}
        for name in context.tracks
    }
    views = {
        name: context.views(name, radius, gate, gate_stage) for name in context.tracks
    }
    for threshold in thresholds:
        row: dict = {
            "stage": stage,
            "arm": context.arm.name,
            "radius": radius,
            "area_min": gate.area_min,
            "sum_mode": gate.sum_mode,
            "sum_value": gate.sum_value,
            "tag_allow": "|".join(str(tag) for tag in gate.tag_allow),
            "gate_stage": gate_stage,
            "th_base": threshold,
        }
        for name in context.tracks:
            keeps = masks[name] if gate_stage == "post" else None
            metric = track_metrics(context, name, views[name], threshold, keeps)
            for key in ("f1", "tp", "fp", "fn", "recall", "precision", "fn_absent", "fn_compete", "det", "gt"):
                row[f"{name}_{key}"] = metric[key]
            if name in gated_tracks:
                row[f"{name}_gate_pass"] = bool(metric["f1"] >= gate_f1_min and metric["fp"] <= gate_fp_max)
        base = context.baselines.get(primary)
        if base is not None:
            delta_tp = row[f"{primary}_tp"] - base["tp"]
            delta_fp = row[f"{primary}_fp"] - base["fp"]
            row[f"d_{primary}_f1"] = row[f"{primary}_f1"] - base["f1"]
            row[f"d_{primary}_tp"] = delta_tp
            row[f"d_{primary}_fp"] = delta_fp
            row[f"d_{primary}_fn"] = row[f"{primary}_fn"] - base["fn"]
            row["bar_tp_per_fp"] = (delta_tp / delta_fp) if delta_fp else float("nan")
            row["bar_breakeven"] = breakeven_tp_per_fp(base["tp"], base["fp"], base["gt"])
            row["bar_fp_per_tp"] = (base["fp"] + base["gt"]) / base["tp"] if base["tp"] else float("inf")
            # Exact F1 margin: (FP+GT)*dTP - TP*dFP > 0  <=>  dF1 > 0.
            row["bar_margin"] = (base["fp"] + base["gt"]) * delta_tp - base["tp"] * delta_fp
            row["bar_exact_pass"] = bool(row[f"d_{primary}_f1"] > 0)
            # The study's literal "1.21" bar, direction aware: one true positive may move at most
            # (FP+GT)/TP false positives. A knob that ADDS candidates must add fewer than that per TP;
            # a knob that REMOVES candidates must remove more than that per TP.
            cost = abs(delta_fp) / abs(delta_tp) if delta_tp else float("nan")
            row["bar_fp_per_tp_cost"] = cost
            if delta_tp > 0 and delta_fp > 0:
                row["bar_strict_1p21_pass"] = bool(np.isfinite(cost) and cost < row["bar_fp_per_tp"])
            elif delta_tp < 0 and delta_fp < 0:
                row["bar_strict_1p21_pass"] = bool(np.isfinite(cost) and cost > row["bar_fp_per_tp"])
            else:
                row["bar_strict_1p21_pass"] = bool(row["bar_margin"] > 0)
            row["bar_strict_1p2065_literal"] = bool(
                np.isfinite(cost)
                and ((cost < STRICT_BAR) if (delta_tp > 0 and delta_fp > 0) else
                     (cost > STRICT_BAR) if (delta_tp < 0 and delta_fp < 0) else row["bar_margin"] > 0)
            )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------
# Diagnostics (quantify the intermediate quantities behind a negative conclusion)
# --------------------------------------------------------------------------------------------------


def _in_box(point: tuple[float, float], box: np.ndarray) -> bool:
    """Point-in-box test against one ground-truth box."""
    return abs(point[0] - box[0]) <= box[2] / 2 and abs(point[1] - box[1]) <= box[3] / 2


def diagnose_sequence(
    pool: SequencePool,
    view: dict[int, list[int]],
    truth: dict[int, np.ndarray],
    threshold: float,
) -> tuple[list[tuple[bool, int]], list[dict]]:
    """Label every baseline detection TP/FP and classify every missed ground truth.

    The matching mirrors ``count_matches`` (greedy score-descending, point-in-bbox), but the labels are kept so
    that the per-candidate attributes can be split into the TP and the FP population, and so that every missed
    ground truth can be classified. The caller asserts the counts against ``count_matches``.
    """
    detections: list[tuple[bool, int]] = []
    missed: list[dict] = []
    for position, frame_id in enumerate(pool.frame_ids):
        boxes = truth.get(frame_id)
        boxes = None if boxes is None or not len(boxes) else np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        rows = view.get(frame_id, [])
        if rows:
            scores = np.asarray([pool.scores_list[row] for row in rows], dtype=np.float32)
            order = np.argsort(-scores, kind="stable")
            rows = [rows[index] for index in order]
            scores = scores[order]
            selected = [index for index, value in enumerate(scores) if value >= threshold]
        else:
            selected = []
        taken = np.zeros(0 if boxes is None else len(boxes), dtype=bool)
        for index in selected:
            row = rows[index]
            point = (pool.xs[row], pool.ys[row])
            hit = False
            if boxes is not None and len(boxes):
                free = np.nonzero(~taken)[0]
                if len(free):
                    distances = ((boxes[free, :2] - np.asarray(point, dtype=np.float32)) ** 2).sum(axis=1)
                    nearest = int(np.argmin(distances))
                    box = free[nearest]
                    if _in_box(point, boxes[box]):
                        taken[box] = True
                        hit = True
            detections.append((hit, row))
        if boxes is None:
            continue
        for index in range(len(boxes)):
            if taken[index]:
                continue
            # Upstream split: a box that a threshold-passing detection did hit is fn_compete, otherwise fn_absent.
            competed = any(
                _in_box((pool.xs[rows[i]], pool.ys[rows[i]]), boxes[index]) for i in selected
            )
            # Recoverability: does ANY cached candidate (any scale, any peak) fall inside the box?
            in_box_rows = []
            for raw in pool.pooled[position]:
                if _in_box((pool.xs[raw], pool.ys[raw]), boxes[index]):
                    in_box_rows.append(raw)
            best = max(in_box_rows, key=lambda raw: pool.scores_list[raw]) if in_box_rows else None
            if not in_box_rows:
                kind = "no_candidate_ever_in_box"
            elif best is not None and pool.scores_list[best] >= threshold:
                # The box holds a candidate strong enough to pass the threshold, but the cross-scale de-duplication
                # kept a higher-scoring candidate elsewhere in the frame => reachable with a smaller dedup radius.
                kind = "in_box_candidate_above_th_but_suppressed_by_dedup"
            else:
                kind = "in_box_candidate_below_th_recoverable"
            missed.append(
                {
                    "frame_id": frame_id,
                    "kind": kind,
                    "competed": bool(competed),
                    "best_in_box_score": None if best is None else float(pool.scores_list[best]),
                    "best_in_box_area": None if best is None else int(pool.area[best]),
                    "best_in_box_tag": None if best is None else int(pool.tag[best]),
                }
            )
    return detections, missed


def run_diagnostics(
    context: ArmContext,
    track_name: str,
    radius: float,
    threshold: float,
) -> dict:
    """Quantify the FP / TP attribute separation and the recoverability of every missed ground truth."""
    views = context.views(track_name, radius, Gate(), "post")
    per_tag: dict[int, dict[str, list[float]]] = {
        tag: {"tp_area": [], "fp_area": [], "tp_sums": [], "fp_sums": [], "tp_score": [], "fp_score": []}
        for tag in SCALE_TAGS
    }
    fn_kinds: dict[str, int] = {}
    counts = {"tp": 0, "fp": 0, "fn_absent": 0, "fn_compete": 0}
    for sequence, pool in context.pools[track_name].items():
        truth = context.truths[track_name][sequence]
        detections, missed = diagnose_sequence(pool, views[sequence], truth, threshold)
        authoritative = score_view(pool, views[sequence], truth, threshold)
        if authoritative["tp"] != sum(1 for hit, _ in detections if hit):
            raise AssertionError(f"diagnostic TP mismatch on {sequence}")
        if authoritative["fp"] != sum(1 for hit, _ in detections if not hit):
            raise AssertionError(f"diagnostic FP mismatch on {sequence}")
        counts["tp"] += authoritative["tp"]
        counts["fp"] += authoritative["fp"]
        counts["fn_absent"] += authoritative["fn_absent"]
        counts["fn_compete"] += authoritative["fn_compete"]
        for hit, row in detections:
            bucket = "tp" if hit else "fp"
            per_tag[int(pool.tag[row])][f"{bucket}_area"].append(float(pool.area[row]))
            per_tag[int(pool.tag[row])][f"{bucket}_sums"].append(float(pool.sums[row]))
            per_tag[int(pool.tag[row])][f"{bucket}_score"].append(float(pool.score[row]))
        for record in missed:
            fn_kinds[record["kind"]] = fn_kinds.get(record["kind"], 0) + 1
    quantiles = (10, 25, 50, 75, 90)

    def summary(values: list[float]) -> dict | None:
        """Percentile summary of one attribute population."""
        if not values:
            return None
        return {f"p{percentile}": round(float(np.percentile(values, percentile)), 4) for percentile in quantiles}

    attributes = {}
    for tag, buckets in per_tag.items():
        attributes[str(tag)] = {
            "n_tp": len(buckets["tp_area"]),
            "n_fp": len(buckets["fp_area"]),
            "tp_area": summary(buckets["tp_area"]),
            "fp_area": summary(buckets["fp_area"]),
            "tp_sums": summary(buckets["tp_sums"]),
            "fp_sums": summary(buckets["fp_sums"]),
            "tp_score": summary(buckets["tp_score"]),
            "fp_score": summary(buckets["fp_score"]),
        }
    recoverable = fn_kinds.get("in_box_candidate_below_th_recoverable", 0)
    suppressed = fn_kinds.get("in_box_candidate_above_th_but_suppressed_by_dedup", 0)
    unrecoverable = fn_kinds.get("no_candidate_ever_in_box", 0)
    return {
        "arm": context.arm.name,
        "track": track_name,
        "radius": radius,
        "th_base": threshold,
        "counts": counts,
        "fn_kinds": fn_kinds,
        "recoverable_fn_frames": recoverable,
        "suppressed_fn_frames": suppressed,
        "unrecoverable_fn_frames": unrecoverable,
        "decision_layer_reachable_fn_frames": recoverable + suppressed,
        "recoverable_fraction_of_fn": recoverable / max(counts["fn_absent"], 1),
        "attributes_by_tag": attributes,
    }


# --------------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the sweep rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value, digits: int = 6) -> str:
    """Format a float for the markdown tables."""
    if value is None:
        return "-"
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def build_summary(
    out_dir: Path,
    meta: dict,
    contexts: list[ArmContext],
    rows: list[dict],
    primary: str,
    gated_tracks: list[str],
    gate_f1_min: float,
    gate_fp_max: int,
    diagnostics: list[dict],
) -> str:
    """Render the human-readable summary of the sweep."""
    lines: list[str] = []
    add = lines.append
    add("# 决策层旋钮扫参报告（零 GPU / 纯 CPU）")
    add("")
    add(f"- 生成时刻：`{meta['generated_at']}`")
    add(f"- 执行机：`{meta['machine']}`")
    add("- 脚本：`manu/evaluation/sweep_decision_layer.py`")
    add(f"- 命令：`{meta['command']}`")
    add(f"- 墙钟截止：`{meta['until']}`（到点停止并把已有结果落盘）")
    add(f"- 总耗时：`{meta['elapsed_s']:.1f}s`")
    add("- 说明：**未运行任何推理/缓存/门禁脚本**，只读取已落盘的 `ms_cache`；"
        "参照系 `reference_frame.json` 未被改动。")
    add("")

    add("## 0. 基线复现（no-gate，radius=768，th_base=0.30）")
    add("")
    add("| 臂 | 轨 | TP | FP | FN | F1 | 硬门禁 |")
    add("|:--|:--|--:|--:|--:|--:|:--|")
    for context in contexts:
        for name in context.tracks:
            base = context.baselines.get(name)
            if base is None:
                continue
            gate = context.baseline_gate.get(name)
            gate_text = "-" if gate is None else ("PASS" if gate else "FAIL")
            add(
                f"| {context.arm.name} | {name} | {base['tp']} | {base['fp']} | {base['fn']} | "
                f"{format_float(base['f1'])} | {gate_text} |"
            )
    add("")
    if meta.get("self_check"):
        add("自检（与冻结门禁 JSON 逐项对比）：")
        add("")
        add("| 臂 | 轨 | 冻结 TP/FP/FN/F1 | 实测 TP/FP/FN/F1 | 结论 |")
        add("|:--|:--|:--|:--|:--|")
        for check in meta["self_check"]:
            add(
                f"| {check['arm']} | {check['track']} | {check['expected_tp']}/{check['expected_fp']}/"
                f"{check['expected_fn']}/{format_float(check['expected_f1'])} | {check['actual_tp']}/"
                f"{check['actual_fp']}/{check['actual_fn']}/{format_float(check['actual_f1'])} | {check['verdict']} |"
            )
        add("")

    add("## 1. K1｜(th_base × dedup_radius) 网格")
    add("")
    for context in contexts:
        arm_rows = [row for row in rows if row["stage"] == "k1" and row["arm"] == context.arm.name]
        if not arm_rows:
            continue
        add(f"### 臂 `{context.arm.name}`")
        add("")
        radii = sorted({row["radius"] for row in arm_rows}, reverse=True)
        thresholds = sorted({row["th_base"] for row in arm_rows})

        def cell(radius: float, threshold: float) -> dict | None:
            """Find one grid cell."""
            return next(
                (row for row in arm_rows if row["radius"] == radius and row["th_base"] == threshold), None
            )

        add(f"{primary} F1：")
        add("")
        add("| dedup_radius | " + " | ".join(f"th={th:g}" for th in thresholds) + " |")
        add("|:--|" + "--:|" * len(thresholds))
        for radius in radii:
            cells = []
            for threshold in thresholds:
                match = cell(radius, threshold)
                cells.append(format_float(match[f"{primary}_f1"], 4) if match else "-")
            add(f"| {radius:g} | " + " | ".join(cells) + " |")
        add("")
        add(f"{primary} TP/FP/FN：")
        add("")
        add("| dedup_radius | " + " | ".join(f"th={th:g}" for th in thresholds) + " |")
        add("|:--|" + "--:|" * len(thresholds))
        for radius in radii:
            cells = []
            for threshold in thresholds:
                match = cell(radius, threshold)
                cells.append(
                    f"{match[f'{primary}_tp']}/{match[f'{primary}_fp']}/{match[f'{primary}_fn']}" if match else "-"
                )
            add(f"| {radius:g} | " + " | ".join(cells) + " |")
        add("")
        for name in gated_tracks:
            add(f"{name} F1 / FP（硬门禁 F1 ≥ {gate_f1_min} 且 FP ≤ {gate_fp_max}）：")
            add("")
            add("| dedup_radius | " + " | ".join(f"th={th:g}" for th in thresholds) + " |")
            add("|:--|" + "--:|" * len(thresholds))
            for radius in radii:
                cells = []
                for threshold in thresholds:
                    match = cell(radius, threshold)
                    if not match:
                        cells.append("-")
                        continue
                    flag = "PASS" if match.get(f"{name}_gate_pass") else "FAIL"
                    cells.append(f"{match[f'{name}_f1']:.6f}/{match[f'{name}_fp']} {flag}")
                add(f"| {radius:g} | " + " | ".join(cells) + " |")
            add("")

    add("## 2. K2/K3｜判别式候选门（area_min × sum_min × th_base）")
    add("")
    add("判据（等价于 `sweep_decision_layer.py::Gate.mask`）：")
    add("")
    add("```text")
    add("候选存活  <=>  area >= area_min")
    add("          AND   sums >= sum_threshold(tag)")
    add("          AND   tag in tag_allow")
    add("          AND   score >= th_base")
    add("")
    add("sum_threshold(tag):")
    add("  sum_mode = 'abs'  ->  绝对积分能量下限（对所有 tag 同值，跨尺度不可比，仅作对照组）")
    add("  sum_mode = 'tagq' ->  该 tag 在本臂 cache_root 全体候选上的分位数（跨尺度可比）")
    add("```")
    add("")
    add("F1 盈亏判据（由 F1 = 2TP/(TP+FP+GT)、GT 固定推出的精确式）：")
    add("")
    add("```text")
    add("dF1 > 0  <=>  (FP_base+GT)*dTP - TP_base*dFP > 0            （= bar_margin 列）")
    add("           <=>  dTP/dFP > TP_base/(FP_base+GT_base)          （= bar_breakeven 列）")
    add("")
    add("本研究引用的「1.21」门槛 = (FP_base+GT_base)/TP_base = 每 1 个 TP 最多允许动 1.2065 个 FP：")
    add("  * 加候选的旋钮（降 th）：每多赚 1 个 TP，代价的 FP 必须 < 1.2065 个；")
    add("  * 减候选的旋钮（候选门）：每多移除 1 个 TP，必须同时净移除 > 1.2065 个 FP。")
    add("  两种方向在 CSV 里都落在 bar_fp_per_tp_cost / bar_strict_1p21_pass 两列上（方向敏感）。")
    add("```")
    add("")
    for context in contexts:
        arm_rows = [row for row in rows if row["stage"].startswith("k2k3") and row["arm"] == context.arm.name]
        if not arm_rows:
            continue
        add(f"### 臂 `{context.arm.name}`")
        add("")
        base = context.baselines.get(primary, {})
        breakeven = breakeven_tp_per_fp(base.get("tp", 0), base.get("fp", 0), base.get("gt", 0))
        fp_per_tp = (base.get("fp", 0) + base.get("gt", 0)) / max(base.get("tp", 1), 1)
        add(
            f"基线：TP={base.get('tp')} FP={base.get('fp')} F1={format_float(base.get('f1', 0.0))}；"
            f"盈亏平衡 dTP/dFP > {format_float(breakeven, 4)}"
            f"（等价：每 TP 可付 {format_float(fp_per_tp, 4)} 个 FP）"
        )
        add("")
        ranked = sorted(arm_rows, key=lambda row: -row.get(f"d_{primary}_f1", -1.0))
        add(f"ΔF1 前 20 行（全部 {len(arm_rows)} 行见 CSV）：")
        add("")
        add(
            f"| # | gate_stage | area_min | sum_mode | sum_value | th | {primary} F1 | TP | FP | FN | "
            f"ΔF1 | dTP | dFP | dTP/dFP | 严格1.21 | 门禁 |"
        )
        add("|--:|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|")
        for index, row in enumerate(ranked[:20], 1):
            gate_text = " | ".join(
                f"{name}:{row.get(f'{name}_f1', 0.0):.6f}/FP{row.get(f'{name}_fp', 0)}"
                f"{'PASS' if row.get(f'{name}_gate_pass') else 'FAIL'}"
                for name in gated_tracks
            )
            ratio = row.get("bar_tp_per_fp", float("nan"))
            ratio_text = "nan" if not np.isfinite(ratio) else f"{ratio:.4f}"
            add(
                f"| {index} | {row['gate_stage']} | {row['area_min']} | {row['sum_mode']} | "
                f"{row['sum_value']:g} | {row['th_base']:g} | {row[f'{primary}_f1']:.6f} | "
                f"{row[f'{primary}_tp']} | {row[f'{primary}_fp']} | {row[f'{primary}_fn']} | "
                f"{row.get(f'd_{primary}_f1', 0.0) * 100:+.4f} pp | {row.get(f'd_{primary}_tp', 0):+d} | "
                f"{row.get(f'd_{primary}_fp', 0):+d} | {ratio_text} | "
                f"{'Y' if row.get('bar_strict_1p21_pass') else 'n'} | {gate_text} |"
            )
        add("")
        best = ranked[0] if ranked else None
        if best is not None:
            improved = [row for row in arm_rows if row.get(f"d_{primary}_f1", -1.0) > 0]
            strict = [row for row in arm_rows if row.get("bar_strict_1p21_pass")]
            add(
                f"- 最强点：ΔF1 = {best.get(f'd_{primary}_f1', 0.0) * 100:+.4f} pp "
                f"（{primary} F1 {base.get('f1', 0.0):.6f} → {best[f'{primary}_f1']:.6f}；"
                f"gate_stage={best['gate_stage']}, area_min={best['area_min']}, "
                f"sum_mode={best['sum_mode']}, sum_value={best['sum_value']:g}, th={best['th_base']:g}）"
            )
            add(f"- ΔF1 > 0 的点数：**{len(improved)} / {len(arm_rows)}**")
            add(f"- 严格判据 `dTP/dFP > {STRICT_BAR}` 通过的点数：**{len(strict)} / {len(arm_rows)}**")
            both = [row for row in arm_rows if row.get("bar_exact_pass") and row.get("bar_strict_1p21_pass")]
            add(f"- 精确判据 `ΔF1 > 0` 与严格判据同时成立的点数：**{len(both)} / {len(arm_rows)}**")
        add("")

    add("## 2b. 三臂共同可用的门（一套参数成组应用，三臂都涨 T3 且三臂 T2 都过硬门禁）")
    add("")
    arms_present = sorted({row["arm"] for row in rows if row["stage"].startswith("k2k3")})
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if not row["stage"].startswith("k2k3"):
            continue
        key = (
            row["stage"],
            row["gate_stage"],
            row["area_min"],
            row["sum_mode"],
            row["sum_value"],
            row["tag_allow"],
            row["th_base"],
        )
        grouped.setdefault(key, {})[row["arm"]] = row
    common = [
        (key, per_arm)
        for key, per_arm in grouped.items()
        if len(per_arm) == len(arms_present) and arms_present
    ]
    safe = [
        (key, per_arm)
        for key, per_arm in common
        if all(per_arm[arm].get(f"d_{primary}_f1", -1.0) > 0 for arm in arms_present)
        and all(per_arm[arm].get("t2_gate_pass") for arm in arms_present)
    ]
    add(f"共同参数点共 {len(common)} 个；其中 **{len(safe)} 个**满足「三臂 T3 全涨 **且** 三臂 T2 全过硬门禁」。")
    add("")
    if safe:
        # NOTE: `per_arm` only ever existed as a *list-comprehension* variable, and in
        # Python 3 a comprehension has its own scope -> the lambda below could not see it
        # ("NameError: free variable 'per_arm' referenced before assignment").  Take the
        # per-arm mapping straight out of the sorted item instead.
        safe.sort(
            key=lambda item: min(item[1][arm].get(f"d_{primary}_f1", -1.0) for arm in arms_present)
        )
        add("按「最差臂的涨幅」排序（最差臂决定这条曲线能不能上线）：")
        add("")
        add(
            "| # | gate_stage | area_min | sum_mode | sum_value | tags | th | "
            + " | ".join(f"{arm} dF1 / T2 F1 / T2 FP / gate" for arm in arms_present)
            + " | min dF1 | mean dF1 |"
        )
        add("|--:|:--|--:|:--|--:|:--|--:|" + "--:|" * (len(arms_present) + 2))
        for index, (key, per_arm) in enumerate(safe[:15], 1):
            cells = []
            for arm in arms_present:
                row = per_arm[arm]
                cells.append(
                    f"{row.get(f'd_{primary}_f1', 0.0) * 100:+.4f}pp / {row.get('t2_f1', 0.0):.6f} / "
                    f"{row.get('t2_fp', 0)} / {'PASS' if row.get('t2_gate_pass') else 'FAIL'}"
                )
            deltas = [per_arm[arm].get(f"d_{primary}_f1", 0.0) for arm in arms_present]
            add(
                f"| {index} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {key[5]} | {key[6]} | "
                + " | ".join(cells)
                + f" | {min(deltas) * 100:+.4f}pp | {sum(deltas) / len(deltas) * 100:+.4f}pp |"
            )
        add("")
    add("## 3. 中间量（否定性结论的量化依据）")
    add("")
    add("基线工作点（radius=768，th_base=0.30）上，每一个漏检框的归类：")
    add("")
    add("| 臂 | 轨 | fn_absent | 框内**无任何候选**（真没冒头，决策层救不回） | 框内有候选但峰值低于阈值（降阈值可救） | 框内有候选却被跨尺度去重挤掉（缩小 radius 可救） |")
    add("|:--|:--|--:|--:|--:|--:|")
    for record in diagnostics:
        kinds = record["fn_kinds"]
        add(
            f"| {record['arm']} | {record['track']} | {record['counts']['fn_absent']} | "
            f"{kinds.get('no_candidate_ever_in_box', 0)} | "
            f"{kinds.get('in_box_candidate_below_th_recoverable', 0)} | "
            f"{kinds.get('in_box_candidate_above_th_but_suppressed_by_dedup', 0)} |"
        )
    add("")
    add("TP / FP 的候选属性分布（分 tag，因为不同尺度 tag 的量纲不可直接比）：")
    add("")
    add(
        "| 臂 | 轨 | tag | nTP | nFP | TP area p10/p50 | FP area p10/p50 | TP sums p10/p50 | FP sums p10/p50 | "
        "TP score p10/p50 | FP score p10/p50 |"
    )
    add("|:--|:--|--:|--:|--:|:--|:--|:--|:--|:--|:--|")
    for record in diagnostics:
        for tag, attributes in record["attributes_by_tag"].items():
            def pair(name: str) -> str:
                """Format one percentile pair."""
                values = attributes.get(name)
                return "-" if not values else f"{values['p10']:.1f}/{values['p50']:.1f}"

            add(
                f"| {record['arm']} | {record['track']} | {tag} | {attributes['n_tp']} | {attributes['n_fp']} | "
                f"{pair('tp_area')} | {pair('fp_area')} | {pair('tp_sums')} | {pair('fp_sums')} | "
                f"{pair('tp_score')} | {pair('fp_score')} |"
            )
    add("")

    add("## 4. 产物")
    add("")
    add("| 文件 | 内容 |")
    add("|:--|:--|")
    add("| `sweep_k1.csv` | K1 网格（stage=k1），一行 = 一个 (臂, radius, th) |")
    add("| `sweep_k2k3.csv` | K2/K3 网格（stage=k2k3*），一行 = 一个 (臂, gate_stage, gate, th) |")
    add("| `sweep_all.csv` | 上面两张表的合并 |")
    add("| `sweep_results.json` | 全部结果 + 基线 + 能量分位表 + 自检结论 |")
    add("| `SUMMARY.md` | 本文件 |")
    add("")
    add("读法：")
    add("")
    add(f"- `{primary}_f1` = 该参数点在本臂 **{primary} 轨**上的 F1（K1/K2/K3 通用）。")
    add(f"- `{primary}_tp` / `{primary}_fp` / `{primary}_fn` = 同一参数点的 TP / FP / FN。")
    for name in gated_tracks:
        add(f"- `{name}_f1` / `{name}_fp` = 同一参数点在 **{name} 轨**上的 F1 / FP。")
        add(f"- `{name}_gate_pass` = 该参数点是否仍满足硬门禁（F1 ≥ {gate_f1_min} 且 FP ≤ {gate_fp_max}）"
            "—— **必须为 True**，否则该点不可用。")
    add(f"- `d_{primary}_f1` / `d_{primary}_tp` / `d_{primary}_fp` = 相对该臂基线（radius=768, th=0.30, 无门）的差。")
    add("- `bar_tp_per_fp` = `dTP/dFP`（带符号）；`bar_breakeven` = 精确盈亏平衡值；`bar_fp_per_tp` = 每 TP 可动的 FP 数。")
    add("- `bar_margin` = `(FP_base+GT)*dTP - TP_base*dFP`，>0 ⇔ F1 上升（精确判据）。")
    add("- `bar_fp_per_tp_cost` = `|dFP|/|dTP|`，方向敏感的净赚比；")
    add("  `bar_strict_1p21_pass` = 加候选时要求 cost < 1.2065、减候选时要求 cost > 1.2065；")
    add("  `bar_strict_1p2065_literal` = 同上但用固定的 1.2065 常数。")
    add(f"- 能量分位表口径：`sum_quantile_scope = {meta.get('sum_quantile_scope', 'arm')}`"
        f"（排除序列：{meta.get('sum_quantile_excluded', {})}）。")
    add("")
    text = "\n".join(lines) + "\n"
    (out_dir / "SUMMARY.md").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="CPU-only decision-layer sweep over landed multi-scale heatmap caches"
    )
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=CACHE_ROOT",
                        help="One cached arm; repeatable")
    parser.add_argument("--track", action="append", default=[], metavar="NAME=SEQ1,SEQ2",
                        help="One evaluation track pooled into a single metric; repeatable")
    parser.add_argument("--labels-root", required=True, help="Final_Labels directory")
    parser.add_argument("--out-dir", required=True, help="Directory for the CSV / JSON / MD artefacts")
    parser.add_argument("--stage", default="all", choices=["all", "k1", "k2k3"])
    parser.add_argument("--primary-track", default="", help="Track used for the delta columns (default: first)")
    parser.add_argument("--gated-track", action="append", default=[], metavar="NAME",
                        help="Track that must still clear the hard gate")
    parser.add_argument("--gate-f1-min", type=float, default=DEFAULT_GATE_F1_MIN)
    parser.add_argument("--gate-fp-max", type=int, default=DEFAULT_GATE_FP_MAX)
    parser.add_argument("--th-list", default="0.22,0.24,0.26,0.28,0.30,0.32,0.34")
    parser.add_argument("--radius-list", default="64,128,256,512,768")
    parser.add_argument("--k2-area-list", default="1,2,3,4,6,8,12")
    parser.add_argument("--k2-sum-list", default="0,0.05,0.1,0.2,0.3",
                        help="per-tag quantile levels of the integral energy floor")
    parser.add_argument("--k2-sum-abs-list", default="0,5,10,20,40",
                        help="absolute integral energy floors (cross-scale incomparable, control group)")
    parser.add_argument("--k2-th-list", default="0.26,0.28,0.30,0.32")
    parser.add_argument("--k2-radius", type=float, default=768.0)
    parser.add_argument("--gate-stage", default="both", choices=["pre", "post", "both"])
    parser.add_argument("--tag-allow", default="0,1,2")
    parser.add_argument("--sum-quantile-scope", default="arm", choices=["arm", "exclude-primary"],
                        help="population of the per-tag energy quantiles; 'exclude-primary' keeps the blind "
                             "hold-out out of its own energy threshold (out-of-sample measurement)")
    parser.add_argument("--baseline-radius", type=float, default=768.0)
    parser.add_argument("--baseline-th", type=float, default=0.30)
    parser.add_argument("--check-json", action="append", default=[], metavar="[ARM:]TRACK=PATH",
                        help="Frozen gate report (track*.json) compared against the computed baseline; "
                             "the arm prefix restricts the comparison to that arm")
    parser.add_argument("--until", default="2026-11-05 07:00", help="Wall-clock deadline, 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_deadline(text: str) -> datetime | None:
    """Parse the wall-clock deadline; an unparsable value disables the deadline."""
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def main() -> None:
    """Run the sweep."""
    args = parse_args()
    started = time.perf_counter()
    deadline = parse_deadline(args.until)

    arms: list[Arm] = []
    for item in args.arm:
        if "=" not in item:
            raise SystemExit(f"--arm must be NAME=CACHE_ROOT, got {item!r}")
        name, root = item.split("=", 1)
        arms.append(Arm(name=name, cache_root=Path(root)))
    tracks: list[Track] = []
    for item in args.track:
        if "=" not in item:
            raise SystemExit(f"--track must be NAME=SEQ1,SEQ2, got {item!r}")
        name, sequences = item.split("=", 1)
        tracks.append(Track(name=name, sequences=[part for part in sequences.split(",") if part]))
    if not arms or not tracks:
        raise SystemExit("at least one --arm and one --track are required")

    primary = args.primary_track or tracks[0].name
    gated_tracks = set(args.gated_track)
    labels_root = Path(args.labels_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_floats(args.th_list)
    radii = parse_floats(args.radius_list)
    k2_areas = parse_ints(args.k2_area_list)
    k2_sum_levels = parse_floats(args.k2_sum_list)
    k2_sum_abs = parse_floats(args.k2_sum_abs_list)
    k2_thresholds = parse_floats(args.k2_th_list)
    tag_allow = tuple(sorted(parse_ints(args.tag_allow)))
    gate_stages = ["pre", "post"] if args.gate_stage == "both" else [args.gate_stage]

    truth_cache: dict[str, dict[int, np.ndarray]] = {}
    contexts: list[ArmContext] = []
    rows: list[dict] = []
    stopped = False

    for arm in arms:
        if deadline is not None and datetime.now() >= deadline:
            stopped = True
            break
        if not args.quiet:
            print(f"[ARM] {arm.name} <- {arm.cache_root}", flush=True)
        context = load_arm_context(
            arm, tracks, labels_root, truth_cache, args.sum_quantile_scope, primary,
            quantile_levels=k2_sum_levels,
        )
        compute_baseline(
            context, args.baseline_radius, args.baseline_th, gated_tracks, args.gate_f1_min, args.gate_fp_max
        )
        contexts.append(context)
        for name, base in context.baselines.items():
            if not args.quiet:
                print(
                    f"[BASE] {arm.name}/{name}: TP={base['tp']} FP={base['fp']} FN={base['fn']} "
                    f"F1={base['f1']:.6f}",
                    flush=True,
                )
        if not args.quiet:
            print(
                f"[SCOPE] {arm.name}: sum_quantile_scope={context.quantile_scope} "
                f"excluded_sequences={context.quantile_excluded}",
                flush=True,
            )

    # ---- K1 ------------------------------------------------------------------------------------
    if args.stage in ("all", "k1"):
        for context in contexts:
            for radius in radii:
                if deadline is not None and datetime.now() >= deadline:
                    stopped = True
                    break
                gate = Gate(area_min=0, sum_mode="none", sum_value=0.0, tag_allow=tag_allow)
                arm_rows = sweep_rows(
                    context, "k1", radius, gate, thresholds, primary, gated_tracks,
                    args.gate_f1_min, args.gate_fp_max, "pre",
                )
                rows.extend(arm_rows)
                if not args.quiet:
                    best = max(arm_rows, key=lambda row: row[f"{primary}_f1"])
                    print(
                        f"[K1] {context.arm.name} radius={radius:g} best th={best['th_base']:g} "
                        f"{primary} F1={best[f'{primary}_f1']:.6f}",
                        flush=True,
                    )
            if stopped:
                break

    # ---- K2 / K3 -------------------------------------------------------------------------------
    if args.stage in ("all", "k2k3") and not stopped:
        for context in contexts:
            quantiles = context.sum_quantiles
            for area_min in k2_areas:
                configurations: list[Gate] = []
                for level in k2_sum_levels:
                    thresholds_by_tag = {tag: quantiles.get(tag, {}).get(level, 0.0) for tag in SCALE_TAGS}
                    configurations.append(
                        Gate(area_min=area_min, sum_mode="tagq", sum_value=level,
                             tag_allow=tag_allow, thresholds=thresholds_by_tag)
                    )
                for absolute in k2_sum_abs:
                    configurations.append(
                        Gate(area_min=area_min, sum_mode="abs", sum_value=absolute, tag_allow=tag_allow)
                    )
                for gate in configurations:
                    for gate_stage in gate_stages:
                        if deadline is not None and datetime.now() >= deadline:
                            stopped = True
                            break
                        rows.extend(
                            sweep_rows(
                                context, "k2k3", args.k2_radius, gate, k2_thresholds, primary, gated_tracks,
                                args.gate_f1_min, args.gate_fp_max, gate_stage,
                            )
                        )
                    if stopped:
                        break
                    if not args.quiet:
                        arm_rows = [
                            row for row in rows if row["stage"] == "k2k3" and row["arm"] == context.arm.name
                        ]
                        best = max(arm_rows, key=lambda row: row[f"{primary}_f1"])
                        print(
                            f"[K2K3] {context.arm.name} area>={gate.area_min} {gate.describe()} "
                            f"best {primary} F1={best[f'{primary}_f1']:.6f} "
                            f"(dF1={best.get(f'd_{primary}_f1', 0.0) * 100:+.4f} pp, {len(arm_rows)} rows)",
                            flush=True,
                        )
                if stopped:
                    break
            if stopped:
                break

    # ---- scale-source sub-sweep ---------------------------------------------------------------
    if args.stage in ("all", "k2k3") and not stopped:
        for context in contexts:
            for allowed in ((0,), (0, 1), (0, 1, 2)):
                if deadline is not None and datetime.now() >= deadline:
                    stopped = True
                    break
                gate = Gate(area_min=0, sum_mode="none", sum_value=0.0, tag_allow=allowed)
                rows.extend(
                    sweep_rows(
                        context, "k2k3_tag", args.k2_radius, gate, k2_thresholds, primary, gated_tracks,
                        args.gate_f1_min, args.gate_fp_max, "pre",
                    )
                )
            if stopped:
                break

    # ---- diagnostics: quantify the intermediate quantities behind the conclusion -------------------
    diagnostics: list[dict] = []
    for context in contexts:
        for name in context.tracks:
            if deadline is not None and datetime.now() >= deadline:
                stopped = True
                break
            diagnostics.append(run_diagnostics(context, name, args.baseline_radius, args.baseline_th))
            if not args.quiet:
                record = diagnostics[-1]
                print(
                    f"[DIAG] {context.arm.name}/{name}: FN 可救回 "
                    f"{record['recoverable_fn_frames']} / 不可救回 {record['unrecoverable_fn_frames']} "
                    f"(fn_absent={record['counts']['fn_absent']})",
                    flush=True,
                )

    # ---- outputs --------------------------------------------------------------------------------
    k1_rows = [row for row in rows if row["stage"] == "k1"]
    k2k3_rows = [row for row in rows if row["stage"].startswith("k2k3")]
    write_csv(out_dir / "sweep_k1.csv", k1_rows)
    write_csv(out_dir / "sweep_k2k3.csv", k2k3_rows)
    write_csv(out_dir / "sweep_all.csv", rows)
    (out_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # ---- self-check against the frozen gate reports ---------------------------------------------
    self_check: list[dict] = []
    for item in args.check_json:
        if "=" not in item:
            continue
        left, path = item.split("=", 1)
        if ":" in left:
            wanted_arm, track_name = left.split(":", 1)
        else:
            wanted_arm, track_name = "", left
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = payload["overall"]
        for context in contexts:
            if wanted_arm and context.arm.name != wanted_arm:
                continue
            actual = context.baselines.get(track_name)
            if actual is None:
                continue
            same = (
                int(actual["tp"]) == int(expected["tp"])
                and int(actual["fp"]) == int(expected["fp"])
                and int(actual["fn"]) == int(expected["fn"])
                and abs(float(actual["f1"]) - float(expected["f1"])) < 1e-12
            )
            self_check.append(
                {
                    "arm": context.arm.name,
                    "track": track_name,
                    "source": str(path),
                    "expected_tp": int(expected["tp"]),
                    "expected_fp": int(expected["fp"]),
                    "expected_fn": int(expected["fn"]),
                    "expected_f1": float(expected["f1"]),
                    "actual_tp": int(actual["tp"]),
                    "actual_fp": int(actual["fp"]),
                    "actual_fn": int(actual["fn"]),
                    "actual_f1": float(actual["f1"]),
                    "verdict": "PASS" if same else "FAIL",
                }
            )
            if not args.quiet:
                print(
                    f"[CHECK] {context.arm.name}/{track_name}: {'PASS' if same else 'FAIL'} "
                    f"(expected TP/FP/FN/F1 = {expected['tp']}/{expected['fp']}/{expected['fn']}/"
                    f"{expected['f1']:.6f}, actual = {actual['tp']}/{actual['fp']}/{actual['fn']}/"
                    f"{actual['f1']:.6f})",
                    flush=True,
                )

    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "machine": "huangzhe@192.168.99.40",
        "command": " ".join(sys.argv),
        "until": args.until,
        "elapsed_s": time.perf_counter() - started,
        "stopped_by_deadline": stopped,
        "primary_track": primary,
        "gated_tracks": sorted(gated_tracks),
        "gate_f1_min": args.gate_f1_min,
        "gate_fp_max": args.gate_fp_max,
        "arms": {context.arm.name: str(context.arm.cache_root) for context in contexts},
        "sum_quantile_scope": args.sum_quantile_scope,
        "sum_quantile_excluded": {
            context.arm.name: context.quantile_excluded for context in contexts
        },
        "sum_quantiles": {
            context.arm.name: {
                str(tag): {f"{level:g}": value for level, value in levels.items()}
                for tag, levels in context.sum_quantiles.items()
            }
            for context in contexts
        },
        "baselines": {
            context.arm.name: {name: base for name, base in context.baselines.items()}
            for context in contexts
        },
        "self_check": self_check,
    }
    (out_dir / "sweep_results.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    build_summary(
        out_dir, meta, contexts, rows, primary, sorted(gated_tracks), args.gate_f1_min, args.gate_fp_max,
        diagnostics,
    )

    print(f"[SUCCESS] rows={len(rows)} (k1={len(k1_rows)}, k2k3={len(k2k3_rows)}) -> {out_dir.resolve()}")
    if stopped:
        print("[STOPPED] wall-clock deadline reached; partial results were flushed to disk.")
    print(f"[SUCCESS] elapsed {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
