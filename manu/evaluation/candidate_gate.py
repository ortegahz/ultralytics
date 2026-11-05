#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Discriminative candidate gate for the Longquanshan decision layer.

What this module is
-------------------
A **post-processing** step that removes weak candidates from the merged candidate pool of the multi-scale
heatmap cache, *after* the cross-scale de-duplication and *before* the score threshold is applied.  The
candidates carry three attributes the cache already stores but the decision layer used to throw away:

* ``area``  -- the pixel count of the connected component (on the grid of its own scale),
* ``sums``  -- the integral of the heatmap inside the component (the "energy" of the blob),
* ``tag``   -- the scale the candidate came from (0 = native 640, 1 = 160, 2 = 80).

The gate keeps a candidate when **all** of the following hold::

    area  >= area_min
    sums  >= sum_threshold(tag)
    tag   in tag_allow

with ``sum_threshold(tag)`` being either an absolute floor (``sum_mode = "abs"``) or a per-tag quantile
level resolved against a frozen table (``sum_mode = "tagq"``).  The global score threshold ``th_base`` is
**not** touched by the gate.

What this module is not
-----------------------
* It is **not** a new detector, not a re-training and not an inference path: it never touches a GPU and never
  rebuilds a cache.
* It is **not** a threshold change.  ``th_base`` / ``dedup_radius`` stay exactly as the frozen delivery
  configuration has them.
* It is **not** a second implementation of the evaluation.  The matching and the metric aggregation are
  imported from the existing gate chain (``manu/evaluation/eval_multiscale_cache_th.py`` and
  ``manu/evaluation/system_sota_longquan.py``).
* It is **not** a released feature.  The evidence behind the shipped parameters is *exploratory*: the
  quantiles were searched on the evaluation pools themselves, so the numbers are not out-of-sample.  The
  delivered parameter file therefore ships the gate **disabled**.

How to use it
-------------
Offline (offline evidence, CPU only, reads the landed ``ms_cache`` pickles)::

    from manu.evaluation.candidate_gate import CandidateGate, build_gate_thresholds
    thresholds = build_gate_thresholds(cache_root, [0.2, 0.3], exclude_sequences={blind_sequence})
    gate = CandidateGate(area_min=1, sum_mode="tagq", sum_value=0.2, tag_allow=(0, 1, 2),
                         thresholds={tag: thresholds[tag][0.2] for tag in (0, 1, 2)})

In the pipeline (the gate is optional and disabled by default)::

    python manu/evaluation/system_sota_longquan.py --params-json <params with cand_gate.enabled = true>

The per-tag quantile table is **not** re-implemented here: it is delegated to
``manu.evaluation.sweep_decision_layer.build_sum_quantiles`` so that there is a single definition of the
energy distribution.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from manu.evaluation.eval_multiscale_cache_th import iter_frames
from manu.evaluation.sweep_decision_layer import build_sum_quantiles

# The scale sources of the cache; the baked fused row (255) is deliberately excluded.
SCALE_TAGS = (0, 1, 2)

# The accepted values of ``sum_mode``.
SUM_MODES = ("none", "abs", "tagq")


def build_gate_thresholds(
    cache_root: Path,
    levels: Sequence[float],
    exclude_sequences: set[str] | None = None,
) -> dict[int, dict[float, float]]:
    """Per-tag ``sums`` quantile table of one arm, delegated to the sweep implementation.

    ``exclude_sequences`` removes sequences from the population, which is how the blind hold-out is kept out
    of its own energy threshold (the out-of-sample measurement).
    """
    return build_sum_quantiles(cache_root, list(levels), exclude_sequences)


@dataclass
class CandidateGate:
    """The discriminative candidate gate.

    The fields are semantically identical to ``manu.evaluation.sweep_decision_layer.Gate``; the class exists
    so that the shipped pipeline does not have to import the sweep script.

    ``area_min``   absolute connected-component area floor, in the pixel grid of the candidate's own scale.
    ``sum_mode``   ``none`` (no energy floor), ``abs`` (absolute ``sums`` floor) or ``tagq`` (per-tag quantile).
    ``sum_value``  the floor itself: an absolute ``sums`` value in ``abs`` mode, a quantile level in ``tagq``
                   mode.
    ``tag_allow``  the scale sources that survive the gate (0 = native 640, 1 = 160, 2 = 80).
    ``thresholds`` the resolved per-tag ``sums`` thresholds of a ``tagq`` gate, keyed by the scale tag.
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

    def mask(self, area, sums, tag) -> np.ndarray:
        """Boolean mask of the candidates that satisfy the gate.

        The comparison dtypes are the ones of the sweep (``area`` int32, ``sums`` float32, ``tag`` int16) so
        that the gate of the pipeline and the gate of the sweep select exactly the same candidates.
        """
        if self.sum_mode not in SUM_MODES:
            raise ValueError(f"unknown sum_mode {self.sum_mode!r}, expected one of {SUM_MODES}")
        area_array = np.asarray(area, dtype=np.int32)
        sums_array = np.asarray(sums, dtype=np.float32)
        tag_array = np.asarray(tag, dtype=np.int16)
        mask = area_array >= self.area_min
        if self.sum_mode == "abs":
            mask &= sums_array >= self.sum_value
        elif self.sum_mode == "tagq":
            floors = np.asarray(
                [self.thresholds.get(int(value), -np.inf) for value in tag_array],
                dtype=np.float32,
            )
            mask &= sums_array >= floors
        mask &= np.isin(tag_array, np.asarray(self.tag_allow, dtype=np.int16))
        return mask

    @classmethod
    def from_params(cls, block: dict) -> "CandidateGate":
        """Build a gate from the ``cand_gate`` block of a parameter file."""
        thresholds = {
            int(tag): float(value) for tag, value in (block.get("thresholds") or {}).items()
        }
        tag_allow = block.get("tag_allow", SCALE_TAGS)
        return cls(
            area_min=int(block.get("area_min", 0)),
            sum_mode=str(block.get("sum_mode", "none")),
            sum_value=float(block.get("sum_value", 0.0)),
            tag_allow=tuple(int(tag) for tag in tag_allow),
            thresholds=thresholds,
        )

    def to_params(self) -> dict:
        """Serialise the gate into the ``cand_gate`` block of a parameter file."""
        return {
            "area_min": int(self.area_min),
            "sum_mode": str(self.sum_mode),
            "sum_value": float(self.sum_value),
            "tag_allow": [int(tag) for tag in self.tag_allow],
            "thresholds": {str(tag): float(value) for tag, value in sorted(self.thresholds.items())},
        }


def apply_candidate_gate(candidates: Sequence[Any], gate: CandidateGate, attrs: Sequence[Sequence]) -> list:
    """Return the candidates that survive the gate.

    ``candidates`` is the candidate list of one frame (any element type) and ``attrs`` carries the
    ``(area, sums, tag)`` triple of every candidate, aligned one to one.  The surviving candidates keep their
    original order, i.e. the score-descending order of the merged pool.

    The gate stage is ``post``: the candidates are the ones that survived the cross-scale de-duplication.
    """
    items = list(candidates)
    if not items:
        return []
    if len(attrs) != len(items):
        raise ValueError(f"attrs/candidates length mismatch: {len(attrs)} != {len(items)}")
    area = np.asarray([row[0] for row in attrs], dtype=np.int32)
    sums = np.asarray([row[1] for row in attrs], dtype=np.float32)
    tag = np.asarray([row[2] for row in attrs], dtype=np.int16)
    keep = gate.mask(area, sums, tag)
    return [item for item, flag in zip(items, keep) if flag]


def filter_merged_frames(merged: dict[int, tuple], gate: CandidateGate) -> dict[int, tuple]:
    """Filter a merged candidate pool, keeping the ``(scores, points)`` pair of every frame.

    ``merged`` is the output of ``manu.evaluation.eval_multiscale_cache_th.merge_scales_with_attrs``, i.e.
    ``{frame_index: (scores, points, area, sums, tag)}``.  The frames are all kept (a frame whose candidates
    are all rejected keeps an empty pair), so that the frame set of the pipeline does not depend on the gate.
    """
    result: dict[int, tuple] = {}
    for frame_index, payload in merged.items():
        scores, points, area, sums, tag = payload
        keep = gate.mask(area, sums, tag)
        result[frame_index] = (np.asarray(scores)[keep], np.asarray(points)[keep])
    return result


def load_predictions_with_attrs(
    cache_root: Path,
    sequence: str,
    tags: Sequence[int],
    floor: float = 0.0,
) -> tuple[dict, dict[int, dict[int, tuple]]]:
    """Per-scale rows of one cached sequence, with the per-candidate attributes of the cache.

    This mirrors ``manu.evaluation.eval_multiscale_cache_th.load_predictions`` exactly (same row selection, same
    order, same per-tag grouping) and adds the ``area`` / ``sums`` / ``tag`` attributes of every row, so that
    the merged pool can be filtered by the candidate gate.

    The returned per-frame payload is ``(scores, points, area, sums, tag)`` with the dtypes of the gate:
    ``area`` int32, ``sums`` float32, ``tag`` int16.
    """
    cache = pickle.loads((Path(cache_root) / f"{sequence}.pkl").read_bytes())
    by_tag: dict[int, dict[int, tuple]] = {int(tag): {} for tag in tags}
    for frame_index, points, scores, _sizes, sums, area, row_tag in iter_frames(cache):
        rows = np.isin(row_tag, tags)
        if not rows.any():
            continue
        selected = np.nonzero(rows & (scores.astype(np.float32) >= floor))[0]
        if not len(selected):
            continue
        order = selected[np.argsort(-scores[selected], kind="stable")]
        for tag in np.unique(row_tag[order]):
            subset = order[row_tag[order] == tag]
            by_tag[int(tag)][int(frame_index)] = (
                scores[subset].astype(np.float32),
                points[subset],
                area[subset].astype(np.int32),
                sums[subset].astype(np.float32),
                row_tag[subset].astype(np.int16),
            )
    return cache["meta"], by_tag
