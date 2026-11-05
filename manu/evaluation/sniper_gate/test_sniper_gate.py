#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression stub for the no-GMC high-precision sniper gate package.

What this file is
-----------------
A **pure CPU, zero GPU, zero network, zero dataset** regression test for the controlled package in
``manu/evaluation/sniper_gate/`` (the ``trial_0049`` sniper tier, gate frozen at ``area>=1``,
``sum_mode=abs``, ``sum_value=14.5``, ``gate_stage=post``, ``tag_allow=[0,1,2]``, ``th_base=0.30``).

It reuses the **real** implementations of the decision layer:

* ``manu/evaluation/candidate_gate.py`` -- loaded **by path** with ``importlib`` and never modified,
  never copied.  ``CandidateGate.mask`` / ``from_params`` / ``to_params`` and
  ``filter_merged_frames`` / ``apply_candidate_gate`` are the upstream code that the board port has to
  match.
* ``manu/evaluation/eval_multiscale_cache_th.py`` -- the merge and the ``>=`` score-threshold rule.
* ``manu/evaluation/system_sota_longquan.py`` -- the real ``build_candidate_gate`` entry point, i.e. the
  function that decides whether the gate is built at all.

It checks four things:

A. the parameter package parses and is self-consistent (absolute energy mode, ``tag_allow`` inside
   ``{0,1,2}``, ``th_base`` frozen at 0.30, gate shipped disabled);
B. the frozen expected numbers agree with the configuration (the ``sums>=14.5`` threshold is paired with
   T3 F1 ``0.911096427`` and TP/FP/FN ``8962/176/1573``) and are reproduced **digit for digit** from the
   parameter file, from ``EXPECTED.json`` and from the constants below;
C. the boundary behaviour of the real gate: ``area<1`` rejected, ``sums<14.5`` rejected, ``sums>=14.5``
   accepted, a ``tag`` outside ``tag_allow`` rejected, and the base score threshold ``th_base`` in effect
   with its exact ``>=`` semantics;
D. no regression: with ``enabled=false`` the pipeline builds no gate at all and the candidate pool is
   **bit for bit** identical to the pre-gate pool for the same synthetic input.

Run it with::

    python3 manu/evaluation/sniper_gate/test_sniper_gate.py

Requirements: CPython 3.8+, ``numpy``.  No GPU, no network, no dataset, no cached artifacts, no training,
no inference, no gate evaluation.  Every check is a pure in-memory computation over synthetic detection
points; the ``.pkl`` caches and the ground truth are never touched.

Exit code: ``0`` when every check passes, non-zero with a loud ``[FAIL]`` report otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

# --------------------------------------------------------------------------------------------------
# Paths and the real implementation under test
# --------------------------------------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[2]  # .../ultralytics
PARAMS_PATH = PKG_DIR / "sniper_gate_params.json"
EXPECTED_PATH = PKG_DIR / "EXPECTED.json"
CANDIDATE_GATE_PATH = REPO_ROOT / "manu" / "evaluation" / "candidate_gate.py"

# The frozen values this package claims.  They are compared digit for digit against the parameter file
# and against EXPECTED.json -- the numbers come from the landed artifacts, see EXPECTED.json :: sources.
FROZEN = {
    "sum_value": 14.5,
    "area_min": 1,
    "th_base": 0.30,
    "t1_f1": 0.8909556600604635,
    "t1_tp": 21219,
    "t1_fp": 1302,
    "t1_fn": 3892,
    "t1_delta_pp": -0.4941,
    "t2_f1": 0.998979,
    "t2_f1_artifact_exact": 0.9989791016094163,
    "t2_tp": 16635,
    "t2_fp": 31,
    "t2_fn": 3,
    "t3_f1": 0.911096427,
    "t3_f1_artifact_exact": 0.9110964265744929,
    "t3_tp": 8962,
    "t3_fp": 176,
    "t3_fn": 1573,
    "rc2_t1_f1": 0.901909759,
    "rc2_t2_f1": 0.998949170,
    "rc2_t3_f1": 0.897693732,
    "rc2_t3_fp": 482,
    "rc2_t3_tp": 8972,
    "rc2_t3_fn": 1563,
    "t3_delta_vs_rc2_pp": 1.3403,
    "t3_fp_reduction_pct": 63.49,
    "weights_md5": "93627d0605c5865701b0a75d855faf0b",
    # The superseded 14.0 tier and the delta of the delivered tier against it.
    "previous_sum_value": 14.0,
    "previous_t3_f1": 0.9108548788540661,
    "previous_t3_tp": 8966,
    "previous_t3_fp": 186,
    "previous_t3_fn": 1569,
    "t3_delta_vs_previous_tier_pp": 0.0242,
    "t3_fp_change_vs_previous_tier": -10,
}

# The gate configuration this package freezes.
FROZEN_GATE = {
    "area_min": 1,
    "sum_mode": "abs",
    "sum_value": 14.5,
    "gate_stage": "post",
    "tag_allow": [0, 1, 2],
    "th_base": 0.30,
}


def _install_fusion_shim() -> str:
    """Make ``manu.evaluation.system_sota_longquan`` importable without scipy/torch/cv2.

    The decision layer under test is pure post-processing, but the module graph of
    ``system_sota_longquan`` reaches ``eval_bidirectional_track_fusion``, which imports scipy and torch.
    Neither is needed by the gate predicate, so when the real module cannot be imported a minimal shim
    that raises on call is registered instead.  The gate logic itself is always the real upstream code.

    Returns ``"real"`` or ``"shim"`` so the caller can report which mode was used.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        import manu.evaluation.eval_bidirectional_track_fusion  # noqa: F401

        return "real"
    except Exception:
        import types

        shim = types.ModuleType("manu.evaluation.eval_bidirectional_track_fusion")

        def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError(
                "the bidirectional fusion module is not available in this environment and is not used "
                "by the candidate gate"
            )

        shim.evaluate_sequence_bidirectional = _unavailable
        shim.natural_sort_key = _unavailable
        sys.modules["manu.evaluation.eval_bidirectional_track_fusion"] = shim
        return "shim"


def _load_candidate_gate() -> Any:
    """Load ``manu/evaluation/candidate_gate.py`` by path with importlib (the real file, unmodified)."""
    spec = importlib.util.spec_from_file_location("manu.evaluation.candidate_gate", CANDIDATE_GATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {CANDIDATE_GATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register it under its canonical name so that the upstream modules (system_sota_longquan) reuse
    # exactly this instance instead of importing a second copy.
    sys.modules["manu.evaluation.candidate_gate"] = module
    spec.loader.exec_module(module)
    return module


FUSION_MODE = _install_fusion_shim()
CG = _load_candidate_gate()

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    count_matches,
    merge_scales,
    merge_scales_with_attrs,
    metrics,
)
from manu.evaluation.system_sota_longquan import build_candidate_gate  # noqa: E402

# --------------------------------------------------------------------------------------------------
# Tiny check harness: [PASS] / [FAIL] lines, loud failure, non-zero exit
# --------------------------------------------------------------------------------------------------


class Checker:
    """Collects check results and prints one ``[PASS]`` / ``[FAIL]`` line per check."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0

    def run(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a failing check must never abort the run silently
            detail = f"{type(exc).__name__}: {exc}"
            self.failures.append(f"{name}: {detail}")
            print(f"[FAIL] {name} -- {detail}")
            traceback.print_exc()
            return
        self.passed += 1
        print(f"[PASS] {name}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def f1_of(tp: int, fp: int, fn: int) -> float:
    """The upstream F1, recomputed from the counts by the real ``metrics()`` helper.

    The upstream formula is ``2 * P * R / (P + R)`` with ``P = tp/(tp+fp)`` and ``R = tp/(tp+fn)``.
    It is NOT ``2*tp/(2*tp+fp+fn)``; the two differ in the last float digit, and only the upstream
    form reproduces the artifact value bit for bit.
    """
    counts = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fn_absent": 0,
        "fn_compete": 0,
        "det": tp + fp,
    }
    return float(metrics(counts, frames=1)["f1"])


# --------------------------------------------------------------------------------------------------
# Synthetic detection points (no dataset, no cache, no ground truth)
# --------------------------------------------------------------------------------------------------


def synthetic_candidates() -> dict[int, dict[int, tuple]]:
    """Two scales, one frame, four candidates -- one of them violates every gate condition.

    The payload shape is the one of ``manu/evaluation/eval_multiscale_cache_th.py``:
    ``(scores, points)`` for the plain merge and ``(scores, points, area, sums, tag)`` for the
    attribute-carrying merge.
    """
    scores = np.asarray([0.90, 0.70, 0.60, 0.40], dtype=np.float32)
    points = np.asarray(
        [[100.0, 100.0], [300.0, 300.0], [500.0, 500.0], [700.0, 700.0]], dtype=np.float32
    )
    area = np.asarray([1, 4, 0, 9], dtype=np.int32)  # index 2 has area 0 -> rejected by area_min=1
    sums = np.asarray([20.0, 14.499, 50.0, 14.5], dtype=np.float32)  # index 1 has sums < 14.5 -> rejected
    tag = np.asarray([0, 1, 2, 3], dtype=np.int16)  # index 3 has tag 3 -> rejected by tag_allow
    plain = {0: {0: (scores, points)}, 1: {0: (scores, points)}}
    with_attrs = {0: {0: (scores, points, area, sums, tag)}, 1: {0: (scores, points, area, sums, tag)}}
    return {"plain": plain, "with_attrs": with_attrs}


def merged_frames(merged: dict[int, tuple]) -> dict[int, tuple]:
    return merged


# --------------------------------------------------------------------------------------------------
# A. parameter package parses and is self-consistent
# --------------------------------------------------------------------------------------------------


def check_params_parse() -> None:
    params = load_json(PARAMS_PATH)
    assert params["package"] == "sniper_gate_trial_0049", params["package"]
    assert params["schema_version"] == "sniper_gate_params/1.1", params["schema_version"]
    assert params["enabled"] is False, "the package must ship disabled"
    assert params["upstream"]["weights_md5"] == FROZEN["weights_md5"]
    assert params["upstream"]["weights_path"].endswith("trial_0049/weights/epoch20.pt")


def check_gate_block_self_consistent() -> None:
    params = load_json(PARAMS_PATH)
    gate = params["gate"]
    assert gate["sum_mode"] == "abs", gate["sum_mode"]
    assert gate["area_min"] == FROZEN_GATE["area_min"], gate["area_min"]
    assert gate["sum_value"] == FROZEN_GATE["sum_value"], gate["sum_value"]
    assert gate["gate_stage"] == FROZEN_GATE["gate_stage"], gate["gate_stage"]
    assert gate["th_base"] == FROZEN_GATE["th_base"], gate["th_base"]
    allowed = gate["tag_allow"]
    assert isinstance(allowed, list) and allowed, allowed
    assert set(allowed) <= {0, 1, 2}, allowed
    assert allowed == FROZEN_GATE["tag_allow"], allowed


def check_cand_gate_block_matches_gate() -> None:
    params = load_json(PARAMS_PATH)
    block = params["cand_gate_block"]
    assert block["enabled"] is False, "the gate block must ship disabled"
    assert block["area_min"] == params["gate"]["area_min"]
    assert block["sum_mode"] == params["gate"]["sum_mode"]
    assert block["sum_value"] == params["gate"]["sum_value"]
    assert block["tag_allow"] == params["gate"]["tag_allow"]
    assert block["gate_stage"] == params["gate"]["gate_stage"]
    assert block["thresholds"] == {}, "an absolute energy floor must not carry a quantile table"


def check_th_base_frozen() -> None:
    """The gate must not move the frozen base score threshold."""
    params = load_json(PARAMS_PATH)
    assert params["gate"]["th_base"] == FROZEN_GATE["th_base"], params["gate"]["th_base"]
    delivery = load_json(REPO_ROOT / "manu" / "evaluation" / "system_sota_longquan_delivery_params.json")
    frozen_th_base = delivery["params"]["th_base"]
    assert params["gate"]["th_base"] == frozen_th_base, (
        f"gate th_base {params['gate']['th_base']} != frozen delivery th_base {frozen_th_base}"
    )
    # The gate object itself carries no score threshold at all.
    built = CG.CandidateGate.from_params(params["cand_gate_block"])
    assert not hasattr(built, "th_base"), "the gate must not own a score threshold"
    assert "th_base" not in built.to_params()


# --------------------------------------------------------------------------------------------------
# B. the frozen expected numbers agree with the configuration
# --------------------------------------------------------------------------------------------------


def check_expected_matches_params() -> None:
    params = load_json(PARAMS_PATH)
    expected = load_json(EXPECTED_PATH)
    assert params["expected"] == expected["expected"], "params and EXPECTED.json disagree"
    assert params["gate"] == expected["gate"], "params and EXPECTED.json disagree on the gate block"
    assert params["upstream"]["weights_md5"] == expected["upstream"]["weights_md5"]


def check_frozen_constants() -> None:
    expected = load_json(EXPECTED_PATH)["expected"]
    assert expected["t3"]["f1"] == FROZEN["t3_f1"], expected["t3"]["f1"]
    assert expected["t3"]["tp"] == FROZEN["t3_tp"], expected["t3"]["tp"]
    assert expected["t3"]["fp"] == FROZEN["t3_fp"], expected["t3"]["fp"]
    assert expected["t3"]["fn"] == FROZEN["t3_fn"], expected["t3"]["fn"]
    assert expected["t1"]["f1"] == FROZEN["t1_f1"], expected["t1"]["f1"]
    assert expected["t1"]["f1_vs_upstream_red_line_pp"] == FROZEN["t1_delta_pp"]
    assert expected["t2"]["f1"] == FROZEN["t2_f1"], expected["t2"]["f1"]
    assert expected["t2"]["f1_artifact_exact"] == FROZEN["t2_f1_artifact_exact"]
    assert expected["t2"]["tp"] == FROZEN["t2_tp"] and expected["t2"]["fp"] == FROZEN["t2_fp"]
    assert expected["t2"]["fn"] == FROZEN["t2_fn"]
    rc2 = expected["rc2"]
    assert rc2["t1_f1"] == FROZEN["rc2_t1_f1"] and rc2["t2_f1"] == FROZEN["rc2_t2_f1"]
    assert rc2["t3_f1"] == FROZEN["rc2_t3_f1"] and rc2["t3_fp"] == FROZEN["rc2_t3_fp"]
    assert rc2["t3_tp"] == FROZEN["rc2_t3_tp"] and rc2["t3_fn"] == FROZEN["rc2_t3_fn"]


def check_threshold_metric_linkage() -> None:
    """sums>=14.5 must be paired with T3 F1 0.911096427 and TP/FP/FN 8962/176/1573."""
    params = load_json(PARAMS_PATH)
    assert params["gate"]["sum_value"] == FROZEN["sum_value"] == FROZEN_GATE["sum_value"]
    t3 = params["expected"]["t3"]
    assert t3["f1"] == FROZEN["t3_f1"], t3["f1"]
    assert (t3["tp"], t3["fp"], t3["fn"]) == (FROZEN["t3_tp"], FROZEN["t3_fp"], FROZEN["t3_fn"])
    # The frozen F1 must be reproducible from the frozen counts.
    assert f1_of(FROZEN["t3_tp"], FROZEN["t3_fp"], FROZEN["t3_fn"]) == FROZEN["t3_f1_artifact_exact"]
    assert f1_of(FROZEN["t2_tp"], FROZEN["t2_fp"], FROZEN["t2_fn"]) == FROZEN["t2_f1_artifact_exact"]


def check_rc2_comparison() -> None:
    expected = load_json(EXPECTED_PATH)["expected"]
    rc2 = expected["rc2"]
    t3 = expected["t3"]
    # The frozen deltas are quoted rounded (1.3403 pp, 63.49 %), so they are compared at that precision.
    delta_pp = (t3["f1"] - rc2["t3_f1"]) * 100.0
    assert round(delta_pp, 4) == FROZEN["t3_delta_vs_rc2_pp"], delta_pp
    reduction = (1.0 - t3["fp"] / rc2["t3_fp"]) * 100.0
    assert round(reduction, 2) == FROZEN["t3_fp_reduction_pct"], reduction
    assert expected["t1"]["f1_vs_upstream_red_line_pp"] == FROZEN["t1_delta_pp"]
    assert expected["t1"]["passes_upstream_red_line"] is False
    assert expected["t2"]["passes_upstream_hard_gate"] is True


def check_previous_tier_linkage() -> None:
    """The delivered tier supersedes 14.0 and improves on it without touching the T2 margin."""
    expected = load_json(EXPECTED_PATH)
    previous = expected["previous_tier"]
    t3 = expected["expected"]["t3"]
    t2 = expected["expected"]["t2"]
    assert previous["sum_value"] == FROZEN["previous_sum_value"], previous["sum_value"]
    assert previous["superseded_by"] == FROZEN["sum_value"], previous["superseded_by"]
    assert previous["t3_f1"] == FROZEN["previous_t3_f1"], previous["t3_f1"]
    assert (previous["t3_tp"], previous["t3_fp"], previous["t3_fn"]) == (
        FROZEN["previous_t3_tp"],
        FROZEN["previous_t3_fp"],
        FROZEN["previous_t3_fn"],
    )
    # The delivered tier is better on T3 and no worse on T2.
    delta_pp = (t3["f1_artifact_exact"] - previous["t3_f1"]) * 100.0
    assert round(delta_pp, 4) == FROZEN["t3_delta_vs_previous_tier_pp"], delta_pp
    assert t3["fp"] - previous["t3_fp"] == FROZEN["t3_fp_change_vs_previous_tier"]
    assert t2["fp"] <= previous["t2_fp"], "the T2 margin must not shrink"
    assert t2["f1_artifact_exact"] >= previous["t2_f1"], "the T2 F1 must not shrink"


# --------------------------------------------------------------------------------------------------
# C. boundary behaviour of the real gate
# --------------------------------------------------------------------------------------------------


def build_gate() -> Any:
    return CG.CandidateGate.from_params(load_json(PARAMS_PATH)["cand_gate_block"])


def check_area_boundary() -> None:
    gate = build_gate()
    area = np.asarray([0, 1, 2], dtype=np.int32)
    sums = np.asarray([20.0, 20.0, 20.0], dtype=np.float32)
    tag = np.asarray([0, 0, 0], dtype=np.int16)
    mask = gate.mask(area, sums, tag)
    assert list(mask) == [False, True, True], list(mask)


def check_sums_boundary() -> None:
    gate = build_gate()
    area = np.asarray([5, 5, 5, 5], dtype=np.int32)
    # 14.499 (below), 14.5 (exactly on the floor), 14.500001 (above), and a value that is below 14.5 in
    # exact arithmetic but rounds to 14.5 in float32.
    sums = np.asarray([14.499, 14.5, 14.500001, 14.4999999], dtype=np.float32)
    tag = np.asarray([0, 0, 0, 0], dtype=np.int16)
    mask = gate.mask(area, sums, tag)
    expected = [False, True, True, True]
    assert list(mask) == expected, f"{list(mask)} != {expected}"
    # The last case is float32 semantics: 14.4999999 is not representable and becomes exactly 14.5f.
    assert float(np.float32(14.4999999)) == 14.5, float(np.float32(14.4999999))


def check_tag_boundary() -> None:
    gate = build_gate()
    area = np.asarray([5, 5, 5, 5], dtype=np.int32)
    sums = np.asarray([20.0, 20.0, 20.0, 20.0], dtype=np.float32)
    tag = np.asarray([0, 1, 2, 3], dtype=np.int16)
    mask = gate.mask(area, sums, tag)
    expected = [True, True, True, False]
    assert list(mask) == expected, f"{list(mask)} != {expected}"


def check_fused_row_tag_rejected() -> None:
    """tag 255 (the baked fused row) is outside tag_allow and must be rejected."""
    gate = build_gate()
    area = np.asarray([5], dtype=np.int32)
    sums = np.asarray([100.0], dtype=np.float32)
    tag = np.asarray([255], dtype=np.int16)
    assert not gate.mask(area, sums, tag)[0]


def check_gate_is_score_independent() -> None:
    """The gate predicate has no score input, so it cannot relax th_base."""
    gate = build_gate()
    area = np.asarray([5], dtype=np.int32)
    sums = np.asarray([20.0], dtype=np.float32)
    tag = np.asarray([0], dtype=np.int16)
    assert bool(gate.mask(area, sums, tag)[0]) is True


def check_th_base_semantics() -> None:
    """th_base is a ``>=`` threshold: 0.30 is kept, anything below it is not a detection."""
    th_base = FROZEN_GATE["th_base"]
    boxes = np.asarray([[100.0, 100.0, 20.0, 20.0]], dtype=np.float32)  # cx, cy, w, h
    point = np.asarray([[100.0, 100.0]], dtype=np.float32)

    below = count_matches([(boxes, point, np.asarray([0.29], dtype=np.float32))], th_base, 8.0, "inbox")
    assert below["tp"] == 0 and below["det"] == 0, below
    assert below["fn"] == 1, below

    on = count_matches([(boxes, point, np.asarray([0.30], dtype=np.float32))], th_base, 8.0, "inbox")
    assert on["tp"] == 1 and on["det"] == 1, on
    assert on["fn"] == 0, on


def check_apply_candidate_gate_keeps_order() -> None:
    """apply_candidate_gate keeps the survivors in the original (score-descending) order."""
    gate = build_gate()
    candidates = ["a", "b", "c", "d"]
    attrs = [(1, 20.0, 0), (4, 14.499, 1), (0, 50.0, 2), (9, 14.5, 3)]
    survivors = CG.apply_candidate_gate(candidates, gate, attrs)
    assert survivors == ["a"], survivors


# --------------------------------------------------------------------------------------------------
# D. no regression: the disabled gate leaves the pipeline bit for bit unchanged
# --------------------------------------------------------------------------------------------------


def check_disabled_gate_builds_no_gate() -> None:
    params = load_json(PARAMS_PATH)
    block = dict(params["cand_gate_block"])
    assert build_candidate_gate({"cand_gate": block}) is None, "a disabled gate must not be built"
    block["enabled"] = True
    built = build_candidate_gate({"cand_gate": block})
    assert built is not None, "an enabled gate must be built"
    assert built.key() == (FROZEN_GATE["area_min"], "abs", FROZEN_GATE["sum_value"], (0, 1, 2)), built.key()


def check_closed_gate_is_bit_identical() -> None:
    """With the gate closed the pool is bit-identical to the pre-gate pool for the same input."""
    data = synthetic_candidates()
    radius = 768.0
    frozen = merge_scales(data["plain"], radius)
    gated_merge = merge_scales_with_attrs(data["with_attrs"], radius)
    assert set(frozen) == set(gated_merge), (sorted(frozen), sorted(gated_merge))
    for frame in frozen:
        frozen_scores, frozen_points = frozen[frame]
        merged_scores, merged_points = gated_merge[frame][:2]
        assert frozen_scores.tobytes() == merged_scores.tobytes(), frame
        assert frozen_points.tobytes() == merged_points.tobytes(), frame


def check_open_gate_that_rejects_nothing_is_bit_identical() -> None:
    """An enabled gate that rejects nothing leaves the pool bit-identical to the closed path.

    This is the regression property that makes ``enabled=false`` safe: the gate is a pure filter on top
    of the same merge, so turning it on can only remove candidates, never reorder or alter them.
    """
    data = synthetic_candidates()
    radius = 768.0
    closed = merge_scales(data["plain"], radius)
    # A gate that rejects nothing: no area floor, no energy floor, every tag allowed.
    permissive = CG.CandidateGate(area_min=0, sum_mode="none", tag_allow=(0, 1, 2, 3))
    opened = CG.filter_merged_frames(merge_scales_with_attrs(data["with_attrs"], radius), permissive)
    assert set(closed) == set(opened), (sorted(closed), sorted(opened))
    for frame in closed:
        closed_scores, closed_points = closed[frame]
        opened_scores, opened_points = opened[frame]
        assert closed_scores.tobytes() == opened_scores.tobytes(), frame
        assert closed_points.tobytes() == opened_points.tobytes(), frame


def check_enabled_gate_filters_the_pool() -> None:
    """With the gate open the very same pool loses exactly the non-compliant candidates."""
    data = synthetic_candidates()
    radius = 768.0
    gate = build_gate()
    merged = merge_scales_with_attrs(data["with_attrs"], radius)
    filtered = CG.filter_merged_frames(merged, gate)
    for frame in filtered:
        scores, points = filtered[frame]
        kept = len(scores)
        assert kept == 1, f"expected 1 survivor in frame {frame}, got {kept}"
        # The survivor is the first candidate: area 1, sums 20, tag 0.
        assert float(points[0][0]) == 100.0, points
    # The frame set itself is preserved -- a fully rejected frame keeps an empty pair.
    assert set(filtered) == set(merged)


def check_gate_does_not_change_frozen_chain() -> None:
    """A disabled gate block must leave the frozen parameter set untouched."""
    params = load_json(PARAMS_PATH)
    delivery = load_json(REPO_ROOT / "manu" / "evaluation" / "system_sota_longquan_delivery_params.json")
    frozen_params = delivery["params"]
    assert "cand_gate" not in frozen_params, "the frozen delivery file must not carry a cand_gate key"
    assert build_candidate_gate(frozen_params) is None
    # The gate block of this package must not restate a frozen delivery parameter either.
    block = params["cand_gate_block"]
    for key in block:
        assert key not in frozen_params or key == "cand_gate", key


# --------------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------------

CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("A1 params package parses", check_params_parse),
    ("A2 gate block is self-consistent", check_gate_block_self_consistent),
    ("A3 cand_gate_block matches the gate block", check_cand_gate_block_matches_gate),
    ("A4 th_base stays frozen at 0.30", check_th_base_frozen),
    ("B1 EXPECTED.json agrees with the params file", check_expected_matches_params),
    ("B2 frozen constants are reproduced digit for digit", check_frozen_constants),
    ("B3 sums>=14.5 is paired with the frozen T3 numbers", check_threshold_metric_linkage),
    ("B4 RC2 comparison deltas are reproducible", check_rc2_comparison),
    ("B5 the delivered tier supersedes 14.0 and improves on it", check_previous_tier_linkage),
    ("C1 area<1 is rejected, area>=1 is kept", check_area_boundary),
    ("C2 sums<14.5 is rejected, sums>=14.5 is kept", check_sums_boundary),
    ("C3 a tag outside tag_allow is rejected", check_tag_boundary),
    ("C4 the fused row tag 255 is rejected", check_fused_row_tag_rejected),
    ("C5 the gate predicate is score independent", check_gate_is_score_independent),
    ("C6 th_base uses >= semantics (0.30 kept, 0.29 dropped)", check_th_base_semantics),
    ("C7 apply_candidate_gate keeps the survivor order", check_apply_candidate_gate_keeps_order),
    ("D1 a disabled gate builds no gate", check_disabled_gate_builds_no_gate),
    ("D2 the closed gate is bit for bit identical to the pre-gate pool", check_closed_gate_is_bit_identical),
    ("D3 an enabled gate that rejects nothing changes nothing", check_open_gate_that_rejects_nothing_is_bit_identical),
    ("D4 the open gate filters exactly the non-compliant candidates", check_enabled_gate_filters_the_pool),
    ("D5 the disabled gate does not touch the frozen chain", check_gate_does_not_change_frozen_chain),
]


def main() -> int:
    print(f"sniper gate regression stub -- package {PKG_DIR}")
    print(f"  candidate_gate.py : {CANDIDATE_GATE_PATH}")
    print(f"  fusion dependency : {FUSION_MODE} (the gate predicate itself is always the real upstream code)")
    print(f"  numpy             : {np.__version__}")
    print("-" * 78)
    checker = Checker()
    for name, fn in CHECKS:
        checker.run(name, fn)
    print("-" * 78)
    total = len(CHECKS)
    if checker.failures:
        print(f"[FAIL] {len(checker.failures)} of {total} checks failed:")
        for failure in checker.failures:
            print(f"  - {failure}")
        return 1
    print(f"[PASS] all {total} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
