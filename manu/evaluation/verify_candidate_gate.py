#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Self-proof of the discriminative candidate gate (zero GPU, CPU only).

What this script is
-------------------
The evidence script of ``manu/evaluation/candidate_gate.py``.  It answers three questions with measured
numbers instead of claims:

* **(a) gate closed** -- does the pipeline reproduce the frozen gate-chain numbers **bit for bit** when the
  gate is absent?  Both the sweep path (``manu/evaluation/sweep_decision_layer.py``) and the shipped path
  (``manu/evaluation/system_sota_longquan.py``) are checked against the frozen reports.
* **(b) gate open** -- does the gate reproduce the measured sweep rows (arm-scope energy quantiles) and what
  does it do with **out-of-sample** energy quantiles (the blind hold-out excluded from its own threshold)?
  The script also measures the **delivered default** (an absolute energy floor, i.e. a distribution-free
  constant) and scans the absolute floors and the quantile levels, always reporting the hard gate of the
  golden pool (F1 >= 0.985 and FP <= 112).
* **(c) plumbing** -- is ``merge_scales_with_attrs`` bit-identical to ``merge_scales`` on the real cache, and
  does the shipped gate path give the same numbers as the sweep path?

What this script is not
-----------------------
* It is **not** a gate, a training run or an inference run: it only reads the landed ``ms_cache`` pickles
  (pure pickle, no GPU) and the label files.
* It does **not** write into any run directory; the optional JSON report is the only artefact.
* It does **not** claim that the gate is released: the quantiles were searched on the evaluation pools, so the
  gate ships **disabled** (see ``manu/evaluation/system_sota_longquan_candgate_params.json``).

How to use it
-------------
::

    cd /tmp/pycharm_project_10ae9e2e
    source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
    export PYTHONPATH=.
    python manu/evaluation/verify_candidate_gate.py \\
        --arm trial_0049=/mnt/data/siping/runs/nogmc_t3/trials/trial_0049/gate/epoch20/ms_cache \\
        --arm cs_t0008p2=/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0008p2/ms_cache \\
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \\
        --out-json /tmp/verify_candidate_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.candidate_gate import (  # noqa: E402
    CandidateGate,
    apply_candidate_gate,
    build_gate_thresholds,
    filter_merged_frames,
    load_predictions_with_attrs,
)
from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    load_ground_truth,
    load_predictions,
    merge_scales,
    merge_scales_with_attrs,
)
from manu.evaluation.sweep_decision_layer import (  # noqa: E402
    Arm,
    Gate,
    Track,
    compute_baseline,
    load_arm_context,
    track_metrics,
)
from manu.evaluation.system_sota_longquan import (  # noqa: E402
    SCALE_TAGS,
    aggregate_metrics,
    build_candidate_gate,
    build_candidates,
    evaluate_biz,
    load_cache_meta,
    make_records,
    single_frame_dets,
)

# --------------------------------------------------------------------------------------------------
# Frozen reference numbers
# --------------------------------------------------------------------------------------------------

# The blind hold-out (T3) and the golden pool (T2) of the Longquanshan delivery.
BLIND_SEQUENCE = "VIDEO00007_19700101_001324"
GOLDEN_SEQUENCES = (
    "VIDEO00009_19700101_001531",
    "VIDEO00029_19700101_014053",
    "VIDEO00033_19700101_015000",
    "VIDEO00034_19700101_015052",
    "VIDEO00006_19700101_000703",
    "VIDEO00008_19700101_001307",
    "VIDEO00007_19700101_000818",
)

# The frozen gate reports (gate/track3_blind.json and gate/track2_golden.json of the two arms), i.e. the
# numbers the delivery chain must reproduce bit for bit when the candidate gate is absent.
FROZEN_BASELINE = {
    ("trial_0049", "t3"): (9005, 330, 1530, 0.906391545042778),
    ("trial_0049", "t2"): (16635, 38, 3, 0.9987691753474829),
    ("cs_t0008p2", "t3"): (8575, 613, 1960, 0.8695431729452923),
    ("cs_t0008p2", "t2"): (16634, 42, 4, 0.9986191991354986),
}

# The measured gate-open rows of the sweep (arm-scope energy quantiles, i.e. the blind sequence *is* part of
# the quantile population).  These are the numbers the candidate gate has to reproduce.
EXPECTED_GATE_OPEN = {
    ("trial_0049", "t3"): (8955, 171, 1580, 0.910940440465897),
    ("trial_0049", "t2"): (16635, 31, 3, 0.9989791016094163),
    ("cs_t0008p2", "t3"): (8524, 315, 2011, 0.8799421905646744),
    ("cs_t0008p2", "t2"): (16392, 29, 246, 0.9916815390665173),
}

# The gate configuration of each arm (the energy quantile level was picked on the evaluation pools).
GATE_LEVEL = {"trial_0049": 0.2, "cs_t0008p2": 0.3}
GATE_AREA_MIN = 1
GATE_TAG_ALLOW = (0, 1, 2)

# The delivered default of the gate: an ABSOLUTE energy floor, i.e. a hard constant that does not depend on
# any dataset distribution (no quantile table, hence no in-sample / out-of-sample distinction).
GATE_ABS_SUM = 10.0   # the arm-robust absolute floor (sums>=20 is better on cs_t0008p2 but breaks the T2 hard gate on trial_0049)
GATE_ABS_SCAN = (5.0, 10.0, 20.0, 40.0)

# The hard gate of the current gate chain on the golden pool (track2_golden: F1 >= 0.985 and FP <= 112).
T2_GATE_F1_MIN = 0.985
T2_GATE_FP_MAX = 112

# The frozen decision layer of the delivery; the gate does not touch any of them.
DEDUP_RADIUS = 768.0
TH_BASE = 0.30

TOLERANCE_CLOSED = 1e-12
TOLERANCE_OPEN = 1e-9


def t2_gate_verdict(metric: dict) -> str:
    """The hard-gate verdict of the golden pool: F1 >= 0.985 and FP <= 112."""
    passed = float(metric["f1"]) >= T2_GATE_F1_MIN and int(metric["fp"]) <= T2_GATE_FP_MAX
    return "PASS" if passed else "FAIL"


def parse_args() -> argparse.Namespace:
    """Command line of the verification script."""
    parser = argparse.ArgumentParser(description="Self-proof of the discriminative candidate gate")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=CACHE_ROOT",
        help="Arm name and the directory holding the <sequence>.pkl heatmap caches",
    )
    parser.add_argument("--labels-root", required=True, help="Final_Labels directory")
    parser.add_argument("--out-json", default="", help="Optional JSON report path")
    parser.add_argument(
        "--skip-production-path",
        action="store_true",
        help="Only run the sweep path (the shipped path is the slow half)",
    )
    parser.add_argument(
        "--oos-scan-levels",
        default="0,0.02,0.05,0.1,0.15,0.2,0.3",
        help="Comma separated quantile levels of the out-of-sample scan",
    )
    return parser.parse_args()


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Print one check line and return its verdict."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' | ' + detail) if detail else ''}", flush=True)
    return ok


def compare(label: str, actual: dict, expected: tuple, tolerance: float) -> bool:
    """Compare a metric row against a frozen ``(tp, fp, fn, f1)`` tuple."""
    tp, fp, fn, f1 = expected
    same = (
        int(actual["tp"]) == tp
        and int(actual["fp"]) == fp
        and int(actual["fn"]) == fn
        and abs(float(actual["f1"]) - f1) <= tolerance
    )
    return check(
        label,
        same,
        f"expected TP={tp} FP={fp} FN={fn} F1={f1!r} | actual TP={actual['tp']} FP={actual['fp']} "
        f"FN={actual['fn']} F1={float(actual['f1'])!r} | dF1={float(actual['f1']) - f1:+.3e}",
    )


def production_metrics(
    cache_root: Path,
    labels_root: Path,
    sequences: tuple[str, ...],
    gate: CandidateGate | None,
    truth_cache: dict,
) -> dict:
    """Score one track the way the shipped pipeline does.

    ``build_candidates`` -> ``make_records`` -> ``single_frame_dets`` -> ``evaluate_biz`` ->
    ``aggregate_metrics``; only the ground truth is cached, so that the label directory is scanned once per
    sequence instead of once per call.
    """
    per_sequence = []
    num_frames = 0
    for sequence in sequences:
        candidates = build_candidates(cache_root, sequence, DEDUP_RADIUS, 0.0, gate)
        meta = load_cache_meta(cache_root, sequence)
        key = f"{labels_root}::{sequence}"
        if key not in truth_cache:
            truth_cache[key] = load_ground_truth(
                labels_root, sequence, int(meta["native_size"][0]), int(meta["native_size"][1])
            )
        records = make_records(candidates, truth_cache[key], sequence)
        dets = single_frame_dets(records, TH_BASE)
        per_sequence.append(evaluate_biz(records, dets, num_frames=int(meta["frames"])))
        num_frames += int(meta["frames"])
    return aggregate_metrics(per_sequence, num_frames)


def gate_from_thresholds(level: float, thresholds: dict[int, dict[float, float]]) -> CandidateGate:
    """Build the candidate gate of one arm from a per-tag quantile table."""
    return CandidateGate(
        area_min=GATE_AREA_MIN,
        sum_mode="tagq",
        sum_value=level,
        tag_allow=GATE_TAG_ALLOW,
        thresholds={tag: float(thresholds.get(tag, {}).get(level, 0.0)) for tag in SCALE_TAGS},
    )


def merge_equivalence(cache_root: Path, sequences: tuple[str, ...], radius: float) -> tuple[bool, str]:
    """Check that ``merge_scales_with_attrs`` keeps exactly the candidates of ``merge_scales``."""
    for sequence in sequences:
        _meta, by_tag = load_predictions(cache_root, sequence, list(SCALE_TAGS), 0.0)
        _meta2, by_tag_attrs = load_predictions_with_attrs(cache_root, sequence, list(SCALE_TAGS), 0.0)
        plain = merge_scales(by_tag, radius)
        rich = merge_scales_with_attrs(by_tag_attrs, radius)
        if set(plain) != set(rich):
            return False, f"{sequence}: frame sets differ ({len(plain)} vs {len(rich)})"
        for frame_index, (scores, points) in plain.items():
            rich_scores, rich_points = rich[frame_index][0], rich[frame_index][1]
            if scores.tobytes() != rich_scores.tobytes() or points.tobytes() != rich_points.tobytes():
                return False, f"{sequence}/{frame_index}: kept candidates differ"
    return True, f"{len(sequences)} sequences, {radius:g}px radius: byte-identical (scores, points)"


def gate_api_equivalence(
    cache_root: Path,
    sequences: tuple[str, ...],
    radius: float,
    gate: CandidateGate,
) -> tuple[bool, str]:
    """Check that ``apply_candidate_gate`` keeps exactly the candidates of ``CandidateGate.mask``.

    The public API works on a candidate list plus the aligned ``(area, sums, tag)`` triples; the mask works
    on the three attribute arrays.  Both must select the same candidates of every frame.
    """
    checked = 0
    for sequence in sequences:
        _meta, by_tag = load_predictions_with_attrs(cache_root, sequence, list(SCALE_TAGS), 0.0)
        merged = merge_scales_with_attrs(by_tag, radius)
        filtered_frames = filter_merged_frames(merged, gate)
        for frame_index, payload in merged.items():
            scores, points, area, sums, tag = payload
            candidates = [
                (float(scores[index]), float(points[index][0]), float(points[index][1]))
                for index in range(len(scores))
            ]
            attrs = [
                (int(area[index]), float(sums[index]), int(tag[index])) for index in range(len(scores))
            ]
            kept = apply_candidate_gate(candidates, gate, attrs)
            mask = gate.mask(area, sums, tag)
            expected = [candidate for index, candidate in enumerate(candidates) if mask[index]]
            if len(kept) != len(expected):
                return False, f"{sequence}/{frame_index}: {len(kept)} kept, {len(expected)} expected"
            for kept_item, expected_item in zip(kept, expected):
                if kept_item != expected_item:
                    return False, f"{sequence}/{frame_index}: kept candidate differs"
            # the shipped per-frame filter must keep the very same candidates
            filtered_scores, filtered_points = filtered_frames[frame_index]
            if len(filtered_scores) != len(kept):
                return False, f"{sequence}/{frame_index}: filter_merged_frames kept {len(filtered_scores)}"
            for index, item in enumerate(kept):
                score, x, y = item
                if float(filtered_scores[index]) != score or (
                    float(filtered_points[index][0]) != x or float(filtered_points[index][1]) != y
                ):
                    return False, f"{sequence}/{frame_index}: filter_merged_frames kept a different candidate"
            checked += len(candidates)
    return True, f"{checked} merged candidates over {len(sequences)} sequences: identical selection"


def check_params_files() -> tuple[bool, str]:
    """The frozen delivery parameters must be untouched and the new file must keep every frozen value.

    The frozen file (``system_sota_longquan_delivery_params.json``) carries the note "Do not edit it in place",
    so it must not even *contain* a ``cand_gate`` key.  The candidate-gate files must repeat every frozen
    parameter value unchanged, add the ``cand_gate`` block, and ship it disabled.
    """
    frozen_path = PROJECT_ROOT / "manu/evaluation/system_sota_longquan_delivery_params.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if "cand_gate" in frozen.get("params", {}):
        return False, "the frozen delivery file has a cand_gate key -- it must not be edited in place"
    if build_candidate_gate(frozen["params"]) is not None:
        return False, "the frozen parameter file must not build a gate"
    details = []
    for name in (
        "system_sota_longquan_candgate_params.json",
        "system_sota_longquan_candgate_oos.json",
    ):
        path = PROJECT_ROOT / "manu/evaluation" / name
        if not path.is_file():
            return False, f"missing {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        new_params = dict(payload.get("params", {}))
        block = new_params.pop("cand_gate", None)
        if block is None:
            return False, f"{name}: no cand_gate block"
        if new_params != frozen["params"] or list(new_params) != list(frozen["params"]):
            return False, f"{name}: the frozen parameter values differ"
        if block.get("enabled") is not False:
            return False, f"{name}: the candidate gate must ship disabled"
        if build_candidate_gate(payload["params"]) is not None:
            return False, f"{name}: a disabled cand_gate block must not build a gate"
        if block.get("sum_mode") == "abs" and block.get("thresholds"):
            return False, f"{name}: an absolute energy floor must not carry a quantile table"
        if block.get("sum_mode") == "tagq" and not block.get("thresholds"):
            return False, f"{name}: a quantile gate needs its per-tag threshold table"
        details.append(f"{name}[{block['sum_mode']} sum_value={block['sum_value']} disabled]")
    return True, (
        "frozen file has no cand_gate key | 34 frozen values identical in both new files | "
        + " | ".join(details)
        + " | no file builds a gate"
    )


def main() -> int:
    """Run the verification and report the verdict."""
    args = parse_args()
    arms: list[Arm] = []
    for item in args.arm:
        if "=" not in item:
            raise SystemExit(f"--arm must be NAME=CACHE_ROOT, got {item!r}")
        name, root = item.split("=", 1)
        arms.append(Arm(name=name, cache_root=Path(root)))
    labels_root = Path(args.labels_root)
    tracks = [Track(name="t3", sequences=[BLIND_SEQUENCE]), Track(name="t2", sequences=list(GOLDEN_SEQUENCES))]
    truth_cache: dict = {}
    levels = sorted({level for level in GATE_LEVEL.values()})
    report: dict = {"arms": {}, "checks": []}
    failures = 0

    print("=" * 100)
    print("[CFG] decision layer: dedup_radius=768px (frozen) | th_base=0.30 (frozen) | "
          f"gate_stage=post | area_min={GATE_AREA_MIN} | sum_mode=tagq | tag_allow={list(GATE_TAG_ALLOW)}")
    print(f"[CFG] blind hold-out (excluded from the out-of-sample quantiles): {BLIND_SEQUENCE}")
    print(f"[CFG] golden pool (the out-of-sample quantile population): {len(GOLDEN_SEQUENCES)} sequences")
    print("=" * 100)

    print("\n[D] PARAMETER FILES (the frozen delivery configuration must be untouched)", flush=True)
    ok, detail = check_params_files()
    failures += 0 if check("frozen delivery params untouched / cand_gate file valid", ok, detail) else 1
    report["params_check"] = {"ok": ok, "detail": detail}

    for arm in arms:
        level = GATE_LEVEL.get(arm.name)
        if level is None:
            raise SystemExit(f"no gate level known for arm {arm.name!r}")
        print(f"\n### ARM {arm.name}  ({arm.cache_root})", flush=True)

        context = load_arm_context(
            arm,
            tracks,
            labels_root,
            truth_cache,
            quantile_scope="arm",
            primary_track="t3",
            quantile_levels=levels,
        )
        compute_baseline(context, DEDUP_RADIUS, TH_BASE, {"t2"}, 0.985, 112)

        # ---- (a) gate closed -------------------------------------------------------------------
        print("\n[A] GATE CLOSED -- must reproduce the frozen gate reports bit for bit", flush=True)
        for track in tracks:
            actual = context.baselines[track.name]
            ok = compare(
                f"{arm.name}/{track.name} (sweep path)",
                actual,
                FROZEN_BASELINE[(arm.name, track.name)],
                TOLERANCE_CLOSED,
            )
            failures += 0 if ok else 1
            report["checks"].append(
                {
                    "arm": arm.name,
                    "track": track.name,
                    "stage": "closed",
                    "path": "sweep",
                    "actual": {key: actual[key] for key in ("tp", "fp", "fn", "f1")},
                    "expected": FROZEN_BASELINE[(arm.name, track.name)],
                    "ok": ok,
                }
            )

        if not args.skip_production_path:
            for track in tracks:
                actual = production_metrics(
                    arm.cache_root, labels_root, tuple(track.sequences), None, truth_cache
                )
                ok = compare(
                    f"{arm.name}/{track.name} (shipped path, system_sota_longquan)",
                    actual,
                    FROZEN_BASELINE[(arm.name, track.name)],
                    TOLERANCE_CLOSED,
                )
                failures += 0 if ok else 1
                report["checks"].append(
                    {
                        "arm": arm.name,
                        "track": track.name,
                        "stage": "closed",
                        "path": "shipped",
                        "actual": {key: actual[key] for key in ("tp", "fp", "fn", "f1")},
                        "expected": FROZEN_BASELINE[(arm.name, track.name)],
                        "ok": ok,
                    }
                )

        # ---- energy thresholds ------------------------------------------------------------------
        arm_thresholds = context.sum_quantiles
        oos_thresholds = build_gate_thresholds(arm.cache_root, levels, {BLIND_SEQUENCE})
        print("\n[T] ENERGY THRESHOLDS (per tag)", flush=True)
        print(f"  {'tag':>4} {'scope':>10} {'level':>7} {'threshold':>14}")
        for tag in SCALE_TAGS:
            for scope, table in (("arm(all)", arm_thresholds), ("oos(gold)", oos_thresholds)):
                for candidate_level in levels:
                    value = table.get(tag, {}).get(candidate_level)
                    if value is None:
                        continue
                    print(f"  {tag:>4} {scope:>10} {candidate_level:>7g} {float(value):>14.6f}")
        print("  [note] 'arm(all)' includes the blind hold-out in the quantile population (the population the "
              "sweep used);")
        print("         'oos(gold)' is computed on the 7 golden sequences only, i.e. the blind hold-out is "
              "excluded")
        print("         from its own energy threshold (the out-of-sample measurement).")

        # ---- (b) gate open ---------------------------------------------------------------------
        print("\n[B] GATE OPEN", flush=True)
        gate_arm = gate_from_thresholds(level, arm_thresholds)
        gate_oos = gate_from_thresholds(level, oos_thresholds)
        print(f"  gate(arm scope) : {gate_arm.describe()} | thresholds="
              f"{ {tag: round(value, 6) for tag, value in gate_arm.thresholds.items()} }")
        print(f"  gate(oos scope) : {gate_oos.describe()} | thresholds="
              f"{ {tag: round(value, 6) for tag, value in gate_oos.thresholds.items()} }")

        for track in tracks:
            views = context.views(track.name, DEDUP_RADIUS, gate_arm, "post")
            # the sweep mask and the shipped mask must select exactly the same candidates
            masks_shipped = {
                sequence: gate_arm.mask(pool.area, pool.sums, pool.tag)
                for sequence, pool in context.pools[track.name].items()
            }
            masks_sweep = {
                sequence: Gate(
                    area_min=gate_arm.area_min,
                    sum_mode=gate_arm.sum_mode,
                    sum_value=gate_arm.sum_value,
                    tag_allow=gate_arm.tag_allow,
                    thresholds=gate_arm.thresholds,
                ).mask(pool)
                for sequence, pool in context.pools[track.name].items()
            }
            same_mask = all(
                np.array_equal(masks_shipped[sequence], masks_sweep[sequence])
                for sequence in masks_shipped
            )
            if not check(f"{arm.name}/{track.name} shipped mask == sweep mask", same_mask):
                failures += 1

            actual = track_metrics(context, track.name, views, TH_BASE, masks_shipped)
            ok = compare(
                f"{arm.name}/{track.name} gate open (arm-scope quantiles)",
                actual,
                EXPECTED_GATE_OPEN[(arm.name, track.name)],
                TOLERANCE_OPEN,
            )
            failures += 0 if ok else 1
            report["checks"].append(
                {
                    "arm": arm.name,
                    "track": track.name,
                    "stage": "open-arm-scope",
                    "actual": {key: actual[key] for key in ("tp", "fp", "fn", "f1")},
                    "expected": EXPECTED_GATE_OPEN[(arm.name, track.name)],
                    "ok": ok,
                }
            )

            views_oos = context.views(track.name, DEDUP_RADIUS, gate_oos, "post")
            masks_oos = {
                sequence: gate_oos.mask(pool.area, pool.sums, pool.tag)
                for sequence, pool in context.pools[track.name].items()
            }
            actual_oos = track_metrics(context, track.name, views_oos, TH_BASE, masks_oos)
            print(
                f"  -> {arm.name}/{track.name} IN-SAMPLE LEVEL under the OOS threshold: "
                f"TP={actual_oos['tp']} FP={actual_oos['fp']} FN={actual_oos['fn']} "
                f"F1={float(actual_oos['f1']):.6f} T2gate={t2_gate_verdict(actual_oos)} "
                f"(arm-scope: TP={actual['tp']} FP={actual['fp']} F1={float(actual['f1']):.6f})",
                flush=True,
            )
            report["checks"].append(
                {
                    "arm": arm.name,
                    "track": track.name,
                    "stage": "open-oos-scope",
                    "actual": {key: actual_oos[key] for key in ("tp", "fp", "fn", "f1")},
                    "expected": None,
                    "note": "informational: the out-of-sample threshold has no frozen expectation",
                    "ok": None,
                }
            )

            if not args.skip_production_path:
                shipped = production_metrics(
                    arm.cache_root, labels_root, tuple(track.sequences), gate_oos, truth_cache
                )
                same = (
                    int(shipped["tp"]) == int(actual_oos["tp"])
                    and int(shipped["fp"]) == int(actual_oos["fp"])
                    and int(shipped["fn"]) == int(actual_oos["fn"])
                    and abs(float(shipped["f1"]) - float(actual_oos["f1"])) <= TOLERANCE_OPEN
                )
                ok = check(
                    f"{arm.name}/{track.name} shipped path == sweep path (gate open, oos quantiles)",
                    same,
                    f"shipped TP={shipped['tp']} FP={shipped['fp']} FN={shipped['fn']} "
                    f"F1={float(shipped['f1'])!r} | sweep TP={actual_oos['tp']} FP={actual_oos['fp']} "
                    f"FN={actual_oos['fn']} F1={float(actual_oos['f1'])!r}",
                )
                failures += 0 if ok else 1
                report["checks"].append(
                    {
                        "arm": arm.name,
                        "track": track.name,
                        "stage": "open-oos-shipped",
                        "actual": {key: shipped[key] for key in ("tp", "fp", "fn", "f1")},
                        "expected": {
                            key: actual_oos[key] for key in ("tp", "fp", "fn", "f1")
                        },
                        "ok": ok,
                    }
                )

        # ---- (b2) out-of-sample level scan ------------------------------------------------------
        scan_levels = sorted({float(item) for item in args.oos_scan_levels.split(",") if item.strip()})
        scan_thresholds = build_gate_thresholds(arm.cache_root, scan_levels, {BLIND_SEQUENCE})
        print(
            "\n[B2] PER-TAG QUANTILE LEVEL SCAN with OUT-OF-SAMPLE thresholds (the blind hold-out is\n"
            "     never part of the quantile population).  dF1 is against the gate-closed baseline and\n"
            "     'T2gate' is the hard gate of the golden pool (F1 >= 0.985 and FP <= 112).\n"
            "     The level that was picked INSIDE the sample (t49 0.2 / p2 0.3) fails out of sample;\n"
            "     a milder level keeps the T3 gain and still passes the T2 hard gate.",
            flush=True,
        )
        print(f"  {'level':>6} {'track':>4} {'TP':>7} {'FP':>6} {'FN':>6} {'F1':>10} {'dF1':>11} {'T2gate':>7}")
        scan_rows = []
        for candidate_level in scan_levels:
            gate_scan = gate_from_thresholds(candidate_level, scan_thresholds)
            for track in tracks:
                views_scan = context.views(track.name, DEDUP_RADIUS, gate_scan, "post")
                masks_scan = {
                    sequence: gate_scan.mask(pool.area, pool.sums, pool.tag)
                    for sequence, pool in context.pools[track.name].items()
                }
                metric = track_metrics(context, track.name, views_scan, TH_BASE, masks_scan)
                closed = context.baselines[track.name]
                delta = float(metric["f1"]) - float(closed["f1"])
                verdict = t2_gate_verdict(metric) if track.name == "t2" else "-"
                print(
                    f"  {candidate_level:>6g} {track.name:>4} {metric['tp']:>7} {metric['fp']:>6} "
                    f"{metric['fn']:>6} {float(metric['f1']):>10.6f} {delta * 100:>+10.4f}pp {verdict:>7}",
                    flush=True,
                )
                scan_rows.append(
                    {
                        "arm": arm.name,
                        "track": track.name,
                        "level": candidate_level,
                        "thresholds": {
                            str(tag): float(value) for tag, value in gate_scan.thresholds.items()
                        },
                        "tp": int(metric["tp"]),
                        "fp": int(metric["fp"]),
                        "fn": int(metric["fn"]),
                        "f1": float(metric["f1"]),
                        "d_f1": delta,
                        "t2_gate": verdict,
                    }
                )
        report.setdefault("oos_scan", []).extend(scan_rows)

        # ---- (b3) the delivered default: an absolute energy floor ---------------------------------
        print(
            "\n[B3] DELIVERED DEFAULT -- absolute energy floor (abs), post, th_base=0.30.\n"
            "     An absolute floor is a hard constant: it does not depend on any dataset distribution and\n"
            "     therefore needs no quantile table (no in-sample / out-of-sample distinction).",
            flush=True,
        )
        gate_default = CandidateGate(
            area_min=GATE_AREA_MIN,
            sum_mode="abs",
            sum_value=GATE_ABS_SUM,
            tag_allow=GATE_TAG_ALLOW,
        )
        print(f"  gate: {gate_default.describe()}")
        for track in tracks:
            views_default = context.views(track.name, DEDUP_RADIUS, gate_default, "post")
            masks_default = {
                sequence: gate_default.mask(pool.area, pool.sums, pool.tag)
                for sequence, pool in context.pools[track.name].items()
            }
            metric = track_metrics(context, track.name, views_default, TH_BASE, masks_default)
            closed = context.baselines[track.name]
            delta = float(metric["f1"]) - float(closed["f1"])
            verdict = t2_gate_verdict(metric) if track.name == "t2" else "-"
            print(
                f"  {arm.name}/{track.name}: TP={metric['tp']} FP={metric['fp']} FN={metric['fn']} "
                f"F1={float(metric['f1']):.6f} dF1={delta * 100:+.4f}pp "
                f"T2gate={verdict} (closed: TP={closed['tp']} FP={closed['fp']} "
                f"F1={float(closed['f1']):.6f})",
                flush=True,
            )
            report.setdefault("delivered_default", []).append(
                {
                    "arm": arm.name,
                    "track": track.name,
                    "gate": gate_default.to_params(),
                    "tp": int(metric["tp"]),
                    "fp": int(metric["fp"]),
                    "fn": int(metric["fn"]),
                    "f1": float(metric["f1"]),
                    "d_f1": delta,
                    "t2_gate": verdict,
                }
            )
            if not args.skip_production_path:
                shipped = production_metrics(
                    arm.cache_root, labels_root, tuple(track.sequences), gate_default, truth_cache
                )
                same = (
                    int(shipped["tp"]) == int(metric["tp"])
                    and int(shipped["fp"]) == int(metric["fp"])
                    and int(shipped["fn"]) == int(metric["fn"])
                    and abs(float(shipped["f1"]) - float(metric["f1"])) <= TOLERANCE_OPEN
                )
                ok = check(
                    f"{arm.name}/{track.name} shipped path == sweep path (delivered default, abs)",
                    same,
                    f"shipped TP={shipped['tp']} FP={shipped['fp']} FN={shipped['fn']} "
                    f"F1={float(shipped['f1'])!r} | sweep TP={metric['tp']} FP={metric['fp']} "
                    f"FN={metric['fn']} F1={float(metric['f1'])!r}",
                )
                failures += 0 if ok else 1

        # ---- (b4) absolute floor scan ----------------------------------------------------------
        print("\n[B4] ABSOLUTE ENERGY FLOOR SCAN (post, th_base=0.30)", flush=True)
        print(f"  {'floor':>6} {'track':>4} {'TP':>7} {'FP':>6} {'FN':>6} {'F1':>10} {'dF1':>11} {'T2gate':>7}")
        for floor in GATE_ABS_SCAN:
            gate_floor = CandidateGate(
                area_min=GATE_AREA_MIN,
                sum_mode="abs",
                sum_value=floor,
                tag_allow=GATE_TAG_ALLOW,
            )
            for track in tracks:
                views_floor = context.views(track.name, DEDUP_RADIUS, gate_floor, "post")
                masks_floor = {
                    sequence: gate_floor.mask(pool.area, pool.sums, pool.tag)
                    for sequence, pool in context.pools[track.name].items()
                }
                metric = track_metrics(context, track.name, views_floor, TH_BASE, masks_floor)
                closed = context.baselines[track.name]
                delta = float(metric["f1"]) - float(closed["f1"])
                verdict = t2_gate_verdict(metric) if track.name == "t2" else "-"
                print(
                    f"  {floor:>6g} {track.name:>4} {metric['tp']:>7} {metric['fp']:>6} {metric['fn']:>6} "
                    f"{float(metric['f1']):>10.6f} {delta * 100:>+10.4f}pp {verdict:>7}",
                    flush=True,
                )
                report.setdefault("abs_scan", []).append(
                    {
                        "arm": arm.name,
                        "track": track.name,
                        "floor": floor,
                        "tp": int(metric["tp"]),
                        "fp": int(metric["fp"]),
                        "fn": int(metric["fn"]),
                        "f1": float(metric["f1"]),
                        "d_f1": delta,
                        "t2_gate": verdict,
                    }
                )

        # ---- (c) plumbing -----------------------------------------------------------------------
        print("\n[C] PLUMBING", flush=True)
        ok, detail = merge_equivalence(
            arm.cache_root, tuple(GOLDEN_SEQUENCES) + (BLIND_SEQUENCE,), DEDUP_RADIUS
        )
        failures += 0 if check(f"{arm.name} merge_scales_with_attrs == merge_scales", ok, detail) else 1
        ok, detail = gate_api_equivalence(
            arm.cache_root, tuple(GOLDEN_SEQUENCES) + (BLIND_SEQUENCE,), DEDUP_RADIUS, gate_oos
        )
        failures += 0 if check(f"{arm.name} apply_candidate_gate == CandidateGate.mask", ok, detail) else 1

        report["arms"][arm.name] = {
            "cache_root": str(arm.cache_root),
            "gate_level": level,
            "thresholds_arm_scope": {
                str(tag): {f"{lvl:g}": value for lvl, value in table.items()}
                for tag, table in arm_thresholds.items()
            },
            "thresholds_oos": {
                str(tag): {f"{lvl:g}": value for lvl, value in table.items()}
                for tag, table in oos_thresholds.items()
            },
        }

    print("\n" + "=" * 100)
    print(f"[VERDICT] {'PASS' if failures == 0 else f'FAIL ({failures} failed checks)'}")
    print("=" * 100)

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"[SUCCESS] report -> {path.resolve()}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
