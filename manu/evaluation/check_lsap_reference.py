#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Differential validation of ``lsap_reference.linear_sum_assignment_ref`` against SciPy 1.15.3.

The reference implementation in :mod:`lsap_reference` is a pure-Python port of the C++ solver that
SciPy ships as ``scipy.optimize._lsap``.  This script is the acceptance gate for that port: it
compares the reference against ``scipy.optimize.linear_sum_assignment`` on a large, seed-fixed
corpus of matrices and reports *any* deviation.

Checks performed for every matrix
----------------------------------
1. **Pair set** -- the set of ``(row, col)`` pairs is identical (pairs sorted before comparison).
   This is the strict check: SciPy's *tie-breaking* is part of the contract the C++ port has to
   reproduce, so merely finding *an* optimal assignment is not enough.
2. **Total cost** -- the total cost of the two assignments is identical, compared with
   ``math.fsum`` (correctly rounded, order independent).

Exit code
---------
``0`` when every case matches, ``1`` when at least one case differs (the offending matrix is
written to ``/tmp/lsap_mismatch.npy``).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment as scipy_lsa

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from lsap_reference import linear_sum_assignment_ref  # noqa: E402

SEED = 20261119
N_RANDOM = 2000
N_LARGE = 400
MISMATCH_NPY = Path("/tmp/lsap_mismatch.npy")
MISMATCH_JSON = Path("/tmp/lsap_mismatch.json")

# Shapes cycled through by the random generator.  Includes tall (transposed), wide and square
# problems plus the degenerate 1xN / Nx1 rows.
SHAPES: list[tuple[int, int]] = [
    (2, 2),
    (3, 4),
    (5, 5),
    (7, 3),
    (1, 5),
    (5, 1),
    (4, 4),
    (6, 2),
    (2, 7),
    (3, 3),
]


def fixed_cases() -> list[tuple[str, np.ndarray]]:
    """Hand-picked matrices: SciPy's documented examples, tie-heavy symmetric matrices, repeats."""
    cases: list[tuple[str, np.ndarray]] = []

    # --- classic examples from the SciPy documentation -------------------------------
    doc_examples = [
        ("doc_2x3", [[1, 2, 3], [2, 1, 1]]),
        ("doc_3x3", [[5, 4, 3], [2, 0, 1], [3, 1, 2]]),
        ("doc_3x3_b", [[10, 19, 8], [15, 5, 13], [4, 3, 6]]),
        ("doc_3x3_c", [[10, 19, 8], [15, 5, 13], [4, 3, 6]]),
    ]
    for name, mat in doc_examples:
        cases.append((name, np.asarray(mat, dtype=np.float64)))

    # --- matrices with a large number of equally optimal solutions --------------------
    tie_cases = [
        ("tie_2x2_01", [[0, 1], [1, 0]]),
        ("tie_2x2_11", [[1, 1], [1, 1]]),
        ("tie_2x3_zeros", [[0, 0, 0], [0, 0, 0]]),
        ("tie_3x3_zeros", np.zeros((3, 3))),
        ("tie_4x4_zeros", np.zeros((4, 4))),
        ("tie_3x3_five", [[5, 5, 5], [5, 5, 5], [5, 5, 5]]),
        ("tie_3x3_alt", [[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        ("tie_2x2_neg", [[-1, -1], [-1, -1]]),
        ("tie_4x4_rows", [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
        ("tie_3x3_block", [[0, 0, 9], [0, 0, 9], [9, 9, 0]]),
    ]
    for name, mat in tie_cases:
        cases.append((name, np.asarray(mat, dtype=np.float64)))

    # --- matrices with many repeated values ------------------------------------------
    repeat_cases = [
        ("rep_3x3", [[1, 1, 2], [1, 1, 2], [3, 3, 0]]),
        ("rep_2x4", [[2, 2, 2, 2], [2, 2, 2, 2]]),
        ("rep_4x2", [[1, 2], [1, 2], [1, 2], [1, 2]]),
        ("rep_3x3_frac", [[0.5, 0.5, 0.25], [0.5, 0.5, 0.25], [0.75, 0.75, 0.5]]),
    ]
    for name, mat in repeat_cases:
        cases.append((name, np.asarray(mat, dtype=np.float64)))

    # --- degenerate shapes -------------------------------------------------------------
    cases.append(("degen_1x1", np.asarray([[7.0]])))
    cases.append(("degen_1x6", np.asarray([[3.0, 1.0, 2.0, 9.0, 9.0, 0.5]])))
    cases.append(("degen_6x1", np.asarray([[3.0], [1.0], [2.0], [9.0], [9.0], [0.5]])))
    cases.append(("degen_1x1_zero", np.asarray([[0.0]])))
    cases.append(("degen_8x1_tie", np.asarray([[1.0]] * 8)))
    cases.append(("degen_1x8_tie", np.asarray([[1.0]] * 8)))

    # --- large-ish square matrix with heavy ties --------------------------------------
    rng = np.random.default_rng(SEED + 1)
    cases.append(("tie_random_8x8_binary", rng.integers(0, 2, size=(8, 8)).astype(np.float64)))
    cases.append(("tie_random_6x9_binary", rng.integers(0, 2, size=(6, 9)).astype(np.float64)))
    cases.append(("tie_random_5x5_int", rng.integers(0, 4, size=(5, 5)).astype(np.float64)))

    return cases


def random_cases(n: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """Seed-fixed random matrices, cycling through shapes and value distributions."""
    rng = np.random.default_rng(seed)
    out: list[tuple[str, np.ndarray]] = []
    for idx in range(n):
        nr, nc = SHAPES[idx % len(SHAPES)]
        # Randomise the degenerate dimensions a little so 1xN / Nx1 get varying lengths.
        if nr == 1:
            nc = int(rng.integers(1, 12))
        elif nc == 1:
            nr = int(rng.integers(1, 12))

        mode = idx % 8
        if mode == 0:  # uniform [0, 1)
            mat = rng.random((nr, nc))
        elif mode == 1:  # uniform [0, 100)
            mat = rng.random((nr, nc)) * 100.0
        elif mode == 2:  # small integers -> many ties
            mat = rng.integers(0, 5, size=(nr, nc)).astype(np.float64)
        elif mode == 3:  # binary -> maximal tie multiplicity
            mat = rng.integers(0, 2, size=(nr, nc)).astype(np.float64)
        elif mode == 4:  # signed
            mat = rng.random((nr, nc)) * 20.0 - 10.0
        elif mode == 5:  # tiny magnitudes: float32 rounding stress
            mat = rng.random((nr, nc)) * 1e-3
        elif mode == 6:  # coarse quantisation
            mat = np.round(rng.random((nr, nc)) * 4.0) / 2.0
        else:  # mostly-equal matrix with a couple of perturbations
            mat = np.full((nr, nc), 0.25)
            flat = mat.reshape(-1)
            if flat.size:
                pos = rng.choice(flat.size, size=min(2, flat.size), replace=False)
                flat[pos] = rng.random(len(pos))

        out.append((f"rand_{idx:05d}_{nr}x{nc}_m{mode}", mat))
    return out


def large_cases(n: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """Seed-fixed *larger* matrices (up to 39x39) to exercise the dual-variable updates and the
    tie-breaking on problems that are big enough for the shortest-augmenting-path search to loop
    several times per row."""
    rng = np.random.default_rng(seed)
    out: list[tuple[str, np.ndarray]] = []
    for idx in range(n):
        nr = int(rng.integers(1, 40))
        nc = int(rng.integers(1, 40))
        mode = idx % 5
        if mode == 0:
            mat = rng.random((nr, nc))
        elif mode == 1:
            mat = rng.integers(0, 3, size=(nr, nc)).astype(np.float64)
        elif mode == 2:
            mat = rng.integers(0, 2, size=(nr, nc)).astype(np.float64)
        elif mode == 3:
            mat = np.round(rng.random((nr, nc)) * 3.0)
        else:
            mat = rng.random((nr, nc)) * 1e-4
        out.append((f"large_{idx:05d}_{nr}x{nc}_m{mode}", mat))
    return out


def error_parity_cases() -> list[tuple[str, np.ndarray]]:
    """Matrices SciPy rejects; the reference must reject them the same way."""
    return [
        ("err_nan", np.array([[1.0, np.nan], [2.0, 3.0]])),
        ("err_neginf", np.array([[1.0, -np.inf], [2.0, 3.0]])),
        ("err_all_posinf", np.array([[np.inf, np.inf], [np.inf, np.inf]])),
    ]


def _pairs(row_ind: np.ndarray, col_ind: np.ndarray) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in zip(np.asarray(row_ind), np.asarray(col_ind)))


def _total(cost64: np.ndarray, row_ind: np.ndarray, col_ind: np.ndarray) -> float:
    return math.fsum(
        float(cost64[int(r), int(c)]) for r, c in zip(np.asarray(row_ind), np.asarray(col_ind))
    )


def compare_one(name: str, cost: np.ndarray) -> dict | None:
    """Compare reference vs SciPy on one matrix.

    Returns ``None`` on agreement, otherwise a dict describing the mismatch.
    """
    cost64 = np.ascontiguousarray(cost, dtype=np.float64)

    scipy_rows, scipy_cols = scipy_lsa(cost)
    ref_rows, ref_cols = linear_sum_assignment_ref(cost)

    scipy_pairs = _pairs(scipy_rows, scipy_cols)
    ref_pairs = _pairs(ref_rows, ref_cols)

    scipy_cost = _total(cost64, scipy_rows, scipy_cols)
    ref_cost = _total(cost64, ref_rows, ref_cols)

    pairs_ok = scipy_pairs == ref_pairs
    cost_ok = scipy_cost == ref_cost
    dtype_ok = np.asarray(ref_rows).dtype == np.int64 and np.asarray(ref_cols).dtype == np.int64

    if pairs_ok and cost_ok and dtype_ok:
        return None

    return {
        "name": name,
        "shape": list(cost.shape),
        "dtype": str(cost.dtype),
        "pairs_ok": pairs_ok,
        "cost_ok": cost_ok,
        "dtype_ok": dtype_ok,
        "scipy_rows": np.asarray(scipy_rows).tolist(),
        "scipy_cols": np.asarray(scipy_cols).tolist(),
        "ref_rows": np.asarray(ref_rows).tolist(),
        "ref_cols": np.asarray(ref_cols).tolist(),
        "scipy_total": repr(scipy_cost),
        "ref_total": repr(ref_cost),
        "cost_delta": repr(ref_cost - scipy_cost),
        "matrix": cost64,
    }


def main() -> int:
    print("=== LSAP reference differential check ===")
    print(f"scipy version : {scipy.__version__}")
    print(f"numpy version : {np.__version__}")
    print(f"seed          : {SEED}")
    print(f"random cases  : {N_RANDOM} matrices x 2 dtypes (float32 + float64)")
    print(f"large cases   : {N_LARGE} matrices (up to 39x39) x 2 dtypes")
    print()

    mismatches: list[dict] = []
    total_cases = 0
    total_dets = 0

    def run_suite(cases: list[tuple[str, np.ndarray]], dtypes: tuple, label: str) -> None:
        nonlocal total_cases, total_dets
        n_label = len(cases) * len(dtypes)
        print(f"--- {label} ({len(cases)} matrices x {len(dtypes)} dtypes = {n_label} cases) ---")
        for name, mat in cases:
            for dtype in dtypes:
                total_cases += 1
                cost = mat.astype(dtype)
                total_dets += int(mat.shape[0]) * int(mat.shape[1])
                info = compare_one(name, cost)
                if info is not None:
                    info["case_dtype"] = str(dtype)
                    mismatches.append(info)
        print(f"    running ... {len(mismatches)} mismatch(es) so far")

    run_suite(fixed_cases(), (np.float32, np.float64), "fixed cases")
    run_suite(random_cases(N_RANDOM, SEED), (np.float32, np.float64), "random cases")
    run_suite(large_cases(N_LARGE, SEED + 7), (np.float32, np.float64), "large cases")

    # --- error parity -------------------------------------------------------------------
    print("--- error parity (scipy rejects, reference must reject identically) ---")
    err_ok = True
    for name, mat in error_parity_cases():
        scipy_exc = ref_exc = None
        try:
            scipy_lsa(mat)
        except Exception as exc:  # noqa: BLE001
            scipy_exc = f"{type(exc).__name__}: {exc}"
        try:
            linear_sum_assignment_ref(mat)
        except Exception as exc:  # noqa: BLE001
            ref_exc = f"{type(exc).__name__}: {exc}"
        same = scipy_exc == ref_exc
        err_ok = err_ok and same
        print(f"    {name:22s} scipy={scipy_exc!s:60s} ref={ref_exc!s:60s} {'OK' if same else 'MISMATCH'}")
    if not err_ok:
        mismatches.append({"name": "error_parity", "pairs_ok": False, "cost_ok": False})

    # --- report --------------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"total comparison cases : {total_cases}")
    print(f"total matrix cells     : {total_dets}")
    print(f"mismatched cases       : {len(mismatches)}")
    rate = 100.0 * (total_cases - len(mismatches)) / total_cases if total_cases else 0.0
    print(f"consistency rate       : {rate:.6f}%")
    print("=" * 72)

    if mismatches:
        first = mismatches[0]
        print("\n### FIRST MISMATCH ###")
        print(f"case name  : {first.get('name')}")
        print(f"input dtype: {first.get('dtype')}  (case dtype {first.get('case_dtype')})")
        print(f"shape      : {first.get('shape')}")
        print(f"pairs_ok   : {first.get('pairs_ok')}")
        print(f"cost_ok    : {first.get('cost_ok')}")
        mat = first.get("matrix")
        if mat is not None:
            print("matrix:")
            print(np.array2string(np.asarray(mat), precision=17, max_line_width=200))
        print(f"scipy rows : {first.get('scipy_rows')}")
        print(f"scipy cols : {first.get('scipy_cols')}")
        print(f"ref   rows : {first.get('ref_rows')}")
        print(f"ref   cols : {first.get('ref_cols')}")
        print(f"scipy total: {first.get('scipy_total')}")
        print(f"ref   total: {first.get('ref_total')}")
        print(f"cost delta : {first.get('cost_delta')}")

        np.save(MISMATCH_NPY, np.asarray(mat))
        MISMATCH_JSON.write_text(
            json.dumps(
                {k: v for k, v in first.items() if k != "matrix"},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nmatrix written to : {MISMATCH_NPY}")
        print(f"details written to: {MISMATCH_JSON}")
        print(f"\nRESULT: FAIL ({len(mismatches)} mismatching case(s))")
        return 1

    print("\nRESULT: PASS (100% agreement with scipy.optimize.linear_sum_assignment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
