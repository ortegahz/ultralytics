#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Bit-compatible Python reference for ``scipy.optimize.linear_sum_assignment`` (scipy 1.15.3).

This module re-implements the rectangular linear-sum-assignment solver that SciPy ships as the
compiled extension ``scipy.optimize._lsap``.  The SciPy extension is a direct port of

    D. F. Crouse, "On implementing 2D rectangular assignment algorithms",
    IEEE Transactions on Aerospace and Electronic Systems 52(4):1679-1796, Aug. 2016,
    doi: 10.1109/TAES.2016.140952

and the C++ source of that port (``rectangular_lsap.cpp`` / ``rectangular_lsap.h``) is the
canonical specification for everything below.  The algorithm body is implemented here in pure
Python (NumPy is only used for array plumbing), so the C++ side of the time-space fusion port can
be diffed line-by-line against a *readable* reference instead of against SciPy's opaque binary.

Key SciPy behaviours reproduced verbatim
----------------------------------------
* SciPy's Python wrapper converts the input with ``PyArray_ContiguousFromAny(obj, NPY_DOUBLE, 0, 0)``
  and then rejects anything that is not 2-D.  **Every** cost matrix therefore enters the solver as
  C-contiguous ``float64``; a ``float32`` matrix is solved in ``float64`` precision.
* The solver transposes the matrix when ``nc < nr`` (tall matrices), solves the transposed problem and
  un-transposes the answer at the end.
* ``u`` / ``v`` (the dual variables) start at zero and are updated with the shortest-augmenting-path
  deltas.
* ``remaining`` is initialised in *reverse* order (``remaining[it] = nc - it - 1``) so that the solution
  of a constant cost matrix is the identity matrix (SciPy issue #11602).
* Ties are broken with ``shortestPathCosts[j] < lowest || (shortestPathCosts[j] == lowest &&
  row4col[j] == -1)`` -- i.e. among equal-cost columns an *unassigned* column wins.
* ``NaN`` and ``-inf`` entries are rejected with ``ValueError("matrix contains invalid numeric
  entries")``; an all-infinite (unreachable) row raises ``ValueError("cost matrix is infeasible")``.

All arithmetic is performed on IEEE-754 binary64 Python floats in exactly the same order as the C++
source, so the reference reproduces SciPy's *chosen* optimal assignment (not merely *an* optimal one)
bit-for-bit on tie-heavy inputs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["linear_sum_assignment_ref", "lsap_total_cost"]

_INF = float("inf")


def _augmenting_path(
    nc: int,
    cost: list[float],
    u: list[float],
    v: list[float],
    path: list[int],
    row4col: list[int],
    shortest_path_costs: list[float],
    i: int,
    sr: list[bool],
    sc: list[bool],
    remaining: list[int],
) -> tuple[int, float]:
    """Find a shortest augmenting path for row ``i``.

    Faithful port of ``augmenting_path()`` from SciPy's ``rectangular_lsap.cpp``.  ``cost`` is a flat
    row-major list of ``float`` (C-contiguous ``float64``); the remaining arguments are the mutable
    solver state and are updated in place.

    Returns ``(sink, min_val)``.  ``sink == -1`` means the matrix is infeasible (every remaining
    column is unreachable), which SciPy maps to ``ValueError("cost matrix is infeasible")``.
    """
    num_remaining = nc
    for it in range(nc):
        # Reverse order ensures that the solution of a constant cost matrix is the identity matrix
        # (SciPy issue #11602).
        remaining[it] = nc - it - 1

    for k in range(len(sr)):
        sr[k] = False
    for k in range(len(sc)):
        sc[k] = False
    for k in range(nc):
        shortest_path_costs[k] = _INF

    sink = -1
    min_val = 0.0
    while sink == -1:
        index = -1
        lowest = _INF
        sr[i] = True

        for it in range(num_remaining):
            j = remaining[it]

            r = min_val + cost[i * nc + j] - u[i] - v[j]
            if r < shortest_path_costs[j]:
                path[j] = i
                shortest_path_costs[j] = r

            # When multiple nodes have the minimum cost, select the one that gives us a new sink node.
            # This is particularly important for integer cost matrices with small coefficients.
            if shortest_path_costs[j] < lowest or (
                shortest_path_costs[j] == lowest and row4col[j] == -1
            ):
                lowest = shortest_path_costs[j]
                index = it

        min_val = lowest
        if min_val == _INF:  # infeasible cost matrix
            return -1, min_val

        j = remaining[index]
        if row4col[j] == -1:
            sink = j
        else:
            i = row4col[j]

        sc[j] = True
        num_remaining -= 1
        remaining[index] = remaining[num_remaining]

    return sink, min_val


def _solve(nr: int, nc: int, cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the (already ``float64``) rectangular assignment problem.

    Port of ``solve()`` from SciPy's ``rectangular_lsap.cpp`` with ``maximize=False`` (the SciPy
    wrapper negates the matrix itself before calling into C++).
    """
    if nr == 0 or nc == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Tall rectangular cost matrices must be transposed.
    transpose = nc < nr
    if transpose:
        # Row-major copy of the transpose, exactly like the C++ `temp[j * nr + i] = cost[i * nc + j]`.
        cost = np.ascontiguousarray(cost.T)
        nr, nc = nc, nr

    # Test for NaN and -inf entries (the C++ scan happens after the transposition).
    if np.isnan(cost).any() or np.isneginf(cost).any():
        raise ValueError("matrix contains invalid numeric entries")

    cost_flat = cost.reshape(-1).tolist()

    # Initialize variables.
    u = [0.0] * nr
    v = [0.0] * nc
    shortest_path_costs = [0.0] * nc
    path = [-1] * nc
    col4row = [-1] * nr
    row4col = [-1] * nc
    sr = [False] * nr
    sc = [False] * nc
    remaining = [0] * nc

    # Iteratively build the solution.
    for cur_row in range(nr):
        sink, min_val = _augmenting_path(
            nc,
            cost_flat,
            u,
            v,
            path,
            row4col,
            shortest_path_costs,
            cur_row,
            sr,
            sc,
            remaining,
        )
        if sink < 0:
            raise ValueError("cost matrix is infeasible")

        # Update dual variables.
        u[cur_row] += min_val
        for i in range(nr):
            if sr[i] and i != cur_row:
                u[i] += min_val - shortest_path_costs[col4row[i]]
        for j in range(nc):
            if sc[j]:
                v[j] -= min_val - shortest_path_costs[j]

        # Augment the previous solution.
        j = sink
        while True:
            i = path[j]
            row4col[j] = i
            col4row[i], j = j, col4row[i]
            if i == cur_row:
                break

    if transpose:
        # `col4row` is a permutation of the original row indices; sorting it yields the rows in
        # ascending order and the matching columns alongside.
        col4row_arr = np.asarray(col4row, dtype=np.int64)
        order = np.argsort(col4row_arr, kind="stable").astype(np.int64)
        a = col4row_arr[order]
        b = order
    else:
        a = np.arange(nr, dtype=np.int64)
        b = np.asarray(col4row, dtype=np.int64)

    return a, b


def linear_sum_assignment_ref(
    cost_matrix: np.ndarray, maximize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the linear sum assignment problem exactly like SciPy 1.15.3 does.

    Args:
        cost_matrix: 2-D array-like of shape ``(nr, nc)``.  Any dtype is accepted; the matrix is
            converted to C-contiguous ``float64`` first (this is what SciPy's wrapper does, so a
            ``float32`` input is solved in double precision).
        maximize: when ``True`` the matrix is negated before solving, i.e. the *maximum* total
            assignment is returned.  Mirrors SciPy's ``maximize`` keyword.

    Returns:
        ``(row_ind, col_ind)`` -- two ``int64`` arrays of length ``min(nr, nc)`` describing the
        optimal assignment.  ``cost_matrix[row_ind, col_ind].sum()`` is the optimal total cost.

    Raises:
        ValueError: if the input is not 2-D, contains ``NaN`` or ``-inf``, or is infeasible (every
            remaining column unreachable, e.g. an all-``+inf`` matrix).
    """
    arr = np.asarray(cost_matrix)
    if arr.ndim != 2:
        raise ValueError(f"expected a matrix (2-D array), got a {arr.ndim} array")

    # SciPy: np.PyArray_ContiguousFromAny(obj_cost, NPY_DOUBLE, 0, 0) -> C-contiguous float64.
    arr = np.ascontiguousarray(arr, dtype=np.float64)

    if maximize:
        # SciPy negates *before* the NaN/-inf validation, so `+inf` becomes an invalid entry.
        arr = -arr

    nr, nc = arr.shape
    if nr == 0 or nc == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return _solve(nr, nc, arr)


def lsap_total_cost(
    cost_matrix: np.ndarray, row_ind: np.ndarray, col_ind: np.ndarray
) -> float:
    """Total cost of an assignment, summed exactly (``math.fsum``) and order-independently.

    ``math.fsum`` returns the correctly rounded sum, so two assignments covering the same multiset
    of cost entries always produce the same value regardless of the summation order.
    """
    import math

    arr = np.asarray(cost_matrix, dtype=np.float64)
    return math.fsum(
        float(arr[int(r), int(c)]) for r, c in zip(np.asarray(row_ind), np.asarray(col_ind))
    )
