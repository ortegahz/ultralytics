#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Trajectory-level parity between the BOARD fused output and the SERVER reference.

``board_vs_server_24seq.py`` compares *counts* (tp/fp/fn/F1/FAR) that both sides score
against the same ground truth.  This script answers the complementary question: **do the two
sides point at the same pixels?**

For every board detection the nearest server detection *in the same frame* is looked up and the
Euclidean distance is recorded.  A detection counts as ``matched`` when that distance is within
``--tol`` (8 px by default, the same tolerance the shared ground-truth matcher uses).

Both sides are read through the parsers of ``board_far_parity_check`` so the board TSV and the
server JSON are decoded with exactly the same rules used by the F1/FAR reconciliation.

Usage::

    python board_traj_parity.py \
        --board-tsv-dir  /mnt/data/siping/runs/trial0080_board24_20261120/tsv \
        --server-preds-dir /mnt/data/siping/runs/trial0080_onnx/eval/shortlist/preds_cc2_md10 \
        --out /mnt/data/siping/runs/trial0080_board24_20261120/board_traj_parity.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ULTRALYTICS_ROOT = os.environ.get(
    "ULTRALYTICS_ROOT", "/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics"
)
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.board_far_parity_check import (  # noqa: E402
    parse_board_tsv,
    parse_server_preds,
)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (no numpy dependency on the board side)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(q / 100.0 * len(ordered) + 0.5))))
    return ordered[rank - 1]


def compare_seq(
    board: dict[int, list[tuple[float, float, float]]],
    server: dict[int, list[tuple[float, float, float]]],
    tol: float,
) -> dict:
    """Nearest-neighbour comparison of every board detection against the server frame."""
    matched: list[float] = []
    unmatched = 0
    total = 0
    per_frame = []
    for frame_idx in sorted(board):
        dets = board[frame_idx]
        if not dets:
            continue
        ref = server.get(frame_idx, [])
        total += len(dets)
        frame_matched = 0
        for x, y, _score in dets:
            if ref:
                nearest = min((x - rx) ** 2 + (y - ry) ** 2 for rx, ry, _ in ref) ** 0.5
                if nearest <= tol:
                    matched.append(nearest)
                    frame_matched += 1
                    continue
            unmatched += 1
        per_frame.append((frame_idx, len(dets), frame_matched))
    result = {
        "det": total,
        "matched": len(matched),
        "unmatched": unmatched,
        "match_rate": (len(matched) / total) if total else float("nan"),
        "tol": tol,
    }
    if matched:
        result.update({
            "dist_mean": statistics.fmean(matched),
            "dist_p50": percentile(matched, 50),
            "dist_p95": percentile(matched, 95),
            "dist_max": max(matched),
        })
    else:
        result.update({"dist_mean": float("nan"), "dist_p50": float("nan"),
                       "dist_p95": float("nan"), "dist_max": float("nan")})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-tsv-dir", required=True, type=Path)
    parser.add_argument("--server-preds-dir", required=True, type=Path)
    parser.add_argument("--tol", type=float, default=8.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    seqs = sorted(p.name[: -len("_fused.tsv")] for p in args.board_tsv_dir.glob("*_fused.tsv"))
    if not seqs:
        print(f"[ERR] no <SEQ>_fused.tsv under {args.board_tsv_dir}")
        return 1

    rows = []
    for seq in seqs:
        server_json = args.server_preds_dir / f"{seq}.json"
        if not server_json.is_file():
            print(f"[ERR] missing server reference for {seq}: {server_json}")
            return 1
        board = parse_board_tsv(args.board_tsv_dir / f"{seq}_fused.tsv")
        server = parse_server_preds(server_json)
        row = compare_seq(board, server, args.tol)
        row["sequence"] = seq
        rows.append(row)
        print(f"[TRAJ] {seq:<30} matched={row['matched']}/{row['det']} "
              f"({row['match_rate'] * 100:.2f}%)  dist mean={row['dist_mean']:.3f} "
              f"p95={row['dist_p95']:.3f} max={row['dist_max']:.3f}")

    det = sum(r["det"] for r in rows)
    matched = sum(r["matched"] for r in rows)
    payload = {
        "tol": args.tol,
        "board_tsv_dir": str(args.board_tsv_dir),
        "server_preds_dir": str(args.server_preds_dir),
        "sequences": rows,
        "global": {
            "det": det,
            "matched": matched,
            "unmatched": det - matched,
            "match_rate": (matched / det) if det else float("nan"),
        },
    }
    print(f"[GLOBAL] matched={matched}/{det} "
          f"({payload['global']['match_rate'] * 100:.2f}%)")
    if args.out:
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OUT] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
