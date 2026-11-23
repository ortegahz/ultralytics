#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Quantify the **threshold edge** that separates the board's detections from the server's.

The cross-feed experiment (``board_crossfeed_verify.py``) shows that the board's fused TSV is
reproduced bit for bit by the server's Python mirror when both are fed the **same** candidate
stream.  What is left is therefore the candidate stream itself: the board runs the RKNN in
**FP16** over an **RGA**-preprocessed frame, the server runs its own heat map in FP32 over an
**OpenCV**-preprocessed frame.

This probe measures that difference directly, on the raw (pre-threshold) candidate pools:

  * for every ground-truth target and every frame, the **best board peak** within
    ``--match-radius`` px of the target and the **best server peak** within the same radius;
  * the distribution of ``board_peak - server_peak`` (which side is more sensitive);
  * the targets whose **threshold decision flips** between the two sides, and how close to
    the threshold they sit (the "edge" effect);
  * the counterfactual: how many targets a threshold lowered by 0.01 / 0.02 / 0.05 would
    bring back on the board.

Usage
-----
    python manu/evaluation/board_fp16_edge_probe.py \
        --tsv-dir /mnt/data/siping/runs/trial0080_board24_20261120/tsv \
        --cache-root /mnt/data/siping/runs/trial0080_onnx/ms_cache \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --sequences VIDEO00007_19700101_001324 \
        --out /tmp/crossfeed/edge_probe.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = str(Path(__file__).resolve().parents[2])
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402


def load_board_cands(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read the board's raw candidate pool: ``{frame: (points (n,2) float32, peaks (n,) float32)}``."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            frame = int(parts[0])
            count = int(parts[1])
            values = parts[2:]
            points = np.zeros((count, 2), dtype=np.float32)
            peaks = np.zeros((count,), dtype=np.float32)
            for k in range(count):
                points[k, 0] = float(values[3 * k + 0])
                points[k, 1] = float(values[3 * k + 1])
                peaks[k] = float(values[3 * k + 2])
            out[frame] = (points, peaks)
    return out


def load_cache(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read the server's cached multi-scale pool: ``{frame: (points, scores)}``."""
    cache = pickle.loads(path.read_bytes())
    offset = cache["offset"]
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for position, frame_index in enumerate(cache["frame_index"]):
        start, end = int(offset[position]), int(offset[position + 1])
        out[int(frame_index)] = (
            np.asarray(cache["points"][start:end], dtype=np.float32),
            np.asarray(cache["scores"][start:end], dtype=np.float32),
        )
    return out


def best_near(
    pool: tuple[np.ndarray, np.ndarray] | None, x: float, y: float, radius: float
) -> tuple[float, float] | None:
    """Return ``(peak, distance)`` of the strongest candidate within ``radius`` px, or ``None``."""
    if pool is None:
        return None
    points, peaks = pool
    if not points.shape[0]:
        return None
    dist = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
    inside = dist <= radius
    if not inside.any():
        return None
    sel = np.nonzero(inside)[0]
    best = sel[int(np.argmax(peaks[sel]))]
    return float(peaks[best]), float(dist[best])


def describe(values: list[float]) -> dict:
    """Compact description of a numeric sample."""
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv-dir", required=True, type=Path, help="the board's TSV artifact root")
    parser.add_argument("--cache-root", required=True, type=Path, help="the server's multi-scale cache root")
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--sequences", required=True, help="comma separated")
    parser.add_argument("--th-base", type=float, default=0.48)
    parser.add_argument("--th-ground", type=float, default=0.26)
    parser.add_argument("--th-salvage", type=float, default=0.06)
    parser.add_argument("--sky-ratio", type=float, default=1.0)
    parser.add_argument("--match-radius", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    sky_boundary = args.height * args.sky_ratio

    rows = []
    for sequence in sequences:
        board = load_board_cands(args.tsv_dir / f"{sequence}_cands.tsv")
        server = load_cache(args.cache_root / f"{sequence}.pkl")
        truth = load_ground_truth(args.labels_root, sequence, args.width, args.height)

        deltas: list[float] = []
        flip_board_below: list[float] = []
        flip_server_below: list[float] = []
        board_miss_margin: list[float] = []
        server_miss_margin: list[float] = []
        n_targets = 0
        n_both = 0
        n_board_only = 0
        n_server_only = 0
        n_neither = 0
        n_flips = 0
        board_miss_server_hit = 0
        server_miss_board_hit = 0
        recover = {0.01: 0, 0.02: 0, 0.05: 0}

        for frame, boxes in truth.items():
            board_pool = board.get(frame)
            server_pool = server.get(frame)
            for row in boxes:
                x, y = float(row[0]), float(row[1])
                tau = args.th_base if y < sky_boundary else args.th_ground
                n_targets += 1
                hit_board = best_near(board_pool, x, y, args.match_radius)
                hit_server = best_near(server_pool, x, y, args.match_radius)
                board_peak = hit_board[0] if hit_board else None
                server_peak = hit_server[0] if hit_server else None

                board_pass = board_peak is not None and board_peak >= tau
                server_pass = server_peak is not None and server_peak >= tau

                if board_peak is not None and server_peak is not None:
                    n_both += 1
                    deltas.append(board_peak - server_peak)
                elif board_peak is not None:
                    n_board_only += 1
                elif server_peak is not None:
                    n_server_only += 1
                else:
                    n_neither += 1

                if board_pass != server_pass:
                    n_flips += 1
                    if board_pass and server_peak is not None:
                        flip_server_below.append(server_peak - tau)
                    if server_pass and board_peak is not None:
                        flip_board_below.append(tau - board_peak)

                if server_pass and not board_pass:
                    board_miss_server_hit += 1
                    if board_peak is not None:
                        board_miss_margin.append(tau - board_peak)
                        for delta in recover:
                            if tau - delta <= board_peak < tau:
                                recover[delta] += 1
                if board_pass and not server_pass:
                    server_miss_board_hit += 1
                    if server_peak is not None:
                        server_miss_margin.append(tau - server_peak)

        row = {
            "sequence": sequence,
            "gt_targets": n_targets,
            "targets_with_both_peaks": n_both,
            "targets_board_peak_only": n_board_only,
            "targets_server_peak_only": n_server_only,
            "targets_without_any_peak": n_neither,
            "peak_delta_board_minus_server": describe(deltas),
            "fraction_board_lower": float(np.mean(np.asarray(deltas) < 0)) if deltas else None,
            "threshold_flips": n_flips,
            "flip_margin_board_below_threshold": describe(flip_board_below),
            "flip_margin_server_below_threshold": describe(flip_server_below),
            "board_miss_server_hit": board_miss_server_hit,
            "board_miss_margin_to_threshold": describe(board_miss_margin),
            "server_miss_board_hit": server_miss_board_hit,
            "server_miss_margin_to_threshold": describe(server_miss_margin),
            "board_recoverable_by_threshold_lowering": {str(k): v for k, v in recover.items()},
        }
        rows.append(row)
        print(
            f"[{sequence}] targets={n_targets} both={n_both} "
            f"delta(med)={row['peak_delta_board_minus_server'].get('median', float('nan')):.6f} "
            f"board_lower={row['fraction_board_lower']:.4f} flips={n_flips} "
            f"board_miss={board_miss_server_hit} server_miss={server_miss_board_hit} "
            f"recover(-0.02)={recover[0.02]}"
        )

    payload = {
        "tsv_dir": str(args.tsv_dir),
        "cache_root": str(args.cache_root),
        "labels_root": str(args.labels_root),
        "th_base": args.th_base,
        "th_ground": args.th_ground,
        "sky_ratio": args.sky_ratio,
        "match_radius": args.match_radius,
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"[OUT] {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
