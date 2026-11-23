#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Board-vs-server FAR/F1 parity check on one real trial_0080 sequence.

The board runs the *mosaic* chain (RK3588 NPU -> mosaic candidates -> ``mosaic_track_fusion``);
the server publishes the *multiscale* chain predictions for the very same sequence under the
delivered parameter group ``cc2_md10``.  This script scores BOTH sides against the SAME
ground truth and the SAME frame window, so the two numbers are directly comparable.

Inputs
------
``--board-tsv``
    Track dump written by the board (``mosaic_track_dump`` / the standalone align engine)::

        <frame_idx> <n> [<track_id> <x> <y> <score> <infilled>] x n

    ``frame_idx`` is the physical camera frame id.  Empty frames may be omitted (the ``.so``
    fusion path only writes non-empty frames) -- missing frames are treated as "no detection"
    and therefore contribute to the false-negative denominator.
``--server-preds``
    The server JSON produced by the delivery evaluation::

        {"config": {...}, "metrics": {...}, "per_frame": {"<frame_idx>": [{"pos": [x, y],
         "score": s, "track_id": t, "infilled": false}, ...]}}

``--labels-root``
    Directory holding ``<sequence>__<index:06d>.txt`` ground-truth boxes.

Usage
-----
    python manu/evaluation/board_far_parity_check.py \
        --board-tsv  /root/align_dn_20261120/board_fused_video07_001324.tsv \
        --server-preds /tmp/boardfar/preds_cc2_md10_VIDEO00007.json \
        --labels-root /tmp/boardfar/gt \
        --sequence VIDEO00007_19700101_001324 \
        --window 0 2500 --out /tmp/boardfar/parity.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = str(Path(__file__).resolve().parents[2])
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    count_matches,
    load_ground_truth,
    metrics,
)


def parse_board_tsv(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    """Parse the board track dump into ``{frame_idx: [(x, y, score), ...]}``."""
    out: dict[int, list[tuple[float, float, float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: expected '<frame_idx> <n>', got {line!r}")
            frame_idx = int(parts[0])
            count = int(parts[1])
            values = parts[2:]
            if len(values) != 5 * count:
                raise ValueError(
                    f"{path}:{lineno}: frame {frame_idx} declares {count} tracks "
                    f"but carries {len(values)} values (expected {5 * count})"
                )
            dets = []
            for k in range(count):
                base = 5 * k
                dets.append((float(values[base + 1]), float(values[base + 2]), float(values[base + 3])))
            out[frame_idx] = dets
    return out


def parse_server_preds(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    """Parse the server per-frame predictions into ``{frame_idx: [(x, y, score), ...]}``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, list[tuple[float, float, float]]] = {}
    for key, entries in payload["per_frame"].items():
        dets = []
        for entry in entries:
            pos = entry["pos"]
            dets.append((float(pos[0]), float(pos[1]), float(entry["score"])))
        out[int(key)] = dets
    return out


def score_side(
    fused: dict[int, list[tuple[float, float, float]]],
    truth: dict[int, np.ndarray],
    window: tuple[int, int],
    tol: float,
) -> dict:
    """Score one side over the frame window with the shared matcher."""
    start, end = window
    frames = []
    for frame_id in range(start, end):
        boxes = truth.get(frame_id)
        dets = fused.get(frame_id)
        if dets:
            ranked = sorted(dets, key=lambda item: -item[2])
            points = np.asarray([(d[0], d[1]) for d in ranked], dtype=np.float32)
            scores = np.asarray([d[2] for d in ranked], dtype=np.float32)
        else:
            points, scores = None, None
        frames.append((boxes, points, scores))
    counts = count_matches(frames, 0.0, tol, "inbox")
    result = metrics(counts, len(frames))
    result["window"] = [start, end]
    result["frames"] = len(frames)
    result["frames_with_detection"] = sum(1 for f in range(start, end) if fused.get(f))
    return result


def compare_tracks(
    board: dict[int, list[tuple[float, float, float]]],
    server: dict[int, list[tuple[float, float, float]]],
    window: tuple[int, int],
    max_dist: float,
) -> dict:
    """Greedy nearest-neighbour agreement between the two track sets, per frame."""
    start, end = window
    d_pos: list[float] = []
    d_score: list[float] = []
    matched = 0
    board_total = 0
    server_total = 0
    for frame_id in range(start, end):
        b = sorted(board.get(frame_id, []), key=lambda item: -item[2])
        s = sorted(server.get(frame_id, []), key=lambda item: -item[2])
        board_total += len(b)
        server_total += len(s)
        if not b or not s:
            continue
        used = set()
        for bx, by, bs in b:
            best, best_d = None, None
            for idx, (sx, sy, ss) in enumerate(s):
                if idx in used:
                    continue
                dist = float(np.hypot(bx - sx, by - sy))
                if best_d is None or dist < best_d:
                    best, best_d = idx, dist
            if best is not None and best_d is not None and best_d <= max_dist:
                used.add(best)
                matched += 1
                d_pos.append(best_d)
                d_score.append(abs(bs - s[best][2]))
    return {
        "board_tracks": board_total,
        "server_tracks": server_total,
        "matched": matched,
        "match_rate_vs_board": matched / board_total if board_total else 0.0,
        "match_rate_vs_server": matched / server_total if server_total else 0.0,
        "pos_delta_mean": statistics.fmean(d_pos) if d_pos else None,
        "pos_delta_p95": float(np.percentile(d_pos, 95)) if d_pos else None,
        "pos_delta_max": max(d_pos) if d_pos else None,
        "score_delta_mean": statistics.fmean(d_score) if d_score else None,
        "score_delta_p95": float(np.percentile(d_score, 95)) if d_score else None,
        "score_delta_max": max(d_score) if d_score else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-tsv", required=True, type=Path)
    parser.add_argument("--server-preds", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--window", type=int, nargs=2, default=(0, 0),
                        help="[start, end) camera frame ids; end=0 means 'use the server window'")
    parser.add_argument("--tol", type=float, default=8.0)
    parser.add_argument("--match-dist", type=float, default=8.0,
                        help="Max centroid distance for two tracks to count as the same target")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    board = parse_board_tsv(args.board_tsv)
    server = parse_server_preds(args.server_preds)
    truth = load_ground_truth(args.labels_root, args.sequence, args.width, args.height)

    if args.window[1] <= 0:
        common = sorted(set(board) & set(server))
        window = (min(common), max(common) + 1)
    else:
        window = (args.window[0], args.window[1])

    board_m = score_side(board, truth, window, args.tol)
    server_m = score_side(server, truth, window, args.tol)
    agreement = compare_tracks(board, server, window, args.match_dist)

    print(f"[SEQ] {args.sequence} window={window[0]}..{window[1] - 1} "
          f"({window[1] - window[0]} frames)  gt_frames_in_window={board_m['gt']}")
    for name, m in (("BOARD ", board_m), ("SERVER", server_m)):
        print(f"[{name}] tp={m['tp']} fp={m['fp']} fn={m['fn']} gt={m['gt']} "
              f"recall={m['recall']:.6f} precision={m['precision']:.6f} f1={m['f1']:.7f} "
              f"far_per_frame={m['far_per_frame']:.7f}")
    print(f"[DELTA] f1={board_m['f1'] - server_m['f1']:+.7f}  "
          f"far={board_m['far_per_frame'] - server_m['far_per_frame']:+.7f}  "
          f"tp={board_m['tp'] - server_m['tp']:+d}  fp={board_m['fp'] - server_m['fp']:+d}  "
          f"fn={board_m['fn'] - server_m['fn']:+d}")
    print(f"[AGREE] matched={agreement['matched']}/{agreement['board_tracks']} board tracks "
          f"({agreement['match_rate_vs_board'] * 100:.3f}%); "
          f"pos_delta mean={agreement['pos_delta_mean']} p95={agreement['pos_delta_p95']} "
          f"max={agreement['pos_delta_max']}; "
          f"score_delta mean={agreement['score_delta_mean']} p95={agreement['score_delta_p95']}")

    result = {
        "sequence": args.sequence,
        "board_tsv": str(args.board_tsv),
        "server_preds": str(args.server_preds),
        "labels_root": str(args.labels_root),
        "tol": args.tol,
        "window": list(window),
        "board": board_m,
        "server": server_m,
        "delta": {
            "f1": board_m["f1"] - server_m["f1"],
            "far_per_frame": board_m["far_per_frame"] - server_m["far_per_frame"],
            "tp": board_m["tp"] - server_m["tp"],
            "fp": board_m["fp"] - server_m["fp"],
            "fn": board_m["fn"] - server_m["fn"],
        },
        "agreement": agreement,
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OUT] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
