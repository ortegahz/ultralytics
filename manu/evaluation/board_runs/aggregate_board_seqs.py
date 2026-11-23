#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Aggregate the BOARD-side fused-track TSVs of many sequences into one delivery report.

This is the 24-sequence counterpart of ``manu/evaluation/eval_board_fused_tsv.py``: the same
parser, the same matcher and the same metric definitions, but the per-sequence counts are
aggregated the way ``manu/evaluation/system_sota_longquan.py`` aggregates them, i.e. the COUNT
KEYS are summed over the sequences and ``metrics()`` is recomputed ONCE on the summed counts.
Per-sequence F1 values are never averaged.

Input (one file per sequence, the format written by ``mosaic_track_dump``)::

    <fused_dir>/<SEQ>_fused.tsv

    <frame_idx> <n> [<track_id1> <x1> <y1> <score1> <infilled1> ...]

``frame_idx`` is the PHYSICAL camera frame id (0-based source frame index), ``x``/``y`` are
centroids in native 640x512 pixels, ``score`` is the track score.  The ground truth is read
from the server-only label directory ``<labels_root>/<SEQ>__<frame6>.txt``.

Two calibers are supported, matching ``eval_board_fused_tsv.py``:

* ``all``      -- every camera frame of a sequence takes part; a frame that is absent from the
                  fused TSV counts as a MISS (its ground truth feeds the FN) and the FAR
                  denominator is the FULL sequence length.  This is the delivery-gate caliber.
* ``delivered``-- only the frames the board actually delivered take part; the FAR denominator
                  is the number of delivered frames (the jump-frame caliber).

Usage::

    python manu/evaluation/board_runs/aggregate_board_seqs.py \
        --fused-dir /mnt/data/siping/runs/board24/fused \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --caliber all --out /mnt/data/siping/runs/board24/board_24seq_all.json

Set ``ULTRALYTICS_ROOT`` to run the very same file from the server checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = os.environ.get(
    "ULTRALYTICS_ROOT", "/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics"
)
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    COUNT_KEYS,
    count_matches,
    load_ground_truth,
    metrics,
)


def parse_fused(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    """Parse ``<frame_idx> <n> <track_id> <x> <y> <score> <infilled> ...`` into centroids+scores."""
    out: dict[int, list[tuple[float, float, float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: expected at least '<frame_idx> <n>', got {line!r}")
            frame_idx = int(parts[0])
            count = int(parts[1])
            values = parts[2:]
            if len(values) != 5 * count:
                raise ValueError(
                    f"{path}:{lineno}: frame {frame_idx} declares {count} detections "
                    f"but carries {len(values)} values (expected {5 * count})"
                )
            dets = []
            for k in range(count):
                base = 5 * k
                x = float(values[base + 1])
                y = float(values[base + 2])
                score = float(values[base + 3])
                dets.append((x, y, score))
            out[frame_idx] = dets
    return out


def _frame_entry(
    truth: dict[int, np.ndarray], fused: dict[int, list[tuple[float, float, float]]], frame_id: int
) -> tuple:
    """Build the ``(boxes, points, scores)`` triple of one frame, sorted by descending score."""
    dets = fused.get(frame_id)
    if dets:
        ranked = sorted(dets, key=lambda item: -item[2])
        points = np.asarray([(d[0], d[1]) for d in ranked], dtype=np.float32)
        scores = np.asarray([d[2] for d in ranked], dtype=np.float32)
    else:
        points, scores = None, None
    return truth.get(frame_id), points, scores


def score_sequence(
    fused: dict[int, list[tuple[float, float, float]]],
    truth: dict[int, np.ndarray],
    total_frames: int,
) -> dict:
    """Return the ``all`` and ``delivered`` counts/metrics of one board sequence.

    ``all`` iterates ``range(total_frames)``; a frame that carries no detection counts as a MISS
    so its ground truth still feeds the FN denominator, and the FAR denominator is
    ``total_frames``.  ``delivered`` iterates only the frame ids present in the fused TSV and the
    FAR denominator is the number of those delivered frames.
    """
    all_frames = [_frame_entry(truth, fused, frame_id) for frame_id in range(total_frames)]
    counts_all = count_matches(all_frames, 0.0, 8.0, "inbox")
    metrics_all = metrics(counts_all, total_frames)

    delivered_ids = sorted(fused)
    delivered_frames = [_frame_entry(truth, fused, frame_id) for frame_id in delivered_ids]
    counts_delivered = count_matches(delivered_frames, 0.0, 8.0, "inbox")
    metrics_delivered = metrics(counts_delivered, len(delivered_ids))

    return {
        "total_frames": total_frames,
        "delivered_frames": len(delivered_ids),
        "delivered_frames_with_output": sum(1 for f in delivered_ids if fused.get(f)),
        "fused_frames": len(fused),
        "gt_frames": len(truth),
        "all": {"counts": counts_all, "metrics": metrics_all},
        "delivered": {"counts": counts_delivered, "metrics": metrics_delivered},
    }


def aggregate_counts(rows: list[dict], caliber: str) -> dict:
    """Sum the COUNT KEYS over the per-sequence rows (never average per-sequence F1 values)."""
    return {key: int(sum(row[caliber]["counts"][key] for row in rows)) for key in COUNT_KEYS}


def discover_sequences(fused_dir: Path) -> list[str]:
    """Auto-discover the sequences by listing the ``<SEQ>_fused.tsv`` files of ``fused_dir``."""
    return sorted(path.name[: -len("_fused.tsv")] for path in fused_dir.glob("*_fused.tsv"))


def load_total_frames(path: Path | None) -> dict[str, int]:
    """Load the optional ``{sequence: total_frames}`` JSON override."""
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object mapping sequence -> total frames")
    return {str(key): int(value) for key, value in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-dir", required=True, type=Path,
                        help="Directory holding the per-sequence <SEQ>_fused.tsv files")
    parser.add_argument("--labels-root", required=True, type=Path,
                        help="Flat Final_Labels directory of {sequence}__{index:06d}.txt")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Sequences to aggregate; default = every <SEQ>_fused.tsv of --fused-dir")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--caliber", choices=["all", "delivered"], default="all",
                        help="'all' = every camera frame (FAR denominator = full sequence length); "
                             "'delivered' = only the frames the board delivered")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--total-frames-json", type=Path, default=None,
                        help="Optional JSON mapping sequence -> full sequence length; overrides the "
                             "max(frame id)+1 / max(GT id)+1 default")
    args = parser.parse_args()

    if not args.fused_dir.is_dir():
        raise SystemExit(f"[ERR] --fused-dir is not a directory: {args.fused_dir}")
    if not args.labels_root.is_dir():
        raise SystemExit(f"[ERR] --labels-root is not a directory: {args.labels_root}")

    sequences = args.sequences if args.sequences else discover_sequences(args.fused_dir)
    if not sequences:
        raise SystemExit(f"[ERR] no sequence found in {args.fused_dir} (expected <SEQ>_fused.tsv)")

    overrides = load_total_frames(args.total_frames_json)

    rows: list[dict] = []
    for sequence in sequences:
        fused_tsv = args.fused_dir / f"{sequence}_fused.tsv"
        if not fused_tsv.is_file():
            raise SystemExit(f"[ERR] missing fused TSV for {sequence}: {fused_tsv}")
        fused = parse_fused(fused_tsv)
        truth = load_ground_truth(args.labels_root, sequence, args.width, args.height)

        total = overrides.get(sequence, 0)
        if total <= 0:
            total = max([0, *fused.keys(), *truth.keys()]) + 1

        row = score_sequence(fused, truth, total)
        row["sequence"] = sequence
        row["fused_tsv"] = str(fused_tsv)
        rows.append(row)

        m = row[args.caliber]["metrics"]
        print(
            f"[SEQ] {sequence} total_frames={row['total_frames']} "
            f"delivered={row['delivered_frames']} (with output {row['delivered_frames_with_output']}) "
            f"gt_frames={row['gt_frames']} fused_frames={row['fused_frames']} "
            f"tp={m['tp']} fp={m['fp']} fn={m['fn']} f1={m['f1']:.7f} far={m['far_per_frame']:.7f}",
            flush=True,
        )

    # Global aggregation: sum the COUNT KEYS over the sequences, then recompute the metrics ONCE.
    global_counts = aggregate_counts(rows, args.caliber)
    if args.caliber == "all":
        global_frames = sum(row["total_frames"] for row in rows)
    else:
        global_frames = sum(row["delivered_frames"] for row in rows)
    global_metrics = metrics(global_counts, global_frames)

    print("")
    print(f"{'sequence':<28} {'tp':>7} {'fp':>7} {'fn':>7} {'f1':>12} {'far':>12}")
    for row in rows:
        m = row[args.caliber]["metrics"]
        print(f"{row['sequence']:<28} {m['tp']:>7} {m['fp']:>7} {m['fn']:>7} "
              f"{m['f1']:>12.7f} {m['far_per_frame']:>12.7f}")
    print("-" * 74)
    print(f"{'GLOBAL(' + args.caliber + ')':<28} {global_metrics['tp']:>7} {global_metrics['fp']:>7} "
          f"{global_metrics['fn']:>7} {global_metrics['f1']:>12.7f} "
          f"{global_metrics['far_per_frame']:>12.7f}")
    print(f"[GLOBAL] caliber={args.caliber} frames={global_frames} gt={global_metrics['gt']} "
          f"recall={global_metrics['recall']:.6f} precision={global_metrics['precision']:.6f} "
          f"fn_absent={global_metrics['fn_absent']} fn_compete={global_metrics['fn_compete']} "
          f"det={global_metrics['det']}")
    print(f"[COUNTS] {global_counts}")

    if args.out:
        payload = {
            "caliber": args.caliber,
            "labels_root": str(args.labels_root),
            "fused_dir": str(args.fused_dir),
            "width": args.width,
            "height": args.height,
            "sequences": [row["sequence"] for row in rows],
            "per_sequence": [
                {
                    "sequence": row["sequence"],
                    "fused_tsv": row["fused_tsv"],
                    "total_frames": row["total_frames"],
                    "delivered_frames": row["delivered_frames"],
                    "delivered_frames_with_output": row["delivered_frames_with_output"],
                    "fused_frames": row["fused_frames"],
                    "gt_frames": row["gt_frames"],
                    "counts": row[args.caliber]["counts"],
                    "metrics": row[args.caliber]["metrics"],
                }
                for row in rows
            ],
            "total_frames": global_frames,
            "global_counts": global_counts,
            "global_metrics": global_metrics,
            "calibers": {
                caliber: {
                    "total_frames": (
                        sum(row["total_frames"] for row in rows) if caliber == "all"
                        else sum(row["delivered_frames"] for row in rows)
                    ),
                    "global_counts": aggregate_counts(rows, caliber),
                    "global_metrics": metrics(
                        aggregate_counts(rows, caliber),
                        sum(row["total_frames"] for row in rows) if caliber == "all"
                        else sum(row["delivered_frames"] for row in rows),
                    ),
                }
                for caliber in ("all", "delivered")
            },
        }
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OUT] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
