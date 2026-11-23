#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Score a BOARD-side fused-track TSV against the Longquan ground truth.

This is the board-side counterpart of ``system_sota_longquan.py``: same matcher, same
metric definitions, so the numbers are directly comparable with the server-side rows of
``track_all_v3_optgapfit.json``.

Input (the format written by ``mosaic_track_dump``)::

    <frame_idx> <n> [<track_id1> <x1> <y1> <score1> <infilled1> ...]

``frame_idx`` is the PHYSICAL camera frame id (0-based source frame index),
``x``/``y`` are centroids in native 640x512 pixels, ``score`` is the track score.

Two calibers are reported, matching ``rt_fps_jitter_search.py``:

* ``all``      -- every camera frame of the sequence takes part; the FAR denominator is the
                  FULL sequence length (this is the "per-frame" caliber of the delivery gate).
* ``delivered``-- only the frames the board actually delivered (the jump-frame caliber);
                  the FAR denominator is the number of delivered frames.

Usage::

    python eval_board_fused_tsv.py --fused-tsv /tmp/track_fused.tsv \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --sequence VIDEO00007_19700101_001324 --total-frames 11810 --out /tmp/board_fused_score.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = "/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics"
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
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


def score(fused: dict[int, list[tuple[float, float, float]]], truth: dict[int, np.ndarray],
          total_frames: int, delivered: set[int] | None = None) -> dict:
    """Return the ``all`` and ``delivered`` metrics of one board run.

    ``delivered`` is the set of camera frames the detector actually delivered (the
    jump-frame caliber of ``rt_fps_jitter_search.py``).  ``None`` means "every frame
    present in the fused file", which is the dense per-frame caliber.
    """
    all_frames = []
    for frame_id in range(total_frames):
        boxes = truth.get(frame_id)
        dets = fused.get(frame_id)
        if dets:
            ranked = sorted(dets, key=lambda item: -item[2])
            points = np.asarray([(d[0], d[1]) for d in ranked], dtype=np.float32)
            scores = np.asarray([d[2] for d in ranked], dtype=np.float32)
        else:
            points, scores = None, None
        all_frames.append((boxes, points, scores))

    m_all = metrics(count_matches(all_frames, 0.0, 8.0, "inbox"), len(all_frames))

    # The SERVER convention (manu/evaluation/rt_fps_jitter_search.score_run): the
    # delivered caliber iterates over EVERY delivered camera frame, and a frame the
    # tracker produced nothing for counts as a MISS (its ground truth feeds the FN
    # denominator).  Dropping the frames that are missing from the fused file (the
    # old behaviour of this script) silently removed them from both the recall and
    # the FAR denominator, which flattered the numbers.
    if delivered is None:
        delivered_ids = sorted(fused)
    else:
        delivered_ids = sorted(delivered)
    delivered_frames = []
    for frame_id in delivered_ids:
        dets = fused.get(frame_id)
        if dets:
            ranked = sorted(dets, key=lambda item: -item[2])
            points = np.asarray([(d[0], d[1]) for d in ranked], dtype=np.float32)
            scores = np.asarray([d[2] for d in ranked], dtype=np.float32)
        else:
            points, scores = None, None
        delivered_frames.append((truth.get(frame_id), points, scores))
    far_frames = len(delivered) if delivered is not None else len(delivered_ids)
    m_del = metrics(count_matches(delivered_frames, 0.0, 8.0, "inbox"), far_frames)

    return {
        "all": m_all,
        "delivered": m_del,
        "total_frames": total_frames,
        "delivered_frames": len(delivered_ids),
        "delivered_frames_with_output": sum(1 for f in delivered_ids if fused.get(f)),
        "gt_frames": len(truth),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-tsv", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--total-frames", type=int, default=0,
                        help="Full sequence length; 0 = max(frame id)+1 and max(GT id)+1")
    parser.add_argument("--delivered-ids", type=Path, default=None,
                        help="File with the delivered camera frame ids (one per line). "
                             "Selects the jump-frame caliber: the FAR denominator becomes the "
                             "number of delivered frames (server convention).")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    fused = parse_fused(args.fused_tsv)
    truth = load_ground_truth(args.labels_root, args.sequence, args.width, args.height)

    total = args.total_frames
    if total <= 0:
        total = max([0, *fused.keys(), *truth.keys()]) + 1

    delivered = None
    if args.delivered_ids:
        delivered = set()
        for raw in args.delivered_ids.read_text(encoding="utf-8").split():
            delivered.add(int(raw))

    result = score(fused, truth, total, delivered)
    result["sequence"] = args.sequence
    result["fused_tsv"] = str(args.fused_tsv)

    print(f"[SEQ] {args.sequence} total_frames={total} delivered={result['delivered_frames']} "
          f"(with output {result['delivered_frames_with_output']}) "
          f"gt_frames={result['gt_frames']} fused_frames={len(fused)}")
    for caliber in ("all", "delivered"):
        m = result[caliber]
        print(f"[{caliber:>8}] tp={m['tp']} fp={m['fp']} fn={m['fn']} gt={m['gt']} "
              f"recall={m['recall']:.6f} precision={m['precision']:.6f} f1={m['f1']:.7f} "
              f"far_per_frame={m['far_per_frame']:.7f}")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OUT] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
