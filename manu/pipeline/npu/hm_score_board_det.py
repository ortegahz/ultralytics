#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Score a board ``--detections-out`` TSV against the CPU reference detections.

Why this exists
---------------
The board binary only prints its per-frame detections when the batch has <= 12
frames, so a 600-frame run produced nothing unless the raw tensors were dumped.
Dumping them costs 1.2 GB over NFS and measured at **~2 frames/minute** -- 583
minutes for 600 frames. The compact TSV is one line per frame, so this path is
about three orders of magnitude cheaper and gives exactly the same TP/FP/FN.

TSV format written by ``rknn_heatmap_smoke --detections-out``::

    <stem>\\t<ndet>\\t<x y score>\\t<x y score> ...

coordinates are canvas-space and already stride-compensated, matching
``hm_common.extract_peaks_np`` and ``hm_reference.py``'s ``det_canvas``.

Every proportion is reported as absolute numbers + a Wilson 95% interval, and
F1 is reported bare (F1 is not a proportion -- rules.md section 3.6b).

Usage:
    python hm_score_board_det.py --tsv <board.tsv> --ref <ref_dir> [--th 0.25]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.pipeline.npu.hm_seq_stats import wilson  # noqa: E402


def parse_tsv(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        # The board writes the basename WITH its extension; the reference keys on
        # Path(...).stem. Normalising here keeps the two joinable -- and a silent
        # mismatch would otherwise score as a perfect 0/0 rather than an error.
        stem = Path(parts[0]).stem
        n = int(parts[1])
        dets = []
        for k in range(n):
            x, y, sc = parts[2 + k].split()
            dets.append((float(x), float(y), float(sc)))
        out[stem] = dets
    return out


def greedy_match(dets, gts, dist: float):
    """Greedy nearest-neighbour matching, same 8 px convention as the reference."""
    if not dets or not gts:
        return 0, len(dets), len(gts)
    d = [[math.hypot(a[0] - b[0], a[1] - b[1]) for b in gts] for a in dets]
    used_g, used_d, tp = set(), set(), 0
    for i in range(len(dets)):
        order = sorted(range(len(gts)), key=lambda j: d[i][j])
        for j in order:
            if j in used_g:
                continue
            if d[i][j] <= dist:
                used_g.add(j)
                used_d.add(i)
                tp += 1
            break
    return tp, len(dets) - tp, len(gts) - tp


def f1_of(p: float, r: float) -> float:
    return 0.0 if (p + r) <= 0 else 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="board --detections-out file")
    ap.add_argument("--ref", required=True, help="CPU reference dir (has detections.json)")
    ap.add_argument("--dist", type=float, default=8.0)
    ap.add_argument("--out", default=None, help="write a per-frame JSON here")
    args = ap.parse_args()

    board = parse_tsv(Path(args.tsv))
    recs = json.loads((Path(args.ref) / "detections.json").read_text())

    tot_tp = tot_fp = tot_fn = tot_gt = 0
    per_frame = []
    missing = 0
    for r in recs:
        stem = Path(r["name"]).stem
        if stem not in board:
            missing += 1
            continue
        gt = [tuple(p) for p in r["gt_canvas"]]
        tp, fp, fn = greedy_match(board[stem], gt, args.dist)
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn
        tot_gt += len(gt)
        per_frame.append({"stem": stem, "seq": r["seq"], "tp": tp, "fp": fp,
                          "fn": fn, "n_gt": len(gt),
                          "cpu_tp": r["tp"], "cpu_fp": r["fp"], "cpu_fn": r["fn"]})

    n = len(per_frame)
    p = tot_tp / max(1, tot_tp + tot_fp)
    rc = tot_tp / max(1, tot_gt)
    plo, phi = wilson(tot_tp, tot_tp + tot_fp)
    rlo, rhi = wilson(tot_tp, tot_gt)

    print(f"frames scored : {n}   (no board line for {missing})")
    print(f"GT {tot_gt}  TP {tot_tp}  FP {tot_fp}  FN {tot_fn}")
    print(f"precision = {tot_tp}/{tot_tp + tot_fp} = {p:.4f}  Wilson [{plo:.4f}, {phi:.4f}]")
    print(f"recall    = {tot_tp}/{tot_gt} = {rc:.4f}  Wilson [{rlo:.4f}, {rhi:.4f}]")
    print(f"F1        = {f1_of(p, rc):.4f}   (no interval: F1 is not a proportion)")

    if args.out:
        Path(args.out).write_text(json.dumps(per_frame, indent=1))
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
