#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Board-side FAR/F1 probe: how much of the board/server gap is the missing candidate dedup?

Background
----------
The board runs the *mosaic* chain (RK3588 NPU -> mosaic heatmap peaks -> ``mosaic_track_fusion``).
The server runs the *multiscale* chain, whose candidate construction merges the peaks of the
different scales with ``dedup_radius`` (768 px on the delivery caliber).  The C++ fusion layer
has no such knob -- ``dedup_radius`` is a *candidate construction* parameter, not a fusion one --
so every raw heatmap peak reaches the board's fusion and a single target that produced two peaks
becomes two tracks (and therefore one extra false positive per frame).

This probe quantifies that: it re-runs the **server's own Python fusion mirror**
(``sdk_track_fusion_ref.py --physical-frame-index``) on the **board's own candidate stream**, with
and without a cross-peak dedup of radius R, and scores every variant against the same ground
truth and the same frame window.  The server's published per-frame predictions for the very same
sequence are scored with the identical matcher, so all rows are directly comparable.

Usage
-----
    python manu/evaluation/board_far_dedup_probe.py \
        --cands manu/memory/align_dn_20261120/real_0080/real_cands.tsv \
        --server-preds /tmp/boardfar/preds_cc2_md10_VIDEO00007.json \
        --labels-root /tmp/boardfar/gt \
        --sequence VIDEO00007_19700101_001324 \
        --params manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json \
        --radius 0,16,20,24,32,64,768 \
        --out /tmp/boardfar/dedup_sweep.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = str(Path(__file__).resolve().parents[2])
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.align_dn_delta_check import read_frames, write_candidates  # noqa: E402
from manu.evaluation.eval_multiscale_cache_th import (  # noqa: E402
    count_matches,
    load_ground_truth,
    metrics,
)


def parse_candidates(path: Path) -> list[list[tuple[float, float, float]]]:
    """Read the board candidate stream (``<frame_idx> <n> <cx> <cy> <peak> ...``)."""
    frames: list[list[tuple[float, float, float]]] = []
    for body in read_frames(path):
        count = int(body[0])
        dets = []
        for k in range(count):
            base = 1 + 3 * k
            dets.append((float(body[base]), float(body[base + 1]), float(body[base + 2])))
        frames.append(dets)
    return frames


def dedup_peaks(frames: list[list[tuple[float, float, float]]], radius: float):
    """Greedy cross-peak de-duplication, identical to the server's ``_merge_scale_rows`` rule.

    Candidates of a frame are pooled, sorted by descending score (stable), and a candidate is
    dropped when it lies within ``radius`` of an already kept candidate.  ``radius <= 0`` keeps
    the stream untouched (the board's current behaviour).
    """
    if radius <= 0:
        return [list(dets) for dets in frames]
    radius_squared = radius * radius
    out = []
    for dets in frames:
        kept: list[tuple[float, float, float]] = []
        kept_xy: list[tuple[float, float]] = []
        for det in sorted(dets, key=lambda d: -d[2]):
            x, y = det[0], det[1]
            if any((x - px) ** 2 + (y - py) ** 2 <= radius_squared for px, py in kept_xy):
                continue
            kept.append(det)
            kept_xy.append((x, y))
        out.append(kept)
    return out


def parse_fused(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    """Read a fused track dump (``<frame_idx> <n> [<id> <x> <y> <score> <infilled>] ...``)."""
    fused: dict[int, list[tuple[float, float, float]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        frame_idx = int(parts[0])
        count = int(parts[1])
        dets = []
        for k in range(count):
            base = 2 + 5 * k
            dets.append((float(parts[base + 1]), float(parts[base + 2]), float(parts[base + 3])))
        if dets:
            fused[frame_idx] = dets
    return fused


def parse_server_preds(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    """Read the server's published per-frame predictions."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, list[tuple[float, float, float]]] = {}
    for key, entries in payload["per_frame"].items():
        out[int(key)] = [(float(e["pos"][0]), float(e["pos"][1]), float(e["score"])) for e in entries]
    return out


def score(fused: dict[int, list[tuple[float, float, float]]],
          truth: dict[int, np.ndarray], window: tuple[int, int], tol: float) -> dict:
    """Score one side over the frame window with the shared matcher."""
    start, end = window
    frames = []
    for frame_id in range(start, end):
        boxes = truth.get(frame_id)
        dets = fused.get(frame_id)
        if dets:
            ranked = sorted(dets, key=lambda item: -item[2])
            frames.append((boxes,
                           np.asarray([(d[0], d[1]) for d in ranked], dtype=np.float32),
                           np.asarray([d[2] for d in ranked], dtype=np.float32)))
        else:
            frames.append((boxes, None, None))
    result = metrics(count_matches(frames, 0.0, tol, "inbox"), len(frames))
    result["frames"] = len(frames)
    return result


def run_fusion(cands: Path, params: Path, out: Path, repo_root: Path) -> None:
    """Run the server's Python fusion mirror on a candidate stream."""
    ref = repo_root / "manu" / "evaluation" / "sdk_track_fusion_ref.py"
    env = {"PYTHONPATH": str(repo_root), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    proc = subprocess.run(
        [sys.executable, str(ref), "--cands", str(cands), "--params", str(params),
         "--out", str(out), "--physical-frame-index"],
        cwd=repo_root, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"fusion reference failed: rc={proc.returncode} {proc.stderr[:400]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cands", required=True, type=Path)
    parser.add_argument("--server-preds", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--radius", default="0,16,20,24,32,64,768",
                        help="Comma separated dedup radii in px (0 = board's current behaviour)")
    parser.add_argument("--window", type=int, nargs=2, default=(0, 0),
                        help="[start, end) camera frame ids; end=0 -> use the candidate stream length")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tol", type=float, default=8.0)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo_root = Path(ULTRALYTICS_ROOT)
    frames = parse_candidates(args.cands)
    truth = load_ground_truth(args.labels_root, args.sequence, args.width, args.height)
    server = parse_server_preds(args.server_preds)
    window = (args.window[0], args.window[1]) if args.window[1] > 0 else (0, len(frames))

    work = args.workdir or Path(tempfile.mkdtemp(prefix="board_far_probe_"))
    work.mkdir(parents=True, exist_ok=True)

    server_m = score(server, truth, window, args.tol)
    print(f"[SEQ] {args.sequence} window={window[0]}..{window[1] - 1} ({window[1] - window[0]} frames)")
    print(f"[SERVER] tp={server_m['tp']} fp={server_m['fp']} fn={server_m['fn']} "
          f"f1={server_m['f1']:.6f} far_per_frame={server_m['far_per_frame']:.6f}")

    rows = []
    for radius in [float(r) for r in args.radius.split(",")]:
        deduped = dedup_peaks(frames, radius)
        cands = work / f"cands_r{int(radius)}.tsv"
        write_candidates(
            [[str(len(dets))] + [f"{value:.9g}" for det in dets for value in det] for dets in deduped],
            list(range(len(deduped))), cands)
        fused_path = work / f"fused_r{int(radius)}.tsv"
        run_fusion(cands, args.params, fused_path, repo_root)
        board_m = score(parse_fused(fused_path), truth, window, args.tol)
        row = {
            "radius": radius,
            "candidates_kept": sum(len(d) for d in deduped),
            "candidates_per_frame": sum(len(d) for d in deduped) / max(1, len(deduped)),
            "board": board_m,
            "delta_f1": board_m["f1"] - server_m["f1"],
            "delta_far": board_m["far_per_frame"] - server_m["far_per_frame"],
        }
        rows.append(row)
        label = "raw (board today)" if radius <= 0 else f"R={radius:g} px"
        print(f"[{label:>18}] kept/frame={row['candidates_per_frame']:>6.3f} "
              f"tp={board_m['tp']:>5} fp={board_m['fp']:>5} fn={board_m['fn']:>5} "
              f"f1={board_m['f1']:.6f} far_per_frame={board_m['far_per_frame']:.6f} "
              f"| dF1={row['delta_f1']:+.6f} dFAR={row['delta_far']:+.6f}")

    result = {
        "sequence": args.sequence,
        "window": list(window),
        "cands": str(args.cands),
        "server_preds": str(args.server_preds),
        "labels_root": str(args.labels_root),
        "params": str(args.params),
        "server": server_m,
        "rows": rows,
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OUT] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
