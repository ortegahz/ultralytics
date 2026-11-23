#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Verify the SDK's cross-scale candidate merge against the server's Python mirror.

What this proves
----------------
The board's FAR was 12.8x over the red line (0.3832 vs 0.03) because the SDK fed the
track fusion engine the **raw** candidate stream: one target that fired on several
scales became several tracks.  The server's multiscale chain merges the scales BEFORE
the fusion (``merge_scales(dedup_radius)``); the SDK had no such step.

``mosaic::MergeCandidatesByRadius`` is the SDK's counterpart.  This script runs the
**C++** merge + the **C++** fusion (through ``track_fusion_align_test``, which has no
third-party dependency and therefore compiles natively on the host and on the board)
over the **board's own candidate stream**, and

  1. compares the result **bit for bit** with the server's Python mirror
     (``sdk_track_fusion_ref.py``) run over the same merged stream, and
  2. scores every radius against the same ground truth with the same matcher, so the
     radius that will run on the board is chosen on measured numbers.

Because (1) is bit-exact, the scores of (2) are exactly what the board will produce
once it runs with the same radius: the offline projection becomes board-measurable.

Usage
-----
    python manu/evaluation/board_cand_merge_verify.py \
        --cands manu/memory/align_dn_20261120/real_0080/real_cands.tsv \
        --params manu/memory/align_dn_20261120/params_delivery.txt \
        --labels-root /tmp/boardfar/gt \
        --sequence VIDEO00007_19700101_001324 \
        --window 0 2500 \
        --radii 0,16,20,24,32 \
        --out /tmp/boardfar/merge_verify.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ULTRALYTICS_ROOT = str(Path(__file__).resolve().parents[2])
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.board_far_parity_check import parse_board_tsv, score_side  # noqa: E402
from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402

# The C++ flags of the delivery build (see manu/pipeline/npu/build/build_lib.sh):
# no FMA contraction, no fast-math => bit-reproducible with the board.
CPP_FLAGS = [
    "-std=c++17",
    "-O2",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-fno-unsafe-math-optimizations",
]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_align_test(source_dir: Path, out_dir: Path) -> Path:
    """Compile the zero-dependency alignment tool on this machine."""
    binary = out_dir / "track_fusion_align_test"
    cmd = [
        "g++",
        *CPP_FLAGS,
        "-o",
        str(binary),
        str(source_dir / "track_fusion_align_test.cpp"),
        str(source_dir / "mosaic_track_fusion.cpp"),
        "-I",
        str(source_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"compilation failed: rc={proc.returncode}\n{proc.stderr[:2000]}")
    return binary


def run_align_test(binary: Path, cands: Path, params: Path, radius: float, out: Path) -> None:
    cmd = [str(binary), "--cands", str(cands), "--params", str(params), "--out", str(out)]
    if radius > 0:
        cmd += ["--dedup-radius", f"{radius:g}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"align test failed: rc={proc.returncode}\n{proc.stderr[:2000]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cands", required=True, type=Path, help="the BOARD's candidate stream")
    parser.add_argument("--params", required=True, type=Path, help="the flat `key value` params file")
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--window", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--radii", default="0,16,20,24,32")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tol", type=float, default=8.0)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(ULTRALYTICS_ROOT) / "manu" / "pipeline" / "npu",
        help="the SDK sources to compile (the repo tree or the delivery tree)",
    )
    parser.add_argument(
        "--python-mirror-dir",
        type=Path,
        default=None,
        help="directory holding fused_r<R>.tsv produced by the Python mirror (for the bit-exact check)",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    radii = [float(r) for r in args.radii.split(",")]
    truth = load_ground_truth(args.labels_root, args.sequence, args.width, args.height)
    work = Path(tempfile.mkdtemp(prefix="board_merge_verify_"))
    try:
        binary = build_align_test(args.source_dir, work)
        rows = []
        for radius in radii:
            fused = work / f"fused_cpp_r{int(radius)}.tsv"
            run_align_test(binary, args.cands, args.params, radius, fused)
            metrics = score_side(parse_board_tsv(fused), truth, (args.window[0], args.window[1]), args.tol)
            row = {
                "radius": radius,
                "fused_tsv_md5": md5(fused),
                "fused_tsv": str(fused),
                "board": metrics,
            }
            if args.python_mirror_dir is not None:
                mirror = args.python_mirror_dir / f"fused_r{int(radius)}.tsv"
                row["python_mirror_tsv"] = str(mirror) if mirror.exists() else None
                row["identical_to_python_mirror"] = bool(mirror.exists() and md5(mirror) == md5(fused))
            rows.append(row)
        payload = {
            "sequence": args.sequence,
            "cands": str(args.cands),
            "params": str(args.params),
            "source_dir": str(args.source_dir),
            "labels_root": str(args.labels_root),
            "window": list(args.window),
            "rows": rows,
        }
        text = json.dumps(payload, indent=2, sort_keys=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
            print(f"[OUT] {args.out}")
        else:
            print(text)
        for row in rows:
            board = row["board"]
            flag = "" if row.get("identical_to_python_mirror", True) else "  <-- DIFFERS from the python mirror"
            print(
                f"R={row['radius']:>5.1f}  tp={board['tp']:>5} fp={board['fp']:>5} fn={board['fn']:>5} "
                f"F1={board['f1']:.7f} FAR={board['far_per_frame']:.4f}{flag}"
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
