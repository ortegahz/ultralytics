#!/usr/bin/env python3
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Cross-feed verification: the BOARD's candidate stream -> the server's Python fusion + tracking.

What this proves
----------------
The board and the server disagree on the per-frame detections of a sequence.  The question is
WHERE the divergence enters the chain:

    board NPU (FP16) --> candidates --> merge --> fusion/tracker --> fused.tsv
    server cache      --> candidates --> merge --> fusion/tracker --> fused.tsv

This script feeds the **board's own candidate stream** (``_cands.tsv``) into the **server's
Python mirror** (``sdk_track_fusion_ref.py``) and into the **C++** mirror
 (``track_fusion_align_test``), then compares the three outputs

  1. **bit for bit** -- the md5 of the non-empty frames of the three fused TSVs, and
  2. **metric for metric** -- tp/fp/fn/F1/FAR of the three outputs against the same ground
     truth, with the same matcher and the same tolerance.

If the Python mirror reproduces the board's fused TSV bit for bit, then the C++ merge + the
C++ tracker carry **zero** porting error: every remaining board-vs-server difference must
come from the stage *before* the tracker -- i.e. from the RKNN FP16 heat map versus the
server's own heat map.

Usage
-----
    python manu/evaluation/board_crossfeed_verify.py \
        --tsv-dir /mnt/data/siping/runs/trial0080_board24_20261120/tsv \
        --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json \
        --params-flat manu/memory/align_dn_20261120/params_delivery.txt \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --sequences VIDEO00007_19700101_001324,VIDEO00009_19700101_001531 \
        --radii 20 \
        --out /tmp/crossfeed/crossfeed_verify.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ULTRALYTICS_ROOT = str(Path(__file__).resolve().parents[2])
if ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ULTRALYTICS_ROOT)

from manu.evaluation.board_cand_merge_verify import build_align_test, run_align_test  # noqa: E402
from manu.evaluation.board_far_parity_check import parse_board_tsv, score_side  # noqa: E402
from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402


def md5_text(text: str) -> str:
    """md5 of a text payload."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def md5_file(path: Path) -> str:
    """md5 of a file's bytes."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def non_empty_lines(path: Path) -> list[str]:
    """Keep only the frames that carry at least one detection.

    The board's ``_fused.tsv`` holds only the frames with detections (the C++ side writes the
    flush output as it comes), while the Python mirror writes **every** frame of the axis
    (empty frames included, as ``<frame> 0``).  Comparing the two files therefore requires
    dropping the empty frames first -- an empty frame carries no information.
    """
    out: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and int(parts[1]) > 0:
                out.append(line)
    return out


def normalized_md5(path: Path) -> str:
    """md5 of the non-empty frames of a fused TSV (order preserved)."""
    return md5_text("\n".join(non_empty_lines(path)) + "\n")


def count_frames(path: Path) -> int:
    """Number of frames the file covers (last frame index + 1)."""
    last = -1
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            last = max(last, int(line.split()[0]))
    return last + 1


def diff_fused(left: dict[int, list], right: dict[int, list]) -> dict:
    """Frame-level difference between two parsed fused TSVs."""
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    common = sorted(set(left) & set(right))
    max_pos = 0.0
    max_score = 0.0
    mismatched = 0
    for frame in common:
        det_a = sorted(left[frame], key=lambda item: -item[2])
        det_b = sorted(right[frame], key=lambda item: -item[2])
        if len(det_a) != len(det_b):
            mismatched += 1
            continue
        for one, two in zip(det_a, det_b):
            max_pos = max(max_pos, abs(one[0] - two[0]), abs(one[1] - two[1]))
            max_score = max(max_score, abs(one[2] - two[2]))
    return {
        "frames_only_left": len(only_left),
        "frames_only_right": len(only_right),
        "frames_common": len(common),
        "frames_common_count_mismatch": mismatched,
        "max_pos_delta": max_pos,
        "max_score_delta": max_score,
        "sample_only_left": only_left[:10],
        "sample_only_right": only_right[:10],
    }


def run_python_mirror(
    cands: Path,
    params_json: Path,
    radius: float,
    out_path: Path,
    repo_root: Path,
    physical: bool = True,
) -> str:
    """Run the server's Python mirror over the board's candidate stream."""
    cmd = [
        sys.executable,
        str(repo_root / "manu" / "evaluation" / "sdk_track_fusion_ref.py"),
        "--cands",
        str(cands),
        "--params",
        str(params_json),
        "--dedup-radius",
        f"{radius:g}",
        "--out",
        str(out_path),
    ]
    if physical:
        cmd.append("--physical-frame-index")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"python mirror failed: rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv-dir", required=True, type=Path, help="the board's TSV artifact root")
    parser.add_argument("--params-json", required=True, type=Path, help="params JSON for the Python mirror")
    parser.add_argument("--params-flat", required=True, type=Path, help="flat params file for the C++ align test")
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--sequences", default="", help="comma separated; empty = every *_fused.tsv found")
    parser.add_argument("--radii", default="20")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tol", type=float, default=8.0)
    parser.add_argument("--total-frames-json", type=Path, default=None)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(ULTRALYTICS_ROOT) / "manu" / "pipeline" / "npu",
        help="the SDK sources to compile for the C++ align test",
    )
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo_root = Path(ULTRALYTICS_ROOT)
    radii = [float(r) for r in args.radii.split(",") if r.strip()]
    totals = {}
    if args.total_frames_json and args.total_frames_json.is_file():
        totals = json.loads(args.total_frames_json.read_text(encoding="utf-8"))

    if args.sequences.strip():
        sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    else:
        sequences = sorted(p.name[: -len("_fused.tsv")] for p in args.tsv_dir.glob("*_fused.tsv"))

    args.work_dir.mkdir(parents=True, exist_ok=True)
    binary = build_align_test(args.source_dir, args.work_dir)

    rows = []
    for sequence in sequences:
        board_cands = args.tsv_dir / f"{sequence}_cands.tsv"
        board_fused = args.tsv_dir / f"{sequence}_fused.tsv"
        if not board_cands.is_file() or not board_fused.is_file():
            print(f"[ERR] missing board artifacts for {sequence} under {args.tsv_dir}")
            continue
        truth = load_ground_truth(args.labels_root, sequence, args.width, args.height)
        n_frames = int(totals.get(sequence, count_frames(board_cands)))
        window = (0, n_frames)

        board_parsed = parse_board_tsv(board_fused)
        board_metrics = score_side(board_parsed, truth, window, args.tol)
        board_key = normalized_md5(board_fused)

        seq_dir = args.work_dir / sequence
        seq_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "sequence": sequence,
            "window": list(window),
            "frames": n_frames,
            "board_fused_tsv": str(board_fused),
            "board_fused_md5": md5_file(board_fused),
            "board_fused_nonempty_md5": board_key,
            "board": board_metrics,
            "radii": [],
        }
        for radius in radii:
            py_out = seq_dir / f"fused_py_r{int(radius)}.tsv"
            cpp_out = seq_dir / f"fused_cpp_r{int(radius)}.tsv"
            run_python_mirror(board_cands, args.params_json, radius, py_out, repo_root)
            run_align_test(binary, board_cands, args.params_flat, radius, cpp_out)

            py_parsed = parse_board_tsv(py_out)
            cpp_parsed = parse_board_tsv(cpp_out)
            py_metrics = score_side(py_parsed, truth, window, args.tol)
            cpp_metrics = score_side(cpp_parsed, truth, window, args.tol)
            py_key = normalized_md5(py_out)
            cpp_key = normalized_md5(cpp_out)

            row["radii"].append(
                {
                    "radius": radius,
                    "python_mirror_tsv": str(py_out),
                    "python_mirror_nonempty_md5": py_key,
                    "cpp_align_tsv": str(cpp_out),
                    "cpp_align_nonempty_md5": cpp_key,
                    "python_identical_to_board": py_key == board_key,
                    "cpp_identical_to_board": cpp_key == board_key,
                    "python_identical_to_cpp": py_key == cpp_key,
                    "python": py_metrics,
                    "cpp": cpp_metrics,
                    "diff_board_vs_python": diff_fused(board_parsed, py_parsed),
                    "diff_board_vs_cpp": diff_fused(board_parsed, cpp_parsed),
                }
            )

            print(
                f"[{sequence}] R={radius:g}  "
                f"board F1={board_metrics['f1']:.7f} FAR={board_metrics['far_per_frame']:.7f} | "
                f"py F1={py_metrics['f1']:.7f} FAR={py_metrics['far_per_frame']:.7f} | "
                f"cpp F1={cpp_metrics['f1']:.7f} FAR={cpp_metrics['far_per_frame']:.7f} | "
                f"md5 board={board_key[:8]} py={py_key[:8]} cpp={cpp_key[:8]} "
                f"identical={py_key == board_key and cpp_key == board_key}"
            )
        rows.append(row)

    payload = {
        "tsv_dir": str(args.tsv_dir),
        "params_json": str(args.params_json),
        "params_flat": str(args.params_flat),
        "labels_root": str(args.labels_root),
        "tol": args.tol,
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
