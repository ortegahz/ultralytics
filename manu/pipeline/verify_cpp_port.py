#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Bit-exactness verifier for the C++ port of the streaming feature engine
(``manu/pipeline/cpp/gmc_stream.cpp``).

This script deliberately does NOT re-implement the feature pipeline. It drives the
already-frozen Python engine (``manu.pipeline.streaming_feature_pipeline.OnlineFeaturePipeline``)
so that any divergence is attributable to the C++ port, never to a second Python
implementation that drifted from the first.

Gates
-----
G0  empty control / port fidelity
    With ``--anchor-step 2`` the anchor grid is every even lag {2,4,...,42}, so the
    composition depth is ZERO and the new arm is *mathematically the frozen native
    pipeline* -- every lag gets its own long-baseline direct fit. That degenerates the
    tested variable to the identity, so any residual here is pure C++ port drift (or a
    broken harness), never a feature-design difference. Non-zero G0 invalidates every
    other number in the report. This is the "zero-code empty control" construction:
    make the variable under test the identity and whatever remains is the confound.

G1  bit-exact arrays
    C++ output vs Python output, compared on the PRE-ENCODE uint8 arrays via a streaming
    md5 per frame. This is the only strict-zero criterion in the report, and it is the one
    that must hold. Comparing decoded JPGs is explicitly NOT used here: three-channel JPEG
    is 4:2:0 chroma-subsampled and JPEG-quantised, so a decoded-pixel difference measures
    the storage format, not the feature.

G2  re-encoded files vs the frozen dataset
    Only meaningful for ``--anchor-step 2``: both sides run the same encoder on
    bit-identical arrays, so the files must be byte-identical. Reported as file-level
    md5 plus a decoded-level MAE, the latter purely informational.

[MATS]  per-lag transform matrices
    C++ vs Python for every lag, so a first divergence can be localised to an anchor fit
    or a composed lag instead of showing up only as a final-pixel mismatch.

Reporting discipline (from the project's measurement rules)
    Every gate prints how many samples it actually compared. A gate that compared nothing
    prints SKIP and never PASS -- a false PASS is more dangerous than a FAIL.

Usage on the server:
    PYTHONPATH=. python manu/pipeline/verify_cpp_port.py \
        --cpp-bin manu/pipeline/cpp/build/gmc_stream \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --sequence wg2022_ir_052_split_08 --limit 200 \
        --anchor-step 2 \
        --frozen-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --cpp-out-dir runs/cpp_port/g2_jpg --md5-out runs/cpp_port/cpp.md5 \
        --dump-dir runs/cpp_port/dump --out-json runs/cpp_port/verify.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_sample_median_dataset import (  # noqa: E402
    IMAGE_SUFFIXES,
    find_sequence_folder,
    natural_key,
    parse_seq_and_frame,
)
from manu.pipeline.streaming_feature_pipeline import (  # noqa: E402
    OnlineFeaturePipeline,
    SequenceFrameSource,
)


def md5_bytes(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


# --------------------------------------------------------------------------------------
# C++ runner
# --------------------------------------------------------------------------------------
def run_cpp(args, sequences: list[str]) -> tuple[dict[str, str], str]:
    cmd = [
        args.cpp_bin,
        "--raw-root", args.raw_root,
        "--anchor-step", str(args.anchor_step),
        "--window", str(args.window),
        "--stride-step", str(args.stride_step),
        "--downscale", str(args.downscale),
        "--limit", str(args.limit),
    ]
    for s in sequences:
        cmd += ["--sequence", s]
    if args.cpp_out_dir:
        cmd += ["--out-dir", args.cpp_out_dir]
    if args.dump_dir:
        cmd += ["--dump-dir", args.dump_dir]
    if args.md5_out:
        Path(args.md5_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md5_out).write_bytes(b"")  # truncate; the binary appends
        cmd += ["--md5-out", args.md5_out]
    if args.rng_seed_per_fit:
        cmd.append("--rng-seed-per-fit")
    if args.timing:
        cmd.append("--timing")

    print("=" * 104)
    print("   C++ PORT BIT-EXACTNESS VERIFIER")
    print("=" * 104)
    print(f"  binary       : {args.cpp_bin}")
    print(f"  anchor_step  : {args.anchor_step}"
          f"{'   (= exact-equivalence arm; G0 empty control)' if args.anchor_step == 2 else ''}")
    print(f"  window       : {args.window}   stride_step: {args.stride_step}   downscale: {args.downscale}")
    print(f"  sequences    : {sequences}")
    print(f"  limit/seq    : {args.limit if args.limit > 0 else 'all'}")
    if args.rng_seed_per_fit:
        print("  [!] --rng-seed-per-fit is ON; the C++ side reseeds before every fit.")
        print("      This does NOT reproduce the frozen files (built without reseeding).")
    print("=" * 104, flush=True)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(f"\n[FATAL] C++ binary exited {proc.returncode}\n{proc.stderr}", file=sys.stderr)
        raise SystemExit(proc.returncode)

    seq_md5: dict[str, str] = {}
    for m in re.finditer(r"^\[SEQ\]\s+(\S+)\s+frames=(\d+)\s+md5=([0-9a-f]{32})", proc.stdout, re.M):
        seq_md5[m.group(1)] = m.group(3)
    if not seq_md5:
        print("[FATAL] no [SEQ] md5 lines parsed from the C++ stdout -- the binary output "
              "format changed or it ran zero frames.", file=sys.stderr)
        raise SystemExit(4)
    return seq_md5, proc.stdout


# --------------------------------------------------------------------------------------
# Python golden side
# --------------------------------------------------------------------------------------
def python_side(args, sequences: list[str]):
    """Drive the frozen Python engine.

    Returns {seq: md5}, {seq: per-frame tensors}, and {seq: transform matrices}.

    The matrices are captured on the FIRST pushed frame, because that is what the C++ binary
    dumps (`_mats.npy` is written at push index 0). Using `pipe.last_mats` after the loop --
    which is what a naive implementation does -- compares frame 0 against the last frame and
    therefore always reports a bogus max|diff|. They are also kept per sequence: a single
    shared variable would silently compare the first sequence's C++ matrices against the
    LAST sequence's Python ones on a multi-sequence run.
    """
    seq_md5: dict[str, str] = {}
    frames_by_seq: dict[str, list[np.ndarray]] = {}
    mats_by_seq: dict[str, dict[int, np.ndarray]] = {}

    for seq in sequences:
        pipe = OnlineFeaturePipeline(
            window=args.window,
            stride_step=args.stride_step,
            anchor_step=args.anchor_step,
            downscale=args.downscale,
        )
        src = SequenceFrameSource(args.raw_root, seq, limit=args.limit)
        h = hashlib.md5()
        kept: list[np.ndarray] = []
        first_mats: dict[int, np.ndarray] = {}
        for i, out in enumerate(pipe.run(src)):
            d = md5_bytes(out)
            h.update(d.encode())
            kept.append(out)
            if i == 0:
                first_mats = dict(pipe.last_mats or {})
        mats_by_seq[seq] = first_mats
        seq_md5[seq] = h.hexdigest()
        frames_by_seq[seq] = kept
        print(f"[PY ] {seq:<32} frames={len(kept):<6} md5={h.hexdigest()}")
    return seq_md5, frames_by_seq, mats_by_seq


# --------------------------------------------------------------------------------------
# push index -> frozen manifest name
# --------------------------------------------------------------------------------------
def build_push_to_manifest(raw_root: str, frozen_root: str, seq: str, limit: int):
    """Map a C++ push index to the frozen dataset's manifest filename.

    The C++ binary (like ``SequenceFrameSource``) walks the raw frames in ``natural_key``
    order. The frozen dataset is keyed by manifest names whose frame number resolves to a
    raw list index through ``idx_map`` in the offline builder. Reusing that builder's own
    helpers keeps the mapping single-sourced -- a hand-rolled name guess is exactly how a
    verification ends up comparing different frames while reporting zero drift.
    """
    seq_dir = find_sequence_folder(Path(raw_root), seq, {})
    if seq_dir is None:
        return {}
    raw = sorted([p.name for p in seq_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES], key=natural_key)
    idx_map: dict[int, int] = {}
    for list_i, f in enumerate(raw):
        m = re.search(r"(\d+)$", Path(f).stem)
        idx_map[int(m.group(1)) if m else list_i] = list_i

    frozen = Path(frozen_root) / "images" / "val"
    if not frozen.is_dir():
        return {}
    # `idx_map` is already frame_number -> list_index, which is exactly the direction needed:
    # the manifest carries a FRAME NUMBER and we need the push index. Inverting it and then
    # looking up by frame number silently matches the wrong key (and misses most frames),
    # which would make G2 compare unrelated frames while reporting a clean result.
    out: dict[int, str] = {}
    for p in frozen.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            s, fi = parse_seq_and_frame(p.name)
        except ValueError:
            continue
        if s != seq:
            continue
        li = idx_map.get(fi)
        if li is not None:
            out[li] = p.name
    if limit > 0:
        out = {k: v for k, v in out.items() if k < limit}
    return out


def g2_frozen_check(args, sequences: list[str]) -> dict:
    import cv2

    out_dir = Path(args.cpp_out_dir) if args.cpp_out_dir else None
    frozen_dir = Path(args.frozen_root) / "images" / "val"
    if out_dir is None:
        return {"usable": False, "reason": "--cpp-out-dir not given, so no re-encoded files"}
    if not frozen_dir.is_dir():
        return {"usable": False, "reason": f"{frozen_dir} not found"}

    summary = []
    for seq in sequences:
        mapping = build_push_to_manifest(args.raw_root, args.frozen_root, seq, args.limit)
        if not mapping:
            print(f"[G2] {seq:<28} no manifest mapping -> SKIP (compared 0 frames)")
            summary.append({"seq": seq, "frames_compared": 0, "files_byte_identical": 0,
                            "mae": [0.0, 0.0, 0.0], "ge2": [0, 0, 0], "max_abs": [0, 0, 0]})
            continue
        mae = np.zeros(3, dtype=np.float64)
        ge2 = np.zeros(3, dtype=np.int64)
        maxd = np.zeros(3, dtype=np.int64)
        n_arr = n_file = 0
        for push_idx, fname in sorted(mapping.items()):
            fz = frozen_dir / fname
            mine = out_dir / f"{seq}_{push_idx:06d}.jpg"
            if not fz.is_file() or not mine.is_file():
                continue
            n_file += int(fz.read_bytes() == mine.read_bytes())
            a = cv2.imread(str(fz), cv2.IMREAD_UNCHANGED)
            b = cv2.imread(str(mine), cv2.IMREAD_UNCHANGED)
            if a is None or b is None or a.shape != b.shape:
                continue
            if a.ndim == 2:
                a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
                b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
            n_arr += 1
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            mae += d.sum(axis=(0, 1), dtype=np.float64)
            ge2 += (d.reshape(-1, 3) >= 2).sum(axis=0)
            maxd = np.maximum(maxd, d.reshape(-1, 3).max(axis=0))
        pix = max(1, n_arr)
        summary.append({
            "seq": seq, "frames_compared": n_arr, "files_byte_identical": n_file,
            "mae": (mae / pix).tolist(), "ge2": ge2.tolist(), "max_abs": maxd.tolist(),
        })
        tag = "" if n_arr else "   SKIP (compared 0 frames)"
        print(f"[G2] {seq:<28} compared {n_arr:>5} arrays, byte-identical files {n_file:>5}{tag}")
    return {"usable": any(s["frames_compared"] > 0 for s in summary), "per_sequence": summary}


def compare_mats(dump_dir: str, seq: str, py_mats: dict[int, np.ndarray]):
    p = Path(dump_dir) / f"{seq}_000000_mats.npy"
    if not p.is_file() or not py_mats:
        return 0, float("nan")
    cpp = np.load(p)
    lags = sorted(py_mats)
    if cpp.shape[0] != len(lags):
        print(f"[MATS] LAYOUT MISMATCH: C++ dumped {cpp.shape[0]} matrices, Python has {len(lags)}")
        return 0, float("nan")
    ref = np.stack([np.asarray(py_mats[l], dtype=np.float64).reshape(-1) for l in lags])
    return len(lags), float(np.max(np.abs(ref - cpp.astype(np.float64))))


def main() -> int:
    p = argparse.ArgumentParser(description="Verify the C++ feature-engine port is bit-exact")
    p.add_argument("--cpp-bin", default="manu/pipeline/cpp/build/gmc_stream")
    p.add_argument("--raw-root", default="/mnt/data/siping/datasets/manu/anti-uav")
    p.add_argument("--sequence", action="append", default=[])
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--anchor-step", type=int, default=2,
                   help="2 = exact-equivalence arm (empty control); 10 = shipping config")
    p.add_argument("--window", type=int, default=21)
    p.add_argument("--stride-step", type=int, default=2)
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--frozen-root", default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    p.add_argument("--cpp-out-dir", default="")
    p.add_argument("--dump-dir", default="")
    p.add_argument("--md5-out", default="")
    p.add_argument("--rng-seed-per-fit", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--skip-g2", action="store_true")
    p.add_argument("--skip-python", action="store_true",
                   help="Do not drive the Python engine; run only the C++ binary and the\n                         G2 frozen-dataset gate. Halves wall time on the 24-sequence run,\n                         where G2 alone already decides the anchor-step 2 claim.")
    p.add_argument("--out-json", default="runs/cpp_port/verify_cpp_port.json")
    a = p.parse_args()

    import cv2
    cv2.setNumThreads(1)

    if not a.sequence:
        p.error("--sequence is required (repeatable); an implicit sequence choice is exactly how "
                "a local check gets mistaken for a full-set one")

    if a.skip_python:
        print("[NOTE] --skip-python: the Python engine is NOT driven. G1 is skipped by construction;\n"
              "       only G2 (C++ vs the frozen dataset) is evaluated. Do not read this as G1 evidence.")
        py_md5, frames_by_seq, py_mats = {}, {s: [] for s in a.sequence}, {}
    else:
        py_md5, frames_by_seq, py_mats = python_side(a, a.sequence)
    cpp_md5, _cpp_stdout = run_cpp(a, a.sequence)

    # ---- G1: bit-exact arrays ------------------------------------------------------
    print("\n" + "-" * 104)
    print("[G1] PRE-ENCODE ARRAY IDENTITY  (C++ vs frozen Python engine, streaming md5/frame)")
    print("-" * 104)
    g1_rows = []
    if a.skip_python:
        print("  (G1 not evaluated: --skip-python)")
    for seq in a.sequence:
        if a.skip_python:
            break
        n = len(frames_by_seq[seq])
        pm, cm = py_md5.get(seq, ""), cpp_md5.get(seq, "")
        ok = bool(pm) and pm == cm
        g1_rows.append({"seq": seq, "frames": n, "python_md5": pm, "cpp_md5": cm, "match": ok})
        print(f"  {seq:<28} frames={n:<6} python={pm or 'MISSING'}  cpp={cm or 'MISSING'}  "
              f"{'MATCH' if ok else 'MISMATCH'}")
        if not ok and a.md5_out and Path(a.md5_out).is_file():
            py_frames = [md5_bytes(f) for f in frames_by_seq[seq]]
            cpp_frames = [ln.split("\t")[2].strip() for ln in Path(a.md5_out).read_text().splitlines()
                          if ln.startswith(seq + "\t")]
            bad = [i for i, (x, y) in enumerate(zip(py_frames, cpp_frames)) if x != y]
            print(f"       first differing frame index: {bad[0] if bad else 'n/a'}; "
                  f"{len(bad)}/{min(len(py_frames), len(cpp_frames))} frames differ")
            if bad:
                print(f"       per-frame md5 in {a.md5_out}; --dump-dir .npy files hold the "
                      f"differing tensors for inspection")

    total_frames = sum(r["frames"] for r in g1_rows)
    g1_matched = sum(r["frames"] for r in g1_rows if r["match"])
    g1_ok = total_frames > 0 and all(r["match"] for r in g1_rows)
    g1_evaluated = not a.skip_python

    # ---- [MATS] --------------------------------------------------------------------
    # Per sequence, and against that sequence's OWN frame-0 matrices. Comparing sequence[0]
    # against a shared variable would mis-pair the two halves on any multi-sequence run.
    print()
    mats_rows: list[dict] = []
    if a.dump_dir:
        for seq in a.sequence:
            n_lags, mats_max = compare_mats(a.dump_dir, seq, py_mats.get(seq, {}))
            if not n_lags:
                print(f"[MATS] {seq:<28} SKIP -- no matrices dumped (compared 0 lags)")
                continue
            mats_rows.append({"seq": seq, "lags": n_lags, "max_abs": mats_max})
            line = (f"[MATS] {seq:<28} compared {n_lags} lags, max|diff| = {mats_max:.3e}")
            print(line)
            if mats_max > 0:
                print("       Anchor matrices already differ -> the GMC estimator or its inputs "
                      "differ; a composition-level fix cannot help.")
        if not mats_rows:
            print("[MATS] SKIP for all sequences (compared 0 lags)")
    else:
        print("[MATS] SKIP -- --dump-dir not given")

    # ---- G2 ------------------------------------------------------------------------
    g2 = {"usable": False, "reason": "not applicable: G2 only runs on the anchor-step 2 arm"}
    if not a.skip_g2 and a.anchor_step == 2:
        print("\n" + "-" * 104)
        print("[G2] RE-ENCODED FILE IDENTITY vs FROZEN DATASET (anchor-step 2 arm only)")
        print("-" * 104)
        g2 = g2_frozen_check(a, a.sequence)

    if a.anchor_step == 2:
        g0_note = ("anchor-step 2 => composition depth 0 => this arm IS the frozen native "
                   "pipeline; any non-zero residual is pure port drift")
    else:
        g0_note = (f"anchor-step {a.anchor_step} => this arm is NOT the native baseline; G1 only "
                   f"certifies C++==Python, not equivalence to the frozen native features")

    g2_pass = bool(g2.get("usable")) and g2["usable"] and all(
        s["files_byte_identical"] == s["frames_compared"] for s in g2.get("per_sequence", [])) \
        and sum(s["frames_compared"] for s in g2.get("per_sequence", [])) > 0
    verdict = ("PASS" if (g1_ok if g1_evaluated else g2_pass) else "FAIL")
    print("\n" + "=" * 104)
    if g1_evaluated:
        print(f"   [G1] C++ vs Python, pre-encode arrays : "
              f"{'PASS' if g1_ok else 'FAIL'}  ({g1_matched}/{total_frames} frames matched)")
    else:
        print("   [G1] C++ vs Python, pre-encode arrays : SKIP  (--skip-python; G2 only)")
    if g2.get("usable"):
        n_tot = sum(s["frames_compared"] for s in g2["per_sequence"])
        n_same = sum(s["files_byte_identical"] for s in g2["per_sequence"])
        print(f"   [G2] re-encoded files vs frozen      : "
              f"{'PASS' if n_same == n_tot else 'FAIL'}  ({n_same}/{n_tot} files byte-identical)")
    else:
        print(f"   [G2] re-encoded files vs frozen      : SKIP  ({g2.get('reason')})")
    print(f"   [G0] {g0_note}")
    print(f"\n[VERDICT] {verdict}")
    print(f"[SCOPE] sequences={len(a.sequence)} frames/seq={a.limit} -- a LOCAL check. "
          f"Full-dataset evidence needs a 24-sequence run.")
    print("=" * 104)

    if a.out_json:
        out = Path(a.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "anchor_step": a.anchor_step, "window": a.window, "stride_step": a.stride_step,
            "downscale": a.downscale, "sequences": a.sequence, "limit": a.limit,
            "g1": {"rows": g1_rows, "matched_frames": g1_matched, "pass": g1_ok},
            "mats": mats_rows, "g2": g2, "verdict": verdict,
            "scope_note": f"sequences={len(a.sequence)} frames/seq={a.limit} "
                          f"-- LOCAL check, not full-dataset evidence",
        }, indent=2), encoding="utf-8")
        print(f"[REPORT] {out}")
    return 0 if g1_ok else 1


if __name__ == "__main__":
    sys.exit(main())