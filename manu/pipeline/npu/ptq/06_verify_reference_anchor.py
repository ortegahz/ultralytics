#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
06_verify_reference_anchor.py -- prove that the BMP path reproduces the numbers in memory.

Two independent proofs, in increasing strength
-----------------------------------------------
A. PIXEL PROOF (cheap, exhaustive over a sample)
   For every sampled frame: `cv2.imread(BMP)` must equal `cv2.imread(JPG)` **bit for
   bit**. A BMP is a lossless re-encode of the JPEG's decoded pixels, so this MUST
   hold; if it does not, the staging is broken and every downstream number is void.

B. END-TO-END PROOF (expensive, one sequence, zero code duplication)
   The stored gate cache (`trial_0080/gate/epoch3/ms_cache/<seq>.pkl`) holds the
   multi-scale fused detections that produced the numbers recorded in memory
   (T1 0.901910 / T2 0.998949 / T3 0.897694). It was built by
   `manu/data/multiscale_heatmap_cache.py` reading the **JPG** features.
   This script re-runs THAT EXACT SCRIPT on the **BMP** features and compares the
   resulting detections frame by frame against the stored cache.

   The trick that makes this possible without touching the pipeline: the staging
   BMPs are exposed through a symlink tree whose names follow the pattern the
   pipeline expects (`<seq>__frame_%06d.jpg`) but whose CONTENT is the BMP.
   OpenCV picks the decoder from the file header, not from the extension, so the
   pipeline reads the BMPs while believing it reads JPEGs. The pipeline code is
   therefore provably identical in both arms -- the only variable is the pixels.

Usage
-----
    python 06_verify_reference_anchor.py --seq VIDEO00009_19700101_001531
    python 06_verify_reference_anchor.py --pixel-sample 4000 --skip-e2e
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

SERVER_ROOT = C.SERVER_ROOT
GATE_CACHE = SERVER_ROOT / "runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"
WORK = C.WORK_ROOT
LINK_ROOT = WORK / "bmp_as_jpg"


def _pixel_worker(pair):
    import cv2

    jpg, bmp = pair
    a = cv2.imread(jpg, cv2.IMREAD_COLOR)
    b = cv2.imread(bmp, cv2.IMREAD_COLOR)
    if a is None or b is None:
        return (jpg, bmp, "unreadable", -1)
    if a.shape != b.shape:
        return (jpg, bmp, "shape", -1)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return (jpg, bmp, "ok", int(d.max()))


def pixel_proof(sample: int, seed: int = 20261022) -> dict:
    import multiprocessing as mp
    import random

    pairs = []
    for seq in C.TRACK2_GOLDEN + C.TRACK3_BLIND:
        ids = C.seq_frames_from_official(C.FEATURE_OFFICIAL_VAL, seq)
        for fid in ids:
            src = C.official_feature_path(C.FEATURE_OFFICIAL_VAL, seq, fid)
            bmp = C.BMP_ROOT / seq / f"{seq}__{fid:06d}.bmp"
            if src.is_file() and bmp.is_file():
                pairs.append((str(src), str(bmp)))
    if not pairs:
        raise SystemExit("[FATAL] no JPG/BMP pairs found -- the staging did not run?")
    rng = random.Random(seed)
    if sample and sample < len(pairs):
        pairs = rng.sample(pairs, sample)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=40) as pool:
        res = pool.map(_pixel_worker, pairs, chunksize=32)

    bad = [r for r in res if r[2] != "ok" or r[3] != 0]
    maxdiff = max((r[3] for r in res if r[3] >= 0), default=-1)
    print(f"[pixel] compared {len(res)} frame pairs: mismatches={len(bad)} max_abs_diff={maxdiff}")
    for r in bad[:5]:
        print(f"   MISMATCH {r[0]} vs {r[1]}  ({r[2]}, maxdiff={r[3]})")
    return {"compared": len(res), "mismatches": len(bad), "max_abs_diff": maxdiff,
            "examples": [list(map(str, b)) for b in bad[:5]]}


def build_symlink_tree(seqs: list[str]) -> Path:
    """Expose the staged BMPs under the names the pipeline expects."""
    LINK_ROOT.mkdir(parents=True, exist_ok=True)
    n = 0
    for seq in seqs:
        src_dir = C.BMP_ROOT / seq
        if not src_dir.is_dir():
            continue
        for bmp in sorted(src_dir.glob("*.bmp")):
            fid = int(bmp.stem.rsplit("__", 1)[1])
            link = LINK_ROOT / f"{seq}__frame_{fid:06d}.jpg"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(bmp)
            n += 1
    print(f"[link] {n} symlinks -> {LINK_ROOT}")
    return LINK_ROOT


def load_cache_detections(pkl: Path) -> dict:
    """The stored cache's fused detections, keyed by frame index."""
    obj = pickle.load(open(pkl, "rb"))
    pts = obj.get("points")
    scores = obj.get("scores")
    offset = obj.get("offset")
    out = {}
    if pts is None:
        return out
    for i, fi in enumerate(obj.get("frame_index", [])):
        a, b = offset[i], offset[i + 1]
        out[int(fi)] = {
            "points": np.asarray(pts[a:b], np.float32).reshape(-1, 2),
            "scores": np.asarray(scores[a:b], np.float32).reshape(-1),
        }
    return out


def e2e_proof(seq: str, ablate_p0: bool = True) -> dict:
    """Re-run the real cache builder on the BMP tree and diff against the stored one."""
    stored_pkl = GATE_CACHE / f"{seq}.pkl"
    if not stored_pkl.is_file():
        raise SystemExit(f"[FATAL] no stored gate cache for {seq}: {stored_pkl}")

    link_root = build_symlink_tree([seq])
    out_dir = WORK / "anchor_ms_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "manu/data/multiscale_heatmap_cache.py",
           "--features-root", str(link_root),
           "--cache-root", str(out_dir),
           "--scales", "160,80",
           "--region-threshold", "0.06", "--main-threshold", "0.22",
           "--fusion-dilate", "32", "--min-area", "4",
           "--device", "0", "--chunk-size", "64", "--batch-size", "8",
           "--cpu-threads", "2", "--hm-weights", str(C.WEIGHTS), "--overwrite"]
    if ablate_p0:
        cmd.append("--ablate-p0")
    print(f"[e2e] {' '.join(cmd)}")
    t0 = __import__("time").time()
    p = subprocess.run(cmd, cwd=str(SERVER_ROOT), capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(SERVER_ROOT)})
    dt = __import__("time").time() - t0
    print(f"[e2e] rc={p.returncode} in {dt:.0f}s")
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        return {"status": "cache_build_failed", "rc": p.returncode}

    new_pkl = out_dir / f"{seq}.pkl"
    if not new_pkl.is_file():
        return {"status": "no_new_cache"}
    stored = load_cache_detections(stored_pkl)
    new = load_cache_detections(new_pkl)
    common = sorted(set(stored) & set(new))
    if not common:
        return {"status": "no_common_frames", "stored": len(stored), "new": len(new)}

    exact = 0
    n_det_stored = n_det_new = 0
    max_pt_diff = 0.0
    max_score_diff = 0.0
    for fi in common:
        s, nw = stored[fi], new[fi]
        n_det_stored += len(s["points"])
        n_det_new += len(nw["points"])
        if len(s["points"]) == len(nw["points"]):
            same = np.allclose(s["points"], nw["points"], atol=1e-4) and \
                np.allclose(s["scores"], nw["scores"], atol=1e-5)
            if same:
                exact += 1
                continue
        if len(s["points"]) and len(nw["points"]):
            k = min(len(s["points"]), len(nw["points"]))
            max_pt_diff = max(max_pt_diff, float(np.abs(
                s["points"][:k] - nw["points"][:k]).max()))
            max_score_diff = max(max_score_diff, float(np.abs(
                s["scores"][:k] - nw["scores"][:k]).max()))
    print(f"[e2e] frames={len(common)} identical_frames={exact} "
          f"det_stored={n_det_stored} det_new={n_det_new} "
          f"max_pt_diff={max_pt_diff:.6f} max_score_diff={max_score_diff:.6f}")
    return {"status": "ok", "frames": len(common), "identical_frames": exact,
            "detections_stored": n_det_stored, "detections_new": n_det_new,
            "max_point_diff": max_pt_diff, "max_score_diff": max_score_diff,
            "stored_pkl": str(stored_pkl), "new_pkl": str(new_pkl), "seconds": dt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="VIDEO00009_19700101_001531",
                    help="sequence for the end-to-end proof (smallest golden-pool one)")
    ap.add_argument("--pixel-sample", type=int, default=4000)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--out", default=str(WORK / "anchor" / "anchor_report.json"))
    args = ap.parse_args()

    report = {"pixel": pixel_proof(args.pixel_sample)}
    if not args.skip_e2e:
        report["end_to_end"] = e2e_proof(args.seq)
    C.write_json(args.out, report)
    print(f"\n[report] -> {args.out}")

    p = report["pixel"]
    print("\n=== 锚点判定 ===")
    if p["mismatches"] == 0 and p["max_abs_diff"] == 0:
        print("  [PASS] BMP 与 JPG 解码结果逐像素完全一致 ⇒ BMP 路径等价于 memory 记录的路径")
    else:
        print(f"  [FAIL] {p['mismatches']} 个像素不一致（max={p['max_abs_diff']}）")
    e = report.get("end_to_end")
    if e:
        print(f"  [E2E] {e.get('status')}  identical_frames="
              f"{e.get('identical_frames')}/{e.get('frames')}")


if __name__ == "__main__":
    main()
