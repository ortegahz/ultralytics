#!/usr/bin/env python3
"""27_check_calib_leak.py -- PROVE that a calibration set contains no evaluation frame.

WHY THIS SCRIPT EXISTS
----------------------
On 2026-10-23 the calibration-set v2 family (set_1 / set_2 / set_3) was found
to be built from frames that also appear in the evaluation tracks:

    set_1 : 120 x 001324 (T3 blind) + 114 x 000818 + 10 x 000703 + ... (T2)
            -> 250 / 250 frames are evaluation frames
    set_2 : 200 x train + 50 x (T2/T3)
    set_3 : 85 x train + 85 x Anti-UAV train + 67 x (T2/T3)

The quantiser's activation range IS the range of the calibration frames, so a
calibration set that has seen an evaluation frame has seen that frame's
activation distribution => the int8 F1 on that track is optimistic and the arm
cannot be compared with a clean one. On the board, set_3 and set_A came out
IDENTICAL on 001324 (best F1 0.7090 @ 0.45 both) although the PC simulator had
put set_3 +0.050 ahead -- so the simulator's advantage was not real.

Two checks, both cheap:
  1. PATH check  -- a calibration frame must not live under an evaluation
     split ("/images/val/", "/labels/val/") and must not be under the val-side
     BMP pool ($W/bmp/).
  2. SEQUENCE check -- the sequence names of the calibration frames must not
     intersect the sequence lists of the three evaluation manifests.

Usage (server):
    python manu/pipeline/npu/ptq/27_check_calib_leak.py \\
        --calib $W/calib2/set_3.txt \\
        --calib-sources $W/calib2/set_3_sources.txt \\
        --manifests $W/fast/anti_uav_sub2500.json $W/fast/001324_sub2000.json \\
                  $W/fast/golden_pool_sub1200.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# Evaluation-side path fragments. The calibration may only use TRAIN splits.
EVAL_PATH_MARKERS = ("/images/val/", "/labels/val/", "/bmp/", "/pcfeat_bmp/")
# The val-side BMP pool the evaluation frames live in (absolute prefix).
EVAL_BMP_ROOTS = ("/mnt/data/siping/ptq_work/bmp/",)


def seq_of(path: str) -> str:
    """Sequence name of a feature file, for both naming conventions."""
    name = Path(path).name
    stem = name.rsplit(".", 1)[0]
    if "__" in stem:
        return stem.rsplit("__", 1)[0]
    if "_frame_" in stem:
        return stem.split("_frame_", 1)[0]
    return stem


def load_manifest_seqs(path: Path) -> set[str]:
    d = json.loads(path.read_text())
    rows = d["items"] if isinstance(d, dict) and "items" in d else d
    out = set()
    for r in rows:
        seq = r.get("seq")
        if seq:
            out.add(str(seq))
        else:
            bmp = r.get("bmp") or ""
            if bmp:
                out.add(seq_of(bmp))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--calib-sources", default="")
    ap.add_argument("--manifests", nargs="+", default=[])
    ap.add_argument("--allow-seq", nargs="*", default=[],
                    help="sequence names that are legitimately shared")
    args = ap.parse_args()

    calib = [l.strip() for l in Path(args.calib).read_text().splitlines() if l.strip()]
    src_path = Path(args.calib_sources) if args.calib_sources else None
    srcs = ([l.strip() for l in src_path.read_text().splitlines() if l.strip()]
            if src_path and src_path.is_file() else [])

    print(f"[calib] {args.calib}: {len(calib)} frames"
          + (f"  (+{len(srcs)} source paths)" if srcs else ""))

    # ---- check 1: path-level --------------------------------------------------
    bad_paths = []
    for p in calib + srcs:
        low = p.lower()
        if any(m in low for m in EVAL_PATH_MARKERS) or any(low.startswith(r) for r in EVAL_BMP_ROOTS):
            bad_paths.append(p)
    print(f"[paths] evaluation-side frames in the calibration: {len(bad_paths)}")
    for p in bad_paths[:5]:
        print(f"        {p}")

    # ---- check 2: sequence-level ---------------------------------------------
    eval_seqs: dict[str, set[str]] = {}
    for m in args.manifests:
        mp = Path(m)
        if not mp.is_file():
            print(f"[manifest] missing: {mp}")
            continue
        s = load_manifest_seqs(mp)
        eval_seqs[mp.stem] = s
        print(f"[manifest] {mp.name}: {len(s)} sequences")

    calib_seqs = {seq_of(p) for p in (srcs or calib)}
    allow = set(args.allow_seq)
    hits = []
    for name, s in eval_seqs.items():
        inter = (calib_seqs & s) - allow
        if inter:
            hits.append((name, inter))
        print(f"[seq] vs {name}: {len(inter)} shared sequences"
              + (f"  -> {sorted(inter)[:6]}" if inter else ""))

    ok = (not bad_paths) and (not hits)
    print()
    print("[leak] %s  (%d path hits, %d sequence hits)" %
          ("PASS -- the calibration set is clean" if ok else "FAIL -- LEAKED", len(bad_paths), len(hits)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
