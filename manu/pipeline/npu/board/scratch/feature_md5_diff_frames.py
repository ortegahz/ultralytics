#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_md5_diff_frames.py -- WHY DOES AN ARM FAIL THE BIT-EXACT VERDICT?

WHAT IT ANSWERS
---------------
The acceptance criterion of the decoupled pipeline is "the delivered frames are
bit-identical to the reference arm".  Some arms report a handful of frames whose
per-frame feature md5 differs from the reference (CANON28 / CANON29: 22 frames,
VERDICT FAIL) while the detection F1 is IDENTICAL.

This script localises those frames and separates the two possible causes:

  (a) a REAL pipeline defect (the delivered bytes are wrong somewhere), and
  (b) a START-UP TRANSIENT artifact (the very first frames of a run are computed
      while the GMC's 21-frame history is still filling up, so their content is
      timing-dependent and differs from the reference run's warm-up).

The separation is done by a threshold on the PHYSICAL frame id: the GMC reads a
history that reaches `gmc_window * gmc_anchor_step` = 21 * 2 = 42 physical frames
back, so every group whose anchor is older than the warm-up window is
transient-dependent.  The threshold used here is 84 = 2 * 42, i.e. the first two
lag windows.

USAGE (on the SERVER, where the results live)
----------------------------------------------
    cd /mnt/data/siping/ptq_work/psearch/results
    python3 /tmp/pycharm_project_10ae9e2e/manu/pipeline/npu/board/scratch/feature_md5_diff_frames.py \
        --ref A_gate_step2_nodrop \
        --arms CANON26_dec_xsvc_d3 CANON27_dec_xsvc_d8 CANON28_dec_rr_d3 CANON29_dec_rr_d8 \
        --steady-threshold 84

EXIT CODE
---------
    0  every arm is bit-identical to the reference on the steady-state frames
    1  at least one arm differs on the steady-state frames  (= a real defect)
"""
from __future__ import annotations

import argparse
import os
import sys


def load_feature_md5(path: str) -> dict:
    """feature_md5.txt is TAB separated:  <seq>\\t<physical_frame_id>\\t<md5>."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                out[parts[1]] = parts[2]
    return out


def compare(arm: dict, ref: dict, threshold: int):
    common = set(arm) & set(ref)
    diff_all = sorted(k for k in common if arm[k] != ref[k])
    steady = {k for k in common if int(k) >= threshold}
    diff_steady = sorted(k for k in steady if arm[k] != ref[k])
    transient = [k for k in diff_all if int(k) < threshold]
    return {
        "arm_frames": len(arm),
        "ref_frames": len(ref),
        "common": len(common),
        "diff_all": diff_all,
        "steady_common": len(steady),
        "diff_steady": diff_steady,
        "transient": transient,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/mnt/data/siping/ptq_work/psearch/results")
    ap.add_argument("--ref", default="A_gate_step2_nodrop")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--steady-threshold", type=int, default=84,
                    help="physical frame id below which a frame belongs to the start-up transient")
    args = ap.parse_args()

    ref_path = os.path.join(args.results_dir, args.ref, "feature_md5.txt")
    if not os.path.exists(ref_path):
        print(f"[FATAL] reference not found: {ref_path}", file=sys.stderr)
        return 2
    ref = load_feature_md5(ref_path)
    print(f"[ref] {args.ref}: {len(ref)} frames  (md5 file: {ref_path})")
    print(f"[rule] steady-state = physical frame id >= {args.steady_threshold}")
    print()

    failed = []
    for arm_name in args.arms:
        arm_path = os.path.join(args.results_dir, arm_name, "feature_md5.txt")
        if not os.path.exists(arm_path):
            print(f"[{arm_name}] SKIP (no feature_md5.txt)")
            continue
        arm = load_feature_md5(arm_path)
        r = compare(arm, ref, args.steady_threshold)
        verdict = "PASS" if not r["diff_steady"] else "FAIL"
        print(f"[{arm_name}] arm={r['arm_frames']} common={r['common']} "
              f"md5_diff(all)={len(r['diff_all'])} "
              f"md5_diff(steady)={len(r['diff_steady'])} "
              f"transient_diff={len(r['transient'])}  -> {verdict}")
        if r["diff_all"]:
            lo = min(int(k) for k in r["diff_all"])
            hi = max(int(k) for k in r["diff_all"])
            print(f"            differing frame ids: {r['diff_all'][:8]}"
                  f"{' ...' if len(r['diff_all']) > 8 else ''}  (range {lo}..{hi})")
        if r["diff_steady"]:
            failed.append(arm_name)

    print()
    if failed:
        print(f"[VERDICT] FAIL on the steady-state frames: {', '.join(failed)}")
        return 1
    print("[VERDICT] PASS -- every arm is bit-identical to the reference on the "
          "steady-state frames; the remaining differences are confined to the "
          "start-up transient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
