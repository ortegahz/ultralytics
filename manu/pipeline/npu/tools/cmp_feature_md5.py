#!/usr/bin/env python3
"""Compare the per-frame GMC feature fingerprints of the three producers.

    server  : runs/pyref/py_step2_300.md5      (frozen Python OnlineFeaturePipeline)
    offline : /tmp/s2_cpu.md5                  (board gmc_stream_ocl --fused cpu)
    service : <out>/feature_md5.txt            (board rknn_mosaic_e2e, GPU tail)

All three write the SAME line format:  <seq>\t<frame index>\t<md5 hex>
and the SAME fingerprint recipe:  gmcpp::MD5 over the PRE-ENCODE 3 x (H*W)
channel planes.  So the files are comparable line by line, and a differing
line names the exact frame that diverged.

Usage:
    cmp_feature_md5.py REF FILE [FILE ...]
"""
from __future__ import annotations

import sys
from pathlib import Path


def load(path: str) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise SystemExit(f"[BAD LINE] {path}:{ln}: {line!r}")
            out[int(parts[1])] = parts[2]
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    ref_path, paths = argv[1], argv[2:]
    ref = load(ref_path)
    print(f"[REF] {ref_path}: {len(ref)} frames")
    rc = 0
    for path in paths:
        cur = load(path)
        common = sorted(set(ref) & set(cur))
        same = [i for i in common if ref[i] == cur[i]]
        diff = [i for i in common if ref[i] != cur[i]]
        only_ref = sorted(set(ref) - set(cur))
        only_cur = sorted(set(cur) - set(ref))
        print(f"\n[CMP] {path}")
        print(f"  frames: ref={len(ref)} cur={len(cur)} common={len(common)}")
        print(f"  SAME: {len(same)}   DIFF: {len(diff)}")
        if diff:
            pct = 100.0 * len(diff) / max(1, len(common))
            print(f"  diff rate: {pct:.2f}%")
            print(f"  first 5 differing frames: {diff[:5]}")
            print(f"  last differing frame: {diff[-1]}")
            print(f"  sample @ {diff[0]}: ref={ref[diff[0]]}  cur={cur[diff[0]]}")
        else:
            print("  ==> BIT-IDENTICAL to the reference")
        if only_ref:
            print(f"  frames only in ref: {len(only_ref)} (first={only_ref[0]})")
        if only_cur:
            print(f"  frames only in cur: {len(only_cur)} (first={only_cur[0]})")
        if diff or only_ref or only_cur:
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
