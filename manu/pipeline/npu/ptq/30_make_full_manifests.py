#!/usr/bin/env python3
"""30_make_full_manifests.py -- FULL-track manifests for the three evaluation tracks.

WHY THIS SCRIPT EXISTS
----------------------
The threshold sweep (19_board_threshold_sweep.py) scores the board TSVs
against the ROWS OF THE MANIFEST, not against every frame in the TSV. The
"fast" manifests are subsets (001324_sub2000 = 2000 rows, golden_pool_sub1200,
anti_uav_sub2500), so sweeping a FULL-track TSV with a subset manifest silently
produces a SUBSET score -- and that is how a "+0.0137" subset reading became
"the full-track number" in one table.

The frame lists are taken from the full-scale driver's own collector
(14_run_full_scale_nofs.collect_rows), so the manifests are exactly the frames
the 60,610-frame run scored: 11,810 / 17,187 / 31,613.

Usage (server):
    python manu/pipeline/npu/ptq/30_make_full_manifests.py --out /mnt/data/siping/ptq_work/fast_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

# The full-scale driver starts with a digit, so it cannot be imported by name;
# load it from its path instead (same trick the smoke binary uses).
import importlib.util  # noqa: E402

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "ptq_full_scale", _here / "14_run_full_scale_nofs.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.WORK_ROOT / "fast_full"))
    ap.add_argument("--bmp-root", default=str(C.BMP_ROOT))
    ap.add_argument("--anti-root", default=str(C.WORK_ROOT / "pcfeat_bmp" / "images"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = {
        "blind": Path(args.bmp_root),
        "golden": Path(args.bmp_root),
        "anti": Path(args.anti_root),
    }
    for track, root in roots.items():
        gt_root = F.DEFAULT_GT_ROOT.get(track)
        rows = F.collect_rows(track, root, None, gt_root)
        mf = out / f"{track}_full.json"
        mf.write_text(json.dumps(rows, ensure_ascii=False))
        print(f"[full] {track:7s} {len(rows):6d} rows -> {mf}")


if __name__ == "__main__":
    main()
