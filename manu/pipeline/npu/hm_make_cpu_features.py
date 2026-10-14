#!/usr/bin/env python3
"""
hm_make_cpu_features.py -- build the CPU authority arm's FEATURES from a
decoded video, so the ``--video`` demo path can be accuracy-reconciled.

Why this exists
---------------
The sequence demo reconciles against ``uav_gmc_median/images/val`` -- a frozen
feature set that the board can be compared to directly. A video has no such set,
so without this tool the video path could report speed but **no accuracy at
all**. This fills that gap: it runs the very same Python streaming engine the
frozen set was built with, on the very frames the board will consume.

The one rule that makes the comparison valid
--------------------------------------------
Both arms must see the SAME PIXELS. So this script reads the frames by decoding
the video file itself -- it never touches the raw sequence directory. If the
board stages its input from the same decode, then any detection difference is
attributable to the quantisation and nothing else.

Naming
------
Features are written as ``<seq>__<id>.png`` where ``id`` is the *sequence frame
number*, matching ``labels/val`` and therefore ``hm_reference.label_for``.
Decoded video frame ``i`` maps to id ``--id-base + i``.

Usage:
    python hm_make_cpu_features.py --video /tmp/c.avi --seq 34_1 --id-base 3 \\
        --out /path/featdir --list /path/list.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "data"))

from streaming_feature_pipeline import OnlineFeaturePipeline  # noqa: E402


def stack_hwc(planes: np.ndarray) -> np.ndarray:
    """[3,H,W] uint8 -> HxWx3 uint8, the builder's channel order."""
    return np.ascontiguousarray(np.transpose(planes, (1, 2, 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--id-base", type=int, default=3,
                    help="sequence id of decoded frame 0 (3 = first id that has GT)")
    ap.add_argument("--out", required=True, help="directory for <seq>__<id>.png")
    ap.add_argument("--list", required=True, help="frame list for hm_reference.py")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--png-level", type=int, default=1,
                    help="cv2.IMWRITE_PNG_LEVEL: 1=fast, 0=uncompressed(no loss, big)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] cannot open {args.video}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = OnlineFeaturePipeline()
    paths: list[str] = []
    i = 0
    while args.limit <= 0 or i < args.limit:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        planes = pipe.push(gray)
        fid = args.id_base + i
        p = out_dir / f"{args.seq}__{fid:06d}.png"
        if not cv2.imwrite(str(p), stack_hwc(planes),
                           [cv2.IMWRITE_PNG_COMPRESSION, args.png_level]):
            raise SystemExit(f"[ERROR] cannot write {p}")
        paths.append(str(p))
        if (i + 1) % 50 == 0:
            st = pipe.stats()
            print(f"[feat] {i + 1} frames  fit={st['fit_ms']:.1f}ms "
                  f"warp={st['warp_ms']:.1f}ms median={st['median_ms']:.1f}ms "
                  f"total={st['total_ms']:.1f}ms", flush=True)
        i += 1
    cap.release()

    Path(args.list).write_text("\n".join(paths) + "\n")
    st = pipe.stats()
    print(f"[feat] {len(paths)} frames -> {out_dir}")
    print(f"[feat] id range {args.id_base}..{args.id_base + len(paths) - 1}"
          f"   (decoded frame 0 <-> sequence id {args.id_base})")
    print(f"[feat] engine stage means: fit={st['fit_ms']:.2f}ms "
          f"warp={st['warp_ms']:.2f}ms median={st['median_ms']:.2f}ms "
          f"total={st['total_ms']:.2f}ms over {st['pushes']} pushes")
    print(f"[list] {args.list}")


if __name__ == "__main__":
    main()
