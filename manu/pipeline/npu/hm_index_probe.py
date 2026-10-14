#!/usr/bin/env python3
"""
hm_index_probe.py -- establish, BY MEASUREMENT, the index mapping between

  (a) a push counter into the raw-frame sequence, and
  (b) the frozen dataset's ``<sequence>__<frame number>.png`` feature names,

and report whether the Python streaming engine reproduces those frozen features
pixel-for-pixel.

Why this has to be measured instead of reasoned about
----------------------------------------------------
Three independent counters exist and none of them agree:

  * the raw sequence's own file numbers (``000001.png`` ... ``000538.png``),
  * the push index fed to OnlineFeaturePipeline / gmc_stream_ocl (``fi`` starts at 0),
  * the dataset's ``__NNNNNN`` frame id.

The frozen features start at ``__000003`` for a 538-frame raw sequence, so some
offset exists -- but guessing which one is exactly how a demo silently reconciles
against the wrong label. This script settles it by comparing pixels: for every push
index it searches the candidate dataset ids and keeps only the ones that match.

Output
------
* a mapping table (push index -> dataset frame id), and
* the residual ``max|diff|`` / mean |diff| / exact-pixel-rate per compared pair.

Usage
-----
    python hm_index_probe.py --sequence 34_1 --pushes 60
    python hm_index_probe.py --sequence 34_1 --pushes 0     # probe the whole sequence (slow)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # manu/pipeline
sys.path.insert(0, str(_HERE.parent.parent / "data"))

DEFAULT_RAW = "/home/manu/mnt/data/siping/datasets/manu/anti-uav"
DEFAULT_FEAT = "/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val"


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def stack_hwc(planes: np.ndarray) -> np.ndarray:
    """[3,H,W] uint8 -> HxWx3 uint8, matching the builder's channel order."""
    return np.ascontiguousarray(np.transpose(planes, (1, 2, 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=DEFAULT_RAW)
    ap.add_argument("--feat-root", default=DEFAULT_FEAT)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--pushes", type=int, default=60,
                    help="how many raw frames to push (0 = all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the raw frame list before pushing")
    ap.add_argument("--search-window", type=int, default=8,
                    help="candidate dataset ids to try per push (offset search radius)")
    ap.add_argument("--anchor", type=int, default=20,
                    help="a push index whose offset is already known; its match seeds the search")
    ap.add_argument("--out-json", default="", help="write the mapping + residuals here")
    args = ap.parse_args()

    from streaming_feature_pipeline import OnlineFeaturePipeline, SequenceFrameSource

    src = SequenceFrameSource(args.raw_root, args.sequence, limit=args.limit)
    raw_paths = src.paths
    print(f"[probe] sequence      : {args.sequence}")
    print(f"[probe] raw dir       : {src.directory}")
    print(f"[probe] raw frames    : {len(raw_paths)}  (first={raw_paths[0].name}, "
          f"last={raw_paths[-1].name})")

    feat_dir = Path(args.feat_root)
    dataset_ids = sorted(
        int(p.stem.split("__")[-1])
        for p in feat_dir.iterdir()
        if p.stem.startswith(f"{args.sequence}__")
    )
    if not dataset_ids:
        die(f"no frozen features for {args.sequence} under {feat_dir}")
    print(f"[probe] frozen ids    : {len(dataset_ids)}  "
          f"(min={dataset_ids[0]}, max={dataset_ids[-1]})")

    n_push = args.pushes if args.pushes > 0 else len(raw_paths)
    n_push = min(n_push, len(raw_paths))

    pipe = OnlineFeaturePipeline()
    img_cache: dict[int, np.ndarray] = {}

    def frozen(nid: int) -> np.ndarray | None:
        if nid not in img_cache:
            p = feat_dir / f"{args.sequence}__{nid:06d}.png"
            if not p.is_file():
                return None
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                return None
            img_cache[nid] = im
        return img_cache[nid]

    def compare(hwc: np.ndarray, nid: int) -> tuple[float, float, float] | None:
        ref = frozen(nid)
        if ref is None or ref.shape != hwc.shape:
            return None
        d = np.abs(ref.astype(np.int16) - hwc.astype(np.int16))
        return float(d.max()), float(d.mean()), float((d == 0).mean())

    rows: list[dict] = []
    offset: int | None = None

    for fi in range(n_push):
        frame = cv2.imread(str(raw_paths[fi]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            die(f"cannot read {raw_paths[fi]}")
        planes = pipe.push(frame)
        hwc = stack_hwc(planes)

        best = None
        if offset is None:
            # Unknown mapping: search the whole id range once, on one anchor push.
            if fi == args.anchor:
                for nid in dataset_ids:
                    r = compare(hwc, nid)
                    if r is None:
                        continue
                    mx, mean, exact = r
                    if mx == 0.0:
                        offset = nid - fi
                        best = (nid, mx, mean, exact)
                        break
                if offset is None:
                    die(f"no exact pixel match for push {fi} over {len(dataset_ids)} ids -- "
                        f"the streaming engine does not reproduce the frozen set for this sequence")
        else:
            nid = fi + offset
            r = compare(hwc, nid)
            if r is not None:
                mx, mean, exact = r
                best = (nid, mx, mean, exact)

        if best is not None:
            nid, mx, mean, exact = best
            rows.append({"push": fi, "raw": raw_paths[fi].name, "id": nid,
                         "max_abs": mx, "mean_abs": mean, "exact_px": exact})
            tag = "EXACT" if mx == 0.0 else f"max|d|={mx:.1f}"
            print(f"[probe] push {fi:4d} ({raw_paths[fi].name}) -> __ {nid:06d}  "
                  f"{tag}  mean|d|={mean:.3e}  exact_px={exact * 100:.2f}%")

        if (fi + 1) % 25 == 0:
            st = pipe.stats()
            print(f"[probe]   ...{fi + 1}/{n_push}  fit={st['fit_ms']:.1f}ms "
                  f"warp={st['warp_ms']:.1f}ms median={st['median_ms']:.1f}ms "
                  f"total={st['total_ms']:.1f}ms", flush=True)

    print("\n[probe] ===== summary =====")
    print(f"[probe] push index -> dataset id offset : {offset}  "
          f"(dataset_id = push_index + {offset})")
    if rows:
        mx = max(r["max_abs"] for r in rows)
        exact_all = all(r["max_abs"] == 0.0 for r in rows)
        print(f"[probe] compared {len(rows)} pushes, worst max|diff| = {mx}")
        print(f"[probe] pixel-exact on every push      : {exact_all}")
        if not exact_all:
            bad = [r for r in rows if r["max_abs"] != 0.0]
            print(f"[probe] non-exact pushes              : {len(bad)} / {len(rows)}")
            for r in bad[:10]:
                print(f"[probe]    push {r['push']:4d} -> id {r['id']:06d}  "
                      f"max|d|={r['max_abs']} mean|d|={r['mean_abs']:.3e} "
                      f"exact_px={r['exact_px'] * 100:.2f}%")

    st = pipe.stats()
    print(f"[probe] engine stage means over {st['pushes']} pushes: "
          f"fit={st['fit_ms']:.2f}ms warp={st['warp_ms']:.2f}ms "
          f"median={st['median_ms']:.2f}ms total={st['total_ms']:.2f}ms "
          f"(fits/push={st['fits'] / max(1, st['pushes']):.2f})")

    if args.out_json:
        import json
        Path(args.out_json).write_text(json.dumps({
            "sequence": args.sequence,
            "offset_push_to_id": offset,
            "rows": rows,
            "stage_ms": st,
        }, indent=2))
        print(f"[probe] wrote {args.out_json}")


if __name__ == "__main__":
    main()
