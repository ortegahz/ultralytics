#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Build several RKNN variants of the stock YOLO26n and measure which build knob
actually moves the board's NPU time.

Why this file exists
--------------------
The board run showed stock YOLO26n int8 at ~23.7 ms per rknn_run. For a
6.1 GFLOP model on a 3-core NPU at 1 GHz that is far below the silicon's
rated throughput, so the question is not "which architecture" but "which build
knob". The toolkit exposes exactly the knobs that matter for a bandwidth- or
compute-bound graph:

    core_mask      how the graph is split across the three NPU cores
                   (NPU_CORE_0_1_2 = 0x7 splits it; AUTO = 0 lets the
                   runtime decide, and on this board it lands on one core)
    sram_prefer    keep operands in the NPU's on-chip SRAM instead of DDR
    compress_weight hardware weight decompression (less DDR read traffic)
    optimization_level  depth of operator fusion (3 = toolkit default)

The point of building them as ARMS in one script is that each arm differs
from the baseline in exactly one knob, so the measured delta is attributable.
Every arm writes its .rknn plus the exact config that produced it, so the run
is reproducible without reading this file.

Usage:
    python yolo26n_speed_ab.py \
        --onnx /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.onnx \
        --outdir /media/manu/1TB-Volume/rknn_work/yolo26n/arms \
        --calib-list /media/manu/1TB-Volume/rknn_work/yolo26n/calib_train_features.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ARMS = [
    # name,          config kwargs
    ("baseline", {}),
    ("sram", {"sram_prefer": True}),
    ("compress", {"compress_weight": True}),
    ("sram_compress", {"sram_prefer": True, "compress_weight": True}),
    # NOTE: nbuf_prefer takes a BOOL in rknn-toolkit2 2.3.0 -- passing the
    # string "high_perf" raises
    #   ValueError: The value of nbuf_prefer only support <class 'bool'> !
    ("nbuf", {"nbuf_prefer": True}),
    ("sram_nbuf", {"sram_prefer": True, "nbuf_prefer": True}),
]

# NOTE on core_mask: in rknn-toolkit2 2.3.0 the Python API does NOT expose a
# build-time core mask -- `core_mask` belongs to `init_runtime()`, which only
# matters when the HOST drives a connected device. `config(single_core_mode=)`
# is the only core-related build knob (default False = multi-core capable).
# The runtime-side `rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2)` was tried on
# the board and CRASHED the RKNPU driver (job timeout -> soft reset -> SIGILL),
# so it is explicitly not part of this sweep.


def md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--calib-list", required=True)
    ap.add_argument("--only", default="", help="comma-separated arm names to build (default: all)")
    args = ap.parse_args()

    from rknn.api import RKNN

    onnx = Path(args.onnx).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    calib = Path(args.calib_list).resolve()
    if not onnx.is_file():
        sys.exit(f"[FATAL] missing ONNX: {onnx}")
    if not calib.is_file():
        sys.exit(f"[FATAL] missing calibration list: {calib}")

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    manifest = []
    for name, cfg_kw in ARMS:
        if wanted and name not in wanted:
            continue
        out = outdir / f"yolo26n_{name}.rknn"
        log = outdir / f"build_{name}.log"
        print(f"\n================ arm: {name} ================")
        print(f"  config kwargs: {cfg_kw}")

        rknn = RKNN(verbose=True)
        cfg = dict(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform="rk3588",
            optimization_level=3,
        )
        cfg.update(cfg_kw)
        rknn.config(**cfg)
        ret = rknn.load_onnx(model=str(onnx))
        if ret != 0:
            sys.exit(f"[FATAL] {name}: load_onnx failed ret={ret}")
        ret = rknn.build(do_quantization=True, dataset=str(calib))
        if ret != 0:
            sys.exit(f"[FATAL] {name}: build failed ret={ret}")
        ret = rknn.export_rknn(str(out))
        if ret != 0:
            sys.exit(f"[FATAL] {name}: export_rknn failed ret={ret}")
        rknn.release()

        if not out.is_file():
            sys.exit(f"[FATAL] {name}: export reported success but {out} is absent")
        with open(out, "rb") as f:
            magic = f.read(4)
        if magic != b"RKNN":
            sys.exit(f"[FATAL] {name}: {out} is not an RKNN file (magic={magic!r})")

        entry = {
            "arm": name,
            "rknn": str(out),
            "bytes": out.stat().st_size,
            "md5": md5(out),
            "config_kwargs": {k: v for k, v in cfg_kw.items()},
            "onnx": str(onnx),
            "onnx_md5": md5(onnx),
            "calib_list": str(calib),
            "calib_count": sum(1 for _ in open(calib) if _.strip()),
        }
        manifest.append(entry)
        print(f"  -> {out}  {entry['bytes'] / 1e6:.2f} MB  md5={entry['md5']}")

    mf = outdir / "manifest.json"
    mf.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[OK] manifest -> {mf}")
    for e in manifest:
        print(f"  {e['arm']:<12} {e['bytes'] / 1e6:>6.2f} MB  md5={e['md5']}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
