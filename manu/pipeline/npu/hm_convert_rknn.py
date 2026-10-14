#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Convert the Trial-0474 heatmap ONNX into an RK3588 .rknn.

Calibration data policy (explicit, because it is the single most consequential
choice in this script):

  * The calibration set is built from `uav_gmc_median/images/{split}` -- these
    are ALREADY-BUILT 3-channel FEATURE images
        ch0 = I_t            (raw IR grayscale)
        ch1 = |I_t - W(I_{t-2})|   GMC short difference
        ch2 = (I_t - B_t)^+        21-frame temporal median residual
    Feeding raw IR frames here would calibrate the quantizer on a completely
    different input distribution than the one the model ever sees, and the
    resulting model would quantize fine and detect fine on paper while quietly
    losing the sub-threshold pulses the heatmap head lives on.

  * The split defaults to `train`, never `val`. Quantization is not a metric,
    but calibrating on the evaluation images is avoidable contamination.

  * Sampling is stratified across sequences and deterministic (sorted + fixed
    stride), so the calibration set is reproducible and auditable; the exact
    list is written next to the model.

Input quantization: mean 0 / std 255, i.e. the graph consumes plain uint8 RGB
and RKNN applies (x - 0) / 255. That is bit-identical to what PyTorch consumes
(uint8/255), so the host and the board see the same numbers and any difference
downstream is attributable to the NPU alone.

Usage:
    python hm_convert_rknn.py --out models/trial0474_hm_int8.rknn
    python hm_convert_rknn.py --out models/trial0474_hm_fp16.rknn --no-quant
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_ROOT = Path("/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median")
DEFAULT_ONNX = REPO_ROOT / "runs/onnx/trial0474_hm.onnx"
WORK_ROOT = Path("/media/manu/1TB-Volume/rknn_work/hm")


def seq_of(path: Path) -> str:
    return re.sub(r"__\d+$", "", path.stem)


def build_calib_list(split: str, n: int, out_file: Path, seed: int = 0) -> list[Path]:
    """Deterministic, sequence-stratified sample of pre-built feature JPEGs."""
    img_dir = DATASET_ROOT / "images" / split
    if not img_dir.is_dir():
        sys.exit(f"[FATAL] no such directory: {img_dir}")
    by_seq: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            by_seq[seq_of(p)].append(p)
    if not by_seq:
        sys.exit(f"[FATAL] {img_dir} holds no images")

    seqs = sorted(by_seq)
    rng = random.Random(seed)
    picked: list[Path] = []
    # round-robin over sequences so every recording contributes
    order = list(range(len(seqs)))
    rng.shuffle(order)
    idx_in_seq = {seqs[i]: 0 for i in order}
    while len(picked) < n:
        progressed = False
        for i in order:
            if len(picked) >= n:
                break
            s = seqs[i]
            lst = by_seq[s]
            k = idx_in_seq[s]
            if k >= len(lst):
                continue
            picked.append(lst[k])
            idx_in_seq[s] = k + 1
            progressed = True
        if not progressed:
            break
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(str(p) for p in picked) + "\n")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--out", default=str(WORK_ROOT / "models/trial0474_hm_int8.rknn"))
    ap.add_argument("--calib-split", default="train")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--calib-list", default=str(WORK_ROOT / f"calib_{'train'}_features.txt"))
    ap.add_argument("--no-quant", action="store_true", help="fp16 arm (no calibration)")
    ap.add_argument("--optimization-level", type=int, default=3,
                    help="rknn.config optimization_level (3 = deepest fusion; "
                         "NOTE: this is already the toolkit default, see "
                         "rknn/api/rknn.py)")
    ap.add_argument("--compress-weight", action="store_true",
                    help="hardware weight decompression (less DDR read traffic)")
    ap.add_argument("--sram-prefer", action="store_true",
                    help="rknn.config sram_prefer (bool): bias operand placement "
                         "toward the NPU's on-chip SRAM instead of DDR")
    ap.add_argument("--nbuf-prefer", default=None,
                    help="rknn.config nbuf_prefer (none/high_perf/low_power)")
    ap.add_argument("--enable-flash-attention", action="store_true",
                    help="rknn.config enable_flash_attention (bool)")
    ap.add_argument("--single-core-mode", action="store_true",
                    help="rknn.config single_core_mode")
    args = ap.parse_args()

    from rknn.api import RKNN

    if not Path(args.onnx).is_file():
        sys.exit(f"[FATAL] missing ONNX: {args.onnx} -- run hm_export_onnx.py first")
    out_abs = Path(args.out).resolve()
    out_abs.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(out_abs.parent)  # keep any toolkit scratch files next to the model

    print(f"[onnx ] {args.onnx} ({os.path.getsize(args.onnx) / 1e6:.2f} MB)")
    print(f"[out  ] {out_abs}")
    print(f"[quant] {'NO (fp16 comparison arm)' if args.no_quant else 'int8'}")

    calib = []
    if not args.no_quant:
        calib = build_calib_list(args.calib_split, args.calib_n, Path(args.calib_list))
        seqs = {seq_of(p) for p in calib}
        print(f"[calib] {len(calib)} pre-built FEATURE images from split '{args.calib_split}' "
              f"across {len(seqs)} sequences")
        print(f"[calib] list -> {args.calib_list}")
        for p in calib[:3]:
            print(f"[calib]   e.g. {p}")

    rknn = RKNN(verbose=True)

    # These knobs exist to be measured, not assumed:
    #   optimization_level    : depth of operator fusion (toolkit default is 3).
    #   compress_weight       : hardware weight decompression; trades a little
    #                          compute for less DDR read traffic.
    #   sram_prefer           : how much operand stays in the NPU's on-chip SRAM
    #                          rather than spilling to DDR. This is the knob that
    #                          matters when the model is bandwidth-bound.
    #   enable_flash_attention: the graph contains an SDPA block; letting the
    #                          compiler map it onto the NPU's fused attention.
    print("--> Config model (mean 0 / std 255 => model consumes uint8 RGB)")
    cfg = dict(mean_values=[[0, 0, 0]],
               std_values=[[255, 255, 255]],
               target_platform="rk3588",
               optimization_level=args.optimization_level,
               compress_weight=args.compress_weight,
               enable_flash_attention=args.enable_flash_attention,
               single_core_mode=args.single_core_mode)
    if args.sram_prefer:
        cfg["sram_prefer"] = True
    if args.nbuf_prefer:
        cfg["nbuf_prefer"] = args.nbuf_prefer
    print("     " + ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "mean_values"))
    rknn.config(**cfg)
    print("done")

    print("--> Loading ONNX")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        sys.exit(f"[FATAL] load_onnx failed ret={ret}")
    print("done")

    print("--> Building")
    ret = rknn.build(do_quantization=not args.no_quant,
                     dataset=args.calib_list if calib else None)
    if ret != 0:
        sys.exit(f"[FATAL] build failed ret={ret}")
    print("done")

    print("--> Exporting")
    ret = rknn.export_rknn(str(out_abs))
    if ret != 0:
        sys.exit(f"[FATAL] export_rknn failed ret={ret}")
    rknn.release()
    print("done")

    if not out_abs.is_file():
        sys.exit(f"[FATAL] export reported success but {out_abs} is absent")
    with open(out_abs, "rb") as f:
        head = f.read(4)
    if head != b"RKNN":
        sys.exit(f"[FATAL] {out_abs} does not start with the RKNN magic (got {head!r})")
    print(f"\n[OK] {out_abs}  {out_abs.stat().st_size / 1e6:.2f} MB  magic=RKNN")


if __name__ == "__main__":
    main()
