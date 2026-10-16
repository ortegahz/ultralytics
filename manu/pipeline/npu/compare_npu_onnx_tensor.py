#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
compare_npu_onnx_tensor.py -- NPU output tensors vs the ORIGINAL ONNX graph.

Why this file exists
--------------------
The board dumps the exact uint8 NHWC buffer it fed into the NPU
(`--dump-input`, 640x640x3) together with the float32 outputs it read back
(`--dump`, `_hm.bin` / `_off.bin`). Feeding that *same* buffer into onnxruntime
turns "the NPU model is faithful to the source graph" into a measurement
instead of an assumption: no re-letterboxing, no re-decoding, no preprocessing
drift can hide inside the number.

Input convention (must match hm_common.to_input_tensor and the rknn config):
    uint8 NHWC [640,640,3]  ->  NCHW  ->  /255.0  ->  float32 [1,3,640,640]
The RKNN `std_values=[[255,255,255]]` undoes that /255 on the board, so both
sides must start from identical bytes.

Usage:
  python compare_npu_onnx_tensor.py \
      --onnx trial0474_hm_nop0.onnx \
      --input-dir f16i --hm-dir f16d --off-dir f16d --tag f16 \
      --out-json cmp_f16.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

# The board names its files <tag>_<stem>_{hm,off}.bin; the input dumps are
# <stem>.input.bin. Strip the tag prefix to recover the shared stem.
def _stem_of(path: Path, tag: str) -> str:
    name = path.name
    for marker in (".input.bin", "_hm.bin", "_off.bin"):
        if name.endswith(marker):
            base = name[: -len(marker)]
            break
    else:
        raise ValueError(f"unrecognised dump filename: {path.name}")
    pref = f"{tag}_"
    base = base[len(pref):] if base.startswith(pref) else base
    # The input dump keeps the extension (dots -> underscores, so "x.bmp" ->
    # "x_bmp") while the output dump strips those trailing 4 chars. Realign.
    for ext in ("_bmp", "_jpg", "_png", "_jpeg"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


def load_input(path: Path) -> np.ndarray:
    """uint8 NHWC [640,640,3] -> float32 NCHW [1,3,640,640] in [0,1]."""
    a = np.fromfile(path, dtype=np.uint8)
    if a.size != 640 * 640 * 3:
        raise SystemExit(f"[FATAL] {path.name}: {a.size} B, expected {640*640*3}")
    return np.ascontiguousarray(a.reshape(640, 640, 3).transpose(2, 0, 1)[None]
                                .astype(np.float32) / 255.0)


def load_out(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(shape)


def stats(got: np.ndarray, ref: np.ndarray) -> dict:
    d = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-6)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "max_rel": float((d / denom).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input-dir", required=True, help="dir of *.input.bin")
    ap.add_argument("--hm-dir", required=True)
    ap.add_argument("--off-dir", required=True)
    ap.add_argument("--tag", required=True, help="file-name prefix used on the board")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--tol-hm", type=float, default=2e-2,
                    help="gate on max_abs for the heatmap")
    ap.add_argument("--tol-off", type=float, default=2e-1,
                    help="gate on max_abs for the offset (regression target, "
                         "deeper range, no sigmoid)")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    ind = Path(args.input_dir)
    rows = []
    for ip in sorted(ind.glob("*.input.bin")):
        stem = _stem_of(ip, args.tag)
        hm_b = Path(args.hm_dir) / f"{args.tag}_{stem}_hm.bin"
        off_b = Path(args.off_dir) / f"{args.tag}_{stem}_off.bin"
        if not hm_b.is_file() or not off_b.is_file():
            raise SystemExit(f"[FATAL] missing output dump for {stem}")

        x = load_input(ip)
        ref_hm, ref_off = sess.run(["heatmap", "offset"], {"images": x})
        got_hm = load_out(hm_b, (1, 1, 320, 320))
        got_off = load_out(off_b, (1, 2, 320, 320))

        row = {
            "stem": stem,
            "heatmap": stats(got_hm, ref_hm),
            "offset": stats(got_off, ref_off),
            # Peak agreement: the heatmap peak is what the detector actually
            # reads out, so a matching argmax matters more than a global mean.
            "hm_peak_ref": float(ref_hm.max()),
            "hm_peak_npu": float(got_hm.max()),
            "hm_peak_d": float(got_hm.max() - ref_hm.max()),
            "hm_argmax_same": bool(np.unravel_index(np.argmax(got_hm), got_hm.shape)
                                   == np.unravel_index(np.argmax(ref_hm), ref_hm.shape)),
        }
        rows.append(row)
        print(f"[{args.tag}] {stem:<30} "
              f"hm max_abs={row['heatmap']['max_abs']:.3e} mean={row['heatmap']['mean_abs']:.3e} "
              f"peak ref={row['hm_peak_ref']:.4f} npu={row['hm_peak_npu']:.4f} "
              f"argmax_same={row['hm_argmax_same']} | "
              f"off max_abs={row['offset']['max_abs']:.3e} mean={row['offset']['mean_abs']:.3e}")

    if not rows:
        raise SystemExit("[FATAL] no frames compared")

    agg = {}
    for key in ("heatmap", "offset"):
        agg[key] = {
            "max_abs_worst": max(r[key]["max_abs"] for r in rows),
            "mean_abs_avg": float(np.mean([r[key]["mean_abs"] for r in rows])),
            "max_rel_worst": max(r[key]["max_rel"] for r in rows),
        }
    agg["hm_peak_d_worst"] = max(abs(r["hm_peak_d"]) for r in rows)
    agg["hm_argmax_agree"] = sum(r["hm_argmax_same"] for r in rows)

    tol_hm, tol_off = args.tol_hm, args.tol_off
    ok = (agg["heatmap"]["max_abs_worst"] <= tol_hm
          and agg["offset"]["max_abs_worst"] <= tol_off)

    print("=" * 78)
    print(f"[{args.tag}] frames={len(rows)}")
    for key in ("heatmap", "offset"):
        print(f"  {key:<8} worst_max_abs={agg[key]['max_abs_worst']:.6e} "
              f"avg_mean_abs={agg[key]['mean_abs_avg']:.6e} "
              f"worst_max_rel={agg[key]['max_rel_worst']:.6e}")
    print(f"  heatmap peak: worst |delta|={agg['hm_peak_d_worst']:.6e}  "
          f"argmax agree={agg['hm_argmax_agree']}/{len(rows)}")
    print(f"  GATE tol_hm={tol_hm:.1e} tol_off={tol_off:.1e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    print("=" * 78)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(
            {"tag": args.tag, "onnx": args.onnx, "n_frames": len(rows),
             "aggregate": agg, "tol_hm": tol_hm, "tol_off": tol_off,
             "gate": "PASS" if ok else "FAIL", "per_frame": rows},
            indent=2))
        print(f"[WROTE] {args.out_json}")

    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
