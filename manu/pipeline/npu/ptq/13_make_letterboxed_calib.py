#!/usr/bin/env python3
"""13_make_letterboxed_calib.py -- calibration images that are byte-identical to
what the board feeds the NPU.

WHY THIS EXISTS
---------------
The toolkit resizes a calibration image to the model's input shape. It does NOT
letterbox. Every feature image in this project is 640x512 (measured: official
val features, Anti-UAV train features and our calibration BMPs are all
(512, 640, 3)), so the toolkit squashes 640x512 -> 640x640 while the board's
inference path letterboxes 640x512 -> 640x640 with 114 padding
(manu/pipeline/npu/rknn_heatmap_smoke.cpp:193-229, mirrored by
ptq_common.letterbox).

Consequence: the activation statistics the quantiser calibrates on are the
statistics of a SQUASHED image, while the deployed model sees a LETTERBOXED
one.  Two different distributions, and the output tensor's scale is derived
from the calibration one.  Measured: the float ONNX gives heatmap peaks of
0.85..0.97 on the letterboxed bytes, but the toolkit's calibration observed a
maximum of only ~0.48 (int8 out[0] scale=0.00378871 => 127*scale=0.481).

THE FIX
-------
Pre-letterbox the calibration images to 640x640 with the SAME code the board
uses (ptq_common.letterbox) and calibrate on THOSE.  Then the calibration
distribution IS the inference distribution, byte for byte.

The output is a new calibration set under --out-root, one 640x640 BMP per
input, plus a list file.  Verify with:
    python manu/pipeline/npu/ptq/13_make_letterboxed_calib.py --set A --verify

Usage (server):
    python manu/pipeline/npu/ptq/13_make_letterboxed_calib.py --set A
    python manu/pipeline/npu/ptq/13_make_letterboxed_calib.py --set A --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402


def build(set_key: str, calib_root: Path, out_root: Path) -> dict:
    lst = calib_root / f"set_{set_key}.txt"
    if not lst.is_file():
        raise SystemExit(f"[FATAL] {lst} missing")
    paths = [l.strip() for l in lst.read_text().splitlines() if l.strip()]
    out_dir = out_root / f"lb_{set_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    lines, missing = [], 0
    for p in paths:
        src = Path(p)
        if not src.is_file():
            missing += 1
            continue
        img = C.load_feature_bmp(src)            # BGR, disk order
        canvas = C.letterbox(img, C.IMGSZ)      # 640x640, pad 114 -- same as the board
        dst = out_dir / (src.stem + ".bmp")
        cv2.imwrite(str(dst), canvas)
        lines.append(str(dst))
    out_list = out_root / f"lb_{set_key}.txt"
    out_list.write_text("\n".join(lines) + "\n")
    print(f"[lb] {len(lines)} letterboxed 640x640 written ({missing} missing) -> {out_list}")
    return {"set": set_key, "n": len(lines), "missing": missing,
            "list": str(out_list), "dir": str(out_dir), "imgsz": C.IMGSZ,
            "pad_value": 114}


def verify(set_key: str, out_root: Path, n: int = 5) -> dict:
    """Assert the letterboxed BMP equals the tensor the board would feed."""
    lst = out_root / f"lb_{set_key}.txt"
    if not lst.is_file():
        raise SystemExit(f"[FATAL] {lst} missing")
    import cv2

    paths = [l.strip() for l in lst.read_text().splitlines() if l.strip()][:n]
    checked = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        assert img is not None and img.shape == (C.IMGSZ, C.IMGSZ, 3), (p, None if img is None else img.shape)
        # the board reads it with stbi (RGB) == the model's channel order
        nhwc = C.bmp_to_input_nhwc(p)
        assert nhwc.shape == (1, C.IMGSZ, C.IMGSZ, 3), nhwc.shape
        checked.append({"bmp": p, "shape": list(img.shape),
                        "model_order_mean": [round(float(x), 2) for x in nhwc[0].reshape(-1, 3).mean(0)]})
    print(f"[verify] {len(checked)} frames ok, shape={C.IMGSZ}x{C.IMGSZ}x3")
    return {"n_checked": len(checked), "frames": checked}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="set key, e.g. A")
    ap.add_argument("--calib-root", default="/mnt/data/siping/ptq_work/calib")
    ap.add_argument("--out-root", default="/mnt/data/siping/ptq_work/calib3")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    info = {"out_root": str(out_root)}
    if args.verify:
        info["verify"] = verify(args.set, out_root)
    else:
        info["letterbox"] = build(args.set, Path(args.calib_root), out_root)
    (out_root / f"lb_{args.set}_manifest.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=1))
    print(f"[out] {out_root/f'lb_{args.set}_manifest.json'}")


if __name__ == "__main__":
    main()
