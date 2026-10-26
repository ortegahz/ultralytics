#!/usr/bin/env python3
"""11_make_calib_variants.py -- build the calibration-set variants for the
"why is the int8 output range 4x too narrow" experiment.

THE EXPERIMENT
--------------
Measured facts:
  * the trial-0080 int8 model has out[0] (heatmap) zp=-128 scale=0.00094072, i.e.
    it can only represent [0, 0.2399]; the decision threshold is 0.3, so every
    peak is clipped below the threshold => 0 detections on every frame
    (board probe: th=0.3 -> 0/500 frames, th=0.1 -> 389/500, score_max=0.2399).
  * the same 250 calibration frames run through the FLOAT ONNX give heatmap
    peaks of 0.85..0.97 (measured by 10_probe_calib_peak.py), which implies an
    output scale of ~0.0075, not 0.00094.
  * the trial-0474 int8 model that actually shipped had out[0] scale=0.00365272
    and was calibrated with 200 JPEG feature images listed in
    /media/manu/1TB-Volume/rknn_work/hm/calib_train_features.txt (Anti-UAV
    train features, round-robin over 199 sequences).

So the toolkit's calibration pass did NOT see what ONNX Runtime sees. Three
candidate causes, and this script prepares the inputs that separate them:

  A. the toolkit cannot decode our 24-bit BMP calibration files (=> the
     calibration ran on nothing).  Test: the SAME frames re-encoded as PNG.
  B. the calibration CONTENT is wrong (=> the frames we picked are too weak /
     from the wrong dataset).  Test: the frozen trial-0474 list.
  C. the toolkit version changed the behaviour (2.3.0 -> 2.3.2).  Test: build
     the same set with 2.3.0 if that environment exists (checked separately).

Outputs (all under --calib-root, default /mnt/data/siping/ptq_work/calib2):
    calib2/set_A.txt          the 250 frames re-encoded as PNG (paths only)
    calib2/png/<name>.png     the PNG files themselves
    calib2/t0474.txt          the frozen trial-0474 list, rewritten to the
                              server's path prefix
    calib2/t0474_manifest.json  where it came from, for auditability

Usage (server):
    python manu/pipeline/npu/ptq/11_make_calib_variants.py --set A
    python manu/pipeline/npu/ptq/11_make_calib_variants.py --t0474
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CALIB = Path("/mnt/data/siping/ptq_work/calib")
DEFAULT_OUT = Path("/mnt/data/siping/ptq_work/calib2")
# the frozen trial-0474 calibration list lives on the workstation
T0474_SRC = Path("/media/manu/1TB-Volume/rknn_work/hm/calib_train_features.txt")
# the list was written against the workstation's NFS mount; on the server the
# same tree is at /mnt/data/siping
T0474_PREFIX_FROM = "/home/manu/mnt/data/siping/"
T0474_PREFIX_TO = "/mnt/data/siping/"


def make_png(set_key: str, calib_root: Path, out_root: Path) -> dict:
    lst = calib_root / f"set_{set_key}.txt"
    if not lst.is_file():
        raise SystemExit(f"[FATAL] {lst} missing")
    paths = [l.strip() for l in lst.read_text().splitlines() if l.strip()]
    png_dir = out_root / "png" / f"set_{set_key}"
    png_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    out_lines = []
    missing = 0
    for p in paths:
        src = Path(p)
        if not src.is_file():
            missing += 1
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            missing += 1
            continue
        dst = png_dir / (src.stem + ".png")
        cv2.imwrite(str(dst), img)
        out_lines.append(str(dst))
    out_list = out_root / f"set_{set_key}.txt"
    out_list.write_text("\n".join(out_lines) + "\n")
    print(f"[png] set_{set_key}: {len(out_lines)} written, {missing} missing -> {out_list}")
    return {"set": set_key, "n": len(out_lines), "missing": missing,
            "list": str(out_list), "png_dir": str(png_dir)}


def make_t0474(out_root: Path, src: Path = T0474_SRC) -> dict:
    if not src.is_file():
        raise SystemExit(f"[FATAL] {src} missing (workstation-only asset; copy it to the "
                         f"server and pass --t0474-src)")
    raw = [l.strip() for l in src.read_text().splitlines() if l.strip()]
    lines, missing = [], 0
    for p in raw:
        q = p.replace(T0474_PREFIX_FROM, T0474_PREFIX_TO)
        if not Path(q).is_file():
            missing += 1
            continue
        lines.append(q)
    out_list = out_root / "t0474.txt"
    out_list.write_text("\n".join(lines) + "\n")
    meta = {"src": str(src), "src_prefix": T0474_PREFIX_FROM,
            "dst_prefix": T0474_PREFIX_TO, "n_src": len(raw), "n_written": len(lines),
            "missing": missing, "list": str(out_list),
            "note": "frozen trial-0474 calibration list: 200 Anti-UAV train feature JPGs, "
                    "round-robin over 199 sequences"}
    (out_root / "t0474_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[t0474] {len(lines)}/{len(raw)} written ({missing} missing) -> {out_list}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=None, help="set key to re-encode as PNG (A/B/C/D)")
    ap.add_argument("--t0474", action="store_true", help="also write the frozen trial-0474 list")
    ap.add_argument("--t0474-src", default=str(T0474_SRC))
    ap.add_argument("--calib-root", default=str(DEFAULT_CALIB))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    info = {"out_root": str(out_root)}
    if args.set:
        info["png"] = make_png(args.set, Path(args.calib_root), out_root)
    if args.t0474:
        info["t0474"] = make_t0474(out_root, Path(args.t0474_src))
    (out_root / "variants_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"[out] {out_root/'variants_manifest.json'}")


if __name__ == "__main__":
    main()
