#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""37_mosaic_pc_control.py -- the PC-side control arm of the board mosaic experiment.

WHY THIS EXISTS
---------------
The board arm runs ONE forward on a 640x640 canvas that packs three scales
(native 640x512 + thumb160 + thumb80).  The PC reference arm runs THREE separate
forwards, one per scale, on three separate letterboxed 640x640 canvases.  Those
are not the same computation, so a single F1 delta cannot be attributed:

    delta_f1(board_mosaic, pc_per_scale)
        = f16 / board-quantisation loss
        + mosaic packing loss (canvas geometry + one-forward coupling)

This script produces the missing middle arm: the SAME mosaic canvas, the SAME
single forward, the SAME back-projection and the SAME connected-component
detector -- but on the PT float32 checkpoint on the GPU.  Comparing it against
the board arm isolates the f16 loss, and comparing it against the PC per-scale
arm isolates the packing loss.

It deliberately reads the SAME BMP files the board reads (``ptq_common.BMP_ROOT``),
so the pixels are byte-identical on both sides.

OUTPUT
------
* ``<out>/<seq>_pc_mosaic.tsv`` -- the board's TSV format, so the same scorer
  consumes both arms without any special-casing.
* ``<out>/<seq>_pc_mosaic.pkl`` -- an ``ms_cache``-shaped pickle.
* ``<out>/<seq>_pc_mosaic.json`` -- the metrics, produced by the SAME
  ``manu/evaluation/system_sota_longquan.py --no-pipeline`` path the PC
  reference numbers come from.

The mosaic layout here is byte-for-byte the layout the board program uses:

    canvas 640x640, background 114 in ALL THREE channels
      native    : source 640x512 at (0,   0)   640x512  (1:1)
      thumb160  : source 640x512 at (0, 512)   160x128  (1/4 down)
      thumb80   : source 640x512 at (160,512)   80x64   (1/8 down)

and the back-projection is "crop the scale's sub-rectangle out of the 320x320
output grid, then resize it to 640x512" -- the same order the board C++ uses.

Usage
-----
    python 37_mosaic_pc_control.py --seq VIDEO00007_19700101_001324 --limit 200
    python 37_mosaic_pc_control.py --seq VIDEO00007_19700101_001324   # full 11,810
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# .../manu/pipeline/npu/ptq/37_mosaic_pc_control.py  ->  parents[4] is the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.multiscale_heatmap_cache import detect_components  # noqa: E402
from manu.data.preannotate_hm_bbox import load_hm_model, resolve_weights  # noqa: E402
from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

# ---------------------------------------------------------------------------
# Mosaic layout -- MUST match manu/pipeline/npu/rknn_mosaic_heatmap.cpp
# ---------------------------------------------------------------------------
CANVAS = 640
NATIVE_W, NATIVE_H = 640, 512
TILES = (
    # (tag, canvas_x, canvas_y, width, height, scale_divisor)
    (0, 0, 0, 640, 512, 1),
    (1, 0, 512, 160, 128, 4),
    (2, 160, 512, 80, 64, 8),
)
PAD_VALUE = 114
REGION_THRESHOLD = 0.06
MIN_AREA = 4
MAX_DETECTIONS = 100
HM_W, HM_H = 320, 320  # the 640x640 canvas at stride 2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default=C.TRACK3_BLIND[0])
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out-root", default=str(C.WORK_ROOT / "mosaic_pc"))
    ap.add_argument("--bmp-root", default=str(C.BMP_ROOT))
    ap.add_argument("--weights", default=str(C.WEIGHTS))
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--params-json", default="", help="gate params json (th_base etc.)")
    ap.add_argument("--labels-root", default=str(C.GT_ROOT))
    ap.add_argument("--thumb-interp", choices=("area", "linear"), default="area",
                    help="thumbnail downscale: area = PC reference (cv2.INTER_AREA); "
                         "linear = the board's RGA (always bilinear)")
    ap.add_argument("--score", action="store_true", help="run the PC scorer on the produced pkl")
    ap.add_argument("--cache-root", default=str(C.SERVER_ROOT / "runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"))
    ap.add_argument("--params-source", default=str(C.SERVER_ROOT / "runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/track3_blind.json"))
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Canvas assembly
# ---------------------------------------------------------------------------
def model_order_from_bmp(path: Path) -> np.ndarray:
    """BMP -> HxWx3 uint8 in the MODEL channel order.

    The BMP stores BGR (``cv2.imwrite(cv2.imread(jpg))``).  ``[..., ::-1]`` turns
    BGR into the RGB byte order that the board's ``stbi_load`` hands over -- and
    that RGB byte order IS the model channel order [residual, diff, I_t].
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read bmp: {path}")
    if img.shape[:2] != (NATIVE_H, NATIVE_W):
        raise ValueError(f"{path}: expected {NATIVE_W}x{NATIVE_H}, got {img.shape[1]}x{img.shape[0]}")
    return np.ascontiguousarray(img[..., ::-1])


def build_canvas(rgb: np.ndarray, thumb_interp: str = "area") -> np.ndarray:
    """Assemble the 640x640 mosaic canvas from one 640x512 model-order frame.

    ``thumb_interp`` selects the downscale used for the two thumbnails:
      "area"   -> cv2.INTER_AREA (block mean).  This is what the PC reference arm
                  (gate cache) used, via downsample_pad().
      "linear" -> cv2.INTER_LINEAR.  The board's RGA is always bilinear (the
                  INTER_NEAREST / INTER_LINEAR pair produced IDENTICAL output in
                  the RGA probe), so "linear" reproduces the BOARD's thumbnail
                  pixels.  The two differ by up to 131/255 on the target pixels
                  -- which is exactly the evidence for the thumbnail-resampling
                  breakdown between the board arm and the PC reference arm.
    """
    canvas = np.full((CANVAS, CANVAS, 3), PAD_VALUE, dtype=np.uint8)
    interp = cv2.INTER_AREA if thumb_interp == "area" else cv2.INTER_LINEAR
    for _tag, x, y, w, h, div in TILES:
        if div == 1:
            patch = rgb
        else:
            patch = cv2.resize(rgb, (w, h), interpolation=interp)
        canvas[y : y + h, x : x + w] = patch
    return canvas


def backproject(heatmap: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop one tile out of the 320x320 output grid and resample it to 640x512.

    ``heatmap_x = canvas_x * HM_W / CANVAS`` -- for a 320-wide grid on a 640
    canvas that is simply a factor of two.
    """
    x0 = int(round(x * HM_W / CANVAS))
    y0 = int(round(y * HM_H / CANVAS))
    x1 = int(round((x + w) * HM_W / CANVAS))
    y1 = int(round((y + h) * HM_H / CANVAS))
    sub = heatmap[y0:y1, x0:x1]
    return cv2.resize(sub, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    bmp_dir = Path(args.bmp_root) / args.seq
    if not bmp_dir.is_dir():
        raise SystemExit(f"[FATAL] no bmp dir: {bmp_dir}")
    frames = sorted(bmp_dir.glob("*.bmp"))
    if args.start:
        frames = frames[args.start :]
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit("[FATAL] no frames selected")
    print(f"[INFO] seq={args.seq} frames={len(frames)} device={device}", flush=True)

    # --ablate-p0 mirrors the board model (p0_off_A_none.rknn has no P0 stream).
    model, stride = load_hm_model(resolve_weights(args.weights), device, use_p0_highway=False)
    print(f"[INFO] weights={args.weights} stride={stride} p0=dropped", flush=True)

    tsv_path = out_dir / f"{args.seq}_pc_mosaic.tsv"
    frame_index, points, sizes, scores, sums, areas, tags = [], [], [], [], [], [], []
    per_frame_counts = []
    per_tag = {t: 0 for t, *_ in TILES}
    started = time.perf_counter()

    with tsv_path.open("w", encoding="utf-8") as tsv:
        for start in range(0, len(frames), args.batch_size):
            chunk = frames[start : start + args.batch_size]
            canvases = np.stack([build_canvas(model_order_from_bmp(p), args.thumb_interp) for p in chunk])
            tensor = torch.from_numpy(canvases.transpose(0, 3, 1, 2)).to(device).float() / 255.0
            with torch.no_grad():
                prediction = model(tensor)
            heatmaps = prediction["heatmap"][:, 0].cpu().numpy()
            for index, path in enumerate(chunk):
                heatmap = heatmaps[index]
                row = []
                for tag, x, y, w, h, _div in TILES:
                    native = backproject(heatmap, x, y, w, h)
                    entries = detect_components(native, REGION_THRESHOLD, 1, MIN_AREA, MAX_DETECTIONS)
                    per_tag[tag] += len(entries)
                    for cx, cy, bw, bh, area, peak, total in entries:
                        points.append((cx, cy))
                        sizes.append((bw, bh))
                        scores.append(peak)
                        sums.append(total)
                        areas.append(area)
                        tags.append(tag)
                        row.append((tag, cx, cy, bw, bh, area, peak, total))
                frame_index.append(int(path.stem.rsplit("__", 1)[1]))
                per_frame_counts.append(len(row))
                stem = path.stem
                tsv.write(f"{stem}\t{len(row)}")
                for tag, cx, cy, bw, bh, area, peak, total in row:
                    tsv.write(
                        f"\t{tag} {cx:.4f} {cy:.4f} {bw:d} {bh:d} {area:d} {peak:.6f} {total:.4f}"
                    )
                tsv.write("\n")
            if (start // args.batch_size) % 20 == 0:
                done = start + len(chunk)
                rate = done / max(1e-9, time.perf_counter() - started)
                print(
                    f"[{done}/{len(frames)}] {rate:.1f} fps | det/frame="
                    f"{(len(points) / max(1, len(frame_index))):.2f}",
                    flush=True,
                )
    meta = {
        "sequence": args.seq,
        "frames": len(frame_index),
        "native_size": [NATIVE_W, NATIVE_H],
        "imgsz": CANVAS,
        "stride": stride,
        "scales": [0, 160, 80],
        "fused_tag": 255,
        "coordinate_space": "native_pixel",
        "source": f"pc_mosaic_f32_{args.thumb_interp}",
        "region_threshold": REGION_THRESHOLD,
        "main_threshold": 0.22,
        "fusion_dilate": 32,
        "min_area": MIN_AREA,
        "features_root": str(bmp_dir),
        "hm_weights": str(args.weights),
        "use_p0_highway": False,
        "adapter_weights": "",
        "adapter_mid_channels": 0,
        "adapter_guide_channel": -1,
        "adapter_single": False,
        "canvas_layout": "native(0,0,640,512) thumb160(0,512,160,128) thumb80(160,512,80,64)",
        "thumb_interp": args.thumb_interp,
        "backproject": "crop-then-resize",
    }
    cache = {
        "meta": meta,
        "frame_index": np.asarray(frame_index, dtype=np.int32),
        "offset": np.concatenate([[0], np.cumsum(np.asarray(per_frame_counts, dtype=np.int64))]).astype(np.int64),
        "points": np.asarray(points, dtype=np.float32).reshape(-1, 2),
        "sizes": np.asarray(sizes, dtype=np.float32).reshape(-1, 2),
        "scores": np.asarray(scores, dtype=np.float16),
        "sums": np.asarray(sums, dtype=np.float32),
        "area": np.asarray(areas, dtype=np.int32),
        "tag": np.asarray(tags, dtype=np.uint8),
    }
    pkl_path = out_dir / f"{args.seq}.pkl"
    pkl_path.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))

    summary = {
        "sequence": args.seq,
        "frames": len(frame_index),
        "detections": len(tags),
        "per_tag_counts": per_tag,
        "tsv": str(tsv_path),
        "pkl": str(pkl_path),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    print(f"[DONE] {json.dumps(summary, ensure_ascii=False)}", flush=True)

    if args.score:
        score_one(args, pkl_path, summary)


def score_one(args: argparse.Namespace, pkl_path: Path, summary: dict) -> None:
    """Score the produced cache with the very same path the PC reference uses."""
    cache_root = pkl_path.parent
    params_json = args.params_json
    if not params_json:
        params_json = str(cache_root / "_gate_params.json")
        source = Path(args.params_source)
        if not source.is_file():
            raise SystemExit(f"[FATAL] gate params source missing: {source}")
        params = json.loads(source.read_text(encoding="utf-8"))["config"]["params"]
        Path(params_json).write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    out_json = cache_root / f"{args.seq}_pc_mosaic_scores.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "manu/evaluation/system_sota_longquan.py"),
        "--cache-root", str(cache_root),
        "--labels-root", str(args.labels_root),
        "--seq", args.seq,
        "--no-pipeline",
        "--params-json", params_json,
        "--out-json", str(out_json),
    ]
    print(f"[score] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(C.SERVER_ROOT), capture_output=True, text=True)
    print(result.stdout[-4000:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr, flush=True)
        raise SystemExit(f"[FATAL] scorer failed rc={result.returncode}")
    report = json.loads(out_json.read_text(encoding="utf-8"))
    summary["scores"] = report.get("overall")
    summary["scores_json"] = str(out_json)
    summary["params_json"] = params_json
    print(f"[SCORE] {json.dumps(report.get('overall'), ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
