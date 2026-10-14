#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
CPU/PyTorch reference for the Trial-0474 heatmap detector.

Produces, for a list of pre-built feature JPEGs:
  * <stem>_hm.npy / <stem>_off.npy   raw float32 outputs (same layout the board dumps)
  * detections.json                  peaks decoded with hm_common.extract_peaks_np
  * a TP/FP/FN summary against the dataset labels at <= 8 px

This is the ground truth every NPU number is compared against, so it refuses to
run unless the model is in eval mode -- see the torch.onnx.export note in
hm_export_onnx.py: BatchNorm in train mode silently uses per-batch statistics,
which is a ~0.28 heatmap / ~1.7 offset error that looks exactly like a broken
export and is not.

Usage:
    python hm_reference.py --frames list.txt --out /path/to/out --th 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.pipeline.npu.hm_common import (  # noqa: E402
    canvas_to_src,
    extract_peaks_np,
    gt_from_label,
    letterbox_geometry,
    letterbox_to_input,
    load_feature_jpg,
    match_distance,
    split_seq_frame,
)

DEFAULT_LABEL_ROOT = Path("/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/labels/val")
DEFAULT_WEIGHTS = "/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_p0_nas/trial_0474/weights/best.pt"

# Overridden by --label-root. The x86 workstation and the training server mount the
# same dataset at different prefixes, so the default here only works on the former;
# a hardcoded root is what previously made this script unrunnable off-box.
LABEL_ROOT = DEFAULT_LABEL_ROOT


def label_for(image_path: Path) -> Path:
    return LABEL_ROOT / f"{image_path.stem}.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--label-root", default=str(DEFAULT_LABEL_ROOT),
                    help="directory of <stem>.txt labels; override when the dataset is "
                         "mounted at a different prefix (e.g. the training server)")
    ap.add_argument("--frames", required=True, help="text file, one feature JPEG per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--th", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--dist", type=float, default=8.0)
    ap.add_argument("--no-dump-tensors", action="store_true")
    ap.add_argument("--no-gt", action="store_true", help="skip label lookup (unlabelled frames)")
    ap.add_argument("--ablate-p0", action="store_true",
                    help="drop the P0 residual highway (same semantics as "
                         "cache_trial0474_inferences.py --ablate-p0)")
    args = ap.parse_args()

    global LABEL_ROOT
    LABEL_ROOT = Path(args.label_root)

    from manu.pipeline.npu.hm_export_onnx import build_model

    model, ckpt = build_model(args.weights, ablate_p0=args.ablate_p0)
    stride = ckpt.get("stride", 2)
    assert not model.training, "reference must run in eval mode"

    frames = [Path(l.strip()) for l in Path(args.frames).read_text().splitlines() if l.strip()]
    if not frames:
        sys.exit("[FATAL] empty frame list")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tot_tp = tot_fp = tot_fn = tot_gt = 0
    records = []
    with torch.no_grad():
        for i, img_path in enumerate(frames):
            rgb = load_feature_jpg(img_path)
            geom = letterbox_geometry(rgb.shape[:2])
            x = torch.from_numpy(letterbox_to_input(img_path))
            # model() returns a dict, so it must be indexed by key -- unpacking
            # it would silently yield the key strings instead of the tensors.
            out = model(x)
            hm_t, off_t = out["heatmap"], out["offset"]
            hm = hm_t[0].numpy()      # (1, H, W)
            off = off_t[0].numpy()     # (2, H, W)

            det = extract_peaks_np(hm, off, stride=stride, conf_thresh=args.th, top_k=args.topk)
            pts_canvas = det["points"]
            pts_src = canvas_to_src(pts_canvas, geom)

            rec = {
                "file": str(img_path),
                "name": img_path.name,
                "seq": split_seq_frame(img_path.name)[0],
                "frame_index": split_seq_frame(img_path.name)[1],
                "src_hw": list(rgb.shape[:2]),
                "th": args.th,
                "stride": stride,
                "n_det": int(len(pts_canvas)),
                "det_canvas": pts_canvas.astype(float).tolist(),
                "det_src": pts_src.astype(float).tolist(),
                "scores": det["scores"].astype(float).tolist(),
            }

            if not args.no_gt:
                lp = label_for(img_path)
                gt_canvas, gt_boxes = gt_from_label(lp, geom)
                rec["gt_canvas"] = gt_canvas.astype(float).tolist()
                rec["gt_src"] = canvas_to_src(gt_canvas, geom).astype(float).tolist()
                rec["gt_boxes_cxcywh_canvas"] = gt_boxes.astype(float).tolist()
                tp, fp, fn = match_distance(
                    np.concatenate([pts_canvas, det["scores"][:, None]], 1)
                    if len(pts_canvas) else np.zeros((0, 3), np.float32),
                    gt_canvas, args.dist)
                rec["tp"], rec["fp"], rec["fn"] = tp, fp, fn
                tot_tp += tp
                tot_fp += fp
                tot_fn += fn
                tot_gt += len(gt_canvas)

            if not args.no_dump_tensors:
                np.save(out_dir / f"{img_path.stem}_hm.npy", hm.astype(np.float32))
                np.save(out_dir / f"{img_path.stem}_off.npy", off.astype(np.float32))
            records.append(rec)
            if (i + 1) % 25 == 0:
                print(f"  ... {i + 1}/{len(frames)} frames")

    (out_dir / "detections.json").write_text(json.dumps(records, indent=1))

    n = len(records)
    prec = tot_tp / max(1, tot_tp + tot_fp)
    rec_ = tot_tp / max(1, tot_gt)
    f1 = 2 * prec * rec_ / max(1e-9, prec + rec_)
    print(f"\n[ref] {n} frames, th={args.th}, dist<={args.dist}px, stride={stride}")
    print(f"[ref] GT={tot_gt}  TP={tot_tp}  FP={tot_fp}  FN={tot_fn}")
    print(f"[ref] precision={prec:.4f}  recall={rec_:.4f}  F1={f1:.4f}")
    print(f"[ref] tensors+json -> {out_dir}")


if __name__ == "__main__":
    main()
