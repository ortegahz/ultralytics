#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Attribute an adapter's behaviour to its two branches on real frames.

The dual-branch adapter emits ``delta_h = +positive(head_recall(feat)) - positive(head_suppress(feat))``,
so a regression can only come from one of three places: the recall branch failing to raise weak GT
peaks, the suppress branch erasing them, or the ``clamp(h_base + delta_h, 0, 1)`` saturating so hard
that gradients die and peaks are destroyed. This probe measures all three on the actual validation
sequences, at the coordinates that matter (GT centres vs background), instead of on global averages
that can hide a branch that only misbehaves in a few pixels.

    python manu/diagnostics/adapter_branch_probe.py \
        --adapter-weights runs/adapter_train_v5/adapter_epoch1.pt --mid-channels 16 \
        --features-root /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --sequence VIDEO00009_19700101_001531 --frames 191 --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.mine_adapter_training_samples import build_native_canvas  # noqa: E402
from manu.data.preannotate_hm_bbox import load_hm_model, resolve_weights  # noqa: E402
from manu.models.adapter_model import HeatmapAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-branch attribution of a HeatmapAdapter")
    parser.add_argument("--adapter-weights", required=True)
    parser.add_argument("--mid-channels", type=int, default=16)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--ablate-p0", action="store_true",
                        help="Match the no-P0 cache arm the Longquan baselines were built with")
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--guide-channel", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def load_gt_boxes(labels_root: Path, sequence: str, frame_index: int, width: int, height: int) -> np.ndarray:
    """Return GT centres in native pixels as (N, 2); empty when the frame is unlabelled.

    The Final_Labels rows are ``class cx cy w h`` with cx/cy/w/h NORMALISED by the native width
    and height -- exactly how eval_multiscale_cache_th.load_ground_truth reads them. Column 0 is
    the class, so slicing from index 0 silently shifts every field by one.
    """
    path = labels_root / f"{sequence}__{frame_index:06d}.txt"
    if not path.exists():
        return np.zeros((0, 2), dtype=np.float32)
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if len(row) >= 5]
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    values = np.asarray([[float(item) for item in row[1:5]] for row in rows], dtype=np.float32)
    values *= np.float32([width, height, width, height])
    return np.stack([values[:, 0], values[:, 1]], axis=1)


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, stride = load_hm_model(resolve_weights(args.hm_weights), device, use_p0_highway=not args.ablate_p0)
    adapter = HeatmapAdapter(in_channels=2, mid_channels=args.mid_channels, dual=not args.single).to(device)
    adapter.load_state_dict(torch.load(args.adapter_weights, map_location=device))
    adapter.eval()
    print(f"[ADAPTER] {adapter.describe()} | guide=ch{args.guide_channel} (model order)")

    features_root = Path(args.features_root)
    labels_root = Path(args.labels_root)
    frame_paths = sorted(features_root.glob(f"{args.sequence}__frame_*.jpg"))[: args.frames]
    if not frame_paths:
        raise FileNotFoundError(f"no frames for {args.sequence} under {features_root}")

    # ---- accumulators, split by GT / background so a branch cannot hide behind global averages ----
    gt_recall, gt_suppress, gt_delta, gt_base = [], [], [], []
    bg_recall, bg_suppress, bg_delta = [], [], []
    killed_gt, total_gt, raised_bg, total_bg = 0, 0, 0, 0

    with torch.no_grad():
        for frame_path in frame_paths:
            frame_index = int(frame_path.stem.rsplit("_", 1)[1])
            feature = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if feature is None:
                continue
            height, width = feature.shape[:2]
            canvas, _ = build_native_canvas(feature, args.imgsz, width, height)

            heat = model(torch.from_numpy(canvas[None]).to(device).float() / 255.0)["heatmap"][:, 0]
            grid = heat.shape[-1]
            guide = torch.nn.functional.interpolate(
                torch.from_numpy(canvas[None, args.guide_channel : args.guide_channel + 1]).to(device).float(),
                size=(grid, grid), mode="bilinear", align_corners=False,
            )
            if not adapter.dual:
                print("[FATAL] this probe needs a dual-branch adapter", file=sys.stderr)
                raise SystemExit(2)
            final, delta_recall, delta_suppress = adapter.forward_split(heat[:, None], guide)

            # native -> grid: the canvas is letterboxed to imgsz, so undo that before indexing
            scale = min(args.imgsz / height, args.imgsz / width)
            new_w, new_h = round(width * scale), round(height * scale)
            left, top = (args.imgsz - new_w) // 2, (args.imgsz - new_h) // 2

            def to_grid(points: np.ndarray) -> np.ndarray:
                if len(points) == 0:
                    return points
                return np.stack([(points[:, 0] * scale + left) / args.imgsz * grid,
                                 (points[:, 1] * scale + top) / args.imgsz * grid], axis=1)

            r_np = delta_recall[0, 0].cpu().numpy()
            s_np = delta_suppress[0, 0].cpu().numpy()
            d_np = (delta_recall + delta_suppress)[0, 0].cpu().numpy()
            b_np = heat[0].cpu().numpy()
            f_np = final[0, 0].cpu().numpy()

            centres = to_grid(load_gt_boxes(labels_root, args.sequence, frame_index, width, height))
            mask = np.zeros((grid, grid), dtype=bool)
            for cx, cy in centres:
                x, y = int(round(cx)), int(round(cy))
                if 0 <= x < grid and 0 <= y < grid:
                    mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = True

            if mask.any():
                gt_recall.append(r_np[mask]); gt_suppress.append(s_np[mask])
                gt_delta.append(d_np[mask]); gt_base.append(b_np[mask])
                total_gt += int(mask.sum())
                killed_gt += int(((b_np[mask] > 0.22) & (f_np[mask] < 0.22)).sum())
            bg = ~mask
            if bg.any():
                bg_recall.append(r_np[bg]); bg_suppress.append(s_np[bg]); bg_delta.append(d_np[bg])
                total_bg += int(bg.sum())
                raised_bg += int(((b_np[bg] < 0.10) & (f_np[bg] >= 0.10)).sum())

    def flat(items):
        return np.concatenate(items) if items else np.zeros(0, dtype=np.float32)

    gr, gs, gd, gb = flat(gt_recall), flat(gt_suppress), flat(gt_delta), flat(gt_base)
    br, bs, bd = flat(bg_recall), flat(bg_suppress), flat(bg_delta)

    report = {
        "adapter_weights": args.adapter_weights,
        "sequence": args.sequence,
        "frames": len(frame_paths),
        "stride": stride,
        "gt_pixels": total_gt,
        "bg_pixels": total_bg,
        "gt": {
            "delta_recall_mean": float(gr.mean()) if len(gr) else 0.0,
            "delta_suppress_mean": float(gs.mean()) if len(gs) else 0.0,
            "delta_net_mean": float(gd.mean()) if len(gd) else 0.0,
            "delta_recall_max": float(gr.max()) if len(gr) else 0.0,
            "delta_suppress_min": float(gs.min()) if len(gs) else 0.0,
            "delta_net_min": float(gd.min()) if len(gd) else 0.0,
            "h_base_mean": float(gb.mean()) if len(gb) else 0.0,
            "killed_above_th": killed_gt,
            "kill_rate_above_th": killed_gt / total_gt if total_bg else 0.0,
        },
        "background": {
            "delta_recall_mean": float(br.mean()) if len(br) else 0.0,
            "delta_suppress_mean": float(bs.mean()) if len(bs) else 0.0,
            "delta_net_mean": float(bd.mean()) if len(bd) else 0.0,
            "delta_suppress_min": float(bs.min()) if len(bs) else 0.0,
            "raised_above_th": raised_bg,
            "raise_rate": raised_bg / total_bg if total_bg else 0.0,
        },
        "global": {
            "delta_recall_abs_max": float(np.abs(gr).max() if len(gr) else 0.0, ),
            "delta_suppress_abs_max": float(np.abs(bs).max() if len(bs) else 0.0),
            "clamp_saturation_share": float((np.abs(d_np) > 1.0).mean()) if len(d_np) else 0.0,
        },
    }

    print("\n" + "=" * 78)
    print(f"BRANCH ATTRIBUTION  {args.sequence}  frames={len(frame_paths)}  grid={grid}")
    print("=" * 78)
    print(f"{'region':<12}{'d_recall(mean)':>16}{'d_suppress(mean)':>18}{'d_net(mean)':>14}{'d_net(min)':>13}")
    print(f"{'GT':<12}{report['gt']['delta_recall_mean']:>16.5f}"
          f"{report['gt']['delta_suppress_mean']:>18.5f}"
          f"{report['gt']['delta_net_mean']:>14.5f}{report['gt']['delta_net_min']:>13.5f}")
    print(f"{'background':<12}{report['background']['delta_recall_mean']:>16.5f}"
          f"{report['background']['delta_suppress_mean']:>18.5f}"
          f"{report['background']['delta_net_mean']:>14.5f}"
          f"{report['background']['delta_suppress_min']:>13.5f}")
    print("-" * 78)
    print(f"GT h_base mean            {report['gt']['h_base_mean']:.5f}")
    print(f"GT peaks dropped below th  {killed_gt} / {total_gt} "
          f"({report['gt']['kill_rate_above_th']*100:.3f}% of GT pixels)")
    print(f"background raised above th {raised_bg} / {total_bg} "
          f"({report['background']['raise_rate']*100:.4f}%)")
    print(f"|delta_recall| max         {report['global']['delta_recall_abs_max']:.5f}")
    print(f"|delta_suppress| max       {report['global']['delta_suppress_abs_max']:.5f}")
    print("=" * 78)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[WROTE] {args.out_json}")


if __name__ == "__main__":
    main()
