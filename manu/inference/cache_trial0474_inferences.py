#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate ultra-fast lightweight inference cache (.pkl) directly from the official GMC Median dataset
using the new Single-Frame Absolute Champion model: Trial 0474 (runs/optuna_p0_nas/trial_0474/weights/best.pt).

Key architecture & design:
1. Micro-Architecture: standard_dw + pixel_unshuffle + diff_only (depth=2, mid_channels=16, scalar_gate).
2. Deep extraction down to conf=0.02 (top_k=100) to capture all sub-threshold candidate pulses.
3. Strict YOLO DataLoader alignment with official validation set (/mnt/data/siping/datasets/manu/uav_gmc_median).
4. Save compact float16 detections (< 20MB) enabling instant multi-strategy track fusion sweeps (<2s per run).
5. Strictly adheres to Rules: file size <= 50MB, no dense tensors stored.

Usage on Server:
    python manu/inference/cache_trial0474_inferences.py \
        --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
        --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
        --output runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --device 0 \
        --batch 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.evaluation.heatmap_evaluate import extract_peaks
from manu.models.heatmap_model import YOLO26HeatmapDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Inference Cache for Trial 0474 Model")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_p0_nas/trial_0474/weights/best.pt",
        help="Path to Trial 0474 checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to uav_gmc_median data.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Destination pickle path",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf-thresh", type=float, default=0.02, help="Deep harvest threshold (default: 0.02)")
    parser.add_argument("--top-k", type=int, default=100, help="Max candidate peaks to store per frame (default: 100)")
    parser.add_argument(
        "--ablate-p0",
        action="store_true",
        help="Ablation: build the model WITHOUT the P0 residual highway (self.p0_highway=None). "
             "Every p0_highway.* tensor in the checkpoint is then skipped and the detector is exactly "
             "the frozen base path. Asserts that no p0 tensor was loaded -- see the GATE below.",
    )
    parser.add_argument(
        "--adapter-weights",
        type=str,
        default="",
        help="Optional HeatmapAdapter .pt. When set, the frozen base heatmap is corrected by "
             "H_final = clamp(H_base + delta, 0, 1) BEFORE extract_peaks, so the cache format, the "
             "downstream A/B evaluator and the frozen headline all stay byte-comparable.",
    )
    parser.add_argument("--adapter-mid-channels", type=int, default=16,
                        help="16 keeps the dual-branch adapter at ~2.3%% of the base model's MACs")
    parser.add_argument("--adapter-single", dest="adapter_dual", action="store_false", default=True,
                        help="Use the single-branch reference layout instead of the dual-branch one")
    parser.add_argument(
        "--adapter-guide-channel",
        type=int,
        default=2,
        help="Which channel of the GMC+median input guides the adapter. Measured on-disk order of "
             "uav_gmc_median is [I_t, motion, (I_t-B_t)^+], so 2 = median residual (default, sparsest).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("   UAV Tiny Object Detection: Offline Inference Cacher for Trial 0474 (New SOTA)")
    print(f"   Checkpoint  : {args.weights}")
    print(f"   Dataset     : {args.data}")
    print(f"   Output PKL  : {args.output}")
    print(f"   Conf Gating : {args.conf_thresh:.2f} (top {args.top_k} peaks)")
    print("=" * 80)

    # 1. Resolve paths
    weights_path = Path(args.weights)
    if not weights_path.exists():
        alt = REPO_ROOT / args.weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Load Model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[INFO] Using Device: {device}")

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    stride = ckpt.get("stride", args.stride)

    # Retrieve p0_kwargs from checkpoint or fallback to Trial 0474 golden architecture
    p0_kwargs = ckpt.get("p0_kwargs", {
        "use_spatial_gate": True,
        "stem_type": "standard_dw",
        "downsample_mode": "pixel_unshuffle",
        "gate_input_mode": "diff_only",
        "gate_mid_channels": 16,
        "gate_depth": 2,
        "fusion_mode": "scalar_gate",
    })

    print(f"[INFO] P0 Micro-Architecture: {p0_kwargs}")

    # Instantiate model with use_p0_highway=True and specific p0_kwargs
    model = YOLO26HeatmapDetector(
        stride=stride,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=not args.ablate_p0,
        p0_highway_kwargs=p0_kwargs,
    )

    # Load trained state_dict directly
    matched, skipped, n_p0_skipped = 0, 0, 0
    own_state = model.state_dict()
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_k in own_state and own_state[clean_k].shape == v.shape:
            own_state[clean_k].copy_(v)
            matched += 1
        else:
            skipped += 1
            if clean_k.startswith("p0_highway"):
                n_p0_skipped += 1

    # ABLATION GATE: --ablate-p0 is only meaningful if the model really has no P0 stream AND
    # every p0_highway tensor was actually dropped. Silently "succeeding" with a P0 model would
    # make the whole ablation a no-op that still prints a lower F1, so fail loudly instead.
    if args.ablate_p0:
        if model.p0_highway is not None:
            raise RuntimeError("[ABLATE-P0 GATE] model.p0_highway is not None -- ablation did not take effect")
        if n_p0_skipped == 0:
            raise RuntimeError("[ABLATE-P0 GATE] 0 p0_highway.* tensors were dropped -- nothing was ablated")
        print(colorstr("yellow", f"[ABLATE-P0 GATE] PASS: p0_highway=None, {n_p0_skipped} p0 tensors dropped"))

    # hasattr() is True even when p0_highway is None (ablation arm), so test the value, not the
    # attribute -- reading .gate off a None highway raises AttributeError.
    actual_gate = (
        model.p0_highway.gate.item()
        if getattr(model, "p0_highway", None) is not None and hasattr(model.p0_highway.gate, "item")
        else 0.0
    )
    print(f"[INFO] Loaded State Dict: {matched} layers matched, {skipped} skipped.")
    print(colorstr("cyan", f"[INFO] Active Gate α Value: {actual_gate:.6f}"))

    model.to(device)
    model.eval()

    # 2b. Optional lightweight adapter. Loaded AFTER the base model so the frozen detector is
    #     never touched; the adapter only rescales the heatmap that feeds extract_peaks.
    adapter = None
    if args.adapter_weights:
        from manu.models.adapter_model import HeatmapAdapter

        adapter = HeatmapAdapter(in_channels=2, mid_channels=args.adapter_mid_channels,
                                 dual=args.adapter_dual)
        payload = torch.load(args.adapter_weights, map_location="cpu")
        adapter.load_state_dict(payload.get("state_dict", payload) if isinstance(payload, dict) else payload)
        adapter.to(device)
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        print(colorstr("cyan", f"[ADAPTER] {adapter.describe()} | guide=ch{args.adapter_guide_channel} "
                               f"| weights={args.adapter_weights}"))

    # 3. Setup DataLoader
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)
    print(colorstr("bold", f"[INFO] Processing {len(val_dataset)} images in official val split..."))

    records = []
    t0 = time.time()
    identity_checked = adapter is None

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Trial 0474 Inference & Caching"):
            # Keep the raw 0..255 copy: the adapter was TRAINED on guidance channels that come
            # straight off disk as uint8 (AdapterFrameDataset does payload["guide"].astype(float32)
            # with no /255), so feeding it imgs_tensor/255 would shrink the guidance by 255x and
            # silently flatten the correction. The detector itself still wants 0..1.
            imgs_u8 = batch["img"].to(device, non_blocking=True).float()
            imgs_tensor = imgs_u8 / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            im_files = batch.get("im_file", [""] * imgs_tensor.shape[0])
            bs = imgs_tensor.shape[0]

            preds = model(imgs_tensor)

            if adapter is not None:
                heatmap = preds["heatmap"]
                grid_h, grid_w = heatmap.shape[-2:]
                guide = torch.nn.functional.interpolate(
                    imgs_u8[:, args.adapter_guide_channel : args.adapter_guide_channel + 1],
                    size=(grid_h, grid_w),
                    mode="bilinear",
                    align_corners=False,
                )
                corrected, delta_h = adapter(heatmap, guide)

                if not identity_checked:
                    # Epoch-0 gate, expressed in the official pipeline: a freshly zero-initialised
                    # delta head must leave the heatmap bit-identical, so the cache produced with an
                    # untrained adapter has to reproduce the frozen headline exactly.
                    base_delta = float(delta_h.abs().max())
                    base_drift = float((corrected - heatmap).abs().max())
                    identity_checked = True
                    if base_delta == 0.0 and base_drift != 0.0:
                        raise RuntimeError(
                            f"[ADAPTER IDENTITY GATE] FAILED: delta is zero yet the heatmap moved by "
                            f"{base_drift:.3e}; the clamp path is not a pure identity"
                        )
                    print(colorstr(
                        "yellow",
                        f"[ADAPTER IDENTITY GATE] first batch: max|delta_h|={base_delta:.3e} "
                        f"max|H_final-H_base|={base_drift:.3e} "
                        + ("(untrained adapter == frozen base)" if base_delta == 0.0 else "(adapter is trained)"),
                    ))

                preds = dict(preds)
                preds["heatmap"] = corrected

            peaks_list = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
                conf_thresh=args.conf_thresh,
                top_k=args.top_k,
            )

            for b in range(bs):
                mask_b = b_idx == b
                gt_norm = bboxes[mask_b].cpu().numpy()
                im_name = Path(im_files[b]).name if im_files[b] else f"img_{b}"

                gt_pts = []
                gt_bboxes = []
                for box in gt_norm:
                    gt_x = float(box[0] * args.imgsz)
                    gt_y = float(box[1] * args.imgsz)
                    gt_w = float(box[2] * args.imgsz)
                    gt_h = float(box[3] * args.imgsz)
                    gt_pts.append([gt_x, gt_y])
                    gt_bboxes.append([gt_x, gt_y, gt_w, gt_h])
                gt_pts = np.array(gt_pts, dtype=np.float32) if len(gt_pts) > 0 else np.zeros((0, 2), dtype=np.float32)
                gt_bboxes = np.array(gt_bboxes, dtype=np.float32) if len(gt_bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)

                # float32 is mandatory. Coordinates live in the 640x640 letterbox space, where
                # float16 has a 0.5 spacing across [512, 1024) and 0.25 across [256, 512), i.e. up
                # to 0.25-0.5 px of quantisation error per axis. At an 8 px match tolerance that is
                # enough to flip borderline frames: measured against the authoritative float32
                # evaluator it cost 3 TP and added 3 FP, which is exactly the -0.00016 F1 gap that
                # previously could not be attributed. Scores may stay float16 safely, but keeping
                # one dtype removes the temptation to reintroduce the bug. The cache only doubles
                # (5.1 -> ~10 MB), far inside the 100 MB budget.
                records.append({
                    "im_name": im_name,
                    "gt_pts": gt_pts.astype(np.float32),
                    "gt_bboxes": gt_bboxes.astype(np.float32),
                    "pred_points": peaks_list[b]["points"].astype(np.float32),
                    "pred_scores": peaks_list[b]["scores"].astype(np.float16),
                })

    elapsed = time.time() - t0
    fps = len(records) / max(0.1, elapsed)
    print(f"\n[INFO] Inference finished in {elapsed:.1f}s ({fps:.1f} imgs/s).")

    print(f"[INFO] Writing {len(records)} records to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[INFO] Record dtypes: pred_points/gt_pts/gt_bboxes = float32 "
          f"(mandatory for 8 px matching), pred_scores = float16")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(colorstr("bold", colorstr("green", f"[SUCCESS] Inference cache saved ({file_size_mb:.1f} MB) -> {output_path.resolve()}\n")))


if __name__ == "__main__":
    main()
