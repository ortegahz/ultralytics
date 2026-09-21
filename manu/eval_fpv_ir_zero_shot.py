#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Zero-Shot Inference, Cache Generation, and SOTA Evaluation for FPV Infrared Dataset.

Workflow:
1. Loads SOTA model checkpoint (default: Trial 0474 best.pt).
2. Runs DataLoader on the generated FPV IR 3-channel dataset (images/val).
3. Extracts candidate peaks (CenterNet heatmap maxpool提峰) and caches to a lightweight .pkl file.
4. Executes System SOTA Bidirectional Tracking & Rigid Static Pruning.
5. Prints comprehensive evaluation metrics (Recall, Precision, F1, TP, FP, FN, FAR)
   under both Point-in-BBox and Strict Distance criteria.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG, colorstr
from manu.heatmap_evaluate import extract_peaks
from manu.heatmap_model import YOLO26HeatmapDetector
from manu.eval_bidirectional_track_fusion import (
    evaluate_sequence_bidirectional,
    extract_seq_name,
    match_predictions_to_gt,
    natural_sort_key,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Shot SOTA Evaluation on FPV IR Dataset")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/optuna_p0_nas/trial_0474/weights/best.pt",
        help="Path to SOTA model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/fpv_ir_gmc_median/data.yaml",
        help="Path to converted fpv_ir_gmc_median data.yaml",
    )
    parser.add_argument(
        "--cache-output",
        type=str,
        default="runs/zero_shot_eval/fpv_ir_trial0474_cache.pkl",
        help="Path to save extracted inference cache .pkl",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    parser.add_argument("--conf-thresh", type=float, default=0.02, help="Peak harvesting floor")
    parser.add_argument("--top-k", type=int, default=100, help="Max peaks stored per frame")
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Evaluation tolerance (default: 8.0px)")
    parser.add_argument("--match-mode", type=str, default="bbox", choices=["bbox", "dist"], help="Matching mode")
    parser.add_argument("--skip-inference-if-cached", action="store_true", help="Skip model forward if cache pkl already exists")
    parser.add_argument("--search-threshold", action="store_true", help="Search the best single-frame confidence threshold")
    parser.add_argument("--th-min", type=float, default=0.05, help="Minimum threshold for search")
    parser.add_argument("--th-max", type=float, default=0.50, help="Maximum threshold for search")
    parser.add_argument("--th-step", type=float, default=0.01, help="Threshold search step")
    return parser.parse_args()


def load_sota_model(weights_path: Path, device: torch.device):
    print(colorstr("bold", f"[INFO] Loading SOTA Checkpoint: {weights_path}"))
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    stride = ckpt.get("stride", 2)

    p0_kwargs = ckpt.get("p0_kwargs", {
        "use_spatial_gate": True,
        "stem_type": "standard_dw",
        "downsample_mode": "pixel_unshuffle",
        "gate_input_mode": "diff_only",
        "gate_mid_channels": 16,
        "gate_depth": 2,
        "fusion_mode": "scalar_gate",
    })

    model = YOLO26HeatmapDetector(
        stride=stride,
        num_classes=1,
        temporal_mode="standard",
        use_p0_highway=True,
        p0_highway_kwargs=p0_kwargs,
    )

    matched, skipped = 0, 0
    own_state = model.state_dict()
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "").replace("model.model.", "").replace("model.", "")
        if clean_k in own_state and own_state[clean_k].shape == v.shape:
            own_state[clean_k].copy_(v)
            matched += 1
        else:
            skipped += 1

    print(f"[INFO] Weights loaded: {matched} layers matched, {skipped} skipped.")
    model.to(device)
    model.eval()
    return model, stride


def run_caching(model, stride, data_path: Path, cache_out: Path, args, device):
    data_dict = check_det_dataset(str(data_path))
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = str(data_path)

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=0, shuffle=False)
    print(colorstr("bold", f"[INFO] Processing {len(val_dataset)} images in FPV IR val split..."))

    records = []
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="FPV IR Inference & Caching"):
            imgs_tensor = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes = batch["bboxes"]
            b_idx = batch["batch_idx"]
            im_files = batch.get("im_file", [""] * imgs_tensor.shape[0])
            bs = imgs_tensor.shape[0]

            preds = model(imgs_tensor)
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

                pts_fp16 = peaks_list[b]["points"].astype(np.float16)
                scs_fp16 = peaks_list[b]["scores"].astype(np.float16)
                gt_fp16 = gt_pts.astype(np.float16)
                gt_bbox_fp16 = gt_bboxes.astype(np.float16)

                records.append({
                    "im_name": im_name,
                    "gt_pts": gt_fp16,
                    "gt_bboxes": gt_bbox_fp16,
                    "pred_points": pts_fp16,
                    "pred_scores": scs_fp16,
                })

    cache_out.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_out, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(colorstr("green", f"✅ Cache saved -> {cache_out.resolve()} (Total frames: {len(records)}) in {time.time()-t0:.1f}s"))
    return records


def evaluate_single_frame_threshold(records: list[dict], threshold: float, dist_thresh: float, match_mode: str) -> dict:
    stats = {"tp": 0, "fp": 0, "gt": 0}
    for record in records:
        gt_pts = np.asarray(record["gt_pts"], dtype=np.float32)
        gt_bboxes = np.asarray(record.get("gt_bboxes", []), dtype=np.float32)
        pred_pts = np.asarray(record["pred_points"], dtype=np.float32)
        pred_scores = np.asarray(record["pred_scores"], dtype=np.float32)
        pred_pts = pred_pts[pred_scores >= threshold]
        tp, fp, _ = match_predictions_to_gt(
            gt_pts,
            pred_pts,
            dist_thresh,
            gt_bboxes=gt_bboxes,
            match_mode=match_mode,
        )
        stats["tp"] += tp
        stats["fp"] += fp
        stats["gt"] += len(gt_pts)
    recall = 100.0 * stats["tp"] / max(1, stats["gt"])
    precision = 100.0 * stats["tp"] / max(1, stats["tp"] + stats["fp"])
    stats["recall"] = recall
    stats["precision"] = precision
    stats["f1"] = 2.0 * recall * precision / max(1e-6, recall + precision)
    return stats


def evaluate_sota_on_cache(records: list[dict], args):
    if args.search_threshold:
        thresholds = np.arange(args.th_min, args.th_max + args.th_step * 0.5, args.th_step)
        results = [evaluate_single_frame_threshold(records, float(th), args.dist_thresh, args.match_mode) for th in thresholds]
        best = max(results, key=lambda result: (result["f1"], result["recall"], result["precision"]))
        best_index = results.index(best)
        print("\nSINGLE-FRAME THRESHOLD SEARCH")
        print(f"Best threshold: {thresholds[best_index]:.3f} | TP={best['tp']} FP={best['fp']} GT={best['gt']} | "
              f"Recall={best['recall']:.2f}% Precision={best['precision']:.2f}% F1={best['f1']:.4f}")
        print("Threshold search candidates:")
        for threshold, result in zip(thresholds, results):
            print(f"  th={threshold:.3f}: R={result['recall']:.2f}% P={result['precision']:.2f}% F1={result['f1']:.4f} TP={result['tp']} FP={result['fp']}")
        return

    print(colorstr("bold", "\n========================================================================================="))
    print(colorstr("bold", f"EVALUATING SYSTEM SOTA ON FPV IR DATASET ({len(records)} frames)"))
    print(colorstr("bold", "========================================================================================="))

    seq_records: dict[str, list[dict]] = {}
    for r in records:
        seq = extract_seq_name(r["im_name"])
        seq_records.setdefault(seq, []).append(r)

    print(f"Total Unique Sequences: {len(seq_records)}")

    tracker_cfg = {
        "max_age": 3,
        "min_hits": 3,
        "match_dist": 12.0,
        "max_match_dist": 18.0,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_displacement": 2.5,
        "sky_ratio": 0.60,
        "img_h": args.imgsz,
    }
    smoother_cfg = {
        "stitch_max_gap": 4,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": 5,
        "max_infill_gap": 3,
        "min_track_hits": 3,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_rigid_displacement": 2.0,
        "max_rigid_variance": 0.5,
        "min_hits_for_prune": 8,
    }

    grand_base = {"tp": 0, "fp": 0, "gt": 0}
    grand_bidi = {"tp": 0, "fp": 0, "gt": 0}

    print(f"{'Sequence Name':<28} | {'Single-Frame (th=0.22)':<28} | {'System SOTA (Bidi+Prune)':<28}")
    print("-" * 92)

    for seq_name in sorted(seq_records.keys(), key=natural_sort_key):
        recs = seq_records[seq_name]
        eval_res = evaluate_sequence_bidirectional(
            records=recs,
            dist_thresh=args.dist_thresh,
            th_base=0.22,
            th_salvage=0.06,
            th_ground=0.35,
            sky_ratio=0.60,
            img_h=args.imgsz,
            tracker_config=tracker_cfg,
            smoother_config=smoother_cfg,
            match_mode=args.match_mode,
        )

        mb = eval_res["baseline"]
        ms = eval_res["bidirectional"]

        for k in ("tp", "fp", "gt"):
            grand_base[k] += int(mb[k])
            grand_bidi[k] += int(ms[k])

        base_str = f"R:{mb['recall']:.1f}% P:{mb['precision']:.1f}% F1:{mb['f1']:.3f}"
        sota_str = f"R:{ms['recall']:.1f}% P:{ms['precision']:.1f}% F1:{ms['f1']:.3f}"
        print(f"{seq_name:<28} | {base_str:<28} | {sota_str:<28}")

    # Overall Summary
    print("=" * 92)
    b_rec = grand_base["tp"] / max(1, grand_base["gt"]) * 100.0
    b_prec = grand_base["tp"] / max(1, grand_base["tp"] + grand_base["fp"]) * 100.0
    b_f1 = 2 * b_rec * b_prec / max(1e-6, b_rec + b_prec)

    s_rec = grand_bidi["tp"] / max(1, grand_bidi["gt"]) * 100.0
    s_prec = grand_bidi["tp"] / max(1, grand_bidi["tp"] + grand_bidi["fp"]) * 100.0
    s_f1 = 2 * s_rec * s_prec / max(1e-6, s_rec + s_prec)

    print(colorstr("bold", f"OVERALL ZERO-SHOT SOTA EVALUATION REPORT (Match Mode: {args.match_mode.upper()})"))
    print(f"Total Ground-Truth Frames : {grand_bidi['gt']}")
    print(f"1. Single-Frame Baseline (th=0.22): TP={grand_base['tp']}, FP={grand_base['fp']}, Recall={b_rec:.2f}%, Precision={b_prec:.2f}%, F1={b_f1:.4f}")
    print(colorstr("green", colorstr("bold", f"2. System SOTA (Bidi + Pruning): TP={grand_bidi['tp']}, FP={grand_bidi['fp']}, Recall={s_rec:.2f}%, Precision={s_prec:.2f}%, F1={s_f1:.4f}")))
    print("=" * 92 + "\n")


def main():
    args = parse_args()
    data_path = Path(args.data)
    cache_out = Path(args.cache_output)
    weights_path = Path(args.weights)

    if not cache_out.is_absolute():
        cache_out = PROJECT_ROOT / cache_out

    if args.skip_inference_if_cached and cache_out.exists():
        print(f"[INFO] Loading existing inference cache: {cache_out}")
        with open(cache_out, "rb") as f:
            records = pickle.load(f)
    else:
        if not weights_path.exists():
            cand = PROJECT_ROOT / weights_path
            if cand.exists():
                weights_path = cand
            else:
                raise FileNotFoundError(f"Weights not found: {weights_path}")

        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
        model, stride = load_sota_model(weights_path, device)
        records = run_caching(model, stride, data_path, cache_out, args, device)

    evaluate_sota_on_cache(records, args)


if __name__ == "__main__":
    main()
