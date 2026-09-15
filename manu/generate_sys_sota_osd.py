#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate System SOTA OSD Verification Video for a given sequence.
Applies Trial 0474 + Bidirectional Temporal Smoothing + Rigid Static Pruner (System SOTA).

Usage:
    python manu/generate_sys_sota_osd.py \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
        --seq DJI_0175_2 \
        --output-dir runs/sys_sota_osd \
        --dist-thresh 8.0 \
        --th-base 0.22 \
        --th-salvage 0.06 \
        --th-ground 0.35 \
        --min-hits-infill 5 \
        --min-rigid-disp 2.0 \
        --max-rigid-var 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import colorstr
from manu.eval_bidirectional_track_fusion import (
    evaluate_sequence_bidirectional,
    extract_seq_name,
    natural_sort_key,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render System SOTA OSD video for a sequence")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/gmc_eval/uav_median_trial0474_cache.pkl",
        help="Path to Trial 0474 cache file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
        help="Path to data.yaml (or dataset root) to locate original images",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="DJI_0175_2",
        help="Target sequence name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/sys_sota_osd",
        help="Directory to save output MP4 video",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="GJB pixel tolerance (default: 8.0px)")
    parser.add_argument("--th-base", type=float, default=0.22, help="Seed detection threshold (default: 0.22)")
    parser.add_argument("--th-salvage", type=float, default=0.06, help="Kinematic salvage threshold (default: 0.06)")
    parser.add_argument("--th-ground", type=float, default=0.35, help="Clutter/ground threshold (default: 0.35)")
    parser.add_argument("--min-hits-infill", type=int, default=5, help="Min track hits for infill (default: 5)")
    parser.add_argument("--min-rigid-disp", type=float, default=2.0, help="Min displacement for static pruner")
    parser.add_argument("--max-rigid-var", type=float, default=0.5, help="Max coordinate variance for static pruner")
    parser.add_argument("--img-h", type=int, default=640)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--fps", type=float, default=25.0)
    return parser.parse_args()


def build_image_lookup(val_source: str | Path | list) -> dict[str, Path]:
    """Index image filenames to disk paths for loading full images."""
    lookup = {}
    if isinstance(val_source, (str, Path)):
        val_dirs = [Path(val_source)]
    else:
        val_dirs = [Path(p) for p in val_source]

    for d in val_dirs:
        if d.is_file():
            with open(d, "r", encoding="utf-8") as f:
                for line in f:
                    p = Path(line.strip())
                    if p.exists():
                        lookup[p.name] = p
        elif d.is_dir():
            for p in d.rglob("*.*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    lookup[p.name] = p
    return lookup


def render_osd_frame(
    frame: np.ndarray,
    gt_pts: np.ndarray,
    sys_dets: list[dict],
    dist_thresh: float,
    seq_name: str,
    frame_idx: int,
    total_frames: int,
    cum_tp: int,
    cum_fp: int,
    cum_gt: int,
) -> tuple[np.ndarray, int, int, int]:
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    pred_pts = np.array([d["pos"] for d in sys_dets], dtype=np.float32) if len(sys_dets) > 0 else np.zeros((0, 2), dtype=np.float32)
    scores = np.array([d["score"] for d in sys_dets], dtype=np.float32) if len(sys_dets) > 0 else np.zeros((0,), dtype=np.float32)
    infilled = np.array([d.get("infilled", False) for d in sys_dets], dtype=bool) if len(sys_dets) > 0 else np.zeros((0,), dtype=bool)
    tids = [d.get("track_id", 0) for d in sys_dets]

    matched_gt = set()
    matched_pred = set()

    if len(pred_pts) > 0 and len(gt_pts) > 0:
        diff = pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        p_inds, g_inds = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
        for p_i, g_i in zip(p_inds, g_inds):
            if dists[p_i, g_i] > dist_thresh:
                break
            if p_i not in matched_pred and g_i not in matched_gt:
                matched_pred.add(p_i)
                matched_gt.add(g_i)

    tp_count = len(matched_gt)
    fp_count = len(pred_pts) - tp_count
    fn_count = len(gt_pts) - tp_count

    # 1. Draw Ground Truths
    for g_i, (gx, gy) in enumerate(gt_pts):
        ix, iy = int(round(gx)), int(round(gy))
        is_hit = g_i in matched_gt
        color = (0, 255, 0) if is_hit else (0, 165, 255)  # Green for Hit, Orange for Missed
        box_r = int(round(dist_thresh))

        cv2.rectangle(canvas, (ix - box_r, iy - box_r), (ix + box_r, iy + box_r), color, 1)
        cv2.drawMarker(canvas, (ix, iy), color, cv2.MARKER_CROSS, 6, 1)

        status_text = f"GT-{g_i+1}" if is_hit else f"GT-{g_i+1}[FN]"
        cv2.putText(canvas, status_text, (ix + box_r + 2, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # 2. Draw Predictions (System SOTA)
    for p_i, ((px, py), sc, inf, tid) in enumerate(zip(pred_pts, scores, infilled, tids)):
        ix, iy = int(round(px)), int(round(py))
        is_tp = p_i in matched_pred
        color = (0, 255, 0) if is_tp else (0, 0, 255)  # Green for TP, Red for FP

        cv2.circle(canvas, (ix, iy), 3, color, -1)
        cv2.circle(canvas, (ix, iy), 7, color, 1)

        tag = "Infill" if inf else f"{sc:.2f}"
        if not is_tp:
            tag += " [FP]"
        label = f"T{tid}:{tag}"
        cv2.putText(canvas, label, (ix + 6, iy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # 3. Draw Top HUD Banner
    hud_h = 32
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    cur_rec = cum_tp / max(1, cum_gt) * 100.0
    cur_prec = cum_tp / max(1, cum_tp + cum_fp) * 100.0
    cur_f1 = 2 * cur_rec * cur_prec / max(1e-6, cur_rec + cur_prec)

    info_left = f"SYS SOTA | {seq_name} | F:{frame_idx:04d}/{total_frames:04d} | Tol={dist_thresh:.1f}px"
    cv2.putText(canvas, info_left, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    info_right = f"Cum: Rec:{cur_rec:.1f}% Prec:{cur_prec:.1f}% F1:{cur_f1:.2f} | Frame TP:{tp_count} FP:{fp_count} FN:{fn_count}"
    text_size = cv2.getTextSize(info_right, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    cv2.putText(canvas, info_right, (w - text_size[0] - 12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas, tp_count, fp_count, fn_count


def main():
    args = parse_args()

    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        for cand in [
            PROJECT_ROOT / cache_path,
            Path("/tmp/pycharm_project_10ae9e2e") / cache_path,
            Path("/home/manu/mnt/pycharm_project_10ae9e2e") / cache_path,
        ]:
            if cand.exists():
                cache_path = cand
                break

    if not cache_path.exists():
        print(colorstr("red", f"[ERROR] Cache file not found: {args.cache_file}"))
        sys.exit(1)

    print(f"[INFO] Loading inference cache from: {cache_path}")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)

    # Filter records for the target sequence
    seq_records = []
    for r in records:
        if args.seq in r["im_name"]:
            seq_records.append(r)

    if not seq_records:
        print(colorstr("red", f"[ERROR] No records found for sequence '{args.seq}' in cache!"))
        sys.exit(1)

    print(f"[INFO] Found {len(seq_records)} frames for sequence '{args.seq}'. Running System SOTA tracking...")

    tracker_config = {
        "max_age": 3,
        "min_hits": 3,
        "match_dist": 12.0,
        "max_match_dist": 18.0,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_displacement": 2.5,
        "sky_ratio": 0.60,
        "img_h": args.img_h,
    }

    smoother_config = {
        "stitch_max_gap": 4,
        "stitch_max_dist": 25.0,
        "min_hits_for_infill": args.min_hits_infill,
        "max_infill_gap": 3,
        "min_track_hits": 3,
        "min_track_score": 0.08,
        "instant_conf": 0.25,
        "min_rigid_displacement": args.min_rigid_disp,
        "max_rigid_variance": args.max_rigid_var,
        "min_hits_for_prune": 8,
    }

    eval_res = evaluate_sequence_bidirectional(
        records=seq_records,
        dist_thresh=args.dist_thresh,
        th_base=args.th_base,
        th_salvage=args.th_salvage,
        th_ground=args.th_ground,
        sky_ratio=0.60,
        img_h=args.img_h,
        tracker_config=tracker_config,
        smoother_config=smoother_config,
    )

    bidi_metrics = eval_res["bidirectional"]
    bidi_frame_dets = eval_res["bidi_frame_dets"]
    records_sorted = eval_res["records_sorted"]

    print("\n" + "=" * 80)
    print(colorstr("bold", f"SYSTEM SOTA RESULTS FOR {args.seq}:"))
    print(
        f"GT: {bidi_metrics['gt']} | TP: {bidi_metrics['tp']} | FP: {bidi_metrics['fp']} | "
        f"Recall: {bidi_metrics['recall']:.2f}% | Precision: {bidi_metrics['precision']:.2f}% | F1: {bidi_metrics['f1']:.2f}"
    )
    print("=" * 80 + "\n")

    # Locate image files
    img_lookup = {}
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if data_path.is_file() and data_path.suffix in (".yaml", ".yml"):
        data_dict = check_det_dataset(str(data_path))
        val_source = data_dict.get("val")
        if val_source:
            print(f"[INFO] Indexing dataset images from: {val_source}")
            img_lookup = build_image_lookup(val_source)

    if not img_lookup:
        # Fallback directory check
        for cand in [
            Path("/mnt/data/siping/datasets/manu/uav_gmc_median/images/val"),
            Path("/home/manu/mnt/datasets/manu/uav_gmc_median/images/val"),
            Path("/mnt/data/siping/datasets/manu/uav/images/val"),
            Path("/home/manu/mnt/datasets/manu/uav/images/val"),
        ]:
            if cand.exists():
                print(f"[INFO] Indexing fallback directory: {cand}")
                img_lookup = build_image_lookup(cand)
                break

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_video_path = out_dir / f"{args.seq}_sys_sota_osd.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    cum_tp, cum_fp, cum_gt = 0, 0, 0
    total_frames = len(records_sorted)

    print(f"[INFO] Rendering System SOTA OSD video to: {out_video_path}...")
    for f_idx, r in enumerate(tqdm(records_sorted, desc=f"Rendering {args.seq}")):
        im_name = r["im_name"]
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        sys_dets = bidi_frame_dets[f_idx]

        # Load image
        frame = r.get("img_hwc")
        if frame is None:
            p = img_lookup.get(im_name)
            if p is None or not p.exists():
                for ext in [".jpg", ".png", ".jpeg"]:
                    alt = img_lookup.get(Path(im_name).stem + ext)
                    if alt and alt.exists():
                        p = alt
                        break
            if p and p.exists():
                frame = cv2.imread(str(p))

        if frame is None:
            # Fallback black canvas
            frame = np.zeros((args.img_h, args.img_w, 3), dtype=np.uint8)

        if frame.shape[0] != args.img_h or frame.shape[1] != args.img_w:
            frame = cv2.resize(frame, (args.img_w, args.img_h))

        osd_frame, tp, fp, fn = render_osd_frame(
            frame=frame,
            gt_pts=gt_pts,
            sys_dets=sys_dets,
            dist_thresh=args.dist_thresh,
            seq_name=args.seq,
            frame_idx=f_idx,
            total_frames=total_frames,
            cum_tp=cum_tp,
            cum_fp=cum_fp,
            cum_gt=cum_gt,
        )

        cum_tp += tp
        cum_fp += fp
        cum_gt += len(gt_pts)

        if writer is None:
            writer = cv2.VideoWriter(str(out_video_path), fourcc, args.fps, (args.img_w, args.img_h))

        writer.write(osd_frame)

    if writer is not None:
        writer.release()

    print(colorstr("green", f"\n[SUCCESS] OSD Video generated successfully: {out_video_path}"))
    print(f"Final Cumulative Metrics: GT={cum_gt} TP={cum_tp} FP={cum_fp} FN={cum_gt - cum_tp} | "
          f"Recall={cum_tp / max(1, cum_gt) * 100:.2f}% Prec={cum_tp / max(1, cum_tp + cum_fp) * 100:.2f}%\n")


if __name__ == "__main__":
    main()
