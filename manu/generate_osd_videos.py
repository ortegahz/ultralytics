#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate per-sequence OSD verification videos directly from inference cache (.pkl).

Key features:
1. Pure post-processing: Zero model forward pass. Reads precomputed predictions directly
   from inference_cache.pkl (or fallback to image disk lookup).
2. Professional OSD Overlays:
   - GT targets: Green box + center cross + target ID.
   - Predictions: Red circle / cross + confidence score + sub-pixel coordinates.
   - Matching metrics on frame: TP (hit), FP (false alarm), FN (missed).
   - Real-time HUD stats banner: Sequence Name, Frame Number, Threshold, Hit Count.
3. Groups frames automatically by video sequence and naturally sorts frames.
4. Flexible filtering: Option to render all sequences, Top-N worst sequences, or specific sequences.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Render OSD videos for sequences from inference cache (.pkl)")
    parser.add_argument(
        "--cache-file",
        type=str,
        default="runs/badcase_analysis/inference_cache.pkl",
        help="Path to inference_cache.pkl",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml to locate original images if cache only stores points",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/badcase_analysis/osd_videos",
        help="Output directory to save OSD mp4 videos",
    )
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for predictions")
    parser.add_argument(
        "--conf-sky",
        type=float,
        default=None,
        help="Optional lower confidence threshold for sky region (e.g. 0.06). If not set, uses --conf everywhere.",
    )
    parser.add_argument(
        "--sky-ratio",
        type=float,
        default=0.60,
        help="Upper fraction of image treated as sky (default: 0.60, meaning top 60%% is sky)",
    )
    parser.add_argument("--dist-thresh", type=float, default=8.0, help="Distance tolerance in pixels (default: 8.0px)")
    parser.add_argument("--fps", type=float, default=25.0, help="Video framerate (default: 25)")
    parser.add_argument(
        "--sequences",
        type=str,
        default="",
        help="Comma-separated sequence filters (e.g. '02_6321,wg2022_ir_020'). Empty means all sequences.",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=0,
        help="Max number of sequences to export (0 means unlimited)",
    )
    return parser.parse_args()


def natural_sort_key(s: str):
    """Sort strings using natural numeric ordering."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_seq_and_frame(im_name: str) -> tuple[str, int]:
    """
    Extract sequence identifier and frame number from image file name.
    Examples:
      02_6321_0274-2773___000416.jpg -> ('02_6321_0274-2773', 416)
      wg2022_ir_020_split_000123.jpg -> ('wg2022_ir_020_split', 123)
    """
    stem = Path(im_name).stem
    if "___" in stem:
        parts = stem.split("___")
        seq = parts[0]
        match = re.search(r"(\d+)$", parts[1])
        frame_idx = int(match.group(1)) if match else 0
        return seq, frame_idx

    match = re.search(r"^(.*?)(?:[_-]+)?(\d+)$", stem)
    if match:
        seq = match.group(1).rstrip("_-")
        frame_idx = int(match.group(2))
        return seq, frame_idx

    return stem, 0


def build_image_lookup(val_source: str | Path | list) -> dict[str, Path]:
    """Index image filenames to disk paths for loading full images."""
    print("Indexing dataset images on disk...")
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
    print(f"Indexed {len(lookup)} images.")
    return lookup


def render_osd_frame(
    frame: np.ndarray,
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
    pred_scores: np.ndarray,
    dist_thresh: float,
    conf_thresh: float,
    seq_name: str,
    frame_idx: int,
    conf_sky: float | None = None,
    sky_ratio: float = 0.60,
) -> tuple[np.ndarray, int, int, int]:
    """
    Draw HUD and bounding marks:
      - Green Box & Cross: GT target
      - Red Circle & Label: Model Prediction (TP: highlighted green ring, FP: red ring)
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Filter predictions by confidence (supports dual-domain partitioned threshold)
    if conf_sky is not None and len(pred_pts) > 0:
        sky_y_boundary = h * sky_ratio
        is_sky = pred_pts[:, 1] < sky_y_boundary
        keep_sky = is_sky & (pred_scores >= conf_sky)
        keep_ground = (~is_sky) & (pred_scores >= conf_thresh)
        keep = keep_sky | keep_ground
    else:
        keep = pred_scores >= conf_thresh

    preds = pred_pts[keep]
    scores = pred_scores[keep]

    # Perform matching
    matched_gt = set()
    matched_pred = set()

    if len(preds) > 0 and len(gt_pts) > 0:
        diff = preds[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))

        p_inds, g_inds = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
        for p_i, g_i in zip(p_inds, g_inds):
            if dists[p_i, g_i] > dist_thresh:
                break
            if p_i not in matched_pred and g_i not in matched_gt:
                matched_pred.add(p_i)
                matched_gt.add(g_i)

    tp_count = len(matched_gt)
    fp_count = len(preds) - tp_count
    fn_count = len(gt_pts) - tp_count

    # 1. Draw Ground Truths
    for g_i, (gx, gy) in enumerate(gt_pts):
        ix, iy = int(round(gx)), int(round(gy))
        is_hit = g_i in matched_gt
        color = (0, 255, 0) if is_hit else (0, 165, 255)  # Green for Hit, Orange for Missed
        box_r = int(round(dist_thresh))

        # Target bounding circle / box
        cv2.rectangle(canvas, (ix - box_r, iy - box_r), (ix + box_r, iy + box_r), color, 1)
        cv2.drawMarker(canvas, (ix, iy), color, cv2.MARKER_CROSS, 6, 1)

        status_text = f"GT-{g_i+1}" if is_hit else f"GT-{g_i+1}[FN]"
        cv2.putText(canvas, status_text, (ix + box_r + 2, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # 2. Draw Predictions
    for p_i, ((px, py), sc) in enumerate(zip(preds, scores)):
        ix, iy = int(round(px)), int(round(py))
        is_tp = p_i in matched_pred
        color = (0, 255, 0) if is_tp else (0, 0, 255)  # Green for TP, Red for FP

        # Predicted point
        cv2.circle(canvas, (ix, iy), 3, color, -1)
        cv2.circle(canvas, (ix, iy), 7, color, 1)

        label = f"{sc:.2f}"
        if not is_tp:
            label += " [FP]"
        cv2.putText(canvas, label, (ix + 6, iy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # 3. Draw HUD Information Bar (Top Banner)
    hud_h = 32
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    conf_info = f"Conf>={conf_thresh:.2f}" if conf_sky is None else f"Sky>={conf_sky:.2f}|Gnd>={conf_thresh:.2f}"
    info_left = f"SEQ: {seq_name} | Frame: {frame_idx:04d} | {conf_info} | Tol={dist_thresh:.1f}px"
    cv2.putText(canvas, info_left, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    info_right = f"GT: {len(gt_pts)}  TP: {tp_count}  FP: {fp_count}  FN: {fn_count}"
    text_size = cv2.getTextSize(info_right, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
    cv2.putText(canvas, info_right, (w - text_size[0] - 15, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas, tp_count, fp_count, fn_count


def main():
    args = parse_args()
    cache_path = Path(args.cache_file)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    if not cache_path.exists():
        raise FileNotFoundError(f"Inference cache not found: {cache_path}. Run diagnose_heatmap_badcases.py first!")

    print(f"\n>>> Loading cached inferences from: {cache_path}")
    with open(cache_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} image predictions from cache.\n")

    # Group records by sequence
    seq_groups: dict[str, list[dict]] = {}
    for r in records:
        im_name = r["im_name"]
        seq_name, frame_idx = extract_seq_and_frame(im_name)
        r["_frame_idx"] = frame_idx
        seq_groups.setdefault(seq_name, []).append(r)

    # Filter sequences if specified
    if args.sequences:
        targets = [s.strip() for s in args.sequences.split(",") if s.strip()]
        filtered_groups = {}
        for s_name, group in seq_groups.items():
            if any(t in s_name for t in targets):
                filtered_groups[s_name] = group
        seq_groups = filtered_groups

    if args.max_seqs > 0:
        sorted_keys = sorted(seq_groups.keys(), key=lambda k: len(seq_groups[k]), reverse=True)[: args.max_seqs]
        seq_groups = {k: seq_groups[k] for k in sorted_keys}

    print(f"Total sequences to render: {len(seq_groups)}")

    # Prepare Image Disk Lookup if needed
    data_dict = check_det_dataset(args.data)
    img_lookup = build_image_lookup(data_dict["val"])

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Render each sequence into an MP4 video
    for seq_name, group in sorted(seq_groups.items()):
        # Sort frames chronologically
        group.sort(key=lambda r: (r["_frame_idx"], natural_sort_key(r["im_name"])))

        video_path = out_dir / f"{seq_name}_osd.mp4"
        writer = None

        total_tp, total_fp, total_fn = 0, 0, 0

        pbar = tqdm(group, desc=f"Rendering {seq_name[:24]} ({len(group)}f)", leave=False)
        for r in pbar:
            im_name = r["im_name"]
            gt_pts = r["gt_pts"]
            pred_pts = r["pred_points"]
            pred_scores = r["pred_scores"]
            frame_idx = r["_frame_idx"]

            # Obtain raw image (from cached numpy array or from disk)
            frame = r.get("img_hwc")
            if frame is None:
                img_path = img_lookup.get(im_name)
                if img_path is None or not img_path.exists():
                    for ext in [".jpg", ".png", ".jpeg"]:
                        alt = img_lookup.get(im_name + ext)
                        if alt and alt.exists():
                            img_path = alt
                            break
                if img_path and img_path.exists():
                    frame = cv2.imread(str(img_path))

            if frame is None:
                continue

            # Resize to 640x640 if not already
            if frame.shape[0] != 640 or frame.shape[1] != 640:
                frame = cv2.resize(frame, (640, 640))

            osd_frame, tp, fp, fn = render_osd_frame(
                frame=frame,
                gt_pts=gt_pts,
                pred_pts=pred_pts,
                pred_scores=pred_scores,
                dist_thresh=args.dist_thresh,
                conf_thresh=args.conf,
                seq_name=seq_name,
                frame_idx=frame_idx,
                conf_sky=args.conf_sky,
                sky_ratio=args.sky_ratio,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            h, w = osd_frame.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))

            writer.write(osd_frame)

        if writer is not None:
            writer.release()
            total_gt = total_tp + total_fn
            rec = total_tp / (total_gt + 1e-6) * 100
            prec = total_tp / (total_tp + total_fp + 1e-6) * 100
            print(f"[OK] {video_path.name:<45} | GT:{total_gt:<4} TP:{total_tp:<4} FP:{total_fp:<4} FN:{total_fn:<4} | Rec:{rec:5.1f}% Prec:{prec:5.1f}%")

    print("\n" + "=" * 75)
    print(f"ALL VIDEOS GENERATED IN: {out_dir}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
