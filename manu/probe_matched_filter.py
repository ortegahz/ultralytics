#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Phase-1 Zero-Training Matched-Filter Feasibility Probe (ROC measurement).

Question
--------
The learned front-end produces a candidate within 8px of the target in only 26.4% of
wg2022_ir_020_split_03 GT frames (hard ceiling for post-processing). A classical
matched filter is the textbook detector for a point target on a smooth background, and the
raw channel carries ~+9 gray over background sigma ~2. This probe measures, with NO
training and NO model, whether a matched filter can separate the target from background:

    PD(k)  = fraction of GT with a local-max candidate (Z >= k) within 8px
    CAND/f = mean number of local-max candidates per frame above k
    FP/f   = same but excluding candidates within 8px of any GT

If even this idealized detector cannot reach a useful PD at an acceptable FP/f, the
candidate-generation route is dead and Phase 2 must not be attempted.

Method (classical IRST small-target detector)
---------------------------------------------
1. Band-pass matched filter (DoG):      R = G(sigma_c) * I - G(sigma_s) * I
2. CFAR normalization:                  Z = (R - local_mean(R)) / (local_std(R) + eps)
3. Detection: 3x3 local maxima of Z above threshold k (top-k not truncated)

Coordinate contract
-------------------
Labels are normalized to the NATIVE image, so GT native pixel = (box[0]*W, box[1]*H).
The 8px hit radius is therefore in native pixels. For 640x512 sequences the letterbox
scale is 1.0 (x unchanged, y padded), so native distance equals letterbox distance; for
512x512 controls the letterbox upscales by 1.25, so native 8px is slightly stricter.

Usage on server:
    python manu/probe_matched_filter.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --seq wg2022_ir_020_split_03 \
        --output-csv runs/badcase_analysis/matched_filter_roc.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr

DEFAULT_CONTROLS = [
    "wg2022_ir_011_split_03",
    "DJI_0175_2",
    "wg2022_ir_011_split_02",
    "wg2022_ir_020_split_01",
    "wg2022_ir_047_split_01",
]


def natural_sort_key(p: Path):
    s = p.stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_args():
    p = argparse.ArgumentParser(description="Zero-training matched-filter feasibility probe (ROC)")
    p.add_argument("--data-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    p.add_argument("--seq", type=str, default="wg2022_ir_020_split_03")
    p.add_argument("--control-seqs", type=str, default=",".join(DEFAULT_CONTROLS))
    p.add_argument(
        "--configs",
        type=str,
        default="1.0,4.0,8.0;0.8,3.0,6.0;1.5,6.0,10.0",
        help="Semicolon-separated sigma_c,sigma_s,sigma_n configs",
    )
    p.add_argument("--ks", type=str, default="2.5,3.0,3.5,4.0", help="Thresholds for the ROC table")
    p.add_argument("--hit-radius", type=float, default=8.0)
    p.add_argument("--margin", type=int, default=12, help="Border margin excluded from detection")
    p.add_argument("--max-frames-control", type=int, default=0, help="Hard frame cap per control sequence (0 = all)")
    p.add_argument("--max-gt-control", type=int, default=200, help="Stop a control sequence after this many GT boxes (0 = all)")
    p.add_argument("--max-frames-target", type=int, default=0, help="Frame cap for the target sequence (0 = all)")
    p.add_argument("--save-examples", type=int, default=0, help="Save N side-by-side examples for the target seq")
    p.add_argument("--example-dir", type=str, default="runs/badcase_analysis/mf_examples")
    p.add_argument("--output-csv", type=str, default="runs/badcase_analysis/matched_filter_roc.csv")
    return p.parse_args()


def read_gt_boxes(lbl_path: Path) -> List[Tuple[float, float, float, float]]:
    """Return native-normalized [cx, cy, w, h] rows."""
    if not lbl_path.exists():
        return []
    boxes = []
    with lbl_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                boxes.append(tuple(float(v) for v in parts[1:5]))
    return boxes


def matched_response(gray: np.ndarray, sigma_c: float, sigma_s: float) -> np.ndarray:
    center = cv2.GaussianBlur(gray, (0, 0), sigma_c)
    surround = cv2.GaussianBlur(gray, (0, 0), sigma_s)
    return center - surround


def cfar_zscore(response: np.ndarray, sigma_n: float) -> np.ndarray:
    mean = cv2.GaussianBlur(response, (0, 0), sigma_n)
    mean_sq = cv2.GaussianBlur(response * response, (0, 0), sigma_n)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return (response - mean) / (np.sqrt(var) + 1e-6)


def local_maxima(z: np.ndarray, margin: int, kmin: float):
    h, w = z.shape
    if h <= 2 * margin or w <= 2 * margin:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    core = z[margin : h - margin, margin : w - margin]
    dil = cv2.dilate(core, np.ones((3, 3), np.uint8))
    mask = (core >= dil - 1e-6) & (core >= kmin)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    coords = np.stack([xs.astype(np.float32) + margin, ys.astype(np.float32) + margin], axis=1)
    return coords, core[ys, xs].astype(np.float32)


def process_sequence(
    img_dir: Path,
    lbl_dir: Path,
    seq: str,
    configs: List[Tuple[float, float, float]],
    ks: List[float],
    hit_radius: float,
    margin: int,
    max_frames: int,
    kmin: float,
    max_gt: int = 0,
    save_examples: int = 0,
    example_dir: Path | None = None,
) -> List[dict]:
    files = sorted(
        [p for p in img_dir.glob(f"{seq}__*") if p.suffix.lower() in (".jpg", ".png")],
        key=natural_sort_key,
    )
    if max_frames and len(files) > max_frames:
        files = files[:max_frames]

    acc = [
        {
            "gt_best_z": [],
            "frame_max_z": [],
            "cand": {k: [] for k in ks},
            "fp": {k: [] for k in ks},
            "frames": 0,
            "gt": 0,
        }
        for _ in configs
    ]
    saved = 0
    gt_seen = 0

    for p in tqdm(files, desc=f"{seq[:26]:<26}", file=sys.stdout, leave=False):
        boxes = read_gt_boxes(lbl_dir / f"{p.stem}.txt")
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        gray = (img[:, :, 0] if img.ndim == 3 else img).astype(np.float32)
        h, w = gray.shape[:2]

        gts = []
        for b in boxes:
            cx, cy = b[0] * w, b[1] * h
            if margin <= cx < w - margin and margin <= cy < h - margin:
                gts.append((cx, cy))

        for ci, (sc, ss, sn) in enumerate(configs):
            response = matched_response(gray, sc, ss)
            z = cfar_zscore(response, sn)
            cand, zs = local_maxima(z, margin, kmin)
            acc[ci]["frames"] += 1
            acc[ci]["gt"] += len(gts)
            if z.shape[0] > 2 * margin and z.shape[1] > 2 * margin:
                acc[ci]["frame_max_z"].append(float(z[margin : h - margin, margin : w - margin].max()))

            for gx, gy in gts:
                if cand.shape[0]:
                    d = np.linalg.norm(cand - np.array([gx, gy], dtype=np.float32), axis=1)
                    near = d <= hit_radius
                    acc[ci]["gt_best_z"].append(float(zs[near].max()) if near.any() else 0.0)
                else:
                    acc[ci]["gt_best_z"].append(0.0)

            for k in ks:
                sel = zs >= k
                n = int(sel.sum())
                acc[ci]["cand"][k].append(n)
                if n == 0 or not gts:
                    acc[ci]["fp"][k].append(n)
                    continue
                pts = cand[sel]
                gt_arr = np.asarray(gts, dtype=np.float32)
                dmin = np.min(np.linalg.norm(pts[:, None, :] - gt_arr[None, :, :], axis=2), axis=1)
                acc[ci]["fp"][k].append(int(np.sum(dmin > hit_radius)))

            if ci == 0 and save_examples and saved < save_examples and gts:
                example_dir.mkdir(parents=True, exist_ok=True)
                vis = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                zdisp = np.clip((z - kmin) / 4.0, 0, 1)
                zvis = cv2.applyColorMap((zdisp * 255).astype(np.uint8), cv2.COLORMAP_JET)
                for gx, gy in gts:
                    cv2.circle(vis, (int(round(gx)), int(round(gy))), 8, (0, 0, 255), 1)
                    cv2.circle(zvis, (int(round(gx)), int(round(gy))), 8, (255, 255, 255), 1)
                cv2.imwrite(str(example_dir / f"{p.stem}_gray.png"), vis)
                cv2.imwrite(str(example_dir / f"{p.stem}_z.png"), zvis)
                saved += 1

        gt_seen += len(gts)
        if max_gt and gt_seen >= max_gt:
            break

    rows = []
    for ci, (sc, ss, sn) in enumerate(configs):
        a = acc[ci]
        gt_best = np.asarray(a["gt_best_z"], dtype=np.float32)
        frame_max = np.asarray(a["frame_max_z"], dtype=np.float32)
        row = {
            "seq": seq,
            "sigma_c": sc,
            "sigma_s": ss,
            "sigma_n": sn,
            "frames": a["frames"],
            "gt": a["gt"],
            "med_gt_z": float(np.median(gt_best)) if gt_best.size else 0.0,
            "p90_gt_z": float(np.percentile(gt_best, 90)) if gt_best.size else 0.0,
            "med_frame_max_z": float(np.median(frame_max)) if frame_max.size else 0.0,
        }
        for k in ks:
            pd = float(np.mean(gt_best >= k) * 100) if gt_best.size else 0.0
            row[f"PD@{k}"] = pd
            row[f"cand@{k}"] = float(np.mean(a["cand"][k])) if a["cand"][k] else 0.0
            row[f"fp@{k}"] = float(np.mean(a["fp"][k])) if a["fp"][k] else 0.0
        rows.append(row)
    return rows


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    img_dir = data_root / "images" / "val"
    lbl_dir = data_root / "labels" / "val"
    if not img_dir.exists():
        alt = Path("/home/manu/mnt/datasets/manu/uav_gmc_median")
        img_dir = alt / "images" / "val"
        lbl_dir = alt / "labels" / "val"

    configs = []
    for token in args.configs.split(";"):
        token = token.strip()
        if not token:
            continue
        sc, ss, sn = (float(v) for v in token.split(","))
        configs.append((sc, ss, sn))
    ks = [float(v) for v in args.ks.split(",")]
    kmin = min(ks)

    controls = [s.strip() for s in args.control_seqs.split(",") if s.strip()]
    sequences = [(args.seq, args.max_frames_target, 0)] + [
        (s, args.max_frames_control, args.max_gt_control) for s in controls
    ]

    all_rows = []
    for seq, frame_cap, gt_cap in sequences:
        rows = process_sequence(
            img_dir, lbl_dir, seq, configs, ks, args.hit_radius, args.margin, frame_cap, kmin,
            max_gt=gt_cap,
            save_examples=args.save_examples if seq == args.seq else 0,
            example_dir=Path(args.example_dir),
        )
        all_rows.extend(rows)

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[INFO] Wrote ROC table: {out}")

    for ci, (sc, ss, sn) in enumerate(configs):
        print("\n" + "=" * 150)
        print(colorstr("bold", colorstr("cyan", f"CONFIG  sigma_center={sc}  sigma_surround={ss}  sigma_noise={sn}")))
        print("=" * 150)
        header = (
            f"{'Sequence':<28} | {'GT':<6} | {'medGTz':<8} | {'medFrameMaxZ':<12} | "
            + " | ".join(f"{'PD@' + str(k):<9}" for k in ks)
            + " | "
            + " | ".join(f"{'FP/f@' + str(k):<10}" for k in ks)
        )
        print(header)
        print("-" * 150)
        for row in all_rows:
            if row["sigma_c"] != sc or row["sigma_s"] != ss or row["sigma_n"] != sn:
                continue
            pd_cols = " | ".join(f"{row['PD@' + str(k)]:>8.1f}% " for k in ks)
            fp_cols = " | ".join(f"{row['fp@' + str(k)]:>9.2f} " for k in ks)
            line = (
                f"{row['seq']:<28} | {row['gt']:<6} | {row['med_gt_z']:>8.2f} | {row['med_frame_max_z']:>12.2f} | "
                f"{pd_cols}| {fp_cols}"
            )
            if row["seq"] == args.seq:
                print(colorstr("bold", colorstr("magenta", line)))
            else:
                print(line)
        print("=" * 150)
        print("medGTz = median matched-filter Z at GT | medFrameMaxZ = median of each frame's max Z (background peak level)")
        print("PD@k   = % of GT with a local-max candidate (Z>=k) within hit radius | FP/f@k = non-target candidates per frame")
        print("Feasibility: PD must clearly exceed the learned front-end's 26.4% coverage while FP/f stays low.")

    print(colorstr("bold", "\n[READ] If medGTz << medFrameMaxZ, the target ranks below typical background peaks and no single"))
    print("       threshold can separate it. If medGTz > medFrameMaxZ, separability exists and the FP cost per PD decides.")


if __name__ == "__main__":
    main()
