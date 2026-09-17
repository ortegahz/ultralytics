#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Total Integrated Energy Proxy Estimator for All Validation Sequences.

Hypothesis under test
---------------------
The per-pixel SNR (MAD-Z / peak contrast) is optimistic for tiny targets because it
measures only the brightest pixel, ignoring target AREA. A detector ultimately responds
to the total integrated signal ~ area x |brightness|. wg2022_ir_020_split_03 has a 2px,
+9-gray target whose total energy is far lower than sequences whose per-pixel contrast
is comparable (e.g. DJI_0175_2: 8px, +7-gray).

This script computes, per GT box, a label-free proxy:
    area  = box_width_px * box_height_px
    signal = polarity-aware signed deviation (peak or trough vs local background median)
    energy = area * |signal|
and ranks all sequences by median energy ascending (worst first).

It optionally joins the model's response rate (% GT with any peak within 8px) from the
inference cache so the energy <-> response relationship is visible in one table.

Coordinate contract
-------------------
All pixel sampling uses NATIVE coordinates (box[0]*W, box[1]*H). Response rates are read
from the cache in its own letterbox space and never mixed with the pixel sampling.

Usage on server:
    python manu/measure_total_energy.py \
        --data-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --cache-file runs/gmc_eval/uav_median_trial0474_cache.pkl
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.utils import colorstr


def natural_sort_key(p: Path):
    s = p.stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def extract_seq_name(im_name: str) -> str:
    stem = Path(im_name).stem
    if "___" in stem:
        return stem.split("___")[0]
    if "__" in stem:
        return stem.split("__")[0]
    m = re.search(r"^(.*?)(?:[_-]+)?\d{3,}$", stem)
    return m.group(1).rstrip("_-") if m else stem


def parse_args():
    p = argparse.ArgumentParser(description="Total integrated energy proxy across all sequences")
    p.add_argument("--data-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    p.add_argument("--cache-file", type=str, default="runs/gmc_eval/uav_median_trial0474_cache.pkl", help="Optional: join response rate")
    p.add_argument("--no-cache", action="store_true", help="Skip cache join")
    p.add_argument("--dist-thresh", type=float, default=8.0)
    return p.parse_args()


def signed_deviation(ch0: np.ndarray, cx: int, cy: int) -> float:
    """Polarity-aware signal: peak-bg_median if bright dominates, else trough-bg_median (negative)."""
    h, w = ch0.shape[:2]
    if not (10 <= cx < w - 10 and 10 <= cy < h - 10):
        return 0.0
    target = ch0[cy - 2 : cy + 3, cx - 2 : cx + 3].astype(np.float32)
    outer = ch0[cy - 10 : cy + 11, cx - 10 : cx + 11].astype(np.float32)
    mask = np.ones((21, 21), dtype=bool)
    mask[5:16, 5:16] = False
    bg = outer[mask]
    med = float(np.median(bg))
    peak = float(np.max(target))
    trough = float(np.min(target))
    up = peak - med
    down = med - trough
    return up if abs(up) >= abs(down) else -down


def response_rate(records: List[dict], dist_thresh: float) -> Dict[str, float]:
    n_near = 0
    n_gt = 0
    top1_scores = []
    for rec in records:
        gt = np.asarray(rec["gt_pts"], dtype=np.float32)
        pred = np.asarray(rec["pred_points"], dtype=np.float32)
        scs = np.asarray(rec["pred_scores"], dtype=np.float32)
        if gt.size == 0:
            continue
        top1_scores.append(float(scs.max()) if scs.size else 0.0)
        if pred.shape[0] == 0:
            n_gt += gt.shape[0]
            continue
        for g in gt:
            d = np.linalg.norm(pred - g, axis=1)
            if float(d.min()) <= dist_thresh:
                n_near += 1
            n_gt += 1
    return {
        "pct_near": float(n_near / n_gt * 100) if n_gt else 0.0,
        "median_top1": float(np.median(top1_scores)) if top1_scores else 0.0,
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    img_dir = data_root / "images" / "val"
    lbl_dir = data_root / "labels" / "val"
    if not img_dir.exists():
        alt = Path("/home/manu/mnt/datasets/manu/uav_gmc_median")
        img_dir = alt / "images" / "val"
        lbl_dir = alt / "labels" / "val"

    files = sorted(
        [p for p in img_dir.glob("*.jpg")] + [p for p in img_dir.glob("*.png")],
        key=natural_sort_key,
    )
    print(f"[INFO] Scanning {len(files):,} validation frames for GT boxes...")

    # cache join
    cache = None
    grouped_cache: Dict[str, List[dict]] = {}
    if not args.no_cache:
        cache_path = Path(args.cache_file)
        if not cache_path.is_absolute():
            for cand in [PROJECT_ROOT / cache_path, Path("/tmp/pycharm_project_10ae9e2e") / cache_path]:
                if cand.exists():
                    cache_path = cand
                    break
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            for rec in cache:
                grouped_cache.setdefault(extract_seq_name(rec["im_name"]), []).append(rec)
            print(f"[INFO] Loaded cache ({len(cache):,} frames) for response-rate join.")

    seq_areas: Dict[str, List[float]] = {}
    seq_signals: Dict[str, List[float]] = {}
    seq_energy: Dict[str, List[float]] = {}
    resolutions: Dict[str, Dict[str, int]] = {}

    for p in tqdm(files, desc="Measuring energy", file=sys.stdout):
        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        boxes = []
        with lbl.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    boxes.append([float(v) for v in parts[1:5]])
        if not boxes:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        seq = extract_seq_name(p.name)
        resolutions.setdefault(seq, {})[f"{w}x{h}"] = resolutions.setdefault(seq, {}).get(f"{w}x{h}", 0) + 1
        ch0 = img[:, :, 0] if img.ndim == 3 else img
        for b in boxes:
            cx, cy = int(round(b[0] * w)), int(round(b[1] * h))
            bw, bh = b[2] * w, b[3] * h
            area = max(bw * bh, 1e-3)
            sig = signed_deviation(ch0, cx, cy)
            seq_areas.setdefault(seq, []).append(area)
            seq_signals.setdefault(seq, []).append(abs(sig))
            seq_energy.setdefault(seq, []).append(area * abs(sig))

    rows = []
    for seq, energy in seq_energy.items():
        rows.append({
            "seq": seq,
            "n_boxes": len(energy),
            "med_area": float(np.median(seq_areas[seq])),
            "med_signal": float(np.median(seq_signals[seq])),
            "med_energy": float(np.median(energy)),
            "mean_energy": float(np.mean(energy)),
            "resolution": ",".join(resolutions[seq].keys()),
            "pct_near": None,
            "median_top1": None,
        })
        if seq in grouped_cache:
            rr = response_rate(grouped_cache[seq], args.dist_thresh)
            rows[-1]["pct_near"] = rr["pct_near"]
            rows[-1]["median_top1"] = rr["median_top1"]

    rows.sort(key=lambda r: r["med_energy"])

    print("\n" + "=" * 140)
    print(colorstr("bold", colorstr("cyan", "TOTAL INTEGRATED ENERGY PROXY LEADERBOARD (SORTED ASCENDING, WORST FIRST)")))
    print("=" * 140)
    hdr = (
        f"{'Rank':<4} | {'Sequence':<28} | {'boxes':<6} | {'res':<8} | {'medArea(px2)':<12} | "
        f"{'med|sig|(gray)':<13} | {'medEnergy':<12} | {'meanEnergy':<12} | {'%near8px':<9} | {'top1Score':<9}"
    )
    print(hdr)
    print("-" * 140)
    for i, r in enumerate(rows, 1):
        near = f"{r['pct_near']:.1f}%" if r["pct_near"] is not None else "-"
        top1 = f"{r['median_top1']:.3f}" if r["median_top1"] is not None else "-"
        line = (
            f"{i:<4} | {r['seq']:<28} | {r['n_boxes']:<6} | {r['resolution']:<8} | {r['med_area']:>12.1f} | "
            f"{r['med_signal']:>13.2f} | {r['med_energy']:>12.1f} | {r['mean_energy']:>12.1f} | {near:>9} | {top1:>9}"
        )
        if r["seq"] == "wg2022_ir_020_split_03":
            print(colorstr("bold", colorstr("magenta", f"👉 {line} 👈 [TARGET PROBE]")))
        else:
            print(line)
    print("=" * 140)
    print("energy proxy = median( box_area_px * |signed gray deviation| ). Lower = less total signal for the detector.")
    print("Signed deviation is polarity-aware (dark targets counted by |trough - bg_median|).")

    probe = "wg2022_ir_020_split_03"
    for r in rows:
        if r["seq"] == probe:
            print("\n" + colorstr("bold", "🎯 VERIFICATION:"))
            print(f"• {probe} rank by median total energy: #{rows.index(r) + 1} / {len(rows)}")
            print(f"• median area={r['med_area']:.1f}px2 | median |signal|={r['med_signal']:.2f} | median energy={r['med_energy']:.1f}")
            if rows.index(r) == 0:
                print(colorstr("bold", colorstr("green", "✅ CONFIRMED: wg020_03 is the LOWEST total-energy sequence.")))
            else:
                print(colorstr("yellow", f"⚠️ wg020_03 is rank #{rows.index(r) + 1}, not the lowest; {rows[0]['seq']} is."))
            break
    print()


if __name__ == "__main__":
    main()
