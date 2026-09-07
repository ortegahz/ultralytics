#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Targeted Diagnostic Cropper:
Exports badcase crops for the TOP missed video sequences, organized by sequence subdirectories.
Reads images directly from disk based on records in fn_missed_analysis.csv, taking only seconds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.utils import check_det_dataset
from manu.diagnose_heatmap_badcases import make_diagnostic_crop


def parse_args():
    parser = argparse.ArgumentParser(description="Export FN crops for Top-N worst sequences")
    parser.add_argument(
        "--csv-file",
        type=str,
        default="runs/badcase_analysis/fn_missed_analysis.csv",
        help="Path to fn_missed_analysis.csv",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/badcase_analysis/top_missed_sequences",
        help="Output directory to save sequence crops",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--crop-size", type=int, default=64, help="Crop window size")
    parser.add_argument("--max-per-seq", type=int, default=100, help="Max crops to save per sequence")
    parser.add_argument("--top-k-seqs", type=int, default=5, help="Number of worst sequences to dump")
    return parser.parse_args()


def build_image_lookup(val_img_dir: Path | list[Path]) -> dict[str, Path]:
    """Build a fast lookup dict from image filename to its absolute Path."""
    print("Indexing validation images...")
    lookup = {}
    if isinstance(val_img_dir, (str, Path)):
        val_dirs = [Path(val_img_dir)]
    else:
        val_dirs = [Path(p) for p in val_img_dir]

    for d in val_dirs:
        if d.is_file():
            # Sometimes data.yaml points to a txt file listing image paths
            with open(d, "r", encoding="utf-8") as f:
                for line in f:
                    p = Path(line.strip())
                    if p.exists():
                        lookup[p.name] = p
        elif d.is_dir():
            for p in d.rglob("*.*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    lookup[p.name] = p
    print(f"Indexed {len(lookup)} validation images.")
    return lookup


def main():
    args = parse_args()
    csv_path = Path(args.csv_file)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV report not found: {csv_path}")

    # 1. 读取 CSV 中所有真正的 Zero-Response 漏检
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    zero_rows = [r for r in rows if "True Zero-Response" in r.get("type", "")]
    print(f"Total True Zero-Response FN records: {len(zero_rows)}")

    # 2. 按视频序列聚合统计
    seq_records: dict[str, list[dict]] = {}
    for r in zero_rows:
        im_name = r["image"]
        seq = im_name.split("___")[0] if "___" in im_name else im_name[:20]
        seq_records.setdefault(seq, []).append(r)

    # 排序选出前 Top-K 个最严重的视频序列
    sorted_seqs = sorted(seq_records.items(), key=lambda x: len(x[1]), reverse=True)
    top_seqs = sorted_seqs[: args.top_k_seqs]

    print("\n" + "=" * 70)
    print(f"TOP {args.top_k_seqs} WORST SEQUENCES (ZERO-RESPONSE LEAKAGE):")
    print("=" * 70)
    for rank, (seq_name, recs) in enumerate(top_seqs, 1):
        ratio = (len(recs) / len(zero_rows)) * 100
        print(f"Rank {rank}: {seq_name:<30} -> {len(recs):<5} missed ({ratio:.1f}%)")
    print("=" * 70 + "\n")

    # 3. 构建原图绝对路径索引
    data_dict = check_det_dataset(args.data)
    val_source = data_dict["val"]
    img_lookup = build_image_lookup(val_source)

    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = PROJECT_ROOT / out_base
    out_base.mkdir(parents=True, exist_ok=True)

    # 4. 逐个视频序列抽取并导出切片
    for rank, (seq_name, recs) in enumerate(top_seqs, 1):
        seq_dir_name = f"rank{rank:02d}_{seq_name}_{len(recs)}missed"
        seq_save_dir = out_base / seq_dir_name
        seq_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting up to {args.max_per_seq} crops for: {seq_name}...")

        # 均匀抽样或按顺序提取最多 max_per_seq 张
        total_missed = len(recs)
        if total_missed > args.max_per_seq:
            # 均匀抽样覆盖整个视频时间线
            indices = np.linspace(0, total_missed - 1, args.max_per_seq, dtype=int)
            sample_recs = [recs[i] for i in indices]
        else:
            sample_recs = recs

        saved_count = 0
        for r in sample_recs:
            im_name = r["image"]
            img_path = img_lookup.get(im_name)

            if img_path is None or not img_path.exists():
                # 尝试加上常见后缀匹配
                for ext in [".jpg", ".png", ".jpeg"]:
                    alt = img_lookup.get(im_name + ext)
                    if alt and alt.exists():
                        img_path = alt
                        break

            if img_path is None or not img_path.exists():
                continue

            # 读取三通道预处理图 (Ch0=Gray, Ch1=Diff1, Ch2=Diff2)
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            if img_bgr.shape[0] != args.imgsz or img_bgr.shape[1] != args.imgsz:
                img_bgr = cv2.resize(img_bgr, (args.imgsz, args.imgsz))

            gx = float(r["gt_x"])
            gy = float(r["gt_y"])
            dist = float(r["min_dist_to_peak"])
            hm = float(r["heatmap_val_at_gt"])

            # 制造空 Heatmap，真实呈现零响应状态
            feat_size = args.imgsz // 2
            dummy_hm = np.zeros((feat_size, feat_size), dtype=np.float32)

            tag = f"Rank{rank} | {seq_name} | dist={dist:.1f}px, hm={hm:.2f}"
            crop_vis = make_diagnostic_crop(
                img_hwc=img_bgr,
                heatmap_hw=dummy_hm,
                center_xy=(gx, gy),
                crop_size=args.crop_size,
                gt_xy=(gx, gy),
                pred_xy=None,
                tag=tag,
            )

            saved_count += 1
            save_name = f"crop_{saved_count:03d}_{Path(im_name).stem}.jpg"
            cv2.imwrite(str(seq_save_dir / save_name), crop_vis)

        print(f"  -> Saved {saved_count} diagnostic crops to {seq_save_dir.name}/")

    print("\n" + "=" * 70)
    print(f"ALL DONE! Saved Top sequences to: {out_base}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
