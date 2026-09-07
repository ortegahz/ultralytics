#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Find outlier GT boxes in sequence wg2022_ir_020_split_03.
Prints raw label lines, coordinates, and images that have unusually large bbox.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.utils import check_det_dataset
from manu.stat_gt_bbox_sizes import find_label_file


def main():
    data_dict = check_det_dataset("/mnt/data/siping/datasets/manu/uav/data.yaml")
    val_source = data_dict["val"]

    val_dirs = [Path(val_source)] if isinstance(val_source, (str, Path)) else [Path(p) for p in val_source]
    img_paths = []
    for d in val_dirs:
        if d.is_file():
            with open(d, "r", encoding="utf-8") as f:
                for line in f:
                    p = Path(line.strip())
                    if p.exists() and "wg2022_ir_020_split_03" in p.name:
                        img_paths.append(p)
        elif d.is_dir():
            for p in d.rglob("*.*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") and "wg2022_ir_020_split_03" in p.name:
                    img_paths.append(p)

    img_paths.sort()
    print(f"Found {len(img_paths)} frames for wg2022_ir_020_split_03")

    large_boxes = []

    for p in img_paths:
        lbl_p = find_label_file(p)
        if not lbl_p or not lbl_p.exists():
            continue

        with open(lbl_p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = parts[0]
                    cx = float(parts[1])
                    cy = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])

                    bw_px = bw * 640.0
                    bh_px = bh * 512.0

                    if bw_px > 30.0 or bh_px > 30.0:
                        large_boxes.append({
                            "img": p.name,
                            "lbl_path": str(lbl_p),
                            "line_idx": line_idx,
                            "raw": line.strip(),
                            "bw_norm": bw,
                            "bh_norm": bh,
                            "bw_px": bw_px,
                            "bh_px": bh_px,
                        })

    print(f"\nTotal boxes with w > 30px or h > 30px: {len(large_boxes)}")
    print("=" * 80)
    for item in large_boxes:
        print(f"File: {item['img']} (line {item['line_idx']})")
        print(f"  Raw label : {item['raw']}")
        print(f"  Size px   : width={item['bw_px']:.1f}px, height={item['bh_px']:.1f}px")
        print(f"  Label file: {item['lbl_path']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
