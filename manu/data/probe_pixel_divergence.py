#!/usr/bin/env python
"""定点诊断：ULT 管线与 LQ 画法的**输入像素**到底差在哪一层。

分层比对，逐层收敛：
  L0 解码层   cv2.imread  vs  ultralytics.utils.patches.imread
  L1 画布层   letterbox_gray(...) vs Ultralytics LetterBox 的输出（未反转）
  L2 通道序   L1 的 [::-1] 是否与 Format._format_img 的 img[::-1] 等价

用法::

    python manu/data/probe_pixel_divergence.py --frames 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils import DEFAULT_CFG  # noqa: E402
from ultralytics.utils.patches import imread as ultra_imread  # noqa: E402

from manu.data.preannotate_hm_bbox import letterbox_gray  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="像素差异分层诊断")
    parser.add_argument("--data", default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    ds = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    loader = build_dataloader(ds, batch=args.batch, workers=2, shuffle=False)

    lb = LetterBox(new_shape=(args.imgsz, args.imgsz), scaleup=False)
    print("=" * 100)
    done = 0
    for batch in loader:
        imgs = batch["img"].numpy()  # (B,3,H,W) uint8
        im_files = list(batch.get("im_file", []))
        for b in range(imgs.shape[0]):
            if done >= args.frames:
                break
            path = im_files[b]
            # L0: 解码层
            a_cv = cv2.imread(str(path), cv2.IMREAD_COLOR)
            b_ul = ultra_imread(str(path), flags=cv2.IMREAD_COLOR)
            d0 = float(np.abs(a_cv.astype(np.int32) - b_ul.astype(np.int32)).max())
            # L1: 画布层（未反转）
            lq_canvas = letterbox_gray(a_cv, args.imgsz)          # HWC uint8
            ult_canvas = lb(image=b_ul)                            # HWC uint8 (Ultralytics LetterBox)
            d1 = float(np.abs(lq_canvas.astype(np.int32) - ult_canvas.astype(np.int32)).max())
            n_diff_px = int((np.abs(lq_canvas.astype(np.int32) - ult_canvas.astype(np.int32)).sum(axis=2) > 0).sum())
            # 形状与通道均值
            ult_t = imgs[b]  # CHW uint8
            # L2: 通道序等价性 —— 把两边都摆成 [res,diff,I_t] 再比
            lq_chw = np.ascontiguousarray(lq_canvas[..., ::-1].transpose(2, 0, 1))
            ult_chw = np.ascontiguousarray(ult_t[:, ::-1])  # Format 已把 CHW 反转过，再翻回来对齐磁盘序
            d2 = float(np.abs(lq_chw.astype(np.int32) - ult_chw.astype(np.int32)).max())

            print(f"--- {Path(path).name} ---")
            print(f"  L0 解码: cv2.imread vs patches.imread  max|d|={d0:.1f}   shape cv2={a_cv.shape} ultra={b_ul.shape}")
            print(
                f"  L1 画布: letterbox_gray vs LetterBox     max|d|={d1:.1f}  不同像素数={n_diff_px}"
                f"/{lq_canvas.shape[0]*lq_canvas.shape[1]}"
            )
            print(f"     LQ  canvas shape={lq_canvas.shape}  ULT canvas shape={ult_canvas.shape}")
            print(f"     LQ  通道均值={lq_canvas.reshape(-1,3).mean(axis=0).round(3)}")
            print(f"     ULT 通道均值={ult_canvas.reshape(-1,3).mean(axis=0).round(3)}")
            print(f"  L2 通道序: 对齐磁盘序后 max|d|={d2:.1f}")
            # 逐行/逐列差异定位
            diff = np.abs(lq_canvas.astype(np.int32) - ult_canvas.astype(np.int32)).sum(axis=2)
            if n_diff_px:
                ys, xs = np.nonzero(diff)
                print(
                    f"     差异像素 y∈[{ys.min()},{ys.max()}] x∈[{xs.min()},{xs.max()}]"
                    f"  最大单像素差={int(diff.max())}"
                )
            done += 1
        if done >= args.frames:
            break
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
