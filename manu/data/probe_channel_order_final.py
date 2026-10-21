#!/usr/bin/env python
"""最终裁决：ULT 管线与 LQ 画法在**内容区**是否逐字节相同（即通道序是否真正一致）。

已定位的差异源
--------------
``manu/data/preannotate_hm_bbox.py::letterbox_gray`` 用 ``cv2.copyMakeBorder(..., value=114)``。
OpenCV 把标量 114 转成 ``Scalar(114, 0, 0, 0)`` ⇒ **只有 0 号通道被填 114**，
而 Ultralytics ``LetterBox`` 用 ``value=(114, 114, 114)`` ⇒ 三通道都填 114。
两者仅在**上下 padding 带**（640x640 画布里 128 行）不同，与通道序无关。

本脚本证明：剔除 padding 带后，两条路径的输入**逐字节相同** ⇒ 通道序一致，
混训两侧同构，通道序阻塞项正式解除。

用法::

    python manu/data/probe_channel_order_final.py --frames 4
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

from manu.data.preannotate_hm_bbox import letterbox_gray  # noqa: E402


def scalar_padding_probe() -> dict:
    """验证 OpenCV 标量 value=114 只填 0 号通道这一事实。"""
    src = np.zeros((4, 4, 3), np.uint8)
    out = cv2.copyMakeBorder(src, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=114)
    out_tuple = cv2.copyMakeBorder(src, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return {
        "scalar_value_114_channel_means": [round(float(out[:, :, c].mean()), 3) for c in range(3)],
        "tuple_value_114_114_114_channel_means": [
            round(float(out_tuple[:, :, c].mean()), 3) for c in range(3)
        ],
        "confirmed_scalar_only_fills_channel0": bool(out[:, :, 0].mean() == 114.0 and out[:, :, 1].mean() == 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="通道序最终裁决")
    parser.add_argument("--data", default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    print("=" * 100)
    scalar = scalar_padding_probe()
    print("  OpenCV 标量 padding 行为验证:")
    print(f"    value=114          → 通道均值 {scalar['scalar_value_114_channel_means']}")
    print(f"    value=(114,114,114)→ 通道均值 {scalar['tuple_value_114_114_114_channel_means']}")
    print(f"    标量只填 0 号通道: {'确认 ✅' if scalar['confirmed_scalar_only_fills_channel0'] else '不成立 ❌'}")
    print("=" * 100)

    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    ds = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    loader = build_dataloader(ds, batch=args.batch, workers=2, shuffle=False)
    lb = LetterBox(new_shape=(args.imgsz, args.imgsz), scaleup=False)

    content_max_diff = 0
    pad_max_diff = 0
    content_mismatch_px = 0
    checked = 0
    pad_rows = None

    for batch in loader:
        imgs = batch["img"].numpy()
        im_files = list(batch.get("im_file", []))
        for b in range(imgs.shape[0]):
            if checked >= args.frames:
                break
            path = im_files[b]
            feature = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if feature is None:
                continue
            ult_t = imgs[b]  # CHW uint8，Format 已做 BGR->RGB 反转
            lq_t = np.ascontiguousarray(
                letterbox_gray(feature, args.imgsz)[..., ::-1].transpose(2, 0, 1)
            )  # CHW uint8，模型序 [res, diff, I_t]
            if ult_t.shape != lq_t.shape:
                print(f"  [SKIP] {Path(path).name} 形状不同 {ult_t.shape} vs {lq_t.shape}")
                continue

            # 内容区 = 去掉上下 padding 带。LetterBox: r=min(640/512,640/640)=1.0 → dh=(640-512)/2=64
            # → top=64, bottom=64 ⇒ 内容区 y∈[64, 576)
            top = 64
            bottom = args.imgsz - 64
            pad_rows = (top, bottom)
            d_content = np.abs(ult_t[:, top:bottom].astype(np.int32) - lq_t[:, top:bottom].astype(np.int32))
            d_pad = np.abs(ult_t[:, :top].astype(np.int32) - lq_t[:, :top].astype(np.int32))
            content_max_diff = max(content_max_diff, int(d_content.max()))
            content_mismatch_px += int((d_content.sum(axis=0) > 0).sum())
            pad_max_diff = max(pad_max_diff, int(d_pad.max()))

            print(
                f"  {Path(path).name:<32} 内容区 max|d|={int(d_content.max()):3d} "
                f"(不同像素={int((d_content.sum(axis=0) > 0).sum())})   padding带 max|d|={int(d_pad.max()):3d}"
            )
            checked += 1
        if checked >= args.frames:
            break

    verdict = (
        "CHANNEL_ORDER_IDENTICAL_PAD_BAND_ONLY_DIFFERS"
        if content_max_diff == 0
        else "CONTENT_REGION_DIFFERS_CHANNEL_ORDER_OR_GEOMETRY"
    )
    report = {
        "data": args.data,
        "frames_checked": checked,
        "content_rows": list(pad_rows) if pad_rows else None,
        "content_region_max_abs_diff": content_max_diff,
        "content_region_mismatch_pixels": content_mismatch_px,
        "pad_band_max_abs_diff": pad_max_diff,
        "opencv_scalar_padding_probe": scalar,
        "verdict": verdict,
        "meaning": {
            "CHANNEL_ORDER_IDENTICAL_PAD_BAND_ONLY_DIFFERS": (
                "ULT(训练/Track1) 与 LQ(Track2/3) 在内容区逐字节相同 ⇒ 通道序完全一致，"
                "混训两侧同构；唯一差异是 padding 带的填充值（letterbox_gray 的 OpenCV 标量 value 陷阱），"
                "与通道序无关，且两轨基线/微调后用同一口径，不影响门禁可比性"
            ),
            "CONTENT_REGION_DIFFERS_CHANNEL_ORDER_OR_GEOMETRY": "内容区仍有差异，必须继续排查",
        }[verdict],
    }
    print("=" * 100)
    print(f"  内容区最大逐元素差 = {content_max_diff}（0 ⇒ 逐字节相同）")
    print(f"  padding 带最大逐元素差 = {pad_max_diff}")
    print(f"  判定: {verdict}")
    print(f"  含义: {report['meaning']}")
    print("=" * 100)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
