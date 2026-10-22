#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 `multiscale_heatmap_cache.py` 的**现场** GMC+median 特征是否与官方落盘特征一致。

为什么必须验
------------
stride=1 诊断有两种取特征的路：
  A. `--frames-root`：缓存脚本在 GPU 前向前**现场重算** GMC+median（省磁盘，但语义可能与官方不同）；
  B. 另建 stride=1 的特征集（与官方逐字节同源，但要新跑一次全量特征构建）。

官方 `build_full_median_dataset.py` 的通道定义是
  ch0 = I_t，ch1 = |I_t − W(I_{t-2})|（2-lag GMC 对齐差分），ch2 = median residual
缓存脚本 `compute_features` 声称复现同一口径。本脚本**不靠声称，靠实测**：
用同一段实现现场重算，逐帧与官方落盘特征比对（官方侧是 JPEG，有损，故看逐通道
max|Δ| / 平均 |Δ| 落在什么量级：几单位 = JPEG 噪声，说明 GMC 完全一致；
几十上百 = GMC 语义不同，路 A 作废）。

用法（执行机：服务器，根目录 `/tmp/pycharm_project_10ae9e2e`）：

```bash
python manu/data/verify_onthefly_features.py \
  --raw-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
  --features-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train \
  --sequence VIDEO00002_19700101_001415 --n-frames 200 --stride 5
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manu.data.multiscale_heatmap_cache import compute_features  # noqa: E402
from manu.data.build_full_median_dataset import FastGMCEstimator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="现场 GMC 特征 vs 官方落盘特征 逐帧比对")
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=5, help="官方落盘特征的采样间隔")
    ap.add_argument("--median-window", type=int, default=21)
    ap.add_argument("--temporal-stride", type=int, default=2)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=50, help="前 N 帧为 GMC 历史预热，不参与判定")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    raw_dir = Path(args.raw_root) / args.sequence
    if not raw_dir.is_dir():
        raise SystemExit(f"[FATAL] 原始帧目录不存在：{raw_dir}")

    raw_paths = sorted(raw_dir.glob("frame_*.jpg"))
    if not raw_paths:
        raise SystemExit(f"[FATAL] {raw_dir} 下没有 frame_*.jpg")
    n = min(args.n_frames, len(raw_paths))
    print(f"[INFO] 原始帧 {len(raw_paths)} 张，本次取前 {n} 张", flush=True)

    grays = []
    for path in raw_paths[:n]:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"[FATAL] 读帧失败：{path}")
        grays.append(img)

    ns = argparse.Namespace(median_window=args.median_window, temporal_stride=args.temporal_stride)
    gmc = FastGMCEstimator(downscale=args.downscale)
    feats, _ = compute_features(grays, [], ns, gmc)

    feat_dir = Path(args.features_root)
    rows = []
    for pos in range(args.warmup, n, args.stride):
        disk = feat_dir / f"{args.sequence}__frame_{pos:06d}.jpg"
        if not disk.is_file():
            continue
        ref = cv2.imread(str(disk), cv2.IMREAD_COLOR)
        if ref is None:
            continue
        cur = feats[pos]
        if ref.shape != cur.shape:
            rows.append({"position": pos, "error": f"shape {ref.shape} vs {cur.shape}"})
            continue
        diff = np.abs(ref.astype(np.int16) - cur.astype(np.int16))
        rows.append(
            {
                "position": pos,
                "max_abs_diff_per_channel": [int(diff[:, :, c].max()) for c in range(3)],
                "mean_abs_diff_per_channel": [round(float(diff[:, :, c].mean()), 4) for c in range(3)],
                "pixels_over_8_per_channel": [int((diff[:, :, c] > 8).sum()) for c in range(3)],
            }
        )

    if not rows:
        raise SystemExit("[FATAL] 没有可比对的帧（官方落盘特征与原始帧位置对不上？）")

    max_by_ch = [max(r["max_abs_diff_per_channel"][c] for r in rows if "max_abs_diff_per_channel" in r) for c in range(3)]
    mean_by_ch = [
        round(float(np.mean([r["mean_abs_diff_per_channel"][c] for r in rows if "mean_abs_diff_per_channel" in r])), 4)
        for c in range(3)
    ]
    over8 = [sum(r["pixels_over_8_per_channel"][c] for r in rows if "pixels_over_8_per_channel" in r) for c in range(3)]
    total_px = sum(int(np.prod(grays[0].shape)) for _ in rows)

    verdict = "IDENTICAL_UP_TO_JPEG" if max(max_by_ch) <= 16 else "SEMANTICS_DIFFER"
    print(f"\n[比对帧数] {len(rows)}（预热跳过前 {args.warmup} 帧）")
    print(f"[逐通道 max|Δ|] ch0(I_t)={max_by_ch[0]} ch1(GMC diff)={max_by_ch[1]} ch2(median res)={max_by_ch[2]}")
    print(f"[逐通道 平均|Δ|] ch0={mean_by_ch[0]} ch1={mean_by_ch[1]} ch2={mean_by_ch[2]}")
    print(f"[|Δ|>8 的像素数] ch0={over8[0]} ch1={over8[1]} ch2={over8[2]} / 总 {total_px}")
    print(f"[判定] {verdict}")
    print("  说明：max|Δ| ≤ 16 且 >8 像素占比极低 ⇒ 现场 GMC 与官方落盘一致（JPEG 噪声量级）；"
          "若 ch1/ch2 出现几十上百的 max|Δ| ⇒ 语义不同，stride=1 必须另建特征集。")

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sequence": args.sequence,
                    "n_frames": n,
                    "stride": args.stride,
                    "warmup": args.warmup,
                    "compared_frames": len(rows),
                    "max_abs_diff_per_channel": max_by_ch,
                    "mean_abs_diff_per_channel": mean_by_ch,
                    "pixels_over_8_per_channel": over8,
                    "verdict": verdict,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[OK] -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
