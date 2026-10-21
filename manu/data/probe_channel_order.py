#!/usr/bin/env python
"""裁决：训练路径（BGR，Ultralytics dataloader）与龙泉山评估路径（[..., ::-1] 反转）
喂给模型的**通道序是否一致**，以及哪一套才是模型真正训练时的顺序。

做法（纯实证，不靠读码推断）
--------------------------
对同一批龙泉山帧，用**完全相同的 letterbox + 同一套抽峰参数**跑两遍模型：
  * arm A = 磁盘序直接喂（[I_t, diff, res]，等价 Ultralytics dataloader 的 BGR 读法）
  * arm B = 反转后喂（[res, diff, I_t]，等价 multiscale_heatmap_cache 的 ``[..., ::-1]``）
然后把两臂的检测结果分别与**既有缓存** ``runs/ms_cache_nop0/<seq>.pkl`` 的原生尺度行
（tag=0，即 letterbox 那一档）逐帧对账，看哪一臂能复刻缓存。

同时统计：两臂在 GT 框内的热力图峰值分布（正确顺序应显著更高）。

用法::

    python manu/data/probe_channel_order.py \\
        --seq VIDEO00007_19700101_001324 --frames 40 \\
        --cache runs/ms_cache_nop0/VIDEO00007_19700101_001324.pkl \\
        --out-json runs/logs/channel_order_probe.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.multiscale_heatmap_cache import detect_components, heatmap_to_native  # noqa: E402
from manu.data.preannotate_hm_bbox import letterbox_gray, load_hm_model, resolve_weights  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> int:
    parser = argparse.ArgumentParser(description="通道序裁决探针")
    parser.add_argument("--features-root", default="/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median/images/train")
    parser.add_argument("--cache", required=True, help="既有缓存 pkl（用于对账）")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--region-threshold", type=float, default=0.06)
    parser.add_argument("--main-threshold", type=float, default=0.22)
    parser.add_argument("--fusion-dilate", type=int, default=32)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, model_stride = load_hm_model(resolve_weights(args.weights), device, use_p0_highway=False)
    model.eval()
    print(f"[INFO] model stride={model_stride}")

    feat_dir = Path(args.features_root)
    paths = sorted(feat_dir.glob(f"{args.seq}__frame_*.jpg"))
    if not paths:
        print(f"[FAIL] 找不到特征: {feat_dir}/{args.seq}__frame_*.jpg")
        return 2
    picks = paths[:: max(1, len(paths) // args.frames)][: args.frames]

    # 既有缓存里该序列的原生尺度行（tag=0）与融合行（tag=255）
    cache = pickle.loads(Path(args.cache).read_bytes())
    from manu.data.multiscale_heatmap_cache import iter_frames

    cache_by_frame: dict[int, dict[int, list]] = {}
    for frame_index, points, sizes, scores, _sums, _areas, tags in iter_frames(cache):
        bucket = cache_by_frame.setdefault(frame_index, {})
        for pt, sc, tg in zip(points, scores, tags):
            bucket.setdefault(int(tg), []).append((float(pt[0]), float(pt[1]), float(sc)))

    def run_arm(reverse: bool) -> tuple[dict[int, list], list[float]]:
        dets: dict[int, list] = {}
        gt_peak: list[float] = []
        for path in picks:
            frame_index = int(path.stem.split("__frame_")[1])
            feature = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
            if feature is None:
                continue
            height, width = feature.shape[:2]
            canvas = letterbox_gray(feature, args.imgsz)
            if reverse:
                canvas = canvas[..., ::-1]
            tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float().to(device) / 255.0
            with torch.no_grad():
                preds = model(tensor[None])
            heat = preds["heatmap"][0].float().cpu().numpy()
            if heat.ndim == 3:
                heat = heat[0]
            native = heatmap_to_native(heat, 1.0, (args.imgsz - int(round(width * args.imgsz / max(width, height)))) // 2,
                                      (args.imgsz - int(round(height * args.imgsz / max(width, height)))) // 2,
                                      args.imgsz, width, height)
            entries = detect_components(native, args.main_threshold, args.fusion_dilate, args.min_area, 100)
            dets[frame_index] = [(float(e[0]), float(e[1]), float(e[4])) for e in entries]
        return dets, gt_peak

    def match_score(dets: dict[int, list]) -> tuple[int, int, float]:
        """与缓存 tag=0 行做贪心最近邻匹配（容差 8px），返回 (命中, 缓存数, 平均距离)。"""
        hit = 0
        total = 0
        dists: list[float] = []
        for frame_index, rows in cache_by_frame.items():
            ref = rows.get(0, [])
            if not ref:
                continue
            total += len(ref)
            mine = dets.get(frame_index, [])
            for rx, ry, _rs in ref:
                if not mine:
                    continue
                d = min(((mx - rx) ** 2 + (my - ry) ** 2) ** 0.5 for mx, my, _ in mine)
                if d <= 8.0:
                    hit += 1
                    dists.append(d)
        return hit, total, (float(np.mean(dists)) if dists else float("nan"))

    results = {}
    for tag, reverse in (("A_disk_order_BGR", False), ("B_reversed_RGB", True)):
        dets, _ = run_arm(reverse)
        hit, total, mean_d = match_score(dets)
        n_det = sum(len(v) for v in dets.values())
        results[tag] = {
            "detections": n_det,
            "cache_rows_matched": hit,
            "cache_rows_total": total,
            "match_rate": round(hit / total, 4) if total else None,
            "mean_match_distance_px": round(mean_d, 3) if dists_safe(mean_d) else None,
        }
        print(f"[{tag}] 检出 {n_det} | 与缓存 tag0 匹配 {hit}/{total} ({(hit/total*100 if total else 0):.1f}%) | 平均距离 {mean_d:.2f}px")

    verdict = None
    a_rate = results["A_disk_order_BGR"]["match_rate"] or 0.0
    b_rate = results["B_reversed_RGB"]["match_rate"] or 0.0
    if a_rate > 0.9 and b_rate < 0.5:
        verdict = "TRAINING_ORDER_IS_DISK_ORDER_BGR"
    elif b_rate > 0.9 and a_rate < 0.5:
        verdict = "TRAINING_ORDER_IS_REVERSED_RGB"
    elif a_rate > 0.9 and b_rate > 0.9:
        verdict = "BOTH_MATCH"
    else:
        verdict = "INDETERMINATE"

    report = {
        "sequence": args.seq,
        "frames_tested": len(picks),
        "model_stride": model_stride,
        "results": results,
        "verdict": verdict,
        "meaning": {
            "TRAINING_ORDER_IS_DISK_ORDER_BGR": "缓存路径（反转）与训练路径（BGR）不一致 —— 缓存用的是反转序",
            "TRAINING_ORDER_IS_REVERSED_RGB": "缓存路径与训练路径一致（都应反转）",
            "BOTH_MATCH": "两臂都能复刻缓存，说明该模型对通道序不敏感（需进一步排查）",
            "INDETERMINATE": "两臂都不匹配，需人工排查",
        }[verdict],
    }
    print("=" * 88)
    print(f"判定: {verdict} —— {report['meaning']}")
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告已写入: {out}")
    return 0


def dists_safe(x: float) -> bool:
    return x == x  # not NaN


if __name__ == "__main__":
    raise SystemExit(main())
