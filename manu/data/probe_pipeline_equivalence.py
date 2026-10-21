#!/usr/bin/env python
"""实证裁决：三条推理/训练路径喂给模型的**通道序与画布几何**是否真的同一构。

背景
----
同一个模型不可能同时适配两种通道序。三条路径各自独立地读盘：

* **ULT**（Track1 Anti-UAV + 训练）：Ultralytics ``build_yolo_dataset(mode=...)``
  → ``LetterBox(640x640)`` → ``Format._format_img`` 里 ``img[::-1]``（BGR→RGB）
* **LQ**（Track2/3 龙泉山）：``manu/data/multiscale_heatmap_cache.py`` 里
  ``letterbox_gray(feature, 640)[..., ::-1].transpose(2, 0, 1)``
* **DISK**（对照组）：磁盘原序，**不反转**

只读码不能作为结论（铁律）。本脚本用同一批帧、同一套权重、同一 canvas 尺寸
把三条臂各跑一遍，直接比对**热力图张量**：

* ULT 与 LQ 的最大逐元素差应当 ≈ 0（等价路径）
* DISK 与它们的差应当显著非零（对照组必须能区分，否则测试无分辨力）

另外统计每臂热力图的 max/mean/对比度，用作「模型是否真的偏好某一序」的旁证。

用法::

    python manu/data/probe_pipeline_equivalence.py \\
        --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \\
        --frames 32 --out-json runs/logs/pipeline_equivalence_uav.json
"""

from __future__ import annotations

import argparse
import json
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
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils import DEFAULT_CFG  # noqa: E402

from manu.data.preannotate_hm_bbox import letterbox_gray, load_hm_model, resolve_weights  # noqa: E402


def heat_stats(heat: np.ndarray) -> dict:
    """热力图响应的旁证统计：模型对「正确通道序」应当给出更高更尖的响应。"""
    return {
        "max": round(float(heat.max()), 6),
        "mean": round(float(heat.mean()), 6),
        "p99": round(float(np.percentile(heat, 99)), 6),
        "frac_over_0.22": round(float((heat > 0.22).mean()), 8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="三条推理路径的通道序/几何等价性实证")
    parser.add_argument("--data", required=True, help="data.yaml（Ultralytics 格式）")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--device-heatmap-stride", type=int, default=2)
    parser.add_argument("--ablate-p0", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, model_stride = load_hm_model(
        resolve_weights(args.weights), device, stride_fallback=2, use_p0_highway=not args.ablate_p0
    )
    model.eval()
    print(f"[INFO] model stride={model_stride} | data={args.data}")

    # ---- ULT 臂：完全照搬 cache_trial0474_inferences.py 的建集方式 ----
    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data
    val_dataset = build_yolo_dataset(
        cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=args.stride
    )
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)

    diffs_ult_lq: list[float] = []
    diffs_disk_ult: list[float] = []
    stats: dict[str, list[dict]] = {"ULT": [], "LQ": [], "DISK": []}
    checked = 0
    mismatched_shape: list[str] = []

    with torch.no_grad():
        for batch in val_loader:
            ult = batch["img"].to(device).float() / 255.0
            im_files = list(batch.get("im_file", []))
            for b in range(ult.shape[0]):
                if checked >= args.frames:
                    break
                path = im_files[b]
                feature = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if feature is None:
                    continue
                # LQ 臂：multiscale_heatmap_cache 的画法
                lq = letterbox_gray(feature, args.imgsz)[..., ::-1].transpose(2, 0, 1)
                # DISK 臂：磁盘原序，不反转（对照组）
                disk = letterbox_gray(feature, args.imgsz).transpose(2, 0, 1)

                t_ult = ult[b]
                if tuple(t_ult.shape) != tuple(lq.shape):
                    mismatched_shape.append(
                        f"{Path(path).name}: ULT{tuple(t_ult.shape)} vs LQ{tuple(lq.shape)}"
                    )
                    continue
                t_lq = torch.from_numpy(np.ascontiguousarray(lq)).float().to(device) / 255.0
                t_disk = torch.from_numpy(np.ascontiguousarray(disk)).float().to(device) / 255.0

                h_ult = model(t_ult[None])["heatmap"][0].float().cpu().numpy()
                h_lq = model(t_lq[None])["heatmap"][0].float().cpu().numpy()
                h_disk = model(t_disk[None])["heatmap"][0].float().cpu().numpy()
                if h_ult.ndim == 3:
                    h_ult, h_lq, h_disk = h_ult[0], h_lq[0], h_disk[0]

                diffs_ult_lq.append(float(np.abs(h_ult - h_lq).max()))
                diffs_disk_ult.append(float(np.abs(h_disk - h_ult).max()))
                stats["ULT"].append(heat_stats(h_ult))
                stats["LQ"].append(heat_stats(h_lq))
                stats["DISK"].append(heat_stats(h_disk))
                checked += 1
            if checked >= args.frames:
                break

    def agg(rows: list[dict], key: str) -> float:
        return round(float(np.mean([r[key] for r in rows])), 6) if rows else float("nan")

    d_ult_lq = float(np.max(diffs_ult_lq)) if diffs_ult_lq else float("nan")
    d_disk_ult = float(np.max(diffs_disk_ult)) if diffs_disk_ult else float("nan")
    mean_ult_lq = float(np.mean(diffs_ult_lq)) if diffs_ult_lq else float("nan")
    mean_disk_ult = float(np.mean(diffs_disk_ult)) if diffs_disk_ult else float("nan")

    # 判据：ULT 与 LQ 等价（<1e-3），且 DISK 对照组确实不同（>1e-3）——否则测试无分辨力
    equivalent = bool(d_ult_lq < 1e-3) if diffs_ult_lq else False
    discriminating = bool(d_disk_ult > 1e-3) if diffs_disk_ult else False
    if equivalent and discriminating:
        verdict = "ULT_EQ_LQ_AND_DISK_DIFFERS"
    elif equivalent and not discriminating:
        verdict = "ULT_EQ_LQ_BUT_DISK_IDENTICAL_NO_DISCRIMINATION"
    elif not equivalent:
        verdict = "ULT_NE_LQ_MISMATCH"
    else:
        verdict = "INDETERMINATE"

    report = {
        "data": args.data,
        "frames_checked": checked,
        "model_stride": model_stride,
        "weights": args.weights,
        "ablate_p0": bool(args.ablate_p0),
        "shape_mismatches": mismatched_shape[:5],
        "ULT_vs_LQ": {
            "max_abs_diff": round(d_ult_lq, 8),
            "mean_abs_diff": round(mean_ult_lq, 8),
            "note": "ULT=Ultralytics build_yolo_dataset 管线; LQ=multiscale_heatmap_cache 的 [..., ::-1] 画法",
        },
        "DISK_vs_ULT": {
            "max_abs_diff": round(d_disk_ult, 8),
            "mean_abs_diff": round(mean_disk_ult, 8),
            "note": "DISK=磁盘原序不反转（对照组）",
        },
        "heat_stats": {
            arm: {
                "mean_max": agg(rows, "max"),
                "mean_mean": agg(rows, "mean"),
                "mean_p99": agg(rows, "p99"),
                "mean_frac_over_0.22": agg(rows, "frac_over_0.22"),
            }
            for arm, rows in stats.items()
        },
        "verdict": verdict,
        "meaning": {
            "ULT_EQ_LQ_AND_DISK_DIFFERS": "训练/Track1 管线与 Track2/3 管线喂给模型的通道序完全一致，且磁盘原序确实不同 —— 混训两侧同构，通道序阻塞解除",
            "ULT_EQ_LQ_BUT_DISK_IDENTICAL_NO_DISCRIMINATION": "两条管线一致，但磁盘原序也一致 —— 模型对通道序不敏感，本次实验无分辨力，需换更强判据",
            "ULT_NE_LQ_MISMATCH": "两条管线不一致 —— 必须先查清差异再开训",
            "INDETERMINATE": "数据不足，无法判定",
        }[verdict],
    }

    print("=" * 96)
    print(f"  帧数={checked}  ULT_vs_LQ max|d|={d_ult_lq:.3e}  DISK_vs_ULT max|d|={d_disk_ult:.3e}")
    for arm in ("ULT", "LQ", "DISK"):
        s = report["heat_stats"][arm]
        print(
            f"  {arm:<5} 热力图 mean_max={s['mean_max']:.4f} mean_p99={s['mean_p99']:.4f} "
            f"frac>0.22={s['mean_frac_over_0.22']:.3e}"
        )
    print(f"  判定: {verdict}")
    print(f"  含义: {report['meaning']}")
    print("=" * 96)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
