#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Inference & Visualization script for UAV Tiny Object Heatmap Detector.
Detects 3x3 point targets and saves both detected points and Heatmap overlays.

Usage:
    python manu/infer_heatmap.py --weights runs/heatmap_uav/exp/weights/best_recall.pt --source ultralytics/assets/bus.jpg --conf 0.20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from manu.heatmap_model import YOLO26HeatmapDetector
from manu.heatmap_evaluate import extract_peaks


def find_gt_boxes(img_path: Path) -> list[tuple[float, float, float, float]]:
    """Automatically find corresponding YOLO format label file and parse GT boxes."""
    # Common mappings: /images/ -> /labels/
    str_path = str(img_path)
    txt_candidates = []
    if "/images/" in str_path:
        txt_candidates.append(Path(str_path.replace("/images/", "/labels/")).with_suffix(".txt"))
    # Or in the same directory with .txt
    txt_candidates.append(img_path.with_suffix(".txt"))

    for txt_p in txt_candidates:
        if txt_p.exists():
            boxes = []
            try:
                with open(txt_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # class, cx, cy, w, h
                            boxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            except Exception:
                pass
            return boxes
    return []


def parse_args():
    parser = argparse.ArgumentParser(description="Infer with UAV Heatmap Detector")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--stride", type=int, default=4, help="Feature stride (4 for P2, 2 for P1)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (0.15~0.3 for high recall)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or cpu")
    parser.add_argument("--save-dir", type=str, default="runs/heatmap_infer", help="Output directory")
    return parser.parse_args()


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # Load weights
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    stride = ckpt.get("stride", args.stride)
    imgsz = ckpt.get("imgsz", args.imgsz)

    model = YOLO26HeatmapDetector(stride=stride, num_classes=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    source_path = Path(args.source)
    if source_path.is_dir():
        img_paths = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    else:
        img_paths = [source_path]

    print(f"Found {len(img_paths)} image(s) to process. Running on {device}...")

    for path in img_paths:
        orig_img = cv2.imread(str(path))
        if orig_img is None:
            continue
        h0, w0 = orig_img.shape[:2]

        # Preprocess
        img_resized, r, (dw, dh) = letterbox(orig_img, (imgsz, imgsz))
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_t)
            peaks = extract_peaks(
                heatmap=preds["heatmap"],
                offset=preds["offset"],
                stride=stride,
                conf_thresh=args.conf,
                top_k=100,
            )[0]

        pred_pts = peaks["points"]  # in (imgsz, imgsz) space
        scores = peaks["scores"]

        # Map back to original image
        vis_img = orig_img.copy()

        # 1. 绘制真实标签 GT (绿色方框 + 绿色实心中心点)
        gt_boxes = find_gt_boxes(path)
        if gt_boxes:
            print(f"[{path.name}] Found {len(gt_boxes)} GT target(s).")
            for cx, cy, w, h in gt_boxes:
                gx = int(cx * w0)
                gy = int(cy * h0)
                gw = int(max(w * w0, 4))  # 确保至少 4 像素宽以便肉眼可见
                gh = int(max(h * h0, 4))  # 确保至少 4 像素高以便肉眼可见
                x1, y1 = max(0, gx - gw // 2), max(0, gy - gh // 2)
                x2, y2 = min(w0 - 1, gx + gw // 2), min(h0 - 1, gy + gh // 2)

                # 画绿色真实框与中心
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.circle(vis_img, (gx, gy), 1, (0, 255, 0), -1)
                cv2.putText(vis_img, "GT", (x1 - 2, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 2. 绘制模型预测 Pred (红色圆圈 + 置信度标签)
        print(f"[{path.name}] Detected {len(pred_pts)} tiny targets (conf >= {args.conf:.2f}):")

        for (px, py), score in zip(pred_pts, scores):
            ox = (px - dw) / r
            oy = (py - dh) / r
            ox = int(np.clip(ox, 0, w0 - 1))
            oy = int(np.clip(oy, 0, h0 - 1))

            print(f"  -> Pred: ({ox}, {oy}) | Confidence: {score:.3f}")

            # Draw circle and center point in RED
            cv2.circle(vis_img, (ox, oy), 6, (0, 0, 255), 1)
            cv2.circle(vis_img, (ox, oy), 1, (0, 0, 255), -1)
            cv2.putText(vis_img, f"{score:.2f}", (ox + 8, oy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Save result
        out_file = save_dir / f"res_{path.name}"
        cv2.imwrite(str(out_file), vis_img)

        # Optionally save Heatmap overlay
        hm = preds["heatmap"][0, 0].cpu().numpy()
        hm_norm = np.uint8(255 * hm / (hm.max() + 1e-6))
        hm_color = cv2.applyColorMap(cv2.resize(hm_norm, (w0, h0)), cv2.COLORMAP_JET)
        hm_overlay = cv2.addWeighted(orig_img, 0.6, hm_color, 0.4, 0)
        cv2.imwrite(str(save_dir / f"hm_{path.name}"), hm_overlay)

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
