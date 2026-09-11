#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Export YOLO26HeatmapDetector to ONNX format for Netron visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.heatmap_model import YOLO26HeatmapDetector


class ONNXHeatmapWrapper(nn.Module):
    """Wrapper to output tuple (heatmap, offset) instead of dict for ONNX export."""

    def __init__(self, model: YOLO26HeatmapDetector):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out["heatmap"], out["offset"]


def export_onnx(
    weights: str | Path,
    output: str | Path,
    stride: int = 2,
    scale: str = "s",
    imgsz: int = 640,
    opset: int = 12,
):
    weights_path = Path(weights)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Building YOLO26HeatmapDetector (stride={stride}, scale='{scale}') ...")
    model = YOLO26HeatmapDetector(stride=stride, scale=scale, weights=weights_path if weights_path.exists() else None)
    model.eval()

    wrapper = ONNXHeatmapWrapper(model)
    dummy_input = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32)

    print(f"[EXPORT] Exporting to ONNX: {output_path} (imgsz={imgsz}, opset={opset}) ...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["heatmap", "offset"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "heatmap": {0: "batch_size"},
            "offset": {0: "batch_size"},
        },
    )
    print(f"[SUCCESS] ONNX model successfully saved to: {output_path.resolve()}")
    print(f"  Input shape: (B, 3, {imgsz}, {imgsz})")
    print(f"  Heatmap output shape: (B, 1, {imgsz // stride}, {imgsz // stride})")
    print(f"  Offset output shape:  (B, 2, {imgsz // stride}, {imgsz // stride})")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26HeatmapDetector to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/manu/mnt/pycharm_project_10ae9e2e/runs/scale_ab_test/yolo26s_p2_5ep/weights/best_f1.pt",
        help="Path to checkpoint .pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/onnx/yolo26s_p1_heatmap.onnx",
        help="Output ONNX path",
    )
    parser.add_argument("--stride", type=int, default=2, help="Feature stride (2 for P1, 4 for P2)")
    parser.add_argument("--scale", type=str, default="s", choices=["n", "s"], help="Model scale ('n' or 's')")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    export_onnx(
        weights=args.weights,
        output=args.output,
        stride=args.stride,
        scale=args.scale,
        imgsz=args.imgsz,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
