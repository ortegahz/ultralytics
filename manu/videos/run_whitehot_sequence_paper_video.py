#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run white-hot inversion, GMC+Median inference, and paper-style video rendering."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import cv2
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_fpv_ir_gmc_median import IMAGE_SUFFIXES, natural_key, process_single_ir_sequence
from manu.evaluation.eval_fpv_ir_zero_shot import load_sota_model, run_caching
from manu.videos.generate_collaboration_diagnostic_videos import generate_sequence_video


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a white-hot Trial 0474 diagnostic video for one sequence")
    parser.add_argument(
        "--source-seq",
        default="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg/VIDEO00032_19700101_014453",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/data/siping/datasets/manu/longquanshan_whitehot_VIDEO00032_19700101_014453",
    )
    parser.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--stride-step", type=int, default=2)
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--right-panel", choices=["heatmap", "features"], default="heatmap")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def invert_sequence(source_seq: Path, inverted_seq: Path, overwrite: bool):
    source_images = sorted(
        (path for path in source_seq.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    if not source_images:
        raise FileNotFoundError(f"No images found in {source_seq}")
    inverted_images = inverted_seq / "ir" / "images"
    if inverted_seq.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {inverted_seq}; use --overwrite")
        shutil.rmtree(inverted_seq)
    inverted_images.mkdir(parents=True, exist_ok=True)

    for source_path in tqdm(source_images, desc="Inverting black-hot frames"):
        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read {source_path}")
        destination = inverted_images / source_path.name
        if not cv2.imwrite(str(destination), 255 - image):
            raise RuntimeError(f"Failed to write {destination}")
    return len(source_images)


def write_data_yaml(dataset_root: Path):
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "data.yaml").write_text(
        f"path: {dataset_root.resolve()}\ntrain: images/train\nval: images/train\ntest: images/train\nnames:\n  0: uav\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    source_seq = Path(args.source_seq).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    inverted_root = output_root / "inverted" / source_seq.name
    feature_root = output_root / "features"
    feature_images = feature_root / "images" / "train"
    feature_labels = feature_root / "labels" / "train"
    cache_path = output_root / "trial0474_whitehot_cache.pkl"
    video_root = output_root / "paper_diagnostic"

    frame_count = invert_sequence(source_seq, inverted_root, args.overwrite)
    feature_images.mkdir(parents=True, exist_ok=True)
    feature_labels.mkdir(parents=True, exist_ok=True)
    result = process_single_ir_sequence(
        seq_name=source_seq.name,
        seq_dir_str=str(inverted_root),
        out_img_dir_str=str(feature_images),
        out_lbl_dir_str=str(feature_labels),
        window=args.window,
        stride_step=args.stride_step,
        downscale=args.downscale,
    )
    if result["success"] != frame_count:
        raise RuntimeError(f"Feature generation mismatch: expected {frame_count}, got {result['success']}")
    write_data_yaml(feature_root)

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    model, stride = load_sota_model(Path(args.weights), device)
    cache_args = SimpleNamespace(imgsz=640, batch=args.batch, conf_thresh=0.02, top_k=100)
    records = run_caching(model, stride, feature_root / "data.yaml", cache_path, cache_args, device)
    cache_by_stem = {Path(record["im_name"]).stem: record for record in records}
    generate_sequence_video(
        seq_name=source_seq.name,
        data_root=feature_root,
        out_dir=video_root,
        cache_by_stem=cache_by_stem,
        right_panel_mode=args.right_panel,
        show_dets=True,
        imgsz=640,
        fps=args.fps,
        conf_thresh=args.conf,
        split="train",
    )
    print(f"[DONE] White-hot video output: {video_root}")


if __name__ == "__main__":
    main()
