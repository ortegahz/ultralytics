#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multi-scale Trial 0474 pre-annotation worker, executed on the training server.

This runs *inside* the torch-capable ``uav`` environment; the web server spawns it over SSH so the model
is only ever loaded when the annotator explicitly asks for it. It writes YOLO proposals into the
``prelabels/<seq_id>/`` tree and never touches ``labels/``, so human edits can never be clobbered by a rerun.

The inference contract is reproduced from ``manu/data/multiscale_heatmap_cache.py`` rather than
re-derived: disk-channel order ``[I_t, GMC diff, median residual]`` for feature construction, reversed to
the trained model order ``[median residual, GMC diff, I_t]`` on the way in, then per-pixel ``np.maximum``
across scales, threshold, dilate and connected-component reduction on the native pixel grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.annotate.labels import format_yolo  # noqa: E402
from manu.data.build_full_median_dataset import FastGMCEstimator  # noqa: E402
from manu.data.preannotate_hm_bbox import letterbox_gray, load_hm_model, resolve_weights  # noqa: E402
from manu.diagnostics.probe_heatmap_regions_video import downsample_pad  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-scale Trial 0474 pre-annotation for the web annotator")
    parser.add_argument("--source", required=True, help="Video file or frame directory, on the server filesystem")
    parser.add_argument("--seq-id", required=True)
    parser.add_argument("--prelabel-dir", required=True, help="Directory that will hold <seq_id>/<frame>.txt")
    parser.add_argument("--status-file", default="", help="Progress JSON refreshed during the run")
    parser.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--scales", default="160,80", help="Downsample targets besides the native letterbox input")
    parser.add_argument("--main-threshold", type=float, default=0.22)
    parser.add_argument("--fusion-dilate", type=int, default=9)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole sequence; used by smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="Recompute frames that already have proposals")
    args = parser.parse_args()
    args.scales = sorted({int(item) for item in args.scales.split(",") if item.strip() and int(item) > 0}, reverse=True)
    return args


def write_status(path: str, payload: dict) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def iter_gray_frames(source: Path, limit: int):
    """Yield grayscale frames from a video file or a frame directory, in order."""
    if source.is_dir():
        paths = sorted(
            (item for item in source.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda item: item.name,
        )
        if not paths:
            raise FileNotFoundError(f"no frames under {source}")
        for path in paths[:limit] if limit else paths:
            frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise RuntimeError(f"unreadable frame: {path}")
            yield frame
        return

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    try:
        seen = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            seen += 1
            if limit and seen >= limit:
                break
    finally:
        capture.release()


def build_features(chunk_grays: list[np.ndarray], history: list[np.ndarray], args, gmc: FastGMCEstimator):
    """Reproduce the raw pipeline in disk channel order [I_t, GMC diff, median residual]."""
    features, rolling = [], list(history)
    for current in chunk_grays:
        lag = rolling[max(0, len(rolling) - args.temporal_stride)] if rolling else current
        motion = cv2.absdiff(current, gmc.warp(lag, gmc.compute_affine(lag, current)))
        aligned = []
        for step in range(1, args.median_window + 1):
            history_frame = rolling[max(0, len(rolling) - step * args.temporal_stride)] if rolling else current
            aligned.append(gmc.warp(history_frame, gmc.compute_affine(history_frame, current)))
        background = np.median(np.stack(aligned, axis=0), axis=0).astype(np.float32)
        residual = np.clip(current.astype(np.float32) - background, 0, 255).astype(np.uint8)
        features.append(np.stack([current, motion, residual], axis=2))
        rolling.append(current)
    return features, rolling


def scale_canvases(feature: np.ndarray, scales: list[int], imgsz: int, height: int, width: int):
    """One model-ready CHW uint8 canvas per scale, reversed into the trained channel order."""
    base_scale = min(imgsz / height, imgsz / width)
    new_w, new_h = round(width * base_scale), round(height * base_scale)
    canvases = [letterbox_gray(feature, imgsz)[..., ::-1].transpose(2, 0, 1)]
    mappings = [(base_scale, (imgsz - new_w) // 2, (imgsz - new_h) // 2)]
    for value in scales:
        canvas, scale, pad_x, pad_y = downsample_pad(feature, value, imgsz)
        canvases.append(canvas[..., ::-1].transpose(2, 0, 1))
        mappings.append((scale, pad_x, pad_y))
    return np.stack(canvases), mappings


def heatmap_to_native(heatmap: np.ndarray, scale: float, left: int, top: int, imgsz: int, width: int, height: int):
    """Undo one scale's letterbox and downsample so every scale lands on the native pixel grid."""
    full = cv2.resize(heatmap, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    new_w, new_h = round(width * scale), round(height * scale)
    return cv2.resize(full[top : top + new_h, left : left + new_w], (width, height), interpolation=cv2.INTER_LINEAR)


def detect_components(heatmap: np.ndarray, threshold: float, dilate: int, min_area: int, limit: int):
    """Connected components over the native-resolution heatmap, ordered by heatmap integral.

    Ordering matters because ``limit`` truncates: this mirrors ``multiscale_heatmap_cache.py``'s
    ``entries.sort(key=lambda item: -item[6])`` so a proposal is dropped for the same reason here as in
    every other cached run, and the ordering is stable rather than dependent on label numbering.
    """
    mask = (heatmap >= threshold).astype(np.uint8)
    if dilate > 1:
        size = dilate if dilate % 2 else dilate + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8), iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    entries = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < min_area:
            continue
        values = heatmap[labels == label]
        entries.append(
            (
                float(centroids[label][0]),
                float(centroids[label][1]),
                int(width),
                int(height),
                int(area),
                float(values.max()),
                float(values.sum()),
            )
        )
    entries.sort(key=lambda item: -item[6])
    return entries[:limit]


def run_heatmaps(model, canvases: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    """Run the heatmap detector over pre-padded canvases, halving the batch on CUDA OOM."""
    collected: list[np.ndarray] = []
    start, size = 0, max(1, batch)
    while start < len(canvases):
        count = min(size, len(canvases) - start)
        try:
            with torch.no_grad():
                tensor = torch.from_numpy(canvases[start : start + count]).to(device).float() / 255.0
                prediction = model(tensor)
                collected.extend(prediction["heatmap"][:, 0].cpu().numpy())
            start += count
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            if size == 1:
                raise
            size = max(1, count // 2)
            print(f"[WARN] CUDA OOM, reducing batch size to {size}", flush=True)
    return np.stack(collected)


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(max(1, args.cpu_threads))
    source = Path(args.source)
    out_dir = Path(args.prelabel_dir) / args.seq_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"state": "running", "seq_id": args.seq_id, "done": 0, "total": 0, "detections": 0, "message": "loading model"}
    write_status(args.status_file, status)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, stride = load_hm_model(resolve_weights(args.weights), device)
    print(f"[INFO] seq={args.seq_id} device={device} stride={stride} scales={[0] + args.scales}", flush=True)

    frame_iter = iter_gray_frames(source, args.max_frames)
    gmc = FastGMCEstimator(downscale=2)
    history: list[np.ndarray] = []
    history_window = args.median_window * args.temporal_stride + 2

    frame_index = 0
    detections_total = 0
    pending_gray: list[np.ndarray] = []
    pending_indices: list[int] = []
    scores: dict[str, list[float]] = {}
    progress = tqdm(unit="frame", dynamic_ncols=True, ascii=True)

    def flush(features: list[np.ndarray]) -> None:
        nonlocal detections_total
        if not features:
            return
        first = features[0]
        height, width = first.shape[:2]
        batch_maps = []
        canvases_list = []
        for feature in features:
            canvases, mappings = scale_canvases(feature, args.scales, args.imgsz, height, width)
            canvases_list.append(canvases)
            batch_maps.append(mappings)
        stacked = np.concatenate(canvases_list, axis=0)  # (frames * scales, C, S, S)
        heatmaps = run_heatmaps(model, stacked, device, args.batch_size)
        scale_count = 1 + len(args.scales)
        for position, index in enumerate(pending_indices):
            fused = None
            for slot in range(scale_count):
                raw = heatmaps[position * scale_count + slot]
                scale, left, top = batch_maps[position][slot]
                native = heatmap_to_native(raw, scale, left, top, args.imgsz, width, height)
                fused = native if fused is None else np.maximum(fused, native)
            entries = detect_components(fused, args.main_threshold, args.fusion_dilate, args.min_area, args.max_detections)
            boxes = [
                {"cls": 0, "x1": cx - bw / 2.0, "y1": cy - bh / 2.0, "x2": cx + bw / 2.0, "y2": cy + bh / 2.0}
                for cx, cy, bw, bh, _area, _peak, _total in entries
            ]
            detections_total += len(boxes)
            scores[f"{index:06d}"] = [round(peak, 5) for *_rest, peak, _total in entries]
            (out_dir / f"{index:06d}.txt").write_text(format_yolo(boxes, width, height), encoding="utf-8")
        progress.update(len(batch_maps))
        status.update({"done": frame_index, "detections": detections_total, "state": "running"})
        write_status(args.status_file, status)

    for gray in frame_iter:
        pending_gray.append(gray)
        frame_index += 1
        status["total"] = frame_index
        if len(pending_gray) >= args.chunk_size:
            features, history = build_features(pending_gray, history, args, gmc)
            history = history[-history_window:]
            pending_indices = list(range(frame_index - len(pending_gray), frame_index))
            pending_gray = []
            flush(features)

    if pending_gray:
        features, history = build_features(pending_gray, history, args, gmc)
        pending_indices = list(range(frame_index - len(pending_gray), frame_index))
        pending_gray = []
        flush(features)

    # Peak confidence per proposal, kept beside the labels: YOLO format has no score column, but the
    # annotator needs one to filter a noisy proposal set down to the plausible targets.
    (out_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    progress.close()
    elapsed = time.perf_counter() - started
    status.update(
        {
            "state": "done",
            "done": frame_index,
            "total": frame_index,
            "detections": detections_total,
            "elapsed": round(elapsed, 2),
            "fps": round(frame_index / elapsed, 2) if elapsed > 0 else 0.0,
            "message": f"{frame_index} frames, {detections_total} proposals",
        }
    )
    write_status(args.status_file, status)
    print(f"[DONE] {args.seq_id}: {frame_index} frames, {detections_total} proposals, {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
