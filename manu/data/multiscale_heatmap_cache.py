#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multi-scale Trial 0474 heatmap inference over restored sequences, caching sparse detections as pkl.

Each scale runs the same frozen SOTA heatmap detector and its heatmap is resampled back to native pixels;
the fused heatmap is the per-pixel ``np.maximum`` across scales, then thresholded, dilated and reduced to
connected components. This mirrors the last row of ``probe_heatmap_regions_video.py --fusion-row``.

Each sequence writes one ``<sequence>.pkl`` of CSR-style arrays, so a 165k-frame run stays far inside the
100 MB per-cache budget. ``points`` stays float32 on purpose: float16 at letterbox magnitudes quantises to
0.25-0.5 px and already cost TP-3/FP+3 once in this project.

    cache["frame_index"]  int32   (N,)     global frame index
    cache["offset"]       int64   (N+1,)   slice bounds into the detection arrays
    cache["points"]       float32 (M, 2)  centroid cx/cy in native pixels, never letterbox coordinates
    cache["sizes"]        float32 (M, 2)  component bounding box w/h in native pixels
    cache["scores"]       float16 (M,)    component peak heatmap value in [0, 1]
    cache["sums"]         float32 (M,)    component heatmap integral
    cache["area"]         int32   (M,)    component pixel count, the dilation/energy gate input
    cache["tag"]          uint8   (M,)    index into cache["meta"]["scales"], or 255 for the fused row

``cache["meta"]["source"]`` records whether frames came from raw grayscale ("raw_frames", files named
``frame_%06d.jpg``) or prebuilt features ("features", files named ``<sequence>__frame_%06d.jpg``); rebuild the
file name from ``frame_index`` with the matching pattern. Use ``iter_frames`` to walk a cache row by row.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_full_median_dataset import FastGMCEstimator  # noqa: E402
from manu.data.preannotate_hm_bbox import letterbox_gray, load_hm_model, resolve_weights  # noqa: E402
from manu.diagnostics.probe_heatmap_regions_video import downsample_pad  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
FUSED_TAG = 255
CACHE_BUDGET_BYTES = 100 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-scale Trial 0474 heatmap inference with pkl cache")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-root", help="Root of <sequence>/frame_XXXXXX.jpg; GMC+median recomputed on the fly")
    source.add_argument("--features-root", help="Root of <sequence>/__frame_XXXXXX.jpg 3-channel feature JPGs")
    parser.add_argument("--cache-root", required=True, help="Directory receiving one <sequence>.pkl per sequence")
    parser.add_argument("--sequences", default="", help="Comma-separated sequence names; empty means all")
    parser.add_argument("--shard", default="", help="Shard this run as INDEX/COUNT over the sorted sequence list")
    parser.add_argument("--scales", default="160,80", help="Downsample targets besides the native letterbox input")
    parser.add_argument("--region-threshold", type=float, default=0.06, help="Threshold of the per-scale rows")
    parser.add_argument("--main-threshold", type=float, default=0.22, help="Threshold of the fused row")
    parser.add_argument("--fusion-dilate", type=int, default=9, help="Odd dilation kernel applied before fusing")
    parser.add_argument("--min-area", type=int, default=4, help="Minimum connected-component pixel count")
    parser.add_argument("--max-detections", type=int, default=100, help="Detections cached per frame per source")
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=64, help="Frames held and inferred per chunk")
    parser.add_argument("--batch-size", type=int, default=8, help="GPU batch size, auto-halved on CUDA OOM")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--limit-per-sequence", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.scales = sorted({int(item) for item in args.scales.split(",") if item.strip() and int(item) > 0}, reverse=True)
    if min(args.chunk_size, args.max_detections) < 1:
        parser.error("--chunk-size and --max-detections must be positive")
    if args.min_area < 1:
        parser.error("--min-area must be positive")
    args.shard_index, args.shard_count = None, None
    if args.shard:
        try:
            args.shard_index, args.shard_count = (int(item) for item in args.shard.split("/"))
        except ValueError:
            parser.error("--shard must look like INDEX/COUNT")
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            parser.error(f"--shard index must satisfy 0 <= INDEX < COUNT, got {args.shard}")
    return args


def iter_frames(cache: dict):
    """Yield (frame_index, points, sizes, scores, sums, areas, tags) for one cached sequence."""
    offset = cache["offset"]
    for position, frame_index in enumerate(cache["frame_index"]):
        start, end = offset[position], offset[position + 1]
        yield (
            int(frame_index),
            cache["points"][start:end],
            cache["sizes"][start:end],
            cache["scores"][start:end],
            cache["sums"][start:end],
            cache["area"][start:end],
            cache["tag"][start:end],
        )


def collect_sequences(source_root: Path, features: bool, limit: int) -> dict[str, list[Path]]:
    """Group frames into {sequence: [paths]}, accepting both a per-sequence tree and a flat feature directory."""
    grouped: dict[str, list[Path]] = {}
    for directory in sorted(path for path in source_root.iterdir() if path.is_dir()):
        # pathlib glob drops a leading "_" from the pattern, so the feature prefix must stay non-empty
        glob = "*__frame_*" if features else "frame_*"
        paths = sorted(path for path in directory.glob(glob) if path.suffix.lower() in IMAGE_SUFFIXES)
        if paths:
            grouped[directory.name] = paths[:limit] if limit else paths

    if not grouped and features:  # flat layout, e.g. longquanshan_ir_gmc_median/images/train
        for path in sorted(source_root.glob("*__frame_*.jpg")):
            sequence, marker, _ = path.stem.partition("__frame_")
            if marker:
                grouped.setdefault(sequence, []).append(path)
        for sequence in grouped:
            grouped[sequence] = grouped[sequence][:limit] if limit else grouped[sequence]
    return grouped


def is_cuda_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def compute_features(
    chunk_grays: list[np.ndarray], history: list[np.ndarray], args: argparse.Namespace, gmc: FastGMCEstimator
) -> tuple[list[np.ndarray], list[np.ndarray]]:
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


def scale_inputs(feature: np.ndarray, scales: list[int], imgsz: int, height: int, width: int):
    """Build one model-ready CHW uint8 canvas per scale, plus the scale/offset mapping back to native pixels.

    ``feature`` uses the disk order [I_t, GMC diff, median residual]; every canvas is reversed to the model
    order [median residual, GMC diff, I_t] that Trial 0474 was trained on.
    """
    base_scale = min(imgsz / height, imgsz / width)
    new_w, new_h = round(width * base_scale), round(height * base_scale)
    canvases = [letterbox_gray(feature, imgsz)[..., ::-1].transpose(2, 0, 1)]
    mappings = [(base_scale, (imgsz - new_w) // 2, (imgsz - new_h) // 2)]
    for value in scales:
        canvas, scale, pad_x, pad_y = downsample_pad(feature, value, imgsz)
        canvases.append(canvas[..., ::-1].transpose(2, 0, 1))
        mappings.append((scale, pad_x, pad_y))
    return canvases, mappings


def heatmap_to_native(heatmap: np.ndarray, scale: float, left: int, top: int, imgsz: int, width: int, height: int):
    """Undo one scale's letterbox and downsample so every scale lands on the same native pixel grid."""
    full = cv2.resize(heatmap, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    new_w, new_h = round(width * scale), round(height * scale)
    return cv2.resize(full[top : top + new_h, left : left + new_w], (width, height), interpolation=cv2.INTER_LINEAR)


def detect_components(heatmap: np.ndarray, threshold: float, dilate: int, min_area: int, limit: int):
    """Connected components over the native-resolution heatmap, scored by component peak."""
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
    start = 0
    size = max(1, batch)
    while start < len(canvases):
        count = min(size, len(canvases) - start)
        try:
            with torch.no_grad():
                tensor = torch.from_numpy(canvases[start : start + count]).to(device).float() / 255.0
                prediction = model(tensor)
                collected.extend(prediction["heatmap"][:, 0].cpu().numpy())
            start += count
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            torch.cuda.empty_cache()
            if size == 1:
                raise
            size = max(1, count // 2)
            print(f"[WARN] CUDA OOM during heatmap inference, reducing batch size to {size}", flush=True)
    return np.stack(collected)


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(max(1, args.cpu_threads))
    source_root = Path(args.frames_root or args.features_root)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "summaries").mkdir(exist_ok=True)

    sequences = collect_sequences(source_root, bool(args.features_root), args.limit_per_sequence)
    wanted = {item.strip() for item in args.sequences.split(",") if item.strip()}
    if wanted:
        unknown = wanted - set(sequences)
        if unknown:
            raise KeyError(f"Unknown sequences under {source_root}: {sorted(unknown)}")
        sequences = {name: paths for name, paths in sequences.items() if name in wanted}
    if args.shard_count:
        sequences = {
            name: paths
            for index, (name, paths) in enumerate(sequences.items())
            if index % args.shard_count == args.shard_index
        }
    if not sequences:
        raise FileNotFoundError(f"No sequences selected under {source_root}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model, stride = load_hm_model(resolve_weights(args.hm_weights), device)
    all_scales = [0] + args.scales
    print(
        f"[INFO] sequences={len(sequences)} | scales={all_scales} (0 = native letterbox) | stride={stride} | device={device}\n"
        f"[INFO] fused row = per-pixel max across scales >= {args.main_threshold} dilated by {args.fusion_dilate}, "
        f"components with area >= {args.min_area}",
        flush=True,
    )

    started = time.perf_counter()
    for position, (sequence_name, frame_paths) in enumerate(sequences.items(), 1):
        cache_path = cache_root / f"{sequence_name}.pkl"
        if cache_path.exists() and not args.overwrite:
            print(f"[WARN] {sequence_name}: {cache_path.name} exists, skipped (pass --overwrite)", flush=True)
            continue

        print(f"\n[SEQUENCE {position}/{len(sequences)}] {sequence_name} | frames={len(frame_paths)}", flush=True)
        frame_indices, per_frame_counts = [], []
        points, sizes, scores, sums, areas, tags = [], [], [], [], [], []
        fused_total, per_scale_total = 0, [0] * len(all_scales)
        gmc = FastGMCEstimator(downscale=2)
        history: list[np.ndarray] = []
        history_window = args.median_window * args.temporal_stride + 2

        for start in tqdm(
            range(0, len(frame_paths), args.chunk_size), desc=f"{sequence_name}", unit="chunk", dynamic_ncols=True
        ):
            chunk_paths = frame_paths[start : start + args.chunk_size]
            if args.features_root:
                chunk_features = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in chunk_paths]
                if any(item is None or item.ndim != 3 or item.shape[2] != 3 for item in chunk_features):
                    raise RuntimeError(f"Invalid 3-channel feature image in {sequence_name}")
            else:
                chunk_grays = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in chunk_paths]
                if any(item is None for item in chunk_grays):
                    raise RuntimeError(f"Unreadable frame in {sequence_name}")
                chunk_features, history = compute_features(chunk_grays, history, args, gmc)
                history = history[-history_window:]
            if start == 0:
                height, width = chunk_features[0].shape[:2]
                if args.features_root is None and (width, height) != (640, 512):
                    raise ValueError(f"{sequence_name}: expected 640x512 restored native, got {width}x{height}")

            canvases = np.zeros((len(chunk_features) * len(all_scales), 3, args.imgsz, args.imgsz), dtype=np.uint8)
            chunk_mappings = []
            for index, feature in enumerate(chunk_features):
                scale_canvases, mappings = scale_inputs(feature, args.scales, args.imgsz, height, width)
                canvases[index * len(all_scales) : (index + 1) * len(all_scales)] = np.stack(scale_canvases)
                chunk_mappings.append(mappings)

            heatmaps = run_heatmaps(model, canvases, device, args.batch_size)
            if device.type == "cuda":
                torch.cuda.empty_cache()

            for index in range(len(chunk_features)):
                span = slice(index * len(all_scales), (index + 1) * len(all_scales))
                native = [
                    heatmap_to_native(
                        heatmaps[span][scale_id], *chunk_mappings[index][scale_id], args.imgsz, width, height
                    )
                    for scale_id in range(len(all_scales))
                ]
                frame_index = start + index
                per_frame_counts.append(0)

                for scale_id, heatmap in enumerate(native):
                    entries = detect_components(heatmap, args.region_threshold, 1, args.min_area, args.max_detections)
                    per_scale_total[scale_id] += len(entries)
                    for cx, cy, box_w, box_h, area, peak, total in entries:
                        points.append((cx, cy))
                        sizes.append((box_w, box_h))
                        scores.append(peak)
                        sums.append(total)
                        areas.append(area)
                        tags.append(scale_id)
                        per_frame_counts[-1] += 1

                fused = detect_components(
                    np.maximum.reduce(native),
                    args.main_threshold,
                    args.fusion_dilate,
                    args.min_area,
                    args.max_detections,
                )
                fused_total += len(fused)
                for cx, cy, box_w, box_h, area, peak, total in fused:
                    points.append((cx, cy))
                    sizes.append((box_w, box_h))
                    scores.append(peak)
                    sums.append(total)
                    areas.append(area)
                    tags.append(FUSED_TAG)
                    per_frame_counts[-1] += 1

                frame_indices.append(frame_index)

        cache = {
            "meta": {
                "sequence": sequence_name,
                "frames": len(frame_indices),
                "native_size": [width, height],
                "imgsz": args.imgsz,
                "stride": stride,
                "scales": all_scales,
                "fused_tag": FUSED_TAG,
                "coordinate_space": "native_pixel",
                "source": "features" if args.features_root else "raw_frames",
                "region_threshold": args.region_threshold,
                "main_threshold": args.main_threshold,
                "fusion_dilate": args.fusion_dilate,
                "min_area": args.min_area,
            },
            "frame_index": np.asarray(frame_indices, dtype=np.int32),
            "offset": np.concatenate([[0], np.cumsum(np.asarray(per_frame_counts, dtype=np.int64))]).astype(np.int64),
            "points": np.asarray(points, dtype=np.float32).reshape(-1, 2),
            "sizes": np.asarray(sizes, dtype=np.float32).reshape(-1, 2),
            "scores": np.asarray(scores, dtype=np.float16),
            "sums": np.asarray(sums, dtype=np.float32),
            "area": np.asarray(areas, dtype=np.int32),
            "tag": np.asarray(tags, dtype=np.uint8),
        }
        cache_path.write_bytes(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
        size_mb = cache_path.stat().st_size / 1024 / 1024
        per_scale = {str(all_scales[value]): per_scale_total[value] for value in range(len(all_scales))}
        summary = {
            "sequence": sequence_name,
            "frames": cache["meta"]["frames"],
            "detections": int(len(cache["tag"])),
            "per_scale_counts": per_scale,
            "fused_count": fused_total,
            "cache": str(cache_path.resolve()),
            "cache_mb": round(size_mb, 3),
        }
        (cache_root / "summaries" / f"{sequence_name}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[DONE] {sequence_name}: fused={fused_total} | per-scale={per_scale} | cache={size_mb:.2f} MB", flush=True
        )
        if size_mb * 1024 * 1024 > CACHE_BUDGET_BYTES:
            print(
                f"[WARN] {cache_path.name} exceeds the 100 MB per-cache budget; lower --max-detections and rerun "
                "with --overwrite",
                flush=True,
            )

    print(f"\n[SUCCESS] sequences={len(sequences)} | cache -> {cache_root.resolve()}", flush=True)
    print(f"[SUCCESS] elapsed {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
