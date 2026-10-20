"""Two-stage mining of adapter training samples for the frozen Trial 0474 heatmap detector.

Stage A (CPU, cache only)
    Reads ``runs/ms_cache_nop0/<sequence>.pkl`` -- thresholded connected-component peaks --
    plus the ground-truth boxes, and produces an OVERSAMPLED candidate pool per class.
    The cache cannot answer "what is the response INSIDE a box that never reached
    ``main_threshold``", so nothing is finalised here.

Stage B (GPU, dense re-run)
    Re-runs the frozen base model on the surviving candidate frames to obtain the dense
    heatmap, then applies the criteria the cache cannot express:
      * hard-FN : box max response < ``--fn-response`` (the 0.08~0.18 drowned band) or >= 2
                  connected peaks inside one box (target splitting);
      * silent  : whole-frame max response < ``--silent-response``;
      * anchor  : Anti-UAV box area in [4, 36] with base response >= ``--anchor-response``.

Sequence governance (non-negotiable, enforced in code, see ``FORBIDDEN``):
    the 3 validation sequences and the 10 user-``<ignore>`` sequences are never mined,
    never used as anchors and never scored by this script.

Outputs
    <out-dir>/manifest.json          one record per accepted frame
    <out-dir>/samples/<seq>__<idx>.npz   h_base (float16, model grid) + guide (uint8, 640x640)
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manu.data.multiscale_heatmap_cache import collect_sequences, run_heatmaps  # noqa: E402
from manu.data.preannotate_hm_bbox import (  # noqa: E402
    letterbox_gray,
    letterbox_params,
    load_hm_model,
    resolve_weights,
)
from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402

# --------------------------------------------------------------------------------------
# Sequence pools -- sourced from docs/dataset/long-quan.md (2026-10-19 human review)
# --------------------------------------------------------------------------------------
VALIDATION_SEQUENCES = [
    "VIDEO00028_19700101_013631",  # big-target splitting probe
    "VIDEO00008_19700101_001307",  # tree/city clutter probe
    "VIDEO00009_19700101_001531",  # pure-sky canary
]

#: 10 sequences the user marked ``<ignore>`` on 2026-10-19 -- excluded from every stage.
IGNORE_SEQUENCES = [
    "VIDEO00032_19700101_014453",  # black-hot
    "VIDEO00036_19700101_015740",  # black-hot, very low SNR
    "VIDEO00033_19700101_015000",  # black-hot prefix
    "VIDEO00031_19700101_014227",  # black-hot suffix
    "VIDEO00035_19700101_015138",  # black-hot tail
    "VIDEO00007_19700101_001324",  # brief black-hot switch
    "VIDEO00006_19700101_000322",  # IR not calibrated
    "VIDEO00003_19700101_002111",  # IR not calibrated
    "VIDEO00004_19700101_002438",  # not calibrated + IR fog
    "VIDEO00012_19700101_002251",  # out-of-scope top-down ground view
]

FORBIDDEN = set(VALIDATION_SEQUENCES) | set(IGNORE_SEQUENCES)

SAMPLE_CLASSES = ("hard_fp", "hard_fn", "silent_sky", "anchor")


def feature_path_for(features_root: Path, sequence: str, frame_index: int) -> Path:
    """Locate one feature JPEG in either supported layout.

    ``longquanshan_ir_gmc_median/images/train`` is FLAT (``<seq>__frame_000123.jpg``); other
    trees keep one directory per sequence. Both exist in the wild, so try both instead of
    assuming -- assuming cost one wasted run already.
    """
    name = f"{sequence}__frame_{frame_index:06d}.jpg"
    flat = features_root / name
    if flat.exists():
        return flat
    return features_root / sequence / name


def resolve_train_pool(sequence_root: Path) -> list[str]:
    """All trainable sequences under the feature root minus validation and ignored."""
    available = set(collect_sequences(sequence_root, True, 0))
    if not available:
        raise FileNotFoundError(f"[SEQ GATE] no feature sequence found under {sequence_root}")
    train = sorted(available - FORBIDDEN)
    leaked = sorted(set(train) & FORBIDDEN)
    if leaked:
        raise RuntimeError(f"[SEQ GATE] FAILED: forbidden sequences reached the train pool: {leaked}")
    if not train:
        raise FileNotFoundError(f"[SEQ GATE] no trainable sequence under {sequence_root}")
    return train


def cache_frames(cache: dict) -> dict[int, np.ndarray]:
    """frame_index -> cached peak array ``(cx, cy, score)`` in native pixels."""
    per_frame: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for frame_index, points, scores, _, _, _, _ in _iter_cache(cache):
        for point, score in zip(points, scores):
            per_frame[int(frame_index)].append((float(point[0]), float(point[1]), float(score)))
    return {index: np.asarray(rows, dtype=np.float32) for index, rows in per_frame.items()}


def _iter_cache(cache: dict):
    frame_index = cache["frame_index"]
    offset = cache["offset"]
    for position in range(len(frame_index)):
        start, end = int(offset[position]), int(offset[position + 1])
        yield (
            int(frame_index[position]),
            cache["points"][start:end],
            cache["scores"][start:end].astype(np.float32),
            cache["sizes"][start:end],
            cache["sums"][start:end],
            cache["area"][start:end],
            cache["tag"][start:end],
        )


def peak_distance_to_boxes(peaks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Euclidean distance from every peak centre to the nearest GT box centre."""
    if len(boxes) == 0:
        return np.full(len(peaks), np.inf, dtype=np.float32)
    centers = boxes[:, :2]
    diff = peaks[:, None, :2] - centers[None, :, :]
    return np.sqrt((diff**2).sum(axis=2)).min(axis=1)


def peaks_inside_boxes(peaks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Boolean mask of peaks whose centre falls inside any GT box (inbox rule)."""
    if len(peaks) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.zeros(len(peaks), dtype=bool)
    for cx, cy, w, h in boxes:
        mask |= (np.abs(peaks[:, 0] - cx) <= w / 2) & (np.abs(peaks[:, 1] - cy) <= h / 2)
    return mask


# --------------------------------------------------------------------------------------
# Stage A
# --------------------------------------------------------------------------------------
def stage_a_candidates(args: argparse.Namespace, sequences: list[str]) -> dict[str, dict[str, list[dict]]]:
    cache_root = Path(args.cache_root)
    labels_root = Path(args.labels_root)
    pools: dict[str, list[dict]] = {name: [] for name in SAMPLE_CLASSES}
    oversample = max(1.0, args.oversample)
    rng = np.random.default_rng(args.seed)

    for position, sequence in enumerate(sequences, 1):
        cache_path = cache_root / f"{sequence}.pkl"
        if not cache_path.exists():
            print(f"[WARN] {sequence}: no cache at {cache_path}, skipped", flush=True)
            continue
        cache = pickle.loads(cache_path.read_bytes())
        meta = cache["meta"]
        if meta.get("sequence") != sequence:
            raise RuntimeError(f"[CACHE GATE] {cache_path} carries sequence={meta.get('sequence')}")
        if meta.get("use_p0_highway", True):
            raise RuntimeError(
                f"[CACHE GATE] {sequence}: cache was built WITH the P0 highway; "
                "the adapter must be tuned against the de-P0 arm"
            )
        width, height = int(meta["native_size"][0]), int(meta["native_size"][1])
        total_frames = int(meta["frames"])

        truth = load_ground_truth(labels_root, sequence, width, height)
        peaks_by_frame = cache_frames(cache)

        # stride bookkeeping: same coarse location must not be mined twice in a row
        last_taken: dict[tuple[int, int], int] = {}

        def stride_ok(frame_index: int, cell: tuple[int, int]) -> bool:
            previous = last_taken.get(cell)
            # the document asks for a 30~50 frame stride; draw the bound per acceptance so a
            # stationary false positive is re-sampled at an irregular cadence rather than on a
            # fixed lattice that could alias with the sequence's own periodicity
            bound = int(rng.integers(args.stride_min, args.stride_max + 1))
            if previous is None or frame_index - previous >= bound:
                last_taken[cell] = frame_index
                return True
            return False

        for frame_index in range(total_frames):
            peaks = peaks_by_frame.get(frame_index)
            if peaks is None:
                peaks = np.zeros((0, 3), dtype=np.float32)
            boxes = truth.get(frame_index)
            boxes = np.zeros((0, 4), dtype=np.float32) if boxes is None else boxes.astype(np.float32)
            area = boxes[:, 2] * boxes[:, 3] if len(boxes) else np.zeros(0, dtype=np.float32)

            # (1) hard false positives: confident peaks far from every ground-truth centre
            if len(peaks):
                confident = peaks[peaks[:, 2] >= args.fp_min_score]
                if len(confident):
                    far = peak_distance_to_boxes(confident, boxes) > args.fp_min_distance
                    for point in confident[far]:
                        cell = (int(point[0]) // args.stride_cell, int(point[1]) // args.stride_cell)
                        if not stride_ok(frame_index, cell):
                            continue
                        pools["hard_fp"].append(
                            {
                                "sequence": sequence,
                                "frame_index": frame_index,
                                "peak_xy": [round(float(point[0]), 2), round(float(point[1]), 2)],
                                "peak_score": round(float(point[2]), 4),
                                "priority": float(point[2]),
                            }
                        )

            # (2) hard false negatives: big or split boxes. The response-level test is
            #     deferred to stage B because the cache cannot see sub-threshold peaks.
            if len(boxes):
                counts = np.zeros(len(boxes), dtype=np.int32)
                for index, box in enumerate(boxes):
                    counts[index] = int(
                        (
                            (np.abs(peaks[:, 0] - box[0]) <= box[2] / 2)
                            & (np.abs(peaks[:, 1] - box[1]) <= box[3] / 2)
                        ).sum()
                    )
                previous = truth.get(frame_index - 1)
                previous = np.zeros((0, 4), dtype=np.float32) if previous is None else previous
                for index, box in enumerate(boxes):
                    box_area = float(area[index])
                    split = counts[index] >= args.split_peaks
                    big_and_blind = box_area > args.fn_min_area and counts[index] == 0
                    hovering = False
                    shift = float("inf")
                    if len(previous):
                        shift = float(
                            np.sqrt(((previous[:, :2] - box[None, :2]) ** 2).sum(axis=1)).min()
                        )
                        hovering = shift <= args.fn_hover_shift
                    if not (split or big_and_blind or hovering):
                        continue
                    pools["hard_fn"].append(
                        {
                            "sequence": sequence,
                            "frame_index": frame_index,
                            "box_xywh": [round(float(v), 2) for v in box],
                            "box_area": round(box_area, 2),
                            "cached_peaks_in_box": int(counts[index]),
                            "split": bool(split),
                            "hovering": bool(hovering),
                            "frame_shift_px": round(shift, 2) if np.isfinite(shift) else None,
                            "priority": float(counts[index] * 10 + math.log1p(box_area) + (5 if hovering else 0)),
                        }
                    )

            # (3) silent sky: frames with no cached peak at all are the only cache-visible
            #     proxy for "frame max response is low"; stage B measures the real value.
            if len(peaks) == 0 and not len(boxes):
                pools["silent_sky"].append(
                    {
                        "sequence": sequence,
                        "frame_index": frame_index,
                        "priority": 0.0,
                    }
                )

        print(
            f"[STAGE A {position}/{len(sequences)}] {sequence:<30} "
            f"fp={len(pools['hard_fp']):<6} fn={len(pools['hard_fn']):<6} "
            f"sky={len(pools['silent_sky']):<6}",
            flush=True,
        )

    # cap each pool so stage B does not have to re-run the world
    caps = {
        "hard_fp": int(args.quota_fp * oversample),
        "hard_fn": int(args.quota_fn * oversample),
        "silent_sky": int(args.quota_silent * oversample),
        "anchor": int(args.quota_anchor * oversample),
    }
    for name, items in pools.items():
        items.sort(key=lambda row: -row["priority"])
        pools[name] = items[: caps[name]]
    return pools


# --------------------------------------------------------------------------------------
# Stage B
# --------------------------------------------------------------------------------------
def build_native_canvas(feature: np.ndarray, imgsz: int, width: int, height: int):
    """Model-ready CHW canvas for the native scale plus its letterbox mapping."""
    scale, left, top, _, _ = letterbox_params(width, height, imgsz)
    canvas = letterbox_gray(feature, imgsz)[..., ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(canvas), (scale, left, top)


def guide_channel(canvas: np.ndarray, index: int) -> np.ndarray:
    """Pick one channel of the model-ordered canvas as the adapter's guidance image."""
    return canvas[index].astype(np.uint8)


def stage_b_filter(args, candidates, model, device, sequence_dims, batch, out_dir):
    """Re-run the frozen model on candidates, apply dense criteria, save npz + records."""
    records = []
    saved = 0
    rejected = defaultdict(int)
    for position, ((sequence, frame_index), items) in enumerate(candidates, 1):
        width, height = sequence_dims[sequence]
        feature_path = feature_path_for(Path(args.features_root), sequence, frame_index)
        if not feature_path.exists():
            rejected["missing_feature"] += 1
            continue
        feature = cv2.imread(str(feature_path), cv2.IMREAD_COLOR)
        if feature is None:
            rejected["unreadable_feature"] += 1
            continue
        canvas, (scale, left, top) = build_native_canvas(feature, args.imgsz, width, height)
        heatmaps = run_heatmaps(model, canvas[None, ...], device, batch)
        heat = heatmaps[0].astype(np.float16)

        guide = guide_channel(canvas, args.guide_channel)
        kept_here = []
        for item in items:
            kind = item["sample_class"]
            dense = cv2.resize(heat.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            if kind == "hard_fp":
                # A confident peak far from every GT centre IS a false positive; there is no
                # dense criterion to apply. Emit it with the peak already expressed in 640-canvas
                # fractions so the dataset can rasterise the suppression mask without re-deriving
                # the letterbox mapping.
                px, py = item["peak_xy"]
                kept_here.append(
                    dict(
                        item,
                        peak_canvas_xy=[
                            round((px * scale + left) / args.imgsz, 6),
                            round((py * scale + top) / args.imgsz, 6),
                        ],
                        peak_canvas_radius=args.fp_mask_radius,
                    )
                )
            elif kind == "hard_fn":
                cx, cy, box_w, box_h = item["box_xywh"]
                x0, x1 = int(max(0, cx - box_w / 2)), int(min(width, cx + box_w / 2))
                y0, y1 = int(max(0, cy - box_h / 2)), int(min(height, cy + box_h / 2))
                if x1 <= x0 or y1 <= y0:
                    rejected["empty_box"] += 1
                    continue
                box_max = float(dense[y0:y1, x0:x1].max())
                if item["split"] or box_max < args.fn_response:
                    # store the box already in 640-canvas fractions so training never re-derives the map
                    canvas_box = (
                        (cx * scale + left) / args.imgsz,
                        (cy * scale + top) / args.imgsz,
                        box_w * scale / args.imgsz,
                        box_h * scale / args.imgsz,
                    )
                    item = dict(
                        item,
                        box_max=round(box_max, 4),
                        box_canvas_cxcywh=[round(float(v), 6) for v in canvas_box],
                    )
                    kept_here.append(item)
                else:
                    rejected["fn_response_too_high"] += 1
            elif kind == "silent_sky":
                frame_max = float(dense.max())
                if frame_max < args.silent_response:
                    kept_here.append(dict(item, frame_max=round(frame_max, 4)))
                else:
                    rejected["sky_not_silent"] += 1
            else:
                rejected["unexpected_class"] += 1

        if not kept_here:
            continue
        sample_path = out_dir / "samples" / f"{sequence}__{frame_index:06d}.npz"
        np.savez_compressed(sample_path, h_base=heat, guide=guide)
        for item in kept_here:
            record = dict(item)
            record.pop("priority", None)
            record["npz"] = str(sample_path.relative_to(out_dir))
            records.append(record)
            saved += 1

        if position % 100 == 0:
            print(f"[STAGE B] {position}/{len(candidates)} frames | accepted={saved} | {dict(rejected)}", flush=True)
    return records, dict(rejected), saved


def mine_anchor_samples(args, model, device, out_dir) -> tuple[list[dict], dict]:
    """Anti-UAV golden anchors mined from the ORIGINAL TRAINING FEATURES.

    These are the only samples carrying ``is_anchor=1``, so they are what makes the one-sided
    ``L_anchor`` protection term in the loss do anything at all.

    The frames come from ``uav_gmc_median/images/train`` -- the exact dataset Trial 0474 was
    trained on -- and are streamed through the project's own YOLO validation dataloader, so the
    letterbox geometry, the guide channel and the GT convention are identical to
    ``cache_trial0474_inferences.py``. Anything less would make the anchor inputs a different
    distribution from the blind-test inputs.

    ``taken`` counts ACCEPTED frames, so a sequence keeps scanning past its candidates until it
    reaches the per-sequence cap instead of stopping at the first rejection.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    yaml_path = Path(args.anchor_data_yaml)
    if not yaml_path.is_file():
        print(f"[ANCHOR] data.yaml {yaml_path} missing -> anchor quota 0", flush=True)
        return [], {"reason": "missing_data_yaml"}

    data_dict = check_det_dataset(str(yaml_path))
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = str(yaml_path)
    dataset = build_yolo_dataset(
        cfg, data_dict["train"], batch=args.batch_size, data=data_dict, mode="val", stride=32
    )
    loader = build_dataloader(dataset, batch=args.batch_size, workers=4, shuffle=False)
    print(
        f"[ANCHOR] {yaml_path} train split = {len(dataset)} images | "
        f"per-sequence cap={args.anchor_per_sequence} | target={args.quota_anchor}",
        flush=True,
    )

    records: list[dict] = []
    rejected: dict[str, int] = defaultdict(int)
    taken: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc="anchor mining", unit="batch", dynamic_ncols=True):
            if len(records) >= args.quota_anchor:
                break
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            bboxes, b_idx = batch["bboxes"], batch["batch_idx"]
            im_files = batch.get("im_file", [""] * imgs.shape[0])

            heatmap = model(imgs)["heatmap"]
            grid_h, grid_w = heatmap.shape[-2:]
            guide = torch.nn.functional.interpolate(
                imgs[:, args.anchor_guide_channel : args.anchor_guide_channel + 1],
                size=(grid_h, grid_w), mode="bilinear", align_corners=False,
            )

            for b in range(imgs.shape[0]):
                if len(records) >= args.quota_anchor:
                    break
                stem = Path(im_files[b]).stem if im_files[b] else ""
                sequence = stem.partition("__")[0]
                if not sequence:
                    rejected["no_name"] += 1
                    continue
                if taken[sequence] >= args.anchor_per_sequence:
                    continue

                boxes = bboxes[b_idx == b].cpu().numpy()
                if len(boxes) == 0:
                    rejected["no_box"] += 1
                    continue
                # areas are measured in the 640 letterbox space, exactly as the cache script
                # stores GT, so the criterion means the same thing here and in the blind test
                areas = (boxes[:, 2] * args.imgsz) * (boxes[:, 3] * args.imgsz)
                small = np.nonzero((areas >= args.anchor_min_area) & (areas <= args.anchor_max_area))[0]
                if not len(small):
                    rejected["no_small_target"] += 1
                    continue

                heat = heatmap[b, 0].float().cpu().numpy()
                # store the guide at the 640 canvas resolution, exactly like the Longquan path,
                # so the dataset resizes one geometry for both sources
                guide_np = (imgs[b, args.anchor_guide_channel].cpu().numpy() * 255.0).clip(0, 255)
                chosen = None
                for index in small:
                    cx = float(boxes[index][0]) * grid_w
                    cy = float(boxes[index][1]) * grid_h
                    half_w = float(boxes[index][2]) * grid_w / 2
                    half_h = float(boxes[index][3]) * grid_h / 2
                    x0, x1 = int(max(0, round(cx - half_w))), int(min(grid_w, round(cx + half_w) + 1))
                    y0, y1 = int(max(0, round(cy - half_h))), int(min(grid_h, round(cy + half_h) + 1))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    response = float(heat[y0:y1, x0:x1].max())
                    if response >= args.anchor_response:
                        chosen = (index, response)
                        break
                if chosen is None:
                    rejected["anchor_response_too_low"] += 1
                    continue

                index, response = chosen
                sample_path = out_dir / "samples" / f"ANTIUAV_{sequence}_{stem.split('__')[-1]}.npz"
                np.savez_compressed(
                    sample_path,
                    h_base=heatmap[b, 0].cpu().numpy().astype(np.float16),
                    guide=guide_np.astype(np.uint8),
                )
                records.append(
                    {
                        "sample_class": "anchor",
                        "sequence": sequence,
                        "source": "uav_gmc_median_train",
                        "stem": stem,
                        # batch["bboxes"] is already [cx, cy, w, h] with NO class column
                        # (cache_trial0474_inferences.py reads box[0] as cx), so slice [0:4].
                        "box_canvas_cxcywh": [round(float(v), 6) for v in boxes[index][0:4]],
                        "box_area_px": round(float(areas[index]), 2),
                        "box_response": round(response, 4),
                        "npz": str(sample_path.relative_to(out_dir)),
                    }
                )
                taken[sequence] += 1

    print(
        f"[ANCHOR] accepted={len(records)} (target {args.quota_anchor}) over "
        f"{len(taken)} sequences | {dict(rejected)}",
        flush=True,
    )
    if len(records) < args.quota_anchor:
        print(
            f"[ANCHOR][WARN] short of quota by {args.quota_anchor - len(records)}; "
            f"raise --anchor-per-sequence (currently {args.anchor_per_sequence}) to widen the pool",
            flush=True,
        )
    return records, dict(rejected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine adapter training samples (two-stage)")
    parser.add_argument("--cache-root", default="runs/ms_cache_nop0")
    parser.add_argument("--labels-root", required=True, help="Final_Labels_merged_20261010")
    parser.add_argument("--features-root", required=True, help="longquanshan_ir_gmc_median/images/train")
    parser.add_argument("--out-dir", default="runs/adapter_mine")
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--guide-channel", type=int, default=0, choices=[0, 1, 2],
                        help="Longquan only, AFTER scale_inputs() reversal: 0=median residual, 1=GMC diff, 2=current frame")
    parser.add_argument("--anchor-guide-channel", type=int, default=2, choices=[0, 1, 2],
                        help="Anchors only, RAW on-disk order of uav_gmc_median which is NOT reversed: 0=I_t, 1=motion, 2=median residual. Defaults to 2 so both sources feed the same residual channel.")
    # quotas
    parser.add_argument("--quota-fp", type=int, default=600)
    parser.add_argument("--quota-fn", type=int, default=500)
    parser.add_argument("--quota-silent", type=int, default=200)
    parser.add_argument("--quota-anchor", type=int, default=200)
    parser.add_argument("--oversample", type=float, default=4.0)
    # stage A criteria
    parser.add_argument("--fp-min-score", type=float, default=0.30)
    parser.add_argument("--fp-min-distance", type=float, default=48.0)
    parser.add_argument("--fn-min-area", type=float, default=80.0)
    parser.add_argument("--split-peaks", type=int, default=2)
    parser.add_argument("--fp-mask-radius", type=float, default=0.01,
                        help="Suppression-mask radius as a fraction of the 640 canvas (~6.4 px)")
    parser.add_argument("--stride-min", type=int, default=30)
    parser.add_argument("--stride-cell", type=int, default=64)
    # stage B criteria
    parser.add_argument("--fn-response", type=float, default=0.22,
                        help="Keep a hard-FN only when the in-box dense max is below this")
    parser.add_argument("--silent-response", type=float, default=0.10)
    parser.add_argument("--anchor-response", type=float, default=0.50)
    parser.add_argument("--anchor-min-area", type=float, default=4.0)
    parser.add_argument("--anchor-max-area", type=float, default=36.0)
    parser.add_argument("--anchor-per-sequence", type=int, default=5,
                        help="Maximum anchor frames taken from any single Anti-UAV sequence")
    parser.add_argument("--anchor-data-yaml", default="/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml",
                        help="ORIGINAL TRAINING feature dataset; its train split supplies the anchors")
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261019)
    # hover-specific hard-FN band (document §3.2): a hovering target whose in-box response
    # sits in this band is the case the adapter must lift; outside it the general rule applies.
    parser.add_argument("--fn-hover-shift", type=float, default=3.0,
                        help="Consecutive-frame centre shift under which a target counts as hovering")
    parser.add_argument("--fn-hover-band", default="0.08,0.18",
                        help="In-box response band that marks a hovering hard-FN")
    parser.add_argument("--stride-max", type=int, default=50,
                        help="Upper end of the 30~50 frame temporal sampling stride")
    parser.add_argument("--skip-stage-b", action="store_true", help="Stage A only (dry run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    feature_root = Path(args.features_root)
    train_pool = resolve_train_pool(feature_root)
    print(f"[SEQ GATE] train pool={len(train_pool)} sequences (validation {len(VALIDATION_SEQUENCES)} "
          f"and ignore {len(IGNORE_SEQUENCES)} hard-excluded)", flush=True)
    print(f"[SEQ GATE] train pool = {train_pool}", flush=True)

    pools = stage_a_candidates(args, train_pool)
    print("\n[STAGE A SUMMARY] " + "  ".join(f"{name}={len(rows)}" for name, rows in pools.items()), flush=True)

    records: list[dict] = []
    reject_report: dict = {}
    if not args.skip_stage_b:
        device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
        )
        # Single model, single arm: Trial 0474 WITH its P0 highway, which is exactly the frozen
        # SOTA headline. The de-P0 ablation is an RK3588 shipping decision and must not be a
        # switch threaded through mining -- that is how the npz heatmaps ended up on the wrong
        # arm relative to inference once already.
        model, stride = load_hm_model(resolve_weights(args.hm_weights), device, use_p0_highway=True)
        print(f"[STAGE B] model loaded (WITH-P0) stride={stride} device={device}", flush=True)

        dims = {}
        for sequence in train_pool:
            cache_path = Path(args.cache_root) / f"{sequence}.pkl"
            if cache_path.exists():
                meta = pickle.loads(cache_path.read_bytes())["meta"]
                dims[sequence] = (int(meta["native_size"][0]), int(meta["native_size"][1]))

        # group candidates by (sequence, frame) so the base forward runs once per frame
        grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for name in ("hard_fp", "hard_fn", "silent_sky"):
            for item in pools[name]:
                item = dict(item, sample_class=name)
                grouped[(item["sequence"], item["frame_index"])].append(item)
        candidates = sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        print(f"[STAGE B] candidate frames={len(candidates)}", flush=True)

        records, reject_report, _ = stage_b_filter(
            args, candidates, model, device, dims, args.batch_size, out_dir
        )

        # Anchors come from a different dataset (the ORIGINAL TRAINING features), a different
        # dataloader and the WITH-P0 base model, so they are mined in their own pass and simply
        # appended to the Longquan records.
        print(f"\n[STAGE C] mining {args.quota_anchor} Anti-UAV anchors from the training features", flush=True)
        anchors, anchor_rejects = mine_anchor_samples(args, model, device, out_dir)
        records.extend(anchors)
        reject_report["anchor"] = anchor_rejects

    manifest = {
        "created_from": {
            "cache_root": args.cache_root,
            "labels_root": args.labels_root,
            "features_root": args.features_root,
            "hm_weights": args.hm_weights,
        },
        "governance": {
            "validation_sequences": VALIDATION_SEQUENCES,
            "ignore_sequences": IGNORE_SEQUENCES,
            "train_pool": train_pool,
        },
        "criteria": {
            "fp_min_score": args.fp_min_score,
            "fp_min_distance": args.fp_min_distance,
            "fn_min_area": args.fn_min_area,
            "fn_response": args.fn_response,
            "split_peaks": args.split_peaks,
            "silent_response": args.silent_response,
            "stride_min": args.stride_min,
            "guide_channel": args.guide_channel,
        },
        "counts": {name: sum(1 for row in records if row["sample_class"] == name) for name in SAMPLE_CLASSES},
        "stage_a_candidates": {name: len(rows) for name, rows in pools.items()},
        "stage_b_rejections": reject_report,
        "samples": records,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[MANIFEST] {manifest_path} | counts={manifest['counts']} | total={len(records)}", flush=True)
    if not records and not args.skip_stage_b:
        print("[WARN] stage B accepted zero samples; inspect stage_b_rejections before training", flush=True)


if __name__ == "__main__":
    main()
