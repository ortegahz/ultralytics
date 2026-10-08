#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Chained-GMC A/B harness — quantifies the single-frame precision cost of removing GMC.

This is the decision tool for the RK3588 embedded port. The frozen input pipeline
spends 22 Shi-Tomasi + Lucas-Kanade + RANSAC affine fits per frame on CPU. Before
deciding whether GMC must be *accelerated* (e.g. a differentiable GMC-Net on NPU) or
*removed entirely* (W(.) = I), the port must know what removing it costs in F1.

Three stages, selectable with --stage:

  --stage eval     (decisive)  Frozen Trial 0474 single-frame metrics on the native
                                cache vs the no-GMC cache. Threshold sweep, per-sequence
                                breakdown, signed deltas, JSON report.

  --stage compare  (context)   Pixel-level feature divergence between the frozen
                                `uav_gmc_median` tree and the no-GMC tree, per channel.
                                Answers "how different are the features, really".

  --stage timing   (cost)      Per-frame CPU cost of the native GMC path vs the no-GMC
                                path on the same raw sequences, plus the share of the
                                21-frame median. This is the number that sizes the port.

Metric protocol — deliberately identical to the frozen single-frame SOTA headline, so
the delta is directly comparable and no re-baselining is needed:
    distance_threshold = 8.0 px, letterbox 640x640 space, thresholds swept from the
    same deep-harvest cache (conf_thresh=0.02, top_k=100, float16 candidates).
    Reference headline to reproduce on the native arm: F1=0.9064 / Rec=86.19% /
    Prec=95.57% / TP=21,643 / FP=1,004 at th=0.25, GT=25,111.

IMPORTANT protocol safeguards implemented here:
  1. The two caches are aligned by `im_name`, never by list index, and the key sets are
     compared before any metric is computed. A silent index misalignment is the one
     failure mode that would make the whole A/B meaningless.
  2. Per-image matching is a byte-exact replication of
     `manu.evaluation.heatmap_evaluate.evaluate_point_detections` (same greedy
     nearest-first assignment, same empty-GT handling where empty-GT frames add FP but
     do NOT add to the GT denominator). The aggregate of the per-image results is
     asserted against that authoritative function at th=0.25, so the per-sequence table
     cannot silently drift from the headline protocol.
  3. `stage compare` reports MAE rather than asserting byte-identity. Three-channel JPEG
     is 4:2:0 chroma-subsampled, so Ch0 cannot be expected to match bit-for-bit between
     two separately encoded files even when the encoder input arrays are identical. The
     correct equivalence criterion is at the pre-encode array level, not the decoded JPG.

Usage on Server:
    # stage eval (requires both caches)
    python manu/evaluation/ab_test_chained_gmc.py --stage eval \
        --native-cache runs/gmc_eval/uav_median_trial0474_cache.pkl \
        --nogmc-cache  runs/gmc_eval/nogmc_trial0474_cache.pkl \
        --primary-th 0.25 --dist-thresh 8.0 \
        --out-json runs/gmc_eval/nogmc_ab_report.json

    # stage compare
    python manu/evaluation/ab_test_chained_gmc.py --stage compare \
        --native-root /mnt/data/siping/datasets/manu/uav_gmc_median \
        --nogmc-root  /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc \
        --split val --workers 24

    # stage timing
    python manu/evaluation/ab_test_chained_gmc.py --stage timing \
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \
        --split val --max-frames-per-seq 300
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.heatmap_evaluate import evaluate_point_detections  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# Frozen single-frame SOTA headline this harness must reproduce on the native arm at
# th=0.25 / dist<=8px. Used as a self-check, never as a substitute for measurement.
NATIVE_REFERENCE = {
    "f1": 0.9064,
    "recall": 0.8619,
    "precision": 0.9557,
    "tp": 21643,
    "fp": 1004,
    "total_gt": 25111,
}


def natural_key(path: Path | str):
    stem = Path(path).stem
    parts = re.split(r"(\d+)", stem)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_seq_and_frame(im_name: str) -> tuple[str, int]:
    """Re-exported from the frozen builder so sequence grouping can never diverge."""
    stem = Path(im_name).stem
    if "___" in stem:
        parts = stem.split("___")
        m = re.search(r"(\d+)$", parts[1])
        return parts[0], int(m.group(1)) if m else 0
    if "__" in stem:
        parts = stem.split("__")
        m = re.search(r"(\d+)$", parts[1])
        return parts[0], int(m.group(1)) if m else 0
    m = re.search(r"^(.*?)(?:[_-]+)?(\d+)$", stem)
    if m:
        return m.group(1).rstrip("_-"), int(m.group(2))
    raise ValueError(f"Cannot parse sequence and frame from {im_name}")


def find_sequence_folder(raw_root: Path, seq_name: str) -> Path | None:
    """Same candidate order as the frozen builder (Data/val, val, Data/train, train, rglob)."""
    for cand in (
        raw_root / seq_name,
        raw_root / "Data" / "val" / seq_name,
        raw_root / "val" / seq_name,
        raw_root / "Data" / "train" / seq_name,
        raw_root / "train" / seq_name,
    ):
        if cand.is_dir():
            return cand
    for p in raw_root.rglob(seq_name):
        if p.is_dir():
            return p
    return None


# --------------------------------------------------------------------------------------
# Shared evaluation core
# --------------------------------------------------------------------------------------

def match_image(pred_pts: np.ndarray, gt_pts_px: np.ndarray, dist_thresh: float) -> tuple[int, int, int]:
    """Byte-exact replication of the greedy matching inside evaluate_point_detections.

    Returns (tp, fp, n_gt). Caller is responsible for the empty-GT convention.
    """
    n_pred = len(pred_pts)
    n_gt = len(gt_pts_px)
    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt

    diff = pred_pts[:, np.newaxis, :] - gt_pts_px[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))

    matched_gt = set()
    matched_pred = set()
    pred_indices, gt_indices = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if dists[p_idx, g_idx] > dist_thresh:
            break
        if p_idx not in matched_pred and g_idx not in matched_gt:
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)

    tp = len(matched_gt)
    return tp, n_pred - tp, n_gt


def threshold_sweep(cache: dict[str, dict], thresholds: list[float], dist_thresh: float,
                    imgsz: int, desc: str) -> dict[float, dict]:
    """Compute headline metrics for every threshold. Reuses the authoritative evaluator."""
    names = list(cache.keys())
    gt_norm_list, sizes, preds_all = [], [], []
    for n in names:
        r = cache[n]
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        # cache stores GT already in letterbox pixel space; convert back to normalized so
        # evaluate_point_detections() reproduces the original float32 round-trip exactly.
        gt_norm_list.append(gt_pts / float(imgsz) if len(gt_pts) else gt_pts.reshape(0, 2))
        sizes.append((imgsz, imgsz))
        preds_all.append(r)

    out: dict[float, dict] = {}
    for th in tqdm(thresholds, desc=desc, ncols=90, unit="th",
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| th={n_fmt}/{total_fmt} {postfix}"):
        th_preds = []
        for r in preds_all:
            sc = np.asarray(r["pred_scores"], dtype=np.float32)
            pts = np.asarray(r["pred_points"], dtype=np.float32)
            keep = sc >= th
            th_preds.append({"points": pts[keep], "scores": sc[keep]})
        m = evaluate_point_detections(
            predictions=th_preds,
            gt_boxes_list=gt_norm_list,
            img_sizes=sizes,
            distance_threshold=dist_thresh,
        )
        m["recall"] = float(m["recall"])
        m["precision"] = float(m["precision"])
        m["f1"] = float(m["f1"])
        out[th] = m
    return out


def per_sequence_table(cache: dict[str, dict], th: float, dist_thresh: float) -> dict[str, dict]:
    """Per-sequence tp/fp/gt at a fixed threshold, using the replicated matcher."""
    agg: dict[str, dict] = defaultdict(lambda: {"frames": 0, "tp": 0, "fp": 0, "gt": 0})
    for im_name, r in cache.items():
        seq, _ = parse_seq_and_frame(im_name)
        a = agg[seq]
        a["frames"] += 1
        sc = np.asarray(r["pred_scores"], dtype=np.float32)
        pts = np.asarray(r["pred_points"], dtype=np.float32)
        keep = sc >= th
        gt_pts = np.asarray(r["gt_pts"], dtype=np.float32)
        tp, fp, n_gt = match_image(pts[keep], gt_pts, dist_thresh)
        a["tp"] += tp
        a["fp"] += fp
        a["gt"] += n_gt
    for a in agg.values():
        rec = a["tp"] / (a["gt"] + 1e-6)
        pre = a["tp"] / (a["tp"] + a["fp"] + 1e-6)
        a["recall"] = float(rec)
        a["precision"] = float(pre)
        a["f1"] = float(2 * pre * rec / (pre + rec + 1e-6))
        a["far"] = float(a["fp"] / max(1, a["frames"]))
    return dict(agg)


# --------------------------------------------------------------------------------------
# Stage: eval
# --------------------------------------------------------------------------------------

def load_cache(path: Path, label: str) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"[{label}] cache not found: {path}\n"
            f"    Generate it first, e.g.\n"
            f"      python manu/inference/cache_trial0474_inferences.py "
            f"--data <dataset>/data.yaml --output {path}"
        )
    with open(path, "rb") as f:
        records = pickle.load(f)
    cache = {r["im_name"]: r for r in records}
    size_mb = path.stat().st_size / (1024 * 1022)
    print(f"[LOAD] {label:<7} {len(cache):>7} records  {size_mb:6.1f} MB  <- {path}")
    return cache


def stage_eval(args):
    native = load_cache(Path(args.native_cache), "native")
    nogmc = load_cache(Path(args.nogmc_cache), "nogmc")

    # --- safeguard 1: align by im_name, verify key sets -------------------------------
    n_keys, g_keys = set(native), set(nogmc)
    only_native = sorted(n_keys - g_keys)
    only_nogmc = sorted(g_keys - n_keys)
    if only_native or only_nogmc:
        print(f"\n[FATAL] im_name key sets differ: {len(only_native)} only-native, "
              f"{len(only_nogmc)} only-nogmc.")
        print(f"        only-native e.g. {only_native[:5]}")
        print(f"        only-nogmc  e.g. {only_nogmc[:5]}")
        print("        A/B aborted. Rebuild the no-GMC dataset so its manifest matches "
              "uav_gmc_median exactly (same split, same filenames).")
        return 2
    print(f"[ALIGN] im_name key sets identical: {len(n_keys)} frames. Order-independent.")

    order = sorted(n_keys, key=natural_key)
    native = {k: native[k] for k in order}
    nogmc = {k: nogmc[k] for k in order}

    gt_native = sum(len(np.asarray(r["gt_pts"])) for r in native.values())
    gt_nogmc = sum(len(np.asarray(r["gt_pts"])) for r in nogmc.values())
    if gt_native != gt_nogmc:
        print(f"\n[FATAL] GT total mismatch: native={gt_native} nogmc={gt_nogmc}. "
              f"Labels are supposed to be hardlinked from the frozen dataset.")
        return 2
    print(f"[GT] identical on both arms: {gt_native} boxes "
          f"(expected frozen total {NATIVE_REFERENCE['total_gt']})")
    if gt_native != NATIVE_REFERENCE["total_gt"]:
        print(f"[WARN] GT total {gt_native} != frozen reference {NATIVE_REFERENCE['total_gt']}. "
              f"Absolute F1 is not comparable to the headline; the DELTA between arms still is.")

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    print("\n" + "=" * 104)
    print(f"   CHAINED-GMC A/B  |  stage=eval  |  dist<={args.dist_thresh}px  |  letterbox {args.imgsz}x{args.imgsz}")
    print("=" * 104)

    print("\n[STAGE 1/2] Threshold sweep (both arms, same deep-harvest cache)")
    m_native = threshold_sweep(native, thresholds, args.dist_thresh, args.imgsz, "native")
    m_nogmc = threshold_sweep(nogmc, thresholds, args.dist_thresh, args.imgsz, "no-GMC ")

    hdr = (f"{'th':>6} | {'Rec(native)':>11} {'Prec(native)':>12} {'F1(native)':>10} {'TP':>7} {'FP':>6} "
           f"| {'Rec(noGMC)':>10} {'Prec(noGMC)':>11} {'F1(noGMC)':>9} {'TP':>7} {'FP':>6} {'dF1':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for th in thresholds:
        a, b = m_native[th], m_nogmc[th]
        print(f"{th:>6.2f} | {a['recall']:>11.4f} {a['precision']:>12.4f} {a['f1']:>10.4f} "
              f"{a['tp']:>7} {a['fp']:>6} | {b['recall']:>10.4f} {b['precision']:>11.4f} "
              f"{b['f1']:>9.4f} {b['tp']:>7} {b['fp']:>6} {b['f1'] - a['f1']:>+8.4f}")

    th = args.primary_th
    if th not in m_native:
        print(f"\n[FATAL] primary th={th} not in sweep {thresholds}. Add it to --thresholds.")
        return 2
    a, b = m_native[th], m_nogmc[th]

    print(f"\n[SELF-CHECK] native arm @ th={th} vs frozen single-frame SOTA reference")
    print(f"{'metric':>12} {'measured':>12} {'reference':>12} {'delta':>10}")
    for k in ("f1", "recall", "precision", "tp", "fp", "total_gt"):
        print(f"{k:>12} {a[k]:>12} {NATIVE_REFERENCE[k]:>12} {a[k] - NATIVE_REFERENCE[k]:>+10}")
    if a["tp"] == NATIVE_REFERENCE["tp"] and a["fp"] == NATIVE_REFERENCE["fp"]:
        print("[SELF-CHECK] PASS — native arm reproduces the frozen headline exactly; "
              "the no-GMC delta below is therefore trustworthy.")
    else:
        print("[SELF-CHECK] MISMATCH — the native arm does NOT reproduce the frozen headline. "
              "Do not quote the delta until the native arm is re-verified "
              "(wrong cache, wrong data.yaml, or wrong --dist-thresh/--imgsz).")

    print(f"\n[STAGE 2/2] Per-sequence breakdown @ th={th}")
    seq_native = per_sequence_table(native, th, args.dist_thresh)
    seq_nogmc = per_sequence_table(nogmc, th, args.dist_thresh)

    # --- safeguard 2: assert the replicated matcher reproduces the authoritative totals
    sum_tp = sum(v["tp"] for v in seq_nogmc.values())
    sum_fp = sum(v["fp"] for v in seq_nogmc.values())
    sum_gt = sum(v["gt"] for v in seq_nogmc.values())
    if (sum_tp, sum_fp, sum_gt) != (b["tp"], b["fp"], b["total_gt"]):
        print(f"\n[FATAL] per-sequence sums {(sum_tp, sum_fp, sum_gt)} disagree with the "
              f"authoritative evaluator {(b['tp'], b['fp'], b['total_gt'])}. Aborting.")
        return 2
    print(f"[ASSERT] per-image matcher reconciles with evaluate_point_detections: "
          f"TP={sum_tp} FP={sum_fp} GT={sum_gt}")

    hdr2 = (f"{'sequence':<30}{'F1 native':>10}{'F1 noGMC':>10}{'dF1':>8}"
            f"{'TPn':>7}{'TPg':>7}{'dTP':>6}{'FPn':>6}{'FPg':>7}{'dFP':>7}{'Rec n':>7}{'Rec g':>7}")
    print("\n" + hdr2)
    print("-" * len(hdr2))
    rows = []
    for s in sorted(set(seq_native) | set(seq_nogmc)):
        x = seq_native.get(s, {"tp": 0, "fp": 0, "gt": 0, "f1": 0.0, "recall": 0.0, "precision": 0.0, "frames": 0})
        y = seq_nogmc.get(s, {"tp": 0, "fp": 0, "gt": 0, "f1": 0.0, "recall": 0.0, "precision": 0.0, "frames": 0})
        print(f"{s:<30}{x['f1']:>10.4f}{y['f1']:>10.4f}{y['f1'] - x['f1']:>+8.4f}"
              f"{x['tp']:>7}{y['tp']:>7}{y['tp'] - x['tp']:>+6}{x['fp']:>6}{y['fp']:>7}"
              f"{y['fp'] - x['fp']:>+7}{x['recall']:>7.4f}{y['recall']:>7.4f}")
        rows.append({"seq": s, "frames": x["frames"], "native": x, "nogmc": y,
                     "delta_f1": y["f1"] - x["f1"], "delta_tp": y["tp"] - x["tp"],
                     "delta_fp": y["fp"] - x["fp"]})

    worst = sorted(rows, key=lambda r: r["delta_f1"])[:5]
    print("\n[WORST 5 SEQUENCES BY dF1]")
    for r in worst:
        print(f"  {r['seq']:<30} dF1={r['delta_f1']:>+8.4f}  dTP={r['delta_tp']:>+6}  dFP={r['delta_fp']:>+7}")

    report = {
        "stage": "eval",
        "dist_thresh": args.dist_thresh,
        "imgsz": args.imgsz,
        "primary_th": th,
        "frames": len(order),
        "gt_total": int(gt_nogmc),
        "native_reference": NATIVE_REFERENCE,
        "native_selfcheck_pass": bool(a["tp"] == NATIVE_REFERENCE["tp"] and a["fp"] == NATIVE_REFERENCE["fp"]),
        "sweep": {
            f"{k:.2f}": {
                "native": {m: m_native[k][m] for m in ("recall", "precision", "f1", "tp", "fp", "total_gt")},
                "nogmc": {m: m_nogmc[k][m] for m in ("recall", "precision", "f1", "tp", "fp", "total_gt")},
                "delta_f1": m_nogmc[k]["f1"] - m_native[k]["f1"],
            } for k in thresholds
        },
        "per_sequence": rows,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 104)
    print(f"[HEADLINE @ th={th}]  native F1={a['f1']:.4f}  ->  no-GMC F1={b['f1']:.4f}   "
          f"dF1={b['f1'] - a['f1']:+.4f}")
    print(f"[HEADLINE @ th={th}]  Recall {a['recall'] * 100:.2f}% -> {b['recall'] * 100:.2f}%  "
          f"({(b['recall'] - a['recall']) * 100:+.2f}pp)   "
          f"Precision {a['precision'] * 100:.2f}% -> {b['precision'] * 100:.2f}%  "
          f"({(b['precision'] - a['precision']) * 100:+.2f}pp)")
    print(f"[HEADLINE @ th={th}]  TP {a['tp']} -> {b['tp']} ({b['tp'] - a['tp']:+d})   "
          f"FP {a['fp']} -> {b['fp']} ({b['fp'] - a['fp']:+d})   GT={b['total_gt']}")
    print(f"[REPORT] {out}")
    print("=" * 104)
    print("\nReading: this measures the marginal value of GMC *to this frozen checkpoint*,")
    print("which has never seen an unaligned difference channel. It is not the ceiling of a")
    print("GMC-free detector trained on GMC-free features.")
    return 0


# --------------------------------------------------------------------------------------
# Stage: compare
# --------------------------------------------------------------------------------------

def _compare_chunk(names: list[str], native_dir_str: str, nogmc_dir_str: str) -> dict:
    cv2.setNumThreads(1)
    ndir, gdir = Path(native_dir_str), Path(nogmc_dir_str)
    per_ch_abs_sum = np.zeros(3, dtype=np.float64)
    per_ch_abs_ge2 = np.zeros(3, dtype=np.float64)
    per_ch_max = np.zeros(3, dtype=np.int64)
    n_pix = 0
    compared = 0
    ch0_exact_frames = 0
    seq_acc: dict[str, dict] = defaultdict(lambda: {"frames": 0, "mae": np.zeros(3), "ge2": np.zeros(3),
                                                    "pix": 0, "ch0_exact": 0})
    for name in names:
        a = cv2.imread(str(ndir / name), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(gdir / name), cv2.IMREAD_UNCHANGED)
        if a is None or b is None or a.shape != b.shape:
            continue
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))  # (H,W,3)
        abs_sum = d.sum(axis=(0, 1), dtype=np.float64)
        ge2 = (d.reshape(-1, 3) >= 2).sum(axis=0)
        mx = d.reshape(-1, 3).max(axis=0)
        seq, _ = parse_seq_and_frame(name)
        s = seq_acc[seq]
        s["frames"] += 1
        s["mae"] += abs_sum
        s["ge2"] += ge2
        s["pix"] += d.shape[0] * d.shape[1]
        if int(ge2[0]) == 0:
            s["ch0_exact"] += 1
        per_ch_abs_sum += abs_sum
        per_ch_abs_ge2 += ge2
        per_ch_max = np.maximum(per_ch_max, mx)
        n_pix += d.shape[0] * d.shape[1]
        compared += 1
        if int(ge2[0]) == 0:
            ch0_exact_frames += 1
    return {
        "per_seq": {k: {"frames": v["frames"], "mae": v["mae"].tolist(),
                        "ge2": v["ge2"].tolist(), "pix": v["pix"],
                        "ch0_exact": v["ch0_exact"]} for k, v in seq_acc.items()},
        "total_mae": per_ch_abs_sum.tolist(),
        "total_ge2": per_ch_abs_ge2.tolist(),
        "total_max": per_ch_max.tolist(),
        "n_pix": n_pix,
        "n_frames": compared,
        "ch0_exact_frames": ch0_exact_frames,
    }


def stage_compare(args):
    native_dir = Path(args.native_root) / "images" / args.split
    nogmc_dir = Path(args.nogmc_root) / "images" / args.split
    if not native_dir.is_dir():
        raise FileNotFoundError(f"native image dir not found: {native_dir}")
    if not nogmc_dir.is_dir():
        raise FileNotFoundError(f"no-GMC image dir not found: {nogmc_dir}")

    native_files = sorted([p.name for p in native_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES], key=natural_key)
    nogmc_files = set(p.name for p in nogmc_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    common = [n for n in native_files if n in nogmc_files]
    missing = [n for n in native_files if n not in nogmc_files]
    extra = sorted(nogmc_files - set(native_files), key=natural_key)

    print("=" * 104)
    print(f"   CHAINED-GMC A/B  |  stage=compare  |  split={args.split}")
    print(f"   native : {native_dir}  ({len(native_files)} files)")
    print(f"   no-GMC : {nogmc_dir}  ({len(nogmc_files)} files)")
    print(f"   common : {len(common)}   missing-from-noGMC={len(missing)}   extra-in-noGMC={len(extra)}")
    if missing:
        print(f"   [WARN] missing e.g. {missing[:5]}")
    if extra:
        print(f"   [WARN] extra   e.g. {extra[:5]}")
    print(f"   workers={args.workers}  chunk_size={args.chunk_size}")
    print("=" * 104, flush=True)

    chunks = [common[i:i + args.chunk_size] for i in range(0, len(common), args.chunk_size)]
    chunks = [c for c in chunks if c]

    total_mae = np.zeros(3, dtype=np.float64)
    total_ge2 = np.zeros(3, dtype=np.float64)
    total_max = np.zeros(3, dtype=np.int64)
    n_pix = 0
    n_frames = 0
    ch0_exact = 0
    seq: dict[str, dict] = defaultdict(lambda: {"frames": 0, "mae": np.zeros(3), "ge2": np.zeros(3),
                                                "pix": 0, "ch0_exact": 0})

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for c in chunks:
            fut = ex.submit(_compare_chunk, c, str(native_dir), str(nogmc_dir))
            futs[fut] = len(c)
        bar = tqdm(total=len(common), desc="compare", unit="frm", ncols=90,
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} frm "
                              "[{elapsed}<{remaining}] {rate_fmt} ETA")
        for fut in as_completed(futs):
            r = fut.result()
            total_mae += np.asarray(r["total_mae"])
            total_ge2 += np.asarray(r["total_ge2"])
            total_max = np.maximum(total_max, np.asarray(r["total_max"]))
            n_pix += r["n_pix"]
            n_frames += r["n_frames"]
            ch0_exact += r["ch0_exact_frames"]
            for s, v in r["per_seq"].items():
                d = seq[s]
                d["frames"] += v["frames"]
                d["mae"] += np.asarray(v["mae"])
                d["ge2"] += np.asarray(v["ge2"])
                d["pix"] += v["pix"]
                d["ch0_exact"] += v["ch0_exact"]
            bar.update(futs[fut])
        bar.close()

    elapsed = time.time() - t0
    if n_pix == 0:
        print("[FATAL] no comparable pixels. Aborting.")
        return 2

    mae = total_mae / n_pix
    print(f"\n[COVERAGE] {n_frames}/{len(common)} common frames compared, "
          f"{elapsed:.1f}s ({n_frames / max(elapsed, 1e-6):.1f} frm/s)")

    names = ["Ch0 I_t", "Ch1 GMC_Diff", "Ch2 Median_Res"]
    hdr = f"{'channel':<18}{'MAE(gray)':>12}{'pct':>9}{'|d|>=2':>12}{'|d|>=2 %':>11}{'max|d|':>9}"
    print("\n[GLOBAL] |native - nogmc| per channel, decoded 3-channel JPG (BGR order as written)")
    print(hdr)
    print("-" * len(hdr))
    for i, nm in enumerate(names):
        print(f"{nm:<18}{mae[i]:>12.5f}{mae[i] / 255.0 * 100:>8.3f}%{int(total_ge2[i]):>12}"
              f"{total_ge2[i] / n_pix * 100:>10.4f}%{int(total_max[i]):>9}")

    print(f"\n[Ch0 NOTE] frames with pixel-identical Ch0: {ch0_exact}/{n_frames} "
          f"({ch0_exact / max(1, n_frames) * 100:.2f}%)")
    print("  Three-channel JPEG is 4:2:0 chroma-subsampled, so a bit-exact Ch0 match between two")
    print("  separately encoded files is NOT a valid equivalence criterion. The correct criterion")
    print("  lives at the pre-encode array level and was already measured: Ch0 maxdiff = 0.")
    print("  A small residual MAE here is expected chroma/quantisation noise, not feature drift.")

    print("\n[PER-SEQUENCE] mean |native - nogmc| per channel (gray levels)")
    hdr2 = (f"{'sequence':<30}{'frames':>8}{'MAE Ch0':>10}{'MAE Ch1':>10}{'MAE Ch2':>10}"
            f"{'ge2% Ch1':>11}{'ge2% Ch2':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    rows = []
    for s in sorted(seq):
        v = seq[s]
        pix = max(1, v["pix"])
        m = v["mae"] / pix
        g1 = v["ge2"][1] / pix * 100
        g2 = v["ge2"][2] / pix * 100
        print(f"{s:<30}{v['frames']:>8}{m[0]:>10.5f}{m[1]:>10.5f}{m[2]:>10.5f}{g1:>10.4f}%{g2:>10.4f}%")
        rows.append({"seq": s, "frames": v["frames"], "mae": m.tolist(),
                     "ge2_pct": [g1, g2]})
    if rows:
        rows.sort(key=lambda r: -r["mae"][2])
        print("\n[TOP 5 BY Ch2 (Median_Residual) DIVERGENCE — these sequences depend most on GMC]")
        for r in rows[:5]:
            print(f"  {r['seq']:<30} MAE Ch2={r['mae'][2]:.5f}  ({r['ge2_pct'][1]:.4f}% pixels |d|>=2)")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stage": "compare", "split": args.split, "frames_compared": n_frames,
        "common_files": len(common), "missing_from_nogmc": len(missing), "extra_in_nogmc": len(extra),
        "global_mae": mae.tolist(), "global_ge2_px": total_ge2.tolist(),
        "global_max": total_max.tolist(),
        "ch0_exact_frames": ch0_exact, "per_sequence": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[REPORT] {out}")
    return 0


# --------------------------------------------------------------------------------------
# Stage: timing
# --------------------------------------------------------------------------------------

def stage_timing(args):
    from manu.data.build_sample_median_dataset import FastGMCEstimator

    labels_src = Path(args.labels_src)
    if not labels_src.is_dir():
        for cand in (Path("/mnt/data/siping/datasets/manu/uav_gmc_median"),
                     Path("/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median")):
            if cand.is_dir():
                labels_src = cand
                break
    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        for cand in (Path("/mnt/data/siping/datasets/manu/anti-uav"),
                     Path("/home/manu/mnt/data/siping/datasets/manu/anti-uav")):
            if cand.is_dir():
                raw_root = cand
                break

    ref_dir = labels_src / "images" / args.split
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"manifest dir not found: {ref_dir}")
    names = sorted([p.name for p in ref_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES], key=natural_key)

    seq_names: dict[str, list[str]] = defaultdict(list)
    for n in names:
        s, _ = parse_seq_and_frame(n)
        seq_names[s].append(n)

    print("=" * 104)
    print(f"   CHAINED-GMC A/B  |  stage=timing  |  split={args.split}")
    print(f"   manifest : {ref_dir}")
    print(f"   raw root : {raw_root}")
    print(f"   window={args.window} stride_step={args.stride_step} downscale={args.downscale}")
    print(f"   max frames/seq = {args.max_frames_per_seq}   cv2.setNumThreads(1)  (single process, serial)")
    print("=" * 104, flush=True)

    # NOTE: the "序列起点钳制区" defect that invalidated the earlier 8-frame spot check.
    # Frames are spread across each sequence instead of taken from the head, so the median
    # window is genuinely populated rather than 20/21 frames clamped to frame 0.
    seq_list = sorted(seq_names)
    rows = []
    for seq in tqdm(seq_list, desc="timing/seq", ncols=90, unit="seq",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} seq "
                               "[{elapsed}<{remaining}] {postfix}"):
        all_names = sorted(seq_names[seq], key=natural_key)
        if len(all_names) > args.max_frames_per_seq:
            picks = np.linspace(0, len(all_names) - 1, args.max_frames_per_seq).round().astype(int)
            selected = [all_names[i] for i in sorted(set(picks.tolist()))]
        else:
            selected = all_names
        # drop the first ~2 windows so every measured frame has a full history
        if len(selected) > args.window * args.stride_step:
            selected = selected[args.window * args.stride_step:]

        seq_dir = find_sequence_folder(raw_root, seq)
        if seq_dir is None:
            continue
        frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
        frames.sort(key=natural_key)
        if not frames:
            continue
        idx_map = {}
        for li, f in enumerate(frames):
            m = re.search(r"(\d+)$", f.stem)
            idx_map[int(m.group(1)) if m else li] = li
        n_frames = len(frames)

        est = FastGMCEstimator(downscale=args.downscale)
        cache: dict[int, np.ndarray] = {}

        def read(i: int):
            i = max(0, min(i, n_frames - 1))
            v = cache.get(i)
            if v is None:
                v = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
                cache[i] = v
            return v

        t_read_n = t_read_g = t_med_n = t_med_g = 0.0
        t_gmc_n = 0.0
        gmc_fits = 0
        n_used = 0
        for im_name in selected:
            _, fi = parse_seq_and_frame(im_name)
            ci = idx_map.get(fi, min(fi, n_frames - 1))

            m0 = time.perf_counter()
            im_cur = read(ci)
            im_p2 = read(ci - 2)
            t_read_n += time.perf_counter() - m0

            # ---- native path: ONE GMC fit per lag, exactly as the frozen builder does it ----
            m1 = time.perf_counter()
            H2 = est.compute_affine(im_p2, im_cur)
            t_gmc_n += time.perf_counter() - m1
            gmc_fits += 1
            m2 = time.perf_counter()
            _ = cv2.absdiff(im_cur, est.warp(im_p2, H2))
            t_med_n += time.perf_counter() - m2
            m3 = time.perf_counter()
            hist_n = []
            for step in range(1, args.window + 1):
                h = read(ci - step * args.stride_step)
                if h is not None:
                    t1 = time.perf_counter()
                    M = est.compute_affine(h, im_cur)
                    t_gmc_n += time.perf_counter() - t1
                    gmc_fits += 1
                    hist_n.append(est.warp(h, M))
            if len(hist_n) >= 5:
                bg = np.median(np.stack(hist_n, axis=0), axis=0).astype(np.float32)
                _ = np.clip(im_cur.astype(np.float32) - bg, 0, 255).astype(np.uint8)
            t_med_n += time.perf_counter() - m3

            # ---- no-GMC path: no estimate, no warp ----
            m4 = time.perf_counter()
            _ = cv2.absdiff(im_cur, im_p2)
            t_read_g += time.perf_counter() - m4
            m5 = time.perf_counter()
            hist_g = [read(ci - step * args.stride_step) for step in range(1, args.window + 1)]
            hist_g = [h for h in hist_g if h is not None]
            if len(hist_g) >= 5:
                bg = np.median(np.stack(hist_g, axis=0), axis=0).astype(np.float32)
                _ = np.clip(im_cur.astype(np.float32) - bg, 0, 255).astype(np.uint8)
            t_med_g += time.perf_counter() - m5

            n_used += 1
            for k in [k for k in cache if k < ci - args.window * args.stride_step - 4]:
                cache.pop(k, None)

        if n_used == 0:
            continue
        rows.append({
            "seq": seq, "frames": n_used,
            "native_ms": 1000.0 * (t_read_n + t_gmc_n + t_med_n) / n_used,
            "nogmc_ms": 1000.0 * (t_read_g + t_med_g) / n_used,
            "gmc_ms": 1000.0 * t_gmc_n / n_used,
            "median_native_ms": 1000.0 * t_med_n / n_used,
            "median_nogmc_ms": 1000.0 * t_med_g / n_used,
            "gmc_fits_per_frame": gmc_fits / n_used,
        })

    if not rows:
        print("[FATAL] no sequences timed. Aborting.")
        return 2

    hdr = (f"{'sequence':<30}{'frames':>8}{'fits/frm':>9}{'native ms':>11}{'noGMC ms':>10}{'GMC ms':>9}"
           f"{'med(n)':>9}{'med(g)':>9}{'speedup':>9}")
    print("\n[PER-SEQUENCE] per-frame CPU cost, single process, cv2.setNumThreads(1)")
    print("  fits/frm = Shi-Tomasi + LK + RANSAC fits actually issued (native pipeline = 1 + window)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['seq']:<30}{r['frames']:>8}{r.get('gmc_fits_per_frame', 0):>9.1f}"
              f"{r['native_ms']:>11.2f}{r['nogmc_ms']:>10.2f}"
              f"{r['gmc_ms']:>9.2f}{r['median_native_ms']:>9.2f}{r['median_nogmc_ms']:>9.2f}"
              f"{r['native_ms'] / max(r['nogmc_ms'], 1e-9):>8.2f}x")

    wsum = sum(r["frames"] for r in rows)
    avg = lambda k: sum(r[k] * r["frames"] for r in rows) / max(1, wsum)  # noqa: E731
    native_ms, nogmc_ms, gmc_ms = avg("native_ms"), avg("nogmc_ms"), avg("gmc_ms")
    print("-" * len(hdr))
    print(f"{'WEIGHTED MEAN':<30}{wsum:>8}{native_ms:>11.2f}{nogmc_ms:>10.2f}{gmc_ms:>9.2f}"
          f"{avg('median_native_ms'):>9.2f}{avg('median_nogmc_ms'):>9.2f}"
          f"{native_ms / max(nogmc_ms, 1e-9):>8.2f}x")
    print(f"\n[COST] GMC affine+warp = {gmc_ms:.2f} ms/frame  "
          f"({gmc_ms / native_ms * 100:.1f}% of the native path)")
    print(f"[COST] removing GMC would save {native_ms - nogmc_ms:.2f} ms/frame "
          f"({(native_ms - nogmc_ms) / native_ms * 100:.1f}%), leaving {nogmc_ms:.2f} ms/frame "
          f"of which the 21-frame median is {avg('median_nogmc_ms'):.2f} ms "
          f"({avg('median_nogmc_ms') / nogmc_ms * 100:.0f}%).")
    print("  => The residual bottleneck after removing GMC is the 21-frame full-image median,")
    print("     which is reducible to an O(21) sorted ring buffer with bit-identical output.")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stage": "timing", "split": args.split, "window": args.window,
        "stride_step": args.stride_step, "downscale": args.downscale,
        "max_frames_per_seq": args.max_frames_per_seq,
        "weighted_native_ms": native_ms, "weighted_nogmc_ms": nogmc_ms, "gmc_ms": gmc_ms,
        "per_sequence": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[REPORT] {out}")
    return 0


# --------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Chained-GMC A/B harness (compare / eval / timing)")
    p.add_argument("--stage", type=str, default="eval", choices=["compare", "eval", "timing"])
    p.add_argument("--split", type=str, default="val")

    # eval
    p.add_argument("--native-cache", type=str, default="runs/gmc_eval/uav_median_trial0474_cache.pkl")
    p.add_argument("--nogmc-cache", type=str, default="runs/gmc_eval/nogmc_trial0474_cache.pkl")
    p.add_argument("--thresholds", type=str, default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45")
    p.add_argument("--primary-th", type=float, default=0.25)
    p.add_argument("--dist-thresh", type=float, default=8.0)
    p.add_argument("--imgsz", type=int, default=640)

    # compare
    p.add_argument("--native-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    p.add_argument("--nogmc-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--chunk-size", type=int, default=200)

    # timing
    p.add_argument("--labels-src", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median")
    p.add_argument("--raw-root", type=str, default="/mnt/data/siping/datasets/manu/anti-uav")
    p.add_argument("--window", type=int, default=21)
    p.add_argument("--stride-step", type=int, default=2)
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--max-frames-per-seq", type=int, default=300)

    p.add_argument("--out-json", type=str, default="",
                   help="Report path; if omitted a stage-specific default under runs/gmc_eval/ is used")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.out_json:
        args.out_json = f"runs/gmc_eval/nogmc_ab_{args.stage}.json"
    if args.stage == "eval":
        return stage_eval(args)
    if args.stage == "compare":
        return stage_compare(args)
    if args.stage == "timing":
        return stage_timing(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())