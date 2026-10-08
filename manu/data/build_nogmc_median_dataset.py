#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
No-GMC Temporal Median Dataset Builder — embedded-port single-variable ablation.

Research question:
    Global Motion Compensation is the dominant CPU cost of the frozen input pipeline
    (22 Shi-Tomasi + Lucas-Kanade + RANSAC fits per frame, ~5-25 ms/frame on CPU).
    For an NPU/embedded port, how much single-frame F1 / Recall / Precision is lost if the
    warping operator is replaced by the identity, i.e. W(.) = I?

THE ONLY DIFFERENCE vs the frozen official dataset `uav_gmc_median`
(built by `manu/data/build_sample_median_dataset.py`):
    native : Ch1 = |I_t - W(I_{t-2})|          Ch2 = (I_t - median{W(I_{t-2k})})^+   (22 GMC fits/frame)
    no-GMC : Ch1 = |I_t - I_{t-2}|             Ch2 = (I_t - median{I_{t-2k}})^+      (0 GMC fits/frame)
    tree   : Ch1 = |I_t - W(I_{t-2})|          Ch2 = (I_t - median{W(I_{t-2k})})^+
             with W direct on the anchor grid only and chained for the rest (--mode tree)

Everything else is byte-identical code, not a re-implementation:
    filename manifest, frame-index resolution, sequence start clamping, history lags
    (2,4,...,42), median expression, uint8 rounding, `np.stack` channel order,
    `cv2.imwrite` JPEG encoding, and label files (hardlinked from the frozen dataset).

    Channel 0: I_t                    raw infrared grayscale frame
    Channel 1: |I_t - I_{t-2}|        unaligned 2-lag difference
    Channel 2: (I_t - B_t)^+          unaligned 21-frame temporal median residual
    B_t = median{ I_{t-2k} }_{k=1..21}

Interpretation caveat, stated up front:
    The frozen detector has never seen an unaligned difference channel. Any global platform
    motion now lands directly in Ch1, so a drop here measures the *marginal value of GMC to
    this particular frozen checkpoint*, not the achievable ceiling of a GMC-free detector that
    was trained on GMC-free features. A large drop is the expected outcome; the number is still
    the correct thing to report to decide whether GMC must be replaced or merely accelerated.

Modes
    --mode nogmc (default) : W(.) = IDENTITY, 0 GMC fits/frame.
    --mode tree            : bounded-depth anchored tree. Direct fits only on the anchor grid
                            {stride_step, +anchor_step, ...} U {max_lag}; every other lag is
                            R_L = p_{ci-L} . p_{ci-L+2} . ... . p_{ci-a-2} . R_a, where a is the
                            largest anchor <= L, so the composition depth is exactly
                            (L - a) / stride_step <= anchor_step/stride_step - 1.
                            With the frozen lags 2..42, anchor_step=10 gives 5 direct fits and
                            max depth 4 (76% fewer direct fits). The bounded tree was verified to be
                            BIT-IDENTICAL to the full-length chain on synthetic transforms
                            (max|diff| = 0.0), so the topology itself adds no error.

                            What this does NOT prove: that five long-baseline direct fits plus reused
                            short-baseline steps estimate the same transforms as 21 independent
                            long-baseline fits. That residual estimation bias can only be settled by
                            the end-to-end F1 delta, never by the algebra.

Usage (server):
    python manu/data/build_nogmc_median_dataset.py \
        --labels-src /mnt/data/siping/datasets/manu/uav_gmc_median \
        --raw-root   /mnt/data/siping/datasets/manu/anti-uav \
        --output     /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc \
        --split val --window 21 --stride-step 2 --workers 24

    python manu/data/build_nogmc_median_dataset.py --mode tree --anchor-step 10 \
        --labels-src /mnt/data/siping/datasets/manu/uav_gmc_median \
        --raw-root   /mnt/data/siping/datasets/manu/anti-uav \
        --output     /mnt/data/siping/datasets/manu/uav_gmc_median_tree \
        --split val --window 21 --stride-step 2 --downscale 2 --workers 24
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
import shutil
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_sample_median_dataset import (  # noqa: E402
    IMAGE_SUFFIXES,
    FastGMCEstimator,
    find_sequence_folder,
    natural_key,
    parse_seq_and_frame,
)

IDENTITY = np.eye(2, 3, dtype=np.float32)


def _to33(m: np.ndarray) -> np.ndarray:
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = m
    return h


def _compose(new: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Compose two 2x3 transforms, `new` applied after `acc`, in homogeneous form."""
    h = _to33(new) @ _to33(acc)
    n = h[2, 2]
    if not np.isfinite(n) or abs(n) < 1e-12:
        return IDENTITY.copy()
    h = h / n
    if not np.isfinite(h).all():
        return IDENTITY.copy()
    return h[:2, :].astype(np.float32)


def anchor_grid(stride_step: int, anchor_step: int, max_lag: int) -> list[int]:
    """Even lags carrying a direct GMC fit; every other lag is reached by bounded composition."""
    if anchor_step % stride_step:
        raise ValueError(f"anchor_step={anchor_step} must be a multiple of stride_step={stride_step} "
                         f"so the chain recursion lands exactly on the anchors")
    lags = sorted(set(range(stride_step, max_lag + 1, anchor_step)) | {max_lag})
    return [lag for lag in lags if lag % stride_step == 0]


def anchors_pretty(stride_step: int, anchor_step: int, max_lag: int) -> str:
    return "{" + ",".join(str(x) for x in anchor_grid(stride_step, anchor_step, max_lag)) + "}"


def parse_args():
    p = argparse.ArgumentParser(description="GMC-variant temporal median residual dataset builder")
    p.add_argument("--mode", type=str, default="nogmc", choices=["nogmc", "tree"],
                   help="nogmc: W(.) = IDENTITY, 0 fits/frame. tree: direct fits on a bounded anchor "
                        "grid, every other lag reached by chained composition")
    p.add_argument("--anchor-step", type=int, default=10,
                   help="tree only: lag spacing between direct fits. 10 -> 5 anchors {2,12,22,32,42} "
                        "for lags 2..42, max composition depth 4")
    p.add_argument("--labels-src", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median",
                   help="Frozen dataset root used ONLY as filename manifest + hardlinked labels")
    p.add_argument("--raw-root", type=str, default="/mnt/data/siping/datasets/manu/anti-uav",
                   help="Root directory containing raw sequences")
    p.add_argument("--output", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--sequences", type=str, default="",
                   help="Comma-separated subset; empty = all sequences in the manifest")
    p.add_argument("--window", type=int, default=21, help="Median window size (frozen = 21)")
    p.add_argument("--stride-step", type=int, default=2, help="Lag stride (frozen = 2 -> lags 2..42)")
    p.add_argument("--downscale", type=int, default=2, help="tree only: GMC estimation downscale (frozen = 2)")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--chunk-size", type=int, default=200,
                   help="Frames per work unit; smaller = smoother progress bar and better load balance")
    p.add_argument("--dry-run", type=int, default=0, help="If > 0 only process this many frames per sequence")
    return p.parse_args()


def process_sequence_chunk(
    seq_name: str,
    img_names: list[str],
    label_dir_str: str,
    raw_root_str: str,
    out_img_dir_str: str,
    out_lbl_dir_str: str,
    window: int,
    stride_step: int,
    dry_run: int,
    mode: str = "nogmc",
    anchor_step: int = 10,
    downscale: int = 2,
) -> dict:
    cv2.setNumThreads(1)
    raw_root = Path(raw_root_str)
    out_img_p = Path(out_img_dir_str)
    out_lbl_p = Path(out_lbl_dir_str)
    lbl_src_p = Path(label_dir_str)

    seq_dir = find_sequence_folder(raw_root, seq_name, {})
    if seq_dir is None:
        return {"seq": seq_name, "ok": 0, "fail": len(img_names), "status": "missing_seq"}

    frames = [f for f in seq_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
    frames.sort(key=natural_key)
    if not frames:
        return {"seq": seq_name, "ok": 0, "fail": len(img_names), "status": "no_raw_frames"}

    idx_map: dict[int, int] = {}
    for list_i, f in enumerate(frames):
        m = re.search(r"(\d+)$", f.stem)
        idx_map[int(m.group(1)) if m else list_i] = list_i

    n_frames = len(frames)
    max_lag = window * stride_step
    anchors = anchor_grid(stride_step, anchor_step, max_lag) if mode == "tree" else []
    est = FastGMCEstimator(downscale=downscale) if mode == "tree" else None
    frame_cache: dict[int, np.ndarray] = {}
    step_cache: dict[int, np.ndarray] = {}
    ok_cnt = fail_cnt = 0
    anchor_fits = step_fits = compose_ops = 0
    max_depth_seen = 0
    t_read = t_hist = t_write = t_gmc = 0.0
    t0 = time.time()

    def read(i: int) -> np.ndarray | None:
        i = max(0, min(i, n_frames - 1))
        v = frame_cache.get(i)
        if v is None:
            v = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
            frame_cache[i] = v
        return v

    def fit_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = est.compute_affine(a, b)
        if m is None or not np.isfinite(m).all():
            return IDENTITY.copy()
        return m.astype(np.float32)

    def fit(prev: np.ndarray | None, curr: np.ndarray | None) -> np.ndarray:
        nonlocal anchor_fits
        anchor_fits += 1
        if prev is None or curr is None:
            return IDENTITY.copy()
        return fit_matrix(prev, curr)

    def pstep(i: int) -> np.ndarray:
        """H_{i -> i+stride_step}, fitted at most once per chunk. The stride MUST match the lag grid,
        otherwise composing it onto a longer-lag matrix leaves a gap of stride_step-1 frames."""
        nonlocal step_fits
        m = step_cache.get(i)
        if m is None:
            step_fits += 1
            a, b = read(i), read(i + stride_step)
            m = IDENTITY.copy() if a is None or b is None else fit_matrix(a, b)
            step_cache[i] = m
        return m

    def tree_transforms(ci: int, curr: np.ndarray) -> dict[int, np.ndarray]:
        """Direct fit on the anchor grid; every other lag reached from the nearest anchor BELOW it by
        at most anchor_step/stride_step - 1 compositions. Deliberately NOT the full-length chain,
        which accumulates one composition per frame and defeats the whole point of the tree."""
        nonlocal compose_ops, max_depth_seen
        out: dict[int, np.ndarray] = {}
        for lag in anchors:
            out[lag] = fit(read(ci - lag), curr)
        for lag in range(stride_step, max_lag + 1, stride_step):
            if lag in out:
                continue
            below = [a for a in anchors if a <= lag]
            if not below:
                continue
            a = below[-1]
            acc = out[a]
            for j in range(ci - a - stride_step, ci - lag - 1, -stride_step):
                acc = _compose(pstep(j), acc)
                compose_ops += 1
            out[lag] = acc
        depth = {a: 0 for a in anchors}
        for lag in sorted(out):
            if lag in depth:
                continue
            a = max((x for x in anchors if x <= lag), default=None)
            if a is not None:
                depth[lag] = (lag - a) // stride_step
        if depth:
            max_depth_seen = max(max_depth_seen, max(depth.values()))
        return out

    sorted_names = sorted(img_names, key=natural_key)
    if dry_run > 0:
        sorted_names = sorted_names[:dry_run]

    for im_name in sorted_names:
        mark = time.perf_counter()
        _, frame_idx = parse_seq_and_frame(im_name)
        ci = idx_map.get(frame_idx, min(frame_idx, n_frames - 1))
        im_curr = read(ci)
        if im_curr is None:
            fail_cnt += 1
            continue
        im_prev2 = read(ci - 2)
        t_read += time.perf_counter() - mark

        mark = time.perf_counter()
        if mode == "tree":
            mats = tree_transforms(ci, im_curr)
            diff2 = cv2.absdiff(im_curr, est.warp(im_prev2, mats[stride_step]))
            history = []
            for lag in range(stride_step, max_lag + 1, stride_step):
                im_h = read(ci - lag)
                m = mats.get(lag)
                if im_h is not None and m is not None:
                    history.append(est.warp(im_h, m))
        else:
            diff2 = cv2.absdiff(im_curr, im_prev2)
            history = []
            for lag in range(stride_step, max_lag + 1, stride_step):
                im_h = read(ci - lag)
                if im_h is not None:
                    history.append(im_h)
        t_gmc += time.perf_counter() - mark

        mark = time.perf_counter()
        if len(history) >= 5:
            stack = np.stack(history, axis=0)
            median_bg = np.median(stack, axis=0).astype(np.float32)
            res_median = np.clip(im_curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
        else:
            res_median = diff2
        t_hist += time.perf_counter() - mark

        mark = time.perf_counter()
        cv2.imwrite(str(out_img_p / im_name), np.stack([im_curr, diff2, res_median], axis=-1))
        t_write += time.perf_counter() - mark

        stem = Path(im_name).stem
        dst_lbl = out_lbl_p / f"{stem}.txt"
        if not dst_lbl.exists():
            src_lbl = lbl_src_p / f"{stem}.txt"
            if src_lbl.exists():
                try:
                    src_lbl.hardlink_to(dst_lbl)
                except OSError:
                    shutil.copyfile(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("", encoding="utf-8")
        ok_cnt += 1

        for i in [k for k in frame_cache if k < ci - window * stride_step - 4]:
            frame_cache.pop(i, None)

    elapsed = time.time() - t0
    n = max(1, ok_cnt)
    return {
        "seq": seq_name,
        "ok": ok_cnt,
        "fail": fail_cnt,
        "status": "ok",
        "seconds": round(elapsed, 2),
        "gmc_calls": anchor_fits + step_fits,
        "anchor_fits_per_frame": round(anchor_fits / n, 2),
        "step_fits_per_frame": round(step_fits / n, 2),
        "compose_per_frame": round(compose_ops / n, 2),
        "max_chain_depth": max_depth_seen,
        "ms_per_frame": round(1000.0 * elapsed / n, 2),
        "ms_read": round(1000.0 * t_read / n, 2),
        "ms_gmc": round(1000.0 * t_gmc / n, 2),
        "ms_median": round(1000.0 * t_hist / n, 2),
        "ms_write": round(1000.0 * t_write / n, 2),
    }


def main():
    args = parse_args()
    labels_src = Path(args.labels_src)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.output)

    if not labels_src.is_dir():
        for cand in (Path("/mnt/data/siping/datasets/manu/uav_gmc_median"),
                     Path("/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median")):
            if cand.is_dir():
                labels_src = cand
                break
    if not raw_root.is_dir():
        for cand in (Path("/mnt/data/siping/datasets/manu/anti-uav"),
                     Path("/home/manu/mnt/data/siping/datasets/manu/anti-uav")):
            if cand.is_dir():
                raw_root = cand
                break

    ref_img_dir = labels_src / "images" / args.split
    src_lbl_dir = labels_src / "labels" / args.split
    out_img_dir = out_dir / "images" / args.split
    out_lbl_dir = out_dir / "labels" / args.split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    ref_images = [p.name for p in ref_img_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    ref_images.sort(key=natural_key)
    missing = [n for n in ref_images[:200] if not (src_lbl_dir / f"{Path(n).stem}.txt").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/200 manifest images have no label under {src_lbl_dir}, e.g. {missing[0]}. "
            f"The label directory must be <root>/labels/{args.split}, not <root>/labels."
        )

    target = [s.strip() for s in args.sequences.split(",") if s.strip()]
    seq_groups: dict[str, list[str]] = {}
    for name in ref_images:
        seq, _ = parse_seq_and_frame(name)
        if not target or any(t == seq or t in seq for t in target):
            seq_groups.setdefault(seq, []).append(name)

    total = sum(len(v) for v in seq_groups.values())

    # Disk preflight (Rule 2: always state a GiB estimate BEFORE materialising a dataset).
    # Measured from the frozen dataset itself rather than guessed, so the estimate is
    # calibrated to the exact same encoder / resolution / channel layout.
    ref_bytes = sum(p.stat().st_size for p in ref_img_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    ref_n = len(ref_images)
    per_file_kb = (ref_bytes / ref_n / 1024.0) if ref_n else 0.0
    est_gib = per_file_kb * total / (1024.0 * 1024.0)
    print("=" * 100)
    print(f"   GMC-VARIANT DATASET BUILDER  |  mode={args.mode}  (embedded-port ablation)")
    print(f"   Frozen manifest/labels : {labels_src}  ({len(ref_images)} files)")
    print(f"   Raw sequences          : {raw_root}")
    print(f"   Output                 : {out_dir}")
    print(f"   Window={args.window} stride_step={args.stride_step} lags 2..{args.window * args.stride_step}"
          + (f" | anchor lags {anchors_pretty(args.stride_step, args.anchor_step, args.window * args.stride_step)}"
             f" downscale={args.downscale}" if args.mode == "tree" else " | W(.) = IDENTITY, 0 GMC fits/frame"))
    print(f"   Sequences selected     : {len(seq_groups)} ({total} images)")
    print(f"   Workers={args.workers} chunk_size={args.chunk_size}")
    print(f"   DISK ESTIMATE          : ~{est_gib:.2f} GiB "
          f"({per_file_kb:.1f} KiB/img x {total} imgs, measured from frozen dataset); "
          f"labels are hardlinked (+0)")
    print("=" * 100, flush=True)

    chunks: list[tuple[str, list[str]]] = []
    for seq, names in seq_groups.items():
        ordered = sorted(names, key=natural_key)
        if args.dry_run > 0:
            ordered = ordered[: args.dry_run]
        for i in range(0, len(ordered), args.chunk_size):
            chunks.append((seq, ordered[i: i + args.chunk_size]))
    unit_total = sum(len(n) for _, n in chunks)

    t0 = time.time()
    results = []
    done_seqs: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for seq, names in chunks:
            fut = ex.submit(process_sequence_chunk, seq, names, str(src_lbl_dir), str(raw_root),
                            str(out_img_dir), str(out_lbl_dir), args.window, args.stride_step, 0,
                            args.mode, args.anchor_step, args.downscale)
            futs[fut] = (len(names), seq)
        bar = tqdm(total=unit_total, desc="build", unit="frm", ncols=96,
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} frm "
                              "[{elapsed}<{remaining}] {rate_fmt} seq={postfix}")
        for f in as_completed(futs):
            results.append(f.result())
            bar.update(futs[f][0])
            done_seqs.append(futs[f][1])
            bar.set_postfix(ok=sum(r["ok"] for r in results), fail=sum(r["fail"] for r in results),
                            last=done_seqs[-1], refresh=False)
        bar.set_postfix(ok=sum(r["ok"] for r in results), fail=sum(r["fail"] for r in results))
        bar.close()

    ok = sum(r["ok"] for r in results)
    fail = sum(r["fail"] for r in results)
    agg: dict[str, dict] = {}
    for r in results:
        a = agg.setdefault(r["seq"], {"ok": 0, "fail": 0, "seconds": 0.0, "read": 0.0,
                                      "gmc": 0.0, "median": 0.0, "write": 0.0,
                                      "anchor": 0.0, "step": 0.0, "compose": 0.0, "depth": 0})
        n = max(1, r["ok"])
        a["ok"] += r["ok"]
        a["fail"] += r["fail"]
        a["seconds"] += r.get("seconds", 0.0)
        a["read"] += r.get("ms_read", 0.0) * n
        a["gmc"] += r.get("ms_gmc", 0.0) * n
        a["median"] += r.get("ms_median", 0.0) * n
        a["write"] += r.get("ms_write", 0.0) * n
        a["anchor"] += r.get("anchor_fits_per_frame", 0.0) * n
        a["step"] += r.get("step_fits_per_frame", 0.0) * n
        a["compose"] += r.get("compose_per_frame", 0.0) * n
        a["depth"] = max(a["depth"], r.get("max_chain_depth", 0))
    for a in agg.values():
        n = max(1, a["ok"])
        a["ms_per_frame"] = 1000.0 * a["seconds"] / n
        for k in ("read", "gmc", "median", "write", "anchor", "step", "compose"):
            a[k] /= n
    seq_rows = []
    for seq in sorted(agg):
        a = agg[seq]
        seq_rows.append({"seq": seq, "ok": a["ok"], "fail": a["fail"], "seconds": round(a["seconds"], 2),
                         "ms_per_frame": round(a["ms_per_frame"], 2), "ms_read": round(a["read"], 2),
                         "ms_gmc": round(a["gmc"], 2), "ms_median": round(a["median"], 2),
                         "ms_write": round(a["write"], 2),
                         "anchor_fits_per_frame": round(a["anchor"], 2),
                         "step_fits_per_frame": round(a["step"], 2),
                         "compose_per_frame": round(a["compose"], 2),
                         "max_chain_depth": a["depth"]})

    print("\n[PER-SEQUENCE]")
    print(f"{'sequence':<32}{'frames':>7}{'ms/frm':>8}{'read':>7}{'gmc':>8}{'median':>8}{'write':>7}"
          f"{'anch/f':>8}{'step/f':>8}{'comp/f':>8}{'depth':>6}")
    for r in seq_rows:
        print(f"{r['seq']:<32}{r['ok']:>7}{r['ms_per_frame']:>8.2f}{r['ms_read']:>7.2f}"
              f"{r['ms_gmc']:>8.2f}{r['ms_median']:>8.2f}{r['ms_write']:>7.2f}"
              f"{r['anchor_fits_per_frame']:>8.2f}{r['step_fits_per_frame']:>8.2f}"
              f"{r['compose_per_frame']:>8.2f}{r['max_chain_depth']:>6}")

    elapsed = time.time() - t0
    ms = 1000.0 * elapsed / max(1, ok)
    wt_anchor = sum(r["anchor_fits_per_frame"] * r["ok"] for r in seq_rows) / max(1, ok)
    wt_step = sum(r["step_fits_per_frame"] * r["ok"] for r in seq_rows) / max(1, ok)
    wt_comp = sum(r["compose_per_frame"] * r["ok"] for r in seq_rows) / max(1, ok)
    print(f"\n[DONE] {ok} frames written, {fail} failed, {elapsed:.1f}s wall "
          f"({ok / max(elapsed, 1e-6):.1f} imgs/s, {ms:.2f} ms/frame with {args.workers} workers)")
    print(f"[MANIFEST] expected {total} images, wrote {ok}"
          f"{'  <== MISMATCH' if args.dry_run == 0 and ok != total else ''}")
    print(f"[COST] read {sum(r['ms_read'] * r['ok'] for r in seq_rows) / max(1, ok):.2f} | "
          f"gmc+warp {sum(r['ms_gmc'] * r['ok'] for r in seq_rows) / max(1, ok):.2f} | "
          f"median {sum(r['ms_median'] * r['ok'] for r in seq_rows) / max(1, ok):.2f} | "
          f"write {sum(r['ms_write'] * r['ok'] for r in seq_rows) / max(1, ok):.2f}  ms/frame")
    print(f"[FITS] anchor {wt_anchor:.2f} + single-step {wt_step:.2f} = {wt_anchor + wt_step:.2f} fits/frame"
          f"  (frozen native = 22.00)  |  compositions {wt_comp:.2f}/frame"
          f"  |  max chain depth {max(r['max_chain_depth'] for r in seq_rows)}")
    if args.mode == "tree":
        print(f"[TREE] anchor lags = {anchors_pretty(args.stride_step, args.anchor_step, args.window * args.stride_step)}"
              f"  (anchor_step={args.anchor_step}, stride_step={args.stride_step})")

    (out_dir / "data.yaml").write_text(
        f"""# UAV GMC-Variant Temporal Median Residual Dataset [I_t, |I_t - W(I_t-2)|, (I_t - B_t)^+]
# mode={args.mode}: nogmc = W(.) IDENTITY (0 GMC fits/frame); tree = direct fits on the bounded anchor grid
#   {anchors_pretty(args.stride_step, args.anchor_step, args.window * args.stride_step) if args.mode == 'tree' else 'n/a'}
# Window / lags / encoding identical to frozen uav_gmc_median.
# Channel order on disk matches the model input contract, because official
# Format._format_img only performs the BGR->RGB flip when channels == 3.
path: {out_dir.resolve()}
train: images/{args.split}
val: images/{args.split}

names:
  0: uav
""", encoding="utf-8")
    print(f"[SUCCESS] Dataset configuration created: {(out_dir / 'data.yaml').resolve()}")
    print("\n[STATS_JSON]")
    print(json.dumps({"mode": args.mode, "window": args.window, "stride_step": args.stride_step,
                      "anchor_step": args.anchor_step,
                      "anchor_lags": anchors_pretty(args.stride_step, args.anchor_step, args.window * args.stride_step),
                      "gmc_calls_per_frame": round(wt_anchor + wt_step, 3),
                      "compose_per_frame": round(wt_comp, 3),
                      "max_chain_depth": max(r["max_chain_depth"] for r in seq_rows),
                      "frames": ok, "wall_seconds": round(elapsed, 1),
                      "ms_per_frame": round(ms, 2),
                      "per_sequence": seq_rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
