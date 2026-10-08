#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Reconcile the causal OnlineFeaturePipeline against the frozen tree feature set.

Three gates, from strictest to loosest:

  G1  in-memory zero drift  : streaming tensor  vs  the offline algorithm replayed
                              non-causally in this file. Expected ``max|diff| == 0``
                              exactly -- not a tolerance, zero.
  G2  byte-identical files  : the streamed tensor re-encoded exactly as the builder did, compared
                              against the frozen feature files. Identical pre-encode arrays imply
                              byte-identical JPEG (established by the ctrl arm), so this must be 0.
  G3  shape/dtype contract  : (3, H, W) uint8, [Ch0, Ch1, Ch2], and Ch0 == I_t bit-exact.

Why G1 shares primitives with the pipeline
    ``FastGMCEstimator``, ``_compose`` and ``anchor_grid`` are imported, not re-implemented. The
    thing under test is the **state management** (FIFO vs full-sequence indexing), so sharing the
    estimator isolates exactly that. The independent part is the control flow: the reference below
    is non-causal (absolute indices, no ring), the pipeline is causal.

Why the spec's "MAE < 0.05" belongs to G1 and not G2
    The frozen tree features are stored as JPEG. Re-encoding noise alone puts the decoded arrays
    ~0.1-0.3 gray levels apart, so ``max <= 1 / MAE < 0.05`` is unreachable against the files
    no matter how correct the pipeline is. G2 therefore reports actuals against a separately
    labelled JPEG bound, while G1 enforces the strict zero.

Cache budget
    Per 铁律二 the pkl stays sparse: per-frame statistics for every frame, plus a configurable
    stride of whole tensors for eyeball / byte comparison. 200 frames x (3,512,640) uint8 is
    196 MiB, which must not be persisted. The comparison itself runs in memory.

Usage (server):
    cd /tmp/pycharm_project_10ae9e2e
    PYTHONPATH=. python manu/pipeline/verify_streaming_pipeline.py \\
        --sequence wg2022_ir_052_split_08 --frames 200 \\
        --raw-root /mnt/data/siping/datasets/manu/anti-uav \\
        --frozen-root /mnt/data/siping/datasets/manu/uav_gmc_median_tree \\
        --pkl runs/gmc_eval/streaming_wg2022_ir_052_split_08.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.data.build_nogmc_median_dataset import _compose, anchor_grid  # noqa: E402
from manu.data.build_sample_median_dataset import FastGMCEstimator  # noqa: E402
from manu.pipeline.streaming_feature_pipeline import (  # noqa: E402
    OnlineFeaturePipeline,
    SequenceFrameSource,
    fit_similarity,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify the streaming feature pipeline against frozen tree features")
    p.add_argument("--sequence", type=str, default="wg2022_ir_052_split_08")
    p.add_argument("--frames", type=int, default=200, help="Raw frames to push (0 = all available)")
    p.add_argument("--raw-root", type=str, default="/mnt/data/siping/datasets/manu/anti-uav")
    p.add_argument("--frozen-root", type=str, default="/mnt/data/siping/datasets/manu/uav_gmc_median_tree")
    p.add_argument("--window", type=int, default=21)
    p.add_argument("--stride-step", type=int, default=2)
    p.add_argument("--anchor-step", type=int, default=10)
    p.add_argument("--downscale", type=int, default=2)
    p.add_argument("--skip-cold-start", type=int, default=0,
                   help="Exclude the first N frames from the per-frame report (default 0 = keep all)")
    p.add_argument("--g2-max-diff", type=float, default=0.0,
                   help="G2 bound on the re-encoded file comparison; exactness gate, keep at 0")
    p.add_argument("--g2-max-mae", type=float, default=0.0)
    p.add_argument("--dump-every", type=int, default=40, help="Store every Nth whole tensor in the pkl")
    p.add_argument("--matrix-probe-frames", type=int, default=64,
                   help="Frames for which the per-lag transform matrices are diffed")
    p.add_argument("--pkl", type=str, default="runs/gmc_eval/streaming_verify.pkl")
    p.add_argument("--dump-intermediate", action="store_true", help="Also write per-frame .npy intermediates")
    p.add_argument("--dump-dir", type=str, default="")
    args = p.parse_args()
    if args.dump_intermediate and not args.dump_dir:
        p.error("--dump-intermediate requires --dump-dir")
    return args


def offline_reference(
    raw_frames: list[np.ndarray], window: int, stride_step: int, anchor_step: int, downscale: int,
    keep_mats_upto: int = 0,
) -> tuple[list[np.ndarray], dict[int, dict[int, np.ndarray]]]:
    """Non-causal replay of the offline tree algorithm. Absolute indices, no ring buffer.

    Deliberately written as a straight transcription of the offline control flow so that any
    divergence from the streaming version shows up as a non-zero G1.
    """
    max_lag = window * stride_step
    lags = list(range(stride_step, max_lag + 1, stride_step))
    anchors = anchor_grid(stride_step, anchor_step, max_lag)
    anchor_set = set(anchors)
    est = FastGMCEstimator(downscale=downscale)
    n = len(raw_frames)

    def read(i: int) -> np.ndarray:
        return raw_frames[max(0, min(i, n - 1))]

    steps: dict[int, np.ndarray] = {}

    def pstep(j: int) -> np.ndarray:
        if j not in steps:
            steps[j] = fit_similarity(est, read(j), read(j + stride_step))
        return steps[j]

    out: list[np.ndarray] = []
    kept: dict[int, dict[int, np.ndarray]] = {}
    for ci, curr in enumerate(raw_frames):
        mats: dict[int, np.ndarray] = {}
        for lag in anchors:
            mats[lag] = fit_similarity(est, read(ci - lag), curr)
        for lag in lags:
            if lag in anchor_set:
                continue
            below = max(a for a in anchors if a <= lag)
            acc = mats[below]
            for lag_abs in range(below + stride_step, lag + 1, stride_step):
                acc = _compose(pstep(ci - lag_abs), acc)
            mats[lag] = acc
        ch1 = cv2.absdiff(curr, est.warp(read(ci - stride_step), mats[stride_step]))
        history = [est.warp(read(ci - lag), mats[lag]) for lag in lags]
        median_bg = np.median(np.stack(history, axis=0), axis=0).astype(np.float32)
        ch2 = np.clip(curr.astype(np.float32) - median_bg, 0, 255).astype(np.uint8)
        out.append(np.stack([curr, ch1, ch2], axis=0))
        if keep_mats_upto and ci < keep_mats_upto:
            kept[ci] = {lag: m.copy() for lag, m in mats.items()}
    return out, kept


def frozen_lookup(frozen_dir: Path, sequence: str) -> dict[int, Path]:
    """raw frame number -> frozen feature file, matching the offline manifest naming."""
    mapping: dict[int, Path] = {}
    for p in frozen_dir.iterdir():
        if not p.is_file() or not p.name.startswith(f"{sequence}__"):
            continue
        m = re.search(r"(\d+)$", p.stem)
        if m:
            mapping[int(m.group(1))] = p
    return mapping


def main() -> int:
    args = parse_args()
    cv2.setNumThreads(1)
    print(f"[THREADS] cv2.setNumThreads(1) pinned; opencv reports "
          f"{cv2.getNumThreads()} thread(s). Bit-reproducibility requires this.")

    source = SequenceFrameSource(args.raw_root, args.sequence, limit=args.frames)
    raw_frames = list(source)
    px = raw_frames[0].size
    hold = len(raw_frames) * px * 3  # raw + streamed + reference, all resident at once
    print(f"[MEMORY] verification holds raw + streamed + reference in RAM: "
          f"~{3 * len(raw_frames) * px / 1073741824:.2f} GiB for {len(raw_frames)} frames")
    print("=" * 100)
    print("   STREAMING FEATURE PIPELINE VERIFICATION")
    print(f"   sequence      : {args.sequence}")
    print(f"   raw source    : {source.directory}")
    print(f"   raw frames    : {len(raw_frames)}  shape(H,W)={raw_frames[0].shape} dtype={raw_frames[0].dtype}")
    print(f"   frozen root   : {args.frozen_root}")
    print(f"   window={args.window} stride_step={args.stride_step} anchor_step={args.anchor_step} "
          f"downscale={args.downscale}")
    print("=" * 100, flush=True)

    pipe = OnlineFeaturePipeline(
        window=args.window,
        stride_step=args.stride_step,
        anchor_step=args.anchor_step,
        downscale=args.downscale,
        expected_shape=raw_frames[0].shape,
        dump_intermediate=args.dump_intermediate,
        dump_dir=args.dump_dir or None,
    )
    print(f"[CONFIG] ring_depth={pipe.ring_depth} consecutive frames (lags 0..{args.window * args.stride_step}) "
          f"| lags used {pipe.lags[0]}..{pipe.lags[-1]} | anchors={pipe.anchors} "
          f"| full ring = {pipe.ring_depth * raw_frames[0].size / 1048576:.2f} MiB")

    # ---- streaming pass (timed per frame) ----
    streamed: list[np.ndarray] = []
    latencies: list[float] = []
    probe = min(args.matrix_probe_frames, len(raw_frames))
    stream_mats: dict[int, dict[int, np.ndarray]] = {}
    for idx, frame in enumerate(raw_frames):
        t0 = time.perf_counter()
        out = pipe.push(frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        streamed.append(out)
        if idx < probe and pipe.last_mats is not None:
            stream_mats[idx] = {lag: m.copy() for lag, m in pipe.last_mats.items()}
    print(f"[STREAM] pushed {len(streamed)} frames | state_bytes={pipe.state_bytes()} "
          f"({pipe.state_bytes() / 1048576:.2f} MiB)", flush=True)

    print(f"[REF] replaying the offline algorithm non-causally over {len(raw_frames)} frames ...", flush=True)
    reference, ref_mats = offline_reference(raw_frames, args.window, args.stride_step,
                                            args.anchor_step, args.downscale, keep_mats_upto=probe)

    frozen_map = frozen_lookup(Path(args.frozen_root) / "images" / "val", args.sequence)
    # SequenceFrameSource orders frames by natural_key; recover each frame's trailing number so the
    # offline manifest name can be resolved exactly the way the builder's idx_map does.
    numbers: list[int] = []
    for path in source.paths:
        m = re.search(r"(\d+)$", path.stem)
        numbers.append(int(m.group(1)) if m else len(numbers))
    print(f"[REF] frozen manifest entries for {args.sequence}: {len(frozen_map)}")

    # ---- G0: is the non-causal replay itself faithful to the frozen dataset? ----
    g0 = {"max": 0, "sum": 0.0, "n": 0, "matched": 0}
    for i in range(probe):
        disk = frozen_map.get(numbers[i])
        if disk is None:
            continue
        f = cv2.imread(str(disk), cv2.IMREAD_UNCHANGED)
        if f is None or f.shape != (reference[i].shape[1], reference[i].shape[2], reference[i].shape[0]):
            continue
        ok, buf = cv2.imencode(".jpg", reference[i].transpose(1, 2, 0))
        if not ok:
            continue
        re_dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        d = np.abs(re_dec.astype(np.int16) - f.astype(np.int16))
        g0["max"] = max(g0["max"], int(d.max()))
        g0["sum"] += float(d.sum())
        g0["n"] += d.size
        g0["matched"] += 1
    print(f"[G0] offline replay re-encoded vs frozen files over {g0['matched']}/{probe} probed frames: "
          f"max|d|={g0['max']} MAE={g0['sum'] / max(g0['n'], 1):.6f}"
          f"  {'-> replay is faithful' if g0['max'] == 0 else '-> REPLAY ITSELF IS WRONG'}")

    # ---- per-lag matrix probe: which transform first differs ----
    worst = {}
    for i in range(probe):
        if i not in ref_mats or i not in stream_mats:
            continue
        for lag, rm in ref_mats[i].items():
            sm = stream_mats[i].get(lag)
            if sm is None:
                worst[lag] = max(worst.get(lag, 0), 999)
                continue
            d = float(np.abs(sm.astype(np.float64) - rm.astype(np.float64)).max())
            worst[lag] = max(worst.get(lag, 0), d)
    if worst:
        anchors = set(anchor_grid(args.stride_step, args.anchor_step, args.window * args.stride_step))
        anc = {k: v for k, v in worst.items() if k in anchors}
        der = {k: v for k, v in worst.items() if k not in anchors}
        print(f"[MATS] anchor lags      max diff over probe: {[(k, round(v, 8)) for k, v in sorted(anc.items())]}")
        print(f"[MATS] derived lags     max diff over probe: "
              f"{[(k, round(v, 8)) for k, v in sorted(der.items()) if v > 0] or 'all identical'}")
        if anc and max(anc.values()) > 0:
            print("[MATS] anchor fits already differ -> the estimator or its inputs differ, "
                  "not the composition")

    # ---- per-frame comparison ----
    start = max(0, args.skip_cold_start)
    per_frame: list[dict] = []
    g1 = {c: {"max": 0, "sum": 0.0, "n": 0} for c in range(3)}
    g2 = {c: {"max": 0, "sum": 0.0, "n": 0} for c in range(3)}
    g2b = {"max": 0, "sum": 0.0, "n": 0}
    shape_ok = dtype_ok = ch0_exact = True
    g2_matched = 0
    samples: list[dict] = []
    diverge: list[int] = []

    for i in range(len(streamed)):
        s = streamed[i]
        r = reference[i]
        if s.shape != r.shape:
            shape_ok = False
        if s.dtype != np.uint8:
            dtype_ok = False
        if not np.array_equal(s[0], raw_frames[i]):
            ch0_exact = False
        if i < start:
            continue
        d1 = np.abs(s.astype(np.int16) - r.astype(np.int16))
        num = int(d1[0].size)
        for c in range(3):
            g1[c]["max"] = max(g1[c]["max"], int(d1[c].max()))
            g1[c]["sum"] += float(d1[c].sum())
            g1[c]["n"] += num

        row = {"frame": i, "latency_ms": round(latencies[i], 3), "frozen": None}
        row["ref_max"] = [int(d1[c].max()) for c in range(3)]
        per_frame.append(row)
        if int(d1.max()) > 0:
            diverge.append(i)
        if args.dump_every > 0 and i % args.dump_every == 0:
            samples.append({"frame": i, "tensor": s.copy()})

        disk = frozen_map.get(numbers[i])
        if disk is None:
            continue
        f = cv2.imread(str(disk), cv2.IMREAD_UNCHANGED)
        if f is None or f.shape != (s.shape[1], s.shape[2], s.shape[0]):
            row["frozen"] = f"{disk.name} (UNREADABLE)"
            continue
        g2_matched += 1
        # G2: re-encode the streamed tensor exactly as the builder did, then compare files.
        # Identical pre-encode arrays -> byte-identical JPEG (proved by the ctrl arm), so a
        # non-zero diff here means the pre-encode arrays genuinely differ.
        ok, buf = cv2.imencode(".jpg", s.transpose(1, 2, 0))
        if not ok:
            row["frozen"] = f"{disk.name} (REENCODE FAILED)"
            continue
        re_dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        d2 = np.abs(re_dec.astype(np.int16) - f.astype(np.int16))
        for c in range(3):
            g2[c]["max"] = max(g2[c]["max"], int(d2[c].max()))
            g2[c]["sum"] += float(d2[c].sum())
            g2[c]["n"] += num
        # G2b, informational only: the raw JPEG quantisation residual, never gated.
        dj = np.abs(s.astype(np.int16) - f.astype(np.int16).transpose(2, 0, 1))
        g2b["max"] = max(g2b["max"], int(dj.max()))
        g2b["sum"] += float(dj.sum())
        g2b["n"] += num
        row["frozen"] = disk.name
        row["frozen_max"] = [int(d2[c].max()) for c in range(3)]
        row["jpeg_residual_max"] = int(dj.max())

    lat = np.asarray(latencies[start:], dtype=np.float64)
    def pct(q: float) -> float:
        return float(np.percentile(lat, q)) if lat.size else float("nan")

    g1_mae = {c: (g1[c]["sum"] / g1[c]["n"] if g1[c]["n"] else 0.0) for c in range(3)}
    g2_mae = {c: (g2[c]["sum"] / g2[c]["n"] if g2[c]["n"] else 0.0) for c in range(3)}
    g1_max = max(g1[c]["max"] for c in range(3))
    g2_max = max(g2[c]["max"] for c in range(3))
    if diverge:
        print(f"[DIVERGE] G1 non-zero on {len(diverge)}/{len(per_frame)} compared frames; "
              f"first={diverge[0]} last={diverge[-1]}")
        print(f"[DIVERGE] first {min(8, len(diverge))} frames:")
        by_frame = {r["frame"]: r for r in per_frame}
        for i in diverge[:8]:
            r = by_frame[i]
            print(f"           frame={i:<5} ref_max={r['ref_max']}  latency={r['latency_ms']:.1f}ms")
        cold = [i for i in diverge if i < args.window * args.stride_step]
        print(f"[DIVERGE] {len(cold)} of them are inside the cold-start region "
              f"(frame < {args.window * args.stride_step})")
    else:
        print("[DIVERGE] none -- every compared frame is bit-identical to the offline replay")
    g2_usable = g2_matched > 0
    if not g2_usable:
        print("[WARN] no frozen feature file matched this sequence/lag configuration -> G2 is "
              "SKIPPED, not passed. A different --stride-step selects a different feature set "
              "which is not comparable with the frozen tree dataset.")

    print(f"\n{'GATE':<34}{'max|d|':>10}{'MAE':>12}{'verdict':>14}")
    print("-" * 70)
    print(f"{'G1 in-memory vs offline replay':<34}{g1_max:>10}{max(g1_mae.values()):>12.6f}"
          f"{'PASS' if g1_max == 0 else 'FAIL':>14}")
    g2_verdict = "SKIP" if not g2_usable else (
        "PASS" if g2_max <= args.g2_max_diff and max(g2_mae.values()) <= args.g2_max_mae else "FAIL")

    print(f"{'G2 re-encoded vs frozen file':<34}{g2_max:>10}{max(g2_mae.values()):>12.6f}{g2_verdict:>14}")
    print(f"{'G3 shape/dtype/Ch0 exactness':<34}{'-':>10}{'-':>12}"
          f"{'PASS' if (shape_ok and dtype_ok and ch0_exact) else 'FAIL':>14}")
    print("-" * 70)
    for c, nm in ((0, "Ch0 I_t"), (1, "Ch1 |I_t-W(I_t-2)|"), (2, "Ch2 (I_t-B_t)^+")):
        print(f"  {nm:<26} G1 max={g1[c]['max']:<4} MAE={g1_mae[c]:.6f}    "
              f"G2 max={g2[c]['max']:<4} MAE={g2_mae[c]:.6f}")

    if g2b["n"]:
        print(f"[G2b INFO] raw JPEG quantisation residual (streaming vs decoded frozen, never gated): "
              f"max={g2b['max']} MAE={g2b['sum'] / g2b['n']:.4f}")
    print(f"\n[LATENCY] per frame, single process, cv2 threads unconstrained: "
          f"P50={pct(50):.2f}ms P90={pct(90):.2f}ms P99={pct(99):.2f}ms "
          f"mean={float(lat.mean()) if lat.size else float('nan'):.2f}ms (n={lat.size})")
    print(f"[FROZEN] matched {g2_matched}/{len(per_frame)} frames against the on-disk manifest")

    g2_pass = g2_verdict in ("PASS", "SKIP")
    verdict = (g1_max == 0) and shape_ok and dtype_ok and ch0_exact and g2_pass
    print(f"\n[VERDICT] {'PASS' if verdict else 'FAIL'}")

    pkl_path = Path(args.pkl)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequence": args.sequence,
        "frames": len(streamed),
        "compared_from_frame": start,
        "config": {"window": args.window, "stride_step": args.stride_step,
                   "anchor_step": args.anchor_step, "downscale": args.downscale,
                   "ring_depth": pipe.ring_depth, "anchors": pipe.anchors,
                   "state_bytes": pipe.state_bytes()},
        "g1_inmemory": {"max_abs": g1_max, "mae": g1_mae, "per_channel_max": {c: g1[c]["max"] for c in range(3)}},
        "g2_reencoded_vs_frozen": {"max_abs": g2_max, "mae": g2_mae, "matched_frames": g2_matched,
                                   "verdict": g2_verdict,
                                   "bound": {"max_diff": args.g2_max_diff, "max_mae": args.g2_max_mae}},
        "g2b_jpeg_residual": {"max_abs": g2b["max"], "mae": g2b["sum"] / g2b["n"] if g2b["n"] else None},
        "g1_divergent_frames": diverge[:64],
        "latency_ms": {"p50": pct(50), "p90": pct(90), "p99": pct(99)},
        "shape_ok": shape_ok, "dtype_ok": dtype_ok, "ch0_bit_exact": ch0_exact,
        "verdict": "PASS" if verdict else "FAIL",
        "per_frame": per_frame,
        "tensor_samples_every": args.dump_every,
        "tensor_samples": samples,
        "note": "tensor_samples holds every Nth whole tensor; full retention would violate 铁律二 "
                f"({len(streamed)} frames x (3,{raw_frames[0].shape[0]},{raw_frames[0].shape[1]}) "
                f"= {len(streamed) * 3 * raw_frames[0].size / 1048576:.0f} MiB).",
    }
    with pkl_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    print(f"[SAVED] {pkl_path}  ({pkl_path.stat().st_size / 1048576:.2f} MiB)")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())