#!/usr/bin/env python3
"""28_make_calib_strong_train.py -- build a LEAK-FREE "strong target" calibration set.

WHY THIS EXISTS
---------------
Two facts drive the design:

1. The quantiser's OUTPUT range is the range of the calibration frames. The
   trial-0474 int8 model that shipped was calibrated with 200 Anti-UAV TRAIN
   features and its output scale was wide enough for the real peaks (0.90-0.97);
   the 2026-10-23 trial-0080 int8 model clips those peaks (T1 loses 643 TP).
   So the calibration must CONTAIN frames whose heatmap peak is high.

2. The v2 family that tried to do exactly that (set_1 / set_2 / set_3) picked
   its "strong" frames from the EVALUATION side -- 001324 (T3 blind) and the
   golden-pool sequences (T2).  That leaks: the quantiser has seen the
   evaluation frames' activation distribution, so the T2/T3 numbers of those
   arms are optimistic and the arms are not comparable with a clean one
   (27_check_calib_leak.py proves it).

=> This script rebuilds the idea from the TRAIN side only:
     Anti-UAV train features   /mnt/data/siping/datasets/manu/uav_gmc_median/images/train
     Longquan official train   /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train
   The float ONNX gives each candidate's heatmap peak; the strong ones become
   the calibration set, the rest keeps the domain diversity set_A had.

NO FEATURE IS RECOMPUTED: the features already exist as JPGs; this script only
re-encodes the selected ones to BMP (lossless, so the pixels the toolkit sees
are exactly the pixels the peak probe saw) and writes the calibration list.

Usage (server):
    python manu/pipeline/npu/ptq/28_make_calib_strong_train.py \\
        --out /mnt/data/siping/ptq_work/calib4 --key 4 \\
        --n-strong 100 --n-diverse 150 --probe 2000 --peak-th 0.95
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

UAV_TRAIN = Path("/mnt/data/siping/datasets/manu/uav_gmc_median/images/train")
LQ_TRAIN = Path("/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train")
UAV_LABELS = Path("/mnt/data/siping/datasets/manu/uav_gmc_median/labels/train")
LQ_LABELS = Path("/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/labels/train")


def seq_frames(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in sorted(root.glob("*")):
        if not p.is_file():
            continue
        out.setdefault(C.split_seq_frame(p.name)[0], []).append(p)
    return out


def frame_index_of(name: str) -> int:
    """Frame index of a feature file.

    The two trees name frames differently:
      Anti-UAV  : 01_4485_2985-4484__000061.jpg
      Longquan  : VIDEO00002_19700101_001415__frame_000230.jpg
    but the LABELS are uniform: {seq}__{frame_index:06d}.txt in both.
    """
    stem = Path(name).stem
    if "__frame_" in stem:
        try:
            return int(stem.rsplit("__frame_", 1)[1])
        except ValueError:
            return -1
    return C.split_seq_frame(name)[1]


def has_target(seq: str, fid: int, labels_root: Path) -> bool:
    """Both label namings are in use:
         Anti-UAV  : {seq}__{fid:06d}.txt
         Longquan  : {seq}__frame_{fid:06d}.txt
    """
    if fid < 0:
        return False
    for name in (f"{seq}__{fid:06d}.txt", f"{seq}__frame_{fid:06d}.txt"):
        p = labels_root / name
        if p.is_file() and p.read_text().strip():
            return True
    return False


def peak_of(sess, path: Path) -> float:
    """Max heatmap value of one feature, in the BOARD's byte order."""
    u8 = C.bmp_to_input_nhwc(path) if path.suffix.lower() == ".bmp" else None
    if u8 is None:
        img = C.letterbox(C.load_feature_bmp(path), C.IMGSZ)
        u8 = img[None]
    nchw = np.ascontiguousarray(u8.transpose(0, 3, 1, 2)).astype(np.float32) / 255.0
    out = sess.run(None, {"images": nchw})
    return float(np.asarray(out[0], np.float32).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/data/siping/ptq_work/calib4")
    ap.add_argument("--key", default="4")
    ap.add_argument("--n-strong", type=int, default=100)
    ap.add_argument("--n-diverse", type=int, default=150)
    ap.add_argument("--probe", type=int, default=2000,
                    help="how many target-bearing candidate frames to probe")
    ap.add_argument("--peak-th", type=float, default=0.85,
                    help="absolute floor; the strong set is the TOP-N by peak")
    ap.add_argument("--per-seq-cap", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--onnx", default="/tmp/pycharm_project_10ae9e2e/runs/onnx/trial_0080_hm_nop0.onnx")
    ap.add_argument("--reuse-peaks", default="",
                    help="JSON cache of the probe results; skips the (slow) peak probe")
    ap.add_argument("--peak-cache", default="",
                    help="where to write the probe cache (default: <out>/peaks.json)")
    ap.add_argument("--pools", default="both", choices=["both", "uav", "lq"],
                    help="which TRAIN pool(s) to draw from: both = Anti-UAV + "
                         "Longquan official, lq = Longquan official only (the "
                         "business domain), uav = Anti-UAV only")
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    out_root = Path(args.out)
    bmp_dir = out_root / f"set_{args.key}_bmp"
    bmp_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---- candidate pools (TRAIN side only) ---------------------------------
    pools = []
    want = {"both": ("uav", "lq"), "uav": ("uav",), "lq": ("lq",)}[args.pools]
    for root, labels, tag in ((UAV_TRAIN, UAV_LABELS, "uav"), (LQ_TRAIN, LQ_LABELS, "lq")):
        if tag not in want:
            continue
        if not root.is_dir():
            print(f"[skip] {root} missing")
            continue
        for seq, frames in seq_frames(root).items():
            for fr in frames:
                pools.append((fr, labels, seq, tag))
    print(f"[pool] {len(pools)} train frames "
          f"({sum(1 for p in pools if p[3]=='uav')} Anti-UAV + "
          f"{sum(1 for p in pools if p[3]=='lq')} Longquan official)")

    # only frames that actually carry a target can be "strong"
    cand = []
    for fr, labels, seq, tag in pools:
        fid = frame_index_of(fr.name)
        if has_target(seq, fid, labels):
            cand.append((fr, labels, seq, tag))
    print(f"[cand] {len(cand)} target-bearing train frames")
    rng.shuffle(cand)
    probe_list = cand[: args.probe]

    # ---- probe the float peak ----------------------------------------------
    cache_path = Path(args.peak_cache) if args.peak_cache else out_root / f"peaks_{args.key}.json"
    if args.reuse_peaks and Path(args.reuse_peaks).is_file():
        blob = json.loads(Path(args.reuse_peaks).read_text())
        peaks = [(float(pk), Path(p), seq, tag) for pk, p, seq, tag in blob["peaks"]]
        print(f"[probe] reused {len(peaks)} cached peaks from {args.reuse_peaks}")
    else:
        peaks = []
    for i, (fr, labels, seq, tag) in enumerate([] if args.reuse_peaks else probe_list):
        try:
            pk = peak_of(sess, fr)
        except Exception as e:                                   # noqa: BLE001
            print(f"[probe] {fr.name} failed: {e}")
            continue
        peaks.append((pk, fr, seq, tag))
        if (i + 1) % 200 == 0:
            print(f"[probe] {i+1}/{len(probe_list)}  "
                  f"max so far {max(p[0] for p in peaks):.4f}", flush=True)
    peaks.sort(key=lambda t: -t[0])
    try:
        cache_path.write_text(json.dumps(
            {"peaks": [[pk, str(p), seq, tag] for pk, p, seq, tag in peaks]}, ensure_ascii=False))
        print(f"[probe] cache -> {cache_path}")
    except Exception as e:                                       # noqa: BLE001
        print(f"[probe] cache write failed: {e}")
    print(f"[probe] done: {len(peaks)} frames, peak p50={np.percentile([p[0] for p in peaks],50):.4f} "
          f"p95={np.percentile([p[0] for p in peaks],95):.4f} max={peaks[0][0]:.4f}")

    # ---- select: strong (capped per sequence) + diverse ---------------------
    # The strong set is the TOP-N by peak, with an absolute floor and a per
    # sequence cap. An absolute quantile like 0.95 is not portable: the train
    # pool peaks lower than the val pool did (observed max 0.918), so a fixed
    # 0.95 would have selected nothing.
    strong, per_seq = [], {}
    for pk, fr, seq, tag in peaks:            # already sorted by peak desc
        if pk < args.peak_th:
            break
        if per_seq.get(seq, 0) >= args.per_seq_cap:
            continue
        per_seq[seq] = per_seq.get(seq, 0) + 1
        strong.append((pk, fr, seq, tag))
        if len(strong) >= args.n_strong:
            break
    if strong:
        print(f"[select] strong peaks: min={strong[-1][0]:.4f} "
              f"max={strong[0][0]:.4f} over {len(per_seq)} sequences")
    print(f"[select] strong: {len(strong)} frames at peak >= {args.peak_th} "
          f"over {len(per_seq)} sequences")

    used = {str(f) for _, f, _, _ in strong}
    rest = [(pk, fr, seq, tag) for pk, fr, seq, tag in peaks if str(fr) not in used]
    diverse = rest[: args.n_diverse]
    print(f"[select] diverse: {len(diverse)} frames")

    picks = strong + diverse
    # ---- write BMPs + the list ---------------------------------------------
    lines, sources, missing = [], [], 0
    for pk, fr, seq, tag in picks:
        dst = bmp_dir / (fr.stem + ".bmp")
        if not dst.is_file():
            try:
                C.write_feature_bmp(fr, dst)
            except Exception as e:                              # noqa: BLE001
                print(f"[bmp] {fr} failed: {e}")
                missing += 1
                continue
        lines.append(str(dst))
        sources.append(str(fr))
    lst = out_root / f"set_{args.key}.txt"
    lst.write_text("\n".join(lines) + "\n")
    src = out_root / f"set_{args.key}_sources.txt"
    src.write_text("\n".join(sources) + "\n")

    manifest = {
        "key": args.key,
        "out_root": str(out_root),
        "n": len(lines), "missing": missing,
        "n_strong": len(strong), "n_diverse": len(diverse),
        "peak_th": args.peak_th, "probe": args.probe,
        "peak_max": peaks[0][0] if peaks else None,
        "pool": "TRAIN side only (Anti-UAV train + Longquan official train)",
        "list": str(lst), "sources": str(src),
        "leak_note": "built from train splits ONLY -- 27_check_calib_leak.py must pass",
    }
    (out_root / f"set_{args.key}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"[out] {lst}  ({len(lines)} frames, {missing} missing)")


if __name__ == "__main__":
    main()
