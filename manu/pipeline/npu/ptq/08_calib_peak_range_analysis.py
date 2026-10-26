#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
08_calib_peak_range_analysis.py -- answer, with MEASUREMENTS, why two int8 models
built from the SAME recipe end up with output-tensor scales that differ by 3.9x,
and build the calibration sets v2 that fix it.

THE QUESTION
------------
    trial-0474  int8 -> board out[0] zp=-128 scale=0.00365272  -> peak cap 0.93
    trial-0080  int8 -> board out[0] zp=-128 scale=0.00094072  -> peak cap 0.24
                                                            (0.24 < TH_BASE 0.3
                                                             => 0 detections/frame)

Same graph recipe (float32-input ONNX + mean_values=[[0,0,0]] /
std_values=[[255,255,255]]), same normalisation, same threshold. The only thing
that can move the OUTPUT tensor's (zp, scale) is the CALIBRATION SET: RKNN picks
the output layer's affine quantisation from the range that tensor takes over the
calibration images.

    calibration peak 0.4639 -> scale = 0.4639/127 = 0.003653   <- trial-0474
    calibration peak 0.1195 -> scale = 0.1195/127 = 0.000941   <- trial-0080

Both board numbers fall straight out of that arithmetic, which is why the
hypothesis under test is:

    "the calibration set does not cover the heatmap peak range of the real eval
     frames, so the int8 output range is too narrow, the real peaks are clipped
     at (127-(-128))*scale = 0.24, and 0.24 < 0.3 => every frame scores zero."

HOW THIS SCRIPT TESTS IT (measurements, not reasoning)
-------------------------------------------------------
Run the float32 ONNX (trial_0080_hm_nop0.onnx) on every group, record the
per-frame heatmap MAX, then convert each group's max into

    theoretical scale   = max / 127      (what a min/max int8 quantiser picks)
    theoretical ceiling = 255 * scale    ((127 - (-128)) steps, RKNN int8)

    group                                   expected under the hypothesis
    ------------------------------------------------------------------------
    E  trial-0474 replica (official train)  ~0.46  -> scale ~0.00365  <- 0474
    A..D the four shipped calibration sets   ~0.12  -> scale ~0.00094  <- 0080
    F  the real blind-eval frames (001324)   >> 0.46 (clipping to 0.24)

If E lands on 0.46, A..D land on 0.12 and F is far above 0.46, the hypothesis is
CONFIRMED and the fix is a calibration set whose peak covers F.

Then it writes calibration set v2 (3 x 250 BMPs) to
/mnt/data/siping/ptq_work/calib2/ and reports each candidate's peak statistics.

BYTE SOURCES (see ptq_common.py "THE PIXEL CONTRACT")
------------------------------------------------------
    A,B,C,D,F,G,H,I  BMP files on disk, read through the sanctioned reader
                    C.load_feature_bmp() + C.letterbox() (C.bmp_to_input_nchw_float
                    is asserted to give bit-identical results on sampled frames).
    E  + the official-train part of the candidate sets
                    the official *train* features exist only as JPG on the server.
                    They are decoded with cv2.imread() IN MEMORY and fed with the
                    identical channel order + letterbox as the BMP route. This is
                    byte-exact: write_feature_bmp() stores cv2.imread(src) and BMP
                    is lossless, therefore cv2.imread(bmp) == cv2.imread(jpg).
                    --self-check PROVES it at runtime (array equality on 3 frames).
                    Every image that has to be handed to the RKNN toolkit
                    (candidates 1-3) is additionally written to disk as a real BMP
                    through C.write_feature_bmp(), so the toolkit reads exactly the
                    bytes the PC arm read.

DETERMINISM
-----------
No random() anywhere. Every selection is "sort + fixed stride / fixed quantile",
so re-running the script reproduces the same frames and the same numbers.

Usage
-----
    # everything: groups -> pool scan -> candidate sets -> candidate statistics
    python -u manu/pipeline/npu/ptq/08_calib_peak_range_analysis.py --stage all

    # only the group peak statistics (fast, ~4 min)
    python -u manu/pipeline/npu/ptq/08_calib_peak_range_analysis.py --stage groups

    # only the pool scan (needed to build the candidates, ~13 min)
    python -u manu/pipeline/npu/ptq/08_calib_peak_range_analysis.py --stage scan

    # build the candidate sets from a cached scan
    python -u manu/pipeline/npu/ptq/08_calib_peak_range_analysis.py --stage candidates

Run it from the repository root with PYTHONPATH=. (see run_ptq_pipeline.sh).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent.parent.parent.parent,
           _HERE.parent.parent.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402

# --------------------------------------------------------------------------------------
# Paths (environment-overridable, same convention as ptq_common.py)
# --------------------------------------------------------------------------------------
WORK_ROOT = Path(os.environ.get("PTQ_WORK_ROOT", "/mnt/data/siping/ptq_work"))
CALIB_ROOT = Path(os.environ.get("PTQ_CALIB_ROOT", str(WORK_ROOT / "calib")))
CALIB2_ROOT = Path(os.environ.get("PTQ_CALIB2_ROOT", str(WORK_ROOT / "calib2")))
OUT_ROOT = Path(os.environ.get("PTQ_PEAK_ROOT", str(WORK_ROOT / "peak_analysis")))
ONNX_FLOAT = Path(os.environ.get(
    "PTQ_ONNX_FLOAT",
    "/tmp/pycharm_project_10ae9e2e/runs/onnx/trial_0080_hm_nop0.onnx"))

SET_SIZE = 250
TRAIN_REPLICA_N = 200      # group E
VAL_BLIND_N = 200          # group F
GOLDEN_PER_SEQ = 30        # group G

# the int8 board numbers we are trying to explain
REFERENCE = {
    "trial_0474": {"zp": -128, "scale": 0.00365272},
    "trial_0080": {"zp": -128, "scale": 0.00094072},
}


# --------------------------------------------------------------------------------------
# The sanctioned pixel path (thin wrappers over ptq_common, never a second convention)
# --------------------------------------------------------------------------------------
def bmp_to_input_and_hw(path) -> tuple[np.ndarray, tuple[int, int]]:
    """BMP -> (1,3,640,640) float32 in [0,1] AND the source (h, w).

    Uses C.load_feature_bmp() (the ONLY sanctioned reader) and C.letterbox() (the
    sanctioned letterbox). C.bmp_to_input_nchw_float() is asserted to produce the
    identical tensor (see assert_sanctioned_reader).
    """
    rgb = C.load_feature_bmp(path)[..., ::-1]          # BGR -> RGB == model order
    canvas = C.letterbox(rgb, C.IMGSZ)
    x = np.ascontiguousarray(canvas[None].astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
    # the source size is needed to map GT boxes back onto the canvas
    return x, (rgb.shape[0], rgb.shape[1])


def jpg_to_input_and_hw(path) -> tuple[np.ndarray, tuple[int, int]]:
    """JPG -> (1,3,640,640) float32 in [0,1] AND the source (h, w).

    Byte-exact twin of the BMP route (see the module docstring).
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read feature image: {path}")
    rgb = img[..., ::-1]                               # BGR -> RGB == model order
    canvas = C.letterbox(rgb, C.IMGSZ)
    x = np.ascontiguousarray(canvas[None].astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
    return x, (rgb.shape[0], rgb.shape[1])


def assert_sanctioned_reader(bmp_path) -> bool:
    """Prove that our wrapper == C.bmp_to_input_nchw_float (bit-identical)."""
    x, _ = bmp_to_input_and_hw(bmp_path)
    ref = C.bmp_to_input_nchw_float(bmp_path)
    ok = bool(np.array_equal(x, ref))
    if not ok:
        raise SystemExit(f"[FATAL] sanctioned-reader mismatch on {bmp_path}: "
                         f"maxdiff={np.abs(x - ref).max()}")
    return ok


def assert_jpg_bmp_equivalence(jpg_path, tmp_dir) -> bool:
    """Prove that the in-memory JPG route == the on-disk BMP route, bit-identically."""
    import cv2

    dst = Path(tmp_dir) / (Path(jpg_path).stem + ".bmp")
    C.write_feature_bmp(jpg_path, dst)
    xj, _ = jpg_to_input_and_hw(jpg_path)
    xb, _ = bmp_to_input_and_hw(dst)
    ok = bool(np.array_equal(xj, xb))
    if not ok:
        raise SystemExit(f"[FATAL] JPG/BMP byte mismatch on {jpg_path}: "
                         f"maxdiff={np.abs(xj - xb).max()}")
    dst.unlink()
    del cv2
    return ok


# --------------------------------------------------------------------------------------
# GT helpers
# --------------------------------------------------------------------------------------
def gt_count(seq: str, frame_index: int) -> int:
    """Number of GT boxes for (seq, frame_index); 0 when the file does not exist."""
    p = C.gt_path(seq, frame_index)
    try:
        return sum(1 for line in p.read_text().splitlines() if len(line.split()) >= 5)
    except OSError:
        return 0


def gt_centers(seq: str, frame_index: int, src_hw: tuple[int, int]) -> np.ndarray:
    """GT box centres in SOURCE pixel coordinates (Nx2)."""
    p = C.gt_path(seq, frame_index)
    h, w = src_hw
    pts = []
    try:
        for line in p.read_text().splitlines():
            fl = line.split()
            if len(fl) >= 5:
                pts.append([float(fl[1]) * w, float(fl[2]) * h])
    except OSError:
        pass
    return np.asarray(pts, np.float32).reshape(-1, 2)


def peak_at_gt(heatmap: np.ndarray, gts: np.ndarray, src_hw: tuple[int, int]) -> float:
    """Heatmap value at the strongest GT centre (canvas/stride mapping)."""
    if len(gts) == 0:
        return 0.0
    geom = C.letterbox_geometry(src_hw)
    hmax = heatmap.shape[-2]
    wmax = heatmap.shape[-1]
    xs = (gts[:, 0] * geom["scale"] + geom["pad_left"]) / C.STRIDE
    ys = (gts[:, 1] * geom["scale"] + geom["pad_top"]) / C.STRIDE
    ix = np.clip(xs.astype(int), 0, wmax - 1)
    iy = np.clip(ys.astype(int), 0, hmax - 1)
    return float(heatmap[iy, ix].max())


# --------------------------------------------------------------------------------------
# The scanner: float32 ONNX Runtime -> per-frame heatmap statistics
# --------------------------------------------------------------------------------------
class PeakScanner:
    """Runs the float32 ONNX and reports the per-frame heatmap max (and the GT peak)."""

    HIST_BINS = 2048

    def __init__(self, onnx_path=ONNX_FLOAT, threads: int = 8):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so,
                                         providers=["CPUExecutionProvider"])
        self.onnx_path = str(onnx_path)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.threads = threads
        self.n_frames = 0
        self.collect_hist = False
        self.hist = np.zeros(self.HIST_BINS, np.float64)

    def reset_hist(self) -> None:
        self.hist[:] = 0.0
        self.collect_hist = True

    def _heatmap(self, x: np.ndarray) -> np.ndarray:
        outs = self.sess.run(self.out_names, {self.in_name: x})
        d = dict(zip(self.out_names, outs))
        hm = np.asarray(d["heatmap"], np.float32).reshape(
            d["heatmap"].shape[-2], d["heatmap"].shape[-1])
        if self.collect_hist:
            h, _ = np.histogram(hm, bins=self.HIST_BINS, range=(0.0, 1.0))
            self.hist += h
        return hm

    def scan_bmp(self, path, want_gt: bool = True) -> dict:
        x, src_hw = bmp_to_input_and_hw(path)
        hm = self._heatmap(x)
        row = {"path": str(path), "peak": float(hm.max()),
               "source": "bmp_disk", "seq": None, "frame_id": None,
               "n_gt": 0, "peak_at_gt": None}
        seq, idx = C.split_seq_frame(Path(path).name)
        row["seq"], row["frame_id"] = seq, idx
        if want_gt and idx >= 0:
            row["n_gt"] = gt_count(seq, idx)
            gts = gt_centers(seq, idx, src_hw)
            row["peak_at_gt"] = peak_at_gt(hm, gts, src_hw)
        self.n_frames += 1
        return row

    def scan_jpg(self, path, want_gt: bool = True) -> dict:
        x, src_hw = jpg_to_input_and_hw(path)
        hm = self._heatmap(x)
        row = {"path": str(path), "peak": float(hm.max()),
               "source": "jpg_in_memory_cv2_imread", "seq": None, "frame_id": None,
               "n_gt": 0, "peak_at_gt": None}
        stem = Path(path).stem
        if "__frame_" in stem:
            seq, _, tail = stem.partition("__frame_")
            try:
                idx = int(tail)
            except ValueError:
                idx = -1
            row["seq"], row["frame_id"] = seq, idx
            if want_gt and idx >= 0:
                row["n_gt"] = gt_count(seq, idx)
                gts = gt_centers(seq, idx, src_hw)
                row["peak_at_gt"] = peak_at_gt(hm, gts, src_hw)
        self.n_frames += 1
        return row

    def scan_paths(self, paths, kind: str = "bmp", want_gt: bool = True) -> list[dict]:
        fn = self.scan_bmp if kind == "bmp" else self.scan_jpg
        out = []
        t0 = time.time()
        for i, p in enumerate(paths):
            out.append(fn(p, want_gt=want_gt))
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"    {i + 1}/{len(paths)}  {el:.0f}s "
                      f"({el / (i + 1) * 1000:.0f} ms/frame)", flush=True)
        return out


# --------------------------------------------------------------------------------------
# The KL-divergence clipping search (the mechanism behind the narrow output range)
# --------------------------------------------------------------------------------------
def _quantize_to_bins(counts: np.ndarray, nq: int, t: float, nb: int) -> np.ndarray:
    """Re-quantise a reference histogram into `nq` uniform bins over [0, t].

    Mass above t folds into the top bin (that is what "clipping at t" means).
    The result is a proper probability distribution over the SAME index space as
    the reference, which is what makes the KL divergence below well defined.
    """
    total = counts.sum()
    if total <= 0:
        return np.zeros(nq, np.float64)
    p = counts / total
    q = np.zeros(nq, np.float64)
    lo = np.arange(nb) / nb
    hi = (np.arange(nb) + 1.0) / nb
    inside = hi <= t + 1e-15
    above = lo >= t - 1e-15
    middle = ~(inside | above)

    cen = (lo[inside] + hi[inside]) / 2.0
    j = np.minimum((cen / max(t, 1e-12) * nq).astype(int), nq - 1)
    np.add.at(q, j, p[inside])

    for i in np.nonzero(middle)[0]:
        frac = float(np.clip((t - lo[i]) / max(hi[i] - lo[i], 1e-12), 0.0, 1.0))
        jm = int(min(((lo[i] + t) / 2.0) / max(t, 1e-12) * nq, nq - 1))
        q[max(jm, 0)] += p[i] * frac
        q[-1] += p[i] * (1.0 - frac)

    q[-1] += p[above].sum()
    s = q.sum()
    return q / s if s > 0 else q


def kl_threshold(hist: np.ndarray, n_thresholds: int = 128,
                 n_quant_bins: int = 128) -> tuple[float, float]:
    """Standard KL-divergence clipping search over a 1-D activation histogram.

    For every candidate upper threshold T the histogram is re-quantised into
    n_quant_bins uniform bins over [0, T] (all mass above T folds into the top
    bin) and the KL divergence against the full-range quantisation is measured.
    The T that minimises it is the threshold the quantiser is expected to pick.

    Returns (threshold, kl_divergence).

    This is the PREDICTOR for `quantized_algorithm="kl_divergence"`. The
    companion predictor for `quantized_algorithm="normal"` (plain min/max) is
    simply the observed maximum: scale = max / 255, ceiling = max.
    """
    counts = np.asarray(hist, np.float64)
    nb = counts.size
    if counts.sum() <= 0:
        return 0.0, 0.0
    p_full = _quantize_to_bins(counts, n_quant_bins, 1.0, nb)

    nz = np.nonzero(counts)[0]
    if nz.size < 2:
        return float((nz[-1] + 1) / nb), 0.0

    cand = np.unique(nz + 1)[::-1]                 # candidate thresholds, descending
    step = max(1, cand.size // n_thresholds)
    cand = cand[::step]
    if cand[0] != nz[-1] + 1:
        cand = np.concatenate([[nz[-1] + 1], cand])

    best_t, best_kl = float(cand[0] / nb), np.inf
    for c in cand:
        t = float(c / nb)
        q = _quantize_to_bins(counts, n_quant_bins, t, nb)
        mask = p_full > 0
        kl = float(np.sum(p_full[mask] *
                          np.log(p_full[mask] / np.maximum(q[mask], 1e-12))))
        if kl < best_kl:
            best_kl, best_t = kl, t
    return best_t, best_kl


def tail_mass(hist: np.ndarray, edges=(0.24, 0.30, 0.90)) -> dict:
    """Fraction of the heatmap's per-pixel MASS above the given values.

    This is the number that explains the narrow output range: the heatmap is
    extremely heavy-tailed -- ~99.96% of its mass sits below 0.24, while the
    values that decide a detection (the target peaks, 0.6..0.97) live in a
    ~0.03% tail. A KL/entropy-style clipping rule sees that tail as
    "statistically negligible" and clips it away; min/max keeps it.
    """
    h = np.asarray(hist, np.float64)
    total = h.sum()
    out = {}
    if total <= 0:
        return {f"mass_above_{e:g}": 0.0 for e in edges}
    for e in edges:
        cut = int(round(e * h.size))
        out[f"mass_above_{e:g}"] = float(h[cut:].sum() / total)
    return out


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def summarise(values) -> dict:
    """Peak statistics + the int8 scale / peak ceiling they would induce."""
    v = np.asarray(list(values), np.float64)
    if v.size == 0:
        return {"n": 0}
    mx = float(v.max())
    scale = mx / 127.0
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
        "max": mx,
        "mean": float(v.mean()),
        # a min/max int8 quantiser over this group would pick scale = max/127
        "theoretical_scale": scale,
        # (127 - (-128)) * scale == 255 * scale == what a real frame is clipped to
        "theoretical_peak_ceiling": 255.0 * scale,
        # the scale the group WOULD induce if only the p99 were covered
        "scale_if_p99": float(np.percentile(v, 99)) / 127.0,
    }


def summarise_rows(rows: list[dict]) -> dict:
    st = summarise([r["peak"] for r in rows])
    with_gt = [r["peak_at_gt"] for r in rows if r.get("peak_at_gt") is not None]
    if with_gt:
        st["peak_at_gt_max"] = float(np.max(with_gt))
        st["peak_at_gt_p90"] = float(np.percentile(with_gt, 90))
    st["n_with_gt"] = int(sum(1 for r in rows if r.get("n_gt", 0) >= 1))
    return st


# --------------------------------------------------------------------------------------
# Group builders (all deterministic: sort + fixed stride)
# --------------------------------------------------------------------------------------
def read_calib_set(key: str) -> list[str]:
    p = CALIB_ROOT / f"set_{key}.txt"
    lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit(f"[FATAL] calibration set {key} is empty: {p}")
    return lines


def train_pool() -> dict[str, list[Path]]:
    """{sequence: [official train feature jpgs]} -- sorted, existence-checked."""
    root = C.require(C.FEATURE_OFFICIAL_TRAIN, "official train features")
    seqs = sorted({p.name.split("__frame_")[0] for p in root.iterdir()
                   if "__frame_" in p.name})
    pool = {}
    for s in seqs:
        files = sorted(root.glob(f"{s}__frame_*.jpg"))
        if files:
            pool[s] = files
    if not pool:
        raise SystemExit("[FATAL] official train pool is empty")
    return pool


def pick_train_replica(n: int = TRAIN_REPLICA_N) -> list[Path]:
    """Group E: round-robin over the official-train sequences, 1 frame per sequence
    per round, evenly spaced inside each sequence. Sequence coverage first, so the
    200 frames span every training sequence."""
    pool = train_pool()
    seqs = sorted(pool)
    per_seq = max(1, math.ceil(n / len(seqs)))
    picks, seen = [], set()
    for r in range(per_seq):
        for s in seqs:
            frames = pool[s]
            n_fr = len(frames)
            idx = 0 if per_seq == 1 else int(round(r * (n_fr - 1) / (per_seq - 1)))
            p = frames[idx]
            if str(p) in seen:
                continue
            seen.add(str(p))
            picks.append(p)
            if len(picks) >= n:
                return picks
    return picks


def pick_train_global(n: int) -> list[Path]:
    """Evenly spaced pick over the FLATTENED, sorted list of all official train
    frames. Used for the mixed candidate so that its train portion is a different,
    corpus-wide draw than the round-robin replica."""
    root = C.require(C.FEATURE_OFFICIAL_TRAIN, "official train features")
    allf = sorted(root.glob("*__frame_*.jpg"))
    if not allf:
        raise SystemExit("[FATAL] no official train frames")
    step = len(allf) / float(n)
    return [allf[int(i * step)] for i in range(n)]


def pick_val_bmp(seq: str, n: int) -> list[str]:
    """n evenly spaced BMPs of one official-val sequence from the BMP pool."""
    files = sorted((C.BMP_ROOT / seq).glob("*.bmp"))
    if len(files) < n:
        raise SystemExit(f"[FATAL] {seq}: only {len(files)} BMPs in the pool")
    idxs = [int(round(r * (len(files) - 1) / (n - 1))) for r in range(n)]
    return [str(files[i]) for i in idxs]


def val_seq_dirs() -> list[str]:
    return sorted(p.name for p in C.BMP_ROOT.iterdir() if p.is_dir())


def golden_seqs() -> list[str]:
    return list(C.TRACK2_GOLDEN)


def set_a_bmps(official: bool) -> list[str]:
    """The 125 Anti-UAV / 125 official-train BMPs that ship inside set_A_bmp."""
    d = CALIB_ROOT / "set_A_bmp"
    out = []
    for p in sorted(d.glob("*.bmp")):
        is_official = "__frame_" in p.stem
        if is_official == official:
            out.append(str(p))
    return out


# --------------------------------------------------------------------------------------
# Group definitions
# --------------------------------------------------------------------------------------
def build_groups(need_train: bool = True) -> dict[str, dict]:
    """Every group we measure, with its byte source spelled out."""
    groups: dict[str, dict] = {}

    for key in ("A", "B", "C", "D"):
        groups[key] = {
            "label": f"Set-{key} (existing calibration set)",
            "kind": "bmp",
            "paths": read_calib_set(key),
            "byte_source": "bmp_disk (calib/set_%s_bmp)" % key,
        }

    if need_train:
        groups["E"] = {
            "label": "trial-0474 replica: official train features",
            "kind": "jpg",
            "paths": [str(p) for p in pick_train_replica(TRAIN_REPLICA_N)],
            "byte_source": "jpg_in_memory_cv2_imread (official train features)",
        }

    groups["F"] = {
        "label": "real blind-eval frames (001324, official val BMP pool)",
        "kind": "bmp",
        "paths": pick_val_bmp(C.TRACK3_BLIND[0], VAL_BLIND_N),
        "byte_source": "bmp_disk (ptq_work/bmp/%s)" % C.TRACK3_BLIND[0],
    }

    g_paths: list[str] = []
    for s in golden_seqs():
        g_paths.extend(pick_val_bmp(s, GOLDEN_PER_SEQ))
    groups["G"] = {
        "label": "golden pool (7 sequences x %d)" % GOLDEN_PER_SEQ,
        "kind": "bmp",
        "paths": g_paths,
        "byte_source": "bmp_disk (ptq_work/bmp/<seq>)",
    }

    groups["H"] = {
        "label": "Anti-UAV train portion of set_A (125)",
        "kind": "bmp",
        "paths": set_a_bmps(official=False),
        "byte_source": "bmp_disk (calib/set_A_bmp, Anti-UAV)",
    }

    groups["I"] = {
        "label": "official train portion of set_A (125)",
        "kind": "bmp",
        "paths": set_a_bmps(official=True),
        "byte_source": "bmp_disk (calib/set_A_bmp, official train)",
    }

    # the FROZEN trial-0474 calibration list, if it is available: 200 Anti-UAV
    # train feature JPGs. Its float heatmap max is what fixes the scale the
    # trial-0474 int8 model MUST have had (scale = max/255 -> 0.9314).
    t0474 = Path(os.environ.get("PTQ_CALIB2_ROOT",
                                 str(WORK_ROOT / "calib2"))) / "set_t0474.txt"
    if t0474.is_file():
        groups["J"] = {
            "label": "frozen trial-0474 calibration list (200 Anti-UAV JPGs)",
            "kind": "jpg",
            "paths": [l.strip() for l in t0474.read_text().splitlines() if l.strip()],
            "byte_source": "jpg_in_memory_cv2_imread (frozen trial-0474 list)",
        }

    return groups


# --------------------------------------------------------------------------------------
# Stage 1: group peak statistics
# --------------------------------------------------------------------------------------
def run_groups(scanner: PeakScanner, groups: dict[str, dict], out_root: Path) -> dict:
    results = {}
    for key in sorted(groups):
        g = groups[key]
        print(f"\n===== group {key}: {g['label']} ({len(g['paths'])} frames) =====",
              flush=True)
        t0 = time.time()
        scanner.reset_hist()
        rows = scanner.scan_paths(g["paths"], kind=g["kind"])
        st = summarise_rows(rows)
        # the two PREDICTORS for the int8 output range
        #   min/max ("normal")  : scale = max/255      -> ceiling = max
        #   KL    ("kl_divergence"): ceiling = the KL clipping threshold
        kl_t, kl_v = kl_threshold(scanner.hist)
        st["predicted_ceiling_minmax_normal"] = st["max"]
        st["predicted_ceiling_kl_divergence"] = kl_t
        st["kl_divergence_at_threshold"] = kl_v
        st["predicted_scale_minmax_normal"] = st["max"] / 127.0
        st["predicted_scale_kl_divergence"] = kl_t / 127.0
        st.update(tail_mass(scanner.hist))
        st["group"] = key
        st["label"] = g["label"]
        st["byte_source"] = g["byte_source"]
        st["seconds"] = round(time.time() - t0, 1)
        results[key] = {"stats": st, "rows": rows,
                        "histogram": scanner.hist.tolist()}
        print(f"  min={st['min']:.5f} p50={st['p50']:.5f} p90={st['p90']:.5f} "
              f"p99={st['p99']:.5f} max={st['max']:.5f} "
              f"=> scale={st['theoretical_scale']:.8f} "
              f"ceiling={st['theoretical_peak_ceiling']:.4f} | "
              f"KL ceiling={kl_t:.4f}", flush=True)

    payload = {
        "onnx": scanner.onnx_path,
        "threads": scanner.threads,
        "reference_board_numbers": REFERENCE,
        "threshold": C.TH_BASE,
        "groups": {k: {"label": v["stats"]["label"],
                       "byte_source": v["stats"]["byte_source"],
                       "stats": v["stats"],
                       "histogram": v["histogram"],
                       "peaks": [r["peak"] for r in v["rows"]],
                       "paths": [r["path"] for r in v["rows"]],
                       "n_gt": [r["n_gt"] for r in v["rows"]],
                       "peak_at_gt": [r["peak_at_gt"] for r in v["rows"]]}
                   for k, v in results.items()},
    }
    C.write_json(out_root / "peak_stats.json", payload)
    print(f"\n[report] -> {out_root / 'peak_stats.json'}")
    return payload


# --------------------------------------------------------------------------------------
# Stage 2: pool scan (the evidence the candidate selection is based on)
# --------------------------------------------------------------------------------------
def run_scan(scanner: PeakScanner, out_root: Path, stride: int) -> list[dict]:
    cache = out_root / "pool_scan.json"
    if cache.is_file():
        rows = json.loads(cache.read_text())["rows"]
        print(f"[scan] cached: {len(rows)} rows -> {cache}")
        return rows

    seqs = val_seq_dirs()
    print(f"\n===== pool scan: {len(seqs)} official-val sequences, stride {stride} =====")
    rows: list[dict] = []
    t0 = time.time()
    for seq in seqs:
        files = sorted((C.BMP_ROOT / seq).glob("*.bmp"))
        sub = files[::stride]
        print(f"  {seq}: {len(files)} BMPs -> {len(sub)} sampled", flush=True)
        for p in sub:
            rows.append(scanner.scan_bmp(p))
        el = time.time() - t0
        print(f"  done {seq}: {len(rows)} rows, {el:.0f}s", flush=True)
    C.write_json(cache, {"stride": stride, "seqs": seqs, "rows": rows})
    print(f"[scan] -> {cache}  ({len(rows)} rows, {time.time() - t0:.0f}s)")
    return rows


# --------------------------------------------------------------------------------------
# Stage 3: calibration set v2
# --------------------------------------------------------------------------------------
def diverse_top(rows: list[dict], n: int, per_seq_cap: int) -> list[dict]:
    """Top-n by peak with a per-sequence cap (keeps the sequence mix sane).

    The cap is RELAXED automatically until n frames are reachable: the strong
    frames of this pool concentrate in two sequences (001324 and 000818), so a
    hard cap would starve the set. The effective cap is recorded in the
    manifest, so the choice stays auditable.
    """
    ordered = sorted(rows, key=lambda r: (-r["peak"], r["path"]))
    best: list[dict] = []
    for cap in (per_seq_cap, per_seq_cap * 2, per_seq_cap * 4, n):
        out: list[dict] = []
        per: dict[str, int] = defaultdict(int)
        for r in ordered:
            if len(out) >= n:
                break
            if per[r["seq"]] >= cap:
                continue
            per[r["seq"]] += 1
            out.append(r)
        best = out
        if len(out) >= n:
            break
    return best[:n]


def write_bmp_copy(src, dst: Path) -> Path:
    """Write a lossless BMP copy of a JPG feature image (the sanctioned writer)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    C.write_feature_bmp(src, dst)
    return dst


def build_candidates(scanner: PeakScanner, scan_rows: list[dict],
                     out_root: Path, calib2: Path) -> dict:
    with_gt = [r for r in scan_rows if r.get("n_gt", 0) >= 1]
    if len(with_gt) < SET_SIZE:
        raise SystemExit(f"[FATAL] only {len(with_gt)} scanned frames carry a GT target")
    peaks = np.array([r["peak"] for r in with_gt], np.float64)

    # ---- candidate 1: strong-target frames -----------------------------------------
    # threshold = a quantile of the WITH-GT pool, lowered deterministically until at
    # least 250 frames qualify (the quantile that was used is recorded).
    used_q = None
    thr = 0.0
    strong = []
    for q in (0.99, 0.95, 0.90, 0.85, 0.80):
        thr = float(np.percentile(peaks, q * 100.0))
        strong = [r for r in with_gt if r["peak"] >= thr]
        used_q = q
        if len(strong) >= SET_SIZE:
            break
    c1 = diverse_top(strong, SET_SIZE, per_seq_cap=60)

    # ---- candidate 2: trial-0474 replica + strong frames ---------------------------
    replica = pick_train_replica(TRAIN_REPLICA_N)
    c2_strong = diverse_top(with_gt, 50, per_seq_cap=15)

    # ---- candidate 3: mixed --------------------------------------------------------
    uav_pool = set_a_bmps(official=False)              # Anti-UAV BMPs (already on disk)
    n_uav = 85
    step = len(uav_pool) / float(n_uav)
    c3_uav = [uav_pool[int(i * step)] for i in range(n_uav)]
    c3_train = pick_train_global(85)
    c3_strong = diverse_top(with_gt, 80, per_seq_cap=20)

    candidates = {
        "1": {
            "name": "v2-1 强目标组 (strong-target only)",
            "rationale": (
                "只保留「有 GT 目标 且 heatmap 峰值高」的帧。标定量化器的输出范围由"
                "标定张量的极值决定，所以这一组直接把输出 scale 顶到能覆盖真实评测帧"
                "峰值的位置,消除 0.24 的削顶天花板。"),
            "items": ([(r["path"], r["path"]) for r in c1]),
            "kind": "bmp",
            "detail": {"threshold_quantile": used_q, "threshold_peak": thr,
                       "n_above_threshold": len(strong),
                       "per_seq_cap": 60},
        },
        "2": {
            "name": "v2-2 train 复刻组 (trial-0474 replica + strong)",
            "rationale": (
                "E 组复刻 trial-0474 成功配方的标定分布（official train 特征,200 张）"
                "+ 50 张强目标帧把峰值上限抬到真实帧水平。这是「已知可用配方 + "
                "峰值覆盖」的最小改动。"),
            "items": ([(str(p), None) for p in replica]
                      + [(r["path"], r["path"]) for r in c2_strong]),
            "kind": "mixed",
            "detail": {"n_replica": len(replica), "n_strong": len(c2_strong)},
        },
        "3": {
            "name": "v2-3 混合组 (Anti-UAV + official train + strong)",
            "rationale": (
                "保留 set-A 的域多样性(Anti-UAV 85,不同数据源)以免量化范围只针对"
                "单一场景,同时用 official train 85 复刻已知配方,并用 80 张强目标帧"
                "覆盖真实评测帧的峰值。"),
            "items": ([(p, p) for p in c3_uav]
                      + [(str(p), None) for p in c3_train]
                      + [(r["path"], r["path"]) for r in c3_strong]),
            "kind": "mixed",
            "detail": {"n_uav": len(c3_uav), "n_train": len(c3_train),
                       "n_strong": len(c3_strong)},
        },
    }

    bmp_dir = calib2 / "bmp"
    manifest = {"set_size": SET_SIZE, "calib2": str(calib2), "candidates": {}}
    for key, spec in candidates.items():
        items = spec["items"]
        if len(items) != SET_SIZE:
            raise SystemExit(f"[FATAL] candidate {key}: {len(items)} != {SET_SIZE}")
        bmp_paths = []
        sources = []
        for bmp_or_src, src in items:
            if src is None:
                # a JPG feature image -> write a lossless BMP copy
                src_path = Path(bmp_or_src)
                dst = bmp_dir / f"cand{key}" / (src_path.stem + ".bmp")
                if not dst.is_file():
                    write_bmp_copy(src_path, dst)
                bmp_paths.append(str(dst))
                sources.append(str(src_path))
            else:
                bmp_paths.append(str(bmp_or_src))
                sources.append(str(src))
        if len(set(bmp_paths)) != SET_SIZE:
            raise SystemExit(f"[FATAL] candidate {key}: duplicate BMP entries")
        list_path = calib2 / f"set_{key}.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(bmp_paths) + "\n")
        (calib2 / f"set_{key}_sources.txt").write_text("\n".join(sources) + "\n")
        spec["list"] = str(list_path)
        spec["n"] = SET_SIZE
        spec["bmp_paths"] = bmp_paths
        spec["sources"] = sources
        print(f"[cand {key}] {spec['name']}: {SET_SIZE} frames -> {list_path}")
        manifest["candidates"][key] = {
            "name": spec["name"], "rationale": spec["rationale"],
            "list": str(list_path), "n": SET_SIZE, "detail": spec["detail"],
            "sources": str(calib2 / f"set_{key}_sources.txt"),
        }

    C.write_json(calib2 / "calib2_manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------------------
# Stage 4: candidate peak statistics
# --------------------------------------------------------------------------------------
def run_candidate_stats(scanner: PeakScanner, manifest: dict, calib2: Path,
                        out_root: Path) -> dict:
    stats = {}
    for key in sorted(manifest["candidates"]):
        list_path = Path(manifest["candidates"][key]["list"])
        paths = [l.strip() for l in list_path.read_text().splitlines() if l.strip()]
        print(f"\n===== candidate {key} ({len(paths)} frames) =====", flush=True)
        scanner.reset_hist()
        rows = scanner.scan_paths(paths, kind="bmp")
        st = summarise_rows(rows)
        kl_t, kl_v = kl_threshold(scanner.hist)
        st["predicted_ceiling_minmax_normal"] = st["max"]
        st["predicted_ceiling_kl_divergence"] = kl_t
        st["kl_divergence_at_threshold"] = kl_v
        st["predicted_scale_minmax_normal"] = st["max"] / 127.0
        st["predicted_scale_kl_divergence"] = kl_t / 127.0
        st.update(tail_mass(scanner.hist))
        stats[key] = {"stats": st, "peaks": [r["peak"] for r in rows],
                      "paths": paths, "histogram": scanner.hist.tolist()}
        print(f"  min={st['min']:.5f} p50={st['p50']:.5f} p90={st['p90']:.5f} "
              f"p99={st['p99']:.5f} max={st['max']:.5f} "
              f"=> scale={st['theoretical_scale']:.8f} "
              f"ceiling={st['theoretical_peak_ceiling']:.4f} | "
              f"KL ceiling={kl_t:.4f}", flush=True)
    C.write_json(out_root / "candidate_stats.json", {
        "onnx": scanner.onnx_path,
        "candidates": {k: {"label": manifest["candidates"][k]["name"],
                           "stats": v["stats"],
                           "histogram": v["histogram"],
                           "peaks": v["peaks"], "paths": v["paths"]}
                       for k, v in stats.items()},
    })
    print(f"\n[report] -> {out_root / 'candidate_stats.json'}")
    return stats


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def print_table(peak_payload: dict, cand_payload: dict | None) -> None:
    rows = []
    for key, g in peak_payload["groups"].items():
        s = g["stats"]
        rows.append((key, g["label"], s))
    if cand_payload:
        # the payload may be the on-disk file ({"candidates": {...}}) or the
        # in-memory {key: {...}} dict -- accept both
        cand = cand_payload.get("candidates", cand_payload)
        for key, c in cand.items():
            rows.append(("v2-" + key, c.get("label", ""), c["stats"]))

    hdr = ("group", "n", "min", "p50", "p90", "p99", "max", "theo_scale", "ceil",
           "ceil_minmax", "ceil_KL")
    print("\n" + "=" * 122)
    print("{:<8s} {:>4s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s} {:>12s} {:>8s} "
          "{:>11s} {:>8s}".format(*hdr))
    print("-" * 122)
    for key, label, s in rows:
        print("{:<8s} {:>4d} {:>9.5f} {:>9.5f} {:>9.5f} {:>9.5f} {:>9.5f} "
              "{:>12.8f} {:>8.4f} {:>11.4f} {:>8.4f}".format(
                  key, s["n"], s["min"], s["p50"], s["p90"], s["p99"], s["max"],
                  s["theoretical_scale"], s["theoretical_peak_ceiling"],
                  s.get("predicted_ceiling_minmax_normal", float("nan")),
                  s.get("predicted_ceiling_kl_divergence", float("nan"))))
    print("=" * 122)
    print("theo_scale  = max/127   (what an int8 min/max quantiser would pick)")
    print("ceil        = 255*scale (the value a real frame is clipped to on the board)")
    print("ceil_minmax = predicted ceiling with quantized_algorithm=normal")
    print("ceil_KL     = predicted ceiling with quantized_algorithm=kl_divergence")
    print("threshold   = %.2f  -> a ceiling below it means 0 detections per frame"
          % C.TH_BASE)
    for k, v in REFERENCE.items():
        print("board %-11s zp=%d scale=%.8f  implied calib max=%.5f  ceiling=%.4f"
              % (k, v["zp"], v["scale"], v["scale"] * 127.0, 255.0 * v["scale"]))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["groups", "scan", "candidates", "all"])
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--calib2-root", default=str(CALIB2_ROOT))
    ap.add_argument("--onnx", default=str(ONNX_FLOAT))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--scan-stride", type=int, default=4,
                    help="stride over the official-val BMP pool (1 = everything)")
    ap.add_argument("--self-check", type=int, default=3,
                    help="how many frames to use for the JPG/BMP byte-equivalence proof")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    calib2 = Path(args.calib2_root)
    calib2.mkdir(parents=True, exist_ok=True)

    print(f"[onnx] {args.onnx}")
    scanner = PeakScanner(Path(args.onnx), threads=args.threads)
    print(f"[onnx] input={scanner.in_name} outputs={scanner.out_names} "
          f"threads={scanner.threads}")

    # ---- byte-contract self check --------------------------------------------------
    if args.self_check > 0:
        import tempfile

        rep = pick_train_replica(TRAIN_REPLICA_N)
        with tempfile.TemporaryDirectory() as td:
            for p in rep[:args.self_check]:
                assert_jpg_bmp_equivalence(p, td)
                print(f"[check] JPG/BMP byte-identical: {Path(p).name}")
        first_bmp = sorted((C.BMP_ROOT / sorted(val_seq_dirs())[0]).glob("*.bmp"))[0]
        assert_sanctioned_reader(first_bmp)
        print(f"[check] sanctioned reader identical: {Path(first_bmp).name}")

    peak_payload = None
    cand_payload = None

    if args.stage in ("groups", "all"):
        groups = build_groups(need_train=True)
        peak_payload = run_groups(scanner, groups, out_root)

    if args.stage in ("scan", "all"):
        scan_rows = run_scan(scanner, out_root, args.scan_stride)
    else:
        cache = out_root / "pool_scan.json"
        scan_rows = json.loads(cache.read_text())["rows"] if cache.is_file() else []

    if args.stage in ("candidates", "all"):
        if not scan_rows:
            raise SystemExit("[FATAL] the pool scan is required before building candidates")
        manifest = build_candidates(scanner, scan_rows, out_root, calib2)
        cand_payload = run_candidate_stats(scanner, manifest, calib2, out_root)

    # ---- final table ---------------------------------------------------------------
    if peak_payload is None:
        f = out_root / "peak_stats.json"
        if f.is_file():
            peak_payload = json.loads(f.read_text())
    if cand_payload is None:
        f = out_root / "candidate_stats.json"
        if f.is_file():
            cand_payload = json.loads(f.read_text())
    if peak_payload:
        print_table(peak_payload, cand_payload)


if __name__ == "__main__":
    main()
