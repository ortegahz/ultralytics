#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
ptq_common.py -- shared contracts for the RK3588 PTQ (post-training quantisation) sweep.

Everything that three consumers must agree on bit-for-bit lives here exactly once:

  1. 01_build_calibration_sets.py   -- builds the 4 calibration sets
  2. 02_batch_ptq_pc_screening.py   -- builds + PC-simulator-screens the RKNN models
  3. 03_make_fast_eval_manifest.py  -- builds the fast-eval manifests
  4. 04_run_board_fast_eval.py      -- drives the board and scores it against PT

--------------------------------------------------------------------------------------
THE PIXEL CONTRACT (this is the part that silently destroys PTQ numbers)
--------------------------------------------------------------------------------------
A feature image on disk is a 3-channel GMC+median-residual map. Two decoders are in
play and they are NOT interchangeable:

  * the PC  reads with cv2.imread()  -> returns BGR
  * the board reads with stbi_load() -> returns RGB

`stbi_load` of a file is exactly `reverse(cv2.imread(file))`. The training pipeline
(multiscale_heatmap_cache.py:369,385) does `cv2.imread(...)` then `[..., ::-1]`, i.e.
it feeds the model `reverse(cv2.imread(file))` -- the SAME bytes the board feeds the
NPU. So the two arms agree **iff** the BMP on disk was written from the array that
`cv2.imread` returns, i.e.:

    cv2.imwrite(bmp_path, cv2.imread(src_path))

After that round-trip:
    PC   : cv2.imread(bmp) -> BGR -> [...] -> model sees reverse(cv2.imread(bmp))
    board: stbi_load(bmp)  -> RGB = reverse(cv2.imread(bmp))              (identical)
    PNG/JPG in the middle would break this: their decoders differ between
    libjpeg-turbo / stb_image, so the two arms would see different pixels.
    => every file that crosses the PC/board boundary is BMP (lossless, exact).

`load_feature_bmp()` is the ONLY sanctioned reader, and `write_feature_bmp()` the
ONLY sanctioned writer. Never introduce a second convention.

--------------------------------------------------------------------------------------
THE NORMALISATION CONTRACT
--------------------------------------------------------------------------------------
RKNN config is mean_values=[[0,0,0]], std_values=[[255,255,255]] => the graph
consumes plain uint8 RGB and RKNN applies (x - 0)/255 internally. That is exactly
what PyTorch consumes (uint8/255), so the host and the board see the same numbers
and any difference downstream is attributable to the NPU alone.

The ONNX graph, by contrast, takes float32 in [0,1]. The simulator is fed uint8
NHWC (1,640,640,3) because the RKNN toolkit's simulator declares its input as
NHWC -- the same layout the board's stb_image path produces.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Paths (all overridable from the environment so the same code runs on the x86 host,
# the training server and the board-side staging box without editing)
# --------------------------------------------------------------------------------------
SERVER_ROOT = Path(os.environ.get("PTQ_SERVER_ROOT", "/tmp/pycharm_project_10ae9e2e"))
WEIGHTS = Path(os.environ.get(
    "PTQ_WEIGHTS", SERVER_ROOT / "runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt"))

# official semantic features == the distribution the network was trained on
# (official per-lag GMC fitting). FPV (`longquanshan_ir_gmc_median`) is a DIFFERENT
# semantics and must never be mixed into a calibration set -- see memory §五十四.
FEATURE_OFFICIAL_VAL = Path(os.environ.get(
    "PTQ_FEATURE_OFFICIAL_VAL",
    "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/val"))
FEATURE_OFFICIAL_TRAIN = Path(os.environ.get(
    "PTQ_FEATURE_OFFICIAL_TRAIN",
    "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train"))
FEATURE_UAV_VAL = Path(os.environ.get(
    "PTQ_FEATURE_UAV_VAL", "/mnt/data/siping/datasets/manu/uav_gmc_median/images/val"))
FEATURE_UAV_TRAIN = Path(os.environ.get(
    "PTQ_FEATURE_UAV_TRAIN", "/mnt/data/siping/datasets/manu/uav_gmc_median/images/train"))

GT_ROOT = Path(os.environ.get(
    "PTQ_GT_ROOT", "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"))

# The Anti-UAV track has its OWN label root.  Its files use exactly the same
# naming convention ("{seq}__{frame_index:06d}.txt") and exactly the same
# content format (one line "0 cx cy w h", YOLO-normalised) as the 龙泉山 labels,
# but they live in a different tree -- so the anti track must be pointed at
# this root instead of GT_ROOT, or every anti detection is scored as a FP.
ANTI_GT_ROOT = Path(os.environ.get(
    "PTQ_ANTI_GT_ROOT", "/mnt/data/siping/datasets/manu/uav_gmc_median/labels/val"))

ONNX_DIR = Path(os.environ.get("PTQ_ONNX_DIR", SERVER_ROOT / "runs/onnx"))
WORK_ROOT = Path(os.environ.get("PTQ_WORK_ROOT", "/mnt/data/siping/ptq_work"))
BMP_ROOT = Path(os.environ.get("PTQ_BMP_ROOT", WORK_ROOT / "bmp"))

# the board's NFS mount point (host export: /run/media/manu/SSD960-LOCAL/nfs)
# The staging root on the HOST (this workstation exports it over NFS). The board
# sees the very same tree at /mnt/manu/ptq, so a file at <BOARD_NFS>/bmp/<seq>/x.bmp
# is /mnt/manu/ptq/bmp/<seq>/x.bmp on the board. Every path that crosses the
# PC/board boundary is expressed relative to this root and re-anchored per side.
BOARD_NFS = Path(os.environ.get("PTQ_BOARD_NFS", "/run/media/manu/SSD960-LOCAL/nfs/ptq"))
BOARD_NFS_MOUNT = "/mnt/manu/ptq"

IMGSZ = 640
PAD_VALUE = 114
STRIDE = 2

# the frozen gate / delivery thresholds (manu/evaluation/system_sota_longquan_*.json)
TH_BASE = 0.3          # delivery (track) arm
TH_SINGLE = 0.22       # single-frame arm
TH_SALVAGE = 0.06      # region threshold of the multi-scale fusion
MAIN_THRESHOLD = 0.22  # main threshold of the multi-scale fusion
FUSION_DILATE = 32
MIN_AREA = 4
DIST_THR = 8.0         # px, greedy nearest-GT match radius

# multi-scale plan of the delivery pipeline (multiscale_heatmap_cache.py --scales 160,80)
SCALES = [0, 160, 80]

# --------------------------------------------------------------------------------------
# The two model variants under test
# --------------------------------------------------------------------------------------
#   p0_on  : trial_0080 with the P0 residual highway  (the original model)
#   p0_off : trial_0080 with P0 ablated                (the gate / business-domain口径:
#             auto_epoch_gate_check.py passes --ablate-p0 for ALL THREE tracks, and
#             run_trial0080_longquan_eval.sh does too -- so the numbers recorded in
#             memory are p0_off numbers and only p0_off can be compared against them)
VARIANTS = {
    "p0_on":  {"ablate_p0": False, "onnx": ONNX_DIR / "trial_0080_hm_p0.onnx",
               "onnx_u8": ONNX_DIR / "trial_0080_hm_p0_u8.onnx",
               "label": "trial_0080 (原始模型, 带 P0)"},
    "p0_off": {"ablate_p0": True,  "onnx": ONNX_DIR / "trial_0080_hm_nop0.onnx",
               "onnx_u8": ONNX_DIR / "trial_0080_hm_nop0_u8.onnx",
               "label": "trial_0080 --ablate-p0 (门禁/业务域口径)"},
}

# --------------------------------------------------------------------------------------
# The three tracks of the gate
# --------------------------------------------------------------------------------------
TRACK1_SEQS = None       # Anti-UAV val: flat directory, no sequence grouping
TRACK2_GOLDEN = [
    "VIDEO00006_19700101_000703",
    "VIDEO00007_19700101_000818",
    "VIDEO00008_19700101_001307",
    "VIDEO00009_19700101_001531",
    "VIDEO00029_19700101_014053",
    "VIDEO00033_19700101_015000",
    "VIDEO00034_19700101_015052",
]
TRACK3_BLIND = ["VIDEO00007_19700101_001324"]

# --------------------------------------------------------------------------------------
# BMP I/O -- the ONLY sanctioned pixel path (see the pixel contract above)
# --------------------------------------------------------------------------------------
def write_feature_bmp(src_path: str | Path, dst_path: str | Path) -> tuple[int, int]:
    """Decode a feature image and store it losslessly as a BMP the board can read.

    Returns (width, height). The bytes stored are `reverse(cv2.imread(src))` as seen
    through the file format, i.e. exactly what `stbi_load` hands to the NPU.
    """
    import cv2

    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read feature image: {src_path}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"feature image is not 3-channel: {src_path} shape={img.shape}")
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), img):  # BMP -> lossless, no JPEG in the loop
        raise OSError(f"cannot write {dst}")
    return img.shape[1], img.shape[0]


def load_feature_bmp(path: str | Path) -> np.ndarray:
    """Read a BMP the way the PC arm must read it: BGR, then reversed to model order.

    `load_feature_bmp(path)[..., ::-1]` is byte-identical to what the board's
    stbi_load() produces for the same file. Use `model_order_bmp()` directly.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read bmp: {path}")
    return img


def model_order_bmp(path: str | Path) -> np.ndarray:
    """HxWx3 uint8 in the model's channel order == the board's RGB byte order."""
    return load_feature_bmp(path)[..., ::-1]


def letterbox(img: np.ndarray, size: int = IMGSZ) -> np.ndarray:
    """Letterbox to size x size with 114 padding (mirrors ultralytics val path)."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)          # NOTE: no cap -- the loader upscales too
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (size - new_unpad[0]) / 2
    dh = (size - new_unpad[1]) / 2
    if new_unpad != (w, h):
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)


def letterbox_geometry(src_hw: tuple[int, int], size: int = IMGSZ) -> dict:
    h, w = src_hw
    r = min(size / h, size / w)
    left = int(round((size - int(round(w * r))) / 2 - 0.1))
    top = int(round((size - int(round(h * r))) / 2 - 0.1))
    return {"scale": r, "pad_left": left, "pad_top": top, "src_hw": (h, w), "size": size}


def canvas_to_src(points_xy: np.ndarray, geom: dict) -> np.ndarray:
    p = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2).copy()
    p[:, 0] = (p[:, 0] - geom["pad_left"]) / geom["scale"]
    p[:, 1] = (p[:, 1] - geom["pad_top"]) / geom["scale"]
    return p


def bmp_to_input_nhwc(path: str | Path, size: int = IMGSZ) -> np.ndarray:
    """BMP -> (1, H, W, 3) uint8, model channel order, letterboxed.

    Layout matches the board's NPU input (uint8 NHWC RGB).
    """
    rgb = model_order_bmp(path)
    canvas = letterbox(rgb, size)
    return np.ascontiguousarray(canvas[None])


def bmp_to_input_nchw_float(path: str | Path, size: int = IMGSZ) -> np.ndarray:
    """BMP -> (1, 3, H, W) float32 in [0,1]: exactly what the PyTorch model consumes."""
    nhwc = bmp_to_input_nhwc(path, size).astype(np.float32) / 255.0
    return np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2))


# --------------------------------------------------------------------------------------
# Peak extraction -- numpy port of the board's C++ extract_peaks (manu/pipeline/npu/hm_common.py)
# --------------------------------------------------------------------------------------
def _maxpool3x3(x: np.ndarray) -> np.ndarray:
    p = np.pad(x.astype(np.float32), ((1, 1), (1, 1)), mode="constant",
               constant_values=-np.inf)
    h, w = x.shape
    out = p[0:h, 0:w]
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out = np.maximum(out, p[dy:dy + h, dx:dx + w])
    return out


def extract_peaks_np(heatmap: np.ndarray, offset: np.ndarray, stride: int = STRIDE,
                     conf_thresh: float = TH_BASE, top_k: int = 100) -> dict:
    hm = np.asarray(heatmap, np.float32).reshape(heatmap.shape[-2], heatmap.shape[-1])
    off = np.asarray(offset, np.float32).reshape(2, *hm.shape)
    hmax = _maxpool3x3(hm)
    mask = (hmax == hm) & (hm >= conf_thresh)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {"points": np.zeros((0, 2), np.float32), "scores": np.zeros((0,), np.float32)}
    scores = hm[ys, xs]
    if scores.size > top_k:
        keep = np.argsort(-scores, kind="stable")[:top_k]
        keep = keep[np.argsort(keep, kind="stable")]
        ys, xs, scores = ys[keep], xs[keep], scores[keep]
    dx = off[0, ys, xs]
    dy = off[1, ys, xs]
    pts = np.stack([(xs.astype(np.float32) + dx) * stride,
                    (ys.astype(np.float32) + dy) * stride], axis=1)
    return {"points": pts.astype(np.float32), "scores": scores.astype(np.float32)}


def match_distance(dets: np.ndarray, gts: np.ndarray, dist_thr: float = DIST_THR):
    """Greedy nearest-GT matching at <= dist_thr. Returns (tp, fp, fn)."""
    if len(gts) == 0:
        return 0, len(dets), 0
    if len(dets) == 0:
        return 0, 0, len(gts)
    used = np.zeros(len(gts), bool)
    tp = 0
    for i in np.argsort(-dets[:, 2]):
        d = np.hypot(gts[:, 0] - dets[i, 0], gts[:, 1] - dets[i, 1])
        j = int(d.argmin())
        if d[j] <= dist_thr and not used[j]:
            used[j] = True
            tp += 1
    return tp, len(dets) - tp, int((~used).sum())


# --------------------------------------------------------------------------------------
# Sequence / frame naming
# --------------------------------------------------------------------------------------
def split_seq_frame(filename: str) -> tuple[str, int]:
    """'01_4485_1167-2666__000123.bmp' -> ('01_4485_1167-2666', 123)"""
    stem = Path(filename).stem
    if "__" in stem:
        seq, idx = stem.rsplit("__", 1)
        try:
            return seq, int(idx)
        except ValueError:
            return seq, -1
    return stem, -1


def official_feature_path(root: Path, seq: str, frame_index: int) -> Path:
    return root / f"{seq}__frame_{frame_index:06d}.jpg"


def seq_frames_from_official(root: Path, seq: str) -> list[int]:
    """Frame indices available for `seq` under an official-semantic feature root."""
    idx = []
    for p in root.glob(f"{seq}__frame_*.jpg"):
        tail = p.stem.rsplit("__frame_", 1)[1]
        try:
            idx.append(int(tail))
        except ValueError:
            continue
    return sorted(idx)


def gt_path(seq: str, frame_index: int, root=None) -> Path:
    """The GT label file of one frame.

    `root` overrides the global GT_ROOT for the tracks that carry their labels
    in a different tree (the Anti-UAV track -> ANTI_GT_ROOT).  Omitting it
    keeps the historical behaviour byte for byte.
    """
    return (Path(root) if root else GT_ROOT) / f"{seq}__{frame_index:06d}.txt"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def md5_file(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    return p


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def require(path: str | Path, what: str = "path") -> Path:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[FATAL] {what} does not exist: {p}")
    return p


def _sensitive_worker(pair):
    """Module level on purpose: a nested function cannot be pickled by the pool."""
    import cv2

    seq, fid = pair
    p = official_feature_path(FEATURE_OFFICIAL_VAL, seq, fid)
    gt = gt_path(seq, fid)
    try:
        n_gt = sum(1 for l in gt.read_text().splitlines() if len(l.split()) >= 5)
    except OSError:
        n_gt = 0
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = img[..., ::-1]
    return (seq, fid, n_gt, float(rgb[..., 2].astype(np.float32).mean()),
            float(rgb[..., 0].astype(np.float32).std()))


def pick_sensitive_frames(n_dark: int = 50, n_clutter: int = 50, seed: int = 20261022):
    """100 frames that carry the two failure modes of interest.

    The selection is driven by the GT (a frame with a target is a frame where a
    detection can actually be lost) plus two cheap feature statistics:
      * dark-weak   : a target IS present and the scene is in the darkest quartile
      * clutter     : a target IS present and the scene is in the busiest decile

    The thresholds are QUANTILES of a deterministic sample of the pool, not
    absolute levels: this dataset's raw IR sits around 96/255, so a fixed
    "<40 is dark" rule would select nothing. A 3,000-frame sample is enough to
    place the quartiles and is cheap (one parallel pass, ~1 s).
    """
    import multiprocessing as mp
    import random

    root = FEATURE_OFFICIAL_VAL
    rows = []
    for seq in TRACK3_BLIND + TRACK2_GOLDEN:
        for fid in seq_frames_from_official(root, seq):
            rows.append((seq, fid))
    if not rows:
        raise SystemExit("[FATAL] no official features found for the Tier-1 pool")

    rng = random.Random(seed)
    sample = rng.sample(rows, min(3000, len(rows)))

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=40) as pool:
        stats = [s for s in pool.map(_sensitive_worker, sample, chunksize=32) if s]
    if not stats:
        raise SystemExit("[FATAL] no usable frames for the Tier-1 pool")

    with_target = [s for s in stats if s[2] >= 1]
    if not with_target:
        raise SystemExit("[FATAL] no frame with a GT target in the Tier-1 pool")
    ir = np.array([s[3] for s in with_target])
    con = np.array([s[4] for s in with_target])
    q_ir = float(np.quantile(ir, 0.25))
    q_con = float(np.quantile(con, 0.90))
    print(f"[frames] pool={len(stats)} with_target={len(with_target)} "
          f"q_ir_25={q_ir:.2f} q_contrast_90={q_con:.2f}")

    dark = [s for s in with_target if s[3] <= q_ir]
    clutter = [s for s in with_target if s[4] >= q_con and s not in dark]
    # fill up from the remaining frames if a stratum is short
    for s in with_target:
        if len(dark) >= n_dark and len(clutter) >= n_clutter:
            break
        if len(dark) < n_dark and s not in dark and s not in clutter:
            dark.append(s)
        elif len(clutter) < n_clutter and s not in dark and s not in clutter:
            clutter.append(s)

    out = []
    for s in (dark[:n_dark] + clutter[:n_clutter]):
        seq, fid, n_gt, irm, con_ = s
        out.append({
            "seq": seq, "frame_id": fid,
            "bmp": str(BMP_ROOT / seq / f"{seq}__{fid:06d}.bmp"),
            "feature_src": str(official_feature_path(root, seq, fid)),
            "gt": str(gt_path(seq, fid)), "n_gt": n_gt,
            "ir_mean": irm, "contrast": con_,
        })
    if len(out) < n_dark + n_clutter:
        raise SystemExit(f"[FATAL] only {len(out)} sensitive frames could be selected")
    return out


# --------------------------------------------------------------------------------------
# Worker: build one configuration, screen it, write one JSON
# --------------------------------------------------------------------------------------


def selftest() -> None:
    """Cheap gate that must pass before a 9-minute build is started.

    Twelve configurations failed in a row once because of a single NameError in
    the letterbox helper. Every worker now runs this first: it is ~50 ms and it
    catches the whole class of "the helper was never exercised" mistakes.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    fake = rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)

    canvas = letterbox(fake)
    assert canvas.shape == (IMGSZ, IMGSZ, 3), canvas.shape
    assert canvas.dtype == np.uint8
    # the padding must be exactly the 114 constant
    assert canvas[0, 0, 0] == PAD_VALUE, canvas[0, 0, 0]

    geom = letterbox_geometry((512, 640))
    assert geom["scale"] == 1.0 and geom["pad_left"] == 0 and geom["pad_top"] == 64, geom
    back = canvas_to_src(np.array([[0.0, 64.0]]), geom)
    assert abs(back[0][1]) < 1e-6, back

    hm = rng.random((320, 320)).astype(np.float32)
    off = rng.random((2, 320, 320)).astype(np.float32)
    det = extract_peaks_np(hm, off, stride=STRIDE, conf_thresh=0.9, top_k=10)
    assert set(det) == {"points", "scores"}
    assert det["points"].shape[1] == 2

    # the peak decoder must find the single maximum
    hm2 = np.zeros((320, 320), np.float32)
    hm2[100, 200] = 0.9
    off2 = np.zeros((2, 320, 320), np.float32)
    d2 = extract_peaks_np(hm2, off2, stride=2, conf_thresh=0.5, top_k=10)
    assert len(d2["points"]) == 1, d2["points"]
    assert abs(d2["points"][0][0] - 400.0) < 1e-6, d2["points"]
    assert abs(d2["points"][0][1] - 200.0) < 1e-6, d2["points"]

    t, f, n = match_distance(np.array([[0.0, 0.0, 0.9]], np.float32),
                              np.array([[1.0, 1.0]], np.float32), 8.0)
    assert (t, f, n) == (1, 0, 0), (t, f, n)

    for v in VARIANTS.values():
        assert v["ablate_p0"] in (True, False)
    print("[selftest] ptq_common OK")


if __name__ == "__main__":
    selftest()
