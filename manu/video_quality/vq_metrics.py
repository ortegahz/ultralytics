#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Per-frame video quality metrics for good/bad UAV capture clips.

Why the normalisation matters (the two clip groups are not comparable naively):
- *Resolution* differs (640x512 IR raw vs 1920x1080 mp4) -> sharpness/noise metrics are computed on
  a canonical 640 px wide crop, motion metrics are expressed as a fraction of the frame width.
- *Frame rate* differs (60 fps vs 25 fps vs 20 fps) -> motion metrics are additionally expressed per
  second.
- *Black pillarbox borders* are present in some clips. Their hard edges dominate Laplacian / gradient
  energy and would make a bad clip look artificially sharp, so the active content box is detected and
  cropped before any metric is computed.

Every metric is stored per frame; sequence level statistics use extreme quantiles (P5 for "worst
moment" quality, P95 for "worst case" motion) instead of the mean, because a bad clip is often bad
only during part of its length.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)  # never let OpenCV fan out over nproc (rules.md §〇之四)

CANON_WIDTH = 640
FLOW_WIDTH = 320
BLACK_THRESH = 16
RAW_SUFFIXES = {".h264", ".264", ".h265", ".hevc", ".265"}

# Per-frame metric -> "higher is worse" (1) / "lower is worse" (-1) / neutral (0).
METRIC_DIRECTION = {
    "shift_frac": 1,
    "shift_frac_max": 1,
    "shift_rate": 1,
    "rot_deg": 1,
    "rot_rate": 1,
    "scale_pct": 1,
    "flow_med_frac": 1,
    "flow_resid_frac": 1,
    "lap_var": -1,
    "edge_density": -1,
    "entropy": -1,
    "noise_sigma": 1,
    "snr_db": -1,
    "grad_energy_ratio": 1,
    "row_bias": 1,
    "col_bias": 1,
    "stripe_idx_col": 1,
    "stripe_idx_row": 1,
    "blockiness": 1,
    "global_contrast": -1,
    "sat_ratio": 1,
    "dark_ratio": 1,
    "illum_jump": 1,
    "dog_snr_max": -1,
    "dog_snr_top8": -1,
    "frame_mean": 0,
}

# Which quantile of the per-frame series is the "clip level" quality number.
# Motion / instability -> worst case (P95); clarity / contrast -> worst moment (P5).
METRIC_STAT = {
    "shift_frac": "p95",
    "shift_frac_max": "max",
    "shift_rate": "p95",
    "rot_deg": "p95",
    "rot_rate": "p95",
    "scale_pct": "p95",
    "flow_med_frac": "p95",
    "flow_resid_frac": "p95",
    "lap_var": "p05",
    "edge_density": "p05",
    "entropy": "p05",
    "noise_sigma": "p95",
    "snr_db": "p05",
    "grad_energy_ratio": "p95",
    "row_bias": "p95",
    "col_bias": "p95",
    "stripe_idx_col": "p95",
    "stripe_idx_row": "p95",
    "blockiness": "p95",
    "global_contrast": "p05",
    "sat_ratio": "p95",
    "dark_ratio": "p95",
    "illum_jump": "p95",
    "dog_snr_max": "p05",
    "dog_snr_top8": "p05",
    "frame_mean": "p50",
}

METRIC_LABEL_CN = {
    "shift_frac": "帧间位移(占画幅宽)",
    "shift_frac_max": "帧间最大位移(占画幅宽)",
    "shift_rate": "帧间位移速率(占画幅宽/秒)",
    "rot_deg": "帧间旋转(度/帧)",
    "rot_rate": "旋转速率(度/秒)",
    "scale_pct": "帧间缩放(%)",
    "flow_med_frac": "光流整体位移(占画幅宽)",
    "flow_resid_frac": "光流非刚性残差(占画幅宽)",
    "lap_var": "拉普拉斯清晰度方差",
    "edge_density": "边缘密度",
    "entropy": "灰度熵",
    "noise_sigma": "噪声 sigma",
    "snr_db": "图像信噪比(dB)",
    "grad_energy_ratio": "梯度能量比 rho",
    "row_bias": "行向非均匀性",
    "col_bias": "列向非均匀性",
    "stripe_idx_col": "列条纹指数",
    "stripe_idx_row": "行条纹指数",
    "blockiness": "块效应(压缩伪影)",
    "global_contrast": "全局对比度",
    "sat_ratio": "过曝像素比",
    "dark_ratio": "死黑像素比",
    "illum_jump": "帧间亮度跳变",
    "dog_snr_max": "最强小目标对比度(SNR)",
    "dog_snr_top8": "前8强小目标对比度(SNR)",
    "frame_mean": "平均亮度",
}

# Metrics that are not stored as their own per-frame series but derived from another one.
# ``shift_frac_max`` is the reference method's ``shift_max``: the worst frame-to-frame shift of a
# window instead of its P95.  Resolving it on the fly keeps a single source of truth (the stored
# ``shift_frac`` series) and needs no re-extraction of the per-frame arrays.
METRIC_SOURCE = {"shift_frac_max": "shift_frac"}


def metric_array(arrays: dict, metric: str) -> np.ndarray | None:
    """Per-frame series of ``metric``, resolving derived metrics onto their source series."""
    source = METRIC_SOURCE.get(metric)
    if source is None:
        return arrays.get(metric)
    return arrays.get(metric) if metric in arrays else arrays.get(source)


def probe_video(path: Path) -> dict:
    """Read stream geometry with ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,pix_fmt",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    stream = streams[0]

    def _rate(value: str) -> float:
        if not value or "/" not in value:
            return float(value or 0.0)
        num, den = value.split("/")
        den = float(den)
        return float(num) / den if den else 0.0

    fps = _rate(stream.get("r_frame_rate", "0")) or _rate(stream.get("avg_frame_rate", "0"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "nb_frames": int(stream.get("nb_frames") or 0),
        "duration": float(stream.get("duration") or 0.0),
        "codec": stream.get("codec_name", ""),
        "pix_fmt": stream.get("pix_fmt", ""),
    }


def open_frames(path: Path, width: int, height: int):
    """Yield grayscale frames decoded by ffmpeg (handles bare .h264 elementary streams)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if path.suffix.lower() in RAW_SUFFIXES:
        cmd += ["-f", "h264" if "264" in path.suffix.lower() else "hevc"]
    cmd += ["-threads", "1", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 24)
    frame_bytes = width * height
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
    finally:
        proc.stdout.close()
        proc.wait()


def detect_content_box(frame: np.ndarray, black_thresh: int = BLACK_THRESH) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) black border sizes of a single frame."""
    height, width = frame.shape
    row_max = frame.max(axis=1)
    col_max = frame.max(axis=0)

    def _run(mask: np.ndarray) -> tuple[int, int]:
        leading = 0
        for value in mask:
            if not value:
                break
            leading += 1
        trailing = 0
        for value in mask[::-1]:
            if not value:
                break
            trailing += 1
        return leading, trailing

    top, bottom = _run(row_max <= black_thresh)
    left, right = _run(col_max <= black_thresh)
    if (height - top - bottom) < 0.25 * height or (width - left - right) < 0.25 * width:
        return 0, 0, 0, 0
    return top, bottom, left, right


def noise_sigma_immerkaer(frame: np.ndarray) -> float:
    """Immerkaer noise estimate on a grayscale frame."""
    mean = cv2.blur(frame, (3, 3))
    mad = cv2.absdiff(frame, mean)
    height, width = frame.shape
    denominator = 6.0 * max(1, width - 2) * max(1, height - 2)
    return float(cv2.sumElems(mad)[0] * np.sqrt(np.pi / 2.0) / denominator)


def stripe_index(profile: np.ndarray) -> float:
    """Strength of the strongest periodic component of a 1-D profile (row or column means)."""
    centered = profile.astype(np.float64) - float(profile.mean())
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size < 4:
        return 0.0
    spectrum[0] = 0.0
    rest = spectrum[1:]
    return float(rest.max() / (rest.mean() + 1e-6))


def blockiness(frame: np.ndarray, block: int = 8) -> float:
    """Energy concentration on coding block borders (higher = stronger compression artefacts)."""
    height, width = frame.shape
    cols = np.arange(block, width, block)
    rows = np.arange(block, height, block)
    fh = cv2.absdiff(frame[:, cols], frame[:, cols - 1]) if cols.size else None
    fv = cv2.absdiff(frame[rows, :], frame[rows - 1, :]) if rows.size else None
    bh = float(fh.mean()) if fh is not None and fh.size else 0.0
    bv = float(fv.mean()) if fv is not None and fv.size else 0.0
    mean_h = float(cv2.absdiff(frame[:, 1:], frame[:, :-1]).mean()) + 1e-6
    mean_v = float(cv2.absdiff(frame[1:, :], frame[:-1, :]).mean()) + 1e-6
    return (bh / mean_h + bv / mean_v) / 2.0


def canonical(frame: np.ndarray, width: int = CANON_WIDTH) -> np.ndarray:
    """Resize to the canonical analysis width (INTER_AREA keeps thin structures when shrinking)."""
    if frame.shape[1] == width:
        return frame
    scale = width / frame.shape[1]
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def hanning_window(height: int, width: int) -> np.ndarray:
    """2-D Hanning window for phase correlation."""
    return np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)


def estimate_motion(prev_small: np.ndarray, cur_small: np.ndarray) -> tuple[float, float, float, float]:
    """Return (rot_deg, scale_pct, translation_frac, fit_residual_frac) from dense optical flow."""
    flow = cv2.calcOpticalFlowFarneback(prev_small, cur_small, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    height, width = flow.shape[:2]
    step = 16
    ys, xs = np.mgrid[step // 2 : height : step, step // 2 : width : step]
    xs = xs.astype(np.float32).ravel()
    ys = ys.astype(np.float32).ravel()
    cols = xs.astype(int).clip(0, width - 1)
    rows = ys.astype(int).clip(0, height - 1)
    dst = np.stack([xs + flow[rows, cols, 0], ys + flow[rows, cols, 1]], axis=1)
    src = np.stack([xs, ys], axis=1)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return 0.0, 0.0, 0.0, 0.0
    a, b, tx = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
    c, d, ty = float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])
    rotation = float(np.degrees(np.arctan2(c, a)))
    scale = float(np.sqrt(a * a + c * c))
    pred_x = a * xs + b * ys + tx
    pred_y = c * xs + d * ys + ty
    residual = float(np.median(np.hypot(dst[:, 0] - pred_x, dst[:, 1] - pred_y)) / width)
    return rotation, abs(scale - 1.0) * 100.0, float(np.hypot(tx, ty) / width), residual


def analyze_video(path: Path, stride: int = 1, max_frames: int = 0) -> tuple[dict, dict]:
    """Decode one clip and return (per-clip result, per-frame metric arrays)."""
    info = probe_video(path)
    width, height = info["width"], info["height"]
    frames = open_frames(path, width, height)

    crop: tuple[int, int, int, int] | None = None
    prev_canon = None
    prev_small = None
    per_frame: dict[str, list[float]] = {name: [] for name in list(METRIC_DIRECTION) + ["_pc_response"]}
    borders: list[tuple[int, int, int, int]] = []
    index = 0
    processed = 0

    for frame in frames:
        index += 1
        if stride > 1 and (index - 1) % stride:
            continue
        if crop is None:
            crop = detect_content_box(frame)
            borders.append(crop)
        else:
            borders.append(detect_content_box(frame))
        top, bottom, left, right = crop
        content = frame[top : height - bottom, left : width - right]
        if content.size == 0:
            content = frame

        canon = canonical(content)
        small_f = canonical(canon, FLOW_WIDTH).astype(np.float32)

        # --- sharpness / texture -------------------------------------------------
        lap_var = float(cv2.Laplacian(canon, cv2.CV_64F).var())
        edge_density = float((cv2.Canny(canon, 60, 160) > 0).mean())
        hist = cv2.calcHist([canon], [0], None, [256], [0, 256]).ravel()
        prob = hist / max(1.0, float(hist.sum()))
        nz = prob[prob > 0]
        entropy = float(-(nz * np.log2(nz)).sum())

        # --- noise / non-uniformity ---------------------------------------------
        sigma = noise_sigma_immerkaer(canon)
        mean_val = float(canon.mean()) + 1e-6
        snr_db = float(20.0 * np.log10(mean_val / (sigma + 1e-6)))
        grad_x = float(cv2.absdiff(canon[:, 1:], canon[:, :-1]).sum())
        grad_y = float(cv2.absdiff(canon[1:, :], canon[:-1, :]).sum())
        grad_energy_ratio = (grad_x + grad_y) / (float(canon.sum()) + 1e-6)
        row_profile = canon.mean(axis=1)
        col_profile = canon.mean(axis=0)
        row_bias = float(row_profile.std() / mean_val)
        col_bias = float(col_profile.std() / mean_val)
        stripe_row = stripe_index(row_profile)
        stripe_col = stripe_index(col_profile)
        blk = blockiness(content)

        # --- contrast / saturation ----------------------------------------------
        p1, p99 = np.percentile(canon, [1.0, 99.0])
        global_contrast = float((p99 - p1) / (p99 + p1 + 1e-6))
        sat_ratio = float((canon >= 250).mean())
        dark_ratio = float((canon <= 5).mean())

        # --- small target saliency (DoG response in noise sigma units) -----------
        dog = cv2.GaussianBlur(canon, (0, 0), 1.0).astype(np.float32) - cv2.GaussianBlur(
            canon, (0, 0), 4.0
        ).astype(np.float32)
        snr_map = dog / (sigma + 1e-6)
        dog_max = float(snr_map.max())
        top8 = np.partition(snr_map.ravel(), -8)[-8:]
        dog_top8 = float(top8.mean())

        # --- motion --------------------------------------------------------------
        shift_frac = 0.0
        rot_deg = 0.0
        scale_pct = 0.0
        flow_trans = 0.0
        flow_resid = 0.0
        illum_jump = 0.0
        pc_response = 0.0
        if prev_canon is not None:
            window = hanning_window(canon.shape[0], canon.shape[1])
            (dx, dy), pc_response = cv2.phaseCorrelate(
                prev_canon.astype(np.float32), canon.astype(np.float32), window
            )
            shift_frac = float(np.hypot(dx, dy) / canon.shape[1]) / stride
            rot_deg, scale_pct, flow_trans, flow_resid = estimate_motion(prev_small, small_f)
            rot_deg, scale_pct, flow_trans, flow_resid = (
                rot_deg / stride,
                scale_pct / stride,
                flow_trans / stride,
                flow_resid / stride,
            )
            illum_jump = abs(float(canon.mean()) - float(prev_canon.mean())) / (
                float(prev_canon.mean()) + 1e-6
            )
        prev_canon = canon
        prev_small = small_f

        per_frame["shift_frac"].append(shift_frac)
        per_frame["rot_deg"].append(rot_deg)
        per_frame["scale_pct"].append(scale_pct)
        per_frame["flow_med_frac"].append(flow_trans)
        per_frame["flow_resid_frac"].append(flow_resid)
        per_frame["lap_var"].append(lap_var)
        per_frame["edge_density"].append(edge_density)
        per_frame["entropy"].append(entropy)
        per_frame["noise_sigma"].append(sigma)
        per_frame["snr_db"].append(snr_db)
        per_frame["grad_energy_ratio"].append(grad_energy_ratio)
        per_frame["row_bias"].append(row_bias)
        per_frame["col_bias"].append(col_bias)
        per_frame["stripe_idx_row"].append(stripe_row)
        per_frame["stripe_idx_col"].append(stripe_col)
        per_frame["blockiness"].append(blk)
        per_frame["global_contrast"].append(global_contrast)
        per_frame["sat_ratio"].append(sat_ratio)
        per_frame["dark_ratio"].append(dark_ratio)
        per_frame["illum_jump"].append(illum_jump)
        per_frame["dog_snr_max"].append(dog_max)
        per_frame["dog_snr_top8"].append(dog_top8)
        per_frame["frame_mean"].append(float(canon.mean()))
        per_frame["_pc_response"].append(float(pc_response))

        processed += 1
        if max_frames and processed >= max_frames:
            break

    if not per_frame["shift_frac"]:
        raise RuntimeError(f"no frames decoded from {path}")

    fps = info["fps"] or 25.0
    arrays = {name: np.asarray(values, dtype=np.float32) for name, values in per_frame.items()}
    arrays["shift_rate"] = arrays["shift_frac"] * fps * stride
    arrays["rot_rate"] = arrays["rot_deg"] * fps * stride
    border_arr = np.asarray(borders, dtype=np.int32)
    border_median = np.median(border_arr, axis=0)
    border_dev = int(np.abs(border_arr - border_median).max())

    summary: dict[str, dict[str, float]] = {}
    for name in METRIC_DIRECTION:
        if name not in arrays:
            continue
        values = arrays[name]
        if name in {"shift_frac", "shift_rate", "rot_deg", "rot_rate", "illum_jump"}:
            values = values[1:]
        if values.size == 0:
            continue
        summary[name] = {
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    result = {
        "path": str(path),
        "info": info,
        "frames_decoded": int(index),
        "frames_analysed": int(processed),
        "stride": int(stride),
        "fps_used": fps,
        "crop": {"top": crop[0], "bottom": crop[1], "left": crop[2], "right": crop[3]},
        "content_size": [
            int(width - crop[2] - crop[3]),
            int(height - crop[0] - crop[1]),
        ],
        "border_deviation_px": border_dev,
        "summary": summary,
    }
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-frame video quality metrics from one clip")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="Output .npz for per-frame arrays")
    parser.add_argument("--json-out", type=str, default="", help="Optional JSON summary path")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.video)
    result, arrays = analyze_video(path, stride=args.stride, max_frames=args.max_frames)
    np.savez_compressed(args.out, **arrays)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
