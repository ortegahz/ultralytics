#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Render a short consecutive-frame detection sequence as an mp4.

Why a video in addition to the contact-sheet PNG: the strip proves the
detections look right on individual frames, but the thing a UAV detector
actually has to do is hold a track across frames. The crop follows the GT of
each frame, so the target stays centred while the scene scrolls underneath it,
which is what makes a continuous sequence legible at all -- a 3-pixel UAV in a
640x512 frame is a handful of grey pixels.

Every frame is stamped with the source of each number (NPU vs CPU vs GT), and
the crop rectangle is drawn on the thumbnail so the viewer can see where the
zoom came from. If the NPU tensors are missing the video says so per frame
rather than silently showing CPU detections.

Usage:
    python hm_seq_video.py --ref <ref_dir> --npu <npu_dir> --npu-tag int8 \
        --frames <list.txt> --out out.mp4 [--zoom 60] [--fps 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.pipeline.npu.hm_osd import C_CPU, C_GT, C_NPU, C_TXT, load_npu  # noqa: E402
from manu.pipeline.npu.hm_common import (  # noqa: E402
    canvas_to_src,
    extract_peaks_np,
    letterbox_geometry,
    load_feature_jpg,
)

CANVAS = (1000, 600)  # W, H


def render_frame(ax_main, ax_thumb, ax_txt, rec, stem, npu, args):
    rgb = load_feature_jpg(Path(rec["file"]))
    geom = letterbox_geometry(rgb.shape[:2])
    ir = rgb[..., 0]
    gt = np.array(rec["gt_src"], np.float64).reshape(-1, 2)
    cpu_pts = np.array(rec["det_src"], np.float64).reshape(-1, 2)
    cpu_sc = np.array(rec["scores"], np.float64)

    npu_pts = np.zeros((0, 2))
    npu_sc = np.zeros((0,))
    if npu is not None:
        h, o = npu
        d = extract_peaks_np(h[None], o, stride=args.stride, conf_thresh=args.th)
        npu_pts = canvas_to_src(d["points"], geom)
        npu_sc = d["scores"]

    anchor = gt[0] if len(gt) else (npu_pts[0] if len(npu_pts) else (cpu_pts[0] if len(cpu_pts) else None))
    ax_main.clear()
    ax_thumb.clear()
    if anchor is None:
        ax_main.text(0.5, 0.5, "no GT and no detection", color=C_TXT, ha="center", va="center",
                     transform=ax_main.transAxes)
        ax_thumb.imshow(ir, cmap="gray", vmin=0, vmax=255)
        lines = ["no anchor: nothing to crop"]
    else:
        cx, cy = float(anchor[0]), float(anchor[1])
        z = args.zoom
        x0, x1 = int(max(0, cx - z)), int(min(ir.shape[1], cx + z))
        y0, y1 = int(max(0, cy - z)), int(min(ir.shape[0], cy + z))
        crop = ir[y0:y1, x0:x1]
        ax_main.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for pts, col in ((npu_pts, C_NPU), (cpu_pts, C_CPU), (gt, C_GT)):
            for p in pts:
                ax_main.plot(p[0] - x0, p[1] - y0, "o", mfc="none", mec=col, mew=1.8, ms=13)
        ax_main.set_xlim(0, x1 - x0)
        ax_main.set_ylim(y1 - y0, 0)
        ax_main.set_title(f"{stem}   crop {2 * z}px  @({cx:.1f}, {cy:.1f})", color=C_TXT, fontsize=11)

        ax_thumb.imshow(ir, cmap="gray", vmin=0, vmax=255)
        from matplotlib.patches import Rectangle
        ax_thumb.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=C_TXT, lw=1.2))
        ax_thumb.set_title("full frame", color=C_TXT, fontsize=9)

        lines = []
        if len(gt):
            lines.append(f"GT        x={gt[0][0]:8.2f}  y={gt[0][1]:8.2f}   score      -")
        if len(npu_pts):
            lines.append(f"NPU int8  x={npu_pts[0][0]:8.2f}  y={npu_pts[0][1]:8.2f}   score {npu_sc[0]:.4f}")
        if len(cpu_pts):
            lines.append(f"CPU fp32  x={cpu_pts[0][0]:8.2f}  y={cpu_pts[0][1]:8.2f}   score {cpu_sc[0]:.4f}")
        lines.append("")
        if len(npu_pts) and len(gt):
            lines.append(f"NPU -> GT distance = {np.hypot(npu_pts[0][0]-gt[0][0], npu_pts[0][1]-gt[0][1]):.2f} px "
                         f"(match tolerance 8.0 px)")
        if len(npu_pts) and len(cpu_pts):
            lines.append(f"NPU -> CPU distance = {np.hypot(npu_pts[0][0]-cpu_pts[0][0], npu_pts[0][1]-cpu_pts[0][1]):.2f} px")
        if len(npu_pts) == 0:
            lines.append("[RED] no NPU detection above threshold "
                         f"{args.th:.2f} in this frame")
        lines.append("")
        lines.append(f"letterbox scale={geom['scale']:.3f} padL={geom['pad_left']} padT={geom['pad_top']}")
        lines.append("all coordinates are SOURCE-IMAGE pixels")
    ax_txt.clear()
    ax_txt.axis("off")
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", color=C_TXT,
                fontsize=10, family="monospace", transform=ax_txt.transAxes)


def main():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--npu", default=None)
    ap.add_argument("--npu-tag", default="int8")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", type=int, default=60)
    ap.add_argument("--th", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--title", default="Trial 0474 on RK3588 NPU int8")
    args = ap.parse_args()

    import json
    recs = json.loads((Path(args.ref) / "detections.json").read_text())
    by_name = {Path(r["name"]).stem: r for r in recs}
    stems = [Path(l.strip()).stem for l in Path(args.frames).read_text().splitlines() if l.strip()]
    stems = [s for s in stems if s in by_name]
    if not stems:
        sys.exit("[FATAL] no frame in --frames matches the reference detections")

    npu_dir = Path(args.npu) if args.npu else None
    fig = plt.figure(figsize=(CANVAS[0] / 100, CANVAS[1] / 100), dpi=100)
    fig.patch.set_facecolor("#101014")
    ax_main = fig.add_axes([0.02, 0.06, 0.46, 0.86])
    ax_thumb = fig.add_axes([0.51, 0.58, 0.47, 0.34])
    ax_txt = fig.add_axes([0.51, 0.06, 0.47, 0.46])
    ax_main.set_axis_off()
    fig.suptitle(args.title, color="#ffffff", fontsize=13)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, CANVAS)
    if not vw.isOpened():
        sys.exit("[FATAL] cv2.VideoWriter failed to open (no mp4v encoder)")

    n_npu = 0
    for stem in stems:
        npu = load_npu(npu_dir, stem, args.npu_tag)
        n_npu += npu is not None
        render_frame(ax_main, ax_thumb, ax_txt, by_name[stem], stem, npu, args)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        vw.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
    vw.release()
    size = Path(args.out).stat().st_size
    print(f"[video] wrote {args.out}  {len(stems)} frames @ {args.fps:.0f} fps, "
          f"{size / 1e6:.2f} MB  (NPU tensors present for {n_npu}/{len(stems)})")


if __name__ == "__main__":
    main()
