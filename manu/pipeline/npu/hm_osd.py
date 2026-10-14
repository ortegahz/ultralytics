#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Render OSD figures for the Trial-0474 heatmap detector on RK3588.

Two modes:
  --mode single     one frame, large IR panel with GT / NPU / CPU overlays,
                    the three feature channels, and the NPU heatmap
  --mode sequence   a strip of consecutive frames, one small panel each

Honesty rules baked in (rules.md 2c / 4.2d):
  * Every panel is labelled with WHERE its detections came from. A CPU/PyTorch
    panel is never drawn with NPU styling, and if the NPU tensors are missing the
    figure says so instead of quietly showing the CPU result.
  * Coordinates are printed as explicit `centre_x=` / `centre_y=` in the
    source image's own pixel frame, plus the letterbox canvas frame they were
    computed in, so no reader has to guess which of the two they are looking at.

Usage:
    python hm_osd.py --mode single --image <feature.jpg> --ref <ref_dir> \
                     --npu <npu_dir> --tag 01_4485_1167-2666__000106 --out out.png
    python hm_osd.py --mode sequence --ref <ref_dir> --npu <npu_dir> --out strip.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manu.pipeline.npu.hm_common import (  # noqa: E402
    canvas_to_src,
    extract_peaks_np,
    letterbox_geometry,
    load_feature_jpg,
    match_distance,
)

C_GT = "#00e676"     # ground truth
C_NPU = "#ff1744"    # RK3588 NPU
C_CPU = "#2979ff"    # CPU / PyTorch reference
C_TXT = "#e0e0e0"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval -- a bare rate with no interval is not a result."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def load_npu(npu_dir: Path | None, stem: str, tag: str = "int8"):
    """The board names its dumps "<tag>_<frame-stem>_hm.bin"; keep that in one place."""
    if not npu_dir:
        return None
    hm = npu_dir / f"{tag}_{stem}_hm.bin"
    off = npu_dir / f"{tag}_{stem}_off.bin"
    if not (hm.is_file() and off.is_file()):
        return None
    h = np.fromfile(hm, dtype=np.float32).reshape(320, 320)
    o = np.fromfile(off, dtype=np.float32).reshape(2, 320, 320)
    return h, o


def draw_points(ax, pts, scores, color, label, radius_px=7, lw=1.6, ms=4):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], s=ms**2, facecolors="none", edgecolors=color,
               linewidths=lw, zorder=5, label=label)
    if scores is not None and len(scores):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=color, zorder=6, marker="+", linewidths=1.0)
    # a ring makes the callout readable over both bright and dark background
    for p in pts:
        ax.add_patch(plt_circle(p, radius_px, color))


def plt_circle(p, r, color):
    from matplotlib.patches import Circle
    return Circle((p[0], p[1]), r, fill=False, edgecolor=color, lw=1.1, alpha=0.85, zorder=4)


def zoom_inset(ax, img, pts, color, pad=26, label="NPU"):
    """Magnified crop around the strongest detection -- targets are 3 px tall."""
    if not len(pts):
        return None
    p = np.asarray(pts, np.float64)
    cx, cy = float(p[0][0]), float(p[0][1])
    x0, x1 = int(max(0, cx - pad)), int(min(img.shape[1], cx + pad))
    y0, y1 = int(max(0, cy - pad)), int(min(img.shape[0], cy + pad))
    ins = ax.inset_axes([0.655, 0.60, 0.33, 0.33])
    ins.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ins.plot([cx - x0], [cy - y0], marker="+", ms=9, mew=1.4, color=C_NPU)
    ins.set_xlim(0, x1 - x0)
    ins.set_ylim(y1 - y0, 0)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color(color)
        s.set_linewidth(1.4)
    ins.set_title(f"x8 zoom @ ({cx:.0f},{cy:.0f})", fontsize=7, color=color, pad=2)
    return ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "sequence"], default="single")
    ap.add_argument("--image", help="feature JPEG (single mode)")
    ap.add_argument("--ref", required=True, help="hm_reference.py output dir")
    ap.add_argument("--npu", help="directory of board-dumped _hm.bin / _off.bin")
    ap.add_argument("--tag", help="frame stem (single mode)")
    ap.add_argument("--npu-tag", default="int8",
                    help="file prefix the board used when dumping (e.g. int8, fp16)")
    ap.add_argument("--frames", help="json or txt listing stems (sequence mode)")
    ap.add_argument("--th", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", type=int, default=0,
                    help="sequence mode: crop +/- N source px around the target (a 3 px UAV "
                         "is invisible in a full 640x512 panel)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--dist", type=float, default=8.0, help="GT match tolerance in px")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ref_dir = Path(args.ref)
    recs = json.loads((ref_dir / "detections.json").read_text())
    by_name = {Path(r["name"]).stem: r for r in recs}
    npu_dir = Path(args.npu) if args.npu else None
    have_npu = npu_dir is not None and any(npu_dir.glob("*_hm.bin"))

    if args.mode == "single":
        stem = args.tag or Path(args.image).stem
        rec = by_name[stem]
        img_path = Path(rec["file"]) if args.image is None else Path(args.image)
        rgb = load_feature_jpg(img_path)
        geom = letterbox_geometry(rgb.shape[:2])
        ir = rgb[..., 0]

        cpu_pts = np.array(rec["det_src"]).reshape(-1, 2)
        cpu_sc = np.array(rec["scores"])
        npu = load_npu(npu_dir, stem, args.npu_tag)
        npu_pts = np.zeros((0, 2))
        npu_sc = np.zeros((0,))
        hm_img = None
        if npu is not None:
            h, o = npu
            d = extract_peaks_np(h[None], o, stride=args.stride, conf_thresh=args.th)
            npu_pts = canvas_to_src(d["points"], geom)
            npu_sc = d["scores"]
            hm_img = h

        fig = plt.figure(figsize=(19.0, 9.6), dpi=115)
        fig.patch.set_facecolor("#101014")
        gs = GridSpec(3, 6, figure=fig, hspace=0.34, wspace=0.30,
                      left=0.030, right=0.985, top=0.905, bottom=0.05)

        # ---- big IR panel ---------------------------------------------------
        ax = fig.add_subplot(gs[:, 0:4])
        ax.imshow(ir, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        gt = np.array(rec["gt_src"]).reshape(-1, 2)
        if len(gt):
            ax.scatter(gt[:, 0], gt[:, 1], s=110, facecolors="none", edgecolors=C_GT,
                       linewidths=1.8, zorder=5, label=f"GT ({len(gt)})")
        draw_points(ax, cpu_pts, cpu_sc, C_CPU, "CPU / PyTorch", lw=1.4)
        draw_points(ax, npu_pts, npu_sc, C_NPU, "RK3588 NPU int8")
        ax.legend(loc="upper left", fontsize=8, facecolor="#181820", edgecolor="#444",
                  labelcolor=C_TXT, framealpha=0.9)
        ax.set_title(f"raw IR (feature channel 0)  {img_path.name}  {ir.shape[1]}x{ir.shape[0]} px",
                     color=C_TXT, fontsize=10)
        ax.tick_params(colors=C_TXT, labelsize=7)
        for s in ax.spines.values():
            s.set_color("#555")
        anchor = npu_pts if len(npu_pts) else cpu_pts
        zoom_inset(ax, ir, anchor, C_NPU if len(npu_pts) else C_CPU)

        # ---- heatmap --------------------------------------------------------
        axh = fig.add_subplot(gs[0, 4:6])
        if hm_img is not None:
            im = axh.imshow(hm_img, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
            # The heatmap panel is indexed by CELL, not by source pixel:
            # source -> canvas -> cell is  (src * scale + pad) / stride.
            # Multiplying by stride instead (as an earlier version did) put
            # every marker at the far edge of the image: 31/31 wrong.
            if len(anchor):
                hp = np.clip(np.round((anchor * geom["scale"]
                                       + [geom["pad_left"], geom["pad_top"]]) / args.stride), 0, 319).astype(int)
            else:
                hp = np.zeros((0, 2), int)
            if len(hp):
                axh.scatter(hp[:, 0], hp[:, 1], s=60, facecolors="none", edgecolors=C_NPU,
                            linewidths=1.2, zorder=5)
            cb = fig.colorbar(im, ax=axh, fraction=0.035, pad=0.03)
            cb.ax.tick_params(colors=C_TXT, labelsize=7)
            axh.set_anchor("N")
            axh.set_title(f"RK3588 NPU heatmap 320x320 (th={args.th}, stride={args.stride})",
                          color=C_TXT, fontsize=10)
        else:
            axh.text(0.5, 0.5, "NO NPU TENSORS\n(board run not available)", ha="center", va="center",
                     color=C_NPU, fontsize=13, transform=axh.transAxes)
            axh.set_title("RK3588 NPU heatmap -- unavailable", color=C_TXT, fontsize=10)
        axh.set_box_aspect(1)
        axh.tick_params(colors=C_TXT, labelsize=7)

        # ---- the three feature channels -------------------------------------
        for k, (axc, name) in enumerate(zip(
                [fig.add_subplot(gs[1, 4]), fig.add_subplot(gs[1, 5]), fig.add_subplot(gs[2, 4])],
                ["ch0  I_t  (raw IR gray)",
                 "ch1  |I_t - W(I_{t-2})|  (GMC short diff)",
                 "ch2  (I_t - B_t)^+  (21-frame median resid)"])):
            axc.imshow(rgb[..., k], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axc.set_title(name, color=C_TXT, fontsize=8)
            axc.tick_params(colors=C_TXT, labelsize=6)
            for s in axc.spines.values():
                s.set_color("#555")
            if len(gt):
                axc.scatter(gt[:, 0], gt[:, 1], s=45, facecolors="none", edgecolors=C_GT, linewidths=1.0)

        # ---- per-detection table -------------------------------------------
        axt = fig.add_subplot(gs[2, 5])
        axt.axis("off")
        lines = [f"{'source':<8}{'centre_x':>10}{'centre_y':>10}{'score':>9}", "-" * 40]
        for i in range(max(len(cpu_pts), len(npu_pts))):
            for tag, pts, sc, col in (("cpu", cpu_pts, cpu_sc, C_CPU), ("npu", npu_pts, npu_sc, C_NPU)):
                if i < len(pts):
                    lines.append(f"{tag:<8}{pts[i][0]:>10.2f}{pts[i][1]:>10.2f}{sc[i]:>9.4f}")
            if i < len(gt):
                lines.append(f"{'GT':<8}{gt[i][0]:>10.2f}{gt[i][1]:>10.2f}{'-':>9}")
        lines += ["", "coordinates: SOURCE-IMAGE pixels",
                  f"letterbox scale={geom['scale']:.3f} padL={geom['pad_left']} padT={geom['pad_top']}"]
        if len(gt) and len(anchor):
            dmin = np.hypot(anchor[:, 0] - gt[0][0], anchor[:, 1] - gt[0][1]).min()
            lines += ["", f"nearest det-to-GT = {dmin:.2f} px  (tol 8.0 px)"]
        axt.text(0, 1, "\n".join(lines), va="top", ha="left", color=C_TXT,
                 fontsize=7.0, family="monospace", transform=axt.transAxes)

        fig.suptitle(args.title or
                     "Trial 0474 heatmap detector -- single-frame detection on RK3588 NPU (int8)",
                     color="#ffffff", fontsize=13)
        fig.savefig(args.out, facecolor=fig.get_facecolor())
        print(f"[osd] wrote {args.out}")
        return

    # ---------------- sequence mode -----------------------------------------
    if args.frames:
        stems = [Path(l.strip()).stem for l in Path(args.frames).read_text().splitlines() if l.strip()]
    else:
        stems = [Path(r["name"]).stem for r in recs]
    stems = [s for s in stems if s in by_name]
    if args.max_frames:
        stems = stems[: args.max_frames]
    n = len(stems)
    cols = min(n, args.cols)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, (4.6 if args.zoom else 3.3) * rows),
                             dpi=105, squeeze=False)
    fig.patch.set_facecolor("#101014")
    n_npu = 0
    tp = fp = fn = 0
    for i, stem in enumerate(stems):
        ax = axes[i // cols][i % cols]
        rec = by_name[stem]
        rgb = load_feature_jpg(Path(rec["file"]))
        geom = letterbox_geometry(rgb.shape[:2])
        ir = rgb[..., 0]
        gt = np.array(rec["gt_src"]).reshape(-1, 2)
        cpu_pts = np.array(rec["det_src"]).reshape(-1, 2)
        npu = load_npu(npu_dir, stem, args.npu_tag)
        npu_pts = np.zeros((0, 2))
        npu_sc = np.zeros((0,))
        if npu is not None:
            n_npu += 1
            h, o = npu
            d = extract_peaks_np(h[None], o, stride=args.stride, conf_thresh=args.th)
            npu_pts = canvas_to_src(d["points"], geom)
            npu_sc = d["scores"]
            # Score the NPU detections against GT HERE. rec["tp"/"fp"/"fn"]
            # were written by hm_reference.py from the PyTorch arm, so reusing
            # them would print CPU numbers under an "NPU" label.
            arr = (np.concatenate([d["points"], d["scores"][:, None]], 1)
                   if len(d["points"]) else np.zeros((0, 3), np.float32))
            gt_c = np.array(rec["gt_canvas"], np.float64).reshape(-1, 2)
            t_, f_, n_ = match_distance(arr, gt_c, args.dist)
            tp += t_; fp += f_; fn += n_

        if args.zoom:
            # A 3-pixel UAV is invisible in a 640x512 panel, so crop around the
            # target. The crop centre is the GT when there is one, otherwise the
            # strongest detection -- the panel then shows the thing the detector
            # is actually being asked to find.
            anchor = gt[0] if len(gt) else (npu_pts[0] if len(npu_pts) else cpu_pts[0])
            z = args.zoom
            cx, cy = float(anchor[0]), float(anchor[1])
            x0, x1 = int(max(0, cx - z)), int(min(ir.shape[1], cx + z))
            y0, y1 = int(max(0, cy - z)), int(min(ir.shape[0], cy + z))
            ax.imshow(ir[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            if len(gt):
                ax.plot(gt[:, 0] - x0, gt[:, 1] - y0, "o", mfc="none", mec=C_GT, mew=1.8, ms=13)
            for pts, col in ((npu_pts, C_NPU), (cpu_pts, C_CPU)):
                for p in pts:
                    ax.plot(p[0] - x0, p[1] - y0, "o", mfc="none", mec=col, mew=1.5, ms=9)
            ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
            sc = f"  npu s={npu_sc.max():.3f}" if len(npu_sc) else ""
            ax.set_title(f"{stem.split('__')[-1].replace('.jpg','')}  crop {2*z}px @({cx:.0f},{cy:.0f}){sc}",
                         color=C_TXT, fontsize=8)
        else:
            ax.imshow(ir, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            if len(gt):
                ax.plot(gt[:, 0], gt[:, 1], "o", mfc="none", mec=C_GT, mew=1.8, ms=13)
            for pts, col in ((npu_pts, C_NPU), (cpu_pts, C_CPU)):
                for p in pts:
                    ax.plot(p[0], p[1], "o", mfc="none", mec=col, mew=1.4, ms=9)
            ax.set_title(f"{stem}   n={len(npu_pts) or len(cpu_pts)}", color=C_TXT, fontsize=8)
        ax.axis("off")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    lo, hi = wilson(tp, max(1, tp + fp))
    fig.suptitle(
        args.title or (f"Trial 0474 on RK3588 NPU -- {n} consecutive val frames\n"
                       f"green = GT   red = RK3588 NPU int8   blue = CPU/PyTorch   |   "
                       f"NPU TP={tp} FP={fp} FN={fn}  P={p:.3f} [95% CI {lo:.3f},{hi:.3f}] R={r:.3f}"),
        color="#ffffff", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(f"[osd] wrote {args.out}  ({n} frames, NPU tensors present for {n_npu})")


if __name__ == "__main__":
    main()
