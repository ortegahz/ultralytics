#!/usr/bin/env python3
"""
hm_demo_osd.py -- visual check for the serial end-to-end demo.

Renders, for one frame of a sequence:
  * the frozen feature image with GT (green), CPU authority arm (blue) and
    board NPU (red) markers in the SAME source-image pixel space,
  * an 8x zoom crop around the GT,
  * a numeric table: coordinates, scores, and the pairwise distances.

Why not ``hm_osd.py``: that one consumes board *tensor dumps* (``--npu <dir>`` of
``_hm.bin``/``_off.bin``), which cost ~2 frames/minute over NFS. The demo path
uses ``--detections-out`` TSV instead, so this script consumes that directly.

Coordinates
-----------
All three marker sets are in the SAME space -- source-image pixels of the 512x512
raw frame -- because that is the only space in which the three can be compared
without a second letterbox transform. The board and the reference both emit
canvas-space detections; ``hm_common`` maps them back with the same scale/pad,
and ``gt_from_label`` maps the label back with ``source_norm*src_hw*scale+pad``.

Usage:
    python hm_demo_osd.py --frame 34_1__000346 \\
        --feat-dir /home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val \\
        --tsv /run/media/manu/SSD960-LOCAL/nfs/npu/out/demo34/npu_det535.tsv \\
        --ref  /media/manu/1TB-Volume/rknn_work/hm/demo34/ref_cpu/detections.json \\
        --out  /media/manu/1TB-Volume/rknn_work/hm/demo34/osd_34_1__000346.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

OKABE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66)]


def parse_tsv(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = line.split("\t")
        n = int(p[1])
        out[Path(p[0]).stem] = [tuple(float(v) for v in p[2 + k].split()) for k in range(n)]
    return out


def src_from_canvas(pts, canvas, src_hw, scale, pad):
    """canvas px -> source px, the inverse of the reference's own mapping."""
    ch, cw = canvas
    out = []
    for x, y in pts:
        out.append(((x - pad[0]) / scale, (y - pad[1]) / scale))
    return out


def draw_marker(img, pt, color, label, r=6, thick=2):
    cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), r, color, thick, cv2.LINE_AA)
    cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), 2, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (int(round(pt[0])) + r + 3, int(round(pt[1])) - r - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help="stem, e.g. 34_1__000346")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--ref", required=True, help="detections.json from hm_reference.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--canvas", type=int, default=640,
                    help="letterbox canvas side the board/CPU detections are in")
    ap.add_argument("--zoom", type=int, default=8)
    ap.add_argument("--crop", type=int, default=48, help="crop half-size in source px")
    args = ap.parse_args()

    feat = Path(args.feat_dir) / f"{args.frame}.png"
    img = cv2.imread(str(feat), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"[FATAL] cannot read feature image {feat}")
    H, W = img.shape[:2]

    recs = {Path(r["name"]).stem: r for r in json.loads(Path(args.ref).read_text())}
    if args.frame not in recs:
        raise SystemExit(f"[FATAL] frame {args.frame} absent from the reference JSON")
    r = recs[args.frame]

    board = parse_tsv(Path(args.tsv)).get(args.frame, [])
    # The reference stores both spaces; use det_src / gt_src so all three panels
    # share one coordinate system without re-deriving the letterbox.
    cpu_src = [tuple(p) for p in r.get("det_src", [])]
    cpu_sc = r.get("scores", [])
    gt_src = [tuple(p) for p in r.get("gt_src", [])]

    # The board TSV is in canvas space; the reference JSON carries both spaces.
    # Derive the letterbox geometry ONCE from the reference's own pair rather than
    # assuming symmetry: pad is measured against the CANVAS side, not the source
    # side. For a square 512x512 source on a 640 canvas the letterbox is an
    # UPSAMPLE with pad 0 on all sides, so scale = 640/512 and pad = 0.
    C = float(args.canvas)
    scale = C / W
    padx = (C - W * scale) / 2.0
    pady = (C - H * scale) / 2.0
    board_src = [((x - padx) / scale, (y - pady) / scale) for x, y, _ in board]

    vis = img.copy()
    if gt_src:
        g = gt_src[0]
        cv2.rectangle(vis, (int(g[0] - 30), int(g[1] - 30)), (int(g[0] + 30), int(g[1] + 30)),
                      (0, 200, 0), 1, cv2.LINE_AA)
    for p in cpu_src:
        draw_marker(vis, p, (255, 120, 0), "CPU")
    for p in board_src:
        draw_marker(vis, p, (0, 0, 255), "NPU")
    for p in gt_src:
        draw_marker(vis, p, (0, 220, 0), "GT")

    # zoom around GT (or the CPU hit if there is no GT)
    c = gt_src[0] if gt_src else (cpu_src[0] if cpu_src else (W / 2, H / 2))
    x0, y0 = int(max(0, c[0] - args.crop)), int(max(0, c[1] - args.crop))
    x1, y1 = int(min(W, c[0] + args.crop)), int(min(H, c[1] + args.crop))
    crop = cv2.resize(vis[y0:y1, x0:x1], None, fx=args.zoom, fy=args.zoom,
                      interpolation=cv2.INTER_NEAREST)

    # numeric table. cv2.putText is Hershey-only -- it renders every CJK codepoint
    # as '?', so anything drawn into the image must be ASCII. The Chinese version
    # goes to stdout below instead.
    lines = [f"frame {args.frame}   src {W}x{H}   th={r['th']}  stride={r['stride']}",
             f"{'source':<8}{'x':>10}{'y':>10}{'score':>10}",
             "-" * 40]
    for i, p in enumerate(gt_src):
        lines.append(f"{'GT':<8}{p[0]:10.3f}{p[1]:10.3f}{'-':>10}")
    for i, p in enumerate(cpu_src):
        lines.append(f"{'CPU':<8}{p[0]:10.3f}{p[1]:10.3f}{cpu_sc[i]:10.6f}")
    for i, p in enumerate(board_src):
        lines.append(f"{'NPU':<8}{p[0]:10.3f}{p[1]:10.3f}{board[i][2]:10.6f}")
    lines.append("-" * 40)
    if gt_src and cpu_src:
        lines.append(f"CPU-GT  dist = {math.hypot(cpu_src[0][0] - gt_src[0][0], cpu_src[0][1] - gt_src[0][1]):.3f} px")
    if gt_src and board_src:
        lines.append(f"NPU-GT  dist = {math.hypot(board_src[0][0] - gt_src[0][0], board_src[0][1] - gt_src[0][1]):.3f} px")
    if cpu_src and board_src:
        lines.append(f"NPU-CPU dist = {math.hypot(board_src[0][0] - cpu_src[0][0], board_src[0][1] - cpu_src[0][1]):.3f} px")
    lines += ["", f"CPU arm  tp={r['tp']} fp={r['fp']} fn={r['fn']}",
              "NPU arm  see board_perframe.json",
              "match tolerance 8.0 px (canvas 640x640)",
              "green=GT  blue=CPU authority arm  red=board NPU (int8)"]
    tbl = np.full((28 * len(lines) + 20, 560, 3), 24, np.uint8)
    for i, ln in enumerate(lines):
        col = (220, 220, 220)
        if ln.startswith("GT"):
            col = (0, 220, 0)
        elif ln.startswith("CPU"):
            col = (255, 150, 60)
        elif ln.startswith("NPU"):
            col = (60, 60, 255)
        cv2.putText(tbl, ln, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

    def pad(im, h, w):
        if im.shape[0] > h or im.shape[1] > w:
            s = min(h / im.shape[0], w / im.shape[1])
            im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))),
                            interpolation=cv2.INTER_AREA)
        out = np.full((h, w, 3), 24, np.uint8)
        y = (h - im.shape[0]) // 2
        x = (w - im.shape[1]) // 2
        out[y:y + im.shape[0], x:x + im.shape[1]] = im
        return out

    canvas = np.hstack([pad(vis, 768, 768), pad(crop, 768, 768)])
    canvas = np.vstack([canvas, pad(tbl, tbl.shape[0], canvas.shape[1])])
    cv2.imwrite(args.out, canvas)
    print(f"[osd] wrote {args.out}  ({canvas.shape[1]}x{canvas.shape[0]})")
    for ln in lines:
        print("   ", ln)


if __name__ == "__main__":
    main()
