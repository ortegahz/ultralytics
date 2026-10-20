#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render the system-SOTA final detections of one Longquan sequence as an OSD review video.

The panel carries exactly three things, and nothing else:

    red      one circle per FINAL fused system detection (solid dot r=3, ring r=7, label T<id>:<score>)
    green    the ground-truth box that a detection hit
    orange   the ground-truth box that nothing hit

Per-scale diagnostic lines are deliberately NOT drawn. The previous round shipped the blue (s640),
magenta (s160) and cyan (s80) rows next to the fused box and the reviewer read them as extra
detections -- "split into several small boxes plus one large false positive" -- even though those rows
never enter the FP statistics. They are gone from this renderer for good, so a red circle is always
one and only one final system detection.

Detection positions come from the system-SOTA prediction JSON (<preds-dir>/<sequence>.json); the ground
truth comes from the flat Final_Labels directory. Both are read through the helpers the business-domain
reviewer video already uses, so the TP/FP/FN counts printed here follow the very same inbox rule
(prediction centre inside the GT box, greedy by descending score, one GT absorbs at most one detection)
that produced the prediction JSON in the first place.

Usage:
    python manu/videos/render_sys_sota_longquan.py \
        --preds-dir runs/sys_sota_preds \
        --labels-root /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
        --frames-root /mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg \
        --seq VIDEO00009_19700101_001531 \
        --output /tmp/VIDEO00009_19700101_001531.mp4 \
        --out-json /tmp/VIDEO00009_19700101_001531.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The ground-truth reader, the inbox matcher, the box helper and the zoom panel are reused from the
# business-domain reviewer video on purpose: the prediction JSON was produced under the same inbox
# rule, so sharing the helpers is what keeps the counts printed here identical to that evaluation.
from manu.videos.render_pred_gt_osd_video import draw_box, load_ground_truth, match_frame, zoom_panel

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
COLOR_DET = (0, 0, 255)  # final system detection -- always red, hit or not


class FrameWriter:
    """Frame sink for the OSD video.

    Two back ends:

    * ``mp4v`` -- OpenCV's own writer, MPEG-4 part 2.  This is the historical default.
    * ``ffmpeg:<codec>`` -- the raw BGR frames are piped into the system ``ffmpeg`` binary, which is the
      only way to get H.264 here: the FFmpeg build that ships inside OpenCV has no H.264 *encoder*
      ("Could not find encoder for codec_id=27"), so ``avc1`` cannot be opened through ``cv2.VideoWriter``.
      Piping the raw frames means the file is encoded exactly once, with no generational loss, and it is
      far more portable across players (H.264/yuv420p + ``+faststart``).

    Both back ends expose the same ``write`` / ``release`` pair so the rendering loop does not care which
    one is in use.
    """

    def __init__(self, output: Path, fps: float, canvas: tuple[int, int], codec: str):
        self.output = Path(output)
        self.mode = codec
        self.process: subprocess.Popen | None = None
        self.writer: cv2.VideoWriter | None = None
        if codec.startswith("ffmpeg:"):
            encoder = codec.split(":", 1)[1]
            width, height = canvas
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-c:v", encoder, "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(self.output),
            ]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
            if self.process.stdin is None:
                raise RuntimeError("ffmpeg stdin pipe was not created")
        else:
            self.writer = cv2.VideoWriter(str(self.output), cv2.VideoWriter_fourcc(*codec), fps, tuple(canvas))
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot open {self.output}")

    def write(self, frame: np.ndarray) -> None:
        if self.process is not None:
            assert self.process.stdin is not None
            self.process.stdin.write(frame.tobytes())
        else:
            assert self.writer is not None
            self.writer.write(frame)

    def release(self) -> None:
        if self.process is not None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            code = self.process.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg exited with {code} while writing {self.output}")
        elif self.writer is not None:
            self.writer.release()
COLOR_GT_HIT = (0, 255, 0)  # ground truth that a detection hit
COLOR_GT_MISS = (0, 165, 255)  # ground truth that nothing hit
HUD_H = 56
PAD = 8
LEGEND = (
    ("RED=system final detection", COLOR_DET),
    ("GREEN=GT(hit)", COLOR_GT_HIT),
    ("ORANGE=GT(missed)", COLOR_GT_MISS),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the system-SOTA final detections of one sequence")
    parser.add_argument("--preds-dir", required=True, help="Directory of <sequence>.json prediction files")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-root", required=True, help="Flat Final_Labels directory")
    parser.add_argument("--frames-root", required=True, help="Root of <sequence>/frame_XXXXXX.jpg")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0, help="Inclusive last frame index; 0 means the last frame")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--canvas", default="1920x1080", help="Output canvas, for example 1920x1080")
    parser.add_argument("--zoom", type=int, default=360, help="Edge of the zoom panel; 0 disables it")
    parser.add_argument("--crop-size", type=int, default=48, help="Zoom crop edge in native pixels")
    parser.add_argument("--tol", type=float, default=8.0, help="Pixel tolerance shown in the HUD")
    parser.add_argument("--out-json", default="", help="Optional per-frame match JSON")
    parser.add_argument(
        "--codec",
        default="mp4v",
        help=(
            "Video back end: mp4v (OpenCV, MPEG-4 part 2, historical default) or ffmpeg:<encoder>, "
            "e.g. ffmpeg:libx264 for H.264 (OpenCV's bundled FFmpeg has no H.264 encoder, so it must "
            "go through the system ffmpeg binary)"
        ),
    )
    args = parser.parse_args()
    if not args.output.lower().endswith(".mp4"):
        parser.error("--output must end with .mp4")
    try:
        canvas_width, canvas_height = (int(value) for value in args.canvas.lower().split("x"))
    except ValueError:
        parser.error("--canvas must look like WIDTHxHEIGHT")
    if canvas_width < 640 or canvas_height < 480:
        parser.error("--canvas must be at least 640x480")
    if canvas_width % 2 or canvas_height % 2:
        parser.error("--canvas must have even width and height")
    args.canvas = (canvas_width, canvas_height)
    if args.zoom and (args.crop_size <= 0 or args.crop_size % 2):
        parser.error("--crop-size must be a positive even number when the zoom panel is enabled")
    return args


def load_predictions(path: Path) -> tuple[dict, dict]:
    """Read the system-SOTA prediction file into (config, per_frame) with native-pixel positions."""
    data = json.loads(path.read_text(encoding="utf-8"))
    per_frame: dict[int, list[dict]] = {}
    for key, rows in (data.get("per_frame") or {}).items():
        detections = []
        for row in rows or []:
            position = row.get("pos")
            if not position or len(position) < 2:
                continue
            detections.append(
                {
                    "pos": (float(position[0]), float(position[1])),
                    "score": float(row.get("score", 0.0)),
                    "track_id": int(row.get("track_id", 0)),
                    "infilled": bool(row.get("infilled", False)),
                }
            )
        per_frame[int(key)] = detections
    return data.get("config") or {}, per_frame


def write_text(canvas: np.ndarray, text: str, x: int, y: int, color, scale: float = 0.42) -> int:
    """Draw one text run and return the width it consumed."""
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return width


def draw_legend(canvas: np.ndarray, y: int) -> None:
    """Colour each legend segment, so the reviewer never has to guess which hue means what."""
    x = PAD
    for index, (text, color) in enumerate(LEGEND):
        if index:
            x += write_text(canvas, " | ", x, y, (170, 170, 170))
        x += write_text(canvas, text, x, y, color)


def main() -> None:
    args = parse_args()
    pred_path = Path(args.preds_dir) / f"{args.seq}.json"
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    pred_config, per_frame = load_predictions(pred_path)

    labels_root, frames_root = Path(args.labels_root), Path(args.frames_root)
    sequence_dir = frames_root / args.seq
    available = {
        int(path.stem.rpartition("_")[2]): path
        for path in sequence_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and path.stem.rpartition("_")[2].isdigit()
    }
    if not available:
        raise FileNotFoundError(f"No frames under {sequence_dir}")

    probe = cv2.imread(str(available[min(available)]), cv2.IMREAD_GRAYSCALE)
    if probe is None:
        raise RuntimeError(f"Undecodable frame {available[min(available)]}")
    native_h, native_w = probe.shape[:2]
    width = int(pred_config.get("img_w") or native_w)
    height = int(pred_config.get("img_h") or native_h)
    if (width, height) != (native_w, native_h):
        raise RuntimeError(f"prediction JSON native size {width}x{height} != frame size {native_w}x{native_h}")

    tol = float(args.tol)
    if "tol" in pred_config and float(pred_config["tol"]) != tol:
        print(f"[WARN] {args.seq}: --tol {tol} differs from the prediction JSON tol {pred_config['tol']}", flush=True)

    wanted = sorted(number for number in available if args.start <= number <= (args.end or 10**18))
    if args.limit:
        wanted = wanted[: args.limit]
    if not wanted:
        raise ValueError("No frames selected")
    truth = load_ground_truth(labels_root, args.seq, width, height, set(wanted))

    zoom = min(args.zoom, args.canvas[1] - HUD_H - 2 * PAD) if args.zoom else 0
    frame_area_w = args.canvas[0] - 2 * PAD - (zoom + PAD if zoom else 0)
    frame_area_h = args.canvas[1] - HUD_H - 2 * PAD
    factor = min(frame_area_w / width, frame_area_h / height)
    panel_size = (int(round(width * factor)), int(round(height * factor)))
    panel_x, panel_y = PAD, HUD_H + PAD

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FrameWriter(output, args.fps, tuple(args.canvas), args.codec)
    print(f"[INFO] codec={args.codec} -> {output}", flush=True)

    total_frames = len(wanted)
    totals = {"tp": 0, "fp": 0, "fn": 0, "det": 0}
    records: list[dict] = []
    cum_tp = cum_fp = cum_fn = 0
    print(
        f"[INFO] {args.seq} | frames={total_frames} | canvas={args.canvas[0]}x{args.canvas[1]} "
        f"frame_scale={factor:.3f} zoom={zoom} | preds={pred_path}",
        flush=True,
    )
    started = time.perf_counter()
    try:
        for number in tqdm(wanted, desc=f"{args.seq}", unit="frame", dynamic_ncols=True):
            gray = cv2.imread(str(available[number]), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape[:2] != (height, width):
                raise RuntimeError(f"Undecodable or mismatched frame {number}")

            boxes = truth.get(number)
            # Greedy by descending score: the strongest detection claims its ground truth first.
            detections = sorted(per_frame.get(number, []), key=lambda item: -item["score"])
            points = np.asarray([item["pos"] for item in detections], np.float32).reshape(-1, 2)
            scores = np.asarray([item["score"] for item in detections], np.float32)
            hits, missed = match_frame(boxes, points, scores)
            tp = int(hits.sum())
            fp = int(len(detections) - tp)
            fn = int(len(missed))
            missed_set = set(int(index) for index in missed)

            panel = cv2.cvtColor(
                cv2.resize(gray, panel_size, interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_GRAY2BGR,
            )

            # Ground truth first, so a detection marker stays readable on top of it.
            if boxes is not None:
                for index in range(len(boxes)):
                    hit = index not in missed_set
                    draw_box(
                        panel,
                        float(boxes[index][0]) * factor,
                        float(boxes[index][1]) * factor,
                        max(2.0, float(boxes[index][2]) * factor),
                        max(2.0, float(boxes[index][3]) * factor),
                        COLOR_GT_HIT if hit else COLOR_GT_MISS,
                        f"GT-{index + 1}" if hit else f"GT-{index + 1}[FN]",
                        thickness=2,
                    )

            # Final system detections: one red circle each, no per-scale diagnostic lines.
            label_slots: list[tuple[int, int]] = []
            for detection in detections:
                x = int(round(detection["pos"][0] * factor))
                y = int(round(detection["pos"][1] * factor))
                cv2.circle(panel, (x, y), max(2, int(round(3 * factor))), COLOR_DET, -1)
                cv2.circle(panel, (x, y), max(4, int(round(7 * factor))), COLOR_DET, 1)
                tag = "Infill" if detection["infilled"] else f"{detection['score']:.2f}"
                # The label sits below the marker: the ground-truth label lives above its box, and on a
                # small target the two would otherwise overlap into an unreadable smear. Detections that
                # share a position (a real one and the infill that filled its gap) get stacked labels.
                offset = 14
                while any(abs(slot_x - x) < 80 and abs(slot_y - (y + offset)) < 12 for slot_x, slot_y in label_slots):
                    offset += 13
                label_slots.append((x, y + offset))
                cv2.putText(
                    panel,
                    f"T{detection['track_id']}:{tag}",
                    (x + 6, y + offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    COLOR_DET,
                    1,
                    cv2.LINE_AA,
                )

            canvas = np.zeros((args.canvas[1], args.canvas[0], 3), np.uint8)
            canvas[0:HUD_H] = (20, 20, 20)
            canvas[panel_y : panel_y + panel.shape[0], panel_x : panel_x + panel.shape[1]] = panel

            if zoom:
                if boxes is not None and len(boxes):
                    centre = (float(boxes[0][0]), float(boxes[0][1]))
                    color = COLOR_GT_HIT if 0 not in missed_set else COLOR_GT_MISS
                elif detections:
                    centre = (float(detections[0]["pos"][0]), float(detections[0]["pos"][1]))
                    color = COLOR_DET
                else:
                    centre, color = (width / 2, height / 2), (120, 120, 120)
                patch = zoom_panel(gray, centre, args.crop_size, zoom, color)
                zoom_x = args.canvas[0] - PAD - zoom
                zoom_y = HUD_H + PAD
                canvas[zoom_y : zoom_y + zoom, zoom_x : zoom_x + zoom] = patch
                cv2.putText(
                    canvas,
                    f"ZOOM {args.crop_size}px",
                    (zoom_x, zoom_y + zoom + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            header = (
                f"SYS SOTA | {args.seq} | F:{number:06d}/{total_frames:06d} | tol={tol:.1f}px | "
                f"cum TP:{cum_tp} FP:{cum_fp} FN:{cum_fn} | frame TP:{tp} FP:{fp} FN:{fn}"
            )
            write_text(canvas, header, PAD, 21, (255, 255, 255), 0.46)
            draw_legend(canvas, 43)

            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            totals["det"] += len(detections)
            cum_tp += tp
            cum_fp += fp
            cum_fn += fn
            records.append({"frame": number, "tp": tp, "fp": fp, "fn": fn})
            writer.write(canvas)
    finally:
        try:
            writer.release()
        except Exception as exc:  # the original error, if any, is the one worth reporting
            print(f"[WARN] writer release failed: {exc}", flush=True)

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(
        f"[DONE] {args.seq} | rendered {total_frames} frames | TP={totals['tp']} FP={totals['fp']} "
        f"FN={totals['fn']} det={totals['det']} | Recall={recall * 100:.2f}% "
        f"Precision={precision * 100:.2f}% F1={f1 * 100:.2f}%",
        flush=True,
    )
    print(f"[SUCCESS] {output.resolve()} | elapsed {time.perf_counter() - started:.1f}s", flush=True)

    pred_metrics = json.loads(pred_path.read_text(encoding="utf-8")).get("metrics") or {}
    if pred_metrics:
        same = (int(pred_metrics.get("tp", -1)) == totals["tp"] and int(pred_metrics.get("fp", -1)) == totals["fp"]
                and int(pred_metrics.get("fn", -1)) == totals["fn"])
        print(
            f"[PRED] {args.seq} | prediction JSON metrics TP={pred_metrics.get('tp')} FP={pred_metrics.get('fp')} "
            f"FN={pred_metrics.get('fn')} | rendered range = "
            f"{'MATCH' if same else 'DIFFERENT (rendered frame range differs from the scored range)'}",
            flush=True,
        )

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {
                        "sequence": args.seq,
                        "pred_file": str(pred_path),
                        "preds_dir": str(Path(args.preds_dir)),
                        "labels_root": str(labels_root),
                        "frames_root": str(frames_root),
                        "frames": [wanted[0], wanted[-1]],
                        "frame_count": total_frames,
                        "fps": args.fps,
                        "canvas": list(args.canvas),
                        "native_size": [width, height],
                        "tol": tol,
                        "match_mode": "inbox",
                        "zoom": zoom,
                        "crop_size": args.crop_size,
                        "params": pred_config.get("params", {}),
                        "pred_metrics": pred_metrics,
                    },
                    "totals": totals,
                    "metrics": {"recall": recall, "precision": precision, "f1": f1},
                    "per_frame": records,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"[SUCCESS] JSON -> {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
