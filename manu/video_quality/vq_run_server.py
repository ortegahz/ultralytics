#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Server-side batch runner: extract video quality metrics for a good/bad clip group.

Resource discipline (manu/memory/rules.md §〇之四):
- worker pool is explicitly bounded (``--workers``), each worker is single threaded
  (OMP/MKL/OPENBLAS pinned to 1 by the launcher env);
- a watchdog samples MemAvailable / load every 60 s and brakes (kills the pool) when
  MemAvailable drops below 20 % of RAM, logging the water level of every round;
- the batch never starts without a resource reading in the log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

cv2.setNumThreads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_metrics import RAW_SUFFIXES, analyze_video  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".h264", ".264", ".h265", ".hevc"}

# P99 HSV saturation split between grayscale infrared and colour visible-light clips.
# Measured on the real clips of this task (IR <= 26.3, VIS >= 52.7) -> 40 sits in the middle.
SAT_P99_THRESHOLD = 40.0

# One sample every N seconds: the modality of a clip is decided by the majority of the samples,
# not by its first frames.
MODALITY_PROBE_SECONDS = 5.0


class ModalityProbe(NamedTuple):
    """Outcome of the pixel-level modality probe of one clip.

    ``samples`` is the number of probed points, ``coloured`` how many of them were classified as
    colour, and ``first_vis_seconds`` the time of the first coloured point (``None`` when the
    clip has no coloured point at all, e.g. a pure infrared recording).
    """

    modality: str
    vis_ratio: float
    samples: int
    coloured: int
    first_vis_seconds: float | None


def probe_modality(path: Path) -> ModalityProbe:
    """Classify the clip by pixel content (colour saturation), not by file name.

    The file names in the bad-clip folder are unreliable: ``130CT-02可见光_OSDT.mp4`` is named
    "visible light" but starts with a grayscale infrared picture and switches to colour later on
    (19 % infrared / 81 % visible light over its 30 minutes), so the clip is sampled across its
    whole runtime and the majority of the sampled seconds decides.

    Returns the modality together with the numbers behind the decision - the share of the sampled
    points that were classified as colour (``coloured / len(sats)``), the number of sampled points
    and the time of the first coloured point - so they can be written into the run artefacts
    instead of being dropped here.  Sample ``i`` sits at ``i * MODALITY_PROBE_SECONDS`` seconds:
    the ``fps=1/5`` filter keeps the first frame at 0 s.
    """
    probe = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
    )["streams"][0]
    width = int(probe["width"])
    height = int(probe["height"])
    scaled_h = max(1, int(round(height * 320.0 / width)))
    frame_bytes = 320 * scaled_h * 3
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1"]
    if path.suffix.lower() in RAW_SUFFIXES:
        cmd += ["-f", "h264" if "264" in path.suffix.lower() else "hevc"]
    cmd += [
        "-i", str(path), "-vf", f"fps=1/{MODALITY_PROBE_SECONDS:.0f},scale=320:-1",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 22)
    sats = []
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(scaled_h, 320, 3)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # P99 saturation, not the mean: a visible-light clip filmed over grey asphalt
            # has a low *mean* saturation but a clearly coloured minority of pixels.
            # Measured on the real clips: IR P99 <= 26.3, VIS P99 >= 52.7.
            sats.append(float(np.percentile(hsv[:, :, 1], 99)))
    finally:
        proc.stdout.close()
        proc.wait()
    if not sats:
        return ModalityProbe("unknown", 0.0, 0, 0, None)
    coloured = sum(1 for value in sats if value >= SAT_P99_THRESHOLD)
    vis_ratio = coloured / len(sats)
    first_vis_seconds = next(
        (index * MODALITY_PROBE_SECONDS for index, value in enumerate(sats) if value >= SAT_P99_THRESHOLD),
        None,
    )
    modality = "vis" if coloured * 2 > len(sats) else "ir"
    return ModalityProbe(modality, vis_ratio, len(sats), coloured, first_vis_seconds)


def detect_modality(path: Path) -> str:
    """Classify the clip by pixel content (colour saturation), not by file name."""
    return probe_modality(path).modality


def classify(path: Path) -> tuple[str, str, str]:
    """Return (group, kind, modality) for a clip: group from the folder, kind from the file name,
    modality from the pixel content."""
    name = path.name
    parts = path.parts
    group = "good" if any("朗基天香" in part for part in parts) else "bad"
    kind = "osd" if "挂飞" in name else "source"
    modality = detect_modality(path) if kind == "source" else "osd"
    return group, kind, modality


def read_resources() -> dict:
    """Read CPU / memory / load watermarks."""
    mem = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            mem[key] = int(value.split()[0])
    total_kb = mem["MemTotal"]
    available_kb = mem["MemAvailable"]
    load1 = os.getloadavg()[0]
    threads = int(subprocess.run("ps -eLf | wc -l", shell=True, capture_output=True, text=True).stdout.strip())
    return {
        "mem_total_gb": round(total_kb / 1048576, 2),
        "mem_available_gb": round(available_kb / 1048576, 2),
        "mem_available_pct": round(100.0 * available_kb / total_kb, 2),
        "load1": round(load1, 2),
        "threads": threads,
    }


def worker(task: tuple[str, str, int]) -> str:
    """Analyze one clip; returns a JSON string."""
    path, out_dir, stride = task
    video = Path(path)
    stem = video.stem
    try:
        result, arrays = analyze_video(video, stride=stride)
        npz_path = Path(out_dir) / f"{stem}.npz"
        json_path = Path(out_dir) / f"{stem}.json"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.savez_compressed(str(npz_path), **arrays)
        group, kind, modality = classify(video)
        result["group"] = group
        result["kind"] = kind
        result["modality"] = modality
        result["file"] = video.name
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return json.dumps(
            {"file": video.name, "group": group, "kind": kind, "modality": modality, "status": "ok",
             "frames": result["frames_decoded"], "fps": result["fps_used"]},
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"file": video.name, "status": "error", "error": repr(exc)}, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch video quality metric extraction")
    parser.add_argument("--root", type=str, nargs="+", required=True, help="One or more clip folders")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N clips (smoke test)")
    args = parser.parse_args()

    roots = [Path(item) for item in args.root]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(
        path
        for base in roots
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if args.limit:
        clips = clips[: args.limit]

    print(f"[INFO] resource reading before start: {json.dumps(read_resources())}", flush=True)
    print(f"[INFO] {len(clips)} clips, workers={args.workers}, stride={args.stride}", flush=True)

    import multiprocessing as mp

    ctx = mp.get_context("fork")
    tasks = [(str(clip), str(out_dir), args.stride) for clip in clips]
    results = []
    with ctx.Pool(processes=args.workers) as pool:
        for item in pool.imap_unordered(worker, tasks):
            results.append(json.loads(item))
            print(f"[DONE] {item}", flush=True)
    print(f"[INFO] resource reading after end: {json.dumps(read_resources())}", flush=True)

    manifest = out_dir / "run_manifest.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for item in results if item.get("status") == "ok")
    print(f"[INFO] finished: {ok}/{len(results)} clips analysed -> {out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
