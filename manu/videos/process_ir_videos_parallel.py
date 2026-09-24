#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run the SOTA video processor concurrently and stream every video's live progress."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from collections import deque
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSOR = PROJECT_ROOT / "manu" / "videos" / "process_ir_video_sota.py"

FRAME_RE = re.compile(r"(\d+)frame\s*\[([^\]]*)\]")
RATIO_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[([^\]]*)\]")
FPS_RE = re.compile(r"([\d.]+)frame/s")
DET_RE = re.compile(r"detections=(\d+)")
TRK_RE = re.compile(r"tracks=(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Process many H.264 videos in parallel with live progress")
    parser.add_argument("--input-root", required=True, help="Directory containing input videos")
    parser.add_argument("--output-dir", default="runs/paper_diagnostic_videos")
    parser.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs, for example 0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=0, help="0 uses GPU count times workers-per-gpu")
    parser.add_argument("--pattern", default="*.h264")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos whose output already exists")
    parser.add_argument("--refresh", type=float, default=1.0, help="Status board refresh interval in seconds")
    parser.add_argument("--log-dir", default="", help="Per-video log directory (default: <output-dir>/logs)")
    parser.add_argument("--no-restore-native", action="store_true", help="Disable native 640x512 restoration")
    parser.add_argument("--processor-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


class StatusBoard:
    """Fixed-height terminal board that streams progress of every running video."""

    def __init__(self, title: str, order: list[str], total: int, refresh: float, log_dir: Path):
        self.title = title
        self.order = order
        self.total = total
        self.refresh = refresh
        self.log_dir = log_dir
        self.lock = threading.Lock()
        self.state: dict[str, dict] = {name: {"status": "queued", "frames": 0, "fps": 0.0, "message": "", "gpu": ""} for name in order}
        self.events: deque[str] = deque(maxlen=5)
        self.drawn = 0
        self.is_tty = sys.stdout.isatty()
        self.started = time.perf_counter()
        self.stop = threading.Event()

    def update(self, name: str, **fields):
        with self.lock:
            self.state.setdefault(name, {})
            self.state[name].update(fields)

    def event(self, text: str):
        with self.lock:
            self.events.append(f"{time.strftime('%H:%M:%S')} {text}")

    def _rows(self) -> list[str]:
        with self.lock:
            snapshot = {name: dict(fields) for name, fields in self.state.items()}
            events = list(self.events)
        done = sum(1 for fields in snapshot.values() if fields.get("status") == "done")
        failed = sum(1 for fields in snapshot.values() if fields.get("status") == "failed")
        running = sum(1 for fields in snapshot.values() if fields.get("status") == "running")
        queued = sum(1 for fields in snapshot.values() if fields.get("status") == "queued")
        elapsed = time.perf_counter() - self.started
        rows = [
            "=" * 110,
            f"{self.title} | total {self.total} | done {done} | running {running} | queued {queued} | failed {failed} | elapsed {elapsed/60:.1f}min",
            "-" * 110,
        ]
        for name in self.order:
            fields = snapshot.get(name, {})
            status = fields.get("status", "queued")
            if status in ("done", "failed"):
                continue
            frames = fields.get("frames", 0)
            fps = fields.get("fps", 0.0)
            detections = fields.get("detections", "-")
            tracks = fields.get("tracks", "-")
            gpu = fields.get("gpu", "")
            elapsed_video = fields.get("elapsed", 0.0)
            message = fields.get("message", "")
            rows.append(
                f"[GPU{gpu}] {name:<42} status={status:<8} frames={frames:<7} fps={fps:<6.2f} det={detections:<4} trk={tracks:<4} t={elapsed_video:>6.1f}s  {message}"
            )
        if self.log_dir:
            rows.append("-" * 110)
            rows.append(f"logs: {self.log_dir}")
        if events:
            rows.append("-" * 110)
            rows.extend(events)
        return rows

    def _flush(self, rows: list[str]):
        if not self.is_tty:
            sys.stdout.write("\n".join(rows) + "\n")
            sys.stdout.flush()
            self.drawn = 0
            return
        count = max(self.drawn, len(rows))
        buffer = []
        if self.drawn:
            buffer.append(f"\x1b[{self.drawn}A")
        for index in range(count):
            buffer.append("\x1b[2K")
            if index < len(rows):
                buffer.append(rows[index])
            buffer.append("\n")
        sys.stdout.write("".join(buffer))
        sys.stdout.flush()
        self.drawn = count

    def run(self):
        while not self.stop.wait(self.refresh):
            self._flush(self._rows())

    def stop_display(self):
        self.stop.set()

    def finalize(self):
        self._flush(self._rows())


def run_one(
    video_path: Path,
    output_dir: Path,
    weights: str,
    gpu: str,
    processor_args: list[str],
    board: StatusBoard,
    task_id: int,
):
    output_path = output_dir / f"{video_path.stem}_sota.mp4"
    log_path = board.log_dir / f"{video_path.stem}.log"
    command = [
        sys.executable,
        "-u",
        str(PROCESSOR),
        "--input",
        str(video_path),
        "--output",
        str(output_path),
        "--weights",
        weights,
        "--device",
        gpu,
        *processor_args,
    ]
    started = time.perf_counter()
    board.update(video_path.name, status="running", gpu=gpu, message="launching")
    board.event(f"[START] GPU{gpu} {video_path.name}")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                log_file.write(line + "\n")
                fields: dict = {"elapsed": time.perf_counter() - started, "message": line[:60]}
                ratio = RATIO_RE.search(line)
                if ratio:
                    fields["frames"] = int(ratio.group(1))
                    fields["total"] = int(ratio.group(2))
                else:
                    frame = FRAME_RE.search(line)
                    if frame:
                        fields["frames"] = int(frame.group(1))
                fps = FPS_RE.search(line)
                if fps:
                    fields["fps"] = float(fps.group(1))
                detections = DET_RE.search(line)
                if detections:
                    fields["detections"] = int(detections.group(1))
                tracks = TRK_RE.search(line)
                if tracks:
                    fields["tracks"] = int(tracks.group(1))
                board.update(video_path.name, **fields)
        return_code = process.wait()
    seconds = time.perf_counter() - started
    return {
        "task_id": task_id,
        "video": video_path.name,
        "output": output_path,
        "log": log_path,
        "gpu": gpu,
        "return_code": return_code,
        "seconds": seconds,
    }


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if not PROCESSOR.exists():
        raise FileNotFoundError(PROCESSOR)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU ID")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")

    videos = sorted(input_root.rglob(args.pattern) if args.recursive else input_root.glob(args.pattern))
    if args.skip_existing:
        videos = [video for video in videos if not (output_dir / f"{video.stem}_sota.mp4").exists()]
    if not videos:
        raise FileNotFoundError(f"No pending videos matching {args.pattern} under {input_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_restore_native:
        args.processor_args = ["--restore-native", *args.processor_args]
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    max_workers = args.max_workers or len(gpus) * args.workers_per_gpu
    max_workers = max(1, min(max_workers, len(videos)))

    board = StatusBoard(f"Batch: {input_root}", [video.name for video in videos], len(videos), args.refresh, log_dir)
    print(f"[INFO] Videos: {len(videos)} | GPUs: {','.join(gpus)} | Workers: {max_workers} | Logs: {log_dir.resolve()}", flush=True)
    display = threading.Thread(target=board.run, daemon=True)
    display.start()

    results, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future, tuple[Path, str]] = {}
        for index, video in enumerate(videos):
            gpu = gpus[index % len(gpus)]
            future = executor.submit(run_one, video, output_dir, args.weights, gpu, args.processor_args, board, index + 1)
            futures[future] = (video, gpu)
        for future in as_completed(futures):
            video, gpu = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"video": video.name, "gpu": gpu, "return_code": -1, "seconds": 0.0, "log": log_dir / f"{video.stem}.log", "error": str(error)}
            results.append(result)
            if result["return_code"] == 0:
                board.update(video.name, status="done", message="completed")
                board.event(f"[DONE] GPU{result['gpu']} {result['video']} {result['seconds']:.1f}s {result['output'].name}")
            else:
                board.update(video.name, status="failed", message=f"exit={result['return_code']}")
                failed.append(result)
                detail = result.get("error", f"exit={result['return_code']}")
                board.event(f"[FAIL] GPU{result['gpu']} {result['video']} {detail} -> {result['log']}")

    board.stop_display()
    display.join(timeout=2.0)
    board.finalize()
    print(f"[SUMMARY] Total={len(videos)} Done={len(results) - len(failed)} Failed={len(failed)} Logs={log_dir.resolve()}")
    if failed:
        print("[FAILED]")
        for result in failed:
            print(f"  GPU={result['gpu']} {result['video']} | log={result['log']}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
