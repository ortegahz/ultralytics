#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dispatch HM+BBox pseudo-label generation for all sequences across GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.process_ir_videos_parallel import RATIO_RE, FPS_RE, StatusBoard


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel pseudo-label generation across sequences")
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--yolo-batch-size", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1, help="OpenCV thread count per worker process")
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--sequences", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hm-weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    parser.add_argument("--bbox-weights", default="runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--hm-conf", type=float, default=0.06)
    parser.add_argument("--hm-top-k", type=int, default=50)
    parser.add_argument("--bbox-conf", type=float, default=0.06)
    parser.add_argument("--bbox-iou", type=float, default=0.70)
    parser.add_argument("--median-window", type=int, default=21)
    parser.add_argument("--temporal-stride", type=int, default=2)
    parser.add_argument("--coarse-min", type=float, default=6.0)
    parser.add_argument("--coarse-max", type=float, default=40.0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--limit-per-sequence", type=int, default=0)
    parser.add_argument("--include-bbox-only", action="store_true")
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--log-dir", default="")
    return parser.parse_args()


def build_worker_command(sequence: str, gpu: str, args) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "manu" / "preannotate_all_sequences.py"),
        "--frames-root", args.frames_root,
        "--output-root", args.output_root,
        "--split", args.split,
        "--sequences", sequence,
        "--worker",
        "--device", gpu,
        "--cpu-threads", str(args.cpu_threads),
        "--hm-weights", args.hm_weights,
        "--bbox-weights", args.bbox_weights,
        "--imgsz", str(args.imgsz),
        "--hm-conf", str(args.hm_conf),
        "--hm-top-k", str(args.hm_top_k),
        "--bbox-conf", str(args.bbox_conf),
        "--bbox-iou", str(args.bbox_iou),
        "--median-window", str(args.median_window),
        "--temporal-stride", str(args.temporal_stride),
        "--coarse-min", str(args.coarse_min),
        "--coarse-max", str(args.coarse_max),
        "--chunk-size", str(args.chunk_size),
        "--batch-size", str(args.batch_size),
        "--yolo-batch-size", str(args.yolo_batch_size),
    ]
    if args.limit_per_sequence:
        command += ["--limit-per-sequence", str(args.limit_per_sequence)]
    if args.include_bbox_only:
        command.append("--include-bbox-only")
    if args.no_copy:
        command.append("--no-copy")
    return command


def run_worker(sequence: str, gpu: str, args, board: StatusBoard) -> dict:
    log_path = board.log_dir / f"{sequence}.log"
    command = build_worker_command(sequence, gpu, args)
    started = time.perf_counter()
    board.update(sequence, status="running", gpu=gpu, message="launching")
    board.event(f"[START] GPU{gpu} {sequence}")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(args.cpu_threads),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
        "MKL_NUM_THREADS": str(args.cpu_threads),
    }
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                log_file.write(line + "\n")
                fields: dict = {"elapsed": time.perf_counter() - started, "message": line[:70]}
                ratio = RATIO_RE.search(line)
                if ratio:
                    fields["frames"] = int(ratio.group(1))
                fps = FPS_RE.search(line)
                if fps:
                    fields["fps"] = float(fps.group(1))
                if line.startswith("[DONE]"):
                    fields["counts"] = line
                board.update(sequence, **fields)
        return_code = process.wait()
    return {"sequence": sequence, "gpu": gpu, "return_code": return_code, "seconds": time.perf_counter() - started, "log": log_path}


def main():
    args = parse_args()
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    if not frames_root.is_dir():
        raise NotADirectoryError(frames_root)
    if output_root.resolve() == frames_root.resolve() or frames_root.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("Refusing to overwrite frames-root with output-root")

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU ID")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")

    sequence_filter = {item.strip() for item in args.sequences.split(",") if item.strip()}
    sequences = [path.name for path in sorted(frames_root.iterdir()) if path.is_dir() and (not sequence_filter or path.name in sequence_filter)]
    if not sequences:
        raise FileNotFoundError(f"No sequences under {frames_root}")
    if args.overwrite:
        print(f"[WARN] Removing existing {output_root / 'images'}, {output_root / 'labels'}, {output_root / 'summaries'}", flush=True)
        shutil.rmtree(output_root / "images", ignore_errors=True)
        shutil.rmtree(output_root / "labels", ignore_errors=True)
        shutil.rmtree(output_root / "summaries", ignore_errors=True)
        for stale_file in [output_root / "data.yaml", output_root / "manifest.json", *output_root.glob("provenance_*.json")]:
            stale_file.unlink(missing_ok=True)
    (output_root / "images" / args.split).mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / args.split).mkdir(parents=True, exist_ok=True)
    (output_root / "summaries").mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.workers_per_gpu > 2:
        print(f"[WARN] workers-per-gpu={args.workers_per_gpu} may exhaust VRAM; 1-2 is recommended", flush=True)
    max_workers = args.max_workers or len(gpus) * args.workers_per_gpu
    max_workers = max(1, min(max_workers, len(sequences)))
    board = StatusBoard(f"Pseudo-labels: {frames_root.name}", sequences, len(sequences), args.refresh, log_dir)
    print(f"[INFO] Sequences: {len(sequences)} | GPUs: {','.join(gpus)} | Workers: {max_workers} | Logs: {log_dir.resolve()}", flush=True)
    display = threading.Thread(target=board.run, daemon=True)
    display.start()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, sequence in enumerate(sequences):
            gpu = gpus[index % len(gpus)]
            futures[executor.submit(run_worker, sequence, gpu, args, board)] = sequence
        for future in as_completed(futures):
            sequence = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"sequence": sequence, "gpu": "", "return_code": -1, "seconds": 0.0, "log": log_dir / f"{sequence}.log", "error": str(error)}
            results.append(result)
            if result["return_code"] == 0:
                board.update(sequence, status="done", message="completed")
                board.event(f"[DONE] {sequence} {result['seconds']:.0f}s")
            else:
                board.update(sequence, status="failed", message=f"exit={result['return_code']}")
                failed.append(result)
                board.event(f"[FAIL] {sequence} -> {result['log']}")
    board.stop_display()
    display.join(timeout=2.0)
    board.finalize()

    global_counts = {0: 0, 1: 0, 2: 0}
    summaries = []
    for summary_path in sorted((output_root / "summaries").glob("*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        for key, value in summary.get("counts", {}).items():
            global_counts[int(key)] += int(value)

    names = {0: "uav_bbox", 1: "uav_hm_coarse"}
    if args.include_bbox_only:
        names[2] = "uav_bbox_only"
    (output_root / "data.yaml").write_text(
        f"path: {output_root.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n"
        + "\n".join(f"  {key}: {value}" for key, value in names.items())
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "frames_root": str(frames_root.resolve()),
        "output_root": str(output_root.resolve()),
        "split": args.split,
        "sequence_count": len(summaries),
        "sequences": summaries,
        "global_counts": global_counts,
        "failed": [item["sequence"] for item in failed],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUMMARY] sequences={len(summaries)} c0={global_counts[0]} c1={global_counts[1]} c2={global_counts[2]} failed={len(failed)}")
    if failed:
        for item in failed:
            print(f"  FAILED {item['sequence']} -> {item['log']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
