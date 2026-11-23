#!/usr/bin/env python3
"""Batch driver for the click-initiated SOT probe (``vq_sot_link.py``).

Reads a job table (TSV: ``path \\t group \\t kind \\t modality``) and runs one clip per
worker process.  Each clip is decoded once, sequentially, with no seeking - the frame
index therefore lines up with ``report/windows.csv`` by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def run_one(job: tuple[str, str, str, str, str, dict]) -> str:
    """Run one clip inside a worker process; returns a one-line status."""
    path, group, kind, modality, out_dir, options = job
    target = Path(out_dir) / f"{Path(path).stem}.json"
    if target.exists():
        return f"[SKIP] {Path(path).name}"
    import vq_sot_link as probe

    cli = argparse.Namespace(
        input=path,
        input_root=None,
        pattern="*",
        out_dir=out_dir,
        group=group,
        kind=kind,
        modality=modality,
        short_frames=options["short_frames"],
        long_seconds=options["long_seconds"],
        clicks_per_epoch=options["clicks_per_epoch"],
        warmup_seconds=options["warmup_seconds"],
        start_frame=options["start_frame"],
        max_frames=options["max_frames"],
        threads=options["threads"],
        click_mode=options["click_mode"],
        ncc_thresh=options["ncc_thresh"],
        closure_thresh=options["closure_thresh"],
        limit=0,
    )
    started = time.perf_counter()
    result = probe.analyse_clip(cli, Path(path))
    target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    summary = result["summary"]
    last = "h%d" % summary["horizons"][-1]
    return (
        "[DONE] %s frames=%d windows=%d clicks=%d xcorr_%s=%.4f patch_%s=%.4f (%.1fs)"
        % (
            Path(path).name,
            summary["frames_decoded"],
            summary["windows"],
            summary["clicks"],
            last,
            summary["xcorr"][last]["lock_rate"],
            last,
            summary["patch"][last]["lock_rate"],
            time.perf_counter() - started,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch driver for the click-initiated SOT probe")
    parser.add_argument("--jobs", required=True, help="TSV: path \\t group \\t kind \\t modality")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--short-frames", type=int, default=5)
    parser.add_argument("--long-seconds", type=float, default=1.0)
    parser.add_argument("--clicks-per-epoch", type=int, default=16)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6, help="Threads inside one clip")
    parser.add_argument("--click-mode", default="alternating", choices=["alternating", "center", "corner"])
    parser.add_argument("--ncc-thresh", type=float, default=0.65)
    parser.add_argument("--closure-thresh", type=float, default=3.0)
    cli = parser.parse_args()

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "short_frames": int(cli.short_frames),
        "long_seconds": float(cli.long_seconds),
        "clicks_per_epoch": int(cli.clicks_per_epoch),
        "warmup_seconds": float(cli.warmup_seconds),
        "start_frame": int(cli.start_frame),
        "max_frames": int(cli.max_frames),
        "threads": int(cli.threads),
        "click_mode": str(cli.click_mode),
        "ncc_thresh": float(cli.ncc_thresh),
        "closure_thresh": float(cli.closure_thresh),
    }

    jobs = []
    for line in Path(cli.jobs).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        jobs.append((parts[0], parts[1], parts[2], parts[3], str(out_dir), options))

    print("[DRIVER] %d jobs, %d workers" % (len(jobs), cli.workers), flush=True)
    started = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            done += 1
            print("%s  [%d/%d, %.0fs]" % (future.result(), done, len(jobs), time.perf_counter() - started), flush=True)
    print("[DRIVER] finished %d jobs in %.1fs" % (len(jobs), time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    main()
