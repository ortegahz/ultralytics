#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Server-side batch runner for the uniform-grid spatial-trackability metric.

Resource discipline (manu/memory/rules.md §〇之四):
- the worker pool is explicitly bounded (``--workers``) and every worker is single threaded
  (the launcher pins OMP/MKL/OPENBLAS to 1 and ``cv2.setNumThreads(1)`` is set here too);
- the batch never starts without a resource reading in the log;
- every finished clip is written out immediately (JSON summary + per-cell NPZ + per-pair CSV),
  so an interrupted batch can be resumed without re-decoding the finished clips.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_grid import analyse_clip  # noqa: E402
from vq_run_server import VIDEO_SUFFIXES, classify, read_resources  # noqa: E402

ROW_SKIP = {"cells_raw_map", "cells_dog_map", "flat_map"}


def worker(task: tuple[str, str, str, int, int, int]) -> str:
    """Evaluate the grid for one clip; returns a JSON status line."""
    path, meta_dir, out_dir, gap, pairs_per_window, max_windows = task
    video = Path(path)
    stem = video.stem
    try:
        meta: dict = {}
        meta_path = Path(meta_dir) / f"{stem}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        group, kind, modality = classify(video)
        meta.setdefault("group", group)
        meta.setdefault("kind", kind)
        meta.setdefault("modality", modality)
        meta["file"] = video.name

        started = time.time()
        result, rows, arrays = analyse_clip(
            video, meta, gap=gap, pairs_per_window=pairs_per_window, max_windows=max_windows
        )
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(str(out / f"{stem}.npz"), **arrays)
        if rows:
            fields = [k for k in rows[0] if k not in ROW_SKIP]
            with (out / f"{stem}.rows.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        return json.dumps(
            {"file": video.name, "group": meta["group"], "kind": meta["kind"], "modality": meta["modality"],
             "status": "ok", "evaluations": result["evaluations"], "frames": result["frames_decoded"],
             "tgr_raw_p50": result["summary"]["tgr_raw_p50"], "tgr_dog_p50": result["summary"]["tgr_dog_p50"],
             "tgr_dog_best_block_p50": result["summary"]["tgr_dog_best_block_p50"],
             "flat_cells_p50": result["summary"]["flat_cells_p50"], "seconds": round(time.time() - started, 1)},
            ensure_ascii=False,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"file": video.name, "status": "error", "error": repr(exc)}, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch uniform-grid trackability extraction")
    parser.add_argument("--root", type=str, nargs="+", required=True, help="One or more clip folders")
    parser.add_argument("--meta-dir", type=str, required=True, help="Directory with the per-clip metric JSONs")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gap", type=int, default=5)
    parser.add_argument("--pairs-per-window", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
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
    print(f"[INFO] {len(clips)} clips, workers={args.workers}, gap={args.gap}, "
          f"pairs/window={args.pairs_per_window}", flush=True)

    import multiprocessing as mp

    ctx = mp.get_context("fork")
    tasks = [(str(clip), args.meta_dir, str(out_dir), args.gap, args.pairs_per_window, args.max_windows)
             for clip in clips]
    results = []
    with ctx.Pool(processes=args.workers) as pool:
        for item in pool.imap_unordered(worker, tasks):
            results.append(json.loads(item))
            print(f"[DONE] {item}", flush=True)
    print(f"[INFO] resource reading after end: {json.dumps(read_resources())}", flush=True)

    manifest = out_dir / "run_manifest.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for item in results if item.get("status") == "ok")
    print(f"[INFO] finished: {ok}/{len(results)} clips -> {out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
