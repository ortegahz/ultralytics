#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
🔴🔴🔴 【已作废 · 禁止运行】DEPRECATED — DO NOT RUN. 逻辑不成立，整版数据作废。

**作废原因（2026-11-20 用户纠偏）**：本脚本拿**反无人机检测器**去跑**没有无人机**的测试视频。
- 好片测出的「连续航迹」其实是**背景虚警**（检测器把楼角/路灯/树梢当成疑似弱小目标，
  地物纹理清晰 ⇒ 运动补偿后被时空关联当成连续轨迹）⇒ 测到的不是「目标可跟踪性」，是**虚警率**；
- 坏片「零航迹」只是因为**糊图把背景锐利边缘抹平、连虚警都触发不了**（0.167%）
  ⇒ 「零航迹」**不能**证明「画质差 ⇒ 跟不住」；
- 真实业务交互是**操作员在画面任意位置点一下 → SOT 锁住局部特征块**，
  「检测器报不报候选」与「用户点选后能不能跟住」是两码事。

⇒ **本脚本及其全部产物（好片有航迹 / 坏片零航迹）不得写进任何正式报告。**

**正解已落地（用这两套，不要用本脚本）**：
- 第八节「48 宫格模拟点击 + 双向闭环」：``manu/video_quality/vq_grid*.py``
- 第九节「点选式 SOT 实测 + 画质合格线」：``manu/video_quality/vq_sot_link.py`` /
  ``vq_sot_link_driver.py`` / ``vq_sot_compare.py`` / ``vq_sot_verify.py``
  （汇报件 ``manu/video_quality/report/REPORT_视频质量指标.md`` 第九节；
  主档 ``manu/memory/memory.md`` §284；专题档 §八）

本文件仅作历史留档，保留以便追溯当时的口径与代码。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import random
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_job(job: tuple[str, str, str, str, int, int], cli: argparse.Namespace) -> str:
    """Run one segment; return a one-line status string."""
    path, group, kind, modality, start, segment = job
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(path).stem}_s{start:07d}"
    target = out_dir / f"{tag}.json"
    if cli.skip_existing and target.exists():
        return f"SKIP {tag}"
    device = random.randrange(cli.gpus)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "manu/video_quality/vq_track_link.py"),
        "--input", path,
        "--out-dir", str(out_dir),
        "--device", str(device),
        "--threshold-arms", cli.threshold_arms,
        "--crop-mode", cli.crop_mode,
        "--start-frame", str(start),
        "--max-frames", str(segment),
        "--group", group,
        "--kind", kind,
        "--modality", modality,
        "--cpu-threads", str(cli.cpu_threads),
    ]
    log_path = out_dir / f"log_{tag}.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        return f"FAIL {tag} rc={result.returncode} ({elapsed:.0f}s) -> {log_path}"
    return f"OK   {tag} {segment}f {elapsed:.0f}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch driver for vq_track_link.py")
    parser.add_argument("--jobs", required=True, help="Job TSV: path/group/kind/modality/start/frames")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--threshold-arms", default="0.20,0.48")
    parser.add_argument("--crop-mode", default="center1280")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    jobs: list[tuple[str, str, str, str, int, int]] = []
    for line in Path(cli.jobs).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path, group, kind, modality, start, frames = line.split("\t")
        jobs.append((path, group, kind, modality, int(start), int(frames)))
    total_frames = sum(job[5] for job in jobs)
    print(
        f"[DRIVER] jobs={len(jobs)} frames={total_frames} workers={cli.workers} gpus={cli.gpus} "
        f"cpu_threads={cli.cpu_threads} arms={cli.threshold_arms} crop={cli.crop_mode}",
        flush=True,
    )

    started = time.perf_counter()
    done = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=cli.workers) as pool:
        for status in pool.map(lambda job: run_job(job, cli), jobs):
            done += 1
            if status.startswith("FAIL"):
                failures += 1
                print(status, flush=True)
            elif done % 10 == 0 or done == len(jobs):
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(
                    f"[DRIVER] {done}/{len(jobs)} | {time.perf_counter() - started:.0f}s elapsed | "
                    f"{rate:.2f} jobs/s | failures={failures}",
                    flush=True,
                )
    print(
        f"[DRIVER] finished {len(jobs)} jobs in {time.perf_counter() - started:.0f}s | failures={failures}",
        flush=True,
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
