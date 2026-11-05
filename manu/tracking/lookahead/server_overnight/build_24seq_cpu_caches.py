#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""24 序官方口径的 **CPU 口径** ms_cache 构建器（线程预算调度）。

为什么要有 CPU 口径：本服务器无 GPU，官方 24 序 RC2 缓存是 GPU 建的；CPU 与 GPU 的
FP32 前后处理顺序不同 ⇒ 检出数会差个位数（实测 4977 vs 4974）⇒ **CPU 缓存不能直接
与 GPU 缓存拼在一起比**。本脚本给两个模型包**各建一份 CPU 缓存**，于是
``cs_t0043(CPU) vs rc2(CPU)`` 是同口径内部一致的相对 A/B（绝对数字 ≠ GPU 交付档）。

调度原理：``--shard`` 是序列级切分，所以「一条序列一个进程」时关键路径 = 最长序列。
这里用**线程预算**给每条序列分配 ``--cpu-threads``，在总线程预算内把关键路径压到最短：
先按帧数降序，给每条序列选「能落在目标墙钟内的最小线程数」，再二分目标墙钟使总线程
刚好用满预算。

线程-耗时模型（服务器实测标定，chunk = 64 帧）：
    sec_per_chunk(t) ≈ 31.5 + 113/t      （t=1 实测 144.6 s/chunk；t=6 实测 50.3 s/chunk）
模型只用于**分配线程**，不用于报告任何结论；实际墙钟从日志实测。

用法：
    python manu/tracking/lookahead/server_overnight/build_24seq_cpu_caches.py --budget 56
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PY = "/home/huangzhe/anaconda3/envs/uav/bin/python"

C = 31.5  # 固定开销秒/chunk
P = 113.0  # 可并行部分秒/chunk * 线程
CHUNK = 64

REF_CACHE = REPO / "runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache"

JOBSETS = {
    "nogmc": {
        "features": "/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc/images/train",
        "weights": "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt",
        "out": "/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq_cpu_nogmc",
        "logdir": "runs/lookahead/fpv8_24_cpu/nogmc",
    },
    "gmc": {
        "features": "/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train",
        "weights": str(REPO / "runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt"),
        "out": str(REPO / "runs/lookahead/fpv8_24_cpu/gmc/ms_cache"),
        "logdir": "runs/lookahead/fpv8_24_cpu/gmc",
    },
}

CACHE_ARGS = [
    "--scales", "160,80",
    "--region-threshold", "0.06",
    "--main-threshold", "0.22",
    "--fusion-dilate", "32",
    "--min-area", "4",
    "--device", "cpu",
    "--chunk-size", "64",
    "--batch-size", "8",
    "--ablate-p0",
]


def read_sequences() -> dict:
    """{序列名: 帧数} —— 真源 = 官方 RC2 24 序缓存的 meta（两份特征集帧清单一致）。"""
    out = {}
    for pkl in sorted(REF_CACHE.glob("*.pkl")):
        meta = pickle.load(open(pkl, "rb"))["meta"]
        out[meta["sequence"]] = int(meta["frames"])
    return out


def sec_per_chunk(threads: int) -> float:
    return C + P / threads


def min_threads(chunks: int, target_s: float) -> int:
    for t in range(1, 33):
        if chunks * sec_per_chunk(t) <= target_s:
            return t
    return 32


def plan(jobs: list, budget: int) -> tuple:
    """二分目标墙钟，使 Σ min_threads 刚好用满预算。"""
    lo, hi = 600.0, 24 * 3600.0
    best = None
    for _ in range(40):
        mid = (lo + hi) / 2
        alloc = [min_threads(j["chunks"], mid) for j in jobs]
        total = sum(alloc)
        if total <= budget:
            best = (mid, alloc)
            hi = mid
        else:
            lo = mid
    if best is None:
        target = hi
        alloc = [1] * len(jobs)
    else:
        target, alloc = best
    for j, t in zip(jobs, alloc):
        j["threads"] = t
        j["eta_s"] = j["chunks"] * sec_per_chunk(t)
    return target, alloc


def main() -> int:
    ap = argparse.ArgumentParser(description="24 序 CPU 口径 ms_cache 构建器")
    ap.add_argument("--budget", type=int, default=56, help="总线程预算")
    ap.add_argument("--sets", default="nogmc,gmc")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="目标 pkl 已存在则跳过（幂等重跑）")
    ap.add_argument("--plan-json", default="runs/lookahead/fpv8_24_cpu/plan.json")
    args = ap.parse_args()

    frames = read_sequences()
    if len(frames) != 24:
        print(f"[FATAL] 官方缓存里读到 {len(frames)} 条序列（应为 24）", file=sys.stderr)
        return 3

    jobs = []
    for tag in args.sets.split(","):
        spec = JOBSETS[tag]
        for seq, n in frames.items():
            jobs.append(
                {"tag": tag, "seq": seq, "frames": n, "chunks": -(-n // CHUNK), "threads": 1, "eta_s": 0.0, **{
                    k: spec[k] for k in ("features", "weights", "out", "logdir")
                }}
            )
    jobs.sort(key=lambda j: -j["chunks"])
    target, _ = plan(jobs, args.budget)
    total_threads = sum(j["threads"] for j in jobs)
    eta_h = max(j["eta_s"] for j in jobs) / 3600
    print(f"[PLAN] 目标墙钟 {target / 3600:.2f} h | 线程预算 {args.budget}（实配 {total_threads}）"
          f" | 关键路径 ETA {eta_h:.2f} h | 作业数 {len(jobs)}")
    for j in jobs:
        print(f"  {j['tag']:6s} {j['seq']:32s} frames={j['frames']:6d} chunks={j['chunks']:4d} "
              f"threads={j['threads']:2d} eta={j['eta_s'] / 3600:.2f}h")

    Path(args.plan_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.plan_json).write_text(
        json.dumps({"target_h": target / 3600, "budget": args.budget, "threads": total_threads,
                    "eta_h": eta_h, "jobs": jobs}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    if args.dry_run:
        return 0

    env = {
        "PYTHONPATH": str(REPO), "MPLCONFIGDIR": "/tmp/mpl",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/home/huangzhe",
    }
    procs = []
    skipped = 0
    t0 = time.time()
    for j in jobs:
        if args.skip_existing and (Path(j['out']) / f"{j['seq']}.pkl").is_file():
            skipped += 1
            continue
        logdir = REPO / j["logdir"]
        logdir.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "manu/data/multiscale_heatmap_cache.py",
               "--features-root", j["features"], "--cache-root", j["out"],
               "--sequences", j["seq"], "--cpu-threads", str(j["threads"]), *CACHE_ARGS,
               "--hm-weights", j["weights"], "--overwrite"]
        log = open(logdir / f"{j['seq']}.log", "w")
        procs.append((j, subprocess.Popen(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, env=env), log))
    print(f"[RUN] 已启动 {len(procs)} 个进程（跳过已完成 {skipped} 条）")

    fails = []
    for j, p, log in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            fails.append((j["tag"], j["seq"], rc))
        print(f"[{'DONE' if rc == 0 else 'FAIL'}] {j['tag']:6s} {j['seq']:32s} rc={rc} "
              f"用时 {(time.time() - t0) / 3600:.2f}h")
    print(f"[SUMMARY] 用时 {(time.time() - t0) / 3600:.2f} h | 失败 {len(fails)}/{len(procs)}")
    for tag, seq, rc in fails:
        print(f"  FAIL {tag} {seq} rc={rc}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
