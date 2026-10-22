#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train 划分容量 / 过拟合诊断：原始底模 vs 微调权重（RC1 = Ep3）在 **训练集** 上的精度对照。

动机
----
路线 B 首轮试跑（5 epoch）已在 val 三轨上做完对照，但**训练集**的拟合程度从未量过。
本脚本补上这一块，用来回答三个诊断问题：

1. **欠拟合检查**：训练帧上的召回率是否健康爬到 92%~96%？
   若连反复见过的 `002001`（train 划分 core_positive_mining）单帧召回仍只有 80% 左右，
   则实锤「全图 1:3.09 混训下大背景稀释了小目标梯度」⇒ 未来要压召回必须上局部切片放大。
2. **过拟合检查**：train F1 是否已逼近 99.5%+（容量用尽 / 开始过拟合建筑物高频边缘），
   还是停在 95%~97%（判别流形平滑、泛化余量充足）。
3. **顽固误检定位**：train 各序列的 FP 分布 ⇒ 是否存在与 val 侧 `000703` 同形态的虚警，
   若 train 里这类误报早已压为 0、唯独未见过的 `000703` 爆出 25 个 FP，
   即属**特定负样本先验缺失**（补几张该形态纯背景切片即可歼灭）。

口径铁律
--------
评测脚本、匹配规则、参数**一律不动**，与 §四十三 门禁完全同一条路径：

    manu/data/multiscale_heatmap_cache.py   (--features-root / --ablate-p0 / 固定超参)
        → manu/evaluation/system_sota_longquan.py (--no-pipeline --all-seqs)

只换 `--hm-weights`（缓存）与 `--labels-root`（train 专属位置索引标签根，见
`manu/data/build_train_labels_pos.py`）。两个权重的缓存参数完全一致 ⇒ 对照公平。

用法（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，
项目根 `/tmp/pycharm_project_10ae9e2e`）：

```bash
python manu/training/train_fit_diagnostic.py \
  --weights baseline=runs/optuna_p0_nas/trial_0474/weights/best.pt \
            ft=runs/mixed_ft_route_b/RC1/epoch3.pt \
  --features-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train \
  --labels-root runs/train_fit_diag/labels_train_pos \
  --out-dir runs/train_fit_diag \
  --gpus auto
```
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

# 与 manu/training/auto_epoch_gate_check.py 完全一致的缓存超参（口径不得漂移）
CACHE_ARGS = {
    "scales": "160,80",
    "region_threshold": "0.06",
    "main_threshold": "0.22",
    "fusion_dilate": "32",
    "min_area": "4",
    "chunk_size": "64",
    "batch_size": "8",
    "cpu_threads": "1",
}
DEFAULT_PARAMS_JSON = "manu/evaluation/system_sota_longquan_delivery_params.json"


def free_gpus(min_free_mb: int = 2000) -> list[int]:
    """列出显存余量足够的 GPU 编号。"""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:
        return []
    ids: list[int] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, used, total = int(parts[0]), float(parts[1]), float(parts[2])
        if total - used >= min_free_mb:
            ids.append(idx)
    return ids


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode


def build_cache(
    weights: str,
    source_root: str,
    source: str,
    cache_root: Path,
    gpu_ids: list[int],
    logs_dir: Path,
    sequences: str = "",
    n_shards: int = 0,
) -> tuple[bool, float]:
    cache_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    # GMC+median 现场重算是 CPU 密集，服务器 80 核 ⇒ 分片数可以多于 GPU 数，
    # 用 round-robin 把分片映射到 GPU 上，让 CPU 并行度拉满。
    n = n_shards or len(gpu_ids)
    procs = []
    logs = []
    for i in range(n):
        cmd = [
            sys.executable,
            "manu/data/multiscale_heatmap_cache.py",
            "--features-root" if source == "features" else "--frames-root",
            source_root,
            "--cache-root",
            str(cache_root),
            "--shard",
            f"{i}/{n}",
            "--scales",
            CACHE_ARGS["scales"],
            "--region-threshold",
            CACHE_ARGS["region_threshold"],
            "--main-threshold",
            CACHE_ARGS["main_threshold"],
            "--fusion-dilate",
            CACHE_ARGS["fusion_dilate"],
            "--min-area",
            CACHE_ARGS["min_area"],
            "--device",
            str(gpu_ids[i % len(gpu_ids)]),
            "--chunk-size",
            CACHE_ARGS["chunk_size"],
            "--batch-size",
            CACHE_ARGS["batch_size"],
            "--cpu-threads",
            CACHE_ARGS["cpu_threads"],
            "--hm-weights",
            weights,
            "--ablate-p0",
            "--overwrite",
        ]
        if sequences:
            cmd += ["--sequences", sequences]
        log = logs_dir / f"ms_cache_shard{i}.log"
        logs.append(log)
        fh = log.open("w", encoding="utf-8")
        # 每个分片单线程：GMC 靠多进程并行，不靠进程内线程（否则 18 个进程 × 80 线程互相抢核）
        env = dict(os.environ)
        env.update(
            {
                "OPENCV_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        procs.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh))
    started = time.time()
    ok = True
    for proc, fh in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            ok = False
    return ok, time.time() - started


def evaluate(
    cache_root: Path, labels_root: str, params_json: str, out_json: Path, log: Path
) -> tuple[bool, dict | None]:
    cmd = [
        sys.executable,
        "manu/evaluation/system_sota_longquan.py",
        "--cache-root",
        str(cache_root),
        "--labels-root",
        labels_root,
        "--params-json",
        params_json,
        "--all-seqs",
        "--no-pipeline",
        "--out-json",
        str(out_json),
    ]
    rc = run(cmd, log)
    if rc != 0 or not out_json.is_file():
        return False, None
    return True, json.loads(out_json.read_text(encoding="utf-8"))


METRIC_KEYS = ("gt", "tp", "fp", "fn", "recall", "precision", "f1", "far_per_frame")


def row_metrics(row: dict) -> dict:
    out = {}
    for key in METRIC_KEYS:
        if key in row:
            out[key] = row[key]
    for key in ("fn_absent", "fn_compete", "frames"):
        if key in row:
            out[key] = row[key]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="train 划分容量/过拟合诊断：底模 vs 微调权重")
    ap.add_argument("--weights", nargs="+", required=True, help="tag=path，可写多个（空格分隔）")
    ap.add_argument("--features-root", default="", help="数据集 images/train（特征 JPG 根，--source features 时必填）")
    ap.add_argument("--frames-root", default="", help="原始帧根（--source frames 时必填：现场重算 GMC）")
    ap.add_argument("--labels-root", required=True, help="位置索引化标签根（build_train_labels_pos.py 产出）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--params-json", default=DEFAULT_PARAMS_JSON)
    ap.add_argument("--gpus", default="auto", help="auto | 逗号分隔 GPU 号 | 个数")
    ap.add_argument("--skip-cache", action="store_true", help="复用已有缓存，不重建")
    ap.add_argument("--sequences", default="", help="逗号分隔序列名（限定评测/建缓存的序列）")
    ap.add_argument("--shards", type=int, default=0, help="分片数（0=与 GPU 数相同；可多于 GPU 数以压满 CPU）")
    ap.add_argument("--sequential", action="store_true", help="串行构建各权重缓存（默认并行）")
    ap.add_argument(
        "--source",
        choices=["features", "frames"],
        default="features",
        help="features=读现成特征 JPG；frames=从原始 IR 帧现场重算 GMC（stride=1 用）",
    )
    args = ap.parse_args()

    if args.source == "features" and not args.features_root:
        raise SystemExit("[FATAL] --source features 需要 --features-root")
    if args.source == "frames" and not args.frames_root:
        raise SystemExit("[FATAL] --source frames 需要 --frames-root")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gpus.strip() == "auto":
        gpu_ids = free_gpus()
    elif "," in args.gpus:
        gpu_ids = [int(x) for x in args.gpus.replace(" ", "").split(",") if x]
    else:
        gpu_ids = list(range(int(args.gpus)))
    if not gpu_ids:
        print("[FATAL] 没有可用 GPU", flush=True)
        return 3
    print(f"[GPU] 使用 {gpu_ids}", flush=True)

    entries: list[tuple[str, str]] = []
    for item in args.weights:
        if "=" not in item:
            raise SystemExit(f"[FATAL] --weights 需要 tag=path 形式：{item}")
        tag, path = item.split("=", 1)
        entries.append((tag.strip(), path.strip()))
    results: dict[str, dict] = {}

    # ---- 并行构建各权重的缓存 ----
    # GMC+median 现场重算是 CPU 密集（每帧一次 affine 估计 + 21 帧中值），是整条链路的瓶颈；
    # 各权重互不依赖 ⇒ 缓存并行构建，把 CPU 并行度拉满。
    build_tasks = []
    for tag, weights in entries:
        cache_root = out_dir / tag / "ms_cache"
        tag_dir = out_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        build_tasks.append((tag, weights, cache_root, tag_dir))

    def _build(task):
        tag, weights, cache_root, tag_dir = task
        if args.skip_cache:
            return tag, (True, 0.0)
        ok, sec = build_cache(
            weights,
            args.features_root if args.source == "features" else args.frames_root,
            args.source,
            cache_root,
            gpu_ids,
            tag_dir / "logs",
            sequences=args.sequences,
            n_shards=args.shards,
        )
        return tag, (ok, sec)

    build_results: dict[str, tuple[bool, float]] = {}
    if args.sequential or len(build_tasks) == 1:
        for task in build_tasks:
            tag, (ok, sec) = _build(task)
            print(f"[CACHE] {tag}: ok={ok} {sec:.1f}s", flush=True)
            build_results[tag] = (ok, sec)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(build_tasks)) as pool:
            for tag, (ok, sec) in pool.map(_build, build_tasks):
                print(f"[CACHE] {tag}: ok={ok} {sec:.1f}s", flush=True)
                build_results[tag] = (ok, sec)

    for tag, weights, cache_root, tag_dir in build_tasks:
        cache_ok, cache_s = build_results.get(tag, (False, 0.0))
        if not cache_ok:
            results[tag] = {"status": "cache_failed"}
            continue
        n_pkl = len(list(cache_root.glob("*.pkl")))
        print(f"[CACHE] {tag}: {n_pkl} 个序列 pkl", flush=True)
        out_json = tag_dir / "train_eval.json"
        eval_ok, data = evaluate(
            cache_root, args.labels_root, args.params_json, out_json, tag_dir / "train_eval.log"
        )
        if not eval_ok or data is None:
            results[tag] = {"status": "eval_failed"}
            continue
        results[tag] = {
            "status": "ok",
            "weights": weights,
            "cache_root": str(cache_root),
            "cache_seconds": cache_s,
            "elapsed_seconds": data.get("elapsed_seconds"),
            "overall": {k: data["overall"][k] for k in data["overall"] if not isinstance(data["overall"][k], (dict, list))},
            "per_sequence": {s: row_metrics(r) for s, r in data["per_sequence"].items()},
        }
        o = results[tag]["overall"]
        print(
            f"[EVAL] {tag}: F1={o.get('f1'):.6f} TP={o.get('tp')} FP={o.get('fp')} FN={o.get('fn')} "
            f"R={o.get('recall', 0) * 100:.2f}% P={o.get('precision', 0) * 100:.2f}%",
            flush=True,
        )

    ok_tags = [t for t, _ in entries if results.get(t, {}).get("status") == "ok"]
    if len(ok_tags) < 2:
        print("[FATAL] 至少需要两个成功评测的权重才能对照", flush=True)
        (out_dir / "train_fit_diagnostic.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 3

    base_tag, ft_tag = ok_tags[0], ok_tags[1]
    base = results[base_tag]
    ft = results[ft_tag]

    # ---- 逐序列对照表 ----
    seqs = sorted(set(base["per_sequence"]) | set(ft["per_sequence"]))
    table = []
    for s in seqs:
        b = base["per_sequence"].get(s, {})
        f = ft["per_sequence"].get(s, {})
        table.append(
            {
                "sequence": s,
                "frames": f.get("frames", b.get("frames")),
                "gt": f.get("gt", b.get("gt")),
                "base_f1": b.get("f1"),
                "ft_f1": f.get("f1"),
                "delta_f1": (f.get("f1") - b.get("f1")) if (b.get("f1") is not None and f.get("f1") is not None) else None,
                "base_tp": b.get("tp"),
                "ft_tp": f.get("tp"),
                "delta_tp": (f.get("tp") - b.get("tp")) if (b.get("tp") is not None and f.get("tp") is not None) else None,
                "base_fp": b.get("fp"),
                "ft_fp": f.get("fp"),
                "delta_fp": (f.get("fp") - b.get("fp")) if (b.get("fp") is not None and f.get("fp") is not None) else None,
                "base_fn": b.get("fn"),
                "ft_fn": f.get("fn"),
                "delta_fn": (f.get("fn") - b.get("fn")) if (b.get("fn") is not None and f.get("fn") is not None) else None,
                "base_recall": b.get("recall"),
                "ft_recall": f.get("recall"),
                "delta_recall": (f.get("recall") - b.get("recall")) if (b.get("recall") is not None and f.get("recall") is not None) else None,
                "base_precision": b.get("precision"),
                "ft_precision": f.get("precision"),
                "delta_precision": (f.get("precision") - b.get("precision")) if (b.get("precision") is not None and f.get("precision") is not None) else None,
            }
        )

    bo, fo = base["overall"], ft["overall"]
    summary = {
        "base_tag": base_tag,
        "ft_tag": ft_tag,
        "base_weights": base["weights"],
        "ft_weights": ft["weights"],
        "overall": {
            "base": {k: bo.get(k) for k in ("f1", "tp", "fp", "fn", "recall", "precision", "gt", "far_per_frame")},
            "ft": {k: fo.get(k) for k in ("f1", "tp", "fp", "fn", "recall", "precision", "gt", "far_per_frame")},
            "delta": {
                k: (fo.get(k) - bo.get(k))
                for k in ("f1", "tp", "fp", "fn", "recall", "precision")
                if bo.get(k) is not None and fo.get(k) is not None
            },
        },
        "per_sequence_table": table,
    }

    doc = {"config": vars(args), "results": results, "summary": summary}
    (out_dir / "train_fit_diagnostic.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (out_dir / "train_fit_diagnostic.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    def fmt(v, nd=6):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    md = []
    md.append("# Train 划分容量 / 过拟合诊断：原始底模 vs 微调 RC1(Ep3)\n")
    md.append(f"- 底模（baseline）：`{base['weights']}`\n- 微调（ft）：`{ft['weights']}`\n")
    md.append(
        f"- 特征来源：`{args.features_root or args.frames_root}`"
        f"（source={args.source}；**特征零重算**，直接读现成特征/现场同口径 GMC）\n"
    )
    md.append(f"- 标签根：`{args.labels_root}`（位置索引化，解决 stride=5 帧号错位）\n")
    md.append("- 口径：与 §四十三 门禁完全一致（`multiscale_heatmap_cache.py` → `system_sota_longquan.py --no-pipeline --all-seqs`），只换权重\n")
    md.append("\n## 1. 训练集整体对照\n")
    md.append("| 指标 | 底模 | 微调 Ep3 | Δ |")
    md.append("|:--|--:|--:|--:|")
    for key, label, nd in (
        ("f1", "F1", 6),
        ("recall", "Recall", 6),
        ("precision", "Precision", 6),
        ("tp", "TP", 0),
        ("fp", "FP", 0),
        ("fn", "FN", 0),
    ):
        md.append(
            f"| {label} | {fmt(summary['overall']['base'].get(key), nd)} | "
            f"{fmt(summary['overall']['ft'].get(key), nd)} | {fmt(summary['overall']['delta'].get(key), nd)} |"
        )
    md.append("\n## 2. 逐序列对照（train 9 条）\n")
    md.append("| 序列 | GT | 底模 F1 | Ep3 F1 | ΔF1 | 底模 TP | Ep3 TP | 底模 FP | Ep3 FP | ΔFP | 底模 FN | Ep3 FN | ΔFN |")
    md.append("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for row in table:
        md.append(
            f"| {row['sequence']} | {row['gt']} | {fmt(row['base_f1'])} | {fmt(row['ft_f1'])} | "
            f"{fmt(row['delta_f1'])} | {row['base_tp']} | {row['ft_tp']} | {row['base_fp']} | {row['ft_fp']} | "
            f"{fmt(row['delta_fp'], 0)} | {row['base_fn']} | {row['ft_fn']} | {fmt(row['delta_fn'], 0)} |"
        )
    (out_dir / "train_fit_diagnostic.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n[OK] 对照表 -> {out_dir / 'train_fit_diagnostic.md'}", flush=True)
    print(f"[OK] 明细   -> {out_dir / 'train_fit_diagnostic.json'}", flush=True)
    print(f"[OK] CSV    -> {out_dir / 'train_fit_diagnostic.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
