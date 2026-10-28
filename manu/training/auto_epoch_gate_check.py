#!/usr/bin/env python
"""每 epoch 自动三轨门禁：换权重、同口径重跑，判定是否熔断。

设计原则（铁律：判据口径一律不动）
----------------------------------
三轨完全沿用各自**现有**脚本与匹配规则，只把权重换成新 checkpoint：

  Track1 Anti-UAV 官方大盘
    cache_trial0474_inferences.py  →  eval_arm_pc_gmc_ab.py（8px 点对点，15 档门限扫��）
    基线臂 = 冻结的 trial_0474（去 P0）缓存，复用不重算；新臂 = 本次权重重算
  Track2 龙泉山黄金池 7 条
    multiscale_heatmap_cache.py（--ablate-p0）→  system_sota_longquan.py --no-pipeline
    口径：point-in-bbox（落框内即 TP）
  Track3 盲测 001324
    同 Track2 的缓存，单序列评测；**观察轨**，只报告不熔断

红线
----
  Track1: F1 ≥ 基线（默认 0.905736 = 去 P0 冻结基线）
  Track2: F1 ≥ 0.985 且 FP ≤ 26+10=36   —— **一票否决**
  Track3: 观察（默认不熔断；可用 --track3-strict 打开）

退出码
------
  0 = 通过（可能带观察告警）    1 = **指标未达标**（质量熔断）
  3 = **基础设施失败**（推理 OOM / 缓存重建失败 / 评测脚本报错）—— 与质量失败严格区分，
      避免把环境故障误判成"模型退化"而错误熔断。

用法::

    python manu/training/auto_epoch_gate_check.py \\
        --weights runs/mixed_ft_route_b/weights/epoch1.pt \\
        --out-dir runs/mixed_ft_route_b/gate/epoch1 \\
        --infer-gpu 1 --cache-gpus 1,2,3 \\
        --strict-gate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------- 冻结基线常量
# 2026-10-21 冻结的 official 语义 Baseline v2（去 P0 口径）
BASE = {
    "track1_uav": {"f1": 0.905736, "tp": 21624, "fp": 1014, "gt": 25111, "frames": 31613},
    "track1_uav_withp0": {"f1": 0.906361, "tp": 21643, "fp": 1004},
    "track2_golden": {"f1": 0.9884359729262149, "tp": 16283, "fp": 26, "fn": 355, "gt": 16638, "frames": 17187},
    "track3_blind": {"f1": 0.8373308251032032, "tp": 8012, "fp": 590, "fn": 2523, "gt": 10535, "frames": 11810},
}

FT_ROOT = "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official"
UAV_DATA = "/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml"
LABELS_ROOT = "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
PARAMS_JSON = "manu/evaluation/system_sota_longquan_delivery_params.json"
GOLDEN = ",".join(
    [
        "VIDEO00009_19700101_001531",
        "VIDEO00029_19700101_014053",
        "VIDEO00033_19700101_015000",
        "VIDEO00034_19700101_015052",
        "VIDEO00006_19700101_000703",
        "VIDEO00008_19700101_001307",
        "VIDEO00007_19700101_000818",
    ]
)
BLIND = "VIDEO00007_19700101_001324"


def run(cmd: list[str], log: Path, env: dict | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = {**os.environ, **(env or {})}
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), stdout=fh, stderr=subprocess.STDOUT, env=merged)
    return rc


def run_parallel(cmds: list[list[str]], logs: list[Path], env: dict | None = None) -> list[int]:
    """并行跑多条命令（分片用），返回各自退出码。"""
    procs = []
    for cmd, log in zip(cmds, logs):
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "w", encoding="utf-8")
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        merged = {**os.environ, **(env or {})}
        procs.append(
            (
                subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    env=merged,
                ),
                fh,
            )
        )
    codes = []
    for p, fh in procs:
        codes.append(p.wait())
        fh.close()
    return codes


def pick_free_gpus(min_free_mb: int = 2000) -> list[int]:
    """探测当前**空闲**的 GPU（显存余量 > min_free_mb）。

    为什么需要：门禁由训练进程拉起，训练进程本身驻留在某张卡上（约 13.5 GB）。
    若门禁再占用同一张卡，会逼近 24 GB 爆显存（实测 23,809/24,564 MiB，余量仅 755 MiB）。
    因此门禁自动避开被占用的卡。
    """
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=30,
        )
    except Exception:
        return []
    free = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            idx, used, total = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if total - used > min_free_mb:
            free.append(idx)
    return free


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="每 epoch 三轨自动门禁")
    p.add_argument("--weights", required=True, help="待评测的 checkpoint")
    p.add_argument("--out-dir", required=True, help="本轮门禁产物目录")
    p.add_argument("--strict-gate", action="store_true", help="任一红线未过则退出码非 0（熔断）")
    p.add_argument("--gpus", type=int, default=0, help=">0 时按此数分片重建龙泉山缓存")
    p.add_argument("--track1-strict", action="store_true", default=True)
    p.add_argument("--track3-strict", action="store_true", help="把观察轨也升为红线")
    p.add_argument("--min-track1-f1", type=float, default=BASE["track1_uav"]["f1"])
    p.add_argument("--min-track2-f1", type=float, default=0.985)
    p.add_argument("--max-track2-fp", type=int, default=36, help="= 基线 26 + 允许增量 10")
    # ---- 参照系注入（**默认 = 现有 GMC 冻结基线 ⇒ 不传任何新参数时行为逐位不变**）----
    # 换特征语义（例如 no-GMC）时，基线数字与特征根目录必须一起换，
    # 否则会拿 GMC 语义的基线去判 no-GMC 模型 ⇒ 结构性误判。
    p.add_argument("--uav-data", default=UAV_DATA,
                   help="Track1 Anti-UAV 特征集 data.yaml（默认 = 冻结 GMC 集）")
    p.add_argument("--ft-root", default=FT_ROOT,
                   help="龙泉山特征集根目录（需含 images/val；默认 = 冻结 GMC 集）")
    p.add_argument("--base-track1-f1", type=float, default=BASE["track1_uav"]["f1"])
    p.add_argument("--base-track1-tp", type=int, default=BASE["track1_uav"]["tp"])
    p.add_argument("--base-track1-fp", type=int, default=BASE["track1_uav"]["fp"])
    p.add_argument("--base-track2-f1", type=float, default=BASE["track2_golden"]["f1"])
    p.add_argument("--base-track2-tp", type=int, default=BASE["track2_golden"]["tp"])
    p.add_argument("--base-track2-fp", type=int, default=BASE["track2_golden"]["fp"])
    p.add_argument("--base-track2-fn", type=int, default=BASE["track2_golden"]["fn"])
    p.add_argument("--base-track3-f1", type=float, default=BASE["track3_blind"]["f1"])
    p.add_argument("--base-track3-tp", type=int, default=BASE["track3_blind"]["tp"])
    p.add_argument("--base-track3-fp", type=int, default=BASE["track3_blind"]["fp"])
    p.add_argument("--base-track3-fn", type=int, default=BASE["track3_blind"]["fn"])
    p.add_argument("--skip-track1", action="store_true")
    p.add_argument("--skip-longquan", action="store_true")
    # ---- 资源参数（**不影响评测口径**：batch/chunk 只影响速度与显存，推理是确定性的、BN 用 eval）----
    p.add_argument("--infer-batch", type=int, default=32,
                   help="Track1 Anti-UAV 推理的 batch（默认 32；显存紧张时调小，不改变数值）")
    p.add_argument("--cache-batch-size", type=int, default=8,
                   help="龙泉山 val 热图缓存重建的 batch（默认 8）")
    p.add_argument("--cache-chunk-size", type=int, default=64,
                   help="龙泉山 val 热图缓存重建的 chunk（默认 64）")
    p.add_argument(
        "--baseline-cache",
        default="runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl",
        help="Track1 基线臂缓存（**去 P0** 冻结基线，F1=0.905736 TP=21624 FP=1014）",
    )
    p.add_argument(
        "--baseline-weights",
        default="runs/optuna_p0_nas/trial_0474/weights/best.pt",
        help="基线缓存缺失时用它重新生成（去 P0 口径）",
    )
    p.add_argument("--device", default="0")
    p.add_argument(
        "--infer-gpu",
        default="auto",
        help="Track1 推理所用 GPU。'auto' = 自动挑一张空闲卡（避开训练进程占用的卡）",
    )
    p.add_argument(
        "--cache-gpus",
        default="auto",
        help="龙泉山缓存分片所用 GPU，逗号分隔（如 1,2,3）。'auto' = 自动挑空闲卡",
    )
    return p.parse_args()


def effective_base(args: argparse.Namespace) -> dict:
    """按 CLI 覆盖构造**生效基线**。

    默认（不传任何 ``--base-*``）⇒ 与模块常量 ``BASE`` **逐位相等**（旧行为不变）。
    换特征语义（no-GMC 等）时用 CLI 覆盖，避免拿 GMC 语义的基线判新语义模型。
    """
    #      （变量名刻意叫 base_eff：下面 track2/track3 循环里已有一个 `base` 局部变量）
    #      先整份复制 BASE（含 track1_uav_withp0 等信息项），再逐条局部覆盖。
    base_eff = {k: dict(v) for k, v in BASE.items()}
    base_eff["track1_uav"].update(
        {"f1": args.base_track1_f1, "tp": args.base_track1_tp, "fp": args.base_track1_fp}
    )
    base_eff["track2_golden"].update(
        {"f1": args.base_track2_f1, "tp": args.base_track2_tp, "fp": args.base_track2_fp, "fn": args.base_track2_fn}
    )
    base_eff["track3_blind"].update(
        {"f1": args.base_track3_f1, "tp": args.base_track3_tp, "fp": args.base_track3_fp, "fn": args.base_track3_fn}
    )
    return base_eff


def main() -> int:
    args = parse_args()
    base_eff = effective_base(args)
    # ---- GPU 分配：'auto' 解析为当前空闲卡（训练进程占用的卡必须避开）----
    free_gpus = pick_free_gpus()
    if args.infer_gpu.strip() == "auto":
        args.infer_gpu = str(free_gpus[0] if free_gpus else "0")
    if args.cache_gpus.strip() == "auto":
        args.cache_gpus = ",".join(str(g) for g in free_gpus) if free_gpus else ""
    print(f"[GPU] 空闲卡={free_gpus or '无(全部占用)'} | 推理 GPU={args.infer_gpu} | 缓存分片 GPU={args.cache_gpus or '默认 0..N-1'}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict = {
        "weights": args.weights,
        "out_dir": str(out),
        "baselines": base_eff,
        "tracks": {},
        "gates": {},
    }
    print("=" * 100, flush=True)
    print(f"  三轨门禁 | weights={args.weights}", flush=True)
    print("=" * 100, flush=True)

    # ---------------------------------------------------------- Track1 Anti-UAV
    if not args.skip_track1:
        # 基线臂：去 P0 冻结基线缓存。缺失则用原始权重重新生成（保证自包含可复现）
        base_cache = Path(args.baseline_cache)
        if not base_cache.is_file():
            print(f"[BASELINE] 基线缓存缺失，用 {args.baseline_weights} 重新生成（去 P0）…", flush=True)
            base_cache.parent.mkdir(parents=True, exist_ok=True)
            rc = run(
                [
                    sys.executable,
                    "manu/inference/cache_trial0474_inferences.py",
                    "--weights",
                    args.baseline_weights,
                    "--data",
                    args.uav_data,
                    "--output",
                    str(base_cache),
                    "--imgsz",
                    "640",
                    "--stride",
                    "2",
                    "--batch",
                    str(args.infer_batch),
                    "--device",
                    args.infer_gpu,
                    "--conf-thresh",
                    "0.02",
                    "--top-k",
                    "100",
                    "--ablate-p0",
                ],
                out / "baseline_cache_gen.log",
            )
            if rc != 0:
                print(f"[FAIL] 基线缓存生成失败 rc={rc}", flush=True)
                return 3  # 基础设施失败（非指标未达标）
            print(f"[BASELINE] 已生成 {base_cache}", flush=True)

        cache = out / "uav_cache.pkl"
        rc = run(
            [
                sys.executable,
                "manu/inference/cache_trial0474_inferences.py",
                "--weights",
                args.weights,
                "--data",
                args.uav_data,
                "--output",
                str(cache),
                "--imgsz",
                "640",
                "--stride",
                "2",
                "--batch",
                str(args.infer_batch),
                "--device",
                args.infer_gpu,
                "--conf-thresh",
                "0.02",
                "--top-k",
                "100",
                "--ablate-p0",
            ],
            out / "track1_infer.log",
        )
        if rc != 0:
            print(f"[FAIL] Track1 推理失败 rc={rc}，见 {out/'track1_infer.log'}", flush=True)
            report["tracks"]["track1_uav"] = {"status": "infer_failed", "rc": rc}
            if args.strict_gate:
                report["verdict"] = "FAIL"
                (out / "gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                return 3  # 基础设施失败（非指标未达标）
        else:
            ab_json = out / "track1_ab.json"
            rc2 = run(
                [
                    sys.executable,
                    "manu/evaluation/eval_arm_pc_gmc_ab.py",
                    "--arm-a",
                    str(base_cache),
                    "--label-a",
                    "BASELINE",
                    "--arm-b",
                    str(cache),
                    "--label-b",
                    "FINETUNE",
                    "--tol",
                    "8.0",
                    "--expect-f1",
                    str(base_eff["track1_uav"]["f1"]),
                    "--expect-tp",
                    str(base_eff["track1_uav"]["tp"]),
                    "--expect-fp",
                    str(base_eff["track1_uav"]["fp"]),
                    "--out-json",
                    str(ab_json),
                ],
                out / "track1_ab.log",
            )
            if rc2 != 0 or not ab_json.is_file():
                print(f"[FAIL] Track1 A/B 评测失败 rc={rc2}（基线臂未复现冻结 headline）", flush=True)
                report["tracks"]["track1_uav"] = {"status": "ab_failed", "rc": rc2}
                if args.strict_gate:
                    report["verdict"] = "FAIL"
                    (out / "gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                    return 3  # 基础设施失败（非指标未达标）
            else:
                ab = json.loads(ab_json.read_text(encoding="utf-8"))
                peak_b = ab["arm_b"]["peak"]
                peak_a = ab["arm_a"]["peak"]
                report["tracks"]["track1_uav"] = {
                    "status": "ok",
                    "baseline_peak": peak_a,
                    "finetune_peak": peak_b,
                    "delta": ab["delta"],
                    "min_f1_required": args.min_track1_f1,
                }
                ok = peak_b["f1"] >= args.min_track1_f1
                report["gates"]["track1_uav"] = {
                    "rule": f"F1 ≥ {args.min_track1_f1:.6f}（去 P0 冻结基线）",
                    "value": peak_b["f1"],
                    "pass": bool(ok),
                    "veto": True,
                }
                print(
                    f"[Track1] 基线 F1={peak_a['f1']:.6f} TP={peak_a['tp']} FP={peak_a['fp']} | "
                    f"微调 F1={peak_b['f1']:.6f} TP={peak_b['tp']} FP={peak_b['fp']} "
                    f"(th={peak_b['th']:.2f}) | ΔF1={ab['delta']['peak_f1']:+.6f} | "
                    f"{'PASS' if ok else 'FAIL'}",
                    flush=True,
                )

    # ---------------------------------------------------------- Track2/3 龙泉山
    if not args.skip_longquan:
        cache_root = out / "ms_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        ngpu = args.gpus or max(1, len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) or 1)
        try:
            import torch

            ngpu = args.gpus or max(1, torch.cuda.device_count())
        except Exception:
            ngpu = args.gpus or 1
        # 分片实际使用的 GPU 列表：--cache-gpus 优先（用于与训练进程占用的 GPU 错开）
        if args.cache_gpus.strip():
            gpu_ids = [int(x) for x in args.cache_gpus.replace(" ", "").split(",") if x != ""]
        else:
            gpu_ids = list(range(ngpu))
        ngpu = len(gpu_ids)
        pids = []
        cmds = []
        logs = []
        for i in range(ngpu):
            cmds.append(
                [
                    sys.executable,
                    "manu/data/multiscale_heatmap_cache.py",
                    "--features-root",
                    f"{args.ft_root}/images/val",
                    "--cache-root",
                    str(cache_root),
                    "--shard",
                    f"{i}/{ngpu}",
                    "--scales",
                    "160,80",
                    "--region-threshold",
                    "0.06",
                    "--main-threshold",
                    "0.22",
                    "--fusion-dilate",
                    "32",
                    "--min-area",
                    "4",
                    "--device",
                    str(gpu_ids[i]),
                    "--chunk-size",
                    str(args.cache_chunk_size),
                    "--batch-size",
                    str(args.cache_batch_size),
                    "--cpu-threads",
                    "1",
                    "--hm-weights",
                    args.weights,
                    "--ablate-p0",
                    "--overwrite",
                ]
            )
            logs.append(out / f"ms_cache_shard{i}.log")
        print(f"[CACHE] 龙泉山 val 缓存重建：{ngpu} 片并行 …", flush=True)
        t_cache = time.time()
        codes = run_parallel(cmds, logs)
        for i, rc in enumerate(codes):
            if rc != 0:
                print(f"[FAIL] 龙泉山缓存分片 {i} 失败 rc={rc}", flush=True)
                report["tracks"]["track2_golden"] = {"status": "cache_failed", "rc": rc}
                if args.strict_gate:
                    report["verdict"] = "FAIL"
                    (out / "gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                    return 3  # 基础设施失败（非指标未达标）
        n_pkl = len(list(cache_root.glob("*.pkl")))
        print(f"[CACHE] 龙泉山 val 缓存 {n_pkl} 个 pkl（{time.time()-t_cache:.1f}s）", flush=True)

        for track, seq_arg, seq_list, key, base in (
            ("track2_golden", "--seqs", GOLDEN, "track2_golden", base_eff["track2_golden"]),
            ("track3_blind", "--seq", BLIND, "track3_blind", base_eff["track3_blind"]),
        ):
            js = out / f"{track}.json"
            cmd = [
                sys.executable,
                "manu/evaluation/system_sota_longquan.py",
                "--cache-root",
                str(cache_root),
                "--labels-root",
                LABELS_ROOT,
                "--params-json",
                PARAMS_JSON,
                seq_arg,
                seq_list,
                "--no-pipeline",
                "--out-json",
                str(js),
            ]
            rc = run(cmd, out / f"{track}.log")
            if rc != 0 or not js.is_file():
                print(f"[FAIL] {track} 评测失败 rc={rc}", flush=True)
                report["tracks"][key] = {"status": "eval_failed", "rc": rc}
                continue
            data = json.loads(js.read_text(encoding="utf-8"))
            overall = data["overall"]
            f1 = float(overall["f1"])
            fp = int(overall["fp"])
            tp = int(overall["tp"])
            fn = int(overall["fn"])
            rec = {
                "status": "ok",
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "baseline": base,
                "delta_f1": f1 - base["f1"],
                "delta_fp": fp - base["fp"],
                "delta_fn": fn - base["fn"],
            }
            report["tracks"][key] = rec
            if key == "track2_golden":
                ok = f1 >= args.min_track2_f1 and fp <= args.max_track2_fp
                report["gates"][key] = {
                    "rule": f"F1 ≥ {args.min_track2_f1} 且 FP ≤ {args.max_track2_fp}（基线 {base['fp']}+10）",
                    "f1": f1,
                    "fp": fp,
                    "pass": bool(ok),
                    "veto": True,
                }
                verdict = "PASS" if ok else "FAIL"
                print(
                    f"[Track2] 黄金池 F1={f1:.6f} (基线 {base['f1']:.6f}, Δ={f1-base['f1']:+.6f}) "
                    f"TP={tp} FP={fp} (基线 {base['fp']}, Δ={fp-base['fp']:+d}) FN={fn} | {verdict}",
                    flush=True,
                )
            else:
                ok = f1 >= base["f1"]
                report["gates"][key] = {
                    "rule": f"观察轨：F1 vs 基线 {base['f1']:.6f}",
                    "f1": f1,
                    "pass": bool(ok),
                    "veto": bool(args.track3_strict),
                }
                print(
                    f"[Track3] 001324 F1={f1:.6f} (基线 {base['f1']:.6f}, Δ={f1-base['f1']:+.6f}) "
                    f"TP={tp} FP={fp} FN={fn} (基线 FN {base['fn']}, Δ={fn-base['fn']:+d}) | "
                    f"{'PASS' if ok else 'WARN(观察轨不熔断)'}",
                    flush=True,
                )

    # ---------------------------------------------------------- 汇总
    veto_fail = [k for k, v in report["gates"].items() if v.get("veto") and not v.get("pass")]
    soft_fail = [k for k, v in report["gates"].items() if not v.get("veto") and not v.get("pass")]
    report["veto_failures"] = veto_fail
    report["observation_warnings"] = soft_fail
    report["verdict"] = "FAIL" if veto_fail else ("PASS" if not soft_fail else "PASS_WITH_WARNING")
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    (out / "gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100, flush=True)
    print(f"  门禁结论: {report['verdict']} | 一票否决失败: {veto_fail or '无'} | 观察告警: {soft_fail or '无'}", flush=True)
    print(f"  报告: {out/'gate_report.json'} | 耗时 {report['elapsed_seconds']}s", flush=True)
    print("=" * 100, flush=True)

    if veto_fail and args.strict_gate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
