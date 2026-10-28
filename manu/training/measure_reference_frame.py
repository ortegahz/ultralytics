#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
参照系测量工具：把「冻结基线」放到**指定特征语义**上重测，产出参照系包 JSON。

为什么需要它
------------
判定链路的基线数字（``mix_ratio_judgement.py`` 的 ``FROZEN_BASELINE`` /
``RC1_EPOCH3``，以及 ``auto_epoch_gate_check.py`` 的 ``BASE``）都是**在 GMC 语义特征上
实测**的。一旦换特征语义（例如 no-GMC），这些数字**全部失效** ⇒
拿 GMC 基线去判 no-GMC 模型 = 结构性误判（一票否决项会凭空触发）。

本工具按**与门禁完全相同**的口径重测基线臂，产出：
  ``reference_frame.json`` —— 可直接喂给
  ``optuna_mix_ratio_search.py --reference-frame`` 与
  ``auto_epoch_gate_check.py --base-*``。

口径一致性（与 `auto_epoch_gate_check.py` 逐项对齐，**不要改**）
  Track1: cache_trial0474_inferences.py（--ablate-p0, conf 0.02, top-k 100, imgsz 640, stride 2）
          → eval_arm_pc_gmc_ab.py（tol 8.0，15 档门限扫参）
  Track2/3: multiscale_heatmap_cache.py（--scales 160,80 --region-threshold 0.06
            --main-threshold 0.22 --fusion-dilate 32 --min-area 4 --ablate-p0）
          → system_sota_longquan.py --no-pipeline

用法（服务器）
------------
    python manu/training/measure_reference_frame.py \
        --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
        --ft-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc \
        --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
        --out-dir runs/nogmc_ref --label nogmc --gpus 0,1,2,3

产出
----
    runs/nogmc_ref/
      ├── uav_cache_base.pkl      # T1 基线臂缓存（<20 MB，float16 稀疏）
      ├── track1_baseline.json    # T1 基线臂评测
      ├── ms_cache/               # T2/T3 热图缓存
      ├── track2_baseline.json    # T2 黄金池基线臂
      ├── track3_baseline.json    # T3 盲测基线臂
      └── reference_frame.json    # ★ 参照系包（喂给 --reference-frame）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 与 auto_epoch_gate_check.py 完全一致的三轨定义
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
LABELS_ROOT = "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
PARAMS_JSON = "manu/evaluation/system_sota_longquan_delivery_params.json"

# 门禁阈值口径（与 GMC 口径保持一致，换的是**基线数字**不是阈值定义）
GATE_TRACK2_F1_MIN = 0.985
GATE_TRACK2_FP_MAX_INCREMENT = 10
GATE_TRACK1_MIN_DELTA = -0.005


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.call([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    print(f"   rc={rc} | 日志 {log}", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="在指定特征语义上重测冻结基线，产出参照系包 JSON")
    # ⚠️ 这两个参数在「只填 rc1」模式（--fill-rc1-from）下**不需要**，
    # 因此不能设 required=True —— argparse 的必填校验发生在任何业务分支之前，
    # 设了会让 `measure_reference_frame.py --out-dir X --fill-rc1-from Y` 直接
    # 以退出码 2 失败（2026-10-27 实测复现），无人值守管线的 rc1 自动回填就静默废了。
    # 改为 default="" + 在**测量分支内**显式校验 ⇒ 测量模式的必填语义逐字保留。
    ap.add_argument("--uav-data", default="", help="Anti-UAV 特征集 data.yaml（--fill-rc1-from 模式不需要）")
    ap.add_argument("--ft-root", default="", help="龙泉山特征集根目录（含 images/val）（--fill-rc1-from 模式不需要）")
    ap.add_argument("--weights", default="runs/optuna_p0_nas/trial_0474/weights/best.pt", help="冻结基线权重")
    ap.add_argument("--out-dir", default="runs/nogmc_ref", help="产物目录")
    ap.add_argument("--label", default="nogmc", help="参照系标签（写进 JSON 便于审计）")
    ap.add_argument("--gpus", default="0", help="GPU 列表（逗号分隔）；LQ 缓存按此数分片")
    ap.add_argument("--infer-gpu", default="", help="T1 推理 GPU（默认取 gpus 第一个）")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip-lq", action="store_true", help="只测 T1")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument(
        "--fill-rc1-from",
        type=str,
        default="",
        help="只做这一步：把某个 trial 的 gate_report.json 写进 --out-dir/reference_frame.json 的 rc1 字段"
             "（用于把 natural 对照 trial 设为该特征语义下的新擂主）",
    )
    args = ap.parse_args()

    # ---------------------------------------------------------------- 只填 rc1 的模式
    if args.fill_rc1_from:
        sys.path.insert(0, str(REPO_ROOT))
        from manu.training.mix_ratio_judgement import metrics_from_gate_report

        rf_path = Path(args.out_dir) / "reference_frame.json"
        if not rf_path.is_file():
            print(f"[FAIL] 找不到参照系包：{rf_path}", flush=True)
            return 3
        ref = json.loads(rf_path.read_text(encoding="utf-8"))
        m = metrics_from_gate_report(args.fill_rc1_from)
        ref["rc1"] = {
            "track1_f1": m["track1_f1"], "track1_fp": m["track1_fp"],
            "track2_f1": m["track2_f1"], "track2_fp": m["track2_fp"],
            "track3_f1": m["track3_f1"], "track3_fp": m["track3_fp"],
        }
        rf_path.write_text(json.dumps(ref, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[RC1] 已把 {args.fill_rc1_from} 写入 {rf_path} 的 rc1：")
        print(f"       T1 F1={m['track1_f1']} FP={m['track1_fp']} | T2 F1={m['track2_f1']} FP={m['track2_fp']} | "
              f"T3 F1={m['track3_f1']} FP={m['track3_fp']}")
        return 0

    # ---- 测量模式：仍强制要求这两个参数（保持原必填语义，只是报错时机后移到这里）----
    if not args.uav_data or not args.ft_root:
        ap.error("测量模式必须同时提供 --uav-data 与 --ft-root"
                 "（若只是回填 rc1，请改用 --fill-rc1-from）")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    infer_gpu = args.infer_gpu or (gpus[0] if gpus else "0")

    t0 = time.time()
    base_cache = out / "uav_cache_base.pkl"

    # ---------------------------------------------------------------- Track1 基线臂
    print("\n================ Track1：基线臂推理缓存 ================", flush=True)
    rc = run(
        [
            sys.executable,
            "manu/inference/cache_trial0474_inferences.py",
            "--weights", args.weights,
            "--data", args.uav_data,
            "--output", base_cache,
            "--imgsz", "640", "--stride", "2",
            "--batch", str(args.batch),
            "--device", infer_gpu,
            "--conf-thresh", "0.02",
            "--top-k", "100",
            "--ablate-p0",
        ],
        out / "track1_infer.log",
    )
    if rc != 0 and not args.dry_run:
        print(f"[FAIL] Track1 推理失败 rc={rc}", flush=True)
        return 3

    print("\n================ Track1：基线臂自评（arm_a=arm_b ⇒ 取 arm_a 自身指标） ================", flush=True)
    t1_json = out / "track1_baseline.json"
    rc = run(
        [
            sys.executable,
            "manu/evaluation/eval_arm_pc_gmc_ab.py",
            "--arm-a", base_cache, "--label-a", f"{args.label}_BASE",
            "--arm-b", base_cache, "--label-b", f"{args.label}_BASE_SELF",
            "--tol", "8.0",
            "--out-json", t1_json,
            "--quiet",
        ],
        out / "track1_ab.log",
    )
    if rc != 0 and not args.dry_run:
        print(f"[FAIL] Track1 自评失败 rc={rc}", flush=True)
        return 3

    # ---------------------------------------------------------------- Track2/3 基线臂
    t2 = t3 = None
    if not args.skip_lq:
        print("\n================ Track2/3：龙泉山热图缓存（基线权重） ================", flush=True)
        cache_root = out / "ms_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        ngpu = len(gpus) or 1
        cmds, logs = [], []
        for i, g in enumerate(gpus):
            cmds.append(
                [
                    sys.executable,
                    "manu/data/multiscale_heatmap_cache.py",
                    "--features-root", f"{args.ft_root}/images/val",
                    "--cache-root", cache_root,
                    "--shard", f"{i}/{ngpu}",
                    "--scales", "160,80",
                    "--region-threshold", "0.06",
                    "--main-threshold", "0.22",
                    "--fusion-dilate", "32",
                    "--min-area", "4",
                    "--device", g,
                    "--chunk-size", "64",
                    "--batch-size", "8",
                    "--cpu-threads", "1",
                    "--hm-weights", args.weights,
                    "--ablate-p0",
                    "--overwrite",
                ]
            )
            logs.append(out / f"ms_cache_shard{i}.log")
        for c, lg in zip(cmds, logs):
            rc = run(c, lg)
            if rc != 0 and not args.dry_run:
                print(f"[FAIL] 龙泉山缓存重建失败 rc={rc}", flush=True)
                return 3

        for track, seq_arg, seqs, js_name in (
            ("track2_golden", "--seqs", GOLDEN, "track2_baseline.json"),
            ("track3_blind", "--seq", BLIND, "track3_baseline.json"),
        ):
            print(f"\n================ {track}：基线臂评测 ================", flush=True)
            js = out / js_name
            rc = run(
                [
                    sys.executable,
                    "manu/evaluation/system_sota_longquan.py",
                    "--cache-root", cache_root,
                    "--labels-root", LABELS_ROOT,
                    "--params-json", PARAMS_JSON,
                    seq_arg, seqs,
                    "--no-pipeline",
                    "--out-json", js,
                ],
                out / f"{track}.log",
            )
            if rc != 0 and not args.dry_run:
                print(f"[FAIL] {track} 评测失败 rc={rc}", flush=True)
                return 3
            if js.is_file():
                data = json.loads(js.read_text(encoding="utf-8"))
                o = data["overall"]
                rec = {"f1": float(o["f1"]), "tp": int(o["tp"]), "fp": int(o["fp"]), "fn": int(o["fn"])}
                if track == "track2_golden":
                    t2 = rec
                else:
                    t3 = rec
                print(f"[{track}] 基线臂: F1={rec['f1']:.6f} TP={rec['tp']} FP={rec['fp']} FN={rec['fn']}", flush=True)

    # ---------------------------------------------------------------- 汇总
    if args.dry_run:
        print("\n[DRY-RUN] 未执行，命令已打印完毕", flush=True)
        return 0

    t1 = {}
    if t1_json.is_file():
        ab = json.loads(t1_json.read_text(encoding="utf-8"))
        pk = ab["arm_a"]["peak"]
        t1 = {"f1": float(pk["f1"]), "tp": int(pk["tp"]), "fp": int(pk["fp"])}

    if not t1 or not t2 or not t3:
        print(f"[FAIL] 缺基线数字：T1={bool(t1)} T2={bool(t2)} T3={bool(t3)}", flush=True)
        return 3

    ref = {
        "label": args.label,
        "note": (
            "在指定特征语义上实测的冻结基线（trial_0474）。"
            "**换特征语义必须整套替换**，否则会拿旧语义的基线判新语义模型。"
            "rc1 需由该语义下的 natural 对照 trial 填入后再判 RC2。"
        ),
        "frozen_baseline": {
            "track1_f1": t1["f1"], "track1_fp": t1["fp"],
            "track2_f1": t2["f1"], "track2_fp": t2["fp"],
            "track3_f1": t3["f1"], "track3_fp": t3["fp"],
        },
        "rc1": {},
        "gate": {
            "track2_f1_min": GATE_TRACK2_F1_MIN,
            "track2_fp_max_increment": GATE_TRACK2_FP_MAX_INCREMENT,
            "track1_min_delta": GATE_TRACK1_MIN_DELTA,
        },
        "gate_base": {
            "track1_f1": t1["f1"], "track1_tp": t1["tp"], "track1_fp": t1["fp"],
            "track2_f1": t2["f1"], "track2_tp": t2["tp"], "track2_fp": t2["fp"], "track2_fn": t2["fn"],
            "track3_f1": t3["f1"], "track3_tp": t3["tp"], "track3_fp": t3["fp"], "track3_fn": t3["fn"],
        },
    }
    rf = out / "reference_frame.json"
    rf.write_text(json.dumps(ref, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"[参照系包] {rf}")
    print(f"  T1 F1={t1['f1']:.6f} TP={t1['tp']} FP={t1['fp']}")
    print(f"  T2 F1={t2['f1']:.6f} TP={t2['tp']} FP={t2['fp']} FN={t2['fn']}")
    print(f"  T3 F1={t3['f1']:.6f} TP={t3['tp']} FP={t3['fp']} FN={t3['fn']}")
    print(f"  用时 {(time.time()-t0)/60:.1f} min")
    print("  ⚠️ rc1 为空：需先用该语义跑 natural 对照 trial，再把它的指标填进 rc1，才能判 RC2")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
