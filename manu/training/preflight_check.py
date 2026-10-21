#!/usr/bin/env python
"""试跑前自检：**无干预出**（自检第 5 项）+ 数据满额（第 2/5 项）。

按**磁盘实际文件**逐条核对，不信任任何配置声明（配置≠事实）。
检查项：
  A. 龙泉山 train/val 序列集合互斥
  B. 龙泉山 val 恰为「黄金池 7 条 + 001324」，train 恰为配置里的 9 条
  C. 训练集**逐帧**不含 001324 与黄金池 7 条的任何一帧
  D. Anti-UAV train/val 序列集合互斥
  E. 两源帧数满额（龙泉山 13,918 / Anti-UAV 43,008）

用法::

    python manu/training/preflight_check.py --out-json runs/logs/preflight_leakage.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FT_ROOT = Path("/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official")
UAV_ROOT = Path("/mnt/data/siping/datasets/manu/uav_gmc_median")
SPLIT_JSON = PROJECT_ROOT / "manu/data/longquan_ft_split.json"

GOLDEN = [
    "VIDEO00009_19700101_001531",
    "VIDEO00029_19700101_014053",
    "VIDEO00033_19700101_015000",
    "VIDEO00034_19700101_015052",
    "VIDEO00006_19700101_000703",
    "VIDEO00008_19700101_001307",
    "VIDEO00007_19700101_000818",
]
BLIND = "VIDEO00007_19700101_001324"


def lq_seq(p: Path) -> str:
    return p.stem.split("__")[0]


def uav_seq(p: Path) -> str:
    return p.stem.split("__")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="试跑前自检：无干预出 + 数据满额")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    report: dict = {"checks": {}}
    failures: list[str] = []

    # ---------------- A/B/C 龙泉山 ----------------
    lq_train = sorted(FT_ROOT.glob("images/train/*.jpg"))
    lq_val = sorted(FT_ROOT.glob("images/val/*.jpg"))
    tr_seqs = sorted({lq_seq(p) for p in lq_train})
    va_seqs = sorted({lq_seq(p) for p in lq_val})
    report["checks"]["A_lq_seq_disjoint"] = {
        "train_seqs": tr_seqs,
        "val_seqs": va_seqs,
        "overlap": sorted(set(tr_seqs) & set(va_seqs)),
    }
    if set(tr_seqs) & set(va_seqs):
        failures.append(f"A: 龙泉山 train/val 序列重叠 {sorted(set(tr_seqs) & set(va_seqs))}")

    cfg = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    cfg_val = sorted(s["sequence"] for s in cfg["groups"]["val"]["sequences"])
    cfg_train = sorted(s["sequence"] for s in cfg["groups"]["train"]["sequences"])
    report["checks"]["B_lq_split_matches_config"] = {
        "val_actual": va_seqs,
        "val_config": cfg_val,
        "val_expected": sorted(GOLDEN + [BLIND]),
        "train_actual": tr_seqs,
        "train_config": cfg_train,
        "val_match": va_seqs == cfg_val == sorted(GOLDEN + [BLIND]),
        "train_match": tr_seqs == cfg_train,
    }
    if not (va_seqs == cfg_val == sorted(GOLDEN + [BLIND])):
        failures.append("B: 龙泉山 val 序列集与配置/期望不一致")
    if tr_seqs != cfg_train:
        failures.append("B: 龙泉山 train 序列集与配置不一致")

    # 逐帧级留出核验
    val_basenames = {p.name for p in lq_val}
    leaked = sorted(n for n in (f"{s}__frame_{i:06d}.jpg" for s in GOLDEN + [BLIND] for i in range(0, 12000, 5)) if n in val_basenames)
    train_basenames = {p.name for p in lq_train}
    leaked_into_train = sorted(n for n in val_basenames if n in train_basenames)
    report["checks"]["C_lq_frame_level_holdout"] = {
        "val_frames": len(lq_val),
        "train_frames": len(lq_train),
        "identical_basenames_in_both": len(leaked_into_train),
        "examples": leaked_into_train[:5],
    }
    if leaked_into_train:
        failures.append(f"C: 有 {len(leaked_into_train)} 个同名帧同时出现在 train 与 val")

    # ---------------- D Anti-UAV ----------------
    uav_train = sorted(UAV_ROOT.glob("images/train/*.jpg")) + sorted(UAV_ROOT.glob("images/train/*.png"))
    uav_val = sorted(UAV_ROOT.glob("images/val/*.jpg")) + sorted(UAV_ROOT.glob("images/val/*.png"))
    ut_seqs = {uav_seq(p) for p in uav_train}
    uv_seqs = {uav_seq(p) for p in uav_val}
    uav_overlap = sorted(ut_seqs & uv_seqs)
    report["checks"]["D_uav_seq_disjoint"] = {
        "train_seqs": len(ut_seqs),
        "val_seqs": len(uv_seqs),
        "overlap_count": len(uav_overlap),
        "overlap_examples": uav_overlap[:5],
    }
    if uav_overlap:
        failures.append(f"D: Anti-UAV train/val 序列重叠 {len(uav_overlap)} 条")

    # ---------------- E 满额 ----------------
    report["checks"]["E_counts"] = {
        "longquan_train": len(lq_train),
        "longquan_val": len(lq_val),
        "uav_train": len(uav_train),
        "uav_val": len(uav_val),
        "expected": {"longquan_train": 13918, "uav_train": 43008, "uav_val": 31613},
    }
    if len(lq_train) != 13918:
        failures.append(f"E: 龙泉山 train 帧数 {len(lq_train)} ≠ 13918")
    if len(uav_train) != 43008:
        failures.append(f"E: Anti-UAV train 帧数 {len(uav_train)} ≠ 43008")

    report["failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"

    print("=" * 100)
    print(f"  龙泉山 train 序列 {len(tr_seqs)} 条 / {len(lq_train):,} 帧")
    print(f"  龙泉山 val   序列 {len(va_seqs)} 条 / {len(lq_val):,} 帧")
    print(f"  Anti-UAV train {len(ut_seqs)} 条 / {len(uav_train):,} 帧 | val {len(uv_seqs)} 条 / {len(uav_val):,} 帧")
    print("-" * 100)
    for k, v in report["checks"].items():
        print(f"  [{'PASS' if not any(f.startswith(k.split('_')[0]) for f in failures) else 'FAIL'}] {k}: {json.dumps(v, ensure_ascii=False)[:220]}")
    print("-" * 100)
    print(f"  判定: {report['verdict']}")
    if failures:
        for f in failures:
            print(f"    ❌ {f}")
    print("=" * 100)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
