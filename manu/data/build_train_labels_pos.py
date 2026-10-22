#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把龙泉山 **train 划分** 的真值重编号成「位置索引」标签根，解决 stride≠1 下的帧号错位。

为什么必须有这个脚本（两个独立的坑）
------------------------------------
1. **帧号错位（致命）**：`manu/data/multiscale_heatmap_cache.py:403` 写的是
   `frame_index = start + index`，即**排序位置**，而不是文件名里的原始帧号。
   val 划分是 stride=1，位置 == 帧号，所以**恰好**对齐；
   但 train 划分是 **stride=5**（13,918 帧来自原始 69,000+ 帧），
   位置 k 对应原始帧号 5k ⇒ 直接把原始真值目录喂给评测脚本会**整体错位**，
   评测结果完全无意义（而且不会报错，只会静默给出垃圾数字）。

2. **命名陷阱（静默清零）**：`manu/evaluation/eval_multiscale_cache_th.py:100`
   的 `load_ground_truth` 只认 `{sequence}__{digits}.txt`，
   digits 必须是纯数字；数据集自带的 `labels/train/{seq}__frame_000000.txt`
   里 digits 段是 `frame_000000` ⇒ `isdigit()` 为假 ⇒ **该文件被整份跳过**。
   若直接用数据集的 `labels/train` 当 `--labels-root`，
   所有帧都会被判成"无真值" ⇒ TP=0 / FN=0 ⇒ 指标全 0 而**不报任何错**。

本脚本做什么
------------
读取数据集 `labels/train/` 的文件名，解析出 (序列, 原始帧号)，
按原始帧号排序后**重新编号为位置索引**，输出一份
`{sequence}__{position:06d}.txt` 的标签根，
其内容与原文件**逐字节一致**（评测脚本内部照常做它自己的去重），
并与原始合并真值目录做**逐帧内容对账**。

「无真值帧」的处理与 val 口径保持一致：原始真值目录里空文件被
`load_ground_truth` 跳过 ⇒ 该帧没有真值条目；本脚本同样**不输出**空帧的标签文件。

用法（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，
项目根 `/tmp/pycharm_project_10ae9e2e`）：

```bash
python manu/data/build_train_labels_pos.py \
  --dataset-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official \
  --orig-labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --out-root runs/train_fit_diag/labels_train_pos \
  --out-json runs/train_fit_diag/labels_train_pos_manifest.json
```
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# {sequence}__frame_000000
FEATURE_STEM = re.compile(r"^(?P<seq>.+?)__frame_(?P<frame>\d+)$")
# {sequence}__000000
PLAIN_STEM = re.compile(r"^(?P<seq>.+?)__(?P<frame>\d+)$")


def parse_rows(text: str) -> list[tuple[float, ...]]:
    """按 `load_ground_truth` 的口径解析标签行：取第 2~5 列并按 (cx,cy,w,h) 去重。"""
    rows: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for line in text.splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        try:
            row = tuple(float(v) for v in values[1:5])
        except ValueError:
            continue
        if row in seen:
            continue
        seen.add(row)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="生成位置索引化的 train 标签根 + 与原始真值对账")
    ap.add_argument("--dataset-root", required=True, help="official 数据集根目录")
    ap.add_argument("--orig-labels-root", default="", help="原始合并真值目录（用于逐帧对账，可空）")
    ap.add_argument("--out-root", required=True, help="输出的位置索引标签根")
    ap.add_argument("--out-json", default="", help="清单/对账报告输出路径")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    src_dir = dataset_root / "labels" / "train"
    if not src_dir.is_dir():
        raise SystemExit(f"[FATAL] 找不到标签目录：{src_dir}")

    files = sorted(src_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"[FATAL] 标签目录为空：{src_dir}")

    # ---- 解析 (序列, 原始帧号) ----
    by_seq: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    unmatched: list[str] = []
    for path in files:
        m = FEATURE_STEM.match(path.stem)
        if not m:
            unmatched.append(path.name)
            continue
        by_seq[m.group("seq")].append((int(m.group("frame")), path))
    if unmatched:
        print(f"[WARN] {len(unmatched)} 个文件名不符合 `<seq>__frame_<num>` 约定，示例：{unmatched[:3]}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    orig_root = Path(args.orig_labels_root) if args.orig_labels_root else None
    report_seq: list[dict] = []
    total_positions = 0
    total_frames_with_gt = 0
    total_boxes = 0
    total_mismatch = 0
    total_missing_orig = 0
    placeholder_rows = 0
    mismatch_examples: list[dict] = []

    for seq in sorted(by_seq):
        entries = sorted(by_seq[seq], key=lambda item: item[0])
        positions = len(entries)
        frames_with_gt = 0
        boxes = 0
        mismatches = 0
        missing_orig = 0
        seq_mismatch_examples: list[dict] = []
        mapping_sample: list[dict] = []

        for position, (orig_frame, src_path) in enumerate(entries):
            text = src_path.read_text(encoding="utf-8")
            rows = parse_rows(text)
            # 占位行（0 0 0 0 0 之类）不是真目标，必须剔除，否则会在 (0,0) 生成假 GT
            real_rows = [r for r in rows if r[2] > 0 and r[3] > 0]
            placeholder_rows += len(rows) - len(real_rows)
            if real_rows:
                frames_with_gt += 1
                boxes += len(real_rows)
                dst = out_root / f"{seq}__{position:06d}.txt"
                dst.write_text(text, encoding="utf-8")
            else:
                # 与 val 口径一致：无真值的帧不产出标签文件（load_ground_truth 会跳过空文件）
                pass

            if position < 3:
                mapping_sample.append({"position": position, "original_frame": orig_frame})

            # ---- 与原始合并真值逐帧对账 ----
            if orig_root is not None:
                orig_file = orig_root / f"{seq}__{orig_frame:06d}.txt"
                if not orig_file.is_file():
                    missing_orig += 1
                else:
                    orig_rows = [r for r in parse_rows(orig_file.read_text(encoding="utf-8")) if r[2] > 0 and r[3] > 0]
                    src_real = real_rows
                    if len(orig_rows) != len(src_real) or any(
                        abs(a - b) > 1e-6 for ra, rb in zip(orig_rows, src_real) for a, b in zip(ra, rb)
                    ):
                        mismatches += 1
                        if len(seq_mismatch_examples) < 3:
                            seq_mismatch_examples.append(
                                {
                                    "position": position,
                                    "original_frame": orig_frame,
                                    "dataset_rows": len(src_real),
                                    "orig_rows": len(orig_rows),
                                }
                            )

        total_positions += positions
        total_frames_with_gt += frames_with_gt
        total_boxes += boxes
        total_mismatch += mismatches
        total_missing_orig += missing_orig
        mismatch_examples.extend(seq_mismatch_examples)

        report_seq.append(
            {
                "sequence": seq,
                "positions": positions,
                "frames_with_gt": frames_with_gt,
                "frames_without_gt": positions - frames_with_gt,
                "gt_boxes": boxes,
                "position_to_original_sample": mapping_sample,
                "mismatch_vs_orig_labels": mismatches,
                "missing_in_orig_labels": missing_orig,
                "mismatch_examples": seq_mismatch_examples,
            }
        )
        print(
            f"[SEQ] {seq}: positions={positions} 有真值帧={frames_with_gt} "
            f"无真值帧={positions - frames_with_gt} GT框={boxes} 对账不一致={mismatches}",
            flush=True,
        )

    doc = {
        "dataset_root": str(dataset_root),
        "source_label_dir": str(src_dir),
        "orig_labels_root": str(orig_root) if orig_root else None,
        "out_root": str(out_root),
        "sequences": report_seq,
        "totals": {
            "sequences": len(report_seq),
            "positions": total_positions,
            "frames_with_gt": total_frames_with_gt,
            "frames_without_gt": total_positions - total_frames_with_gt,
            "gt_boxes": total_boxes,
            "mismatch_vs_orig_labels": total_mismatch,
            "missing_in_orig_labels": total_missing_orig,
            "placeholder_rows_dropped": placeholder_rows,
        },
        "notes": [
            "缓存里的 frame_index 是排序位置（multiscale_heatmap_cache.py:403 `frame_index = start + index`），"
            "因此标签必须按位置编号，val 因 stride=1 才恰好天然对齐。",
            "数据集自带 labels/train 的文件名含 `frame_` 段，load_ground_truth 的 digits.isdigit() 会整份跳过，"
            "直接使用会得到 TP=0/FN=0 的静默错误结果。",
            "无真值帧不产出标签文件，与 val 口径（空文件被跳过）保持一致。",
        ],
    }
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] 清单 -> {path}")

    print(
        f"[SUMMARY] 序列={len(report_seq)} 位置总数={total_positions} 有真值帧={total_frames_with_gt} "
        f"GT框={total_boxes} 对账不一致={total_mismatch} 原始真值缺失={total_missing_orig} "
        f"占位行剔除={placeholder_rows}",
        flush=True,
    )
    if total_mismatch:
        print(f"[FAIL] 与原始真值存在 {total_mismatch} 处不一致，示例：{mismatch_examples[:5]}", flush=True)
        return 1
    print("[OK] 与原始合并真值逐帧对账全部一致", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
