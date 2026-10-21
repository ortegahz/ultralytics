#!/usr/bin/env python
"""校验龙泉山微调特征数据集的完整性，并冻结一份可复现的验收报告。

校验项
------
1. 结构：images/<group> 与 labels/<group> 严格 1:1 镜像
2. 划分：train / val 序列集合互斥；ignore 序列完全缺席；24 条序列全覆盖
3. 采样：val 帧号连续 0..N-1（含第 0 帧，与 2026-10-20 冻结 baseline 帧集合一致）；
        train 帧号等差 step=5
4. GT 对账：val 侧 GT 行数必须等于 baseline_reference 中冻结的 GT（golden pool 16638 /
        001324 10535）——这是标签链路正确性的硬指标
5. 标签合法性：5 字段、类 id 合法、cx/cy/w/h ∈ [0,1]、w/h > 0
6. 特征健康度：抽样检查 3 通道、尺寸、三通道内容互异（排除全黑/全灰/通道串位）
7. 源一致性：抽样比对数据集标签与合并真值源文件内容
8. data.yaml：路径与 train/val 目录可解析

用法::

    python manu/data/check_longquan_ft_dataset.py \\
        --dataset-root "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median" \\
        --split-config manu/data/longquan_ft_split.json \\
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \\
        --out-json runs/logs/longquan_ft_dataset_check.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

NAME_RE = re.compile(r"^(?P<seq>.+)__frame_(?P<idx>\d{6})\.jpg$")


def read_label_rows(path: Path) -> tuple[list[tuple[float, ...]], int]:
    """与评测端一致的读取 + 去重（键 = cx,cy,w,h 四元组）。"""
    if not path.is_file():
        return [], 0
    seen: set[tuple[float, ...]] = set()
    rows: list[tuple[float, ...]] = []
    raw_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        raw_count += 1
        key = tuple(float(value) for value in values[1:5])
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    return rows, raw_count


def main() -> int:
    parser = argparse.ArgumentParser(description="校验龙泉山微调特征数据集")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--sample-features", type=int, default=40)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    config = json.loads(Path(args.split_config).read_text())
    groups_cfg = config["groups"]
    sampling = config.get("sampling", {})
    labels_root = Path(args.labels_root)

    failures: list[str] = []
    warnings: list[str] = []
    report: dict = {
        "dataset_root": str(root),
        "checks": {},
        "groups": {},
        "failures": failures,
        "warnings": warnings,
    }

    # ---------- 0. data.yaml ----------
    data_yaml = root / "data.yaml"
    yaml_ok = data_yaml.is_file()
    yaml_text = data_yaml.read_text() if yaml_ok else ""
    for required in ("train:", "val:", "names:"):
        if required not in yaml_text:
            failures.append(f"data.yaml 缺少字段 {required}")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        if not (root / sub).is_dir():
            failures.append(f"目录缺失: {sub}")
    report["checks"]["data_yaml"] = {"ok": yaml_ok, "text": yaml_text}

    # ---------- 1~5. 逐组校验 ----------
    expected_gt = {
        "golden_pool": config["baseline_reference"]["golden_pool_single_frame"]["gt"],
        "blind_001324": config["baseline_reference"]["blind_001324_single_frame"]["gt"],
    }
    golden_seqs = {
        e["sequence"]
        for e in groups_cfg["val"]["sequences"]
        if e.get("role") == "golden_pool"
    }
    blind_seqs = {
        e["sequence"]
        for e in groups_cfg["val"]["sequences"]
        if e.get("role") == "blind_hardcase"
    }

    seq_to_group: dict[str, str] = {}
    for group in ("val", "train"):
        img_dir = root / "images" / group
        lbl_dir = root / "labels" / group
        if not img_dir.is_dir():
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".jpg")
        labels = sorted(p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt") if lbl_dir.is_dir() else []

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        missing_lbl = img_stems - lbl_stems
        orphan_lbl = lbl_stems - img_stems
        if missing_lbl:
            failures.append(f"{group}: {len(missing_lbl)} 张图缺标签（例 {sorted(missing_lbl)[:3]}）")
        if orphan_lbl:
            failures.append(f"{group}: {len(orphan_lbl)} 个标签无对应图（例 {sorted(orphan_lbl)[:3]}）")

        # 帧号解析
        per_seq: dict[str, list[int]] = defaultdict(list)
        for path in images:
            m = NAME_RE.match(path.name)
            if not m:
                failures.append(f"{group}: 文件名不符合 <seq>__frame_<idx:06d>.jpg: {path.name}")
                continue
            per_seq[m.group("seq")].append(int(m.group("idx")))

        # 标签统计
        gt_rows = 0
        gt_rows_raw = 0
        empty_labels = 0
        bad_rows = 0
        bad_samples: list[str] = []
        class_ids: set[int] = set()
        seq_gt: dict[str, int] = defaultdict(int)
        for path in labels:
            rows, raw_count = read_label_rows(path)
            gt_rows += len(rows)
            gt_rows_raw += raw_count
            if not rows:
                empty_labels += 1
            seq = path.stem.split("__frame_")[0]
            seq_gt[seq] += len(rows)
            for line in path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) < 5:
                    bad_rows += 1
                    continue
                class_ids.add(int(float(values[0])))
                cx, cy, w, h = (float(v) for v in values[1:5])
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    bad_rows += 1
                    if len(bad_samples) < 5:
                        bad_samples.append(f"{path.name}: {line}")
        if bad_rows:
            failures.append(f"{group}: {bad_rows} 行标签越界或格式非法（例 {bad_samples}）")
        if class_ids - {0}:
            failures.append(f"{group}: 出现非法类 id {sorted(class_ids - {0})}")

        # 采样口径
        expected_stride = sampling.get("val_stride", 1) if group == "val" else sampling.get("train_stride", 5)
        expected_start = sampling.get("val_start", 0) if group == "val" else sampling.get("train_start", 0)
        for seq, idxs in per_seq.items():
            idxs_sorted = sorted(idxs)
            if len(idxs_sorted) != len(set(idxs_sorted)):
                failures.append(f"{group}/{seq}: 帧号有重复")
            diffs = {idxs_sorted[i + 1] - idxs_sorted[i] for i in range(len(idxs_sorted) - 1)}
            if expected_stride == 1:
                if diffs and diffs != {1}:
                    failures.append(f"{group}/{seq}: 期望连续帧，实际步长集合 {sorted(diffs)}")
                if idxs_sorted and idxs_sorted[0] != expected_start:
                    failures.append(f"{group}/{seq}: 首帧应为 {expected_start}，实际 {idxs_sorted[0]}")
                if idxs_sorted and idxs_sorted[-1] - idxs_sorted[0] + 1 != len(idxs_sorted):
                    failures.append(f"{group}/{seq}: 帧号不连续")
            else:
                if diffs and diffs != {expected_stride}:
                    failures.append(f"{group}/{seq}: 期望等差 {expected_stride}，实际步长集合 {sorted(diffs)}")
            seq_to_group[seq] = group

        # 与配置声明的序列集合比对
        declared = {e["sequence"] for e in groups_cfg[group]["sequences"]}
        actual = set(per_seq)
        if declared != actual:
            failures.append(f"{group}: 序列集合与配置不符，多={sorted(actual - declared)} 少={sorted(declared - actual)}")

        report["groups"][group] = {
            "images": len(images),
            "labels": len(labels),
            "sequences": len(per_seq),
            "gt_rows_deduped": gt_rows,
            "gt_rows_raw": gt_rows_raw,
            "duplicate_rows_removed": gt_rows_raw - gt_rows,
            "empty_label_files": empty_labels,
            "bad_rows": bad_rows,
            "class_ids": sorted(class_ids),
            "expected_stride": expected_stride,
            "per_sequence": {
                seq: {
                    "frames": len(per_seq[seq]),
                    "first": min(per_seq[seq]),
                    "last": max(per_seq[seq]),
                    "gt": seq_gt.get(seq, 0),
                }
                for seq in sorted(per_seq)
            },
        }

    # ---------- 2. 划分互斥 / ignore 缺席 ----------
    train_seqs = {s for s, g in seq_to_group.items() if g == "train"}
    val_seqs = {s for s, g in seq_to_group.items() if g == "val"}
    overlap = train_seqs & val_seqs
    if overlap:
        failures.append(f"train/val 序列泄漏: {sorted(overlap)}")
    ignore_seqs = {e["sequence"] for e in groups_cfg.get("ignore", {}).get("sequences", [])}
    leaked_ignore = ignore_seqs & (train_seqs | val_seqs)
    if leaked_ignore:
        failures.append(f"ignore 序列进入了数据集: {sorted(leaked_ignore)}")
    all_24 = {e["sequence"] for g in ("val", "train", "ignore") for e in groups_cfg[g]["sequences"]}
    report["checks"]["split"] = {
        "train_sequences": sorted(train_seqs),
        "val_sequences": sorted(val_seqs),
        "ignore_sequences": sorted(ignore_seqs),
        "train_val_overlap": sorted(overlap),
        "ignore_leaked": sorted(leaked_ignore),
        "configured_total": len(all_24),
        "covered": len(train_seqs | val_seqs | ignore_seqs),
    }
    if len(train_seqs | val_seqs | ignore_seqs) != len(all_24):
        failures.append("三组合计未覆盖全部 24 条序列")

    # ---------- 4. GT 对账（硬指标） ----------
    gt_val = report["groups"].get("val", {}).get("per_sequence", {})
    golden_gt = sum(gt_val.get(s, {}).get("gt", 0) for s in golden_seqs)
    blind_gt = sum(gt_val.get(s, {}).get("gt", 0) for s in blind_seqs)
    golden_expect = expected_gt["golden_pool"]
    blind_expect = expected_gt["blind_001324"]
    golden_ok = golden_gt == golden_expect
    blind_ok = blind_gt == blind_expect
    if not golden_ok:
        failures.append(f"黄金池 GT 对账失败: 数据集 {golden_gt} vs baseline {golden_expect}")
    if not blind_ok:
        failures.append(f"001324 GT 对账失败: 数据集 {blind_gt} vs baseline {blind_expect}")
    report["checks"]["gt_reconciliation"] = {
        "golden_pool": {"dataset": golden_gt, "baseline": golden_expect, "match": golden_ok},
        "blind_001324": {"dataset": blind_gt, "baseline": blind_expect, "match": blind_ok},
    }

    # ---------- 6. 特征健康度抽样 ----------
    sample_results = []
    rng = random.Random(20261021)
    for group in ("val", "train"):
        img_dir = root / "images" / group
        if not img_dir.is_dir():
            continue
        pool = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".jpg")
        if not pool:
            continue
        picks = rng.sample(pool, min(args.sample_features, len(pool)))
        for path in picks:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                failures.append(f"{group}: 特征图无法解码 {path.name}")
                continue
            if img.ndim != 3 or img.shape[2] != 3:
                failures.append(f"{group}: 特征图不是 3 通道 {path.name} shape={img.shape}")
                continue
            ch = [img[:, :, i].astype(np.float32) for i in range(3)]
            same_01 = float(np.abs(ch[0] - ch[1]).max())
            same_02 = float(np.abs(ch[0] - ch[2]).max())
            nonzero = [float((c > 0).mean()) for c in ch]
            entry = {
                "group": group,
                "file": path.name,
                "shape": list(img.shape),
                "mean": [round(float(c.mean()), 3) for c in ch],
                "nonzero_ratio": [round(v, 4) for v in nonzero],
                "ch0_vs_ch1_maxdiff": round(same_01, 3),
                "ch0_vs_ch2_maxdiff": round(same_02, 3),
            }
            if same_01 == 0.0 and same_02 == 0.0:
                failures.append(f"{group}: 三通道内容完全相同（疑似通道串位）{path.name}")
            if all(v == 0.0 for v in nonzero):
                failures.append(f"{group}: 特征图全黑 {path.name}")
            sample_results.append(entry)
    report["checks"]["feature_health"] = {
        "sampled": len(sample_results),
        "all_3ch": all(e["shape"][2] == 3 for e in sample_results) if sample_results else None,
        "distinct_channels": all(
            not (e["ch0_vs_ch1_maxdiff"] == 0.0 and e["ch0_vs_ch2_maxdiff"] == 0.0) for e in sample_results
        )
        if sample_results
        else None,
        "samples": sample_results[:10],
    }

    # ---------- 7. 源标签一致性抽样 ----------
    mismatch = 0
    checked = 0
    for group in ("val", "train"):
        lbl_dir = root / "labels" / group
        if not lbl_dir.is_dir():
            continue
        pool = sorted(lbl_dir.glob("*.txt"))
        if not pool:
            continue
        for path in rng.sample(pool, min(20, len(pool))):
            seq, _, idx = path.stem.partition("__frame_")
            src = labels_root / f"{seq}__{int(idx):06d}.txt"
            a, _ = read_label_rows(path)
            b, _ = read_label_rows(src)
            checked += 1
            if a != b:
                mismatch += 1
                if mismatch <= 3:
                    failures.append(f"标签与真值源不一致: {path.name}")
    report["checks"]["source_label_parity"] = {"checked": checked, "mismatch": mismatch}

    # ---------- 汇总 ----------
    report["summary"] = {
        "status": "PASS" if not failures else "FAIL",
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "val_images": report["groups"].get("val", {}).get("images", 0),
        "train_images": report["groups"].get("train", {}).get("images", 0),
        "val_gt_rows": report["groups"].get("val", {}).get("gt_rows_deduped", 0),
        "train_gt_rows": report["groups"].get("train", {}).get("gt_rows_deduped", 0),
    }

    print("=" * 92)
    print(f"数据集校验: {report['summary']['status']}   ({args.dataset_root})")
    print("=" * 92)
    for group, info in report["groups"].items():
        print(
            f"  {group:>5}: images={info['images']:>7} labels={info['labels']:>7} seqs={info['sequences']:>3} "
            f"GT={info['gt_rows_deduped']:>7} (去重前 {info['gt_rows_raw']}, 重复 {info['duplicate_rows_removed']}) "
            f"空标签={info['empty_label_files']:>6} 非法行={info['bad_rows']}"
        )
    gt = report["checks"]["gt_reconciliation"]
    print(
        f"  GT 对账: 黄金池 {gt['golden_pool']['dataset']} vs baseline {gt['golden_pool']['baseline']} "
        f"[{'MATCH' if gt['golden_pool']['match'] else 'MISMATCH'}] | "
        f"001324 {gt['blind_001324']['dataset']} vs baseline {gt['blind_001324']['baseline']} "
        f"[{'MATCH' if gt['blind_001324']['match'] else 'MISMATCH'}]"
    )
    print(f"  划分: train={len(train_seqs)} val={len(val_seqs)} ignore={len(ignore_seqs)} "
          f"泄漏={len(overlap)} ignore入组={len(leaked_ignore)}")
    print(f"  特征抽样: {report['checks']['feature_health']['sampled']} 张, "
          f"全3通道={report['checks']['feature_health']['all_3ch']}, "
          f"通道互异={report['checks']['feature_health']['distinct_channels']}")
    print(f"  源标签一致性: 检查 {checked} 个, 不一致 {mismatch} 个")

    if failures:
        print(f"\n[FAIL] {len(failures)} 项：")
        for item in failures:
            print(f"   - {item}")
    else:
        print("\n[PASS] 全部校验项通过。")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n报告已写入: {out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
