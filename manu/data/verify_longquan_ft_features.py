#!/usr/bin/env python
"""龙泉山微调数据集 —— 特征逐字节验证 + 缺失帧补齐（copy 模式）。

用法::

    python manu/data/verify_longquan_ft_features.py \\
        --dataset-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median \\
        --cache-root /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train \\
        --split-config manu/data/longquan_ft_split.json \\
        --fill-missing --out-json runs/logs/lq_ft_feature_parity.json

设计要点
--------
* **补齐由标签目录驱动**：``labels/<group>/`` 里的文件清单就是该组应有的目标帧集合，
  因此不会像「按缓存全量拷」那样把 train 组多拷进未抽帧的中间帧。
* **只补特征图，绝不碰标签**：旧缓存 ``longquanshan_ir_gmc_median/labels/train`` 的标签
  全是 0 字节空文件，标签必须来自 ``Final_Labels_merged_20261010``。
* **逐字节比对**：sha256 逐文件比对，输出逐序列报告。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="龙泉山微调数据集特征逐字节验证 / 补齐")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cache-root", required=True, help="既有特征缓存目录（扁平布局）")
    parser.add_argument("--split-config", default="")
    parser.add_argument("--groups", default="val,train")
    parser.add_argument("--fill-missing", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    cache = Path(args.cache_root)
    report: dict = {"dataset_root": str(root), "cache_root": str(cache), "groups": {}}
    failures: list[str] = []

    declared: dict[str, list[str]] = {}
    if args.split_config:
        config = json.loads(Path(args.split_config).read_text())
        for group, info in config["groups"].items():
            declared[group] = [e["sequence"] for e in info["sequences"]]

    for group in [g.strip() for g in args.groups.split(",") if g.strip()]:
        img_dir = root / "images" / group
        lbl_dir = root / "labels" / group
        if not img_dir.is_dir():
            continue

        # 目标帧集合 = 标签目录清单（权威，且天然是抽帧后的子集）
        targets: dict[str, list[str]] = {}
        if lbl_dir.is_dir():
            for lbl in sorted(lbl_dir.glob("*.txt")):
                seq = lbl.stem.split("__frame_")[0]
                targets.setdefault(seq, []).append(lbl.stem + ".jpg")
        for seq in declared.get(group, []):
            targets.setdefault(seq, [])

        per_seq: dict[str, dict] = {}
        filled = 0
        for seq, names in sorted(targets.items()):
            entry = {"expected": len(names), "identical": 0, "mismatch": [], "missing_in_cache": 0, "absent": 0}
            if args.fill_missing:
                # 先以既有缓存为准把该序列的目标帧补齐/对齐，再重新统计
                for name in names:
                    src = cache / name
                    if not src.is_file():
                        continue
                    dst = img_dir / name
                    if dst.is_file() and sha256(dst) == sha256(src):
                        continue
                    if dst.exists():
                        dst.unlink()
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                    filled += 1
            for name in names:
                dst = img_dir / name
                src = cache / name
                if not src.is_file():
                    entry["missing_in_cache"] += 1
                    continue
                if not dst.is_file():
                    entry["absent"] += 1
                    continue
                if sha256(dst) == sha256(src):
                    entry["identical"] += 1
                else:
                    entry["mismatch"].append(name)
            per_seq[seq] = entry

        # 数据集里多出来的（不属于目标集合的）文件
        expected_names = {name for names in targets.values() for name in names}
        extra = [p.name for p in img_dir.glob("*.jpg") if p.name not in expected_names]

        total = sum(e["expected"] for e in per_seq.values())
        identical = sum(e["identical"] for e in per_seq.values())
        mismatch = sum(len(e["mismatch"]) for e in per_seq.values())
        absent = sum(e["absent"] for e in per_seq.values())
        missing_cache = sum(e["missing_in_cache"] for e in per_seq.values())

        report["groups"][group] = {
            "sequences": {
                seq: {
                    "expected": e["expected"],
                    "identical": e["identical"],
                    "mismatch": len(e["mismatch"]),
                    "absent": e["absent"],
                    "missing_in_cache": e["missing_in_cache"],
                }
                for seq, e in per_seq.items()
            },
            "files_filled": filled,
            "extra_files": extra[:20],
            "extra_count": len(extra),
        }

        print("=" * 96)
        print(
            f"[{group}] 逐字节比对: 相同 {identical}/{total}  不一致 {mismatch}  缺失 {absent}  "
            f"缓存中无 {missing_cache}  多余文件 {len(extra)}  本次补齐 {filled}"
        )
        for seq, e in sorted(per_seq.items()):
            flag = "OK " if e["identical"] == e["expected"] else "!! "
            print(f"  {flag}{seq:<32} {e['identical']:>6}/{e['expected']:<6} 不一致={len(e['mismatch'])} 缺失={e['absent']}")
            if e["mismatch"]:
                failures.append(f"{group}/{seq}: {len(e['mismatch'])} 帧与既有缓存不一致（例 {e['mismatch'][:3]}）")
            if e["absent"]:
                failures.append(f"{group}/{seq}: {e['absent']} 帧缺失且未补齐")
            if e["missing_in_cache"]:
                failures.append(f"{group}/{seq}: {e['missing_in_cache']} 帧在既有缓存中不存在")

    if failures:
        print(f"\n[FAIL] {len(failures)} 项")
        for item in failures:
            print(f"   - {item}")
    else:
        print("\n[PASS] 全部特征与既有缓存逐字节一致。")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告已写入: {out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
