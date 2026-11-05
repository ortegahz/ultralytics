#!/usr/bin/env python3
"""24 序 no-GMC 特征集 vs 8 序 no-GMC：**逐字节等价性自检**（8 条 gate 序列）。

为什么必须做：
  24 序 no-GMC 数据集（--ref-dataset longquanshan_ir_gmc_median）和 8 序 no-GMC 数据集
  （longquanshan_ft_gmc_median_nogmc）用的是**同一份代码、同一个 raw-root**，
  只是参照集的序列集合不同 ⇒ 落在同一条序列上的帧，W≡IDENTITY 的计算结果必须**完全一样**。
  同输入数组 ⇒ 同 JPEG 编码 ⇒ **md5 必须逐字节相同**。
  这条自检是「8 序实验的结论能不能外推到 24 序」的地基。

用法：
  python manu/tracking/lookahead/server_overnight/check_nogmc24_byte_equal.py \
      --new /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc/images/train \
      --old /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/images/val \
      --out-json runs/lookahead/nogmc24_byte_equal.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

GATE_SEQS = [
    "VIDEO00006_19700101_000703",
    "VIDEO00007_19700101_000818",
    "VIDEO00007_19700101_001324",
    "VIDEO00008_19700101_001307",
    "VIDEO00009_19700101_001531",
    "VIDEO00029_19700101_014053",
    "VIDEO00033_19700101_015000",
    "VIDEO00034_19700101_015052",
]


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="24 序 no-GMC 的 images/train")
    ap.add_argument("--old", required=True, help="8 序 no-GMC 的 images/val")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    report: dict = {
        "new_root": args.new,
        "old_root": args.old,
        "sequences": [],
        "total_files": 0,
        "identical_files": 0,
        "mismatch_files": 0,
        "missing_in_new": 0,
        "missing_in_old": 0,
    }

    all_ok = True
    for seq in GATE_SEQS:
        new_files = sorted(f for f in os.listdir(args.new) if f.startswith(seq + "__frame_"))
        old_files = sorted(f for f in os.listdir(args.old) if f.startswith(seq + "__frame_"))
        entry = {
            "sequence": seq,
            "new_count": len(new_files),
            "old_count": len(old_files),
            "count_equal": len(new_files) == len(old_files),
            "identical": 0,
            "mismatched": [],
            "missing_in_new": [],
            "missing_in_old": [],
        }
        old_set = set(old_files)
        for fname in new_files:
            if fname not in old_set:
                entry["missing_in_old"].append(fname)
                continue
            if md5(os.path.join(args.new, fname)) == md5(os.path.join(args.old, fname)):
                entry["identical"] += 1
            else:
                entry["mismatched"].append(fname)
        new_set = set(new_files)
        for fname in old_files:
            if fname not in new_set:
                entry["missing_in_new"].append(fname)
        report["sequences"].append(entry)
        report["total_files"] += len(new_files)
        report["identical_files"] += entry["identical"]
        report["mismatch_files"] += len(entry["mismatched"])
        report["missing_in_new"] += len(entry["missing_in_new"])
        report["missing_in_old"] += len(entry["missing_in_old"])
        ok = (
            entry["count_equal"]
            and not entry["mismatched"]
            and not entry["missing_in_new"]
            and not entry["missing_in_old"]
        )
        all_ok = all_ok and ok
        print(
            f"{'PASS' if ok else 'FAIL'}  {seq}  new={entry['new_count']} old={entry['old_count']} "
            f"identical={entry['identical']} mismatch={len(entry['mismatched'])} "
            f"missing_new={len(entry['missing_in_new'])} missing_old={len(entry['missing_in_old'])}"
        )

    report["verdict"] = "PASS" if all_ok else "FAIL"
    print(
        f"\n总计 {report['total_files']} 文件 | 逐字节相同 {report['identical_files']} | "
        f"不一致 {report['mismatch_files']} | 缺失(new/old) {report['missing_in_new']}/{report['missing_in_old']}"
    )
    print(f"VERDICT: {report['verdict']}")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"写出 {args.out_json}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
