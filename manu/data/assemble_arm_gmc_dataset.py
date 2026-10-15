#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把板卡 GMC 输出组装成 ultralytics 可读的数据集目录

命名映射（关键，已用逐像素比对实证）
--------------------------------
板卡写的是   <seq>_<k:06d>.bmp        k 从 0 开始，对应 raw 目录 natural 排序后的第 k+1 个文件
官方 val 是  <seq>__<i:06d>.bmp        i 从 3 开始

依据冻结构建器 build_full_median_dataset.py 的索引逻辑：
  idx_map = {int(stem 尾部数字): list_index}     # 由 raw 目录 natural 排序建立
  curr_list_idx = idx_map[frame_idx]
  => <seq>__<i> 取的是 raw 列表第 i 个（即文件 00000i.bmp）

板卡 feat k 是 raw 列表第 k 个 => raw 文件 00000(k+1).bmp => 官方 __<k+1>。

所以  <seq>_<k>  <->  <seq>__<k+1>。
官方 val 从 __000003 起算，故 k=0,1 两帧（对应 i=1,2）在官方里不存在，必须丢弃。
每条序列: 板卡 feat 数 = raw 数，丢弃 2 帧后正好等于官方帧数
        （31,661 raw -> 31,613 官方，24x2 = 48）。

实测验证（cmp_arm_pc_features.py 偏移扫描，ARM vs PC 同源重建）：
  25_1     off=1 MAE=0.00022 Max|D|=2     off=3 MAE=0.44577 Max|D|=7
  34_1     off=1 MAE=0.00041 Max|D|=2     off=3 MAE=0.53627 Max|D|=68
  01_4485  off=1 MAE=0.06503              off=3 MAE=1.25581 Max|D|=248
=> off=1 是唯一给出近乎逐字节一致的偏移；off=3 会引入视频运动量级的误差。

实现用硬链接：staging 与数据集同盘，零拷贝、零额外空间。
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

OFFSET = 1          # __<k+OFFSET>，实证值
DROP_BELOW = 2      # k < 2 对应 i=1,2，官方 val 不存在


def parse_args():
    p = argparse.ArgumentParser(description="Assemble board GMC features into a YOLO dataset dir")
    p.add_argument("--stage", required=True, help="staging root holding <seq>/feat/<seq>_<k>.bmp")
    p.add_argument("--out", required=True, help="output dataset root")
    p.add_argument("--labels-from", required=True, help="dataset root to hardlink labels/val from")
    p.add_argument("--expect-total", type=int, default=31613)
    p.add_argument("--verify-md5", type=int, default=0, help="md5-verify N randomly chosen links")
    return p.parse_args()


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    a = parse_args()
    stage = Path(a.stage)
    out_img = Path(a.out) / "images" / "val"
    out_lbl = Path(a.out) / "labels" / "val"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # ---- labels: hardlink copy (只读文件名/内容，不做重压缩) ----
    lbl_src = Path(a.labels_from) / "labels" / "val"
    n_lbl = 0
    for f in sorted(lbl_src.glob("*.txt")):
        dst = out_lbl / f.name
        if not dst.exists():
            try:
                os.link(f, dst)
            except OSError:
                dst.write_bytes(f.read_bytes())
        n_lbl += 1
    print(f"[labels] hardlinked {n_lbl} from {lbl_src}")

    # ---- images: <seq>_<k>.bmp -> <seq>__<k+3>.bmp ----
    seqs = sorted(d for d in stage.iterdir() if d.is_dir() and (d / "feat").is_dir())
    total = dropped = 0
    per_seq = {}
    for sd in seqs:
        seq = sd.name
        n = 0
        for f in sorted((sd / "feat").glob(f"{seq}_*.bmp")):
            k = int(f.stem.rsplit("_", 1)[1])
            if k < DROP_BELOW:
                dropped += 1
                continue
            dst = out_img / f"{seq}__{k + OFFSET:06d}.bmp"
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    dst.write_bytes(f.read_bytes())
            n += 1
        per_seq[seq] = n
        total += n

    print(f"[images] sequences={len(seqs)}  linked={total}  dropped(k<{DROP_BELOW})={dropped}")
    for s in sorted(per_seq):
        print(f"          {s:28s} {per_seq[s]}")

    if a.expect_total and total != a.expect_total:
        print(f"[FAIL] 帧数 {total} != 期望 {a.expect_total}")
        sys.exit(2)
    print(f"[OK]   帧数校验通过: {total}")

    # ---- 抽样 md5 校验（链接是否真的指向板卡原文件） ----
    if a.verify_md5 > 0:
        import random
        files = sorted(out_img.glob("*.bmp"))
        random.seed(20261015)
        picked = random.sample(files, min(a.verify_md5, len(files)))
        ok = 0
        for p in picked:
            seq = p.name.split("__")[0]
            i = int(p.stem.split("__")[1])
            src = stage / seq / "feat" / f"{seq}_{i - OFFSET:06d}.bmp"
            if src.exists() and md5(src) == md5(p):
                ok += 1
            else:
                print(f"[FAIL] md5 不一致: {p.name} <- {src}")
        print(f"[md5 ] 抽样 {len(picked)} 条, 一致 {ok}/{len(picked)}")
        if ok != len(picked):
            sys.exit(2)

    (Path(a.out) / "data.yaml").write_text(
        f"path: {Path(a.out).resolve()}\ntrain: images/val\nval: images/val\n\nnames:\n  0: uav\n",
        encoding="utf-8")
    print(f"[WROTE] {Path(a.out).resolve()}/data.yaml")


if __name__ == "__main__":
    main()
