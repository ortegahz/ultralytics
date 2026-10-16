#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板卡 e2e demo 的检测 TSV -> npu_tsv_to_cache.py 能吃的 stem 口径

为什么需要这一步
----------------
两臂的 TSV 第一列都不是 GT 的 im_name：

  串行臂 e2e_demo      `<seq>_<k:06d>.bmp`     例 `34_1_000000.bmp`
  并行臂 e2e_par2_demo `frame_<k:06d>`         例 `frame_000000`（两 worker 交错写）

而 npu_tsv_to_cache.py 是按 **GT 的 im_name stem** 去查表的
（npu_tsv_to_cache.py:104-110），PC cache 里是 `<seq>__<idx:06d>.bmp`。
所以必须先把第一列改写成 GT 口径，否则配对数为 0。

帧偏移
------
demo 从原始目录的第 1 张开始读，k=0 <-> 原始 `000001.bmp`；
GT stem 从 `<seq>__000003` 起（头两帧无标注）。
故  k -> <seq>__{k+1:06d}.bmp，且 k=0,1 丢弃。

  每序列 demo 输出 want+2 帧，评测用 want 帧
  24 序列：demo 31661 帧 -> 评测 31613 帧（31661 - 24*2 = 31613）。

并行臂合并
----------
npu_tsv_to_cache.py 只 glob `*/npu_det.tsv`，而 par2 写的是 `npu_det.w{0,1}.tsv`。
本脚本把 w0/w1 按 k 合并成一个 `npu_det.tsv`（行序按 k 升序重建，与串行臂一致）。

硬闸门（任一不过 -> 非零退出，不产出半成品）
---------------------------------------------
  * 每个序列的行数必须等于 expected.tsv 的 want，差 1 帧都不放行
  * 同一臂内 k 不得重复
  * k 必须是 want+2 的连续区间
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="realign board e2e demo detection TSV to GT im_name stems")
    p.add_argument("--tsv-root", required=True,
                   help="dir holding <seq>/npu_det.tsv (serial) or <seq>/npu_det.w{0,1}.tsv (parallel)")
    p.add_argument("--arm", required=True, choices=["serial", "parallel"])
    p.add_argument("--expected", required=True,
                   help="expected.tsv: <seq>\\t<want> per line")
    p.add_argument("--out", required=True, help="output dir to receive <seq>/npu_det.tsv")
    p.add_argument("--expect-total", type=int, default=31613,
                   help="sum of want across all sequences (hard gate)")
    return p.parse_args()


def read_expected(path):
    want = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split("\t") if "\t" in ln else ln.split()
        want[parts[0]] = int(parts[1])
    return want


def parse_k(stem, seq):
    """从 stem 取 k。容忍串行臂的 `<seq>_<k:06d>.bmp` 与并行臂的 `frame_<k:06d>`。"""
    tail = stem.rsplit("_", 1)[-1].rsplit(".", 1)[0]  # 串行臂带 .bmp 后缀，并行臂不带
    if not tail.isdigit():
        raise ValueError(f"{seq}: stem {stem!r} 末段不是数字，取不到 k")
    return int(tail)


def main():
    a = parse_args()
    root = Path(a.tsv_root)
    out_root = Path(a.out)
    want = read_expected(a.expected)
    print(f"[EXP ] {len(want)} 序列，GT 帧合计 {sum(want.values())}")

    fail = []
    seq_hashes = {}
    rows_out_total = 0
    payload_chunks = {}

    for seq, w in sorted(want.items()):
        if a.arm == "serial":
            src = [root / seq / "npu_det.tsv"]
        else:
            src = [root / seq / "npu_det.w0.tsv", root / seq / "npu_det.w1.tsv"]

        by_k = defaultdict(list)
        for f in src:
            if not f.exists():
                fail.append(f"{seq}: 缺 TSV {f}")
                continue
            with open(f) as fh:
                for ln in fh:
                    ln = ln.rstrip("\n")
                    if not ln:
                        continue
                    parts = ln.split("\t")
                    if len(parts) < 2:
                        fail.append(f"{seq}: 非法 TSV 行 {ln!r}")
                        continue
                    k = parse_k(parts[0], seq)
                    by_k[k].append("\t".join(parts[1:]))

        if fail:
            break

        n_in = len(by_k)
        if n_in != w + 2:
            fail.append(f"{seq}: TSV 帧数 {n_in} != want+2 = {w + 2}")
            continue
        if set(by_k) != set(range(w + 2)):
            miss = sorted(set(range(w + 2)) - set(by_k))
            extra = sorted(set(by_k) - set(range(w + 2)))
            fail.append(f"{seq}: k 不连续，缺 {miss[:5]} 多 {extra[:5]}")
            continue

        # k=0,1 无 GT，丢弃；k>=2 -> <seq>__<k+1:06d>.bmp
        lines, per_seq = [], []
        for k in range(2, w + 2):
            dets = by_k[k]
            lines.append(f"{seq}__{k + 1:06d}.bmp\t" + "\t".join(dets))
            per_seq.append((seq, k, dets))

        if len(lines) != w:
            fail.append(f"{seq}: 重建后 {len(lines)} 行 != want {w}")
            continue

        (out_root / seq).mkdir(parents=True, exist_ok=True)
        body = "\n".join(lines) + "\n"
        (out_root / seq / "npu_det.tsv").write_text(body)
        seq_hashes[seq] = hashlib.md5(body.encode()).hexdigest()
        payload_chunks[seq] = per_seq
        rows_out_total += len(lines)
        # dets[0] == "<ndet>\t<x y score>\t..."，拆 tab 后段数 = 1 + ndet
        ndet = sum(int(d[2][0].split("\t")[0]) for d in per_seq)
        print(f"[OK  ] {seq:<28} frames={len(lines):>5} dets={ndet:>5} md5={seq_hashes[seq]}")

    if fail:
        print(f"[FAIL] {len(fail)} 项，前 5 条：")
        for m in fail[:5]:
            print(f"        {m}")
        sys.exit(2)

    if rows_out_total != a.expect_total:
        print(f"[FAIL] 总行数 {rows_out_total} != 期望 {a.expect_total}")
        sys.exit(3)

    # 全臂 payload 哈希（只含 k>=2，即评测口径），供两臂逐帧比对
    allh = hashlib.md5()
    for seq in sorted(payload_chunks):
        for s, k, dets in payload_chunks[seq]:
            allh.update(f"{s}__{k + 1:06d}.bmp|{';'.join(dets)}\n".encode())
    print(f"[OUT ] 总行 {rows_out_total}（== expect {a.expect_total}）")
    print(f"[HASH] {a.arm} payload md5 = {allh.hexdigest()}")

    (out_root.parent / f"{a.arm}_payload_md5.txt").write_text(allh.hexdigest() + "\n")


if __name__ == "__main__":
    main()
