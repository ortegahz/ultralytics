#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板卡 NPU TSV -> 与 PC 侧同格式的 inference cache pkl

为什么 GT 要从 PC cache 取，而不是板卡的 gt_canvas
--------------------------------------------------
板端 `hm_score_board_det.py` 用的 GT 来源是 `r["gt_canvas"]`（CPU 侧另行算好），
与 PC cache 里的 `gt_pts` 不是同一条产出链。本次要做的三臂对照要求
**GT 逐项一致**（eval_arm_pc_gmc_ab.py 会硬校验），所以 GT 一律从 PC cache 取，
只把预测点换成板卡 NPU 的输出。任何一臂的 GT 都不是我们放的。

坐标
----
板端写的是 640x640 letterbox 画布像素：
    x_canvas = (gx + dx) * stride      (rknn_heatmap_smoke.cpp:260)
PC cache 同式：
    pred_x   = (x_coords + dx) * stride (heatmap_evaluate.py:72-73)
两式逐字相同 ⇒ 同一坐标系，无需映射。已实测 34_1 前 40 帧 top1 距离
中位数 0.223 px、最大 1.354 px，40/40 在 8px 内。

TSV 格式（rknn_heatmap_smoke.cpp:606-616）
------------------------------------------
    <basename>\\t<ndet>\\t<x y score>\\t<x y score>...
每组三元组各占一个独立 tab 列。0 检出写 `name\\t0`（仅两列），整行不会跳过；
缺失帧 = 该 stem 完全没有行（通常意味着整轮中止，不是 0 检出）。

dtype
-----
pred_points 必须 float32：fp16 曾造成 3 TP / 3 FP 的口径漂移
（cache_trial0474_inferences.py:221-228 注释有记录）。pred_scores 用 fp16 无妨，
这里统一存 float32，避免三臂之间出现无谓差异。
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="board NPU TSV -> inference cache pkl")
    p.add_argument("--tsv-root", required=True, help="board-side dir holding <seq>/npu_det.tsv")
    p.add_argument("--gt-cache", required=True, help="PC cache pkl to take GT from")
    p.add_argument("--out", required=True)
    p.add_argument("--expect-total", type=int, default=31613)
    p.add_argument("--im-ext", default=".bmp")
    return p.parse_args()


def parse_tsv(path):
    """返回 {stem: (points (N,2) float32, scores (N,) float32)}；ndet==0 是「在场但空」。"""
    out = {}
    with open(path) as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 2:
                raise ValueError(f"列数 <2（不是合法 TSV 行）: {ln!r}")
            stem = Path(parts[0]).stem
            n = int(parts[1])
            if n == 0:
                out[stem] = (np.zeros((0, 2), np.float32), np.zeros((0,), np.float32))
                continue
            if len(parts) < 2 + n:
                raise ValueError(f"{stem}: 声称 {n} 个检出，实际只有 {len(parts)-2} 列")
            arr = np.array([[float(v) for v in parts[2 + k].split()] for k in range(n)],
                           dtype=np.float32)
            if arr.shape[1] != 3:
                raise ValueError(f"{stem}: 三元组列宽 {arr.shape[1]} != 3")
            out[stem] = (np.ascontiguousarray(arr[:, :2]), np.ascontiguousarray(arr[:, 2]))
    return out


def main():
    a = parse_args()
    root = Path(a.tsv_root)

    gt_rows = pickle.load(open(a.gt_cache, "rb"))
    print(f"[GT ] 来源 {a.gt_cache}，{len(gt_rows)} 条")

    dets, n_zero_frames, bad = {}, 0, []
    files = sorted(root.glob("*/npu_det.tsv"))
    print(f"[TSV] {len(files)} 个序列 TSV")
    for f in files:
        try:
            d = parse_tsv(f)
        except ValueError as e:
            bad.append((str(f), str(e)))
            continue
        for k, v in d.items():
            if len(v[0]) == 0:
                n_zero_frames += 1
        dets.update(d)

    if bad:
        print(f"[FAIL] {len(bad)} 个 TSV 解析失败，前 3 条：")
        for f, e in bad[:3]:
            print(f"        {f}: {e}")
        raise SystemExit(2)

    print(f"[TSV] 帧 {len(dets)}（其中 0 检出 {n_zero_frames}）")

    records, missing = [], []
    for r in gt_rows:
        im = r["im_name"]
        stem = im.rsplit(".", 1)[0]
        if stem not in dets:
            missing.append(im)
            continue
        pts, sc = dets[stem]
        records.append({
            "im_name": im,
            "gt_pts": np.asarray(r["gt_pts"], dtype=np.float32),
            "gt_bboxes": np.asarray(r["gt_bboxes"], dtype=np.float32),
            "pred_points": pts.astype(np.float32),
            "pred_scores": sc.astype(np.float32),
        })

    print(f"[OUT] 配对成功 {len(records)} / GT {len(gt_rows)}"
          + (f"，缺 {len(missing)}（前 3: {missing[:3]}）" if missing else ""))
    if len(records) != a.expect_total:
        print(f"[FAIL] 记录数 {len(records)} != 期望 {a.expect_total}")
        raise SystemExit(3)

    npts = np.array([len(r["pred_points"]) for r in records])
    print(f"[STAT] 每帧检出数: 均值 {npts.mean():.2f}  最大 {npts.max()}  "
          f"0 检出帧 {(npts == 0).sum()}  >=80 的帧 {(npts >= 80).sum()}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(records, f)
    print(f"[WROTE] {a.out}")


if __name__ == "__main__":
    main()
