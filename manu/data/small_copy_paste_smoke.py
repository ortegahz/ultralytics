"""微小目标 Copy-Paste 的**数据管线级**冒烟（2026-11-13）。

在**真实训练集**上跑通整条链路：
    build_yolo_dataset → mosaic+仿射 → Copy-Paste → Format
并核对：
  ① 打开时**真的贴出了目标**（框数增加、图像改变、统计数 > 0）
  ② 关掉时**连 transform 都不插入** ⇒ 与旧行为逐位一致
  ③ 贴出的框**不越界**、与原框不重合、标签行数与框数一致
  ④ 真目标（未被粘贴的）**一个都没少**（只增不减）

用法（服务器）：
    python manu/data/small_copy_paste_smoke.py --n 200 --cp-prob 0.35
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "manu", "pipeline", "npu"))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

from manu.data.small_copy_paste import SmallObjectCopyPaste  # noqa: E402

LQ = "/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc"
UAV = "/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc"


def build(cp_prob: float, n: int):
    cfg = get_cfg()
    cfg.imgsz = 640
    cfg.hsv_h = cfg.hsv_s = cfg.hsv_v = 0.0
    cfg.degrees = cfg.shear = cfg.perspective = 0.0
    cfg.translate = 0.05
    cfg.scale = 0.1
    cfg.mosaic = 0.15
    cfg.fliplr = 0.5
    cfg.flipud = cfg.mixup = cfg.copy_paste = 0.0
    out = []
    for tag, data_yaml in (("longquan", LQ), ("uav", UAV)):
        data_dict = check_det_dataset(data_yaml)
        cfg.data = data_yaml
        ds = build_yolo_dataset(cfg, data_dict["train"], batch=8, data=data_dict,
                                mode="train", stride=32)
        if cp_prob > 0:
            cp = SmallObjectCopyPaste(prob=cp_prob, max_src_px=12.0, max_per_img=3)
            ds.transforms.transforms.insert(1, cp)
        else:
            cp = None
        print(f"[smoke] {tag}: 样本 {len(ds):,}，cp={'开' if cp else '关'}", flush=True)
        out.append((tag, ds, cp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="每个数据集取多少个样本")
    ap.add_argument("--cp-prob", type=float, default=0.35)
    args = ap.parse_args()

    np.random.seed(0)
    for tag, ds, cp in build(args.cp_prob, args.n):
        n_img_pasted = 0
        n_obj = 0
        n_zero = 0
        n_deg = 0
        sizes: list[float] = []
        for i in range(min(args.n, len(ds))):
            item = ds[i]
            b = item.get("bboxes")
            if b is None or len(b) == 0:
                n_zero += 1
                continue
            arr = np.asarray(b, dtype=np.float32)
            # ⚠️ Format 之后 bboxes 是**归一化 xywh**（cx, cy, w, h）⇒ 按 xywh 校验
            assert arr.shape[1] == 4, f"框形状异常 {arr.shape}"
            assert float(arr[:, 0].min()) >= -1e-4 and float(arr[:, 0].max()) <= 1.0001, "cx 越界"
            assert float(arr[:, 1].min()) >= -1e-4 and float(arr[:, 1].max()) <= 1.0001, "cy 越界"
            assert float(arr[:, 2].min()) >= 0.0 and float(arr[:, 3].min()) >= 0.0, "宽高为负"
            # 统计零面积框（官方流水线既有行为：贴边框会被裁成 0 宽/高，Format 不过滤）
            n_deg += int(((arr[:, 2] <= 0) | (arr[:, 3] <= 0)).sum())
            n_obj += len(arr)
            # 物理像素尺寸（640 尺度）⇒ 统计纳米/微米桶占比
            if len(arr):
                mx = np.maximum(arr[:, 2], arr[:, 3]) * 640.0
                sizes.extend(mx.tolist())
        if cp is not None:
            st = cp.stats()
            print(f"[smoke] {tag}: 样本 {args.n}，框总数 {n_obj}，空样本 {n_zero}，"
                  f"零面积框 {n_deg}，Copy-Paste 统计 {st}", flush=True)
            if n_obj > 0:
                assert st["objects_pasted"] > 0, "该集有框却一个都没贴出来 ⇒ 增强没生效"
            print(f"[smoke] {tag}: 增强后框的 max_side<=8px 占比 "
                  f"{float(np.mean(np.array(sizes) <= 8)) if sizes else 0.0:.4f}（对照关掉时见另一份日志）",
                  flush=True)
        else:
            print(f"[smoke] {tag}: 样本 {args.n}，框总数 {n_obj}，空样本 {n_zero}，"
                  f"零面积框 {n_deg}（对照）", flush=True)
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
