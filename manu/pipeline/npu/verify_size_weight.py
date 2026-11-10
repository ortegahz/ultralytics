#!/usr/bin/env python3
"""尺寸加权 focal 损失的**逐位一致性**自检（方向三① 的前置闸门）。

要证明两件事：
1. **默认关闭 ⇒ 逐位不变**：``size_weight_small/large == 1.0`` 时，
   ``generate_heatmaps_and_targets`` 的返回值与**旧实现逐位相同**（heatmap/offset/ind/mask），
   且**不产生 weight_map**；``HeatmapLoss`` 的数值与旧实现逐位相同。
2. **打开后只改「正样本梯度」**：weight_map 只在目标中心那一格取 w，
   其余全 1 ⇒ focal 正项按 w 线性放大，负项与 offset 项**逐位不变**。

用法（服务器）：
    python manu/pipeline/npu/verify_size_weight.py --old-module /tmp/old_heatmap_loss.py
    # 不传 --old-module 时只做「打开后行为」自检
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

import torch

REPO = "/tmp/pycharm_project_10ae9e2e"


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_boxes() -> tuple[torch.Tensor, torch.Tensor]:
    """确定性样例：4 张图，混合 nano(≤4px/≤16px²)、micro、regular。"""
    # 归一化框 (cx, cy, w, h)，相对 640×512 原图
    raw = [
        # (img, cx, cy, w_px, h_px)
        (0, 0.10, 0.10, 3.0, 3.0),    # nano：最长边 3px、面积 9
        (0, 0.30, 0.20, 4.0, 4.0),    # nano：面积 16（边界）
        (0, 0.50, 0.30, 20.0, 20.0),  # regular
        (1, 0.20, 0.40, 2.0, 8.0),    # nano：面积 16、最长边 8（细长）
        (1, 0.60, 0.50, 8.0, 8.0),    # small
        (2, 0.15, 0.60, 40.0, 40.0),  # regular
        (3, 0.80, 0.80, 5.0, 5.0),    # nano：面积 25 但最长边 5 → 不算 nano（按 NanoBench 规则）
    ]
    boxes = []
    bids = []
    for img, cx, cy, w, h in raw:
        boxes.append([cx, cy, w / 640.0, h / 512.0])
        bids.append(img)
    return (torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(bids, dtype=torch.long))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-module", default="", help="旧版 heatmap_loss.py 路径（用于逐位对照）")
    ap.add_argument("--grid", type=int, default=320, help="监督网格边长（0031 = 320）")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    from manu.models.heatmap_loss import HeatmapLoss, generate_heatmaps_and_targets  # noqa: PLC0415

    ok = True
    boxes, bids = make_boxes()
    bs = 4
    feat = (args.grid, args.grid)

    # ---------- ① 默认关闭：与旧实现逐位对照 ----------
    new_t = generate_heatmaps_and_targets(boxes, bids, bs, feat, stride=2, min_radius=1)
    if "weight_map" in new_t:
        print("[FAIL] 默认参数竟然产生了 weight_map")
        ok = False
    else:
        print("[PASS] 默认参数不产生 weight_map（零额外开销）")

    if args.old_module:
        old = load_module(args.old_module, "old_heatmap_loss")
        old_t = old.generate_heatmaps_and_targets(boxes, bids, bs, feat, stride=2, min_radius=1)
        for k in ("heatmap", "offset", "ind", "mask"):
            same = torch.equal(old_t[k], new_t[k])
            print(f"[{'PASS' if same else 'FAIL'}] 默认参数下 `{k}` 与旧实现逐位一致")
            ok = ok and same

        # loss 数值逐位对照
        torch.manual_seed(0)
        pred = torch.rand((bs, 1, args.grid, args.grid))
        off = torch.rand((bs, 2, args.grid, args.grid))
        crit_old = old.HeatmapLoss(hm_weight=1.0, offset_weight=0.4, focal_alpha=2.0, focal_beta=2.75)
        crit_new = HeatmapLoss(hm_weight=1.0, offset_weight=0.4, focal_alpha=2.0, focal_beta=2.75)
        l_old, _ = crit_old({"heatmap": pred, "offset": off}, old_t)
        l_new, _ = crit_new({"heatmap": pred, "offset": off}, new_t)
        same = torch.equal(l_old, l_new)
        print(f"[{'PASS' if same else 'FAIL'}] 默认参数下 loss 逐位一致（{l_old.item():.10f}）")
        ok = ok and same

    # ---------- ② 打开后：weight_map 只在目标中心取 w ----------
    wt = generate_heatmaps_and_targets(
        boxes, bids, bs, feat, stride=2, min_radius=1,
        size_weight_small=2.0, size_weight_large=1.0,
        size_thr_px=16.0, size_max_side_px=4.0, img_w=640.0, img_h=512.0,
    )
    wm = wt.get("weight_map")
    if wm is None:
        print("[FAIL] 打开后没有 weight_map")
        return 1
    # 目标中心（按 generate 的取整规则）
    W = args.grid
    expect = {}
    for i, b in enumerate(boxes):
        img = int(bids[i])
        cx = min(int(b[0].item() * W), W - 1)
        cy = min(int(b[1].item() * W), W - 1)
        w_px = b[2].item() * 640.0
        h_px = b[3].item() * 512.0
        is_nano = ((max(w_px, h_px) <= 4.0 + 1e-3) or (w_px * h_px <= 16.0 + 1e-3))
        expect[(img, cy, cx)] = 2.0 if is_nano else 1.0
        got = float(wm[img, 0, cy, cx])
        want = expect[(img, cy, cx)]
        flag = "PASS" if abs(got - want) < 1e-6 else "FAIL"
        print(f"[{flag}] 目标{i} img={img} ({cx},{cy}) {w_px:.1f}×{h_px:.1f}px "
              f"nano={is_nano} ⇒ weight={got} (期望 {want})")
        ok = ok and flag == "PASS"
    nz = (wm != 1.0).sum().item()
    print(f"[INFO] weight_map 中非 1 的格子数 = {nz}（应等于 nano 目标数）")

    # ---------- ③ 负项/offset 项逐位不变，只有正项被放大 ----------
    torch.manual_seed(0)
    pred = torch.rand((bs, 1, args.grid, args.grid))
    off = torch.rand((bs, 2, args.grid, args.grid))
    crit = HeatmapLoss(hm_weight=1.0, offset_weight=0.4, focal_alpha=2.0, focal_beta=2.75)
    base_t = generate_heatmaps_and_targets(boxes, bids, bs, feat, stride=2, min_radius=1)
    l0, i0 = crit({"heatmap": pred, "offset": off}, base_t)
    l1, i1 = crit({"heatmap": pred, "offset": off}, wt)
    same_off = abs(i0["loss_offset"] - i1["loss_offset"]) <= 1e-9
    print(f"[{'PASS' if same_off else 'FAIL'}] offset 损失逐位不变（{i0['loss_offset']:.8f}）")
    ok = ok and same_off
    print(f"[INFO] hm 损失 {i0['loss_hm']:.8f} → {i1['loss_hm']:.8f}（应增大）")
    ok = ok and i1["loss_hm"] > i0["loss_hm"]

    print("\n==> " + ("ALL PASS" if ok else "HAS FAILURE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
