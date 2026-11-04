#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**缺口归因**：把「零延迟天花板 vs 交付档」的命中差，逐帧判定是**检测器**的问题还是**外挂**的问题。

背景（§34 的结论：外挂已在物理极限）：
  剩余命中缺口 **不是**外挂的锅，但要给出**可执行的检测器改进规格**，否则「该修检测器」只是口号。

口径（与 `diag_miss_funnel.py` 完全一致）：
  * `original` 臂 = **零延迟管线**在**当前帧 t** 的输出 ⇒ 它命中 = 「检测器在这一帧本来就能给对」。
  * `replay`   臂 = 交付档在帧 t 的上报。
  * 缺口帧 = `original` 命中 且 `replay` 未命中的帧（⇒ 理论可救回，且检测器当帧是好的）。

缺口帧再按**外挂的因果链**逐级分类：
  ┌ src<0 / 源帧无 GT ................ 边界帧（目标刚出现）
  ├ 源帧零延迟管线**无检测** .......... A · 检测器**源帧漏检**（时序不稳定）
  ├ 源帧有检测但锚点自身 > tol ....... B · 检测器**源帧定位错**（外挂只是继承）
  └ 源帧检测是对的
      ├ 外挂**未上报**（弃权/回退失败） C · **外挂侧**弃权门/回退链
      └ 外挂上报了但位置错 ............ D · **外挂侧**走歪（位移估计）
                                     （或目标在 Δ 内真动了 ⇒ 物理上限）

用法::

    python manu/tracking/lookahead/diag_gap_attribution.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --original runs/lookahead/final/gateA_newcode/preds24 \
        --replay   runs/lookahead/final/final24/preds_delivery \
        --delta 8 --out-json runs/lookahead/final/final24/diag_gap_attribution.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

IMG_W, IMG_H = 640, 512


def load_gt(labels_root: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    for f in sorted(labels_root.glob(f"{seq}__*.txt")):
        idx = int(f.stem.split("__")[-1])
        pts = []
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                pts.append((float(parts[1]) * IMG_W, float(parts[2]) * IMG_H))
        if pts:
            out[idx] = pts
    return out


def load_per_frame(preds_dir: Path, seq: str) -> dict[int, list[tuple[float, float]]]:
    path = preds_dir / f"{seq}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8")).get("per_frame", {})
    return {
        int(k): [(float(d["pos"][0]), float(d["pos"][1])) for d in (dets or [])]
        for k, dets in payload.items()
    }


def nearest(points, targets):
    if not points or not targets:
        return None
    return min(math.hypot(px - gx, py - gy) for px, py in points for gx, gy in targets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--original", required=True, help="零延迟管线 preds 目录（探针）")
    ap.add_argument("--replay", required=True, help="交付档 preds 目录")
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--seqs", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    orig_dir, rep_dir = Path(args.original), Path(args.replay)
    delta, tol = int(args.delta), float(args.hit_tol_px)
    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()] or sorted(
        p.stem for p in rep_dir.glob("*.json") if p.stem != "summary"
    )

    grand: Counter = Counter()
    per_seq: dict[str, dict] = {}
    for seq in seqs:
        gt = load_gt(labels_root, seq)
        orig = load_per_frame(orig_dir, seq)
        rep = load_per_frame(rep_dir, seq)
        row: Counter = Counter()
        for t, gts in gt.items():
            row["gt"] += 1
            orig_hit = (nearest(orig.get(t, []), gts) or 1e9) <= tol
            rep_hit = (nearest(rep.get(t, []), gts) or 1e9) <= tol
            if orig_hit:
                row["orig_hit"] += 1
            if rep_hit:
                row["rep_hit"] += 1
            if orig_hit and not rep_hit:
                # ↓↓↓ 缺口帧：检测器当帧是好的，外挂没救回来
                row["gap"] += 1
                src = t - delta
                if src < 0:
                    row["gap_edge_src"] += 1
                    continue
                if not gt.get(src):
                    row["gap_edge_gt"] += 1
                    continue
                anchor_err = nearest(orig.get(src, []), gt[src])
                if anchor_err is None:
                    row["gap_A_no_src_det"] += 1          # 检测器源帧漏检
                elif anchor_err > tol:
                    row["gap_B_anchor_wrong"] += 1       # 检测器源帧定位错（外挂只是继承）
                else:
                    err = nearest(rep.get(t, []), gts)
                    if err is None:
                        row["gap_C_no_output"] += 1       # 外挂未上报（弃权门 / 回退失败）
                    else:
                        row["gap_D_wrong_pos"] += 1       # 外挂上报了但走歪
                        # ↓ 再按目标在 Δ 内**真实位移**细分：走歪是「外挂估错」还是「目标剧烈机动」
                        src_gts = gt[src]
                        # 追踪同一个目标：用 src 处最近的 GT 中心作为该目标的锚
                        s_pt = min(src_gts, key=lambda g: math.hypot(g[0] - orig[src][0][0], g[1] - orig[src][0][1])) \
                            if orig.get(src) else None
                        t_pt = min(gts, key=lambda g: math.hypot(g[0] - s_pt[0], g[1] - s_pt[1])) if s_pt else None
                        if s_pt is not None and t_pt is not None:
                            disp = math.hypot(t_pt[0] - s_pt[0], t_pt[1] - s_pt[1])
                            row["D_disp_sum"] += disp
                            row["D_disp_max"] = max(row["D_disp_max"], disp)
                            if disp <= 5.0:
                                row["D_disp_le5"] += 1       # 目标几乎没动却走歪 ⇒ **真 bug，可修**
                            elif disp <= 20.0:
                                row["D_disp_5_20"] += 1
                            else:
                                row["D_disp_gt20"] += 1      # 剧烈机动 ⇒ 线性外推的物理上限
            elif rep_hit and not orig_hit:
                row["rep_only"] += 1                      # 外挂比零延迟检测器还准（补了当帧漏检）
        per_seq[seq] = dict(row)
        grand.update(row)

    total = max(1, grand["gt"])
    print(f"=== 缺口归因 {args.label}（Δ={delta}, tol={tol}px）===")
    print(f"GT 帧 {grand['gt']}  |  零延迟命中 {grand['orig_hit']} ({grand['orig_hit'] / total * 100:.2f}%)"
          f"  |  交付档命中 {grand['rep_hit']} ({grand['rep_hit'] / total * 100:.2f}%)")
    print(f"★ 缺口 = {grand['gap']} 帧（{grand['gap'] / total * 100:.2f} pp of GT）"
          f"  |  外挂反超零延迟的帧 = {grand['rep_only']}（{grand['rep_only'] / total * 100:.2f} pp）")
    print(f"{'归因':44s} {'帧数':>8s} {'占缺口':>8s} {'占GT':>7s}")
    gap = max(1, grand["gap"])
    for key, name in (
        ("gap_A_no_src_det", "A 检测器源帧漏检（时序不稳定）"),
        ("gap_B_anchor_wrong", "B 检测器源帧定位错（外挂只是继承）"),
        ("gap_C_no_output", "C 外挂未上报（弃权门/回退失败）"),
        ("gap_D_wrong_pos", "D 外挂上报但走歪（位移估计）"),
        ("gap_edge_src", "边界：源帧不存在"),
        ("gap_edge_gt", "边界：源帧无 GT（目标刚出现）"),
    ):
        v = grand.get(key, 0)
        print(f"{name:44s} {v:8d} {v / gap * 100:7.2f}% {v / total * 100:6.2f}%")
    detector = grand.get("gap_A_no_src_det", 0) + grand.get("gap_B_anchor_wrong", 0)
    plugin = grand.get("gap_C_no_output", 0) + grand.get("gap_D_wrong_pos", 0)
    print(f"--- 汇总：检测器侧 {detector} 帧（占缺口 {detector / gap * 100:.1f}%，占 GT {detector / total * 100:.2f} pp）"
          f" | 外挂侧 {plugin} 帧（占缺口 {plugin / gap * 100:.1f}%，占 GT {plugin / total * 100:.2f} pp）")
    d_n = max(1, grand.get("gap_D_wrong_pos", 0))
    print(f"--- D 桶按「目标在 Δ=8 内真实位移」细分（判定：外挂估错 vs 物理上限）---")
    for key, name in (("D_disp_le5", "位移 ≤ 5 px（目标几乎没动却走歪 ⇒ **真 bug，可修**）"),
                      ("D_disp_5_20", "位移 5~20 px（线性外推够得着但没跟上）"),
                      ("D_disp_gt20", "位移 > 20 px（剧烈机动 ⇒ 线性外推的物理上限）")):
        v = grand.get(key, 0)
        print(f"  {name:46s} {v:7d}  占 D {v / d_n * 100:6.2f}%  占 GT {v / total * 100:5.2f} pp")
    if grand.get("gap_D_wrong_pos"):
        print(f"  D 桶平均真实位移 = {grand.get('D_disp_sum', 0) / d_n:.2f} px  （最大 {grand.get('D_disp_max', 0):.1f} px）")
    print("--- 逐序（按缺口帧数排序）---")
    for seq, row in sorted(per_seq.items(), key=lambda kv: -kv[1].get("gap", 0)):
        g = max(1, row.get("gt", 0))
        if not row.get("gap"):
            continue
        print(f"  {seq:34s} GT {g:6d}  缺口 {row.get('gap', 0):6d}（占 GT {row.get('gap', 0) / g * 100:5.2f}%）"
              f"  A={row.get('gap_A_no_src_det', 0):5d} B={row.get('gap_B_anchor_wrong', 0):5d}"
              f" C={row.get('gap_C_no_output', 0):4d} D={row.get('gap_D_wrong_pos', 0):5d}")

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "label": args.label,
                    "delta": delta,
                    "hit_tol_px": tol,
                    "overall": dict(grand),
                    "per_sequence": per_seq,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"[SAVED] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
