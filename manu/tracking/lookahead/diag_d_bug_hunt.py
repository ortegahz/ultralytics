#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**D 桶猎 bug**：目标在 Δ 内几乎没动（≤5 px），外挂却走歪 >5 px 的帧 —— 外挂侧唯一的"真 bug"。

这些帧锚点（源帧检测）是对的，目标也没动，外挂却把坐标报到了 5 px 以外 ⇒ 一定是外挂自己的问题。
本脚本判定**三种失效形态**（决定修法）：
  * ``stale_anchor`` ：上报点 ≈ 源帧锚点 ⇒ **位移没生效**（回退到 hold / 锚点原样搬运）
  * ``wrong_lock``   ：上报点既不在 GT 附近、也不在锚点附近 ⇒ **跟错目标**（锁到别的东西）
  * ``drift``        ：上报点 = 锚点 + 一个与真实位移**方向不符/幅度不符**的偏移 ⇒ **位移估计错**

用法::

    python manu/tracking/lookahead/diag_d_bug_hunt.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --original runs/lookahead/final/gateA_newcode/preds24 \
        --replay   runs/lookahead/final/final24/preds_delivery \
        --delta 8 --out-json runs/lookahead/final/final24/diag_d_bug_hunt.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

IMG_W, IMG_H = 640, 512


def load_gt(labels_root: Path, seq: str):
    out = {}
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


def load_full(preds_dir: Path, seq: str):
    path = preds_dir / f"{seq}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8")).get("per_frame", {})
    return {
        int(k): [(float(d["pos"][0]), float(d["pos"][1]), int(d.get("track_id", -1)), bool(d.get("infilled", False))) for d in (dets or [])]
        for k, dets in payload.items()
    }


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--disp-tol-px", type=float, default=5.0, help="「目标几乎没动」的判定阈值")
    ap.add_argument("--max-report", type=int, default=40)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    orig_dir, rep_dir = Path(args.original), Path(args.replay)
    delta, tol, dtol = int(args.delta), float(args.hit_tol_px), float(args.disp_tol_px)
    seqs = sorted(p.stem for p in rep_dir.glob("*.json") if p.stem != "summary")

    shapes: Counter = Counter()
    err_buckets: Counter = Counter()
    border_rows: list = []
    same_track_rows: list = []
    inf_rows: list = []
    per_seq: dict[str, Counter] = defaultdict(Counter)
    cases = []
    for seq in seqs:
        gt = load_gt(labels_root, seq)
        orig = load_full(orig_dir, seq)
        rep = load_full(rep_dir, seq)
        for t, gts in gt.items():
            src = t - delta
            if src < 0 or not gt.get(src) or not orig.get(src) or not gts:
                continue
            rep_d = rep.get(t, [])
            if not rep_d:
                continue                      # C 桶（未上报）另算
            err = min(dist(p, g) for p in rep_d for g in gts)
            if err <= tol:
                continue                      # 命中
            # 锚点（源帧最近 GT 中心处的检测）
            anchor = min(orig[src], key=lambda p: dist(p, min(gt[src], key=lambda g: dist(p, g))))
            a_pt = min(gt[src], key=lambda g: dist(anchor, g))
            t_pt = min(gts, key=lambda g: dist(a_pt, g))
            real_disp = dist(a_pt, t_pt)
            if real_disp > dtol:
                continue                      # 目标动了 → 不是「几乎没动」类
            if dist(anchor, a_pt) > tol:
                continue                      # 锚点本身错 → B 桶
            out = min(rep_d, key=lambda p: dist(p, t_pt))
            to_anchor = dist(out, anchor)
            shape = "stale_anchor" if to_anchor <= tol else "wrong_lock"
            shapes[shape] += 1
            per_seq[seq][shape] += 1
            b = "<=10" if err <= 10 else ("10-20" if err <= 20 else ("20-50" if err <= 50 else ">50"))
            err_buckets[b] += 1
            border = min(anchor[0], IMG_W - anchor[0], anchor[1], IMG_H - anchor[1])
            # 锚点是不是管线「插值」出来的假位置（infilled）？那是快核起步就错的经典原因
            anchor_inf = bool(anchor[3]) if len(anchor) > 3 else False
            out_inf = bool(out[3]) if len(out) > 3 else False
            cases.append({
                "seq": seq, "frame": t, "src": src,
                "err": round(err, 2), "to_anchor": round(to_anchor, 2),
                "real_disp": round(real_disp, 2),
                "anchor": [round(anchor[0], 1), round(anchor[1], 1)],
                "gt_t": [round(t_pt[0], 1), round(t_pt[1], 1)],
                "out": [round(out[0], 1), round(out[1], 1)],
                "track_id": out[2], "anchor_track": anchor[2],
                "anchor_infilled": anchor_inf, "out_infilled": out_inf, "shape": shape,
                "anchor_border_px": round(border, 1),
                "same_track": bool(out[2] == anchor[2] and out[2] >= 0),
            })
            border_rows.append((border, shape))
            inf_rows.append((anchor_inf, out_inf, shape))
            same_track_rows.append(shape == "stale_anchor" and out[2] == anchor[2] and out[2] >= 0)

    n = max(1, sum(shapes.values()))
    print(f"=== D 桶猎 bug（目标在 Δ={delta} 内位移 ≤{dtol}px，锚点对，但外挂报 >{tol}px）===")
    print(f"命中 {n} 帧")
    for k, v in shapes.most_common():
        print(f"  {k:16s} {v:6d}  {v / n * 100:6.2f}%")
    print(f"--- 误差幅度分桶 ---")
    for k in ("<=10", "10-20", "20-50", ">50"):
        v = err_buckets.get(k, 0)
        if v:
            print(f"  {k:>6s}  {v:6d}  {v / n * 100:6.2f}%")
    print("--- 逐序（stale_anchor = 位移没生效；wrong_lock = 跟错目标）---")
    for seq, c in sorted(per_seq.items(), key=lambda kv: -sum(kv[1].values())):
        if sum(c.values()):
            print(f"  {seq:34s} {sum(c.values()):5d}  stale_anchor={c.get('stale_anchor', 0):5d}  wrong_lock={c.get('wrong_lock', 0):5d}")
    # --- 机制线索 1：锚点是否贴边（贴边 ⇒ ROI 被裁 ⇒ 质心/峰偏）---
    near = sum(1 for b, _ in border_rows if b <= 16)
    print(f"--- 机制线索：锚点贴边（距边界 ≤16 px）{near} / {len(border_rows)} 帧（{near / max(1, len(border_rows)) * 100:.1f}%）"
          f"  ⇒ 贴边占比 {near / max(1, len(border_rows)) * 100:.1f}%，全体 GT 帧贴边基准见 memory §30.9.x")
    # --- 机制线索 2：是否换了 track_id（换了 ⇒ 跟错目标）---
    st = [x for x in same_track_rows]
    print(f"--- 机制线索：stale_anchor 帧里 track_id 与锚点相同 = {sum(st)} / {max(1, len(st))}"
          f"（不同 ⇒ 外挂换了一条轨迹去上报）")
    # --- 机制线索 3：时间聚簇（连续帧 ⇒ 一次跟丢后拉不回来）---
    by_seq: dict = defaultdict(list)
    for c in cases:
        by_seq[c["seq"]].append(c["frame"])
    runs = []
    for seq, frames in by_seq.items():
        frames.sort()
        cur = 1
        for i in range(1, len(frames)):
            if frames[i] == frames[i - 1] + 1:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
    if runs:
        runs.sort(reverse=True)
        print(f"--- 机制线索：时间聚簇 {len(runs)} 段，最长 {runs[0]} 帧；"
              f"≥3 帧的连段 {sum(1 for r in runs if r >= 3)} 段（共 {sum(r for r in runs if r >= 3)} 帧）")
    # --- 机制线索 4：锚点是不是管线插值出来的（infilled）假位置 ---
    a_inf = sum(1 for a, _, _ in inf_rows if a)
    o_inf = sum(1 for _, o, _ in inf_rows if o)
    print(f"--- 机制线索：锚点 infilled（管线插值假位置）{a_inf} / {len(inf_rows)} 帧"
          f"（{a_inf / max(1, len(inf_rows)) * 100:.1f}%）；上报点 infilled {o_inf} 帧")
    for shp in ("stale_anchor", "wrong_lock"):
        sub = [r for r in inf_rows if r[2] == shp]
        if sub:
            print(f"    {shp:14s} 锚点 infilled {sum(1 for a, _, _ in sub)} / {len(sub)}"
                  f"  ({sum(1 for a, _, _ in sub) / len(sub) * 100:.1f}%)")
    # --- 机制线索 5：track_id 是否与锚点一致（stale_anchor 桶单独统计）---
    st_sub = [c for c in cases if c["shape"] == "stale_anchor"]
    if st_sub:
        same = sum(1 for c in st_sub if c["same_track"])
        print(f"--- 机制线索：stale_anchor 帧 track_id 与锚点相同 {same} / {len(st_sub)}"
              f"（不同 ⇒ 外挂换轨上报）")
    cases.sort(key=lambda c: -c["err"])
    print(f"--- 最严重的 {min(args.max_report, len(cases))} 例 ---")
    for c in cases[: args.max_report]:
        print(f"  {c['seq']} f{c['frame']:<6d} err={c['err']:7.2f} to_anchor={c['to_anchor']:7.2f} "
              f"real_disp={c['real_disp']:5.2f} track={c['track_id']} anchor={c['anchor']} gt={c['gt_t']} out={c['out']}")

    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "delta": delta, "hit_tol_px": tol, "disp_tol_px": dtol,
            "shapes": dict(shapes), "err_buckets": dict(err_buckets),
            "per_sequence": {k: dict(v) for k, v in per_seq.items()},
            "worst_cases": cases[:200],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[SAVED] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
