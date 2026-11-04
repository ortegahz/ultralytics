#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**误差三分解**：把回放臂的输出误差拆成「锚点继承 e(f)」+「位移估计残差 δ」。

  O(t) = A(f) + D_est(f→t)        （f = t − Δ）
  GT(t) = GT(f) + D_true(f→t)
  ⇒ 误差(t) = O(t) − GT(t) = [A(f) − GT(f)] + [D_est − D_true] = e(f) + δ

其中：
  * **A(f)** = 回放用的锚点 = 管线在源帧的输出位置。
    取自 ``stale`` 臂在 t 帧的输出（``stale`` 臂的定义就是 ``report[t] = D(t−Δ)``，
    位置原样搬运 ⇒ 逐位等于锚点）。
  * **e(f)** = |A(f) − GT(f)|：锚点自身误差（被原样继承）。
  * **δ**   = (O(t) − A(f)) − (GT(t) − GT(f))：位移估计残差。
  * **e(t)** = |O_off(t) − GT(t)|：零延迟管线在**当前帧**的误差（对照基准）。

⇒ 若 |δ| ≫ |e(f)| ⇒ 该修**位移估计**；若 |e(f)| ≫ |δ| ⇒ 该修**锚点**。

用法::

    python manu/tracking/lookahead/diag_error_decomp2.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --stale runs/lookahead/final/preds_final_stale \
        --replay runs/lookahead/final/final8/preds_delivery \
        --original runs/lookahead/final/preds_off --delta 8 --out-json ...json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

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
    if not points:
        return None
    return min(math.hypot(px - gx, py - gy) for px, py in points for gx, gy in targets)


def pct(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--stale", required=True, help="stale 臂 preds 目录（= 锚点 A(f)）")
    ap.add_argument("--replay", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--hit-tol-px", type=float, default=5.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    stale_dir, rep_dir, off_dir = Path(args.stale), Path(args.replay), Path(args.original)
    seqs = sorted(p.stem for p in rep_dir.glob("*.json") if p.stem != "summary")
    d = int(args.delta)

    e_hist, delta_hist, err_hist, eoff_hist = [], [], [], []
    miss_e, miss_delta = [], []
    per_seq = {}
    for seq in seqs:
        gt = load_gt(labels_root, seq)
        st = load_per_frame(stale_dir, seq)
        rp = load_per_frame(rep_dir, seq)
        off = load_per_frame(off_dir, seq)
        row = {"n": 0, "hit": 0, "miss": 0}
        for t, gts in gt.items():
            f = t - d
            if f < 0 or f not in gt or not gt.get(f):
                continue
            a = st.get(t)
            o = rp.get(t)
            if not a or not o:
                continue
            row["n"] += 1
            e = nearest(a, gt[f])
            delta_vec = np.asarray(o[0], dtype=np.float64) - np.asarray(a[0], dtype=np.float64)
            true_vec = np.asarray(gt[t][0], dtype=np.float64) - np.asarray(gt[f][0], dtype=np.float64)
            # 用**最近 GT 中心**重算真值位移，避免多目标时配错
            j = int(np.argmin([math.hypot(gx - a[0][0], gy - a[0][1]) for gx, gy in gt[f]]))
            true_vec = np.asarray(gt[t][j], dtype=np.float64) - np.asarray(gt[f][j], dtype=np.float64)
            delta = float(np.linalg.norm(delta_vec - true_vec))
            err = nearest(o, gts)
            eoff = nearest(off.get(t, []), gts)
            e_hist.append(e)
            delta_hist.append(delta)
            err_hist.append(err)
            if eoff is not None:
                eoff_hist.append(eoff)
            if err is not None and err <= args.hit_tol_px:
                row["hit"] += 1
            else:
                row["miss"] += 1
                miss_e.append(e)
                miss_delta.append(delta)
        per_seq[seq] = row

    print(f"=== 误差三分解 {args.label}（Δ={d}）===")
    print(f"  锚点误差 e(f)      : p50={pct(e_hist,50):6.3f}  p90={pct(e_hist,90):6.3f}  p99={pct(e_hist,99):6.3f}")
    print(f"  位移残差 δ         : p50={pct(delta_hist,50):6.3f}  p90={pct(delta_hist,90):6.3f}  p99={pct(delta_hist,99):6.3f}")
    print(f"  输出误差 |e+δ|     : p50={pct(err_hist,50):6.3f}  p90={pct(err_hist,90):6.3f}  p99={pct(err_hist,99):6.3f}")
    print(f"  零延迟管线 e(t)     : p50={pct(eoff_hist,50):6.3f}  p90={pct(eoff_hist,90):6.3f}  p99={pct(eoff_hist,99):6.3f}")
    print(f"  失手帧 {len(miss_e)}：e(f) p50={pct(miss_e,50):6.3f}   δ p50={pct(miss_delta,50):6.3f}")
    print("  --- 逐序 ---")
    for seq, row in per_seq.items():
        n = max(1, row["n"])
        print(f"  {seq:34s} n={row['n']:6d} hit {row['hit'] / n * 100:6.2f}%")
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "label": args.label,
                    "anchor_err": {"p50": pct(e_hist, 50), "p90": pct(e_hist, 90), "p99": pct(e_hist, 99)},
                    "disp_residual": {"p50": pct(delta_hist, 50), "p90": pct(delta_hist, 90), "p99": pct(delta_hist, 99)},
                    "output_err": {"p50": pct(err_hist, 50), "p90": pct(err_hist, 90), "p99": pct(err_hist, 99)},
                    "zero_delay_err": {"p50": pct(eoff_hist, 50), "p90": pct(eoff_hist, 90), "p99": pct(eoff_hist, 99)},
                    "miss": {"n": len(miss_e), "anchor_p50": pct(miss_e, 50), "delta_p50": pct(miss_delta, 50)},
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
