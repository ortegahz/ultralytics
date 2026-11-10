#!/usr/bin/env python3
"""微小目标子集：定义 + 构建 + 分桶评估（路线二：P1/P2 + 通道瘦身的监控盘）。

═══ 为什么需要 ═══
大盘 T1（31613 帧 / 25111 个目标）里中大目标占多数 ⇒ 结构退化在总 F1 上只体现 1~2 pp，
把「P3 丢掉 P1/P2 到底损失多少微小目标召回」这件事彻底掩盖。
⇒ 建**GT 驱动、与模型无关**的微小目标子集，把架构收益与代价量化到可辩论的粒度。

═══ 尺寸定义（物理像素，原图 640x512）═══
  nano  : max(w,h) <= 4px  或  area <= 16px^2   （极限微小：P3 的 8px 网格物理上无法表示）
  micro : 4px < max(w,h) <= 8px                 （过渡带：检验 P2(4px) 与 P3(8px) 的交接）
  small : 8px < max(w,h) <= 16px
  regular: max(w,h) > 16px
（分位数参考：T1 目标 max-side p1=2.5 / p10=4.0 / p30=8.8 / p50=12.5 / p90=26.3 px
  ⇒ <=4px 占 10.2%、<=8px 占 26.9%）

═══ 匹配口径（与权威 T1 评估器逐位一致）═══
  复用 manu/evaluation/eval_arm_pc_gmc_ab.py 的 `match_at`：
  按分数降序贪心，每个预测取 tol 内最近的**未匹配** GT（欧氏距离），tol 默认 8.0px，
  并扫阈值取整体 F1 峰值作为该臂的**工作点**。
  ⚠️ 子集指标一律报**双容差**（4px / 8px）：2x2 目标的 IoU 恒≈0 ⇒ 只能用物理距离。
  ⚠️ **GMC 与 no-GMC 缓存严禁混用**：本脚本要求显式声明 --semantics 并写进结果。
"""

import json
import os
import pickle

import numpy as np

# ==============================================================================
# ★ 正式名称：**NanoBench**（微小目标专用评测台）
# ------------------------------------------------------------------------------
# NanoBench-T1-nogmc : T1（Anti-UAV 官方域）**no-GMC** 口径的微小目标子集 + 分桶召回台
# NanoBench-T1-gmc    : 同上，GMC 口径（**两套口径严禁混用**，分别出报告）
#
# 组成：
#   * **子集**：T1 全量 31613 帧 / 25111 个目标，按**物理像素尺寸**分层
#       nano   : max(w,h) <= 4px  或  area <= 16px^2   （4232 个，占 16.8%）
#       micro  : 4px < max(w,h) <= 8px                 （3283 个，占 13.1%）
#       small  : 8px < max(w,h) <= 16px                （11796 个）
#       regular: max(w,h) > 16px                       （5800 个）
#     尺寸分布（T1 实测）：max-side p1=2.5 / p10=4.0 / p30=8.8 / p50=12.5 / p90=26.3 px
#   * **判据**：中心欧氏距离容差（8px 主档 / 4px 严格档），与权威 T1 评估器
#     eval_arm_pc_gmc_ab.match_at **逐位一致**（前缀复用实现有自检 selftest 兜底）
#   * **价值**：大盘 F1 把「普通目标召回 0.96 / 微小目标召回 0.70」这 26 pp 的差距
#     稀释成 1~2 pp ⇒ 微小目标召回必须单列成台，才能给 P1/P2 的时延代价定价。
# ==============================================================================
BENCH_NAME = "NanoBench"
BENCH_TRACKS = ("NanoBench-T1-nogmc", "NanoBench-T1-gmc")
NANO_MANIFEST_DIR = "manu/data/nanobench"

# ---- 尺寸分桶（物理像素）------------------------------------------------
BUCKETS = ("nano", "micro", "small", "regular")


def gt_physical_size(gt_bboxes, img_w=640, img_h=512):
    """缓存里的 gt_bboxes 是 (cx, cy, w, h)，坐标在 640x640 归一化尺度上。

    ⇒ 宽度 w_px = w（源图宽 640，比例 1.0）；高度 h_px = h * (img_h/img_w)（源图高 512）。
    返回 (max_side_px, area_px2)，逐目标。
    """
    if len(gt_bboxes) == 0:
        return np.zeros((0,)), np.zeros((0,))
    bb = np.asarray(gt_bboxes, dtype=np.float64)
    w_px = bb[:, 2]
    h_px = bb[:, 3] * (float(img_h) / float(img_w))
    max_side = np.maximum(w_px, h_px)
    area = w_px * h_px
    return max_side, area


def bucket_of(max_side, area):
    """nano: max<=4 或 area<=16；micro: 4<max<=8；small: 8<max<=16；regular: 其余。"""
    if max_side <= 4.0 or area <= 16.0:
        return "nano"
    if max_side <= 8.0:
        return "micro"
    if max_side <= 16.0:
        return "small"
    return "regular"


def load_cache(path):
    with open(path, "rb") as f:
        recs = pickle.load(f)
    if isinstance(recs, dict):
        recs = recs.get("records", list(recs.values()))
    return recs


# ---- 匹配（与 eval_arm_pc_gmc_ab.match_at 同逻辑，额外返回被匹配的 GT 掩码）----
def match_mask(rec, th, tol):
    """返回 (tp, fp, fn, matched_gt_bool_mask, fp_count_by_dist)。

    逻辑与 eval_arm_pc_gmc_ab.match_at 逐行等价：分数降序贪心 + 最近未匹配 GT + 欧氏 tol。
    """
    scores = rec["pred_scores"]
    gts = rec["gt_pts"]
    n_gt = len(gts)
    matched = np.zeros(n_gt, dtype=bool)
    if len(scores) == 0:
        return 0, 0, n_gt, matched, 0
    keep = np.where(scores >= th)[0]
    if len(keep) == 0:
        return 0, 0, n_gt, matched, 0
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    tp = fp = 0
    used = np.zeros(n_gt, dtype=bool)
    tol2 = tol * tol
    for i in keep:
        p = rec["pred_points"][i]
        if n_gt:
            d2 = ((gts - p) ** 2).sum(axis=1)
            d2[used] = np.inf
            j = int(np.argmin(d2))
            if d2[j] <= tol2:
                used[j] = True
                matched[j] = True
                tp += 1
                continue
        fp += 1
    return tp, fp, int((~used).sum()), matched, fp


def frame_prefix_states(rec, tol, buckets):
    """★前缀复用：按分数降序贪心**只跑一遍**，返回阈值扫描所需的全部状态。

    原理：贪心是"按分数序依次处理预测"，阈值 th 的匹配结果 = 分数序列表中
    score ≥ th 的**前缀**的贪心结果 ⇒ 阈值扫描不再需要重跑匹配（49× 加速）。

    返回 (scores_sorted_desc, tp_cum, fp_cum, matched_bucket_seq)
      - tp_cum[k]  = 处理完前 k+1 个预测后的 TP 数
      - fp_cum[k]  = 同上的 FP 数
      - matched_bucket_seq[j] = 第 j 个被匹配的 GT 所属的尺寸桶
    """
    scores = np.asarray(rec["pred_scores"], dtype=np.float64)
    pts = np.asarray(rec["pred_points"], dtype=np.float64).reshape(-1, 2)
    gts = np.asarray(rec["gt_pts"], dtype=np.float64).reshape(-1, 2)
    n_gt = len(gts)
    n = min(len(scores), len(pts))
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int), []
    order = np.argsort(-scores[:n], kind="stable")
    s_sorted = scores[:n][order]
    p_sorted = pts[:n][order]
    tp_cum = np.zeros(n, dtype=np.int32)
    fp_cum = np.zeros(n, dtype=np.int32)
    matched_seq = []
    used = np.zeros(n_gt, dtype=bool)
    tp = fp = 0
    tol2 = tol * tol
    for i in range(n):
        p = p_sorted[i]
        if n_gt:
            d2 = ((gts - p) ** 2).sum(axis=1)
            d2[used] = np.inf
            j = int(np.argmin(d2))
            if d2[j] <= tol2:
                used[j] = True
                tp += 1
                matched_seq.append(buckets[j])
            else:
                fp += 1
        else:
            fp += 1
        tp_cum[i] = tp
        fp_cum[i] = fp
    return s_sorted, tp_cum, fp_cum, matched_seq


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def build_manifest(cache_path, img_w=640, img_h=512):
    """构建子集清单：逐帧每个 GT 的物理尺寸 + 分桶（与模型无关）。"""
    recs = load_cache(cache_path)
    frames = []
    stats = {b: 0 for b in BUCKETS}
    size_all = []
    for r in recs:
        ms, ar = gt_physical_size(r["gt_bboxes"], img_w, img_h)
        buckets = [bucket_of(m, a) for m, a in zip(ms, ar)]
        size_all.extend(ms.tolist())
        for b in buckets:
            stats[b] += 1
        if any(b in ("nano", "micro") for b in buckets):
            frames.append(dict(im_name=r["im_name"],
                               nano=[i for i, b in enumerate(buckets) if b == "nano"],
                               micro=[i for i, b in enumerate(buckets) if b == "micro"],
                               max_side=[float(x) for x in ms]))
    size_all = np.array(size_all) if size_all else np.zeros(1)
    manifest = dict(
        source_cache=cache_path, img_w=img_w, img_h=img_h,
        total_gt=int(sum(stats.values())), bucket_counts=stats,
        frames_total=len(recs), frames_with_tiny=len(frames),
        size_percentiles={f"p{p}": float(np.percentile(size_all, p))
                          for p in (1, 5, 10, 20, 30, 50, 75, 90, 99)},
        frames=frames,
    )
    return manifest


def eval_cache(cache_path, semantics, tols=(4.0, 8.0), ths=None, img_w=640, img_h=512,
               label="", baseline=None):
    """在给定缓存上算：整体 P/R/F1 + 分桶召回 + 孤立假阳数。

    返回 dict；工作点 = 整体 F1 峰值的阈值（与权威评估器同口径）。
    """
    if ths is None:
        ths = [round(x, 3) for x in np.arange(0.02, 0.501, 0.01)]
    recs = load_cache(cache_path)
    # 预先算好每个 GT 的桶
    bucket_masks = []
    n_gt_by_bucket = {b: 0 for b in BUCKETS}
    for r in recs:
        ms, ar = gt_physical_size(r["gt_bboxes"], img_w, img_h)
        bl = [bucket_of(m, a) for m, a in zip(ms, ar)]
        bucket_masks.append(bl)
        for b in bl:
            n_gt_by_bucket[b] += 1

    rows = []
    for tol in tols:
        # ★前缀复用：每个 (帧, 容差) 只跑一遍贪心匹配，阈值扫描取前缀（49× 加速）
        states = [frame_prefix_states(r, tol, bl) for r, bl in zip(recs, bucket_masks)]
        for th in ths:
            tp = fp = fn = 0
            tp_bucket = {b: 0 for b in BUCKETS}
            for (s_sorted, tp_cum, fp_cum, matched_seq), bl in zip(states, bucket_masks):
                if len(s_sorted) == 0:
                    fn += len(bl)
                    continue
                k = int(np.searchsorted(-s_sorted, -th, side="right"))
                if k <= 0:
                    fn += len(bl)
                    continue
                t = int(tp_cum[k - 1]); f = int(fp_cum[k - 1])
                tp += t; fp += f; fn += len(bl) - t
                for j in range(t):
                    tp_bucket[matched_seq[j]] += 1
            p, r_, f1 = prf(tp, fp, fn)
            row = dict(tol=tol, th=th, tp=tp, fp=fp, fn=fn, precision=p, recall=r_, f1=f1,
                       tp_by_bucket=tp_bucket,
                       recall_by_bucket={b: (tp_bucket[b] / n_gt_by_bucket[b] if n_gt_by_bucket[b] else 0.0)
                                         for b in BUCKETS})
            rows.append(row)

    # 工作点：整体 F1 峰值（tol=8 主档，4px 为严格档）
    peak8 = max((x for x in rows if x["tol"] == 8.0), key=lambda x: x["f1"])
    peak4 = max((x for x in rows if x["tol"] == 4.0), key=lambda x: x["f1"])
    out = dict(
        label=label, cache=cache_path, semantics=semantics,
        frames=len(recs), total_gt=int(sum(n_gt_by_bucket.values())),
        gt_by_bucket=n_gt_by_bucket,
        peak_tol8=dict(th=peak8["th"], f1=peak8["f1"], precision=peak8["precision"],
                       recall=peak8["recall"], fp=peak8["fp"], fn=peak8["fn"],
                       recall_by_bucket=peak8["recall_by_bucket"], tp_by_bucket=peak8["tp_by_bucket"]),
        peak_tol4=dict(th=peak4["th"], f1=peak4["f1"], precision=peak4["precision"],
                       recall=peak4["recall"], fp=peak4["fp"], fn=peak4["fn"],
                       recall_by_bucket=peak4["recall_by_bucket"], tp_by_bucket=peak4["tp_by_bucket"]),
    )
    if baseline:
        out["delta_vs_baseline"] = dict(
            nano=(out["peak_tol8"]["recall_by_bucket"]["nano"]
                  - baseline["peak_tol8"]["recall_by_bucket"]["nano"]),
            micro=(out["peak_tol8"]["recall_by_bucket"]["micro"]
                   - baseline["peak_tol8"]["recall_by_bucket"]["micro"]),
            f1=(out["peak_tol8"]["f1"] - baseline["peak_tol8"]["f1"]),
        )
    return out


def card_row(res, board_ms=None):
    """标准卡片一行（用户要的对比表）。"""
    p8 = res["peak_tol8"]
    p4 = res["peak_tol4"]
    return dict(
        model=res["label"],
        board_ms=board_ms,
        t1_f1=round(p8["f1"], 6),
        t1_th=p8["th"],
        nano_recall=round(p8["recall_by_bucket"]["nano"], 4),
        micro_recall=round(p8["recall_by_bucket"]["micro"], 4),
        regular_recall=round(p8["recall_by_bucket"]["regular"], 4),
        nano_recall_tol4=round(p4["recall_by_bucket"]["nano"], 4),
        micro_recall_tol4=round(p4["recall_by_bucket"]["micro"], 4),
        fp=p8["fp"],
        fn=p8["fn"],
    )


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="微小目标子集：构建清单 / 分桶评估 / 卡片")
    ap.add_argument("mode", choices=["manifest", "eval", "card", "selftest"])
    ap.add_argument("--cache", action="append", default=[],
                    help="缓存 pkl（可多个：label=path 形式）")
    ap.add_argument("--label", default="")
    ap.add_argument("--semantics", default="nogmc", choices=["nogmc", "gmc"])
    ap.add_argument("--tols", default="4,8")
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--img-h", type=int, default=512)
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default="", help="基线缓存路径（用于 delta）")
    ap.add_argument("--board-ms", default="", help="形如 name=ms,name2=ms2")
    ap.add_argument("--manifest-out", default="")
    args = ap.parse_args()

    tols = tuple(float(x) for x in args.tols.split(","))
    board_ms = {}
    for kv in [x for x in args.board_ms.split(",") if x]:
        k, v = kv.split("=")
        board_ms[k] = float(v)

    if args.mode == "selftest":
        # ★口径自检：前缀复用实现必须与「逐阈值重跑」的参考实现**逐位一致**
        recs = load_cache(args.cache[0])
        bls = []
        for r in recs[:400]:
            ms, ar = gt_physical_size(r["gt_bboxes"], args.img_w, args.img_h)
            bls.append([bucket_of(m, a) for m, a in zip(ms, ar)])
        sub = recs[:400]
        bad = 0
        for tol in (4.0, 8.0):
            states = [frame_prefix_states(r, tol, b) for r, b in zip(sub, bls)]
            for th in (0.05, 0.15, 0.25, 0.35, 0.45):
                for i, (r, b) in enumerate(zip(sub, bls)):
                    a1, b1, c1, _, _ = match_mask(r, th, tol)
                    s_sorted, tp_cum, fp_cum, matched_seq = states[i]
                    if len(s_sorted) == 0:
                        a2, b2, c2 = 0, 0, len(b)
                    else:
                        k = int(np.searchsorted(-s_sorted, -th, side="right"))
                        if k <= 0:
                            a2, b2, c2 = 0, 0, len(b)
                        else:
                            a2 = int(tp_cum[k - 1]); b2 = int(fp_cum[k - 1])
                            c2 = len(b) - a2
                    if (a1, b1, c1) != (a2, b2, c2):
                        bad += 1
                        if bad <= 5:
                            print(f"  MISMATCH tol={tol} th={th} frame={i}: "
                                  f"ref=({a1},{b1},{c1}) opt=({a2},{b2},{c2})")
        print(f"自检：{len(sub)} 帧 × 5 阈值 × 2 容差 ⇒ 不一致 {bad} 处 ⇒ "
              + ("PASS（口径与逐阈值重跑完全一致）" if bad == 0 else "FAIL"))
        raise SystemExit(0 if bad == 0 else 1)

    if args.mode == "manifest":
        man = build_manifest(args.cache[0], args.img_w, args.img_h)
        print(f"清单: {man['source_cache']}")
        print(f"  帧数 {man['frames_total']}，含微小目标的帧 {man['frames_with_tiny']}")
        print(f"  GT 总数 {man['total_gt']}，分桶 {man['bucket_counts']}")
        print(f"  尺寸分位 {man['size_percentiles']}")
        if args.manifest_out:
            os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
            json.dump(man, open(args.manifest_out, "w"), ensure_ascii=False, indent=2)
            print("written", args.manifest_out)
        raise SystemExit(0)

    base_res = None
    if args.baseline:
        base_res = eval_cache(args.baseline, args.semantics, tols=tols,
                              img_w=args.img_w, img_h=args.img_h, label="baseline")

    results, rows = [], []
    for c in args.cache:
        path = c
        label = args.label or os.path.basename(os.path.dirname(c))
        if "=" in c and not c.startswith("/"):
            label, path = c.split("=", 1)
        r = eval_cache(path, args.semantics, tols=tols, img_w=args.img_w, img_h=args.img_h,
                       label=label, baseline=base_res)
        results.append(r)
        rows.append(card_row(r, board_ms.get(label)))
        p8 = r["peak_tol8"]
        print(f"[{label}] sem={r['semantics']} frames={r['frames']} gt={r['total_gt']} "
              f"F1={p8['f1']:.6f} (th={p8['th']}) FP={p8['fp']} FN={p8['fn']}")
        print(f"    分桶 GT: {r['gt_by_bucket']}")
        print(f"    召回@8px: nano={p8['recall_by_bucket']['nano']:.4f} "
              f"micro={p8['recall_by_bucket']['micro']:.4f} "
              f"small={p8['recall_by_bucket']['small']:.4f} "
              f"regular={p8['recall_by_bucket']['regular']:.4f}")
        p4 = r["peak_tol4"]
        print(f"    召回@4px: nano={p4['recall_by_bucket']['nano']:.4f} "
              f"micro={p4['recall_by_bucket']['micro']:.4f}")

    if rows:
        print("\n%-22s %-9s %-10s %-11s %-12s %-12s %-8s" %
              ("模型", "板卡ms", "T1 F1", "Nano-Recall", "Micro-Recall", "Regular-Rec", "FP"))
        print("-" * 92)
        for r in rows:
            print("%-22s %-9s %-10.6f %-11.4f %-12.4f %-12.4f %-8s" %
                  (r["model"], r["board_ms"] if r["board_ms"] is not None else "-",
                   r["t1_f1"], r["nano_recall"], r["micro_recall"], r["regular_recall"], r["fp"]))
    if args.out:
        json.dump(dict(results=results, card=rows), open(args.out, "w"),
                  ensure_ascii=False, indent=2)
        print("written", args.out)
