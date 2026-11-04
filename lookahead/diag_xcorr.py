"""判决性诊断：模板匹配位移 vs 质心链位移（8 序列，只读）。

要回答的问题：重推链上报的**位移**误差（1.66 px p50）是不是来自「质心偏置在链两端
发生变化」？如果是，那么用**同一份模板**在源帧/目标帧之间做相关（偏置精确抵消）
应当显著降低位移误差，从而抬高 hit@5px。

口径：
* 锚点 A(t−Δ) = 原臂（原脚本管线）在源帧的上报位置；
* 现方案位移 = replay 臂上报(t) − A(t−Δ)；
* 相关位移 = 模板匹配(Source 帧 A 处模板, Target 帧) 得到的位移；
* 两种位移都与「GT 位移」对账，并各自算上报位置的 hit@5px。
"""
import argparse
import json
import os

import cv2
import numpy as np

W, H = 640, 512
SEQS = [
    "VIDEO00006_19700101_000703",
    "VIDEO00007_19700101_000818",
    "VIDEO00007_19700101_001324",
    "VIDEO00008_19700101_001307",
    "VIDEO00009_19700101_001531",
    "VIDEO00029_19700101_014053",
    "VIDEO00033_19700101_015000",
    "VIDEO00034_19700101_015052",
]


def load_gt(labels_root, seq):
    truth = {}
    prefix = seq + "__"
    for name in os.listdir(labels_root):
        if not name.startswith(prefix) or not name.endswith(".txt"):
            continue
        idx = int(name[len(prefix):-4])
        rows = []
        with open(os.path.join(labels_root, name), encoding="utf-8") as handle:
            for line in handle:
                v = line.split()
                if len(v) < 5:
                    continue
                rows.append((float(v[1]) * W, float(v[2]) * H))
        if rows:
            truth[idx] = np.asarray(rows, dtype=np.float64)
    return truth


def load_preds(path):
    doc = json.load(open(path, encoding="utf-8"))
    out = {}
    for key, dets in doc["per_frame"].items():
        if not dets:
            continue
        best = max(dets, key=lambda d: float(d.get("score", 0.0)))
        out[int(key)] = np.asarray(best["pos"], dtype=np.float64)
    return out


def window(img, cx, cy, size):
    """取以 (cx,cy) 为中心、边长 size 的窗；越界时整体平移（与快核同一约定）。"""
    h, w = img.shape
    half = size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1, y1 = x0 + size, y0 + size
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        x0 -= x1 - w
        x1 = w
    if y1 > h:
        y0 -= y1 - h
        y1 = h
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        return None, (0, 0)
    return img[y0:y1, x0:x1], (x0, y0)


def make_template(img, xy, tsize, bg_size, th_k):
    """源帧模板：anchor 处的 tsize×tsize 权重图（按底噪阈值去掉背景）。"""
    patch, (ox, oy) = window(img, xy[0], xy[1], tsize)
    if patch is None:
        return None
    bgpatch, _ = window(img, xy[0], xy[1], bg_size)
    if bgpatch is None:
        return None
    sample = bgpatch[:: max(1, bg_size // 32), :: max(1, bg_size // 32)].astype(np.float32)
    bg = float(np.median(sample))
    mad = float(np.median(np.abs(sample - bg)))
    sigma = max(1.4826 * mad, 1e-3)
    weights = np.clip(patch.astype(np.float32) - (bg + th_k * sigma), 0.0, None)
    if float(weights.sum()) <= 1e-6:
        return None
    return weights


def match_displacement(target_img, template, anchor_xy, center_xy, radius, tsize):
    """在目标帧里匹配模板，返回 (位移, 峰值)。"""
    size = 2 * radius + tsize
    search, (ox, oy) = window(target_img, center_xy[0], center_xy[1], size)
    if search is None:
        return None, 0.0
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(search.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    px, py = maxloc
    dx = dy = 0.0
    if 0 < px < res.shape[1] - 1:
        a, b, c = float(res[py, px - 1]), float(res[py, px]), float(res[py, px + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dx = 0.5 * (a - c) / den
    if 0 < py < res.shape[0] - 1:
        a, b, c = float(res[py - 1, px]), float(res[py, px]), float(res[py + 1, px])
        den = a - 2 * b + c
        if abs(den) > 1e-9:
            dy = 0.5 * (a - c) / den
    # 模板中心（= 锚点）在目标帧里的绝对位置
    cx = ox + px + dx + (tsize - 1) / 2.0
    cy = oy + py + dy + (tsize - 1) / 2.0
    disp = np.asarray([cx - anchor_xy[0], cy - anchor_xy[1]], dtype=np.float64)
    return disp, float(maxv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--original-preds", required=True, help="原臂产物目录（锚点来源）")
    ap.add_argument("--replay-preds", required=True, help="replay 臂产物目录（现方案）")
    ap.add_argument("--delta", type=int, default=8)
    ap.add_argument("--tsize", type=int, default=24)
    ap.add_argument("--bg-size", type=int, default=96)
    ap.add_argument("--th-k", type=float, default=2.5)
    ap.add_argument("--radius", type=int, default=48)
    ap.add_argument("--min-peak", type=float, default=0.25)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    stats = {
        "chain_err": [], "xcorr_err": [], "hyb_err": [],
        "chain_pos": [], "xcorr_pos": [], "hyb_pos": [],
        "disp_chain": [], "disp_xcorr": [], "peak": [], "n_xcorr": 0, "n_hybrid": 0,
    }
    per_seq = {}
    for seq in SEQS:
        truth = load_gt(args.labels_root, seq)
        orig = load_preds(os.path.join(args.original_preds, seq + ".json"))
        rep = load_preds(os.path.join(args.replay_preds, seq + ".json"))
        fdir = os.path.join(args.frames_root, seq)
        seq_stat = {k: [] for k in ("chain", "xcorr", "hyb")}
        for idx in sorted(truth):
            src = idx - args.delta
            if src < 0 or idx not in rep or src not in orig:
                continue
            gt_now = truth[idx]
            anchor = orig[src]
            gt_src = truth.get(src)
            if gt_src is None:
                continue
            # 目标身份：取源帧离锚点最近的 GT
            j = int(np.argmin(np.linalg.norm(gt_src - anchor[None, :], axis=1)))
            gt_src_pt = gt_src[j]
            gt_now_pt = gt_now[j] if j < len(gt_now) else gt_now[0]
            gt_disp = gt_now_pt - gt_src_pt

            chain_pos = rep[idx]
            chain_pos_err = float(np.linalg.norm(chain_pos - gt_now_pt))
            chain_disp = chain_pos - anchor
            chain_disp_err = float(np.linalg.norm(chain_disp - gt_disp))

            f_src = cv2.imread(os.path.join(fdir, "frame_%06d.jpg" % src), cv2.IMREAD_GRAYSCALE)
            f_dst = cv2.imread(os.path.join(fdir, "frame_%06d.jpg" % idx), cv2.IMREAD_GRAYSCALE)
            if f_src is None or f_dst is None:
                continue
            template = make_template(f_src, anchor, args.tsize, args.bg_size, args.th_k)
            if template is None:
                seq_stat["chain"].append(chain_pos_err)
                continue
            disp, peak = match_displacement(f_dst, template, anchor, chain_pos, args.radius, args.tsize)
            if disp is None:
                seq_stat["chain"].append(chain_pos_err)
                continue
            xcorr_pos = anchor + disp
            xcorr_pos_err = float(np.linalg.norm(xcorr_pos - gt_now_pt))
            xcorr_disp_err = float(np.linalg.norm(disp - gt_disp))
            # 混合：相关峰值够高、且与链位移不矛盾 ⇒ 用相关；否则保留链
            consistent = float(np.linalg.norm(disp - chain_disp)) <= args.radius
            use_x = peak >= args.min_peak and consistent
            hyb_pos = xcorr_pos if use_x else chain_pos
            hyb_pos_err = float(np.linalg.norm(hyb_pos - gt_now_pt))

            stats["chain_err"].append(chain_pos_err)
            stats["xcorr_err"].append(xcorr_pos_err)
            stats["hyb_err"].append(hyb_pos_err)
            stats["chain_pos"].append(chain_pos)
            stats["disp_chain"].append(chain_disp_err)
            stats["disp_xcorr"].append(xcorr_disp_err)
            stats["peak"].append(peak)
            if use_x:
                stats["n_xcorr"] += 1
                stats["n_hybrid"] += 1
            seq_stat["chain"].append(chain_pos_err)
            seq_stat["xcorr"].append(xcorr_pos_err)
            seq_stat["hyb"].append(hyb_pos_err)
        per_seq[seq] = {
            k: {
                "n": len(v),
                "hit5": float(np.mean([x <= 5.0 for x in v]) * 100) if v else 0.0,
                "p50": float(np.percentile(v, 50)) if v else 0.0,
                "p95": float(np.percentile(v, 95)) if v else 0.0,
            }
            for k, v in seq_stat.items()
        }
        print(f"[SEQ] {seq} chain={per_seq[seq]['chain']} xcorr={per_seq[seq]['xcorr']}", flush=True)

    def summarize(values):
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "hit5": float(np.mean(arr <= 5.0) * 100),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(arr.mean()),
        }

    summary = {
        "config": vars(args),
        "position_error": {
            "chain": summarize(stats["chain_err"]),
            "xcorr": summarize(stats["xcorr_err"]),
            "hybrid": summarize(stats["hyb_err"]),
        },
        "displacement_error": {
            "chain": summarize(stats["disp_chain"]),
            "xcorr": summarize(stats["disp_xcorr"]),
        },
        "peak": summarize(stats["peak"]),
        "per_sequence": per_seq,
    }
    print(json.dumps(summary["position_error"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["displacement_error"], ensure_ascii=False, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] -> {args.out_json}")


if __name__ == "__main__":
    main()
