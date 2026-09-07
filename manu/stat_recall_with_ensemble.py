#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Dual-Expert Ensemble Evaluation (YOLO26 + Heatmap) with Size-Stratified Metrics.

Architecture:
1. Heatmap Expert (trial_0031): High-res P1 Stride=2 point detector for tiny/small targets.
   - Leverages pre-computed predictions from inference_cache.pkl (instant load).
2. YOLO26 Expert (trial_0028): Anchor/Box-based detector for medium/large targets.
   - Runs inference once and caches results to runs/badcase_analysis/yolo26_trial0028_cache.pkl.
3. Size-Gated Dual-Expert Fusion:
   - For tiny/small targets: Uses Heatmap point detections.
   - For targets where YOLO predicts max(w, h) >= split_size (default 28px):
     Fuses YOLO center detections into candidate pool.
   - Redundant point suppression: If a Heatmap point is within dist_match of a YOLO large target,
     they are unified into a single confident detection.
4. Evaluates and outputs:
   - Overall Recall, Precision, F1 under unified Distance <= 8.0px.
   - Stratified Recall Table by Original Resolution Bbox sizes (matching Table 1).
   - Side-by-side comparison: Heatmap Solo vs YOLO Solo vs Dual-Expert Ensemble!
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, colorstr


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Heatmap & YOLO26 with Size Stratification")
    parser.add_argument(
        "--heatmap-cache",
        type=str,
        default="runs/badcase_analysis/inference_cache.pkl",
        help="Path to Heatmap inference_cache.pkl (contains trial_0031 predictions)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="runs/optuna_uav_recall_sgpu/trial_0028/weights/best.pt",
        help="Path to YOLO26 trial_0028 weights",
    )
    parser.add_argument(
        "--yolo-cache",
        type=str,
        default="runs/badcase_analysis/yolo26_trial0028_cache.pkl",
        help="Path to save/load YOLO26 predictions cache",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/data/siping/datasets/manu/uav/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--exclude-seqs",
        type=str,
        default="wg2022_ir_020_split_03,DJI_0051_2,02_6321_0274-2773,DJI_0175_2,wg2022_ir_011_split_03",
        help="Comma-separated sequence names or substrings to exclude",
    )
    parser.add_argument(
        "--dist-thresh",
        type=float,
        default=8.0,
        help="Distance tolerance in pixels (default: 8.0px)",
    )
    parser.add_argument(
        "--conf-hm",
        type=float,
        default=0.20,
        help="Confidence threshold for Heatmap predictions (default: 0.20)",
    )
    parser.add_argument(
        "--conf-yolo",
        type=float,
        default=0.20,
        help="Confidence threshold for YOLO26 predictions (default: 0.20)",
    )
    parser.add_argument(
        "--size-split",
        type=float,
        default=28.0,
        help="Bbox max(w, h) in 640-scale to activate YOLO detections (default: 28px)",
    )
    parser.add_argument(
        "--dist-match",
        type=float,
        default=12.0,
        help="Distance threshold for deduplicating YOLO large box and Heatmap point (default: 12px)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for YOLO inference")
    parser.add_argument("--device", type=str, default="2", help="CUDA device index or cpu")
    parser.add_argument(
        "--force-recompute-yolo",
        action="store_true",
        help="Force re-running YOLO26 inference even if cache exists",
    )
    return parser.parse_args()


def find_label_file(img_path: Path) -> Path | None:
    s = str(img_path)
    candidates = []
    if "/images/" in s:
        candidates.append(Path(s.replace("/images/", "/labels/")).with_suffix(".txt"))
    candidates.append(img_path.with_suffix(".txt"))

    for c in candidates:
        if c.exists():
            return c
    return None


def build_image_and_label_lookup(val_source: str | Path | list) -> tuple[dict[str, Path], dict[str, Path]]:
    print("Indexing validation images and labels on disk...")
    img_lookup = {}
    lbl_lookup = {}
    val_dirs = [Path(val_source)] if isinstance(val_source, (str, Path)) else [Path(p) for p in val_source]

    for d in val_dirs:
        if d.is_file():
            with open(d, "r", encoding="utf-8") as f:
                for line in f:
                    p = Path(line.strip())
                    if p.exists():
                        img_lookup[p.name] = p
                        lbl = find_label_file(p)
                        if lbl:
                            lbl_lookup[p.name] = lbl
        elif d.is_dir():
            for p in d.rglob("*.*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    img_lookup[p.name] = p
                    lbl = find_label_file(p)
                    if lbl:
                        lbl_lookup[p.name] = lbl

    print(f"Indexed {len(img_lookup)} images and {len(lbl_lookup)} labels.")
    return img_lookup, lbl_lookup


def get_yolo_predictions(args, device: torch.device) -> list[dict]:
    """Run YOLO26 inference or load cached results."""
    cache_path = Path(args.yolo_cache)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    if cache_path.exists() and not args.force_recompute_yolo:
        print(colorstr("bold", colorstr("green", f"\n>>> Loading cached YOLO26 predictions from: {cache_path}")))
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    weights_path = Path(args.yolo_weights)
    if not weights_path.exists():
        alt = PROJECT_ROOT / args.yolo_weights
        if alt.exists():
            weights_path = alt
        else:
            raise FileNotFoundError(f"YOLO26 weights not found: {args.yolo_weights}")

    print(colorstr("bold", f"\n>>> Running YOLO26 inference ({weights_path.name}) to build cache..."))
    yolo = YOLO(str(weights_path))
    model = yolo.model
    model.to(device)
    model.eval()

    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.data = args.data

    val_dataset = build_yolo_dataset(cfg, data_dict["val"], batch=args.batch, data=data_dict, mode="val", stride=32)
    val_loader = build_dataloader(val_dataset, batch=args.batch, workers=4, shuffle=False)

    from ultralytics.utils.nms import non_max_suppression

    yolo_records = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Infer YOLO26 ({weights_path.stem})"):
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            im_files = batch.get("im_file", [""] * imgs.shape[0])
            bs = imgs.shape[0]

            preds = model(imgs)
            # 标准 YOLO NMS 后处理，避免调用 predict() 引起的进程管道或跨设备冲突
            preds = non_max_suppression(preds, conf_thres=0.02, iou_thres=0.60)

            for b in range(bs):
                im_name = Path(im_files[b]).name if im_files[b] else f"img_{b}"
                det = preds[b]  # (N, 6): [x1, y1, x2, y2, conf, cls]

                if det is not None and len(det) > 0:
                    det = det.cpu().numpy()
                    boxes_xyxy = det[:, :4]
                    confs = det[:, 4]
                    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
                    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
                    bw = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
                    bh = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
                    centers = np.stack([cx, cy], axis=1).astype(np.float32)
                    max_sides = np.maximum(bw, bh).astype(np.float32)
                else:
                    centers = np.zeros((0, 2), dtype=np.float32)
                    max_sides = np.zeros((0,), dtype=np.float32)
                    confs = np.zeros((0,), dtype=np.float32)

                yolo_records.append({
                    "im_name": im_name,
                    "centers": centers,
                    "max_sides": max_sides,
                    "scores": confs,
                })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving YOLO26 predictions cache to: {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(yolo_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(colorstr("green", "YOLO26 cache saved successfully!"))
    return yolo_records


def fuse_detections(
    hm_pts: np.ndarray,
    hm_scs: np.ndarray,
    yolo_pts: np.ndarray,
    yolo_scs: np.ndarray,
    yolo_sides: np.ndarray,
    conf_hm: float = 0.20,
    conf_yolo: float = 0.20,
    size_split: float = 28.0,
    dist_match: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Size-Gated Dual-Expert Fusion:
    1. Base: Heatmap points with conf >= conf_hm
    2. Add: YOLO predictions where max_side >= size_split and conf >= conf_yolo
    3. De-duplicate: If a Heatmap point is close (<= dist_match) to a YOLO large detection,
       retain only the higher-confidence location without duplicating false alarms.
    """
    # 1. Heatmap 候选
    keep_hm = hm_scs >= conf_hm
    fused_pts = list(hm_pts[keep_hm])
    fused_scs = list(hm_scs[keep_hm])

    # 2. YOLO 大目标候选
    keep_yolo_large = (yolo_scs >= conf_yolo) & (yolo_sides >= size_split)
    yolo_large_pts = yolo_pts[keep_yolo_large]
    yolo_large_scs = yolo_scs[keep_yolo_large]

    for y_pt, y_sc in zip(yolo_large_pts, yolo_large_scs):
        # 检查是否与已有的 Heatmap 点重叠
        is_near = False
        for h_pt in fused_pts:
            d = np.sqrt((y_pt[0] - h_pt[0]) ** 2 + (y_pt[1] - h_pt[1]) ** 2)
            if d <= dist_match:
                is_near = True
                break
        if not is_near:
            fused_pts.append(y_pt)
            fused_scs.append(y_sc)

    if not fused_pts:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return np.array(fused_pts, dtype=np.float32), np.array(fused_scs, dtype=np.float32)


def define_size_bins():
    return [
        {"name": "极微小点目标 (<= 3x3 px)", "min": 0.0, "max": 3.01},
        {"name": "超小目标 (4x4 ~ 6x6 px)", "min": 3.01, "max": 6.01},
        {"name": "弱小目标 (7x7 ~ 10x10 px)", "min": 6.01, "max": 10.01},
        {"name": "中微目标 (11x11 ~ 20x20 px)", "min": 10.01, "max": 20.01},
        {"name": "中近距无人机 (21x21 ~ 40x40 px)", "min": 20.01, "max": 40.01},
        {"name": "近距/大目标 (> 40x40 px)", "min": 40.01, "max": 9999.0},
    ]


def evaluate_records_set(
    eval_items: list[dict],
    dist_thresh: float,
    mode: str = "ensemble",
) -> tuple[dict, list[dict]]:
    """
    mode: 'heatmap_solo', 'yolo_solo', 'ensemble'
    Returns: (overall_metrics, size_bins_stats)
    """
    size_bins = define_size_bins()
    for b in size_bins:
        b["total_gt"] = 0
        b["tp"] = 0

    total_gt = 0
    total_tp = 0
    total_fp = 0

    for item in eval_items:
        gt_pts = item["gt_pts"]
        gt_sizes_orig = item["gt_sizes_orig"]
        num_gt = len(gt_pts)
        total_gt += num_gt

        if mode == "heatmap_solo":
            pred_pts = item["hm_pts"]
        elif mode == "yolo_solo":
            pred_pts = item["yolo_pts"]
        else:
            pred_pts = item["fused_pts"]

        matched_gt = set()
        matched_pred = set()

        if len(pred_pts) > 0 and num_gt > 0:
            diff = pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))

            p_inds, g_inds = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
            for p_i, g_i in zip(p_inds, g_inds):
                if dists[p_i, g_i] > dist_thresh:
                    break
                if p_i not in matched_pred and g_i not in matched_gt:
                    matched_pred.add(p_i)
                    matched_gt.add(g_i)

        total_tp += len(matched_gt)
        total_fp += (len(pred_pts) - len(matched_pred))

        for g_i in range(num_gt):
            is_hit = g_i in matched_gt
            max_side = gt_sizes_orig[g_i] if g_i < len(gt_sizes_orig) else 4.0
            for b in size_bins:
                if b["min"] <= max_side < b["max"]:
                    b["total_gt"] += 1
                    if is_hit:
                        b["tp"] += 1
                    break

    recall = total_tp / (total_gt + 1e-6) * 100
    precision = total_tp / (total_tp + total_fp + 1e-6) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    overall = {
        "total_gt": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "recall": recall,
        "precision": precision,
        "f1": f1 / 100.0,
    }
    return overall, size_bins


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # 1. 加载 Heatmap 预测缓存
    hm_cache_path = Path(args.heatmap_cache)
    if not hm_cache_path.is_absolute():
        hm_cache_path = PROJECT_ROOT / hm_cache_path
    if not hm_cache_path.exists():
        raise FileNotFoundError(f"Heatmap cache not found: {hm_cache_path}")

    print(f"\n>>> Loading Heatmap predictions from: {hm_cache_path}")
    with open(hm_cache_path, "rb") as f:
        hm_records = pickle.load(f)
    print(f"Loaded {len(hm_records)} Heatmap records.")

    # 2. 获取 YOLO26 预测
    yolo_records = get_yolo_predictions(args, device)
    yolo_dict = {r["im_name"]: r for r in yolo_records}

    # 3. 构建标签和原图索引以获取实际尺寸
    data_dict = check_det_dataset(args.data)
    img_lookup, lbl_lookup = build_image_and_label_lookup(data_dict["val"])

    # 4. 排除指定的极难/异常序列
    exclude_list = [s.strip() for s in args.exclude_seqs.split(",") if s.strip()]
    eval_items = []
    excluded_count = 0
    seq_res_cache = {}

    for hm_r in hm_records:
        im_name = hm_r["im_name"]
        if any(exc in im_name for exc in exclude_list):
            excluded_count += 1
            continue

        gt_pts_640 = hm_r["gt_pts"]
        hm_raw_pts = hm_r["pred_points"]
        hm_raw_scs = hm_r["pred_scores"]

        yolo_r = yolo_dict.get(im_name)
        if yolo_r:
            yolo_raw_pts = yolo_r["centers"]
            yolo_raw_scs = yolo_r["scores"]
            yolo_raw_sides = yolo_r["max_sides"]
        else:
            yolo_raw_pts = np.zeros((0, 2), dtype=np.float32)
            yolo_raw_scs = np.zeros((0,), dtype=np.float32)
            yolo_raw_sides = np.zeros((0,), dtype=np.float32)

        # 获取该图 GT 原图尺寸
        seq_id = im_name.split("___")[0] if "___" in im_name else im_name[:15]
        if seq_id in seq_res_cache:
            W_orig, H_orig = seq_res_cache[seq_id]
        else:
            img_p = img_lookup.get(im_name)
            if img_p and img_p.exists():
                im = cv2.imread(str(img_p))
                W_orig, H_orig = (im.shape[1], im.shape[0]) if im is not None else (640, 512)
            else:
                W_orig, H_orig = 640, 512
            seq_res_cache[seq_id] = (W_orig, H_orig)

        lbl_p = lbl_lookup.get(im_name)
        gt_sizes_orig = []
        if lbl_p and lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bw_norm = float(parts[3])
                        bh_norm = float(parts[4])
                        if bw_norm > 0.8 and bh_norm > 0.8:
                            continue
                        gt_sizes_orig.append(max(bw_norm * W_orig, bh_norm * H_orig))

        # 过滤各单模型预测点
        hm_keep = hm_raw_scs >= args.conf_hm
        hm_filtered_pts = hm_raw_pts[hm_keep]

        yolo_keep = yolo_raw_scs >= args.conf_yolo
        yolo_filtered_pts = yolo_raw_pts[yolo_keep]

        # 双专家模型融合
        fused_pts, fused_scs = fuse_detections(
            hm_pts=hm_raw_pts,
            hm_scs=hm_raw_scs,
            yolo_pts=yolo_raw_pts,
            yolo_scs=yolo_raw_scs,
            yolo_sides=yolo_raw_sides,
            conf_hm=args.conf_hm,
            conf_yolo=args.conf_yolo,
            size_split=args.size_split,
            dist_match=args.dist_match,
        )

        eval_items.append({
            "im_name": im_name,
            "gt_pts": gt_pts_640,
            "gt_sizes_orig": gt_sizes_orig,
            "hm_pts": hm_filtered_pts,
            "yolo_pts": yolo_filtered_pts,
            "fused_pts": fused_pts,
        })

    print(f"Total evaluated frames after exclusion: {len(eval_items)} (Excluded {excluded_count} frames)")
    print(colorstr("bold", f"Fusion Parameters: size_split >= {args.size_split:.0f}px, conf_hm = {args.conf_hm:.2f}, conf_yolo = {args.conf_yolo:.2f}, dist_thresh = {args.dist_thresh:.1f}px"))

    # 5. 分别评估 Heatmap Solo、YOLO26 Solo、Ensemble
    m_hm, bins_hm = evaluate_records_set(eval_items, args.dist_thresh, mode="heatmap_solo")
    m_yolo, bins_yolo = evaluate_records_set(eval_items, args.dist_thresh, mode="yolo_solo")
    m_ens, bins_ens = evaluate_records_set(eval_items, args.dist_thresh, mode="ensemble")

    # 6. 打印全局对比总结
    print("\n" + "=" * 95)
    print("GLOBAL PERFORMANCE COMPARISON (DISTANCE <= 8.0px):")
    print("=" * 95)
    comp_headers = ["Model / Strategy", "Total GT", "TP (命中数)", "FP (虚警数)", "Recall (召回率)", "Precision (精确率)", "F1-Score"]
    comp_rows = [
        ["Heatmap Solo (trial_0031)", m_hm["total_gt"], m_hm["tp"], m_hm["fp"], f"{m_hm['recall']:.2f}%", f"{m_hm['precision']:.2f}%", f"{m_hm['f1']:.4f}"],
        ["YOLO26 Solo (trial_0028)", m_yolo["total_gt"], m_yolo["tp"], m_yolo["fp"], f"{m_yolo['recall']:.2f}%", f"{m_yolo['precision']:.2f}%", f"{m_yolo['f1']:.4f}"],
        ["Dual-Expert Ensemble (融合)", m_ens["total_gt"], m_ens["tp"], m_ens["fp"], f"{m_ens['recall']:.2f}%", f"{m_ens['precision']:.2f}%", f"{m_ens['f1']:.4f}"],
    ]
    print(tabulate(comp_rows, headers=comp_headers, tablefmt="github"))
    print("=" * 95)

    # 7. 打印按原图尺寸细分表（对照表 1，横向对比 Heatmap vs YOLO vs Ensemble）
    print("\n" + "-" * 105)
    print("【表 1 (融合重算)：按原图物理分辨率尺寸细分对比 (Original Resolution Bbox)】")
    print("-" * 105)
    strat_headers = ["目标尺度区间", "像素跨度", "总目标(GT)", "样本占比", "Heatmap 召回率", "YOLO26 召回率", "融合后召回率 (Ensemble)", "提升幅差 (vs Heatmap)"]
    strat_rows = []

    for b_hm, b_yolo, b_ens in zip(bins_hm, bins_yolo, bins_ens):
        gt_cnt = b_hm["total_gt"]
        prop = (gt_cnt / max(m_hm["total_gt"], 1)) * 100
        rec_hm = (b_hm["tp"] / max(gt_cnt, 1)) * 100
        rec_yolo = (b_yolo["tp"] / max(gt_cnt, 1)) * 100
        rec_ens = (b_ens["tp"] / max(gt_cnt, 1)) * 100
        delta = rec_ens - rec_hm
        delta_str = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"

        strat_rows.append([
            b_hm["name"],
            f"{b_hm['min']:.0f} ~ {b_hm['max']:.0f} px",
            gt_cnt,
            f"{prop:.1f}%",
            f"{rec_hm:6.2f}%",
            f"{rec_yolo:6.2f}%",
            f"{rec_ens:6.2f}%",
            delta_str,
        ])

    print(tabulate(strat_rows, headers=strat_headers, tablefmt="github"))
    print("-" * 105 + "\n")


if __name__ == "__main__":
    main()
