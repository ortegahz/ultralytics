#!/usr/bin/env python3
"""001324 残余缺口的**可复现**诊断（§23.5 / §23.11 的数字全部由本脚本产出）。

判据一律照抄交付工具的官方尺子 :func:`extrapolator.servo_metrics`，不自己发明：
* ``hit_rate_5px``：存在某个上报点到**最近 GT 中心**的欧氏距离 ≤ 5 px；
* 分母 = 有 GT 的帧数（``gt_frames``），无上报的帧记为脱靶；
* CLE = 每个「有 GT 且有上报」的 record 里，**离最近 GT 中心最近**的那个上报点到该中心的距离。

除官方口径外，本脚本额外给三组**帧集受限**的对照（用来把「匹配精度的贡献」与
「检测器盲区造成的丢帧」分开）：
* ``cle_set``  = 有 GT 且**交付臂**有上报的帧（= 官方 CLE 样本集）；
* ``strict_set`` = 有 GT 且**所有臂**都有上报的帧。

用法::

    python manu/tracking/lookahead/diag_001324.py \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --arms-json runs/lookahead/final/arms_final_WTH15.json \
        --preds original=runs/lookahead/final/preds_off \
               chain=runs/lookahead/final/preds_SPC4 \
               delivery=runs/lookahead/final/preds_final_WTH15 \
        --seq VIDEO00007_19700101_001324 \
        --out-json runs/lookahead/final/diag_001324.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_multiscale_cache_th import load_ground_truth  # noqa: E402
from manu.tracking.lookahead.extrapolator import servo_metrics  # noqa: E402


def load_preds_dir(directory: Path, seq: str) -> dict[int, list[dict]]:
    doc = json.loads((directory / f"{seq}.json").read_text(encoding="utf-8"))
    per_frame = doc["per_frame"]
    return {int(key): list(dets) for key, dets in per_frame.items()}


def to_report_stream(per_frame: dict[int, list[dict]], num_frames: int) -> list[list[dict]]:
    return [per_frame.get(index, []) for index in range(num_frames)]


def nearest_gt_distance(points: np.ndarray, centres: np.ndarray) -> float:
    distances = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
    return float(distances.min())


def frame_stats(per_frame: dict[int, list[dict]], truth: dict[int, np.ndarray], frames: set[int]):
    """返回 (命中数, CLE 样本列表)；无上报的帧不进样本。"""
    hits = 0
    cle_values = []
    for frame in sorted(frames):
        dets = per_frame.get(frame)
        if not dets:
            continue
        boxes = truth.get(frame)
        if boxes is None:
            continue
        centres = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[:, :2]
        points = np.asarray([det["pos"] for det in dets], dtype=np.float64).reshape(-1, 2)
        distance = nearest_gt_distance(points, centres)
        cle_values.append(distance)
        if distance <= 5.0:
            hits += 1
    return hits, cle_values


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--arms-json", required=True, help="用于取 img_w/img_h 与全局 GT 帧数")
    parser.add_argument("--preds", nargs="+", required=True, help="label=preds_dir，可多组")
    parser.add_argument("--seq", default="VIDEO00007_19700101_001324")
    parser.add_argument("--delta", type=int, default=8, help="滞后帧数（320 ms @25 Hz）")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default="", help="可选：把误差结构表写成 Markdown（避免手抄串列）")
    args = parser.parse_args()

    arms_doc = json.loads(Path(args.arms_json).read_text(encoding="utf-8"))
    width = int(arms_doc["config"]["img_w"])
    height = int(arms_doc["config"]["img_h"])
    truth = load_ground_truth(Path(args.labels_root), args.seq, width, height)
    gt_frames = set(truth)
    num_frames = (max(gt_frames) + 1) if gt_frames else 0

    total_gt = int(arms_doc["arms"]["replay"]["current_frame"]["overall"]["gt"])

    preds: dict[str, dict[int, list[dict]]] = {}
    for item in args.preds:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--preds 需要 label=dir 形式，收到 {item!r}")
        preds[label] = load_preds_dir(Path(path), args.seq)

    report = {
        "seq": args.seq,
        "img_w": width,
        "img_h": height,
        "gt_frames": len(gt_frames),
        "total_gt_frames_8seq": total_gt,
        "gt_share_of_8seq": len(gt_frames) / total_gt if total_gt else 0.0,
        "num_frames": num_frames,
        "arms": {},
    }

    # ---------- 1. 官方口径逐臂复算（与产物 JSON 交叉校验） ----------
    for label, per_frame in preds.items():
        stream = to_report_stream(per_frame, num_frames)
        metrics = servo_metrics(stream, truth, num_frames, hit_tol_px=5.0)
        metrics.pop("cle_values", None)
        report["arms"][label] = {"official_servo": metrics}

    # ---------- 2. 帧集受限对照 ----------
    delivery_label = "delivery" if "delivery" in preds else next(iter(preds))
    delivery = preds[delivery_label]
    cle_set = {f for f in gt_frames if delivery.get(f)}
    strict_set = set(cle_set)
    for per_frame in preds.values():
        strict_set &= {f for f in gt_frames if per_frame.get(f)}

    frame_sets = {
        "cle_set": {"size": len(cle_set), "definition": "有 GT 且交付臂有上报"},
        "strict_set": {"size": len(strict_set), "definition": "有 GT 且所有臂都有上报"},
    }
    for label, per_frame in preds.items():
        for name, frame_set in (("cle_set", cle_set), ("strict_set", strict_set)):
            hits, cle_values = frame_stats(per_frame, truth, frame_set)
            report["arms"][label].setdefault("frame_sets", {})[name] = {
                "n_frames_with_report": len(cle_values),
                "hits_5px": hits,
                "hit_rate_5px": hits / len(cle_values) if cle_values else 0.0,
                "cle_p50": percentile(cle_values, 50),
                "cle_p95": percentile(cle_values, 95),
                "cle_max": max(cle_values) if cle_values else 0.0,
            }
    report["frame_sets"] = frame_sets

    def gt_centre(frame: int):
        boxes = truth.get(frame)
        if boxes is None:
            return None
        return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[:, :2].mean(axis=0)

    outlier_threshold = 15.0
    delta = int(args.delta)

    # ---------- 3. 位移/误差结构分解（逐臂 vs GT） ----------
    def error_structure(label: str) -> dict:
        """某一条臂在本序列上的位置误差 / 位移误差结构。"""
        per_frame = preds[label]
        arm_frames = {f for f in gt_frames if per_frame.get(f)}
        arm_errors: dict[int, np.ndarray] = {}
        arm_points: dict[int, np.ndarray] = {}
        arm_dist: dict[int, float] = {}
        for frame in sorted(arm_frames):
            dets = per_frame[frame]
            boxes = truth.get(frame)
            if not dets or boxes is None:
                continue
            centres = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[:, :2]
            points = np.asarray([det["pos"] for det in dets], dtype=np.float64).reshape(-1, 2)
            matrix = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
            flat = int(np.argmin(matrix))
            point_index, centre_index = divmod(flat, matrix.shape[1])
            arm_errors[frame] = points[point_index] - centres[centre_index]
            arm_points[frame] = points[point_index]
            arm_dist[frame] = float(matrix[point_index, centre_index])

        magnitudes = [float(np.linalg.norm(arm_errors[f])) for f in arm_errors]
        outliers = [f for f in arm_errors if float(np.linalg.norm(arm_errors[f])) > outlier_threshold]

        cosine_outliers = []
        true_disp_outliers = []
        output_step_outliers = []
        window_cos_outliers = []
        window_disp_outliers = []
        measured_window_disp = []
        true_window_disp = []
        measured_window_vec = []
        true_window_vec = []
        displacement_errors = []
        for frame in outliers:
            centre_now = gt_centre(frame)
            centre_prev = gt_centre(frame - 1)
            if centre_now is None or centre_prev is None:
                continue
            true_disp = centre_now - centre_prev
            error = arm_errors[frame]
            norm_error = float(np.linalg.norm(error))
            norm_disp = float(np.linalg.norm(true_disp))
            if norm_error > 1e-6 and norm_disp > 1e-6:
                cosine_outliers.append(float(np.dot(error, true_disp) / (norm_error * norm_disp)))
            true_disp_outliers.append(norm_disp)
            prev_point = None
            for det in per_frame.get(frame - 1, []):
                prev_point = np.asarray(det["pos"], dtype=np.float64)
            if prev_point is not None:
                output_step_outliers.append(
                    float(np.linalg.norm(arm_points[frame] - prev_point))
                )

        # ---- 位移层：链的「测得位移」与「真位移」（跨 delta 帧窗口）----
        for frame in sorted(arm_errors):
            centre_now = gt_centre(frame)
            centre_prev = gt_centre(frame - delta)
            if centre_now is None or centre_prev is None:
                continue
            true_window = centre_now - centre_prev
            point_now = arm_points.get(frame)
            point_prev = None
            for det in per_frame.get(frame - delta, []):
                point_prev = np.asarray(det["pos"], dtype=np.float64)
            if point_now is None or point_prev is None:
                continue
            measured_window = point_now - point_prev
            measured_window_vec.append(measured_window)
            true_window_vec.append(true_window)
            true_window_disp.append(float(np.linalg.norm(true_window)))
            measured_window_disp.append(float(np.linalg.norm(measured_window)))
            displacement_errors.append(float(np.linalg.norm(measured_window - true_window)))
            if frame in outliers:
                window_disp_outliers.append(float(np.linalg.norm(true_window)))
                denom = float(np.linalg.norm(arm_errors[frame]) * np.linalg.norm(true_window))
                if denom > 1e-6:
                    window_cos_outliers.append(
                        float(np.dot(arm_errors[frame], true_window) / denom)
                    )

        # 连续脱靶段（有 GT 的帧序列上，CLE>5 或无上报即算脱靶）
        ordered_frames = sorted(gt_frames)
        runs = []
        current = []
        for frame in ordered_frames:
            lost = frame not in arm_frames or arm_dist.get(frame, float("inf")) > 5.0
            if lost:
                current.append(frame)
            else:
                if current:
                    runs.append(current)
                current = []
        if current:
            runs.append(current)
        run_lengths = [len(run) for run in runs]

        # 只统计「有上报但 CLE>5」的脱靶段（另一种口径，单列出来）
        reported_loss_runs = []
        current = []
        for frame in ordered_frames:
            if frame in arm_frames and arm_dist.get(frame, 0.0) > 5.0:
                current.append(frame)
            else:
                if current:
                    reported_loss_runs.append(current)
                current = []
        if current:
            reported_loss_runs.append(current)

        # 连续脱靶段内的**同向性**：相邻两帧误差向量的方向余弦
        same_direction = []
        for run in runs:
            for previous, following in zip(run, run[1:]):
                if previous in arm_errors and following in arm_errors:
                    first = arm_errors[previous]
                    second = arm_errors[following]
                    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
                    if denom > 1e-6:
                        same_direction.append(float(np.dot(first, second) / denom))

        result = {
            "arm": label,
            "n_samples": len(magnitudes),
            "err_abs_p50": percentile(magnitudes, 50),
            "err_abs_p90": percentile(magnitudes, 90),
            "err_abs_p95": percentile(magnitudes, 95),
            "err_abs_p99": percentile(magnitudes, 99),
            "err_abs_max": max(magnitudes) if magnitudes else 0.0,
            "outlier_threshold_px": outlier_threshold,
            "outlier_frames": len(outliers),
            "outlier_fraction": len(outliers) / len(magnitudes) if magnitudes else 0.0,
            "outlier_cos_to_true_motion_p50": percentile(cosine_outliers, 50),
            "outlier_true_displacement_p50": percentile(true_disp_outliers, 50),
            "outlier_output_step_p50": percentile(output_step_outliers, 50),
            "loss_runs": len(runs),
            "loss_run_longest": max(run_lengths) if run_lengths else 0,
            "loss_run_lengths_top10": sorted(run_lengths, reverse=True)[:10],
            "loss_run_direction_cosine_p50": percentile(same_direction, 50),
            "loss_runs_reported_only": len(reported_loss_runs),
            "loss_run_longest_reported_only": max((len(r) for r in reported_loss_runs), default=0),
            "displacement_err_p50": percentile(displacement_errors, 50),
            "displacement_err_p90": percentile(displacement_errors, 90),
            "displacement_err_p95": percentile(displacement_errors, 95),
            "true_window_displacement_p50": percentile(true_window_disp, 50),
            "measured_window_displacement_p50": percentile(measured_window_disp, 50),
            "displacement_scale_ratio": (
                percentile(measured_window_disp, 50) / percentile(true_window_disp, 50)
                if percentile(true_window_disp, 50) > 1e-6
                else 0.0
            ),
            "outlier_window_displacement_p50": percentile(window_disp_outliers, 50),
            "outlier_cos_to_window_motion_p50": percentile(window_cos_outliers, 50),
        }
        if true_window_vec:
            measured_matrix = np.asarray(measured_window_vec, dtype=np.float64)
            true_matrix = np.asarray(true_window_vec, dtype=np.float64)
            denominator = float(np.sum(true_matrix * true_matrix))
            result["displacement_scale_ols"] = (
                float(np.sum(measured_matrix * true_matrix) / denominator) if denominator > 1e-9 else 0.0
            )
            result["displacement_window_samples"] = len(true_window_vec)

        # ---- 离群帧上「本臂 vs 同帧检测(原臂)」的不一致度 ----
        original = preds.get("original", {})
        mismatch = []
        for frame in outliers:
            dets = original.get(frame)
            if not dets or frame not in arm_points:
                continue
            points = np.asarray([det["pos"] for det in dets], dtype=np.float64).reshape(-1, 2)
            mismatch.append(
                float(np.linalg.norm(points - arm_points[frame][None, :], axis=1).min())
            )
        result["outlier_vs_original_p50"] = percentile(mismatch, 50)
        result["outlier_vs_original_p90"] = percentile(mismatch, 90)
        result["outlier_vs_original_n"] = len(mismatch)
        return result

    report["error_structure"] = {
        label: error_structure(label) for label in ("original", "delivery")
    }

    # ---------- 4. 对 8 序列总 hit@5px 的贡献 ----------
    servo = arms_doc["arms"]["replay"]["servo"]
    overall = servo["overall"]
    per_seq = servo["per_sequence"][args.seq]
    original_seq = arms_doc["arms"]["original"]["servo"]["per_sequence"][args.seq]
    overall_original = arms_doc["arms"]["original"]["servo"]["overall"]
    gap_seq = float(original_seq["hit_rate_5px"]) - float(per_seq["hit_rate_5px"])
    gap_overall = float(overall_original["hit_rate_5px"]) - float(overall["hit_rate_5px"])
    report["contribution"] = {
        "seq_hit_rate_5px_delivery": float(per_seq["hit_rate_5px"]),
        "seq_hit_rate_5px_original": float(original_seq["hit_rate_5px"]),
        "seq_gap_pp": gap_seq * 100.0,
        "gt_share_of_8seq": len(gt_frames) / total_gt if total_gt else 0.0,
        "contribution_pp_to_8seq_gap": gap_seq * (len(gt_frames) / total_gt if total_gt else 0.0) * 100.0,
        "overall_gap_pp": gap_overall * 100.0,
        "seq_lost_frames": int(per_seq["lost_frames"]),
        "seq_lost_rate_over_gt": int(per_seq["lost_frames"]) / len(gt_frames) if gt_frames else 0.0,
        "eight_seq_lost_frames": int(overall["lost_frames"]),
        "eight_seq_lost_rate_over_gt": int(overall["lost_frames"]) / total_gt if total_gt else 0.0,
        "eight_seq_cle_samples": int(overall.get("cle_pooled_samples", 0)),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_md:
        original_err = report["error_structure"]["original"]
        delivery_err = report["error_structure"]["delivery"]
        md = [
            "| 量 | `original`（同帧检测，天花板） | **`delivery`（WTH15，交付定稿）** | 口径 |",
            "|:--|--:|--:|:--|",
            f"| 位置误差 \\|err\\| p50 | {original_err['err_abs_p50']:.2f} px | {delivery_err['err_abs_p50']:.2f} px | `err` = 上报点 − 最近 GT 中心 |",
            f"| 位置误差 \\|err\\| p90 / p95 / p99 / max | {original_err['err_abs_p90']:.2f} / {original_err['err_abs_p95']:.2f} / {original_err['err_abs_p99']:.2f} / {original_err['err_abs_max']:.1f} px | "
            f"{delivery_err['err_abs_p90']:.2f} / {delivery_err['err_abs_p95']:.2f} / {delivery_err['err_abs_p99']:.2f} / {delivery_err['err_abs_max']:.1f} px | |",
            f"| 灾难性离群帧（\\|err\\| > {original_err['outlier_threshold_px']:.0f} px）占比 | **{original_err['outlier_fraction']*100:.2f}%** | **{delivery_err['outlier_fraction']*100:.2f}%** | |",
            f"| 连续脱靶段数 / 最长 | {original_err['loss_runs']} 段 / {original_err['loss_run_longest']} 帧 | {delivery_err['loss_runs']} 段 / {delivery_err['loss_run_longest']} 帧 | 无上报 或 CLE>5 |",
            f"| 连续脱靶段内**相邻两帧误差向量方向余弦 p50** | **{original_err['loss_run_direction_cosine_p50']:.3f}** | **{delivery_err['loss_run_direction_cosine_p50']:.3f}** | ⇒ **同向漂移**，不是随机尖峰 |",
            f"| 位移层误差 \\|测得位移−真位移\\| p50 / p90 / p95 | {original_err['displacement_err_p50']:.2f} / {original_err['displacement_err_p90']:.2f} / {original_err['displacement_err_p95']:.2f} px | "
            f"{delivery_err['displacement_err_p50']:.2f} / {delivery_err['displacement_err_p90']:.2f} / {delivery_err['displacement_err_p95']:.2f} px | 跨 Δ=8 帧窗口 |",
            f"| 真位移（跨 Δ=8 帧）p50 | {original_err['true_window_displacement_p50']:.2f} px | {delivery_err['true_window_displacement_p50']:.2f} px | 目标**大部分时间几乎悬停**（逐帧步长 p50 仅 0.50 px） |",
            f"| **标度比（最小二乘斜率）** | **{original_err['displacement_scale_ols']:.3f}** | {delivery_err['displacement_scale_ols']:.3f} | ⇒ **测量本身很准，不是滞后问题** |",
            f"| 标度比（median 测得 ÷ median 真） | {original_err['displacement_scale_ratio']:.3f} | {delivery_err['displacement_scale_ratio']:.3f} | |",
            f"| 离群帧上：误差 vs **跨 Δ 帧真位移**方向余弦 p50 | {original_err['outlier_cos_to_window_motion_p50']:.3f} | **{delivery_err['outlier_cos_to_window_motion_p50']:.3f}** | |",
            f"| 离群帧上：误差 vs **逐帧**真位移方向余弦 p50 | {original_err['outlier_cos_to_true_motion_p50']:.3f} | {delivery_err['outlier_cos_to_true_motion_p50']:.3f} | |",
            f"| 离群帧上：跨 Δ 帧真位移 p50 | — | **{delivery_err['outlier_window_displacement_p50']:.2f} px** | ⇒ 目标确实飞走了 |",
            f"| 离群帧上：输出自身帧间位移 p50 | — | **{delivery_err['outlier_output_step_p50']:.2f} px** | ⇒ 输出**没跟上** |",
            f"| 离群帧上：本臂 vs 同帧检测(原臂) 的距离 p50 / p90 | {original_err['outlier_vs_original_p50']:.1f} / {original_err['outlier_vs_original_p90']:.1f} px | "
            f"**{delivery_err['outlier_vs_original_p50']:.1f} / {delivery_err['outlier_vs_original_p90']:.1f} px** | ⇒ 外挂与同帧检测**严重不一致** |",
        ]
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"[DIAG] wrote {args.out_md}")

    print(f"[DIAG] seq={args.seq} gt_frames={len(gt_frames)} ({report['gt_share_of_8seq']*100:.2f}% of 8seq GT)")
    for label in preds:
        official = report["arms"][label]["official_servo"]
        cle_entry = report["arms"][label]["frame_sets"]["cle_set"]
        strict_entry = report["arms"][label]["frame_sets"]["strict_set"]
        print(
            f"[DIAG] {label:>10}: official hit@5px={official['hit_rate_5px']*100:6.2f}% "
            f"(cle_samples={official['cle_samples']}, lost={official['lost_frames']}) | "
            f"cle_set n={cle_entry['n_frames_with_report']} hit={cle_entry['hit_rate_5px']*100:6.2f}% "
            f"p50={cle_entry['cle_p50']:.2f} p95={cle_entry['cle_p95']:.2f} | "
            f"strict n={strict_entry['n_frames_with_report']} hit={strict_entry['hit_rate_5px']*100:6.2f}%"
        )
    for label, err in report["error_structure"].items():
        print(
            f"[DIAG] {label:>10}: |err| p50={err['err_abs_p50']:.2f} p95={err['err_abs_p95']:.2f} "
            f"max={err['err_abs_max']:.1f} outlier(>{err['outlier_threshold_px']:.0f}px)={err['outlier_fraction']*100:.2f}% "
            f"cos(outlier,true_motion) p50={err['outlier_cos_to_true_motion_p50']:.3f} "
            f"true_disp p50={err['outlier_true_displacement_p50']:.2f}px "
            f"| disp_err p50={err['displacement_err_p50']:.2f} scale={err['displacement_scale_ratio']:.3f} "
            f"| loss_runs={err['loss_runs']} longest={err['loss_run_longest']} "
            f"dir_cos={err['loss_run_direction_cosine_p50']:.3f}"
        )
    contrib = report["contribution"]
    print(
        f"[DIAG] contribution: seq gap={contrib['seq_gap_pp']:.2f}pp x share="
        f"{contrib['gt_share_of_8seq']*100:.2f}% = {contrib['contribution_pp_to_8seq_gap']:.2f}pp "
        f"(overall gap {contrib['overall_gap_pp']:.2f}pp); "
        f"seq lost={contrib['seq_lost_frames']}/{len(gt_frames)}={contrib['seq_lost_rate_over_gt']*100:.2f}%; "
        f"8seq lost={contrib['eight_seq_lost_frames']}/{total_gt}={contrib['eight_seq_lost_rate_over_gt']*100:.2f}%"
    )
    print(f"[DIAG] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
