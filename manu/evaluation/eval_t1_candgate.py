#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T1（Anti-UAV 官方域）候选门评估器 —— 把 area/sums 判别式门移植到 T1 管线。

它是什么
--------
一个**离线评测脚本**：读一份 T1 推理缓存（``uav_cache*.pkl``），在
（置信度阈值 × area_min × sums 门）三维上扫参，报告 T1 的最优 F1 / TP / FP / FN，
并明确标出哪些配置满足 T1 红线。

它不是什么
----------
* 不是新的检测器、不是训练、不碰 GPU。
* 不是新的匹配口径：匹配与聚合**直接复用**
  ``manu/evaluation/eval_arm_pc_gmc_ab.py`` 的 ``match_at`` / ``prf`` / ``sweep`` /
  ``DEFAULT_THS``（8 px 点对点、按分数降序贪心、15 档阈值扫描）。
* 不是阈值改动：门只过滤候选，全局置信度阈值照扫。
* 不是 T3 配方的逐字搬运：``area<=`` / ``sums<=`` 两个**上限门**是本脚本额外加的，
  用来证伪"T1 的虚警是"胖"杂波"这一假设；T3 原配方只有下限门。

怎么用
------
::

    python manu/evaluation/eval_t1_candgate.py \
        --cache /path/uav_cache.pkl --label trial_0049 \
        --redline 0.8958967482400269 \
        --area-min 0,1,2,3 --sums-abs 0.3,0.6,1.0 --sums-q 0.1,0.2,0.3 \
        --out-json /path/t1_candgate.json

旧缓存（没有 ``pred_area`` / ``pred_sums`` 键）照读：缺键时按「不设门」处理
（``area = +inf`` / ``sums = +inf``），门对它们恒为 no-op。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.eval_arm_pc_gmc_ab import DEFAULT_THS, match_at, prf, sweep

# The frozen T1 red line of the current gate chain (frozen_baseline T1 - 0.005).
RED_LINE = 0.8958967482400269


def parse_args() -> argparse.Namespace:
    """Command line of the evaluator."""
    p = argparse.ArgumentParser(description="T1 candidate gate (area / sums) evaluation")
    p.add_argument("--cache", action="append", required=True, help="cache .pkl of one arm (repeatable)")
    p.add_argument("--label", action="append", default=[], help="label of the arm (aligned with --cache)")
    p.add_argument("--tol", type=float, default=8.0, help="matching tolerance in native px")
    p.add_argument("--ths", default="", help="comma separated thresholds; default = 15-point 0.05..0.75")
    p.add_argument("--area-min", default="0,1,2,3", help="comma separated connected-component area floors")
    p.add_argument("--area-max", default="", help="comma separated connected-component area ceilings")
    p.add_argument("--sums-abs", default="", help="comma separated absolute sums floors")
    p.add_argument("--sums-max", default="", help="comma separated absolute sums ceilings")
    p.add_argument("--sums-q", default="", help="comma separated sums quantile levels (of the whole pool)")
    p.add_argument("--redline", type=float, default=RED_LINE, help="T1 red line")
    p.add_argument("--expect-f1", type=float, default=None, help="self-calibration gate on the closed gate")
    p.add_argument("--expect-tp", type=int, default=None)
    p.add_argument("--expect-fp", type=int, default=None)
    p.add_argument("--cross-check-frames", type=int, default=200,
                   help="frames cross-checked against the reference match_at (0 disables)")
    p.add_argument("--out-json", default="")
    return p.parse_args()


def load_cache_with_attrs(path: str) -> tuple[dict, dict]:
    """Load a T1 cache exactly like ``eval_arm_pc_gmc_ab.load_cache`` plus the optional attributes.

    The numeric part (gt_pts / pred_points / pred_scores, all float64) is built with the same
    ``np.asarray(..., dtype=np.float64).reshape(...)`` calls, so the matching sees bit-identical
    numbers.  A cache without ``pred_area`` / ``pred_sums`` gets ``+inf`` for both, i.e. the gate
    is a no-op for it.

    Returns:
        ``(cache, stats)`` where ``cache`` maps ``im_name`` to the record and ``stats`` reports
        how many frames carry the attributes and how many candidates they cover.
    """
    with open(path, "rb") as f:
        rows = pickle.load(f)
    cache: dict[str, dict] = {}
    n_with_attrs = 0
    n_candidates = 0
    n_attr_candidates = 0
    for r in rows:
        points = np.asarray(r.get("pred_points", np.zeros((0, 2))), dtype=np.float64).reshape(-1, 2)
        area_raw = r.get("pred_area", None)
        sums_raw = r.get("pred_sums", None)
        if area_raw is None or sums_raw is None:
            area = np.full(len(points), np.inf, dtype=np.float64)
            sums = np.full(len(points), np.inf, dtype=np.float64)
        else:
            area = np.asarray(area_raw, dtype=np.float64).reshape(-1)
            sums = np.asarray(sums_raw, dtype=np.float64).reshape(-1)
            if len(area) != len(points) or len(sums) != len(points):
                raise RuntimeError(
                    f"[CAND-GATE] attribute length mismatch in {r.get('im_name')}: "
                    f"points={len(points)} area={len(area)} sums={len(sums)}"
                )
            n_with_attrs += 1
            n_attr_candidates += len(points)
        n_candidates += len(points)
        cache[r["im_name"]] = {
            "gt_pts": np.asarray(r.get("gt_pts", np.zeros((0, 2))), dtype=np.float64).reshape(-1, 2),
            "pred_points": points,
            "pred_scores": np.asarray(r.get("pred_scores", np.zeros((0,))), dtype=np.float64).reshape(-1),
            "pred_area": area,
            "pred_sums": sums,
        }
    stats = {
        "frames": len(cache),
        "candidates": n_candidates,
        "frames_with_attrs": n_with_attrs,
        "candidates_with_attrs": n_attr_candidates,
    }
    return cache, stats


def greedy_cumulative(gt_pts: np.ndarray, pts: np.ndarray, tol2: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy match of a score-descending candidate list, accumulated over the prefix.

    The matching rule is the one of ``eval_arm_pc_gmc_ab.match_at``: the candidates are processed
    in descending score order and every candidate takes the closest still-unmatched GT within the
    tolerance.  Because the rule only depends on the *order* of the candidates, the counts at a
    confidence threshold ``th`` are the counts of the prefix that contains every candidate with
    ``score >= th`` -- i.e. all thresholds come out of a single pass.

    Args:
        gt_pts: (M, 2) GT centres.
        pts: (N, 2) candidate points, already sorted by descending score.
        tol2: squared matching tolerance.

    Returns:
        ``(cum_tp, cum_fp)``, both of length N + 1: the TP / FP counts after the first k
        candidates have been processed.
    """
    n = len(pts)
    cum_tp = np.zeros(n + 1, dtype=np.int64)
    cum_fp = np.zeros(n + 1, dtype=np.int64)
    if n == 0:
        return cum_tp, cum_fp
    if len(gt_pts):
        d2 = ((gt_pts[None, :, :] - pts[:, None, :]) ** 2).sum(axis=2)
    else:
        d2 = None
    used = np.zeros(len(gt_pts), dtype=bool)
    tp = fp = 0
    for k in range(n):
        if d2 is not None:
            row = d2[k].copy()
            row[used] = np.inf
            j = int(np.argmin(row))
            if row[j] <= tol2:
                used[j] = True
                tp += 1
                cum_tp[k + 1] = tp
                cum_fp[k + 1] = fp
                continue
        fp += 1
        cum_tp[k + 1] = tp
        cum_fp[k + 1] = fp
    return cum_tp, cum_fp


def prepare_frames(cache: dict, tol: float) -> list[dict]:
    """Pre-compute the per-frame candidate order and the closed-gate greedy counts.

    The candidates of every frame are sorted by descending score once; the closed-gate counts are
    computed once as well, so a gate that does not touch a frame can reuse them.
    """
    tol2 = tol * tol
    frames = []
    for name in sorted(cache):
        rec = cache[name]
        scores = rec["pred_scores"]
        n_gt = len(rec["gt_pts"])
        order = np.argsort(-scores, kind="stable") if len(scores) else np.zeros(0, dtype=np.int64)
        pts = rec["pred_points"][order]
        cum_tp, cum_fp = greedy_cumulative(rec["gt_pts"], pts, tol2)
        frames.append({
            "name": name,
            "n_gt": n_gt,
            "gt_pts": rec["gt_pts"],
            "order": order,
            "scores": scores[order],
            "pts": pts,
            "area": rec["pred_area"][order],
            "sums": rec["pred_sums"][order],
            "cum_tp": cum_tp,
            "cum_fp": cum_fp,
        })
    return frames


def sweep_gate(frames: list[dict], ths: list[float], tol: float, area_min: int,
               sums_floor: float | None, area_max: float | None = None,
               sums_max: float | None = None) -> list[dict]:
    """Sweep the confidence thresholds of one gate configuration.

    The gate keeps a candidate when ``area >= area_min``, ``area <= area_max``,
    ``sums >= sums_floor`` and ``sums <= sums_max`` (the ceilings are optional and only used to
    test the *broad clutter* hypothesis).  A candidate that the gate rejects is removed
    **before** the matching, i.e. it can neither become a TP nor an FP.
    """
    tol2 = tol * tol
    th_arr = np.asarray(ths, dtype=np.float64)
    totals = np.zeros((len(ths), 3), dtype=np.int64)
    for fr in frames:
        n_gt = fr["n_gt"]
        n_cand = len(fr["scores"])
        if n_cand == 0:
            totals[:, 2] += n_gt
            continue
        keep = np.ones(n_cand, dtype=bool)
        if area_min >= 1:
            keep &= fr["area"] >= area_min
        if area_max is not None:
            keep &= fr["area"] <= area_max
        if sums_floor is not None:
            keep &= fr["sums"] >= sums_floor
        if sums_max is not None:
            keep &= fr["sums"] <= sums_max
        if keep.all():
            cum_tp, cum_fp = fr["cum_tp"], fr["cum_fp"]
            kept_scores = fr["scores"]
        else:
            kept = np.nonzero(keep)[0]
            kept_scores = fr["scores"][kept]
            cum_tp, cum_fp = greedy_cumulative(fr["gt_pts"], fr["pts"][kept], tol2)
        # number of candidates with score >= th (kept_scores is descending)
        m = np.searchsorted(-kept_scores, -th_arr, side="right")
        # cum_tp[k] / cum_fp[k] are the counts after the first k candidates
        tp = cum_tp[m]
        fp = cum_fp[m]
        totals[:, 0] += tp
        totals[:, 1] += fp
        totals[:, 2] += n_gt - tp
    rows = []
    for i, th in enumerate(ths):
        tp, fp, fn = int(totals[i, 0]), int(totals[i, 1]), int(totals[i, 2])
        precision, recall, f1 = prf(tp, fp, fn)
        rows.append({"th": float(th), "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1})
    return rows


def cross_check(frames: list[dict], cache: dict, ths: list[float], tol: float, n_frames: int,
                area_min: int, sums_floor: float | None, area_max: float | None = None,
                sums_max: float | None = None) -> int:
    """Compare the fast sweep against the reference ``match_at`` on a sample of frames.

    Returns the number of checked (frame, threshold) pairs; raises when a single pair differs.
    """
    if n_frames <= 0:
        return 0
    tol2 = tol * tol
    rng = np.random.default_rng(0)
    picks = rng.choice(len(frames), size=min(n_frames, len(frames)), replace=False)
    checked = 0
    for i in picks:
        fr = frames[int(i)]
        rec = cache[fr["name"]]
        n_cand = len(fr["scores"])
        keep = np.ones(n_cand, dtype=bool)
        if area_min >= 1:
            keep &= fr["area"] >= area_min
        if area_max is not None:
            keep &= fr["area"] <= area_max
        if sums_floor is not None:
            keep &= fr["sums"] >= sums_floor
        if sums_max is not None:
            keep &= fr["sums"] <= sums_max
        idx = fr["order"][keep]
        gated = {
            "gt_pts": rec["gt_pts"],
            "pred_points": rec["pred_points"][idx],
            "pred_scores": rec["pred_scores"][idx],
        }
        kept_scores = fr["scores"][keep]
        if keep.all():
            cum_tp, cum_fp = fr["cum_tp"], fr["cum_fp"]
        else:
            cum_tp, cum_fp = greedy_cumulative(rec["gt_pts"], fr["pts"][keep], tol2)
        for th in ths:
            ref = match_at(gated, th, tol)
            m = int(np.searchsorted(-kept_scores, -th, side="right"))
            if m == 0:
                got = (0, 0, fr["n_gt"])
            else:
                tp = int(cum_tp[m])
                fp = int(cum_fp[m])
                got = (tp, fp, fr["n_gt"] - tp)
            checked += 1
            if tuple(int(v) for v in ref) != tuple(int(v) for v in got):
                raise RuntimeError(
                    f"[CAND-GATE] cross-check failed on frame {fr['name']} th={th}: "
                    f"reference={ref} fast={got}"
                )
    return checked


def gate_configs(area_mins: list[int], sums_abs: list[float], sums_q: list[float],
                 quantile_floors: dict[float, float], area_maxs: list[float],
                 sums_maxs: list[float]) -> list[dict]:
    """Enumerate the gate configurations of the sweep."""
    configs = [{"area_min": 0, "sums_floor": None, "area_max": None, "sums_max": None,
                "label": "CLOSED"}]
    for area_min in area_mins:
        if area_min < 1:
            continue
        configs.append({"area_min": area_min, "sums_floor": None, "area_max": None,
                        "sums_max": None, "label": f"area>={area_min}"})
        for floor in sums_abs:
            configs.append({"area_min": area_min, "sums_floor": floor, "area_max": None,
                            "sums_max": None, "label": f"area>={area_min},sums>={floor:g}"})
        for level in sums_q:
            floor = quantile_floors[level]
            configs.append({"area_min": area_min, "sums_floor": floor, "area_max": None,
                            "sums_max": None,
                            "label": f"area>={area_min},sums>=q{level:g}({floor:.4f})"})
    for area_max in area_maxs:
        configs.append({"area_min": 0, "sums_floor": None, "area_max": area_max,
                        "sums_max": None, "label": f"area<={area_max:g}"})
        for floor in sums_abs:
            configs.append({"area_min": 0, "sums_floor": floor, "area_max": area_max,
                            "sums_max": None,
                            "label": f"area<={area_max:g},sums>={floor:g}"})
    for sums_max in sums_maxs:
        configs.append({"area_min": 0, "sums_floor": None, "area_max": None,
                        "sums_max": sums_max, "label": f"sums<={sums_max:g}"})
        for floor in sums_abs:
            configs.append({"area_min": 0, "sums_floor": floor, "area_max": None,
                            "sums_max": sums_max,
                            "label": f"sums<={sums_max:g},sums>={floor:g}"})
    return configs


def main() -> None:
    """Entry point."""
    a = parse_args()
    ths = [float(x) for x in a.ths.split(",")] if a.ths else list(DEFAULT_THS)
    labels = a.label + [f"CACHE{i}" for i in range(len(a.label), len(a.cache))]
    area_mins = [int(x) for x in a.area_min.split(",") if x != ""]
    area_maxs = [float(x) for x in a.area_max.split(",") if x != ""]
    sums_abs = [float(x) for x in a.sums_abs.split(",") if x != ""]
    sums_maxs = [float(x) for x in a.sums_max.split(",") if x != ""]
    sums_q = [float(x) for x in a.sums_q.split(",") if x != ""]

    print("=" * 100)
    print("  T1 candidate gate (area / sums)   tol=%.1fpx  thresholds=%d  redline=%.16f"
          % (a.tol, len(ths), a.redline))
    for path, label in zip(a.cache, labels):
        print(f"  {label}: {path}")
    print("=" * 100)

    report = {"tol": a.tol, "ths": ths, "redline": a.redline, "arms": []}

    for path, label in zip(a.cache, labels):
        cache, stats = load_cache_with_attrs(path)
        frames = prepare_frames(cache, a.tol)
        names = sorted(cache)
        print(f"\n[{label}] frames={stats['frames']} candidates={stats['candidates']} "
              f"frames_with_attrs={stats['frames_with_attrs']} "
              f"candidates_with_attrs={stats['candidates_with_attrs']}")
        if stats["frames_with_attrs"] == 0:
            print(f"[{label}] [WARN] the cache carries no pred_area / pred_sums -> every gate is a no-op")

        # the quantile population: every candidate of the cache that owns a component.
        # Candidates without a component carry sums = 0 (their score is below component_thresh);
        # keeping them in the population would make every quantile degenerate to 0.
        quantile_floors: dict[float, float] = {}
        if sums_q:
            pool = np.concatenate([fr["sums"] for fr in frames]) if frames else np.zeros(1)
            finite = pool[np.isfinite(pool) & (pool > 0)]
            if finite.size == 0:
                print(f"[{label}] [WARN] no candidate with a component -> the quantile gates are no-ops")
                quantile_floors = {level: float("inf") for level in sums_q}
            else:
                quantile_floors = {level: float(np.percentile(finite, level * 100.0)) for level in sums_q}
                print(f"[{label}] quantile population: {finite.size} candidates with a component; "
                      + " ".join(f"q{level:g}={floor:.4f}" for level, floor in quantile_floors.items()))

        # reference cross-check of the closed gate against eval_arm_pc_gmc_ab.sweep
        reference_rows = sweep(cache, names, ths, a.tol)
        configs = gate_configs(area_mins, sums_abs, sums_q, quantile_floors, area_maxs, sums_maxs)
        rows_out = []
        for cfg in configs:
            rows = sweep_gate(frames, ths, a.tol, cfg["area_min"], cfg["sums_floor"],
                              cfg["area_max"], cfg["sums_max"])
            if cfg["label"] == "CLOSED":
                same = all(
                    r["th"] == q["th"] and r["tp"] == q["tp"] and r["fp"] == q["fp"] and r["fn"] == q["fn"]
                    for r, q in zip(rows, reference_rows)
                )
                identical_f1 = all(r["f1"] == q["f1"] for r, q in zip(rows, reference_rows))
                print(f"[{label}] [CLOSED GATE] reference sweep identical: tp/fp/fn={same} f1_bitwise={identical_f1}")
                if not same:
                    raise RuntimeError("[CAND-GATE] the closed gate does not reproduce the reference sweep")
            best = max(rows, key=lambda r: r["f1"])
            rows_out.append({"label": cfg["label"], "area_min": cfg["area_min"],
                             "sums_floor": cfg["sums_floor"], "area_max": cfg["area_max"],
                             "sums_max": cfg["sums_max"], "rows": rows, "best": best})

        hdr = (f"{'gate':<34} | {'th':>5} {'TP':>7} {'FP':>7} {'FN':>7} {'P':>8} {'R':>8} "
               f"{'F1':>10} | {'dF1(pp)':>8} {'dFP':>6} {'dTP':>6} | redline")
        print("\n" + hdr)
        print("-" * len(hdr))
        closed_best = rows_out[0]["best"]
        for entry in sorted(rows_out, key=lambda e: -e["best"]["f1"]):
            b = entry["best"]
            d_f1 = (b["f1"] - closed_best["f1"]) * 100.0
            d_fp = b["fp"] - closed_best["fp"]
            d_tp = b["tp"] - closed_best["tp"]
            mark = "PASS" if b["f1"] >= a.redline else "fail"
            print(f"{entry['label']:<34} | {b['th']:>5.2f} {b['tp']:>7} {b['fp']:>7} {b['fn']:>7} "
                  f"{b['precision']:>8.5f} {b['recall']:>8.5f} {b['f1']:>10.6f} | "
                  f"{d_f1:>+8.4f} {d_fp:>+6} {d_tp:>+6} | {mark}")

        passing = [e for e in rows_out if e["best"]["f1"] >= a.redline]
        best = max(rows_out, key=lambda e: e["best"]["f1"])
        print(f"[{label}] [BEST] {best['label']}  th={best['best']['th']:.2f}  "
              f"F1={best['best']['f1']:.9f}  TP={best['best']['tp']} FP={best['best']['fp']} "
              f"FN={best['best']['fn']}  -> {'PASS' if best['best']['f1'] >= a.redline else 'FAIL'}"
              f" (redline {a.redline:.9f}, margin {(best['best']['f1'] - a.redline) * 100:+.4f} pp)")
        print(f"[{label}] passing configurations: {len(passing)}/{len(rows_out)}")
        for entry in passing[:12]:
            b = entry["best"]
            print(f"[{label}]   PASS {entry['label']:<32} th={b['th']:.2f} F1={b['f1']:.9f} "
                  f"TP={b['tp']} FP={b['fp']} FN={b['fn']}")

        if a.cross_check_frames > 0:
            checked = cross_check(frames, cache, ths, a.tol, a.cross_check_frames,
                                  0, None, None, None)
            print(f"[{label}] [CROSS-CHECK] {checked} (frame, threshold) pairs identical to "
                  f"eval_arm_pc_gmc_ab.match_at")
            for entry in rows_out[1:]:
                if entry["sums_floor"] is not None or entry["area_max"] is not None \
                        or entry["sums_max"] is not None:
                    checked = cross_check(frames, cache, ths, a.tol, 20,
                                          entry["area_min"], entry["sums_floor"],
                                          entry["area_max"], entry["sums_max"])
                    print(f"[{label}] [CROSS-CHECK] gate '{entry['label']}': {checked} pairs identical")
                    break

        if a.expect_f1 is not None:
            got = closed_best["f1"]
            ok = abs(got - a.expect_f1) < 1e-12
            tp_ok = a.expect_tp is None or closed_best["tp"] == a.expect_tp
            fp_ok = a.expect_fp is None or closed_best["fp"] == a.expect_fp
            print(f"[{label}] [GATE] closed gate expected F1={a.expect_f1:.12f} "
                  f"TP={a.expect_tp} FP={a.expect_fp} | measured F1={got:.12f} "
                  f"TP={closed_best['tp']} FP={closed_best['fp']} -> "
                  f"{'PASS' if (ok and tp_ok and fp_ok) else 'FAIL'}")
            if not (ok and tp_ok and fp_ok):
                raise SystemExit(2)

        report["arms"].append({
            "label": label, "path": path, "stats": stats,
            "quantile_floors": quantile_floors,
            "closed": rows_out[0],
            "configs": rows_out,
        })

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[WROTE] {a.out_json}")


if __name__ == "__main__":
    main()
