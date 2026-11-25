#!/usr/bin/env python3
"""把「点选跟踪(SOT)」结果与画质指标对上，并反解**画质合格线**。

输入
----
``--sot-dir``      ``vq_sot_link.py`` 的逐片 JSON（逐点击明细 + 逐窗口汇总）
``--windows``      ``report/windows.csv``（逐秒画质指标，26 列）
``--grid-summary`` ``report/grid_summary.csv``（逐片画质汇总 + UGR/TGR，§271 产物）

口径（与 §271 的 48 宫格探针严格对齐）
------------------------------------
同一个 6×8 = 48 格、同一个 32×32 模板、同一个 NCC 0.65 / 闭环 3 px 判据；本脚本新增的是

* **纹理地板 τ_det**：单格拉普拉斯方差 ≥ τ 才算「点得下去」。
  **τ 只用同模态「好片」的 P10 标定**（与 §271 同一纪律：**不用坏片标定**，避免循环论证）。
* **跟踪器家族**：``xcorr``（仓库自有配准核，逐帧重切模板）与 ``patch``（产品快核 + 极性自适应），
  由 ``--arms`` 指定（逗号分隔、按顺序去重；新增家族如 ``kcf`` 只需改这一个参数）。
* **两个时程**：``short`` = 5 帧（与 vq_grid 的 gap 相同，可与 UGR 逐项对照）、
  ``long`` = 1 秒（业务意义：「点一下能跟住多久」）。
* **两种点选方式**：``center``（点格心 = 随便点）与 ``corner``（点格内最优角点 = 挑了个看得见的结构）。

主指标
------
``ugr`` ＝ (有纹理 **且** 跟住 且 闭环成立) 的点击占比 —— 与 §271 的 **UGR** 同名同义；
``lock`` ＝ (跟住 且 闭环成立) 的点击占比 —— 与 §271 的 **TGR** 同名同义（不加纹理地板）。

输出
----
``report/sot_windows.csv``       逐秒：画质指标 + 点选跟踪指标
``report/sot_clips.csv``         逐片
``report/sot_groups.csv``        分组 × 点选方式 汇总（汇报主表）
``report/sot_discrimination.csv``每个画质指标的判别力（AUC / Spearman）
``report/sot_thresholds.csv``    **画质合格线**：满足目标可用率所需的画质阈值
``report/sot_tau_det.csv``       纹理地板标定记录
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from vq_metrics import METRIC_DIRECTION  # noqa: E402

# §260 判定有效的 5 个指标（3 个跨模态 + 2 个 IR 独有）；其余仅作对照
KEY_METRICS = (
    "lap_var",
    "edge_density",
    "global_contrast",
    "blockiness",
    "entropy",
    "snr_db",
    "noise_sigma",
    "dog_snr_max",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_arms(text: str) -> list[str]:
    """Ordered, de-duplicated tracker families (``xcorr`` / ``patch`` / ``kcf`` ...)."""
    arms: list[str] = []
    for part in str(text).split(","):
        name = part.strip()
        if name and name not in arms:
            arms.append(name)
    return arms


def union_fields(rows: list[dict], preferred: list[str]) -> list[str]:
    """Ordered union of the row keys (clips with different fps carry different horizon tags)."""
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q / 100.0 * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def auc_good_higher(good: list[float], bad: list[float]) -> float:
    """AUC of "good (higher) vs bad" = P(good > bad) + 0.5 P(good == bad)."""
    if not good or not bad:
        return float("nan")
    wins = 0.0
    for a in good:
        for b in bad:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(good) * len(bad))


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2.0 + 1.0
            for k in range(index, stop + 1):
                ranks[order[k]] = average
            index = stop + 1
        return ranks

    rx = rank([p[0] for p in pairs])
    ry = rank([p[1] for p in pairs])
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def best_threshold(values: list[float], scores: list[float]) -> tuple[float, float, float, int, int]:
    """Threshold on ``values`` maximising the mean-score gap (values already "larger = better")."""
    pairs = [(v, s) for v, s in zip(values, scores) if math.isfinite(v) and math.isfinite(s)]
    if len(pairs) < 40:
        return float("nan"), float("nan"), float("nan"), 0, 0
    pairs.sort(key=lambda p: p[0])
    candidates = sorted({p[0] for p in pairs})
    best = (float("nan"), float("nan"), float("nan"), 0, 0)
    best_gap = -1e9
    for index in range(1, len(candidates)):
        threshold = candidates[index]
        above = [s for v, s in pairs if v >= threshold]
        below = [s for v, s in pairs if v < threshold]
        if len(above) < 30 or len(below) < 30:
            continue
        gap = sum(above) / len(above) - sum(below) / len(below)
        if gap > best_gap:
            best_gap = gap
            best = (threshold, sum(above) / len(above), sum(below) / len(below), len(above), len(below))
    return best


def admission_line(good_pairs: list[tuple[float, float]], bad_pairs: list[tuple[float, float]],
                   pass_level: float = 0.5, min_tpr: float = 0.90,
                   max_fpr: float = 0.01) -> dict:
    """Quality lines derived from a per-window binary pass/fail on the tracking score.

    🔴 **Why a binary per-window pass/fail and not a median**: a median constraint is a *vacuous*
    constraint — the median of any subset is always >= the median of the whole, so "median of the
    above-set >= overall median" is satisfied by every candidate line (and yields a useless line
    that admits ~99% of everything).  A per-window binary is not vacuous.

    Definition (per modality, because IR and VIS metric scales differ by an order of magnitude):

    * a window counts as **trackable** when its UGR >= ``pass_level``
      (i.e. at least half of that second's simulated clicks locked on and closed the loop);
    * the **admission line** L is the **lowest** quality threshold such that
      **P(trackable | metric >= L, good) >= min_tpr**  (the guarantee: above the line, footage of
      this quality really does track), while
      **P(trackable | metric >= L, bad) <= max_fpr**  (the false-accept side: unusable footage
      must not slip through the pre-filter);
    * candidates are the 1%..99% quantiles of the GOOD distribution, so the line cannot be parked
      on a single outlier window.

    ⚠️ **Empirically only 1 of 24 (scope x metric) combinations admits a line that satisfies
    BOTH constraints** — the good clips have far too much per-window spread to reach
    ``TPR >= 0.90`` without collapsing the coverage.  Three lines are therefore reported
    side by side instead of hiding the failure behind one number:

    * ``line_strict``  — the admission line: lowest threshold with TPR >= min_tpr **and**
      FPR <= max_fpr.  ``NaN`` when no candidate qualifies (the honest answer for most metrics);
    * ``line_rec``     — the recommended line: the candidate with the **highest good TPR**
      among those with FPR <= max_fpr (trades coverage for a stronger guarantee);
    * ``line_minfpr``  — the lowest threshold that still keeps FPR <= max_fpr (maximises the
      share of good footage that is admitted).

    Every line carries its own TPR / FPR / good-coverage so the trade-off stays visible.
    ``fallback`` stays True whenever the strict pair could not be satisfied.
    """
    empty = {
        "line": float("nan"), "good_tpr": float("nan"), "bad_fpr": float("nan"),
        "good_coverage": float("nan"), "good_above_median": float("nan"),
        "good_below_median": float("nan"), "bad_below_median": float("nan"),
        "n_good_above": 0, "n_bad_above": 0, "fallback": False,
        "line_strict": float("nan"), "line_strict_tpr": float("nan"),
        "line_strict_fpr": float("nan"), "line_strict_cov": float("nan"),
        "line_rec": float("nan"), "line_rec_tpr": float("nan"),
        "line_rec_fpr": float("nan"), "line_rec_cov": float("nan"),
        "line_minfpr": float("nan"), "line_minfpr_tpr": float("nan"),
        "line_minfpr_fpr": float("nan"), "line_minfpr_cov": float("nan"),
    }
    if len(good_pairs) < 40 or not bad_pairs:
        return empty
    ordered = sorted(v for v, _ in good_pairs)
    candidates = []
    for step in range(1, 100):
        position = step / 100.0 * (len(ordered) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        candidates.append(ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low))
    scored = []
    for threshold in sorted(set(candidates)):
        good_above = [u for v, u in good_pairs if v >= threshold]
        good_below = [u for v, u in good_pairs if v < threshold]
        bad_above = [u for v, u in bad_pairs if v >= threshold]
        bad_below = [u for v, u in bad_pairs if v < threshold]
        if len(good_above) < 20:
            continue
        tpr = sum(1 for u in good_above if u >= pass_level) / len(good_above)
        fpr = (sum(1 for u in bad_above if u >= pass_level) / len(bad_above)) if bad_above else 0.0
        scored.append({
            "threshold": threshold, "tpr": tpr, "fpr": fpr,
            "good_coverage": len(good_above) / len(good_pairs),
            "good_above_median": percentile(good_above, 50),
            "good_below_median": percentile(good_below, 50) if good_below else float("nan"),
            "bad_below_median": percentile(bad_below, 50) if bad_below else float("nan"),
            "n_good_above": len(good_above), "n_bad_above": len(bad_above),
        })
    if not scored:
        return empty
    safe_pool = [c for c in scored if c["fpr"] <= max_fpr]
    strict_pool = [c for c in safe_pool if c["tpr"] >= min_tpr]
    # 三条线：① 准入线（最低的 TPR≥min_tpr 且 FPR≤max_fpr）② 推荐线（FPR 可接受里 TPR 最高）
    #         ③ 最低安全线（FPR≤max_fpr 里阈值最低 ⇒ 好片覆盖最高）
    strict = min(strict_pool, key=lambda c: c["threshold"]) if strict_pool else None
    rec = max(safe_pool, key=lambda c: (c["tpr"], -c["threshold"])) if safe_pool else None
    minfpr = min(safe_pool, key=lambda c: c["threshold"]) if safe_pool else None
    # 兜底：三条都不存在时取「通过率−误放率」最大者，仅作参考
    fallback_best = max(scored, key=lambda c: (c["tpr"] - c["fpr"], -c["threshold"]))

    def pack(candidate: dict | None, prefix: str) -> dict:
        if candidate is None:
            return {prefix: float("nan"), prefix + "_tpr": float("nan"),
                    prefix + "_fpr": float("nan"), prefix + "_cov": float("nan")}
        return {prefix: candidate["threshold"], prefix + "_tpr": candidate["tpr"],
                prefix + "_fpr": candidate["fpr"], prefix + "_cov": candidate["good_coverage"]}

    best = dict(fallback_best)
    best["good_tpr"] = best.pop("tpr")
    best["bad_fpr"] = best.pop("fpr")
    if strict is not None:
        best["line"] = strict["threshold"]
        best["good_tpr"] = strict["tpr"]
        best["bad_fpr"] = strict["fpr"]
        best["good_coverage"] = strict["good_coverage"]
        best["good_above_median"] = strict.get("good_above_median", float("nan"))
        best["good_below_median"] = strict.get("good_below_median", float("nan"))
        best["bad_below_median"] = strict.get("bad_below_median", float("nan"))
        best["n_good_above"] = strict["n_good_above"]
        best["n_bad_above"] = strict["n_bad_above"]
        best["fallback"] = False
    else:
        best["line"] = fallback_best["threshold"]
        best["fallback"] = True
    best.update(pack(strict, "line_strict"))
    best.update(pack(rec, "line_rec"))
    best.update(pack(minfpr, "line_minfpr"))
    return best


def derive_thresholds(scoped: list[dict], good: list[dict], bad: list[dict],
                      score_key: str, scope: str) -> list[dict]:
    """Derive the quality lines inside one scope (e.g. ``all`` / ``ir`` / ``vis``).

    🔴 **必须按模态分开跑**：IR 与 VIS 的指标量纲差一个数量级（好片 ``lap_var`` 中位
    IR 100 vs VIS 2102）⇒ 混池出来的线对其中一个模态是错的（IR 好片会被 ``lap_var ≥ 755``
    这类线全判不合格）。混池结果仅作「不分模态的整体参考」。
    """
    rows: list[dict] = []
    for metric in KEY_METRICS:
        direction = METRIC_DIRECTION.get(metric, 0)
        if direction == 0:
            continue
        signed = [
            (float(row[metric]) * (-direction), float(row[score_key]))
            for row in scoped
            if row.get(metric) not in (None, "") and row.get(score_key) not in (None, "")
        ]
        if len(signed) < 40:
            continue
        values = [v for v, _ in signed]
        scores = [s for _, s in signed]
        threshold, mean_above, mean_below, n_above, n_below = best_threshold(values, scores)
        good_target = percentile(
            [float(row[score_key]) for row in good if row.get(metric) not in (None, "")], 50
        )
        good_pairs = [
            (float(row[metric]) * (-direction), float(row[score_key]))
            for row in good if row.get(metric) not in (None, "") and row.get(score_key) not in (None, "")
        ]
        bad_pairs = [
            (float(row[metric]) * (-direction), float(row[score_key]))
            for row in bad if row.get(metric) not in (None, "") and row.get(score_key) not in (None, "")
        ]
        line = admission_line(good_pairs, bad_pairs)
        cov_thr = line["line"]
        cov_mean = line["good_above_median"]
        cov_above = line["n_good_above"]
        cov_below = line["n_bad_above"]
        rows.append({
            "scope": scope,
            "metric": metric,
            "direction": "higher_better" if direction < 0 else "lower_better",
            "raw_threshold": threshold,
            "metric_threshold": round(threshold * (-direction), 6) if math.isfinite(threshold) else float("nan"),
            "score": score_key,
            "mean_score_above": round(mean_above, 6),
            "mean_score_below": round(mean_below, 6),
            "gap": round(mean_above - mean_below, 6),
            "n_above": n_above,
            "n_below": n_below,
            "cover_threshold": round(cov_thr * (-direction), 6) if math.isfinite(cov_thr) else float("nan"),
            "cover_good_above_median": round(cov_mean, 6),
            "cover_good_below_median": round(line["good_below_median"], 6) if math.isfinite(line["good_below_median"]) else float("nan"),
            "cover_bad_below_median": round(line["bad_below_median"], 6) if math.isfinite(line["bad_below_median"]) else float("nan"),
            "cover_good_tpr": round(line["good_tpr"], 6) if math.isfinite(line["good_tpr"]) else float("nan"),
            "cover_bad_fpr": round(line["bad_fpr"], 6) if math.isfinite(line["bad_fpr"]) else float("nan"),
            "cover_n_good_above": cov_above,
            "cover_n_bad_above": cov_below,
            "cover_good_coverage": round(line["good_coverage"], 6) if math.isfinite(line["good_coverage"]) else float("nan"),
            "cover_fallback": line["fallback"],
            "cover_target_score": round(good_target, 6),
            # ---- 三条线（按第 9 节要求同时给出，FALLBACK 不隐藏）----
            "admit_threshold": round(line["line_strict"] * (-direction), 6) if math.isfinite(line["line_strict"]) else float("nan"),
            "admit_good_tpr": round(line["line_strict_tpr"], 6) if math.isfinite(line["line_strict_tpr"]) else float("nan"),
            "admit_bad_fpr": round(line["line_strict_fpr"], 6) if math.isfinite(line["line_strict_fpr"]) else float("nan"),
            "admit_good_coverage": round(line["line_strict_cov"], 6) if math.isfinite(line["line_strict_cov"]) else float("nan"),
            "admit_exists": bool(math.isfinite(line["line_strict"])),
            "rec_threshold": round(line["line_rec"] * (-direction), 6) if math.isfinite(line["line_rec"]) else float("nan"),
            "rec_good_tpr": round(line["line_rec_tpr"], 6) if math.isfinite(line["line_rec_tpr"]) else float("nan"),
            "rec_bad_fpr": round(line["line_rec_fpr"], 6) if math.isfinite(line["line_rec_fpr"]) else float("nan"),
            "rec_good_coverage": round(line["line_rec_cov"], 6) if math.isfinite(line["line_rec_cov"]) else float("nan"),
            "safe_threshold": round(line["line_minfpr"] * (-direction), 6) if math.isfinite(line["line_minfpr"]) else float("nan"),
            "safe_good_tpr": round(line["line_minfpr_tpr"], 6) if math.isfinite(line["line_minfpr_tpr"]) else float("nan"),
            "safe_bad_fpr": round(line["line_minfpr_fpr"], 6) if math.isfinite(line["line_minfpr_fpr"]) else float("nan"),
            "safe_good_coverage": round(line["line_minfpr_cov"], 6) if math.isfinite(line["line_minfpr_cov"]) else float("nan"),
        })
    return rows


def load_clicks(sot_dirs: list[Path], arms: list[str]) -> tuple[list[dict], list[dict]]:
    """Flatten the per-click records of every probe output."""
    clicks: list[dict] = []
    clips: list[dict] = []
    for sot_dir in sot_dirs:
        for path in sorted(sot_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload["summary"]
            clips.append(summary)
            horizons = sorted(int(h) for h in summary["horizons"])
            short_h, long_h = horizons[0], horizons[-1]
            for click in payload.get("clicks", []):
                record = {
                    "file": summary["video"],
                    "group": summary["group"],
                    "kind": summary["kind"],
                    "modality": summary["modality"],
                    "window": click["window"],
                    "cell": click["cell"],
                    "click_mode": click.get("click_mode", "center"),
                    "tex_lap_var": click.get("tex_lap_var", float("nan")),
                    "tex_std": click.get("tex_std", float("nan")),
                    "tex_lam": click.get("tex_lam", float("nan")),
                }
                for arm in arms:
                    block = click.get(arm, {})
                    arm_horizons = block.get("horizons", {})
                    for horizon, label in ((short_h, "short"), (long_h, "long")):
                        # 点击级 horizons 以 str(h) 为键（"5"/"30"），窗口级聚合才用 "h5"/"h30"
                        entry = arm_horizons.get(str(horizon), arm_horizons.get("h%d" % horizon, {}))
                        prefix = "%s_%s" % (arm, label)
                        record[prefix + "_survived"] = bool(entry.get("survived", False))
                        record[prefix + "_closure"] = entry.get("closure", float("nan"))
                        record[prefix + "_ncc"] = entry.get("ncc", float("nan"))
                clicks.append(record)
    return clicks, clips


def calibrate_tau(clicks: list[dict]) -> dict[str, float]:
    """Texture floor per modality: P10 of the per-click Laplacian variance on the GOOD clips.

    Same discipline as §271: calibrated on good clips only (never on the bad ones).
    """
    taus: dict[str, float] = {}
    for modality in sorted({click["modality"] for click in clicks}):
        values = [
            float(click["tex_lap_var"])
            for click in clicks
            if click["modality"] == modality and click["group"] == "good" and math.isfinite(click["tex_lap_var"])
        ]
        taus[modality] = percentile(values, 10)
    return taus


def aggregate_windows(clicks: list[dict], taus: dict[str, float], arms: list[str]) -> list[dict]:
    """Per (file, window, click_mode) tracking rates, with and without the texture floor."""
    buckets: dict[tuple[str, int, str], list[dict]] = {}
    for click in clicks:
        key = (click["file"], int(click["window"]), click["click_mode"])
        buckets.setdefault(key, []).append(click)

    rows: list[dict] = []
    for (file, window, mode), items in sorted(buckets.items()):
        tau = taus.get(items[0]["modality"], float("nan"))
        record = {
            "file": file,
            "group": items[0]["group"],
            "kind": items[0]["kind"],
            "modality": items[0]["modality"],
            "window": window,
            "click_mode": mode,
            "n_clicks": len(items),
            "tau_det": tau,
            "n_flat": sum(1 for c in items if c["tex_std"] < 3.0),
            "n_textured": sum(1 for c in items if math.isfinite(c["tex_lap_var"]) and c["tex_lap_var"] >= tau),
        }
        for arm in arms:
            for horizon in ("short", "long"):
                locked = 0
                usable = 0
                closures = []
                for click in items:
                    survived = click["%s_%s_survived" % (arm, horizon)]
                    closure = click["%s_%s_closure" % (arm, horizon)]
                    if not survived:
                        continue
                    if math.isfinite(closure) and closure <= 3.0:
                        locked += 1
                        closures.append(closure)
                        # UGR 口径：有纹理 **且** 跟住 **且** 闭环成立（与 §271 一致）
                        if math.isfinite(click["tex_lap_var"]) and click["tex_lap_var"] >= tau:
                            usable += 1
                record["%s_%s_lock" % (arm, horizon)] = round(locked / len(items), 6)
                record["%s_%s_ugr" % (arm, horizon)] = round(usable / len(items), 6)
                record["%s_%s_clos" % (arm, horizon)] = (
                    round(percentile(closures, 50), 6) if closures else float("nan")
                )
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Join the click-SOT probe with the quality metrics")
    parser.add_argument("--sot-dir", required=True, help="Comma separated list of probe output dirs")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--grid-summary", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score", default="xcorr_long_ugr", help="Column used as the tracking score")
    parser.add_argument("--kind", default="source", help="Only this kind takes part in the good/bad discrimination")
    parser.add_argument("--click-mode", default="all", choices=["all", "center", "corner"])
    parser.add_argument("--arms", default="xcorr,patch",
                        help="Comma separated tracker families (e.g. xcorr,patch,kcf); every family gets its own columns")
    cli = parser.parse_args()

    arms = parse_arms(cli.arms)
    if not arms:
        parser.error("--arms needs at least one tracker family")
    out_dir = Path(cli.out_dir)
    sot_dirs = [Path(p) for p in str(cli.sot_dir).split(",") if p.strip()]
    clicks, clips = load_clicks(sot_dirs, arms)
    taus = calibrate_tau(clicks)

    tau_rows = [
        {
            "modality": modality,
            "tau_det_p10_good": round(value, 6),
            "calibration": "P10 of per-click Laplacian variance over GOOD clips only (center+corner clicks, no click_mode filter)",
            "n_clicks": sum(
                1
                for click in clicks
                if click["modality"] == modality
                and click["group"] == "good"
                and math.isfinite(click["tex_lap_var"])
            ),
        }
        for modality, value in sorted(taus.items())
    ]
    write_csv(out_dir / "sot_tau_det.csv", tau_rows, list(tau_rows[0].keys()))

    window_rows = aggregate_windows(clicks, taus, arms)
    if cli.click_mode != "all":
        window_rows = [row for row in window_rows if row["click_mode"] == cli.click_mode]

    quality_by_key = {}
    for row in read_csv(Path(cli.windows)):
        quality_by_key[(row["file"], int(row["window"]))] = row

    joined: list[dict] = []
    for row in window_rows:
        quality = quality_by_key.get((row["file"], int(row["window"])))
        if quality is None:
            continue
        merged = dict(row)
        for key, value in quality.items():
            if key in ("file", "group", "kind", "modality", "window"):
                continue
            merged[key] = value
        joined.append(merged)
    if joined:
        write_csv(out_dir / "sot_windows.csv", joined, union_fields(joined, list(joined[0].keys())))

    grid_by_file = {}
    if cli.grid_summary and Path(cli.grid_summary).exists():
        for row in read_csv(Path(cli.grid_summary)):
            grid_by_file[row["file"]] = row

    clip_rows: list[dict] = []
    for summary in clips:
        record = {
            "file": summary["video"],
            "group": summary["group"],
            "kind": summary["kind"],
            "modality": summary["modality"],
            "frames": summary["frames_decoded"],
            "fps": summary["fps"],
            "windows": summary["windows"],
            "clicks": summary["clicks"],
            "elapsed_s": summary["elapsed_s"],
        }
        for arm in arms:
            for horizon in ("short", "long"):
                for key, source in (("lock", "lock_rate"), ("surv", "survived_ratio"), ("clos", "closure_median")):
                    record["%s_%s_%s" % (arm, horizon, key)] = summary[arm].get(
                        ("h%d" % min(summary["horizons"])) if horizon == "short" else ("h%d" % max(summary["horizons"])),
                        {},
                    ).get(source, float("nan"))
        grid = grid_by_file.get(summary["video"], {})
        for key in ("ugr_p50", "ugr", "tgr_dog_p50", "tgr_raw_p50", "det_ratio_p50", "flat_cells_p50",
                    "lap_var", "edge_density", "global_contrast", "blockiness", "entropy", "tier",
                    "core_violations", "weak_violations"):
            if key in grid:
                record["grid_%s" % key] = grid[key]
        clip_rows.append(record)
    if clip_rows:
        clip_rows.sort(key=lambda r: (r["group"] != "good", r["kind"] != "source", r["file"]))
        write_csv(out_dir / "sot_clips.csv", clip_rows, union_fields(clip_rows, list(clip_rows[0].keys())))

    # ---------------- 分组 × 点选方式 汇总（汇报主表） ----------------
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in joined:
        groups.setdefault((row["group"], row["kind"], row["click_mode"]), []).append(row)
    group_rows: list[dict] = []
    for (group, kind, mode), rows in sorted(groups.items()):
        record = {
            "group": group,
            "kind": kind,
            "click_mode": mode,
            "clips": len({row["file"] for row in rows}),
            "windows": len(rows),
            "clicks": sum(int(row["n_clicks"]) for row in rows),
            "textured_ratio": round(
                sum(int(row["n_textured"]) for row in rows) / max(1, sum(int(row["n_clicks"]) for row in rows)), 6
            ),
        }
        for arm in arms:
            for horizon in ("short", "long"):
                for key in ("lock", "ugr"):
                    values = [
                        float(row["%s_%s_%s" % (arm, horizon, key)])
                        for row in rows
                        if row.get("%s_%s_%s" % (arm, horizon, key)) not in (None, "")
                    ]
                    if values:
                        record["%s_%s_%s_p50" % (arm, horizon, key)] = round(percentile(values, 50), 6)
                        record["%s_%s_%s_mean" % (arm, horizon, key)] = round(sum(values) / len(values), 6)
        group_rows.append(record)
    if group_rows:
        write_csv(out_dir / "sot_groups.csv", group_rows, union_fields(group_rows, list(group_rows[0].keys())))

    # ---------------- 判别力（只用 --kind 指定的源片） ----------------
    scoped = [row for row in joined if row["kind"] == cli.kind]
    good = [row for row in scoped if row["group"] == "good"]
    bad = [row for row in scoped if row["group"] == "bad"]
    score_key = cli.score
    if joined and score_key not in joined[0]:
        print("[FATAL] score column %r is not produced by --arms %s" % (score_key, ",".join(arms)))
        raise SystemExit(2)

    discrimination: list[dict] = []
    thresholds: list[dict] = []
    for metric in KEY_METRICS:
        direction = METRIC_DIRECTION.get(metric, 0)
        if direction == 0:
            continue
        good_values = [float(row[metric]) for row in good if row.get(metric) not in (None, "")]
        bad_values = [float(row[metric]) for row in bad if row.get(metric) not in (None, "")]
        good_scores = [float(row[score_key]) for row in good if row.get(metric) not in (None, "")]
        bad_scores = [float(row[score_key]) for row in bad if row.get(metric) not in (None, "")]
        auc_metric = auc_good_higher(good_values, bad_values)
        if direction > 0:
            auc_metric = 1.0 - auc_metric
        auc_score = auc_good_higher(good_scores, bad_scores)
        rho = spearman(good_values + bad_values, good_scores + bad_scores)
        discrimination.append({
            "metric": metric,
            "direction": "higher_better" if direction < 0 else "lower_better",
            "good_median": round(percentile(good_values, 50), 6),
            "bad_median": round(percentile(bad_values, 50), 6),
            "auc_metric_vs_group": round(auc_metric, 6),
            "auc_score_vs_group": round(auc_score, 6),
            "spearman_metric_vs_score": round(rho, 6),
            "n_good": len(good_values),
            "n_bad": len(bad_values),
        })
    # ---- 画质合格线：混池（参考）+ 按模态分别（正式用） ----
    threshold_scopes: list[tuple[str, list[dict], list[dict], list[dict]]] = [
        ("all", scoped, good, bad),
    ]
    for modality in sorted({row["modality"] for row in scoped}):
        sub = [row for row in scoped if row["modality"] == modality]
        sub_good = [row for row in good if row["modality"] == modality]
        sub_bad = [row for row in bad if row["modality"] == modality]
        if len(sub_good) >= 40 and len(sub_bad) >= 40:
            threshold_scopes.append((modality, sub, sub_good, sub_bad))
    for scope_name, sub_scoped, sub_good, sub_bad in threshold_scopes:
        scope_rows = derive_thresholds(sub_scoped, sub_good, sub_bad, score_key, scope_name)
        thresholds.extend(scope_rows)
        name = "sot_thresholds.csv" if scope_name == "all" else "sot_thresholds_%s.csv" % scope_name
        if scope_rows:
            write_csv(out_dir / name, scope_rows, list(scope_rows[0].keys()))
        for row in scope_rows:
            print("  LINE[%s] %-18s thr=%.4g  above=%.3f below=%.3f gap=%.3f (n=%d/%d)"
                  % (scope_name, row["metric"], row["raw_threshold"], row["mean_score_above"],
                     row["mean_score_below"], row["gap"], row["n_above"], row["n_below"]))
            if math.isfinite(row["cover_threshold"]):
                print("       %-18s LINE=%.4g  good_TPR=%.3f  bad_FPR=%.4f  good_cov=%.1f%%  good_below_median=%.3f  %s"
                      % (row["metric"], row["cover_threshold"], row["cover_good_tpr"],
                         row["cover_bad_fpr"], 100.0 * row["cover_good_coverage"],
                         row["cover_good_below_median"],
                         "[FALLBACK 无候选同时满足]" if row["cover_fallback"] else ""))
            else:
                print("       %-18s 无交叉（该模态无此指标 / 样本不足）" % row["metric"])
            # 三条线（① 准入线 ② 推荐线 ③ 最低安全线），FALLBACK 明示
            for prefix, label in (("admit", "①准入线"), ("rec", "②推荐线"), ("safe", "③最低安全线")):
                thr = row["%s_threshold" % prefix]
                if not math.isfinite(thr):
                    print("       %-18s %s=无解" % (row["metric"], label))
                    continue
                print("       %-18s %s=%.4g  good_TPR=%.3f  bad_FPR=%.4f  good_cov=%.1f%%"
                      % (row["metric"], label, thr, row["%s_good_tpr" % prefix],
                         row["%s_bad_fpr" % prefix], 100.0 * row["%s_good_coverage" % prefix]))

    if discrimination:
        write_csv(out_dir / "sot_discrimination.csv", discrimination, list(discrimination[0].keys()))

    print("[SUMMARY] clicks=%d windows=%d (kind=%s: good=%d bad=%d) clips=%d"
          % (len(clicks), len(joined), cli.kind, len(good), len(bad), len(clip_rows)))
    print("[TAU] " + "  ".join("%s=%.4f" % (k, v) for k, v in sorted(taus.items())))
    for row in group_rows:
        print("  GROUP %-5s %-6s %-6s clips=%-3d windows=%-6d %s"
              % (row["group"], row["kind"], row["click_mode"], row["clips"], row["windows"],
                 " ".join("%s=%s" % (k, v) for k, v in row.items() if k.endswith("_p50") and "lock" in k or k.endswith("_ugr_p50"))))
    for row in discrimination:
        print("  %-18s auc_metric=%.3f auc_score=%.3f spearman=%+.3f good_med=%.4g bad_med=%.4g"
              % (row["metric"], row["auc_metric_vs_group"], row["auc_score_vs_group"],
                 row["spearman_metric_vs_score"], row["good_median"], row["bad_median"]))


if __name__ == "__main__":
    main()
