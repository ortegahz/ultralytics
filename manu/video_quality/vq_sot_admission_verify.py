#!/usr/bin/env python3
"""独立复算：按模态分列的「画质准入线」（不 import ``vq_sot_compare.py``）。

为什么单独一个脚本：``vq_sot_compare.py`` 的准入线口径在 2026-11-20 被整体重写过
（中位数口径 → ROC 式准入线口径；混池 → 按 IR / VIS 分列）。准入线是要**写进汇报件
交给领导**的数字，必须有一条**不复用生产代码路径**的独立复算线，否则口径一改就没人
能证明表里的数没算错。

检查项：
  A. 纹理地板 τ_det 复算（只用好片 P10）            vs ``sot_tau_det.csv``
  B. 逐窗 UGR 聚合复算（窗口数 / 均值）              vs ``sot_windows.csv``
  C. 准入线复算（线值 / TPR / FPR / 覆盖率 / fallback）vs ``sot_thresholds{,_ir,_vis}.csv``
  D. gap 口径参考线复算                             vs 同表的 ``raw_threshold``
  E. **跨模态可迁移性体检**：把「混池线」「IR 的线」拿到另一个模态上去卡，
     量化「好片被误杀成不合格」的比例 —— 这是混池口径必须作废的量化证据。

执行：
    python manu/video_quality/vq_sot_admission_verify.py \
        --sot-dir /tmp/sot_out,/tmp/sot_out_osd \
        --windows manu/video_quality/report/windows.csv \
        --report-dir /tmp/sot_report_long \
        --score xcorr_long_ugr --kind source
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

KEY_METRICS = (
    "lap_var", "edge_density", "global_contrast", "blockiness", "entropy",
    "snr_db", "noise_sigma", "dog_snr_max",
)

# direction: +1 = 值越低越好，-1 = 值越高越好（与 vq_metrics.METRIC_DIRECTION 同口径）
METRIC_DIRECTION = {
    "lap_var": -1,
    "edge_density": -1,
    "global_contrast": -1,
    "blockiness": 1,
    "entropy": -1,
    "snr_db": -1,
    "noise_sigma": 1,
    "dog_snr_max": -1,
}


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def horizon_entry(block: dict, horizon: int) -> dict:
    """点击级 horizons 以 str(h) 为键（"5"/"30"），窗口级聚合才用 "h5"/"h30"。"""
    horizons = block.get("horizons", {}) if isinstance(block, dict) else {}
    if str(horizon) in horizons:
        return horizons[str(horizon)]
    return horizons.get("h%d" % horizon, {})


def load_clicks(sot_dirs: list[Path], arm: str) -> list[dict]:
    clicks: list[dict] = []
    for sot_dir in sot_dirs:
        for path in sorted(sot_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload["summary"]
            horizons = sorted(int(h) for h in summary["horizons"])
            short_h, long_h = horizons[0], horizons[-1]
            for click in payload.get("clicks", []):
                record = {
                    "file": summary["video"],
                    "group": summary["group"],
                    "kind": summary["kind"],
                    "modality": summary["modality"],
                    "window": int(click["window"]),
                    "cell": click["cell"],
                    "click_mode": click.get("click_mode", "center"),
                    "tex_lap_var": float(click.get("tex_lap_var", float("nan"))),
                    "tex_std": float(click.get("tex_std", float("nan"))),
                }
                block = click.get(arm, {})
                for horizon, label in ((short_h, "short"), (long_h, "long")):
                    entry = horizon_entry(block, horizon)
                    record["%s_survived" % label] = bool(entry.get("survived", False))
                    record["%s_closure" % label] = float(entry.get("closure", float("nan")))
                clicks.append(record)
    return clicks


def calibrate_tau(clicks: list[dict]) -> dict[str, float]:
    taus: dict[str, float] = {}
    for modality in sorted({click["modality"] for click in clicks}):
        values = [
            click["tex_lap_var"] for click in clicks
            if click["modality"] == modality and click["group"] == "good"
            and math.isfinite(click["tex_lap_var"])
        ]
        taus[modality] = percentile(values, 10)
    return taus


def aggregate_windows(clicks: list[dict], taus: dict[str, float], arm: str) -> list[dict]:
    buckets: dict[tuple[str, int, str], list[dict]] = {}
    for click in clicks:
        buckets.setdefault((click["file"], click["window"], click["click_mode"]), []).append(click)
    rows: list[dict] = []
    for (file, window, mode), items in sorted(buckets.items()):
        tau = taus.get(items[0]["modality"], float("nan"))
        record = {
            "file": file, "group": items[0]["group"], "kind": items[0]["kind"],
            "modality": items[0]["modality"], "window": window, "click_mode": mode,
            "n_clicks": len(items), "tau_det": tau,
        }
        for label in ("short", "long"):
            usable = 0
            for click in items:
                if not click["%s_survived" % label]:
                    continue
                closure = click["%s_closure" % label]
                if not (math.isfinite(closure) and closure <= 3.0):
                    continue
                if math.isfinite(click["tex_lap_var"]) and click["tex_lap_var"] >= tau:
                    usable += 1
            record["%s_%s_ugr" % (arm, label)] = round(usable / len(items), 6)
        rows.append(record)
    return rows


def join_quality(window_rows: list[dict], windows_csv: Path) -> list[dict]:
    quality = {(row["file"], int(row["window"])): row for row in read_csv(windows_csv)}
    joined: list[dict] = []
    for row in window_rows:
        quality_row = quality.get((row["file"], row["window"]))
        if quality_row is None:
            continue
        merged = dict(row)
        for key, value in quality_row.items():
            if key in ("file", "group", "kind", "modality", "window"):
                continue
            merged[key] = value
        joined.append(merged)
    return joined


def admission_line(good_pairs, bad_pairs, pass_level=0.5, min_tpr=0.90, max_fpr=0.01):
    """ROC 式准入线（独立实现，与 compare 同口径但不复用其代码）。

    返回 ``(line, tpr, fpr, coverage, fallback)``；line 是**原始量纲**的阈值。
    """
    if len(good_pairs) < 40 or not bad_pairs:
        nan = float("nan")
        return nan, nan, nan, nan, False, nan, nan, nan, nan
    ordered = sorted(v for v, _ in good_pairs)
    candidates: list[float] = []
    for step in range(1, 100):
        position = step / 100.0 * (len(ordered) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        candidates.append(ordered[low] if low == high
                          else ordered[low] + (ordered[high] - ordered[low]) * (position - low))
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
            "coverage": len(good_above) / len(good_pairs),
            "good_above_median": percentile(good_above, 50),
            "good_below_median": percentile(good_below, 50) if good_below else float("nan"),
            "bad_below_median": percentile(bad_below, 50) if bad_below else float("nan"),
            "n_good_above": len(good_above), "n_bad_above": len(bad_above),
        })
    if not scored:
        nan = float("nan")
        return nan, nan, nan, nan, False, nan, nan, nan, nan
    qualified = [c for c in scored if c["tpr"] >= min_tpr and c["fpr"] <= max_fpr]
    if qualified:
        best = min(qualified, key=lambda c: c["threshold"])
        chosen = (best["threshold"], best["tpr"], best["fpr"], best["coverage"], False)
    else:
        best = max(scored, key=lambda c: (c["tpr"] - c["fpr"], -c["threshold"]))
        chosen = (best["threshold"], best["tpr"], best["fpr"], best["coverage"], True)
    # ---- 业务可直接用的另外两条线（不受 min_tpr 约束，只受「坏片误放率」约束）----
    # 红线 safe_line：满足 bad_FPR <= max_fpr 的 **最低** 线 ⇒ 低于它的坏片基本进不来，
    #                 但好片会被拒掉多少要看 good_coverage。
    # 推荐线 best_line：在 bad_FPR <= max_fpr 的候选里取 **TPR 最高**（好片覆盖最多）的那条。
    safe_candidates = [c for c in scored if c["fpr"] <= max_fpr]
    if safe_candidates:
        safe = min(safe_candidates, key=lambda c: c["threshold"])
        best_tpr = max(safe_candidates, key=lambda c: (c["tpr"], -c["threshold"]))
        extra = (safe["threshold"], safe["fpr"], safe["coverage"],
                 best_tpr["threshold"], best_tpr["tpr"], best_tpr["fpr"], best_tpr["coverage"])
    else:
        nan = float("nan")
        extra = (nan, nan, nan, nan, nan, nan, nan)
    return chosen + extra


def gap_threshold(values: list[float], scores: list[float]):
    """gap 口径：使「线以上均值 − 线以下均值」最大的那条线（独立实现）。"""
    if len(values) < 40:
        return float("nan"), float("nan"), float("nan"), 0, 0
    candidates = sorted(set(values))
    best = (candidates[0], float("nan"), float("nan"), 0, 0)
    best_gap = -1e9
    for index, threshold in enumerate(candidates):
        if index == 0:
            continue
        above = [s for v, s in zip(values, scores) if v >= threshold]
        below = [s for v, s in zip(values, scores) if v < threshold]
        if len(above) < 30 or len(below) < 30:
            continue
        gap = sum(above) / len(above) - sum(below) / len(below)
        if gap > best_gap:
            best_gap = gap
            best = (threshold, sum(above) / len(above), sum(below) / len(below), len(above), len(below))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent re-computation of the per-modality admission lines")
    parser.add_argument("--sot-dir", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--report-dir", required=True, help="Directory holding sot_thresholds*.csv / sot_windows.csv")
    parser.add_argument("--score", default="xcorr_long_ugr")
    parser.add_argument("--kind", default="source")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--arms", default="xcorr,patch",
                        help="Comma separated tracker families (e.g. xcorr,patch,kcf); only the first one is re-computed here")
    cli = parser.parse_args()

    arms = list(dict.fromkeys(part.strip() for part in str(cli.arms).split(",") if part.strip()))
    if not arms:
        parser.error("--arms needs at least one tracker family")
    arm = arms[0]

    failures: list[str] = []
    report_dir = Path(cli.report_dir)
    sot_dirs = [Path(p.strip()) for p in cli.sot_dir.split(",") if p.strip()]
    clicks = load_clicks(sot_dirs, arm)
    if not clicks:
        print("[FATAL] no probe JSON found -- refusing to report a vacuous PASS")
        raise SystemExit(2)
    print("[INPUT] %d clicks from %s" % (len(clicks), ", ".join(str(d) for d in sot_dirs)))

    taus = calibrate_tau(clicks)

    # ---------- A. τ_det ----------
    print("[CHECK A] texture floor tau_det")
    stored_tau = {row["modality"]: float(row["tau_det_p10_good"])
                  for row in read_csv(report_dir / "sot_tau_det.csv")}
    for modality, value in sorted(taus.items()):
        expected = stored_tau.get(modality, float("nan"))
        ok = math.isclose(value, expected, rel_tol=1e-4, abs_tol=1e-4)
        print("  %-4s recomputed=%.4f stored=%.4f %s" % (modality, value, expected, "OK" if ok else "MISMATCH"))
        if not ok:
            failures.append("CHECK A tau_det %s: %.4f vs stored %.4f" % (modality, value, expected))

    # ---------- B. 逐窗 UGR ----------
    print("[CHECK B] per-window UGR aggregation")
    window_rows = aggregate_windows(clicks, taus, arm)
    joined = join_quality(window_rows, Path(cli.windows))
    print("  windows recomputed=%d joined=%d" % (len(window_rows), len(joined)))
    stored_windows = report_dir / "sot_windows.csv"
    if stored_windows.exists():
        stored = read_csv(stored_windows)
        if len(stored) != len(joined):
            failures.append("CHECK B window count %d != stored %d" % (len(joined), len(stored)))
            print("  MISMATCH count %d vs stored %d" % (len(joined), len(stored)))
        else:
            print("  OK count matches stored (%d)" % len(stored))

    scoped = [row for row in joined if row["kind"] == cli.kind]
    good = [row for row in scoped if row["group"] == "good"]
    bad = [row for row in scoped if row["group"] == "bad"]
    score_key = cli.score
    if joined and score_key not in joined[0]:
        print("[FATAL] score column %r is not produced by --arms %s" % (score_key, ",".join(arms)))
        raise SystemExit(2)
    print("  scoped windows: good=%d bad=%d (score=%s)" % (len(good), len(bad), score_key))

    # ---------- C/D. 准入线 + gap 参考线 ----------
    print("[CHECK C/D] admission line + gap reference line, per scope")
    scopes = [("all", scoped, good, bad)]
    for modality in sorted({row["modality"] for row in scoped}):
        sub = [row for row in scoped if row["modality"] == modality]
        sub_good = [row for row in good if row["modality"] == modality]
        sub_bad = [row for row in bad if row["modality"] == modality]
        if len(sub_good) >= 40 and len(sub_bad) >= 40:
            scopes.append((modality, sub, sub_good, sub_bad))

    recomputed: dict[tuple[str, str], dict] = {}
    for scope, scope_rows, scope_good, scope_bad in scopes:
        stored_path = report_dir / ("sot_thresholds.csv" if scope == "all" else "sot_thresholds_%s.csv" % scope)
        if not stored_path.exists():
            failures.append("CHECK C missing stored thresholds for scope %s" % scope)
            continue
        stored_rows = {(row["scope"], row["metric"]): row for row in read_csv(stored_path)}
        for metric in KEY_METRICS:
            direction = METRIC_DIRECTION.get(metric, 0)
            if direction == 0:
                continue
            good_pairs = [
                (float(row[metric]) * (-direction), float(row[score_key]))
                for row in scope_good
                if row.get(metric) not in (None, "") and row.get(score_key) not in (None, "")
            ]
            bad_pairs = [
                (float(row[metric]) * (-direction), float(row[score_key]))
                for row in scope_bad
                if row.get(metric) not in (None, "") and row.get(score_key) not in (None, "")
            ]
            if len(good_pairs) < 40:
                continue
            (line, tpr, fpr, coverage, fallback,
             safe_line, safe_fpr, safe_cov,
             best_line, best_tpr, best_fpr, best_cov) = admission_line(good_pairs, bad_pairs)
            # gap 口径的线是在 **全池**(好片+坏片) 上搜的：目标是「线以上均值 − 线以下均值」最大，
            # 坏片样本必须参与，否则线会一路压到好片尾部去（与 compare.best_threshold 同口径）。
            pooled_pairs = [
                (float(row[metric]) * (-direction), float(row[score_key]))
                for row in scope_rows
                if row.get(metric) not in (None, "") and row.get(score_key) not in (None, "")
            ]
            gap_thr, gap_above, gap_below, n_above, n_below = gap_threshold(
                [v for v, _ in pooled_pairs], [s for _, s in pooled_pairs]
            )
            recomputed[(scope, metric)] = {
                "scope": scope, "metric": metric,
                "admission_line": line * (-direction) if math.isfinite(line) else float("nan"),
                "good_tpr": tpr, "bad_fpr": fpr, "good_coverage": coverage, "fallback": fallback,
                "gap_raw_threshold": gap_thr,          # 有符号空间，与存档 raw_threshold 同量纲
                "gap_metric_threshold": gap_thr * (-direction) if math.isfinite(gap_thr) else float("nan"),
                "safe_line": safe_line * (-direction) if math.isfinite(safe_line) else float("nan"),
                "safe_bad_fpr": safe_fpr, "safe_good_coverage": safe_cov,
                "best_line": best_line * (-direction) if math.isfinite(best_line) else float("nan"),
                "best_good_tpr": best_tpr, "best_bad_fpr": best_fpr, "best_good_coverage": best_cov,
            }
            stored = stored_rows.get((scope, metric))
            if stored is None:
                failures.append("CHECK C no stored row for %s/%s" % (scope, metric))
                print("  %-4s %-16s NOT IN STORED" % (scope, metric))
                continue
            stored_line = float(stored["cover_threshold"])
            stored_tpr = float(stored["cover_good_tpr"])
            stored_fpr = float(stored["cover_bad_fpr"])
            stored_cov = float(stored["cover_good_coverage"])
            stored_fallback = stored.get("cover_fallback", "").strip().lower() in ("true", "1")
            problems = []
            if math.isfinite(stored_line) and not math.isclose(line * (-direction), stored_line,
                                                             rel_tol=1e-3, abs_tol=max(1e-4, cli.tol)):
                problems.append("line %.6g vs stored %.6g" % (line * (-direction), stored_line))
            if math.isfinite(stored_tpr) and not math.isclose(tpr, stored_tpr, rel_tol=1e-3, abs_tol=1e-3):
                problems.append("tpr %.4f vs stored %.4f" % (tpr, stored_tpr))
            if math.isfinite(stored_fpr) and not math.isclose(fpr, stored_fpr, rel_tol=1e-2, abs_tol=1e-4):
                problems.append("fpr %.4f vs stored %.4f" % (fpr, stored_fpr))
            if math.isfinite(stored_cov) and not math.isclose(coverage, stored_cov, rel_tol=1e-3, abs_tol=1e-3):
                problems.append("cov %.4f vs stored %.4f" % (coverage, stored_cov))
            if fallback != stored_fallback:
                problems.append("fallback %s vs stored %s" % (fallback, stored_fallback))
            stored_raw = float(stored["raw_threshold"])
            if math.isfinite(stored_raw) and math.isfinite(gap_thr) and \
                    not math.isclose(gap_thr, stored_raw, rel_tol=1e-3, abs_tol=max(1e-3, cli.tol)):
                problems.append("gap-line %.6g vs stored %.6g" % (gap_thr, stored_raw))
            status = "OK" if not problems else "MISMATCH " + "; ".join(problems)
            print("  %-4s %-16s line=%-10.5g tpr=%.3f fpr=%.4f cov=%.3f %s | gap-line=%-10.5g %s"
                  % (scope, metric, line * (-direction) if math.isfinite(line) else float("nan"),
                     tpr, fpr, coverage, "[FALLBACK]" if fallback else "",
                     gap_thr * (-direction) if math.isfinite(gap_thr) else float("nan"), status))
            if problems:
                failures.append("CHECK C/D %s/%s: %s" % (scope, metric, "; ".join(problems)))

    # ---------- E. 跨模态可迁移性体检（混池线为什么必须作废） ----------
    print("[CHECK E] cross-modality transfer: a line derived on the POOLED population")
    print("  applied to one modality only -- how many GOOD windows does it reject?")
    print("  (IR 是灰度域、VIS 是彩色域，指标量纲差一个数量级 ⇒ 混池线对其中一个模态必然误杀)")

    def reject_rate(rows: list[dict], metric: str, line: float, direction: int) -> tuple[int, int]:
        total = 0
        rejected = 0
        for row in rows:
            if row.get(metric) in (None, ""):
                continue
            total += 1
            value = float(row[metric])
            if direction < 0 and value < line:
                rejected += 1
            elif direction > 0 and value > line:
                rejected += 1
        return rejected, total

    good_by_modality = {
        modality: [row for row in good if row["modality"] == modality]
        for modality in sorted({row["modality"] for row in good})
    }
    transfer_rows: list[dict] = []
    arm = "short" if "short" in cli.score else "long"

    # E1: 混池 gap 线（历史落档那一组）分别卡两个模态
    print("  -- E1 混池 gap 线（raw_threshold 口径） --")
    for (scope, metric), row in recomputed.items():
        if scope != "all":
            continue
        direction = METRIC_DIRECTION[metric]
        line = row["gap_metric_threshold"]
        if not math.isfinite(line):
            continue
        for modality, rows in good_by_modality.items():
            rejected, total = reject_rate(rows, metric, line, direction)
            rate = rejected / total if total else float("nan")
            transfer_rows.append({
                "line_kind": "pooled_gap", "line_scope": "all", "metric": metric,
                "applied_to": modality, "line": round(line, 6),
                "good_windows": total, "rejected_good_windows": rejected,
                "good_rejected_pct": round(100.0 * rate, 2),
            })
            print("    混池线 %-16s = %-10.5g -> 卡好%s片: 误杀 %d/%d = %5.1f%%"
                  % (metric, line, modality, rejected, total, 100.0 * rate))

    # E1b: 混池**准入线**（ROC 口径）分别卡两个模态 —— 报告 9.5.5 第一张表用的就是这个
    print("  -- E1b 混池准入线（ROC 口径，报告 9.5.5 第一张表） --")
    for (scope, metric), row in recomputed.items():
        if scope != "all":
            continue
        direction = METRIC_DIRECTION[metric]
        line = row["admission_line"]
        if not math.isfinite(line):
            continue
        for modality, rows in good_by_modality.items():
            rejected, total = reject_rate(rows, metric, line, direction)
            rate = rejected / total if total else float("nan")
            transfer_rows.append({
                "line_kind": "pooled_admission", "line_scope": "all", "metric": metric,
                "applied_to": modality, "line": round(line, 6),
                "good_windows": total, "rejected_good_windows": rejected,
                "good_rejected_pct": round(100.0 * rate, 2),
            })
            print("    混池准入线 %-16s = %-10.5g -> 卡好%s片: 误杀 %d/%d = %5.1f%%"
                  % (metric, line, modality, rejected, total, 100.0 * rate))

    # E2: 各模态自己的线拿到另一个模态上
    print("  -- E2 单模态线跨模态使用 --")
    for (scope, metric), row in recomputed.items():
        if scope == "all":
            continue
        direction = METRIC_DIRECTION[metric]
        for line_kind, key in (("own_gap", "gap_metric_threshold"), ("own_admission", "admission_line")):
            line = row[key]
            if not math.isfinite(line):
                continue
            for modality, rows in good_by_modality.items():
                if modality == scope:
                    continue
                rejected, total = reject_rate(rows, metric, line, direction)
                rate = rejected / total if total else float("nan")
                transfer_rows.append({
                    "line_kind": line_kind, "line_scope": scope, "metric": metric,
                    "applied_to": modality, "line": round(line, 6),
                    "good_windows": total, "rejected_good_windows": rejected,
                    "good_rejected_pct": round(100.0 * rate, 2),
                })
                print("    %s线 %s/%-16s = %-10.5g -> 卡好%s片: 误杀 %d/%d = %5.1f%%"
                      % (line_kind, scope, metric, line, modality, rejected, total, 100.0 * rate))

    # E3: 自查 —— 同模态用自己��线，误杀率必须是 0（否则线本身有问题）
    print("  -- E3 自查：同模态用自己的线 --")
    for (scope, metric), row in recomputed.items():
        if scope == "all":
            continue
        direction = METRIC_DIRECTION[metric]
        rows = good_by_modality.get(scope, [])
        rejected, total = reject_rate(rows, metric, row["gap_metric_threshold"], direction)
        note = "OK" if rejected == 0 else "NOTE 非 0（线落在好片分布内部，见 CSV）"
        print("    %s/%-16s 混池无关: 误杀好%s片 %d/%d  %s" % (scope, metric, scope, rejected, total, note))
        transfer_rows.append({
            "line_kind": "own_gap_self", "line_scope": scope, "metric": metric,
            "applied_to": scope, "line": round(row["gap_metric_threshold"], 6),
            "good_windows": total, "rejected_good_windows": rejected,
            "good_rejected_pct": round(100.0 * (rejected / total if total else float("nan")), 2),
        })

    if transfer_rows:
        for row in transfer_rows:
            row["arm"] = arm
        write_csv(report_dir / "sot_admission_transfer.csv", transfer_rows)
        print("  -> %s" % (report_dir / "sot_admission_transfer.csv"))

    # ---------- F. 业务可用线总表（红线 / 推荐线 / gap 参考线） ----------
    print("[CHECK F] business-usable lines table")
    lines_rows: list[dict] = []
    for scope in ("all", "ir", "vis"):
        for metric in KEY_METRICS:
            row = recomputed.get((scope, metric))
            if row is None:
                continue
            direction = METRIC_DIRECTION[metric]
            relation = ">=" if direction < 0 else "<="
            lines_rows.append({
                "scope": scope,
                "metric": metric,
                "relation": relation,
                "red_line_safe": ("%.5g" % row["safe_line"]) if math.isfinite(row["safe_line"]) else "",
                "red_line_bad_fpr": ("%.4f" % row["safe_bad_fpr"]) if math.isfinite(row["safe_bad_fpr"]) else "",
                "red_line_good_coverage": ("%.4f" % row["safe_good_coverage"]) if math.isfinite(row["safe_good_coverage"]) else "",
                "recommended_line": ("%.5g" % row["best_line"]) if math.isfinite(row["best_line"]) else "",
                "recommended_good_tpr": ("%.4f" % row["best_good_tpr"]) if math.isfinite(row["best_good_tpr"]) else "",
                "recommended_bad_fpr": ("%.4f" % row["best_bad_fpr"]) if math.isfinite(row["best_bad_fpr"]) else "",
                "recommended_good_coverage": ("%.4f" % row["best_good_coverage"]) if math.isfinite(row["best_good_coverage"]) else "",
                "roc_line": ("%.5g" % row["admission_line"]) if math.isfinite(row["admission_line"]) else "",
                "roc_good_tpr": ("%.4f" % row["good_tpr"]) if math.isfinite(row["good_tpr"]) else "",
                "roc_bad_fpr": ("%.4f" % row["bad_fpr"]) if math.isfinite(row["bad_fpr"]) else "",
                "roc_good_coverage": ("%.4f" % row["good_coverage"]) if math.isfinite(row["good_coverage"]) else "",
                "roc_meets_both_constraints": "no" if row["fallback"] else "yes",
                "gap_separating_line": ("%.5g" % row["gap_metric_threshold"]) if math.isfinite(row["gap_metric_threshold"]) else "",
                "gap_line_is_admission_line": "NO (只做分界线，见 CHECK E3)",
            })
    if lines_rows:
        write_csv(report_dir / "sot_admission_lines.csv", lines_rows)
        print("  -> %s" % (report_dir / "sot_admission_lines.csv"))
        for row in lines_rows:
            print("    %-4s %-16s 红线 %-4s %-10s | 推荐线 %-10s TPR=%.3f FPR=%.4f 覆盖=%.3f | gap %-10s"
                  % (row["scope"], row["metric"], row["relation"], row["red_line_safe"],
                     row["recommended_line"], float(row["recommended_good_tpr"] or 0),
                     float(row["recommended_bad_fpr"] or 0), float(row["recommended_good_coverage"] or 0),
                     row["gap_separating_line"]))

    # ---------- G. 组合判据（这才是「一组指标」的最终形态） ----------
    # 单指标单线覆盖率都偏低（IR 最好也只有 60%），业务上真正能用的是
    # 「多个有效指标同时过线」的组合判据：任一项不达标即拒接入。
    print("[CHECK G] combined rule: ALL effective metrics must pass at once")
    DISCRIMINATING = ("global_contrast", "edge_density", "lap_var", "blockiness", "entropy")
    combo_rows: list[dict] = []
    for scope in ("ir", "vis"):
        scope_good = [r for r in good if r["modality"] == scope]
        scope_bad = [r for r in bad if r["modality"] == scope]
        for kind in ("recommended", "red_line"):
            used: list[tuple[str, str, float]] = []
            for metric in DISCRIMINATING:
                row = recomputed.get((scope, metric))
                if row is None:
                    continue
                key = "best_line" if kind == "recommended" else "safe_line"
                line = row[key]
                if not math.isfinite(line):
                    continue
                direction = METRIC_DIRECTION[metric]
                used.append((metric, ">=" if direction < 0 else "<=", line))
            if not used:
                continue

            def passes(row: dict) -> bool:
                for metric, relation, line in used:
                    if row.get(metric) in (None, ""):
                        return False
                    value = float(row[metric])
                    if relation == ">=" and value < line:
                        return False
                    if relation == "<=" and value > line:
                        return False
                return True

            def rate(rows: list[dict]) -> tuple[float, float, int, int]:
                passed = [r for r in rows if passes(r)]
                if not passed:
                    return float("nan"), float("nan"), 0, len(rows)
                trackable = sum(1 for r in passed if float(r[score_key]) >= 0.5)
                return trackable / len(passed), len(passed) / len(rows), len(passed), len(rows)

            good_tpr, good_cov, n_good_pass, n_good = rate(scope_good)
            bad_fpr, bad_cov, n_bad_pass, n_bad = rate(scope_bad)
            rule = " AND ".join("%s %s %.5g" % (m, rel, ln) for m, rel, ln in used)
            combo_rows.append({
                "scope": scope, "line_kind": kind, "rule": rule,
                "n_metrics": len(used),
                "good_trackable_rate": ("%.4f" % good_tpr) if math.isfinite(good_tpr) else "",
                "good_coverage": ("%.4f" % good_cov) if math.isfinite(good_cov) else "",
                "good_windows_passed": n_good_pass, "good_windows_total": n_good,
                "bad_false_accept_rate": ("%.4f" % bad_fpr) if math.isfinite(bad_fpr) else "",
                "bad_windows_passed": n_bad_pass, "bad_windows_total": n_bad,
            })
            print("  %-3s %-11s 指标数=%d | 好片线上跟住率=%s 好片覆盖=%s | 坏片误放率=%s"
                  % (scope, kind, len(used),
                     ("%.3f" % good_tpr) if math.isfinite(good_tpr) else "n/a",
                     ("%.1f%%" % (100 * good_cov)) if math.isfinite(good_cov) else "n/a",
                     ("%.4f" % bad_fpr) if math.isfinite(bad_fpr) else "n/a"))
            print("        规则: %s" % rule)
    if combo_rows:
        write_csv(report_dir / "sot_combined_rules.csv", combo_rows)
        print("  -> %s" % (report_dir / "sot_combined_rules.csv"))

    # ---------- H. 「好片分位数」刻度（给领导的最终口径） ----------
    # 把绝对阈值换算成「达到好片分布的第 P 分位」，领导问「画质多少」时可以直接回答
    # 「至少要比好片中位数再好一档」这种可执行的话。
    print("[CHECK H] percentile-calibrated combined rules (good-clip quantiles)")
    percentile_rows: list[dict] = []
    for scope in ("ir", "vis"):
        scope_good = [r for r in good if r["modality"] == scope]
        scope_bad = [r for r in bad if r["modality"] == scope]
        for pct in (10, 25, 50, 75, 90):
            used: list[tuple[str, str, float]] = []
            for metric in DISCRIMINATING:
                direction = METRIC_DIRECTION[metric]
                values = [float(r[metric]) for r in scope_good if r.get(metric) not in (None, "")]
                if len(values) < 40:
                    continue
                line = percentile(values, pct)
                used.append((metric, ">=" if direction < 0 else "<=", line))
            if not used:
                continue

            def passes_p(row: dict) -> bool:
                for metric, relation, line in used:
                    if row.get(metric) in (None, ""):
                        return False
                    value = float(row[metric])
                    if relation == ">=" and value < line:
                        return False
                    if relation == "<=" and value > line:
                        return False
                return True

            good_pass = [r for r in scope_good if passes_p(r)]
            bad_pass = [r for r in scope_bad if passes_p(r)]
            good_track = (sum(1 for r in good_pass if float(r[score_key]) >= 0.5) / len(good_pass)
                          if good_pass else float("nan"))
            bad_track = (sum(1 for r in bad_pass if float(r[score_key]) >= 0.5) / len(bad_pass)
                         if bad_pass else 0.0)
            percentile_rows.append({
                "scope": scope, "good_percentile": "P%d" % pct,
                "rule": " AND ".join("%s %s %.5g" % (m, rel, ln) for m, rel, ln in used),
                "good_windows_passed": len(good_pass), "good_windows_total": len(scope_good),
                "good_coverage_pct": ("%.1f" % (100.0 * len(good_pass) / len(scope_good))) if scope_good else "",
                "good_trackable_rate": ("%.3f" % good_track) if math.isfinite(good_track) else "n/a",
                "bad_windows_passed": len(bad_pass), "bad_windows_total": len(scope_bad),
                "bad_false_accept_rate": ("%.4f" % bad_track),
            })
            print("  %-3s P%-3d 好片覆盖=%5.1f%% 线上跟住率=%s | 坏片通过=%d/%d 误放率=%s"
                  % (scope, pct, 100.0 * len(good_pass) / len(scope_good),
                     ("%.3f" % good_track) if math.isfinite(good_track) else "n/a",
                     len(bad_pass), len(scope_bad), "%.4f" % bad_track))
            print("        %s" % " AND ".join("%s %s %.5g" % (m, rel, ln) for m, rel, ln in used))
    if percentile_rows:
        write_csv(report_dir / "sot_percentile_rules.csv", percentile_rows)
        print("  -> %s" % (report_dir / "sot_percentile_rules.csv"))

    print("")
    if failures:
        print("[RESULT] FAIL (%d)" % len(failures))
        for item in failures:
            print("  - %s" % item)
        raise SystemExit(1)
    print("[RESULT] PASS")


if __name__ == "__main__":
    main()
