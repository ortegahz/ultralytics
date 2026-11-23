#!/usr/bin/env python3
"""独立复算验收：不 import ``vq_sot_compare.py``，直接从原始 JSON / CSV 重算关键数字。

用途（对齐 §七/§八 的「独立复算验收」纪律）：
1. **逐片一致性**：本脚本重算的「跟住率(TGR-like)」「有纹理率(DET-like)」与 §271 的
   ``tgr_raw_p50`` / ``det_ratio_p50`` 应当高度一致 —— 两者是**两套独立实现**
   （§271 = 暴力 NCC 搜索；本轮 = 链式 SOT 跟踪），一致即互为佐证。
2. **合格线复算**：重新搜索阈值，检查与 ``report/sot_thresholds.csv`` 相同。
3. **窗口对齐**：逐秒窗口索引必须与 ``report/windows.csv`` 对得上（帧号 // fps）。
4. **分组口径**：好/坏/桌面录屏三组的中位数与「线两侧」均值独立重算。

执行：
    python manu/video_quality/vq_sot_verify.py --sot-dir /tmp/sot_out \\
        --windows manu/video_quality/report/windows.csv \\
        --grid-summary manu/video_quality/report/grid_summary.csv \\
        --thresholds manu/video_quality/report/sot_thresholds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数（与 compare 同一口径，但此处独立实现以便交叉验收）。"""
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


def horizon_entry(block: dict, tag: str) -> dict:
    """点击级 horizons 以 str(h) 为键（"5"/"30"），窗口级聚合才用 "h5"/"h30"。"""
    horizons = block.get("horizons", {}) if isinstance(block, dict) else {}
    if tag in horizons:
        return horizons[tag]
    plain = tag[1:] if tag.startswith("h") else tag
    return horizons.get(plain, {})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent re-computation check for the click-SOT probe")
    parser.add_argument("--sot-dir", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--grid-summary", required=True)
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--tolerance", type=float, default=0.12, help="Max |SOT - grid| per-clip gap")
    parser.add_argument("--kind", default="source", help="Clip kind used for the threshold re-derivation")
    cli = parser.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    grid = {row["file"]: row for row in read_csv(Path(cli.grid_summary))}
    windows_csv = read_csv(Path(cli.windows))
    window_index = {(row["file"], int(row["window"])) for row in windows_csv}
    frames_per_file: dict[str, int] = {}
    for row in windows_csv:
        frames_per_file[row["file"]] = max(frames_per_file.get(row["file"], 0), int(row["window"]) + 1)

    sot_dirs = [Path(part.strip()) for part in cli.sot_dir.split(",") if part.strip()]
    clips: list[dict] = []
    for sot_dir in sot_dirs:
        for path in sorted(sot_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            clips.append(payload)
    if not clips:
        print("[FATAL] no probe JSON found under %s -- refusing to report a vacuous PASS" % cli.sot_dir)
        raise SystemExit(2)
    print("[INPUT] %d clip JSONs from %s" % (len(clips), ", ".join(str(d) for d in sot_dirs)))

    # ---------- 0) 纹理地板 τ_det（只用好片 P10；§271 同一纪律） ----------
    print("[CHECK 0] texture floor tau_det (P10 over GOOD clips only)")
    taus: dict[str, float] = {}
    for modality in sorted({p["summary"]["modality"] for p in clips}):
        values = [
            float(c["tex_lap_var"])
            for p in clips
            if p["summary"]["modality"] == modality and p["summary"]["group"] == "good"
            for c in p.get("clicks", [])
            if math.isfinite(c.get("tex_lap_var", float("nan")))
        ]
        taus[modality] = percentile(values, 10)
        print("  %-4s tau=%.4f (n=%d good clicks)" % (modality, taus[modality], len(values)))
    if not taus:
        failures.append("CHECK 0 produced no texture floor (no modality seen in the loaded clips)")
    notes.append("tau_det " + " ".join("%s=%.4f" % (k, v) for k, v in sorted(taus.items())))

    # ---------- 1) 逐片：重算 TGR-like / DET-like，与 §271 对照 ----------
    print("[CHECK 1] per-clip SOT vs the §271 grid probe")
    print("  %-34s %-6s | %-22s | %-22s" % ("file", "kind", "SOT lock / textured", "grid tgr_raw / det"))
    deltas: list[float] = []
    for payload in clips:
        summary = payload["summary"]
        clicks = payload.get("clicks", [])
        center = [c for c in clicks if c.get("click_mode") == "center"]
        usable = [c for c in center if c.get("xcorr", {}).get("status") == "ok"]
        short_tag = "h%d" % min(summary["horizons"])
        locked = 0
        for click in usable:
            entry = horizon_entry(click.get("xcorr", {}), short_tag)
            closure = entry.get("closure", float("nan"))
            if entry.get("survived") and math.isfinite(closure) and closure <= 3.0:
                locked += 1
        lock_rate = locked / len(center) if center else float("nan")
        tau = taus.get(summary["modality"], float("nan"))
        textured = sum(
            1
            for c in center
            if math.isfinite(c.get("tex_lap_var", float("nan"))) and c["tex_lap_var"] >= tau
        )
        grid_row = grid.get(summary["video"], {})
        grid_tgr = float(grid_row.get("tgr_raw_p50", "nan")) if grid_row else float("nan")
        grid_det = float(grid_row.get("det_ratio_p50", "nan")) if grid_row else float("nan")
        print("  %-34s %-6s | %.3f / %.3f            | %.3f / %.3f"
              % (summary["video"][:34], summary["kind"], lock_rate, textured / len(center) if center else float("nan"),
                 grid_tgr, grid_det))
        if math.isfinite(grid_tgr) and center:
            deltas.append(abs(lock_rate - grid_tgr))
    if not deltas:
        failures.append("CHECK 1 produced no comparable clips (grid summary and SOT clips do not overlap)")
    if deltas:
        mean_delta = sum(deltas) / len(deltas)
        worst = max(deltas)
        notes.append("per-clip |SOT lock - grid tgr_raw| mean=%.3f worst=%.3f" % (mean_delta, worst))
        if mean_delta > cli.tolerance:
            failures.append("per-clip SOT lock vs grid tgr_raw mean gap %.3f > %.3f" % (mean_delta, cli.tolerance))
        else:
            print("  OK mean |gap| = %.3f (<= %.3f)" % (mean_delta, cli.tolerance))

    # ---------- 2) 窗口对齐 ----------
    print("[CHECK 2] window alignment against report/windows.csv")
    bad_align = 0
    for payload in clips:
        summary = payload["summary"]
        file = summary["video"]
        for row in payload.get("windows", []):
            if (file, int(row["window"])) not in window_index:
                bad_align += 1
    if bad_align:
        failures.append("%d SOT windows have no matching quality window" % bad_align)
    else:
        print("  OK every SOT window maps to a quality window")

    # ---------- 4) 分组中位数独立重算 ----------
    print("[CHECK 4] group medians (independent recomputation)")
    groups: dict[tuple[str, str], list[float]] = {}
    for payload in clips:
        summary = payload["summary"]
        tau = taus.get(summary["modality"], float("nan"))
        short_tag = "h%d" % min(summary["horizons"])
        long_tag = "h%d" % max(summary["horizons"])
        for click in payload.get("clicks", []):
            for tag, horizon in ((short_tag, "short"), (long_tag, "long")):
                entry = horizon_entry(click.get("xcorr", {}), tag)
                closure = entry.get("closure", float("nan"))
                ok = bool(entry.get("survived")) and math.isfinite(closure) and closure <= 3.0
                textured = math.isfinite(click.get("tex_lap_var", float("nan"))) and click["tex_lap_var"] >= tau
                lock_key = (summary["group"], "%s_lock" % horizon)
                ugr_key = (summary["group"], "%s_ugr" % horizon)
                groups.setdefault(lock_key, []).append(1.0 if ok else 0.0)
                groups.setdefault(ugr_key, []).append(1.0 if (ok and textured) else 0.0)
    if not groups:
        failures.append("CHECK 4 produced no group statistics")
    for key in sorted(groups):
        values = groups[key]
        print("  %-24s n=%-7d mean=%.4f" % (key[0] + "/" + key[1], len(values), statistics.fmean(values)))

    # ---------- 5) 合格线复算 ----------
    if cli.thresholds and Path(cli.thresholds).exists():
        print("[CHECK 5] re-derive the quality lines")
        quality = {(row["file"], int(row["window"])): row for row in windows_csv}
        rows = []
        for payload in clips:
            summary = payload["summary"]
            if summary.get("kind") != cli.kind:
                continue
            tau = taus.get(summary["modality"], float("nan"))
            long_tag = "h%d" % max(summary["horizons"])
            # 与生产口径一致：按 (file, window, click_mode) 分桶，center / corner 不混池
            per_bucket: dict[tuple[int, str], list[float]] = {}
            for click in payload.get("clicks", []):
                entry = horizon_entry(click.get("xcorr", {}), long_tag)
                closure = entry.get("closure", float("nan"))
                ok = bool(entry.get("survived")) and math.isfinite(closure) and closure <= 3.0
                textured = math.isfinite(click.get("tex_lap_var", float("nan"))) and click["tex_lap_var"] >= tau
                key = (int(click["window"]), str(click.get("click_mode", "center")))
                bucket = per_bucket.setdefault(key, [0.0, 0.0])
                bucket[1] += 1.0
                if ok and textured:
                    bucket[0] += 1.0
            for (window, _mode), (hit, total) in per_bucket.items():
                quality_row = quality.get((summary["video"], window))
                if quality_row is None:
                    continue
                rows.append((float(quality_row["global_contrast"]), hit / total))
        values = [v for v, _ in rows]
        scores = [s for _, s in rows]
        pairs = sorted(zip(values, scores))
        best = (float("nan"), 0.0)
        for index in range(1, len({v for v, _ in pairs})):
            threshold = sorted({v for v, _ in pairs})[index]
            above = [s for v, s in pairs if v >= threshold]
            below = [s for v, s in pairs if v < threshold]
            if len(above) < 30 or len(below) < 30:
                continue
            gap = statistics.fmean(above) - statistics.fmean(below)
            if gap > best[1]:
                best = (threshold, gap)
        if len(rows) < 40:
            failures.append("CHECK 5 has only %d comparable windows -- cannot re-derive a line" % len(rows))
        stored = [row for row in read_csv(Path(cli.thresholds)) if row["metric"] == "global_contrast"]
        stored_threshold = float(stored[0]["raw_threshold"]) if stored else float("nan")
        print("  global_contrast re-derived thr=%.4f gap=%.3f | stored thr=%.4f"
              % (best[0], best[1], stored_threshold))
        if math.isfinite(stored_threshold) and abs(best[0] - stored_threshold) > 0.05 * max(1.0, abs(stored_threshold)):
            failures.append("global_contrast threshold mismatch: %.4f vs stored %.4f" % (best[0], stored_threshold))
        else:
            print("  OK threshold reproduced")

    print()
    for note in notes:
        print("[NOTE] " + note)
    if failures:
        print("\n[FAIL] %d check(s) failed:" % len(failures))
        for item in failures:
            print("  - " + item)
        sys.exit(1)
    print("\n[PASS] all checks passed")


if __name__ == "__main__":
    main()
