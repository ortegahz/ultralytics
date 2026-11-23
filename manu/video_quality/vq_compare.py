#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compare good vs bad clip groups on the extracted video quality metrics.

Every number produced here comes from :mod:`vq_stats`, the single owner of the statistical
definitions (AUC direction, the length independent video level statistic, the red line and the
significance tests), so the tables, the charts and the report cannot drift apart.

Produces:
- ``metrics_table.csv``  : clip level metric table (one row per clip, one column per metric)
- ``comparison.csv``     : per metric good/bad statistics, AUC and mined red line
- ``windows.csv``        : 1 second window level samples (the statistical power source)
- ``*.png``              : charts for the report
- ``comparison.json``    : machine readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_metrics import METRIC_DIRECTION, METRIC_LABEL_CN, METRIC_STAT  # noqa: E402
from vq_stats import (  # noqa: E402
    PAIRS,
    auc_good_higher,
    auc_rank,
    bootstrap_ci,
    clip_typical_value,
    cluster_bootstrap_ci,
    discrimination,
    exact_permutation_p,
    redline,
    window_values,
)


def load_clips(out_dir: Path) -> list[dict]:
    clips = []
    for json_path in sorted(out_dir.glob("*.json")):
        if json_path.name == "run_manifest.json":
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        npz_path = json_path.with_suffix(".npz")
        arrays = {}
        if npz_path.exists():
            with np.load(npz_path) as handle:
                arrays = {key: handle[key] for key in handle.files}
        clips.append({"meta": data, "arrays": arrays, "name": data["file"], "stem": json_path.stem})
    return clips


def window_samples(clip: dict) -> dict[str, list[float]]:
    """Per 1 second window value of every metric of one clip (the statistical power source)."""
    out: dict[str, list[float]] = {}
    for metric in METRIC_DIRECTION:
        values = window_values(clip["meta"], clip["arrays"], metric)
        if values.size:
            out[metric] = [float(value) for value in values]
    return out


def build_tables(clips: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    clip_rows = []
    window_rows = []
    for clip in clips:
        meta = clip["meta"]
        arrays = clip["arrays"]
        row = {
            "file": clip["name"],
            "group": meta.get("group", "?"),
            "kind": meta.get("kind", "?"),
            "modality": meta.get("modality", "?"),
            "frames": meta.get("frames_decoded", 0),
            "fps": meta.get("fps_used", 0),
            "width": meta["info"]["width"],
            "height": meta["info"]["height"],
            "crop": f'{meta["content_size"][0]}x{meta["content_size"][1]}',
        }
        for metric in METRIC_DIRECTION:
            row[metric] = clip_typical_value(meta, arrays, metric)
        clip_rows.append(row)

        samples = window_samples(clip)
        for index in range(max((len(v) for v in samples.values()), default=0)):
            window_rows.append(
                {
                    "file": clip["name"],
                    "group": meta.get("group", "?"),
                    "kind": meta.get("kind", "?"),
                    "modality": meta.get("modality", "?"),
                    "window": index,
                    **{metric: (samples[metric][index] if index < len(samples.get(metric, ())) else float("nan"))
                       for metric in METRIC_DIRECTION},
                }
            )
    return clip_rows, window_rows


def compare(clip_rows: list[dict], window_rows: list[dict], good_filter, bad_filter, pair: str) -> list[dict]:
    rows = []
    good_clips = [r for r in clip_rows if r["group"] == "good" and good_filter(r)]
    bad_clips = [r for r in clip_rows if r["group"] == "bad" and bad_filter(r)]
    good_win = [r for r in window_rows if r["group"] == "good" and good_filter(r)]
    bad_win = [r for r in window_rows if r["group"] == "bad" and bad_filter(r)]
    if not good_clips or not bad_clips:
        return rows

    def by_clip(rows_: list[dict], metric: str) -> list[np.ndarray]:
        """Window values grouped per clip - the clusters the bootstrap resamples."""
        groups: dict[str, list[float]] = {}
        for row in rows_:
            groups.setdefault(row["file"], []).append(row[metric])
        return [np.asarray(values, dtype=float) for values in groups.values()]

    for metric in METRIC_DIRECTION:
        g = np.array([r[metric] for r in good_clips], dtype=float)
        b = np.array([r[metric] for r in bad_clips], dtype=float)
        gw = np.array([r[metric] for r in good_win], dtype=float)
        bw = np.array([r[metric] for r in bad_win], dtype=float)
        clip_auc = auc_good_higher(g, b)
        win_auc = auc_rank(gw, bw)
        p_value, permutations = exact_permutation_p(g, b)
        clip_low, clip_high = bootstrap_ci(g, b)
        win_low, win_high = cluster_bootstrap_ci(by_clip(good_win, metric), by_clip(bad_win, metric))
        window_p = float("nan")
        if gw.size > 5 and bw.size > 5:
            window_p = float(
                mannwhitneyu(gw[~np.isnan(gw)], bw[~np.isnan(bw)], alternative="two-sided").pvalue
            )
        g_med = float(np.nanmedian(g))
        b_med = float(np.nanmedian(b))
        ratio = max(abs(g_med), abs(b_med)) / (abs(min(g_med, b_med)) + 1e-9)
        separation = "bad_high" if b_med >= g_med else "bad_low"
        rows.append(
            {
                "pair": pair,
                "metric": metric,
                "label_cn": METRIC_LABEL_CN.get(metric, metric),
                "stat": METRIC_STAT[metric],
                "expected_direction": METRIC_DIRECTION[metric],
                "good_median": g_med,
                "bad_median": b_med,
                "ratio": ratio,
                "separation": separation,
                "good_p10": float(np.nanpercentile(g, 10)),
                "good_p90": float(np.nanpercentile(g, 90)),
                "bad_p10": float(np.nanpercentile(b, 10)),
                "bad_p90": float(np.nanpercentile(b, 90)),
                "good_min": float(np.nanmin(g)),
                "good_max": float(np.nanmax(g)),
                "bad_min": float(np.nanmin(b)),
                "bad_max": float(np.nanmax(b)),
                "clip_auc": clip_auc,
                "clip_discrimination": discrimination(clip_auc),
                "clip_p_exact": p_value,
                "clip_permutations": permutations,
                "clip_auc_ci_low": clip_low,
                "clip_auc_ci_high": clip_high,
                "window_auc": win_auc,
                "window_discrimination": discrimination(win_auc),
                "window_auc_cluster_ci_low": win_low,
                "window_auc_cluster_ci_high": win_high,
                "window_p_mw": window_p,
                "n_good_clips": len(good_clips),
                "n_bad_clips": len(bad_clips),
                "n_good_windows": int(gw.size),
                "n_bad_windows": int(bw.size),
                **redline(g, b),
            }
        )
    rows.sort(key=lambda r: (-r["clip_discrimination"], -r["window_discrimination"]))
    return rows


def make_charts(comparison: list[dict], clip_rows: list[dict], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    charts = []
    for pair, title, good_filter, bad_filter in PAIRS:
        rows = [r for r in comparison if r["pair"] == pair]
        if not rows:
            continue
        rows = rows[: min(6, len(rows))]
        figure, axes = plt.subplots(2, 3, figsize=(16, 9))
        for axis, row in zip(axes.flat, rows):
            good = [r[row["metric"]] for r in clip_rows if r["group"] == "good" and good_filter(r)]
            bad = [r[row["metric"]] for r in clip_rows if r["group"] == "bad" and bad_filter(r)]
            good = [v for v in good if v == v]
            bad = [v for v in bad if v == v]
            axis.scatter([0] * len(good), good, color="#2E7D32", s=60, alpha=0.85, label="好视频")
            axis.scatter([1] * len(bad), bad, color="#C62828", s=60, alpha=0.85, label="坏视频")
            if good:
                axis.scatter([0], [float(np.median(good))], color="#1B5E20", marker="_", s=900)
            if bad:
                axis.scatter([1], [float(np.median(bad))], color="#B71C1C", marker="_", s=900)
            axis.set_xticks([0, 1])
            axis.set_xticklabels(["好", "坏"])
            axis.set_title(
                f'{row["label_cn"]} ({row["stat"]})\n视频级判别力={row["clip_discrimination"]:.2f}  '
                f'窗口级判别力={row["window_discrimination"]:.2f}',
                fontsize=11,
            )
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(rows):]:
            axis.axis("off")
        figure.suptitle(f"好/坏视频组质量指标区分度 — {title}", fontsize=15)
        figure.tight_layout()
        path = out_dir / f"chart_metrics_{pair}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        charts.append(str(path))

    # Discrimination overview for every metric (direction free: 0.5 = no discrimination).
    for pair, title, _, _ in PAIRS:
        rows = [r for r in comparison if r["pair"] == pair]
        if not rows:
            continue
        rows = rows[::-1]
        figure, axis = plt.subplots(figsize=(11, 0.42 * len(rows) + 2))
        names = [r["label_cn"] for r in rows]
        clip_disc = [r["clip_discrimination"] for r in rows]
        window_disc = [r["window_discrimination"] for r in rows]
        positions = np.arange(len(rows))
        axis.barh(positions - 0.2, clip_disc, height=0.38, color="#1565C0", label="视频级判别力")
        axis.barh(positions + 0.2, window_disc, height=0.38, color="#EF6C00", label="窗口级判别力 (1s)")
        axis.axvline(0.5, color="grey", linestyle="--", linewidth=1)
        axis.axvline(0.7, color="#C62828", linestyle=":", linewidth=1)
        axis.set_yticks(positions)
        axis.set_yticklabels(names, fontsize=9)
        axis.set_xlim(0.45, 1.02)
        axis.set_xlabel("判别力 (0.5 = 无区分度, 红线 0.7 = 有效)")
        axis.set_title(f"各质量指标区分度排序 — {title}", fontsize=13)
        axis.legend(loc="lower right")
        axis.grid(alpha=0.25, axis="x")
        figure.tight_layout()
        path = out_dir / f"chart_auc_{pair}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        charts.append(str(path))
    return charts


def make_timeseries_charts(comparison: list[dict], window_rows: list[dict], clip_rows: list[dict], out_dir: Path) -> list[str]:
    """Overlay the per-1-second-window series of every clip for the top discriminating metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    charts = []
    for pair, title, good_filter, bad_filter in PAIRS:
        rows = [r for r in comparison if r["pair"] == pair and classify_window(r) == "effective"]
        if not rows:
            continue
        rows = rows[:3]
        good_rows = [r for r in window_rows if r["group"] == "good" and good_filter(r)]
        bad_rows = [r for r in window_rows if r["group"] == "bad" and bad_filter(r)]
        if not good_rows or not bad_rows:
            continue
        figure, axes = plt.subplots(len(rows), 1, figsize=(15, 3.2 * len(rows)), squeeze=False)
        for axis, row in zip(axes[:, 0], rows):
            metric = row["metric"]
            for source, colour, label in ((good_rows, "#2E7D32", "好"), (bad_rows, "#C62828", "坏")):
                per_clip: dict[str, list[float]] = {}
                for item in source:
                    per_clip.setdefault(item["file"], []).append(float(item[metric]))
                for name, series in per_clip.items():
                    axis.plot(np.arange(len(series)), series, color=colour, alpha=0.45, linewidth=0.9, label=label)
            axis.set_title(
                f'{row["label_cn"]} ({row["stat"]})  窗口级判别力={row["window_discrimination"]:.3f}   '
                f'红/绿 = 坏/好视频逐秒曲线',
                fontsize=11,
            )
            axis.set_xlabel("1 秒窗口序号")
            axis.grid(alpha=0.25)
        axes[0, 0].legend(loc="upper right", fontsize=9)
        figure.suptitle(f"逐秒质量指标时间线 — {title}", fontsize=14)
        figure.tight_layout()
        path = out_dir / f"chart_timeseries_{pair}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        charts.append(str(path))
    return charts


def classify_window(row: dict) -> str:
    """effective / weak / none, from the direction free discrimination and the cluster CI.

    The cluster bootstrap resamples clips, so the CI respects the correlation between the windows
    of one clip; the window level Mann-Whitney p is pseudo-replicated and is not used here.
    """
    discrimination = row["window_discrimination"]
    low = row.get("window_auc_cluster_ci_low", float("nan"))
    high = row.get("window_auc_cluster_ci_high", float("nan"))
    significant = (low == low and high == high) and (low > 0.5 or high < 0.5)
    if discrimination >= 0.70 and significant:
        return "effective"
    if discrimination >= 0.55:
        return "weak"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare good vs bad clip quality metrics")
    parser.add_argument("--metrics-dir", type=str, required=True)
    parser.add_argument("--report-dir", type=str, required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    clips = load_clips(metrics_dir)
    clip_rows, window_rows = build_tables(clips)
    comparison = []
    for pair, _, good_filter, bad_filter in PAIRS:
        comparison.extend(compare(clip_rows, window_rows, good_filter, bad_filter, pair))

    import csv

    with (report_dir / "metrics_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clip_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clip_rows)
    with (report_dir / "windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(window_rows[0].keys()))
        writer.writeheader()
        writer.writerows(window_rows)
    if comparison:
        with (report_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
            writer.writeheader()
            writer.writerows(comparison)
    else:
        print("[WARN] no good/bad pair could be built - comparison.csv not written")

    charts = make_charts(comparison, clip_rows, report_dir)
    charts += make_timeseries_charts(comparison, window_rows, clip_rows, report_dir)
    payload = {
        "n_clips": len(clip_rows),
        "n_windows": len(window_rows),
        "comparison": comparison,
        "charts": charts,
        "clip_table": clip_rows,
    }
    (report_dir / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] comparison written to {report_dir} ({len(clip_rows)} clips, {len(window_rows)} windows)")


if __name__ == "__main__":
    sys.exit(main())
