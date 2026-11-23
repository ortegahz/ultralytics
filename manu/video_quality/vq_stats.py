#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Statistics for the good/bad clip comparison - the single owner of every number in the report.

Why this module exists
----------------------
The bad group holds a handful of clips (a few infrared sources, one or two visible light sources and
three ground-station screen recordings), so the naive statistics are all misleading:

* a whole-clip P95 over a 44 000 frame clip is an extreme-value statistic and is **not** comparable
  with the P95 of a 340 frame clip -> every video level number is therefore the *typical second*:
  the median over the 1 second windows of the in-window quantile (length independent);
* treating the 1 second windows as independent samples is pseudo-replication (windows inside one
  clip are highly correlated) -> significance is judged by a **cluster bootstrap over clips** and
  the window level Mann-Whitney p is only kept as a descriptive number;
* the AUC direction must be unambiguous -> every AUC here is ``P(good > bad)`` on the **raw** metric
  values (0.5 = no discrimination, > 0.5 = the good group has the larger value).  The headline
  number is ``discrimination = max(auc, 1 - auc)`` plus an explicit ``bad_side`` column;
* a threshold mined from the same data it is evaluated on always flatters itself -> the red line is
  additionally validated leave-one-bad-clip-out.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_metrics import METRIC_DIRECTION, METRIC_LABEL_CN, METRIC_SOURCE, METRIC_STAT, metric_array  # noqa: E402

# pair key -> (title, good filter, bad filter).  ``kind`` separates the raw aircraft feeds from the
# ground-station screen recordings, ``modality`` separates infrared from visible light.
PAIRS = [
    ("ir", "红外 IR (源视频)", lambda r: r["kind"] == "source" and r["modality"] == "ir",
     lambda r: r["kind"] == "source" and r["modality"] == "ir"),
    ("vis", "可见光 VIS (源视频)", lambda r: r["kind"] == "source" and r["modality"] == "vis",
     lambda r: r["kind"] == "source" and r["modality"] == "vis"),
    ("osd", "挂飞记录 OSD (地面站录屏)", lambda r: r["kind"] == "source",
     lambda r: r["kind"] == "osd"),
]

PAIR_TITLE = {key: title for key, title, _, _ in PAIRS}


def auc_good_higher(good: np.ndarray, bad: np.ndarray) -> float:
    """P(good value > bad value) on the raw metric values; 0.5 = no discrimination."""
    good = np.asarray(good, dtype=float)
    bad = np.asarray(bad, dtype=float)
    good = good[~np.isnan(good)]
    bad = bad[~np.isnan(bad)]
    if good.size == 0 or bad.size == 0:
        return float("nan")
    wins = (good[:, None] > bad[None, :]).sum() + 0.5 * (good[:, None] == bad[None, :]).sum()
    return float(wins / (good.size * bad.size))


def discrimination(auc: float) -> float:
    """Direction free strength of a metric: 0.5 = none, 1.0 = perfect separation."""
    if not np.isfinite(auc):
        return float("nan")
    return float(max(auc, 1.0 - auc))


def auc_rank(good: np.ndarray, bad: np.ndarray) -> float:
    """P(good > bad) from the Mann-Whitney rank sum - O(n log n) instead of the O(n^2) matrix.

    The window level pools thousands of samples, so the pairwise comparison matrix used by
    :func:`auc_good_higher` is too slow there.  The two implementations agree exactly (the rank sum
    is the closed form of the same statistic), and ``vq_verify.py`` checks that they do.
    """
    good = np.asarray(good, dtype=float)
    bad = np.asarray(bad, dtype=float)
    good = good[~np.isnan(good)]
    bad = bad[~np.isnan(bad)]
    if good.size == 0 or bad.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([good, bad]))
    n_good, n_bad = good.size, bad.size
    u_good = ranks[:n_good].sum() - n_good * (n_good + 1) / 2.0
    return float(u_good / (n_good * n_bad))


def window_values(meta: dict, arrays: dict, metric: str) -> np.ndarray:
    """Per 1 second window value of a clip: the in-window quantile of the per-frame series."""
    values = metric_array(arrays, metric)
    if values is None or values.size == 0:
        return np.array([], dtype=float)
    window = max(1, int(round(meta.get("fps_used", 25.0) * meta.get("stride", 1))))
    count = values.size // window
    if count < 1:
        return np.array([], dtype=float)
    trimmed = values[: count * window].reshape(count, window)
    if metric in {"shift_frac", "shift_frac_max", "shift_rate", "rot_deg", "rot_rate", "illum_jump"}:
        trimmed = trimmed[:, 1:]
    if trimmed.size == 0:
        return np.array([], dtype=float)
    stat = METRIC_STAT[metric]
    if stat == "p05":
        return np.percentile(trimmed, 5, axis=1)
    if stat == "p95":
        return np.percentile(trimmed, 95, axis=1)
    if stat == "max":
        return trimmed.max(axis=1)
    return np.percentile(trimmed, 50, axis=1)


def clip_typical_value(meta: dict, arrays: dict, metric: str) -> float:
    """Video level quality number = median over the 1 second windows (the "typical second").

    Length independent, so a 343 frame good clip and a 50 000 frame bad clip are compared on the
    same footing; the whole-clip quantile stays available in the per-clip json for reference.
    """
    windows = window_values(meta, arrays, metric)
    if windows.size == 0:
        source = METRIC_SOURCE.get(metric, metric)
        summary = meta.get("summary", {}).get(source, {})
        return float(summary.get(METRIC_STAT[metric], float("nan")))
    return float(np.median(windows))


def exact_permutation_p(good: np.ndarray, bad: np.ndarray) -> tuple[float, int]:
    """Exact two sided permutation p-value of the AUC, enumerating every label assignment.

    The statistic is the raw AUC ``P(good > bad)``; 0.5 means "no discrimination".  A permutation is
    counted as extreme when it lands at least as far from 0.5 as the observed assignment, in *either*
    direction.  Counting only one direction would report p = 1.0 for a metric that separates the two
    groups perfectly but with the bad group on the high side (its AUC is exactly 0), which reads as
    "not significant" next to a discrimination of 1.0.
    """
    good = good[~np.isnan(good)]
    bad = bad[~np.isnan(bad)]
    values = np.concatenate([good, bad])
    n_total = good.size + bad.size
    if n_total == 0 or good.size == 0 or bad.size == 0:
        return float("nan"), 0
    observed = auc_good_higher(good, bad)
    distance = abs(observed - 0.5)
    extreme = 0
    total = 0
    for combo in combinations(range(n_total), bad.size):
        mask = np.zeros(n_total, dtype=bool)
        mask[list(combo)] = True
        total += 1
        if abs(auc_good_higher(values[~mask], values[mask]) - 0.5) >= distance - 1e-12:
            extreme += 1
    return float(extreme / total), total


def bootstrap_ci(good: np.ndarray, bad: np.ndarray, iterations: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI over clips (resampling clips, not windows)."""
    rng = np.random.default_rng(seed)
    good = good[~np.isnan(good)]
    bad = bad[~np.isnan(bad)]
    if good.size < 2 or bad.size < 2:
        return float("nan"), float("nan")
    scores = [auc_good_higher(good[rng.integers(0, good.size, good.size)],
                              bad[rng.integers(0, bad.size, bad.size)]) for _ in range(iterations)]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def cluster_bootstrap_ci(groups_good: list[np.ndarray], groups_bad: list[np.ndarray],
                         iterations: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Cluster bootstrap CI: resample clips and pool the windows of the resampled clips."""
    rng = np.random.default_rng(seed)
    if len(groups_good) < 2 or len(groups_bad) < 2:
        return float("nan"), float("nan")
    scores = []
    for _ in range(iterations):
        g = np.concatenate([groups_good[i] for i in rng.integers(0, len(groups_good), len(groups_good))])
        b = np.concatenate([groups_bad[i] for i in rng.integers(0, len(groups_bad), len(groups_bad))])
        scores.append(auc_rank(g, b))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def redline(good: np.ndarray, bad: np.ndarray) -> dict:
    """Mine the engineering red line and validate it leave-one-bad-clip-out.

    The red line sits between the good side's edge (P90 of the good clips) and the bad side's edge
    (P10 of the bad clips), exactly as the method note prescribes.  A video is rejected when its
    typical second falls on the bad side of the threshold.
    """
    good = np.asarray(good, dtype=float)
    bad = np.asarray(bad, dtype=float)
    good = good[~np.isnan(good)]
    bad = bad[~np.isnan(bad)]
    if good.size == 0 or bad.size == 0:
        return {}
    side = "high" if np.median(bad) >= np.median(good) else "low"
    if side == "high":
        good_edge, bad_edge = float(np.percentile(good, 90)), float(np.percentile(bad, 10))
    else:
        good_edge, bad_edge = float(np.percentile(good, 10)), float(np.percentile(bad, 90))
    threshold = 0.5 * (good_edge + bad_edge)

    def balanced(values: np.ndarray, other: np.ndarray, cut: float) -> float:
        """Balanced accuracy of 'values sits on the bad side of cut'."""
        if side == "high":
            tpr = float((values > cut).mean())
            tnr = float((other <= cut).mean())
        else:
            tpr = float((values < cut).mean())
            tnr = float((other >= cut).mean())
        return 0.5 * (tpr + tnr)

    balanced_acc = balanced(bad, good, threshold)
    separated = bool(good.max() < bad.min()) if side == "high" else bool(good.min() > bad.max())
    loco = []
    for index in range(bad.size):
        rest = np.delete(bad, index)
        if rest.size == 0:
            continue
        if side == "high":
            cut = 0.5 * (np.percentile(good, 90) + np.percentile(rest, 10))
        else:
            cut = 0.5 * (np.percentile(good, 10) + np.percentile(rest, 90))
        loco.append(balanced(bad, good, cut))
    return {
        "bad_side": side,
        "threshold": float(threshold),
        "good_edge": good_edge,
        "bad_edge": bad_edge,
        "balanced_acc": float(balanced_acc),
        "separated": separated,
        "loco_balanced_acc_min": float(min(loco)) if loco else float("nan"),
        "loco_balanced_acc_mean": float(np.mean(loco)) if loco else float("nan"),
        "loco_folds": len(loco),
    }


def load(metrics_dir: Path) -> list[dict]:
    clips = []
    for json_path in sorted(metrics_dir.glob("*.json")):
        if json_path.name == "run_manifest.json":
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        arrays = {}
        npz_path = json_path.with_suffix(".npz")
        if npz_path.exists():
            with np.load(npz_path) as handle:
                arrays = {key: handle[key] for key in handle.files}
        clips.append({"meta": data, "arrays": arrays, "name": data["file"]})
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation tests and bootstrap CIs for the comparison")
    parser.add_argument("--metrics-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True, help="Output directory for stats.json / stats.csv")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = load(metrics_dir)
    rows = []
    for pair, title, good_filter, bad_filter in PAIRS:
        good_clips = [c for c in clips if c["meta"].get("group") == "good" and good_filter(c["meta"])]
        bad_clips = [c for c in clips if c["meta"].get("group") == "bad" and bad_filter(c["meta"])]
        if not good_clips or not bad_clips:
            print(f"[SKIP] {pair}: good={len(good_clips)} bad={len(bad_clips)}")
            continue
        for metric, direction in METRIC_DIRECTION.items():
            if direction == 0:
                continue
            g = np.array([clip_typical_value(c["meta"], c["arrays"], metric) for c in good_clips], dtype=float)
            b = np.array([clip_typical_value(c["meta"], c["arrays"], metric) for c in bad_clips], dtype=float)
            p_value, permutations = exact_permutation_p(g, b)
            clip_low, clip_high = bootstrap_ci(g, b)
            gw = [window_values(c["meta"], c["arrays"], metric) for c in good_clips]
            bw = [window_values(c["meta"], c["arrays"], metric) for c in bad_clips]
            gw = [values for values in gw if values.size]
            bw = [values for values in bw if values.size]
            pooled_g = np.concatenate(gw) if gw else np.array([], dtype=float)
            pooled_b = np.concatenate(bw) if bw else np.array([], dtype=float)
            win_low, win_high = cluster_bootstrap_ci(gw, bw)
            win_auc = auc_rank(pooled_g, pooled_b)
            rows.append(
                {
                    "pair": pair,
                    "pair_title": title,
                    "metric": metric,
                    "label_cn": METRIC_LABEL_CN.get(metric, metric),
                    "stat": METRIC_STAT[metric],
                    "expected_direction": direction,
                    "n_good_clips": len(good_clips),
                    "n_bad_clips": len(bad_clips),
                    "clip_auc": auc_good_higher(g, b),
                    "clip_discrimination": discrimination(auc_good_higher(g, b)),
                    "clip_p_exact": p_value,
                    "clip_permutations": permutations,
                    "clip_auc_ci_low": clip_low,
                    "clip_auc_ci_high": clip_high,
                    "window_auc": win_auc,
                    "window_discrimination": discrimination(win_auc),
                    "window_auc_cluster_ci_low": win_low,
                    "window_auc_cluster_ci_high": win_high,
                    "good_median": float(np.nanmedian(g)),
                    "bad_median": float(np.nanmedian(b)),
                    **redline(g, b),
                }
            )
            print(
                f"[{pair:<4}] {metric:<22} clip_auc={rows[-1]['clip_auc']:.3f} "
                f"(disc {rows[-1]['clip_discrimination']:.3f}) p_exact={p_value:.4f} ({permutations} perms)  "
                f"window_auc={win_auc:.3f} clusterCI=[{win_low:.3f},{win_high:.3f}]"
            )

    if not rows:
        print("[WARN] no comparable pair found")
        return
    rows.sort(key=lambda r: (r["pair"], -r["clip_discrimination"]))
    (out_dir / "stats.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] stats written to {out_dir}/stats.json and stats.csv")


if __name__ == "__main__":
    sys.exit(main())
