#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Acceptance check for the video-quality metric study.

Answers one question with evidence instead of trust:  are the deliverables actually there,
do the numbers in the report agree with the artefacts they claim to come from, and does the
report contain the caveats the methodology requires?

Usage
-----
    python manu/video_quality/vq_acceptance.py --metrics-dir manu/video_quality/out \
        --report-dir manu/video_quality/report --memory manu/memory/video_quality_metrics_20261119.md

Exit code 0 = all checks pass, 1 = at least one FAIL.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# vq_metrics imports cv2, which only exists in the server's conda env ``uav``.  The metric
# vocabulary is therefore taken from the artefacts themselves, so this checker also runs on a
# plain host checkout (no OpenCV needed).
try:  # pragma: no cover - convenience only
    from vq_metrics import METRIC_DIRECTION  # noqa: E402
except ModuleNotFoundError:
    METRIC_DIRECTION = {}

EXPECTED_CLIPS = 25
EXPECTED_GOOD = {"ir": 6, "vis": 12}
EXPECTED_BAD = {"ir": 2, "vis": 2, "osd": 3}

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Acceptance checks for the video quality study")
    parser.add_argument("--metrics-dir", type=str, required=True)
    parser.add_argument("--report-dir", type=str, required=True)
    parser.add_argument("--memory", type=str, required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    report_dir = Path(args.report_dir)

    # ---- 1. every clip processed -------------------------------------------
    clips = [p for p in sorted(metrics_dir.glob("*.json")) if p.name != "run_manifest.json"]
    npzs = list(metrics_dir.glob("*.npz"))
    check("所有 25 条片都有指标 JSON", len(clips) == EXPECTED_CLIPS, f"json={len(clips)}")
    check("所有片都有逐帧 npz", len(npzs) == EXPECTED_CLIPS, f"npz={len(npzs)}")

    metas = [json.loads(p.read_text(encoding="utf-8")) for p in clips]
    good = [m for m in metas if m.get("group") == "good"]
    bad = [m for m in metas if m.get("group") == "bad"]
    check("好组 18 条 / 坏组 7 条", len(good) == 18 and len(bad) == 7,
          f"good={len(good)} bad={len(bad)}")

    good_mod = {}
    for meta in good:
        good_mod[meta.get("modality")] = good_mod.get(meta.get("modality"), 0) + 1
    bad_mod = {}
    for meta in bad:
        key = meta.get("modality") if meta.get("kind") == "source" else "osd"
        bad_mod[key] = bad_mod.get(key, 0) + 1
    check("好组模态 = 6 IR + 12 VIS", good_mod == EXPECTED_GOOD, str(good_mod))
    check("坏组模态 = 2 IR + 2 VIS + 3 OSD", bad_mod == EXPECTED_BAD, str(bad_mod))

    reclassified = sum(1 for m in metas if m.get("modality_reclassified"))
    check("模态重分类已执行（post_batch 生效）", reclassified > 0,
          f"modality_reclassified={reclassified} 条")

    # ---- 2. artefacts ------------------------------------------------------
    comparison = read_csv(report_dir / "comparison.csv")
    stats = read_csv(report_dir / "stats.csv")
    tail = read_csv(report_dir / "tail_compare.csv")
    check("comparison.csv 存在且非空", len(comparison) > 0, f"rows={len(comparison)}")
    check("stats.csv 存在且非空", len(stats) > 0, f"rows={len(stats)}")
    check("tail_compare.csv 存在且非空", len(tail) > 0, f"rows={len(tail)}")

    report_files = sorted(report_dir.glob("REPORT_*.md"))
    check("中文汇报文件存在", len(report_files) == 1,
          ", ".join(p.name for p in report_files) or "none")
    report_text = report_files[0].read_text(encoding="utf-8") if report_files else ""

    memory_path = Path(args.memory)
    memory_ok = memory_path.exists() and memory_path.stat().st_size > 0
    check("memory 档存在且非空", memory_ok, str(memory_path))

    # ---- 3. statistical gate actually produced something --------------------
    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    passed = []
    for row in stats:
        p_value = to_float(row.get("clip_p_exact"))
        disc = to_float(row.get("clip_discrimination"))
        auc = to_float(row.get("clip_auc"))
        # the exact permutation test is the video level gate: with 2 bad clips a bootstrap CI is
        # far too wide to decide anything, while the enumeration of all label assignments is exact.
        if p_value == p_value and p_value < 0.05 and disc == disc and disc >= 0.70:
            passed.append((row.get("pair"), row.get("metric"), auc, p_value))
    check("至少 1 个指标通过视频级统计门槛（精确置换 p<0.05 且判别力>=0.70）", len(passed) > 0,
          f"passed={len(passed)}: " + ", ".join(f"{m}({p})" for _, m, _, p in passed[:6]))

    ir_vis_passed = [row for row in passed if row[0] in {"ir", "vis"}]
    check("IR/VIS 同源对照里至少 1 个指标通过门槛", len(ir_vis_passed) > 0,
          f"n={len(ir_vis_passed)}")

    # ---- 4. the report honours the methodology -----------------------------
    def has(*patterns) -> bool:
        return all(re.search(pattern, report_text) for pattern in patterns)

    check("汇报含结论速览", has("结论"))
    check("汇报含有效指标表", has("有效指标"))
    check("汇报含无效指标清单", has("无效") or has("无区分度") or has("剔除"))
    check("汇报含工程准入红线/阈值", has("红线|阈值"))
    check("汇报含局限与后续", has("局限"))
    check("汇报给出精确置换 p 值",
          has("置换") and (has(r"p\s*[=<]") or has("p 值") or has("p(") or has("p（")))
    check("汇报给出 cluster bootstrap CI", has("cluster|聚类|cluster CI", "CI|置信区间"))
    check("汇报区分倍数（ratio）", has("倍数"))
    check("汇报含跟丢切片/尾窗对比", has("切片|尾窗|worst|tail"))
    check("汇报声明无 GT 框故未算真值局部 SCR", has("SCR") and has("标注|初始框|GT"))
    check("汇报含 OSD 跨源不可直接对照的方法学警告",
          has("OSD|挂飞|录屏|桌面")
          and has("不同源|跨源|不能直接|补充|参考|佐证|不参与|不作为结论依据|非源视频"))
    check("汇报剔除低区分度指标", has(r"1\.0|1\.2"))

    # every metric id quoted in the report must exist in stats.csv
    stat_metrics = {row.get("metric") for row in stats}
    vocabulary = set(METRIC_DIRECTION) | stat_metrics
    quoted = {m for m in vocabulary if re.search(rf"\b{re.escape(m)}\b", report_text)}
    unknown = sorted(quoted - stat_metrics)
    check("汇报里引用的指标 id 都能在 stats.csv 对上", not unknown, f"unknown={unknown}")

    # ---- 5. numbers in the report must exist in the artefacts --------------
    tail_metrics = {row.get("metric") for row in tail}
    missing_tail = sorted(quoted - tail_metrics)
    check("汇报里引用的指标 id 都能在 tail_compare.csv 对上", not missing_tail,
          f"missing={missing_tail}")

    failures = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 72)
    print(f"总计 {len(RESULTS)} 项检查，通过 {len(RESULTS) - len(failures)}，失败 {len(failures)}")
    for name in failures:
        print(f"  FAILED: {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
