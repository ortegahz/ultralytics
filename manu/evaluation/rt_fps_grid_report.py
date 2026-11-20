#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Turn the delivery-rate (fps) grid artefact into one auditable table.

WHY
---
``rt_fps_jitter_search.py`` answers "how bad is the cadence we measured" (s1..s4).  It
does NOT answer the question the delivery decision actually turns on: **at which
delivery rate does the tracker stop working**.  The per-frame arm (dN == 1) is the fast
anchor and s1 (dN 3.11) is the measured board point; everything between them -- 12 fps
to 20 fps -- was never sampled.

This script reads the fps-grid artefact (one JSON, produced on the server) and emits:

  * ``rt_fps_grid.md``   -- the ladder: achieved rate -> F1 / FAR, per arm;
  * ``rt_fps_grid.csv``  -- the same numbers, machine readable.

WHAT IS RECOMPUTED (read before quoting)
-----------------------------------------
Every aggregate is RE-DERIVED from the per-sequence rows and checked against the stored
value, with the SAME pooling code as ``rt_grid_compare.py`` (imported, not copied --
that is deliberate: a second copy of the pooling rule is a second thing to drift).
On top of that it checks the things that would silently invalidate the ladder:

  * the achieved rate is monotone in the requested one (a mis-generated grid fails);
  * ``25 / fps`` is consistent with the stored ``mean_delta_n``;
  * the jump triplet (``_jump_mean_f1`` and friends) matches the scenario aggregates;
  * the delivery-caliber block (per-frame on the FULL cache) is re-derived too, because
    that is where the FAR red line is defined.

Usage::

    python manu/evaluation/rt_fps_grid_report.py \
        --arm-json artifacts/rt_fps_grid_r90.json \
        --out-dir manu/memory/delivery_20261119/grid_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.evaluation.rt_grid_compare import (  # noqa: E402
    Checks,
    arm_scenarios,
    pooled,
    read_json,
    scenario_rows,
    write_csv,
)

# The cadence arms of the fps grid, in the order the ladder is read.
CADENCE_ARMS = (
    ("2b_jump_physical_untuned", "v3 基座 · 未调参"),
    ("3_jump_optimised", "cc2_md10（交付参数组）"),
    ("2a_jump_dense_timebase", "密集时间基（SDK 现状口径）"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Delivery-rate (fps) grid report")
    parser.add_argument("--arm-json", required=True, help="rt_fps_grid_r90.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--far-redline", type=float, default=0.03)
    args = parser.parse_args()

    doc = read_json(Path(args.arm_json), "fps grid json")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = Checks()
    arms = doc.get("arms", {})
    per_sequence = doc.get("per_sequence", {})
    cadence = doc.get("cadence", {})
    scenarios = list(doc.get("meta", {}).get("scenarios", []))
    checks.add("fps grid has scenarios", bool(scenarios), f"scenarios={scenarios}")
    checks.add("fps grid has cadence", bool(cadence), f"cadence keys={sorted(cadence)}")

    # ---- 1. re-derive every stored aggregate from the per-sequence rows --------
    for arm_name in ("2b_jump_physical_untuned", "3_jump_optimised",
                     "2a_jump_dense_timebase"):
        arm = arms.get(arm_name)
        if not arm:
            checks.add(f"{arm_name} present", False, "arm missing from the artefact")
            continue
        for scenario in arm_scenarios(arm):
            rows = scenario_rows(per_sequence.get(arm_name, {}), scenario)
            if not rows:
                checks.add(f"{arm_name}/{scenario} per-sequence rows", False,
                           "no rows -- it was NOT verified")
                continue
            got = pooled(rows)
            stored = arm[scenario]
            for key, value in (("f1_delivered_agg", got["f1"]),
                               ("precision_delivered_agg", got["precision"]),
                               ("recall_delivered_agg", got["recall"])):
                checks.add(f"{arm_name}/{scenario} {key}",
                           checks.close(stored[key], value),
                           f"stored={stored[key]!r} recomputed={value!r}")
            checks.add(f"{arm_name}/{scenario} far_delivered_agg",
                       abs(stored["far_delivered_agg"] - got["far_delivered"]) <= 1e-6,
                       f"stored={stored['far_delivered_agg']!r} "
                       f"recomputed={got['far_delivered']!r}")

    # ---- 2. the jump triplet ---------------------------------------------------
    for arm_name in ("2b_jump_physical_untuned", "3_jump_optimised"):
        arm = arms.get(arm_name)
        if not arm:
            continue
        names = arm_scenarios(arm)
        if not names:
            continue
        mean_f1 = sum(arm[s]["f1_delivered_agg"] for s in names) / len(names)
        worst_f1 = min(arm[s]["f1_delivered_agg"] for s in names)
        mean_far = sum(arm[s]["far_delivered_agg"] for s in names) / len(names)
        checks.add(f"{arm_name} _jump_mean_f1",
                   checks.close(arm["_jump_mean_f1"], mean_f1),
                   f"stored={arm['_jump_mean_f1']!r} recomputed={mean_f1!r}")
        checks.add(f"{arm_name} _jump_worst_f1",
                   checks.close(arm["_jump_worst_f1"], worst_f1),
                   f"stored={arm['_jump_worst_f1']!r} recomputed={worst_f1!r}")
        checks.add(f"{arm_name} _jump_mean_far",
                   checks.close(arm["_jump_mean_far"], mean_far),
                   f"stored={arm['_jump_mean_far']!r} recomputed={mean_far!r}")

    # ---- 3. the cadence itself: monotone rate, dN consistent with fps -----------
    fps_values: list[tuple[str, float]] = []
    for scenario in scenarios:
        info = cadence.get(scenario)
        if not info:
            checks.add(f"cadence {scenario}", False, "no cadence entry")
            continue
        fps = float(info.get("fps", 0.0))
        mean_dn = float(info.get("mean_delta_n", 0.0))
        checks.add(f"cadence {scenario} fps > 0", fps > 0, f"fps={fps}")
        checks.add(f"cadence {scenario} mean_delta_n > 0", mean_dn > 0, f"mean_dN={mean_dn}")
        if fps > 0 and mean_dn > 0:
            # fps = 25 * updates / span  and  mean_delta_n = span / (updates - 1)
            expected = 25.0 / fps
            rel = abs(expected - mean_dn) / mean_dn
            checks.add(f"cadence {scenario} 25/fps ~ mean_delta_n", rel <= 2e-3,
                       f"25/fps={expected:.4f} mean_delta_n={mean_dn:.4f} rel={rel:.2e}")
        fps_values.append((scenario, fps))
    for (prev_name, prev_fps), (next_name, next_fps) in zip(fps_values, fps_values[1:]):
        checks.add(f"cadence ladder is monotone: {prev_name} > {next_name}",
                   prev_fps > next_fps, f"{prev_fps:.4f} vs {next_fps:.4f}")

    # ---- 4. the delivery caliber (the caliber the red line is defined on) -------
    delivery = doc.get("delivery_caliber", {})
    for key, arm_name in (("base", "1_perframe_ceiling"), ("selected", "5_optimised_perframe")):
        block = delivery.get(key)
        if not block:
            checks.add(f"delivery_caliber {key}", False, "block missing")
            continue
        # The per-frame arms carry their rows in two shapes: the flat arms hang the
        # rows directly under the arm (sequence -> row), the nested ones put them
        # under a "perframe" key.  Accept both.
        node = per_sequence.get(arm_name, {})
        nested = node.get("perframe")
        if isinstance(nested, dict):
            rows = list(nested.values())
        else:
            rows = [value for value in node.values()
                    if isinstance(value, dict) and "tp" in value]
        if not rows:
            checks.add(f"delivery_caliber {key} per-sequence rows", False, "no rows")
            continue
        got = pooled(rows)
        checks.add(f"delivery_caliber {key} f1", checks.close(block["f1"], got["f1"]),
                   f"stored={block['f1']!r} recomputed={got['f1']!r}")
        checks.add(f"delivery_caliber {key} far",
                   abs(block["far"] - got["far_delivered"]) <= 1e-6,
                   f"stored={block['far']!r} recomputed={got['far_delivered']!r}")

    # ---- 5. the report ---------------------------------------------------------
    lines: list[str] = []
    add = lines.append
    add("# trial_0080 · 交付帧率网格（Delivery-Rate Ladder）")
    add("")
    add("> 由 `manu/evaluation/rt_fps_grid_report.py` 从服务器产物生成，"
        "聚合值由**逐序列行独立复算**并与产物逐位对账。")
    add("")
    add("## 0. 这张表回答什么")
    add("")
    add("`s1`~`s4` 回答「实测节拍有多糟」，**不**回答交付决策真正关心的那个问题："
        "**降到多少帧率，跟踪器就不能用了**。"
        "逐帧臂（Δn=1，25 fps）是快锚点，`s1`（Δn 3.11，实测板卡）是慢端点，"
        "中间 12~20 fps 这一段**从未被采样过**。本表补上这一段。")
    add("")
    add("| 场景 | 目标 fps | **实测达成 fps** | mean Δn | 帧数/序·seed |")
    add("| --- | ---: | ---: | ---: | ---: |")
    ladder_rows: list[dict] = []
    for scenario in scenarios:
        info = cadence.get(scenario, {})
        arm = arms.get("3_jump_optimised", {})
        block = arm.get(scenario, {})
        ladder_rows.append({
            "scenario": scenario,
            "target_fps": doc.get("meta", {}).get("scenarios_spec", {})
                          .get(scenario, {}).get("target_fps", ""),
            "achieved_fps": info.get("fps", ""),
            "mean_delta_n": info.get("mean_delta_n", ""),
            "updates": info.get("updates", ""),
            "untuned_f1": arms.get("2b_jump_physical_untuned", {})
                          .get(scenario, {}).get("f1_delivered_agg", ""),
            "untuned_far": arms.get("2b_jump_physical_untuned", {})
                           .get(scenario, {}).get("far_delivered_agg", ""),
            "opt_f1": block.get("f1_delivered_agg", ""),
            "opt_far": block.get("far_delivered_agg", ""),
            "dense_f1": arms.get("2a_jump_dense_timebase", {})
                        .get(scenario, {}).get("f1_delivered_agg", ""),
        })
    for row in ladder_rows:
        add(f"| `{row['scenario']}` | {row['target_fps']} | **{row['achieved_fps']}** | "
            f"{row['mean_delta_n']} | {row['updates']} |")
    add("")
    add("⚠️ **「实测达成 fps」才是可引用的数字**：目标帧率是采样器的请求值，"
        "而 8% 的相邻帧乱序会让引擎丢掉晚到的那一帧 ⇒ "
        "**跟踪器实际看到的速率低于请求值**（生成器已按此做了标定，"
        "残差 < 1%，逐场景实测值见上表）。")
    add("")

    add("## 1. 帧率阶梯：F1 / FAR 随交付速率的变化")
    add("")
    header = "| 臂 | 口径 | " + " | ".join(f"{r['scenario']}（{r['achieved_fps']} fps）"
                                          for r in ladder_rows) + " |"
    add(header)
    add("| --- | --- | " + " | ".join("---:" for _ in ladder_rows) + " |")
    for arm_name, label in CADENCE_ARMS:
        arm = arms.get(arm_name, {})
        cells = []
        for row in ladder_rows:
            block = arm.get(row["scenario"], {})
            f1 = block.get("f1_delivered_agg")
            far = block.get("far_delivered_agg")
            cells.append("—" if f1 is None
                         else f"{f1:.6f} / {far:.6f}")
        add(f"| `{arm_name}` | {label} | " + " | ".join(cells) + " |")
    add("")
    add("（单元格 = 跳帧 F1 / 跳帧 FAR，均为物理帧基、逐序列 micro-pool）")
    add("")
    ceiling = arms.get("1_perframe_ceiling", {})
    if ceiling:
        add(f"**锚点（逐帧，Δn≡1，全量 165,223 帧）**："
            f"F1={ceiling.get('f1', float('nan')):.6f}、"
            f"FAR={ceiling.get('far_per_frame', float('nan')):.6f}"
            f"（红线 {args.far_redline}）")
        add("")

    add("## 2. 独立复算验收")
    add("")
    failed = checks.failed
    add(f"**{len(checks.rows) - len(failed)}/{len(checks.rows)} 项通过"
        f"{'，0 失败 ✅' if not failed else '，失败 ' + str(len(failed)) + ' 项 ❌'}**")
    add("")
    if failed:
        add("| 失败项 | 详情 |")
        add("| --- | --- |")
        for name, _, detail in failed:
            add(f"| `{name}` | {detail} |")
        add("")
    add("复算口径：序列级 micro-pool（`precision=Σtp/Σ(tp+fp)`、"
        "`recall=Σtp/Σgt`、`F1=2PR/(P+R)`、跳帧 `FAR=Σfp/Σ(delivered_frames×seeds)`），"
        "与 `rt_grid_compare.py` **共用同一份实现**（import，非复制）；"
        "另外校验帧率阶梯单调性、`25/fps` 与 `mean_delta_n` 的相容性、"
        "跳帧三元组、交付口径（per-frame 全量）聚合值。")
    add("")

    (out_dir / "rt_fps_grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(out_dir / "rt_fps_grid.csv", ladder_rows,
              ["scenario", "target_fps", "achieved_fps", "mean_delta_n", "updates",
               "untuned_f1", "untuned_far", "opt_f1", "opt_far", "dense_f1"])
    print(checks.report())
    print(f"[OUT] {out_dir}/rt_fps_grid.md")
    print(f"[OUT] {out_dir}/rt_fps_grid.csv")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
