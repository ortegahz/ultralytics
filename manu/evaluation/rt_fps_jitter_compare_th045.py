#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compare the random-FPS simulation against the frozen th045 delivery baseline.

The frozen baseline is ``osd_sys_sota_trial0080_th045_20261022``:

* ``per_sequence_hyperparams.tsv`` -- one row per sequence with the delivered
  F1 / recall / precision / far, the group (pass / un-pass) and the global
  parameters that produced them;
* ``pass/*.mp4`` / ``un-pass/*.mp4`` -- the OSD videos, whose file names carry the
  very same per-sequence F1 (``VIDEO00002_19700101_001415_F1-98.218.mp4``).

This script reads the report written by ``rt_fps_jitter_search.py`` and lines the
per-sequence numbers up with the frozen ones, so that "did the delta-n aware
parameters cost us F1?" can be answered per sequence and not only in aggregate.

Usage::

    python manu/evaluation/rt_fps_jitter_compare_th045.py \
        --th045 "/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_th045_20261022" \
        --report /tmp/rt_final.json --out-md /tmp/rt_final.md --out-tsv /tmp/rt_final.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_th045(root: Path) -> dict:
    """Read the frozen per-sequence TSV and check it against the video file names."""
    tsv = root / "per_sequence_hyperparams.tsv"
    rows: list[dict] = []
    header: list[str] = []
    with tsv.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if not header:
                header = parts
                continue
            rows.append(dict(zip(header, parts)))

    # The file names carry the per-sequence F1 -- check they agree with the TSV.
    names: dict[str, float] = {}
    for group in ("pass", "un-pass"):
        folder = root / group
        if not folder.is_dir():
            continue
        for video in sorted(folder.glob("*.mp4")):
            stem = video.stem
            if "_F1-" not in stem:
                continue
            sequence, _, value = stem.rpartition("_F1-")
            names[sequence] = float(value)

    mismatches = []
    for row in rows:
        sequence = row["sequence"]
        if sequence in names and abs(names[sequence] - float(row["f1_pct"])) > 1e-6:
            mismatches.append((sequence, names[sequence], float(row["f1_pct"])))
    missing = sorted(set(row["sequence"] for row in rows) - set(names))
    extra = sorted(set(names) - set(row["sequence"] for row in rows))

    return {"rows": rows, "names": names, "mismatches": mismatches,
            "missing_in_videos": missing, "extra_in_videos": extra}


def pooled(rows: list[dict]) -> dict:
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    gt = sum(int(r["gt"]) for r in rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"f1_pct": 100.0 * f1, "recall_pct": 100.0 * recall,
            "precision_pct": 100.0 * precision, "tp": tp, "fp": fp, "fn": fn, "gt": gt}


def arm_table(report: dict, arm: str, scenario: str | None = None) -> dict[str, dict]:
    per_seq = report.get("per_sequence", {})
    block = per_seq.get(arm, {})
    if scenario is None:
        # the ceiling arm is a flat {sequence: row} mapping
        return block if isinstance(block, dict) else {}
    return block.get(scenario, {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the delta-n simulation with the frozen th045 baseline")
    parser.add_argument("--th045", required=True, help="the osd_sys_sota_trial0080_th045_20261022 folder")
    parser.add_argument("--report", required=True, help="the JSON written by rt_fps_jitter_search.py")
    parser.add_argument("--arm", default="3_jump_optimised", help="the arm to compare")
    parser.add_argument("--scenario", default="s1", help="the cadence scenario of the arm")
    parser.add_argument("--base-arm", default="1_perframe_ceiling",
                        help="the per-frame reference arm")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-tsv", default="")
    args = parser.parse_args()

    th045 = read_th045(Path(args.th045))
    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    official = {row["sequence"]: row for row in th045["rows"]}
    base = arm_table(report, args.base_arm)
    target = arm_table(report, args.arm, args.scenario)

    lines: list[str] = []
    tsv: list[str] = []

    lines.append("# th045 vs delta-n simulation")
    lines.append("")
    lines.append(f"- frozen baseline : `{args.th045}`")
    lines.append(f"- simulation report: `{args.report}`")
    lines.append(f"- compared arm    : `{args.arm}` / scenario `{args.scenario}`")
    lines.append(f"- reference arm   : `{args.base_arm}` (per frame, dN == 1)")
    lines.append("")
    lines.append("## The video file names vs the frozen TSV")
    lines.append("")
    if th045["mismatches"]:
        for sequence, name_value, tsv_value in th045["mismatches"]:
            lines.append(f"- MISMATCH `{sequence}`: file name {name_value} vs TSV {tsv_value}")
    else:
        lines.append(f"- all {len(th045['names'])} `*_F1-<value>.mp4` file names agree with the TSV, exactly.")
    if th045["missing_in_videos"]:
        lines.append(f"- in the TSV but without a video: {', '.join(th045['missing_in_videos'])}")
    if th045["extra_in_videos"]:
        lines.append(f"- a video without a TSV row: {', '.join(th045['extra_in_videos'])}")
    lines.append("")

    header = ["sequence", "group", "th045_f1", "ours_perframe_f1", "delta_base",
              "ours_target_f1", "delta_vs_th045", "delta_vs_base", "th045_far", "ours_far"]
    tsv.append("\t".join(header))
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    base_rows: list[dict] = []
    target_rows: list[dict] = []
    for sequence, row in official.items():
        off = float(row["f1_pct"])
        base_row = base.get(sequence)
        target_row = target.get(sequence)
        base_f1 = float(base_row["f1_pct"]) if base_row else float("nan")
        target_f1 = float(target_row["f1_pct"]) if target_row else float("nan")
        far_off = float(row["far"])
        far_tgt = float(target_row["far"]) if target_row else float("nan")
        lines.append(
            f"| {sequence} | {row['group']} | {off:.3f} | {base_f1:.3f} | "
            f"{base_f1 - off:+.3f} | {target_f1:.3f} | {target_f1 - off:+.3f} | "
            f"{target_f1 - base_f1:+.3f} | {far_off:.4f} | {far_tgt:.4f} |"
        )
        tsv.append("\t".join([
            sequence, row["group"], f"{off:.3f}", f"{base_f1:.3f}", f"{base_f1 - off:.3f}",
            f"{target_f1:.3f}", f"{target_f1 - off:.3f}", f"{target_f1 - base_f1:.3f}",
            f"{far_off:.4f}", f"{far_tgt:.4f}",
        ]))
        if base_row:
            base_rows.append({"tp": base_row["tp"], "fp": base_row["fp"], "fn": base_row["fn"],
                              "gt": base_row["gt"], "f1_pct": base_row["f1_pct"],
                              "far": base_row["far"]})
        if target_row:
            target_rows.append({"tp": target_row["tp"], "fp": target_row["fp"], "fn": target_row["fn"],
                                "gt": target_row["gt"], "f1_pct": target_row["f1_pct"],
                                "far": target_row["far"]})

    off_rows = [{"tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]),
                 "gt": int(r["gt"]), "f1_pct": float(r["f1_pct"]), "far": float(r["far"])}
                for r in th045["rows"]]
    off_pooled = pooled(off_rows)
    base_pooled = pooled(base_rows) if base_rows else {}
    target_pooled = pooled(target_rows) if target_rows else {}

    lines.append("")
    lines.append("## Pooled over the 24 sequences")
    lines.append("")
    lines.append("| set | F1 % | recall % | precision % | tp | fp | fn | gt |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(f"| th045 frozen (official) | {off_pooled['f1_pct']:.3f} | {off_pooled['recall_pct']:.3f} | "
                 f"{off_pooled['precision_pct']:.3f} | {off_pooled['tp']} | {off_pooled['fp']} | "
                 f"{off_pooled['fn']} | {off_pooled['gt']} |")
    if base_pooled:
        lines.append(f"| ours, per frame (base arm) | {base_pooled['f1_pct']:.3f} | {base_pooled['recall_pct']:.3f} | "
                     f"{base_pooled['precision_pct']:.3f} | {base_pooled['tp']} | {base_pooled['fp']} | "
                     f"{base_pooled['fn']} | {base_pooled['gt']} |")
    if target_pooled:
        lines.append(f"| ours, {args.arm}/{args.scenario} | {target_pooled['f1_pct']:.3f} | "
                     f"{target_pooled['recall_pct']:.3f} | {target_pooled['precision_pct']:.3f} | "
                     f"{target_pooled['tp']} | {target_pooled['fp']} | {target_pooled['fn']} | "
                     f"{target_pooled['gt']} |")

    markdown = "\n".join(lines) + "\n"
    print(markdown)
    if args.out_md:
        Path(args.out_md).write_text(markdown, encoding="utf-8")
        print(f"[out] {args.out_md}")
    if args.out_tsv:
        Path(args.out_tsv).write_text("\n".join(tsv) + "\n", encoding="utf-8")
        print(f"[out] {args.out_tsv}")


if __name__ == "__main__":
    main()
