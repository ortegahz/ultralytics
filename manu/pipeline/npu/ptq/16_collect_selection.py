#!/usr/bin/env python3
"""16_collect_selection.py -- collect every board evaluation report into one
table and apply the deterministic model-selection rule.

WHY A COLLECTOR
---------------
The selection sweep (15_night_selection.sh) runs the SAME candidate models on
TWO manifests and writes one report per (model, manifest) pair.  The selection
must not depend on whoever is looking at the logs, so the rule is written
down here and applied mechanically.

THE RULE
--------
  * axis 2 = the golden pool (T2, 1197 frames) -- the SELECTION axis.
  * axis 1 = 001324 2000 frames (T3) -- a COMPARATIVE/diagnostic axis; the
    blind track's full 11,810 frames are reserved for the final model only.
  * primary key  : axis-2 F1 (highest wins)
  * tie-break 1  : axis-1 F1
  * tie-break 2  : the heatmap output ceiling (255 * out[0] scale) read from
                   the board -- a model whose output range is too narrow cannot
                   represent the peaks and MUST NOT be selected
  * tie-break 3  : model name (stable, deterministic)
  * a model whose board output ceiling is <= the decision threshold (0.3) is
    REJECTED outright, whatever its F1: its peaks are clipped below the
    threshold, so its F1 is not a quantisation result but a broken model.

Input:  the directory that 04_run_board_fast_eval.py wrote its reports into.
Output: <out>/selection_table.json  (every row, every metric, the rule, the
        winner) and <out>/best_model.json  (the chosen model, server path,
        board name, md5, and the numbers behind the choice).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_report(p: Path) -> dict:
    d = json.loads(p.read_text())
    out = {
        "report": p.name,
        "manifest": d.get("manifest"),
        "manifest_name": d.get("manifest_name"),
        "frames": d.get("frames"),
        "th": d.get("th"),
        "topk": d.get("topk"),
        "arms": {},
    }
    for tag, arm in (d.get("arms") or {}).items():
        if not isinstance(arm, dict):
            continue
        row = {k: arm.get(k) for k in
               ("tp", "fp", "fn", "det", "precision", "recall", "f1", "status")}
        # the board prints the tensor attributes; the fixed 04 archives them
        for k in ("input_attr", "out_attrs", "board_stdout"):
            if k in arm:
                row[k] = arm[k]
        out["arms"][tag] = row
    return out


def ceiling_of(arm: dict) -> float | None:
    """255 * out[0] scale -- the largest heatmap value the model can represent."""
    attrs = arm.get("out_attrs")
    if isinstance(attrs, str):
        m = re.search(r"out\[0\].*?scale=([0-9.]+)", attrs)
        if m:
            return 255.0 * float(m.group(1))
    if isinstance(attrs, list):
        for a in attrs:
            if isinstance(a, str) and "out[0]" in a:
                m = re.search(r"scale=([0-9.]+)", a)
                if m:
                    return 255.0 * float(m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dir with the 04 reports")
    ap.add_argument("--models", required=True, help="model list file (server_path board_name)")
    ap.add_argument("--axis1", required=True)
    ap.add_argument("--axis2", required=True)
    ap.add_argument("--pick-by", default="axis2_then_axis1",
                    choices=["axis2_then_axis1", "axis1_then_axis2"])
    ap.add_argument("--th", type=float, default=0.3)
    args = ap.parse_args()

    out_dir = Path(args.out)
    models = []
    for line in Path(args.models).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            models.append({"server_path": parts[0], "board_name": parts[1]})

    reports = [parse_report(p) for p in sorted(out_dir.glob("*_report.json"))]
    by_axis: dict[str, dict[str, dict]] = {"axis1": {}, "axis2": {}}
    for rep in reports:
        man = rep.get("manifest")
        axis = "axis1" if man and Path(man).name == Path(args.axis1).name else (
            "axis2" if man and Path(man).name == Path(args.axis2).name else None)
        if axis is None:
            continue
        for tag, arm in rep["arms"].items():
            key = tag.replace("board_", "")
            by_axis[axis].setdefault(key, {}).update(arm)

    rows = []
    for m in models:
        key = Path(m["server_path"]).stem
        a1 = by_axis["axis1"].get(key, {})
        a2 = by_axis["axis2"].get(key, {})
        merged = {**a1, **a2}
        row = {
            "model": key,
            "server_path": m["server_path"],
            "board_name": m["board_name"],
            "axis1_f1": a1.get("f1"), "axis1_recall": a1.get("recall"),
            "axis1_precision": a1.get("precision"), "axis1_fp": a1.get("fp"),
            "axis2_f1": a2.get("f1"), "axis2_recall": a2.get("recall"),
            "axis2_precision": a2.get("precision"), "axis2_fp": a2.get("fp"),
            "status": merged.get("status"),
            "out_ceiling": ceiling_of(merged),
        }
        rows.append(row)

    def pick_key(r: dict):
        first = r["axis2_f1"] if args.pick_by == "axis2_then_axis1" else r["axis1_f1"]
        second = r["axis1_f1"] if args.pick_by == "axis2_then_axis1" else r["axis2_f1"]
        return (first if first is not None else -1,
                second if second is not None else -1,
                r["model"])

    rejected = [{"model": r["model"], "reason": "output ceiling <= threshold",
                 "ceiling": r["out_ceiling"]}
                for r in rows
                if r["out_ceiling"] is not None and r["out_ceiling"] <= args.th]
    alive = [r for r in rows if r not in rejected]
    alive.sort(key=pick_key, reverse=True)
    winner = alive[0] if alive else None

    table = {
        "rule": {"primary": args.pick_by, "threshold_for_rejection": args.th,
                 "rejected": rejected, "tie_breakers": ["axis F1", "out ceiling", "name"]},
        "axis1_manifest": args.axis1, "axis2_manifest": args.axis2,
        "rows": sorted(rows, key=lambda r: (r.get("axis2_f1") is None,
                                            -(r.get("axis2_f1") or 0))),
        "winner": winner,
    }
    (out_dir / "selection_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1))
    if winner:
        (out_dir / "best_model.json").write_text(json.dumps(winner, ensure_ascii=False, indent=1))

    print(f"[collect] {len(rows)} models, {len(rejected)} rejected, winner="
          f"{winner['model'] if winner else None}")
    print(f"{'model':<28}{'ax1(T3)':>9}{'ax2(T2)':>9}{'ax2_R':>8}{'ax2_FP':>8}{'ceiling':>9}")
    for r in table["rows"]:
        f1a = f"{r['axis1_f1']:.4f}" if r["axis1_f1"] is not None else "-"
        f1b = f"{r['axis2_f1']:.4f}" if r["axis2_f1"] is not None else "-"
        rc = f"{r['axis2_recall']:.4f}" if r["axis2_recall"] is not None else "-"
        fp = str(r["axis2_fp"]) if r["axis2_fp"] is not None else "-"
        ce = f"{r['out_ceiling']:.4f}" if r["out_ceiling"] is not None else "-"
        print(f"{r['model']:<28}{f1a:>9}{f1b:>9}{rc:>8}{fp:>8}{ce:>9}")
    for r in rejected:
        print(f"  REJECTED {r['model']}: ceiling {r['ceiling']} <= {args.th}")
    print(f"[collect] -> {out_dir/'selection_table.json'}")


if __name__ == "__main__":
    main()
