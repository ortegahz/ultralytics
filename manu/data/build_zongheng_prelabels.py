#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Turn the RC2 per-frame detection records into the web annotator's pre-label layout.

Why this exists
---------------
The 「纵横AI数据采集」 domain has no labels, so the annotator's own pre-annotation worker
(``manu/annotate/worker_infer.py``) would have to re-run the whole multi-scale inference over the
58 videos -- a second GPU pass over data the RC2 OSD run has already streamed.

The RC2 OSD run already produced exactly the numbers an annotator needs, per frame: the detection
centre, the component size and the score. This script converts those records into the annotator's
native pre-label layout, so the reviewer can open the browser and start correcting proposals
immediately instead of waiting for another inference pass.

Two layouts are recognised by the annotator (``manu/annotate/labels.py``): ``<root>/<seq_id>/<frame>.txt``
and the flat ``<root>/<seq_id>__<frame>.txt``. This script writes the **flat** one, because that is
the delivery / evaluation layout (``manu/annotate/export_labels.py``) and it avoids one directory
per sequence.

Each frame also gets a ``scores.json`` sidecar next to the labels, holding the peak score of every
proposal. The annotator uses it to filter a noisy proposal set; without it, imported candidates are
displayed unconditionally.

The proposals are the RC2 **fused multi-scale detections at the delivery working point**, i.e. the
same points the reviewer sees in the OSD video. They are *proposals*, not truth: the human edits
them. A proposal that lands on nothing should simply be deleted in the UI.

Usage
-----
    python manu/data/build_zongheng_prelabels.py \
        --frames-dir /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_20261109/frames \
        --out /mnt/data/siping/datasets/manu/纵横AI数据采集_preannot_rc2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.annotate.labels import format_yolo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RC2 per-frame records into annotator pre-labels")
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory of <sequence>.jsonl per-frame detection records written by the RC2 OSD renderer",
    )
    parser.add_argument("--out", required=True, help="Output directory for the pre-labels")
    parser.add_argument("--native-width", type=int, default=640)
    parser.add_argument("--native-height", type=int, default=512)
    parser.add_argument("--min-visible", type=float, default=0.10, help="Minimum visible fraction of a proposal")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Drop proposals below this score (0 keeps every recorded detection)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        raise SystemExit(f"not a directory: {frames_dir}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources = sorted(frames_dir.glob("*.jsonl"))
    if not sources:
        raise SystemExit(f"no per-frame records under {frames_dir}")

    total_files = 0
    total_proposals = 0
    for source in sources:
        sequence = source.stem
        scores: dict[str, list[float]] = {}
        written = 0
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                detections = [
                    item for item in (record.get("detections") or []) if float(item["score"]) >= args.min_score
                ]
                boxes = [
                    {
                        "cls": 0,
                        "x1": float(item["cx"]) - float(item["w"]) / 2.0,
                        "y1": float(item["cy"]) - float(item["h"]) / 2.0,
                        "x2": float(item["cx"]) + float(item["w"]) / 2.0,
                        "y2": float(item["cy"]) + float(item["h"]) / 2.0,
                    }
                    for item in detections
                ]
                payload = format_yolo(boxes, args.native_width, args.native_height, args.min_visible)
                if not payload:
                    continue
                index = int(record["frame"])
                (out / f"{sequence}__{index:06d}.txt").write_text(payload, encoding="utf-8")
                scores[f"{index:06d}"] = [round(float(item["score"]), 5) for item in detections]
                written += 1
                total_proposals += len(detections)
        # The score sidecar sits next to the labels, named after the sequence.
        (out / f"{sequence}__scores.json").write_text(json.dumps(scores), encoding="utf-8")
        total_files += written
        print(f"[OK] {sequence}: {written} frame files, {sum(len(v) for v in scores.values())} proposals")

    print(f"[SUCCESS] sequences={len(sources)} label_files={total_files} proposals={total_proposals} -> {out.resolve()}")
    print(
        "[NOTE] 这些是**候选**不是真值：人工在 web 标注器里改；"
        f"交付/评测前用 python -m manu.annotate.export_labels 转成 <seq>__<idx>.txt（已是该布局）。"
    )


if __name__ == "__main__":
    main()
