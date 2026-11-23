#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Re-detect the modality (infrared / visible-light) of every analysed clip.

The file names in these folders are unreliable (``130CT-02可见光_OSDT.mp4`` is named
"visible light" but starts with a grayscale infrared picture and switches to colour later on), so
the modality is decided by the pixels: P99 HSV saturation, sampled across the whole runtime and
decided by the majority of the sampled seconds.  The metric arrays themselves do not depend on the
modality, only the grouping does, so a re-classification pass is enough - no re-extraction needed.

Every clip also carries the share of the sampled points that were classified as colour
(``modality_vis_ratio``), how many points were probed (``modality_samples``), how many of them
were colour (``modality_coloured``) and when the first colour point appears
(``modality_first_vis_seconds``), so the modality of a clip can be recomputed from the artefacts
alone.  ``run_manifest.json`` is the batch run record: its ``modality`` field stays the batch-time
first impression and the final grouping is added next to it as ``modality_final``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_run_server import MODALITY_PROBE_SECONDS, probe_modality  # noqa: E402

WORKERS = 4


def sync_manifest(manifest_path: Path, finals: dict) -> None:
    """Add the final grouping modality to the batch run manifest.

    The manifest records the batch-time classification in ``modality``; that value is the run
    history and stays untouched.  ``modality_final`` is the modality the clip ended up in for the
    analysis: the pixel probe for source clips, ``osd`` for the ground-station screen recordings
    (they are grouped by ``kind`` and do not take part in the source-video statistics).  The
    numbers behind the probe are recorded next to them so the decision stays reproducible.
    """
    if not manifest_path.exists():
        print(f"[SKIP] {manifest_path.name}: manifest not found")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated = 0
    for entry in manifest:
        final = finals.get(entry.get("file"))
        if final is None:
            print(f"[SKIP] manifest entry {entry.get('file')}: no analysed clip")
            continue
        kind, probe = final
        entry["modality_final"] = "osd" if kind == "osd" else probe.modality
        entry["modality_vis_ratio"] = round(probe.vis_ratio, 6)
        entry["modality_samples"] = probe.samples
        entry["modality_coloured"] = probe.coloured
        entry["modality_first_vis_seconds"] = probe.first_vis_seconds
        updated += 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] manifest: {updated}/{len(manifest)} entries carry modality_final/modality_vis_ratio/probe counts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-classify modality for analysed clips")
    parser.add_argument("--metrics-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    jobs = []
    for json_path in sorted(metrics_dir.glob("*.json")):
        if json_path.name == "run_manifest.json":
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        source = data.get("path")
        if not source or not Path(source).exists():
            print(f"[SKIP] {json_path.name}: source video not found")
            continue
        jobs.append((json_path, Path(source), data.get("modality")))

    def probe(job):
        json_path, source, previous = job
        return json_path, source, previous, probe_modality(source)

    changed = []
    finals = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for json_path, source, previous, probe in pool.map(probe, jobs):
            modality = probe.modality
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if modality != previous:
                data["modality"] = modality
                data["modality_reclassified"] = True
                changed.append((json_path.name, previous, modality))
            data["modality_vis_ratio"] = round(probe.vis_ratio, 6)
            data["modality_probe_seconds"] = MODALITY_PROBE_SECONDS
            data["modality_samples"] = probe.samples
            data["modality_coloured"] = probe.coloured
            data["modality_first_vis_seconds"] = probe.first_vis_seconds
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            finals[data["file"]] = (data["kind"], probe)
            print(
                f"{json_path.name:<44} {previous} -> {modality}  vis_ratio={probe.vis_ratio:.6f}"
                f"  samples={probe.samples} coloured={probe.coloured}"
                f"  first_vis={probe.first_vis_seconds}",
                flush=True,
            )
    print(f"[INFO] reclassified {len(changed)} clips")
    for name, before, after in changed:
        print(f"  changed: {name} {before} -> {after}")
    sync_manifest(metrics_dir / "run_manifest.json", finals)


if __name__ == "__main__":
    sys.exit(main())
