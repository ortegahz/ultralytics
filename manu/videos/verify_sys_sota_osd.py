#!/usr/bin/env python3
"""Verify the system-SOTA OSD deliverables: 24 videos, filenames carry the F1, metrics agree.

Checks, per the delivery contract:
  * exactly 24 mp4 files, split into pass/ and un-pass/
  * every filename is ``<SEQ>_F1-<f1>.mp4`` and the F1 in the name matches the TSV
  * the video is readable, has the expected frame count and resolution
  * the per-sequence metrics in the TSV match the search JSON

Usage:
    python manu/videos/verify_sys_sota_osd.py \
        --osd-dir "/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_20261020" \
        --search-json runs/logs/sys_sota_longquan_search.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

NAME_RE = re.compile(r"^(?P<seq>VIDEO\d+_\d{8}_\d{6})_F1-(?P<f1>\d+\.\d{3})\.mp4$")


def probe(path: Path) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True)
        return (json.loads(out.stdout).get("streams") or [{}])[0]
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osd-dir", required=True)
    ap.add_argument("--search-json", default="")
    ap.add_argument("--expect", type=int, default=24)
    args = ap.parse_args()

    root = Path(args.osd_dir)
    problems: list[str] = []
    rows: list[tuple[str, str, float, Path]] = []

    for group in ("pass", "un-pass"):
        folder = root / group
        if not folder.is_dir():
            problems.append(f"缺少目录 {folder}")
            continue
        for mp4 in sorted(folder.glob("*.mp4")):
            m = NAME_RE.match(mp4.name)
            if not m:
                problems.append(f"文件名不符合 <SEQ>_F1-<f1>.mp4 约定: {mp4.name}")
                continue
            rows.append((group, m.group("seq"), float(m.group("f1")), mp4))

    print(f"[VERIFY] 产物根目录 {root}")
    print(f"[VERIFY] 视频数 {len(rows)} / 期望 {args.expect}")
    if len(rows) != args.expect:
        problems.append(f"视频数 {len(rows)} != 期望 {args.expect}")

    tsv = root / "per_sequence_hyperparams.tsv"
    tsv_map: dict[str, float] = {}
    if tsv.is_file():
        lines = [ln for ln in tsv.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        header = [c.strip() for c in lines[0].split("\t")]
        try:
            seq_i, f1_i = header.index("sequence"), header.index("f1_pct")
        except ValueError:
            problems.append(f"TSV 缺少 sequence/f1_pct 列: {header}")
        else:
            for line in lines[1:]:
                cells = [c.strip() for c in line.split("\t")]
                if len(cells) > max(seq_i, f1_i):
                    tsv_map[cells[seq_i]] = float(cells[f1_i])
        print(f"[VERIFY] TSV {len(tsv_map)} 行")
    else:
        problems.append(f"缺少 {tsv}")

    search_metrics: dict[str, dict] = {}
    if args.search_json and Path(args.search_json).is_file():
        data = json.loads(Path(args.search_json).read_text())
        best = data.get("best_global") or {}
        search_metrics = best.get("per_sequence") or {}
        print(f"[VERIFY] 搜索 JSON best_global 覆盖 {len(search_metrics)} 条序列")

    for group, seq, f1, mp4 in rows:
        info = probe(mp4)
        if "error" in info:
            problems.append(f"{mp4.name}: ffprobe 失败 {info['error']}")
            continue
        size_mb = mp4.stat().st_size / 1024 / 1024
        line = f"  {group:7s} {seq}  F1={f1:7.3f}  {int(info.get('width', 0))}x{int(info.get('height', 0))}" \
               f"  frames={info.get('nb_frames', '?')}  {size_mb:.0f}MB"
        if seq in tsv_map and abs(tsv_map[seq] - f1) > 0.001:
            problems.append(f"{seq}: 文件名 F1 {f1} 与 TSV {tsv_map[seq]} 不一致")
        if search_metrics.get(seq):
            want = float(search_metrics[seq].get("f1", 0)) * 100
            if abs(want - f1) > 0.01:
                problems.append(f"{seq}: 文件名 F1 {f1} 与搜索 JSON {want:.3f} 不一致")
        print(line)

    print()
    if problems:
        print(f"[VERIFY] ✗ 发现 {len(problems)} 个问题:")
        for p in problems:
            print("   -", p)
        raise SystemExit(1)
    print("[VERIFY] ✅ 全部通过")


if __name__ == "__main__":
    main()
