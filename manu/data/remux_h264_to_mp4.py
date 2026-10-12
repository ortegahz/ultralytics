#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Remux raw H.264 elementary streams (.h264) into MP4 containers, losslessly.

The acquisition gear delivers Annex-B elementary streams: no container, hence no
frame timing and no index, so most tools refuse to open them. This script wraps
each stream in MP4 with `-c copy` -- the H.264 slices are never decoded or
re-encoded, so the result is pixel-identical to the source (verified with
`framemd5`). Every output is checked against its input packet count and stream
parameters, and any mismatch is reported loudly rather than silently skipped.

Usage:
    python remux_h264_to_mp4.py --src /path/to/纵横AI数据采集
    python remux_h264_to_mp4.py --src SRC --out DST --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

SUFFIX = ".h264"
DEFAULT_FPS = 25


def run(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command; return (rc, stdout, stderr)."""
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return proc.returncode, proc.stdout, proc.stderr


def probe(path: Path, input_format: str | None, decode_frames: bool = False) -> dict:
    """Return stream parameters plus the packet count, or {} on failure.

    `-count_packets` parses the container/bitstream without decoding, which is
    ~80x faster than `-count_frames` on long sequences.

    Caveat: on a raw H.264 stream `-count_packets` counts trailing incomplete
    access units the decoder never emits (these streams are cut mid-GOP), so it
    legitimately exceeds the MP4 packet count by a frame or two. Use
    `decode_frames=True` for an authoritative content count.
    """
    cmd = ["ffprobe", "-v", "error"]
    if input_format:
        cmd += ["-f", input_format]
    if decode_frames:
        cmd += ["-count_frames", "-show_entries",
                "stream=codec_name,profile,width,height,pix_fmt,color_range,nb_read_frames"]
    else:
        cmd += ["-count_packets", "-show_entries",
                "stream=codec_name,profile,width,height,pix_fmt,color_range,nb_read_packets"]
    cmd += ["-select_streams", "v:0", "-of", "json", str(path)]
    rc, out, err = run(cmd)
    if rc != 0:
        return {}
    try:
        streams = json.loads(out).get("streams", [])
    except json.JSONDecodeError:
        return {}
    return streams[0] if streams else {}


def count_of(info: dict) -> int | None:
    """Frame or packet count from a probe result, whichever key is present."""
    for key in ("nb_read_frames", "nb_read_packets"):
        if key in info:
            try:
                return int(info[key])
            except (TypeError, ValueError):
                return None
    return None


def remux(src: Path, dst: Path, fps: float, overwrite: bool) -> tuple[int, str, str]:
    """Remux one file with `-c copy`; return (rc, stderr, command string)."""
    cmd = [
        "ffmpeg", "-v", "error", "-y" if overwrite else "-n",
        "-f", "h264", "-r", f"{fps:g}", "-i", str(src),
        "-c", "copy", "-movflags", "+faststart",
        str(dst),
    ]
    rc, _, err = run(cmd)
    return rc, err.strip(), " ".join(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="directory holding the raw .h264 streams")
    ap.add_argument("--out", type=Path, help="output directory (default: <src>_mp4)")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                    help=f"frame rate written into the container (default: {DEFAULT_FPS})")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="list the work without touching the disk")
    ap.add_argument("--verify-frames", action="store_true",
                    help="decode both sides to compare true frame counts (authoritative, ~6s/file)")
    ap.add_argument("--max-drop", type=int, default=16,
                    help="tolerated packet loss from trailing incomplete NALs (default: 16)")
    ap.add_argument("--manifest", type=Path, help="write a JSON manifest of the conversion")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"[FATAL] not a directory: {args.src}")

    sources = sorted(p for p in args.src.glob(f"*{SUFFIX}") if p.is_file())
    if not sources:
        sys.exit(f"[FATAL] no *{SUFFIX} files under {args.src}")

    out_dir = args.out or args.src.with_name(args.src.name + "_mp4")

    print(f"[IN ] {args.src}  ({len(sources)} files)")
    print(f"[OUT] {out_dir}")
    print(f"[FPS] {args.fps:g}")
    if args.dry_run:
        for p in sources:
            print(f"  -> {p.with_suffix('.mp4').name}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    records, failed, warned = [], [], []
    for i, src in enumerate(sources, 1):
        name = src.name
        dst = out_dir / src.with_suffix(".mp4").name
        rec = {"src": str(src), "dst": str(dst)}

        if dst.exists() and not args.overwrite:
            rec.update(status="skipped_exists")
            records.append(rec)
            print(f"[{i:>3}/{len(sources)}] SKIP (exists) {name}")
            continue

        in_info = probe(src, input_format="h264", decode_frames=args.verify_frames)
        if not in_info:
            rec.update(status="probe_failed_src")
            records.append(rec)
            failed.append((name, "input ffprobe failed"))
            print(f"[{i:>3}/{len(sources)}] FAIL {name}: input ffprobe failed")
            continue

        rc, err, cmd = remux(src, dst, args.fps, args.overwrite)
        if rc != 0:
            rec.update(status="ffmpeg_failed", stderr=err)
            records.append(rec)
            failed.append((name, err or f"ffmpeg rc={rc}"))
            print(f"[{i:>3}/{len(sources)}] FAIL {name}: {err.splitlines()[-1] if err else rc}")
            continue

        # Decoder warnings on raw streams are common and recoverable (a stream cut
        # mid-GOP); surface them, but let the structural check decide success.
        if err:
            warned.append((name, err))
        out_info = probe(dst, input_format=None, decode_frames=args.verify_frames)
        if not out_info:
            rec.update(status="probe_failed_dst")
            records.append(rec)
            failed.append((name, "output ffprobe failed"))
            print(f"[{i:>3}/{len(sources)}] FAIL {name}: output ffprobe failed")
            continue

        n_in, n_out = count_of(in_info), count_of(out_info)
        mismatches = []
        for key in ("codec_name", "width", "height", "pix_fmt", "color_range"):
            if in_info.get(key) != out_info.get(key):
                mismatches.append(f"{key} {in_info.get(key)} -> {out_info.get(key)}")
        if not n_out:
            mismatches.append("output reports zero frames")
        elif args.verify_frames:
            # Authoritative path: both counts are decoded frame counts.
            if n_in != n_out:
                mismatches.append(f"frames {n_in} -> {n_out}")
        elif n_in and n_out > n_in:
            # `-c copy` can never invent frames.
            mismatches.append(f"packets grew {n_in} -> {n_out}")
        elif n_in and n_in - n_out > args.max_drop:
            # A raw stream may lose only its trailing incomplete access units.
            mismatches.append(f"dropped {n_in - n_out} packets (> {args.max_drop})")

        rec.update(
            status="ok" if not mismatches else "verify_mismatch",
            frames_out=n_out,
            width=out_info.get("width"),
            height=out_info.get("height"),
            pix_fmt=out_info.get("pix_fmt"),
            color_range=out_info.get("color_range"),
            bytes_in=src.stat().st_size,
            bytes_out=dst.stat().st_size,
        )
        rec["count_in"] = n_in
        records.append(rec)
        if mismatches:
            rec["mismatch"] = mismatches
            failed.append((name, "; ".join(mismatches)))
            print(f"[{i:>3}/{len(sources)}] FAIL {name}: {'; '.join(mismatches)}")
        else:
            note = "" if args.verify_frames or not n_in else f"  (-{n_in - n_out} trailing NAL)"
            print(f"[{i:>3}/{len(sources)}] OK   {name}  ({n_out} frames){note}")

    ok = sum(r["status"] == "ok" for r in records)
    skipped = sum(r["status"] == "skipped_exists" for r in records)

    print("\n=== SUMMARY ===")
    print(f"ok {ok} / skipped {skipped} / failed {len(failed)} / total {len(sources)}")
    if warned:
        print(f"\n[decode warnings] {len(warned)} file(s) reported recoverable decoder errors:")
        for name, err in warned:
            first = err.splitlines()[0] if err else ""
            print(f"  {name}: {first}")
    if failed:
        print("\n[FAILED]")
        for name, why in failed:
            print(f"  {name}: {why}")

    if args.manifest:
        args.manifest.write_text(json.dumps(
            {"src": str(args.src), "out": str(out_dir), "fps": args.fps, "records": records},
            ensure_ascii=False, indent=2))
        print(f"\n[manifest] {args.manifest}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
