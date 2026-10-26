#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22f_make_smoke_manifest.py -- cut the first N frames out of an eval manifest
and write the board-side batch list for them.

WHY
---
A model that is broken in a way that makes the BOARD hang (or makes its own
null control fire) must be caught on a handful of frames, not on 2000.  The
smoke run therefore needs

  * a truncated manifest, so the eval driver scores exactly the same frames
    the board was fed, and
  * the board-side list.txt, because the board reads the BMPs from its own
    NFS mount (/mnt/manu/ptq/...) and not from the server's paths.

Usage (server):
    python manu/pipeline/npu/ptq/22f_make_smoke_manifest.py \
        --manifest /mnt/data/siping/ptq_work/fast/tier1_sensitive.json \
        --n 40 \
        --out-manifest /mnt/data/siping/ptq_work/qat/trial_0080/smoke40.json \
        --out-list /mnt/data/siping/ptq_work/qat/trial_0080/smoke40_list.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from manu.pipeline.npu.ptq import ptq_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-list", required=True)
    ap.add_argument("--board-mount", default="/mnt/manu/ptq")
    args = ap.parse_args()

    raw = C.read_json(args.manifest)
    rows = raw if isinstance(raw, list) else raw["items"]
    rows = rows[: args.n]
    if not rows:
        raise SystemExit(f"[FATAL] manifest {args.manifest} is empty")

    out_m = Path(args.out_manifest)
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_m.write_text(json.dumps({"name": out_m.stem, "items": rows},
                                ensure_ascii=False, indent=1))

    lines = []
    for r in sorted(rows, key=lambda x: (x["seq"], str(x["frame_id"]))):
        bmp = Path(r["bmp"])
        try:
            rel = bmp.relative_to(C.BMP_ROOT)
            lines.append(f"{args.board_mount}/bmp/{rel}")
        except ValueError:
            lines.append(str(bmp))
    out_l = Path(args.out_list)
    out_l.parent.mkdir(parents=True, exist_ok=True)
    out_l.write_text("\n".join(lines) + "\n")

    print(f"[smoke] {len(rows)} frames -> {out_m}")
    print(f"[smoke] board list -> {out_l}")
    print(f"[smoke] first board path: {lines[0]}")
    print(f"[smoke] last  board path: {lines[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
