#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Print one progress line per shard for a batch of render_multiscale_osd_video.py runs.

tqdm redraws in place with carriage returns, so a log holds thousands of stale bars on one physical line.
Reading it as text (or splitting on newlines) therefore shows the first video forever; this parses with
``newline=""`` so every redraw survives as its own segment and only the last one is current.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

BAR = re.compile(r"(\S+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[\d:]+<([\d:]+),\s*([\d.]+)frame/s")
ANNOUNCE = re.compile(r"^\[VIDEO ", re.M)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show per-shard progress of the OSD render logs")
    parser.add_argument("--log-dir", required=True, help="Directory holding shard<N>.log")
    parser.add_argument("--shards", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.log_dir)
    for index in range(args.shards):
        path = root / f"shard{index}.log"
        if not path.is_file():
            print(f"shard{index} [WAIT ] no log yet")
            continue
        with path.open(errors="replace", newline="") as log:
            raw = log.read()
        done = raw.count("[DONE]")
        planned = len(ANNOUNCE.findall(raw))
        state = "[DONE]" if "[SUCCESS] videos=" in raw else ("[RUN ]" if planned else "[WAIT ]")
        current = BAR.search(raw.split("\r")[-1])
        detail = ""
        if current:
            name, percent, index_done, index_total, eta, fps = current.groups()
            detail = f"{name[:26]:28s} {percent:>3}%  {index_done}/{index_total}  ETA {eta}  @{fps}fps"
        print(f"shard{index} {state} {done}/{planned:<3} {detail}")


if __name__ == "__main__":
    sys.exit(main())
