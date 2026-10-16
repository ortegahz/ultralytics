#!/usr/bin/env python3
"""
Patch a hybrid-quantization cfg so a given tensor list is forced to float16.

Why this exists
---------------
RKNN-Toolkit2 2.3.0's two-step hybrid flow crashes on this graph:

    step1 -> OK, and its own quant_optimizer reports
             adjust_tanh_sigmoid: ['/model/head/Sigmoid']
    step2 -> KeyError: '/model/head/feat_conv/feat_conv.1/act/Sigmoid_output_0'
             raised from quant_optimizer._p_adjust_tanh_sigmoid.try_forward_broadcast

The shapes of the two runs differ exactly on this point. step1's optimizer
walked the graph BEFORE the hybrid split and skipped every Sigmoid that was
already in the float16 list, so only one survivor was printed. step2 rebuilds
the graph with the hybrid split applied and re-runs the same optimizer, and now
a Sigmoid whose output tensor has no entry in the cfg at all reaches
try_forward_broadcast and dies on the missing key.

So the fix is to put that tensor in the float16 list too: the optimizer then
skips it exactly as it did in step1, and the crash has no trigger. This is a
workaround for a toolkit defect, not a design decision -- it is recorded as
such so nobody later mistakes it for a precision choice.

The edit is additive and idempotent: re-running on an already-patched file
changes nothing.

Usage:
    python patch_hybrid_cfg.py --cfg <cfg> --add <tensor> [--add <tensor> ...] [--out <cfg>]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--add", action="append", default=[],
                    help="tensor name to force to float16 (repeatable)")
    a = ap.parse_args()

    cfg = Path(a.cfg)
    out = Path(a.out) if a.out else cfg
    if not cfg.is_file():
        sys.exit(f"[FATAL] no cfg at {cfg}")

    lines = cfg.read_text().splitlines()
    # The section header carries a trailing score comment, so it is matched by
    # prefix rather than by equality.
    start = next((i for i, l in enumerate(lines)
                  if l.split("#")[0].strip().startswith("custom_quantize_layers:")), None)
    if start is None:
        sys.exit("[FATAL] cfg has no custom_quantize_layers section")

    # Section runs until the next unindented key.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i] and not lines[i].startswith(" "):
            end = i
            break

    existing = set()
    for l in lines[start + 1:end]:
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        existing.add(s.split(":")[0].strip())

    added = [t for t in a.add if t not in existing]
    if added:
        lines[end:end] = [f"    {t}: float16" for t in added]
        if out != cfg:
            shutil.copyfile(cfg, out)
        out.write_text("\n".join(lines) + "\n")

    print(f"[cfg ] {cfg}")
    print(f"[add ] {len(added)} new / {len(a.add)} requested")
    for t in a.add:
        print(f"        {'+ ' if t in added else '= '}{t}")


if __name__ == "__main__":
    main()
