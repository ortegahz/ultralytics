#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
22m_parse_build_log.py -- read the toolkit's operator table out of a build log.

WHY
---
The RKNN toolkit prints the final operator table only in verbose mode, and it
is the ONLY place that says what the model's INPUT tensor is:

    D RKNN: [09:48:18.191] 0    InputOperator      INT8     CPU    (1,3,640,640)

That single line decides the latency budget: an INT8 input means the host
uploads a uint8 buffer the runtime widens in place (measured 0.34 ms), a
FLOAT16 input means the runtime has to convert 640*640*3 elements in host
space (measured 45.41 ms) and the NPU has to hold a 105,600 KB fp16 tensor
instead of a 20,400 KB int8 one.

So this script answers, for every arm:
    input_op_dtype      the dtype of the InputOperator
    dtype_by_device     how many operators of each dtype run where
    n_operators         the table size
and writes them into a JSON that the delivery report quotes.

Usage (server):
    python manu/pipeline/npu/ptq/22m_parse_build_log.py \
        --log /mnt/data/siping/ptq_work/logs/a2f_build.log \
        --name a2f_qatw_toolkit_int8 \
        --out  /mnt/data/siping/ptq_work/qat/trial_0080/report/build_ops.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# D RKNN: [09:48:18.191] 0    InputOperator      INT8     CPU    (1,3,640,640) ...
ROW = re.compile(
    r"^D RKNN: \[(?P<ts>[^\]]+)\]\s+(?P<idx>\d+)\s+(?P<op>\S+)\s+"
    r"(?P<dtype>[A-Z0-9]+)\s+(?P<dev>\S+)\s+(?P<rest>.*)$"
)


SECTION = re.compile(r"^=+\s*BUILD\s+(\S+)\s+\(mean_std=(\d)\)\s*=+\s*$", re.M)


def split_sections(path: Path) -> dict:
    """A multi-build log (22d_build_qat_arms.sh) holds several builds; split it
    on the '=== BUILD <name> (mean_std=<m>) ===' banner."""
    out: dict = {}
    name = None
    ms = None
    buf: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        m = SECTION.match(line)
        if m:
            if name is not None:
                out[(name, ms)] = "\n".join(buf)
            name, ms = m.group(1), m.group(2)
            buf = []
            continue
        if name is not None:
            buf.append(line)
    if name is not None:
        out[(name, ms)] = "\n".join(buf)
    return out


def parse_log(path: Path) -> dict:
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = ROW.match(line.rstrip("\n"))
            if m:
                rows.append(m.groupdict())
    if not rows:
        raise SystemExit(f"[FATAL] no operator table found in {path} "
                         "(was the build run with verbose=True?)")
    input_rows = [r for r in rows if r["op"] == "InputOperator"]
    dtype_all = Counter(r["dtype"] for r in rows)
    dtype_npu = Counter(r["dtype"] for r in rows if r["dev"] == "NPU")
    dtype_cpu = Counter(r["dtype"] for r in rows if r["dev"] != "NPU")
    ops = Counter(r["op"] for r in rows)
    return {
        "log": str(path),
        "n_operators": len(rows),
        "input_op_dtype": input_rows[0]["dtype"] if input_rows else None,
        "input_op_device": input_rows[0]["dev"] if input_rows else None,
        "input_op_shape": input_rows[0]["rest"].split()[0] if input_rows else None,
        "dtype_all": dict(dtype_all),
        "dtype_npu": dict(dtype_npu),
        "dtype_cpu": dict(dtype_cpu),
        "n_npu_ops": sum(dtype_npu.values()),
        "n_cpu_ops": sum(dtype_cpu.values()),
        "top_ops": dict(ops.most_common(12)),
        "rows": [{"op": r["op"], "dtype": r["dtype"], "dev": r["dev"]} for r in rows],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", action="store_true",
                    help="merge into an existing JSON instead of replacing it")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if args.merge and out.is_file():
        data = json.loads(out.read_text())

    log = Path(args.log)
    sections = split_sections(log) if SECTION.search(
        log.read_text(errors="replace")[:400000].replace("\n", "\n") + "\n") else {}
    if sections:
        # a multi-build log: keep only the mean_std=1 sections (the ones whose
        # mean/std recipe is the usable one)
        for (name, ms), text in sections.items():
            if ms != "1":
                continue
            tmp = log.parent / f"._section_{name}.log"
            tmp.write_text(text)
            info = parse_log(tmp)
            info["name"] = name
            info["mean_std"] = int(ms)
            data[name] = info
            print(f"[ops] {name}: input_op_dtype={info['input_op_dtype']} "
                  f"({info['input_op_device']})  ops={info['n_operators']} "
                  f"npu={info['n_npu_ops']} cpu={info['n_cpu_ops']}")
            print(f"[ops]   dtype_all={info['dtype_all']}")
    else:
        info = parse_log(log)
        info["name"] = args.name
        data[args.name] = info
        print(f"[ops] {args.name}: input_op_dtype={info['input_op_dtype']} "
              f"({info['input_op_device']})  ops={info['n_operators']} "
              f"npu={info['n_npu_ops']} cpu={info['n_cpu_ops']}")
        print(f"[ops]   dtype_all={info['dtype_all']}")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"[ops] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
