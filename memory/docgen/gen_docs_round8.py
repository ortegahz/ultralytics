#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§32 文档生成器：**所有数字从产物 JSON 读取**，不手抄（避免文档与实测漂移）。

产物：
  * 专题档 §32 → ``manu/memory/lookahead_extrapolator_20261102.md``
  * 主档 §30.9.26 → ``manu/memory/memory.md``
  * README §5.16 → ``manu/tracking/lookahead/README.md``
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "runs/lookahead/final"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def arm_row(doc, arm="replay"):
    cur = doc["arms"][arm]["current_frame"]["overall"]
    srv = doc["arms"][arm]["servo"]["overall"]
    return {
        "f1": cur["f1"] * 100,
        "far": cur["far_per_frame"],
        "hit": srv["hit_rate_5px"] * 100,
        "p50": srv.get("cle_pooled_p50", srv.get("cle_p50")),
        "p95": srv.get("cle_pooled_p95", srv.get("cle_p95")),
        "p99": srv.get("cle_pooled_p99", srv.get("cle_p99")),
        "mj": srv.get("max_jitter"),
        "lost": srv.get("lost_frames"),
    }


def collect(patterns):
    """按 (目录, 文件名) 收集，避免 8 序/24 序同名文件互相覆盖。"""
    rows = []
    for pattern in patterns:
        for path in sorted(FINAL.glob(pattern)):
            doc = load(path)
            if "arms" not in doc:
                continue
            try:
                row = arm_row(doc)
            except KeyError:
                continue
            row["tag"] = f"{path.parent.name}/{path.stem.replace('.json', '')}"
            rows.append(row)
    return rows


def main() -> int:
    out = {}
    rows = collect(
        ["chainmp/*.json", "chainroimax/*.json", "mplo/*.json", "chainfar/*.json",
         "final8/arms_delivery.json", "final24/arms_delivery.json", "final24par/*.json"]
    )
    for row in rows:
        out[row["tag"]] = row
    (FINAL / "final8" / "round8_variants.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] {len(out)} variants -> {FINAL / 'final8' / 'round8_variants.json'}")
    for tag in sorted(out):
        row = out[tag]
        print(
            f"  {tag:24s} F1={row['f1']:9.4f} FAR={row['far']:9.6f} hit={row['hit']:8.4f} "
            f"p50={row['p50']:.2f} p95={row['p95']:.2f} mj={row['mj']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
