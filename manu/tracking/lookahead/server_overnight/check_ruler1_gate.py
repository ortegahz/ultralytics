#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""尺子 1（历史帧等价性）硬门禁的**全局**核查。

外挂是纯后处理、绝不回写历史估计 ⇒ 同一产物内四臂（original / stale / kinematic / replay）的
`source_frame.overall.f1` 必须**逐位相等**。本脚本把所有已产出 `arms.json` 扫一遍，
任何一个产物不满足就 FAIL。

用法：
    python manu/tracking/lookahead/server_overnight/check_ruler1_gate.py \
        --out runs/lookahead/fi_nogmc/ruler1_gate_check.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOTS = [
    "runs/lookahead/fi_nogmc/*/d*/arms.json",
    "runs/lookahead/fi_nogmc24/*/d*/arms.json",
    "runs/lookahead/fovgate/runs/*/d*/fov__m*/arms.json",
    "runs/lookahead/fovgate24/runs/*/d*/fov__m*/arms.json",
    "runs/lookahead/fovgate24nogmc/runs/*/d*/fov__m*/arms.json",
    "runs/lookahead/fovgate24nogmc_d2/runs/*/d*/fov__m*/arms.json",
    "runs/lookahead/fovgate24nogmc_d8/runs/*/d*/fov__m*/arms.json",
    "runs/lookahead/fovgate24rc2_d8/runs/*/d*/fov__m*/arms.json",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="尺子1 硬门禁全局核查")
    ap.add_argument("--out", default="runs/lookahead/fi_nogmc/ruler1_gate_check.json")
    args = ap.parse_args()

    files: list[str] = []
    for root in ROOTS:
        files += sorted(glob.glob(root))

    checked, bad = 0, []
    for path in files:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        arms = doc.get("arms")
        if not isinstance(arms, dict):
            continue
        vals = {}
        for arm, body in arms.items():
            overall = ((body or {}).get("source_frame") or {}).get("overall") or {}
            vals[arm] = overall.get("f1")
        if len(set(vals.values())) != 1:
            bad.append({"path": path, "source_frame_f1": vals})
        checked += 1

    result = {
        "checked_products": checked,
        "violations": len(bad),
        "verdict": "PASS" if not bad else "FAIL",
        "bad": bad,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[{result['verdict']}] 尺子1 硬门禁：{checked} 份产物，不逐位相等 {len(bad)} 份 -> {out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
