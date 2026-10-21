#!/usr/bin/env python
"""打印 YOLO26HeatmapDetector 的全部参数名，并按分层冻结方案标注每个参数组。

用途：把用户方案里的「Stem/P1、Stage1/P2、Stage2/P3、P4/P5、Neck、Head」
映射到**真实模块名**，避免用 model.0..model.4 之类的臆测名。

分层冻结方案（路线 B）
---------------------
  Group1 严格冻结 : Stem/P1、Stage1/P2、Stage2/P3
  Group2 微弱解冻 : P4/P5                     lr=3e-6
  Group3 中度解冻 : Neck                      lr=1.5e-5
  Group4 主力     : Head                      lr=3e-5

用法::

    python manu/data/probe_param_groups.py --out-json runs/logs/param_groups.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manu.models.heatmap_model import YOLO26HeatmapDetector  # noqa: E402

# 分层规则：按参数名的前缀判定，顺序敏感（先匹配到的胜出）
GROUP_RULES: list[tuple[str, str, str]] = [
    # (正则, 组名, 说明)
    (r"^b[0-4]\.", "G1_frozen", "Stem/P1 + Stage1/P2 + Stage2/P3（严格冻结）"),
    (r"^b(5|6|7|8|9|10)\.", "G2_p4p5", "P4/P5 特征层（微弱解冻 lr=3e-6）"),
    (r"^(up1|c13|up2|c16|up3|c19|down1|c22|up_p2|fuse_p2)\.", "G3_neck", "Neck（FPN+PAN，lr=1.5e-5）"),
    (r"^head\.", "G4_head", "Head 分类/回归（lr=3e-5）"),
    (r"^p0_highway\.", "P0", "P0 高速支路（本次微调对象为去 P0 模型，应为空）"),
]


def classify(name: str) -> tuple[str, str]:
    for pattern, group, desc in GROUP_RULES:
        if re.search(pattern, name):
            return group, desc
    return "OTHER", "未归类 —— 需人工确认"


def main() -> int:
    parser = argparse.ArgumentParser(description="打印参数分组")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    model = YOLO26HeatmapDetector(
        stride=2, num_classes=1, temporal_mode="standard", use_p0_highway=False
    )
    groups: dict[str, dict] = {}
    rows = []
    for name, param in model.named_parameters():
        group, desc = classify(name)
        g = groups.setdefault(group, {"desc": desc, "params": 0, "tensors": 0, "names": []})
        g["params"] += param.numel()
        g["tensors"] += 1
        g["names"].append(name)
        rows.append({"name": name, "shape": list(param.shape), "numel": param.numel(), "group": group})

    print("=" * 100)
    print(f"  模型总参数: {sum(p.numel() for p in model.parameters()):,} | 张量数: {len(rows)}")
    print("=" * 100)
    for group in ("G1_frozen", "G2_p4p5", "G3_neck", "G4_head", "P0", "OTHER"):
        if group not in groups:
            continue
        g = groups[group]
        print(f"  {group:<10} 参数 {g['params']:>12,}  张量 {g['tensors']:>4}  —— {g['desc']}")
    if "OTHER" in groups:
        print("  ⚠️ OTHER 未归类参数：")
        for n in groups["OTHER"]["names"][:20]:
            print(f"      {n}")
    print("=" * 100)

    # 顶层模块名一览，便于人工核对分层
    tops = sorted({n.split(".")[0] for n, _ in model.named_parameters()})
    print(f"  顶层模块: {tops}")
    print("=" * 100)

    report = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "total_tensors": len(rows),
        "groups": {k: {"desc": v["desc"], "params": v["params"], "tensors": v["tensors"]} for k, v in groups.items()},
        "top_level_modules": tops,
        "unclassified": groups.get("OTHER", {}).get("names", []),
        "rows": rows,
    }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
