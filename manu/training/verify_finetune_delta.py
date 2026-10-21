#!/usr/bin/env python
"""核查微调权重是否「按设计」变化：冻结组必须逐位未动，可训组必须真的更新。

这是区分「配方太猛」与「实现有 bug」的决定性检查：
  - G1（b0..b4，严格冻结）必须与源 ckpt **逐位相同**
  - G2/G3/G4（可训）必须与源 ckpt **存在差异**（否则 lr 没生效）
  - p0 张量：应为源 ckpt 原样带回（不参与前向，仅供评测侧 --ablate-p0 门控）

用法::

    python manu/training/verify_finetune_delta.py \\
        --weights runs/mixed_ft_route_b/weights/epoch1.pt \\
        --source runs/optuna_p0_nas/trial_0474/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

G1 = r"^b[0-4]\."
G2 = r"^b(5|6|7|8|9|10)\."
G3 = r"^(up1|c13|up2|c16|up3|c19|down1|c22|up_p2|fuse_p2|fuse_p1)\."
G4 = r"^head\."


def group_of(name: str) -> str | None:
    for pat, g in ((G1, "G1_frozen"), (G2, "G2_p4p5"), (G3, "G3_neck"), (G4, "G4_head")):
        if re.search(pat, name):
            return g
    return None


def load_state(path: str) -> tuple[dict, dict]:
    ck = torch.load(path, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    meta = {k: ck.get(k) for k in ("epoch", "stride", "imgsz", "p0_kwargs", "finetune") if k in ck}
    return sd, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="核查微调权重变化是否符合分层设计")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="runs/optuna_p0_nas/trial_0474/weights/best.pt")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    new, new_meta = load_state(args.weights)
    old, old_meta = load_state(args.source)
    print("=" * 96)
    print(f"  new  : {args.weights}  epoch={new_meta.get('epoch')} stride={new_meta.get('stride')}")
    print(f"  source: {args.source}  epoch={old_meta.get('epoch')} stride={old_meta.get('stride')}")
    print("=" * 96)

    stats: dict[str, dict] = {}
    unmatched: list[str] = []
    for name, p_new in new.items():
        if name.startswith("p0_highway"):
            continue
        if name not in old:
            unmatched.append(name)
            continue
        p_old = old[name]
        if p_old.shape != p_new.shape:
            unmatched.append(f"{name} (shape {tuple(p_old.shape)} -> {tuple(p_new.shape)})")
            continue
        g = group_of(name)
        if g is None:
            continue
        st = stats.setdefault(g, {"tensors": 0, "identical": 0, "changed": 0, "max_abs_delta": 0.0, "sum_abs": 0.0})
        st["tensors"] += 1
        d = (p_new.float() - p_old.float()).abs()
        md = float(d.max())
        if md == 0.0:
            st["identical"] += 1
        else:
            st["changed"] += 1
        st["max_abs_delta"] = max(st["max_abs_delta"], md)
        st["sum_abs"] += float(d.sum())

    p0_new = {k: v for k, v in new.items() if k.startswith("p0_highway")}
    p0_old = {k: v for k, v in old.items() if k.startswith("p0_highway")}
    p0_same = all(k in p0_old and torch.equal(p0_old[k], p0_new[k]) for k in p0_new) and len(p0_new) == len(p0_old)

    report = {
        "weights": args.weights,
        "source": args.source,
        "new_meta": new_meta,
        "groups": stats,
        "unmatched": unmatched,
        "p0_tensor_count": len(p0_new),
        "p0_identical_to_source": p0_same,
    }

    ok = True
    print("  组别         张量数  未变(应全变/应全不变)  已变   max|Δ|")
    for g, expect_changed in (("G1_frozen", False), ("G2_p4p5", True), ("G3_neck", True), ("G4_head", True)):
        st = stats.get(g)
        if st is None:
            print(f"  {g:<12} ——  该组无参数")
            continue
        verdict = "✅" if (st["changed"] > 0) == expect_changed else "❌"
        if (st["changed"] > 0) != expect_changed:
            ok = False
        print(
            f"  {g:<12} {st['tensors']:>6}   未变 {st['identical']:>5} / 已变 {st['changed']:>5}   "
            f"{st['max_abs_delta']:.6e}   {verdict}"
        )
    print(f"  p0 张量带回: {len(p0_new)} 个，与源逐位相同: {p0_same} {'✅' if p0_same else '❌'}")
    if not p0_same:
        ok = False
    if unmatched:
        ok = False
        print(f"  ❌ 未匹配张量: {unmatched[:8]}")
    print("=" * 96)
    print(f"  判定: {'PASS — 权重变化与分层设计完全一致' if ok else 'FAIL — 实现与设计不符'}")
    print("=" * 96)

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入: {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
