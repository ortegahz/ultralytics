#!/usr/bin/env python3
"""Model Soups / 权重平均（2026-11-02 定点打靶 · 变体 C）

动机
----
周末 114 个 trial 里，**同一配方不同种子**的模型在 no-GMC 语义下有稳定的 T3 差异
（种子噪声带 T3 极差 0.170~1.651 pp）。把**同一盆地**里（都从 GMC RC2 trial_0080
起步 ⇒ 同一初始化）的几个 checkpoint 在**参数层面平均**，往往能：
  · 平滑损失曲面 ⇒ 降低方差、提升 T1 稳定性
  · 利用不同种子的互补性 ⇒ T3 提升
这一步**不花任何训练算力**，是当前性价比最高的一击。

关键实现细节（踩过的坑）
------------------------
1. **p0_highway 张量不能平均**。各 trial 的 p0 highway 来源不同
   （rc2 起步的是训练过的、base474 起步的是原始的）⇒ 平均会污染它。
   门禁用 ``use_p0_highway=False``（会 drop 掉 29 个 p0 张量）⇒ 门禁不受影响，
   但为了让产物**自洽可复用**，这里统一取**第一个模型的 p0 值**。
2. **BN 的 running_mean/var 也不宜平均**（不同模型的统计量尺度可能不同）⇒
   提供 ``--nobn`` 变体：只平均**可训练权重**，BN 统计量与 p0 全部取第一个模型。
3. 输出 ckpt 必须与训练产物**同格式**（``{"model": state_dict, ...}``），
   否则门禁的 ``load_hm_model`` 读不出来。

用法
----
    python manu/training/fuse_weights.py \
        --inputs  a/epoch15.pt b/epoch6.pt c/epoch6.pt \
        --out     /mnt/data/.../soups/soup_a.pt \
        [--nobn] [--name soup_a]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

# p0 highway 张量的名字前缀（门禁以 use_p0_highway=False 加载，会丢弃它们）
P0_PREFIX = "p0_highway"


def load_state(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    return {k: v for k, v in state.items()}


def is_p0(key: str) -> bool:
    return key.startswith(P0_PREFIX)


def is_bn_stat(key: str) -> bool:
    return key.endswith("running_mean") or key.endswith("running_var")


def fuse(paths: list[Path], nobn: bool) -> dict:
    states = [load_state(p) for p in paths]
    keys = set(states[0].keys())
    for i, st in enumerate(states[1:], 1):
        if set(st.keys()) != keys:
            miss = keys ^ set(st.keys())
            raise SystemExit(f"[FATAL] 第 {i+1} 个权重的键集合不同：{sorted(miss)[:10]}")
    out = {}
    n_avg = n_kept = 0
    for k in sorted(keys):
        ref = states[0][k]
        if is_p0(k) or (nobn and is_bn_stat(k)):
            out[k] = ref.clone() if torch.is_tensor(ref) else ref
            n_kept += 1
            continue
        if not torch.is_tensor(ref) or not ref.is_floating_point():
            out[k] = ref.clone() if torch.is_tensor(ref) else ref
            n_kept += 1
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for st in states:
            acc += st[k].to(torch.float64)
        out[k] = (acc / len(states)).to(ref.dtype)
        n_avg += 1
    print(f"[FUSE] 平均张量 {n_avg} 个 | 保留原值 {n_kept} 个"
          f"{'（含 BN 统计量与 p0）' if nobn else ''} | 来源 {len(states)} 个模型")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Model Soups：多 checkpoint 参数平均")
    ap.add_argument("--inputs", nargs="+", required=True, help="要平均的 .pt 路径")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nobn", action="store_true",
                    help="只平均可训练权重；BN running_mean/var 与 p0 取第一个模型")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"[FATAL] 权重不存在：{p}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    state = fuse(paths, args.nobn)
    ref_ckpt = torch.load(paths[0], map_location="cpu")
    ckpt = {
        "epoch": 0,
        "model": state,
        "metrics": {"soup": True, "n_models": len(paths), "nobn": bool(args.nobn)},
        "stride": ref_ckpt.get("stride", 2),
        "imgsz": ref_ckpt.get("imgsz", 640),
        "p0_kwargs": ref_ckpt.get("p0_kwargs", {}),
        "soup": {
            "name": args.name or out.stem,
            "sources": [str(p) for p in paths],
            "nobn": bool(args.nobn),
            "averaged_tensors": len(state),
        },
    }
    torch.save(ckpt, out)
    h = hashlib.md5(out.read_bytes()).hexdigest()
    meta = {
        "name": args.name or out.stem,
        "out": str(out),
        "md5": h,
        "bytes": out.stat().st_size,
        "sources": [str(p) for p in paths],
        "nobn": bool(args.nobn),
        "n_tensors": len(state),
    }
    (out.parent / (out.stem + ".json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[FUSE] 已写出 {out}")
    print(f"[FUSE] md5={h} | {out.stat().st_size:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
