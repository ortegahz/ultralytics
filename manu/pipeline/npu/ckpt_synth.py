#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""权重合成器：把若干同构 checkpoint 的浮点参数做加权平均（SWA / Model Soup）。

用途（2026-11-11 自适应搜索）：**不训练**就能开一条新搜索轴 ——
Model Soup（不同超参/种子的同构权重平均）与 SWA（同一臂相邻 epoch 的权重平均）。

约束：
  * 只平均**同拓扑**的 ckpt（no_p2/head_level/act/head_conv/late_upsample/stride 必须一致），
    否则权重不可加；调用方负责保证同构，本模块会**自检并拒绝**结构不一致的组合；
  * 整数张量（如 BN 的 num_batches_tracked）取第一个，不参与平均；
  * 输出沿用第一个 ckpt 的结构元数据（ckpt 自描述字段），并附 `synth` 溯源块。

用法：
  from ckpt_synth import average_checkpoints
  average_checkpoints(["a.pt", "b.pt"], "out.pt", note="soup2")
"""
import os
import sys


def _struct_sig(ck: dict) -> tuple:
    return (ck.get("no_p2"), ck.get("head_level"), ck.get("act"),
            ck.get("head_conv"), ck.get("late_upsample"), ck.get("stride"))


def average_checkpoints(paths, out_path: str, weights=None, epoch=None, note: str = "") -> str:
    """加权平均若干 ckpt 的 model state_dict，写到 out_path。返回 out_path。"""
    import torch  # 延迟导入：本模块的其余部分不需要 torch

    paths = [p for p in paths if os.path.isfile(p)]
    if len(paths) < 2:
        raise ValueError(f"至少需要 2 个存在的 ckpt，得到 {paths}")
    cks = [torch.load(p, map_location="cpu") for p in paths]
    sigs = {_struct_sig(c) for c in cks}
    if len(sigs) > 1:
        raise ValueError(f"ckpt 结构不一致，禁止平均：{ {s: [p for p, c in zip(paths, cks) if _struct_sig(c) == s] for s in sigs} }")

    states = []
    for c in cks:
        s = c["model"] if "model" in c else c.get("state_dict", c)
        if hasattr(s, "state_dict"):
            s = s.state_dict()
        states.append(s)

    w = list(weights) if weights else [1.0] * len(states)
    if len(w) != len(states):
        raise ValueError("weights 长度与 ckpt 数不一致")
    tot = float(sum(w))

    keys = set(states[0].keys())
    for s in states[1:]:
        keys &= set(s.keys())  # 只平均公共键

    avg = {}
    for k in sorted(keys):
        vals = [s[k] for s in states]
        if not torch.is_tensor(vals[0]):
            avg[k] = vals[0]
            continue
        if vals[0].is_floating_point():
            acc = vals[0].float() * w[0]
            for v, wi in zip(vals[1:], w[1:]):
                acc = acc + v.float() * wi
            avg[k] = (acc / tot).to(vals[0].dtype)
        else:
            avg[k] = vals[0].clone()

    out = dict(cks[0])
    out["model"] = avg
    if epoch is not None:
        out["epoch"] = epoch
    out["synth"] = {"note": note, "members": list(paths), "weights": list(w),
                     "struct": _struct_sig(cks[0])}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    print(average_checkpoints(sys.argv[2:-1], sys.argv[-1], note="cli"))
