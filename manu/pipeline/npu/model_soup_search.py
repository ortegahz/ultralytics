#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨臂 / 跨格权重汤（Cross-Arm / Cross-Cell Model Soup）搜索器 —— **零训练成本**。

背景（2026-11-13 周末计划）：
  台账里 12 个 triad 臂（nano/nano4/f1 三项同时过线）在六项判据上**互补**：
    · A1/A2 家族（12519@ep3 / 12679@ep3 / 12694@ep5 / 12709@ep4）：t1/虚警全过，**只差 t3 1.1e-4~7.8e-4**
    · B2 家族（12053@ep4）：nano/nano4/f1/t2_fp/t3 全过，**只差 t1 9.8e-4**
  ⇒ 若同构权重可线性插值，把 A 家族的 t1 优势与 B2 家族的 t3 优势叠加 ⇒ 可能直接产出**六项全过**候选。
  推理图与 trial_0031 / 9863 **逐字节同构** ⇒ RKNN 时延零变化（不破板卡 T4 红线）。

⚠️ 历史教训（memory §142.1，2026-11-11 实测）：当时的跨臂 Soup **崩盘**
   （soup_multiseed 同 sw 跨种子 → nano 0.2422；soup_dim_top2 → 0.3046；soup_top3 → 0.6545）
   ⇒ 原因：**不同配方 / 不同种子的权重不在同一 loss 盆地**。
   本脚本的目标：**用正确的成员选择（A 家族 × B2 家族 × 9863 基线本体）+ 正确 α 网格**重新验证，
   并把结论写进报告（若仍崩盘 ⇒ 该轴正式封死，不再占用周末算力）。

用法（服务器）：
    PYTHONPATH=. python manu/pipeline/npu/model_soup_search.py --mode pairs --alphas 0.2,0.35,0.5,0.65,0.8
    PYTHONPATH=. python manu/pipeline/npu/model_soup_search.py --mode greedy
    PYTHONPATH=. python manu/pipeline/npu/model_soup_search.py --mode grid --alphas 0.25,0.5,0.75 --gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = "/tmp/pycharm_project_10ae9e2e"
sys.path.insert(0, REPO)

ROOT = "/mnt/data/siping/runs/npu20ms_offcal"
RUNS = "/mnt/data/siping/runs/npu20ms"
SYNTH_ROOT = os.path.join(RUNS, "soup")
LEDGER = os.path.join(RUNS, "nano_search", "nano_search_ledger.json")

BASE = dict(nano=0.708648393194707, nano4=0.6472117202268431,
            f1=0.8897117275427806, t1=0.888086, t2_fp=5, t3=0.948489)

# 12 个 triad 臂（唯一有资格升轨）：(trial, epoch, 家族标签)
TRIAD = [
    (9863, 5, "B1/base"),
    (12053, 4, "B2"),
    (12519, 3, "A1"),
    (12520, 4, "A1"),
    (12600, 3, "A1"),
    (12679, 3, "A2"),
    (12694, 5, "A2"),
    (12709, 4, "A2"),
    (12252, 3, "A6"),
    (12263, 5, "A3"),
    (12300, 3, "A5"),
    (12333, 3, "B1"),
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ckpt_path(trial: int, ep: int) -> str:
    return f"{ROOT}/trials/trial_{trial}/weights/epoch{ep}.pt"


def family(trial: int, ep: int, tag: str) -> str:
    return tag


def make_soup(members: list[tuple[int, int]], weights: list[float], name: str) -> str:
    """加权平均 ⇒ 落到 SYNTH_ROOT/<name>/weights/epoch5.pt（可直接当普通臂用）。"""
    from manu.pipeline.npu.ckpt_synth import average_checkpoints  # noqa: PLC0415
    paths = [ckpt_path(t, e) for t, e in members]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"缺 ckpt：{missing}")
    out_dir = os.path.join(SYNTH_ROOT, name)
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
    out = os.path.join(out_dir, "weights", "epoch5.pt")
    if not os.path.isfile(out):
        if len(paths) == 1:                      # 单成员：不平均，直接复制（口径校验用）
            import shutil
            shutil.copy2(paths[0], out)
        else:
            average_checkpoints(paths, out, weights=weights, epoch=5,
                                note="cross-arm-soup:" + "+".join(f"{t}@{e}" for t, e in members))
    return out


def evaluate(ck: str, name: str, gpu: int) -> dict | None:
    """建 no-GMC 缓存 + 跑 NanoBench ⇒ 三项判据（与主流水线逐位同口径）。"""
    from manu.pipeline.npu import route2_acceptance as RA  # noqa: PLC0415
    pkl = os.path.join(RUNS, "tiny_ref", f"soup_{name}.pkl")
    if not RA.build_cache(ck, pkl, gpu, extra="--no-peak-attrs"):
        log(f"[{name}] 建缓存失败")
        return None
    res = RA.nanobench(pkl, f"soup_{name}")
    if not res:
        return None
    rb8 = res["peak_tol8"]["recall_by_bucket"]
    rb4 = res["peak_tol4"]["recall_by_bucket"]
    out = {"f1": res["peak_tol8"]["f1"], "nano": rb8["nano"], "micro": rb8["micro"],
           "small": rb8["small"], "regular": rb8["regular"],
           "nano4": rb4["nano"], "f1_4": res["peak_tol4"]["f1"], "cache": pkl}
    out["triad"] = bool(out["nano"] >= BASE["nano"] and out["nano4"] >= BASE["nano4"]
                        and out["f1"] >= BASE["f1"])
    return out


def greedy_soup(cands: list[tuple[str, list[tuple[int, int]], list[float]]], gpu: int,
                 report: dict) -> list[dict]:
    """Greedy Soup：按单成员 nano 排序逐个试加入；**只有三项仍全过才保留**。"""
    keep: list[tuple[tuple[int, int], float]] = []
    results = []
    for name, members, weights in cands:
        trial_members = keep + [members[0]]
        trial_weights = [w for _, w in keep] + [1.0]
        sname = "greedy_" + name
        ck = make_soup(trial_members, trial_weights, sname)
        r = evaluate(ck, sname, gpu)
        if not r:
            continue
        r.update({"name": sname, "members": [f"{t}@{e}" for t, e in trial_members]})
        results.append(r)
        log(f"[greedy] {sname} nano={r['nano']:.6f} nano4={r['nano4']:.6f} f1={r['f1']:.6f} "
            f"triad={'Y' if r['triad'] else 'n'}")
        if r["triad"]:
            keep.append((members[0], 1.0))
    report["greedy"] = results
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pairs", choices=["pairs", "greedy", "single"])
    ap.add_argument("--alphas", default="0.35,0.5,0.65")
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(RUNS, "nano_search", "model_soup_report.json"))
    ap.add_argument("--members", default="", help="覆盖默认成员：trial:ep,trial:ep,...")
    args = ap.parse_args()
    sys.path.insert(0, REPO)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    if args.members:
        mem = []
        for tok in args.members.split(","):
            t, e = tok.split(":")
            mem.append((int(t), int(e)))
    else:
        mem = [(t, e) for t, e, _ in TRIAD]
    tag = {t: g for t, _e, g in TRIAD}

    report = {"mode": args.mode, "alphas": alphas, "members": [f"{t}@{e}" for t, e in mem],
              "base": BASE, "results": [], "ts": time.strftime("%F %T")}
    log(f"成员：{report['members']}")

    cands: list[tuple[str, list[tuple[int, int]], list[float]]] = []
    if args.mode == "single":
        for t, e in mem:
            cands.append((f"single_{t}_{e}", [(t, e)], [1.0]))
    elif args.mode == "greedy":
        # 单成员基线（按 nano 排序）→ 逐个加入
        singles = []
        for t, e in mem:
            ck = ckpt_path(t, e)
            if os.path.isfile(ck):
                singles.append((t, e))
        cands = [(f"m{t}", [(t, e)], [1.0]) for t, e in singles]
    else:
        # 两两：A 家族 × B2/基线，以及基线 × A 家族
        base = [(9863, 5)]
        pairs = []
        for (t1, e1) in mem:
            if (t1, e1) == (9863, 5):
                continue
            pairs.append(((9863, 5), (t1, e1)))
        for (t1, e1) in mem:
            if (t1, e1) in ((9863, 5), (12053, 4)):
                continue
            pairs.append(((12053, 4), (t1, e1)))
        for (a, b) in pairs:
            for al in alphas:
                cands.append((f"s_{a[0]}_{b[0]}_a{al}", [a, b], [1.0 - al, al]))

    if args.mode == "greedy":
        greedy_soup(cands, args.gpu, report)
    else:
        for name, members, weights in cands:
            ck = make_soup(members, weights, name)
            r = evaluate(ck, name, args.gpu)
            if not r:
                continue
            r.update({"name": name, "members": [f"{t}@{e}" for t, e in members],
                       "weights": weights,
                       "families": [tag.get(t, "?") for t, _ in members]})
            report["results"].append(r)
            log(f"[{name}] {' + '.join(r['members'])} w={weights} "
                f"nano={r['nano']:.6f} nano4={r['nano4']:.6f} f1={r['f1']:.6f} "
                f"triad={'Y' if r['triad'] else 'n'}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=2)
    log(f"报告已写出 {args.out}")
    ok = [r for r in report.get("results", []) if r.get("triad")]
    log(f"triad 命中：{len(ok)} / {len(report.get('results', []))}")
    for r in ok:
        log(f"  ★ {r['name']} {r.get('members')} nano={r['nano']:.6f} nano4={r['nano4']:.6f} f1={r['f1']:.6f}")


if __name__ == "__main__":
    main()
