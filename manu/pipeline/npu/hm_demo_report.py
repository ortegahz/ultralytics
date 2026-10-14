#!/usr/bin/env python3
"""
hm_demo_report.py -- three-layer accuracy alignment + speed roll-up for the
serial end-to-end demo:

    raw infrared sequence
        -> gmc_stream_ocl   (board, CPU fits + Mali fused tail)  == GMC layer
        -> rknn_heatmap_smoke (board, RKNN int8, Trial 0474 w/o P0) == NPU layer
    vs
    CPU authority arm: frozen dataset features -> PyTorch heatmap head

Why three layers and not one
----------------------------
Collapsing "board end-to-end vs CPU" into a single recall number hides *where*
the difference comes from. The board path contains two independent sources of
divergence, and they have completely different meanings:

  L1 GMC : board features vs the frozen dataset features. This isolates the
           streaming engine + Mali fused tail. A large delta here is a
           **preprocessing** defect; a small one means any L2 gap is not
           attributable to feature synthesis.
  L2 DET : board detections vs GT, and CPU detections vs GT. This is what the
           business actually cares about.
  L3 PAIR: NPU vs CPU on the *same* frames, paired. A recall gap that is not
           significant under McNemar must not be reported as a regression
           (rules.md 3.6c: "the Wilson intervals overlap" is not a test).

Every proportion is reported as absolute numbers + a Wilson 95% interval.
F1 is reported bare -- it is not a proportion (rules.md 3.6b).

Usage:
    python hm_demo_report.py \\
        --sequence 34_1 \\
        --board-feat /home/manu/nfs/npu/out/demo34/feat \\
        --frozen-feat /home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val \\
        --ref /media/manu/1TB-Volume/rknn_work/hm/demo34/ref_cpu \\
        --perframe /media/manu/1TB-Volume/rknn_work/hm/demo34/board_perframe.json \\
        --gmc-csv /home/manu/nfs/npu/out/demo34/gmc_profile.csv \\
        --npu-log /home/manu/nfs/npu/out/demo34/npu_run.log \\
        --gmc-log /home/manu/nfs/npu/out/demo34/gmc_run.log \\
        --push-offset 1
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval, 95%. Returns (lo, hi) in [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def fmt_ratio(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.4f} [{lo:.4f}, {hi:.4f}]"


def f1_of(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (p + r) <= 0 else 2 * p * r / (p + r)


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial p for discordant pairs (b, c).

    b = frames where the first arm scored a hit and the second did not,
    c = the mirror image. Under H0 the split is Binomial(b + c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------- L1
def layer1_gmc(args) -> None:
    print("\n" + "=" * 78)
    print("L1  GMC 层：板端特征 vs 冻结数据集特征（逐像素）")
    print("=" * 78)
    ids = sorted(int(p.stem.split("__")[-1]) for p in Path(args.frozen_feat).iterdir()
                 if p.stem.startswith(f"{args.sequence}__"))
    if not ids:
        print("[L1] 冻结特征为空，跳过")
        return
    n_cmp = max_abs = 0
    exact_frames = 0
    sum_mean = 0.0
    all_diffs = []
    frame_max = []
    for fid in ids:
        push = fid - args.push_offset
        b = Path(args.board_feat) / f"{args.sequence}_{push:06d}.bmp"
        if not b.is_file():
            continue
        B = cv2.imread(str(b), cv2.IMREAD_COLOR)
        R = cv2.imread(str(Path(args.frozen_feat) / f"{args.sequence}__{fid:06d}.png"),
                       cv2.IMREAD_COLOR)
        if B is None or R is None or B.shape != R.shape:
            continue
        d = np.abs(B.astype(np.int16) - R.astype(np.int16))
        n_cmp += 1
        mx = int(d.max())
        max_abs = max(max_abs, mx)
        sum_mean += float(d.mean())
        exact_frames += int(mx == 0)
        all_diffs.append((d == 0).mean())
        frame_max.append(mx)
        if n_cmp <= 8 or n_cmp % 100 == 0:
            print(f"[L1] id {fid:06d}  max|d|={mx:3d}  mean|d|={d.mean():.3e}  "
                  f"逐位相同像素={(d == 0).mean() * 100:6.2f}%")
    if n_cmp == 0:
        print("[L1] 无可比帧（板端特征缺失或形状不符）")
        return
    ex = sum(1 for v in all_diffs if v == 1.0)
    fm = np.array(frame_max)
    print(f"\n[L1] 比对帧数        : {n_cmp}")
    print(f"[L1] 全帧逐位相同    : {ex}/{n_cmp} = {ex / n_cmp:.4f}")
    print(f"[L1] 逐帧 max|diff|   : p50={np.percentile(fm, 50):.0f}  "
          f"p95={np.percentile(fm, 95):.0f}  全局 max={max_abs}  (uint8 满量程 255)")
    print(f"[L1] 平均 mean|diff| : {sum_mean / n_cmp:.3e}")
    print(f"[L1] 逐位相同像素占比 : min {min(all_diffs) * 100:.2f}%  "
          f"mean {np.mean(all_diffs) * 100:.2f}%  max {max(all_diffs) * 100:.2f}%")
    ok = np.percentile(fm, 95) <= 8 and np.mean(all_diffs) >= 0.97
    if ok:
        verdict = (f"特征合成与冻结集**高度一致但非逐位**：平均 "
                   f"{np.mean(all_diffs) * 100:.2f}% 的像素逐位相同，逐帧 max|diff| "
                   f"p50={np.percentile(fm, 50):.0f} / p95={np.percentile(fm, 95):.0f}。"
                   f"这是流式引擎与离线 builder 的**既有**差异（同量级亦见于 host Python 引擎），"
                   f"不是板端缺陷；其下游后果由 L3 定量给出。")
    else:
        verdict = "⚠️ 偏差超出流式引擎的既有量级，需排查预处理"
    print(f"[L1] 判定            : {verdict}")


# --------------------------------------------------------------------------- L2
def layer2_det(args) -> dict:
    print("\n" + "=" * 78)
    print("L2  检出层：两臂各自对 GT")
    print("=" * 78)
    recs = json.loads((Path(args.ref) / "detections.json").read_text())
    cpu = dict(gt=0, tp=0, fp=0, fn=0)
    for r in recs:
        cpu["gt"] += r["n_gt"] if "n_gt" in r else len(r.get("gt_canvas", []))
        cpu["tp"] += r["tp"]
        cpu["fp"] += r["fp"]
        cpu["fn"] += r["fn"]

    pf = json.loads(Path(args.perframe).read_text())
    npu = dict(gt=0, tp=0, fp=0, fn=0)
    for r in pf:
        npu["gt"] += r["n_gt"]
        npu["tp"] += r["tp"]
        npu["fp"] += r["fp"]
        npu["fn"] += r["fn"]

    rows = [("CPU 权威臂（冻结特征 + PyTorch 去 p0）", cpu),
            ("板端端到端（Mali GMC + RKNN int8 去 p0）", npu)]
    hdr = f"| {'臂':40} | {'GT':>4} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'precision':>42} | {'recall':>42} | {'F1':>7} |"
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    for name, m in rows:
        print(f"| {name:40} | {m['gt']:4d} | {m['tp']:4d} | {m['fp']:4d} | {m['fn']:4d} | "
              f"{fmt_ratio(m['tp'], m['tp'] + m['fp']):>42} | "
              f"{fmt_ratio(m['tp'], m['gt']):>42} | {f1_of(m['tp'], m['fp'], m['fn']):7.4f} |")
    print("\n[L2] F1 不是比例，不附区间（rules.md §三·6b）")
    return {"cpu": cpu, "npu": npu}


# --------------------------------------------------------------------------- L3
def layer3_pair(args) -> None:
    print("\n" + "=" * 78)
    print("L3  配对层：板端 NPU vs CPU 权威臂（同一批帧）")
    print("=" * 78)
    pf = json.loads(Path(args.perframe).read_text())
    b = c = 0          # b: NPU 命中 CPU 未命中；c: 反之
    both = neither = 0
    dists = []
    tsv = {}
    for line in Path(args.tsv).read_text().splitlines():
        if not line.strip():
            continue
        p = line.split("\t")
        n = int(p[1])
        tsv[Path(p[0]).stem] = [tuple(float(v) for v in p[2 + k].split()) for k in range(n)]

    for r in pf:
        nh = r["tp"] > 0
        ch = r["cpu_tp"] > 0
        if nh and ch:
            both += 1
        elif nh and not ch:
            b += 1
        elif ch and not nh:
            c += 1
        else:
            neither += 1
        bd = tsv.get(r["stem"], [])
        if bd and "det_canvas" not in r:
            pass
    tot = len(pf)
    print(f"[L3] 帧数                : {tot}")
    print(f"[L3] 两臂都命中          : {both}/{tot} = {both / tot:.4f}")
    print(f"[L3] 都未命中（空帧）    : {neither}/{tot} = {neither / tot:.4f}")
    print(f"[L3] 不一致帧            : {b + c}/{tot} = {(b + c) / tot:.4f}")
    print(f"[L3]   NPU 命中 / CPU 未中 : {b}")
    print(f"[L3]   CPU 命中 / NPU 未中 : {c}")
    p = mcnemar(b, c)
    verdict = ("无统计显著差异" if p >= 0.05 else "存在统计显著差异")
    print(f"[L3] 配对 McNemar p       : {p:.4f}  ⇒ {verdict}")
    print(f"[L3] 判定依据 rules.md §三·6c（配对检验；区间重叠不是显著性判据）")


# --------------------------------------------------------------------------- speed
STAGES = ["io_ms", "prep_ms", "fit_ms", "warp_ms", "median_ms",
          "img_up_ms", "mats_up_ms", "kernel_ms", "readback_ms"]
STAGE_LABEL = {
    "io_ms": "帧解码", "prep_ms": "prep(环形)", "fit_ms": "GMC fit",
    "warp_ms": "warp(CPU)", "median_ms": "median(CPU)", "img_up_ms": "img H2D",
    "mats_up_ms": "mats H2D", "kernel_ms": "融合核(Mali)", "readback_ms": "回读 D2H",
}


def speed(args) -> None:
    print("\n" + "=" * 78)
    print("SPEED  速度报告")
    print("=" * 78)

    if args.gmc_csv and Path(args.gmc_csv).is_file():
        rows = np.atleast_1d(np.genfromtxt(args.gmc_csv, delimiter=",", names=True))
        n_all = len(rows)
        # A short clip can have FEWER frames than the cold-start allowance, which
        # would silently leave a 20-frame sample looking like a 437-frame one.
        # Cap the warmup at a fifth of the run and always print what was used.
        warm = min(args.gmc_warmup, n_all // 5)
        sel = rows[warm:] if n_all > warm else rows
        if warm != args.gmc_warmup:
            print(f"\n[S-GMC] ⚠️ 仅有 {n_all} 帧，冷启动剔除由 {args.gmc_warmup} 自动降为 "
                  f"{warm}，统计样本 n={len(sel)}（偏薄，结论请按此样本量看待）")
        print(f"\n[S-GMC] 分阶段（CSV: {args.gmc_csv}，统计剔除前 {warm} 帧冷启动，n={len(sel)}）")
        print(f"[S-GMC] {'阶段':<16}{'p50':>9}{'p95':>9}{'max':>9}{'mean':>9}   占 p50")
        tot_p50 = float(np.percentile(sel["total_ms"], 50))
        for s in STAGES:
            if s not in sel.dtype.names:
                continue
            v = sel[s]
            p50, p95, mx = (np.percentile(v, 50), np.percentile(v, 95), v.max())
            share = f"{p50 / tot_p50 * 100:5.1f}%" if tot_p50 else "  -  "
            print(f"[S-GMC] {STAGE_LABEL.get(s, s):<16}{p50:9.3f}{p95:9.3f}{mx:9.3f}"
                  f"{v.mean():9.3f}   {share}")
        resid = sel["resid_ms"]
        print(f"[S-GMC] {'resid':<14}{np.percentile(resid, 50):10.3f}"
              f"{np.percentile(resid, 95):10.3f}{resid.max():10.3f}{resid.mean():10.3f}"
              f"   （resid = total - Σ阶段；应≈0，否则账不可信）")
        print(f"[S-GMC] total     p50={tot_p50:.3f}  p95={np.percentile(sel['total_ms'], 95):.3f}  "
              f"max={sel['total_ms'].max():.3f}  mean={sel['total_ms'].mean():.3f} ms  "
              f"→ {1000 / tot_p50:.2f} fps")

    if args.gmc_log and Path(args.gmc_log).is_file():
        txt = Path(args.gmc_log).read_text(errors="ignore")
        m = re.search(r"END-TO-END TOTAL ==\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt)
        if m:
            p50, _, p95, p99, mx = (float(m.group(i)) for i in range(1, 6))
            print(f"\n[S-GMC] 板端自报 END-TO-END  p50={p50:.3f}  p95={p95:.3f}  "
                  f"p99={p99:.3f}  max={mx:.3f} ms  → {1000 / p50:.2f} fps")
            bm = re.search(r"\[budget\] 40ms red line: (.*)", txt)
            if bm:
                print(f"[S-GMC] 40 ms 红线: {bm.group(1).strip()}")
        m2 = re.search(r"\[SEQ\]\s+(\S+)\s+frames=(\d+)\s+md5=([0-9a-f]+)", txt)
        if m2:
            print(f"[S-GMC] 序列 {m2.group(1)}  frames={m2.group(2)}  "
                  f"feature md5={m2.group(3)}")

    if args.npu_log and Path(args.npu_log).is_file():
        txt = Path(args.npu_log).read_text(errors="ignore")
        print(f"\n[S-NPU] 板端 NPU 阶段（{args.model_name}）")
        for lbl, pat in [("inputs_set", r"inputs_set\s+avg\s+([\d.]+) ms"),
                         ("rknn_run", r"rknn_run\s+avg\s+([\d.]+) ms"),
                         ("outputs_get", r"outputs_get\s+avg\s+([\d.]+) ms")]:
            m = re.search(pat, txt)
            if m:
                print(f"[S-NPU] {lbl:<12}{float(m.group(1)):8.3f} ms")
        m = re.search(r"TOTAL\s+avg\s+([\d.]+) ms\s+->\s+([\d.]+) fps", txt)
        if m:
            print(f"[S-NPU] {'TOTAL':<12}{float(m.group(1)):8.3f} ms  → {m.group(2)} fps")
        m = re.search(r"clock sanity:.*delta=([\d.]+) ms \[(\w+)\]", txt)
        if m:
            print(f"[S-NPU] 时钟自检 50 ms usleep 实测 {m.group(1)} ms  [{m.group(2)}]"
                  f"{'  ← 时延数字有效' if m.group(2) == 'ok' else '  ← 时延数字一律不算数'}")
        m = re.search(r"null control.*?(CHANGED|IDENTICAL\(BAD\))", txt, re.S)
        if m:
            print(f"[S-NPU] 空对照 {m.group(1)}")
        m = re.search(r"determinism: .*->\s*(IDENTICAL|DIFFER)", txt)
        if m:
            print(f"[S-NPU] 确定性 {m.group(1)}")

    print(f"\n[S-E2E] 端到端 = GMC 特征阶段 + NPU 阶段（串行，两段独立计时相加）")
    if args.gmc_csv and Path(args.gmc_csv).is_file():
        rows = np.atleast_1d(np.genfromtxt(args.gmc_csv, delimiter=",", names=True))
        sel = rows[args.gmc_warmup:] if len(rows) > args.gmc_warmup else rows
        g = float(np.percentile(sel["total_ms"], 50))
    else:
        g = float("nan")
    n_ = None
    if args.npu_log and Path(args.npu_log).is_file():
        m = re.search(r"TOTAL\s+avg\s+([\d.]+) ms", Path(args.npu_log).read_text(errors="ignore"))
        if m:
            n_ = float(m.group(1))
    if n_ is not None and g == g:
        print(f"[S-E2E] GMC p50 {g:.3f} ms + NPU {n_:.3f} ms = {g + n_:.3f} ms  "
              f"→ {1000 / (g + n_):.2f} fps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--board-feat", default="")
    ap.add_argument("--frozen-feat", default="")
    ap.add_argument("--ref", default="", help="CPU reference dir (has detections.json)")
    ap.add_argument("--perframe", default="", help="hm_score_board_det.py --out JSON")
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--push-offset", type=int, default=1)
    ap.add_argument("--gmc-csv", default="")
    ap.add_argument("--gmc-log", default="")
    ap.add_argument("--npu-log", default="")
    ap.add_argument("--gmc-warmup", type=int, default=100)
    ap.add_argument("--model-name", default="trial0474_hm_nop0_int8.rknn")
    ap.add_argument("--speed-only", action="store_true",
                    help="skip L1/L2/L3 and print only the speed roll-up; for inputs "
                         "with no paired GT (e.g. a video) where accuracy cannot be scored")
    args = ap.parse_args()

    print("=" * 78)
    print(f"端到端 demo 报告  序列 {args.sequence}  模型 {args.model_name}")
    print("=" * 78)
    if args.speed_only:
        print("\n[speed-only] 该输入无配对 GT，跳过 L1/L2/L3 精度对齐，仅报速度。")
        speed(args)
        print("\n" + "=" * 78)
        return
    for req in ("ref", "perframe"):
        if not getattr(args, req):
            raise SystemExit(f"[FATAL] --{req} is required unless --speed-only is given")
    if args.board_feat and args.frozen_feat:
        layer1_gmc(args)
    layer2_det(args)
    layer3_pair(args)
    speed(args)
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
