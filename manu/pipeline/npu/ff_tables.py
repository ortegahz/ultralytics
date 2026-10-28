#!/usr/bin/env python3
"""Emit the final markdown tables for the GMC full-frame GPU probe report."""
import sys
sys.path.insert(0, '/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu')
from ff_summarize import load, stats, pctl

D = '/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu/out/ff/'
RUNS = [
    ("hard600", "最难窗口 (abs 551)", 600, 1),
    ("hard500", "最难窗口 (abs 551)", 500, 1),
    ("hard450", "最难窗口 (abs 551)", 450, 1),
    ("hard600_par5", "最难窗口 (abs 551)", 600, 5),
    ("normal600", "普通窗口 (abs 3149)", 600, 1),
    ("normal500", "普通窗口 (abs 3149)", 500, 1),
    ("normal450", "普通窗口 (abs 3149)", 450, 1),
]
EXTRA = [("hard600_new", "最难窗口 (abs 551)", 600, 1),
         ("hard600_nocache", "最难窗口 (abs 551) 无缓存", 600, 1)]


def row(name, hdr, rows, key, label):
    v = [float(r[key]) for r in rows]
    s = stats(v)
    return (f"| {label} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | "
            f"{s['max']:.2f} | {s['fps']:.2f} | {s['over40']:.1f}% |")


def load_run(tag):
    try:
        hdr, rows = load(D + "ff_%s.tsv" % tag)
        return hdr, rows
    except Exception:
        return None, None


print("### 表 1 — 速度：CPU 路径 vs GPU 路径\n")
print("| 窗口 | 角点预算 | 路径 | mean | p50 | p95 | p99 | max | fps | >40ms 占比 |")
print("| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
cache = {}
for tag, win, budget, cput in RUNS + EXTRA:
    hdr, rows = load_run(tag)
    if rows is None:
        print(f"| {win} | {budget} | ({tag}: MISSING) | | | | | | | |")
        continue
    cache[tag] = (hdr, rows)
    cpu_lbl = "CPU (OpenCV 单线程)" if cput == 1 else f"CPU (锚点并行 {cput} 线程)"
    print(row(tag, hdr, rows, "cpu_total", f"{win} | {budget} | {cpu_lbl}"))
    print(row(tag, hdr, rows, "gpu_total", f"{win} | {budget} | **GPU 全帧**"))

print("\n### 表 2 — GPU 每帧时间分解 (mean ms)\n")
print("| 窗口 | 角点预算 | resize | 角点(gFTT) | 角点(CPU选取) | 金字塔 | 上传 | LK批量 | LK回读 | RANSAC | 合计 |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for tag, win, budget, cput in RUNS + EXTRA:
    if tag not in cache:
        continue
    hdr, rows = cache[tag]
    g = lambda c: sum(float(r[c]) for r in rows) / len(rows)
    lkd = g("gpu_lk_dispatch") if "gpu_lk_dispatch" in hdr else g("gpu_lk_batch")
    lkr = g("gpu_lk_read") if "gpu_lk_read" in hdr else 0.0
    print(f"| {win} | {budget} | {g('gpu_resize'):.2f} | {g('gpu_gftt'):.2f} | {g('gpu_select'):.2f} | "
          f"{g('gpu_pyr'):.2f} | {g('gpu_upload'):.2f} | {lkd:.2f} | {lkr:.2f} | "
          f"{g('gpu_ransac'):.2f} | **{g('gpu_total'):.2f}** |")

print("\n### 表 3 — 批量 vs 顺序 dispatch (LK 阶段, 同一份已上传金字塔)\n")
print("| 窗口 | 角点 | 1 锚点批量 (同 kernel, nfits=1) | 6 拟合批量 (1 次 NDRange) | 6 次顺序 dispatch | 批量/顺序 | 批量(6)/批量(1) |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
for tag, win, budget, cput in RUNS + EXTRA:
    if tag not in cache:
        continue
    hdr, rows = cache[tag]
    g = lambda c: sum(float(r[c]) for r in rows) / len(rows)
    b1, bb, sq = g("gpu_lk_1anchor"), g("gpu_lk_batch"), g("gpu_lk_seq")
    print(f"| {win} | {budget} | {b1:.2f} | **{bb:.2f}** | {sq:.2f} | {sq/bb:.2f}x | {bb/b1:.2f}x |")

print("\n### 表 4 — 尾延迟归因 (>40 ms 的帧 vs 其余帧, GPU 分解 mean ms)\n")
print("| 窗口 | 角点 | LK批量 (慢帧) | LK批量 (快帧) | RANSAC (慢帧) | RANSAC (快帧) | 慢帧 LK 占比 |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
for tag, win, budget, cput in RUNS:
    if tag not in cache:
        continue
    hdr, rows = cache[tag]
    slow = [r for r in rows if float(r["gpu_total"]) > 40.0]
    fast = [r for r in rows if float(r["gpu_total"]) <= 40.0]
    if not slow or not fast:
        continue
    m = lambda rs, c: sum(float(r[c]) for r in rs) / len(rs)
    print(f"| {win} | {budget} | {m(slow,'gpu_lk_batch'):.2f} | {m(fast,'gpu_lk_batch'):.2f} | "
          f"{m(slow,'gpu_ransac'):.2f} | {m(fast,'gpu_ransac'):.2f} | "
          f"{100*m(slow,'gpu_lk_batch')/m(slow,'gpu_total'):.0f}% |")

print("\n### 表 5 — 角点预算 → p99 曲线\n")
print("| 窗口 | 角点预算 | corner_mean | 撞顶帧数 | GPU mean | GPU p95 | GPU p99 | GPU max | GPU fps | >40ms |")
print("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
for tag, win, budget, cput in RUNS:
    if tag not in cache:
        continue
    hdr, rows = cache[tag]
    v = [float(r["gpu_total"]) for r in rows]
    s = stats(v)
    cc = [float(r["ncorners_gpu"]) for r in rows]
    capped = sum(1 for x in cc if x >= budget - 0.5)
    print(f"| {win} | {budget} | {sum(cc)/len(cc):.1f} | {capped}/{len(cc)} | {s['mean']:.2f} | "
          f"{s['p95']:.2f} | {s['p99']:.2f} | {s['max']:.2f} | {s['fps']:.2f} | {s['over40']:.1f}% |")
