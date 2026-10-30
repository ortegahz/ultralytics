#!/usr/bin/env python3
"""compare_e2e_timing.py -- 三个后处理模式的 e2e timing.txt 逐段对比。

用法： compare_e2e_timing.py <dense.txt> <sparse.txt> <gpu.txt>
"""
import re
import sys


def load(path):
    out = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
        elif len(parts) == 1:
            out[parts[0]] = None
        # "key=value" 形式（后半段的标量指标）
        if "=" in line and " " not in line.split("=")[0]:
            k, _, v = line.partition("=")
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    # timing.txt 的第二段是 "segment total_ms avg_ms/frame"
    seg = {}
    rows = []
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    in_seg = False
    for line in lines:
        if line.startswith("segment"):
            in_seg = True
            continue
        if in_seg:
            p = line.split()
            if len(p) == 3:
                try:
                    seg[p[0]] = (float(p[1]), float(p[2]))
                except ValueError:
                    pass
            elif line.strip() == "":
                in_seg = False
    out["__seg__"] = seg
    return out


KEYS = [
    ("fps", "端到端 fps", 1),
    ("total_wall", "total_wall ms/帧", 1),
    ("backproject_cc", "后处理 backproject_cc ms/帧", 1),
    ("post_wait", "POST 线程等待 ms/帧", 1),
    ("gmc_total", "GMC 合计 ms/帧", 1),
    ("gmc_fit", "GMC 拟合 ms/帧", 1),
    ("gmc_ingest_in", "GMC ingest 输入 ms/帧", 1),
    ("rknn_run", "rknn_run ms/帧", 1),
    ("outputs_get", "outputs_get ms/帧", 1),
    ("rga_import", "rga_import ms/帧", 1),
    ("rga_improcess", "rga_improcess ms/帧", 1),
    ("stage1_ms", "Stage1(pre) ms/帧", 1),
    ("stage1_wait_ms", "Stage1 等待 ms/帧", 1),
    ("stage2_ms", "Stage2(NPU) ms/帧", 1),
    ("stage2_wait_ms", "Stage2 等待 ms/帧", 1),
    ("bufA_upload_ms", "BufferA 上传 ms/帧", 1),
    ("npu_busy_ms", "NPU 忙 ms/帧", 1),
    ("npu_util_pct", "NPU 利用率 %", 1),
    ("bufA_dropped", "BufferA 丢弃", 1),
    ("dropped", "相机丢帧", 1),
    ("features_dropped", "特征丢弃", 1),
    ("detections_dropped", "检测丢弃", 1),
    ("gmc_groups_done", "GMC 组数", 1),
    ("gmc_frames_skipped", "GMC 跳过帧", 1),
    ("deadline_blows", "deadline 超限", 1),
    ("gmc_history_gaps", "历史缺口", 1),
    ("e2e_latency_p50", "端到端延时 p50 ms", 1),
    ("e2e_latency_p95", "端到端延时 p95 ms", 1),
    ("e2e_latency_p99", "端到端延时 p99 ms", 1),
    ("e2e_latency_max", "端到端延时 max ms", 1),
]


def main():
    paths = sys.argv[1:4]
    names = ["dense", "sparse", "gpu"]
    data = [load(p) for p in paths]
    print("| 指标 | dense | sparse | gpu | sparse vs dense | gpu vs dense |")
    print("|:--|--:|--:|--:|--:|--:|")
    for key, label, _ in KEYS:
        vals = [d.get(key) for d in data]
        if all(v is None for v in vals):
            continue
        cells = []
        for v in vals:
            cells.append("n/a" if v is None else f"{v:.4f}" if key == "fps" else f"{v:.1f}")
        deltas = []
        for v in vals[1:]:
            b = vals[0]
            if v is None or b is None or b == 0:
                deltas.append("—")
            else:
                deltas.append(f"{(v - b) / b * 100:+.1f}%")
        print(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {deltas[0]} | {deltas[1]} |")


if __name__ == "__main__":
    main()
