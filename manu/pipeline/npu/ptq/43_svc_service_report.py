#!/usr/bin/env python3
"""Generate the SERVICE-ARCHITECTURE comparison table.

Reads only PRODUCED artifacts:
  * the pre-service baseline timing.txt (the old, non-service library)
  * one timing.txt per service arm
  * the detection TSV of each arm, for the byte-identity check

and derives:
  * the per-service critical path  (what the pipeline's throughput limit IS)
  * the scheduling overhead          (wall clock minus the critical path)
  * the GMC margin against the 40 ms/frame budget of a 25 fps source
  * the fallback counters            (dropped / deadline_blows)

Nothing here computes a metric of its own: every number is either read or
arithmetic on read numbers.

Usage:
  python 43_svc_service_report.py \
      --baseline <timing.txt of the pre-service run> \
      --arm NAME=<timing.txt>[:<detection.tsv>] ... \
      --baseline-tsv <certified detection.tsv> \
      --out <report.md>
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

# The 25 fps source: one frame every 40 ms.  This is the budget the GMC
# feature stage has to fit into, with the rest of the pipeline around it.
FRAME_BUDGET_MS = 1000.0 / 25.0


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_timing(path: Path) -> dict:
    """Read the e2e timing.txt: a segment table plus key=value lines."""
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and not parts[0].startswith("#"):
            try:
                out[parts[0]] = float(parts[2])
                out[parts[0] + "_total_ms"] = float(parts[1])
            except ValueError:
                continue
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def derive(t: dict) -> dict:
    """The per-service critical path of one arm."""
    npu_workers = int(t.get("npu_workers", 1))
    gmc_workers = int(t.get("gmc_workers", 1))
    # The PRE-SERVICE build had no POST service at all: the back-projection
    # ran INSIDE the NPU worker thread.  Its timing.txt therefore has no
    # `post_workers` key -- that absence is the discriminator.  Counting the
    # back-projection as a separate service there would UNDERSTATE its NPU
    # chain by 10.9 ms/frame and make the scheduling overhead look smaller
    # than it was.
    has_post_service = "post_workers" in t
    post_workers = int(t.get("post_workers", 1))

    # The NPU service thread does, in this order and serially:
    #   RGA import -> RGA improcess -> upload -> rknn_run -> outputs_get
    # (the back-projection moved OUT into the POST service.)
    npu_service_ms = (t.get("rga_import", 0.0) + t.get("rga_improcess", 0.0) +
                      t.get("upload", 0.0) + t.get("rknn_run", 0.0) +
                      t.get("outputs_get", 0.0))
    if not has_post_service:
        npu_service_ms += t.get("backproject_cc", 0.0)
    # The GMC service: gmc_total is the SUM over all engines, so the stream
    # only carries gmc_total / gmc_workers.
    gmc_service_ms = t.get("gmc_total", 0.0) / max(1, gmc_workers)
    # The POST service: one worker does the back-projection + connected
    # components for every frame.
    post_service_ms = (t.get("backproject_cc", 0.0) / max(1, post_workers)
                       if has_post_service else 0.0)

    npu_critical_ms = npu_service_ms / max(1, npu_workers)
    critical_ms = max(npu_critical_ms, gmc_service_ms, post_service_ms)
    wall = t.get("total_wall", 0.0)
    # The counters below were only added in the SERVICE build; the pre-service
    # run did not measure them, so they must show as "not measured" and NOT as
    # a zero (a zero would be a claim we did not earn).
    def opt(key):
        return t[key] if key in t else None
    return {
        "has_post_service": has_post_service,
        "npu_service_ms": npu_service_ms,
        "npu_critical_ms": npu_critical_ms,
        "gmc_service_ms": gmc_service_ms,
        "post_service_ms": post_service_ms,
        "critical_ms": critical_ms,
        "overhead_ms": wall - critical_ms,
        "wall_ms": wall,
        "fps": t.get("fps", 0.0),
        "gmc_margin_ms": FRAME_BUDGET_MS - gmc_service_ms,
        "gmc_margin_pct": 100.0 * (FRAME_BUDGET_MS - gmc_service_ms) / FRAME_BUDGET_MS,
        "gmc_budget_pct": 100.0 * gmc_service_ms / FRAME_BUDGET_MS,
        "gmc_fit_max_ms": opt("gmc_fit_max"),
        "gmc_fit_p50_ms": opt("gmc_fit_p50"),
        "gmc_fit_p95_ms": opt("gmc_fit_p95"),
        "gmc_fit_p99_ms": opt("gmc_fit_p99"),
        "gmc_corner_mean": opt("gmc_corner_mean"),
        "gmc_corner_capped": opt("gmc_corner_capped"),
        "dropped": opt("dropped"),
        "deadline_blows": opt("deadline_blows"),
        "npu_wait_ms": opt("npu_wait"),
        "gmc_wait_ms": opt("gmc_wait"),
        "post_wait_ms": opt("post_wait"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="the pre-service timing.txt")
    ap.add_argument("--baseline-tsv", default=None)
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME=timing.txt[:detection.tsv]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_t = parse_timing(Path(args.baseline))
    base = derive(base_t)

    arms: list[tuple[str, dict, dict, str | None]] = []
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        tsv_part = ""
        if ":" in rest:
            rest, _, tsv_part = rest.partition(":")
        t = parse_timing(Path(rest))
        arms.append((name, t, derive(t), tsv_part or None))

    lines: list[str] = []
    lines.append("# 服务化架构：四轮全量对照表\n")
    lines.append(f"> 全部数字读自板端产物 `timing.txt`；生成脚本 "
                 f"`manu/pipeline/npu/ptq/43_svc_service_report.py`\n")

    # ---- table S1: the per-service critical path ---------------------------
    cols: list[tuple[str, dict]] = [("服务化前（旧架构）", base)]
    cols += [(name, d) for name, _, d, _ in arms]

    lines.append("## 表 S1：关键路径推导（ms/帧）\n")
    lines.append("> ⚠️ **「服务化前」那一列的结构与其它列不同**：旧架构里**没有 POST 服务**，"
                 "反投影在 NPU worker 线程内部 ⇒ 它的「NPU 服务串行链」**含** `backproject_cc`，"
                 "「POST 服务」为空。脚本按 `post_workers` 字段是否存在来区分，不靠人工指定。\n")
    lines.append("| 段 / 服务 | " + " | ".join(c[0] for c in cols) + " |")
    lines.append("|" + "|".join(["--:"] * (len(cols) + 1)) + "|")

    def row(label, fn):
        lines.append("| " + label + " | " +
                     " | ".join(f"{fn(d):.3f}" for _, d in cols) + " |")

    row("**墙钟 total_wall**", lambda d: d["wall_ms"])
    row("**帧率 fps**", lambda d: d["fps"])
    row("NPU 服务：单服务串行链", lambda d: d["npu_service_ms"])
    row("　→ 除以 NPU 服务数 = 关键路径", lambda d: d["npu_critical_ms"])
    row("GMC 服务：引擎并行后每帧摊到", lambda d: d["gmc_service_ms"])
    row("POST 服务：反投影+连通域", lambda d: d["post_service_ms"])
    row("**理论关键路径 max(服务)**", lambda d: d["critical_ms"])
    row("**调度损耗（墙钟 − 关键路径）**", lambda d: d["overhead_ms"])
    lines.append("")

    # ---- table S2: the GMC margin against the 25 fps budget ----------------
    lines.append("## 表 S2：GMC 对 40 ms/帧预算（25 fps 源）的余量\n")
    lines.append(f"> 预算 = 1000/25 = {FRAME_BUDGET_MS:.1f} ms/帧。"
                 "GMC 服务占用 = `gmc_total / gmc_workers`（引擎并行后每帧摊到的）。"
                 "「—」= 该次运行没有测这个指标（旧架构没有这些计数器）。\n")
    lines.append("| 指标 | " + " | ".join(c[0] for c in cols) + " |")
    lines.append("|" + "|".join(["--:"] * (len(cols) + 1)) + "|")
    for label, key, fmt in [
        ("GMC 服务占用 ms/帧", "gmc_service_ms", "{:.3f}"),
        ("占 40 ms 预算比例", "gmc_budget_pct", "{:.1f}%"),
        ("**余量 ms/帧**", "gmc_margin_ms", "{:.3f}"),
        ("**余量比例**", "gmc_margin_pct", "{:.1f}%"),
        ("GMC 单帧拟合 p50 ms", "gmc_fit_p50_ms", "{:.3f}"),
        ("GMC 单帧拟合 p95 ms", "gmc_fit_p95_ms", "{:.3f}"),
        ("GMC 单帧拟合 p99 ms", "gmc_fit_p99_ms", "{:.3f}"),
        ("GMC 单帧最坏（毛刺）ms", "gmc_fit_max_ms", "{:.3f}"),
        ("角点均值（预算上限见头文件）", "gmc_corner_mean", "{:.1f}"),
        ("角点预算触顶帧数", "gmc_corner_capped", "{:.0f}"),
        ("NPU 饥饿 ms/帧", "npu_wait_ms", "{:.3f}"),
        ("GMC 饥饿 ms/帧", "gmc_wait_ms", "{:.3f}"),
        ("POST 饥饿 ms/帧", "post_wait_ms", "{:.3f}"),
        ("**丢帧 dropped**", "dropped", "{:.0f}"),
        ("**兜底触发 deadline_blows**", "deadline_blows", "{:.0f}"),
    ]:
        vals = []
        for _, d in cols:
            v = d[key]
            vals.append("—" if v is None else fmt.format(v))
        lines.append("| " + label + " | " + " | ".join(vals) + " |")
    lines.append("")

    # ---- table S3: the byte-identity check ---------------------------------
    if args.baseline_tsv:
        ref = md5(Path(args.baseline_tsv))
        lines.append("## 表 S3：检测结果逐字节一致性\n")
        lines.append(f"> 认证基准 `{Path(args.baseline_tsv).name}` md5 `{ref}`\n")
        lines.append("| 臂 | 检测 TSV md5 | 与认证基准 |")
        lines.append("|:--|:--|:--|")
        for name, _, _, tsv in arms:
            if not tsv:
                continue
            m = md5(Path(tsv))
            lines.append(f"| {name} | `{m}` | {'✅ **逐字节相同**' if m == ref else '❌ 不同'} |")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[ok] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
