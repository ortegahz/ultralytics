#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板卡稳温 / 零降频监控器（RK3588 / Orange Pi CM5）。

做两件事：

1. **持续采样**（默认 2 s 一次）：全部 thermal zone 温度、全部 cooling device 的
   ``cur_state``、全部 cpufreq policy 的 min/max/cur、全部 devfreq 的 cur/max/target、
   CPU 每核利用率、系统负载、内存。原始样本写成一张宽表 CSV。
2. **每分钟打印一次关键状态摘要**（``--print-interval``，默认 60 s）：
   温度、当前频率 vs 允许的最高频率、**是否发生降频**。

降频判据（三条都算「降频」，任一命中即置位）：

* **A. cooling 动作**：任一 ``/sys/class/thermal/cooling_device*/cur_state`` > 0
  —— 内核 thermal 框架已经开始压频/压流。
* **B. 频率低于上限**：任一 cpufreq ``scaling_cur_freq`` < ``scaling_max_freq``，
  或任一 devfreq ``cur_freq`` < ``max_freq``。
* **C. 内核事件**：``dmesg`` 里出现新的 thermal / throttle / cooling 相关行。

用法（板卡上）::

    python3 board_thermal_monitor.py --duration 3600 --interval 2 --print-interval 60 \
        --out /tmp/thermal_run --label newboard_sink_1h

只读：全程只读 /sys 与 /proc，不写任何系统节点，不会自己触发或解除降频。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from typing import Any

# dmesg 里与「温控动作」有关的行（判据 C）
DMSG_RE = re.compile(
    r"thermal|throttl|cooling|trip point|tsadc|scmi|pvtm|devfreq.*(down|limit)",
    re.IGNORECASE,
)


def _read(path: str) -> str | None:
    """读一个 sysfs/procfs 文件，失败返回 None（节点可能热插拔）。"""
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    raw = _read(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_float(path: str) -> float | None:
    raw = _read(path)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def discover() -> dict[str, Any]:
    """发现板卡上的温控/频率拓扑。拓扑在测试期间不变，只在启动时做一次。"""
    zones: list[dict[str, Any]] = []
    for zpath in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        ztype = _read(os.path.join(zpath, "type")) or os.path.basename(zpath)
        zones.append({"path": zpath, "name": ztype, "trip": _read(os.path.join(zpath, "trip_point_0_temp"))})

    cdevs: list[dict[str, Any]] = []
    for cpath in sorted(glob.glob("/sys/class/thermal/cooling_device*")):
        cdevs.append({"path": cpath, "type": _read(os.path.join(cpath, "type")) or os.path.basename(cpath)})

    policies: list[dict[str, Any]] = []
    for ppath in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
        policies.append(
            {
                "path": ppath,
                "name": os.path.basename(ppath),
                "cpus": _read(os.path.join(ppath, "affected_cpus")) or "",
            }
        )

    devfreqs: list[dict[str, Any]] = []
    for dpath in sorted(glob.glob("/sys/class/devfreq/*")):
        devfreqs.append(
            {
                "path": dpath,
                "name": os.path.basename(dpath),
                "real": _read(os.path.join(dpath, "name")) or os.path.basename(dpath),
            }
        )

    fans: list[dict[str, Any]] = []
    for hpath in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if _read(os.path.join(hpath, "name")) == "pwmfan":
            fans.append({"path": hpath, "name": "pwmfan"})

    # PVTM 的热天花板：设备树 OPP 表里的 rockchip,high-temp-max-freq。
    # SoC 温度 >= rockchip,high-temp 时，PVTM（rockchip-system-monitor）通过 PM QoS
    # 把各域的频率上限压到这个 ceiling —— 这是「真降频」，比 cooling device 更早、更隐蔽。
    opps: list[dict[str, Any]] = []
    for t in ("cluster0", "cluster1", "cluster2", "gpu", "npu"):
        d = "/sys/firmware/devicetree/base/%s-opp-table" % t
        if not os.path.isdir(d):
            continue
        opps.append(
            {
                "name": t,
                "dir": d,
                "high_temp": _read_be32(os.path.join(d, "rockchip,high-temp")),
                "ceiling": _read_be32(os.path.join(d, "rockchip,high-temp-max-freq")),
            }
        )

    # 各域的「额定上限」（PVTM 未介入时应等于它）
    nominal: dict[str, int] = {}
    for p in policies:
        v = _read_int(os.path.join(p["path"], "cpuinfo_max_freq"))
        if v is not None:
            nominal["cpu_" + p["name"]] = v
    for dpath, nom in (("/sys/class/devfreq/fb000000.gpu", 1000000000),
                       ("/sys/class/devfreq/fdab0000.npu", 1000000000),
                       ("/sys/class/devfreq/dmc", 2112000000)):
        if os.path.isdir(dpath):
            nominal["dv_" + os.path.basename(dpath)] = nom

    gpu_util_path = "/sys/devices/platform/fb000000.gpu/utilisation"
    gpu_util_period = "/sys/devices/platform/fb000000.gpu/utilisation_period"
    rknpu_load_path = "/sys/kernel/debug/rknpu/load"

    return {
        "zones": zones,
        "cdevs": cdevs,
        "policies": policies,
        "devfreqs": devfreqs,
        "fans": fans,
        "opps": opps,
        "nominal": nominal,
        "gpu_util_path": gpu_util_path,
        "gpu_util_period": gpu_util_period,
        "rknpu_load_path": rknpu_load_path,
    }


def _read_be32(path: str) -> int | None:
    """读设备树里的大端 u32 属性（rockchip,high-temp 等）。"""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(4)
    except OSError:
        return None
    if len(raw) < 4:
        return None
    return int.from_bytes(raw, "big")


def read_cpu_stat() -> tuple[list[int], int]:
    """读 /proc/stat 的 CPU 行，返回 (每核 jiffies 累计, 总 jiffies)。"""
    lines = _read("/proc/stat") or ""
    per_cpu: list[int] = []
    total = 0
    for line in lines.splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        vals = [int(p) for p in parts[1:] if p.isdigit()]
        if not vals:
            continue
        s = sum(vals)
        total += s
        if parts[0] == "cpu":
            continue
        per_cpu.append(s)
    return per_cpu, total


def cpu_util(prev_per: list[int], prev_tot: int, cur_per: list[int], cur_tot: int) -> list[float]:
    """两次采样之间的每核利用率（%）。"""
    out: list[float] = []
    dt = cur_tot - prev_tot
    for i, cur in enumerate(cur_per):
        old = prev_per[i] if i < len(prev_per) else 0
        d = cur - old
        out.append(0.0 if dt <= 0 else 100.0 * d / dt)
    return out


def dmesg_tail_marker() -> int:
    """返回 dmesg 的行数，作为「读到哪儿了」的游标。"""
    out = subprocess.run(["dmesg"], capture_output=True, text=True, errors="replace")
    return len(out.stdout.splitlines())


def dmesg_new_lines(cursor: int) -> tuple[list[str], int]:
    """返回 (cursor 之后的新行, 新的 cursor)。"""
    out = subprocess.run(["dmesg"], capture_output=True, text=True, errors="replace")
    lines = out.stdout.splitlines()
    if cursor >= len(lines):
        return [], cursor
    new = lines[cursor:]
    return new, len(lines)


def sample(topo: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """采一个样本。返回 (本样本的扁平字典, 本样本命中的降频原因列表)。"""
    row: dict[str, Any] = {}
    reasons: list[str] = []

    for z in topo["zones"]:
        t = _read_int(os.path.join(z["path"], "temp"))
        row["temp_" + z["name"]] = (t / 1000.0) if t is not None else None

    for c in topo["cdevs"]:
        cur = _read_int(os.path.join(c["path"], "cur_state"))
        row["cdev_" + c["type"]] = cur
        if cur is not None and cur > 0:
            reasons.append("cooling device %s cur_state=%d (>0 ⇒ 内核已在压频)" % (c["type"], cur))

    for p in topo["policies"]:
        cur = _read_int(os.path.join(p["path"], "scaling_cur_freq"))
        mx = _read_int(os.path.join(p["path"], "scaling_max_freq"))
        mn = _read_int(os.path.join(p["path"], "scaling_min_freq"))
        nom = _read_int(os.path.join(p["path"], "cpuinfo_max_freq"))
        row["cpu_%s_cur" % p["name"]] = cur
        row["cpu_%s_max" % p["name"]] = mx
        row["cpu_%s_min" % p["name"]] = mn
        row["cpu_%s_nominal" % p["name"]] = nom
        row["cpu_%s_gov" % p["name"]] = _read(os.path.join(p["path"], "scaling_governor"))
        if cur is not None and mx is not None and cur < mx:
            reasons.append("CPU %s 频率 %d < 上限 %d" % (p["name"], cur, mx))
        # ---- PVTM 钳位：scaling_max_freq 被压到额定值以下 ----
        if mx is not None and nom is not None and mx < nom:
            reasons.append("PVTM 钳位 CPU %s：scaling_max=%d < 额定 %d" % (p["name"], mx, nom))
        for o in topo["opps"]:
            if o["ceiling"] and mx is not None and mx == o["ceiling"] and nom is not None and mx < nom:
                row["cpu_%s_pvtm_ceiling" % p["name"]] = o["ceiling"]

    for d in topo["devfreqs"]:
        cur = _read_int(os.path.join(d["path"], "cur_freq"))
        mx = _read_int(os.path.join(d["path"], "max_freq"))
        tgt = _read_int(os.path.join(d["path"], "target_freq"))
        nom = topo["nominal"].get("dv_" + d["name"])
        row["dv_%s_cur" % d["name"]] = cur
        row["dv_%s_max" % d["name"]] = mx
        row["dv_%s_tgt" % d["name"]] = tgt
        row["dv_%s_nominal" % d["name"]] = nom
        row["dv_%s_gov" % d["name"]] = _read(os.path.join(d["path"], "governor"))
        row["dv_%s_load" % d["name"]] = _read_int(os.path.join(d["path"], "load"))
        if cur is not None and mx is not None and cur < mx:
            reasons.append("%s 频率 %d < 上限 %d" % (d["real"], cur, mx))
        if mx is not None and nom is not None and mx < nom:
            reasons.append("PVTM 钳位 %s：max_freq=%d < 额定 %d" % (d["real"], mx, nom))

    # ---- 三个域的实时负载 ----
    gpu_u = _read_int(topo["gpu_util_path"])
    row["gpu_util"] = gpu_u
    row["gpu_util_period_ms"] = _read_int(topo["gpu_util_period"])
    rk = _read(topo["rknpu_load_path"]) or ""
    for i in range(3):
        m = re.search(r"Core%d:\s*(\d+)%%" % i, rk)
        row["rknpu_core%d" % i] = int(m.group(1)) if m else None

    for f in topo["fans"]:
        row["fan_pwm"] = _read_int(os.path.join(f["path"], "pwm1"))

    loadavg = (_read("/proc/loadavg") or "").split()
    if len(loadavg) >= 3:
        row["load1"] = float(loadavg[0])
        row["load5"] = float(loadavg[1])
        row["load15"] = float(loadavg[2])
    mem = _read("/proc/meminfo") or ""
    for key, name in (("MemAvailable:", "mem_avail_kb"), ("MemFree:", "mem_free_kb"), ("SwapTotal:", "swap_total_kb"), ("SwapFree:", "swap_free_kb")):
        for line in mem.splitlines():
            if line.startswith(key):
                row[name] = int(line.split()[1])
                break

    return row, reasons


def fmt_khz(v: int | None) -> str:
    """cpufreq 的单位是 kHz。"""
    if v is None:
        return "?"
    return "%.0fMHz" % (v / 1000.0)


def fmt_hz(v: int | None) -> str:
    """devfreq 的单位是 Hz。"""
    if v is None:
        return "?"
    return "%.0fMHz" % (v / 1000000.0)


def pvtm_hit(elapsed: float, reasons: list[str], stop_log: list, args) -> None:
    """PVTM（rockchip-system-monitor）是否已经钳住了频率上限。

    判据：``sample()`` 产出的原因里出现 "PVTM 钳位" —— 即 cpufreq 的
    ``scaling_max_freq`` 被压到 ``cpuinfo_max_freq`` 以下，或 devfreq 的
    ``max_freq`` 被压到额定值以下。PVTM 走 PM QoS，用户态写回会被它覆盖，
    是「真降频」，比 cooling device 更早、更隐蔽。
    """
    pvtm = [r for r in reasons if "PVTM 钳位" in r]
    if pvtm and not getattr(args, "_pvtm_logged", False):
        args._pvtm_logged = True
        stop_log.append((elapsed, "PVTM: " + " | ".join(pvtm)))
        print("[%7.0fs] !! PVTM 已钳住频率上限: %s" % (elapsed, " | ".join(pvtm)), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="RK3588 板卡稳温 / 零降频监控器")
    ap.add_argument("--duration", type=float, default=3600.0, help="总时长（秒），默认 3600 = 1 小时")
    ap.add_argument("--interval", type=float, default=2.0, help="采样间隔（秒），默认 2")
    ap.add_argument("--print-interval", type=float, default=60.0, help="关键状态打印间隔（秒），默认 60")
    ap.add_argument("--out", default="/tmp/thermal_run", help="输出前缀（CSV + 文本摘要）")
    ap.add_argument("--label", default="", help="本次测试的标签")
    ap.add_argument(
        "--stop-cmd",
        default="",
        help="确认降频后执行的命令（用于立刻停掉压测负载）；空 = 不执行",
    )
    ap.add_argument(
        "--throttle-confirm",
        type=int,
        default=3,
        help="连续多少个采样点都判定降频才确认（防瞬时误报），默认 3",
    )
    args = ap.parse_args()

    topo = discover()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    csv_path = args.out + "_samples.csv"
    txt_path = args.out + "_summary.txt"

    soc = None
    for z in topo["zones"]:
        if z["name"] == "soc-thermal":
            soc = z["name"]
    if soc is None and topo["zones"]:
        soc = topo["zones"][0]["name"]

    print("== 板卡稳温 / 零降频监控器 ==")
    print("  标签      : %s" % (args.label or "-"))
    print("  时长      : %.0f s (采样 %.1f s, 打印 %.0f s)" % (args.duration, args.interval, args.print_interval))
    print("  thermal   : %s" % ", ".join(z["name"] for z in topo["zones"]))
    print("  cooling   : %s" % ", ".join(c["type"] for c in topo["cdevs"]))
    print("  cpufreq   : %s" % ", ".join(p["name"] for p in topo["policies"]))
    print("  devfreq   : %s" % ", ".join(d["name"] for d in topo["devfreqs"]))
    print("  输出      : %s , %s" % (csv_path, txt_path))
    print("  起始温度  : %s" % _read(os.path.join(next(z["path"] for z in topo["zones"] if z["name"] == soc), "temp")))
    print("  开始监控 ...", flush=True)

    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    all_reasons: list[tuple[float, str]] = []
    dmesg_events: list[tuple[float, str]] = []
    cursor = dmesg_tail_marker()

    throttle_streak = 0
    stop_fired = False
    pvtm_events: list[tuple[float, str]] = []
    stop_log: list[tuple[float, str]] = []

    prev_per, prev_tot = read_cpu_stat()
    t0 = time.monotonic()
    wall0 = time.time()
    next_print = t0 + args.print_interval
    n = 0

    while True:
        now = time.monotonic()
        elapsed = now - t0
        if elapsed > args.duration:
            break

        row, reasons = sample(topo)
        cur_per, cur_tot = read_cpu_stat()
        util = cpu_util(prev_per, prev_tot, cur_per, cur_tot)
        prev_per, prev_tot = cur_per, cur_tot
        row["t_s"] = round(elapsed, 2)
        row["wall"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for i, u in enumerate(util):
            row["cpu%d_util" % i] = round(u, 1)

        new_lines, cursor = dmesg_new_lines(cursor)
        for ln in new_lines:
            if DMSG_RE.search(ln):
                dmesg_events.append((elapsed, ln.strip()))
                reasons.append("dmesg: " + ln.strip()[:120])

        if reasons:
            all_reasons.append((elapsed, " | ".join(reasons)))
        rows.append(row)
        n += 1

        # ---- 降频确认（连续 N 个采样点）+ 触发停负载 ----
        if reasons:
            throttle_streak += 1
        else:
            throttle_streak = 0
        pvtm_hit(elapsed, reasons, stop_log, args)
        if throttle_streak >= max(1, args.throttle_confirm) and not stop_fired:
            stop_fired = True
            stop_log.append((elapsed, " | ".join(reasons)))
            print(
                "[%7.0fs] !! 连续 %d 个采样点判定降频，确认触发：%s"
                % (elapsed, throttle_streak, " | ".join(reasons)),
                flush=True,
            )
            if args.stop_cmd:
                print("[%7.0fs] !! 执行 stop-cmd: %s" % (elapsed, args.stop_cmd), flush=True)
                try:
                    subprocess.run(args.stop_cmd, shell=True, timeout=60)
                except Exception as exc:  # noqa: BLE001
                    print("[%7.0fs] !! stop-cmd 执行异常: %s" % (elapsed, exc), flush=True)

        if now >= next_print or elapsed > args.duration:
            # ---- 每分钟关键状态打印 ----
            soc_t = row.get("temp_" + soc)
            gpu_t = row.get("temp_gpu-thermal")
            npu_t = row.get("temp_npu-thermal")
            cpu_cur = "/".join(
                "%s" % fmt_khz(row.get("cpu_%s_cur" % p["name"])) for p in topo["policies"]
            )
            cpu_max = "/".join(
                "%s" % fmt_khz(row.get("cpu_%s_max" % p["name"])) for p in topo["policies"]
            )
            dv_parts = []
            for d in topo["devfreqs"]:
                if d["name"] in ("fb000000.gpu", "fdab0000.npu"):
                    dv_parts.append(
                        "%s %s(max %s)"
                        % (d["real"], fmt_hz(row.get("dv_%s_cur" % d["name"])), fmt_hz(row.get("dv_%s_max" % d["name"])))
                    )
            cdev_bad = [c["type"] for c in topo["cdevs"] if (row.get("cdev_" + c["type"]) or 0) > 0]
            throttled = bool(reasons)
            tag = "🔴降频" if throttled else "🟢正常"
            # 三域负载
            utils = [row.get("cpu%d_util" % i) for i in range(8) if row.get("cpu%d_util" % i) is not None]
            cpu_load = (sum(utils) / len(utils)) if utils else 0.0
            gpu_load = row.get("gpu_util")
            npu_load = row.get("dv_fdab0000.npu_load")
            # PVTM 是否已钳住频率上限
            pvtm = []
            for p in topo["policies"]:
                mx, nom = row.get("cpu_%s_max" % p["name"]), row.get("cpu_%s_nominal" % p["name"])
                if mx is not None and nom is not None and mx < nom:
                    pvtm.append("CPU%s" % p["name"])
            for d in topo["devfreqs"]:
                mx, nom = row.get("dv_%s_max" % d["name"]), row.get("dv_%s_nominal" % d["name"])
                if mx is not None and nom is not None and mx < nom:
                    pvtm.append(d["real"])
            pvtm_s = ("已触发:" + ",".join(pvtm)) if pvtm else "未触发"
            print(
                "[%7.0fs] %s | 温度 soc=%.1f°C gpu=%.1f°C npu=%.1f°C | CPU %s (max %s) | %s | "
                "负载 CPU=%.0f%% GPU=%s%% NPU=%s%% | cooling=%s | PVTM:%s | load=%.2f | %s"
                % (
                    elapsed,
                    time.strftime("%H:%M:%S", time.localtime()),
                    soc_t if soc_t is not None else -1,
                    gpu_t if gpu_t is not None else -1,
                    npu_t if npu_t is not None else -1,
                    cpu_cur,
                    cpu_max,
                    " ".join(dv_parts),
                    cpu_load,
                    ("%d" % gpu_load) if gpu_load is not None else "?",
                    ("%d" % npu_load) if npu_load is not None else "?",
                    ("!" + ",".join(cdev_bad)) if cdev_bad else "0",
                    pvtm_s,
                    row.get("load1", 0.0),
                    tag,
                ),
                flush=True,
            )
            if throttled:
                for r in reasons:
                    print("        ↳ %s" % r, flush=True)
            next_print = now + args.print_interval

        time.sleep(args.interval)

    # ---- 落盘 ----
    if rows:
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
        with open(csv_path, "w") as fh:
            fh.write(",".join(columns) + "\n")
            for r in rows:
                fh.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in columns) + "\n")

    temps = [r.get("temp_" + z["name"]) for r in rows for z in topo["zones"] if r.get("temp_" + z["name"]) is not None]
    tail_start = len(rows) // 2
    tail_temps = [
        r.get("temp_" + z["name"])
        for r in rows[tail_start:]
        for z in topo["zones"]
        if r.get("temp_" + z["name"]) is not None
    ]
    with open(txt_path, "w") as fh:
        fh.write("标签: %s\n" % (args.label or "-"))
        fh.write("开始: %s  结束: %s  时长: %.1f s  样本数: %d\n" % (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall0)),
            time.strftime("%Y-%m-%d %H:%M:%S"),
            time.monotonic() - t0, n))
        fh.write("采样 CSV: %s\n\n" % csv_path)
        fh.write("=== 各温度区统计 (°C) ===\n")
        for z in topo["zones"]:
            vals = [r.get("temp_" + z["name"]) for r in rows if r.get("temp_" + z["name"]) is not None]
            if not vals:
                continue
            fh.write("  %-18s min=%.1f  max=%.1f  last=%.1f\n" % (z["name"], min(vals), max(vals), vals[-1]))
        fh.write("\n=== 降频事件 ===\n")
        if all_reasons:
            for t, why in all_reasons[:200]:
                fh.write("  [%7.0fs] %s\n" % (t, why))
            fh.write("  合计 %d 个采样点出现降频迹象\n" % len(all_reasons))
        else:
            fh.write("  无 —— 全程没有任何降频迹象\n")
        if dmesg_events:
            fh.write("\n=== dmesg 温控事件 ===\n")
            for t, ln in dmesg_events:
                fh.write("  [%7.0fs] %s\n" % (t, ln))
        else:
            fh.write("\n=== dmesg 温控事件 ===\n  无\n")
        pvtm_lines = [x for x in stop_log if x[1].startswith("PVTM:")]
        if pvtm_lines:
            fh.write("\n=== PVTM 钳位事件 ===\n")
            for t, ln in pvtm_lines:
                fh.write("  [%7.0fs] %s\n" % (t, ln))
        else:
            fh.write("\n=== PVTM 钳位事件 ===\n  无 —— 全程未触发 PVTM 降额\n")
        fh.write("\n=== 后半程（稳态）温度极差 ===\n")
        if tail_temps:
            fh.write("  min=%.1f  max=%.1f  极差=%.2f\n" % (min(tail_temps), max(tail_temps), max(tail_temps) - min(tail_temps)))

    print("")
    print("== 监控结束 ==")
    print("  样本 %d 个 -> %s" % (n, csv_path))
    print("  摘要      -> %s" % txt_path)
    print("  降频事件  : %d 个采样点" % len(all_reasons))
    print("  PVTM 钳位 : %s" % ("已触发" if any(x[1].startswith("PVTM:") for x in stop_log) else "未触发"))
    if temps:
        print("  最高温度  : %.1f °C" % max(temps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
