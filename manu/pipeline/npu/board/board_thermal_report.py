#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从主机侧观测器的采样 CSV 生成稳温/零降频测试报告（markdown 表格）。

输入：`board_host_watch.py` 产出的 `*_samples.csv`
      （**按列名解析**，探针增删列不会错位；列名见观测脚本的 PROBE_FIELDS）
输出：stdout 为 markdown，可直接贴进报告；`--out` 可落盘。

用法：
    python3 manu/pipeline/npu/board/board_thermal_report.py \
        --csv /tmp/bts_1h_r5_samples.csv --dmesg /tmp/bts_1h_r5_dmesg.txt \
        --label newboard_1h_1client_cpu8 --per-minute
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

# thermal_zone 实测布局（Orange Pi CM5 / RK3588）
#   0 soc-thermal(PVTM 温度源) 1/2 bigcore0/1 3 littlecore 4 center 5 gpu 6 npu
TEMP_COLS = ["soc_c", "soc_max_c", "gpu_c", "npu_c", "little_c", "big0_c", "big1_c", "center_c"]
CD_COLS = ["cd0", "cd1", "cd2", "cd3"]
# (显示名, 当前频率列, 上限列, 额定值列或常量)
FREQ_DOMAINS = [
    ("CPU0 (A55 簇)", "cpu0_cur", "cpu0_max", "nom0"),
    ("CPU4 (A76 簇1)", "cpu4_cur", "cpu4_max", "nom4"),
    ("CPU6 (A76 簇2)", "cpu6_cur", "cpu6_max", "nom6"),
    ("NPU", "npu_cur", "npu_max", "npu_rated"),
    ("GPU", "gpu_cur", "gpu_max", "gpu_rated"),
    ("DMC", "dmc_cur", "dmc_max", "dmc_rated"),
]
# PVTM 硬天花板（设备树 OPP 表 rockchip,high-temp-max-freq，内核 PM QoS 钳位值）
PVTM_CEILING = {"CPU0 (A55 簇)": 1608000, "CPU4 (A76 簇1)": 2208000,
                "CPU6 (A76 簇2)": 2208000, "NPU": 800000000,
                "GPU": 800000000, "DMC": 2112000000}
RATED_CONST = {"npu_rated": 1000000000, "gpu_rated": 1000000000, "dmc_rated": 2112000000}

TRIP_CRIT = 115.0
TRIP_PASSIVE_DEFAULT = 85.0
PVTM_THRESHOLD_DEFAULT = 85.0


def fmt(value, digits=1):
    return "n/a" if value is None else ("%.*f" % (digits, value))


def ols_slope(xs, ys):
    """最小二乘斜率 y = a*x + b；返回 (slope, r2)。样本不足或 x 无变化时返回 (None, None)。"""
    n = len(xs)
    if n < 10:
        return None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None, None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = None if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return a, r2


def stat(values):
    if not values:
        return None
    s = sorted(values)
    return {"min": s[0], "max": s[-1], "avg": sum(s) / len(s),
            "p95": s[max(0, int(len(s) * 0.95) - 1)]}


def load_rows(path: Path):
    """按列名解析；返回 (rows[list[dict]], short_count, header)"""
    lines = path.read_text(errors="replace").splitlines()
    if not lines:
        return [], 0, []
    header = lines[0].split(",")
    rows, short = [], 0
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != len(header):
            short += 1
            continue
        rows.append(dict(zip(header, parts)))
    return rows, short, header


def num(row, name):
    v = row.get(name)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dmesg", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-minute", action="store_true")
    ap.add_argument("--steady-after", type=float, default=600.0,
                    help="稳态统计起点（相对 t_s 秒），默认前 10 分钟算预热")
    ap.add_argument("--trip-passive", type=float, default=TRIP_PASSIVE_DEFAULT,
                    help="thermal_zone0 passive trip 阈值(℃)，用于算余量")
    ap.add_argument("--pvtm-threshold", type=float, default=PVTM_THRESHOLD_DEFAULT,
                    help="PVTM 硬天花板(℃)，用于算余量")
    ap.add_argument("--boot", default="",
                    help="只统计指定 boot_id 的那一段（默认取样本最多的那一段）")
    args = ap.parse_args()

    rows, short, header = load_rows(Path(args.csv))
    if not rows:
        print("没有可用样本", file=sys.stderr)
        return 1
    has = lambda name: name in header          # noqa: E731

    # ---- 按 boot_id 分段 ------------------------------------------------------
    # 板卡复位会把 CSV 切成两段。把两段混在一起统计，峰值/极差/斜率全废
    # （"复位前的 75℃" 减 "复位后的 32℃" ⇒ 极差 43℃，毫无意义）。
    segs = []
    for r in rows:
        b = r.get("boot_id", "")
        if segs and segs[-1][0] == b:
            segs[-1][1].append(r)
        else:
            segs.append((b, [r]))
    seg_desc = ""
    if len(segs) > 1:
        if args.boot:
            pick = [s for s in segs if s[0].startswith(args.boot)]
            if not pick:
                print("找不到 boot 段 %s（可选：%s）" %
                      (args.boot, ", ".join(s[0][:8] for s in segs)), file=sys.stderr)
                return 2
            rows = pick[0][1]
        else:
            rows = max((s[1] for s in segs), key=len)
        seg_desc = ("**⚠️ CSV 含 %d 个 boot 段（板卡发生过复位）**：%s\n\n"
                    "**本表只统计选中的那一段**（boot `%s`，%d 个样本，%s → %s）。\n" %
                    (len(segs),
                     "；".join("`%s` %d 样本 %s→%s" %
                               (b[:8], len(rs), rs[0]["wall"], rs[-1]["wall"]) for b, rs in segs),
                     rows[0].get("boot_id", "")[:8], len(rows),
                     rows[0]["wall"], rows[-1]["wall"]))

    # 实测采样间隔（不写死 2 s：探针间隔会变，写死会让"最后 N 个样本"的秒数差一倍）
    t_first, t_last = num(rows[0], "t_s"), num(rows[-1], "t_s")
    interval = ((t_last - t_first) / (len(rows) - 1)) if (len(rows) > 1 and t_first is not None
                                                       and t_last is not None) else 2.0

    out = []
    a = out.append
    if seg_desc:
        a(seg_desc)
        a("")
    a("### 采样概览（label=%s）" % (args.label or "-"))
    a("")
    a("| 项 | 值 |")
    a("|---|---|")
    a("| 有效样本 | %d（间隔 %.1f s） |" % (len(rows), interval))
    a("| 覆盖时长 | %.1f s（t_s %.1f → %.1f） |" %
      (num(rows[-1], "t_s"), num(rows[0], "t_s"), num(rows[-1], "t_s")))
    a("| 墙钟区间 | %s → %s |" % (rows[0]["wall"], rows[-1]["wall"]))
    a("| boot_id（本表所统计的段） | %s |" % (rows[0].get("boot_id", "") or "n/a"))
    a("| 探针不完整行 | %d |" % short)
    a("| 采样列数 | %d |" % len(header))
    a("")

    # ---------------- 温度 ----------------
    a("### 温度（℃）")
    a("")
    a("温控阈值配置：passive trip = %.1f℃（运行时；DT 原值 75℃ 已由开机启动项 "
      "`thermal-trip.service` 抬到 85℃，**固件未改**）；PVTM 硬天花板 = %.1f℃；critical = %.1f℃" %
      (args.trip_passive, args.pvtm_threshold, TRIP_CRIT))
    a("")
    a("| 域 | 最小 | 均值 | P95 | **峰值** | 稳态段峰值(t≥%.0fs) | 到 passive %.0f℃ 余量 | 到 PVTM %.0f℃ 余量 | 超 PVTM 样本数 |" %
      (args.steady_after, args.trip_passive, args.pvtm_threshold))
    a("|---|---|---|---|---|---|---|---|---|")
    for name in TEMP_COLS:
        if not has(name):
            continue
        vals = [v / 1000 for v in (num(r, name) for r in rows) if v is not None]
        if not vals:
            continue
        st = stat(vals)
        steady = [num(r, name) / 1000 for r in rows
                  if (num(r, "t_s") or 0) >= args.steady_after and num(r, name) is not None]
        sst = stat(steady)
        over = sum(1 for v in vals if v > args.pvtm_threshold)
        a("| %s | %s | %s | %s | **%s** | %s | %s | %s | %d |" % (
            name, fmt(st["min"]), fmt(st["avg"]), fmt(st["p95"]), fmt(st["max"]),
            fmt(sst["max"]) if sst else "n/a",
            fmt(args.trip_passive - st["max"]), fmt(args.pvtm_threshold - st["max"]), over))
    a("")
    a("#### 温度稳定性判据（斜率 + 平台）")
    a("")
    a("只看「极差」不够 —— 缓慢爬升的曲线极差也可能很小。这里对**稳态段**（t_s ≥ %.0f s）"
      "做最小二乘拟合：斜率 ≈ 0 才叫稳定；若斜率显著为正，说明还在爬升、并未到热平衡。" %
      args.steady_after)
    a("")
    a("| 温度通道 | 稳态段样本 | 稳态均值 | 稳态极差 | **斜率** | 折合 | R² | 到达平台时刻（进入峰值 1℃ 带内） |")
    a("|---|---|---|---|---|---|---|---|")
    for tag, col in (("SoC (zone0, PVTM 温度源)", "soc_c"),
                     ("全板最热区 (soc_max_c)", "soc_max_c"),
                     ("GPU (zone5)", "gpu_c"), ("NPU (zone6)", "npu_c")):
        if not has(col):
            continue
        pts = [(num(r, "t_s"), num(r, col) / 1000) for r in rows
               if (num(r, "t_s") or -1) >= args.steady_after and num(r, col) is not None]
        if len(pts) < 10:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        slope, r2 = ols_slope(xs, ys)
        peak = max(ys)
        plateau = next((t for t, y in pts if y >= peak - 1.0), None)
        a("| %s | %d | %s | %s | **%s** | %s | %s | %s |" % (
            tag, len(pts), fmt(sum(ys) / len(ys)), fmt(max(ys) - min(ys)),
            "%+.4f ℃/s" % slope if slope is not None else "n/a",
            "%+.2f ℃/小时" % (slope * 3600) if slope is not None else "n/a",
            fmt(r2, 3) if r2 is not None else "n/a",
            "t=%.0f s" % plateau if plateau is not None else "未进入"))
    a("")
    for tag, col in (("SoC (soc-thermal，PVTM 温度源)", "soc_c"),
                     ("全板最热区 (soc_max_c)", "soc_max_c")):
        if not has(col):
            continue
        vals = [v / 1000 for v in (num(r, col) for r in rows) if v is not None]
        if not vals:
            continue
        tail = vals[-900:] if len(vals) >= 900 else vals
        a("**%s 稳定性**：全程极差 **%.1f℃**（min %.1f / max %.1f / 均值 %.1f）；"
          "最后 %d 个样本(约 %.0f s)极差 **%.1f℃**。" %
          (tag, max(vals) - min(vals), min(vals), max(vals), sum(vals) / len(vals),
           len(tail), len(tail) * interval, max(tail) - min(tail)))
    a("")

    # ---------------- 降频 ----------------
    a("### 降频判定（逐域实际频率 vs 额定值）")
    a("")
    a("| 域 | 额定频率 | 实测 cur 最小/均值/最大 | 低于额定的样本数 | PVTM 天花板 | 判定 |")
    a("|---|---|---|---|---|---|")
    any_low = False
    for label, cur_c, max_c, rated_c in FREQ_DOMAINS:
        if not (has(cur_c) and has(max_c)):
            continue
        rated = num(rows[0], rated_c) if rated_c in RATED_CONST else None
        if rated is None:
            rated = next((num(r, rated_c) for r in rows if num(r, rated_c)), None)
        if rated is None:
            rated = RATED_CONST.get(rated_c)
        cur_vals = [v for v in (num(r, cur_c) for r in rows) if v is not None]
        if not cur_vals or not rated:
            continue
        low = sum(1 for v in cur_vals if v < rated)
        max_low = sum(1 for r in rows
                      if (num(r, max_c) is not None) and (num(r, max_c) < rated))
        any_low = any_low or low or max_low
        ceiling = PVTM_CEILING.get(label)
        verdict = "✅ 全程满频" if (low == 0 and max_low == 0) else "❌ 发生降频"
        # cpufreq 的频率单位是 kHz，devfreq 是 Hz
        k = 1e3 if label.startswith("CPU") else 1e6
        a("| %s | %s | %s / %s / %s | %d | %s | %s |" % (
            label, fmt(rated / k, 3) + " MHz",
            fmt(min(cur_vals) / k, 3), fmt(stat(cur_vals)["avg"] / k, 3),
            fmt(max(cur_vals) / k, 3),
            low, fmt(ceiling / k, 3) + " MHz" if ceiling else "n/a", verdict))
    a("")
    cd_hits = [(r["wall"], c, num(r, c)) for r in rows for c in CD_COLS
               if (num(r, c) or 0) > 0]
    pvtm_col = "pvtm"
    pvtm_hits = [(r["wall"], r.get(pvtm_col, "")) for r in rows
                 if r.get(pvtm_col) and r.get(pvtm_col) != "未触发"]
    a("| 判据 | 命中次数 | 结论 |")
    a("|---|---|---|")
    a("| A. cooling_device cur_state > 0（温控真的动手） | %d | %s |" %
      (len(cd_hits), "✅ 全程无温控动作" if not cd_hits else "❌ " + str(cd_hits[:3])))
    a("| B. 任一域 cur_freq < 额定值 | %d | %s |" %
      (sum(1 for r in rows for _l, c, _m, rt in FREQ_DOMAINS
           if has(c) and (num(r, c) or 0) > 0
           and (num(r, c) < (RATED_CONST.get(rt) or
                             next((num(x, rt) for x in rows if num(x, rt)), 0)))),
         "✅ 全程未低于额定值" if not any_low else "❌ 见上表"))
    a("| C. PVTM 钳位标记（观测器判定） | %d | %s |" %
      (len(pvtm_hits), "✅ 未触发" if not pvtm_hits else "❌ " + str(pvtm_hits[:3])))
    dmesg_hits = []
    if args.dmesg and Path(args.dmesg).exists():
        pattern = re.compile(r"thermal|throttl|cooling|pvtm|devfreq.*(down|limit)|watchdog", re.I)
        dmesg_hits = [l for l in Path(args.dmesg).read_text(errors="replace").splitlines()
                      if pattern.search(l)]
    a("| D. dmesg 温控增量 | %d | %s |" %
      (len(dmesg_hits), "✅ 无" if not dmesg_hits else "⚠️ " + str(len(dmesg_hits)) + " 行"))
    a("")
    if has("guard_on"):
        g = [num(r, "guard_on") for r in rows if num(r, "guard_on") is not None]
        off = sum(1 for v in g if v == 0)
        a("**满频锁看门狗**：maxfreq-guard.timer 每 10 s 重设满频，但 SoC ≥ 78℃ 后主动停手；"
          "本次采样中停手样本 %d/%d（%.1f%%）。停手之后的任何频率下降都是真实降频。" %
          (off, len(g), 100.0 * off / len(g)) if g else "")
        a("")

    # ---------------- NPU 真实负载 ----------------
    if has("rknpu0"):
        rk = [v for v in (num(r, "rknpu0") for r in rows) if v is not None]
        if rk:
            busy = sum(1 for v in rk if v > 0)
            a("### NPU 真实负载（rknpu Core0，%）")
            a("")
            a("| 指标 | 值 |")
            a("|---|---|")
            a("| 采样数 | %d |" % len(rk))
            a("| Core0 > 0 的采样占比（NPU 占空比） | %d/%d = **%.1f%%** |" %
              (busy, len(rk), 100.0 * busy / len(rk)))
            a("| 最小 / 均值 / 峰值 | %.0f / %s / %.0f |" %
              (min(rk), fmt(stat(rk)["avg"], 1), max(rk)))
            a("| Core0 = 0 的采样数（说明 NPU 空闲/客户端卡死） | %d |" % (len(rk) - busy))
            a("")

    # ---------------- 负载与风扇 ----------------
    a("### 负载与风扇")
    a("")
    a("| 项 | 值 |")
    a("|---|---|")
    if has("load1"):
        load1 = [v for v in (num(r, "load1") for r in rows) if v is not None]
        if load1:
            a("| load1 均值 / 峰值 | %s / %s |" %
              (fmt(stat(load1)["avg"], 2), fmt(stat(load1)["max"], 2)))
    if has("nload_proc"):
        nproc = [v for v in (num(r, "nload_proc") for r in rows) if v is not None]
        a("| 负载进程数（stress+mosaic）峰值 | %d |" % (max(nproc) if nproc else 0))
    if has("fan_pwm"):
        fan = [v for v in (num(r, "fan_pwm") for r in rows) if v is not None]
        if fan:
            a("| 风扇 pwm 最小 / 均值 / 最大 | %d / %s / %d |" %
              (min(fan), fmt(stat(fan)["avg"], 0), max(fan)))
    a("")

    if args.per_minute:
        a("### 每分钟关键状态")
        a("")
        a("| 时刻 | SoC℃ | 最热℃ | GPU℃ | NPU℃ | CPU0 cur/max | CPU6 cur/max | NPU cur/max | GPU cur/max | rknpu0% | cdev | fan | load | 判定 |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        seen_min = set()
        for r in rows:
            minute = r["wall"][:5]
            if minute in seen_min:
                continue
            seen_min.add(minute)
            cdev_hit = any((num(r, c) or 0) > 0 for c in CD_COLS if has(c))
            pvtm_hit = r.get("pvtm") not in (None, "", "未触发")
            a("| %s | %s | %s | %s | %s | %s/%s | %s/%s | %s/%s | %s/%s | %s | %s | %s | %s | %s |" % (
                r["wall"][11:] or r["wall"],
                fmt((num(r, "soc_c") or 0) / 1000),
                fmt((num(r, "soc_max_c") or num(r, "soc_c") or 0) / 1000),
                fmt((num(r, "gpu_c") or 0) / 1000),
                fmt((num(r, "npu_c") or 0) / 1000),
                r.get("cpu0_cur", "-"), r.get("cpu0_max", "-"),
                r.get("cpu6_cur", "-"), r.get("cpu6_max", "-"),
                r.get("npu_cur", "-"), r.get("npu_max", "-"),
                r.get("gpu_cur", "-"), r.get("gpu_max", "-"),
                r.get("rknpu0") or "-",
                ",".join(r.get(c, "-") for c in CD_COLS),
                r.get("fan_pwm", "-"), r.get("load1", "-"),
                "🔴降频" if (cdev_hit or pvtm_hit) else "🟢"))
        a("")

    text = "\n".join(out)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
