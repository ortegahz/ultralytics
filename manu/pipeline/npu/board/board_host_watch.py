#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主机侧板卡观测器（不依赖板卡本地落盘 ⇒ 板卡重启也不丢证据）。

每 N 秒拉一次板卡紧凑状态，写主机侧 CSV；每 60 s 打印一行关键状态；
同时把 dmesg 尾部做「变化检测」，只把新出现的内核行追加到单独文件
⇒ 板卡复位前最后几秒的内核消息会被留在主机上。

用法:
    python3 board_host_watch.py --duration 4200 --interval 2 --print-interval 60 \
        --out /tmp/bts_watch --label newboard_1h
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

# SSH 走外部 helper（把口令留在 helper 里，不进仓库）
SSH_HELPER = os.environ.get("BTS_SSH_HELPER", "/tmp/bssh.sh")

# 板卡侧一行紧凑状态（只读 /sys /proc，命令尽量少、尽量快）
# 字段顺序 == CSV 列顺序（CSV = wall,t_s + 本列表 + pvtm,ssh_ok）
# 关键点：频率一律记 **cur_freq（实际在跑的频率）**。PM QoS(PVTM) 只压低实际频率，
#       sysfs 的 max_freq 可能仍是额定值，所以只看 max_freq 会漏报降频。
TZ = "/sys/class/thermal/thermal_zone"
POL = "/sys/devices/system/cpu/cpufreq/policy"
GPUD = "/sys/class/devfreq/fb000000.gpu"
NPUD = "/sys/class/devfreq/fdab0000.npu"
DMCD = "/sys/class/devfreq/dmc"

PROBE_FIELDS = [
    ("soc_c",      "cat %s0/temp 2>/dev/null" % TZ),          # soc-thermal = PVTM 温度源
    ("soc_max_c",  "cat %s*/temp 2>/dev/null | sort -n | tail -1" % TZ),
    ("gpu_c",      "cat %s5/temp 2>/dev/null" % TZ),          # gpu-thermal
    ("npu_c",      "cat %s6/temp 2>/dev/null" % TZ),          # npu-thermal
    ("little_c",   "cat %s3/temp 2>/dev/null" % TZ),          # littlecore-thermal
    ("big0_c",     "cat %s1/temp 2>/dev/null" % TZ),          # bigcore0-thermal
    ("big1_c",     "cat %s2/temp 2>/dev/null" % TZ),          # bigcore1-thermal
    ("center_c",   "cat %s4/temp 2>/dev/null" % TZ),          # center-thermal
    ("cpu0_cur",   "cat %s0/scaling_cur_freq 2>/dev/null" % POL),
    ("cpu0_max",   "cat %s0/scaling_max_freq 2>/dev/null" % POL),
    ("cpu4_cur",   "cat %s4/scaling_cur_freq 2>/dev/null" % POL),
    ("cpu4_max",   "cat %s4/scaling_max_freq 2>/dev/null" % POL),
    ("cpu6_cur",   "cat %s6/scaling_cur_freq 2>/dev/null" % POL),
    ("cpu6_max",   "cat %s6/scaling_max_freq 2>/dev/null" % POL),
    ("npu_cur",    "cat %s/cur_freq 2>/dev/null" % NPUD),
    ("npu_max",    "cat %s/max_freq 2>/dev/null" % NPUD),
    ("gpu_cur",    "cat %s/cur_freq 2>/dev/null" % GPUD),
    ("gpu_max",    "cat %s/max_freq 2>/dev/null" % GPUD),
    ("dmc_cur",    "cat %s/cur_freq 2>/dev/null" % DMCD),
    ("dmc_max",    "cat %s/max_freq 2>/dev/null" % DMCD),
    ("npu_load",   "cat %s/load 2>/dev/null" % NPUD),
    ("gpu_util",   "cat /sys/devices/platform/fb000000.gpu/utilisation 2>/dev/null"),
    ("cd0",        "cat /sys/class/thermal/cooling_device0/cur_state 2>/dev/null"),
    ("cd1",        "cat /sys/class/thermal/cooling_device1/cur_state 2>/dev/null"),
    ("cd2",        "cat /sys/class/thermal/cooling_device2/cur_state 2>/dev/null"),
    ("cd3",        "cat /sys/class/thermal/cooling_device3/cur_state 2>/dev/null"),
    ("fan_pwm",    "cat /sys/class/hwmon/hwmon7/pwm1 2>/dev/null"),
    ("load1",      "cut -d' ' -f1 /proc/loadavg"),
    ("uptime",     "cut -d' ' -f1 /proc/uptime"),
    ("boot_id",    "cat /proc/sys/kernel/random/boot_id 2>/dev/null"),
    ("nload_proc", "ps -eo comm 2>/dev/null | grep -cE '^(stress|mosaic_async_client)$'"),
    ("rknpu0",     "sed -n 's/.*Core0: *\\([0-9][0-9]*\\)%.*/\\1/p' /sys/kernel/debug/rknpu/load 2>/dev/null | head -1"),
    # guard_on=1 表示 maxfreq-guard.timer 还在每 10s 重设满频（SoC<78C）；
    # >=78C 后 guard 主动停手，此后任何频率下降都是真实降频，不再是锁被顶掉
    ("guard_on",   "t=$(cat %s0/temp 2>/dev/null); [ -n \"$t\" ] && [ \"$t\" -lt 78000 ] && echo 1 || echo 0" % TZ),
    ("nom0",       "cat %s0/cpuinfo_max_freq 2>/dev/null" % POL),
    ("nom4",       "cat %s4/cpuinfo_max_freq 2>/dev/null" % POL),
    ("nom6",       "cat %s6/cpuinfo_max_freq 2>/dev/null" % POL),
]

PROBE = "\n".join("v%d=$(%s)" % (i, expr) for i, (_n, expr) in enumerate(PROBE_FIELDS))
PROBE += "\nprintf '" + " ".join(["%s"] * len(PROBE_FIELDS)) + "'"
PROBE += " " + " ".join('"$v%d"' % i for i in range(len(PROBE_FIELDS)))
PROBE += "\n"

# 额定频率（降频判据的基准）：GPU/NPU/DMC 的额定值由 maxfreq-lock.sh 锁定在额定最高频
RATED_GPU = 1000000000
RATED_NPU = 1000000000
RATED_DMC = 2112000000
# PVTM 硬天花板（设备树 OPP 表 rockchip,high-temp-max-freq，内核 PM QoS 钳位值）
PVTM_CEILING = {"cpu0": 1608000, "cpu4": 2208000, "cpu6": 2208000,
                "gpu": 800000000, "npu": 800000000, "dmc": 2112000000}


def ssh(cmd: str, timeout: int = 12) -> tuple[int, str]:
    try:
        p = subprocess.run([SSH_HELPER, cmd], capture_output=True, text=True,
                           timeout=timeout, errors="replace")
        return p.returncode, p.stdout.strip()
    except subprocess.TimeoutExpired:
        return 255, ""
    except Exception as exc:  # noqa: BLE001
        return 255, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4200.0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--print-interval", type=float, default=60.0)
    ap.add_argument("--out", default="/tmp/bts_watch")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    csv_path = args.out + "_samples.csv"
    dmesg_path = args.out + "_dmesg.txt"
    log_path = args.out + "_log.txt"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # 列顺序 = wall,t_s + PROBE_FIELDS + pvtm,ssh_ok（索引由 PROBE_FIELDS 生成，杜绝手写错位）
    cols = ["wall", "t_s"] + [n for n, _ in PROBE_FIELDS] + ["pvtm", "ssh_ok"]
    IX = {n: 2 + i for i, (n, _e) in enumerate(PROBE_FIELDS)}   # CSV 列名 -> 列号
    fh = open(csv_path, "w")
    fh.write(",".join(cols) + "\n")
    dh = open(dmesg_path, "a")
    lh = open(log_path, "a")

    def say(msg: str) -> None:
        line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        lh.write(line + "\n")
        lh.flush()

    say("== 主机侧观测器启动 label=%s duration=%.0fs interval=%.0fs ==" %
        (args.label or "-", args.duration, args.interval))

    t0 = time.monotonic()
    next_print = args.print_interval      # 相对 elapsed（秒）
    last_dmesg = ""
    last_bid = ""
    n = 0
    fails = 0

    while time.monotonic() - t0 < args.duration:
        rc, out = ssh(PROBE)
        el = time.monotonic() - t0
        if rc == 0 and out:
            f = out.split()
            if len(f) >= len(PROBE_FIELDS):
                # npu_load 格式 "100@1000000000Hz" ⇒ 只取数字
                f[IX["npu_load"] - 2] = re.sub(r"[^0-9].*$", "", f[IX["npu_load"] - 2])

                def g(name):
                    return f[IX[name] - 2]

                def gi(name):
                    try:
                        return int(float(g(name)))
                    except (TypeError, ValueError):
                        return None

                # ---- 降频/PVTM 判定 ----
                # 判据 1：实际跑频 cur_freq < 额定值  ⇒ 已经被压过（含 PM QoS/PVTM 钳位）
                # 判据 2：允许上限 max_freq < 额定值  ⇒ 上限本身被压
                # 判据 3：cooling_device cur_state > 0 ⇒ 温控真的动了
                # PVTM 额定值：CPU 取 cpuinfo_max_freq；GPU/NPU/DMC 取满频锁的目标值
                rated = {"cpu0": gi("nom0") or 0, "cpu4": gi("nom4") or 0,
                         "cpu6": gi("nom6") or 0, "gpu": RATED_GPU,
                         "npu": RATED_NPU, "dmc": RATED_DMC}
                pvtm, throttled = [], []
                for dom, curc, maxc in (("cpu0", "cpu0_cur", "cpu0_max"),
                                        ("cpu4", "cpu4_cur", "cpu4_max"),
                                        ("cpu6", "cpu6_cur", "cpu6_max"),
                                        ("npu", "npu_cur", "npu_max"),
                                        ("gpu", "gpu_cur", "gpu_max"),
                                        ("dmc", "dmc_cur", "dmc_max")):
                    cur_v, max_v, rt = gi(curc), gi(maxc), rated[dom]
                    if not rt:
                        continue
                    if cur_v is not None and cur_v < rt:
                        throttled.append("%s(%d<%d)" % (dom, cur_v, rt))
                        if cur_v <= PVTM_CEILING.get(dom, 0):
                            pvtm.append(dom.upper())
                    elif max_v is not None and max_v < rt:
                        throttled.append("%s_max(%d<%d)" % (dom, max_v, rt))
                cdev_hit = [g("cd0"), g("cd1"), g("cd2"), g("cd3")]
                cdev_on = any(gi(c) and gi(c) > 0 for c in ("cd0", "cd1", "cd2", "cd3"))
                if cdev_on:
                    throttled.append("cdev")
                pvtm_s = "已触发:" + ",".join(pvtm) if pvtm else "未触发"
                f.append(pvtm_s)
                row = ([time.strftime("%H:%M:%S"), "%.1f" % el] + f[0:len(PROBE_FIELDS)]
                       + [pvtm_s, "1"])
                fh.write(",".join(str(x) for x in row[:len(cols)]) + "\n")
                fh.flush()
                n += 1
                bid = g("boot_id")
                if bid and not last_bid:
                    # 首次观测：只记录基线，不算“重启”（旧版会把每次开跑都误报成重启）
                    last_bid = bid
                    say("== 首次观测：boot_id=%s uptime=%s ==" %
                        (bid, g("uptime")))
                elif bid and bid != last_bid:
                    say("*** 板卡重启！boot_id %s -> %s（此前 uptime=%s）" %
                        (last_bid, bid, g("uptime")))
                    last_bid = bid
                    last_dmesg = ""
                if el >= next_print:
                    def tc(name):
                        v = gi(name)
                        return "%.1f" % (v / 1000) if v is not None else "-"
                    say("t=%5.0fs soc=%s hottest=%s(big=%s/%s) gpu=%s npu=%s | "
                        "CPU0 %s/%s CPU4 %s/%s CPU6 %s/%s | NPU %s/%s GPU %s/%s DMC %s/%s | "
                        "rknpu0=%s%% GPU_util=%s%% cdev=%s,%s,%s,%s fan=%s guard=%s | load=%s | "
                        "PVTM:%s | %s" % (
                            el,
                            tc("soc_c"), tc("soc_max_c"), tc("big0_c"), tc("big1_c"),
                            tc("gpu_c"), tc("npu_c"),
                            g("cpu0_cur"), g("cpu0_max"), g("cpu4_cur"), g("cpu4_max"),
                            g("cpu6_cur"), g("cpu6_max"),
                            g("npu_cur"), g("npu_max"), g("gpu_cur"), g("gpu_max"),
                            g("dmc_cur"), g("dmc_max"),
                            g("rknpu0") or "-", g("gpu_util") or "-",
                            g("cd0"), g("cd1"), g("cd2"), g("cd3"), g("fan_pwm"),
                            g("guard_on"),
                            g("load1"), pvtm_s,
                            ("🔴降频:" + ",".join(throttled)) if throttled else "🟢正常"))
                    next_print = el + args.print_interval
        else:
            fails += 1
            if el >= next_print:
                say("t=%5.0fs !! ssh 失败 (rc=%s) 连续失败 %d 次 —— 板卡可能正在重启" % (el, rc, fails))
                next_print = el + args.print_interval

        # dmesg 变化检测：只把新行写到主机（板卡复位前的内核消息就靠这个留下）
        rc2, dm = ssh("dmesg | tail -25")
        if rc2 == 0 and dm:
            dl = dm.splitlines()
            newl = [x for x in dl if x not in last_dmesg.splitlines()] if last_dmesg else dl
            for line in newl:
                dh.write("%s %s\n" % (time.strftime("%H:%M:%S"), line))
                dh.flush()
            last_dmesg = dm
        time.sleep(args.interval)

    say("== 观测结束：有效样本 %d，ssh 失败 %d 次 -> %s ==" % (n, fails, csv_path))
    fh.close()
    dh.close()
    lh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
