#!/bin/sh
# maxfreq-guard.sh -- watchdog that re-asserts the maximum-frequency lock.
#
# It guards against everything EXCEPT the PVTM high-temperature limit:
#   * cpufrequtils.service / orangepi-config resetting the governor
#   * the thermal governor (power_allocator) lowering scaling_max_freq or
#     the devfreq min_freq after a temperature excursion
#
# It deliberately does NOT fight the PVTM clamp: the clamp engages above
# 85 degC and releases below 80 degC (5 degC hysteresis), so above 78 degC
# this script only records what it sees and leaves the frequencies alone.
# That ceiling is the silicon's stability/lifetime guarantee and must be
# respected -- Rockchip's CPUFreq guide RK-KF-YF-184 states the wide
# temperature window is -40..85 degC and that the derating exists to protect
# chip lifetime.
#
# Run every 10 s from maxfreq-guard.timer.

LOCK=/usr/local/bin/maxfreq-lock.sh
HOT=78000
Z=/sys/class/thermal/thermal_zone0/temp
LOG=/var/log/maxfreq-guard.log

t=$(cat "$Z" 2>/dev/null)
[ -z "$t" ] && exit 0
if [ "$t" -ge "$HOT" ]; then
    echo "$(date '+%F %T') SoC $((t/1000))C >= $((HOT/1000))C - PVTM clamp may be active, not overriding" >> "$LOG"
    exit 0
fi
sh "$LOCK" >> "$LOG" 2>&1
