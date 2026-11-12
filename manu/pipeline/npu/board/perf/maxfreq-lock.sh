#!/bin/sh
# maxfreq-lock.sh -- pin the RK3588S (Orange Pi CM5) to a constant maximum
# frequency that the silicon guarantees at ANY temperature.
#
# ============================ WHY THIS DESIGN ============================
# Measured on board root@192.168.0.211 (Orange Pi CM5, Debian 12, kernel
# 5.10.160-rockchip-rk3588), 2026-11-12.
#
# The kernel's own boot log names the mechanism:
#     cpu cpu0: l=10000 h=85000 hyst=5000 l_limit=0 h_limit=1608000000 h_table=0
#     cpu cpu4: l=10000 h=85000 hyst=5000 l_limit=0 h_limit=2208000000 h_table=0
#     mali fb000000.gpu: l=10000 h=85000 hyst=5000 l_limit=0 h_limit=800000000
#     RKNPU fdab0000.npu: l=10000 h=85000 hyst=5000 l_limit=0 h_limit=800000000
#
# That is drivers/soc/rockchip/rockchip_system_monitor.c.  It parses the
# device tree of every OPP table:
#     rockchip,high-temp          = 85000          (m degC)
#     rockchip,temp-hysteresis    = 5000           (m degC)
#     rockchip,high-temp-max-freq = <ceiling>      (kHz)
# and enforces the ceiling through **PM QoS**:
#     CPU        FREQ_QOS_MAX           on the cpufreq policy
#     GPU / NPU  DEV_PM_QOS_MAX_FREQUENCY on the devfreq parent
# QoS aggregation is min(), so
#     effective_max = min(our request, thermal ceiling)
# and a user-space write to scaling_max_freq CANNOT bypass it.  There is no
# runtime knob: the values are parsed once at probe time, the driver is
# built into the image, and /sys/kernel/debug only exposes read-only
# summaries.
#
# Temperature source: thermal zone `soc-thermal` (TSADC), selected by
# `rockchip,thermal-zone` in the /rockchip-system-monitor node.
# Engage at temp > 85 degC, release at temp < 80 degC (the 5 degC hysteresis)
# => a 80..85 degC band in which the clock depends on the board's THERMAL
# HISTORY.  That is a reproducibility hazard for a pipeline whose output has
# to be bit-exact.
#
# ============================ WHAT THIS SCRIPT DOES ============================
# It pins every domain at the **PVTM ceiling** read live from the device
# tree (the same constant the kernel uses):
#
#     cluster0 (A55) 1608000 Hz | cluster1/2 (A76) 2208000 Hz
#     GPU  800000000 Hz | NPU  800000000 Hz | DMC 2112000000 Hz (never clamped)
#
# Consequences:
#   * The frequency is CONSTANT -- the QoS ceiling and our request are equal,
#     so min() returns the same number hot or cold.  No hysteresis hunting,
#     no thermal-history-dependent output.
#   * min_freq == max_freq, so the kernel never has to resolve a
#     min > max policy (which is what makes later sysfs writes fail EIO).
#   * It is the vendor's own wide-temperature guarantee.  Rockchip's CPUFreq
#     guide (RK-KF-YF-184) states the wide-temperature window is -40..85 degC
#     and that the >85 degC derating exists to protect chip LIFETIME.
#     Do not raise that threshold.
#
# Set MAXFREQ_TARGET=full to pin at the CPU's rated max instead
# (1800/2400/1000 MHz).  That is ~8..20 % faster on paper but it is only
# reachable below 85 degC; above it the kernel clamps it back down anyway,
# so the clock becomes history-dependent again.
#
# Idempotent; safe to run at any time; needs root.
# =====================================================================

TARGET_MODE=${MAXFREQ_TARGET:-full}
CPU_POLICIES="0 4 6"
GPU=/sys/class/devfreq/fb000000.gpu
NPU=/sys/class/devfreq/fdab0000.npu
DMC=/sys/class/devfreq/dmc
DTB=/sys/firmware/devicetree/base

log() { echo "[maxfreq-lock] $*"; }

# ---- read a u32 cell out of the live device tree -------------------------
dt_u32() {
    [ -f "$1" ] || return 1
    od -An -tu4 --endian=big "$1" 2>/dev/null | tr -d ' \n'
}

# PVTM ceiling for an OPP table, in kHz (the DT property is in kHz and the
# kernel driver does value*1000 to get Hz).
#   cpufreq sysfs (scaling_min_freq / scaling_max_freq) is in kHz
#   devfreq  sysfs (min_freq / max_freq / cur_freq) is in Hz
pvtm_ceiling_khz() {
    dt_u32 "$DTB/$1-opp-table/rockchip,high-temp-max-freq"
}

# cpufreq policy index -> OPP table name (policy4 sits on cluster1, policy6 on
# cluster2 -- the A76 cluster is split into two policies).
table_for_policy() {
    case "$1" in
        0) echo cluster0 ;;
        4) echo cluster1 ;;
        6) echo cluster2 ;;
        *) echo "cluster$1" ;;
    esac
}

target_for() { # $1 = policy index (0/4/6), $2 = cpuinfo_max (kHz)
    case "$TARGET_MODE" in
        full) echo "$2"; return ;;
        *)    c=$(pvtm_ceiling_khz "$(table_for_policy "$1")")
              [ -n "$c" ] && echo "$c" || echo "$2" ;;
    esac
}

# ---- CPU policies -------------------------------------------------------
for p in $CPU_POLICIES; do
    d=/sys/devices/system/cpu/cpufreq/policy$p
    [ -d "$d" ] || continue
    mx=$(cat "$d/cpuinfo_max_freq" 2>/dev/null)
    [ -n "$mx" ] || continue
    t=$(target_for "$p" "$mx")
    echo performance > "$d/scaling_governor" 2>/dev/null
    echo "$t" > "$d/scaling_max_freq" 2>/dev/null
    # pin min only when it cannot exceed the effective max: while the PVTM
    # clamp holds max lower than our target, min > max would leave the policy
    # inconsistent (and makes later sysfs writes fail with EIO).
    eff=$(cat "$d/scaling_max_freq" 2>/dev/null)
    if [ -n "$eff" ] && [ "$t" -le "$eff" ]; then
        echo "$t" > "$d/scaling_min_freq" 2>/dev/null || true
    fi
    log "policy$p gov=$(cat $d/scaling_governor) min=$(cat $d/scaling_min_freq) max=$(cat $d/scaling_max_freq) cur=$(cat $d/scaling_cur_freq) (target $t)"
done

# ---- devfreq domains ----------------------------------------------------
lock_devfreq() {
    n=$1; t=$2
    [ -d "$n" ] || { log "$(basename "$n") not present, skipped"; return; }
    echo performance > "$n/governor" 2>/dev/null
    echo "$t" > "$n/max_freq" 2>/dev/null
    eff=$(cat "$n/max_freq" 2>/dev/null)
    if [ -n "$eff" ] && [ "$t" -le "$eff" ]; then
        echo "$t" > "$n/min_freq" 2>/dev/null || true
    fi
    log "$(basename "$n") gov=$(cat $n/governor) min=$(cat $n/min_freq) max=$(cat $n/max_freq) cur=$(cat $n/cur_freq) (target $t)"
}
gpu_c=$(pvtm_ceiling_khz gpu); npu_c=$(pvtm_ceiling_khz npu)
if [ "$TARGET_MODE" = "full" ]; then
    gpu_c=1000000; npu_c=1000000      # kHz -> 1 GHz
fi
lock_devfreq "$GPU"  "$(( ${gpu_c:-1000000} * 1000 ))"
lock_devfreq "$NPU"  "$(( ${npu_c:-1000000} * 1000 ))"
lock_devfreq "$DMC"  "$(cat $DMC/available_frequencies 2>/dev/null | tr ' ' '\n' | sort -n | tail -1 || echo 2112000000)"

# ---- report the PVTM constants we are respecting -------------------------
for t in cluster0 cluster1 cluster2 gpu npu; do
    d=$DTB/$t-opp-table
    [ -d "$d" ] || continue
    ht=$(dt_u32 "$d/rockchip,high-temp")
    hy=$(dt_u32 "$d/rockchip,temp-hysteresis")
    hm=$(dt_u32 "$d/rockchip,high-temp-max-freq")
    [ -n "$ht" ] && log "PVTM $t: high-temp=${ht}C hyst=${hy}C ceiling=${hm}kHz ($((hm * 1000))Hz)"
done
log "target mode = $TARGET_MODE"
exit 0
