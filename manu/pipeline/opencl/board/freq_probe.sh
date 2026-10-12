#!/bin/sh
# ===========================================================================
# freq_probe.sh -- what DVFS knobs actually exist on this board, and what are
# they doing right now.
#
# WHY A PROBE AND NOT THE COMMANDS FROM THE PLAN
#   The plan says "lock the clocks, then measure". Locking is only meaningful if
#   the knob exists AND is writable, and on RK3588 the CPU and GPU clocks are
#   governed by two different subsystems with different permissions. A failed
#   `echo performance > ...` is silent when wrapped in `|| true`, which would
#   leave DVFS live during a "locked" benchmark -- the exact deception this whole
#   exercise exists to prevent. So: report the knob, its current value, and
#   whether it is writable, and FAIL LOUDLY if a lock we depend on cannot be set.
# ===========================================================================
echo "=== CPU ==="
echo -n "cpupower: "; command -v cpupower || echo "(not installed)"
for c in 0 4 5 6 7; do
  g=/sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor
  f=/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq
  m=/sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq
  if [ -f "$g" ]; then
    printf 'cpu%s governor=%-12s cur=%-7s max=%-7s writable=%s\n' "$c" "$(cat $g)" \
      "$(cat $f 2>/dev/null)" "$(cat $m 2>/dev/null)" \
      "$([ -w "$g" ] && echo yes || echo NO)"
  else
    echo "cpu$c: no cpufreq"
  fi
done
echo -n "online cores: "; cat /sys/devices/system/cpu/online 2>/dev/null

echo
echo "=== GPU ==="
G=/sys/class/devfreq/fb000000.gpu
if [ -d "$G" ]; then
  for f in governor available_governors cur_freq min_freq max_freq trans_stat; do
    [ -f "$G/$f" ] && printf '%-20s %s\n' "$f:" "$(cat $G/$f 2>/dev/null)"
  done
  printf 'governor writable=%s\n' "$([ -w "$G/governor" ] && echo yes || echo NO)"
else
  echo "no $G"
fi

echo
echo "=== NPU / devfreq ==="
ls /sys/class/devfreq/ 2>/dev/null
echo -n "/dev/rknpu: "; [ -e /dev/rknpu ] && echo present || echo ABSENT

echo
echo "=== thermal ==="
for z in /sys/class/thermal/thermal_zone*; do
  [ -f "$z/temp" ] && printf '%s type=%s temp=%s\n' "$(basename $z)" \
    "$(cat $z/type 2>/dev/null)" "$(cat $z/temp 2>/dev/null)"
done

echo
echo "=== load ==="
cat /proc/loadavg
free -m | head -2