#!/bin/bash
# 抓取板卡温控配置的完整快照（供测试报告与嵌入式组核对）
# 用法: board_thermal_config_snapshot.sh [输出文件]
OUT="${1:-/tmp/board_thermal_config.txt}"
Z=/sys/class/thermal/thermal_zone0
{
echo "=== 采集时间 / boot ==="
date '+%F %T'; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"; uptime
echo
echo "=== thermal zones ==="
for z in /sys/class/thermal/thermal_zone*; do
  echo "[$z] type=$(cat $z/type 2>/dev/null) temp=$(cat $z/temp 2>/dev/null) policy=$(cat $z/policy 2>/dev/null) mode=$(cat $z/mode 2>/dev/null)"
done
echo
echo "=== soc-thermal trips（运行时生效值）==="
for i in 0 1 2; do
  echo "  trip_point_$i: temp=$(cat $Z/trip_point_${i}_temp 2>/dev/null) type=$(cat $Z/trip_point_${i}_type 2>/dev/null) hyst=$(cat $Z/trip_point_${i}_hyst 2>/dev/null)"
done
echo
echo "=== cooling device <-> trip 绑定（关键！）==="
for c in $Z/cdev[0-9]*; do
  id=$(basename "$c")
  [ "$id" = "cdev${id#cdev}_trip_point" ] && continue
  case "$id" in *_trip_point|*_weight) continue;; esac
  echo "  $id type=$(cat $c/type 2>/dev/null) cur_state=$(cat $c/cur_state 2>/dev/null) max_state=$(cat $c/max_state 2>/dev/null) trip_index=$(cat $Z/${id}_trip_point 2>/dev/null)"
done
echo
echo "=== 设备树原始值（未改固件）==="
for n in trip-point-0 trip-point-1 soc-crit; do
  printf "  %s.temperature = " "$n"; od -An -tx1 /sys/firmware/devicetree/base/thermal-zones/soc-thermal/trips/$n/temperature | tr -d ' \n'; echo
done
echo "=== 设备树 cooling-maps（trip 指向哪个 phandle）==="
for m in /sys/firmware/devicetree/base/thermal-zones/soc-thermal/cooling-maps/map*; do
  printf "  %s trip_phandle=" "$(basename $m)"; od -An -tx4 $m/trip | tr -d ' \n'; echo
done
echo
echo "=== 频率上限 / governor ==="
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  echo "  $(basename $p): cur=$(cat $p/scaling_cur_freq 2>/dev/null) max=$(cat $p/scaling_max_freq 2>/dev/null) gov=$(cat $p/scaling_governor 2>/dev/null)"
done
for d in /sys/class/devfreq/*; do
  echo "  devfreq $(basename $d): cur=$(cat $d/cur_freq 2>/dev/null) max=$(cat $d/max_freq 2>/dev/null) gov=$(cat $d/governor 2>/dev/null)"
done
echo
echo "=== PVTM（rockchip-pvtm）==="
for p in /sys/kernel/debug/pvtm/*/value; do echo "  $p = $(cat $p 2>/dev/null)"; done
echo
echo "=== 开机启动项（本次新增/生效）==="
systemctl list-unit-files 2>/dev/null | grep -E "thermal-trip|nfs-mount" || true
systemctl is-enabled thermal-trip.service nfs-mount.service 2>/dev/null
echo "  /etc/fstab 中 NFS 行:"; grep -n "SSD960-LOCAL/nfs" /etc/fstab 2>/dev/null
echo
echo "=== 风扇 ==="
for f in /sys/class/hwmon/hwmon*/pwm1; do [ -e "$f" ] && echo "  $f = $(cat $f)"; done
} > "$OUT" 2>&1
cat "$OUT"
