#!/bin/bash
# 🟢 晨报调度器（2026-11-19 11:40）
#   根因：晨报原先由 day_supervisor 的 mreport 分支拉起，
#     ① supervisor 的「每通道重启上限 30 次」已被 mreport 吃光 ⇒ 再也不会拉起；
#     ② AT 解析曾指向**过去**的时间 ⇒ 脚本秒退 ⇒ 每 90s 重启一次（26~30 次）⇒ 空烧机时。
#   修法：独立长驻调度器，每天 08:40 出一份报告，直到 2026-11-30 09:00。
#        flock 互斥 + PID 文件 ⇒ supervisor 可安全地按 PID 判活重启。
set -u
REPO=/tmp/pycharm_project_10ae9e2e
W=/mnt/data/siping/runs/npu20ms/nano_search
END="${MRPT_END:-2026-11-30 09:00}"
AT_HHMM="08:55"
exec 9>"$W/.lock_mreport_sched"
flock -n 9 || { echo "[$(date '+%F %T')] [DUP] 已有调度器 ⇒ 退出" >> $W/morning_report_sched.log; exit 0; }
echo $$ > "$W/.pid_mreport_sched"
cleanup(){ [ "$(cat "$W/.pid_mreport_sched" 2>/dev/null)" = "$$" ] && rm -f "$W/.pid_mreport_sched"; }
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { echo "[$(date '+%F %T')] [截止] $END ⇒ 退出" >> $W/morning_report_sched.log; break; }
  if [ "$(date +%H%M)" -lt "$AT_HHMM" ]; then
    AT=$(date "+%Y-%m-%d $AT_HHMM")
  else
    AT=$(date -d "tomorrow" "+%Y-%m-%d $AT_HHMM")
  fi
  echo "[$(date '+%F %T')] 下一份晨报时刻 $AT" >> $W/morning_report_sched.log
  while [ "$(date +%s)" -lt "$(date -d "$AT" +%s)" ]; do
    [ "$(date +%s)" -lt "$end" ] || break
    sleep 30
  done
  [ "$(date +%s)" -lt "$end" ] || { echo "[$(date '+%F %T')] [截止] $END => 退出" >> $W/morning_report_sched.log; break; }
  sleep 20
  echo "[$(date '+%F %T')] 生成晨报 $AT" >> $W/morning_report_sched.log
  bash "$REPO/manu/pipeline/npu/run_morning_report4.sh" "$AT" >> $W/morning_report_runner.log 2>&1
  sleep 90
done
