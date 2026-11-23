#!/bin/bash
# 守望的守望：cron 每 3 分钟检查 day_supervisor_d1.sh 是否存活 ⇒ 死了就拉起
W=/mnt/data/siping/runs/npu20ms/nano_search
if ! ps -eo args | grep -q "[d]ay_supervisor_d1.sh"; then
  setsid nohup bash $W/scripts/day_supervisor_d1.sh >> $W/day_supervisor_runner.log 2>&1 < /dev/null 0<&- 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&- &
  echo "[$(date "+%F %T")] [CRON] 守望已死 ⇒ 已拉起" >> $W/day_supervisor.log
fi
