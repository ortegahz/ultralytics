#!/bin/bash
# 🟢 白天波 D1 通道自愈守望（2026-11-18 09:35）
#   每 5 分钟检查 4 条通道 + 4 个守望进程，掉线即拉起（幂等：号段/配方固定 ⇒ 复用同臂号）。
#   铁律：拉起前查 MemAvailable（<40GB ⇒ 等 120s）；每个通道重启上限 30 次。
set -u
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/day_supervisor.log
END="${SUP_END:-2026-11-30 09:00}"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
log(){ echo "[$(date "+%F %T")] $*" >> $LOG; }
# 判活：PAT 是 PID 文件路径 ⇒ 读 PID 后 kill -0；PAT 是正则 ⇒ 走 pgrep（已锚定，不会自匹配）
alive(){
  local pat="$1" pid
  case "$pat" in
    /*) [ -f "$pat" ] || return 1; pid=$(cat "$pat" 2>/dev/null); [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null ;;
    *)   pgrep -f "$pat" >/dev/null 2>&1 ;;
  esac
}
exec 9>"$W/.lock_day_supervisor"
flock -n 9 || { log "[DUP] 已有守望实例在跑 ⇒ 本实例退出"; exit 0; }
pkill -f "synth_gpu2.sh" 2>/dev/null
pkill -f "synth_gpu3.sh" 2>/dev/null
log "[启动] 守望实例 pid=$$ （flock 互斥已生效）"
declare -A PAT REST
# 🟢🟢 2026-11-18 10:10 **判活口径全面改为 PID 文件 / 锚定正则**
#   坑：`pgrep -f "<脚本名>"` 会匹配**任何提到该名字的 shell**（运维 ssh 命令里也含这些名字）
#        ⇒ 守望误判「已在跑」⇒ 通道永远不重启（11-18 10:03 真实踩到：synth2/synth3/gateC 都没起来）。
PAT[gateA]="$W/.pid_gateA"
PAT[gateB]="$W/.pid_gateB"
# 🟢 2026-11-18 10:25 新增门禁通道 C（GPU0，高权重交叉段）
PAT[gateC]="$W/.pid_gateC"
# 🟢 2026-11-18 11:15 新增门禁通道 D/E（sb3 家族按权重段/成员集切分 ⇒ 通道间不抢臂）
PAT[gateD]="$W/.pid_gateD"
PAT[gateE]="$W/.pid_gateE"
PAT[synth2]="$W/.pid_synth2d4"
PAT[synth3]="$W/.pid_synth_gpu3"
PAT[boardw]="^/home/huangzhe/anaconda3/envs/uav/bin/python -u manu/pipeline/npu/board_t4_watch.py"
PAT[promow]="^/home/huangzhe/anaconda3/envs/uav/bin/python -u manu/pipeline/npu/auto_promote_watch.py"
PAT[guard]="baseline_guard.py"
# 🟢 2026-11-18 16:10：新增 GPU0/GPU1 两条合成波 + Track-B 升轨守望
PAT[synth0]="$W/.pid_synth_gpu0"
PAT[synth1]="$W/.pid_synth_gpu1"
PAT[trackbw]="^/home/huangzhe/anaconda3/envs/uav/bin/python -u manu/pipeline/npu/trackb_promote_watch.py"
PAT[tipdose]="$W/.pid_tipdose"
# 🛡 nano 夜间管线内存看门狗（门禁缓存构建会把内存吃穿 ⇒ 必须有分级刹车）
PAT[nw]="^bash /mnt/data/siping/runs/npu20ms/nano_search/scripts/nano_watchdog.sh"
# 🔴 2026-11-19 11:50：mreport 改为「PID 文件判活 + 独立调度器」
#   坑：原 PAT 是正则 `^bash .*run_morning_report4.sh`，但 supervisor 的「每通道重启上限 30 次」
#        已被 mreport 吃光（26~30 次）⇒ 之后永远是「[跳过] mreport 重启超限」⇒ 晨报彻底不出现。
#        且 run_morning_report4.sh 的 AT 一旦解析成过去时间就会秒退 ⇒ 每 90s 重启一次空烧机时。
#   修法：改用 PID 文件判活（.pid_mreport_sched）+ 独立长驻调度器 morning_report_sched.sh。
PAT[mreport]="$W/.pid_mreport_sched"
for k in "${!PAT[@]}"; do REST[$k]=0; done
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] $END ⇒ 守望退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  # 🟢 2026-11-19 13:00 soup3_orphan_reaper：波壳被杀 ⇒ soup3 变孤儿(ppid==1)继续烧卡
  #   ⇒ 每轮收割一次；活着的波（timeout 父进程在）不受影响。
  for _o in $(ps -eo pid,ppid,args | grep "night_soup3[.]py" | awk '$2==1{print $1}'); do
    kill -9 "$_o" 2>/dev/null && log "[收割] 孤儿 soup3 pid=$_o"
  done
  for k in gateA gateB gateC guard synth0 synth1 synth2 synth3 trackbw boardw promow mreport nw tipdose; do
    if alive "${PAT[$k]}"; then continue; fi
    [ "${REST[$k]}" -lt 30 ] || { log "[跳过] $k 重启超限"; continue; }
    # 🔴🔴🔴 2026-11-20 16:58 **汤轴全停开关**（§280）
    #   汤空间已证伪封死（轴⑨/⑭ 30 臂三轴全过 0 个；同成员集 ≈ 同一模型；供体/骨架交集为空）
    #   ⇒ 周末算力全部让给训练轴。开关 = $W/.no_waves 存在 ⇒ 不再拉起任何汤产线。
    if [ -f "$W/.no_waves" ]; then
      case "$k" in
        synth0|synth1|synth2|synth3|tipdose) log "[汤轴停] .no_waves ⇒ 跳过 $k"; continue ;;
      esac
    fi
    if [ "${avail:-0}" -lt 40960 ]; then log "[闸门] avail=${avail}MB < 40960MB ⇒ 暂不重启 $k"; continue; fi
    REST[$k]=$(( ${REST[$k]} + 1 ))
    log "[重启] $k（第 ${REST[$k]} 次）avail=${avail}MB"
    case $k in
      gateA) setsid nohup bash $W/scripts/gate_loop_d1.sh >> $W/day_gateA.log 2>&1 < /dev/null 9>&- & ;;
      gateB) setsid nohup bash $W/scripts/gate_loop_d1b.sh >> $W/day_gateB.log 2>&1 < /dev/null 9>&- & ;;
      gateC) setsid nohup bash $W/scripts/gate_loop_d1c.sh >> $W/day_gateC.log 2>&1 < /dev/null 9>&- & ;;
      gateD) : ;;  # 🔴 2026-11-18 17:35 停用：门禁通道铁律 ≤3（A/B/C）⇒ D/E 不再拉起
      gateE) : ;;  #    （旧守望实例解析的是含 D/E 的旧循环体 ⇒ 曾把门禁开到 5 通道 ×4 分片 ×4.5GB = 68GB 把内存打爆）
      synth0) setsid nohup bash $W/scripts/synth_gpu0_d6.sh >> $W/d6_gpu0.log 2>&1 < /dev/null 9>&- & ;;
      synth1) setsid nohup bash $W/scripts/synth_gpu1_d5.sh >> $W/d5_gpu1.log 2>&1 < /dev/null 9>&- & ;;
      trackbw) setsid nohup $PY -u manu/pipeline/npu/trackb_promote_watch.py --watch --interval 600 \
               --deadline "${SUP_END:-2026-11-30 09:00}" >> $W/trackb_promote_watch.log 2>&1 < /dev/null 9>&- & ;;
      synth2) setsid nohup bash $W/scripts/synth_gpu2_d4.sh >> $W/d4_gpu2.log 2>&1 < /dev/null 9>&- & ;;
      synth3) setsid nohup bash $W/scripts/synth_gpu3_d3.sh >> $W/d3_gpu3.log 2>&1 < /dev/null 9>&- & ;;
      boardw) setsid nohup $PY -u manu/pipeline/npu/board_t4_watch.py --watch --interval 600 \
               --deadline "${SUP_END:-2026-11-30 09:00}" >> $W/board_t4_watch.log 2>&1 < /dev/null 9>&- & ;;
      promow) setsid nohup $PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 \
               --deadline "${SUP_END:-2026-11-30 09:00}" >> $W/auto_promote_watch.log 2>&1 < /dev/null 9>&- & ;;
      guard) setsid nohup $PY -u /tmp/baseline_guard.py "${SUP_END:-2026-11-30 09:00}" >> $W/baseline_guard.log 2>&1 < /dev/null 9>&- & ;;
      nw) setsid nohup bash $W/scripts/nano_watchdog.sh >> $W/nano_watchdog_runner.log 2>&1 < /dev/null 9>&- & ;;
      mreport) setsid nohup bash $W/scripts/morning_report_sched.sh \
               >> $W/morning_report_sched_runner.log 2>&1 < /dev/null 9>&- & ;;
      tipdose) setsid nohup bash $W/scripts/tipdose_loop.sh >> $W/tipdose_loop_runner.log 2>&1 < /dev/null 9>&- & ;;
    esac
    sleep 30
  done
  # ---- sb3 家族标签统一（粗粒度 ⇒ 关A/关B 能攒票）----
  $PY -u manu/pipeline/npu/scripts/retag_sb3.py >> $W/retag_sb3.log 2>&1 9>&-
  # ---- 六项全过猎手（补升轨链「家族 n>=4」门槛造成的漏报）----
  $PY -u $W/scripts/six_hunter.py >> $W/six_hunter_runner.log 2>&1 9>&-
  # ---- 台账回填（把磁盘 gate_report.json 的官方三轨读数补进台账；2026-11-19 一次补 92 臂）----
  $PY -u $W/scripts/backfill_official.py >> $W/backfill_official.log 2>&1 9>&-
  # ---- 清理「基础设施失败」的门禁目录（结论不可信 ⇒ 放回队列重打）----
  $PY -u $W/scripts/clean_infra.py >> $W/clean_infra.log 2>&1 9>&-
  # ---- 自动刷新门禁优先队列（近失锚点邻域优先 + 回收僵死 claim）----
  $PY -u $W/scripts/refresh_prio.py >> $W/refresh_prio.log 2>&1 9>&-
  sleep 60 9>&-
done
