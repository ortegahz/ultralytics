#!/bin/bash
# 🔴🔴 夜间心跳 v4（2026-11-18 17:45）
# v1 自匹配 bug → v2 方括号修复 → v3 残留 flock 清理 → v4 **各组件独立日志**
#   （v3 把所有组件的 stdout 都接到 heartbeat.log ⇒ board_t4_watch.log 等**看起来「停了」**，
#     实际都在 heartbeat.log 里 ⇒ 极易误判。v4 恢复「一组件一日志」。）
set -u
W=/mnt/data/siping/runs/npu20ms/nano_search
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
LOG=$W/heartbeat.log
cd "$REPO" || exit 1
log(){ echo "[$(date "+%F %T")] $*" >> "$LOG"; }
AVAIL=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
alive_p(){ ps -eo args | grep -q -- "$1"; }

need_light(){ # need_light <pat> <logfile> <cmd...>
  local pat="$1" lf="$2"; shift 2
  alive_p "$pat" && return 0
  setsid nohup "$@" >> "$lf" 2>&1 < /dev/null 9>&- &
  log "[拉起-轻] $pat → $(basename "$lf")"
}
need_heavy(){ # need_heavy <pat> <logfile> <cmd...>
  local pat="$1" lf="$2"; shift 2
  alive_p "$pat" && return 0
  if [ "${AVAIL:-0}" -lt 40960 ]; then log "[跳过-重] $pat（avail=${AVAIL}MB < 40960MB）"; return 0; fi
  setsid nohup "$@" >> "$lf" 2>&1 < /dev/null 9>&- &
  log "[拉起-重] $pat → $(basename "$lf")（avail=${AVAIL}MB）"
}
need_wave(){ # need_wave <pat> <lockfile> <script> <logfile>
  local pat="$1" lock="$2" scr="$3" lf="$4"
  alive_p "$pat" && return 0
  [ -e "$lock" ] && { rm -f "$lock"; log "[清残留锁] $(basename "$lock")"; }
  if [ "${AVAIL:-0}" -lt 40960 ]; then log "[跳过-波] $pat（avail=${AVAIL}MB < 40960MB）"; return 0; fi
  setsid nohup bash "$scr" >> "$lf" 2>&1 < /dev/null 9>&- &
  log "[拉起-波] $pat → $(basename "$lf")（avail=${AVAIL}MB）"
}

need_train(){ # need_train <pat> <logfile> <cmd...>  —— 训练轴专用（内存门槛与训练轴自身一致 = 26 GB）
  local pat="$1" lf="$2"; shift 2
  alive_p "$pat" && return 0
  if [ "${AVAIL:-0}" -lt 26624 ]; then log "[跳过-训] $pat（avail=${AVAIL}MB < 26624MB）"; return 0; fi
  setsid nohup "$@" >> "$lf" 2>&1 < /dev/null 9>&- &
  log "[拉起-训] $pat → $(basename "$lf")（avail=${AVAIL}MB）"
}

need_light "[d]ay_supervisor_d1.sh"    $W/day_supervisor_runner.log bash $W/scripts/day_supervisor_d1.sh
need_light "[n]ano_watchdog.sh"        $W/nano_watchdog_runner.log bash $W/scripts/nano_watchdog.sh
# 🔴 2026-11-20 16:08 **截止时刻对齐到 09:20**
#   坑：本脚本与 day_supervisor_d1.sh 是**两个竞争拉起器**（本脚本 cron 每 2 min，守望每 ~90 s），
#        两者都负责拉 board_t4_watch / trackb_promote_watch / auto_promote_watch / baseline_guard。
#        本脚本里写的是**旧的 08:30**，守望里是 09:20 ⇒ **谁先跑谁说了算**
#        ⇒ 实测 16:02 由本脚本拉起 ⇒ 三个守望又变回 08:30（比用户要的 09:00 早 30 min）。
#   修法：把本脚本的 08:30 全部改成 09:20 ⇒ 两个拉起器**同一口径** ⇒ 谁先跑都一样。
need_light "[b]oard_t4_watch.py"       $W/board_t4_watch.log       $PY -u manu/pipeline/npu/board_t4_watch.py --watch --interval 600 --deadline "2026-11-23 09:20"
need_light "[t]rackb_promote_watch.py" $W/trackb_promote_watch.log $PY -u manu/pipeline/npu/trackb_promote_watch.py --watch --interval 600 --deadline "2026-11-23 09:20"
need_light "[a]uto_promote_watch.py"   $W/auto_promote_watch.log   $PY -u manu/pipeline/npu/auto_promote_watch.py --watch --interval 600 --deadline "2026-11-23 09:20"
need_light "[b]aseline_guard.py"       $W/baseline_guard.log       $PY -u /tmp/baseline_guard.py "2026-11-23 09:20"

need_heavy "[g]ate_loop_d1.sh"         $W/day_gateA.log bash $W/scripts/gate_loop_d1.sh
need_heavy "[g]ate_loop_d1b.sh"        $W/day_gateB.log bash $W/scripts/gate_loop_d1b.sh
need_heavy "[g]ate_loop_d1c.sh"        $W/day_gateC.log bash $W/scripts/gate_loop_d1c.sh

# 🟢🔴 2026-11-20 16:45 **训练轴自愈拉起**（§280）—— 周末 60 h 无人值守的最后一块拼图
#   背景：训练轴此前**只由手工脚本拉起**（`weekend_train_axis_20261120.sh`）⇒ 一旦它挂掉
#         （例如「连续 8 次训练失败 ⇒ 熔断停机」，或被 OOM/SIGKILL），**没有任何组件会把它拉回来**
#         ⇒ 整个周末的算力就白扔了。本行把它纳入心跳管理（**cron 每 2 min 重跑 ⇒ 天然拿到新版本**）。
#   幂等：`need_heavy` 先判活（`night_train_axis.py` 在跑就跳过）；真拉起时走
#         `weekend_train_axis_20261120.sh`，该脚本自身会先停旧 runner / 训练子进程再拉起，
#         且训练轴**按臂号幂等续跑**（权重已存在 ⇒ 跳过训练直接评测）⇒ 不会重复烧算力、不会撞臂号目录。
#   口径：ARMS=3 / GPU 0,2,3 / 截止 2026-11-23 08:00 / polish 600 / pt1 30（与当前实例完全一致）。
need_train "[n]ight_train_axis.py"     $W/night_train_axis_runner.log bash $W/scripts/weekend_train_axis_20261120.sh

# 🔴🔴🔴 2026-11-20 16:58 **汤轴全停开关**（§280）
#   实测结论：汤（权重平均）空间**已证伪封死** —— ① 轴⑨(2050xxx)+轴⑭(2060xxx) 共 30 臂
#   「三轴全过」= **0 个**（历史臂是 301/1170 = 25.7%）；② 同成员集 ≈ 同一模型
#   （基线成员集 1178 臂，nano 极差仅 2.36e-4 = 1 阶，T1 逐位相同）；
#   ③ 供体/骨架关键交集为**空集** ⇒ 汤空间封死。
#   ⇒ 周末 4 条 GPU 通道全部让给**训练轴**（唯一能产出新冠军成员的通道）。
#   开关：$W/.no_waves 存在 ⇒ 本脚本不再拉起任何波次壳（随时删掉即可回滚）。
if [ -f "$W/.no_waves" ]; then
  log "[汤轴停] .no_waves 存在 ⇒ 不拉起波次壳（算力让给训练轴）"
else
  need_wave "[s]ynth_gpu0_d6.sh" "$W/.lock_d6_gpu0" "$W/scripts/synth_gpu0_d6.sh" "$W/d6_gpu0.log"
  need_wave "[s]ynth_gpu1_d5.sh" "$W/.lock_d5_gpu1" "$W/scripts/synth_gpu1_d5.sh" "$W/d5_gpu1.log"
  need_wave "[s]ynth_gpu2_d4.sh" "$W/.lock_d4_gpu2" "$W/scripts/synth_gpu2_d4.sh" "$W/d4_gpu2.log"
  need_wave "[s]ynth_gpu3_d3.sh" "$W/.lock_d3_gpu3" "$W/scripts/synth_gpu3_d3.sh" "$W/d3_gpu3.log"
fi

log "[心跳] avail=${AVAIL}MB 组件检查完成"
