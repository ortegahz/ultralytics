#!/bin/bash
# 明早自动出报告（host 已关机也能跑：独立进程，PPID=1）
# 用法（服务器）：
#   setsid nohup bash manu/pipeline/npu/run_morning_report.sh "2026-11-13 09:10" >/dev/null 2>&1 </dev/null &
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
W=/mnt/data/siping/runs/npu20ms/nano_search
AT="${1:-2026-11-13 09:10}"
STAMP=$(date -d "$AT" +%Y%m%d)
OUT=$W/morning_report_${STAMP}.txt
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

while [ "$(date +%s)" -lt "$(date -d "$AT" +%s)" ]; do sleep 60; done
sleep 45

{
  echo "===== 明早自动报告 $(date "+%F %T") ====="
  echo "--- ① 进程状态（调度器到点应已退出） ---"
  pgrep -af "adaptive_runner.py --deadline" || echo "调度器已退出 ✔"
  pgrep -af "run_adaptive_watchdog.sh" || echo "守护已退出 ✔"
  echo
  echo "--- ② 轮次状态 ---"
  cat $W/round_state.json 2>/dev/null; echo
  echo
  echo "--- ③ 热升级核查（新代码是否在 R8 生效） ---"
  grep -n "\[R8\]\|优先格\|PRIORITY\|多epoch\|multi_epoch" $W/adaptive.log | tail -25 || echo "未见 R8 标记"
  echo "（守护日志里应有『代码已更新…在轮次边界生效』一行）"
  grep -n "代码已更新" $W/watchdog.log | tail -5
  echo
  echo "--- ④ adaptive.log 尾部 60 行 ---"
  tail -60 $W/adaptive.log
  echo
  echo "--- ⑤ watchdog.log 尾部 15 行 ---"
  tail -15 $W/watchdog.log
  echo
  echo "--- ⑥ 夜汇总（四块：总量/六项全过/最接近候选/配方组统计） ---"
  $PY manu/pipeline/npu/night_summary.py 2>&1
  echo
  echo "--- ⑦ 升轨候选清单 ---"
  $PY manu/pipeline/npu/promote_candidate.py --list 2>&1 | head -60
  echo
  echo "--- ⑧ 板卡状态与恢复指令 ---"
  ls -la /home/huangzhe/.ssh/ | grep -i board
  cat $W/BOARD_PHASE_DISABLED.txt 2>/dev/null
} > "$OUT" 2>&1

echo "[$(date "+%F %T")] 明早报告已写入 $OUT"
