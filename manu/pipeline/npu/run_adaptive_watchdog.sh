#!/bin/bash
# 调度器守护 v2（2026-11-11 18:15，host 关机前加固）
#   ① 调度器进程不在            ⇒ 自动重启
#   ② 日志静默 > 阈值(45min)    ⇒ 判定卡死 ⇒ 杀掉并重启（重启前删可能写一半的 epoch5.pt）
#   ③ 调度器代码 md5 变了 且 刚跑完一轮 ⇒ **在轮次边界重启**，让新代码生效
#   ④ 续跑轮号 = max(台账已完成轮+1, 最近开跑轮)
#      🔴 v2 修：v1 只看台账 `rounds`，而轮记录是**轮末**才写 ⇒ 轮中途崩溃时台账查不到该轮
#         ⇒ v1 会算成「从第 1 轮重跑」（浪费 ~2.5 h + 重复配方污染 recipe_stats 判优）
#         ⇒ v2 增加 round_state.json（runner 每轮开头原子写 last_started_round）
#   ⑤ 重启时按**进程组**杀（kill -9 -PGID）⇒ 不留孤儿训练进程占着显卡
#   ⑥ 轮次边界判定用「台账已完成轮」而非 len(rounds)（v1 已修，v2 保留）
# 用法（服务器）：setsid nohup bash manu/pipeline/npu/run_adaptive_watchdog.sh <deadline> <silence_s> [pending_code_md5] > /dev/null 2>&1 &
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/watchdog.log
STATE=$W/watchdog_state.json
# 监控**所有会改变判优/搜索行为的代码文件**（2026-11-11 18:05）
#   只盯 adaptive_runner.py 是不够的：nano_search.py（选 epoch 规则 / 台账写盘）变了也一样要生效。
CODE=manu/pipeline/npu/adaptive_runner.py
CODE2=manu/pipeline/npu/nano_search.py
TRIALS=/mnt/data/siping/runs/npu20ms_offcal/trials
DEADLINE="${1:-2026-11-12 09:00}"
SILENCE="${2:-2700}"
PENDING_MD5="${3:-}"
DL=$(date -d "$DEADLINE" +%s)

state_get() {
  python3 -c "
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: d={}
print(d.get('code_md5',''), d.get('rounds',0))
" "$STATE"
}
state_set() {
  python3 -c "
import json,sys
json.dump({'code_md5':sys.argv[2],'rounds':int(sys.argv[3])}, open(sys.argv[1],'w'))
" "$STATE" "$1" "$2"
}
# 台账里**已完成**的最大轮号（轮末才写 ⇒ 轮中途崩溃时查不到该轮）
completed_round() {
  python3 -c "
import json,sys
try:
    rs=[int(r.get('round',0)) for r in (json.load(open(sys.argv[1])).get('rounds') or []) if isinstance(r,dict)]
    print(max(rs) if rs else 0)
except Exception: print(0)
" "$W/nano_search_ledger.json"
}
# 最近**开跑**的轮号（runner 每轮开头原子写；轮中途被杀时它仍指向那一轮）
started_round() {
  python3 -c "
import json,sys
try: print(int(json.load(open(sys.argv[1])).get('last_started_round',0)))
except Exception: print(0)
" "$W/round_state.json"
}
# 续跑轮号 = max(已完成+1, 最近开跑, 1) ⇒ 崩在哪一轮就从哪一轮续
next_round() {
  python3 -c "
import sys
print(max(int(sys.argv[1])+1, int(sys.argv[2]), 1))
" "$1" "$2"
}

if [ ! -f "$STATE" ]; then
  CUR0=$(md5sum $CODE $CODE2 | md5sum | cut -d" " -f1)
  state_set "${PENDING_MD5:-$CUR0}" "$(completed_round)"
  echo "[$(date "+%F %T")] [watchdog] 启动｜deadline=$DEADLINE｜静默阈值=${SILENCE}s｜运行中代码 md5=${PENDING_MD5:-$CUR0}" >> $LOG
fi

while true; do
  sleep 300
  NOW=$(date +%s)
  if [ "$NOW" -gt "$DL" ]; then echo "[$(date "+%F %T")] [watchdog] 已过截止 ⇒ 退出" >> $LOG; break; fi
  PID=$(pgrep -f "adaptive_runner.py --deadline" | head -1 || true)
  MT=$(stat -c %Y "$W/adaptive.log" 2>/dev/null || echo 0)
  AGE=$((NOW - MT))
  read -r S_MD5 S_ROUNDS <<< "$(state_get)"
  CUR=$(md5sum $CODE $CODE2 | md5sum | cut -d" " -f1)
  COMP=$(completed_round)
  ST=$(started_round)
  NEXT=$(next_round "$COMP" "$ST")
  REASON=""
  HARD=0
  if [ -z "${PID:-}" ]; then
    REASON="调度器不在"
  elif [ "$AGE" -gt "$SILENCE" ]; then
    REASON="日志静默 ${AGE}s（疑似卡死）"
    HARD=1
  elif [ "$CUR" != "$S_MD5" ] && [ "$COMP" -gt "$S_ROUNDS" ]; then
    REASON="代码已更新（${S_MD5:0:8}→${CUR:0:8}），在轮次边界生效"
  fi
  if [ -n "$REASON" ]; then
    echo "[$(date "+%F %T")] [watchdog] 重启调度器：$REASON ⇒ --start-round $NEXT（已完成=$COMP 最近开跑=$ST）" >> $LOG
    if [ -n "${PID:-}" ]; then
      PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ')
      if [ -n "${PGID:-}" ]; then kill -9 -"$PGID" 2>/dev/null || true; fi
      kill -9 "$PID" 2>/dev/null || true
      pkill -9 -f "optuna_mix_ratio_searc[h].py" 2>/dev/null || true
      sleep 5
    fi
    if [ "$HARD" = "1" ]; then
      find "$TRIALS" -name "epoch5.pt" -mmin -1 -delete 2>/dev/null || true
    fi
    setsid nohup bash manu/pipeline/npu/run_adaptive_night.sh "$DEADLINE" 6 18 "$NEXT" >> /dev/null 2>&1 < /dev/null &
    sleep 20
    state_set "$CUR" "$COMP"
  elif [ "$COMP" -gt "$S_ROUNDS" ]; then
    state_set "$S_MD5" "$COMP"
  fi
done
