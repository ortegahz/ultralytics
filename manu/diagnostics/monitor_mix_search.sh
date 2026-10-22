#!/usr/bin/env bash
# 搜索监控：记录每次检查的快照（本地执行，只读服务器）
set -uo pipefail
OUT=/tmp/mix_search_monitor.log
SSH="ssh -p 32222 huangzhe@192.168.99.40"

{
  echo "===== $(date '+%F %T') ====="
  $SSH 'cd /tmp/pycharm_project_10ae9e2e && date "+server %T" && \
    echo "dispatcher: $(ps -eo pid,etime,args | grep "optuna_mix_ratio_search.py --output-root" | grep -v grep | head -1)" && \
    echo "settled: $(ls runs/optuna_mix_ratio/trials/*/trial_result.json 2>/dev/null | wc -l)" && \
    echo "--- 在跑 ---" && ps -eo args | grep -o "trial-number [0-9]*" | sort -u | tr "\n" " " && echo && \
    echo "--- 对照组结果 ---" && \
    for t in 0088 0089 0090 0091; do \
      f=runs/optuna_mix_ratio/trials/trial_$t/trial_result.json; \
      if [ -f "$f" ]; then echo "trial_$t SETTLED"; else echo "trial_$t pending"; fi; \
    done' 2>&1 | grep -v "WARNING\|post-quantum\|store now\|may need"
  echo
} >> "$OUT"
