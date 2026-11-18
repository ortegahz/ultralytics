#!/bin/bash
# 明早自动报告 v3（2026-11-17 17:5x）：加「权威六项审计」+ 定向门禁通道尾巴
#   改动：① 时间提前到 08:40（用户 09:00 到）② 加 six_criteria_audit（以磁盘 gate_report.json 为准）
#         ③ 加 gate_priority（岭线 w56 定向门禁）日志 ④ 门槛票数/双关进度
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
W=/mnt/data/siping/runs/npu20ms/nano_search
AT="${1:-2026-11-18 08:40}"
STAMP=$(date -d "$AT" +%Y%m%d)
OUT=$W/morning_report_${STAMP}.txt
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

while [ "$(date +%s)" -lt "$(date -d "$AT" +%s)" ]; do sleep 60; done
sleep 30

{
  echo "===== 权威六项审计（磁盘 gate_report.json 为准）====="
  $PY manu/pipeline/npu/six_criteria_audit.py --top 15
  echo
  echo "===== 通用晨报 ====="
  $PY manu/pipeline/npu/morning_report_nano.py
  echo
  echo "===== 升轨记录 promotion_ready.json ====="
  cat "$W/../promotion_ready.json" 2>/dev/null || cat /mnt/data/siping/runs/npu20ms/promotion_ready.json 2>/dev/null || echo "（未升轨）"
  echo
  echo "===== 定向门禁通道（岭线 w56）====="
  tail -n 25 "$W/gate_priority_run.log" 2>/dev/null
  tail -n 10 "$W/gate_priority.log" 2>/dev/null
  echo
  echo "===== 门禁优先循环尾部 ====="
  tail -n 30 "$W/gate_watch_run.log" 2>/dev/null
  echo
  echo "===== 种子复现波尾部 ====="
  tail -n 40 "$W/seed_replicate_run.log" 2>/dev/null
  echo
  echo "===== 岭线精扫 / 同盆地三兄弟 / 复现汤 尾部 ====="
  for f in soup_new_1220000.log soup_new_1230000.log soup_new_1240000.log soup_new_1250000.log soup_new_1310000.log; do
    echo "--- $f"; tail -n 14 "$W/$f" 2>/dev/null
  done
  echo
  echo "===== 板卡 T4 守望尾部 ====="
  tail -n 20 "$W/board_t4_watch.log" 2>/dev/null
  echo
  echo "===== 升轨守望尾部 ====="
  tail -n 20 "$W/auto_promote_watch.log" 2>/dev/null
  echo
  echo "===== 资源水位（watchdog 最后 5 条）====="
  tail -n 5 /tmp/load_watchdog.log 2>/dev/null
} > "$OUT" 2>&1

cp -f "$OUT" "$W/morning_report_latest.txt"
echo "报告已生成：$OUT"
