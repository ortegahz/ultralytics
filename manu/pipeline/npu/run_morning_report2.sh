#!/bin/bash
# 明早自动报告（夜间训练轴专用版，2026-11-17 §196；2026-11-17 17:50 加「夜间汤族/种子复现」日志）
#   用法（服务器）：
#     setsid nohup bash manu/pipeline/npu/run_morning_report2.sh "2026-11-18 09:10" >/dev/null 2>&1 </dev/null &
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
W=/mnt/data/siping/runs/npu20ms/nano_search
AT="${1:-2026-11-18 09:10}"
STAMP=$(date -d "$AT" +%Y%m%d)
OUT=$W/morning_report_${STAMP}.txt
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

while [ "$(date +%s)" -lt "$(date -d "$AT" +%s)" ]; do sleep 60; done
sleep 45

{
  $PY manu/pipeline/npu/morning_report_nano.py
  echo
  echo "===== 升轨记录 promotion_ready.json ====="
  cat "$W/../promotion_ready.json" 2>/dev/null || cat /mnt/data/siping/runs/npu20ms/promotion_ready.json 2>/dev/null || echo "（未升轨）"
  echo
  echo "===== 种子复现波（冠军配方换种子）尾部 ====="
  tail -n 40 "$W/seed_replicate.log" 2>/dev/null
  echo
  echo "===== 门禁优先循环尾部 ====="
  tail -n 30 "$W/gate_watch_run.log" 2>/dev/null
  echo
  echo "===== 岭线精扫 / 复现汤 尾部 ====="
  for f in soup_new_1220000.log soup_new_1210000.log soup_new_1310000.log; do
    echo "--- $f"; tail -n 12 "$W/$f" 2>/dev/null
  done
  echo
  echo "===== 训练轴日志尾部（最后 40 行）====="
  tail -n 40 "$W/night_train_axis.log" 2>/dev/null
} > "$OUT" 2>&1

cp -f "$OUT" "$W/morning_report_latest.txt"
echo "报告已生成：$OUT"
