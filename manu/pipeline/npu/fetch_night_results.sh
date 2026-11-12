#!/bin/bash
# 把夜间搜索产物从训练服务器拉回本地 workspace（host 开机后跑一次即可）
#   用法：bash manu/pipeline/npu/fetch_night_results.sh [日期标签]
#   拉回：台账 / 报告 / 夜间台账 / 可升轨候选 + md5 双侧比对
set -u
LOCAL_ROOT=/media/manu/1TB-Volume/workspace/ultralytics
TAG="${1:-$(date +%Y%m%d)}"
SSH="ssh -F /dev/null -p 32222 -o ConnectTimeout=10 -o StrictHostKeyChecking=no huangzhe@192.168.99.40"
SCP="scp -F /dev/null -P 32222 -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
REMOTE_REPO=/tmp/pycharm_project_10ae9e2e
WORK=/mnt/data/siping/runs/npu20ms/nano_search
DST="$LOCAL_ROOT/manu/memory/night_$TAG"
mkdir -p "$DST"

FILES="nano_search_ledger.json final_report.md beat_to_pass_report.md night_log.md promotion_ready.json adaptive.log watchdog.log triad_gate_report.json specific_gate_report.json round_state.json watchdog_state.json BOARD_PHASE_DISABLED.txt"
for f in $FILES; do
  $SCP huangzhe@192.168.99.40:$WORK/$f "$DST/$f" >/dev/null 2>&1 || echo "[skip] $f"
done
$SCP huangzhe@192.168.99.40:$REMOTE_REPO/manu/memory/adaptive_night_20261111.md "$DST/adaptive_night_20261111.md" >/dev/null 2>&1 || echo "[skip] adaptive_night"

echo "=== md5 双侧比对 ==="
cd "$DST" || exit 1
for f in $(ls); do
  L=$(md5sum "$f" | cut -d" " -f1)
  R=$($SSH "md5sum $WORK/$f 2>/dev/null || md5sum $REMOTE_REPO/manu/memory/$f 2>/dev/null" 2>/dev/null | cut -d" " -f1)
  if [ "$L" = "$R" ]; then echo "[OK]   $f  $L"; else echo "[DIFF] $f  local=$L remote=$R"; fi
done
echo "=== 落盘目录：$DST ==="
