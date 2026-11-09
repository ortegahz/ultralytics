#!/bin/bash
# 重启 th=0.20 档的服务器端收尾脚本（用修好的 pgrep 锚定模式）
set -u
cd /tmp/pycharm_project_10ae9e2e || exit 2

DATA_ROOT=/mnt/data/siping/datasets/manu
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_th020_20261109"
LOG=/mnt/data/siping/runs/zongheng_rc2_osd_th020_20261109.log
FINAL_LOG=/mnt/data/siping/runs/zongheng_rc2_osd_th020_finalize.log

echo "[RESTART] $(date '+%F %T') stopping the old finalizer (it predates the pgrep anchoring fix)"
pkill -f "finalize_rc2_zongheng_osd.sh" 2>/dev/null || true
sleep 3

export OUT LOG
export SPLIT_LOCAL=manu/data/zongheng_ft_split_th020.json
export REVIEW_LOCAL=manu/docs/dataset/zongheng-tk-th020.md
export PRELABEL_OUT="$DATA_ROOT/纵横AI数据采集_preannot_rc2_th020"

setsid nohup bash manu/videos/finalize_rc2_zongheng_osd.sh > "$FINAL_LOG" 2>&1 < /dev/null &
sleep 8
echo "[RESTART] new finalizer:"
head -5 "$FINAL_LOG"
echo "alive: $(pgrep -cf '^bash .*finalize_rc2_zongheng_osd\.sh' || echo 0)"
