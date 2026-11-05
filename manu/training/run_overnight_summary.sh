#!/bin/bash
# ==============================================================================
# 过夜收尾：到点自动产出一份 OVERNIGHT_SUMMARY.md / .json（2026-11-04 夜）
#
# 为什么单独一个调度器
# ------------------
# · host 18:00 关机 ⇒ 明早 09:00 的第一件事必须是「**只读一个文件**」就能拿到全部结论；
# · 四条链（clip 受控 / 插值扫参 / 训练臂 / 决策层）各自落盘，汇总器负责**统一判定**；
# · 汇总器**幂等**：可重复跑；墙钟截止后仍可手动再跑一次。
#
# 用法（服务器）：
#   setsid nohup bash manu/training/run_overnight_summary.sh > /mnt/data/siping/runs/nogmc_t3/logs/overnight_summary.out 2>&1 < /dev/null &
#   # 手动提前跑一次：
#   bash manu/training/run_overnight_summary.sh --now
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
OUT="${OUT:-/mnt/data/siping/runs/nogmc_t3}"
REF="${REF:-/mnt/data/siping/runs/nogmc_ref_weekend}"
LOG="$OUT/logs/overnight_summary.log"
AT="${AT:-2026-11-05 07:05}"
NOW=0
[ "${1:-}" = "--now" ] && NOW=1

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT/logs"

say(){ printf '[%s] [overnight] %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; echo "[$(date '+%F %T')] [overnight] $*"; }

build(){
  say "生成总汇总 …"
  python manu/training/build_overnight_summary.py \
    --out-root "$OUT" \
    --out-md "$OUT/OVERNIGHT_SUMMARY.md" \
    --out-json "$OUT/OVERNIGHT_SUMMARY.json" >> "$LOG" 2>&1
  local rc=$?
  say "汇总 rc=$rc ⇒ $OUT/OVERNIGHT_SUMMARY.md"
  [ $rc -eq 0 ] || return $rc
  say "──── 判定摘要 ────"
  sed -n '/## ⑥ 裁定/,/## ⑦/p' "$OUT/OVERNIGHT_SUMMARY.md" | head -40 >> "$LOG" 2>/dev/null || true
  return 0
}

if [ "$NOW" = "1" ]; then
  build
  exit $?
fi

END=$(date -d "$AT" '+%s' 2>/dev/null || echo 0)
say "等待到 $AT 再汇总（host 18:00 关机不影响本进程）"
while :; do
  [ "$(date +%s)" -ge "$END" ] && break
  sleep 60
done
build
say "✅ 过夜汇总完成 ⇒ $OUT/OVERNIGHT_SUMMARY.md"
