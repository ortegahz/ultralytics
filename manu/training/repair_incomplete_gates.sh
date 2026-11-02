#!/bin/bash
# 补跑"门禁基础设施失败"的臂（CUDA OOM 瞬态问题，权重完好），然后重判 + 重出 SUMMARY
#
# 关键点（踩过的坑）：
#   · 从日志抽出的命令行带 "$ " 前缀（回显），eval 前必须去掉，否则 rc=127；
#   · auto_epoch_gate_check.py 的 --gpus 是**分片数(int)**，不是 GPU 列表；
#     GPU 列表是 --cache-gpus；--device / --infer-gpu 是单卡 index。
#   · 4 个门禁并发会 CUDA OOM ⇒ 这里**串行**跑，每个用 4 卡并行重建缓存。
set -uo pipefail
cd /tmp/pycharm_project_10ae9e2e
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
export PYTHONPATH=.
ROOT=/mnt/data/siping/runs/nogmc_weekend
LOG=/mnt/data/siping/runs/nogmc_repair.log
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for T in trial_0006 trial_0011 trial_0068; do
  CMD=$(grep -h 'auto_epoch_gate_check.py' "$ROOT/logs/$T.log" | grep '^\$' | tail -1 | sed 's/^\$ //')
  if [ -z "$CMD" ]; then say "SKIP $T: 取不到门禁命令"; continue; fi
  NEW=$(printf '%s' "$CMD" | sed -E 's/--gpus [0-9]+/--gpus 4/; s/--device [0-9]+/--device 0/; s/--infer-gpu [0-9]+/--infer-gpu 0/; s/--cache-gpus [0-9,]+/--cache-gpus 0,1,2,3/')
  say "重跑 $T 门禁（4 卡并行缓存）"
  say "  CMD: $NEW"
  eval "$NEW" >> "$ROOT/trials/$T/gate_repair.log" 2>&1
  say "  $T 门禁 rc=$?"
  sleep 15
done

say "重判（用参照系重算所有 trial 判决并写回 CSV）"
python manu/training/rejudge_with_reference.py \
  --out-root "$ROOT" --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --epochs 0 --apply 2>&1 | tee -a "$LOG"
say "重出 SUMMARY"
python manu/training/summarize_nogmc_search.py \
  --out-root "$ROOT" --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json --top 20 2>&1 | tail -6 | tee -a "$LOG"
cp -f "$ROOT/SUMMARY.md" /mnt/data/siping/runs/nogmc_weekend_SUMMARY_final.md
cp -f "$ROOT/optuna_summary.csv" /mnt/data/siping/runs/nogmc_weekend_summary_final.csv
STAMP=$(date '+%Y%m%d_%H%M')
mkdir -p "/mnt/data/siping/runs/nogmc_weekend_backup_$STAMP"
cp -f "$ROOT/SUMMARY.md" "$ROOT/optuna_summary.csv" "$ROOT/study.db" "/mnt/data/siping/runs/nogmc_weekend_backup_$STAMP/" 2>/dev/null
say "修复完成，备份 /mnt/data/siping/runs/nogmc_weekend_backup_$STAMP"
