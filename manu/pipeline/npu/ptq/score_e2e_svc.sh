#!/usr/bin/env bash
# ===========================================================================
# score_e2e_svc.sh -- score the SERVICE-ARCHITECTURE board runs.
#
# The board produced one directory per arm under
#   /mnt/manu/mosaic_out/svc/<arm>/
# each holding
#   VIDEO00007_19700101_001324_mosaic.tsv   the detections
#   timing.txt                              the per-service timing
#
# This script only MOVES and SCORES; it changes no number.
#
# The arms (see the report for what each one varies):
#   svc_A_idle        the delivery configuration, idle board
#   svc_B_load_other  the same, but other processes use cores 0-3
#   svc_C_load_ours   the same, but other processes take cores 4-7 (ours),
#                     with the wall-clock watchdog armed
#   svc_D_step2       anchor_step=2 (the official GMC semantics) instead of 10
#
# Usage: bash manu/pipeline/npu/ptq/score_e2e_svc.sh [arm ...]
# ===========================================================================
set -euo pipefail

NFS_ROOT=/run/media/manu/SSD960-LOCAL/nfs/mosaic_out/svc
SERVER=huangzhe@192.168.99.40
PORT=32222
REPO=/tmp/pycharm_project_10ae9e2e
SEQ=VIDEO00007_19700101_001324
LOCAL_OUT=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu/results/svc_arms

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(svc_A_idle svc_B_load_other svc_C_load_ours svc_D_step2 corners300 corners250)
fi

echo "== 1. 本地取 NFS 产物 =="
mkdir -p "$LOCAL_OUT"
for arm in "${ARMS[@]}"; do
  src="$NFS_ROOT/$arm"
  if [ ! -d "$src" ]; then
    echo "[WARN] missing $src -- skipped"
    continue
  fi
  mkdir -p "$LOCAL_OUT/$arm"
  cp "$src/${SEQ}_mosaic.tsv" "$src/timing.txt" "$LOCAL_OUT/$arm/"
  echo "--- $arm ---"
  md5sum "$LOCAL_OUT/$arm/${SEQ}_mosaic.tsv" "$LOCAL_OUT/$arm/timing.txt"
done

echo "== 2. 送到服务器 =="
ssh -p $PORT "$SERVER" "mkdir -p '$REPO/manu/pipeline/npu/ptq/board_tsv_svc'"
for arm in "${ARMS[@]}"; do
  [ -d "$LOCAL_OUT/$arm" ] || continue
  ssh -p $PORT "$SERVER" "mkdir -p '$REPO/manu/pipeline/npu/ptq/board_tsv_svc/$arm'"
  scp -P $PORT "$LOCAL_OUT/$arm/${SEQ}_mosaic.tsv" "$LOCAL_OUT/$arm/timing.txt" \
      "$SERVER:$REPO/manu/pipeline/npu/ptq/board_tsv_svc/$arm/"
done

echo "== 3. 出对照表 =="
for arm in "${ARMS[@]}"; do
  [ -d "$LOCAL_OUT/$arm" ] || continue
  echo "--- $arm ---"
  ssh -p $PORT "$SERVER" "source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav && cd $REPO && \
    python manu/pipeline/npu/ptq/38_mosaic_final_report.py \
      --group track3_blind=$SEQ \
      --board-tsv-dir manu/pipeline/npu/ptq/board_tsv \
      --e2e-tsv-dir manu/pipeline/npu/ptq/board_tsv_svc/$arm \
      --pc-area-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_area_all \
      --pc-linear-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_linear_all \
      --e2e-timing manu/pipeline/npu/ptq/board_tsv_svc/$arm/timing.txt \
      --out /mnt/data/siping/ptq_work/final_report_svc/$arm"
done

echo "== 4. 取回报告 =="
mkdir -p "$LOCAL_OUT"
for arm in "${ARMS[@]}"; do
  [ -d "$LOCAL_OUT/$arm" ] || continue
  mkdir -p "$LOCAL_OUT/$arm"
  scp -P $PORT "$SERVER:/mnt/data/siping/ptq_work/final_report_svc/$arm/mosaic_vs_f32_report.md" \
      "$LOCAL_OUT/$arm/" 2>/dev/null || true
  scp -P $PORT "$SERVER:/mnt/data/siping/ptq_work/final_report_svc/$arm/mosaic_vs_f32_report.json" \
      "$LOCAL_OUT/$arm/" 2>/dev/null || true
done
echo "[ok] $LOCAL_OUT/"
