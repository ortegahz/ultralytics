#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Wait for the batch to finish, then re-detect the modality of every analysed clip.
#
# The batch workers were started with the first version of detect_modality(), which used the
# MEAN HSV saturation and therefore mislabelled colour clips filmed over grey asphalt
# (VIDEO00020_19700101_011424VIS was read as infrared).  The metrics themselves are unaffected -
# only the grouping is - so a re-classification pass over the JSON sidecars is enough.
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/tmp/pycharm_project_10ae9e2e}"
# 产物落点（铁律二十五）：中间产物一律出仓库目录；与 vq_launch_server.sh 的 OUT 保持同一根。
PRODUCT_ROOT="${PRODUCT_ROOT:-/mnt/data/siping/runs/video_quality}"
OUT="${1:-${PRODUCT_ROOT}/out}"
LOG="${OUT}/post_batch.log"

cd "${PROJECT_DIR}"
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

{
  echo "[POST] waiting for the batch to finish ($(date -Is))"
  while true; do
    if grep -q "=== done rc=" "${OUT}/run.log" 2>/dev/null; then
      echo "[POST] batch finished at $(date -Is), re-classifying"
      python manu/video_quality/vq_reclassify.py --metrics-dir "${OUT}" 2>&1
      echo "[POST] done at $(date -Is)"
      break
    fi
    sleep 20
  done
} >> "${LOG}" 2>&1
