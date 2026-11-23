#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Batch launcher for the uniform-grid spatial-trackability metric.
# Resource discipline follows manu/memory/rules.md §〇之四:
#   - workers are explicitly bounded and every thread pool is pinned to 1
#   - a resident watchdog samples MemAvailable / load every 60 s and brakes under 20 %
#   - the resource reading before the start is written into the same log
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/tmp/pycharm_project_10ae9e2e}"
WORKERS="${1:-4}"
# 产物落点（铁律二十五）：中间产物一律出仓库目录。
PRODUCT_ROOT="${PRODUCT_ROOT:-/mnt/data/siping/runs/video_quality}"
OUT="${2:-${PRODUCT_ROOT}/grid_out}"
LOG="${3:-${OUT}/run.log}"
META_DIR="${META_DIR:-${PRODUCT_ROOT}/out}"
GOOD_ROOT="${GOOD_ROOT:-/mnt/data/xxxu/朗基天香20260903}"
BAD_ROOT="${BAD_ROOT:-/mnt/data/xxxu/纵横挂飞}"

cd "${PROJECT_DIR}"
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
mkdir -p "${OUT}"

{
  echo "=== start $(date -Is) workers=${WORKERS} ==="
  echo "--- free -m"; free -m | head -2
  echo "--- nproc / uptime"; nproc; uptime
  echo "--- iostat -x 1 2"; iostat -x 1 2 2>/dev/null | tail -6
  echo "--- threads"; ps -eLf | wc -l
} >> "${LOG}"

(
  while true; do
    sleep 60
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    total=$(awk '/MemTotal/{print $2}' /proc/meminfo)
    pct=$((avail * 100 / total))
    echo "[WATCH $(date -Is)] MemAvailable=${avail}kB (${pct}%) load1=$(cut -d' ' -f1 /proc/loadavg) threads=$(ps -eLf | wc -l)" >> "${LOG}"
    if [ "${pct}" -lt 20 ]; then
      echo "[WATCH] BRAKE: MemAvailable < 20 % -> killing the batch" >> "${LOG}"
      pkill -f "vq_grid_run_server.py" || true
      break
    fi
  done
) &
WATCH_PID=$!

python manu/video_quality/vq_grid_run_server.py \
  --root "${GOOD_ROOT}" "${BAD_ROOT}" \
  --meta-dir "${META_DIR}" \
  --out "${OUT}" \
  --workers "${WORKERS}" >> "${LOG}" 2>&1
RC=$?
kill "${WATCH_PID}" 2>/dev/null || true
{
  echo "--- free -m"; free -m | head -2
  echo "=== done rc=${RC} $(date -Is) ==="
} >> "${LOG}"
exit "${RC}"
