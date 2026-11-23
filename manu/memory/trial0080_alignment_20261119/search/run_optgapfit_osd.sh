#!/usr/bin/env bash
# trial_0080 + v3 + Δn 重搜出的最优 gapfit 参数：渲染 OSD 复核视频（pass / un-pass 分档）
set -uo pipefail
cd /mnt/data/siping/runs/npu20ms/nano_search/repo || exit 2
export PYTHONPATH=/mnt/data/siping/ptq_work/ortgpu:/mnt/data/siping/runs/npu20ms/nano_search/repo
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

EVAL_DIR=/mnt/data/siping/runs/trial0080_onnx/eval
SEARCH_JSON="$EVAL_DIR/track_all_v3_optgapfit.json"
PREDS_DIR="$EVAL_DIR/preds_v3_optgapfit"
OUT_DIR="/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_optgapfit_20261119"

echo "[wait] 等评测产物 $SEARCH_JSON ..."
for i in $(seq 1 240); do
  [ -f "$SEARCH_JSON" ] && break
  sleep 20
done
if [ ! -f "$SEARCH_JSON" ]; then echo "[FAIL] 评测产物未出现"; exit 3; fi
sleep 10
NP=$(ls -1 "$PREDS_DIR"/*.json 2>/dev/null | grep -v summary.json$ | wc -l)
echo "[info] preds=$NP / 24"
if [ "$NP" -ne 24 ]; then echo "[FAIL] 预测数不足"; exit 3; fi

# 渲染器拒绝写非空目录；这里再确认一次，且绝不碰上一轮用非最优 gapfit 参数渲的半成品
if [ -e "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  echo "[FAIL] 输出目录已存在且非空: $OUT_DIR"; exit 4
fi
echo "[info] 对照目录（只读，不动）: /mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_gapfit_20261119"

PREDS_DIR="$PREDS_DIR" SEARCH_JSON="$SEARCH_JSON" OUT_DIR="$OUT_DIR" OUT_CUT=88.0 JOBS=4 CODEC=mp4v \
  bash manu/videos/render_trial0080_longquan_osd.sh
echo "[done] $(date +%H:%M:%S)"
