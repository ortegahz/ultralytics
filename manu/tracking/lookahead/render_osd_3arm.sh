#!/bin/bash
# 交付档三臂 OSD 对照：stale(老逻辑，真吃 320 ms 滞后) / replay(外挂) / original(零延迟天花板)。
#
# **窗口沿用**已有的 windows/<SEQ>_3arm.json（当初按「stale vs replay 分离度最大」挑的），
# 只把 replay 臂换成当前交付档产物 ⇒ 与历史视频逐帧可比。
#
#   bash manu/tracking/lookahead/render_osd_3arm.sh [8|24]
set -u
cd "$(dirname "$0")/../../.."        # → 项目根
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
FRAMES="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
F=runs/lookahead/final
OUT=runs/lookahead/osd
TAG=${1:-8}
case "$TAG" in
  8)  DELIV="$F/final8/preds_delivery"; STALE="$F/preds_final_stale"; OFF="$F/preds_off" ;;
  24) DELIV="$F/final24/preds_delivery"; STALE="$F/preds_n1_p0_24seq_stale"; OFF="$F/preds_off_24seq" ;;
  *) echo "usage: $0 [8|24]"; exit 2 ;;
esac
mkdir -p "$OUT/videos" "$OUT/windows"
for w in "$OUT/windows"/*_3arm.json; do
  [ -e "$w" ] || continue
  SEQ=$(basename "$w" _3arm.json)
  START=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['start'])" "$w")
  LIMIT=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['limit'])" "$w")
  echo "[render] $SEQ start=$START limit=$LIMIT"
  for pair in "stale:$STALE" "replay:$DELIV" "original:$OFF"; do
    arm=${pair%%:*}; preds=${pair#*:}
    $PY manu/videos/render_sys_sota_longquan.py --preds-dir "$preds" --seq "$SEQ" \
      --output "$OUT/videos/${SEQ}_3arm_${arm}.mp4" --labels-root "$GT" --frames-root "$FRAMES" \
      --start "$START" --limit "$LIMIT" --codec ffmpeg:h264 \
      --out-json "$OUT/videos/${SEQ}_3arm_${arm}.json" > "$OUT/videos/${SEQ}_3arm_${arm}.log" 2>&1 \
      && echo "[OK] $SEQ $arm" || { echo "[FAIL] render $SEQ $arm"; tail -5 "$OUT/videos/${SEQ}_3arm_${arm}.log"; }
  done
done
echo "OSD_3ARM_DONE"
