#!/bin/bash
# 新交付档的 OSD 三臂对照：stale（老逻辑）/ replay（外挂交付档）/ original（零延迟天花板）
#   bash render_osd_3arm.sh <SEQ> [window_frames]
# ⚠️ 路径按**服务器**写；host 侧请改 PY / GT / FRAMES。
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FINAL=runs/lookahead/final
OUT=runs/lookahead/osd
SEQ=${1:?usage: render_osd_3arm.sh <SEQ> [window]}
WINDOW=${2:-300}
mkdir -p "$OUT/videos" "$OUT/windows" "$OUT/logs"

STALE="$FINAL/preds_final_stale"
REPLAY="$FINAL/final8/preds_delivery"
ORIG=runs/lookahead/ref/preds_original_v3

$PY manu/tracking/lookahead/osd/pick_osd_window_3arm.py \
  --labels-root "$GT" --seq "$SEQ" --window "$WINDOW" \
  --stale "$STALE" --replay "$REPLAY" --original "$ORIG" \
  --out-json "$OUT/windows/${SEQ}_3arm.json" > "$OUT/windows/${SEQ}_3arm.pick.log" 2>&1 \
  || { echo "[FAIL] pick $SEQ"; tail -20 "$OUT/windows/${SEQ}_3arm.pick.log"; exit 1; }

read -r START LIMIT END < <($PY -c "
import json
d = json.load(open('$OUT/windows/${SEQ}_3arm.json'))
print(d['start'], d['limit'], d['end'])
")
echo "[render] $SEQ start=$START limit=$LIMIT end=$END"

for pair in "stale:$STALE" "replay:$REPLAY" "original:$ORIG"; do
  arm=${pair%%:*}; preds=${pair#*:}
  $PY manu/videos/render_sys_sota_longquan.py --preds-dir "$preds" --seq "$SEQ" \
    --output "$OUT/videos/${SEQ}_3arm_${arm}.mp4" --labels-root "$GT" --frames-root "$FRAMES" \
    --start "$START" --limit "$LIMIT" --codec ffmpeg:libx264 \
    --out-json "$OUT/videos/${SEQ}_3arm_${arm}.json" > "$OUT/logs/${SEQ}_3arm_${arm}.log" 2>&1 \
    && echo "[OK] $SEQ $arm" || { echo "[FAIL] render $SEQ $arm"; tail -10 "$OUT/logs/${SEQ}_3arm_${arm}.log"; }
done

$PY -c "
import json
d = json.load(open('$OUT/windows/${SEQ}_3arm.json'))
print('window', d['start'], '-', d['end'], ' separation(stale-replay) =', round(d['separation_score'],2),
      ' replay_gap_to_ceiling =', round(d['replay_gap_to_ceiling'],2))
for k,v in d['per_arm'].items():
    print(f\"  {k:9s} cle_mean={v['cle_mean']:8.2f} p50={v['cle_p50']:7.2f} max={v['cle_max']:8.2f} hit5={v['hit_5px']*100:5.1f}%\")
"
