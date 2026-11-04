#!/bin/bash
# Render one OSD comparison sequence: stale (old logic) / replay (WTH15) / original (ceiling).
#
#   render_osd_one.sh <SEQ> [window_frames] [criterion]
#
# criterion is "base:cand" and selects which arm pair the window maximises the separation of:
#   stale:replay       the old logic loses the target (high-speed manoeuvre sequences)
#   replay:original    the external replay arm drifts onto the wrong thing (dark hover)
#
# Steps: pick the window where the stale arm is worst while the replay arm holds, then render the
# three arms over that window into runs/lookahead/osd/videos/<SEQ>_<arm>.mp4 (h264) with the
# renderer's per-frame match JSON next to each video.
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
FRAMES="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
FINAL=runs/lookahead/final
OUT=runs/lookahead/osd
SEQ=${1:?usage: render_osd_one.sh <SEQ> [window] [criterion]}
WINDOW=${2:-300}
CRITERION=${3:-stale:replay}
mkdir -p "$OUT/videos" "$OUT/windows" "$OUT/logs"

# The stale arm has no 24-sequence product, so it is regenerated per sequence into
# $OUT/preds_stale_<SEQ>/; replay and original ship as 24-sequence products.
STALE="$OUT/preds_stale_${SEQ}"
if [ ! -f "$STALE/${SEQ}.json" ]; then
  echo "[STALE] generating stale arm for $SEQ"
  /mnt/data/codex-overnight/run_stale_arm.sh "$SEQ" || exit 1
fi

$PY /mnt/data/codex-overnight/pick_osd_windows.py \
  --labels-root "$GT" --seq "$SEQ" --window "$WINDOW" --criterion "$CRITERION" \
  --out-json "$OUT/windows/$SEQ.json" > "$OUT/windows/$SEQ.pick.log" 2>&1 \
  || { echo "[FAIL] pick $SEQ"; tail -20 "$OUT/windows/$SEQ.pick.log"; exit 1; }

read -r START LIMIT END < <($PY -c "
import json
d = json.load(open('$OUT/windows/$SEQ.json'))
print(d['start'], d['limit'], d['end'])
")
echo "[render] $SEQ start=$START limit=$LIMIT end=$END criterion=$CRITERION"

for pair in \
  "stale:$STALE" \
  "replay:$FINAL/preds_final_WTH15_24seq" \
  "original:$FINAL/ref_preds_original_v3_24seq"; do
  arm=${pair%%:*}; preds=${pair#*:}
  $PY manu/videos/render_sys_sota_longquan.py --preds-dir "$preds" --seq "$SEQ" \
    --output "$OUT/videos/${SEQ}_${arm}.mp4" --labels-root "$GT" --frames-root "$FRAMES" \
    --start "$START" --limit "$LIMIT" --codec ffmpeg:libx264 \
    --out-json "$OUT/videos/${SEQ}_${arm}.json" > "$OUT/videos/${SEQ}_${arm}.log" 2>&1 \
    && echo "[OK] $SEQ $arm" || { echo "[FAIL] render $SEQ $arm"; tail -10 "$OUT/videos/${SEQ}_${arm}.log"; }
done

$PY -c "
import json
d = json.load(open('$OUT/windows/$SEQ.json'))
print('window', d['start'], '-', d['end'], 'criterion', d['criterion'], 'separation', round(d['separation_score'], 2))
for k, v in d['per_arm'].items():
    print(f\"  {k:9s} cle_mean={v['cle_mean']:7.2f} p50={v['cle_p50']:6.2f} p90={v['cle_p90']:6.2f} max={v['cle_max']:7.2f} hit5={v['hit_5px']*100:5.1f}%\")
"
echo "OSD_DONE $SEQ"
