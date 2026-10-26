#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# take_over_full_scale.sh -- 18:33 emergency takeover.
#
# What went wrong:
#   * 20_finalize_and_launch.py sampled "is 15 still running?" during the ~4 s
#     window in which 15 was being restarted, saw zero, found a STALE
#     best_model.json from the previous sweep and therefore finalised the
#     WRONG model: it picked p0_off_A_none, which is the fp16 REFERENCE arm,
#     not an int8 candidate.  The full-scale run that started at 17:53:53 is
#     therefore running the wrong model AND it is failing every chunk anyway,
#     because the two selection axes hold both RKNN contexts.
#   * p0_on_A_normal HANGS the board: the demo never returns (26 min without
#     a result).  That is the real reason for its ZERO_DETECTIONS -- not a
#     clipped output range (its out[0] scale 0.00380537 => ceiling 0.9704).
#
# What this script does:
#   1. stops the stuck selection sweep and both stuck board drivers;
#   2. stops the mis-selected full-scale run;
#   3. starts the full-scale run again with the int8 model that actually has
#      board evidence: p0_off_A_normal (T3 2000-frame F1 0.6448 / FP 527,
#      golden pool F1 0.9420, out ceiling 0.9661 > 0.3 => alive model);
#   4. uses a FRESH output directory, because 14 refuses to reuse a chunk
#      whose model md5 differs.
# ---------------------------------------------------------------------------
set -uo pipefail
W=/mnt/data/siping/ptq_work
REPO=/tmp/pycharm_project_10ae9e2e
MODEL=$W/screen_n/p0_off_A_normal.rknn
OUT=$W/full_scale_int8

echo "=== take over $(date '+%H:%M:%S') ==="

# 1. stop the stuck sweep and its drivers
for p in $(ps -eo pid,args | awk "/night[_]selection[.]sh/ {print \$1}"); do
  echo "kill sweep $p"; kill $p 2>/dev/null
done
for p in $(ps -eo pid,args | awk "/0[4]_run_board_fast_eval[.]py/ {print \$1}"); do
  echo "kill board driver $p"; kill $p 2>/dev/null
done
sleep 3

# 2. stop the mis-selected full-scale run (fp16) and its lock
for p in $(ps -eo pid,args | awk "/1[4]_run_full_scale_nofs[.]py/ {print \$1}"); do
  echo "kill wrong full-scale $p"; kill $p 2>/dev/null
done
sleep 3
# the lock file of the aborted run has to go, otherwise the new run refuses
# to start ("another instance already holds .../.lock")
find $W/full_scale -maxdepth 1 -name ".lock" -delete 2>/dev/null
echo "old lock removed"

# 3. make sure the board is free before the new run grabs a context
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  ctx=$(ssh -o ConnectTimeout=8 ptq-board 'ps -eo comm | grep -c "^rknn_heatmap"' < /dev/null 2>/dev/null | tr -d ' \n')
  ctx=${ctx:-0}
  echo "board ctx = $ctx"
  [ "$ctx" -eq 0 ] && break
  sleep 10
done
# the p0_on demo may still be running on the board; kill it explicitly
ssh -o ConnectTimeout=8 ptq-board 'pkill -f rknn_heatmap_smoke 2>/dev/null; sleep 1; ps -eo comm | grep -c "^rknn_heatmap"' < /dev/null 2>/dev/null
# clean the per-run scratch dirs the killed drivers left behind
ssh -o ConnectTimeout=8 ptq-board 'find /root -maxdepth 1 -name "ptq*" -type d -exec find {} -mindepth 1 -delete \; 2>/dev/null; find /root -maxdepth 1 -name "ptq*" -type d -empty -delete 2>/dev/null; true' < /dev/null 2>/dev/null
echo "board cleaned: ctx=$(ssh -o ConnectTimeout=8 ptq-board 'ps -eo comm | grep -c "^rknn_heatmap"' < /dev/null 2>/dev/null)"

# 4. start the full-scale run with the int8 model, detached from any session
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd "$REPO"
mkdir -p "$W/logs" "$OUT"
setsid nohup python -u manu/pipeline/npu/ptq/14_run_full_scale_nofs.py \
  --tracks blind golden anti \
  --model "$MODEL" \
  --frames-root "$W/bmp" \
  --frames-root "anti=$W/pcfeat_bmp/images" \
  --stage-dir /userdata/ptq_stage \
  --chunk 500 --th 0.30 --topk 100 \
  --out "$OUT" \
  --pt-device cuda:1 \
  < /dev/null > "$W/logs/full_scale_int8.log" 2>&1 &
PID=$!
echo "$PID" > "$W/logs/full_scale_int8.pid"
echo "full-scale INT8 started pid=$PID model=$MODEL out=$OUT at $(date '+%H:%M:%S')"
sleep 20
echo "--- first lines of the log ---"
tail -12 "$W/logs/full_scale_int8.log"
