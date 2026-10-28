#!/bin/bash
# =============================================================================
# pipeline_throughput_only.sh   (runs ON the RK3588 board)
#
# THE THROUGHPUT SWEEP -- THE OBJECTIVE OF THE WHOLE ARCHITECTURE.
#
# "Run every stage flat out and get as many PROCESSED frames out per second
#  as the board can produce."
#
# WHAT THE MEASURED BASELINE SAYS
#   P2 (step10, 2 GMC engines, live 25 fps camera, npu_workers=1):
#     GMC fit capacity 29.79 fits/s, EFFECTIVE 24.4 fits/s (the history-window
#       guard blocked the ingest 1182 times for 85 s -- THAT is what made the
#       camera queue overflow 274 times)
#     rknn_run 105.75 ms/frame  -> 9.45 fps with ONE context; the library's
#       DEFAULT is two contexts, and P2 ran with one, so half the NPU was idle
#     END TO END 8.022 fps
#   memory (already certified, not re-tried here):
#     the RKNN context limit is 2 => the NPU ceiling is 15.1 fps
#       (132.6 ms per worker per frame, two workers)
#     cpu affinity 4-7 (4x A76) beats 0-7 (which drags in A55): 12.833 vs
#       12.405 fps -- so 4-7 is the CPU configuration
#     GMC engines 3/4 are SLOWER (fit +47.6%/+23.5%): the one-step fit cache
#       collapses across engines.  gmc_workers stays at 2.
#
# THE ARMS (step10, live 25 fps camera, 4000 frames)
#   S0  npu_workers=1                 the P2 baseline, reproduced
#   S1  npu_workers=2                 the NPU ceiling
#   S2  npu_workers=2 + ring_slack 128  does a deeper history ring release the
#                                      ingest from the window guard and let the
#                                      GMC's effective rate reach its capacity?
#   S3  S2 + npu_core_mask 7 (all three NPU cores)
#
# The 4000-frame limit is deliberate: this sequence's corner density goes from
# sparse to dense around frame 2700 and the GMC fit cost doubles there, so a
# short prefix measures the easy part and ranks the arms wrongly.
# =============================================================================
set -u

RES=/root/psearch/results
PUB=/mnt/srv2/psearch/results
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
E2E=${E2E:-/root/rknn_mosaic_e2e_sweep3}
LIMIT=${LIMIT:-4000}
SUMMARY=/root/psearch/thr_run.log

echo "=== throughput sweep (objective: processed frames per second) ===" | tee "$SUMMARY"
date | tee -a "$SUMMARY"

run_arm () {
  local arm=$1; shift
  local out=$RES/$arm
  mkdir -p "$out"
  echo "" | tee -a "$SUMMARY"
  echo "--- $arm : $* ---" | tee -a "$SUMMARY"
  date | tee -a "$SUMMARY"
  local attempt rc
  for attempt in 1 2; do
    sleep 10
    taskset -c 4-7 "$E2E" \
      --raw-root "$RAW_ROOT" --sequence "$SEQ" \
      --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
      --limit "$LIMIT" --anchor-step 10 --pace-ms 40 \
      --input-queue 48 --jitter-depth 3 --gmc-workers 2 \
      --gmc-phys-anchor 1 "$@" > "$out/run.log" 2>&1
    rc=$?
    echo "[$arm] attempt=$attempt exit=$rc" | tee -a "$SUMMARY"
    [ "$rc" = "0" ] && break
    grep -E "Illegal|FATAL|failed to submit" "$out/run.log" | head -3 | sed 's/^/    /' | tee -a "$SUMMARY"
  done
  grep -E "^\[timing\]|fallback summary|camera ran" "$out/run.log" | tail -5 | tee -a "$SUMMARY"
  mkdir -p "$PUB/$arm"
  cp -f "$out/${SEQ}_mosaic.tsv" "$out/feature_md5.txt" "$out/timing.txt" "$PUB/$arm/" 2>/dev/null
}

run_arm S0_npu1              --npu-workers 1 --post-workers 1 --cpu-affinity 4-7
run_arm S1_npu2              --npu-workers 2 --post-workers 1 --cpu-affinity 4-7
run_arm S2_npu2_slack128     --npu-workers 2 --post-workers 1 --gmc-ring-slack 128 --cpu-affinity 4-7
run_arm S3_npu2_slack128_mask7 --npu-workers 2 --post-workers 1 --gmc-ring-slack 128 --npu-core-mask 7 --cpu-affinity 4-7

echo "" | tee -a "$SUMMARY"
echo "=== the throughput table (the fps column is the objective) ===" | tee -a "$SUMMARY"
printf "%-28s %8s %8s %9s %9s %9s %9s %9s %10s\n" \
  arm frames dropped feat_drop gmc_fit rknn_run wall fps gaps | tee -a "$SUMMARY"
for arm in S0_npu1 S1_npu2 S2_npu2_slack128 S3_npu2_slack128_mask7; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || { printf "%-28s %8s\n" "$arm" "(missing)"; continue; }
  get() { grep -E "^$1=" "$t" | head -1 | cut -d= -f2; }
  fit=$(grep -E "^gmc_fit " "$t" | awk '{print $3}')
  rknn=$(grep -E "^rknn_run " "$t" | awk '{print $3}')
  wall=$(grep -E "^total_wall " "$t" | awk '{print $3}')
  printf "%-28s %8s %8s %9s %9s %9s %9s %9s %10s\n" "$arm" \
    "$(get frames)" "$(get dropped)" "$(get features_dropped)" "$fit" "$rknn" "$wall" \
    "$(get fps)" "$(get gmc_history_gaps)" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== arm detail ===" | tee -a "$SUMMARY"
for arm in S0_npu1 S1_npu2 S2_npu2_slack128 S3_npu2_slack128_mask7; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || continue
  echo "--- $arm ---" | tee -a "$SUMMARY"
  grep -E "^(gmc_total|gmc_post|gmc_ingest|gmc_ingest_window|gmc_wait|npu_wait|post_wait|upload|outputs_get|backproject_cc|gmc_queue_depth_mean|gmc_window_blocks|gmc_ring_slack|gmc_fit_p50|gmc_fit_p95|gmc_fit_max|rknn_run_p50|rknn_run_max|gmc_corner_mean|gmc_corner_capped)=" "$t" | sed 's/^/    /' | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== throughput sweep done ===" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
