#!/bin/bash
# =============================================================================
# pipeline_throughput_sweep.sh   (runs ON the RK3588 board)
#
# THE THROUGHPUT SWEEP FOR THE DECOUPLED PIPELINE.
#
# THE GOAL OF THE ARCHITECTURE
#   Run every stage flat out and get as many PROCESSED frames out per second
#   as the board can produce.  Accuracy is a constraint ("do not corrupt the
#   frames"), not the objective function.
#
# WHAT THE MEASURED BASELINE SAYS (P2_step10_live25, the certified-path run)
#   camera            25.0 fps, 11810 frames pushed
#   GMC               29.79 fits/s of FITTING capacity, but only 24.4 fits/s
#                     EFFECTIVE -- the history-window guard blocked the ingest
#                     1182 times for 85 s, and the camera queue overflowed
#                     274 times because of it
#   NPU               rknn_run 105.75 ms/frame -> 9.45 fps with ONE context.
#                     The library's DEFAULT is two contexts; P2 was run with
#                     --npu-workers 1, so half of the NPU was left unused.
#   POST              backproject_cc 10.93 ms/frame -> not a limit
#   END TO END       8.022 fps  (3793 frames in 472.8 s)
#
# So there are two independent levers and this sweep measures both:
#   (1) the NPU context count / core mask      -- the end-to-end ceiling
#   (2) the history-ring slack (gmc_ring_slack) -- how far the ingest may run
#       ahead of the engines, which is what pins the GMC's effective rate
#       BELOW its fitting capacity and makes the camera queue overflow
#
# THE ARMS
#   Every arm: anchor_step 10, a live 25 fps camera (--pace-ms 40),
#   --gmc-phys-anchor 1, --limit 4000.  The limit is 4000 and not 300: the
#   corner density of this sequence goes from sparse to dense around frame
#   2700, and the GMC fit cost doubles there -- a shorter prefix measures the
#   EASY part of the sequence and would rank the arms wrongly.
#
#   S0  the P2 baseline, reproduced on the new binary (1 NPU ctx)
#   S1  2 NPU contexts                       -> the end-to-end ceiling
#   S2  2 NPU ctx + 2 POST workers
#   S3  2 NPU ctx + deeper history ring (slack 128) -> frees the ingest
#   S4  2 NPU ctx + slack 256
#   S5  2 NPU ctx + slack 128 + 3 GMC engines
#   S6  2 NPU ctx + slack 128 + all 8 cores for the CPU side
#   S7  the best of the above, re-measured, with phys-anchor OFF, to price
#       the anchoring change
#
# THE METRICS (per arm)
#   fps                  frames written / wall seconds  <-- THE OBJECTIVE
#   frames               TSV rows
#   dropped              camera-queue overflow (the frames the GMC never saw)
#   features_dropped     frames the inference skipped (acceptable by design)
#   gmc_fit / rknn_run   the two per-frame costs
#   starvation           per-frame wait of each service
#   gmc_history_gaps     history holes the physical anchor DETECTED
# =============================================================================
set -u

RES=/root/psearch/results
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
E2E=${E2E:-/root/rknn_mosaic_e2e_sweep3}
LIMIT=${LIMIT:-4000}
OUT=/root/psearch/sweep_thr
SUMMARY=$OUT/SUMMARY.txt

mkdir -p "$OUT"
echo "=== decoupled-pipeline THROUGHPUT sweep (objective: frames out per second) ===" | tee "$SUMMARY"
date | tee -a "$SUMMARY"

run_arm () {
  local arm=$1; shift
  local out=$RES/$arm
  mkdir -p "$out"
  echo "" | tee -a "$SUMMARY"
  echo "--- $arm : $* ---" | tee -a "$SUMMARY"
  date | tee -a "$SUMMARY"
  # The RK3588 RKNPU driver occasionally refuses a job with
  #   "RKNN: failed to submit!, op id: ... task number: 10"
  # right after a previous context was released (observed once on this board
  # when an arm started <10 s after the previous one exited).  It is a
  # driver-side state issue, not a pipeline defect, so an arm gets ONE retry
  # after a settle pause -- otherwise a single transient failure silently
  # deletes an arm from the table.
  local attempt rc
  for attempt in 1 2; do
    # settle: give the driver time to release the previous context
    sleep 8
    taskset -c 0-7 "$E2E" \
      --raw-root "$RAW_ROOT" --sequence "$SEQ" \
      --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
      --anchor-step 10 --pace-ms 40 --limit "$LIMIT" \
      --gmc-phys-anchor 1 --input-queue 48 --jitter-depth 3 \
      --cpu-affinity 4-7 "$@" > "$out/run.log" 2>&1
    rc=$?
    echo "[$arm] attempt=$attempt exit=$rc" | tee -a "$SUMMARY"
    [ "$rc" = "0" ] && break
    echo "[$arm] non-zero exit -- the arm is RETRIED after a settle pause" | tee -a "$SUMMARY"
  done
  grep -E "^\[timing\]" "$out/run.log" | tee -a "$SUMMARY"
}

run_arm S0_npu1_baseline      --npu-workers 1 --post-workers 1 --gmc-workers 2
run_arm S1_npu2              --npu-workers 2 --post-workers 1 --gmc-workers 2
run_arm S2_npu2_post2        --npu-workers 2 --post-workers 2 --gmc-workers 2
run_arm S3_npu2_slack128     --npu-workers 2 --post-workers 1 --gmc-workers 2 --gmc-ring-slack 128
run_arm S4_npu2_slack256     --npu-workers 2 --post-workers 1 --gmc-workers 2 --gmc-ring-slack 256
run_arm S5_npu2_slack128_w3  --npu-workers 2 --post-workers 1 --gmc-workers 3 --gmc-ring-slack 128
run_arm S6_npu2_slack128_all8 --npu-workers 2 --post-workers 1 --gmc-workers 2 --gmc-ring-slack 128 --cpu-affinity 0-7

# -----------------------------------------------------------------------------
# THE TABLE
# -----------------------------------------------------------------------------
echo "" | tee -a "$SUMMARY"
echo "=== THROUGHPUT TABLE (the objective is the fps column) ===" | tee -a "$SUMMARY"
printf "%-26s %8s %8s %8s %8s %9s %9s %9s %8s\n" \
  arm frames dropped feat_drop gmc_fit rknn_run fps hist_gaps | tee -a "$SUMMARY"
for arm in S0_npu1_baseline S1_npu2 S2_npu2_post2 S3_npu2_slack128 \
           S4_npu2_slack256 S5_npu2_slack128_w3 S6_npu2_slack128_all8; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || { printf "%-26s %8s\n" "$arm" "(missing)"; continue; }
  get() { grep -E "^$1=" "$t" | head -1 | cut -d= -f2; }
  frames=$(get frames); dropped=$(get dropped); fd=$(get features_dropped)
  fit=$(grep -E "^gmc_fit " "$t" | awk '{print $3}')
  rknn=$(grep -E "^rknn_run " "$t" | awk '{print $3}')
  fps=$(get fps); gaps=$(get gmc_history_gaps)
  printf "%-26s %8s %8s %8s %8s %9s %9s %9s %8s\n" \
    "$arm" "$frames" "$dropped" "$fd" "$fit" "$rknn" "$fps" "$gaps" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== per-arm detail ===" | tee -a "$SUMMARY"
for arm in S0_npu1_baseline S1_npu2 S2_npu2_post2 S3_npu2_slack128 \
           S4_npu2_slack256 S5_npu2_slack128_w3 S6_npu2_slack128_all8; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || continue
  echo "--- $arm ---" | tee -a "$SUMMARY"
  grep -E "^(gmc_total|gmc_post|gmc_ingest|gmc_ingest_window|gmc_ingest_space|gmc_wait|npu_wait|post_wait|rga_import|rga_improcess|upload|outputs_get|backproject_cc|total_wall|gmc_queue_depth_mean|gmc_window_blocks|gmc_ring_slack|gmc_fit_p50|gmc_fit_p95|gmc_fit_max|rknn_run_p50|rknn_run_max)=" "$t" | sed 's/^/    /' | tee -a "$SUMMARY"
  grep -E "^\[timing\] GMC fit distribution" "$RES/$arm/run.log" | sed 's/^/    /' | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== sweep done ===" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
