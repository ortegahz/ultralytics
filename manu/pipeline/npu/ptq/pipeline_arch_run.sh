#!/bin/bash
# =============================================================================
# pipeline_arch_run.sh   (runs ON the RK3588 board)
#
# THE CONSOLIDATED ARCHITECTURE RUN.
#
# WHY ONE SCRIPT AND NOT MANY
# ---------------------------
# This board's RKNPU and RGA drivers leak across process exits: every e2e
# invocation prints "rga: buffer[0] mm release buffer failed!" on teardown,
# and after enough invocations the NPU job times out (6 s), the driver does a
# "soft reset", and the NEXT rknn_init dies with SIGILL -- which took the
# whole board down.  So the number of e2e invocations is a scarce resource.
# Everything that has to run on the board therefore goes into ONE script,
# in one order, with a settle pause and a health check in front of each arm.
#
# THE THREE QUESTIONS THIS RUN ANSWERS
# -------------------------------------
# Q1  Can step2 reach the 25 fps regime?  The anchor LK stage is written to
#     run concurrently (phase 2) with the RANSAC still serial in the anchor
#     order (phase 3), so the numbers stay BIT-IDENTICAL.  step2 has 21
#     anchors per frame and step10 only 5 -- that is the whole 204 vs 60 ms
#     difference.  --gmc-anchor-threads (clamp 1..8) + --gmc-affinity-group
#     spread those 21 fits over the board's cores.  If step2 can reach 25 fps,
#     the LIVE step2 arm can run with ZERO camera drops, and then "the frames
#     step2 processed must match the reference frame by frame" is a
#     meaningful statement.
#     ARMS: K0 serial (the reference path) / K1 4 threads on cores 4-7 /
#            K2 8 threads on cores 0-7 / K3 8 threads on 0-7 with the NPU on
#            6,7.  EVERY arm is compared to K0 frame by frame: if the anchor
#            parallelism is not bit-identical it is not a free speed knob.
#
# Q2  What is the maximum number of processed frames per second?  The NPU is
#     the end-to-end ceiling (memory: the RKNN context limit is 2, 132.6 ms
#     per worker per frame => 15.1 fps theoretical).  The sweep measures:
#     S0 one NPU context (the P2 baseline) / S1 two contexts /
#     S2 two contexts + a deeper history ring (the ring slack is what
#     decides whether the GMC's effective rate reaches its 29.79 fits/s
#     capacity, or stays pinned at 24.4 by the window guard) /
#     S3 two contexts + NPU core mask 7 (all three cores).
#     ARMS THAT MEMORY ALREADY REFUTED ARE NOT RE-RUN: gmc_workers 3/4 (fit
#     +47.6%/+23.5% -- the one-step fit cache collapses across engines) and
#     cpu_affinity 0-7 (12.405 vs 12.833 fps on cores 4-7).
#
# Q3  Does the physical-frame-number anchor change anything when nothing is
#     dropped?  It must not: K0 (anchor off) and the anchor-on twin are
#     compared frame by frame.
#
# ALL arms are step2 or step10 with --gmc-phys-anchor 1 except where the
# question is the anchor itself.
# =============================================================================
set -u

RES=/root/psearch/results
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
E2E=${E2E:-/root/rknn_mosaic_e2e_sweep3}
OUT=/root/psearch/arch
SUMMARY=$OUT/SUMMARY.txt

mkdir -p "$OUT"
echo "=== consolidated architecture run (throughput-first) ===" | tee "$SUMMARY"
date | tee -a "$SUMMARY"

health () {
  # The NPU health check: a 3-frame run.  If rknn_init dies with SIGILL the
  # driver is in the bad state and every later arm would be lost, so the
  # check is loud instead of silent.
  local d=$OUT/health_$1
  mkdir -p "$d"
  timeout 180 "$E2E" --raw-root "$RAW_ROOT" --sequence "$SEQ" --model "$MODEL" \
    --out "$d" --kernel-dir "$KERNELS" --cpu-affinity 4-7 --npu-workers 1 \
    --post-workers 1 --anchor-step 10 --pace-ms 0 --limit 3 > "$d/log.txt" 2>&1
  local rc=$?
  echo "[health:$1] exit=$rc" | tee -a "$SUMMARY"
  if [ "$rc" != "0" ]; then
    echo "[health:$1] NPU HEALTH CHECK FAILED -- the driver is in the bad state;" | tee -a "$SUMMARY"
    echo "[health:$1] tail:" | tee -a "$SUMMARY"
    tail -4 "$d/log.txt" | sed 's/^/    /' | tee -a "$SUMMARY"
    return 1
  fi
  return 0
}

run_arm () {
  local arm=$1; local limit=$2; shift 2
  local out=$RES/$arm
  mkdir -p "$out"
  echo "" | tee -a "$SUMMARY"
  echo "--- $arm : $* ---" | tee -a "$SUMMARY"
  date | tee -a "$SUMMARY"
  local attempt rc
  for attempt in 1 2; do
    sleep 10          # let the driver release the previous context
    taskset -c 4-7 "$E2E" \
      --raw-root "$RAW_ROOT" --sequence "$SEQ" \
      --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
      --limit "$limit" "$@" > "$out/run.log" 2>&1
    rc=$?
    echo "[$arm] attempt=$attempt exit=$rc" | tee -a "$SUMMARY"
    [ "$rc" = "0" ] && break
    grep -E "Illegal|FATAL|failed to submit|timeout" "$out/run.log" | head -3 | sed 's/^/    /' | tee -a "$SUMMARY"
  done
  grep -E "^\[timing\]|anchor LK stage on" "$out/run.log" | tail -6 | tee -a "$SUMMARY"
}

###############################################################################
# Q1 -- the anchor-LK parallelism probe (step2, zero-drop, 600 frames)
###############################################################################
health q1 || { echo "ABORT: the NPU is unhealthy before Q1" | tee -a "$SUMMARY"; exit 1; }

run_arm K0_serial      600 --anchor-step 2 --pace-ms 0 --jitter-depth 8 --input-queue 8 \
      --npu-workers 1 --post-workers 1 --gmc-workers 1 --gmc-anchor-threads 1 \
      --cpu-affinity 4-7 --gmc-phys-anchor 0
run_arm K1_thr4        600 --anchor-step 2 --pace-ms 0 --jitter-depth 8 --input-queue 8 \
      --npu-workers 1 --post-workers 1 --gmc-workers 1 --gmc-anchor-threads 4 \
      --gmc-affinity-group 4-7 --cpu-affinity 4-7 --gmc-phys-anchor 0
run_arm K2_thr8        600 --anchor-step 2 --pace-ms 0 --jitter-depth 8 --input-queue 8 \
      --npu-workers 1 --post-workers 1 --gmc-workers 1 --gmc-anchor-threads 8 \
      --gmc-affinity-group 0-7 --cpu-affinity 4-7 --gmc-phys-anchor 0
run_arm K3_thr8_anchor 600 --anchor-step 2 --pace-ms 0 --jitter-depth 8 --input-queue 8 \
      --npu-workers 1 --post-workers 1 --gmc-workers 1 --gmc-anchor-threads 8 \
      --gmc-affinity-group 0-7 --cpu-affinity 4-7 --gmc-phys-anchor 1

###############################################################################
# Q2 -- the throughput sweep (step10, live 25 fps camera, 4000 frames)
###############################################################################
health q2 || { echo "WARN: the NPU is unhealthy before Q2 -- trying anyway" | tee -a "$SUMMARY"; }

run_arm S0_npu1        4000 --anchor-step 10 --pace-ms 40 --input-queue 48 --jitter-depth 3 \
      --npu-workers 1 --post-workers 1 --gmc-workers 2 --gmc-phys-anchor 1 --cpu-affinity 4-7
run_arm S1_npu2        4000 --anchor-step 10 --pace-ms 40 --input-queue 48 --jitter-depth 3 \
      --npu-workers 2 --post-workers 1 --gmc-workers 2 --gmc-phys-anchor 1 --cpu-affinity 4-7
run_arm S2_npu2_slack128 4000 --anchor-step 10 --pace-ms 40 --input-queue 48 --jitter-depth 3 \
      --npu-workers 2 --post-workers 1 --gmc-workers 2 --gmc-ring-slack 128 --gmc-phys-anchor 1 --cpu-affinity 4-7
run_arm S3_npu2_slack128_mask7 4000 --anchor-step 10 --pace-ms 40 --input-queue 48 --jitter-depth 3 \
      --npu-workers 2 --post-workers 1 --gmc-workers 2 --gmc-ring-slack 128 \
      --npu-core-mask 7 --gmc-phys-anchor 1 --cpu-affinity 4-7

###############################################################################
# THE TABLES
###############################################################################
echo "" | tee -a "$SUMMARY"
echo "=== Q1: the anchor-LK parallelism (step2) ===" | tee -a "$SUMMARY"
printf "%-20s %10s %10s %10s %10s %12s\n" arm gmc_fit total_wall fps diff_vs_K0 first_diff | tee -a "$SUMMARY"
for arm in K0_serial K1_thr4 K2_thr8 K3_thr8_anchor; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || { printf "%-20s %10s\n" "$arm" "(missing)"; continue; }
  fit=$(grep -E "^gmc_fit " "$t" | awk '{print $3}')
  wall=$(grep -E "^total_wall " "$t" | awk '{print $3}')
  fps=$(grep -E "^fps=" "$t" | cut -d= -f2)
  res=$(python3 - "$RES/K0_serial/feature_md5.txt" "$RES/$arm/feature_md5.txt" <<'PY'
import sys, os
def load(p):
    d = {}
    if not os.path.exists(p): return None
    for line in open(p):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            try: d[int(parts[1])] = parts[2]
            except ValueError: pass
    return d
a = load(sys.argv[1]); b = load(sys.argv[2])
if a is None or b is None:
    print("- -"); raise SystemExit
common = sorted(set(a) & set(b))
diff = [k for k in common if a[k] != b[k]]
print(f"{len(diff)} {diff[0] if diff else '-'}")
PY
)
  printf "%-20s %10s %10s %10s %10s %12s\n" "$arm" "$fit" "$wall" "$fps" \
    "$(echo "$res" | awk '{print $1}')" "$(echo "$res" | awk '{print $2}')" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== Q2: the throughput sweep (step10, live 25 fps) ===" | tee -a "$SUMMARY"
printf "%-26s %8s %8s %8s %9s %9s %9s %9s %10s\n" \
  arm frames dropped feat_drop gmc_fit rknn_run wall fps gaps | tee -a "$SUMMARY"
for arm in S0_npu1 S1_npu2 S2_npu2_slack128 S3_npu2_slack128_mask7; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || { printf "%-26s %8s\n" "$arm" "(missing)"; continue; }
  get() { grep -E "^$1=" "$t" | head -1 | cut -d= -f2; }
  fit=$(grep -E "^gmc_fit " "$t" | awk '{print $3}')
  rknn=$(grep -E "^rknn_run " "$t" | awk '{print $3}')
  wall=$(grep -E "^total_wall " "$t" | awk '{print $3}')
  printf "%-26s %8s %8s %8s %9s %9s %9s %9s %10s\n" "$arm" \
    "$(get frames)" "$(get dropped)" "$(get features_dropped)" "$fit" "$rknn" "$wall" \
    "$(get fps)" "$(get gmc_history_gaps)" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== arm detail ===" | tee -a "$SUMMARY"
for arm in K0_serial K1_thr4 K2_thr8 K3_thr8_anchor S0_npu1 S1_npu2 S2_npu2_slack128 S3_npu2_slack128_mask7; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || continue
  echo "--- $arm ---" | tee -a "$SUMMARY"
  grep -E "^(gmc_total|gmc_post|gmc_ingest|gmc_ingest_window|gmc_wait|npu_wait|post_wait|upload|outputs_get|backproject_cc|gmc_queue_depth_mean|gmc_window_blocks|gmc_ring_slack|gmc_fit_p50|gmc_fit_p95|gmc_fit_max|rknn_run_p50|rknn_run_max|gmc_corner_mean|gmc_corner_capped)=" "$t" | sed 's/^/    /' | tee -a "$SUMMARY"
  grep -E "^\[timing\] GMC fit distribution" "$RES/$arm/run.log" | sed 's/^/    /' | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== run done ===" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
