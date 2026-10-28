#!/bin/bash
# =============================================================================
# pipeline_search_driver.sh   (runs ON the RK3588 board)
#
# Searches the DECOUPLED-PIPELINE configuration space and reports, per arm:
#   speed   : gmc_fit per frame, the GMC throughput ceiling, the ingest wait
#             breakdown, and the end-to-end fps
#   drops   : dropped (camera-side, MUST be 0) and features_dropped (allowed)
#
# It runs DETACHED (setsid nohup) and its data source is the SERVER's
# /mnt/srv2, so it survives the host going down.
#
# ARMS
# ----
# The question is whether a DEEP camera-side buffer alone rescues 25 fps, or
# whether GMC's steady-state throughput is the real wall.  The arms are chosen
# to separate those two regimes:
#
#   A1  q8   w1   the CURRENT shipped configuration (the baseline to beat)
#   A2  q48  w1   deep Raw buffer, ONE GMC engine       <- the proposal alone
#   A3  q48  w2   deep Raw buffer, TWO GMC engines      <- proposal + throughput
#   A4  q96  w2   very deep Raw buffer, TWO engines     <- headroom check
#   A5  q48  w2  iters10  deep buffer, 2 engines, LK iters=10 (the cheap fit)
#
# Every arm uses --pace-ms 40 (a real 25 fps camera) and the FULL 11810-frame
# sequence.  A short prefix would be misleading: the first few hundred frames
# are cheap (few corners) and the sequence gets ~2x more expensive later.
#
# The fit knobs are ALSO swept, because a deep buffer absorbs spikes but not a
# sustained rate deficit -- and if GMC's mean fit cost is above the 40 ms
# budget, no buffer depth will ever hold.
# =============================================================================
set -u

REMOTE=/root/psearch
RES=$REMOTE/results
# The data source is the SERVER's copy, reached over the board's SSHFS mount.
# It does NOT live on the host's NFS export (/mnt/manu), which dies with the
# host -- this driver must keep working after the host shuts down.
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
# A LOCAL copy of the model, for the same reason: /mnt/manu/ptq is on the host.
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
# The GPU kernels, also copied locally.
KERNELS=${KERNELS:-/root/psearch/kernels}
E2E=${E2E:-/root/rknn_mosaic_e2e_sweep}   # the sweep build (has the fit knobs)
LIMIT=${LIMIT:-0}                          # 0 = the whole sequence

mkdir -p "$RES"
SUMMARY=$RES/SUMMARY.txt
echo "=== pipeline search driver ===" | tee "$SUMMARY"
date | tee -a "$SUMMARY"

# The smoke-test directory is not an arm; keep it out of the results.
rm -rf "$RES/_smoke" 2>/dev/null

# The library's own numbers, in one line, from the arm's timing.txt.
emit () {
  local arm=$1
  local t=$RES/$arm/timing.txt
  [ -f "$t" ] || { echo "$arm | NO_TIMING" | tee -a "$SUMMARY"; return; }
  local gfit gmc_frames dropped feat_drop fps corners capped p50 p95 p99
  gfit=$(grep -E "^gmc_fit " "$t" | awk '{print $3}')
  gmc_frames=$(grep -E "^gmc_frames=" "$t" | cut -d= -f2)
  dropped=$(grep -E "^dropped=" "$t" | cut -d= -f2)
  feat_drop=$(grep -E "^features_dropped=" "$t" | cut -d= -f2)
  fps=$(grep -E "^fps=" "$t" | cut -d= -f2)
  corners=$(grep -E "^gmc_corner_mean=" "$t" | cut -d= -f2)
  capped=$(grep -E "^gmc_corner_capped=" "$t" | cut -d= -f2)
  p50=$(grep -E "^gmc_fit_p50=" "$t" | cut -d= -f2)
  p95=$(grep -E "^gmc_fit_p95=" "$t" | cut -d= -f2)
  p99=$(grep -E "^gmc_fit_p99=" "$t" | cut -d= -f2)
  local ing in_w space qdepth wblocks sblocks total
  ing=$(grep -E "^gmc_ingest " "$t" | awk '{print $3}')
  in_w=$(grep -E "^gmc_ingest_in " "$t" | awk '{print $3}')
  space=$(grep -E "^gmc_ingest_space " "$t" | awk '{print $3}')
  qdepth=$(grep -E "^gmc_queue_depth_mean=" "$t" | cut -d= -f2)
  wblocks=$(grep -E "^gmc_window_blocks=" "$t" | cut -d= -f2)
  sblocks=$(grep -E "^gmc_space_blocks=" "$t" | cut -d= -f2)
  total=$(grep -E "^total_wall " "$t" | awk '{print $3}')
  printf "%-18s | fit=%-8s | gmc_frames=%-6s | dropped=%-6s | feat_drop=%-6s | fps=%-7s | p50=%-7s p95=%-7s p99=%-7s | corners=%-6s capped=%-6s | ingest=%-8s in=%-8s space=%-8s | qdepth=%-7s wblk=%-6s sblk=%-6s | wall=%s\n" \
    "$arm" "$gfit" "$gmc_frames" "$dropped" "$feat_drop" "$fps" "$p50" "$p95" "$p99" \
    "$corners" "$capped" "$ing" "$in_w" "$space" "$qdepth" "$wblocks" "$sblocks" "$total" | tee -a "$SUMMARY"
}

run_arm () {
  local arm=$1; shift
  local extra=("$@")
  echo "--- arm $arm : ${extra[*]} ---" | tee -a "$SUMMARY"
  date | tee -a "$SUMMARY"

  # Skip an arm that already completed (re-runnable without losing work).
  if [ -f "$RES/$arm/timing.txt" ]; then
    echo "[skip] $arm already has a timing.txt" | tee -a "$SUMMARY"
    emit "$arm"
    return 0
  fi

  local out=$RES/$arm
  mkdir -p "$out"
  # --cpu-affinity 4-7 keeps the CPU stage off the cores the NPU needs.
  taskset -c 4-7 "$E2E" \
    --raw-root "$RAW_ROOT" --sequence "$SEQ" \
    --model "$MODEL" \
    --out "$out" --kernel-dir "$KERNELS" \
    --pace-ms 40 --cpu-affinity 4-7 \
    --npu-workers 1 --post-workers 1 \
    "${extra[@]}" > "$out/run.log" 2>&1
  local rc=$?
  echo "[arm $arm] exit=$rc" | tee -a "$SUMMARY"
  grep -E "^\[timing\]|camera ran|fallback" "$out/run.log" | tee -a "$SUMMARY"
  emit "$arm"

  # Publish to the SERVER-visible path so the result is on the server even if
  # this host goes down.
  mkdir -p "/mnt/srv2/psearch/results/$arm" 2>/dev/null
  cp -f "$out/timing.txt" "/mnt/srv2/psearch/results/$arm/" 2>/dev/null
  cp -f "$out/run.log"   "/mnt/srv2/psearch/results/$arm/" 2>/dev/null
  cp -f "$out"/*_mosaic.tsv "/mnt/srv2/psearch/results/$arm/" 2>/dev/null
}

# ---------------------------------------------------------------------------
# THE SEARCH
# ---------------------------------------------------------------------------
# ORDER MATTERS: the host may shut down at any moment, so the arms are ordered
# by how much each one ANSWERS, not by how cheap it is.  A2 first: it is the
# single most informative arm -- does a deep Raw buffer alone fix the drop?
# A1 (the shipped baseline) second, because every later number is only
# meaningful against it.  Then A3 (the proposal plus a second engine), A5
# (plus the cheap LK), A4 (headroom), A6 (the RANSAC cap).
#
# Every arm is a FULL 11810-frame sequence at --pace-ms 40 (a real 25 fps
# camera).  A short prefix would be misleading: the first few hundred frames
# are cheap (few corners) and the sequence gets ~2x more expensive later, so
# a prefix run says "fast" when the full run says "slow".
# ---------------------------------------------------------------------------
# A2: the proposal ALONE -- deep Raw buffer, ONE engine.  Does the deep buffer
#     absorb the fit-cost variance on its own?
run_arm A2_q48_w1_it20     --input-queue 48  --jitter-depth 3 --gmc-workers 1 --anchor-step 10

# A1: the shipped baseline (q=8, one engine) -- the number to beat.
run_arm A1_q8_w1_it20      --input-queue 8   --jitter-depth 8 --gmc-workers 1 --anchor-step 10

# A3: deep Raw buffer + TWO engines.  The engines are what actually raise the
#     GMC throughput ceiling.
run_arm A3_q48_w2_it20     --input-queue 48  --jitter-depth 3 --gmc-workers 2 --anchor-step 10

# A5: deep buffer + two engines + the cheap LK (iters=10).  This is the arm
#     that should clear 40 ms/frame if the fit cost is the wall.
run_arm A5_q48_w2_it10     --input-queue 48  --jitter-depth 3 --gmc-workers 2 --anchor-step 10 --gmc-fit-iters 10

# A4: a very deep buffer, to see whether the queue depth itself has any
#     effect on the drop count once the engines are adequate.
run_arm A4_q96_w2_it20     --input-queue 96  --jitter-depth 3 --gmc-workers 2 --anchor-step 10

# A6: the same, with the RANSAC sampling capped (if the new CLI is present).
#     RANSAC maxIters changes the RNG stream, so it is a semantic change and
#     must be scored separately on the server.
if "$E2E" --help 2>&1 | grep -q "gmc-ransac-max-iters"; then
  run_arm A6_q48_w2_it10_r200 --input-queue 48 --jitter-depth 3 --gmc-workers 2 \
         --anchor-step 10 --gmc-fit-iters 10 --gmc-ransac-max-iters 200
else
  echo "[skip] A6: the RANSAC knob is not in this build" | tee -a "$SUMMARY"
fi

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
echo "" | tee -a "$SUMMARY"
echo "=== VERDICT ===" | tee -a "$SUMMARY"
{
  echo "Acceptance: dropped == 0 (camera-side).  features_dropped is allowed."
  echo ""
  printf "%-24s %10s %10s %10s %12s\n" ARM FIT_ms FPS DROPPED FEAT_DROP
  for arm in A2_q48_w1_it20 A1_q8_w1_it20 A3_q48_w2_it20 A5_q48_w2_it10 A4_q96_w2_it20 A6_q48_w2_it10_r200; do
    t=$RES/$arm/timing.txt
    [ -f "$t" ] || continue
    printf "%-24s %10s %10s %10s %12s\n" "$arm" \
      "$(grep -E '^gmc_fit ' "$t" | awk '{print $3}')" \
      "$(grep -E '^fps=' "$t" | cut -d= -f2)" \
      "$(grep -E '^dropped=' "$t" | cut -d= -f2)" \
      "$(grep -E '^features_dropped=' "$t" | cut -d= -f2)"
  done
  echo ""
  echo "25 fps budget = 40.0 ms/frame.  GMC throughput ceiling = 1000/gmc_fit fps."
} | tee -a "$SUMMARY"

cp -f "$SUMMARY" "/mnt/srv2/psearch/" 2>/dev/null
echo "SEARCH_DONE" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
