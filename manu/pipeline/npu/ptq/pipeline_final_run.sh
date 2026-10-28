#!/bin/bash
# =============================================================================
# pipeline_final_run.sh   (runs ON the RK3588 board)
#
# THE LAST CONSOLIDATED RUN.  Everything that is still outstanding, in ONE
# script, in one order, with a health check in front of every arm.
#
# WHY ONE SCRIPT, ONE ORDER, NO INTERRUPTIONS
# ---------------------------------------------
# This board's RKNPU driver dies if it is pushed: a refused submit -> a 6 s
# job timeout -> a driver soft reset -> SIGILL in rknn_init -> and twice now
# the WHOLE BOARD went dark and needed a physical power cycle.  Killing an
# e2e mid-flight leaves its contexts un-released and is exactly what dirties
# the driver.  So:
#   * every arm is preceded by a 3-frame health check,
#   * every arm gets ONE retry after a settle pause,
#   * NO arm is ever killed while it runs,
#   * the number of e2e invocations is kept as small as possible.
#
# THE ARMS
#   B_live_step2_25fps   step2 (the exact anchor semantics) under a real
#                        camera on its own clock.  The raw ring is 48 frames
#                        and the feature queue 3, as the architecture says.
#                        The camera rate is derived from the gate arm's
#                        measured capacity; the camera must be SLOWER than the
#                        GMC, because only the GMC's rate decides whether the
#                        camera queue overflows -- the NPU's skipping is an
#                        accepted loss.
#   C_live_step10_25fps  step10 (the lite GMC), same camera discipline.
#   S0..S3               the throughput sweep (the objective: processed frames
#                        per second).
#
# AFTERWARDS the script compares, frame by frame:
#   reference vs B,  reference vs C,  B vs C
# and publishes every arm to the server-visible path so the server's scoring
# watcher produces the val3 report for each of them.
# =============================================================================
set -u

RES=/root/psearch/results
PUB=/mnt/srv2/psearch/results
REF=/mnt/srv2/psearch/results/P1b_step2_nodrop_orig
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
E2E=${E2E:-/root/rknn_mosaic_e2e_sweep3}
SUMMARY=/root/psearch/final_run.log
# MUST be quoted wherever it is used: an unquoted "0-3;4-7" is split by the
# shell, the library then sees ONE core group, pins BOTH GMC engines to
# core 0, and the anchor workers all land on the same four cores.
GMC_GROUP=${GMC_GROUP:-0-3;4-7}
RING_SLACK=${RING_SLACK:-256}
# THE REAL CAMERA RATE.  Both live arms run at 25 fps (--pace-ms 40) -- the
# camera is NOT slowed down to suit the pipeline.  The pipeline processes what
# it can and the rest is skipped, and the question is only whether the frames
# that WERE reported agree with the zero-drop experiment on the same frames.
PACE_B=${PACE_B:-40}
PACE_C=${PACE_C:-40}
SWEEP_LIMIT=${SWEEP_LIMIT:-4000}

mkdir -p "$RES"
echo "=== FINAL consolidated run (live arms + throughput sweep) ===" | tee "$SUMMARY"
date | tee -a "$SUMMARY"

health () {
  local d=/root/psearch/health_final
  mkdir -p "$d"
  timeout 240 "$E2E" --raw-root "$RAW_ROOT" --sequence "$SEQ" --model "$MODEL" \
    --out "$d" --kernel-dir "$KERNELS" --cpu-affinity 4-7 --npu-workers 1 \
    --post-workers 1 --anchor-step 10 --pace-ms 0 --limit 3 > "$d/log.txt" 2>&1
  echo "[health] exit=$?" | tee -a "$SUMMARY"
}

run_arm () {
  local arm=$1; local pace=$2; local limit=$3; shift 3
  local out=$RES/$arm
  mkdir -p "$out"
  echo "" | tee -a "$SUMMARY"
  echo "--- $arm (--pace-ms $pace --limit $limit) : $* ---" | tee -a "$SUMMARY"
  date | tee -a "$SUMMARY"
  local attempt rc
  for attempt in 1 2; do
    health
    sleep 10
    taskset -c 4-7 "$E2E" \
      --raw-root "$RAW_ROOT" --sequence "$SEQ" \
      --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
      --limit "$limit" --pace-ms "$pace" \
      --input-queue 48 --jitter-depth 3 --gmc-ring-slack "$RING_SLACK" \
      --npu-workers 2 --post-workers 1 --gmc-workers 2 \
      --gmc-phys-anchor 1 "$@" > "$out/run.log" 2>&1
    rc=$?
    echo "[$arm] attempt=$attempt exit=$rc" | tee -a "$SUMMARY"
    [ "$rc" = "0" ] && break
    grep -E "Illegal|FATAL|failed to submit" "$out/run.log" | head -3 | sed 's/^/    /' | tee -a "$SUMMARY"
  done
  grep -E "^\[timing\]|fallback summary|camera ran" "$out/run.log" | tail -6 | tee -a "$SUMMARY"
  # the contiguity of the produced frames IS the zero-drop proof
  python3 - "$out/feature_md5.txt" <<'PY' | tee -a "$SUMMARY"
import sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print("    no feature_md5.txt"); raise SystemExit
ids = []
for line in open(p):
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 3:
        try: ids.append(int(parts[1]))
        except ValueError: pass
gaps = [(a, b, b - a - 1) for a, b in zip(ids, ids[1:]) if b - a != 1]
print(f"    produced={len(ids)} first={ids[0] if ids else '-'} last={ids[-1] if ids else '-'}"
      f"  gap_events={len(gaps)}  missing={sum(g[2] for g in gaps)}")
if gaps:
    print(f"    first gaps: {gaps[:5]}")
PY
  mkdir -p "$PUB/$arm"
  cp -f "$out/${SEQ}_mosaic.tsv" "$out/feature_md5.txt" "$out/timing.txt" "$PUB/$arm/" 2>/dev/null
}

# --- B: step2 under a real camera -----------------------------------------
run_arm B_live_step2_25fps "$PACE_B" 0 --anchor-step 2 --gmc-anchor-threads 4 \
      --gmc-affinity-group "$GMC_GROUP" --cpu-affinity 0-7

# --- C: step10 under a real camera ----------------------------------------
run_arm C_live_step10_25fps "$PACE_C" 0 --anchor-step 10 --cpu-affinity 4-7

# --- the throughput sweep --------------------------------------------------
run_arm S0_npu1              40 "$SWEEP_LIMIT" --anchor-step 10 --npu-workers 1 --cpu-affinity 4-7
run_arm S1_npu2              40 "$SWEEP_LIMIT" --anchor-step 10 --npu-workers 2 --cpu-affinity 4-7
run_arm S2_npu2_slack256     40 "$SWEEP_LIMIT" --anchor-step 10 --npu-workers 2 --cpu-affinity 4-7
run_arm S3_npu2_mask7        40 "$SWEEP_LIMIT" --anchor-step 10 --npu-workers 2 --npu-core-mask 7 --cpu-affinity 4-7

echo "" | tee -a "$SUMMARY"
echo "=== per-frame comparison ===" | tee -a "$SUMMARY"
python3 - "$REF/feature_md5.txt" "$RES/B_live_step2_25fps/feature_md5.txt" \
         "$RES/C_live_step10_25fps/feature_md5.txt" <<'PY' | tee -a "$SUMMARY"
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
ref = load(sys.argv[1]); b = load(sys.argv[2]); c = load(sys.argv[3])

def gaps_of(d):
    """the frame ids the arm DID NOT report -- the camera-side drops"""
    if not d: return []
    ids = sorted(d)
    out = []
    for a, z in zip(ids, ids[1:]):
        if z - a != 1:
            out.append((a, z, z - a - 1))
    return out

def cmp(name, x, y, show_gaps=False):
    if x is None or y is None:
        print(f"  {name}: MISSING"); return
    common = sorted(set(x) & set(y))
    diff = [k for k in common if x[k] != y[k]]
    print(f"  {name}")
    print(f"    x={len(x)} frames  y={len(y)} frames  common={len(common)}")
    print(f"    IDENTICAL={len(common)-len(diff)}  DIFFERENT={len(diff)}"
          + (f"  first_diff={diff[0]}" if diff else "  <-- BIT-IDENTICAL"))
    if diff:
        runs = []
        for v in diff:
            if runs and v == runs[-1][1] + 1: runs[-1][1] = v
            else: runs.append([v, v])
        print(f"    differing runs = {runs[:8]}{' ...' if len(runs)>8 else ''}  (total {len(runs)} runs)")
    if show_gaps:
        g = gaps_of(y)
        miss = sum(t[2] for t in g)
        print(f"    the arm did NOT report {len(g)} frame id(s) ({miss} frames missing)"
              + (f"; first gap {g[0][0]} -> {g[0][1]}" if g else ""))
        if g and diff:
            print(f"    CROSS-CHECK: the first missing frame is {g[0][1]}, the first differing"
                  f" frame is {diff[0]}  ->  "
                  + ("the divergence starts at/after the first drop (the drop explains it)"
                     if diff[0] >= g[0][1] else
                     "the divergence PRECEDES the first drop (something else causes it)"))

cmp("step2 LIVE  vs the certified zero-drop step2 reference  [md5, frame by frame]",
    ref, b, show_gaps=True)
cmp("step10 LIVE vs the certified zero-drop step2 reference  [md5, frame by frame]",
    ref, c, show_gaps=True)
cmp("step10 LIVE vs step2 LIVE (the lite GMC's cost, on the shared frames)", b, c)
PY

echo "" | tee -a "$SUMMARY"
echo "=== the throughput table (the fps column is the objective) ===" | tee -a "$SUMMARY"
printf "%-22s %8s %8s %9s %9s %9s %9s %9s\n" arm frames dropped feat_drop gmc_fit rknn_run wall fps | tee -a "$SUMMARY"
for arm in B_live_step2_25fps C_live_step10_25fps S0_npu1 S1_npu2 S2_npu2_slack256 S3_npu2_mask7; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || { printf "%-22s %8s\n" "$arm" "(missing)"; continue; }
  get() { grep -E "^$1=" "$t" | head -1 | cut -d= -f2; }
  printf "%-22s %8s %8s %9s %9s %9s %9s %9s\n" "$arm" \
    "$(get frames)" "$(get dropped)" "$(get features_dropped)" \
    "$(grep -E '^gmc_fit ' "$t" | awk '{print $3}')" \
    "$(grep -E '^rknn_run ' "$t" | awk '{print $3}')" \
    "$(grep -E '^total_wall ' "$t" | awk '{print $3}')" \
    "$(get fps)" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== per-stage detail (the live arms are the throughput + correctness arms) ===" | tee -a "$SUMMARY"
for arm in B_live_step2_25fps C_live_step10_25fps S0_npu1 S1_npu2 S2_npu2_slack256 S3_npu2_mask7; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || continue
  echo "--- $arm ---" | tee -a "$SUMMARY"
  grep -E "^(gmc_total|gmc_post|gmc_ingest|gmc_ingest_window|gmc_ingest_space|gmc_wait|npu_wait|post_wait|upload|outputs_get|backproject_cc|rga_import|rga_improcess|gmc_queue_depth_mean|gmc_window_blocks|gmc_space_blocks|gmc_ring_slack|gmc_fit_p50|gmc_fit_p95|gmc_fit_p99|gmc_fit_max|rknn_run_p50|rknn_run_max|gmc_corner_mean|gmc_corner_capped|pushed|gmc_frames|det_frames|dropped|features_dropped|detections_dropped|gmc_history_gaps|pace_ms)=" "$t" | sed 's/^/    /' | tee -a "$SUMMARY"
  grep -E "^\[timing\] GMC fit distribution" "$RES/$arm/run.log" 2>/dev/null | sed 's/^/    /' | tee -a "$SUMMARY"
  grep -E "the camera ran at|fallback summary" "$RES/$arm/run.log" 2>/dev/null | sed 's/^/    /' | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== final run done ===" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
