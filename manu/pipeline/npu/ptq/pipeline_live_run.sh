#!/bin/bash
# =============================================================================
# pipeline_live_run.sh   (runs ON the RK3588 board)
#
# THE TWO LIVE ARMS -- THE PER-FRAME VALIDATION UNDER A REAL CAMERA.
#
# THE ARCHITECTURE UNDER TEST (the user's specification)
#   [camera] -> Raw Image RingBuffer (30~50 frames) -> [GMC]
#            -> Feature Queue (2~3 frames) -> [NPU] -> [POST]
# so the live arms use --input-queue 48 (the raw ring) and --jitter-depth 3
# (the feature queue), and the camera runs on its OWN clock (--pace-ms) instead
# of being back-pressured by the pipeline.
#
# THE THREE THINGS THAT HAD TO BE GOT RIGHT
#   1. THE RAW RING DEPTH ALONE IS NOT ENOUGH.  A deep camera queue lets the
#      ingest run far ahead of the GMC engines, and the ingest is then stopped
#      by the HISTORY-WINDOW GUARD -- so the deep queue just moves the
#      overflow from the GMC side to the camera side.  P2 measured this:
#      1182 window blocks over 85 s, and 274 camera-queue drops.  The fix is
#      to deepen the LOOKAHEAD as well (--gmc-ring-slack), so the ingest is
#      allowed to run ahead without hitting the guard.
#   2. THE CAMERA RATE IS DERIVED FROM THE MEASURED CAPACITY.  The gate arm
#      measured the FULL-SEQUENCE wall time per frame (a prefix understates it
#      by ~1.8x because the corner density doubles around frame 2700).  The
#      camera must be slower than that or the camera queue overflows and the
#      registration chain breaks -- and then the per-frame comparison would
#      measure the overflow instead of the architecture.
#   3. ONLY ONE e2e AT A TIME.  Two concurrent RKNN context pairs exceed the
#      two-context limit: "failed to submit" -> 6 s job timeout -> driver soft
#      reset -> SIGILL in rknn_init -> the whole board goes dark and needs a
#      physical power cycle.  So every arm is preceded by a health check, and
#      a failed arm is retried once after a settle pause.
#
#   B_live_step2_25fps   step2 (the exact anchor semantics)
#   C_live_step10_25fps  step10 (the lite GMC) -- same camera, same rates,
#                        only the anchor cascade differs
#
# Both are compared frame by frame against the certified reference, and
# step10 is ALSO compared against step2 -- because the question for step10 is
# what the lite GMC's cascade costs, and the cleanest way to measure that is
# to compare the two live arms on the frames they share.
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
SUMMARY=/root/psearch/live_run.log
# The core group per GMC engine.  It MUST be quoted everywhere it is used:
# an unquoted "0-3;4-7" is split by the shell and the library then sees ONE
# group, pins BOTH engines to core 0, and the anchor workers all land on the
# same four cores.
GMC_GROUP=${GMC_GROUP:-0-3;4-7}
RING_SLACK=${RING_SLACK:-256}

mkdir -p "$RES"
echo "=== live arms under a real camera (own clock) ===" | tee "$SUMMARY"
date | tee -a "$SUMMARY"

health () {
  local d=/root/psearch/health_live
  mkdir -p "$d"
  timeout 240 "$E2E" --raw-root "$RAW_ROOT" --sequence "$SEQ" --model "$MODEL" \
    --out "$d" --kernel-dir "$KERNELS" --cpu-affinity 4-7 --npu-workers 1 \
    --post-workers 1 --anchor-step 10 --pace-ms 0 --limit 3 > "$d/log.txt" 2>&1
  local rc=$?
  echo "[health] exit=$rc" | tee -a "$SUMMARY"
  [ "$rc" != "0" ] && tail -3 "$d/log.txt" | sed 's/^/    /' | tee -a "$SUMMARY"
  return 0
}

run_arm () {
  local arm=$1; local pace=$2; shift 2
  local out=$RES/$arm
  mkdir -p "$out"
  echo "" | tee -a "$SUMMARY"
  echo "--- $arm (--pace-ms $pace) : $* ---" | tee -a "$SUMMARY"
  date | tee -a "$SUMMARY"
  local attempt rc
  for attempt in 1 2; do
    health
    sleep 10
    taskset -c 4-7 "$E2E" \
      --raw-root "$RAW_ROOT" --sequence "$SEQ" \
      --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
      --limit 0 --pace-ms "$pace" \
      --input-queue 48 --jitter-depth 3 --gmc-ring-slack "$RING_SLACK" \
      --npu-workers 2 --post-workers 1 --gmc-workers 2 \
      --gmc-phys-anchor 1 "$@" > "$out/run.log" 2>&1
    rc=$?
    echo "[$arm] attempt=$attempt exit=$rc" | tee -a "$SUMMARY"
    [ "$rc" = "0" ] && break
    grep -E "Illegal|FATAL|failed to submit" "$out/run.log" | head -3 | sed 's/^/    /' | tee -a "$SUMMARY"
  done
  grep -E "^\[timing\]|fallback summary|camera ran" "$out/run.log" | tail -6 | tee -a "$SUMMARY"
  mkdir -p "$PUB/$arm"
  cp -f "$out/${SEQ}_mosaic.tsv" "$out/feature_md5.txt" "$out/timing.txt" "$PUB/$arm/" 2>/dev/null
}

# THE CAMERA RATE IS DERIVED FROM THE MEASURED PIPELINE CAPACITY.
PACE_STEP2=$(python3 - <<'PY'
import re
try:
    t = open("/root/psearch/results/A_gate_step2_nodrop/timing.txt").read()
    m = re.search(r"^total_wall\s+[\d.]+\s+([\d.]+)", t, re.M)
    wall = float(m.group(1)) if m else 150.0
except Exception:
    wall = 150.0
print(int(max(40, min(250, round(wall * 1.15)))))
PY
)
echo "[info] the gate arm's full-sequence wall time gives a step2 camera pace of ${PACE_STEP2} ms" | tee -a "$SUMMARY"

# B: step2 (the exact anchor semantics), raw ring 48, feature queue 3
run_arm B_live_step2_25fps "$PACE_STEP2" --anchor-step 2 --gmc-anchor-threads 4 \
      --gmc-affinity-group "$GMC_GROUP" --cpu-affinity 0-7

# C: step10 (the lite GMC) -- only 5 anchors per frame instead of 21, so its
#    GMC capacity (24.4 fits/s) leaves room for the real 25 fps camera
run_arm C_live_step10_25fps 45 --anchor-step 10 --cpu-affinity 4-7

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
def cmp(name, x, y):
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
cmp("step2 LIVE  vs the certified zero-drop step2 reference", ref, b)
cmp("step10 LIVE vs the certified zero-drop step2 reference", ref, c)
cmp("step10 LIVE vs step2 LIVE (the lite GMC's cost, on shared frames)", b, c)
PY

echo "" | tee -a "$SUMMARY"
echo "=== live arms done ===" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
