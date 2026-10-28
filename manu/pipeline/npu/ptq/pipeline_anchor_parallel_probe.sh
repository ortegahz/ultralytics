#!/bin/bash
# =============================================================================
# pipeline_anchor_parallel_probe.sh   (runs ON the RK3588 board)
#
# THE LEVER THAT CAN PUT step2 INTO THE 25 fps REGIME.
#
# WHY THIS PROBE
# --------------
# The GMC's cost per frame is NOT one fit.  The anchor grid at
# anchor_step=2 has 21 anchors (lags 2,4,...,42), and EVERY one of them is an
# independent LK fit; at anchor_step=10 there are only 5.  That is the whole
# reason step2 costs 204.6 ms/frame and step10 costs 59.9 ms/frame.
#
# The anchor LK stage is ALREADY written to be parallel: phase 1 (the cache
# work) is serial, phase 2 (the LK fits) runs concurrently over the anchors,
# phase 3 (the RANSAC + the gate) is serial IN THE ANCHOR ORDER.  Only the
# RANSAC stage touches OpenCV's global RNG, and it is still drawn in the
# reference's order -- so the numbers stay BIT-IDENTICAL and only the
# scheduling changes.  The knobs exist: --gmc-anchor-threads (clamped 1..8)
# and --gmc-affinity-group (one core group per engine).
#
# So step2's 21 anchor fits can be spread over up to 8 cores.  If the LK phase
# is the bulk of the 204 ms, step2 can reach the 25 fps regime -- and then the
# LIVE step2 arm can run at 25 fps with ZERO camera drops, which is what makes
# the "step2 must match the reference frame by frame" check meaningful.
#
# THE ARMS (600 frames, zero-drop, step2)
#   K0  the baseline: 1 engine, 1 anchor thread           (the reference path)
#   K1  1 engine, 4 anchor threads on cores 4-7
#   K2  1 engine, 8 anchor threads on cores 0-7           (max GMC throughput)
#   K3  1 engine, 6 anchor threads on cores 0-5, NPU+POST on 6-7
#   K4  2 engines, 4 anchor threads each, groups 0-3 and 4-7
#   K5  2 engines, 8 anchor threads each, one group 0-7 (both share 8 cores)
#
# EVERY arm is compared to K0 frame by frame: the anchor parallelism must be
# bit-identical, or it is not a free speed knob and the whole plan changes.
# =============================================================================
set -u

RES=/root/psearch/results
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
E2E=${E2E:-/root/rknn_mosaic_e2e_sweep3}
LIMIT=${LIMIT:-600}
OUT=/root/psearch/anchor_par
SUMMARY=$OUT/SUMMARY.txt

mkdir -p "$OUT"
echo "=== anchor-LK parallelism probe (step2, zero-drop, 600 frames) ===" | tee "$SUMMARY"
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
    sleep 6
    taskset -c 0-7 "$E2E" \
      --raw-root "$RAW_ROOT" --sequence "$SEQ" \
      --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
      --anchor-step 2 --pace-ms 0 --jitter-depth 8 --input-queue 8 \
      --npu-workers 1 --post-workers 1 --limit "$LIMIT" \
      --gmc-phys-anchor 1 "$@" > "$out/run.log" 2>&1
    rc=$?
    echo "[$arm] attempt=$attempt exit=$rc" | tee -a "$SUMMARY"
    [ "$rc" = "0" ] && break
  done
  grep -E "anchor LK stage on|^\[timing\]" "$out/run.log" | tail -5 | tee -a "$SUMMARY"
}

run_arm K0_serial            --gmc-workers 1 --gmc-anchor-threads 1 --cpu-affinity 4-7
run_arm K1_thr4_grp4_7       --gmc-workers 1 --gmc-anchor-threads 4 --gmc-affinity-group 4-7 --cpu-affinity 0-7
run_arm K2_thr8_grp0_7       --gmc-workers 1 --gmc-anchor-threads 8 --gmc-affinity-group 0-7 --cpu-affinity 0-7
run_arm K3_thr6_grp0_5_npu67 --gmc-workers 1 --gmc-anchor-threads 6 --gmc-affinity-group 0-5 --npu-affinity 6,7 --cpu-affinity 0-7
run_arm K4_w2_thr4           --gmc-workers 2 --gmc-anchor-threads 4 --gmc-affinity-group 0-3;4-7 --cpu-affinity 0-7
run_arm K5_w2_thr8           --gmc-workers 2 --gmc-anchor-threads 8 --gmc-affinity-group 0-7 --cpu-affinity 0-7

echo "" | tee -a "$SUMMARY"
echo "=== speed + bit-identity table ===" | tee -a "$SUMMARY"
printf "%-24s %10s %10s %10s %10s %12s\n" arm gmc_fit total_wall fps diff_vs_K0 first_diff | tee -a "$SUMMARY"
K0=$RES/K0_serial/feature_md5.txt
for arm in K0_serial K1_thr4_grp4_7 K2_thr8_grp0_7 K3_thr6_grp0_5_npu67 K4_w2_thr4 K5_w2_thr8; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || { printf "%-24s %10s\n" "$arm" "(missing)"; continue; }
  fit=$(grep -E "^gmc_fit " "$t" | awk '{print $3}')
  wall=$(grep -E "^total_wall " "$t" | awk '{print $3}')
  fps=$(grep -E "^fps=" "$t" | cut -d= -f2)
  res=$(python3 - "$K0" "$RES/$arm/feature_md5.txt" <<'PY'
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
    print("-"); raise SystemExit
common = sorted(set(a) & set(b))
diff = [k for k in common if a[k] != b[k]]
print(f"{len(diff)} {diff[0] if diff else '-'}")
PY
)
  d=$(echo "$res" | awk '{print $1}'); fd=$(echo "$res" | awk '{print $2}')
  printf "%-24s %10s %10s %10s %10s %12s\n" "$arm" "$fit" "$wall" "$fps" "$d" "$fd" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "=== probe done ===" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
