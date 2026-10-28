#!/bin/bash
# =============================================================================
# pipeline_gate_run.sh   (runs ON the RK3588 board)
#
# STAGE 0 -- the 600-frame bisect that the driver's crash cut short
# STAGE 1 -- THE GATE: the zero-drop, full-sequence step2 run whose E-arm F1
#           must land on the certified chain (C = 0.895555 for the board f16
#           mosaic arm; the e2e arms sit at 0.891676 step10 / 0.891871 step2)
#
# WHY STAGE 0 EXISTS
# ------------------
# The regression probe found that every new-binary arm differs from the
# certified zero-drop reference (P1b) starting at frame id 60.  The probe's
# arms used npu_workers=1 / input_queue=64 while P1b's own timing.txt says
# npu_workers=2 / input_queue=8 -- so the difference is a CONFIG difference,
# but the bisect that was supposed to isolate which knob died with the NPU
# driver.  It is re-run here, and it is the FIRST thing in the script because
# it is cheap (600 frames) and it decides whether the gate arm's result can
# be read as "the code is right" or "something moved".
#
#   G1  the OLD certified binary with P1b's exact config
#   G2  the NEW binary with P1b's exact config
#   Both are compared against P1b's first 600 frames.  If G1 matches and G2
#   does not, the refactor moved a number and must be found.  If both match,
#   the difference was the config and the new binary is inert.
#
# THE GATE ARM'S CONFIG
# ---------------------
# P1b's exact config, plus --gmc-workers 2.  The regression probe PROVED that
# two GMC engines are bit-identical to one (same per-frame fingerprints, and
# the wall clock drops from 247 to 151 ms/frame), so the two engines are a
# free wall-clock saving on a 30-minute run.  The arm's feature_md5.txt is
# compared against P1b's frame by frame afterwards: a 100% match means the
# whole chain -- code, driver, board -- reproduces the certified artefact
# exactly, and the F1 that comes out of the scorer MUST then be 0.891871.
# =============================================================================
set -u

RES=/root/psearch/results
REF=/mnt/srv2/psearch/results/P1b_step2_nodrop_orig
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
OLD=/root/rknn_mosaic_e2e
NEW=/root/rknn_mosaic_e2e_sweep3
OUT=/root/psearch/gate
SUMMARY=$OUT/SUMMARY.txt

mkdir -p "$OUT"
echo "=== gate run: bisect (600 frames) + the zero-drop full-sequence gate ===" | tee "$SUMMARY"
date | tee -a "$SUMMARY"

run_arm () {
  local arm=$1; local limit=$2; shift 2
  local out=$RES/$arm
  mkdir -p "$out"
  echo "" | tee -a "$SUMMARY"
  echo "--- $arm : $* ---" | tee -a "$SUMMARY"
  date | tee -a "$SUMMARY"
  local attempt rc
  for attempt in 1 2; do
    sleep 10
    taskset -c 4-7 "$1" \
      --raw-root "$RAW_ROOT" --sequence "$SEQ" \
      --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
      --limit "$limit" "$@" > "$out/run.log" 2>&1
    rc=$?
    echo "[$arm] attempt=$attempt exit=$rc" | tee -a "$SUMMARY"
    [ "$rc" = "0" ] && break
    grep -E "Illegal|FATAL|failed to submit" "$out/run.log" | head -2 | sed 's/^/    /' | tee -a "$SUMMARY"
  done
  grep -E "^\[timing\]" "$out/run.log" | tail -4 | tee -a "$SUMMARY"
}

###############################################################################
# STAGE 0 -- the bisect, 600 frames, P1b's exact config
###############################################################################
run_arm G1_old_p1b_cfg 600 "$OLD" --anchor-step 2 --pace-ms 0 --jitter-depth 8 \
      --input-queue 8 --npu-workers 2 --post-workers 1 --gmc-workers 1 --gmc-phys-anchor 0
run_arm G2_new_p1b_cfg 600 "$NEW" --anchor-step 2 --pace-ms 0 --jitter-depth 8 \
      --input-queue 8 --npu-workers 2 --post-workers 1 --gmc-workers 1 --gmc-phys-anchor 0
run_arm G3_new_p1b_cfg_anchor1 600 "$NEW" --anchor-step 2 --pace-ms 0 --jitter-depth 8 \
      --input-queue 8 --npu-workers 2 --post-workers 1 --gmc-workers 1 --gmc-phys-anchor 1

echo "" | tee -a "$SUMMARY"
echo "=== STAGE 0 result: against the certified reference ===" | tee -a "$SUMMARY"
python3 - "$REF/feature_md5.txt" "$RES/G1_old_p1b_cfg" "$RES/G2_new_p1b_cfg" "$RES/G3_new_p1b_cfg_anchor1" <<'PY' | tee -a "$SUMMARY"
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
ref = load(sys.argv[1])
print(f"  certified reference frames = {len(ref)}")
arms = {}
for d in sys.argv[2:]:
    name = os.path.basename(d)
    got = load(os.path.join(d, "feature_md5.txt"))
    if got is None:
        print(f"  {name:26s} NO feature_md5.txt"); continue
    arms[name] = got
    common = sorted(set(ref) & set(got))
    diff = [k for k in common if ref[k] != got[k]]
    print(f"  {name:26s} frames={len(got):5d} common={len(common):5d} "
          f"identical={len(common)-len(diff):5d} DIFFERENT={len(diff):5d}"
          + (f"  first_diff={diff[0]}" if diff else "  <-- MATCHES the certified reference"))
names = list(arms)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = arms[names[i]], arms[names[j]]
        common = sorted(set(a) & set(b))
        diff = [k for k in common if a[k] != b[k]]
        print(f"  pairwise {names[i]} vs {names[j]}: common={len(common):5d} DIFFERENT={len(diff):5d}"
              + (f"  first_diff={diff[0]}" if diff else "  <-- identical"))
PY

###############################################################################
# STAGE 1 -- THE GATE: the zero-drop, full-sequence step2 run
###############################################################################
# P1b's exact config, plus the two changes that the regression probe PROVED
# are bit-identical:
#   * --gmc-workers 2      (same per-frame fingerprints, 1.64x faster wall)
#   * --gmc-anchor-threads (the anchor LK stage in parallel; the RANSAC stays
#                           serial in the anchor order, so the RNG stream and
#                           therefore the numbers are unchanged)
# and --npu-workers 2, which is P1b's own value.
# phys-anchor 0: this arm runs the CERTIFIED anchoring mode, so it must
# reproduce the certified artefact exactly.
# The arm's feature_md5.txt is compared against P1b's frame by frame: a 100%
# match means the whole chain -- code, driver, board -- reproduces the
# certified artefact, and the F1 the scorer produces MUST then be 0.891871.
GMC_WORKERS=${GMC_WORKERS:-2}
ANCHOR_THREADS=${ANCHOR_THREADS:-4}
AFFINITY_GROUP=${AFFINITY_GROUP:-0-3;4-7}
CPU_AFF=${CPU_AFF:-0-7}
NPU_WORKERS=${NPU_WORKERS:-2}

run_arm A_gate_step2_nodrop 0 "$NEW" --anchor-step 2 --pace-ms 0 --jitter-depth 8 \
      --input-queue 8 --npu-workers "$NPU_WORKERS" --post-workers 1 \
      --gmc-workers "$GMC_WORKERS" --gmc-anchor-threads "$ANCHOR_THREADS" \
      --gmc-affinity-group "$AFFINITY_GROUP" --cpu-affinity "$CPU_AFF" \
      --gmc-phys-anchor 0

echo "" | tee -a "$SUMMARY"
echo "=== STAGE 1: the gate arm against the certified reference (11810 frames) ===" | tee -a "$SUMMARY"
python3 - "$REF/feature_md5.txt" "$RES/A_gate_step2_nodrop/feature_md5.txt" <<'PY' | tee -a "$SUMMARY"
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
ref = load(sys.argv[1]); got = load(sys.argv[2])
if ref is None or got is None:
    print("  a feature_md5.txt is missing"); raise SystemExit
common = sorted(set(ref) & set(got))
diff = [k for k in common if ref[k] != got[k]]
print(f"  certified reference frames = {len(ref)}   gate arm frames = {len(got)}")
print(f"  common = {len(common)}   identical = {len(common)-len(diff)}   DIFFERENT = {len(diff)}")
if diff:
    print(f"  FIRST differing frame id = {diff[0]}")
    print("  => the gate arm does NOT reproduce the certified artefact: read the F1 with care")
else:
    print("  => the gate arm reproduces the certified artefact BIT-IDENTICALLY on every common frame")
PY
grep -E "^(frames|gmc_frames|dropped|features_dropped|gmc_history_gaps|gmc_fit |gmc_total|rknn_run|fps|total_wall)" "$RES/A_gate_step2_nodrop/timing.txt" | sed 's/^/    /' | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "=== gate run done ===" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
