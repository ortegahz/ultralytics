#!/bin/bash
# =============================================================================
# pipeline_regression_probe.sh   (runs ON the RK3588 board)
#
# THE REGRESSION GATE FOR THE PHYSICAL-FRAME-NUMBER ANCHORING REFACTOR.
#
# WHY THIS PROBE EXISTS
# ---------------------
# The refactor changed HOW the GMC addresses its history: from the INGEST
# COUNTER to the CAMERA'S PHYSICAL FRAME NUMBER, plus a gap detector and a
# camera-queue drop-policy switch.  The certified reference (the zero-drop
# step2 run whose E-arm F1 = 0.891871 is recorded in memory) was produced by
# the OLD binary.  A refactor of the GMC's history addressing is exactly the
# kind of change that can silently move every number, so the refactor is only
# acceptable if the CERTIFIED PATH IS BIT-IDENTICAL.
#
# THE REFERENCE
#   /mnt/srv2/psearch/results/P1b_step2_nodrop_orig/feature_md5.txt
#   the zero-drop step2 run, 11810 frames, the artifact that scored 0.891871.
#   It is reachable from the board through /mnt/srv2 (the server's SSHFS), so
#   the probe compares against the ACTUAL reference artifact, not against
#   another run of the new binary.
#
# THE FOUR ARMS
#   PR_A  old certified binary            -- the reference run of this probe
#   PR_B  new binary, phys-anchor OFF     -- MUST equal PR_A (refactor is inert)
#   PR_C  new binary, phys-anchor ON      -- MUST equal PR_A (zero-drop makes
#                                            the two anchorings the same frame
#                                            sequence; this is the safety proof)
#   PR_D  new binary, phys-anchor OFF, 2 GMC engines
#                                         -- is a MULTI-ENGINE run bit-identical
#                                            to a single-engine one?  The GMC
#                                            draws from OpenCV's global RNG,
#                                            which is THREAD-LOCAL, so a second
#                                            engine thread has its own RNG
#                                            stream and the RANSAC sampling of
#                                            frame k can differ.  This arm
#                                            MEASURES that instead of assuming
#                                            it, because the answer decides
#                                            whether the zero-drop full-sequence
#                                            gate can be run with 2 engines
#                                            (20 min) or must use 1 (40 min).
#
# ALL arms are ZERO-DROP (--pace-ms 0: the producer waits for the pipeline,
# so the camera queue cannot overflow) and step2 -- the certified configuration.
# =============================================================================
set -u

RES=/root/psearch/results
REF=/mnt/srv2/psearch/results/P1b_step2_nodrop_orig/feature_md5.txt
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
LIMIT=${LIMIT:-600}
OUT=/root/psearch/regress
OLD=/root/rknn_mosaic_e2e
NEW=/root/rknn_mosaic_e2e_sweep3

mkdir -p "$OUT"
LOG=$OUT/probe.log
: > "$LOG"

echo "=== anchoring refactor regression probe ===" | tee -a "$LOG"
date | tee -a "$LOG"
echo "reference: $REF" | tee -a "$LOG"

run_arm () {
  local name=$1; local bin=$2; shift 2
  local out=$RES/$name
  mkdir -p "$out"
  echo "--- $name : $bin $* ---" | tee -a "$LOG"
  date | tee -a "$LOG"
  taskset -c 4-7 "$bin" \
    --raw-root "$RAW_ROOT" --sequence "$SEQ" \
    --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
    --cpu-affinity 4-7 --npu-workers 1 --post-workers 1 \
    --anchor-step 2 --pace-ms 0 --input-queue 64 --jitter-depth 8 \
    --limit "$LIMIT" "$@" > "$out/run.log" 2>&1
  echo "[$name] exit=$?" | tee -a "$LOG"
}

run_arm PR_A_old_certified   "$OLD"
run_arm PR_B_new_anchor0     "$NEW" --gmc-phys-anchor 0
run_arm PR_C_new_anchor1     "$NEW" --gmc-phys-anchor 1
run_arm PR_D_new_anchor0_w2  "$NEW" --gmc-phys-anchor 0 --gmc-workers 2

# -----------------------------------------------------------------------------
# THE COMPARISON
# -----------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== per-frame fingerprint comparison ===" | tee -a "$LOG"
python3 - "$REF" "$RES/PR_A_old_certified" "$RES/PR_B_new_anchor0" \
         "$RES/PR_C_new_anchor1" "$RES/PR_D_new_anchor0_w2" <<'PY' | tee -a "$LOG"
import sys, os

def load(path):
    """-> {frame_id: md5}  (the file is TAB-separated: <something> <id> <md5>)"""
    d = {}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    d[int(parts[1])] = parts[2]
                except ValueError:
                    continue
    return d

ref_path = sys.argv[1]
arms = sys.argv[2:]
ref = load(ref_path)
if ref is None:
    print("  !! reference not found:", ref_path)
    raise SystemExit(1)
print(f"  reference (certified zero-drop step2) frames = {len(ref)}")

for arm_dir in arms:
    name = os.path.basename(arm_dir)
    got = load(os.path.join(arm_dir, "feature_md5.txt"))
    if got is None:
        print(f"  {name:22s} NO feature_md5.txt")
        continue
    common = sorted(set(ref) & set(got))
    diff = [k for k in common if ref[k] != got[k]]
    print(f"  {name:22s} frames={len(got):5d}  common={len(common):5d}  "
          f"identical={len(common)-len(diff):5d}  DIFFERENT={len(diff):5d}")
    if diff:
        print(f"      FIRST differing frame id = {diff[0]}")
    # also compare the whole-file md5 of the overlapping prefix
    same_prefix = all(ref[k] == got[k] for k in common)
    print(f"      overlapping prefix identical: {same_prefix}")
PY

echo "" | tee -a "$LOG"
echo "=== arm timings ===" | tee -a "$LOG"
for a in PR_A_old_certified PR_B_new_anchor0 PR_C_new_anchor1 PR_D_new_anchor0_w2; do
  t=$RES/$a/timing.txt
  [ -f "$t" ] || continue
  echo "--- $a ---" | tee -a "$LOG"
  grep -E "^(frames|gmc_frames|dropped|features_dropped|gmc_history_gaps|gmc_fit |gmc_total|rknn_run|fps|total_wall)" "$t" | sed 's/^/    /' | tee -a "$LOG"
  grep -E "^\[timing\] GMC fit distribution" "$RES/$a/run.log" | sed 's/^/    /' | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== probe done ===" | tee -a "$LOG"
date | tee -a "$LOG"
