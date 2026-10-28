#!/bin/bash
# =============================================================================
# pipeline_precision_protocol.sh   (runs ON the RK3588 board)
#
# THE THREE-STEP PRECISION PROTOCOL for the decoupled-pipeline architecture.
#
# STEP 1  Zero-drop, full-sequence val T3 scoring of step2.
#         The reference experiment.  A step2 run that drops NOTHING must score
#         exactly what yesterday's certified run scored (E-arm F1 0.891871).
#         If it does not, the harness is wrong -- not the architecture.
#
# STEP 2  The same step2 features under a LIVE 25 fps camera.
#         Question: are the frames that DID get processed bit-identical to the
#         zero-drop run's frames?  They must be: the GMC math depends only on
#         pixels, and a frame's history is addressed by its own position.
#         A mismatch here would mean the pipeline is corrupting frames.
#         NOTE: the comparison is over the frames present in BOTH runs.  The
#         paced run may produce FEWER frames (the inference stage skips what it
#         cannot keep up with) -- that is allowed.  What must not happen is a
#         surviving frame whose CONTENT differs.
#
# STEP 3  The live 25 fps camera running step10 instead of step2.
#         The question: how much accuracy does the cheaper anchor cascade
#         cost, compared with step2?  Answered by scoring the live step10 arm
#         and comparing its E-arm F1 with step2's.
#
# WHY THE COMPARISON IS PER-FRAME, NOT JUST A SCORE
# A score can hide a lot.  A per-frame fingerprint comparison says exactly
# WHICH frame first differs, so a failure is diagnosable instead of merely
# "the F1 is lower".
#
# THE ANCHORING FACT THAT GOVERNS STEP 2
# --------------------------------------
# The GMC addresses its history by an ABSOLUTE COUNTER (abs_counter) that is
# incremented once per INGESTED frame, while the physical camera frame number
# is a different counter (next_id) incremented once per PUSHED frame.  The two
# are equal only while nothing is dropped.  So:
#
#   dropped == 0  =>  the counter IS the physical frame number => the 21-frame
#                    median window covers exactly 21 physical frames (0.84 s
#                    at 25 fps) => the geometry is the training-set geometry.
#   dropped  > 0  =>  the window silently compresses => the chain is broken.
#
# So a deep buffer CANNOT substitute for zero drops; it can only delay the
# moment the accounting diverges.  This protocol measures both.
# =============================================================================
set -u

REMOTE=/root/psearch
PROTO=$REMOTE/protocol
RES=$REMOTE/results
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
E2E=${E2E:-/root/rknn_mosaic_e2e_sweep}
LIMIT=${LIMIT:-0}

mkdir -p "$PROTO"
OUT=$PROTO/SUMMARY.txt
echo "=== decoupled-pipeline precision protocol (3 steps) ===" | tee "$OUT"
date | tee -a "$OUT"

run () {
  local arm=$1; shift
  local out=$RES/$arm
  mkdir -p "$out"
  echo "--- run $arm : $* ---" | tee -a "$OUT"
  date | tee -a "$OUT"
  taskset -c 4-7 "$E2E" \
    --raw-root "$RAW_ROOT" --sequence "$SEQ" \
    --model "$MODEL" \
    --out "$out" --kernel-dir "$KERNELS" \
    --cpu-affinity 4-7 --npu-workers 1 --post-workers 1 \
    "${extra[@]}" > "$out/run.log" 2>&1
  echo "[$arm] exit=$?" | tee -a "$OUT"
  grep -E "^\[timing\]|camera ran|fallback" "$out/run.log" | tee -a "$OUT"
}

# ===========================================================================
# STEP 1 -- zero-drop, full-sequence step2.  The reference.
# --pace-ms 0 = self-paced: the producer waits for the pipeline, so the
# camera queue cannot overflow and NOTHING is dropped.  This is the run whose
# E-arm F1 must reproduce 0.891871.
# ===========================================================================
echo "" | tee -a "$OUT"
echo "### STEP 1: step2, zero-drop, full sequence (reference)" | tee -a "$OUT"
extra=(--anchor-step 2 --pace-ms 0 --input-queue 64 --jitter-depth 8 --gmc-workers 1 --limit "$LIMIT")
run P1_step2_nodrop "${extra[@]}"

# ===========================================================================
# STEP 2 -- a LIVE 25 fps camera.  THE ARM THAT ACTUALLY TESTS THE ARCHITECTURE.
# ===========================================================================
# NOTE ON THE ANCHOR STEP HERE.  A live step2 run cannot answer the question:
# step2 fits at 204 ms/frame, i.e. 4.9 fps against a 25 fps camera, so the
# camera queue overflows within seconds and the whole sequence is a broken
# chain.  Comparing "surviving frames" of such a run measures the collapse,
# not the architecture.
#
# So the LIVE arm uses step10 with two engines, the configuration that actually
# sustains ~24 fps (measured: svc_Q).  That is the regime the architecture is
# FOR, and it is the only regime in which the question "does the live pipeline
# corrupt the frames it processes?" has a meaningful answer.
#
# The frames are compared against the ZERO-DROP step2 reference (P1), so the
# comparison answers TWO things at once:
#   - does the live pipeline corrupt a frame?  (a surviving frame must be
#     bit-identical to the reference)
#   - what does the step10 cascade cost?  (a surviving frame's step10 hash vs
#     the step2 reference: equal => the cascade is free)
# ===========================================================================
echo "" | tee -a "$OUT"
echo "### STEP 2: step10 + 2 engines under a live 25 fps camera" | tee -a "$OUT"
extra=(--anchor-step 10 --pace-ms 40 --input-queue 48 --jitter-depth 3 --gmc-workers 2 --limit "$LIMIT")
run P2_step10_live25 "${extra[@]}"

# ===========================================================================
# STEP 3 -- the live 25 fps camera running step2.  The reference semantics
# under the same live conditions: how far the real-time path degrades the
# accuracy when the anchor cascade is the expensive one.
# ===========================================================================
echo "" | tee -a "$OUT"
echo "### STEP 3: step2 under a live 25 fps camera" | tee -a "$OUT"
extra=(--anchor-step 2 --pace-ms 40 --input-queue 48 --jitter-depth 3 --gmc-workers 2 --limit "$LIMIT")
run P3_step2_live25 "${extra[@]}"

# ===========================================================================
# THE PER-FRAME COMPARISON
# ===========================================================================
# WHAT TO EXPECT, AND WHY
# ----------------------
# The GMC addresses its history by an ABSOLUTE COUNTER (abs_counter) that is
# incremented once per INGESTED frame, while the physical camera frame number
# is a different counter (next_id) incremented once per PUSHED frame.  The two
# are equal only while nothing is dropped.  Therefore:
#
#   dropped == 0  =>  counter == physical id => the 21-frame median window
#                    covers exactly 21 physical frames (0.84 s at 25 fps)
#                    => the geometry IS the training-set geometry
#                    => every surviving frame is BIT-IDENTICAL to the
#                       zero-drop run.
#
#   dropped  > 0  =>  the counters diverge => the window stretches (the GMC
#                    asks for "lag 2" and gets physical lag 3) => the
#                    registration chain is broken from that point on, and
#                    every later frame differs from the zero-drop run.
#
# So the comparison MUST report the FIRST differing frame id and the first
# dropped frame id, and check whether they coincide.  A single summary
# number would hide exactly the information that matters.
# ===========================================================================
echo "" | tee -a "$OUT"
echo "=== STEP 2 per-frame fingerprint comparison (P1 zero-drop step2 vs P2 live step10) ===" | tee -a "$OUT"
F2=$RES/P2_step10_live25/feature_md5.txt
# The first frame the live run did NOT produce tells where the camera queue
# overflowed (or where the inference stage started skipping).  It must line up
# with the first differing frame in the comparison below -- that coincidence
# IS the proof of the mechanism.
if [ -f "$F2" ]; then
  awk -F"\t" "{print \$2}" "$F2" | awk '
      {n=$1+0; if (NR>1 && n != prev+1) {print "  first GAP in the live run: frame " prev " -> " n "  (missing " n-prev-1 ")"; exit} prev=n}
    ' | tee -a "$OUT"
fi
F1=$RES/P1_step2_nodrop/feature_md5.txt
if [ ! -f "$F1" ]; then
  # The zero-drop step2 reference already exists from the earlier run
  # (svc_S2_w1_nodrop); it was rescued to the server-visible path before the
  # host shut down.  Use it as P1 so the protocol does not have to re-run it.
  F1=/mnt/srv2/psearch/results/P1b_step2_nodrop_orig/feature_md5.txt
  echo "[info] using the rescued zero-drop step2 reference: $F1" | tee -a "$OUT"
fi
if [ -f "$F1" ] && [ -f "$F2" ]; then
  n1=$(wc -l < "$F1"); n2=$(wc -l < "$F2")
  echo "P1 (zero-drop) frames = $n1   P2 (live 25fps) frames = $n2" | tee -a "$OUT"
  # Compare only the frame ids present in BOTH.  A frame missing from P2 was
  # skipped by the inference stage -- that is allowed and is NOT a failure.
  # A frame present in both with a DIFFERENT hash is a real corruption.
  python3 - "$F1" "$F2" <<'PY' | tee -a "$OUT"
import sys
def load(p):
    d = {}
    with open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                d[parts[1]] = parts[2]
    return d
a = load(sys.argv[1]); b = load(sys.argv[2])
common = sorted(set(a) & set(b))
diff = [k for k in common if a[k] != b[k]]
print(f"  frames compared        = {len(common)}")
print(f"  identical              = {len(common) - len(diff)}")
print(f"  DIFFERENT              = {len(diff)}")
if diff:
    print(f"  FIRST differing frame id = {diff[0]}   (of {len(common)} common frames)")
    print("  => the divergence point: everything BEFORE it is bit-identical to the")
    print("     zero-drop run, everything AFTER it is not.  Cross-check this id")
    print("     against the first camera-side drop of the live run.")
else:
    print("  => every surviving frame is BIT-IDENTICAL: the live pipeline does not")
    print("     corrupt frames; it only ever SKIPS them.")
missing = sorted(set(a) - set(b))
print(f"  frames only in P1 (not produced by the live run) = {len(missing)}")
if missing:
    print(f"  first such frame id  = {missing[0]}")
PY
else
  echo "(a run is missing its feature_md5.txt)" | tee -a "$OUT"
fi

# ===========================================================================
# STEP 3's PER-FRAME COMPARISON  (P2 step2-live  vs  P3 step10-live)
# ===========================================================================
# THIS IS THE POINT OF STEP 3, and it is NOT the score.
#
# The question is: for the frames that BOTH live runs produced, is the step10
# feature BIT-IDENTICAL to the step2 feature?  If it is, then the cheaper
# anchor cascade costs ZERO accuracy -- not "a little" -- because the two
# cascade geometries agree exactly on those frames.
#
# Only the frames present in BOTH runs are compared: a frame the live run
# skipped (inference stage could not keep up) is not evidence of anything.
# This is the "只比对存活帧" rule.
# ===========================================================================
echo "" | tee -a "$OUT"
echo "=== STEP 3 per-frame fingerprint comparison (P2 step10-live vs P3 step2-live) ===" | tee -a "$OUT"
FA=$RES/P2_step10_live25/feature_md5.txt
FB=$RES/P3_step2_live25/feature_md5.txt
if [ -f "$FA" ] && [ -f "$FB" ]; then
  na=$(wc -l < "$FA"); nb=$(wc -l < "$FB")
  echo "P2 (step10 live) frames = $na   P3 (step2 live) frames = $nb" | tee -a "$OUT"
  python3 - "$FA" "$FB" <<'PY' | tee -a "$OUT"
import sys
def load(p):
    d = {}
    with open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                d[parts[1]] = parts[2]
    return d
a = load(sys.argv[1]); b = load(sys.argv[2])
common = sorted(set(a) & set(b))
diff = [k for k in common if a[k] != b[k]]
print(f"  frames compared (surviving in BOTH live runs) = {len(common)}")
print(f"  identical                                     = {len(common) - len(diff)}")
print(f"  DIFFERENT                                     = {len(diff)}")
if diff:
    print(f"  FIRST differing frame id = {diff[0]}  (of {len(common)} common)")
    print("  => the step10 cascade diverges from step2 here; score the arms to")
    print("     quantify what that divergence costs.")
else:
    print("  => step10 and step2 produce BIT-IDENTICAL features on every frame")
    print("     that survived in both live runs => the cheaper anchor cascade")
    print("     costs ZERO feature-level accuracy.")
onlyA = sorted(set(a) - set(b)); onlyB = sorted(set(b) - set(a))
print(f"  only in P2 (step10 live) = {len(onlyA)}" + (f", first = {onlyA[0]}" if onlyA else ""))
print(f"  only in P3 (step2 live)  = {len(onlyB)}" + (f", first = {onlyB[0]}" if onlyB else ""))
PY
else
  echo "(a run is missing its feature_md5.txt)" | tee -a "$OUT"
fi

# ---------------------------------------------------------------------------
# THE THIRD COMPARISON: step10-live vs the ZERO-DROP step2 reference.
# ---------------------------------------------------------------------------
# The P2-vs-P3 comparison above mixes TWO effects: the anchor cascade (step10
# vs step2) AND the different drop positions (P2 and P3 overflow the camera
# queue at different frames, so the two runs' surviving windows are not the
# same).  Comparing BOTH live runs against the SAME zero-drop reference
# separates them: the reference has no drops at all, so any divergence is
# attributable to the drop chain, and the P2-vs-P3 divergence on frames that
# are clean in both is attributable to the cascade.
# ---------------------------------------------------------------------------
F0=/mnt/srv2/psearch/results/P1b_step2_nodrop_orig/feature_md5.txt
if [ -f "$F0" ]; then
  echo "" | tee -a "$OUT"
  echo "=== STEP 3b: each live run vs the ZERO-DROP step2 reference ===" | tee -a "$OUT"
  for pair in "P2_step10:$FA" "P3_step2:$FB"; do
    tag=${pair%%:*}; f=${pair#*:}
    [ -f "$f" ] || continue
    python3 - "$F0" "$f" "$tag" <<'PY' | tee -a "$OUT"
import sys
def load(p):
    d = {}
    with open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                d[parts[1]] = parts[2]
    return d
ref = load(sys.argv[1]); x = load(sys.argv[2]); tag = sys.argv[3]
common = sorted(set(ref) & set(x))
diff = [k for k in common if ref[k] != x[k]]
first = diff[0] if diff else None
print(f"  {tag} vs zero-drop reference: common={len(common)}  identical={len(common)-len(diff)}  different={len(diff)}")
if first is not None:
    print(f"    first differing frame id = {first}")
    # How many frames BEFORE the first divergence are identical?  That is the
    # "clean prefix" -- the frames produced before the camera queue overflowed.
    before = [k for k in common if int(k) < int(first)]
    clean = [k for k in before if ref[k] == x[k]]
    print(f"    clean prefix length = {len(clean)} frames  (identical before the overflow)")
PY
  done
else
  echo "(the zero-drop reference is not available)" | tee -a "$OUT"
fi

# ===========================================================================
# THE NUMBERS
# ===========================================================================
echo "" | tee -a "$OUT"
echo "=== SUMMARY ===" | tee -a "$OUT"
printf "%-22s %10s %10s %10s %12s %12s\n" ARM FIT_ms FPS DROPPED FEAT_DROP CORNERS
for arm in P1_step2_nodrop P2_step10_live25 P3_step2_live25; do
  t=$RES/$arm/timing.txt
  [ -f "$t" ] || continue
  printf "%-22s %10s %10s %10s %12s %12s\n" "$arm" \
    "$(grep -E '^gmc_fit ' "$t" | awk '{print $3}')" \
    "$(grep -E '^fps=' "$t" | cut -d= -f2)" \
    "$(grep -E '^dropped=' "$t" | cut -d= -f2)" \
    "$(grep -E '^features_dropped=' "$t" | cut -d= -f2)" \
    "$(grep -E '^gmc_corner_mean=' "$t" | cut -d= -f2)"
done
echo "" | tee -a "$OUT"
echo "P1/P2/P3 are scored on the server by:" | tee -a "$OUT"
echo "  bash manu/pipeline/npu/ptq/score_pipeline_arms.sh P1_step2_nodrop P2_step10_live25 P3_step2_live25" | tee -a "$OUT"
echo "  references: step2 zero-drop E-arm F1 = 0.891871 ; step10 zero-drop E-arm F1 = 0.891676" | tee -a "$OUT"

cp -f "$OUT" "/mnt/srv2/psearch/" 2>/dev/null
echo "PROTOCOL_DONE" | tee -a "$OUT"
date | tee -a "$OUT"
