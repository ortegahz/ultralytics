#!/bin/bash
# =============================================================================
# pipeline_equiv_check.sh
#
# PROVES (or disproves) the central claim of the decoupled architecture:
#
#   "As long as the geometric registration behind a feature is anchored to the
#    original 25 fps TIME GRID, the feature quality is 100% equivalent to the
#    training set -- the precision does not decay."
#
# The claim has two halves, and they are NOT the same claim:
#
#   (1) PIPELINE-EQUIVALENCE: the SAME frames, processed by pipelines with
#       DIFFERENT queue depths / worker counts, must produce IDENTICAL
#       features.  This is what the absolute frame_id anchoring buys.  If it
#       holds, the architecture can be retuned freely without touching the
#       model -- the features are the same objects.
#
#   (2) TRAINING-EQUIVALENCE: the features must equal the FROZEN training set
#       (uav_gmc_median).  This is a different question and is checked on the
#       server by the T3 scorer, not here.
#
# This script checks (1): it runs the same frame range under several pipeline
# configurations and compares the per-frame feature fingerprints.
#
# A PASS means: the queue depths and the worker count are PURE THROUGHPUT
# KNOBS.  They change how many frames the inference stage skips; they do not
# change a single feature.  That is exactly the freedom the architecture
# needs.
#
# A FAIL means: the architecture has a latent coupling that would force a
# retrain, and the queue depth would be a SEMANTIC knob after all.
#
# NOTE ON THE DURATION: this uses a FRAME PREFIX (--limit).  That is safe
# HERE, unlike in the throughput search: the question is whether two
# configurations agree on the SAME frames, not how fast they are.  Both runs
# see identical pixels, so any difference is a coupling, not a content effect.
# =============================================================================
set -u

REMOTE=/root/psearch
EQUIV=$REMOTE/equiv
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
E2E=${E2E:-/root/rknn_mosaic_e2e_sweep}
LIMIT=${LIMIT:-400}

mkdir -p "$EQUIV"
OUT=$EQUIV/SUMMARY.txt
echo "=== pipeline equivalence check (same frames, different pipeline shapes) ===" | tee "$OUT"
date | tee -a "$OUT"

run () {
  local tag=$1; shift
  local d=$EQUIV/$tag
  mkdir -p "$d"
  # --pace-ms 0 = self-paced (the producer waits for the pipeline).  The
  # pipeline SHAPE still differs between the arms, which is what is being
  # compared.  A paced run would additionally let the inference stage skip
  # frames, and those frames would have no feature to compare.
  taskset -c 4-7 "$E2E" \
    --raw-root "$RAW_ROOT" --sequence "$SEQ" \
    --model "$MODEL" \
    --out "$d" --kernel-dir "$KERNELS" \
    --pace-ms 0 --cpu-affinity 4-7 \
    --limit "$LIMIT" "${extra[@]}" > "$d/run.log" 2>&1
  local rc=$?
  local n=$(wc -l < "$d/feature_md5.txt" 2>/dev/null || echo 0)
  echo "[$tag] exit=$rc  feature_md5_lines=$n" | tee -a "$OUT"
  grep -E "^\[timing\] gmc_fit|dropped=" "$d/run.log" | head -2 | tee -a "$OUT"
}

# E1: the shipped shape -- q=8, one engine, jitter 8.
echo "--- E1: q8  w1 jitter8 (shipped shape) ---" | tee -a "$OUT"
extra=(--input-queue 8 --jitter-depth 8 --gmc-workers 1 --anchor-step 10)
run E1_q8_w1_j8

# E2: the proposal's shape -- deep raw buffer, shallow feature queue.
echo "--- E2: q48 w1 jitter3 (proposal shape) ---" | tee -a "$OUT"
extra=(--input-queue 48 --jitter-depth 3 --gmc-workers 1 --anchor-step 10)
run E2_q48_w1_j3

# E3: deep buffer + TWO engines.  The engine count changes which core runs
#     which fit -- the strongest possible test of the absolute anchoring.
echo "--- E3: q48 w2 jitter3 (two engines) ---" | tee -a "$OUT"
extra=(--input-queue 48 --jitter-depth 3 --gmc-workers 2 --anchor-step 10)
run E3_q48_w2_j3

# E4: a very deep buffer.
echo "--- E4: q96 w2 jitter3 (very deep buffer) ---" | tee -a "$OUT"
extra=(--input-queue 96 --jitter-depth 3 --gmc-workers 2 --anchor-step 10)
run E4_q96_w2_j3

# ---- compare -----------------------------------------------------------------
echo "" | tee -a "$OUT"
echo "=== COMPARISON ===" | tee -a "$OUT"
REF=$EQUIV/E1_q8_w1_j8/feature_md5.txt
for tag in E2_q48_w1_j3 E3_q48_w2_j3 E4_q96_w2_j3; do
  f=$EQUIV/$tag/feature_md5.txt
  [ -f "$f" ] || { echo "$tag | MISSING" | tee -a "$OUT"; continue; }
  # The md5 stream is per-frame (id + hash), so a line-by-line diff of the
  # (id, hash) pairs is the strictest comparison: it shows the FIRST frame
  # that differs, not just "the files differ".
  if diff -q "$REF" "$f" >/dev/null 2>&1; then
    echo "$tag | IDENTICAL to E1 ($(wc -l < "$f") frames)" | tee -a "$OUT"
  else
    first=$(diff "$REF" "$f" 2>/dev/null | grep -m1 '^<' | cut -f2)
    echo "$tag | DIFFERS from E1; first differing frame id: ${first:-unknown}" | tee -a "$OUT"
  fi
done

# The whole-stream hash, for the record.
for tag in E1_q8_w1_j8 E2_q48_w1_j3 E3_q48_w2_j3 E4_q96_w2_j3; do
  f=$EQUIV/$tag/feature_md5.txt
  [ -f "$f" ] || continue
  echo "$tag stream_md5=$(md5sum "$f" | cut -d' ' -f1)" | tee -a "$OUT"
done

cp -f "$OUT" "/mnt/srv2/psearch/" 2>/dev/null
echo "EQUIV_DONE" | tee -a "$OUT"
date | tee -a "$OUT"
