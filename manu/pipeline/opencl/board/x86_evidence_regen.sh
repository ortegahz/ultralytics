#!/usr/bin/env bash
# ===========================================================================
# x86_evidence_regen.sh -- regenerate the handle census and the timing ledger on
# x86, with every intermediate artefact LEFT ON DISK.
#
# Why this script exists: the numbers in the report were produced by earlier
# one-off invocations whose .npy dumps and per-sequence logs were cleaned up
# afterwards. `.bench_out/` currently contains ZERO `cornercap` lines, so the
# trigger census in the report cannot be re-derived from anything on disk. A
# number that cannot be reproduced from a retained artefact is a number nobody
# can audit. This regenerates them, keeps them, and prints the provenance.
#
# It is NOT a substitute for the board run. x86 latency is not RK3588 latency.
# What x86 can settle authoritatively: the corner BUDGET's effect on output
# (CPU arm, bit-level, architecture-independent) and the handle TRIGGER COUNTS
# (a pure Shi-Tomasi count, shown SIMD-insensitive in simd_corner_portability).
# What only the board can settle: latency. See RK3588_PREPROCESS_BENCH_REPORT.md
# section 3 for the board baseline and section 7 for what stays unverified.
#
# Usage: bash x86_evidence_regen.sh [census|timing|both]
# ===========================================================================
set -u
HERE=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board
BIN=/tmp/gmc_stream_ocl_cap
B=$HERE/.bench_bmp
K=$HERE/../kernels
OUT=$HERE/.bench_out/REGEN${REGEN_TAG_SUFFIX:-}
STAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUT"
{
  echo "### x86_evidence_regen stamp=$STAMP"
  echo "### binary=$BIN md5=$(md5sum "$BIN" | cut -d' ' -f1)"
  echo "### host=$(uname -srm) nproc=$(nproc)"
  echo "### corpus=$(find "$B" -name '*.bmp' | wc -l) bmp"
  echo "### kernels=$K"
} | tee "$OUT/PROVENANCE.txt"

SEQS="VIDEO00034_19700101_015052
      VIDEO00028_19700101_013631
      VIDEO00032_19700101_014453
      VIDEO00002_19700101_001415
      VIDEO00036_19700101_015740"

WARMUP=100
LIMIT=400

# One process per sequence, same as bench_run.sh. On x86 the reason is OOM
# hygiene of the OpenCL device pool rather than the board's 8 GiB ceiling, but
# the protocol is kept identical so the two ledgers are comparable.
run_seq() {
  tag=$1; s=$2; shift 2
  d=$OUT/$tag
  mkdir -p "$d"
  "$BIN" --raw-root "$B" --sequence "$s" --limit $LIMIT --fused gpu \
         --pyr-cache --threads 4 --affinity a76 --hw-linear 0 \
         --kernel-dir "$K" \
         --profile-csv "$d/profile_${tag}_${s}.csv" --profile-warmup $WARMUP \
         --md5-out "$d/md5_${tag}.txt" --timing "$@" \
         > "$d/$s.log" 2>&1
  rc=$?
  echo "### $tag $s rc=$rc rows=$(($(wc -l < "$d/profile_${tag}_${s}.csv" 2>/dev/null || echo 1) - 1)) flags=$*"
  return $rc
}

MODE=${1:-both}

if [ "$MODE" = "census" ] || [ "$MODE" = "both" ]; then
  # ---- CENSUS -------------------------------------------------------------
  # UNC: --fit-corners 5000, i.e. far above any corner count in this corpus, so
  # Shi-Tomasi returns everything it found. The `corners mean=` on this arm is
  # the TRUE per-frame corner count. It is what the budget-200 numbers have to be
  # read against -- a clipped mean sitting just under 600 looks like "pressing
  # against the ceiling", which is exactly the misreading section 5A.2 corrects.
  # UNC also records cornercap=0f/... which is the control proving the probe
  # reports zero when nothing is truncated.
  echo "=== CENSUS: unclipped (fit-corners 5000) ==="
  for s in $SEQS; do run_seq UNC "$s" --fit-corners 5000 --corner-probe 1; done

  # B200: the delivery budget, applied to ALL five sequences on purpose. The
  # shipped policy enables it for VIDEO00036 only (section 5A.4), but the census
  # and the accuracy cost have to be measured on all five -- a policy justified
  # by "we did not measure the others" is not a justification.
  echo "=== CENSUS: budget 200 on all five ==="
  for s in $SEQS; do run_seq B200 "$s" --fit-corners 200 --corner-probe 1; done
fi

if [ "$MODE" = "timing" ] || [ "$MODE" = "both" ]; then
  # ---- TIMING -------------------------------------------------------------
  # A: baseline as delivered before this change set. No prealloc, default
  #    corners. This is the arm every "did it get slower" question is against.
  echo "=== TIMING arm A: baseline ==="
  for s in $SEQS; do run_seq A "$s" --corner-probe 1; done

  # B: the delivered configuration, except that the corner budget is applied to
  #    all five rather than only to VIDEO00036. Running it wide makes the
  #    per-sequence decision falsifiable: if a sequence's B-vs-A delta is inside
  #    noise and its accuracy cost is nonzero, not shipping the budget there is
  #    right for a measurable reason instead of a remembered one.
  #    --prealloc-pad 1024 == kMaxPadDevice, so no runtime reallocation can occur.
  echo "=== TIMING arm B: prealloc 1024 + fit-corners 200 ==="
  for s in $SEQS; do run_seq B "$s" --prealloc-pad 1024 --fit-corners 200 --corner-probe 1; done
fi

echo
echo "=== [SEQ] lines (the handle census itself) ==="
grep -h "^\[SEQ\]" "$OUT"/UNC/*.log "$OUT"/B200/*.log 2>/dev/null | sed 's/  */ /g'
echo
echo "artefacts kept under $OUT"