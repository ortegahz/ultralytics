#!/usr/bin/env bash
# corner_config_sensitivity.sh -- the trigger census is CONFIGURATION dependent,
# and the two existing experiments in the report used different configurations.
#
# The observed contradiction:
#   - report census      : VIDEO00032 cornercap = 306/400 frames
#   - regenerated census : VIDEO00032 cornercap = 170/400 frames
#   - BUT the accuracy table reproduced to five decimal places (Ch1 max 68,
#     84.7318% exact), and output at a given budget cannot be identical if a
#     different number of frames had been truncated.
#
# The two runs differed in exactly one flag. corner_budget_full.sh (accuracy)
#   --threads 4 --kernel-dir K --fit-corners N --dump-dir D        <- no --pyr-cache
# x86_evidence_regen.sh (census)
#   --pyr-cache --threads 4 --affinity a76 --hw-linear 0 ...      <- with --pyr-cache
#
# --pyr-cache reuses the previous frame's Shi-Tomasi corners, so a frame can be
# fitted from cached corners that would not have been re-detected. That changes
# WHICH frames exceed the budget, and it changes the unclipped mean (221.3 ->
# 229.6 for VIDEO00032) without changing the accuracy experiment, which never
# enabled the cache.
#
# This matters commercially, not just scientifically: bench_run.sh, the board
# harness, DOES pass --pyr-cache. So the board census will follow the pyr-cache
# column, and quoting 306 as the expected board count would be wrong.
set -u
HERE=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board
cd "$HERE"
B=$HERE/.bench_bmp
K=$HERE/../kernels
OUT=$HERE/.bench_out/REGEN/CFGSENS
mkdir -p "$OUT"

# VIDEO00002 is deliberately excluded: at budget 200 it truncates on 398/398
# detect calls either way, so it cannot discriminate between the two
# configurations and would only add four runs of wall time.
SEQS="VIDEO00032_19700101_014453 VIDEO00036_19700101_015740"

one() {
  tag=$1; s=$2; budget=$3; shift 3
  /tmp/gmc_stream_ocl_cap --raw-root "$B" --sequence "$s" --limit 400 --fused gpu \
     --threads 4 --affinity a76 --hw-linear 0 --kernel-dir "$K" \
     --fit-corners $budget --corner-probe 1 "$@" > "$OUT/$tag-$s.log" 2>&1
  L=$(grep -h '^\[SEQ\]' "$OUT/$tag-$s.log")
  cap=$(echo "$L" | grep -o 'cornercap=[0-9]*f/[0-9]*calls([0-9]*pts)' | cut -d= -f2)
  mean=$(echo "$L" | grep -o 'corners mean=[0-9.]*' | cut -d= -f2)
  dets=$(echo "$L" | grep -o 'over [0-9]* dets' | cut -d' ' -f2)
  md5=$(echo "$L" | grep -o 'md5=[0-9a-f]*' | cut -d= -f2)
  printf '%-26s %-12s budget=%-5s cornercap=%-22s mean=%-8s dets=%-5s md5=%s\n' \
     "$tag" "${s%%_*}" "$budget" "${cap:-ERR}" "${mean:-ERR}" "${dets:-ERR}" "${md5:-ERR}"
}

echo "=== trigger census: --pyr-cache ON vs OFF (x86 GPU arm, 400 frames) ==="
echo
for s in $SEQS; do
  one "budget5000_pyrON"  "$s" 5000 --pyr-cache
  one "budget5000_pyrOFF" "$s" 5000
  one "budget200_pyrON"   "$s" 200  --pyr-cache
  one "budget200_pyrOFF"  "$s" 200
  echo
done
echo "artefacts: $OUT"