#!/bin/bash
# =============================================================================
# pipeline_bisect_ref_diff.sh   (runs ON the RK3588 board)
#
# WHY
# ---
# The regression probe found that all three new-binary arms differ from the
# CERTIFIED zero-drop reference (P1b) starting at frame id 60, while the
# certified-path arm (the OLD binary) matched the reference on the 55 frames
# it managed to produce before an RKNPU submit failure.
#
# The reference's own timing.txt shows the difference:
#     P1b (reference) : npu_workers=2  input_queue=8  jitter_depth=8  gmc_workers=1
#     the probe arms  : npu_workers=1  input_queue=64 jitter_depth=8  gmc_workers=1
# So the arms and the reference differ in TWO knobs, and a 600-frame prefix is
# not the same experiment as the reference's full 11810-frame run.
#
# The bisection below isolates it.  Every arm is 600 frames of the same
# zero-drop step2 configuration, and every arm is compared to the reference's
# first 600 frames:
#
#   Q1  OLD certified binary, the PROBE's config (npu1, in_q64)
#       -> if it also differs at frame 60, the BINARY is innocent: the config
#          is the cause.
#   Q2  NEW binary, the REFERENCE's exact config (npu2, in_q8)
#       -> if it matches, the config is confirmed as the cause.
#   Q3  NEW binary, npu_workers=2, in_q64   -> isolates input_queue
#   Q4  NEW binary, npu_workers=1, in_q8    -> isolates npu_workers
#   Q5  NEW binary, npu_workers=2, in_q8, gmc_workers=2
#       -> the reference config PLUS a second GMC engine: does the multi-engine
#          split stay bit-identical when the reference's own knobs are used?
#
# An RKNN "failed to submit" is retried once after a settle pause (the driver
# occasionally refuses the first job right after a context was released).
# =============================================================================
set -u

RES=/root/psearch/results
REF=/mnt/srv2/psearch/results/P1b_step2_nodrop_orig/feature_md5.txt
RAW_ROOT=${RAW_ROOT:-/mnt/srv2/lq_ir_bmp}
SEQ=VIDEO00007_19700101_001324
MODEL=${MODEL:-/root/psearch/models/p0_off_A_none.rknn}
KERNELS=${KERNELS:-/root/psearch/kernels}
LIMIT=${LIMIT:-600}
OUT=/root/psearch/bisect
OLD=/root/rknn_mosaic_e2e
NEW=/root/rknn_mosaic_e2e_sweep3
SUMMARY=$OUT/SUMMARY.txt

mkdir -p "$OUT"
echo "=== bisect: why do the new arms differ from the certified reference at frame 60? ===" | tee "$SUMMARY"
date | tee -a "$SUMMARY"

run_arm () {
  local arm=$1; local bin=$2; shift 2
  local out=$RES/$arm
  mkdir -p "$out"
  echo "" | tee -a "$SUMMARY"
  echo "--- $arm : $bin $* ---" | tee -a "$SUMMARY"
  date | tee -a "$SUMMARY"
  local attempt rc
  for attempt in 1 2; do
    sleep 8
    taskset -c 4-7 "$bin" \
      --raw-root "$RAW_ROOT" --sequence "$SEQ" \
      --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
      --cpu-affinity 4-7 --post-workers 1 \
      --anchor-step 2 --pace-ms 0 --jitter-depth 8 \
      --limit "$LIMIT" "$@" > "$out/run.log" 2>&1
    rc=$?
    echo "[$arm] attempt=$attempt exit=$rc" | tee -a "$SUMMARY"
    [ "$rc" = "0" ] && break
  done
  grep -E "^\[timing\]" "$out/run.log" | tail -4 | tee -a "$SUMMARY"
}

run_arm Q1_old_probe_cfg   "$OLD" --npu-workers 1 --input-queue 64 --gmc-workers 1
run_arm Q2_new_ref_cfg     "$NEW" --npu-workers 2 --input-queue 8  --gmc-workers 1
run_arm Q3_new_npu2_q64    "$NEW" --npu-workers 2 --input-queue 64 --gmc-workers 1
run_arm Q4_new_npu1_q8     "$NEW" --npu-workers 1 --input-queue 8  --gmc-workers 1
run_arm Q5_new_ref_cfg_w2  "$NEW" --npu-workers 2 --input-queue 8  --gmc-workers 2

echo "" | tee -a "$SUMMARY"
echo "=== comparison against the certified reference ===" | tee -a "$SUMMARY"
python3 - "$REF" "$RES/Q1_old_probe_cfg" "$RES/Q2_new_ref_cfg" "$RES/Q3_new_npu2_q64" \
         "$RES/Q4_new_npu1_q8" "$RES/Q5_new_ref_cfg_w2" <<'PY' | tee -a "$SUMMARY"
import sys, os

def load(path):
    d = {}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                try:
                    d[int(parts[1])] = parts[2]
                except ValueError:
                    pass
    return d

ref = load(sys.argv[1])
print(f"  reference frames = {len(ref)}")
arms = {}
for arm_dir in sys.argv[2:]:
    name = os.path.basename(arm_dir)
    got = load(os.path.join(arm_dir, "feature_md5.txt"))
    if got is None:
        print(f"  {name:22s} NO feature_md5.txt")
        continue
    arms[name] = got
    common = sorted(set(ref) & set(got))
    diff = [k for k in common if ref[k] != got[k]]
    print(f"  {name:22s} frames={len(got):4d} common={len(common):4d} "
          f"identical={len(common)-len(diff):4d} DIFFERENT={len(diff):4d}"
          + (f"  first_diff={diff[0]}" if diff else "  <-- MATCHES the reference"))
# pairwise: the arms against EACH OTHER, which is the question that matters for
# "does the anchoring / the engine count change the numbers?"
names = list(arms)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = arms[names[i]], arms[names[j]]
        common = sorted(set(a) & set(b))
        diff = [k for k in common if a[k] != b[k]]
        print(f"  pairwise {names[i]} vs {names[j]}: common={len(common):4d} DIFFERENT={len(diff):4d}"
              + (f"  first_diff={diff[0]}" if diff else "  <-- identical"))
PY

echo "" | tee -a "$SUMMARY"
echo "=== bisect done ===" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
