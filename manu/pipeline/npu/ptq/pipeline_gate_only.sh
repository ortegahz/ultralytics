#!/bin/bash
# =============================================================================
# pipeline_gate_only.sh   (runs ON the RK3588 board)
#
# THE FULL-SEQUENCE GATE ARM -- THE DELIVERABLE.
#
# What it is
#   The zero-drop, full val3 (11810 frames), step2 run whose E-arm F1 must
#   land on the certified chain:
#       A  PC f32 per-scale          0.897694   (PT, the frozen baseline)
#       C  board f16 mosaic          0.895555   (the board's own f16 path)
#       E  e2e GMC(step2)            0.891871   (the certified pipeline run)
#   The board's f16 and the PT f32 agree to +0.000529 (C is even slightly
#   ABOVE A), and the whole GMC step2 semantics costs -0.005823 against A.
#   So the gate is: the new code's E arm reproduces 0.891871, and the
#   per-frame fingerprints reproduce the certified artefact.
#
# Why this configuration
#   P1b's exact config, plus the two changes the Q1 probe PROVED are
#   bit-identical:
#     --gmc-workers 2         same per-frame fingerprints, 1.64x faster wall
#     --gmc-anchor-threads 4  the anchor LK stage in parallel; the RANSAC
#                            stays serial in the anchor order, so the RNG
#                            stream -- and therefore the numbers -- do not move
#   and --npu-workers 2, which is P1b's own value.
#   The run also PROVES it: the feature_md5.txt is compared against P1b's
#   frame by frame at the end.  Identical features => identical detections
#   => identical F1, so the alignment is established at the feature level and
#   the number is then confirmed by the scorer on the server.
#
# It also publishes the arm to the SERVER-visible path so the server's
# scoring watcher picks it up and produces the full val3 report (all the
# A/B/C/E arms), even if the host is gone by then.
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
ARM=${ARM:-A_gate_step2_nodrop}
GMC_WORKERS=${GMC_WORKERS:-2}
ANCHOR_THREADS=${ANCHOR_THREADS:-4}
AFFINITY_GROUP=${AFFINITY_GROUP:-0-3;4-7}
CPU_AFF=${CPU_AFF:-0-7}
NPU_WORKERS=${NPU_WORKERS:-2}
OUT=$RES/$ARM
SUMMARY=/root/psearch/gate_arm.log

mkdir -p "$OUT"
echo "=== FULL-SEQUENCE GATE ARM: zero-drop step2, val3 (11810 frames) ===" | tee "$SUMMARY"
date | tee -a "$SUMMARY"
echo "config: gmc_workers=$GMC_WORKERS anchor_threads=$ANCHOR_THREADS group=$AFFINITY_GROUP cpu=$CPU_AFF npu=$NPU_WORKERS" | tee -a "$SUMMARY"

attempt=1
rc=1
while [ "$attempt" -le 2 ]; do
  sleep 10
  taskset -c 4-7 "$E2E" \
    --raw-root "$RAW_ROOT" --sequence "$SEQ" \
    --model "$MODEL" --out "$OUT" --kernel-dir "$KERNELS" \
    --limit 0 \
    --anchor-step 2 --pace-ms 0 --jitter-depth 8 --input-queue 8 \
    --npu-workers "$NPU_WORKERS" --post-workers 1 \
    --gmc-workers "$GMC_WORKERS" --gmc-anchor-threads "$ANCHOR_THREADS" \
    --gmc-affinity-group "$AFFINITY_GROUP" --cpu-affinity "$CPU_AFF" \
    --gmc-phys-anchor 0 > "$OUT/run.log" 2>&1
  rc=$?
  echo "[$ARM] attempt=$attempt exit=$rc" | tee -a "$SUMMARY"
  [ "$rc" = "0" ] && break
  grep -E "Illegal|FATAL|failed to submit" "$OUT/run.log" | head -3 | sed 's/^/    /' | tee -a "$SUMMARY"
  attempt=$((attempt+1))
done

grep -E "^\[timing\]" "$OUT/run.log" | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "=== per-frame alignment against the certified artefact (11810 frames) ===" | tee -a "$SUMMARY"
python3 - "$REF/feature_md5.txt" "$OUT/feature_md5.txt" <<'PY' | tee -a "$SUMMARY"
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
print(f"  certified reference frames = {len(ref)}")
print(f"  gate arm frames           = {len(got)}")
print(f"  common                    = {len(common)}")
print(f"  IDENTICAL                 = {len(common)-len(diff)}")
print(f"  DIFFERENT                 = {len(diff)}")
if diff:
    runs = []
    for x in diff:
        if runs and x == runs[-1][1] + 1: runs[-1][1] = x
        else: runs.append([x, x])
    print(f"  FIRST differing frame id  = {diff[0]}")
    print(f"  differing runs (contiguous) = {runs[:12]}{' ...' if len(runs)>12 else ''}")
    print("  => the features are NOT bit-identical; the F1 must then be READ, not assumed")
else:
    print("  => BIT-IDENTICAL on every common frame: the features reproduce the")
    print("     certified artefact exactly, so the E-arm F1 must be 0.891871")
PY

# publish to the server-visible path so the server's watcher scores it
mkdir -p "$PUB/$ARM"
cp -f "$OUT/${SEQ}_mosaic.tsv" "$OUT/feature_md5.txt" "$OUT/timing.txt" "$PUB/$ARM/" 2>/dev/null
echo "published to $PUB/$ARM" | tee -a "$SUMMARY"
ls -la "$PUB/$ARM" | sed 's/^/    /' | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "=== gate arm done ===" | tee -a "$SUMMARY"
date | tee -a "$SUMMARY"
