#!/bin/sh
# ===========================================================================
# board_experiments.sh -- the board experiment plan, in the order the Leader
# asked for.  EVERY RUN IS SEQUENTIAL: the board tolerates exactly two RKNN
# contexts, and two overlapping e2e runs take the NPU down (that is what
# wedged the pipeline on 2026-10-27 12:36).
#
# Run it on the board:  sh /root/board_experiments.sh
# Every run writes its own timing file, so every number in the report can be
# traced to one file.
# ===========================================================================
cd /root || exit 1
SEQ=VIDEO00007_19700101_001324
MODEL=/mnt/manu/ptq/models/p0_off_A_none.rknn
RAW=/mnt/manu/lq_ir_bmp
OUT=/mnt/manu/mosaic_out/svc
BASE="--raw-root $RAW --sequence $SEQ --model $MODEL --kernel-dir /root/kernels --anchor-step 10 --gmc-workers 2 --npu-workers 2 --post-workers 1 --jitter-depth 8 --gmc-fit-corners 600 --cpu-affinity 4-7"

# ---- F3: the board must be FREE before the batch starts --------------------
N=$(ps -eo comm | grep -c "^rknn_")
echo "### rknn_ processes on the board before the batch: $N"
if [ "$N" != "0" ]; then
  echo "### ABORT: the board is busy ($N rknn_ processes).  Wait, do not overlap."
  exit 1
fi

run() {
  tag=$1; shift
  mkdir -p "$OUT/$tag"
  echo "##### RUN $tag : $* #####"
  ./rknn_mosaic_e2e $BASE --out "$OUT/$tag" --timing "$OUT/$tag/timing.txt" "$@" \
      > "$OUT/$tag.log" 2>&1
  rc=$?
  echo "### $tag EXIT CODE: $rc"
  grep -E "^\[timing\]|^\[i\] TSV|^\[i\] feature md5|^\[FATAL\]|^\[WARN\]|core mask" \
      "$OUT/$tag.log" | tail -30
  echo "##### END $tag #####"
}

# ===========================================================================
# 2) THE FAULT INJECTION (F1 + F2)
# ===========================================================================
# (a) the DETERMINISTIC one: the library's own FATAL-error path.  Before the
#     fix this set the abort flag but woke nobody, so every thread stayed
#     asleep and the demo hung for ever.
run svc_F_faultinj --fault-inject-at 20 --watchdog-ms 20000
# (b) the WATCHDOG itself: a 1 ms budget must trip and report the counters.
run svc_F_watchdog --watchdog-ms 1 --limit 5

# ===========================================================================
# 3) THE CORE MASK TABLE
#    1 context and 2 contexts, each with CORE_0 / CORE_0_1 / CORE_0_1_2 and
#    the shipped default (0 = do not call rknn_set_core_mask).
#    400 frames each; the timing file carries rknn_run's mean/p50/p95/p99.
# ===========================================================================
for W in 1 2; do
  for M in 0 1 3 7; do
    run "svc_M_c${W}_m${M}" --npu-workers $W --npu-core-mask $M --limit 400
  done
done
echo "##### CORE MASK TABLE DONE #####"

# ===========================================================================
# 4) THE UPLOAD / POST A/B
# ===========================================================================
# (a) the upload path.  set is the CERTIFIED one; iomem_nhwc binds the input
#     tensor to host memory, so the runtime's staging moves inside rknn_run.
#     setf16 was MEASURED slower (the runtime copies 2.45 MB at ~60-75 MB/s).
run svc_U_set      --input-mode set      --limit 400
run svc_U_iomemnhwc --input-mode iomem_nhwc --limit 400
run svc_U_setf16   --input-mode setf16   --limit 400
# (b) the POST parallelism.  The POST stage is frame-independent: every
#     mutable buffer is the per-worker PostWorkspace, and the Session is only
#     read.  So two workers must give the SAME numbers.
run svc_Q_post1 --post-workers 1 --limit 400
run svc_Q_post2 --post-workers 2 --limit 400
echo "##### UPLOAD / POST A/B DONE #####"

# ===========================================================================
# 5) THE FULL BIT-EXACTNESS ARM (the acceptance criterion)
#    The TSV md5 must be 0bf598a1d890040bcfa8d157dc69c032 and the per-frame
#    feature fingerprints must be identical to the reference.
# ===========================================================================
run svc_S_parfull --gmc-anchor-threads 5 --gmc-affinity-group "4-5;6-7"

echo "##### THE MD5 RECONCILIATION #####"
md5sum "$OUT"/svc_S_parfull/*_mosaic.tsv
md5sum "$OUT"/svc_S_parfull/feature_md5.txt
echo "--- the certified reference ---"
md5sum /mnt/manu/mosaic_out/svc/svc_I_bitexact_opt/*_mosaic.tsv
md5sum /mnt/manu/mosaic_out/svc/svc_I_bitexact_opt/feature_md5.txt
echo "--- the A/B arms (must be identical to each other) ---"
md5sum "$OUT"/svc_U_set/*_mosaic.tsv "$OUT"/svc_U_iomemnhwc/*_mosaic.tsv \
       "$OUT"/svc_U_setf16/*_mosaic.tsv "$OUT"/svc_Q_post1/*_mosaic.tsv \
       "$OUT"/svc_Q_post2/*_mosaic.tsv
echo "##### ALL DONE #####"
