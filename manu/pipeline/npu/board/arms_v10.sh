#!/usr/bin/env bash
# ===========================================================================
# arms_v10.sh -- the experiment matrix for the THREE-STAGE DECOUPLED pipeline
# (API v10).  Every arm is run SERIALLY on the board: two concurrent e2e
# runs corrupt the RKNN contexts and need a physical power cycle.
#
# THE MATRIX
# ----------
#  CANON14_fused_ref  fused, CANON8's EXACT parameters (old binary's arm).
#                     => the REGRESSION arm: the refactor must be numerically
#                        inert, so this arm must reproduce CANON8 frame for
#                        frame (bit-exact feature md5, same delivered count).
#
#  CANON13_dec        decoupled, USER TOPOLOGY
#                     GMC 4-5 (2 A76) | Pre+Post 6-7 | NPU stage 6-7
#                     external business 0-2 (A55)
#
#  CANON15_fused      fused, the SAME core allocation as CANON13
#                     => isolates the effect of the STAGE SPLIT from the
#                        effect of the core re-allocation.
#
#  CANON16_dec_jit    CANON13 + a bursty CPU burner on ALL cores (0-7)
#  CANON17_fused_jit  CANON15 + the SAME burner
#                     => the CPU-jitter firewall: the NPU must stay busy.
#
# The NPU hardware load (/sys/kernel/debug/rknpu/load) is sampled once a
# second by run_arm.sh into <ARM>.npu_load.log.
#
# Usage:  arms_v10.sh <e2e-binary> [arm ...]
# ===========================================================================
set -uo pipefail

BIN="${1:?e2e binary path required}"
shift
WANT="${*:-CANON14_fused_ref CANON13_dec CANON15_fused CANON16_dec_gmc47 CANON17_dec_jit CANON18_fused_jit}"
RUNNER="/root/psearch/run_arm_v2.sh"

# ---- the shared base --------------------------------------------------------
# Everything the arms have in common.  CANON14_fused_ref keeps CANON8's own
# GMC placement (4-7, 4 anchor threads) on purpose: it is the regression arm.
BASE_COMMON="--raw-root /mnt/manu/lq_ir_bmp --sequence VIDEO00007_19700101_001324
 --model /root/psearch/models/p0_off_A_none.rknn --kernel-dir /root/psearch/kernels
 --limit 0 --input-queue 64 --anchor-step 2 --gmc-phys-anchor 1 --gmc-workers 1
 --jitter-depth 3 --gmc-ring-slack 1 --cpu-affinity 0-7 --npu-workers 2
 --post-workers 1 --pace-ms 40 --gmc-min-stride 1"

# USER TOPOLOGY: GMC 4-5 | Pre+Post 6-7 | NPU stage 6-7 | external 0-2
GMC_USER="--gmc-anchor-threads 2 --gmc-affinity-group 4-5"
PRE_USER="--pre-affinity 6-7 --npu-stage-affinity 6-7"

for ARM in ${WANT}; do
  case "${ARM}" in
    CANON14_fused_ref)
      # CANON8's exact parameters, fused mode, on the NEW binary.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --decoupled-stages 0"
      BURN=none; BURN_SEC=0
      ;;
    CANON13_dec)
      ARGS="${BASE_COMMON} ${GMC_USER} ${PRE_USER}
             --decoupled-stages 1 --buffer-a-depth 3 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON15_fused)
      # the same core allocation as CANON13, but the stages are NOT split
      ARGS="${BASE_COMMON} ${GMC_USER} --npu-affinity 6-7
             --decoupled-stages 0 --buffer-a-depth 3 --buffer-b-depth 3"
      BURN=none; BURN_SEC=0
      ;;
    CANON16_dec_gmc47)
      # The decoupled pipeline WITHOUT starving the GMC: the GMC keeps all
      # four A76 cores (as in CANON8) and the pre/post stages share 6-7 with
      # it.  This is the arm that answers "how fast CAN the decoupled
      # pipeline go when the GMC is not the bottleneck" -- i.e. it exposes the
      # NPU stage's own ceiling.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 6-7 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 3 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON17_dec_jit)
      ARGS="${BASE_COMMON} ${GMC_USER} ${PRE_USER}
             --decoupled-stages 1 --buffer-a-depth 3 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=0-7; BURN_SEC=1400
      ;;
    CANON18_fused_jit)
      ARGS="${BASE_COMMON} ${GMC_USER} --npu-affinity 6-7
             --decoupled-stages 0 --buffer-a-depth 3 --buffer-b-depth 3"
      BURN=0-7; BURN_SEC=1400
      ;;
    CANON19_dec_jit67)
      # The PRECISE firewall arm: the burner takes ONLY the A76 cores the three
      # stages live on (6-7).  The GMC keeps its own cores (4-5) and stays at
      # full speed, so the ONLY thing that changes is how much CPU the
      # pre/post work gets.  This is the scenario the user described: an
      # external business (RTSP decode / radar fusion / gimbal comms) taking
      # the A76s away from us.
      ARGS="${BASE_COMMON} ${GMC_USER} ${PRE_USER}
             --decoupled-stages 1 --buffer-a-depth 3 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=6-7; BURN_SEC=1400
      ;;
    CANON20_fused_jit67)
      # The control for CANON19: the SAME cores taken away, but the stages are
      # NOT split, so the upload sits on the NPU thread and its stall is the
      # NPU's stall.
      ARGS="${BASE_COMMON} ${GMC_USER} --npu-affinity 6-7
             --decoupled-stages 0 --buffer-a-depth 3 --buffer-b-depth 3"
      BURN=6-7; BURN_SEC=1400
      ;;
    CANON22_dec_bufa16)
      # CANON21's topology, but Buffer A is DEEPER (16 slots instead of 3).
      # WHY 16: the GMC at full tilt produces 16.1 groups/s and the NPU stage
      # can consume 16.3 fps -- they are neck and neck, so ANY jitter in
      # either side turns into a growing backlog.  The GMC's fit tail (p95
      # ~200 ms, max ~700 ms) is followed by a catch-up burst of ~10+ frames,
      # and a 3-slot ring cannot hold that: CANON21 dropped 1567 of 7195
      # groups (21.8%) while the NPU stage was only 73% busy.
      # 16 x 2,457,600 B = 37.7 MB of DMA per NPU service (75 MB total).
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 16 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON21_dec_a55)
      # THE PRESCRIPTION ARM.  The user's own rule is that the pre/post work
      # only has to stay UNDER the NPU's per-frame time -- and it does, by
      # 10~23x.  So they do NOT need A76 cores: they go DOWN to the small
      # cores, and the four A76s go back to the GMC (which needs ~3 of them
      # for its 4-worker anchor fit).
      #   GMC 4-7 (4 anchor workers) | Pre+Post 0-3 (A55) | Stage 2 6-7
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 3 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON23_dec_a16)
      # CANON21's topology, EXACTLY as it is, with ONE variable changed: the
      # Buffer A depth.  CANON22 asked for 16 but the clamp table's ceiling was
      # 8, and an out-of-range request falls back to the reference default
      # (3) -- so CANON22 was a depth-3 repeat of CANON21 and the deep ring
      # has NEVER actually run.  The ceiling is 64 now, so 16 is what the
      # pipeline really gets.
      # 16 x 2,457,600 B x 2 services = 78.7 MB of DMA on the board.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 16 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON24_dec_a32)
      # CANON23 with 32 slots: 32 x 2,457,600 B x 2 services = 157 MB of DMA.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 32 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON25_dec_a64)
      # CANON23 with 64 slots (the knob's ceiling): 64 x 2,457,600 B x 2
      # services = 315 MB of DMA.  If the drops are gone at 16, this arm says
      # how much of the ring is actually needed.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 64 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON26_dec_xsvc_d3)
      # CROSS-SERVICE DELIVERY ("whoever has room takes the frame"), depth 3.
      # CANON21's topology, byte for byte, with ONE change: the Stage 1 thread
      # no longer feeds its own paired NPU service -- it feeds whichever
      # service's ring has room.  The 1:1 pairing was the cause of the 1.63:1
      # imbalance (per-service frames_run 3616 vs 2223), and the imbalance is
      # where 100% of the Buffer A losses came from.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 3 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON27_dec_xsvc_d8)
      # Cross-service delivery AND a ring deep enough to absorb the GMC burst.
      # Depth 8 x 2,457,600 B x 2 services = 39.3 MB of DMA.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 8 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON28_dec_rr_d3)
      # ROUND-ROBIN delivery: the frame goes to the service the cursor points
      # at (scanning forward until one has room).  CANON26 proved that the
      # "shortest READY queue" rule has a SYSTEMATIC bias: the fast service's
      # queue is empty most of the time, so the tie-break (lower index) kept
      # handing every frame to service 0 -- the imbalance only moved from
      # 1.63:1 to 1.52:1.  The round-robin cursor removes the bias itself.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 3 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON29_dec_rr_d8)
      # Round-robin delivery AND a deeper ring (8 slots x 2,457,600 B x 2
      # services = 39.3 MB of DMA) to absorb the GMC's burst.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 8 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON30_dec_rr_d16)
      # Round-robin delivery AND a 16-slot ring (16 x 2,457,600 B x 2 services
      # = 78.7 MB of DMA).  CANON29 (D=8) measured 850 reclaims, ALL of them on
      # service 0, while service 1 sat idle 44% of the time: the two rings were
      # NOT both full, so the loss came from the ring that the round-robin kept
      # feeding.  A deeper ring holds the GMC's catch-up burst, which is the
      # only loss channel that a deeper buffer can absorb.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 16 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON31_dec_rr_d32)
      # CANON30 with 32 slots (32 x 2,457,600 B x 2 services = 157 MB of DMA).
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 32 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON32_dec_wfq_d8)
      # CAPACITY-WEIGHTED delivery (v11) with the D=8 ring: the frame goes to
      # the service whose MEASURED capacity (frames_run / stage2_busy_ms) is
      # the highest per unit of load already put on it.  CANON29 proved that
      # balancing the FRAMES (1.000:1) is the wrong invariant -- the slow
      # service still overflowed its ring (all 850 reclaims) while the fast
      # one idled 44% of the time.  Balancing the CAPACITY is the right one.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 8 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON36_dec_rr_d64)
      # Round-robin + the knob's CEILING depth (64 slots x 2,457,600 B x 2
      # services = 314.6 MB of DMA).  The measured depth ladder is
      #   D=3 -> 12.880 fps (1039 drops)
      #   D=8 -> 13.245 fps ( 850 drops)
      #   D=16-> 13.543 fps ( 713 drops)
      #   D=32-> 13.933 fps ( 486 drops)
      # i.e. every doubling of the ring buys ~2.4% fps and takes away latency
      # (p95 481 -> 1097 -> 2098 -> 4116 ms).  D=64 is the last rung.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 64 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON37_dec_rr_d64_npu_a55)
      # D=64 AND the NPU stage moved onto the A55s (0-3), so the GMC's four
      # A76 anchor workers (cores 4-7) stop sharing cores 6-7 with Stage 2.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 0-3
             --decoupled-stages 1 --buffer-a-depth 64 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON34_dec_rr_d16_npu_a55)
      # ROUND-ROBIN + D=16, with the NPU stage MOVED OFF the A76 cores.
      #
      # WHY: the GMC's group period (66.4 ms) is a SINGLE-ENGINE SERIAL chain
      # -- prepare_fit(21 anchors) -> LK(21 anchors on 4 workers = 6 rounds,
      # ~39.7 ms) -> RANSAC(21, serial by design because it consumes the
      # global RNG in anchor order) -> GPU tail (~8.9 ms) -> post (MD5
      # fingerprint 5-7 ms + NEON interleave + enqueue).  The 3 anchor helper
      # threads are IDLE for ~30 ms of every cycle.  AND the GMC's anchor
      # workers live on cores 4-7 while Stage 2 (NPU) is pinned to 6-7 --
      # i.e. TWO of the GMC's four workers share a core with a Stage 2 thread.
      # Moving Stage 2 onto the A55s (0-3, where Pre/Post already sit) gives
      # the GMC its four A76s without contention.
      #
      # THIS CHANGES NO FEATURE BYTES: it only re-places threads.  The GMC's
      # anchor set, the lag set and the RANSAC order are untouched.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 0-3
             --decoupled-stages 1 --buffer-a-depth 16 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    CANON35_dec_rr_d16_nofp)
      # ROUND-ROBIN + D=16 with the GMC's per-frame feature MD5 fingerprint
      # DISABLED.  The fingerprint costs 5-7 ms of CPU per group (~10% of the
      # GMC's 66.4 ms group period) and it only READS the output, so the
      # feature bytes are unchanged.  THIS ARM IS FOR THE CEILING MEASUREMENT
      # ONLY: with the fingerprint off there is no feature_md5.txt, so the
      # bit-exact accuracy check CANNOT be run on it.
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 16 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2 --gmc-feature-fingerprint 0"
      BURN=none; BURN_SEC=0
      ;;
    CANON33_dec_wfq_d16)
      # CANON32 with a 16-slot ring (16 x 2,457,600 B x 2 services = 78.7 MB).
      ARGS="${BASE_COMMON} --gmc-anchor-threads 4 --gmc-affinity-group 4-7
             --pre-affinity 0-3 --npu-stage-affinity 6-7
             --decoupled-stages 1 --buffer-a-depth 16 --buffer-b-depth 3
             --buf-a-mode dma --pre-threads 2"
      BURN=none; BURN_SEC=0
      ;;
    *)
      echo "[arms] unknown arm ${ARM}" >&2; exit 2 ;;
  esac

  echo "=============================================================="
  echo "[arms] ${ARM}  (burn=${BURN} ${BURN_SEC}s)"
  echo "[arms] ${BIN} ${ARGS}"
  echo "=============================================================="
  # shellcheck disable=SC2086
  E2E_BIN="${BIN}" "${RUNNER}" "${ARM}" "${BURN}" "${BURN_SEC}" -- ${ARGS}
  echo "[arms] ${ARM} done (rc=$?)"
done
