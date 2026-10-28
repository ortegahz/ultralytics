#!/bin/bash
# =============================================================================
# deploy_pipeline_search.sh
#
# Deploys the DECOUPLED-PIPELINE search driver to the RK3588 board and starts
# it detached.  The driver survives the HOST going down: its data source is the
# SERVER's /mnt/srv2 (SSHFS to huangzhe@192.168.99.40), NOT the host's NFS
# export at /mnt/manu (which dies with the host).
#
# WHAT IS BEING TESTED
# --------------------
# The decoupled three-stage architecture:
#   [camera] -> Raw Image RingBuffer -> [GMC] -> Feature Queue -> [NPU] -> [POST]
#
# The claim under test: deepening the camera-side buffer absorbs the GMC
# fit-cost variance, so the camera NEVER drops a frame even though the GMC
# stage is slower than 25 fps on average.
#
# The hard acceptance criterion:  dropped == 0 over the whole sequence.
#   dropped    = frames that never reached the GMC  -> breaks the registration
#                chain.  MUST be 0.
#   features_dropped = frames the INFERENCE stage skipped.  Allowed by design.
#
# The controls sweep the two buffer depths and the GMC worker count.  All of
# them are EXISTING CLI knobs (no code change):
#   --input-queue N     camera-side Raw Image RingBuffer depth
#   --jitter-depth N    GMC -> NPU Feature Queue depth
#   --gmc-workers N     GMC engines
#   --gmc-ring-slack N  how far the ingest may run ahead of the engines
#
# WHY THE DEEP BUFFER MAY NOT BE ENOUGH (the script measures this, not assumes):
# a deeper buffer absorbs SPIKES; it cannot absorb a sustained RATE deficit.
# If GMC's steady-state throughput < the camera rate, the queue grows without
# bound and eventually overflows anyway.  The driver therefore ALSO reports the
# steady-state GMC throughput and the ingest wait breakdown, so the two
# regimes (spike-limited vs rate-limited) can be told apart from the data.
#
# RESULTS
#   board local : /root/psearch/results/<arm>/
#   server-side : /mnt/srv2/psearch/results/<arm>/   (visible from the server)
# =============================================================================
set -u

BOARD=root@192.168.0.64
BOARD_PORT=22
REMOTE=/root/psearch
# The SERVER's copy of the raw IR.  Immune to the host going down.
RAW_ROOT=/mnt/srv2/lq_ir_bmp
SEQ=VIDEO00007_19700101_001324
MODEL=/mnt/manu/ptq/models/p0_off_A_none.rknn

echo "[deploy] staging the driver to $BOARD"
ssh -p $BOARD_PORT "$BOARD" "mkdir -p $REMOTE" || exit 1

# --- the driver itself -------------------------------------------------------
scp -P $BOARD_PORT "$(dirname "$0")/pipeline_search_driver.sh" "$BOARD:$REMOTE/" || exit 1
scp -P $BOARD_PORT "$(dirname "$0")/pipeline_search_driver.sh" "$BOARD:$REMOTE/" || exit 1
ssh -p $BOARD_PORT "$BOARD" "chmod +x $REMOTE/pipeline_search_driver.sh" || exit 1

# --- start it detached -------------------------------------------------------
# setsid + nohup so it survives the SSH session AND the host's shutdown.
ssh -p $BOARD_PORT "$BOARD" \
  "setsid nohup $REMOTE/pipeline_search_driver.sh > $REMOTE/driver.log 2>&1 < /dev/null & sleep 2; echo '[deploy] driver launched'; head -5 $REMOTE/driver.log"

echo "[deploy] done.  Watch: ssh -p $BOARD_PORT $BOARD 'tail -f $REMOTE/driver.log'"
echo "[deploy] results land in $REMOTE/results/ and /mnt/srv2/psearch/results/"
