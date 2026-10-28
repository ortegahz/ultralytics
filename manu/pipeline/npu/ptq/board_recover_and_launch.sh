#!/bin/bash
# =============================================================================
# board_recover_and_launch.sh   (runs on the HOST)
#
# Waits for the RK3588 board to come back after a power cycle, checks that the
# environment it needs is intact, deploys the consolidated architecture run and
# launches it detached on the board.
#
# WHY A WATCHER AND NOT A MANUAL STEP
# ------------------------------------
# The board lost the NPU driver (two RKNPU job timeouts -> soft reset ->
# SIGILL in rknn_init) and had to be power-cycled.  Everything the run needs
# lives on the board's own disk (/root/psearch/models, /root/psearch/kernels,
# the sweep3 binaries), so the recovery is: wait, verify, deploy the script,
# launch.
#
# THE CHECKS, IN ORDER, AND WHY EACH ONE IS A HARD GATE
#   1. ssh works
#   2. the raw data source is reachable.  The board reads the sequence from
#      /mnt/srv2/lq_ir_bmp, which is the SERVER's ptq_work over SSHFS.  If the
#      SSHFS mount did not come back after the reboot, the run has no input
#      and would fail 90 s in, so the mount is re-established (or the run is
#      reported as blocked).
#   3. the model and the kernels are on the board's local disk
#   4. the sweep3 binaries are present with the expected md5
#   5. an NPU HEALTH CHECK: a 3-frame run.  If the driver is still in the bad
#      state the health check dies with SIGILL and the long run is NOT
#      launched -- a 40-minute run that dies at minute 1 is worse than no run.
# =============================================================================
set -u

BOARD=root@192.168.0.64
PORT=22
LOCAL=/media/manu/1TB-Volume/workspace/ultralytics
SCRIPT=$LOCAL/manu/pipeline/npu/ptq/pipeline_arch_run.sh
LOG=/tmp/board_recover.log

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for $BOARD"
while true; do
  if timeout 12 ssh -p $PORT -o ConnectTimeout=6 -o BatchMode=yes \
       -o StrictHostKeyChecking=no $BOARD 'echo UP' >/dev/null 2>&1; then
    say "board is up"
    break
  fi
  sleep 15
done

# ---- 2/3/4: environment -----------------------------------------------------
say "checking the environment"
ssh -p $PORT $BOARD '
  set -u
  echo "--- uptime ---"; uptime
  echo "--- srv2 mount ---"; mountpoint -q /mnt/srv2 && echo "MOUNTED" || echo "NOT MOUNTED"
  echo "--- raw data ---"; ls /mnt/srv2/lq_ir_bmp 2>/dev/null | head -3
  echo "--- model ---"; ls -la /root/psearch/models/ 2>/dev/null
  echo "--- kernels ---"; ls /root/psearch/kernels/ 2>/dev/null | head -4
  echo "--- binaries ---"; md5sum /root/rknn_mosaic_e2e_sweep3 /root/libmosaic_pipeline_sweep3.a /root/mosaic_api_test_sweep3 2>/dev/null
  echo "--- results kept ---"; ls /root/psearch/results/ 2>/dev/null
' 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"

# The raw data: if the SSHFS mount is down, try to bring it back.
NEED_MOUNT=$(ssh -p $PORT $BOARD 'mountpoint -q /mnt/srv2 && echo no || echo yes' 2>/dev/null | tail -1)
if [ "$NEED_MOUNT" = "yes" ]; then
  say "the /mnt/srv2 SSHFS mount is down -- trying to re-establish it"
  ssh -p $PORT $BOARD '
    set -u
    grep -n srv2 /etc/fstab 2>/dev/null && { mount /mnt/srv2 2>&1 || true; }
    mountpoint -q /mnt/srv2 && echo "MOUNTED NOW" || echo "STILL NOT MOUNTED"
    ls /mnt/srv2/lq_ir_bmp 2>/dev/null | head -3
  ' 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"
fi

# ---- deploy -----------------------------------------------------------------
say "deploying the consolidated run"
scp -P $PORT "$SCRIPT" $BOARD:/root/psearch/ 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"
ssh -p $PORT $BOARD 'chmod +x /root/psearch/pipeline_arch_run.sh; md5sum /root/psearch/pipeline_arch_run.sh' 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"

# ---- 5: NPU health check ----------------------------------------------------
say "NPU health check (3 frames)"
ssh -p $PORT $BOARD '
  d=/root/psearch/health_recover; mkdir -p $d
  cd /root
  timeout 240 ./rknn_mosaic_e2e_sweep3 --raw-root /mnt/srv2/lq_ir_bmp \
    --sequence VIDEO00007_19700101_001324 --model /root/psearch/models/p0_off_A_none.rknn \
    --out $d --kernel-dir /root/psearch/kernels --cpu-affinity 4-7 --npu-workers 1 \
    --post-workers 1 --anchor-step 10 --pace-ms 0 --limit 3 > $d/log.txt 2>&1
  echo "health exit=$?"
  tail -3 $d/log.txt
' 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"

HEALTH=$(ssh -p $PORT $BOARD 'd=/root/psearch/health_recover; grep -c "^\[timing\]" $d/log.txt 2>/dev/null' 2>/dev/null | tail -1)
if [ "${HEALTH:-0}" -lt 1 ]; then
  say "THE NPU IS STILL UNHEALTHY -- the consolidated run is NOT launched"
  say "the board needs another power cycle (the driver state survived the reboot)"
  exit 1
fi
say "the NPU is healthy -- launching the consolidated run"

# ---- launch -----------------------------------------------------------------
ssh -p $PORT $BOARD 'cd /root/psearch && nohup setsid ./pipeline_arch_run.sh > /root/psearch/arch_run.log 2>&1 & sleep 3; pgrep -af pipeline_arch_run | head -1' 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"
say "launched; follow it with: ssh -p 22 root@192.168.0.64 'tail -f /root/psearch/arch_run.log'"
