#!/bin/bash
# =============================================================================
# board_recover_final.sh   (runs on the HOST)
#
# Waits for the board to come back after a power cycle, deploys the final
# consolidated run and launches it.  The board's local disk already holds the
# model, the kernels and the sweep3 binaries, so the recovery is: wait, push
# the script, run.
#
# It does NOT touch the already-published artefacts, and it does NOT start
# anything until the NPU health check passes -- a long run that dies in its
# first minute costs more than no run at all.
# =============================================================================
set -u

BOARD=root@192.168.0.64
PORT=22
LOCAL=/media/manu/1TB-Volume/workspace/ultralytics
SCRIPT=$LOCAL/manu/pipeline/npu/ptq/pipeline_final_run.sh
LOG=/tmp/board_final.log

say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for $BOARD"
while true; do
  if timeout 12 ssh -p $PORT -o ConnectTimeout=6 -o BatchMode=yes \
       -o StrictHostKeyChecking=no $BOARD 'echo UP' >/dev/null 2>&1; then
    say "the board is up"
    break
  fi
  sleep 15
done

say "checking the environment"
ssh -p $PORT $BOARD '
  uptime
  echo "--- srv2 ---"; mount | grep srv2 | head -1
  echo "--- raw ---"; ls /mnt/srv2/lq_ir_bmp 2>/dev/null | head -2
  echo "--- model/kernels ---"; ls /root/psearch/models/ /root/psearch/kernels/ 2>/dev/null | head -8
  echo "--- binaries ---"; md5sum /root/rknn_mosaic_e2e_sweep3 2>/dev/null
' 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"

say "deploying the final consolidated run"
scp -P $PORT "$SCRIPT" $BOARD:/root/psearch/ 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"
ssh -p $PORT $BOARD 'chmod +x /root/psearch/pipeline_final_run.sh; md5sum /root/psearch/pipeline_final_run.sh' 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"

say "NPU health check"
ssh -p $PORT $BOARD '
  d=/root/psearch/health_boot; mkdir -p $d; cd /root
  timeout 240 ./rknn_mosaic_e2e_sweep3 --raw-root /mnt/srv2/lq_ir_bmp \
    --sequence VIDEO00007_19700101_001324 --model /root/psearch/models/p0_off_A_none.rknn \
    --out $d --kernel-dir /root/psearch/kernels --cpu-affinity 4-7 --npu-workers 1 \
    --post-workers 1 --anchor-step 10 --pace-ms 0 --limit 3 > $d/log.txt 2>&1
  echo "health exit=$?"; tail -3 $d/log.txt
' 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"

HEALTH=$(ssh -p $PORT $BOARD 'grep -c "^\[timing\]" /root/psearch/health_boot/log.txt 2>/dev/null' 2>/dev/null | tail -1)
if [ "${HEALTH:-0}" -lt 1 ]; then
  say "THE NPU IS UNHEALTHY -- the final run is NOT launched"
  exit 1
fi
say "the NPU is healthy -- launching the final consolidated run"
ssh -p $PORT $BOARD 'cd /root/psearch && nohup setsid ./pipeline_final_run.sh > /root/psearch/final_run_launch.log 2>&1 & sleep 4; pgrep -af pipeline_final_run | head -1 | cut -c1-90' 2>&1 | grep -vE "WARNING|vulnerable|upgraded" | tee -a "$LOG"
say "follow it with: ssh -p 22 root@192.168.0.64 'tail -f /root/psearch/final_run.log'"
