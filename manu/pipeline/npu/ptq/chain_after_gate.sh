#!/bin/bash
# =============================================================================
# chain_after_gate.sh   (runs ON the RK3588 board)
#
# Waits for the gate arm's e2e process to exit, then launches the live arms.
# The board is the serial bottleneck for every long run, so nothing may sit
# idle between the arms: this chain removes the dead time.
#
# IT ALSO GUARDS THE ORDER
#   Only ONE e2e may run at a time on this board: two concurrent RKNN
#   contexts over the two-context limit cause "failed to submit", then a 6 s
#   job timeout, then a driver soft reset, then SIGILL in rknn_init, and
#   finally the whole board goes dark and needs a physical power cycle.
#   So the chain waits for the previous e2e to be GONE, not just for the
#   script to return.
# =============================================================================
set -u

LOG=/root/psearch/chain.log
say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for the gate arm to finish"
while pgrep -f "rknn_mosaic_e2e_sweep3 .*A_gate_step2_nodrop" >/dev/null 2>&1; do
  sleep 15
done
say "the gate arm is done -- settling 20 s before the next e2e"
sleep 20

say "launching the live arms"
cd /root/psearch
nohup setsid ./pipeline_live_run.sh > /root/psearch/live_launch.log 2>&1 &
sleep 5
pgrep -af pipeline_live_run | head -1 | cut -c1-100
say "live arms launched"
