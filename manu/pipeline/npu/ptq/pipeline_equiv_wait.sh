#!/bin/bash
# =============================================================================
# pipeline_equiv_check.sh  -- SERIALISED after the throughput search.
#
# The board has ONE meaningful CPU stage and a bounded RKNN context count, so
# two experiments at once would contaminate BOTH.  This waiter runs the
# equivalence check only after the throughput search has finished.
#
# It waits on the driver's completion marker in the driver log.
# =============================================================================
set -u

REMOTE=/root/psearch
LOG=$REMOTE/driver.log
WAIT_MAX=${WAIT_MAX:-14400}      # 4 h cap
SCRIPT=$REMOTE/pipeline_equiv_check.sh

echo "[equiv-wait] started, waiting for the throughput search to finish"
t0=$(date +%s)
while true; do
  if grep -q "SEARCH_DONE" "$LOG" 2>/dev/null; then
    echo "[equiv-wait] the search finished; running the equivalence check"
    break
  fi
  now=$(date +%s)
  if [ $((now - t0)) -ge $WAIT_MAX ]; then
    echo "[equiv-wait] TIMED OUT after ${WAIT_MAX}s; the search is still running"
    exit 1
  fi
  sleep 20
done

bash "$SCRIPT"
echo "[equiv-wait] done"
