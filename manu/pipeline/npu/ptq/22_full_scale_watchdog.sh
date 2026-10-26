#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 22_full_scale_watchdog.sh -- keep the int8 full-scale run alive until every
# one of the 60,610 frames is scored.
#
# WHY
# ---
# The workstation is being powered off around 18:30.  Its NFS export is what
# the board mounts at /mnt/manu, so the moment the host goes down the board's
# NFS mount can wedge: an ssh to the board then times out during the banner
# exchange and every board operation fails.  The full-scale driver therefore
# cannot simply be started once and forgotten -- it has to be retried until
# the board is healthy AND every chunk is scored.
#
# The driver itself is resumable: a chunk whose record says OK, whose model
# md5 still matches and whose TSV still passes the row/name check is skipped.
# So the watchdog's job is only to (re)start it when it is not running, and to
# stop once the run is complete.
#
# COMPLETION TEST
#   status.json must report 0 failed chunks and 60,610 scored frames.
#
#   ⚠️ FIELD NAMES (fixed 2026-10-23): the driver writes `frames_done`,
#   `done_chunks`, `total_chunks` and `errors` into status.json -- NOT
#   `frames_scored` and NOT a `chunks` dict.  The old test read the wrong
#   keys, always got None/-1, and the watchdog could therefore never
#   recognise a finished run (it looked like a permanent failure and kept
#   the driver alive).
#
# BOARD OWNERSHIP (added 2026-10-23)
#   The watchdog CLEANS the board (`pkill -x rknn_heatmap_sm`) before it
#   restarts the driver.  On 2026-10-23 the board was handed over to a
#   DIFFERENT harness that owns it for inference-speed work -- running this
#   watchdog then would have killed the other harness's job.
#   => Board touching is now OPT-IN: set PTQ_ALLOW_BOARD_CLEAR=1 in the
#      environment.  Without it the watchdog only observes and restarts the
#      driver; it never runs pkill and never probes the board.
# ---------------------------------------------------------------------------
set -uo pipefail
W=/mnt/data/siping/ptq_work
REPO=/tmp/pycharm_project_10ae9e2e
MODEL=$W/screen_n/p0_off_A_normal.rknn
OUT=$W/full_scale_int8
LOG=$W/logs/full_scale_int8.log
WDLOG=$W/logs/full_scale_watchdog.log
TOTAL=60610
# Opt-in board ownership.  Default: observe only, never touch the board.
ALLOW_BOARD_CLEAR=${PTQ_ALLOW_BOARD_CLEAR:-0}

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$WDLOG"; }

log "watchdog started (model=$MODEL out=$OUT allow_board_clear=$ALLOW_BOARD_CLEAR)"
if [ "$ALLOW_BOARD_CLEAR" != "1" ]; then
  log "board touching DISABLED (set PTQ_ALLOW_BOARD_CLEAR=1 to enable) -- another harness may own the board"
fi

while true; do
  # ---- is the run already finished? -------------------------------------
  if [ -f "$OUT/status.json" ]; then
    done_n=$(python3 - "$OUT/status.json" <<'EOF' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(-1); raise SystemExit
# the driver writes `frames_done`; `frames_scored` is kept only as a
# fallback for a differently-named status file
sc = d.get("frames_done")
if sc is None:
    sc = d.get("frames_scored")
if sc is None:
    ch = d.get("chunks")
    if isinstance(ch, dict):
        sc = sum(v.get("frames", 0) for v in ch.values())
    elif isinstance(ch, list):
        sc = sum(v.get("frames", 0) for v in ch)
print(-1 if sc is None else sc)
EOF
)
    failed_n=$(python3 - "$OUT/status.json" <<'EOF' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(-1); raise SystemExit
if "errors" in d:
    print(len(d.get("errors") or []))
elif "chunks_failed" in d:
    print(int(d.get("chunks_failed") or 0))
else:
    print(-1)
EOF
)
    if [ "$done_n" = "$TOTAL" ] && [ "$failed_n" = "0" ]; then
      log "COMPLETE: $done_n/$TOTAL frames, 0 failed chunks -- watchdog exits"
      break
    fi
    log "status.json: frames_done=$done_n failed=$failed_n (need $TOTAL/0)"
  fi

  # ---- is the driver running? -------------------------------------------
  running=$(ps -eo args | grep -c "[1]4_run_full_scale_nofs[.]py")
  if [ "$running" -gt 0 ]; then
    sleep 60; continue
  fi

  # ---- is the board reachable? ------------------------------------------
  # NOTE: keep the probes SPARSE.  sshd's MaxStartups drops new connections
  # when too many are still in the unauthenticated state, and a probe that
  # lingers for its whole ConnectTimeout occupies a slot the whole time.
  # A 20 s probe every 120 s is enough and does not pile up.
  if [ "$ALLOW_BOARD_CLEAR" != "1" ]; then
    # Not authorised to touch the board: do not probe it, do not clear it.
    # The driver will fail on its own and the next round will retry.
    log "board touching disabled -- restarting the driver without touching the board"
  elif ! timeout 20 ssh -o ConnectTimeout=15 -o BatchMode=yes \
        -o ControlMaster=no -o ControlPath=none ptq-board \
        'ps -eo comm | grep -c "^rknn_heatmap"' < /dev/null > "$W/logs/board_ctx.txt" 2>/dev/null; then
    log "board unreachable (sshd kex reset / NFS wedge) -- waiting 120s"
    sleep 120; continue
  fi

  if [ "$ALLOW_BOARD_CLEAR" = "1" ]; then
    ctx=$(tr -d ' \n' < "$W/logs/board_ctx.txt" 2>/dev/null); ctx=${ctx:-0}
    if [ "$ctx" -gt 0 ]; then
      log "board has $ctx demo process(es) alive -- clearing stale ones"
      # -x matches the PROCESS NAME exactly (comm is truncated to 15 chars:
      # "rknn_heatmap_sm").  `pkill -f` would match the whole command line and
      # could kill the sshd session wrappers as well -- do not use -f here.
      timeout 20 ssh -o ConnectTimeout=10 ptq-board \
        'pkill -x rknn_heatmap_sm 2>/dev/null; true' < /dev/null >/dev/null 2>&1
      sleep 20
    fi
  fi

  # ---- (re)start the resumable driver -----------------------------------
  log "starting the full-scale driver (resume)"
  source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
  conda activate uav
  cd "$REPO"
  setsid nohup python -u manu/pipeline/npu/ptq/14_run_full_scale_nofs.py \
    --tracks blind golden anti \
    --model "$MODEL" \
    --frames-root "$W/bmp" \
    --frames-root "anti=$W/pcfeat_bmp/images" \
    --stage-dir /userdata/ptq_stage \
    --chunk 500 --th 0.30 --topk 100 \
    --out "$OUT" \
    --pt-device cuda:1 \
    < /dev/null >> "$LOG" 2>&1 &
  log "driver pid=$!"
  sleep 120
done
