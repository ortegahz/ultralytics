#!/bin/bash
# =============================================================================
# score_watcher.sh   (runs ON the SERVER, detached)
#
# WHY A WATCHER AND NOT A ONE-SHOT SCORE
# --------------------------------------
# The host is going to be shut down.  The board cannot reach the server over
# ssh (it has no public key), so the board PUBLISHES its arm artefacts to
# /mnt/srv2/psearch/results/<arm>/ -- which is the server's own ptq_work over
# SSHFS, and therefore visible to the server without the host.
#
# But the scoring itself needs the server's python + the repo at
# /tmp/pycharm_project_10ae9e2e, and it has to be LAUNCHED on the server.
# If the host dies first, nobody can launch it.
#
# So the watcher is launched ONCE, now, while the host is still up, and it
# polls the publish directory forever.  Every arm that appears and has not
# been scored yet is scored, and the result is written next to it.  The
# watcher is idempotent: an arm with a finished report is skipped.
#
# IT SCORES TWO WAYS
#   full-GT    the ordinary report (38_mosaic_final_report.py).  For a PARTIAL
#              arm (a live arm that only produced some frames) its E-arm F1
#              is meaningless -- the uncovered frames are counted as FN -- so
#              that number is reported but ALWAYS labelled as partial.
#   surviving  the restricted-to-the-produced-frames report
#              (score_surviving_frames.py), which is the number that means
#              something for a live arm.
#
# Usage:  nohup setsid bash score_watcher.sh > /tmp/score_watcher.log 2>&1 &
# =============================================================================
set -u

REPO=/tmp/pycharm_project_10ae9e2e
PUB=/mnt/data/siping/ptq_work/psearch/results
OUT_ROOT=/mnt/data/siping/ptq_work/final_report_svc
SURV_OUT=/mnt/data/siping/ptq_work/surviving_frames
SEQ=VIDEO00007_19700101_001324
PARAMS=$REPO/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/track3_blind.json
REF_TSV=$REPO/manu/pipeline/npu/ptq/board_tsv_svc/P1b_step2_nodrop_orig/${SEQ}_mosaic.tsv
LOG=/tmp/score_watcher.log
DONE=$SURV_OUT/.scored_arms

mkdir -p "$OUT_ROOT" "$SURV_OUT"
touch "$DONE"

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd "$REPO" || exit 1

echo "[watcher] started $(date)" | tee -a "$LOG"

while true; do
  for d in "$PUB"/*/; do
    arm=$(basename "$d")
    tsv="$d/${SEQ}_mosaic.tsv"
    [ -f "$tsv" ] || continue
    grep -qxF "$arm" "$DONE" && continue
    rows=$(wc -l < "$tsv")
    echo "[watcher] scoring $arm ($rows TSV rows) $(date)" | tee -a "$LOG"

    # 1) the ordinary report (full GT).  For a partial arm this number is
    #    NOT comparable to the certified 0.891871 -- it is still written out
    #    because the report also carries the timing table.
    python manu/pipeline/npu/ptq/38_mosaic_final_report.py \
      --group track3_blind=$SEQ \
      --board-tsv-dir manu/pipeline/npu/ptq/board_tsv \
      --e2e-tsv-dir "$PUB/$arm" \
      --pc-area-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_area_all \
      --pc-linear-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_linear_all \
      --e2e-timing "$d/timing.txt" \
      --out "$OUT_ROOT/$arm" >> "$LOG" 2>&1

    # 2) the surviving-frames report: the number that means something for a
    #    live arm, because the ground truth is restricted to the frames the
    #    arm actually produced.
    if [ -f "$REF_TSV" ]; then
      python manu/pipeline/npu/ptq/score_surviving_frames.py \
        --arm-tsv "$tsv" --ref-tsv "$REF_TSV" --seq "$SEQ" \
        --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
        --params-json "$PARAMS" \
        --work-dir "/tmp/surv_work_$arm" \
        --out-json "$SURV_OUT/${arm}.json" \
        --out-md "$SURV_OUT/${arm}.md" >> "$LOG" 2>&1
    fi

    grep -qxF "$arm" "$DONE" || echo "$arm" >> "$DONE"
    echo "[watcher] done $arm" | tee -a "$LOG"
  done
  sleep 60
done
