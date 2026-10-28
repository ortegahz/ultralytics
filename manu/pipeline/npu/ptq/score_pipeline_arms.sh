#!/bin/bash
# =============================================================================
# score_pipeline_arms.sh
#
# Scores the decoupled-pipeline arms on the SERVER.
#
# WHY IT IS SEPARATE FROM THE BOARD RUN
# -------------------------------------
# The board cannot reach the server over ssh (no public key on the board), and
# the board's raw data lives on the host's NFS export -- which dies with the
# host.  So the two halves are split:
#
#   board  : produces the arm TSVs, and publishes them to the SERVER-visible
#            path /mnt/srv2/psearch/results/<arm>/  (the server's ptq_work
#            over SSHFS).  This works with the host down.
#   server : scores them.  This needs the host's repo mirror only for the
#            scripts, which the server already has.
#
# Run this from the host AFTER the board search finishes (or whenever):
#   bash score_pipeline_arms.sh <arm> [arm...]
# With no argument it scores every arm the board has published.
#
# WHAT IS BEING CHECKED
# ---------------------
# The architecture must NOT cost accuracy.  The E arm (the e2e pipeline) is
# scored against the frozen baselines (A/B/C) with the track3_blind
# parameters, and compared with the CERTIFIED step10 zero-drop baseline:
#
#   svc_A_idle  E-arm F1 = 0.891676   <- the reference to match
#   svc_D_step2 E-arm F1 = 0.891871   (step2, +0.000195 over step10)
#
# IMPORTANT: only FULL-SEQUENCE arms are comparable.  A partial arm (fewer
# than 11810 rows) has its uncovered frames counted as FN in the scorer's
# denominator, so its E-arm F1 collapses and means nothing.
#
# Run from the local repo (the host):
#   bash manu/pipeline/npu/ptq/score_pipeline_arms.sh [arm ...]
# =============================================================================
set -u

SRV=huangzhe@192.168.99.40
PORT=32222
REPO_SRV=/tmp/pycharm_project_10ae9e2e
SEQ=VIDEO00007_19700101_001324
ARM_SRV_ROOT=$REPO_SRV/manu/pipeline/npu/ptq/board_tsv_svc
OUT_SRV_ROOT=/mnt/data/siping/ptq_work/final_report_svc
LOCAL_OUT=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu/results/svc_arms
# The board publishes here, and it is the same directory the server sees.
PUB_LOCAL=/mnt/srv2/psearch/results

if [ $# -ge 1 ]; then
  ARMS="$*"
else
  # Every arm the board has published.
  ARMS=$(ls "$PUB_LOCAL" 2>/dev/null | tr '\n' ' ')
fi

if [ -z "${ARMS// /}" ]; then
  echo "no arms found under $PUB_LOCAL" >&2
  exit 1
fi

echo "=== scoring arms: $ARMS ==="
for arm in $ARMS; do
  echo "=================== arm: $arm ==================="
  srcdir=$PUB_LOCAL/$arm
  if [ ! -f "$srcdir/${SEQ}_mosaic.tsv" ]; then
    echo "[skip] $arm: no TSV at $srcdir"
    continue
  fi
  rows=$(wc -l < "$srcdir/${SEQ}_mosaic.tsv")
  echo "[info] TSV rows = $rows  (a full-sequence arm has 11810)"

  # 1) stage the arm on the server
  ssh -p $PORT $SRV "mkdir -p '$ARM_SRV_ROOT/$arm'" || continue
  scp -P $PORT "$srcdir/${SEQ}_mosaic.tsv" "$srcdir/timing.txt" "$SRV:$ARM_SRV_ROOT/$arm/" || continue

  # 2) score it
  ssh -p $PORT $SRV "source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav && cd $REPO_SRV && \
python manu/pipeline/npu/ptq/38_mosaic_final_report.py \
  --group track3_blind=$SEQ \
  --board-tsv-dir manu/pipeline/npu/ptq/board_tsv \
  --e2e-tsv-dir manu/pipeline/npu/ptq/board_tsv_svc/$arm \
  --pc-area-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_area_all \
  --pc-linear-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_linear_all \
  --e2e-timing manu/pipeline/npu/ptq/board_tsv_svc/$arm/timing.txt \
  --out $OUT_SRV_ROOT/$arm" 2>&1 | tail -12

  # 3) pull the report back
  mkdir -p "$LOCAL_OUT/$arm"
  scp -P $PORT "$SRV:$OUT_SRV_ROOT/$arm/mosaic_vs_f32_report.md" \
             "$SRV:$OUT_SRV_ROOT/$arm/mosaic_vs_f32_report.json" \
      "$LOCAL_OUT/$arm/" 2>/dev/null

  # 4) print the E-arm line, which is the number that matters
  if [ -f "$LOCAL_OUT/$arm/mosaic_vs_f32_report.md" ]; then
    echo "--- $arm : E-arm ---"
    grep -E '^\| \*\*E\*\*' "$LOCAL_OUT/$arm/mosaic_vs_f32_report.md" | head -1
  else
    echo "[warn] $arm: the report did not come back"
  fi
  echo
done

echo "=== reference (certified step10, zero drop) : E-arm F1 = 0.891676 ==="
echo "=== reference (step2,      zero drop)      : E-arm F1 = 0.891871 ==="
