#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 09_board_threshold_probe.sh -- board-side threshold sweep probe.
#
# WHY THIS PROBE EXISTS
# ---------------------
# The trial-0080 int8 model returns ZERO detections on every frame on the
# board (F1 0.0000 / FP 0 / FN 2000) while the PC PyTorch reference on the
# SAME BMP bytes gives F1 0.7484.  Two candidate causes:
#
#   (a) the int8 graph INPUT is fed wrong (the "uint8 upload into an INT8
#       input tensor" story), or
#   (b) the OUTPUT heatmap tensor was calibrated into too narrow a range, so
#       every real peak is clipped at the tensor's maximum and lands BELOW the
#       decision threshold 0.3.
#
# The probe separates them without changing anything: if the model is fine and
# only the RANGE is clipped, then LOWERING --th must make detections come back
# (with lower scores), while the fp16 control arm stays flat across the same
# sweep.  If the input path were corrupt, lowering --th would only produce
# garbage boxes.
#
# It also prints the board's own tensor attributes (type / zp / scale) for the
# input and both outputs, which is where the clipping range is visible.
#
# USAGE (on the board):
#   ./09_board_threshold_probe.sh <model.rknn> <TAG> [outdir] [seq] [start] [n] [th-list]
#   ./09_board_threshold_probe.sh /mnt/manu/ptq/models/p0_off_C_kl_divergence.rknn int8
#
# The BMP pool is the board's NFS view of the host staging area:
#   host  /run/media/manu/SSD960-LOCAL/nfs/ptq/bmp/<seq>/<seq>__%06d.bmp
#   board /mnt/manu/ptq/bmp/<seq>/<seq>__%06d.bmp
# ---------------------------------------------------------------------------
set -uo pipefail

MODEL="${1:?model.rknn}"
TAG="${2:?tag}"
OUTDIR="${3:-/root/ptq/probe}"
SEQ="${4:-VIDEO00007_19700101_001324}"
START="${5:-4000}"
N="${6:-500}"
THS="${7:-0.3 0.1 0.05 0.02}"

BIN=/root/rknn_heatmap_smoke
BMPROOT=/mnt/manu/ptq/bmp

mkdir -p "$OUTDIR"
LIST="$OUTDIR/${TAG}_list.txt"
: > "$LIST"
for ((i = START; i < START + N; i++)); do
  printf '%s/%s/%s__%06d.bmp\n' "$BMPROOT" "$SEQ" "$SEQ" "$i" >> "$LIST"
done
echo "[probe] $TAG  frames=$N  start=$START  seq=$SEQ"
echo "[probe] list=$LIST  ($(wc -l < "$LIST") lines)"

for TH in $THS; do
  TSV="$OUTDIR/${TAG}_th${TH}.tsv"
  LOG="$OUTDIR/${TAG}_th${TH}.log"
  "$BIN" "$MODEL" --batch "$LIST" --th "$TH" --topk 80 \
      --detections-out "$TSV" --tag "${TAG}_th${TH}" > "$LOG" 2>&1
  rc=$?
  # frames with >=1 detection, total detections, score max
  stats=$(awk -F'\t' '
    { n = $2 + 0; if (n > 0) nz++; s += n
      if (n > 0) { for (i = 3; i <= NF; i++) { split($i, a, " "); if (a[3] + 0 > mx) mx = a[3] + 0 } } }
    END { printf "frames_with_det=%d total_det=%d score_max=%.4f", nz + 0, s + 0, mx + 0 }' "$TSV")
  echo "[probe] th=$TH rc=$rc $stats"
done

echo "[probe] ---- board tensor attributes (from the last run's log) ----"
grep -E 'input attr|out\[|in \[|dims=|type=' "$OUTDIR/${TAG}_th${TH}.log" | head -8
echo "[probe] done -> $OUTDIR"
