#!/bin/bash
# ===========================================================================
# batch_eval.sh -- runs ON THE BOARD. Sweeps a folder of images through one
# .rknn model and writes one detections file per image into an NFS output dir.
#
# Invoked as:  batch_eval.sh <model.rknn> <out_dir> [img_dir]
#
# This lives on the board rather than being inlined into an SSH command because
# board_run.py --exec only accepts a SINGLE-LINE command: a multi-line `for`
# loop embedded in the exec string gets fed to the pty with its newlines intact
# and the prompt matcher never fires, so the command silently produces nothing.
# Deploying the loop as a file keeps the exec a one-liner.
#
# --loops 1 is used because rknn_npu_smoke is bit-deterministic and the official
# demo always burns 10 inference passes per invocation; a 300-image sweep is ~10x
# cheaper this way with identical detections.
# ===========================================================================
set -u

MODEL="${1:?model.rknn}"
OUT="${2:?output dir}"
IMG_DIR="${3:-/mnt/manu/npu/h300}"

BIN=/root/rknn_npu_smoke

mkdir -p "$OUT" || exit 1

n=0
det=0
for img in "$IMG_DIR"/*.jpg; do
  b=$(basename "$img" .jpg)
  "$BIN" "$MODEL" "$img" --loops 1 --warmup 0 2>/dev/null \
    | grep -E '^ +#' \
    | sed -e 's/^ *//' -e 's/  */ /g' -e 's/[[:space:]]*$//' > "$OUT/$b.txt"
  c=$(wc -l < "$OUT/$b.txt")
  det=$((det + c))
  n=$((n + 1))
  # heartbeat every 25 images so a stalled sweep is visible rather than silent
  if [ $((n % 25)) -eq 0 ]; then
    echo "  ... $n images, $det detections so far"
  fi
done

echo "DONE images=$n detections=$det out=$OUT"