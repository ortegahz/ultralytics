#!/usr/bin/env bash
# ===========================================================================
# print_yolo26n_ref_table.sh -- read the board's measurement results back and
# print a comparison table.
#
# The board writes every raw timing line to <NFS>/yolo26n_ref/ref_results.txt.
# The packed session also emits one machine-readable summary line per model:
#
#     @@SUMMARY@@ <label>|<source>|<per-frame ms>
#
# This script turns those into a table and (with --raw) dumps the raw timing
# lines verbatim.
#
# USAGE
#   bash manu/pipeline/npu/print_yolo26n_ref_table.sh
#   bash manu/pipeline/npu/print_yolo26n_ref_table.sh --raw
#   RES=/path/to/ref_results.txt bash manu/pipeline/npu/print_yolo26n_ref_table.sh
# ===========================================================================
set -uo pipefail

NFS_HOST_DIR="${RKNPU_NFS_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu/yolo26n_ref}"
RES="${RES:-$NFS_HOST_DIR/ref_results.txt}"

RAW=0
[ "${1:-}" = "--raw" ] && RAW=1

if [ ! -f "$RES" ]; then
  echo "[ERROR] no results file: $RES" >&2
  echo "        (has the board session been run after the power cycle?)" >&2
  exit 1
fi

echo "results file : $RES"
echo "size         : $(stat -c %s "$RES") bytes, $(wc -l < "$RES") lines"
echo

if [ "$RAW" = "1" ]; then
  echo "=================== RAW TIMING LINES (verbatim) ==================="
  grep -E "average run|once run use|rknn_run|NPU total|inputs_set|outputs_get|fps|SUMMARY|wall clock|HEALTH|health check" "$RES"
  echo "==================================================================="
  echo
fi

# ---- GFLOPs per model (int8, same input size as the measurement) ----------
gflops_for() {
  case "$1" in
    *mobilenet*) echo "0.57" ;;
    *yolov5s*)   echo "7.2"  ;;
    *yolo26n*)   echo "6.1"  ;;
    *)           echo "?"    ;;
  esac
}

printf '%-34s %-38s %12s %10s %12s\n' "model" "source" "ms/frame" "GFLOPs" "implied GOPS"
printf '%-34s %-38s %12s %10s %12s\n' "----" "------" "--------" "------" "------------"

awk -F'|' '
  /@@SUMMARY@@/ {
    # line format: "@@SUMMARY@@ <label>|<source>|<ms>"
    label=$1; sub(/^@@SUMMARY@@ /, "", label);
    src=$2; ms=$3+0;
    printf "%-34s %-38s %12.3f\n", label, src, ms;
  }
' "$RES" | while IFS= read -r line; do
  label=$(printf '%s' "$line" | awk '{print $1}')
  src=$(printf '%s' "$line" | awk '{print $2}')
  ms=$(printf '%s' "$line" | awk '{print $3}')
  gf=$(gflops_for "$label")
  if [ "$gf" != "?" ]; then
    gops=$(awk -v g="$gf" -v m="$ms" 'BEGIN{ if (m>0) printf "%.0f", (g*1000.0)/m; else print "?" }')
  else
    gops="?"
  fi
  printf '%-34s %-38s %12s %10s %12s\n' "$label" "$src" "$ms" "$gf" "$gops"
done

echo
echo "NOTE  implied GOPS = GFLOPs * 1000 / ms_per_frame  (counts 1 MAC = 2 ops,"
echo "      so it is the classic GOPS figure and is directly comparable with"
echo "      the RK3588 NPU's quoted int8 TOPS)."
