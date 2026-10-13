#!/usr/bin/env bash
# ===========================================================================
# run_coco_bench.sh -- run several .rknn models over a folder of images on the
# board, capturing detections per (model, image) for later box-by-box diffing.
#
# The official demo writes out.jpg into its CWD, so each image is processed in
# a private sub-directory. Its own per-detection lines ("person @ (l t r b) s")
# go to a log that lands on the host via NFS.
#
# Usage: RK3588_PASSWORD=... ./run_coco_bench.sh <img_dir_on_board> <out_tag>
# ===========================================================================
set -euo pipefail

IMG_DIR="${1:?image dir on board}"
TAG="${2:?output tag}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="$(cd "$HERE/../opencl/board" && pwd)"

# Run from the NFS demo dir so the demo finds model/coco_80_labels_list.txt.
WORK="/mnt/manu/npu/bench/$TAG"
DEMO=/root/rknn_demo/rknn_yolov5_demo
LIB=/root/rknn_demo/lib
MODEL_DIR=/mnt/manu/npu/demo/model/RK3588

python3 "$BOARD/board_run.py" --exec \
  "mkdir -p $WORK && cd /mnt/manu/npu/demo && \
   for img in $IMG_DIR/*.jpg; do \
     b=\$(basename \$img); \
     mkdir -p $WORK/\$b; \
     LD_LIBRARY_PATH=$LIB $DEMO $MODEL_DIR/\$MODEL $img 2>/dev/null | grep '@ (' > $WORK/\$b.txt || true; \
     cp out.jpg $WORK/\$b.jpg 2>/dev/null || true; \
   done; \
   echo BENCH_DONE" | tail -3