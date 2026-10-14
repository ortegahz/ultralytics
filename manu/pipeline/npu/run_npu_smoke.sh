#!/usr/bin/env bash
# ===========================================================================
# run_npu_smoke.sh -- build, deploy and run the NPU smoke test on the board.
#
# Storage split, per the project's board-storage rule (board root has ~1.7 GiB
# free, so anything bulky goes through the NFS exchange area):
#
#   * binaries  -> board local disk via scp   (~148 KB)
#       NFS is NOT used for executables: the NFS server still has the write in
#       flight when the board execs the file, which yields "Text file busy"
#       (see falsified_archive.md 35.6).
#   * model/image -> /mnt/manu/npu/data  (NFS == host /run/media/manu/SSD960-LOCAL/nfs/npu/data)
#       Nothing bulky ever lands on the board's own filesystem.
#
# Host-side prerequisites:
#   ssh root@192.168.0.64 'mkdir -p /mnt/manu/npu/data && chown 1000:1000 /mnt/manu/npu/data'
#   (the board mounts NFS with no_root_squash and no uid=/gid=, so without the
#    chown the host user cannot write into a board-created directory)
#
# Usage: ./run_npu_smoke.sh [--loops N] [--perf]
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="$(cd "$HERE/../opencl/board" && pwd)"

REMOTE_BIN="/root/rknn_npu_smoke"
REMOTE_MODEL="/mnt/manu/npu/data/yolov5s-640-640.rknn"
REMOTE_IMAGE="/mnt/manu/npu/data/bus.jpg"
EXTRA_ARGS="${*:-}"

echo "== build =="
"$HERE/build_rknn_smoke_arm64.sh" | tail -3

echo
echo "== deploy =="
python3 "$BOARD/board_run.py" --put "$HERE/out/rknn_npu_smoke:$REMOTE_BIN" | tail -4

echo
echo "== run =="
python3 "$BOARD/board_run.py" \
  --exec "$REMOTE_BIN $REMOTE_MODEL $REMOTE_IMAGE --loops 30 --warmup 10 $EXTRA_ARGS"