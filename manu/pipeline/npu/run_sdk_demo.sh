#!/usr/bin/env bash
# ===========================================================================
# run_sdk_demo.sh -- deploy and run Rockchip's rknn_yolov5_demo on the board.
#
# This is the reference arm. Its printed boxes are what rknn_npu_smoke is
# checked against; see README.md section 7.
#
# Storage split:
#   executable  -> board local disk (/root/rknn_demo/, ~7.2 MB) because NFS
#                  exec yields "Text file busy"
#   librga.so   -> board local disk (133 KB; the demo's RPATH is relative "lib")
#   model/image -> /mnt/manu/npu/demo (NFS == host storage, not board disk)
#
# The demo resolves labels at ./model/coco_80_labels_list.txt relative to its
# CWD, so it is run from the NFS demo directory.
#
# Usage: RK3588_PASSWORD=... ./run_sdk_demo.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="$(cd "$HERE/../opencl/board" && pwd)"
SDK="${RKNN_SDK:-/media/manu/1TB-Volume/rk3588/rk3588_sdk/EVM3588-A_20250815/02Linux软件资料/SDK&Image/SDK/evm3588_linux_sdk_v2.1.1.0/external/rknpu2}"

# Two different roots for the SAME directory -- do not conflate them:
#   board side : /mnt/manu/npu/demo   (the board's NFS mount point)
#   host  side : /home/manu/nfs/npu/demo
# The host cannot use the board path and vice versa; that mismatch is the
# "same path, different meaning" trap recorded in rules.md.
RGA="$SDK/examples/3rdparty/rga/RK3588/lib/Linux/aarch64/librga.so"
DEMO_DIR="/mnt/manu/npu/demo"                                    # on the board
DEMO_DIR_HOST="/home/manu/nfs/npu/demo"                          # on this host

echo "== prepare NFS demo dir (board creates it and chowns to the host uid) =="
# Order matters: the board's NFS mount has no uid=/gid= mapping and the export
# carries no_root_squash, so anything the board creates lands on the host as
# root:root and the host user then cannot write into it. Create + chown from the
# board FIRST, then the host can copy files in.
python3 "$BOARD/board_run.py" --exec \
  "mkdir -p $DEMO_DIR/model/RK3588 && chown -R 1000:1000 $DEMO_DIR" | tail -2

echo
echo "== stage assets on the NFS exchange area =="
mkdir -p "$DEMO_DIR_HOST/model/RK3588"
cp -f "$SDK/examples/rknn_yolov5_demo/model/coco_80_labels_list.txt" "$DEMO_DIR_HOST/model/"
cp -f "$SDK/examples/rknn_yolov5_demo/model/bus.jpg" "$DEMO_DIR_HOST/"
cp -f "$SDK/examples/rknn_yolov5_demo/model/RK3588/yolov5s-640-640.rknn" "$DEMO_DIR_HOST/model/RK3588/"

echo "== build =="
"$HERE/build_sdk_demo_arm64.sh" | tail -4

echo
echo "== deploy =="
python3 "$BOARD/board_run.py" \
  --put "$HERE/.demo-build/rknn_yolov5_demo:/root/rknn_demo/rknn_yolov5_demo" \
  --put "$RGA:/root/rknn_demo/lib/librga.so" \
  | tail -4

echo
echo "== run =="
python3 "$BOARD/board_run.py" --exec \
  "cd $DEMO_DIR && LD_LIBRARY_PATH=/root/rknn_demo/lib /root/rknn_demo/rknn_yolov5_demo model/RK3588/yolov5s-640-640.rknn bus.jpg"
echo
echo "reference OSD: host /home/manu/nfs/npu/demo/out.jpg"