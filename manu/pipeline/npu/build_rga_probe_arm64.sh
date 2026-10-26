#!/usr/bin/env bash
# ===========================================================================
# build_rga_probe_arm64.sh -- cross-compile rga_probe.cpp for the RK3588.
#
# Same discipline as build_rknn_heatmap_arm64.sh:
#   * no OpenCV; stb/hand-rolled image IO only
#   * link the BOARD's own librga.so.2 -- never the SDK's stale copies.
#     The SDK debs do NOT match what the board runs:
#        board  : /usr/lib/aarch64-linux-gnu/librga.so.2.1.0
#                 192712 B  md5 7136ae5469eb217f068ef3efc6f26d53
#        SDK 2.1: 94032 B  md5 6c42f98c9e8b1de3d4fe87d3cec644a3   (MISMATCH)
#        SDK 2.2: 114408 B md5 01e9c74111d44d7750c5327972609600  (MISMATCH)
#     So the library is staged from the board into RT_LIB_DIR and the RGA
#     headers are staged from the board into .build-include/rga.
#   * assert the DT_NEEDED entry is the plain SONAME (librga.so.2), not a path
#   * -ffp-contract=off + explicit fast-math rejection
#   * headers staged privately; never add -I/usr/include (rules.md 2b)
#
# Usage: ./build_rga_probe_arm64.sh
# ===========================================================================
set -euo pipefail

# readelf localises its "Shared library:" label; force the C locale so the
# DT_NEEDED assertion below can match it.
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"
STAGE="$HERE/.build-include"

TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
RT_LIB_DIR="${RT_LIB_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu}"
RGA_HDR_DIR="${RGA_HDR_DIR:-$STAGE/rga}"

CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
READELF="$TC/bin/aarch64-rockchip-linux-gnu-readelf"
SRC="$HERE/rga_probe.cpp"
BIN="$OUT/rga_probe"
FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT"
[ -x "$CXX" ] || die "cross compiler missing: $CXX"
[ -f "$SRC" ] || die "missing source: $SRC"

# ---- staged RGA headers (the board's own set) ------------------------------
[ -d "$RGA_HDR_DIR" ] || die \
  "RGA headers not staged in $RGA_HDR_DIR.
   Fetch them from the board (they differ from the SDK debs):
     mkdir -p $RGA_HDR_DIR
     ssh root@192.168.0.64 'cd /usr/include/rga && tar cf - .' | tar xf - -C $RGA_HDR_DIR"
for h in im2d.h im2d_type.h im2d_single.h im2d_task.h im2d_buffer.h im2d_common.h im2d_version.h \
         im2d_expand.h im2d_mpi.h rga.h; do
  [ -f "$RGA_HDR_DIR/$h" ] || die "missing staged RGA header: $RGA_HDR_DIR/$h"
done

# ---- staged RGA library (the board's own build) ----------------------------
[ -f "$RT_LIB_DIR/librga.so" ] || die \
  "board librga.so not found in $RT_LIB_DIR -- stage it first:
     ssh root@192.168.0.64 'cat /usr/lib/aarch64-linux-gnu/librga.so.2.1.0' > $RT_LIB_DIR/librga.so.2.1.0
     ln -sf librga.so.2.1.0 $RT_LIB_DIR/librga.so"

echo "== cross-compiling rga_probe.cpp -> aarch64 =="
# gnu++17 (not c++17): librga's imcheck() macro uses GNU statement expressions.
"$CXX" -std=gnu++17 -O2 -Wall -Wextra \
  $FP_FLAGS \
  -I"$RGA_HDR_DIR" \
  "$SRC" -o "$BIN" \
  -L"$RT_LIB_DIR" -lrga -Wl,-rpath-link,"$RT_LIB_DIR" -lpthread -ldl -lm

echo
echo "[ok] $BIN"
file "$BIN" | cut -d, -f1-3
echo "    NEEDED:"
"$READELF" -d "$BIN" | grep NEEDED | sed 's/^/      /'

# Strict SONAME assertion: the dependency must be the bare SONAME, not a path.
if ! "$READELF" -d "$BIN" | grep -qE 'Shared library: \[librga\.so\.2\]'; then
  die "DT_NEEDED does not contain the plain SONAME librga.so.2"
fi
echo "    [ok] resolves librga.so.2 via the board's own runtime"

echo "    md5: $(md5sum "$BIN" | cut -d' ' -f1)"
echo
echo "    copy to the board with: scp -P 22 $BIN root@192.168.0.64:/root/"
