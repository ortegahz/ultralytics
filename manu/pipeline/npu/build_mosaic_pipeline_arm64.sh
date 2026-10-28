#!/usr/bin/env bash
# ===========================================================================
# build_mosaic_pipeline_arm64.sh -- cross-compile the mosaic pipeline LIBRARY
# and the demo that drives it, for the RK3588.
#
# Deliverables (all aarch64, in manu/pipeline/npu/out/):
#
#   libmosaic_pipeline.a    the library: GMC/OpenCL + RGA mosaic + RKNN fp16
#                           + back-projection, behind the C ABI in
#                           mosaic_pipeline.h
#   rknn_mosaic_e2e         the TEST PROGRAM: calls ONLY the public C ABI,
#                           drives a raw infrared sequence, writes the TSV
#   rknn_mosaic_heatmap     the refactored BMP-driven binary, rebuilt here so
#                           the refactor is covered by the same compile
#
# DISCIPLINE (inherited from build_rknn_mosaic_arm64.sh and
# build_gmc_ocl_arm64.sh, both of which are certified):
#
#   * no OpenCV in the mosaic binary; the library links the CROSS-BUILT
#     OpenCV 4.10.0 (statically), because the GMC engine's primitives
#     (Shi-Tomasi / pyramidal LK / RANSAC / warpAffine / resize) must be the
#     SAME C++ translation units the Python reference calls -- that is what
#     makes the port bit-exact, not "careful writing".
#   * link the BOARD's own librknnrt.so (2.3.0) and librga.so.2 (2.1.0), and
#     assert the DT_NEEDED entries are the plain SONAMEs.
#   * NEVER -I/usr/include: the host headers are x86 and would poison the
#     build (rules.md 2b).  The Khronos CL headers are staged privately.
#   * -ffp-contract=off + explicit fast-math rejection.
#   * export LC_ALL=C: readelf localises its "Shared library:" label, and a
#     Chinese label makes the DT_NEEDED assertions silently pass/fail.
#
# Usage: ./build_mosaic_pipeline_arm64.sh
# ===========================================================================
set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$(cd "$HERE/.." && pwd)"
OUT="$HERE/out"
STAGE="$HERE/.build-include"

SDK="${RKNN_SDK:-/media/manu/1TB-Volume/rk3588/rk3588_sdk/EVM3588-A_20250815/02Linux软件资料/SDK&Image/SDK/evm3588_linux_sdk_v2.1.1.0/external/rknpu2}"
TC="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"
INSTALL="${OPENCV_INSTALL:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install}"
LIBDIR="${BOARD_LIBS:-/media/manu/1TB-Volume/rk3588/boardlibs}"
RT_LIB_DIR="${RT_LIB_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu}"

CXX="$TC/bin/aarch64-rockchip-linux-gnu-g++"
READELF="$TC/bin/aarch64-rockchip-linux-gnu-readelf"
AR="$TC/bin/aarch64-rockchip-linux-gnu-ar"

FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"
TP_LIB="$INSTALL/lib/opencv4/3rdparty"
OPENCV_LIBS="-lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core -ltegra_hal -lzlib"

die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT"
[ -x "$CXX" ] || die "cross compiler missing: $CXX"
[ -x "$AR" ] || die "cross ar missing: $AR"
[ -d "$INSTALL/lib" ] || die "cross OpenCV missing (run build_opencv_arm64.sh)"
[ -f "$LIBDIR/libOpenCL.so" ] || die "board libOpenCL not staged"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "board librknnrt.so not staged in $RT_LIB_DIR"
[ -f "$RT_LIB_DIR/librga.so" ] || die "board librga.so not staged in $RT_LIB_DIR"

# ---- staged headers -------------------------------------------------------
for h in rknn_api.h; do
  [ -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/runtime/RK3588/Linux/librknn_api/include/$h" "$STAGE/$h"
done
for h in stb_image.h; do
  [ -f "$SDK/examples/3rdparty/stb/$h" ] || die "SDK header not found: $h"
  cp -f "$SDK/examples/3rdparty/stb/$h" "$STAGE/$h"
done
for h in im2d.h im2d_type.h im2d_single.h im2d_buffer.h im2d_common.h im2d_version.h \
         im2d_expand.h im2d_mpi.h rga.h RgaUtils.h; do
  [ -f "$STAGE/rga/$h" ] || die "missing staged RGA header: $STAGE/rga/$h"
done
[ -d "$STAGE/CL" ] || die "Khronos CL headers not staged in $STAGE/CL"
echo "[stage] headers ready"

# ---------------------------------------------------------------------------
# 1. the LIBRARY
# ---------------------------------------------------------------------------
LIB_CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  $FP_FLAGS -DCL_TARGET_OPENCL_VERSION=300 \
  -I$STAGE -I$STAGE/rga -I$PIPE/cpp -I$PIPE/opencl/host -I$INSTALL/include/opencv4"

echo "== compiling libmosaic_pipeline.a =="
"$CXX" $LIB_CXXFLAGS -c "$HERE/mosaic_pipeline.cpp" -o "$OUT/mosaic_pipeline.o"
rm -f "$OUT/libmosaic_pipeline.a"
"$AR" rcs "$OUT/libmosaic_pipeline.a" "$OUT/mosaic_pipeline.o"
echo "[ok] $OUT/libmosaic_pipeline.a ($(stat -c%s "$OUT/libmosaic_pipeline.a") B)"

# ---------------------------------------------------------------------------
# 2. the TEST PROGRAM (links the library, calls only the public C ABI)
# ---------------------------------------------------------------------------
echo "== linking rknn_mosaic_e2e =="
"$CXX" $LIB_CXXFLAGS "$HERE/rknn_mosaic_e2e.cpp" -o "$OUT/rknn_mosaic_e2e" \
  "$OUT/libmosaic_pipeline.a" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm
echo "[ok] $OUT/rknn_mosaic_e2e"

# ---------------------------------------------------------------------------
# 2b. the API contract test (links the library, touches only the public ABI)
# ---------------------------------------------------------------------------
echo "== building mosaic_api_test (library contract test) =="
"$CXX" -std=gnu++17 -O2 -Wall -Wextra \
  $FP_FLAGS \
  -I"$STAGE" -I"$STAGE/rga" -I"$PIPE/cpp" -I"$PIPE/opencl/host" -I"$INSTALL/include/opencv4" \
  "$HERE/mosaic_api_test.cpp" "$OUT/libmosaic_pipeline.a" -o "$OUT/mosaic_api_test" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm
echo "[ok] $OUT/mosaic_api_test"

# ---------------------------------------------------------------------------
# 3. the refactored BMP-driven binary (same core, no OpenCV)
# ---------------------------------------------------------------------------
echo "== rebuilding rknn_mosaic_heatmap (mosaic core, no OpenCV) =="
"$CXX" -std=gnu++17 -O2 -Wall -Wextra \
  $FP_FLAGS \
  -I"$STAGE" -I"$STAGE/rga" \
  "$HERE/rknn_mosaic_heatmap.cpp" -o "$OUT/rknn_mosaic_heatmap" \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" -lpthread -ldl -lm
echo "[ok] $OUT/rknn_mosaic_heatmap"

# ---------------------------------------------------------------------------
# DT_NEEDED assertions
# ---------------------------------------------------------------------------
check_needed() {
  local bin="$1"
  echo
  echo "== $bin =="
  file "$bin" | cut -d, -f1-3
  "$READELF" -d "$bin" | grep NEEDED | sed 's/^/    /'
  if ! "$READELF" -d "$bin" | grep -qE 'Shared library: \[librknnrt\.so\]'; then
    die "$bin: DT_NEEDED does not contain the plain SONAME librknnrt.so"
  fi
  if ! "$READELF" -d "$bin" | grep -qE 'Shared library: \[librga\.so\.2\]'; then
    die "$bin: DT_NEEDED does not contain the plain SONAME librga.so.2"
  fi
  if "$READELF" -d "$bin" | grep -qE 'Shared library: \[libc\+\+\.so'; then
    die "$bin: DT_NEEDED contains libc++ -- a host-runtime difference, refuse to ship"
  fi
  echo "    [ok] librknnrt.so + librga.so.2 via the board's own runtime, no libc++"
  echo "    md5: $(md5sum "$bin" | cut -d' ' -f1)"
}

check_needed "$OUT/rknn_mosaic_e2e"
check_needed "$OUT/mosaic_api_test"
check_needed "$OUT/rknn_mosaic_heatmap"

echo
echo "== deploy to the board =="
# ACTUALLY deploy.  This block used to only ECHO the scp lines, so a "build +
# deploy" run silently left the OLD binaries on the board -- which is how a
# stale, flaky contract test kept being reported as "42 passed, 0 failed".
# The deploy is best-effort: a board that is unreachable must not destroy a
# successful build, so failures warn instead of aborting.
BOARD="${MOSAIC_BOARD:-root@192.168.0.64}"
BOARD_PORT="${MOSAIC_BOARD_PORT:-22}"
if [ "${MOSAIC_NO_DEPLOY:-0}" = "1" ]; then
  echo "  [skip] MOSAIC_NO_DEPLOY=1"
else
  if scp -P "$BOARD_PORT" "$OUT/libmosaic_pipeline.a" "$OUT/rknn_mosaic_e2e" \
             "$OUT/mosaic_api_test" "$BOARD:/root/" && \
     scp -P "$BOARD_PORT" "$OUT/rknn_mosaic_heatmap" "$BOARD:/root/"; then
    echo "  [ok] deployed to $BOARD:/root/"
    echo "  board md5:"
    ssh -p "$BOARD_PORT" "$BOARD" 'md5sum /root/libmosaic_pipeline.a /root/rknn_mosaic_e2e /root/mosaic_api_test /root/rknn_mosaic_heatmap' | sed 's/^/    /'
  else
    echo "  [WARN] deploy failed (board unreachable?); the build itself is fine" >&2
  fi
fi

# ---------------------------------------------------------------------------
# 4. the GMC GPU PROBE (standalone measurement binary)
# ---------------------------------------------------------------------------
# APPENDED, 2026-10-27.  This section adds ONE new target and changes NOTHING
# above it: libmosaic_pipeline.a / rknn_mosaic_e2e / mosaic_api_test /
# rknn_mosaic_heatmap keep their exact compile flags and their exact
# artifacts.
#
# gmc_gpu_probe is a PROBE, not a pipeline component:
#   * it links NOBODY from libmosaic_pipeline.a (it is self-contained),
#   * it is not called by the pipeline, it does not touch the e2e TSV,
#   * it exists to measure "GMC feature prep on Mali-G610: speed and accuracy
#     cost" -- see manu/pipeline/npu/gmc_gpu_probe.cpp.
#
# The probe needs OpenCV (the CPU reference arm: goodFeaturesToTrack /
# calcOpticalFlowPyrLK / estimateAffinePartial2D) and OpenCL (the GPU arm).
# Same $FP_FLAGS discipline as the library: -ffp-contract=off, no fast-math.
echo
echo "== building gmc_gpu_probe (standalone GMC feature-prep GPU probe) =="
"$CXX" $LIB_CXXFLAGS "$HERE/gmc_gpu_probe.cpp" -o "$OUT/gmc_gpu_probe" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$TP_LIB" $OPENCV_LIBS -lOpenCL -lpthread -ldl -lm
echo "[ok] $OUT/gmc_gpu_probe"
file "$OUT/gmc_gpu_probe" | cut -d, -f1-3
echo "    md5: $(md5sum "$OUT/gmc_gpu_probe" | cut -d' ' -f1)"
# The probe is NOT auto-deployed: the deploy block above belongs to the
# certified artifacts and must stay byte-identical in behaviour.  Deploy it
# explicitly when measuring:
#   scp -P 22 $OUT/gmc_gpu_probe root@192.168.0.64:/root/
#   scp -P 22 ../opencl/kernels/gmc_feature_gpu.cl root@192.168.0.64:/root/kernels/
