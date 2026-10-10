#!/usr/bin/env bash
# ===========================================================================
# build_opencv_arm64.sh -- cross-compile OpenCV 4.10.0 for the RK3588 board.
#
# Why this exists
# ---------------
# The feature-channel work needs the CPU half of the pipeline (GMC fit, feature
# detection, LK flow, warpAffine resize) to run on the board next to the fused
# OpenCL kernel. The board has no OpenCV: apt offers only 4.2.0, eight minor
# versions behind the x86 reference, and installing it would silently change the
# numerics that everything else was validated against. So we build the *same
# version* from source instead.
#
# Build scope
# -----------
# The sysroot has 334 shared objects but every one of them is a gconv
# character-set module -- no zlib, libpng, libjpeg or libtiff headers exist, so
# imgcodecs cannot be built against any of them. JPEG/PNG/TIFF/WebP stay off.
#
# BMP is the escape hatch: OpenCV's BMP codec (modules/imgcodecs/src/grfmt_bmp.cpp)
# is collected by an unconditional `file(GLOB ... grfmt*.cpp)` and includes only
# its own two headers -- no zlib, no SIMD, no external library. BMP is also
# uncompressed, so both machines read the identical bytes instead of each
# running a different libjpeg-turbo build. That matters here: the board ships
# libjpeg-turbo 1.5.2 (.so.62) and 2.0.3 (.so.8), x86 has 2.1.5, and a lossy
# decoder difference at the input would invalidate every downstream comparison.
# Sequence frames are therefore converted to BMP on x86 and read back with
# cv::imread on the board.
#
# WITH_OPENJPEG must stay OFF. imgcodecs compiles grfmt_jpeg2000_openjpeg.cpp
# unconditionally once the module is in BUILD_LIST, and OpenCV's own bundled
# OpenJPEG is not linked into the static libs -- so leaving it on produces an
# imagecodecs.a full of unresolved opj_* symbols that only surface at link time
# in programs that call imread, not in the module list. BMP needs none of it.
#
# The remaining modules are those the pipeline actually calls:
#
#   core        median()
#   imgproc     warpAffine(), resize()
#   imgcodecs   imread()/imwrite() for the BMP sequence (no external codec)
#   features2d  goodFeaturesToTrack()
#   video       calcOpticalFlowPyrLK()
#   calib3d     estimateAffinePartial2D()
#   flann       KNN indexer used internally by features2d/video
#
# Usage: ./build_opencv_arm64.sh [jobs]
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${1:-4}"

SRC="${OPENCV_SRC:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64}"
BUILD="${OPENCV_BUILD:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-build}"
INSTALL="${OPENCV_INSTALL:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install}"
TOOLCHAIN="$HERE/rk3588-aarch64-toolchain.cmake"

# The system CMake is 4.x, which hard-errors on OpenCV 4.10's
# `cmake_minimum_required(VERSION 3.1)`. A local 3.x is used instead of patching
# the worktree, so the source stays byte-identical to the 4.10.0 tag.
CMAKE="${CMAKE:-/media/manu/1TB-Volume/workspace/tools/cmake-3.31.6-linux-x86_64/bin/cmake}"
NINJA="${NINJA:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu/bin/other_exe/ninja}"

die() { echo "[ERROR] $*" >&2; exit 1; }

[ -x "$CMAKE" ] || die "cmake 3.x not found: $CMAKE (see script header)"
[ -x "$NINJA" ] || die "ninja not found: $NINJA"
[ -f "$TOOLCHAIN" ] || die "missing toolchain file: $TOOLCHAIN"
[ -f "$SRC/CMakeLists.txt" ] || die "missing OpenCV source: $SRC"

# Refuse to build anything that is not exactly 4.10.0. "Close enough" here
# would invalidate every downstream comparison against the x86 reference.
VER="$(sed -n 's/^#define CV_VERSION_REVISION  *\([0-9]*\).*/\1/p' \
        "$SRC/modules/core/include/opencv2/core/version.hpp")"
MAJ="$(sed -n 's/^#define CV_VERSION_MAJOR  *\([0-9]*\).*/\1/p' \
        "$SRC/modules/core/include/opencv2/core/version.hpp")"
MIN="$(sed -n 's/^#define CV_VERSION_MINOR  *\([0-9]*\).*/\1/p' \
        "$SRC/modules/core/include/opencv2/core/version.hpp")"
[ "$MAJ.$MIN.$VER" = "4.10.0" ] || die "source tree is $MAJ.$MIN.$VER, expected 4.10.0"

mkdir -p "$BUILD"

"$CMAKE" -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DCMAKE_MAKE_PROGRAM="$NINJA" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL" \
  -DBUILD_LIST=core,imgproc,imgcodecs,features2d,video,calib3d,flann \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCS=OFF \
  -DBUILD_opencv_java=OFF \
  -DBUILD_opencv_python2=OFF \
  -DBUILD_opencv_python3=OFF \
  -DBUILD_PROTOBUF=OFF \
  -DWITH_IPP=OFF \
  -DWITH_OPENCL=OFF \
  -DWITH_FFMPEG=OFF \
  -DWITH_GTK=OFF \
  -DWITH_QT=OFF \
  -DWITH_1394=OFF \
  -DWITH_V4L=OFF \
  -DWITH_GSTREAMER=OFF \
  -DWITH_JPEG=OFF \
  -DWITH_PNG=OFF \
  -DWITH_TIFF=OFF \
  -DWITH_WEBP=OFF \
  -DWITH_OPENEXR=OFF \
  -DWITH_JASPER=OFF \
  -DWITH_OPENJPEG=OFF \
  -DWITH_LAPACK=OFF \
  -DWITH_EIGEN=OFF \
  -DWITH_TBB=OFF \
  -DWITH_ITT=OFF \
  -DWITH_OPENCL_D3D11_NV=OFF \
  -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DENABLE_FAST_MATH=OFF \
  -DCV_DISABLE_OPTIMIZATION=OFF \
  -DCV_TRACE=OFF \
  -DCV_ENABLE_INTRINSICS=ON \
  -DBUILD_WITH_DEBUG_INFO=OFF \
  -DCV_BUILD_TESTS=OFF

echo "[build] ninja -j$JOBS  (low job count on purpose: this host has 5 GiB free)"
"$CMAKE" --build "$BUILD" -- -j"$JOBS"

"$CMAKE" --install "$BUILD"

echo
echo "[ok] installed to $INSTALL"
ls -la "$INSTALL/lib/"*.a 2>/dev/null || true
