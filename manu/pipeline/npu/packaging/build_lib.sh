#!/usr/bin/env bash
# ===========================================================================
# build_lib.sh -- cross-compile ONLY the mosaic pipeline LIBRARY for RK3588.
#
#   output: out/libmosaic_pipeline.so      (共享库, SONAME=libmosaic_pipeline.so)
#
# ---------------------------------------------------------------------------
# 【2026-11-11 新板交付版】与上一版的根本区别
#
#   上一版：交付包自带 third_party/（交叉编译版 OpenCV **静态库** .a + 板卡
#           libOpenCL.so / librknnrt.so / librga.so 的副本）⇒ 包是"静态自包含"。
#   本版： **交付包不再自带任何第三方 .so / .a**。
#           OpenCV / RKNN Runtime / RGA / OpenCL 全部**动态链接板卡自带的库**
#           （嵌入式组的要求：直接用板卡上原来就有的 .so）。
#           ⇒ 包里只剩：源码 + 第三方头 + kernel 资产 + 构建脚本 + 预编译产物。
#           ⇒ 库本身也改成**共享库 .so** 交付（不再是静态库 .a）。
#
#   ⚠️ 头与运行库必须**版本严格配套**：本包用板卡 /usr/include/opencv4 的头
#      （OpenCV 4.6.0）去链接板卡 /usr/lib/aarch64-linux-gnu 的 libopencv_*.so.406
#      —— 头与 .so 同版本，SONAME 天然一致（libopencv_core.so.406）。
#      不要拿 4.8 的运行库去配 4.6 的头（OpenCV 次版本会改 C++ ABI）。
#
#   * -ffp-contract=off + 显式拒绝 fast-math（位级可复现移植）。
#   * NEVER -I/usr/include: 宿主机的头是 x86 的，会污染构建。
#     ⚠️ 这里的 "板卡默认路径" 指**目标板卡**的文件系统；
#        交叉编译在宿主机上做时，必须把它们指向**从板卡取回来的那份副本**
#        （见 fetch_board_libs.sh），不能真用宿主机的 /usr。
#   * export LC_ALL=C: readelf 本地化 "Shared library:" 标签。
#
# 用法: ./build_lib.sh
# ===========================================================================
set -euo pipefail
export LC_ALL=C

die() { echo "[ERROR] $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/out"

# ---------------------------------------------------------------------------
# 依赖定位 —— 全部指向**板卡自带的库**，包内不再有任何 third_party 副本。
#
#   OPENCV_INCLUDE_DIR  板卡 OpenCV **头**目录（含 opencv2/）
#                        板卡默认 /usr/include/opencv4
#   OPENCV_LIB_DIR      板卡 OpenCV **动态库**目录（libopencv_*.so.406）
#                        板卡默认 /usr/lib/aarch64-linux-gnu
#   BOARD_LIBS          板卡 libOpenCL.so.1 / librga.so 所在目录
#                        板卡默认 /usr/lib/aarch64-linux-gnu
#   RT_LIB_DIR          板卡 librknnrt.so 所在目录
#                        板卡默认 /usr/lib
#   EXTRA_LIB_DIRS      额外链接搜索路径（空格分隔），用于放 OpenCV 的传递依赖
#                        （libjpeg/libpng/libtiff/libgdal/...），交叉链接时需要
#
# 换板卡 / 换 BSP 时用上面这些环境变量覆盖即可，不用改脚本。
# ---------------------------------------------------------------------------

# 板卡上的多架构库目录不固定（aarch64-linux-gnu / 其它），自动探测一次。
detect_multiarch() {
  for d in /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib; do
    [ -d "$d" ] && { echo "$d"; return 0; }
  done
  echo /usr/lib
}

BOARD_SYS_LIBDIR="$(detect_multiarch)"

OPENCV_INCLUDE_DIR="${OPENCV_INCLUDE_DIR:-/usr/include/opencv4}"
OPENCV_LIB_DIR="${OPENCV_LIB_DIR:-$BOARD_SYS_LIBDIR}"
BOARD_LIBS="${BOARD_LIBS:-$BOARD_SYS_LIBDIR}"
RT_LIB_DIR="${RT_LIB_DIR:-/usr/lib}"
EXTRA_LIB_DIRS="${EXTRA_LIB_DIRS:-}"

# 兼容旧变量名：OPENCV_INSTALL=<根>/... 时拆成 头目录 + 库目录
if [ -n "${OPENCV_INSTALL:-}" ]; then
  [ -n "${OPENCV_INCLUDE_DIR:-}" ] || OPENCV_INCLUDE_DIR="$OPENCV_INSTALL/include/opencv4"
  [ -n "${OPENCV_LIB_DIR:-}" ]    || OPENCV_LIB_DIR="$OPENCV_INSTALL/lib"
fi

# ---------------------------------------------------------------------------
# 交叉编译器（不在包里 —— 编译器没法随包发）
# ---------------------------------------------------------------------------
TC_DEFAULT="/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu"
TC_CANDIDATES="${RK3588_TOOLCHAIN:-$TC_DEFAULT}"

# 工具链里的编译器名字各家不一样（rockchip / none / 通用 aarch64-linux-gnu）。
# ⚠️ 光看「文件在不在」不够：buildroot 的 `*-g++` 是指向 `*.br_real` 的包装器，
#    改名/错链之后驱动自己就找不到后端 ⇒ 编译时才炸，报错还很难懂。
#    ⇒ 必须**真的跑一下**（预处理一个空文件：能过 = 驱动 + 前端 + 头路径都通）。
cxx_works() { [ -x "$1" ] && "$1" -E -x c++ - </dev/null >/dev/null 2>&1; }

find_cxx() {
  local tc="$1" n c
  for n in aarch64-rockchip-linux-gnu-g++ aarch64-none-linux-gnu-g++ aarch64-linux-gnu-g++; do
    if cxx_works "$tc/bin/$n"; then echo "$tc/bin/$n"; return 0; fi
  done
  # 兜底：bin 下任何 *-g++，且能真的跑起来
  while IFS= read -r c; do
    if cxx_works "$c"; then echo "$c"; return 0; fi
  done < <(find "$tc/bin" -maxdepth 1 -name '*-g++' -type f 2>/dev/null | LC_ALL=C sort)
  return 1
}
find_ar() {
  local tc="$1" n c
  for n in aarch64-rockchip-linux-gnu-ar aarch64-none-linux-gnu-ar aarch64-linux-gnu-ar; do
    [ -x "$tc/bin/$n" ] && { echo "$tc/bin/$n"; return 0; }
  done
  c="$(find "$tc/bin" -maxdepth 1 -name '*-ar' -type f 2>/dev/null | LC_ALL=C sort | head -1)"
  [ -n "$c" ] && { echo "$c"; return 0; }
  return 1
}

TC=""
IFS=':' read -r -a _tc_list <<< "$TC_CANDIDATES"
for _tc in "${_tc_list[@]}"; do
  [ -n "$_tc" ] || continue
  if find_cxx "$_tc" >/dev/null 2>&1; then TC="$_tc"; break; fi
done
[ -n "$TC" ] || die "找不到**可用**的交叉编译器。
       候选路径：$TC_CANDIDATES
       ⇒ 「可用」= 编译器能真的跑起来（驱动 + 前端 + 头路径都通）；
         目录里有 g++ 但跑不起来（常见于 buildroot 的 *.br_real 包装器被改名/错链）不算。
       ⇒ 请设置 RK3588_TOOLCHAIN=<工具链根目录>（可用 ':' 分隔多个，按顺序试），
         或直接设 CROSS_CXX=<aarch64 交叉 g++ 的绝对路径>。"

CXX="${CROSS_CXX:-}"
[ -n "$CXX" ] || CXX="$(find_cxx "$TC")"
AR="${CROSS_AR:-}"
[ -n "$AR" ] || AR="$(find_ar "$TC" || echo "$TC/bin/ar")"
[ -x "$CXX" ] || die "cross compiler 不可用：$CXX"
[ -x "$AR" ]  || die "cross ar 不可用：$AR"

FP_FLAGS="-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

# 板卡 OpenCV 的模块（**共享库**，不再是静态 .a）
OPENCV_LIBS="-lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_imgproc -lopencv_core"

# 链接搜索路径：板卡自带库 + 额外的依赖闭包目录
LIB_PATHS=(-L"$RT_LIB_DIR" -L"$BOARD_LIBS" -L"$OPENCV_LIB_DIR")
for d in $EXTRA_LIB_DIRS; do LIB_PATHS+=(-L"$d"); done
RPATH_LINKS=(-Wl,-rpath-link,"$RT_LIB_DIR" -Wl,-rpath-link,"$BOARD_LIBS" -Wl,-rpath-link,"$OPENCV_LIB_DIR")

# 板卡 OpenCV 是共享库，它自己还链着一长串传递依赖（libjpeg/libpng/libtiff/
# libgdal/libOpenEXR/libtbb/...）。那些依赖**只在板卡运行时由 ldconfig 解析**，
# 交叉链接时不必（也不该）把它们全搬到宿主机 ⇒ 允许共享库带未定义符号。
#   ⇒ 集成方只需提供板卡那 7 个 libopencv_*.so + 3 个板卡库即可链接。
RELAX="-Wl,--allow-shlib-undefined"

# ---------------------------------------------------------------------------
# 0. environment gate -- fail early and loudly, never half-build
# ---------------------------------------------------------------------------
[ -d "$OPENCV_INCLUDE_DIR/opencv2" ] || die "找不到板卡 OpenCV 头：$OPENCV_INCLUDE_DIR/opencv2
       ⇒ 这应当是**板卡**的 OpenCV 头（板卡上为 /usr/include/opencv4）。
         交叉编译时必须指向**从板卡取回的那份副本**（fetch_board_libs.sh），
         不能用宿主机的 /usr（x86 的头会污染构建）。
         可用 OPENCV_INCLUDE_DIR=<目录> 覆盖。"

_have_cv=0
for m in core imgproc imgcodecs calib3d features2d flann video; do
  if ls "$OPENCV_LIB_DIR"/libopencv_$m.so* >/dev/null 2>&1; then
    _have_cv=$((_have_cv+1))
  fi
done
[ "$_have_cv" -eq 7 ] || die "板卡 OpenCV 动态库不全（找到 $_have_cv/7 个模块）：$OPENCV_LIB_DIR
       ⇒ 需要 libopencv_{core,imgproc,imgcodecs,calib3d,features2d,flann,video}.so*
         可用 OPENCV_LIB_DIR=<目录> 覆盖。"

[ -f "$BOARD_LIBS/libOpenCL.so" ] || [ -f "$BOARD_LIBS/libOpenCL.so.1" ] || die "找不到板卡 libOpenCL：$BOARD_LIBS
       ⇒ 可用 BOARD_LIBS=<目录> 覆盖。"
[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "找不到板卡 librknnrt.so：$RT_LIB_DIR
       ⇒ 可用 RT_LIB_DIR=<目录> 覆盖。"
[ -f "$BOARD_LIBS/librga.so" ] || [ -f "$BOARD_LIBS/librga.so.2" ] || die "找不到板卡 librga：$BOARD_LIBS
       ⇒ 可用 BOARD_LIBS=<目录> 覆盖。"
[ -f "$HERE/src/npu/mosaic_pipeline.cpp" ] || die "missing source: src/npu/mosaic_pipeline.cpp"

echo "  工具链   : $TC"
echo "  编译器   : $CXX"
echo "  OpenCV头 : $OPENCV_INCLUDE_DIR"
echo "  OpenCV库 : $OPENCV_LIB_DIR"
echo "  板卡库   : $BOARD_LIBS"
echo "  RKNN 库  : $RT_LIB_DIR"
[ -n "$EXTRA_LIB_DIRS" ] && echo "  额外路径 : $EXTRA_LIB_DIRS"

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# 1. compile + link the SHARED library
#
#    -fPIC            : 共享库必需
#    -soname          : libmosaic_pipeline.so（集成方按这个名字链接）
#    链接的是**板卡自带的**动态库 ⇒ 交付包里不需要带任何 .so
# ---------------------------------------------------------------------------
LIB_CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  $FP_FLAGS -fPIC -DCL_TARGET_OPENCL_VERSION=300 \
  -I$HERE/include -I$HERE/include/rga -I$HERE/src/cpp -I$HERE/src/opencl/host -I$OPENCV_INCLUDE_DIR"

echo "== compiling libmosaic_pipeline.so =="
"$CXX" $LIB_CXXFLAGS -c "$HERE/src/npu/mosaic_pipeline.cpp" -o "$OUT/mosaic_pipeline.o"

echo "== linking  libmosaic_pipeline.so =="
rm -f "$OUT/libmosaic_pipeline.so"
"$CXX" -shared -Wl,-soname,libmosaic_pipeline.so $RELAX \
  -o "$OUT/libmosaic_pipeline.so" "$OUT/mosaic_pipeline.o" \
  "${LIB_PATHS[@]}" "${RPATH_LINKS[@]}" \
  $OPENCV_LIBS -lOpenCL -lrknnrt -lrga -lpthread -ldl -lm

echo
echo "== artifact =="
echo "  path : $OUT/libmosaic_pipeline.so"
echo "  size : $(stat -c%s "$OUT/libmosaic_pipeline.so") B"
echo "  md5  : $(md5sum "$OUT/libmosaic_pipeline.so" | cut -d' ' -f1)"
echo
echo "  -- 运行时依赖（全部为板卡自带库，板卡 ldconfig 可解析）--"
readelf -d "$OUT/libmosaic_pipeline.so" | grep NEEDED | sed 's/.*\[/    /;s/\]//'
echo
echo "[ok] shared library built.  Next: bash build_demo.sh"
