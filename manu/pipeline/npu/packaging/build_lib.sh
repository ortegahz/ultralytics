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
#   ⚠️ 头与运行库必须**版本严格配套**（新板实测，2026-11-25）：
#      板上同时存在**两套 OpenCV**：
#        /usr/lib/libopencv_*.so.408                    ← 4.8（板上 app 与本 SDK 链的）
#        /usr/lib/aarch64-linux-gnu/libopencv_*.so.406   ← 4.6（Debian 系统包）
#      而板上的**头**只有 4.6（/usr/include/opencv4）⇒ 4.8 的头必须外部提供
#      （本包 include/opencv4 就是随包发布的那份 4.8 头）。
#      ⇒ 所以：运行库目录按 **SONAME 后缀（ABI）** 选，不按"哪个目录存在"选；
#        头版本与 ABI 的配套关系**当场校验**，不配套就 die。
#        （选错目录 ⇒ 同一进程里同时装进 4.6 和 4.8 两套 OpenCV，cv::Mat 这类
#          C++ 类型跨 .so 边界串味 ⇒ 事故。）
#
#   ⚠️ OpenCL：新板（Debian 12）只装了 **ICD loader**
#      /usr/lib/aarch64-linux-gnu/libOpenCL.so.1，**没有 libOpenCL.so 开发软链**
#      ⇒ -lOpenCL 必然链不出来。脚本会退化成**按绝对路径**直接链接那个版本化
#      文件（它的 SONAME 就是 libOpenCL.so.1 ⇒ 产物 NEEDED 与 -lOpenCL 一致）。
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
#   OPENCV_ABI          OpenCV 的 **SONAME 后缀**（408 = 4.8，406 = 4.6，
#                        410 = 4.10）。默认 408。
#   OPENCV_VERSION      期望的**头**版本，形如 4.8；默认由 OPENCV_ABI 推导。
#   OPENCV_INCLUDE_DIR  OpenCV **头**目录（含 opencv2/）。默认优先级：
#                        ① 用户显式指定 ② $HERE/include/opencv4（包内随包发布
#                        的 4.8 头）③ /usr/include/opencv4
#   OPENCV_LIB_DIR      OpenCV **动态库**目录（libopencv_*.so.$OPENCV_ABI）。
#                        默认按 ABI 自动探测（见 detect_opencv_lib_dir）。
#   BOARD_LIBS          板卡 libOpenCL.so.1 / librga.so 所在目录
#                        板卡默认 /usr/lib/aarch64-linux-gnu
#   RT_LIB_DIR          板卡 librknnrt.so 所在目录
#                        板卡默认 /usr/lib
#   EXTRA_LIB_DIRS      额外链接搜索路径（空格分隔），用于放 OpenCV 的传递依赖
#                        （libjpeg/libpng/libtiff/libgdal/...），交叉链接时需要
#
# 换板卡 / 换 BSP 时用上面这些环境变量覆盖即可，不用改脚本。
# ---------------------------------------------------------------------------

# ---- OpenCV 版本：ABI 是唯一真相来源 --------------------------------------
# OpenCV 的 SONAME 后缀 = major*100 + minor（4.6 → 406，4.8 → 408，4.10 → 410）
# ⇒ ABI 能唯一反推版本号 ⇒ 头校验与选目录共用同一套数字，不会各说各话。
OPENCV_ABI="${OPENCV_ABI:-408}"
case "$OPENCV_ABI" in
  ''|*[!0-9]*) die "OPENCV_ABI 必须是纯数字（OpenCV 的 SONAME 后缀，如 408/406/410）：$OPENCV_ABI" ;;
esac
_cv_abi_num=$((10#$OPENCV_ABI))
[ "$_cv_abi_num" -ge 100 ] || die "OPENCV_ABI 不合法（应为 major*100+minor，如 408）：$OPENCV_ABI"
_cv_abi_major=$((_cv_abi_num / 100))
_cv_abi_minor=$((_cv_abi_num % 100))
# 分工：OPENCV_ABI 决定**运行库**（选目录 + 校验 .so 的 SONAME）；
#       OPENCV_VERSION 决定**头**的期望版本（用户显式设置时以用户为准）。
OPENCV_VERSION="${OPENCV_VERSION:-$_cv_abi_major.$_cv_abi_minor}"

# ---- OpenCV 运行库目录：按 ABI 选，绝不按"目录是否存在"选 ------------------
# 新板上同时存在两套 OpenCV：
#   /usr/lib/libopencv_core.so.408                   ← 4.8（板上 app 与本 SDK 链的）
#   /usr/lib/aarch64-linux-gnu/libopencv_core.so.406  ← 4.6（Debian 系统包）
# 旧写法"哪个 multiarch 目录存在就选哪个"会选到 aarch64-linux-gnu（4.6）⇒
# 同一进程里同时装进 4.6 与 4.8 两套 OpenCV ⇒ cv::Mat 跨 .so 边界串味 ⇒ 事故。
# ⇒ 只接受**恰好含 libopencv_core.so.$OPENCV_ABI** 的目录。
OPENCV_LIB_CANDIDATES="/usr/lib /usr/lib/aarch64-linux-gnu /usr/lib64 /root/mosaic/opencv /home/orangepi/lib"

detect_opencv_lib_dir() {
  local d found
  for d in $OPENCV_LIB_CANDIDATES; do
    [ -d "$d" ] || continue
    [ -f "$d/libopencv_core.so.$OPENCV_ABI" ] && { echo "$d"; return 0; }
  done
  {
    echo "       需要 libopencv_core.so.$OPENCV_ABI（OPENCV_ABI=$OPENCV_ABI）"
    echo "       扫描过的候选目录 / 实际找到的 libopencv_core.so*："
    for d in $OPENCV_LIB_CANDIDATES; do
      if [ -d "$d" ]; then
        found="$(ls -1 "$d"/libopencv_core.so* 2>/dev/null | while IFS= read -r f; do basename "$f"; done | tr '\n' ' ')" || found=""
        echo "         $d : ${found:-(无)}"
      else
        echo "         $d : (目录不存在)"
      fi
    done
  } >&2
  die "按 OpenCV ABI 找不到板卡 OpenCV 运行库。
       ⇒ 板卡上可能只有**另一个次版本**的 OpenCV（例如只有 .so.406，而这里要 .so.408）。
       ⇒ 可用 OPENCV_LIB_DIR=<含 libopencv_core.so.$OPENCV_ABI 的目录> 覆盖，
         或用 OPENCV_ABI=<板卡实际提供的后缀> 换一套配套的头。"
}

# 用户显式指定的（含旧变量名 OPENCV_INSTALL 拆出来的）
_cv_inc_user="${OPENCV_INCLUDE_DIR:-}"
_cv_lib_user="${OPENCV_LIB_DIR:-}"

# 兼容旧变量名：OPENCV_INSTALL=<根>/... 时拆成 头目录 + 库目录
if [ -n "${OPENCV_INSTALL:-}" ]; then
  [ -n "$_cv_inc_user" ] || _cv_inc_user="$OPENCV_INSTALL/include/opencv4"
  [ -n "$_cv_lib_user" ] || _cv_lib_user="$OPENCV_INSTALL/lib"
fi

# 头目录：包内随包发布的 4.8 头优先（板上 /usr/include/opencv4 只有 4.6 的头）
if [ -n "$_cv_inc_user" ]; then
  OPENCV_INCLUDE_DIR="$_cv_inc_user"
elif [ -d "$HERE/include/opencv4" ]; then
  OPENCV_INCLUDE_DIR="$HERE/include/opencv4"
else
  OPENCV_INCLUDE_DIR="/usr/include/opencv4"
fi

# 运行库目录：按 ABI 探测（找不到就 die，并把每个候选目录里实际有的版本列出来）
if [ -n "$_cv_lib_user" ]; then
  OPENCV_LIB_DIR="$_cv_lib_user"
else
  OPENCV_LIB_DIR="$(detect_opencv_lib_dir)"
fi

BOARD_LIBS="${BOARD_LIBS:-/usr/lib/aarch64-linux-gnu}"
RT_LIB_DIR="${RT_LIB_DIR:-/usr/lib}"
EXTRA_LIB_DIRS="${EXTRA_LIB_DIRS:-}"

# ---- 头版本解析：CV_VERSION_MAJOR / CV_VERSION_MINOR ------------------------
# 从 opencv2/core/version.hpp 里取头版本（# 与 define 之间允许多余空格）
cv_version_of() {
  local f="$1/opencv2/core/version.hpp" maj min
  [ -f "$f" ] || return 0
  maj="$(sed -n 's/^[[:space:]]*#[[:space:]]*define[[:space:]]\{1,\}CV_VERSION_MAJOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$f" | head -1)" || maj=""
  min="$(sed -n 's/^[[:space:]]*#[[:space:]]*define[[:space:]]\{1,\}CV_VERSION_MINOR[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p' "$f" | head -1)" || min=""
  [ -n "$maj" ] && [ -n "$min" ] && echo "$maj.$min"
  return 0
}

# ---- 运行库完整性：按**精确版本**判定 --------------------------------------
# 旧写法 `libopencv_$m.so*`（通配）会把 4.6 的 .so.406 也算成"有" ⇒ 明明链的
# 是 4.6 却以为对上了 4.8。⇒ 改成精确匹配 libopencv_$m.so.$OPENCV_ABI；无版本号
# 的 .so 软链也接受（新板 /usr/lib/libopencv_core.so 就是这种），但必须**核对它
# 的 SONAME** —— 链接器 -lopencv_<m> 吃的正是这个 .so，它的 SONAME 决定产物里
# 的 NEEDED，对不上就等于悄悄链了另一个次版本。
cv_module_ok() {
  local d="$1" m="$2" abi="$3" so son
  # ① 精确版本文件
  if [ -f "$d/libopencv_$m.so.$abi" ]; then
    so="$d/libopencv_$m.so"
    if [ -f "$so" ]; then
      son="$(readelf -d "$so" 2>/dev/null | sed -n 's/.*SONAME.*\[\([^]]*\)\].*/\1/p' | head -1)" || son=""
      [ "$son" = "libopencv_$m.so.$abi" ] && return 0
      return 1
    fi
    return 0
  fi
  # ② 只有无版本号的 .so（开发软链）⇒ 核对它的 SONAME
  so="$d/libopencv_$m.so"
  if [ -f "$so" ]; then
    son="$(readelf -d "$so" 2>/dev/null | sed -n 's/.*SONAME.*\[\([^]]*\)\].*/\1/p' | head -1)" || son=""
    [ "$son" = "libopencv_$m.so.$abi" ] && return 0
  fi
  return 1
}

# ---- OpenCL 链接项解析 ----------------------------------------------------
# 新板（Debian 12）只有 **ICD loader** /usr/lib/aarch64-linux-gnu/libOpenCL.so.1，
# **没有 libOpenCL.so 开发软链** ⇒ -lOpenCL 必然链不出来。
# ⇒ 退化：直接把这个版本化文件按**绝对路径**放进链接行。它的 SONAME 就是
#    libOpenCL.so.1 ⇒ 产物里的 NEEDED 仍是 libOpenCL.so.1，与 -lOpenCL 一致。
resolve_opencl_link() {
  if [ -f "$BOARD_LIBS/libOpenCL.so" ]; then
    echo "-lOpenCL"; return 0
  fi
  local f
  for f in "$BOARD_LIBS/libOpenCL.so.1" /lib/aarch64-linux-gnu/libOpenCL.so.1 /usr/lib/aarch64-linux-gnu/libOpenCL.so.1; do
    [ -f "$f" ] && { echo "$f"; return 0; }
  done
  die "找不到板卡 libOpenCL（既没有 libOpenCL.so，也没有 libOpenCL.so.1）。
       ⇒ 板卡只有运行库、没有开发软链时，脚本会退化成直接链接版本化的 .so.1；
         这里连 .so.1 都没有 ⇒ 板上可能压根没装 OpenCL ICD loader。
       ⇒ 可用 BOARD_LIBS=<目录> 覆盖。"
}
OPENCL_LINK="$(resolve_opencl_link)"

# ---------------------------------------------------------------------------
# 交叉编译器（不在包里 —— 编译器没法随包发）
# ---------------------------------------------------------------------------
# ⚠️ 2026-11-23：**不再硬编码宿主机盘符**。
#    宿主机重装 / 换盘之后 `/media/manu/1TB-Volume` 会变成 `/run/media/manu/<UUID>`
#    ⇒ 写死的路径必然失效（而且失效得很隐蔽：脚本会在「找不到编译器」这一行才炸）。
#    ⇒ 改成**自动发现**：按 新式盘符 → 旧式盘符 → $HOME 顺序扫，候选之间用 ':' 分隔。
TC_REL="rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu"
TC_DEFAULT=""
for _tc_root in /run/media/manu/* /media/manu/* "$HOME"; do
  [ -d "$_tc_root/$TC_REL" ] && TC_DEFAULT="${TC_DEFAULT:+:}$_tc_root/$TC_REL"
done
[ -n "$TC_DEFAULT" ] || TC_DEFAULT="/media/manu/1TB-Volume/$TC_REL"
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
       ⇒ 这应当是**板卡**的 OpenCV 头（板卡上为 /usr/include/opencv4，只有 4.6；
         本 SDK 要的是 $OPENCV_VERSION 的头，包内 include/opencv4 就是那份）。
         交叉编译时必须指向**从板卡取回的那份副本**（fetch_board_libs.sh），
         不能用宿主机的 /usr（x86 的头会污染构建）。
         可用 OPENCV_INCLUDE_DIR=<目录> 覆盖。"

# ---- 头版本 ↔ 运行库 ABI 的一致性校验 --------------------------------------
# OpenCV 次版本会改 C++ ABI ⇒ 拿 4.6 的头去编 4.8 的运行库就是 ABI 错配，
# 症状是运行期莫名其妙崩在 cv::Mat 相关的调用上 ⇒ 必须在链接前**当场**挡住。
_cv_hdr_ver="$(cv_version_of "$OPENCV_INCLUDE_DIR")"
[ -n "$_cv_hdr_ver" ] || die "解析不出 OpenCV 头版本（CV_VERSION_MAJOR/MINOR）：$OPENCV_INCLUDE_DIR/opencv2/core/version.hpp
       ⇒ 这不是一个完整的 OpenCV 头目录。"
_cv_hdr_major="${_cv_hdr_ver%%.*}"
_cv_hdr_minor="${_cv_hdr_ver##*.}"
_cv_exp_major="${OPENCV_VERSION%%.*}"
_cv_exp_minor="${OPENCV_VERSION##*.}"
# MAJOR+MINOR 逐位比（4.8 → 48，4.10 → 410）：比的是**次版本**，补丁号不算
[ "$_cv_hdr_major$_cv_hdr_minor" = "$_cv_exp_major$_cv_exp_minor" ] || {
  _cv_board_ver="$(cv_version_of /usr/include/opencv4)"
  _cv_pkg_ver="$(cv_version_of "$HERE/include/opencv4")"
  die "OpenCV 头版本与运行库 ABI 不配套：
       头目录   : $OPENCV_INCLUDE_DIR → 实际 OpenCV $_cv_hdr_ver
       期望版本 : $OPENCV_VERSION（OPENCV_ABI=$OPENCV_ABI，ABI 前两位 ${OPENCV_ABI:0:2}）
       板上 /usr/include/opencv4 的版本: ${_cv_board_ver:-（本机未找到；这块板上是 4.6 的头）}
       ⇒ OpenCV 次版本会改 C++ ABI：$_cv_hdr_ver 的头配 $OPENCV_ABI 的运行库是错配，
         同一进程里会同时出现两套 OpenCV ⇒ cv::Mat 跨边界串味。
       ⇒ 必须显式指定 $OPENCV_VERSION 的头目录：
         OPENCV_INCLUDE_DIR=<含 opencv2/ 的 $OPENCV_VERSION 头目录>
         并用 OPENCV_ABI=$OPENCV_ABI 让运行库与它配套。
         包内 include/opencv4：${_cv_pkg_ver:-（这个包里没有）}"
}

_cv_missing=""
for m in core imgproc imgcodecs calib3d features2d flann video; do
  cv_module_ok "$OPENCV_LIB_DIR" "$m" "$OPENCV_ABI" || _cv_missing="$_cv_missing $m"
done
if [ -n "$_cv_missing" ]; then
  {
    echo "       缺（要 libopencv_*.so.$OPENCV_ABI）：$_cv_missing"
    echo "       目录里实际有的版本："
    for m in $_cv_missing; do
      _cv_found="$(ls -1 "$OPENCV_LIB_DIR"/libopencv_$m.so* 2>/dev/null | while IFS= read -r f; do basename "$f"; done | tr '\n' ' ')" || _cv_found=""
      echo "         libopencv_$m : ${_cv_found:-(无)}"
    done
  } >&2
  die "板卡 OpenCV 动态库不全 / 版本不对（OPENCV_ABI=$OPENCV_ABI）：$OPENCV_LIB_DIR
       ⇒ 新板上 4.8 在 /usr/lib（.so.408），Debian 的 4.6 在 /usr/lib/aarch64-linux-gnu（.so.406）。
       ⇒ 可用 OPENCV_LIB_DIR=<目录> 覆盖，或用 OPENCV_ABI=406 明确要 4.6 那套。"
fi

[ -f "$RT_LIB_DIR/librknnrt.so" ] || die "找不到板卡 librknnrt.so：$RT_LIB_DIR
       ⇒ 可用 RT_LIB_DIR=<目录> 覆盖。"
# RGA：按**绝对路径**参与链接，不用 `-lrga`。
#   🔴 为什么：`-L` 顺序是 RT_LIB_DIR → BOARD_LIBS → OPENCV_LIB_DIR，而板卡上
#      RT_LIB_DIR 默认就是 **/usr/lib** —— 那里有**另一份** librga.so
#      （Rockchip BSP 版，md5 与 multiarch 那份不同；app 的 ldd 解析到的也不是它）
#      ⇒ `-lrga` 会挑到 BSP 那份 ⇒ 必须用绝对路径锁定 BOARD_LIBS 这一份
#      （与 build_board_native.sh 同口径；链接器取其 SONAME ⇒ NEEDED 仍是 librga.so.2）。
if [ -f "$BOARD_LIBS/librga.so" ]; then
  RGA_LINK="$BOARD_LIBS/librga.so"
elif [ -f "$BOARD_LIBS/librga.so.2" ]; then
  RGA_LINK="$BOARD_LIBS/librga.so.2"
else
  die "找不到板卡 librga：$BOARD_LIBS
       ⇒ 可用 BOARD_LIBS=<目录> 覆盖（板卡上是 /usr/lib/aarch64-linux-gnu）。"
fi
[ -f "$HERE/src/npu/mosaic_pipeline.cpp" ] || die "missing source: src/npu/mosaic_pipeline.cpp"

echo "  工具链   : $TC"
echo "  编译器   : $CXX"
echo "  OpenCV头 : $OPENCV_INCLUDE_DIR (v$_cv_hdr_ver)"
echo "  OpenCV库 : $OPENCV_LIB_DIR (ABI $OPENCV_ABI)"
echo "  OpenCL   : $OPENCL_LINK"
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
# v10.5: the spatio-temporal track fusion kernel.  It is a SEPARATE translation
# unit (zero third-party dependencies) and it MUST be compiled with the same
# $FP_FLAGS: with aarch64's default -ffp-contract=fast the FMA contraction
# changes the rounding and the result is no longer bit-identical to the
# Python reference (measured: a different output MD5).
"$CXX" $LIB_CXXFLAGS -c "$HERE/src/npu/mosaic_track_fusion.cpp" -o "$OUT/mosaic_track_fusion.o"

echo "== linking  libmosaic_pipeline.so =="
find "$OUT" -maxdepth 1 -type f -name 'libmosaic_pipeline.so' -delete
"$CXX" -shared -Wl,-soname,libmosaic_pipeline.so $RELAX \
  -o "$OUT/libmosaic_pipeline.so" "$OUT/mosaic_pipeline.o" "$OUT/mosaic_track_fusion.o" \
  "${LIB_PATHS[@]}" "${RPATH_LINKS[@]}" \
  $OPENCV_LIBS "$OPENCL_LINK" -lrknnrt "$RGA_LINK" -lpthread -ldl -lm

# ---------------------------------------------------------------------------
# 1b. archiving  libmosaic_pipeline.a  (STATIC library)
#
#     为什么同时交付 .so 与 .a：嵌入式组要求静态库（要单文件交付 / 不想在运行
#     环境里带我们的 .so）；但 .so 那套已经被认证过 => 两种都留着，客户要哪
#     种都有，行为逐字节一致（静态库与共享库同源同对象文件）。
#
#     成员只有 mosaic_pipeline.o + mosaic_track_fusion.o，故意不含
#     stat_compat.o：那个垫片转发的是 glibc 的 __xstat（glibc 2.33 起已移除）
#     => 把它打进预编译归档，等于给这份 .a 锁死一个 glibc 大版本 => 目标板
#     glibc >= 2.33 时直接链接失败。stat_compat.o 继续单独留在 out/，由集成
#     方按目标板自己决定要不要带（包内 src/npu/stat_compat.c 也带上了源码）。
# ---------------------------------------------------------------------------
echo "== archiving libmosaic_pipeline.a (static library) =="
find "$OUT" -maxdepth 1 -type f -name 'libmosaic_pipeline.a' -delete
# -D = deterministic（归档里不留 uid/gid/mtime => 同一份源码重复构建 md5 稳定）
if ! "$AR" rcsD "$OUT/libmosaic_pipeline.a" "$OUT/mosaic_pipeline.o" "$OUT/mosaic_track_fusion.o" 2>/dev/null; then
  "$AR" rcs "$OUT/libmosaic_pipeline.a" "$OUT/mosaic_pipeline.o" "$OUT/mosaic_track_fusion.o"
fi
[ -f "$OUT/libmosaic_pipeline.a" ] || die "静态库归档失败（ar: $AR）"

echo
echo "== artifact =="
echo "  path : $OUT/libmosaic_pipeline.so"
echo "  size : $(stat -c%s "$OUT/libmosaic_pipeline.so") B"
echo "  md5  : $(md5sum "$OUT/libmosaic_pipeline.so" | cut -d' ' -f1)"
echo "  path : $OUT/libmosaic_pipeline.a"
echo "  size : $(stat -c%s "$OUT/libmosaic_pipeline.a") B"
echo "  md5  : $(md5sum "$OUT/libmosaic_pipeline.a" | cut -d' ' -f1)"
echo "  members: $("$AR" t "$OUT/libmosaic_pipeline.a" | tr '\n' ' ')"
echo
echo "  -- 运行时依赖（全部为板卡自带库，板卡 ldconfig 可解析）--"
readelf -d "$OUT/libmosaic_pipeline.so" | grep NEEDED | sed 's/.*\[/    /;s/\]//'
echo
echo "[ok] shared library + static library built.  Next: bash build_demo.sh"
