#!/usr/bin/env bash
# ===========================================================================
# make_build_dir.sh -- 从工程里把交付嵌入式组所需的【全部交付文件】整理进
#                      build/。
#
# 本脚本从工程里把交付嵌入式组所需的**全部交付文件**整理进 `build/`，
# 可反复重跑（幂等）：重跑会刷新全部文件、删掉不再需要的陈旧文件、
# 重新生成 MANIFEST.md。
#
#   manu/pipeline/npu/packaging/   ← 本脚本 + 模板 + 脚本原件（手写、可维护）
#   manu/pipeline/npu/build/        ← ★ 生成物（不要手改，改源码后重跑本脚本）
#
# 本脚本【不编译、不拷到 NFS、不 ssh】：
#   * 编译        → build/build_lib.sh + build/build_demo.sh
#   * 拷到 NFS    → build/copy_to_nfs.sh
#
# ---------------------------------------------------------------------------
# 【2026-11-11 新板交付版】交付包结构变了
#
#   旧版：包内自带 third_party/（交叉编译版 OpenCV 头 + **静态库** .a、
#         板卡 libOpenCL.so / librknnrt.so / librga.so 的副本）⇒ 220 个文件。
#   新版：**包内不携带任何第三方 .so/.a**（嵌入式组要求：直接用板卡上
#         原来就有的 .so）⇒ 没有 third_party/ 目录。
#         库本身也从**静态库 .a 改成共享库 .so**：
#           out/libmosaic_pipeline.so（SONAME = libmosaic_pipeline.so）
#         四个测试程序用 `-lmosaic_pipeline` + `-Wl,-rpath,'$ORIGIN'` 链接它
#         ⇒ 整包可搬、可直接跑，不需要设 LD_LIBRARY_PATH。
#
#   ⚠️ 头与运行库必须**同版本**：用板卡 OpenCV 4.6.0 的头去链板卡
#      4.6.0 的 libopencv_*.so.406；不要拿 4.8 的运行库配 4.6 的头。
# ---------------------------------------------------------------------------
#
# 用法: bash manu/pipeline/npu/packaging/make_build_dir.sh
# ===========================================================================
set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEST="$ROOT/build"
STAGE="$ROOT/.build-include"

die() { echo "[ERROR] $*" >&2; exit 1; }

echo "==========================================================="
echo " 从工程里整理交付包 -> build/"
echo "==========================================================="
echo "  脚本目录 : $HERE"
echo "  工程根   : $ROOT"
echo "  生成目录 : $DEST"
echo

# ---------------------------------------------------------------------------
# 占位符取值（与 build_lib.sh / build_demo.sh / copy_to_nfs.sh 的默认值一致）
# ---------------------------------------------------------------------------
TOOLCHAIN_V="${RK3588_TOOLCHAIN:-/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"

# ---------------------------------------------------------------------------
# 板卡依赖的【定位】。
#
# 本包**不再携带任何第三方 .so/.a** ⇒ OpenCV / RKNN Runtime / RGA / OpenCL
# 全部**动态链接板卡自带的库**。下面的默认值就是**板卡的系统路径**：
#
#   OPENCV_INCLUDE_DIR  板卡 OpenCV **头**（含 opencv2/）   板卡默认 /usr/include/opencv4
#   OPENCV_LIB_DIR      板卡 OpenCV **动态库**（.so.406）   板卡默认 /usr/lib/aarch64-linux-gnu
#   BOARD_LIBS          板卡 libOpenCL.so.1 / librga.so.2   板卡默认 /usr/lib/aarch64-linux-gnu
#   RT_LIB_DIR          板卡 librknnrt.so                   板卡默认 /usr/lib
#
# ⚠️ **交叉编译在宿主机上做时，上面这些路径指的是「目标板卡的文件系统」**：
#    宿主机的 /usr 是 x86 的，**绝对不能直接用**（会污染构建）⇒
#    必须先把板卡的那份（头 + .so）取回宿主机，再把变量指过去。
#
# build_lib.sh / build_demo.sh 用的是同一套变量（名字与默认值完全一致）。
# ---------------------------------------------------------------------------
OPENCV_INCLUDE_DIR_V="${OPENCV_INCLUDE_DIR:-/usr/include/opencv4}"
OPENCV_LIB_DIR_V="${OPENCV_LIB_DIR:-/usr/lib/aarch64-linux-gnu}"
OPENCV_V="${OPENCV_INSTALL:-/usr/include/opencv4}"     # 旧键名（兼容用；现在等价于板卡 OpenCV 头目录）
BOARD_LIBS_V="${BOARD_LIBS:-/usr/lib/aarch64-linux-gnu}"
RT_LIB_DIR_V="${RT_LIB_DIR:-/usr/lib}"
NFS_ROOT_V="${NFS_ROOT:-/run/media/manu/SSD960-LOCAL/nfs}"
BOARD_V="${BOARD:-root@192.168.0.211}"
BOARD_PORT_V="${BOARD_PORT:-22}"

# 填占位符。模板里 {{...}} 形式的占位符由本函数按实况替换。
fill_placeholders() {
  local src="$1" dst="$2"
  sed -E \
    -e "s|\{\{SRC_ROOT\}\}|$ROOT|g" \
    -e "s|\{\{PKG_ROOT\}\}|$DEST|g" \
    -e "s|\{\{NFS_ROOT\}\}|$NFS_ROOT_V|g" \
    -e "s|\{\{BOARD\}\}|$BOARD_V|g" \
    -e "s|\{\{BOARD_PORT\}\}|$BOARD_PORT_V|g" \
    -e "s|\{\{TOOLCHAIN\}\}|$TOOLCHAIN_V|g" \
    -e "s|\{\{OPENCV_INSTALL\}\}|$OPENCV_V|g" \
    -e "s|\{\{OPENCV_INCLUDE_DIR\}\}|$OPENCV_INCLUDE_DIR_V|g" \
    -e "s|\{\{OPENCV_LIB_DIR\}\}|$OPENCV_LIB_DIR_V|g" \
    -e "s|\{\{BOARD_LIBS\}\}|$BOARD_LIBS_V|g" \
    -e "s|\{\{RT_LIB_DIR\}\}|$RT_LIB_DIR_V|g" \
    -e "s|\{\{FILE_COUNT\}\}|$FILE_COUNT|g" \
    -e "s|\{\{SRC_COUNT\}\}|$SRC_COUNT|g" \
    -e "s|\{\{HDR_COUNT\}\}|$HDR_COUNT|g" \
    -e "s|\{\{KERNEL_COUNT\}\}|$KERNEL_COUNT|g" \
    -e "s|\{\{OUT_COUNT\}\}|$OUT_COUNT|g" \
    -e "s|\{\{TOTAL_COUNT\}\}|$TOTAL_COUNT|g" \
    -e "s|\{\{OBJ_MD5\}\}|$OBJ_MD5|g" \
    -e "s|\{\{LIB_SO_MD5\}\}|$LIB_SO_MD5|g" \
    -e "s|\{\{LIB_A_MD5\}\}|$LIB_SO_MD5|g" \
    -e "s|\{\{API_TEST_MD5\}\}|$API_TEST_MD5|g" \
    -e "s|\{\{CLIENT_EXAMPLE_MD5\}\}|$CLIENT_EXAMPLE_MD5|g" \
    -e "s|\{\{CONTRACT_TEST_MD5\}\}|$CONTRACT_TEST_MD5|g" \
    "$src" > "$dst"
}

# 生成物里绝不能残留 {{...}}：模板漏改一个占位符就必须当场失败，
# 而不是产出一份带 {{}} 的交付文档。
check_no_placeholder() {
  local f="$1" n
  n=$(grep -c '{{' "$f" || true)
  if [ "$n" != "0" ]; then
    echo "[ERROR] $f 里还有 $n 行残留的 {{...}} 占位符（模板漏改了？）：" >&2
    grep -n '{{' "$f" >&2 || true
    exit 1
  fi
}

# 产物 md5；文件不存在（还没编译）时填「（尚未编译）」
md5_or_placeholder() {
  local f="$1"
  if [ -f "$f" ]; then md5sum "$f" | cut -d' ' -f1; else echo "（尚未编译）"; fi
}

# ---------------------------------------------------------------------------
# 用途说明（写进 MANIFEST.md 的表里）
# ---------------------------------------------------------------------------
purpose_of() {
  case "$1" in
    README.md)                    echo "使用说明（四条命令 + 交付/零参数/NFS/一键拷贝 各节）" ;;
    MANIFEST.md)                  echo "**本交付清单**（由 make_build_dir.sh 生成）" ;;
    build_lib.sh)                 echo "构建脚本①：只编译库 -> out/libmosaic_pipeline.so（**共享库**，SONAME=libmosaic_pipeline.so）" ;;
    build_demo.sh)                echo "构建脚本②：只编译测试程序 -> out/mosaic_api_test + out/mosaic_client_example + out/mosaic_contract_test + out/rknn_mosaic_e2e（用 -lmosaic_pipeline + -rpath \$ORIGIN 链接同目录的 .so）" ;;
    copy_to_nfs.sh)               echo "一键脚本：编译 + 拷贝到 NFS + md5 回验 +（可选）板卡验证" ;;
    fetch_board_libs.sh)           echo "把**板卡自带的**库与头取回宿主机（ssh 打包 → tar → 解包），并打印 build_lib.sh / build_demo.sh 要用的四个路径变量" ;;
    known_hosts)                  echo "板卡主机密钥基线（ssh 严格校验用；板卡换密钥会直接失败，这是故意的）" ;;
    src/npu/mosaic_pipeline.cpp)  echo "库实现（GMC/OpenCL + RGA mosaic + RKNN fp16 + 反投影）" ;;
    src/npu/mosaic_pipeline.h)    echo "**公开 C ABI 头**（集成方写代码时 include 它）" ;;
    src/npu/mosaic_core.h)        echo "库内部核心头（GMC / RGA / RKNN 胶水层 + 后处理三路径）" ;;
    src/npu/mosaic_post_gpu.h)    echo "GPU 后处理路径（OpenCL 局部极大；kernel 已内嵌，**不需要**外置 .cl）" ;;
    src/npu/mosaic_api_test.cpp)  echo "示例 demo 源（只调公开 C ABI）" ;;
    src/npu/mosaic_client_example.cpp) echo "**最简集成示范源**（嵌入方照着抄；含 set_post_mode 钉死 sparse）" ;;
    src/npu/mosaic_contract_test.cpp) echo "契约测试源（验收门禁）" ;;
    src/npu/rknn_mosaic_e2e.cpp)  echo "**端到端测试程序源**（吞吐/延时测量，只调公开 C ABI）" ;;
    src/cpp/gmc_stream_ocl.cpp)   echo "GMC/OpenCL 引擎（被 mosaic_pipeline.cpp 以 ../cpp/ 相对包含）" ;;
    src/cpp/md5.h)                echo "MD5 实现（被 gmc_stream_ocl.cpp 以 md5.h 相对包含）" ;;
    src/opencl/host/ocl_host.h)   echo "OpenCL host 封装（被 gmc_stream_ocl.cpp 以 ../opencl/host/ 相对包含）" ;;
    src/opencl/host/post_local_max.cl) echo "GPU 后处理的 kernel 源码**可读副本**（运行时不需要，源码已内嵌进 mosaic_post_gpu.h）" ;;
    include/rknn_api.h)           echo "RKNN Runtime API 头（第三方）" ;;
    include/stb_image.h)          echo "stb 图像解码（第三方）" ;;
    include/stb_image_resize.h)   echo "stb 图像缩放（第三方）" ;;
    include/stb_image_write.h)    echo "stb 图像写出（第三方）" ;;
    runtime/kernels/warp_median_fused.cl)  echo "**运行时必需**：warp+median 融合 OpenCL kernel" ;;
    runtime/kernels/sortnet_generated.inc) echo "**运行时必需**：sortnet 生成的 OpenCL include" ;;
    runtime/kernels/gmc_feature_gpu.cl)    echo "可选：GPU 侧 GMC 特征 kernel（**库不加载**，仅探针 gmc_gpu_probe 用）" ;;
    include/rga/*)                echo "RGA API 头（第三方）" ;;
    include/CL/*)                 echo "Khronos OpenCL 头（第三方）" ;;
    *)                            echo "—" ;;
  esac
}

# out/ 段的说明（**刻意不列 md5**）
out_note_of() {
  case "$1" in
    out/mosaic_pipeline.o)     echo "编译中间产物（库的 .o）" ;;
    out/libmosaic_pipeline.so) echo "**共享库**（客户链接它；SONAME=libmosaic_pipeline.so；动态链接板卡自带库）" ;;
    out/mosaic_api_test)       echo "示例 demo 可执行文件（给客户的示例）" ;;
    out/mosaic_client_example) echo "**最简集成示范可执行文件**（嵌入方照着抄这份）" ;;
    out/mosaic_contract_test)  echo "契约测试可执行文件（验收门禁）" ;;
    out/rknn_mosaic_e2e)       echo "端到端测试程序（吞吐/延时测量）" ;;
    *)                         echo "构建产物" ;;
  esac
}

# 字节数加千位分隔符（与交付文档的排版一致）
fmt_num() {
  printf '%s' "$1" | sed -E ':a;s/([0-9])([0-9]{3})($|,)/\1,\2\3/;ta'
}

# 本次拷贝清单（相对 DEST）。stale 清理与 md5 表都以它为准。
EXPECTED=()
add_expected() { EXPECTED+=("$1"); }

is_expected() {
  local rel="$1" e
  for e in "${EXPECTED[@]}"; do
    [ "$rel" = "$e" ] && return 0
  done
  return 1
}

# ---------------------------------------------------------------------------
# 0. 先决条件：暂存的第三方头
# ---------------------------------------------------------------------------
echo "== [1/8] 检查暂存的第三方头 =="
[ -d "$STAGE" ] || die "缺少暂存的第三方头：先跑一次 MOSAIC_NO_DEPLOY=1 bash build_mosaic_pipeline_arm64.sh（它会自动 stage 这些头）"
[ -d "$STAGE/rga" ] || die "暂存的第三方头缺少 rga/：$STAGE/rga"
[ -d "$STAGE/CL" ]  || die "暂存的第三方头缺少 CL/：$STAGE/CL"
echo "  [ok] $STAGE（rga/ + CL/ 齐备）"

# ---------------------------------------------------------------------------
# 1. 拷贝源码、脚本、kernel 资产
# ---------------------------------------------------------------------------
echo
echo "== [2/8] 拷贝源码 / 脚本 / kernel 资产 =="
mkdir -p "$DEST/src/npu" "$DEST/src/cpp" "$DEST/src/opencl/host" \
         "$DEST/runtime/kernels" "$DEST/include/rga" "$DEST/include/CL"

copy_one() {
  local src="$1" rel="$2"
  [ -f "$src" ] || die "源文件不存在：$src"
  cp -f "$src" "$DEST/$rel"
  add_expected "$rel"
  printf "  [ok] %-46s <- %s\n" "$rel" "${src#$ROOT/}"
}

# 库与 demo 的源
copy_one "$ROOT/mosaic_pipeline.cpp"            "src/npu/mosaic_pipeline.cpp"
copy_one "$ROOT/mosaic_pipeline.h"              "src/npu/mosaic_pipeline.h"
copy_one "$ROOT/mosaic_core.h"                  "src/npu/mosaic_core.h"
copy_one "$ROOT/mosaic_post_gpu.h"              "src/npu/mosaic_post_gpu.h"
copy_one "$ROOT/mosaic_api_test.cpp"            "src/npu/mosaic_api_test.cpp"
copy_one "$ROOT/mosaic_client_example.cpp"      "src/npu/mosaic_client_example.cpp"
copy_one "$ROOT/mosaic_contract_test.cpp"       "src/npu/mosaic_contract_test.cpp"
copy_one "$ROOT/rknn_mosaic_e2e.cpp"            "src/npu/rknn_mosaic_e2e.cpp"
# 被相对包含的引擎 / 工具 / OpenCL host
copy_one "$ROOT/../cpp/gmc_stream_ocl.cpp"      "src/cpp/gmc_stream_ocl.cpp"
copy_one "$ROOT/../cpp/md5.h"                   "src/cpp/md5.h"
copy_one "$ROOT/../opencl/host/ocl_host.h"       "src/opencl/host/ocl_host.h"
# GPU 后处理路径的 kernel 源码（mosaic_post_gpu.h 里已内嵌同一份，
# 这里带的是**可读副本**，方便集成方查阅/改写；运行时不需要它）
copy_one "$ROOT/../opencl/kernels/post_local_max.cl" "src/opencl/host/post_local_max.cl"
# 运行时 OpenCL kernel 资产
copy_one "$ROOT/../opencl/kernels/warp_median_fused.cl"  "runtime/kernels/warp_median_fused.cl"
copy_one "$ROOT/../opencl/kernels/sortnet_generated.inc" "runtime/kernels/sortnet_generated.inc"
copy_one "$ROOT/../opencl/kernels/gmc_feature_gpu.cl"    "runtime/kernels/gmc_feature_gpu.cl"
# 脚本与文档（packaging/ 下是原件，拷进 build/ 后是运行版）
# README.md 不在这里拷贝：它由本脚本从 packaging/README.template.md **生成**
#   （见下面的 [6/8]），所以这里只把它登记进清单。
copy_one "$HERE/build_lib.sh"       "build_lib.sh"
copy_one "$HERE/build_demo.sh"      "build_demo.sh"
copy_one "$HERE/copy_to_nfs.sh"     "copy_to_nfs.sh"
copy_one "$HERE/fetch_board_libs.sh" "fetch_board_libs.sh"
copy_one "$HERE/known_hosts"        "known_hosts"
chmod +x "$DEST/build_lib.sh" "$DEST/build_demo.sh" "$DEST/copy_to_nfs.sh" "$DEST/fetch_board_libs.sh"
EXPECTED+=("README.md")   # 生成物，但在交付清单里（md5 在生成后才有）

# ---------------------------------------------------------------------------
# 2. 拷贝第三方头
# ---------------------------------------------------------------------------
echo
echo "== [3/8] 拷贝第三方头（include/）=="
for h in rknn_api.h stb_image.h stb_image_resize.h stb_image_write.h; do
  copy_one "$STAGE/$h" "include/$h"
done
for sub in rga CL; do
  while IFS= read -r f; do
    copy_one "$f" "include/$sub/$(basename "$f")"
  done < <(find "$STAGE/$sub" -type f | sort)
done

# ---------------------------------------------------------------------------
# 4. 【本轮改造】第三方依赖**不再随包携带** —— 包里一个 .so/.a 都不带
#
#    旧设计：包内自带 third_party/（交叉编译版 OpenCV 的头 + **静态库** .a、
#    板卡 libOpenCL.so / librknnrt.so / librga.so 的副本）⇒ 220 个文件。
#    新设计（嵌入式组要求）：**直接用板卡上原来就有的 .so**
#
#      OpenCV 4.6.0     头 /usr/include/opencv4
#                       运行库 /usr/lib/aarch64-linux-gnu/libopencv_*.so.406
#      RKNN Runtime 2.2.0   /usr/lib/librknnrt.so
#      RGA                 /usr/lib/aarch64-linux-gnu/librga.so.2
#      OpenCL 3.0(Mali-G610) /usr/lib/aarch64-linux-gnu/libOpenCL.so.1
#
#    ⇒ 交付包**没有 third_party/**；库本身也改成**共享库**交付：
#      out/libmosaic_pipeline.so（SONAME = libmosaic_pipeline.so），
#      四个测试程序用 `-lmosaic_pipeline` + `-Wl,-rpath,'$ORIGIN'` 链接它。
#
#    ⚠️ 头与运行库必须**同版本**：用板卡 4.6.0 的头去链板卡 4.6.0 的 .so.406；
#       不要拿 4.8 的运行库配 4.6 的头（OpenCV 次版本会改 C++ ABI）。
#    ⚠️ 交叉编译在**宿主机**上做时，上面那些路径指的是**目标板卡的文件系统**；
#       宿主机的 /usr 是 x86 的，**绝对不能直接用**（会污染构建）⇒
#       要先把板卡的那份（头 + .so）取回宿主机，再把变量指过去。
#
#    旧包残留的 third_party/ 由 [5/8] 的**目录级清理**整棵删掉。
# ---------------------------------------------------------------------------
echo
echo "== [4/8] 第三方依赖：不再随包携带 =="
echo "  [ok] 本包**不携带任何第三方 .so/.a**："
echo "       OpenCV / RKNN Runtime / RGA / OpenCL 全部动态链接**板卡自带**的库"
echo "       编译期头 ：$OPENCV_INCLUDE_DIR_V"
echo "       链接期库 ：$OPENCV_LIB_DIR_V  $RT_LIB_DIR_V  $BOARD_LIBS_V"

# ---------------------------------------------------------------------------
# 5. 清理陈旧文件（不在本次清单里的旧文件）—— 用 find -delete，不用 rm -rf
#
#    ⚠️ **目录级清理**：旧设计（第三方依赖随包）留下的 `third_party/` 有 220 个
#    文件，本包**不再携带任何第三方 .so/.a** ⇒ 整棵目录必须删掉，
#    否则旧包的副本会污染新包（MANIFEST 回验也会因「漏登记」直接失败）。
#    逐文件 delete 慢且会留下空目录 ⇒ 这里整棵删。
# ---------------------------------------------------------------------------
echo
echo "== [5/8] 清理陈旧文件 =="
stale=0
for sub in src include runtime; do
  [ -d "$DEST/$sub" ] || continue
  while IFS= read -r f; do
    rel="${f#$DEST/}"
    if ! is_expected "$rel"; then
      find "$f" -type f -delete
      echo "  [stale] 已删除 $rel"
      stale=$((stale + 1))
    fi
  done < <(find "$DEST/$sub" -type f | sort)
done

# 旧包残留的 third_party/：整棵删掉（目录级）
if [ -d "$DEST/third_party" ]; then
  _tp_n=$(find "$DEST/third_party" -type f | wc -l)
  rm -rf "$DEST/third_party"
  echo "  [stale] 已删除整个目录 third_party/（旧设计的第三方依赖副本，$_tp_n 个文件）"
  stale=$((stale + 1))
fi
echo "  [ok] 删除 $stale 个陈旧文件/目录"

# ---------------------------------------------------------------------------
# 5. 生成 README（内容先落临时文件，让 MANIFEST 能登记它的 md5）
# ---------------------------------------------------------------------------
echo
echo "== [6/8] 生成 README =="
README_TPL="$HERE/README.template.md"
[ -f "$README_TPL" ] || die "缺模板：$README_TPL"

# 计数与 md5 都按【当前磁盘实况】算
SRC_COUNT=$(find "$DEST/src" -type f 2>/dev/null | wc -l)
HDR_COUNT=$(find "$DEST/include" -type f 2>/dev/null | wc -l)
KERNEL_COUNT=$(find "$DEST/runtime" -type f 2>/dev/null | wc -l)
FILE_COUNT=$(( ${#EXPECTED[@]} + 1 ))   # + MANIFEST.md 自身
# out/ 的产物**不进受管清单**（md5 会随构建变），单独计数后再加总
if [ -d "$DEST/out" ]; then OUT_COUNT=$(find "$DEST/out" -type f | wc -l); else OUT_COUNT=0; fi
# 磁盘上真实存在的文件总数 = 受管 + 编译产物
# （**不含 third_party**：本包不携带任何第三方 .so/.a）
TOTAL_COUNT=$(( FILE_COUNT + OUT_COUNT ))
OBJ_MD5=$(md5_or_placeholder "$DEST/out/mosaic_pipeline.o")
LIB_SO_MD5=$(md5_or_placeholder "$DEST/out/libmosaic_pipeline.so")
API_TEST_MD5=$(md5_or_placeholder "$DEST/out/mosaic_api_test")
CLIENT_EXAMPLE_MD5=$(md5_or_placeholder "$DEST/out/mosaic_client_example")
CONTRACT_TEST_MD5=$(md5_or_placeholder "$DEST/out/mosaic_contract_test")

README_TMP="$(mktemp)"
fill_placeholders "$README_TPL" "$README_TMP"
check_no_placeholder "$README_TMP"
echo "  [ok] README 内容已生成（计数：$TOTAL_COUNT = $FILE_COUNT + $OUT_COUNT / $SRC_COUNT / $HDR_COUNT / $KERNEL_COUNT）"

# ---------------------------------------------------------------------------
# 6. 生成 build/MANIFEST.md（模板 + 自动生成的 md5 表）
# ---------------------------------------------------------------------------
echo
echo "== [7/8] 生成 MANIFEST.md =="
TEMPLATE="$HERE/MANIFEST.template.md"
[ -f "$TEMPLATE" ] || die "缺模板：$TEMPLATE"

TABLE="$(mktemp)"
trap 'rm -f "$TABLE" "$README_TMP"' EXIT

{
  echo "**共 ${#EXPECTED[@]} 个交付文件（不含 \`out/\` 的 $OUT_COUNT 个编译产物与本清单自身）。**"
  echo
  echo "### 交付文件清单"
  echo
  echo "| 路径 | md5 | 字节数 | 用途 |"
  echo "|---|---|---:|---|"
  for rel in $(printf '%s\n' "${EXPECTED[@]}" | sort); do
    f="$DEST/$rel"
    # README.md 是刚生成的（还没落盘），它的 md5 取自临时文件
    if [ "$rel" = "README.md" ]; then
      f="$README_TMP"
    fi
    [ -f "$f" ] || die "清单里的文件不存在：$f"
    printf '| `%s` | `%s` | %s | %s |\n' \
      "$rel" "$(md5sum "$f" | cut -d' ' -f1)" "$(fmt_num "$(stat -c%s "$f")")" "$(purpose_of "$rel")"
  done
  echo
  echo "### 构建产物（不作为清单基线）"
  echo
  echo "> ℹ️ **本段刻意不列 md5**：\`out/\` 里是**构建产物**，随工具链/编译路径变化，"
  echo "> 列进来只会误导人（README 顶部有一张本次构建的快照表）。"
  echo "> 需要核对产物时手动 \`md5sum out/*\`。"
  echo ">"
  echo "> 本版交付的是**共享库** \`out/libmosaic_pipeline.so\`（上一版是 \`.a\` 归档）："
  echo "> \`.so\` **没有** \`ar\` 归档时间戳那种「每次构建都变」的问题 ⇒ 在**同一个"
  echo "> 包根路径**下重编，md5 是稳定的。"
  echo
  echo "| 路径 | 说明 |"
  echo "|---|---|"
  if [ -d "$DEST/out" ]; then
    while IFS= read -r f; do
      rel="${f#$DEST/}"
      printf '| `%s` | %s |\n' "$rel" "$(out_note_of "$rel")"
    done < <(find "$DEST/out" -type f | sort)
  fi
} > "$TABLE"

MANIFEST_TMP="$(mktemp)"
# 先填模板里的占位符，再把自动生成的 md5 表拼进去
{
  sed -n '1,/<!-- MD5_TABLE:BEGIN -->/p' "$TEMPLATE" | sed '$d'
  cat "$TABLE"
  sed -n '/<!-- MD5_TABLE:END -->/,$p' "$TEMPLATE"
} > "$MANIFEST_TMP.raw"
fill_placeholders "$MANIFEST_TMP.raw" "$MANIFEST_TMP"
rm -f "$MANIFEST_TMP.raw"
check_no_placeholder "$MANIFEST_TMP"
mv -f "$MANIFEST_TMP" "$DEST/MANIFEST.md"
# mktemp 建出来的是 600，交付给别人（换用户解压）就读不了 → 统一放开只读权限
chmod 0644 "$DEST/MANIFEST.md"

echo "  [ok] $DEST/MANIFEST.md（模板 + 自动生成表）"

# ---------------------------------------------------------------------------
# 7. README 落盘（**最后生成的文件**）
# ---------------------------------------------------------------------------
cp -f "$README_TMP" "$DEST/README.md"
chmod 0644 "$DEST/README.md"
check_no_placeholder "$DEST/README.md"
echo "  [ok] $DEST/README.md（模板 + 占位符填充）"

# ---------------------------------------------------------------------------
# 8. 回验：表内 md5 与磁盘一致；表外不该有漏登记的文件
# ---------------------------------------------------------------------------
echo
echo "== [8/8] 回验 MANIFEST =="
bad=0
while IFS= read -r line; do
  rel=$(printf '%s' "$line" | sed -E 's/^\| `([^`]+)`.*/\1/')
  want=$(printf '%s' "$line" | sed -E 's/^\| `[^`]+` \| `([0-9a-f]{32})`.*/\1/')
  got=$(md5sum "$DEST/$rel" | cut -d' ' -f1)
  if [ "$want" != "$got" ]; then
    echo "  [ERROR] md5 不一致：$rel  表=$want 磁盘=$got" >&2
    bad=$((bad + 1))
  fi
done < <(grep -E '^\| `[^`]+` \| `[0-9a-f]{32}` \|' "$DEST/MANIFEST.md")
[ "$bad" = "0" ] || die "MANIFEST 里 $bad 个 md5 与磁盘不一致"
echo "  [ok] 表内 md5 全部与磁盘一致"

# 表里有、磁盘上没有
miss=0
while IFS= read -r rel; do
  [ -f "$DEST/$rel" ] || { echo "  [ERROR] 清单登记了但磁盘上没有：$rel" >&2; miss=$((miss + 1)); }
done < <(grep -E '^\| `[^`]+` \| `[0-9a-f]{32}` \|' "$DEST/MANIFEST.md" | sed -E 's/^\| `([^`]+)`.*/\1/')
[ "$miss" = "0" ] || die "MANIFEST 登记了 $miss 个不存在的文件"

# 磁盘上有、表里没有（out/ 与 MANIFEST.md 自身除外）
extra=0
while IFS= read -r f; do
  rel="${f#$DEST/}"
  case "$rel" in
    out/*|MANIFEST.md) continue ;;
  esac
  if ! grep -qE "^\| \`$rel\` \| \`[0-9a-f]{32}\` \|" "$DEST/MANIFEST.md"; then
    echo "  [ERROR] 磁盘上有但清单里没登记：$rel" >&2
    extra=$((extra + 1))
  fi
done < <(find "$DEST" -type f | sort)
[ "$extra" = "0" ] || die "有 $extra 个文件漏登记（清单不完整）"
echo "  [ok] 无漏登记文件"

# ---------------------------------------------------------------------------
# 9. 摘要
# ---------------------------------------------------------------------------
src_n=$(find "$DEST/src" -type f 2>/dev/null | wc -l)
inc_n=$(find "$DEST/include" -type f 2>/dev/null | wc -l)
ker_n=$(find "$DEST/runtime" -type f 2>/dev/null | wc -l)
out_n=$(find "$DEST/out" -type f 2>/dev/null | wc -l)
all_n=$(find "$DEST" -type f | wc -l)
echo
echo "== 完成 =="
echo "  build/ 文件总数 : $all_n"
echo "  源码/脚本      : $src_n 个（src/）"
echo "  第三方头       : $inc_n 个（include/）"
echo "  kernel 资产    : $ker_n 个（runtime/kernels/）"
echo "  共享库         : out/libmosaic_pipeline.so（SONAME=libmosaic_pipeline.so）← 动态链接板卡自带库"
echo "  编译产物       : $out_n 个（out/：共享库 + 4 个测试程序 + .o）"
echo
echo "  下一步（⚠️ 顺序不能颠倒）："
echo "    cd $DEST && bash build_lib.sh && bash build_demo.sh   # ① 编译（库 + 4 个测试程序）"
echo "    cd <repo>/manu/pipeline/npu && bash packaging/make_build_dir.sh   # ② 再打包一次"
echo "         └─ 把新产物的 md5 和「包内文件总数」写回 README/MANIFEST"
echo "           （不重跑的话，交付出去的 README 会写着「（尚未编译）」+ 文件数少 6）"
echo "    cd $DEST && bash copy_to_nfs.sh --verify              # ③ 拷贝 + 上板验证"
echo "         └─ 已编好时用 --verify；想让它自己编用 --build --verify（会自动补②）"
