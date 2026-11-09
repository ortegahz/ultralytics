#!/usr/bin/env bash
# ===========================================================================
# make_build_dir.sh -- 从工程里把交付嵌入式组所需的【全部依赖文件】整理进
#                      build/。
#
# 本脚本从工程里把交付嵌入式组所需的**全部依赖文件**整理进 `build/`，
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
OPENCV_V="${OPENCV_INSTALL:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install}"
BOARD_LIBS_V="${BOARD_LIBS:-/media/manu/1TB-Volume/rk3588/boardlibs}"
RT_LIB_DIR_V="${RT_LIB_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu}"

# ---------------------------------------------------------------------------
# 第三方依赖的【来源】。这三个是编译期的硬依赖（缺一个就编不出来），
# 以前只存在于开发机的绝对路径里 ⇒ 交付出去的包**根本没法原地重编**。
# 现在把它们一并打进 third_party/，包才真正自包含。
#
#   OPENCV_INSTALL   → 交叉编译版 OpenCV（头 + **静态库** .a）
#   BOARD_LIBS       → 板卡 libOpenCL.so
#   RT_LIB_DIR       → 板卡 librknnrt.so / librga.so
#
# 交付态的包里，build_lib.sh / build_demo.sh 默认改用 third_party/ 里的副本；
# 这三个环境变量仍可覆盖（换板卡、换 BSP 时用）。
# ---------------------------------------------------------------------------
TP_SRC_OPENCV="${OPENCV_INSTALL:-/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install}"
TP_SRC_BOARDLIBS="${BOARD_LIBS:-/media/manu/1TB-Volume/rk3588/boardlibs}"
TP_SRC_RT="${RT_LIB_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu}"
NFS_ROOT_V="${NFS_ROOT:-/run/media/manu/SSD960-LOCAL/nfs}"
BOARD_V="${BOARD:-root@192.168.0.64}"
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
    -e "s|\{\{BOARD_LIBS\}\}|$BOARD_LIBS_V|g" \
    -e "s|\{\{RT_LIB_DIR\}\}|$RT_LIB_DIR_V|g" \
    -e "s|\{\{FILE_COUNT\}\}|$FILE_COUNT|g" \
    -e "s|\{\{TP_COUNT\}\}|$TP_COUNT|g" \
    -e "s|\{\{TP_SIZE\}\}|$TP_SIZE|g" \
    -e "s|\{\{SRC_COUNT\}\}|$SRC_COUNT|g" \
    -e "s|\{\{HDR_COUNT\}\}|$HDR_COUNT|g" \
    -e "s|\{\{KERNEL_COUNT\}\}|$KERNEL_COUNT|g" \
    -e "s|\{\{OUT_COUNT\}\}|$OUT_COUNT|g" \
    -e "s|\{\{TOTAL_COUNT\}\}|$TOTAL_COUNT|g" \
    -e "s|\{\{OBJ_MD5\}\}|$OBJ_MD5|g" \
    -e "s|\{\{LIB_A_MD5\}\}|$LIB_A_MD5|g" \
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
    build_lib.sh)                 echo "构建脚本①：只编译库 -> out/libmosaic_pipeline.a" ;;
    build_demo.sh)                echo "构建脚本②：只编译测试程序 -> out/mosaic_api_test + out/mosaic_client_example + out/mosaic_contract_test + out/rknn_mosaic_e2e" ;;
    copy_to_nfs.sh)               echo "一键脚本：编译 + 拷贝到 NFS + md5 回验 +（可选）板卡验证" ;;
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
    third_party/MANIFEST.md)      echo "**第三方依赖逐文件 md5 清单**（OpenCV / libOpenCL / librknnrt / librga）" ;;
    third_party/README.md)        echo "**第三方依赖说明**（是什么 / 编译期怎么用 / 运行时怎么装到板卡）" ;;
    *)                            echo "—" ;;
  esac
}

# out/ 段的说明（**刻意不列 md5**）
out_note_of() {
  case "$1" in
    out/mosaic_pipeline.o)     echo "编译中间产物（库的 .o）" ;;
    out/libmosaic_pipeline.a)  echo "**静态库**（客户链接它）" ;;
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
#   （见下面的 [5/7]），所以这里只把它登记进清单。
copy_one "$HERE/build_lib.sh"       "build_lib.sh"
copy_one "$HERE/build_demo.sh"      "build_demo.sh"
copy_one "$HERE/copy_to_nfs.sh"     "copy_to_nfs.sh"
copy_one "$HERE/known_hosts"        "known_hosts"
chmod +x "$DEST/build_lib.sh" "$DEST/build_demo.sh" "$DEST/copy_to_nfs.sh"
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
# 4. 拷贝第三方依赖到 third_party/  ★ 本次改造的核心
#
#    这三个是**编译期硬依赖**（缺一个就编不出来），以前只存在于开发机的
#    绝对路径里 ⇒ 交付出去的包**根本没法原地重编**，README 里
#    「放到哪都能原地重编」那句话是假的：
#
#      OPENCV_INSTALL  交叉编译版 OpenCV（头 + **静态库** .a）
#      BOARD_LIBS      板卡 libOpenCL.so
#      RT_LIB_DIR      板卡 librknnrt.so / librga.so
#
#    现在打进包里 ⇒ 包真正自包含。包内副本优先，环境变量仍可覆盖。
#
#    ⚠️ 符号链接一律**转成实文件副本**：
#       包内文件要参与 find -type f 枚举与 md5 回验，符号链接会被漏掉
#       （copy_to_nfs.sh 的直写模式就是按 find -type f 逐个 cp 的）。
#       链接器要 libX.so、运行时要 libX.so.1（SONAME）⇒ 两个名字各存一份。
# ---------------------------------------------------------------------------
echo
echo "== [4/8] 拷贝第三方依赖到 third_party/ =="
rm -rf "$DEST/third_party"
mkdir -p "$DEST/third_party/opencv/include/opencv4" \
         "$DEST/third_party/opencv/lib/opencv4/3rdparty" \
         "$DEST/third_party/boardlibs" \
         "$DEST/third_party/rknnrt"

TP_N=0
tp_add() {   # tp_add <源文件> <third_party 内相对路径>
  local src="$1" rel="$2"
  [ -f "$src" ] || die "第三方依赖缺失：$src"
  mkdir -p "$(dirname "$DEST/third_party/$rel")"
  cp -f "$src" "$DEST/third_party/$rel"
  TP_N=$((TP_N + 1))
}

# --- OpenCV：整个 opencv2 头树 + 7 个静态库 + 2 个 3rdparty 静态库 ---
_ocv_hdr="$TP_SRC_OPENCV/include/opencv4/opencv2"
[ -d "$_ocv_hdr" ] || die "交叉编译版 OpenCV 缺失：$_ocv_hdr"
while IFS= read -r f; do
  tp_add "$f" "opencv/include/opencv4/opencv2/${f#"$_ocv_hdr/"}"
done < <(find "$_ocv_hdr" -type f | LC_ALL=C sort)
for l in core imgproc imgcodecs calib3d features2d flann video; do
  tp_add "$TP_SRC_OPENCV/lib/libopencv_$l.a" "opencv/lib/libopencv_$l.a"
done
# 3rdparty **整个目录**一起带上（当前链接行只用 tegra_hal + zlib，但 liblibopenjp2.a
# 也在里面 ⇒ 集成方换链接方式时不会缺东西；多 642 KB，不值得为省它而留隐患）
for a in "$TP_SRC_OPENCV/lib/opencv4/3rdparty/"*.a; do
  [ -f "$a" ] || die "OpenCV 3rdparty 静态库缺失：$a"
  tp_add "$a" "opencv/lib/opencv4/3rdparty/$(basename "$a")"
done

# --- 板卡 libOpenCL：链接名 libOpenCL.so + 运行名 libOpenCL.so.1（SONAME）---
_cl_real="$(readlink -f "$TP_SRC_BOARDLIBS/libOpenCL.so.1" 2>/dev/null || true)"
[ -n "$_cl_real" ] && [ -f "$_cl_real" ] || \
  die "板卡 libOpenCL 缺失：$TP_SRC_BOARDLIBS/libOpenCL.so.1"
tp_add "$_cl_real" "boardlibs/libOpenCL.so"
tp_add "$_cl_real" "boardlibs/libOpenCL.so.1"

# --- 板卡 RKNN Runtime / RGA：librknnrt.so（文件名即 SONAME）+ librga.so / librga.so.2 ---
[ -f "$TP_SRC_RT/librknnrt.so" ] || die "板卡 librknnrt.so 缺失：$TP_SRC_RT/librknnrt.so"
tp_add "$TP_SRC_RT/librknnrt.so" "rknnrt/librknnrt.so"
_rga_real="$(readlink -f "$TP_SRC_RT/librga.so" 2>/dev/null || true)"
[ -n "$_rga_real" ] && [ -f "$_rga_real" ] || \
  die "板卡 librga 缺失：$TP_SRC_RT/librga.so"
tp_add "$_rga_real" "rknnrt/librga.so"
tp_add "$_rga_real" "rknnrt/librga.so.2"

echo "  [ok] 第三方依赖 $TP_N 个文件（OpenCV 头+静态库 / libOpenCL / librknnrt+librga）"

# third_party/MANIFEST.md：逐文件 md5（资产有 220+ 个，单独成册，
# 免得顶层 MANIFEST 被撑爆；顶层清单只登记这一份）
TP_MANIFEST="$DEST/third_party/MANIFEST.md"
{
  echo '# `third_party/` 清单（第三方依赖 · 逐文件 md5）'
  echo
  echo "**共 $TP_N 个依赖文件。**它们是**编译期硬依赖**——以前只存在于开发机的"
  echo "绝对路径里，导致交付包根本没法原地重编；现在一并打包 ⇒ 包自包含。"
  echo
  echo "| 类别 | 何时需要 | 说明 |"
  echo "|---|---|---|"
  # ⚠️ 用单引号：反引号在双引号里会被 bash 当成**命令替换**执行掉 ⇒ 单元格会变空
  echo '| `opencv/include/**` | 编译期 | OpenCV 头（交叉编译版） |'
  echo '| `opencv/lib/**` | 编译期 | OpenCV **静态库** `.a` ⇒ **静态链接**，运行时不需要板卡装 OpenCV |'
  echo '| `boardlibs/**` | 编译期 + 运行时 | 板卡 `libOpenCL.so`（OpenCL ICD） |'
  echo '| `rknnrt/**` | 编译期 + 运行时 | 板卡 `librknnrt.so` / `librga.so`（RKNN Runtime / RGA） |'
  echo
  echo "| 路径 | md5 | 字节数 |"
  echo "|---|---|---:|"
  while IFS= read -r f; do
    printf '| `%s` | `%s` | %s |\n' \
      "${f#"$DEST/third_party/"}" "$(md5sum "$f" | cut -d' ' -f1)" "$(fmt_num "$(stat -c%s "$f")")"
  done < <(find "$DEST/third_party" -type f ! -name MANIFEST.md ! -name README.md | LC_ALL=C sort)
} > "$TP_MANIFEST"
chmod 0644 "$TP_MANIFEST"
echo "  [ok] $TP_MANIFEST（$TP_N 行明细）"

# third_party/README.md：给集成方看的「这三个依赖到底是什么/怎么用」
cat > "$DEST/third_party/README.md" <<'TPREADME'
# `third_party/` —— 交付包自带的第三方依赖

这个目录里的东西以前**不在交付包里**：它们只存在于开发机的绝对路径上
（OpenCV 交叉编译版、板卡 `libOpenCL.so`、板卡 `librknnrt.so`/`librga.so`），
导致交付出去的包**编不出来**，README 里「放到哪都能原地重编」那句话是假的。

现在它们一并打包 ⇒ **包真正自包含**：源码 + 依赖 + 脚本全在里面，
拷到任意目录都能原地重编，不需要再去找开发机上的路径。

| 目录 | 是什么 | 什么时候需要 | 大小 |
|---|---|---|---|
| `opencv/include/` | 交叉编译版 OpenCV 头（整个 `opencv2/` 树） | 编译期 | 7.3 MB |
| `opencv/lib/` | OpenCV **静态库** `.a`（7 个模块 + `tegra_hal`/`zlib`） | 编译期 | 25 MB |
| `boardlibs/` | 板卡 `libOpenCL.so`（OpenCL ICD） | 编译期 + 运行时 | 70 KB |
| `rknnrt/` | 板卡 `librknnrt.so`（RKNN Runtime）、`librga.so`（RGA） | 编译期 + 运行时 | 7.5 MB |

## 编译期怎么用

`build_lib.sh` / `build_demo.sh` **默认就用这个目录里的副本**
（路径相对脚本自身 ⇒ 拷到哪都能编）。要换掉它，用这三个环境变量：

```bash
export OPENCV_INSTALL=/path/to/opencv-arm64-install   # 默认 <包>/third_party/opencv
export BOARD_LIBS=/path/to/boardlibs                  # 默认 <包>/third_party/boardlibs
export RT_LIB_DIR=/path/to/npu                        # 默认 <包>/third_party/rknnrt
```

交叉编译器**不在包里**（编译器没法随包发），用这个指定：

```bash
export RK3588_TOOLCHAIN=/path/to/toolchain-root       # 可以用 ':' 分隔多个候选，按顺序试
export CROSS_CXX=/path/to/aarch64-linux-gnu-g++       # 可选：直接指定编译器
export CROSS_AR=/path/to/aarch64-linux-gnu-ar         # 可选：直接指定 ar
```

## 运行时怎么用

OpenCV 是**静态链接**的 ⇒ **板卡不需要装 OpenCV**。

`librknnrt.so` / `librga.so` / `libOpenCL.so` 是运行时依赖。
板卡 BSP 里一般已经有了（`ldconfig -p | grep -E 'rknnrt|librga|OpenCL'` 查一下）。
如果你们的板卡没有，把这三个文件装到板卡上：

```bash
# ⚠️ 板卡 NFS 是 soft 挂载：>1 MB 的文件从 NFS 读会被读坏
#    ⇒ 必须先 cat 到板卡 /tmp（tmpfs）再落到 /usr/lib，并对账 md5
ssh <板卡> 'cat /mnt/manu/mosaic_sdk/third_party/rknnrt/librknnrt.so > /tmp/librknnrt.so && \
            cat /mnt/manu/mosaic_sdk/third_party/rknnrt/librga.so.2   > /tmp/librga.so.2 && \
            cat /mnt/manu/mosaic_sdk/third_party/boardlibs/libOpenCL.so.1 > /tmp/libOpenCL.so.1 && \
            md5sum /tmp/librknnrt.so /tmp/librga.so.2 /tmp/libOpenCL.so.1'
# 三个 md5 与本包 third_party/MANIFEST.md 里的一致 ⇒ 传输没坏
ssh <板卡> 'install -m755 /tmp/librknnrt.so /usr/lib/ && \
            install -m755 /tmp/librga.so.2 /usr/lib/ && \
            install -m755 /tmp/libOpenCL.so.1 /usr/lib/ && ldconfig'
```

不想装进系统也行 —— 用 `LD_LIBRARY_PATH` 指到本包（**同样先 cat 到 /tmp**）：

```bash
export LD_LIBRARY_PATH=/tmp/tp:$LD_LIBRARY_PATH
```

## 逐文件 md5

见同目录的 [`MANIFEST.md`](MANIFEST.md)（本目录每个文件一行，含 md5 与字节数）。
TPREADME
chmod 0644 "$DEST/third_party/README.md"
echo "  [ok] $DEST/third_party/README.md"

# 这两份说明文件登记进顶层清单（依赖资产本身由 third_party/MANIFEST.md 逐条登记）
EXPECTED+=("third_party/MANIFEST.md")
EXPECTED+=("third_party/README.md")

# ---------------------------------------------------------------------------
# 5. 清理陈旧文件（不在本次清单里的旧文件）—— 用 find -delete，不用 rm -rf
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
echo "  [ok] 删除 $stale 个陈旧文件"

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
TP_COUNT="$TP_N"                       # 第三方依赖资产数（不含 third_party/ 下那两份说明文件）
FILE_COUNT=$(( ${#EXPECTED[@]} + 1 ))   # + MANIFEST.md 自身
# out/ 的产物**不进受管清单**（md5 会随构建变），单独计数后再加总
if [ -d "$DEST/out" ]; then OUT_COUNT=$(find "$DEST/out" -type f | wc -l); else OUT_COUNT=0; fi
# 磁盘上真实存在的文件总数 = 受管（含 third_party 两份说明）+ 依赖资产 + 编译产物
TOTAL_COUNT=$(( FILE_COUNT + TP_COUNT + OUT_COUNT ))
TP_SIZE="$(du -sh "$DEST/third_party" 2>/dev/null | cut -f1)"
OBJ_MD5=$(md5_or_placeholder "$DEST/out/mosaic_pipeline.o")
LIB_A_MD5=$(md5_or_placeholder "$DEST/out/libmosaic_pipeline.a")
API_TEST_MD5=$(md5_or_placeholder "$DEST/out/mosaic_api_test")
CLIENT_EXAMPLE_MD5=$(md5_or_placeholder "$DEST/out/mosaic_client_example")
CONTRACT_TEST_MD5=$(md5_or_placeholder "$DEST/out/mosaic_contract_test")

README_TMP="$(mktemp)"
fill_placeholders "$README_TPL" "$README_TMP"
check_no_placeholder "$README_TMP"
echo "  [ok] README 内容已生成（计数：$TOTAL_COUNT = $FILE_COUNT + $TP_COUNT + $OUT_COUNT / $SRC_COUNT / $HDR_COUNT / $KERNEL_COUNT）"

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
  echo "> ⚠️ **本段刻意不列 md5**：\`ar\` 会把成员时间戳写进归档索引，"
  echo "> \`out/libmosaic_pipeline.a\` 的 md5 **每次构建都变**，列进来只会误导人。"
  echo "> 需要核对产物时手动 \`md5sum out/*\`：\`mosaic_pipeline.o\` 与两个可执行"
  echo "> 文件的 md5 是稳定的，\`.a\` 的不是。"
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
    third_party/*)
      # third_party 的**依赖资产**由 third_party/MANIFEST.md 逐条登记（220+ 行），
      # 顶层清单只登记那两份说明文件 ⇒ 资产文件不参与本检查
      case "$rel" in
        third_party/MANIFEST.md|third_party/README.md) ;;
        *) continue ;;
      esac
      ;;
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
tp_n=$(find "$DEST/third_party" -type f 2>/dev/null | wc -l)
all_n=$(find "$DEST" -type f | wc -l)
echo
echo "== 完成 =="
echo "  build/ 文件总数 : $all_n"
echo "  源码/脚本      : $src_n 个（src/）"
echo "  第三方头       : $inc_n 个（include/）"
echo "  kernel 资产    : $ker_n 个（runtime/kernels/）"
echo "  第三方依赖     : $tp_n 个（third_party/，含 2 份说明文件）← OpenCV/libOpenCL/librknnrt/librga"
echo
echo "  下一步（⚠️ 顺序不能颠倒）："
echo "    cd $DEST && bash build_lib.sh && bash build_demo.sh   # ① 编译（库 + 4 个测试程序）"
echo "    cd <repo>/manu/pipeline/npu && bash packaging/make_build_dir.sh   # ② 再打包一次"
echo "         └─ 把新产物的 md5 和「包内文件总数」写回 README/MANIFEST"
echo "           （不重跑的话，交付出去的 README 会写着「（尚未编译）」+ 文件数少 6）"
echo "    cd $DEST && bash copy_to_nfs.sh --verify              # ③ 拷贝 + 上板验证"
echo "         └─ 已编好时用 --verify；想让它自己编用 --build --verify（会自动补②）"
