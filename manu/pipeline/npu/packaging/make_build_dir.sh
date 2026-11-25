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
#         五个测试程序用 `-lmosaic_pipeline` + `-Wl,-rpath,'$ORIGIN'` 链接它
#         ⇒ 整包可搬、可直接跑，不需要设 LD_LIBRARY_PATH。
#
#   ⚠️ 头与运行库必须**同版本**。
#      【2026-11-25 新板（Orange Pi 5 / Debian 12）】板上运行库是 **4.8**
#      （libopencv_*.so.408），但板上**只有 4.6 头**（/usr/include/opencv4）
#      ⇒ 4.8 头必须**外部提供**：本包把它随包发布到 **include/opencv4/**
#      （OpenCV 4.8 头，第三方 BSD 许可）⇒ 集成方在板上可直接原地构建。
#      旧板（4.6 头 + 4.6 库）不受影响：头与库仍然必须同版本。
# ---------------------------------------------------------------------------
#
# 用法: bash manu/pipeline/npu/packaging/make_build_dir.sh
# ===========================================================================
set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
# 2026-11-23（铁律二十五）：交付包是**中间产物**，默认落仓库外的 cache 区，
# 仓库内的 build/ 只是历史默认；用 MOSAIC_SDK_BUILD_DIR 可覆盖到任意目录。
DEST="${MOSAIC_SDK_BUILD_DIR:-$ROOT/build}"
# 第三方头暂存区（rga/ + CL/ + rknn_api.h + stb_image.h）——
# 🔴 铁律二十三：中间产物一律出仓库目录 ⇒ 默认落 cache（仓库外）；
#    可用 MOSAIC_STAGE_DIR 覆盖。旧口径 $HERE/.build-include 已废止。
STAGE="${MOSAIC_STAGE_DIR:-/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/pipeline-artifacts/npu_stage_include}"
mkdir -p "$STAGE"

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
TOOLCHAIN_V="${RK3588_TOOLCHAIN:-/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu}"

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
# 🔴 2026-11-25（实际部署板 = Orange Pi 5 / Debian 12）：这两个默认值**必须**改。
#    板上 OpenCV **运行库 4.8** 在 /usr/lib（libopencv_*.so.408），
#    而**系统包 4.6** 在 /usr/lib/aarch64-linux-gnu（libopencv_*.so.406）；
#    板上 app 与本 SDK 链的都是 **4.8** ⇒ 默认必须是 /usr/lib。
#    板上**只有 4.6 头**（/usr/include/opencv4）⇒ 头必须用**本包随包发布的
#    4.8 头**（$DEST/include/opencv4，源由 OPENCV_HEADERS_SRC 指定）。
OPENCV_INCLUDE_DIR_V="${OPENCV_INCLUDE_DIR:-$DEST/include/opencv4}"
OPENCV_LIB_DIR_V="${OPENCV_LIB_DIR:-/usr/lib}"
OPENCV_V="${OPENCV_INSTALL:-/usr/include/opencv4}"     # 旧键名（兼容用；现在等价于板卡 OpenCV 头目录）
BOARD_LIBS_V="${BOARD_LIBS:-/usr/lib/aarch64-linux-gnu}"
RT_LIB_DIR_V="${RT_LIB_DIR:-/usr/lib}"
NFS_ROOT_V="${NFS_ROOT:-/run/media/manu/SSD960-LOCAL/nfs}"
# RKNN 模型（.rknn）—— 本轮起**随包交付**，放在 models/ 下。
#   默认从 NFS 的 ptq/models/ 取；可用 MODEL_RKNN 指向别的副本。
#   设 MODEL_RKNN="" 显式表示「不打包模型」（只给不想带模型的场合）。
MODEL_RKNN="${MODEL_RKNN-$NFS_ROOT_V/ptq/models/p0_off_A_none.rknn}"
BOARD_V="${BOARD:-root@192.168.1.115}"
BOARD_PORT_V="${BOARD_PORT:-22}"

# ---------------------------------------------------------------------------
# 目标板 OpenCV 的 **ABI / 版本**（2026-11-25 新板 Orange Pi 5 / Debian 12）
#
#   板上运行库 = 4.8（/usr/lib/libopencv_*.so.408，SONAME 后缀 408）
#   板上**头**   = 只有 4.6（/usr/include/opencv4，CV_VERSION_MINOR 6）
#   ⇒ 头与运行库**必须同版本**，否则是 ABI 错配（同进程加载两套 OpenCV，
#     cv::Mat 跨版本串味）⇒ 本包把 **4.8 头**随包发布到 include/opencv4/。
#
#   OPENCV_ABI 是运行库的 SONAME 后缀（408 ⇒ OpenCV 4.8；406 ⇒ 4.6）。
#   头文件版本必须**精确匹配**它，否则一律拒绝（板卡 /usr/include/opencv4
#   那份 4.6 头**不能**拿来配 4.8 运行库）。
# ---------------------------------------------------------------------------
OPENCV_ABI_V="${OPENCV_ABI:-408}"
case "$OPENCV_ABI_V" in
  [0-9][0-9][0-9]) ;;
  *) die "OPENCV_ABI 必须是三位数字（OpenCV 运行库的 SONAME 后缀，如 408）：$OPENCV_ABI_V" ;;
esac
OCV_MAJOR="${OPENCV_ABI_V:0:1}"          # 408 -> 4
OCV_MINOR=$((10#${OPENCV_ABI_V:1}))      # 408 -> 8（408=>4.8, 406=>4.6, 410=>4.10）
OCV_VER="${OCV_MAJOR}.${OCV_MINOR}"      # 408 -> 4.8

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
    -e "s|\{\{STAT_COMPAT_OBJ_MD5\}\}|$STAT_COMPAT_OBJ_MD5|g" \
    -e "s|\{\{LIB_SO_MD5\}\}|$LIB_SO_MD5|g" \
    -e "s|\{\{LIB_SO_SIZE\}\}|$LIB_SO_SIZE|g" \
    -e "s|\{\{LIB_A_MD5\}\}|$LIB_A_MD5|g" \
    -e "s|\{\{API_TEST_MD5\}\}|$API_TEST_MD5|g" \
    -e "s|\{\{API_TEST_STATIC_MD5\}\}|$API_TEST_STATIC_MD5|g" \
    -e "s|\{\{CLIENT_EXAMPLE_MD5\}\}|$CLIENT_EXAMPLE_MD5|g" \
    -e "s|\{\{ASYNC_EXAMPLE_MD5\}\}|$ASYNC_EXAMPLE_MD5|g" \
    -e "s|\{\{CONTRACT_TEST_MD5\}\}|$CONTRACT_TEST_MD5|g" \
    -e "s|\{\{TRACK_FUSION_OBJ_MD5\}\}|$TRACK_FUSION_OBJ_MD5|g" \
    -e "s|\{\{E2E_MD5\}\}|$E2E_MD5|g" \
    -e "s|\{\{CADENCE_PROBE_MD5\}\}|$CADENCE_PROBE_MD5|g" \
    -e "s|\{\{TRACK_DUMP_MD5\}\}|$TRACK_DUMP_MD5|g" \
    -e "s|\{\{TRACKED_CB_EXAMPLE_MD5\}\}|$TRACKED_CB_EXAMPLE_MD5|g" \
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

# 产物字节数；文件不存在（还没编译）时填「（尚未编译）」
size_or_placeholder() {
  local f="$1"
  if [ -f "$f" ]; then fmt_num "$(stat -c%s "$f")"; else echo "（尚未编译）"; fi
}

# ---------------------------------------------------------------------------
# 用途说明（写进 MANIFEST.md 的表里）
# ---------------------------------------------------------------------------
purpose_of() {
  case "$1" in
    README.md)                    echo "使用说明（四条命令 + 交付/零参数/NFS/一键拷贝 各节）" ;;
    MANIFEST.md)                  echo "**本交付清单**（由 make_build_dir.sh 生成）" ;;
    build_lib.sh)                 echo "构建脚本①：只编译库 -> out/libmosaic_pipeline.so（**共享库**，SONAME=libmosaic_pipeline.so）" ;;
    build_demo.sh)                echo "构建脚本②：只编译测试程序 -> out/mosaic_api_test + out/mosaic_client_example + out/mosaic_async_client_example + out/mosaic_contract_test + out/rknn_mosaic_e2e（用 -lmosaic_pipeline + -rpath \$ORIGIN 链接同目录的 .so）" ;;
    copy_to_nfs.sh)               echo "一键脚本：编译 + 拷贝到 NFS + md5 回验 +（可选）板卡验证" ;;
    fetch_board_libs.sh)           echo "把**板卡自带的**库与头取回宿主机（ssh 打包 → tar → 解包），并打印 build_lib.sh / build_demo.sh 要用的四个路径变量" ;;
    fetch_opencv_headers.sh)       echo "**取 OpenCV $OCV_VER 头**的辅助脚本（从本地目录 --src 或从板卡 --board 拉取，校验版本必须是 $OCV_VER），供集成方自行补齐头文件" ;;
    known_hosts)                  echo "板卡主机密钥基线（ssh 严格校验用；板卡换密钥会直接失败，这是故意的）" ;;
    src/npu/mosaic_pipeline.cpp)  echo "库实现（GMC/OpenCL + RGA mosaic + RKNN fp16 + 反投影）" ;;
    src/npu/mosaic_pipeline.h)    echo "**公开 C ABI 头**（集成方写代码时 include 它）" ;;
    src/npu/mosaic_track_fusion.cpp) echo "时空（帧间）后处理内核实现（独立编译单元，零第三方依赖；**必须**带 -ffp-contract=off 才能与 Python 参考逐位一致）" ;;
    src/npu/mosaic_track_fusion.h)   echo "时空后处理内核头（参数结构 + 引擎声明，库内部用）" ;;
    src/npu/mosaic_core.h)        echo "库内部核心头（GMC / RGA / RKNN 胶水层 + 后处理三路径）" ;;
    src/npu/mosaic_post_gpu.h)    echo "GPU 后处理路径（OpenCL 局部极大；kernel 已内嵌，**不需要**外置 .cl）" ;;
    src/npu/mosaic_paths.h)        echo "**板卡无关的默认路径解析头**（demo / 探针用：env > 交付包内 > 板卡部署路径；换板不用改代码）" ;;
    src/npu/mosaic_api_test.cpp)  echo "示例 demo 源（只调公开 C ABI）" ;;
    src/npu/mosaic_client_example.cpp) echo "**最简集成示范源**（嵌入方照着抄；含 set_post_mode 钉死 sparse）" ;;
    src/npu/mosaic_async_client_example.cpp) echo "**多线程生产-消费 + 回调模板源**（嵌入式组的工程级集成骨架：相机线程不阻塞 + 严格一推一收 + 回调分发）" ;;
    src/npu/mosaic_contract_test.cpp) echo "契约测试源（验收门禁）" ;;
    src/npu/rknn_mosaic_e2e.cpp)  echo "**端到端测试程序源**（吞吐/延时测量，只调公开 C ABI）" ;;
    src/npu/mosaic_cadence_probe.cpp) echo "**板卡验证探针源**（节奏 / Δn / 帧间隔 + 时空融合参数回读；**带交付工作点的显式 9 字段覆盖**）" ;;
    src/npu/mosaic_track_dump.cpp) echo "**板卡验证探针源**（逐帧导出候选流 / 融合结果 TSV，供离线对 GT 打分；**带交付工作点的显式 9 字段覆盖**）" ;;
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
    include/opencv4/*)            echo "**OpenCV $OCV_VER 头**（第三方，BSD 许可，随包发布；板卡上只有 4.6 头，不能拿来配 $OCV_VER 运行库 ⇒ 集成方必须用这份）" ;;
    models/p0_off_A_none.rknn)    echo "**RKNN 模型**（fp16，640x640x3 -> 2x[1,1,320,320]）—— 随包交付，客户可直接用；板上使用前需先 cat 到 /tmp（NFS soft 挂载直读会被读坏）" ;;
    *)                            echo "—" ;;
  esac
}

# out/ 段的说明（**刻意不列 md5**）
out_note_of() {
  case "$1" in
    out/mosaic_pipeline.o)     echo "编译中间产物（库的 .o）" ;;
    out/mosaic_track_fusion.o)  echo "编译中间产物（时空后处理内核的 .o）" ;;
    out/stat_compat.o)          echo "编译中间产物（**buildroot glibc 2.29 的 \`stat\` hidden symbol 垫片**，铁律十八；链接可执行文件时必需）" ;;
    out/libmosaic_pipeline.so) echo "**共享库**（客户链接它；SONAME=libmosaic_pipeline.so；动态链接板卡自带库）" ;;
    out/libmosaic_pipeline.a)  echo "**静态库**（客户要静态链接时用这个；与同名 .so **同源同对象文件** => 行为逐字节一致；**不含** stat_compat.o 垫片，链接可执行文件时仍需按目标板 glibc 版本另行带上）" ;;
    out/mosaic_api_test)       echo "示例 demo 可执行文件（给客户的示例）" ;;
    out/mosaic_api_test_static) echo "**静态链接版的示例 demo**（mosaic_api_test 链 .a 编出来；**静态库可用性的活证据**；运行输出与链 .so 的版本逐字节一致）" ;;
    out/mosaic_client_example) echo "**最简集成示范可执行文件**（嵌入方照着抄这份）" ;;
    out/mosaic_async_client_example) echo "**多线程生产-消费 + 回调模板可执行文件**（工程级集成骨架；嵌入式组照着这份搭自己的相机线程）" ;;
    out/mosaic_contract_test)  echo "契约测试可执行文件（验收门禁）" ;;
    out/rknn_mosaic_e2e)       echo "端到端测试程序（吞吐/延时测量）" ;;
    out/mosaic_cadence_probe)  echo "**板卡验证探针**（本地工具，不属于集成面）：逐帧推/收，量节奏（fps / Δn / 帧间隔）+ 打印时空融合的**有效参数回读**" ;;
    out/mosaic_track_dump)     echo "**板卡验证探针**（本地工具，不属于集成面）：逐帧导出候选流 / 融合结果 TSV，供离线对 GT 打分" ;;
    out/mosaic_tracked_cb_example) echo "**跟踪结果回调示范可执行文件**（v10.7）：注册 mosaic_pipeline_set_tracked_callback 拿**一目标一框**的跟踪结果，并与原始候选框做同帧对照" ;;
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
         "$DEST/runtime/kernels" "$DEST/include/rga" "$DEST/include/CL" \
         "$DEST/models"

copy_one() {
  local src="$1" rel="$2" quiet="${3:-}"
  [ -f "$src" ] || die "源文件不存在：$src"
  cp -f "$src" "$DEST/$rel"
  add_expected "$rel"
  # 第三个参数可选：传 quiet 则**不打印**这一行（登记进 EXPECTED 的效果一样）。
  #   OpenCV 头有 211 个文件，逐行打印会刷屏 ⇒ 用它静默登记。
  if [ "$quiet" != "quiet" ]; then
    printf "  [ok] %-46s <- %s\n" "$rel" "${src#$ROOT/}"
  fi
}

# 库与 demo 的源
copy_one "$ROOT/mosaic_pipeline.cpp"            "src/npu/mosaic_pipeline.cpp"
copy_one "$ROOT/mosaic_pipeline.h"              "src/npu/mosaic_pipeline.h"
# v10.5: the spatio-temporal track fusion kernel (own translation unit; the
# public C ABI to drive it lives in mosaic_pipeline.h).
copy_one "$ROOT/mosaic_track_fusion.cpp"        "src/npu/mosaic_track_fusion.cpp"
copy_one "$ROOT/mosaic_track_fusion.h"          "src/npu/mosaic_track_fusion.h"
copy_one "$ROOT/mosaic_core.h"                  "src/npu/mosaic_core.h"
copy_one "$ROOT/mosaic_post_gpu.h"              "src/npu/mosaic_post_gpu.h"
copy_one "$ROOT/mosaic_paths.h"                 "src/npu/mosaic_paths.h"
copy_one "$ROOT/mosaic_api_test.cpp"            "src/npu/mosaic_api_test.cpp"
copy_one "$ROOT/mosaic_client_example.cpp"      "src/npu/mosaic_client_example.cpp"
copy_one "$ROOT/mosaic_async_client_example.cpp" "src/npu/mosaic_async_client_example.cpp"
copy_one "$ROOT/mosaic_contract_test.cpp"       "src/npu/mosaic_contract_test.cpp"
copy_one "$ROOT/rknn_mosaic_e2e.cpp"            "src/npu/rknn_mosaic_e2e.cpp"
copy_one "$ROOT/mosaic_cadence_probe.cpp"       "src/npu/mosaic_cadence_probe.cpp"
copy_one "$ROOT/mosaic_track_dump.cpp"          "src/npu/mosaic_track_dump.cpp"
copy_one "$ROOT/mosaic_tracked_cb_example.cpp" "src/npu/mosaic_tracked_cb_example.cpp"
# 【2026-11-23 上游】glibc<2.33 的 `stat` 公开符号垫片（交叉编译可执行文件必需，
#   见 packaging/build_demo.sh 里的说明；没有它，链接 demo 会报
#   "hidden symbol `stat' ... is referenced by DSO"）。
copy_one "$ROOT/stat_compat.c"                   "src/npu/stat_compat.c"
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
copy_one "$HERE/fetch_opencv_headers.sh" "fetch_opencv_headers.sh"
copy_one "$HERE/build_board_native.sh" "build_board_native.sh"
copy_one "$HERE/build_probes.sh"      "build_probes.sh"
copy_one "$HERE/known_hosts"        "known_hosts"
chmod +x "$DEST/build_lib.sh" "$DEST/build_demo.sh" "$DEST/copy_to_nfs.sh" \
         "$DEST/fetch_board_libs.sh" "$DEST/fetch_opencv_headers.sh"
EXPECTED+=("README.md")   # 生成物，但在交付清单里（md5 在生成后才有）

# RKNN 模型（.rknn）—— **随包交付**（models/p0_off_A_none.rknn）。
#   为什么必须打包：交付包原来**一个 .rknn 都没有**，README 与交付脚本却让客户用
#   `/mnt/manu/ptq/models/p0_off_A_none.rknn`（**我们内部 NFS 的路径**）⇒
#   客户照抄命令必然失败（路径不存在）。模型进包后统一用包内路径。
#   注意：模型在 NFS 上是 soft 挂载，**板卡直读会被读坏** ⇒ 板上仍要先 cat 到 /tmp。
if [ -n "$MODEL_RKNN" ]; then
  [ -f "$MODEL_RKNN" ] || die "找不到 RKNN 模型：$MODEL_RKNN
       ⇒ 本包现在**自带模型**（models/p0_off_A_none.rknn）。
         可用 MODEL_RKNN=<路径> 指向别的副本；
         确实不想带模型时设 MODEL_RKNN=""（显式置空）。"
  copy_one "$MODEL_RKNN" "models/p0_off_A_none.rknn"
else
  echo "  [--] MODEL_RKNN 为空 ⇒ **本次不打包模型**（客户需自备 .rknn）"
fi

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
# 3. 【2026-11-25 新板】**OpenCV 4.8 头**随包发布到 include/opencv4/
#
#    板卡实测（/tmp/board115_facts.md；Orange Pi 5 / Debian 12）：
#      运行库 = 4.8   /usr/lib/libopencv_*.so.408（SONAME 后缀 408）
#      头     = **只有 4.6**  /usr/include/opencv4（CV_VERSION_MINOR 6）
#    ⇒ 板卡原生构建会「4.6 头配 4.8 运行库」= **ABI 错配**：
#      OpenCV 次版本会改 C++ ABI（cv::Mat 布局 / 内联函数），头与库不同版本
#      ⇒ 同一进程里会同时加载两套 OpenCV，cv::Mat 跨版本串味
#      ⇒ 集成方在这块板上**无法原地构建**。
#    ⇒ 决定：把 OpenCV 4.8 头（211 个文件，7.2 MB，BSD 许可，可再分发）
#      **随包发布**到 include/opencv4/。
#
#    源目录：OPENCV_HEADERS_SRC 优先；未指定时按下面顺序自动搜索。
#    版本校验：源的 opencv2/core/version.hpp 里 CV_VERSION_MAJOR/MINOR 必须
#      **精确等于**目标 ABI（OPENCV_ABI=408 ⇒ 4.8）⇒ 不符一律 die。
#      （板卡 /usr/include/opencv4 那份 4.6 头**不能**拿来配 4.8 运行库。）
#    找不到头 ⇒ **die**（不许静默产出一个「在板上编不了」的包）；
#      实在要跳过：SKIP_OPENCV_HEADERS=1（会打**醒目警告**）。
# ---------------------------------------------------------------------------

# 读 <dir>/opencv2/core/version.hpp → "4.8"；读不到返回非 0
ocv_version_of() {
  local d="$1" hdr="$1/opencv2/core/version.hpp"
  [ -f "$hdr" ] || return 1
  local major minor
  major=$(sed -nE 's/^[[:space:]]*#[[:space:]]*define[[:space:]]+CV_VERSION_MAJOR[[:space:]]+([0-9]+).*/\1/p' "$hdr" | head -1)
  minor=$(sed -nE 's/^[[:space:]]*#[[:space:]]*define[[:space:]]+CV_VERSION_MINOR[[:space:]]+([0-9]+).*/\1/p' "$hdr" | head -1)
  [ -n "$major" ] && [ -n "$minor" ] || return 1
  printf '%s.%s\n' "$major" "$minor"
}

# 版本不符 ⇒ die（必须点名实际版本与期望版本）
ocv_die_version() {
  local actual="$1" src="$2"
  die "OpenCV 头版本不匹配：$src 是 **OpenCV $actual** 的头，
       目标 ABI 需要 **OpenCV $OCV_VER**（运行库 SONAME 后缀 libopencv_*.so.$OPENCV_ABI_V）。
     ⇒ ⚠️ 本板（Orange Pi 5 / Debian 12）上 /usr/include/opencv4 是 **4.6 头**，
       **不能**拿来配 **4.8 运行库**：OpenCV 次版本会改 C++ ABI
       （cv::Mat 布局 / 内联函数）⇒ 同一进程会同时加载两套 OpenCV，
       cv::Mat 跨版本串味。
     ⇒ 请换一份 OpenCV $OCV_VER 的头：用包内 fetch_opencv_headers.sh 取
       （--src <本地头目录> 或 --board <板卡>），或手工从 OpenCV $OCV_VER
       源码的 modules/*/include 拼出 opencv2/ 树。"
}

if [ "${SKIP_OPENCV_HEADERS:-0}" = "1" ]; then
  echo "  [!!] SKIP_OPENCV_HEADERS=1 ⇒ **本次不把 OpenCV 头放进包**"
  echo "  [!!] ╔═════════════════════════════════════════════════════════════╗"
  echo "  [!!] ║ 警告：这个交付包**无法在目标板上原生构建**！               ║"
  echo "  [!!] ║ 板上只有 OpenCV 4.6 头（/usr/include/opencv4），             ║"
  echo "  [!!] ║ 运行库却是 4.8（libopencv_*.so.$OPENCV_ABI_V）⇒ ABI 错配。   ║"
  echo "  [!!] ║ 集成方必须自己提供 OpenCV $OCV_VER 头（fetch_opencv_headers.sh）。║"
  echo "  [!!] ╚═════════════════════════════════════════════════════════════╝"
else
  # 候选源（OPENCV_HEADERS_SRC 未指定时按顺序找；第 4 项只有 4.8 才会被接受）
  OCV_CANDIDATES=(
    "$STAGE/opencv4"
    "/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/pipeline-artifacts/boardlibs_orangepi115_20261125/opencv/include/opencv4"
    "${HOME:-}/rk3588_boardlibs/opencv/include/opencv4"
    "/usr/include/opencv4"
  )
  OCV_SRC=""
  OCV_SEEN=""
  OCV_REJECTED=""
  if [ -n "${OPENCV_HEADERS_SRC:-}" ]; then
    OCV_SRC="${OPENCV_HEADERS_SRC%/}"
    [ -d "$OCV_SRC" ] || die "OPENCV_HEADERS_SRC 指定的目录不存在：$OCV_SRC"
  else
    for cand in "${OCV_CANDIDATES[@]}"; do
      [ -n "$cand" ] || continue
      OCV_SEEN="$OCV_SEEN
       - $cand"
      [ -d "$cand" ] || continue
      v="$(ocv_version_of "$cand" || true)"
      if [ "$v" = "$OCV_VER" ]; then
        OCV_SRC="$cand"
        break
      fi
      if [ -z "$v" ]; then
        OCV_REJECTED="$OCV_REJECTED
       - $cand （缺 opencv2/core/version.hpp，读不出版本）"
      else
        OCV_REJECTED="$OCV_REJECTED
       - $cand （是 OpenCV $v 的头，与目标 $OCV_VER 不符）"
      fi
    done
  fi

  if [ -z "$OCV_SRC" ]; then
    die "找不到 **OpenCV $OCV_VER** 的头（要随包发布到 include/opencv4/）。
       搜索过：${OCV_SEEN:-（无）}
       存在但版本不符的：${OCV_REJECTED:-（无）}
     ⇒ **怎么拿到 $OCV_VER 的头**：
       1) 用包内自带辅助脚本（集成方也能用）：
            bash fetch_opencv_headers.sh --src <本地 OpenCV $OCV_VER 头目录> --dest <落地目录>
            bash fetch_opencv_headers.sh --board root@<板卡IP> --board-path /usr/include/opencv4 --dest <落地目录>
       2) 手工：从 OpenCV $OCV_VER 源码的 modules/*/include 拼出 opencv2/ 树。
       3) 临时跳过（⚠️ 该包将**无法在这块板上原生构建**）：SKIP_OPENCV_HEADERS=1"
  fi

  # 版本必须精确匹配（显式指定的 OPENCV_HEADERS_SRC 也要过这一关）
  OCV_VER_ACTUAL="$(ocv_version_of "$OCV_SRC" || true)"
  [ -n "$OCV_VER_ACTUAL" ] || die "OpenCV 头目录里缺 opencv2/core/version.hpp：$OCV_SRC
       ⇒ 读不出 CV_VERSION_MAJOR/MINOR 就无法确认版本 ⇒ 拒绝使用
         （宁可打包失败，也不要配错 ABI）。"
  if [ "$OCV_VER_ACTUAL" != "$OCV_VER" ]; then
    ocv_die_version "$OCV_VER_ACTUAL" "$OCV_SRC"
  fi

  mkdir -p "$DEST/include/opencv4"
  OCV_N=0
  while IFS= read -r f; do
    rel="${f#$OCV_SRC/}"
    mkdir -p "$DEST/include/opencv4/$(dirname "$rel")"
    copy_one "$f" "include/opencv4/$rel" quiet
    OCV_N=$((OCV_N + 1))
  done < <(find "$OCV_SRC" -type f | sort)
  echo "  [ok] OpenCV $OCV_VER_ACTUAL 头：$OCV_N 个文件 -> include/opencv4/（源：$OCV_SRC）"
fi

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
#      五个测试程序用 `-lmosaic_pipeline` + `-Wl,-rpath,'$ORIGIN'` 链接它。
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
echo "       OpenCV **头**：随包发布在 include/opencv4/（OpenCV $OCV_VER，第三方 BSD 许可）"
echo "       编译期头 ：$OPENCV_INCLUDE_DIR_V"
echo "       链接期库 ：$OPENCV_LIB_DIR_V  $RT_LIB_DIR_V  $BOARD_LIBS_V"

# ---------------------------------------------------------------------------
# 5. 清理陈旧文件（不在本次清单里的旧文件）—— 一律 find -delete + rmdir
#
#    ⚠️ **目录级清理**：旧设计（第三方依赖随包）留下的 `third_party/` 有 220 个
#    文件，本包**不再携带任何第三方 .so/.a** ⇒ 整棵目录必须删掉，
#    否则旧包的副本会污染新包（MANIFEST 回验也会因「漏登记」直接失败）。
#    逐文件 delete 慢且会留下空目录 ⇒ 这里整棵删。
# ---------------------------------------------------------------------------
echo
echo "== [5/8] 清理陈旧文件 =="
stale=0
for sub in src include runtime models; do
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

# include/ 下因「删陈旧文件」而留下的空目录（例如上一次打包留下的 opencv4/）
#   也要清掉，否则包里会留一个空壳目录。用 find -delete + rmdir。
find "$DEST/include" -mindepth 1 -type d -empty -delete 2>/dev/null || true

# 旧包残留的 third_party/：整棵删掉（目录级）
if [ -d "$DEST/third_party" ]; then
  _tp_n=$(find "$DEST/third_party" -type f | wc -l)
  # 🔴 铁律：删目录 = find -mindepth 1 -delete（清内容）+ rmdir（删空目录）
  find "$DEST/third_party" -mindepth 1 -delete 2>/dev/null || true
  rmdir "$DEST/third_party" 2>/dev/null || true
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
LIB_SO_SIZE=$(size_or_placeholder "$DEST/out/libmosaic_pipeline.so")
LIB_A_MD5=$(md5_or_placeholder "$DEST/out/libmosaic_pipeline.a")
API_TEST_MD5=$(md5_or_placeholder "$DEST/out/mosaic_api_test")
API_TEST_STATIC_MD5=$(md5_or_placeholder "$DEST/out/mosaic_api_test_static")
CLIENT_EXAMPLE_MD5=$(md5_or_placeholder "$DEST/out/mosaic_client_example")
ASYNC_EXAMPLE_MD5=$(md5_or_placeholder "$DEST/out/mosaic_async_client_example")
CONTRACT_TEST_MD5=$(md5_or_placeholder "$DEST/out/mosaic_contract_test")
TRACK_FUSION_OBJ_MD5=$(md5_or_placeholder "$DEST/out/mosaic_track_fusion.o")
STAT_COMPAT_OBJ_MD5=$(md5_or_placeholder "$DEST/out/stat_compat.o")
E2E_MD5=$(md5_or_placeholder "$DEST/out/rknn_mosaic_e2e")
CADENCE_PROBE_MD5=$(md5_or_placeholder "$DEST/out/mosaic_cadence_probe")
TRACK_DUMP_MD5=$(md5_or_placeholder "$DEST/out/mosaic_track_dump")
TRACKED_CB_EXAMPLE_MD5=$(md5_or_placeholder "$DEST/out/mosaic_tracked_cb_example")

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
trap 'find "$TABLE" "$README_TMP" -maxdepth 0 -type f -delete 2>/dev/null || true' EXIT

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
find "$MANIFEST_TMP.raw" -maxdepth 0 -type f -delete 2>/dev/null || true
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
mdl_n=$(find "$DEST/models" -type f 2>/dev/null | wc -l)
if [ -d "$DEST/out" ]; then out_n=$(find "$DEST/out" -type f | wc -l); else out_n=0; fi
all_n=$(find "$DEST" -type f | wc -l)
echo
echo "== 完成 =="
echo "  build/ 文件总数 : $all_n"
echo "  源码/脚本      : $src_n 个（src/）"
echo "  第三方头       : $inc_n 个（include/）"
echo "  kernel 资产    : $ker_n 个（runtime/kernels/）"
echo "  RKNN 模型      : $mdl_n 个（models/）"
echo "  共享库         : out/libmosaic_pipeline.so（SONAME=libmosaic_pipeline.so）← 动态链接板卡自带库"
echo "  编译产物       : $out_n 个（out/：共享库 .so + **静态库 .a** + 测试程序 + 板卡验证探针 + .o）"
echo
echo "  下一步（⚠️ 顺序不能颠倒）："
echo "    cd $DEST && bash build_lib.sh && bash build_demo.sh && bash build_probes.sh   # ① 编译（库 + 测试程序 + 探针）"
echo "    cd <repo>/manu/pipeline/npu && bash packaging/make_build_dir.sh   # ② 再打包一次"
echo "         └─ 把新产物的 md5 和「包内文件总数」写回 README/MANIFEST"
echo "           （不重跑的话，交付出去的 README 会写着「（尚未编译）」+ 文件数少 6）"
echo "         └─ ⚠️ **完整顺序必须是 ①②③**（make_build_dir.sh 会覆盖正在运行的脚本，"
echo "            绝不能在同一脚本运行期间再调它：bash 按文件偏移读脚本，被覆盖后读到"
echo "            的就是错位内容 ⇒ 后半段静默不执行、退出码还可能是 0）。"
echo "            copy_to_nfs.sh --build 现在只做「编译」，且**编译前会校验包目录里的源码"
echo "            与工程根是否一致**（不一致直接 die），但**不会**替你同步、也不会替你回填 md5。"
