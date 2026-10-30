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
# 用途说明（写进 MANIFEST.md 的表里）
# ---------------------------------------------------------------------------
purpose_of() {
  case "$1" in
    README.md)                    echo "使用说明（四条命令 + 交付/零参数/NFS/一键拷贝 各节）" ;;
    MANIFEST.md)                  echo "**本交付清单**（由 make_build_dir.sh 生成）" ;;
    build_lib.sh)                 echo "构建脚本①：只编译库 -> out/libmosaic_pipeline.a" ;;
    build_demo.sh)                echo "构建脚本②：只编译测试程序 -> out/mosaic_api_test + out/mosaic_contract_test" ;;
    copy_to_nfs.sh)               echo "一键脚本：编译 + 拷贝到 NFS + md5 回验 +（可选）板卡验证" ;;
    src/npu/mosaic_pipeline.cpp)  echo "库实现（GMC/OpenCL + RGA mosaic + RKNN fp16 + 反投影）" ;;
    src/npu/mosaic_pipeline.h)    echo "**公开 C ABI 头**（集成方写代码时 include 它）" ;;
    src/npu/mosaic_core.h)        echo "库内部核心头（GMC / RGA / RKNN 胶水层）" ;;
    src/npu/mosaic_api_test.cpp)  echo "示例 demo 源（只调公开 C ABI）" ;;
    src/npu/mosaic_contract_test.cpp) echo "契约测试源（验收门禁）" ;;
    src/cpp/gmc_stream_ocl.cpp)   echo "GMC/OpenCL 引擎（被 mosaic_pipeline.cpp 以 ../cpp/ 相对包含）" ;;
    src/cpp/md5.h)                echo "MD5 实现（被 gmc_stream_ocl.cpp 以 md5.h 相对包含）" ;;
    src/opencl/host/ocl_host.h)   echo "OpenCL host 封装（被 gmc_stream_ocl.cpp 以 ../opencl/host/ 相对包含）" ;;
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
    out/libmosaic_pipeline.a)  echo "**静态库**（客户链接它）" ;;
    out/mosaic_api_test)       echo "示例 demo 可执行文件（给客户的示例）" ;;
    out/mosaic_contract_test)  echo "契约测试可执行文件（验收门禁）" ;;
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
echo "== [1/6] 检查暂存的第三方头 =="
[ -d "$STAGE" ] || die "缺少暂存的第三方头：先跑一次 MOSAIC_NO_DEPLOY=1 bash build_mosaic_pipeline_arm64.sh（它会自动 stage 这些头）"
[ -d "$STAGE/rga" ] || die "暂存的第三方头缺少 rga/：$STAGE/rga"
[ -d "$STAGE/CL" ]  || die "暂存的第三方头缺少 CL/：$STAGE/CL"
echo "  [ok] $STAGE（rga/ + CL/ 齐备）"

# ---------------------------------------------------------------------------
# 1. 拷贝源码、脚本、kernel 资产
# ---------------------------------------------------------------------------
echo
echo "== [2/6] 拷贝源码 / 脚本 / kernel 资产 =="
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
copy_one "$ROOT/mosaic_api_test.cpp"            "src/npu/mosaic_api_test.cpp"
copy_one "$ROOT/mosaic_contract_test.cpp"       "src/npu/mosaic_contract_test.cpp"
# 被相对包含的引擎 / 工具 / OpenCL host
copy_one "$ROOT/../cpp/gmc_stream_ocl.cpp"      "src/cpp/gmc_stream_ocl.cpp"
copy_one "$ROOT/../cpp/md5.h"                   "src/cpp/md5.h"
copy_one "$ROOT/../opencl/host/ocl_host.h"       "src/opencl/host/ocl_host.h"
# 运行时 OpenCL kernel 资产
copy_one "$ROOT/../opencl/kernels/warp_median_fused.cl"  "runtime/kernels/warp_median_fused.cl"
copy_one "$ROOT/../opencl/kernels/sortnet_generated.inc" "runtime/kernels/sortnet_generated.inc"
copy_one "$ROOT/../opencl/kernels/gmc_feature_gpu.cl"    "runtime/kernels/gmc_feature_gpu.cl"
# 脚本与文档（packaging/ 下是原件，拷进 build/ 后是运行版）
copy_one "$HERE/README.md"          "README.md"
copy_one "$HERE/build_lib.sh"       "build_lib.sh"
copy_one "$HERE/build_demo.sh"      "build_demo.sh"
copy_one "$HERE/copy_to_nfs.sh"     "copy_to_nfs.sh"
chmod +x "$DEST/build_lib.sh" "$DEST/build_demo.sh" "$DEST/copy_to_nfs.sh"

# ---------------------------------------------------------------------------
# 2. 拷贝第三方头
# ---------------------------------------------------------------------------
echo
echo "== [3/6] 拷贝第三方头（include/）=="
for h in rknn_api.h stb_image.h stb_image_resize.h stb_image_write.h; do
  copy_one "$STAGE/$h" "include/$h"
done
for sub in rga CL; do
  while IFS= read -r f; do
    copy_one "$f" "include/$sub/$(basename "$f")"
  done < <(find "$STAGE/$sub" -type f | sort)
done

# ---------------------------------------------------------------------------
# 3. 清理陈旧文件（不在本次清单里的旧文件）—— 用 find -delete，不用 rm -rf
# ---------------------------------------------------------------------------
echo
echo "== [4/6] 清理陈旧文件 =="
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
# 4. 生成 build/MANIFEST.md（模板 + 自动生成的 md5 表）
# ---------------------------------------------------------------------------
echo
echo "== [5/6] 生成 MANIFEST.md =="
TEMPLATE="$HERE/MANIFEST.template.md"
[ -f "$TEMPLATE" ] || die "缺模板：$TEMPLATE"

TABLE="$(mktemp)"
trap 'rm -f "$TABLE"' EXIT

{
  echo "**共 ${#EXPECTED[@]} 个交付文件（不含 \`out/\` 与本清单自身）。**"
  echo
  echo "### 交付文件清单"
  echo
  echo "| 路径 | md5 | 字节数 | 用途 |"
  echo "|---|---|---:|---|"
  for rel in $(printf '%s\n' "${EXPECTED[@]}" | sort); do
    f="$DEST/$rel"
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

{
  sed -n '1,/<!-- MD5_TABLE:BEGIN -->/p' "$TEMPLATE" | sed '$d'
  cat "$TABLE"
  sed -n '/<!-- MD5_TABLE:END -->/,$p' "$TEMPLATE"
} > "$DEST/MANIFEST.md"

echo "  [ok] $DEST/MANIFEST.md（模板 + 自动生成表）"

# ---------------------------------------------------------------------------
# 5. 回验：表内 md5 与磁盘一致；表外不该有漏登记的文件
# ---------------------------------------------------------------------------
echo
echo "== [6/6] 回验 MANIFEST =="
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
  case "$rel" in out/*|MANIFEST.md) continue ;; esac
  if ! grep -qE "^\| \`$rel\` \| \`[0-9a-f]{32}\` \|" "$DEST/MANIFEST.md"; then
    echo "  [ERROR] 磁盘上有但清单里没登记：$rel" >&2
    extra=$((extra + 1))
  fi
done < <(find "$DEST" -type f | sort)
[ "$extra" = "0" ] || die "有 $extra 个文件漏登记（清单不完整）"
echo "  [ok] 无漏登记文件"

# ---------------------------------------------------------------------------
# 6. 摘要
# ---------------------------------------------------------------------------
src_n=$(find "$DEST/src" -type f 2>/dev/null | wc -l)
inc_n=$(find "$DEST/include" -type f 2>/dev/null | wc -l)
ker_n=$(find "$DEST/runtime" -type f 2>/dev/null | wc -l)
all_n=$(find "$DEST" -type f | wc -l)
echo
echo "== 完成 =="
echo "  build/ 文件总数 : $all_n"
echo "  源码/脚本      : $src_n 个（src/）"
echo "  第三方头       : $inc_n 个（include/）"
echo "  kernel 资产    : $ker_n 个（runtime/kernels/）"
echo
echo "  下一步："
echo "    cd $DEST && bash build_lib.sh && bash build_demo.sh   # 编译"
echo "    cd $DEST && bash copy_to_nfs.sh --build --verify     # 拷贝 + 上板验证"
