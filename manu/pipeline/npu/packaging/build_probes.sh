#!/usr/bin/env bash
# ===========================================================================
# build_probes.sh -- build the two BOARD-SIDE VERIFICATION PROBES
#
#   outputs: out/mosaic_track_dump
#            out/mosaic_cadence_probe
#
# ---------------------------------------------------------------------------
# 【为什么需要这个脚本】
#
#   两个探针是**板卡侧本地工具**（源码在包里，注释写明「不进交付库」的本意，
#   但 §267 把源码收进了包，让集成方/板卡线能自己重编）⇒ 在此之前**全网没有
#   任何脚本会编它们**，板上的二进制是手敲 g++ 编出来的（`.comment` =
#   GCC 9.4.0 Ubuntu 20.04 ⇒ 板卡原生编译）。
#
#   🔴 **跨尺度候选合并（2026-11-20）必须两个都重编**：
#      · 合并逻辑在 mosaic_pipeline.cpp（进 .so）
#      · 开关 --cand-dedup-radius 在 mosaic_track_dump.cpp / mosaic_cadence_probe.cpp
#      ⇒ **只重编一个 = 合并不生效**（板端 FAR 仍是 0.3832，红线 12.8 倍）。
#
#   探针**零第三方依赖**（只 include mosaic_pipeline.h，无 OpenCV/RKNN）
#   ⇒ 三个 OpenCV 变体（ocv46 / ocv48 / ocv42）**通用**，编一次到处可用。
#
#   用法:
#     ./build_probes.sh                 # 在**包根目录**跑（out/ 与 .so 同目录）
#     CROSS_CXX=<aarch64 交叉 g++> ./build_probes.sh    # 交叉编译
#     REQUIRE_MERGE=1 ./build_probes.sh # .so 里没有合并 API 就**直接失败**
#
#   环境变量:
#     CROSS_CXX     交叉编译器（默认用宿主机的 g++ ⇒ 板卡上原生编译）
#     PKG_ROOT      包根目录（默认 = 本脚本所在目录）
#     ALLOW_LEGACY  1 = 允许拿「修复前」的 .so 编探针（链接会失败，属预期）
#     REQUIRE_MERGE 1 = 强校验：.so 必须自带跨尺度候选合并能力（v10.6 起合并半径是库常量）
# ===========================================================================
set -euo pipefail
export LC_ALL=C

die() { echo "[ERROR] $*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PKG_ROOT：显式给 > 脚本所在目录就是包根（脚本在包内时）> 当前工作目录
# （脚本留在工程里、在包根目录里 `bash .../packaging/build_probes.sh` 时最顺手）。
if [ -n "${PKG_ROOT:-}" ]; then
  :
elif [ -d "$HERE/src/npu" ]; then
  PKG_ROOT="$HERE"
else
  PKG_ROOT="$PWD"
fi
SRC="$PKG_ROOT/src/npu"
OUT="$PKG_ROOT/out"
CXX="${CROSS_CXX:-g++}"

# 与库、Python 参考实现同一套数值开关（位级可复现的前提）。
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations"

[ -d "$SRC" ] || die "找不到探针源码目录：$SRC（PKG_ROOT 设错？）"
[ -d "$OUT" ]  || die "找不到 out/ 目录：$OUT（探针必须与 .so 同目录 ⇒ RPATH=\$ORIGIN）"
[ -f "$OUT/libmosaic_pipeline.so" ] || die "找不到 out/libmosaic_pipeline.so（先跑 build_lib.sh）"
[ -f "$SRC/mosaic_pipeline.h" ] || die "找不到 src/npu/mosaic_pipeline.h"

echo "== build probes =="
echo "  包根     : $PKG_ROOT"
echo "  编译器   : $CXX"
echo "  库       : $OUT/libmosaic_pipeline.so"

# ---------------------------------------------------------------------------
# 0) **前置闸门**：.so 必须自带「跨尺度候选合并」能力（v10.6 起合并半径是**库常量**）。
#
#    🔴 v10.6 之后的**新危险**：探针不再调用任何合并 setter（那个公开接口已删除），
#    改用环境变量 `MOSAIC_TRACK_MERGE_RADIUS` 把半径交给库 ⇒
#    **旧 .so 会「安静地」忽略这个环境变量** ⇒ 合并保持关闭 ⇒
#    板端逐帧 FAR 0.3832（红线 0.03 的 12.8 倍）—— **不报错、只是结果错**。
#    ⇒ 这里用符号把两种 .so 区分开：旧 .so 仍导出那个已删的 setter。
# ---------------------------------------------------------------------------
# 🔴 2026-11-23：`nm ... | grep -q` 在 `set -o pipefail` 下会踩 **SIGPIPE** 陷阱：
#    `grep -q` 一命中就退出 ⇒ 上游 `nm` 收到 SIGPIPE（退出码 141）⇒ **整条管道判失败**
#    ⇒ 一个明明导出了该符号的 .so 会被判成「没导出」。
#    （符号表比管道缓冲大时必然触发；小 .so 侥幸不触发 ⇒ 表现为「时灵时不灵」。）
#    ⇒ 改成**先把符号表抓进变量**，再在变量里 grep，全程没有管道。
SO_SYMS="$(nm -D --defined-only "$OUT/libmosaic_pipeline.so" 2>/dev/null || true)"
if printf '%s\n' "$SO_SYMS" | grep -q "mosaic_pipeline_set_candidate_dedup_radius"; then
  HAS_MERGE=0
  echo "  [!!] .so 仍导出 mosaic_pipeline_set_candidate_dedup_radius ⇒ 这是**旧契约**的 .so"
  echo "       ⇒ 旧 .so 不认 MOSAIC_TRACK_MERGE_RADIUS ⇒ 合并会是**关**的（静默错误！"
  echo "          板端 FAR 0.3832，红线 12.8 倍）⇒ 探针会跑出一份「看着正常、其实没合并」的结果。"
  if [ "${ALLOW_LEGACY:-0}" = "1" ]; then
    echo "       ⇒ ALLOW_LEGACY=1 ⇒ 继续编（仅用于与旧 .so 做对照实验）"
  else
    die "这个 .so 是**旧契约**（合并半径还是调用侧 setter，库内置那套没有生效）
       ⇒ 探针用环境变量传的半径会被**静默忽略** ⇒ 跑出来的 FAR 会是 0.3832。
       **先重编 .so，再编探针**（两者必须同版本）。
       确实要编「旧契约」的探针对：ALLOW_LEGACY=1（只用于对照实验）。
       .so 重编见 build_lib.sh；板卡侧原生编 .so 见 §267 的 /root/sdk 流程。"
  fi
elif printf '%s\n' "$SO_SYMS" | grep -q "MergeCandidatesByRadius"; then
  echo "  [ok] .so 自带跨尺度候选合并能力（MergeCandidatesByRadius，v10.6 库内置半径）"
  HAS_MERGE=1
else
  HAS_MERGE=0
  echo "  [!!] .so **没有**合并能力（既无 MergeCandidatesByRadius，也不是旧契约）"
  if [ "${ALLOW_LEGACY:-0}" = "1" ]; then
    echo "       ⇒ ALLOW_LEGACY=1 ⇒ 继续编（仅用于对照实验）"
  else
    die "这个 .so 里**没有**跨尺度候选合并能力 ⇒ 合并不会发生 ⇒ FAR 会是 0.3832。
       **先重编 .so，再编探针**（两者必须同版本）。"
  fi
fi

for probe in mosaic_track_dump mosaic_cadence_probe; do
  [ -f "$SRC/$probe.cpp" ] || die "缺少探针源码：$SRC/$probe.cpp"
  echo "  compiling $probe ..."
  # --allow-shlib-undefined：.so 自己还依赖板卡上的 libopencv_*.so.406 /
  # librknnrt / libOpenCL / librga，交叉编译时这些在本机**不存在** ⇒ 不放宽
  # 就会报一堆 "undefined reference to cv::..."（那些符号本来就该由板卡上的
  # 动态库在运行期提供）。与 build_lib.sh 的 RELAX 同一口径。
  "$CXX" $CXXFLAGS -I"$SRC" \
    -o "$OUT/$probe" "$SRC/$probe.cpp" \
    -L"$OUT" -lmosaic_pipeline -Wl,-rpath,'$ORIGIN' -Wl,--allow-shlib-undefined \
    -lpthread -ldl -lm
  chmod 0755 "$OUT/$probe"
done

echo
echo "== 自检 =="
for probe in mosaic_track_dump mosaic_cadence_probe; do
  if strings -a "$OUT/$probe" 2>/dev/null | grep -- "--cand-dedup-radius" >/dev/null 2>&1; then
    echo "  [ok] $probe 带 --cand-dedup-radius 开关"
  else
    echo "  [!!] $probe **没有** --cand-dedup-radius 开关（源码不是修复版？）"
  fi
  echo "  md5 : $(md5sum "$OUT/$probe" | cut -d' ' -f1)  $OUT/$probe"
done
echo "  .so md5 : $(md5sum "$OUT/libmosaic_pipeline.so" | cut -d' ' -f1)"

[ "$HAS_MERGE" = "1" ] || echo "  ⇒ 记住：.so 没重编之前，这套探针**不能**用来验证合并效果"
echo
echo "== 完成 =="
