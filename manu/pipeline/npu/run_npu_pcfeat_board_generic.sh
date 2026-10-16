#!/usr/bin/env bash
# =============================================================================
# run_npu_pcfeat_board_generic.sh -- 24 序列全量 PC 特征推理，模型可换
#
# 这是 run_npu_pcfeat_board_f16.sh 的参数化版本，唯一区别是配置块
# （MODEL / MODEL_MD5 / OUTROOT / DONE_MARKER）改为环境变量。
# 推理循环、重试语义、TSV 行数硬校验、缺失特征即 exit 3 全部与原脚本逐字一致，
# 可用 `diff <(sed -n '/^set -u/,$p' 原脚本) <(sed -n '/^set -u/,$p' 本脚本)` 自查。
#
# 为什么需要它：那套脚本里模型路径与 md5 是写死的，而混合精度模型有三个候选
# （hybridL1/L2/L3），每一臂都必须绑定自己的 md5 硬校验——否则「跑的到底是哪个
# 模型」就只能靠人记，正是 md5 铁律要堵的洞。
#
# 用法：
#   MODEL=... MODEL_MD5=... OUTROOT=... TAG=w0 DONE_MARKER=... \
#     ./run_npu_pcfeat_board_generic.sh <序列清单> <标识>
#
# 硬校验（同原脚本，缺一不可）：
#   * 模型存在且 md5 与声明一致，否则 exit 2
#   * 每个序列的 npu_det.tsv 行数必须等于 expected.tsv 的 want，否则记 FAIL
#   * 特征图缺失立刻 exit 3
#   * 全部结束后打印 DONE_MARKER，供流水线轮询
# =============================================================================
set -u

FEAT=${FEAT:-/mnt/manu/pcfeat_bmp/images}
MODEL=${MODEL:?必须显式指定 MODEL}
MODEL_MD5=${MODEL_MD5:?必须显式指定 MODEL_MD5}
OUTROOT=${OUTROOT:?必须显式指定 OUTROOT}
NPU=${NPU:-/root/rknn_heatmap_smoke}
DONE_MARKER=${DONE_MARKER:-NPU_PCFEAT_ALL_DONE}
TH=${TH:-0.08}
TOPK=${TOPK:-80}
EXPECT=${EXPECT:-/mnt/manu/npu/out/demo/expected.tsv}
LIST=${1:-$EXPECT}
TAG=${2:-w0}

mkdir -p "$OUTROOT"
[ -f "$MODEL" ] || { echo "[FATAL] model missing: $MODEL"; exit 2; }
got=$(md5sum "$MODEL" | cut -d' ' -f1)
[ "$got" = "$MODEL_MD5" ] || { echo "[FATAL] model md5 $got != $MODEL_MD5"; exit 2; }
echo "model md5 OK: $got"

prog=$OUTROOT/progress.$TAG.tsv
: > "$prog"
total=0; ok=0; fail=0

while IFS="$(printf '\t')" read -r seq want; do
    [ -n "${seq:-}" ] || continue
    total=$((total + 1))
    mkdir -p "$OUTROOT/$seq"

    L=$OUTROOT/$seq/list.txt
    T=$OUTROOT/$seq/npu_det.tsv
    : > "$L"
    i=3
    end=$((want + 2))
    while [ "$i" -le "$end" ]; do
        f="$FEAT/${seq}__$(printf '%06d' $i).bmp"
        [ -f "$f" ] || { echo "[FATAL] 缺特征 $f"; exit 3; }
        echo "$f" >> "$L"
        i=$((i + 1))
    done

    if [ -f "$T" ] && [ "$(wc -l < "$T")" -eq "$want" ]; then
        printf '%s\tSKIP\t%s\n' "$seq" "$want" >> "$prog"
        echo "[$seq] SKIP (TSV 已是 $want 行)"
        ok=$((ok + 1))
        continue
    fi

    echo "[$seq] NPU start ($(wc -l < "$L") 帧)"
    # 与 int8/fp16 两臂同一套重试语义：NPU 偶发 6s job timeout -> 驱动软复位。
    rc=1; n=0; attempt=1
    while [ "$attempt" -le 3 ]; do
        "$NPU" "$MODEL" --batch "$L" --th "$TH" --topk "$TOPK" \
            --detections-out "$T" > "$OUTROOT/$seq/npu.log" 2>&1
        rc=$?
        n=$(wc -l < "$T" 2>/dev/null || echo 0)
        if [ "$rc" -eq 0 ] && [ "$n" -eq "$want" ]; then
            break
        fi
        echo "[$seq] attempt $attempt 失败 rc=$rc tsv=$n，重试"
        attempt=$((attempt + 1))
    done
    echo "[$seq] rc=$rc tsv=$n (期望 $want)"
    if [ "$rc" -ne 0 ]; then
        printf '%s\tFAIL_RC\t%s\t%s\n' "$seq" "$n" "$rc" >> "$prog"
        fail=$((fail + 1)); continue
    fi
    if [ "$n" -ne "$want" ]; then
        printf '%s\tSHORT\t%s\t%s\n' "$seq" "$n" "$want" >> "$prog"
        echo "[$seq] ⚠ TSV $n != 期望 $want"
        fail=$((fail + 1)); continue
    fi
    printf '%s\tOK\t%s\n' "$seq" "$n" >> "$prog"
    ok=$((ok + 1))
done < "$LIST"

echo "=== NPU($MODEL) 结束 $(date '+%F %T') ==="
echo "total=$total ok=$ok fail=$fail"
echo "$DONE_MARKER"
