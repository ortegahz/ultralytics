#!/bin/sh
# =============================================================================
# ARM NPU **fp16** (去 P0) 全量推理 —— 与 run_npu_pcfeat_board.sh (int8) 逐项同构
#
# 本脚本是 run_npu_pcfeat_board.sh 的 fp16 对照臂，**唯一三处差异**：
#   1. MODEL / MODEL_MD5  -> trial0474_hm_nop0_fp16.rknn (07c893e4...)
#   2. OUTROOT            -> /mnt/manu/npu_pcfeat_f16  (不覆盖 int8 的结果)
#   3. 注释
# 其余全部照抄，保证「特征不变、抽峰不变、坐标不变，只换模型精度」：
#   特征  = PC 版 BMP   /mnt/manu/pcfeat_bmp/images/<seq>__<i>.bmp  (31,613 帧)
#   抽峰  = --th 0.08 --topk 80  (对齐 PC 侧 --conf-thresh 0.08 --top-k 80)
#   二进制= /root/rknn_heatmap_smoke (与 int8 臂同一个可执行文件)
#
# 模型来源（可复现）：与 int8 交付模型同一脚本、同一 ONNX、同一 config，
# 只加 --no-quant：
#   /media/manu/1TB-Volume/conda-envs/rknn230/bin/python \
#     manu/pipeline/npu/hm_convert_rknn.py \
#     --onnx /media/manu/1TB-Volume/rknn_work/hm/models/trial0474_hm_nop0.onnx \
#     --out   /media/manu/1TB-Volume/rknn_work/hm/models/trial0474_hm_nop0_fp16.rknn \
#     --compress-weight --no-quant
#   ONNX md5 028f23c655f421fe61a5e43575715fca / rknn-toolkit2 2.3.0
#
# 已知的、与 int8 臂不同的硬件事实（构建日志 build_nop0_fp16.log）：
#   - CPU 计算算子 0（与 int8 一致）
#   - **1 条构建回退**: 'Conv:/model/b5/conv/Conv' 3Core -> single core
#   - **NPU 内部内存 105,600 KB**，而 int8 只有 20,400 KB (5.18x)
#     => P=2 并行是否可行必须实测，不能假设
#   - 实测单进程: inputs_set 21.0 ms + rknn_run 102.6 ms
#
# 用法：N 个 worker 各跑一份**互不相交**的序列清单（本脚本参数 $1 清单 / $2 标识）
# =============================================================================
set -u

FEAT=/mnt/manu/pcfeat_bmp/images
MODEL=/mnt/manu/npu/data/trial0474_hm_nop0_fp16.rknn
MODEL_MD5=07c893e4d2ca3af9a47a4bf88335fe33
NPU=/root/rknn_heatmap_smoke
OUTROOT=/mnt/manu/npu_pcfeat_f16
TH=0.08
TOPK=80
EXPECT=/mnt/manu/npu/out/demo/expected.tsv
LIST=${1:-$EXPECT}          # 本 worker 负责的序列清单（默认全量）
TAG=${2:-w0}                 # worker 标识，用于区分进度文件

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

    echo "[$seq] NPU-fp16 start ($(wc -l < "$L") 帧)"
    # 与 int8 臂同一套重试语义：NPU 偶发 6s job timeout -> 驱动软复位。
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

echo "=== NPU-fp16(pcfeat) 结束 $(date '+%F %T') ==="
echo "total=$total ok=$ok fail=$fail"
echo "NPU_F16_PCFEAT_ALL_DONE"
