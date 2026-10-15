#!/bin/sh
# =============================================================================
# ARM NPU int8 (去 P0) 全量推理 —— 特征用 PC 版本，只替换推理后端
#
# 与 hm_val_board_loop.sh 的区别（这是本实验的唯一改动点）：
#   hm_val_board_loop.sh : raw -> gmc_stream_ocl -> feat -> NPU
#   本脚本               : --------------------------> NPU
#   也就是说，本脚本完全跳过 GMC，直接把 PC 版特征喂给 NPU，
#   从而实现「特征不变、只替换推理」的受控 A/B。
#
# 特征来源（host NFS，板卡可见）：
#   host  /run/media/manu/SSD960-LOCAL/nfs/pcfeat_bmp/images/<seq>__<i>.bmp
#   board /mnt/manu/pcfeat_bmp/images/<seq>__<i>.bmp
#   该目录由 PC 侧 armcmp_pcbmp/images/val 拷贝而来（31,613 帧，BMP 无损）。
#
# 模型：trial0474_hm_nop0_int8.rknn  md5 ef302edb1b5aff1d376204bb7ad77550
#       （Trial 0474 热图头，去 P0 highway，int8，CPU 计算算子 0）
# 抽峰：--th 0.08 --topk 80   ← 与 PC 侧 cache_trial0474_inferences.py
#                              --conf-thresh 0.08 --top-k 80 严格对齐
# 坐标：输出为 640x640 letterbox 画布像素，与 PC cache 同框（已实测 40/40 吻合）
#
# 可续跑：某序列 TSV 行数已等于期望帧数时跳过。
#
# 并行：干净条件实测（bench_npu_parallel.sh，每档冷页缓存 + dmesg 稳定性监控）
#   P=1 60帧   6,050ms  9.92 f/s  完成1/1  无崩溃
#   P=2 120帧  6,110ms 19.64 f/s  完成2/2  无崩溃      <== 安全上限
#   P=3 180帧  5,990ms 20.03 f/s  完成2/3  内核崩溃1    <== 无增益
#   P=4 240帧 18,940ms  6.34 f/s  完成2/4  劣化3倍
#   P=6 360帧 19,010ms  6.31 f/s  完成2/6  劣化3倍
#   P=9 540帧 19,070ms  6.29 f/s  完成2/9  劣化3倍
# 结论：RK3588 标称 3 核，但**第 3 个并发上下文一提交就撞 RKNPU 6s job timeout**，
#       驱动 soft reset 整个 NPU，在飞上下文全部陪葬（rc=1/139）。
#       **任何 P>2 都不会更快，只会空耗 ~19s 超时窗口。生产只能 P=2。**
#       P>=4 时"吞吐"是假的——9 进程里只有 2 个真的产出了 60 帧。
# 注意 /sys/class/devfreq/fdab0000.npu/load 单进程就显示 100@1GHz，
# 那是三核合计口径，不能用来判断空核，必须实测吞吐+看 dmesg。
# 用法：N 个 worker 各跑一份**互不相交**的序列清单（本脚本参数 $1 清单 / $2 标识）。
#       生产取 N=2。N>2 见上方实测，必须先加失败重试并接受其不可用性。
# =============================================================================
set -u

FEAT=/mnt/manu/pcfeat_bmp/images
MODEL=/mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn
MODEL_MD5=ef302edb1b5aff1d376204bb7ad77550
NPU=/root/rknn_heatmap_smoke
OUTROOT=/mnt/manu/npu_pcfeat
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

    # ---- 该序列的 PC 特征清单（官方 val 命名，id 从 3 起）
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

    # ---- 续跑检查
    if [ -f "$T" ] && [ "$(wc -l < "$T")" -eq "$want" ]; then
        printf '%s\tSKIP\t%s\n' "$seq" "$want" >> "$prog"
        echo "[$seq] SKIP (TSV 已是 $want 行)"
        ok=$((ok + 1))
        continue
    fi

    echo "[$seq] NPU start ($(wc -l < "$L") 帧)"
    # NPU 偶发 6s job timeout -> 驱动软复位 -> rknn_run failed（rc=1/139）。
    # 单序列重试即可恢复；重试仍失败才记 FAIL，避免整轮带病跑完。
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

echo "=== NPU(pcfeat) 结束 $(date '+%F %T') ==="
echo "total=$total ok=$ok fail=$fail"
echo "NPU_PCFEAT_ALL_DONE"
