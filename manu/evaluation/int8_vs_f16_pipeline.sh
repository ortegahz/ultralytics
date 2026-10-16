#!/bin/sh
# =============================================================================
# int8 vs fp16 收尾流水线（host 侧执行）
#
#   1. 等板卡 fp16 全量推理结束（NPU_F16_PCFEAT_ALL_DONE）
#   2. 硬校验：24 个序列 TSV 行数逐一对齐 expected.tsv，且总计 31613
#      —— 不对齐就 exit，绝不拿残缺数据去算 F1
#   3. rsync fp16 TSV -> 训练服务器
#   4. 跑 int8-vs-fp16 评测编排（转 pkl + 2 次两两对比 + 闸门 + 四臂汇总表）
#
# 与 three_arm_pipeline.sh 同构，只是换模型与输出目录。
# 失败一律响亮退出，不产出半成品结论。
# =============================================================================
set -eu

BOARD_HOST=192.168.0.64
BOARD_PORT=22
BOARD_USER=root
SRV_HOST=192.168.99.40
SRV_PORT=32222
SRV_USER=huangzhe

LOCAL_REPO=/media/manu/1TB-Volume/workspace/ultralytics
MIRROR=/home/manu/mnt/pycharm_project_10ae9e2e
NFS_TSV=/run/media/manu/SSD960-LOCAL/nfs/npu_pcfeat_f16
SRV_ROOT=/tmp/pycharm_project_10ae9e2e
SRV_TSV=$SRV_ROOT/runs/npu_pcfeat_f16_tsv
LOG=$LOCAL_REPO/runs/int8_vs_f16_pipeline.log

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== [1/4] 等待板卡 fp16 全量推理结束 ==="
i=0
while [ $i -ge 0 ]; do
    if ssh -p $BOARD_PORT -o ConnectTimeout=10 $BOARD_USER@$BOARD_HOST \
        'grep -q NPU_F16_PCFEAT_ALL_DONE /tmp/f16_full_w0.log /tmp/f16_full_w1.log 2>/dev/null' 2>/dev/null; then
        say "推理已结束（等待 $((i*30)) 秒）"
        break
    fi
    i=$((i + 1))
    if [ $i -ge 160 ]; then
        say "[FATAL] 等了 80 分钟仍未结束"
        exit 3
    fi
    if [ $((i % 4)) -eq 0 ]; then
        ssh -p $BOARD_PORT -o ConnectTimeout=10 $BOARD_USER@$BOARD_HOST \
            'for d in /mnt/manu/npu_pcfeat_f16/*/; do s=$(basename $d); n=$(wc -l < $d/npu_det.tsv 2>/dev/null || echo 0); printf "%s=%s " "$s" "$n"; done; echo' \
            2>/dev/null | tee -a "$LOG" || true
    fi
    sleep 30
done

say "=== [2/4] 硬校验 24 序列完整性 ==="
VERIFY=$(ssh -p $BOARD_PORT -o ConnectTimeout=15 $BOARD_USER@$BOARD_HOST '
    bad=0; tot=0; nseq=0
    while IFS="$(printf "\t")" read -r s w; do
        [ -n "$s" ] || continue
        nseq=$((nseq + 1))
        n=$(wc -l < /mnt/manu/npu_pcfeat_f16/$s/npu_det.tsv 2>/dev/null || echo 0)
        tot=$((tot + n))
        if [ "$n" -ne "$w" ]; then
            echo "  [MISMATCH] $s: $n != $w"
            bad=$((bad + 1))
        fi
    done < /mnt/manu/npu/out/demo/expected.tsv
    echo "  序列数=$nseq 不齐=$bad 总行数=$tot"
    echo "BAD=$bad TOT=$tot NSEQ=$nseq"
')
echo "$VERIFY" | tee -a "$LOG"
BAD=$(echo "$VERIFY" | grep -o 'BAD=[0-9]*' | cut -d= -f2)
TOT=$(echo "$VERIFY" | grep -o 'TOT=[0-9]*' | cut -d= -f2)
NSEQ=$(echo "$VERIFY" | grep -o 'NSEQ=[0-9]*' | cut -d= -f2)
if [ "$BAD" != "0" ]; then
    say "[FATAL] 有 $BAD 个序列行数不符，拒绝评估"
    exit 3
fi
if [ "$TOT" != "31613" ]; then
    say "[FATAL] 总行数 $TOT != 31613，拒绝评估"
    exit 3
fi
if [ "$NSEQ" != "24" ]; then
    say "[FATAL] 序列数 $NSEQ != 24，拒绝评估"
    exit 3
fi
say "完整性校验通过：24/24 序列，$TOT 行"

say "=== [3/4] 同步 fp16 TSV 到训练服务器 ==="
rsync -a --delete -e "ssh -p $SRV_PORT -o ConnectTimeout=10" \
    "$NFS_TSV/" "$SRV_USER@$SRV_HOST:$SRV_TSV/"
for f in manu/evaluation/run_int8_vs_f16_eval.sh \
         manu/diagnostics/npu_tsv_to_cache.py \
         manu/evaluation/eval_arm_pc_gmc_ab.py; do
    rsync -a -e "ssh -p $SRV_PORT -o ConnectTimeout=10" \
        "$MIRROR/$f" "$SRV_USER@$SRV_HOST:$SRV_ROOT/$f"
done
REMOTE_N=$(ssh -p $SRV_PORT -o ConnectTimeout=10 $SRV_USER@$SRV_HOST \
    "find $SRV_TSV -name npu_det.tsv | wc -l")
say "服务器端 TSV 文件数 = $REMOTE_N"
if [ "$REMOTE_N" != "24" ]; then
    say "[FATAL] 服务器端 TSV 数 $REMOTE_N != 24"
    exit 3
fi

say "=== [4/4] 跑 int8-vs-fp16 评测 ==="
ssh -p $SRV_PORT -o ConnectTimeout=10 $SRV_USER@$SRV_HOST \
    "sh $SRV_ROOT/manu/evaluation/run_int8_vs_f16_eval.sh" 2>&1 | tee -a "$LOG"

say "=== 完成 ==="
say "INT8_VS_F16_PIPELINE_DONE"
