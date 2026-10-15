#!/bin/sh
# =============================================================================
# 端到端收尾流水线（host 侧执行）
#
#   1. 等板卡单进程全量推理结束（NPU_PCFEAT_ALL_DONE）
#   2. 硬校验：24 个序列 TSV 行数逐一对齐 expected.tsv，且总计 31613
#      —— 不对齐就 exit，绝不拿残缺数据去算 F1
#   3. rsync TSV -> 训练服务器
#   4. 跑三臂评估编排（转换 + 3 次两两对比 + 闸门 + 汇总表）
#
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
NFS_TSV=/run/media/manu/SSD960-LOCAL/nfs/npu_pcfeat
SRV_ROOT=/tmp/pycharm_project_10ae9e2e
SRV_TSV=$SRV_ROOT/runs/npu_pcfeat_tsv
LOG=$LOCAL_REPO/runs/three_arm_pipeline.log

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "=== [1/4] 等待板卡全量推理结束 ==="
i=0
while [ $i -lt 120 ]; do
    if ssh -p $BOARD_PORT -o ConnectTimeout=10 $BOARD_USER@$BOARD_HOST \
        'grep -q NPU_PCFEAT_ALL_DONE /mnt/manu/npu_pcfeat.log' 2>/dev/null; then
        say "推理已结束（等待 $((i*30)) 秒）"
        break
    fi
    i=$((i + 1))
    if [ $((i % 4)) -eq 0 ]; then
        ssh -p $BOARD_PORT -o ConnectTimeout=10 $BOARD_USER@$BOARD_HOST \
            'echo -n "  进度: "; cat /mnt/manu/npu_pcfeat.log | grep -c "rc=0\|SKIP"' \
            2>/dev/null | tee -a "$LOG" || true
    fi
    sleep 30
done
if [ $i -ge 120 ]; then
    say "[FATAL] 等了 60 分钟仍未结束"
    exit 3
fi

say "=== [2/4] 硬校验 24 序列完整性 ==="
VERIFY=$(ssh -p $BOARD_PORT -o ConnectTimeout=15 $BOARD_USER@$BOARD_HOST '
    bad=0; tot=0
    while IFS="$(printf "\t")" read -r s w; do
        [ -n "$s" ] || continue
        n=$(wc -l < /mnt/manu/npu_pcfeat/$s/npu_det.tsv 2>/dev/null || echo 0)
        tot=$((tot + n))
        if [ "$n" -ne "$w" ]; then
            echo "  [MISMATCH] $s: $n != $w"
            bad=$((bad + 1))
        fi
    done < /mnt/manu/npu/out/demo/expected.tsv
    echo "  序列数=24 不齐=$bad 总行数=$tot"
    echo "BAD=$bad TOT=$tot"
')
echo "$VERIFY" | tee -a "$LOG"
BAD=$(echo "$VERIFY" | grep -o 'BAD=[0-9]*' | cut -d= -f2)
TOT=$(echo "$VERIFY" | grep -o 'TOT=[0-9]*' | cut -d= -f2)
if [ "$BAD" != "0" ]; then
    say "[FATAL] 有 $BAD 个序列行数不符，拒绝评估"
    exit 3
fi
if [ "$TOT" != "31613" ]; then
    say "[FATAL] 总行数 $TOT != 31613，拒绝评估"
    exit 3
fi
say "完整性校验通过：24/24 序列，$TOT 行"

say "=== [3/4] 同步 TSV 到训练服务器 ==="
rsync -a --delete -e "ssh -p $SRV_PORT -o ConnectTimeout=10" \
    "$NFS_TSV/" "$SRV_USER@$SRV_HOST:$SRV_TSV/"
rsync -a -e "ssh -p $SRV_PORT -o ConnectTimeout=10" \
    "$MIRROR/manu/evaluation/run_three_arm_eval.sh" \
    "$SRV_USER@$SRV_HOST:$SRV_ROOT/manu/evaluation/run_three_arm_eval.sh"
rsync -a -e "ssh -p $SRV_PORT -o ConnectTimeout=10" \
    "$MIRROR/manu/diagnostics/npu_tsv_to_cache.py" \
    "$SRV_USER@$SRV_HOST:$SRV_ROOT/manu/diagnostics/npu_tsv_to_cache.py"
rsync -a -e "ssh -p $SRV_PORT -o ConnectTimeout=10" \
    "$MIRROR/manu/evaluation/eval_arm_pc_gmc_ab.py" \
    "$SRV_USER@$SRV_HOST:$SRV_ROOT/manu/evaluation/eval_arm_pc_gmc_ab.py"
REMOTE_N=$(ssh -p $SRV_PORT -o ConnectTimeout=10 $SRV_USER@$SRV_HOST \
    "find $SRV_TSV -name npu_det.tsv | wc -l")
say "服务器端 TSV 文件数 = $REMOTE_N"
if [ "$REMOTE_N" != "24" ]; then
    say "[FATAL] 服务器端 TSV 数 $REMOTE_N != 24"
    exit 3
fi

say "=== [4/4] 跑三臂评估 ==="
ssh -p $SRV_PORT -o ConnectTimeout=10 $SRV_USER@$SRV_HOST \
    "sh $SRV_ROOT/manu/evaluation/run_three_arm_eval.sh" 2>&1 | tee -a "$LOG"

say "=== 完成 ==="
say "PIPELINE_DONE"
