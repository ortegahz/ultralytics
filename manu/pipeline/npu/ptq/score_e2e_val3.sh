#!/usr/bin/env bash
# ===========================================================================
# score_e2e_val3.sh -- 把板端 e2e 全量产物送到服务器并出对照表。
#
# 前提：板端已跑完
#   /root/rknn_mosaic_e2e --raw-root /mnt/manu/lq_ir_bmp \
#     --sequence VIDEO00007_19700101_001324 \
#     --model /mnt/manu/ptq/models/p0_off_A_none.rknn \
#     --out /mnt/manu/mosaic_out/e2e_val3_final --kernel-dir /root/kernels \
#     --anchor-step 10 --npu-workers 2 --cpu-affinity 4-7 \
#     --timing /mnt/manu/mosaic_out/e2e_val3_final/timing.txt
#
# 本脚本只做「搬运 + 打分」，不改任何数值。
# ===========================================================================
set -euo pipefail

NFS=/run/media/manu/SSD960-LOCAL/nfs/mosaic_out/e2e_val3_final
SERVER=huangzhe@192.168.99.40
PORT=32222
REPO=/tmp/pycharm_project_10ae9e2e
STAGE=/tmp/e2e_val3_final
DST="$REPO/manu/pipeline/npu/ptq/board_tsv_e2e_final"
SEQ=VIDEO00007_19700101_001324

echo "== 1. 本地取 NFS 产物 =="
mkdir -p "$STAGE"
cp "$NFS/${SEQ}_mosaic.tsv" "$NFS/timing.txt" "$STAGE/"
ls -la "$STAGE"
md5sum "$STAGE"/*

echo "== 2. 送到服务器 =="
ssh -p $PORT "$SERVER" "mkdir -p '$DST'"
scp -P $PORT "$STAGE/${SEQ}_mosaic.tsv" "$STAGE/timing.txt" "$SERVER:$DST/"
ssh -p $PORT "$SERVER" "ls -la '$DST'; md5sum '$DST'/*"

echo "== 3. 出对照表 =="
ssh -p $PORT "$SERVER" "source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav && cd $REPO && \
  python manu/pipeline/npu/ptq/38_mosaic_final_report.py \
    --group track3_blind=$SEQ \
    --board-tsv-dir manu/pipeline/npu/ptq/board_tsv \
    --e2e-tsv-dir manu/pipeline/npu/ptq/board_tsv_e2e_final \
    --pc-area-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_area_all \
    --pc-linear-tsv-dir /mnt/data/siping/ptq_work/mosaic_pc_linear_all \
    --board-timing-dir manu/pipeline/npu/ptq/board_timing_dir \
    --e2e-timing manu/pipeline/npu/ptq/board_tsv_e2e_final/timing.txt \
    --out /mnt/data/siping/ptq_work/final_report_e2e_final"

echo "== 4. 取回报告 =="
mkdir -p /tmp/e2e_report
scp -P $PORT "$SERVER:/mnt/data/siping/ptq_work/final_report_e2e_final/mosaic_vs_f32_report.md" /tmp/e2e_report/ || true
scp -P $PORT "$SERVER:/mnt/data/siping/ptq_work/final_report_e2e_final/mosaic_vs_f32_report.json" /tmp/e2e_report/ || true
echo "[ok] /tmp/e2e_report/"
