#!/bin/bash
# =============================================================================
# 串行 vs 并行 e2e 精度一致性消融 —— 评测编排（在 PC 工作站执行）
#
# 唯一变量：执行结构
#   臂 S  e2e_demo         单进程串行（FP-GMC + letterbox + rknn_run + 抽峰）
#   臂 P  e2e_par2_demo    4 级流水线 × 2 worker（绑核 4/5/6/7，NPU core0+core1）
# 其余全部冻结：同一 fp16 模型 trial0474_hm_nop0_fp16.rknn、同一板卡 GMC、
# 同一批原始 BMP（/mnt/manu/aval）、th=0.08、topk=100、stride=2、
# 同一 GT（pc_bmp_nop0_cache.pkl）、同一评测器（eval_arm_pc_gmc_ab.py）、
# 同一 15 档门限 0.05..0.75、同一匹配容差 8.0px。
#
# 硬闸门（任一不过立即非零退出）
#   G1 两臂 realign 后帧数各 == 31613，每序列 == expected.tsv 的 want
#   G2 两臂 realign 后 payload md5 相同 -> 逐帧逐检测完全一致
#   G3 npu_tsv_to_cache 配对成功 == 31613
#   G4 eval_arm_pc_gmc_ab 严格同集配对 + GT 逐项一致
# =============================================================================
set -euo pipefail

ROOT=${ROOT:-/media/manu/1TB-Volume/workspace/ultralytics}
NFS=${NFS:-/run/media/manu/SSD960-LOCAL/nfs/e2e24}
WORK=${WORK:-/media/manu/1TB-Volume/workspace/ultralytics/runs/e2e24_ablation}
TS_S=${TS_S:-$NFS/full_serial}
TS_P=${TS_P:-$NFS/full_par}

# 训练服务器
TS_HOST=${TS_HOST:-192.168.99.40}
TS_PORT=${TS_PORT:-32222}
TS_USER=${TS_USER:-huangzhe}
TSH="ssh -q -p $TS_PORT -o BatchMode=yes -o ConnectTimeout=30 $TS_USER@$TS_HOST"
TSR="rsync -q -a -e ssh -p $TS_PORT -o BatchMode=yes -o ConnectTimeout=30"

PY=$TSH "$TS_USER@$TS_HOST"' /home/huangzhe/anaconda3/envs/uav/bin/python'
TROOT=/tmp/pycharm_project_10ae9e2e
GT=$TROOT/runs/gmc_eval/pc_bmp_nop0_cache.pkl

mkdir -p "$WORK"
echo "### 阶段 0：期望清单"
# expected.tsv 只在板卡上；若本地没有就拉一份
if [ ! -f "$WORK/expected.tsv" ]; then
  scp -q -P 22 -o BatchMode=yes -o ConnectTimeout=30 \
      root@192.168.0.64:/mnt/manu/npu/out/demo/expected.tsv "$WORK/expected.tsv"
fi
EXP=$WORK/expected.tsv
awk -F'\t' '{s+=$2} END{print "[EXP ] "NR" 序列, GT 帧合计 "s}' "$EXP"

echo
echo "### 阶段 1：两臂 TSV -> GT stem 口径（realign，含逐序列硬校验）"
for ARM in serial parallel; do
  SRC=$TS_S; [ "$ARM" = parallel ] && SRC=$TS_P
  rm -rf "$WORK/ra_$ARM"
  python3 "$ROOT/manu/diagnostics/npu_tsv_realign.py" \
      --tsv-root "$SRC" --arm "$ARM" --expected "$EXP" \
      --out "$WORK/ra_$ARM" --expect-total 31613 | tee "$WORK/realign_$ARM.log"
done

echo
echo "### G2：两臂 payload 逐帧一致性"
HS=$(grep -o '[0-9a-f]\{32\}' "$WORK/realign_serial.log"   | tail -1)
HP=$(grep -o '[0-9a-f]\{32\}' "$WORK/realign_parallel.log" | tail -1)
echo "  serial   payload md5 = $HS"
echo "  parallel payload md5 = $HP"
if [ "$HS" != "$HP" ]; then
  echo "[FAIL] G2 不通过：两臂 payload 哈希不同，存在逐帧差异"; exit 2
fi
echo "  [OK] G2 通过：两臂 31613 帧检测 payload 完全一致"

# 逐序列 md5 也全等才算数
diff <(grep '^\[OK  \]' "$WORK/realign_serial.log"   | awk '{print $2, $NF}') \
     <(grep '^\[OK  \]' "$WORK/realign_parallel.log" | awk '{print $2, $NF}') \
  && echo "  [OK] 24 个序列的逐序列 md5 也全部相同"

echo
echo "### 阶段 2：推训练服务器并转 cache pkl"
$TSR "$WORK/ra_serial/"   "$TS_USER@$TS_HOST:/tmp/e2e24_ra_serial/"
$TSR "$WORK/ra_parallel/" "$TS_USER@$TS_HOST:/tmp/e2e24_ra_par/"
for ARM in serial parallel; do
  RT=/tmp/e2e24_ra_serial; [ "$ARM" = parallel ] && RT=/tmp/e2e24_ra_par
  $TSH "$TS_USER@$TS_HOST" "$PY $TROOT/manu/diagnostics/npu_tsv_to_cache.py \
      --tsv-root $RT --gt-cache $GT --out /tmp/e2e24_$ARM.pkl --expect-total 31613"
done

echo
echo "### 阶段 3：F1 / P / R 对照（15 档扫描取峰值，tol=8.0px）"
$TSH "$TS_USER@$TS_HOST" "$PY $TROOT/manu/evaluation/eval_arm_pc_gmc_ab.py \
    --arm-a /tmp/e2e24_serial.pkl   --label-a 'serial_e2e_demo' \
    --arm-b /tmp/e2e24_parallel.pkl --label-b 'par2_e2e_demo' \
    --tol 8.0 --out-json /tmp/cmp_e2e24_serial_vs_par.json"

echo
echo "### 阶段 4：归档产物到训练服务器"
$TSH "$TS_USER@$TS_HOST" "mkdir -p $TROOT/runs/gmc_eval/e2e24_ablation && \
    cp /tmp/e2e24_serial.pkl /tmp/e2e24_parallel.pkl /tmp/cmp_e2e24_serial_vs_par.json \
       $TROOT/runs/gmc_eval/e2e24_ablation/ && \
    ls -la $TROOT/runs/gmc_eval/e2e24_ablation/"

echo
echo "### DONE"
