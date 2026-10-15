#!/bin/bash
# =============================================================================
# ARM GMC 精度评估 —— 全量可复现编排脚本（铁律九：实验必须留命令与脚本入口）
#
# 对应结论：memory.md §12 / memory_compact.md §14
# 实验记录：manu/docs/experiments/2026-10-15_arm_gmc_precision_eval.md
#
# 唯一变量 = 「GMC 特征由谁生成」：
#   PC-BMP 臂 = 冻结构建器 build_full_median_dataset.py（md5 7729e77299f8933faffa73577e8e9400）
#   ARM-BMP 臂 = 板卡 evm3588 上实机运行的 gmc_stream_ocl（md5 aa99da811356694d66a0bce6de108b92）
# 两侧同为 BMP 无损容器、同一批 raw、同一份标签、同一权重、同一抽峰口径。
#
# 运行环境：
#   本机(host)   : NFS 挂载 /run/media/manu/SSD960-LOCAL/nfs
#   板卡(board)  : ssh -p 22 root@192.168.0.64
#   训练服务器    : ssh -p 32222 huangzhe@192.168.99.40
#   解释器        : /home/huangzhe/anaconda3/envs/uav/bin/python  (py3.10.20 / cv2 4.10.0 / np 1.26.4)
#   GPU          : RTX 4090 D
#
# 用法： bash manu/evaluation/run_arm_pc_gmc_full_ab.sh [stage]
#        stage ∈ {all, board, xfer, assemble, infer, ab, pix}   默认 all
# =============================================================================
set -u

STAGE="${1:-all}"

SS="-p 32222"
SRV="huangzhe@192.168.99.40"
PY="/home/huangzhe/anaconda3/envs/uav/bin/python"
PROJ="/tmp/pycharm_project_10ae9e2e"

NFS="/run/media/manu/SSD960-LOCAL/nfs/armfeat_s2"          # 板卡 GMC 输出（host 视角）
RAWROOT="/mnt/data/siping/datasets/manu/anti_uav_bmp_all/val"  # 无损 BMP 原帧（两侧共用输入）
FROZEN="/mnt/data/siping/datasets/manu/uav_gmc_median"     # 冻结参照（帧名 + 标签来源）
REF_BMP="/mnt/data/siping/datasets/manu/ref_bmpval"        # 强制 .bmp 扩展的无损 ref
STAGE_DIR="/mnt/data/siping/datasets/manu/armfeat_stage"   # 板卡产物落地区
PC_DS="/mnt/data/siping/datasets/manu/armcmp_pcbmp"        # PC 臂数据集
ARM_DS="/mnt/data/siping/datasets/manu/arm_gmc_full"       # ARM 臂数据集
WEIGHTS="runs/optuna_p0_nas/trial_0474/weights/best.pt"   # md5 5b07f255735f9fbadf3e0dfe95a7708f

BOARD_TOTAL=31661   # 板卡产出 = raw 总帧数（每序列含前 2 帧）
FINAL=31613         # 丢弃 k<2 的 48 帧后 = 官方 val 帧数

log(){ echo "[$(date +%H:%M:%S)] $*"; }

# -----------------------------------------------------------------------------
stage_board() {
  log "=== stage: board (在板卡上生成 anchor_step=2 特征) ==="
  # 板卡侧脚本已固化在仓库：manu/pipeline/npu/run_step2_board.sh
  # （可执行体与板卡 /root/run_step2.sh 逐字节一致，md5 4ebde86eff698475b170b4c9c539e2da）
  log "--- 部署并启动板卡脚本 ---"
  scp -P 22 manu/pipeline/npu/run_step2_board.sh root@192.168.0.64:/root/run_step2.sh
  ssh -p 22 root@192.168.0.64 'chmod +x /root/run_step2.sh && \
      nohup /root/run_step2.sh > /mnt/manu/armfeat_s2.log 2>&1 & echo "[board pid] $!"'

  log "期望清单：/mnt/manu/npu/out/demo/expected.tsv（24 行 <seq>\\t<官方帧数>）"
  log "等待产出 $BOARD_TOTAL 帧（= raw 总帧数，每序列含前 2 帧；组装阶段再丢 48 帧）..."
  while :; do
    N=$(ssh -p 22 root@192.168.0.64 "find /mnt/manu/armfeat_s2 -name '*.bmp' 2>/dev/null | wc -l" | tr -d ' ')
    # 注意：不要用 `ls .../*/feat/*.bmp` —— 3 万个路径展开进 argv 会撞 ARG_MAX 静默返回 0
    log "  board=$N/$BOARD_TOTAL"
    [ "$N" = "$BOARD_TOTAL" ] && break
    [ "$N" -gt "$BOARD_TOTAL" ] 2>/dev/null && { log "OVERPRODUCED"; return 3; }
    sleep 60
  done
}

# -----------------------------------------------------------------------------
stage_xfer() {
  log "=== stage: xfer (host -> 训练服务器，6 流并行) ==="
  log "板卡 root 写的文件在 host 上属 root:root，需先 chown"
  ssh -p 22 root@192.168.0.64 'chown -R 1000:1000 /mnt/manu/armfeat_s2 && chmod -R a+rX /mnt/manu/armfeat_s2'
  (cd "$NFS" && find . -name '*.bmp' | sort) > /tmp/arm_bmp_list.txt
  log "list = $(wc -l < /tmp/arm_bmp_list.txt) (expect $BOARD_TOTAL)"
  ssh $SS $SRV "rm -rf $STAGE_DIR; mkdir -p $STAGE_DIR"
  rm -f /tmp/armpart_*
  split -n l/6 -d /tmp/arm_bmp_list.txt /tmp/armpart_
  T0=$(date +%s)
  # 6 流并行：单流 13.8 MB/s，6 流实测 78~81 MB/s（5.7x）
  for p in /tmp/armpart_*; do
    rsync -az -e "ssh $SS" --files-from="$p" --relative "$NFS/" "$SRV:$STAGE_DIR/" >/dev/null 2>>/tmp/armrsync.err &
  done
  wait
  log "transfer done in $(( $(date +%s) - T0 ))s"
  R=$(ssh $SS $SRV "find $STAGE_DIR -name '*.bmp' | wc -l" | tr -d ' ')
  log "staged = $R (expect $BOARD_TOTAL)"
  [ "$R" != "$BOARD_TOTAL" ] && { tail -5 /tmp/armrsync.err; return 5; }
}

# -----------------------------------------------------------------------------
stage_assemble() {
  log "=== stage: assemble (PC 臂无损重建 + ARM 臂组装) ==="

  # --- PC 臂：构造强制 .bmp 扩展的无损 ref（不动构建器，守第〇之二节）---
  # 构建器只读 ref 的文件名与标签，从不读 ref 图像像素（build_full_median_dataset.py:268-272）
  # 故全改 .bmp + 软链回原图即可让 cv2.imwrite 输出无损 BMP。
  log "--- 构造 ref_bmpval ---"
  ssh $SS $SRV "bash -s" <<REMOTE
set -eu
REF=$FROZEN
OUT=$REF_BMP
rm -rf "\$OUT"; mkdir -p "\$OUT/images/val" "\$OUT/labels/val"
for f in "$REF"/images/val/*; do
  b=\$(basename "\$f"); stem="\${b%.*}"
  ln -sf "\$f" "\$OUT/images/val/\$stem.bmp"
done
for f in "$REF"/labels/val/*.txt; do
  ln -f "\$f" "\$OUT/labels/val/\$(basename "\$f")" 2>/dev/null || cp "\$f" "\$OUT/labels/val/\$(basename "\$f")"
done
printf 'path: %s\ntrain: images/val\nval: images/val\n\nnames:\n  0: uav\n' "\$OUT" > "\$OUT/data.yaml"
echo "images=\$(find \$OUT/images/val -type l | wc -l)  labels=\$(ls -1 \$OUT/labels/val | wc -l)"
REMOTE

  # --- PC 臂重建 ---
  log "--- PC 臂无损重建（$FINAL 帧）---"
  ssh $SS $SRV "cd $PROJ && md5sum manu/data/build_full_median_dataset.py && \
    $PY manu/data/build_full_median_dataset.py \
      --ref-dataset $REF_BMP \
      --raw-root $RAWROOT \
      --output $PC_DS \
      --splits val --workers 16"

  # --- ARM 臂组装：<seq>_<k>.bmp -> <seq>__<k+1>.bmp，丢弃 k<2 ---
  log "--- ARM 臂组装（OFFSET=1，丢弃 k<2 的 48 帧）---"
  ssh $SS $SRV "cd $PROJ && $PY manu/data/assemble_arm_gmc_dataset.py \
      --stage $STAGE_DIR \
      --out $ARM_DS \
      --labels-from $REF_BMP \
      --expect-total $FINAL --verify-md5 120"
}

# -----------------------------------------------------------------------------
stage_infer() {
  log "=== stage: infer (两臂同权重同口径) ==="
  for pair in "pc_bmp_full_cache.pkl:$PC_DS" "arm_gmc_full_cache.pkl:$ARM_DS"; do
    OUT="${pair%%:*}"; DS="${pair##*:}"
    log "--- $DS -> $OUT ---"
    ssh $SS $SRV "cd $PROJ && $PY manu/inference/cache_trial0474_inferences.py \
        --weights $WEIGHTS \
        --data $DS/data.yaml \
        --output runs/gmc_eval/$OUT \
        --imgsz 640 --stride 2 --batch 32 --device 0 \
        --conf-thresh 0.08 --top-k 80"
  done
}

# -----------------------------------------------------------------------------
stage_ab() {
  log "=== stage: ab (容器对称 A/B 评估) ==="
  ssh $SS $SRV "cd $PROJ && $PY manu/evaluation/eval_arm_pc_gmc_ab.py \
      --arm-a runs/gmc_eval/pc_bmp_full_cache.pkl --label-a PC-BMP \
      --arm-b runs/gmc_eval/arm_gmc_full_cache.pkl --label-b ARM-BMP \
      --tol 8.0 \
      --out-json /mnt/data/siping/datasets/manu/gmc_arm_vs_pc_ab.json"
}

stage_pix() {
  log "=== stage: pix (全量逐像素) ==="
  ssh $SS $SRV "cd $PROJ && $PY manu/diagnostics/cmp_arm_pc_features.py \
      --arm $ARM_DS/images/val --pc $PC_DS/images/val \
      --samples 0 \
      --out-json /mnt/data/siping/datasets/manu/gmc_arm_vs_pc_pixdiff.json"
}

# -----------------------------------------------------------------------------
case "$STAGE" in
  all)      stage_assemble; stage_infer; stage_ab; stage_pix ;;
  board)    stage_board ;;
  xfer)     stage_xfer ;;
  assemble) stage_assemble ;;
  infer)    stage_infer ;;
  ab)       stage_ab ;;
  pix)      stage_pix ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
log "stage '$STAGE' done"
