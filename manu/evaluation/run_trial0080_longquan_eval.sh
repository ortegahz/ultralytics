#!/usr/bin/env bash
# 龙泉山业务域全量评估 —— trial_0080 一次性流水线（ONNX 后端）
#
# 2026-11-19 更新：原 `runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt` 连同整棵
# `runs/optuna_mix_ratio` 树已被删除，原脚本无法复现。幸存替身是同一份权重导出的 ONNX
# （`hm_plain.onnx`），它与 torch 参考在 3297 帧上逐阈值 tp/fp/fn 一致（见
# manu/pipeline/npu/ptq/22q_onnx_vs_pt_ref.py + ptq_work/qat/trial_0080/report/onnx_vs_pt_ref_verdict.json）。
# 缓存用 ONNX Runtime 重建后，v3 工作点的 24 序逐序列指标与 2026-10-22 官方记录**逐项一致**。
#
# 另外：交付嵌入式的 `p0_off_A_none.rknn` 就是这份权重的 fp16 版本
# （6,326,305 B, md5 1a3faf2f9459cad8a389ffa81810234c），源图即 `trial_0080_hm_nop0.onnx`。
#
# 做什么（一条命令跑完，分 4 段）：
#   1) 等训练让出 GPU（optuna 搜索 20h 截止后自动收尾）
#   2) 用 trial_0080 的 ONNX 导出在 FPV 语义特征上重建**全量 24 序列**多尺度热图缓存
#      —— 特征目录 longquanshan_ir_gmc_median/images/train 就是冻结基线 ms_cache_nop0 的同源特征，
#         已算好的 GMC+median 特征直接读盘复用，不重算 GMC/中值（唯一要新算的是 heatmap 推理）
#   3) 单帧臂（--no-pipeline，th_base=0.22 / dedup_radius=768，对齐冻结单帧基线 F1 87.070%）
#   4) 带 track 臂（系统管线：双向平滑+插补+刚性剪枝+连续帧确认），工作点 = **v3**
#      （th_base 0.48 / th_ground 0.26 / sky_ratio 1.0 / confirm_consec 7，官方 F1 94.7469%、FAR 0.02082）
#
# 为什么必须重建缓存而不能直接用 trial_0080/gate/epoch3/ms_cache：
#   那 8 条缓存的特征源是 longquanshan_ft_gmc_median_official（official 逐 lag 直接拟合语义），
#   与冻结基线的 FPV 仿射连乘语义**逐像素差 26~70/255**，混用会让"全量 F1"变成两个分布的加权平均，
#   无法与 87.070% / 90.605% 做差值归因（详见 runs/logs/lq_feature_source.json 与本脚本头注释）。
#
# 关键路径（全部相对于 /tmp/pycharm_project_10ae9e2e）：
#   模型   hm_plain.onnx (= 交付 rknn p0_off_A_none.rknn 的源图 trial_0080_hm_nop0.onnx)
#   特征   /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train   (165,223 帧 / 24 序列)
#   GT     /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010
#   缓存   runs/trial0080_onnx/lq_eval/ms_cache      (实际跑过的产物在 /mnt/data/siping/runs/trial0080_onnx/ms_cache)
#   报告   runs/trial0080_onnx/lq_eval/*.json        (实际跑过的产物在 /mnt/data/siping/runs/trial0080_onnx/eval/)
#   预测   runs/trial0080_onnx/lq_eval/preds_v3       (OSD 渲染输入)
#
# 用法：
#   bash manu/evaluation/run_trial0080_longquan_eval.sh
#   WAIT_GPU=0 bash manu/evaluation/run_trial0080_longquan_eval.sh     # 不等 GPU，直接抢（不推荐）
#   GPU_LIST=2,3,1 bash ...                                           # 指定物理卡（错峰 30s，最多 3 路并发）
#   FORCE_REBUILD=1 bash ...                                          # 强制重算缓存
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=.

WAIT_GPU="${WAIT_GPU:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# The torch checkpoint is gone; the ONNX export of the SAME weights is the surviving stand-in.
ONNX="${HM_ONNX:-/mnt/data/siping/ptq_work/qdq_probe/onnx/hm_plain.onnx}"
TRIAL_DIR="runs/trial0080_onnx"
FEATURES="/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median/images/train"
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
OUT_DIR="$TRIAL_DIR/lq_eval"
CACHE_ROOT="$OUT_DIR/ms_cache"
SINGLE_PARAMS="manu/evaluation/system_sota_longquan_singleframe_params.json"
# v3 is the current baseline (best F1 AND lowest FAR of the three frozen trial_0080 points).
TRACK_PARAMS="manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json"

echo "=============================================================="
echo " 0. 环境与前置检查"
echo "=============================================================="
python -c "import torch;print(' torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())"
for p in "$ONNX" "$FEATURES" "$GT" "$SINGLE_PARAMS" "$TRACK_PARAMS"; do
  [ -e "$p" ] || { echo "[FAIL] 缺失: $p"; exit 2; }
done
echo " 模型   : $ONNX  (trial_0080 epoch3 的 ONNX 导出)"
echo " 特征   : $FEATURES"
echo " GT     : $GT"
echo " 产物   : $OUT_DIR"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------- 1. 等 GPU
echo
echo "=============================================================="
echo " 1. 等待 GPU 空闲（训练搜索让出卡）"
echo "=============================================================="
if [ "$WAIT_GPU" = "1" ]; then
  while true; do
    # 只认**真正的 python 搜索进程**：`pgrep -f` 会匹配到命令行里含该字样的监控 shell 自身
    # （踩过的坑：等待条件永远不成立，GPU 已释放却仍在空转）。
    # 因此这里用 `ps | grep python.*optuna_...`，再叠加"GPU 上无计算进程"两个条件。
    # 计数用 `wc -l` 而非 `grep -c`：`grep -c` 无匹配时 exit 1，会让 `|| echo 0` 再补一个 0，
    # 变量变成 "0\n0"，字符串比较 `= "0"` 永远不成立（第二个坑）。
    NPROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    NSEARCH=$(ps -eo pid,cmd --no-headers 2>/dev/null | grep -E "python.*optuna_mix_ratio_search" | grep -v grep | wc -l | tr -d ' ')
    echo "  [wait] gpu_procs=$NPROC search_procs=$NSEARCH  $(date +%H:%M:%S)"
    if [ "$NPROC" = "0" ] && [ "$NSEARCH" = "0" ]; then break; fi
    sleep 60
  done
  echo "  GPU 已全部释放，等 20s 让显存回收"
  sleep 20
else
  echo "  [warn] WAIT_GPU=0 —— 将在训练进程仍占卡时启动推理，显存不足会自动降 batch"
fi

# ---------------------------------------------------------------- 2. 建缓存
echo
echo "=============================================================="
echo " 2. 重建全量 24 序列多尺度热图缓存（trial_0080/epoch3，FPV 特征源）"
echo "=============================================================="
NGPU=$(python -c "import torch;print(max(1,torch.cuda.device_count()))")
# 资源铁律：通道并发 ≤ 3、必须错峰（间隔 ≥ 30s）、禁止一次 `&` 起多路后台。
# GPU_LIST 可指定**物理卡号**（逗号分隔）；默认取 0..NGPU-1 的前 3 张。
GPU_LIST="${GPU_LIST:-$(seq -s, 0 $((NGPU-1)) | cut -d, -f1-3)}"
NSHARD=$(awk -F, '{print NF}' <<< "$GPU_LIST")
echo "  GPU 数: $NGPU | 使用物理卡: $GPU_LIST | shard 数: $NSHARD"
EXPECT_SEQ=24
if [ "$FORCE_REBUILD" = "1" ]; then
  rm -rf "$CACHE_ROOT"
fi
mkdir -p "$CACHE_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PIDS=()
idx=0
for phys in ${GPU_LIST//,/ }; do
  if [ "$idx" -gt 0 ]; then
    echo "  [stagger] 错峰 30s 后再起下一个 shard（铁律：间隔 ≥ 30s）"
    sleep 30
  fi
  CUDA_VISIBLE_DEVICES="$phys" python manu/data/multiscale_heatmap_cache.py \
    --features-root "$FEATURES" \
    --cache-root "$CACHE_ROOT" \
    --shard "$idx/$NSHARD" \
    --scales 160,80 \
    --region-threshold 0.06 --main-threshold 0.22 \
    --fusion-dilate 32 --min-area 4 \
    --device 0 --chunk-size 64 --batch-size 8 --cpu-threads 1 \
    --hm-onnx "$ONNX" --ablate-p0 \
    --overwrite > "$OUT_DIR/ms_cache_shard$idx.log" 2>&1 &
  PIDS+=($!)
  echo "  [start] shard $idx/$NSHARD -> 物理卡 GPU$phys (pid ${PIDS[-1]}) $(date +%H:%M:%S)"
  idx=$((idx+1))
done
for p in "${PIDS[@]}"; do wait "$p"; done
NSEL=$(ls "$CACHE_ROOT"/*.pkl 2>/dev/null | wc -l)
echo "  缓存 pkl 数: $NSEL / $EXPECT_SEQ"
if [ "$NSEL" -ne "$EXPECT_SEQ" ]; then
  echo "[FAIL] 缓存序列数不足，终止（不产出误导性的部分结果）"
  grep -h "Traceback\|Error" "$OUT_DIR"/ms_cache_shard*.log | head -20
  exit 3
fi

# ---------------------------------------------------------------- 3. 单帧臂
echo
echo "=============================================================="
echo " 3. 单帧臂（--no-pipeline，th_base=0.22 / dedup_radius=768）"
echo "=============================================================="
python manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE_ROOT" --labels-root "$GT" --params-json "$SINGLE_PARAMS" \
  --all-seqs --no-pipeline \
  --out-json "$OUT_DIR/single_frame_all.json" \
  --out-preds-dir "$OUT_DIR/preds_single" \
  2>&1 | tee "$OUT_DIR/single_frame_all.log" | grep -E "^\[SEQ\]|^\[OVERALL\]|^\[GATE\]"

# 3b. 单帧臂的正确性闸门：期望值外置成 JSON（trial_0080 单帧臂实测基线）。
#     注意：内置 GATE_EXPECTED 绑定的是冻结 trial_0474 缓存，对本缓存必然 FAIL —— 只能显式给基线。
echo
echo "--- 3b. 单帧臂 self-check（显式基线）---"
python manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE_ROOT" --labels-root "$GT" --all-seqs \
  --self-check --self-check-baseline manu/evaluation/system_sota_longquan_trial0080_singleframe_gate_baseline.json \
  2>&1 | tee "$OUT_DIR/self_check.log" | grep -E "^\[GATE\]|expected|actual" || {
    echo "[FAIL] 单帧臂 self-check 未通过"; exit 4;
  }

# ---------------------------------------------------------------- 4. 带 track 臂
echo
echo "=============================================================="
echo " 4. 带 track 臂（系统管线：双向平滑+插补+刚性剪枝+连续帧确认）"
echo "=============================================================="
python manu/evaluation/system_sota_longquan.py \
  --cache-root "$CACHE_ROOT" --labels-root "$GT" --params-json "$TRACK_PARAMS" \
  --all-seqs \
  --out-json "$OUT_DIR/track_all_v3.json" \
  --out-preds-dir "$OUT_DIR/preds_v3" \
  2>&1 | tee "$OUT_DIR/track_all.log" | grep -E "^\[SEQ\]|^\[OVERALL\]|^\[GATE\]"

echo
echo "=============================================================="
echo " 评估完成"
echo "=============================================================="
ls -la "$OUT_DIR"/*.json
echo " 预测目录 (OSD 输入): $OUT_DIR/preds_v3"

# ---------------------------------------------------------------- 5. 与 2026-10-22 官方记录对账
echo
echo "=============================================================="
echo " 5. 与官方记录 osd_sys_sota_trial0080_v3_20261022 逐序列对账"
echo "=============================================================="
python manu/evaluation/cmp_seq_vs_official.py \
  --eval-json "$OUT_DIR/track_all_v3.json" \
  --official-json "/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_v3_20261022/search_json_for_render.json" \
  --out-json "$OUT_DIR/cmp_v3.json" \
  --require-gt-identical --max-df1-pp 0.1
echo "[GATE] 官方逐序列对账闸门：GT 必须 24/24 一致、每序 |dF1| ≤ 0.1pp"
