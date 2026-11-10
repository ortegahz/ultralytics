#!/usr/bin/env bash
# ==============================================================================
# 路线二启动器：P1/P2 + 通道瘦身 + relu（2026-11-10 新目标）
#
# 目标：在保住微小目标召回的前提下，把板卡 int8 压到 <=30 ms。
#   * act = relu（用户口径：NPU 推理效率最高；板卡实测 relu 比 silu 快 ~1.5x）
#   * head 分辨率保留 P2（4px 特征）⇒ 微小目标不被 8px 网格物理销毁
#   * head 通道瘦身（hc64 -> 48/40/32/24）⇒ 用通道换延迟
#   * late_upsample ⇒ 输出仍是 2px 网格（320x320 输出，160x160 计算）
#   * init = RC2(trial_0080) ⇒ 骨干/颈继承已训好的高分辨率特征通路
#   * head_init = merge_abs ⇒ 换 head_src/宽度时**按通道合并继承**再取绝对值
#     （不把头打回原形；abs 是因为 silu 训出的负权重在 relu 下会死）
#   * unfreeze_b0b4 = 1 ⇒ all-relu 必须解冻底模早期层（旧 all-relu 线 T2 虚警 FP=118 的根因之一）
#   * min_radius = 0 + train_on_native_grid = 1 ⇒ 监督落在头的原生网格，单格点目标也算正样本
#   * epochs 50（用户口径 50~100，不再用 8 轮）
#
# 用法：bash launch_route2_p2.sh <arm_id> <gpu>
#   arm_id ∈ {hc32, hc64, hc32late, hc48late, hc32late_s05, p1hc64, p1late32, p1hcs05}
# ==============================================================================
#
# ★ 2026-11-10 10:05 板卡实测补充（一次会话，8 探针 + 2 锚点，锚点漂移 <0.4%）：
#   真 2px（P1 全路径，320 头）**能进 30 ms**：p1_hc32 + 颈收窄0.5 = 28.714 ms（整帧 29.954）
#   p1_late（P1 细节 avg_pool 后与 P2 融合，160 算 / 320 输出）= 21.2~22.6 ms
#   ⇒ 新增两个臂：**p1late32**（零形状变化 ⇒ RC2 权重全继承，22.590 ms）
#                     **p1hcs05**（真 2px 极限方案，28.714 ms，颈 2 个 conv 需重训）
# ==============================================================================
set -euo pipefail

ARM="${1:?用法: launch_route2_p2.sh <arm_id> <gpu>}"
GPU="${2:?用法: launch_route2_p2.sh <arm_id> <gpu>}"

REPO=/tmp/pycharm_project_10ae9e2e
RUNS=/mnt/data/siping/runs/npu20ms
ROOT=/mnt/data/siping/runs/npu20ms_offcal
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
RC2=/mnt/data/siping/runs/nogmc_weekend/trials/trial_0080/weights/epoch8.pt
BASE_CACHE=/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl
REF=/mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json
LQ=/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc
UAVD=/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc

TRIAL_BASE="${TRIAL_BASE:-9600}"
EPOCHS="${EPOCHS:-50}"
SEED_BASE="${SEED_BASE:-19000}"
HEAD_INIT="${HEAD_INIT:-merge_abs}"
# ★ 2026-11-10 10:30：init 起点可切换（受控 A/B 用）
#   默认 RC2(trial_0080) —— 但 RC2 的 nano 召回已被搜索配方打掉 3 pp（0.6978→0.6678）
#   INIT_WEIGHTS=fb ⇒ 冻结基线 trial_0474（nano 0.6978 的最佳资产，P0 由训练侧自动跳过）
FB=/mnt/data/siping/runs/nogmc_weekend/seed_weights/frozen_base_trial_0474_best.pt
INIT_WEIGHTS="${INIT_WEIGHTS:-$RC2}"
# ★ 2026-11-10 11:10：混训比例可切换
#   默认 w2.0（P(UAV)=0.667）；冻结基线 trial_0474 是 **UAV 单域**（P0 NAS 研究的 --data 只给 UAV）
#   ⇒ 「LQ 稀释微小目标分布」是可检验的机制 ⇒ MIX_SPEC=w8.0（P(UAV)=0.889）做单变量对照
MIX_SPEC="${MIX_SPEC:-w2.0}"
# UNFREEZE：底模 b0..b4 是否解冻（all-relu 线教训 ⇒ 默认 1；交付件 trial_0031 用的是 uf0）
UNFREEZE="${UNFREEZE:-1}"

case "$ARM" in
  hc32)        TRIAL=$((TRIAL_BASE+1)); HC=32; LATE=0; SC=1.0; DESC="P2 160输出 hc32" ;;
  hc64)        TRIAL=$((TRIAL_BASE+2)); HC=64; LATE=0; SC=1.0; DESC="P2 160输出 hc64(满宽天花板)" ;;
  hc32late)    TRIAL=$((TRIAL_BASE+3)); HC=32; LATE=1; SC=1.0; DESC="★P2+late 320输出 hc32(主力候选)" ;;
  hc48late)    TRIAL=$((TRIAL_BASE+4)); HC=48; LATE=1; SC=1.0; DESC="P2+late 320输出 hc48" ;;
  hc32late_s05) TRIAL=$((TRIAL_BASE+5)); HC=32; LATE=1; SC=0.5; DESC="P2+late 320输出 hc32 neck收窄0.5" ;;
  p1hc64)      TRIAL=$((TRIAL_BASE+6)); HC=64; LATE=0; SC=1.0; HEAD_SRC=p1; HEAD_LEVEL=p1; DESC="P1 320头 hc64(召回基准,>30ms)" ;;
  # ★ 2026-11-10 新增：P1 细节参与融合（late）—— 21.2~22.6 ms，RC2 权重**零形状变化全继承**
  p1late32)    TRIAL=$((TRIAL_BASE+3)); HC=32; LATE=1; SC=1.0; HEAD_SRC=p1; HEAD_LEVEL=p1; DESC="★P1细节融合+late hc32(320输出,22.590ms实测)" ;;
  # ★ 2026-11-10 13:10：head_conv=64 ⇒ **头张量与冻结基线的 P1 头逐形状一致 ⇒ 可完整继承**
  #   （hc32 必须裁剪合并、丢一半滤波器）⇒ P1/P2 家族对 nano 的最后一张牌
  p1late64)    TRIAL=$((TRIAL_BASE+3)); HC=64; LATE=1; SC=1.0; HEAD_SRC=p1; HEAD_LEVEL=p1; DESC="★P1细节融合+late hc64(头可完整继承,速度待测)" ;;
  # ★ 真 2px 进 30 ms 的唯一结构（28.714 ms，整帧 29.954）；颈收窄 ⇒ fuse_p1/fuse_p2 需重训
  p1hcs05)     TRIAL=$((TRIAL_BASE+4)); HC=32; LATE=0; SC=0.5; HEAD_SRC=p1; HEAD_LEVEL=p1; DESC="★真2px hc32+颈0.5(28.714ms实测)" ;;
  # ★★ 2026-11-10 11:40：**P3 交付架构 + 最佳起点/配方**（对照用，非 P1/P2 路线）
  #   P3/hc64/late 的**头张量与冻结基线的 P1 头逐形状兼容**（48→64 / 64→64 / 64→128，
  #   late 分支无权重）⇒ **可 100% 继承冻结基线的头**（nano 能力所在）⇒ 最高先验臂
  p3late64)    TRIAL=$((TRIAL_BASE+5)); HC=64; LATE=1; SC=1.0; HEAD_SRC=p3; HEAD_LEVEL=p3; DESC="P3 320输出 hc64 + 最佳起点/配方(对照交付架构)" ;;
  *) echo "未知 arm: $ARM" >&2; exit 2 ;;
esac
: "${HEAD_SRC:=p2_fused}"
: "${HEAD_LEVEL:=p2_fused}"

SEED=$((SEED_BASE+TRIAL))
LOGDIR="$ROOT/logs/route2"
mkdir -p "$LOGDIR"

echo "[ROUTE2] arm=$ARM trial=$TRIAL gpu=$GPU hc=$HC late=$LATE sc=$SC head=$HEAD_SRC | $DESC"
echo "[ROUTE2] init=$(basename "$INIT_WEIGHTS") head_init=$HEAD_INIT epochs=$EPOCHS mix=$MIX_SPEC"
cd "$REPO"
export PYTHONPATH=.
exec "$PY" -u manu/training/optuna_mix_ratio_search.py --worker \
  --trial-number "$TRIAL" --gpu-id "$GPU" --output-root "$ROOT" \
  --longquan-data "$LQ" --uav-data "$UAVD" \
  --weights "$INIT_WEIGHTS" --baseline-cache "$BASE_CACHE" \
  --epochs "$EPOCHS" --sched-epochs "$EPOCHS" --batch 32 --imgsz 640 --workers 8 --seed "$SEED" \
  --mix-spec "$MIX_SPEC" --lr-scale 3.5 --warmup-epochs 1.5 --weight-decay 0.000269 \
  --focal-beta 2.75 --offset-weight 0.4 --mosaic 0.15 --scale 0.10 --translate 0.05 \
  --clip-max-norm 5.65 --freeze-bn 0 --backbone-lr-scale 1.0 --freeze-backbone-epochs 0 \
  --no-p2 0 --head-level "$HEAD_LEVEL" --act relu --head-conv "$HC" --late-upsample "$LATE" \
  --unfreeze-b0b4 "$UNFREEZE" --head-src "$HEAD_SRC" --p2-out-scale "$SC" \
  --head-init "$HEAD_INIT" --head-seed "$SEED" \
  --train-on-native-grid 1 --min-radius 0 \
  --reference-frame "$REF" --skip-gate \
  > "$LOGDIR/route2_${ARM}_t${TRIAL}_${HEAD_INIT}.log" 2>&1
