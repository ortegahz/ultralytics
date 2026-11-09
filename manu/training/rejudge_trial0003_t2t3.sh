#!/bin/bash
# ==============================================================================
# rejudge_trial0003_t2t3.sh —— 用**已训好的** trial_0003 权重重跑一次 T2/T3 门禁
#
# 为什么值得做（2026-11-06）
# --------------------------
# 上一轮（nogmc_struct4）的 trial_0003 是 **head_conv=32** 的结构臂，
# 当时 T2/T3 全 0 ⇒ 被判成「死模型」⇒ 搜索计划据此把 hc32 整个降级。
# ★ 后查明那是**代码 bug**：`load_hm_model()` 没从 ckpt 读 `head_conv`/`late_upsample`
#   ⇒ T2/T3 推理端把模型重建成 hc64 ⇒ head 权重形状不匹配被静默丢弃
#   ⇒ 带随机头推理 ⇒ 热图全 0 ⇒ det=0。
#   （T1 路径读对了 ckpt ⇒ T1=0.873964 是有效的）
# ⇒ 现在 loader 已修，**用同一份权重重跑一次 T2/T3 门禁**即可直接回答：
#     「hc32 到底行不行？」（零训练成本，约 17 min 纯推理）
#
# ⚠️ **铁律一：评测脚本由用户执行** ⇒ 本脚本交给你手动跑，不自动执行。
#
# 用法（服务器，conda uav）：
#     bash manu/training/rejudge_trial0003_t2t3.sh
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
SRC_OUT="${SRC_OUT:-/mnt/data/siping/runs/nogmc_struct4}"
TRIAL="${TRIAL:-trial_0003}"
EPOCH="${EPOCH:-epoch20}"
OUT_ROOT="${OUT_ROOT:-/mnt/data/siping/runs/nogmc_struct4_rejudge}"
GPU="${GPU:-3}"
BASELINE_CACHE="${BASELINE_CACHE:-/mnt/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav || exit 1

CKPT="$SRC_OUT/trials/$TRIAL/weights/$EPOCH.pt"
[ -f "$CKPT" ] || { echo "[FATAL] 找不到权重：$CKPT"; exit 1; }
[ -f "$BASELINE_CACHE" ] || { echo "[FATAL] 找不到基线缓存：$BASELINE_CACHE"; exit 1; }

mkdir -p "$OUT_ROOT/logs"
LOG="$OUT_ROOT/logs/rejudge_${TRIAL}_${EPOCH}.log"

echo "[REJUDGE] ckpt   = $CKPT"
echo "[REJUDGE] 输出   = $OUT_ROOT"
echo "[REJUDGE] 日志   = $LOG"
echo "[REJUDGE] GPU    = $GPU"
echo ""
echo "[REJUDGE] 先看 ckpt 的结构自描述（应当 head_conv=32 / late_upsample=False）："
python - << 'PYEOF'
import torch, sys
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("  ", {k: c.get(k) for k in ("head_conv", "late_upsample", "head_src",
                                  "head_level", "act", "no_p2", "p2_out_scale", "stride")})
PYEOF
echo ""

echo "[REJUDGE] 跑 T2/T3 门禁（纯推理，约 17 min）…"
# ⚠️ --skip-track1：T1 上一轮已经算过（0.873964 有效），这里只补 T2/T3。
#    参数口径照抄 dispatcher（optuna_mix_ratio_search.py:1291-1318）⇒ 数字可比。
#    ⚠️ 参照系数字（--base-*）取 no-GMC 参照系包（与本轮搜索同口径）。
REF_FRAME="${REF_FRAME:-/mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json}"
UAV_DATA="${UAV_DATA:-/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml}"
LQ_DATA="${LQ_DATA:-/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/data.yaml}"

python - "$REF_FRAME" << 'PYEOF' > /tmp/_rej_base_args.txt
import json, sys
ref = json.load(open(sys.argv[1]))
base = ref.get("gate_base") or ref.get("gate") or {}
pairs = [("track1_f1", "--base-track1-f1"), ("track1_tp", "--base-track1-tp"),
         ("track1_fp", "--base-track1-fp"), ("track2_f1", "--base-track2-f1"),
         ("track2_tp", "--base-track2-tp"), ("track2_fp", "--base-track2-fp"),
         ("track2_fn", "--base-track2-fn"), ("track3_f1", "--base-track3-f1"),
         ("track3_tp", "--base-track3-tp"), ("track3_fp", "--base-track3-fp"),
         ("track3_fn", "--base-track3-fn")]
out = []
for k, flag in pairs:
    if k in base:
        out += [flag, str(base[k])]
print("\n".join(out))
PYEOF

python -u manu/training/auto_epoch_gate_check.py \
  --weights "$CKPT" \
  --out-dir "$OUT_ROOT" \
  --gpus 1 \
  --device "$GPU" \
  --infer-gpu "$GPU" \
  --cache-gpus "$GPU" \
  --baseline-cache "$BASELINE_CACHE" \
  --skip-track1 \
  --uav-data "$UAV_DATA" \
  --ft-root "$(dirname "$LQ_DATA")" \
  $(tr '\n' ' ' < /tmp/_rej_base_args.txt) \
  2>&1 | tee "$LOG"

echo ""
echo "[REJUDGE] 完成。关键看两处："
echo "  ① ms_cache*.log 里打印的 [head_conv] 必须是 32（不是 64）"
echo "     ⇒ 若还是 64，说明 loader 没生效"
echo "  ② gate_report.json 里 track2_golden.f1 / track3_blind.f1"
echo "     ⇒ 若 f1 接近 0.99/0.84（基线水平）⇒ hc32 不是死模型，"
echo "        之前的「死模型」结论作废，320 输出家族值得投预算"
echo "     ⇒ 若仍是 0 ⇒ 这次是真的容量问题"
echo ""
echo "[REJUDGE] 验收总表可用："
echo "  python manu/training/npu20ms_acceptance.py --root $OUT_ROOT \\"
echo "      --speed-table manu/pipeline/npu/struct_speed_table.json"
