#!/bin/bash
# ==============================================================================
# 宿主侧重判 trial_0003 的 T2/T3 门禁（不占服务器 GPU）
#
# 为什么在宿主上跑
# ----------------
# 宿主 18:30 关机，但**服务器 4 张卡被周末轮占满** ⇒ 重判这 17 min 纯推理
# 放宿主上，零 GPU 争抢。宿主有 torch 2.4.1 + 两套数据集（SSHFS 挂载）。
#
# 为什么值得跑
# ------------
# 上一轮 trial_0003 是 **head_conv=32** 的结构臂，当时 T2/T3 全 0 ⇒ 被判成「死模型」
# ⇒ 整个搜索计划据此降级。★ 后查明那是 **loader bug**（T2/T3 推理端重建模型时丢了
#   head_conv ⇒ head 权重被静默丢弃 ⇒ 带随机头推理 ⇒ 热图全 0）。
#   loader 已修 ⇒ **用同一份权重重跑一次 T2/T3 门禁**即可直接回答
#   「hc32 到底行不行？」（零训练成本）。
#
# 用法（宿主）：
#     bash manu/training/rejudge_trial0003_t2t3_host.sh
# ==============================================================================
set -uo pipefail

REPO="${REPO:-/media/manu/1TB-Volume/workspace/ultralytics}"
MNT="${MNT:-/home/manu/mnt}"
SRC_OUT="$MNT/data/siping/runs/nogmc_struct4"
TRIAL="${TRIAL:-trial_0003}"
EPOCH="${EPOCH:-epoch20}"
OUT_ROOT="${OUT_ROOT:-$MNT/data/siping/runs/nogmc_struct4_rejudge}"
PY="${PY:-/media/manu/1TB-Volume/conda-envs/rknn230/bin/python}"
BASELINE_CACHE="$MNT/data/siping/runs/nogmc_ref_weekend/uav_cache_base.pkl"
REF_FRAME="$MNT/data/siping/runs/nogmc_ref_weekend/reference_frame.json"

cd "$REPO" || exit 1
CKPT="$SRC_OUT/trials/$TRIAL/weights/$EPOCH.pt"
[ -f "$CKPT" ] || { echo "[FATAL] 找不到权重：$CKPT"; exit 1; }
[ -f "$BASELINE_CACHE" ] || { echo "[FATAL] 找不到基线缓存：$BASELINE_CACHE"; exit 1; }

mkdir -p "$OUT_ROOT/logs"
LOG="$OUT_ROOT/logs/rejudge_${TRIAL}_${EPOCH}.log"

echo "[REJUDGE] ckpt = $CKPT"
echo "[REJUDGE] 输出 = $OUT_ROOT"
echo ""
echo "[REJUDGE] ckpt 的结构自描述（应当 head_conv=32）："
"$PY" - "$CKPT" << 'PYEOF'
import torch, sys
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("  ", {k: c.get(k) for k in ("head_conv", "late_upsample", "head_src", "head_level",
                                  "act", "no_p2", "p2_out_scale", "stride")})
PYEOF
echo ""

# 参照系基线数字（no-GMC 口径）
"$PY" - "$REF_FRAME" << 'PYEOF' > /tmp/_rej_base_args.txt
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
print(" ".join(out))
PYEOF

echo "[REJUDGE] 跑 T2/T3 门禁（纯推理，CPU；约 10~25 min）…"
# ⚠️ --skip-track1：T1 上一轮已算过（0.873964 有效），这里只补 T2/T3。
#    宿主无 GPU ⇒ --device cpu。
"$PY" -u manu/training/auto_epoch_gate_check.py \
  --weights "$CKPT" \
  --out-dir "$OUT_ROOT" \
  --gpus 0 \
  --device cpu \
  --infer-gpu 0 \
  --cache-gpus 0 \
  --baseline-cache "$BASELINE_CACHE" \
  --skip-track1 \
  --uav-data "$MNT/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml" \
  --ft-root "$MNT/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc" \
  $(tr '\n' ' ' < /tmp/_rej_base_args.txt) \
  2>&1 | tee "$LOG"

echo ""
echo "[REJUDGE] 完成。关键看两处："
echo "  ① ms_cache*.log 里的 [head_conv] 必须是 32（不是 64）⇒ loader 生效"
echo "  ② gate_report.json 的 track2_golden.f1 / track3_blind.f1"
echo "     · 接近基线（0.98/0.76 量级）⇒ hc32 不是死模型 ⇒ 320 输出家族值得投预算"
echo "     · 仍是 0 ⇒ 这次是真的容量问题"
