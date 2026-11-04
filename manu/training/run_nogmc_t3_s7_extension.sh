#!/bin/bash
# ==============================================================================
# **t0033 家族补到 4 粒子** 的自主续跑（2026-11-04 09:05 上线）
#
# 背景（2026-11-04 09:01 实测）
# ==========================
#   `cs_t0033` = focus `trial_0033` + 昨晚 `t45` 的 **2 粒子**跨种子平均
#     ⇒ T1 = **0.894861**（红线 0.8958967，**只差 0.001036 pp ≈ 2.5 个 TP**）
#        T2 = 0.998679 / FP 40 ✅　T3 = 0.872633 ✅　verdict FAIL（**只因 T1**）
#   而**同配方 4 粒子**的 `cs_t0008` 的 T1 = **0.895901 已过线**
#   ⇒ **「同配方粒子越多、soup 的 T1 越高」**
#   ⇒ 补 2 条同配方同权重空间（**rc2-init**）的种子 ⇒ t0033 家族 4 粒子 ⇒ 目标跨过红线
#   ⇒ 这正是判断文档点名的「补跑 2 个同配方种子做跨种子平均」的**完整落地**。
#
# 本脚本做完全部事（**服务器侧自主，不依赖任何在线会话**）：
#   ① 等当前 3 个变体门禁跑完（GPU 释放；铁律：评估不得与训练同卡）
#   ② 补投 2 条臂（role `INS_T0033_S6` / `INS_T0033_S7`，w3.0·lr3.0·8ep·rc2-init）
#   ③ 派发 worker（GPU 0 / 1），worker 自带训练 + 门禁
#   ④ 重建 8 个 soup 家族（t0033 变 4 粒子）
#   ⑤ 对新的 `cs_t0033` 跑门禁（4 卡）
#   ⑥ 重跑 finalize（让新候选进判定链）+ 重跑交付件自检
#
# 用法（服务器）：
#   cd /tmp/pycharm_project_10ae9e2e
#   setsid nohup bash manu/training/run_nogmc_t3_s7_extension.sh \
#     >> /mnt/data/siping/runs/nogmc_t3/logs/s7_extension.log 2>&1 < /dev/null &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_t3}"
LOG="${LOG:-$OUT_ROOT/logs/s7_extension.log}"
REF="$RUNS_BASE/nogmc_ref_weekend/reference_frame.json"
STUDY_NAME="mix_ratio_search_nogmc_t3"
DS_LQ="/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc"
DS_UAV="/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc"
RC2_W="$PROJECT_ROOT/runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs"

say(){ local m="[$(date '+%F %T')] [s7-ext] $*"; printf '%s\n' "$m" >> "$LOG"; return 0; }

say "════ t0033 补到 4 粒子 · 自主续跑启动 ════"
say "  产物根 $OUT_ROOT"

# ---- ① 等当前门禁结束（GPU 释放）-------------------------------------------
say "① 等当前变体门禁跑完（GPU 释放）…"
WAITED=0
while pgrep -f "auto_epoch_gate_check" > /dev/null 2>&1; do
  sleep 30
  WAITED=$((WAITED + 30))
  if [ "$WAITED" -ge 5400 ]; then
    say "⚠️ 等门禁超时（90 min）⇒ 仍继续（但 GPU 可能被占）"
    break
  fi
done
say "   GPU 已释放（等待 $WAITED s）"

# ---- ② 补投 2 条臂 ---------------------------------------------------------
say "② 补投 2 条 w3.0·lr3.0·8ep·rc2-init 臂（INS_T0033_S6 / INS_T0033_S7）…"
python manu/training/build_nogmc_t3_extra_plan.py \
  --pick INS_T0033_S6,INS_T0033_S7 --out manu/training/nogmc_t3_extra_plan.json >> "$LOG" 2>&1
python manu/training/enqueue_extra_arms.py --out-root "$OUT_ROOT" --study-name "$STUDY_NAME" \
  --plan manu/training/nogmc_t3_plan.json --plan-extra manu/training/nogmc_t3_extra_plan.json \
  >> "$LOG" 2>&1
say "   enqueue rc=$?"

# ---- ③ 取 trial 号 ---------------------------------------------------------
read -r T_A T_B <<< "$(python3 - "$OUT_ROOT" <<'PY'
import optuna, sys
st = optuna.load_study(study_name="mix_ratio_search_nogmc_t3",
                        storage="sqlite:///" + sys.argv[1] + "/study.db")
want = ["INS_T0033_S6", "INS_T0033_S7"]
# ⚠️ **只取 WAITING + role 匹配 + 编号最大的一条**（2026-11-04 09:12 修）：
#    原来会把**陈旧孤儿臂**（同 role、状态 PRUNED、params 全 None、seed=327）
#    一起选中 ⇒ python 打出 3 个号 ⇒ shell 的 `read` 只取前两个
#    ⇒ **误派了 t25 这个 TPE 垃圾臂**，且第二个 worker 的参数变成 `--trial-number 47 48`
#    ⇒ argparse 直接报错退出 ⇒ 续跑链静默走空（门禁在训练没跑完时就重建 soup）。
#    ⇒ 现在：按 role 各取**最新的 WAITING** 臂，且下面校验数量必须正好 2。
best = {}
for t in st.get_trials(deepcopy=False):
    r = (t.user_attrs or {}).get("role")
    if r in want and t.state.name == "WAITING":
        if r not in best or t.number > best[r]:
            best[r] = t.number
print(" ".join(str(best[r]) for r in sorted(want) if r in best))
PY
)"
say "   trial 号：$T_A / $T_B"
if [ -z "${T_A:-}" ] || [ -z "${T_B:-}" ] || [ "${T_A}" = "${T_B}" ]; then
  say "❌ 拿不到**两个不同**的 trial 号 ⇒ 中止（防再次误派）"
  exit 1
fi

# ---- ④ 派发 worker（GPU 0 / 1）-------------------------------------------
run_worker(){
  local tn="$1" gpu="$2" seed="$3"
  python -u manu/training/optuna_mix_ratio_search.py --worker \
    --trial-number "$tn" --gpu-id "$gpu" \
    --output-root "$OUT_ROOT" \
    --longquan-data "$DS_LQ/data.yaml" --uav-data "$DS_UAV/data.yaml" \
    --weights "$RC2_W" --gate-script manu/training/auto_epoch_gate_check.py \
    --baseline-cache "$RUNS_BASE/nogmc_ref_weekend/uav_cache_base.pkl" \
    --epochs 8 --batch 32 --imgsz 640 --workers 8 --gate-epochs last --gate-epoch 0 \
    --seed "$seed" --mix-spec w3.0 --lr-scale 3.0 --warmup-epochs 1.5 \
    --weight-decay 0.000269 --focal-beta 2.75 --offset-weight 0.4 \
    --mosaic 0.15 --scale 0.1 --translate 0.05 --clip-max-norm 5.65 \
    --freeze-bn 0 --backbone-lr-scale 1.0 --freeze-backbone-epochs 0 \
    --init-key rc2 --sched-epochs 8 --reference-frame "$REF" \
    >> "$OUT_ROOT/logs/worker_t${tn}.log" 2>&1
}
say "③ 派发 worker：t$T_A → GPU 0（seed 383）｜ t$T_B → GPU 1（seed 384）"
run_worker "$T_A" 0 383 &
P1=$!
run_worker "$T_B" 1 384 &
P2=$!
wait $P1; say "   worker t$T_A rc=$?"
wait $P2; say "   worker t$T_B rc=$?"

# ---- ⑤ 重建 8 家族 soup（t0033 应变 4 粒子）--------------------------------
say "④ 重建跨种子 soup 家族…"
python manu/training/build_crossseed_soups.py \
  --families-json manu/training/nogmc_t3_families.json \
  --focus-root "$RUNS_BASE/nogmc_focus" --weekend-root "$RUNS_BASE/nogmc_weekend" \
  --out-dir "$OUT_ROOT/swa" --repo "$PROJECT_ROOT" >> "$LOG" 2>&1
say "   soup 构建 rc=$?"
grep -E "平均 [0-9]+ 个种子" "$LOG" | tail -8 >> "$LOG"

# ---- ⑥ 新的 cs_t0033 跑门禁（4 卡）----------------------------------------
say "⑤ 对 4 粒子的 cs_t0033 跑门禁…"
GD="$OUT_ROOT/swa/gate/cs_t0033"
# ⚠️ **先备份旧的 2 粒子门禁报告**（万一新的 4 粒子门禁失败，可回退）
if [ -f "$GD/gate_report.json" ]; then
  cp -a "$GD" "${GD}_2p_backup_$(date '+%m%d_%H%M')" && say "   已备份旧的 2 粒子门禁报告"
fi
rm -rf "$GD"; mkdir -p "$GD"
python manu/training/auto_epoch_gate_check.py \
  --weights "$OUT_ROOT/swa/cs_t0033.pt" --out-dir "$GD" \
  --gpus 4 --device 0 --infer-gpu 0 --cache-gpus 0,1,2,3 \
  --baseline-cache "$RUNS_BASE/nogmc_ref_weekend/uav_cache_base.pkl" \
  --min-track2-f1 0.985 --max-track2-fp 112 --min-track1-f1 0.8958967482400269 \
  --uav-data "$DS_UAV/data.yaml" --ft-root "$DS_LQ" \
  --base-track1-f1 0.9008967482400269 --base-track1-tp 21499 --base-track1-fp 1118 \
  --base-track2-f1 0.981074481074481 --base-track2-tp 16070 --base-track2-fp 52 --base-track2-fn 568 \
  --base-track3-f1 0.7578120617110798 --base-track3-tp 6754 --base-track3-fp 536 --base-track3-fn 3781 \
  >> "$OUT_ROOT/logs/gate_cs_t0033_v2.log" 2>&1
say "   cs_t0033 门禁 rc=$?"

# ---- ⑦ 重跑 finalize + 自检 ------------------------------------------------
say "⑥ 重跑 finalize（让新候选进判定链）…"
LOG="$RUNS_BASE/nogmc_t3_finalize.log" SWA_WAIT_MIN=0 \
  bash manu/training/finalize_nogmc_t3.sh >> "$RUNS_BASE/nogmc_t3_finalize.log" 2>&1
say "   finalize rc=$?"

say "⑦ 重跑交付件自检…"
python manu/training/check_nogmc_t3_deliverables.py --out-root "$OUT_ROOT" --ref "$REF" \
  > "$OUT_ROOT/DELIVERABLE_CHECK.txt" 2>&1
say "   自检 rc=$? ⇒ $OUT_ROOT/DELIVERABLE_CHECK.txt"

say "✅ t0033 补到 4 粒子 · 自主续跑全部完成"
