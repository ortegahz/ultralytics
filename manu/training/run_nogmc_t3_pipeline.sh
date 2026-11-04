#!/bin/bash
# ==============================================================================
# **no-GMC T3 前沿** 定点打靶流水线（2026-11-03，窗口到 2026-11-04 09:00）
#
# 目标（用户原话）
# ================
#   「参考判断文档，再尝试一些搜索空间，看能否尽可能接近带 GMC 的 RC2，
#     特别是 T3 能否接近 0.89」
#
# 本轮扩的三处搜索空间（依据 119 条 COMPLETE 记录的受控对分析）
# ==============================================================
#   · `lr_scale`  [0.8, 3.0] → **[0.8, 4.0]**（20/20 受控对里 lr↑ ⇒ T3↑ 单调；
#                                            最优 T3 全挤在 3.0 的天花板上）
#   · `epochs`    {6,8,10,15} → **{6,8,10,15,20,25}**（ep↑ ⇒ T3↑ +0.77~+2.58 pp，
#                                                而 T1 只掉 0.03~0.38 pp；ep=15 也是天花板）
#   · `mix_spec`  收窄到 **{natural,w2.0,w2.5,w3.0,w6.0}**（mix 比↑ ⇒ T3↓，砍掉 w8.0）
#
# 为什么必须开**新 study**
# ========================
#   Optuna 4.9 的分布兼容检查是 study 级：`CategoricalDistribution` 换 choices 会抛
#   "does not support dynamic value space" ⇒ `epochs` / `mix_spec` 一换就必须换 study。
#   （`lr_scale` 是 Float，low/high 其实不参与兼容检查 —— 但既然 `epochs` 已锁死，
#     统一开新 study 更干净，也不会让两个 study 的语义互相污染。）
#
# 时间闸门（全部可用环境变量临场调）
# ==============================
#   TRAIN_STOP  停止派新臂        （默认 2026-11-04 04:10）
#   HARD_CUT    硬截止：派单前判断「这条臂能不能在截止前跑完」⇒ 跑不完就不派
#               （默认 2026-11-04 05:00）
#   ⇒ 门禁最早 05:00 起（评估不得与训练同卡）→ 06:15 收尾 → 09:00 交付
#
# 用法（服务器 `ssh -p 32222 huangzhe@192.168.99.40`，conda `uav`）：
#   setsid nohup bash manu/training/run_nogmc_t3_pipeline.sh \
#     >> /mnt/data/siping/runs/nogmc_t3/logs/pipeline.log 2>&1 < /dev/null &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_t3}"
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
OLD_FOCUS_ROOT="${OLD_FOCUS_ROOT:-$RUNS_BASE/nogmc_focus}"
OLD_WEEKEND_ROOT="${OLD_WEEKEND_ROOT:-$RUNS_BASE/nogmc_weekend}"
LOG="${LOG:-$RUNS_BASE/nogmc_t3_pipeline.log}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_t3}"
PLAN="${PLAN:-manu/training/nogmc_t3_plan.json}"
INIT_MAP="${INIT_MAP:-manu/training/nogmc_weekend_init_weights.json}"
TRAIN_STOP="${TRAIN_STOP:-2026-11-04 04:10}"
HARD_CUT="${HARD_CUT:-2026-11-04 05:00}"
N_TRIALS="${N_TRIALS:-40}"          # 计划 30 条 + 预留 10 条补投位（hard_cut 会挡住 TPE 垃圾臂）
WATCH_ONLY="${WATCH_ONLY:-0}"       # 1 = 只当看门狗：不再启动调度器、不做收尾（收尾交给 autorun）
GPUS="${GPUS:-0,1,2,3}"
WORKERS="${WORKERS:-8}"

# ---- 新空间的三个覆盖（调度器新增的 CLI 参数）----
LR_SCALE_LOW="${LR_SCALE_LOW:-0.8}"
LR_SCALE_HIGH="${LR_SCALE_HIGH:-4.0}"
EPOCH_CHOICES="${EPOCH_CHOICES:-6,8,10,15,20,25}"
MIX_CHOICES="${MIX_CHOICES:-natural,w2.0,w2.5,w3.0,w6.0}"
INIT_CHOICES="${INIT_CHOICES:-rc2,base474}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/trials" "$OUT_ROOT/swa"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

REF="$REF_DIR/reference_frame.json"
BASE="$REF_DIR/uav_cache_base.pkl"
[ -f "$REF" ]  || { say "[FATAL] 缺参照系包 $REF"; exit 1; }
[ -f "$BASE" ] || { say "[FATAL] 缺基线缓存 $BASE"; exit 1; }
[ -f "$PLAN" ] || { say "[FATAL] 缺计划文件 $PLAN"; exit 1; }
[ -f "$INIT_MAP" ] || { say "[FATAL] 缺初始化权重映射 $INIT_MAP"; exit 1; }

# ---- 运行期配置：调度器每派一条臂都会读它（可**不重启调度器**改协议）----
#   stop_at   墙钟硬截止（到点不再派新臂）
#   hard_cut  派单前的「能不能跑完」预判（跑不完 ⇒ 不派，不浪费 GPU）
cat > "$OUT_ROOT/runtime_config.json" <<EOF
{
  "stop_at": "$TRAIN_STOP",
  "hard_cut": "$HARD_CUT",
  "workers": $WORKERS,
  "_note": "2026-11-03 T3 前沿轮；stop_at=停止派单，hard_cut=派单前跑不完就不派"
}
EOF
say "runtime_config: $(cat "$OUT_ROOT/runtime_config.json" | tr -d '\n')"

STOP_S=$(date -d "$TRAIN_STOP" '+%s' 2>/dev/null || echo 0)
CUT_S=$(date -d "$HARD_CUT" '+%s' 2>/dev/null || echo 0)
LAUNCH_H=$(( (STOP_S - $(date '+%s')) / 3600 ))
[ "$LAUNCH_H" -lt 1 ] && LAUNCH_H=1
say "════ T3 前沿流水线启动 ════  训练停派 $TRAIN_STOP | 硬截止 $HARD_CUT | 预算 ${LAUNCH_H} h"

# ---------------------------------------------------------------- 启动调度器
launch_dispatcher(){
  local max_hours="$1"
  setsid nohup python -u manu/training/optuna_mix_ratio_search.py \
    --output-root "$OUT_ROOT" --study-name "$STUDY_NAME" \
    --gpus "$GPUS" --n-trials "$N_TRIALS" --epochs 3 --batch 32 --imgsz 640 --workers "$WORKERS" \
    --max-hours "$max_hours" \
    --longquan-data /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/data.yaml \
    --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
    --baseline-cache "$BASE" --reference-frame "$REF" \
    --extended-space --focus-space \
    --lr-scale-low "$LR_SCALE_LOW" --lr-scale-high "$LR_SCALE_HIGH" \
    --epoch-choices "$EPOCH_CHOICES" --mix-choices "$MIX_CHOICES" --init-choices "$INIT_CHOICES" \
    --init-weights-map "$INIT_MAP" --enqueue-plan "$PLAN" \
    >> "$OUT_ROOT/logs/dispatcher.log" 2>&1 < /dev/null &
  echo $!
}

if [ "$WATCH_ONLY" = "1" ]; then
  # ⚠️ **只当看门狗**：调度器已在跑（PID 记在 dispatcher.pid）⇒ 不再启动第二个，
  #    也不做收尾（收尾由 run_nogmc_t3_autorun.sh 负责 ⇒ 避免两边同时写 optuna_summary.csv）。
  DISP_PID=$(cat "$OUT_ROOT/dispatcher.pid" 2>/dev/null || echo "")
  [ -n "$DISP_PID" ] || { say "[FATAL] WATCH_ONLY 但读不到 dispatcher.pid"; exit 1; }
  say "看门狗模式：接管已有调度器 PID=$DISP_PID（不启动新的、不做收尾）"
else
  DISP_PID=$(launch_dispatcher "$LAUNCH_H")
  echo "$DISP_PID" > "$OUT_ROOT/dispatcher.pid"
  disown 2>/dev/null || true
  say "调度器 PID=$DISP_PID（study=$STUDY_NAME，n-trials=$N_TRIALS，gpus=$GPUS）"
fi

# ---------------------------------------------------------------- 看门狗
# ⚠️ **只在「还有臂没跑完」时重启**：
#    调度器跑完 N_TRIALS 条臂后会**自己退出** ⇒ 若看门狗无脑重启，
#    新调度器会把 WAITING 用尽后**凭空造 TPE 臂**（昨晚实测造出 45 条无 role 的垃圾 PRUNED）。
# ⇒ 这里用 optuna 直接数「计划臂里已终态的条数」，全部终态就不重启。
RESTARTS=0
while :; do
  NOW_S=$(date '+%s')
  if [ "$STOP_S" -gt 0 ] && [ "$NOW_S" -ge "$STOP_S" ]; then say "到达训练停派时刻 ⇒ 退出看门狗"; break; fi
  sleep 60
  kill -0 "$DISP_PID" 2>/dev/null && continue

  # 调度器没了 ⇒ 判断是不是「活已经干完」
  DONE_N=$(OUT_ROOT="$OUT_ROOT" PLAN="$PLAN" STUDY_NAME="$STUDY_NAME" python3 - <<'PY' 2>/dev/null || echo 0
import json, os, optuna
from pathlib import Path
root = Path(os.environ["OUT_ROOT"])
roles = {str(x.get("role", "")) for x in json.loads(Path(os.environ["PLAN"]).read_text(encoding="utf-8"))}
st = optuna.load_study(study_name=os.environ["STUDY_NAME"], storage="sqlite:///" + str(root / "study.db"))
term = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL)
n = sum(1 for t in st.get_trials(deepcopy=False)
        if (t.user_attrs or {}).get("role") in roles and t.state in term)
print(n)
PY
)
  TOTAL_N=$(python3 -c "import json;print(len(json.load(open('$PLAN'))))" 2>/dev/null || echo 0)
  if [ "${DONE_N:-0}" -ge "${TOTAL_N:-0}" ] && [ "${TOTAL_N:-0}" -gt 0 ]; then
    say "计划臂全部终态（$DONE_N/$TOTAL_N）⇒ 不重启调度器"
    break
  fi

  # 真崩了 ⇒ 清孤儿 worker（调度器死了但它派出去的训练子进程还活着 ⇒ 继续占卡）
  say "⚠️ 调度器退出但仍有臂未完成（$DONE_N/$TOTAL_N）⇒ 杀孤儿 worker 后重启"
  OUT_ROOT="$OUT_ROOT" python3 - <<'PY' 2>/dev/null || true
import json, os, signal, time
from pathlib import Path
p = Path(os.environ["OUT_ROOT"]) / "workers.json"
if not p.exists():
    raise SystemExit(0)
try:
    ws = (json.loads(p.read_text(encoding="utf-8")).get("workers") or {})
except Exception:
    raise SystemExit(0)
for _k, w in ws.items():
    pid = int(w.get("pid") or 0)
    if not pid:
        continue
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
            print(f"[ORPHAN] {w.get('trial_name')} pid={pid} <- {sig.name}")
        except Exception:
            break
        time.sleep(3)
PY
  RESTARTS=$((RESTARTS+1))
  REMAIN=$(( (STOP_S - $(date '+%s')) / 3600 )); [ "$REMAIN" -lt 1 ] && REMAIN=1
  say "第 $RESTARTS 次重启调度器（剩余 ${REMAIN}h）"
  DISP_PID=$(launch_dispatcher "$REMAIN")
  echo "$DISP_PID" > "$OUT_ROOT/dispatcher.pid"
  disown 2>/dev/null || true
  sleep 45
done
say "训练阶段结束（看门狗共重启 $RESTARTS 次）"

# ---------------------------------------------------------------- 收尾
if [ "$WATCH_ONLY" = "1" ]; then
  say "看门狗模式 ⇒ 收尾交给 run_nogmc_t3_autorun.sh（避免两边同时写 optuna_summary.csv）"
  exit 0
fi
say "③ 收尾：补采 → 重判 → SUMMARY → 备份"
python manu/training/backfill_summary_rows.py --out-root "$OUT_ROOT" \
  --storage "sqlite:///$OUT_ROOT/study.db" --study-name "$STUDY_NAME" --apply >> "$LOG" 2>&1
python manu/training/rejudge_with_reference.py --out-root "$OUT_ROOT" --ref "$REF" --epochs 0 --apply >> "$LOG" 2>&1
python manu/training/summarize_nogmc_search.py --out-root "$OUT_ROOT" --ref "$REF" --top 20 >> "$LOG" 2>&1
cp -f "$OUT_ROOT/SUMMARY.md" "$RUNS_BASE/nogmc_t3_SUMMARY.md" 2>/dev/null
cp -f "$OUT_ROOT/optuna_summary.csv" "$RUNS_BASE/nogmc_t3_summary.csv" 2>/dev/null
STAMP=$(date '+%Y%m%d_%H%M')
mkdir -p "$RUNS_BASE/nogmc_t3_backup_$STAMP"
cp -f "$OUT_ROOT/SUMMARY.md" "$OUT_ROOT/optuna_summary.csv" "$OUT_ROOT/study.db" \
      "$RUNS_BASE/nogmc_t3_backup_$STAMP/" 2>/dev/null
say "✅ 流水线完成；备份 $RUNS_BASE/nogmc_t3_backup_$STAMP"
