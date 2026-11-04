#!/bin/bash
# ==============================================================================
# 服务器端**自主收尾编排**（2026-11-02 18:5x）
#
# 为什么必须有它
# ==============
# 主机 **2026-11-03 06:30 定时关机** ⇒ 我的会话在 06:30~09:00 离线
# ⇒ 收尾链（③ 收尾 → 两条门禁 → finalize → 备份）**不能依赖我在线点命令**，
#   必须在服务器上**自己跑完**（setsid + nohup ⇒ 脱离 SSH 会话）。
#
# 它做四件事
# ==========
#   ① **接管监督**：流水线自带看门狗被停掉（否则它会在调度器退出后反复重启它），
#      本脚本负责「调度器意外退出 ⇒ 断点续跑重启」；到训练截止后进入收尾。
#   ② **③ 收尾**：backfill → rejudge → SUMMARY（与流水线同一套命令）
#   ③ **保险层**：**先**出一版完整可交付件（判定 + 固化 RC + T3 可达性）
#      ⇒ 哪怕后面门禁/关机出事，交付物**已经躺在磁盘上**
#   ④ **等门禁（硬截止）→ 再 finalize**（整合 SWA 附录C；若 SWA 命中双目标则自动封存）
#      → 备份 → 写 AUTORUN_DONE.json
#
# 时间闸门（全部可用环境变量临场调）
# ======================================
#   STOP_AT        训练截止（默认读 runtime_config.json 的 stop_at —— 与调度器同源）
#   GATE_DEADLINE  门禁硬截止（到点放弃剩余候选，**保住 finalize 的时间**）
#   FINAL_DEADLINE finalize 硬截止
#
# 用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
#   setsid nohup bash manu/training/run_nogmc_tail_autorun.sh \
#     >> /mnt/data/siping/runs/nogmc_focus/logs/autorun.log 2>&1 < /dev/null &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
export OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_focus}"
OLD_ROOT="${OLD_ROOT:-$RUNS_BASE/nogmc_weekend}"
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
LOG="${LOG:-$OUT_ROOT/logs/autorun.log}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_focus}"
PLAN="${PLAN:-manu/training/nogmc_focus_plan.json}"
INIT_MAP="${INIT_MAP:-manu/training/nogmc_weekend_init_weights.json}"
GATE_DEADLINE="${GATE_DEADLINE:-2026-11-03 06:05}"
FINAL_DEADLINE="${FINAL_DEADLINE:-2026-11-03 06:25}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs"
say(){ echo "[$(date '+%F %T')] [autorun] $*" | tee -a "$LOG"; }

REF="$REF_DIR/reference_frame.json"
BASE="$REF_DIR/uav_cache_base.pkl"
[ -f "$REF" ] || { say "缺参照系包 $REF ⇒ 退出"; exit 1; }
[ -f "$BASE" ] || { say "缺基线缓存 $BASE ⇒ 退出"; exit 1; }

# ---- 训练截止时刻（与调度器用的 runtime_config.stop_at **同源**）-----------
STOP_AT="${STOP_AT:-$(python3 -c "import json;print(json.load(open('$OUT_ROOT/runtime_config.json')).get('stop_at',''))" 2>/dev/null || echo "")}"
STOP_S=$(date -d "$STOP_AT" '+%s' 2>/dev/null || echo 0)
GATE_S=$(date -d "$GATE_DEADLINE" '+%s' 2>/dev/null || echo 0)

say "════ 自主收尾编排启动 ════"
say "  训练截止 STOP_AT=$STOP_AT ｜ 门禁硬截止=$GATE_DEADLINE ｜ finalize 硬截止=$FINAL_DEADLINE"

# ---- ① 接管监督：停掉流水线自带的看门狗 ------------------------------------
#
# ⚠️ **为什么必须接管**（2026-11-02 18:5x 读码发现）：
#   流水线看门狗的职责是「调度器退出 ⇒ 重启」，它的重启**不带 stop_at 语义**
#   ⇒ 调度器在 stop_at 优雅收尾后退出，看门狗会在 60 s 内把它**再拉起来**，
#      而新调度器因为 stop_at 已过会**立刻退出** ⇒ 每分钟一次的**重启死循环**。
#   ⇒ 监督职责移交本脚本（重启命令带同样的 stop_at ⇒ 重启后立即优雅收尾，不会空转）。
for p in $(ps -eo pid,args | grep "run_nogmc_focus_pipeline[.]sh" | awk '{print $1}'); do
  kill "$p" 2>/dev/null && say "  已停掉流水线看门狗 PID=$p（监督职责移交本脚本）"
done
sleep 2

# ---- ① 监督循环：调度器意外退出 ⇒ 重启；到点 ⇒ 进入收尾 --------------------
RESTARTS=0
while :; do
  dpid="$(cat "$OUT_ROOT/dispatcher.pid" 2>/dev/null | tr -d '[:space:]')"
  alive=0
  if [ -n "$dpid" ] && kill -0 "$dpid" 2>/dev/null; then alive=1; fi
  if [ "$alive" = "1" ]; then sleep 60; continue; fi

  now_s=$(date '+%s')
  if [ "$STOP_S" -gt 0 ] && [ "$now_s" -ge "$STOP_S" ]; then
    say "① 调度器已退出且已过训练截止 ⇒ 进入收尾阶段"
    break
  fi

  # 意外退出 ⇒ 断点续跑重启（与流水线看门狗同一套命令）
  REMAIN=$(( (STOP_S - now_s) / 3600 )); [ "$REMAIN" -lt 1 ] && REMAIN=1
  RESTARTS=$((RESTARTS+1))
  say "⚠️ 调度器不在 ⇒ 第 $RESTARTS 次自动重启（剩余 ${REMAIN}h）"
  setsid nohup python -u manu/training/optuna_mix_ratio_search.py \
    --output-root "$OUT_ROOT" --study-name "$STUDY_NAME" \
    --gpus 0,1,2,3 --n-trials 4096 --epochs 3 --batch 32 --imgsz 640 --workers 8 \
    --max-hours "$REMAIN" \
    --longquan-data /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/data.yaml \
    --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
    --baseline-cache "$BASE" --reference-frame "$REF" \
    --extended-space --focus-space \
    --init-weights-map "$INIT_MAP" --enqueue-plan "$PLAN" \
    >> "$OUT_ROOT/logs/dispatcher.log" 2>&1 < /dev/null &
  echo $! > "$OUT_ROOT/dispatcher.pid"
  sleep 60
done
say "① 监督阶段结束（期间重启 $RESTARTS 次）"

# ---- ② ③ 收尾（backfill → rejudge → SUMMARY）------------------------------
say "② 收尾：补采 → 重判 → SUMMARY"
python manu/training/backfill_summary_rows.py --out-root "$OUT_ROOT" \
  --storage "sqlite:///$OUT_ROOT/study.db" --study-name "$STUDY_NAME" --apply >> "$LOG" 2>&1
say "   backfill rc=$?"
python manu/training/rejudge_with_reference.py --out-root "$OUT_ROOT" --ref "$REF" --epochs 0 --apply >> "$LOG" 2>&1
say "   rejudge rc=$?"
python manu/training/summarize_nogmc_search.py --out-root "$OUT_ROOT" --ref "$REF" --top 20 >> "$LOG" 2>&1
say "   summarize rc=$?"

# ---- ③ 保险层：先出一版完整可交付件 ---------------------------------------
#
# ⚠️ **为什么要这一层**（2026-11-02 18:5x 决策）：
#   门禁 + finalize 都在训练之后；主机 06:30 关机是**硬边界**。
#   ⇒ 先落一版完整交付件（判定 + 固化 RC + T3 可达性），
#      这样**无论后面发生什么，交接材料都已经存在**。
#   ⇒ SWA_WAIT_MIN=0：不等 SWA 门禁（此刻它还没跑完）⇒ 这一版不含附录C。
say "③ 保险层：先出一版完整交付件（判定 + 固化 + T3 可达性）"
SWA_WAIT_MIN=0 bash manu/training/finalize_nogmc_focus.sh >> "$LOG" 2>&1
say "   finalize(保险层) rc=$?"
say "   → 交付件已落盘：HANDOVER.md / HANDOVER.json / T3_REACHABILITY.md / 冻结 RC"

# ---- ④ 等门禁（硬截止）→ 最终 finalize -------------------------------------
say "④ 等门禁收尾（硬截止 $GATE_DEADLINE）"
while :; do
  now_s=$(date '+%s')
  if [ "$GATE_S" -gt 0 ] && [ "$now_s" -ge "$GATE_S" ]; then
    say "   到达门禁硬截止 ⇒ 不再等"
    break
  fi
  swa_done=0; cs_done=0
  grep -q "SWA 变体门禁全部完成" "$OUT_ROOT/logs/gate_swa.log" 2>/dev/null && swa_done=1
  grep -q "跨种子 soup 门禁完成" "$OUT_ROOT/logs/gate_crossseed.log" 2>/dev/null && cs_done=1
  if [ "$swa_done" = "1" ] && [ "$cs_done" = "1" ]; then
    say "   两条门禁线都已完成"
    break
  fi
  sleep 30
done

# 硬截止 ⇒ 停掉还在跑的门禁（避免继续抢 CPU/显存；已完成的候选结果保留在磁盘上）
for p in $(ps -eo pid,args | grep -E "gate_swa_variants[.]sh|gate_crossseed_soups[.]sh" | awk '{print $1}'); do
  kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
  say "   停掉门禁脚本 PID=$p（进程组）"
done

# ---- ⑤ 最终 finalize（整合 SWA 附录C；命中则自动封存）---------------------
say "⑤ 最终 finalize（整合 SWA 结果；命中双目标则自动封存）"
SWA_WAIT_MIN=2 bash manu/training/finalize_nogmc_focus.sh >> "$LOG" 2>&1
say "   finalize rc=$?"

# ---- ⑥ 备份 + DONE 标记 ---------------------------------------------------
say "⑥ 备份 + DONE 标记"
STAMP=$(date '+%Y%m%d_%H%M')
BAK="$RUNS_BASE/nogmc_focus_backup_$STAMP"
mkdir -p "$BAK"
for f in SUMMARY.md optuna_summary.csv study.db HANDOVER.md HANDOVER.json \
         SWA_SUMMARY.json T3_REACHABILITY.md T3_REACHABILITY.json crossseed_manifest.json; do
  cp -f "$OUT_ROOT/$f" "$BAK/" 2>/dev/null
done
cp -f "$OUT_ROOT/HANDOVER.md" "$RUNS_BASE/nogmc_focus_HANDOVER.md" 2>/dev/null
cp -f "$OUT_ROOT/T3_REACHABILITY.md" "$RUNS_BASE/nogmc_focus_T3_REACHABILITY.md" 2>/dev/null
cp -f "$OUT_ROOT/optuna_summary.csv" "$RUNS_BASE/nogmc_focus_summary.csv" 2>/dev/null
cp -f "$OUT_ROOT/SUMMARY.md" "$RUNS_BASE/nogmc_focus_SUMMARY.md" 2>/dev/null

python3 - "$OUT_ROOT" "$BAK" <<'PY'
import hashlib, json, os, sys, time
out_root, bak = sys.argv[1], sys.argv[2]
arts = {}
for name in ("HANDOVER.md", "HANDOVER.json", "T3_REACHABILITY.md", "T3_REACHABILITY.json",
             "SUMMARY.md", "optuna_summary.csv", "SWA_SUMMARY.json", "AUTORUN_DONE.json"):
    p = os.path.join(out_root, name)
    if not os.path.isfile(p):
        continue
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    arts[name] = {"md5": h, "bytes": os.path.getsize(p)}
payload = {
    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "artifacts": arts,
    "backup_dir": bak,
    "candidates_dir": "/mnt/data/siping/runs/CANDIDATES",
}
with open(os.path.join(out_root, "AUTORUN_DONE.json"), "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
say "✅ 自主收尾全部完成；备份 $BAK"
