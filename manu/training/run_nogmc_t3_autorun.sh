#!/bin/bash
# ==============================================================================
# no-GMC **T3 前沿 study** 的服务器端自主收尾流水线（2026-11-03）
#
# 它做什么
# ========
# 今晚训练由**别人**启动（4 张卡 11:00 起被占满）⇒ 本脚本**不启动训练**，
# 只在服务器上**自己跑完**从「训练结束」到「交付件落盘」的全过程（setsid+nohup
# ⇒ 脱离 SSH 会话；SSH 断了也不影响）。
#
#   ① 等训练结束（**三重判据**，见 _lib_train_end.sh）
#        · 墙钟已过 runtime_config.json 的 stop_at
#        · 调度器进程（<out-root>/dispatcher.pid）已退出
#        · **4 张卡上没有训练进程**（nvidia-smi compute-apps + ps 交叉验证）
#      ⇒ 铁律：**评估不得与训练同卡** ⇒ 门禁开始时刻必然晚于最后一条训练臂完成时刻
#   ② 补采 → 重判 → SUMMARY（backfill / rejudge / summarize）
#   ③ **保险层**：先出一版完整可交付件（判定 + 固化 RC + T3 可达性）
#      ⇒ 哪怕后面门禁超时/机器重启，磁盘上**已经有完整交付物**
#   ④ 变体门禁（跨种子 soup + 沿 epoch 轴 SWA），**按价值排序**、硬截止截断
#   ⑤ **最终 finalize**（把 SWA 结果并进附录 C；命中双目标则自动封存）
#   ⑥ 备份 + 写 AUTORUN_DONE.json（每个产物的路径 + md5 + 完成时间）
#      + 把 SUMMARY.md / optuna_summary.csv 拷到 /mnt/data/siping/runs/nogmc_t3_*
#
# 时间闸门（全部可用环境变量临场调）
# =====================================
#   STOP_AT        训练截止（默认读 <OUT_ROOT>/runtime_config.json 的 stop_at）
#   TRAIN_STOP     等训练结束的墙钟时刻（默认同上；显式给出时优先）
#   GATE_DEADLINE  门禁硬截止（到点放弃剩余候选，**保住 finalize 的时间**）
#   FINAL_DEADLINE finalize 硬截止（到点不再等，直接收尾落盘）
#
# 用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
#   setsid nohup bash manu/training/run_nogmc_t3_autorun.sh \
#     >> /mnt/data/siping/runs/nogmc_t3/logs/autorun.log 2>&1 < /dev/null &
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
export RUNS_BASE PROJECT_ROOT
export OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_t3}"
export OLD_ROOT="${OLD_ROOT:-$RUNS_BASE/nogmc_weekend}"
export REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
export STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_t3}"
export RUN_TAG="${RUN_TAG:-nogmc_t3}"
export SELF_TAG="${SELF_TAG:-t3}"
LOG="${LOG:-$OUT_ROOT/logs/autorun.log}"
PLAN="${PLAN:-manu/training/nogmc_t3_plan.json}"
FAMILIES_JSON="${FAMILIES_JSON:-manu/training/nogmc_t3_families.json}"
GATE_DEADLINE="${GATE_DEADLINE:-}"
FINAL_DEADLINE="${FINAL_DEADLINE:-}"
TRAIN_STOP="${TRAIN_STOP:-}"
WAIT_HARD_CAP_MIN="${WAIT_HARD_CAP_MIN:-360}"
RC_ID="${RC_ID:-Candidate_t3}"
RC_NAME="${RC_NAME:-no-GMC T3 前沿（30 臂）最优交接候选}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs"
# ⚠️ 日志只写进 $LOG 一份；stdout 只在**交互式终端**时再打一遍。
#    （若同时 `>> $LOG 2>&1`，tee 写法会让每行在日志里出现**两遍** ⇒ 这里改成
#      「写文件 + 非 TTY 时不打印」，保证日志里每行只有一份。）
say(){ local m="[$(date '+%F %T')] [autorun-t3] $*"; printf '%s\n' "$m" >> "$LOG"; if [ -t 1 ]; then printf '%s\n' "$m"; fi; return 0; }

# shellcheck source=manu/training/_lib_train_end.sh
source manu/training/_lib_train_end.sh || { say "找不到 _lib_train_end.sh ⇒ 退出"; exit 1; }

REF="$REF_DIR/reference_frame.json"
BASE="$REF_DIR/uav_cache_base.pkl"
[ -f "$REF" ]  || { say "缺参照系包 $REF ⇒ 退出"; exit 1; }
[ -f "$BASE" ] || { say "缺基线缓存 $BASE ⇒ 退出"; exit 1; }
[ -f "$PLAN" ] || say "⚠️ 找不到计划文件 $PLAN（不影响收尾，仅提示）"

# ---- 时间闸门 ---------------------------------------------------------------
# 训练截止：优先 runtime_config.json 的 stop_at（与调度器**同源**）
if [ -z "$TRAIN_STOP" ]; then
  TRAIN_STOP="$(python3 -c "import json;print(json.load(open('$OUT_ROOT/runtime_config.json')).get('stop_at',''))" 2>/dev/null || echo "")"
fi
STOP_AT="${STOP_AT:-$TRAIN_STOP}"
[ -z "$TRAIN_STOP" ] && TRAIN_STOP="$STOP_AT"
STOP_S=$(date -d "$STOP_AT" '+%s' 2>/dev/null || echo 0)
GATE_S=$(date -d "$GATE_DEADLINE" '+%s' 2>/dev/null || echo 0)
FINAL_S=$(date -d "$FINAL_DEADLINE" '+%s' 2>/dev/null || echo 0)

say "════ 自主收尾编排（T3 前沿 study）启动 ════"
say "  产物根 $OUT_ROOT ｜ study $STUDY_NAME"
say "  训练截止 STOP_AT=$STOP_AT ｜ TRAIN_STOP=$TRAIN_STOP"
say "  门禁硬截止=${GATE_DEADLINE:-（不限）} ｜ finalize 硬截止=${FINAL_DEADLINE:-（不限）}"

# ---- ① 等训练结束 + GPU 空闲 ------------------------------------------------
say "① 等待训练结束（墙钟 + 调度器已退出 + 4 张卡无训练进程）…"
wait_train_end "$OUT_ROOT" "$TRAIN_STOP" "$WAIT_HARD_CAP_MIN" >> "$LOG" 2>&1
WAIT_RC=$?
if [ "$WAIT_RC" != "0" ]; then
  say "❌ 等待训练结束失败（rc=$WAIT_RC）⇒ **中止**（绝不在训练同卡时评估）"
  exit "$WAIT_RC"
fi
TRAIN_END_EPOCH=$(date '+%s')
say "① 训练已结束、GPU 已释放（epoch=$TRAIN_END_EPOCH）"

# 最后一条训练臂的完成时刻（用于「门禁开始必须晚于它」的断言）
# ⚠️ **Optuna 的 FrozenTrial 没有 `end_time`**（5.0.0 只有 datetime_start/datetime_complete）
#    ⇒ 用 datetime_complete，并保留 end_time 作为旧版兜底。
LAST_TRAIN_END="$(python3 - "$OUT_ROOT" "$STUDY_NAME" <<'PY' 2>>"$LOG" || echo ""
import json, sys
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
out_root, study = sys.argv[1], sys.argv[2]
try:
    st = optuna.load_study(storage="sqlite:///" + out_root + "/study.db", study_name=study)
except Exception as exc:
    print("[AUTORUN] 读 study 失败（跳过末条训练臂断言）：" + str(exc))
    sys.exit(0)


def _end(t):
    for attr in ("datetime_complete", "end_time"):
        v = getattr(t, attr, None)
        if v is not None:
            return v
    return None


ends = [_end(t) for t in st.trials]
ends = [e for e in ends if e is not None]
if not ends:
    print("[AUTORUN] study 里还没有带完成时刻的 trial ⇒ 跳过末条训练臂断言")
    sys.exit(0)
last = max(ends)
print(json.dumps({"iso": last.isoformat(), "epoch": last.timestamp(),
                  "n_complete": sum(1 for t in st.trials if t.state.name == "COMPLETE")}))
PY
)"
say "   最后一条训练臂完成：$LAST_TRAIN_END"

# ---- ② 补采 → 重判 → SUMMARY ------------------------------------------------
say "② 收尾：补采 → 重判 → SUMMARY"
python manu/training/backfill_summary_rows.py --out-root "$OUT_ROOT" \
  --storage "sqlite:///$OUT_ROOT/study.db" --study-name "$STUDY_NAME" --apply >> "$LOG" 2>&1
say "   backfill rc=$?"
python manu/training/rejudge_with_reference.py --out-root "$OUT_ROOT" --ref "$REF" --epochs 0 --apply >> "$LOG" 2>&1
say "   rejudge rc=$?"
python manu/training/summarize_nogmc_search.py --out-root "$OUT_ROOT" --ref "$REF" --top 20 >> "$LOG" 2>&1
say "   summarize rc=$?"

# ---- ③ 保险层：先出一版完整可交付件 ----------------------------------------
# ⚠️ **为什么要这一层**：门禁 + finalize 都在训练之后；机器随时可能重启/断电。
#   ⇒ 先落一版完整交付件（判定 + 固化 RC + T3 可达性），
#      这样**无论后面发生什么，交接材料都已经存在**。
#   ⇒ SWA_WAIT_MIN=0：不等 SWA 门禁（此刻还没跑）⇒ 这一版不含附录 C。
say "③ 保险层：先出一版完整可交付件（判定 + 固化 + T3 可达性）"
LOG="$RUNS_BASE/nogmc_t3_finalize.log" SWA_WAIT_MIN=0 bash manu/training/finalize_nogmc_t3.sh >> "$LOG" 2>&1
say "   finalize(保险层) rc=$?"
say "   → 交付件已落盘：HANDOVER.md / HANDOVER.json / T3_REACHABILITY.md / 冻结 RC"

# ---- ④ 变体门禁（跨种子 soup + SWA），硬截止截断 ----------------------------
say "④ 变体门禁（soup + SWA）"
GATE_LOG="$OUT_ROOT/logs/gate_variants.log"
GATE_DEADLINE="$GATE_DEADLINE" TRAIN_STOP="$TRAIN_STOP" FAMILIES_JSON="$FAMILIES_JSON" \
  LOG="$GATE_LOG" \
  setsid nohup bash manu/training/gate_nogmc_t3_variants.sh >> "$GATE_LOG" 2>&1 < /dev/null &
GATE_PID=$!
GATE_START_EPOCH=$(date '+%s')
say "   门禁脚本 PID=$GATE_PID（启动时刻 epoch=$GATE_START_EPOCH）"
# 断言：门禁开始时刻 **晚于** 最后一条训练臂完成时刻（铁律：评估不得与训练同卡）
LAST_END_EPOCH="$(printf '%s' "$LAST_TRAIN_END" | python3 -c "import json,sys;print(int(json.load(sys.stdin).get('epoch',0)))" 2>/dev/null || echo 0)"
if [ "$LAST_END_EPOCH" -gt 0 ] && [ "$GATE_START_EPOCH" -le "$LAST_END_EPOCH" ]; then
  say "❌ 门禁开始时刻($GATE_START_EPOCH) ≤ 末条训练臂完成时刻($LAST_END_EPOCH) ⇒ 违反铁律 ⇒ 中止"
  exit 9
fi
say "   ✅ 门禁开始时刻晚于最后一条训练臂完成时刻（铁律满足）"

while :; do
  NOW_S=$(date '+%s')
  if [ "$FINAL_S" -gt 0 ] && [ "$NOW_S" -ge "$FINAL_S" ]; then
    say "   到达 finalize 硬截止 $FINAL_DEADLINE ⇒ 不再等门禁"
    break
  fi
  if [ "$GATE_S" -gt 0 ] && [ "$NOW_S" -ge "$GATE_S" ]; then
    say "   到达门禁硬截止 $GATE_DEADLINE ⇒ 放弃剩余候选（已完成的保留在磁盘上）"
    break
  fi
  if ! kill -0 "$GATE_PID" 2>/dev/null; then
    say "   门禁脚本已退出"
    break
  fi
  sleep 30
done

# 硬截止 ⇒ 停掉还在跑的门禁（避免继续抢 CPU/显存；已完成的候选结果保留在磁盘上）
# ⚠️ **只按 PID 杀进程组**，**不要**用 `ps | grep 脚本名` 的写法：
#    那种写法会命中**任何命令行里提到该脚本名**的进程（包括用户自己的 shell
#    ⇒ `kill -- -$pid` 会把对方的进程组一起杀掉）。实测踩到：测试用的 shell 被误杀。
if kill -0 "$GATE_PID" 2>/dev/null; then
  kill -- -"$GATE_PID" 2>/dev/null || kill "$GATE_PID" 2>/dev/null
  say "   停掉门禁进程组 PID=$GATE_PID"
fi
# 门禁脚本的直接子进程（auto_epoch_gate_check.py 等）也一并停掉（按 PPID 精确匹配）
for p in $(ps -eo pid,ppid 2>/dev/null | awk -v g="$GATE_PID" '$2==g {print $1}'); do
  kill "$p" 2>/dev/null && say "   停掉门禁子进程 PID=$p"
done

# ---- ⑤ 最终 finalize（整合 SWA 附录 C；命中则自动封存）---------------------
say "⑤ 最终 finalize（整合 SWA 结果；命中双目标则自动封存）"
LOG="$RUNS_BASE/nogmc_t3_finalize.log" SWA_WAIT_MIN=0 bash manu/training/finalize_nogmc_t3.sh >> "$LOG" 2>&1
say "   finalize rc=$?"

# ---- ⑥ 备份 + DONE 标记 ----------------------------------------------------
say "⑥ 备份 + DONE 标记"
STAMP=$(date '+%Y%m%d_%H%M')
BAK="$RUNS_BASE/${RUN_TAG}_backup_$STAMP"
mkdir -p "$BAK"
for f in SUMMARY.md optuna_summary.csv study.db HANDOVER.md HANDOVER.json \
         SWA_SUMMARY.json T3_REACHABILITY.md T3_REACHABILITY.json \
         crossseed_manifest.json swa_candidates.json; do
  cp -f "$OUT_ROOT/$f" "$BAK/" 2>/dev/null
done
cp -f "$OUT_ROOT/SUMMARY.md" "$RUNS_BASE/${RUN_TAG}_SUMMARY.md" 2>/dev/null
cp -f "$OUT_ROOT/optuna_summary.csv" "$RUNS_BASE/${RUN_TAG}_summary.csv" 2>/dev/null
[ -f "$OUT_ROOT/HANDOVER.md" ] && cp -f "$OUT_ROOT/HANDOVER.md" "$RUNS_BASE/${RUN_TAG}_HANDOVER.md" 2>/dev/null
[ -f "$OUT_ROOT/T3_REACHABILITY.md" ] && cp -f "$OUT_ROOT/T3_REACHABILITY.md" "$RUNS_BASE/${RUN_TAG}_T3_REACHABILITY.md" 2>/dev/null

python3 - "$OUT_ROOT" "$BAK" "$GATE_START_EPOCH" "$LAST_TRAIN_END" <<'PY'
import hashlib, json, os, sys, time
out_root, bak, gate_start, last_train_end = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


arts = {}
for name in ("HANDOVER.md", "HANDOVER.json", "T3_REACHABILITY.md", "T3_REACHABILITY.json",
             "SUMMARY.md", "optuna_summary.csv", "SWA_SUMMARY.json", "AUTORUN_DONE.json",
             "crossseed_manifest.json", "swa_candidates.json", "study.db"):
    p = os.path.join(out_root, name)
    if not os.path.isfile(p):
        continue
    arts[name] = {"path": p, "md5": _md5(p), "bytes": os.path.getsize(p)}

# 门禁报告（每个候选一份）
greports = {}
gate_root = os.path.join(out_root, "swa", "gate")
if os.path.isdir(gate_root):
    for name in sorted(os.listdir(gate_root)):
        rp = os.path.join(gate_root, name, "gate_report.json")
        if os.path.isfile(rp):
            greports[name] = {"path": rp, "md5": _md5(rp)}

payload = {
    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "out_root": out_root,
    "study": "mix_ratio_search_nogmc_t3",
    "backup_dir": bak,
    "candidates_dir": "/mnt/data/siping/runs/CANDIDATES",
    "gate_started_epoch": gate_start,
    "last_training_arm_end": last_train_end,
    "artifacts": arts,
    "gate_reports": greports,
}
with open(os.path.join(out_root, "AUTORUN_DONE.json"), "w", encoding="utf-8") as fh:
    json.dump(payload, fh, ensure_ascii=False, indent=2)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
say "✅ 自主收尾全部完成；备份 $BAK"
