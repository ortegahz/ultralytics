#!/usr/bin/env bash
# ============================================================
# ★ Δ=3 迭代二：围绕**新基线 B** 的受控单旋钮 A/B（8 序 cs_t0043）
#
#   新基线 B = 交付档 CFG（run_delivery.sh 的 CFG=）
#              + --replay-abstain-peak-below 0.50
#              + --replay-rollback-min-peak 0.80
#              （= 第一轮单旋钮 A/B 选出的推荐旋钮串）
#
#   ★ 受控单旋钮 A/B（唯一变量 = 一个旋钮）：
#     - 每次只**替换** B 里一个旋钮的值，其余一字不改（同一 flag 不出现两次）
#     - 同一模型包缓存 / 同一 28,997 帧 / 同一 GT / 同一工作点 params
#     - CFG 唯一真值 = run_delivery.sh 的 CFG=（多行反斜杠续行，awk 取整块再 eval）
#
#   ★ 硬门禁：**第一个作业**必须是 baseline B 复现门禁 ——
#     与第一轮既有产物
#     runs/lookahead/fi_nogmc/knobscan/runs/cs_t0043/d3/p3__c_peak050_min080/arms.json
#     的 **replay 臂**逐位比对（17 字段严格相等，无容差）。
#     不一致 ⇒ 立刻 exit 3，**不跑任何扫描**。
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3_it2.sh gate   # 只跑 baseline 门禁
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3_it2.sh ab     # 逐旋钮单旋钮 A/B（28 点）
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3_it2.sh mech   # 新机理候选（置信度阈值联动 4 组）
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3_it2.sh all    # gate → ab → mech
#
#   环境变量：
#     JOBS=3      并行度（限并行度，避免与 24 序扫描互相拖慢）
#     THREADS=6   每个进程的 OMP/MKL 线程数
#     OUT=...     产物根（默认 runs/lookahead/fi_nogmc/knobscan2）
#     DELTA=3     目标 Δ
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
OUT="${OUT:-$REPO/runs/lookahead/fi_nogmc/knobscan2}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"
SELF="$REPO/manu/tracking/lookahead/server_overnight/run_knob_scan_d3_it2.sh"
JOBS="${JOBS:-3}"
THREADS="${THREADS:-6}"
DELTA="${DELTA:-3}"
MODEL="cs_t0043"
# 第一轮既有产物（baseline B 的参考）
REF_ARMS="${REF_ARMS:-$REPO/runs/lookahead/fi_nogmc/knobscan/runs/cs_t0043/d3/p3__c_peak050_min080/arms.json}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT/runs" "$OUT/logs"
MANIFEST="$OUT/manifest.tsv"
PROGRESS="$OUT/progress.txt"
[ -f "$OUT/.start_epoch" ] || date +%s > "$OUT/.start_epoch"

# ---- 模型包 --------------------------------------------------------------
declare -A CACHE
CACHE[cs_t0043]="/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache"
CACHE[rc2]="$REPO/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"

FRAME_ARGS=""
if [ -d "$FC" ]; then
  FRAME_ARGS="--frames-cache $FC"
fi

slug() { printf '%s' "$1" | sed -E 's|^--||; s|-|_|g'; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

# ---- 唯一 CFG 真值：直接从交付脚本读取，杜绝配置漂移 ----------------------
# ⚠️ run_delivery.sh 的 CFG 是**多行反斜杠续行**的赋值，单行 grep 会把它截断
#    ⇒ 用 awk 取完整赋值块（直到行尾是 " 的那行），再交给 bash 自己 eval。
load_cfg() {
  CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
  if [ -z "$CFG_ASSIGN" ]; then echo "[FATAL] 无法从 $DELIVERY_SH 解析 CFG" >&2; exit 3; fi
  CFG=""
  eval "$CFG_ASSIGN"
  if [ -z "$CFG" ]; then echo "[FATAL] CFG 解析为空" >&2; exit 3; fi
  # 防截断断言：交付档的关键旋钮必须全部在位（少一个 ⇒ 解析被截断 ⇒ 拒绝跑）
  for _must in "--replay-roi-selfpred-max-px 16" "--replay-rollback-min-peak 0.90" \
               "--replay-polarity auto" "--replay-xcorr-uniq-gate 0.6" "--replay-mode perframe" \
               "--replay-abstain-peak-below 0.30" "--replay-rollback-max-frames 16"; do
    case " $CFG " in
      *" $_must "*) ;;
      *) echo "[FATAL] CFG 缺少必需旋钮 '$_must' ⇒ 解析被截断，拒绝跑" >&2; exit 3 ;;
    esac
  done
  # 新基线 B = 交付档 CFG + peak-below 0.50 + rollback-min-peak 0.80
  local cfg
  cfg="$(printf '%s' "$CFG" | sed -E 's|--replay-abstain-peak-below[[:space:]]+[^[:space:]]+|--replay-abstain-peak-below 0.50|g')"
  cfg="$(printf '%s' "$cfg" | sed -E 's|--replay-rollback-min-peak[[:space:]]+[^[:space:]]+|--replay-rollback-min-peak 0.80|g')"
  for _must in "--replay-abstain-peak-below 0.50" "--replay-rollback-min-peak 0.80"; do
    case " $cfg " in
      *" $_must "*) ;;
      *) echo "[FATAL] 基线 B 缺少必需旋钮 '$_must'" >&2; exit 3 ;;
    esac
  done
  BASE_CFG="$cfg"
}

# ---- 单次运行（唯一变量 = 作业点指定的旋钮）-----------------------------
# $1=model  $2=delta  $3=tag  $4=flag列表(逗号分隔，"-"=不改)  $5=value列表(逗号分隔)
do_run() {
  local m="$1" d="$2" tag="$3" flags="$4" vals="$5"
  local cfg="$BASE_CFG"
  if [ "$flags" != "-" ]; then
    local old_ifs="$IFS"
    IFS=','
    # shellcheck disable=SC2206
    local -a FL=($flags) VA=($vals)
    IFS="$old_ifs"
    if [ "${#FL[@]}" -ne "${#VA[@]}" ]; then echo "[FATAL] flag/value 个数不匹配: $flags / $vals" >&2; return 9; fi
    local i n
    for i in "${!FL[@]}"; do
      cfg="$(printf '%s' "$cfg" | sed -E "s|${FL[$i]}[[:space:]]+[^[:space:]]+|${FL[$i]} ${VA[$i]}|g")"
      case " $cfg " in
        *" ${FL[$i]} ${VA[$i]} "*) ;;
        *) echo "[FATAL] 旋钮替换失败: ${FL[$i]} ${VA[$i]}" >&2; return 9 ;;
      esac
      # 同一 flag 不得出现两次（sed 是全局替换 ⇒ 先确认基线里只出现一次）
      n="$(printf '%s' "$cfg" | grep -o -F -- "${FL[$i]}" | wc -l)"
      if [ "$n" -ne 1 ]; then echo "[FATAL] 旋钮 '${FL[$i]}' 在 CFG 中出现 $n 次" >&2; return 9; fi
    done
  fi
  local o="$OUT/runs/$m/d$d/$tag"
  mkdir -p "$o"
  if [ -s "$o/arms.json" ] && [ -f "$o/.done" ]; then
    log "SKIP $m d$d $tag（已有产物，幂等跳过）"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$tag" "$m" "$d" "$flags" "$vals" "0" "0" "$o/arms.json" "$(date '+%F %T')" "skip" >> "$MANIFEST"
    return 0
  fi
  local t0 t1 rc
  t0=$(date +%s)
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    --cache-root "${CACHE[$m]}" --labels-root "$GT" --frames-root "$FRAMES" $FRAME_ARGS \
    --params-json "$PARAMS" --all-seqs --lookahead replay \
    --arms original,stale,kinematic,replay --delta "$d" $cfg \
    --out-json "$o/arms.json" --out-preds-dir "$o/preds" \
    > "$o/run.log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then touch "$o/.done"; else echo "rc=$rc" > "$o/.rc"; fi
  log "RUN $m d$d $tag ($flags -> $vals) rc=$rc 用时 $((t1-t0))s"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$m" "$d" "$flags" "$vals" "$rc" "$((t1-t0))" "$o/arms.json" "$(date '+%F %T')" "run" >> "$MANIFEST"
  return $rc
}

# ---- baseline B 逐位门禁 -------------------------------------------------
gate_baseline() {
  local ref="$REF_ARMS"
  if [ ! -f "$ref" ]; then echo "[FATAL] 参考产物不存在: $ref" >&2; exit 3; fi
  local tag="gate__baseline_B"
  do_run "$MODEL" "$DELTA" "$tag" "-" "-" || { echo "[FATAL] baseline 运行失败 rc≠0" >&2; exit 3; }
  "$PY" - "$OUT/runs/$MODEL/d$DELTA/$tag/arms.json" "$ref" "$MODEL" "$DELTA" <<'PY'
import json, sys

new_path, ref_path, model, delta = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


def arm_row(path, arm="replay"):
    doc = json.load(open(path))
    a = doc["arms"][arm]
    cur = a["current_frame"]["overall"]
    srv = a["servo"]["overall"]
    return {
        "f1": cur["f1"], "recall": cur["recall"], "precision": cur["precision"],
        "far_per_frame": cur["far_per_frame"], "tp": cur["tp"], "fp": cur["fp"],
        "fn": cur["fn"], "det": cur["det"], "gt": cur["gt"],
        "hit5": srv["hit_rate_5px"],
        "cle_pooled_p50": srv.get("cle_pooled_p50"), "cle_pooled_p95": srv.get("cle_pooled_p95"),
        "cle_pooled_p99": srv.get("cle_pooled_p99"), "cle_pooled_max": srv.get("cle_pooled_max"),
        "cle_pooled_samples": srv.get("cle_pooled_samples"),
        "source_f1": a["source_frame"]["overall"]["f1"],
        "source_far": a["source_frame"]["overall"]["far_per_frame"],
    }


new = arm_row(new_path)
ref = arm_row(ref_path)
bad = [k for k in ref if new.get(k) != ref[k]]
print("=" * 78)
print(f"[GATE] baseline B 逐位校验  model={model} Δ={delta}")
print(f"  新产物: {new_path}")
print(f"  参考  : {ref_path}")
for k in ref:
    flag = "OK  " if new.get(k) == ref[k] else "DIFF"
    print(f"  {flag} {k:22s} new={new.get(k)!r}  ref={ref[k]!r}")
print("=" * 78)
if bad:
    print("[GATE][FAIL] baseline B 与既有产物不一致 ⇒ 扫描器引入了额外变量 ⇒ 停止（不继续扫描）")
    print(f"[GATE][FAIL] 不一致字段: {bad}")
    sys.exit(3)
print(f"[GATE][PASS] baseline B 与既有产物逐位一致（{len(ref)}/{len(ref)} 字段）⇒ 扫描器未引入额外变量")
PY
  local grc=$?
  if [ $grc -ne 0 ]; then exit $grc; fi
}

# ---- worker（xargs -P 并行）----------------------------------------------
if [ "${1:-}" = "--job" ]; then
  load_cfg
  IFS='|' read -r _m _d _tag _flags _vals <<< "$2"
  do_run "$_m" "$_d" "$_tag" "$_flags" "$_vals"
  exit $?
fi

# ---- 作业点表 ------------------------------------------------------------
# 逐旋钮单旋钮 A/B：每个旋钮 4 个取值，其中 1 个取值 = B 本身（跑出来用于自洽性核对）
AB_SPECS=(
  "--replay-abstain-peak-below|0.45"
  "--replay-abstain-peak-below|0.50"
  "--replay-abstain-peak-below|0.55"
  "--replay-abstain-peak-below|0.60"
  "--replay-rollback-min-peak|0.70"
  "--replay-rollback-min-peak|0.75"
  "--replay-rollback-min-peak|0.80"
  "--replay-rollback-min-peak|0.85"
  "--replay-roi-selfpred-max-px|8"
  "--replay-roi-selfpred-max-px|12"
  "--replay-roi-selfpred-max-px|16"
  "--replay-roi-selfpred-max-px|24"
  "--replay-xcorr-uniq-gate|0.5"
  "--replay-xcorr-uniq-gate|0.6"
  "--replay-xcorr-uniq-gate|0.7"
  "--replay-rollback-max-frames|0"
  "--replay-rollback-max-frames|8"
  "--replay-rollback-max-frames|16"
  "--replay-rollback-max-frames|32"
  "--replay-abstain-disp-px|16"
  "--replay-abstain-disp-px|24"
  "--replay-abstain-disp-px|32"
  "--replay-abstain-disp-px|48"
  "--output-max-step-px|16"
  "--output-max-step-px|30"
  "--output-max-step-px|48"
  "--output-max-step-px|64"
)

# 新机理候选：置信度阈值联动（回退阈值随弃权阈值一起收紧/放松）
#   ⚠️ 与第一轮已否证的「弃权/回退**窗口**（disp/frames）联动」不是一回事
MECH_SPECS=(
  "--replay-abstain-peak-below=0.45,--replay-rollback-min-peak=0.70"
  "--replay-abstain-peak-below=0.45,--replay-rollback-min-peak=0.85"
  "--replay-abstain-peak-below=0.55,--replay-rollback-min-peak=0.70"
  "--replay-abstain-peak-below=0.55,--replay-rollback-min-peak=0.85"
)

PHASE="${1:-all}"
JOBS_FILE="$OUT/jobs_${PHASE}.txt"
: > "$JOBS_FILE"

emit_job() {
  local m="$1" d="$2" tag="$3" knobs="$4"
  local fl="" vl=""
  local old_ifs="$IFS"
  IFS=','
  # shellcheck disable=SC2206
  local -a pairs=($knobs)
  IFS="$old_ifs"
  local pr f v
  for pr in "${pairs[@]}"; do
    f="${pr%%=*}"; v="${pr#*=}"
    fl="${fl:+$fl,}$f"; vl="${vl:+$vl,}$v"
  done
  printf '%s|%s|%s|%s|%s\n' "$m" "$d" "$tag" "$fl" "$vl" >> "$JOBS_FILE"
}

case "$PHASE" in
  gate)
    load_cfg
    log "===== PHASE gate：baseline B 逐位门禁（model=$MODEL Δ=$DELTA）====="
    gate_baseline
    ;;
  ab)
    load_cfg
    log "===== PHASE ab：baseline B 逐位门禁 ====="
    gate_baseline
    for spec in "${AB_SPECS[@]}"; do
      flag="${spec%%|*}"; val="${spec##*|}"
      emit_job "$MODEL" "$DELTA" "it2__$(slug "$flag")__${val}" "$flag=$val"
    done
    log "===== PHASE ab：$(wc -l < "$JOBS_FILE") 次单旋钮 A/B（$MODEL Δ=$DELTA，-P $JOBS）====="
    xargs -P "$JOBS" -I{} bash "$SELF" --job "{}" < "$JOBS_FILE"
    ;;
  mech)
    load_cfg
    log "===== PHASE mech：baseline B 逐位门禁 ====="
    gate_baseline
    for spec in "${MECH_SPECS[@]}"; do
      knobs="$spec"
      c="${knobs%%,*}"; mp="${knobs#*,}"
      cv="${c##*=}"; mv="${mp##*=}"
      emit_job "$MODEL" "$DELTA" "it2m__c${cv//./}_min${mv//./}" "$knobs"
    done
    log "===== PHASE mech：$(wc -l < "$JOBS_FILE") 次新机理候选（$MODEL Δ=$DELTA，-P $JOBS）====="
    xargs -P "$JOBS" -I{} bash "$SELF" --job "{}" < "$JOBS_FILE"
    ;;
  all)
    log "===== PHASE all：gate → ab → mech ====="
    bash "$SELF" gate
    bash "$SELF" ab
    bash "$SELF" mech
    ;;
  *)
    echo "用法: $0 {gate|ab|mech|all}"; exit 2
    ;;
esac

date +%s > "$OUT/.end_epoch"
log "PHASE $PHASE 结束。manifest: $MANIFEST"
