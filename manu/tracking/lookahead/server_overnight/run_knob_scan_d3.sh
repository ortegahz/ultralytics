#!/usr/bin/env bash
# ============================================================
# ★ P3 · Δ=3 小延时专属：受控单旋钮 A/B 扫描器 + 新机理候选
#
#   目的：
#     Δ=3 时 replay 已追回 93.7~96.6% 的滞后代价 ⇒ 剩余缺口是
#     FAR（+0.003001 vs oracle）、hit@5px（−0.8354pp）、CLE p95（+0.6700px）。
#     本脚本在 **Δ=3 / 8 序 gate 集** 上做**受控单旋钮 A/B**：
#     每次只改一个旋钮，其余一字不改；扫完 7 个关键旋钮后，
#     再跑「新机理候选」（Δ 自适应 roimax、弃权/回退窗口联动）。
#
#   ★ 严格受控 A/B（唯一变量 = 一个旋钮）：
#     - 同一模型包缓存 / 同一 28,997 帧 / 同一 GT（Final_Labels_merged_20261010）
#     - 同一工作点 params json（system_sota_longquan_trial0080_delivery_v3_params.json）
#     - 同一插件 CFG：**直接从 run_delivery.sh 读取**（杜绝配置漂移）
#     - 改旋钮 = **替换** CFG 里该旋钮的值，不追加第二个同名单 ⇒ 语义唯一
#
#   ★ 门禁：先跑一次 baseline（所有旋钮取交付档值），与既有产物
#     runs/lookahead/fi_nogmc/<model>/d<Δ>/arms.json 的 replay 臂**逐位比对**
#     （F1/R/P/FAR/hit@5px/CLE 严格相等 + 整数计数相等）。
#     不一致 ⇒ 立刻 exit 3，**不再跑任何扫描**（说明扫描器引入了额外变量）。
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh gate   # 只跑 baseline 门禁
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh p1     # cs_t0043 单旋钮 A/B（21 次）
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh p2     # rc2 复跑（P2_KNOBS 指定白名单）
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh p3     # 新机理候选（P3_KNOBS）
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh d24    # Δ=2/4 验证（ROI_FLAG/ROI_VAL）
#     bash manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh all
#
#   环境变量：
#     JOBS=4          并行度（封顶 4；机器 80 核但已有其它扫描在跑）
#     THREADS=6       每个进程的 OMP/MKL 线程数
#     OUT=...         产物根目录（默认 runs/lookahead/fi_nogmc/knobscan）
#     P2_KNOBS=...    p2 阶段的旋钮白名单（空格分隔的 "flag=value"）
#     P3_KNOBS=...    p3 阶段的旋钮白名单（空格分隔的 "tag=flag=value,flag=value"）
#     ROI_FLAG/ROI_VAL d24 阶段要验证的旋钮与取值
#     CACHE_CS=...    cs_t0043 的 gate 缓存根（默认 = 8 序那份）
#     CACHE_RC2=...   rc2 的 gate 缓存根（默认 = 8 序那份）
#     REFROOT=...     baseline 门禁的参考产物根（默认 runs/lookahead/fi_nogmc）
#     REF_ARMS=...    baseline 门禁的参考产物**单文件**（给了就覆盖 REFROOT）
#     EXTRA_BASE_CFG= 追加到交付档 CFG 之后的基线旋钮串（空格分隔）
#                     ⇒ 用于「交付档 CFG + 出框门」这类基线扩展；
#                       追加的 flag 若已存在于 CFG ⇒ 拒绝跑（同 flag 两次 = 语义二义）
#     SPECS=...       p1 阶段作业点表覆盖（空格分隔的 "flag|value"）
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
OUT="${OUT:-$REPO/runs/lookahead/fi_nogmc/knobscan}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"
SELF="$REPO/manu/tracking/lookahead/server_overnight/run_knob_scan_d3.sh"
JOBS="${JOBS:-4}"
THREADS="${THREADS:-6}"
DELTA="${DELTA:-3}"
ROI_FLAG="${ROI_FLAG:---replay-roi-selfpred-max-px}"
ROI_VAL="${ROI_VAL:-8}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT/runs" "$OUT/logs"
MANIFEST="$OUT/manifest.tsv"
PROGRESS="$OUT/progress.txt"

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
  # 基线扩展：把 EXTRA_BASE_CFG 的旋钮追加到交付档 CFG 之后
  # ★ 追加的 flag 若已存在于 CFG ⇒ 同一 flag 出现两次 = 语义二义 ⇒ 拒绝跑
  if [ -n "${EXTRA_BASE_CFG:-}" ]; then
    for _f in $EXTRA_BASE_CFG; do
      case "$_f" in
        --*) case " $CFG " in
               *" $_f "*) echo "[FATAL] EXTRA_BASE_CFG 的 '$_f' 已存在于交付档 CFG ⇒ 同 flag 两次，拒绝跑" >&2; exit 3 ;;
             esac ;;
      esac
    done
    CFG="$CFG $EXTRA_BASE_CFG"
  fi
}

slug() { printf '%s' "$1" | sed -E 's|^--||; s|-|_|g'; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$PROGRESS"; }

# ---- 模型包 / 参考产物 ---------------------------------------------------
declare -A CACHE
declare -A LABEL
CACHE[cs_t0043]="/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache"
# ⚠️ 数组元素必须写 **${CACHE[key]}**（带花括号）：`$CACHE[key]` 会被解析成
#    `$CACHE`（= 元素 0，未定义）+ 字面 `[key]` ⇒ `set -u` 下直接 unbound variable 报错。
CACHE[cs_t0043]="${CACHE_CS:-${CACHE[cs_t0043]}}"
LABEL[cs_t0043]="cs_t0043 / no-GMC 语义（主）"
CACHE[rc2]="$REPO/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"
CACHE[rc2]="${CACHE_RC2:-${CACHE[rc2]}}"
LABEL[rc2]="RC2 trial_0080ep3 / GMC 语义（对照）"
CACHE[trial_0049]="/mnt/data/siping/runs/nogmc_t3/trials/trial_0049/gate/epoch20/ms_cache"
LABEL[trial_0049]="trial_0049 ep20 / no-GMC 语义"
REFROOT="${REFROOT:-$REPO/runs/lookahead/fi_nogmc}"
REF_ARMS="${REF_ARMS:-}"

FRAME_ARGS=""
if [ -d "$FC" ]; then
  FRAME_ARGS="--frames-cache $FC"
fi

# ---- 单次运行 ------------------------------------------------------------
# $1=model  $2=delta  $3=tag  $4=flag列表(逗号分隔，"-"=不改)  $5=value列表(逗号分隔)
do_run() {
  local m="$1" d="$2" tag="$3" flags="$4" vals="$5"
  local cfg="$CFG"
  if [ "$flags" != "-" ]; then
    local old_ifs="$IFS"
    IFS=','
    # shellcheck disable=SC2206
    local -a FL=($flags) VA=($vals)
    IFS="$old_ifs"
    if [ "${#FL[@]}" -ne "${#VA[@]}" ]; then echo "[FATAL] flag/value 个数不匹配: $flags / $vals" >&2; return 9; fi
    local i
    for i in "${!FL[@]}"; do
      n="$(printf '%s' "$cfg" | grep -o -F -- "${FL[$i]}" | wc -l)"
      if [ "$n" -gt 1 ]; then echo "[FATAL] 旋钮 '${FL[$i]}' 在 CFG 中出现 $n 次" >&2; return 9; fi
      if [ "$n" -eq 1 ]; then
        # 基线 CFG 里已有该旋钮 ⇒ **替换**它的值（同一 flag 不出现两次）
        cfg="$(printf '%s' "$cfg" | sed -E "s|${FL[$i]}[[:space:]]+[^[:space:]]+|${FL[$i]} ${VA[$i]}|g")"
      else
        # 基线 CFG 里没有该旋钮 ⇒ 用 CLI 默认值的场景，**追加**（不追加就无法改它）
        cfg="$cfg ${FL[$i]} ${VA[$i]}"
      fi
      case " $cfg " in
        *" ${FL[$i]} ${VA[$i]} "*) ;;
        *) echo "[FATAL] 旋钮替换失败: ${FL[$i]} ${VA[$i]}" >&2; return 9 ;;
      esac
    done
  fi
  local o="$OUT/runs/$m/d$d/$tag"
  mkdir -p "$o"
  if [ -s "$o/arms.json" ] && [ -f "$o/.done" ]; then
    log "SKIP $m d$d $tag（已有产物，幂等跳过）"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$m" "$d" "$flags" "$vals" "0" "0" "$o/arms.json" "$(date '+%F %T')" >> "$MANIFEST"
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$m" "$d" "$flags" "$vals" "$rc" "$((t1-t0))" "$o/arms.json" "$(date '+%F %T')" >> "$MANIFEST"
  return $rc
}

# ---- baseline 逐位门禁 ---------------------------------------------------
gate_baseline() {
  local m="$1" d="$2"
  local ref="${REF_ARMS:-$REFROOT/$m/d$d/arms.json}"
  if [ ! -f "$ref" ]; then echo "[FATAL] 参考产物不存在: $ref" >&2; exit 3; fi
  local tag="gate__baseline"
  do_run "$m" "$d" "$tag" "-" "-" || { echo "[FATAL] baseline 运行失败 rc≠0" >&2; exit 3; }
  "$PY" - "$OUT/runs/$m/d$d/$tag/arms.json" "$ref" "$m" "$d" <<'PY'
import json, sys
new_path, ref_path, model, delta = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ARMS = ("original", "stale", "kinematic", "replay")
FIELDS = ("f1", "recall", "precision", "far_per_frame", "tp", "fp", "fn", "det", "gt")


def rows(path):
    doc = json.load(open(path))
    out = {}
    for arm in ARMS:
        a = doc["arms"][arm]
        cur = a["current_frame"]["overall"]
        srv = a["servo"]["overall"]
        out[arm] = {
            **{k: cur.get(k) for k in FIELDS},
            "hit5": srv.get("hit_rate_5px"),
            "cle_pooled_p50": srv.get("cle_pooled_p50"), "cle_pooled_p95": srv.get("cle_pooled_p95"),
            "cle_pooled_p99": srv.get("cle_pooled_p99"), "cle_pooled_max": srv.get("cle_pooled_max"),
            "cle_pooled_samples": srv.get("cle_pooled_samples"),
            "source_f1": (a.get("source_frame") or {}).get("overall", {}).get("f1"),
            "source_far": (a.get("source_frame") or {}).get("overall", {}).get("far_per_frame"),
        }
    return out


new = rows(new_path)
ref = rows(ref_path)
bad = [f"{arm}.{k}" for arm in ARMS for k in ref[arm] if new[arm].get(k) != ref[arm][k]]
print("=" * 78)
print(f"[GATE] baseline 逐位校验（四臂）  model={model} Δ={delta}")
print(f"  新产物: {new_path}")
print(f"  参考  : {ref_path}")
for arm in ARMS:
    for k in ref[arm]:
        flag = "OK  " if new[arm].get(k) == ref[arm][k] else "DIFF"
        print(f"  {flag} {arm:10s}.{k:18s} new={new[arm].get(k)!r}  ref={ref[arm][k]!r}")
print("=" * 78)
if bad:
    print(f"[GATE][FAIL] baseline 与既有产物不一致（{len(bad)} 处）⇒ 扫描器引入了额外变量 ⇒ 停止（不继续扫描）")
    print(f"[GATE][FAIL] 不一致字段: {bad}")
    sys.exit(3)
print(f"[GATE][PASS] baseline 与既有产物逐位一致（{len(ARMS)} 臂 × {len(ref['replay'])} 字段）⇒ 扫描器未引入额外变量")
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

# ---- 阶段定义 ------------------------------------------------------------
PHASE="${1:-all}"

# p1：cs_t0043 Δ=3 的 7 组单旋钮 A/B（每组 3 个取值，含 baseline 值）
P1_SPECS=(
  "--replay-roi-selfpred-max-px|4"
  "--replay-roi-selfpred-max-px|8"
  "--replay-roi-selfpred-max-px|12"
  "--replay-roi-selfpred-max-px|16"
  "--replay-abstain-disp-px|16"
  "--replay-abstain-disp-px|24"
  "--replay-abstain-disp-px|32"
  "--replay-xcorr-uniq-gate|0.5"
  "--replay-xcorr-uniq-gate|0.6"
  "--replay-xcorr-uniq-gate|0.7"
  "--replay-rollback-max-frames|0"
  "--replay-rollback-max-frames|8"
  "--replay-rollback-max-frames|16"
  "--output-max-step-px|16"
  "--output-max-step-px|30"
  "--output-max-step-px|48"
  "--replay-abstain-peak-below|0.20"
  "--replay-abstain-peak-below|0.30"
  "--replay-abstain-peak-below|0.40"
  "--replay-rollback-min-peak|0.85"
  "--replay-rollback-min-peak|0.90"
  "--replay-rollback-min-peak|0.95"
)

P2_KNOBS="${P2_KNOBS:-}"
P3_KNOBS="${P3_KNOBS:-}"

# ★ 作业清单文件必须**按本次调用唯一**（PID 后缀）——
#   否则并发跑两个阶段时，后一个阶段的 `: > "$JOBS_FILE"` 会把前一个阶段的清单截断，
#   前一个阶段的 xargs 读不到后面的作业 ⇒ **静默丢作业**（本轮真踩到）。
JOBS_FILE="$OUT/jobs_${PHASE}_$$.txt"
: > "$JOBS_FILE"

case "$PHASE" in
  gate)
    load_cfg
    log "===== PHASE gate：baseline 逐位门禁 ====="
    gate_baseline "cs_t0043" "$DELTA"
    ;;
  p1)
    load_cfg
    log "===== PHASE p1：baseline 门禁 ====="
    gate_baseline "cs_t0043" "$DELTA"
    if [ -n "${SPECS:-}" ]; then
      read -r -a P1_SPECS <<< "$SPECS"
    fi
    for spec in "${P1_SPECS[@]}"; do
      flag="${spec%%|*}"; val="${spec##*|}"
      printf '%s|%s|%s|%s|%s\n' "cs_t0043" "$DELTA" "p1__$(slug "$flag")__${val}" "$flag" "$val" >> "$JOBS_FILE"
    done
    log "===== PHASE p1：$(wc -l < "$JOBS_FILE") 次单旋钮 A/B（cs_t0043 Δ=$DELTA，-P $JOBS）====="
    xargs -P "$JOBS" -I{} bash "$SELF" --job "{}" < "$JOBS_FILE"
    ;;
  p2)
    load_cfg
    log "===== PHASE p2：baseline 门禁（rc2）====="
    gate_baseline "rc2" "$DELTA"
    [ -n "$P2_KNOBS" ] || { echo "[FATAL] P2_KNOBS 为空"; exit 3; }
    for spec in $P2_KNOBS; do
      flag="${spec%%=*}"; val="${spec##*=}"
      printf '%s|%s|%s|%s|%s\n' "rc2" "$DELTA" "p2__$(slug "$flag")__${val}" "$flag" "$val" >> "$JOBS_FILE"
    done
    log "===== PHASE p2：$(wc -l < "$JOBS_FILE") 次单旋钮 A/B（rc2 Δ=$DELTA，-P $JOBS）====="
    xargs -P "$JOBS" -I{} bash "$SELF" --job "{}" < "$JOBS_FILE"
    ;;
  p3)
    load_cfg
    log "===== PHASE p3：新机理候选（多旋钮组合）====="
    for spec in $P3_KNOBS; do
      tag="${spec%%=*}"; knobs="${spec#*=}"
      fl=""; vl=""
      IFS=',' read -r -a pairs <<< "$knobs"
      for pr in "${pairs[@]}"; do
        f="${pr%%=*}"; v="${pr##*=}"
        fl="${fl:+$fl,}$f"; vl="${vl:+$vl,}$v"
      done
      printf '%s|%s|%s|%s|%s\n' "cs_t0043" "$DELTA" "p3__${tag}" "$fl" "$vl" >> "$JOBS_FILE"
    done
    log "===== PHASE p3：$(wc -l < "$JOBS_FILE") 次候选（cs_t0043 Δ=$DELTA，-P $JOBS）====="
    xargs -P "$JOBS" -I{} bash "$SELF" --job "{}" < "$JOBS_FILE"
    ;;
  extra)
    # 通用补充扫描：EXTRA_SPECS 每项 = "model|delta|tag|flag=value[,flag=value]"
    # （单旋钮 ⇒ 只给一对；多旋钮组合 ⇒ 逗号分隔；tag 自带阶段前缀，便于报告归类）
    load_cfg
    log "===== PHASE extra：补充扫描 ====="
    for spec in $EXTRA_SPECS; do
      m="${spec%%|*}"; rest="${spec#*|}"
      d="${rest%%|*}"; rest="${rest#*|}"
      tag="${rest%%|*}"; knobs="${rest#*|}"
      fl=""; vl=""
      IFS=',' read -r -a pairs <<< "$knobs"
      for pr in "${pairs[@]}"; do
        f="${pr%%=*}"; v="${pr##*=}"
        fl="${fl:+$fl,}$f"; vl="${vl:+$vl,}$v"
      done
      printf '%s|%s|%s|%s|%s\n' "$m" "$d" "$tag" "$fl" "$vl" >> "$JOBS_FILE"
    done
    log "===== PHASE extra：$(wc -l < "$JOBS_FILE") 次补充扫描（-P $JOBS）====="
    xargs -P "$JOBS" -I{} bash "$SELF" --job "{}" < "$JOBS_FILE"
    ;;
  d24)
    load_cfg
    for d in 2 4; do
      log "===== PHASE d24：Δ=$d baseline 门禁 ====="
      gate_baseline "cs_t0043" "$d"
      if [ "$d" = "2" ] && [ -n "${D24_SPECS:-}" ]; then
        # D24_SPECS 每项 = "delta|value|tag"（可为同一个 Δ 配多个取值）
        for s in $D24_SPECS; do
          sd="${s%%|*}"; srest="${s#*|}"
          sv="${srest%%|*}"; stag="${srest##*|}"
          printf '%s|%s|%s|%s|%s\n' "cs_t0043" "$sd" "$stag" "$ROI_FLAG" "$sv" >> "$JOBS_FILE"
        done
      else
        printf '%s|%s|%s|%s|%s\n' "cs_t0043" "$d" "d${d}__$(slug "$ROI_FLAG")__${ROI_VAL}" "$ROI_FLAG" "$ROI_VAL" >> "$JOBS_FILE"
      fi
    done
    log "===== PHASE d24：Δ=2/4 候选验证（-P $JOBS）====="
    xargs -P "$JOBS" -I{} bash "$SELF" --job "{}" < "$JOBS_FILE"
    ;;
  gated)
    # 只跑 baseline 逐位门禁：GATE_LIST 每项 = "model|delta"
    load_cfg
    for s in $GATE_LIST; do
      gm="${s%%|*}"; gd="${s##*|}"
      log "===== PHASE gated：$gm Δ=$gd baseline 门禁 ====="
      gate_baseline "$gm" "$gd"
    done
    ;;
  all)
    log "===== PHASE all：gate → p1 → p2 → p3 → d24 ====="
    bash "$SELF" gate
    bash "$SELF" p1
    bash "$SELF" p2
    bash "$SELF" p3
    bash "$SELF" d24
    ;;
  *)
    echo "用法: $0 {gate|p1|p2|p3|d24|all}"; exit 2
    ;;
esac

log "PHASE $PHASE 结束。manifest: $MANIFEST"
