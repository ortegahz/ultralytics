#!/usr/bin/env bash
# ============================================================
# ★ P3 新机理 · **出框门**（FOV margin gate）受控扫描
#
#   机理（8 序 cs_t0043 Δ=3 实测的 FAR 增量构成）：
#     replay 相对 original 的 FAR 增量 +0.003000（+87 FP）中
#       · 58% 是「框被推歪 12~136 px」（位移型）
#       · 41% 是「当前帧无检出时凭空上报」（幽灵型，**其中 4 个坐标为负、出画面**）
#     ⇒ 出框门：上报坐标必须落在画面内且距边界 ≥ m px，否则该帧不上报。
#       转台只能瞄画面里的东西 ⇒ 出框坐标必是误报。
#
#   ★ 新旋钮 `--replay-fov-margin-px`，**默认 0 = 关闭 = 逐位不变**。
#     只作用于 replay 臂（stale / kinematic 是对照基线，不加门）。
#
#   ★ 受控 A/B：唯一变量 = m（其余旋钮一字不改，CFG 仍取自 run_delivery.sh）。
#     门禁：m=0 的产物必须与既有 `runs/lookahead/fi_nogmc/<model>/d3/arms.json`
#           的四臂指标**逐位相等**（严格相等，无容差）；不一致 ⇒ 立刻停。
#
#   用法：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_fov_gate_scan.sh
#     env MODELS="cs_t0043 rc2" MARGINS="0 2 4 6 8 12 16" bash .../run_fov_gate_scan.sh
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
OUT="${OUT:-$REPO/runs/lookahead/fovgate}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"
REFROOT="${REFROOT:-$REPO/runs/lookahead/fi_nogmc}"
DELTA="${DELTA:-3}"
THREADS="${THREADS:-6}"

declare -A CACHE=(
  [cs_t0043]="${CACHE_CS:-/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache}"
  [rc2]="${CACHE_RC2:-$REPO/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache}"
)

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT/progress.txt"; }

# ---- 唯一 CFG 真值（多行续行 ⇒ awk 取整块再 eval）+ 防截断断言 ----------
CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
[ -n "$CFG_ASSIGN" ] || { echo "[FATAL] 无法解析 CFG"; exit 3; }
CFG=""; eval "$CFG_ASSIGN"
[ -n "$CFG" ] || { echo "[FATAL] CFG 解析为空"; exit 3; }
for _must in "--replay-roi-selfpred-max-px 16" "--replay-rollback-min-peak 0.90" \
             "--replay-polarity auto" "--replay-xcorr-uniq-gate 0.6" "--replay-mode perframe" \
             "--replay-abstain-peak-below 0.30" "--replay-rollback-max-frames 16"; do
  case " $CFG " in
    *" $_must "*) ;;
    *) echo "[FATAL] CFG 缺少必需旋钮 '$_must'"; exit 3 ;;
  esac
done

# 额外 CFG 覆盖（空格分隔的 "flag=value"，可空）—— 用于「B 旋钮串 × 出框门」组合扫描
EXTRA_CFG="${EXTRA_CFG:-}"
EXTRA_CFG="${EXTRA_CFG//--replay-fov-margin-px/}"   # 出框门单独走 --replay-fov-margin-px

MODELS="${MODELS:-cs_t0043 rc2}"
MARGINS="${MARGINS:-0 2 4 6 8 12 16}"

for m in $MODELS; do
  [ -n "${CACHE[$m]:-}" ] || { echo "[FATAL] 未知模型 $m"; exit 3; }
  for margin in $MARGINS; do
    tag="fov__m${margin}"
    o="$OUT/runs/$m/d$DELTA/$tag"
    mkdir -p "$o"
    if [ -s "$o/arms.json" ] && [ -f "$o/.done" ]; then
      log "SKIP $m Δ=$DELTA $tag（幂等）"; continue
    fi
    t0=$(date +%s)
    "$PY" manu/tracking/lookahead/lookahead_longquan.py \
      --cache-root "${CACHE[$m]}" --labels-root "$GT" --frames-root "$FRAMES" \
      --frames-cache "$FC" --params-json "$PARAMS" --all-seqs \
      --lookahead replay --arms original,stale,kinematic,replay --delta "$DELTA" \
      $CFG $EXTRA_CFG --replay-fov-margin-px "$margin" \
      --out-json "$o/arms.json" --out-preds-dir "$o/preds" > "$o/run.log" 2>&1
    rc=$?; t1=$(date +%s)
    [ $rc -eq 0 ] && touch "$o/.done" || echo "rc=$rc" > "$o/.rc"
    log "RUN $m Δ=$DELTA $tag rc=$rc 用时 $((t1-t0))s"
    [ $rc -ne 0 ] && tail -20 "$o/run.log"
  done
done

# ---- 门禁：m=0 必须与既有产物逐位一致 ----------------------------------
"$PY" - "$OUT" "$REFROOT" "$DELTA" <<'PY'
import json, sys
out, refroot, delta = sys.argv[1], sys.argv[2], int(sys.argv[3])
FIELDS = ("f1", "recall", "precision", "far_per_frame", "tp", "fp", "fn", "det", "gt")


def rows(path):
    doc = json.load(open(path))
    out = {}
    for arm in ("original", "stale", "kinematic", "replay"):
        a = doc["arms"][arm]
        cur = a["current_frame"]["overall"]
        srv = a["servo"]["overall"]
        out[arm] = {
            **{k: cur.get(k) for k in FIELDS},
            "hit5": srv.get("hit_rate_5px"),
            "cle50": srv.get("cle_pooled_p50"),
            "cle95": srv.get("cle_pooled_p95"),
            "cle99": srv.get("cle_pooled_p99"),
            "clemax": srv.get("cle_pooled_max"),
            "samples": srv.get("cle_pooled_samples"),
            "source_f1": (a.get("source_frame") or {}).get("overall", {}).get("f1"),
            "source_far": (a.get("source_frame") or {}).get("overall", {}).get("far_per_frame"),
        }
    return out


bad = 0
for model in ("cs_t0043", "rc2"):
    new = f"{out}/runs/{model}/d{delta}/fov__m0/arms.json"
    ref = f"{refroot}/{model}/d{delta}/arms.json"
    try:
        rn, rr = rows(new), rows(ref)
    except FileNotFoundError:
        print(f"[GATE] {model}: 产物缺失（跳过）")
        continue
    diff = [f"{arm}.{k}: new={rn[arm][k]!r} ref={rr[arm][k]!r}"
            for arm in rn for k in rn[arm] if rn[arm][k] != rr[arm][k]]
    if diff:
        bad += 1
        print(f"[GATE][FAIL] {model} m=0 与既有产物不一致（{len(diff)} 处）：")
        for d in diff[:8]:
            print("   ", d)
    else:
        print(f"[GATE][PASS] {model} m=0 四臂逐位一致（{len(rn) * len(rn['replay'])} 字段）")
sys.exit(3 if bad else 0)
PY
gate_rc=$?
log "门禁 rc=$gate_rc"
exit $gate_rc
