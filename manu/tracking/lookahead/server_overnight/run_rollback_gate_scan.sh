#!/usr/bin/env bash
# ============================================================
# ★ 夜 8 · **回退链峰值置信门**（`--replay-rollback-min-peak`）受控 A/B
#
#   机理（24 序 cs_t0043 Δ=3 离线归因，见 runs/lookahead/fi_nogmc24/reg_oracle/）：
#     「有配准决策但没上报」的静默帧里，最大的一块是**回退链峰值门**：
#       回退链的模板来自更早的帧（source_frame != target_frame − Δ）⇒
#       目标可能已飞走、配准可能锁到背景结构（"幽灵坐标"）⇒ 峰值 < 0.90 时弃权。
#     该门当前把 **2025 帧**整帧丢掉（回退 + 峰值不达标）。
#     记忆里的历史实测（8 序）：放松该门 ⇒ F1↑ 但 FAR↑（符号在 cs_t0043 上完全一致）。
#     24 序 FAR 红线余量只有 0.005663 ⇒ 必须在**官方 24 序口径**上重新量。
#
#   ★ 受控 A/B：**唯一变量 = `--replay-rollback-min-peak`**，
#     其余旋钮逐字取自 run_delivery.sh（awk 取整块再 eval + 7 条防截断断言），
#     外加交付档已选的出框门 `--replay-fov-margin-px 6`。
#
#   ★ 门禁（两条，任一不过 ⇒ 产物作废）：
#     ① rb=0.90 必须与既有交付档产物 `runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/
#        fov__m6/arms.json` **逐位相等**（证明 CFG 管线与交付档完全同源）；
#     ② 所有变体的 original / stale / kinematic 三臂必须**逐位相等**
#        （该旋钮只作用于 replay 臂 ⇒ 变了就说明 A/B 不受控）。
#
#   用法：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/run_rollback_gate_scan.sh
#     env RB_PEAKS="0.0" OUT=... bash .../run_rollback_gate_scan.sh
#
#   ★ 幂等：已跑完（.done）的变体跳过。
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
GT="${GT:-/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FC="${FRAMES_CACHE:-/mnt/data/siping/frame_cache_lq}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
OUT="${OUT:-$REPO/runs/lookahead/rbgate24nogmc}"
DELIVERY_SH="$REPO/manu/tracking/lookahead/run_delivery.sh"
DELTA="${DELTA:-3}"
FOV_MARGIN="${FOV_MARGIN:-6}"
RB_PEAKS="${RB_PEAKS:-0.90 0.80 0.70 0.50 0.0}"
RUN_GATE="${RUN_GATE:-1}"
THREADS="${THREADS:-6}"
CACHE_CS="${CACHE_CS:-/mnt/data/siping/runs/nogmc_t3/swa/ms_cache_24seq}"
REF_M6="${REF_M6:-$REPO/runs/lookahead/fovgate24nogmc/runs/cs_t0043/d3/fov__m6/arms.json}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$OUT"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT/progress.txt"; }

# ---- 唯一 CFG 真值（多行续行 ⇒ awk 取整块再 eval）+ 防截断断言 ----------
CFG_ASSIGN="$(awk '/^CFG="/ {f=1} f {print} f && /"[[:space:]]*$/ {exit}' "$DELIVERY_SH")"
[ -n "$CFG_ASSIGN" ] || { log "[FATAL] 无法解析 $DELIVERY_SH 的 CFG"; exit 3; }
CFG=""; eval "$CFG_ASSIGN"
[ -n "$CFG" ] || { log "[FATAL] CFG 解析为空"; exit 3; }
for _must in "--replay-roi-selfpred-max-px 16" "--replay-rollback-min-peak 0.90" \
             "--replay-polarity auto" "--replay-xcorr-uniq-gate 0.6" "--replay-mode perframe" \
             "--replay-abstain-peak-below 0.30" "--replay-rollback-max-frames 16"; do
  case " $CFG " in
    *" $_must "*) ;;
    *) log "[FATAL] CFG 缺少必需旋钮 '$_must' ⇒ 解析被截断，拒绝跑"; exit 3 ;;
  esac
done
# 把交付档里的该旋钮**摘掉**，再由本次扫描显式给出 ⇒ 保证「唯一变量」逐字成立
CFG="${CFG//--replay-rollback-min-peak 0.90/}"
log "CFG（源自 run_delivery.sh，${#CFG} 字符，已摘掉 rollback-min-peak）: $CFG"

NPKL=$(ls "$CACHE_CS"/*.pkl 2>/dev/null | wc -l)
[ "$NPKL" -eq 24 ] || { log "[FATAL] 24 序缓存 pkl 数=$NPKL ≠ 24"; exit 3; }
log "24 序 no-GMC 缓存 $CACHE_CS pkl=$NPKL；出框门 margin=$FOV_MARGIN；Δ=$DELTA"

for rb in $RB_PEAKS; do
  tag="rb__${rb}"
  o="$OUT/runs/cs_t0043/d$DELTA/$tag"
  mkdir -p "$o"
  if [ -s "$o/arms.json" ] && [ -f "$o/.done" ]; then
    log "SKIP $tag（幂等）"; continue
  fi
  t0=$(date +%s)
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    --cache-root "$CACHE_CS" --labels-root "$GT" --frames-root "$FRAMES" \
    --frames-cache "$FC" --params-json "$PARAMS" --all-seqs \
    --lookahead replay --arms original,stale,kinematic,replay --delta "$DELTA" \
    $CFG --replay-fov-margin-px "$FOV_MARGIN" --replay-rollback-min-peak "$rb" \
    --out-json "$o/arms.json" --out-preds-dir "$o/preds" > "$o/run.log" 2>&1
  rc=$?; t1=$(date +%s)
  [ $rc -eq 0 ] && touch "$o/.done" || echo "rc=$rc" > "$o/.rc"
  log "RUN rb__$rb rc=$rc 用时 $((t1-t0))s"
  [ $rc -ne 0 ] && tail -20 "$o/run.log"
done

# ---- 门禁：① rb=0.90 与交付档逐位一致  ② 三条对照臂跨变体逐位一致 ----
[ "$RUN_GATE" = "1" ] || { log "RUN_GATE=0 ⇒ 跳过门禁（并行分片时用）"; exit 0; }
"$PY" - "$OUT" "$DELTA" "$REF_M6" <<'PY'
import json, os, sys
out, delta, ref_m6 = sys.argv[1], sys.argv[2], sys.argv[3]
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
tags = sorted(
    d for d in os.listdir(f"{out}/runs/cs_t0043/d{delta}")
    if os.path.isdir(f"{out}/runs/cs_t0043/d{delta}/{d}")
)
paths = {t: f"{out}/runs/cs_t0043/d{delta}/{t}/arms.json" for t in tags}
paths = {t: p for t, p in paths.items() if os.path.exists(p)}

# 门禁 ①
base_tag = "rb__0.90"
if base_tag in paths and os.path.exists(ref_m6):
    a, b = rows(paths[base_tag]), rows(ref_m6)
    diff = [f"{arm}.{k}: new={a[arm][k]!r} ref={b[arm][k]!r}"
            for arm in a for k in a[arm] if a[arm][k] != b[arm][k]]
    if diff:
        bad += 1
        print(f"[GATE1][FAIL] {base_tag} 与交付档 m6 产物不一致（{len(diff)} 处）：")
        for d in diff[:8]:
            print("   ", d)
    else:
        print(f"[GATE1][PASS] {base_tag} 与交付档 m6 产物四臂逐位一致（{len(a) * len(a['replay'])} 字段）")
else:
    print(f"[GATE1][SKIP] 缺 {base_tag} 或参照产物")

# 门禁 ②
if base_tag in paths:
    base = rows(paths[base_tag])
    for tag in tags:
        if tag == base_tag:
            continue
        cur = rows(paths[tag])
        diff = [f"{arm}.{k}: {tag}={cur[arm][k]!r} base={base[arm][k]!r}"
                for arm in ("original", "stale", "kinematic") for k in base[arm]
                if cur[arm][k] != base[arm][k]]
        if diff:
            bad += 1
            print(f"[GATE2][FAIL] {tag} 的对照臂与基线不一致（{len(diff)} 处）：")
            for d in diff[:6]:
                print("   ", d)
        else:
            print(f"[GATE2][PASS] {tag} 的 original/stale/kinematic 三臂逐位一致")
sys.exit(3 if bad else 0)
PY
gate_rc=$?
log "门禁 rc=$gate_rc"
exit $gate_rc
