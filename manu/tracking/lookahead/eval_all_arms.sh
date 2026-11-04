#!/usr/bin/env bash
# ============================================================
# 一键出对照表：参考产物 → Gate A → 对照臂(perframe/rolling/refine) → 出表
#
# 六个步骤：
#   ① 参考产物：跑原脚本 manu/evaluation/system_sota_longquan.py（只读，不改）
#   ② Gate A  ：外挂关闭(--lookahead off --arms original) 必须与①逐位一致
#   ③ 对照臂·perframe：--replay-mode perframe（已调优口径：holdv/anchor=delta）
#   ④ 对照臂·rolling ：--replay-mode rolling（已判负的对照臂，按当前实现重跑）
#   ⑤ 对照臂·refine  ：--replay-mode refine（零漂移档位：绝对位置锚管线，快核只修正一步）
#   ⑥ 出表    ：make_report_table.py → $OUT/comparison.md
#
# 所有路径/参数都能用环境变量覆盖：
#   CACHE=... GT=... FRAMES=... FRAMES_CACHE=... PY=... PARAMS=... OUT=... \
#   DELTA=8 CONSISTENCY_GATE_PX=0 REFINE_GATE_PX=24 ROLLING_EXTRA="--rolling-max-step-px 25" \
#   bash manu/tracking/lookahead/eval_all_arms.sh [--dry-run]
#
# --dry-run：只打印每一步的完整命令，不真跑（Gate A 逐文件比对随之跳过）。
# 只跑评测与出表：不训练、不推理、不重建帧缓存。
# ============================================================
set -uo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h | --help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg（只支持 --dry-run / --help）" >&2; exit 2 ;;
  esac
done

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

CACHE="${CACHE:-/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache}"
GT="${GT:-/home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010}"
FRAMES="${FRAMES:-/home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
FRAMES_CACHE="${FRAMES_CACHE:-runs/lookahead/frame_cache}"
PY="${PY:-/media/manu/1TB-Volume/workspace/.conda-envs/lqcache/bin/python}"
PARAMS="${PARAMS:-manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json}"
OUT="${OUT:-runs/lookahead/final}"
DELTA="${DELTA:-8}"
CONSISTENCY_GATE_PX="${CONSISTENCY_GATE_PX:-0}"   # 一致性门（px）：重推结果偏离『锚点+Δ·v』超门 ⇒ 改用运动学预测；0=关
REFINE_GATE_PX="${REFINE_GATE_PX:-24}"             # refine 档位：快核单步修正上限（px）
ROLLING_EXTRA="${ROLLING_EXTRA:-}"                 # 追加给 rolling 臂的整串 flag（空白分隔）
OUTPUT_MAX_STEP_PX="${OUTPUT_MAX_STEP_PX:-30}"          # 输出级限幅（伺服 slew limiter）：每帧上报位移上限（px）；0=不限幅

mkdir -p "$OUT"
FAILED=()

# replay 三个档位共用的口径：帧源（本地 memmap 优先，省掉逐帧 sshfs 读图）、Δ、快核 ROI、一致性门
REPLAY_COMMON=(
  --cache-root "$CACHE" --labels-root "$GT" --all-seqs --params-json "$PARAMS"
  --frames-root "$FRAMES" --frames-cache "$FRAMES_CACHE"
  --lookahead replay --delta "$DELTA" --arms original,stale,kinematic,replay
  --fast-win-size 48 --fast-bg-window 96
  --replay-consistency-gate-px "$CONSISTENCY_GATE_PX"
)

# ROLLING_EXTRA 按空白切成数组（空串 ⇒ 不追加任何参数，避免传空参给 argparse）
ROLLING_EXTRA_ARGS=()
if [ -n "$ROLLING_EXTRA" ]; then
  read -r -a ROLLING_EXTRA_ARGS <<< "$ROLLING_EXTRA"
fi

# 步骤包装：横幅走 stderr（数据走 stdout，便于把某步输出单独重定向到文件）
run(){  # run [--quiet] <步骤名> <命令...>
  local quiet=0
  if [ "${1:-}" = "--quiet" ]; then quiet=1; shift; fi
  local name="$1"; shift
  echo >&2
  echo >&2 ">>> [$(date '+%H:%M:%S')] $name"
  if [ "$DRY_RUN" -eq 1 ]; then
    local rendered=""
    for word in "$@"; do rendered+=" $(printf '%q' "$word")"; done
    printf '[DRY-RUN] %s\n' "$name"
    printf '  %s\n' "${rendered# }"
    return 0
  fi
  if [ "$quiet" -eq 1 ]; then
    "$@" > /dev/null
  else
    "$@"
  fi
  local rc=$?
  echo >&2 "<<< [$(date '+%H:%M:%S')] $name rc=$rc"
  if [ "$rc" -ne 0 ]; then FAILED+=("$name"); fi
  return 0
}

# ---------- 0. 环境摘要 ----------
NPKL=$(find "$CACHE" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
NNPY=$(find "$FRAMES_CACHE" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
cat <<EOF
=== 环境摘要 ===
  缓存(ms_cache) : $CACHE   [$NPKL 个 pkl]
  帧缓存(memmap) : $FRAMES_CACHE   [$NNPY 个 npy]
  帧根           : $FRAMES
  标注根         : $GT
  Δ              : $DELTA 帧 ($((DELTA * 40))ms @25Hz)
  解释器         : $PY
  参数文件       : $PARAMS
  产物根         : $OUT
  一致性门       : $CONSISTENCY_GATE_PX px（0=关）
  refine 修正门  : $REFINE_GATE_PX px
EOF
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  模式         : DRY-RUN（只打印命令，不执行）"
fi
if [ "$NPKL" -eq 0 ]; then
  echo "  !! 警告：缓存里没有 pkl，评测会跑空 —— 用 CACHE= 指向正确的 ms_cache 目录"
fi
if [ "$NNPY" -eq 0 ]; then
  echo "  !! 警告：帧缓存里没有 npy，replay 臂会退回逐帧 sshfs 读图（慢 ~100×）—— 先跑 build_frame_cache.py"
fi
if [ ! -f "$PARAMS" ]; then
  echo "  !! 警告：参数文件不存在：$PARAMS"
fi
if [ ! -x "$PY" ]; then
  echo "  !! 警告：解释器不可执行：$PY —— 用 PY= 指向可用的 python"
fi

# ---------- ① 参考产物（原脚本，只读不改） ----------
run "① 参考产物（原脚本 system_sota_longquan.py）" \
  "$PY" manu/evaluation/system_sota_longquan.py \
    --cache-root "$CACHE" --labels-root "$GT" --all-seqs --params-json "$PARAMS" \
    --out-json "$OUT/ref_original.json" --out-preds-dir "$OUT/preds_ref_original"

# ---------- ② Gate A：外挂关闭必须与①逐位一致 ----------
run "② Gate A（外挂关闭 --lookahead off --arms original）" \
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    --cache-root "$CACHE" --labels-root "$GT" --all-seqs --params-json "$PARAMS" \
    --lookahead off --arms original --delta "$DELTA" \
    --out-json "$OUT/gateA_off.json" --out-preds-dir "$OUT/preds_gateA_off"

IDENT=0
DIFF=0
if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "=== Gate A 逐文件比对：dry-run 跳过（①② 未真实执行，没有产物可比） ==="
else
  echo
  echo "=== Gate A 逐文件比对（$OUT/preds_ref_original vs $OUT/preds_gateA_off） ==="
  for f in "$OUT"/preds_ref_original/*.json; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    g="$OUT/preds_gateA_off/$b"
    if [ ! -f "$g" ]; then
      echo "  MISSING $b"
      DIFF=$((DIFF + 1))
    elif cmp -s "$f" "$g"; then
      IDENT=$((IDENT + 1))
    else
      echo "  DIFFER $b"
      DIFF=$((DIFF + 1))
    fi
  done
  for g in "$OUT"/preds_gateA_off/*.json; do
    [ -e "$g" ] || continue
    b=$(basename "$g")
    if [ ! -f "$OUT/preds_ref_original/$b" ]; then
      echo "  EXTRA $b"
      DIFF=$((DIFF + 1))
    fi
  done
  echo "GATE_A: IDENTICAL=$IDENT DIFFER=$DIFF"
  echo "$DIFF" > "$OUT/gateA_diff_count.txt"
  if [ "$DIFF" -ne 0 ]; then
    echo "!! Gate A 未通过：外挂关闭时输出与原脚本不逐位一致，后续对照不可信 —— 中止"
    exit 1
  fi
  if [ "$((IDENT + DIFF))" -eq 0 ]; then
    echo "!! Gate A 无法比对：参考产物目录里没有可比的 json（① 步骤多半失败了）—— 中止"
    exit 1
  fi
fi

# ---------- ③ 对照臂：perframe / rolling / refine 各一趟 ----------
# perframe：已调优口径（hold 预算 8 + holdv 兜底 + 锚点取管线位移 + hold 不走稳健参考）
run "③ 对照臂·perframe（每物理帧从源帧 t-Δ 重新起链重推 Δ 步）" \
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    "${REPLAY_COMMON[@]}" \
    --replay-mode perframe \
    --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta \
    --no-replay-hold-use-ref --output-max-step-px 0 \
    --out-json "$OUT/arms_perframe.json" --out-preds-dir "$OUT/preds_perframe"

# perframe + 输出级限幅：**交付候选**。限幅把每帧上报位移钉在上限内 ⇒ 转台看到的
# 帧间位移有界 ⇒ 二阶差分（max_jitter）自动有界；实测 014053 抖动 331 px → 40 px。
run "③b 对照臂·perframe+输出级限幅（交付候选）" \
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    "${REPLAY_COMMON[@]}" \
    --replay-mode perframe \
    --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta \
    --no-replay-hold-use-ref --output-max-step-px "$OUTPUT_MAX_STEP_PX" \
    --out-json "$OUT/arms_perframe_limit.json" --out-preds-dir "$OUT/preds_perframe_limit"

# rolling：已判负的对照臂；产物用当前修好 bug 后的实现重跑（不复用旧炸掉的 arms_rolling.json）
run "④ 对照臂·rolling（一条链跨物理帧持续推进）" \
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    "${REPLAY_COMMON[@]}" \
    --replay-mode rolling --rolling-recenter 1 --rolling-hold-mode vel \
    --replay-max-hold 8 --replay-fallback holdc \
    "${ROLLING_EXTRA_ARGS[@]}" \
    --out-json "$OUT/arms_rolling.json" --out-preds-dir "$OUT/preds_rolling"

# refine：绝对位置永远锚在管线上，快核只做一次单步局部修正（零漂移、不拒帧）
run "⑤ 对照臂·refine（快核单步局部修正）" \
  "$PY" manu/tracking/lookahead/lookahead_longquan.py \
    "${REPLAY_COMMON[@]}" \
    --replay-mode refine --refine-gate-px "$REFINE_GATE_PX" \
    --output-max-step-px "$OUTPUT_MAX_STEP_PX" \
    --out-json "$OUT/arms_refine.json" --out-preds-dir "$OUT/preds_refine"

# ---------- ⑥ 出表 ----------
run --quiet "⑥ 出对照表（make_report_table.py）" \
  "$PY" manu/tracking/lookahead/make_report_table.py \
    --runs "$OUT/ref_original.json" "$OUT/arms_perframe.json" \
            "$OUT/arms_perframe_limit.json" \
            "$OUT/arms_rolling.json" "$OUT/arms_refine.json" \
    --labels "原代码(同帧口径)" "外挂·perframe" "外挂·perframe+限幅" \
             "外挂·rolling" "外挂·refine" \
    --delta "$DELTA" --per-sequence --out-md "$OUT/comparison.md"

echo
if [ -f "$OUT/comparison.md" ]; then
  cat "$OUT/comparison.md"
elif [ "$DRY_RUN" -eq 1 ]; then
  echo "(dry-run：未真实执行，对照表未生成：$OUT/comparison.md)"
else
  echo "!! 对照表没生成：$OUT/comparison.md"
fi

# ---------- 汇总 ----------
echo
echo "=== 汇总 ==="
echo "  产物根   : $OUT"
echo "  参考产物 : $OUT/ref_original.json"
echo "  Gate A   : $OUT/gateA_off.json  (IDENTICAL=$IDENT DIFFER=$DIFF，差异数写在 $OUT/gateA_diff_count.txt)"
echo "  perframe : $OUT/arms_perframe.json"
echo "  rolling  : $OUT/arms_rolling.json"
echo "  refine   : $OUT/arms_refine.json"
echo "  对照表   : $OUT/comparison.md"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  DRY-RUN  : 以上命令均未执行"
  exit 0
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "  失败步骤 : ${FAILED[*]}"
  exit 1
fi
echo "  全部步骤成功"
