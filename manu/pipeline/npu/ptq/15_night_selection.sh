#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 15_night_selection.sh -- the two-axis MODEL SELECTION sweep, run on the server.
#
# WHY TWO AXES
# -------------
#   axis 1 (T3)  : 001324 2000 frames -- the HARD track, and the track already
#                   used for root-cause diagnosis.  COMPARATIVE axis.
#   axis 2 (T2)  : golden pool 1197 frames -- the SELECTION axis.  The blind
#                   track (T3, 11,810 frames) stays untouched until the final
#                   model runs the full three tracks exactly once.
#
# THE BOARD HAS A HARD WINDOW
# ---------------------------
#   The board reads frames from the workstation's NFS export
#   (/mnt/manu/ptq/bmp).  The host powers off at 18:30, after which the board
#   cannot read them (the tar-stream driver 14_run_full_scale_nofs.py takes
#   over from there).  So every board evaluation started here must finish
#   before the window closes: the loop checks the clock before each model and
#   stops starting new ones at --deadline (default 18:25).
#
# TWO AXES IN PARALLEL = TWO RKNN CONTEXTS
# ----------------------------------------
#   The board holds at most 2 RKNN contexts (a 3rd triggers an NPU soft
#   reset), so exactly two sequential loops run at a time -- one per axis.
#
# MODEL LIST
# ----------
#   Built from the candidate directories at run time (--model-dirs), so a model
#   that finished building late still gets evaluated.  Order matters only for
#   the clock guard: the models already validated on the board come first.
#
# USAGE
#   bash 15_night_selection.sh <out_dir> [deadline_HH:MM] [axis1] [axis2]
# ---------------------------------------------------------------------------
set -uo pipefail

REPO=/tmp/pycharm_project_10ae9e2e
OUT="${1:?out dir}"
DEADLINE="${2:-18:25}"
WAIT_UNTIL="${WAIT_UNTIL:-17:50}"   # give up waiting for the in-flight builds
AX1="${3:-/mnt/data/siping/ptq_work/fast/001324_sub2000.json}"
AX2="${4:-/mnt/data/siping/ptq_work/fast/golden_pool_sub1200.json}"
BOARD_MODELS=/mnt/manu/ptq/models
LOGS=/mnt/data/siping/ptq_work/logs
LIST="$OUT/model_list.txt"

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd "$REPO" || exit 1
mkdir -p "$OUT" "$LOGS"

# ---- wait for the in-flight builds, but not past the deadline --------------
echo "[wait] builds still running:"
while ps -eo args | grep -q '[0]2_batch_ptq_pc_screening.py --worker'; do
  now=$(date '+%H:%M')
  if [[ "$now" > "$WAIT_UNTIL" ]]; then echo "[wait] $WAIT_UNTIL reached ($now) -- proceed with what is built"; break; fi
  echo "[wait] $now  $(ps -eo args | grep -c '[0]2_batch_ptq_pc_screening.py --worker') workers"
  sleep 60
done

# ---- build the model list --------------------------------------------------
MODEL_DIRS=(
  /mnt/data/siping/ptq_work/screen_n
  /mnt/data/siping/ptq_work/screen_v2
  /mnt/data/siping/ptq_work/calib_ab
  /mnt/data/siping/ptq_work/calib_ab_lb
  /mnt/data/siping/ptq_work/screen_m
  /mnt/data/siping/ptq_work/screen_on
  /mnt/data/siping/ptq_work/screen
)
# WHITELIST: only the models that still need a BOARD verdict, and only as many
# as the 18:30 NFS window allows.  The board budget is 2 RKNN contexts and the
# board reads its frames over the workstation's NFS export, which dies with the
# host at 18:30 -- so the scan must be SHORT.  The fp16 arm and the int8
# set-A control are NOT re-run here: they already have board numbers on both
# axes from the same script, the same board and the same frames, so they are
# directly comparable:
#   p0_off_A_none   (f16)   T3 F1 0.7395 / T2 F1 0.9849
#   p0_off_A_normal (int8)  T3 F1 0.6448 / T2 F1 0.9420
#   p0_off_B/C/D_normal      T3 F1 0.5970 / 0.6407 / 0.6371
#   p0_off_lb_A_normal       T3 F1 0.6426 / T2 F1 0.9436
#   p0_off_{A,B,C}_kl        DEAD MODEL (output range clipped, 0 detections)
#   p0_off_A_mmse            ABANDONED (151 layers, ETA 4h08m)
WHITELIST="${WHITELIST:-p0_off_3_normal|p0_off_1_normal|p0_off_2_normal|p0_on_A_normal}"
# order: the fresh candidates first, then the controls
ORDER_RE='p0_off_3_normal|p0_off_1_normal|p0_off_2_normal|p0_on_A_normal|p0_off_A_mmse|p0_off_A_normal|p0_off_A_none'

: > "$LIST"
for d in "${MODEL_DIRS[@]}"; do
  [ -d "$d" ] || continue
  for f in "$d"/*.rknn; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    case "$b" in
      *kl_divergence*) continue ;;   # proven broken (output range clipped)
    esac
    # whitelist: the board budget is only ~2 contexts, so only the models
    # that still need a board verdict are allowed in
    if ! echo "$b" | grep -qE "($WHITELIST)"; then continue; fi
    echo "$f $b" >> "$LIST"
  done
done
sort -t/ -k9 "$LIST" | awk 'BEGIN{FS=" "} {print}' > "$LIST.sorted"
{
  grep -E "($ORDER_RE)" "$LIST.sorted" || true
  grep -vE "($ORDER_RE)" "$LIST.sorted" || true
} > "$LIST"
echo "[list] $(wc -l < "$LIST") models:"
cat "$LIST"

# ---- stage every candidate on the board (04 verifies the md5) ---------------
echo "[stage] pushing models to the board ..."
while read -r mp bn; do
  [ -z "${mp:-}" ] && continue
  [ -f "$mp" ] || continue
  if [ ! -f "$BOARD_MODELS/$bn" ]; then
    # </dev/null: scp reads stdin and would swallow the rest of the list
    scp -q "$mp" "ptq-board:$BOARD_MODELS/$bn" < /dev/null || echo "[stage] FAIL $bn"
  fi
done < "$LIST"
echo "[stage] done $(date '+%H:%M:%S')"

board_ctx_count() {
  # The board tolerates at most 2 RKNN contexts; a third one triggers an NPU
  # soft reset.
  #
  # COUNT BY `comm`, NOT BY `args`.  The board runs the demo through an
  # `ssh "bash -c '/root/rknn_heatmap_smoke ...'"` wrapper, so an
  # `ps -eo args | grep` count sees the wrapper AND the demo and reports 2 for
  # a single context -- which made this gate believe the board was full and
  # skip models.  `ps -eo comm` is the executable name (truncated to 15
  # chars: "rknn_heatmap_sm"), so it counts real contexts only.
  ssh -o ConnectTimeout=10 ptq-board \
    'ps -eo comm | grep -c "^rknn_heatmap"' < /dev/null 2>/dev/null \
    | tr -d ' \n' || echo 0
}

run_axis() {
  local axis="$1" manifest="$2"
  local stop=0
  while read -r mp bn; do
    [ -z "${mp:-}" ] && continue
    [ -f "$mp" ] || continue
    now=$(date '+%H:%M')
    if [[ "$now" > "$DEADLINE" ]]; then
      echo "[axis$axis] deadline $DEADLINE reached at $now -- stopping"
      break
    fi
    # Board context gate.
    #
    # The two axes run in parallel and each of them holds exactly ONE RKNN
    # context, so "2 contexts" is the NORMAL state, not an overload -- a third
    # one is what soft-resets the NPU.  So the gate waits while the board is
    # at 2 and only starts when it drops below 2; the model is then run.
    #
    # The earlier version gave up after 60 s and SKIPPED the model, which
    # silently dropped p0_on_A_normal from the sweep -- a candidate can
    # never be skipped, only deferred.  The wait is now unbounded (the
    # deadline above is the only exit) and a model is never dropped.
    while :; do
      ctx=$(board_ctx_count)
      ctx=${ctx:-0}
      if [ "$ctx" -lt 2 ]; then break; fi
      nw=$(date '+%H:%M')
      if [[ "$nw" > "$DEADLINE" ]]; then
        echo "[axis$axis] deadline $DEADLINE reached while waiting for a board slot -- stopping"
        stop=1; break
      fi
      echo "[axis$axis] board busy (ctx=$ctx) -- waiting for a free slot at $nw"
      sleep 20
    done
    [ "$stop" -eq 1 ] && break
    tag=$(basename "$mp" .rknn)
    echo "[axis$axis] $tag  start $now"
    # </dev/null is mandatory: the driver inherits stdin, and without this it
    # eats the rest of the model list that the `while read` loop is reading
    # from -- the axis then stops after the FIRST model.
    python -u manu/pipeline/npu/ptq/04_run_board_fast_eval.py \
      --models "$mp" \
      --manifest "$manifest" \
      --variant p0_off --out "$OUT" \
      --th 0.3 --topk 100 --board-models-dir "$BOARD_MODELS" \
      < /dev/null > "$LOGS/sel_axis${axis}_${tag}.log" 2>&1
    echo "[axis$axis] $tag rc=$? end $(date '+%H:%M:%S')"
    grep -E 'F1=|FATAL' "$LOGS/sel_axis${axis}_${tag}.log" | tail -2
  done < "$LIST"
  echo "[axis$axis] ALL DONE $(date '+%H:%M:%S')"
}

run_axis 2 "$AX2" &
AX2_PID=$!
sleep 15
run_axis 1 "$AX1" &
AX1_PID=$!
wait $AX1_PID
wait $AX2_PID
echo "[night_selection] both axes done $(date '+%H:%M:%S')"

python -u manu/pipeline/npu/ptq/16_collect_selection.py \
  --out "$OUT" --models "$LIST" --axis1 "$AX1" --axis2 "$AX2" \
  --pick-by axis2_then_axis1 > "$LOGS/selection_collect.log" 2>&1
echo "[night_selection] collect rc=$?"
tail -40 "$LOGS/selection_collect.log"
echo "[night_selection] FINISHED $(date '+%H:%M:%S')"
