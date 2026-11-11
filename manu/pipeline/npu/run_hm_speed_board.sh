#!/usr/bin/env bash
# ===========================================================================
# run_hm_speed_board.sh -- the **board speed pipeline** for the no-GMC struct
#                           search line (cs_t0043 -> delete P2 fusion + all ReLU).
#
# WHAT IT DOES (one command, one packed board session)
#   [PREPARE]  arm ckpt -> hm_export_onnx.py -> hm_convert_rknn.py (int8)
#              (the ckpt is **self-describing**: no_p2 / head_level / act /
#               head_conv / late_upsample are read from the ckpt, so the
#               exported graph IS the trained structure -- zero extra flags)
#   [STAGE]    scp the .rknn arms onto the board's **own local disk**
#   [MEASURE]  ONE ssh session on the board:
#                0. health check (3 NPU frames)
#                1. CONTROL  = cs_t0043 same-source base arm  -> the baseline
#                2. every new arm
#              each via the board's heatmap-contract smoke tool
#              (rknn_heatmap_smoke) -> the `rknn_run avg` ms line is tabulated.
#
# WHY THE CONTROL IS MEASURED EVERY TIME
#   The RK3588 RKNPU driver drifts between days, so "55.5 ms" from a previous
#   session is NOT a valid comparison.  Every session re-measures the control
#   (cs_t0043 same-source base) so the arm-vs-baseline delta is same-session.
#
# HARD CONSTRAINTS (learned the hard way -- do not violate)
#   * The board's /usr/lib/librknnrt.so (md5 9cb5f17b...) SIGILLs inside
#     rknn_run (exit 132).  The known-good copy is md5 56e425dc... and must be
#     FIRST on LD_LIBRARY_PATH.
#   * NEVER call rknn_set_core_mask: verified to wedge the driver directly.
#   * The driver is fragile: repeated long tasks -> "job timeout" -> soft
#     reset -> rknn_init returns 132 forever -> **physical power cycle**.
#     Hence: ONE board invocation, settle pause + health check before EVERY model.
#   * exit 132 with the good lib already first => the NPU is wedged => power cycle.
#
# USAGE
#   # measure the arms that are already converted to .rknn
#   ARMS_SRC=/path/to/rknn bash manu/pipeline/npu/run_hm_speed_board.sh
#
#   # full pipeline: ckpts -> onnx -> rknn -> board
#   CKPT_DIR=/mnt/data/siping/runs/nogmc_struct/trials/trial_0007 \
#     bash manu/pipeline/npu/run_hm_speed_board.sh
#
#   bash manu/pipeline/npu/run_hm_speed_board.sh --dry-run
#
# ENV
#   ARMS_SRC     dir of .rknn arms           (default $OUT_ROOT/rknn_arms)
#   CKPT_DIR     dir of arm .ckpt to convert  (optional)
#   OUT_ROOT     everything lands here       (default /mnt/data/siping/runs/nogmc_struct/npu_speed)
#   SSH_KEY      board key (server: ~/.ssh/id_board ; host: ~/.ssh/id_ed25519)
#   LOOPS/WARMUP timing loops per model      (default 30/10)
# ===========================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

BOARD="${RKNPU_BOARD:-root@192.168.0.211}"
BOARD_PORT="${RKNPU_BOARD_PORT:-22}"
BOARD_DIR="${RKNPU_BOARD_DIR:-/root/hm_speed}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_board}"

OUT_ROOT="${OUT_ROOT:-/mnt/data/siping/runs/nogmc_struct/npu_speed}"
ARMS_SRC="${ARMS_SRC:-$OUT_ROOT/rknn_arms}"
CKPT_DIR="${CKPT_DIR:-}"

CONTROL_NAME="${HM_CONTROL_NAME:-base_int8.rknn}"
GOOD_MD5_EXPECTED="56e425dcc7b59ab0e1aec685b178b99f"
GOOD_LIB="${HM_GOOD_LIB:-/run/media/manu/SSD960-LOCAL/nfs/npu/librknnrt.so}"

# no-GMC feature dataset (calibration source -- the GMC default root would
# quantise the no-GMC model against the WRONG feature distribution)
NOGMC_UAV_IMAGES="${NOGMC_UAV_IMAGES:-/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/images/train}"
CALIB_N="${CALIB_N:-200}"

REF_BASE_MS="${HM_REF_BASE_MS:-55.5}"
LOOPS="${HM_LOOPS:-30}"
WARMUP="${HM_WARMUP:-10}"
SETTLE="${HM_SETTLE:-5}"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
SCP_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
[ -f "$SSH_KEY" ] && { SSH_OPTS="$SSH_OPTS -i $SSH_KEY"; SCP_OPTS="$SCP_OPTS -i $SSH_KEY"; }

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

say() { echo "[run_hm_speed_board] $*"; }
die() { echo "[run_hm_speed_board][ERROR] $*" >&2; exit 1; }

mkdir -p "$OUT_ROOT" "$ARMS_SRC" "$OUT_ROOT/logs"

# ===========================================================================
# STEP 0 -- board reachable + known-good runtime present
# ===========================================================================
say "STEP 0: board reachability (key=$SSH_KEY)"
ssh $SSH_OPTS -p "$BOARD_PORT" "$BOARD" "echo ok" >/dev/null 2>&1 \
  || die "cannot ssh to $BOARD with key $SSH_KEY"
BOARD_LIB_MD5=$(ssh $SSH_OPTS -p "$BOARD_PORT" "$BOARD" \
  "md5sum $BOARD_DIR/lib/librknnrt.so 2>/dev/null | awk '{print \$1}'" 2>/dev/null | tr -d '\r')
if [ "$BOARD_LIB_MD5" = "$GOOD_MD5_EXPECTED" ]; then
  say "  board already carries the known-good runtime (md5=$BOARD_LIB_MD5)"
elif [ -f "$GOOD_LIB" ]; then
  say "  staging known-good runtime from $GOOD_LIB"
  scp $SCP_OPTS -P "$BOARD_PORT" "$GOOD_LIB" "$BOARD:$BOARD_DIR/lib/librknnrt.so" || die "scp lib failed"
else
  die "board has no known-good runtime and $GOOD_LIB is not reachable from here"
fi

# ===========================================================================
# STEP 1 -- PREPARE: ckpt -> onnx -> rknn(int8)
# ===========================================================================
if [ -n "$CKPT_DIR" ]; then
  say "STEP 1: PREPARE ckpt -> onnx -> rknn from $CKPT_DIR"
  [ -d "$CKPT_DIR" ] || die "CKPT_DIR does not exist: $CKPT_DIR"
  CALIB_LIST="$OUT_ROOT/calib_nogmc_train.txt"
  if [ ! -f "$CALIB_LIST" ]; then
    say "  building calibration list (no-GMC features, n=$CALIB_N)"
    find "$NOGMC_UAV_IMAGES" -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' \) \
      | sort | head -"$CALIB_N" > "$CALIB_LIST" || die "cannot build the calibration list"
  fi
  say "  calibration list: $CALIB_LIST ($(wc -l < "$CALIB_LIST") images)"
  for ck in "$CKPT_DIR"/*.pt; do
    [ -e "$ck" ] || continue
    # 名字带上 trial 目录名（trial_0007/epoch20.pt -> trial_0007_epoch20）
    # ⇒ 板卡结果里的 arm 名能直接对回验收表的 trial 号
    name="$(basename "$(dirname "$ck")")_$(basename "$ck" .pt)"
    onnx="$OUT_ROOT/onnx/$name.onnx"
    rknn="$ARMS_SRC/${name}_int8.rknn"
    mkdir -p "$OUT_ROOT/onnx"
    say "  --- $name"
    if [ ! -f "$onnx" ]; then
      python -u "$HERE/hm_export_onnx.py" --weights "$ck" --out "$onnx" --skip-check \
        > "$OUT_ROOT/logs/export_$name.log" 2>&1 || die "export failed: $ck (see $OUT_ROOT/logs/export_$name.log)"
    fi
    grep -E "^\[STRUCT\]|^\[model\] stride" "$OUT_ROOT/logs/export_$name.log" 2>/dev/null | sed 's/^/      /'
    if [ ! -f "$rknn" ]; then
      python -u "$HERE/hm_convert_rknn.py" --onnx "$onnx" --out "$rknn" \
        --calib-list "$CALIB_LIST" --calib-list-only \
        > "$OUT_ROOT/logs/convert_$name.log" 2>&1 || die "convert failed: $onnx (see $OUT_ROOT/logs/convert_$name.log)"
    fi
    # ★ 边车文件：记下该臂的输出 stride ⇒ 板卡工具的 --stride 才能对上
    #   （no_p2+p3 ⇒ 8；no_p2+p2 ⇒ 4；原结构 ⇒ 2）
    _st=$(grep -oE "^\[model\] stride=[0-9]+" "$OUT_ROOT/logs/export_$name.log" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
    [ -n "$_st" ] && echo "$_st" > "$ARMS_SRC/${name}_int8.stride"
    say "      -> $rknn ($(stat -c %s "$rknn") bytes)  stride=${_st:-?}"
  done
else
  say "STEP 1: PREPARE skipped (no CKPT_DIR)"
fi

# ===========================================================================
# STEP 2 -- stage the arms onto the board's local disk
# ===========================================================================
say "STEP 2: staging arms from $ARMS_SRC"
N_ARMS=0
ARM_FILES=()
if [ -d "$ARMS_SRC" ]; then
  for a in "$ARMS_SRC"/*.rknn; do
    [ -e "$a" ] || continue
    ARM_FILES+=("$a"); N_ARMS=$((N_ARMS + 1))
  done
fi
say "  found $N_ARMS arm(s)"
if [ "$DRY_RUN" != "1" ] && [ "$N_ARMS" -gt 0 ]; then
  ssh $SSH_OPTS -p "$BOARD_PORT" "$BOARD" "mkdir -p $BOARD_DIR/arms && rm -f $BOARD_DIR/arms/*" \
    || die "cannot prepare $BOARD_DIR/arms"
  for a in "${ARM_FILES[@]}"; do
    scp $SCP_OPTS -P "$BOARD_PORT" "$a" "$BOARD:$BOARD_DIR/arms/" || die "scp failed: $a"
    # ★ 边车文件（stride）必须一起传：板卡侧的 --stride 从**板卡本地**读。
    #   命名约定：<arm>.rknn ⇒ <arm>.stride（**替换** .rknn 后缀，不是追加）
    _sd="${a%.rknn}.stride"
    if [ -f "$_sd" ]; then
      scp $SCP_OPTS -P "$BOARD_PORT" "$_sd" "$BOARD:$BOARD_DIR/arms/" || die "scp stride failed: $_sd"
    fi
    say "    staged $(basename "$a")  stride=$(cat "${a%.rknn}.stride" 2>/dev/null || echo '?')"
  done
fi

# ===========================================================================
# STEP 3 -- board-side runner
# ===========================================================================
MEASURE_SH="$OUT_ROOT/measure_hm_speed_board.sh"
cat > "$MEASURE_SH" <<'MEASURE_EOF'
#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# measure_hm_speed_board.sh -- RUNS ON THE BOARD.
# Deployed by run_hm_speed_board.sh; do not run it by hand.
# ---------------------------------------------------------------------------
set -uo pipefail
DIR="${BOARD_DIR:-/root/hm_speed}"
RES="$DIR/hm_speed_results.txt"
IMG="$DIR/feat.jpg"
CONTROL="$DIR/${CONTROL_NAME:-base_int8.rknn}"
LOOPS="${LOOPS:-30}"; WARMUP="${WARMUP:-10}"; SETTLE="${SETTLE:-5}"

# known-good runtime FIRST (the board's /usr/lib copy SIGILLs inside rknn_run)
export LD_LIBRARY_PATH="$DIR/lib:/mnt/manu/npu:/usr/lib"

: > "$RES"
log() { printf '%s\n' "$*" | tee -a "$RES"; }

say_lib() {
  log "librknnrt in use  : $DIR/lib/librknnrt.so  md5=$(md5sum "$DIR/lib/librknnrt.so" 2>/dev/null | cut -d' ' -f1)"
  log "board /usr/lib    : /usr/lib/librknnrt.so     md5=$(md5sum /usr/lib/librknnrt.so 2>/dev/null | cut -d' ' -f1)"
}

health_check() {
  local tag="$1" out rc
  out=$(timeout 90 "$DIR/rknn_heatmap_smoke" "$CONTROL" "$IMG" \
          --loops 3 --warmup 1 --tag "health($tag)" 2>&1)
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log ""; log "########## NPU HEALTH CHECK FAILED  tag=$tag  rc=$rc ##########"
    printf '%s\n' "$out" | tee -a "$RES"
    if [ "$rc" -eq 132 ] || printf '%s' "$out" | grep -q "Illegal instruction"; then
      log ">>>>>> BOARD NPU NEEDS POWER CYCLE <<<<<<"
      log "SIGILL (exit 132) inside rknn_run, with the known-good librknnrt.so already"
      log "first on LD_LIBRARY_PATH => the NPU itself is wedged (usually after an"
      log "'RKNPU: job timeout' -> 'soft reset'). It needs a PHYSICAL POWER CYCLE."
    else
      log ">>>>>> health check failed for a non-SIGILL reason -- see output above <<<<<<"
    fi
    exit 10
  fi
  log ""; log "########## health check OK  tag=$tag ##########"
  printf '%s\n' "$out" | grep -E "rknn_run|NPU\+dequant|clock sanity" | tee -a "$RES"
}

run_model() {
  local model="$1" label="$2" tmp="$DIR/.last_hm.log" rc val total host stride="${3:-}"
  log ""; log "===== rknn_heatmap_smoke :: $label ====="
  # ⚠️ --stride 必须**按臂**传：板卡工具会校验 "input/grid ratio == 640/stride"。
  #    no_p2+p3 臂输出 80x80 ⇒ stride 8；no_p2+p2 臂输出 160x160 ⇒ stride 4；
  #    原结构输出 320x320 ⇒ stride 2。
  #    不传就默认 2 ⇒ 新结构臂被 [FATAL] input/grid ratio 8 does not match --stride 2 拒绝。
  local stride_arg=()
  [ -n "$stride" ] && stride_arg=(--stride "$stride")
  log "cmd: ./rknn_heatmap_smoke $model $IMG --loops $LOOPS --warmup $WARMUP ${stride_arg[*]} --tag $label"
  timeout 600 "$DIR/rknn_heatmap_smoke" "$model" "$IMG" \
        --loops "$LOOPS" --warmup "$WARMUP" "${stride_arg[@]}" --tag "$label" > "$tmp" 2>&1
  rc=$?
  cat "$tmp" | tee -a "$RES"
  log "--- exit=$rc ---"
  # "    rknn_run     avg  55.736 ms  (98.3%)"  -> $3
  val=$(grep -E 'rknn_run +avg' "$tmp" | awk '{print $3}' | tail -1)
  total=$(grep -E 'NPU\+dequant +avg' "$tmp" | awk '{print $3}' | tail -1)
  host=$(grep -E '\+ host +avg' "$tmp" | awk '{print $4}' | tail -1)
  [ -n "$val" ]   && log "@@SUMMARY@@ $label|rknn_run_avg_ms|$val"
  [ -n "$total" ] && log "@@SUMMARY@@ $label|npu_dequant_avg_ms|$total"
  [ -n "$host" ]  && log "@@SUMMARY@@ $label|full_frame_avg_ms|$host"
}

log "==================== heatmap model speed session on RK3588 ===================="
log "date            : $(date)"
log "board dir       : $DIR"
log "control model   : $CONTROL"
say_lib
log "===================================================================================="

health_check "initial"

sleep "$SETTLE"
run_model "$CONTROL" "control_base_cs_t0043"

for arm in "$DIR"/arms/*.rknn; do
  [ -e "$arm" ] || continue
  sleep "$SETTLE"
  health_check "before-$(basename "$arm")"
  # --stride 从边车文件读（没有就退回 2 = 原结构）
  # 边车文件在**板卡本地**（STEP 2 一起 scp 过去的），命名 = <arm>.stride
  _st=$(cat "$DIR/arms/$(basename "$arm" .rknn).stride" 2>/dev/null || echo "")
  [ -n "$_st" ] || _st=2
  run_model "$arm" "arm_$(basename "$arm" .rknn)" "$_st"
done

log ""; log "==================== session complete ===================="
log "results: $RES"
exit 0
MEASURE_EOF
chmod +x "$MEASURE_SH"

if [ "$DRY_RUN" = "1" ]; then
  say "DRY-RUN: not touching the board. Planned:"
  echo "  scp -P $BOARD_PORT $MEASURE_SH $BOARD:$BOARD_DIR/measure_hm_speed_board.sh"
  echo "  ssh -p $BOARD_PORT $BOARD 'BOARD_DIR=$BOARD_DIR CONTROL_NAME=$CONTROL_NAME LOOPS=$LOOPS WARMUP=$WARMUP SETTLE=$SETTLE bash $BOARD_DIR/measure_hm_speed_board.sh'   # <== the ONE board invocation"
  exit 0
fi

# ===========================================================================
# STEP 4 -- deploy + verify
# ===========================================================================
say "STEP 4: deploying runner"
scp $SCP_OPTS -P "$BOARD_PORT" "$MEASURE_SH" "$BOARD:$BOARD_DIR/measure_hm_speed_board.sh" || die "scp failed"
ssh $SSH_OPTS -p "$BOARD_PORT" "$BOARD" \
  "cd $BOARD_DIR && md5sum rknn_heatmap_smoke feat.jpg $CONTROL_NAME lib/librknnrt.so measure_hm_speed_board.sh" 2>/dev/null \
  | sed 's/^/    board: /'
BOARD_LIB_MD5=$(ssh $SSH_OPTS -p "$BOARD_PORT" "$BOARD" "md5sum $BOARD_DIR/lib/librknnrt.so" 2>/dev/null | awk '{print $1}' | tr -d '\r')
[ "$BOARD_LIB_MD5" = "$GOOD_MD5_EXPECTED" ] || die "board librknnrt.so md5 $BOARD_LIB_MD5 is NOT the known-good one"

# ===========================================================================
# STEP 5 -- THE single board invocation
# ===========================================================================
say "STEP 5: running the packed session on the board (one invocation)"
ssh $SSH_OPTS -p "$BOARD_PORT" "$BOARD" \
  "BOARD_DIR=$BOARD_DIR CONTROL_NAME=$CONTROL_NAME LOOPS=$LOOPS WARMUP=$WARMUP SETTLE=$SETTLE bash $BOARD_DIR/measure_hm_speed_board.sh" \
  || die "board session failed -- see $BOARD_DIR/hm_speed_results.txt on the board"

# ===========================================================================
# STEP 6 -- read back + tabulate
# ===========================================================================
say "STEP 6: results"
RES_LOCAL="$OUT_ROOT/hm_speed_results.txt"
scp $SCP_OPTS -P "$BOARD_PORT" "$BOARD:$BOARD_DIR/hm_speed_results.txt" "$RES_LOCAL" || die "read-back failed"
echo
printf '%-52s %14s %12s %10s\n' "model" "rknn_run_ms" "vs_base" "speedup"
printf '%s\n' "----------------------------------------------------------------------------------------"
BASE_VAL=$(grep -E '^@@SUMMARY@@ control_base_cs_t0043\|rknn_run_avg_ms\|' "$RES_LOCAL" | tail -1 | awk -F'|' '{print $3}')
[ -z "$BASE_VAL" ] && BASE_VAL="$REF_BASE_MS"
grep -E '^@@SUMMARY@@ .*\|rknn_run_avg_ms\|' "$RES_LOCAL" | while IFS='|' read -r _ tag name val; do
  [ "$tag" = "@@SUMMARY@@" ] || continue
  printf '%-52s %14s %12s %10s\n' "$name" "$val" \
    "$(awk -v a="$val" -v b="$BASE_VAL" 'BEGIN{printf "%+.2f", a-b}')" \
    "$(awk -v a="$val" -v b="$BASE_VAL" 'BEGIN{if(b>0) printf "%.2fx", b/a; else print "-"}')"
done
echo
say "  historical reference baseline (same board family) = ${REF_BASE_MS} ms"
say "  in-session control (cs_t0043 same-source base)   = ${BASE_VAL} ms"
say "  raw output: $RES_LOCAL"
say "  next step: python manu/training/nogmc_struct_acceptance.py --speed-results $RES_LOCAL"
