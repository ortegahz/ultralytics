#!/usr/bin/env bash
# ===========================================================================
# run_yolo26n_ref_board.sh -- ONE packed RK3588 board measurement session.
#
# PURPOSE
#   Establish what this board's NPU can actually deliver, using ROCKCHIP'S OWN
#   SDK demos as the control, so that "our model is slow" can be told apart
#   from "this board / this runtime is slow".
#
#   It measures, in a single board invocation:
#     1. SDK rknn_yolov5_demo   on yolov5s  (Rockchip's own demo, "average run X ms")
#     2. SDK rknn_mobilenet_demo on mobilenet_v1  (Rockchip's own demo)
#     3. rknn_bench_generic      on mobilenet_v1 (clean rknn_run-only ms; the
#                                SDK mobilenet demo has NO internal timer, so
#                                this is the only per-frame figure for it)
#     4. rknn_yolo26n_smoke      on the stock YOLO26n int8 model
#     5. rknn_yolo26n_smoke      on every built A/B arm in
#                                /media/manu/1TB-Volume/rknn_work/yolo26n/arms/
#                                (baseline / sram / compress / sram_compress /
#                                 nbuf_high / sram_nbuf -- whichever exist)
#     so the sram_prefer / compress_weight / nbuf_prefer knobs are measured on
#     the same board in the same session.
#
#   Every raw timing line goes to  <NFS>/yolo26n_ref/ref_results.txt  which the
#   board writes and the host reads back.
#
# WHY A SINGLE INVOCATION
#   This board's RKNPU driver is fragile: a long task or a repeat of the same
#   model can trip the 6 s job timeout -> soft reset -> the board then needs a
#   physical power cycle. So everything is packed into ONE ssh, with a settle
#   pause and a health check before every model.
#
# THE HEALTH CHECK
#   A 3-frame NPU run happens FIRST. If it fails, the script prints
#   "BOARD NPU NEEDS POWER CYCLE" and exits immediately, so the operator is
#   told before any model is touched.
#
#   NOTE (measured 2026-11-05): SIGILL/exit-132 on this board has TWO causes,
#   and the script distinguishes them:
#     (a) the board's /usr/lib/librknnrt.so (md5 9cb5f17b...) is a build
#         that SIGILLs inside rknn_run. The known-good copy (md5 56e425dc...)
#         is staged into <board>/lib and put first on LD_LIBRARY_PATH, so this
#         cause is normally already excluded;
#     (b) the NPU itself is wedged (e.g. after a job-timeout soft reset) and
#         genuinely needs a physical power cycle.
#   If (a) is already ruled out and the health check still SIGILLs, it is (b).
#
# USAGE
#   bash manu/pipeline/npu/run_yolo26n_ref_board.sh              # full session
#   bash manu/pipeline/npu/run_yolo26n_ref_board.sh --dry-run    # stage only,
#                                                                  # print every
#                                                                  # command
#   bash manu/pipeline/npu/print_yolo26n_ref_table.sh             # read back
# ===========================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"

# --- endpoints / paths ------------------------------------------------------
BOARD="${RKNPU_BOARD:-root@192.168.0.64}"
BOARD_PORT="${RKNPU_BOARD_PORT:-22}"
BOARD_DIR="${RKNPU_BOARD_DIR:-/root/yolo26n_ref}"
NFS_HOST_DIR="${RKNPU_NFS_DIR:-/run/media/manu/SSD960-LOCAL/nfs/npu/yolo26n_ref}"
NFS_BOARD_DIR="/mnt/manu/npu/yolo26n_ref"

SDK="${RKNN_SDK:-/media/manu/1TB-Volume/rk3588/rk3588_sdk/EVM3588-A_20250815/02Linux软件资料/SDK&Image/SDK/evm3588_linux_sdk_v2.1.1.0/external/rknpu2}"
MODELS_SRC="${RKNPU_MODELS_SRC:-/media/manu/1TB-Volume/rknn_work/models}"
ARMS_SRC="${RKNPU_ARMS_SRC:-/media/manu/1TB-Volume/rknn_work/yolo26n/arms}"
YOLO26N_SRC="${RKNPU_YOLO26N_SRC:-/media/manu/1TB-Volume/rknn_work/yolo26n}"
NFS_ROOT="$(dirname "$NFS_HOST_DIR")"          # for bus.jpg and the good lib

# --- binaries (built by scripts in this directory) --------------------------
DEMO_YOLOV5="$HERE/.demo-build/rknn_yolov5_demo"
DEMO_MOBILENET="$HERE/.demo-build-mobilenet/rknn_mobilenet_demo"
SMOKE="$HERE/out/rknn_yolo26n_smoke"
BENCH="$HERE/out/rknn_bench_generic"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"
SCP_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

say()  { echo "[run_yolo26n_ref] $*"; }
die()  { echo "[run_yolo26n_ref][ERROR] $*" >&2; exit 1; }

# ===========================================================================
# STEP 0 -- build anything that is missing, so the session is one command
# ===========================================================================
say "STEP 0: ensuring binaries exist"
if [ ! -x "$DEMO_YOLOV5" ]; then
  say "  building rknn_yolov5_demo (Rockchip SDK demo)"
  bash "$HERE/build_sdk_demo_arm64.sh" >/dev/null || die "build_sdk_demo_arm64.sh failed"
fi
if [ ! -x "$DEMO_MOBILENET" ]; then
  say "  building rknn_mobilenet_demo (Rockchip SDK demo)"
  bash "$HERE/build_sdk_mobilenet_demo_arm64.sh" >/dev/null || die "build_sdk_mobilenet_demo_arm64.sh failed"
fi
if [ ! -x "$SMOKE" ]; then
  say "  building rknn_yolo26n_smoke"
  bash "$HERE/build_rknn_yolo26n_arm64.sh" >/dev/null || die "build_rknn_yolo26n_arm64.sh failed"
fi
if [ ! -x "$BENCH" ]; then
  say "  building rknn_bench_generic"
  bash "$HERE/build_rknn_bench_generic_arm64.sh" >/dev/null || die "build_rknn_bench_generic_arm64.sh failed"
fi
for b in "$DEMO_YOLOV5" "$DEMO_MOBILENET" "$SMOKE" "$BENCH"; do
  [ -x "$b" ] || die "missing binary: $b"
  say "  ok $(basename "$b")  $(stat -c %s "$b") bytes  md5=$(md5sum "$b" | cut -d' ' -f1)"
done

# ===========================================================================
# STEP 1 -- stage models / labels / image / known-good runtime onto NFS
# ===========================================================================
say "STEP 1: staging NFS dir $NFS_HOST_DIR"
mkdir -p "$NFS_HOST_DIR/arms" || die "cannot create $NFS_HOST_DIR"
cp -f "$MODELS_SRC/yolov5s_rknn230.rknn"                        "$NFS_HOST_DIR/yolov5s_rknn230.rknn"
cp -f "$SDK/examples/rknn_mobilenet_demo/model/RK3588/mobilenet_v1.rknn" "$NFS_HOST_DIR/mobilenet_v1.rknn"
cp -f "$SDK/examples/rknn_yolov5_demo/model/coco_80_labels_list.txt"    "$NFS_HOST_DIR/coco_80_labels_list.txt"
cp -f "$YOLO26N_SRC/yolo26n_int8.rknn"                         "$NFS_HOST_DIR/yolo26n_int8.rknn"
cp -f "$NFS_ROOT/data/bus.jpg"                                 "$NFS_HOST_DIR/bus.jpg"
# known-good librknnrt (the board's /usr/lib copy SIGILLs inside rknn_run)
if [ -f "$NFS_ROOT/librknnrt.so" ]; then
  cp -f "$NFS_ROOT/librknnrt.so" "$NFS_HOST_DIR/librknnrt_good.so"
fi
rm -f "$NFS_HOST_DIR/arms/"*.rknn 2>/dev/null
for a in "$ARMS_SRC"/*.rknn; do
  [ -e "$a" ] && cp -f "$a" "$NFS_HOST_DIR/arms/"
done
say "  staged:"
( cd "$NFS_HOST_DIR" && md5sum * arms/*.rknn 2>/dev/null | sed 's/^/    /' ) || true

# ===========================================================================
# STEP 2 -- write the board-side runner
# ===========================================================================
MEASURE_SH="$HERE/out/measure_yolo26n_ref_board.sh"
mkdir -p "$HERE/out"
cat > "$MEASURE_SH" <<'MEASURE_EOF'
#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# measure_yolo26n_ref_board.sh -- RUNS ON THE BOARD.
# Deployed by run_yolo26n_ref_board.sh; do not run it by hand.
# ---------------------------------------------------------------------------
set -uo pipefail

DIR="${BOARD_DIR:-/root/yolo26n_ref}"   # binaries + local lib   (board's own disk)
NFS=/mnt/manu/npu/yolo26n_ref  # models + results       (NFS)
RES="$NFS/ref_results.txt"
IMG="$NFS/bus.jpg"
SETTLE="${SETTLE:-5}"

mkdir -p "$DIR/model" "$DIR/lib"
export LD_LIBRARY_PATH="$DIR/lib:/usr/lib"

# The results file is overwritten at the start of every session so a stale
# file from a previous (possibly dead) session can never be mistaken for
# this one's numbers.
: > "$RES"

log() { printf '%s\n' "$*" | tee -a "$RES"; }

say_lib() {
  log "librknnrt in use  : $DIR/lib/librknnrt.so  md5=$(md5sum "$DIR/lib/librknnrt.so" 2>/dev/null | cut -d' ' -f1)"
  log "board /usr/lib    : /usr/lib/librknnrt.so     md5=$(md5sum /usr/lib/librknnrt.so 2>/dev/null | cut -d' ' -f1)"
}

# ---------------------------------------------------------------------------
# health check: 3 NPU frames.  Fatal if it fails.
# ---------------------------------------------------------------------------
health_check() {
  local tag="$1" out rc
  if [ -x "$DIR/rknn_bench_generic" ] && [ -f "$NFS/mobilenet_v1.rknn" ]; then
    out=$(timeout 60 "$DIR/rknn_bench_generic" "$NFS/mobilenet_v1.rknn" \
            --warmup 1 --loops 3 --label "health($tag)" 2>&1)
  else
    out=$(timeout 60 "$DIR/rknn_yolo26n_smoke" "$NFS/yolo26n_int8.rknn" "$IMG" \
            --loops 3 --warmup 1 --pass-through 1 2>&1)
  fi
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log ""
    log "########## NPU HEALTH CHECK FAILED  tag=$tag  rc=$rc ##########"
    printf '%s\n' "$out" | tee -a "$RES"
    if [ "$rc" -eq 132 ] || printf '%s' "$out" | grep -q "Illegal instruction"; then
      log ">>>>>> BOARD NPU NEEDS POWER CYCLE <<<<<<"
      log "SIGILL (exit 132) inside rknn_run."
      if [ -f "$DIR/lib/librknnrt.so" ]; then
        log "The known-good librknnrt.so IS already first on LD_LIBRARY_PATH, so the"
        log "broken /usr/lib/librknnrt.so is NOT the cause: the NPU itself is wedged"
        log "(typically after an 'RKNPU: job timeout' -> 'soft reset'). It needs a"
        log "PHYSICAL POWER CYCLE by a human before any model can run."
      else
        log "No known-good librknnrt.so was staged, so this may be the broken"
        log "/usr/lib/librknnrt.so rather than a wedged NPU."
      fi
    else
      log ">>>>>> health check failed for a non-SIGILL reason -- see output above <<<<<<"
    fi
    exit 10
  fi
  log ""
  log "########## health check OK  tag=$tag ##########"
  printf '%s\n' "$out" | grep -E "sdk version|rknn_run|NPU total" | tee -a "$RES"
}

# ---------------------------------------------------------------------------
# Rockchip's own yolov5 demo.  Its "average run" number INCLUDES
# post_process (PERF_WITH_POST=1) -- that is the SDK's number, verbatim.
# ---------------------------------------------------------------------------
run_sdk_yolov5() {
  local model="$1" label="$2" tmp="$DIR/.last_yolov5.log" rc val
  log ""
  log "===== SDK rknn_yolov5_demo :: $label ====="
  log "cmd: (cd $DIR && ./rknn_yolov5_demo $model $IMG)"
  ( cd "$DIR" && timeout 300 ./rknn_yolov5_demo "$model" "$IMG" ) > "$tmp" 2>&1
  rc=$?
  cat "$tmp" | tee -a "$RES"
  log "--- SDK yolov5 demo exit=$rc ---"
  grep -E "once run use|average run" "$tmp" | tee -a "$RES"
  val=$(grep -oE 'average run +[0-9.]+' "$tmp" | grep -oE '[0-9.]+$' | tail -1)
  [ -n "$val" ] && log "@@SUMMARY@@ $label|sdk_yolov5_demo_average_run_ms|$val"
}

# ---------------------------------------------------------------------------
# Rockchip's own mobilenet demo.  NOTE: it has NO internal timer -- it runs
# the model once and prints Top5 -- so its wall clock is reported instead and
# the clean per-frame figure comes from rknn_bench_generic right after.
# ---------------------------------------------------------------------------
run_sdk_mobilenet() {
  local model="$1" tmp="$DIR/.last_mobilenet.log" rc t0 t1
  log ""
  log "===== SDK rknn_mobilenet_demo :: mobilenet_v1 ====="
  log "cmd: (cd $DIR && ./rknn_mobilenet_demo $model $IMG)"
  t0=$(date +%s%N)
  ( cd "$DIR" && timeout 300 ./rknn_mobilenet_demo "$model" "$IMG" ) > "$tmp" 2>&1
  rc=$?
  t1=$(date +%s%N)
  cat "$tmp" | tee -a "$RES"
  log "--- SDK mobilenet demo exit=$rc ---"
  log "--- wall clock of the whole process: $(( (t1 - t0) / 1000000 )) ms (model load + 1 inference; the SDK demo prints no ms line) ---"
}

# ---------------------------------------------------------------------------
# clean rknn_run-only timing, model agnostic
# ---------------------------------------------------------------------------
run_bench() {
  local model="$1" label="$2" tmp="$DIR/.last_bench.log" rc val
  log ""
  log "===== rknn_bench_generic :: $label ====="
  log "cmd: ./rknn_bench_generic $model --warmup 10 --loops 50"
  timeout 300 "$DIR/rknn_bench_generic" "$model" --warmup 10 --loops 50 --label "$label" > "$tmp" 2>&1
  rc=$?
  cat "$tmp" | tee -a "$RES"
  log "--- exit=$rc ---"
  # 'rknn_run +avg' only -- the tool also prints an "rknn_run only -> N fps"
  # line that would otherwise be picked up by a looser pattern.
  val=$(grep -E 'rknn_run +avg' "$tmp" | awk '{print $3}' | tail -1)
  [ -n "$val" ] && log "@@SUMMARY@@ $label|rknn_bench_generic_rknn_run_ms|$val"
}

# ---------------------------------------------------------------------------
# our own smoke tool: warmup + timed loop, rknn_run average reported
# ---------------------------------------------------------------------------
run_smoke() {
  local model="$1" label="$2" loops="${3:-30}" tmp="$DIR/.last_smoke.log" rc val
  log ""
  log "===== rknn_yolo26n_smoke :: $label ====="
  log "cmd: ./rknn_yolo26n_smoke $model $IMG --loops $loops --warmup 10 --pass-through 1"
  timeout 300 "$DIR/rknn_yolo26n_smoke" "$model" "$IMG" \
            --loops "$loops" --warmup 10 --pass-through 1 > "$tmp" 2>&1
  rc=$?
  cat "$tmp" | tee -a "$RES"
  log "--- exit=$rc ---"
  val=$(grep -E 'rknn_run +avg' "$tmp" | awk '{print $3}' | tail -1)
  [ -n "$val" ] && log "@@SUMMARY@@ $label|rknn_yolo26n_smoke_rknn_run_ms|$val"
}

# ===========================================================================
# session
# ===========================================================================
log "==================== YOLO26n reference measurement on RK3588 ===================="
log "date            : $(date)"
log "board           : $DIR (binaries) / $NFS (models+results)"
say_lib
log "===================================================================================="

health_check "initial"

sleep "$SETTLE"
run_sdk_yolov5 "$NFS/yolov5s_rknn230.rknn" "yolov5s_rknn230"

sleep "$SETTLE"
health_check "before-mobilenet"
run_sdk_mobilenet "$NFS/mobilenet_v1.rknn"
run_bench "$NFS/mobilenet_v1.rknn" "mobilenet_v1"

sleep "$SETTLE"
health_check "before-yolo26n-stock"
run_smoke "$NFS/yolo26n_int8.rknn" "yolo26n_int8_stock" 30

for arm in "$NFS"/arms/*.rknn; do
  [ -e "$arm" ] || continue
  sleep "$SETTLE"
  health_check "before-$(basename "$arm")"
  run_smoke "$arm" "arm_$(basename "$arm" .rknn)" 30
done

log ""
log "==================== session complete ===================="
log "results: $RES"
exit 0
MEASURE_EOF
chmod +x "$MEASURE_SH"
say "STEP 2: board-side runner written -> $MEASURE_SH"

# ===========================================================================
# STEP 3 -- deploy to the board's LOCAL disk (never the NFS mount) + md5 check
# ===========================================================================
DEPLOY_LIST=(
  "$DEMO_YOLOV5"
  "$DEMO_MOBILENET"
  "$SMOKE"
  "$BENCH"
  "$MEASURE_SH"
)
# md5 manifest, host side
MANIFEST="$HERE/out/.deploy_manifest.md5"
: > "$MANIFEST"
for f in "${DEPLOY_LIST[@]}"; do md5sum "$f" >> "$MANIFEST"; done
if [ -f "$NFS_HOST_DIR/librknnrt_good.so" ]; then
  md5sum "$NFS_HOST_DIR/librknnrt_good.so" | sed 's#  .*/#  lib/#' >> "$MANIFEST"
fi
md5sum "$NFS_HOST_DIR/coco_80_labels_list.txt" | sed 's#  .*/#  model/#' >> "$MANIFEST"

if [ "$DRY_RUN" = "1" ]; then
  say "DRY-RUN: not touching the board. Planned commands:"
  echo "  ssh -p $BOARD_PORT $BOARD 'mkdir -p $BOARD_DIR/lib $BOARD_DIR/model'"
  for f in "${DEPLOY_LIST[@]}"; do
    echo "  scp -P $BOARD_PORT $f $BOARD:$BOARD_DIR/"
  done
  [ -f "$NFS_HOST_DIR/librknnrt_good.so" ] && \
    echo "  scp -P $BOARD_PORT $NFS_HOST_DIR/librknnrt_good.so $BOARD:$BOARD_DIR/lib/librknnrt.so"
  echo "  scp -P $BOARD_PORT $NFS_HOST_DIR/coco_80_labels_list.txt $BOARD:$BOARD_DIR/model/"
  echo "  ssh -p $BOARD_PORT $BOARD 'md5sum ...'  # compared against $MANIFEST"
  echo "  ssh -p $BOARD_PORT $BOARD 'bash $BOARD_DIR/measure_yolo26n_ref_board.sh'   # <== the ONE board invocation"
  echo
  echo "DRY-RUN: planned board-side invocation:"
  cat "$MEASURE_SH"
  exit 0
fi

say "STEP 3: deploying to $BOARD:$BOARD_DIR (board's local disk)"
ssh $SSH_OPTS -p "$BOARD_PORT" "$BOARD" "rm -rf $BOARD_DIR && mkdir -p $BOARD_DIR/lib $BOARD_DIR/model" \
  || die "cannot create $BOARD_DIR on the board"
for f in "${DEPLOY_LIST[@]}"; do
  scp $SCP_OPTS -P "$BOARD_PORT" "$f" "$BOARD:$BOARD_DIR/" || die "scp failed: $f"
done
if [ -f "$NFS_HOST_DIR/librknnrt_good.so" ]; then
  scp $SCP_OPTS -P "$BOARD_PORT" "$NFS_HOST_DIR/librknnrt_good.so" "$BOARD:$BOARD_DIR/lib/librknnrt.so" \
    || die "scp failed: librknnrt.so"
fi
scp $SCP_OPTS -P "$BOARD_PORT" "$NFS_HOST_DIR/coco_80_labels_list.txt" "$BOARD:$BOARD_DIR/model/" \
  || die "scp failed: coco_80_labels_list.txt"

say "STEP 3b: verifying md5 on the board"
BOARD_MD5=$(ssh $SSH_OPTS -p "$BOARD_PORT" "$BOARD" \
  "cd $BOARD_DIR && md5sum rknn_yolov5_demo rknn_mobilenet_demo rknn_yolo26n_smoke rknn_bench_generic measure_yolo26n_ref_board.sh lib/librknnrt.so model/coco_80_labels_list.txt" \
  2>/dev/null)
printf '%s\n' "$BOARD_MD5" | sed 's/^/    board: /'
# Compare the SET OF HASHES, not "hash path" pairs: the same file legitimately
# lands under a different relative path on the board (lib/librknnrt.so vs
# librknnrt_good.so, model/coco_80_labels_list.txt vs coco_80_labels_list.txt),
# so a path-sensitive comparison reports a false mismatch.
HOST_MD5=$(awk '{print $1}' "$MANIFEST" | sort)
BOARD_SORT=$(printf '%s\n' "$BOARD_MD5" | awk '{print $1}' | sort)
if [ "$HOST_MD5" != "$BOARD_SORT" ]; then
  echo "--- host manifest ---"; printf '%s\n' "$HOST_MD5"
  echo "--- board manifest ---"; printf '%s\n' "$BOARD_SORT"
  die "md5 mismatch between host and board -- NOT running the benchmark"
fi
say "  md5 verified for all deployed files"

# ===========================================================================
# STEP 4 -- THE single board invocation
# ===========================================================================
say "STEP 4: running the packed measurement session on the board (one invocation)"
ssh $SSH_OPTS -p "$BOARD_PORT" "$BOARD" "BOARD_DIR=$BOARD_DIR bash $BOARD_DIR/measure_yolo26n_ref_board.sh" \
  || die "board session failed (exit $?) -- see $NFS_HOST_DIR/ref_results.txt"

# ===========================================================================
# STEP 5 -- read back and tabulate
# ===========================================================================
say "STEP 5: results"
bash "$HERE/print_yolo26n_ref_table.sh" || true
