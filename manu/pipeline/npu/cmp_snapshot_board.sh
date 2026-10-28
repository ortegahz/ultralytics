#!/usr/bin/env bash
# ===========================================================================
# cmp_snapshot_board.sh -- the ZERO-DROP, BIT-IDENTICAL acceptance of the GMC
# computation snapshot (the two-level cache), ON THE BOARD.
#
# WHAT IT PROVES
#   The snapshot version of the pipeline and the baseline version produce the
#   SAME GMC feature fingerprint, frame by frame, on a sequence where nothing
#   was dropped.  That is the whole claim: the snapshot only MOVES the pixels
#   the fit reads, it does not change them.
#
#   It also checks the snapshot's own coverage counter: every history read of
#   every frame must have been served from the PRIVATE snapshot.  A non-zero
#   count means the run fell back to the shared ring -- the numbers would
#   still be right, but the structural guarantee would be OFF, and that must
#   not be reported as a pass.
#
# WHY ZERO-DROP MATTERS
#   The baseline's history guarantee is a TUNED one: the ring is deeper than
#   the window by `gmc_ring_slack` frames, so the producer may run that far
#   ahead of the oldest owed frame.  On a sequence where frames were DROPPED,
#   the baseline itself already reads substituted history frames (nearest
#   older), and the two versions could legitimately differ there.  So the
#   acceptance has to run on a sequence where the camera queue never
#   overflowed -- hence --input-queue 1024, which is far deeper than the
#   GMC can fall behind by.
#
# USAGE (on the board, as root):
#   bash cmp_snapshot_board.sh [BASE_BIN] [SNAP_BIN] [WORKDIR]
#
#   BASE_BIN  the BASELINE binary (before the snapshot).  Default:
#             /root/rknn_mosaic_e2e_sweep4
#   SNAP_BIN  the SNAPSHOT binary.  Default: /root/rknn_mosaic_e2e_snap
#   WORKDIR   where the output directories are written.  Default: /tmp/snapcmp
#
# EXIT CODE: 0 = every comparison passed, 1 = at least one did not.
# ===========================================================================
set -uo pipefail

BASE="${1:-/root/rknn_mosaic_e2e_sweep4}"
SNAP="${2:-/root/rknn_mosaic_e2e_snap}"
WORK="${3:-/tmp/snapcmp}"

RAW_ROOT="/mnt/srv2/lq_ir_bmp"
SEQ="VIDEO00007_19700101_001324"
MODEL="/root/psearch/models/p0_off_A_none.rknn"
KERNELS="/root/psearch/kernels"
LIMIT=600

die() { echo "[ERROR] $*" >&2; exit 1; }
[ -x "$BASE" ] || die "baseline binary not executable: $BASE"
[ -x "$SNAP" ] || die "snapshot binary not executable: $SNAP"
[ -d "$RAW_ROOT" ] || die "raw root missing: $RAW_ROOT"
[ -f "$MODEL" ] || die "model missing: $MODEL"
[ -d "$KERNELS" ] || die "kernel dir missing: $KERNELS"

mkdir -p "$WORK" || die "cannot create $WORK"

echo "=== binaries ==="
md5sum "$BASE" "$SNAP"
echo

# ---------------------------------------------------------------------------
# The two runs.  They differ ONLY in --anchor-step and the affinity layout:
# step 10 is the shipping config (5 anchors), step 2 is the exact-equivalence
# config (21 anchors).  Everything else is identical, including the
# --input-queue 1024 that makes the sequence zero-drop.
# ---------------------------------------------------------------------------
run_one() {   # $1=binary $2=outdir $3=anchor-step $4=affinity [$5=extra...]
  local bin="$1" out="$2" astep="$3" aff="$4"; shift 4
  rm -rf "$out"; mkdir -p "$out"
  "$bin" --raw-root "$RAW_ROOT" --sequence "$SEQ" \
    --model "$MODEL" --out "$out" --kernel-dir "$KERNELS" \
    --limit "$LIMIT" --pace-ms 40 --input-queue 1024 --jitter-depth 3 \
    --gmc-ring-slack 256 --npu-workers 2 --post-workers 1 --gmc-workers 2 \
    --gmc-phys-anchor 1 --anchor-step "$astep" --cpu-affinity "$aff" "$@" \
    > "$out/stdout.log" 2> "$out/stderr.log"
  return $?
}

# The per-frame fingerprint comparison.  feature_md5.txt is
#   <sequence>\t<frame id, 6 digits>\t<md5 hex>
# so the frame id is column 2 and the hash is column 3.  The frames are
# matched BY ID, not by line number: a dropped frame would shift the lines,
# and a shifted comparison would silently compare different frames.
compare_md5() {   # $1=baseline dir $2=snapshot dir $3=label
  local a="$1/feature_md5.txt" b="$2/feature_md5.txt"
  [ -f "$a" ] || { echo "  [FAIL] $3: baseline feature_md5.txt missing"; return 1; }
  [ -f "$b" ] || { echo "  [FAIL] $3: snapshot feature_md5.txt missing"; return 1; }
  awk -F'\t' '{print $2"\t"$3}' "$a" | sort > "$WORK/.a.tsv"
  awk -F'\t' '{print $2"\t"$3}' "$b" | sort > "$WORK/.b.tsv"
  local na nb same
  na=$(wc -l < "$WORK/.a.tsv"); nb=$(wc -l < "$WORK/.b.tsv")
  same=$(comm -12 "$WORK/.a.tsv" "$WORK/.b.tsv" | wc -l)
  echo "  [$3] baseline frames=$na  snapshot frames=$nb  identical=$same"
  if [ "$na" -ne "$nb" ]; then
    echo "  [FAIL] $3: frame counts differ ($na vs $nb) -- the sequence was not zero-drop"
    return 1
  fi
  if [ "$same" -ne "$na" ]; then
    echo "  [FAIL] $3: only $same/$na frames are bit-identical"
    diff "$WORK/.a.tsv" "$WORK/.b.tsv" | head -20
    return 1
  fi
  echo "  [OK] $3: $same/$na frames bit-identical"
  return 0
}

# The snapshot's own coverage: every history read served from the private
# copy.  This is the counter that says the structural guarantee was ACTIVE.
check_snapshot_counter() {   # $1=dir $2=label
  local line
  line=$(grep -h '\[SNAPSHOT\]' "$1/stderr.log" 2>/dev/null)
  if [ -z "$line" ]; then
    echo "  [FAIL] $2: no [SNAPSHOT] line in stderr -- is this the snapshot build?"
    return 1
  fi
  echo "  $2: $line"
  if echo "$line" | grep -qvE '= 0 '; then
    echo "  [FAIL] $2: the snapshot did NOT cover every history read (see above)"
    return 1
  fi
  echo "  [OK] $2: every history read came from the private snapshot"
  return 0
}

rc=0

# ---------------------------------------------------------------- anchor 10
echo "==================================================================="
echo "=== anchor-step 10 (shipping config: 5 anchors) ==="
echo "==================================================================="
echo "--- baseline ---"
run_one "$BASE" "$WORK/base_a10" 10 "4-7" || die "baseline run failed (step 10)"
echo "--- snapshot ---"
run_one "$SNAP" "$WORK/snap_a10" 10 "4-7" || die "snapshot run failed (step 10)"
compare_md5 "$WORK/base_a10" "$WORK/snap_a10" "anchor-step 10" || rc=1
check_snapshot_counter "$WORK/snap_a10" "anchor-step 10" || rc=1
echo

# ----------------------------------------------------------------- anchor 2
# The affinity group is passed as ONE argument, so the ';' must survive the
# shell: it is quoted on purpose.
echo "==================================================================="
echo "=== anchor-step 2 (exact-equivalence config: 21 anchors) ==="
echo "==================================================================="
echo "--- baseline ---"
run_one "$BASE" "$WORK/base_a2" 2 "0-7" --gmc-anchor-threads 4 --gmc-affinity-group "0-3;4-7" \
  || die "baseline run failed (step 2)"
echo "--- snapshot ---"
run_one "$SNAP" "$WORK/snap_a2" 2 "0-7" --gmc-anchor-threads 4 --gmc-affinity-group "0-3;4-7" \
  || die "snapshot run failed (step 2)"
compare_md5 "$WORK/base_a2" "$WORK/snap_a2" "anchor-step 2" || rc=1
check_snapshot_counter "$WORK/snap_a2" "anchor-step 2" || rc=1
echo

echo "==================================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: PASS -- the snapshot is bit-identical to the baseline and covered every read."
else
  echo "RESULT: FAIL -- see the [FAIL] lines above."
fi
echo "==================================================================="
exit $rc
