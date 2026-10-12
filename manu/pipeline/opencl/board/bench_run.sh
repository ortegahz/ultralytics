#!/bin/sh
# ===========================================================================
# bench_run.sh -- the channel-prep latency benchmark, on the board.
#
# PRODUCTION CONFIG, not a benchmark config: --fused gpu --pyr-cache --threads 4
# --affinity a76 --hw-linear 0. A latency number from a different configuration
# is not a latency number for the thing being delivered.
#
# ONE PROCESS PER SEQUENCE -- and this is a measured requirement, not a style
# choice. Running all five sequences in one process gets OOM-killed on this 8 GiB
# board: RSS climbs +2.6 GiB at every sequence boundary. Diagnosis (mem_where2):
# at RSS = 5065 MiB, /proc/PID/smaps accounts for only ~147 MiB across all 205
# mappings ([heap] 65 MiB, [anon] 40 MiB, libmali 35 MiB). The host allocations
# are small and flat; the growth lives in Mali device-memory that the driver does
# not release until the context is destroyed. FusedGpu::configure() is grow-only
# by design, and a single context outlives every sequence, so the driver's
# unreclaimed pool accumulates across boundaries until the global OOM killer
# fires. Isolating each sequence into its own process bounds the pool to one
# sequence's working set and destroys the context in between.
#
# EXPERIMENT A  clean system -> baseline ledger.
# EXPERIMENT B  background load on the LITTLE cores (cpu0-3). The question is not
#               "is it slower" but "does P99/Max break the 40 ms red line". The
#               load is deliberately NOT on cpu4-7: the pipeline is pinned there,
#               and loading the same cores would measure lost parallelism rather
#               than system interference.
#
# Usage: sh bench_run.sh <A|B> <outdir> <warmup> <limit>
# ===========================================================================
set -u

EXP=$1
OUTDIR=$2
WARMUP=$3
LIMIT=$4

ROOT=/userdata/prof
BIN=$ROOT/gmc_stream_ocl
BMP=$ROOT/bmp

# FRAME LIMIT IS 400, AND THAT IS A RESULT, NOT A COMPROMISE.
#   The limit used to be 350 because VIDEO00028/VIDEO00032 were SIGKILLed at ~380:
#   the fit produced a finite-but-absurd outlier, required_pad returned 4863 px,
#   and the padded image became 10366x10238 = 106 Mpixel per slot x 45 slots
#   ~ 4.8 GB, which the global OOM killer took as file-rss:7623308kB.
#   That cause is now fixed at the source (three-layer fit plausibility gate +
#   bounded, quantised pad), so all five sequences run the full 400 frames.
#   If one ever dies again, that is a regression to investigate, not a reason to
#   quietly lower this number -- a benchmark that stops before the interesting
#   frame cannot tell you the interesting frame is gone.
#
# BENCH_EXTRA lets a caller append flags without editing this file, e.g.
#   BENCH_EXTRA="--prealloc-pad 1024" sh bench_run.sh C /userdata/prof/out/C 100 400
# It is echoed into the log so the numbers carry their own configuration.
EXTRA=${BENCH_EXTRA:-}

# BENCH_FUSED selects which arm computes the tail. Default "gpu" is the latency
# configuration; "both" runs both arms on the SAME fits and A/Bs them, which is
# the accuracy configuration and is far slower. Keeping it here rather than in
# the caller's own loop means every arm gets the retry/backoff below for free.
FUSED=${BENCH_FUSED:-gpu}

# BENCH_EXTRA_MAP applies DIFFERENT flags to different sequences:
#   BENCH_EXTRA_MAP="VIDEO00036_19700101_015740=--fit-corners 200" sh bench_run.sh ...
# Entries are "<sequence key>=<flags>", separated by semicolons. The key must
# match the sequence id exactly or be followed by an underscore, so "VIDEO00036"
# is enough for VIDEO00036_19700101_015740 while "VIDEO0003" claims nothing else. The corner budget is
# per-sequence by design (report section 5A.4): VIDEO00032 pays the largest
# accuracy cost in the corpus -- Ch1 max 68 / Ch2 max 48 -- for a speedup its
# p99 of 30.37 ms does not need, so a single global budget is not acceptable.
# Written with a `for` and no pipe on purpose: `echo ... | while read` runs the
# body in a SUBSHELL, so the accumulated flags die with it and every sequence
# silently falls back to the global EXTRA. That failure is invisible -- the run
# completes, it just quietly applies the wrong configuration to the two
# sequences whose accuracy cost was measured to be the highest.
extra_for() {
  _s=$1
  _acc=$EXTRA
  _rest=$BENCH_EXTRA_MAP
  while [ -n "$_rest" ]; do
    _entry=${_rest%%;*}
    if [ "$_entry" = "$_rest" ]; then _rest=; else _rest=${_rest#*;}; fi
    [ -n "$_entry" ] || continue
    _key=${_entry%%=*}
    # Exact match, or the key followed by an underscore. A bare prefix glob would
    # let a key like VIDEO0003 silently claim VIDEO00034, VIDEO00036 and
    # VIDEO00032 at once -- and the failure is invisible, because the run just
    # applies the wrong budget to three sequences and reports numbers normally.
    case "$_s" in
      "$_key"|"$_key"_*) _acc="$_acc ${_entry#*=}" ;;
    esac
  done
  echo "$_acc"
}

SEQS="VIDEO00034_19700101_015052 \
      VIDEO00028_19700101_013631 \
      VIDEO00032_19700101_014453 \
      VIDEO00002_19700101_001415 \
      VIDEO00036_19700101_015740"
# VIDEO00032 is BACK IN. It was excluded because the fit produced a finite outlier
# that drove devpad to ~100 Mpixel/slot and an OOM kill. With the fit plausibility
# gate + a shrink-capable configure() in place it is expected to run, and its
# [FIT-GATE] line is the evidence that the gate is what made the difference.
# If it still dies, that is a finding, not a reason to drop it again.

mkdir -p "$OUTDIR"
LOG=$OUTDIR/bench_$EXP.log

# Clock state goes INTO the log: the numbers then carry their own provenance and
# nobody has to trust that "locking" worked.
{
  echo "### environment $(date -u +%FT%TZ)  experiment=$EXP limit=$LIMIT warmup=$WARMUP"
  for c in 4 5 6 7; do
    printf '### cpu%s governor=%s cur_khz=%s\n' "$c" \
      "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)" \
      "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq)"
  done
  echo "### gpu governor=$(cat /sys/class/devfreq/fb000000.gpu/governor) cur=$(cat /sys/class/devfreq/fb000000.gpu/cur_freq)"
  echo "### soc_temp_mC=$(cat /sys/class/thermal/thermal_zone0/temp)"
  echo "### bench_extra=${EXTRA:-<none>}"
  echo "### bench_extra_map=${BENCH_EXTRA_MAP:-<none>}"
  echo "### fused=${FUSED}"
} > "$LOG" 2>&1

BENCH_EXTRA_MAP=${BENCH_EXTRA_MAP:-}
LOADPIDS=""
if [ "$EXP" = "B" ]; then
  taskset -c 0-3 sh -c 'while :; do :; done' &
  LOADPIDS="$LOADPIDS $!"
  echo "### background load on cpu0-3 pid(s):$LOADPIDS" >> "$LOG"
  sleep 2
fi

# No separate warm-up pass. It cost a whole extra process (Mali JIT + context
# build) per experiment and bought nothing the per-sequence --profile-warmup window
# does not already do: the first $WARMUP frames of every sequence are computed and
# recorded in the CSV but excluded from the statistics.

RC=0
run_seq() {
  s=$1
  EXTRA="$(extra_for "$s")"
  echo "### ---- $s ---- [$EXTRA]" >> "$LOG"
  "$BIN" --raw-root "$BMP" --sequence "$s" --limit "$LIMIT" --fused "$FUSED" \
         --pyr-cache --threads 4 --affinity a76 --hw-linear 0 \
         --kernel-dir "$ROOT/kernels" \
         --profile-csv "$OUTDIR/profile_${EXP}_${s}.csv" --profile-warmup "$WARMUP" \
         --md5-out "$OUTDIR/md5_${EXP}.txt" --timing $EXTRA >> "$LOG" 2>&1
}

for s in $SEQS; do
  r=1
  attempt=0
  # Retry, do not just sleep.
  #
  # Measured: a run can be SIGKILLed instantly at start-up -- right after the
  # [THREADS] banner, before a single frame -- while MemAvailable sits at 7.4 GiB.
  # That is the Mali driver's device pool: it is NOT reflected in /proc/meminfo, so
  # waiting on free memory returns immediately and never helps. The failure is real
  # (the global OOM killer fires on the driver's file-rss accounting) but its cause
  # is the predecessor's pool not yet released. Backing off and retrying is the
  # reliable fix; a fixed sleep is a guess at the same thing.
  while [ $attempt -lt 3 ]; do
    attempt=$((attempt+1))
    run_seq "$s"
    r=$?
    if [ $r -eq 0 ]; then break; fi
    echo "### RETRY $s attempt=$attempt rc=$r; backing off 25s" >> "$LOG"
    sleep 25
  done
  [ $r -ne 0 ] && { RC=$r; echo "### FAILED $s rc=$r after $attempt attempts" >> "$LOG"; }
  echo "### $s done rc=$r attempts=$attempt csv_rows=$(wc -l < "$OUTDIR/profile_${EXP}_${s}.csv" 2>/dev/null || echo 0)" >> "$LOG"
  sleep 8
done

if [ -n "$LOADPIDS" ]; then
  for p in $LOADPIDS; do kill $p 2>/dev/null; done
fi

{
  echo "### soc_temp_end_mC=$(cat /sys/class/thermal/thermal_zone0/temp)"
  echo "### gpu_transitions: $(cat /sys/class/devfreq/fb000000.gpu/available_governors | tr '\n' ' ')"
  echo "### rc=$RC"
} >> "$LOG"

echo "DONE rc=$RC"