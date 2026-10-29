#!/usr/bin/env bash
# ===========================================================================
# run_arm.sh -- run ONE pipeline arm on the RK3588 board, with the two
# instruments the decoupling report needs:
#
#   1. the NPU HARDWARE load (/sys/kernel/debug/rknpu/load, sampled once a
#      second) -- the proof that the NPU cores stay busy even when the CPU
#      under the pre/post stages is being stolen;
#   2. an optional BURSTY CPU burner -- the "external business" that steals
#      the cores, in the shape the report argues about (bursts, not a
#      constant load).
#
# Usage:
#   run_arm.sh <ARM> <burn_cores|none> <burn_seconds> -- <e2e args...>
#
# Examples:
#   run_arm.sh CANON13_dec none 0 -- --pace-ms 40 --gmc-min-stride 1 ...
#   run_arm.sh CANON15_dec_jit 0-7 1200 -- --pace-ms 40 ...
#
# NOTHING here is destructive: no kill of a running e2e, no pkill -f, and
# the burner is ALWAYS torn down by the EXIT trap, so a crashed run cannot
# leave the board pegged.
# ===========================================================================
set -uo pipefail

ARM="${1:?ARM name required}"
BURNCORES="${2:-none}"
BURNSEC="${3:-0}"
shift 3 || true
[ "${1:-}" = "--" ] && shift

OUTDIR="/root/psearch/results/${ARM}"
LOG="/root/psearch/results/${ARM}.run.log"
LOADLOG="/root/psearch/results/${ARM}.npu_load.log"
BIN="${E2E_BIN:-/root/rknn_mosaic_e2e_v5}"
TIMEOUT="${E2E_TIMEOUT:-1500}"

rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}"

# ---- sweep the leftovers of a PREVIOUS arm ---------------------------------
# The first version of this script ran the burner in a subshell, so the EXIT
# trap could not see its children: four orphaned `yes` processes kept three
# A55 cores and one A76 core busy for eleven minutes and contaminated the two
# arms that followed.  The burner now records its children in a PID FILE, and
# this sweep kills anything a previous arm left behind.
for f in /root/psearch/.burn_pids_*; do
  [ -f "${f}" ] || continue
  while read -r p; do [ -n "${p}" ] && kill "${p}" 2>/dev/null; done < "${f}"
  rm -f "${f}"
done

# ---- the NPU hardware load sampler -----------------------------------------
# /sys/kernel/debug/rknpu/load prints one line per sample:
#   NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,
( while :; do cat /sys/kernel/debug/rknpu/load 2>/dev/null; sleep 1; done ) \
    > "${LOADLOG}" 2>/dev/null &
LOADPID=$!

# ---- the bursty CPU burner --------------------------------------------------
# "External business" in the shape the report argues about: a variable number
# of user-space threads (1..6) on the chosen cores, in bursts of 50..400 ms,
# with quiet gaps between the bursts.  `yes` is pure user-space compute.
BURN_PIDFILE="/root/psearch/.burn_pids_${ARM}"
: > "${BURN_PIDFILE}"
BURN_PIDS=()
burner() {
  local end=$(( $(date +%s) + BURNSEC ))
  while [ "$(date +%s)" -lt "${end}" ]; do
    local k=$(( RANDOM % 6 + 1 )) d
    : > "${BURN_PIDFILE}"
    for _ in $(seq 1 "${k}"); do
      taskset -c "${BURNCORES}" yes > /dev/null 2>&1 &
      echo "$!" >> "${BURN_PIDFILE}"
    done
    d=$(( RANDOM % 350 + 50 ))
    sleep "0.$(printf '%03d' "${d}")"
    while read -r p; do [ -n "${p}" ] && kill "${p}" 2>/dev/null; done < "${BURN_PIDFILE}"
    sleep "0.$(printf '%03d' $(( RANDOM % 200 + 50 )))"
  done
  : > "${BURN_PIDFILE}"
}

cleanup() {
  if [ -f "${BURN_PIDFILE}" ]; then
    while read -r p; do [ -n "${p}" ] && kill "${p}" 2>/dev/null; done < "${BURN_PIDFILE}"
    rm -f "${BURN_PIDFILE}"
  fi
  kill "${LOADPID}" 2>/dev/null
  wait "${LOADPID}" 2>/dev/null
}
trap cleanup EXIT INT TERM

if [ "${BURNCORES}" != "none" ] && [ "${BURNSEC}" -gt 0 ]; then
  echo "[run_arm] CPU burner: cores=${BURNCORES} bursts for ${BURNSEC}s" | tee -a "${LOG}"
  burner &
  BURNPID=$!
else
  BURNPID=""
fi

# ---- the arm ----------------------------------------------------------------
# `timeout` is what makes a remote long run self-killing: stopping the local
# ssh does NOT stop the remote process, but the timeout does.
# The caller's own --out is dropped: the output directory is THIS script's
# business, so the arm can never write into a certified result directory.
ARGS=()
for a in "$@"; do [ "${a}" = "--out" ] && continue; ARGS+=("${a}"); done
echo "[run_arm] ${ARM}: ${BIN} ${ARGS[*]} --out ${OUTDIR}" | tee -a "${LOG}"
timeout "${TIMEOUT}" "${BIN}" ${ARGS[@]+"${ARGS[@]}"} --out "${OUTDIR}" > "${LOG}" 2>&1
RC=$?
echo "[run_arm] ${ARM} EXIT=${RC}" | tee -a "${LOG}"

if [ -n "${BURNPID}" ]; then kill "${BURNPID}" 2>/dev/null; wait "${BURNPID}" 2>/dev/null; fi

# ---- the NPU load summary ----------------------------------------------------
# The RK3588 has THREE NPU cores; the runtime spreads the two contexts over
# them.  The per-core mean matters (a pipeline that uses 2 of the 3 cores at
# 100% and leaves the third at 0% is FULL), so the busiest-two mean is the
# number the report quotes.
#
# The parser is robust to the exact format ("Core0: 12.5%, " / "Core0: 0%,")
# because the idle and the busy lines are formatted by the same driver but
# not necessarily with the same precision.
if [ -s "${LOADLOG}" ]; then
  awk '
    /NPU load/ {
      samples++
      line = $0; act = 0
      while (match(line, /Core[0-9]+:[ ]*[0-9]+(\.[0-9]+)?%/)) {
        seg = substr(line, RSTART, RLENGTH)
        c = seg; sub(/:.*/, "", c); sub(/^Core/, "", c)
        v = seg; sub(/^[^:]*:[ ]*/, "", v); sub(/%/, "", v); v += 0
        s[c] += v; n[c]++; tot += v; tn++
        if (v > mx[c]) mx[c] = v
        act += v
        line = substr(line, RSTART + RLENGTH)
      }
      if (act > 0) { as++; atot += act }
    }
    END {
      if (tn == 0) { print "npu_load_samples=0"; exit }
      printf "npu_load_samples=%d\n", samples
      printf "npu_load_active_samples=%d\n", as + 0
      for (c in s) printf "npu_load_core%s_mean_pct=%.2f  npu_load_core%s_peak_pct=%.1f\n", c, s[c]/n[c], c, mx[c]
      best1 = -1; best2 = -1
      for (c in s) {
        m = s[c]/n[c]
        if (m > best1) { best2 = best1; best1 = m }
        else if (m > best2) { best2 = m }
      }
      printf "npu_load_busiest2_mean_pct=%.2f\n", (best1 + best2) / 2
      printf "npu_load_all_cores_mean_pct=%.2f\n", tot / tn
      if (as > 0) printf "npu_load_active_total_pct=%.2f\n", atot / as
    }' "${LOADLOG}" > "${OUTDIR}/npu_load.txt"
  cat "${OUTDIR}/npu_load.txt"
fi

exit ${RC}
