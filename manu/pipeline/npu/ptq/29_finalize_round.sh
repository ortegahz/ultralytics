#!/usr/bin/env bash
# =============================================================================
# 29_finalize_round.sh -- finish the 2026-10-23 PTQ round end to end:
#   1. wait for the full-scale three-track run to finish
#   2. sweep the full-scale TSVs offline (ZERO board cost) -> the deployment
#      threshold per track for the champion
#   3. stage the new candidates onto the board and screen them on the three
#      tracks (T1 / T3 / T2 subsets)
#   4. print one consolidated ranking table
#
# Usage: 29_finalize_round.sh <work_dir> <out_dir>
#   work_dir : the server-side work root (e.g. /mnt/data/siping/ptq_work)
#   out_dir  : where the sweep + screening artefacts go
# =============================================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
W="${1:?work dir}"
OUT="${2:?out dir}"
BOARD_MODELS_DIR="${BOARD_MODELS_DIR:-/userdata/ptq_models}"
THRESHOLDS="${THRESHOLDS:-0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70}"
MANIFESTS="${MANIFESTS:-fast/anti_uav_sub2500.json fast/001324_sub2000.json fast/golden_pool_sub1200.json}"

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd "$REPO"
mkdir -p "$OUT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- 1. wait for the full-scale driver -------------------------------------
log "waiting for the full-scale driver to finish ..."
for _ in $(seq 1 240); do
    n=$(ps -eo args | grep -c "[1]4_run_full_scale_nofs")
    [ "$n" -eq 0 ] && break
    sleep 20
done
log "full-scale driver finished (status below)"
cat "$W/full_scale_r2/status.json" 2>/dev/null | head -12

# ---- 2. offline threshold sweep of the full-scale TSVs (no board) -----------
# The sweep script globs *.tsv in --tsv-dir, so each track gets its own dir
# (symlinks) -- pointing it at the whole directory would MIX the three tracks.
for track in blind golden anti; do
    src="$W/full_scale_r2/${track}_board.tsv"
    [ -f "$src" ] || { log "[warn] missing $src"; continue; }
    d="$OUT/tsv_$track"
    mkdir -p "$d"
    ln -sf "$src" "$d/${track}_board.tsv"
    mf="$W/fast/${track}_sub2000.json"
    case "$track" in
        blind)   mf="$W/fast/001324_sub2000.json" ;;
        golden)  mf="$W/fast/golden_pool_sub1200.json" ;;
        anti)    mf="$W/fast/anti_uav_sub2500.json" ;;
    esac
    log "sweeping $track (manifest=$mf)"
    python -u manu/pipeline/npu/ptq/19_board_threshold_sweep.py \
        --manifest "$mf" --tsv-dir "$d" \
        --thresholds "$THRESHOLDS" --skip-pt \
        --out "$OUT/sweep_$track" 2>&1 | grep -E "^\[tsv\]|best F1" | tail -4
done

# ---- 3. board screening of the new candidates ------------------------------
CANDIDATES="${CANDIDATES:-\
$W/screen_r3/p0_off_6_normal_ah.rknn \
$W/screen_s8/p0_off_8_normal.rknn \
$W/screen_s9/p0_off_9_normal.rknn \
$W/screen_6_100/p0_off_6_100_normal.rknn \
$W/screen_r2/p0_off_6_normal.rknn}"

STAGE=""
for m in $CANDIDATES; do
    [ -f "$m" ] || { log "[warn] missing $m"; continue; }
    scp -P 22 -o BatchMode=yes "$m" ptq-board:"$BOARD_MODELS_DIR/" >/dev/null 2>&1 \
        && STAGE="$STAGE $m" || log "[warn] cannot stage $m"
done
log "staged $(echo $STAGE | wc -w) candidates on the board"

if [ -n "$STAGE" ]; then
    MANIFESTS="$MANIFESTS" BOARD_MODELS_DIR="$BOARD_MODELS_DIR" \
        bash manu/pipeline/npu/ptq/26_board_screening_round.sh "$OUT/board" $STAGE \
        2>&1 | tee "$OUT/board_screen.log" | grep -E "^\[board\]|best F1|@0.30 F1|mean=" | tail -40
fi

# ---- 4. consolidated table -------------------------------------------------
echo
echo "################ CONSOLIDATED ################"
python - "$OUT" "$W" <<'EOF'
import json, sys
from pathlib import Path
out, W = Path(sys.argv[1]), Path(sys.argv[2])

def curve(path, arm):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    a = (d.get("arms") or {}).get(arm)
    cur = (a or {}).get("curve") or []
    return cur

rows = []
for f in sorted(out.glob("board/*/*sweep*.json")):
    d = json.load(open(f))
    track = d.get("manifest_name") or f.parent.name
    for tag, arm in (d.get("arms") or {}).items():
        cur = arm.get("curve") or []
        if not cur:
            continue
        b = max(cur, key=lambda c: c["f1"])
        at30 = next((c for c in cur if abs(c["th"] - 0.30) < 1e-6), None)
        rows.append((tag, track, b["f1"], b["th"], (at30 or {}).get("f1", float("nan")),
                     (at30 or {}).get("tp", 0), (at30 or {}).get("fp", 0)))
print("%-28s %-22s %-18s %-18s %s" % ("arm", "track", "best F1@th", "F1@0.30 (TP/FP)", ""))
for tag, track, f1, th, f30, tp, fp in sorted(rows, key=lambda r: (r[1], -r[2])):
    print("%-28s %-22s %.4f@%-7.2f %.4f (%d/%d)" % (tag, track, f1, th, f30, tp, fp))

print()
for track in ("blind", "golden", "anti"):
    p = out / f"sweep_{track}" / f"{track}_threshold_sweep.json"
    if not p.is_file():
        continue
    d = json.load(open(p))
    for arm, a in (d.get("arms") or {}).items():
        cur = a.get("curve") or []
        if not cur:
            continue
        b = max(cur, key=lambda c: c["f1"])
        at30 = next((c for c in cur if abs(c["th"] - 0.30) < 1e-6), None)
        print("[full-scale sweep] %-8s %-22s best %.4f@%.2f | @0.30 %.4f (TP %d FP %d)  n=%d frames"
              % (track, arm, b["f1"], b["th"], (at30 or {}).get("f1", 0), (at30 or {}).get("tp", 0),
                 (at30 or {}).get("fp", 0), a.get("raw_detections", 0)))
EOF
echo "################ DONE $(date '+%H:%M:%S') ################"
