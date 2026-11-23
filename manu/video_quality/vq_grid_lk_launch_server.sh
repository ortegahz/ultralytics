#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Batch launcher for the Lucas-Kanade control experiment of the uniform-grid
# spatial-trackability metric.
#
# Resource discipline follows manu/memory/rules.md §〇之四:
#   - the worker pool is explicitly bounded (--workers) and every worker is pinned to 1 thread
#   - a resident watchdog samples MemAvailable / load every 60 s and brakes under 20 %
#   - the resource reading before the start is written into the same log
#   - a clip whose JSON already exists is skipped, so an interrupted batch can be resumed
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/tmp/pycharm_project_10ae9e2e}"
WORKERS="${1:-4}"
# 产物落点（铁律二十五）：中间产物一律出仓库目录。
PRODUCT_ROOT="${PRODUCT_ROOT:-/mnt/data/siping/runs/video_quality}"
OUT="${2:-${PRODUCT_ROOT}/grid_lk_out}"
LOG="${3:-${OUT}/run.log}"
META_DIR="${META_DIR:-${PRODUCT_ROOT}/out}"
GOOD_ROOT="${GOOD_ROOT:-/mnt/data/xxxu/朗基天香20260903}"
BAD_ROOT="${BAD_ROOT:-/mnt/data/xxxu/纵横挂飞}"

cd "${PROJECT_DIR}"
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
mkdir -p "${OUT}"

{
  echo "=== start $(date -Is) workers=${WORKERS} ==="
  echo "--- free -m"; free -m | head -2
  echo "--- nproc / uptime"; nproc; uptime
  echo "--- threads"; ps -eLf | wc -l
} >> "${LOG}"

(
  while true; do
    sleep 60
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    total=$(awk '/MemTotal/{print $2}' /proc/meminfo)
    pct=$((avail * 100 / total))
    echo "[WATCH $(date -Is)] MemAvailable=${avail}kB (${pct}%) load1=$(cut -d' ' -f1 /proc/loadavg) threads=$(ps -eLf | wc -l)" >> "${LOG}"
    if [ "${pct}" -lt 20 ]; then
      echo "[WATCH] BRAKE: MemAvailable < 20 % -> killing the batch" >> "${LOG}"
      pkill -f "vq_grid_lk.py" || true
      break
    fi
  done
) &
WATCH_PID=$!

python - "$WORKERS" "$OUT" "$META_DIR" "$GOOD_ROOT" "$BAD_ROOT" >> "${LOG}" 2>&1 <<'PY'
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

workers, out, meta_dir, good_root, bad_root = sys.argv[1:6]
out = Path(out)
suffixes = {".mp4", ".avi", ".mov", ".mkv", ".h264", ".h265", ".ts"}
clips = sorted(
    path for base in (good_root, bad_root) for path in Path(base).rglob("*")
    if path.is_file() and path.suffix.lower() in suffixes
)


REQUIRED_KEYS = ("lk_trackable_p50", "lk_trackable_mean", "lk_trackable_p50_fb2",
                 "lk_valid_fwd_p50", "lk_fb_err_p50", "lk_disp_p50", "lk_diverged_p50",
                 "lk_flat_cells_p50")


def existing_is_valid(json_path: Path) -> bool:
    """A resume may only reuse a result that is complete AND physically plausible.

    An aborted run can leave a half-written or *stale-schema* JSON behind; skipping such a
    file silently keeps the old numbers in the batch.  Measured on this task: a leftover JSON
    from the earlier (buggy) LK revision carried a 1130 px forward-backward error on a
    640-px-wide frame and was picked up as "already done" by the resume logic.
    """
    if not json_path.exists():
        return False
    try:
        summary = json.loads(json_path.read_text(encoding="utf-8")).get("summary", {})
    except (ValueError, OSError):
        return False
    if any(key not in summary for key in REQUIRED_KEYS):
        return False
    # a forward-backward error or a displacement larger than the frame is not a measurement
    return float(summary["lk_fb_err_p50"]) < 640.0 and float(summary["lk_disp_p50"]) < 640.0


def run_one(video: Path) -> str:
    json_path = out / f"{video.stem}.json"
    if existing_is_valid(json_path):
        return json.dumps({"file": video.name, "status": "skipped"})
    cmd = [
        "python", "manu/video_quality/vq_grid_lk.py",
        "--video", str(video),
        "--meta-dir", meta_dir,
        "--out", str(json_path),
        "--rows-csv", str(out / f"{video.stem}.rows.csv"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-1:] or [""]
    return json.dumps({"file": video.name, "rc": proc.returncode, "last": tail[0][:400]})


with ThreadPoolExecutor(max_workers=int(workers)) as pool:
    for line in pool.map(run_one, clips):
        print(f"[DONE] {line}", flush=True)
PY
RC=$?
kill "${WATCH_PID}" 2>/dev/null || true
{
  echo "--- free -m"; free -m | head -2
  echo "=== done rc=${RC} $(date -Is) ==="
} >> "${LOG}"
exit "${RC}"
