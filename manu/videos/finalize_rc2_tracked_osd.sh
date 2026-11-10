#!/bin/bash
# 纵横AI RC2 时空轨迹档 —— 渲染完成后自动收尾：序列划分 + 复核底稿 + 预标注候选 + 核验
#
# 与前两档的收尾脚本同构，只是产物名不同（tracked 后缀），三档互不覆盖。
set -uo pipefail
cd /tmp/pycharm_project_10ae9e2e || exit 2
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.

DATA_ROOT=/mnt/data/siping/datasets/manu
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_tracked_th020_20261110"
SPLIT_LOCAL=manu/data/zongheng_ft_split_tracked.json
REVIEW_LOCAL=manu/docs/dataset/zongheng-tk-tracked.md
PRELABEL_OUT="$DATA_ROOT/纵横AI数据采集_preannot_rc2_tracked"

echo "[FINALIZE] $(date '+%F %T') waiting for the tracked render to finish..."
# 模式锚定在 python 上：pgrep -f 匹配完整命令行，模式串出现在别的 shell 里会让计数永远不为 0。
for i in $(seq 1 720); do
  n=$(pgrep -cf "^python -u manu/videos/render_rc2_tracked_osd_video\.py" || true)
  [ "${n:-0}" -eq 0 ] && break
  sleep 30
done
sleep 10
echo "[FINALIZE] renderers left: ${n:-0}"

echo
echo "=============================================================="
echo " 1. 产物核验"
echo "=============================================================="
echo " mp4       = $(ls -1 "$OUT"/*.mp4 2>/dev/null | wc -l) / 58"
echo " frames    = $(ls -1 "$OUT/frames"/*.jsonl 2>/dev/null | wc -l) / 58"
echo " preds     = $(ls -1 "$OUT/preds_track"/*.json 2>/dev/null | wc -l) / 58"
echo " cache     = $(ls -1 "$OUT/ms_cache"/*.pkl 2>/dev/null | wc -l) / 58"
for i in 0 1 2 3; do
  grep -h "^\[SUCCESS\]" "$OUT/logs/shard$i.log" 2>/dev/null | tail -1
done

echo
echo "=============================================================="
echo " 2. 序列划分 + 复核底稿"
echo "=============================================================="
python manu/data/build_zongheng_ft_split.py \
  --frames-dir "$OUT/frames" \
  --out "$SPLIT_LOCAL" \
  --review-sheet "$REVIEW_LOCAL"

echo
echo "=============================================================="
echo " 3. 预标注候选（web 标注器可直接读）"
echo "=============================================================="
python manu/data/build_zongheng_prelabels.py \
  --frames-dir "$OUT/frames" \
  --layout sequence \
  --out "$PRELABEL_OUT"

echo
echo "=============================================================="
echo " 4. 收尾核验"
echo "=============================================================="
python3 - "$SPLIT_LOCAL" "$REVIEW_LOCAL" "$PRELABEL_OUT" "$OUT" <<'PY'
import glob, json, sys
from pathlib import Path

split, review, prelabel, out = (Path(p) for p in sys.argv[1:5])
data = json.loads(split.read_text(encoding="utf-8"))
groups = data["groups"]
point = data["baseline_reference"]["operating_point"]
summary = data["baseline_reference"]["this_domain_detection_summary"]
print(f" split: {split} ({split.stat().st_size} bytes)")
for name in ("val", "train", "ignore"):
    print(f"   {name:6s} = {len(groups[name]['sequences'])} 条")
print(f"   operating point: th={point['threshold']} r={point['merge_radius_px']} tracking={point['tracking']}")
print(f"   frames={summary['frames']} tracked_frames={summary['frames_with_detection']} "
      f"({summary['frames_with_detection']/max(summary['frames'],1):.1%})")
print(f" review: {review} ({review.stat().st_size} bytes)")
labels = glob.glob(str(prelabel / "**" / "*.txt"), recursive=True)
print(f" prelabels: {prelabel} -> {len(labels)} label files")
# 逐序列：跟踪后 0 轨迹的序列（可能被刚性静态剪枝剪掉）—— 人工要单独看
per = {}
for p in sorted(Path(out, "frames").glob("*.jsonl")):
    n = sum(1 for line in p.open(encoding="utf-8") if line.strip() and json.loads(line)["detections"])
    per[p.stem] = n
empty = sorted(k for k, v in per.items() if v == 0)
print(f" sequences with 0 tracked detection: {len(empty)}")
for name in empty:
    print(f"   - {name}")
PY

echo
echo "[FINALIZE] done $(date '+%F %T')"
echo " 视频目录 : $OUT"
echo " 划分清单 : $SPLIT_LOCAL"
echo " 复核底稿 : $REVIEW_LOCAL"
echo " 标注候选 : $PRELABEL_OUT"
