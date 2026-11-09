#!/bin/bash
# 纵横AI RC2 OSD —— 渲染完成后自动收尾：序列划分 + 预标注候选 + 产物核验
#
# 为什么要有这个脚本（铁律：长链条收尾必须由服务器端自主执行，不能靠会话还在不在）：
#   渲染是 58 条视频 × 4 卡的长跑（约 3 小时）。渲完之后还要做三件事：
#     1) 出正式序列划分清单 manu/data/zongheng_ft_split.json
#     2) 出人工复核底稿 manu/docs/dataset/zongheng-tk.md
#     3) 出标注候选（web 标注器可直接读）manu/data/build_zongheng_prelabels.py
#   这三步都只依赖渲染产物，**不需要 GPU**，因此可以在服务器端一次性跑完。
set -uo pipefail
cd /tmp/pycharm_project_10ae9e2e || exit 2
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.

OUT="${OUT:-/mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_20261109}"
FRAMES="$OUT/frames"
LOG="${LOG:-/mnt/data/siping/runs/zongheng_rc2_osd_20261109.log}"
SPLIT_LOCAL="${SPLIT_LOCAL:-manu/data/zongheng_ft_split.json}"
REVIEW_LOCAL="${REVIEW_LOCAL:-manu/docs/dataset/zongheng-tk.md}"
PRELABEL_OUT="${PRELABEL_OUT:-/mnt/data/siping/datasets/manu/纵横AI数据采集_preannot_rc2}"

echo "[FINALIZE] $(date '+%F %T') waiting for the render to finish..."
echo "[FINALIZE] out=$OUT"
echo "[FINALIZE] split=$SPLIT_LOCAL review=$REVIEW_LOCAL prelabels=$PRELABEL_OUT"

# 等渲染进程全部退出（最长等 6 小时）。
# ⚠️ 模式必须**锚定**在 python 上：`-f` 匹配完整命令行，若模式出现在别的 shell 的命令行里
# （例如监控脚本、pgrep 自己），计数就永远不为 0 ⇒ 这个等待循环会**永不退出**。
for i in $(seq 1 720); do
  n=$(pgrep -cf "^python .*render_rc2_center_osd_video\.py" || true)
  [ "${n:-0}" -eq 0 ] && break
  sleep 30
done
sleep 10
echo "[FINALIZE] render procs left: ${n:-0}"

echo
echo "=============================================================="
echo " 1. 产物核验"
echo "=============================================================="
NMP4=$(ls -1 "$OUT"/*.mp4 2>/dev/null | wc -l)
NJSONL=$(ls -1 "$FRAMES"/*.jsonl 2>/dev/null | wc -l)
echo " mp4        = $NMP4 / 58"
echo " frames jsonl = $NJSONL / 58"
if [ "$NMP4" -ne 58 ] || [ "$NJSONL" -ne 58 ]; then
  echo "[WARN] 数量不足 58，查 $OUT/logs/shard*.log"
fi
echo " --- 每分片统计 ---"
for i in 0 1 2 3; do
  grep -h "^\[DONE\]" "$OUT"/logs/shard$i.log 2>/dev/null | wc -l | tr -d "\n"
  echo " 条 (shard$i)"
done

echo
echo "=============================================================="
echo " 2. 序列划分 + 人工复核底稿"
echo "=============================================================="
python manu/data/build_zongheng_ft_split.py \
  --frames-dir "$FRAMES" \
  --out "$SPLIT_LOCAL" \
  --review-sheet "$REVIEW_LOCAL"

echo
echo "=============================================================="
echo " 3. 标注候选（web 标注器可直接读）"
echo "=============================================================="
python manu/data/build_zongheng_prelabels.py \
  --frames-dir "$FRAMES" \
  --layout sequence \
  --out "$PRELABEL_OUT"

echo
echo "=============================================================="
echo " 4. 收尾核验"
echo "=============================================================="
python3 - "$SPLIT_LOCAL" "$REVIEW_LOCAL" "$PRELABEL_OUT" <<'PY'
import json, sys
from pathlib import Path

split = Path(sys.argv[1])
review = Path(sys.argv[2])
prelabel = Path(sys.argv[3])

data = json.loads(split.read_text(encoding="utf-8"))
groups = data["groups"]
print(f" split: {split} ({split.stat().st_size} bytes)")
for name in ("val", "train", "ignore"):
    print(f"   {name:6s} = {len(groups[name]['sequences'])} 条")
summary = data["baseline_reference"]["this_domain_detection_summary"]
print(f"   frames={summary['frames']} frames_with_det={summary['frames_with_detection']} "
      f"({summary['frames_with_detection']/max(summary['frames'],1):.1%}) detections={summary['detections']}")
print(f" review: {review} ({review.stat().st_size} bytes)")
labels = sorted(prelabel.glob("*.txt"))
print(f" prelabels: {prelabel} -> {len(labels)} label files")
PY

echo
echo "[FINALIZE] done $(date '+%F %T')"
echo " 视频目录 : $OUT"
echo " 划分清单 : $SPLIT_LOCAL"
echo " 复核底稿 : $REVIEW_LOCAL"
echo " 标注候选 : $PRELABEL_OUT"
