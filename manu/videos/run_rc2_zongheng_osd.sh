#!/usr/bin/env bash
# 纵横AI数据采集 —— RC2 (trial_0080) 全量「中心点」OSD 复核视频渲染
#
# 做什么：
#   用 RC2 = trial_0080 epoch3 对「纵横AI数据采集」58 个 mp4 做 **单帧** 多尺度检测，
#   把每个检测的**中心点**画成红色圆点（与龙泉山 OSD 同一套视觉语言），产出 58 个复核视频
#   + 每帧检测记录（JSONL），供人工评估检测效果、划分序列、为后续标注与训练做准备。
#
# 关键路径：
#   帧源   /mnt/data/siping/datasets/manu/纵横AI数据采集_mp4          （58 个 mp4，1920x1080）
#   权重   runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt  （RC2）
#   输出   /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_20261109   （新目录，不覆盖旧 OSD）
#   旧 OSD /mnt/data/siping/datasets/manu/纵横AI数据采集_osd                （trial_0474 时代，只读不动）
#
# 工作点（**实测**得来，不是拍的）：
#   在真实龙泉山 RC2 缓存上用 manu/evaluation/eval_multiscale_cache_th.py 扫出来的
#   merge_radius=768px / th=0.44 ⇒ F1 94.069%、Recall 90.74%、Precision 97.65%、FAR 0.0176/帧
#   （768px 在 640x512 原生画幅上 ≈ 每帧最多一个检测点；半径 128→93.923%、64→93.857% 都更差）
#
# 与旧 OSD 的差别：
#   1) 模型：trial_0474 → trial_0080(RC2)，且用 use_p0_highway=False（与 RC2 门禁/龙泉山缓存同配置）
#   2) 画法：矩形框 → **只有中心点**（红点 + 圆环 + 分数）
#   3) 无 track：每帧独立打分，没有跨帧状态
#
# 用法：
#   bash manu/videos/run_rc2_zongheng_osd.sh
#   SHARDS=1/4 LIMIT=150 bash manu/videos/run_rc2_zongheng_osd.sh        # 单分片冒烟
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=.

source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav

DATA_ROOT=/mnt/data/siping/datasets/manu
SRC="$DATA_ROOT/纵横AI数据采集_mp4"
OLD_OSD="$DATA_ROOT/纵横AI数据采集_osd"
# Default = the RC2 delivery working point (th 0.44 / r 768), measured on the real Longquan RC2 cache.
# A lower threshold (e.g. TH=0.20) is the *sensitive* variant: more frames carry a detection centre,
# which is what a reviewer wants when the point is to catch every candidate. Give it its OWN output
# directory (OUT=) so the two working points never overwrite each other.
OUT="${OUT:-$DATA_ROOT/纵横AI数据采集_osd_rc2_20261109}"
SHARDS="${SHARDS:-4}"
DEVICES="${DEVICES:-}"          # 空 = 分片 i 用 GPU i；否则逗号分隔的卡号列表
RESUME="${RESUME:-0}"           # 1 = 允许在非空输出目录上续跑（已完成的跳过，半截的重渲）
LIMIT="${LIMIT:-0}"
BATCH="${BATCH:-8}"
TH="${TH:-0.44}"
RADIUS="${RADIUS:-768}"
WEIGHTS="${WEIGHTS:-runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt}"

echo "=============================================================="
echo " 0. 前置检查"
echo "=============================================================="
[ -d "$SRC" ] || { echo "[FAIL] 缺少帧源目录 $SRC"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "[FAIL] 缺少 RC2 权重 $WEIGHTS"; exit 1; }
NV=$(ls -1 "$SRC"/*.mp4 2>/dev/null | wc -l)
echo " 帧源 mp4: $NV 个"
if [ "$NV" -ne 58 ]; then
  echo "[WARN] 帧源不是预期的 58 个，实际 $NV"
fi
mkdir -p "$OUT/logs"
echo " 旧 OSD（只读，不动）: $OLD_OSD ($(ls -1 "$OLD_OSD"/*.mp4 2>/dev/null | wc -l) 个 mp4)"

# 目录保护：**空目录**才允许开跑；非空目录只在「全是已完成的视频」时按续跑处理。
# 判定「已完成」= MP4 与其每帧记录**都**存在（记录只在视频渲完后落盘 ⇒ 半截文件会被识别出来）。
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  if [ "$RESUME" = "1" ]; then
    echo " 输出目录已存在，按 RESUME=1 续跑（已完成的视频会被跳过，半截的会重渲）"
  else
    echo "[FAIL] 输出目录已存在且非空: $OUT —— 换一个 OUT，绝不覆盖既有 OSD"
    echo "       （确需在**同一目录**上续跑时加 RESUME=1）"
    exit 1
  fi
fi

echo
echo "=============================================================="
echo " 1. 渲染（$SHARDS 分片）"
echo "=============================================================="
# DEVICES 允许把分片映射到**指定的** GPU 列表（例如另一条训练臂占着 1 号卡时
# 写 DEVICES="0,2,3"），分片序号与卡号解耦。
IFS=',' read -r -a DEV_ARR <<< "$DEVICES"
PIDS=()
for ((i = 0; i < SHARDS; i++)); do
  DEV="${DEV_ARR[$i]:-$i}"
  python manu/videos/render_rc2_center_osd_video.py \
    --input-root "$SRC" \
    --output-dir "$OUT" \
    --shard "$i/$SHARDS" \
    --hm-weights "$WEIGHTS" \
    --threshold "$TH" \
    --merge-radius "$RADIUS" \
    --batch-size "$BATCH" \
    --cpu-threads 2 \
    --device "$DEV" \
    --limit "$LIMIT" \
    --skip-existing \
    --out-json "$OUT/summary_shard$i.json" \
    --out-frames-dir "$OUT/frames" \
    > "$OUT/logs/shard$i.log" 2>&1 &
  PIDS+=($!)
  echo " shard $i/$SHARDS -> pid ${PIDS[-1]} (gpu $DEV)"
done

RC=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || RC=1
done
echo " 渲染 rc=$RC"

echo
echo "=============================================================="
echo " 2. 产物核验"
echo "=============================================================="
NMP4=$(ls -1 "$OUT"/*.mp4 2>/dev/null | wc -l)
echo " mp4  = $NMP4 / 58"
if [ "$NMP4" -ne 58 ]; then
  echo "[WARN] 视频数不足 58，查 $OUT/logs/shard*.log"
fi
for f in "$OUT"/summary_shard*.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
videos = data.get("videos", [])
frames = sum(v["rendered_frames"] for v in videos)
hits = sum(v["frames_with_detections"] for v in videos)
dets = sum(v["detections"] for v in videos)
print(f" {Path(sys.argv[1]).name}: videos={len(videos)} frames={frames} frames_with_det={hits} ({hits/max(frames,1)*100:.1f}%) det={dets}")
PY
done
du -sh "$OUT"
echo " 旧 OSD 仍在： $OLD_OSD ($(ls -1 "$OLD_OSD"/*.mp4 2>/dev/null | wc -l) 个 mp4)"
