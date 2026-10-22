#!/usr/bin/env bash
# 龙泉山全量 OSD 视频渲染 —— trial_0080 (带 track 臂)
#
# 做什么：
#   用 trial_0080 的带 track 系统级预测（preds_track）渲染**全量 24 序列** OSD 复核视频，
#   按项目自身交付判据 F1 >= 88% 分成 pass / un-pass 两级，落到**新建目录**，不碰旧 OSD。
#
# 为什么复用 render_sys_sota_osd.sh 而不是重写：
#   它已经封装了「一个全局工作点 + 每序列自己的分数」的契约：pred JSON 给出每序列 metrics，
#   SEARCH_JSON 给出全局超参，per_sequence_hyperparams.tsv 记录每行用的哪套超参与落在哪个目录，
#   分组规则因此可从 TSV 重新切分。渲染器本身（render_sys_sota_longquan.py）只画
#   红=最终系统检出 / 绿=命中的GT / 橙=漏掉的GT，无逐尺度诊断线。
#
# 关键路径：
#   预测   runs/optuna_mix_ratio/trials/trial_0080/lq_eval/preds_track/<SEQ>.json
#   GT     /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010
#   帧源   /mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg
#   输出   /mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_20261022   （新目录）
#   旧 OSD /mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_20261020              （只读，不动）
#
# 用法：
#   bash manu/videos/render_trial0080_longquan_osd.sh
#   OUT_DIR=/path/to/new OUT_CUT=88.0 JOBS=12 bash manu/videos/render_trial0080_longquan_osd.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PYTHONPATH=.

OUT_DIR="${OUT_DIR:-/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_trial0080_20261022}"
OUT_CUT="${OUT_CUT:-88.0}"
JOBS="${JOBS:-12}"
CODEC="${CODEC:-mp4v}"

TRIAL_DIR="runs/optuna_mix_ratio/trials/trial_0080"
PREDS_DIR="$TRIAL_DIR/lq_eval/preds_track"
SEARCH_JSON="$TRIAL_DIR/lq_eval/track_all.json"
D="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
F="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"

echo "=============================================================="
echo " 0. 前置检查"
echo "=============================================================="
[ -d "$PREDS_DIR" ] || { echo "[FAIL] 缺少预测目录 $PREDS_DIR（先跑 run_trial0080_longquan_eval.sh）"; exit 1; }
[ -d "$D" ] || { echo "[FAIL] 缺少 GT 目录 $D"; exit 1; }
[ -d "$F" ] || { echo "[FAIL] 缺少帧目录 $F"; exit 1; }
[ -f "$SEARCH_JSON" ] || { echo "[FAIL] 缺少评估报告 $SEARCH_JSON"; exit 1; }
NPRED=$(ls "$PREDS_DIR"/*.json 2>/dev/null | grep -v '/summary.json$' | wc -l)
echo " 预测 JSON: $NPRED 条"
if [ "$NPRED" -lt 24 ]; then
  echo "[FAIL] 预测只有 $NPRED 条（应 24 条），中止渲染（不产出残缺 OSD 集）"
  exit 1
fi

# 旧 OSD 目录保护：确认不会覆盖既有产物
echo " 旧 OSD 目录（只读）："
for d in /mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_20261020 \
         /mnt/data/siping/datasets/manu/龙泉山/osd_global_20261019 \
         /mnt/data/siping/datasets/manu/龙泉山/osd_review_20261019; do
  [ -d "$d" ] && echo "   - $d  ($(ls "$d"/pass/*.mp4 "$d"/un-pass/*.mp4 2>/dev/null | wc -l) 个 mp4，保持不动)"
done
if [ -e "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  echo "[FAIL] 输出目录已存在且非空: $OUT_DIR —— 换一个 OUT_DIR，绝不覆盖既有 OSD"
  exit 1
fi
mkdir -p "$OUT_DIR/pass" "$OUT_DIR/un-pass" "$OUT_DIR/logs"

echo
echo "=============================================================="
echo " 1. 把评估报告转成渲染器期望的 search JSON 结构"
echo "=============================================================="
# render_sys_sota_osd.sh 的 TSV 生成器读 data["best_global"]["params"|"metrics"|"per_sequence"]，
# 而 system_sota_longquan.py 写出的是 {"config","overall","per_sequence"}。这里做一次结构转换，
# 数值逐字段搬运，不做任何重算 —— TSV 里的分数仍来自评估本身。
RENDER_SEARCH_JSON="$OUT_DIR/search_json_for_render.json"
python - "$SEARCH_JSON" "$RENDER_SEARCH_JSON" <<'PY'
import json, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(src.read_text(encoding="utf-8"))
config = data.get("config") or {}
payload = {
    "best_global": {
        "params": config.get("params") or {},
        "metrics": data.get("overall") or {},
        "per_sequence": data.get("per_sequence") or {},
    },
    "_source": str(src),
}
dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] {dst} | params={len(payload['best_global']['params'])} "
      f"per_sequence={len(payload['best_global']['per_sequence'])}")
PY
SEARCH_JSON="$RENDER_SEARCH_JSON"

echo
echo "=============================================================="
echo " 2. 渲染全部序列（工作点=带 track 交付参数，阈值 F1 >= $OUT_CUT）"
echo "=============================================================="
PREDS_DIR="$PREDS_DIR" SEARCH_JSON="$SEARCH_JSON" OUT_DIR="$OUT_DIR" \
PASS_CUT="$OUT_CUT" JOBS="$JOBS" CODEC="$CODEC" \
  bash manu/videos/render_sys_sota_osd.sh --preds-dir "$PREDS_DIR" --search-json "$SEARCH_JSON" \
       --out-dir "$OUT_DIR" --jobs "$JOBS" --pass-cut "$OUT_CUT" --codec "$CODEC"

echo
echo "=============================================================="
echo " 3. 产物核验"
echo "=============================================================="
NP=$(ls -1 "$OUT_DIR/pass"/*.mp4 2>/dev/null | wc -l)
NU=$(ls -1 "$OUT_DIR/un-pass"/*.mp4 2>/dev/null | wc -l)
echo " pass    = $NP"
echo " un-pass = $NU"
echo " 合计    = $((NP + NU)) / 24"
if [ $((NP + NU)) -ne 24 ]; then
  echo "[WARN] 视频数不足 24，请查 $OUT_DIR/logs/*.log"
fi
echo " 旧 OSD 仍在："
for d in /mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_20261020; do
  [ -d "$d" ] && echo "   - $d  ($(ls "$d"/pass/*.mp4 "$d"/un-pass/*.mp4 2>/dev/null | wc -l) 个 mp4)"
done
du -sh "$OUT_DIR"
