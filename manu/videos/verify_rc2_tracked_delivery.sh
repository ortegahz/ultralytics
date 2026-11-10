#!/bin/bash
# 纵横AI RC2 时空轨迹档 —— 最终验收（服务器端）
#
# 检查渲染产物、收尾产物、工作点标注三件事是否自洽。
# 重点是**工作点不能串档**：本档必须报 th=0.20 + tracking=True。
set -uo pipefail
cd /tmp/pycharm_project_10ae9e2e || exit 2
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
conda activate uav
export PYTHONPATH=.

DATA_ROOT=/mnt/data/siping/datasets/manu
OUT="$DATA_ROOT/纵横AI数据采集_osd_rc2_tracked_th020_20261110"
PRELABEL="$DATA_ROOT/纵横AI数据采集_preannot_rc2_tracked"
SPLIT_LOCAL=manu/data/zongheng_ft_split_tracked.json
REVIEW_LOCAL=manu/docs/dataset/zongheng-tk-tracked.md
FAIL=0

echo "=============================================================="
echo " 纵横AI RC2 时空轨迹档 —— 最终验收"
echo "=============================================================="

echo "--- 1. 渲染产物 (期望 58) ---"
for pair in "mp4:$OUT/*.mp4" "frames:$OUT/frames/*.jsonl" "preds:$OUT/preds_track/*.json" "cache:$OUT/ms_cache/*.pkl"; do
  name=${pair%%:*}; pat=${pair#*:}
  n=$(ls -1 $pat 2>/dev/null | wc -l)
  mark="OK "; [ "$n" -ne 58 ] && { mark="FAIL"; FAIL=1; }
  printf "  %-7s %3d / 58   %s\n" "$name" "$n" "$mark"
done
n_sum=$(ls -1 $OUT/summary_shard*.json 2>/dev/null | wc -l)
mark="OK "; [ "$n_sum" -lt 4 ] && { mark="FAIL"; FAIL=1; }
printf "  %-7s %3d / 4    %s\n" "summary" "$n_sum" "$mark"

echo
echo "--- 2. 工作点（必须 th=0.20 + tracking=True） ---"
python3 - "$OUT" "$SPLIT_LOCAL" "$REVIEW_LOCAL" "$PRELABEL" <<'PY'
import glob, json, sys
from pathlib import Path

out, split, review, prelabel = (Path(p) for p in sys.argv[1:5])
fail = 0

print("  -- 渲染 summary --")
for p in sorted(out.glob("summary_shard*.json")):
    cfg = json.loads(p.read_text(encoding="utf-8")).get("config", {})
    th = cfg.get("threshold")
    tr = cfg.get("tracking")
    ok_th = abs(float(th) - 0.20) < 1e-9
    ok_tr = bool(tr) and str(tr).lower() not in {"false", "none", "off"}
    fail += 0 if (ok_th and ok_tr) else 1
    print(f"    {p.name}: threshold={th} merge_radius={cfg.get('merge_radius')} "
          f"tracking={'YES' if ok_tr else 'NO'} confirm_consec={cfg.get('confirm_consec')} "
          f"-> {'OK' if (ok_th and ok_tr) else 'FAIL'}")

print("  -- 划分清单 --")
data = json.loads(split.read_text(encoding="utf-8"))
point = data["baseline_reference"]["operating_point"]
summary = data["baseline_reference"]["this_domain_detection_summary"]
ok_th = abs(float(point["threshold"]) - 0.20) < 1e-9
ok_tr = bool(point.get("tracking"))
fail += 0 if (ok_th and ok_tr) else 1
print(f"    threshold={point['threshold']} tracking={point.get('tracking')} "
      f"merge_radius={point.get('merge_radius_px')} -> {'OK' if (ok_th and ok_tr) else 'FAIL'}")
for name in ("val", "train", "ignore"):
    print(f"    {name:6s} = {len(data['groups'][name]['sequences'])} 条")
print(f"    frames={summary['frames']} frames_with_detection={summary['frames_with_detection']} "
      f"({summary['frames_with_detection']/max(summary['frames'],1):.1%})")

print("  -- 复核底稿 --")
text = review.read_text(encoding="utf-8")
ok_tier = "时空轨迹档" in text
fail += 0 if ok_tier else 1
tier_line = [ln for ln in text.splitlines() if "工作点档位" in ln]
print(f"    {tier_line[0] if tier_line else '(no tier line)'} -> {'OK' if ok_tier else 'FAIL'}")
n_seq = text.count("_F1-")
print(f"    复核条目行数 = {n_seq}")

print("  -- 预标注候选 --")
labels = glob.glob(str(prelabel / "**" / "*.txt"), recursive=True)
print(f"    label files = {len(labels)}")

print("  -- 跟踪后 0 轨迹的序列（人工需单独看） --")
per = {}
for p in sorted((out / "frames").glob("*.jsonl")):
    n = sum(1 for line in p.open(encoding="utf-8") if line.strip() and json.loads(line)["detections"])
    per[p.stem] = n
empty = sorted(k for k, v in per.items() if v == 0)
print(f"    共 {len(empty)} 条：{', '.join(empty) if empty else '(无)'}")
print(f"\n  [VERIFY] failures = {fail}")
sys.exit(1 if fail else 0)
PY
[ $? -ne 0 ] && FAIL=1

echo
if [ "$FAIL" -eq 0 ]; then
  echo "[VERIFY] ALL GREEN"
else
  echo "[VERIFY] HAS FAILURES"
fi
exit "$FAIL"
