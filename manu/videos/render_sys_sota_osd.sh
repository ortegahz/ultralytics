#!/usr/bin/env bash
# Render one OSD review video per business-domain sequence from the system-SOTA final detections.
#
# A SINGLE global working point: every sequence is rendered from the same global best hyper-parameter
# set, so the pass / un-pass split reflects one system configuration instead of 24 cherry-picked ones.
# pass / un-pass split on the project's own delivery criterion F1 >= 88% (audit_substandard_cases.py:78-81).
#
# Colours: red = final system detection, green = ground truth that was hit, orange = ground truth that
# was missed. No per-scale diagnostic lines are drawn (render_sys_sota_longquan.py).
#
# Overridable by environment variable or by the matching command-line flag:
#   PREDS_DIR   runs/sys_sota_preds                       --preds-dir
#   SEARCH_JSON runs/logs/sys_sota_longquan_search.json    --search-json
#   OUT_DIR     .../osd_sys_sota_20261020                  --out-dir
#   JOBS        12                                         --jobs
#   PASS_CUT    88.0                                       --pass-cut
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /tmp/pycharm_project_10ae9e2e

PREDS_DIR="${PREDS_DIR:-runs/sys_sota_preds}"
SEARCH_JSON="${SEARCH_JSON:-runs/logs/sys_sota_longquan_search.json}"
OUT_DIR="${OUT_DIR:-/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_20261020}"
JOBS="${JOBS:-12}"
PASS_CUT="${PASS_CUT:-88.0}"
CODEC="${CODEC:-mp4v}"
D=/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010
F=/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg

while [ $# -gt 0 ]; do
  case "$1" in
    --preds-dir)  PREDS_DIR="$2"; shift 2 ;;
    --search-json) SEARCH_JSON="$2"; shift 2 ;;
    --out-dir)     OUT_DIR="$2"; shift 2 ;;
    --jobs)        JOBS="$2"; shift 2 ;;
    --pass-cut)    PASS_CUT="$2"; shift 2 ;;
    --codec)       CODEC="$2"; shift 2 ;;
    -h|--help)     grep '^#' "$0" | cut -c3-; exit 0 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

[ -d "$PREDS_DIR" ] || { echo "缺少 pred JSON 目录 $PREDS_DIR"; exit 1; }
[ -d "$D" ] || { echo "缺少 GT 目录 $D"; exit 1; }
[ -d "$F" ] || { echo "缺少原始帧目录 $F"; exit 1; }
if [ ! -f "$SEARCH_JSON" ]; then
  echo "[WARN] 缺少 $SEARCH_JSON —— 全局超参列留空，指标改用各序列 pred JSON 的 metrics 字段"
fi

mkdir -p "$OUT_DIR/pass" "$OUT_DIR/un-pass" "$OUT_DIR/logs"

# The global working point and the per-sequence verdict live in one file: the "#" line records the
# hyper-parameter combination every row was rendered with, the rows below carry each sequence's own
# score, and the last column is the output directory so the split rule is re-splittable from the TSV.
# The header has no merge_radius/th column any more: this round has one global working point.
python - "$SEARCH_JSON" "$PREDS_DIR" "$PASS_CUT" > "$OUT_DIR/per_sequence_hyperparams.tsv" <<'PY'
import json, sys
from pathlib import Path

search_path, preds_dir, cut = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
params, per_seq, search_metrics = {}, {}, {}
if search_path.is_file():
    data = json.loads(search_path.read_text(encoding="utf-8"))
    best = data.get("best_global") or {}
    params = best.get("params") or {}
    per_seq = best.get("per_sequence") or data.get("per_sequence") or {}
    search_metrics = best.get("metrics") or {}

sequences = set(per_seq)
for path in sorted(preds_dir.glob("*.json")):
    if path.stem != "summary":
        sequences.add(path.stem)

def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0

def from_pred(path):
    """Fall back to the prediction JSON's own metrics when the search JSON has no per-sequence rows."""
    row = json.loads(path.read_text(encoding="utf-8")).get("metrics") or {}
    tp, fp = int(row.get("tp", 0)), int(row.get("fp", 0))
    fn = int(row.get("fn", 0))
    gt = int(row.get("gt", tp + fn))
    recall = float(row.get("recall", ratio(tp, tp + fn)))
    precision = float(row.get("precision", ratio(tp, tp + fp)))
    f1 = float(row.get("f1", ratio(2 * precision * recall, precision + recall)))
    return {
        "tp": tp, "fp": fp, "fn": fn, "gt": gt, "recall": recall,
        "precision": precision, "f1": f1, "far_per_frame": float(row.get("far_per_frame", 0.0)),
    }

rows, mismatched = [], []
for sequence in sorted(sequences):
    row = per_seq.get(sequence)
    pred_path = preds_dir / f"{sequence}.json"
    if not row and pred_path.is_file():
        row = from_pred(pred_path)
    if not row:
        continue
    tp, fp = int(row.get("tp", 0)), int(row.get("fp", 0))
    fn = int(row.get("fn", 0))
    gt = int(row.get("gt", tp + fn))
    recall = float(row.get("recall", ratio(tp, tp + fn)))
    precision = float(row.get("precision", ratio(tp, tp + fp)))
    f1 = float(row.get("f1", ratio(2 * precision * recall, precision + recall)))
    far = float(row.get("far_per_frame", 0.0))
    if pred_path.is_file():
        other = from_pred(pred_path)
        if (other["tp"], other["fp"], other["fn"]) != (tp, fp, fn):
            mismatched.append(sequence)
    rows.append((sequence, f1, recall, precision, far, gt, tp, fp, fn))

global_params = " ".join(
    f"{key}={params[key]}" for key in
    ("th_base", "th_salvage", "th_ground", "sky_ratio", "confirm_consec") if key in params)
print("# global params: " + (global_params or "(not available -- search JSON missing or empty)"))
print("sequence\tf1_pct\trecall_pct\tprecision_pct\tfar\tgt\ttp\tfp\tfn\tgroup\t"
      "th_base\tth_salvage\tth_ground\tsky_ratio\tconfirm_consec")
for sequence, f1, recall, precision, far, gt, tp, fp, fn in sorted(rows, key=lambda item: -item[1]):
    group = "pass" if f1 * 100 >= cut else "un-pass"
    print(f"{sequence}\t{f1 * 100:.3f}\t{recall * 100:.2f}\t{precision * 100:.2f}\t{far:.4f}\t"
          f"{gt}\t{tp}\t{fp}\t{fn}\t{group}\t"
          f"{params.get('th_base', '')}\t{params.get('th_salvage', '')}\t{params.get('th_ground', '')}\t"
          f"{params.get('sky_ratio', '')}\t{params.get('confirm_consec', '')}")
if mismatched:
    print(f"# WARNING: search JSON and prediction JSON disagree on tp/fp/fn for {len(mismatched)} "
          f"sequence(s): {', '.join(mismatched)}", file=sys.stderr)
PY

if ! grep -q '^sequence' "$OUT_DIR/per_sequence_hyperparams.tsv"; then
  echo "[ERROR] per_sequence_hyperparams.tsv 生成失败（搜索 JSON 或 pred JSON 解析错误）"; exit 1
fi

echo "[INFO] pred JSON 目录 = $PREDS_DIR"
echo "[INFO] 搜索 JSON     = $SEARCH_JSON"
echo "[INFO] 输出目录      = $OUT_DIR | 阈值 F1 >= $PASS_CUT | 并行 $JOBS | 编码 $CODEC"
grep '^#' "$OUT_DIR/per_sequence_hyperparams.tsv" || true
awk -F'\t' '$10=="pass"||$10=="un-pass"{c[$10]++} END{for(g in c) printf "  %-8s %d 条\n", g, c[g]}' \
  "$OUT_DIR/per_sequence_hyperparams.tsv"

render_one() {
  seq="$1"; grp="$2"; f1="$3"
  # F1 goes into the filename so a folder listing is already the per-sequence scoreboard.
  out="$OUT_DIR/$grp/${seq}_F1-${f1}.mp4"
  python manu/videos/render_sys_sota_longquan.py \
    --preds-dir "$PREDS_DIR" --labels-root "$D" --frames-root "$F" \
    --seq "$seq" --output "$out" \
    --fps 25 --canvas 1920x1080 --zoom 360 --codec "$CODEC" \
    --out-json "$OUT_DIR/logs/$seq.json" > "$OUT_DIR/logs/$seq.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -s "$out" ]; then
    echo "  [OK]   $grp/${seq}_F1-${f1}.mp4  $(du -h "$out" | cut -f1)"
  else
    echo "  [FAIL] $grp/${seq}_F1-${f1}.mp4 rc=$rc"
    tail -3 "$OUT_DIR/logs/$seq.log" | sed 's/^/         /'
  fi
}
export -f render_one
export OUT_DIR D F PREDS_DIR CODEC

# Frames come off a network mount, so these processes spend most of their time in iowait -- parallelising
# across sequences is far more effective than any per-process tuning.
grep -v '^#' "$OUT_DIR/per_sequence_hyperparams.tsv" \
  | tail -n +2 \
  | awk -F'\t' -v OFS='\t' '{print $1, $10, $2}' \
  | xargs -P "$JOBS" -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r s g f <<< "{}"; render_one "$s" "$g" "$f"'

echo
echo "[RESULT] pass    = $(ls -1 "$OUT_DIR/pass"/*.mp4 2>/dev/null | wc -l)"
echo "[RESULT] un-pass = $(ls -1 "$OUT_DIR/un-pass"/*.mp4 2>/dev/null | wc -l)"
du -sh "$OUT_DIR"
