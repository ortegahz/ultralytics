#!/usr/bin/env bash
# Render one OSD video per business-domain sequence at that sequence's own hyper-parameter optimum.
#
# Each sequence gets its own (merge_radius, threshold) taken from the per-sequence search JSON, so the
# video shows what the sequence can actually reach rather than what a single global working point gives.
# pass / un-pass split on the project's own delivery criterion F1 >= 88% (audit_substandard_cases.py:78-81).
#
# Colours: 640/160/80 scale boxes = orange/magenta/cyan thin lines, fused box = green when it hits a GT
# box and red when it is a false alarm (render_pred_gt_osd_video.py SCALE_COLORS / COLOR_TP / COLOR_FP).
set -u
source ~/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /tmp/pycharm_project_10ae9e2e

# MODE=perseq : every sequence rendered at ITS OWN best (radius, th)   -> osd_review_20261019
# MODE=global : every sequence rendered at the SINGLE global working point (768px / 0.22)
# Both modes read the same search JSON; they differ only in which per-sequence record is used.
# The at_global record already carries the global (radius, th), so no separate constant is needed.
MODE="${MODE:-perseq}"
case "$MODE" in
  perseq) OUT=/mnt/data/siping/datasets/manu/龙泉山/osd_review_20261019;  FIELD=best ;;
  global) OUT=/mnt/data/siping/datasets/manu/龙泉山/osd_global_20261019;   FIELD=at_global ;;
  *) echo "MODE 只能是 perseq 或 global（当前 $MODE）"; exit 2 ;;
esac

HYPER=runs/logs/perseq_hyper_nop0_inbox_r768.json
D=/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010
F=/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg
PASS_CUT=88.0

[ -f "$HYPER" ] || { echo "缺少 $HYPER，先跑 search_per_sequence_hyperparams.py"; exit 1; }
mkdir -p "$OUT/pass" "$OUT/un-pass" "$OUT/logs"

# Emit "seq<TAB>radius<TAB>th<TAB>f1<TAB>dir" so the split rule is visible in one place and re-splittable.
# The last column is the output directory itself -- it must be the directory name, not the verdict word,
# or the mkdir list and the write path drift apart and videos land in two parallel folder trees.
python - "$HYPER" "$PASS_CUT" "$FIELD" > "$OUT/per_sequence_hyperparams.tsv" <<'PY'
import json, sys
data = json.load(open(sys.argv[1])); cut = float(sys.argv[2]); field = sys.argv[3]
print("sequence\tmerge_radius\tth\tf1_pct\trecall_pct\tprecision_pct\tfar\tgroup")
for r in sorted(data["per_sequence"], key=lambda x: -x[field]["f1"]):
    b = r[field]
    print(f"{r['sequence']}\t{b['merge_radius']:.0f}\t{b['th']:.2f}\t{b['f1']*100:.3f}\t"
          f"{b['recall']*100:.2f}\t{b['precision']*100:.2f}\t{b['far_per_frame']:.4f}\t"
          f"{'pass' if b['f1']*100 >= cut else 'un-pass'}")
PY

echo "[MODE] $MODE (工作点记录 = $FIELD) | 输出 = $OUT"
awk -F'\t' 'NR>1{c[$8]++} END{for(g in c) printf "  %-8s %d 条\n", g, c[g]}' "$OUT/per_sequence_hyperparams.tsv"

render_one() {
  seq="$1"; rad="$2"; th="$3"; grp="$4"; f1="$5"
  # F1 goes into the filename so a folder listing is already the per-sequence scoreboard.
  out="$OUT/$grp/${seq}_F1-${f1}.mp4"
  python manu/videos/render_pred_gt_osd_video.py \
    --cache-root runs/ms_cache_nop0 \
    --labels-root "$D" --frames-root "$F" \
    --seq "$seq" --output "$out" \
    --threshold "$th" --merge-radius "$rad" \
    --show-scales 0,1,2 --fps 25 --canvas 1920x1080 --zoom 360 \
    --out-json "$OUT/logs/$seq.json" > "$OUT/logs/$seq.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "  [OK]   $grp/${seq}_F1-${f1}.mp4  r=$rad th=$th  $(du -h "$out" | cut -f1)"
  else
    echo "  [FAIL] $grp/${seq}_F1-${f1}.mp4 r=$rad th=$th rc=$rc"
    tail -3 "$OUT/logs/$seq.log" | sed 's/^/         /'
  fi
}
export -f render_one
export OUT D F

# Frames come off a network mount, so these processes spend most of their time in iowait -- parallelising
# across sequences is far more effective than any per-process tuning.
tail -n +2 "$OUT/per_sequence_hyperparams.tsv" \
  | awk -F'\t' -v OFS='\t' '{print $1, $2, $3, $8, $4}' \
  | xargs -P "${JOBS:-12}" -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r s r t g f <<< "{}"; render_one "$s" "$r" "$t" "$g" "$f"'

echo
echo "[RESULT] pass    = $(ls -1 "$OUT/pass"/*.mp4 2>/dev/null | wc -l)"
echo "[RESULT] un-pass = $(ls -1 "$OUT/un-pass"/*.mp4 2>/dev/null | wc -l)"
du -sh "$OUT"
