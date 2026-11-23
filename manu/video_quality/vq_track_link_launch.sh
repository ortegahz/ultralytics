#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# 🔴🔴🔴 【已作废 · 禁止运行】DEPRECATED — DO NOT RUN. 逻辑不成立，整版数据作废。
#
# 作废原因（2026-11-20 用户纠偏）：本脚本拿「反无人机检测器」去跑「没有无人机」的测试视频 ——
# 好片测出的「连续航迹」其实是背景虚警，坏片「零航迹」只是糊图抑制了虚警，
# 两者都不能证明「画质差 ⇒ 跟不住」。真实业务是「操作员点一下 → SOT 跟住」。
# ⇒ 本脚本产物不得写进任何正式报告。
#
# 正解：manu/video_quality/vq_grid*.py（第八节） / vq_sot_*.py（第九节，画质合格线）
# 本文件仅作历史留档。
set -euo pipefail

WORKERS="${WORKERS:-32}"
SEGMENT_FRAMES="${SEGMENT_FRAMES:-1500}"
THRESHOLD_ARMS="${THRESHOLD_ARMS:-0.20,0.48}"
OUT_DIR="${OUT_DIR:-/tmp/vq_track_out}"
JOBS_FILE="${JOBS_FILE:-/tmp/vq_track_jobs.txt}"
CROP_MODE="${CROP_MODE:-center1280}"
CPU_THREADS="${CPU_THREADS:-2}"
GPUS="${GPUS:-4}"

GOOD_DIR="/mnt/data/xxxu/朗基天香20260903"
BAD_DIR="/mnt/data/xxxu/纵横挂飞"

cd /tmp/pycharm_project_a7d966e6
mkdir -p "$OUT_DIR"
: > "$JOBS_FILE"

emit_jobs() { # $1=path $2=group $3=kind $4=modality $5=frames
  local path="$1" group="$2" kind="$3" modality="$4" frames="$5"
  local start=0 seg
  while [ "$start" -lt "$frames" ]; do
    seg=$(( frames - start ))
    [ "$seg" -gt "$SEGMENT_FRAMES" ] && seg="$SEGMENT_FRAMES"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$path" "$group" "$kind" "$modality" "$start" "$seg" >> "$JOBS_FILE"
    start=$(( start + seg ))
  done
}

# --- good group: 18 source clips (IR + VIS), each one contiguous job -------------------
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  frames="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$f" | tr -d ',')"
  frames="${frames:-0}"
  if [ "$frames" -eq 0 ]; then
    frames="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -show_entries format=duration -of csv=p=0 "$f" | awk -F, '{print $4}')"
    frames="$(python3 -c "print(int(float('$frames')*25))" 2>/dev/null || echo 1500)"
  fi
  case "$base" in
    *IR.h264|*红外02.mp4|*红外_IRT.mp4) modality=ir ;;
    *) modality=vis ;;
  esac
  emit_jobs "$f" good source "$modality" "$frames"
done < <(find "$GOOD_DIR" -maxdepth 1 -type f \( -name '*.h264' -o -name '*.mp4' \) -print0 | sort -z)

# --- bad group: 4 source clips (the 3 osd screen recordings are excluded on purpose) -----
for f in "$BAD_DIR"/*红外02.mp4 "$BAD_DIR"/*红外_IRT.mp4 "$BAD_DIR"/*可见光.mp4 "$BAD_DIR"/*可见光_OSDT.mp4; do
  [ -f "$f" ] || continue
  base="$(basename "$f")"
  frames="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$f" | tr -d ',')"
  frames="${frames:-0}"
  if [ "$frames" -eq 0 ]; then
    dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")"
    frames="$(python3 -c "print(int(float('$dur')*25))" 2>/dev/null || echo 1500)"
  fi
  case "$base" in
    *红外*) modality=ir ;;
    *) modality=vis ;;
  esac
  emit_jobs "$f" bad source "$modality" "$frames"
done

TOTAL=$(wc -l < "$JOBS_FILE")
echo "[LAUNCH] jobs=$TOTAL workers=$WORKERS segment_frames=$SEGMENT_FRAMES arms=$THRESHOLD_ARMS crop=$CROP_MODE out=$OUT_DIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_one() {
  local path="$1" group="$2" kind="$3" modality="$4" start="$5" seg="$6"
  local gpu=$(( RANDOM % GPUS ))
  python manu/video_quality/vq_track_link.py \
    --input "$path" --out-dir "$OUT_DIR" --device "$gpu" \
    --threshold-arms "$THRESHOLD_ARMS" --crop-mode "$CROP_MODE" \
    --start-frame "$start" --max-frames "$seg" \
    --group "$group" --kind "$kind" --modality "$modality" \
    --cpu-threads "$CPU_THREADS" > "$OUT_DIR/log_$(echo "$path" | md5sum | cut -c1-8)_${start}.log" 2>&1
}
export -f run_one
export OUT_DIR THRESHOLD_ARMS CROP_MODE CPU_THREADS GPUS

awk -F'\t' '{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6}' "$JOBS_FILE" \
  | xargs -P "$WORKERS" -I{} -d '\n' bash -c 'IFS=$'"'"'\t'"'"' read -r p g k m s n <<< "{}"; run_one "$p" "$g" "$k" "$m" "$s" "$n"'

echo "[LAUNCH] done; outputs in $OUT_DIR"
