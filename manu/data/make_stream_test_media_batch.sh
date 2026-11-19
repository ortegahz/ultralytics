#!/usr/bin/env bash
# 批量生成嵌入式组推流测试素材（龙泉山 pass 序列集）
#
# 每个序列产出三件套（命名/规格与 make_stream_test_media.sh 完全一致）：
#   <SEQ>_original_1920x1080.mp4      裸 H.264 流无损封装（-c copy，像素级无损）
#   <SEQ>_640x512_lossless_gray.mp4  BMP 处理链 JPG 序列 -> x264 qp0（逐像素校验参照）
#   <SEQ>_640x512.mp4                BMP 处理链 JPG 序列 -> x264 crf18 yuv420p（推流主用）
#
# 设计要点：
#   * 只做「素材转换」，不碰任何 OSD 视频 —— 输入是原始采集裸流 + BMP 链 JPG 序列
#   * 断点续跑：已存在且帧数完整的序列直接跳过（--force 可强制重做）
#   * 失败响亮：任一序列退出码非 0 ⇒ 整个批次退出码非 0，并在日志里点名
#   * 并行受控：--jobs N（N 个序列并发），每个 ffmpeg 内部 threads=4
#
# 用法:
#   bash make_stream_test_media_batch.sh --seq-list <清单文件> --out <输出目录> [选项]
#   选项:
#     --jobs N            并发序列数（默认 3）
#     --src-root DIR      原始裸流根（默认 /mnt/data/xxxu/龙泉山/ir）
#     --jpg-root DIR      BMP 链 JPG 帧根（默认 /mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg）
#     --single-script F   单序列脚本路径（默认 与本脚本同目录的 make_stream_test_media.sh）
#     --force             已完成的序列也重做
#     --only-seq SEQ      只做这一个序列（可重复）
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEQ_LIST=""; OUT=""; JOBS=3
SRC_ROOT="/mnt/data/xxxu/龙泉山/ir"
JPG_ROOT="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
SINGLE="$SELF_DIR/make_stream_test_media.sh"
FORCE=0
ONLY=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seq-list)      SEQ_LIST="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --jobs)          JOBS="$2"; shift 2 ;;
    --src-root)      SRC_ROOT="$2"; shift 2 ;;
    --jpg-root)      JPG_ROOT="$2"; shift 2 ;;
    --single-script) SINGLE="$2"; shift 2 ;;
    --force)         FORCE=1; shift ;;
    --only-seq)      ONLY+=("$2"); shift 2 ;;
    -h|--help)       sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "[FATAL] 未知参数: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$SEQ_LIST" && -n "$OUT" ]] || { echo "[FATAL] 必须给 --seq-list 与 --out" >&2; exit 2; }
[[ -f "$SINGLE" ]] || { echo "[FATAL] 单序列脚本不存在: $SINGLE" >&2; exit 2; }
mkdir -p "$OUT" || exit 2
LOGDIR="$OUT/_logs"; mkdir -p "$LOGDIR"
TMPROOT="${TMPROOT:-/tmp/stm_batch_tmp}"; mkdir -p "$TMPROOT"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

# ---- 组装序列清单 ----
if [[ ${#ONLY[@]} -gt 0 ]]; then
  printf '%s\n' "${ONLY[@]}" > "$LOGDIR/seqs.selected"
else
  grep -vE '^\s*(#|$)' "$SEQ_LIST" | sed 's/[[:space:]]*$//' | sort -u > "$LOGDIR/seqs.all"
fi
LIST="$LOGDIR/seqs.all"
[[ ${#ONLY[@]} -gt 0 ]] && LIST="$LOGDIR/seqs.selected"
NSEQ=$(wc -l < "$LIST")
echo "[BATCH] 序列数=$NSEQ  并发=$JOBS  输出=$OUT"

# ---- 单序列任务体（跑在子进程里） ----
run_one() {
  local seq="$1"
  local src="$SRC_ROOT/$seq.h264"
  local jpg="$JPG_ROOT/$seq"
  local o="$OUT/${seq}_original_1920x1080.mp4"
  local l="$OUT/${seq}_640x512_lossless_gray.mp4"
  local s="$OUT/${seq}_640x512.mp4"
  local log="$LOGDIR/$seq.log"

  {
    echo "[SEQ] $seq  start=$(date -Is)"
    if [[ $FORCE -eq 0 && -f "$o" && -f "$l" && -f "$s" ]]; then
      # 三件套都必须能解码出且帧数一致，否则视为残件重做
      local nf nl nsr bad=""
      nf=$(ffprobe -v error -select_streams v:0 -count_frames \
             -show_entries stream=nb_read_frames -of csv=p=0 "$s" 2>/dev/null | tail -1)
      nl=$(ffprobe -v error -select_streams v:0 -count_frames \
             -show_entries stream=nb_read_frames -of csv=p=0 "$l" 2>/dev/null | tail -1)
      nsr=$(ffprobe -v error -select_streams v:0 -count_frames \
              -show_entries stream=nb_read_frames -of csv=p=0 "$o" 2>/dev/null | tail -1)
      [[ -n "$nf" && "$nf" -gt 0 ]] || bad="stream:帧数=$nf"
      [[ -n "$nl" && "$nl" == "$nf" ]] || bad="$bad lossless:帧数=$nl"
      [[ -n "$nsr" && "$nsr" == "$nf" ]] || bad="$bad original:帧数=$nsr"
      if [[ -z "$bad" ]]; then
        echo "[SEQ] $seq  SKIP（三件套已齐全且帧数一致: $nf）"
        echo "[SEQ] $seq  done=$(date -Is)"; exit 0
      fi
      echo "[SEQ] $seq  存在但不合格（$bad）⇒ 重做"
    fi
    if [[ ! -f "$src" ]]; then echo "[FATAL] $seq 缺原始裸流: $src"; exit 1; fi
    if [[ ! -d "$jpg" ]]; then echo "[FATAL] $seq 缺 JPG 帧目录: $jpg"; exit 1; fi
    # 原子落位：先在临时目录产出，三件套齐了再 mv 进交付目录
    #   （否则交付目录会短暂出现半成品，下游一拉就拿到坏文件）
    local tmp="$TMPROOT/$seq"
    rm -rf "$tmp"; mkdir -p "$tmp"
    bash "$SINGLE" "$src" "$tmp" "$jpg" || { echo "[FATAL] $seq 转换失败"; rm -rf "$tmp"; exit 1; }
    for f in "$o" "$l" "$s"; do
      [[ -s "$tmp/$(basename "$f")" ]] || { echo "[FATAL] $seq 缺产物 $(basename "$f")"; rm -rf "$tmp"; exit 1; }
    done
    for f in "$o" "$l" "$s"; do
      mv -f "$tmp/$(basename "$f")" "$f" || { echo "[FATAL] $seq 落位失败 $f"; exit 1; }
    done
    rm -rf "$tmp"
    local nf nsr
    nf=$(ffprobe -v error -select_streams v:0 -count_frames \
           -show_entries stream=nb_read_frames -of csv=p=0 "$s" 2>/dev/null | tail -1)
    nsr=$(ffprobe -v error -select_streams v:0 -count_frames \
            -show_entries stream=nb_read_frames -of csv=p=0 "$o" 2>/dev/null | tail -1)
    [[ -n "$nf" && "$nf" == "$nsr" ]] || { echo "[FATAL] $seq 帧数不一致: stream=$nf original=$nsr"; exit 1; }
    echo "[SEQ] $seq  OK frames=$nf  done=$(date -Is)"
  } >>"$log" 2>&1
  return $?
}
export -f run_one
export OUT LOGDIR SRC_ROOT JPG_ROOT SINGLE FORCE TMPROOT

# ---- 并发执行 ----
RC=0
: > "$LOGDIR/batch.fail"
while IFS= read -r seq; do
  [[ -z "$seq" ]] && continue
  echo "$seq"
done < "$LIST" | xargs -d '\n' -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} || RC=1

# ---- 收口对账：清单里的每个序列都必须有齐全三件套 ----
MISS=0
while IFS= read -r seq; do
  [[ -z "$seq" ]] && continue
  for f in "$OUT/${seq}_original_1920x1080.mp4" \
           "$OUT/${seq}_640x512_lossless_gray.mp4" \
           "$OUT/${seq}_640x512.mp4"; do
    [[ -s "$f" ]] || { echo "[MISS] $f" | tee -a "$LOGDIR/batch.fail"; MISS=$((MISS+1)); }
  done
  # 三件套帧数必须一致（截断的残件在这里会被抓出来）
  nf=$(ffprobe -v error -select_streams v:0 -count_frames \
         -show_entries stream=nb_read_frames -of csv=p=0 \
         "$OUT/${seq}_640x512.mp4" 2>/dev/null | tail -1)
  nl=$(ffprobe -v error -select_streams v:0 -count_frames \
         -show_entries stream=nb_read_frames -of csv=p=0 \
         "$OUT/${seq}_640x512_lossless_gray.mp4" 2>/dev/null | tail -1)
  nsr=$(ffprobe -v error -select_streams v:0 -count_frames \
          -show_entries stream=nb_read_frames -of csv=p=0 \
          "$OUT/${seq}_original_1920x1080.mp4" 2>/dev/null | tail -1)
  if [[ -z "$nf" || "$nf" -le 0 || "$nl" != "$nf" || "$nsr" != "$nf" ]]; then
    echo "[FRAME-MISMATCH] $seq stream=$nf lossless=$nl original=$nsr" | tee -a "$LOGDIR/batch.fail"
    MISS=$((MISS+1))
  fi
done < "$LIST"
if [[ $MISS -gt 0 ]]; then
  echo "[BATCH][FAIL] 缺 $MISS 个产物"; exit 1
fi
echo "[BATCH][OK] $NSEQ 个序列 × 3 件套齐全  输出=$OUT  完成=$(date -Is)"
exit $RC
