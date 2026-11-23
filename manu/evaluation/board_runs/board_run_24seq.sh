#!/bin/bash
# board_run_24seq.sh -- 服务器侧 24 序板端全量验证驱动（RK3588 vs 服务器参考）
#
# 执行机：服务器 test-Super-Server（huangzhe@192.168.99.40:32222）
# 板卡：  RK3588 evm3588（服务器 ~/.ssh/config 里的 `ptq-board` 别名 → 192.168.0.64）
#
# 每序串行（板卡 NPU 串行铁律：同一时刻只允许 1 个 NPU 上下文）：
#   ① 服务器原始 IR JPG --tar/ssh--> 板卡 /userdata/lq24/<SEQ>/
#   ② 板卡 ffmpeg -vf format=gray --> /userdata/lq24_bmp/<SEQ>/frame_%06d.bmp（8bit 灰度、无损）
#   ③ 板卡 mosaic_track_dump（--tag-mask 7 --cand-dedup-radius 20 --thr 0.02）
#        --> /userdata/lq24_out/<SEQ>_{cands,tracked,fused}.tsv
#   ④ 服务器 scp 拉回 TSV --> 板卡删帧（腾出 /userdata）
#
# 用法（服务器上）：
#   bash board_run_24seq.sh --list                 # 打印计划（不执行）
#   bash board_run_24seq.sh --seq VIDEO00009_19700101_001531   # 只跑一序（链路验证）
#   bash board_run_24seq.sh                        # 跑全部 24 序（按帧数升序）
#   bash board_run_24seq.sh --resume               # 跳过已有 TSV 的序（断点续跑）
#
# 关键口径：
#   * 帧号：mosaic_track_dump 按文件名排序后用**下标**当帧号 ⇒ BMP 必须连续零基
#     （ffmpeg 用 `-start_number 0` 保证 frame_000000.bmp 起）。
#   * --frames 必须传该序真实帧数（工具默认 400，会静默只跑前 400 帧）。
#   * 合并半径用交付值 20 px，**不能**照搬服务器的 768 px（那会让每帧只剩 1 个候选）。
set -uo pipefail

BOARD=${BOARD:-ptq-board}
JPG_ROOT=${JPG_ROOT:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}
OUT_ROOT=${OUT_ROOT:-/mnt/data/siping/runs/trial0080_board24_20261120}
R_JPG=${R_JPG:-/userdata/lq24}
R_BMP=${R_BMP:-/userdata/lq24_bmp}
R_OUT=${R_OUT:-/userdata/lq24_out}
B_OUT=${B_OUT:-/root/merge_20261120/out}
MODEL=${MODEL:-/root/mosaic/runtime/ir_uav_det.rknn}
KERNELS=${KERNELS:-/root/mosaic/runtime/kernels}
TAG_MASK=${TAG_MASK:-7}
DEDUP=${DEDUP:-20}
THR=${THR:-0.02}

TSV_DIR=$OUT_ROOT/tsv
LOG_DIR=$OUT_ROOT/logs
mkdir -p "$TSV_DIR" "$LOG_DIR"
LOG=$LOG_DIR/board_run_24seq.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- 板卡远端执行 ----
bsh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$BOARD" "$@"; }

usage() { sed -n '2,30p' "$0"; exit 0; }

ONLY=""
RESUME=0
LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --seq) ONLY=${2:-}; shift 2;;
    --resume) RESUME=1; shift;;
    --list) LIST=1; shift;;
    -h|--help) usage;;
    *) echo "未知参数：$1" >&2; usage;;
  esac
done

# ---- 序列清单（按帧数升序：先小序验证链路，早出结果）----
ALL_SEQS=(
  VIDEO00009_19700101_001531 VIDEO00029_19700101_014053 VIDEO00030_19700101_014144
  VIDEO00036_19700101_015740 VIDEO00033_19700101_015000 VIDEO00034_19700101_015052
  VIDEO00009_19700101_003002 VIDEO00006_19700101_000703 VIDEO00010_19700101_001550
  VIDEO00008_19700101_001307 VIDEO00031_19700101_014227 VIDEO00011_19700101_001810
  VIDEO00003_19700101_002111 VIDEO00007_19700101_000818 VIDEO00032_19700101_014453
  VIDEO00028_19700101_013631 VIDEO00035_19700101_015138 VIDEO00007_19700101_001324
  VIDEO00002_19700101_001415 VIDEO00004_19700101_002438 VIDEO00012_19700101_002251
  VIDEO00005_19700101_002959 VIDEO00006_19700101_000322 VIDEO00008_19700101_002001
)

declare -a PLAN=()
for s in "${ALL_SEQS[@]}"; do
  [ -n "$ONLY" ] && [ "$s" != "$ONLY" ] && continue
  n=$(ls "$JPG_ROOT/$s"/frame_*.jpg 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { say "[WARN] $s 没有原始 JPG，跳过"; continue; }
  PLAN+=("$s:$n")
done

if [ "$LIST" = 1 ]; then
  echo "序                                   帧数"
  for p in "${PLAN[@]}"; do echo "${p%:*}  ${p##*:}"; done
  echo "合计帧数 = $(for p in "${PLAN[@]}"; do echo "${p##*:}"; done | paste -sd+ | bc)"
  exit 0
fi

say "===== board_run_24seq 启动  序数=${#PLAN[@]}  板卡=$BOARD ====="
bsh 'df -h /userdata | tail -1; md5sum '"$B_OUT"'/libmosaic_pipeline.so' 2>&1 | tee -a "$LOG"

for entry in "${PLAN[@]}"; do
  SEQ=${entry%:*}; N=${entry##*:}
  say "----- $SEQ  ($N 帧) -----"

  if [ "$RESUME" = 1 ] && [ -s "$TSV_DIR/${SEQ}_fused.tsv" ]; then
    say "[SKIP] $SEQ 已有 _fused.tsv，断点续跑跳过"; continue
  fi

  # ① 传原始 JPG（tar 流式，不在服务器落中间文件）
  T0=$(date +%s)
  bsh "mkdir -p '$R_JPG/$SEQ' '$R_BMP/$SEQ' '$R_OUT'" || { say "[FATAL] $SEQ 板卡建目录失败"; continue; }
  tar -C "$JPG_ROOT/$SEQ" -cf - . | ssh -o BatchMode=yes "$BOARD" "tar -C '$R_JPG/$SEQ' -xf -"
  T1=$(date +%s); say "[$SEQ] ① 传输 JPG 用时 $((T1-T0))s"

  # ② 板卡转 8bit 灰度 BMP（连续零基）
  bsh "ffmpeg -hide_banner -loglevel error -y -i '$R_JPG/$SEQ/frame_%06d.jpg' -vf format=gray -vsync 0 -start_number 0 '$R_BMP/$SEQ/frame_%06d.bmp'" \
    >"$LOG_DIR/$SEQ.ffmpeg.log" 2>&1
  RC=$?
  T2=$(date +%s); say "[$SEQ] ② 转 BMP rc=$RC 用时 $((T2-T1))s"
  if [ $RC -ne 0 ]; then say "[FATAL] $SEQ ffmpeg 失败，见 $LOG_DIR/$SEQ.ffmpeg.log"; continue; fi

  NB=$(bsh "ls '$R_BMP/$SEQ'/frame_*.bmp 2>/dev/null | wc -l")
  say "[$SEQ] BMP 帧数=$NB（期望 $N）"
  if [ "$NB" != "$N" ]; then say "[FATAL] $SEQ BMP 帧数与源不符（$NB vs $N）"; bsh "rm -rf '$R_JPG/$SEQ' '$R_BMP/$SEQ'"; continue; fi

  # ③ 板卡 NPU 推理 + 融合（串行铁律：跑前确认无其它 NPU 上下文）
  BUSY=$(bsh "ps -eo comm | grep -E 'rknn|mosaic|e2e' | grep -v grep" | tr -d '\r')
  if [ -n "$BUSY" ]; then say "[FATAL] $SEQ 板卡仍有 NPU 进程：$BUSY —— 串行铁律，不抢跑"; bsh "rm -rf '$R_JPG/$SEQ' '$R_BMP/$SEQ'"; continue; fi

  T3=$(date +%s)
  bsh "export LD_PRELOAD=/root/mosaic/libstatcompat.so; export LD_LIBRARY_PATH=$B_OUT:/root/mosaic/opencv:/root/mosaic:/root/hm_speed/lib:/lib:/lib/aarch64-linux-gnu:/usr/lib; $B_OUT/mosaic_track_dump --dir '$R_BMP/$SEQ' --model '$MODEL' --kernel-dir '$KERNELS' --seq '$SEQ' --out-prefix '$R_OUT/$SEQ' --frames $N --npu-workers 1 --thr $THR --fusion 1 --tag-mask $TAG_MASK --cand-dedup-radius $DEDUP" \
    >"$LOG_DIR/$SEQ.dump.log" 2>&1
  RC=$?
  T4=$(date +%s)
  say "[$SEQ] ③ 板端推理 rc=$RC 用时 $((T4-T3))s ($(awk -v n=$N -v a=$T4 -v b=$T3 'BEGIN{printf "%.2f", n/(a-b)}') fps)"
  if [ $RC -ne 0 ]; then say "[FATAL] $SEQ dump 失败，见 $LOG_DIR/$SEQ.dump.log"; bsh "rm -rf '$R_JPG/$SEQ' '$R_BMP/$SEQ'"; continue; fi

  # ④ 拉回 TSV + 板卡删帧
  scp -q "$BOARD:$R_OUT/$SEQ"'_cands.tsv' "$BOARD:$R_OUT/$SEQ"'_tracked.tsv' "$BOARD:$R_OUT/$SEQ"'_fused.tsv' "$TSV_DIR/" 2>>"$LOG"
  bsh "rm -rf '$R_JPG/$SEQ' '$R_BMP/$SEQ' '$R_OUT/$SEQ'_cands.tsv '$R_OUT/$SEQ'_tracked.tsv '$R_OUT/$SEQ'_fused.tsv"
  T5=$(date +%s); say "[$SEQ] ④ 拉回+清理 用时 $((T5-T4))s  累计 $((T5-T0))s"
  say "[$SEQ] TSV: $(ls -l "$TSV_DIR" | grep -E "${SEQ}_(cands|tracked|fused)\\.tsv" | awk '{printf "%s(%sB) ", $9, $5}')"
done

say "===== board_run_24seq 结束 ====="
