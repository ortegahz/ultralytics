#!/usr/bin/env bash
# ===========================================================================
# run_hm_demo.sh -- serial end-to-end demo on the RK3588 board.
#
#   raw infrared sequence
#       -> [board] gmc_stream_ocl        (CPU fits + Mali fused tail)  -> features
#       -> [board] rknn_heatmap_smoke    (RKNN int8, Trial 0474 w/o P0) -> detections
#       -> [host] hm_demo_report.py       three-layer accuracy + speed roll-up
#
# SERIAL ON PURPOSE. No fused binary, no shared memory, no pipeline overlap:
# the two stages are already-certified binaries, and keeping them separate means
# each stage keeps its own honest timer, which is what makes the speed report
# attributable rather than merely fast.
#
# What "serial" costs and buys
# ----------------------------
#   cost : one extra round of feature I/O (512x512x3 BMP, ~786 KB/frame)
#   buy  : each stage is independently verified and independently timed, and
#          either can be swapped without touching the other
#
# Prerequisites
# -------------
#   * RK3588_PASSWORD exported (never written to disk)
#   * board reachable at $RK3588_HOST (default 192.168.0.64)
#   * cross-built binaries:
#       manu/pipeline/opencl/board/out/gmc_stream_ocl_arm64
#       manu/pipeline/npu/out/rknn_heatmap_smoke
#   * the feature kernel pair staged on the board at $BOARD_KERNEL_DIR
#
# Usage
# -----
#   ./run_hm_demo.sh --sequence 34_1 [--limit 0] [--th 0.25] [--skip-build]
#   ./run_hm_demo.sh --video /path/to/clip.mp4 --seq-name clip [--frames 300] [--id-base N]
#
# --video accepts any container cv2.VideoCapture reads. There is NO per-sequence
# raw .mp4 in this project -- the archived sequences are frame directories, and the
# only .mp4 files under the datasets (label_review.mp4, gt_review_*/..._gt_full.mp4)
# are review renders. The gt_ ones have boxes already drawn into the pixels, so they
# are NOT valid input for a detector; use --sequence for GT-aligned runs.
# ===========================================================================
set -euo pipefail

SEQ=""
VIDEO=""
SEQNAME=""
FRAMES=0
ID_BASE=0
LIMIT=0
TH=0.25
SKIP_BUILD=0
OUT_ROOT="${OUT_ROOT:-/media/manu/1TB-Volume/rknn_work/hm}"

# ---- fixed wiring (single source of truth; every path below derives from these)
RAW_ROOT_LOCAL="${RAW_ROOT_LOCAL:-/home/manu/mnt/data/siping/datasets/manu/anti-uav/train}"
FROZEN_ROOT="${FROZEN_ROOT:-/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val}"
LABEL_ROOT="${LABEL_ROOT:-/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/labels/val}"
NFS_LOCAL="/home/manu/nfs/npu/out/demo"
BOARD_NFS="/mnt/manu/npu/out/demo"
WEIGHTS_DEFAULT="/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_p0_nas/trial_0474/weights/best.pt"
MODEL="trial0474_hm_nop0_int8.rknn"
MODEL_MD5="ef302edb1b5aff1d376204bb7ad77550"
PUSH_OFFSET=1          # measured: dataset_id = push_index + 1 (see hm_index_probe.py)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE_ROOT="$(cd "$HERE/.." && pwd)"        # manu/pipeline  (this file lives in npu/)
PY="${PY:-/media/manu/1TB-Volume/conda-envs/rknn230/bin/python}"

usage() { sed -n '2,40p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --sequence)   SEQ="$2"; shift 2 ;;
    --video)      VIDEO="$2"; shift 2 ;;
    --seq-name)   SEQNAME="$2"; shift 2 ;;
    --frames)     FRAMES="$2"; shift 2 ;;
    --id-base)    ID_BASE="$2"; shift 2 ;;
    --limit)      LIMIT="$2"; shift 2 ;;
    --th)         TH="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help)    usage 0 ;;
    *) echo "[ERROR] unknown arg: $1" >&2; usage 2 ;;
  esac
done
if [ -n "$VIDEO" ]; then
  [ -n "$SEQNAME" ] || SEQNAME="$(basename "${VIDEO%.*}")"
  [ -f "$VIDEO" ] || { echo "[ERROR] no such video: $VIDEO" >&2; exit 1; }
  SEQ="$SEQNAME"
elif [ -z "$SEQ" ]; then
  echo "[ERROR] one of --sequence or --video is required" >&2; usage 2
fi

: "${RK3588_PASSWORD:?RK3588_PASSWORD must be set (never written to disk)}"
BR="$PIPE_ROOT/opencl/board/board_run.py"
[ -f "$BR" ] || { echo "[ERROR] board_run.py not found at $BR" >&2; exit 1; }

WORK="$OUT_ROOT/${SEQ}"
NFS_DIR="$NFS_LOCAL/$SEQ"
BOARD_DIR="$BOARD_NFS/$SEQ"
mkdir -p "$WORK" "$NFS_DIR"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 0. build
if [ "$SKIP_BUILD" -eq 0 ]; then
  say "0. 交叉编译板端二进制"
  ( cd "$PIPE_ROOT/opencl/board" && ./build_gmc_ocl_arm64.sh )
  ( cd "$PIPE_ROOT/npu" && ./build_rknn_heatmap_arm64.sh )
fi

# ---------------------------------------------------------------- 1. raw -> BMP
say "1. host: 输入 -> 8bit 灰度 BMP（板端 OpenCV 只有内置 BMP 编解码）"
RAWDIR="$NFS_DIR/raw/$SEQ"
mkdir -p "$RAWDIR"
if [ -n "$VIDEO" ]; then
  echo "[video] $VIDEO  (--frames ${FRAMES:-0} = 不限)"
  "$PY" - "$VIDEO" "$RAWDIR" "$SEQ" "$FRAMES" <<'PYEOF'
import os, sys, cv2
video, dst, seq, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
c = cv2.VideoCapture(video)
if not c.isOpened():
    raise SystemExit(f"[ERROR] cannot open {video}")
n = 0
while cap <= 0 or n < cap:
    ok, bgr = c.read()
    if not ok:
        break
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(dst, f"{seq}_{n:06d}.bmp"), gray)
    n += 1
c.release()
if n == 0:
    raise SystemExit("[ERROR] decoded zero frames")
print(f"[video] {n} frames -> {dst}  ({gray.shape[1]}x{gray.shape[0]})")
PYEOF
else
  SRC_DIR="$RAW_ROOT_LOCAL/$SEQ"
  [ -d "$SRC_DIR" ] || { echo "[ERROR] no such sequence: $SRC_DIR" >&2; exit 1; }
  "$PY" - "$SRC_DIR" "$RAWDIR" <<'PYEOF'
import os, sys, cv2
src, dst = sys.argv[1], sys.argv[2]
fs = sorted(f for f in os.listdir(src) if f.lower().endswith((".png", ".jpg", ".bmp", ".tif")))
if not fs:
    raise SystemExit(f"[ERROR] no image frames in {src}")
for f in fs:
    im = cv2.imread(os.path.join(src, f), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise SystemExit(f"[ERROR] unreadable {f}")
    cv2.imwrite(os.path.join(dst, os.path.splitext(f)[0] + ".bmp"), im)
print(f"[bmp] {len(fs)} frames -> {dst}")
PYEOF
fi

# ---------------------------------------------------------------- 2. deploy
say "2. 推送二进制与 OpenCL kernel（走 SSH，禁走 NFS）"
python3 "$BR" --put "$PIPE_ROOT/opencl/board/out/gmc_stream_ocl_arm64:/root/gmc_stream_ocl"
python3 "$BR" --put "$PIPE_ROOT/npu/out/rknn_heatmap_smoke:/root/rknn_heatmap_smoke"
python3 "$BR" --put "$PIPE_ROOT/opencl/kernels/warp_median_fused.cl:/root/kernels/warp_median_fused.cl"
python3 "$BR" --put "$PIPE_ROOT/opencl/kernels/sortnet_generated.inc:/root/kernels/sortnet_generated.inc"

say "2b. 板卡状态（频率必须随任何时延数字一起报）"
python3 "$BR" --exec '
  echo -n "npu governor/cur_freq : "; cat /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null | tr -d "\n"; echo -n " @ "; cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null
  echo -n "cpu0 / cpu4           : "; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null | tr -d "\n"; echo -n " / "; cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq 2>/dev/null
  df -h / | tail -1'

# ---------------------------------------------------------------- 3. GMC stage
say "3. BOARD 阶段 A：gmc_stream_ocl（原始红外 -> 三通道特征）"
python3 "$BR" --exec "
cd $BOARD_DIR && mkdir -p feat &&
/root/gmc_stream_ocl --raw-root raw --sequence $SEQ ${LIMIT:+--limit $LIMIT} \
  --out-dir feat --out-ext bmp --fused gpu --kernel-dir /root/kernels \
  --timing --profile-csv gmc_profile.csv > gmc_run.log 2>&1
echo GMC_RC=\$?
grep -E 'END-TO-END TOTAL|budget|MD5-SUM|\[SEQ\]' gmc_run.log"

# ---------------------------------------------------------------- 4. naming
say "4. host: push 索引 -> 帧 id"
# The board names its outputs by PUSH INDEX. Two cases:
#   * frozen features exist for this sequence  -> the id list is the dataset's,
#     and dataset_id = push_index + PUSH_OFFSET (measured, see hm_index_probe.py);
#   * video input, no frozen set              -> identity, id = push_index.
# In both cases the symlink target must be RELATIVE: the board mounts this tree at
# a different prefix, so a symlink written with an absolute host path resolves to
# nothing over there and stbi_load fails silently.
"$PY" - "$NFS_DIR/feat" "$NFS_DIR/feat_named" "$BOARD_NFS/$SEQ/feat_named" \
       "$SEQ" "$PUSH_OFFSET" "$FROZEN_ROOT" "$ID_BASE" <<'PYEOF'
import os, sys
from pathlib import Path
feat, named, board_named, seq, off, frozen, id_base = sys.argv[1:8]
off, id_base = int(off), int(id_base)
frozen_ids = sorted(int(Path(p).stem.split("__")[-1]) for p in os.listdir(frozen)
                    if p.startswith(seq + "__"))
if frozen_ids:
    ids, off = frozen_ids, off
    how = f"dataset ids {ids[0]}..{ids[-1]}, id = push + {off}"
else:
    n = len([f for f in os.listdir(feat) if f.lower().endswith(".bmp")])
    # --id-base lets a video clip carry the SEQUENCE frame numbering, so that
    # labels/val/<seq>__<id>.txt is found and the arms stay GT-alignable.
    # push index is 0-based, the id is --id-base based, so the symlink offset is
    # id_base -- NOT 0. (Getting this wrong points the last ids at files that
    # were never written, and the board dies with a bare "stbi_load failed".)
    ids, off = list(range(id_base, id_base + n)), id_base
    how = f"video input -> id = push + {id_base} ({n} frames, ids {ids[0]}..{ids[-1]})"
os.makedirs(named, exist_ok=True)
for fid in ids:
    d = os.path.join(named, f"{seq}__{fid:06d}.bmp")
    if os.path.lexists(d):
        os.unlink(d)
    os.symlink(f"../feat/{seq}_{fid - off:06d}.bmp", d)
with open(os.path.join(named, "list_board.txt"), "w") as f:
    f.write("\n".join(f"{board_named}/{seq}__{i:06d}.bmp" for i in ids) + "\n")
print(f"[map] {len(ids)} frames, {how}")
PYEOF

# ---------------------------------------------------------------- 5. NPU stage
say "5. BOARD 阶段 B：rknn_heatmap_smoke（特征 -> RKNN int8 -> 检测）"
python3 "$BR" --exec "
cd /mnt/manu/npu &&
/root/rknn_heatmap_smoke data/$MODEL --batch $BOARD_DIR/feat_named/list_board.txt \
  --th $TH --stride 2 --detections-out $BOARD_DIR/npu_det.tsv \
  > $BOARD_DIR/npu_run.log 2> $BOARD_DIR/npu_run.err
echo NPU_RC=\$?
grep -E 'clock sanity|null control|determinism|TOTAL ' $BOARD_DIR/npu_run.log
head -2 $BOARD_DIR/npu_run.err"

# ---------------------------------------------------------------- 6. CPU arm
# Whether an accuracy alignment is POSSIBLE is decided by the labels actually
# existing for this id range -- not by which flag was used. A video encoded from
# a labelled sequence (see hm_make_video.py) is fully GT-alignable; a review
# render is not.
HAS_GT=0
GT_FOUND=0
for t in "$LABEL_ROOT"/"$SEQ"__*.txt; do
  [ -e "$t" ] && GT_FOUND=1 && break
done
if [ "$GT_FOUND" -eq 1 ]; then
  HAS_GT=1
fi

if [ "$HAS_GT" -eq 0 ]; then
  say "6. CPU 权威臂 —— 跳过"
  echo "[ref] $LABEL_ROOT 下没有 $SEQ__*.txt，无配对 GT，只报速度与检出分布。"
  echo "      需要精度对齐请用 --sequence，或先用 hm_make_video.py 把带标签的序列编成无损视频。"
else
  say "6. CPU 权威臂（$LABEL_ROOT 找到 $SEQ 标签，三层精度对齐可用）"
  if [ -f "$WORK/ref_cpu/detections.json" ]; then
    echo "[ref] 已有 $WORK/ref_cpu/detections.json，跳过（如需重跑请先删除）"
  else
    echo "[ref] 在训练服务器跑（torch+ultralytics+GPU）；本地无该环境"
    echo "      参考命令："
    echo "        cd /tmp/pycharm_project_10ae9e2e && ~/anaconda3/envs/uav/bin/python -m manu.pipeline.npu.hm_reference \\"
    echo "          --frames <list> --out <ref_cpu> --th $TH --ablate-p0 --no-dump-tensors \\"
    echo "          --label-root $LABEL_ROOT"
    echo "      帧清单：序列路径 = $FROZEN_ROOT/$SEQ__*.png；视频路径 = hm_make_cpu_features.py 的输出"
  fi
fi

# ---------------------------------------------------------------- 7. score
say "7. host: 检出级对账"
if [ "$HAS_GT" -eq 0 ]; then
  echo "[skip] 无 GT，跳过 TP/FP/FN 对账"
  "$PY" - "$NFS_DIR/npu_det.tsv" <<'PYEOF'
import sys
from collections import Counter
c, n = Counter(), 0
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    p = line.split("\t")
    k = int(p[1]); c[k] += 1; n += 1
print(f"[det] {n} 帧检出数分布: " + ", ".join(f"{k}个检出:{v}帧" for k, v in sorted(c.items())))
PYEOF
else
"$PY" "$PIPE_ROOT/npu/hm_score_board_det.py" \
  --tsv "$NFS_DIR/npu_det.tsv" --ref "$WORK/ref_cpu" --dist 8 \
  --out "$WORK/board_perframe.json"
fi

say "8. host: 精度对齐 + 速度报告"
if [ "$HAS_GT" -eq 1 ]; then
"$PY" "$PIPE_ROOT/npu/hm_demo_report.py" \
  --sequence "$SEQ" \
  --board-feat "$NFS_DIR/feat" --frozen-feat "$FROZEN_ROOT" \
  --ref "$WORK/ref_cpu" --perframe "$WORK/board_perframe.json" \
  --tsv "$NFS_DIR/npu_det.tsv" \
  --gmc-csv "$NFS_DIR/gmc_profile.csv" --gmc-log "$NFS_DIR/gmc_run.log" \
  --npu-log "$NFS_DIR/npu_run.log" \
  --push-offset "$PUSH_OFFSET" --gmc-warmup 100 --model-name "$MODEL" \
  | tee "$WORK/report_${SEQ}.txt"
else
# no GT -> speed only, via the report's own flag (never by filtering its output:
# a swallowed traceback would still be a failed run wearing a success hat)
"$PY" "$PIPE_ROOT/npu/hm_demo_report.py" --sequence "$SEQ" --speed-only \
  --tsv "$NFS_DIR/npu_det.tsv" \
  --gmc-csv "$NFS_DIR/gmc_profile.csv" --gmc-log "$NFS_DIR/gmc_run.log" \
  --npu-log "$NFS_DIR/npu_run.log" --gmc-warmup 100 --model-name "$MODEL" \
  | tee "$WORK/speed_${SEQ}.txt"
fi

say "完成"
{
  echo "产物："
  if [ "$HAS_GT" -eq 1 ]; then
    echo "  $WORK/report_${SEQ}.txt          报告（L1 特征 / L2 检出 / L3 配对 + 速度）"
    echo "  $WORK/board_perframe.json        逐帧对账（含 CPU 侧并列）"
  else
    echo "  $WORK/speed_${SEQ}.txt           报告（仅速度；本输入无配对 GT，未做精度对齐）"
  fi
  echo "  $NFS_DIR/npu_det.tsv             板端检测 TSV"
  echo "  $NFS_DIR/gmc_profile.csv         板端 GMC 分阶段账本"
  echo "  $NFS_DIR/gmc_run.log             板端 GMC 自报计时"
  echo "  $NFS_DIR/npu_run.log             板端 NPU 自报计时"
  echo "  $NFS_DIR/feat/                   板端特征 BMP"
  echo
  echo "模型：$MODEL（md5 $MODEL_MD5，去 P0 Highway，0 个 CPU 计算算子）"
}
