#!/bin/bash
# board_xcomp_ab_check.sh -- 板端 A/B：宿主机交叉编译版 `.so` vs 板卡原生版 `.so` 的行为等价证明。
#
# 执行机：服务器 test-Super-Server（huangzhe@192.168.99.40:32222）→ 板卡 ptq-board（192.168.0.64）
# 用法：
#   bash board_xcomp_ab_check.sh <SEQ> [XCOMP_DIR]
#     <SEQ>       要验证的序列（默认用已跑过的同一序，保证帧完全一致）
#     XCOMP_DIR  宿主机交叉编译产物目录（默认 /root/xcomp/out，需已推上板）
#
# 判定：两次运行的 `_cands.tsv` / `_tracked.tsv` / `_fused.tsv` **md5 逐字节相同** ⇒ 行为等价。
#
# ⚠️ 板卡串行铁律：同一时刻只允许 1 个 NPU 上下文 ⇒ 跑前必须确认无 rknn/mosaic 进程。
set -uo pipefail

SEQ=${1:?sequence}
XCOMP_DIR=${2:-/root/xcomp/out}
BOARD=${BOARD:-ptq-board}
JPG_ROOT=${JPG_ROOT:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}
OUT_ROOT=${OUT_ROOT:-/mnt/data/siping/runs/trial0080_board24_20261120}
R_JPG=/userdata/ab_jpg
R_BMP=/userdata/ab_bmp
R_OUT=/userdata/ab_out
NATIVE_OUT=/root/merge_20261120/out
LOG=$OUT_ROOT/logs/ab_xcomp_$SEQ.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
bsh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$BOARD" "$@"; }

N=$(ls "$JPG_ROOT/$SEQ"/frame_*.jpg 2>/dev/null | wc -l)
[ "$N" -gt 0 ] || { say "[FATAL] $SEQ 没有原始 JPG"; exit 1; }
say "===== A/B 开始  seq=$SEQ frames=$N  xcomp=$XCOMP_DIR ====="

BUSY=$(bsh "ps -eo comm | grep -E 'rknn|mosaic|e2e' | grep -v grep" | tr -d '\r')
if [ -n "$BUSY" ]; then say "[FATAL] 板卡仍有 NPU 进程：$BUSY —— 串行铁律，不抢跑"; exit 1; fi

say "--- 板卡原生 .so ---"; bsh "md5sum $NATIVE_OUT/libmosaic_pipeline.so $NATIVE_OUT/mosaic_track_dump" | tee -a "$LOG"
say "--- 宿主机交叉编译 .so ---"; bsh "md5sum $XCOMP_DIR/libmosaic_pipeline.so $XCOMP_DIR/mosaic_track_dump" | tee -a "$LOG"

# ① 传帧 + 转 BMP（与主跑批同一路径，保证像素完全一致）
bsh "mkdir -p '$R_JPG/$SEQ' '$R_BMP/$SEQ' '$R_OUT'" || exit 1
tar -C "$JPG_ROOT/$SEQ" -cf - . | ssh -o BatchMode=yes "$BOARD" "tar -C '$R_JPG/$SEQ' -xf -"
bsh "ffmpeg -hide_banner -loglevel error -y -i '$R_JPG/$SEQ/frame_%06d.jpg' -vf format=gray -vsync 0 -start_number 0 '$R_BMP/$SEQ/frame_%06d.bmp'" >/dev/null 2>&1
say "BMP 帧数=$(bsh "ls '$R_BMP/$SEQ'/frame_*.bmp | wc -l")（期望 $N）"

run_dump() {  # $1=输出前缀 $2=二进制目录
  bsh "export LD_PRELOAD=/root/mosaic/libstatcompat.so; export LD_LIBRARY_PATH=$2:$NATIVE_OUT:/root/mosaic/opencv:/root/mosaic:/root/hm_speed/lib:/lib:/lib/aarch64-linux-gnu:/usr/lib; $2/mosaic_track_dump --dir '$R_BMP/$SEQ' --model /root/mosaic/runtime/ir_uav_det.rknn --kernel-dir /root/mosaic/runtime/kernels --seq '$SEQ' --out-prefix '$1' --frames $N --npu-workers 1 --thr 0.02 --fusion 1 --tag-mask 7 --cand-dedup-radius 20" >"$OUT_ROOT/logs/ab_${3}_$SEQ.dump.log" 2>&1
  echo "rc=$?"
}

# ② A 臂：板卡原生版
say "--- A 臂：板卡原生版 ---"
T0=$(date +%s); RA=$(run_dump "$R_OUT/${SEQ}__native" "$NATIVE_OUT" native); T1=$(date +%s)
RC_A=${RA#rc=}
say "A 臂 rc=$RC_A 用时 $((T1-T0))s"

# ③ B 臂：宿主机交叉编译版
say "--- B 臂：宿主机交叉编译版 ---"
T2=$(date +%s); RB=$(run_dump "$R_OUT/${SEQ}__xcomp" "$XCOMP_DIR" xcomp); T3=$(date +%s)
RC_B=${RB#rc=}
say "B 臂 rc=$RC_B 用时 $((T3-T2))s"

if [ "$RC_A" != "0" ] || [ "$RC_B" != "0" ]; then
  say "[FATAL] 有臂跑失败（native=$RC_A xcomp=$RC_B）"
  exit 1
fi

# ④ 逐字节比对
say "--- md5 比对 ---"
bsh "md5sum $R_OUT/${SEQ}__native_*.tsv $R_OUT/${SEQ}__xcomp_*.tsv" | tee -a "$LOG"
SAME=$(bsh "for f in cands tracked fused; do a=\$(md5sum $R_OUT/${SEQ}__native_\$f.tsv | cut -d' ' -f1); b=\$(md5sum $R_OUT/${SEQ}__xcomp_\$f.tsv | cut -d' ' -f1); if [ \"\$a\" = \"\$b\" ]; then echo \"\$f SAME \$a\"; else echo \"\$f DIFF native=\$a xcomp=\$b\"; fi; done")
echo "$SAME" | tee -a "$LOG"
if echo "$SAME" | grep -q DIFF; then say "[VERDICT] ❌ 不等价"; else say "[VERDICT] ✅ 行为等价（cands/tracked/fused 三份 TSV md5 全同）"; fi

# ⑤ 拉回 + 清理
mkdir -p "$OUT_ROOT/ab"
scp -q "$BOARD:$R_OUT/${SEQ}__"*.tsv "$OUT_ROOT/ab/" 2>/dev/null
bsh "rm -rf '$R_JPG/$SEQ' '$R_BMP/$SEQ' $R_OUT/${SEQ}__*.tsv"
say "===== A/B 结束 ====="
