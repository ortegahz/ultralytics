#!/bin/bash
# ==============================================================================
# board_measure_from_server.sh —— **在服务器上**独立完成「结构 → 板卡实测 NPU ms」
#
# 为什么需要这个脚本（2026-11-06）
# -------------------------------
# 宿主今天 18:30 关机、板卡同时断电。⇒ 周末训出来的最终模型，**在宿主上无法再测速**。
# ⇒ 验证过：服务器 **自带 rknn-toolkit2 2.3.0 / 2.3.2**（conda 环境 rknn230 / rknn），
#   且**能直连板卡**（ssh -i ~/.ssh/id_board root@192.168.0.64）⇒
#   「导出 ONNX → 转 RKNN int8 → 上板实测」可以在服务器独立完成，**不依赖宿主**。
#
# 用法（服务器，conda uav）：
#     bash manu/training/board_measure_from_server.sh \
#         --ckpt /mnt/data/siping/runs/npu20ms/trials/trial_0002/weights/epoch20.pt \
#         --out-root /mnt/data/siping/runs/npu20ms/npu_speed
#
# ⚠️ 板卡断电后本脚本无法运行（板卡 2026-11-06 18:30 断电）。
#    ⇒ 若板卡已断电，速度请查结构→速度表（结构决定速度，与权重无关）。
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
BOARD="${RKNPU_BOARD:-root@192.168.0.64}"
BOARD_PORT="${RKNPU_BOARD_PORT:-22}"
BOARD_DIR="${RKNPU_BOARD_DIR:-/root/hm_speed}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_board}"
GOOD_MD5_EXPECTED="56e425dcc7b59ab0e1aec685b178b99f"
GOOD_LIB="${HM_GOOD_LIB:-/run/media/manu/SSD960-LOCAL/nfs/npu/librknnrt.so}"
LOOPS="${HM_LOOPS:-30}"
WARMUP="${HM_WARMUP:-10}"
SETTLE="${HM_SETTLE:-5}"
CALIB_LIST="${CALIB_LIST:-/mnt/data/siping/runs/nogmc_struct2/npu_speed/calib_nogmc_train.txt}"

CKPT=""
OUT_ROOT="/mnt/data/siping/runs/npu20ms/npu_speed"
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    *) echo "未知参数：$1"; exit 1 ;;
  esac
done

# ⚠️ 标定列表里的图片路径是**宿主路径**（/home/manu/mnt/...），服务器上没有这个挂载
#    ⇒ 在服务器上必须把它改写成服务器路径，否则转换会报
#    「200/200 calibration images do not exist」。
# ⇒ 这里自动生成一份服务器路径版本（宿主路径 → 服务器路径）。
#    ⚠️ 必须在参数解析**之后**算：OUT_ROOT 可能被 --out-root 覆盖。
CALIB_LIST_SRV="$OUT_ROOT/calib_nogmc_train_server.txt"

[ -n "$CKPT" ] || { echo "[FATAL] 必须给 --ckpt"; exit 1; }
[ -f "$CKPT" ] || { echo "[FATAL] 找不到 ckpt：$CKPT"; exit 1; }
[ -f "$CALIB_LIST" ] || { echo "[FATAL] 找不到标定列表：$CALIB_LIST"; exit 1; }

# ---- 生成服务器路径版标定列表（宿主路径 → 服务器路径）----
mkdir -p "$OUT_ROOT"
sed 's#/home/manu/mnt/data/#/mnt/data/#g' "$CALIB_LIST" > "$CALIB_LIST_SRV"
MISSING=$(while read -r p; do [ -f "$p" ] || echo "$p"; done < "$CALIB_LIST_SRV" | head -3)
if [ -n "$MISSING" ]; then
  echo "[FATAL] 服务器路径版标定列表仍有缺失文件，例如："
  echo "$MISSING" | sed 's/^/        /'
  echo "       （检查 /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/images/train 是否存在）"
  exit 1
fi
echo "[标定] $CALIB_LIST_SRV（$(wc -l < "$CALIB_LIST_SRV") 张，服务器路径校验通过）"

cd "$PROJECT_ROOT" || exit 1
# ★ 用服务器的 rknn 环境（宿主已验证：rknn230 = 2.3.0，rknn = 2.3.2）
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh
# ⚠️ 具体用哪个环境在下面分步决定（导出=uav，转换=rknn）⇒ 这里**不提前激活**。

mkdir -p "$OUT_ROOT/onnx" "$OUT_ROOT/rknn_arms" "$OUT_ROOT/logs"
NAME="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt)"
ONNX="$OUT_ROOT/onnx/${NAME}.onnx"
RKNN="$OUT_ROOT/rknn_arms/${NAME}_int8.rknn"

SSH_OPTS="-p $BOARD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"
SCP_OPTS="-P $BOARD_PORT -i $SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"

echo "[1/5] 板卡连通性与 runtime 校验"
if ! ssh $SSH_OPTS "$BOARD" "echo ok" >/dev/null 2>&1; then
  echo "[FATAL] 连不上板卡（$BOARD）—— 板卡可能已断电（2026-11-06 18:30 断电）"; exit 1
fi
BOARD_MD5=$(ssh $SSH_OPTS "$BOARD" "md5sum $BOARD_DIR/lib/librknnrt.so 2>/dev/null | awk '{print \$1}'" 2>/dev/null | tr -d '\r')
if [ "$BOARD_MD5" != "$GOOD_MD5_EXPECTED" ]; then
  echo "[WARN] 板卡 runtime md5=$BOARD_MD5 ≠ 已知可用 $GOOD_MD5_EXPECTED"
  if [ -f "$GOOD_LIB" ]; then
    echo "       从 $GOOD_LIB 暂存已知可用 runtime…"
    ssh $SSH_OPTS "$BOARD" "mkdir -p $BOARD_DIR/lib" >/dev/null 2>&1
    scp $SCP_OPTS "$GOOD_LIB" "$BOARD:$BOARD_DIR/lib/librknnrt.so" || { echo "[FATAL] 暂存 runtime 失败"; exit 1; }
  else
    echo "[FATAL] 板卡 runtime 不对，且 $GOOD_LIB 不可达"; exit 1
  fi
fi

echo "[2/5] 导出 ONNX（conda uav）"
# ★★ **导出用 uav 环境、转换用 rknn 环境**（2026-11-06 实测）：
#    · conda `uav`      ⇒ 有 torch / PIL / numpy ⇒ 导出 ONNX 没问题
#    · conda `rknn230`  ⇒ 有 rknn-toolkit2 2.3.0，但**缺 PIL** ⇒ 在它里面跑导出会挂在
#      `ultralytics/utils/patches.py`（ModuleNotFoundError: No module named 'PIL'）
#    ⇒ 两个环境分工跑，别试图在一个环境里干完。
conda activate uav || { echo "[FATAL] conda activate uav 失败"; exit 1; }
[ -f "$ONNX" ] || python -u manu/pipeline/npu/hm_export_onnx.py \
  --weights "$CKPT" --out "$ONNX" --skip-check --ablate-p0 \
  > "$OUT_ROOT/logs/export_${NAME}.log" 2>&1 || {
    echo "[FATAL] 导出失败，见 $OUT_ROOT/logs/export_${NAME}.log"; tail -20 "$OUT_ROOT/logs/export_${NAME}.log"; exit 1; }
grep -E "^\[STRUCT\]|^\[model\] stride|^\[head_conv\]|^\[onnx" "$OUT_ROOT/logs/export_${NAME}.log" | sed 's/^/    /'

echo "[3/5] 转 RKNN int8（conda ${RKNN_ENV:-rknn230}）"
conda activate "${RKNN_ENV:-rknn230}" || { echo "[FATAL] conda activate ${RKNN_ENV:-rknn230} 失败"; exit 1; }
[ -f "$RKNN" ] || python -u manu/pipeline/npu/hm_convert_rknn.py \
  --onnx "$ONNX" --out "$RKNN" --calib-list "$CALIB_LIST_SRV" --calib-list-only \
  > "$OUT_ROOT/logs/convert_${NAME}.log" 2>&1 || {
    echo "[FATAL] 转换失败，见 $OUT_ROOT/logs/convert_${NAME}.log"; tail -20 "$OUT_ROOT/logs/convert_${NAME}.log"; exit 1; }

# stride 边车（板卡工具校验 input/grid ratio == 640/stride）
ST=$(grep -oE "^\[model\] stride=[0-9]+" "$OUT_ROOT/logs/export_${NAME}.log" | head -1 | grep -oE "[0-9]+$")
[ -n "$ST" ] && echo "$ST" > "$OUT_ROOT/rknn_arms/${NAME}_int8.stride"
echo "    stride=${ST:-?}  rknn=$(stat -c %s "$RKNN" 2>/dev/null) bytes"

echo "[4/5] 暂存到板卡"
ssh $SSH_OPTS "$BOARD" "mkdir -p $BOARD_DIR/arms && rm -f $BOARD_DIR/arms/*" || exit 1
scp $SCP_OPTS "$RKNN" "$BOARD:$BOARD_DIR/arms/" || exit 1
[ -f "$OUT_ROOT/rknn_arms/${NAME}_int8.stride" ] && \
  scp $SCP_OPTS "$OUT_ROOT/rknn_arms/${NAME}_int8.stride" "$BOARD:$BOARD_DIR/arms/"

echo "[5/5] 上板实测（一次会话，含 health check）"
# ⚠️ 板卡驱动脆弱 ⇒ 一次会话内测完，不要反复连接。
cat > "$OUT_ROOT/measure_board.sh" << 'MEASURE_EOF'
#!/usr/bin/env bash
set -uo pipefail
DIR="${BOARD_DIR:-/root/hm_speed}"
export LD_LIBRARY_PATH="$DIR/lib:/mnt/manu/npu:/usr/lib"
RES="$DIR/hm_speed_results.txt"
: > "$RES"
log() { printf '%s\n' "$*" | tee -a "$RES"; }
log "==== heatmap model speed session (from server) ===="
log "date: $(date)"
log "librknnrt: $(md5sum $DIR/lib/librknnrt.so 2>/dev/null | cut -d' ' -f1)"
# health check（3 帧）
# ⚠️ **必须按臂传 --stride**（2026-11-06 实测踩到）：板卡工具校验
#    `input/grid ratio == stride`（rknn_heatmap_smoke.cpp:1159），
#    160 输出的模型 ratio=4，而工具默认 stride=2 ⇒ 会报
#    「[FATAL] input/grid ratio 4 does not match --stride 2」。
#    ⇒ health check 用第一个臂自己的边车 stride。
_hm_arm=$(ls "$DIR"/arms/*.rknn 2>/dev/null | head -1)
_hm_st=2
[ -n "$_hm_arm" ] && _hm_st=$(cat "$DIR/arms/$(basename "$_hm_arm" .rknn).stride" 2>/dev/null || echo 2)
log "health check: arm=$(basename "${_hm_arm:-none}") stride=$_hm_st"
timeout 90 "$DIR/rknn_heatmap_smoke" "$_hm_arm" \
  "$DIR/feat.jpg" --loops 3 --warmup 1 --stride "$_hm_st" --tag health 2>&1 \
  | grep -E "rknn_run|FATAL|Illegal|implied stride" | tee -a "$RES"
for arm in "$DIR"/arms/*.rknn; do
  [ -e "$arm" ] || continue
  st=$(cat "$DIR/arms/$(basename "$arm" .rknn).stride" 2>/dev/null || echo 2)
  sleep "${SETTLE:-5}"
  timeout 600 "$DIR/rknn_heatmap_smoke" "$arm" "$DIR/feat.jpg" \
    --loops "${LOOPS:-30}" --warmup "${WARMUP:-10}" --stride "$st" \
    --tag "$(basename "$arm")" 2>&1 | grep -E "rknn_run|NPU.dequant|FATAL|CHANGED|IDENTICAL" | tee -a "$RES"
done
log "==== session complete ===="
MEASURE_EOF
scp $SCP_OPTS "$OUT_ROOT/measure_board.sh" "$BOARD:$BOARD_DIR/measure_board.sh" || exit 1
ssh $SSH_OPTS "$BOARD" "cd $BOARD_DIR && BOARD_DIR=$BOARD_DIR LOOPS=$LOOPS WARMUP=$WARMUP SETTLE=$SETTLE bash $BOARD_DIR/measure_board.sh" \
  || { echo "[WARN] 板卡会话异常（详见 $BOARD_DIR/hm_speed_results.txt）"; }

scp $SCP_OPTS "$BOARD:$BOARD_DIR/hm_speed_results.txt" "$OUT_ROOT/hm_speed_results.txt" || exit 1
echo ""
echo "=== 结果 ==="
grep -E "rknn_run +avg" "$OUT_ROOT/hm_speed_results.txt" | tail -3
echo ""
echo "[OK] 原始输出：$OUT_ROOT/hm_speed_results.txt"
echo "[OK] 把它喂给验收脚本："
echo "  python manu/training/npu20ms_acceptance.py --root /mnt/data/siping/runs/npu20ms \\"
echo "      --speed-table manu/pipeline/npu/struct_speed_table.json"
