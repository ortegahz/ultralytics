#!/usr/bin/env bash
# ============================================================
# ★ 服务器端「变体矩阵」评测（铁律十三：重活一律上服务器）
#
#   用法（用户在自己终端执行）：
#     cd /media/manu/1TB-Volume/workspace/ultralytics
#     bash manu/tracking/lookahead/server_overnight/run_matrix_server.sh
#
#   一次推送 → 服务器连跑 N 个变体（每个都是完整 8 序列）→ 全部拉回本地
#
#   变体（可用 VARIANTS 环境变量覆盖，逗号分隔）：
#     delta_s30   交付候选：anchor=delta + 输出级限幅 30
#     pred_r24_s30  ★ anchor=pred（ROI 钉在"锚点+匀速预测"）+ 限幅 30 —— 治 190px 跑飞
#     pred_r12_s30 / pred_r40_s30  pred 的回看半径扫描
#     gateA       关掉开关的逐位一致性复验（Gate A）
# ============================================================
set -euo pipefail

LOCAL_REPO="${LOCAL_REPO:-/media/manu/1TB-Volume/workspace/ultralytics}"
SSH_HOST="${SSH_HOST:-huangzhe@192.168.99.40}"
SSH_PORT="${SSH_PORT:-32222}"
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/pycharm_project_10ae9e2e}"
SSH_OPTS=(-p "$SSH_PORT" -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
BUNDLE_REL="manu/tracking/lookahead/server_overnight"

DELTA="${DELTA:-8}"
VARIANTS="${VARIANTS:-delta_s30,w32_bg96,w48_sg20,w32_sg20,gateA}"

REMOTE_CACHE="$REMOTE_ROOT/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"
REMOTE_LABELS="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
REMOTE_FRAMES="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
REMOTE_OUT="$REMOTE_ROOT/runs/lookahead/final"
LOCAL_OUT="$LOCAL_REPO/runs/lookahead/final"

echo "== [1/4] 打包 + 上传 =="
TMP_TAR="$(mktemp /tmp/lookahead_matrix.XXXXXX.tar.gz)"
tar -C "$LOCAL_REPO" --exclude='__pycache__' --exclude='*.pyc' -czf "$TMP_TAR" \
    manu/tracking \
    manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
    manu/memory/lookahead_extrapolator_20261102.md
ls -lh "$TMP_TAR"
scp "${SSH_OPTS[@]}" "$TMP_TAR" "$SSH_HOST:/tmp/lookahead_matrix.tar.gz"

echo "== [2/4] 服务器解包 + 预检 =="
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "set -e
cd '$REMOTE_ROOT'
tar -xzf /tmp/lookahead_matrix.tar.gz
mkdir -p '$REMOTE_OUT'
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=\${REMOTE_PY:-}
if [ -z \"\$PY\" ]; then
  for c in /home/huangzhe/anaconda3/envs/uav/bin/python /home/huangzhe/miniconda3/envs/uav/bin/python \
           /home/huangzhe/anaconda3/bin/python python3; do
    if command -v \"\$c\" >/dev/null 2>&1 && \"\$c\" -c 'import numpy,cv2' 2>/dev/null; then PY=\"\$c\"; break; fi
  done
fi
[ -z \"\$PY\" ] && { echo 'FATAL: 服务器上找不到可用 python（numpy+cv2）'; exit 2; }
echo \"USING_PY=\$PY\"
for p in '$REMOTE_CACHE' '$REMOTE_LABELS' '$REMOTE_FRAMES'; do
  [ -e \"\$p\" ] && echo \"  OK      \$p\" || echo \"  MISSING \$p\"
done
echo \"ms_cache 条目数: \$(ls '$REMOTE_CACHE' 2>/dev/null | wc -l)\"
echo \"--- 开始变体矩阵 ---\"

run_one() {
  local tag=\"\$1\"; shift
  local j=\"\$REMOTE_OUT/arms_\$tag.json\"
  local log=\"\$REMOTE_OUT/arms_\$tag.log\"
  echo \"### [RUN] \$tag :: \$*\"
  ( cd '$REMOTE_ROOT' && \$PY manu/tracking/lookahead/lookahead_longquan.py \\
      --cache-root '$REMOTE_CACHE' --labels-root '$REMOTE_LABELS' --all-seqs \\
      --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \\
      --frames-root '$REMOTE_FRAMES' \\
      --delta $DELTA \\\$@ \\
      --out-json \"\$j\" --out-preds-dir \"\$REMOTE_OUT/preds_\$tag\" ) > \"\$log\" 2>&1
  local rc=\$?
  echo \"### [DONE] \$tag rc=\$rc\"
  grep -E '^\[OVERALL\]|^\[ARM\]|^\[GATE\]' \"\$log\" || tail -5 \"\$log\"
  return \$rc
}

for v in \$(echo '$VARIANTS' | tr ',' ' '); do
  case \"\$v\" in
    delta_s30)    run_one delta_s30    --lookahead replay --arms original,stale,kinematic,replay --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta --no-replay-hold-use-ref --fast-win-size 48 --fast-bg-window 96 --output-max-step-px 30 ;;
    pred_r12_s30) run_one pred_r12_s30 --lookahead replay --arms original,stale,kinematic,replay --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --replay-anchor pred --replay-pred-radius-px 12 --no-replay-hold-use-ref --fast-win-size 48 --fast-bg-window 96 --output-max-step-px 30 ;;
    pred_r24_s30) run_one pred_r24_s30 --lookahead replay --arms original,stale,kinematic,replay --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --replay-anchor pred --replay-pred-radius-px 24 --no-replay-hold-use-ref --fast-win-size 48 --fast-bg-window 96 --output-max-step-px 30 ;;
    pred_r40_s30) run_one pred_r40_s30 --lookahead replay --arms original,stale,kinematic,replay --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --replay-anchor pred --replay-pred-radius-px 40 --no-replay-hold-use-ref --fast-win-size 48 --fast-bg-window 96 --output-max-step-px 30 ;;
    w32_bg96)     run_one w32_bg96     --lookahead replay --arms original,stale,kinematic,replay --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta --no-replay-hold-use-ref --fast-win-size 32 --fast-bg-window 96 --output-max-step-px 30 ;;
    w48_sg20)     run_one w48_sg20     --lookahead replay --arms original,stale,kinematic,replay --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta --no-replay-hold-use-ref --fast-win-size 48 --fast-bg-window 96 --replay-step-gate-px 20 --output-max-step-px 30 ;;
    w32_sg20)     run_one w32_sg20     --lookahead replay --arms original,stale,kinematic,replay --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta --no-replay-hold-use-ref --fast-win-size 32 --fast-bg-window 96 --replay-step-gate-px 20 --output-max-step-px 30 ;;
    gateA)        run_one gateA        --lookahead off --arms original ;;
    *) echo \"### [SKIP] 未知变体 \$v\" ;;
  esac
done
echo \"--- 变体矩阵结束 ---\"
"

echo "== [3/4] 拉回全部产物 =="
mkdir -p "$LOCAL_OUT"
for tag in $(echo "$VARIANTS" | tr ',' ' '); do
  scp "${SSH_OPTS[@]}" "$SSH_HOST:$REMOTE_OUT/arms_$tag.json" "$LOCAL_OUT/arms_$tag.json" 2>/dev/null || echo "  (缺 $tag.json)"
  scp "${SSH_OPTS[@]}" "$SSH_HOST:$REMOTE_OUT/arms_$tag.log" "$LOCAL_OUT/arms_$tag.log" 2>/dev/null || true
done

echo "== [4/4] 本地汇总 =="
for tag in $(echo "$VARIANTS" | tr ',' ' '); do
  echo "───────── $tag ─────────"
  grep -E '^\[OVERALL\]|^\[ARM\]|^\[GATE\]' "$LOCAL_OUT/arms_$tag.log" 2>/dev/null || echo "  (无日志)"
done
