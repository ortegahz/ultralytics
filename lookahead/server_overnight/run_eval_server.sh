#!/usr/bin/env bash
# ============================================================
# ★ 把重活放到服务器跑（铁律十三）
#   用法（在【用户自己的终端】执行，沙箱内出网被封）：
#     cd /media/manu/1TB-Volume/workspace/ultralytics
#     bash manu/tracking/lookahead/server_overnight/run_eval_server.sh
#
#   流程：本地打包 → scp 到服务器 → 服务器预检 → 服务器跑 8 序列评测
#         → scp 拉回产物 → 本地打印汇总表
#
#   可覆盖的环境变量：
#     OUT_TAG        产物标签（默认 perframe_s30_v2）
#     ARM_LIMIT      --output-max-step-px（默认 30）
#     DELTA          滞后帧数（默认 8 = 320ms @25Hz）
#     REMOTE_PY      指定服务器 python（默认自动探测）
#     SKIP_PUSH=1    跳过上传（代码没改时用）
# ============================================================
set -euo pipefail

LOCAL_REPO="${LOCAL_REPO:-/media/manu/1TB-Volume/workspace/ultralytics}"
SSH_HOST="${SSH_HOST:-huangzhe@192.168.99.40}"
SSH_PORT="${SSH_PORT:-32222}"          # ★ 服务器 sshd 在 32222，22 是关闭的
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/pycharm_project_10ae9e2e}"
SSH_OPTS=(-p "$SSH_PORT" -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

OUT_TAG="${OUT_TAG:-perframe_s30_v2}"
ARM_LIMIT="${ARM_LIMIT:-30}"
DELTA="${DELTA:-8}"
BUNDLE_REL="manu/tracking/lookahead/server_overnight"

REMOTE_CACHE="$REMOTE_ROOT/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache"
REMOTE_LABELS="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
REMOTE_FRAMES="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
REMOTE_OUT="$REMOTE_ROOT/runs/lookahead/final/$OUT_TAG"
REMOTE_LOG="$REMOTE_OUT/server_run.log"

echo "== [0/5] 探活 =="
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "echo SERVER_OK; nproc; free -g | head -2; ls -d '$REMOTE_CACHE' >/dev/null 2>&1 && echo CACHE_OK || echo CACHE_MISSING"

if [ "${SKIP_PUSH:-0}" != "1" ]; then
  echo "== [1/5] 打包本地代码 =="
  TMP_TAR="$(mktemp /tmp/lookahead_eval.XXXXXX.tar.gz)"
  tar -C "$LOCAL_REPO" \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
      -czf "$TMP_TAR" \
      manu/tracking \
      manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
      manu/memory/lookahead_extrapolator_20261102.md
  ls -lh "$TMP_TAR"
  echo "== [2/5] 上传 =="
  scp "${SSH_OPTS[@]}" "$TMP_TAR" "$SSH_HOST:/tmp/lookahead_eval.tar.gz"
else
  echo "== [1-2/5] SKIP_PUSH=1，跳过打包上传 =="
fi

echo "== [3/5] 服务器预检 =="
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "set -e
cd '$REMOTE_ROOT'
tar -xzf /tmp/lookahead_eval.tar.gz 2>/dev/null || true
echo '--- python 候选 ---'
for cand in \${REMOTE_PY:-} /home/huangzhe/anaconda3/envs/uav/bin/python /home/huangzhe/miniconda3/envs/uav/bin/python \
            /home/huangzhe/anaconda3/bin/python /home/huangzhe/miniconda3/bin/python python3; do
  [ -z \"\$cand\" ] && continue
  if command -v \"\$cand\" >/dev/null 2>&1 || [ -x \"\$cand\" ]; then
    if \"\$cand\" -c 'import numpy,cv2,sys;print(\"PY_OK\",sys.executable,numpy.__version__,cv2.__version__)' 2>/dev/null; then
      echo \"GOOD_PY=\$cand\"; break
    else
      echo \"  (候选不可用: \$cand)\"
    fi
  fi
done
echo '--- 关键路径 ---'
for p in '$REMOTE_CACHE' '$REMOTE_LABELS' '$REMOTE_FRAMES' '$REMOTE_ROOT/manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json'; do
  if [ -e \"\$p\" ]; then echo \"  OK      \$p\"; else echo \"  MISSING \$p\"; fi
done
echo '--- ms_cache 序列数 ---'
ls '$REMOTE_CACHE' 2>/dev/null | wc -l
"

echo "== [4/5] 服务器端评测（前台 ssh，输出到 $REMOTE_LOG）==="
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "set -e
cd '$REMOTE_ROOT'
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
mkdir -p '$REMOTE_OUT'
\$PY manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root '$REMOTE_CACHE' --labels-root '$REMOTE_LABELS' --all-seqs \
  --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
  --frames-root '$REMOTE_FRAMES' \
  --lookahead replay --delta $DELTA --arms original,stale,kinematic,replay --replay-mode perframe \
  --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta --no-replay-hold-use-ref \
  --fast-win-size 48 --fast-bg-window 96 --output-max-step-px $ARM_LIMIT \
  --out-json '$REMOTE_OUT/arms_$OUT_TAG.json' --out-preds-dir '$REMOTE_OUT/preds_$OUT_TAG' \
  2>&1 | tee '$REMOTE_LOG'
"

echo "== [5/5] 拉回产物 =="
LOCAL_OUT="$LOCAL_REPO/runs/lookahead/final"
mkdir -p "$LOCAL_OUT"
scp "${SSH_OPTS[@]}" "$SSH_HOST:$REMOTE_OUT/arms_$OUT_TAG.json" "$LOCAL_OUT/arms_$OUT_TAG.json"
scp "${SSH_OPTS[@]}" "$SSH_HOST:$REMOTE_LOG" "$LOCAL_OUT/arms_$OUT_TAG.log"
scp -r "${SSH_OPTS[@]}" "$SSH_HOST:$REMOTE_OUT/preds_$OUT_TAG" "$LOCAL_OUT/preds_$OUT_TAG"

echo
echo "================ 服务器端汇总 ================"
grep -E '^\[SEQ\]|^\[OVERALL\]|^\[ARM\]|^\[GATE\]|^\[SUCCESS\]' "$LOCAL_OUT/arms_$OUT_TAG.log" || true
echo "=============================================="
echo "产物：$LOCAL_OUT/arms_$OUT_TAG.json"
echo "      $LOCAL_OUT/preds_$OUT_TAG/"
echo "      $LOCAL_OUT/arms_$OUT_TAG.log"
