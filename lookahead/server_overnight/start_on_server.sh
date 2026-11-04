#!/usr/bin/env bash
# ============================================================
# 服务器端入口: 夜间作业总控
#   A) 确定性作业(不依赖 LLM): 环境自检 -> 单测 -> 24序列原逻辑/外挂评测 -> 对照表
#   B) 服务器端 codex 会话(带完整上下文 TASK.md): 算法返工 + 交互式排查
# 两者并行跑, 互为兜底。
# ============================================================
set -uo pipefail

ROOT="${ROOT:-/tmp/pycharm_project_10ae9e2e}"
BUNDLE="$ROOT/manu/tracking/lookahead/server_overnight"
OUT="$ROOT/runs/lookahead/server_out"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
# codex 是用 `curl ... | sh` 装的, 默认在 ~/.local/bin; 非交互 shell 的 PATH 里往往没有
export PATH="$HOME/.local/bin:$HOME/.codex/bin:$PATH"

log(){ echo "[$(date '+%F %T')] $*"; }

[ -f "$OUT/progress.md" ] || cat > "$OUT/progress.md" <<'PM'
# 夜间作业进度
- 启动时间: (见 overnight.log 首行)
- 状态: 刚启动, 尚未产出
PM
log "===== 环境自检 ====="
python -V 2>&1 | tee -a "$OUT/env.log"
(nvidia-smi -L 2>&1 || echo "NO GPU") | tee -a "$OUT/env.log"
python - <<'PY' 2>&1 | tee -a "$OUT/env.log"
try:
    import numpy, torch
    print("numpy", numpy.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("IMPORT_ERR", type(e).__name__, e)
PY
python -c "import numpy;print('numpy_ok',numpy.__version__)" 2>&1 | tee -a "$OUT/env.log"

log "===== 定位 24 序列官方 ms_cache（服务器端自查） ====="
for d in \
  "$ROOT/runs/optuna_mix_ratio/trials/trial_0080/lq_eval/ms_cache" \
  "$ROOT/runs/optuna_mix_ratio/trials/trial_0080/ms_cache" \
  "$ROOT/runs/optuna_mix_ratio/trials/trial_0080/gate/ms_cache" ; do
  [ -d "$d" ] && log "FOUND: $d"
done
CACHE_ROOT="$(timeout 120 find "$ROOT/runs" -maxdepth 6 -type d -name ms_cache 2>/dev/null | head -1)"
[ -n "${CACHE_ROOT:-}" ] && log "FOUND(by find): $CACHE_ROOT" || log "WARN: 未找到 ms_cache, 需在服务器用 GPU 重建"
echo "${CACHE_ROOT:-}" > "$OUT/cache_root.txt"

log "===== 单元测试 (manu/tracking/tests) ====="
python -m pytest manu/tracking/tests -q 2>&1 | tail -25 | tee "$OUT/pytest.log" || \
python -m unittest discover -s manu/tracking/tests -t . -v 2>&1 | tail -25 | tee "$OUT/pytest.log"

log "===== 启动确定性夜间评测 (run_overnight_eval.sh) ====="
setsid nohup bash "$BUNDLE/run_overnight_eval.sh" > "$OUT/eval.log" 2>&1 < /dev/null &
sleep 3
log "eval pid=$! 日志: $OUT/eval.log"

log "===== 启动服务器端 codex 会话 (TASK.md 上下文) ====="
command -v codex >/dev/null 2>&1 || log "WARN: PATH 里没有 codex, 尝试 ~/.local/bin/codex"
CODEX_BIN="$(command -v codex || echo "$HOME/.local/bin/codex")"
if [ -x "$CODEX_BIN" ]; then
  setsid nohup "$CODEX_BIN" exec \
      --skip-git-repo-check -C "$ROOT" \
      -c approval_policy='"never"' -c sandbox_mode='"workspace-write"' \
      -o "$OUT/codex_last_message.txt" \
      - < "$BUNDLE/TASK.md" > "$OUT/codex_session.log" 2>&1 < /dev/null &
  log "codex exec pid=$! 日志: $OUT/codex_session.log"
else
  log "WARN: 服务器上找不到 codex 命令, 只跑确定性作业"
fi

log "===== 总控启动完毕, 各子任务后台运行中 ====="
