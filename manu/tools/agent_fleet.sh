#!/usr/bin/env bash
# 多 Agent 物理并行编排器 —— 方案二的落地（worktree 隔离 + 独立会话实例）
#
# 用途：为「需要物理级并行、互不阻塞」的长任务拉起 N 个独立 Hermes 实例。
# 与 delegate_task 的分工：
#   - delegate_task  = 分钟级、共享父会话上下文、子 agent 自动回收
#   - 本脚本        = 小时级、独立会话/独立终端、可跨进程存活、需真隔离写入
#
# 用法：
#   bash manu/tools/agent_fleet.sh plan  <repo> <base-branch> <N> [roles,csv]
#   bash manu/tools/agent_fleet.sh up    <repo> <base-branch> <N> [session-prefix]
#   bash manu/tools/agent_fleet.sh status
#   bash manu/tools/agent_fleet.sh down
#   bash manu/tools/agent_fleet.sh send   <session> <message>
#   bash manu/tools/agent_fleet.sh capture <session>
#
# 后端自动降级：tmux → screen → nohup（本机 screen 已装；tmux 需 sudo 安装）。
# 会话名带后缀后缀（f1-tmux / f1-screen），send/capture 用全名。
#
# 铁律（继承 manu/memory/rules.md）：
#   - 只在本地工程目录操作；worktree 建在本地，不在服务器
#   - 起 GPU 重活前先查卡（nvidia-smi --query-compute-apps=...）
#
set -euo pipefail

HERMES_BIN="${HERMES_BIN:-hermes}"
REPO_ROOT="${REPO_ROOT:-/media/manu/1TB-Volume/workspace/ultralytics}"
WT_ROOT="${WT_ROOT:-/media/manu/1TB-Volume/workspace/.hermes-fleet}"
STATE_DIR="${WT_ROOT}/.state"
ROLE_DEFAULT="worker"

mkdir -p "$STATE_DIR"

# ---- 后端选择（tmux > screen > nohup） --------------------------------------
detect_backend() {
  if command -v tmux >/dev/null 2>&1; then echo tmux
  elif command -v screen >/dev/null 2>&1; then echo screen
  else echo nohup
  fi
}
BACKEND="${FLEET_BACKEND:-$(detect_backend)}"

die() { echo "ERROR: $*" >&2; exit 1; }
ok()  { echo "  $*"; }

be_new() {  # be_new <session> <workdir> <cmd...>
  local sess="$1" wd="$2"; shift 2
  case "$BACKEND" in
    tmux)   tmux new-session -d -s "$sess" -x 200 -y 50 -c "$wd" "$@" ;;
    screen) screen -dmS "$sess" -c bash -lc "cd '$wd' && exec $*" ; sleep 1 ;;
    nohup)  ( cd "$wd" && setsid nohup "$@" >"$STATE_DIR/$sess.log" 2>&1 & ) ;;
  esac
}
be_alive() {
  local sess="$1"
  case "$BACKEND" in
    tmux)   tmux has-session -t "$sess" 2>/dev/null ;;
    screen) screen -ls 2>/dev/null | grep -q "[.]${sess}[[:space:]]" ;;
    nohup)  pgrep -f "$sess" >/dev/null 2>&1 ;;
  esac
}
be_send() {
  local sess="$1"; shift
  case "$BACKEND" in
    tmux)   tmux send-keys -t "$sess" "$*" Enter ;;
    screen) screen -S "$sess" -p 0 -X stuff "$*$(printf '\r')" ;;
    nohup)  echo "[nohup 后端不支持交互发送] $*" ;;
  esac
}
be_capture() {
  local sess="$1"
  case "$BACKEND" in
    tmux)   tmux capture-pane -t "$sess" -p | tail -40 ;;
    screen) screen -S "$sess" -X hardcopy -h "$STATE_DIR/$sess.txt" 2>/dev/null && tail -40 "$STATE_DIR/$sess.txt" ;;
    nohup)  tail -40 "$STATE_DIR/$sess.log" 2>/dev/null || echo "(无日志)" ;;
  esac
}
be_kill() {
  local sess="$1"
  case "$BACKEND" in
    tmux)   tmux kill-session -t "$sess" 2>/dev/null || true ;;
    screen) screen -S "$sess" -X quit 2>/dev/null || true ;;
    nohup)  pkill -f "$sess" 2>/dev/null || true ;;
  esac
}

cmd_plan() {
  local repo="$1" base="$2" n="$3" roles="${4:-}"
  [[ -d "$repo/.git" ]] || die "不是 git 仓库: $repo"
  (( n >= 1 && n <= 16 )) || die "N 必须在 1..16（Hermes 并发上限见 delegation.max_concurrent_children）"
  echo "Fleet plan: base=$base  N=$n  backend=$BACKEND  roles=${roles:-默认 worker}"
  echo
  printf "%-4s %-38s %s\n" "#" "worktree" "branch"
  printf "%-4s %-38s %s\n" "---" "--------------------------------------" "------"
  local i=1
  while (( i <= n )); do
    local role="$ROLE_DEFAULT"
    if [[ -n "$roles" ]]; then
      IFS=',' read -ra arr <<< "$roles"
      role="${arr[$((i-1))]}"
      [[ -n "$role" ]] || role="$ROLE_DEFAULT"
    fi
    printf "%-4s %-38s %s\n" "$i" "${WT_ROOT}/${role}-${i}" "${base}/${role}-${i}"
    i=$((i+1))
  done
  echo
  echo "提示：同一分支不要分配给两个 worker（写冲突）。"
  echo "      每个 worktree 内跑独立 hermes 实例，角色由 prompt 首句声明。"
}

cmd_up() {
  local repo="$1" base="$2" n="$3" prefix="${4:-f}"
  [[ -d "$repo/.git" ]] || die "不是 git 仓库: $repo"
  mkdir -p "$WT_ROOT"

  # 起物理并行前先查 GPU 是否被占（铁律：评估不得与训练同卡）
  if command -v nvidia-smi >/dev/null 2>&1; then
    local busy
    busy="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | wc -l)"
    [[ "$busy" -gt 0 ]] && echo "WARN: 当前有 $busy 个 GPU 进程在跑；派发前确认它们不是训练/评估同卡。"
  fi

  local i=1
  while (( i <= n )); do
    local wt="${WT_ROOT}/w${i}" branch="${base}/w${i}" sess="${prefix}${i}-${BACKEND}"
    if [[ ! -d "$wt" ]]; then
      git -C "$repo" worktree add -b "$branch" "$wt" >/dev/null 2>&1 \
        || git -C "$repo" worktree add "$wt" "$branch" >/dev/null 2>&1 \
        || die "worktree 创建失败: $wt"
      ok "worktree: $wt  (branch $branch)"
    else
      ok "worktree 已存在，复用: $wt"
    fi
    be_new "$sess" "$wt" "$HERMES_BIN -w"
    sleep 2
    echo "$sess|$wt|$branch" >> "$STATE_DIR/sessions"
    ok "session: $sess (cwd $wt)"
    i=$((i+1))
  done
  echo
  echo "启动后：bash $0 send <session> '<你的角色 prompt>'   （session 名带 -$BACKEND 后缀）"
  echo "查看：bash $0 status"
}

cmd_status() {
  echo "== fleet sessions (backend=$BACKEND) =="
  if [[ -s "$STATE_DIR/sessions" ]]; then
    while IFS='|' read -r sess wt branch; do
      local alive="dead"
      be_alive "$sess" && alive="alive"
      printf "  %-18s %-8s %s\n" "$sess" "$alive" "$wt"
    done < "$STATE_DIR/sessions"
  else
    echo "  (无)"
  fi
  echo
  echo "== git worktrees =="
  git -C "$REPO_ROOT" worktree list 2>/dev/null | sed 's/^/  /' || echo "  (无)"
}

cmd_send() {
  local sess="$1"; shift
  be_send "$sess" "$*"
  echo "sent to $sess"
}

cmd_capture() {
  local sess="$1"
  be_capture "$sess"
}

cmd_down() {
  if [[ -s "$STATE_DIR/sessions" ]]; then
    while IFS='|' read -r sess wt branch; do
      if be_alive "$sess"; then be_kill "$sess"; ok "killed $sess"; fi
    done < "$STATE_DIR/sessions"
    : > "$STATE_DIR/sessions"
  fi
  echo "sessions stopped（worktree 保留；用 'git worktree remove' 手动清理）"
}

case "${1:-}" in
  plan)    shift; cmd_plan "$@" ;;
  up)      shift; cmd_up "$@" ;;
  status)  cmd_status ;;
  down)    cmd_down ;;
  send)    shift; cmd_send "$@" ;;
  capture) shift; cmd_capture "$@" ;;
  *) cat <<EOF
用法: agent_fleet.sh <plan|up|status|down|send|capture> ...
  plan    <repo> <base-branch> <N> [roles,csv]
  up      <repo> <base-branch> <N> [session-prefix]
  status | down
  send    <session> <message>
  capture <session>
EOF
  exit 1 ;;
esac
