#!/usr/bin/env bash
# =============================================================================
# 把本机工程代码 upload 到服务器（host → server）
#
# 为什么需要这个封装：
#   2026-11-16 起，大件产物（runs/ 60 GB + 板端构建产物 0.8 GB）已迁到
#   /media/manu/1TB-Volume/workspace/cache/ultralytics/；
#   工程目录内**已无软链接、也无产物目录**（按用户要求：不要链接，直接删）。
#   ⇒ upload 载荷实测 82.5 MB（清单里已无 runs / .rknn / build 产物）。
#
#   🔴 但排除项**不能删**：本机已没有 `runs/`，若开 --delete，rsync 会把**服务器侧**的
#      `runs/`（当前 nano_search 正在写的 runs/mixed_ft_route_b/）当成"本机已删除"而删掉。
#      实测（rsync 3.4.1）：rsync -a --delete → 目标非空目录被清空；加 --exclude → 安全。
#   ⇒ 本脚本把排除项写死，并默认禁止 --delete。
#
# 用法:
#   DRY_RUN=1 bash manu/videos/upload_to_server.sh    # 默认：只看要传什么，不真传
#   DRY_RUN=0 bash manu/videos/upload_to_server.sh    # 真传
#   ALLOW_DELETE=1 DRY_RUN=0 bash ...                 # 确实要删服务器侧多余文件时（危险）
#   EXCLUDE_EXTRA="--exclude=xxx" ...                 # 临时再加排除项
#
# 铁律：本脚本**只做 upload，绝不碰服务器上已有的数据**（除非显式 ALLOW_DELETE=1）。
# =============================================================================
set -euo pipefail

REPO="${REPO:-/media/manu/1TB-Volume/workspace/ultralytics}"
SSH_HOST="${SSH_HOST:-huangzhe@192.168.99.40}"
SSH_PORT="${SSH_PORT:-32222}"
# 🔴 2026-11-16 用户铁律：**服务器路径一律不动**。
#    默认落点绝不能是 /tmp/pycharm_project_10ae9e2e —— 那是 nano_search harness 的活仓库
#    （30+ 个 harness 脚本 `cd /tmp/pycharm_project_10ae9e2e`，且它是指向
#    /mnt/data/siping/runs/npu20ms/nano_search/repo 的软链；往里写=写别人的活）。
# 🔴 2026-11-16 用户铁律：**服务器路径一律不动**。
#    惯用落点仍是 /tmp/pycharm_project_10ae9e2e（2026-11-16 15:19 重启后重传 ⇒ 它现在是**真实目录**）。
#    ⚠️ 但它历史上是指向 /mnt/data/siping/runs/npu20ms/nano_search/repo 的**软链**（harness 的活仓库）
#       ⇒ 落点若是软链，rsync 会**穿透**写进 harness 正在跑的活目录 ⇒ 守卫⓪ 会拦下。
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/pycharm_project_10ae9e2e}"
DRY_RUN="${DRY_RUN:-1}"
ALLOW_DELETE="${ALLOW_DELETE:-0}"
EXCLUDE_EXTRA="${EXCLUDE_EXTRA:-}"
SSH_OPTS=(-p "$SSH_PORT" -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

# ── ssh 可用性自愈：本机 /etc/ssh/ssh_config.d/*.conf 若属主/权限不对，ssh 会直接拒绝读配置
#    （"Bad owner or permissions on ..."）⇒ 自动降级为 -F /dev/null 绕过系统配置。
if ! ssh "${SSH_OPTS[@]}" "$SSH_HOST" true 2>/dev/null; then
  if ssh -F /dev/null "${SSH_OPTS[@]}" "$SSH_HOST" true 2>/dev/null; then
    SSH_OPTS+=(-F /dev/null)
    echo "[fix] 系统 ssh_config 不可用 ⇒ 本次改用 -F /dev/null"
  else
    echo "[FAIL] 连不上 $SSH_HOST:$SSH_PORT（ssh 与 -F /dev/null 都失败）" >&2
    exit 1
  fi
fi

# ── 排除项 ────────────────────────────────────────────────────────────────────
# 现在本机已无这些产物路径，排除项的作用变成**保护服务器侧的同名目录不被 --delete 删掉**。
# 注意：必须以**不带行尾斜杠**的路径形式写，否则 --delete 仍会删掉服务器侧的真实目录。
EXCLUDES=(
  --exclude=/runs
  --exclude=/manu/pipeline/npu/out
  --exclude=/manu/pipeline/npu/results
  --exclude=/manu/pipeline/npu/build
  --exclude=/manu/pipeline/npu/.demo-build
  --exclude=/manu/pipeline/npu/.demo-build-mobilenet
  --exclude=/manu/pipeline/npu/convert/*.rknn
  --exclude=/manu/pipeline/opencl/board/out
  --exclude=/manu/pipeline/opencl/board/.bench_bmp
  --exclude=.git
  --exclude=__pycache__
  --exclude='*.pyc'
  --exclude=.pytest_cache
  --exclude=.ruff_cache
  --exclude=.idea
  --exclude=.codex
  --exclude=.minimax
  --exclude='*.log'
)

RSYNC_ARGS=(-a --checksum --compress --partial --info=stats2 "${EXCLUDES[@]}")
# 🔴 DRY_RUN 必须真的加 --dry-run，否则「预览」会变成真传（2026-11-16 自查修）
[ "$DRY_RUN" = "1" ] && RSYNC_ARGS+=(--dry-run)
[ -n "$EXCLUDE_EXTRA" ] && RSYNC_ARGS+=($EXCLUDE_EXTRA)
[ "$ALLOW_DELETE" = "1" ] && RSYNC_ARGS+=(--delete) \
  || echo "[guard] 未设 ALLOW_DELETE=1 ⇒ 不带 --delete（服务器上的旧文件保持不动）"

echo "== [1/3] 目标 =="
echo "  本机   : $REPO"
echo "  服务器 : $SSH_HOST:$SSH_PORT$REMOTE_ROOT"
echo "  模式   : $([ "$DRY_RUN" = "1" ] && echo 'DRY-RUN（不落盘）' || echo '真传')"

# ── 落点守卫：上传前先看清服务器那边现在有什么，别覆盖别人的东西 ─────────────
echo "== [1.5/3] 落点现状（只读） =="
DEST_LISTING=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "ls -A '$REMOTE_ROOT' 2>/dev/null | tr '\\n' ' '; echo; readlink -f '$REMOTE_ROOT' 2>/dev/null" 2>/dev/null || true)
if [ -z "$DEST_LISTING" ]; then
  echo "  (落点不存在或不可读 ⇒ 将新建)"
else
  echo "  落点实路径: $(echo "$DEST_LISTING" | tail -1)"
  echo "  已有条目  : $(echo "$DEST_LISTING" | head -1)"
fi

# ── 落点守卫（两条，缺一不可）─────────────────────────────────────────────────
# ── 守卫 ⓪（红线，最高优先）：落点既不能是 harness 的活路径，也不能是软链 ──────
# 软链落点 = rsync 会「穿过去」写进软链指向的真实目录（nano_search 的活 repo）⇒ 踩别人的活。
# 2026-11-16 用户铁律：服务器路径一律不动，run 数据以 host 侧备份为准。
HARNESS_PATH="/tmp/pycharm_project_10ae9e2e"
if [ "${FORCE_TARGET:-0}" != "1" ]; then
  # 判据 = 落点**是不是软链**、以及**软链后面是不是 harness 的活仓库**。
  # 真实目录（正常落点）⇒ 放行；软链 ⇒ 穿透风险 ⇒ 拦下。
  IS_LINK=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "[ -L '$REMOTE_ROOT' ] && echo YES || echo NO" 2>/dev/null || echo UNKNOWN)
  REAL_PATH=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "readlink -f '$REMOTE_ROOT' 2>/dev/null" 2>/dev/null || true)
  if [ "$IS_LINK" = "YES" ]; then
    echo >&2
    echo "[FAIL] 落点是软链：$REMOTE_ROOT ⇒ rsync 会穿链写入它的真实目标。" >&2
    echo "       真实目标：${REAL_PATH:-未知}" >&2
    case "$REAL_PATH" in
      /mnt/data/siping/runs/npu20ms/nano_search/*)
        echo "       ⇒ 这正是 nano_search harness 的活仓库（30+ 个 harness 脚本正 cd 进去）。" >&2
        echo "       ⇒ 往那儿 upload = 写别人的活作业。铁律：服务器路径一律不动。" >&2
        ;;
      *)
        echo "       ⇒ 软链落点一律拒绝：穿链写入的真实目标未必属于本工程。" >&2
        ;;
    esac
    echo "       ⇒ 确认无误后加 FORCE_TARGET=1 可放行；或换 REMOTE_ROOT 指到真实目录。" >&2
    exit 2
  fi
  case "$REAL_PATH" in
    /mnt/data/siping/runs/npu20ms/nano_search/*)
      echo >&2
      echo "[FAIL] 落点实路径落在 harness 的活仓库下：$REAL_PATH" >&2
      echo "       ⇒ 那是 nano_search 正在跑的活目录，往里写 = 踩别人的作业。" >&2
      echo "       ⇒ 换 REMOTE_ROOT；确认无误后加 FORCE_TARGET=1 可放行。" >&2
      exit 2
      ;;
  esac
  if [ -z "$REAL_PATH" ] || [ "$REAL_PATH" = "$REMOTE_ROOT" ]; then
    :
  else
    echo "  落点实路径: $REAL_PATH"
  fi
  if [ "$IS_LINK" = "UNKNOWN" ]; then
    echo >&2
    echo "[FAIL] 落点状态探测失败（ssh 读不到落点属性）⇒ 为安全起见中止。" >&2
    exit 2
  fi
fi

# 落点是不是「共享 / harness 的活目录」？共享 ⇒ 下面「正在被写」是硬拦；私有 ⇒ 只告警。
IS_SHARED=0
case "$REAL_PATH" in
  /mnt/data/siping/runs/npu20ms/nano_search/*) IS_SHARED=1 ;;
esac
[ "$IS_LINK" = "YES" ] && IS_SHARED=1

# 🔴 背景（2026-11-16 实测）：服务器 13:47 僵死重启后 /tmp 被清空，完整工程 checkout 丢失，
#    /tmp/pycharm_project_10ae9e2e 被改指到 nano_search 的瘦身 repo，其 runs/ 里是**正在被写**的
#    runs/mixed_ft_route_b/（官方 T1 基线缓存落点）⇒ 往那儿传会覆盖别人的活。
#
# ① 「正在被写」判据（主）：落点里 runs/ 的 mtime 在 ACTIVE_WINDOW_MIN 分钟内 ⇒ 有活作业在写 ⇒ 中止。
# ② 「布局不符」判据（辅）：落点根目录有 memory.md 却没有 memory/ ⇒ 不是本工程 ⇒ 中止。
#    （注：harness 会自建 memory/，所以这条会随时间失效 —— 真正兜底的是 ①）
# 两条都可用 FORCE_TARGET=1 放行（确认过再放）。
ACTIVE_WINDOW_MIN="${ACTIVE_WINDOW_MIN:-30}"
if [ "${FORCE_TARGET:-0}" != "1" ]; then
  PROBE=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
    "if [ -e '$REMOTE_ROOT/runs' ]; then stat -c %Y '$REMOTE_ROOT/runs'; else echo NORUNS; fi; date +%s" 2>/dev/null || true)
  RUNS_MTIME=$(echo "$PROBE" | head -1)
  NOW_EPOCH=$(echo "$PROBE" | tail -1)
  if [ "$RUNS_MTIME" = "NORUNS" ]; then
    echo "  落点无 runs/（本工程已清空产物 ⇒ 正常）"
  elif [ -n "$RUNS_MTIME" ] && [ -n "$NOW_EPOCH" ] && [ "$RUNS_MTIME" -gt 0 ] 2>/dev/null; then
    AGE_MIN=$(( (NOW_EPOCH - RUNS_MTIME) / 60 ))
    if [ "$AGE_MIN" -lt "$ACTIVE_WINDOW_MIN" ]; then
      if [ "$IS_SHARED" = "1" ]; then
        echo >&2
        echo "[FAIL] 落点正在被写：$REMOTE_ROOT/runs 的 mtime 是 ${AGE_MIN} 分钟前（< ${ACTIVE_WINDOW_MIN} 分钟）" >&2
        echo "       ⇒ 这是**共享**落点（harness 活目录），有活作业正在里面跑。" >&2
        echo "       ⇒ 往这儿 upload 会覆盖对方正在写的东西 ⇒ 中止。" >&2
        echo "       ⇒ 改 REMOTE_ROOT 指向属于你的落点，或确认无误后加 FORCE_TARGET=1 重跑。" >&2
        exit 2
      else
        echo "[warn] 落点 runs/ 最近 ${AGE_MIN} 分钟内有改动（私有落点 ⇒ 只告警，不中止）"
      fi
    else
      echo "  落点 runs/ 最后修改于 ${AGE_MIN} 分钟前（> ${ACTIVE_WINDOW_MIN} 分钟，判定为空闲）"
    fi
  fi
  D=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
      "cd '$REMOTE_ROOT' 2>/dev/null && { [ -f memory.md ] && [ ! -d memory ]; } && echo LAYOUT_MISMATCH || echo LAYOUT_OK" \
      2>/dev/null || echo UNKNOWN)
  if [ "$D" = "LAYOUT_MISMATCH" ]; then
    echo >&2
    echo "[FAIL] 落点布局与本工程不符：那边根目录有 memory.md 但没有 memory/ 目录。" >&2
    echo "       ⇒ 通常是服务器上另一个工程的 checkout。请改 REMOTE_ROOT 或加 FORCE_TARGET=1。" >&2
    exit 2
  fi
fi

echo "== [2/3] rsync =="
# -e 指定端口；--delete 不带 ⇒ 服务器多余文件不会被删
rsync "${RSYNC_ARGS[@]}" -e "ssh ${SSH_OPTS[*]}" "$REPO/" "$SSH_HOST:$REMOTE_ROOT/"

if [ "$DRY_RUN" = "1" ]; then
  echo "== [3/3] DRY-RUN 结束（未落盘）。确认无误后：DRY_RUN=0 再跑一次 =="
  exit 0
fi

echo "== [3/3] 同步后校验（铁律三：已同步 ≠ 已是最新，必须比对） =="
# 挑一个本次一定传了的源码文件做双侧 md5 比对
SENTINEL="${SENTINEL:-manu/memory/rules.md}"
LOCAL_MD5=$(md5sum "$REPO/$SENTINEL" | awk '{print $1}')
REMOTE_MD5=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "md5sum '$REMOTE_ROOT/$SENTINEL' 2>/dev/null | awk '{print \$1}'")
if [ "$LOCAL_MD5" = "$REMOTE_MD5" ] && [ -n "$REMOTE_MD5" ]; then
  echo "  [OK] $SENTINEL 双侧 md5 一致：$LOCAL_MD5"
else
  echo "  [FAIL] $SENTINEL md5 不一致：本地=$LOCAL_MD5 服务器=$REMOTE_MD5" >&2
  exit 1
fi
echo "== 完成 =="
