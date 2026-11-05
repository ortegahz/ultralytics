#!/usr/bin/env bash
# ============================================================
# 交付包**一键复核**（host 侧）：一条命令跑完全部复核，给出 PASS/FAIL 摘要
#
#   做四件事：
#     1. 包级清单 `md5sum -c`（覆盖包内每个文件，**含**镜像清单 `repo/MD5SUMS.txt`）
#     2. 镜像级清单 `md5sum -c`（覆盖镜像每个文件）
#     3. 包内六份审计（**包内模式**，只用包内 `repo/` 镜像）
#     4. 变异测试（证明审计**有判别力**：篡改必报警）
#
#   ★ 全程在**临时副本**里跑（`mktemp -d`）⇒ **不改动交付包本体**
#     （包内复跑审计会把运行产物写进 `repo/runs/lookahead/fi_nogmc/audit_round7/`，
#      直接在包内跑会污染交付包 ⇒ 必须走副本）。
#
#   ★ 执行**崩溃**（进程异常退出 / traceback）≠ 执行**判了失败**：
#     一次偶发环境抖动若被当成判据失败，就会误报「交付包不可信」⇒ 那是假警报，
#     比不报警更坏。⇒ 崩溃**重试**（最多 3 次）；真判据失败（有结论、无 traceback）
#     ⇒ 立即 FAIL，不重试。
#
#   用法：bash verify_all.sh            # 随包：bash <包>/code/verify_all.sh
#         bash manu/tracking/lookahead/server_overnight/verify_all.sh   # 仓内
#   退出码：0 = 全部通过；1 = 有 FAIL。
# ============================================================
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
if [ "$(basename "$SELF_DIR")" = "code" ]; then
  PKG="$(cd "$SELF_DIR/.." && pwd)"          # 随包：<包>/code/verify_all.sh
else
  # ★ 脚本不在名为 code 的目录下 ⇒ **先按脚本自身位置找包**（host 上服务器路径不存在，
  #   写死兜底会让「拿到包就复核」这条路直接 exit 2）⇒ 显式 PKG > 脚本位置 > 服务器路径。
  if [ -z "${PKG:-}" ] && [ -f "$SELF_DIR/../code/audit_pack_selfsufficient.py" ]; then
    PKG="$(cd "$SELF_DIR/.." && pwd)"
  else
    PKG="${PKG:-/tmp/pycharm_project_10ae9e2e/runs/lookahead/final/nogmc_fi_20261104}"
  fi
fi
[ -f "$PKG/code/audit_pack_selfsufficient.py" ] || { echo "[FATAL] 交付包不存在或缺 code/：$PKG"; exit 2; }

# ★ **md5sum 存在性闸**：没有 md5sum（macOS 只有 `md5`）⇒ 两道清单校验的命令**不存在**，
#   报错被重定向进日志 ⇒ `grep -c` 得 0 ⇒ 被当成「没有失败项」⇒ **两道锁被跳过却报绿**。
#   ⇒ 缺 md5sum 必须 [FATAL] 退出，绝不「没验就说验过」。
command -v md5sum >/dev/null 2>&1 || { echo "[FATAL] 缺 md5sum（本包的两道清单校验依赖它）"; exit 2; }

# ★ 运行环境闸：**Python ≥ 3.8**（交付链刻意只用 3.8 通用语法/API ⇒ 系统 python3 即可）。
#   早先审计脚本用了 `str.removeprefix`（3.9+）⇒ 在 python3.8 的机器上**必崩**
#   （实测 AttributeError）⇒ host 复核直接失败。现已改为 3.7+ 通用写法。
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' \
  || { echo "[FATAL] 需要 Python ≥ 3.8（当前 $PY：$("$PY" -V 2>&1)）"; exit 2; }

SB="$(mktemp -d /tmp/nogmc_fi_verify_XXXXXX)" || { echo "[FATAL] mktemp 失败"; exit 2; }
[ -n "$SB" ] && [ -d "$SB" ] || { echo "[FATAL] 沙箱目录创建失败：$SB"; exit 2; }
trap 'rm -rf "$SB"' EXIT
cp -a "$PKG" "$SB/pkg"
cd "$SB/pkg" || exit 2

echo "交付包：$PKG"
echo "复核副本：$SB/pkg（★ 交付包本体不被改动）"
echo "========================================================================"

fail=0
n_crash=0

# ★ 执行**崩溃**（进程异常退出 / traceback）≠ 执行**判了失败**：
#   崩溃 ⇒ 重试（最多 3 次）；真判据失败（有结论、无 traceback）⇒ 立即 FAIL、不重试。
#   输出：<rc> <本次崩溃次数>（用子 shell 调用 ⇒ 计数靠返回值带回，不能改全局变量）。
run_retry() {   # $1=日志  $2=重试判据正则  $3..=命令
  local log="$1" pat="$2"; shift 2
  local attempt=0 rc=1 nc=0
  while [ "$attempt" -lt 3 ]; do
    attempt=$((attempt + 1))
    "$@" > "$log" 2>&1; rc=$?
    if [ "$rc" -eq 0 ]; then break; fi
    if ! grep -qE "$pat" "$log"; then break; fi     # 真判据失败 ⇒ 不重试
    nc=$((nc + 1))
    echo "    （第 $attempt 次执行崩溃 ⇒ 环境异常 ⇒ 重试）" >&2
    sleep 2
  done
  echo "$rc $nc"
}

echo "=== 1/5 包级清单 md5sum -c ==="
md5sum -c MD5SUMS.txt > "$SB/pkg_sums.log" 2>&1
n_ok=$(grep -c ': OK$' "$SB/pkg_sums.log"); n_bad=$(grep -c 'FAILED' "$SB/pkg_sums.log")
echo "  OK $n_ok 条 / FAILED $n_bad 条"
# ★ 判据必须是「**有**通过项 **且** 无失败项」—— 只看 n_bad 会在「一条都没验」时误判为通过
[ "$n_ok" -gt 0 ] && [ "$n_bad" -eq 0 ] || { fail=1; grep 'FAILED' "$SB/pkg_sums.log" | head -5; }

echo "=== 2/5 镜像级清单 md5sum -c ==="
( cd repo && md5sum -c MD5SUMS.txt ) > "$SB/mir_sums.log" 2>&1
m_ok=$(grep -c ': OK$' "$SB/mir_sums.log"); m_bad=$(grep -c 'FAILED' "$SB/mir_sums.log")
echo "  OK $m_ok 条 / FAILED $m_bad 条"
[ "$m_ok" -gt 0 ] && [ "$m_bad" -eq 0 ] || { fail=1; grep 'FAILED' "$SB/mir_sums.log" | head -5; }

echo "=== 3/5 包内六份审计（包内模式） ==="
audit_ok=0
for s in audit_sec5_deliverables audit_sec5_acceptance audit_sec16_gap \
         audit_sec17_reg_oracle audit_sec17_rbgate audit_pack_selfsufficient; do
  _o=$(run_retry "$SB/audit_$s.log" "Traceback \(most recent call last\)" "$PY" "code/$s.py")
  rc=${_o%% *}; _nc=${_o##* }
  n_crash=$((n_crash + _nc))
  nf=$(grep -oE "FAIL [0-9]+ 项" "$SB/audit_$s.log" | head -1)
  _rt=""; [ "$_nc" -gt 0 ] && _rt="（崩溃重试 $_nc 次）"
  echo "  $s rc=$rc ${nf:-（无失败项）}$_rt"
  if [ "$rc" -eq 0 ]; then audit_ok=$((audit_ok+1)); else fail=1; fi
done

echo "=== 4/5 变异测试（审计的判别力） ==="
_o=$(run_retry "$SB/mutation.log" "Traceback \(most recent call last\)|变异测试.*ENV" \
     "$PY" code/mutation_test_selfsufficient.py)
rc_mut=${_o%% *}; _nc=${_o##* }
n_crash=$((n_crash + _nc))
grep -E "变异测试：|判定：" "$SB/mutation.log" | sed 's/^/  /'
[ "$rc_mut" -eq 0 ] || fail=1

echo "=== 5/5 摘要 ==="
echo "  包级清单：OK $n_ok / FAILED $n_bad"
echo "  镜像清单：OK $m_ok / FAILED $m_bad"
echo "  包内审计：$audit_ok / 6 份 rc=0"
echo "  执行崩溃重试：$n_crash 次（崩溃 ≠ 判据失败 ⇒ 已重试；重试后仍崩 ⇒ 环境异常）"
echo "========================================================================"
if [ "$fail" -eq 0 ]; then
  echo "结论：交付包复核**全部通过**（清单逐位一致 + 六份审计 rc=0 + 变异测试无违例）"
else
  echo "结论：交付包复核**有 FAIL**（详见上方日志）"
fi
exit "$fail"
