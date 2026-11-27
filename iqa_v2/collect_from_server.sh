#!/bin/bash
# ============================================================================
# 服务器侧交付物回收 + 门禁复跑 + 逐字节比对（**在 host 上跑**）
#
#   用法: bash iqa_v2/collect_from_server.sh
#   可选: REGEN_MANIFEST=1 bash iqa_v2/collect_from_server.sh
#         （在**服务器侧**重生成 iqa_v2/output/manifest_v2.md5 —— **默认关闭**：
#           它会改服务器状态，且会让 md5sum -c 退化成「对刚生成的清单自证」；
#           只有确认产物已定稿、且门禁红确实由清单过期造成时才显式打开）
#
# 流程（**任何一步失败 ⇒ 打印错误原文 ⇒ exit 1**，绝不假绿）：
#   0) [可选] 服务器侧重生成 md5 清单（REGEN_MANIFEST=1）→ gates/manifest_regen.txt
#   1) 拉回交付物：iqa_v2/*.py|*.md|*.sh ＋ output/ **整棵子树** ＋ 退化明细底表
#      → host 的 iqa_v2/server_out/files/（**独立目录**，绝不覆盖 host 工作副本），
#      并**逐项核对**：服务器 output/ 的每个文件、每个子目录都必须在本地存在，
#      文件数必须一致 ⇒ 缺任一项 ⇒ exit 1
#      ⚠️ 拉回**先于**门禁：host 随时可能关机，产出物必须先落地；
#         门禁红 ⇒ 立刻 exit 1（但产出物已经在 host 手里）。
#   2) 检查 md5 清单是否**过期**（清单比产物旧 ⇒ md5sum -c 无意义）
#   3) 服务器侧三道门禁（pytest / verify_recalc / md5sum -c）：**逐条看退出码**，
#      三道都跑完（证据完整），任一非 0 ⇒ 打印原文 ⇒ exit 1（不再往下跑，
#      绝不打印任何绿色结论）
#   4) 服务器 ↔ host 逐文件 md5 比对（**双向清单**；共有文件 0 个 ⇒ exit 1）
#   5) 打印一页结论（GATES: pytest=… verify=… md5sum=… manifest=…）
#
# ⚠️ host 上没有 cv2、也没有 /mnt/data ⇒ **pytest / verify_recalc 只能在服务器上跑**
#    （本脚本就是代跑 + 回传证据），host 只能做 md5 与人工审阅。
# ============================================================================
set -euo pipefail

HOST=huangzhe@192.168.99.40
PORT=32222
SSH="ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -o ConnectTimeout=10 -p $PORT $HOST"
SP=/tmp/pycharm_project_a7d966e6
HP="$(cd "$(dirname "$0")" && pwd)"
OUT="$HP/server_out"
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
TMP="$(mktemp -d /tmp/iqa_collect.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# 每轮**清空**回收目录 ⇒ 计数与比对都是「本轮」口径，不掺上一轮遗留
rm -rf "$OUT/files" "$OUT/gates"
mkdir -p "$OUT/gates" "$OUT/files"

RC_PYTEST=1
RC_VERIFY=1
RC_MD5=1
MANIFEST_STATE=UNKNOWN
GATE_FAIL=0

# ---------- 小工具 ----------
# 服务器只读命令：stdout/stderr 全落证据文件，**退出码由调用方判**
srv() {          # srv <证据名> <远程命令>
  local tag="$1"; shift
  local rc=0
  $SSH "$*" > "$OUT/gates/$tag.txt" 2>&1 || rc=$?
  echo "$rc" > "$OUT/gates/$tag.rc"
  return "$rc"
}

# scp：stderr 落证据文件、退出码由调用方判；**恒带 -r**（目录不递归 ⇒ 整棵子树丢失）
pull() {         # pull <证据名> <本地目标> <远端路径...>
  local tag="$1" dest="$2"; shift 2
  local rc=0 spec
  local specs=()
  for spec in "$@"; do specs+=("$HOST:$spec"); done
  scp -r -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
      -o BatchMode=yes -o ConnectTimeout=10 -P "$PORT" "${specs[@]}" "$dest" \
      > "$OUT/gates/$tag.log" 2>&1 || rc=$?
  echo "$rc" > "$OUT/gates/$tag.rc"
  return "$rc"
}

# 总判定行（结论页与失败路径共用）
verdict() {
  local py=FAIL vf=FAIL md=FAIL
  if [ "$RC_PYTEST" -eq 0 ]; then py=PASS; fi
  if [ "$RC_VERIFY" -eq 0 ]; then vf=PASS; fi
  if [ "$RC_MD5" -eq 0 ]; then md=PASS; fi
  echo "GATES: pytest=$py verify=$vf md5sum=$md manifest=$MANIFEST_STATE"
}

# 硬失败：打印说明 +（可选）错误原文 ⇒ exit 1
fail() {          # fail <说明> [错误原文文件]
  echo
  echo "🔴 [FAIL] $1"
  if [ -n "${2:-}" ] && [ -s "${2:-}" ]; then
    echo "──── 错误原文（${2}，前 30 行）────"
    head -30 "${2}"
    echo "────────────────────────────────────"
  fi
  verdict
  echo "⇒ ❌ 不可据此宣告交付完成"
  echo "证据目录: $OUT/gates/  ·  回收文件: $OUT/files/"
  exit 1
}

# ---------- 0) [可选] 服务器侧重生成 md5 清单 ----------
echo "===== 0) md5 清单重生成（服务器侧）====="
if [ "${REGEN_MANIFEST:-0}" = "1" ]; then
  echo "⚠️  REGEN_MANIFEST=1 ⇒ 将改写服务器 iqa_v2/output/manifest_v2.md5"
  rc=0
  srv manifest_regen "cd $SP && $PY iqa_v2/make_manifest_v2.py" || rc=$?
  cat "$OUT/gates/manifest_regen.txt"
  if [ "$rc" -ne 0 ]; then
    fail "服务器侧重生成 md5 清单失败（rc=$rc）" "$OUT/gates/manifest_regen.txt"
  fi
  MANIFEST_STATE=REGENERATED
  echo "✅ 清单已在服务器侧重生成（gates/manifest_regen.txt）"
else
  echo "⏭️  跳过（**默认关闭**；需要时 REGEN_MANIFEST=1）"
  echo "skipped (REGEN_MANIFEST!=1)" > "$OUT/gates/manifest_regen.txt"
  MANIFEST_STATE=UNCHANGED
fi

# ---------- 1) 拉回交付物（先于门禁：host 随时可能关机）----------
echo
echo "===== 1) 拉回交付物（进独立目录，不覆盖工作副本）====="
# 服务器侧 output/ 清单（**拷贝前**取）—— 用来逐项核对拷贝结果
rc=0
srv out_files "cd $SP && find iqa_v2/output -type f | sed 's|^iqa_v2/output/||' | sort" || rc=$?
if [ "$rc" -ne 0 ]; then
  fail "服务器侧取 output/ 文件清单失败（ssh rc=$rc）" "$OUT/gates/out_files.txt"
fi
rc=0
srv out_dirs "cd $SP && find iqa_v2/output -mindepth 1 -type d | sed 's|^iqa_v2/output/||' | sort" || rc=$?
if [ "$rc" -ne 0 ]; then
  fail "服务器侧取 output/ 目录清单失败（ssh rc=$rc）" "$OUT/gates/out_dirs.txt"
fi

# 顶层交付物（可能还没生成 ⇒ 缺失不算错，但证据留档）
for f in REPORT_V2_FINAL.md MEMORY_SECTION.md PROGRESS_SERVER.md; do
  if pull "scp_$f" "$OUT/files/" "$SP/iqa_v2/$f"; then
    echo "拉回 $f"
  else
    echo "（$f 尚未生成，属正常；见 gates/scp_$f.log）"
  fi
done
# 代码 / 文档 / 脚本（扁平落到 files/）
if ! pull scp_code "$OUT/files/" "$SP/iqa_v2/*.py" "$SP/iqa_v2/*.md" "$SP/iqa_v2/*.sh"; then
  fail "拉回代码/文档/脚本失败（scp 报错见 gates/scp_code.log）" "$OUT/gates/scp_code.log"
fi
# output/ 产物：**-r** 才能把 post_check/ 这类目录整棵树带回来
if ! pull scp_output "$OUT/files/" "$SP/iqa_v2/output/."; then
  fail "拉回 output/ 失败（scp -r 报错见 gates/scp_output.log）" "$OUT/gates/scp_output.log"
fi
# 退化仿真明细底表（不在 output/ 里，在 cache 目录；R1/R2 审计要用）
# ⚠️ 全量与冒烟**必须落成不同文件名** —— 同名会让后拉取的冒烟覆盖全量，
#    审计就会判在 60 窗口的抽样上（R1/R2 结论全错）。
if pull scp_degrade_full "$OUT/files/degrade_detail.full.csv" "/tmp/iqa_v2_cache/degrade_detail.csv"; then
  echo "拉回全量明细 degrade_detail.full.csv"
else
  fail "拉回全量明细 degrade_detail.csv 失败（R1/R2 审计缺底表）" "$OUT/gates/scp_degrade_full.log"
fi
if pull scp_degrade_smoke "$OUT/files/degrade_detail.smoke.csv" "/tmp/iqa_v2_cache_smoke/degrade_detail.csv"; then
  echo "拉回冒烟明细 degrade_detail.smoke.csv"
else
  echo "（无冒烟明细；不阻断）"
fi

# 逐项核对：服务器 output/ 的每个文件、每个子目录都必须落到本地
n_exp=0
n_got=0
missing=""
while IFS= read -r rel; do
  if [ -z "$rel" ]; then continue; fi
  n_exp=$((n_exp + 1))
  if [ -f "$OUT/files/$rel" ]; then
    n_got=$((n_got + 1))
  else
    missing="$missing $rel"
  fi
done < "$OUT/gates/out_files.txt"

n_dirs=0
d_missing=""
while IFS= read -r rel; do
  if [ -z "$rel" ]; then continue; fi
  n_dirs=$((n_dirs + 1))
  if [ ! -d "$OUT/files/$rel" ]; then
    d_missing="$d_missing $rel"
  fi
done < "$OUT/gates/out_dirs.txt"

n_pulled=$(find "$OUT/files" -type f | wc -l)
echo "服务器 output/: 文件 $n_exp 个、目录 $n_dirs 个 ｜ 本地命中文件 $n_got 个 ｜ 本轮共拉回 $n_pulled 个文件"
if [ "$n_exp" -eq 0 ]; then
  fail "服务器 output/ 里一个文件都没有 ⇒ 产物没生成（或清单取错）" "$OUT/gates/out_files.txt"
fi
if [ -n "$missing" ]; then
  echo "🔴 缺失文件:$missing"
fi
if [ -n "$d_missing" ]; then
  echo "🔴 缺失目录:$d_missing（子树没跟回来 ⇒ 证据缺了一整块）"
fi
if [ "$n_got" -ne "$n_exp" ] || [ -n "$missing" ] || [ -n "$d_missing" ]; then
  fail "拷贝核对不通过（缺文件/缺目录/数量不符）⇒ 证据树不完整，不能当作「产物正常」" "$OUT/gates/scp_output.log"
fi
echo "✅ 拷贝核对通过：output/ 的 $n_exp 个文件 + $n_dirs 个子目录全部落到本地"

# ---------- 2) md5 清单新鲜度 ----------
echo
echo "===== 2) md5 清单新鲜度（清单旧于产物 ⇒ md5sum -c 无意义）====="
rc=0
srv manifest_stale "cd $SP && find iqa_v2 -type f -newer iqa_v2/output/manifest_v2.md5 -not -path '*post_check*' -not -name 'manifest_v2.md5' | sort" || rc=$?
if [ "$rc" -ne 0 ]; then
  fail "服务器侧查清单新鲜度失败（ssh rc=$rc）" "$OUT/gates/manifest_stale.txt"
fi
n_stale=$(grep -c . "$OUT/gates/manifest_stale.txt" || true)
if [ "$n_stale" -gt 0 ]; then
  echo "🔴 [FAIL] md5 清单**已过期**：有 $n_stale 个文件比清单新（md5sum -c 的结果不可信）"
  echo "     前 10 条: $(head -10 "$OUT/gates/manifest_stale.txt" | tr '\n' ' ')"
  echo "     ⇒ 需在服务器侧重跑 'python iqa_v2/make_manifest_v2.py'（或 REGEN_MANIFEST=1）后再复跑本脚本"
  MANIFEST_STATE=STALE
  GATE_FAIL=1
else
  echo "✅ md5 清单新鲜（无产物比清单新）"
  MANIFEST_STATE=FRESH
fi

# ---------- 3) 服务器侧三道门禁 ----------
echo
echo "===== 3) 服务器侧三道门禁（逐条看退出码）====="
rc=0
srv pytest "cd $SP && export PYTHONPATH=/tmp/pylibs_cvcontrib && $PY -m pytest iqa_v2/test_iqa_v2.py -v" || rc=$?
RC_PYTEST=$rc
printf '[门禁] %-12s rc=%s\n' pytest "$rc"

rc=0
srv verify "cd $SP && export PYTHONPATH=/tmp/pylibs_cvcontrib && $PY iqa_v2/verify_recalc_v2.py --samples 120" || rc=$?
RC_VERIFY=$rc
printf '[门禁] %-12s rc=%s\n' verify "$rc"

rc=0
srv md5 "cd $SP && md5sum -c iqa_v2/output/manifest_v2.md5" || rc=$?
RC_MD5=$rc
printf '[门禁] %-12s rc=%s\n' md5 "$rc"

# pytest 的判定**以退出码为准**（FAILED 行前面有 ANSI 转义，不能只看 grep 计数）
n_fail=$(grep -acE '(FAILED|ERROR)' "$OUT/gates/pytest.txt" || true)
if [ "$n_fail" -gt 0 ]; then n_fail_txt="$n_fail"; else n_fail_txt="无匹配行"; fi
n_ok=$(grep -ac ': OK$' "$OUT/gates/md5.txt" || true)
n_md5fail=$(grep -ac 'FAILED' "$OUT/gates/md5.txt" || true)
# verify 必须能打出「最大绝对误差」结论行，否则视为「无法确认」（不能当 PASS）
verify_line=$(grep -aiE '最大绝对误差|max.*err|max_abs' "$OUT/gates/verify.txt" | tail -1 || true)

gate_failed=0
if [ "$RC_PYTEST" -ne 0 ]; then
  echo "🔴 [FAIL] pytest 退出码 $RC_PYTEST（失败/错误行数: $n_fail_txt）—— **以退出码为准**"
  gate_failed=1
fi
if [ "$RC_VERIFY" -ne 0 ]; then
  echo "🔴 [FAIL] verify_recalc 退出码 $RC_VERIFY"
  gate_failed=1
fi
if [ "$RC_MD5" -ne 0 ]; then
  echo "🔴 [FAIL] md5sum -c 退出码 $RC_MD5（OK $n_ok 行 / FAILED $n_md5fail 行）"
  gate_failed=1
elif [ "$n_ok" -eq 0 ]; then
  echo "🔴 [FAIL] md5sum -c 退出码 0 但 0 行 OK ⇒ 清单为空/无有效条目（比对无效）"
  gate_failed=1
fi
if [ "$RC_VERIFY" -eq 0 ] && [ -z "$verify_line" ]; then
  echo "🔴 [FAIL] verify 退出码 0 但输出里没有「最大绝对误差」结论行 ⇒ 模式不命中/输出异常，不能当 PASS"
  gate_failed=1
fi
if [ "$gate_failed" -ne 0 ]; then
  for g in pytest verify md5; do
    if [ "$(cat "$OUT/gates/$g.rc")" -ne 0 ]; then
      echo "──── $g 错误原文（前 30 行，完整见 $OUT/gates/$g.txt）────"
      head -30 "$OUT/gates/$g.txt"
      echo "──────────────────────────────────────────────"
    fi
  done
  fail "服务器侧门禁未通过 ⇒ 不可据此宣告交付完成"
fi
if [ "$MANIFEST_STATE" = "STALE" ]; then
  fail "md5 清单已过期（$n_stale 个文件比清单新）⇒ md5sum -c 与逐字节比对都不可信；请在服务器侧重生成清单后复跑"
fi

# ---------- 4) 服务器 ↔ host 逐文件 md5 比对 ----------
echo
echo "===== 4) 服务器 ↔ host 逐文件 md5 比对（双向清单）====="
rc=0
srv files_server "cd $SP && find iqa_v2 -type f -not -path '*__pycache__*' -not -path '*post_check*' -exec md5sum {} +" || rc=$?
if [ "$rc" -ne 0 ]; then
  fail "服务器侧 md5 清单抓取失败（ssh rc=$rc）" "$OUT/gates/files_server.txt"
fi
# 服务器路径 iqa_v2/output/x.csv ⇒ 归一成 x.csv（与 host 拉回后的扁平布局对齐）
# ⚠️ 只收 md5 形态的行（长度 32 的十六进制）⇒ ssh 报错文本不会混进清单
awk 'length($1)==32 && $1 ~ /^[0-9a-fA-F]+$/ {n=$2; sub(/^iqa_v2\//,"",n); sub(/^output\//,"",n); print tolower($1)"  "n}' \
  "$OUT/gates/files_server.txt" | sort -k2 > "$TMP/srv.md5"
n_srv=$(grep -c . "$TMP/srv.md5" || true)
if [ "$n_srv" -eq 0 ]; then
  fail "服务器侧 md5 清单为空/不可解析 ⇒ 比对无效（**不是**「一致」）" "$OUT/gates/files_server.txt"
fi
(cd "$OUT/files" && find . -type f -not -path '*post_check*' -exec md5sum {} + | sed 's|  \./|  |' | sort -k2) > "$TMP/hst.md5"
n_hst=$(grep -c . "$TMP/hst.md5" || true)

awk '{print $2}' "$TMP/srv.md5" | sort > "$TMP/srv_names"
awk '{print $2}' "$TMP/hst.md5" | sort > "$TMP/hst_names"
comm -12 "$TMP/srv_names" "$TMP/hst_names" > "$TMP/both_names"
n_both=$(grep -c . "$TMP/both_names" || true)
# 🔴 用 FILENAME==ARGV[1] 而不是 NR==FNR：第一个文件为空时 NR==FNR 会对第二份恒真 ⇒ 结果恒空 ⇒ 假绿
awk 'FILENAME==ARGV[1]{b[$2]=$1; next} ($2 in b){print b[$2]"  "$2}' \
    "$TMP/srv.md5" "$TMP/hst.md5" | sort -k2 > "$TMP/srv_sub"
# 🔴 共有文件名单是**单列** ⇒ 用 $1 建表，再按 $2 **精确**匹配
#    （不用 grep -F -f：'windows.csv' 会子串误配 'old_windows.csv.bak'）
awk 'FILENAME==ARGV[1]{common[$1]=1; next} ($2 in common){print}' \
    "$TMP/both_names" "$TMP/hst.md5" | sort -k2 > "$TMP/hst_sub"
echo "服务器侧记录: $n_srv 条 ｜ host 侧拉回: $n_hst 条 ｜ 共有: $n_both"
echo "仅服务器有: $(comm -23 "$TMP/srv_names" "$TMP/hst_names" | tr '\n' ' ')"
echo "仅 host 有 : $(comm -13 "$TMP/srv_names" "$TMP/hst_names" | tr '\n' ' ')"
if [ "$n_both" -eq 0 ]; then
  fail "共有文件 0 个 ⇒ 逐字节比对**无效**（比 0 个文件等于没比对，不能报「一致」）"
fi
if diff -q "$TMP/srv_sub" "$TMP/hst_sub" >/dev/null 2>&1; then
  echo "✅ 共有 $n_both 个文件**逐字节一致**"
else
  echo "🔴 [FAIL] 共有文件里有不一致："
  diff "$TMP/srv_sub" "$TMP/hst_sub" | head -20
  fail "拉回的内容与服务器不一致（服务器若仍在改文件，等流水线停下后再复核一次）"
fi
# 两侧清单落盘 ⇒ 比对证据可复现
cp "$TMP/srv.md5" "$OUT/gates/files_server_norm.md5"
cp "$TMP/hst.md5" "$OUT/gates/files_host.md5"

# ---------- 5) 结论 ----------
echo
echo "===== 5) 结论 ====="
echo "pytest  退出码       : $RC_PYTEST"
echo "pytest  判定         : ✅ PASS（退出码 0；失败/错误行数: $n_fail）"
echo "pytest  summary      : $(grep -aE '[0-9]+ (passed|failed|error)' "$OUT/gates/pytest.txt" | tail -1 || true)"
echo "verify  退出码       : $RC_VERIFY"
echo "verify  最大绝对误差行 : $verify_line"
echo "md5sum  退出码       : $RC_MD5"
echo "md5sum  OK 行数      : $n_ok"
echo "md5sum  FAILED 行数  : $n_md5fail"
echo "md5 清单状态         : $MANIFEST_STATE（过期文件数 $n_stale）"
echo "本轮拉回文件数       : $n_pulled（服务器 output/ 命中 $n_got/$n_exp，子目录 $n_dirs 个）"
echo "共有文件数           : $n_both"
verdict
if [ "$GATE_FAIL" -eq 0 ]; then
  echo "⇒ ✅ 全绿（可据此宣告交付完成）"
else
  echo "⇒ ❌ 有门禁未通过，不可据此宣告交付完成"
fi
echo "证据目录: $OUT/gates/  ·  回收文件: $OUT/files/"
exit "$GATE_FAIL"
