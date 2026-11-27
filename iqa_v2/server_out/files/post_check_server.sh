#!/bin/bash
# ============================================================================
# 服务器侧「完工自检」：等 codex 主会话退出 → 自动跑三道门禁 + 完成度审计
#   目的: host 关机后无人值守，证据仍然留在服务器上（host 回来即可直接取）
#
#   监控目标 PID = 2505229（codex exec 主进程）
#   产物: $SP/iqa_v2/output/post_check/{post_check.log,gates/,audit/,stage/}
#   本脚本只读 iqa_v2/ 的产物，只写自己的 post_check/ 目录 ⇒ 不与主会话抢文件
# ============================================================================
SP=/tmp/pycharm_project_a7d966e6
PID=2505229
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
OUT="$SP/iqa_v2/output/post_check"
mkdir -p "$OUT/gates" "$OUT/audit" "$OUT/stage"
LOG="$OUT/post_check.log"

exec >>"$LOG" 2>&1
echo "=========================================================="
echo "[$(date '+%F %T')] watcher 启动，监控 PID=$PID"
echo "[$(date '+%F %T')] 等待主会话结束（每 20s 轮询）..."
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
echo "[$(date '+%F %T')] 主会话 PID=$PID 已退出 ⇒ 开始门禁"
cd "$SP" || { echo "FATAL: cd $SP 失败"; exit 1; }
export PYTHONPATH=/tmp/pylibs_cvcontrib

echo; echo "=== [1/4] pytest iqa_v2/test_iqa_v2.py ==="
$PY -m pytest iqa_v2/test_iqa_v2.py -v 2>&1 | tail -45 | tee "$OUT/gates/pytest_tail.txt"
$PY -m pytest iqa_v2/test_iqa_v2.py -q 2>&1 | tail -6 > "$OUT/gates/pytest_summary.txt"

echo; echo "=== [2/4] verify_recalc_v2.py --samples 120 ==="
$PY iqa_v2/verify_recalc_v2.py --samples 120 2>&1 | tail -18 | tee "$OUT/gates/verify.txt"

echo; echo "=== [3/4] md5sum -c iqa_v2/output/manifest_v2.md5 ==="
md5sum -c iqa_v2/output/manifest_v2.md5 2>&1 | tee "$OUT/gates/md5.txt" | tail -20
echo "md5 OK=$(grep -c ': OK$' "$OUT/gates/md5.txt")  FAILED=$(grep -c 'FAILED' "$OUT/gates/md5.txt")"

echo; echo "=== [4/4] audit_final.py 完成度审计 ==="
# 拼一个 staging 目录：output/*.csv + 缓存目录里的退化明细底表
rm -f "$OUT/stage"/*.csv
cp -f iqa_v2/output/*.csv "$OUT/stage/" 2>/dev/null
for c in /tmp/iqa_v2_cache/degrade_detail.csv /tmp/iqa_v2_cache_smoke/degrade_detail.csv; do
  [ -f "$c" ] && cp -f "$c" "$OUT/stage/" && echo "staged: $c"
done
ls -la "$OUT/stage/" | tee "$OUT/audit/stage_listing.txt"
if [ -f iqa_v2/audit_final.py ]; then
  $PY iqa_v2/audit_final.py "$OUT/stage" iqa_v2/REPORT_V2_FINAL.md 2>&1 | tee "$OUT/audit/audit_final.txt"
else
  echo "WARN: 服务器上无 iqa_v2/audit_final.py ⇒ 跳过完成度审计"
fi

echo; echo "=== 交付物清单 ==="
ls -la iqa_v2/REPORT_V2_FINAL.md iqa_v2/MEMORY_SECTION.md iqa_v2/PROGRESS_SERVER.md 2>&1 | tee "$OUT/audit/deliverables.txt"
echo; echo "=== 主会话最后 6 条 agent_message ==="
grep -o '"type":"agent_message","text":"[^"]*"' /tmp/codex_iqa_v2.log | tail -6 | cut -c1-300 | tee "$OUT/audit/last_messages.txt"

echo; echo "[$(date '+%F %T')] ✅ 完工自检全部结束"
