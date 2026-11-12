#!/bin/bash
# 明早自动出报告（host 已关机也能跑：独立进程，PPID=1）
# 用法（服务器）：
#   setsid nohup bash manu/pipeline/npu/run_morning_report.sh "2026-11-13 09:10" >/dev/null 2>&1 </dev/null &
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
W=/mnt/data/siping/runs/npu20ms/nano_search
AT="${1:-2026-11-13 09:10}"
STAMP=$(date -d "$AT" +%Y%m%d)
OUT=$W/morning_report_${STAMP}.txt
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

while [ "$(date +%s)" -lt "$(date -d "$AT" +%s)" ]; do sleep 60; done
sleep 45

{
  echo "===== 明早自动报告 $(date "+%F %T") ====="
  echo "--- ① 进程状态（调度器到点应已退出） ---"
  pgrep -af "adaptive_runner.py --deadline" || echo "调度器已退出 ✔"
  pgrep -af "run_adaptive_watchdog.sh" || echo "守护已退出 ✔"
  echo
  echo "--- ② 轮次状态 ---"
  cat $W/round_state.json 2>/dev/null; echo
  echo
  echo "--- ③ 热升级核查（新代码在哪一轮生效 + 新分配是否真的跑了） ---"
  echo "（各轮次队列规模）"
  grep -n "训练 .* 臂" $W/adaptive.log | tail -12
  echo "（新分配的证据：近失复现 / 种子扫描 / 优先格 mixw1.9 / 突变）"
  grep -n "近失复现\|种子扫描\|优先格 .*mixw1\.9\|突变 " $W/adaptive.log | tail -30
  echo
  echo "--- ③b 今夜算力配比是否按 §150.7.4 生效（种子格 20 / 突变 ≤5 / 优先格 9）---"
  echo "（各轮开跑臂数：'训练 N 臂' 行）"
  grep -nE "训练 [0-9]+ 臂|本轮共 [0-9]+ 臂" $W/adaptive.log | tail -10
  echo "（种子扫描臂数分母：'种子扫描 i/20' ⇒ 配比已从 14 提到 20）"
  grep -oE "种子扫描 [0-9]+/[0-9]+" $W/adaptive.log | sort -u | tail -6
  echo "（新增互补格 lrf0.10_mixw1.9 + 近失 lrf0.10 的 official 回填证据）"
  grep -nE "lrf0\.10_mixw1\.9|official 回填" $W/adaptive.log | tail -8
  echo "（守护日志里应有『代码已更新…在轮次边界生效』一行）"
  grep -n "代码已更新" $W/watchdog.log | tail -5
  echo
  echo "--- ④ adaptive.log 尾部 60 行 ---"
  tail -60 $W/adaptive.log
  echo
  echo "--- ⑤ watchdog.log 尾部 15 行 ---"
  tail -15 $W/watchdog.log
  echo
  echo "--- ⑥ 夜汇总（四块：总量/六项全过/最接近候选/配方组统计） ---"
  $PY manu/pipeline/npu/night_summary.py 2>&1
  echo
  echo "--- ⑥b 漏斗 / 近失前沿（analyze_near_miss）---"
  $PY manu/pipeline/npu/analyze_near_miss.py --ledger $W/nano_search_ledger.json --top 20 2>&1 | head -60
  echo
  echo "--- ⑦ 升轨候选清单 ---"
  $PY manu/pipeline/npu/promote_candidate.py --list 2>&1 | head -60
  echo
  echo "--- ⑧b 板卡同板 A/B 测速口径（板卡已换成旧板卡 64 = 192.168.0.64）---"
  cat <<'EOF'
① 确认板卡空闲（别抢别人的任务）：
   ssh -p 22 root@192.168.0.64 "cat /proc/loadavg; ps -eo cmd | grep -E 'rknn|npu' | grep -v grep"
② 恢复板卡 key：mv /home/huangzhe/.ssh/id_board.disabled_20261112_nano_search /home/huangzhe/.ssh/id_board
③ 改流水线 BOARD 常量为 192.168.0.64（nano_search.py 受守护 md5 监控 ⇒ 09:00 之后再改，别在跑轮次里改）
④ 判据：**候选 vs trial_0031 同板同会话 A/B**，候选_ms ≤ 0031_ms × 1.02，重复 3 次取中位数
   ⛔ 不与 23.528 ms（11-11 新板读数）比较 —— 板卡已更换，跨板不可比
EOF
  echo
  echo "--- ⑧ 板卡状态与恢复指令 ---"
  ls -la /home/huangzhe/.ssh/ | grep -i board
  cat $W/BOARD_PHASE_DISABLED.txt 2>/dev/null
  echo
  echo "--- ⑨ 整夜资源曲线（avail/swap/GPU util/在训臂数，每 5 min 一行）---"
  tail -120 $W/mem_samples.log 2>/dev/null
  echo
  echo "--- ⑩ 本轮改动清单（供回溯）---"
  md5sum manu/pipeline/npu/adaptive_runner.py manu/pipeline/npu/nano_search.py 2>/dev/null
  grep -n "TRAIN_WORKERS\|NANO_BENCH_WORKERS" manu/pipeline/npu/nano_search.py | head -6
} > "$OUT" 2>&1

echo "[$(date "+%F %T")] 明早报告已写入 $OUT"
