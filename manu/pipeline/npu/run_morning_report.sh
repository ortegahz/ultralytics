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
  echo "--- ⓪ 结论速览：**六项全过 = 可升轨 Baseline_v2.1**（判据 = Baseline_v2.0/trial_9863 的六项）---"
  $PY manu/pipeline/npu/six_criteria_audit.py --top 8 2>&1 | head -26
  echo
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
  echo "--- ③b 今夜算力配比是否按 §150.7.4 + §152.5 生效（种子格 24 / 突变 ≤5 / 优先格 9）---"
  echo "（各轮开跑臂数：'训练 N 臂' 行）"
  grep -nE "训练 [0-9]+ 臂|本轮共 [0-9]+ 臂" $W/adaptive.log | tail -10
  echo "（种子扫描臂数分母：'种子扫描 i/24' ⇒ 配比 14→20→24）"
  grep -oE "种子扫描 [0-9]+/[0-9]+" $W/adaptive.log | sort -u | tail -6
  echo "（新增互补格 lrf0.10_mixw1.9 + 近失 lrf0.10 的 official 回填证据）"
  grep -nE "lrf0\.10_mixw1\.9|official 回填" $W/adaptive.log | tail -8
  echo "（§152.5 新增格：sw4.0 与 sw4.0_thr12ms3 是否真的开跑了）"
  grep -nE "优先格 sw4\.0" $W/adaptive.log | tail -8
  grep -cE "优先格 sw4\.0" $W/adaptive.log | sed "s/^/    sw4.0 优先格臂数 = /"
  echo "（守护日志里应有『代码已更新…在轮次边界生效』一行）"
  grep -n "代码已更新" $W/watchdog.log | tail -5
  echo
  echo "--- ③c 🟢 §153.16~153.18 新杠杆是否真的开跑（基线派生格 / LAWA 加权 SWA / 新种子分配）---"
  echo "（① 基线派生绝对格：全部 BASE_* 格（含 §153.29 新增 thr12ms3×lrf0.1 家族））"
  grep -oE "基线派生格 BASE_[A-Za-z0-9_.]+" $W/adaptive.log | sort | uniq -c | sort -rn | head -20
  echo "（①b 各格实际臂数汇总：BASE_A2 / BASE_A1 / BASE_Ai / BASE_B2 …）"
  grep -nE "基线派生格 BASE_(A2|A1|Ai|A2i|A1i|A1hi|sw5.0_lrf0.1)" $W/adaptive.log | tail -18
  echo "（② LAWA 加权 SWA：窗口名带 _w / post_swa_of_*_lawa）"
  grep -nE "swa_(late|neigh|wide)_w|post_swa_of_[0-9]+_lawa" $W/adaptive.log | tail -12
  echo "（③ 种子扫描新分配：B1(9863 本家)/B2(12053 本家) 应拿到多数臂）"
  grep -oE "种子扫描 [0-9]+/[0-9]+（[^|｜]*｜[^）]*)" $W/adaptive.log | tail -12
  echo "（④ 各格臂数汇总（台账实测）：按 sw×thr×lrf 统计本夜各格训练臂数）"
  $PY - <<'PYEOF' 2>&1 | tail -20
import json, collections
led = json.load(open("/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"))
c = collections.Counter()
for t, st in (led.get("arms") or {}).items():
    p = st.get("params") or {}
    if st.get("synth") or p.get("init") != "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt":
        continue
    c[(p.get("sw"), p.get("thr"), p.get("maxside"), p.get("lr_floor"))] += 1
for k, v in c.most_common(14):
    print("    sw=%-5s thr=%-5s ms=%-5s lrf=%-5s  臂数=%d" % (k[0], k[1], k[2], k[3], v))
PYEOF
  echo
  echo "--- ③c2 🟢🟢 §153.29 算力错配修复是否生效：E 排序 + thr12ms3×lrf0.1 家族配比 ---"
  echo "（① 门禁窗口的新措辞：'期望产出率前 8' ⇒ 旧措辞是 'nano 前 8'）"
  grep -nE "门禁窗口" $W/adaptive.log | tail -8
  echo "（② 本夜各格训练臂数（sw×thr×ms×lrf）⇒ A2/A1 家族应占大头）"
  $PY - <<'PYEOF3' 2>&1 | tail -16
import json, collections
led = json.load(open("/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"))
c = collections.Counter(); o = collections.Counter()
for t, st in (led.get("arms") or {}).items():
    p = st.get("params") or {}
    if st.get("synth") or p.get("init") != "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt":
        continue
    if int(t) < 12898:
        continue
    k = (p.get("sw"), p.get("thr"), p.get("maxside"), p.get("lr_floor"))
    c[k] += 1
    of = st.get("official") or {}
    if of and "skipped" not in of and "stale" not in of:
        o[k] += 1
print("    本夜臂数 / 有官方读数的臂数：")
for k, v in c.most_common(14):
    print("      sw=%-5s thr=%-5s ms=%-5s lrf=%-5s  臂数=%-3d 官方读数=%d" % (k[0], k[1], k[2], k[3], v, o.get(k, 0)))
PYEOF3
  echo "（③b 无偏口径：全臂 p_triad（格级 sw×thr×ms×lrf）⇒ 这才是训练臂分配的真实依据）"
  $PY - <<'PYEOF5' 2>&1 | tail -14
import sys
sys.path.insert(0, "manu/pipeline/npu")
import adaptive_runner as ar, nano_search as ns
led = ns.load_ledger(); base = ar.base_ref(led)
cs = ar.cell_stats(led, base)
rows = sorted(cs.items(), key=lambda kv: -(kv[1]["triad"] / max(1, kv[1]["n"])))
for k, d in rows[:12]:
    print("    p_triad=%.3f  n=%-4d triad=%-3d T3|triad=%d/%d   sw=%s thr=%s ms=%s lrf=%s" % (
        d["triad"] / max(1, d["n"]), d["n"], d["triad"], d["t3"], d["ntri"], k[0], k[1], k[2], k[3]))
PYEOF5
  echo "（③ E 期望产出率表实跑（A2/A1 应居首，B1 应远低于它们））"
  $PY - <<'PYEOF4' 2>&1 | tail -12
import sys
sys.path.insert(0, "manu/pipeline/npu")
import adaptive_runner as ar, nano_search as ns
led = ns.load_ledger(); base = ar.base_ref(led)
y = ar.cell_yield_table(led, base)
for k, v in sorted(y.items(), key=lambda kv: -kv[1])[:10]:
    print("    E=%.4f  sw=%s thr=%s ms=%s lrf=%s" % (v, k[0], k[1], k[2], k[3]))
PYEOF4
  echo
  echo "--- ③d 🔴 §153.20 末轮训练臂硬上限（arm_budget）是否生效 ---"
  echo "（末轮应出现：末轮训练臂硬上限 N ⇒ [arm_budget] 上限 N 臂 ⇒ 保留 M 臂）"
  grep -nE "末轮训练臂硬上限|\[arm_budget\]" $W/adaptive.log | tail -12
  echo "（各轮实际训练臂数：'训练 N 臂' 行，末轮应是缩过的小数）"
  grep -nE "训练 [0-9]+ 臂" $W/adaptive.log | tail -8
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
  echo "--- ⑥c 🔴 六项判据**全盘复算**审计（磁盘 gate_report 为权威，含非 picked epoch）---"
  $PY manu/pipeline/npu/six_criteria_audit.py --top 25 2>&1 | head -70
  echo
  echo "--- ⑥d 🟢 §153.25 逐格联合产出率（哪一格最该投算力）---"
  $PY manu/pipeline/npu/cell_yield_audit.py --top 12 2>&1 | head -40
  echo
  echo "--- ⑥e 🟢 §153.25 种子噪声定标（σ 与 T3 过线率标定）---"
  $PY manu/pipeline/npu/seed_noise_audit.py --top 8 2>&1 | head -34
  echo
  echo "--- ⑥f 🟢 §153.25 重训浪费核查（(配方,seed) 重复对数；R14 起应为 0）---"
  $PY - <<'PYEOF2' 2>&1
import json, collections
led = json.load(open("/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"))
arms = led.get("arms") or led
def rk(p):
    return "|".join(str(p.get(k)) for k in ("sw","swl","thr","maxside","lr_floor","mix","sched"))
by = collections.defaultdict(list)
for t, st in arms.items():
    if st.get("synth") or (st.get("params") or {}).get("init") != "/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt":
        continue
    by[(rk(st.get("params") or {}), st.get("seed"))].append(int(t))
dups = {k: v for k, v in by.items() if len(v) > 1}
print("    真训练臂 (配方,seed) 唯一组合 = %d，重复对 = %d，浪费臂 = %d" % (
    len(by), len(dups), sum(len(v) - 1 for v in dups.values())))
recent = {k: v for k, v in by.items() if max(v) >= 12900 and len(v) > 1}
print("    R14 之后（trial≥12900）的重复对 = %d" % len(recent))
PYEOF2
  echo
  echo "--- ⑥b 漏斗 / 近失前沿（analyze_near_miss）---"
  $PY manu/pipeline/npu/analyze_near_miss.py --ledger $W/nano_search_ledger.json --top 20 2>&1 | head -60
  echo
  echo "--- ⑦ 升轨候选清单 ---"
  $PY manu/pipeline/npu/promote_candidate.py --list 2>&1 | head -60
  echo
  echo "--- ⑧a 板卡同会话实测（旧板 192.168.0.64 / evm3588，2026-11-12 16:40）---"
  grep -aE "@@SUMMARY@@" /mnt/data/siping/runs/npu20ms/board_measure_20261112_old64/hm_speed_results.txt 2>/dev/null || echo "（无板卡会话文件）"
  echo "（对照：新板 192.168.0.211 同会话 = t31 23.528 / n9863 22.009 ms）"
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
