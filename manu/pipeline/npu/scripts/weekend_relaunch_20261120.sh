#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# 🟢 周末重启（2026-11-20）—— 让三件事在**无人值守**下生效
#
#   ① **新截止时刻**：波次 → 2026-11-30 09:00；门禁 → 08:55；守望子进程 → 09:20；
#      顶层守望 → 09:30。（原为 07:00 / 07:30 / 08:00 / 08:30 / 08:45，**全部早于用户要的 09:00**）
#   ② **新搜索轴立刻投产**：`gen_w13_recipes.py` 的队首三条新轴
#      （ax_t1donor / ax_t1donor6 / ax_fam3t3）在本波循环体**第一位**
#      ⇒ 重启波次壳后**第一轮就跑新轴**（原本要等 w12→…→d5 全跑完，约 +17 h）。
#   ③ **板卡同会话参照修复生效**：`board_t4_watch.py` 现在每次会话都把**当前基线臂**放上板
#      ⇒ `ref_ms` 不再是 None ⇒「同会话相对基线」口径恢复
#      ⇒ 板卡热漂移 ±0.5 ms 下，基线同图臂不再被误判 FAIL ⇒ **升轨链不再断链**。
#
# ⚠️ 本脚本**只重启**，不启动任何训练/推理任务（铁律一：训练/推理由用户手工执行）。
# ⚠️ 幂等：可重复执行。
# 用法：
#     bash weekend_relaunch_20261120.sh --dry-run    # 只打印将要做什么
#     bash weekend_relaunch_20261120.sh              # 真执行
# ══════════════════════════════════════════════════════════════════════════
set -u
W=/mnt/data/siping/runs/npu20ms/nano_search
REPO=/tmp/pycharm_project_10ae9e2e
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

say(){ echo "[$(date "+%F %T")] $*"; }
run(){ if [ "$DRY" = "1" ]; then say "DRY-RUN: $*"; else eval "$@"; fi; }

say "=== 周末重启开始（DRY=$DRY）==="

# ── ① 先停顶层守望 ──────────────────────────────────────────────────────
#    顺序关键：旧守望进程内存里是**旧截止字面量**（08:30/08:45），
#    先停它 ⇒ 后面拉起的一切子进程都用**新脚本**（09:20/09:30）。
say "① 停顶层守望 day_supervisor_d1.sh"
run "pkill -f 'bash $W/scripts/day_supervisor_d1.sh'"
sleep 2

# ── ② 停 4 条波次壳 + tipdose + 3 条门禁壳 ───────────────────────────
#    波壳的 trap 会清掉自己的 PID 文件 ⇒ 新守望会按新脚本重拉。
say "② 停波次壳 / tipdose / 门禁壳"
for k in .pid_synth_gpu0 .pid_synth_gpu1 .pid_synth2d4 .pid_synth_gpu3 .pid_tipdose \
         .pid_gateA .pid_gateB .pid_gateC; do
  p=$(cat "$W/$k" 2>/dev/null)
  if [ -n "$p" ]; then run "kill -TERM $p"; say "   stop $k pid=$p"; else say "   skip $k（无 PID 文件）"; fi
done

# ── ③ 停 3 个守望进程（否则仍是旧代码，板卡参照修复不生效）─────────────
say "③ 停守望进程（board_t4_watch / auto_promote_watch / baseline_guard）"
run "pkill -f 'manu/pipeline/npu/board_t4_watch.py'"
run "pkill -f 'manu/pipeline/npu/auto_promote_watch.py'"
run "pkill -f '/tmp/baseline_guard.py'"
# 🔴🔴 2026-11-20 16:25 **补停两个漏掉的旧实例**（子 Agent 独立审计发现）
#   ① `trackb_promote_watch.py` —— 旧实例带 **--deadline 08:30**，
#      而新脚本要的是 09:20 ⇒ 不停它 ⇒ 守望的「按 PID 文件判活」会认为它活着
#      ⇒ **永远不会被新守望用新截止重启** ⇒ 周一 08:30 后升轨链静默停摆。
#   ② `morning_report_sched.sh` —— 旧实例内存里是 **AT 08:40 / END 08:45**，
#      而新脚本是 AT 08:55 / END 09:05 ⇒ 不停它 ⇒ 周一**会多出一份 08:40 的早报**，
#      且那份早报统计的是 08:40 的台账（比正式晨报早 15 min）。
#   ⚠️ `gen_watchdog.sh` / `nano_watchdog.sh` **故意不停**：它们是守门狗，重启无益。
say "③b 补停 trackb_promote_watch / morning_report_sched（旧截止字面量残留）"
run "pkill -f 'manu/pipeline/npu/trackb_promote_watch.py'"
_mr=$(cat "$W/.pid_mreport_sched" 2>/dev/null)
if [ -n "$_mr" ]; then run "kill -TERM $_mr"; say "   stop .pid_mreport_sched pid=$_mr"; fi
run "pkill -f 'morning_report_sched.sh'"

# ── ④ 收割在跑的 night_soup3（波壳被杀后子进程会继续烧卡）────────────────
#   ⚠️ 坑（2026-11-20 14:5x 实测）：`timeout` 包装器**自建独立进程组**
#      （pgid=timeout_pid ≠ 波壳 pid；实测 1802103/1802104 的 pgid=1802103，
#      波壳 3815016 的 pgid=3815016）⇒ `kill -PGID` 打不到 python 真身。
#      且原写法只按 `ppid==1` 收割 ⇒ 在「父壳刚死、子进程刚被 reparent」的
#      窗口里**一个都收割不到** ⇒ python 真身继续占着 GPU，
#      新波壳再起一条 ⇒ 同卡双跑（+5.3GB 缓存构建 ⇒ 内存打穿）。
#   ⇒ 改法：**按 cmdline 精确匹配**（python 真身 + timeout 包装器都算）
#      先 TERM 再 KILL，最后按 ppid==1 再兜一遍。
say "④ 收割在跑的 night_soup3"
soup_pids(){ ps -eo pid,args | grep "night_soup3[.]py" | grep -v grep | awk '{print $1}'; }
for p in $(soup_pids); do run "kill -TERM $p 2>/dev/null"; done
sleep 3
for p in $(soup_pids); do run "kill -9 $p 2>/dev/null"; say "   killed soup3 pid=$p"; done
sleep 1
for p in $(ps -eo pid,ppid,args | grep "night_soup3[.]py" | grep -v grep | awk '$2==1{print $1}'); do
  run "kill -9 $p 2>/dev/null"; say "   reaped orphan soup3 pid=$p"
done
say "④ 收割后残留 night_soup3 进程数 = $(soup_pids | wc -l)（应为 0）"

# ── ⑤ 拉起新守望（flock 互斥；子进程截止 09:20，自己活到 09:30）────────
say "⑤ 拉起新守望"
if [ "$DRY" = "1" ]; then
  say "DRY-RUN: cd $REPO && setsid nohup bash $W/scripts/day_supervisor_d1.sh >> $W/day_supervisor.log 2>&1 < /dev/null 9>&- &"
else
  cd "$REPO" || exit 1
  setsid nohup bash "$W/scripts/day_supervisor_d1.sh" >> "$W/day_supervisor.log" 2>&1 < /dev/null 9>&- &
fi
sleep 20

# ── ⑥ 自检 ─────────────────────────────────────────────────────────────
say "⑥ 自检：进程清单"
ps -eo pid,etime,args | grep -E "day_supervisor_d1|synth_gpu[0-9]_|gate_loop_d1|tipdose_loop|board_t4_watch|auto_promote_watch|baseline_guard|morning_report_sched|nano_watchdog" \
  | grep -v grep | sed 's/^/   /'
say "⑥ 自检：各脚本 END 截止字面量（波次 08:30 / 门禁 08:55 / 顶层 09:30）"
for s in synth_gpu0_d6 synth_gpu1_d5 synth_gpu2_d4 synth_gpu3_d3 tipdose_loop \
         gate_loop_d1 gate_loop_d1b gate_loop_d1c day_supervisor_d1; do
  say "   $s.sh  $(grep -o 'END="[^"]*"' "$W/scripts/$s.sh" 2>/dev/null | head -1)"
done
say "⑥ 自检：运行中进程实际带的 --deadline（守望子进程应为 09:20 / soup3 应为 08:30）"
ps -eo args | grep -o "\-\-deadline 2026-11-23 [0-9:]*" | sort | uniq -c | sed 's/^/   /'
say "⑥ 自检：旧截止字面量应已消失（重启前会看到 07:30/08:00/08:30；重启后应只剩 08:30/08:55/09:20/09:30）"
say "⑥ 自检：新轴是否已进入在跑波次的命令行（找 1603002/1610001/1609001 供体）"
for p in $(pgrep -f "night_soup3[.]py" 2>/dev/null); do
  n=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | grep -o "1603002\|1610001\|1609001\|1603004" | sort -u | tr '\n' ' ')
  say "   soup3 pid=$p  新轴供体=${n:-（本轮未含新轴，正常：下一轮才有）}"
done
# 🆕 2026-11-20 16:15 自检追加：**T1 供体微剂量轴**（§270）是否已进入在跑波次
say "⑥b 自检：T1 供体微剂量轴（§270 · gen_t1donor_recipes.py）是否已上线"
for p in $(pgrep -f "night_soup3[.]py" 2>/dev/null); do
  n=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | grep -o "14963\|12543\|12703\|15510\|1610001\|1104642" | sort -u | tr '\n' ' ')
  say "   soup3 pid=$p  T1供体=${n:-（本轮未含新轴，正常：下一轮才有）}"
done
say "   （生成器本体：$W/scripts/gen_t1donor_recipes.py md5=$(md5sum $W/scripts/gen_t1donor_recipes.py 2>/dev/null | cut -d' ' -f1)）"
say "   （期望 md5 = 89ec60c976e56cdddbf0215a5b265d20 —— 2026-11-20 16:30 加了实测汤增益继承）"
say "⑥b2 自检：T1 骨架补偿轴（轴⑭ · gen_t1fix_recipes.py）是否已上线"
for p in $(pgrep -f "night_soup3[.]py" 2>/dev/null); do
  n=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | grep -o "2000321\|2010351\|1920134\|1920144" | sort -u | tr '\n' ' ')
  say "   soup3 pid=$p  T1骨架=${n:-（本轮未含新轴，正常：下一轮才有）}"
done
say "   （生成器本体：$W/scripts/gen_t1fix_recipes.py md5=$(md5sum $W/scripts/gen_t1fix_recipes.py 2>/dev/null | cut -d' ' -f1)）"
say "   （期望 md5 = 17f1a181559d1c894b14a7996f6bdba9）"
say "⑥c 自检：4 条波次壳是否含新轴调用"
for s in synth_gpu0_d6 synth_gpu1_d5 synth_gpu2_d4 synth_gpu3_d3; do
  say "   $s.sh  t1donor=$(grep -c 'gen_t1donor_recipes' $W/scripts/$s.sh 2>/dev/null) 处｜t1fix=$(grep -c 'gen_t1fix_recipes' $W/scripts/$s.sh 2>/dev/null) 处"
done
say "=== 完成：截止时刻 = 波次 08:30 / 门禁 08:55 / 守望子进程 09:20 / 顶层守望 09:30 ==="
say ""
say "ℹ️ 本脚本**不含**新成员生产线（汤空间已证伪饱和 ⇒ 冠军只能来自新训练成员）。"
say "   要同时起训练轴，请另跑（它会启动训练，故与本脚本分开）："
say "     bash $W/scripts/weekend_train_axis_20261120.sh --dry-run   # 先看"
say "     bash $W/scripts/weekend_train_axis_20261120.sh             # 真起"
