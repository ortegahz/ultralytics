#!/bin/bash
# 明早自动报告 v3（2026-11-17 17:5x）：加「权威六项审计」+ 定向门禁通道尾巴
#   改动：① 时间提前到 08:40（用户 09:00 到）② 加 six_criteria_audit（以磁盘 gate_report.json 为准）
#         ③ 加 gate_priority（岭线 w56 定向门禁）日志 ④ 门槛票数/双关进度
set -u
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
W=/mnt/data/siping/runs/npu20ms/nano_search
AT="${1:-2026-11-18 08:40}"
STAMP=$(date -d "$AT" +%Y%m%d)
OUT=$W/morning_report_${STAMP}.txt
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

while [ "$(date +%s)" -lt "$(date -d "$AT" +%s)" ]; do sleep 60; done
sleep 30

{
  echo "===== 权威六项审计（磁盘 gate_report.json 为准）====="
  $PY manu/pipeline/npu/six_criteria_audit.py --top 15
  echo
  echo "===== 🧭 周一升轨命令（promote_ready_cmd · 只读 · 可直接复制粘贴）====="
  # 🟢 2026-11-20 15:40 新增：周末新轴产出的是**新成员/单臂** ⇒ `auto_promote_watch` 的
  #   「同族 n≥4 + 种子复现」关天然过不了 ⇒ 必须有人工兜底。
  #   本节把「哪个臂 / 六项读数 / 差哪一关 / **复制粘贴跑哪条命令**」直接印出来。
  #   `timeout 420` ⇒ 绝不吃掉晨报 08:55→09:05 的 10 分钟窗口（超时只影响本节）。
  timeout 420 $PY manu/pipeline/npu/promote_ready_cmd.py --top 8 \
    || echo "（🟡 升轨命令生成器超时/异常：本节跳过，请手工跑 promote_ready_cmd.py）"
  echo
  echo "===== 通用晨报 ====="
  $PY manu/pipeline/npu/morning_report_nano.py
  echo
  echo "===== 升轨记录 promotion_ready.json ====="
  cat "$W/../promotion_ready.json" 2>/dev/null || cat /mnt/data/siping/runs/npu20ms/promotion_ready.json 2>/dev/null || echo "（未升轨）"
  echo
  echo "===== 定向门禁通道（岭线 w56）====="
  tail -n 25 "$W/gate_priority_run.log" 2>/dev/null
  tail -n 10 "$W/gate_priority.log" 2>/dev/null
  echo
  echo "===== 门禁优先循环尾部 ====="
  tail -n 30 "$W/gate_watch_run.log" 2>/dev/null
  echo
  echo "===== 种子复现波尾部 ====="
  tail -n 40 "$W/seed_replicate_run.log" 2>/dev/null
  echo
  echo "===== 🟢 白天波 D1：三路汤（sb3）合成/门禁尾部 ====="
  for f in soup3_synth.log soup3_round2.log day_gateB.log gate_loop_d1b.log retag_sb3.log; do
    echo "--- $f"; tail -n 16 "$W/$f" 2>/dev/null
  done
  echo
  echo "===== 🟢 白天波 D1：1100277 高权重段（找 T2_fp=1）尾部 ====="
  for f in soup_new_1420000.log soup_new_1460000.log day_gateA.log gate_loop_d1.log; do
    echo "--- $f"; tail -n 14 "$W/$f" 2>/dev/null
  done
  echo
  echo "===== 岭线精扫 / 同盆地三兄弟 / 复现汤 尾部 ====="
  for f in soup_new_1220000.log soup_new_1230000.log soup_new_1240000.log soup_new_1250000.log soup_new_1310000.log; do
    echo "--- $f"; tail -n 14 "$W/$f" 2>/dev/null
  done
  echo
  echo "===== 板卡 T4 守望尾部 ====="
  tail -n 20 "$W/board_t4_watch.log" 2>/dev/null
  echo
  echo "===== 升轨守望尾部 ====="
  tail -n 20 "$W/auto_promote_watch.log" 2>/dev/null
  echo
  echo "===== 🧵 Track-B：交付红线内（T2_fp≤5）但非六项全过的更强候选 ====="
  $PY $W/scripts/trackb_report.py
  echo
  echo "===== 🏆 六项全过猎手（任何 6/6 臂，含家族不足 4 的孤例冠军）====="
  tail -n 40 "$W/six_hunter.log" 2>/dev/null
  echo
  echo "----- 附加候选表 -----"
  cat "$W/morning_report_extra.md" 2>/dev/null
  echo
  echo "===== 🌙 今夜新波次合成尾部（h3/h4/h5/jit/m26/m26b/m27/d5c/frontier）====="
  for f in soup3_h3.log soup3_h4.log soup3_h5.log soup3_jit.log soup3_m26.log soup3_m26b.log soup3_m27.log soup3_d5c.log soup3_d4_frontier.log; do
    echo "--- $f"; tail -n 8 "$W/$f" 2>/dev/null
  done
  echo
  echo "===== 🆕 今晚主波（h32~h46 · D3/D4/D5/D6 四卡）合成与门禁尾部 ====="
  for f in soup3_h32.log soup3_h33.log soup3_h34.log soup3_h35.log soup3_h36.log soup3_h37.log soup3_h38.log soup3_h39.log soup3_h40.log soup3_h41.log soup3_h42.log soup3_h43.log soup3_h45.log soup3_h46.log d6_gpu0.log d5_gpu1.log d4_gpu2.log d3_gpu3.log; do
    echo "--- $f"; tail -n 6 "$W/$f" 2>/dev/null
  done
  echo
  echo "===== 🛠 夜间自愈 / 运维水位（守望 + 看门狗 + 板卡）====="
  for f in day_supervisor.log nano_watchdog.log board_t4_watch.log trackb_promote_watch.log refresh_prio.log; do
    echo "--- $f"; tail -n 10 "$W/$f" 2>/dev/null
  done
  echo "--- 守望进程"; ps -eo pid,etime,args | grep -E "day_supervisor_d1|synth_gpu[0-9]|board_t4_watch|trackb_promote|auto_promote|nano_watchdog" | grep -v grep | cut -c1-120
  echo "--- 当前 GPU"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
  echo "--- 内存/负载"; free -m | head -2; uptime
  echo
  echo "===== 内存看门狗尾部（nano_watchdog）====="
  tail -n 10 "$W/nano_watchdog.log" 2>/dev/null
  echo
  echo "===== 资源水位（watchdog 最后 5 条）====="
  tail -n 5 /tmp/load_watchdog.log 2>/dev/null
} > "$OUT" 2>&1

cp -f "$OUT" "$W/morning_report_latest.txt"
echo "报告已生成：$OUT"
