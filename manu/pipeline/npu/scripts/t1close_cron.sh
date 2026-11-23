#!/bin/bash
# 🟢 t1close 合成通道守望（2026-11-23 新增）—— cron 每 5 分钟判活，死了就按 PID 文件拉起
#   铁律：**按 PID 判活 + 禁 pkill -f**；脚本自带 flock ⇒ 绝不双实例
#   🔴 **门禁优先闸门**：同一张 GPU 上还有 `--only-gate` 门禁通道在跑 ⇒ 本轮**不拉起合成**
#      （门禁是瓶颈机时，合成只是上游供给 ⇒ 门禁优先）
W=/mnt/data/siping/runs/npu20ms/nano_search
END="${T1CLOSE_END:-2026-11-23 21:00}"
now=$(date +%s); end=$(date -d "$END" +%s)
[ "$now" -lt "$end" ] || exit 0
for spec in $T1CLOSE_SPEC; do
  GPU="${spec%%:*}"; PAR="${spec##*:}"
  # 🔴 门禁优先：同卡门禁在跑 ⇒ 让位
  if ps -eo args | grep -q "[n]ight_soup_batch.py --only-gate.*--gpu $GPU "; then
    echo "[$(date "+%F %T")] [CRON] gpu=$GPU 门禁通道在跑 ⇒ 本轮让位" >> $W/t1close_cron.log
    continue
  fi
  PIDF=$W/.pid_t1close_gpu$GPU
  alive=0
  if [ -f "$PIDF" ]; then
    pid=$(cat "$PIDF" 2>/dev/null)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
  fi
  if [ "$alive" -eq 0 ]; then
    setsid nohup env T1CLOSE_GPU="$GPU" T1CLOSE_PARITY="$PAR" T1CLOSE_END="$END" \
      bash $W/scripts/t1close_synth.sh >> "$W/t1close_gpu${GPU}_runner.log" 2>&1 < /dev/null 0<&- 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&- &
    echo "[$(date "+%F %T")] [CRON] t1close gpu=$GPU(parity=$PAR) 已死 ⇒ 已拉起" >> $W/t1close_cron.log
  fi
done

# 2026-11-23 10:55 GPU2 从合成改为第 4 条门禁通道（门禁是 8 倍瓶颈）
# T1CLOSE_GATE_SPEC="<GPU>:<通道>" 把某张卡派给 t1close_gate.sh
# 门禁通道总数仍为 4 条（3 永久 A/B/C + 1 额外 D），不新增第 5 条
# 同卡已有 --only-gate 门禁在跑 ⇒ 本轮让位，避免抢同一张卡
if [ -n "${T1CLOSE_GATE_SPEC:-}" ]; then
for spec in $T1CLOSE_GATE_SPEC; do
  GPU="${spec%%:*}"; CHAN="${spec##*:}"
  if ps -eo args | grep -q "[n]ight_soup_batch.py --only-gate.*--gpu $GPU "; then
    echo "[$(date "+%F %T")] [CRON] gpu=$GPU 已有门禁通道在跑 ⇒ 本轮让位" >> $W/t1close_cron.log
    continue
  fi
  PIDF=$W/.pid_t1close_gate_gpu$GPU
  alive=0
  if [ -f "$PIDF" ]; then
    pid=$(cat "$PIDF" 2>/dev/null)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
  fi
  if [ "$alive" -eq 0 ]; then
    setsid nohup env T1CLOSEGATE_GPU="$GPU" T1CLOSEGATE_CHAN="$CHAN" T1CLOSEGATE_END="$END" \
      bash $W/scripts/t1close_gate.sh >> "$W/t1close_gate_gpu${GPU}_runner.log" 2>&1 < /dev/null 0<&- 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&- &
    echo "[$(date "+%F %T")] [CRON] t1close 门禁 gpu=$GPU(chan=$CHAN) 已死 ⇒ 已拉起" >> $W/t1close_cron.log
  fi
done
fi
