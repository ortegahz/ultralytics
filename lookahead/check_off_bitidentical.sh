#!/bin/bash
# 「关开关必须逐位一致」复核：新旋钮（--replay-abstain-*）关闭时，replay 臂的逐帧产物
# 必须与**上一档冠军**（弃权功能存在之前跑的）逐字节相同。
#
#   bash manu/tracking/lookahead/check_off_bitidentical.sh
#
# 对照口径（注意：**不是** final8/final24 —— 那是「新旋钮打开」的产物，本来就该不同）：
#   8 序：noguard8/preds_delivery   vs uniqrematch2/preds_u060
#   24 序：noguard24/preds_delivery vs uniq24b/preds_u060
set -u
cd "$(dirname "$0")/../../.."        # → 项目根
F=runs/lookahead/final
rc=0
for pair in "noguard8:uniqrematch2/preds_u060:8" "noguard24:uniq24b/preds_u060:24"; do
  IFS=: read -r a b tag <<< "$pair"
  same=0; diff=0; miss=0
  for f in "$F/$a/preds_delivery"/*.json; do
    [ -e "$f" ] || continue
    n=$(basename "$f"); g="$F/$b/$n"
    if [ ! -f "$g" ]; then miss=$((miss+1)); continue; fi
    if [ "$(md5sum "$f" | cut -d' ' -f1)" = "$(md5sum "$g" | cut -d' ' -f1)" ]; then
      same=$((same+1))
    else
      echo "DIFFER  $tag  $n"; diff=$((diff+1))
    fi
  done
  echo "[$tag seq] IDENTICAL=$same DIFFER=$diff MISSING=$miss"
  [ "$diff" -eq 0 ] && [ "$miss" -eq 0 ] || rc=1
done
[ "$rc" -eq 0 ] && echo "OFF_BITIDENTICAL_OK" || echo "OFF_BITIDENTICAL_FAIL"
exit "$rc"
