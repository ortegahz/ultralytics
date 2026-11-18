#!/bin/bash
# 🟢 白天波 D2 · GPU2（2026-11-18）：2 路汤「T2_fp=1」高权重密网格（bs640001 家族）
#   物理假设（11-18 晨读数）：
#     · T2_fp 随「父主导权重」升高而下降：1100277@2 族 w≤.65⇒4 / .70⇒3 / .75⇒2 / .725⇒2
#       ⇒ 推 w≥.80 ⇒ T2_fp 可能落到 **1**（升轨唯一硬墙）。
#     · bench（nano/nano4/F1）随 1100277 权重升高而**上升**（nano .7205→.7235，nano4 .6562→.6607）
#       ⇒ 与 T2_fp 同向 ⇒ 高权重段是最优方向。
#   家族口径：soup_family = `bs640001:w<round(w*100)>` ⇒ **1% 一个桶** ⇒ 每桶必须 ≥4 个点
#       才能满足升轨「同族 n≥4 六项全过」⇒ 本脚本按**每桶 4~5 点、0.25% 步长**铺网格。
#   铁律：flock 互斥（防守望重复拉起 ⇒ 两个进程写同一个 sid ⇒ 权重写坏）；起跑前查 MemAvailable。
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="2026-11-19 07:30"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
exec 9>"$W/.lock_d2_gpu2"
flock -n 9 || { echo "[$(date "+%F %T")] [D2-GPU2] 已有实例在跑 ⇒ 退出" >> "$W/d2_gpu2.log"; exit 0; }
log(){ echo "[$(date "+%F %T")] $*" >> $W/d2_gpu2.log; }
# 🟢🟢 2026-11-18 10:10 **PID 文件判活**（替代守望里的 `pgrep -f`）
#   坑：`pgrep -f "<脚本名>"` 会**匹配到任何提到该名字的 shell**（例如运维 ssh 命令本身）
#        ⇒ 守望误判「已在跑」⇒ 通道永远不重启。⇒ 改为脚本自己登记 PID + kill -0 验活。
PIDF=$W/.pid_synth2
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT

run(){  # run <sid_base> <weights> <logfile>
  local sb=$1 ws=$2 lf=$3
  timeout 7200 $PY -u manu/pipeline/npu/night_soup_batch.py --gpu 2 --parents 640001:5 --children-arms "1100277:2" \
    --weights "$ws" --sid-base "$sb" --sid-span 200 --max-gates 0 --gate-family-quota 8 \
    --mem-gate-gb 40 --deadline "$END" >> "$lf" 2>&1
}
run3sb(){  # 三路汤（sb3）合成：run3sb <sid_base> <recipes> <logfile>
  timeout 7200 $PY -u manu/pipeline/npu/night_soup3.py --gpu 2 --sid-base "$1" --sid-span 200 \
    --max-gates 0 --mem-gate-gb 40 --deadline "$END" --recipes "$2" >> "$3" 2>&1
}
while :; do
  now=$(date +%s); end=$(date -d "$END" +%s)
  [ "$now" -lt "$end" ] || { log "[截止] $END ⇒ 退出"; break; }
  avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  if [ "${avail:-0}" -lt 40 ]; then log "[闸门] avail=${avail}GB < 40GB ⇒ 等 120s"; sleep 120; continue; fi
  log "[轮] D2-GPU2 段开始 avail=${avail}GB"
  # 段1：w84 / w86 桶（.835~.865）—— 补 1460xxx 网格的 w84 桶（原来只有 3 点）
  run 1500000 "0.8350,0.8375,0.8400,0.8425,0.8450,0.8550,0.8575,0.8600,0.8625" $W/soup_new_1500000.log
  # 段2：w88 桶补点 + w90 桶（.885~.905）
  run 1501000 "0.8850,0.8950,0.9000,0.9025,0.9050" $W/soup_new_1501000.log
  # 段3：w92 / w94 桶（.915~.9425）—— 父主导更极端段
  run 1502000 "0.9150,0.9175,0.9200,0.9225,0.9350,0.9375,0.9400,0.9425" $W/soup_new_1502000.log
  # 段4：w76 桶（.755~.7625）—— 万一 T2_fp=1 的边界比预期更靠下
  run 1503000 "0.7550,0.7575,0.7600,0.7625" $W/soup_new_1503000.log
  # 🟢🟢 段5（09:40 新增，依据门禁读数 1420005@w.88 ⇒ **T2fp=1 但 T3=.948374 掉出判据线**）
  #   ⇒ 单腿加重 1100277 会**同时**压 T2_fp 和压 T3 ⇒ 必须在「重增益子 + 小剂量 T3 锚」处取平衡。
  #   ⇒ 三路汤 (640001@5, 1100296@1, 1100277@2) 取 1100296∈{.05,.10,.15,.20}、1100277∈{.45,.50}
  #      ⇒ bw45/bw50 各 4 个拆分 ⇒ 每族 ≥4 票；T3 靠 1100296 锚保住，T2_fp 靠大剂量 1100277 压下来。
  run3sb 1530000 "640001:5:0.45,1100296:1:0.10,1100277:2:0.45;640001:5:0.40,1100296:1:0.15,1100277:2:0.45;640001:5:0.50,1100296:1:0.05,1100277:2:0.45;640001:5:0.35,1100296:1:0.20,1100277:2:0.45;640001:5:0.40,1100296:1:0.10,1100277:2:0.50;640001:5:0.45,1100296:1:0.05,1100277:2:0.50;640001:5:0.35,1100296:1:0.15,1100277:2:0.50;640001:5:0.30,1100296:1:0.20,1100277:2:0.50" $W/soup3_d2_bw45_50.log
  # 段6：2 路 w.82~.885 密扫（0.25% 步长）—— 精确定位「T2fp 2→1」与「T3 掉线」的交叉点
  run 1531000 "0.8200,0.8225,0.8250,0.8275,0.8300,0.8325,0.8350,0.8375,0.8400,0.8425,0.8450,0.8475,0.8500,0.8525,0.8550,0.8575,0.8600,0.8625,0.8650,0.8675,0.8700,0.8725,0.8750,0.8775,0.8800,0.8825,0.8850" $W/soup_new_1531000.log
  # 🟢🟢 段7（D3，10:30）：**交叉窗口超密扫** w∈[.7900,.8900] step .000125→.00125 步长 0.125%
  #   ⇒ 每个 1% 桶（w79..w89）拿到 **8 个点** ⇒ 即使 T2fp=1 的边界落在桶中间，
  #      该桶仍有 ≥4 个六项全过臂 ⇒ **升轨「同族 n≥4」可满足**（这是 4 点/桶密网格的关键补丁）。
  W2=$($PY -c "print(','.join('%.4f'%(0.79+0.00125*i) for i in range(81)))")
  run 1534000 "$W2" $W/soup_new_1534000.log
  log "[轮] D2-GPU2 段结束 ⇒ 睡 60s 再巡（幂等：已做组合自动跳过）"
  sleep 60
done

