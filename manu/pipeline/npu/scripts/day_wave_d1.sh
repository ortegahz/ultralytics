#!/bin/bash
# 🟢🟢 2026-11-18 白天波 D1 —— **三路汤轴 + 1100277 高权重段扫描**
#   昨夜结论（台账实证）：
#     · 子臂 1100296@1 家族(bs640001:*): bench **钉死**在基线 ⇒ 只能刷 T1（连续量）
#     · 子臂 1100277@2 家族(bs640001:*): bench **更高**(nano .7235 / nano4 .6607 / T1 .8908)
#       且 T3 ∈ [.9487598, .9490200] **全部 ≥ 基线 .9485822**
#       ⇒ **唯一短板 = T2_fp**：w≤.65⇒4, w=.70⇒3, w=.75⇒2 ⇒ **父主导越高越干净**
#       ⇒ 本波直接扫 w ∈ [.70, .88]（.725/.75/.775/.80/.825/.85/.875）找 **T2_fp=1 且 bench 仍 ≥ 基线** 的点
#     · 1100277 的 bench/nano4 增益 + 1100296 的 T2 洁净/T3 台阶 ⇒ **三路汤**同时兼顾
#   通道（4 卡，内存闸门 40GB 起）：
#     GPU0 = 门禁通道 A（bs640001 家族，优先新臂）
#     GPU1 = 门禁通道 B（三路汤 sb3 家族）
#     GPU2 = 合成通道（先三路汤 19 配方，再 1100277 高权重段 7 点）
#     GPU3 = 合成通道（1100277 高权重段 7 点 + 锚族 T1 细扫 57 点）
#   门禁并发 = 2（铁律：单门禁 ≈25GB RSS）
set -u
REPO=/tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search
LOG=$W/day_wave_d1.log
END="${END:-2026-11-19 07:00}"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
log(){ echo "[$(date "+%F %T")] $*" >> $LOG; }

# ---- 🔴 铁律：开跑前读数（内存/负载/IO/GPU/线程）----
avail=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
load=$(awk '{print $1}' /proc/loadavg); ncpu=$(nproc); thr=$(ps -eLf | wc -l)
log "===== 白天波 D1 启动读数：MemAvailable=${avail}GB load=${load}/${ncpu} threads=${thr} ====="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader >> $LOG
if [ "${avail:-0}" -lt 40 ]; then log "[闸门] MemAvailable=${avail}GB < 40GB ⇒ 拒绝启动"; exit 1; fi
if awk "BEGIN{exit !($load > $ncpu * 0.9)}"; then log "[闸门] load=${load} > 0.9*${ncpu} ⇒ 拒绝启动"; exit 1; fi

RECIPES="640001:5:0.40,1100296:1:0.45,1100277:2:0.15;640001:5:0.40,1100296:1:0.40,1100277:2:0.20;640001:5:0.45,1100296:1:0.40,1100277:2:0.15;640001:5:0.35,1100296:1:0.45,1100277:2:0.20;640001:5:0.45,1100296:1:0.45,1100277:2:0.10;640001:5:0.50,1100296:1:0.35,1100277:2:0.15;640001:5:0.30,1100296:1:0.50,1100277:2:0.20;640001:5:0.50,1100296:1:0.40,1100277:2:0.10;640001:5:0.40,1100296:1:0.45,1100275:1:0.15;640001:5:0.45,1100296:1:0.40,1100275:1:0.15;640001:5:0.40,1100296:1:0.40,1100275:1:0.20;640001:5:0.40,1100296:1:0.45,1102047:1:0.15;640001:5:0.45,1100296:1:0.40,1102047:1:0.15;1120019:5:0.55,1100296:1:0.45;1120008:5:0.55,1100296:1:0.45;1120019:5:0.60,1100296:1:0.40;1120008:5:0.60,1100296:1:0.40;1100296:1:0.60,1100277:2:0.40;1100296:1:0.65,1100277:2:0.35"
HIW="0.70,0.725,0.75,0.775,0.80,0.825,0.85,0.875"
FINE=$(python3 - <<'PY'
v=[]
x=0.5530
while x<=0.5615001:
    v.append("%.6f"%x); x+=0.000125
print(",".join(v))
PY
)

launch(){ # name cmd...
  local nm="$1"; shift
  log "[启动] $nm :: $*"
  setsid nohup "$@" >> $W/day_${nm}.log 2>&1 < /dev/null &
  sleep 30
}

# ---- 门禁通道 A（GPU0）：bs640001 家族（含 1100277 高权重段新臂）----
launch gateA $PY -u manu/pipeline/npu/night_soup_batch.py --only-gate --gate-family-prefix bs640001 \
  --gpu 0 --gate-family-quota 40 --max-gates 40 --mem-gate-gb 40 \
  --sid-base 1190000 --sid-span 1000 --deadline "$END"
# ---- 门禁通道 B（GPU1）：三路汤家族 ----
launch gateB $PY -u manu/pipeline/npu/night_soup3.py --only-gate --gpu 1 \
  --max-gates 40 --mem-gate-gb 40 --deadline "$END"
# ---- 合成通道（GPU2）：三路汤 19 配方 ⇒ 再 1100277 高权重段 ----
cat > /tmp/synth_gpu2.sh <<'EOS'
#!/bin/bash
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="2026-11-19 07:00"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
timeout 9000 $PY -u manu/pipeline/npu/night_soup3.py --gpu 2 --sid-base 1400000 --sid-span 400 \
  --max-gates 0 --mem-gate-gb 40 --deadline "$END" --recipes "$RECIPES3" >> $W/soup3_synth.log 2>&1
timeout 9000 $PY -u manu/pipeline/npu/night_soup_batch.py --gpu 2 --parents 640001:5 --children-arms "1100277:2" \
  --weights "$HIW2" --sid-base 1420000 --sid-span 100 --max-gates 0 --gate-family-quota 8 \
  --mem-gate-gb 40 --deadline "$END" >> $W/soup_new_1420000.log 2>&1
EOS
sed -e "s|\$RECIPES3|$RECIPES|" -e "s|\$HIW2|$HIW|" /tmp/synth_gpu2.sh > $W/scripts/synth_gpu2.sh
chmod +x $W/scripts/synth_gpu2.sh
launch synth2 bash $W/scripts/synth_gpu2.sh
# ---- 合成通道（GPU3）：锚族 T1 细扫 57 点（只合成，门禁由通道 A/B 捡）----
cat > /tmp/synth_gpu3.sh <<'EOS'
#!/bin/bash
set -u
REPO=/tmp/pycharm_project_10ae9e2e; PY=/home/huangzhe/anaconda3/envs/uav/bin/python
W=/mnt/data/siping/runs/npu20ms/nano_search; END="2026-11-19 07:00"
cd $REPO || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
timeout 12000 $PY -u manu/pipeline/npu/night_soup_batch.py --gpu 3 --parents 640001:5 --children-arms "1100296:1" \
  --weights "$FINE3" --sid-base 1410000 --sid-span 100 --max-gates 0 --gate-family-quota 8 \
  --mem-gate-gb 40 --deadline "$END" >> $W/soup_new_1410000.log 2>&1
EOS
sed -e "s|\$FINE3|$FINE|" /tmp/synth_gpu3.sh > $W/scripts/synth_gpu3.sh
chmod +x $W/scripts/synth_gpu3.sh
launch synth3 bash $W/scripts/synth_gpu3.sh

log "[完成] 白天波 D1 四通道已拉起（门禁并发=2）"
