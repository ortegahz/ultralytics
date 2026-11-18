#!/bin/bash
# 夜间接力 v4（2026-11-17 15:00）—— **递归彩票波**：用本夜新产生的「三轴全过」臂当子，
# 继续跟 640001@5 做 2 成员汤 ⇒ **全部堆进同一个家族 `soup:bs640001:w2`**（权重不进家族键）。
# 动机（15:00 实测）：batchD 的 w=0.53/0.56 两臂（1117001/1117002）**三轴全过** ⇒
#   「冠军当子」这条路确实能批量产三轴全过臂 ⇒ 那就把它做成**自喂**的循环：
#   每波把上一波的三轴全过臂收进子池 → 再合成 → 再收。
# 铁律：MemAvailable ≥ 48GB 才启新臂；OMP/MKL/OPENBLAS/NUMEXPR/NUMBA=4；通道 ≤ 4；deadline 07:00。
W=/mnt/data/siping/runs/npu20ms/nano_search
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
REPO=/tmp/pycharm_project_10ae9e2e
cd "$REPO" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4

avail_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
now_s() { date +%s; }
END_S=$(date -d "2026-11-18 07:00" +%s)

wait_gpu3() {
  while pgrep -f "night_soup_batch.py --gpu 3" > /dev/null 2>&1; do sleep 60; done
}

# ---- 收子：家族 bs640001:* 里「三轴全过 + 还没门禁」的臂（按边际排序，取前 12）----
harvest() {
  "$PY" - <<'PYEOF'
import json, os
L = "/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json"
G = "/mnt/data/siping/runs/npu20ms_offcal/gate"
led = json.load(open(L)); arms = led.get("arms") or {}
base = led.get("baseline_v2") or {}
bn, b4, bf = base.get("nano"), base.get("nano4"), base.get("f1")
rows = []
for sid, a in arms.items():
    sf = str(a.get("soup_family") or "")
    if not sf.startswith("bs640001:"):
        continue
    if not (a.get("synth") and a.get("soup_members")):
        continue
    pk = a.get("picked") or {}
    if pk.get("nano") is None:
        continue
    if not (pk["nano"] > bn and pk["nano4"] > b4 and pk["f1"] > bf):
        continue
    # 🟢 15:20 修正：**冠军（已门禁）也是最好的子** ⇒ 不再排除「已门禁」的臂
    #   （排除只是为了不重复门禁；本脚本是**造新汤**，不重复门禁）
    gp = os.path.join(G, "NANO_%s_ep5_official" % sid, "gate_report.json")
    strict = 0
    if os.path.exists(gp):
        try:
            r = json.load(open(gp)); gg = r.get("gates") or {}
            t1 = (gg.get("track1_uav") or {}).get("value"); fp = (gg.get("track2_golden") or {}).get("fp")
            t3 = (gg.get("track3_blind") or {}).get("f1")
            if t1 is not None and t3 is not None:
                strict = int(t1 >= base["t1"] and fp <= base["t2_fp"] and t3 >= base["t3"])
        except Exception:
            pass
    if not os.path.isfile(os.path.join("/mnt/data/siping/runs/npu20ms_offcal/trials",
                                       "trial_%s" % sid, "weights", "epoch5.pt")):
        continue
    mg = min((pk["nano"] - bn) / 0.0042, (pk["nano4"] - b4) / 0.0060, (pk["f1"] - bf) / 0.0008)
    rows.append((-strict, -mg, int(sid)))
rows.sort()
print(",".join("%d:5" % sid for _s, _mg, sid in rows[:14]))
PYEOF
}

if [ "${NO_WAIT:-0}" != "1" ]; then wait_gpu3; fi
WAVE=0
while [ "$(now_s)" -lt "$END_S" ]; do
  KIDS=$(harvest)
  N=$(echo "$KIDS" | tr ',' '\n' | grep -c ':' || true)
  echo "[$(date '+%F %T')] 波次 $WAVE：收子 $N 个 ⇒ ${KIDS}"
  if [ "$N" -lt 3 ]; then
    echo "[$(date '+%F %T')] 子池不足 ⇒ 等 600s 再收"
    sleep 600
    continue
  fi
  while true; do
    a=$(avail_gb)
    if [ "${a:-0}" -ge 48 ]; then break; fi
    echo "[$(date '+%F %T')] 内存闸门 avail=${a}GB < 48GB ⇒ 等 120s"
    sleep 120
  done
  WAVE=$((WAVE+1))
  "$PY" -u manu/pipeline/npu/night_soup_batch.py --gpu 3 --parents "640001:5" \
      --children-arms "$KIDS" --weights "0.55,0.58,0.62" --sid-base 1120000 \
      --max-gates 30 --gate-family-quota 12 --mem-gate-gb 44 \
      --deadline "2026-11-18 07:00" >> "$W/soup_batch_recursion.log" 2>&1
  echo "[$(date '+%F %T')] 波次 $WAVE 结束"
  sleep 60
done
echo "[$(date '+%F %T')] 递归波结束"
