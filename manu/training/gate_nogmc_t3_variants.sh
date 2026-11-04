#!/bin/bash
# ==============================================================================
# no-GMC T3 study 的**变体门禁**（跨种子 soup + 沿 epoch 轴 SWA）—— 2026-11-03
#
# 与旧脚本（gate_swa_variants.sh / gate_crossseed_soups.sh）的关系
# -------------------------------------------------------------
# 旧脚本把候选列表**写死**了（swa_0077_last3 / cs_t0077 …）⇒ 换 study 就要改代码。
# 本脚本把候选来源改成**数据驱动**：
#   · soup：build_crossseed_soups.py --families-json（家族定义外置，跨 3 个 study 取成员）
#   · SWA ：build_nogmc_t3_swa_candidates.py（扫 study.db ⇒ 每个 trial 的末 3 个 epoch）
# ⇒ **同一个脚本对任何 study 都能用**，不用改一行代码。
#
# 铁律
# ----
# · **评估不得与训练同卡** ⇒ 本脚本先等「训练结束 + GPU 空闲」
#   （判据在 _lib_train_end.sh：墙钟已过 + 调度器已退出 + 4 张卡上没有训练进程）。
# · 门禁**串行**（同一时刻只跑 1 个），避免 4 并发抢显存。
# · 产物落 /mnt/data（物理盘）。
# · **按价值排序**跑：价值高的先跑，时间不够就自然截断（低价值的排在后面被砍掉）。
#
# 用法（服务器，conda `uav`，项目根 /tmp/pycharm_project_10ae9e2e）：
#   setsid nohup bash manu/training/gate_nogmc_t3_variants.sh \
#     >> /mnt/data/siping/runs/nogmc_t3/logs/gate_variants.log 2>&1 < /dev/null &
#
# 环境变量
# ---------
#   OUT_ROOT        study 产物根（默认 /mnt/data/siping/runs/nogmc_t3）
#   STUDY_NAME      Optuna study 名（默认 mix_ratio_search_nogmc_t3）
#   TRAIN_STOP      训练截止墙钟（默认读 <OUT_ROOT>/runtime_config.json 的 stop_at）
#   GATE_DEADLINE   门禁硬截止（到点放弃剩余候选）
#   FAMILIES_JSON   家族定义文件（默认 manu/training/nogmc_t3_families.json）
#   GATE_PRIORITY   价值排序表（空格分隔；先命中者排前面）
# ==============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tmp/pycharm_project_10ae9e2e}"
RUNS_BASE="${RUNS_BASE:-/mnt/data/siping/runs}"
export OUT_ROOT="${OUT_ROOT:-$RUNS_BASE/nogmc_t3}"
REF_DIR="${REF_DIR:-$RUNS_BASE/nogmc_ref_weekend}"
LOG="${LOG:-$OUT_ROOT/logs/gate_variants.log}"
STUDY_NAME="${STUDY_NAME:-mix_ratio_search_nogmc_t3}"
FAMILIES_JSON="${FAMILIES_JSON:-manu/training/nogmc_t3_families.json}"
SWA_DIR="$OUT_ROOT/swa"
BASE="$REF_DIR/uav_cache_base.pkl"
GATE_DEADLINE="${GATE_DEADLINE:-}"
WAIT_HARD_CAP_MIN="${WAIT_HARD_CAP_MIN:-240}"

cd "$PROJECT_ROOT" || exit 1
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav || exit 1
export PYTHONPATH=.
mkdir -p "$OUT_ROOT/logs" "$SWA_DIR/gate"
# ⚠️ 日志只写进 $LOG 一份；stdout 只在**交互式终端**时再打一遍
#    （避免 `>> $LOG 2>&1` 时 tee 让每行出现两遍）。
say(){ local m="[$(date '+%F %T')] [t3-gate] $*"; printf '%s\n' "$m" >> "$LOG"; if [ -t 1 ]; then printf '%s\n' "$m"; fi; return 0; }

# shellcheck source=manu/training/_lib_train_end.sh
source manu/training/_lib_train_end.sh || { say "找不到 _lib_train_end.sh ⇒ 退出"; exit 1; }

[ -f "$BASE" ] || { say "缺基线缓存 $BASE ⇒ 退出"; exit 1; }

# ---- 训练截止：优先读 runtime_config.json 的 stop_at ------------------------
if [ -z "$TRAIN_STOP" ]; then
  TRAIN_STOP="$(python3 -c "import json;print(json.load(open('$OUT_ROOT/runtime_config.json')).get('stop_at',''))" 2>/dev/null || echo "")"
fi
[ -z "$TRAIN_STOP" ] && TRAIN_STOP="${GATE_DEADLINE:-$(date '+%F %T')}"
say "训练截止 TRAIN_STOP=$TRAIN_STOP ｜ 门禁硬截止=${GATE_DEADLINE:-（不限）}"

# ---- ① 等训练彻底结束（铁律：评估不得与训练同卡）--------------------------
say "① 等待训练结束 + GPU 空闲 …"
wait_train_end "$OUT_ROOT" "$TRAIN_STOP" "$WAIT_HARD_CAP_MIN" >> "$LOG" 2>&1
WAIT_RC=$?
if [ "$WAIT_RC" != "0" ]; then
  say "❌ 等待失败（rc=$WAIT_RC）⇒ **中止门禁**（绝不在训练同卡时评估）"
  exit "$WAIT_RC"
fi
say "① 训练已结束、GPU 已释放 ⇒ 开始变体门禁"

# ---- ② 构建跨种子 soup（跨 3 个 study 取成员）------------------------------
say "② 构建跨种子 soup（家族定义 $FAMILIES_JSON）…"
python manu/training/build_crossseed_soups.py \
  --families-json "$FAMILIES_JSON" \
  --focus-root "$RUNS_BASE/nogmc_focus" \
  --weekend-root "$RUNS_BASE/nogmc_weekend" \
  --out-dir "$SWA_DIR" --repo "$PROJECT_ROOT" >> "$LOG" 2>&1
say "   soup 构建 rc=$?"

# ---- ③ 生成 SWA 候选（每个 trial 的末 3 个 epoch 平均）----------------------
say "③ 生成 SWA 候选清单 + 融合 …"
python manu/training/build_nogmc_t3_swa_candidates.py \
  --out-root "$OUT_ROOT" --study-name "$STUDY_NAME" \
  --swa-dir "$SWA_DIR" --repo "$PROJECT_ROOT" --fuse >> "$LOG" 2>&1
say "   SWA 候选 rc=$?"

# ---- ④ 按价值排序 -----------------------------------------------------------
# 排序表（先命中者排前面）：
#   · cs_t0033 / cs_t0008 —— 两个「差一点点过线」的家族，soup 降方差最可能补上缺口
#   · 15ep 的 tier A 主攻臂 —— 本轮主攻方向（T3 目标 0.87）
#   · cs_t0027 / 15ep soup —— 其余家族
#   · 其余 SWA（种子副本、tier B/C）—— 兜底
GATE_PRIORITY="${GATE_PRIORITY:-cs_t0033 cs_t0008 T3_W2_LR30_15EP T3_W3_LR30_15EP T3_W3_LR2882_15EP cs_t0027 cs_w2lr30_15 cs_w3lr30_15 INS_W2_LR30_15EP_S2 INS_W3_LR30_15EP_S2 INS_T0033 INS_T0008 INS_T0027}"
say "④ 价值排序表：$GATE_PRIORITY"

ORDER_FILE="$SWA_DIR/gate_order.txt"
python3 - "$SWA_DIR" "$GATE_PRIORITY" > "$ORDER_FILE" <<'PY'
import json, sys
from pathlib import Path

swa_dir = Path(sys.argv[1])
tokens = sys.argv[2].split()

cands = []   # (key, name, kind)
mp = swa_dir / "crossseed_manifest.json"
if mp.is_file():
    try:
        doc = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ORDER] manifest 解析失败：{exc}", file=sys.stderr)
        doc = {}
    for n in doc.get("made") or []:
        stem = Path(n).stem          # cs_t0033
        cands.append((stem, stem, "soup"))
sp = swa_dir / "swa_candidates.json"
if sp.is_file():
    try:
        sdoc = json.loads(sp.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ORDER] swa 清单解析失败：{exc}", file=sys.stderr)
        sdoc = {}
    for c in sdoc.get("candidates") or []:
        name = c.get("name")
        if not name:
            continue
        key = c.get("role") or name
        cands.append((key, name, "swa"))


def rank(key: str) -> int:
    for i, tok in enumerate(tokens):
        if tok in key:
            return i
    return 999


cands.sort(key=lambda c: (rank(c[0]), c[0]))
for key, name, kind in cands:
    print(f"{name}\t{kind}\t{key}")
print(f"[ORDER] 候选 {len(cands)} 个（soup {sum(1 for c in cands if c[2]=='soup')} / "
      f"swa {sum(1 for c in cands if c[2]=='swa')}）", file=sys.stderr)
PY
say "   排序结果 $(wc -l < "$ORDER_FILE") 行 ⇒ $ORDER_FILE"

# ---- ⑤ 串行门禁 ------------------------------------------------------------
GATE_S=$(date -d "$GATE_DEADLINE" '+%s' 2>/dev/null || echo 0)
while IFS=$'\t' read -r NAME KIND KEY; do
  [ -n "$NAME" ] || continue
  case "$NAME" in \#*) continue ;; esac
  if [ "$GATE_S" -gt 0 ]; then
    NOW_S=$(date '+%s')
    if [ "$NOW_S" -ge "$GATE_S" ]; then
      say "⏰ 到达门禁硬截止 $GATE_DEADLINE ⇒ 放弃剩余候选（已完成的保留在磁盘上）"
      break
    fi
  fi
  W="$SWA_DIR/$NAME.pt"
  [ -f "$W" ] || { say "跳过 $NAME（权重不存在）"; continue; }
  GD="$SWA_DIR/gate/$NAME"
  if [ -f "$GD/gate_report.json" ] || ls "$GD"/epoch*/gate_report.json >/dev/null 2>&1; then
    say "跳过 $NAME（已有门禁结果）"; continue
  fi
  mkdir -p "$GD"
  say "门禁 $NAME（$KIND · $KEY）…"
  python manu/training/auto_epoch_gate_check.py \
    --weights "$W" --out-dir "$GD" \
    --gpus 4 --device 0 --infer-gpu 0 --cache-gpus 0,1,2,3 \
    --baseline-cache "$BASE" \
    --min-track2-f1 0.985 --max-track2-fp 112 --min-track1-f1 0.8958967482400269 \
    --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
    --ft-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc \
    --base-track1-f1 0.9008967482400269 --base-track1-tp 21499 --base-track1-fp 1118 \
    --base-track2-f1 0.981074481074481 --base-track2-tp 16070 --base-track2-fp 52 --base-track2-fn 568 \
    --base-track3-f1 0.7578120617110798 --base-track3-tp 6754 --base-track3-fp 536 --base-track3-fn 3781 \
    >> "$LOG" 2>&1
  say "  $NAME 门禁 rc=$?"
  sleep 10
done < "$ORDER_FILE"

say "✅ 跨种子 soup 门禁完成；结果在 $SWA_DIR/gate/"
say "✅ SWA 变体门禁全部完成；结果在 $SWA_DIR/gate/"
