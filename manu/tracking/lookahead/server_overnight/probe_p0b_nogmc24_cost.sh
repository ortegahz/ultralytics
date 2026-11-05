#!/usr/bin/env bash
# ============================================================
# ★ P0b —— 24 序 no-GMC 口径的**可行性探测**（只测代价，不建全量）
#
#   目的：量出
#     (1) 24 序 no-GMC **特征集**全量构建的 CPU 墙钟 ETA；
#     (2) ms_cache 构建在 **CPU 上**的速率 ⇒ 全量 ETA ⇒「CPU 能否今夜内完成」的判断；
#     (3) CPU 建的缓存与 GPU 建的缓存**是否逐位相同**（直接拿 8 序 cs_t0043 那份比对）。
#
#   ★ 绝不执行全量构建（165,223 帧的特征集 / 24 序 ms_cache 都不建）。
#   ★ 探测产物一律落 /tmp/p0b_probe/，绝不写进正式数据集目录。
#   ★ 幂等：已有产物跳过。
#
#   用法（服务器）：
#     cd /tmp/pycharm_project_10ae9e2e
#     bash manu/tracking/lookahead/server_overnight/probe_p0b_nogmc24_cost.sh
#     env SEQ=VIDEO00033_19700101_015000 bash .../probe_p0b_nogmc24_cost.sh
# ============================================================
set -uo pipefail

REPO="${REPO:-/tmp/pycharm_project_10ae9e2e}"
PY="${PY:-/home/huangzhe/anaconda3/envs/uav/bin/python}"
RAW_ROOT="${RAW_ROOT:-/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg}"
REF24="${REF24:-/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median}"
REF8_NOGMC="${REF8_NOGMC:-/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/images/val}"
GPU_CACHE8="${GPU_CACHE8:-/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache}"
HM_W="${HM_W:-/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt}"
PROBE="${PROBE:-/tmp/p0b_probe}"
SEQ="${SEQ:-VIDEO00009_19700101_001531}"        # 24 序里帧数最少的一条（191 帧）
CACHE_SEQ="${CACHE_SEQ:-VIDEO00034_19700101_015052}"
THREADS="${THREADS:-6}"
WORKERS="${WORKERS:-1}"

export PYTHONPATH="$REPO"
export MPLCONFIGDIR=/tmp/mpl
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

cd "$REPO" || exit 2
mkdir -p "$PROBE"
RES="$PROBE/probe_results.json"
log(){ echo "[$(date '+%F %T')] $*"; }

# =====================================================================
# 阶段 1：特征集构建速率（单序列，临时 mini ref-dataset）
#   build_nogmc_rebuild_dataset.py 只用 ref-dataset 的**文件名清单 + 标签**，
#   图像内容自己从 --raw-root 重算 ⇒ 用 symlink 造一个 mini ref-dataset 即可单序列探测。
# =====================================================================
log "阶段1：特征集构建速率（序列 $SEQ）"
REF="$PROBE/ref_mini_$SEQ"
if [ ! -d "$REF/images/train" ]; then
  mkdir -p "$REF/images/train" "$REF/labels/train"
  n=0
  for f in "$REF24/images/train/${SEQ}__frame_"*.jpg; do
    ln -sf "$f" "$REF/images/train/$(basename "$f")"; n=$((n+1))
  done
  for f in "$REF24/labels/train/${SEQ}__frame_"*.txt; do
    cp -f "$f" "$REF/labels/train/$(basename "$f")" 2>/dev/null || true
  done
  log "  mini ref-dataset: $n 个特征文件（symlink）+ 标签拷贝"
fi
NFILES=$(ls "$REF/images/train"/*.jpg 2>/dev/null | wc -l)

OUT1="$PROBE/feat_mini_$SEQ"
if [ ! -f "$OUT1/images/train/summary.done" ]; then
  rm -rf "$OUT1"
  t0=$(date +%s)
  $PY manu/data/build_nogmc_rebuild_dataset.py \
    --ref-dataset "$REF" --raw-root "$RAW_ROOT" --output "$OUT1" \
    --splits train --mode identity --window 21 --stride-step 2 \
    --workers "$WORKERS" --verify-ch0-sample 24 \
    > "$PROBE/feat_build_${SEQ}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  log "  特征集构建 rc=$rc 用时 $((t1-t0))s（$NFILES 帧，workers=$WORKERS）"
  [ "$rc" -eq 0 ] && touch "$OUT1/images/train/summary.done"
else
  log "  SKIP 特征集构建（已存在）"
fi
grep -E "Completed in|VERIFY|VERIFY PASSED|FATAL" "$PROBE/feat_build_${SEQ}.log" 2>/dev/null | tail -6

# =====================================================================
# 阶段 2：ms_cache 在 **CPU** 上的速率（几十帧探针）
#   特征源用**已存在的 8 序 no-GMC 特征集**（同一模型/同口径 ⇒ 单帧成本与 24 序一致；
#   只是为了量速率）。缓存输出到临时目录。
# =====================================================================
log "阶段2：ms_cache CPU 速率（序列 $CACHE_SEQ）"
for LIM in 32 64 128; do
  OC="$PROBE/ms_cache_cpu_l$LIM"
  if [ -f "$OC/.done" ]; then log "  SKIP limit=$LIM"; continue; fi
  rm -rf "$OC"; mkdir -p "$OC"
  t0=$(date +%s)
  $PY manu/data/multiscale_heatmap_cache.py \
    --features-root "$REF8_NOGMC" --cache-root "$OC" \
    --sequences "$CACHE_SEQ" --limit-per-sequence "$LIM" \
    --scales 160,80 --region-threshold 0.06 --main-threshold 0.22 \
    --fusion-dilate 32 --min-area 4 --device cpu \
    --chunk-size 64 --batch-size 8 --cpu-threads "$THREADS" \
    --hm-weights "$HM_W" --ablate-p0 --overwrite \
    > "$PROBE/ms_cache_cpu_l$LIM.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "$((t1-t0))" > "$OC/.secs"
  [ "$rc" -eq 0 ] && touch "$OC/.done"
  log "  limit=$LIM rc=$rc 用时 $((t1-t0))s"
done

# =====================================================================
# 阶段 3：CPU 建的缓存 vs GPU 建的缓存（8 序 cs_t0043）是否逐位相同
# =====================================================================
log "阶段3：CPU 缓存 vs GPU 缓存（$CACHE_SEQ）逐位比对"
$PY - <<'PY'
import pickle, hashlib, json
from pathlib import Path
seq = "VIDEO00034_19700101_015052"
gpu = Path("/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache") / f"{seq}.pkl"
cpu = Path("/tmp/p0b_probe/ms_cache_cpu_l128") / f"{seq}.pkl"
out = {}
if gpu.exists() and cpu.exists():
    a = pickle.loads(gpu.read_bytes()); b = pickle.loads(cpu.read_bytes())
    out["gpu_md5"] = hashlib.md5(gpu.read_bytes()).hexdigest()
    out["cpu_md5"] = hashlib.md5(cpu.read_bytes()).hexdigest()
    out["byte_identical"] = gpu.read_bytes() == cpu.read_bytes()
    out["meta_gpu"] = a["meta"]; out["meta_cpu"] = b["meta"]
    out["meta_equal_except_root_weights"] = {
        k: (a["meta"].get(k), b["meta"].get(k))
        for k in sorted(set(a["meta"]) | set(b["meta"]))
        if a["meta"].get(k) != b["meta"].get(k)
    }
    out["payload_keys_equal"] = sorted(a) == sorted(b)
    for k in ("frame_index", "offset", "points", "sizes", "scores", "sums", "area", "tag"):
        if k in a and k in b:
            import numpy as np
            va, vb = a[k], b[k]
            out[f"array_{k}_equal"] = bool(np.array_equal(np.asarray(va), np.asarray(vb)))
    print(json.dumps({k: v for k, v in out.items() if k.startswith(("byte", "meta_equal", "array", "payload"))}, ensure_ascii=False, indent=2))
Path("/tmp/p0b_probe/cache_cpu_vs_gpu.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
PY

# =====================================================================
# 阶段 1b：**多序列并行**吞吐（workers 缩放）
#   脚本的并行粒度 = **每序列一个 task** ⇒ workers=24 就能 24 条序列同时跑。
#   用 4 条序列的 mini ref + workers=4 实测墙钟，再外推。
# =====================================================================
SEQS4="VIDEO00009_19700101_001531 VIDEO00029_19700101_014053 VIDEO00036_19700101_015740 VIDEO00030_19700101_014144"
REF4="$PROBE/ref_mini_4seq"
if [ ! -d "$REF4/images/train" ]; then
  mkdir -p "$REF4/images/train" "$REF4/labels/train"
  for sq in $SEQS4; do
    for f in "$REF24/images/train/${sq}__frame_"*.jpg; do
      ln -sf "$f" "$REF4/images/train/$(basename "$f")"
    done
    for f in "$REF24/labels/train/${sq}__frame_"*.txt; do
      cp -f "$f" "$REF4/labels/train/$(basename "$f")" 2>/dev/null || true
    done
  done
fi
N4=$(ls "$REF4/images/train"/*.jpg 2>/dev/null | wc -l)
OUT1B="$PROBE/feat_mini_4seq"
if [ ! -f "$OUT1B/images/train/summary.done" ]; then
  rm -rf "$OUT1B"
  t0=$(date +%s)
  $PY manu/data/build_nogmc_rebuild_dataset.py \
    --ref-dataset "$REF4" --raw-root "$RAW_ROOT" --output "$OUT1B" \
    --splits train --mode identity --window 21 --stride-step 2 \
    --workers 4 --verify-ch0-sample 24 \
    > "$PROBE/feat_build_4seq.log" 2>&1
  rc=$?; t1=$(date +%s)
  log "  4序列特征集构建 rc=$rc 用时 $((t1-t0))s（$N4 帧，workers=4）"
  [ "$rc" -eq 0 ] && touch "$OUT1B/images/train/summary.done"
else
  log "  SKIP 4序列特征集构建（已存在）"
fi
grep -E "Completed in|VERIFY" "$PROBE/feat_build_4seq.log" 2>/dev/null | tail -4

# =====================================================================
# 阶段 3b：**整序列** CPU 缓存 vs GPU 缓存逐位比对
#   阶段 3 用 --limit-per-sequence 128 比 1239 帧的 GPU 缓存 ⇒ 帧范围不同、数组不可比。
#   这里用**整条序列**（1239 帧）重跑 CPU ⇒ 帧范围对齐，才能真正回答
#   「CPU 重建的缓存与 GPU 重建的是否逐位相同」。
# =====================================================================
log "阶段3b：整序列 CPU 缓存 vs GPU 缓存逐位比对"
OCFULL="$PROBE/ms_cache_cpu_full"
if [ ! -f "$OCFULL/.done" ]; then
  rm -rf "$OCFULL"; mkdir -p "$OCFULL"
  t0=$(date +%s)
  $PY manu/data/multiscale_heatmap_cache.py \
    --features-root "$REF8_NOGMC" --cache-root "$OCFULL" \
    --sequences "$CACHE_SEQ" \
    --scales 160,80 --region-threshold 0.06 --main-threshold 0.22 \
    --fusion-dilate 32 --min-area 4 --device cpu \
    --chunk-size 64 --batch-size 8 --cpu-threads "$THREADS" \
    --hm-weights "$HM_W" --ablate-p0 --overwrite \
    > "$PROBE/ms_cache_cpu_full.log" 2>&1
  rc=$?; t1=$(date +%s)
  echo "$((t1-t0))" > "$OCFULL/.secs"
  [ "$rc" -eq 0 ] && touch "$OCFULL/.done"
  log "  整序列 CPU 缓存 rc=$rc 用时 $((t1-t0))s（1239 帧）"
fi

$PY - <<'PY2'
import pickle, hashlib, json
import numpy as np
from pathlib import Path
seq = "VIDEO00034_19700101_015052"
gpu = Path("/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache") / f"{seq}.pkl"
cpu = Path("/tmp/p0b_probe/ms_cache_cpu_full") / f"{seq}.pkl"
out = {}
if gpu.exists() and cpu.exists():
    gb, cb = gpu.read_bytes(), cpu.read_bytes()
    a = pickle.loads(gb); b = pickle.loads(cb)
    out["gpu_md5"] = hashlib.md5(gb).hexdigest()
    out["cpu_md5"] = hashlib.md5(cb).hexdigest()
    out["byte_identical"] = gb == cb
    out["meta_diff"] = {k: (a["meta"].get(k), b["meta"].get(k))
                        for k in sorted(set(a["meta"]) | set(b["meta"]))
                        if a["meta"].get(k) != b["meta"].get(k)}
    for k in ("frame_index", "offset", "points", "sizes", "scores", "sums", "area", "tag"):
        if k in a and k in b:
            va, vb = np.asarray(a[k]), np.asarray(b[k])
            out[f"array_{k}_equal"] = bool(np.array_equal(va, vb))
            if va.shape == vb.shape:
                out[f"array_{k}_max_abs_diff"] = int(np.abs(va.astype(np.int64) - vb.astype(np.int64)).max()) if va.size else 0
                out[f"array_{k}_ndiff"] = int((va != vb).sum())
            else:
                out[f"array_{k}_shape"] = [list(va.shape), list(vb.shape)]
    print(json.dumps(out, ensure_ascii=False, indent=2))
Path("/tmp/p0b_probe/cache_cpu_vs_gpu_full.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
PY2

log "探测完成 ⇒ $RES"
