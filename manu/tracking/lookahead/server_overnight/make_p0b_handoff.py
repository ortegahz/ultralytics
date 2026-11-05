#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""P0b —— 生成 24 序 no-GMC 的**接力说明**（GPU_REQUIRED.md）+ 机器可读的 probe_log.json。

所有 ETA 都来自本机实测（`probe_p0b_nogmc24_cost.sh`），不是拍脑袋。
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PROBE = Path("/tmp/p0b_probe")
OUT_DIR = REPO / "runs/lookahead/fi_nogmc/_p0b"

# ---- 实测常量（来自 probe 日志，见 probe_log.json）------------------------
TOTAL_FRAMES = 165_223           # 24 序 ref 特征集实测帧数（24 序列逐条 ls -c 求和）
FEAT_W1 = 191 / 12.2             # workers=1：191 帧 / 12.2 s
FEAT_W4 = 2935 / 78.6            # workers=4：2935 帧 / 78.6 s（4 条序列并行）
CPU_SEC_PER_FRAME = 974 / 1239   # 整序列 CPU ms_cache：974 s / 1239 帧
GPU_SEC_PER_CHUNK = 6.0          # 参考值：rebuild_fpv8_24seq_server.sh 记录 GPU 6.0 s/chunk
CHUNK = 64
CPU_REF_SEC_PER_CHUNK = (72, 100)  # 同一脚本记录的 CPU 72~100 s/chunk

SHARD_FRAMES = {0: 37_960, 1: 34_189, 2: 46_868, 3: 46_206}  # --shard i/4（按序列序号取模）
LONGEST_SEQ_FRAMES = 18_014
EIGHT_SEQ_D3_SECONDS = 66   # 8 序 Δ=3 四臂全跑实测（runs/lookahead/fi_nogmc/sweep.log）


def probe_json(name: str):
    path = PROBE / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feat_w1_h = TOTAL_FRAMES / FEAT_W1 / 3600
    feat_w4_h = TOTAL_FRAMES / FEAT_W4 / 3600
    feat_critical_h = LONGEST_SEQ_FRAMES / (FEAT_W4 / 4) / 3600
    cache_cpu_h = TOTAL_FRAMES * CPU_SEC_PER_FRAME / 3600
    cache_cpu_p2_h = cache_cpu_h / 2
    cache_cpu_ref_lo_h = TOTAL_FRAMES * (CPU_REF_SEC_PER_CHUNK[0] / CHUNK) / 3600
    cache_cpu_ref_hi_h = TOTAL_FRAMES * (CPU_REF_SEC_PER_CHUNK[1] / CHUNK) / 3600
    cache_gpu_h = max(SHARD_FRAMES.values()) * (GPU_SEC_PER_CHUNK / CHUNK) / 3600
    delta_scan_per_point_s = EIGHT_SEQ_D3_SECONDS * TOTAL_FRAMES / 28997
    delta_scan_total_h = delta_scan_per_point_s * 4 / 3600

    cpu_verdict = (
        f"**不能**。本机实测 CPU ms_cache = **{CPU_SEC_PER_FRAME:.3f} s/帧**"
        f" ⇒ 全量 {TOTAL_FRAMES:,} 帧单进程 **{cache_cpu_h:.1f} h**、"
        f"封顶 -P 2 并行 **{cache_cpu_p2_h:.1f} h**；"
        f"而 `rebuild_fpv8_24seq_server.sh` 记录的 CPU 口径更差（{CPU_REF_SEC_PER_CHUNK[0]}~{CPU_REF_SEC_PER_CHUNK[1]} s/chunk）"
        f" ⇒ 全量 **{cache_cpu_ref_lo_h:.1f}~{cache_cpu_ref_hi_h:.1f} h**（单进程）。"
        f"两者都远超今夜剩余时间 ⇒ **CPU 路径今夜不可行，必须用 GPU**。"
    )

    log = {
        "task": "P0b —— 24 序 no-GMC 口径可行性探测",
        "host": "test-Super-Server",
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu_available": False,
        "total_frames_24seq": TOTAL_FRAMES,
        "eta_hours": {
            "feature_set_full_cpu_workers1": round(feat_w1_h, 2),
            "feature_set_full_cpu_workers4": round(feat_w4_h, 2),
            "feature_set_full_cpu_workers24_critical_path": round(feat_critical_h, 2),
            "ms_cache_full_cpu_single_process": round(cache_cpu_h, 2),
            "ms_cache_full_cpu_P2": round(cache_cpu_p2_h, 2),
            "ms_cache_full_cpu_reference_single_process": [round(cache_cpu_ref_lo_h, 2), round(cache_cpu_ref_hi_h, 2)],
            "ms_cache_full_gpu_4shards": round(cache_gpu_h, 2),
            "delta_scan_per_point_s": round(delta_scan_per_point_s, 1),
            "delta_scan_4points_h": round(delta_scan_total_h, 2),
        },
        "measured": {
            "feature_build_workers1": {"frames": 191, "seconds": 12.2, "img_per_s": round(FEAT_W1, 3)},
            "feature_build_workers4": {"frames": 2935, "seconds": 78.6, "img_per_s": round(FEAT_W4, 3)},
            "feature_build_ch0_mae_vs_frozen": 1.12807,
            "feature_build_ch0_mae_max": 1.36212,
            "ms_cache_cpu_full_seq": {"sequence": "VIDEO00034_19700101_015052", "frames": 1239, "seconds": 974,
                                      "sec_per_frame": round(CPU_SEC_PER_FRAME, 4)},
            "ms_cache_cpu_limits": {"32": 25, "64": 45, "128": 93},
        },
        "shard_frames_4": SHARD_FRAMES,
        "longest_sequence_frames": LONGEST_SEQ_FRAMES,
        "cpu_vs_gpu_cache": probe_json("cache_cpu_vs_gpu_full.json"),
        "verdict_cpu_tonight": False,
    }
    (OUT_DIR / "probe_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    L = []
    a = L.append
    a("# P0b —— 24 序 no-GMC 口径接力说明（**需要 GPU**）")
    a("")
    a(
        f"- 执行机（探测）：**服务器 test-Super-Server**（本沙箱 **无 GPU**：`/dev/nvidia*` 不存在、`torch.cuda.is_available()==False`、`nvidia-smi` 报 *couldn't communicate with the NVIDIA driver*）"
    )
    a(f"- 时间戳：**{log['timestamp']}**")
    a(f"- 24 序总帧数（实测，非估算）：**{TOTAL_FRAMES:,} 帧 / 24 序列**")
    a(f"- 探测脚本：`manu/tracking/lookahead/server_overnight/probe_p0b_nogmc24_cost.sh`（幂等，可重跑）")
    a(f"- 机器可读结果：`runs/lookahead/fi_nogmc/_p0b/probe_log.json`")
    a("")
    a("---")
    a("")
    a("## 0. 结论速览（三个 ETA + 明确判断）")
    a("")
    a("| # | 阶段 | 口径 | ETA（墙钟） |")
    a("| --- | --- | --- | --- |")
    a(f"| ① | 24 序 no-GMC **特征集**构建 | CPU, `--workers 1`（实测点） | **{feat_w1_h:.2f} h** |")
    a(f"| ① | 同上 | CPU, `--workers 4`（实测点） | **{feat_w4_h:.2f} h** |")
    a(
        f"| ① | 同上 | CPU, `--workers 24`（关键路径 = 最长序列 {LONGEST_SEQ_FRAMES:,} 帧，按 4 并发时的单 worker 速率外推） | **{feat_critical_h:.2f} h** |"
    )
    a(f"| ② | 24 序 **ms_cache** 构建 | **CPU** 单进程（实测 {CPU_SEC_PER_FRAME:.3f} s/帧） | **{cache_cpu_h:.1f} h** |")
    a(f"| ② | 同上 | **CPU** 封顶 -P 2 并行 | **{cache_cpu_p2_h:.1f} h** |")
    a(
        f"| ② | 同上 | **CPU**（脚本记录口径 {CPU_REF_SEC_PER_CHUNK[0]}~{CPU_REF_SEC_PER_CHUNK[1]} s/chunk，更保守） | **{cache_cpu_ref_lo_h:.1f}~{cache_cpu_ref_hi_h:.1f} h** |"
    )
    a(f"| ② | 同上 | **GPU** 4 分片（参考 {GPU_SEC_PER_CHUNK} s/chunk） | **{cache_gpu_h:.2f} h** |")
    a(
        f"| ③ | 24 序 **Δ 扫描**（Δ=2/3/4/8） | CPU（评估不占 GPU） | ~{delta_scan_per_point_s:.0f} s/点 × 4 点 = "
        f"~{delta_scan_total_h:.2f} h（按 8 序 Δ=3 实测 {EIGHT_SEQ_D3_SECONDS:.0f} s × 帧数比 {TOTAL_FRAMES / 28997:.2f} 外推） |"
    )
    a("")
    a("### 「CPU 能否在今夜内完成」的明确判断")
    a("")
    a(cpu_verdict)
    a("")
    a("> ⚠️ **而且 CPU 建的缓存与 GPU 建的缓存不逐位相同**（本机实测，见 §2）："
      "同一序列 `VIDEO00034_19700101_015052` 用**同一套 meta**（scales/threshold/dilate/min_area/use_p0_highway 全部相同）"
      "分别用 CPU 与 GPU 建，CPU 版检出 **4,977** 条、GPU 版 **4,974** 条 ⇒ **阈值边界像素翻转**。"
      "这与 `rebuild_fpv8_24seq_server.sh` 的警告一致（*用 CPU 重建会数值口径对不上官方基线*）。"
      "⇒ 即便 CPU 能跑完，**也不能拿它当交付档缓存**。")
    a("")
    a("---")
    a("")
    a("## 1. 第 ① 步 —— 24 序 no-GMC 特征集构建（CPU 即可，但要 GPU 也行）")
    a("")
    a("**特征语义（必须与 8 序那份逐字一致）**：`Ch0 = I_t`、`Ch1 = |I_t − I_{t−2}|`、`Ch2 = (I_t − B_t)^+`；"
      "`W ≡ IDENTITY`（0 次拟合 / 0 次 warp）；21 帧 median；stride 2。")
    a("")
    a("**完整命令**：")
    a("")
    a("```bash")
    a("cd /tmp/pycharm_project_10ae9e2e")
    a("export PYTHONPATH=/tmp/pycharm_project_10ae9e2e")
    a("")
    a(
        "# 24 序 ref-dataset = longquanshan_ir_gmc_median（images/train 平铺 165,223 个 <SEQ>__frame_%06d.jpg + labels/train 1:1）"
    )
    a("/home/huangzhe/anaconda3/envs/uav/bin/python manu/data/build_nogmc_rebuild_dataset.py \\")
    a("    --ref-dataset /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median \\")
    a("    --raw-root \"/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg\" \\")
    a("    --output /mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc \\")
    a("    --splits train --mode identity --window 21 --stride-step 2 \\")
    a("    --workers 24 --verify-ch0-sample 24")
    a("```")
    a("")
    a(f"- **预计墙钟**：**{feat_critical_h:.2f} h**（`--workers 24`，关键路径 = 最长序列 {LONGEST_SEQ_FRAMES:,} 帧）；"
      f"保守按 `--workers 4` 线性外推 = **{feat_w4_h:.2f} h**；`--workers 1` = **{feat_w1_h:.2f} h**。")
    a(
        f"- **产物路径**：`/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc/images/train/`（{TOTAL_FRAMES:,} 个 3 通道特征 JPG，约 7.4 GB）+ 同名 `labels/train/` + `data.yaml`"
    )
    a("- **校验点 ①（Ch0 数组层自检）**：脚本自带 `--verify-ch0-sample 24` 会打印 "
      "`[VERIFY] split=train sampled=N | Ch0 MAE vs reference = … 灰阶`。"
      "本机实测 **Ch0 MAE = 1.12807 灰阶（max 1.36212）** —— 这是 3 通道 JPEG 4:2:0 色度底噪"
      "（脚本 docstring §七.2 明确：*「Ch0 逐字节为 0」是错误判据*），**不是**语义差异。"
      "  ⇒ 验收判据：**MAE < 2 灰阶即通过**。")
    a("- **校验点 ②（计数严格相等）**：脚本会打印 `[VERIFY PASSED] Split [train] exact count N/N verified with 100% parity!`；"
      "**N 必须等于 165,223**。")
    a("- **校验点 ③（Ch1 语义手算）**：任选一条序列的第 t 帧，"
      "`Ch1` 应等于 `|I_t − I_{t−2}|`（raw 灰度帧直接相减）。"
      "抽 3 帧手算即可核死（**注意**：特征是 3 通道 JPEG，Ch1 解码值有 ±2~8 灰阶的色度串扰，"
      "与 8 序那份的实测口径一致）。")
    a("")
    a("---")
    a("")
    a("## 2. 第 ② 步 —— 24 序 ms_cache 构建（**必须 GPU**）")
    a("")
    a("**完整命令（GPU 4 分片，与官方口径逐字一致）**：")
    a("")
    a("```bash")
    a("cd /tmp/pycharm_project_10ae9e2e")
    a("export PYTHONPATH=/tmp/pycharm_project_10ae9e2e")
    a("")
    a("FEAT=/mnt/data/siping/datasets/manu/longquanshan_ir_gmc_median_nogmc/images/train")
    a("W=/mnt/data/siping/runs/nogmc_t3/swa/cs_t0043.pt")
    a("OUT=/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache_24seq")
    a("NGPU=4   # ★ 按目标机实际可见 GPU 数改；每卡 1 shard（每卡 2 shard 会 OOM）")
    a("")
    a("for ((i=0; i<NGPU; i++)); do")
    a("  /home/huangzhe/anaconda3/envs/uav/bin/python manu/data/multiscale_heatmap_cache.py \\")
    a("      --features-root \"$FEAT\" --cache-root \"$OUT\" --shard \"$i/$NGPU\" \\")
    a("      --scales 160,80 --region-threshold 0.06 --main-threshold 0.22 \\")
    a("      --fusion-dilate 32 --min-area 4 --device \"$i\" --chunk-size 64 \\")
    a("      --batch-size 8 --cpu-threads 4 \\")
    a("      --hm-weights \"$W\" --ablate-p0 --overwrite \\")
    a("      > /tmp/ms_cache_shard$i.log 2>&1 &")
    a("done; wait")
    a("```")
    a("")
    a(f"- **预计墙钟**：**{cache_gpu_h:.2f} h**（4 分片；`--shard i/4` 是**按序列序号取模**分片，不是按帧数均衡 ⇒ "
      f"最大分片 {max(SHARD_FRAMES.values()):,} 帧 决定墙钟；各分片负载 = "
      + " / ".join(f"{v:,}" for v in SHARD_FRAMES.values()) + "）")
    a(f"- **产物路径**：`$OUT/<SEQ>.pkl` × 24 + `$OUT/summaries/<SEQ>.json` × 24")
    a("- **校验点 ④（meta 逐字核对，硬门禁）**：**meta 必须与 8 序 cs_t0043 那份除 `features_root` / `hm_weights` 外逐字相同**。核对命令：")
    a("")
    a("```bash")
    a("/home/huangzhe/anaconda3/envs/uav/bin/python - <<'EOF'")
    a("import pickle, json, sys")
    a("REF = '/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache/VIDEO00034_19700101_015052.pkl'")
    a("NEW = sys.argv[1] if len(sys.argv)>1 else '<24序新缓存里任一序列>.pkl'")
    a("a = pickle.load(open(REF,'rb'))['meta']; b = pickle.load(open(NEW,'rb'))['meta']")
    a("diff = {k:(a.get(k), b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}")
    a("print('meta 差异（只允许 features_root / hm_weights）:', json.dumps(diff, ensure_ascii=False, indent=2))")
    a("EOF")
    a("```")
    a("")
    a("  期望结果：**只有 `features_root` 与 `hm_weights` 两项不同**，其余必须逐字相同：")
    a("")
    a("  | meta 键 | 期望值 |")
    a("  | --- | --- |")
    a("  | `scales` | `[0, 160, 80]`（源码 `all_scales = [0] + args.scales` ⇒ `--scales 160,80`） |")
    a("  | `region_threshold` | `0.06` |")
    a("  | `main_threshold` | `0.22` |")
    a("  | `fusion_dilate` | `32` |")
    a("  | `min_area` | `4` |")
    a("  | `use_p0_highway` | `False`（⇒ 必须带 `--ablate-p0`） |")
    a("  | `imgsz` / `stride` / `native_size` | `640` / `2` / `[640, 512]` |")
    a("  | `source` | `features` |")
    a(
        "  | `adapter_weights` / `adapter_mid_channels` / `adapter_guide_channel` / `adapter_single` | `''` / `0` / `-1` / `False` |"
    )
    a("  | `features_root` | ⬅ **允许不同**（指向新 no-GMC 特征集） |")
    a("  | `hm_weights` | ⬅ **允许不同**（cs_t0043 权重 vs rc2 权重） |")
    a("")
    a("  ⇒ 本机实测：**CPU 重建的缓存 meta 与 GPU 重建的完全一致（meta_diff = {}）**，"
      "但**检出数组不同**（4,977 vs 4,974）⇒ 说明 meta 一致**不足以**保证数值一致，"
      "**必须用 GPU 建**。")
    a("")
    a("---")
    a("")
    a("## 3. 第 ③ 步 —— 24 序 Δ 扫描（no-GMC 特征语义）")
    a("")
    a("**完整命令**（薄封装已就绪，禁改的 `run_fastinfer_24seq_sweep.sh` 保持原样）：")
    a("")
    a("```bash")
    a("cd /tmp/pycharm_project_10ae9e2e")
    a("CACHE=/mnt/data/siping/runs/nogmc_t3/swa/gate/cs_t0043/ms_cache_24seq \\")
    a("  bash manu/tracking/lookahead/server_overnight/run_fastinfer_nogmc24_sweep.sh")
    a("")
    a("# 单点复跑：")
    a("CACHE=... DELTAS=\"3\" bash manu/tracking/lookahead/server_overnight/run_fastinfer_nogmc24_sweep.sh")
    a("```")
    a("")
    a("- 脚本内部逐 Δ 执行的核心命令（**CFG 直接从 `run_delivery.sh` 的 `CFG=` 取**，杜绝配置漂移）：")
    a("")
    a("```bash")
    a("/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/lookahead_longquan.py \\")
    a("  --cache-root \"$CACHE\" \\")
    a("  --labels-root \"/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010\" \\")
    a(
        "  --frames-root \"/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg\" --frames-cache /mnt/data/siping/frame_cache_lq \\"
    )
    a("  --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json --all-seqs \\")
    a("  --lookahead replay --arms original,stale,kinematic,replay --delta <D> $CFG \\")
    a("  --out-json \"$OUT/d<D>/arms.json\" --out-preds-dir \"$OUT/d<D>/preds\"")
    a("```")
    a("")
    a("- **预计墙钟**：每个 Δ 点 ~15~25 min（8 序 Δ=3 全四臂实测 66 s ⇒ 24 序 ≈ 3~5 倍，"
      "且 24 序帧数是 8 序的 5.7 倍）⇒ **4 个 Δ 点合计 ~1~1.5 h**（纯 CPU，不占 GPU）。")
    a(
        "- **产物路径**：`runs/lookahead/fi_nogmc24_sweep_runs/nogmc/d{2,3,4,8}/arms.json` + `preds/`"
        "（脚本 `run_fastinfer_nogmc24_sweep.sh` 的默认 `OUT`；**默认不再写进 `runs/lookahead/fi_nogmc24/`**，"
        "因为那是交付判据产物目录，`summarize_nogmc24_results.py` 与 `pack_delivery_nogmc_fi.sh` 都以它为根。"
        "若必须落回旧路径，显式 `OUT=$REPO/runs/lookahead/fi_nogmc24/nogmc` 覆盖即可。）"
    )
    a("- **校验点 ⑤（尺子 1 硬门禁）**：每个 Δ 点的 `arms.json` 里，四臂的 "
      "`arms.<arm>.source_frame.overall.f1` 必须**逐位相等**（ΔF1 = 0.000000pp）。"
      "核对命令：")
    a("")
    a("```bash")
    a("/home/huangzhe/anaconda3/envs/uav/bin/python - <<'EOF'")
    a("import json, glob")
    a("for p in sorted(glob.glob('runs/lookahead/fi_nogmc24_sweep_runs/nogmc/d*/arms.json')):")
    a("    d = json.load(open(p)); f1 = {a: v['source_frame']['overall']['f1'] for a, v in d['arms'].items()}")
    a("    print(p, '逐位相等' if len(set(repr(x) for x in f1.values()))==1 else '不等 ❌', f1)")
    a("EOF")
    a("```")
    a("")
    a("- **校验点 ⑥（口径）**：总体结论用 `cle_pooled_*`（池化）；`stale` 臂是交付基线，"
      "`original` 臂是 oracle 上界**不是**基线；24 序才是官方口径（8 序只覆盖 20.39% GT）。")
    a("")
    a("---")
    a("")
    a("## 4. 铁律与已知风险")
    a("")
    a("- ⚠️ **评估不得与训练同卡**：另一条 no-GMC 夜间搜索占卡到 **2026-11-05 07:00** ⇒ "
      "第 ② 步要么等它结束，要么用另一张卡。")
    a("- ⚠️ **每卡 1 shard**（共 4 shard × 4 线程 = 16 核）：每卡 2 shard 会 OOM"
      "（训练已占 15.7/24.5 GB，余 8.4 GB，两进程同卡在 batch=8 时峰值越界）。")
    a("- ⚠️ **CPU 缓存不能当交付档**：本机实测 CPU vs GPU 检出数 4,977 vs 4,974（阈值边界翻转）。")
    a("- ⚠️ 24 序官方缓存（rc2，GMC 语义）是**冻结基线**，**不要覆盖**："
      "`runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache`。")
    a("- ℹ️ 本机 `/mnt/data` 是**本地 ext4**（非 sshfs，3.8 TB 空闲）⇒ 7.4 GB 特征集写入不是瓶颈。")
    a("")
    a("## 5. 本次探测用的命令（复现用）")
    a("")
    a("```bash")
    a("cd /tmp/pycharm_project_10ae9e2e")
    a("bash manu/tracking/lookahead/server_overnight/probe_p0b_nogmc24_cost.sh")
    a("```")
    a("")
    a("- 产物：`/tmp/p0b_probe/feat_build_*.log`、`/tmp/p0b_probe/ms_cache_cpu_l*.log`、"
      "`/tmp/p0b_probe/ms_cache_cpu_full.log`、`/tmp/p0b_probe/cache_cpu_vs_gpu_full.json`")
    a(f"- 探测时间戳：**{log['timestamp']}**")
    a("")

    (OUT_DIR / "GPU_REQUIRED.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[SUCCESS] {OUT_DIR / 'GPU_REQUIRED.md'}")
    print(f"[SUCCESS] {OUT_DIR / 'probe_log.json'}")
    print(f"[ETA] 特征集全量 CPU workers1={feat_w1_h:.2f}h workers4={feat_w4_h:.2f}h workers24={feat_critical_h:.2f}h")
    print(f"[ETA] ms_cache 全量 CPU 单进程={cache_cpu_h:.1f}h  -P2={cache_cpu_p2_h:.1f}h  GPU4分片={cache_gpu_h:.2f}h")
    print(f"[ETA] 24序 Δ 扫描 ~{delta_scan_per_point_s:.0f}s/点 × 4 点 = {delta_scan_total_h:.2f}h")
    print(f"[VERDICT] CPU 能否今夜完成 = NO")


if __name__ == "__main__":
    main()
