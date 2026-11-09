# 纵横AI数据采集 — 交付域档案

> 本文件是「纵横AI数据采集」域的**入口文档**。数据本身**没有 GT**，
> 本域的一切指标都来自 **RC2 的单帧多尺度检测**（不是标注精度）。
> 详细证据见 `manu/memory/memory.md` §91。

## 1. 域事实

| 项 | 值 |
|:--|:--|
| 帧源 | `/mnt/data/siping/datasets/manu/纵横AI数据采集_mp4` |
| 序列数 | **58** 个 mp4 |
| 原始画幅 | 1920×1080 |
| 裁剪契约 | `--crop-left 320 --crop-top 0 --crop-width 1280 --crop-height 1024` → 原生 **640×512** |
| 特征 | 内存现算：GMC（逐 lag 直接拟合）+ 21 帧中值，通道序 `[I_t, GMC diff, median residual]`，推理时反转为 `[median_residual, gmc_diff, I_t]` |
| GT | **无**（本域尚未标注） |

🔴 **裁剪后原生画幅与龙泉山完全一致（640×512）** ⇒ 龙泉山 RC2 实测出来的工作点可以**原样搬过来**，
不需要为这个域重新扫阈值。

## 2. 模型与工作点

| 项 | 值 | 依据 |
|:--|:--|:--|
| 模型 | **RC2 = `runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt`** | 三轨门禁交杖的精度冠军（T1 0.901910 / T2 0.998949 / T3 0.897694） |
| P0 主干 | **`use_p0_highway=False`**（等价 `--ablate-p0`） | RC2 的门禁跑法与龙泉山缓存构建**都**用它；用 `on` 会静默建出另一张图 |
| 结构自描述 | `head_conv=64`、`late_upsample=False` 走 ckpt 自带字段 | 漏传就重建出另一张图（§80「hc32 死模型」事故根因） |
| 尺度 | native 640 + 160 + 80 | 与龙泉山 RC2 缓存一致 |
| 阈值 | **th = 0.44** | **实测**于真实龙泉山 RC2 缓存 |
| 跨尺度合并半径 | **768 px**（原生像素） | 同上；768px 在 640×512 上 ≈ 每帧最多一个检测点 |
| 跟踪 | **无**（每帧独立打分） | 用户要求 |

工作点是**测出来的**，不是拍的（`manu/evaluation/eval_multiscale_cache_th.py`，CPU-only）：

| merge_radius (px) | 最优 th | F1 | Recall | Precision | FAR/帧 |
|---|---|---|---|---|---|
| **768** | **0.44** | **94.069%** | 90.74% | 97.65% | 0.0176 |
| 128 | 0.44 | 93.923% | 90.77% | 97.30% | 0.0203 |
| 64 | 0.44 | 93.857% | 90.82% | 97.11% | 0.0218 |

⚠️ **旧 OSD 用的不是这个口径**：2026-10-12 那批 `纵横AI数据采集_osd` 是 trial_0474 + `--threshold 0.30 --merge-radius 64`
⇒ **新旧 OSD 不可逐帧比对**。

## 3. 产物

| 产物 | 路径 | 说明 |
|:--|:--|:--|
| 复核视频 | `/mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_20261109/<SEQ>_osd.mp4` | 58 条，1920×1080，左主画面 + 右 48px 局部放大 |
| 每帧检测记录 | `.../纵横AI数据采集_osd_rc2_20261109/frames/<SEQ>.jsonl` | 每帧一行：`{sequence, video, frame, detections:[{cx, cy, score}]}`，原生像素 |
| 分片统计 | `.../summary_shard{0..3}.json` | 每条序列的帧数 / 检出帧数 / 检出数 / 首末检出帧 |
| 序列划分 | `manu/data/zongheng_ft_split.json` | 见 §4 |
| 人工复核底稿 | `manu/docs/dataset/zongheng-tk.md` | `# pass` / `# unpass` 格式，同 `long-quan-tk.md` |

**旧 OSD `/mnt/data/siping/datasets/manu/纵横AI数据采集_osd`（trial_0474 时代）保持不动**，新产物一律进新目录。

## 4. 序列划分的口径

🔴 **本域没有 GT ⇒ 划分不是质量判定**，它只记录「模型看到了什么」，最终取舍由人工复核拍板。

| 桶 | 判据 | role | 含义 |
|:--|:--|:--|:--|
| `val` | 检测覆盖 **≥ 50%** | `golden_pool` | 稳定正样本候选，本域黄金池 |
| `train` | 检测覆盖 **2% ~ 50%** | `core_positive_mining` | 间断检出，提供正负切片养料 |
| `ignore` | 检测覆盖 **< 2%** | — | 无 GT ⇒ **无法区分空场景与漏检**，先隔离 |

字段名**逐字沿用** `manu/data/longquan_ft_split.json`（SOP 明令不自创字段名）：
顶层 9 键（`version` / `created` / `author` / `purpose` / `rationale` / `sampling` / `feature_spec` / `groups` / `baseline_reference`）、
`groups.{val,train,ignore}`、val/train 条目带 `role` + `note`、ignore 条目只有 `sequence` + `reason`。
`single_frame_f1` **标注完成后回填**（不另起字段名）。

## 5. 人工复核怎么做

1. 看 `.../纵横AI数据采集_osd_rc2_20261109/<SEQ>_osd.mp4`：红点 = RC2 检出的中心点。
2. 判「红点是否落在真实目标上」—— 漏检（该有目标却没点）/ 虚警（点在背景上）/ 位置偏移（点在目标边上）。
3. 把结论写进 `manu/docs/dataset/zongheng-tk.md`：一行一条，`[pass]|[ignore] <SEQ>_F1-<val>.mp4 - <原因>`。
4. 定级后再回填 `zongheng_ft_split.json` 的 `single_frame_f1`，并据此调整 val/train/ignore 三桶。

## 6. 后续（标注与训练）

* 标注落 `labels/<seq_id>/<frame:06d>.txt`（annotator 原生布局，类别恒为 0，像素归一化 6 位小数）；
  交付/评测前用 `python -m manu.annotate.export_labels` 转扁平 `<seq>__<idx>.txt`。
* 训练集构建用 `manu/data/build_longquan_ft_dataset.py --split-config manu/data/zongheng_ft_split.json`
  （`ignore` 桶不构建）。

## 7. 复现命令

```bash
# 全量渲染（服务器，4 卡分片，detached）
bash manu/videos/launch_rc2_zongheng_osd.sh

# 序列划分（可从半成品出：记录按视频增量落盘）
python manu/data/build_zongheng_ft_split.py \
  --frames-dir /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_20261109/frames \
  --out manu/data/zongheng_ft_split.json \
  --review-sheet manu/docs/dataset/zongheng-tk.md
```
