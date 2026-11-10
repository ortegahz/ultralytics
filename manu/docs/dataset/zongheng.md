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

本域跑了**三档**，产物分目录存放、互不覆盖：

| 档位 | th | 后处理 | 用途 | 龙泉山实测 Recall / Prec / F1 / FAR |
|:--|:--|:--|:--|:--|
| **交付档** | **0.44** | 单帧 | 看**最终结论**、评估上线质量 | 90.74% / 97.65% / **94.07%** / **0.0176** |
| **敏感档** | **0.20** | 单帧 | 人工评估 / 标注播种（弱目标也捞出来） | **93.91%** / 90.13% / 91.98% / 0.0830 |
| **时空轨迹档** | **0.20** | **龙泉山时空轨迹** | 参照龙泉山业务域处理，断续检出连成轨迹 | 同上阈值，后处理另算 |

🔴 **0.20 档不能当交付档**：FAR 0.0830 是 0.44 档的 4.7 倍，已越过业务域交付上限 0.030。
**交付结论仍以 0.44 档为准**；0.20 档只是复核/预标注的工作档。

### 2.1 时空轨迹档的后处理链

直接复用龙泉山代码，不重写：

```
多尺度逐尺度行 → 跨尺度合并(r=768) → OnlineAdaptiveTracker（双向跟踪，天/地区分阈值）
→ BidirectionalTemporalSmoother（拼接 / 插补 / 刚性静态剪枝 / 悬停锁定）
→ 连续帧确认(confirm_consec=7) → 每个最终系统检出画一个红圈 + T<id>:<score>
```

轨迹参数**从龙泉山 RC2 交付点 v3 文件读回**
（`manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`），不复制到本仓库，
避免与交付臂漂移。

🔴 **本域无 GT，所以有两个坑**（都已在代码里处理）：

1. **跟踪器不需要 GT**（GT 只用于打分与补齐空帧），但龙泉山的 `make_records` 用
   「候选帧 ∪ GT 帧」定帧集 ⇒ 无 GT 时**空帧被整帧删除** ⇒ tracker 的帧索引被压扁 ⇒
   `max_age` / `min_hits` / `confirm_consec` 全部失真。
   ⇒ 本实现按 cache meta 的 `frames` **补齐完整时间轴**。
2. **刚性静态剪枝会杀掉静止目标**（净位移 < 2 px 且位置方差小 ⇒ 判为「冻结坏点/静态反光」）。
   龙泉山本意就是去掉静态反光，但**静止悬停的真目标**也会被同样剪掉。
   ⇒ 最终清单会单独列出「跟踪后 0 轨迹」的序列，供人工判断。

### 2.2 时空轨迹档 vs 单帧档：覆盖率会**降**，这是设计使然

同为 th=0.20 逐序列对比（26 条）：**21 条覆盖率下降、0 条上升**，最大 `VIDEO00033` −10.2 pp。

| 归因（按检出段长度） | 帧数 | 占比 |
|---|---|---|
| 孤立闪烁 run=1 | 439 | 13.9% |
| 短段 run 2~6（未达 confirm_consec=7） | 1,400 | 44.3% |
| **持续段 run ≥ 7** | **1,324** | **41.9%** ⚠️ 真的目标也被删 |

🔴 **但这不等于跟踪变差**：龙泉山**有 GT** 的实测是
**跟踪臂 F1 94.747% vs 单帧臂 F1 87.070%（+7.68 pp）** ⇒ 在有 GT 的域上，被删的帧**大多是虚警**，删掉是净赚。
本域**无 GT**，同样的删除**无法验证**是真目标还是虚警。

⇒ **用法定位**：
* **时空轨迹档** = 降虚警的**复核视图**（看最终系统结论，画面干净）；
* **th=0.20 单帧档** = **标注播种**（宁可多给候选，人来删）——**播种以它为准**。

### 2.3 覆盖率损失的三级归因（更精确、可执行）

上表按「检出段长度」分桶，是**现象**层归因。按**机制**重新拆一次（`diag_zongheng_fast_movers.py`，
门限从龙泉山 v3 参数文件读回，**全 58 序列 / 173,781 帧**实测）：

| 机制 | 帧数 | 占比 | 性质 |
|---|---:|---:|:--|
| 短段（run < `confirm_consec=7`） | 3,470 | **58.2%** | ✅ 设计意图（滤孤立闪烁） |
| **关联门限受限**（median step > `max_match_dist`=24 px） | 126 | 2.1% | ⚠️ **域差**，见下 |
| 残差（可关联长段仍丢帧） | 2,369 | 39.7% | ✅ 刚性静态剪枝（净位移 ≤2 px 的固定结构） |

总计删 **5,965 帧**（单帧档 102,139 命中帧 → 跟踪档 **96,174**，占 5.84%）。

🔴 **根因：龙泉山跟踪器的关联门限有硬上限 `max_match_dist = 24 px`**
（`eval_bidirectional_track_fusion.py:214`：`cur_gate = min(max_match_dist, match_dist + 0.5*|v|)`）。
**每帧位移 > 24 px 的目标永远关联不上** ⇒ 每帧新建轨迹 ⇒ 达不到 `confirm_consec=7` ⇒ **整段被删**。
本域实测 **11 段**这种目标（每帧 25~460 px，0.3~0.4 s 横穿画面）：
`VIDEO00036:785-799` / `VIDEO00036:801-807` / `VIDEO00043:1664-1671` / `VIDEO00044:16-23` /
`VIDEO00044:910-917` / `VIDEO00047:2830-2842` / `VIDEO00058:69-75` / `VIDEO00073:354-361` /
`VIDEO00074:41-48` / `VIDEO00084:1301-1311` / `VIDEO00086:2482-2514`
⇒ **这是龙泉山口径与纵横AI域的域差，不是 bug**（龙泉山目标的帧间位移在 24 px 以内）。

🔴 **刚性静态剪枝的 66 段 / 1,197 帧，净位移全部 ≤ 2 px**，抽帧目视（OSD 的 ZOOM 面板）确认
**全是固定人工结构**：`VIDEO00046:441` 与 `VIDEO00060` = **T 形灯杆/横臂**，`VIDEO00033:1792` = 亮树冠边缘
⇒ **这些删除是正确的降虚警，不是误删**。

「跟踪后 0 轨迹」只有 2 条：

| 序列 | 单帧档 | 跟踪档 | 判定 |
|---|---:|---:|:--|
| `VIDEO00060_19700101_011137` | **243/243 帧（100%）** | 0 | **灯杆，剪对**（ZOOM 面板目视确认） |
| `VIDEO00057_19700101_010013` | 2/449 帧（score 0.212） | 0 | 可忽略（刚过阈值的两帧闪烁） |

### 2.4 三档共用

| 项 | 值 | 依据 |
|:--|:--|:--|
| 模型 | **RC2 = `runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt`** | 三轨门禁交杖的精度冠军（T1 0.901910 / T2 0.998949 / T3 0.897694） |
| P0 主干 | **`use_p0_highway=False`**（等价 `--ablate-p0`） | RC2 的门禁跑法与龙泉山缓存构建**都**用它；用 `on` 会静默建出另一张图 |
| 结构自描述 | `head_conv=64`、`late_upsample=False` 走 ckpt 自带字段 | 漏传就重建出另一张图（§80「hc32 死模型」事故根因） |
| 尺度 | native 640 + 160 + 80 | 与龙泉山 RC2 缓存一致 |
| 跨尺度合并半径 | **768 px**（原生像素） | 实测最优；128⇒93.923%、64⇒93.857% 都更差 |
| 跟踪 | 前两档**无**（每帧独立打分）；第三档**有** | 用户要求 |

工作点是**测出来的**，不是拍的（`manu/evaluation/probe_rc2_threshold_table.py`，CPU-only，
复用交付评估代码 ⇒ 数值与交付臂一致）。完整档位对照：

| th | Recall | Prec | F1 | FAR/帧 |
|---|---|---|---|---|
| 0.16 | 94.31% | 87.72% | 90.90% | 0.1065 |
| 0.18 | 94.12% | 89.03% | 91.50% | 0.0936 |
| **0.20** | **93.91%** | 90.13% | 91.98% | 0.0830 |
| 0.22 | 93.69% | 91.07% | 92.36% | 0.0741 |
| 0.25 | 93.36% | 92.35% | 92.85% | 0.0624 |
| 0.30 | 92.76% | 94.35% | 93.55% | 0.0448 |
| 0.35 | 92.04% | 95.84% | 93.90% | 0.0322 |
| 0.40 | 91.33% | 96.91% | 94.04% | 0.0235 |
| **0.44** | 90.74% | 97.65% | **94.07%** | 0.0176 |

⚠️ **旧 OSD 用的不是这个口径**：2026-10-12 那批 `纵横AI数据采集_osd` 是 trial_0474 + `--threshold 0.30 --merge-radius 64`
⇒ **新旧 OSD 不可逐帧比对**。

## 3. 产物

| 产物 | 路径 | 说明 |
|:--|:--|:--|
| **交付档 OSD**（th=0.44，单帧） | `/mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_20261109/` | 58 条，1920×1080，左主画面 + 右 48px 局部放大 |
| **敏感档 OSD**（th=0.20，单帧） | `/mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_th020_20261109/` | 同上，**召回更高、虚警更多** |
| **时空轨迹档 OSD**（th=0.20，带跟踪） | `/mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_tracked_th020_20261110/` | 同上，画的是**最终系统检出**（红圈 + `T<id>:<score>`，插补帧标 `Infill`） |
| 每帧检测记录 | 各自目录下的 `frames/<SEQ>.jsonl` | 每帧一行：`{sequence, video, frame, detections:[{cx, cy, w, h, score}]}`（跟踪档只有 cx/cy/score + track_id），**原生 640×512 像素** |
| 分片统计 | 各自目录下的 `summary_shard{0..3}.json` | 每条序列的帧数 / 检出帧数 / 检出数 |
| 序列划分 | `manu/data/zongheng_ft_split{,_th020,_tracked}.json` | 见 §4 |
| 人工复核底稿 | `manu/docs/dataset/zongheng-tk{,-th020,-tracked}.md` | `# pass` / `# unpass` 格式，同 `long-quan-tk.md` |
| 标注候选 | `.../纵横AI数据采集_preannot_rc2{,_th020,_tracked}` | web 标注器原生布局 `<SEQ>/<帧号>.txt` + `<SEQ>/scores.json` |

**旧 OSD `/mnt/data/siping/datasets/manu/纵横AI数据采集_osd`（trial_0474 时代）保持不动**，新产物一律进新目录。

### 3.1 三档最终结果（2026-11-10）

| 档位 | OSD | 检出帧 / 总帧 | 覆盖率 | 划分 val/train/ignore | 预标注数 |
|:--|:--|:--|--:|:--|--:|
| 交付档 th=0.44 单帧 | 58/58 | 86,595 / 173,781 | 49.8% | 26 / 29 / 3 | 86,595 |
| 敏感档 th=0.20 单帧 | 58/58 | 102,139 / 173,781 | 58.8% | 31 / 25 / 2 | 102,139 |
| **时空轨迹档 th=0.20 + 跟踪** | **58/58** | **96,174 / 173,781** | **55.3%** | **30 / 24 / 4** | **96,174** |

时空轨迹档的验收（`manu/videos/verify_rc2_tracked_delivery.sh`）= **ALL GREEN，failures=0**：
58 mp4 / 58 frames / 58 preds / 58 cache / 4 summary 全部齐；4 个 summary 的工作点
均为 `threshold=0.2 / merge_radius=768.0 / tracking=YES / confirm_consec=7`；
划分清单读回 `threshold=0.2 / tracking=True`；复核底稿档位行为「**时空轨迹档**」。

🔴 **坐标空间必须对齐**：每帧记录在**原生 640×512** 网格上，而 web 标注器显示的是**原始 1920×1080** 视频。
直接写原生坐标会让每个候选整体偏移（2 倍 + 320px）。预标注导出器默认 `--frame-space full`
（`full_x = crop_left + native_x * crop_width / native_width`），已用真实视频逐帧比对验证。
若标注器扫的是**裁剪后的帧目录**，则要改用 `--frame-space native`。

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

### 6.1 怎么把标注器挂上这个域

web 标注器既可以扫 mp4（`VideoSequence`），也可以扫帧目录（`ImageDirSequence`）：

* **扫 mp4**（推荐，零抽帧开销）⇒ 标注坐标在 **1920×1080** ⇒ 预标注用 `--frame-space full`（默认）。
  注意标注器 worker 自己走的是**另一条口径**（`worker_infer.py` 不做裁剪、也不做跨尺度合并），
  所以**本域的预标注用本仓库的 RC2 记录导出**，不要点标注器自带的「预标注」。
* **扫裁剪后的帧目录**（`extract_video_frames.py --restore-native`）⇒ 坐标在 **640×512** ⇒ 预标注用
  `--frame-space native`。

启动示例：

```bash
python -m manu.annotate.server \
  --root /mnt/data/siping/datasets/manu/纵横AI数据采集_mp4 \
  --workspace /home/manu/mnt/pycharm_project_10ae9e2e/manu/annotate_workspace \
  --port 8777
```

🔴 **预标注树要放进 workspace 自己的 `prelabels/`，不要只挂 `--prelabel-source`**：

| 放法 | 候选框 | 分数 / 阈值滑块 |
|:--|:--|:--|
| 复制到 `<workspace>/prelabels/<SEQ>/` | ✅ | ✅（读 `<workspace>/prelabels/<SEQ>/scores.json`） |
| 只 `--prelabel-source <tree>` | ✅ | ❌（外部树无分数 ⇒ 一律「始终显示」，滑块失效） |

⇒ 推荐：
```bash
cp -r /mnt/data/siping/datasets/manu/纵横AI数据采集_preannot_rc2/* \
      /home/manu/mnt/pycharm_project_10ae9e2e/manu/annotate_workspace/prelabels/
```
（workspace 必须在 SSHFS 内，服务器才看得到。）

已实测：记录 `cx=545.05, cy=373.47, w=11, h=8, score=0.491`
⇒ 标签 `0.734427 0.691611 0.011458 0.014815`（1920×1080 归一化）⇒ 服务端返回同一条框 + `scores:[0.491]`。

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
