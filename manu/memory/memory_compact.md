# 红外弱小无人机检测项目精简背景

> 用途：向外部高性能模型提供项目上下文。详细实验证据以 `manu/memory/memory.md` 及 `manu/memory/` 专题文档为准。
> 更新时间：**2026-10-20**（新增 **§26 微调前基线冻结**：**三条全部单帧臂**（`--no-pipeline`）—— Anti-UAV 0.906361 / 黄金池 7 条 F1 98.788% / 盲测 `001324` F1 83.479%，三线零漂移；**§0 业务域快照已按 `memory.md` §三十九 更新为 F1 90.605%、三条交付判据全过**）
>
> ⚠️ **本文件编号说明**：`## 11/12/13` 各出现两次（第 427/462/472 行与第 492/506/520 行）。
> 前一组＝§11 Anti-UAV val 子集 / §12 板卡访问免密 / §13 NPU 段耗时归因；
> 后一组＝无损 PNG 参考基线 / 官方特征构建脚本恢复 / 本轮收口总结。
> 本文件内部引用「§12」「§13」一律指**后一组**（唯一例外：第 466 行的「原 §11」指前一组）。
> ⚠️ `memory.md` 有**独立编号体系**（一…二十六 + 子节），其「§12/§13」不与本文件对应。

## 0. 当前工作状态快照（截至 2026-10-10，详见 memory.md 零之二）

- **算法主线已冻结收口**：Trial 0474 单帧 F1 0.9064、系统级 In-BBox F1 0.9209，不再改模型。
- **业务域交付（2026-10-20 已更新，取代 2026-10-16 旧数字）**：GT `Final_Labels_merged_20261010`（correct 之后版本，去重后 **GT 分母 133,257**）。🔥 **系统级（SOTA 后处理已接）24 序列聚合：F1 90.605% / R 85.881% / P 95.879% / TP 114,443 / FP 4,919 / FN 18,814 / FAR 0.02977** ⇒ **三条交付判据首次全部达标**（F1≥88 ✅ R≥85 ✅ FAR≤0.030 ✅，FAR 只剩 0.8% 余量）。单一全局工作点 34 参数，机制 = `dedup_radius=768`（本域每帧只有一个目标 ⇒ 每帧只留最高分 1 个候选）+ `confirm_consec=6` 连续确认门（FP −55.5%）。⚠️ 旧**单帧候选池**口径 F1 87.070% / FAR 0.0632 仍是同批数据的另一口径，引用时必须写明是哪一档。完整证据见 `memory.md` **§三十九**。
- 🔥 **微调前基线已冻结（2026-10-20，见本文 §26）**：**三条全部是单帧臂**（`--no-pipeline`）——Anti-UAV **0.906361 / TP 21,643 / FP 1,004**（15 档门限曲线同时冻结）、Val_Golden_Pool 7 条 **F1 98.788% / TP 16,264 / FP 25**、盲测 `VIDEO00007_19700101_001324` **F1 83.479% / TP 7,981 / FP 605**（`fn_compete`=0 ⇒ 漏检全是「帧内无检出」，后处理救不了 ⇒ **是前端检测器瓶颈**）。Replay 微调**尚未启动**，方案在裁决中。
- 🔥 **业务域已定一条新方案（2026-10-13，代码未实现、零实测）**：`Trial 0474 双专家 MoE Adapter`
  （Expert_Large 大目标补盲 ~12k + Expert_Suppress 路灯压制 ~15k，底模 100% 冻结、确定性融合
  `H_final=(H_base+Δ_pos)×(1-M_gate)`、~30k 参数、留 `--disable-adapter` 逃生开关）。
  详见 §9 与 `dual_expert_moe_adapter.md`。⚠️ **实施前须先裁决 3 个硬冲突**（缓存体积踩铁律二、缺铁律六零初始化门控、挖掘口径≠评测口径）。
- 🔥 **嵌入式时延已从「调研」推进到「性能收口」**：
  - **GMC fit 板卡 18.46 → 2.65 ms（6.96×），端到端 → 128.8 fps，全程位精确（MD5 不变）**。
    关键认知：①「半分辨率」本就是冻结默认，不是收益；②实测 **gFTT 占 fit 65.8%、金字塔仅 3.2%**，
    真杠杆是**按绝对帧号缓存每帧派生量**；③**多线程在本工程位精确**（证伪旧铁律）；
    ④自写标量 `median_residual` 线程池碰不到，单独并行化后再提速 2.1~2.4×。
  - 原「时延口径 5 倍分歧」已裁决：唯一权威口径 = `--timing` 分段表 + 板卡实测。
  - 🔥 **NPU 已跑通（2026-10-14 更新，本条取代此前「驱动未使能」的记载）**：RK3588 NPU 节点为
    **`/dev/dri/card1`**（驱动 0.9.8、runtime api 2.3.0），**governor = performance、cur_freq 满频 1 GHz**，
    该频率必须写进任何时延报告。热图检测器已从 152.4 ms / 6.6 fps 优化到 **56.7 ms / 17.7 fps（2.69×，仅 NPU 阶段）**
    且 **0 个 CPU 计算算子**；⚠️ **全链路端到端只有 88.852 ms / 11.25 fps（512×512，预处理 31.9 + NPU 56.9）**，
    且预处理时延随分辨率近似线性增长 ⇒ **实时帧率不得只引 NPU 阶段数字**（见 §6 与 `heatmap_npu_trial0474.md`）。
  - 🔴 **RK3588 移植参考模型（ref model）已裁决 = 「去 P0 Highway」的模型**（2026-10-14 用户裁决）：
    精度代价 ΔF1 仅 **−0.000625**（CPU fp32 全量 31,613 帧实测，0.906361 → 0.905736），
    换板卡 `rknn_run` **55.5 ms** + **0 个 CPU 计算算子/0 条 fallback**。
    ⚠️ 原记的 **151.7 ms / 6.5 fps 是带 P0 的 v1 档，已作废为历史档**。详见 §10 与 `heatmap_npu_trial0474.md` §六之二。
  - ⚠️ **GMC fit 的收益强依赖纹理**，引用加速比须同时报点数。

## 1. 项目目标

面向低空安全场景，研究复杂背景下 1~3 像素红外弱小无人机检测。核心评价集为 Anti-UAV 红外数据，采用防数据泄漏的场景划分和全量统一评测。

统一评测规模：24 个验证序列、31,613 帧、25,111 个 GT，匹配标准为 Distance <= 8.0 px。

## 2. 当前最高基线

### 单帧模型：Trial 0474

- 权重：`runs/optuna_p0_nas/trial_0474/weights/best.pt`
- 架构：YOLO26 Heatmap Detector，P0-NAS 微架构 `standard_dw + pixel_unshuffle + diff_only (depth=2)`
- 输入契约：`[I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]`
- 其中 GMC 在原始灰度帧上估计运动并对齐历史帧，`B_t` 为 21 帧时域中值背景
- 单帧指标：Recall 86.19%、Precision 95.57%、F1 0.9064、TP 21,643、FP 1,004、FAR 0.0318/帧
- Trial 0474 是当前永久冻结的单帧基线
- 🔴 **复现路径（2026-10-14 实测更正）**：`val_heatmap_resolution.py` **无法复现**（不传 `use_p0_highway` / `p0_highway_kwargs`）。
  正确路径：`manu/inference/cache_trial0474_inferences.py --conf-thresh 0.08 --top-k 80 --stride 2 --imgsz 640`
  → `manu/evaluation/eval_bidirectional_track_fusion.py --match-mode dist --dist-thresh 8.0 --th-base 0.25`。
  ✅ 已零漂移复现：**TP 21,643 / FP 1,004 / GT 25,111 三项逐个对齐**，F1 **0.906361**（= `results.csv` epoch 3 冻结标称）。
- **P0 Highway 消融**：摘除后 F1 **0.905736**，**ΔF1 仅 +0.0006**；P0 用 **4,418 参数（0.25%）**换边缘收益，
  **不足以支撑「P0 显著有效」的结论**（跨 15 档门限均值 +0.00033、2/15 档为负）。详见 §10。

### 系统级方案

Trial 0474 + 双向时空平滑 + 碎片缝合 + 高确信插补 + 刚性坏点剪枝：

- In-BBox：Recall 90.22%、Precision 94.05%、F1 0.9209、TP 22,654、FP 1,434
- 严格距离口径：F1 约 0.9194
- 交付参数：`th_base=0.22`、`th_salvage=0.06`、`th_ground=0.35`、`stitch_gap=4`、`infill_gap=3`、`min_hits_infill=5`、`min_rigid_disp=2.0`、`max_rigid_var=0.50`、`min_hits_prune=8`

## 3. 关键演进

1. 公司数据摸底后确认场景多样性和极小目标样本不足，转向公开 Anti-UAV 红外数据。
2. 建立 5-Fold 防泄漏划分和统一 24 序列评测大盘。
3. YOLO26-P2 Bbox 基线 `trial_0028`：F1 0.7631。
4. Anchor-free 高分辨率 Heatmap 范式 `uav_gpu23_heatmap`：F1 0.8331。
5. GMC + 21 帧中值背景建模显著提升输入质量，形成 Trial 22：F1 0.9057。
6. P0-NAS 搜索产出 Trial 0474：单帧 F1 0.9064。
7. 双向时空后处理和刚性坏点剪枝将系统级 F1 提升至 0.9209。
8. 自定义 DataLoader 曾造成样本腰斩和指标漂移，已修复并由官方 DataLoader 零漂移复现。

## 4. 四项代表性失败实验

1. **Tubelet 时序注意力**：空间坐标未对齐，历史目标能量无法注入；F1=0.8954，门控收敛为微负。
2. **浅层多尺度时序 NAS**：58 组搜索均停留在 F1=0.9063~0.9064，未超过 Trial 0474，说明短时局部卷积已触及信息增益天花板。
3. **点流 DP-TBD**：48 组搜索最佳 F1=0.9057；低阈值候选中的白噪声被时空串联，FP 增至 2,369，低于系统基线。
4. **亚像素积分能量守恒 Focal Loss**：单帧 Recall 短暂升至 86.32%，但破坏热图峰值曲率和时间连续性，系统级净丢 223 个真值，正式停止检测头损失微调。

其他已证伪路线：原始 3-Frame 堆叠、Hybrid Corr、960 免训练跨尺度推理、IRSTD-UNet 从零训练、PixelShuffle 直接微调、Top-Hat 零训练推理、Soft-IoU、TTA 热图平均、全参数高斯松弛微调、形态学侧支无差别微调。

## 5. 当前技术定论

- 1~3 像素点目标更适合高分辨率 Heatmap 点回归，不适合传统 Bbox 回归。
- GMC 必须在原始灰度图上估计，不能在差分图上估计。
- 输入管道永久锁定三通道语义，不得擅自改变通道顺序或破坏灰度直流基底。
- 训练和推理尺度必须物理对齐，不能直接免训练放大推理。
- 单帧模型和底模 Trial 0474 已冻结；后续突破优先从系统级时空物理后处理和候选池分析入手。
- 任何重大非继承性改动必须执行同源 Twin A/B 冷启动受控协议。
- 任何侧支实验必须底模 100% 冻结、零初始化恒等门控，并先完成 Epoch 0 零回归校验。
- 任何新实验必须使用官方 DataLoader，训练样本 43,008、验证样本 31,613，且基线需浮点级复现。
- 涉及混合分辨率数据时，原生图像坐标和 640x640 Letterbox 坐标严禁混用；不得硬编码 `imgsz=640` 进行原生坐标采样。
- 修改已有多通道数据时，已有通道必须读取官方数据集图像，禁止从 raw 重新生成导致基线漂移。

## 6. 领域迁移 Bad Case（当前业务数据发现）

以下 8 类问题是业务域落地中的核心失效模式；所列工程解法除已有输入契约外均需 A/B 和定量验证，不能直接视为已证实收益。

1. 黑热模式：`(I_t-B_t)^+` 截断暗目标响应。优先在原始灰度帧级执行 `I'=255-I`，再重新生成 GMC+21 帧中值特征；不能只反转已有通道或 HM。无元数据时动态极性识别仍待验证。
2. 悬停目标：超过 21 帧后可能被中值背景吸收。候选方案是高置信航迹驻留保护和预测位置局部背景冻结，必须防止固定坏点被锁定。
3. 近距离大目标 Heatmap 多峰：机体热部件产生多个峰。候选方案是按空间距离和连通能量聚类，以加权质心合并。
4. 近距离大目标 Bbox 漏检：差分自抵消和辐射纹理断裂。候选方案是原始灰度 320x320 Bbox 旁路，或 Track 层按同速和相对距离稳定性组装点目标。
5. NUC 温漂与固定条纹：FPN 破坏 GMC 和中值背景。候选方案是轻量列去条纹和适度 CLAHE，需验证不破坏原域基线。
6. 镜头水雾/油污：光学散射使峰值降低。候选方案是降低双门限并提高轨迹连续性约束，禁止单独降门限。
7. 吊舱大动态机动：GMC 在大视角转动、变焦和视差下失效。候选方案是按残差 MSE/内点率健康度门控，异常时静音差分和中值通道，降级纯灰度输入。
8. 野外复杂热杂波：树木、岩石和地表热斑产生持续伪峰。候选方案是按净位移/总路程比 `R` 做曲折度剪枝，并结合刚性位移约束。
9. 下沉入地/入城：从天空进入建筑、道路和高温地表后，局部背景方差暴增、GMC亚像素残差在强边缘产生 `(I_t-B_t)^+` 毛刺，目标可能被压低、粘连或使航迹中断。**首要任务是确认临界段热图/Ch1/Ch2属于完全无峰、低于 `th_base=0.22` 的残存峰，还是被边缘峰吸引**；候选方向为已确认航迹的地表ROI软保护、边缘感知残差抑制、受控运动管道积分和GMC异常降级，均待A/B验证。新增业务分桶指标：Recall、FAR、连续漏检长度、航迹中断率、恢复延迟、误保护率。

迁移验证统一保留 Trial 0474 对照，统计 Recall、Precision、FAR、连续漏检长度和轨迹误删率，~~并通过 Anti-UAV 原始验证集不可退化闸门。~~ 🔴 **该闸门已于 2026-10-13 废止为硬约束**（见 §9.5），Anti-UAV 降级为防崩底板但**仍须每轮测量并如实上报**。

### 新交付域「纵横AI数据采集」58 条视频（2026-10-12）

- 数据：`/mnt/data/siping/datasets/manu/纵横AI数据采集_mp4`，58 个 mp4 / 1920×1080 / 25fps / 174k 帧，**无 GT**，长序列约 3150 帧，短序列 243~1741 帧。
- 🔴 **画幅契约与龙泉山不同**：letterbox 内容框实测 **x[320,1599] y[0,1023]**（ffprobe + 逐帧投影，58/58 一致）⇒ 还原参数 `--crop-left 320 --crop-top 0 --crop-width 1280 --crop-height 1024` → native 640×512；**龙泉山是 `--crop-top 28`，静默沿用会整体错位**。
- 新增 `manu/videos/render_multiscale_osd_video.py`：直接吃 mp4 流式出 OSD 视频，**不落帧、不落 pkl**；复用 `multiscale_heatmap_cache.py` 与 `render_pred_gt_osd_video.py` 的既有算子，**推理链与评测大盘同一份代码**。默认工作点同业务域评测：`--scales 160,80 --threshold 0.30 --merge-radius 64 --region-threshold 0.06`。`--shard i/n` 多卡、`--skip-existing` 续跑、`--out-json` 落每视频 `frames_with_detections / first_detected_frame / last_detected_frame`。
- 🔴 **`--start` 只门控渲染、不门控 GMC 历史**（早于 start 的帧仍喂 GMC 与 21 帧中值窗口），避免渲染帧坐在冷历史上——与 tree-GMC A/B「前 22 帧全同」是同一类陷阱。

### 新增定性结果：近距离大目标直接 160 下采样

- 原始 640x512 三通道特征直接推理时，近距离大目标存在热图能量摊平、峰值低于 `th_base=0.22` 的尺度失配问题。
- “下采样再放大回原尺寸”没有明显收益；改为整张 `[I_t, GMC_Diff, Median_Residual]` 同步缩小到 160 px 后，直接通道感知 Padding 到 640x640 输入 Trial 0474。
- Ch0 用边缘灰度均值填充，Ch1/Ch2 严格填 0，避免通道语义错误和 Padding 阶跃边缘。
- `VIDEO00005_19700101_002959` 视频中近距离无人机出现明显、稳定的定性检出，支持“尺度/感受野失配”假设；目前没有 GT 定量指标，暂不能宣称 Recall/F1 提升。
- 当前定位：近距离大目标 P1 候选旁路，需与原始分支做人工 GT A/B，统计检出率、连续漏检长度、FAR、中心偏差和峰值响应。

### Final_Labels 人工终审标注集（2026-10-09）

- 交付标注已升级为 `龙泉山/label/Final_Labels`，**取代** `pre_label/part1 + part2` 试标注。
- **扁平目录，无 part1/part2**：单层 `{sequence}__{index:06d}.txt`，共 **165,223 个**标签 + `classes.txt`（仅 `airplane`，单类 class 0），**24 个序列**，索引 `0..N-1` 连续，与 `frames_ir_jpg/<seq>/frame_%06d.jpg` 一一对应。
- 非空标签 **132,570（80.2%）**，格式 `class cx cy w h`，归一化到原生 640×512，class id 全为 0。
- ⚠️ **数据缺陷：132,592 行 vs 132,570 非空文件 ⇒ 22 个文件含完全重复行**（例 `VIDEO00002_19700101_001415__009082.txt`）；且**全库无任何一帧存在两个不同目标**。按行数统计目标数前必须去重，否则虚高。
- 新增 `manu/videos/render_gt_osd_video.py`：读扁平 Final_Labels + frames_ir_jpg，按 `--seq` 渲染「全帧 OSD + 局部放大」双联 GT 复核视频；`--list` 列出全部 24 序列；内置去重并打印 `duplicate_rows_dropped`；选中范围内标签找不到帧即 `RuntimeError`，不做静默兜底。

### 标注公司 patch_label 回补 + 合并基线重建（2026-10-10，详见 `memory.md` 4c 节）

- **数据位置（服务器 `/mnt/data/…`，本机 SSHFS 挂载 `/home/manu/mnt/data/…`，同一份数据）**：
  - 回补 patch（**只读输入**）：`/mnt/data/siping/datasets/manu/龙泉山/label/patch_label`，内含
    `VIDEO00006_19700101_000322_correct_labels/`（3,264 文件，跨 `_000322`+`_000703` 两段）与
    `VIDEO00036_19700101_015740_correct_labels/`（996 文件），另有各序列 `.zip` 原包未动；
  - 合并底板：`…/龙泉山/label/Final_Labels`（165,223 标签 + `classes.txt`）；
  - **当前评测真值基线**：`…/龙泉山/label/Final_Labels_merged_20261010`（165,224 项 / 532 MB）；
  - 原始帧：`…/龙泉山/frames_ir_jpg`；复核视频：`…/龙泉山/gt_review_20261010`（5 个 mp4 / 1.3 GB）。
- **patch 不是可直接使用的标注集**：它是**嵌套两层**（`patch_label/<SEQ>_correct_labels/`）且**只含改动帧**；`render_gt_osd_video.py` 走 `iterdir()` 非递归，直接指向会读到 0 个标签（响亮失败，非静默空视频）。
- ✅ 新增 `manu/data/merge_patch_labels.py`：以 `Final_Labels` 打底 + patch 覆盖，产出完整扁平根，并逐序列断言「标签索引全部能寻址到帧」（24/24 序列 labels == frames）。
- **回补量（脚本 diff 权威）**：`_000322` 补 **541** 帧空标签（区间 15617-15810 / 15871-16159 / 16369-16426）；`_00036` 补 **146** 帧（13 个区间散布全序列）、另调 104 帧框；`_000703` 仅 **2 帧框调整（1739-1740），不是漏标**。
- ✅ **双路互证**：渲染脚本独立统计的 `labeled_frames` 14,821 / 731 减去旧 GT 14,280 / 585，**恰为 541 / 146**，与 diff 一致。
- ✅ **GT OSD 复核视频已出 5 个**（ffprobe 帧数时长全部匹配）：3 条全序列 + `_000322` 定点段 15497-17016 + `_000703` 定点段 1700-1780（scale 4）。
- ⚠️ `_00036` 补标后 **Recall 分母变大、F1 先降后升**，属正常，不得据此判算法退化。
- ⚠️ 自我更正：曾误记「25 个序列前缀」，实为 `classes.txt` 被 `sed` 计入；**权威值 24 个序列**。

### 多尺度 Trial 0474 热图 + pkl cache（2026-10-09 `manu/data/multiscale_heatmap_cache.py`）

- **模型只有一个：冻结 Trial 0474**，各尺度跑同一模型；**不含 YOLO26 BBox**（YOLO 伪标签分支是 `preannotate_hm_bbox.py` 那条独立路线）。
- **融合口径对齐 `probe_heatmap_regions_video.py --fusion-row`**：各尺度热图先逆映射回**原生 640×512 网格** → `np.maximum.reduce` → 按 `main_threshold`(0.22) 二值化 → 膨胀 `fusion_dilate`(9) → 8 连通域 → `area >= min_area`(4)。逐尺度行**不膨胀**，用 `region_threshold`(0.06)。
- 每个检出 `(cx, cy, w, h, area, peak, sum)`，score = 连通域热图峰值，**按 sum 降序**（与视频脚本一致，非按峰值）。
- **Cache CSR 布局**：`frame_index`/`offset`/`points`(**float32**)/`sizes`/`scores`(f16)/`sums`/`area`(i32)/`tag`(尺度下标，**255=融合行**)，配 `iter_frames(cache)`。
- **多卡靠 `--shard i/n`**；每序列独立写 `{seq}.pkl`，无共享 manifest。
- ⚠️ 落盘序 `[I_t, diff, median]` 必须**每尺度** `[..., ::-1]` 转成模型序 `[median, diff, I_t]`；`copyMakeBorder(value=114)` 只填第 0 通道；`glob("__frame_*.jpg")` 返回 0，须写 `"*__frame_*.jpg"`。
- **尺度量化误差**（热图往返）：原生 0.0px、160 尺度 1.0px、**80 尺度 4.5px**（1 输入像素 = 8 原生像素，固有），仍 < 8px 匹配容差。

### RK3588 板卡到手 + NFS 打通（2026-10-09；**2026-10-12 挂载方向反转**）

- 登录：**`ssh -p 22 root@192.168.0.64`，密码 `ematech`**（明文 root 口令，禁止写入汇报材料/代码仓库；完整规范见 `rules.md` 第二节 1b）。
- 🔥 **2026-10-12 挂载方向反转为「板卡挂 host」，2026-10-14 导出目录换到 SSD960-LOCAL 大盘**，板卡执行
  `mount -t nfs 192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu -o nolock,soft,timeo=5,retrans=2`
  ⇒ 板卡写 `/mnt/manu` 落在 **host 磁盘**上，不再吃板卡那 1.7G。本机 IP `192.168.0.115`。
- ✅ **2026-10-14 双端实测通过**：`/etc/exports` = `/run/media/manu/SSD960-LOCAL/nfs 192.168.0.0/24(rw,sync,no_subtree_check,insecure,no_root_squash)`；`/proc/fs/nfsd/exports` 内核实况**确有**该 rw 导出；`etab` mtime **晚于** `/etc/exports` ⇒ **确实跑过 `exportfs -ra`**。板卡侧 `mount` 实测 `...SSD960-LOCAL/nfs on /mnt/manu type nfs4 (rw,...)`，`df` = `880G / 54G used / 782G avail`。**写链路已回环验证**（板卡写 ⇒ host 新盘当场可见，属主 `root:root`）。
- ⚠️ **旧路径 `/home/manu/nfs` 已作废**：仍在 host 上留着一份 4.5G 拷贝（内容一致、非 bind mount），**极易误判**，但已不被导出，照旧命令挂载必失败。清理需用户 `sudo`。
- 🟢 **红线已解除**：旧「host 根分区仅 15G 可用（93%）」不再成立 ⇒ 现为独立大盘 **`782G` 可用**，全量 val 的 33G 级暂存不再受限。GiB 估算义务（铁律二）不变。
- ⚠️ **同路径语义已变（最易踩）**：`板卡:/mnt/manu` 现在 = `host:/run/media/manu/SSD960-LOCAL/nfs`；旧文档里的 `/mnt/manu/ocvparity/`、`/mnt/manu/fused/` 等**在板卡上已不存在**（那是旧方向下板卡本地盘的内容）。
- **（旧方向，已拆除）本机挂板卡**：`sudo mount -t nfs 192.168.0.64:/mnt/manu /home/manu/mnt/nfs -o nolock` ⇒ 生效为 `nfs4 vers=4.2 hard proto=tcp`；挂载点目录已于 2026-10-12 删除，**别再照抄这条命令**。SSH 横幅 `OpenSSH_8.2p1 Ubuntu-4ubuntu0.12` ⇒ 板卡是 **Ubuntu base 的 arm64 Linux**（结论仍有效）。
- RK3588 从「纯 x86 侧 C++ 移植 + Golden Reference」推进到「有真机可验」；`cpp_port_rk3588.md` 的两项未决（**L3 全量位精确**、**真实板卡时延**）有了执行载体。
- ⚠️ **板卡本地盘仍仅剩 1.7G 可用（16G 总量 / 14G 已用 / 89%）** —— 转 rknn、装 runtime、拷模型前必须先 `df -h /` 并给 GiB 估算（同铁律二的缓存体积红线）；数据类改走 `/mnt/manu`。
- ⚠️ **NFS 不能替代代码同步**：只含数据交换，代码仓库不在其内；铁律三的自动同步仍只覆盖 x86 挂载。
- ⚠️ **板卡上没有任何 `runs/` 权重**（`runs/` 从未纳入同步链路），板卡推理必须把权重与数据双端就位列为前置检查。
- ⚠️ **板卡侧是 `soft` 挂载**：经 `/mnt/manu` 的耗时含网络往返，**不得计入板卡算子时延预算**；时延只认本地盘数据 + `--timing` 分段表。
- ⚠️ **PoCL/CPU 模拟时延严禁当作板卡性能**。

### 板卡 OpenCL 环境探针（2026-10-09，详见 `opencl_fused_rk3588.md` 第八节）

- **交付**：`manu/pipeline/opencl/board/`（`board_cl_probe.cpp` + `build_board_probe.sh` + `deploy_and_run.py` + README）。枚举平台/设备、解析 ICD、跑平凡 kernel 并逐元素校验。**已在板卡实跑 `PROBE RESULT: PASS` / exit 0**，传输前后 MD5 一致（`ec641df354e998b21e51a22ee016159f`）。
- **设计要点**：用 **dlopen** 而非 `-lOpenCL`——工具链 sysroot 无任何 OpenCL、SDK 只有 buildroot 配方，链接期依赖无法满足；改为运行期决定后，「无 OpenCL」变成干净报告而非链接错误，并能报出是 ICD loader 还是直连 Mali。
- ✅ **板卡 `evm3588` 实测**：Ubuntu 20.04.6 LTS，内核 `5.10.160`；8 核 = 4×A76 + 4×A55；**RAM 7.7 GiB**（⚠️ memory 里「16G/14G/1.7G」是**磁盘**，且 `/mnt/manu` 与 `/` 同一文件系统）；GPU 已就绪（`/dev/mali0`，内核内建驱动，`lsmod` 查不到属正常）。
- 🔴 **板卡 OpenCV C++ SDK 专项探测（`board_opencv_probe`，双证据一致 = 无）**：**板卡有完整编译器**（`gcc/g++ 9.4.0`、`make`、`/usr/local/bin/cmake`、`pkg-config`，**可原生编译**）；但 OpenCV C++ SDK **不存在**——`dpkg` opencv 包 0 个、`pkg-config opencv4/opencv` not found、全盘扫**头文件 0 / `libopencv*.so` 0**、`dlopen libopencv_core.so` 失败，**且实际 `g++ -lopencv_core` 两次均报 `fatal error: opencv2/core.hpp: No such file or directory`**。
  ⚠️ **`import cv2` 成功 ≠ 装了 OpenCV**：4.12.0 是 pip wheel 的 `cv2.abi3.so` 自包含模块，不带头文件/可链接库/`.pc`。
  ✅ **但 apt 有候选 `libopencv-dev 4.2.0+dfsg-5`** ——⚠️ **远低于 x86 侧 4.10.0**，直接装会破坏「调用同一批 OpenCV C++ 原语」的位精确前提，装前须做等价性验证。
  - ✅ **已解决：自行交叉编译 arm64 OpenCV 4.10.0，板卡实跑通过（2026-10-09）**，不再依赖 apt 的 4.2.0。
    - **产物**：`/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install`（静态库，模块 `core imgproc features2d video calib3d flann`，全关 IPP/OCL/FFMPEG/图像编解码/LAPACK/TBB），**约 4 分钟**编译完成；源码 worktree `/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64`（用户原有 5.x 树未动）。
    - **跨架构一致性实测（59 帧 × 19,333,120 元素，逐算子 dump 对比）**：`resize` / `goodFeaturesToTrack` / LK status / LK 输入点 / **RANSAC 内点掩码** = **全部位精确**；LK 亚像素位置 max **1.06e-4 像素**、RANSAC 矩阵 max **1.64e-5**；`warpAffine` **56/59 帧位精确**，19.3M 像素中**仅 11 个不同（0.000057%）**、最大 2/255；21 元素 median **38/39 帧位精确**，12.8M 中**仅 1 个不同**。
      ⇒ **判决：arm64 OpenCV 可用于特征通道**；RANSAC 判定在两架构完全一致，差异停在 float32 末位。
    - 🔴 **FMA 归因假设被实测证伪**：同架构 x86 上 `OPENCV_CPU_DISABLE=AVX2,AVX512_SKX,FP16,SSE4_1,SSE4_2` 关掉带 FMA3 的 dispatch 后，**13 个 stage 全部 59/59 位精确** ⇒ LK 差异与 FMA 无关，真实来源是**架构特定 SIMD 实现（x86 SSE/AVX vs aarch64 NEON/carotene）+ 编译器版本差 6 年（gcc 15.2.0 vs gcc 9.3.0）**，不可消除。
      ⚠️ 顺带更正我自己的错误推理：x86 参考版 `Baseline: SSE SSE2` 无 FMA，**但有 runtime dispatch 实际走带 FMA3 的 AVX2/AVX512 路径**，基线无 FMA ≠ 执行路径无 FMA。`OPENCV_CPU_DISABLE=FMA3` 无效（FMA3 不是独立 dispatch 名）。
    - 🔴 **样本量教训**：首轮只跑 23 帧，`warp_dst`/`median_out` 恰好全一致，我据此判「差异未传播」；**扩到 59 帧后被推翻**（warp 有 3 帧、median 有 1 帧存在差异）。稀疏差异必须用**跨全部元素的绝对计数**报告，**「零观测」≠「不存在」**。
    - **三个静默构建坑**：① 系统 CMake 4.2.3 会**硬拒** OpenCV 4.10 的 `cmake_minimum_required(3.1)`，须用 3.31.6（不改源码以保持与 tag 逐字节一致）；② 链接报 `carotene_o4t::split*` 未定义，实际库名是 **`tegra_hal`**，装在 `lib/opencv4/3rdparty/`（**不在 `lib/`**）；③ 即使 `WITH_JPEG=OFF`，`persistence.cpp` 仍引用 zlib，须加 **`-lzlib`**（文件名 `libzlib.a`）。
    - ⚠️ **交叉前缀更正**：板卡 glibc **2.31**，sysroot **2.29**；`aarch64-rockchip930-linux-gnu-` **无 sysroot 目录**，早期「板卡对应 rockchip930-」的推断是**错的**，应用 `aarch64-rockchip-linux-gnu-`。
    - 详见 `memory/opencv_arm64_parity.md`；工具在 `pipeline/opencl/board/`（`build_opencv_arm64.sh`、`rk3588-aarch64-toolchain.cmake`、`opencv_parity.cpp`、`gen_opencv_parity_case.cpp`、`compare_opencv_parity.py`、`push_and_run.py`）。
    - ⚠️ 跑对比脚本要用 **`/home/manu/anaconda3/bin/python`**（系统 `python3` 无 numpy）。
    - ✅ **2026-10-10：`gmc_stream_ocl.cpp` 原样交叉编译后在板卡跑通完整链路**（读 BMP → GMC fit → fused kernel → Ch0/Ch1/Ch2，**未改任何源码**）。
      **跨架构精度 PASS**：板卡 vs **x86 `gmc_stream.cpp` 纯 CPU 权威基线**同参数同输入，**Ch0/Ch1 Max|Diff|=0 全位精确**，Ch2 仅第 41 帧 1 个像素差 1 级（前后帧完全一致 ⇒ 舍入抖动非系统性漂移）；transform 矩阵 126/126 全等。**三条链路闭合**：`板卡 GPU` ≈ `x86 gmc_stream.cpp` ≈ `x86 gmc_stream_ocl`；x86 上两实现 MD5 相同互为印证。
      **性能**（重复 3 次中位数，四配置）：板卡 GPU 臂 total **156.50 ms（6.39 fps）**；板卡 CPU 臂 150.36；x86 OCL(PoCL) 176.52；**x86 `gmc_stream.cpp` 纯 CPU 68.69（14.56 fps）**。GPU 加速 **24.6×**（132.4→5.38，kernel 3.84）；只用 GPU 尾段可达 **44 fps**。
      🔴 **板卡 CPU 比 x86 CPU 慢 2.19×**（同源码/同输入/单线程）⇒ CPU 路径在板卡不划算，这是选 GPU 融合的现实依据。
      🔴 **x86 上 GPU 比 CPU 慢 0.59×**（PoCL 模拟），同二进制两硬件差 **19.8 倍** —— 再次证实 PoCL 结果不可当板卡性能。
      🔴 **当前瓶颈是 CPU 不是 GPU**：CPU 占 97%、GPU 仅 3.4%。
    - 🔴 **三条推翻旧认知的事实**：① **x86 参考实际链接系统 libjpeg-turbo 2.1.5**（非内建 3.1.2），源码树 3.0.3、板卡 1.5.2/2.0.3 —— **四者无一匹配** ⇒ 改走 **BMP**（OpenCV 内置、零外部依赖、无压缩无解码分歧，实测往返无损）；② **板卡链接期 OpenCL 可用**（`libOpenCL.so.1` 34 KB 在 ldconfig 里，探针 `context=OK`），旧记录「无 loader」不准确，`ocl_host.h` 无需改 dlopen；③ **`imgcodecs` 进 BUILD_LIST 必须加 `-DWITH_OPENJPEG=OFF`**，否则链接期报 `opj_*` undefined。
    - ⚠️ **逐算子比端到端更严格**：`opencv_parity` 里 `warp_dst` 差 11/19.3M 像素，端到端跑完只剩 **1 个**像素浮出。
    - 🔴 **本轮推翻了 `opencl_fused_rk3588.md` 的两条旧结论**：① 「无任何 arm64 OpenCV」已不成立；② 「fit 从未在板卡测过」已解决，且旧的「5 倍分歧」获裁决——**板卡实测 fit 仅 17.3 ms/帧，不是折算的 30~41 ms**；③ **真正瓶颈是 CPU 的 warp+median（86.1%），不是 fit（11.1%）也不是 GPU（3.4%）**。
    - ⚠️ **四个可复用的坑**（详见 `falsified_archive.md` 第三十七节）：**build info ≠ 运行时链接的库**（x86 实际是 turbo 2.1.5，非配置串里的 3.1.2）；**依赖版本对不上时换格式而非硬凑版本**（改 BMP）；**新增模块后必须有真调 API 的链接测试**（`imgcodecs` 的 OpenJPEG 只在链接时炸）；**`.icd` 坏 ≠ ICD 机制不可用**（loader 一直在，只是它指向的库不在）。
    - 详见 `memory/gmc_ocl_board_run.md`。
- ✅ **Mali OpenCL 实测可用**：`Mali-G610 r0p0`，OpenCL 3.0 `v1.g13p0-01eac0`，FULL_PROFILE，**4 CU**、max WG 1024、global 7902.1 MiB、local 32 KiB。**ICD 是目录式**（ARM 布局，`mali.icd`），且 **`.icd` 指向的 `libMaliOpenCL.so.1` 根本不存在**，真正驱动是 `libmali.so.1.9.0`（包 `libmali-valhall-g610-g13p0-x11-gbm`）——排查时直接 `find`，别信 `.icd`。
- 🔴 **真机推翻三条原计划**：①**无 `cl_khr_fp64`** ⇒ double-float 回退作废（有 `cl_khr_fp16`）；②**事件级 profiling 不可用** ⇒ 板卡取不到 device-side 时间，只能给 `clEnqueue+clFinish` 墙钟**上界**；③Mali **首次启动某 kernel 会现场编译**（探针 0.475 ms → 二次 0.344 ms），单次数字必须区分首次/稳态。可用扩展：`cl_khr_image2d_from_buffer`、`cl_khr_egl_image`、`cl_khr_suggested_local_work_size`、完整 subgroup 系列。
- 🔴 ~~**NPU 当前不可用**~~ —— **🔴 2026-10-13 实测推翻：NPU 驱动一直正常，RKNN 推理已跑通。**
  驱动 **RKNPU 0.9.8** 以 **DRM 子系统**注册，真实设备节点是 **`/dev/dri/card1`**（不是 `/dev/rknpu`），
  `dmesg | grep -c rknpu` = **16**，`lsmod` 查不到属正常（内建驱动）。
  **旧的三条判据全部假阴性** ⇒ 正确探针是 `readlink -f /sys/class/drm/card*/device/driver`。
  `librknnrt.so` **2.3.0** 可用。已实跑 `init→inputs_set→run→outputs_get`，检出正确，
  **`rknn_run` 24.4 ms（占 97%）/ 端到端 25.1 ms / 39.8 fps**；开 `--perf` 标志慢 54%（进预算取 `rknn_run`）。
  ✅ 已出 **OSD 图**并与 **Rockchip 官方 demo 逐框对账一致（偏差 ≤6 px）**；官方 demo 也已可复现运行（参照臂）。
  ✅ **ONNX→RKNN 重新量化闭环已跑通**（`manu/pipeline/npu/convert/` + `CONVERT.md`）：从 `examples/onnx/yolov5/yolov5s.onnx` 重新量化的 `.rknn`，与官方 `yolov5s-640-640.rknn` 检出结果一致。
  ⚠️ **一致性比例必须在 ≥300 样本上测**：10 张图给的「94.1% 匹配」，扩到 **300 张 / 1369 框**后是 **84.5%（95% CI [82.5,86.3]）** —— 小样本把比例**抬高约 10 个百分点**，还会**凭空造出「分歧全在决策边界」的假规律**（扩样后同类率变成 94.5~100% 的平滑单调，无边界效应）。详见 `rules.md` 三·6b。
  🔴 两个重量化模型都**系统性多检出 18~19%**（1622/1629 vs 1369），是敏感性差异不是噪声；且**标定集从 1 张扩到 120 张并不能缩小与官方模型的差距**（84.5% vs 84.1%），差异来自源 ONNX 不同。
  🔴 **toolkit 版本必须与板卡 runtime 的 commit 对齐**：SDK 自带的 1.4.0 产出的模型会被 NPU 在**提交阶段**拒绝，且**不报错退出**，只表现为「单次 6395 ms（正常 23 ms）+ 分数全挤在 0.9976 + 框退化」；换 toolkit 2.3.0（与板卡同 commit `c949ad889d`）一次通过。
  🔴 踩到 **6 个静默失效坑**，最隐蔽的是 **center→left 做了两次 ⇒ 框整体左上歪**（不报错、指标不变，只能靠与参照实现对账发现）。详见 `memory/rknn_npu_board_run.md`。
  ⚠️ **教训：断言「硬件不可用」前必须先证明探测手段对该版本有效。** ⚠️ **OpenCL 走 GPU 与 NPU 无关，两条线分别取证。**
- ✅ **板卡命令现可由 Agent 自动执行**：本机虽无 `sshpass`，但 **`pexpect` 可用**，已封装 `deploy_and_run.py`（base64 单会话传输 + 部署 + 摸底 + 运行，`--exec` 任意查询）；密码不入文件，由 `RK3588_PASSWORD` 环境变量提供。**取代**「无法非交互登录板卡」的限制。
- **四个编译期查不出的坑**（已入 `falsified_archive.md` 第三十四节）：① `CL_SUCCESS==0` 使 `!rc` 把成功读成失败；② `clEnqueueNDRangeKernel` 真实 ABI 顺序是 `(offset, size, local)`，三参数同型无从校验；③ 未检查的 `clSetKernelArg` 返回值被 **PoCL 误报为 `-52 CL_INVALID_GLOBAL_WORK_SIZE`** 而非 `-51`；④ **ICD 目录/文件布局混淆**——`fopen` 目录会成功而首读 EISDIR ⇒ 伪报「ICD 为空」，与真实故障无法区分。

### 🔴 真机精度验证定论（2026-10-09，Mali-G610，详见 `opencl_fused_rk3588.md` 8.2）

交付 `manu/pipeline/opencl/board/{gen_fused_case.cpp, board_fused_accuracy.cpp, fused_case_format.h}`：
x86 用**真实 OpenCV** 算 CPU 基准（**当时**全链路无任何 arm64 OpenCV ⚠️**该前提 2026-10-10 已不成立**，见本节后续条目），在板卡重实现 OpenCV 等于自证），
板卡无 OpenCV、只 dlopen OpenCL，跑**两个采样臂**。

- **判决一：`FUSED_USE_HW_LINEAR` 必须为 0（manual 双线性）。** 数据：Anti-UAV `01_4485_1167-2666` 640×512、**真实 GMC**、3 case、pad=67。
  - `hw-linear`：**FAIL**（Ch0 Max|Diff|=**130**、MAE 0.285、精确率仅 88.5%；Ch1/Ch2 MAE 0.31~0.42）
  - `manual`：**PASS**（**Ch0 Max|Diff|=0、100.00% 精确**；Ch1 Max|Diff|=5~6、MAE 0.0011；Ch2 Max|Diff|=1~4、MAE 0.00075）
  - **决定性证据是 Ch0**：整数坐标直读本应逐位还原却错 130 ⇒ **Mali 的 `CLK_FILTER_LINEAR` 连整数坐标都不还原原值**。
  - manual 臂 Ch1/Ch2 与已记录的 x86 基线一致 ⇒ 工具正确，差异来自硬件。
- **判决二：Mali 支持 1 字节/像素**，`{CL_R, CL_UNORM_INT8}` 可用 ⇒ 22 张 padded 图 **10.5 MiB**（CL_RGBA 为 42 MiB）；`clCreateImage`(2.0) 在 Mali 正常工作。**「CL_R8 vs CL_RGBA 带宽」项结案。**
- ⚠️ **判决三（2026-10-09 已更正）**：首轮「3.7~4.8 ms」**只圈了 kernel**（upload 在计时窗口之前、readback 在之后），**不是尾段总成本**。三段实测（manual 臂）：**upload 2.72~3.33 + kernel 3.87~5.71 + readback 0.11~0.12 = 合计 6.70~9.15 ms/帧**，upload 约占 40%。同批 PoCL 合计 131~135 ms ⇒ 真机约 **15~20×**。
   🔴 **本节合计仍不含 GMC fit**（mats 由 x86 下发）。⚠️ **此处对 fit 的判断已被 2026-10-10 实测推翻**（当时无 arm64 OpenCV；fit 实测仅 **17.34 ms/帧**，非折算的 30~41 ms）。真实瓶颈是 **CPU 的 warp+median（86.1%）**，非 fit 非 GPU。墙钟均为**上界**（事件级 profiling 不可用），首次含 JIT。
- 🔴 **顺带证伪 `ocl_host.h` 两个规格常量**：`CL_R8=0x10D0` 实为 `CL_SNORM_INT8`（非法 channel_order，正确是 `{CL_R,CL_UNORM_INT8}`）；`CL_IMAGE_OBJECT_2D=0x10F0` 实为 `CL_MEM_OBJECT_BUFFER`（应为 `0x10F1`）。**在 x86 被双重掩盖**（PoCL 拒绝→退 CL_RGBA；`clCreateImage` 失败→走不读 `image_desc` 的 `clCreateImage2D`），**上 Mali 两条同时生效**。
- **可执行文件禁止经 NFS 投递**：x86 写入后在板卡执行报 `Text file busy`，换新名也无效 ⇒ **二进制走 SSH、数据走 NFS**。
- 另有四个坑已入 `falsified_archive.md` 第三十五节：`/*__SORTNET__*/` 未拼接导致 **Ch2 静默错误而 Ch0/Ch1 完美**（已加 FATAL 闸门）、上传未按 `nch` 扩展行距致越界、`clCreateSampler` 是 **5 参数**（漏 `normalized_coords` 即参数左移崩溃）、`clSetKernelArgSampler` 在 ocl-icd 上不存在（改用 `clSetKernelArg` 传 `cl_sampler`）。
- **本地交叉编译器**：`/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu`；**SDK 资料**：`/media/manu/1TB-Volume/rk3588/rk3588_sdk`。二者只放 x86 本地，严禁拷到板卡（磁盘仅剩 1.7G）；换工具链**不得绕过** `-ffp-contract=off` / 拒绝 fast-math 的浮点契约。

### RK3588 NPU：Trial 0474 热图检测器移植（2026-10-13/14，详见 `heatmap_npu_trial0474.md`）

- 单帧 SOTA 首次上板：`best.pt → ONNX → RKNN int8（4.09 MB）→ 板卡`。⚠️ **v1（带 p0）的 `rknn_run` 151.701 ms / 端到端 152.917 ms → 6.5 fps 已是历史档**；
  **当前交付版 = v3 去 p0 highway**（`trial0474_hm_nop0_int8.rknn`，3.80 MB）：`rknn_run` **55.5 ms**、端到端 **56.5 ms → 17.6–17.7 fps**、
  **CPU 计算算子 0、fallback 警告 0**，三代累计 **2.69×**。空白拒答 / 空对照 / 确定性三闸门 + 时钟自检 `[ok]` 全过。标定用 **200 张已建特征 JPEG、跨 199 序列、只取 train**。
- CPU 参考臂 600 帧分层：P **402/419 = 0.9594** [0.9360, 0.9745]、R **402/490 = 0.8204** [0.7840, 0.8519]、F1 **0.8845**（F1 非比例，不附区间）；**88 个 FN 里 42 个来自两个 DJI（512×512）序列**，是模型域弱点、不是移植缺陷。✅ 600 帧 NPU↔CPU 对账：一致帧 **572/600 = 0.9533** [0.9334, 0.9675]，**「两边都检出但相距 >8 px」= 0/600**（int8 从不改变检出位置，只在阈值附近影响 28 帧）；🔴 剔除 DJI 后召回 **0.8955 → 0.8727（−2.27 pp）**，配对 McNemar **p = 0.0129 ⇒ 显著**（全 600 帧 p = 0.0002）；⚠️ 初版曾用「Wilson 区间重叠」判为不显著，**该做法对配对数据无效**，已列为新规 §三·6c。
- 🔴 **新规 §三·10**：`torch.onnx.export` 返回后模型**仍在 train 态**，BN 改用批统计 ⇒ 导出后取参考值会得到 heatmap diff **0.2801** / offset **1.677**（eval 态应为 **5.07e-7 / 2.48e-5**）；**参考值必须在导出前抓**。
- 🔴 **新规 §四·2e**：原始标签按**源尺寸**归一（`source_norm*src_hw*scale+pad`），与 dataloader 的画布归一 `batch["bboxes"]` **禁止混用** —— 同一 200 帧 recall 因此从 **31/158 = 0.1962** [0.1418, 0.2650] 摆到 **144/158 = 0.9114** [0.8568, 0.9465]。
- 🔴 **新规 §四·2f**：**对比臂必须自己上过板**；~~同一份 ONNX 的 **fp16 构建在本板卡提交失败**而 int8 正常 ⇒ 不得由「int8 能跑」推出「fp16 能跑」~~
  ⚠️ **2026-10-16 更正适用范围**：失败算子 `Conv:/model/p0_highway/p0_dw/Conv` **在 P0 支路内部**；
  **去 P0 的 fp16 实测可提交、可全量跑完 31,613 帧**（见 §22）。该条应读作「**图含 p0 支路时** fp16 提交失败」，
  **不是**「这块板子不支持 fp16」。铁律本身（对照臂必须自己上过板）**依然完全正确** —— 板子没变，是对照臂换了图。
- 🔴 **v2（0 CPU 计算算子）：PixelUnshuffle 的回退是布局类，且上限由空间乘积决定**。导出后 rknn-toolkit2 把 `nn.PixelUnshuffle` 折成 `Reshape` 到 (1,16,4,**102400**)，随后的 Transpose 触发超限 `height(4)*width(102400)=409600 > 16384` ⇒ **1 个计算算子落 CPU**；**上限卡在 `H*W ≤ 16384`，与张量怎么拼无关** —— 改写成 slice+stack+reshape 同样逐位精确，却落到同样的 409600 且 CPU 算子 **4→10** 更慢 ⇒ **改写张量表达式救不了布局类回退**。**解法：把布局变换改写成算术** —— 2×2/stride-2 的 **depthwise conv**，`groups=C` 时输出通道天然编为 `c·f²+i·f+j`，与 PixelUnshuffle 定义一致、无需任何 transpose；四种形状（(1,16,640,640)/(1,3,320,320)/(2,16,128,96)/(1,64,8,8)）**全部 `torch.equal` EXACT**，整网 **31 帧 NPU 输出与旧模型逐字节相同** ⇒ **`rknn_run` 151.2 → 103.7 ms，零精度代价**。实现 `hm_pixelshuffle.py`（`--verify` 自检）。
- ✅ **当前交付版 = v3 去 p0**：`trial0474_hm_nop0_int8.rknn`（**3,799,304 B**，md5 `ef302edb1b5aff1d376204bb7ad77550`），端到端 **56.5 / 56.7 ms → 17.6~17.7 fps**、**CPU 计算算子 0**、fallback 警告 0；**v1 152.4 ms / 6.6 fps → v3 56.7 ms / 17.7 fps，累计 2.69×**。P0 可删的依据：底模 Trial 22 **100% 物理冻结**（1,790,531 参数）、支路仅 1,954 参数零初始化冷启动、**Step 0 绝对数学恒等**，训练收益仅召回 **+0.15 pp**、收敛后 F1 反而略降；**600 帧配对检验：CPU 带 p0 vs 去 p0 不一致帧仅 1/600，配对 McNemar p = 1.0000 ⇒ 去 p0 无可归因精度代价**（TP 402→401、FP 17→16、F1 0.8845 → 0.8842）。参考版本严格采用 `cache_trial0474_inferences.py --ablate-p0` 的构造，**ONNX 对 ONNX 字节级验证 MD5 完全相同 `028f23c655f421fe61a5e43575715fca`**（7,210,756 B，494/494 权重逐位相同）。
- 🔴 **最反直觉：v2→v3 MAC 只降 6.8%（23.11 → 21.54 G）、时延却降 46.5%（103.7 → 55.5 ms）**，差近 7 倍 —— p0 那几层是 **640×640 / C=16** 的卷积，既大分辨率（走 DDR 搬运）又极窄通道（MAC 单元空转），**同时踩中 NPU 两个最差工况**。⇒ **算子回退（47 ms）与结构劣化（48 ms）是两笔独立的账、代价相当，只查回退会漏掉后一半。**
- 🔴 **结构性发现：`p0_highway` 与 stride 4 不兼容**（内部只下采样一次、残差恒为 stride 2，无法与 160×160 的 neck 相加）⇒ **该支路把整网锁死在 stride 2、锁死在 320×320 head**，删掉它顺带解锁 stride 4。⚠️ **10 ms 目标尚未验证可达**：最激进变体 `s4_head1x1` 为 6.68 G MAC、粗略外推约 17 ms，但**线性外推已被证伪，必须实测**。
- 🔴 **新规 §四·9e（用户铁律）：int8 上板模型必须 0 个 CPU 计算算子**，判定命令见 `rules.md`，`InputOperator`/`OutputOperator` 豁免；三条配套 —— `fallback` 警告必须同时为 0、改写须**逐位等价**、改写张量表达式救不了布局类回退。**本轮该闸门实测把交付时延从 151 ms 压到 104 ms。**
- ⚠️ **如实记录**：v3 的 600 帧 NPU↔CPU 走的是**检测级**对账（板端新增 `--detections-out` 紧凑 TSV；dump 原始张量经 NFS 约 2 帧/分钟、600 帧需 583 分钟），**未做逐元素张量对账**；逐元素口径在 v2 上已验过（heatmap mean|diff| **3.316e-4**、offset **6.120e-2**）。
- ✅ **端到端串行 demo 已跑通（2026-10-14）**：`原始红外序列 --板--> gmc_stream_ocl（CPU fit + Mali-G610 融合尾）→ 三通道特征 --板--> rknn_heatmap_smoke（RKNN int8 去 p0）→ 检测`。
  **刻意串行、不融合二进制**：两段都是已认证的现成二进制，代价是多一轮特征 I/O（512×512×3 BMP ≈ **786 KB/帧**），收益是**每段各有独立可归因的计时器**、可单独替换。驱动脚本 `manu/pipeline/npu/run_hm_demo.sh`（`--sequence` 有 GT 走三层精度对齐；`--video` 无 GT 只报速度）。
  - 实测 `34_1`（Anti-UAV val，**535 帧 / GT 503 / 512×512**）：板端端到端 **TP 496 / FP 49 / FN 7**，P **496/545 = 0.9101** [0.8831, 0.9313]、R **496/503 = 0.9861** [0.9716, 0.9932]、F1 **0.9466**；CPU 权威臂 **TP 502 / FP 72 / FN 1**，P **502/574 = 0.8746** [0.8450, 0.8992]、R **502/503 = 0.9980** [0.9888, 0.9996]、F1 **0.9322**。
    ⚠️ **板端 precision 与 F1 反而更高**（int8 压掉了部分低分伪峰），代价是少 6 个召回 ⇒ **不得只报 recall 单边**。
  - 确定性：两次独立运行的 GMC 特征 **md5 完全相同** `5caa3c6fc72eb7fad4dd5dfd08ebf4ad`；NPU 时钟自检 **50.190 ms `[ok]`**、空对照 **CHANGED**、两次运行 **IDENTICAL**。
- 🔴 **三层精度对齐法：把「板端端到端 vs CPU」压成单一 recall 会掩盖差异来源** —— 板端链路含**两个互相独立、含义完全不同的偏差源**（int8 量化 vs GMC 特征合成），故分三层：
  - **L1 GMC 层**（板端特征 vs 冻结数据集特征逐像素）：平均 **98.75% 像素逐位相同**、逐帧 `max|diff|` p50 = **3** / p95 = **7** ⇒ 差异小 ⇒ **L2 的任何差异不得归因于特征合成**；
  - **L2 检出层**：两臂各自对 GT，业务真正关心的指标（上条 P/R/F1）；
  - **L3 配对层**：NPU vs CPU 同批帧**配对 McNemar**，非显著的 recall 差**不得**报为回归（配对数据用「Wilson 区间重叠」判显著性无效，同 §三·6c）。
  - 本轮 L3 实测：不一致 **6/535 = 0.0112** [0.0051, 0.0242]，**NPU 独中 0 帧 / CPU 独中 6 帧**，McNemar **p = 0.0312 ⇒ 统计显著的一侧性劣势**。
- 🔴 **贴边目标弱点 + 分辨率决定预处理时延 + 两条新铁律（2026-10-14）**：
  - 🔴 **6 个不一致帧的 GT canvas x 全部落在 4.38 ~ 80.63**（画布宽度的前 **12.6%**），而全序列 GT x 中位数 **352.5** ⇒ **不是随机分布、全部集中在左边界**；5 帧 NPU 完全漏检、1 帧（`000347`）检出但位移约 **13 px** 超 **8 px** 容差，两臂共有伪峰 canvas ≈ **(443, 625)**。⚠️ **是否构成业务阻塞待裁决**。
  - 🔴 **预处理时延随分辨率近似线性增长、NPU 阶段恒定**：512×512 时 GMC fit **23.8 ms** / 预处理合计 **31.9 ms**（**40 ms 红线 PASS**）、NPU **56.9 ms**、**端到端 88.852 ms → 11.25 fps**；1024×576 时 GMC fit **73.4 ms** / 合计 **86.6 ms**（**红线 FAIL**，p99 **91.4**）、NPU **56.6 ms**、端到端 **~143 ms**。⇒ **GMC 放大约 3.1× 而 NPU 完全不变**（模型输入恒为 640×640），瓶颈已从 NPU 翻转到 CPU 预处理。
  - 🔴 **新规 §四·9g**：验收输出**不得靠 `grep`/`sed` 过滤判成败** —— 过滤会把 traceback 吞掉、让失败运行看起来成功；**产物 0 字节必须当失败追查**。
  - 🔴 **新规 §四·9h**：预处理时延随分辨率近似线性增长，**必须实测并绑定分辨率引用**，禁止跨分辨率搬数字（与 §四·9f「禁止用 MAC 外推时延」同源）。
  - ⚠️ **遗留**：视频路径无配对 GT 故未做精度对齐（本项目 `.mp4` 全是审片渲染，归档序列是帧目录）；端到端距 **10 ms** 目标仍远；贴边目标 6 帧显著损失待裁决。

## 7. 领域迁移阶段收口边界

- Trial 0474 单帧 F1=0.9064；Trial 0474 加双向平滑、插补和刚性剪枝的系统 In-BBox F1=0.9209，不能混写为底模指标。
- 近距大目标直接 160 下采样 + Ch0 灰度均值/114 Padding、Ch1/Ch2 填 0，已获得稳定定性正向结果；双尺度 0.5x/0.25x、主从 NMS、`sum>=15` 能量兜底尚未完成定量 A/B，不能宣称彻底解决。
- 黑热正确路径是原始灰度逐帧反相后重新计算 GMC+21 帧中值；已有定性验证，但人工 GT 定量指标仍待补齐，无元数据动态极性识别仍属候选。
- GMC 健康度门控、NUC 列去条纹、CLAHE、曲折度剪枝、恶劣天气动态门限均为待固化防御项，尚无完整业务域定量回归。
- 所有迁移方案必须与原始 Trial 0474 做同一连续帧 A/B。
  ~~并通过 Anti-UAV 原始验证集不可退化闸门。~~ 🔴 **该闸门已于 2026-10-13 废止为硬约束**（见 §9.5），Anti-UAV 降级为防崩底板但**仍须每轮测量并如实上报**。

### 待验证方案 A：原生 1:1 双拼多尺度输入

- 640x640 画布上部保留原始 640x512 三通道特征，完全不缩放，保护 1~3 像素目标。
- 画布下部左侧放同一特征的严格等比 0.25x 版本，即 160x128，用于压缩近距大目标；其余区域按通道填充：Ch0 灰度用 114/边缘均值，Ch1/Ch2 填 0。
- 区域 A 坐标保持原生；区域 B 坐标按缩放和 y 偏移反算。该方案目前只是无畸变设计，尚未完成 Trial 0474 接入、双拼训练、融合和 GT 定量验证。
- 若配合 Wh Head，Wh 只能在标注中心稀疏监督，底模保持冻结；Wh 误差和融合效果需独立报告。

### Wh/BBox 尺寸回归结果

- 已完成冻结 Trial 0474 的轻量 WhHead 四卡训练，仅新增约 13,954 个可训练参数；底模、Heatmap、Offset 和 BN 状态保持冻结。
- Epoch 0 与训练结束均完成 SOTA 对账，原始单帧指标保持 `Recall=0.861853 / Precision=0.955665 / F1=0.906338`，说明尺寸支路没有破坏原始检测器。
- 10 Epoch 后 `train_wh=0.05120`、`val_wh=0.05316`；该损失是 log(w/h) 中心点 SmoothL1，不是像素误差。
- 业务视频上估计 bbox 基本可用，但精度低于专用 YOLO Bbox；适合多峰吸收、尺度融合和 Track 关联，不应视为替代 Bbox 检测器。
- 后续以 mAP50-95、宽高像素 MAE、绝对/相对误差中位数及误差覆盖率共同评估。

### 无监督运动解耦试跑记录（2026-09-29）

测试缓存已成功构建，单卡 Optuna Trial 能启动并被正常剪枝；单卡 Trial 已完整跑通20 Epoch，无崩溃/OOM；最终 fitness 约0.063（无监督物理先验分数，不等于Recall/F1）。显著图通过 mean_target=0.05、barrier=2.0、bias=-2.8、Top-K峰值保持和梯度裁剪稳定。stride=12下龙泉山仅13,770窗口，正式缓存前需决定减小stride扩充，或接受约4.5万总量。训练已改为三阶段：先训练EgoMotionNet，再冻结相机分支训练显著分支，最后低学习率联合微调；评估改为有效batch均值，只有塌陷比例超过50%才判定Trial失败。Anti-UAV验证链路已新增：训练用原始无标签灰度序列，官方val用uav_gmc_median的BGR通道0作为I_t并读取labels/val GT；支持checkpoint、阈值扫描、Distance<=8px指标和热图可视化。首次10 Epoch单Trial结果：最佳fitness=0.063099，但官方val最佳阈值th=0.08仅TP=10、FP=211,384、FN=21,858，Recall=0.0457%、Precision=0.00473%、F1=0.0000857。结论是当前无监督显著性未学到可用无人机响应，fitness不能代表检测性能；禁止进入4096 Trial，后续需GT弱监督/显式前景约束或冻结Trial0474受控A/B。

### ST-BgNet 无监督背景重构与 GMC-Net 立项（2026-09-30）

- 新增 `manu/models/st_bgnet.py`：`STBgNet` 参数量 19,609，三阶段盲孔设计（网络只看 `W(I_{t-6}), W(I_{t-2}), W(I_{t-1})`，重建目标为 `I_t`），损失为 Charbonnier + 非对称正残差松弛（系数 0.05）+ TV。
- 数据划分已纠偏为：三源 `anti-uav` / `fpv_data` / `frames_ir_jpg` **全部只做训练**；**官方 Anti-UAV `images/val` 单独做验证**。严禁再对三源随机切分。
- 三条工程教训：①几何契约必须走 YOLO26 风格等比例 letterbox 到 640×640、padding=114，**严禁 resize 拉伸**（会把 512×512 横向拉变形）；②640×640 四帧预对齐缓存实测约 **500 GiB，已手工删除**，任何新缓存必须先给 GiB 估算；③在线 GMC 仍是当前唯一路径，已把特征估计降采样提到 1/8、特征点降到 120，四卡 batch=16 约 5~7 batch/s，CPU 为瓶颈。
- **GMC-Net 待定实验**（详见 `manu/memory/gmc_net_pending.md`）：以 4 点位移参数化 + 可微 DLT 回归单应性替代 OpenCV GMC，目标 30K~60K 参数、NPU 延迟 < 0.8 ms，攻克 CPU 算力黑洞与大晃动内点率崩塌导致 `Ch1` 撕裂引爆 FP；配套免标注真实帧评测（集合 A 大晃动失效集 / 集合 B 稳态强纹理集，指标 CCD 闭环漂移 < 2.0 px、ELE 边缘泄漏能量）。**当前仅设计定稿、代码未实现、零实测指标。**
**口径漂移已定位并修复**：根因是 `cache_trial0474_inferences.py` 把坐标存成 **float16**（letterbox 量级 256~640 处 float16 间隔 0.25~0.5 px，8 px 容差下足以推翻边界帧，致 TP −3 / FP +3）。改为 float32 + 权威抽峰 `0.08/80` 后 **`[SELF-CHECK] PASS`：TP 21,643 / FP 1,004 / GT 25,111 完全对齐**，F1 0.9063607 与标称 0.9064 仅差 −3.9e-05（四位舍入）。此前记录的 `0.9062` 作废；「抽峰深度」归因已被证伪。**权威产出脚本是 `optuna_p0_nas_distributed.py`**，不是 `train_p0_residual_highway.py`（后者 P0 默认架构不同）；memory 原记的 `val_heatmap_resolution.py` 亦不成立（未开 `use_p0_highway`，无法加载该 checkpoint）。
- **Tree 锚点级联 GMC 已判决（2026-10-08，详见 `manu/memory/gmc_net_pending.md` 第八~九、十二节）**：`W(.)` 只在锚点网格 `{2,12,22,32,42}` 直接拟合，其余 lag 由缓存 stride 步连乘（深度 ≤4），**拟合 22 → 6.19/帧（省 71.9%），但 warp 仍 22 次/帧未减少**。**权威口径三臂定稿**：native `F1 0.906361 / R 86.19% / P 95.57% / TP 21643 / FP 1004`；**tree `F1 0.906050（ΔF1 −0.000311）/ R 86.18% / P 95.51% / TP 21641 / FP 1018`**；no-GMC `F1 0.900945（ΔF1 −0.005417）`。**tree 把 GMC 精度代价压掉 94.3%**，dF1 跨 10 门限 6 正 4 负（不可区分），而 no-GMC 10 门限全负（真实退化）；tree 的缺口几乎全部来自 `DJI_0051_2` 一条，`wg2022_ir_052_split_08` dTP −74→**0**。> **作废**：旧口径 `0.9062 / 0.9060 / 0.9009` 与 `TP 21640/21639/21504` 全部作废（float16 缓存伪值，详见第十一节）；`0.9064` 是 `trial_0474/results.csv` epoch 3 冻结标称，引用须注明。> **禁止入汇报**：「几何误差 <0.2px」从未测量；「消灭 75% 算力」只有拟合省了、warp 未动。> **空对照臂角色已被流式引擎 `G2=0` 取代**，ctrl 可取消。> **RK3588 真正阻塞项是 Trial 0474 无任何 NPU 部署产物**（全仓唯一 onnx 属已止损的 YOLO26s 路线）。
- **anchor_step 端点与【阻塞中】的时延不自洽（2026-10-08，详见 `gmc_net_pending.md` 第十三节）**：**精确等价只在 `--anchor-step 2`**（锚点=每个偶数 lag、零连乘），已由流式引擎直接对冻结 native 集验证 `G0/G1/G2` 全 0、21 个锚点 lag 逐个 diff 全 0（范围 1 序列 / 198 帧 / 640×512，**非全量铁证**）⇒ step=2 指标即 **F1 0.906361 / R 86.19% / P 95.57%**，`ΔF1=0.000000`。**「短基线连乘 == 长基线拟合」不成立** —— 合成代数 bit-exact 不能当估计等价的证据。`anchor_step∈{4,6,8}` 未测（从 10 到 6 仅多花 ~3.6% 算力、深度由 4 降到 2）。> ⚠️ **本线唯一阻塞项**：时延测量差 **5 倍**（由两个 `anchor_step` 反解 **9.06 ms/次拟合** vs `stage timing` 实测 **1.80 ms/次**），机制未查明，**两套数字均禁止作为嵌入式预算依据**。> 报「等价」必须同时给出序列数 / 帧数 / 分辨率（本项目两次把局部验证称作全量）。
- **RK3588 嵌入式移植启动（2026-10-08）**：新增部署侧驱动目标，把冻结 Trial 0474 搬到 RK3588 NPU。首个决策问题即「GMC 是**替换**还是**仅加速**」，No-GMC 全量 F1 掉分是其定量输入（三臂权威口径见上）。`--stage timing` 修正了早期抽检的采样缺陷（原 8 帧全落在序列起点钳制区、`ci−2×21<0`，不代表稳态），改为每序列跨全序列均匀抽样并剔除前 2 个窗口。**代码已就绪；RK3588 真正阻塞项是 Trial 0474 无任何 NPU 部署产物**（全仓唯一 onnx 属已止损的 YOLO26s 路线）。
- **C++ 移植 L2 端到端位精确已通过（2026-10-08）**：`manu/pipeline/cpp/gmc_stream.cpp` + 验证器 `manu/pipeline/verify_cpp_port.py`（详见 `manu/memory/cpp_port_rk3588.md`）。**`--anchor-step 2` 臂在双分辨率 2 序列各 200 帧上：G1 400/400 与 Python 引擎逐位一致、G2 396/396 与冻结 `uav_gmc_median` 逐字节相同、`[MATS] max|diff|=0`（各 21 lag）、二进制自报 `fits/frm=21.00 compose/frm=0.00`（结构上印证零连乘）。即 C++ 输出 == Trial 0474 取得 F1 0.906361 所用的特征集，按构造继承 SOTA 单帧精度。** 口径：覆盖 **2/24 序列、400/31,613 帧（1.27%）**，报告自带 `[SCOPE] LOCAL check`，**是局部证据不是全量铁证**，L3 未做。**step 10 臂 G1 亦 400/400 逐位一致，但 step 10 ≠ SOTA 特征**（逐帧 400 帧中 354 帧不同，两序列均自第 23 帧起分歧，第 0~22 帧为冷启动钳制区全同），与 tree F1 0.906050 / ΔF1 −0.000311 一致；「非逐字节相同」须区分「复现 step10 精确」与「step10 等于 SOTA」两件事。
- **第四套时延数据（2026-10-08，仍不可进预算）**：C++ `--timing` 在 **Debug `-O0`** 下 median 占 **58~64%**（`DJI_0051_2` 512×512：fit 101.0 / warp 22.0 / median 216.3 / total 339.6 ms；`wg2022_ir_052_split_08` 640×512：137.9 / 40.3 / 251.0 / 429.5 ms）。瓶颈是 `gmc_stream.cpp:530-534` 逐像素 `std::nth_element` 标量循环，**加速优先级 median ≫ warp > fit**（warp 超线性 1.83x）。`fit` 折算 4.8~6.6 ms/次，落在反解 9.06 与实测 1.80 之间，**但 -O0 与优化构建不可比，无法裁决 13.4 的 5 倍分歧，该项仍是本线唯一阻塞项**。
- **两个工程教训**：①`CMakeLists.txt` 曾写 `${CMAKE_CXX_FLAGS_<CONFIG>}`（`<` 是非法变量名字符）⇒ CMake 语法错误、CLion 完全配不出来，而**逐配置槽位恰是唯一能抓到发行版往 `CMAKE_CXX_FLAGS_DEBUG` 注入 `-ffast-math` 的入口**，即浮点守卫从未生效；已修为 `string(TOUPPER ...)`，并用**反向注入测试**（注入 `-ffast-math` 必须 FATAL_ERROR）证明守卫会失败。②编译产物只在**本地** checkout（`manu/pipeline/cpp/cmake-build-debug/`），服务器挂载上**没有** `gmc_stream`，故 `--cpp-bin` 必须给绝对路径；数据集在 `/home/manu/mnt/data/...` 且序列位于 `anti-uav/train/<seq>/`。
- **OpenCL 融合内核已端到端接入流水线（2026-10-09，详见 `manu/memory/opencl_fused_rk3588.md`）**：`manu/pipeline/cpp/gmc_stream_ocl.cpp` 是 `gmc_stream.cpp` 的超集，CPU 保留 Shi-Tomasi/LK/RANSAC/锚点网格/连乘，**只有 warp + 21 帧中值 + 三通道组装这一段移到 GPU**。**fork 保真已被证明**：`--fused cpu` 的 MD5 与 `gmc_stream` 完全一致（`5e0d336b971948f631640dc983acfc23`），在此之前所有 GPU 数字都不可采信。`--fused both` 两臂消费同一批拟合（每 push 只算一次），排除「两边输入不同」的假通过。**端到端 A/B（`--anchor-step 2`，`--limit 120`）：Ch0 三序列均 100.0000% 逐位精确；Ch1 Max|Diff|=7~8 / MAE 0.0003~0.0010；Ch2 Max|Diff|=6~8 / MAE 0.0001~0.0003。** `--mode nogmc`（W=IDENTITY）下**三通道全帧 100% 逐位相同**——身份变换是最强的全链路正确性证明（padding/ring 索引/通道序/平面排布/坐标约定）。**排序网络 109 次比较，全枚举 2^21 = 2,097,152 个 0/1 输入完备验证通过**（Knuth 0/1 原理，非抽样）⇒ **验收标准写的「91 次比较」复现不出来**，根因是公开的 Batcher 奇偶归并伪码只在 n 为 2 的幂时成立。> **PoCL 上 GPU 臂比 CPU 臂慢 1.3~2.9 倍，这是预期结果不是缺陷**（PoCL 是 CPU 模拟，与 CPU 臂抢同 12 核；且手动 2×2 每采样 4 次 NEAREST 读 vs 真 GPU 1 次滤波读）。**能迁移到 RK3588 的只有传输列**：upload 0.024~0.036 ms、readback 0.325~0.347 ms 是总线流量为真实数字；kernel 112~118 ms 是 CPU 模拟属性，**不得进入嵌入式预算**。> **埋点抓到的两个测量缺陷**：①`clEnqueueNDRangeKernel` 在入队即返回，围它计时得 0.04 ms 而阻塞回读报 116 ms——计算藏在同步里，第一版「回读主导」结论完全错误，须 enqueue 后立刻 `clFinish`；②设备槽位按 `索引 % N` 键入时，冷启动被钳帧与当前帧可同余→跳过上传→**Ch0 31% 像素错、偏 80 灰阶、预热后消失**，现按内容身份分配槽位。

### 龙泉山域工程交付判据与结论（2026-10-09，详见 `memory.md` 第 5d 节）

- **判据**：F1 ≥ **88.0%** 且 Recall ≥ **85.0%**，且工作门限下虚警受控。依据项目内已成文的 `manu/diagnostics/audit_substandard_cases.py:78-81`（`--min-f1 88.0`/`--min-recall 85.0`/`--min-prec 90.0`/`--max-pure-bg-far 0.030`，三者为**与关系**）；⚠️ 该线原为 Anti-UAV **系统级**所设，不可直接套到单帧候选池。
- **实测全局**（24 序列 / 165,223 帧 / 132,570 GT，dist 8px，多尺度 r=64）：**F1 73.58% / R 69.06% / P 78.74% / FAR 0.1496** ⇒ **三指标全部不达标，Recall 线零通过**。
- **multi-scale > single-scale 已确证**：单尺度 640 F1 66.698% → 多尺度 r=64 F1 **73.581%**，**TP +17,321 / FN −17,321 / Recall +13.07**，最优 th 同为 0.30，唯一变量是尺度集合。合并半径是真超参（r=0→32 F1 52.29%→72.98%；**r=0 裸并集 FP 达 64,392 直接崩盘 ⇒ 多尺度必须配跨尺度去重**）。
- ⚠️ **FAR 0.1496/帧 = 军工光电合格线(0.030)的 5 倍**，是最大工程障碍，必须接双向平滑/插补/刚性剪枝。
- ⚠️ **8px 容差对本域偏严**：GT 中位框 49×19px，8px 已达中位最短边的 42%（p10 小框的 80%），而 8px 是照 Anti-UAV 1~3px 弱点定的。同一批预测换 inbox 口径 F1 76.47%→82.53% ⇒ **73.58% 是保守低估**。两口径都报，**严禁静默换口径**。
- **FN 必须拆解**：`fn_absent`（附近无检出=前端召回上限）/ `fn_compete`（有检出被更高分抢走=合并代价）——判断半径是否过大的唯一依据。
- **7 条 UN-PASS**（seqBestF1 远高于全局 F1 者=工作点/预处理可解；本身就低者=能力缺口）：
  `00006_000322`(63.97%，头号事故 FN 6,847) / `00008_002001`(85.16%) / `00012_002251`(**45%→97.11%@th0.62**，纯门限问题) / `00032_014453`(**64%→89.42%@th0.08**，黑热截断) / `00007_001324`(83.32%) / `00003_002111`(80.86%) / `00036_015740`(24.03%，短序列塌陷)。
- ⚠️ `00032` 的 **Precision 97.46% + Recall 47.10%** 组合与 memory §九.4 记录的黑热截断特征高度吻合 ⇒ **疑似预处理问题而非标注问题**，解法是 `I'=255-I` 反相重算特征，**不是打回标注员**。

## 8. 下一步研发方向

1. 以 Trial 0474 作为冻结单帧基线，统一所有实验口径。
2. 按 P0 优先验证黑热极性对齐和 GMC 健康度门控；按 P1/P2 分层验证其余业务 Bad Case 解法。
3. 分析系统级候选池覆盖、轨迹碎片、刚性坏点和虚警来源。
4. 对典型难例进行前端漏检与后处理误删分层诊断。
5. 新方案必须先完成零回归校验，再进行全盘指标比较。
6. 交互执行规则：每次因报错、代码修复或参数修改要求重跑，必须重新提供包含工作目录、环境变量和全部参数的完整可复制命令，不能只要求用户重跑上一条命令。
7. 🔴 **交互执行规则（最高优先级，2026-10-10 用户强调）**：**永远先改本地工程代码，再 cp 同步到服务器；严禁直接修改服务器代码。** 可写编辑区唯一为本地 checkout `/media/manu/1TB-Volume/workspace/ultralytics/`（含 `manu/` 与 `manu/memory/`）；服务器 SSHFS 挂载目录 `/home/manu/mnt/pycharm_project_10ae9e2e/` **一律只读，禁止 Edit/Write**；`/run/media/manu/SSD960-LOCAL/nfs/`（板卡 NFS 挂载的 host 导出目录，**2026-10-14 由 `/home/manu/nfs` 换址而来**）是**数据交换区，不是编辑区**。强制三步：**本地编辑 → 本地校验 → cp 同步 + md5 双侧比对**。⚠️ 本日实测事故：Agent 曾把三处 memory 编辑直接打在 SSHFS 挂载盘上，本地一行未变，造成**本地与服务器静默分叉且无报错**。分叉修复必须**以本地为准重做，严禁把服务器版本 cp 回本地**。完整规范见 `rules.md` 第〇节。
8. ~~**交互执行规则（2026-10-10 新增）**：每次被要求「读 memory / 了解背景」时，第一步必须先校验三条挂载链路~~ —— 🔴 **已于 2026-10-12 经用户要求废止**。memory 的权威副本在本地 checkout（`/media/manu/1TB-Volume/workspace/ultralytics/manu/memory/`），读 memory 不再依赖任何 SSHFS/NFS 挂载，因此「读前必校验挂载、缺失即自动补挂」这条前置闸门**整体撤销**，不再作为任何操作的前置条件。
9. 🔥 **业务域攻坚方向已转向 §9 双专家 MoE Adapter**（2026-10-13 定稿，用户定「后续有时间再实施」）。实施前**先裁决三个硬冲突**（铁律二缓存豁免 / 铁律六零初始化门控 / 挖掘口径对齐），并建议 Step 0 先做 FN/FAR 拆解确认 +14.5 F1 点分布。
10. 🔥 **交互执行规则（2026-10-14 用户强调，token 预算充足）**：**可并行的独立子任务必须委派子 Agent 并行执行，主 Agent 只保留结论与整合。** 全量扫描、跨文件查漏、多机核对、一致性审计这类「只读取证」工作若在主上下文里做，会把成万行内容灌进上下文挤掉后续推理空间。判定标准：需读 ≥3 个文件、是「还有没有别处残留」这类穷尽性问题、与当前工作无依赖可并行、或有可验收产物。角色：只读取证 → `explore`（禁改文件）；范围内改动 → `worker`（须声明独占文件）；独立复核 → `verifier`；🔴 **并行写同一文件禁止**（须串行或拆文件集）。**主 Agent 保留**：范围界定、方案决策、对结论负责、对外交付；子 Agent 说「完成」**不等于**事实成立，必须复核后才可写入 memory。委派 prompt 必须自包含（目标、**已确认事实与判定基准**、范围、约束、输出格式）—— 缺判定基准会让子 Agent 把合规的历史记录误判为待修项。完整规范见 `rules.md` §一·8。
11. 🔴🔴 **交互执行规则（最高优先级，2026-10-16 用户强调，原话「我让你做什么你就做什么，绝对不允许自作主张跑别的任务」）**：**只做本轮用户明确交代的事，任务完成即停。** 🔴 **禁止**擅自启动任何未被要求的实验、测量、压测、补测、benchmark（包括板卡/服务器跑批）；🔴 **禁止**把「我发现的缺口 / 待办 / 可优化点」自动升级为待执行任务 —— 发现缺口只能**记下来并在回复里问用户**，然后停下等回答；🔴 **禁止**把「目标里写着 X」当作执行 X 的授权。**判断某件事「值不值得做」不是 Agent 的权限。** 拿不准是否越权 ⇒ **默认不做，先问**。⚠️ 本日实测事故：用户在「只更新 memory」的指令完成后，Agent 因自行判断「还有一项没测」而擅自上板卡跑 fp16 e2e 帧率实验并留下产物目录 —— 技术上「有价值」但仍是越权。验收类只读检查（确认自己刚写的文件在位、md5 是否一致）属交付的一部分**不算越权**；任何改动外部状态的动作都不算。与铁律四/七 不冲突：自动写入 memory 的是**已授权执行过的**实验事实；未授权的一律记为「未测 / 待裁决」而非顺手做掉。完整规范见 `rules.md` 第〇节之三。

## 9. 双专家 MoE Adapter 业务域攻坚方案（2026-10-13 定稿，**未实施**）

> 完整方案、训练细节与风险登记见 `manu/memory/dual_expert_moe_adapter.md` 与 `memory.md` §十二。
> **状态：代码未实现，零实测指标。不得当作已验证方案引用或对外宣称收益。**

### 9.1 方案骨架

- **动机**：只打业务域两大痛点 —— **路灯/建筑反光虚警 (FP)** 与 **近距大目标多峰破碎漏检 (FN)**；
  外挂旁路、底模 100% 冻结，**不改底模卷积与内存排布** ⇒ `gmc_stream_ocl.cpp` 等已有底层加速算子**不受影响、不作废**。
- **架构**：`Expert_Large`（3×3Conv+BN+ReLU+1×1Conv+Sigmoid，≈12k）输出 `Δ_pos∈[0,1]` 正向热图增量，
  把近距大目标弥散峰值从 **0.08~0.15 拉到 0.40+**；
  `Expert_Suppress`（3×3 **DilatedConv(d=2)**+BN+ReLU+1×1Conv+Sigmoid，≈15k）输出 `M_gate∈[0,1]` 空间门控，
  用扩张卷积抓「双翼路灯对称高亮 + 细长灯杆 + 建筑刚性边缘」的空间指纹，把 0.60 虚警衰减到 **0.03 ≪ 0.22**。
- **确定性物理融合**：`H_final = (H_base + Δ_pos) × (1 - M_gate)`；平坦纯天空区 `M_gate→0`、`Δ_pos→0` ⇒ 严格还原 `H_base`。
  推理留 `--disable-adapter` 逃生开关，100% 恢复底模。
  🔴 **实现必须钉死**：融合发生在**抽峰之前**，`H_final` 替代 `H_base` 进入现有抽峰/连通域/多尺度/Track 全链路，
  **下游门限 `th_base=0.22` 等一律不变**；若改成抽峰后压制，则已冻结的交付参数口径全部作废须重标。
- **数据**：三桶切片挖掘 —— 硬负样本（GT 空但峰值 ≥0.22）100% 灌入；漏检大目标（框面积 >100 px² 但框内响应 <0.22，目标为宽核高斯）100% 灌入；
  黄金锚点（峰值 ≥0.50 且 IoU 良好）降采样 10% 混入防漂移；收敛到 **2,000~3,000 帧**。
- **增强**：X/Y 各 ±30~60 px 随机平移（破解「死背路灯绝对坐标」）+ 50% 随机水平翻转。
- **损失**：`L = L_det(Modified Focal) + λ·L_silent`，`L_silent = mean(M_gate² + Δ_pos²)`，`λ = 0.05~0.1`；
  物理含义是**无梯度驱动时门控必须停在 0**，不在没把握的背景上乱放抑制或虚假补偿。
- **验证**：🔴 **序列级封存** 1~2 条业务视频（1 帧不进训练池）后再挖 Mini-Golden Val（~1,500 帧，三段各约 500 帧）；
  **严禁从训练切片库抽 20% 当 Val**（前后帧泄漏，指标虚高至 99%）。
  三大物理探针一票否决早停：`P_suppress<0.10`（路灯消音）、`R_large≥90%`（大目标召回）、`S_anchor≥0.95`（天空锚点零误伤），预期 5~8 Epoch。

### 9.2 🔴 实施前必须裁决的三个硬冲突

1. **踩铁律二缓存红线**：拟缓存 FP16 P0 密集特征 ≈ **3.9 GiB** + Val **1.8 GB**，而铁律二上限 **100 MB** 且明令禁止缓存高维稠密特征。
   数量级比原动机小约 10×，但**「可控」≠「合规」** ⇒ 需用户**显式豁免**（建议写死 8 GiB 上限）或改设计（int8 / 降分辨率 / 在线前向）。
2. **缺铁律六的零初始化恒等门控 + Epoch-0 零回归校验**：两条支路都带 BN 却未声明恒等门控。
   🔴 且**末层 weight/bias 全零初始化是错的** —— 全零 bias 经 Sigmoid 恒输出 0.5（非 0 且梯度被截断）；
   正确做法是 bias 设为使 Sigmoid 输出 ≈0 的负值 / 末层接 `α=0.0` 门控 / `ReLU+α`；BN 全锁 `eval()`。
   验收标准是 **Epoch 0 实测 `Δ_pos≡0 且 M_gate≡0`**。可复用 `manu/training/train_trial0474_wh.py` 已跑通的冻结与对账模式。
3. **挖掘口径 ≠ 评测口径**：挖掘用单尺度 `0.22`，评测大盘是多尺度 `scales 160,80 / th 0.30 / r=64 / region-th 0.06`；
   挖出的硬负样本未必是大盘真实 FP 来源 ⇒ 必须复用 `manu/data/multiscale_heatmap_cache.py` 既有算子（推理链与评测大盘同一份代码），
   并在 Step 0 统计桶内 FP 与大盘 FP 一致率。

### 9.3 ⚠️ 覆盖面不足的自陈

业务域 F1 73.58% → 判据 88%，缺口约 **+14.5 F1 / +16 R**。本方案只覆盖路灯虚警与近距大目标；
`00006_000322`（头号事故 FN 6,847）、`00036_015740`（F1 24.03%）属能力缺口，`00032_014453` 属黑热截断，**均不在射程内**。
⇒ 即便完全成功也不保证达标，**不得把「路灯消音 + 大目标召回」等同于「F1 ≥ 88%」**；建议 Step 0 先做 FN/FAR 拆解（`fn_absent`/`fn_compete`）。

### 9.4 长期演进阶梯

交付期 = 单一双专家 Adapter；容量饱和 = **保持外挂接口不变**，内部升级为软路由 **MoA（Mixture of Adapters）**，
按地表特性扩原型分支（人造强反差刚性专家 / 自然杂波海浪专家）+ 轻量 Router；
数据质变 = 把 Anti-UAV 作为联合监督集之一，启动下一代大主干联合重训，**把 Adapter 能力内化进主干**。

### 9.5 🔴 指标口径政策变更（覆盖全文旧表述）

- **Anti-UAV「不可退化闸门」自 2026-10-13 起不再作为硬约束**，理由：0.9064 浮点级零退化假设非合同硬约束。
- **新口径**：交付核心指标 = **业务域综合效果**；Anti-UAV 降级为**防崩底板**，容忍单帧 F1 **0.82~0.88**、系统级 F1 **0.88+**。
- ⚠️ **降级 ≠ 免测 ≠ 免报**：每轮仍须测量并如实上报 Anti-UAV 对照数字；跌破下限属「未达标」而非「豁免」。
- 全文出现的旧「不可退化闸门」表述（本文 §7 前一句、`memory.md` §十.9/§十.10/§十一.3）已就地标注废止，以本条为准。

## 10. SOTA 零漂移复现 + P0 Highway 消融（2026-10-14 实测）

> 口径：官方 val **31,613 帧 / 25,111 GT**、`conf 0.08 / top_k 80`、`th=0.25`、**Strict Distance ≤ 8.0px**；
> 两臂共用**同一 checkpoint** `trial_0474/weights/best.pt`。完整证据见 `memory.md` §十三。

- ✅ **SOTA 零漂移复现**：TP **21,643** / FP **1,004** / GT **25,111** 三项逐个对齐，F1 **0.906361**，
  与 `results.csv` epoch 3 逐位互证 ⇒ `0.9064` 就是该冻结标称。
- **P0 消融**（`--ablate-p0`，训练后移除）：F1 **0.905736**、TP 21,624、FP 1,014，
  **ΔF1 仅 +0.000625**；P0 支路共 **4,418 参数 = 总量 0.25%**。
- **判决**：跨 15 档门限，**ΔTP 15/15 全为正**（均值 +18，召回增益系统性），但 **ΔF1 均值仅 +0.00033 且 2/15 档为负**
  ⇒ 按 §三·6 噪声纪律（±3 TP ≈ ±0.0002，结论须 > 2×）**属边缘、不可作有效结论**。
- 🔴 **决策（2026-10-14 用户裁决）**：**去 P0 臂 = RK3588 移植的参考模型（ref model）与交付对象**。
  三条依据：① 精度代价 ΔF1 **−0.000625**，距 Anti-UAV 防崩底板（单帧 0.82~0.88）绝对余量 > 0.026；
  ② 板卡 `rknn_run` **103.7 → 55.5 ms**，且 **Conv MAC 只降 6.8% 而时延降 46.5%**
  （p0 那几层 640×640/C=16 是 **NPU 最劣形状**，§四·9f 禁止用 MAC 外推时延）；③ **0 CPU 计算算子 / 0 fallback**。
  ⚠️ **两个代价不可相加**：上表是 **CPU fp32 全量**口径；int8 的独立代价另测为 **−2.27 pp 召回、配对 McNemar p = 0.0129**。
  去 P0 臂**未为 int8 重新标定或重训**。
- ⚠️ **两条必须随消融结论一起引用的局限**：①这是**训练后移除**，不回答「从一开始就不用 P0 会不会一样好」，
  后者须走 **Twin A/B 冷启动**；②同 `th_base=0.25` 下去 P0 臂的**系统级后处理**反而 **+0.0014**
  （0.917706 vs 0.916324），但 0.25 非交付档（交付档 0.22）且后处理对候选池分布**非单调**，
  **仅登记为待查现象，不得据此宣称「去 P0 更好」**。
- 🔴 **新踩坑**：可选组件的开关检查**不能用 `hasattr`** —— 属性值为 `None` 时 `hasattr` 仍返回 `True`，
  本轮据此在消融臂上触发 `AttributeError`（只在该臂才暴露）。一律用 `is not None`。

## 11. Anti-UAV val 子集端到端精度+速度验证（2026-10-14 实测，**推翻了 §10/§十五 的两条单序列结论**）

> 板端 = RK3588 端到端（Mali GMC 特征 + RKNN int8 去 P0 热图头）vs **CPU 权威臂**（FP32、冻结特征、th=0.25、ablate-p0）。
> CPU 权威臂跑完全量 31,613 帧（双侧 md5 一致）：P 21,624/22,636 = 0.9553 [0.9525, 0.9579] / R 21,624/25,111 = 0.8611 [0.8568, 0.8654] / F1 0.9058。

- 🔴 **范围＝子集**：板卡串行 164 ms/帧、全量需 ~86 min ⇒ 只跑 **10/24 序列、10,641/31,613 帧 = 0.3366 [0.3314, 0.3418]（33.7%）**
  （512×512 4,682 / 640×512 5,959 帧；角点 12.2~600.0 全量程；png 3,656 / jpg 6,985）。
  **所有板端结论只在这 10,641 帧上成立，剩余 20,972 帧未测。**
- ✅ **L1 特征对齐通过**：帧账目硬断言 期望 10,641 = 实测 10,641 = 跳过 0；**png 无损容器 identical_frac mean 0.9674**
  （min 0.8683 / max 1.0000）。jpg 有损 0.5558 是**容器压缩误差、不是错帧**。
  `push_offset` **实测 10/10 序列 k=1**（原为单序列假设）；⚠️ **jpg 不可用 0.9 绝对门槛**（会 20/20 全误判 FAIL），改用「最优/次优差距 ≥ 0.05」。
- 🔴 **精度显著单侧劣化**（同批 9,375 GT）：F1 **0.8869 → 0.8601（−2.68 pp）**、
  recall **7823/9375 = 0.8345 [0.8268, 0.8418] → 7392/9375 = 0.7885 [0.7801, 0.7966]（−4.60 pp）**，
  而 **precision 零损失 −0.04 pp**（**7823/8266 = 0.9464** [0.9413, 0.9511] → **7392/7814 = 0.9460** [0.9408, 0.9508]）；劣化全来自漏检（FN 1552→1983），误报反降（FP 443→422）。
  L3 配对 10,641 帧：不一致 **515/10,641 = 0.0484 [0.0445, 0.0526]**、NPU 独中 42 / CPU 独中 473
  ⇒ **McNemar p < 0.0001，统计显著**。**单序列小样本优势不能外推**（`34_1` 的「虚警 72→49」红利在 10 序列上消失）。
- 🔴 **40 ms 实时红线 6/10 序列不达标（推翻「PASS」）**：512×512 p50 20.995 / p99 59.431 / max 63.964（n=4,104）；
  640×512 p50 26.574 / p99 63.117 / **max 132.601**（n=5,567）。**p50 通过、p90/p99 普遍不通过。**
  此前「PASS」只在 **`34_1`** 测得，而 `34_1` 的 **std 仅 0.47 ms 是全子集最稳序列 —— 那是运气**。
  **尾部不由角点均值决定**（`34_1` corners 267/std 0.47 vs `43_1` 245/std 9.10，角点几乎同而 std 差 19 倍）
  ⇒ 尾部由**逐帧角点尖峰**驱动。治理：❌ **不得用全局 clamp**（会污染 L1 对齐），应做**逐帧自适应角点预算**（未实施）。
- **NPU 阶段恒定 56.8~57.2 ms，与分辨率无关**（模型输入恒 640×640）⇒ **17.6 fps**；
  端到端算力 77.9 ms(p50)/116.3 ms(p99) @512×512，83.5/120.0 @640×512；
  **实测墙钟 164 ms/帧**，差额 **~87 ms** = NFS 特征往返（GMC 写 ~31 GB + NPU 读回 ~31 GB），同进程/共享内存可消除（未验证）。
- **新增 5 个工具**（`manu/pipeline/npu/`，md5 双侧一致）：`hm_stage_val_bmp.py`（原始帧→8bit 灰度 BMP 暂存，幂等，
  必须落 `raw/<seq>/` **子目录**）、`hm_val_prepare_links.py`（板端软链，**必须相对路径**）、`hm_val_board_loop.sh`
  （板端逐序列 GMC+NPU，序列级 checkpoint）、`hm_val_verify.py`（偏移实测 + L1 分层 + 分辨率分层速度）、
  `hm_val_report.py`（L2/L3 编排 + 4 道硬断言）。
- 🔴 **7 条新坑（可复用）**：①GMC 输入必须落 `raw/<seq>/` 子目录，否则 `sequence folder not found`；
  ②`| tail; echo $?` 取到的是 **tail 的退出码**，把 FATAL 伪装成成功；③stdout 块缓冲（8 KB≈285 行≈滞后 7,000 帧）
  使 CPU 臂**看似卡死实则满载**，判卡死看进程不看日志；④**帧数对账不足以发现 id 错位**（`--push-offset 2` 账目齐平但逐帧全错）；
  ⑤**按文本猜字段会静默失效**（`hm_score_board_det.py` 无 "missing" 字样 ⇒ 正则得 `None` ⇒ `if miss:` 恒假、校验永远通过）；
  ⑥**L2 两行分母必须相等**（`layer2_det` 不过滤 perframe，子集分析须先把 CPU 参考裁到同帧集）；
  ⑦`missing == |ref stems| - |board stems|`，写成「必须为 0」会**静默禁止子集分析**。

## 12. 板卡访问免密化 + 串行两段墙钟重新分解（2026-10-14）

- 🔴 **板卡 SSH 已配公钥免密 ⇒ 不再需要 `RK3588_PASSWORD`，不得再向用户索要板卡口令**：本机 `id_ed25519.pub` 已入板卡 `/root/.ssh/authorized_keys`，免密实测 `KEY_AUTH_OK`（内核 5.10.160 / `evm3588`）；`board_run.py` 已改为优先免密，仅公钥缺失时才回落 pexpect + 环境变量。**凭据仍不落盘**——用户曾要求写进 memory，已拒绝并改为免密：**正确解法是消除口令依赖本身，不是记住口令**（写进去既救不了失效凭据，又把设备凭据散进会被 `cp` 到服务器的 207 KB 知识库）。
- `board_run.py` 免密分支两个 rc 缺陷已修：`; __rc=$?` 拼在以换行结尾的命令后 ⇒ **bash 语法错误、数据照打但 rc 丢失**；命令内 `exit N` ⇒ 远端 shell 提前死、包装行没机会输出 rc。改为**子 shell 包裹** `( cmd )`，4 项回归（`exit 7→7` / `false→1` / `true→0` / 多行 for）全过。
- **串行两段墙钟 183 ms/帧**（原 §11 的 164 ms 系总量倒推）：按 `all_run.log` 起始时间 + 各 `npu_det.tsv` mtime 串成串行时间轴拟合 `段墙钟 = g·GMC帧 + n·NPU帧 + c` ⇒ GMC 67.2 ms/帧、NPU 116.1 ms/帧、每序列常数 28.0 s，**R²=0.9944、ΔAIC=11.2**。全量外推 **~97 min**。⚠️ 拟合**必须剔除 `34_1`**（TSV 是旧 demo 遗留，会产生 −4,814 s 负段）。⚠️ **该拟合的 `g`/`n` 已被下条实测证伪，实测值优先。**
- 🔴 **「放大全在 NFS 特征往返、提速 2.5×」已被板端 tmpfs 对照实验否决**：`DJI_0175_2` 448 帧三臂对照（NFS 冷 75,684 ms / tmpfs 72,801 ms / NFS 冷 75,693 ms ⇒ 169.0 / 162.5 / 169.0 ms/帧），**搬到内存只快 6.5 ms/帧（3.8%）**。三臂 TSV md5 全等 `8b8de575fadd92c1e93cefe50996f3a3` 且与循环产出逐字节相同（实验有效）。
- **真正的缺口 = NPU 程序自身 CPU 空转**：50 帧/448 帧两点解出**斜率 168.2 ms/帧、截距仅 433 ms**（模型装载可忽略），而程序自报仅 57.1 ms/帧 ⇒ **≈111 ms/帧 未计时**，其中 letterbox 预处理实测 **30.69 ms/帧**（`512x512 -> 640x640`）。⇒ **NPU 在 168 ms 里只忙 57 ms，空闲率约 66%**；⚠️ **模型输入恒为 640×640**，512 源被上采样是纯浪费。**优化靶子应是预处理/后处理，不是 I/O 也不是 NPU**；余下 ≈74 ms（BMP 解码+后处理+落盘）需在二进制内加分段计时才能拆分。
- 宿主 NFS 读 0.94 MiB 冷 2.285 ms / 热 0.107 ms（≈400 MB/s），板卡经 NFS 只剩 ≈15.8 MB/s，差 25 倍 ⇒ 慢在**板卡侧 NFS 客户端**，非链路、非宿主页缓存（但该 I/O 本身只占 3.8%，不是主要矛盾）。
- 板卡资源：`Mem 7909 MB / available 7350 MB`、`/dev/shm` tmpfs **3.9 G**、**无 swap**、root 可 `drop_caches`（⇒ 冷热可强制切换，page cache 不再是混杂因素）。

## 13. NPU 段每帧耗时完整归因 + 根因修复（2026-10-14）

- 🔴 **根因：batch 模式每帧把 NPU 推理跑了两遍**——第 567 行那次产出检测但**不计时**，第 613 行计时块（`loops=0` 默认 ⇒ `n=1`）**再跑一遍、结果直接丢弃**只为填 `[7]` 报告。每帧白白多跑 ≈54 ms `rknn_run`，而报告只统计其中一遍 ⇒ **168 ms/帧的墙钟看起来像 57 ms/帧**。「NPU 空闲 66%」的真相在此，**不是预处理慢**。
- **修复**：`t_stage` 扩到 4 槽并新增 **`dequant`** 计时（原先在 `a0..a3` 窗口外从未计入）；batch 分支改为直接给产出检测那次推理计时；计时块前置 `if (batch_list) continue;`，**单图 `--warmup/--loops` 基准行为保留**；补齐 `stbi_load`/`letterbox`/minmax/`extract_peaks`/`det write` 计时；修掉一处**误导性 printf**（letterbox 源尺寸被写死成 640×640，显示成 `640x640 -> 640x640`，掩盖了真实缩放）。
- **结果位精确、零代价**（重复推理产出被丢弃 + `[5] determinism` 已证确定性）：两条序列 TSV 与修复前**逐字节相同**（`DJI_0175_2`=`8b8de575fadd92c1e93cefe50996f3a3`、`02_6321`=`c5c150bf98132a736fc45b504f4d8f08`）。**墙钟 168.2 → 100.3 ms/帧，1.68×**。
- **完整分解（绑定分辨率，§四·9h）**：512×512 = `rknn_run` 53.9 / `letterbox` **25.2** / `stbi_load` 14.1 / dequant 2.7 / minmax 1.3 / peaks 0.6 / write 0.03 ⇒ **真实 98.8 ms，host 占 41.8%**；640×512 = `rknn_run` 54.8 / `stbi_load` 17.6 / **`letterbox` 仅 0.2** / minmax 3.4 ⇒ **真实 80.4 ms，host 占 27.2%**。
- 🔴 **「低分辨率更快」在本流水线不成立**：512×512 → 640×640 是 **`r=1.25` 上采样**（letterbox 真缩放，25.2 ms），640×512 → 640×640 是 **`r=1.0` 纯补边**（letterbox 0.216 ms，差 117 倍）⇒ **512 序列反而更慢**。任何速度结论不绑定分辨率必错。
- **后续靶子**（按实测收益，均未实施）：letterbox 上采样 25.2 ms（改 RGA/GPU 或免上采样）＞ `stbi_load` 14~17.6 ms（让 GMC 直出 NHWC buffer，免「写 BMP→读→解码」）＞ dequant 2.7 / minmax 1.3~3.4。

## 附. 详细资料导航

- `manu/memory/memory.md`：项目总索引、完整演进路线和当前结论
- `manu/memory/rules.md`：实验、数据、缓存、服务器和同步铁律
- `manu/memory/dual_expert_moe_adapter.md`：**双专家 MoE Adapter 业务域攻坚方案 + 三个实施前硬冲突登记**（§9 的完整出处）
- `manu/memory/gmc_median_sota.md`：GMC、21 帧中值和 Trial 0474
- `manu/memory/heatmap_npu_trial0474.md`：**Trial 0474 热图检测器 RK3588 NPU 移植**（2026-10-13/14；**当前交付 = v3 去 P0 Highway**：int8 模型 3,799,304 B、**NPU 阶段** 56.7 ms / 17.7 fps / 0 个 CPU 计算算子 / 较 v1 累计 2.69×、**板端全链路端到端 88.852 ms / 11.25 fps（512×512）**；600 帧分层 P 402/419 = 0.9594 [0.9360, 0.9745] / R 402/490 = 0.8204 [0.7840, 0.8519] / F1 0.8845，缺口主要是 DJI 域弱点；int8 的 −2.27 pp 召回代价经配对 McNemar p=0.0129 判为显著）+ **端到端串行 demo 的三层精度对齐法**（L1 GMC 逐像素平均 98.75% 位相同 ⇒ L2 差异不得归因于特征合成；L3 McNemar p=0.0312 判为显著一侧性劣势）与 **`34_1` 贴边目标弱点**（6 帧不一致帧 GT x 全落在画布前 12.6%，待裁决）+ 八条新规 §三·10 / §三·6c / §四·2e / §四·2f / **§四·9e（int8 上板 0 CPU 计算算子，用户铁律）** / **§四·9f（禁止用 MAC/FLOPs 外推 NPU 时延：实测 MAC 降 6.8% 而时延降 46.5%）** / **§四·9g（验收输出不得靠 grep/sed 过滤判成败，0 字节产物当失败）** / **§四·9h（预处理时延随分辨率线性增长，须绑定分辨率实测引用）**；手册见 `pipeline/npu/CONVERT_HM.md`
- `manu/docs/experiments/2026-10-16_npu_fp16_vs_int8.md`：**NPU fp16 vs int8 受控对照**（§22 的完整出处）。全量 31,613 帧：fp16 臂 F1 **0.901401**（th 0.25）/ R **85.81%**，int8 臂 F1 0.882231（th 0.20）/ R 82.90% ⇒ **fp16 挽回 +0.019171（15 档负档仅 1/15），对 PC-FP32 仅 −0.000533，挽回率 97.3%**；代价 `rknn_run` 55.3→104.4 ms。含构建配方、CPU 算子验收、tensor 级对比（fp16 heatmap max_abs 7.1e-03 vs int8 1.9e-01）、P=2 并行实测、与板卡既有 `fp_no_p0.rknn` 的逐字节等价性补测
- `manu/memory/falsified_archive.md`：失败实验与工程避坑档案
- `manu/memory/spatio_temporal_track.md`：双向时空平滑系统
- `manu/memory/hard_cases.md`：难例诊断与攻坚记录

## 11. 无损 PNG 参考基线（2026-10-14，为嵌入式 OpenCL GMC 对齐而建）

- **产物**：`/mnt/data/siping/datasets/manu/uav_gmc_median_png/`（`images/val` 全 PNG + `labels/val` 软链 + `data.yaml`），
  **31,613 文件 / 7.55 GB**；转换脚本 `manu/data/jpg_features_to_png.py`，逐文件做「解码→写→重解码→`array_equal`」闸门。
- 🔴 **官方 val 是混合格式**：**27,957 JPG（20 序列）+ 3,656 PNG（`25_1`/`34_1`/`43_1`/`5_1`）**
  ⇒ **`F1 0.9064` 这个标称本身就是 JPG+PNG 混合的数字**（此前未记录）。
- ✅ **PNG 路径指标与 JPG 路径逐格相同**：`pred_points` / `pred_scores` 差异帧 **0 / 31,613**，15 档门限扫描每格一致。
  正确解读：转的是「JPEG 解码后的像素」，像素未变故输出必然逐位相同 ⇒ **基线由解码像素唯一确定、与容器无关**。
- PNG 路径两臂：SOTA（带 P0）**F1 0.906361 / TP 21,643 / FP 1,004**；去 P0（RK3588 参考模型）**F1 0.905736 / TP 21,624 / FP 1,014**。
- ⚠️ **只换了容器，没有从 raw 重算 GMC/中值**。逐算子对账仍须走 `cpp_port_rk3588.md` 的 G0/G1/G2 + `--anchor-step 2`，本节不替代它。
- 🔴 **本轮踩坑（脚本自身缺陷）**：初版只 `glob("*.jpg")`，**静默只转 27,957/31,613 却打印 `[SUCCESS]`**——
  那 4 条序列本来就是 PNG 被漏掉。**「无损数据集」建立在部分匹配之上比没有数据集更危险**；
  已加总数闸门（`matched != scandir 计数` 即 `exit`）并打印 M/N，源端 `.png` 改硬链接。

## 12. 官方特征构建脚本恢复（2026-10-15）—— 已升为铁律·基线冻结

- 🔴 **`manu/data/build_full_median_dataset.py` = 生成官方 `uav_gmc_median` 的唯一脚本**，是 Trial 0474
  全部指标（0.9064 / 0.9209）的输入来源，**严禁任何修改**（`rules.md` 第〇之二节铁律）。
  曾被暗残差实验改写成 4 通道版覆盖掉，**已从提交 `622b179cb` 逐字节恢复**（`diff` 校验一致）；
  原 4 通道版另存为 `build_full_median_signed_dataset.py`。
- **定位依据**：服务器 `~/.bash_history` 的当年调用 + `git log --follow`（原路径 `manu/build_full_median_dataset.py`）。
- ✅ **全量验收**：重建 31,613 张特征，**MD5 与官方 31,613 / 31,613 完全相同（100.0000%）**，标签全一致，0 失败。
- 🔴 **官方算法 = 逐 lag 直接拟合**（`compute_affine(im_h, im_curr)`），
  **不是**相邻帧仿射连乘 ⇒ **`build_fpv_ir_gmc_median.py` 不可用于重建官方特征**（实测 F1 掉 0.004）。
- ⚠️ 官方 val **JPG/PNG 混合**：20 序列 27,957 张 JPG + 4 序列（`25_1`/`34_1`/`43_1`/`5_1`）3,656 张 PNG
  （输出扩展名镜像自 ref 数据集），**不是 bug**。
- 🔴 **作废**：`uav_gmc_median_ctrl` / `uav_gmc_median_rawpng` / `uav_gmc_median_png`（错误构建器产物，不得引用）。

## 13. 本轮收口总结（2026-10-14 ~ 10-15）

### 13.1 三件已验收的事实

1. **SOTA 零漂移复现**：TP 21,643 / FP 1,004 / GT 25,111 三项逐个对齐，F1 **0.906361**，
   与 `trial_0474/results.csv` epoch 3 双路互证。
2. **P0 Highway 消融**（`--ablate-p0`，默认关）：去 P0 F1 **0.905736**，**494/494 非 P0 张量在所有
   P0-NAS trial 间逐位相同** ⇒ 底模全程物理冻结，去 P0 臂即 0474 自身底模，不是另一个模型。
   P0 贡献 **ΔF1 +0.000625 / 4,418 参数（0.25%）**；跨 15 档门限均值仅 +0.000326 且 2/15 档为负
   ⇒ 按 §三·6 **属边缘，不可作有效结论**。
3. **官方构建脚本恢复 + 全量验收**（见 §12）：特征 **31,613/31,613 MD5 相同**、
   cache 级差异帧 0、指标级全部零偏差。

### 13.2 冻结数字（引用须带口径：`th=0.25`、Distance ≤ 8.0px、`conf 0.08 / top_k 80`、官方 val 31,613 帧）

| 口径 | R | P | F1 | TP | FP | GT |
| :-- | --: | --: | --: | --: | --: | --: |
| **0474 带 P0** | 86.19% | 95.57% | **0.906361** | 21,643 | 1,004 | 25,111 |
| **0474 去 P0 = RK3588 ref** | 86.11% | 95.52% | **0.905736** | 21,624 | 1,014 | 25,111 |

### 13.3 ✅ raw→PNG 干净对照已完成（2026-10-15）—— **存储格式代价 ≈ 0**

**设计**：官方构建器**零改动**，只改 `--raw-root`（`anti-uav` 原 JPG vs `anti_uav_png_flat` 原图 PNG）。
A 臂 / E 臂两臂 G2 闸门均 `[VERIFY PASSED] 31613/31613`、Fail 0（643.7 s vs 651 s）。
**A 臂零漂移**：SELF-CHECK 复现 F1 **0.9064** / TP 21,643 / FP 1,004，与 §13.2 冻结值逐项一致。

| 臂 | 峰值 th | F1 | R | P | TP | FP |
| :-- | --: | --: | --: | --: | --: | --: |
| A 官方 JPEG | 0.20 | 0.916794 | 89.45% | 94.03% | 22,461 | 1,427 |
| **E raw→PNG 干净重建** | 0.20 | **0.916816** | 89.45% | 94.03% | 22,462 | 1,427 |
| | | **ΔF1 +0.000022** | | | | |

跨 15 档（0.10~0.80）：**ΔF1 均值 −0.000038**，区间 **[−0.000597, +0.000291]**，负档 6/15。
🔴 噪声下限 ≈ **±0.0002** ⇒ **15 档全部落在噪声量级内、正负双向分布 ⇒ 无系统性差异**。

**三条判决**：
1. ✅ **raw→PNG 无损化零代价** ⇒ **ARM 侧「读原图 PNG 现场算 GMC+中值」可放心采用**。
2. 🔴 **正式结清旧悬案**：「换输入源掉 0.006 F1」**归因证伪**。干净对照只差两个数量级且方向相反
   ⇒ 旧差异真实来源是 **①构建器用错（连乘 vs 逐 lag 直接拟合，实测占 −0.0040）②旧口径 th 起点不同**，
   **与存储格式无关**。
3. ⚠️ **本节未覆盖「去 P0 臂」**（当前唯一待办）；且未重跑「原图转 PNG」步骤（复用 10-14 暂存，
   已用像素 md5 证明与原图一致，结论等价但流程省略）。
   完整记录：`manu/docs/experiments/2026-10-15_rawpng_clean_rebuild_ab.md`

**官方算法仍是逐 lag 直接拟合**（`compute_affine(im_h, im_curr)`），**不是**相邻帧仿射连乘
⇒ **`build_fpv_ir_gmc_median.py` 不可用于重建官方特征**。
作废产物：`uav_gmc_median_ctrl` / `uav_gmc_median_rawpng` / `uav_gmc_median_png`。

### 13.4 仍然有效的两条环境事实

- 🔴 **官方 val 与 raw 都是 JPG/PNG 混合**：raw 27,997 jpg + 3,664 png；官方 val 27,957 jpg + 3,656 png
  （`25_1`/`34_1`/`43_1`/`5_1` 四条原图即 PNG）⇒ **「官方 0.9064」是混合格式数字**，引用时须说明。
- **帧映射**：val `<seq>__<i>` ⟷ raw `<seq>/<i>`，`i` 从 **3** 到该序列图像末帧
  （`31,661 − 24×2 = 31,613`）；raw 另有 24 个 `IR_label.json` 旁挂文件，**不是帧**。

### 13.5 下一步唯一待办

**用恢复后的官方脚本做干净 jpg→png 对照**：只改 `--raw-root`
（`anti-uav` vs `anti_uav_png_staging`），脚本零改动。两份原图暂存已就绪。
ARM 侧尚未开始，属后续独立工作。

---

## 14. ✅✅ ARM GMC 精度损失评估（2026-10-15）—— **板卡实机，容器对称全量对照：判定无回归**

> 评估对象 = `manu/pipeline/cpp/gmc_stream_ocl.cpp` 的 GMC 阶段。**板卡 aarch64 + Mali-G610 实机执行。**
> 详细版见 `memory.md` §12；实验记录 `manu/docs/experiments/2026-10-15_arm_gmc_precision_eval.md`

### 14.1 结论

**ARM GMC 未引入精度回归。** 唯一变量 = 「GMC 特征由谁生成」，全量 24 序列 / 31,613 帧 / GT 25,111：

| | th | F1 | P | R | TP | FP |
| :-- | --: | --: | --: | --: | --: | --: |
| PC-BMP | 0.25 | 0.902316 | 94.87% | 86.02% | 21,601 | 1,167 |
| ARM-BMP | 0.25 | 0.902116 | 94.84% | 86.01% | 21,598 | 1,174 |

峰值 ΔF1 = **−0.000201**（= **3 TP**，恰好一个噪声单位）；跨 15 档均值 **+0.000074**，
区间 [−0.000201, +0.000490]，负档 6/15 ⇒ **全部落在 ±0.0002 噪声地板内**。
两臂峰值门限同为 `th=0.25`，无工作点漂移。正确表述是「**两者精度等价**」。

### 14.2 🔴 三条必须随结论一起引用的修正

1. **旧结论「ΔF1 = +0.0047，15/15 档全正」已作废。** 根因：冻结构建器
   `build_full_median_dataset.py:272` 输出扩展名**继承自 ref 文件名**，旧 PC 对照用含 6,985 帧 JPG 的
   冻结 ref ⇒ 输出被 **JPEG q95 二次压缩**，而 ARM 臂是 BMP ⇒ 65.7% 帧上混入压缩误差。
   **修复不动构建器**：它只读 ref 的文件名与标签、从不读 ref 像素（`:268-272`），
   构造 `ref_bmpval`（全改 `.bmp` + 软链原图）即可强制无损。构建器 md5 全程 `7729e772...` 未变。
2. **索引映射是 `__<k+1>` 不是 `__<k+3>`。** 板卡 `<seq>_<k>` ↔ raw 第 k+1 个 ↔ 官方 `__<k+1>`；
   板卡**每序列会多产出前 2 帧**（板卡 31,661 = raw 总帧数），组装丢弃 `k<2` 的 48 帧后正好 31,613。
   **错位代价 7~10 倍**（MAE 0.18 vs 1.32）⇒ 不做像素抽样，后续所有 F1 都会是错的。
3. **像素校验不能用 `Max|D|` 单一指标判死刑。** 高运动序列单点最大差达 254 但整体 MAE 仅 0.06；
   正确判据是 **MAE + `>5` 像素占比**。

### 14.3 像素佐证（31,613 帧全量逐像素）

| 分组 | Max\|D\| | MAE | >5 比例 |
| :-- | --: | --: | --: |
| 低运动 25_1/34_1/43_1/5_1 | 2 / 2 / 3 / 5 | 0.000297 / 0.000332 / 0.000432 / 0.000749 | 0.000% |
| **24 序列合计** | **254** | **0.037025** | **0.1443%** |

低运动序列与 §13.3 历史记录逐项吻合 ⇒ 低运动下移植近乎位精确；
高运动序列 MAE 升高但 p99 仅 0~4，是 21 帧时域中值被 warp 末位浮点差异放大造成的
「中值换帧」，属跨架构浮点末位效应（呼应 `opencv_arm64_parity.md`）。

### 14.4 🔴 复现命令（铁律九登记项，缺此项本节视为不合格）

一键入口：`bash manu/evaluation/run_arm_pc_gmc_full_ab.sh [all|board|xfer|assemble|infer|ab|pix]`
（md5 `29d7c1b1e0a0a7a5a52453b5f637d9bf`）；完整参数见实验记录 §5。

```sh
# 1 板卡（root@192.168.0.64）：脚本已固化在仓库 run_step2_board.sh
#   可执行体与板卡 /root/run_step2.sh 逐字节一致（md5 4ebde86e...，21 行）
scp -P 22 manu/pipeline/npu/run_step2_board.sh root@192.168.0.64:/root/run_step2.sh
ssh -p 22 root@192.168.0.64 'chmod +x /root/run_step2.sh && \
    nohup /root/run_step2.sh > /mnt/manu/armfeat_s2.log 2>&1 &'
#   清单 /mnt/manu/npu/out/demo/expected.tsv（24 行）；内部逐序列调用：
#   /root/gmc_stream_ocl --raw-root raw --sequence "$seq" --anchor-step 2 \
#     --out-dir /mnt/manu/armfeat_s2/$seq/feat --out-ext bmp \
#     --fused gpu --kernel-dir /root/kernels --timing      # -> 31,661 帧，约 75 min

# 2 传输（host，6 流并行；单流 13.8 MB/s -> 6 流 81 MB/s）
split -n l/6 -d /tmp/arm_bmp_list.txt /tmp/armpart_
for p in /tmp/armpart_*; do rsync -az -e "ssh -p 32222" --files-from="$p" --relative \
  /run/media/manu/SSD960-LOCAL/nfs/armfeat_s2/ \
  huangzhe@192.168.99.40:/mnt/data/siping/datasets/manu/armfeat_stage/ & done; wait

# 3 无损 ref：冻结 ref 全改 .bmp + 软链回原图，标签硬链
#   （构建器只读文件名与标签，从不读 ref 像素）

# 4 PC 臂无损重建（冻结构建器，零改动）
$PY manu/data/build_full_median_dataset.py \
  --ref-dataset /mnt/data/siping/datasets/manu/ref_bmpval \
  --raw-root   /mnt/data/siping/datasets/manu/anti_uav_bmp_all/val \
  --output     /mnt/data/siping/datasets/manu/armcmp_pcbmp --splits val --workers 16

# 5 ARM 臂组装（OFFSET=1，丢 k<2 的 48 帧）
$PY manu/data/assemble_arm_gmc_dataset.py \
  --stage /mnt/data/siping/datasets/manu/armfeat_stage \
  --out   /mnt/data/siping/datasets/manu/arm_gmc_full \
  --labels-from /mnt/data/siping/datasets/manu/ref_bmpval \
  --expect-total 31613 --verify-md5 120

# 6 双臂推理（同权重同口径）
$PY manu/inference/cache_trial0474_inferences.py \
  --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --data <DS>/data.yaml --output runs/gmc_eval/<NAME>.pkl \
  --imgsz 640 --stride 2 --batch 32 --device 0 --conf-thresh 0.08 --top-k 80

# 7 自校准闸门（必先跑，复现不出冻结 headline 即中止）
$PY manu/evaluation/eval_arm_pc_gmc_ab.py \
  --arm-a runs/gmc_eval/official_rebuild_trial0474_cache.pkl --label-a SOTA \
  --arm-b runs/gmc_eval/official_rebuild_trial0474_cache.pkl --label-b SELFTEST \
  --expect-f1 0.906361 --expect-tp 21643 --expect-fp 1004

# 8 A/B 评估
$PY manu/evaluation/eval_arm_pc_gmc_ab.py \
  --arm-a runs/gmc_eval/pc_bmp_full_cache.pkl  --label-a PC-BMP \
  --arm-b runs/gmc_eval/arm_gmc_full_cache.pkl --label-b ARM-BMP \
  --tol 8.0 --out-json /mnt/data/siping/datasets/manu/gmc_arm_vs_pc_ab.json

# 9 全量逐像素
$PY manu/diagnostics/cmp_arm_pc_features.py \
  --arm /mnt/data/siping/datasets/manu/arm_gmc_full/images/val \
  --pc  /mnt/data/siping/datasets/manu/armcmp_pcbmp/images/val \
  --samples 0 --out-json /mnt/data/siping/datasets/manu/gmc_arm_vs_pc_pixdiff.json
```

`$PY=/home/huangzhe/anaconda3/envs/uav/bin/python`（py3.10.20 / cv2 4.10.0 / numpy 1.26.4）；
项目 `/tmp/pycharm_project_10ae9e2e`；GPU RTX 4090 D。

⚠️ 板卡轮询**禁止** `ls .../*/feat/*.bmp | wc -l` —— 3 万路径展开进 argv 撞 `ARG_MAX`
**静默返回 0**，曾误判板卡产物清零。改用 `find ... -name '*.bmp' | wc -l`。

### 14.5 评估脚本自校准闸门（新增铁律级做法）

`manu/evaluation/eval_arm_pc_gmc_ab.py` 要求**先复现冻结 headline 才允许继续**：
`F1 0.906361 / TP 21,643 / FP 1,004` ⇒ **PASS**；自比（SOTA vs SOTA）15 档 ΔF1 恒为 `+0.000000`。
复现不了就不该往下走。同批新增 `manu/diagnostics/cmp_arm_pc_features.py`、
`manu/data/assemble_arm_gmc_dataset.py`（OFFSET=1 已写进脚本 docstring）。

### 14.6 ⚠️ 四条限定

1. **两臂 F1（~0.902）都低于冻结 headline 0.906361，差约 0.004** —— 这是**两臂共有**因素
   （本实验输入是 `anti_uav_bmp_all` 的 BMP 原帧，冻结集用的是原帧容器），A/B 中被抵消，
   **不影响 ΔF1**，但**绝对 F1 不得与 headline 直接比大小**。
2. **仅覆盖 `anchor_step=2`**；`anchor_step=10` 未测（用户指示后续再测），且该档 ≠ SOTA 冻结特征，
   **须另设参照系**。
3. **未覆盖去 P0 臂**（同 §13.5 限定 2）。
4. 板卡二进制 `/root/gmc_stream_ocl` md5 `aa99da811356694d66a0bce6de108b92`，
   ELF machine `0xb7` = **AArch64** —— 此前「x86 版不可执行」的记录**已过时**。

---

## 15. ✅✅ ARM GMC `anchor_step=10` 评估（2026-10-15）—— **e2e 实际配置，无精度损失**

> 第二轮。评估对象、参照系、口径、评估链路与 §14 **完全一致**，唯一变量是 ARM 臂特征
> 从 `anchor_step=2` 换成 `anchor_step=10`（e2e demo 实际参数）。PC 臂完全复用未重跑。
> 用户要求「带 bad case handle 的结果」——闸门本就是默认开启，本轮即含真实回退行为。
> 独立实验记录：`manu/docs/experiments/2026-10-15_arm_gmc_precision_eval_s10.md`
> 详细版见 `memory.md` §13。

### 15.1 结论

**三臂峰值 F1 落在 0.9021~0.9025 的 0.0004 窄带内，两两差异都在噪声下限（±3 TP ≈ ±0.0002）以内。**
**anchor_step 从 2 换到 10，在这份 val 上分不出差异。**

| 臂 | anchor_step | th | F1 | P | R | TP | FP |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| PC-BMP | — | 0.25 | 0.902316 | 94.87% | 86.02% | 21,601 | 1,167 |
| ARM-BMP | 2 | 0.25 | 0.902116 | 94.84% | 86.01% | 21,598 | 1,174 |
| **ARM-BMP** | **10** | **0.25** | **0.902513** | 94.91% | 86.03% | 21,603 | 1,159 |

| 对比 | ΔF1 | 折合 TP |
| :-- | --: | --: |
| PC vs ARM-step2 | −0.000201 | −3 |
| **PC vs ARM-step10** | **+0.000197** | **+2** |
| ARM-step2 vs ARM-step10 | +0.000397 | +5 |

跨 15 档均值 **−0.000078**，区间 [−0.000653, +0.000272]，负档 7/15，正负各半。

⚠️ 口径：本节 0.902 是**对 PC-BMP 臂**；旧记的 tree 臂 `0.906050 / ΔF1 −0.000311`
是**对冻结 headline 0.906361**。两口径**不可直接相减**。

### 15.2 🔴 e2e demo 确实用 `anchor_step=10`（三处独立证据）

1. `manu/pipeline/npu/run_hm_demo.sh:170-172` 调 `gmc_stream_ocl` **未传 `--anchor-step`** ⇒ 走默认
2. `manu/pipeline/cpp/gmc_stream_ocl.cpp:2217`：`int limit = 0, anchor_step = 10, ...`
3. 板卡按 demo 写法实跑，二进制自报 `[BUILD] ... anchor_step=10 ... mode=tree` +
   `[NOTE] mode=tree is the shipping arm; it is NOT equivalent to the frozen native set`

⇒ §14 的「移植无回归」是**精确等价档（step=2）**下的结论，不能直接外推；本节补齐了 e2e 实际档，结论同样成立。

### 15.3 ✅ bad case handle 默认开启，本轮即含真实回退

`run_step10_board.sh` 未传任何闸门参数，三层 handle 全默认开启。全量 24 序列触发统计：

| 层 | 默认 | 触发后 | 本轮实测 |
| :-- | :-- | :-- | :-- |
| `fitgate` 合理性闸门 | **开** | `W=IDENTITY` | 10 条序列触发，最多 `wg2022_ir_012_split_08` **116** 次；其余 13 条为 0 |
| `chain` 连乘链二次闸门 | **开** | `W=IDENTITY` | 最多 **74** 次 |
| `fitdeadline` 超时预算 | **关** | identity / hold | 全程 **0** |

01_4485 细节：`rejected=57/8838 (0.645%) [shift=13 other=44]` —— 多数不是「平移超标」
而是 scale/extent 类，印证 `scale×extent+translation` 不独立才是真凶。

**为什么坏拟合回退 IDENTITY 而非 clamp**（源码 `:641-668`）：实测出现过平移 **4863 px** 的
有限但物理荒谬解，`all_finite()` 放行 → 设备图 106 Mpixel/槽 × 45 槽 ≈ **4.8 GB** → 板卡
OOM killer 杀进程且**无断言无报错无非零 rc**。真实运动峰值仅 ~76 px。
注释原话：`Clamping the translation to the cap still yields a wildly wrong warp`。

**为什么默认关超时臂**（`:2047-2049`）：`OFF is deliberate: a wall-clock trigger makes the
output depend on machine load.` ⇒ 开了它精度对账失效。

### 15.4 附带：step=10 比 step=2 快约 3×

| 配置 | min | p50 | max | fits/frm |
| :-- | --: | --: | --: | --: |
| step=2 | 37.385 | **111.109** | 216.527 | 21.00（零连乘） |
| step=10 | 13.519 | **35.127** | 63.870 | 6.02 |

ms/frame，板卡全量 24 序列。**但仍受 §15.5 限定 3 约束，不得直接进嵌入式预算。**

### 15.5 🔴 复现命令（铁律九登记项）

PC 臂与评估链路全部沿用 §14.4，**只有 ARM 特征侧不同**：

```sh
# 1 板卡 step=10 生成（脚本已固化：manu/pipeline/npu/run_step10_board.sh）
scp -P 22 manu/pipeline/npu/run_step10_board.sh root@192.168.0.64:/root/run_step10.sh
ssh -p 22 root@192.168.0.64 'chmod +x /root/run_step10.sh && \
    nohup /root/run_step10.sh > /mnt/manu/armfeat_s10.log 2>&1 &'   # -> 31,661 帧

# 2 传输（6 流并行，31,661 帧 / 325 s ≈ 81 MB/s）
split -n l/6 -d /tmp/s10list.txt /tmp/s10part_
for p in /tmp/s10part_*; do rsync -az -e "ssh -p 32222" --files-from="$p" --relative \
  /run/media/manu/SSD960-LOCAL/nfs/armfeat_s10/ \
  huangzhe@192.168.99.40:/mnt/data/siping/datasets/manu/armfeat_stage_s10/ & done; wait

# 3 组装（OFFSET=1，丢 k<2 的 48 帧）
$PY manu/data/assemble_arm_gmc_dataset.py \
  --stage /mnt/data/siping/datasets/manu/armfeat_stage_s10 \
  --out   /mnt/data/siping/datasets/manu/arm_gmc_full_s10 \
  --labels-from /mnt/data/siping/datasets/manu/ref_bmpval \
  --expect-total 31613 --verify-md5 60

# 4 推理（PC 臂沿用 pc_bmp_full_cache.pkl，不重跑）
$PY manu/inference/cache_trial0474_inferences.py \
  --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --data /mnt/data/siping/datasets/manu/arm_gmc_full_s10/data.yaml \
  --output runs/gmc_eval/arm_gmc_s10_cache.pkl \
  --imgsz 640 --stride 2 --batch 32 --device 0 --conf-thresh 0.08 --top-k 80

# 5 A/B 评估（对照臂换成 step10 臂）
$PY manu/evaluation/eval_arm_pc_gmc_ab.py \
  --arm-a runs/gmc_eval/pc_bmp_full_cache.pkl  --label-a PC-BMP \
  --arm-b runs/gmc_eval/arm_gmc_s10_cache.pkl  --label-b ARM-S10 \
  --tol 8.0 --out-json /mnt/data/siping/datasets/manu/gmc_arm_s10_vs_pc_ab.json
```

产物 md5：`pc_bmp_full_cache.pkl`=`59c0a19b...`（未动）、`arm_gmc_full_cache.pkl`=`7b02d122...`（step=2，未动）、
`arm_gmc_s10_cache.pkl`=`721f53fd...`（step=10）。
**路径隔离**：step=10 产物全带 `_s10` 后缀；**step=2 产物一条未删**（用户明确要求保留）。
板卡二进制两轮共用同一份 `aa99da81...`，仅 `--anchor-step` 不同。

### 15.6 ⚠️ 四条限定

1. **只覆盖 step=10 与 2 两档**；`{4,6,8}` 仍未测，**不得声称 10 最优**。
2. **未做 ARM(step10) vs PC-side tree 的交叉校验** ⇒ 若要断言「移植在 step=10 下仍保真」，
   需另建 PC-side tree 参照臂；本节 0.9025 与 0.906050 **不可直接相减**。
3. **§15.4 时延不得直接进嵌入式预算**（旧记录已两度指出该线 5 倍不自洽，
   且 `-O0` Debug 与优化构建不可比）。
4. **三臂共同少 ~0.004 F1**（相对冻结 headline），属共有因素、A/B 中抵消，
   但**绝对 F1 不得与 headline 直接比大小**。

---

## 16. ✅ NPU int8 去 P0 vs PC SOTA：三臂受控消融（2026-10-15）

设计：PC SOTA 配置与脚本一字不改、特征仍用 PC 版本，**只换推理后端**（铁律五 ⇒ 强制三臂）。

| 臂 | 峰值 th | F1 | P% | R% | TP | FP |
|---|--:|--:|--:|--:|--:|--:|
| 臂1 PC-FP32 **+P0**（冻结基准） | 0.25 | **0.902316** | 94.87 | 86.02 | 21,601 | 1,167 |
| 臂2 PC-FP32 **−P0** | 0.25 | **0.901935** | 94.86 | 85.96 | 21,586 | 1,169 |
| 臂3 **NPU-int8 −P0** | **0.20** | **0.882231** | 94.27 | 82.90 | 20,818 | 1,265 |

| 分解项 | 峰值 ΔF1 | ΔTP | 15 档均值 | 负档 |
|---|--:|--:|--:|--:|
| 去 P0（臂1→臂2） | −0.000382 | −15 | −0.004296 | 11/15 |
| **int8 量化（臂2→臂3）** | **−0.019704** | −768 | −0.016615 | **15/15** |
| **合计（臂1→臂3）** | **−0.020085** | −783 | −0.020911 | 14/15 |

同点 th=0.25：ΔF1 = **−0.020986**。

### 16.1 结论

1. **损失 98% 来自 int8 量化，不是去 P0**：去 P0 仅 15 TP（噪声下限 ±3 TP），int8 是其 **51 倍**。
   ⇒「去 P0 不掉点」在 FP32 域成立，**不可外推到 int8**。
2. **int8 全档位一致劣化**（15/15 负，区间上界仅 −0.001469），系统性下移而非抖动。
3. **NPU 最优阈值 0.25 → 0.20**：e2e 固化的 `TH=0.25` **不是 NPU 最优工作点**。
4. 召回是唯一受创项（R −3.06pt，TP −768；P 仅 −0.59pt）。NPU 每帧均 0.83 检出，
   **24.8% 帧零检出**（PC 最多 11 个/帧）。

### 16.2 ⚠️ RK3588 NPU 并行度硬上限 = 2（附带发现）

第 3 个并发上下文一提交即撞 **RKNPU 6s job timeout** → 驱动 `soft reset` 整个 NPU → 在飞全陪葬（rc=1/139）。

| P | 1 | **2** | 3 | 4 | 6 | 9 |
|---|--:|--:|--:|--:|--:|--:|
| 成功 worker | 1/1 | **2/2** | 2/3 | 2/4 | 2/6 | 2/9 |
| 真实吞吐 f/s | 9.92 | **19.64** | 20.03 | 6.34 | 6.31 | 6.29 |
| 内核崩溃 | 0 | 0 | **1** | 0 | 0 | 0 |

**6/9 进程不可行**，P>2 只空耗 ~19s 超时窗口；P≥4 的吞吐数字是假的。
`/sys/class/devfreq/fdab0000.npu/load` 是三核合计口径，**不能**判断空核，必须实测 + 看 dmesg。
本轮最终走**单进程**（稳优先），零失败零重试。

### 16.3 踩坑

1. **benchmark 必须前置校验输入存在性** —— 否则越界 BASE 会打印「480 fps」假数据（已加守卫 + 越界验证）。
2. 并行基准被全量任务污染，首轮数据已作废重测。
3. 自校准闸门此前**完全失效**（编排脚本没传 `--expect-*`），本轮接上，两道全 PASS。
4. `__<k+1>` 偏移只属旧 ARM-GMC 装配路径，本链路无偏移风险；`BOARD_TOTAL=31661` 是错的（实际 31,613）。

### 16.4 脚本

`manu/pipeline/npu/run_npu_pcfeat_board.sh`、`manu/pipeline/npu/bench_npu_parallel.sh`、
`manu/diagnostics/npu_tsv_to_cache.py`、`manu/evaluation/run_three_arm_eval.sh`、
`manu/evaluation/three_arm_pipeline.sh`、`manu/diagnostics/check_npu_tsv_vs_cache.py`

**完整记录（命令 / 环境 / 产物 md5）**：`manu/docs/experiments/2026-10-15_npu_int8_vs_pc_three_arm.md`

### 16.5 ⚠️ 三条限定

1. **不得与 SOTA headline 0.906361 直接比大小**（共有因素少 ~0.004，A/B 内部相减才有效）。
2. 臂3 峰值在 th=0.20、臂1/2 在 0.25；同点口径 −0.020986，**两口径不可混用**。
3. **未做 PTQ 校准敏感性分析** ⇒「int8 代价 −0.0197」是该**具体模型**结论，
   不可外推为 int8 的普遍代价。

---

## 17. ✅ `e2e_demo.cpp` 单进程纯 C++ demo，与 shell demo 逐字节一致（2026-10-15）

弃用 `run_hm_demo.sh` 的 shell 编排，单进程纯 C++ 跑完 raw → 三通道特征 → RKNN int8 → heatmap。
验证序列 `34_1`（537 帧）：

| 产物 | 一致数 |
|---|--:|
| 三通道特征 BMP | **537 / 537** |
| heatmap + offset 张量（float32 `.bin`） | **1070 / 1070** |
| 检出 TSV | **535 / 535** |

两边 GMC 特征序列 md5 相同 `5caa3c6fc72eb7fad4dd5dfd08ebf4ad`，回退计数均为 0。
检出总数 545 vs 547，差 2 恰为 e2e 多出的 push 0/1 两帧，非差异。

### 17.1 方法论（可迁移）

`e2e_core.cpp` **#include 两个生产源、main 编译掉**，调同一批类与函数
（`OnlineFeaturePipeline`/`FusedGpu`、`letterbox`/`run_once`/`dequant`/`extract_peaks`），
对既有文件**只加一行 `#ifndef` 包裹 main，函数体一字未改**。

**等价性最强证据**：重编的两个原二进制与板上原版**逐字节一致**
（`rknn_heatmap_smoke` `170379506ad...`、`gmc_stream_ocl_arm64` `aa99da8113...`）
⇒ 「同一份代码」是可证事实。

**推广**：任何「A/B 必须等价」的改造，都应走「只加编译开关、不动被测函数体、
再用产物 md5 自证」这条路。

### 17.2 帧号映射（PUSH_OFFSET=1）

参考按 id 3..537 读（= push 2..536），e2e 按 push 0..536 读
⇒ **参考 id N ≡ e2e push N−1**。shell 版需要 host 侧 python 建符号链接改名，
是因为两个进程必须就文件名达成一致；单进程后该步骤结构性消失。
不做这层映射会得到假的「一致」或假的「差异」。

### 17.3 并行预留 + 硬约束

`extract_features()` 返回特征路径列表，`run_infer()` 接受任意 frames 子集
⇒ 并行 demo 可按序列或按帧切分。
**但 RK3588 NPU 并行上限 = 2**（§16.2），第 3 个并发上下文撞 6s job timeout 触发驱动软复位。

### 17.4 ⚠️ Mali 驱动锁泄漏 → 必须断电重启（运维教训）

§16.2 那次 NPU 并行扫描触发 RKNPU 软复位 + `BUG: spinlock bad magic on CPU#2, kswapd0/146`
（debug kernel oops）后，**任何打开 `/dev/mali0` 的进程都永久阻塞**：
卡在 `rwsem_down_write_slowpath`，D 状态 `kill -9` 无效，load average 88。

- **`reboot` 在该状态下不可靠**（后台化后随 ssh 断开一起死），最终**手动断电**才恢复。
- **重启后 NFS 挂载丢失**（`/etc/fstab` 无此条），须手工重挂：
  ```sh
  mount -t nfs4 -o rw,vers=4.0,soft,proto=tcp,timeo=5,retrans=2 \
    192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu
  ```
- 长跑任务前先冒烟：`timeout 60 /root/gmc_stream_ocl --help`

### 17.5 ⚠️ 比对脚本自身的坑

`cmp_e2e_outputs.py` 自己出过两个 bug，**都制造了假结论**：
① 把参考 `_hm.bin` 和 e2e `_off.bin` 比（offset 是 heatmap 两倍大），报「全不一致」实际全一致；
② 帧号解析漏剥 `.bmp` 抛异常。
⇒ **比对脚本自身必须先自证可靠**（用已知该一致/该不一致的样本验证），
第一个跑出来的数字最容易被直接采信，也最容易是脚本自己的错。

### 17.6 脚本

`manu/pipeline/npu/e2e_demo.cpp`、`e2e_core.h`、`e2e_core.cpp`、
`build_e2e_demo_arm64.sh`、`cmp_e2e_outputs.py`；板卡 `/root/e2e_demo`。

**完整记录（命令 / 环境 / md5）**：
`manu/docs/experiments/2026-10-15_e2e_demo_cpp_vs_shell_demo.md`

### 17.7 ⚠️ 两条限定

1. **只测了 `34_1` 一个序列 537 帧**，且全程 `anchor_step=10 / --fused gpu / th=0.25`；
   其余 23 序列、其它 anchor_step、`--fused cpu` 均未覆盖。
2. **e2e_demo 多了 `release(c, &out)`，原 rknn 批处理模式没有**。释放上一帧 1.2 MB 缓冲
   不改变 NPU 输出（数值由 `rknn_run` 产生），且 1070/1070 已实测一致；
   但这是**两者唯一的行为差异**，说「代码完全相同」时需一并说明。

---

## 18. ✅ 并行流水线 e2e demo：GMC ∥ NPU，逐位一致，1.76×（2026-10-15）

序列 `34_1`（537 帧），`anchor_step=10 / --fused gpu / th=0.25`，预热 100 帧剔除：

| 配置 | FPS | 墙钟 | GMC p50 | GMC p95 | GMC max | ≤40ms |
|---|--:|--:|--:|--:|--:|---|
| 原版（串行 + 落盘） | 5.78 | 93.0 s | 27.33 | 28.11 | 28.65 | PASS |
| 去中间落盘（串行） | 8.16 | 65.8 s | 27.71 | 28.56 | 28.99 | PASS |
| **去落盘 + 并行流水线** | **10.15** | **52.9 s** | 27.40 | 28.10 | 30.71 | PASS |

**累计 1.76×，40 ms 全程守住。** 去落盘 +41%，并行重叠再 +24%。

### 18.1 等价性（三重证明）

- 逐字节 vs shell demo 基准：三通道特征 **537/537**、heatmap+offset **1070/1070**
- **哈希链跨 I/O 边界：文件路 vs 无 I/O 路逐帧链式 fnv1a，三档全 `d5e81977`**
  ⇒ 绕开 BMP 落盘**没有偷换通道**，且不落一字节即可证明与文件路逐位相同
- GMC 序列 md5 四档均 `5caa3c6fc72eb7fad4dd5dfd08ebf4ad`；
  `fitgate=0 chain=0 fitdeadline=0 cornercap=0`（bad case handle 全保留未触发）

### 18.2 NPU 归因（ms/帧）

| 阶段 | 文件路 | 无 I/O |
|---|--:|--:|
| load（读文件+BMP 解码） | 8.46 | **0** |
| letterbox（512→640 双线性） | 27.51 | 25~29 |
| **rknn** | 55.32 | **59.96** |
| dequant / peaks | 2.73 / 0.35 | 同 |
| **write（heatmap/det → NFS）** | **27.39** | **0** |

⇒ 带落盘时**每帧 35.85 ms 纯 I/O**（NPU 段的 31%）——这就是"5.78 fps 不可信"的来源
（每帧往 NFS 写 2×1.2 MB heatmap）。
真瓶颈：**rknn ~60 ms**，其次 **letterbox ~29 ms**（`rknn_heatmap_smoke.cpp:193`）。

### 18.3 ⚠️⚠️ RK3588 CPU 不同质

cpu0-3 = A55（capacity 405 / 1.8 GHz），**cpu4-7 = A76（1024 / 2.352 GHz）**。
**算力最重的 GMC 必须绑大核。** 首次默认绑 `cpu-feat 0-3` ⇒ GMC p50 **27 ms → 93 ms**（3.4× 劣化），
代码看起来完全正确。最终默认 `--cpu-feat 4-7` / `--cpu-npu 0-3` / `--npu-nice 5`，
绑定到位后并行版 GMC **p95 从 62.05 ms 回落到 28.10 ms**。

⇒ **并行化前必须先读 `/sys/devices/system/cpu/cpu*/cpu_capacity` 确认拓扑**。

### 18.4 设计要点

- 批量路径收成流式薄壳，帧体抽成共享 `infer_frame()`（文件路与内存路共用）
  ⇒ 串行/并行跑同一份逐帧代码，一致是构造性的。
- 两个 demo 编进**同一二进制**（`--mode serial|parallel`）、同一套计时
  ⇒ FPS 对比不是两个测量口径的产物。
- **通道序**：`cv::imwrite` 存 BGR、`stbi_load` 读回 RGB ⇒ 模型实际看到 `[Ch2,Ch1,Ch0]`，
  内存路径必须原样复现这个翻转。
- 缓存是**有界队列**（`--depth` 默认 6），无界会把整个序列的 BMP 攥在内存。

### 18.5 ⚠️ 四个"看起来对"的 bug

1. **帧名丢目录前缀**（改构造函数时漏拼 `seq_dir`）→ `failed to read 000001.bmp`
2. **`setpriority(PRIO_PROCESS, 0, n)` 降的是整个进程**，`who=0` 指进程非线程 ⇒ GMC p50 27→91 ms。
   改为按 tid：`setpriority(PRIO_PROCESS, syscall(SYS_gettid), n)`。
   ⇒ **per-thread 调优 API 的 `who=0` 语义必须确认**。
3. **双重析构**（placement-new 时代的显式析构残留）→ `double free detected in tcache 2`，
   且只在 stdout 重定向时显形、块缓冲还吞掉崩溃前日志
   ⇒ 已加 `setvbuf(stdout, nullptr, _IOLBF, 0)`。
4. **`--no-io` 写在组装之后** ⇒ "不落盘"其实还在写 heatmap，write 仍 25 ms，
   FPS 只从 5.74 走到 6.59 几乎看不出差别。
   ⇒ **配置覆盖必须早于它要影响的那次组装**。

### 18.6 脚本

`manu/pipeline/npu/e2e_par_demo.cpp`、`e2e_core.h`、`e2e_core.cpp`、`e2e_demo.cpp`、
`build_e2e_demo_arm64.sh`；板卡 `/root/e2e_par_demo`、`/root/e2e_demo`。

**完整记录（命令 / 环境 / md5 / NPU 归因）**：
`manu/docs/experiments/2026-10-15_e2e_par_pipeline.md`

### 18.7 ⚠️ 三条限定

1. **只测 `34_1` 一序列 537 帧**，`anchor_step=10 / --fused gpu / th=0.25`、双线程；
   其余 23 序列、`--fused cpu`、更深缓存未覆盖。
2. **10.15 fps 依赖 NFS 状态**：写本地存储则 `write` 27.39 ms 大降、FPS 更高；
   若 heatmap 必须落盘则无 I/O 路不可用。
3. **`letterbox` ~29 ms 是下一个优化点**（`rknn_heatmap_smoke.cpp:193` float 双线性）。
   本轮未动——动它会改像素、进而可能改 heatmap，必须重走逐位验证。

---

## 19. ✅ 三级流水线：letterbox 独立成一级，14.07 fps（累计 2.45×）（2026-10-15）

承接 §18。§18.2 的归因显示 NPU 线程 94.45 ms 里只有 59.96 ms 是硬件，
**28.60 ms 是挤在同一线程上的 CPU letterbox**，硬件全程空转。拆成独立一级（`--split-lb`）：

| 配置 | FPS | 帧时间 | GMC p50 | GMC max | ≤40ms | hash |
|---|--:|--:|--:|--:|---|---|
| 原版（串行 + 落盘） | 5.74 | 174.3 ms | 27.52 | 28.83 | PASS | `d5e81977` |
| 去落盘（串行） | 8.16 | 122.5 ms | 27.71 | 28.99 | PASS | `d5e81977` |
| 去落盘 + 并行 2 级 | 10.15 | 98.5 ms | 27.40 | 30.71 | PASS | `d5e81977` |
| **去落盘 + 并行 3 级** | **14.07** | **71.1 ms** | 27.93 | 32.10 | **PASS** | **`d5e81977`** |

**累计 2.45×（5.74 → 14.07），四档 hash 全 `d5e81977` 逐位等价。**
> ⚠️ **2026-10-16 待复核**：`e2e_par_demo` 读 raw 帧用的是 `cv::imread`，而配套 arm64 OpenCV
> **未编图像编解码** ⇒ 现存 PNG 版 raw 目录读不出帧（RC=0 但 `frames=0`），
> BMP 版可读。本节 14.07 fps 当初用的 raw 来源**尚未查清**，复核前不要当作可直接复现的基线。
> 详见 §22「仍欠」第 4 条。

三级（ms/帧）：GMC 33.19（cpu 4-5）、**letterbox 12.23（cpu 6-7）**、
NPU 66.99（cpu 0-3：rknn 60.77 + dequant 2.75 + peaks 0.35）。

### 19.1 两条可迁移结论

1. **letterbox 独占线程只要 12.23 ms，挂在 NPU 线程上要 28.60 ms——差 2.3 倍是争抢不是算法。**
   ⇒ **"这段代码慢"必须先分清算法慢 vs 被抢慢**：挪到独占线程再测一次，
   代价近乎为零，变慢就说明瓶颈在争抢。
2. **等价性白送**：三级调的是**同一个 `letterbox()`、同样参数**，只换线程，
   hash 一致即可证。**纯搬运/纯计算的无状态阶段挪到独立线程不需要重验精度**。

### 19.2 当前瓶颈与天花板

- 帧时间 71.1 ms，NPU 线程 66.99 ms（94%）
- 理论上限 66.99 ms → **14.93 fps**（当前重叠效率 94%）
- **硬地板 63.87 ms → 15.7 fps**
- 再往下只能拆 `rknn`（`inputs_set/run/outputs_get` 现合并统计），
  需 `RKNN_FLAG_COLLECT_PERF_MASK`。**本轮未测，不得声称已定位。**

### 19.3 脚本

`e2e_par_demo.cpp`（新增 `--split-lb` / `--cpu-lb`）、`e2e_core.h/.cpp`
（新增 `letterbox_feature()` / `infer_preboxed()`）；板卡 `/root/e2e_par_demo`。

**完整记录**：`manu/docs/experiments/2026-10-15_e2e_par_pipeline.md`（§7）

---

## 20. ✅ 等价性复核：逐位比对重跑，修正三个「比对方法」坑（2026-10-15）

交付最终报告时按裸 `diff -rq` 重比 §18/§19 产物，报 1070/1070 不一致。重做后：
**数据没变，是比对方法错了。** 复核后 `ref` vs 串行/并行两档，
heatmap+offset **1070/1070 逐字节一致**、检出 TSV 535 可比帧 **0 行差异**、
特征 537 张聚合 md5 三档全同、`detections=547`、bad case handle 全 `0`。

### 三个坑

1. **帧名偏移**：`ref`（shell demo）从第 3 帧起、`e2e` 从第 0 帧起，
   裸 `diff -rq` 把名字对不上当内容不一致。正确映射 **`ref = serial + 1`，且是唯一解**
   （扫 0~4，仅 +1 为 0，邻位全 1070 不一致）。
   **⇒ 铁律级：比对前先扫偏移确定唯一解，再看该解是否为零差异。
   只看"某个偏移是 0"不够——全零文件任何偏移都是 0。**
2. **证据链断裂**：三级档 `--no-io` 只有链值 `d5e81977`，落盘档（已证 == `ref`）没开
   `--hash` 无链值，两组各证一半。已用**独立实现**按 `e2e_core.cpp:485-487` 规则重算落盘档
   `.bin`（659,865,600 字节）得 `d5e81977`，桥搭上，三级档 == `ref` 传递可证。
   FNV 是字节级串行链，Python 跑不动，用 `/tmp/fnvchain.c`（`gcc -O2`）。
3. **误导产物目录**：`noio_serial`/`noio_parallel` 是修 `--no-io` 位置 bug **之前**的旧运行
   （头部写 `io=NONE` 但 `write` 仍 25 ms，FPS 低估成 6.59/8.16），用 `v2_*`；
   `e2e/` 是更早一轮 `e2e_demo` 产物，所有偏移都对不上，不参与本轮结论。
   **⇒ 判据：头部参数声明与 breakdown 数字必须自洽，不自洽的目录直接作废。**

### 速度表最终口径

原版档复核后按 `io_serial/run.log` **实测 5.74 fps / 93.61 s / 174.3 ms**
（此前记 5.78 是另一次运行；单次运行未取方差）。§19/§23 四级表首行已统一改 5.74，
累计倍率 2.43× → **2.45×（5.74 → 14.07）**。§18/§22 的 5.78 保留原样。

### rknn 三段拆分（已改，**未上板跑**）

`e2e_core.cpp:413` 原把 `inputs_set/run/outputs_get` 加总成 `rknn_ms`，已拆成
`inputs_set_ms`/`run_ms`/`outputs_get_ms`（`rknn_ms` 保留作和以兼容旧口径）。
`--rknn-perf`（`COLLECT_PERF_MASK`）**判定不可用**：run_duration 差分得 744 ms/帧
远超 rknn wall 69 ms，且该 flag 拖慢运行（14.07 → 13.16 fps）。
**复核更正（重要）**：此前记的产出二进制 md5 `5599bcde…` **是错的**。板上 `/root/e2e_par_demo`
最后一次推送 **17:11:59**，实际 md5 **`d314f84ecb1c2711d69d5cb302cdd0e6`**；
`io_serial`(17:50:58) ~ `split_lb`(18:07:29) 全在其之后，**§19 全部数字由 `d314f84e…` 产出**。
新编 `cfc750ad…` 从未上过板。`rknn 60.77 ms` 内部构成仍未定位。

实验记录：`manu/docs/experiments/2026-10-15_e2e_par_pipeline.md` §8（含可复制比对命令）

---

## 21. ✅【收口】e2e 三级流水线：2.45× 且与原版逐位一致（2026-10-15）

详细报告：`manu/docs/experiments/2026-10-15_e2e_par_final_report.md`
实验记录：`manu/docs/experiments/2026-10-15_e2e_par_pipeline.md`

### 速度（34_1 / 537 帧 / anchor_step=10 / --fused gpu / th=0.25 / 预热 100 帧剔除）

| # | 配置 | FPS | 墙钟 | 帧时间 | GMC p50 | GMC max | ≤40ms |
|:--|---|--:|--:|--:|--:|--:|---|
| 1 | 原版（串行+落盘） | 5.74 | 93.61 s | 174.3 ms | 27.52 | 28.83 | PASS |
| 2 | 去中间落盘（串行） | 8.16 | 65.84 s | 122.5 ms | 27.71 | 28.99 | PASS |
| 3 | 去落盘+并行 2 级 | 10.15 | 52.91 s | 98.5 ms | 27.40 | 30.71 | PASS |
| 4 | **去落盘+并行 3 级** | **14.07** | **38.18 s** | **71.1 ms** | 27.93 | 32.10 | PASS |

**2.45×**。+42%（去落盘，每帧省 ~38 ms 纯 I/O）→ +24%（两级重叠 1.279×）→ +39%（letterbox 拆独立级）。
三级绑核：GMC cpu4-5/33.19ms、letterbox cpu6-7/12.23ms、NPU cpu0-3/66.99ms。
天花板 max(...)=66.99→14.93 fps；硬地板 rknn 60.77+dequant 2.75+peaks 0.35=63.87→15.7 fps。

### NPU 段（ms/帧）

| 阶段 | 原版 | 去落盘 | 并行2级 | 并行3级 |
|---|--:|--:|--:|--:|
| load | 8.38 | 0 | 0 | 0 |
| letterbox | 28.38 | 25.10 | 28.60 | **0** |
| **rknn** | 55.79 | 56.88 | 59.96 | **60.77** |
| dequant / peaks | 2.72 / 0.34 | 2.72 / 0.35 | 2.75 / 0.35 | 2.75 / 0.35 |
| write | **29.55** | 0 | 0 | 0 |

⇒ **落盘路径每帧纯 I/O ≈37.9 ms，占 NPU 段 32%**，这就是"原版 5.74 fps 不可信"的来源。
⇒ **letterbox 独占线程 12.23 ms vs 挂 NPU 线程 28.60 ms，快 2.3 倍**——差额是争抢不是算法。
**"某段代码慢"先分清算法慢还是被抢慢，挪到独占线程再测。**

### 精度：五条独立证据全一致

特征 BMP 聚合 md5 三档全 `f47f9135…`（各 537 张）／GMC 序列 md5 `5caa3c6f…`（6 档）／
heatmap+offset **1070/1070 逐字节**（偏移 `ref=serial+1` 扫描为唯一解）／检出 TSV **535 可比帧 0 行差异**，均 547／
链式哈希 `d5e81977`（已用独立实现重算闭合证据链）。
坏例 `fitgate=0 chain=0 fitdeadline=0 cornercap=0` 四档全 0；`--fit-deadline-ms` 保持 0 保证确定性。
通道序 `[Ch2,Ch1,Ch0]` 已在内存路复现，链值一致即证未偷换。

### 产物与环境

产出全部数字的板卡二进制 **`/root/e2e_par_demo` `d314f84ecb1c2711d69d5cb302cdd0e6`**（推送 17:11:59）。
模型 `trial0474_hm_nop0_int8.rknn` `ef302edb1b5aff1d376204bb7ad77550`（CPU 算子 0）。
CPU：**cpu0-3 A55 小核@1.80GHz ／ cpu4-7 A76 大核@2.352GHz**，GMC 必须绑大核。
GPU Mali-G610 4CU，OpenCL 3.0 v1.g13p0。

### 仍欠的一项

`rknn 60.77 ms` 内部构成未拆开。代码已改（三段计时），新版 `cfc750ad…` **从未上过板**，需授权。
`--rknn-perf` 已判定不可用。大头若在 `outputs_get`（~1.2 MB/帧回读）有优化余地，若在 `rknn_run` 则是地板。

---

## 22. ✅✅ NPU fp16 vs int8 受控对照（2026-10-16）—— **fp16 挽回 int8 量化损失的 97.3%，代价 ×1.85 时延**

**headline（全量 31,613 帧 / GT 25,111 / tol 8.0px / 15 档，两臂闸门均 PASS）**

| 臂 | 峰值 th | F1 | P% | R% | TP | FP |
|---|--:|--:|--:|--:|--:|--:|
| 臂1  PC-FP32 +P0（冻结基准） | 0.25 | 0.902316 | 94.87 | 86.02 | 21601 | 1167 |
| 臂2  PC-FP32 −P0 | 0.25 | 0.901935 | 94.86 | 85.96 | 21586 | 1169 |
| 臂3  NPU-int8 −P0 | **0.20** | 0.882231 | 94.27 | 82.90 | 20818 | 1265 |
| 臂4  NPU-**fp16** −P0 | **0.25** | **0.901401** | **94.93** | **85.81** | 21548 | 1151 |

- int8 量化代价（臂2→3）= **−0.019704**（−768 TP）｜**fp16 量化代价（臂2→4）= −0.000533**（−38 TP）
- **fp16 挽回（臂3→4）= +0.019171（+730 TP），15 档负档仅 1/15 ⇒ 挽回率 97.3%**
- 对照 FP32 的 15 档：fp16 峰值 −0.000533、均值 −0.002132、**负档 12/15**（高阈值区系统性略低）
- ⚠️ −38 TP 超出噪声下限（±3 TP ≈ ±0.0002 F1）约 12 倍 ⇒ **fp16 的 −0.000533 是真实小损失，不是抖动**

### 🔴 业务判据分水岭

业务线判据 **F1 ≥ 88% 且 R ≥ 85%**：int8 臂 R 82.90% **不达标**；
**fp16 臂 R 85.81% 达标**（F1 0.9014 两者都远超 88%）。
⇒ 若召回硬线必须过，**int8 在这块板子上就不合格，fp16 合格**。
且 fp16 峰值 th = 0.25（与 PC-FP32 一致），int8 峰值 th = 0.20 ⇒ e2e 固化的 `TH=0.25`
**对 fp16 是正确工作点，对 int8 偏保守**。

### 🔴 旧禁令的适用边界被推翻（重要）

2026-10-13 记的「同图 fp16 在板卡提交失败」，失败算子 `Conv:/model/p0_highway/p0_dw/Conv`
**位于 P0 支路内部**。交付模型是**去 P0** 图 ⇒ 该算子不存在 ⇒ **fp16 实测可提交、可全量跑完**。

**旧结论应改写为「带 P0 的图上 fp16 提交失败」，而不是「这块板子不支持 fp16」。**
已同步修正 `CONVERT_HM.md` 两处（模型表 + 坑表 #3）。
`rules.md` §四·2f 的「对照臂必须自己上过板」**依然正确且本轮再次证明其价值** ——
板子没变，是对照臂换了图。

### 模型构建（可复现）

```sh
cd /media/manu/1TB-Volume/rknn_work/hm
/media/manu/1TB-Volume/conda-envs/rknn230/bin/python \
  /media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu/hm_convert_rknn.py \
  --onnx models/trial0474_hm_nop0.onnx --out models/trial0474_hm_nop0_fp16.rknn \
  --compress-weight --no-quant
```

- 产物 `trial0474_hm_nop0_fp16.rknn` 6,327,393 B，md5 **`07c893e4d2ca3af9a47a4bf88335fe33`**
- 源 ONNX `trial0474_hm_nop0.onnx` md5 **`028f23c655f421fe61a5e43575715fca`**（7,210,756 B，与 int8 臂同一份）
- 构建日志 `logs/build_nop0_fp16.log`
- 🔴 **toolkit 环境陷阱**：host 上有两个 rknn env ——
  `conda-envs/rknn` 是 **1.4.0**（弃用），`conda-envs/rknn230` 才是 **2.3.0**（与板卡 runtime
  `c949ad889d` 同版本）。**转换前必须核对 `importlib.metadata.version('rknn-toolkit2')`**。

### 构建日志验收

| 项 | int8 | fp16 |
|---|--:|--:|
| CPU 计算算子 | 0 | **0** ✅ |
| 构建回退警告 | 0 | **1** ⚠️ `Conv:/model/b5/conv/Conv` 3Core→单核 |
| NPU 内部内存 | 20,400 KB | **105,600 KB（5.18×）** |

🔴 **105.6 MB 内部内存没有卡死并发** —— 实测 P=2 双 worker 跑完 31,613 帧，
**0 次 soft reset / job timeout**，稳定 12.6 fps。
⇒ 「Total Internal Memory Size」**不能**当每上下文预留量线性外推并发上限
（与「禁止用 MAC/FLOPs 外推时延」同源教训）。

### tensor 级精度（用户指定的关键校验，板卡实见输入闭环喂原始 ONNX）

新增脚本 `manu/pipeline/npu/compare_npu_onnx_tensor.py`；12 帧
（板卡 `--dump-input` 落盘实见 uint8 NHWC，host 用同一份字节喂 onnxruntime）：

| vs ONNX-FP32 | **fp16** | int8 | fp16 优势 |
|---|--:|--:|--:|
| heatmap worst max_abs | **7.099e-03** | 1.914e-01 | **27×** |
| heatmap 峰值 worst \|Δ\| | **4.665e-03** | 1.914e-01 | **41×** |
| heatmap argmax 一致 | **10/12** | 8/12 | — |
| offset avg mean_abs | **2.840e-03** | 7.462e-02 | **26×** |

闸门 `tol_hm=2e-2 / tol_off=2e-1` → fp16 **PASS**，int8 **FAIL**。
**机制**：int8 误差主要打在 **offset（位置）** 上 ⇒ 峰值被搬离 GT ⇒ 掉召回。
⚠️ 限定：fp16 offset worst 1.978e-1 已达门限 99%，只是勉强过；`max_rel` 被近零分母放大
（1.8e+02 / 1.4e+05）**不可作判据**，判据一律用 `max_abs`。

### 时延与并行度

| 阶段 | int8 | fp16 |
|---|--:|--:|
| `inputs_set` | 0.184 ms | **21.0 ms（P=2 下 30.9 ms）** |
| `rknn_run` | 55.3 ms | **104.4 ms** |
| 单进程整帧 | 84.4 ms / 11.8 fps | **159.2 ms / 6.3 fps** |
| **P=2 实测** | 19.64 fps | **12.6 fps（2.02×）** |

**fp16 多出的 21–31 ms 全在 `inputs_set`（uint8→fp16 转换），不是 NPU 计算本身** ⇒
一处独立于模型精度、尚未处理的可优化点。
温度 47–49 °C、NPU 锁 1 GHz performance、CPU 满频 ⇒ **慢与降频无关**。

### ⚠️ 为什么精度口径没有用 3 级并行程序（用户要求提速，此处不成立）

`e2e_par_demo` 强制 `--raw-root`，其 stage A 就是 GMC
（`e2e_core.cpp:36-40` 把 `gmc_stream_ocl.cpp` 与 `rknn_heatmap_smoke.cpp` 编进同一 TU，
且 `FeatureStream` 里 `OnlineFeaturePipeline pipe(...)` **无条件构造** ——
**没有「喂预建特征」的分支**）。昨天的受控口径刻意跳过 GMC 直喂 PC 特征；
把 PC 特征当 `--raw-root` 会让 GMC 跑两遍 ⇒ 口径失效。
**实际提速用 P=2 双 worker（int8 臂同一套策略，两臂同条件）**。

### 与板卡既有 fp_no_p0.rknn 的等价性（补测）

板上 `fp_no_p0.rknn`（`hm_struct_variant.py` 结构变体产物，fnv1a32 `e2443a11`）
与本轮重建版（fnv1a32 `aaf9cf62`）：**输入 12/12 逐字节一致，输出 24/24 文件逐字节一致**
⇒ **功能等价**，fnv1a32 差异只来自 RKNN 容器元数据。
但仍**一律使用重建版**：结论只建立在可复现路径上，不依赖来源不明的历史产物。

### 新增/变更产物

- `manu/pipeline/npu/compare_npu_onnx_tensor.py`（新）
- `manu/pipeline/npu/run_npu_pcfeat_board_f16.sh`（新）
- `manu/evaluation/run_int8_vs_f16_eval.sh`（新）
- `manu/evaluation/int8_vs_f16_pipeline.sh`（新）
- `manu/docs/experiments/2026-10-16_npu_fp16_vs_int8.md`（新，完整记录）
- 板卡 fp16 产物目录 `/mnt/manu/npu_pcfeat_f16`（24 序列 TSV，31,613 帧）
- 服务器 cache `runs/gmc_eval/arm_npu_f16_nop0_cache.pkl`、
  对比结果 `runs/gmc_eval/int8_vs_f16/cmp{F,I}_*.json`

### 仍欠

- e2e 三级口径下 fp16 的 fps 未测（int8 是 14.07 fps），若决定上 fp16 必须补
- `Conv:/model/b5/conv/Conv` 回退是否为时延劣化主因未归因（MAC 数相同，回退只是候选解释）
- `inputs_set` 21–31 ms 的预转换优化未做
- ⚠️ **`e2e_par_demo` 的 raw 输入路径当前读不了 PNG，14.07 fps 的输入来源需复核**（2026-10-16 实测）：
  `e2e_core.cpp:173` 用 `cv::imread(path, IMREAD_GRAYSCALE)` 读 raw 帧，而配套的 arm64 OpenCV 4.10.0
  是**全关图像编解码**编译的（模块只有 core/imgproc/features2d/video/calib3d/flann）。
  实测：`--raw-root raw`（`/mnt/manu/armfeat/raw/34_1`，**PNG** 512×512）⇒ `frames=0`、
  `[FATAL] failed to read raw/34_1/000001.png`，**但进程 RC=0**；换成
  `--raw-root bmpraw`（**BMP**）⇒ 正常读帧。
  ⇒ 昨天 §19/§21 那组 14.07 fps 用的是**哪一份 raw**、为什么当时能读，**尚未查清**。
  **在复核之前，不要把 14.07 fps 当作可直接复现的基线引用**；
  另注意「RC=0 但 0 帧」正是铁律 §四·9g 说的假成功形态。

---

## 23. ✅✅ e2e 双 NPU 并行架构：CPU 绑核让 fp16 过 12 fps（2026-10-16）—— **11.89 → 15.26 fps，537/537 逐帧一致**

**headline（fp16 / `34_1` / 537 帧 / 2 workers / anchor_step=10 / --fused gpu / th=0.25）**

| 方案 | GMC | letterbox | npu0 | npu1 | **fps** | per-ctx | `inputs_set` | `rknn_run` |
|---|---|---|---|---|--:|--:|--:|--:|
| A 原方案 | 4-5 | 6-7 | 0-1 | 2-3 | 11.80 | 162.97 | 45.41 | 111.46 |
| P 反例 | 0-1 | 2-3 | 4-5 | 6-7 | **9.12** | 119.72 | 14.73 | 102.27 |
| Q 反例 | 0-1 | 2-3 | 4 | 5 | 9.22 | 120.36 | 14.72 | 103.16 |
| **T 采纳** | **4** | **5** | **6** | **7** | **15.26** | 126.99 | 14.77 | 109.62 |
| T 复跑 | 4 | 5 | 6 | 7 | **15.25** | 126.99 | 14.77 | 109.69 |

**关键点**

- 旧方案 11.89 fps 时理论上限本就是 `2×1000/162.97 = 12.27 fps`——**贴着天花板跑**，
  再加流水线级数也翻不过 12.3。必须压 `per_context_ms` 本身。
- **`inputs_set` 45.41 ms 是小核上的标量循环**：45.41 ms ÷ 1,228,800 = 37 ns/元素
  ≈ 74 cycles @2GHz。同一份 uint8 上传在 int8 图上只要 **0.34 ms**。
  ⇒ 挪到大核后降到 **14.77 ms**（×0.32）。
- **臂 P/Q 是反例**：NPU 级压到 59.9 ms/帧（理论 16.7 fps），但 GMC 被挤到 A55 后
  32.09 → **106.25 ms**，GMC 变新瓶颈，总帧率反而 9.12。**异构核上优化一级会转移瓶颈。**
- **臂 Q 证明 `inputs_set` 吃不满多核**：1 核 14.72 vs 2 核 14.73，一模一样。
- **采纳解 T**：四个线程一人一个 A76 大核，cpu 0-3 的 A55 全程闲置。
  GMC 33.40（对 A 档 32.09 仅退化 4%），理论 15.75 / 实测 **15.26**（97% 流水线效率）。
  int8 同绑核 **29.83 fps**（原 26.88）。

**`pass_through` 路线证伪（驱动自相矛盾）**

- `RKNN_QUERY_INPUT_ATTR` 报 `type=1(FLOAT16) fmt=1(NHWC) n_elems=1228800 size=2457600`（2 B/px），
  但驱动的直通检查要 `4915200`（4 B/px）：`E RKNN: rknn_inputs_set,
  param input size(2457600) < model input size(4915200)`。按声明给 → 报错；补齐到 4 B/px → **SIGSEGV**
  （`__memcpy_generic` ← `librknnrt.so` ← `rknn_inputs_set`）。
- 改声明 `type=FLOAT32` 也不过直通 ⇒ `inputs_set` 46.74 / 47.78 ms 没降；
  先除 255 ⇒ 537 帧 **0 检测**（运行时又除一次 255），不除 ⇒ 571 检测（对，但没提速）。
- ⇒ 不是调用姿势问题，是驱动 bug。代码保留 `--f16-in` / `--f32-in` 作负面对照。
- ⚠️ **旧误判更正**：「arm I 挂死」实为尺寸校验失败后 worker 吞错误继续跑、队列不排空。

**唯一未执行的剩余路线（需用户裁决）**：转模型时把输入固化为 uint8 节点，让 Cast/Mul 成为
图内算子跑到 NPU——这正是 int8 档 `inputs_set` 只有 0.34 ms 的原因。要改 ONNX 输入 dtype +
rknn-toolkit2 2.3.0 重转，属模型侧改动。

**闸门（全 PASS）**：T/Tr/T2/I8T 各 537 行无 0 字节产物；**T 与基线 537/537 逐帧 IDENTICAL**
（571 检测），复跑同样 537/537；改代码后 `e2e_par_demo` int8 chain hash 仍 **`d5e81977`**
与 10-15 基线逐位一致；`dmesg` 本轮 0 错误（16 行全是开机初始化）；2 个上下文未触发
第 3 上下文 soft reset。

**天花板**：`rknn_run` 109.62 ms 占 per-ctx 的 86%，两上下文并行下
`2/109.62 ms = 18.2 fps` 是绝对上限。

**产物**：`e2e_par2_demo.cpp`、`F32Boxed`/`letterbox_feature_f32`/`infer_preboxed_f32`/
`u8_to_f32`(NEON)、`run_once` f32 分支 + 输入 attr 打印；
记录 `docs/experiments/2026-10-16_e2e_par2_npu_affinity.md`；
板卡 `/mnt/manu/npu/out/par2i/{P,Q}`、`par2j/{T,T2}`、`par2k/{Tr,I8T,REG}`。
五个源文件已同步镜像，md5 双侧一致。

**复现**（`--raw-root` 必须 `bmpraw`，arm64 OpenCV 没编 PNG 解码器）：
```sh
/root/e2e_par2_demo --raw-root bmpraw --sequence 34_1 --out-root <OUT> \
  --model /mnt/manu/npu/data/trial0474_hm_nop0_fp16.rknn --kernel-dir /root/kernels \
  --fused gpu --anchor-step 10 --th 0.25 --workers 2 --depth 6 \
  --cpu-feat 4 --cpu-lb 5 --cpu-npu0 6 --cpu-npu1 7
```

**补充复核：目标里的「两个 npu core(0 1) worker」= NPU core，不是 CPU 核**

代码此前**从未调 `rknn_set_core_mask`**，两个上下文一直在默认 `RKNN_NPU_CORE_AUTO` 下，
所以此前 fps 只证明「两个 worker」，没证明「worker N 占住 NPU core N」。
新增 `infer_set_npu_core` + `--npu-core0/--npu-core1`（1/2/4 = core 0/1/2；被拒即 FATAL，不静默吞）。
逐 core 占用由 0.25 s 间隔采样 `/sys/kernel/debug/rknpu/load` 得到：

| 臂 | w0 | w1 | **fps** | per-ctx | `rknn_run` | Core0 | Core1 | Core2 |
|---|---|---|--:|--:|--:|--:|--:|--:|
| v0auto | AUTO | AUTO | **15.28** | 126.57 | 109.38 | 84.1% | 75.2% | **0.0%** |
| v1c01 | `0x1` | `0x2` | **15.18** | 127.94 | 110.24 | 79.4% | 79.9% | **0.0%** |
| v2c02 | `0x1` | `0x4` | **失败**（`failed to submit!`，只出 49/537 行） | — | — | — | — | — |

- **AUTO 本来就用 core 0 + core 1，core 2 全程 0%** ⇒ 目标里的 core(0,1) 成立，但只能靠采样证明。
- **显式切分只让负载更均衡**（79.4/79.9 vs 84.1/75.2），帧率不变 ⇒ **NPU core 分配无余量**；
  `rknn_run` 109 ms 是真实计算不是排队。
- ⚠️ **RK3588 只支持 0/1 这对相邻 core，0+2 直接 `failed to submit!`**（残缺产物 49/537 行，
  按铁律 §四·9g 判失败）。与「第 3 个上下文触发 soft reset」是同一约束的两面。
- **采纳默认 AUTO**（功能/帧率等价，少两个参数少一个失败模式），开关保留在代码里。
- 闸门：v0auto / v1c01 vs 基线各 **537/537 IDENTICAL**；新增代码后
  `e2e_par_demo` int8 chain hash 仍 **`d5e81977`**（13.86 fps）。板卡 `/mnt/manu/npu/out/par2l/`。

---

## 24. ❌ 混合精度（int8 输入 + 敏感层 fp16）：toolkit 2.3.0 上**建不出来**（2026-10-16）

**结论**：四条独立路径逐一实测封死。记录 `docs/experiments/2026-10-16_hybrid_precision_int8_input.md`。

**为什么「输入必须 int8」是硬约束**：int8 图输入 1 B/px ⇒ `inputs_set` **0.34 ms**；
fp16 图输入 2 B/px ⇒ **45.41 ms**（运行时用户态展宽 1,228,800 元素，37 ns/元素）。
133× 差距，混合模型若输入是 fp16 则一分不省。

**逐层量化损失（做出来了，两套方法互相印证）**

- toolkit 自己（`hybrid_quantization_step1(proposal=True, size=1)`，rc=0）：
  给出 **76 个应保 float16 的张量**，分数 0.977864–0.999718。
  Conv 29 / Mul 24 / Concat 5 / Add 4 / Slice 3 / MaxPool 2。
  最低分：`/model/b1/act/Mul` 0.9779、`/model/b0/act/Mul` 0.9882、`/model/b2/Concat` 0.99635。
- 我独立算的 SQNR（426 张量，12 张标定图，per-tensor 非对称仿射 int8）：
  p1=**27.7 dB** / p10=34.4 / 中位 40.3。最差：`/model/b2/cv2/conv/Conv` **27.24 dB**、
  `attn/Softmax` 28.73、`head/Sigmoid`/`Clip` 29.98、`/model/b2/cv1/conv` 30.16。
- ⇒ **两套方法都把 `/model/b2` 排在最前**，其次 attn Softmax 与 head 输出。损失高度集中。

**四条封死的路径**

| 路径 | 结果 |
|---|---|
| 两步式 step1/step2 | step1 通、给出层清单；**step2 崩**（`KeyError` on Sigmoid） |
| `quantized_hybrid_level=1/2/3` | **三档输入全 FLOAT16、内部内存全 105,600 KB（= fp16）** |
| UINT8 图输入 ONNX | **abort rc=134** `Not Support Dtype: 2`，quant / no-quant 同样崩 ⇒ 与量化无关 |
| `quantized_dtype='w8a16'` | toolkit 拒绝：`not support in 'rk3588'` |
| `custom_string` 混合 DSL | 2.3.0 已移除 |

**根因**：toolkit 把「图输入 dtype」与「计算精度」绑死 —— 没有开关能让输入停在 int8 而图内混精度。
step2 崩溃的具体机制：原 ONNX **77 个 Sigmoid**，图融合后只剩 **1 个**；step2 重建图后优化器
仍按 ONNX 级列表查融合后图的参数表 → KeyError。**toolkit 自身缺陷。**

**已建成但不合用**：hybridL1/L2/L3（5.74/5.74/4.48 MB，md5 `c43c94a5`/`2e87dc88`/`8a347afc`），
输入都是 FLOAT16，`inputs_set` 仍 45 ms。

**两个自己的坑（已修并注释）**：① 注入常量是 0 维标量 → 改 shape (1)；
② **`TensorProto.UINT8` 是 2 不是 1**（1 是 FLOAT），硬编码导致"输入已 uint8 ⇒ 不传 std_values"
静默失效、模型被除两次 255。**枚举值必须 import，不能硬编码。**

**板卡污染事故（已升铁律）**：失败实验的残留进程占着 NPU 上下文，加上新实验 2 worker = 3 上下文
⇒ `failed to submit!` + IOMMU page fault + soft reset，板卡掉线需断电。
⇒ **板卡测量前必须先确认无残留 demo 进程**（污染是静默的：进程还在跑、行数还在涨）。

**继续的两条路线**：① 换更新 toolkit（所有失败都在 toolkit 侧，先验 76 层清单能否喂给 step2）；
② ONNX 层自建隔离子图——需 toolkit 支持"子图豁免量化"，**目前无证据表明有**，属研究路线。

---

## 25. ✅✅✅【最新】System SOTA 后处理移植业务域：F1 87.070% → **90.605%**，三条交付判据首次全达标（2026-10-20）

> **背景**：把 Anti-UAV 的 system SOTA 后处理管线（Kalman 跟踪 + 双向时空平滑 + 碎片缝合 + 高确信插补 + 刚性剪枝）移植到龙泉山业务域，做**全 24 序列聚合的单一全局超参搜索**，并新增用户点名的「**必须连续多帧才新建目标**」连续确认门。

### 25.1 结果(业务域 inbox 口径,全 24 序列 / 165,223 帧)

| 口径 | TP | FP | FN | R% | P% | **F1%** | FAR |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 冻结单帧基线(r=768/th=0.22) | 110,788 | 10,435 | 22,469 | 83.139 | 91.392 | 87.070 | 0.0632 |
| 单帧对照(同 th_base=0.30、同 r=768、**不跑后处理**) | 104,449 | 5,710 | 28,808 | 78.382 | 94.817 | 85.819 | 0.0346 |
| **系统级全局最优(本轮)** | **114,443** | **4,919** | **18,814** | **85.881** | **95.879** | **90.605** | **0.02977** |
| 域内 14 条(剔除 10 条 `<ignore>`) | 67,801 | 1,110 | 5,187 | 92.893 | 98.389 | 95.562 | — |

- ⭐ **归因(同候选池同门限,只差跑不跑后处理)**:TP +9,994、FN −9,994、FP −791、**F1 +4.79 点**、精度还升 1.06 点 ⇒ **后处理是"用时序换召回",两者可兼得,不是变差**。
- ⭐ **冻结基线原本三条交付判据全不达标**(F1 87.07<88、R 83.14<85、FAR 0.0632>0.030);本轮**首次三条全过**。

### 25.2 全局最优超参(唯一一组,全 24 序列聚合)

`th_base=0.30` `th_salvage=0.06` `th_ground=0.22` `sky_ratio=0.60` `min_hits=3` `max_age=3`
`match_dist=16.0` `max_match_dist=24.0` `instant_conf=0.25` `min_disp=2.5` `min_track_score=0.08`
`stitch_gap=6` `infill_gap=3` `min_hits_infill=5` `min_rigid_disp=2.0` `max_rigid_var=1.0` `min_hits_prune=4`
**`dedup_radius=768.0`** **`confirm_consec=6`** `confirm_gap_tol=0` `confirm_trim_prefix=1`(Phase-A 扩展全 OFF)

> 📌 34 项参数全集的完整表格 + **可复制的完整命令** 在 `memory.md` §三十九「8. 可复制命令」/「8.1 参数全集」;完整搜索空间在 `runs/logs/sys_sota_longquan_search.json` 的 `config` 字段。

### 25.3 连续确认门消融(用户点名的效果)

| 配置 | TP | FP | F1% | FAR |
|:---|---:|---:|---:|---:|
| 无连续门(consec=1) | 116,806 | 11,051 | 89.467 | 0.0669 |
| consec=3 保留前缀 | 116,105 | 8,175 | 90.166 | 0.0495 |
| consec=3 裁前缀 | 115,881 | 7,313 | 90.373 | 0.0443 |
| **consec=6 裁前缀(交付)** | 114,443 | **4,919** | **90.605** | **0.02977** |

⇒ **FP −55.5%、F1 +1.14 点、FAR −55.5%,代价 TP 仅 −1.98%** ⇒ 该逻辑对业务域是**净收益,应固化为默认**。

### 25.4 ⭐ `dedup_radius` 是决定性变量

F1 随半径单调上升、**r≥768 饱和**(768/1024 逐位相同):r=8 → F1 76.280%;r=32 → 83.832%;r=128 → 87.980%;r=256 → 89.060%;**r=768 → 90.114%**。
机理:本域**几乎每帧只有一个目标**,r=768 等价于"每帧只留最高分的 1 个候选";半径小 → 同帧多个低分候选(多为误检)全进池 → FP 爆炸。
🔴 **上一轮把 `dedup_radius=32` 当"丰富候选池"是错的**——那是往池里灌噪声,不是让跟踪器有米下锅。
✅ **二次佐证**:最终参数组下 sweep 里「只差 `dedup_radius`」的 4 条记录 = r384 F1 90.439 / r512 90.572 / **r768 90.605** / r1024 90.605(F1 与 FAR 均与 r768 **逐位相同**)⇒ 饱和结论独立复现两次。

### 25.5 🔬 误检归因(逐帧取证,`manu/diagnostics/fp_attribution_sys_sota.py`)

把 4,919 个 FP 逐帧拆开:**`FP_DUP` = 0(0.0%)、`FP_BG` = 4,919(100.0%)**,其中插补框贡献 605(12.3%)。

- ⭐ **本域不可能"一帧多个框"**:`dedup_radius=768` ⇒ 每帧只留**最高分 1 个候选** ⇒ 24 条序列**FP 帧数 == FP 数**(每帧最多 1 个 FP)。人工若报"分裂成多个小框",先怀疑**渲染/口径**,不是算法。
- ⭐ **FP 不是边缘伪影**:87.2% 落在画面四边**内部** ⇒ "加边缘抑制"最多砍 13.8% 的 FP,不是杠杆。
- ⭐ **FP/TP 分数大量重叠**:FP 中位 0.313 / TP 中位 0.664;阈值 0.30 会同时命中 46% 的 FP 和 10% 的 TP ⇒ **靠门限清 FP 必然丢召回**。
- 🔎 **`001531` 97.872→95.436(−2.44)的根因**:用 5 个独立配置(单帧对照 / legacy / best / 去掉连续门 / 另一条路线)交叉验证,FP **恒为 11** ⇒ **与本轮调参、与连续确认门都无关**,差异全部来自"跑不跑后处理"。11 个 FP 全是 `FP_BG`,来自 2 条轨迹:`track 4476` 1 帧(最左缘)+ `track 4477` **连续 10 帧**(帧 121~130,恒在**最右缘** x≈594~610,score 0.136~0.414)。机理:`min_disp` 的静态杂波抑制**只对非天空区生效**,而该轨迹在天空区(y≈175~200 < 307)⇒ 不被抑制;再因 `instant_conf=0.25` 被 0.414 分起轨、`confirm_consec=6` 因"连续 10 帧"合法通过。人工清单的 97.872 来自**上一轮另一条管线**(`osd_global_20261019`),不是 system-SOTA。
- ⭐ **两条域内缺口的漏检是"零检出"主导**:`003002` 140/202、`002001` 2645/3076 ⇒ **检测器在复杂背景下召回塌陷**,后处理救不了(只能续上"已见过"的目标)⇒ **下一轮靶子应在检测器侧**。`001531` 召回反而是完美的(115/115)。

### 25.6 产物

- **OSD**:`/mnt/data/siping/datasets/manu/龙泉山/osd_sys_sota_20261020/{pass,un-pass,logs,per_sequence_hyperparams.tsv}`,17 pass / 7 un-pass,5.5 GB,**文件名带 F1**(`<SEQ>_F1-<f1>.mp4`)
- **画面**:🔴 红圈 = 最终系统检出(含插补 `T{id}:Infill`);🟢 绿框 = GT 命中;🟠 橙框 = GT 未命中;**逐尺度诊断细线已全部移除**
- **搜索 JSON**:`runs/logs/sys_sota_longquan_search.json`(343,081 B,**173 条 sweep** + 消融 + 逐序列,md5 `a9aa7299ead003ccaaad8965f06e3f12`)
- **未采用路线**:`runs/logs/sys_sota_longquan_search_altrun.json`(393,172 B,206 条 sweep,md5 `cbebd7c9656d3be7052b186b21cd1b22`)——F1 更高但 FAR 超线,保留备查
- **pred JSON**:`runs/sys_sota_preds/{SEQ}.json` × 24
- **新脚本**:`manu/evaluation/system_sota_longquan.py`(含 `--self-check` 正确性门)、`manu/postprocess/tune_sys_sota_longquan.py`(fork 并行)、`manu/postprocess/rank_sweep_delivery.py`、`manu/videos/render_sys_sota_longquan.py`、`manu/videos/render_sys_sota_osd.sh`、`manu/videos/verify_sys_sota_osd.py`、**`manu/diagnostics/fp_attribution_sys_sota.py`(逐帧 FP 归因)**
- `eval_bidirectional_track_fusion.py` **一行未改**(保住 Anti-UAV 0.9209 的可复现性);连续门在适配器里**后置实现**

### 25.7 性能与坑

- 管线 0.23 ms/帧 ⇒ 全 24 序列一次评估 **31~42 秒**;**173 次评估 / 24 路 fork 并行 = 829.97 秒**(墙钟预算 0.45 h)
- 渲染 24 路并行 ≈ **5 分钟**(瓶颈是 1080p mp4v 编码 153 fps/核,不是读帧 854 fps/核)
- 🔴 `CandidateStore` 预加载全部 14 个半径会 **OOM**(`_candidates` 缓存"每序列 × 每半径"合并候选,fork 后 COW 撑爆)⇒ 用 `--radii` 限半径 + `--base-params` 定基准
- 🔴 `--radii` 必须与 G5 搜索空间一致,否则 `KeyError`;🔴 `pkill -f <脚本名>` 会连自己所在 shell 一起杀;🔴 `python ... | tail` 会吞掉 python 的退出码,**必须重定向到日志文件**
- 🔴 **本地 SSHFS 镜像会被"旧版同名文件"污染**:本轮本地 `sys_sota_longquan_search.json` 存的是被覆盖那条路线(206 sweep),正本是 173 sweep ⇒ **文件大小 + sweep 条数是最快的判别位**;每次同步后必须 md5 双侧比对
- 🔴 **选型必须按交付硬约束,不能按 F1 高低**:另一条路线 F1 91.000% 但 FAR 0.0376 **超线 25%** ⇒ 不可交付;其 206 条 sweep 重排后**三判据全达标者 = 0 条**。FAR 与 Recall 在本域**互斥**,压 FAR 就掉 Recall
- 🎬 **"同样的代码渲出不一样的 OSD"**:渲染器**确定性已用逐帧像素比对证明**(同参数同输入重渲,4,378 帧平均绝对差 0.588/最大 0.807,仅 H.264 量化)⇒ 差异只能来自**输入不同**。本轮出现过**三套超参** ⇒ 三套 pred JSON ⇒ 三套 F1 ⇒ 三套文件名(013631: 97.242 / 98.412 / **97.897**;001810: 98.185 / 98.993 / **98.238**;014227: 96.491 / 98.459 / **96.460**)。🔴 **文件名内嵌 F1 ⇒ 换参后旧文件不会被覆盖、只会被新增** ⇒ 必须按最终清单显式清理
- 🔴 **"播放器打不开"≠ 文件坏**:用户报 3 条打不开,实测那 3 个路径是**已被删除的第一轮旧文件**(现役 97.897/98.238/96.460 ffprobe 全健康)⇒ 根因是**挂载/文件面板缓存保留了已删文件的条目**,刷新后即恢复。**清理旧文件后必须在交付说明里提示用户刷新挂载**
- 🎥 **OpenCV 自带的 FFmpeg 没有 H.264 编码器**(`codec_id=27` 找不到)⇒ `avc1` 走 `cv2.VideoWriter` 必失败;已给渲染器加 `--codec`,`ffmpeg:libx264` 走**系统 ffmpeg CLI 原始帧管道**(单次编码无代际损失,H.264/High/yuv420p/+faststart,66 fps)。**交付的 24 条视频编码未改、文件未替换**

### 25.8 ⚠️ 本文件索引缺口(未回填,勿误以为已同步)

`memory.md` 的 **§25~§38**(2026-10-16~10-19:e2e 三级流水线、fp16/int8 对照、双 NPU 并行、混合精度、Head 分支合并与 DW+PW 证伪、Late Upsample、head_conv、龙泉山全量重跑、OSD 复核视频、人工复核 24 序列归因、官方 SOTA 评测管线、Adapter 首轮失败根因)**尚未回填到本 compact**。本节只覆盖 §39(本轮)。**补充:§40(微调前基线冻结)已回填为本文 §26;§39 的核心结论已同步进本文 §0 快照**(业务域 F1 90.605%、三条交付判据全过)——旧 §0 里「F1 87.070% / 差 0.93 点 / FAR 2.11 倍未解决」的说法**已作废**。**需要这些章节的完整上下文时,请直接读 `memory.md`。**

---

## 26. 🔒 微调前基线冻结（2026-10-20 实测，三条全部零漂移）

> 目的：为「龙泉山业务域 Replay 微调」建立**微调前**的**可机读**基线。全部在**冻结工作点**下重跑，与已交付产物**逐位一致**。
> **微调本身未启动**，方案与判据在裁决中。详细证据与命令见 `memory.md` **§四十**。

### 26.1 三条基线

| 轨道 | 口径 | 基线值 |
|:--|:--|:--|
| **Anti-UAV 官方大盘**（防退化红线） | 单帧，Distance ≤ 8.0 px，31,613 帧 / GT 25,111 | 峰值 th=0.25：**F1 0.906361 / R 86.189% / P 95.567% / TP 21,643 / FP 1,004 / FAR 0.0317557**；`[GATE] PASS`；**四重互证**（冻结 headline ↔ `trial_0474/results.csv` ep3 ↔ 已交付 OSD ↔ 本次重跑） |
| **Val_Golden_Pool**（7 条） | **单帧臂**，17,187 帧 / GT 16,638 | **F1 98.788% / R 97.75% / P 99.85% / TP 16,264 / FP 25 / FN 374 / FAR 0.0015**；`[GATE] PASS` |
| **盲测 `001324`**（一帧未训） | **单帧臂**，11,810 帧 / GT 10,535 | **F1 83.479% / R 75.76% / P 92.95% / TP 7,981 / FP 605 / FN 2,554 / FAR 0.0512**；`[GATE] PASS` |

- 🔴 **口径铁律：三条基线全部是【单帧】臂**（`--no-pipeline`，无跟踪/缝合/插补/剪枝），因为微调改的是**前端检测器**，防退化必须卡在检测器层。系统级（带跟踪）数字另列为**补充参考**，**不可混比**。
- 系统级补充参考：黄金池 F1 **99.756%**（TP 16,574 / FP 17）、盲测 `001324` F1 **84.357%**（TP 8,024 / FP 465）。
- 产物：`runs/logs/tk_baseline_antiuav_gate.json`、`runs/logs/tk_baseline_golden_pool_single.json`、`runs/logs/tk_baseline_blind_001324_single.json` + `runs/baseline_preds_golden_single/`、`runs/baseline_preds_001324_single/`。
- **新增资产**：`manu/evaluation/system_sota_longquan.py` 加 `--seqs`（子集，保序，缺缓存报错）+ `--expect-f1/--expect-tp/--expect-fp` 闸门（不达标 `exit 2`）；新增 `manu/evaluation/system_sota_longquan_delivery_params.json`（**冻结交付 34 参数**，由 `summary.json` 的 params 块直接生成 ⇒ 不会漂移）。

### 26.2 🔴 门禁设计的两条修正（本次基线暴露，务必先改）

1. **黄金池已饱和到几乎无灵敏度**（单帧 micro F1 98.788%，掉 1.0pp 需 FP ×8）⇒ 「综合 F1 ≥ 98.5%」只能当**灾难否决**，不是精度守门员。
   逐序列「波动 ≤ 0.5%」的**量化步长就是 1 个 FP**，各序列宽容度差 **42 倍**（`001531` 最脆 ≈ +2 FP，`000818` ≈ +84 FP）。
   ⭐ **单帧口径已剥离跟踪器噪声**：`001531` 系统级 FP 11（其中 8 个是右缘噪声轨）vs **单帧只有 3** ⇒ **前端门禁必须用单帧口径**，否则门限被压到「+1.27 个 FP」的不可用水平。
   ⇒ 建议逐序列门禁改「**FP 增量绝对数**」（如 `FP ≤ 基线+2`，`001531` 单列），聚合条款保留作灾难否决。**真正灵敏度在盲测 `001324`。**
2. **Track 2「`001324` → F1 ≥ 92%」的换算**（单帧基线 TP 7,981 / FP 605；GT 固定 10,535，`F1 = 2TP/(TP+FP+10535)`）：FP 605→须 TP **9,490（+1,509）**；FP 700→9,571；FP 1,000→9,827。⇒ **每多 100 个 FP 就得多补 71 个 TP** ⇒ 任何「召回涨了」的结论必须**同时报 FP**。

### 26.3 关键判断（已被单帧对照证实）

- 🔥 **24 序列上后处理逐条都是净正收益**（`003002` +23.91pp R、`000322` +8.76pp、`002001` +1.87pp、`001324` +0.41pp；全量单帧 R 78.38% → 系统 85.88%）⇒ **前端检测器才是瓶颈，不是后处理**。
- `001324` 的 **2,511 个 FN 全部是 `fn_absent`（`fn_compete`=0）** ⇒ 帧内根本没有检出 ⇒ **后处理救不了**，必须检测器侧真实能力。
- ⚠️ **`001324` 自身 FAR 0.0394 > 0.030** ⇒ FAR 判据只能对 **24 序列聚合**用，不能按单序列卡。
- ⚠️ 微调后配对比较 Anti-UAV 时**不要加 `--expect-*`**（那三个值是自校准用，微调后必然变，会误 `exit 2`）；应看 15 档整条曲线是否整体下移。

### 26.4 🔬 跟踪器贡献分解 ⇒ 靶点排序被推翻

**挽回率 = (单帧FN − 系统FN)/单帧FN**;`1−挽回率` = **纯检测器缺口**占比。域内 17 条:单帧 FN 10,000 → 系统 7,945(挽回仅 **20.5%**)⇒ **域内约 80% 的漏检是纯检测器缺口**。

| 靶点 | 残留FN | 挽回率 | 纯检测器占比 |
|:--|--:|--:|--:|
| **`001324`**(盲测) | 2,511 | **1.7%** | **98.3%** ← 最纯 |
| **`002001`** | 3,076 | **8.9%** | **91.1%** |
| `002959` | 802 | 11.3% | 88.7% |
| `001415` | 493 | 32.1% | 67.9% |
| `003002` | **202** | 44.0% | 56.0% ← **性价比最低**(单帧 R 仅 45.71% 看着最惨,但 44% 已被跟踪器救回) |
| `000322`(ignore) | **6,919** | 15.8% | 84.2% ← 残留 FN 第一,但被标 ignore ⇒ 需裁决**按段拆分** |

- ⭐ **`001324` 留作盲测是正确的**:它是**最纯的检测器缺口**、也是最灵的探针(挽回率仅 1.7% ⇒ 每个漏检都直接对应"检测器没看见")。
- 🔴 **`001550` 跟踪器净损失 40 个真目标**(单帧 TP 2109/FN 53 → 系统 TP 2069/FN 93,挽回率 **−75.5%**,全 24 条唯一负值;系统 F1 96.121% < 单帧 97.41%)⇒ 快速运动时轨迹断链,属**后处理侧**问题,重搜参数若能救回即"白捡"。
- 🔴 **口径发现:单帧臂与系统级臂的候选池不同**,不是"加不加跟踪"的关系 —— 单帧只取 `sc >= th_base(0.30)`;系统级额外捞**天空区 `[th_salvage 0.06, th_base)` 弱候选入轨**(`eval_bidirectional_track_fusion.py:815`)。实证 `001531` 单帧 FP **3** → 系统级 FP **11**(新增 8 个全是右缘弱候选连成轨)。⇒ **两臂不可互相推算,门禁必须写死口径**。

### 26.5 待裁决（**未启动**）

① 底模是否解冻（铁律六 vs Replay 微调方案）；② 损失定义与负样本权重上限（历史 focal 负项 20,000:1 失衡已致两域双退化）；③ 切片尺寸与标签策略（宽核高斯 vs 中心峰）；④ **`000322` 是否按段拆分**（它独占 36.8% 的 FN）；⑤ 「物理辐射反转 / 机身变黑」目前**只有目视证据、无量测**；⑥ 🔥 **靶点排序按「纯检测器占比」而非 FN 绝对量**（§26.4）；⑦ 🔴 **门禁必须写死口径**（单帧臂 vs 系统级臂候选池不同）。
