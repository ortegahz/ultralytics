# No-GMC 特征重建（2026-10-27）—— 专题档

> 主题：为「去 GMC 预处理」重训 RC2 重新准备特征集。
> 核心命题：**训练集本身就是离线预处理的产物**（3 通道特征图），所以「去掉 GMC」≠ 换网络，
> 而是**必须重造特征**；且**判据基准必须同步重建**，否则门禁会拿 GMC 语义基线判 no-GMC 模型。

---

## 一、结论速览

| 问题 | 答案 |
| :--- | :--- |
| 训练读的是"特征"还是原始帧？ | **特征**。数据集目录里就是 3 通道特征图：`Ch0=I_t`、`Ch1=\|I_t−W(I_{t−2})\|`、`Ch2=(I_t−B_t)^+` |
| 去掉 GMC 要不要重造特征？ | **要**。W(·) 变了 ⇒ Ch1/Ch2 像素全变 ⇒ 训练集、门禁基线缓存、T2/T3 缓存、交付工作点**全部要重建** |
| 去 GMC ≠ 喂原始灰度帧？ | **对**。原始帧单通道，模型输入契约是 3 通道。正确含义 = **保持 3 通道结构，只把 W 换成 IDENTITY** |
| Anti-UAV 的 no-GMC 集现成吗？ | val 现成（31,613，旧脚本产出）；**train 缺失**，本次已补齐 |
| 龙泉山的 no-GMC 集现成吗？ | **完全没有**。`build_longquan_ft_dataset.py` 的 `--gmc-semantics` 只有 `{fpv, official}`，无 identity |

---

## 二、新脚本：`manu/data/build_nogmc_rebuild_dataset.py`

**蓝本**：`manu/data/build_full_median_dataset.py`（**冻结**，铁律〇之二）—— 该脚本产出的特征
与冻结集**逐字节一致**（memory `gmc_net_pending.md` §九.4 空对照臂：`200/200 抽样文件 md5 与冻结集完全相同`、
`dF1 = +0.0000`）。

**唯一变量**：`W(·) ≡ IDENTITY`（0 次仿射拟合、0 次 warp 重采样）。

**逐字复用（不构成额外变量）**：文件名清单与标签 1:1 镜像、帧索引解析（`parse_seq_and_frame`/`idx_map`）、
序列起点钳位（`max(0,...)`，21 帧历史在序列开头钳到 frame 0）、history lags（stride 递推 2,4,…,42）、
`np.median` 表达式与 `astype(np.float32)`/`np.clip(...,0,255)`/uint8 取整、`np.stack` 通道序、
`cv2.imwrite` JPEG 编码路径、计数严格校验（success≠期望 ⇒ RuntimeError）。

**内置空对照臂**：`--mode gmc` 逐字复刻 rebuild 的 GMC 路径，**仅用于验证**；
跑它重建冻结集，md5 必须逐字节相同 ⇒ 证明除 GMC 外二进制等价。

### 参数

| 参数 | 默认 | 说明 |
| :--- | :--- | :--- |
| `--ref-dataset` | `/mnt/.../manu/uav` | 参照 YOLO 集（提供文件名清单 + 标签，1:1 镜像） |
| `--raw-root` | `/mnt/.../manu/anti-uav` | 原始帧根 |
| `--output` | `/mnt/.../manu/uav_gmc_median_nogmc` | 输出目录 |
| `--splits` | `train,val` | **多 split 一次建完**（旧 nogmc 脚本是单 `--split`，data.yaml 会被覆写） |
| `--mode` | `identity` | `identity`=W≡I（交付物）｜`gmc`=逐字复刻 rebuild（仅验证） |
| `--window/--stride-step/--workers/--downscale` | 21/2/16/2 | 与 rebuild 同默认 |
| `--verify-ch0-sample N` | 0 | 建完抽 N 个文件报 Ch0 数组层 MAE vs 参照集 |

### 两套数据源共用同一脚本（official 语义两边逐行一致）

- Anti-UAV：`build_full_median_dataset.py:234-265`（rebuild 蓝本）
- 龙泉山：`build_longquan_ft_dataset.py:219-235`（official 分支）—— 逐行等价：
  `idx_prev2=max(0,ci−2)` → `compute_affine` → `warp` → `absdiff`；
  history `h_idx=max(0,ci−step*stride_step)` → `compute_affine` → `warp`；
  `len(history)>=3` → `np.median` → `np.clip(...,0,255)` → uint8；`np.stack([Ch0,Ch1,Ch2])`。

---

## 三、验证证据（本次全部实测）

| # | 验证 | 结果 |
| :-- | :--- | :--- |
| 1 | **本地合成数据空对照臂**（`/tmp/nogmc_selftest.py`） | **5/5 通过**：`--mode gmc` vs 冻结 rebuild **40/40 逐字节相同**；identity 数组层 Ch0/Ch1/Ch2 同编码路径往返后误差 **=0**；labels 1:1；计数一致；data.yaml 指向正确 |
| 2 | **服务器真实数据空对照臂**（`/tmp/verify_empty_control.py`） | **1,946/1,946 与冻结集逐字节相同**（`DJI_0175_2` 448 + `wg2022_ir_012_split_08` 1,498），不同 0、缺失 0 ⇒ **新脚本非 GMC 路径与冻结 rebuild 二进制等价** |
| 3 | **帧号覆盖检查** | Anti-UAV 199+24 序列 / 龙泉山 9+8 序列，**零缺序列、零帧号未覆盖** ⇒ 无静默 clamp 风险 |
| 4 | 吞吐标定 | 4,776 帧 / 16 s ≈ **300 fps**（24 worker，真实 640×512）；单进程 ~43 ms/帧（与 §七.4 的 34.84 ms@512×512 ×1.25 吻合） |

> ⚠️ 判据纪律（§七.2 教训）：**「Ch0 必须逐字节为 0」是错误判据**。3 通道 JPEG 的 4:2:0 色度下采样
> 会让 Ch1/Ch2 一变就带动 Ch0 解码值抖 1~8 灰阶。正确判据 = **同编码路径往返**（期望数组也走一遍
> imwrite/imread 再比）或**编码前数组层**。

---

## 四、产物与命令

### 4.1 Anti-UAV（源域，T1）

```bash
python manu/data/build_nogmc_rebuild_dataset.py \
  --ref-dataset /mnt/data/siping/datasets/manu/uav \
  --raw-root /mnt/data/siping/datasets/manu/anti-uav \
  --output /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc \
  --splits train,val --mode identity --window 21 --stride-step 2 --workers 24 \
  --verify-ch0-sample 200
```

### 4.2 龙泉山（业务域，T2 黄金池 / T3 001324）

```bash
python manu/data/build_nogmc_rebuild_dataset.py \
  --ref-dataset /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official \
  --raw-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
  --output /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc \
  --splits train,val --mode identity --window 21 --stride-step 2 --workers 24 \
  --verify-ch0-sample 200
```

标签：直接从参照集 1:1 复制（龙泉山用合并真值 `Final_Labels_merged_20261010` 的落盘结果，零字节=背景帧）。

---

## 五、训练侧必须同步重建的东西（否则门禁结构性误判）

1. **门禁基线常量是硬编码的 GMC 语义**：`manu/training/auto_epoch_gate_check.py:54-58`
   `BASE = {track1_uav: f1 0.905736/tp 21624/fp 1014, track2_golden: 0.988436/26, track3_blind: 0.837331/590}`；
   默认基线缓存 `runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl`。
   ⇒ no-GMC 必须**重测三轨基线**并参数化，否则拿 GMC 基线判 no-GMC 模型。
2. **门禁的数据根是硬编码常量**：`FT_ROOT = .../longquanshan_ft_gmc_median_official`（L61）、
   `UAV_DATA = .../uav_gmc_median/data.yaml`（L62）；`LABELS_ROOT`（L63）两套特征通用。
   ⇒ 需要给门禁加 `--ft-root` / `--uav-data` 覆盖（默认值保持不变）。
3. **U 值参照系要重建**：旧 RC1/Ep3（U=+0.027579）是 GMC 语义下测的，**no-GMC 的 U 不可横向比较**；
   必须先跑一个 **no-GMC 的 natural 对照 trial** 当新擂主，再判 RC2。

---

## 六、已有 no-GMC 消融数字的正确解读（不要误用）

2026-10-08 冻结底模消融（官方 val 24 序列 / 31,613 帧 / 25,111 GT，th=0.25，Distance≤8.0px）：

| 臂 | F1 | ΔF1 | Recall | Precision | TP | FP |
| :--- | --: | --: | --: | --: | --: | --: |
| native（GMC 冻结集） | 0.906361 | — | 86.19% | 95.57% | 21,643 | 1,004 |
| tree（anchor_step=10，交付方案） | 0.906050 | −0.000311 | 86.18% | 95.51% | 21,641 | 1,018 |
| **no-GMC（W≡I）** | **0.900945** | **−0.005417** | 85.64% | 95.03% | 21,506 | 1,124 |

- dF1 在 th 0.10~0.45 **全区间恒定** −0.0033~−0.0054、两臂峰值同在 th=0.25 ⇒ **不是阈值平移，重调门限救不回**
- 掉分**高度集中**：5 条 GMC 空转序列上删 GMC **净收益 +0.0066 F1**；总 dTP −136 中
  `wg2022_ir_052_split_08` 独占 −74（54%）、`DJI_0051_2` −22（16%）
- 🔴 **限定**：这是「**GMC 对当前冻结权重的边际价值**」，**不是「GMC-free 方案的可达上限」**。
  冻结权重从未见过未对齐的差分通道。§七.5 已写明：*"若掉分不可接受，正确下一步是铁律五的
  Twin A/B 冷启动在 no-GMC 数据上同构重训，而不是继续拿冻结权重硬扛域偏移（历史「改模型」0 胜 8 负）"*。
  **本次重训正是来补这个洞的。**
- 成本侧：删 GMC 共省 **121.54 ms/帧（66.3%）**，单帧 183.32 → 61.78 ms（**2.97×**）；
  但残留 61.78 ms 里 **61.71 ms（99.9%）是 21 帧全图 median**（大头是 21 次 warp ~80 ms，不是拟合 39.66 ms）。

---

## 七、构建实测结果（2026-10-27，服务器 80 核 / 125 GB / 4×4090D 全空闲）

### 7.1 逐 split 耗时（24 worker）

| 集 | split | 帧数 | 序列数 | 耗时 | 吞吐 | 失败 |
|:--|:--|--:|--:|--:|--:|--:|
| Anti-UAV | train | 43,008 | 199 | **143.6 s** | **299.5 fps** | 0 |
| Anti-UAV | val | 31,613 | 24 | **134.8 s** | **234.4 fps** | 0 |
| 龙泉山 | train | 13,918 | 9 | **281.6 s** | 49.4 fps | 0 |
| 龙泉山 | val | 28,997 | 8 | **806.6 s** | 35.9 fps | 0 |
| **合计** | | **117,536** | 240 | **1,366.6 s ≈ 22.8 min** | | **0** |

**两端速率差 6~7 倍，原因已定位（不是代码问题）**：

1. **并行度**：Anti-UAV 有 199/24 条序列 ⇒ 24 worker 全忙；龙泉山只有 9/8 条 ⇒ 只有 9/8 个 worker 在跑，
   尾部由最长序列决定（`VIDEO00007_19700101_001324` 一条 11,810 帧）。
2. **原始 JPEG 体积**：龙泉山原始帧 **142 KB/张** vs Anti-UAV **11 KB/张**（12.7×）⇒ 解码成本差一个量级。

⇒ **预估口径修正**：特征准备 ≈ **23 min**（不是 10~12 min），瓶颈在龙泉山大码率帧的 JPEG 解码，
不是 `np.median`、不是 I/O 带宽。若要提速：把龙泉山序列按帧段切片再并行（当前按序列切分）。

### 7.2 产物验收

| 项 | Anti-UAV | 龙泉山 |
|:--|:--|:--|
| train images / labels | 43,008 / 43,008（零字节标签 2,247=背景帧） | 13,918 / 13,918（零字节 1,798） |
| val images / labels | 31,613 / 31,613（零字节标签 6,502） | 28,997 / 28,997（零字节 1,824） |
| 零字节图像 | **0** | **0** |
| 计数严格校验 | `[VERIFY PASSED] 43008/43008` | `[VERIFY PASSED] 13918/13918`、`28997/28997` |
| 体积 | 4.7 GB | 2.4 GB |
| `data.yaml` | `train: images/train` / `val: images/val` ✓ | 同左 ✓ |
| Ch0 数组层自检（vs 参照集，抽样 200） | train MAE **1.81**（max 3.72）／val MAE **1.32**（max 2.84） | train MAE **0.98**（max 3.47）／val MAE **1.33**（max 4.62） |

Ch0 自检值即 JPEG 4:2:0 色度底噪量级（memory §七.2：判据下到数组层，**不是**「逐字节为 0」）。

### 7.3 🔑 最强的一条交叉验证

**新产物 vs 旧 nogmc 产物（不同蓝本的两个独立实现）：31,613/31,613 逐字节完全相同（不同 0）。**

- 旧 nogmc 脚本蓝本 = `build_sample_median_dataset.py`；新脚本蓝本 = `build_full_median_dataset.py`（rebuild）
- 两个**独立实现**在 no-GMC 语义下产出**逐字节一致** ⇒ 语义实现无歧义
- 叠加「`--mode gmc` 与冻结 rebuild 1,946/1,946 逐字节相同」⇒
  **no-GMC 特征集与冻结 GMC 特征集的差异，有且仅有 GMC 本身**

### 7.4 产物路径

```
/mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/            # T1 源域，4.7 GB
  ├── images/train  43,008   images/val  31,613
  ├── labels/train  43,008   labels/val  31,613
  └── data.yaml                # train: images/train ✔（旧脚本曾误指 images/val，已修正）
/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc/ # T2/T3 业务域，2.4 GB
  ├── images/train  13,918   images/val  28,997
  ├── labels/train  13,918   labels/val  28,997
  └── data.yaml
```

龙泉山 val 的 8 条序列包含 **T2 黄金池 7 条**（`VIDEO00009_19700101_001531` 等）+ **T3 盲测
`VIDEO00007_19700101_001324`（11,810 帧）** ⇒ 三轨全覆盖。

### 7.5 新脚本三侧 md5

`manu/data/build_nogmc_rebuild_dataset.py` = `2a35a4b0110a7a95ecd8d817dbf29c89`
（本地仓库 = SSHFS 镜像 = 服务器，三侧一致，铁律〇）

### 7.6 训练数据管线冒烟（**纯 CPU、零 GPU、零训练**，2026-10-27）

复用 worker 自己的函数（`parse_mix_spec` / `LQ_EXPECT` / `UAV_EXPECT`）跑通训练前置链路：

| 检查 | 结果 |
|:--|:--|
| A. 两份 no-GMC `data.yaml` 解析 | ✅ train 均指向 `images/train`、classes = `['uav']` |
| B. `build_yolo_dataset` 样本数 | ✅ LQ **13,918** / UAV **43,008** ⇒ **DATA-GATE 会 PASS**（满额 56,926，与 trial_0080 同值） |
| C. 口径 1 加权采样实测（w2.5, seed=80） | ✅ LQ 16,087 / UAV 40,839 = **1:2.539**（目标 1:2.500；trial_0080 实测 1:2.524，同量级）；总抽样量 56,926 = 满额 |
| D. 真实 batch | ✅ `img` **(32, 3, 640, 640) uint8**（3 通道 ✓）；`bboxes` (31,4)；`batch_idx` 长度 = 框数且取值 ∈ [0,bs)；heatmap target = `{'heatmap':(32,1,320,320), 'offset':(32,128,2), 'ind':(32,128), 'mask':(32,128)}`，**无 NaN** |
| E. 标签完整性 | ✅ LQ 有标签 **12,120** / 背景 1,798；UAV 有标签 **40,761** / 背景 2,247 —— 与磁盘真值、ultralytics 扫描行（`0 corrupt`）**逐项吻合** |

> 与 memory「口径3 = 12,120 : 40,761」**完全一致** ⇒ no-GMC 特征集在**结构层面**与冻结 GMC 集等价，
> 差异只存在于 Ch1/Ch2 的像素内容（§7.3 的逐通道体检已证）。

**两个把坑记录下来的点（我自己踩的）**：

1. `ds.labels` 在这个 ultralytics fork 里是 **`list[dict]`**（每元素键：`bbox_format/bboxes/cls/im_file/
   keypoints/normalized/segments/shape`），**不是** `list[np.ndarray]` ⇒ 用 `len(lbl)` 当"框数"会把
   **dict 键数(8)** 当框数，会得出"每图恰好 8 框"这种假象。判框数要取 `len(lbl["bboxes"])`。
2. `batch_idx` 是**每框一个**（ultralytics 语义），不是每图一个 ⇒ 长度应等于框数，不等于 batch size。

**副产物**：ultralytics 在两个 no-GMC 数据集的 `labels/` 下生成了 `train.cache`（标准行为，训练也会生成；
它同时把首次 epoch 的标签扫描时间省掉）。

### 7.7 判定链端到端自检（**纯 CPU、零 GPU**，2026-10-27）

用**合成 `gate_report.json`** 走一遍 worker 的判定链，验证参照系注入真的生效：

| # | 检查 | 结果 |
|:--|:--|:--|
| A | 同一候选在 **GMC 参照系**（模块常量）下 | ✅ 一级红线 **FAIL_TRACK1_RETREAT**（U = −inf） |
| B | 同一候选在 **no-GMC 参照系**下 | ✅ **PASS**，U(vs 冻结基线) = **+0.046000** |
| C | 结论**确实翻转** ⇒ 注入不是空操作 | ✅（不改参照系就必然误判：no-GMC 候选会被 GMC 基线判死） |
| D | U(vs RC1) 用**注入的 rc1** | ✅ 注入 rc1 → **+0.010500**；模块常量 RC1_EPOCH3（GMC 语义）→ **−0.009218**（与 no-GMC 候选比无意义） |
| E | 缺字段的 gate_report | ✅ 抛 `GateReportIncomplete`（基础设施失败 ≠ 指标未达标） |

> 这条自检是**「为什么必须换参照系」的最短证明**：同一个候选，换参照系前 FAIL、换后 PASS。

---

---

## 八、判定链路的「参照系注入」改造（2026-10-27 已完成）

### 8.1 改了什么

| 文件 | 改动 | md5 |
|:--|:--|:--|
| `manu/training/auto_epoch_gate_check.py` | 新增 `--uav-data` / `--ft-root`（默认 = 原硬编码常量 `UAV_DATA`/`FT_ROOT`）+ `--base-track{1,2,3}-{f1,tp,fp,fn}`（默认 = `BASE`）；新增模块级 `effective_base(args)` 便于测试 | `b067e3d6ee59446df022a0ea95088643` |
| `manu/training/optuna_mix_ratio_search.py` | 新增 `--reference-frame`（参照系包 JSON，worker 与调度器都收）；把 `--uav-data`/`--ft-root`/`--base-*` 透传给门禁；判定改用注入的 `frozen_baseline`/`rc1`/阈值 | `3e300fba3c2cffb0883312df92ab9948` |
| `manu/training/measure_reference_frame.py` | **新增工具**：在指定特征语义上实测冻结基线（口径与门禁逐项对齐），产出 `reference_frame.json` | `49d589148d3a753bed1a68088822881a` |

`manu/training/mix_ratio_judgement.py` **零改动** —— 它的 `hard_gate`/`utility`/`evaluate_model_quality`
本来就接受 `baseline` 参数，注入即可。

### 8.2 参照系包 JSON 结构（`reference_frame.json`）

```json
{
  "frozen_baseline": {"track1_f1":..,"track1_fp":..,"track2_f1":..,"track2_fp":..,
                      "track3_f1":..,"track3_fp":..},
  "rc1":             {同上},          // ← 由该语义下的 natural 对照 trial 填入
  "gate":            {"track2_f1_min":0.985,"track2_fp_max_increment":10,
                      "track1_min_delta":-0.005},
  "gate_base":       {"track1_f1":..,"track1_tp":..,"track1_fp":.., ... 12 项}
}
```

### 8.3 训练时的完整命令（一条命令一套参照系）

```bash
# ① 测 no-GMC 参照系（T1 基线臂 + T2/T3 基线臂）
python manu/training/measure_reference_frame.py \
  --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median_nogmc/data.yaml \
  --ft-root  /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_nogmc \
  --weights  runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --out-dir  runs/nogmc_ref --label nogmc --gpus 0,1,2,3

# ② natural 对照 trial（新 RC1 等价擂主）→ 把它的 gate_report.json 写进参照系包的 rc1
python manu/training/measure_reference_frame.py \
  --out-dir runs/nogmc_ref --fill-rc1-from \
  runs/nogmc_ref/trials/trials/trial_0090/gate/epoch3/gate_report.json

# ③ RC2（trial_0080 配置）—— 4 卡并行（natural 对照 + RC2 + RC2 两个种子）
bash manu/training/run_nogmc_trials.sh launch
bash manu/training/run_nogmc_trials.sh status
```

### 8.3.1 `run_nogmc_trials.sh`（新增，4 卡并行启动器）

| trial | GPU | mix_spec | seed | 用途 |
|:--|--:|:--|--:|:--|
| 0090 | 0 | `natural` | 0 | **natural 对照** ⇒ 新 RC1 等价擂主的来源 |
| **0080** | 1 | `w2.5` | 0 | **RC2 = trial_0080 配置**（采样 seed = 0+80 = 80，与原 trial_0080 同） |
| 0091 | 2 | `w2.5` | 91 | RC2 种子确认 |
| 0092 | 3 | `w2.5` | 92 | RC2 种子确认 |

三种模式：`smoke`（1 trial / 2 iters / 跳过门禁，提交前预检）｜`launch`（4 卡并行）｜`status`（打印各 trial 的 verdict / U / U(vs RC1)）。
超参与 trial_0080 **逐项一致**（3 epoch + 5 epoch cosine 周期、w2.5、lr_scale 2.270182993957786、warmup 0.5877289830563912、
wd 0.00026905793288047476、focal 2.75、offset 0.40、mosaic 0.15、scale 0.10、translate 0.05、clip 5.649530591182026、freeze_bn 0），
**唯一变量 = 特征语义 + 参照系**。

> ⚠️ `launch` 会先检查 `reference_frame.json` 存在且 `rc1` 非空，否则**拒绝启动**（防止拿 GMC 参照系判 no-GMC 模型）。

### 8.4 顺带发现的一个既有小 bug（不影响管线）

`manu/data/multiscale_heatmap_cache.py --help` 会崩：
`TypeError: %u format: a real number is required, not dict`（某个 help 字符串里有 `%u`）。
**与本次改动无关**（门禁一直用真实参数调用它，从不走 `--help`）。修它属独立小任务。

---

## 九、执行清单（2026-10-27 已启动，跨夜运行中 → 详见 §十）

1. ✅ **参照系重建**：474 去 P0 在 no-GMC 三轨实测 ⇒ `runs/nogmc_ref/reference_frame.json`
   （T1 已出：F1 0.900897 / TP 21499 / FP 1118，ΔF1 −0.004840 vs GMC；T2/T3 见包内）
2. 🔄 **natural 对照 trial**（新擂主）⇒ 编排脚本在 trial_0000 出报告后**自动填 rc1**（§10.3）
3. 🔄 **搜索进行中**：`runs/optuna_nogmc`（4 卡，19 h 预算，约 100~110 trial）
   —— 里面的 **w2.5 档** 就是 no-GMC 版的 RC2（trial_0080 的配置）
4. ⏳ **多种子确认**：从 w2.5 档取多种子重跑，**交接线要按 no-GMC 口径重新定标**
   （不能套 GMC 的 +0.045 —— 见 §10.7）
5. 4 卡并行，门禁阶段自动挑空闲卡；每 trial 门禁固定成本 ≈ 1,116 s（其中龙泉山 val 缓存重建 970 s，
   `auto_epoch_gate_check.py` 强制 `--overwrite`，**无复用开关**）。



## 十、no-GMC RC2 重训流水线（2026-10-27 启动 · 跨夜无人值守）

> ⚠️ **本机（host）晚上断电、mcode 会话会退出，但服务器不断电** ⇒ 所有跨会话的步骤
> 必须落在服务器上、用 `setsid nohup` 常驻。本节记录的就是那条链。

### 10.1 底模确认（用户记忆核对）

用户记忆「RC2 0080 是基于 SOTA model 474 微调的（w/o p0）」—— **核对成立**：
- `trial_0080` 的 worker 用 `--weights runs/optuna_p0_nas/trial_0474/weights/best.pt`
- 且 `load_hm_model(..., use_p0_highway=False)` ⇒ **去 P0**
- 基线锚 `BASE["track1_uav_withp0"]`（0.906361/21643/1004）就是**带 P0** 的版本，
  而判据用的是**去 P0** 的 0.905736/21624/1014 ⇒ 两者不是一回事，别混。

### 10.2 四个新增/改动的脚本

| 文件 | 作用 | md5 |
|:--|:--|:--|
| `manu/training/launch_nogmc_mix_ratio_search.sh` | **no-GMC 版搜索启动器**：与 `launch_mix_ratio_search.sh` **配置逐项完全一致**（底模/epochs 3/batch 32/imgsz 640/workers 8/4 卡/TPE/种子 trial/门禁脚本/截止机制/断点续跑），**只换三样**：① 数据 → no-GMC 特征集 ② 基线缓存+参照系 → no-GMC 语义 ③ 输出根+study 名 → `runs/optuna_nogmc` / `mix_ratio_search_nogmc`（独立 study.db，不污染 GMC 研究） | `584b79889d34789c8db35da2eb671657` |
| `manu/training/run_nogmc_pipeline.sh` | **无人值守编排**：① 等参照系 → ② 起搜索 → ③ natural 对照 trial_0000 出报告后**自动填 rc1** → ④ 搜索结束补采 CSV 漏行 → ⑤ 生成 `SUMMARY.md`；含 **07:00 硬截止保护**（SIGTERM 调度器，它有优雅处理：停派新 trial、等在跑的收尾） | `72022a46ce9c474663fd77507b794a83` |
| `manu/training/summarize_nogmc_search.py` | 结果汇总器 → `runs/optuna_nogmc/SUMMARY.md`：**U 一律重算**（`utility()`，不信 CSV 列）+ 分档统计 + Top-N + 参照系 + 口径声明 | `17e7b11eef440ccd4408fbfe7c411eba` |
| `manu/training/measure_reference_frame.py` | 参照系测量器（含 `--fill-rc1-from`） | `21832e9aeeff068f7c5ad0a265df7960` |

### 10.3 关键机制：rc1 为什么必须自动填

- 搜索的**种子 trial**是 `natural / w2.0 / w3.0 / w4.0 / w6.0`（`enqueue_trial`，trial_0000 = **natural 对照**）
- no-GMC 语义的**新擂主**就是这个 natural 对照 ⇒ 它出报告后必须写进参照系包的 `rc1`
- **worker 每次启动都重新读 `--reference-frame`** ⇒ 编排脚本在 trial_0000 落地后回填，
  **后续 trial 自动用上新擂主**，无需重启调度器（这正是 `runtime_config.json` 那套"不重启改协议"的思路）
- 前几个 trial 的 `utility_vs_rc1` 列在回填前是用 GMC 常量算的（**无意义**，但不影响 objective）

### 10.4 no-GMC 参照系基线（474 去 P0 · 2026-10-27 实测）

**T1 Anti-UAV 大盘 val（31,613 帧 / 25,111 GT）**：

| 指标 | GMC 特征 | no-GMC 特征 | Δ |
|:--|--:|--:|--:|
| F1 | 0.905736 | **0.900897** | **−0.004840（−0.48 pp）** |
| TP | 21,624 | 21,499 | −125 |
| FP | 1,014 | 1,118 | +104 |
| Recall | 86.115% | 85.615% | −0.50 pp |
| Precision | 95.522% | 95.057% | −0.47 pp |

**与 memory §8.1 的交叉验证**：§8.1 记的 no-GMC 消融是 F1 0.900945 / ΔF1 −0.005417（**固定 th=0.25** 口径）；
上表是 **15 档扫参的 peak**（与门禁基线同口径）⇒ 差在口径，结论一致（**T1 掉约 0.5 pp**）。
且新旧 no-GMC val 集 **31,613/31,613 逐字节相同** ⇒ 是同一份特征上的独立复测。

**换算后的一级红线（新基线）**：T1 F1 ≥ 0.900897 − 0.005 = **0.895897**（旧 GMC 口径是 0.900736）——
已由参照系包自动带进每个 trial 的门禁与判定（`--base-*` + `--reference-frame`）。

**T2 黄金池 / T3 盲测 001324**：见 `runs/nogmc_ref/reference_frame.json`（生成于 2026-10-27 上午）。

### 10.5 时间线（截止 2026-10-28 09:00）

| 阶段 | 计划 | 结束 |
|:--|:--|:--|
| 参照系测量（474 去 P0 三轨） | 09:53 起，T1 已完成 | ~10:20 |
| 搜索（4 卡，19 h 预算） | 参照系就绪即自动起 | 派单止于次日 05:20，末批收尾 ~06:05 |
| 补采 CSV + SUMMARY.md | 自动 | ~06:10 |
| **缓冲** | | **≈ 2.8 h** |

预计 **100~110 个 trial**（上次 GMC 搜索 103，同量级）。

### 10.6 明早怎么读（会话重启后）

```bash
ssh -p 32222 huangzhe@192.168.99.40 "cat /tmp/pycharm_project_10ae9e2e/runs/optuna_nogmc/SUMMARY.md"
```

其它入口：
- 实时状态：`python manu/training/status_mix_ratio_search.py --out-root runs/optuna_nogmc --study-name mix_ratio_search_nogmc`
- 管线日志：`runs/nogmc_pipeline.log`｜调度器日志：`runs/optuna_nogmc/logs/dispatcher.log`
- 逐 trial 日志：`runs/optuna_nogmc/logs/trial_XXXX.log`｜产物：`runs/optuna_nogmc/trials/trial_XXXX/`
- 参照系包：`runs/nogmc_ref/reference_frame.json`（含 `frozen_baseline` / `rc1` / `gate_base`）

### 10.7 口径纪律（给明早的读盘人）

1. **no-GMC 的 U 与 GMC 的 U 不可横向比较**（参照系不同）。只能说"在 no-GMC 语义下最好的是 X"。
2. **no-GMC 的 RC2 交接线要重新定标**：不能套 GMC 的 +0.045 —— 应取本表内 w2.0~w2.5 档 U 的上分位
   + 多种子中位数重新定（§10.3 的 rc1 是"新擂主"，不是"交接线"）。
3. **U 一律重算**，不信 CSV 列。
4. 掉点归因沿用 §8.2：**dF1 在 th 0.10~0.45 全区间恒定 ⇒ 不是阈值平移**；
   掉点集中在 `wg2022_ir_052_split_08`、`DJI_0051_2` 等少数序列。

> 原则：**纯增量、默认值 = 现有 GMC 常量 ⇒ 不传任何新参数时行为逐位不变**。
> 自检：门禁 4/4 过、worker 17/17 过（本地 + 服务器 `uav` 环境各跑一遍）。


---

## 十一、GMC vs no-GMC 三轨对照（同一底模 474 去 P0，2026-10-27 10:12 实测）

### 11.1 可比性前提（必须成立，否则表不成立）

| 维度 | GMC 侧 | no-GMC 侧 | 是否同 |
|:--|:--|:--|:--|
| 底模 | `runs/optuna_p0_nas/trial_0474/weights/best.pt`（去 P0） | 同 | ✅ |
| 工作点参数 | `manu/evaluation/system_sota_longquan_delivery_params.json` | 同 | ✅ |
| **尺度口径（T2/T3 热图缓存）** | `multiscale_heatmap_cache.py --scales 160,80 --region-threshold 0.06 --main-threshold 0.22 --fusion-dilate 32 --min-area 4 --ablate-p0` ⇒ **多尺度**（640+160+80 三尺度热图逐像素 `np.maximum` 融合） | **逐参数相同** | ✅ |
| **尺度口径（T1 推理）** | `cache_trial0474_inferences.py --imgsz 640 --conf-thresh 0.02 --top-k 100 --ablate-p0` ⇒ **单次 640×640 前向**（无多尺度融合环节） | **逐参数相同** | ✅ |
| **后处理口径（T2/T3）** | `system_sota_longquan.py --no-pipeline` ⇒ **单帧臂**（无跟踪 / 无缝合 / 无插补 / 无剪枝） | 同（实测 `track2_baseline.json` 里 `pipeline: false`） | ✅ |
| **评测口径（T1）** | `eval_arm_pc_gmc_ab.py --tol 8.0` 15 档门限扫 ⇒ **取峰值** | 同 | ✅ |
| T1 帧集 / GT | 31,613 帧 / 25,111 | 同 | ✅ |
| T2 黄金池 7 序列 / GT | 17,187 帧 / 16,638 | 同 | ✅ |
| T3 盲测 001324 / GT | 11,810 帧 / 10,535 | 同 | ✅ |
| 唯一变量 | W(·) = GMC | W(·) = IDENTITY | — |

GMC 侧数值来源 = 冻结常量 `manu/training/auto_epoch_gate_check.py::BASE`（2026-10-21 冻结）；
no-GMC 侧 = 服务器实测 `runs/nogmc_ref/track{1,2,3}_baseline.json`。
两侧 F1 均由 TP/FP/FN 独立重算并与报告值逐项自洽（差 < 1e-6），TP+FN=GT 校验通过。

### 11.2 对照表

| 轨道 | 指标 | GMC | no-GMC | Δ(绝对) | Δ(pp) |
|:--|:--|--:|--:|--:|--:|
| **T1** 源域大盘 | F1 | 0.905736 | **0.900897** | −0.004839 | **−0.484** |
| | Recall | 0.861137 | 0.856159 | −0.004978 | −0.498 |
| | Precision | 0.955208 | 0.950568 | −0.004640 | −0.464 |
| | TP | 21,624 | 21,499 | −125 | — |
| | FP | 1,014 | 1,118 | +104 | — |
| | FN | 3,487 | 3,612 | +125 | — |
| **T2** 黄金池 | F1 | 0.988436 | **0.981074** | −0.007361 | **−0.736** |
| | Recall | 0.978663 | 0.965861 | −0.012802 | −1.280 |
| | Precision | 0.998406 | 0.996775 | −0.001631 | −0.163 |
| | TP | 16,283 | 16,070 | −213 | — |
| | FP | 26 | 52 | +26 | — |
| | FN | 355 | 568 | +213 | — |
| **T3** 盲测 001324 | F1 | 0.837331 | **0.757812** | −0.079519 | **−7.952** |
| | Recall | 0.760513 | 0.641101 | −0.119411 | **−11.941** |
| | Precision | 0.931411 | 0.926475 | −0.004937 | −0.494 |
| | TP | 8,012 | 6,754 | −1,258 | — |
| | FP | 590 | 536 | −54 | — |
| | FN | 2,523 | 3,781 | +1,258 | — |

### 11.3 解读

1. **T3 是真正的代价所在**：F1 −7.95 pp、Recall −11.94 pp（漏检多 1,258 个），
   而 Precision 只掉 0.49 pp、FP 反而少了 54 ⇒ **掉的是召回，不是虚警**。
   物理解释：001324 是移动背景下的小目标难例，GMC 负责把全局运动压掉；
   去掉 GMC 后背景运动被当成目标特征，检测被稀释 ⇒ 漏检。
2. **T2 黄金池只掉 0.74 pp**，且 FP 从 26 涨到 52（绝对值仍极小，16,638 GT 里误报率
   0.156% → 0.312%）；Recall 掉 1.28 pp ⇒ 黄金池**未崩**。
3. **T1 源域 −0.48 pp**，量级与 GMC 侧微调本身的代价（ΔF1_T1 ≈ −0.0038~−0.0054）同档。
4. **GMC 的作用排序：T3 >> T2 > T1** ⇒ GMC 主要买的是**难例召回**，不是大盘精度。

### 11.4 对一级红线的影响（判据口径本身未改，只换参照系）

| 红线 | GMC 侧 | no-GMC 侧 | 状态 |
|:--|:--|:--|:--|
| T2 绝对线 F1 ≥ 0.985 | 基线 0.98844 过线 | **基线 0.98107 本身不过线** | ⚠️ **唯一风险项** |
| T2 ΔFP ≤ +10 | 基线 FP 26 ⇒ 上限 36 | 基线 FP 52 ⇒ 上限 62 | ✅ 相对口径已随基线自动平移 |
| T1 ΔF1 ≥ −0.005 | 相对 0.905736 ⇒ 下限 0.900736 | 相对 0.900897 ⇒ **下限 0.895897** | ✅ 已自动带入（`--base-*` + `--reference-frame`） |

**绝对线 0.985 是"交付物在黄金池上的业务质量要求"，与特征语义无关 ⇒ 保留不动**。
但 no-GMC 下基线离它只剩 −0.00393，裕量比 GMC 侧（+0.00344）更薄：

- GMC 侧实测：微调把 T2 从 0.988436 抬到 0.998979（**+0.01054**）。
- no-GMC 侧若拿到相近增益 ⇒ T2 ≈ 0.9916 ⇒ **绝对线仍可达**（预计裕量 +0.0066）。
- **决策规则（写死，不临场拍脑袋）**：以 **trial_0000（natural 对照）** 的实测 T2 F1 为准。
  - 若 trial_0000 的 T2 F1 **≥ 0.985** ⇒ 绝对线可用，一切照常。
  - 若 trial_0000 的 T2 F1 **< 0.985** ⇒ 说明该绝对业务线在 no-GMC 语义下**不可达**，
    此时**不改代码、不擅自放宽容差**，出报告请用户裁决（建议二选一：
    ① 绝对线重标定到 no-GMC 基线 −0.005 ≈ 0.976；② 改成"相对基线 ΔF1_T2 ≥ +0.008"）。
- 受控放宽通道的 `track2_f1_min_gain = +0.008` 是**相对基线**，量纲自动正确，不受影响。

### 11.5 U 量纲（不可横比）

- no-GMC 参照系下 **基线 U = 0**（`U = ΔF1_T3 + 0.5·ΔF1_T2 − 2.5·max(0,−ΔF1_T1) − 0.0005·max(0,ΔFP_T2)`，
  Δ 全部相对 no-GMC 基线）。
- GMC 侧 trial_0080 的 **U(vs 冻结基线) = +0.052553** 与 no-GMC 侧的任何 U **不可横向比较**；
  no-GMC 侧只能说"相对 no-GMC 基线最好的是 X"。
- 由于 no-GMC 基线的 T3 已低 7.95 pp，no-GMC 侧的**绝对 T3 F1 交付目标必须重新定标**
  （不能沿用 GMC 侧的 0.871684 / 0.897694 之类数值）。

### 11.6 「single-scale 还是 multi-scale」的澄清（2026-10-27 10:30 专项核实）

容易被混的是**两个不同维度的 single/multi**，必须分开说：

| 维度 | 本次对照表的口径 | 说明 |
|:--|:--|:--|
| **热图尺度（scale）** | **多尺度** | T2/T3 走 `multiscale_heatmap_cache.py --scales 160,80` ⇒ **640 + 160 + 80 三尺度热图逐像素 `np.maximum` 融合**；T1 则是**单次 640×640 前向**（本来就没有多尺度融合环节）。**两侧逐参数相同** |
| **帧级后处理（frame）** | **单帧** | T2/T3 两侧都是 `system_sota_longquan.py --no-pipeline`（单帧臂，无跟踪/缝合/插补/剪枝）—— 因为微调改的是**前端检测器**，防退化必须卡在检测器层 |

⚠️ 两个容易踩的坑：

1. **「单尺度 ONNX」不在本链路里**。只跑 640×640 一次的单尺度 ONNX 是 **RK3588 上板 / 全量 PT 臂**的口径
   （见 `manu/memory/PTQ_RK3588_int8_报告.md`），**门禁与评测大盘都是多尺度**。别拿上板数字和本表混。
2. **文档里的 `_single` 后缀指的是「单帧」不是「单尺度」**。
   `manu/docs/baselines/lq_baseline_semantics_compare.md` 的 `ft_golden_pool_single` /
   `ft_blind_001324_single` 指的是 `--no-pipeline` 的**单帧裸跑**，其缓存**仍是多尺度 160,80**。

**实证闭环（GMC 侧基线用同一脚本重扫，2026-10-27 10:28，服务器，纯 CPU 零 GPU）**：

```bash
python manu/evaluation/eval_arm_pc_gmc_ab.py \
  --arm-a runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl --label-a GMC_BASE \
  --arm-b runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl --label-b GMC_BASE_SELF \
  --tol 8.0 --out-json <OUT>/track1_baseline.json --quiet
```

| | 峰值 th | F1 | TP | FP | FN | Precision | Recall | 帧 / GT |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| GMC（冻结缓存重扫） | **0.25** | 0.905736 | 21,624 | 1,014 | 3,487 | 0.955208 | 0.861137 | 31,613 / 25,111 |
| no-GMC（同脚本同参数） | **0.25** | 0.900897 | 21,499 | 1,118 | 3,612 | 0.950568 | 0.856159 | 31,613 / 25,111 |

⇒ **同工作点（th=0.25）、同脚本、同 tol、同帧集同 GT、同 peak 口径**，
且 GMC 侧冻结常量 0.905736/21,624/1,014 被独立复现 ⇒ §11.2 的对照表**口径成立**。

---

## 十二、跨夜运行期的踩坑与修复（2026-10-27 上午）

### 12.1 🔴 已修复：rc1 自动回填会被 argparse 静默打死（真 bug，已实证复现）

| 项 | 内容 |
|:--|:--|
| 现象 | 无人值守管线 `run_nogmc_pipeline.sh` 第③步执行 `python manu/training/measure_reference_frame.py --out-dir runs/nogmc_ref --fill-rc1-from <GR>` ⇒ **argparse 退出码 2**，rc1 根本没写进去 |
| 根因 | `--uav-data` / `--ft-root` 被设成 `required=True`，而 **argparse 的必填校验发生在任何业务分支之前** ⇒ 即使用了 `--fill-rc1-from`（这个模式根本不需要这两个参数），也直接被拒 |
| 二次伤害 | 管线那一步 `python ... | tee -a "$PIPE_LOG"` 之后**无条件** `RC1_DONE=1` ⇒ 失败也会打印「rc1 填充：完成」 ⇒ **静默失败** |
| 实证 | 2026-10-27 10:22 服务器实测：`error: the following arguments are required: --uav-data, --ft-root`，退出码 2 |
| 修复 | `manu/training/measure_reference_frame.py`：两个参数改 `default=""`，并在**测量分支内**显式 `ap.error(...)` ⇒ **测量模式的必填语义逐字保留**，只解锁 fill 模式 |
| 回归 | 5/5 过：① fill 免参可用 ② 测量模式缺参仍退出码 2 ③ 测量模式带参行为不变 ④ 重复填幂等 ⑤ 坏文件明确报 `GateReportIncomplete`（不静默降级） |
| 三侧 md5 | `2ecebaeb9e092ffb0cc4e29cdc6ed8d3`（本地 / SSHFS / 服务器一致） |

### 12.2 🛡 第二道保险：`manu/training/nogmc_rc1_watchdog.sh`（服务器常驻）

19 小时的 GPU 产出不能只押在管线一处。watchdog（PID 2085077，`setsid nohup`）逻辑：

1. 等 `runs/optuna_nogmc/trials/trial_0000/gate/epoch3/gate_report.json` 落地；
2. **故意晚 180 秒**再动手 —— 让管线自己先填，**规避两个进程并发写同一个非原子 JSON** 的竞态；
3. 管线已填好 ⇒ 只校验不重复写；管线没填上 ⇒ 由 watchdog 补填；
4. 最后校验 rc1 六个字段齐全，不齐就以非零码退出并留日志。

幂等（同值覆盖），不影响正在跑的管线（**不修改正在执行的 bash 脚本本体** —— 改运行中的脚本会因
bash 按字节偏移增量读取而解析错乱）。

### 12.3 ✅ 澄清：种子 trial 的 U 没有污染搜索与排名

| 对象 | 参照系 | 是否有污染 |
|:--|:--|:--|
| **Optuna objective**（`result["utility"]` = `evaluate_model_quality(metrics, frozen_baseline)`） | `frozen_baseline`（10:12 起就是正确的 no-GMC 基线） | **无污染** ✅ |
| **一级门禁** `hard_gate(metrics, frozen_baseline, ...)` | 同上 | **无污染** ✅ |
| `result["utility_vs_rc1"]`（仅记录字段） | `rc1`（10:12~11:00 期间为空 ⇒ 回落成 **GMC 常量 RC1_EPOCH3**） | **trial_0000~0003 这几个字段偏** ⚠️ |
| **`SUMMARY.md`** | 汇总器用**最终参照系包**逐 trial `utility(m, base)` / `utility(m, rc1)` **重算** | **无污染** ✅ |

⇒ 搜索目标（Optuna 采样依据）、门禁判定、明早的最终排名**三者都不受影响**；
只有那 4 个种子 trial 的 `trial_result.json` 里 `utility_vs_rc1` 这一**历史字段**是旧参照系，
以 SUMMARY.md 的重算值为准。

### 12.4 门禁报告路径核对（无问题）

`optuna_mix_ratio_search.py:664-665`：`gate_epoch = args.gate_epoch if args.gate_epoch > 0 else args.epochs`
⇒ 调度器传的 `--gate-epochs last --gate-epoch 0` 解析为 `gate_epoch=3` ⇒ 落盘目录 `gate/epoch3/`
⇒ 与管线第③步等待的 `trials/trial_0000/gate/epoch3/gate_report.json` **完全一致** ✅

---

## 十三、收尾链路（管线第⑤⑥步）的端到端验证（2026-10-27 10:30）

管线最后两步要在 **19 小时后**才执行（05:20 派单截止 → ~06:05 末批收尾 → ~06:10 汇总）。
如果在那个时刻崩掉，明早就什么都没有 ⇒ **用真实的 GMC 搜索数据（104 个 trial）提前冒烟**。

### 13.1 ① 汇总器 `summarize_nogmc_search.py` —— 通过

```bash
# 服务器（真实 GMC 数据，输出写 /tmp 不污染 GMC 目录）
python manu/training/summarize_nogmc_search.py \
  --out-root runs/optuna_mix_ratio --ref /tmp/gmc_ref_for_smoke.json \
  --top 5 --out-md /tmp/smoke_summary.md
```

**交叉验证命中**：trial_0080 重算 U(vs 基线) = **+0.052554**、U(vs RC1) = **+0.024975**，
T1/T2/T3 = 0.901910 / 0.998949 / 0.897694，耗时 45.92 min
⇒ 与历史记录（+0.052553 / +0.024974）**逐位吻合** ⇒ "U 一律重算"这条路径可信。

### 13.2 ② 补采 `backfill_summary_rows.py` —— 通过（且还原行逐字节一致）

在**副本**上测（线上 GMC study.db / CSV 未被触碰，md5 与时间戳均未变）：

```
[INFO] 已结算 trial：103 个；CSV 缺行：1 个
  + trial_0103  status=SUCCESS  mix=w2.5  U=0.03285339244042796  判决=PASS
[OK] 已补写 1 行 → .../optuna_summary.csv
```

补写行与线上原始行 **逐字节一致**（含 `utility` / `utility_vs_rc1` / 全部超参列）✅
⇒ 调度器在截止时刻漏收的那批 trial，明早能**全部**进榜。

### 13.3 🛠 修掉一个"明早读盘误取"的坑：Top 榜混排了红线否决的 trial

实测（用 GMC 真实数据）：混排榜 Top-5 里 **第 2、3 名是 `FAIL_TRACK1_RETREAT`**，
排第 4 的 PASS 候选被压在后面。原因是**一级红线是绝对业务质量线、U 只罚相对退让**，
两者不是同一个量纲 ⇒ 红线否决的 trial U 反而常常更高。

**明早若直接取混排榜第 1 名，就会取到一个不可上线的模型。**

修法（纯增量，不动判定口径、不动原表结构）：

| 改动 | 内容 |
|:--|:--|
| `summarize_nogmc_search.py` §二 总体 | 新增一行「**过一级红线（可上线候选）：N**」 |
| §四 Top-N | 加一句警示（`verdict` 列可见，本榜含不可上线者，**选型看下一节**） |
| **§四之二（新增）** | 「Top-N **可上线候选**（仅过一级红线者）」独立榜 + 直接给出「本轮最优可上线候选：trial_XXXX」 |
| §五 口径声明 | 新增第 5 条：混排榜不可用于选型 |

改后实测（GMC 数据）：可上线候选池 69 个，榜内 trial_0086 / trial_0066 已被正确剔除，
trial_0080 浮到第 2 位。

> 顺带一条 GMC 侧的观察：**GMC 那轮的可上线最优是 `trial_0061`（w2.0 / 实测比 2.026 /
> lr_scale 2.6216 / PASS_RELAXED / U=+0.057861）**，不是 trial_0080。
> 这印证了 §10.7 的口径纪律——**交接线必须按数据重新定标，不能凭记忆套一个数**。

### 13.4 三侧 md5

| 文件 | md5 |
|:--|:--|
| `manu/training/summarize_nogmc_search.py` | `0bd5928f4186981b38ad775f81e8429b` |
| `manu/training/measure_reference_frame.py` | `2ecebaeb9e092ffb0cc4e29cdc6ed8d3` |
| `manu/training/nogmc_rc1_watchdog.sh` | `21b9b026f0c55f2ddf35bb893bd07bec` |
| `manu/memory/nogmc_rebuild_20261027.md` | 见下方同步记录 |

---

## 十四、🔴 又抓一个 bug：混样"实测比"上报字段被硬编码目录名打死（2026-10-27 10:45）

### 14.1 现象

no-GMC 侧 4 个 trial 的日志里，`mix LQ/UAV` **每一行都是 `0/32`**（各 51 个采样点，零例外）：

```
trial_0000: 51× "mix LQ/UAV 0/32"
trial_0001: 51× "mix LQ/UAV 0/32"    ← w2.0 本该约 1:2
trial_0002: 51× "mix LQ/UAV 0/32"    ← w3.0
trial_0003: 51× "mix LQ/UAV 0/32"    ← w4.0
```

GMC 侧 trial_0080 同位置是正常分布（9/23、11/21、8/24、12/20 …，均值 ≈1:2.5）。

### 14.2 根因（一行硬编码）

`manu/training/optuna_mix_ratio_search.py:591`（原）：

```python
n_lq_batch = sum(1 for f in im_files if "longquanshan_ft_gmc_median_official" in str(f))
```

**硬编码了 GMC 的 LQ 目录名**。no-GMC 跑的是 `longquanshan_ft_gmc_median_nogmc` ⇒ 永远匹配不上
⇒ `n_lq_batch ≡ 0` ⇒ `lq_seen ≡ 0` ⇒
`ratio = uav_seen / max(lq_seen, 1)` = 56926/1 = **56926.0**（垃圾值）
⇒ 写进 `trial_result.json` / `optuna_summary.csv` 的 `measured_ratio` 列，
以及 epoch 权重的 `finetune.measured_lq/measured_uav/measured_ratio` 元数据。

### 14.3 ⚠️ 影响面（**关键：不影响训练与判定**）

| 受影响 | 不受影响 |
|:--|:--|
| 日志 `mix LQ/UAV a/b` 列 | **实际采样**（`WeightedRandomSampler` 用**样本数** n_lq/n_uav 构造权重，**不看路径** ⇒ LQ 样本确实按 `P(LQ)=1/(1+r)` 进了 batch） |
| `measured_ratio` / `measured_lq` / `measured_uav`（trial_result.json、CSV、权重元数据） | **模型权重**（训练完全正常） |
| 调度器日志里的"实测 1:{ratio}" | **门禁三轨指标 T1/T2/T3** |
| | **一级红线判定**（`hard_gate` 只吃 metrics + frozen_baseline） |
| | **U 分数**（`utility` 只吃 metrics + baseline） |

⇒ **19 小时的训练产出、门禁判定、U 排名全部有效**；只有"上报字段"是垃圾值。

### 14.4 修复（两段）

**① 根修**（`optuna_mix_ratio_search.py`，md5 `5396673837cab15121725e2cb5044d63`）：
把硬编码换成**从 `--longquan-data` 推导** LQ 数据集根目录（绝对路径 + 目录名两个标记，
兼容 im_file 为相对路径的情况）：

```python
_lq_root = Path(args.longquan_data).resolve().parent
_LQ_MARKERS = (str(_lq_root), _lq_root.name)
def _is_lq_file(f) -> bool:
    s = str(f)
    return any(m in s for m in _LQ_MARKERS)
```

GMC 场景下目录名与旧硬编码**逐位等价** ⇒ 旧行为不变；no-GMC 场景自动正确。
**从下一个 trial 起生效**（调度器每次起新 worker 进程 ⇒ 会重读该文件）。

**② 事后复原**（当时已在跑的 trial_0000~0003 用的是旧进程）：
采样序列由 seeded generator 完全决定 ⇒ **可确定性重放**。
新增 `manu/training/replay_measured_ratio.py`（md5 `333baccfe8bd6a3a8899db60e6d2aba4`）：

```bash
# 校验（对 GMC 研究；硬编码当时是对的 ⇒ 是天然的真值集）
python manu/training/replay_measured_ratio.py --verify --trials-root runs/optuna_mix_ratio/trials
# → [VERIFY] 逐位一致 104 / 不一致 0（共 104），最大偏差 0 个样本   ✅
```

重放配方（GMC 侧 **104/104 逐位一致、零偏差**）：

1. `torch.manual_seed(trial_seed)`（worker 全局播种 ⇒ `torch.initial_seed()`）
2. `g.manual_seed((6148914691236517205 + RANK + trial_seed) % 2**64)`（`RANK=-1`）
3. `torch.empty((), dtype=torch.int64).random_(generator=g)` —— **base_seed 只抽一次**
   （`InfiniteDataLoader` **复用迭代器** ⇒ 第 2/3 个 epoch 不再抽；这一条是"抽 0 次/1 次/每 epoch"
   三种变体里**唯一**能对上的）
4. 每 epoch 一次 `torch.multinomial(weights, 56926, True, generator=g)`
   （torch 1.13.1 的 `WeightedRandomSampler.__iter__` 就是这么干的）
5. **只取最后一个 epoch 的计数**（`lq_seen` 每个 epoch 开头重置 ⇒ 落盘的就是末 epoch 值）
6. `natural` 模式没有 WeightedRandomSampler（普通 RandomSampler 全量无放回）⇒ 恰好 LQ=13918 / UAV=43008

### 14.5 收尾链补一段：修复守护

新增 `manu/training/nogmc_repair_after.sh`（md5 `4a77f101d1fcdf1cb228c748df65169c`，服务器常驻
PID 2097092）：等管线产出 `SUMMARY.md` ⇒ 重放复原实测比 ⇒ 刷新 SUMMARY.md ⇒ 校验无残留异常行。
**故意等 SUMMARY.md 落地再动手**，避免与管线的补采/汇总并发写同一个 CSV。

### 14.6 单 trial 耗时与吞吐（实测）

| 阶段 | 耗时 | 备注 |
|:--|--:|:--|
| 训练 3 ep（5337 步） | **≈ 28 min** | 0.31 s/iter；warmup 890 步（0.5 ep）；**cosine 周期 5 ep = 8895 步**（与 RC1 同口径） |
| 门禁三轨 | **≈ 18 min** | 其中龙泉山 val 多尺度热图缓存重建 **≈ 16 min**（每 trial 强制重建，无复用开关） |
| **单 trial 合计** | **≈ 46~47 min** | GMC 侧实测中位 46.4 min（最短 45.9 / 最长 51.6） |
| 吞吐 | 4 卡并行 ⇒ **每 ~47 min 出 4 个** | 19 h 预算 ⇒ 预计 **~95~100 个 trial**（上次 GMC 103） |

---

## 十五、🔴 ΔFP_T2 红线在 no-GMC 语义下失效 → 用户授权重标定（2026-10-27 11:12）

### 15.1 触发事件

前 4 个种子 trial（trial_0000~0003）门禁结算，**4/4 全被剪枝**：

| trial | mix_spec | T1 F1 (Δ) | T2 F1 (Δ) | **T2 FP (Δ)** | T3 F1 (Δ) | verdict |
|:--|:--|:--|:--|:--|:--|:--|
|0000| natural |0.899462 (−0.0014)|0.996552 (+0.0155)|**98 (+46)**|0.777577 (+0.0198)|FAIL_TRACK2_SAFETY|
|0001| w2.0 |0.898554 (−0.0023)|0.996403 (+0.0153)|**103 (+51)**|0.773346 (+0.0155)|FAIL_TRACK2_SAFETY|
|0002| w3.0 |0.898982 (−0.0019)|0.996552 (+0.0155)|**98 (+46)**|0.773282 (+0.0155)|FAIL_TRACK2_SAFETY|
|0003| w4.0 |0.900056 (−0.0008)|0.996612 (+0.0155)|**95 (+43)**|0.773489 (+0.0157)|FAIL_TRACK2_SAFETY|

**✅ 同时排除了 §11.4 的 T2 绝对线风险**：绝对线 0.985 在 no-GMC 下**完全可达**
（实测 0.996552，余量 +0.0116）⇒ 那条线不用动。

**❌ 但 ΔFP_T2 ≤ +10 这条线 100% 不通过** ⇒ `U=None` ⇒ Optuna 拿不到信号、TPE 退化为随机采样。

### 15.2 为什么说这条线"在 no-GMC 下失效"（不是模型变差，是尺子换了单位）

| | GMC 侧 | no-GMC 侧 |
|:--|:--|:--|
| 基线 T2 FP | 26 | **52**（去 GMC 后虚警本身翻倍） |
| 混训后 T2 FP | 过线 69 个 trial 落在 **[27, 41]** | 实测 **[95, 103]** |
| ⇒ ΔFP 正常带 | **[+1, +15]** | **[+43, +51]** |
| 这条线的原文设计意图 | 「T2 FP 正常水平 [32,37]，单 trial 噪声 ±5，**严格线 +10 正切在噪声带中部，红线在误杀好配置**」⇒ 它是「**别误杀正常混训水平**」的**相对护栏**，**不是**业务虚警上限 | 沿用 +10 ⇒ **100% trial 被剪枝** ⇒ 判据**彻底失去筛选功能** |

**根因（物理）**：虚警高度集中在**一条序列**上——`VIDEO00006_19700101_000703`：

| 序列 | no-GMC FP | GMC FP |
|:--|--:|--:|
| **VIDEO00006_000703** | **70** | 28 |
| VIDEO00008_001307 | 14 | 2 |
| 其余 5 条 | 3/3/3/0/5 | 1/5/1/0/0 |
| 合计 | **98** | **37** |

⇒ 去 GMC 后，「未做全局运动补偿的差分通道」把**移动背景/云层边缘**的能量注进了检测器
⇒ 该序列上的背景误报成倍放大。**这是语义失配的真实后果，不是配置问题。**

### 15.3 用户授权的重标定（已执行）

| 项 | 内容 |
|:--|:--|
| 授权 | 2026-10-27 11:12 问卷，用户明确选定 `recalib_60` |
| 改动 | `runs/nogmc_ref/reference_frame.json` → `gate.track2_fp_max_increment` **10 → 60**（⇒ 绝对 FP ≤ 52+60 = **112**，实测上沿 103） |
| **未改动** | `track2_f1_min` = 0.985（绝对业务线）、`track1_min_delta` = −0.005（T1 红线）、`frozen_baseline`、`rc1`、`gate_base` |
| 备份 | `runs/nogmc_ref/reference_frame.json.bak_20261027_111259` |
| 审计 | 文件内新增 `gate_calibration` 块（改动时间/授权人/理由/证据/未改动项） |
| 写入方式 | **原子替换**（tmp + `os.replace`）—— worker 随时可能并发读该文件，非原子写会让 worker 读到半截 JSON 而崩 |
| 生效机制 | 阈值经 `resolve_reference()` 换算成 `--max-track2-fp 112` 传给门禁脚本；**worker 每次启动都重读参照系包** ⇒ **trial_0008 起立即生效，无需重启调度器** |

### 15.4 离线重判工具 `manu/training/rejudge_with_reference.py`

重标定前已在跑的 trial（trial_0000~0007）是在旧线下判的 ⇒ 需要**权威重判**：
从每个 trial 落盘的 `gate/epoch3/gate_report.json` 取回**真实指标**（**不重新推理、指标一个不重算**），
用最终参照系包的 `frozen_baseline` / `rc1` / `gate` 重跑 `hard_gate` 与 `utility`
（与 worker 的调用**逐参数一致**），同步写回 `trial_result.json` 与 CSV 的
`status` / `gate_verdict` / `utility` / `utility_vs_rc1` 四列。

> ⚠️ 自查时发现并修掉一个**自己的 bug**：第一循环只改了内存对象，落盘循环又从磁盘重读、
> 找不到 `rejudged_at` 就 `continue` ⇒ **全部跳过、什么都没写**（但仍打印"已落盘"）。
> 修法：把重判结果存进 `judged` 字典，落盘循环直接用它，不再从磁盘重读。
> **教训：打印"已落盘"不能代替校验文件真的变了 —— 修完后逐行核对了 CSV 与 trial_result.json。**

重判结果（4 个 trial 全部翻转）：

```
trial_0000: FAIL_TRACK2_SAFETY → PASS | U None → +0.000916
trial_0001: FAIL_TRACK2_SAFETY → PASS | U None → −0.008158
trial_0002: FAIL_TRACK2_SAFETY → PASS | U None → −0.004577
trial_0003: FAIL_TRACK2_SAFETY → PASS | U None → −0.000155
```

### 15.5 ⚠️ 这次重判带出的**更硬的结论**：no-GMC 的微调净收益 ≈ 0

U 的分项（no-GMC 参照系）：

| trial | T3 增益 | T2 增益×0.5 | T1 退让×−2.5 | **FP 惩罚 ×−0.0005** | **U** |
|:--|--:|--:|--:|--:|--:|
|0000| +0.0198 | +0.0077 | −0.0036 | **−0.0230（46 FP）** | **+0.0009** |
|0001| +0.0155 | +0.0077 | −0.0059 | **−0.0255（51 FP）** | **−0.0082** |
|0002| +0.0155 | +0.0077 | −0.0048 | **−0.0230（46 FP）** | **−0.0046** |
|0003| +0.0157 | +0.0078 | −0.0021 | **−0.0215（43 FP）** | **−0.0002** |

⇒ **黄金池虚警的惩罚（−0.0215~−0.0255）把难例召回的增益（+0.0155~+0.0198）几乎完全抵消**
⇒ U ≈ 0 或为负 ⇒ **即使放宽了门禁，no-GMC 语义下微调的净效用也接近于零**。

叠加另一条独立证据：no-GMC 的 ΔF1_T3 只有 **+0.0155~+0.0198**，
而 GMC 侧成功 trial 的中位是 **+0.0445**（最好 +0.0741）⇒ **难例收益缩水约 2/3**。

⇒ **no-GMC 路径的问题不只是"门禁尺子要重标定"，而是"底模在 no-GMC 语义下本身就失配"**：
冻结底模 474 的 P0/highway 层是**为 GMC 差分通道设计并调优**的（P0 通道数 16、640×640）
⇒ 喂 no-GMC 特征天然失配，这解释了三轨全线掉点（−0.48 / −0.74 / −7.95 pp）。
**下一阶段若继续这条路线，应从「no-GMC 语义上重新适配的底模」出发，而不是直接拿 GMC 底模微调。**

### 15.6 收尾守护重启（bash 增量读取陷阱）

`nogmc_repair_after.sh` 加了「②用最终参照系包重判」一步后**必须重启守护**：
bash 是**按字节偏移增量读取脚本**的，改了运行中的脚本 ⇒ 后续命令会读错位、执行错乱。
⇒ `kill -TERM` 旧守护（PID 2097092）→ `setsid nohup` 起新版（PID **2103400**）。
最终守护链：**① 重放复原实测比 → ①之二 重判门禁与 U → ② 刷新 SUMMARY.md → ③ 校验**。

### 15.7 吞吐修正（实测）

单 trial 实测 **52.3~52.8 min**（比 GMC 侧的 46.4 min 慢，因 no-GMC 门禁的龙泉山缓存重建
与 4 个 trial 并发抢 I/O）⇒ 19 h 预算 ⇒ **预计 ~85~90 个 trial**（原估 95~100），
仍在"与上次 103 同量级"附近。截止不变：派单止于次日 05:20，末批收尾 ~06:15，汇总 ~06:20。

### 15.8 中期实测（8 个 trial，12:09）：新线仍具鉴别力，且暴露配置敏感性

| trial | mix | 实测比（重放复原） | T1 F1 | T2 F1 | **T2 FP** | T3 F1 | ΔF1_T3 | 新线(+60)判定 |
|:--|:--|--:|--:|--:|--:|--:|--:|:--|
|0004| w6.0 | 5.885 | 0.900207 | 0.996402 | 99 | 0.769148 | +0.0113 | PASS |
|0005| w5.0 | 4.985 | 0.900618 | 0.995746 | **125** | 0.751101 | **−0.0067** | **FAIL**（ΔFP +73） |
|0007| w5.0 | 4.978 | 0.900507 | 0.996402 | 97 | 0.761745 | +0.0039 | PASS |
|0000| natural | 3.090099 | 0.899462 | 0.996552 | 98 | 0.777577 | +0.0198 | PASS |
|0001| w2.0 | 1.987928 | 0.898554 | 0.996403 | 103 | 0.773346 | +0.0155 | PASS |
|0002| w3.0 | 2.988649 | 0.898982 | 0.996552 | 98 | 0.773282 | +0.0155 | PASS |
|0003| w4.0 | 3.993071 | 0.900056 | 0.996612 | 95 | 0.773489 | +0.0157 | PASS |
|0006| w3.0 | 3.027 | 0.900891 | 0.995807 | **125** | 0.751259 | **−0.0066** | **FAIL**（ΔFP +73） |

> **重放复原的交叉验证**：no-GMC 侧重放出的 4 组 (LQ, UAV) 与 **GMC 侧同 mix_spec、同 trial 号的真值逐位相同**
> （trial_0000 13918/43008、trial_0001 19052/37874、trial_0002 14272/42654、trial_0003 11401/45525）
> ⇒ 两侧数据集规模相同（13,918 + 43,008）且抽样种子相同 ⇒ **抽样序列逐位一致**，
> 重放公式得到独立互证 ✅（GMC 侧 104/104 全一致，no-GMC 侧 4/4 与 GMC 对齐）。

**两个关键判读**：

1. **重标定的 +60 选得准**：8 个样本的 ΔFP 分布 = [+43,+43,+46,+46,+46,+51,**+73,+73**]
   ⇒ +60 这条线**放过 6 个正常配置、抓住 2 个真正的虚警爆炸**（+73）⇒ **不是"放水"，仍有鉴别力**。
   （4 点定标时的实测上沿是 +51，扩到 8 点后尾部冒出 +73，恰好说明线要卡在两者之间。）
2. **配置敏感、且存在负增益配置**：trial_0005(w5.0) 与 trial_0006(w3.0) 的
   **ΔF1_T3 为负**（−0.0067 / −0.0066）⇒ 某些混样比**比冻结基线还差**（难例更差 + 虚警翻 2.4 倍）。
   ⇒ **no-GMC 语义下不存在"随便混一下就好"的安全配置**，必须靠搜索筛。

**汇总器在真实 no-GMC 数据上验证通过**（此前只在 GMC 数据上冒烟）：新章节「§四之二 可上线候选」
正确剔除 FAIL 项、「过一级红线（可上线候选）：4」计数正确、`实测比` 列对 trial_0004 已显示 5.885（新计数器生效）、
对 trial_0000~0003 显示 56926.0（待收尾的重放复原修掉）。

**吞吐实测修正**：单 trial 中位 **50.3 min**（47.8~52.8）⇒ 19 h 预算 ≈ **95 个 trial**。

### 15.9 汇总器新增「§五 与 GMC 侧的绝对指标对照」

**为什么必须加这一节**：U 是「相对各自参照系」的量，**跨语义不可比**；
但三轨的**绝对 F1** 是同帧集（31,613 / 17,187 / 11,810）、同 GT（25,111 / 16,638 / 10,535）、
同工作点（`system_sota_longquan_delivery_params.json`）测出来的 ⇒ **绝对值可比**，
这正是判断「no-GMC 能不能替代 GMC」真正需要的表。

实现要点（`summarize_nogmc_search.py`）：
- GMC 侧三行从**代码常量**读（`auto_epoch_gate_check.py::BASE` 冻结基线、
  `mix_ratio_judgement.py::RC1_EPOCH3`、`runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/gate_report.json`）
  ⇒ **不手抄数字**，来源可审计；
- no-GMC 两行从参照系包与「§四之二 可上线候选榜首」取；
- 口径声明顺延为 §六。

当前（8 trial 中期）的对照表：

| 模型 | T1 源域 F1 | T2 黄金池 F1 | T2 FP | T3 盲测 F1 |
|:--|--:|--:|--:|--:|
| GMC 冻结基线 474（去 P0） | 0.905736 | 0.988436 | 26 | 0.837331 |
| GMC RC1（路线B Ep3） | 0.901918 | 0.998979 | 31 | 0.871684 |
| GMC RC2（trial_0080） | 0.901910 | 0.998949 | 33 | **0.897694** |
| no-GMC 冻结基线 474（去 P0） | 0.900897 | 0.981074 | 52 | 0.757812 |
| **no-GMC 本轮最优可上线（trial_0000）** | 0.899462 | 0.996552 | **98** | 0.777577 |

⇒ **no-GMC 的最优可上线候选比 GMC RC2 在难例上低 12.0 pp（0.7776 vs 0.8977），
黄金池虚警是 3 倍（98 vs 33）** —— 这是明早做「no-GMC 能否替代 GMC」决策的**唯一横向可比口径**。

> ⚠️ 自查时发现并修掉一个**列顺序 bug**：元组是 `(name, T1, T2, T3, FP)`，解包却写成
> `(name, T1, T2, FP, T3)` ⇒ **T2 FP 与 T3 F1 两列互换**（输出里表现为 "T2 FP=0.837331、T3=26.000000"）。
> 教训同 §15.4：**生成物必须真的看一眼**，只看代码会漏掉这种错。

### 15.10 ✅ 新红线在运行系统中确认生效（12:20:42 实测）

`trial_0008`（重标定之后派单的第一个 trial）的**门禁命令回显**（在 worker stdout 日志
`runs/optuna_nogmc/logs/trial_0008.log` 里）：

```
--base-track1-f1 0.9008967482400269     ← no-GMC 基线 T1
--min-track1-f1 0.8958967482400269     ← = 基线 − 0.005（T1 红线，随参照系平移）
--min-track2-f1 0.985                   ← 绝对业务线，未动
--base-track2-fp 52                     ← no-GMC 基线 FP
--max-track2-fp 112                     ← ★ 新线（52 + 60）；旧线对照 trial_0004 = 62
```

⇒ 三个阈值**全部**来自更新后的参照系包，**worker 每次启动重读**的机制验证通过，
**不需要重启调度器**。

> ⚠️ 查这条命令回显时踩了个坑：一开始去门禁目录（`gate/epoch3/*.log`）里找，实际
> **门禁命令的回显在 worker 的 stdout 日志**（`runs/optuna_nogmc/logs/trial_XXXX.log`）里，
> 门禁目录只有各子步骤的日志（track1_infer / track2_golden / track3_blind / ms_cache）。

### 15.11 ✅ 收尾守护链**端到端验证通过**（12:23，在副本上跑）

修复守护 `nogmc_repair_after.sh` 的四步链此前**从未完整跑过**（它要等明早 06:10 才触发）⇒
在**当前状态的副本**（`/tmp/e2e3.*`：study.db + optuna_summary.csv + 12 个 trial 的
`trial_result.json` + 8 个 `gate_report.json`）上**完整跑了一遍**：

```
[12:23:25] ---- ① 确定性重放复原实测比 ----
[12:23:27] ---- ① 之二 用最终参照系包重判所有 trial 的门禁与 U ----
[REJUDGE] trial_result.json 已落盘 8 个
[REJUDGE] CSV 已同步 8 行（备份 .../optuna_summary.csv.bak_rejudge）
[REJUDGE] 共重判 8 个 trial，判定发生变化 4 个
[12:23:27] ---- ② 刷新汇总 ----
[SUMMARY] 已写入 /tmp/e2e3.bxBF/SUMMARY.md
[12:23:27] ---- ③ 校验 ----
[CHECK] /tmp/e2e3.bxBF/optuna_summary.csv 实测比仍异常的行：0 ✅
```

最终汇总核对无误：

- 按 status：SUCCESS 6 / PRUNED 2（**trial_0005、trial_0006 的 ΔFP +73 被新线正确拦住** ✅）
- 过一级红线（可上线候选）：**6**
- 实测比全部复原（3.090 / 1.988 / 2.989 / 3.993 / 5.885 / 4.983 …，不再是 56926.000）✅
- §四之二 可上线候选榜正确剔除了红线否决项 ✅

**期间又抓到两个问题**（都已修）：

1. **③ 校验的路径写死**（`runs/optuna_nogmc/optuna_summary.csv`），在副本测试时读的是**线上**文件 ⇒
   报出"4 行仍异常"的假警报。修法：改用 `OUT_ROOT` 环境变量，并在输出里**打印被检查的文件路径**。
2. ⚠️ **权限网关拦截会连累整条命令**：修①的第一次尝试因为命令里带 `rm -f` 被网关拦下，
   **整条命令（含 `cp` 同步）都没执行** ⇒ 服务器上仍是旧版脚本，而我一度以为已同步。
   ⇒ **教训：命令被网关拦下后，必须单独确认「同步这一步到底跑没跑」（md5 比对），不能凭意图继续。**
   （第 15.4 节「已落盘」的假成功、第 15.9 节列顺序 bug，是同一类问题：**不能凭代码/意图判断结果，
   必须看实际产物**。）

### 15.12 🔧 补完最后两个「不认参照系 / 丢信息」的工具缺口

全链路的工具盘点：`auto_epoch_gate_check` / `optuna_mix_ratio_search`（worker）/ `summarize_nogmc_search`
/ `rejudge_with_reference` / `replay_measured_ratio` 都已参照系感知；最后两个还没跟上：

**① `backfill_summary_rows.py`：补写行的 `duration_min` 写死 0.0**

管线第⑤步（补采 CSV）在**真实 no-GMC 数据**上实测（副本，删掉 trial_0002 那行再补）：

```
[INFO] 已结算 trial：8 个；CSV 缺行：1 个
  + trial_0002  status=SUCCESS  mix=w3.0  U=-0.0045768767646343425  判决=PASS
[OK] 已补写 1 行
```

补是补上了，但还原行的 `duration_min` 是 **0.00**（真实值 52.78）⇒ 会把汇总里
「单 trial 耗时：中位/最短/最长」拉成 0.00。

**根因**：脚本第 136 行 `dur = 0.0`（当时 `trial_result.json` 里没读时长字段）。
**修法**：`trial_result.json` 的 `elapsed_s` 就是 worker 的**全程墙钟**（训练 + 门禁，
实测 3129.4 s = 52.16 min，与调度器自己算的 52.33 min 只差计时起点）⇒
`dur = float(res.get("elapsed_s") or 0.0) / 60.0`。修后还原行 `52.62` vs 线上原行 `52.78`，
**其余字段逐字节一致** ✅（只动一个上报列，不碰任何指标与判定）。

**② `status_mix_ratio_search.py`：实时监控脚本用的是 GMC 侧常量**

它是管线启动脚本里明确让用户用的监控命令，但它硬编码 `FROZEN_BASELINE` / `RC1_EPOCH3`
⇒ 在 no-GMC 语境下会显示**GMC 的基线与擂主**、按 GMC 基线算 U、并给所有 trial 标「⛔触红线」。

**修法（纯增量，不传参 ⇒ 逐位不变）**：新增 `--reference-frame`，注入
`frozen_baseline` / `rc1` / `gate` 阈值；并让**口径组的擂主**在注入时换成参照系包里的
`rc1`（no-GMC 语义下的 natural 对照），否则会拿 GMC 的 RC1_EPOCH3 跟 no-GMC 候选对局（跨语义）。

| 模式 | 冻结基线行 | 擂主行 | 及格线 |
|:--|:--|:--|:--|
| GMC（不传参，**逐位不变**） | 0.905736 / 0.988436 / 0.837331 | 【RC1=Ep3】0.901918 / 0.998979 / 0.871684 | +0.027579 |
| no-GMC（传参照系） | 0.900897 / 0.981074 / 0.757812 | 【现任擂主】0.899462 / 0.996552 / 0.777577 | +0.000916 |

两模式都实测过；no-GMC 侧的 U 与 `rejudge_with_reference` 的重算值**逐位一致**，
trial_0005/0006（ΔFP +73）被正确标 ⛔。

**③ 启动脚本的监控提示**：`launch_nogmc_mix_ratio_search.sh` 原来只印 `--out-root`，
现在补上 `--study-name` 与 `--reference-frame`（并修正了一个笔误：变量名是
`REFERENCE_FRAME` 不是 `REF_FRAME`，否则会打印空路径）。

⇒ **至此，整条 no-GMC 工具链（门禁 / worker / 判定 / 汇总 / 重判 / 重放 / 补采 / 监控）
全部参照系感知，且「不传新参数 ⇒ 行为逐位不变」这条铁律对 GMC 侧依然成立。**

---

## 十五.13、🔧 监控链最后一处跨语义漏洞：决赛段用 GMC 常量判门禁 → KeyError 崩溃（12:40）

### 15.13.1 现象（服务器实跑，不是推演）

```bash
python manu/training/status_mix_ratio_search.py --out-root runs/optuna_nogmc \
  --study-name mix_ratio_search_nogmc --reference-frame runs/nogmc_ref/reference_frame.json
```

跑到 `[6] 决赛` 段直接抛异常：

```
File "manu/training/status_mix_ratio_search.py", line 495, in main
  f"帕累托极大元 {rep['n_pareto_maximal']} 个")
KeyError: 'n_pareto_maximal'
```

⇒ 监控脚本在**决赛结论**这一步就断了（前面 [1]~[5] 正常）。
更严重的是：把 KeyError 修掉之后，结论本身是**错的**（详见 15.13.2 第③层）——
**一个不崩但结论错的监控，比崩了更危险**。

### 15.13.2 根因（三层，逐层剥开）

**第①层：`select_best()` 内部用模块常量判门禁。**
`select_best` 对每个候选调 `hard_gate(metrics, frozen, allow_relaxed=...)` 时**没有传阈值**
⇒ 用 `DEFAULT_GATE_TRACK2_FP_MAX_INCREMENT = +10`。
no-GMC 候选的 ΔFP_T2 正常带是 [+43,+51] ⇒ **9 个候选全被剔除** ⇒ `passed` 为空。

**第②层：早退路径缺 `n_pareto_maximal` 键。**
`select_best` 在 `if not passed:` 处提前 return，那个 report dict 里没有这个键
⇒ 调用方按固定键取值 ⇒ KeyError。

**第③层：`compare_models()` 里也用默认阈值。**
修好 ①②后，"vs 擂主" 判成 `GATE_REJECT` —— 因为 `compare_models` 内部
`hard_gate(a, frozen, ...)` 同样用默认 +10，而且**参照系注入的基线**没有真正进入它的门禁：
no-GMC 擂主相对 **GMC 冻结基线** 的 ΔF1_T1 = 0.899462 − 0.905736 = **−0.006274 < −0.005**
⇒ 触了 **GMC 的** T1 红线 ⇒ 跨语义误判。

### 15.13.3 修复（全部纯增量：新增参数默认 `None` ⇒ 不传新参数逐位等价旧行为）

| 文件 | 改动 |
|:--|:--|
| `mix_ratio_judgement.py` | `select_best(..., gate_config=None)`、`compare_models(..., gate_config=None)`：把 `track2_f1_min` / `track2_fp_max_increment` / `track1_min_delta` 透传给 `hard_gate`；`select_best` 把自己的 `frozen_baseline` 透传给 `compare_models`；早退路径补 `"n_pareto_maximal": 0` |
| `status_mix_ratio_search.py` | [6] 段两处 `select_best` 调用补 `frozen_baseline=base, gate_config=gate_cfg`；`rep['n_pareto_maximal']` → `rep.get('n_pareto_maximal', 0)`；**GMC 侧文案逐字保留**（`✅ 打败 RC1/Ep3`），no-GMC 侧另用「现任擂主」措辞；及格线行 GMC 侧不带正号 |
| `summarize_nogmc_search.py` | 口径声明不再写死「ΔFP_T2 ≤ +10」，改为读参照系包的实际阈值；并自动追加一条 3.1 重标定说明（含 8 点 ΔFP 分布与"仍有鉴别力"的依据） |

### 15.13.4 证据（全部实测，不是推理）

- **自测**：`test_mix_ratio_judgement` 71/71、`test_status_mix_ratio_search` 54/54、`test_optuna_dispatcher` 26/26。
- **GMC 侧字节级对照**（关键）：把 git HEAD 版本拷到独立目录，与新版在**同一份 GMC 产物**上跑
  `status_mix_ratio_search.py --out-root runs/optuna_mix_ratio` ⇒ **111 行逐字节一致**。
  期间只发现一处差异（及格线行的 `+0.027579` vs `0.027579`），已按分支修回。
- **服务器实跑 no-GMC 监控**（修完）：

```
▸ 分组【RC1-Ep3口径】 擂主=参照系（U=+0.000916）
  候选 9 个，过一级红线 7 个，帕累托极大元 5 个
  被红线剔除：[('trial_0006', 'FAIL_TRACK2_SAFETY'), ('trial_0005', 'FAIL_TRACK2_SAFETY')]
  🏆 最优：trial_0000  U=+0.000917
     vs 擂主：UTILITY_A   U=+0.000917 vs +0.000916  ⇒ ✅ 打败本组擂主
```

  ⇒ trial_0005/0006（ΔFP_T2 = +73）被正确剔除，其余 7 个过线。
- **三侧 md5 一致**：
  - `manu/training/mix_ratio_judgement.py` `9aade8dd3e2feb021138c26025af60ec`
  - `manu/training/status_mix_ratio_search.py` `73b05613ca2d847943afaebf44dbac4f`
  - `manu/training/summarize_nogmc_search.py` `59fdbaa0befa23a365d4f3af6bda8885`

### 15.13.5 ⚠️ 纪律教训：**「自测全绿」不等于「逐位不变」**

我第一版的参照系注入把 GMC 侧的结论**文案**也一起改了
（`✅ 打败 RC1/Ep3` → `✅ 打败现任擂主`），自测立刻挂 1 项。
定位方式是：**把 git HEAD 版本拷到独立目录跑同一套测试 + 失败项逐行 diff**
⇒ 确认是**我自己今天引入的**，不是既有失败（HEAD 版 54/54 全绿）。

⇒ 教训固化成两条：
1. 参照系注入类改动**必须做字节级输出 diff**（`diff <(HEAD 版输出) <(新版输出)`），
   只看自测通过会漏掉文案类回归；
2. 「既有失败 vs 新引入」必须用**两个版本对跑**来判定，不能凭印象。

---

## 十五.14、🔬 配对 A/B：**同一条配方**下 GMC 的受控价值（12 个严格配对 trial，12:50）

### 15.14.1 怎么发现的

回查 GMC 侧 104 个 trial 时发现 **GMC `trial_0011` 与 no-GMC `trial_0011` 的参数逐字相同**
（`mix=w5.0`、`lr_scale=2.3391`、实测比 1:4.949）⇒ 两个 study 的**前若干 trial 走的是同一套
种子采样序列**（同 trial 号 = 同超参 + 同种子 + 同样本序列）。

逐维度核对（`mix_spec` + `lr_scale` + `warmup_epochs` + `weight_decay` + `focal_beta` +
`offset_weight` + `mosaic` + `scale` + `translate` + `clip_max_norm` + `freeze_bn`）：

- **trial_0000 ~ trial_0011：12/12 全部逐字相同，0 个不一致** ⇒ **天然配对 A/B**；
- 从 trial_0012 起分叉：GMC 侧走 `lr=1.0000 / warm=0.500` 的固定档（natural / w4.0 / natural / w4.0），
  no-GMC 侧直接进 TPE 采样（w8.0/2.152、w5.0/1.167、w2.0/2.365、w2.0/1.422）
  ⇒ **配对集合上限 = 12 个**，之后两侧不再同源。

### 15.14.2 配对差（no-GMC − GMC，同超参同种子）

| trial | mix | lr_scale | T1 F1 GMC / no-GMC / Δ | T2 F1 GMC / no-GMC / Δ | T3 F1 GMC / no-GMC / Δ | ΔFP_T2 | ΔFP_T3 |
|:--|:--|--:|:--|:--|:--|--:|--:|
| trial_0000 | natural | 1.0000 | 0.902455 / 0.899462 / **−0.299 pp** | 0.998769 / 0.996552 / **−0.222 pp** | 0.864086 / 0.777577 / **−8.651 pp** | +61 | +460 |
| trial_0001 | w2.0 | 1.0000 | 0.900319 / 0.898554 / −0.177 pp | 0.998739 / 0.996403 / −0.234 pp | 0.865463 / 0.773346 / −9.212 pp | +66 | +607 |
| trial_0002 | w3.0 | 1.0000 | 0.901388 / 0.898982 / −0.241 pp | 0.998919 / 0.996552 / −0.237 pp | 0.861058 / 0.773282 / −8.778 pp | +66 | +498 |
| trial_0003 | w4.0 | 1.0000 | 0.902170 / 0.900056 / −0.211 pp | 0.998799 / 0.996612 / −0.219 pp | 0.862080 / 0.773489 / −8.859 pp | +58 | +502 |
| trial_0004 | w6.0 | 1.0000 | 0.903165 / 0.900207 / −0.296 pp | 0.998919 / 0.996402 / −0.252 pp | 0.855909 / 0.769148 / −8.676 pp | +66 | +354 |
| trial_0005 | w5.0 | 1.0139 | 0.903939 / 0.900618 / −0.332 pp | 0.998320 / 0.995746 / −0.257 pp | 0.855073 / 0.751101 / −10.397 pp | +72 | +764 |
| trial_0006 | w3.0 | 1.1106 | 0.904610 / 0.900891 / −0.372 pp | 0.998380 / 0.995807 / −0.257 pp | 0.861535 / 0.751259 / −11.028 pp | +74 | +1009 |
| trial_0007 | w5.0 | 0.4349 | 0.902735 / 0.900507 / −0.223 pp | 0.998709 / 0.996402 / −0.231 pp | 0.850582 / 0.761745 / −8.884 pp | +59 | +290 |
| trial_0008 | w3.5 | 0.3957 | 0.902974 / 0.900192 / −0.278 pp | 0.998230 / 0.995028 / −0.320 pp | 0.850293 / 0.743415 / −10.688 pp | +82 | +754 |
| trial_0009 | w1.0 | 0.3033 | 0.897551 / 0.894762 / −0.279 pp | 0.998529 / 0.996223 / −0.231 pp | 0.848932 / 0.764887 / −8.405 pp | +63 | +364 |
| trial_0010 | w5.0 | 0.9516 | 0.902256 / 0.900410 / −0.185 pp | 0.998050 / 0.994760 / −0.329 pp | 0.861437 / 0.743076 / −11.836 pp | +86 | +1043 |
| **trial_0011** | **w5.0** | **2.3391** | 0.897715 / 0.896825 / **−0.089 pp** | 0.998949 / 0.997421 / **−0.153 pp** | 0.888055 / 0.790082 / **−9.797 pp** | **+44** | +452 |

**汇总（12 个配对，中位数）**：

| 指标 | Δ（no-GMC − GMC） | 区间 |
|:--|--:|--:|
| T1 F1（源域大盘） | **−0.259 pp** | −0.372 ~ −0.089 pp |
| T2 F1（黄金池） | **−0.235 pp** | −0.329 ~ −0.153 pp |
| **T3 F1（难例）** | **−9.048 pp** | −11.836 ~ −8.405 pp |
| T2 虚警 | **+66** | +44 ~ +86 |
| T3 虚警 | **+500** | +290 ~ +1043 |

**召回/精度分解**（由落盘 F1 + FP 与 GT 反解 TP：`TP = F1×(FP+GT)/(2−F1)`；
GT 取 `auto_epoch_gate_check.BASE[*]["gt"]` = 25,111 / 16,638 / 10,535；
F1 为落盘 6 位小数 ⇒ 反解值 ~1e-4 舍入误差，**只用于归因**）：

| 轨 | ΔRecall（no-GMC − GMC） | ΔPrecision（no-GMC − GMC） | 读法 |
|:--|--:|--:|:--|
| T1 源域 | **+0.078 pp** | −0.539 pp | 召回持平、精度略降 ⇒ **基本无感** |
| T2 黄金池 | **−0.084 pp** | −0.394 pp | 召回持平、精度略降 ⇒ **基本无感** |
| **T3 难例** | **−10.783 pp** | **−6.428 pp** | **召回与精度双降** ⇒ 又是漏报又是乱报 |

### 15.14.3 结论（比 §十一 的基线对照更硬）

1. **GMC 的价值高度集中在 T3 难例**：同一配方下，去掉 GMC ⇒ T3 掉 **9.05 pp**，
   而 T1/T2 只掉 0.2~0.3 pp。§十一 的基线对照（−7.95 pp）与本节（−9.05 pp）**互相印证**。
2. **方向修正**：基线侧是「no-GMC 虚警反而少 54、掉的是召回」（§十一.3）；
   **训练后**变成「no-GMC 虚警多 +500、召回也低 10.78 pp、精度也低 6.43 pp」
   ⇒ **混训在 no-GMC 特征上把难例推到了"乱报"区**，不只是"报得少"。
   这解释了 §十五.5 的「净效用 ≈ 0」：**难例增益被 GMC 那一层的差距吃光**。
3. **T1/T2 上 GMC 几乎无感**（ΔRecall +0.08 / −0.08 pp、ΔPrecision −0.54 / −0.39 pp）
   ⇒ 「去掉 GMC」这件事的代价**只在难例上兑现**，源域与黄金池基本不受影响
   ⇒ 若业务只考核 T2 绝对线 0.985，no-GMC **可行**；一旦上难例（T3），no-GMC **不够用**。
3. **决策含义（与 §十五.5 同向、更强）**：no-GMC 线要继续，必须**从 no-GMC 适配过的底模出发**；
   用 GMC 调优过的 474（去 P0）当底模去训 no-GMC，任何配方都得先补 ~9 pp 的难例召回债。
4. ⚠️ **只做配对差，不做跨语义 U 横比**（GMC 的 U 与 no-GMC 的 U 量纲不同）。

### 15.14.4 新增工具与产物

- `manu/training/compare_gmc_nogmc_matched.py`（新增，md5 `fc4193db4c155aa861398cd46ea62d1a`）：
  配对判定 + 配对差表 + 汇总；`collect_pairs()` 供汇总器复用；`--out-json` 可落 JSON。
  ```bash
  cd /tmp/pycharm_project_10ae9e2e
  python manu/training/compare_gmc_nogmc_matched.py \
    --gmc-csv runs/optuna_mix_ratio/optuna_summary.csv \
    --nogmc-csv runs/optuna_nogmc/optuna_summary.csv --max-trials 200
  ```
- 汇总器 `summarize_nogmc_search.py` 新增 **§五之二「与 GMC 侧的配对 A/B」**（md5 `27165f3be3a70d10c3295d8250e653c4`）
  ⇒ **明早读 `SUMMARY.md` 就能直接看到这张配对表**；配对数在收尾时自动重算
  （两侧采样器若后续再同源，配对集会自动变长）。
- 产物：`runs/optuna_nogmc/matched_gmc_vs_nogmc.json`（服务器）。

### 15.14.5 踩坑：内联 ssh 的 heredoc 会落到 Python 2.7

`ssh '... python - <<EOF ... EOF'` 若**不经脚本文件**、也**不先激活 conda**，
`python` 是系统 **Python 2.7** ⇒ `open(..., encoding=)` 报
`TypeError: 'encoding' is an invalid keyword argument`、中文报
`SyntaxError: Non-ASCII character ... no encoding declared`。

⇒ 两条纪律：
1. **内联 ssh 跑 python 脚本必须先** `source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav`
   （实测激活后为 Python 3.10.20）；且 heredoc 内容尽量纯 ASCII。
2. **仓库里的 .sh 脚本本身已在头部激活 conda** ⇒ 脚本内 heredoc 不受影响
   （实测 `nogmc_repair_after.sh` 第③步在真实条件下正常输出 `[CHECK] … 异常的行：N`）——
   ⚠️ 别把「内联 ssh 的坑」误判成「守护脚本的 bug」。

---

## 十五.15、✅ 配置逐项一致性审计（直接对应用户原始要求：「配置完全一样，只换数据集」）

GMC 版启动脚本 `launch_mix_ratio_search.sh` **只传** output-root / study-name / gpus /
n-trials / epochs / max-hours，训练相关项全部走 dispatcher 的 argparse 默认值；
no-GMC 版 `launch_nogmc_mix_ratio_search.sh` 显式传参。⇒ 逐项对照如下
（两边**同一份** `optuna_mix_ratio_search.py`）：

| 参数 | GMC 实际生效 | no-GMC 实际生效 | 差异 |
|:--|:--|:--|:--|
| `--weights`（底模） | `runs/optuna_p0_nas/trial_0474/weights/best.pt` | **同左（逐字相同）** | 无 |
| `--uav-data`（源域训练集） | `.../uav_gmc_median/data.yaml` | `.../uav_gmc_median_nogmc/data.yaml` | **换数据集** |
| `--longquan-data`（业务域训练集） | `.../longquanshan_ft_gmc_median_official/data.yaml` | `.../longquanshan_ft_gmc_median_nogmc/data.yaml` | **换数据集** |
| `--baseline-cache` | `runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl` | `runs/nogmc_ref/uav_cache_base.pkl` | **换缓存**（必须同特征语义） |
| `--reference-frame` | 不传（用模块常量） | `runs/nogmc_ref/reference_frame.json` | **新增**（参照系注入） |
| `--epochs` | 3 | 3 | 无 |
| `--batch` | 32 | 32 | 无 |
| `--imgsz` | 640 | 640 | 无 |
| `--workers` | 8 | 8 | 无 |
| `--gpus` | 0,1,2,3 | 0,1,2,3 | 无 |
| `--n-trials` | 4096 | 4096 | 无 |
| `--gate-epochs` / `--sched-epochs` | last / 5 | last / 5 | 无 |
| `--gate-script` | `manu/training/auto_epoch_gate_check.py` | 同左 | 无 |
| `--output-root` / `--study-name` | `runs/optuna_mix_ratio` / `mix_ratio_search` | `runs/optuna_nogmc` / `mix_ratio_search_nogmc` | **独立 study**（不污染 GMC 研究库） |
| `--max-hours` | 20 | 19 | **纯时间预算**（no-GMC 单 trial ~48 min > GMC ~46 min，且要给 07:00 硬停留余量），与训练配置无关 |

**搜索空间 / TPE / 种子 trial 列表 / 断点续跑 / 门禁脚本 / 截止机制** ⇒ 同一份代码，**完全一致**。

⇒ **结论：除「训练集/测试集 + 基线缓存 + 参照系 + 输出根/study 名 + max-hours（时间预算）」
外，逐项一致** —— 满足用户「配置完全一样，只替换训练集和测试集」的要求，
且额外三项替换都是**语义必需**（不换就会拿 GMC 基线判 no-GMC 模型）。

---

## 十五.16、中期结论（16 trial，13:28）：红线归属清楚了 + 一个**待用户裁决**的两难

### 15.16.1 16 个 trial 的红线归属（实测，非推断）

| 归属 | trial | ΔF1_T1 | ΔFP_T2 | ΔF1_T3 |
|:--|:--|--:|--:|--:|
| **只被 T1 否** | trial_0014（w2.0, lr 2.365） | **−0.675 pp** | +46 | **+4.999 pp** |
| | trial_0009（w1.0, lr 0.303） | −0.613 pp | +53 | +0.707 pp |
| **只被 T2 否** | trial_0013（w5.0, lr 1.167） | −0.165 pp | **+63** | +1.233 pp |
| | trial_0005（w5.0, lr 1.014） | −0.028 pp | +73 | −0.671 pp |
| | trial_0006（w3.0, lr 1.111） | −0.001 pp | +73 | −0.655 pp |
| | trial_0008（w3.5, lr 0.396） | −0.070 pp | +85 | −1.440 pp |
| | trial_0010（w5.0, lr 0.952） | −0.049 pp | +96 | −1.474 pp |
| **全过（9 个）** | 0000/0001/0002/0003/0004/0007/0011/0012/0015 | −0.407 ~ −0.039 pp | +25 ~ +51 | −0.393 ~ +3.227 pp |

⇒ **T1 触线 2 个、T2 触线 5 个、全过 9 个**。

### 15.16.2 ⚠️ 与 ΔFP_T2 的性质**不同**：T1 线**不需要**重标定

- T2 过线的 **11** 个 trial 里，源域代价 ΔF1_T1 = **中位 −0.228 pp**、区间 [−0.675, −0.039] pp，
  低于 −0.5 pp 红线的只有 **2** 个 ⇒ **红线仍然有鉴别力**。
- 对照：ΔFP_T2 沿用 +10 时是 **4/4 全部剪枝、判据完全失效**（必须重标定）；
  T1 −0.5 pp 只否掉 **2/16** ⇒ **不构成"判据失效"**。
  ⇒ **不擅自改动 T1 红线**（用户 11:12 授权时明确写明「绝对业务线 0.985 与 T1 红线 −0.005 未动」）。

### 15.16.3 🔴 真正的两难：`trial_0014` 要不要放行（**业务取舍，需用户裁决**）

`trial_0014` 与 `trial_0015` 是**同配比（w2.0、实测比 1:1.96）、仅 lr_scale 不同**的一对干净对照：

| trial | lr_scale | ΔF1_T1 | ΔFP_T2 | ΔF1_T3 | U(vs 基线) | 判定 |
|:--|--:|--:|--:|--:|--:|:--|
| trial_0014 | **2.3648** | **−0.675 pp** | +46 | **+4.999 pp** | **+0.017973** | ⛔ 触 T1（比红线深 **0.175 pp**） |
| trial_0015 | 1.4221 | −0.266 pp | +37 | +2.967 pp | +0.012459 | ✅ 过线 |

⇒ **`lr_scale` 是「难例增益 ↔ 源域退让」的旋钮**：
高 lr 换难例（T3 +5.00 pp），代价是源域退让 0.68 pp。
no-GMC 语义下最优落点在 **lr ≈ 1.4~2.4**，且 **T1 是约束**（不是 T2）。

⇒ **待用户裁决的问题**：「源域退让 0.68 pp 换难例 +5.0 pp」是否可接受？
- 若可接受 ⇒ `trial_0014` 是本轮**最好的**候选（T3 0.807805，U +0.017973）；
- 若不可接受 ⇒ 本轮最优可上线为 **`trial_0011`**（w5.0 / lr 2.339，T3 +3.227 pp，U +0.017764）。
  ⚠️ 注意两者 T1 都接近红线（−0.407 / −0.675 pp），且**都是单种子** ⇒ 需多种子复现。

### 15.16.4 与 GMC 侧的对照（**⚠️ 本小节的结论已被 §十五.17 更正**）

| | T1 代价 | T3 增益 |
|:--|:--|:--|
| GMC RC1（路线B Ep3） | −0.382 pp | +0.034 pp（0.837331→0.871684） |
| GMC RC2（trial_0080） | −0.383 pp | +0.060 pp（0.837331→0.897694） |
| no-GMC trial_0014 | −0.675 pp | +5.00 pp（0.757812→0.807805） |
| no-GMC trial_0011 | −0.407 pp | +3.23 pp（0.757812→0.790082） |

> 🔴 **更正（13:40，§十五.17 推翻本小节）**：这里拿 **2 个 no-GMC 单 trial** 去比 GMC 的 **2 个特定 trial**，
> 就得出「no-GMC 源域代价整体更大」⇒ **不成立**。
> 按 lr 分档看全分布（§十五.17），GMC 在高 lr 档（n=49）的源域代价中位是 **−0.745 pp**，
> 比 no-GMC 现有 3 个（−0.407 pp）**更重**。正确说法见 §十五.17。
> ⇒ **教训：单点对比（尤其 n=2 对 n=2）不能支撑"整体更大/更小"这种分布级断言。**

---

## 十五.17、🔬 lr_scale 分档：两侧同轴的「源域代价 ↔ 难例增益」曲线（13:40）

### 15.17.1 分档结果（Δ 相对**各自**的冻结基线 ⇒ 两侧可同轴比）

| lr_scale 档 | GMC：n / ΔF1_T1 / ΔF1_T3 | no-GMC：n / ΔF1_T1 / ΔF1_T3 | ΔT3 之比 |
|:--|:--|:--|--:|
| <0.7 | 6 / −0.358 pp / **+1.436 pp** | 3 / −0.070 pp / +0.393 pp | 27% |
| 0.7~1.2 | 22 / −0.373 pp / **+2.968 pp** | 9 / −0.084 pp / +1.233 pp | 42% |
| 1.2~1.8 | 27 / −0.530 pp / **+4.276 pp** | 1 / −0.266 pp / +2.967 pp | 69% |
| 1.8~2.4 | 25 / −0.684 pp / **+5.726 pp** | 3 / −0.407 pp / +3.227 pp | 56% |
| ≥2.4 | 24 / −0.776 pp / **+6.203 pp** | **0（未采样）** | — |

高 lr 档（≥1.8）汇总：**GMC n=49**：ΔT1 中位 **−0.745 pp**（−1.103~−0.383）、ΔT3 中位 **+5.955 pp**（+3.271~+7.647）；
**no-GMC n=3**：ΔT1 中位 **−0.407 pp**（−0.675~−0.228）、ΔT3 中位 **+3.227 pp**（+1.307~+4.999）。

### 15.17.2 三条结论（区分「已证实」与「证据不足」）

1. ✅ **已证实（硬结论）：no-GMC 在每一个 lr 档上难例增益都只有 GMC 的 ~27%~69%（多数 ~50%）**
   ⇒ **这不是「把 lr 调对就能补上」的差距** ⇒ **no-GMC 必须换底模**（§十五.5 / §十五.14 同向）。
2. ⚠️ **证据不足（不能下结论）：源域代价谁更重。**
   - **配对区**（同超参同种子，12 组，lr ≤ 1.1 为主）：no-GMC 的 T1 代价中位比 GMC **高 0.259 pp**（§十五.14）；
   - **高 lr 区**：GMC 有 49 个样本，中位 −0.745 pp；no-GMC 只有 **3** 个（−0.407 pp）⇒ **不可比**。
   ⇒ 正确表述：**「no-GMC 源域代价更大」这句撤回**（§十五.16.4 已标注更正）。
   ⇒ ⚠️ 纪律教训：**n=2 对 n=2 的单点对比不能支撑分布级断言**；这类结论必须落到分档/全样本。
3. ✅ **采样覆盖缺口（今晚要补）**：no-GMC 侧 lr 分布 **≥2.4 档一个都没有**，1.8~2.4 只有 3 个
   （GMC 侧分别是 24 / 25 个）⇒ **no-GMC 的最优操作点还没探到**，
   GMC 的最优区恰好在 ≥2.4（ΔT3 +6.2 pp）。
   ⇒ 夜里的 TPE 若继续往上探 lr，明早可能出现更强的 no-GMC 候选（也可能仍被 T1 否掉）。

### 15.17.3 工具与产物

- `compare_gmc_nogmc_matched.py` 新增 `lr_bands()` / `gmc_base()` / `nogmc_base()`；
  CLI 末尾直接打印分档表；`--ref` 指向参照系包。
- 汇总器新增 **§五之三「`lr_scale` 分档：源域代价 ↔ 难例增益（两侧同轴）」** ⇒ 明早 `SUMMARY.md` 直接可见。
- 两侧基线**都从代码/参照系包读**（GMC 用 `auto_epoch_gate_check.BASE`，no-GMC 用参照系包），**不手抄**。

---

## 十五.18、🔬 种子噪声带实测 + 已投放 4 个「确认重跑」（13:45）

### 15.18.1 发现：GMC 侧天然存在「同超参不同种子」的重复组 ⇒ 种子噪声带

GMC 侧 104 个 trial 里有 **3 组完全同超参**（11 个调参维度逐字相同）的重复组
（trial 号不同 ⇒ 随机种子不同）：

| 配置 | 种子数 | trial | T1 极差 | T2 极差 | T3 极差 | T2FP 极差 | T3FP 极差 |
|:--|--:|:--|--:|--:|--:|--:|--:|
| natural / lr 1.0 | 4 | 0000、0012、0014、0088 | 0.219 pp | 0.018 pp | **0.670 pp** | 6 | 50 |
| w4.0 / lr 1.0 | 3 | 0003、0013、0015 | 0.092 pp | 0.012 pp | **0.579 pp** | 5 | 41 |
| **w2.5 / lr 2.2702**（含 RC2 0080） | 3 | **0080**、0089、0090 | 0.249 pp | 0.009 pp | **0.094 pp** | 3 | 38 |

⇒ **种子噪声带（跨种子极差）**：T1 **0.09~0.25 pp**｜T2 **≤0.02 pp（几乎无噪声）**｜
T3 **0.09~0.67 pp**（lr 高时更小）｜T2FP 3~6｜T3FP 38~50。

### 15.18.2 对「trial_0014 vs trial_0011」待裁决问题的直接含义

| 比较 | 差距 | 对照种子噪声带 | 结论 |
|:--|--:|--:|:--|
| trial_0014 vs trial_0011 的 **T3** | 1.772 pp | ≤ 0.67 pp | ✅ **远超噪声 ⇒ trial_0014 的难例优势是真信号** |
| trial_0014 vs trial_0011 的 **T2** | 0.628 pp | ≤ 0.02 pp | ✅ 可信 |
| trial_0014 的 **T1 距红线** | **0.175 pp** | T1 带 0.09~0.25 pp | ⚠️ **落在噪声带之内/边缘 ⇒ 它的"被否"可能部分来自运气** |

⇒ **正确的处置不是改红线，而是多种子复现**：若 trial_0014 配置换种子后 T1 系统性越线
⇒ 红线判得对；若 T1 在 −0.5 pp 上下摆动 ⇒ 说明**红线对 no-GMC 偏紧**，需要重新定标（届时请示用户）。

### 15.18.3 已投放的 4 个确认重跑（**零额外 GPU、零重启**）

用**现成工具** `manu/training/enqueue_control_trials.py --from-trial X --count N`：
复制已有 trial 的精确超参入队，**换 trial 编号 ⇒ 换随机种子**；
入队 trial 走 `study.ask()` **优先被取走** ⇒ 无需重启调度器。

```bash
cd /tmp/pycharm_project_10ae9e2e
python manu/training/enqueue_control_trials.py \
  --storage "sqlite:///runs/optuna_nogmc/study.db" --study-name mix_ratio_search_nogmc \
  --from-trial trial_0014 --summary-csv runs/optuna_nogmc/optuna_summary.csv --count 2
python manu/training/enqueue_control_trials.py \
  --storage "sqlite:///runs/optuna_nogmc/study.db" --study-name mix_ratio_search_nogmc \
  --from-trial trial_0011 --summary-csv runs/optuna_nogmc/optuna_summary.csv --count 2
```

已入队（WAITING，`user_attrs.role` 可审计）：
`#20 #21 = CONFIRM_trial_0014`（w2.0 / lr 2.3648）｜`#22 #23 = CONFIRM_trial_0011`（w5.0 / lr 2.3391）

**副作用审计（为什么这不占额外资源）**：
- 4 个入队 trial 占的是**本来就会有的 4 个派单槽位**（截止 05:13 的派单窗口内），
  **总 trial 数不变**（仍 ~94），只是其中 4 个由 TPE 采样改为固定配置；
- 不重启调度器、不改搜索空间、不改任何判据/权重/产物；
- 超参取自落盘 CSV（`lr_scale` 4 位小数、`warmup` 3 位等）⇒ 与被复制 trial 的差异 ~1e-4 量级，
  对 lr 乘子影响 ~0.004% ⇒ **对「同超参不同种子」的比较无影响**（且 CSV 是比较基准）。

### 15.18.4 工具改动（纯增量，默认值=旧行为）

`manu/training/enqueue_control_trials.py` 新增 `--summary-csv`（默认仍是 `runs/optuna_mix_ratio/optuna_summary.csv`
⇒ **不传新参数行为逐位不变**），使 `--from-trial` 能读 no-GMC 侧 CSV。
md5 `90c9666414cc5c1248de8e71388740d6`。

### 15.18.5 汇总器新增 §五之四「种子噪声带」

`SUMMARY.md` 自动多一节：GMC 侧重复组的跨种子极差 + 读法（**T2 差异可信；T1/T3 差距要对照这条带**）。
⇒ 明早读盘时，候选榜旁边就摆着噪声带，可直接判断「trial_0014 的 T1 越线是真越线还是运气」。

---

## 十五.19、📊 20 trial 中期榜（14:15）：新榜首 `trial_0019`，且甜点区被 3 个 trial 交叉验证

### 15.19.1 当前榜（U 一律重算）

| trial | mix | lr_scale | 实测比 | T1 F1 | T2 F1 | T2 FP | T3 F1 | U | 判定 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| **trial_0019** | w5.0 | 2.1775 | 1:4.99 | 0.898070 | 0.997840 | **61（Δ+9）** | **0.792752** | **+0.031756** | ✅ 过线 |
| trial_0016 | w3.5 | 2.1017 | 1:3.49 | 0.897827 | 0.997570 | 70（Δ+18） | 0.791941 | +0.025702 | ✅ 过线 |
| trial_0014 | w2.0 | 2.3648 | 1:1.96 | 0.894145 | 0.996793 | 98（Δ+46） | **0.807805** | +0.017973 | ⛔ T1 |
| trial_0011 | w5.0 | 2.3391 | 1:4.95 | 0.896825 | 0.997421 | 77（Δ+25） | 0.790082 | +0.017764 | ✅ 过线 |
| trial_0015 | w2.0 | 1.4221 | 1:1.96 | 0.898240 | 0.996942 | 89（Δ+37） | 0.787479 | +0.012459 | ✅ 过线 |
| trial_0017 | w5.0 | 1.3719 | 1:5.04 | 0.898658 | 0.997271 | 78（Δ+26） | 0.778594 | +0.010283 | ✅ 过线 |
| trial_0018 | w8.0 | 2.7894 | 1:8.25 | 0.897750 | 0.996256 | — | 0.752564 | — | ⛔ T2（ΔFP +64） |

### 15.19.2 三条新结论

1. **新榜首 `trial_0019` 全面优于上一版榜首 `trial_0011`**：U +0.031756 vs +0.017764；
   T3 0.792752（+3.494 pp）｜T2 虚警只有 **61（ΔFP +9）**｜T1 −0.283 pp ✅｜T3FP 725（最低）。
2. **甜点区被 3 个 trial 交叉验证**（不是单点运气）：`trial_0011`（w5.0/2.339）、`trial_0016`（w3.5/2.102）、
   `trial_0019`（w5.0/2.178）**全部落在 lr 2.1~2.4 + 高 UAV 比例**，
   三轨一致地落在 T3 ≈ 0.790~0.793、T2 FP 61~77、T1 −0.28~−0.41 pp ⇒ **no-GMC 的 lr 甜点区 ≈ 2.1~2.4**。
3. **配比方向的取舍**（w2.0 vs w5.0，同为高 lr）：
   `w2.0`（UAV 少）T3 最高（0.807805）但**源域代价最大**（−0.675 pp，触线）；
   `w5.0`（UAV 多）T3 略低（0.7928）但**源域退让小、虚警少** ⇒ **混训比例是「难例增益 ↔ 源域保真」的第二个旋钮**。
   ⇒ 与 lr 旋钮叠加：**高 lr + 高 UAV 比例 = 源域稳、难例中等、虚警最少**（trial_0019 当前最优）。

### 15.19.3 确认重跑已扩到 8 个（覆盖 4 个候选 × 各 2 个新种子）

| 入队 trial | 复现对象 | 目的 |
|:--|:--|:--|
| #20、#21 | trial_0014（w2.0/2.3648） | T3 最高但触 T1 线 ⇒ 判「真越线还是运气」 |
| #22、#23 | trial_0011（w5.0/2.3391） | 上一版榜首 |
| #24、#25 | trial_0016（w3.5/2.1017） | 14:14 新榜首 |
| #26、#27 | trial_0019（w5.0/2.1775） | 14:15 新榜首（U 最高） |

⇒ 8 个确认 trial + 4 个原 trial ⇒ **4 组「同超参 3 种子」** ⇒ no-GMC 侧种子噪声带可**直接实测**，
与 GMC 侧 3 组噪声带同口径对比。**零额外 GPU、零重启、总 trial 数不变（仍 ~94+）**。

---

## 十五.20、✅ 确认重跑结案（15:10，24 trial）：`trial_0014` **不该放行**，且 no-GMC 的种子噪声**比 GMC 大**

### 15.20.1 `trial_0014` 配置（w2.0 / lr 2.3648）三种子 ⇒ **3/3 全部越 T1 红线**

| trial | T1 F1 | ΔF1_T1 | T2 FP | T3 F1 | ΔF1_T3 | U | 判定 |
|:--|--:|--:|--:|--:|--:|--:|:--|
| trial_0014 | 0.894145 | −0.675 pp | +46 | 0.807805 | +4.999 pp | +0.017973 | ⛔ T1 |
| trial_0020 | 0.893631 | **−0.727 pp** | +58 | 0.795494 | +3.768 pp | −0.001878 | ⛔ T1 |
| trial_0021 | 0.895339 | **−0.556 pp** | +55 | 0.805054 | +4.724 pp | +0.013572 | ⛔ T1 |

⇒ §十五.16.3 / §十五.18.2 提出的假设「**trial_0014 的 T1 越线可能只是运气**」**被证伪**：
三个种子的 T1 代价是 −0.556 / −0.675 / −0.727 pp，**全部越 −0.5 pp 红线**（且 0021 离红线只有 0.056 pp）。
⇒ **trial_0014 不放行；T1 红线对 no-GMC 并没有"太紧"** ⇒ **不需要、也不应该再动 T1 线**。
⇒ 而且该配置**本身不稳**：T3FP 三种子 1043 / 1280 / 1145（**极差 237**）⇒ trial_0014 那个"T3 最高 0.807805"
不可靠，换种子就掉到 0.7955。

### 15.20.2 `trial_0011` 配置（w5.0 / lr 2.3391）三种子 ⇒ **3/3 全部过线**（稳定可上线）

| trial | ΔF1_T1 | ΔFP_T2 | ΔF1_T3 | U |
|:--|--:|--:|--:|--:|
| trial_0011 | −0.407 pp | +25 | +3.227 pp | +0.017764 |
| trial_0022 | −0.368 pp | **+14** | +3.030 pp | +0.022476 |
| trial_0023 | −0.316 pp | +24 | +3.475 pp | +0.023050 |

⇒ **配置级稳定性已验证**：该配置 3 个种子 T1 全在 −0.41 pp 内、T2 虚警 14~25、T3 +3.0~+3.5、**U ∈ [+0.0178, +0.0231]**。

### 15.20.3 🔬 种子噪声带：**no-GMC 侧比 GMC 侧大**（尤其 T3）

| 指标 | GMC 侧（3 组） | **no-GMC 侧（2 组）** |
|:--|:--|:--|
| T1 极差 | 0.09~0.25 pp | 0.09~0.17 pp（相当）|
| T2 极差 | ≤0.02 pp | ≤0.05 pp（相当）|
| **T3 极差** | **0.09~0.67 pp** | **0.445 / 1.231 pp（1~2 倍以上）** |
| T2FP 极差 | 3~6 | 11~12（2 倍）|
| **U 极差** | — | **0.0053（w5.0 组）/ 0.0199（w2.0 组）** |

⇒ **关键推论**：当前榜前三名 trial_0019 (+0.031756) / trial_0016 (+0.025702) / trial_0023 (+0.023050)，
**U 的最大差距 0.0087 与实测 U 噪声带（0.0053~0.0199）同量级**
⇒ **现在还不能说 trial_0019 真的比 trial_0016 好**；
⇒ 而 #24~#27（trial_0016 / trial_0019 各 2 个复现）**正是为回答这个问题而投的**（预计 ~16:00 落地）。

⇒ **no-GMC 难例更"飘"**这件事本身也是结论：**no-GMC 语义下难例指标的种子噪声比 GMC 大 1~2 倍**
⇒ 交接线（no-GMC 版的 RC2 交接线）**必须用多种子中位数 + 上分位来定，不能用单点 U**。

### 15.20.4 当前榜（24 trial，15:11）

| trial | mix | lr_scale | T1 F1 | T2 F1 | T2 FP | T3 F1 | U | 判定 |
|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| **trial_0019** | w5.0 | 2.1775 | 0.898070 | 0.997840 | **61** | 0.792752 | **+0.031756** | ✅ |
| trial_0016 | w3.5 | 2.1017 | 0.897827 | 0.997570 | 70 | 0.791941 | +0.025702 | ✅ |
| trial_0023 | w5.0 | 2.3391 | 0.897736 | 0.997481 | 76 | 0.792561 | +0.023050 | ✅ |
| trial_0022 | w5.0 | 2.3391 | 0.897221 | 0.997810 | 66 | 0.788110 | +0.022476 | ✅ |
| trial_0014 | w2.0 | 2.3648 | 0.894145 | 0.996793 | 98 | 0.807805 | +0.017973 | ⛔ T1 |
| trial_0021 | w2.0 | 2.3648 | 0.895339 | 0.996524 | 107 | 0.805054 | +0.013572 | ⛔ T1 |
| trial_0020 | w2.0 | 2.3648 | 0.893631 | 0.996284 | 110 | 0.795494 | −0.001878 | ⛔ T1 |

⇒ 进度：24 完成 / 过线 12 / 剪枝 12 / 异常 0；4 卡满载（util 84~98%）。

---

## 十五.21、✅ 配置级定论（15:50，28 trial）：**冠军 = `trial_0019` 配置（w5.0 / lr 2.1775），3/3 种子全过线**

### 15.21.1 四个候选配置 × 各 3 种子（**交接该用的口径**）

| 配置 | 中位 T1 | 中位 T2 FP | 中位 T3 | **中位 U** | **过线** | U 极差 | T3 极差 |
|:--|--:|--:|--:|--:|:--:|--:|--:|
| **w5.0 / 2.1775**（trial_0019） | 0.898070 | **61** | **0.792253** | **+0.028657** | **3/3** | **0.0051** | **0.170 pp** |
| w3.5 / 2.1017（trial_0016） | 0.898293 | 77 | 0.791941 | +0.025598 | 3/3 | 0.0297 | 1.651 pp |
| w5.0 / 2.3391（trial_0011） | 0.897221 | 76 | 0.790082 | +0.022476 | 3/3 | 0.0053 | 0.445 pp |
| w2.0 / 2.3648（trial_0014） | 0.894145 | 107 | 0.805054 | +0.013572 | **0/3** | 0.0199 | 1.231 pp |

**逐种子原始值**（U）：

| 配置 | 种子 1 | 种子 2 | 种子 3 |
|:--|:--|:--|:--|
| w5.0 / 2.1775 | +0.031756（trial_0019） | +0.026694（trial_0026） | +0.028657（trial_0027） |
| w3.5 / 2.1017 | +0.025702（trial_0016） | +0.025598（trial_0024） | **−0.004043（trial_0025）** |
| w5.0 / 2.3391 | +0.017764（trial_0011） | +0.022476（trial_0022） | +0.023050（trial_0023） |
| w2.0 / 2.3648 | +0.017973（trial_0014） | **−0.001878（trial_0020）** | +0.013572（trial_0021） |

### 15.21.2 四条定论

1. **冠军 = `trial_0019` 的配置（w5.0 / lr_scale 2.1775）**：3/3 过线、
   U 中位 **+0.028657**（四组最高）、T3 中位 **0.792253**（最高）、T2 FP 中位 **61**（最低）、
   **T3 种子极差仅 0.170 pp（四组最稳）**、U 极差 0.0051（次小）。
   ⇒ **不只水平高，还最稳** ⇒ 这是本轮可上线模型。
2. **次席 `trial_0016` 配置（w3.5 / 2.1017）水平接近但不稳**：U 中位 +0.025598，
   但 trial_0025 直接掉到 **−0.004043**、T3 掉到 0.777721（T3 极差 **1.651 pp**、T2FP 极差 33、U 极差 0.0297）
   ⇒ **风险配置，不建议选**（单看 trial_0016 会高估）。
3. **`trial_0011` 配置（w5.0 / 2.3391）是可靠备选**：3/3 过线、U 中位 +0.022476、T3 极差 0.445 pp。
4. **`trial_0014` 配置（w2.0 / 2.3648）出局**：0/3 过线（T1 全越线），
   且 T3 种子极差 1.231 pp、T3FP 极差 **237** ⇒ 「T3 最高 0.807805」是**单种子假象**，不可用。

### 15.21.3 可交付的权重（冠军配置三个种子，gate_report 指向一致、md5 实测）

| trial | epoch3.pt md5 | 大小 | 备注 |
|:--|:--|--:|:--|
| **trial_0019** | `64bfd0a89903bd77091fc9ee40de7399` | 21,301,027 B | **建议交付这个**（该配置 U 最高的单种子：U +0.031756、T2 FP 61、T3 0.792752） |
| trial_0026 | `ff267635fed071f3c5f428875284489d` | 21,301,027 B | 同配置种子 2（U +0.026694） |
| trial_0027 | `24ded8b00f3e6f36ba218af42a0856d0` | 21,301,027 B | 同配置种子 3（U +0.028657） |

路径：`/tmp/pycharm_project_10ae9e2e/runs/optuna_nogmc/trials/trial_0019/weights/epoch3.pt`
（`gate_report.weights` 指向同一文件，md5 实测一致 ⇒ 指标↔权重闭环 ✅）

### 15.21.4 交接口径（写给明早的人）

- **选型口径**：**配置级多种子中位**（不是单 trial 最高 U）—— 本轮= `w5.0 / lr_scale 2.1775`。
- **no-GMC 的 U 不能与 GMC 的 U 横比**；no-GMC 侧种子噪声 T3 达 0.17~1.65 pp ⇒
  **交接线必须按「多种子中位 + 上分位」定标，不能用单点**。
- **T2 黄金池已不是瓶颈**（no-GMC 冠军 T2 0.997840 ≫ 绝对线 0.985、虚警仅 61）；
  **T3 难例仍差 GMC RC2 约 10.5 pp**（0.792753 vs 0.897694）⇒ **no-GMC 不能替代 GMC 上难例**，
  除非下一阶段「从 no-GMC 适配过的底模出发」把难例拉上来。







## 十五.22、🛠 汇总器新增 §四之三「配置级多种子榜」：把选型口径写进产物（15:55，28 trial）

### 15.22.1 为什么加这一节（不是锦上添花，是修一个**会误导人**的缺陷）

到 §十五.21 为止，「冠军 = `trial_0019` 配置」这个结论**只存在于我的分析里**，
`SUMMARY.md` 的 §四 / §四之二 仍然只给**单 trial** 榜 ⇒ 明早读盘的人会：

1. 看到 §四 的单点榜第一是 `trial_0014`（T3 0.807805 全场最高）⇒ 误取为模型；
2. 看不到 `trial_0014` 配置已经 **0/3 过线**；
3. 看不到 `trial_0016` 配置的种子极差高达 **1.651 pp**（单点 U 看着好，换种子就掉）。

⇒ **§四 的单点榜在 no-GMC 语义下是不可直接用于交接的**（§五之四 实测 U 噪声带 0.0051~0.0297）。
所以把「配置级多种子中位」这个口径**写进产物本身**，而不是只写在文档里。

### 15.22.2 实现（`manu/training/summarize_nogmc_search.py`，纯增量）

| 项 | 做法 |
|:--|:--|
| 分组键 | **11 个调参维度逐位相同**（`mix_spec / lr_scale / warmup_epochs / weight_decay / focal_beta / offset_weight / mosaic / scale / translate / clip_max_norm / freeze_bn`）；数值维度按 float 归一到 6 位小数（吸收 `7.21` vs `7.210` 写法差异），`mix_spec / freeze_bn` 按字符串 |
| 组内聚合 | 一律取**中位数**（U 仍由 `utility()` 重算，不信 CSV 列） |
| 表结构 | `§4.3.1` 已复现配置（n ≥ 2，**交接候选**）／`§4.3.2` 单点配置（n = 1，**仅参考未复现**），两张表分开排，混排会再次造成误读 |
| **冠军判定** | 在 **全种子过一级红线**的组里取 U 中位最高者（不是无条件取表首行）；若无全过组 ⇒ 显式写「交接线待定」 |
| 不稳定标记 | 任何组存在未过线种子 ⇒ 自动列进「已出局」清单 |
| 超参指纹 | 每个复现组回显 11 维原始 CSV 字符串 ⇒ 明早可直接照抄配方复现 |
| 单点 vs 冠军 | 用**已复现组实测的 U 种子噪声带**（max 极差）做判据：单点榜首高出冠军中位但**未超噪声带** ⇒ 自动提示「复现冠军仍居首」 |
| 其它 | §四 加「单点榜只作参考，交接以 §四之三 为准」；§六 加第 6 条口径声明 |

### 15.22.3 验证（**生成物必须真的看一眼**）

| 项 | 结果 |
|:--|:--|
| 本地跑（no-GMC 28 行） | ✅ §4.3.1 正确识别 4 个复现组，冠军 = **w5.0 · lr_scale 2.1775**（3/3 过线、U 中位 +0.028657、T3 中位 0.792253、种子极差 0.170 pp） |
| 同一份代码跑 **GMC 侧 104 行**（冒烟） | ✅ 冠军自动落到 **w2.5 · lr_scale 2.2702 = `trial_0080`（3/3 过线、T3 极差 0.094 pp）** ⇒ 与已知 GMC RC2 结论一致 ⇒ 分组与冠军逻辑**没有偏** |
| 不稳定检出 | ✅ 自动列出「`w2.0 · 2.3648`（0/3）」出局；GMC 侧自动列出「`natural · 1.0`（3/4）」「`w4.0 · 1.0`（2/3）」 |
| 服务器实跑（conda `uav`，写临时目录） | ✅ 输出一致，未污染 `runs/optuna_nogmc/SUMMARY.md`（该文件仍等管线收尾链生成） |
| 三侧 md5 | 首版 `f9ea0cdd7da5cdc8683045b0e1ae78dd`；**补「权重闭环」后终版 `a075294508c074a5c62d2617793cd6c1`**（本地 = SSHFS = 服务器，完全一致） |
| 对运行中进程的影响 | **零** —— 本次只改汇总器，未动 `optuna_mix_ratio_search.py` / 门禁脚本 / 判定链 ⇒ **无需重启任何守护**；收尾链是 fresh python 进程，会自动用新版本 |

### 15.22.4 遗留

- `SUMMARY.md` 仍**未生成**（须等 05:13 派单截止 → 末批收尾 → 收尾四步链）。
  届时 §四之三 会自动按最终 trial 集重新分组，**明早读盘以 SUMMARY.md 里的 §四之三 为准**。
- 本节写入时的 trial 集为 28 行；夜间 TPE 若采样到新的高 lr 配置，§4.3.1 排名可能变化
  （§五之三 已记录的「no-GMC 高 lr 档采样缺口」仍未闭合）。

## 十五.23、🔗 §四之三 补「权重闭环」+ 夜间收尾链路顺序审计（16:05）

### 15.23.1 补「权重闭环」：结论 → 产物 这一环原来缺一截

§四之三 落地后发现：它说「冠军是 w5.0/2.1775」，但**没有告诉读者该拿哪个 `epoch3.pt`**；
而 `gate_report.json` 里**只有权重路径、没有 md5**（实测：`gate_report.json` 顶层 10 个 key，
`weights` = `runs/optuna_nogmc/trials/<trial>/weights/epoch3.pt`，无 md5 字段）
⇒ 交接的人还得自己去 `trial_XXXX/weights/` 里翻、猜哪个是 epoch3、再自己算 md5。

**补法**（仍只在汇总器内，纯增量）：
每个已复现配置块下增加「权重（epoch3.pt，md5 现算）」子列表 ⇒
`trial → 权重相对路径｜md5｜字节数｜U`，并对**全种子过线**的组标「组内 U 最高 ⇒ 建议交付」
（**已出局组不标** —— 否则等于把 `trial_0014` 推出去，实测已修：出局组「建议交付」计数 = 0）。

md5 取自**现算**（`hashlib.md5` 分块读），不是抄之前记录的值 ⇒ 顺带成为**每日自校验**：
服务器实算 12 个权重，5 个与 §十五.21 手工记录的值逐一吻合
（`trial_0019` `64bfd0a8…`／`0026` `ff267635…`／`0027` `24ded8b0…`／`0014` `d142b5ee…`／`0011` `67c33b63…`），
另 7 个为新记录。耗时可忽略（12 × 21 MB）。

### 15.23.2 夜间收尾链路顺序审计（**不假设，去读脚本**）

| 脚本 | 实际顺序 | 结论 |
|:--|:--|:--|
| `run_nogmc_pipeline.sh`（⑤） | ① `backfill_summary_rows.py --apply`（**补 CSV 漏行**）→ ② `summarize_nogmc_search.py` | ✅ **backfill 在 summarize 之前** ⇒ 管线的 SUMMARY.md 不含漏行 |
| `nogmc_repair_after.sh` | ① `replay_measured_ratio.py --repair --apply` → ② `rejudge_with_reference.py --apply` → ③ `summarize_nogmc_search.py` → ④ 校验（实测比 >1000 的行数） | ✅ **summarize 是最后一次写 CSV 之后的最后一次渲染**；且 repair 链**不含 backfill**（backfill 归管线）⇒ 两链不重叠、不互相覆盖 |

⇒ **最终 `SUMMARY.md` 的数据来源链是完整的**：
`调度器写 CSV → 管线补漏行 → repair 链修实测比 → repair 链按最终参照系重判 → 汇总渲染`
且 §四之三 的「过线」列反映的是**重判后**的 verdict（`rejudge` 改的正是 `gate_verdict`）。

⚠️ 已排除的两个假设性风险：
1. 「backfill 排在 summarize 之后 ⇒ SUMMARY.md 是旧的」—— **不成立**（见上表，实测脚本顺序相反）；
2. 「repair 链等 `SUMMARY.md` 出现就动手 ⇒ 可能与管线并发写 CSV」—— **不成立**
   （管线里 summarize 是**最后一步**，写完 SUMMARY.md 管线只再打一行日志；且 repair 链
   `cd` 到项目根 ⇒ `--gmc-csv` 默认路径 `runs/optuna_mix_ratio/optuna_summary.csv` 可解析 ⇒
   §五之二/五之三/五之四 与新 §四之三 在最终产物里都会渲染）。

### 15.23.3 本轮三侧 md5

| 文件 | md5 |
|:--|:--|
| `manu/training/summarize_nogmc_search.py` | 首版 `a075294508c074a5c62d2617793cd6c1`（补权重闭环）→ 加 §五之五 → 加次席/取舍依据，**终版 `98cbd151137106c1006d4c990e4ccd0d`** |
| `manu/memory/nogmc_rebuild_20261027.md` | 见本节落盘后实测 |
| `manu/memory/memory.md` | 见 §12.6 增补后实测 |

### 15.23.5 后续补强：§四之三 加「次席 + 取舍依据」（16:45）

跑出来的成品里暴露了一个**新问题**：冠军（w5.0/2.1775，U 中位 +0.028657）与次席
（w3.5/2.1017，U 中位 +0.025598）**只差 0.003059**，而实测 U 种子噪声带最大 **0.029745**
⇒ **这个差距落在噪声带内** ⇒ 「冠军」二字是**统计上分不开的**；
只报冠军不报依据，等于把「实质相当」伪装成「明确胜出」。

补法（§4.3.1 内，纯增量）：

1. **次席行**：次席配置 + 种子数 + 过线数 + U 中位 + **落后幅度**；
2. **差距判读**：落后幅度 vs 实测 U 噪声带 max 极差 ⇒ 显式写「落在噪声带内 ⇒ 实质相当」
   或「超出噪声带 ⇒ 真信号」；
3. **取舍依据**（差距在噪声带内时**必给**，否则等于没结论）：本轮取 **T3 种子极差最小**
   （难例最稳）—— 理由是 no-GMC 侧 T3 噪声最大（0.170~1.651 pp），稳定性是本轮最该被引用的维度。
   实跑结果：选 `w5.0 · 2.1775`（T3 极差 **0.170 pp**），对照次席 `w3.5 · 2.1017`（**1.651 pp**）
   ⇒ 交接口径取冠军（**靠稳定性取胜，不是靠 U**）。

⚠️ 这条同时提示：no-GMC 侧若**新增 2 种子组**（如 §15.24 的 RC2 配方移植），它会被并入
「已复现配置」参与排名 —— 但它只有 2 个种子，中位 U 的证据量弱于 3 种子组，
所以「种子数」必须在表里显式可见（已在列），且**差距判读 + 取舍依据**这两段会自动跟着重算。


## 十五.24、🔬 补一个真实覆盖缺口：**RC2 自己那条配方在 no-GMC 侧从未跑过**（16:10）

### 15.24.1 发现（查 no-GMC ↔ GMC 的配方距离时撞出来的）

用户原始要求是「**参考 RC2 0080 的训练脚本，配置完全一样，只替换训练集和测试集**」。
本轮一直引用的证据是 §五之二 的**配对 A/B**（12 组同超参同种子），
但它覆盖的是**两个 study 前 12 个 trial 的随机配方**（几乎全是 `lr_scale=1.0` 的温和档），
**并不包含 RC2 自己那条配方**（`w2.5 / lr 2.2702 / …`）。

实测（归一化欧氏距离，0 = 配方逐位相同）：

| 项 | 结果 |
|:--|:--|
| no-GMC 侧与 RC2 配方**逐位完全相同**的 trial | **0 个** |
| no-GMC 侧**出现过 `w2.5` 配比档**吗 | **一个都没有**（28 trial 里 0 个）⇒ RC2 所在档位整档缺失 |
| 最近的 6 个邻居 | 距离全 ≥ **1.6167**，且都是 `lr=1.0` 的早期配方（trial_0000~0004 等） |
| 这些邻居的 ΔT3 vs GMC RC2 | **−12.0 ~ −12.9 pp**（距离远的配方当然差得更多，**不能当结论用**） |

⇒ 这是一个**真实、可陈述的覆盖缺口**，不是文字游戏：
**「严格照搬 RC2 配方、只换数据集」在本轮尚未被直接验证过**；
现有 −9~−10.5 pp 的结论建立在「重新搜出来的配方」与「前 12 个温和配方」上。

### 15.24.2 处置：投放 2 条**配方移植** trial（零额外 GPU）

沿用已获先例的 `enqueue_control_trials.py` 机制（占**既有派单槽位**）：
从 **GMC 侧** CSV 取 `trial_0080` 的精确超参 → 投进 **no-GMC** study。

```bash
python manu/training/enqueue_control_trials.py \
  --storage "sqlite:///runs/optuna_nogmc/study.db" --study-name mix_ratio_search_nogmc \
  --from-trial trial_0080 --summary-csv runs/optuna_mix_ratio/optuna_summary.csv \
  --count 2 --role-tag RC2_RECIPE_TRANSPLANT --role-note "…"
```

| 性质 | 状态 |
|:--|:--|
| 额外 GPU | **零**（占既有派单槽位，不新开搜索、不并行抢卡） |
| 搜索空间 | **不变**（`w2.5` ∈ `MIX_SPECS` ⇒ 不会触发 dynamic value space 报错） |
| 判据 / 红线 | **不变** |
| 预计 trial 总数 | ~105 → **~107**（上次 GMC 103；「跟以前一样」的口径不受影响） |
| 对 09:00 截止 | **无影响**（派单窗口仍止于 05:13，末批仍 ~06:05 收尾） |
| 入队结果 | **trial #32 / #33**（`role=RC2_RECIPE_TRANSPLANT`，`src_trial=trial_0080`） |

⚠️ **这是我的主动判断，用户可否决**：理由是它直接对应用户原始要求的那一条，
零额外 GPU、不动判据、不影响截止；代价是占用一个派单槽位（可能挤掉一次 TPE 探索）。

### 15.24.3 工具改造：`enqueue_control_trials.py` 加**可选** `--role-tag` / `--role-note`

原脚本的 role 恒为 `CONFIRM_<from_trial>`、note 恒为「确认重跑：…」。
本次要投的是**配方移植**（语义不同：源 trial 在**另一个 study** 里）⇒ 硬用 `CONFIRM_trial_0080`
会让人误以为它是 no-GMC 侧某个 trial 的复现 ⇒ 必须自解释。

- 两个参数**默认 `""`** ⇒ 不传时 role/note 与原来**逐位相同**（已实测：`role=CONFIRM_trial_0080`、
  `note=确认重跑：…` 未变）。
- 只改 `ua["role"]` / `ua["note"]` 两个字符串，**不动参数构造、不动入队逻辑**。
- 三侧 md5 `a6725d3657915b1fb4305b6fe49e2e7b`。

### 15.24.4 汇总器新增 §五之五「RC2 配方在 no-GMC 侧的对应」

| 子节 | 内容 |
|:--|:--|
| 5.5.1 | GMC RC2 的 11 维配方 + 三轨；no-GMC 侧**逐位相同**的 trial 列表；**ΔT1/ΔT2/ΔT3（绝对指标差，同帧同 GT 可比）**；若为 0 个 ⇒ 明写「覆盖缺口」 |
| 5.5.2 | no-GMC 侧离 RC2 配方最近的 6 个 trial（归一化欧氏距离）+ 各自的 ΔT3 vs RC2；并注明「距离只说明覆盖程度，不代表性能」 |

⇒ 等 #32/#33 落地后，**§五之五 会自动填出「同一条 RC2 配方、只换数据集」的实测三轨差**，
直接回答用户最初那句话，**不用再另跑一次**。

验证：本地 no-GMC 28 行实跑 ✅（5.5.1 正确报「覆盖缺口 0 个」；5.5.2 最近邻 6 行、距离与 §15.24.1 一致）；
三侧 md5 `d5d33fa69bcc4a9182129571bfe7beeb`。

### 15.24.5 派单验收（16:38）—— ✅ **参数逐项吻合**

调度器于 **16:37:56** 取走两条移植 trial（同批还派了 trial_0034/0035，4 卡满载）：

```
[LAUNCH] trial_0032 → GPU 1 | mix=w2.5 | lr_scale=2.270 | 协议 3ep/量末Ep | seed=32
[LAUNCH] trial_0033 → GPU 0 | mix=w2.5 | lr_scale=2.270 | 协议 3ep/量末Ep | seed=33
```

**worker 实际命令行**（从 `ps` 实抓，不是查日志）：

| 参数 | 值 | 与 GMC RC2 一致 |
|:--|--:|:--:|
| `--mix-spec` | `w2.5` | ✅ |
| `--lr-scale` | `2.2702` | ✅ |
| `--warmup-epochs` | `0.588` | ✅ |
| `--weight-decay` | `0.00026906` | ✅ |
| `--focal-beta` | `2.75` | ✅ |
| `--offset-weight` | `0.4` | ✅ |
| `--mosaic` | `0.15` | ✅ |
| `--scale` | `0.1` | ✅ |
| `--translate` | `0.05` | ✅ |
| `--clip-max-norm` | `5.65` | ✅ |
| `--freeze-bn` | `0`（False） | ✅ |

**Optuna study 里的固定参数**（`study.get_trials` 实读）：
`#32/#33 state=0(RUNNING)`，`params` 与 GMC `trial_0080` 逐位一致，
`user_attrs = {role: RC2_RECIPE_TRANSPLANT, src_trial: trial_0080, enqueued_by: enqueue_control_trials.py}`。

⇒ **验收通过**（与前 8 条确认重跑同一套验收方式：看 worker 真实命令行，不看日志措辞）。

⏳ **待落地**：两条约 17:25 结算 ⇒ 服务器侧常驻观察者 `watch_settle.sh`（PID 2260129）会在
它们进 CSV 后**实跑汇总器**并打印 §5.5.1，确认「同配方换数据」的对照表**自动填出**（而不是人工填）。

### 15.24.6 顺带得到的一个数据点

`trial_0030`（**w2.5 / lr_scale 1.403**）结算为 **PRUNED（FAIL_TRACK2_SAFETY）**，
T1 0.897752 / T2 0.995746 / T3 0.745579 ⇒
**同一配比档（w2.5）但 lr 只有 RC2 的 62% 时，虚警就爆了**（ΔFP_T2 > 60）。
⇒ no-GMC 侧的 `w2.5` 档**不是「配比一换就能用」**，必须配高 lr（RC2 的 2.2702）。
这也解释了为什么 §5.5.2 的最近邻全是 lr=1.0 的档：**no-GMC 侧真正的高 lr 甜点区还没被 TPE 覆盖到**。


## 十五.25、🔍 全量对账：28 行 CSV ↔ 各自 `gate_report.json`（16:20）

> 纪律：**最终 SUMMARY.md 的每一行都建在 `optuna_summary.csv` 上** ⇒ 该表若与实测脱节，整张榜都是假的。
> 故对**已结算的 28 行逐行**回查各自 trial 的 `gate/epoch3/gate_report.json`。

| 检查项 | 结果 |
|:--|:--|
| `t1_f1 / t1_fp / t2_f1 / t2_fp / t3_f1 / t3_fp` 六列 ↔ gate_report 对应字段 | **28/28 行逐位一致**（容差 1e-6）✅ |
| 缺 `gate_report.json` 的行 | **0** ✅ |
| `gate_verdict` ↔ gate_report 的 `verdict` | **16 行不同 —— 但全部是**预期内**的（见下） |
| CSV 的 `utility` 列 ↔ `utility()` 重算 U | **7 行差 ≤ 1.1e-6**（trial_0019：CSV 0.03175468 vs 重算 0.03175583）⇒ **不改变任何排名**（相邻名次差 ~1e-3 量级） |

### 15.25.1 那 16 行 `verdict` 差异是什么（**不是 bug，是重标定的正常结果**）

| 来源 | 含义 |
|:--|:--|
| gate_report 的 `verdict` | **跑那一刻**的门禁原始判定，用的是**旧线**（ΔFP_T2 ≤ **10**，且基线是 GMC 语义的）⇒ 大量 trial 记成裸 `FAIL` |
| CSV 的 `gate_verdict` | **按 no-GMC 参照系重判**后的判定（ΔFP_T2 ≤ **60** + no-GMC 冻结基线 + T1 红线 −0.005）⇒ 带**具体原因**（`FAIL_TRACK2_SAFETY` / `FAIL_TRACK1_RETREAT` / `PASS`） |

逐条核对（16 行）全部自洽：
- `trial_0000~0003`：旧线判 FAIL，**新线判 PASS**（ΔFP_T2 +43~+51 ≤ 60）✅
- `trial_0004/0005/0006/0007/0008/0010/0013/0018`：新旧都 FAIL，且新线给出**具体原因** `FAIL_TRACK2_SAFETY` ✅
- `trial_0009/0014/0020/0021`：新旧都 FAIL，原因 `FAIL_TRACK1_RETREAT`（T1 触线，与 §十五.20 的复现结论一致）✅

⇒ **结论：数据层无问题**；而且 **repair 守护在 06:10 还会用最终参照系再全量重判一遍**
（`rejudge_with_reference.py --apply`）⇒ 最终 SUMMARY.md 里的 verdict / 过线列
与 §四之三 的「过线 n/m」都会是**终态**值。

### 15.25.2 这轮审计顺带确认了「U 一律重算」的必要性

CSV 的 `utility` 列与 `utility()` 重算值差最多 **1.1e-6**（落盘时用的基线精度与参照系包里的全精度不同）。
**数值上无碍**（排名间隔 ~1e-3），但这正好说明：**若直接信 CSV 列，U 就会有 1e-6 量级的系统性偏移**
⇒ 汇总器全部用重算值（§四 / §四之二 / §四之三 / §五之二 一致），**这条铁律继续保留**。

## 十五.26、✅ **RC2 配方移植实测落地**（17:25）：交接配置更新为 **w2.5 / 2.2702**（= RC2 原配方）

### 15.26.1 实测结果（§5.5.1 自动填出，未人工填写）

两条移植 trial 已结算（`trial_0032` / `trial_0033`，各 46.82 min，**2/2 过一级红线**）：

| trial | T1 | T2 | T2FP | T3 | T3FP | ΔT1 vs RC2 | ΔT2 vs RC2 | **ΔT3 vs RC2** | verdict |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| trial_0032 | 0.897766 | 0.996852 | 94 | **0.801626** | 1109 | −0.414 pp | −0.210 pp | **−9.607 pp** | PASS |
| trial_0033 | 0.898371 | 0.997331 | 81 | **0.801353** | 984 | −0.354 pp | −0.162 pp | **−9.634 pp** | PASS |
| **中位** | 0.898068 | 0.997092 | 88 | **0.801489** | — | **−0.384 pp** | **−0.186 pp** | **−9.620 pp** | 2/2 |

⇒ **直接回答用户最初那句话**：「**同一条 RC2 配方，只把训练/测试集换成 no-GMC**」的结果是
**T1 −0.38 pp、T2 −0.19 pp、T3 −9.62 pp**。
⇒ 与 §五之二 的配对 A/B（12 组同配方，ΔT3 中位 **−9.048 pp**）**互相印证**
（−9.62 vs −9.05，差 0.57 pp 在种子噪声范围内）⇒
**「T3 难例掉 ~9~10 pp」是 GMC 语义本身带来的，不是配方没调对。**

### 15.26.2 🔴 交接配置因此**更新**：w5.0/2.1775 → **w2.5 / 2.2702**

| 配置 | 种子 | 过线 | U 中位 | **T3 中位** | T3 极差 | T2FP 中位 |
|:--|--:|:--:|--:|--:|--:|--:|
| w5.0 · 2.1775（此前结论） | 3 | 3/3 | **+0.028657** | 0.792253 | 0.170 pp | **61** |
| **w2.5 · 2.2702（= RC2 原配方）** | 2 | **2/2** | +0.026865 | **0.801489** | **0.027 pp** | 88 |

- **U 不可分**：差 0.001792 ≪ 实测 U 噪声带 0.029745；
- **T3 可分**：高 **0.924 pp**，而「中位不确定度」阈值只有 0.117 pp
  （保守取组内种子极差/2/√n：0.027/2/√2 = 0.010、0.170/2/√3 = 0.049）⇒ **超出 8 倍**；
- ⇒ 交接口径按「U 不可分时以 **T3 难例**为主判据」取 **w2.5 / 2.2702**。

**这同时是个有意思的结论**：用户原始要求的「照搬 RC2 配方、只换数据集」这条路线，
**产出的难例表现（0.8015）反而比本轮 TPE 重新搜出来的最优配置（0.7923）更好**；
且它正是 **RC2 自己那条配方** ⇒ 原始需求本身可行，且是当前最好的那条路。

### 15.26.3 🔴 我自己那条取舍规则有两个缺陷，被成品跑出来打脸后修掉

| 缺陷 | 现象 | 修法 |
|:--|:--|:--|
| **n 绑架** | 原规则「T3 种子极差最小者胜」拿 **2 种子组的 0.027 pp** 比 **3 种子组的 0.170 pp** ⇒ **极差随种子数结构性偏小**（2 个点的极差天然更小）⇒ 假优势 | 改用**组内极差折算的中位不确定度**（极差/2/√n）做判别阈值，按 **T3 中位**判可分性；**不再跨组比较极差** |
| **自指 bug** | 选出的组与「对照次席」是同一组 ⇒ 文字变成「选 w2.5/2.2702… 对照次席 w2.5/2.2702」 | 重写为**一条显式可复算的规则**（U 差距 → 是否超噪声带 → 转看 T3 → 阈值判可分 → 否则取种子数多者），并把每一步都打印出来 |

⇒ 新规则的完整输出（可直接复算）：
```
U 榜首 w5.0·2.1775(+0.028657,3种子) vs 次席 w2.5·2.2702(+0.026865,2种子)，差 0.001792
⇒ 落在 U 噪声带 0.029745 内 ⇒ U 不可分 ⇒ 转看 T3 难例
T3: 0.801489(2种子,极差0.027) vs 0.792253(3种子,极差0.170)，差 0.924 pp，
    阈值 2×(0.010+0.049)=0.117 pp ⇒ 超出 ⇒ T3 上可分 ⇒ 交接取 w2.5·2.2702
```
⚠️ 规则**自己指出**：`w2.5·2.2702` 只有 2 个种子（U 榜首有 3 个）⇒ 建议再补 1 个同超参种子。

### 15.26.4 补第 3 种子（零额外 GPU）

按规则自身的提示，投放 **trial #40**（`role=RC2_RECIPE_TRANSPLANT`，同一条 RC2 配方的第 3 个种子）
⇒ 把 `w2.5/2.2702` 补成 3 种子组、**消除 2 vs 3 的证据量差异**。
（仍属既有机制：占派单槽位、不动搜索空间与判据、不影响 05:13 派单截止与 09:00 交付。）

### 15.26.5 终版三侧 md5

| 文件 | md5 |
|:--|:--|
| `manu/training/summarize_nogmc_search.py` | `f31769409e218ed40839da5685102cc1` |
| `manu/training/enqueue_control_trials.py` | `a6725d3657915b1fb4305b6fe49e2e7b` |

## 十五.27、🌅 夜间收官（10-28 05:47 收尾）+ 早上 09:10 读盘 + 单点榜首复现（09:12 起跑）

### 15.27.1 夜间运行结果（**09:00 截止前 3.2 h 完成**）

| 项 | 结果 |
|:--|:--|
| 收尾时刻 | 管线 **05:46:38** ✅ → repair 链 **05:47:10** ✅（校验：实测比异常行 **0**）⇒ 比 09:00 截止**提前 3.2 h** |
| trial 总数 | 派单 **97**，CSV **97 行**；其中 **96 个有完整指标**（`trial_0076` 门禁崩溃、`GATE_REPORT_INCOMPLETE` ⇒ 唯一 1 次失败） |
| 门禁分布 | **PASS 69** ／ `FAIL_TRACK1_RETREAT` 16 ／ `FAIL_TRACK2_SAFETY` 11 ／ PRUNED 27 |
| 单 trial 耗时 | 中位 **47.5 min**（最短 46.1 / 最长 52.8） |
| 夜间 TPE 的收获 | **高 lr 区被探到了**（lr 2.2~2.95 大量新点）⇒ §15.17 记录的「no-GMC 高 lr 采样缺口」**已闭合** |

### 15.27.2 读盘时发现的两处**记账缺口**（数据没丢，是状态没标）

- **4 个 trial 仍是 `RUNNING`**（#93~#96）：但它们的 **gate_report / 指标 / 权重 / CSV 行全都在**
  ⇒ 真实情况是**已跑完**，只是调度器被杀（07:00 硬停）前没来得及 `study.tell()` ⇒ **纯记账缺口，不影响任何指标**。
- 由此 `done_count` 实为 **92**（不是 96）⇒ 复现时的 `--n-trials` 必须按 92 算，否则会多跑。
- 重启调度器时它**自愈**了这些陈旧 RUNNING（标 FAIL，不污染 TPE）⇒ 该机制已内建。

### 15.27.3 配置级交接榜（5 个复现组，全部 3 种子）

| # | 配置 | 种子 | 过线 | U 中位 | T1 中位 | T2 中位 | T2FP 中位 | **T3 中位** | T3 极差 |
|--:|:--|--:|:--:|--:|--:|--:|--:|--:|--:|
| 1 | w5.0 · 2.1775 | 3 | 3/3 | **+0.028657** | 0.898070 | 0.997840 | **61** | 0.792253 | 0.170 pp |
| 2 | w3.5 · 2.1017 | 3 | 3/3 | +0.025598 | 0.898293 | 0.997451 | 77 | 0.791941 | 1.651 pp |
| 3 | **w2.5 · 2.2702（RC2 原配方）** | 3 | **3/3** | +0.022876 | 0.898122 | 0.996852 | 94 | **0.801353** | 0.101 pp |
| 4 | w5.0 · 2.3391 | 3 | 3/3 | +0.022476 | 0.897221 | 0.997481 | 76 | 0.790082 | 0.445 pp |
| 5 | w2.0 · 2.3648 | 3 | **0/3** | +0.013572 | 0.894145 | 0.996524 | 107 | 0.805054 | 1.231 pp |

**交接配置 = `w2.5 · 2.2702`（= RC2 0080 原配方）**：U 不可分但 T3 可分（+0.910 pp vs 阈值 0.156 pp）。
交付权重（md5 已在服务器**独立复核**）：
`trial_0033` `e87f277f14cb9c0409a216615c81ae32`（组内 U 最高，**建议交付**）／
`trial_0032` `07e0901af3086e302642658b0bf2c46c`／`trial_0040` `be8649578e2f55a69af9f92cf3e72295`。

### 15.27.4 🔴 读盘后新发现：**单点榜首比交接配置更好** ⇒ 已触发复现

夜间 TPE 找到的新点里，`trial_0080`（**w3.0 / lr_scale 2.8821**）：
**T1 0.896372 ｜ T2 0.998050（FP 60）｜ T3 0.811897 ｜ U +0.047261 ｜ PASS**
⇒ **T3 比交接配置高 1.054 pp**，是 no-GMC 侧 T3 最高的**过线** trial。

但它是**单点**（n=1）：
- U 差 +0.024385 **未超** U 噪声带 0.029745 ⇒ U 上不可分；
- T3 高 1.054 pp **超过**交接组自身种子极差 0.101 pp，但**未超**全局最大种子极差 1.651 pp
  ⇒ **不能排除是抽到了好种子** ⇒ 必须复现才能换交接线。

⇒ **处置**（09:12 起跑）：`enqueue_control_trials.py --from-trial trial_0080 --summary-csv runs/optuna_nogmc/optuna_summary.csv --count 2`
⇒ 入队 **#97 / #98**（`role=TOP_SINGLE_REPRO`），
再以 `--n-trials 94 --max-hours 1.2` 重启调度器 ⇒ **只跑这 2 条、不重新搜索**
（`done_count`=92 ⇒ 92+2=94 即停；`study.ask()` 优先取走入队 trial；实测已取走 #97→GPU0、#98→GPU1）。
服务器观察者 `watch_repro.sh`（PID 2397300）在两条落地后**自动重跑收尾链**
（backfill → replay → rejudge → summarize）刷新 `SUMMARY.md`。

### 15.27.5 汇总器再修一处逻辑不一致

§4.3.2 对单点榜首原先**只用 U 比较** ⇒ 会输出「复现冠军仍居首」，而单点的 T3 明明更高 ⇒ **误导**。
已改为与 §4.3.1 **同一条两轨判据**：先 U（超噪声带？），再 T3（超交接组自身种子极差？超全局最大种子极差？）
⇒ 现在输出「U 不可分 + T3 高 1.054 pp 但不能排除抽到好种子 ⇒ 须复现」。

终版 `summarize_nogmc_search.py` md5 `340c919d07c5ffce84be8faab38a96c0`。

## 十五.28、🔍 最终产物全量审计（97 行）+ §五之三 说明改为数据驱动 ⇒ **一个更锋利的结论**（09:20）

### 15.28.1 全量对账（之前只查了 28 行，这次 97 行全查）

| 检查项 | 结果 |
|:--|:--|
| 六列指标（t1_f1/t1_fp/t2_f1/t2_fp/t3_f1/t3_fp）↔ `gate_report.json` | **96/96 行逐位一致**（容差 1e-6），**0 不一致**，**0 缺报告** ✅ |
| 无指标行 | **1**（`trial_0076`，门禁崩溃 `GATE_REPORT_INCOMPLETE`）⇒ 汇总器按设计跳过 |
| 实测比 | **0 行** >1000；范围 **0.998 ~ 8.250**（正常）✅ |
| CSV `utility` 列 ↔ `utility()` 重算 | 最大偏差 **1.524e-06**，14 行超 1e-6 ⇒ 再次印证「**U 一律重算**」（不影响排名，排名间隔 ~1e-3） |

⇒ **最终 `SUMMARY.md` 所依赖的数据层是干净的。**

### 15.28.2 🔴 §五之三 的说明文字**过时了**（读产物读出来的）⇒ 改为数据驱动

原文是早先（no-GMC 高 lr 档只有 3 个样本时）写死的：
「⚠️ 源域代价（ΔF1_T1）一列**暂不可比**：no-GMC 在高 lr 档样本太少」＋
「⚠️ **采样覆盖缺口**：no-GMC 侧的高 lr 档可能尚未采样」。

⇒ **夜间 TPE 把高 lr 档填满后（no-GMC ≥2.4 档 n=28、GMC n=24），这两句已经不成立**，
继续留在交付物里会**误导读盘的人**。改成**按实际 n 判定**：
两侧最高 lr 档 n 均 ≥10 ⇒ 输出「✅ 源域代价列现已可比」＋逐档对比；否则才保留旧警告。
同时把「大致只有 GMC 的一半」也改成**实算比例**（两侧 n≥5 的档）。

### 15.28.3 💡 由此得到一个**之前拿不到的新结论**

| lr 档 | GMC：ΔT1 / ΔT3 | no-GMC：ΔT1 / ΔT3 |
|:--|:--|:--|
| <0.7 | −0.358 / +1.436 pp（n=6） | −0.070 / +0.393 pp（n=3） |
| 0.7~1.2 | −0.373 / +2.968 pp（n=22） | −0.077 / +1.390 pp（n=12） |
| 1.2~1.8 | −0.530 / +4.276 pp（n=27） | −0.262 / +2.141 pp（n=16） |
| 1.8~2.4 | −0.684 / +5.726 pp（n=25） | −0.329 / +3.484 pp（n=38） |
| **≥2.4** | **−0.776 / +6.203 pp（n=24）** | **−0.456 / +3.631 pp（n=28）** |

- **难例增益比（no-GMC / GMC，两侧 n≥5 的 4 个档）= 47% ~ 61%**（实算）。
- **源域退让：no-GMC 在所有 lr 档都更小**（最高档 −0.456 vs −0.776 pp）。

⇒ **新结论：no-GMC 的混训「性价比」更低 —— 它更少地动源域，却换来更少的难例收益。**
（此前只能说「难例增益只有一半」，且不敢比较源域代价；现在两侧都可比。）
⇒ 这把诊断**钉死在语义本身**：不是「no-GMC 更伤源域所以难例上不去」，
而是「no-GMC 模式下同样的混训，对源域更温和、对难例却更无效」⇒ **必须换底模**。

终版 `summarize_nogmc_search.py` md5 `1c9bdda7be6a4e5e835ceae3de21863c`
（本地 / SSHFS / 服务器三侧一致；GMC 侧 104 行冒烟复跑正常 ✅）。

## 十五.29、✅ **最终交接配置 = `w3.0 / lr_scale 2.8821`**（10-28 09:59 复现落地，3/3 过线）

### 15.29.1 单点榜首复现成功 ⇒ 交接线切换

`trial_0080`（**w3.0 / lr_scale 2.8821**）的两个复现种子已落地（09:59:34）：

| trial | T1 | T2 | T2FP | **T3** | T3FP | U | verdict |
|:--|--:|--:|--:|--:|--:|--:|:--|
| trial_0080（原单点） | 0.896372 | 0.998050 | 60 | 0.811897 | 850 | +0.047261 | PASS |
| trial_0097 | 0.896828 | 0.997332 | 84 | 0.802374 | 1067 | +0.026518 | PASS |
| trial_0098 | 0.896338 | 0.998349 | 49 | **0.816311** | 713 | +0.055740 | PASS |
| **中位** | **0.896372** | **0.998050** | **60** | **0.811897** | — | **+0.047261** | **3/3** |

⇒ **3/3 过一级红线**，不再是单点 ⇒ 进入配置级榜并**成为新的 U 榜首**（U 中位 +0.047261）。

### 15.29.2 交接规则的复算（§4.3.1 自动输出，可逐位复算）

```
U 榜首 w3.0·2.8821(+0.047261,3种子) vs 次席 w5.0·2.1775(+0.028657,3种子)，差 0.018604
⇒ 落在实测 U 噪声带 0.029745 内 ⇒ U 不可分 ⇒ 转看 T3 难例
T3 中位：w3.0·2.8821 0.811897(3种子,极差1.394pp) vs w2.5·2.2702 0.801353(3种子,极差0.101pp)
    差 1.054 pp，阈值 2×(中位不确定度 0.402+0.029)=0.863 pp ⇒ 超出 ⇒ T3 上可分 ⇒ 交接取 w3.0·2.8821
```
⇒ **交接配置 = `w3.0 / lr_scale 2.8821`**（3 种子、3/3 过线、U 中位 +0.047261、**T3 中位 0.811897**）

交付权重（md5 已在服务器**独立复核**，与 SUMMARY 现算值一致）：

| trial | md5 | 大小 | U | 备注 |
|:--|:--|--:|--:|:--|
| **trial_0098** | `48b6075a605406534cd03d6844509ad7` | 21,301,027 B | +0.055740 | **组内 U 最高 ⇒ 建议交付** |
| trial_0080 | `a2684150b4e91d356020e30c20b87e87` | 21,300,963 B | +0.047261 | 组内 T3 中位所在那条 |
| trial_0097 | `a564d53535f39699a888543a0025233b` | 21,300,963 B | +0.026518 | 最弱的一条 |

路径：`/tmp/pycharm_project_10ae9e2e/runs/optuna_nogmc/trials/trial_0098/weights/epoch3.pt`

### 15.29.3 ⚠️ 交接的**真实取舍**（必须一起看，不能只看 T3）

| 维度 | **w3.0 · 2.8821（新交接）** | w2.5 · 2.2702（RC2 原配方，保守选项） |
|:--|:--|:--|
| T3 中位 | **0.811897** | 0.801353（低 1.05 pp） |
| **T3 种子极差** | **1.394 pp（很飘）** | **0.101 pp（很稳）** |
| T2 F1 / 虚警 | **0.998050 / 60** | 0.996852 / 94（虚警多 34） |
| T1 中位 | 0.896372（**距红线 0.895897 仅 +0.48 pp**） | 0.898122（余量 +2.23 pp） |
| U 中位 | **+0.047261** | +0.022876 |

⇒ **判别只超出阈值 1.22 倍（1.054 vs 0.863 pp），属边际胜出**；
且新配置的 **T1 红线余量只有旧配置的 1/5**。
⇒ **若只交付一个权重、且更看重稳 ⇒ 选 `w2.5 · 2.2702`（T3 极差 0.101 pp）；
若优先难例精度与虚警 ⇒ 选 `w3.0 · 2.8821`（本表按既定规则给出的交接配置）。**
⚠️ 两者**都**不能替代 GMC RC2 的难例（0.897694），差距仍有 **~8.6 pp**。

### 15.29.4 收尾链复跑的结果（watcher 自动执行，幂等性得到验证）

| 步骤 | 结果 |
|:--|:--|
| ① backfill | CSV **99 行**，已结算 99 个，**缺行 0** ⇒ 无需补采 ✅ |
| ② replay 修实测比 | **修 0 个**（重放幂等 ✅，与前一次结果一致） |
| ③ rejudge | 只碰 2 条新 trial（`status None→SUCCESS`、`verdict PASS→PASS`、U 不变）⇒ 旧行**零改动** ✅ |
| ④ 重出 SUMMARY | 13 个章节齐全，§4.3.1 变 **6 组**，交接结论按新规则重算 ✅ |
