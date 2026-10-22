> ⚠️ **本文件不是独立 memory 章节**，它是 `memory.md` **§四十七** 的**逐字源文**（2026-10-21 由
> `manu/docs/route_b/train_fit_diag/memory_section_47.md` 原样追加进 `memory.md` 末尾）。保留此处仅为溯源用。

# §四十七 Train 划分拟合度诊断：Ep3 拟合健康、不过拟合、增益 85.4% 泛化（2026-10-21）

## 1. 结论一句话

用**与 §四十三 门禁完全一致**的口径、只换权重，量了 RC1(Ep3) 在**训练集**上的拟合程度：
**F1 93.34% → 97.74%，Recall 89.67% → 96.92%，FP 反降 43.1%** ⇒
**不欠拟合、不过拟合、泛化余量充足**；**本轮 4.40pp 的 train 增益有 85.4% 泛化到了盲测 001324**。
完整分析见 `manu/docs/route_b/train_fit_diag/train_fit_diagnostic_analysis.md`。

## 2. 训练集整体对照（9 条序列 / 13,918 帧 / 12,120 GT —— 正是模型训练时逐帧看过的样本）

| 指标 | 底模 trial_0474 | 微调 RC1(Ep3) | Δ |
|:--|--:|--:|--:|
| F1 | 0.933396 | **0.977410** | **+0.044014** |
| Recall | 0.896700 | **0.969224** | **+0.072525** |
| Precision | 0.973225 | **0.985735** | **+0.012510** |
| TP | 10,868 | 11,747 | +879 |
| FP | 299 | **170** | **−129（−43.1%）** |
| FN | 1,252 | **373** | **−879（−70.2%）** |

🔑 **Recall 与 Precision 同涨、FP 反降** ⇒ **不是"放宽门限换召回"，是真变强**。
逐序列：**9 条序列 F1 全部上升，无一条下降**。

## 3. 三个诊断信号的判定（用户预设判据 → 实测）

| # | 信号 | 判据 | 实测 | 判定 |
|:--|:--|:--|:--|:--|
| 1 | **欠拟合** | train Recall 爬到 92%~96%；002001 FN 被砍大半 | Recall 89.67%→**96.92%**；**002001** FN **606→229（−62.2%）**、R 81.24%→92.91%、F1 +8.17pp；**003002** FN **65→12（−81.5%）**、F1 **+29.15pp**、R 51.13%→90.98% | ✅ **不欠拟合** |
| 2 | **过拟合** | train F1 逼近 99.5%+ 且 FP→0 | train F1 **97.74%**（远未到 99.5%）；FP 170（未趋零；**2 条序列 FP 已压到 0**：001810、013631；另 003002=2、001415=8） | ✅ **不过拟合，余量充足** |
| 3 | **顽固误检** | train 内 FP 分布，找 000703 同形态 | 002001 **−103**、001415 **−27**、001810 −6、013631 −5、003002 −5、014144 0、014227 +3、001550 +4、**002959 +10** | ⚠️ **机制在 train 内成立，但 002959 反弹** |

### 3.1 两个被证伪的旧假说

- ❌ **「全图 1:3.09 混训下大背景稀释小目标梯度」**（用户预设的欠拟合解释）—— **被证伪**：
  002001 / 003002 这类底模失效序列被显著救回，模型容量不是瓶颈。
- ❌ **「Ep3 已把容量用尽 / 开始对建筑物高频边缘过拟合」**—— **被证伪**：train F1 97.74%，
  FP 未趋零，判别流形仍平滑。
- ⚠️ **train↔blind 落差略扩**：底模 9.61pp → Ep3 **10.25pp**（train 涨得比 blind 快，符合"先拟合后泛化"）。

### 3.2 🔴 新暴露的风险点：002959 与 000703 同构

**002959 是训练集里被反复见过的序列，Ep3 在它上面 FP 反而 45→55（+10）**，
与 val 侧未见过同形态的 `000703`（ΔFP **+25**，占黄金池总误报 88%）**现象同构**
⇒ **压制某类地表切变线的能力没有泛化到同形态新序列，反而在那里新造误报**。
⇒ 处方升级为**同时打中两处**：把 **000703 与 002959** 的失效机理并入训练集负样本（各补若干纯背景切片）。

### 3.3 ⭐ 拟合度 → 泛化度 转化率（本轮最有价值的派生结论）

| 指标 | ΔTrain | Δ盲测 001324 | **泛化率** |
|:--|--:|--:|--:|
| F1 | +0.044014 | +0.037593 | **85.4%** |
| Recall | +0.072525 | +0.054200 | **74.7%** |
| Precision | +0.012510 | +0.013333 | **106.6%（完全泛化）** |

⇒ 4.40pp 的 train 增益里 **85.4% 泛化到了盲测**，14.6% 是纯训练集拟合；
**精度增益 100% 泛化，召回增益只 74.7% 泛化**（45.8% 流失）。
⇒ **下一轮杠杆 = 召回的泛化，不是继续拟合** ⇒ 局部切片放大（方案 A）是对症方向。
⇒ 残余缺口高度集中：Ep3 的 train FN 373 里 **002001 占 229（61.4%）**、FP 170 里 **002001 占 79（46.5%）**
⇒ 切片方案应**针对 002001 这类"暗刚体"**，而不是全局重训。

## 4. ⚠️ 本轮修掉的两个静默坑（train 划分专属，务必记住）

1. **stride≠1 的帧号错位（致命）**：`manu/data/multiscale_heatmap_cache.py:403` 写的是
   `frame_index = start + index` = **排序位置**，不是原始帧号。
   val 是 stride=1 才天然对齐；**train 是 stride=5**（13,918 帧来自 69,589 帧），
   直接把原始真值目录喂给评测脚本会**整体错位且不报错**。
2. **命名陷阱（静默清零）**：`manu/evaluation/eval_multiscale_cache_th.py:100` 的 `load_ground_truth`
   只认 `{seq}__{digits}.txt` 且 digits 必须纯数字；数据集自带 `labels/train/{seq}__frame_000000.txt`
   的 digits 段是 `frame_000000` ⇒ `isdigit()` 为假 ⇒ **整份被跳过** ⇒
   若直接用数据集标签目录，所有帧都被判成"无真值" ⇒ **TP=0 / FN=0，指标全 0 而不报任何错**。

⇒ 修法：新增 `manu/data/build_train_labels_pos.py`（按位置重编号 + 与原始真值逐帧对账，
本次 13,918 位置 / 12,120 有真值帧 / 12,120 GT 框 / **对账不一致 0 处**）。

## 5. 其他实测事实

- **特征确实不用重算**：官方特征 JPG 已在盘上，`--features-root` 直接读。
  （唯一例外：若要 **stride=1 全密度**，必须现场重算 GMC —— 实测 **≈0.25 s/帧**，
  69,589 帧 × 2 权重 ⇒ 关键路径单条 002001 需 **≈75 分钟**，已按用户指示放弃。）
- **现场 GMC 与官方落盘特征等价性已实测**（`manu/data/verify_onthefly_features.py`，
  产物 `runs/train_fit_diag/onthefly_feature_check.json`）：
  **GMC 派生通道 ch1/ch2 与官方落盘逐像素一致**（|Δ|>8 像素占比 0.09% / 0.25%，JPEG 噪声量级）；
  ch0 的差异是源图 double-JPEG 噪声（与权重无关，两臂同源）⇒ stride=1 这条路**口径一致、可行**。
- **旧 train 缓存不可复用**：`runs/ms_cache_v2` / `runs/ms_cache_nop0` 的 002001 是 **18,014 帧**
  （Oct 9 旧特征、stride=1），与当前 official train 的 **3,603 帧** 不是一回事。

## 6. 可复制命令（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，根目录 `/tmp/pycharm_project_10ae9e2e`）

```bash
# ① 位置索引化 train 标签根 + 逐帧对账（必须先跑）
python manu/data/build_train_labels_pos.py \
  --dataset-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official \
  --orig-labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --out-root runs/train_fit_diag/labels_train_pos \
  --out-json runs/train_fit_diag/labels_train_pos_manifest.json

# ② 双权重 train 对照（缓存并行构建，9 分片 = 9 条序列各一片；实测两个权重各 ~175-186s）
python manu/training/train_fit_diagnostic.py \
  --weights baseline=runs/optuna_p0_nas/trial_0474/weights/best.pt \
            ft=runs/mixed_ft_route_b/RC1/epoch3.pt \
  --source features \
  --features-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train \
  --labels-root runs/train_fit_diag/labels_train_pos \
  --sequences VIDEO00002_19700101_001415,VIDEO00005_19700101_002959,VIDEO00008_19700101_002001,VIDEO00009_19700101_003002,VIDEO00010_19700101_001550,VIDEO00011_19700101_001810,VIDEO00028_19700101_013631,VIDEO00030_19700101_014144,VIDEO00031_19700101_014227 \
  --out-dir runs/train_fit_diag_s5 --gpus auto --shards 9

# ③ 现场 GMC 等价性核验
python manu/data/verify_onthefly_features.py \
  --raw-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
  --features-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train \
  --sequence VIDEO00002_19700101_001415 --n-frames 200 --stride 5 --warmup 60 \
  --out-json runs/train_fit_diag/onthefly_feature_check.json
```

**产物**：服务器 `runs/train_fit_diag_s5/train_fit_diagnostic.{md,csv,json}`、
`runs/train_fit_diag/labels_train_pos_manifest.json`、`runs/train_fit_diag/onthefly_feature_check.json`；
本地副本 `manu/docs/route_b/train_fit_diag/`。
**新增工具**：`manu/data/build_train_labels_pos.py`、`manu/data/verify_onthefly_features.py`、
`manu/training/train_fit_diagnostic.py`、**`manu/training/verify_train_fit_report.py`**
（交叉校验：汇总表 vs 两份 `train_eval.log` 原始行 —— 整数 TP/FP/FN 精确比对、
R/P/F1 由整数重算后 1e-9 比对、打印值 1e-4 宽松比对，并**由原始行重算三个诊断信号**；
本轮实测 **PASS**；注意该脚本自身踩过两个坑：日志打印精度 ≠ 报告全精度、delta 方向必须是 ft−base）。
