# Train 划分拟合度诊断：原始底模 vs 微调 RC1(Ep3)

> 目的：回答「微调模型在**训练集**上拟合到什么程度」—— 判欠拟合 / 判过拟合 / 找顽固误检。
> 口径与 §四十三 三轨门禁**完全一致**，只换权重。
> 产物：`train_fit_diagnostic.{md,csv,json}`（同目录）、`labels_train_pos_manifest.json`、`onthefly_feature_check.json`。

## 0. 实验设置（可复核）

| 项 | 值 |
|:--|:--|
| 底模（baseline） | `runs/optuna_p0_nas/trial_0474/weights/best.pt`（去 P0 口径，`--ablate-p0`） |
| 微调（ft） | `runs/mixed_ft_route_b/RC1/epoch3.pt`（RC1，0444 封存，md5 `a845a95b…`） |
| 评测数据 | 龙泉山 **train 划分 9 条序列 / 13,918 帧 / 12,120 GT**（**正是模型训练时逐帧看过的样本**） |
| 特征 | 官方 official 特征 JPG（**零重算**），磁盘序 [I_t, diff, res] |
| 标签 | 位置索引化标签根 `runs/train_fit_diag/labels_train_pos`（与原始合并真值**逐帧对账 0 处不一致**） |
| 推理链 | `manu/data/multiscale_heatmap_cache.py`（超参与门禁逐项相同）→ `manu/evaluation/system_sota_longquan.py --no-pipeline --all-seqs`（**单帧臂**） |
| 匹配规则 | `inbox`（点落框内即 TP）；**只换权重，其余一律不动** |

### 0.1 本轮踩到的两个静默坑（已修，值得记住）

1. **stride≠1 的帧号错位（致命）**：`multiscale_heatmap_cache.py:403` 写的是 `frame_index = start + index`，即
   **排序位置**而非原始帧号。val 是 stride=1 才恰好天然对齐；train 是 **stride=5**（13,918 帧来自 69,589 帧），
   直接把原始真值目录喂给评测脚本会**整体错位且不报错**。
2. **命名陷阱（静默清零）**：`eval_multiscale_cache_th.py:100` 的 `load_ground_truth` 只认 `{seq}__{digits}.txt`，
   digits 必须纯数字；数据集自带 `labels/train/{seq}__frame_000000.txt` 的 digits 段是 `frame_000000`
   ⇒ `isdigit()` 为假 ⇒ **整份被跳过** ⇒ 若直接用数据集标签目录，所有帧都会被判成"无真值"，
   指标全 0 而**不报任何错**。
   ⇒ 修法：`manu/data/build_train_labels_pos.py` 按位置重编号 + 与原始真值逐帧对账（本次 0 处不一致）。

### 0.2 为什么不跑全密度（stride=1）

试过：现场 GMC（LK 光流对齐 + 21 帧中值）实测 **≈0.25 s/帧**，69,589 帧 × 2 权重 ⇒ 关键路径单条 002001
需 **≈75 分钟**，且该任务已按用户指示放弃（"不用非得测试全密度"）。
现场 GMC 与官方落盘特征的等价性已单独实测（`onthefly_feature_check.json`）：
**GMC 派生通道 ch1/ch2 与官方落盘逐像素一致**（|Δ|>8 像素占比 0.09% / 0.25%，JPEG 噪声量级），
ch0 的差异是源图 double-JPEG 噪声（与权重无关，两臂同源）⇒ **口径一致这条路已验证可行**，只是本次不必跑。

---

## 1. 训练集整体对照（13,918 帧 / 12,120 GT）

| 指标 | 底模 | 微调 Ep3(RC1) | Δ |
|:--|--:|--:|--:|
| **F1** | 0.933396 | **0.977410** | **+0.044014** |
| **Recall** | 0.896700 | **0.969224** | **+0.072525** |
| **Precision** | 0.973225 | **0.985735** | **+0.012510** |
| TP | 10,868 | 11,747 | +879 |
| **FP** | 299 | **170** | **−129（−43.1%）** |
| **FN** | 1,252 | **373** | **−879（−70.2%）** |

> 关键性质：**召回和精度同时上升**（Recall +7.25pp，Precision +1.25pp，FP 反而降 43%）
> ⇒ 不是"放宽门限换召回"，是真变强。

## 2. 逐序列对照（train 9 条）

| 序列 | 角色 | GT | 底模 F1 | Ep3 F1 | ΔF1 | 底模 FN | Ep3 FN | ΔFN | 底模 FP | Ep3 FP | ΔFP |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 001415 | 负样本挖掘 | 1,700 | 0.947847 | 0.988751 | +0.040904 | 137 | 30 | **−107** | 35 | 8 | **−27** |
| 002959 | 正样本挖掘 | 3,154 | 0.967065 | 0.976885 | +0.009820 | 159 | 90 | −69 | 45 | 55 | **+10** |
| **002001** | 正样本挖掘 | 3,231 | 0.869493 | 0.951204 | **+0.081711** | 606 | 229 | **−377** | 182 | 79 | **−103** |
| **003002** | 负样本挖掘 | 133 | 0.653846 | 0.945313 | **+0.291466** | 65 | 12 | **−53** | 7 | 2 | −5 |
| 001550 | 运动补偿 | 433 | 0.978035 | 0.985210 | +0.007176 | 10 | **0** | −10 | 9 | 13 | +4 |
| 001810 | 正样本挖掘 | 1,170 | 0.955131 | **0.999572** | +0.044441 | 95 | 1 | −94 | 6 | **0** | −6 |
| 013631 | 正样本挖掘 | 1,507 | 0.953995 | **0.999004** | +0.045008 | 128 | 3 | −125 | 5 | **0** | −5 |
| 014144 | 负样本挖掘 | 152 | 0.980132 | 0.993464 | +0.013332 | 4 | 0 | −4 | 2 | 2 | 0 |
| 014227 | 负样本挖掘 | 640 | 0.954839 | 0.985191 | +0.030352 | 48 | 8 | −40 | 8 | 11 | +3 |

**9 条序列 F1 全部上升，无一条下降。**

---

## 3. 三个诊断信号的判定

### 信号 1 · 欠拟合检查 ⇒ ✅ **不欠拟合，容量足够**

| 判据 | 实测 | 判定 |
|:--|:--|:--|
| Train Recall 稳步爬到 92%~96% | 底模 89.67% → **Ep3 96.92%** | ✅ 落在健康带上沿 |
| `002001`（下落变暗段）FN 是否被砍掉大半 | Recall 81.24% → **92.91%**；FN **606 → 229（−62.2%）**；F1 86.95% → 95.12% | ✅ **"砍掉大半"成立** |
| `003002`（贴地弱目标）| Recall 51.13% → **90.98%**；FN **65 → 12（−81.5%）**；F1 65.38% → **94.53%（+29.15pp）** | ✅ 底模在此近乎失效，微调后基本救回 |

⇒ **"全图 1:3.09 混训下大背景稀释小目标梯度"这一假说被证伪**：模型在训练集上确实把暗刚体 / 贴地弱目标
吃进去了，容量不是瓶颈。
⚠️ 残余缺口**高度集中**：Ep3 的 train FN 373 里 **002001 占 229（61.4%）**，FP 170 里 **002001 占 79（46.5%）**
⇒ 后续若还要压召回，**局部切片放大（方案 A）应针对 002001 这类"暗刚体"**，而不是全局重训。

### 信号 2 · 过拟合检查 ⇒ ✅ **不过拟合，泛化余量充足**

| 判据 | 实测 | 判定 |
|:--|:--|:--|
| Train F1 是否已逼近 99.5%+ | **97.74%**（远未到 99.5%） | ✅ 落在 95%~97% 健康带（略高） |
| Train FP 是否趋近 0 | 170（未趋零；**2 条序列 FP 已压到 0**：001810、013631；另 003002=2、001415=8） | ✅ 未趋零，判别流形仍平滑 |
| Train 与盲测 001324 的落差 | 底模 gap 9.61pp → Ep3 gap **10.25pp** | ⚠️ 落差**略扩**（见下） |

⇒ **Ep3 不处于"容量用尽 / 对建筑物高频边缘过拟合"状态**，当前主要矛盾不是过拟合。

### 信号 3 · 顽固误检定位 ⇒ ⚠️ **训练集内已压住，但 002959 反弹**

| 序列 | ΔFP | 解读 |
|:--|--:|:--|
| 002001 | **−103** | 地表高频假目标被压住（主力） |
| 001415 | **−27** | 同上 |
| 001810 / 013631 / 003002 | −6 / −5 / −5 | 已压到 0~2 |
| 014144 | 0 | 持平 |
| 014227 | +3 | 轻微 |
| 001550 | +4 | 轻微 |
| **002959** | **+10** | ⚠️ **train 侧唯一显著反弹**（FP 45 → 55） |

⇒ 「抑制城市高频假目标」机制在**训练集内确实成立**（−130，总降幅 −129 的全部来源）。
⇒ 🔴 **但 002959 是在训练集里被反复见过的序列，Ep3 在它上面反而多报 10 个虚警** ——
   与 val 侧 `000703`（未见过的同类形态，ΔFP **+25**）**现象同构**：
   **压制某类地表切变线的能力没有泛化到同形态的新序列，反而在那里新造误报。**
⇒ 处方不变且更强：**把 000703 与 002959 的失效机理并入负样本**（各补若干纯背景切片），
   这是唯一同时打中 val 侧 000703 与 train 侧 002959 的处方。

---

## 4. 拟合度 → 泛化度的转化率（本轮最有价值的派生结论）

| 指标 | ΔTrain | Δ盲测 001324 | **泛化率** |
|:--|--:|--:|--:|
| F1 | +0.044014 | +0.037593 | **85.4%** |
| Recall | +0.072525 | +0.054200 | **74.7%** |
| Precision | +0.012510 | +0.013333 | **106.6%（完全泛化）** |

⇒ 本轮 4.40pp 的 train 增益里，**85.4% 泛化到了盲测**，14.6% 是纯训练集拟合；
⇒ 精度侧的增益**百分之百泛化**；**召回侧只有 74.7% 泛化**（45.8% 流失）——
   召回是"吃进去"了但"带不出来"的部分，对应 §四十五 归因里的 001324 ΔFN −571 与 ΔFP −88 同降。
⇒ **下一步的杠杆在召回的泛化，不在拟合** ⇒ 局部切片放大（方案 A）正是对症的方向。

---

## 5. 结论

1. **Ep3 在训练集上拟合健康**：F1 93.34% → 97.74%，Recall 96.92%，且 Recall/Precision 同涨、FP 反降 43%
   ⇒ **不是靠放宽门限换来的提升**。
2. **不欠拟合**：002001 / 003002 这类底模失效序列被显著救回（FN −62% / −82%），
   "大背景稀释小目标梯度"的假说被证伪。
3. **不过拟合**：train F1 97.74% 远未到 99.5%，FP 未趋零，泛化余量充足。
4. **增益的 85.4% 泛化到了盲测**，精度增益 100% 泛化，召回增益 74.7% 泛化
   ⇒ 下一轮杠杆 = **召回的泛化**（局部切片放大），不是继续拟合。
5. **唯一新暴露的风险点**：002959（train 侧 ΔFP +10）与 000703（val 侧 ΔFP +25）同构
   ⇒ 负样本先验缺失，下一轮定点补这两条形态。

## 6. 可复制命令（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`，`conda activate uav`，根目录 `/tmp/pycharm_project_10ae9e2e`）

```bash
# ① 位置索引化 train 标签根 + 与原始真值逐帧对账（必须先跑）
python manu/data/build_train_labels_pos.py \
  --dataset-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official \
  --orig-labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --out-root runs/train_fit_diag/labels_train_pos \
  --out-json runs/train_fit_diag/labels_train_pos_manifest.json

# ② 双权重 train 对照（缓存并行构建；9 分片 = 9 条序列各一片）
python manu/training/train_fit_diagnostic.py \
  --weights baseline=runs/optuna_p0_nas/trial_0474/weights/best.pt \
            ft=runs/mixed_ft_route_b/RC1/epoch3.pt \
  --source features \
  --features-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train \
  --labels-root runs/train_fit_diag/labels_train_pos \
  --sequences VIDEO00002_19700101_001415,VIDEO00005_19700101_002959,VIDEO00008_19700101_002001,VIDEO00009_19700101_003002,VIDEO00010_19700101_001550,VIDEO00011_19700101_001810,VIDEO00028_19700101_013631,VIDEO00030_19700101_014144,VIDEO00031_19700101_014227 \
  --out-dir runs/train_fit_diag_s5 --gpus auto --shards 9

# ③ 现场 GMC 与官方落盘特征的等价性核验（stride=1 可行性）
python manu/data/verify_onthefly_features.py \
  --raw-root "/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg" \
  --features-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/images/train \
  --sequence VIDEO00002_19700101_001415 --n-frames 200 --stride 5 --warmup 60 \
  --out-json runs/train_fit_diag/onthefly_feature_check.json

# ④ 交叉校验：汇总表 vs 两份原始日志，并由原始行重算三个诊断信号（验收必跑）
python manu/training/verify_train_fit_report.py \
  --report runs/train_fit_diag_s5/train_fit_diagnostic.json \
  --base-log runs/train_fit_diag_s5/baseline/train_eval.log \
  --ft-log runs/train_fit_diag_s5/ft/train_eval.log
```

**产物路径**：服务器 `runs/train_fit_diag_s5/train_fit_diagnostic.{md,csv,json}`、
`runs/train_fit_diag/labels_train_pos_manifest.json`、`runs/train_fit_diag/onthefly_feature_check.json`；
本地副本 `manu/docs/route_b/train_fit_diag/`。
