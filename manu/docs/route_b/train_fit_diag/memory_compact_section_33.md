> ⚠️ **本文件不是独立 memory 章节**，它是 `memory_compact.md` **§33** 的**逐字源文**（2026-10-21 由
> `manu/docs/route_b/train_fit_diag/memory_compact_section_33.md` 原样追加进 `memory_compact.md` 末尾）。保留此处仅为溯源用。

## §33 🔬 Train 划分拟合度诊断：Ep3 拟合健康、不过拟合、增益 85.4% 泛化（2026-10-21）

- ✅ **口径**：与 §四十三 门禁**完全一致**（`multiscale_heatmap_cache.py` → `system_sota_longquan.py --no-pipeline --all-seqs`），
  **只换权重**；数据 = 龙泉山 **train 划分 9 条 / 13,918 帧 / 12,120 GT**（**正是模型训练时逐帧看过的样本**）；
  特征用官方落盘特征 JPG（**零重算**）。
- **整体对照**：F1 **0.933396 → 0.977410（+0.044014）**｜Recall **0.896700 → 0.969224**｜
  Precision **0.973225 → 0.985735**｜TP 10,868→11,747｜**FP 299→170（−43.1%）**｜**FN 1,252→373（−70.2%）**
  ⇒ **Recall 与 Precision 同涨、FP 反降** ⇒ **不是"放宽门限换召回"，是真变强**；逐序列 **9 条 F1 全部上升**。
- ✅ **信号1 欠拟合 ⇒ 不欠拟合**：Recall 落在 96.92%（健康带 92%~96% 上沿）；
  **002001** FN **606→229（−62.2%）**、R 81.24%→92.91%、F1 +8.17pp；
  **003002** FN **65→12（−81.5%）**、R 51.13%→**90.98%**、F1 **+29.15pp**。
  ⇒ **「全图 1:3.09 混训下大背景稀释小目标梯度」被证伪**，模型容量不是瓶颈。
- ✅ **信号2 过拟合 ⇒ 不过拟合**：train F1 **97.74%**（远未到 99.5%），FP 170 未趋零
  （**2 条序列 FP 已压到 0**：001810、013631；另 003002=2、001415=8）⇒ 泛化余量充足。
  ⚠️ train↔blind 落差略扩：9.61pp → **10.25pp**。
- ⚠️ **信号3 顽固误检**：train 内 ΔFP = 002001 **−103**、001415 **−27**、001810 −6、013631 −5、003002 −5、
  014144 0、014227 +3、001550 +4、**002959 +10**。
  🔴 **002959 是训练集里见过的序列，Ep3 在它上面 FP 反而 45→55**，与 val 侧未见过的 `000703`（ΔFP +25）**同构**
  ⇒ 处方升级为 **000703 + 002959 同形态负样本一起补**（唯一同时打中两处的处方）。
- ⭐ **拟合度→泛化度 转化率（本轮最有价值的派生结论）**：
  ΔF1：train +0.044014 / blind 001324 +0.037593 ⇒ **泛化率 85.4%**（14.6% 是纯训练集拟合）；
  ΔRecall：+0.072525 / +0.054200 ⇒ **74.7%**；ΔPrecision：+0.012510 / +0.013333 ⇒ **106.6%（完全泛化）**。
  ⇒ **下一轮杠杆 = 召回的泛化，不是继续拟合** ⇒ 局部切片放大（方案 A）对症。
  ⇒ 残余缺口集中在 **002001**（占 Ep3 train FN 373 的 **61.4%**、FP 170 的 **46.5%**）⇒ 切片应针对 002001 这类"暗刚体"。
- ⚠️ **本轮修掉的两个静默坑（train 划分专属）**：
  ① **stride≠1 帧号错位（致命）**：`multiscale_heatmap_cache.py:403` 写的是 `frame_index = start + index`
  = **排序位置**而非原始帧号；val 因 stride=1 才天然对齐，**train 是 stride=5** ⇒ 直接喂原始真值目录会**整体错位且不报错**。
  ② **命名陷阱（静默清零）**：`eval_multiscale_cache_th.py:100` 的 `load_ground_truth` 只认 `{seq}__{digits}.txt`
  且 digits 必须纯数字；数据集自带 `labels/train/{seq}__frame_000000.txt` 的 digits 段是 `frame_000000`
  ⇒ 被**整份跳过** ⇒ 直接用会得到 **TP=0/FN=0 的静默错误结果**。
  ⇒ 修法：新增 `manu/data/build_train_labels_pos.py`（位置重编号 + 逐帧对账，本次 **对账不一致 0 处**）。
- **其他实测**：**特征确实不用重算**；若要 stride=1 全密度则须现场重算 GMC（实测 **≈0.25 s/帧**，
  69,589 帧 × 2 权重 ⇒ 关键路径 002001 单条 ≈**75 分钟**，已按用户指示放弃）；
  现场 GMC 与官方落盘**等价性已实测**（GMC 派生通道 ch1/ch2 逐像素一致，|Δ|>8 像素占比 0.09%/0.25%，
  ch0 差异是源图 double-JPEG 噪声且与权重无关）⇒ stride=1 这条路口径一致、可行；
  旧 `runs/ms_cache_v2` 的 002001 是 **18,014 帧**（Oct 9 旧特征）**不可复用**。
- **新增工具**：`manu/data/build_train_labels_pos.py`、`manu/data/verify_onthefly_features.py`、
  `manu/training/train_fit_diagnostic.py`、**`manu/training/verify_train_fit_report.py`**
  （交叉校验汇总表 vs 两份原始日志，并**由原始行重算三个诊断信号**；本轮实测 **PASS**）；
  **产物** `runs/train_fit_diag_s5/train_fit_diagnostic.{md,csv,json}`，
  本地副本 `manu/docs/route_b/train_fit_diag/`；详细分析 `manu/docs/route_b/train_fit_diag/train_fit_diagnostic_analysis.md`。
  复现命令见 `memory.md` §四十七 · 6。
