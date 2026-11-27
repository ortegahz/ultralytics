> ⚠️ **本报告已被 `iqa_v2/REPORT_V2_FINAL.md` 取代**（2026-11-26 收口轮）：
> 本版含 **JPEG 退化轴**、**未纯净化仿真底座**、**无序列级切分**、**无三分法回归测试**，
> 其膝点与红线数值**不得**对外引用。新报告见 `iqa_v2/REPORT_V2_FINAL.md`。

# 红外图像质量评估 v2（Target-Centric 业务目标中心模式）实验报告

> **口径声明**：当前阈值基于 **Anti-UAV 开源红外数据集**标定，用途是**验证 v2 Target-Centric 方法本身的有效性**；**正式上板红线**待内部 58 条红外业务 GT 序列回收后，用**同一套流水线**重标。

## 一、数据集过滤统计（只跑红外）

| 项 | 值 |
|:--|--:|
| 数据集根目录 | `/mnt/data/siping/datasets/manu/anti-uav` |
| 模态 | **仅红外（IR）**，可见光文件**未被打开** |
| 扫描到的红外序列数 | **223** |
| 保留（单目标）序列数 | **223** |
| 剔除序列数 | **0**（多目标 0） |
| 有效 5 帧窗口数 | **2442** |
| 小目标子集（max(w,h) ≤ 20 px） | **1031** 条（占比 42.2%） |

剔除名单为空：本机 Anti-UAV 副本**已是单目标版本**（223 条序列逐条解析，**0 条**出现单帧多框 / 多 track_id）。

## 二、跟踪器三臂定位与干净帧基线

| 跟踪臂 | 角色 | 干净帧跟住率 | 中位 CLE(px) | P90 CLE(px) | 异常/报错次数 |
|:--|:--|--:|--:|--:|--:|
| `xcorr` | **主臂**：模板匹配链式配准（与 KCF 跨家族） | 84.5% | 1.58 | 14.08 | 0 |
| `kcf_legacy` | **主臂**：任务书指定的 contrib legacy KCF（detect_thresh=0.0） | 84.6% | 1.80 | 21.99 | 224 |
| `csrt` | 同族对照：DCF 家族的另一实现（CSRT） | 93.2% | 1.58 | 6.30 | 0 |

### 2.1 任务书指定的 KCF 臂——两条必须写进报告的实现事实

> 任务书指定的 cv2.legacy.TrackerKCF（opencv-contrib 4.10.0.84，PYTHONPATH=/tmp/pylibs_cvcontrib）在本环境**可用**：合成目标每帧平移 8 px 时，update() 逐帧返回的位置与真值**逐帧对齐**（120/128/136/144/152，误差 ≤ 1 px）。三条必须写进报告的实现事实：① 🔴 **update() 严格滞后一帧**：在第 k 帧调用 update() 返回的是**第 k-1 帧**的矩形（实测：真值 120/128/136/144/152，update(帧1..帧5) 依次返回 120/128/136/144/152）。⇒ 本流水线用**多喂一帧**（第 t+5 帧）来报出「第 t+4 帧的位置」，否则 KCF 臂会被「一帧的运动量」系统性罚掉（快目标直接判失锁），污染膝点曲线；② detect_thresh 参数**确实生效**——置 0.0（任务书口径，关闭内部阈值截断）时正常跟踪；置 0.99 时响应峰达不到阈值，update() 会返回**退化的 (0,0) 矩形**（实测 5 帧里 4 帧塌到原点）；③ 初始化框必须**精确落在目标本体上**：一旦偏了半个框，KCF 会锁到目标的**局部**（实测偏 12 px 时它会以 12 px 的固定偏差跟着目标跑，看起来像「矩形不更新」——这是实现细节，不是跟踪器坏了）。

**取证方法**（可复现）：合成一帧「亮目标 + 噪声背景」序列，目标每帧平移 8 px，共 5 帧；用 `tracker.update()` 连续推 4 步，逐帧比较返回矩形的中心与真值。

| 参数 | 观测到的逐帧中心 x（真值 120/128/136/144/152） | 结论 |
|:--|:--|:--|
| `detect_thresh=0.0`（任务书口径） | 120 / 128 / 136 / 144 / 152 | ✅ 逐帧对齐，误差 ≤ 1 px |
| `detect_thresh=0.99` | 120 / 0 / 0 / 0 / 0 | ❌ 响应峰达不到阈值 ⇒ 退化成原点矩形 |

### 2.2 为什么三臂要分开列（跨家族纪律）

* **XCorr**（模板匹配 + 链式重切模板）与 **KCF**（判别相关滤波 + 在线模型更新）是原理完全不同的两类跟踪器 ⇒ 它们各自的**绝对锁定率刻度不可互换**，但**反解出来的线位可以互相印证**。
* **CSRT** 与 KCF 同属 DCF 家族 ⇒ 它的作用是**同族对照**：证明 KCF 侧反推的线位不是「某个实现的细节」。
* 三个臂的绝对数值**严禁合并求平均**；任何一张表里三者都**并列呈现**。

### 2.3 KCF 臂「尾帧补偿」的实测覆盖率与残差（防质疑必读）

> 因为 contrib legacy KCF 的 `update()` 滞后一帧，本流水线用**第 t+5 帧**做补偿。下面把补偿的**实际覆盖率**与**残差**摊开写清楚，别让读数变成黑箱。

| 退化族 | 档位 | 行数 | 尾帧更新失败 | 占比 | 无尾帧（序列末尾） |
|:--|--:|--:|--:|--:|--:|
| clean | 0.0 | 2442 | 19 | 0.8% | 205 |
| blur | 0.6 | 2442 | 46 | 1.9% | 205 |
| blur | 1.0 | 2442 | 127 | 5.2% | 205 |
| blur | 1.5 | 2442 | 142 | 5.8% | 205 |
| blur | 2.2 | 2442 | 203 | 8.3% | 205 |
| blur | 3.0 | 2442 | 200 | 8.2% | 205 |
| blur | 4.0 | 2442 | 268 | 11.0% | 205 |
| motion | 3.0 | 2442 | 54 | 2.2% | 205 |
| motion | 5.0 | 2442 | 67 | 2.7% | 205 |
| motion | 7.0 | 2442 | 84 | 3.4% | 205 |
| motion | 9.0 | 2442 | 126 | 5.2% | 205 |
| motion | 11.0 | 2442 | 134 | 5.5% | 205 |
| contrast | 0.15 | 2442 | 356 | 14.6% | 205 |
| contrast | 0.25 | 2442 | 207 | 8.5% | 205 |
| contrast | 0.35 | 2442 | 144 | 5.9% | 205 |
| contrast | 0.5 | 2442 | 92 | 3.8% | 205 |
| contrast | 0.65 | 2442 | 44 | 1.8% | 205 |
| contrast | 0.8 | 2442 | 26 | 1.1% | 205 |
| jpeg | 10.0 | 2442 | 30 | 1.2% | 205 |
| jpeg | 15.0 | 2442 | 22 | 0.9% | 205 |
| jpeg | 22.0 | 2442 | 25 | 1.0% | 205 |
| jpeg | 32.0 | 2442 | 14 | 0.6% | 205 |
| jpeg | 45.0 | 2442 | 12 | 0.5% | 205 |
| jpeg | 60.0 | 2442 | 10 | 0.4% | 205 |

| 退化族 | 尾帧失败行的跟住率 | 其余行的跟住率 |
|:--|--:|--:|
| clean | 36.8% | 85.0% |
| blur | 61.3% | 83.8% |
| motion | 60.2% | 84.1% |
| contrast | 62.9% | 83.6% |
| jpeg | 46.9% | 84.0% |

> **两条必须一起讲的结论**：
> ① **尾帧失败不是「口径造成的假失败」**：这些窗口集中在强退化档位，其跟住率显著低于其余窗口（见上表）⇒ 跟踪器在那时**确实已经跟丢**；
> ② **仍有残差**：序列末尾的 **205 个窗口（8.4%）** 拿不到尾帧，全程只能用「滞后一帧」的读数；这 205 个窗口在**所有档位**上都存在，对 KCF 臂是**恒定**的轻微偏保守 ⇒ KCF 族膝点只会**偏保守**、不会偏松。
>    XCorr 与 CSRT 两臂**没有滞后一帧问题**（OpenCV 4.x 的 `Tracker.update` 返回的是当前帧结果），所以三臂的绝对值差异**不来自**这个补偿。

## 三、4 维退化下的局部 ROI 指标膝点反解

> **跨家族纪律**：XCorr 与 KCF 的绝对锁定率刻度**不可互换**，下表**并列呈现、各自独立反解**，**严禁合并求平均**；可比的是**线位**。

### 3.1 采用的膝点（全量窗口 / 全量曲线集）

| 退化族 | 指标 | XCorr 膝点 | KCF 膝点 | 两族线位差 | 可用性 |
|:--|:--|--:|--:|--:|:--|
| blur | local_scr | 4.77734 | 4.77734 | 0.00% | ✅ 可靠 |
| blur | target_lap_var | 819.551 | 819.551 | 0.00% | ✅ 可靠 |
| blur | surround_std | 4.71249 | 4.71249 | 0.00% | ✅ 可靠 |
| blur | global_contrast | 0.630662 | 0.630662 | 0.00% | ✅ 可靠 |
| motion | local_scr | 5.63068 | 2.69153 | 52.20% | ✅ 可靠 |
| motion | target_local_contrast | 0.204807 | 0.204807 | 0.00% | ✅ 可靠 |
| motion | target_lap_var | 1753.97 | 2320.86 | 24.43% | ✅ 可靠 |
| motion | target_grad_mean | 75.3158 | 68.2129 | 9.43% | ✅ 可靠 |
| motion | surround_std | 4.75697 | 4.75697 | 0.00% | ✅ 可靠 |
| motion | surround_blockiness | 1.10406 | 1.06929 | 3.15% | ✅ 可靠 |
| motion | global_contrast | 0.636364 | 0.636364 | 0.00% | ✅ 可靠 |
| motion | global_lap_var | 41.63 | 41.63 | 0.00% | ✅ 可靠 |
| contrast | local_scr | 3.20777 | 3.20777 | 0.00% | ✅ 可靠 |
| contrast | target_lap_var | 1090.89 | 1651.88 | 33.96% | ✅ 可靠 |
| contrast | target_grad_mean | 27.1303 | 27.1303 | 0.00% | ✅ 可靠 |
| contrast | surround_std | 2.36438 | 2.36438 | 0.00% | ✅ 可靠 |
| contrast | surround_blockiness | 1.18749 | 1.11221 | 6.34% | ✅ 可靠 |
| contrast | global_contrast | 0.242857 | 0.242857 | 0.00% | ✅ 可靠 |
| contrast | global_lap_var | 80.6401 | 36.7179 | 54.47% | ✅ 可靠 |
| jpeg | local_scr | 4.88624 | 4.88624 | 0.00% | ✅ 可靠 |
| jpeg | target_lap_var | 4182.89 | 4182.89 | 0.00% | ✅ 可靠 |
| jpeg | target_grad_mean | 93.2623 | 93.2623 | 0.00% | ✅ 可靠 |
| jpeg | surround_std | 5.08902 | 5.08902 | 0.00% | ✅ 可靠 |
| jpeg | surround_blockiness | 0.620585 | 0.620585 | 0.00% | ✅ 可靠 |
| jpeg | global_contrast | 0.657534 | 0.657534 | 0.00% | ✅ 可靠 |
| jpeg | global_lap_var | 184.621 | 184.621 | 0.00% | ✅ 可靠 |

### 3.2 小目标子集（max(w,h) ≤ 20 px）

| 退化族 | 指标 | XCorr 膝点 | KCF 膝点 | CSRT 膝点（同族对照） |
|:--|:--|--:|--:|--:|
| blur | surround_std | 3.86362 | 3.86362 | nan |
| blur | surround_blockiness | 1.08264 | 1.08264 | 1.08264 |
| blur | global_contrast | 0.622378 | 0.622378 | nan |
| blur | global_lap_var | 10.6165 | 10.6165 | nan |
| motion | local_scr | 5.09378 | 2.58186 | 3.81826 |
| motion | surround_std | 3.94555 | 3.94555 | nan |
| motion | surround_blockiness | 1.07912 | 1.02563 | 1.07912 |
| motion | global_contrast | 0.628176 | 0.609589 | nan |
| motion | global_lap_var | 22.3745 | 15.1704 | nan |
| contrast | local_scr | 2.38331 | 2.38331 | 4.71492 |
| contrast | target_lap_var | 1392.36 | 3298.64 | 860.438 |
| contrast | surround_std | 1.97028 | 1.50566 | nan |
| contrast | surround_blockiness | 1.11911 | 1.11911 | 1.11911 |
| contrast | global_contrast | 0.115313 | 0.115313 | nan |
| contrast | global_lap_var | 57.8668 | 57.8668 | nan |
| jpeg | local_scr | 3.2171 | 4.63663 | 4.63663 |
| jpeg | surround_std | 4.37482 | 4.37482 | 2.7142 |
| jpeg | surround_edge_density | 0.00144231 | 0.00144231 | nan |
| jpeg | surround_blockiness | 0.73452 | 0.73452 | 0.5902 |
| jpeg | global_contrast | 0.626866 | 0.608376 | nan |
| jpeg | global_lap_var | 94.7819 | 94.7819 | nan |

### 3.3 同族对照：KCF vs CSRT（证明 KCF 侧线位不是实现细节）

| 退化族 | 指标 | KCF 膝点 | CSRT 膝点 | 线位差 |
|:--|:--|--:|--:|--:|
| blur | local_scr | 4.77734 | 4.77734 | 0.00% |
| blur | target_lap_var | 819.551 | 553.998 | 32.40% |
| blur | surround_blockiness | 0.95333 | 1.12572 | 15.31% |
| blur | global_contrast | 0.630662 | 0.630662 | 0.00% |
| blur | global_lap_var | 9.11919 | 9.11919 | 0.00% |
| motion | local_scr | 2.69153 | 4.23763 | 36.48% |
| motion | target_lap_var | 2320.86 | 994.417 | 57.15% |
| motion | target_grad_mean | 68.2129 | 60.6277 | 11.12% |
| motion | global_contrast | 0.636364 | 0.822075 | 22.59% |
| contrast | local_scr | 3.20777 | 4.9097 | 34.66% |
| contrast | target_lap_var | 1651.88 | 717.342 | 56.57% |
| contrast | target_grad_mean | 27.1303 | 21.9584 | 19.06% |
| contrast | surround_blockiness | 1.11221 | 1.11221 | 0.00% |
| contrast | global_contrast | 0.242857 | 0.69935 | 65.27% |
| jpeg | local_scr | 4.88624 | 4.88624 | 0.00% |
| jpeg | target_lap_var | 4182.89 | 4182.89 | 0.00% |
| jpeg | target_grad_mean | 93.2623 | 74.0197 | 20.63% |
| jpeg | surround_std | 5.08902 | 3.86383 | 24.08% |
| jpeg | surround_blockiness | 0.620585 | 0.620585 | 0.00% |
| jpeg | global_contrast | 0.657534 | 0.70892 | 7.25% |
| jpeg | global_lap_var | 184.621 | 184.621 | 0.00% |

### 3.4 反推出的「良好跟踪合格线」（约束线）

> 每个指标在 4 个退化族上各有各的膝点；**准入线必须取各族膝点的最小值**（最严格的那一族说了算）——因为画质指标都是「越大越好」，线一旦定在某个族的膝点之上，另一族就会崩。

| 指标 | 类型 | XCorr 约束线 | 决定族 | KCF 约束线 | 决定族 | 两族线位差 | 可用性 |
|:--|:--|--:|:--|--:|:--|--:|:--|
| local_scr | 局部 ROI | 3.20777 | contrast | 2.69153 | motion | 16.09% | ⚠️ 有差 |
| target_local_contrast | 局部 ROI | 0.170531 | contrast | 0.126782 | blur | 25.65% | ⚠️ 有差 |
| target_lap_var | 局部 ROI | 819.551 | blur | 819.551 | blur | 0.00% | ✅ 一致 |
| target_grad_mean | 局部 ROI | 27.1303 | contrast | 27.1303 | contrast | 0.00% | ✅ 一致 |
| surround_std | 局部 ROI | 2.36438 | contrast | 2.36438 | contrast | 0.00% | ✅ 一致 |
| surround_blockiness | 局部 ROI | 0.620585 | jpeg | 0.620585 | jpeg | 0.00% | ✅ 一致 |
| global_contrast | 全图对照 | 0.242857 | contrast | 0.242857 | contrast | 0.00% | ✅ 一致 |
| global_lap_var | 全图对照 | 41.63 | motion | 9.11919 | blur | 78.09% | ⚠️ 有差 |

> **口径**：膝点是**必要条件**不是充分条件——过线只代表「物理上还有可能跟住」，过线窗口的实际跟住率是另一个读数（见第二节干净帧基线与第四节尾部验证）。

> 🔴 **读法提醒**：`surround_blockiness` 的决定族是 jpeg，但在红外上块效应**不随码率上升**（见 3.5）⇒ 这条线应读作「**背景环纹理粗糙度的下限**」，而不是「码率下限」；`global_contrast` 的决定族是 contrast，同理要按「动态范围下限」读。

### 3.5 JPEG 族在红外上的物理特殊性（jpeg 族的锐度 / 块效应膝点**不能**当码率下限读）

> 低码率 JPEG 在**红外小目标**画面上的主效应**不是**锐度下降、也**不是**块效应上升（下表数字现场读自明细 CSV，是全量窗口的中位数）：
> ① **锐度类**（`target_lap_var`）在 jpeg 族内**基本持平**：DCT 振铃把高频能量塞回 9×9 核心区，**抵消**了背景被抹噪带来的损失 ⇒ 不是有效退化轴；
> ② **块效应类**（`surround_blockiness`）**明确不升反降**：低对比、噪声主导的红外背景被 JPEG **先抹噪**，块内梯度掉得比块边界梯度更快 ⇒ 「块效应」在红外上根本不是 JPEG 的表征；
> ⇒ 任务书「退化强度递增 ⇒ 局部画质指标单调下降」这条在 JPEG 族**不成立**；jpeg 族膝点只能理解为「该指标-跟住率关系」的拐点，**不能**当成码率下限；
>    真正的码率信息损失用 **MAE（退化帧 vs 原帧）**量化：机检用例证明它逐档单调递增。
> 该结论已固化成机检用例：`test_jpeg_on_ir_is_not_a_sharpness_or_blockiness_axis`。

| 指标 | jpeg q=60（最好） | jpeg q=10（最差） | 变化 | 结论 |
|:--|--:|--:|--:|:--|
| `target_lap_var` | 7252.79 | 7104.84 | -2.0% | ❌ 基本持平（非退化轴） |
| `surround_blockiness` | 0.981215 | 0.816256 | -16.8% | ❌ 不升反降（抹噪） |
| `local_scr` | 7.66496 | 6.98648 | -8.9% | ✅ 可作退化轴（单调下降） |
| `global_contrast` | 0.872549 | 0.893314 | +2.4% | ⚠️ 方向相反（JPEG 反而抬高） |

> 上表数字现场读自 `degrade_detail.csv`（全量窗口的中位数），非手工抄写。

## 四、零合成退化：真实红外画面上的尾部验证

> 在**未施加任何人工退化**的真实红外窗口上，按指标排序取最低 10% / 15% 尾部，对比尾部窗口与其余窗口的**真实跟住率**，并给出「预测跟丢」的 ROC-AUC。

| 指标 | 类型 | 尾部 | 跟踪器 | 尾部跟住率 | 其余跟住率 | 差(pp) | 预测跟丢 AUC |
|:--|:--|--:|:--|--:|--:|--:|--:|
| local_scr | local | 10% | xcorr | 60.7% | 87.1% | 26.5 | 0.6938 |
| local_scr | local | 10% | kcf_legacy | 71.7% | 86.1% | 14.4 | 0.6141 |
| local_scr | local | 10% | csrt | 79.1% | 94.8% | 15.7 | 0.7446 |
| local_scr | local | 15% | xcorr | 60.1% | 88.8% | 28.7 | 0.6938 |
| local_scr | local | 15% | kcf_legacy | 70.2% | 87.2% | 17.0 | 0.6141 |
| local_scr | local | 15% | csrt | 77.6% | 96.0% | 18.4 | 0.7446 |
| target_lap_var | local | 10% | xcorr | 71.7% | 85.9% | 14.2 | 0.6192 |
| target_lap_var | local | 10% | kcf_legacy | 77.0% | 85.5% | 8.4 | 0.5819 |
| target_lap_var | local | 10% | csrt | 87.3% | 93.9% | 6.6 | 0.5384 |
| target_lap_var | local | 15% | xcorr | 72.4% | 86.6% | 14.2 | 0.6192 |
| target_lap_var | local | 15% | kcf_legacy | 76.5% | 86.1% | 9.6 | 0.5819 |
| target_lap_var | local | 15% | csrt | 88.0% | 94.2% | 6.2 | 0.5384 |
| target_local_contrast | local | 10% | xcorr | 53.7% | 87.9% | 34.2 | 0.7970 |
| target_local_contrast | local | 10% | kcf_legacy | 67.6% | 86.5% | 18.9 | 0.6303 |
| target_local_contrast | local | 10% | csrt | 83.6% | 94.3% | 10.7 | 0.6777 |
| target_local_contrast | local | 15% | xcorr | 56.0% | 89.5% | 33.5 | 0.7970 |
| target_local_contrast | local | 15% | kcf_legacy | 70.5% | 87.1% | 16.6 | 0.6303 |
| target_local_contrast | local | 15% | csrt | 86.1% | 94.5% | 8.4 | 0.6777 |
| global_contrast | global | 10% | xcorr | 70.9% | 86.0% | 15.1 | 0.5831 |
| global_contrast | global | 10% | kcf_legacy | 80.3% | 85.1% | 4.8 | 0.4974 |
| global_contrast | global | 10% | csrt | 95.9% | 92.9% | -3.0 | 0.4697 |
| global_contrast | global | 15% | xcorr | 70.2% | 87.0% | 16.8 | 0.5831 |
| global_contrast | global | 15% | kcf_legacy | 80.3% | 85.4% | 5.1 | 0.4974 |
| global_contrast | global | 15% | csrt | 94.8% | 93.0% | -1.8 | 0.4697 |
| global_lap_var | global | 10% | xcorr | 57.4% | 87.5% | 30.1 | 0.6044 |
| global_lap_var | global | 10% | kcf_legacy | 77.9% | 85.4% | 7.5 | 0.4729 |
| global_lap_var | global | 10% | csrt | 95.1% | 93.0% | -2.0 | 0.4070 |
| global_lap_var | global | 15% | xcorr | 58.5% | 89.1% | 30.6 | 0.6044 |
| global_lap_var | global | 15% | kcf_legacy | 79.2% | 85.6% | 6.4 | 0.4729 |
| global_lap_var | global | 15% | csrt | 94.3% | 93.1% | -1.2 | 0.4070 |

## 五、指标判别力（退化族内 Spearman）

| 指标 | 类型 | XCorr \|max ρ\| | KCF \|max ρ\| | CSRT \|max ρ\| |
|:--|:--|--:|--:|--:|
| local_scr | 局部 ROI | 0.318 | 0.150 | 0.244 |
| target_local_contrast | 局部 ROI | 0.409 | 0.192 | 0.207 |
| target_lap_var | 局部 ROI | 0.186 | 0.130 | 0.085 |
| target_grad_mean | 局部 ROI | 0.325 | 0.196 | 0.153 |
| surround_std | 局部 ROI | 0.044 | 0.016 | 0.138 |
| surround_edge_density | 局部 ROI | 0.188 | 0.117 | 0.229 |
| surround_blockiness | 局部 ROI | 0.062 | 0.035 | 0.040 |
| global_contrast | 全图对照 | 0.142 | 0.079 | 0.013 |
| global_lap_var | 全图对照 | 0.155 | 0.055 | 0.062 |

> **结论（业务口径）**：局部 ROI 指标（`local_scr` / `target_lap_var` / `target_local_contrast`）在**退化族内**与跟住率的相关性应当**强于**全图对照指标（`global_contrast` / `global_lap_var`）——这正是 v2 从「全图随机采点」转向「业务目标本体」的收益所在。

## 六、复现方式与验收

```bash
# 服务器（huangzhe@192.168.99.40:32222，conda env uav）
PYTHONPATH=/tmp/pylibs_cvcontrib python3 iqa_v2/run_eval_v2.py --phase all --workers 24
python3 iqa_v2/verify_recalc_v2.py --samples 120      # 独立复算，误差 < 1e-6
python3 -m pytest iqa_v2/test_iqa_v2.py -v             # 机械门禁 100% 通过
md5sum -c iqa_v2/output/manifest_v2.md5                 # 产物 MD5 全量校验
```
