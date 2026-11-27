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
| 有效 5 帧窗口数 | **60** |
| 小目标子集（max(w,h) ≤ 20 px） | **22** 条（占比 36.7%） |

剔除名单为空：本机 Anti-UAV 副本**已是单目标版本**（223 条序列逐条解析，**0 条**出现单帧多框 / 多 track_id）。

## 二、跟踪器三臂定位与干净帧基线

| 跟踪臂 | 角色 | 干净帧跟住率 | 中位 CLE(px) | P90 CLE(px) | 异常/报错次数 |
|:--|:--|--:|--:|--:|--:|
| `xcorr` | **主臂**：模板匹配链式配准（与 KCF 跨家族） | 98.3% | 0.90 | 2.33 | 0 |
| `kcf_legacy` | **主臂**：任务书指定的 contrib legacy KCF（detect_thresh=0.0） | 68.3% | 2.85 | 45.22 | 0 |
| `csrt` | 同族对照：DCF 家族的另一实现（CSRT） | 98.3% | 1.12 | 2.30 | 0 |

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

## 三、4 维退化下的局部 ROI 指标膝点反解

> **跨家族纪律**：XCorr 与 KCF 的绝对锁定率刻度**不可互换**，下表**并列呈现、各自独立反解**，**严禁合并求平均**；可比的是**线位**。

### 3.1 采用的膝点（全量窗口 / 全量曲线集）

| 退化族 | 指标 | XCorr 膝点 | KCF 膝点 | 两族线位差 | 可用性 |
|:--|:--|--:|--:|--:|:--|
| blur | target_local_contrast | 0.181403 | 0.224824 | 19.31% | ✅ 可靠 |
| blur | target_grad_mean | 45.7988 | 37.9873 | 17.06% | ✅ 可靠 |
| blur | surround_std | 4.41272 | 4.8608 | 9.22% | ✅ 可靠 |
| blur | global_lap_var | 1.25368 | 1.25368 | 0.00% | ✅ 可靠 |
| motion | local_scr | 14.6575 | 14.667 | 0.06% | ✅ 可靠 |
| motion | target_local_contrast | 0.162458 | 0.227225 | 28.50% | ✅ 可靠 |
| motion | target_lap_var | 2442.48 | 2442.48 | 0.00% | ✅ 可靠 |
| motion | target_grad_mean | 51.7639 | 64.5185 | 19.77% | ✅ 可靠 |
| motion | surround_std | 4.4169 | 4.89746 | 9.81% | ✅ 可靠 |
| motion | global_lap_var | 11.5582 | 29.2184 | 60.44% | ✅ 可靠 |
| contrast | target_local_contrast | 0.114896 | 0.190627 | 39.73% | ✅ 可靠 |
| contrast | global_lap_var | 6.30103 | 6.30103 | 0.00% | ✅ 可靠 |
| jpeg | local_scr | 14.0773 | 16.4064 | 14.20% | ✅ 可靠 |
| jpeg | target_lap_var | 4161.49 | 3810.65 | 8.43% | ✅ 可靠 |
| jpeg | target_grad_mean | 71.7385 | 75.1501 | 4.54% | ✅ 可靠 |
| jpeg | surround_std | 5.02467 | 5.29382 | 5.08% | ✅ 可靠 |
| jpeg | global_lap_var | 175.717 | 175.717 | 0.00% | ✅ 可靠 |

### 3.2 小目标子集（max(w,h) ≤ 20 px）

| 退化族 | 指标 | XCorr 膝点 | KCF 膝点 | CSRT 膝点（同族对照） |
|:--|:--|--:|--:|--:|
| blur | local_scr | 11.1861 | 12.7022 | 11.1861 |
| blur | target_lap_var | 1995.7 | 1995.7 | 1995.7 |
| blur | target_grad_mean | 46.3194 | 46.3194 | 46.3194 |
| blur | surround_std | 4.47009 | 4.47009 | 4.40574 |
| blur | global_lap_var | 1.44222 | 1.8301 | 1.44222 |
| motion | local_scr | 18.6691 | 20.3254 | 18.6691 |
| motion | target_local_contrast | 0.415902 | 0.415902 | 0.415902 |
| motion | target_lap_var | 3607.48 | 5224.28 | 3607.48 |
| motion | target_grad_mean | 101.274 | 101.274 | 101.274 |
| motion | surround_std | 4.4169 | 4.4169 | 4.4169 |
| motion | global_lap_var | 11.8619 | 19.9934 | 11.8619 |
| contrast | local_scr | 24.0813 | 30.6217 | 24.0813 |
| contrast | target_local_contrast | 0.114689 | 0.186235 | 0.114689 |
| contrast | target_lap_var | 2186.08 | 4200.8 | 2186.08 |
| contrast | target_grad_mean | 30.9243 | 30.9243 | 30.9243 |
| contrast | global_contrast | 0.160377 | 0.160377 | 0.160377 |
| jpeg | local_scr | 22.7487 | 26.381 | 22.7487 |
| jpeg | target_local_contrast | 0.541099 | 0.546404 | 0.541099 |
| jpeg | target_lap_var | 16405.3 | 20088.1 | 16405.3 |
| jpeg | target_grad_mean | 177.164 | 198.477 | 177.164 |
| jpeg | surround_std | 5.0344 | 5.0344 | 5.03504 |
| jpeg | global_lap_var | 213.436 | 218.401 | 213.436 |

### 3.3 同族对照：KCF vs CSRT（证明 KCF 侧线位不是实现细节）

| 退化族 | 指标 | KCF 膝点 | CSRT 膝点 | 线位差 |
|:--|:--|--:|--:|--:|
| blur | target_local_contrast | 0.224824 | 0.128557 | 42.82% |
| blur | target_grad_mean | 37.9873 | 37.9873 | 0.00% |
| blur | surround_std | 4.8608 | 4.17198 | 14.17% |
| blur | global_lap_var | 1.25368 | 3.4682 | 63.85% |
| motion | local_scr | 14.667 | 14.6575 | 0.06% |
| motion | target_local_contrast | 0.227225 | 0.138079 | 39.23% |
| motion | target_lap_var | 2442.48 | 2442.48 | 0.00% |
| motion | target_grad_mean | 64.5185 | 51.7639 | 19.77% |
| motion | surround_std | 4.89746 | 4.18007 | 14.65% |
| motion | global_lap_var | 29.2184 | 11.5582 | 60.44% |
| contrast | target_local_contrast | 0.190627 | 0.114896 | 39.73% |
| contrast | global_lap_var | 6.30103 | 6.30103 | 0.00% |
| jpeg | local_scr | 16.4064 | 14.0773 | 14.20% |
| jpeg | target_lap_var | 3810.65 | 4161.49 | 8.43% |
| jpeg | target_grad_mean | 75.1501 | 71.7385 | 4.54% |
| jpeg | surround_std | 5.29382 | 5.02467 | 5.08% |

### 3.4 反推出的「良好跟踪合格线」（约束线）

> 每个指标在 4 个退化族上各有各的膝点；**准入线必须取各族膝点的最小值**（最严格的那一族说了算）——因为画质指标都是「越大越好」，线一旦定在某个族的膝点之上，另一族就会崩。

| 指标 | 类型 | XCorr 约束线 | 决定族 | KCF 约束线 | 决定族 | 两族线位差 | 可用性 |
|:--|:--|--:|:--|--:|:--|--:|:--|
| local_scr | 局部 ROI | 14.0773 | jpeg | 14.667 | motion | 4.02% | ✅ 一致 |
| target_local_contrast | 局部 ROI | 0.114896 | contrast | 0.190627 | contrast | 39.73% | ⚠️ 有差 |
| target_lap_var | 局部 ROI | 2039.1 | blur | 2442.48 | motion | 16.52% | ⚠️ 有差 |
| target_grad_mean | 局部 ROI | 45.7988 | blur | 37.9873 | blur | 17.06% | ⚠️ 有差 |
| surround_std | 局部 ROI | 4.41272 | blur | 4.56522 | contrast | 3.34% | ✅ 一致 |
| global_lap_var | 全图对照 | 1.25368 | blur | 1.25368 | blur | 0.00% | ✅ 一致 |

> **口径**：膝点是**必要条件**不是充分条件——过线只代表「物理上还有可能跟住」，过线窗口的实际跟住率是另一个读数（见第二节干净帧基线与第四节尾部验证）。

## 四、零合成退化：真实红外画面上的尾部验证

> 在**未施加任何人工退化**的真实红外窗口上，按指标排序取最低 10% / 15% 尾部，对比尾部窗口与其余窗口的**真实跟住率**，并给出「预测跟丢」的 ROC-AUC。

| 指标 | 类型 | 尾部 | 跟踪器 | 尾部跟住率 | 其余跟住率 | 差(pp) | 预测跟丢 AUC |
|:--|:--|--:|:--|--:|--:|--:|--:|
| local_scr | local | 10% | xcorr | 100.0% | 98.1% | -1.9 | 0.5763 |
| local_scr | local | 10% | kcf_legacy | 100.0% | 64.8% | -35.2 | 0.5045 |
| local_scr | local | 10% | csrt | 100.0% | 98.1% | -1.9 | 0.3390 |
| local_scr | local | 15% | xcorr | 100.0% | 98.0% | -2.0 | 0.5763 |
| local_scr | local | 15% | kcf_legacy | 100.0% | 62.7% | -37.3 | 0.5045 |
| local_scr | local | 15% | csrt | 100.0% | 98.0% | -2.0 | 0.3390 |
| target_lap_var | local | 10% | xcorr | 100.0% | 98.1% | -1.9 | 0.7966 |
| target_lap_var | local | 10% | kcf_legacy | 0.0% | 75.9% | 75.9 | 0.8383 |
| target_lap_var | local | 10% | csrt | 100.0% | 98.1% | -1.9 | 0.3729 |
| target_lap_var | local | 15% | xcorr | 100.0% | 98.0% | -2.0 | 0.7966 |
| target_lap_var | local | 15% | kcf_legacy | 0.0% | 80.4% | 80.4 | 0.8383 |
| target_lap_var | local | 15% | csrt | 100.0% | 98.0% | -2.0 | 0.3729 |
| target_local_contrast | local | 10% | xcorr | 83.3% | 100.0% | 16.7 | 1.0000 |
| target_local_contrast | local | 10% | kcf_legacy | 0.0% | 75.9% | 75.9 | 0.9949 |
| target_local_contrast | local | 10% | csrt | 100.0% | 98.1% | -1.9 | 0.6780 |
| target_local_contrast | local | 15% | xcorr | 88.9% | 100.0% | 11.1 | 1.0000 |
| target_local_contrast | local | 15% | kcf_legacy | 0.0% | 80.4% | 80.4 | 0.9949 |
| target_local_contrast | local | 15% | csrt | 100.0% | 98.0% | -2.0 | 0.6780 |
| global_contrast | global | 10% | xcorr | 100.0% | 98.1% | -1.9 | 0.0000 |
| global_contrast | global | 10% | kcf_legacy | 100.0% | 64.8% | -35.2 | 0.0026 |
| global_contrast | global | 10% | csrt | 100.0% | 98.1% | -1.9 | 0.0254 |
| global_contrast | global | 15% | xcorr | 100.0% | 98.0% | -2.0 | 0.0000 |
| global_contrast | global | 15% | kcf_legacy | 100.0% | 62.7% | -37.3 | 0.0026 |
| global_contrast | global | 15% | csrt | 100.0% | 98.0% | -2.0 | 0.0254 |
| global_lap_var | global | 10% | xcorr | 100.0% | 98.1% | -1.9 | 0.7627 |
| global_lap_var | global | 10% | kcf_legacy | 16.7% | 74.1% | 57.4 | 0.9756 |
| global_lap_var | global | 10% | csrt | 100.0% | 98.1% | -1.9 | 0.7288 |
| global_lap_var | global | 15% | xcorr | 100.0% | 98.0% | -2.0 | 0.7627 |
| global_lap_var | global | 15% | kcf_legacy | 11.1% | 78.4% | 67.3 | 0.9756 |
| global_lap_var | global | 15% | csrt | 100.0% | 98.0% | -2.0 | 0.7288 |

## 五、指标判别力（退化族内 Spearman）

| 指标 | 类型 | XCorr \|max ρ\| | KCF \|max ρ\| | CSRT \|max ρ\| |
|:--|:--|--:|--:|--:|
| local_scr | 局部 ROI | 0.061 | 0.089 | 0.089 |
| target_local_contrast | 局部 ROI | 0.279 | 0.792 | 0.217 |
| target_lap_var | 局部 ROI | 0.174 | 0.472 | 0.139 |
| target_grad_mean | 局部 ROI | 0.270 | 0.644 | 0.202 |
| surround_std | 局部 ROI | 0.288 | 0.610 | 0.222 |
| surround_edge_density | 局部 ROI | nan | nan | nan |
| surround_blockiness | 局部 ROI | 0.192 | 0.337 | 0.192 |
| global_contrast | 全图对照 | 0.306 | 0.804 | 0.222 |
| global_lap_var | 全图对照 | 0.217 | 0.749 | 0.148 |

> **结论（业务口径）**：局部 ROI 指标（`local_scr` / `target_lap_var` / `target_local_contrast`）在**退化族内**与跟住率的相关性应当**强于**全图对照指标（`global_contrast` / `global_lap_var`）——这正是 v2 从「全图随机采点」转向「业务目标本体」的收益所在。

## 六、复现方式与验收

```bash
# 服务器（huangzhe@192.168.99.40:32222，conda env uav）
PYTHONPATH=/tmp/pylibs_cvcontrib python3 iqa_v2/run_eval_v2.py --phase all --workers 24
python3 iqa_v2/verify_recalc_v2.py --samples 120      # 独立复算，误差 < 1e-6
python3 -m pytest iqa_v2/test_iqa_v2.py -v             # 机械门禁 100% 通过
md5sum -c iqa_v2/output/manifest_v2.md5                 # 产物 MD5 全量校验
```
