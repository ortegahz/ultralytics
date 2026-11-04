# SPIRE-IRSTD 在 Anti-UAV 上掉分的根因排查

> **文件状态警告 (2026-09-30 17:24)**
> 本文件曾达 768 行。17:04 一次"服务器 -> 本地"同步把本地独有、服务器没有的
> `spire_antiuav_debug.md` 当成删除处理, 导致第一~八节原文丢失, 随后 17:23 的
> `cat >>` 只是在新建的空文件上追加 (只剩 50 行)。
>
> * **第九~十四节**: 本次会话亲笔所写, 逐字恢复, 可信。
> * **第一~八节**: **按章节标题与会话记录重建**, 数据点来自当时的实测输出,
>   但**措辞不是原文**。引用具体数字前请回原始实验日志核对。
>
> 已同步一份到服务器 `/tmp/pycharm_project_10ae9e2e/manu/memory/`,
> 并备份到 `/media/manu/1TB-Volume/workspace/_backup/`。后续写本文件务必两边都写。

---

## 一、数据集事实 (Anti-UAV 侧, 全部实测)

数据集: `/mnt/data/siping/datasets/manu/spire/YOLO26-SPIRE`
`img_idx/train.txt` = 43,008 行, `img_idx/test.txt` = 31,613 行。
严格 **1 目标/图**: 74,621 张图 / 74,621 条标注记录, 但其中 **12% 的标注 `bbox` 为空**
(纯背景帧)。test 集共 **11 段视频**, train/test 是**按视频切分**的。

目标尺寸 (目标 bbox 对角线, 原图像素):

| 集合 | n | 覆盖序列 | 中位 | p10 | p90 | <7px 占比 |
|---|---:|---:|---:|---:|---:|---:|
| 完整 test | 31,613 | 11/11 | 13.60 | 5.00 | 26.63 | **0.171** |
| `test[:8000]` | 8,000 | **8/11** | 21.26 | 12.81 | 103.06 | **0.000** |
| `test[:2000]` | 2,000 | **2/11** | 29.55 | 13.45 | 43.66 | **0.000** |
| 随机 2000 (seed 0) | 2,000 | 11/11 | 13.45 | 5.00 | 28.23 | 0.187 |
| 朴素 `[:200]` | 200 | **1/11** | 14.21 | 12.73 | 17.03 | **0.000** |

**这张表是本次排查最重要的单张表**, 详见第九、十节。

目标局部信噪比 (local SNR): SIRST-UAVB raw 1.43 / **Anti-UAV raw 0.86**
(58.9% 的目标 SNR < 1)。逐通道 (GMCM 特征): ch0 原始灰度 2.31、ch1 GMC 差分 27.57、
ch2 中值残差 **46.55**。

## 二、已定位并修复的代码缺陷

### 缺陷 1 (性能, 已修): 数据集构造 O(N²) 全表扫描
`utils/dataset.py` 构造 `target_list` 时对每张图全表扫描 annotations。
改为 dict 索引后: **4.5 min+ -> 2.7 s**。

### 缺陷 2 (性能, 已修): PRPS 热图贴图的 Python 13×13 双重循环
`utils/transforms.py` 贴高斯用 Python 双重循环。改为 numpy 向量化,
**400 组随机用例 max diff = 0.0000**。

### 缺陷 3 (正确性, 已定位, 可开关): PRPS 二次 min-max 归一化
监督热图被归一化两次。后果实测:
* 峰值 **1.0093** (范围 1.0001~1.0126), **超过 1.0**;
* 100% 样本峰值 >1.0, 最大 4.9026;
* 但**等效增益中位仅 1.01** (弱/强目标放大比 1.0~1.1 倍) —— **"对比度放大器"假设被证伪**。
* 实际危害在第十一节: `gt.eq(1.0)` 取正样本会得到**空集**, `(1-gt)**beta` 直接 **NaN**。
* 平坦 patch NaN: 2/5665 = 0.04%, 最终热图 NaN/Inf 均为 0, **未污染训练**。
* 开关: `--fix_double_norm`。**默认保留原行为** —— 虽然是真 bug, 但实测影响面极小,
  不应在没有多种子验证的情况下改动。

### 缺陷 4 (结构性, 已参数化): stride=4 的定位量化误差
`--heat_stride {1,2,4}`。默认 4 = repo 原行为。**注意: 定位精度诊断已证伪 stride 假设**
(见第十一节与匹配半径扫描), 优先级下调。

### 缺陷 5 (增强方向, 已参数化): 增广只会放大目标
`utils/transforms.py` L204-206:
```python
scale_factor = random.uniform(*self.scale)   # self.scale = (opt.scale_min, 1.15)
```
默认 `scale_min=1.0` -> `scale_factor ∈ [1.00, 1.15]`, **只放大、从不缩小**。
而 test 集 17.1% 的目标 <7px。开关: `--scale_min 0.8`。

### 缺陷 6 (评测, 已修): 固定阈值 + 前缀切片
见第九节 (前缀切片) 与第十一节 (阈值网格)。

## 三、诊断方法论与已建立的工具

控制组: 用官方权重在 SIRST-UAVB test 上复现 **F1 0.9680**
(README 97.05, git log 94.74) —— 证明诊断链路可信。

19 个工具位于 `SPIRE-IRSTD/tools/`:
`analyze_stats.py`、`measure_target_size.py`、`measure_input_contrast.py`、
`inspect_gmc_channels.py`、`diag_collect.py`、`diag_analyze.py`、`diag_localization.py`、
`eval_tp_distance.py`、`build_gmcm_spire.py`、`check_flat_patch_nan.py`、
`analyze_amplification.py`、`bench_dataset.py`、`bench_pipeline.py`、`debug_stamp.py`、
`smoke_stride.py`、`measure_double_norm.py`、`eval_thr_halfsplit.py`(本次新增)、
`run_honest_eval.sh`、`run_loss_ab.sh`、`run_multiseed.sh`、`launch_full_runs.sh`。

数据集: `GMCM-SPIRE` (`tools/build_gmcm_spire.py`) —— 74,621 张软链接至
`uav_gmc_median/images`, 标注复用 YOLO26-SPIRE, 尺寸校验 200/200 一致, 0 缺失。

## 四、推荐配置与可复制命令

见 `tools/run_honest_eval.sh` 与 `tools/run_multiseed.sh`。
**注意: `--val_limit` 现为随机抽样 (见第九节), 旧的前缀切片行为已移除。**

## 五、A/B 受控实验结果 (12k 子集, tpd=5, seed 42)

| 实验 | ep0 | ep5 | ep10 | ep15 |
|:--|--:|--:|--:|--:|
| A2 原始灰度 | 0.0006 | 0.5675 | **0.7421** | 0.7315 |
| D GMC+中值 | 0.0447 | **0.6785** | 0.7284 | 0.6490 |

种子噪声: 同配置无 seed 0.7824 vs seed=42 0.5675, **差 0.215**。
**这是本项目所有 A/B 的噪声下限, 单种子差异小于它一律不下结论。**

## 六、已证伪的假设 (全部有实测数据)

1. **"训练/验证目标尺寸错配 2.57 倍"** —— 用户用 Trial 0474 (F1 0.9064) 反驳, **反例成立**。
   且 "train 中位 36.80px vs val 14.32px" 这个测量本身就是在前缀切片上做的, 修正后不成立。
2. **"输入契约不对 / 该喂 GMC+中值特征"** —— 五个数据点证伪: 只加速收敛, 不抬天花板。
   `F_full_gmcm` ep0=0.7333 (收敛极快) 但 ep4=0.6858 回落, 而 raw 输入 ep0=0.0594 / ep4=0.7521。
3. **"sweep 网格上限 0.5 截断了最优点"** —— 修正协议后曲线是标准倒 U, 峰值在 **thr=0.25**,
   本就在旧网格内; 0.50 处 F1 0.5725 vs 0.25 处 0.5775, 只差 0.005。**证伪。**
4. **"repo 自带的 FocalMSELoss 可以直接启用"** —— 方向是**反的**, 见第十节。

## 七、诊断工具的实测结论

### 7.1 定位精度不是问题
E_full_raw ep4 定位误差中位 **1.80px**; 4px 量化检验: 17.5% 落在 4px 以下,
低于均匀期望 25% -> **无栅格特征, stride 假设被证伪**。

### 7.2 失败模式是"检出"不是"定位"
5328 个 GT 中仅 4001 个有附近预测 -> **24.9% 完全漏检**。
`tp_distance` 放宽到 20px, recall 仍封顶 0.7408 —— 不是匹配半径不够。

### 7.3 匹配半径是显著的协议杠杆 (同一 checkpoint)
`tpd` 3 -> 0.6178、5 -> **0.7421**、8 -> **0.7766**、10 -> 0.7809、15 -> 0.7971、20 -> 0.8006。
用户 Trial 0474 用 `Distance<=8.0px`, 故本项目统一 `tp_distance=8` 对齐口径。

## 八、数据集与输入特征 (GMC 路线)

* Trial 0474 输入特征: `/mnt/data/siping/datasets/manu/uav_gmc_median/images/{train,val}`
* GMCM-SPIRE: `/mnt/data/siping/datasets/manu/spire/GMCM-SPIRE`
* 官方集: `/mnt/data/siping/datasets/manu/spire/{SIRST-UAVB,SIRST4}`

---

## 九、评测协议缺陷: 前缀切片导致所有历史 F1 不可比 (2026-09-30, 最高优先级)

### 9.1 缺陷本体

`utils/dataset.py` 的 `target_list` 顺序来自 `img_idx/*.txt`, 这些文件**按视频序列分块排列**。
`train_ddp.py` 原先的子集开关是:

```python
val_dataset.target_list = val_dataset.target_list[:opt.val_limit]   # 朴素前缀
```

于是 `--val_limit 8000` 只取到 test 集**前 8 段视频**, 而不是全 11 段的随机代表性子集。

### 9.2 实测证据

见第一节的尺寸表。核心: **`test[:8000]` 里一个 <7px 的微小目标都没有, 真实 test 是 17.1%。**
即: 历史报告的 0.7521 / 0.6858 / 0.7421 / 0.9064 等数字都建立在一个"偏易且不具代表性"的子集上。

### 9.3 由此发现的对比基准错误

用户自己的 Trial 0474 (`runs/optuna_p0_nas/trial_0474/results.csv`):

```
epoch  best_th  recall  precision  f1      tp     fp     gt
1      0.25     0.8609  0.9559     0.9059  21618  998    25111
3      0.25     0.8619  0.9557     0.9064  21643  1004   25111
```

`gt = 25111` 对应 **完整 31,613 张 test** (25111/31613 = 0.794, 与实测 box/img 一致)。
我的 E/F 跑在 8000 张前缀上、只有 6969 个 GT。
**这两个数字从来不可比。**

### 9.4 已修复

`train_ddp.py` 的子集逻辑改为固定种子随机抽样, 并新增 `--val_seed` (与 `--seed` 分开,
保证所有实验的 val 集完全一致):

```python
def _subset(ds, n, seed, tag):
    if n and n < len(ds.target_list):
        rng = _np.random.RandomState(seed)
        idx = sorted(rng.permutation(len(ds.target_list))[:n])
        ds.target_list = [ds.target_list[i] for i in idx]
    return ds
```

### 9.5 血泪教训 (违反了我自己写过的铁律)

我在 `tools/eval_thr_halfsplit.py` 里第一次跑出 **F1 = 0.9925** 时, 第一反应是"不可能, 肯定有 bug"。
实际原因只是 `--val_limit 200` 抽到了单段视频 `01_*` (1 段, 目标对角 12.73~17.03px 的窄带)。
**代码是对的, 抽样是错的。** 教训: 任何带 `--val_limit` 的评测工具, 必须内置随机抽样 +
打印子集统计 (序列数 / 目标尺寸分布), 让抽样偏差无处隐藏。现在 `eval_thr_halfsplit.py`
每次都会打印 `subset stats: n_seq=... target_diag: median/p10/p90`。

---

## 十、损失函数方向性分析 (2026-09-30)

### 10.1 失败模式是"检出"不是"定位"

E_full_raw ep4 定位误差中位 1.80px, 但 **5328 个 GT 中 4001 个有附近预测 —— 24.9% 完全漏检**,
且 `tp_distance` 放宽到 20px 时 recall 仍封顶 0.7408。

### 10.2 KpLoss 对目标没有任何偏好 (定量)

用真实高斯 PRPS 热图 (160x160, sigma=2) 实测权重比 (峰值区 heatmap>0.5 vs 背景 heatmap<0.01):

| 损失 | 峰值权重 / 背景权重 | 方向 |
|---|---:|---|
| `KpLoss` (repo 默认, 全图等权 MSE) | **1.000** | 中性 —— 目标像素只占 0.64%, 梯度预算自然被背景吃掉 |
| `FocalMSELoss(gamma=2.0)` | **0.124** | **反向** —— 把梯度预算推离目标 8 倍 |
| `FocalMSELoss(gamma=0.5)` | 0.534 | 反向 |
| `PosFocalMSELoss(beta=3)` (本次新增) | **3.050** | 正向 |

**repo 自带的 `FocalMSELoss` 权重是 `(1-heatmap)**gamma`, 方向是反的。**
它的设计意图是"增强低置信度区域的学习", 但在红外小目标这个任务上, 低置信度区域恰恰是背景。
所以待办清单里"启用 FocalMSELoss"这条路可以直接关掉, 不必浪费一次实验。

### 10.3 新增 `PosFocalMSELoss` (utils/loss.py)

```
w = (1 - y)**gamma_neg * (1 + beta * y)        # 归一化到 batch 内均值 = 1
loss = mean_{H,W}[ w * (pred - y)^2 ] * 2 / bs
```

* `beta=0, gamma_neg=0` 时**严格等于 KpLoss** (实测比值 1.0000), 可作 A/B 基线。
* 权重按均值归一化 -> loss 尺度与 KpLoss 一致 (实测各配置比值 0.9969~1.0007),
  从而把"重加权的效果"与"有效学习率变化的效果"**分离开**。
* `train_ddp.py` 新增 `--loss {kp,posfocal,cntfocal}` `--loss_beta` `--loss_gamma_neg`。

---

## 十一、根因定位: 对比用户自有网络的损失函数 (2026-09-30 16:00, 决定性)

### 11.1 诚实的基线数字 (修正后)

E_full_raw ep4 checkpoint, 在**正确随机抽样**的 2000 张 val (11/11 段, <7px 占 18.7%) 上,
宽阈值网格 0.05..0.95 + 分半验证协议:

```
   thr |    A:F1    B:F1 |  all:F1 |   all:P   all:R
  0.20 |  0.5616  0.5788 |  0.5702 |  0.5631  0.5775
  0.25 |  0.5731  0.5818 |  0.5775 |  0.5933  0.5625   <== 峰值
  0.30 |  0.5693  0.5813 |  0.5754 |  0.6094  0.5450
  0.50 |  0.5510  0.5932 |  0.5725 |  0.7119  0.4788
  0.80 |  0.4829  0.5124 |  0.4979 |  0.9852  0.3331
```

| 口径 | F1 |
|---|---:|
| 有偏前缀 `[:8000]` (此前全部报告) | 0.7521 |
| **诚实随机 val @ thr=0.25** | **0.5775** |
| 分半验证 (阈值在 A 上选, 报 B) | 0.5818 |

**前缀切片把 F1 虚高了 0.175。** 与用户 Trial 0474 的真实差距是 **0.9064 - 0.5775 = 0.33**,
不是此前以为的 0.15。

结果文件: `spire_runs/E_full_raw/halfsplit_ep4_n2000.json` (**目前最有价值的单个产物**)。

### 11.2 假设再次被自己的数据证伪: sweep 网格截断

我曾观察到 "thr 0.40->0.50 仍单调上升", 据此怀疑 `train_ddp.py` 的网格上限 0.50
截断了最优点, 并写了宽网格工具去查。**结论是错的**: 正确随机 val 上曲线是标准倒 U,
峰值在 **thr=0.25**, 完全落在旧网格内部; 旧网格取 0.50 得 0.5725, 真实最优 0.5775,
只差 0.005。**网格截断不是有效问题。**
(网格仍放宽到 0.80, 但那是为了 `--out_sigmoid` 之后高阈值区才值得扫, 不是因为原来漏了。)

巧合但重要: thr=0.25 正是用户 Trial 0474 的 `best_th`。修正后两边协议终于对齐。

### 11.3 真正的根因: KpLoss 的负样本梯度永不衰减

对比 `utils/loss.py::KpLoss` 与用户自有网络的 `manu/models/heatmap_loss.py::FocalLoss`:

**(a) 归一化分母不同**
```python
# SPIRE KpLoss: 25600 个像素等权平均, 目标像素只占 0.64%
loss = (logits - heatmaps).pow(2).mean(dim=[2,3]) * 2 / bs
# 用户 FocalLoss: 负样本项除以正样本个数, 而不是除以像素数
focal_total = -(pos_loss.sum() + neg_loss.sum()) / num_pos
```

**(b) 简单负样本的梯度行为 —— 这是决定性差异**
MSE 在 y=0 时梯度恒为 `2p`, 与像素"有多简单"无关。
CenterNet focal 的负项 `log(1-p) * p^alpha` (alpha=2) 梯度随 p->0 **趋于零**。实测:

| 背景像素 p | CNTFocal 的 \|dL/dp\| |
|---:|---:|
| 0.5 | 1.193147 |
| 0.2 | 0.139257 |
| 0.05 | 0.007761 |
| 0.01 | 0.000302 |
| 0.001 | 0.000003 |

从 p=0.5 到 p=0.001 衰减约 **4x10^5 倍**。
160x160 热图里有 25,600 个背景像素, 在 MSE 下它们**永远不会闭嘴**, 持续把网络往下拽;
这与实测的"24.9% 的 GT 完全无邻近预测、定位误差却只有 1.80px"完全吻合 ——
失败在**检出**, 不在定位。

**(c) 用户自己早就写下了这个诊断** (`manu/models/heatmap_loss.py` 的类文档):
> "Rigidly penalizing the center pixel as if it were an isolated 1.0 peak suppresses
> predicted response peaks down to 0.15~0.22, leading to massive false-negative
> truncations at standard operating threshold th=0.25."

**(d) 模型输出无界**: SPIRE 的 `final_layer` 是裸 1x1 conv, **没有任何输出激活**,
输出是无界 logit, 却被拿 0.35/0.5 去阈值化 —— 该工作点没有概率学意义。
用户的网络输出 sigmoid 概率, `best_th=0.25` 因此是有意义的。

### 11.4 实现 CNTFocalLoss 时踩到的两个地雷 (都已实测确认)

1. **正样本不能用 `gt.eq(1.0)`**。SPIRE 的 PRPS 监督做了二次 min-max 归一化,
   实测热图峰值 = **1.0093** (范围 1.0001~1.0126), **超过 1.0**。
   用 `eq(1.0)` 时 40/40 张图的正样本集合**全为空**。改用 `gt >= 0.99` (得到 1 个正样本)。
2. **`(1-gt)**beta` 必须先 clamp**。gt>1 时是负底数取非整数次幂 -> **NaN** (实测 True);
   clamp 到 [0,1] 后无 NaN。

### 11.5 代码改动

* `model/spire_net.py`: 新增 `out_act` (`none` / `sigmoid`)。Sigmoid 无可学参数,
  **不改变 state_dict 的键**, 老 checkpoint 仍可 `load_state_dict`。
* `utils/loss.py`: 新增 `CNTFocalLoss(alpha, beta, pos_weight, pos_thresh)`。
* `train_ddp.py`: 新增 `--loss {kp,posfocal,cntfocal}` `--out_sigmoid` `--loss_alpha`
  `--loss_beta` `--loss_pos_weight` `--loss_pos_thresh`; `cntfocal` 未加 `--out_sigmoid`
  时直接报错退出。
* sweep 网格 0.05..0.80。

### 11.6 梯度预算的定量对比 (真实 PRPS 热图, 30 张)

构造"典型未收敛"预测 (背景 ~N(0,0.05), 目标区 ~0.3), 反传到像素求梯度绝对值之和:

```
每 1 个正样本像素对应 25,525 个背景像素   (pos px=30, bg px=765,748)

loss          |g| target         |g| bg    bg/target
KpLoss            0.0001          0.08       718.8x
CNTFocal          3.4793        100.93        29.0x
```

* `KpLoss` 下背景拿到的梯度总量是目标的 **719 倍**; 换成 CenterNet focal 后降到 **29 倍**,
  即**梯度预算朝目标搬了 24.8 倍**。
* 注意 `KpLoss` 目标区梯度本身只有 1e-4 量级: 因为它先对 160x160 做 `mean(dim=[2,3])`
  (/25600), 正样本的贡献被稀释到几乎为零。

### 11.7 与用户网络的完整差异清单

| 组件 | 用户自有网络 | SPIRE (repo 原状) | 是否已处理 |
|---|---|---|---|
| 热图损失 | CenterNet focal, `/num_pos` 归一化 | 全图等权 MSE, `mean(dim=[2,3])` | 已加 `CNTFocalLoss` |
| 简单负样本梯度 | `p^alpha`, 随 p->0 衰减 4e5 倍 | `2p`, 仅线性衰减 | 同上 |
| 模型输出 | sigmoid -> (0,1) 概率, `best_th=0.25` 有意义 | **无输出激活**, 无界 logit 被硬阈值 | 已加 `--out_sigmoid` |
| 增广方向 | `scale=0.2` 双向缩放 | `scale_min=1.0` **只放大不缩小** | 已参数化 (`--scale_min`), **未测** |
| 高斯半径 | `min_radius=1` 保护 3x3 目标 | 固定 sigma=2 (与目标大小无关) | 不是问题 (监督斑恒大于小目标) |
| 亚像素 offset | RegL1 + offset 分支 | 无 | 未处理 (对检出无帮助) |
| 能量保持 / 峰值池化 | 有 (`EnergyPreservingFocalLoss`) | 无 | 未处理 (待测) |
| 纯背景帧约束 | `SoftIoULoss` 可选 | 无 | 未处理 (待测) |
| 热图 stride | 搜到 2 (`optuna_heatmap_stride2_640`) | 4 | 已参数化 (`--heat_stride`) |
| 后处理提峰 | 3x3 maxpool NMS (conf 0.20, top_k 100) | 相对置信度抑制 value_range=0.35 | **不是差距来源** |

### 11.8 进行中的受控 A/B (seed 42, 同一随机 val 8000, val_seed 0)

| run | GPU | loss | 命令要点 |
|---|---:|---|---|
| `G_cntfocal` | 0 | cntfocal + out_sigmoid | `--loss cntfocal --out_sigmoid --loss_alpha 2.0 --loss_beta 2.4 --loss_pos_thresh 0.99` |
| `H_kp_ctrl` | 1 | kp (对照) | `--loss kp` |

两者除损失函数外**完全相同** (seed 42 / val 8000 随机 / tp_distance 8 / heat_stride 4 /
nEpochs 10 / eval_interval 2 / sweep)。

---

## 十二、Trial 0474 的完整超参 (从 optuna study.db 读出, 2026-09-30)

```
study: p0_nas_search_4096   trial 474   state=COMPLETE   objective=0.906361
(与 runs/optuna_p0_nas/trial_0474/results.csv 的 f1=0.9064 完全一致)

focal_beta        = 2.7          <-- 确认是 CenterNet focal 家族
lr0               = 0.0007182492929231857
weight_decay      = 8.121031254296626e-05
lrf               = 0.0334249094814197
scale             = 0.2
translate         = 0.1
mosaic            = 0.05
offset_weight     = 0.55
downsample_mode   = pixel_unshuffle
stem_type         = standard_dw
fusion_mode       = scalar_gate
gate_depth        = 2
gate_input_mode   = diff_only
gate_mid_channels = 16
```

读取方式 (供复现):
```bash
ssh -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_10ae9e2e
/home/huangzhe/anaconda3/envs/uav/bin/python -c "
import optuna,optuna.logging; optuna.logging.set_verbosity(optuna.logging.WARNING)
st=optuna.load_study(study_name='p0_nas_search_4096', storage='sqlite:///runs/optuna_p0_nas/study.db')
t=st.trials[474]
print(t.state, t.values); [print(k,'=',t.params[k]) for k in sorted(t.params)]"
```

### 12.1 由此暴露的两个新差异

**(a) 学习率差 7 倍**: 用户 `lr0 = 7.18e-4`, SPIRE 默认 **0.005**。
focal loss 的梯度量级与全图 MSE 完全不同, 所以给 CNTFocalLoss 配 0.005 未必合适。
**注意实验设计**: G/H A/B 刻意保持同 lr=0.005, 目的是**隔离"损失函数"这一个变量**;
lr 扫描是独立的下一步, 不能混进同一次对比。

**(b) 增广方向缺失 —— 这条更可疑**
用户的 `scale = 0.2` 表示**双向**随机缩放 (含缩小)。
SPIRE 的 `AffineTransform(scale=(opt.scale_min, 1.15))` 配默认 `scale_min=1.0`,
即 scale ∈ [1.0, 1.15] —— **只放大、从不缩小** (代码级确认见第二节缺陷 5)。
而验证集有 **17.1% 的目标 <7px**。这条与第十一节的损失函数问题是**正交的**,
应作为独立 A/B 项测试: `--scale_min 0.8`。

### 12.2 用户后处理不是差距来源

`manu/evaluation/heatmap_evaluate.py::extract_peaks` 用的是标准 3x3 maxpool NMS
(`conf_thresh=0.20, top_k=100`), 与 `sweep_eval._peaks_topk` 方式基本一致。
所以**差距不在提峰/后处理, 而在训练目标**。

---

## 十三、G/H 受控 A/B 中间结果 (2026-09-30 17:00)

同协议: seed 42, val = 随机抽样 8000 (val_seed 0, 11/11 段, <7px 占 ~17%),
tp_distance 8, heat_stride 4, lr 0.005, nEpochs 10, eval_interval 2, sweep 0.05~0.80。
**唯一变量是损失函数** (外加 cntfocal 必需的 `--out_sigmoid`)。

| epoch | G_cntfocal F1 | (P / R) | H_kp_ctrl F1 | (P / R) | ΔF1 |
|--:|--:|:--|--:|:--|--:|
| 0 | 0.3369 | 0.3663 / **0.3119** | 0.0841 | 0.2204 / **0.0519** | +0.253 |
| 2 | **0.5609** | 0.7111 / **0.4630** | 0.4923 | 0.8053 / **0.3545** | +0.069 |

### 13.1 方向与机制预测一致, 但**尚不能定论**

* 预测: 换成 CenterNet focal 后**召回应显著上升**(梯度预算朝目标搬 24.8 倍)。
  实测 ep2 召回 0.4630 vs 0.3545, **+0.109**, 方向正确。
* 代价: 精度下降 0.8053 -> 0.7111 (-0.094), 这也是预期的(P/R 权衡沿 ROC 移动)。
* **但是 ΔF1 = +0.069 小于种子噪声 ±0.215**。单种子 A/B 在这个量级上**不构成证据**。

### 13.2 ep0 sweep 里的两条旁证

G_cntfocal (sigmoid 输出, 阈值有概率学意义) 在 ep0:
```
thr=0.3: P=0.3663 R=0.3119 F1=0.3369   <== 最优
thr=0.5: P=0.6465 R=0.1662 F1=0.2644
thr=0.7: P=0.9662 R=0.0314 F1=0.0608
```
**thr=0.7 时 P=0.9662** -> 模型能产生高置信、良分离的预测, 排序能力不差,
瓶颈在"高置信预测的数量"即检出, 与 24.9% 漏检的诊断一致。

对照 H_kp_ctrl (无界 logit) 在 ep0:
```
thr=0.4: P=0.4178 R=0.0140 F1=0.0270
thr=0.5: P=0.0098 R=0.0002 F1=0.0003
thr>=0.6: 全部 0
```
**无界 logit + 固定阈值网格是病态组合**: ep0 时 logit 全部偏小, 网格 [0.05,0.50]
里只有最低的四档有效。这独立印证了"模型缺输出激活"这一条。

### 13.3 冒烟测试

`--loss cntfocal --out_sigmoid --scale_min 0.8` 组合命令已端到端跑通
(CPU, train_limit 200 / nEpochs 1), 下一个实验的命令可以直接用。

---

## 十四、收工交接 (2026-09-30 17:25, 国庆后继续)

### 14.1 全部实验已停止

`G_cntfocal` / `H_kp_ctrl` 在 ep4 计算中被 kill, **没有跑到 10 epoch**。
服务器上已无 SPIRE 相关进程。

### 14.2 已保存的产物 (服务器 `/tmp/pycharm_project_10ae9e2e/spire_runs/`)

| run | 内容 |
|---|---|
| `G_cntfocal/20260930_161337_.../` | ep0/ep2 的 val_result.txt + sweep_result.txt + checkpoint |
| `H_kp_ctrl/20260930_161837_.../` | 同上 (kp 对照) |
| `E_full_raw/` | 旧 run, ep0/ep4 产物 + `halfsplit_ep4_n2000.json` (**诚实随机 val 复评结果**) |
| `F_full_gmcm/` | 旧 run, ep0/ep4 产物 |

**`E_full_raw/*/halfsplit_ep4_n2000.json` 是目前最有价值的单个文件**: 修正后协议的
诚实基线 F1=0.5775 + 完整宽阈值曲线。

### 14.3 国庆后第一件事: 多种子 (不要跳过)

单种子 ΔF1=+0.069 落在种子噪声 ±0.215 内, **目前还不能说 cntfocal 有效**。

```bash
# 脚本已写好: SPIRE-IRSTD/tools/run_multiseed.sh <GPU> <cntfocal|kp> <seed>
# 4 epoch 足够看出次序, 3 seed x 2 arm = 6 次, 两张卡约 2.5 小时
ssh -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_10ae9e2e/SPIRE-IRSTD
for s in 1 2 3; do
  bash tools/run_multiseed.sh 0 cntfocal $s
  bash tools/run_multiseed.sh 1 kp        $s
done
```
汇报口径: 每个 arm 给 mean ± std, 再报 ΔF1 与其标准误。**只有 |Δ| > 2·SE 才算结论。**

### 14.4 之后依次做

2. `--scale_min 0.8` 单独 A/B (默认 `random.uniform(1.0, 1.15)` 只放大不缩小, 验证集 17% 是 <7px)
3. lr 扫描 (用户 7.18e-4 vs SPIRE 0.005, 差 7 倍)
4. 若仍不足: `--heat_stride 2`、energy-preserving/peak-pool 变体

### 14.5 三条必须遵守的实验纪律 (本轮踩过的坑)

1. **任何 `--val_limit` 都是随机抽样**, 不是前缀切片。历史数字 (0.7521 等) 全部作废。
2. **报告数字必须带 val 子集统计** (`tools/eval_thr_halfsplit.py` 会自动打印 `n_seq` 与
   目标尺寸分位数)。曾因 `[:200]` 抽到单段视频而误以为代码有 bug, 实际 F1=0.9925。
3. **单种子 A/B 一律不下结论**, 种子噪声 ±0.215。

### 14.6 memory 文件的同步陷阱 (本次新踩)

`/media/manu/1TB-Volume/workspace/ultralytics/manu/memory/` 与
`/tmp/pycharm_project_10ae9e2e/manu/memory/` 是**双向同步**的。
**只在一侧新建的文件, 会被对端的同步当成"删除"传播掉。**
本次 `spire_antiuav_debug.md` 就是这样丢的 (服务器 17:04 同步 -> 本地被删)。
写 memory 务必**两边都写**, 或只写在已有文件里。
