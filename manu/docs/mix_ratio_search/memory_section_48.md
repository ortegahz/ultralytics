
# §四十八 多轨判别标准（三级漏斗）+ 混合比例超参搜索体系（2026-10-21）

## 1. 决策一句话

建立**可量化、可审计、可复用**的多轨判别标准（一级安全红线 → 二级帕累托占优 → 三级效用裁决），
并据此落地**混合比例 Optuna 超参搜索**（口径 1 = Batch 内采样概率），4 卡并行 / 3 epoch / trial /
后台常驻一晚 / 随时停止继续。判定标准自测 **43/43 通过**（用真实门禁数据固化）。

## 2. 为什么不能加权求和（用户铁律）

多轨（源域大盘 / 黄金池守门员 / 难例盲测）加权求和会**用顺风轨的大涨稀释致命缺陷**
（黄金池虚警爆炸、源域断崖下跌都会被别的轨"平均掉"）。
⇒ 正确范式是**漏斗**（顺序不可交换）：先否决、再占优、最后才用效用分排名。

## 3. 三级漏斗（实现：`manu/training/mix_ratio_judgement.py`）

```text
候选 M vs 对手 B
  ├─【一级】安全红线一票否决（相对**冻结基线**的绝对红线）
  │      · Track2 黄金池：F1 ≥ 0.985  且  ΔFP_T2 ≤ +10
  │      · Track1 源域  ：ΔF1_T1 ≥ −0.005
  │      触碰任一条 ⇒ 剪枝 Pruned（不进二级、不参与 TPE 采样）
  ├─【二级】帕累托占优（相对对手 B）⇒ 直接判胜，**不再看效用分**
  └─【三级】效用裁决（仅当二级不成立）
         U = ΔF1_T3 + 0.5·ΔF1_T2 − 2.5·max(0,−ΔF1_T1) − 0.0005·max(0,ΔFP_T2)
```

冻结基线（去 P0）：T1 F1 0.905736 / T2 F1 0.988436 FP 26 / T3 F1 0.837331。
**RC1(Ep3) 的 U = 0.027579 ⇒ 这是"更好"的及格线。**

### 3.1 ⚠️ 两个实现层面的关键判定（都是用真实门禁数据挖出来的）

**① 效用分**非反对称** ⇒ 三级必须比"同一基准下的 U"**

`max(0,·)` 惩罚项使 `U(B vs A) ≠ −U(A vs B)`。实测
`U(Ep3 vs Ep4) = −0.000903` 而 `U(Ep4 vs Ep3) = −0.001097`（**同号**）。
若按逐对比较会得出与 objective 不一致的结论；正确做法是比**同基准（冻结基线）**的 U：
`U(Ep3)=0.027579 > U(Ep4)=0.026178` ⇒ **Ep3 胜**。
⇒ 搜索 objective 与最终选型**同一尺度**，结论自洽。

**② 帕累托维度集必须显式可配**（否则"无争议占优"会悄悄变味）

| 预设 | 维度 | Ep3 ≻ Ep4？ |
|:--|:--|:--|
| `user` | T1_F1↑, T2_F1↑, T3_FP↓ | ✅ 成立（= 用户例句的隐含维度） |
| `core` | T1_F1↑, T2_F1↑, T3_F1↑ | ❌ 不成立（Ep3 的 T3-F1 实际**低** 0.001） |
| `full` | 上述六维（含三轨 FP） | ❌ 不成立 |

⇒ **两条路径殊途同归**：`user` 下 Ep3 严格占优；`full` 下 Ep1~Ep5 互不占优
（三轨 F1 最优分散在 Ep1/Ep3/Ep5），**效用分裁决同样判 Ep3 胜**。
⇒ **RC1 = Ep3 的"最优"地位在两套独立判据下都被证实**（这是对用户结论的严格化确认）。
工具默认 `full`（最保守），报告同时给三预设判定。

### 3.2 另一条实证：**Ep5 过不了一级红线**

Ep5 的 `ΔF1_T1 = −0.005236 < −0.005` ⇒ **直接被红线剔除**。
用户"锁定 Ep3 而非 Ep5"的结论，现在有了**红线级**理由（不只是边际收益递减）。

## 4. "真正的 1:4"：四种口径（必须统一到口径 1）

| 口径 | 定义 | 1:4 成立？ |
|:--|:--|:--|
| **口径1 · Batch 内采样概率** | P(LQ)=0.20 / P(UAV)=0.80 | ✅ **采用** |
| 口径2 · 数据集总帧数 | 两边各遍历一次 | ❌ 物理不成立：13,918×4 = 55,672 > 43,008，极限 **1:3.09** |
| 口径3 · GT 目标框数 | 12,120 : 40,761 | 1:3.36 |
| 口径4 · 局部切片张数 | 路线 A 之后才成立 | — |

**实现（关键工程点）**：`WeightedRandomSampler`，且**只替换官方 `build_dataloader` 管线里的索引采样器**
—— torch 1.13 的 `DataLoader.__init__` 在 `sampler is None and shuffle=True` 时用**模块级**
`RandomSampler`，故在调用窗口内替换该名字即可；数据集/BatchSampler/_RepeatSampler/collate_fn/
worker_init_fn/generator **全部官方**，杜绝 §十四「自写 DataLoader 致样本腰斩 + 凭空多 576 处假警」复现。
**冒烟实测：目标 1:4.000 → 实测 1:4.106** ✅；官方数据集满额 13,918 / 43,008 ✅；
warmup 890 步 = 0.5 ep ✅（公式正确）；VRAM 峰值 11.28 GB ✅。

### 4.1 ⚠️ 解读必读：加权采样会顺带降低龙泉山帧的每 epoch 覆盖率

口径 1「每 epoch 总抽样量固定 = 56,926」⇒ **比例越高，龙泉山每个 epoch 见到的不同帧越少**
（期望覆盖 ≈ 1 − exp(−抽次/唯一帧数)）：

| mix_spec | P(LQ) | LQ 抽次/epoch | **LQ 期望覆盖** | UAV 期望覆盖 |
|:--|--:|--:|--:|--:|
| `natural` | 0.245 | 13,918 | **100%** | **100%** |
| w2.0 | 0.333 | 18,975 | 74.4% | 58.6% |
| w3.0 | 0.250 | 14,232 | 64.0% | 62.9% |
| **w4.0** | 0.200 | 11,385 | **55.9%** | 65.3% |
| w6.0 | 0.143 | 8,132 | 44.3% | 67.8% |
| w8.0 | 0.111 | 6,325 | 36.5% | 69.2% |

⇒ **归因警示**：高比例下源域"变好"可能来自**两个不同机制**——
(a) Anti-UAV 梯度占比变大（口径 1 本意）；(b) 龙泉山帧被见得更少 ⇒ 目标域漂移更小（**副作用**）。
⇒ 解读 w4.0/w6.0 必须同时看 Track3：**若 T3 同步下滑 ⇒ 那是"少学"而非"学得更好"**。
⇒ 这正是效用分把 `ΔF1_T3` 放主项（权重 1.0）、并把源域损失按 2.5 倍惩罚的意义：
**只涨源域、不涨难例的 trial 拿不到高 U，会被正确地排到后面。**

## 5. 搜索体系（`manu/training/optuna_mix_ratio_search.py`）

* **4 卡并行**，每 trial **单 GPU**、**3 epoch**；门禁在同卡串行跑（只门禁最后一个 epoch，三个权重全留）
* **可随时停止 / 随时继续**：Optuna SQLite study 持久化 + 断点续训状态 + 陈旧 RUNNING trial 自愈为 FAIL
* **后台常驻**：`setsid nohup` 独立 session ⇒ 宿主机断电 / SSH 窗口关闭都不影响
* **状态可查**：`status_mix_ratio_search.py` 随时输出进度 / GPU / Top-N / **按 mix_spec 分组最优** / 决赛结论
* **有截止时间**：`--max-hours` 到点自动收尾（保证第二天早上有结果）
* **objective** = 效用分 U（相对冻结基线）；**一级红线失败 ⇒ `TrialState.PRUNED`**（不参与 TPE 采样）

### 5.1 搜索空间

| 参数 | 取值 | 说明 |
|:--|:--|:--|
| **`mix_spec`** | `natural`（对照=路线 B 自然 1:3.09）+ `w1.0/1.5/2.0/2.5/3.0/3.5/4.0/5.0/6.0/8.0` | ★**关键参数** |
| `lr_scale` | 0.3~3.0 (log) | 路线 B 三组 lr（3e-6/1.5e-5/3e-5）的整体倍率 |
| `warmup_epochs` | 0.2~1.0 | |
| `weight_decay` | 1e-5~5e-4 (log) | |
| `focal_beta` / `offset_weight` / `mosaic` / `scale` / `translate` | 见脚本 | 弱抖动族 |
| `clip_max_norm` | 2.0~8.0 | |
| `freeze_bn` | True/False | ★用户点名的下一轮方向 |

**冻结不变**：G1 严格冻结（Stem/P1、Stage1/P2、Stage2/P3）+ eval 保护 BN、三组 lr 比例 1:5:10、
wd 基准 1e-4、只保留弱抖动、禁色彩与极性翻转、AMP。
**种子 trial（自动注入，头几小时就有信号）**：`natural` 对照 + `w2.0`/`w3.0`/`w4.0`/`w6.0`。

### 5.2 产物与操作

| 内容 | 路径（服务器 `/tmp/pycharm_project_10ae9e2e/`） |
|:--|:--|
| 搜索根 | `runs/optuna_mix_ratio/` |
| Optuna 库（断点续跑依据） | `runs/optuna_mix_ratio/study.db` |
| 汇总表 / 心跳 / worker 登记 | `optuna_summary.csv` / `status.json` / `workers.json` |
| 日志 | `logs/dispatcher.log`、`logs/trial_XXXX.log` |
| 单 trial 产物 | `trials/trial_XXXX/{weights/,gate/,logs/,trial_result.json}` |

```bash
# 启动（后台常驻）
bash manu/training/launch_mix_ratio_search.sh          # MAX_HOURS=20 默认
# 查看（任何时候）
python manu/training/status_mix_ratio_search.py
# 停止（可恢复）
bash manu/training/stop_mix_ratio_search.sh            # 优雅
bash manu/training/stop_mix_ratio_search.sh --force    # 强杀
# 恢复
bash manu/training/launch_mix_ratio_search.sh
```

**运行手册**：`manu/docs/mix_ratio_search/README.md`（本地仓库与服务器同路径）。

## 6. 新增工具清单

| 文件 | 作用 |
|:--|:--|
| `manu/training/mix_ratio_judgement.py` | 三级漏斗判别标准（门禁 / 帕累托 / 效用 / 两两比较器 / 决赛选择器 + CLI） |
| `manu/training/test_mix_ratio_judgement.py` | 判别标准自测（真实门禁数据，43 项全绿） |
| `manu/training/optuna_mix_ratio_search.py` | 混合比例超参搜索（调度器 + worker） |
| `manu/training/status_mix_ratio_search.py` | 状态查询 / 选型报告 |
| `manu/training/launch_mix_ratio_search.sh` | 后台常驻启动（setsid+nohup） |
| `manu/training/stop_mix_ratio_search.sh` | 停止（优雅/强杀） |
