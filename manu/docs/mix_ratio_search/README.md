# 混合比例超参搜索 · 运行手册

> 一句话：**用 Optuna 搜索 dataloader 混合比例（口径 1 = Batch 内采样概率），
> 每个 trial 3 epoch、单 GPU、4 卡并行、后台常驻一晚，用三级漏斗判定标准选型。**

---

## 0. 明天上班第一件事：看状态（**不需要重跑任何评测**）

```bash
ssh -p 32222 huangzhe@192.168.99.40
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
cd /tmp/pycharm_project_10ae9e2e

python manu/training/status_mix_ratio_search.py          # 完整状态报告
python manu/training/status_mix_ratio_search.py --top 20 # 多看几个
tail -f runs/optuna_mix_ratio/logs/dispatcher.log        # 实时日志
```

### 0.1 为什么不用重跑 val

**每个 trial 训练结束时，三轨评测已经自动跑完并全部落盘**（worker 内部串行调用门禁脚本）：

```
[GATE] 三轨门禁（w3.0 / epoch3）…  →  [GATE] rc=0 用时 1145.5s
[JUDGE] PASS | U(vs 冻结基线)=0.015099 | U(vs RC1)=-0.012480
[RESULT] .../trials/trial_0002/trial_result.json
```

每个 trial 的产物（`trials/trial_XXXX/`）：

| 路径 | 内容 |
|:--|:--|
| `weights/epoch{1,2,3}.pt` | **三个 epoch 权重全留**（只门禁最后一个） |
| `gate/epoch3/gate_report.json` | 三轨完整指标（判定的唯一真相源） |
| `gate/epoch3/track1_ab.json` / `track2_golden.json` / `track3_blind.json` | 三轨各自的明细 |
| `trial_result.json` | U / 红线判定 / 实测混合比例 |
| `logs/train_log.csv` | 逐 iteration 的 loss 与混样计数 |

汇总在 `optuna_summary.csv`（人读）与 `study.db`（Optuna 读）里，**明早直接读，不用重算**。

### 0.2 断电/关窗口安全（已核实）

| 风险项 | 结论 |
|:--|:--|
| 本机（宿主机）断电 / SSH 窗口关闭 | ✅ **完全不影响**。dispatcher 是 `setsid nohup` 独立 session，**PPID=1（已被 init 收养）、无 TTY**，不依赖任何 SSH 连接 |
| 服务器重启 | ⚠️ dispatcher 进程会没，但**数据大概率还在**：`/tmp` 在 **ext4 磁盘 `/dev/nvme0n1p2`（3.4T）上，不是 tmpfs**，重启不自动清空。万一真丢了，按 §1 的"恢复"命令断点续跑（已完成/已剪枝的 trial 不重跑） |
| 产物丢失 | ✅ 已额外备份到**另一块物理盘**：`/mnt/data/siping/backup/mix_ratio_search_20261021/`（490MB，含 `study.db` 与全部 trial 产物） |

**万一 dispatcher 死了（状态报告第 [1] 段显示"❌ 已退出"）**：
```bash
bash manu/training/launch_mix_ratio_search.sh    # 断点续跑，已完成的 trial 不重跑
```

### 0.3 ⚠️ 读结果的两个铁律（今晚新发现，务必先看）

**① 口径分界：`trial_0008` 起 cosine 周期改为 5 epoch（与 RC1 同口径）**

RC1/Ep3 是"路线 B 跑 5 epoch"里的第 3 个 epoch 权重，它经历的是 **5 epoch 全长的 cosine 曲线**
（Ep3 结束时 lr = 1.8264e-06 = 峰值的 60.9%）。
而 `trial_0000~0007` 用的是 **3 epoch 自己的 cosine**（Ep3 结束时 lr 已退到 1.0000e-06 = 33.3%）
⇒ **这 8 个 trial 的末 epoch 权重系统性比 RC1 "学得少"，与 mix_spec 效应混在一起、无法归因。**
`trial_0008` 起已修正（worker `--sched-epochs`，默认 5）。

⇒ **明早只把 trial_0008+ 当作可比数据**；前 8 个只作"链路是否健康"的证据。
⇒ **强 sanity check**：新口径下 `natural` 档应**精确复现 RC1**（T3≈0.8717 / T1FP≈941 / T2FP≈31 / U≈0.0276）。
   若复现成功 ⇒ 口径正确；若明显偏低 ⇒ 口径仍有问题，立刻找我。

**② 噪声远大于档位差异 ⇒ 不能用单个 trial 下结论**

| 量 | 实测噪声 |
|:--|:--|
| T2 虚警 FP | **≈ ±5**（红线 `ΔFP_T2 ≤ +10` 正切在噪声带中部，**会误杀 3/4 的好配置**） |
| T3 F1（同配置） | **≈ ±0.008**（trial_0000 与 RC1 同配置却差 0.007598） |
| U（同配置） | **≈ ±0.009** |

⇒ **判"打败 RC1（U > 0.027579）"需 U ≳ 0.045**（+2σ）才算显著；
⇒ 或用**同一 mix_spec 多个 trial 的中位数**（每档 8~9 个的意义所在）——
   状态报告的「按混合比例分组」表就是为此设计，**它已纳入全部档位（含被剪枝的）并标注 ⛔/✅/🟡**。

**③ ⭐ RC1 自己相邻 epoch 就在大幅波动 ⇒ 别拿单点当"应该复现到的值"**

RC1 的 6 点曲线（`runs/mixed_ft_route_b/RC1/RC1.json` → `full_cycle_curve`）：

| epoch | T1 F1 | T1 FP | T2 F1 | T2 FP | T3 F1 | T3 FP |
|:--|--:|--:|--:|--:|--:|--:|
| 1 | 0.902215 | 1232 | 0.998769 | 35 | 0.846895 | 589 |
| 2 | 0.901927 | 988 | 0.998889 | 33 | 0.859587 | 531 |
| **3 = RC1** | **0.901918** | **941** | **0.998979** | **31** | **0.871684** | **461** |
| 4 | 0.901796 | 1057 | 0.998859 | 35 | 0.872648 | 517 |
| 5 | 0.900501 | 1023 | 0.998949 | 33 | 0.874924 | 502 |

⇒ **同一个 run 内**，T3 F1 摆动 **0.8469~0.8749（跨度 0.028）**、T1 FP 摆动 **941~1232（跨度 291）**。
⇒ **trial_0000(natural) 与 RC1/Ep3 的差（T3 F1 −0.0076 / T1 F1 +0.0005 / T2 FP +6）全部落在这个包络内**
   ⇒ **"相同配置复现不了 RC1"是判据错觉，不是管线有未发现的差异**。
⇒ **明早第一件事：看 `#12` 复现对照**（natural + RC1 原值超参 + cosine 周期 5ep）。
   它落回 T3≈0.8717 / T1FP≈941 / T2FP≈31 ⇒ 管线验证通过；
   它明显偏低 ⇒ 口径仍有问题，立刻找我，别急着读比例结论。

**④ 两个口径分界都要标（互相独立）**

| 分界 | 范围 | 含义 |
|:--|:--|:--|
| **调度口径** | `trial_0008` 起 | cosine 周期 5 epoch（与 RC1 同口径）；之前是 3 epoch，末 epoch lr 只有 RC1 的 55% |
| **红线口径** | `trial_0011` 起 | 一级红线用**双通道**（严格 + 受控放宽）；`trial_0010` 及以前都是**严格单通道**判定 |

⚠️ 两者**独立**：一个 trial 可能"新调度 + 旧红线"（0008~0010）。读表时别混。

---

## 1. 常用操作

| 操作 | 命令 | 说明 |
|:--|:--|:--|
| 启动 | `bash manu/training/launch_mix_ratio_search.sh` | 后台常驻；`MAX_HOURS=8` 可改时长 |
| 查看 | `python manu/training/status_mix_ratio_search.py` | 随时可查 |
| 停止 | `bash manu/training/stop_mix_ratio_search.sh` | 优雅（等在跑的收尾） |
| 强停 | `bash manu/training/stop_mix_ratio_search.sh --force` | 立刻杀整组 |
| **继续** | `bash manu/training/launch_mix_ratio_search.sh` | **断点续跑**：已完成的 trial 全部跳过 |
| 单 trial 日志 | `tail -100 runs/optuna_mix_ratio/logs/trial_0007.log` | |
| 调度器日志 | `tail -100 runs/optuna_mix_ratio/logs/dispatcher.log` | |
| **投放对照 trial** | `python manu/training/enqueue_control_trials.py --specs natural,w4.0` | 入队后调度器**自动**取走，无需重启搜索 |
| 看对照队列 | `python manu/training/enqueue_control_trials.py --list` | 列出 WAITING 状态的 trial |

**宿主机断电 / SSH 窗口关闭都不影响搜索**：调度器用 `setsid nohup` 独立 session 运行，
不随终端关闭而终止；所有状态在服务器磁盘上（`runs/optuna_mix_ratio/`）。

---

## 2. 判别标准：三级漏斗（`manu/training/mix_ratio_judgement.py`）

**绝不用加权求和**——那会用顺风轨的大涨稀释致命缺陷。

```text
候选 M vs 对手 B
  ├─【一级】安全红线一票否决（相对**冻结基线**的绝对红线）—— **双通道**
  │      · 严格通道：T2 F1 ≥ 0.985  且  ΔFP_T2 ≤ +10  且  ΔF1_T1 ≥ −0.005
  │      · 受控放宽：T2 F1 ≥ 0.985  且  ΔFP_T2 ≤ +15  且  ΔF1_T2 ≥ +0.008
  │                   且  ΔF1_T3 ≥ +0.020  且  ΔF1_T1 ≥ −0.008
  │        （额外**要求黄金池与难例有实质增益**，非无条件放行；过线标 `PASS_RELAXED`）
  │      触碰任一条 ⇒ 剪枝 Pruned（不进二级、不参与 TPE 采样）
  │      `--strict-only` / `allow_relaxed=False` 可一键退回严格口径
  ├─【二级】帕累托占优（相对对手 B）：三轨主指标↑ + 虚警指标↓ 全向不劣且至少一维严格更优
  │      ⇒ 无争议判定 M ≻ B（**不再看效用分**）
  └─【三级】效用裁决（仅当二级不成立，即互有攻守）
         U = ΔF1_T3 + 0.5·ΔF1_T2 − 2.5·max(0,−ΔF1_T1) − 0.0005·max(0,ΔFP_T2)
```

| 系数 | 值 | 物理含义 |
|:--|--:|:--|
| `ΔF1_T3` | 1.0 | 难例突破，微调主攻目标 |
| `α` | 0.5 | 黄金池巩固（已在 0.988+ 饱和区，涨点只给适度加分） |
| `β` | 2.5 | 源域惩罚倍率：源域每掉 0.1pp，难例轨要涨 0.25pp 才拉平 |
| `γ` | 0.0005 | 黄金池每新增 1 个 FP 扣 0.0005 分 |

### 2.1 两个实现层面的关键判定（已用真实门禁数据固化，自测 60/60）

**① 效用分非反对称 ⇒ 三级必须比"同一基准下的 U"**

`max(0,·)` 惩罚项使 `U(B vs A) ≠ −U(A vs B)`。实测：
`U(Ep3 vs Ep4) = −0.000903`，`U(Ep4 vs Ep3) = −0.001097`（**同号**）。
若按逐对比较会得出错误结论；正确做法是比**同基准（冻结基线）**的 U：
`U(Ep3) = 0.027579 > U(Ep4) = 0.026178` ⇒ **Ep3 胜**。
⇒ 搜索 objective 与最终选型用的是**同一个尺度**，结论自洽。

**② 帕累托维度集必须显式可配（否则"无争议占优"会悄悄变味）**

| 预设 | 维度 | Ep3 ≻ Ep4？ |
|:--|:--|:--|
| `user` | T1_F1↑, T2_F1↑, T3_FP↓ | ✅ 成立（= 用户例句的隐含维度） |
| `core` | T1_F1↑, T2_F1↑, T3_F1↑ | ❌ 不成立（Ep3 的 T3-F1 实际**低** 0.001） |
| `full` | 上述六维（含三轨 FP） | ❌ 不成立 |

⇒ **两条路径殊途同归**：`user` 预设下 Ep3 严格占优；`full` 预设下 Ep1~Ep5 互不占优
（三轨 F1 最优分散在 Ep1/Ep3/Ep5），但**效用分裁决同样判 Ep3 胜**。
⇒ **RC1 = Ep3 的"最优"地位在两套独立判据下都被证实**。
本工具默认用 `full`（最保守），报告里同时给出三个预设的判定。

### 2.2 另一条实证：Ep5 过不了一级红线

Ep5 的 `ΔF1_T1 = −0.005236 < −0.005` ⇒ **直接被红线剔除**。
用户"锁定 Ep3 而非 Ep5"的结论，现在有了**红线级**理由（不只是边际收益递减）。

---

## 3. "真正的 1:4"：四种口径与本搜索采用的口径

| 口径 | 定义 | 1:4 成立？ |
|:--|:--|:--|
| **口径1 · Batch 内采样概率** | P(LQ)=0.20 / P(UAV)=0.80 | ✅ **本搜索采用** |
| 口径2 · 数据集总帧数 | 两边各遍历一次 | ❌ 物理不成立：13,918×4 = 55,672 > 43,008，实际极限 **1:3.09** |
| 口径3 · GT 目标框数 | 12,120 : 40,761 | 1:3.36 |
| 口径4 · 局部切片张数 | 路线 A 之后才成立 | — |

**实现**：`WeightedRandomSampler`（零数据复制、零磁盘开销），
且**只替换官方 `build_dataloader` 管线里的索引采样器**——
数据集/BatchSampler/collate_fn/worker/generator 全部是官方实现，
杜绝 §十四「自写 DataLoader 致样本腰斩 + 凭空多 576 处假警」的复现。

**正式运行第 1 个 epoch 的全量实测**（56,926 样本，2026-10-21 16:41，服务器真实日志）：

| trial | mix_spec | 目标 | **全 epoch 实测** | 误差 |
|:--|:--|:--|:--|--:|
| trial_0000 | `natural` | 1:3.090 | LQ 13,918 / UAV 43,008 = **1:3.090** | 0.0% |
| trial_0001 | `w2.0` | 1:2.000 | LQ 18,742 / UAV 38,184 = **1:2.037** | +1.9% |
| trial_0002 | `w3.0` | 1:3.000 | LQ 14,261 / UAV 42,665 = **1:2.992** | −0.3% |
| trial_0003 | **`w4.0`** | 1:4.000 | LQ 11,362 / UAV 45,564 = **1:4.010** | **+0.25%** ✅ |

⇒ **"真正的 1:4" 已在正式运行中得到确认**（口径 1：Batch 内采样概率 P(LQ)=0.20 / P(UAV)=0.80）。

### 3.1 ⚠️ 解读必读：加权采样会顺带降低龙泉山帧的每 epoch 覆盖率
口径 1 的"每 epoch 总抽样量固定 = 56,926"意味着：**比例越高，龙泉山每个 epoch 能见到的**不同**帧越少**。
（公式：期望覆盖率 ≈ 1 − exp(−抽次 / 唯一帧数)）

| mix_spec | P(LQ) | LQ 抽次/epoch | **LQ 期望覆盖** | UAV 抽次/epoch | UAV 期望覆盖 |
|:--|--:|--:|--:|--:|--:|
| `natural`（对照） | 0.245 | 13,918 | **100%** | 43,008 | **100%** |
| w1.0 | 0.500 | 28,463 | 87.1% | 28,463 | 48.4% |
| w2.0 | 0.333 | 18,975 | 74.4% | 37,951 | 58.6% |
| w3.0 | 0.250 | 14,232 | 64.0% | 42,694 | 62.9% |
| **w4.0** | 0.200 | 11,385 | **55.9%** | 45,541 | 65.3% |
| w6.0 | 0.143 | 8,132 | 44.3% | 48,794 | 67.8% |
| w8.0 | 0.111 | 6,325 | 36.5% | 50,601 | 69.2% |

⇒ **归因警示**：高比例下源域"变好"可能来自**两个不同机制**——
  (a) Anti-UAV 梯度占比变大（口径 1 的本意）；
  (b) 龙泉山帧被见得更少 ⇒ 目标域漂移更小（**副作用**）。
⇒ 解读 w4.0/w6.0 的结果时，必须同时看 Track3（难例）是否同步下滑：
  **若 T3 也降 ⇒ 那是"少学"而非"学得更好"**，不能算真正的 1:4 收益。
  ⇒ 这正是三级漏斗把 `ΔF1_T3` 放在效用分主项（权重 1.0）、并把源域损失按 2.5 倍惩罚的原因：
  **只涨源域、不涨难例的 trial，拿不到高 U，会被正确地排到后面。**

---

## 4. 搜索空间

| 参数 | 类型 | 取值 | 说明 |
|:--|:--|:--|:--|
| **`mix_spec`** | categorical | `natural`（对照=路线 B 自然 1:3.09）、`w1.0/1.5/2.0/2.5/3.0/3.5/4.0/5.0/6.0/8.0` | ★**关键参数**：口径1 的 1:N |
| `lr_scale` | float log | 0.3 ~ 3.0 | 路线 B 三组 lr（3e-6/1.5e-5/3e-5）的整体倍率 |
| `warmup_epochs` | float | 0.2 ~ 1.0 | |
| `weight_decay` | float log | 1e-5 ~ 5e-4 | |
| `focal_beta` | float | 2.2 ~ 2.9 | |
| `offset_weight` | float | 0.40 ~ 0.70 | |
| `mosaic` | float | 0.0 ~ 0.15 | |
| `scale` | float | 0.10 ~ 0.30 | |
| `translate` | float | 0.05 ~ 0.15 | |
| `clip_max_norm` | float | 2.0 ~ 8.0 | |
| `freeze_bn` | categorical | True/False | ★用户点名的下一轮方向 |

**冻结不变**（路线 B 已验证的结构）：G1 严格冻结（Stem/P1、Stage1/P2、Stage2/P3）+ eval 保护 BN、
三组 lr 比例 1:5:10、wd 基准 1e-4、只保留弱抖动、禁色彩与极性翻转、grad clip、AMP。
**种子 trial**（自动注入，保证头几小时就有信号）：`natural` 对照 + `w2.0` / `w3.0` / `w4.0` / `w6.0`。

---

## 5. 产物路径

| 内容 | 路径（服务器 `/tmp/pycharm_project_10ae9e2e/`） |
|:--|:--|
| 搜索根 | `runs/optuna_mix_ratio/` |
| Optuna 库（**断点续跑的依据**） | `runs/optuna_mix_ratio/study.db` |
| 汇总表 | `runs/optuna_mix_ratio/optuna_summary.csv` |
| 每次调度心跳 | `runs/optuna_mix_ratio/status.json` |
| 在跑 worker 登记 | `runs/optuna_mix_ratio/workers.json` |
| 调度器日志 | `runs/optuna_mix_ratio/logs/dispatcher.log` |
| 单 trial 日志 | `runs/optuna_mix_ratio/logs/trial_XXXX.log` |
| 单 trial 产物 | `runs/optuna_mix_ratio/trials/trial_XXXX/{weights/,gate/,logs/,trial_result.json}` |
| **现任擂主 RC1** | `runs/mixed_ft_route_b/RC1/epoch3.pt`（md5 `a845a95b77ac8762b7d033b64f61425e`） |

---

## 6. 选型方法（搜索结束后）

```bash
# 状态报告里已含决赛结论；也可单独跑：
python manu/training/status_mix_ratio_search.py --out-json runs/optuna_mix_ratio/final_selection.json

# 两个候选正面比较（三级漏斗）
python manu/training/mix_ratio_judgement.py \
  --candidate runs/optuna_mix_ratio/trials/trial_0007/gate/epoch3/gate_report.json \
  --baseline  runs/mixed_ft_route_b/gate/epoch3/gate_report.json
```

**及格线**：候选的效用分必须 > **0.027579**（RC1/Ep3 相对冻结基线的 U），
或在与 RC1 的正面对决中判为 `A_DOMINATES` / `UTILITY_A`。

---

## 7. 故障排查

| 现象 | 处理 |
|:--|:--|
| 状态显示"已退出" | 重跑 `launch_mix_ratio_search.sh`（自动跳过已完成 trial，陈旧 RUNNING 自愈为 FAIL） |
| 某 trial `CRASHED` | 看 `runs/optuna_mix_ratio/logs/trial_XXXX.log` 尾部 |
| 某 trial `PRUNED` | **正常**：触了一级红线（源域跌超 0.5pp 或黄金池虚警爆表） |
| 显存不足 | 降 `--batch`（默认 32，实测峰值 11.30GB/24GB） |
| 门禁报基础设施失败（rc=3） | 看 `trials/trial_XXXX/gate/epoch3/` 下的日志；属故障不是指标问题 |
| 门禁 Track1 CUDA OOM（报告里 `track1_f1` 缺失） | **别重跑整个门禁**：用 `--skip-longquan` 只补跑 Track1，再 `merge_gate_report.py` 合并（见 §7.2） |
| 分组表里某档只有 1 个点 | 单 trial 判定不可靠（见 §7.3），用 `status_mix_ratio_search.py` 的分组表看该档多点中位数 |

### 7.1 ⚠️ 上线前踩过的三个致命坑（已修复，勿重蹈）

| # | 坑 | 后果 | 现状 |
|:--|:--|:--|:--|
| 1 | `args.max-hours` 写在 f-string 里（被解析成 `args.max - hours`） | dispatcher **启动即崩**，20 小时搜索**静默空跑一个 trial 都不跑**；`launch` 脚本只看 `kill -0`，查不出来 | 已改 `args.max_hours` |
| 2 | 门禁 `--max-track2-fp` 是**绝对上限**（默认 36 = 基线 26 + 增量 10），却误传增量 `10` | `gate_report.json` 里 `veto_failures` 全是假的（实测 FP=13、ΔFP=−13 远优于红线仍判 FAIL） | 已改传 `GATE_TRACK2_FP_ABS_MAX=36`；`--min-track1-f1` 同理改 `0.900736` |
| 3 | Optuna RDB 的 `trial_id` 是 **1 起**主键、`Trial.number` 是 **0 起**业务编号，**差 1**；且 `get_trials(deepcopy=False)` 返回 **FrozenTrial** | 断点续跑时自愈逻辑 `set_trial_state_values(t.number, ...)` **改错行** ⇒ `UpdateFinishedTrialError` ⇒ **续跑必崩** | 已改用公共 API `study.tell(t.number, state=FAIL)` |

**防呆**：新增调度器离线自测，改完调度器就跑一次（几十秒，不必等真实 trial）：

```bash
python manu/training/test_optuna_dispatcher.py     # 期望：✅ 25/25
```

### 7.2 门禁 Track1 失败后的补测与合并（实测走过一次）

**背景**：Track1 Anti-UAV 推理最吃显存；若与正在训练的 trial 同卡会 CUDA OOM
（实测：`GPU 3; 23.55 GiB total; 310.56 MiB free`），报告里 `track1_f1` 缺失。
**但 Track2/Track3 与 val 缓存通常已完成 ⇒ 只补跑 Track1 即可**（424.9s，而非全门禁 20min）。

```bash
# 1) 只补跑 Track1（--skip-longquan 跳过龙泉山两轨；batch 调小以省显存）
python manu/training/auto_epoch_gate_check.py \
  --weights runs/optuna_mix_ratio/trials/trial_XXXX/weights/epoch3.pt \
  --out-dir runs/optuna_mix_ratio/trials/trial_XXXX/gate/epoch3_retry \
  --gpus 1 --device 3 --infer-gpu 3 --cache-gpus 3 \
  --baseline-cache runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl \
  --skip-longquan --infer-batch 4

# 2) 合并回主报告（残缺原件自动另存 gate_report.incomplete.json 留档）
python manu/training/merge_gate_report.py \
  --main  runs/optuna_mix_ratio/trials/trial_XXXX/gate/epoch3/gate_report.json \
  --retry runs/optuna_mix_ratio/trials/trial_XXXX/gate/epoch3_retry/gate_report.json
```

⚠️ **铁律：探针/任何评估任务不得与正在训练的 trial 同卡**；单卡同时只允许"训练"或"评估"二者之一。

### 7.3 ⚠️ 一级红线的 "ΔFP_T2 ≤ +10" **在单 trial 上不可用**（实测 4 个种子）

| mix_spec | T2 F1 | **T2 FP** | ΔFP_T2 | 红线 | T3 F1 |
|:--|:--|--:|--:|:--|:--|
| natural 1:3.09 | 0.998769 | **37** | +11 | ❌ | 0.864086 |
| w2.0 1:1.99 | 0.998739 | **37** | +11 | ❌ | 0.865463 |
| w3.0 1:2.99 | 0.998919 | **32** | +6 | ✅ | 0.861058 |
| **w4.0 1:3.99** | 0.998799 | **37** | +11 | ❌ | 0.862080 |

**4 个里 3 个只差 1 个虚警**，而 **T2 F1 全部 ≥ 0.9987**（黄金池从 98.84% 提到 99.87~99.89%）。
⇒ **T2 FP 的单 trial 噪声 ≈ ±5，红线切在噪声带正中部 ⇒ 单 trial 判定会误杀 3/4 的好配置。**

⇒ **读结果铁律**：
1. **主看「按 mix_spec 分组」表，同一档取多个 trial 的中位数**；
2. **任何单 trial 的红线判定都只是一个样本，不能用来否定一个比例**；
3. U 值差异同样在噪声内（实测极差仅 0.004）⇒ **单批 4 个 trial 无法区分比例优劣**。

### 7.3.1 ✅ 处置结果：一级红线改为**双通道**（用户授权"按实测数据分布做调整"）

| 通道 | 判据 |
|:--|:--|
| 严格（原样） | T2 F1 ≥ 0.985 且 ΔFP_T2 ≤ **+10** 且 ΔF1_T1 ≥ **−0.005** |
| **受控放宽**（新） | T2 F1 ≥ 0.985 且 ΔFP_T2 ≤ **+15** 且 ΔF1_T2 ≥ **+0.008** 且 ΔF1_T3 ≥ **+0.020** 且 ΔF1_T1 ≥ **−0.008** |

**校准依据（第一批 4 个种子 trial，同一批门禁）**：

| 观测量 | 实测分布 | 推论 |
|:--|:--|:--|
| T2 FP 混训后正常水平 | 32 / 37 / 37 / 37 ⇒ Δ = **+6~+11** | 原红线 +10 **正切噪声带中部** |
| T2 FP 单 trial 噪声 | ≈ **±5** | ⇒ **误杀 3/4 种子** |
| T2 F1（同批全部） | ≥ **0.9987**（Δ≥+0.0103，FN 355→6） | 黄金池远在红线之上，F1 不是瓶颈 |
| ΔF1_T1 | **[−0.005417, −0.003281]**（RC1 自身 −0.003818） | 严格 −0.005 切在分布下沿 |
| ΔF1_T3 | **[+0.023727, +0.028133]** | 放宽取 +0.020 ⇒ 仍要求实质增益 |

⇒ **旧口径 4 个种子：严格 1/4 过 ⇒ 放宽 4/4 过**。
⇒ 放宽通道**不是无条件放行**：它额外要求黄金池(T2)与难例(T3)有实质增益。
⇒ 过线显式标 **`PASS_RELAXED`**（状态报告里显示为 **🟡过线(放宽)**），**`--strict-only` 可一键回退**。
⇒ 代码：`manu/training/mix_ratio_judgement.py` 的 `DEFAULT_RELAXED_GATE` / `allow_relaxed`；自测 **60/60**（服务器实测）。

### 7.4 实测耗时（用于推算还剩多少 trial）

| 环节 | 实测 | 依据 |
|:--|:--|:--|
| 训练 3 epoch | **≈ 28 min** | 3.16 iter/s × 1779 iter × 3 |
| 单卡门禁 | **18.9 min** | `gate_report.json` → `elapsed_seconds=1131.7`（其中 val 热图缓存重建 977.7s / 28,997 帧 / 3 尺度） |
| **单 trial** | **≈ 47 min** | 4 卡并行 ⇒ 20h ≈ **102 个 trial** |

---

## 8. 明天想继续搜索 / 扩大搜索空间时

> 当前决策（2026-10-21）：**搜索空间暂时够用，不扩大、代码不动**。本节只是备忘。

⚠️ **同一个 Optuna study 不能扩大搜索空间**（实测抛 `ValueError: CategoricalDistribution does not support dynamic value space.`）。
给 `mix_spec` 加新档位（如 w4.5 / w7.0）**不能**只改 `MIX_SPECS` 后续跑现有 `study.db`——会崩。

| 场景 | 做法 | 状态 |
|:--|:--|:--|
| **纯续跑**（不扩空间） | 直接 `bash manu/training/launch_mix_ratio_search.sh`（`load_if_exists=True`，新编号接续，已完成的不重跑） | ✅ 实测通过 |
| **扩搜索空间** | **新建 study**（换 `--study-name`）+ 用 `enqueue_trial()` 注入第一阶段 top-K 的参数，让 TPE 从好点起步 | ✅ 实测通过 |
| **某档样本不足** | 单卡手动补跑该档，不需要重启搜索 | ✅ 命令见下 |

补跑单档（执行机：服务器）：

```bash
python -u manu/training/optuna_mix_ratio_search.py --worker --trial-number 9001 --gpu-id 0 \
  --output-root runs/optuna_mix_ratio \
  --longquan-data /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/data.yaml \
  --uav-data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
  --weights runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --baseline-cache runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl \
  --epochs 3 --batch 32 --mix-spec w6.0 --lr-scale 1.0 --warmup-epochs 0.5 \
  --weight-decay 1e-4 --focal-beta 2.7 --offset-weight 0.55 --mosaic 0.05 --scale 0.20 \
  --translate 0.10 --clip-max-norm 5.0 --freeze-bn 0
```

### 8.1 ⚠️ 搜索出"最优比例"之后还差一步

**3 epoch ≠ 5 epoch**：RC1 是 **5 epoch 训练**里选出的 Ep3，而搜索每 trial 只训 **3 epoch**；
高比例档（w6/w8）的龙泉山帧覆盖率差异会**随 epoch 累积放大**。
⇒ **搜索的真正产出是"选比例"**；最终模型应当用选出的比例**再跑一次 5 epoch 正式训练**，再与 RC1 同口径比。

```bash
# 5 epoch 正式训练（用选出的比例）
python -u manu/training/optuna_mix_ratio_search.py --worker --trial-number 9002 --gpu-id 0 \
  --output-root runs/optuna_mix_ratio_final --epochs 5 --mix-spec <选出的档> ... （其余同上）
```

---

## 9. 相关文件

| 文件 | 作用 |
|:--|:--|
| `manu/training/optuna_mix_ratio_search.py` | 搜索主体（调度器 + worker） |
| `manu/training/mix_ratio_judgement.py` | 三级漏斗判别标准（可独立 CLI 用，**一级红线双通道**） |
| `manu/training/enqueue_control_trials.py` | **投放对照组 trial**（AST 解析搜索参数空间，不复制常量） |
| `manu/training/test_mix_ratio_judgement.py` | 判别标准自测（真实门禁数据，**60 项**） |
| `manu/training/test_optuna_dispatcher.py` | **调度器离线自测**（假 worker，26 项） |
| `manu/training/test_status_mix_ratio_search.py` | **状态报告自测**（假产物，**38 项**） |
| `manu/training/merge_gate_report.py` | 门禁**补测合并**（Track1 失败后单独补跑再并回主报告） |
| `manu/training/status_mix_ratio_search.py` | 状态查询 / 选型报告（**✅/🟡/⛔ 三档红线标记**、`--strict-only`） |
| `manu/training/launch_mix_ratio_search.sh` | 后台常驻启动 |
| `manu/training/stop_mix_ratio_search.sh` | 停止（可恢复） |
