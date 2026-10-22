---
name: mixed-ft-orchestrator
description: |
  定义并调度「路线 B 全帧对等混训」（Route B full-frame equal-mix fine-tuning）的标准训练范式：
  冻结策略、混样口径（口径 1 = Batch 内采样概率）、cosine 退火周期绑定、正式训练命令、
  三轨门禁判别与产物持久化。
  When the user asks to 启动路线 B 混训 / 跑 mixed_ft / 复现 trial_0080 / 调混合比例 /
  换一批超参再搜 / 门禁不过要定位 / 把新场景并进混训，load this skill before running
  manu/training/ 下的训练脚本。
  Near-misses（这些不要用本 skill）：新场景数据接入、特征语义与数据划分规范化 →
  domain-data-curator；纯推理/评测脚本调参 → manu/evaluation/ 下的脚本；
  底模预训练与组件扩容（那属于铁律六约束的另一条路线）。
  判据只认已锁定的先验空间：配比 w2.0~w2.5 ＋ lr_scale 2.0~2.8 ＋ 骨干锁死，
  不要凭直觉重新开一轮宽搜索。
descriptions:
  zh-Hans: 路线 B 全帧对等混训的启动与调度标准：口径 1 混样、骨干锁死、退火周期绑定、正式训练命令与三轨门禁判别。
displayNames:
  zh-Hans: 路线 B 混训编排
---

# Mixed FT Orchestrator

## Inputs to collect

动手前确认下列每一项；缺项先问用户或读 `manu/memory/memory.md`，不要用默认值硬上。

- **本轮目的**：复现已上线配置（照抄 `trial_0080`）、在锁定先验空间内做小范围搜索、
  还是只补跑某个 epoch 的权重。三种目的的命令不同。
- **训练数据**：龙泉山 official
  `/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/data.yaml`
  与 Anti-UAV 大盘 `/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml`；
  换场景时两个 `--longquan-data` / `--uav-data` 都要换。
- **起始权重**：`runs/optuna_p0_nas/trial_0474/weights/best.pt`（P0 冻结基线，默认值）。
- **擂主参照系**：`rc1` = 路线 B 5-epoch 运行的第 3 epoch（`runs/mixed_ft_route_b/RC1/epoch3.pt`，
  md5 `a845a95b77ac8762b7d033b64f61425e`）；`baseline` = 冻结基线。
  正式上线判定用 `rc1`，因为同期新搜索的 epoch 数与 RC1 不同。
- **GPU 分配**：哪些卡空闲、哪些卡已被训练 trial 占用。
- **门禁量第几个 epoch**：`--gate-epoch 3`（量 Ep3，与 RC1 同尺度）或末 epoch。
- **持久化目标路径**：关键权重要另存 `/mnt/data`（见步骤 7）。

## Procedure

### 0. 本地优先 + 三侧 md5 同步（铁律〇）

代码永远**先改本地** `/media/manu/1TB-Volume/workspace/ultralytics/`，
再同步到服务器 `/tmp/pycharm_project_10ae9e2e`。
本机的 SSHFS 镜像 `/home/manu/mnt/pycharm_project_10ae9e2e` 是**写入即落服务器**的同一份代码，
但同步是否成功**只能用 md5 比对**判定，不要凭"文件应该在那儿"的直觉。

```bash
# 本地（机器 A）
cd /media/manu/1TB-Volume/workspace/ultralytics
md5sum manu/training/optuna_mix_ratio_search.py

# SSHFS 镜像（机器 A，写入即落服务器）
md5sum /home/manu/mnt/pycharm_project_10ae9e2e/manu/training/optuna_mix_ratio_search.py

# 服务器（机器 B）
ssh -p 32222 huangzhe@192.168.99.40 \
  "md5sum /tmp/pycharm_project_10ae9e2e/manu/training/optuna_mix_ratio_search.py"
```

三行 md5 必须完全一致。不一致就用 SSHFS 镜像重写一次再比对；
**不要**在服务器上直接改代码 —— 那样本地仓库和服务器会分叉，下次同步会把改动冲掉。

服务器环境（conda 环境名 `uav`，服务器上 `python` 无 3 后缀，Python 3.10.20）：

```bash
ssh -p 32222 huangzhe@192.168.99.40
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
```

### 1. 口径 1 = Batch 内采样概率（项目里"真正的 1:4"的唯一定义）

`mix_spec` 定义的是 **Batch 内的采样概率**，不是帧数比、不是数据集占比：

- `wN` → `P(LQ) = 1/(1+N)`、`P(UAV) = N/(1+N)`。例如 `w4.0` ⇒ P(LQ)=0.20 / P(UAV)=0.80。
- `natural` → 不加权，按数据集自然比例遍历（43008/13918 ≈ **1:3.09**）。

实现方式：用 `WeightedRandomSampler` **只替换官方 dataloader 的索引采样器**
（`manu/training/optuna_mix_ratio_search.py:150` 的 `weighted_sampler_patch`），
在调用官方 `build_dataloader` 的窗口内把模块级 `RandomSampler` 换成加权采样器。
数据集本身仍是官方 `build_yolo_dataset`（图片/标签扫描、缓存、mosaic 全官方），
`collate_fn` / batch_idx 语义不变，因此不会出现样本腰斩或凭空多假警。

每 epoch 总采样量保持 `n_lq + n_uav` 不变，这样各 trial 算力与耗时可比。

搜索空间：`mix_spec ∈ {natural, w1.0, w1.5, w2.0, w2.5, w3.0, w3.5, w4.0, w5.0, w6.0, w8.0}`
（`manu/training/optuna_mix_ratio_search.py:110`）。

**已证伪的说法**："1:4 最优"不成立。`w4.0` 实测全 epoch 抽样 **1:4.010**
（口径 1 抽样命中目标比例），但它不是最优配比。最优档在 `w2.5`。

运行日志里会打印两条关键行，用来核对口径与实测比例：

```text
[MIX] 口径1 加权采样：P(LQ)=0.2000 P(UAV)=0.8000 | 每 epoch 抽样 56,926 次（LQ 期望 11,385 / UAV 45,541）
[EPOCH 3] loss ... | 实测混样 LQ <实测计数> / UAV <实测计数> = 1:4.010（目标 1:4.000）
```

（`w4.0` 的全 epoch 实测抽样比例 **1:4.010** 是已记录值；
上面的 LQ/UAV 计数在每次运行日志里逐 epoch 打印，用来核对口径是否真的生效。）

### 2. 冻结策略：骨干锁死（b0~b4 / G1_frozen）

参数分组（`manu/training/optuna_mix_ratio_search.py:105-107`）：

| 组 | 模块归属 | 基准 lr |
|:--|:--|--:|
| `G1_frozen` | `b0 b1 b2 b3 b4`（Stem/P1、Stage1/P2、Stage2/P3） | 冻结，`requires_grad_(False)` + eval 保护 BN，共 **74,848** 参数 |
| `G2_p4p5` | `b5 ~ b10` | 3e-6 |
| `G3_neck` | 其余 neck 参数 | 1.5e-5 |
| `G4_head` | `head.*` | 3e-5 |

- `--lr-scale` 在 `manu/training/optuna_mix_ratio_search.py:406` 处**统一缩放 G2/G3/G4 三组基准 lr**
  （`lr = base_lr × lr_scale`）。历史文档把它记作"Head `lr_scale`"，
  实际作用范围是 neck+head 三组；命名与实现有出入，行为按代码走。
- `--freeze-bn 1` 会把**所有** `bn` 模块（含未冻结段）强制 eval
  （`manu/training/optuna_mix_ratio_search.py:467-473` 的 `set_frozen_eval`）；
  `--freeze-bn 0` 表示不额外冻结 BN 统计，此时冻结段 `b0~b4` 依然因 `requires_grad_(False)` + eval 保护
  而不更新统计量，未冻结段照常更新 BN。
- **铁律六**：组件扩容 / 微调**不解冻底模**，走受控渐进残差规范。
  骨干锁死是已锁定先验空间的一部分，不是可调项。
- **命名对齐**：老文档写"Stem~Stage2"，代码实现叫 `b0~b4`（G1_frozen）。
  两者**行为无差异**，只是命名不同；看到不一致时以代码为准，不要当成两个不同的冻结方案。

启动时会打印分组，可现场核对：

```text
[PARAM] G1_frozen 参数 74,848（严格冻结）
[PARAM] G2_p4p5  lr=6.811e-06 (基准 3.0e-06 × 2.270)  参数 ...
```

### 3. cosine 退火周期必须显式绑定 `--sched-epochs 5`

RC1/Ep3 走的是 **5 epoch 全长的 cosine 曲线**（总步 8895），到 Ep3 结束时 lr 仍在高位。
若按 3 epoch 自己的 cosine（总步 5337），Ep3 结束就退到**绝对 lr 下限**
（`ROUTE_B_DEFAULTS["lr_min"] = 1e-6`，对所有参数组生效），每个 trial 的末 epoch 权重都
系统性地"比 RC1 学得少一点"，与 `mix_spec` 的真实效应混在一起，无法归因。
旧口径下末 epoch lr 只有 RC1 的 **55%**。

因此：

- `--sched-epochs` 语义是"cosine 退火**周期**按几个 epoch 算"，**与训练总 epoch 数解耦**。
- 训练 5 epoch 时保持 `--sched-epochs 5`（默认值即 5）。
- 如果你把训练 epoch 数改成 3，**必须同时显式传 `--sched-epochs 5`**，
  否则 `sched_epochs = args.sched_epochs if > 0 else args.epochs` 会退化成 3-epoch 周期。
- 历史教训已用 `reproduction_check()` 回测证实：旧调度下的 `trial_0000` 判 **NOT_REPRO**，
  正面印证了 cosine 周期诊断。但要注意新调度下的对照 trial（`trial_0012` / `trial_0014`）
  同样判 NOT_REPRO ⇒ NOT_REPRO 判的是"出了 RC1 自身 epoch 间包络"，
  **不是"调度错了"的唯一证据**。拿到 NOT_REPRO 要回到 `[SCHED]` / `[MIX]` / `[PARAM]`
  三行日志逐项核对，不能只凭这一条就下结论。

启动日志确认周期（不看到这行就别往下走）：

```text
# --epochs 5 --sched-epochs 5（周期与总长一致 ⇒ 无后缀标记）
[SCHED] 训练 5 ep（8895 步）| warmup 1046 步（0.588 ep）| cosine 周期 5 ep（8895 步）

# --epochs 3 --sched-epochs 5（周期被显式解耦 ⇒ 出现 ← 与 RC1 同口径 标记）
[SCHED] 训练 3 ep（5337 步）| warmup 1046 步（0.588 ep）| cosine 周期 5 ep（8895 步）  ← 与 RC1 同口径
```

（`warmup 1046 步` = `round(0.588 × 1779 iters/epoch)`；该后缀由代码在
`sched_epochs != epochs` 时自动打印，出现即说明周期已与总长解耦。）

### 4. 正式训练命令（路线 B worker）

路线 B 的混训主体是 `manu/training/optuna_mix_ratio_search.py` 的 **worker 模式**
（`--worker`）。`manu/training/mixed_ft_train.py` 是更早的路线 B 训练器，
**没有** `--mix-spec` / `--sched-epochs` / `--lr-scale` / 冻结相关参数，
不要拿它当混训入口；`manu/training/finetune_median_4gpu.py` 是更早的单场景微调路径，同样不是混训入口。
调度器 `manu/training/launch_route_b_train.sh` 走的是旧的 `mixed_ft_train.py`，只用于复现 RC1。

单 trial 正式训练（服务器，GPU 0）：

```bash
# 服务器
cd /tmp/pycharm_project_10ae9e2e
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
export PYTHONPATH=.

python -u manu/training/optuna_mix_ratio_search.py \
  --worker \
  --trial-number 80 \
  --gpu-id 0 \
  --output-root runs/optuna_mix_ratio \
  --longquan-data /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/data.yaml \
  --uav-data      /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
  --weights       runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --gate-script   manu/training/auto_epoch_gate_check.py \
  --baseline-cache runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl \
  --epochs 5 --sched-epochs 5 \
  --batch 32 --imgsz 640 --workers 16 \
  --gate-epochs last --gate-epoch 3 \
  --seed 0 \
  --mix-spec w2.5 \
  --lr-scale 2.2702 \
  --warmup-epochs 0.588 \
  --weight-decay 2.6906e-04 \
  --focal-beta 2.75 \
  --offset-weight 0.40 \
  --mosaic 0.15 --scale 0.10 --translate 0.05 \
  --clip-max-norm 5.65 \
  --freeze-bn 0
```

注意事项：

- worker 的 `--mix-spec` / `--lr-scale` / `--warmup-epochs` / `--weight-decay` / `--focal-beta` /
  `--offset-weight` / `--mosaic` / `--scale` / `--translate` / `--clip-max-norm` / `--freeze-bn`
  **全部是 `required=True`**，少一个就报错退出。
- `--gate-epoch 3` 会把 `epoch3.pt` 作为门禁权重，门禁报告落在
  `runs/optuna_mix_ratio/trials/trial_XXXX/gate/epoch3/gate_report.json`。
- worker 内置**满额护栏**：龙泉山样本数必须等于 13,918、Anti-UAV 必须等于 43,008，
  否则 `[DATA-GATE]` 直接中止（这是刻意中止，不是 bug；换场景要同步改
  `manu/training/optuna_mix_ratio_search.py:102` 的 `LQ_EXPECT` / `UAV_EXPECT`）。
- `--gate-script` 调 `manu/training/auto_epoch_gate_check.py`，它对特征缓存显式传
  `--fusion-dilate 32`；缓存脚本 `manu/data/multiscale_heatmap_cache.py:69` 的默认值是 **9**，
  自己另写缓存脚本时漏传就会用 9，阈值与膨胀半径一起错。

多 trial 搜索走调度器（可跨多卡并行）：

```bash
# 服务器
bash manu/training/launch_mix_ratio_search.sh          # 起调度器
python manu/training/status_mix_ratio_search.py         # 查状态（--top / --strict-only / --all-calibers）
bash manu/training/stop_mix_ratio_search.sh             # 停
```

**起调度器之前要有用户明确授权**：搜索启动、扩大搜索空间、改训练协议、跑重评估，
都属于消耗算力的动作，没有用户点头就自己开跑，等于替用户做了不可逆的决定。
拿不准时先报出"要跑什么、要占几张卡、大概多久、产物落在哪"，等确认再执行。

调度器可以在**不重启**的前提下改后续 trial 的协议 ——
写 `<out_root>/runtime_config.json`：`{"train_epochs": 5, "gate_epoch": 3, "workers": 16}`。
调度器每派一个 trial 会把 `--sched-epochs` 之外的协议参数拼进 worker 命令；
`--sched-epochs` 不在拼接列表里，因此 worker 用的是自身默认值 5 ——
**这正是我们想要的**，也说明任何"改 epoch 数"的操作都要回头确认 cosine 周期。

### 5. 上线配置：`trial_0080`（正式训练/复现照抄）

| 参数 | 值 |
|:--|--:|
| `mix_spec` | `w2.5`（实测 **1:2.524**） |
| `lr_scale` | 2.2702 |
| `warmup_epochs` | 0.588 |
| `weight_decay` | 2.6906e-04 |
| `focal_beta` | 2.75 |
| `offset_weight` | 0.40 |
| `mosaic` | 0.15 |
| `scale` | 0.10 |
| `translate` | 0.05 |
| `clip_max_norm` | 5.65 |
| `freeze_bn` | 0 |

实测表现：`U = +0.052554`（相对 RC1/Ep3 绝对值），`+0.024975`（相对 RC1 的正增益）。

**已锁定的先验空间**（新增场景直接复用，不要重新开宽搜索）：

- 配比 **w2.0 ~ w2.5**
- `lr_scale` **2.0 ~ 2.8**
- 骨干锁死

依据：`corr(U, lr_scale) = +0.824`（全部新 trial），w2.5 档内部 **+0.916**；
把 `lr_scale` 锁死回 RC1 原值 1.0 后，各配比的 U 中位数几乎无差别
（natural +0.0229 / w4.0 +0.0223 / w2.5 +0.0222 / w2.0 +0.0185）
⇒ **主导因子是 `lr_scale`，配比是次要因子（只在高 lr 区间显现）**。
推论：想继续提升就先动 `lr_scale`，不要先动配比。

另有 8 个只过"放宽通道"但 U 也超 RC1 的 trial（其中 `trial_0061` +0.057861 最高），
它们**源域有退让** ⇒ 上线选 `trial_0080` 这一档，不选最高分那批。

### 6. 三轨门禁与判别

三轨定义（以 `manu/training/mix_ratio_judgement.py` 为准）：

| 轨 | 内容 | 冻结基线 F1 / FP |
|:--|:--|:--|
| **Track1** | 源域 Anti-UAV 大盘（抗混轨） | 0.905736 / 1014 |
| **Track2** | 龙泉山黄金池 7 条（守门员轨） | 0.988436 / 26 |
| **Track3** | 盲测 `001324`（难例轨） | 0.837331 / 590 |

（历史文档里 Track1/Track2 的叫法与代码不一致；行为按代码走。）

判定是**三级漏斗**，顺序不可交换，绝不能加权求和成一个总分：

```text
【一级】安全红线（相对冻结基线，一票否决）
       · Track2 黄金池：F1 ≥ 0.985 且 ΔFP_T2 ≤ +10
       · Track1 源域  ：ΔF1_T1 ≥ −0.005
【二级】帕累托占优（相对对手 B）：三轨 F1↑ 且 FP↓ 全向不劣，且至少一维严格更优
【三级】效用裁决（仅二级不成立时）
       U = ΔF1_T3 + α·ΔF1_T2 − β·max(0, −ΔF1_T1) − γ·max(0, ΔFP_T2)
       α=0.5（黄金池巩固） β=2.5（源域惩罚倍率） γ=0.0005（每新增 1 个黄金池 FP）
```

```bash
# 服务器：候选 vs 擂主，输出判定
python manu/training/mix_ratio_judgement.py \
  --candidate runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/gate_report.json \
  --baseline  rc1 \
  --out-json /tmp/judgement_trial_0080.json

# 复现性回测（EXACT_REPRO / BAND_REPRO / NOT_REPRO）
python manu/training/check_control_repro.py --out-root runs/optuna_mix_ratio --epoch 3
python manu/training/analyze_mix_spec_results.py --out-root runs/optuna_mix_ratio --epoch 3
```

`--strict-only` 只用用户原始严格红线（禁用实测分布校准出的"放宽通道"），用于复现旧结论。

### 7. 产物持久化

- 服务器 `/tmp` 是 ext4 `/dev/nvme0n1p2`（3.4T，**非 tmpfs**）：重启不清空，但**写满会丢**。
  ⇒ 关键权重必须另存 `/mnt/data`（`/dev/sda`，7.4T，独立物理盘）。
- 每次 trial 的 `epoch1/2/3.pt` 全部保留（每个约 21.3 MB），不要只留 best。
- 权重产物路径：`runs/optuna_mix_ratio/trials/trial_XXXX/weights/epoch3.pt`；
  RC1 参照物在 `runs/mixed_ft_route_b/RC1/epoch3.pt`
  （md5 `a845a95b77ac8762b7d033b64f61425e`），RC1 六点曲线在
  `runs/mixed_ft_route_b/RC1/RC1.json` 的 `full_cycle_curve`。

```bash
# 服务器：关键产物另存持久盘
mkdir -p /mnt/data/siping/backup/mix_ratio_search_$(date +%Y%m%d)
cp -a runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt \
      /mnt/data/siping/backup/mix_ratio_search_$(date +%Y%m%d)/
md5sum runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt \
      /mnt/data/siping/backup/mix_ratio_search_$(date +%Y%m%d)/epoch3.pt
```

### 8. GPU 铁律

**探针、门禁、评估不得与正在训练的 trial 同卡；单卡同时只允许"训练"或"评估"二者之一。**

- 起评估/门禁前先 `nvidia-smi` 确认目标卡显存与进程；训练 trial 正在占卡就换卡。
- 门禁若因 CUDA OOM 失败，判定器里 `tracks.track1_uav.finetune_peak.f1` 会是 `null`，
  这是**门禁基础设施失败**，不是模型指标不达标 —— 别据此判负。
- 历史 OOM 事故的归纳就在这条：单卡并发训练+评估 ⇒ OOM ⇒ 训练被打断 ⇒ 权重丢失。

## Output contract

一次完整混训交付应当包含：

1. **训练日志**：调度器派单时写 `runs/optuna_mix_ratio/logs/trial_XXXX.log`
   （`manu/training/optuna_mix_ratio_search.py:908`）；
   单独跑 worker 时 stdout 不会被落盘，要自己重定向：
   `python -u manu/training/optuna_mix_ratio_search.py --worker ... > logs/trial_0080.log 2>&1`。
   日志里必须能看到 `[MIX]`（口径与目标比例）、`[SCHED]`（cosine 周期）、`[PARAM]`（分组与 lr）、
   `[DATA-GATE] PASS`（样本满额）、`[EPOCH n]`（实测混样比例）。
2. **权重** `runs/optuna_mix_ratio/trials/trial_XXXX/weights/epoch{1,2,3}.pt`
   与已另存 `/mnt/data` 的副本 + 两侧 md5。
3. **门禁报告** `runs/optuna_mix_ratio/trials/trial_XXXX/gate/epoch3/gate_report.json`，
   含三轨 F1/FP/TP/FN。
4. **判定结论** `mix_ratio_judgement.py` 的输出（verdict + veto_failures + U 值），
   以及本次用的 `mix_spec` / `lr_scale` / 完整超参表。
5. **一句话结论**：本轮相对擂主 RC1/Ep3 是通过还是未通过；
   若通过，是否已具备"交接权杖"资格（需要多种子确认，中位数 ≥ +0.045 才正式交接）。

## Failure handling

- **执行机搞混**：本 skill 的命令全部针对 Linux 服务器（`ssh` / `conda` / `md5sum` / `nvidia-smi`）。
  Windows（win32）客户端上只做本地仓库的代码编辑与 md5 记录，
  **不要**在 win32 上执行训练、门禁或评估命令。
- **启动即报参数缺失**：worker 的 11 个超参全部 `required=True`，按步骤 4 的完整命令补齐。
- **`[DATA-GATE] 样本数异常` 立即中止**：数据集不完整或 `data.yaml` 指错。先修数据；
  换场景时同步改 `manu/training/optuna_mix_ratio_search.py:102` 的 `LQ_EXPECT` / `UAV_EXPECT`。
- **结果比预期系统性差一截**：先查 `[SCHED]` 那行的 cosine 周期是不是 5。
  3-epoch 周期会让末 epoch lr 只有 RC1 的 55%，训出系统性更差的模型，
  而且这种劣化会和 `mix_spec` 的真实效应混淆 —— 用 `reproduction_check()` 回测确认。
- **`reproduction_check()` 判 NOT_REPRO**：表示"出了 RC1 自身 epoch 间包络"。
  常见原因是协议与基线不一致（cosine 周期、epoch 数、seed、workers）。
  对照 `[SCHED]` / `[MIX]` / `[PARAM]` 三行日志逐项核对后再重跑；
  注意新调度下也有对照 trial 判 NOT_REPRO，别只凭这个标签就断定"调度错了"。
- **门禁报 `f1: null` 或 CUDA OOM**：基础设施失败，不是模型差。换空闲卡重跑门禁，
  并确认没有训练 trial 占用同一张卡。
- **配比调了没效果**：按 `corr(U, lr_scale)=+0.824` 的结论，先调 `lr_scale`（2.0~2.8 区间内），
  配比只在高 lr 区间才显现效果。
- **磁盘写满 / 权重丢失**：`/tmp` 是 ext4 不是 tmpfs，重启不清空但写满会丢。
  训练过程中就把 epoch 权重另存 `/mnt/data`，不要等跑完再拷。
- **必须回报用户而不是硬上的情况**：
  (a) 需要突破已锁定先验空间（配比 w2.0~w2.5 ＋ `lr_scale` 2.0~2.8 ＋ 骨干锁死）才能前进；
  (b) 需要解冻底模或改底模结构（触碰铁律六）；
  (c) 多种子确认未达中位数 ≥ +0.045 的交接门槛；
  (d) 目标卡无法独占，训练与门禁无法分卡；
  (e) 需要起新一轮搜索 / 扩搜索空间 / 改训练协议 / 跑重评估，但没有用户明确授权。
  这五种情况继续跑只会产出不可比的结果，应当带着已完成的证据（门禁 JSON、判定输出、
  参数字典）回报用户并说明取舍。

## Examples

### 例 1（canonical，已真实执行）：复现并上线 `trial_0080`

输入：擂主 `rc1`（`runs/mixed_ft_route_b/RC1/epoch3.pt`）、
锁定先验空间内的候选配置（`w2.5` / `lr_scale=2.2702` / 骨干锁死）、GPU 0 空闲。

执行（服务器）：

```bash
cd /tmp/pycharm_project_10ae9e2e
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
export PYTHONPATH=.

python -u manu/training/optuna_mix_ratio_search.py \
  --worker --trial-number 80 --gpu-id 0 \
  --output-root runs/optuna_mix_ratio \
  --longquan-data /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/data.yaml \
  --uav-data      /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
  --weights       runs/optuna_p0_nas/trial_0474/weights/best.pt \
  --gate-script   manu/training/auto_epoch_gate_check.py \
  --baseline-cache runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl \
  --epochs 5 --sched-epochs 5 --batch 32 --imgsz 640 --workers 16 \
  --gate-epochs last --gate-epoch 3 --seed 0 \
  --mix-spec w2.5 --lr-scale 2.2702 --warmup-epochs 0.588 \
  --weight-decay 2.6906e-04 --focal-beta 2.75 --offset-weight 0.40 \
  --mosaic 0.15 --scale 0.10 --translate 0.05 --clip-max-norm 5.65 --freeze-bn 0

python manu/training/mix_ratio_judgement.py \
  --candidate runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/gate_report.json \
  --baseline rc1 --out-json /tmp/judgement_trial_0080.json
```

输出（实测值）：
- 日志关键行：`[MIX] 口径1 加权采样：P(LQ)=0.2857 P(UAV)=0.7143`（w2.5 ⇒ 1/3.5 与 2.5/3.5），
  `[EPOCH 3] ... 实测混样 LQ ... / UAV ... = 1:2.524（目标 1:2.500）`。
- 门禁报告：`runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/gate_report.json`。
- 判定：`U = +0.052554`（相对 RC1/Ep3 绝对值）、`+0.024975`（正增益），
  一级红线全部通过 ⇒ 列为**暂定 RC2 候选**；因只跑过 1 次，
  正式交接权杖还需多种子确认（中位数 ≥ +0.045）。
- 权重：`runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt`，
  已另存 `/mnt/data/siping/backup/mix_ratio_search_20261022/`。

### 例 2：新场景并进混训时的最小改动

输入：沙漠场景已由 `domain-data-curator` 产出
`/mnt/data/siping/datasets/manu/desert_ft_gmc_median_official/data.yaml`。

执行：
1. 把 worker 的 `--longquan-data` 指向沙漠场景的 `data.yaml`；
2. 同步把 `manu/training/optuna_mix_ratio_search.py:102` 的 `LQ_EXPECT`
   改成沙漠 train 样本数（否则 `[DATA-GATE]` 会直接中止）；
3. 其余超参**先照抄 `trial_0080` 整表**跑 1 个 trial，确认 `[EPOCH 3]` 的实测混样比例
   命中目标（例如 `w2.5` 目标 1:2.500、实测 1:2.52x）后，再在
   `lr_scale 2.0~2.8` 区间内做小范围搜索。

输出：一份"沙漠场景 × `trial_0080` 超参"的首轮结果 + 门禁 JSON；
配比与 `lr_scale` 的调整建议只在新场景数据就绪、门禁基础设施正常后才提交。
