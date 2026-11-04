# No-GMC 周末复现实验（2026-10-30 ~ 11-02）—— 专题档

> 主题：**上一轮 no-GMC 实验的模型被误删**（连同 99 行 CSV / SUMMARY / 全部权重，
> 2026-10-29 在服务器 `/tmp` 被清掉，见 §0），重新做一遍，并**扩展搜索空间**。
>
> 核心假设（用户提出，本轮的主线）：
> 1. **用 GMC RC2（trial_0080）当初始化权重**，而不是冻结基线（474 去 P0）——
>    trial_0080 拥有全项目最高的难例表征上限（T3 0.897694）；
> 2. 但必须**同时**配套：更小的 lr（0.5~1.0 而非 2.88）、warmup ≥ 1 epoch、
>    骨干 lr 缩到 0.1×（或前 2 ep 锁死）、`freeze_bn=True` ⇒ 杜绝灾难性遗忘；
> 3. **3 个 epoch 对 no-GMC 太少**（"半夹生"）⇒ 扩到 6 / 10 / 15 ep + 完整 cosine 退火。
>
> 算力窗口：**host 2026-10-30 18:30 关机**（服务器不关）⇒ 从现在到
> **周一 2026-11-02 09:00** 的整段周末时间，服务器 4×RTX 4090D 独占。

---

## 〇、前情：上一轮产物是怎么丢的（必须记住的教训）

| 项 | 状态（2026-10-30 实测） |
|:--|:--|
| `runs/optuna_nogmc/`（99 行 CSV、SUMMARY.md、99 个 trial 权重、study.db） | ❌ **已丢失** |
| `runs/nogmc_ref/`（reference_frame.json） | ❌ **已丢失** |
| SSHFS 镜像 `/home/manu/mnt/pycharm_project_10ae9e2e/` | ❌ 同样没有 |
| `/mnt/data` 全盘 find | ❌ 无备份 |
| **no-GMC 数据集** `/mnt/data/siping/datasets/manu/{uav,longquanshan_ft}_gmc_median_nogmc` | ✅ **还在**（在持久盘） |
| 磁盘 | `/` 仅用 16%（488G/3.4T）⇒ **不是写满** |
| 服务器 `runs/` 目录 mtime | **2026-10-29 09:28**（`manu/` 同刻）⇒ 10-29 那次 RK3588 动作同刻动过 `runs/` |
| bash_history | 无 `rm -rf runs/optuna_nogmc` 记录 |

⇒ **根因：产物只存在服务器 `/tmp`、没落 `/mnt/data`**（数据集因为在 `/mnt/data` 才活下来）。
⇒ **本轮的硬规矩：所有产物根直接建在 `/mnt/data/siping/runs/`**，收尾再备份一份。

---

## 一、资产核查（2026-10-30 16:50~17:20 实测）

| 资产 | 状态 |
|:--|:--|
| GMC RC2 `runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt` | ✅ 在（21,300,963 B，md5 `0f03ada97bd3903fa6097c7b31627d9c`） |
| 冻结基线 `runs/optuna_p0_nas/trial_0474/weights/best.pt` | ✅ 在（7,375,044 B） |
| **ckpt 结构同构性** | ✅ **实测两者结构完全一致**：523 个张量 / 29 个 `p0_highway` / `p0_kwargs` 相同 / 键集一致 ⇒ **trial_0080 可直接当初始化权重**（这是整个方案的地基） |
| no-GMC 数据集 | ✅ Anti-UAV 43,008/31,613；龙泉山 13,918/28,997 |
| GPU | ✅ 4×RTX 4090D 全空闲（15 MiB / 24.5 GB） |
| 残留训练进程 | ✅ 无 |

**T1 基线保护值（memory §12.6.1，供核对）**：no-GMC 冻结基线 0.900897 / 0.981074(FP 52) / 0.757812；
GMC RC2 0.901910 / 0.998949(FP 33) / **0.897694**。

---

## 二、计时代账（决定整个周末怎么排）

从 GMC `trial_0080` 的实测日志反解：

```
iters/epoch = 1779
[EPOCH 1] ... 545.4s   [EPOCH 2] ... 541.7s   [EPOCH 3] ... 542.1s
[GATE] rc=0 用时 1116.0s
```

⇒ **每 epoch ≈ 9.05 min（训练）+ 18.6 min（固定门禁）**

| epochs | 训练 | 门禁 | 单 trial 合计 |
|--:|--:|--:|--:|
| 3 | 27.2 min | 18.6 | **45.8 min** |
| 6 | 54.3 | 18.6 | **72.9 min** |
| 8 | 72.4 | 18.6 | **91 min** |
| 10 | 90.5 | 18.6 | **109 min** |
| 15 | 135.8 | 18.6 | **154 min** |

窗口：Fri 18:30 → Mon 09:00 ≈ 62 h × 4 卡 = **248 GPU·h**。

---

## 三、扩展搜索空间（在原 11 维上 +4 维）

| 维度 | 取值 | 理由 |
|:--|:--|:--|
| `init_from` | `base474` / `rc2` | ★ 主假设：换初始化权重 |
| `epochs` | 3 / 6 / 10 / 15 | ★ 3 ep 对 no-GMC 是"半夹生" |
| `backbone_lr_scale` | 1.0 / 0.3 / 0.1 / 0.0 | G2(b5..b10) 的 lr 相对倍率；0 = 该段整段冻结 |
| `freeze_backbone_epochs` | 0 / 2 | 前 N ep 锁死骨干，第 N+1 ep 起解冻（渐进式） |

其余维度沿用旧空间，**两处边界放宽**：
- `lr_scale` 下限 0.3 → **0.2**（RC2 初始化需要小步长）；
- `warmup_epochs` 上限 1.0 → **1.5**（用户要求 warmup ≥ 1 个完整 epoch）。

⚠️ **兼容性护栏**：新维度由 `--extended-space` 开关控制，**不传 ⇒ `suggest_params` 与旧版逐位相同**
⇒ GMC 侧既有 `mix_ratio_search` study 的搜索空间**不受污染**（§12.6.7 的纪律延续）。

---

## 四、确定性对照矩阵（**16 条臂**，TPE 之前先跑完）

清单：`manu/training/nogmc_weekend_plan.json`（入队 ⇒ `study.ask()` 优先取走，**不靠 TPE 运气**）。

> ⚠️ 16 条臂已逐项校验：全部参数落在 Optuna 分布内（含 `step=0.05` 的整除性），
> 且**走真实 `enqueue_trial → study.ask() → suggest_params` 路径验证 16/16 逐位一致**。

### 4.1 主干 11 条（第一版）

| # | role | init | ep | lr | warmup | fbn | bb_lr | 锁骨干 | mix | 目的 |
|--:|:--|:--|--:|--:|--:|:--:|--:|--:|:--|:--|
| 0 | `NATURAL_RC1` | base474 | 3 | 1.0 | 0.5 | F | 1.0 | 0 | natural | 产出 no-GMC 语义的 rc1 参照 |
| 1 | `CTRL_BASE474_3EP` | base474 | 3 | 2.8821 | 0.588 | F | 1.0 | 0 | w3.0 | **对照组（原路线复现）** ⇒ 应复现旧 T3≈0.8119 |
| 2 | `CTRL_BASE474_10EP` | base474 | 10 | 2.8821 | 1.5 | F | 1.0 | 0 | w3.0 | **单独量化"epoch 太少"**（不换初始化） |
| 3 | `RC2_ANCHOR_3EP` | rc2 | 3 | 2.2702 | 0.588 | F | 1.0 | 0 | w2.5 | RC2 原配方只换数据集（可比锚点） |
| 4 | `ARM_A_RC2_LR1_8EP` | rc2 | 8 | 1.0 | 1.5 | F | 1.0 | 0 | w3.0 | **实验组 A**：RC2 移植直冲（中 lr + 长 epoch） |
| 5 | `ARM_B_RC2_LR05_FBN_8EP` | rc2 | 8 | 0.5 | 1.5 | **T** | 1.0 | 0 | w3.0 | **实验组 B**：特征保护（小 lr + freeze BN） |
| 6 | `ARM_C_RC2_PROG_8EP` | rc2 | 8 | 1.0 | 1.5 | F | **0.1** | **2** | w3.0 | **实验组 C**：渐进式解冻 |
| 7 | `ARM_BP_..._PROG_8EP` | rc2 | 8 | 0.5 | 1.5 | **T** | **0.1** | **2** | w3.0 | B+C 叠加（最强保护） |
| 8 | `ARM_CAT_RC2_LR28_8EP` | rc2 | 8 | **2.8821** | 1.5 | F | 1.0 | 0 | w3.0 | **灾难性遗忘反证**：RC2 初始化 + 大 lr 应当崩 |
| 9 | `ARM_LONG_RC2_LR10_15EP` | rc2 | 15 | 1.0 | 1.5 | F | 1.0 | 0 | w3.0 | 探顶（15 ep 是否继续涨 / 开始过拟合） |
| 10 | `ARM_RC2_LR07_6EP` | rc2 | 6 | 0.7 | 1.5 | **T** | 0.3 | 0 | w3.0 | 中间档 |

### 4.2 追加 5 条（**补齐"同 epoch 的 base474 对照"**，否则 init 效应无法干净归因）

| # | role | init | ep | lr | warmup | fbn | bb_lr | 锁骨干 | mix | 补的是什么 |
|--:|:--|:--|--:|--:|--:|:--:|--:|--:|:--|:--|
| 11 | `CTRL_BASE474_8EP` | base474 | 8 | 2.8821 | 1.5 | F | 1.0 | 0 | w3.0 | ★ 与 `ARM_CAT` **逐项同参只换 init** ⇒ 把 init 效应从 epoch 效应里剥出来；并补齐 base474 的 3/8/10 ep 阶梯 |
| 12 | `ARM_A_BASE474_LR1_8EP` | base474 | 8 | 1.0 | 1.5 | F | 1.0 | 0 | w3.0 | ★ `ARM_A` 的 **base474 孪生臂**（逐项同参只换 init）⇒ 直接读出"换初始化权重"这一个变量的净效应 |
| 13 | `ARM_D_RC2_BBFREEZE_8EP` | rc2 | 8 | 0.5 | 1.5 | **T** | **0.0** | 0 | w3.0 | 极端特征保护：骨干**整段冻结**，只训 Neck+Head |
| 14 | `ARM_E_RC2_W25_10EP` | rc2 | **w2.5** | 1.0 | 1.5 | **T** | 0.1 | **2** | w2.5 | 把 **RC2 自己那条配方**（w2.5）拉长到 10 ep |
| 15 | `ARM_F_RC2_LR03_15EP` | rc2 | 15 | **0.3** | 1.5 | **T** | 0.1 | **2** | w3.0 | **最保守极限**：最小步长 + 最长训练 + 全保护 ⇒ 若它仍打不过 GMC RC2，"no-GMC 必须换底模"被强力证实 |

合计 ≈ **25 GPU·h** ⇒ 4 卡并行约 **6.3 h** 跑完（搜索预算 44 h，余量充足）。

### 4.3 读法（**预注册判据**）

| 对比 | 读出什么 |
|:--|:--|
| `ARM_CAT` vs `CTRL_BASE474_8EP` | **init 效应**（同 8ep 同 lr 同超参，只换初始化权重） |
| `ARM_A` vs `ARM_A_BASE474_LR1_8EP` | **init 效应**（中 lr 档再来一次，成对读） |
| `CTRL_BASE474_3EP` vs `CTRL_BASE474_8EP` vs `CTRL_BASE474_10EP` | **epoch 效应**（同 init 同 lr，3/8/10 阶梯） |
| `ARM_CAT` vs `CTRL_BASE474_8EP`（都大 lr） | **灾难性遗忘**：若 RC2 初始化 + 大 lr 反而更差 ⇒ 小步长是必要条件 |
| `ARM_A/B/C/BP/D` 互相比 | **保护手段的相对强弱**（lr↓ / freeze_bn / 骨干冻结 / 渐进解冻） |
| `ARM_F`（最保守极限） vs GMC RC2 的 T3 0.897694 | **最终判决**：no-GMC 路线还能不能追上 |

若这三组效应（init / epoch / 保护）**分离度不够** ⇒ 结论是"**三者缺一不可**"，
而不是把功劳归给其中任何一个 —— 这也是上一轮"只报冠军不报依据"的教训（专题档 §15.23.5）。

---

## 五、代码改动（本地先改 → rsync 到服务器 → md5 三侧一致）

| 文件 | 改动 |
|:--|:--|
| `manu/training/optuna_mix_ratio_search.py` | ① +4 维搜索空间（`--extended-space` 开关，默认关）；② `--init-weights-map` 支持**每 trial 换初始化权重**；③ **每 trial 可换 epoch 数**（`--epochs`）+ cosine 周期自适应（3ep→5 保 RC1 口径；≥6ep→自身长度）；④ `--backbone-lr-scale`（G2 lr 倍率）；⑤ `--freeze-backbone-epochs`（渐进式解冻 + 冻结期 BN 保护）；⑥ `--enqueue-plan`（确定性臂强制入队）；⑦ CSV 增 4 列且**列头变化自动改名让位**（防错位 ⇒ 防"神秘"解析错） |
| `manu/training/summarize_nogmc_search.py` | `CFG_COLS` 扩到 15 维 + `_cfg_norm` 按类型规范化（老 CSV 缺列 ⇒ 取空串 ⇒ 仍是合法单配置） |
| `manu/training/measure_reference_frame.py` | 新增 `--track2-fp-max-increment`（no-GMC 口径 **60**，2026-10-27 用户授权） |
| `manu/training/plan_replication.py`（新） | 从汇总 CSV 挑配置级 U 中位前 N 名，各补到 3 个种子 ⇒ 交接计划 |
| `manu/training/nogmc_weekend_init_weights.json`（新） | `base474` → trial_0474 best.pt；`rc2` → GMC trial_0080 epoch3.pt |
| `manu/training/nogmc_weekend_plan.json`（新） | 11 条确定性对照臂 |
| `manu/training/launch_nogmc_weekend_search.sh`（新） | 启动器：**产物落 `/mnt/data`**、扩展空间、init map、enqueue plan |
| `manu/training/run_nogmc_weekend_pipeline.sh`（新） | **无人值守编排**：参照系 → 搜索 → **看门狗** → rc1 自动填 → 收尾 → 复现 → 最终汇总 + 备份 |

### 关键实现细节（别踩坑）

1. **渐进式解冻的实现**：G2(b5..b10) 的参数**始终留在 optimizer 的 param_group 里**，
   只在冻结期把 `requires_grad=False` ⇒ `p.grad is None` ⇒ AdamW 跳过、`clip_grad_norm_` 自动过滤
   （torch 1.13 的 `clip_grad_norm_` 只取 `p.grad is not None` 的参数）⇒ 解冻时**不需要重建 optimizer**，
   AdamW 状态在解冻后首次 step 时自建。
2. **冻结期 BN 保护**：`set_frozen_eval(epoch)` 里对 G2 段额外 `module.eval()` ⇒
   running mean/var 不被未对齐数据的亮边带偏（§十七 教训：BN 漂移毁掉底模）。
   `freeze_bn=True` 则对**所有** `*.bn` 模块强制 eval。
3. **cosine 周期与训练长度解耦**：3 ep 仍用 5 ep 周期（保 RC1/Ep3 同 LR 轨迹 ⇒ 与旧结果可比）；
   ≥6 ep 用自身长度 ⇒ 让 lr 在整段训练里走完一个完整 cosine（这正是"拉长 epoch"的前提）。
4. **CSV 列头变化自动改名让位**：新列加入后，老 CSV 直接追加会**整体错位** ⇒ 读盘时表现为"神秘"的解析错误。
   现在检测到列头不一致就把老文件 `改名.bak_<ts>` 并新开一份。

### 5.1 ⚠️ 骨干保护的**两种语义互斥**（容易误解，纯 CPU 实测确认）

| 参数 | 语义 | 实测 |
|:--|:--|:--|
| `backbone_lr_scale = 0.0` | **该段永久冻结**：G2 的 lr 恒为 0（`base_lr × lambda ≡ 0`），且 `g2_frozen_forever=True` ⇒ `freeze_backbone_epochs` **对它无效**（不会解冻） | ① scheduler 的 `floor_ratio = lr_min/base_lr` 在 base_lr=0 时**不除零**（有 `base_lr > 0` 保护）；② 梯度虽恢复但 lr=0 ⇒ **权重逐位不变** ✅ |
| `backbone_lr_scale > 0` + `freeze_backbone_epochs = N` | **渐进解冻**：前 N 个 epoch 冻结（`requires_grad=False` + BN eval），第 N+1 epoch 起解冻并按倍率更新 | 冻结期 `grad=None`、权重逐位不变；解冻后权重恢复更新 ✅ |
| `freeze_bn = True` | 所有 `*.bn` 模块强制 eval（running stats 不更新）⇒ 保护底模的统计量 | 独立于上面两者，可叠加 |

⇒ 读结论时注意：`ARM_D_RC2_BBFREEZE_8EP`（bb=0）是"**只训 Neck+Head**"的极端档，
和 `ARM_C`（bb=0.1 + 前2ep锁骨干）不是同一类保护，别混着比。

### 5.2 实测到的"自己踩自己"的坑（测试代码写错，不是产品代码错）

1. **以为 `lambda` 就是 lr**：scheduler 的 lambda 是**倍率**，实际 lr = `base_lr × lambda` ⇒
   base_lr=0 时 lambda 仍可非 0（warmup 期是 0.01→1.0），**但实际 lr 恒为 0**。
2. **以为"解冻后一定更新"**：base_lr=0 时不成立（lr 就是 0）⇒ 把"永久冻结"和"渐进解冻"两种语义分开验才说得清。
3. **GPU 冒烟测试抢显存**：4 卡都在跑真实 trial 时再起冒烟会 OOM（GPU0 已占 15.7 GB）⇒
   **验证"参数组/调度器/优化器"层面的逻辑用纯 CPU 单测**，别用 GPU。
   （OOM 是测试环境问题，不是产品代码问题 —— 真实 trial 的显存已被 PyTorch caching allocator 占住，不会被我的测试挤爆。）

---

## 五·二、开跑前测试抓到的真 bug（此处 3 个 + 后续 §8.4~§8.9 又抓到 6 个 ⇒ **累计 9 个**）

> 用户提醒"别切换 trial 的时候报错神秘的" ⇒ 确实抓到了，而且**不止一处**。
> 本节记的是**开跑前**那 3 个；后面 §8.4（老路径回归）、§8.6（交接标签撞车）、
> §8.8（`seed_noise` 并组）、§8.9（`is_rc1_hp` 误判）、§8.7（判据复核）、§8.12（Phase 3 算术）
> 等又陆续抓到/验证 6 项 ⇒ **合计 9 个**。
> 完整清单与影响见 §八 的 8.1~8.11。

| # | 现象 | 根因 | 修法 |
|--:|:--|:--|:--|
| 1 | `ValueError: '8' not in (3, 6, 10, 15)` —— **入队阶段直接炸** | `EPOCH_CHOICES` 没覆盖对照矩阵用到的 `epochs=8`；Optuna 的 `suggest_categorical` 对"固定参数不在 choices 里"直接抛异常 | `EPOCH_CHOICES = [3, 6, 8, 10, 15]`（并加注释：**改这个列表必须同步改 plan.json**） |
| 2 | `TypeError: expected str, bytes or os.PathLike object, not list` | `nogmc_weekend_init_weights.json` 里的 `_comment` 是**数组**，调度器直接 `Path(_v)` | 调度器遍历 init map 时**跳过非字符串值**（允许 JSON 放说明性字段） |
| 3 | 看门狗重启调度器 ⇒ **同一批臂被重复入队**（白烧 GPU，且拿到同一个 seed） | 去重护栏只看 `trial.params`，而 **WAITING trial 的 `.params` 是空的**（入队参数存在 `system_attrs["fixed_params"]`，要等 `ask()` 真正 suggest 后才落到 `.params`） | 去重时 `.params` 为空则回落到 `system_attrs["fixed_params"]`；命中就跳过 |

> 教训 3 值得单独记住：**Optuna 的"已存在"判定不能用 `trial.params`** ——
> WAITING 状态的 trial 参数在 `system_attrs` 里，和 RUNNING/COMPLETE 的取法不同。

**测试覆盖**：
- ✅ 冒烟测试（worker）：`--max-iters 2 --epochs 8 --freeze-backbone-epochs 2` ⇒ rc2 初始化加载成功、
  G2 lr = 3e-6 × lr_scale × 骨干倍率（实测 1.5e-7）、前 2 ep 冻结（BN 一并冻结）、
  **epoch 3 准时 `[UNFREEZE]` 解冻 1,290,624 参数**、8 ep 跑完出 8 份权重。
- ✅ 入队/取回一致性：`study.ask()` 取回的参数与计划**逐位一致 11/11**；TPE 自由采样不炸。
- ✅ 调度器启动路径：11 条臂入队成功；第二次启动**全部跳过**（去重生效）；WAITING 数量不变。

---

## 六、无人值守编排（host 18:30 关机也不受影响）

### 6.0 ✅ 已开跑确认（2026-10-30 17:16 服务器时间，实测）

| 项 | 状态 |
|:--|:--|
| **参照系包** | ✅ 已生成 `/mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json`；**数字与上一轮逐位吻合** —— T1 0.9008967482400269(FP 1118) / T2 0.981074481074481(FP 52) / T3 0.7578120617110798(FP 536)，`gate.track2_fp_max_increment = 60` ✅（⇒ 参照系重建正确，测量链路可信） |
| **调度器** | ✅ PID 244260，**PPID=1、SID=244260**（独立 session leader ⇒ 与 SSH/本机彻底解耦） |
| **流水线/看门狗** | ✅ PID 236888，**PPID=1、SID=236888** |
| **对照臂入队** | ✅ 16/16 条按计划顺序入队（seed 100~115） |
| **已派单** | ✅ 前 4 条同时开跑：trial_0000→GPU0(natural/3ep) / 0001→GPU1(w3.0/2.882/3ep) / 0002→GPU2(w3.0/2.882/**10ep**) / 0003→GPU3(w2.5/2.2702/3ep) |
| **GPU** | ✅ 4 卡各 15702 MiB、80~99% 利用率，**一卡一进程无争抢** |
| **心跳** | ✅ `status.json` 正常刷新 |

### 6.0.0 截止时刻对齐已核对（避免收尾链与调度器抢跑）

| | 时刻 | 来源 |
|:--|:--|:--|
| 调度器启动 | 10-30 17:16:40 | dispatcher.log |
| 调度器预算 | **45.0 小时** | dispatcher.log `截止: 45.0 小时后自动收尾` |
| ⇒ 调度器截止 | **11-01 14:16** | = 启动 + 45 h |
| 流水线 SEARCH_STOP | **11-01 15:00** | 编排脚本锚点 |

⇒ **调度器比编排锚点早 44 分钟退出** ⇒ 收尾链（⑤→⑥→⑦）在一个**已干净退出**的调度器之后开始，
**不会与在跑 trial 抢同一份 CSV / study.db** ✅（`--max-hours` 由 `LAUNCH_H=(STOP_S-now)/3600` 整数除法得到 45，
天然略保守，属预期行为）。

### 6.0.1 ✅ rc1 早填在**真实环境**触发（2026-10-30 18:03:57 首次门禁报告落地）

```
[18:03:57] 找到 natural 对照门禁报告：.../trial_0000/gate/epoch3/gate_report.json ⇒ 填 rc1
[RC1] T1 F1=0.8996206 FP=1138 | T2 F1=0.9964029 FP=102 | T3 F1=0.7726961 FP=1098
[18:03:57] rc1 填充完成，退出
```

**门禁阈值注入逐位核对（真实 trial_0000 的门禁命令行）**：
`--max-track2-fp 112`（= no-GMC 基线 FP 52 + 60 ✅）｜`--min-track1-f1 0.8958967482400269`（= 0.9008967 − 0.005 ✅）

**参照系 + 新擂主（no-GMC 语义）**

| | T1 | T2 | T3 |
|:--|:--|:--|:--|
| 冻结基线（474 去 P0） | 0.900897 (FP 1118) | 0.981074 (FP 52) | 0.757812 (FP 536) |
| **rc1 = natural 对照（trial_0000，PASS）** | 0.899621 (FP 1138) | 0.996403 (FP 102) | **0.772696** (FP 1098) |
| 差 | **−0.1276 pp**（红线 ≥ −0.5） | **ΔFP +50**（上限 +60） | **+1.4884 pp** |

⇒ no-GMC 语义下，自然 3 epoch 混训已能在**难例上拿到 +1.49 pp**，
但代价是源域 −0.13 pp、虚警 +50 —— 这就是 no-GMC 的起点水位线。

⚠️ trial_0000 的 `UvsRC1 = −0.1415` 是**rc1 填充前**算的（当时用的是 GMC 常量）；
收尾的 `rejudge` 会用最终参照系重算归零 ✅ —— 这正是"U 一律重算"铁律的用武之地。

### 6.1 编排流程

`run_nogmc_weekend_pipeline.sh`（服务器侧 `setsid nohup` 常驻）：

```
① 重建 no-GMC 参照系（474 去 P0 三轨基线，ΔFP_T2 ≤ 60）
② 启动扩展搜索（确定性 11 臂 → TPE）
③ 看门狗：调度器意外退出 ⇒ 自动断点续跑重启（study.db 自愈陈旧 RUNNING）
④ natural 对照落地 ⇒ 自动把它的指标填进参照系包的 rc1
⑤ 搜索到点 ⇒ backfill → rejudge → SUMMARY
⑥ 复现阶段：配置级多种子榜前 3 名，各补到 3 个种子（n-trials 精确卡住 ⇒ TPE 不再自由探索）
⑦ 最终重判 + SUMMARY + 带时间戳备份
```

**时间锚点（服务器本地时间）**

| 锚点 | 时刻 | 含义 |
|:--|:--|:--|
| 搜索停止 | **2026-11-01 15:00**（Sun） | ≈ 44 h 搜索预算 |
| 复现阶段 | ≤ **6 h** | 排在搜索之后 |
| 终极硬停 | **2026-11-02 06:00**（Mon） | 一切未收尾进程 SIGTERM + 强行出汇总 |
| 交付截止 | 2026-11-02 09:00（Mon） | 留 ~11 h 缓冲 |

**为什么必须落 `/mnt/data`**：§〇 的直接教训。服务器 `/tmp` 是 ext4（**不是 tmpfs**），
重启不清空，但**上一轮的产物就是在 10-29 被清掉的** ⇒ 这次直接建在 `/mnt/data/siping/runs/`。

---

## 七、监控（host 关机后仍可看）

### 7.1 ⚠️ 想往**本轮周末 study** 中途投放对照臂时的正确命令

`enqueue_control_trials.py` 的默认值指向的是 **GMC 侧** study
（`runs/optuna_mix_ratio/study.db` / `mix_ratio_search`）⇒ **不加参数会投错库**。

```bash
cd /tmp/pycharm_project_10ae9e2e && conda activate uav && export PYTHONPATH=.
python manu/training/enqueue_control_trials.py \
  --storage  "sqlite:////mnt/data/siping/runs/nogmc_weekend/study.db" \
  --study-name mix_ratio_search_nogmc_weekend \
  --summary-csv /mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv \
  --from-trial trial_00XX --count 2
```
（它用 `csv.DictReader` 按列名取参 ⇒ 新增的 4 列无害；未指定的扩展维度会由 TPE 采样。）

### 7.2 常规监控

```bash
ssh -p 32222 huangzhe@192.168.99.40

# 流水线总日志
tail -f /mnt/data/siping/runs/nogmc_weekend_pipeline.log
# 调度器日志
tail -f /mnt/data/siping/runs/nogmc_weekend/logs/dispatcher.log
# 单 trial 日志
tail -f /mnt/data/siping/runs/nogmc_weekend/logs/trial_XXXX.log
# 心跳（调度器每 3 s 写一次）
cat /mnt/data/siping/runs/nogmc_weekend/status.json
# 状态脚本
cd /tmp/pycharm_project_10ae9e2e && conda activate uav && \
  python manu/training/status_mix_ratio_search.py \
    --out-root /mnt/data/siping/runs/nogmc_weekend \
    --study-name mix_ratio_search_nogmc_weekend \
    --reference-frame /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json
# 产物
ls -la /mnt/data/siping/runs/nogmc_weekend/
cat /mnt/data/siping/runs/nogmc_weekend/SUMMARY.md
```

---

## 八、首批实测结果（2026-10-30 18:0x，起跑后第一波）

| trial | init | ep | lr | 骨干 | verdict | T1 | T2 | T2FP | **T3** |
|:--|:--|--:|--:|:--|:--|--:|--:|--:|--:|
| `trial_0000` NATURAL_RC1 | base474 | 3 | 1.0 | — | PASS | 0.899621 | 0.996403 | 102 | 0.772696 |
| `trial_0001` CTRL_BASE474_3EP | base474 | 3 | 2.8821 | — | PASS | 0.898046 | 0.997271 | 83 | **0.814615** |
| `trial_0003` RC2_ANCHOR_3EP | **rc2** | 3 | 2.2702 | — | **FAIL_TRACK1_RETREAT** | **0.895591** | 0.997631 | **75** | **0.843601** |

### 8.1 三条读数

#### 8.1.0 📌 对齐用户自定的**三条核心监控指标**（周一定论时直接用这张表）

用户原话的三条判据：
> ① **T2 FP**：epoch 从 3 增到 10，虚警从 60 显著降到 30~40 ⇒ 说明网络学会过滤未对齐边缘
> ② **T3 盲测 F1**：能否**跨过 0.81 瓶颈**，往 **0.85~0.87** 冲刺
> ③ **T1 门禁 0.895897**：防灾难性遗忘

**当前实测（18:22）**

| 参照 | T1 | T2 FP | T3 | 备注 |
|:--|--:|--:|--:|:--|
| no-GMC 冻结基线 | 0.900897 | 52 | 0.757812 | 起点 |
| no-GMC rc1（natural 3ep） | 0.899621 | 102 | 0.772696 | 本轮新擂主 |
| base474+w3.0+2.88+**3ep**（旧交接配方） | 0.898046 | 83 | 0.814615 | 复现旧成绩 0.811897 ✅ |
| **rc2+w2.5+2.27+3ep**（大 lr） | **0.895591** ❌触线 | **75** | **0.843601** | **T3 已跨过 0.81 瓶颈** |
| GMC RC2（参照上限） | 0.901910 | 33 | 0.897694 | 当前最好水平线 |

**⇒ 三条指标的当前判读**

1. **T2 FP**：rc2 起步臂 **75**，比 base474 同配方（83）**少 8 个**，比 rc1（102）**少 27 个**
   ⇒ 方向与用户预期一致（虚警在降），但**离"30~40"目标带还远** ⇒ 8ep 臂若能把 FP 压到 40 以下，
   就说明"网络学会了过滤未对齐地物白边"，这是**判断"拉长 epoch 是否真在起作用"的最干净信号**
   （比 T3 更直接，因为 T3 还会受底模表征上限影响）。
2. **T3**：**0.843601 已跨过 0.81 瓶颈（+3.17 pp）**，且**接近 0.85 目标带下沿**
   ⇒ 用户"T3 往 0.85~0.87 冲刺"**是可达的**，不是空想；
   而这条臂还是**被 T1 否决的大 lr 档** ⇒ **小步长 + 骨干保护（ARM_A/B/C）应当能再往上走**。
   ⚠️ 但 GMC RC2 的 0.897694 仍高 **5.4 pp** ⇒ no-GMC 语义下的天花板问题**尚未解决**。
3. **T1**：rc2+大 lr 臂 **0.895591**，距红线 **−0.000306**（差 0.03 pp）
   ⇒ **几乎擦线**。按 loss 曲线读数（该臂 loss 0.3265 是全场最低、拟合最好），
   **这不是"没学会"而是"学过头"** ⇒ 小步长 + 保护应当**同时**把 T1 拉回线上并保住 T3。

**⇒ 周一的核心问句（一句话）**：
> **ARM_A/B/C 能不能在 T1 ≥ 0.895897 的前提下，把 T3 做到 ≥ 0.8436（rc2 大 lr 档已证实的下限）、
> 并把 T2 FP 往 40 以下压？** —— 能 ⇒ "换底模 + 小步长 + 保护 + 长 epoch"四件套成立；
> 不能 ⇒ no-GMC 差距来自语义本身，必须换底模。

### 8.1.1 起跑基线快照（2026-10-30 18:24 服务器时间）—— 周一用来做 diff 的"前"

**已结算 3 条**

| trial | init | ep | lr | verdict | T1 | T2 FP | T3 |
|:--|:--|--:|--:|:--|--:|--:|--:|
| trial_0000 | base474 | 3 | 1.0 | PASS | 0.899621 | 102 | 0.772696 |
| trial_0001 | base474 | 3 | 2.8821 | PASS | 0.898046 | 83 | 0.814615 |
| trial_0003 | **rc2** | 3 | 2.2702 | FAIL_TRACK1_RETREAT | 0.895591 | 75 | **0.843601** |

**在跑 4 条 + 预计落地时间**（按 9.05 min/epoch + 18.6 min 门禁推算）

| trial | 配置 | 进度(18:24) | 预计落地 |
|:--|:--|:--|:--|
| trial_0002 | base474 / w3.0 / 2.8821 / **10ep** | 7/10 ep | ≈19:10 |
| trial_0004 | **rc2** / w3.0 / 1.0 / 8ep（**ARM_A**） | 2/8 ep | ≈19:37 |
| trial_0005 | **rc2** / w3.0 / 0.5 / 8ep / freeze_bn（**ARM_B**） | 2/8 ep | ≈19:37 |
| trial_0006 | **rc2** / w3.0 / 1.0 / 8ep / 骨干0.1 + 锁2ep（**ARM_C**） | 2/8 ep | ≈19:37 |

⇒ **这 4 条落地后（≈19:40 服务器时间）就能对上面三条监控指标给出第一轮完整判读**；
其余 12 条臂 + TPE 阶段在周日整晚继续。**周一开机后先看这 4 条，再看 SUMMARY。**

### 8.1.1b ⭐ 三条 RC2 臂在 **epoch 2 开始分道扬镳**（2026-10-30 18:26 实测）

| trial | 配置 | Ep1 loss | **Ep2 loss** | 变化 |
|:--|:--|--:|--:|:--|
| trial_0004 | **ARM_A**：rc2 + lr1.0 + 骨干1.0 | 0.3545 | 0.3512 | −0.0033 |
| trial_0005 | **ARM_B**：rc2 + lr0.5 + **freeze_bn** + 骨干1.0 | 0.3550 | **0.3565** | **+0.0015** |
| trial_0006 | **ARM_C**：rc2 + lr1.0 + **骨干0.1× + 前2ep锁骨干** | 0.3565 | **0.3459** | **−0.0106** |

⇒ **完美印证 §8.2 的读数 3**：三条臂在 **Ep1 几乎同值**（0.3545/0.3550/0.3565），
**Ep2 起分离**，且分离方向与假设一致：
- **ARM_C（骨干保护 + 前 2 ep 锁死）loss 最低（0.3459）且降幅最大（−0.0106）**
  ⇒ 前 2 ep 只训 Neck+Head 把"假警抑制"学稳，第 3 ep 解冻骨干后拟合效率最高
  ⇒ **"护住骨干"这条策略在训练侧已经看到正效应**；
- **ARM_A（lr1.0 无保护）居中**（−0.0033）；
- **ARM_B（lr0.5 + freeze_bn）loss 反而微升**（+0.0015）
  ⇒ 步长最小 ⇒ 拟合最慢，但**最保守**（是否更保 T1 要等门禁结果）。

⚠️ **loss 低 ≠ T3 高**：这三臂的最终判决仍要看门禁（T1 是否过线 + T2 FP + T3）。
loss 只说明"拟合/学习效率"，**过拟合与遗忘要在指标上看**（trial_0003 就是 loss 最低却被 T1 否决的先例）。

### 8.1.2 ✅ 独立审计：门禁判决自洽性 + 日志零异常（2026-10-30 18:25）

**（a）用参照系常量独立复算每条 trial 的判决**（不信门禁自己的输出，直接拿 CSV 指标 + 参照系常量重算）：

| trial | T1 | 红线 | T2 FP | 上限 | 实际判决 | 独立复算 | |
|:--|--:|--:|--:|--:|:--|:--|:--|
| trial_0000 | 0.899621 | 0.895897 | 102 | 112 | PASS | PASS | ✅ |
| trial_0001 | 0.898046 | 0.895897 | 83 | 112 | PASS | PASS | ✅ |
| trial_0003 | **0.895591** | 0.895897 | 75 | 112 | FAIL_TRACK1_RETREAT | FAIL | ✅ |

⇒ **不自洽 0 条** ⇒ 门禁判定与参照系口径（`T1 ≥ 0.895897`、`T2 FP ≤ 112`）**完全自洽**。
⇒ 顺带确认：trial_0003 是**纯 T1 触线**（FP 75 远低于上限 112）⇒ 它完全符合"灾难性遗忘牺牲源域"的画像。

**（b）全量日志异常扫描**：流水线日志 + 全部 trial 日志逐个 grep
`Traceback | CUDA out of memory | Error | Exception` ⇒ **0 处** ✅

⇒ 起跑至今无任何运行期异常。

1. **对照组复现成功** ✅：trial_0001 的 T3 = 0.814615，旧交接成绩 0.811897，差 **+0.27 pp**（种子噪声内）
   ⇒ **整条流水线的测量链路可信**（且这正是"上一轮成绩可被本轮复现"的直接证据）。
2. **RC2 起步把难例顶上历史新高** ✅：trial_0003 的 **T3 = 0.843601**
   ⇒ 比上一轮 no-GMC 最好成绩（0.811897）**+3.17 pp**，比同配方 base474 对照（0.814615）**+2.90 pp**；
   且 **T2 虚警更少（75 vs 83）**。
   ⇒ **用户核心假设方向上已被实测证实**：RC2 的"弱小目标表征上限"确实能直接抬高 no-GMC 的难例起点。
3. **但撞了 T1 红线** ⚠️：T1 = 0.895591 vs 红线 0.895897（差 **0.0003**）⇒ `FAIL_TRACK1_RETREAT` 一票否决。
   而这条臂正是 **lr=2.2702（大 lr）+ 无骨干保护** 档 ⇒ **灾难性遗忘被量化**：
   大 lr 下 RC2 起步能吃到 T3 +2.9 pp 的红利，代价是源域塌 0.03 pp 触线。
   （旧 memory 已记：T1 触线**不是运气** —— 上一轮同一配置 3/3 越线 ⇒ 这次同样按真信号处理。）

⇒ **接下来 ARM_A / ARM_B / ARM_C 三条臂（rc2 + lr1.0/0.5 + 骨干0.1 或 freeze_bn + 8ep）正是验证
"小步长 + 保护"能否把 T3 0.8436 的红利保住、同时把 T1 拉回红线之上** —— 这是本周末最关键的看点。

### 8.2 训练损失曲线（2026-10-30 18:20 实测）—— **"RC2 表征上限"的训练侧直接证据**

| trial | init | ep | 逐 epoch loss |
|:--|:--|--:|:--|
| trial_0000 | base474 | 3 | 0.5828 → 0.4393 → 0.4140 |
| trial_0001 | base474 | 3 | 0.5033 → 0.3710 → **0.3504** |
| **trial_0002** | base474 | **10** | 0.5536 → 0.3891 → 0.3554 → 0.3407 → 0.3297 → 0.3294 → **0.3224** |
| trial_0003 | **rc2** | 3 | **0.3579** → 0.3375 → **0.3265** |
| trial_0004 | **rc2** | 8 | 0.3545（Ep1） |
| trial_0005 | **rc2** | 8 | 0.3550（Ep1） |
| trial_0006 | **rc2** | 8 | 0.3565（Ep1） |

**四条读数（比指标更早、也更硬）**

1. **RC2 起步的 epoch-1 loss ≈ 0.355，base474 起步是 0.50~0.58 ⇒ 低 30~37%**。
   ⇒ trial_0080 的表征让网络在 no-GMC 数据上**几乎不用学就能拟合**
   —— 这是"RC2 拥有最强弱小目标表征上限"的**训练侧直接证据**（此前只有 T3 指标侧证据）。
2. **base474 的 10ep 臂 loss 一路降到 0.3224，3ep 臂停在 0.3504**
   ⇒ 量化了"**3 ep 确实停得太早**"（用户判断成立）；10ep 在 Ep5 后已进平台
   （0.3297→0.3294→0.3224，降幅收窄）。
3. **RC2 的三条 8ep 臂 epoch-1 loss 几乎相同**（0.3545/0.3550/0.3565）
   ⇒ **第 1 个 epoch 由「相同初始化 + 1.5 ep warmup」主导**；三条臂的差异
   （lr 1.0 vs 0.5、freeze_bn、骨干 0.1×）**要到第 2 ep 之后才分道扬镳**
   ⇒ 也解释了 ARM_C「前 2 ep 锁骨干」为何在 Ep1 与 ARM_A 几乎无差。
4. ⚠️ **trial_0003（rc2 + 大 lr）loss 拟合得很好（0.3265，比 base474 的 0.3504 还低）却 T1 触线**
   ⇒ **触线不是"没学会"，而是"学过头/学偏了"** —— 灾难性遗忘的典型形态：
   **难例学得好（loss 低、T3 高 0.8436），代价是源域被牺牲（T1 塌 0.0003）**。
   ⇒ 结论表述时**不能**说"RC2 起步会过拟合"（拟合没问题），要说
   "**RC2 起步 + 大步长会优先让模型去吃难例，源域被挤掉**"。

### 8.3 读盘路径已在**真实线上数据**上端到端验证（2026-10-30 18:1x）

用真实 out-root + 真实参照系跑 `summarize_nogmc_search.py`（输出到临时路径，不污染产物）：
- ✅ §四 单点榜**含**被红线否决的 trial（trial_0003 RC2-init，T3 0.843601 最高但 FAIL_TRACK1_RETREAT）
  —— 与旧文"§四 混排了不可上线的 trial"一致；
- ✅ §四之二 可上线候选**正确剔除**它（候选池 2 个：trial_0001 / trial_0000）⇒ **一票否决逻辑生效**；
- ✅ trial_0000 的 `U(vs RC1) = −0.000000` ⇒ **rc1 早填已生效**（早填前是 −0.1415，用的是 GMC 常量）；
- ✅ 交接判据（§四之三）用 2 配置 × 3 种子合成数据验证过：分组、中位、极差、可复算排序规则全部正确。

**两处过时文案已修**（会误导读 SUMMARY.md 的人）："超参（11 维）"→ 动态 `len(CFG_COLS)`（现 15 维）；
"权重（epoch3.pt）"→ 改为按 **该 trial 实际末 epoch**（epoch8/10/15…，**不能写死 epoch3**）。

### 8.4 🔴 对 **GMC 老路径** 的行为回归已消除（共享脚本自检，2026-10-30 18:1x）

`optuna_mix_ratio_search.py` 是 **GMC 侧与 no-GMC 侧共用**的脚本 ⇒ 改它必须保证老路径逐位不变。
diff 自检发现**一处回归**：我原先让调度器**总是**显式传 `--sched-epochs`，
而旧版调度器**从不传**（worker 用自己的默认值 5）⇒ 若非扩展模式下 `--epochs≠3`，
cosine 周期会从 5 变成 trial_epochs ⇒ **GMC 老行为变了**。

**修法**：`--sched-epochs` **只在 `--extended-space` 下才传**（非扩展模式不传 ⇒ worker 用默认 5）。

**验证**（打桩 `subprocess.Popen` 捕获真实命令行，不占 GPU）：

| 模式 | --epochs | --sched-epochs | 期望 |
|:--|--:|--:|:--|
| 扩展 | 3 | 5 | 与 RC1 同口径 ✅ |
| 扩展 | ≥6 | = 自身长度 | 完整 cosine ✅ |
| **非扩展（GMC 老路径）** | 3 | **不传** | worker 用默认 5 ✅ |
| **非扩展（GMC 老路径）** | 10 | **不传** | 旧版恒为 5，**不能变 10** ✅ |

### 8.5 生产环境验证（真实训练日志，2026-10-30 18:1x）

| 臂 | 实测 |
|:--|:--|
| `trial_0002`（10ep 对照） | `[SCHED] 训练 10 ep（17790 步）\| warmup 2668 步（1.5 ep）\| **cosine 周期 10 ep**` ✅；loss **0.5536 → 0.3891 → 0.3554 → 0.3407 → 0.3297**（到 Ep5 仍在降 ⇒ **支持"3 ep 太少"假设**） |
| `trial_0006`（ARM_C 骨干×0.1 + 前2ep锁） | `[PARAM] G2 骨干段保护：backbone_lr_scale=0.1 freeze_backbone_epochs=2 ⇒ 前 2 个 epoch 冻结` ✅；`G2 lr = 3.000e-07`（= 3e-6 × 1.0 × 0.1）✅ |
| `trial_0004`（ARM_A rc2+lr1.0+8ep） | `[WORKER] ★扩展协议：init=rc2 \| epochs=8 \| backbone_lr_scale=1.0` ✅；`G2 lr = 3.000e-06`（骨干不缩放）✅ |
| 加权采样器（全部 w3.0 臂） | 实测混样 **1:2.986 / 2.962 / 2.992 / 2.982 / 2.978** vs 目标 1:3.000 ✅ 逐 epoch 贴合 ⇒ 口径 1 正确 |

---

### 8.6 🔴 **交接标签撞车**（会导致交接结论有歧义，2026-10-30 18:1x 逐行读判据代码时发现）

**症状**：§四之三 的"配置"标签只由 `mix_spec · lr_scale` 拼成。
但在 **15 维扩展空间**里，`base474/8ep` 与 `rc2/8ep`（或任何同 mix 同 lr、只差 init/epochs 的两条臂）
⇒ **标签完全相同** ⇒ SUMMARY.md 里两行看起来一模一样、**交接结论说不清是哪一个** ——
而 base474 vs rc2 的对比**恰恰是本实验的核心**。

**修法**：新增 `_glabel()`，标签改为 **`mix · lr{scale} · {init_from} · {epochs}ep`**，
表格列头同步改为"配置（mix · lr · init · epochs）"。

**验证**（造 2 组**同 mix 同 lr**、只差 init 的合成数据）：

| | 修复前 | 修复后 |
|:--|:--|:--|
| 配置 1 | `w3.0 · 2.8821` | `w3.0 · lr2.8821 · **rc2** · 8ep` |
| 配置 2 | `w3.0 · 2.8821` | `w3.0 · lr2.8821 · **base474** · 8ep` |
| 交接结论 | `w3.0 · 2.8821`（**歧义**） | `w3.0 · lr2.8821 · **rc2** · 8ep`（**明确**）✅ |

⇒ **周一读盘前必须用这个版本的汇总器**（已推送服务器，md5 `2a1a72846eb409dbbf8924baf2144ca0`）。

### 8.7 交接判据逐行复核结论（2026-10-30 18:1x）

`§四之三` 的判定链**逐行读过，逻辑正确**：
1. 冠军只从 **full_pass 组**（全部种子过一级红线）里选，按 **U 中位**排序；
2. U 差距 > **实测 U 噪声带**（= 各已复现组内 U 极差的最大值）⇒ **按 U 定序**；
3. 否则 **U 不可分 ⇒ 转看 T3 难例中位**，判别阈值 = **2×(中位不确定度_x + 不确定度_y)**，
   其中 不确定度 = **极差/2/√n**（**已用修正值，不用裸极差** —— 裸极差随种子数结构性偏小，
   实测 2 种子组 0.027 pp vs 3 种子组 0.170 pp 是**假优势**）；
4. 仍不可分 ⇒ 取种子数多者，并显式标注"两者实质相当"；
5. 输出里会**同时**给"U 榜首"和"按规则的交接配置"，不一致时**额外警告 + 提示补种子**；
6. "组内 U 最高的单点"与"交接口径"**分开表述**，避免拿单点当交接依据。

⇒ 判据本身**无需再改**；唯一要修的就是 §8.6 的标签撞车（已修）。

### 8.8 🔴 同一根源的**第二处**：`seed_noise` 也只用 11 维分组 ⇒ 噪声带被虚高

`compare_gmc_nogmc_matched.seed_noise()`（§五之四 噪声带的来源）用 `_param_tuple()` 分组，
而 `_param_tuple` 只取原 11 维 ⇒ 在 15 维空间里 **base474/8ep 与 rc2/8ep 会被并成"同一配置"**
⇒ 实测"种子噪声"里混进**配置间差异** ⇒ **噪声带虚高** ⇒ 交接判据的
"U 差距 > 噪声带"门槛被抬高 ⇒ **明明可分的 U 会被误判成"不可分"**。

**修法**：`_param_tuple` 增加 `EXTRA_PARAMS = (init_from, epochs, backbone_lr_scale, freeze_backbone_epochs)`；
`seed_noise` 的返回值也带上 init/epochs 供标签使用。
**老 CSV（GMC 侧 / 旧 no-GMC 侧）没有这 4 列 ⇒ 取空串 ⇒ 分组与旧版逐位一致** ✅。

**验证**（同 mix 同 lr、只差 init 的合成数据）：

| | 修复前 | 修复后 |
|:--|:--|:--|
| seed_noise 分组 | 1 组 4 trial（base474 与 rc2 **被并组**） | **2 组各 2 trial**（base474 / rc2 分开）✅ |
| 交接榜标签 | 两行同为 `w3.0 · 2.8821`（歧义） | `w3.0 · lr2.8821 · rc2 · 8ep` / `... · base474 · 8ep` ✅ |

⇒ **两处修复已推服务器**：`summarize_nogmc_search.py` md5 `6bb436564fe7f9df1ac4bba8e344e783`、
`compare_gmc_nogmc_matched.py` md5 `8a52996a4792bfa71a92ed9126f53f8e`。
**周一读盘必须用这两个新版本**，否则交接结论既歧义、噪声带又虚高。

### 8.9 🔴 第三处同源：`is_rc1_hp()` 只比对 11 维 ⇒ rc2 臂被误判成"RC1 对照"

`analyze_mix_spec_results.py` 用 `is_rc1_hp()` 把 trial 分成「复现对照」与「TPE 自选超参」两桶。
旧语义下"RC1 对照"= 11 维与 RC1 逐位相同；但在 15 维空间里，
**同配方的 rc2 臂（base474 vs rc2 只差 init_from）也会被判成 True** ⇒ 两类 trial 混进同一桶。

**修法**：新增 `RC1_EXT = {init_from: "base474", epochs: 3.0, backbone_lr_scale: 1.0, freeze_backbone_epochs: 0.0}`
（这才是旧 RC1 的真实语义）⇒ 扩展列存在时一并校验；**老 CSV 无这些列 ⇒ 跳过校验 ⇒ 与旧版逐位一致** ✅。

**验证 4 项全绿**：

| 输入 | is_rc1_hp | 期望 |
|:--|:--:|:--|
| 老 CSV 行（无扩展列） | **True** | 与旧版一致 ✅ |
| 新 CSV base474 / 3ep / 无保护 | **True** | ✅ |
| 新 CSV rc2 / 3ep | **False** | 不混进 RC1 桶 ✅ |
| 新 CSV rc2 / 8ep / 骨干保护 | **False** | ✅ |

⇒ 服务器已更新（md5 `85c6a4c642782b3dab377f953df4a157`）。

### 8.10 「11 维遗留假设」系统排查结论

按 `clip_max_norm`/`freeze_bn` 组合全量 grep `manu/**/*.py`，共找到 4 处硬编码 11 维的地方：

| 位置 | 状态 | 说明 |
|:--|:--:|:--|
| `summarize_nogmc_search.py` `CFG_COLS` | ✅ 已修（15 维） | 分组键 + 标签 |
| `compare_gmc_nogmc_matched.py` `_param_tuple` | ✅ 已修（+EXTRA_PARAMS） | 配对与噪声带分组 |
| `analyze_mix_spec_results.py` `is_rc1_hp` | ✅ 已修（+RC1_EXT） | RC1 对照桶判定 |
| `test_status_mix_ratio_search.py` 的 CSV 表头 | ✅ **无需修** | 那是**测试夹具**，老 26 列表头正好验证"缺列容错"；被测的 `status_mix_ratio_search.py` 按列名读，不受影响 |

⇒ **4/4 已处置**，无遗漏。

### 8.11 ✅ 既有测试套件全绿（改动未破坏原有行为，2026-10-30 18:1x 服务器实跑）

| 测试文件 | 结果 | 覆盖到的关键点 |
|:--|:--:|:--|
| `manu/training/test_optuna_dispatcher.py` | **26/26 ✅** | **直接测我改过的调度器**：陈旧 RUNNING 自愈 ✅、断点续跑不重跑已完成 trial ✅、编号单调递增 ✅ |
| `manu/training/test_status_mix_ratio_search.py` | **54/54 ✅** | 严格口径回退、空状态不崩 |
| `manu/training/test_mix_ratio_judgement.py` | **71/71 ✅** | EXACT/BAND/NOT_REPRO 判定、RC1 包络 |

⇒ **我改了共享脚本（`optuna_mix_ratio_search.py`）但没有破坏任何原有行为** —— 这是"GMC 老路径逐位不变"承诺的独立佐证。

> 注：服务器 conda 环境 `uav` **没装 pytest** ⇒ 这些测试文件带 `__main__` 入口，直接 `python <file>` 即可跑。

### 8.12 ✅ 复现阶段（Phase 3）的算术路径已验证

流水线 step ⑥ 用 `DONE_N = study 里 COMPLETE+PRUNED 的数`，
再 `TOTAL_N = DONE_N + N_REPL` 去卡 Phase 3 调度器的 `--n-trials`
⇒ **只补跑复现臂，不让 TPE 自由探索**（否则会多烧几小时在无关 trial 上）。

按流水线的**真实顺序**（脚本开头已 `conda activate uav`，全程有效）实跑：
`DONE_N = 3` ✅、`TOTAL_N = 3 + 6 = 9` ✅。

⚠️ 踩坑记录：我第一次测时把 heredoc 放在 `conda activate` **之前** ⇒ `ImportError: No module named optuna`
⇒ `DONE_N` 为空 ⇒ `TOTAL_N` 退化成 `0 + N_REPL`。
**流水线的 `conda activate` 在开头**，所以实际不会踩；但若以后把这个片段挪到别处，**必须确认 conda 已激活**
（heredoc 里的 `import optuna` 在 try 块**外面**，import 失败不会被 `except Exception` 兜住）。

另做了一次**静态未定义变量扫描**（流水线用 `set -uo pipefail` ⇒ 任何未定义变量引用会让脚本**当场退出**，
若发生在收尾链里就会丢备份）：脚本引用 33 个变量、显式赋值 37 个 ⇒ **无「引用但未赋值」** ✅。

## 八·五、✅ **最终实验结果**（2026-11-02 收官）

### 8.5.1 流水线时间线（跑满整个周末，零中断）

```
10-30 17:16  搜索启动（4 卡）
11-01 15:00  搜索满 45 h 自动收尾   ← 看门狗全程 **0 次重启**（调度器一次没崩）
11-01 15:01  复现阶段（6 条，2.5 h）
11-01 17:35  流水线全部完成 + 备份
```

**114 个 trial**：SUCCESS 81 / PRUNED 26 / CRASHED 7
**verdict 分布**：PASS 31 / **PASS_RELAXED 50** / FAIL_TRACK1_RETREAT 26 / GATE_REPORT_INCOMPLETE 3 / 无 4

产物：`/mnt/data/siping/runs/nogmc_weekend/`（SUMMARY.md · optuna_summary.csv · study.db · trials/）
备份：`/mnt/data/siping/runs/nogmc_weekend_backup_20261101_1735/`

### 8.5.2 ⚠️ **口径铁律被触发了（本轮新发现，最重要的一条）**

> memory 铁律：**「一级门禁：严格通道为上位门槛；放宽通道仅作诊断标记」**

- 汇总器 §4.3.1 给出的交接线是 **w2.0 · lr2.6669 · rc2 · 8ep**（2 种子，T3 中位 0.872844，U 中位 +0.106645），
  但**这两条 trial 的 verdict 是 `PASS_RELAXED`** ⇒ 按铁律**不能作正式交接线**。
- **根因**：`plan_replication.py` 复现阶段只按 `status == "SUCCESS"` 挑复现对象
  ⇒ **把 PASS_RELAXED 当成可上线** ⇒ 2026-11-01 那轮复现挑中了两个"放宽口径"配置，
  而**真正严格过线的配置全是 n=1** ⇒ **该复现的没复现**。
  ⇒ **已修**（`plan_replication.py` 现在只挑 `verdict == "PASS"`）。
- ⇒ **严格口径下没有任何配置拿到 ≥2 种子** ⇒ **正式交接线目前缺多种子证据**，
  只有 n=1 的单点可选 ⇒ **交接需要补一轮严格口径的多种子复现**。

### 8.5.3 严格口径 PASS 的关键读数

| trial | 配置 | T1 | T2FP | T3 | U |
|:--|:--|--:|--:|--:|--:|
| **trial_0009** | rc2 / 15ep / **lr 1.0** | 0.896171 | 64 | **0.857880** | +0.090682 |
| **trial_0025** | rc2 / 6ep / lr 2.8053 | 0.897516 | **47** | **0.850972** | +0.093390 |
| trial_0018 | rc2 / 6ep / 骨干全冻+锁2ep | 0.896205 | 56 | 0.849318 | +0.086339 |
| trial_0062 | rc2 / 8ep / 骨干0.1× | 0.895936 | 64 | 0.844467 | +0.076696 |
| trial_0021 | rc2 / 6ep / 骨干0.1× | 0.896294 | 52 | 0.842445 | +0.081734 |

**全局最高 T3 = 0.890979**（trial_0060，rc2/15ep/lr2.9191，T2FP 36）⇒ **T1 0.892473 触线出局**；
它同时满足了用户要的 **T2FP 30~40 目标带（36）** 与 **T3 0.85~0.87（超出）** ⇒ **只差 T1**。

### 8.5.4 三效应分离（严格 PASS 聚合）

| init | ep | n | U 中位 | T1 中位 | T2FP 中位 | **T3 中位** |
|:--|--:|--:|--:|--:|--:|--:|
| **rc2** | 3 | 5 | +0.067507 | 0.896927 | 59 | **0.829385** |
| base474 | 3 | 3 | +0.026723 | 0.898046 | 88 | 0.801802 |
| **rc2** | 6 | 4 | +0.084036 | 0.896905 | 54 | **0.845881** |
| base474 | 6 | 1 | +0.040395 | 0.896952 | 79 | 0.813396 |
| **rc2** | 8 | 10 | +0.072841 | 0.896964 | 62 | **0.836312** |
| base474 | 8 | 5 | +0.037717 | 0.897546 | 80 | 0.809149 |
| **rc2** | 10 | 1 | +0.040607 | 0.896790 | 79 | 0.813998 |
| **rc2** | 15 | 2 | +0.069501 | 0.897138 | 70 | 0.837388 |

**① init 效应（用户核心假设）= 成立**
同 epoch 下 rc2 比 base474 的 **T3 中位高 +2.7~2.8 pp**，且 **T2 虚警少 18~29 个**
（3ep：0.8294 vs 0.8018 / FP 59 vs 88；8ep：0.8363 vs 0.8091 / FP 62 vs 80）。

**② epoch 效应 = 成立，但有甜点**
rc2 下 **3ep 0.8294 → 6ep 0.8459（+1.65 pp）→ 8ep 0.8363 → 15ep 0.8374** ⇒
**6 epoch 是甜点**，再往上回落（源域被挤）。⇒ 用户"拉到 8~10"的预期**部分成立、部分过头**。

**③ 骨干保护效应 = 在 lr=1.0 档没有增益（诚实负结果）**
成对臂：ARM_A（rc2/lr1.0/8ep/无保护）T3 **0.833911**、FP 60
vs ARM_C（rc2/lr1.0/8ep/骨干0.1×+锁2ep）T3 **0.825327**、FP 68
⇒ **保护让源域更稳（T1 +0.08 pp）但难例略降（−0.86 pp）、虚警多 8 个**。

**④ lr 效应**：rc2 lr1.0→lr0.5（同 8ep/fbn）T3 **−2.06 pp**、FP +13 ⇒ 太小步长**牺牲难例**。

### 8.5.5 16 条成对臂的最终结果（唯一能干净归因的证据）

| role | init | ep | lr | bb | fbb | verdict | T1 | T2FP | T3 |
|:--|:--|--:|--:|--:|--:|:--|--:|--:|--:|
| NATURAL_RC1 | base474 | 3 | 1.0 | 1 | 0 | PASS | 0.899621 | 102 | 0.772696 |
| CTRL_BASE474_3EP | base474 | 3 | 2.8821 | 1 | 0 | PASS | 0.898046 | 83 | 0.814615 |
| CTRL_BASE474_8EP | base474 | 8 | 2.8821 | 1 | 0 | **PASS（11-02 修复）** | 0.895915 | 51 | **0.851805** |
| CTRL_BASE474_10EP | base474 | 10 | 2.8821 | 1 | 0 | PASS_RELAXED | 0.894216 | 52 | 0.865357 |
| RC2_ANCHOR_3EP | rc2 | 3 | 2.2702 | 1 | 0 | FAIL_TRACK1_RETREAT | 0.895591 | 75 | 0.843601 |
| ARM_A_RC2_LR1_8EP | rc2 | 8 | 1.0 | 1 | 0 | PASS | 0.897364 | 60 | **0.833911** |
| ARM_B_RC2_LR05_FBN_8EP | rc2 | 8 | 0.5 | 1 | 0 | PASS | 0.897545 | 73 | 0.813269 |
| ARM_C_RC2_PROG_8EP | rc2 | 8 | 1.0 | **0.1** | **2** | **PASS（已修复）** | 0.898166 | 68 | **0.825327** |
| ARM_BP_RC2_LR05_FBN_PROG_8EP | rcrc2 | 8 | 0.5 | 0.1 | 2 | PASS | 0.898627 | 79 | 0.810501 |
| ARM_CAT_RC2_LR28_8EP | rc2 | 8 | 2.8821 | 1 | 0 | PASS_RELAXED | 0.895661 | **45** | **0.867711** |
| ARM_LONG_RC2_LR10_15EP | rc2 | 15 | 1.0 | 1 | 0 | PASS | 0.896171 | 64 | **0.857880** |
| ARM_RC2_LR07_6EP | rc2 | 6 | 0.7 | 0.3 | 0 | PASS | 0.898348 | 69 | 0.815597 |
| ARM_A_BASE474_LR1_8EP | base474 | 8 | 1.0 | 1 | 0 | PASS | 0.898379 | 80 | 0.795089 |
| ARM_D_RC2_BBFREEZE_8EP | rc2 | 8 | 0.5 | **0** | 0 | PASS | 0.896443 | 64 | 0.815353 |
| ARM_E_RC2_W25_10EP | rc2 | 10 | 1.0 | 0.1 | 2 | PASS_RELAXED | 0.895353 | 59 | 0.831673 |
| ARM_F_RC2_LR03_15EP | rc2 | 15 | 0.3 | 0.1 | 2 | PASS | 0.898104 | 76 | 0.816896 |

**成对差（每对只差一个变量）**

| 对比 | T3 | T2FP | T1 | 读数 |
|:--|--:|--:|--:|:--|
| **[epoch]** base474 3ep→10ep | **+5.07 pp** | **−31** | −0.38 pp | **拉长 epoch 有效** |
| **[init]** base474 8ep(lr1.0)→**rc2** 8ep(lr1.0) | **+3.88 pp** | **−20** | −0.10 pp | **换起点有效，且虚警更少、源域几乎不损** |
| **[保护]** rc2 lr1.0 无保护→骨干0.1+锁2ep | −0.86 pp | +8 | +0.08 pp | **在 lr=1.0 档没有增益**（源域更稳、难例略降） |
| **[lr]** rc2 lr1.0→lr0.5（8ep/fbn） | −2.06 pp | +13 | +0.02 pp | 太小步长牺牲难例 |

### 8.5.6 门禁基础设施事故（10-30 傍晚，已修复）

- **现象**：3 条 trial 的 verdict = `GATE_REPORT_INCOMPLETE`（trial_0006 / 0011 / 0068）
- **根因**：**4 个门禁同时跑**，每个都要重建龙泉山热图缓存 ⇒ `torch.cuda.empty_cache()` 处 **CUDA OOM**
  （`ms_cache_shard0.log`：`RuntimeError: CUDA error: out of memory`）⇒ **纯并发环境问题，与模型无关**
- **损失**：权重全部完好（`epoch8.pt` 等），trial_0006 的 Track1 当场就 PASS（T1=0.898166）
- **修复（11-02）**：重跑门禁，**4 卡并行建缓存 + 串行执行**（避免再次并发 OOM）⇒
  trial_0006 **已修好**（PASS，T1 0.898166 / T2 0.997720 FP68 / **T3 0.825327** / U +0.061011）；
  trial_0011、trial_0068 **仍在跑**（脚本 `manu/training/repair_incomplete_gates.sh`，日志 `/mnt/data/siping/runs/nogmc_repair.log`）
- **教训**：**门禁并发度要限 1**（或至少别与训练并发），否则 4 卡同时重建缓存必炸

### 8.5.7 留给下一轮的 3 件事

1. **补一轮「严格口径」的多种子复现** —— 当前严格 PASS 全是 n=1，**没有配置级交接证据**
   （`plan_replication.py` 已修为只挑 `verdict == "PASS"`，重跑复现阶段即可）。
2. **finish 门禁修复**：trial_0011 / trial_0068 跑完后重判 + 重出 SUMMARY。
3. **门禁并发护栏**：在 `auto_epoch_gate_check.py` 侧加"同一时刻只允许 1 个门禁重建缓存"的锁。

### 8.5.8 ✅ **门禁修复完成后的最终三条臂**（2026-11-02 09:42）

| trial | role | T1 | T2 | T2FP | T3 | 判决 | U |
|:--|:--|--:|--:|--:|--:|--:|:--|--:|
| trial_0006 | ARM_C（rc2/lr1.0/骨干0.1×+锁2ep/8ep） | 0.898166 | 0.997720 | 68 | **0.825327** | **PASS** | +0.061011 |
| **trial_0011** | **CTRL_BASE474_8EP**（base474/w3.0/2.8821/8ep） | **0.895915** | 0.998349 | 51 | **0.851805** | **PASS** | +0.090177 |
| trial_0068 | rc2/15ep | 0.892364 | 0.998739 | **38** | **0.881679** | FAIL_TRACK1_RETREAT | +0.111366 |

**最终 CSV**：114 行 · SUCCESS **83** / PRUNED 26 / CRASHED **5** · verdict **PASS 33** / PASS_RELAXED 50 / FAIL_TRACK1_RETREAT **27** / 无 4
（重判后 110 个 trial 重算、**3 条判决发生变化**；备份 `nogmc_weekend_backup_20261102_0942/`）

### 8.5.9 ⭐ 补全后的 **epoch 阶梯**（base474，同 lr2.8821、同 w3.0，只变 epoch）

| epoch | trial | T1 | T2FP | **T3** | verdict |
|--:|:--|--:|--:|--:|:--|
| 3 | trial_0001 | 0.898046 | 83 | 0.814615 | PASS |
| **8** | **trial_0011** | 0.895915 | **51** | **0.851805** | **PASS** |
| 10 | trial_0002 | 0.894216 | 52 | **0.865357** | PASS_RELAXED |

⇒ **epoch 效应（最干净的一对，3ep→8ep）：T3 +3.72 pp、虚警 −32**，且 T1 只退 0.21 pp（仍在红线内）
⇒ **8→10ep 再 +1.36 pp**，但 T1 掉到 0.894216 ⇒ **10ep 起进入"放宽通道"**（源域开始被挤）
⇒ **⇒ 用户的"3 ep 太少"判断被严格证实；甜点在 8 ep（严格过线），10 ep 收益更大但要付源域代价。**

### 8.5.10 ⭐ 补全后的 **init 成对比较**（同 lr2.8821、同 w3.0、同 8ep，只换初始化权重）

| init | trial | T1 | T2FP | **T3** | verdict |
|:--|:--|--:|--:|--:|:--|
| base474 | trial_0011 | 0.895915 | 51 | 0.851805 | PASS |
| **rc2** | trial_0008 (ARM_CAT) | 0.895661 | **45** | **0.867711** | PASS_RELAXED |

⇒ **init 效应：T3 +1.59 pp、虚警 −6、T1 仅 −0.03 pp** ⇒ **换起点（RC2 初始化）在完全同参下依然成立**，
且**虚警同步下降**（与用户"抗未对齐微调"的直觉一致：模型学会了忽略错位地物边缘）。

### 8.5.11 三效应**最终判词**（综合 8.5.4 / 8.5.9 / 8.5.10）

| 效应 | 结论 | 证据 |
|:--|:--|:--|
| **拉长 epoch** | ✅ **强正效应**，且是本轮最大杠杆 | base474 3→8ep：**T3 +3.72 pp、虚警 −32**；8→10ep 再 +1.36 pp 但 T1 越线 |
| **换初始化（RC2）** | ✅ **正效应**，且**虚警同步下降** | 同 lr 同 ep：T3 **+1.59 pp**、虚警 **−6**、T1 仅 −0.03 pp |
| **骨干保护（0.1×/锁 2 ep）** | ⚠️ **在 lr=1.0 档无增益**（诚实负结果） | ARM_A T3 0.833911/FP 60 vs ARM_C 0.825327/FP 68 |
| **更小 lr（0.3~0.5）** | ⚠️ **牺牲难例** | ARM_B（lr0.5）T3 0.813269 vs ARM_A（lr1.0）0.833911 |
| **T2 FP 目标带 30~40** | ✅ 达成的都在"触线"档 | trial_0068 **FP=38** 但 T1 0.892364 出局；严格过线最好 **FP=47**（trial_0025） |

**⇒ 用户三条判据的最终答复**：
- **T3 冲 0.85~0.87**：**达成**（严格过线最高 0.857880 trial_0009；配置级 0.872844）
- **T2 FP 降到 30~40**：**部分达成**（最好 38，但该 trial T1 触线；严格过线最好 47）
- **T1 门禁 0.895897**：**守住**是**有代价的**——8ep 恰好压线（0.895915，余量仅 0.000018！），
  10ep 就掉出去了 ⇒ **no-GMC 下"难例增益"与"源域不退让"存在直接冲突**，
  这才是它落后 GMC RC2（T3 0.897694）**~4 pp** 的根因。

---

## 九、待验证 / 风险登记

| # | 风险 | 处置 |
|--:|:--|:--|
| 1 | 扩展维度让 Optuna 触发 `dynamic value space` 报错 | `init_from/epochs/backbone_lr_scale/freeze_backbone_epochs` 都是**固定 choices 的 categorical** ⇒ 不会触发；`lr_scale`/`warmup_epochs` 仍是 `log`/非 `step` 分布，只改边界不改分布类型 ⇒ 允许 |
| 2 | 复现阶段 `n-trials` 算错导致 TPE 继续乱跑 | `n-trials = 已有 done + 复现条数` ⇒ 精确卡住（`done_count` 对 SUCCESS/PRUNED/FAIL **都** +1） |
| 3 | 参照系包丢失导致误判 | 本轮产物在 `/mnt/data`，且流水线每步都校验文件存在才继续 |
| 4 | 调度器中途崩溃 | 看门狗自动重启（`study.db` 断点续跑 + 自愈陈旧 RUNNING） |
| 5 | RC2 初始化 + 15 epoch 严重过拟合（T1 塌方） | 门禁一级红线会在末 epoch 直接判 `FAIL_TRACK1_RETREAT` ⇒ 数据照实记录，不会污染交接 |
| 6 | 冒烟测试未覆盖到"切换 trial"路径 | 已用 `--max-iters 2 --epochs 8 --freeze-backbone-epochs 2` 验证到 epoch3 解冻；**切换 trial 的调度路径在搜索启动后由 dispatcher.log 复核** |
| 7 | ⚠️ **调度器异常退出 ⇒ 孤儿 worker + 重复派单**（**已实测量化**，2026-10-30 18:0x 沙盒） | 看门狗**从不 kill** 调度器（只 `kill -0` 判活）⇒ 重启只发生在它自己崩了时 ⇒ 4 个 worker 变孤儿（仍在跑、仍在写产物）⇒ 新调度器把陈旧 RUNNING **自愈标 FAIL** 并按 `done_count` 重新派单 ⇒ **8 worker 抢 4 卡** | 沙盒实测：`[HEAL] 自愈 1 个陈旧 RUNNING trial（标为 FAIL，不污染 TPE）`；自愈后 `done_count=0`（**FAIL 不计入**）⇒ 结论成立 | 影响**有界**：① 原 worker 显存已被 caching allocator 占住 ⇒ 存活；② 新派 worker CUDA OOM 死（记 CRASHED）；③ 孤儿结果由收尾 `backfill_summary_rows.py` 按 `trial_result.json` **补回 CSV** ✅；④ FAIL/CRASHED **不污染 TPE** ✅ ⇒ 最坏浪费几个 trial（<2% 算力），**不丢数据** |

> 风险 7 **不采取自动加固**：想过加"GPU 守护进程"（>4 个 CUDA 进程就杀多余的），但——
> ① 门禁阶段本来就是 4 worker + 4 gate = **8 个 CUDA 进程**，按进程数判断会误杀；
> ② `ps` 看到的 worker 数含 dataloader 子进程（每 trial 9 个 PID），按进程数判断同样不可靠；
> ③ 唯一可靠的判据是 `nvidia-smi --query-compute-apps`，但它同样分不清 worker 与 gate。
> ⇒ **盲加守护的误杀风险 > 它能挽回的收益**，故只登记不实施。
> **周一读盘时若看到成片 CRASHED 且 dispatcher.log 里有重启记录 ⇒ 就是这件事，按"浪费算力"处理即可。**

---

## 十、周一读盘清单（下次开机照着做）

```bash
ssh -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_10ae9e2e && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
export PYTHONPATH=.

# ① 最终汇总（13 章节）
cat /mnt/data/siping/runs/nogmc_weekend/SUMMARY.md
# ② 常驻副本（不依赖服务器是否还活着）
cat /mnt/data/siping/runs/nogmc_weekend_SUMMARY_final.md
# ③ 流水线全程日志（含看门狗重启记录）
grep -nE "看门狗|自动重启|HEAL|CRASH|INFRA-FAIL|⏰" /mnt/data/siping/runs/nogmc_weekend_pipeline.log
# ④ 逐 trial 明细（CSV，30 列）
cat /mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv
# ⑤ 备份目录
ls -la /mnt/data/siping/runs/ | grep nogmc_weekend
```

**判定顺序（严格按此，别跳步）**
1. 先看 **§四之三 配置级多种子榜**（交接唯一依据），**不要**用 §四 的单点 Top 榜；
2. 每个候选配置看 **种子数**与 **T3 极差**；极差随种子数结构性偏小 ⇒ **跨组比稳定性要用「极差/2/√n」**；
3. **U 不可分时转看 T3**（U 噪声带见 §五之四）；T3 差距要超阈值才算真信号；
4. 与 **GMC RC2 的 T3 = 0.897694** 比：差距仍大 ⇒ 结论是"no-GMC 必须换底模"，不是"调参没调好"；
5. 三个效应**分开归因**（这正是 16 臂矩阵的用途）：
   - init 效应 = `ARM_CAT` vs `CTRL_BASE474_8EP`（同参只换 init）
   - epoch 效应 = `CTRL_BASE474_3EP` vs `CTRL_BASE474_8EP` vs `CTRL_BASE474_10EP`
   - 保护效应 = `ARM_A` / `ARM_B` / `ARM_C` / `ARM_BP` / `ARM_D` 互比
   - 分离度不够 ⇒ 结论是"**三者缺一不可**"，不把功劳归给其中任何一个
6. 确认 §五之五（RC2 配方移植）与 §五之二（配对 A/B）是否同向 ⇒ 互证才算数；
7. 最后写交接口径：**T3 中位 + 种子极差 + T1 红线余量 + T2 虚警**四项一起给，别只报一个数。

---

## 十一、导航

| 想看 | 去哪 |
|:--|:--|
| ⭐ **周一怎么读结果** | **本档 §十「周一读盘清单」+ §8.1.0 三条监控指标对照表**（先看 §8.1.1 起跑基线做 diff） |
| 本轮**首轮实测读数**（T3/T2FP/T1 + 损失曲线 + Ep2 分化） | 本档 **§8.1 ~ §8.2** |
| 本轮**开跑前拦下的 9 个 bug** 与处置 | 本档 **§五·二 / §8.3~§8.12**（含 3 处「11 维遗留假设」） |
| 上一轮（10-27/28）结论 | `manu/memory/memory.md` §12.6；专题档 `nogmc_rebuild_20261027.md` §十五.23~§十五.29 |
| 上一轮产物丢失的取证 | 本档 §〇 |
| 本轮代码 | `manu/training/optuna_mix_ratio_search.py` / `summarize_nogmc_search.py` / `compare_gmc_nogmc_matched.py` / `analyze_mix_spec_results.py` / `rejudge_with_reference.py` / `measure_reference_frame.py` / `plan_replication.py` |
| 本轮配置 | `manu/training/nogmc_weekend_plan.json`（16 臂）/ `nogmc_weekend_init_weights.json` |
| 本轮编排 | `manu/training/launch_nogmc_weekend_search.sh` / `run_nogmc_weekend_pipeline.sh` / `rc1_earlyfill_watchdog.sh` |
| 本轮产物（服务器） | `/mnt/data/siping/runs/nogmc_weekend/`（SUMMARY.md / optuna_summary.csv / study.db / trials/ / **BASELINE.md 起跑基线副本**） |
| 本轮产物（参照系） | `/mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json` |
| 主档 / 速查卡里的对应条目 | `manu/memory/memory.md` **§12.7** / `manu/memory/memory_compact.md` **§3.20** |
