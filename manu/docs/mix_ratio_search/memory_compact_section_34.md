
## §34 多轨判别标准（三级漏斗）+ 混合比例超参搜索体系（2026-10-21）

- 🔒 **判别标准**（`manu/training/mix_ratio_judgement.py`，自测 **43/43**）：
  **绝不用加权求和**（会用顺风轨大涨稀释致命缺陷）⇒ 用**三级漏斗**（顺序不可交换）：
  ① **安全红线一票否决**（相对**冻结基线**）：T2 F1 ≥ 0.985 且 ΔFP_T2 ≤ +10 且 ΔF1_T1 ≥ −0.005
  → 触碰任一条 ⇒ **剪枝 Pruned**；② **帕累托占优**（相对对手）⇒ 直接判胜，**不再看效用分**；
  ③ **效用裁决**（仅互有攻守时）：`U = ΔF1_T3 + 0.5·ΔF1_T2 − 2.5·max(0,−ΔF1_T1) − 0.0005·max(0,ΔFP_T2)`。
  **RC1(Ep3) 的 U = 0.027579 ⇒ "更好"的及格线。**
- ⚠️ **两个用真实门禁数据挖出的关键判定**：
  ① **效用分非反对称**（`max(0,·)` 惩罚项所致：U(Ep3 vs Ep4)=−0.000903 与 U(Ep4 vs Ep3)=−0.001097 **同号**）
  ⇒ 三级**必须比"同一基准（冻结基线）下的 U"**，否则与 objective 不一致；
  正确结论 `U(Ep3)=0.027579 > U(Ep4)=0.026178` ⇒ **Ep3 胜**。
  ② **帕累托维度集必须显式可配**（三预设：`user`={T1F1↑,T2F1↑,T3FP↓} / `core`=三轨F1↑ / `full`=六维含FP）：
  Ep3 ≻ Ep4 **仅在 `user` 下成立**（Ep3 的 T3-F1 实际低 0.001）；而 `full` 下 Ep1~Ep5 互不占优
  （三轨 F1 最优分散在 Ep1/Ep3/Ep5），**效用分裁决同样判 Ep3 胜** ⇒ **两条独立路径都证实 RC1=Ep3 最优**。
- 🔴 **另一条实证：Ep5 过不了一级红线**（ΔF1_T1 = −0.005236 < −0.005）⇒ 直接剔除。
  用户"锁定 Ep3 而非 Ep5"现在有了**红线级**理由（不只是边际收益递减）。
- 📐 **"真正的 1:4" = 口径 1（Batch 内采样概率）**：口径2 总帧数**物理不成立**（13,918×4=55,672 > 43,008，极限 1:3.09）、
  口径3 GT 框数 1:3.36、口径4 切片（路线 A 后才成立）。
  实现=`WeightedRandomSampler` 且**只替换官方 `build_dataloader` 管线里的索引采样器**
  （torch 1.13 `DataLoader.__init__` 用模块级 `RandomSampler`；数据集/BatchSampler/collate/worker/generator 全官方）
  ⇒ 杜绝 §十四「自写 DataLoader 致样本腰斩 + 凭空多 576 处假警」复现。
  **冒烟实测：目标 1:4.000 → 实测 1:4.106** ✅；官方数据集满额 13,918/43,008 ✅；warmup 890 步=0.5ep ✅；VRAM 11.28GB ✅。
- 🔬 **搜索体系**（`manu/training/optuna_mix_ratio_search.py`）：**4 卡并行**、每 trial **单 GPU + 3 epoch**、
  门禁在同卡串行（只门禁末 epoch，三个权重全留）；**objective = U**，**一级红线失败 ⇒ `TrialState.PRUNED`**；
  **可随时停止/继续**（SQLite study + 陈旧 RUNNING 自愈）；**后台常驻**（`setsid nohup` 独立 session，
  **宿主机断电/SSH 关闭都不影响**）；`--max-hours` 到点自动收尾；`status_mix_ratio_search.py` 随时可查
  进度/GPU/Top-N/**按 mix_spec 分组最优**/决赛结论。
  **搜索空间**：`mix_spec`（★关键：`natural`=路线 B 对照 1:3.09 + `w1.0/1.5/2.0/2.5/3.0/3.5/4.0/5.0/6.0/8.0`）、
  `lr_scale` 0.3~3.0、`warmup_epochs`、`weight_decay`、`focal_beta`、`offset_weight`、`mosaic`、`scale`、
  `translate`、`clip_max_norm`、**`freeze_bn`**；**种子 trial 自动注入**（natural + w2/w3/w4/w6，头几小时就有信号）。
  冻结不变：G1 严格冻结+eval 保护 BN、三组 lr 比例 1:5:10、wd 1e-4、只保留弱抖动、禁色彩/极性翻转。
- 📊 **时间预算**：每 trial ≈ 训练 21min + 单卡门禁 25min ≈ **46min** ⇒ 4 卡并行 **16h ≈ 83 个 trial / 20h ≈ 103 个**。
- **新增工具**：`mix_ratio_judgement.py`、`test_mix_ratio_judgement.py`、`optuna_mix_ratio_search.py`、
  `status_mix_ratio_search.py`、`launch_mix_ratio_search.sh`、`stop_mix_ratio_search.sh`。
  **运行手册**：`manu/docs/mix_ratio_search/README.md`（本地仓库与服务器同路径）。详见 `memory.md` §四十八。
