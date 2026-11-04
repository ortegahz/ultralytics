# No-GMC 定点打靶（2026-11-02 ~ 11-03 早）—— 专题档

> 新目标：**不再做宽泛超参盲搜**，把周末剩下的算力换成**确定性最高的定点打靶**。
> 窗口：**现在 → 2026-11-03（周二）09:00**（用户"我上班"时刻）⇒ 训练截止 **11-03 06:30**，
> 留 2.5 h 给门禁 + 收尾 + 冻结权重。
>
> ⚠️ **服务器时钟比本机慢约 52 分钟**（本机 10:43 时服务器是 09:51）⇒ **日志时间戳一律按服务器读**。

---

## 〇、定点打靶要解决什么（唯一矛盾）

周末 114 个 trial 的实测把矛盾钉死了：

| 现象 | 证据 |
|:--|:--|
| **T3 ≥ 0.87 的 trial 全部 T1 触线** | trial_0060 T3 **0.890979** / T1 0.892473 ❌ · trial_0087 T3 0.873983 / T1 0.893779 ❌ · trial_0113 T3 0.871705 / T1 0.894299 ❌ |
| **严格过线的 T3 上限只有 0.857880** | trial_0009（rc2/15ep/lr1.0）T1 0.896171 / FP **64** / T3 0.857880 |
| **保住 T1 的代价是虚警高** | 严格过线里 FP 最低的是 trial_0025（47），而 T3 高的都 FP 40~64 |

⇒ **唯一目标：把 T3 从 0.858 推过 0.87，同时把 T1 拉回 ≥ 0.895897、把 FP 从 64 压下来。**

---

## 一、三个变体（严格按用户文档执行）

### 变体 A：源域数据重混比例（`mix_spec` 的 UAV 侧 ↑）
`mix_spec "wN"` 的语义 = **UAV:LQ = N:1**（w3.0 ⇒ 源域 UAV 占 75%）。
用户的"源域权重从 1.0 提至 1.2~1.3"⇒ 相对 w3.0 放大 1.2~1.3× ⇒ **w3.6~w3.9** ⇒ 取 **w6.0 / w8.0**
（w6.0 = 85.7%、w8.0 = 88.9%）。**已在原 MIX_SPECS 里 ⇒ 零新代码。**

### 变体 B：渐进式解冻 + 学习率分层（骨干保护）
前 2 ep 锁死 Backbone（只训 Neck+Head），第 3 ep 起骨干按 `backbone_lr_scale` 低速更新
⇒ `freeze_backbone_epochs=2` + `backbone_lr_scale ∈ {0, 0.1}`。**这两个维度周末已实现且已跑过。**

### 变体 C：Top-3 权重融合（Model Soups / SWA）—— **零训练算力**
`W_fused = 1/N · Σ W_i`，同一盆地（都从 RC2 trial_0080 起步）内平均。
**实现要点（脚本 `manu/training/fuse_weights.py`）**
- **p0_highway 不平均**：各 trial 的 p0 来源不同（rc2 起步=训练过的、base474 起步=原始的）⇒ 取第一个模型的值；
- **BN running_mean/var 不平均**（另有 `--nobn` 变体）⇒ 只平均可训练权重；
- 输出 ckpt 与训练产物**同格式**（`{"model": state_dict, ...}`）⇒ 门禁 `load_hm_model` 能直接读。

---

## 二、聚焦搜索空间（`--focus-space`，把 TPE 从发散盲搜收窄）

| 维度 | 全空间 | **聚焦空间** | 收窄理由 |
|:--|:--|:--|:--|
| `mix_spec` | 11 个 | **w2.0 / w3.0 / w6.0 / w8.0 / natural** | w1.0/w1.5 已被证无望；w6.0/w8.0 是源域高占比方向 |
| `lr_scale` | [0.2, 3.0] | **[0.8, 3.0]** | 实测有效带 1.0~2.92 |
| `epochs` | 3/6/8/10/15 | **6/8/10/15** | **3 ep 已被证"半夹生"** |
| `backbone_lr_scale` | 4 个 | 不变 | 4 个取值都要试 |
| `freeze_backbone_epochs` | 0/2 | 不变 | — |
| `init_from` | base474/rc2 | 不变（rc2 为主，base474 只作对照） | init 效应已证 +1.6~2.9 pp |
| 其余 8 维 | 全空间 | 不变 | — |

⚠️ **必须用新 study-name**（`mix_ratio_search_nogmc_focus`）——categorical 取值集合被 study 记录后不可改。

---

## 三、30 条确定性臂（`manu/training/nogmc_focus_plan.json`，入队必跑；11-02 10:26 增补 8 条近靶心 + 11-02 11:47 增补 4 条近靶心 + 3 条 soup-init）

| role | init | ep | lr | mix | bb | fbb | 蓝本 | 改动 |
|:--|:--|--:|--:|:--|:--|:--:|:--|:--|
| FOCUS_T0025_SEED2 | rc2 | 6 | 2.80528 | w6.0 | 1.0 | 0 | trial_0025 | 换种子（**给最优严格 PASS 补多种子**） |
| FOCUS_T0025_SEED3 | rc2 | 6 | 2.80528 | w6.0 | 1.0 | 0 | trial_0025 | 第三个种子 |
| **FOCUS_A_W6_LR1_15EP** | rc2 | 15 | 1.0 | **w6.0** | 1.0 | 0 | trial_0009 | **变体 A**：mix w3.0→w6.0 |
| **FOCUS_A_W8_LR1_15EP** | rc2 | 15 | 1.0 | **w8.0** | 1.0 | 0 | trial_0009 | 变体 A 更激进 |
| **FOCUS_A_W8_T0025_6EP** | rc2 | 6 | 2.80528 | **w8.0** | 1.0 | 0 | trial_0025 | 变体 A 在甜点 ep 上 |
| **FOCUS_B_T0060_PROT15** | rc2 | 15 | 2.91912 | w3.0 | **0.1** | **2** | trial_0060 | **变体 B 核心**：骨干 0.1×+前2ep锁死，且 warmup 0.77→**1.2**（用户要求 ≥1 ep） |
| **FOCUS_B_T0087_PROT8** | rc2 | 8 | 2.66688 | w2.0 | **0.1** | **2** | trial_0087 | 变体 B：骨干 0→0.1×+锁2ep |
| **FOCUS_B_T0009_PROT15** | rc2 | 15 | 1.0 | w3.0 | **0.1** | **2** | trial_0009 | 变体 B：期望压虚警 64→<50 |
| FOCUS_B_T0018_PROT6 | rc2 | 6 | 2.92230 | natural | 0.0 | **2** | trial_0018 | 骨干已全冻，再加"前2ep锁" |
| FOCUS_C_W6_LR2_10EP | rc2 | 10 | 2.0 | w6.0 | 1.0 | 0 | — | 源域↑+中lr+中ep |
| FOCUS_T0009_W6_SEED2 | rc2 | 15 | 1.0 | w6.0 | 1.0 | 0 | trial_0009 | 变体 A 的多种子确认 |
| FOCUS_BASE474_W6_8EP | **base474** | 8 | 2.8821 | w6.0 | 1.0 | 0 | trial_0011 | **base474 对照**（成对归因） |
| FOCUS_BASE474_W6_10EP | base474 | 10 | 2.8821 | w6.0 | 1.0 | 0 | — | base474 对照 |
| FOCUS_RC2_W6_LR1_10EP | rc2 | 10 | 1.0 | w6.0 | 1.0 | 0 | — | 源域↑+小lr+中ep |
| FOCUS_RC2_NAT_LR1_15EP | rc2 | 15 | 1.0 | natural | 1.0 | 0 | trial_0009 | mix 换 natural（探测最优点） |
| **FOCUS_T0008_REPLAY** | rc2 | 8 | 2.8821 | w3.0 | 1.0 | 0 | trial_0008 | ★★**靶心原配方换种子**（T1 只差 0.000236） |
| **FOCUS_T0008_SEED2** | rc2 | 8 | 2.8821 | w3.0 | 1.0 | 0 | trial_0008 | ★★同上第二个种子 |
| **FOCUS_T0008_PROT8** | rc2 | 8 | 2.8821 | w3.0 | **0.1** | **2** | trial_0008 | ★★骨干保护**打在最准的靶心**上 |
| **FOCUS_T0077_PROT8** | rc2 | 8 | 2.7676 | w3.0 | **0.1** | **2** | trial_0077 | ★★**唯一 T3 已过 0.87** 的靶心 + 保护 |
| **FOCUS_T0077_SEED2** | rc2 | 8 | 2.7676 | w3.0 | 1.0 | 0 | trial_0077 | ★★原配方换种子 |
| **FOCUS_T0082_PROT8** | rc2 | 8 | 2.4220 | w3.0 | **0.1** | **2** | trial_0082 | ★★**缺口最小**（0.000146）靶心 + 保护 |
| **FOCUS_T0082_SEED2** | rc2 | 8 | 2.4220 | w3.0 | 1.0 | 0 | trial_0082 | ★★原配方换种子 |
| **FOCUS_RC2_W6_LR28821_8EP** | rc2 | 8 | 2.8821 | **w6.0** | 1.0 | 0 | trial_0008 | ★mix 插值点（单变量 w3.0→w6.0） |
| **FOCUS_SOUSTRICT3_W3_LR28821_8EP** | **soup_strict3** | 8 | 2.8821 | w3.0 | 1.0 | 0 | soup_strict3 | ★★★**soup 当初始化**（T1 已过线的起点，超参与 trial_0008 逐位相同 ⇒ 干净成对） |
| **FOCUS_SOUHIGH3NOBIN_W3_LR28821_8EP** | **soup_high3_nobn** | 8 | 2.8821 | w3.0 | 1.0 | 0 | soup_high3_nobn | ★★★**唯一 T3 已过 0.87 的起点**当初始化，与 trial_0008 成对 |
| **FOCUS_SOUHIGH3NOBIN_W3_LR15_6EP** | **soup_high3_nobn** | 6 | 1.5 | w3.0 | 0.5 | 1 | soup_high3_nobn | ★★同起点的小步精修（lr 1.5 / 6ep / 骨干 0.5×） |
| **FOCUS_T0056_SEED2** | rc2 | 8 | 2.4620 | w2.5 | 0.3 | 0 | trial_0056 | ★★近靶心多种子（缺口 0.000725） |
| **FOCUS_T0064_SEED2** | rc2 | 6 | 2.7750 | w3.0 | 1.0 | 0 | trial_0064 | ★★近靶心多种子（缺口 0.000788，6ep 更快） |
| **FOCUS_T0108_SEED2** | rc2 | 8 | 2.3670 | w3.0 | 1.0 | 0 | trial_0108 | ★★近靶心多种子（缺口 0.000914，T2 FP 45） |
| **FOCUS_T0034_SEED2** | rc2 | 8 | 2.8041 | w2.5 | 0.3 | 0 | trial_0034 | ★★近靶心多种子（缺口 0.000922） |

**校验**：23 条全部落在聚焦空间内（含 `step=0.05` 的整除性）✅；真实 Optuna 路径 `enqueue → ask → suggest_params` 逐位一致 ✅；TPE 自由采样也稳稳落在聚焦区 ✅。
**同步**：plan 用 **scp→临时文件→校验 md5/JSON→`mv` 原子替换**（避免调度器在 10:32 启动那一刻读到半写文件）⇒ md5 `e5cba8fc4c0ebcbf3d692a92208de6a1` 本地/服务器一致。

**成本**：原 15 条 6×15ep(154min) + 3×10ep(109) + 2×8ep(91) + 4×6ep(73) = **1725 卡·min**；增补 8 条 8ep × 93min ≈ **743 卡·min** ⇒ 合计约 **2468 卡·min ≈ 41 卡·h** ⇒ 4 卡并行约 **10.3 h 训练**（周末实测 8ep 单 trial duration ≈ 92.8 min）；门禁另计。从 11-02 10:33 起算 ⇒ **约 11-02 21:00 完成 23 条确定性臂**，远早于周二 06:30 ⇒ 余量约 9.5 h 留给聚焦 TPE。

---

## 四、Candidate_v1（保底牌，已落盘）

```
/mnt/data/siping/runs/CANDIDATES/Candidate_v1_trial_0025_epoch6.pt   md5 4302b13de7a338e8b8c3882aac606482  21,301,027 B
/mnt/data/siping/runs/CANDIDATES/Candidate_v1.json   （完整 15 维配方 + 指标 + 红线余量）
```
- 出处：**周末 114 trial 里严格口径 PASS 且 U 最高**的单点（trial_0025）
- 指标：T1 **0.897516**（红线余量 **+0.16 pp**）· T2 0.998439（**FP 47**）· **T3 0.850972** · U **+0.093390**
- 配方：`w6.0 | lr2.8053 | warm1.463 | wd2.64e-4 | beta2.85 | off0.55 | mos0.10 | sc0.15 | tr0.05 | clip2.44 | fbnFalse | rc2 | 6ep | bb1.0 | fbb0`
- 选它的理由（vs trial_0009）：**U 更高、T1 余量大 5 倍、虚警少 17 个** ⇒ 更"稳"

---

## 五、本轮新修的 2 个工程问题

| # | 问题 | 根因 | 修法 |
|--:|:--|:--|:--|
| 1 | **门禁 CUDA OOM**（周末炸掉 3 条 trial） | **worker 训练进程仍把模型驻留在同一张卡**（~11 GB），而门禁的缓存重建子进程又跑在那张卡上 ⇒ 11 GB + 缓存峰值 | **worker 调门禁前 `model.to("cpu")` + `del model` + `gc.collect()` + `empty_cache()`**，把整张卡让给门禁子进程（新增 `[GATE-MEM]` 日志） |
| 2 | 聚焦模式与既有 study 冲突 | categorical 取值集合被 study 记录后不可改 | `--focus-space` **必须配新 study-name**；脚本已内置提醒 |

（另：`plan_replication.py` 在上一轮已修为只挑严格 `PASS`。）

---

## 六、流水线与时间锚点（服务器本地时间）

```
11-02 09:51  定点打靶流水线启动（训练截止 11-03 06:30）
             ① 变体 C：4 个 soup 融合权重各跑一次门禁（串行，~10 min/次 ⇒ ~40 min）
             ② 聚焦搜索：15 条确定性臂 → 聚焦 TPE（--focus-space）
             ③ 截止 → backfill → rejudge → SUMMARY → 备份 → 冻结权重
11-03 06:30  训练截止
11-03 ~08:00 门禁收尾 + 汇总 + 交接口径判定
11-03 09:00  交付（用户上班）
```

**产物**：`/mnt/data/siping/runs/nogmc_focus/`（SUMMARY.md · optuna_summary.csv · study.db · trials/ · soups/）
**日志**：`/mnt/data/siping/runs/nogmc_focus_pipeline.log`

---

## 七、监控

```bash
ssh -p 32222 huangzhe@192.168.99.40
tail -f /mnt/data/siping/runs/nogmc_focus_pipeline.log
tail -f /mnt/data/siping/runs/nogmc_focus/logs/dispatcher.log
cat  /mnt/data/siping/runs/nogmc_focus/status.json
# soup 门禁结果
for d in /mnt/data/siping/runs/nogmc_focus/soups/*/gate; do echo "--- $d"; python -c "
import json,sys
r=json.load(open(\"$d/gate_report.json\"))
print(\"  T1=%.6f T2FP=%s T3=%.6f verdict=%s\" % (r[\"tracks\"][\"track1_uav\"][\"finetune_peak\"][\"f1\"], r[\"tracks\"][\"track2_golden\"][\"fp\"], r[\"tracks\"][\"track3_blind\"][\"f1\"], r[\"verdict\"]))" 2>&1 | tail -1; done
```

---

## 八、变体 C 判负 + Pareto 冲突量化 + 靶心定位（11-02 10:30 增补）

### 8.1 变体 C（Model Soups）**四个门禁全部证伪**

| 融合权重 | T1 | T2 (FP) | T3 | 判定 | 说明 |
|:--|--:|--:|--:|:--|:--|
| `soup_strict3` | 0.896846 | 0.998200 (56) | 0.847521 | PASS | 严格过线的 3 模型融合；**T3 低于 Candidate_v1 的 0.850972**，T2 虚警更高 |
| `soup_strict3_nobn` | 0.896172 | 0.997930 (65) | 0.845592 | PASS | `--nobn` 反而更差 ⇒ **BN 不平均并无增益** |
| `soup_high3` | **0.894943** | — | — | **FAIL** | T3 最高的 3 个模型融合后 **T1 仍触线** ⇒ 融合救不回 T1 |
| `soup_high3_nobn` | **0.893832** | — | — | **FAIL** | 同上，且更差 |

**结论**：Model Soups **不构成突破，路线判负**。机理：T1 触线是**系统性偏置**（被融合的 3 个模型本来就全都触线），权重平均无法把一个整体偏置的簇拉回线内；`--nobn` 连"平均"带来的正则化收益都拿不到。

### 8.2 Pareto 冲突的量化（周末 114 条全量）

- 过 T1 红线（≥0.8958967482400269）的只有 **33/114**（⚠️ **修复后数字**，见 §8.4；修复前是 31），它们的 T3 分布：**max 0.857880 / p90 0.849318 / median 0.820477 / min 0.772696**
  ⇒ **严格口径下 T3 的实测上限 = 0.857880（trial_0009）**；T3 ≥ 0.87 在严格口径下**尚无先例**
- **T3 ≥ 0.86 的 31 条全部 T1 触线**，但缺口极小（**前 6 条缺口 < 0.001**）：

| trial | 配方 | T1 | 缺口 | T3 | T2 FP |
|:--|:--|--:|--:|--:|--:|
| trial_0082 | w3.0/lr2.4220/8ep/bb1 | 0.895751 | **−0.000146** | 0.865458 | 45 |
| trial_0008 | w3.0/lr2.8821/8ep/bb1 | 0.895661 | **−0.000236** | 0.867711 | 45 |
| trial_0056 | w2.5/lr2.4620/8ep/bb0.3 | 0.895172 | −0.000725 | 0.860100 | 67 |
| trial_0064 | w3.0/lr2.7750/6ep/bb1 | 0.895109 | −0.000788 | 0.862429 | 57 |
| trial_0108 | w3.0/lr2.3670/8ep/bb1 | 0.894983 | −0.000914 | 0.862801 | 45 |
| trial_0034 | w2.5/lr2.8041/8ep/bb0.3 | 0.894975 | −0.000922 | 0.862606 | 57 |

**⚠️ 修复后新浮现的两条**（原先因 CSV 指标为空被统计漏掉，见 §8.4）：
- **trial_0011** = `base474/w3.0/lr2.8821/8ep/bb1`，T1 **0.895915**（红线余量仅 **+0.0019 pp**）· T2 FP 51 · T3 **0.851805** · U 0.0901766
  ⇒ **严格 PASS 榜 T3 第 2**（仅次于 trial_0009 的 0.857880），但 T1 余量比 trial_0009（+0.0274 pp）还薄 ⇒ **不能当交接线**（一换种子就掉线）
  ⇒ 它同时是 **trial_0008 的 base474 版** ⇒ 与 rc2 版构成一组**干净的 init 成对**：
  **rc2 让 T3 +1.59 pp、T2 虚警 −6，但 T1 −0.00025**（又一次印证 init 效应）
- **trial_0006** = T1 0.898166 · T2 0.997720 · T3 0.825327 · U 0.0610112（严格 PASS，但 T3 偏低）
- **trial_0068** = `natural/lr2.8055/15ep/bb1/base474`，T3 **0.881679**（周末第 4 高）· T2 FP 38，但 T1 缺口 0.003533 ⇒ 不是靶心，仅作记录

**缺口是"硬缺口"（已核 `gate_report.json`）**：T1 门禁取 `tracks.track1_uav.finetune_peak.f1`（**峰值门限下的 F1 = 上界**），trial_0008 的 peak 在 th=0.30、`min_f1_required=0.8958967482400269` ⇒ **换阈值救不了**，只能靠训练本身或种子噪声跨越。
**附产物——trial_0008 属"门限漂移型差一点"而非散架**：`delta.mean_f1 = +0.0037197406`（15 档阈值平均 F1 差值为**正**）、`delta.peak_f1 = -0.0052352660`、`negative_bands = 8/15` ⇒ 它在 15 档里有 7 档优于基线，唯峰值档吃亏于**门限漂移**（模型峰 th=0.30 vs 基线峰 th=0.25）。⇒ 若未来要修订红线口径（**当前铁律不动**），这条会先受益。

### 8.3 靶心定位与打法选择（决定增补 8 条臂的依据）

**同簇内 T3 几乎不动、T1 在抖** —— w3.0/lr≈2.88/8ep/bb1 簇 5 条：

| trial | lr | ep | T1 | T3 |
|:--|--:|--:|--:|--:|
| trial_0008 | 2.8821 | 8 | 0.895661 | 0.867711 |
| trial_0017 | 2.8767 | 8 | 0.893778 | 0.868033 |
| trial_0036 | 2.9959 | 8 | 0.894430 | 0.866337 |
| trial_0081 | 2.9169 | 8 | 0.893407 | 0.864144 |
| trial_0002 | 2.8821 | 10 | 0.894216 | 0.865357 |

⇒ **T1 抖动跨度 0.0023（≈ 2.5 个缺口），而 T3 跨度仅 0.004（几乎不动）**
⇒ **最优打法 = 同配方多种子**（赌种子噪声把 T1 推过线、T3 不掉），而不是继续调超参
⇒ 辅以**骨干保护**（提 T1 的第二条候选机制）与 **mix 插值 w6.0**（实测 mix↑ ⇒ T1↑/T3↓，但 w8.0 代价太大：T1 0.897014 而 T3 掉到 0.841839）

---

## 八·五、第 10 个真 bug：`rejudge` 不回填 `metrics` ⇒ **严格 PASS 的强候选在汇总里凭空消失**（11-02 11:40 修）

### 8.5.1 根因链（实测复现）

```
门禁基础设施失败（Track1 推理 CUDA OOM）
  ⇒ worker 写 result["metrics"] = None + gate_verdict="GATE_REPORT_INCOMPLETE"
  ⇒ 【关键】后来门禁**重跑成功了，但 trial_result.json 再没人更新过**（worker 早已退出）
  ⇒ metrics 永久为 null
  ⇒ worker 写 CSV 时 t1_f1/t2_f1/t3_f1 等 6 列**全空**
  ⇒ summarize_nogmc_search.py 直接 `continue` 跳过这些行（`any(v is None ...)`）
  ⇒ **trial_0006 / trial_0011 这两条严格 PASS 被静默丢掉**
  ⇒ 连带污染：T3≥0.86 的清单漏了 trial_0068（T3 0.881679，周末第 4 高）
```

### 8.5.2 修法（`manu/training/rejudge_with_reference.py`）

重判手里**本来就有完整 `gate_report`**（口径纪律："指标一个都不重算，只用落盘的 gate_report"）⇒ 顺手回填：

1. `res["metrics"] = m`（真实三轨指标）
2. CSV 同步段回填 **6 个指标列**（`t1_f1/t1_fp/t2_f1/t2_fp/t3_f1/t3_fp`），值为 None 时**不覆盖**原值
3. 拿到完整 gate_report ⇒ 旧 `gate_error` / `infrastructure_failure` **已证伪** ⇒ 清掉（并记 `gate_error_resolved_at`）
4. ⚠️ **`status` 必须在回填之后算**：`_status_of` 会看 `infrastructure_failure`，清了它之后指标不达标的 trial 应判 **PRUNED**（剪枝）而不是 CRASHED

### 8.5.3 验证证据

- 周末 114 trial 重跑 rejudge --apply：**110 个重判，只有 1 个判定变化**（`trial_0068: CRASHED→PRUNED`，verdict 不变）⇒ 其余判决**逐位未动**
- 回填结果：`trial_0006` T1 0.898166 / T3 0.825327、`trial_0011` T1 0.895915 / T3 0.851805 ⇒ **原本空白的指标全部补齐**
- 修复后重算：过 T1 红线 **31→33 条**；T3≥0.86 **30→31 条**；严格 PASS 榜 T3 第 2 变成 trial_0011
- 回归：**test_mix_ratio_judgement 71/71 ✅、test_status_mix_ratio_search 54/54 ✅**
- 备份：`/mnt/data/siping/runs/nogmc_weekend_rejudge_bak/`（110 个 trial_result.json，444 KB）

### 8.5.4 新工具：`manu/training/judge_handover_candidate.py`（交接口径判定器，周二交卷用）

把散在 memory 里的交接规则**代码化**（一条命令出 Markdown + JSON 报告）：

```bash
python manu/training/judge_handover_candidate.py \
  --csv   /mnt/data/siping/runs/nogmc_focus/optuna_summary.csv \
  --ref   /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --out-md   /mnt/data/siping/runs/nogmc_focus/HANDOVER.md \
  --out-json /mnt/data/siping/runs/nogmc_focus/HANDOVER.json \
  --min-seeds 2
```

- **严格口径**：`gate_verdict == "PASS"` 才进交接；`PASS_RELAXED` / `FAIL_TRACK1_RETREAT` 只进附录（诊断）
- **指标不全的行一律排除**（无法验证 ≠ 可交接）并显式计数上报
- 按 **15 维配方**分组 → 组内**中位** → **噪声带 = 极差/2/√n**
- 判定链：**U 可分 ⇒ 取 U 最高；U 不可分 ⇒ 看 T3 中位；T3 仍不可分 ⇒ 取种子数多者**
- **阻塞项显式输出**（种子数不足 / T1 余量过薄 / T3 未达 0.87 / T2 虚警超限）
- 退出码：0 = 可交接；**3 = 有候选但种子数不足**；4 = 无严格 PASS
- **冒烟验证（周末 114 条）**：独立复现了人工判定（严格 PASS 33 组全 n=1 ⇒ U/T3 均不可分 ⇒ 取 U 更高者 = **trial_0025**），并如实报出"只有 1 个种子 ⇒ 交接口径尚不成立" + "T3 未达 0.87（差 1.903 pp）"

---

## 八·六、**Candidate_v1 按交接口径站不住**（11-02 11:45 首批结果实测）

### 8.6.1 事实

本轮首批门禁（trial_0000/0001 = **trial_0025 配方的种子 200/201**）：

| trial | T1 | 红线余量 | T2 FP | T3 | verdict |
|:--|--:|--:|--:|--:|:--|
| 周末 trial_0025（原种子） | 0.897516 | **+0.001619** | 47 | 0.850972 | PASS |
| trial_0000 (seed 200) | 0.895692 | **−0.000205** | 65 | 0.845578 | PASS_RELAXED（严格 FAIL） |
| trial_0001 (seed 201) | 0.894412 | −0.001485 | 46 | 0.851568 | PASS_RELAXED（严格 FAIL） |

⇒ **同配方种子噪声跨度 0.0031**（原种子是运气好的高值）⇒
**三种子中位 T1 = 0.895692 < 红线 0.8958967** ⇒
**trial_0025 这个单点虽"严格 PASS 且 U 最高"，按「同超参多种子中位」口径并不成立。**

⇒ 这正是口径铁律要防的情形：**单 trial 的 U 受种子噪声支配**（memory §12.6.2）。

### 8.6.2 应对（打靶重心转移）

把预算从"赌单点运气"转向"**T1 余量能扛住种子噪声的配方 + 多种子**"：
`trial_0083`（余量 **+0.1117 pp**，严格 PASS 里最大）· `trial_0009`（+0.0274 pp，T3 最高 0.857880）·
`trial_0018`（+0.0308 pp）· `trial_0021`（+0.0397 pp）⇒ 全部补"原配方换种子"臂（精确复制，§8.7）。

### 8.6.3 判定口径不变

`PASS_RELAXED` **不得进交接**（铁律）⇒ trial_0000/0001 只能进附录诊断。

---

## 八·七、本轮新增 3 个工具（全部有实测证据）

### 8.7.1 `judge_handover_candidate.py` —— 交接口径判定器（周二交卷用）

一条命令出 Markdown + JSON：严格通道筛选 → 15 维配方分组 → 组内中位 → **噪声带 = 极差/2/√n** →
判定链（**U 可分⇒取 U 最高；U 不可分⇒看 T3 中位；T3 仍不可分⇒取种子数多者**）→ **显式输出阻塞项**。
- 指标不全的行**一律排除**（无法验证 ≠ 可交接）并计数上报
- 退出码：`0` 可交接 / **`3` 有候选但种子数不足** / `4` 无严格 PASS
- **冒烟（周末 114 条）**：独立复现人工判定（trial_0025），并如实报"只有 1 个种子 ⇒ 交接口径尚不成立"

### 8.7.2 `build_focus_arms_from_study.py` —— 从 `study.db` **精确**复制配方

`optuna_summary.csv` 的超参是**写盘时格式化过的**（`lr:.4f` / `clip:.2f`）⇒ **照抄会丢精度**：

| 真实值（study.db） | CSV |
|:--|:--|
| `lr_scale = 2.922306436248727` | `2.9223` |
| `clip_max_norm = 5.563409172338667` | `5.56` |
| `weight_decay = 0.00014963699994362378` | `0.00014964` |

⇒ 照抄 CSV 会让「多种子对照」退化成「近似配方对照」，**证据链失效**。
本工具直读 `trial.params`（原始 float）+ 可选 override ⇒ **实测修正了 12 处精度丢失**（trial_0077/0082/0018/0021/0056/0064/0108/0034/0083…）。
臂定义放在 `manu/training/nogmc_focus_arms.json`（role/seed/trial/note/override）。

### 8.7.3 `seal_candidate.py` —— 按判定结果**固化** RC（替代手工复制）

- **只固化严格 PASS + 指标齐全**的 trial（`PASS_RELAXED` 绝不会被封成 RC）
- 多种子时封**中位 trial**（三轨到组中位距离最小），**不封最高 trial**
- 副本 `chmod 0444`；登记文件含 md5 + sha256 + 大小 + 三轨 + **15 维配方** + 判定链 + **阻塞项** + 复现命令
- **交叉验证（强证据）**：拿周末 trial_0025 冒烟 ⇒ 产出 md5 **`4302b13de7a338e8b8c3882aac606482`**、
  21,301,027 B ⇒ **与上一轮手工固化的 `Candidate_v1` 逐位一致** ⇒ 固化流程可信
- 加载校验：523 张量 / 29 个 p0 / `p0_kwargs` 齐 / ckpt 结构完整 / 登记 md5 == 实际 md5 ✅

### 8.7.4 soup-init 训练路径风险**已排除**（省下一次冒烟训练）

worker 用 `load_hm_model(..., use_p0_highway=False)` 加载（去掉 p0 分支），训练落盘时
`state.update(p0_tensors)` 只是把源 ckpt 的 p0 张量**原样搬运**（**不参与训练**）⇒
soup 的 p0（来自 trial_0087）与 rc2 的 p0（trial_0080）结构一致（`p0_kwargs` 已验证相同），
训练产物格式完全相同 ⇒ **soup 当 init 零结构风险**（且 4 次 soup 门禁已实测 `load_hm_model` 能读）。

---

## 八·八、TPE 采样分布体检 ⇒ 发现 focus 空间仍有浪费，已用确定性臂封堵（11-02 11:55）

### 8.8.1 离线模拟（1500 次，TPESampler 冷启动，与调度器同一 `suggest_params`）

| 维度 | TPE 采样中位 | 周末已知好点 | 浪费后果 |
|:--|--:|:--|:--|
| `warmup_epochs` | **0.795**（75% < 1.08） | 1.125~1.5 | **违反用户「warmup ≥ 1 ep」要求** |
| `lr_scale` | **1.61**（75% < 1.99） | 2.37~2.92 | 远离靶心簇 |
| `init_from` | base474 **872** : rc2 **628** | rc2 全面更优（+2.7~2.8 pp） | **一半预算花在更差的 init 上** |
| `epochs` | **15ep 458 条**（最多） | 8ep 甜点 | 1/3 预算花在最贵的臂上 |
| `mix_spec` | w8.0 394（最多） | w3.0/w2.5 近靶心 | w8.0 实测 T3 掉到 0.8418 |

⇒ focus 空间已排除最离谱的（3ep、w1.0/w1.5、lr 收到 [0.8,3.0]），但**冷启动 TPE 的均匀采样仍大量落在"已验证次优"区**。

### 8.8.2 为什么不能直接改搜索空间

**本轮 study 已启动 ⇒ Optuna 的分布已记录**：
- categorical 集合改 ⇒ `CategoricalDistribution does not support dynamic value space`
- float 区间改 ⇒ `Cannot set different distribution`
⇒ **本轮锁死**；改空间必须换新 `study-name`（而换 study 会让已跑完的 40 臂结果无法复用 ⇒ 更亏）。

### 8.8.3 封堵手段：**用确定性臂替代随机采样**（计划 35 → **40 臂**）

全部基于**最准靶心 trial_0008** 的精确参数 + 单变量 override：
`FOCUS_T0008_WARMUP12`（warmup 1.5→1.2）/ `FOCUS_T0008_WARMUP18`（→1.8）/ `FOCUS_T0008_LR26`（lr→2.6）/
`FOCUS_T0008_BB03`（bb 1.0→0.3、不锁骨，补第三档梯度）/ `FOCUS_T0008_W25`（mix w3.0→w2.5）

⇒ 用"已知好点 × 单变量扰动"**正面覆盖 TPE 会浪费的那几维**。

### 8.8.4 时间预算（40 臂）

`7×15ep(154) + 20×8ep(93) + 11×6ep(75) + 1×10ep(109) = 3872 卡·min ≈ 64.5 卡·h`
⇒ ÷4 卡 ≈ **16.1 h** ⇒ 自 10:33 起 ⇒ **约 11-03 02:40 跑完 40 臂**（含门禁，周末实测 8ep 臂 duration 92.8 min 已含门禁）
⇒ 距训练截止 06:30 只剩 **~3.8 h 给 TPE ≈ 9 个 trial** ⇒ **成败几乎完全由确定性臂决定**（这正是扩容的意义）。

### 8.8.5 顺带修掉 watcher 的一个**死等**逻辑

`restart_dispatcher_inject_arms.py` 原判定是"plan 里所有非 soup 臂都到终态"⇒
plan 运行期一旦增补（35→40 臂），未入队的新臂会让它**永远等不到**（只能等超时）。
⇒ 改为"等**已入队**的 plan 臂（`user_attrs.enqueued_by == "enqueue_plan"`）全部终态"⇒ 新臂不参与等待 ✓
（实测：修前报 `0/37`，修后报 `2/23`）。

---

## 八·九、交付工具链（周二 09:00 一条命令收工）

### 8.9.1 `finalize_nogmc_focus.sh` —— 收尾交付（**已端到端冒烟**）

```
① backfill（补齐 CSV 行）
② rejudge（权威重判 + **回填 metrics**）
③ summarize（SUMMARY.md）
④ judge（**交接口径判定** ⇒ HANDOVER.md / HANDOVER.json）
⑤ seal（按判定结果固化 RC；判定器 rc=4 时**自动拒绝固化**）
⑥ 备份 + 打印交接摘要
```

```bash
cd /tmp/pycharm_project_10ae9e2e && bash manu/training/finalize_nogmc_focus.sh
```
- `SKIP_SEAL=1` 可跳过固化；`RC_ID` / `RC_NAME` 可改候选编号与名称
- **冒烟**（周末数据实测）：判定器 rc=3（候选 trial_0025、n=1）+ 如实报出两个阻塞项 ✅

### 8.9.2 `resume_nogmc_focus.sh` —— **抗主机重启**的断点续跑

`run_nogmc_focus_pipeline.sh` **不可重入**（阶段① 会重跑 4 个 soup 门禁 ≈ 40 分钟纯浪费），
且 host 重启后没人记得该敲哪条命令。本脚本 = **只跑阶段②+③**：
- 复用已落盘的 soup 门禁报告；`study.db` 存在 ⇒ 自动按「参数集+种子」跳过已跑臂
- **训练截止时刻与主流水线一致**（默认 11-03 06:30）
- 阶段③ 额外产出 `HANDOVER.md` / `HANDOVER.json`（主流水线没有）
- `CHECK_ONLY=1` 只校验前置条件不启动（**安全自检**，已实测）

### 8.9.3 `monitor_focus_round.py` —— 常驻事件监控（只读，不干扰训练）

每 5 分钟读 study.db；每个 trial **首次到达终态**就把 `role / 三轨 / verdict / U` 追加进
`logs/events.log`，并每 15 分钟写 `logs/monitor_status.json` 快照。
⇒ 8+ 小时无人工干预期间也有**过程留痕**，不必靠临时起意去查。
（踩坑：`FrozenTrial` **没有** `name` 属性 ⇒ 用 `trial_{number:04d}` 拼。）

---

## 八·十、本轮首批 5 条结果（11-02 13:10）：**暂无更好的模型**，但拿到了 3 个有价值的负面结论

### 8.10.1 结果（服务器 13:31 时已终态 5/23）

| trial | 配方 | verdict | T1 | T2 FP | T3 | U |
|:--|:--|:--|--:|--:|--:|--:|
| trial_0000 | w6.0·lr2.805·6ep (T0025_SEED2) | PASS_RELAXED | 0.895692 | 65 | 0.845578 | 0.076668 |
| trial_0001 | w6.0·lr2.805·6ep (T0025_SEED3) | PASS_RELAXED | 0.894412 | 46 | 0.851568 | 0.086241 |
| trial_0004 | w8.0·lr2.805·6ep (**变体A**) | **PASS** | 0.896388 | 54 | 0.843366 | 0.081859 |
| trial_0002 | w6.0·lr1.0·15ep (**变体A**) | PASS_RELAXED | 0.895665 | 44 | 0.848286 | 0.086108 |
| trial_0003 | w8.0·lr1.0·15ep (**变体A**) | **PASS** | 0.896027 | 50 | 0.841121 | 0.079772 |

**对比周末严格 PASS 榜**：本轮严格 PASS 最高 U **0.081859** < 周末最高 U **0.093390**（trial_0025）；
本轮严格 PASS 最高 T3 **0.843366** < 周末最高 T3 **0.857880**（trial_0009）。
⇒ **暂时没有任何配置比周末已有候选更好。**

### 8.10.2 负面结论 1：**变体 A（源域比例↑）作为"过线手段"有效，作为"提 T3 手段"无效**

w6.0 / w8.0 三条**全部把 T1 稳住了**（0.8960~0.8964，两条严格 PASS），
但 **T3 全部掉到 0.841~0.848**（周末最好 0.857880）⇒
**mix↑ 的 T1 收益 < T3 损失** ⇒ 变体 A 不能解决核心矛盾（要 T3 ≥ 0.87）⇒ **该路线降级为"过线手段"备用**。

### 8.10.3 负面结论 2：**"把 T1 拉过线"的手段在当前配方族里都会同时压 T3**

本轮两条严格 PASS（trial_0003 / trial_0004）的 T3 都比 trial_0025 低 ⇒ 印证 §8.2 的 Pareto 冲突是**硬冲突**。

### 8.10.4 负面结论 3：**Candidate_v1 的"最好"确实是运气**

trial_0025 配方 3 个种子 T1 = **0.897516 / 0.895692 / 0.894412** ⇒ 中位 0.895692 **低于红线 0.8958967**
⇒ 严格按"同超参多种子中位"口径，**Candidate_v1 不成立**（周五已预告，见 §8.6）。

### 8.10.5 翻盘只剩三条未跑的路（20:55 重启后入队）

| 路线 | 臂 | 依据 |
|:--|:--|:--|
| **近靶心原配方换种子** | trial_0008/0082/0077/0056/0064/0108/0034（8 个配方） | T1 缺口 0.000146~0.000922，而**种子噪声跨度实测 0.0031** ⇒ 噪声量级足以跨过缺口 |
| **骨干保护（未验证机制）** | PROT8 ×3 | `bb=0.1×+锁2ep` 在 **lr≈2.88 档从未测过**（周末只在 lr=1.0 档测过） |
| **soup-init** | ×3 | `soup_high3_nobn` 是**唯一 T3 已过 0.87 的起点**（T3 0.874746 · T2 FP **39**） |
| 高余量多种子 | trial_0083(+0.1117 pp) / trial_0009(T3 最高) / trial_0018 / trial_0021 | 余量扛得住种子噪声 |

---

## 八·十一、**受控对分析**：找到了冲突的汇率，并据此设计了组合臂（11-02 14:15）

### 8.11.1 为什么不能用"分组中位 + 相关系数"

工具 `analyze_t3_reachability.py`（新建）先按 `mix_spec` 分组取中位再算相关，得到的结论**与原始单点数据方向相反**
（ratio↔T1 = **−0.464**、ratio↔T3 = **+0.412**，而单点数据是 mix↑⇒T1↑/T3↓）。
**根因**：周末 TPE 采样里 **mix 与 epochs/lr/backbone 强耦合**（w8.0 多落在 15ep+lr1.0、w3.0 多落在 8ep+lr2.88）
⇒ 分组中位把 epoch/lr 的效应误算成 mix 的效应，**连符号都会反**。
⇒ **结论只认「只差一个维度」的受控对**（工具已内置该口径，并自动识别耦合警告）。

### 8.11.2 受控对结果（周末 114 + 本轮 5 = 119 条里共 **12 对**干净受控对）

| 变动维度 | 受控对数 | 该维度变大时 T1 上升 | T3 下降 | ΔT1 中位 | ΔT3 中位 |
|:--|--:|:--:|:--:|--:|--:|
| **mix_spec** | 3 | **3/3** | **3/3** | **+0.000696** | **−0.007165** |
| lr_scale | 2 | 0/2 | 0/2 | −0.002084 | **+0.045258** |
| epochs | 2 | 0/2 | 0/2 | −0.001446 | +0.018761 |
| init_from | 2 | 0/2 | 0/2 | −0.000634 | +0.027364 |
| weight_decay | 2 | 2/2 | 1/2 | +0.002464 | +0.002399 |
| backbone_lr_scale | 1 | 1/1 | 1/1 | +0.001102 | −0.002084 |

### 8.11.3 两条关键结论

1. **mix↑ 的汇率 = 10:1 赔率**：每换来 **T1 +0.0007**，要付 **T3 −0.0072**
   ⇒ 周末缺口 0.000146 需要 mix↑ **0.2 档**（理论）但实际每档代价 0.0072 T3
   ⇒ **单靠 mix↑ 永远补不上 T1 缺口**（且 T3 本来就不够）⇒ **变体 A 不能作为解法**。
2. **提 T3 的一等杠杆是 `lr_scale`（+0.045/档）和 `init_from`（rc2 比 base474 高 +0.027）**，
   而它们的 T1 代价很小（−0.002 / −0.0006）⇒ **T3 应该用 lr 和 init 去拿，不是用 mix**。
   ⇒ 骨干保护的汇率**优于 mix**（T1 +0.0011 / T3 仅 −0.0021）⇒ 是更好的 T1 杠杆。

### 8.11.4 据此新增 3 条**双目标组合臂**（计划 40 → **43 臂**）

之前 40 条全是**单变量**扰动 ⇒ **没有任何一条能同时朝两个目标走**。现补（全部基于最准靶心 trial_0008）：

| role | 组合 | 期望（按受控对效应量推算） |
|:--|:--|:--|
| `FOCUS_T0008_W6_SEED3` | mix↑(w6.0)，其余不变 | **补全 mix 的干净受控对**，标定真实效应量 |
| `FOCUS_T0008_W6_LR30` | mix↑(w6.0) + **lr↑(3.0)** | T1 ≈ +0.0007−0.0020、T3 ≈ +0.045−0.007 ⇒ **第一次真正针对「T1 过线 AND T3≥0.87」设计的组合** |
| `FOCUS_T0008_W6_PROT8` | mix↑(w6.0) + **骨干 0.1×+锁2ep** | 用**汇率更好的骨干保护**补 T1（对照组合 #1 的"用 lr 换 T3"） |

⇒ 时间预算：43 臂 = **4151 卡·min ≈ 69.2 卡·h** ⇒ 23 臂约 20:54 完 + 剩余 20 条约 8.0 h ⇒ **约 04:54 跑完**，距训练截止 06:30 仍留 ~1.6 h 给 TPE。
（**确定性臂是主力**，TPE 只剩 3~4 个 trial 的余量 —— 这是有意的取舍。）

---

## 八·十二、**全库受控对重扫推翻 §8.11.3 的结论** + 修掉 3 个真 bug + 计划重建（11-02 15:00~15:20）

### 8.12.0 触发

本轮 8 条新臂出来后，`trial_0005`（骨干保护 T15）与 `trial_0006`（骨干保护 T8）两条**直接复现**
把 §8.11.3 的"骨干保护是更好的 T1 杠杆"打成了问号 ⇒ 决定**用 study.db 的精确参数**重扫全库
（周末 114 + 本轮 8 = **122 条**，其中 **118 条有指标**）的受控对。

### 8.12.1 【修正 §8.11.3】汇率排序：**init > lr > mix > backbone**

方法：先按「只差一个参数」把 122 条分组（参数取自 **study.db 的原始 float**，不是 CSV 的格式化值），
得到 **10 对干净受控对**（mix 3 / lr 2 / init_from 2 / epochs 2 / backbone_lr_scale 1）
\+ **3 对同参数换种子**（用于标定种子噪声）。

| 维度 | 对数 | ΔT1（该维度变大时） | ΔT3 | **每换 1 单位 T3 要付的 T1** |
|:--|--:|--:|--:|--:|
| **init_from**（rc2→base474） | 2 | +0.000634 | −0.027364 | **0.023** ← 最优 |
| **lr_scale** | 2 | −0.001346/unit | +0.017009/unit | **0.079** |
| **mix_spec**（w6→w8） | 3 | **+0.001011**（3/3） | **−0.005860**（3/3） | **0.173** |
| **epochs** | 2 | +0.000253 | +0.005209 | 0.049 |
| **backbone_lr_scale**（1→0） | 1 | **−0.001102** | +0.002084 | **0.529** ← 最差 |

⇒ **三处修正**：
1. **骨干保护是最差的杠杆**（0.529），**不是** §8.11.3 说的"优于 mix" ⇒ §8.11.3 第 3 条**作废**。
   家族均值直接印证：`ep8_rc2_bb1/0` 的 T3=0.852511 vs `ep8_rc2_bb0.1/2` 的 T3=0.831052（**掉 0.0215**），
   换来 T1 +0.0019 ⇒ 赔率 0.088，比 lr 还差。
2. **mix 是弱杠杆**：家族回归 `T1~mix` 斜率仅 **+0.000123/unit（R²=0.021）**，`T3~mix` −0.003645/unit（R²=0.153）
   ⇒ **mix↓ 几乎免费地换 T3**，mix↑ 只能换到 +0.001 左右的 T1。
3. **lr 才是 T3 一等杠杆**：`family C (ep8/rc2/bb1/0, n=20)` 回归 **T3~lr = +0.017009/unit（R²=0.607）**、
   `T1~lr = −0.001346/unit（R²=0.457）**；`ep15 全家族 (n=25)` 是 **+0.019312（R²=0.644）/ −0.001866（R²=0.730）**。

**家族均值表（n≥3，节选）**：`ep8_rc2_bb1/0` n=20 T1 0.895159/T3 0.852511 FP49 ·
`ep15_base474_bb1/0` n=13 T1 0.892929/T3 **0.864937** FP49 · `ep8_rc2_bb0.1/2` n=3 T1 **0.897064**/T3 0.831052 FP69 ·
`ep8_rc2_bb1/2` n=5 T1 0.894209/T3 **0.859903** FP57 · `ep6_rc2_bb1/0` n=6 T1 0.895771/T3 0.848563 FP55。

**种子噪声实测**（同参数换种子 3 对）：|ΔT1| = 0.00128（trial_0025 家族）/ 0.00052（w2.0 家族）/ 0.00035（base474-natural 家族）；
|ΔT3| = 0.00599 / 0.00228 / 0.00007 ⇒ **σ(T1) ≈ 0.0005~0.0013，σ(T3) ≈ 0.0001~0.006**。

⇒ **最优联合点 = 「低 mix（w2.0）+ 高 lr（2.75~3.0）」**，**不是**原计划的 mix↑(w6.0)+lr↑（mix↑ 的 T1 收益
会被 T3 损失吃掉）。以 trial_0008（T1 0.895661/T3 0.867711，离两个目标只差 0.000236 / 0.002289）为原点推算：

| 配置 | 预测 T3 | 预测 T1 | 判定 |
|:--|--:|--:|:--|
| mix w2.0 + lr 3.0 | **0.8734** | 0.89556 | **T3 过线**，T1 差 0.0003（0.6σ 内） |
| mix w2.0 + lr 2.75 | 0.8691 | **0.89590** | T1 压线，T3 差 0.0009 |
| mix w2.0 + lr 2.5 | 0.8648 | **0.89618** | **严格 PASS 且 T3 明显高于 trial_0009 的 0.8579** |
| mix w3.0 + lr 3.0（纯 lr↑） | 0.8697 | 0.89550 | 干净的 lr 受控对 |

### 8.12.2 【第 11 个真 bug】8 条待注入臂**超出 focus 搜索空间 ⇒ 必崩**

**现象**：计划文件 43 条里有 20 条尚未入队，其中 **8 条**用了 `--focus-space` 白名单外的取值：

| 臂 | 越界项 |
|:--|:--|
| `FOCUS_SOUSTRICT3_W3_LR28821_8EP` / `FOCUS_SOUHIGH3NOBIN_W3_LR28821_8EP` / `FOCUS_SOUHIGH3NOBIN_W3_LR15_6EP` | `init_from` = soup_strict3 / soup_high3_nobn（白名单只有 rc2/base474）；第 3 条还带 bb 0.5 / freeze 1 |
| `FOCUS_T0056_SEED2` / `FOCUS_T0034_SEED2` / `FOCUS_T0008_W25` | `mix_spec` = **w2.5**（白名单 w2.0/w3.0/w6.0/w8.0/natural） |
| `FOCUS_T0021_SEED2` | `mix_spec` = **w4.0** |
| `FOCUS_T0008_WARMUP18` | `warmup_epochs` = **1.8**（上限 1.5） |

**根因（读 Optuna 4.9.0 源码确认，实测复现）**：
1. `optuna/trial/_trial.py::_is_fixed_param` → `distribution.to_internal_repr(value)`
   ⇒ 对 CategoricalDistribution，**值不在 choices 直接抛**
   `ValueError: 'w2.5' not in ('w2.0','w3.0','w6.0','w8.0','natural')`。
   ⇒ **`study.enqueue_trial()` 不校验**（入队时看不出来），**只有 trial 被 ask 的瞬间才炸** ⇒
   日志里表现为莫名其妙的 FAIL，极易漏掉（本轮就差点漏掉 8 条，含 3 条 soup 臂 —— 而重启调度器**唯一**的目的就是注入它们）。
2. `optuna/storages/_rdb/storage.py::_set_trial_param_without_commit` 里 `previous_distribution` 是按
   **(study_id, param_name)** 查 `TrialParamModel` 的**任意一条**记录 ⇒ **兼容性检查是 study 级、不是 trial 级**
   ⇒ 一旦本 study 记录过 `init_from=(rc2, base474)`，**之后任何用新 choices 的 trial 都会报**
   `CategoricalDistribution does not support dynamic value space`（实测）。
   ⇒ **所以「扩 FOCUS_CHOICES」这条路在本 study 上是死的。**
3. `optuna 4.9.0` **没有** `optuna.delete_trial`（实测 `hasattr(optuna,'delete_trial')` = **False**）
   ⇒ 删不掉历史 trial ⇒ 也换不了 study-name。

**修法（两条腿）**：
- **腿 A（改臂）**：5 条越界臂改写成白名单内**最近等价**（w2.5→**w2.0**、w4.0→w3.0、WARMUP18 直接砍）。
- **腿 B（新参数名绕开 study 级检查）**：soup init 走**新参数名 `init_from_x`**
  —— 该名在本 study 里**没有历史记录** ⇒ 不触发兼容性检查。
  改动落在 `manu/training/optuna_mix_ratio_search.py`：
  新增 `INIT_X_CHOICES = ["rc2","base474","soup_strict3","soup_high3_nobn"]`；
  `suggest_params()` 里**仅当 fixed_params 里显式带了 `init_from_x`** 才 suggest 它
  ⇒ 已入队的 11 条 WAITING 臂与 TPE 臂**逐位不变**；调度器取参改成
  `params.get("init_from_x") or params.get("init_from")`；入队白名单 `allowed` 加 `init_from_x`。
  ⇒ 计划里 soup 臂写成 `init_from: "rc2"`（白名单内）+ `init_from_x: "soup_strict3"`。

### 8.12.3 【第 12 个真 bug】调度器重启会**丢掉在跑臂的收割**

`optuna_mix_ratio_search.py` 收到 SIGTERM 后走「收尾：等在跑的 trial 结束（或杀掉）」分支，
那段代码**只 `process.wait()`，不收割**（不 `study.tell()`、不写 summary CSV）⇒
重启时若还有臂在跑：**trial 永久卡在 RUNNING**（`backfill` 也补不回来，因为没有 user_attrs）
**+ 它的 GPU 会被新调度器重新分配 ⇒ GPU 争抢/OOM**。

**修法**（不动正在跑的调度器，改 watcher）：`restart_dispatcher_inject_arms.py` 增加
「**重启前等非 plan 臂收尾**」的保险 —— 原有等待循环已覆盖「在跑的 **plan 臂**」，
但 **TPE 臂**不在集合里，而调度器在最后一个 plan 臂跑完的瞬间可能刚好派出一个 TPE 臂 ⇒ 竞态真实存在。

### 8.12.4 【第 13 个真 bug】4 条臂**去重失效 ⇒ 重启后会被重复入队**

去重指纹 `_cfg_sig()` 把浮点按 `round(v, 6)` 归一。`T0077_PROT8 / T0077_SEED2 / T0082_PROT8 / T0082_SEED2`
这 4 条**入队时用的是从 CSV 抄来的截断值**（lr 2.7676 / clip 4.05 / warmup 1.448 / wd 3.132e-05），
之后计划文件被 `build_focus_arms_from_study.py` 修成了 study.db 的精确值
（lr 2.7675790243131346 / clip 4.05428232735804 …）⇒ `round(...,6)` 后指纹不同 ⇒
**重启后这 4 条会以"新臂"身份再入队一次 ⇒ 白烧 4 个 GPU 槽（约 5.5 卡·h）**。
**修法**：新增 `align_plan_with_study.py` —— **以 study 为准**（study 里那份才是真正会跑的）把计划文件对齐回去。

### 8.12.5 计划重建：43 → **41 臂**（砍 7 / 改写 5 / 新增 5 / **按价值重排**）

- **入队顺序 = 执行顺序**（`study.ask()` 按 trial_id FIFO）⇒ **高价值臂必须排前面**，否则超时被砍的是好臂。
- **砍 7 条**：`SOUHIGH3NOBIN_W3_LR15_6EP`（冗余）· `T0021_SEED2`（w4.0 越界且改后与 trial_0064 家族重合）·
  `T0008_BB03`（bb0.30 家族 T3 0.849974 < bb1/0 的 0.852511）· `T0008_WARMUP18`（warmup 1.8 越界）·
  `T0008_W6_SEED3`（mix 标定已有 3 对受控对）· **`T0008_W6_LR30`**（模型判定 mix↑ 的 −0.011 T3 吃掉 lr↑ 的 +0.002 ⇒ 预测 T3≈0.859 < trial_0008 的 0.8677）·
  **`T0008_W6_PROT8`**（mix↑ 与骨干保护两个负杠杆叠加）。
- **改写 5 条**：3 条 w2.5→w2.0（`T0056_SEED2`/`T0034_SEED2`/`T0008_W25`）+ 2 条 soup 臂加 `init_from_x`。
  ★ `T0008_W25` 改后价值反而更高：**相对 trial_0008 只动 mix 一个维度 ⇒ 干净的 mix↓ 受控对**。
- **新增 5 条**（模型驱动）：`T0008_W20_LR30`(seed248) · `T0008_W20_LR275`(seed247) ·
  `T0008_LR30`(seed249) · `T0008_W20_LR25`(seed250) · `T0008_W20_SEED2`(seed251)
  ⇒ 与 trial_0008 构成 **w3.0/w2.0 × lr 2.75/2.8821/3.0 的受控网格**。
- **算力预算**：待跑 = 11 条已入队 WAITING + 18 条新臂 + 4 条在跑 ≈ **3025 卡·min ÷ 4 卡 ≈ 12.6 h**
  ⇒ 15:20 起算 **约 03:55 收工**，距训练截止 06:30 留 ~2.5 h。

### 8.12.6 本轮新增 3 个永久工具（都带实测证据）

| 工具 | 作用 | 命令（服务器 conda `uav`，`cd /tmp/pycharm_project_10ae9e2e`，`export PYTHONPATH=.`） |
|:--|:--|:--|
| `manu/training/check_plan_vs_search_space.py` | 逐条臂体检：搜索空间 + 与已入队臂**逐位一致** + 调度器 `allowed` 过滤 | `python3 manu/training/check_plan_vs_search_space.py --plan manu/training/nogmc_focus_plan.json --out-root /mnt/data/siping/runs/nogmc_focus --study-name mix_ratio_search_nogmc_focus` |
| `manu/training/dryrun_plan_enqueue.py` | 在 **study 副本**上完整复刻「入队 + ask + suggest_params」，**重启前**证明不会崩 | `python3 manu/training/dryrun_plan_enqueue.py --plan manu/training/nogmc_focus_plan.json --out-root /mnt/data/siping/runs/nogmc_focus --study-name mix_ratio_search_nogmc_focus` |
| `manu/training/align_plan_with_study.py` | 以 study 为准把计划文件的 params 对齐回去（修去重失效） | `python3 manu/training/align_plan_with_study.py --plan manu/training/nogmc_focus_plan.json --out-root /mnt/data/siping/runs/nogmc_focus --study-name mix_ratio_search_nogmc_focus --apply` |
| `manu/training/rebuild_focus_plan.py` | 计划重建脚本（砍/改/增/重排的可复现来源） | `python3 manu/training/rebuild_focus_plan.py` |

**体检与干跑实测输出**：空间体检 **41/41 ✅**；干跑 **29 条（11 已入队 WAITING + 18 新臂）全部 ask+suggest 成功、0 失败**，
其中 `t23 init=soup_strict3`、`t28 init=soup_high3_nobn` ⇒ **`init_from_x` 新路径走通**；
23 条已入队臂被正确跳过（精度对齐后无重复）。真 study 未被触碰（干跑全程在副本上）。

---

## 八·十三、**交接口径判定器的两处口径修正**（11-02 15:20~15:50，周二交付链的关键补丁）

> 这两处不是小修：**不修的话周二 09:00 交出来的判定必然是错的**（候选池漏掉周末 33 条严格 PASS、
> 且"多种子中位"永远算不到 n≥2 ⇒ 交接口径形同虚设）。

### 8.13.1 判定器的**决策单元**必须是「配置」，不是「单条 trial」

**旧逻辑（错）**：只把 `gate_verdict == "PASS"` 的 trial 放进配置组，其余进诊断附录。
**为什么错**：交接口径白纸黑字是「**同超参多种子中位**」⇒ 决策单元是**配置**；
而单条 trial 的 `PASS_RELAXED` 只说明"**这条种子**的 T1 没过红线"，**不代表这个配置的中位没过**。
⇒ 旧逻辑的后果：一个配置 3 条种子只有 1 条 PASS 时组里只有 1 条 ⇒ **n=1 ⇒ 永远凑不够 min-seeds**
⇒ 恰好把"**中位过线**"的配置误判成不合格，**多种子口径等于白做**。

**新逻辑**：
1. **全部种子**（三轨指标齐全）按 15 维配方逐位归组；
2. 用**组的中位**过硬门禁：**T1 中位 ≥ 0.8958967482400269** 且 **T2 中位 F1 ≥ 0.985**
   且 **T2 中位虚警 ≤ 112** ⇒ 该组合格；
3. 单条 trial 的 verdict **只作诊断信息**（附录 A 仍逐条列出）。

**口径铁律怎么守住**（「放宽通道不得进正式交接」）：**若组内一条严格 PASS 的种子都没有**，
则列为**阻塞项**（判定器 rc=3 ⇒ 只能作「候选」不能作「交接线」，交人工裁定）；
固化器**优先封组内的严格 PASS 种子**，一条都没有时才退到中位种子，
并在登记 JSON 里显式写 `sealed_trial_individual_verdict` / `sealed_trial_is_strict_pass`。

### 8.13.2 判定池必须**含周末**，且 trial 名要加来源前缀

**为什么必须合并**：同配方的种子**横跨两个 study** ——
周末 `trial_0009`（严格 PASS 里 T3 最高 0.857880）与本轮 `FOCUS_T0009_SEED2`
是**同一个配方**（w3.0·lr1.0·15ep·rc2·bb1）⇒ **只有合并才能凑够 n≥2**。
只读本轮 CSV ⇒ ① 候选池少了周末的 33 条严格 PASS；② 同配方种子被拆成两组 ⇒ n 永远 ≤1。

**工具改动**（`judge_handover_candidate.py` / `seal_candidate.py` / `finalize_nogmc_focus.sh` / `resume_nogmc_focus.sh`）：
- `--csv` 支持**多份**，格式 `TAG=路径`；`--trials-root` 同样多份，按 TAG 解析。
- ⚠️ **trial 名统一加 `TAG:` 前缀**：两个 study **都有 trial_0009 但不是同一条**
  （周末 = w3.0/lr1.0/15ep；本轮 = w6.0/lr2.0/10ep），权重也各在各自目录
  ⇒ 不加前缀 ⇒ 固化器会**取错权重**。
- ⚠️ **argparse 坑**：`nargs="+"` 的选项**重复写会覆盖而不是追加**
  ⇒ 只能写**一次** `--csv A B`，不能写 `--csv A --csv B`。
  > 🔴 **2026-11-03 更正（本条只对 `judge_handover_candidate.py` / `seal_candidate.py` 成立）**：
  > `analyze_t3_reachability.py` 的 `--csv` 是 **`action="append"`**（每个 CSV 一个 `--csv`）⇒ **用它时不能写 `--csv A B`**，
  > 否则 argparse 报 `unrecognized arguments`（本轮实测 finalize ⑤b rc=2）⇒ 正确写法见 **§11.3** 的复现命令；根因见 **§11.5 bug ②**。

### 8.13.3 实测证据（服务器，conda `uav`，`cd /tmp/pycharm_project_10ae9e2e`，`export PYTHONPATH=.`）

```bash
# 回归：只读周末（应仍复现 trial_0025 为候选、33 组严格 PASS）
python3 manu/training/judge_handover_candidate.py \
  --csv "weekend=/mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv" \
  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --out-md /tmp/_J1.md --out-json /tmp/_J1.json --min-seeds 2

# 合并判定（周末 + 本轮）
python3 manu/training/judge_handover_candidate.py \
  --csv "weekend=.../nogmc_weekend/optuna_summary.csv" "focus=.../nogmc_focus/optuna_summary.csv" \
  --ref .../nogmc_ref_weekend/reference_frame.json \
  --out-md .../nogmc_focus/HANDOVER.md --out-json .../nogmc_focus/HANDOVER.json --min-seeds 2

# 固化干跑（跨 study 解析权重）
python3 manu/training/seal_candidate.py --handover-json .../HANDOVER.json \
  --csv "weekend=...weekend.csv" "focus=...focus.csv" \
  --trials-root "focus=.../nogmc_focus/trials" "weekend=.../nogmc_weekend/trials" \
  --out-dir .../CANDIDATES --rc-id Candidate_v2 --dry-run
```

| 测试 | 结果 |
|:--|:--|
| 周末单跑（回归） | ✅ 仍复现 `trial_0025` 为候选（33 组合格）；trial 名带 `weekend:` 前缀 |
| 合并判定（122 行） | ✅ **34 组合格 / 53 组不合格**；**trial_0025 组 n=3**（focus:trial_0000 + focus:trial_0001 + weekend:trial_0025）⇒ **T1 中位 0.895692 < 红线 0.895897** ⇒ 不合格（**与人工分析完全一致**）；**trial_0087+0113 组 n=2 正确合并**（T3 中位 0.872844） |
| 合并判定 winner | `weekend:trial_0009`（T1 0.896171 +0.0274pp · T2 FP 64 · **T3 0.857880** · U 0.090682），n=1 ⇒ `handover_ready=false`（等 `FOCUS_T0009_SEED2` 补种子） |
| 固化干跑 | ✅ 正确解析到**周末**目录的 `epoch15.pt`（没串到本轮同号 trial）；单条 verdict=PASS |

⇒ **周二 09:00 的交付判定，现在会自动给出与人工分析一致的结论**：
Candidate_v1（trial_0025 配方）**因 3 种子中位 T1 不过线被排除**（理由精确，不再是"只有 1 个种子"），
新的第一候选是 **trial_0009 配方**（T3 最高的严格 PASS）。

---

## 八·十四、**种子噪声实测 0.0025 ≫ T1 缺口 0.0002** ⇒ 策略从「加配置」改为「补种子」（11-02 15:35~15:50）

### 8.14.1 【第 14 类 bug】受控对必须判「差异是否**实质**」

2026-11-02 15:35 修 `analyze_t3_reachability.py` 时发现：两轮 study 里存在**数值几乎相同但源码不同**的维度 ——
周末 `trial_0025` 的 `weight_decay = 0.00026412`，本轮 focus 臂是 `0.00026410`（**差 2e-8**，
源于"抄 CSV"与"抄 study.db"的差别）。若只看"是否只差一个维度"，这一对会被当成
**weight_decay 的受控对**，而它俩的 ΔT1/ΔT3 其实**全是种子噪声**（换了个 seed 而已）
⇒ 实测算出来 **ΔT1/单位 = +123200** 这种荒谬值，差点写进周二交付件。

⇒ **修法**：按维度设「**实质差异门限**」（`MATERIAL_MIN`：lr ≥0.05 · epochs ≥1 · warmup ≥0.05 ·
weight_decay ≥1e-4 · focal/offset/mosaic/scale/translate ≥0.05 · clip ≥0.1 · backbone ≥0.1 ·
freeze ≥1 · mix 比例 ≥0.25）；达不到门限的对**单独归为「种子噪声对」**。
⇒ 同时把受控对改成**用 study.db 的原始 float**（新增 `--study TAG=db:study_name` 参数）——
CSV 的格式化精度会**凭空造出/抹掉**受控对（实测：按 CSV 配对会得到一条假的 weight_decay 对，
同时漏掉真正的 lr / epochs / init_from 对）。

### 8.14.2 【关键硬数字】种子噪声 = 0.0025，而最近点的 T1 缺口只有 0.00015~0.00123

修完后 §4.2 给出**种子噪声的直接实测**（同配方换种子）：

| 量 | 中位 | 最大 |
|:--|--:|--:|
| **\|ΔT1\|** | **0.002464** | 0.003104 |
| \|ΔT3\| | 0.002995 | 0.005394 |

对照**离双目标最近的三个配置**：

| 配置 | T3 | T1 缺口 | 缺口 / σ(T1) |
|:--|--:|--:|--:|
| `trial_0082` 配方 | 0.865458 | **0.000146** | **0.06 σ** |
| `trial_0008` 配方 | 0.867711 | 0.000236 | 0.10 σ |
| `trial_0077` 配方 | **0.871475** | 0.001234 | 0.50 σ |

⇒ **T1 缺口比种子噪声小 1~2 个数量级** ⇒ **"过不过 T1 红线"基本由种子运气决定，不是配置本身决定**
⇒ **交接口径（同超参多种子中位）是唯一能给出可信结论的办法**，而 n=2 的中位仍然很脆。

⇒ **⇒ 策略修正：这一轮剩下的算力应该「补种子」而不是「加新配置」。**
（加新配置只能在 Pareto 前沿上再挪一点，而挪动的幅度还不到噪声的零头；补种子能把运气变成可统计的中位。）

### 8.14.3 计划 41 → **45 臂**：补 4 条种子臂，插到未入队队列**最前**

| 新增臂 | seed | 补的是谁 | 为什么 |
|:--|--:|:--|:--|
| `FOCUS_T0008_W20_LR30_SEED2` | 252 | 模型靶点 `w2.0+lr3.0` 的**第 2 个种子** | 它若跑出好结果但 n=1 ⇒ 交接口径同样不成立 ⇒ **必需配套** |
| `FOCUS_T0008_SEED3` | 253 | `trial_0008` 配方（缺口 0.10 σ）的第 3 种子 | n=2 中位仍脆 ⇒ 补到 n=3 |
| `FOCUS_T0077_SEED3` | 254 | `trial_0077` 配方（**唯一 T3 过 0.87**，缺口 0.50 σ）的第 3 种子 | 它的 T1 是**种子运气问题** ⇒ 必须 n=3 才敢下结论 |
| `FOCUS_T0082_SEED3` | 255 | `trial_0082` 配方（**缺口最小 0.06 σ**）的第 3 种子 | 一旦某种子过线，就是最稳的交接线 |

**入队顺序 = 执行顺序** ⇒ 4 条补种子臂插在未入队队列**最前**（紧跟模型靶点），
保证它们一定跑到；被挤到后面的都是低价值的探针臂。

**实测**：空间体检 **45/45 ✅**（已入队 23 条逐位一致）；干跑 **33 条（11 WAITING + 22 新臂）全部通过、0 失败**。
算力：待跑 11 + 22 = **33 条 ≈ 3200 卡·min ÷ 4 卡 ≈ 13.3 h** ⇒ 15:20 起算 **约 04:40 收工**，距 06:30 留 ~1.8 h。

### 8.14.4 `analyze_t3_reachability.py` 同步修正（周二交付件）

- 新增 `--study TAG=db:study_name` ⇒ 受控对改用**精确参数**；
- 方向约定统一为「**该维度变大时**的 Δ（按单位归一）」，`mix_spec` 按**实际比例**排序（wN→N，natural→3.09），
  纯类别维度（init_from / freeze_bn）**不做方向聚合**、逐对列出；
- 新增「**每换 1 单位 T3 要付的 T1**」列（越小越划算）—— 这是**唯一的杠杆排序依据**；
- 新增 **§4.2 种子噪声量**表（§4.1 的受控对只收「实质差异」的对）。

**修完后的杠杆排序（每换 1 单位 T3 付的 T1）**：
`lr_scale 0.047` < `epochs 0.088` <（`backbone_lr_scale` 与 `mix_spec` 是**反向**的：它们变大时 T3 掉）
；`init_from` 是类别维度：**rc2 → base474 换来 T1 +0.0003~0.0010，代价 T3 −0.016~−0.039**。

### 8.14.5 顺带修掉：**watcher 单实例锁**（两个 watcher 会互相重启调度器）

2026-11-02 15:31 重启 watcher 时**旧实例没被杀掉** ⇒ 两个 watcher 同时在跑 ⇒
两者都会在「已入队臂全终态」时触发重启 ⇒ 一个 SIGTERM 掉调度器、另一个等看门狗拉起新的、
第一个**又把新的 SIGTERM 掉** ⇒ 调度器被反复重启，**新增臂可能入队不全/被跳过**。
⇒ `restart_dispatcher_inject_arms.py` 加 **PID 单实例锁**（`<out-root>/restart_inject.pid`）：
已有活实例就以 rc=3 拒绝启动；锁文件指向已死进程时自动清理。
⚠️ 顺带提醒：`--dry-run` 的检查在**等待循环之后** ⇒ 用它做"锁测试"会**真的进入等待循环变成第二个 watcher**
（实测踩到，已停掉）。要测锁请直接跑、不加 `--dry-run`。
实测：单实例 PID 752127、锁文件一致、第二个实例被拒（`[LOCK] 已存在运行中的实例 ⇒ 拒绝重复启动`，rc=3）。

### 8.14.6 【新路线】沿 **epoch 轴**的 SWA（零训练成本，11-02 16:35 起）

**为什么变体 C 判负、但 SWA 仍值得试**：变体 C 的 Model Soups 是**跨配置**平均
（trial_0009 + trial_0025 + trial_0018），而这三者的 T1/T3 画像差别很大
⇒ 平均结果只是**在两端之间插值**（strict3: T1 0.8968/T3 0.8475；high3_nobn: T1 0.8938/T3 0.8747）。
**SWA 沿 epoch 轴平均是另一回事**：同一 run 的相邻 epoch 在**同一个损失盆地**里
⇒ 平均 = 降方差、**不改变 T1/T3 的权衡位置**，通常还小幅涨 F1。
⇒ **零训练算力**，而当前死结恰好是"最近几个配置 T1 只差 0.00015~0.00123" ⇒ 若 SWA 抬 T1 0.001~0.003
且不掉 T3，就可能**直接命中「T3 ≥ 0.87 且 T1 过线」**。

**已产出的 6 个 SWA 权重**（`/mnt/data/siping/runs/nogmc_focus/swa/`，各 7,384,006 B）：
`swa_0077_last3`（T3 0.871475，唯一过 0.87 的配置）· `swa_0008_last3`（T1 缺口 0.000236）·
`swa_0082_last3`（T1 缺口 0.000146，最小）· `swa_0113_last3`（T3 0.871705）·
`swa_0087_last3`（T3 0.873983，最高）· `swa_0009_last3`（T1 余量 +0.0274pp）。
（8ep 的取末 3 个 epoch；trial_0009 取 13/14/15。）

**格式自检（CPU，已做）**：SWA 与源 ckpt **523 张量完全一致、无缺失无多余**；
**25 个 p0 张量在 epoch6 与 epoch8 之间逐位相同 ⇒ 融合正确保留了 p0**；
变化最大的只是 BN `running_var`（相对变化 1.3%）⇒ **平均后的 BN 统计量与平均权重自洽**
（所以用「BN 统计量也平均」的普通版，不用 `--nobn` 版）。

**门禁脚本**：`manu/training/gate_swa_variants.sh`（PID 763148，已挂后台待命）
⇒ **等训练彻底结束才开跑**（判据：已过 `TRAIN_STOP=06:30` **且** 调度器进程已退出）
⇒ **铁律：评估不得与训练同卡**（实测 4 卡全被训练占满、每卡只剩 8.4 GB，
`pick_free_gpus()` 会回退到训练卡上 ⇒ 现在跑门禁有 OOM 风险 ⇒ 绝不能提前跑）。
门禁**串行**（一次 1 个），按"最可能命中"排序（0077 → 0008 → 0082 → 0113 → 0087 → 0009），
预计 06:30 起、每个约 10 min ⇒ 约 07:30 出全部结果。

**下一手（本轮 focus 种子跑完后）**：用**同配方不同种子**的权重做**跨种子 soup**
（经典 model soup，文献常报 +0.5~1.0 pp F1）⇒ 追加进 SWA 门禁队列。

### 8.14.7 SWA 结果**接进交付链**（否则会漏交一条可能命中的产物）

⚠️ **SWA 产物不在 study 里** ⇒ `judge_handover_candidate.py`（读 CSV）和
`seal_candidate.py`（从 CSV 挑中位 trial）**都看不到它** ⇒ 不单独整合的话，
周二 09:00 交出来的报告会**漏掉一条可能已经命中双目标的产物**。

⇒ 新增 `manu/training/incorporate_swa_results.py`：
扫 `<out-root>/swa/gate/*/epoch*/gate_report.json` ⇒ 取 T1 / T2(F1,FP) / T3
（门禁报告字段：`tracks.track1_uav.finetune_peak.f1`、`tracks.track2_golden.f1/fp`、
`tracks.track3_blind.f1`）⇒ 按**同一套硬门禁**（T1 ≥ 红线 · T2 F1 ≥ 0.985 · T2 虚警 ≤ 112）判定
⇒ 追加 **附录 C** 到 `HANDOVER.md` + 写 `SWA_SUMMARY.json`
⇒ **若某候选既过硬门禁又是 T3 ≥ 0.87 ⇒ 自动封存**（权重 + md5/sha256 + 登记 JSON，含"这是权重后处理产物、
多种子证据来自底层配方"的说明）并打出「★ 命中」。

⇒ `finalize_nogmc_focus.sh` 新增 **④b**：先**等 SWA 门禁收尾**（等 `gate_swa.log` 出现
「SWA 变体门禁全部完成」，最多等 `SWA_WAIT_MIN`=45 分钟）⇒ 再整合
⇒ 若 `SWA_SUMMARY.json` 的 `n_hit > 0` ⇒ **⑤ 步不再封 trial 侧候选**（避免两个 RC 造成歧义）。

**实测**（用假门禁报告做的端到端验证）：命中判定 ✅（T1 0.897012 / T2 0.998619(FP42) / T3 0.873457 ⇒ ★命中）、
未过硬门禁的候选被正确标出 ✅、附录 C 表格渲染 ✅、`SWA_SUMMARY.json` 字段正确 ✅、
**权重缺失时安全拒绝封存（rc=1）** ✅。

### 8.14.8 【下一手】跨种子 soup（同配方·不同种子·末 epoch 平均）+ 又抓一个坑

**跨种子 soup** 与变体 C 的差别：变体 C 是**跨配置**平均（trial_0009 + trial_0025 + trial_0018
的 T1/T3 画像差很多 ⇒ 只是两端插值 ⇒ 判负）；**跨种子**是**同一配方、同一初始化、只换随机种子**
⇒ 三个模型在同一盆地里的不同位置 ⇒ 平均 = 降方差，**通常同时提升 T1 与 T3**
（文献常报 +0.5~1.0 pp F1）。⇒ 正打在当前死结上（T1 缺口 0.00015~0.00123 vs 种子噪声 0.0025）。

**工具**：`build_crossseed_soups.py`（按 role 从本轮 study + 周末 study 取同配方的**末 epoch** 权重做平均，
写 `swa/cs_t0008.pt` / `cs_t0077.pt` / `cs_t0082.pt` + `crossseed_manifest.json`）
+ `gate_crossseed_soups.sh`（PID 779515，已挂后台待命：等训练结束 ⇒ 等 SWA 门禁跑完 ⇒ 构建 ⇒ 串行门禁，
只门禁 **manifest 里列出**的产物 ⇒ 不会把遗留半成品当候选）。

⚠️ **【第 15 类坑】正在训练的 trial，其 `weights/` 里已经有 epoch1..epochN 的半成品权重**。
实测：`t0015`（FOCUS_T0008_REPLAY）在**训练中**，目录里已有 `epoch1/2/3.pt`
⇒ 若构建器取"最后一个 epoch"就会把**半训练模型**混进 soup ⇒ 门禁量的是一个**从未存在过的模型** ⇒ 结论无效。
⇒ **修法**：构建器**只用 `state == COMPLETE` 的 trial**（半成品直接跳过并打印原因）；
已生成的那个半成品 soup 已改名移出 `cs_*.pt` 命名（`stale_cs_t0008_BAD_halfmodel.pt`），
并让门禁只认 manifest 里的产物。

---

## 九、判据（与周末一致，严格口径）

- **交接口径** = **严格 `PASS`**（"严格通道为上位门槛；放宽通道仅作诊断标记"）**的配置级多种子中位**
- 噪声带用**「极差/2/√n」**修正后的不确定度，不用裸极差
- U 不可分 ⇒ 转看 **T3 中位**；T3 仍不可分 ⇒ 取种子数多者
- **T1 红线 0.8958967482400269 / T2 FP ≤ 112**（= 基线 52 + 增量 60）

---

## 十、**关机窗口排期加固**（2026-11-02 18:00~19:00，为 11-03 06:30 主机定时关机）

> **触发**：用户宣布 **2026-11-03 06:30 主机定时关机**，而本轮交付窗口是 **11-03 09:00** ⇒
> **会话在 06:30~09:00 离线**，收尾链（训练截止 → 门禁 → 判定 → 固化 → 备份）**不能靠在线点命令**。
> ⇒ 本节记录：① 关机到底影响什么（实测）② 排期用的实测耗时基线 ③ 本轮为这次关机改了 4 处代码
> ④ 新时间线 ⑤ 可复制的复现命令。

### 10.1 关机影响面判定：关机的是**本机**，训练在**另一台机器**上

**实测两台机器的身份（执行机：本地 + 服务器 `192.168.99.40:32222`）**

| 项 | 训练服务器 | 本地机 |
|:--|:--|:--|
| 地址 | `192.168.99.40:32222` | `192.168.0.115` |
| hostname | `test-Super-Server` | `manu-System-Product-Name` |
| machine-id | `9eb85e0364724034ba6a4851711c6dd7` | `ce2ff254ee6e4890bdf8f355d48f1071` |
| GPU | **4 × RTX 4090D** | **无 GPU**（`nvidia-smi` 命令不存在） |

⇒ **machine-id 不同 + 服务器上没有任何挂载本机磁盘的迹象**（服务器 `mount` 里只有本地 fuse/gvfs 与 nfsd，
**没有指向本机的 nfs/sshfs 挂载**）⇒ **两台机器相互独立，本机关机不会影响服务器上的进程**。

**查得到定时关机吗？查不到。**
两台机器的 `systemctl list-timers --all` 里**没有任何定时关机单元**；`shutdown.target` / `poweroff.target`
都是 `static` 或 `disabled`；`atq` **两机都未安装**（`command not found`）。
⇒ **06:30 关机是用户侧行为/策略，机器上查不到痕迹** ⇒ 只能按「已知会关机」做防御性排期。

**时钟**：两机此刻**完全一致**（实测 `2026-11-02 18:19:02 CST` = `2026-11-02 18:19:02 CST`）；
此前（本轮早些时候）记录过 **52~56 分钟**偏差 ⇒ **偏差是"发生过"而非"恒定"**，
⇒ **纪律不变：日志时间戳一律按服务器读**（服务器那侧才是产出日志的机器）。

**结论（三条）**

1. ✅ **本机关机不会杀掉服务器上的训练/门禁** —— 调度器、两条门禁线、autorun 全部是 `setsid nohup` 启动，
   **已脱离 SSH 会话** ⇒ 本机关掉 SSH / 断网 / 关机都不影响它们继续跑。
2. 🔴 **但主 Agent 会离线到 09:00** ⇒ **收尾链必须服务器端自主执行**（这是本节全部改动的出发点）。
3. 🔴 **本机不是产出机** —— 交付物（`HANDOVER.md` / `study.db` / 权重）都在服务器 `/mnt/data/siping/runs/nogmc_focus/`
   ⇒ 关机前只要**服务器端把交付件落盘**，本机关机就是安全的。

> ✅ **2026-11-02 18:23 用户明确确认**（与上述实测一致）：
> **06:30 只关本地机，服务器（`test-Super-Server`）照常跑到用户 09:00 到岗**。
> ⇒ 本节 §10.3 的 4 处代码改动**全部保留**（自主收尾编排仍然是正解：它让「会话离线」不再是风险），
> 但**墙钟截止可以放宽** —— 由 `stop_at=04:00 / hard_cut=04:05` 放宽到 **`stop_at=hard_cut=06:20`**（+2h20m），
> 换来多跑 **2 条臂**（见 §10.4 / §10.4.1）。
> ⇒ ⚠️ **教训**：**"哪台机器会关机"必须先跟用户确认，不能靠机器上的痕迹反推** ——
> 机器上查不到定时关机单元是常态（本轮两机 `atq` 都没装），**查不到 ≠ 不会发生**，
> 也**不等于**"两台都会关机"。

**复现命令（执行机：本地 / 服务器各跑一遍）**

```bash
# 本地
hostname; cat /etc/machine-id; ip -4 addr show | grep -oP 'inet \K[\d.]+' ; nvidia-smi -L 2>&1 | head -2
systemctl list-timers --all --no-pager | grep -iE "shutdown|poweroff|halt" ; atq 2>&1 | head -3
date "+%F %T %Z"

# 服务器
ssh -p 32222 huangzhe@192.168.99.40 'hostname; cat /etc/machine-id; nvidia-smi --query-gpu=count --format=csv,noheader; mount | grep -Ei "nfs|sshfs"; systemctl list-timers --all --no-pager | grep -iE "shutdown|poweroff|halt"; atq 2>&1 | head -3; date "+%F %T %Z"'
```

### 10.2 排期用的**实测耗时基线**（决定了"训练什么时候必须停"）

**① 单次三轨门禁耗时**（`--gpus 4 --cache-gpus 0,1,2,3` 模式）
⇒ 取自 **2026-11-02 变体 C 的四个 soup 门禁**，`gate_report.json` 的 `elapsed_seconds` 字段：

| 候选 | 耗时 | 判定 |
|:--|--:|:--|
| `soup_strict3` | **588.7 s** | PASS |
| `soup_strict3_nobn` | **590.6 s** | PASS |
| `soup_high3` | **592.8 s** | FAIL |
| `soup_high3_nobn` | **590.9 s** | FAIL |

⇒ **≈ 10 min / 个**。产物路径：`/mnt/data/siping/runs/nogmc_focus/soups/<name>/gate/gate_report.json`。

**② worker 内嵌门禁**（`--gpus 1` 模式）：**1170.3 s ≈ 19.5 min**（`trial_0012`，日志
`/mnt/data/siping/runs/nogmc_focus/logs/trial_0012.log` 的「耗时 1170.3s」）。
⇒ **总吞吐对比**：4 卡模式（一个门禁吃 4 卡）串行 = **6 个/小时**（60÷10）；
1 卡模式 4 并行 = 4÷19.5 min ≈ **12 个/小时** ⇒ **1 卡并行在总吞吐上更优**。
⚠️ **但本轮不切换** —— 4 卡串行是**已验证**的路径（4 条 soup 门禁 + worker 门禁都跑过），
临时改并发模式等于在交付前夜换未验证路径；1 卡并行的结论**留作下一轮的优化项**。

**③ 每臂训练耗时**（周末 study 的 **80 条 `COMPLETE`**，`datetime_start → datetime_complete` 的**中位数**）：

| epochs | 3 | 6 | 8 | 10 | 15 |
|:--|--:|--:|--:|--:|--:|
| **中位数（min）** | **46.9** | **73.8** | **92.2** | **110.3** | **154.8** |
| n | 9 | 10 | 46 | 4 | 11 |

⇒ 每档 min/max 极差 **1.5~2.4 min** ⇒ 耗时**极稳**，可安全用作排期估计
（调度器里就固化成了 `EPOCH_MIN_EST = {3:47, 6:74, 8:92, 10:110, 15:155}`）。
数据源：`/mnt/data/siping/runs/nogmc_weekend/study.db`（`trials.datetime_start/datetime_complete`
+ `trial_params` 里 `epochs` 的 categorical 索引 → `[3,6,8,10,15]`）。

**⇒ 排期结论**：门禁 **9 个候选**（6 SWA + 3 跨种子）× 10 min ≈ **1.5 h**（实测门禁窗口 ~96 min，含跨种子 soup 构建 6 min）
⇒ **训练截止必须比门禁开始时刻提前 ≥ 2 h**（留出收尾 + 判定 + 固化 + 备份）。
⚠️ **注意基准是「门禁开始时刻」，不是「关机时刻」** —— 服务器照常运行到 09:00 ⇒
**本轮定为 `stop_at = hard_cut = 2026-11-03 06:20`**（门禁 ~07:13 跑完、finalize ~07:35 完成、09:00 前就位）。
（早先按「服务器也关机」误判定的 04:00/04:05 已作废，见 §10.4 的用户更正。）

**复现命令（执行机：服务器，conda env `uav`，`export PYTHONPATH=.`，`cd /tmp/pycharm_project_10ae9e2e`）**

```bash
# ① 四个 soup 门禁耗时
python3 -c "
import json,glob
for f in sorted(glob.glob('/mnt/data/siping/runs/nogmc_focus/soups/*/gate/gate_report.json')):
    d=json.load(open(f)); print(d['weights'].split('/')[-1], d['verdict'], d['elapsed_seconds'])
"
# ② worker 内嵌门禁耗时
grep -h "耗时" /mnt/data/siping/runs/nogmc_focus/logs/trial_0012.log | tail -2
# ③ 每臂耗时中位数（从周末 study.db 现算）
python3 -c "
import sqlite3, statistics, datetime
db=sqlite3.connect('/mnt/data/siping/runs/nogmc_weekend/study.db'); c=db.cursor()
q='''select tr.datetime_start, tr.datetime_complete,
 (select param_value from trial_params p where p.trial_id=tr.trial_id and p.param_name=\"epochs\")
 from trials tr where tr.state=\"COMPLETE\" and tr.datetime_start is not null and tr.datetime_complete is not null'''
P=lambda s: datetime.datetime.fromisoformat(s); b={}
for st,cp,ep in c.execute(q).fetchall():
    b.setdefault(int(ep),[]).append((P(cp)-P(st)).total_seconds()/60)
for ep in sorted(b):
    v=b[ep]; print('ep_idx%d n=%d median=%.1f min=%.1f max=%.1f'%(ep,len(v),statistics.median(v),min(v),max(v)))
"
```

### 10.3 本轮为这次关机改的 4 处代码

#### 10.3.1 `manu/training/optuna_mix_ratio_search.py` —— **bug #12 修掉 + 墙钟双截止**

**问题①（bug #12，最致命）**：收割逻辑**只写在主循环里** ⇒ 调度器一旦 `break`
（收到 SIGTERM / 到达截止时间）就**永远不收割在跑的臂** ⇒

- trial 永远卡在 `RUNNING`；
- `summary CSV` 缺行；
- **`backfill` 也补不回来** —— 因为没有 `user_attrs`（backfill 靠 `user_attrs` 认领）；
- ⇒ **一条 92~155 min 的训练白跑**（权重还在，但判定器看不到它）。

**修法**：把收割逻辑抽成**嵌套函数 `harvest(workers)`**（返回 `(仍在跑的, 本次收割条数)`），
**主循环与收尾路径都调用它** ⇒ 停机不丢结果。

**问题②**：收尾等待超时**写死 1800 s（30 min）** ⇒ 15ep 臂要跑 **155 min**，
固定 30 min 超时会把**还在正常训练**的臂杀掉 ⇒ 白跑 92~155 min。
**修法**：用 §10.2 的实测耗时表算「这条臂还需要多久」**+ 10 min 宽限**：
`remain_s = max(300.0, (est_total - elapsed) * 60.0 + 600.0)`，
`est_total = EPOCH_MIN_EST.get(epochs, EPOCH_MIN_EST_DEFAULT)`。

**新增（运行期可改，**不用重启调度器**）**：`<out-root>/runtime_config.json` 加两个键

| 键 | 语义 | 格式 |
|:--|:--|:--|
| `stop_at` | **墙钟停止派单** —— 过了这个时刻不再派新 trial，开始收尾 | `YYYY-MM-DD HH:MM[:SS]` 或 epoch 秒 |
| `hard_cut` | **硬截止** —— 不派「**预计跑不完**」的臂（按 epoch→耗时表 + 10% 余量判断） | 同上 |

配套常量：`EPOCH_MIN_EST = {3:47, 6:74, 8:92, 10:110, 15:155}` ·
`EPOCH_MIN_EST_DEFAULT = 110` · `EPOCH_MIN_EST_MARGIN = 1.10`；
解析函数 `parse_wallclock()`（支持 `YYYY-MM-DD HH:MM` / `HH:MM:SS` / epoch 秒，用**本机本地时区**解释，
与流水线的 `date -d` 一致）。
`read_runtime_config()` 的**白名单从 3 个键扩到 5 个**（新增 `stop_at` / `hard_cut`）。

**问题③（沙箱冒烟抓到的第二个 bug）**：`hard_cut` 触发时**只 `break` 了内层派单循环**，
外层主循环会再进来 ⇒ 又 `study.ask()` 一条 ⇒ 又 CUT ⇒ **循环** ⇒
把**所有**待派臂逐个 PRUNED，待派臂用完后还会**不停新建 TPE trial 再 PRUNED**
（**实测跑到 trial_0069** ⇒ study 被污染）。
**修法**：加 `hard_cut_hit` 标志，**外层主循环一起 break**。

**为什么**：这条链（训练截止 → 收割 → 门禁 → 判定）必须能在**无人干预**的前提下走完，
而「关机前最后一条臂能不能准时跑完」只能用**实测耗时表**回答，不能拍脑袋。

**复现命令（执行机：服务器，conda env `uav`，`export PYTHONPATH=.`，`cd /tmp/pycharm_project_10ae9e2e`）**

```bash
# 运行期改截止时刻（不用重启调度器）
cat /mnt/data/siping/runs/nogmc_focus/runtime_config.json
python3 -c "
import json,pathlib
p=pathlib.Path('/mnt/data/siping/runs/nogmc_focus/runtime_config.json')
d=json.loads(p.read_text()); d.update({'stop_at':'2026-11-03 06:20','hard_cut':'2026-11-03 06:20'})
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)); print(d)
"
# 看调度器是否认到（重启后生效的那次）
grep -nE "DEADLINE|CUT|收尾收割" /mnt/data/siping/runs/nogmc_focus/logs/dispatcher.log | tail -20
# 查 epoch→耗时估计表与白名单是否就位
grep -nE "EPOCH_MIN_EST|def parse_wallclock|stop_at|hard_cut" manu/training/optuna_mix_ratio_search.py | head -20
```

#### 10.3.2 `manu/training/restart_dispatcher_inject_arms.py` —— **杀孤儿 worker + TPE 臂等待 60→8 min**

**问题①（孤儿 worker）**：worker 是调度器的 `subprocess.Popen` 子进程
⇒ **调度器被 SIGKILL 时子进程不会被连带杀掉** ⇒ 变成**孤儿继续占 GPU**；
新调度器起来后把**同一块卡再派一条臂** ⇒ **显存争抢 / OOM**，且孤儿结果**永远回不来**。
**修法**：按调度器**自己写的 `workers.json` 登记表**（`pid` / `gpu` / `trial_name`）
先发 `SIGTERM`，5 s 后仍活着的补 `SIGKILL` ⇒ **重启前必须清理孤儿**。

**问题②（非 plan / TPE 臂的等待时间）**：原来等 **60 min** ⇒ **必然超时**
（一条 TPE 臂要 **92 min**）⇒ **白等 1 h × 4 卡 = 4 GPU·h ≈ 2.6 条 8ep 臂的产出**。
而 TPE 臂**本来就该让位给 plan 臂**（watcher 的设计前提：plan 臂优先、TPE 臂可弃）。
**修法**：等待时长**从 60 min 缩到 8 min**（`wait_deadline = time.time() + 8 * 60`），
超时后照常重启，由上面的「杀孤儿」逻辑兜底清理。

**为什么**：06:30 关机意味着**这轮不能再有"白烧 GPU"的事**；而且重启路径（watcher 会在 ~21:50 重启调度器）
本来就是**预期会发生**的路径，必须把它的浪费和副作用都收干净。

**复现命令（执行机：服务器）**

```bash
# 看孤儿登记表
cat /mnt/data/siping/runs/nogmc_focus/workers.json
# 看重启注入器日志（等 TPE 臂 / 杀孤儿 / 重启）
tail -n 40 /mnt/data/siping/runs/nogmc_focus/logs/restart_inject.log
# 确认等待时长与杀孤儿逻辑在位
grep -nE "wait_deadline|8 \* 60|workers.json|SIGTERM|SIGKILL" manu/training/restart_dispatcher_inject_arms.py
```

#### 10.3.3 `manu/training/finalize_nogmc_focus.sh` —— **⑤b T3 可达性分析 + SWA 等待可调**

**问题①**：T3 可达性分析（受控对重扫 + 汇率排序 + 种子噪声带）**不在交付链里**
⇒ 周二交出来的报告缺这一块（它是"为什么没命中"的**主要解释材料**）。
**修法**：新增 **⑤b**，调 `manu/training/analyze_t3_reachability.py`（**用 study.db 原始 float**
+ **两份 study 的 `--study` 参数**）⇒ 产出 `T3_REACHABILITY.md` / `T3_REACHABILITY.json`，
并把 `T3_REACHABILITY.md` / `T3_REACHABILITY.json` / `crossseed_manifest.json`
**加进备份清单**。

**问题②**：④b 的 SWA 等待时长**写死 45 min** ⇒ 保险层（机器马上要关机）**等不起**。
**修法**：做成**环境变量 `SWA_WAIT_MIN`（默认 45）** ⇒ 保险层可用 **`SWA_WAIT_MIN=0`** 立即出交付件
（那一版不含附录 C），最终版用 `SWA_WAIT_MIN=2` 补上附录 C。

**复现命令（执行机：服务器，conda env `uav`，`export PYTHONPATH=.`，`cd /tmp/pycharm_project_10ae9e2e`）**

```bash
# 立刻出一版交付件（不等 SWA 门禁）
SWA_WAIT_MIN=0 bash manu/training/finalize_nogmc_focus.sh
# 交付件 + T3 可达性
ls -la /mnt/data/siping/runs/nogmc_focus/HANDOVER.md \
       /mnt/data/siping/runs/nogmc_focus/HANDOVER.json \
       /mnt/data/siping/runs/nogmc_focus/T3_REACHABILITY.md \
       /mnt/data/siping/runs/nogmc_focus/T3_REACHABILITY.json
# ⑤b 单独重跑
python manu/training/analyze_t3_reachability.py \
  --study "sqlite:////mnt/data/siping/runs/nogmc_weekend/study.db" \
  --study "sqlite:////mnt/data/siping/runs/nogmc_focus/study.db" \
  --out-md /mnt/data/siping/runs/nogmc_focus/T3_REACHABILITY.md \
  --out-json /mnt/data/siping/runs/nogmc_focus/T3_REACHABILITY.json
```

#### 10.3.4 **新增** `manu/training/run_nogmc_tail_autorun.sh` —— **服务器端自主收尾编排**

> 这是本节**最关键的新增**：把"收尾链"从**依赖在线会话**改成**服务器端自主执行**。

**五步编排（全部 `setsid nohup` 脱离 SSH 会话）**

| 步 | 做什么 | 为什么 |
|:--|:--|:--|
| ① | **停掉流水线自带看门狗**，接管监督：调度器意外退出 ⇒ **断点续跑重启**（命令与流水线同一套，带同样的 `stop_at`） | 看门狗的重启**不带 `stop_at` 语义** ⇒ 调度器在 `stop_at` 优雅退出后，它会**每 60 s 重启一次**，而新调度器因 `stop_at` 已过会**立刻退出** ⇒ **空转死循环** |
| ② | **③ 收尾**：`backfill_summary_rows.py` → `rejudge_with_reference.py` → `summarize_nogmc_search.py` | 与流水线**同一套命令** ⇒ 口径一致 |
| ③ | **保险层**：`SWA_WAIT_MIN=0 bash manu/training/finalize_nogmc_focus.sh` ⇒ **先出一版完整交付件** | 门禁与 finalize 都在训练之后，而 06:30 是**硬边界** ⇒ 先落一版完整交付件，**无论后面发生什么，交接材料都已经躺在磁盘上** |
| ④ | **等门禁**（硬截止 `GATE_DEADLINE`，脚本默认 `2026-11-03 06:05`；**本轮实际以 `2026-11-03 08:00` 启动**）⇒ 到点停掉还在跑的门禁脚本（进程组），已完成的候选结果保留 | 给 finalize 留时间 |
| ⑤ | **最终 finalize**：`SWA_WAIT_MIN=2 bash manu/training/finalize_nogmc_focus.sh`（整合 SWA **附录 C**；若 SWA 命中双目标则自动封存） | 附录 C 必须在门禁跑完后才有内容 |
| ⑥ | **备份 + DONE 标记**：拷 9 个产物到 `/mnt/data/siping/runs/nogmc_focus_backup_<stamp>/`，写 `AUTORUN_DONE.json`（**含各产物 md5**） | 可对账的收尾凭证 |

**复现命令（执行机：服务器，conda env `uav`，`export PYTHONPATH=.`，`cd /tmp/pycharm_project_10ae9e2e`）**

```bash
# 启动自主收尾（脱离 SSH 会话）
setsid nohup bash manu/training/run_nogmc_tail_autorun.sh \
  >> /mnt/data/siping/runs/nogmc_focus/logs/autorun.log 2>&1 < /dev/null &
# 看它跑到哪
tail -n 40 /mnt/data/siping/runs/nogmc_focus/logs/autorun.log
# 收尾凭证
cat /mnt/data/siping/runs/nogmc_focus/AUTORUN_DONE.json
```

### 10.4 新的时间线（**服务器时间**，2026-11-03）

> ⚠️ **2026-11-02 18:23 用户更正（覆盖早先的 04:00/06:05 版本）**：
> **06:30 只关本地机，服务器（`test-Super-Server`）照常跑到用户 09:00 到岗**。
> 早先把墙钟截止定在 04:00 是因为**误判「服务器可能也关机」** ⇒ 现已放宽 2h20m。

**新的 `runtime_config.json`**（已写入服务器 `/mnt/data/siping/runs/nogmc_focus/runtime_config.json`，实测已确认）：

```json
{ "stop_at": "2026-11-03 06:20", "hard_cut": "2026-11-03 06:20" }
```

| 时刻 | 事件 |
|:--|:--|
| **18:25** | `runtime_config` 放宽到 **06:20**；autorun 以 `GATE_DEADLINE=2026-11-03 08:00`、`FINAL_DEADLINE=2026-11-03 08:45` 重启（**PID 798381**，实测 `lstart = Mon Nov 2 18:25:03 2026`） |
| **~21:50** | watcher（**PID 793475**，新代码，`lstart = 18:12:26`）重启调度器 ⇒ **22 条待注入臂入队** |
| **~05:35** | **最后一条臂跑完**（**19/22** 条臂完成；第 20 条在派单时被 `hard_cut` 剪枝、第 21、22 条保持 WAITING） |
| **~05:37** | 调度器**收割收尾并退出** ⇒ 两条门禁线开跑（**SWA 6 个 + 跨种子 3 个**，各 ~10 min）；autorun 同时出**保险层**交付件 |
| **~07:13** | 两条门禁线跑完 |
| **~07:35** | 最终 finalize（判定 + 固化 RC + SWA 附录 C + 备份 + `AUTORUN_DONE.json`）完成 |
| **09:00** | 用户到岗 ⇒ **交付物已就位（比 09:00 早约 1.5 h）** |
| **06:30** | **本地机**关机 ⇒ **不影响服务器上的任何进程**（训练 / 门禁 / 收尾全是 `setsid nohup`） |

### 10.4.1 为什么是 **19/22** 而不是 22/22（**可复用的算账结论**）

**这一步必须先算账再排期，不能反过来。**

| 算项 | 数值 |
|:--|:--|
| 22 条待注入臂的 epoch 构成（`manu/training/nogmc_focus_plan.json` 第 24~45 条） | **18×8ep**（92 min）+ **3×6ep**（74 min）+ **1×15ep**（155 min） |
| **总 GPU 需求** | **2033 GPU·min**（按 `EPOCH_MIN_EST` 常量口径；按实测中位数口径为 2035.8 GPU·min） |
| 4 卡跑完 22 条的**理论 makespan** | **541 min** ⇒ 从 21:50 起算 ⇒ 结束在 **06:51** |
| 门禁窗口 | **~96 min**（9 个候选 × 10 min + 跨种子 soup 构建 6 min） |
| finalize | **~20 min** |
| ⇒ 门禁必须在 **~08:00 前**开始 | 用户 09:00 到岗 + 缓冲 |
| ⇒ 反推训练截止 | **只能到 ~06:20** |
| ⇒ 实际可跑 | **19 条**（**第 20 条**在派单时按 epoch→耗时表被 `hard_cut` 剪枝，**第 21、22 条保持 WAITING、从未派单**） |

**为什么是 19 而不是 20（关键在「派单判定用的是含 10% 余量的估计」）**：

- 从 **21:55** 起按计划顺序做 4 机列表调度 ⇒ **19 条**臂能跑完，**最后一条结束于 05:35**（与原估一致）。
- **第 20 条**（`FOCUS_T0108_SEED2`，8ep）约在 **04:48** 派单，**真实耗时 92 min ⇒ 结束正好 06:20 = `hard_cut` 边界**；
  但派单判定用的是**含 10% 余量的 101 min**（⇒ **06:29 > 06:20**）⇒ **被剪枝**。
- **第 21、22 条**（`FOCUS_T0008_W20_SEED2` / `FOCUS_T0008_WARMUP12`）**保持 WAITING，从未派单**。
- ⇒ 被牺牲的是 **3 条额外种子臂**（`FOCUS_T0108_SEED2` / `FOCUS_T0008_W20_SEED2` / `FOCUS_T0008_WARMUP12`），
  **全是价值最低的补种子臂 —— 不是模型靶心臂，也不是 4 条关键种子臂**。

**为什么不再放宽 `hard_cut` 去抢回第 20 条**（算账结论）：

放宽到 **06:30** ⇒ 最后一条臂 06:20 结束 ⇒ 门禁 **06:22~07:58**（9 个候选 + soup 构建 96 min）
⇒ finalize **07:58~08:18** ⇒ **只剩 42 min 缓冲**，且**跨种子门禁线会贴着 `GATE_DEADLINE=08:00` 被杀**
（**可能砍掉最后 1 个候选**）⇒ **用 1 条低价值种子臂换 1.4 h 缓冲不划算 ⇒ 维持 `hard_cut=06:20`。**

> 🔴 **教训（通用）**：**门禁窗口（候选数 × 单候选耗时）是硬约束，它反推训练截止**。
> ⇒ 「**多跑几条臂**」和「**保住门禁窗口**」必须**一起算**，**不能先按训练排期、再回头想门禁** ——
> 那样算出来的截止时刻一定是错的。
> ⇒ ⚠️ **派单判定必须用「含余量的保守估计」**（本轮 `EPOCH_MIN_EST × 1.10`）：
> 余量不足 ⇒ 会派出**恰好卡在 `hard_cut` 边界**的臂（真实耗时 06:20 = 边界，但判定算出 06:29）⇒ 白等一轮。

**复现命令（执行机：服务器，conda env `uav`，`export PYTHONPATH=.`，`cd /tmp/pycharm_project_10ae9e2e`）**

```bash
# 22 条待注入臂的构成（第 24~45 条）与总 GPU 需求
python3 -c "
import json
from collections import Counter
arms=json.load(open('manu/training/nogmc_focus_plan.json'))[23:]
EST={8:92,6:74,15:155}
print('n =',len(arms),'epochs dist =',Counter(a['params']['epochs'] for a in arms))
print('total GPU·min =',sum(EST[a['params']['epochs']] for a in arms))
for i,a in enumerate(arms,start=24):
    print(i, a['role'], 'ep=', a['params']['epochs'])
"
# 4 机列表调度复算（含 10% 余量的派单判定 vs hard_cut=06:20）
python3 -c "
import json
from datetime import datetime, timedelta
arms=json.load(open('manu/training/nogmc_focus_plan.json'))[23:]
EST={6:74,8:92,15:155}; M=1.10
HC=datetime(2026,11,3,6,20); START=datetime(2026,11,2,21,55)
bins=[START]*4; done=[]; pruned=[]
for i,a in enumerate(arms, start=24):
    ep=a['params']['epochs']; real=EST[ep]; disp=real*M
    k=min(range(4), key=lambda j: bins[j]); dt=bins[k]
    fr=dt+timedelta(minutes=real); fe=dt+timedelta(minutes=disp)
    if fe>HC: pruned.append((i,a['role'],ep,dt,fe,fr)); continue
    bins[k]=fr; done.append((i,a['role'],ep,dt,fr))
print('跑完 =',len(done),' 剪枝 =',len(pruned))
print('最后一条结束 =', max(f for *_,f in done).strftime('%H:%M'))
for i,role,ep,dt,fe,fr in pruned:
    print('  #%d %s ep%d: 派单 %s | 含余量结束 %s (> hard_cut) | 真实结束 %s'%(i,role,ep,dt.strftime('%H:%M'),fe.strftime('%H:%M'),fr.strftime('%H:%M')))
"
```

### 10.4.2 `hard_cut` 的耗时预估**可信**（实测印证）

调度器日志实测：`trial_0013`（10ep）实耗 **110.5 min**、`trial_0012`（10ep）**110.8 min**
⇒ 与新常量表 **`EPOCH_MIN_EST[10] = 110`** 吻合，**误差 < 1 %**
⇒ **说明 `hard_cut` 的耗时预估可信**（它就是靠这张表判断「这条臂跑不跑得完」的）。

**复现命令（执行机：服务器，conda env `uav`，`export PYTHONPATH=.`，`cd /tmp/pycharm_project_10ae9e2e`）**

```bash
# 从 study.db 现算每条臂的实耗（datetime_start → datetime_complete）
python3 -c "
import sqlite3, datetime
db=sqlite3.connect('/mnt/data/siping/runs/nogmc_focus/study.db'); c=db.cursor()
q='''select tr.number, tr.state, tr.datetime_start, tr.datetime_complete,
 (select param_value from trial_params p where p.trial_id=tr.trial_id and p.param_name=\"epochs\")
 from trials tr'''
P=lambda s: datetime.datetime.fromisoformat(s) if s else None
for num,st,s,e,ep in c.execute(q).fetchall():
    if st!='COMPLETE' or not s or not e: continue
    print('trial_%04d epochs_idx=%s dur=%.1f min' % (num, ep, (P(e)-P(s)).total_seconds()/60))
"
# 看常量表与 hard_cut 判定
grep -nE "EPOCH_MIN_EST|hard_cut" manu/training/optuna_mix_ratio_search.py | head -20
```

### 10.5 复现命令总表（**可直接复制执行**）

**服务器实时看板（执行机：服务器 `192.168.99.40:32222`，conda env `uav`，`export PYTHONPATH=.`，`cd /tmp/pycharm_project_10ae9e2e`）**

```bash
date "+%F %T %Z"
cat /mnt/data/siping/runs/nogmc_focus/runtime_config.json
tail -n 20 /mnt/data/siping/runs/nogmc_focus/logs/autorun.log
tail -n 10 /mnt/data/siping/runs/nogmc_focus/logs/restart_inject.log
tail -n 5  /mnt/data/siping/runs/nogmc_focus/logs/gate_swa.log
tail -n 5  /mnt/data/siping/runs/nogmc_focus/logs/gate_crossseed.log
ps -eo pid,etime,args | grep -E "optuna_mix_ratio_search|run_nogmc_tail_autorun|gate_swa_variants|gate_crossseed_soups" | grep -v grep
```

**入队冒烟（沙箱，**不碰真 study** —— 脚本内部把 study 备份到 `/tmp/_nogmc_focus_dryrun.db` 后在副本上跑）**

```bash
python3 manu/training/dryrun_plan_enqueue.py --plan manu/training/nogmc_focus_plan.json \
  --out-root /mnt/data/siping/runs/nogmc_focus --study-name mix_ratio_search_nogmc_focus
# 实测：成功 27 条 / 失败 0 条
```

### 10.6 本节沉淀的**通用结论**

1. **「定时关机」查不到 ⇒ 必须按「一定会关机」排期**，且关机**不等于**进程被杀 ——
   关键是**主动脱离会话**（`setsid nohup`）而不是**赌它不关机**。
2. **收尾链必须服务器端自主执行**：会话离线不可怕，**脚本依赖在线点命令**才可怕。
3. **先出保险层交付件**：再完善的收尾链也要有一版"已经躺在磁盘上"的完整交付件兜底。
4. **墙钟截止要用实测耗时表**，不能拍脑袋 —— §10.2 的每臂耗时极差只有 1.5~2.4 min，
   拿它做估计才敢写 `hard_cut`。
5. **截止逻辑本身也要冒烟**：`hard_cut` 只 break 内层循环 ⇒ study 被污染到 trial_0069
   ⇒ **任何"到点就停"的分支，都要在沙箱里单独验一遍外层循环会不会再进来**。
---

## 十一、⭐⭐ **「定点打靶 overnight」实测结局**（2026-11-03 05:32:53 ~ 08:48，**实测 vs 计划**）

> 本节是 **§十（11-02 晚写的「计划版」）的实测对照**。**§十 原样保留**（那是过程记录），本节只写**实际发生了什么**。
> **执行机**：服务器 `huangzhe@192.168.99.40 -p 32222`（`test-Super-Server`，4×RTX 4090D），项目目录 `/tmp/pycharm_project_10ae9e2e`，conda env `uav`，`export PYTHONPATH=.`。
> **时钟口径**：全部按**服务器**读（本轮两机时钟已一致，18:19:02 = 18:19:02）。
> **产物根目录**：`/mnt/data/siping/runs/nogmc_focus/`（下称 `<OUT>`）。

### 11.0 一句话结局

**双目标（T3 ≥ 0.87 且 T1 ≥ 0.895897）没有达成 —— 但不是「没跑够」，而是「这套先验空间里两者构成硬权衡」。**
交接口径判定器 **rc=3**（有候选、但种子数不足）⇒ **按口径尚不可正式交接**；交付件（判定 / 可达性 / 固化 RC / 备份）**已全部落盘**。

| 项 | §十 的**计划** | **实测**（本节） | 差异 |
|:--|:--|:--|:--|
| 训练截止 | `hard_cut = 06:20` | `hard_cut = 06:20`（`runtime_config.json` 实测确认） | ✅ 一致 |
| 待注入臂 | **19/22**（§10.4.1 的算账结论：第 20 条卡 `hard_cut` 边界、第 21/22 条 WAITING） | **19 条 COMPLETE**；被剪的 3 条 = `FOCUS_T0108_SEED2`/`FOCUS_T0008_W20_SEED2`/`FOCUS_T0008_WARMUP12` | ✅ **条数与身份完全一致**（§十 算出来的就是这 3 条） |
| study 终态 | 未预估 | **94 trial = 39 COMPLETE + 51 PRUNED + 4 FAIL** | — |
| 空转 | 未预估 | 🔴 **48 次自动重启**，其中 **47 次是硬截止后的空转循环**（05:33:10~06:19:10，每 60 s 一次）⇒ 造出 **45 条无 role 的垃圾 PRUNED**（t0049~t0093） | ❌ **计划外的新缺陷** |
| 训练实际结束 | ~05:35 | **05:32:53**（最后 3 条臂 t0043/t0044/t0045 同时完成） | ✅ 早 2 min |
| 门禁开始 | ~05:37（训练一结束就开跑；硬截止 08:00） | **05:33:43**（`gate_swa.log`：「训练已结束、GPU 已释放 ⇒ 开始串行门禁」） | ✅ 早 3 min；**评估未与训练同卡**（铁律遵守） |
| 跨种子 soup 门禁 | ~05:37 与 SWA 一起开跑（SWA 6 个 + 跨种子 3 个，各 ~10 min） | **构建全部失败**（枚举 bug）⇒ 修好后重建 3 个，**08:44:23 开跑、09:14:31 跑完** ⇒ **1 个严格 PASS（`cs_t0008`）+ 2 个 FAIL** | ❌ **延后 ~3.1 h**（构建 bug 是唯一原因）；**结果本身是本轮最有价值的新发现**（见 §11.4.2） |
| 交接口径 | 未出 | **rc=3**（候选 n=2、T1 余量 +0.0102 pp、T3 差 2.477 pp）⇒ **需人工复核** | — |

### 11.1 训练侧终态（实测）

**调度器改动（`manu/training/optuna_mix_ratio_search.py`）**：
`runtime_config.json` 新增 `stop_at` / `hard_cut`（**墙钟截止**，运行期可改）· 抽出 `harvest(workers)`（**停机收割**，主循环与收尾路径共用）·
`EPOCH_MIN_EST = {3:47, 6:74, 8:92, 10:110, 15:155}`（分钟，**实测中位数**）· `hard_cut_hit` 标志（内层 break 后外层一起 break）。

**study `mix_ratio_search_nogmc_focus` 终态（实测 `study.db`）**：

| 项 | 值 |
|:--|--:|
| trial 总数 | **94** |
| COMPLETE / PRUNED / FAIL | **39 / 51 / 4** |
| plan 45 条 role 里**至少有一条 COMPLETE** 的 | **39**（缺 6：`FOCUS_B_T0060_PROT15` / `FOCUS_T0008_W25` / `FOCUS_SOUHIGH3NOBIN_W3_LR28821_8EP` / `FOCUS_T0108_SEED2` / `FOCUS_T0008_W20_SEED2` / `FOCUS_T0008_WARMUP12`） |
| **22 条待注入臂实际完成** | **19 条** |
| 被剪的 3 条 | `FOCUS_T0108_SEED2`(**t0046**) / `FOCUS_T0008_W20_SEED2`(**t0047**) / `FOCUS_T0008_WARMUP12`(**t0048**) —— **全是价值最低的额外种子臂，不是模型靶心臂、不是 4 条关键种子臂** |
| 4 条 FAIL | t0023 / t0024 / t0025 / t0026（TPE 臂，无 role；11-02 白天阶段） |

**🔴 实测抓到一个真缺陷：`hard_cut` 触发 ⇒ 调度器退出 ⇒ autorun 监督循环把它**再拉起来** ⇒ 「启动 → CUT → 退出 → 重启」死循环。**

| 项 | 实测值 |
|:--|:--|
| autorun 自动重启总次数 | **48 次**（`autorun.log`「⚠️ 调度器不在 ⇒ 第 N 次自动重启」；**#1 = 11-02 21:54:07 的正常重启**，**#2~#48 = 47 次空转**） |
| 空转窗口 | **05:33:10 ~ 06:19:10**，**每 60 s 一次**（正好是监督循环的轮询周期） |
| 每次的产物 | 调度器起来 → 立刻 CUT 一条 trial → 退出 ⇒ **每轮多造 1 条 PRUNED** |
| 垃圾 PRUNED | **45 条无 role 的 TPE trial（t0049~t0093）**；另 2 轮（#2/#3）恰好把本来就跑不完的 `t0047`/`t0048` 剪掉 ⇒ **PRUNED 总数 51** |
| 根因 | autorun 的重启判定**只看「调度器是否活着」**，**没看「它是不是被硬截止主动收尾的」** ⇒ 被硬截止"正常退出"误判成"崩了" |
| **修法** | **重启前先判断是否已过 `stop_at`**（已过 ⇒ **不再重启**，直接进收尾阶段） |

**训练结束 / 门禁开始的时序（铁律「评估与训练不得同卡」的自检口径）**：

| 事件 | 服务器时刻 | 证据 |
|:--|:--|:--|
| 最后 3 条臂完成（t0043/t0044/t0045） | **05:32:53** | `study.db` `datetime_complete` = `2026-11-03 05:32:53.916944` |
| 门禁开始（SWA 串行门禁） | **05:33:43** | `logs/gate_swa.log`：「训练已结束、GPU 已释放 ⇒ 开始串行门禁」 |
| ⇒ 判定 | **门禁开始时刻晚于最后一条训练臂的完成时刻 ⇒ ✅ 评估未与训练同卡** | — |

**复现命令（执行机：服务器，conda env `uav`，`export PYTHONPATH=.`，`cd /tmp/pycharm_project_10ae9e2e`）**

```bash
# study 终态（94 = 39 COMPLETE + 51 PRUNED + 4 FAIL）
python3 - <<'EOF'
import optuna, collections
st = optuna.load_study(storage="sqlite:////mnt/data/siping/runs/nogmc_focus/study.db",
                        study_name="mix_ratio_search_nogmc_focus")
rows = [(t.number, t.state.name, t.datetime_start, t.datetime_complete, (t.user_attrs or {}).get("role"))
        for t in st.trials]
print("total:", len(rows), dict(collections.Counter(r[1] for r in rows)))
comp = sorted([r for r in rows if r[1] == "COMPLETE" and r[3]], key=lambda r: r[3])
print("最后一条 COMPLETE:", comp[-1])          # (45, ..., '2026-11-03 05:32:53.916944')
print("PRUNED:", [(r[0], r[4]) for r in rows if r[1] == "PRUNED"])
EOF

# 空转循环：48 次自动重启（#1 正常 + #2~#48 空转）
grep -c "次自动重启" /mnt/data/siping/runs/nogmc_focus/logs/autorun.log     # 96 行（每条日志打印 2 遍）⇒ 48 次
grep "次自动重启" /mnt/data/siping/runs/nogmc_focus/logs/autorun.log | tail -3
# 硬截止日志：每次 CUT 都跟着「硬截止已触发 ⇒ 不再派新 trial，进入收尾」
grep -nE "\[CUT\]|硬截止已触发" /mnt/data/siping/runs/nogmc_focus/logs/dispatcher.log | tail -6

# 门禁 vs 训练的时序（铁律自检）
grep -n "训练已结束、GPU 已释放" /mnt/data/siping/runs/nogmc_focus/logs/gate_swa.log
```

### 11.2 交接口径判定结果（`HANDOVER.md`，2026-11-03 06:33:52 保险层 + 08:45 重跑）

**判定器 rc = 3**（有候选，但**种子数不足**）⇒ **按口径尚不可正式交接**。

**交接候选组**（`focus:trial_0040` + `weekend:trial_0083` = 配方 `w8.0·lr2.8819·rc2·8ep·bb1·fbb0`）：

| 项 | 值 |
|:--|--:|
| 种子数 | **n = 2** |
| U 中位 | **+0.083843 ± 0.000636** |
| T1 中位 | **0.895999**（红线 0.8958967482400269 ⇒ **余量 +0.0102 pp**） |
| T2 中位 | F1 **0.998409** / 虚警中位 **48**（上限 112） |
| T3 中位 | **0.845233 ± 0.002400**（目标 ≥ 0.87 ⇒ **未达，差 2.477 pp**） |

**判定链**：U **不可分** ⇒ T3 **仍不可分** ⇒ 种子数**并列** ⇒ 取 U 更高者 ⇒ **结论标注「需人工复核」**。

**阻塞项（必须显式交代）**：
① 候选 T1 余量仅 **0.0102 pp**（≈ **0.4σ** 种子噪声，种子噪声 |ΔT1| 中位 0.002464）⇒ **建议再补 1 个种子**；
② 候选 T3 中位 **0.845233**，**差 2.477 pp**。

**严格通道配置榜第 1 名（也是离双目标最近的单点）**：`w3.0·lr2.4220·rc2·8ep·bb1·fbb0`（= focus `trial_0031`），
n=1 · U **+0.105918** · T1 **0.896061**（余量 **+0.0164 pp**）· T2 FP **46** · T3 **0.867107**（**距 0.87 只差 0.289 pp**）。

**通道规模**：严格通道 **43 条严格 PASS / 39 组 / 41 个 trial**；放宽通道 **109 条**（仅诊断，不可交接）。

**复现命令（执行机：服务器；`--csv` 是 `nargs="+"` ⇒ **一个 `--csv` 跟多个值**）**

```bash
python3 manu/training/judge_handover_candidate.py \
  --csv "focus=/mnt/data/siping/runs/nogmc_focus/optuna_summary.csv" \
         "weekend=/mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv" \
  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --out-md /mnt/data/siping/runs/nogmc_focus/HANDOVER.md \
  --out-json /mnt/data/siping/runs/nogmc_focus/HANDOVER.json \
  --min-seeds 2
# 实测：rc=3（候选存在但种子数不足）；产物 /mnt/data/siping/runs/nogmc_focus/HANDOVER.md|.json
```

### 11.3 T3 可达性论证（`T3_REACHABILITY.md`，2026-11-03 08:45:37 重生成）

| 项 | 值 |
|:--|--:|
| 纳入分析 | **156 个 trial**（**43** 严格 PASS + **109** 放宽） |
| 严格通道 T1–T3 Pearson / Spearman | **-0.721** / **-0.687**（强负相关；⚠️ **受维度耦合影响，只能当背景**） |
| **严格 PASS 内 T3 实测上界** | **0.867107**（`trial_0031`）⇒ **差 0.289 pp** |
| 受控对总数（只差一个维度） | **74 对** |
| **杠杆分型**（按 **population** 判该维度是「买 T3」还是「换 T1」，再排序） | **买 T3 型**：`lr_scale` **0.045**（**29/32** 对涨 T3）、`epochs` **0.104**（**6/6** 对涨 T3）；**拿 T3 换 T1 型**：`mix_spec`（**仅 3/24** 对涨 T3、**21/24** 对掉 T3、T1 升 15/24）；`scale` 汇率 −0.324 但 **T3 变大的对数 0/2（空 population）⇒ 不参与排序**；`backbone_lr_scale` 仅 1 对 |
| mix 单变量受控对 | **24 对**：mix 变大时 **T1 上升 15/24、T3 下降 21/24** |
| 类别维度 `init_from`（rc2 → base474） | 逐对 T1 Δ ≈ **−0.004~+0.0033**、T3 Δ ≈ **−0.0146~−0.0388** ⇒ **换 init 掉 T3，不划算** |
| **种子噪声** | \|ΔT1\| 中位 **0.002464**（最大 0.003104）· \|ΔT3\| 中位 **0.002995**（最大 0.005394） |
| 放宽通道 T3 最高 | **0.890979**；T3 ≥ 0.87 的有 **25 条但全部 T1 触线**；离红线最近的是 `trial_0033`（T3 **0.872645** / T1 **0.895565**，**仍缺 0.0332 pp**） |

**⇒ 结论**：**T3 ≥ 0.87 不是做不到，而是与 T1 红线构成硬权衡**；两个方向各差 **0.03~0.29 pp** ⇒
**双目标就卡在 Pareto 前沿的拐角上，这套先验空间内没够到**；
且 **0.289 pp 的缺口比种子噪声（0.002995，中位）大一个量级** ⇒ **不是种子运气能补的**；
**要买 T3 走 `lr_scale`（不是 mix）** ⇒ 见 §11.3.1。

#### 11.3.1 🔴 汇率表（§4.1）**口径缺陷已修**：旧表的「走 mix 杠杆」建议**方向是反的**

🔴 **缺陷**：`analyze_t3_reachability.py` 的 `rate`（**每换 1 单位 T3 付的 T1**）**只在 `d_t3 > 0` 的对子子集上取中位**，
而同一行的「ΔT1/单位 / ΔT3/单位」是**全部**实质对子的中位 ⇒ **两列不是同一批样本，符号可以互相矛盾**。

**实测后果**：旧表把 `mix_spec` 排成「最划算的买 T3 杠杆（**−0.128**）」，但 mix 的聚合斜率明明是
**T1↑ +0.000500 / T3↓ −0.005435**（**拿 T3 换 T1**）⇒ **自相矛盾**；
且 `scale` 的汇率 **−0.324** 建立在「**T3 变大的对数 0/2**」的**空 population** 上，却被排到第一。

**修法（已改并同步，服务器 09:02 重生成）**：① 汇率改为在**全部实质对子**上算；② 表格加一列「**其中 T3 变大的对数**」让 population 显式；
③ 结论改为**按 population 分型**后再排序。

**修正后的汇率表（实测，`T3_REACHABILITY.md` §4.1）**：

| 变动维度 | 受控对数 | 该维度变大时 T1 上升 | T3 下降 | **其中 T3 变大的对数** | ΔT1/单位 | ΔT3/单位 | 每换 1 单位 T3 付的 T1 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `scale` | 2 | 0/2 | 2/2 | **0/2** | −0.016610 | −0.045800 | **−0.324**（⚠️ 空 population，不参与排序） |
| `lr_scale` | 32 | 12/32 | 3/32 | **29/32** | −0.000898 | +0.018971 | **0.045** |
| `mix_spec` | 24 | 15/24 | 21/24 | **3/24** | +0.000500 | −0.005435 | **0.062** |
| `epochs` | 6 | 1/6 | 0/6 | **6/6** | −0.000322 | +0.003552 | **0.104** |
| `backbone_lr_scale` | 1 | 1/1 | 1/1 | **0/1** | +0.001102 | −0.002084 | **0.529**（仅 1 对，不排序） |

**⇒ 修正后的结论（按 population 分型）**：

- **买 T3 型**（该维度以 **T3 变大**为主，按汇率升序）：**`lr_scale` 0.045（29/32 对涨 T3）**、**`epochs` 0.104（6/6 对涨 T3）**；
- **拿 T3 换 T1 型**（该维度以 **T3 下降**为主）：**`mix_spec`**（**只有 3/24 对涨 T3**、**21/24 对掉 T3**、**T1 上升 15/24**）。
- ⇒ **要买 T3 走 `lr_scale`；`mix_spec` 是「涨 T1、代价掉 T3」的杠杆，不能当买 T3 的路。**

**直接受控对证据（与修正后结论一致）**：

| 受控对（只差一个维度） | T1 Δ | T3 Δ | 读法 |
|:--|--:|--:|:--|
| 只差 `lr`（2.8821 → 3.0，其余全同）：focus `trial_0029` → `trial_0033` | **−0.011 pp** | **+0.377 pp** | **lr 是买 T3 的杠杆** |
| 只差 `mix`（w2.0 → w3.0，其余全同）：focus `trial_0028` → `trial_0033` | **+0.172 pp** | **−0.837 pp** | **mix 是换 T1 的杠杆** |

#### 11.3.2 `trial_0033` 的精确计数（**用户明确要求过线的候选**）

`focus:trial_0033` = **`FOCUS_T0008_LR30`**（原始 float）：mix **w3.0** / lr **3.0** / **8ep** / `rc2` /
`bb_lr` 1.0 / `fbb` 0 / focal 2.75 / offset 0.4 / mosaic 0.15 / scale 0.1 / translate 0.05 / clip 5.65 / warmup 1.5 / wd 0.00026906。

| 轨 | 规模 / GT | 冻结基线（th 0.25） | 微调（th 0.30） | Δ | 判定 |
|:--|:--|:--|:--|:--|:--|
| **Track1**（Anti-UAV） | 31,613 帧 / GT 25,111 | TP 21,499 / FP 1,118 / FN 3,612 / P 0.950568 / R 0.856159 / F1 **0.9008967** | TP 21,254 / FP 1,100 / FN 3,857 / **P 0.950792（涨）** / R 0.846402 / F1 **0.8955652** | ΔTP **−245** · ΔFP **−18** · ΔFN **+245** · ΔF1 **−0.005332** | ❌ T1 **差 0.000332 F1 = 0.0332 pp**；**FP/FN 不变时再 +76 个 TP 即过线**（recall 需 +0.3027 pp） |
| **Track2**（黄金池） | 17,187 帧 / GT 16,638 | F1 0.981074 | F1 **0.998439** / TP 16,634 / FP **48**（上限 112，比基线 52 还少 4）/ FN **4**（基线 568，**−564**） | ΔF1 **+0.017365** | ✅ **严格 PASS** |
| **Track3**（001324 盲测） | 11,810 帧 / GT 10,535 | F1 0.757812 | F1 **0.8726446**（**超 0.87 目标 +0.2645 pp ✅**）/ TP 8,637 / FP 623 / FN 1,898 | ΔF1 **+0.114833** · FN **−1,883** | ✅ **T3 过线** |

🔴 **两个必须记住的细节**：

1. **不是乱抓掉分，是漏检多 245 个** —— **precision 反而涨（0.950568 → 0.950792）**、**FP 还少 18 个** ⇒ 失分全在 **recall**。
2. **门禁卡峰值，不卡均值** —— `peak_f1` **−0.005332**，而 **`mean_f1` +0.002547**（15 个 band 里 8 个为负）
   ⇒ 按均值口径，这条臂**比基线还高 0.25 pp** ⇒ **别用均值口径给它判死刑**。

**复现命令（执行机：服务器）**

```bash
# ⚠️ analyze_t3_reachability.py 的 --csv 是 action="append" ⇒ 每个 CSV 一个 --csv（见 §11.5 bug ②）
python3 manu/training/analyze_t3_reachability.py \
  --csv "focus=/mnt/data/siping/runs/nogmc_focus/optuna_summary.csv" \
  --csv "weekend=/mnt/data/siping/runs/nogmc_weekend/optuna_summary.csv" \
  --study "focus=/mnt/data/siping/runs/nogmc_focus/study.db:mix_ratio_search_nogmc_focus" \
  --study "weekend=/mnt/data/siping/runs/nogmc_weekend/study.db:mix_ratio_search_nogmc_weekend" \
  --ref /mnt/data/siping/runs/nogmc_ref_weekend/reference_frame.json \
  --out-md /mnt/data/siping/runs/nogmc_focus/T3_REACHABILITY.md \
  --out-json /mnt/data/siping/runs/nogmc_focus/T3_REACHABILITY.json
# 实测：rc=0；156 个 trial（43 严格 + 109 放宽）；strict_t3_max=0.867107(trial_0031)
# ⚠️ `--study` 的格式是 TAG=study.db路径:study_name（study_name 可省，默认 mix_ratio_search_nogmc_focus）
```

### 11.4 两条**零训练成本**新路线的实测结论

#### 11.4.1 epoch 轴 SWA（同一 run 末 3 epoch 平均）

6 个候选**全部门禁跑完**（每个 ~10 min，实测 578.6~592.9 s）：

| 候选 | T1 | T1 缺口 | T2 F1 (FP) | T3 | T3 ≥ 0.87 | 硬门禁 |
|:--|--:|--:|--:|--:|:--:|:--|
| `swa_0082_last3` | **0.896274** | **+0.0378 pp** | 0.998589 (43) | 0.865366 | ❌ | ✅ **唯一严格 PASS** |
| `swa_0009_last3` | 0.895884 | −0.0013 pp | 0.997960 (63) | 0.855812 | ❌ | ❌ T1 触线 |
| `swa_0008_last3` | 0.895354 | −0.0543 pp | 0.998409 (49) | 0.867644 | ❌ | ❌ T1 触线 |
| `swa_0077_last3` | 0.894426 | −0.1471 pp | 0.998679 (40) | **0.870708** | ✅ | ❌ T1 触线 |
| `swa_0113_last3` | 0.894414 | −0.1483 pp | 0.998320 (52) | **0.872660** | ✅ | ❌ T1 触线 |
| `swa_0087_last3` | 0.893881 | −0.2016 pp | 0.998589 (43) | **0.873045** | ✅ | ❌ T1 触线 |

**⇒ SWA 的真实收益：把 `trial_0082` 从「放宽通道」抬进「严格 PASS」（T1 0.895751 → **0.896274**；T2 FP **43**，基线 52、**降 9**），
但没有把任何配置推过 T3 0.87 ⇒ 双目标仍未达成。**
`SWA_SUMMARY.json` 实测 `n_candidates=6 / n_strict_ok=1 / n_hit=0` ⇒ **判定规则要求 T3 ≥ 0.87 才算命中，口径正确**。

#### 11.4.2 跨种子 soup（同配方 · 不同种子 · 末 epoch 平均）

🔴 **首次构建全部失败（一个 soup 都没建成）** ⇒ 根因是 `manu/training/build_crossseed_soups.py` 的**枚举比较 bug**：

- 第 91 行存 **`t.state.value`**（数值：Optuna `TrialState` 枚举 **RUNNING=0 / COMPLETE=1 / PRUNED=2 / FAIL=3 / WAITING=4**），
- 第 107 行却用 **`state != "COMPLETE"`**（字符串）⇒ **恒为真** ⇒ **所有已完成臂都被当成「未完成」跳过**
  ⇒ 每族只剩 1 个 weekend 种子 ⇒ 「可用种子只有 1 个 ⇒ 不够平均，跳过」。
- **修法：存 `t.state.name`**（修后代码已进仓库：`manu/training/build_crossseed_soups.py` 的 `role2trial[r] = (t.number, t.state.name)`）。

**修后重建成功（2026-11-03 08:44:18，rc=0）**：

| soup | 种子数 | 成员 | 体积 | md5 |
|:--|--:|:--|--:|--|
| `cs_t0008` | **4** | weekend 0008 + focus 0015 / 0020 / 0029 | 7,380,920 B | `1964b74f733180178fbadd920c7aca3e` |
| `cs_t0077` | **3** | weekend 0077 + focus 0022 / 0030 | 7,380,856 B | `de425c9573b74ba89dbaf3827db66764` |
| `cs_t0082` | **3** | weekend 0082 + focus 0021 / 0031 | 7,380,856 B | `ceeaf98b17df668232bde2c725a39b1e` |

**跨种子门禁结果（2026-11-03 08:44:23 ~ 09:14:31，三个候选**全部跑完**，各 ~10 min：578.6 / 601.1 / 592.5 s）**：

| soup | 种子数 | T1 | T1 距红线 | T2 F1 (FP) | T3 | T3 ≥ 0.87 | 门禁结论 |
|:--|--:|--:|--:|--:|--:|:--:|:--|
| `cs_t0077` | 3 | 0.894400 | **−0.1497 pp（≈341 个 TP）** | 0.998709 (**39**，基线 52、−13) | **0.870514** | ✅ | **FAIL**（卡 T1） |
| `cs_t0082` | 3 | 0.895868 | **−0.0029 pp（= 7 个 TP）** | 0.998649 (**41**，基线 52、−11) | 0.865307 | ❌（差 0.469 pp ≈ 355 TP） | **FAIL**（卡 T1） |
| `cs_t0008` | 4 | **0.8959005163873961** | **+0.0004 pp（压线过，≈1~2 个 TP）** | 0.998619 (**42**，基线 52、−10) | 0.868572 | ❌（差 0.1428 pp） | ✅ **PASS** |

🔴 **跨种子 soup 的真实价值（本轮最有价值的新发现）**：

- ✅ **`cs_t0008`（4 种子：weekend `trial_0008` + focus `trial_0015`/`0020`/`0029`）= 本轮唯一一条来自 soup 路线的新严格 PASS**
  （T1 **0.8959005** 压线过 + T2 F1 0.998619 / FP **42**）。T1 精确计数：TP **21,253** / FP 1,081 / FN 3,858 /
  P **0.951598（涨）** / R 0.846362；Δpeak_f1 **−0.004996**、**mean_f1 +0.002322**（15 band 里 8 负）⇒ 仍是**漏检吃亏、均值高于基线**。
  ⚠️ **但 T1 余量只有 +0.0004 pp（≈1~2 个 TP）⇒ 极脆，不能当稳态交接件**；T3 **0.868572** 距 0.87 还差 **0.1428 pp** ⇒ **双目标仍差一口气**。
- **`cs_t0082` 只差 7 个真阳就过 T1 红线** ⇒ **跨种子平均把同配方的 T1 抬到了「离红线 7 个 TP」的位置**
  （t0082 族单条：weekend `trial_0082` **0.895751** / focus `trial_0021` **0.894710** / focus `trial_0031` **0.896061**
  ⇒ 族均值 **0.895507** ⇒ **soup 0.895868 = 比族均值高 0.036 pp**）⇒ **这条路线有「把 T1 推过线」的实证先例**。
- **`cs_t0077` 的 T3 0.870514 已过线**，只被 T1 否掉 ⇒ **两条线各自都差一点点** ⇒ 又一次落在 Pareto 拐角上。

⇒ **当前最有希望的下一步**：把跨种子平均用在 **`w3.0·lr3.0·8ep·rc2·bb1·fbb0`** 配方上 ——
它是**唯一「T3 已过线（+0.2645 pp）且 T1 只差 76 个 TP」**的候选（focus `trial_0033`，见 §11.3.2）
⇒ **补 2 个同配方种子（各 ~92 min）→ 3 种子平均 → 门禁**。
（跨种子平均已在 `cs_t0082` 上实证能**把族均值抬 +0.036 pp** ⇒ 若 `trial_0033` 的族均值也接近红线，这一手有机会双线同时过。）

**复现命令（执行机：服务器）**

```bash
# SWA 变体门禁（6 个候选串行，每个 ~10 min）
bash manu/training/gate_swa_variants.sh          # 日志 <OUT>/logs/gate_swa.log；结果 <OUT>/swa/gate/*/gate_report.json
# 把 SWA 结果并进交付链（写 HANDOVER.md 附录 C + SWA_SUMMARY.json）
python3 manu/training/incorporate_swa_results.py --out-root /mnt/data/siping/runs/nogmc_focus
# 实测：n_candidates=6 / n_strict_ok=1 / n_hit=0；产物 <OUT>/SWA_SUMMARY.json

# 跨种子 soup 构建 + 门禁
python3 manu/training/build_crossseed_soups.py --out-root /mnt/data/siping/runs/nogmc_focus
bash manu/training/gate_crossseed_soups.sh        # 日志 <OUT>/logs/gate_crossseed.log；产物 <OUT>/swa/cs_*.pt + <OUT>/swa/gate/cs_*/
```

### 11.5 交付链上今天早上修掉的 **5 个真 bug**（①~③ 交付链 · ④⑤ 口径/汇总）

| # | 位置 | 症状（实测） | 根因 | 修法 |
|:--|:--|:--|:--|:--|
| ① | `manu/training/build_crossseed_soups.py` | **跨种子 soup 一个都建不出来**（构建输出「状态=1 ⇒ 未完成，跳过」× 8 ⇒「完成：无」） | 第 91 行存 `t.state.value`（**数值**）而第 107 行比 `"COMPLETE"`（**字符串**）⇒ **恒为真** ⇒ 已完成臂全被跳过 | **存 `t.state.name`**（修后 3 个 soup 全部构建成功） |
| ② | `manu/training/finalize_nogmc_focus.sh` **⑤b** | **rc = 2** ⇒ **`T3_REACHABILITY.md` 静默停在旧档**（06:33 的备份里 T3_REACHABILITY.md 还是 7,987 B 旧版） | `analyze_t3_reachability.py` 的 `--csv` 是 **`action="append"`**（每个 CSV 一个 `--csv`），而 `judge_handover_candidate.py` / `seal_candidate.py` 的是 **`nargs="+"`**（一个 `--csv` 多个值）⇒ **复用 judge 的参数数组 ⇒ argparse 报 `unrecognized arguments`** | finalize 里**单独建** `T3_CSV_ARGS=(--csv "focus=...")`，再**逐个** `--csv` 追加 |
| ③ | `manu/training/seal_candidate.py` | **重封 rc = 1**（`PermissionError`） | 固化会把封存副本置 **`0444` 只读** ⇒ 第二次固化**同一个 rc_id** 时 `shutil.copy2` 抛 **PermissionError** | 覆盖前先 `os.chmod(sealed, 0o644)` ⇒ **保险层与最终层会封同一个 rc_id ⇒ 必须幂等** |
| ④ | `manu/training/analyze_t3_reachability.py`（**汇率口径**） | 汇率表把 `mix_spec` 排成「最划算的买 T3 杠杆（**−0.128**）」⇒ **方向是反的**（见 §11.3.1） | `rate`（每换 1 单位 T3 付的 T1）**只在 `d_t3 > 0` 的对子子集上取中位**，而同一行的「ΔT1/单位 / ΔT3/单位」是**全部**实质对子的中位 ⇒ **两列不是同一批样本，符号可以互相矛盾**；且 `scale` 的 −0.324 建立在 **0/2** 的空 population 上却被排到第一 | ① 汇率改为在**全部实质对子**上算；② 表格加「**其中 T3 变大的对数**」列让 population 显式；③ 结论改为**按 population 分型**后再排序（**买 T3 型** vs **拿 T3 换 T1 型**）。**服务器 09:02 已按修正口径重生成 `T3_REACHABILITY.md`** |
| ⑤ | `manu/training/incorporate_swa_results.py`（**glob + 幂等**） | **glob 漏真实布局** ⇒ 扫不到 `gate_report.json` ⇒ **`SWA_SUMMARY.json` 根本不写、`HANDOVER.md` 附录 C 空着**；且每次重跑 finalize 会**再追加一份附录 C**（报告里出现两份同节） | 脚本只扫 `<name>/epoch*/gate_report.json`，漏了门禁脚本实际用的 `--out-dir $GD`（报告直接在 `<name>/gate_report.json`）⇒ **两种布局必须都收** | ① glob 同时收**两种布局**；② 追加附录 C 前**先把旧的「## 附录 C.」整段切掉** ⇒ **附录 C 改为幂等** |

> ⚠️ **旧 memory 文档的 `--csv` 写法要更正**：专题档 **§8.13.2 / §8.13.3** 写的
> `--csv "weekend=..." "focus=..."`（**一个 `--csv` 跟多个值**）**只对 judge / seal 成立**；
> **对 `analyze_t3_reachability.py` 是错的**（它是 `action="append"`）⇒ 正确写法见 §11.3 的复现命令。
> 旧文保留原文（过程记录），**更正记于此**。

**复现命令（执行机：服务器；均为只读/沙箱，不改 study）**

```bash
# ① 确认枚举 bug 已修（存的是 .name 不是 .value）
grep -n "t.state.name\|t.state.value" manu/training/build_crossseed_soups.py
# ② 确认 finalize 里 CSV 参数已分流
grep -n "T3_CSV_ARGS\|CSV_ARGS" manu/training/finalize_nogmc_focus.sh
# ③ 确认封存幂等
grep -n "os.chmod(sealed, 0o644)\|os.chmod(sealed, 0o444)" manu/training/seal_candidate.py
# ④ 确认汇率口径已修：rate 与相邻斜率取自同一批对子 + 显式 population 列
grep -n "t3_up\|其中 T3\|rate" manu/training/analyze_t3_reachability.py | head -10
# ⑤ 确认 SWA 汇总脚本同时收两种布局 + 附录 C 幂等
grep -n "glob(\"epoch\*/gate_report.json\")\|gate_report.json\|## 附录 C\." manu/training/incorporate_swa_results.py | head -10
```

### 11.6 冻结产物（交接件清单）

| 产物 | 路径 / 校验 | 备注 |
|:--|:--|:--|
| `Candidate_v1`（保底牌） | `/mnt/data/siping/runs/CANDIDATES/Candidate_v1_trial_0025_epoch6.pt` · md5 **`4302b13de7a338e8b8c3882aac606482`** · 21,301,027 B | 登记 `/mnt/data/siping/runs/CANDIDATES/Candidate_v1.json` |
| `Candidate_v2`（判定候选，**rc=3 ⇒ 未 handover_ready**） | `/mnt/data/siping/runs/CANDIDATES/Candidate_v2/Candidate_v2_trial_0083_epoch8.pt`（= `weekend:trial_0083` epoch8）· md5 **`67e8e65e31a7b4144eb69243726ee84a`** · sha256 **`1bfc05b9e43fe062e35830699fd90be0a4fd8dbe0409acf657b03c8fb092111d`** · 21,301,027 B · 模式 **0444** | 登记 `/mnt/data/siping/runs/CANDIDATES/Candidate_v2/Candidate_v2.json`（`sealed_at_local = 2026-11-03 08:45:35`，`handover_ready = false`，含 15 维配方 + 逐 trial 指标 + 判定链 + 阻塞项） |
| 交付目录 | `/mnt/data/siping/runs/nogmc_focus/`：`HANDOVER.md` / `HANDOVER.json` / `T3_REACHABILITY.md` / `T3_REACHABILITY.json` / `SUMMARY.md` / `SWA_SUMMARY.json` / `AUTORUN_DONE.json` / `optuna_summary.csv` | `SWA_SUMMARY.json` 于 **08:48** 由 `incorporate_swa_results.py` 补齐（**06:33 的保险层里没有它**） |
| 备份 | `/mnt/data/siping/runs/nogmc_focus_backup_20261103_0620`（保险层）· `..._20261103_0633`（终版）· `..._20261103_0848`（08:48 重跑） | 每次含 `HANDOVER.md/.json` + `optuna_summary.csv` + `study.db` + `SUMMARY.md` + `T3_REACHABILITY.md/.json` |
| SWA / 跨种子权重 | `/mnt/data/siping/runs/nogmc_focus/swa/swa_*.pt`（6 个）· `/mnt/data/siping/runs/nogmc_focus/swa/cs_*.pt`（3 个） | 门禁报告在 `swa/gate/<name>/gate_report.json` |
| **`cs_t0008`（本轮唯一新严格 PASS）** | `/mnt/data/siping/runs/nogmc_focus/swa/cs_t0008.pt`（md5 `1964b74f733180178fbadd920c7aca3e`，7,380,920 B，4 种子平均） | 门禁 **PASS**：T1 **0.8959005**（压线过 +0.0004 pp）· T2 F1 0.998619 / FP 42 · T3 0.868572（距 0.87 差 0.1428 pp）⇒ **未进交接判定池（不是 trial，无配方键）**，但**是 soup 路线可行性的实证** |

### 11.7 本节沉淀的**通用结论**

1. **墙钟截止只写 `stop_at` 不够**：必须同时有「**预计跑不完的臂不派**」的硬截止（`hard_cut`）
   **和「调度器被硬截止收尾后不许被看门狗再拉起」** ⇒ 否则就是本轮实测的 **47 次空转 + 45 条垃圾 PRUNED**。
2. **看门狗的「活着吗」判据不够**：必须区分「**崩了**」与「**被硬截止正常收尾**」⇒ 重启前先查是否已过 `stop_at`。
3. **收尾链每个子步骤都要独立 rc 可查**：外层 `rc=0` **不等于**每个产物都是新的（⑤b rc=2 ⇒ `T3_REACHABILITY.md` 停在旧档）⇒ **必须核对每个产物的时间戳**。
4. **封存/固化类步骤必须幂等**（只读副本会挡住覆盖 ⇒ `PermissionError`）。
5. **「评估与训练不得同卡」的可执行自检口径**：**门禁开始时刻必须晚于最后一条训练臂的完成时刻**（本轮 05:32:53 vs 05:33:43 ✅）。
6. **双目标卡在 Pareto 前沿拐角上时，补种子补不动**：0.289 pp 的 T3 缺口比种子噪声中位 0.002995 **大一个量级** ⇒ **该换先验空间，不是该补种子**；
   **但「换先验空间」要按修正后的杠杆分型走**：**买 T3 走 `lr_scale`**，不是 `mix_spec`（见 §11.3.1）。
7. **比率类汇总指标必须与它旁边的聚合斜率取自同一批样本**，否则**符号会互相矛盾**；
   且必须**显式给出 population（多少对支持该方向）**，否则**空 population 的维度会被排到第一**（见 §11.3.1 bug ④）。
8. **「种子平均」是独立于超参搜索的第三条杠杆**：跨种子 soup 已实证能把同配方 T1 抬到「离红线 7 个 TP」（`cs_t0082`）、
   并产出一条**新的严格 PASS**（`cs_t0008`）⇒ **换先验空间之前，先把零训练成本的三条路（SWA / 跨配置 soup / 跨种子 soup）走完**。
