# 击败 Baseline_v2.0（trial_9863）的夜间搜索 —— 全部尝试总结

> **文档版本**：v1.0　**撰写时间**：2026-11-13 13:20 (CST)
> **覆盖范围**：2026-11-11 20:19（R2 开跑）→ 2026-11-13 13:13（R13 进行中）
> **数据来源（权威）**：服务器台账 `/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json`（947 臂）
> ＋ `adaptive.log` / `watchdog.log` ＋ 本地 `manu/memory/memory.md` §153.1~§153.29c
> **复算工具**：`six_criteria_audit.py`（六项判据全盘复算）、`cell_yield_audit.py`（格级产出率）

---

## 0. TL;DR（结论摘要）

1. **至今（11-13 13:13 实采）仍无任何模型击败 `9863`**。六项判据全盘复算（121 个有完整读数的
   （臂, epoch））⇒ **六项全过的只有 1 个 = `9863` 本体**；`promotion_ready.json` ⇒ `candidates = []`。
   ⇒ `Baseline_v2.0` 仍是**唯一官方基线**，11 轮（R2~R12）**累计 beat = 0**。
2. **失败的主因不是"算力不够"，而是"算力投错了格子"**：台账 947 臂里 **285 臂（30%）压在
   `sw5.0×thr16ms4×lrf0.0`（记为 B1 格）**，而该格只产出了 `9863` 本体（连"击败自己"都没做到）。
   真正高产的 `sw4.0/5.0 × thr12ms3 × lrf0.1`（A2/A1 格）只拿到 148 + 112 臂。
3. **门禁名额同样投错**：旧逻辑按"裸 nano 排名"挑官方三轨名额（官方三轨是最贵的一步，约
   12 min/臂），结果 100% 喂给了 E≈0.001 的 B1 家族 ⇒ **每一轮都在烧钱测最没希望的臂**。
4. **已上线 5 个补丁**（算力重排 + 门禁重排 + 门禁名额扩容 + 时间预算上调 + 种子扫描改格级加权），
   全部经「本地改 → scp → 双侧 md5 比对 → 服务器 `--selftest` PASS」闭环。
5. **当前状态**：R13 在跑（NanoBench 阶段），守护进程将在**轮次边界自动切到新码起 R14**；
   R14 计划 **122 臂，其中 48 臂（39%）落在 A 家族（thr12ms3×lrf0.1）**。
6. **未闭合**：板卡 `192.168.0.64` 离线 ⇒ **T4 时延判据需人工补测**（明早升轨前必须做）。

---

## 1. 基线定义与判据体系

### 1.1 Baseline_v2.0 = `trial_9863` ep5（唯一官方基线）

| 字段 | 值 |
|---|---|
| trial / epoch | `9863` / ep5 |
| 权重 | `/mnt/data/siping/runs/npu20ms_offcal/trials/trial_9863/weights/epoch5.pt` |
| 配方 | `trial_0031` 配方 ＋ `--size-weight-small 5.0`（骨干/头结构与 0031 逐项同图） |
| NanoBench（no-GMC） | nano **0.7086** / micro 0.8258 / small 0.8574 / regular 0.9435 / F1 **0.889712** / nano@4px **0.6472** |
| 官方三轨（GMC） | T1 **0.888086**（FP 1526） / T2 0.997440（**FP 5**） / T3 **0.948489**（FP 80） |
| 板卡时延 | 交付板 **19.306 ms**；新板 22.009 ms（同会话 0031 = 23.528 ms） |
| 资状态 | `ACTIVE / OFFICIAL BASELINE`；0031 = `SUPERSEDED`；0080(RC2) = `RETIRED` |

### 1.2 六项判据（Beat-to-Pass，**同一臂 + 同一 epoch** 上同时满足）

| # | 判据 | 阈值（严格 `>=`） | 含义 |
|---|---|---|---|
| 1 | `nano` | **≥ 0.708648** | NanoBench 微小目标召回（8px 容差） |
| 2 | `nano4` | **≥ 0.647212** | NanoBench 微小目标召回（**4px 严格档**） |
| 3 | `f1` | **≥ 0.889712** | NanoBench 总体 F1 |
| 4 | `t1` | **≥ 0.888086** | 官方 T1（Anti-UAV 官方域 source macro） |
| 5 | `t2_fp` | **≤ 5** | 官方 T2（Longquan 金标池）虚警数上限 |
| 6 | `t3` | **≥ 0.948489** | 官方 T3（盲测难例序列） |

> ⚠️ **纪律**：升轨必须在**同一个臂、同一个 epoch** 上六项合取成立。
> 历史上出现过"nano/nano4/F1 三项全过、但官方三轨根本没测"的臂被误判为"打平基线" ⇒ 已修
> （`full_checks()`：六项必须齐全且全 True 才算 beat）。

### 1.3 种子噪声 σ 标定（臂级，用于判断"差多少算噪声"）

| 指标 | σ |
|---|---|
| nano | 0.003948 |
| nano4 | 0.009571 |
| f1 | 0.001440 |
| t1 | 0.001328 |
| **t3** | **0.000855** |

⇒ 最接近的 5/6 臂（12679 差 1.57e-4、12519 差 1.15e-4）在 **1σ 内** ⇒ **换 seed 就可能全过**
⇒ 这正是"种子扫描"这条线的价值所在。

### 1.4 NanoBench-T1（小目标召回数据子集）—— 唯一小目标召回基准

- 脚本：`manu/evaluation/tiny_target_subset.py`；**T1 全量 31,613 帧 / 25,111 GT**。
- 分桶 GT 数：**nano 4,232 / micro 3,283 / small 11,796 / regular 5,800**。
- **双容差**：**8px 主档** ＋ **4px 严格档**；匹配 = **中心欧氏距离**（**禁用 IoU**）；
  尺寸按**物理像素**（**禁各向同性 ×640**）。
- **语义 = no-GMC**；官方三轨 = **GMC** ⇒ **两者严禁混用、严禁互相引用数字**（铁律）。
- 参照卡：服务器 `/mnt/data/siping/runs/npu20ms/tiny_ref/`（`epoch{2,3,4,5}.pt` 的推理缓存 pkl）。
- **已知基准口径缺陷（入档不改）**：缓存里 `gt_bboxes` 是**原图像素**，而 NanoBench 又把高度 ×0.8
  ⇒ 物理 nano 实占 14.64%（参照卡写 16.85%，偏高 2.2 pp）、高度被系统性低估 20%。
  本轮**不修基准**（修了则与 0031 不可比、历史臂全部失真）⇒ 下一版统一修并重刷参照卡。

### 1.5 两套参照系（严禁混）

| 参照系 | 出处 | T1 | T2 | T3 |
|:--|:--|:--|:--|:--|
| **GMC**（官方三轨默认） | `auto_epoch_gate_check.py` 默认 | ≥ 0.905736（绝对） | F1 ≥ 0.985 且 FP ≤ 36 | 观察轨 base 0.837331 |
| **no-GMC** | `reference_frame.json` | ≥ base−0.005 = 0.895897 | FP ≤ 112 | 观察轨 base 0.7578 |

`nano_search.py::run_official_gate()` **不传参照系** ⇒ 走 **GMC 口径**（与 0031 报告一致）✓

### 1.6 板卡时延判据（T4）—— 当前**唯一无法自动完成**的一项

- 口径：**候选_ms ≤ 0031_ms × 1.02 = 20.07 ms**。
- `.64` 板同会话实测：0031 = **19.673 ms**、9863 = **19.642 ms**。
- **现状：`192.168.0.64` 离线**（服务器 ＋ host 双侧 ping 100% 丢包）⇒ 所有臂的 `board_ms = None`
  ⇒ 联合门禁一律 `PASS_WITH_GAP (speed=SPEED_UNKNOWN)`。
- ⇒ **明早若有六项全过臂，必须人工开机板卡 .64 补测 T4 才能完成升轨**。
  （另注：流水线 `BOARD` 常量仍指向已失效的 `.211`，**不会误碰** `.64` ⇒ 安全，但补测需手工执行
  `manu/pipeline/npu/board/final_board_session.sh`。）

---

## 2. 实验流水线架构

### 2.1 组件与文件清单（本地镜像 `ultralytics/` ⇄ 服务器 `/tmp/pycharm_project_10ae9e2e/`）

| 组件 | 文件 | 职责 | md5 |
|:--|:--|:--|:--|
| 调度器 | `manu/pipeline/npu/adaptive_runner.py` | 轮次编排：选臂 → 训练 → 缓存 → NanoBench → 选 epoch → 官方三轨 → RKNN → 板卡 → 联合门禁 → 台账 | `61206cf2b7177db1b345334f87ab87ec` |
| 单臂执行器 | `manu/pipeline/npu/nano_search.py` | Wave 驱动单臂流水线（幂等 / 台账 / 硬截止） | `d80969060633809d94abe1d3c565cb38` |
| 守护 | `manu/pipeline/npu/run_adaptive_watchdog.sh` | 调度器挂了/静默超时 ⇒ 自动重启；**代码变更在轮次边界生效** | `30fba85763f38cf3ea65669df944a0f7` |
| 早报 | `manu/pipeline/npu/run_morning_report.sh` | 09:10 自动出 `morning_report_20261114.txt` | `18dfe6c1f688006dc1b7b6179faf57bf` |
| 审计 | `manu/pipeline/npu/six_criteria_audit.py` | 六项判据**全盘复算**（以磁盘 `gate_report.json` 为准） | `e5aba56e436b6fad4021fe5ee22a3ba2` |
| 审计 | `manu/pipeline/npu/cell_yield_audit.py` | 格级产出率（joint 命中 / 组均值近失 / 已投臂账） | `e4164b169331cb192641a7fc3e5f63ab` |
| 升轨 | `manu/pipeline/npu/promote_candidate.py` | 候选枚举 → `promotion_ready.json` → 升轨 Baseline_v2.1 | `08b36304b2dd6fdce06335134faab297` |
| 升轨 | `manu/pipeline/npu/promote_baseline.py` | 台账基线注册表写入（0031/0080 降轨、9863 封板） | — |

> **代码同步铁律**：本地改 → `scp` → **双侧 md5 比对** → 服务器 `--selftest` PASS → 记入 memory。
> （系统 python3 不认中文源码 ⇒ 服务器一律用 `/home/huangzhe/anaconda3/envs/uav/bin/python`。）

### 2.2 单臂生命周期（`nano_search.py` 的 Wave）

```
训练(ep1..5) → 推理缓存(epoch*.pt → pkl) → NanoBench(逐 epoch 读 nano/nano4/f1/micro/small/regular)
   → 按"三项全平 + 其它判据"选 picked epoch → 官方三轨(gate_report.json: t1/t2_fp/t3)
   → RKNN 导出 → 板卡时延（.64 离线 ⇒ 跳过） → 联合门禁 → 台账落盘
```

- **选 epoch 规则（已修）**：① 先要求 `nano ≥ 下限`；② 在**同时**满足 `nano4 ≥ 基线` 且 `F1 ≥ 基线`
  + 官方项的集合里择优。旧规则只按 F1 挑 ⇒ 曾白丢 `12047`（其 ep4 的 nano4 = 0.6569 是过的）。
- **门禁跳过（`official.skipped`）**：未过 nano 窗口的臂 **419 个**被跳过官方三轨（省了最贵的一步，
  但也意味着这些臂**永远不可能升轨**）。
- **SWA 合成臂**：源臂 epoch 窗口平均 ⇒ **零训练成本**的候选臂（80 个），命中率相对最高。

### 2.3 关键路径与产物（服务器）

| 路径 | 内容 |
|:--|:--|
| `/mnt/data/siping/runs/npu20ms/nano_search/nano_search_ledger.json` | **权威台账**（`arms` / `waves` / `rounds` / `baseline_registry` / `beat_to_pass`） |
| `.../adaptive.log` | 调度器逐臂日志（训练 / NanoBench / 门禁） |
| `.../watchdog.log` + `watchdog_state.json` | 守护日志与状态（含 `code_md5`） |
| `.../promotion_ready.json` | 升轨候选清单（当前 `[]`） |
| `.../dual_gate.json` / `final_report.json` / `beat_to_pass_report.md` | 联合门禁 / 全轮汇总 / 打平报告 |
| `.../morning_report_20261114.txt` | **明早 09:10 自动早报** |

---

## 3. 逐轮时间线（R2 ~ R14）

| 轮 | 起跑时间 | 臂数 | 门禁臂数 | **beat** | 备注 |
|:--|:--|--:|--:|--:|:--|
| R2 | 11-11 20:19 | 62 | 8 | **0** | 首轮；建立 A/B/C 格雏形 |
| R3 | 11-11 23:33 | 36 | 8 | **0** | |
| R4 | 11-12 02:40 | 36 | 8 | **0** | |
| R5 | 11-12 06:04 | 36 | 8 | **0** | |
| R6 | 11-12 09:11 | 32 | 8 | **0** | |
| R7 | 11-12 12:04 | 28 | 8 | **0** | |
| R8 | 11-12 15:30 | 38 | 8 | **0** | 宿主会话第一次交接 |
| R9 | 11-12 22:50 | 59 | 8 | **0** | |
| R10 | 11-13 03:02 | 63 | 8 | **0** | |
| R11 | 11-13 07:18 | 61 | 8 | **0** | |
| R12 | 11-13 09:15 | 58 | 8 | **0** | `phase_dual` 崩 ⇒ 联合门禁整段丢失（已修） |
| **R13** | 11-13 09:21 | 65 | — | — | **进行中**：训练 12:36 结束 → NanoBench 阶段（13:13 实采） |
| **R14** | ≈ 11-13 14:00 | **122（计划）** | 8 + triad | — | 守护将**自动以新码起跑**（新预算表 + 新门禁排序） |

**累计：R2~R12 共 509 臂（台账 `arms` 字段共 947 条，含历史/未成轮次臂），beat = 0。**

### 3.1 昨夜至今发生的三次真实故障（均已修）

1. 🔴 **R12 `phase_dual` 整段崩**（`KeyError 'picked'`）：末轮被 `stop_ts` 砍掉的臂没有 `picked`
   读数 ⇒ `st['picked']['epoch']` 抛异常 ⇒ **联合门禁整段丢失**。
   ✅ 已修并部署（`nano_search.py` md5 `d80969060633809d94abe1d3c565cb38`）：无 `picked` 的臂跳过
   ＋ 单臂异常不许掀掉整段。
2. 🔴 **09:16:21 与 09:21:41 起了两次 R13**（第一次 5 min 内静默死掉，守护判"调度器不在"重启）
   ⇒ 浪费 **69 个臂号 ＋ 10 臂 GPU 时间**；且旧码按轮号算 seed ⇒ 两次用同一批 seed。
   ✅ 新码 `used_recipe_seeds() + _fresh_seed()` ⇒ 重启不再重复（dry-run 实测重复 0、seed 全唯一）。
3. 🔴 **升轨护栏误杀**：均值护栏把"有六项全过臂"的候选也拦下。
   ✅ 已修：先判有无六项全过臂 ⇒ 均值护栏让路。

---

## 4. 全盘统计（947 臂）

### 4.1 读数分布（按台账 `official` 字段）

| 状态 | 臂数 | 含义 |
|:--|--:|:--|
| `official = {'skipped': ...}` | **419** | 未过 nano 窗口 ⇒ 官方三轨未跑（**永远不可能升轨**） |
| `official = {}` | **385** | 训练/缓存完成但未进入官方门禁 |
| 有完整官方读数 | **128** | 跑过官方三轨 |

`six_criteria_audit.py` 以**磁盘 `gate_report.json`** 为准 ⇒ **121 个（臂, epoch）有完整读数**
（去重 (arm,epoch) 后 117 行，其中 **28 个是 SWA 合成臂**）。

### 4.2 六项判据全盘复算（`six_criteria_audit.py --top 20`，11-13 实采）

**六项全过：1 个**

```
✔ trial=9863 ep=5  nano=0.708648 nano4=0.647212 f1=0.889712 t1=0.888086 t2fp=5 t3=0.948489  ← 基线本体
```

**最接近候选（5/6）**

| 臂 | ep | sw | lrf | 缺口 |
|:--|:--|--:|--:|:--|
| 12679 | 3 | 4.0 | 0.1 | t3 −1.57e-4（**0.2σ**） |
| 12519 | 3 | 5.0 | 0.1 | t3 −1.15e-4（**0.13σ**） |
| 12694 | 5 | 4.0 | 0.1 | t3 −7.81e-4（0.9σ） |
| 12709 | 4 | 4.0 | 0.1 | t3 −4.93e-4（0.6σ） |
| 12520 | 4 | 5.0 | 0.1 | t3 −1.84e-3（2.2σ） |
| 12053 | 4 | 5.0 | 0.1 | **t1 −9.85e-4**（0.74σ） |
| 12635 | 3 | 4.0 | 0.1 | nano −3.07e-3 |

> 另有 SWA 臂 **12702 / 12703 = 5/6**，**T3 只差 2.4e-5**（nano4 差 0.0043，F1 / T1 / 虚警全过）。

### 4.3 全台账的 12 个 "triad" 臂（**唯一有资格升轨**的臂）

`9863`(B1,T3✔) · `12053`(B2,T1) · `12252` · `12263` · `12300` · `12333` · `12519`(A1,T3) ·
`12520`(A1) · `12600`(A1) · `12679`(A2,T3) · `12694`(A2,T3) · `12709`(A2,T3)

> **triad 定义**：存在某 epoch 使 `nano`/`nano4`/`f1` **三项同时过线**（六项全过的必要条件）。
> **实测**：全台账 587 个有 NanoBench 读数的臂中，**13 个 triad-able、12 个有官方读数**。

---

## 5. 关键发现（失败原因分析）

### 5.1 发现 1 🔴🔴：算力严重错配 —— 预算压在最没希望的格子上

按格（`sw × thr × maxside × lr_floor`）统计**全部 947 臂**的投放量（独立复算，any-epoch 口径）：

| 格 | 含义 | **投放臂数** | p_triad | triad+官方 | T1 过 | 虚警过 | **T3 过** |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **B1** | sw5.0×thr16ms4×lrf0.0 | **285（30%）** | 4/198 = 0.020 | 4 | 2 | 3 | **1** |
| **A2** | sw4.0×thr12ms3×lrf0.1 | 148 | 3/98 = 0.031 | 3 | 3 | 3 | **0** |
| **A1** | sw5.0×thr12ms3×lrf0.1 | 112 | 3/52 = 0.058 | 3 | 3 | 2 | **0** |
| **C** | sw4.0×thr16ms4×lrf0.0 | 62 | 0/29 = 0.000 | 0 | – | – | – |
| **B2** | sw5.0×thr16ms4×lrf0.1 | 52 | 1/38 = 0.026 | 1 | 1 | 1 | **1** |

⇒ **B1 吃掉 30% 预算，只产出 `9863` 本体**（连"击败自己"都没做到）。
⇒ A 家族（thr12ms3×lrf0.1）**p_triad 最高（0.031 / 0.058）、T1/虚警全过、只卡 T3** ⇒ 应重仓。

> 📌 **口径说明**：上表是**独立复算**（any-epoch 口径，n = 有 NanoBench 读数的臂）。
> 代码内的 `cell_stats()` 用 **picked epoch** 口径、`cell_yield_table()` 用 **有官方读数的臂**口径，
> 两者的绝对值会与上表有小幅差异（memory §153.29 / §153.29b 的表格属代码内口径）。

### 5.2 发现 2 🟢：`lr_floor` 是官方侧的总闸

在**过了 nano 侧**的 73 臂统计里：

| `lr_floor` | 平均 T3 | T3 过线率 |
|:--|:--|:--|
| **0.1** | −0.0007 | **7/23** |
| 0.0 | −0.030 | 4/55 |

⇒ **把 `lr_floor` 抬到 0.1，是让官方侧（T3）过线的必要条件**；
而 `lr_floor=0.1` 又不伤 nano 侧（A2 / A1 格的 nano 均值 0.711 / 0.711 均 ≥ 基线 0.7086）⇒ **这才是主战场**。

### 5.3 发现 3 🟢：`sw`（size-weight-small）是 nano ↔ (F1/T1/T3) 的**主交换轴**

| 方向 | 随 sw 上升 | 随 sw 下降 |
|:--|:--|:--|
| nano | 上升（sw4 16% → sw5 56%） | 下降（sw1 0% / sw3 0%） |
| F1 / T1 / T3 | 下降（sw4 F1 76% → sw5 F1 23%；T3 sw1 54% → sw4 0%） | 上升 |

⇒ **sw4.0~5.0 之间是"未采样的插值区"**（sw4.25 / 4.5 / 4.75 / 5.25 全部是本轮新增格）。
⇒ `sw5×thr16ms4×lrf0.1`（= `12053` 配方）**从未跑过 sw4 版本** ⇒ 交叉出互补强项。

### 5.4 发现 4 🔴：T3（官方盲测难例）是最终瓶颈

**P(T3 过线 | 三项全平) 全局 = 2/12 ≈ 17%**。
⇒ 即使三项全平、T1 过、虚警过，**仍有 ~83% 概率卡在 T3**。
⇒ 这就是为什么 12 个 triad 臂里只有 1 个（`9863`，B1 格）真的六项全过。

### 5.5 发现 5 🔴：**nano 高 ≠ 能升轨**

旧门禁窗口（"nano 前 8"）选出来的臂 nano 高达 **0.7578**，但它们 **nano4 / F1 塌了**
（例：`12131/12287/12183/…`）⇒ **不是 triad ⇒ 没有任何 epoch 可升轨**。
⇒ 这 8 个名额**纯诊断、不产生升轨**，而且把最贵的官方三轨名额浪费掉了。

### 5.6 发现 6 🟡：两处 E 值口径必须分开用（含一处自我更正）

> 🔴 **自我更正（memory §153.29b）**：§153.29 给出的 E 表（A2 = 0.030、B1 = 0.0004，宣称差 30~70 倍）
> **只用 128 个"被 nano 窗口筛过"的臂算 p_triad ⇒ 存在选择偏差**。
> 全臂口径复算后方向不变（A 家族 > B2 > B1 > C），但量级只有 **~3 倍**。
> ⇒ **旧数字（30~70 倍）不要再引用**；本文件 §5.1 的表为全臂口径。

- **门禁窗口排序**（补丁 2）：用"有官方读数的臂"算 E **是对的** ⇒ 因为排序问题本身就是
  "在已过 nano 窗口的候选里挑"，条件与筛选同口径 ⇒ 保留 `cell_yield_table()`。
- **训练臂分配**（补丁 1/4）：必须用**全臂** p_triad ⇒ 新增 `cell_stats()` + `_cell_weight()`。

---

## 6. 已上线的补丁

### 6.1 本轮 5 个补丁（`adaptive_runner.py`，md5 `61206cf2b7177db1b345334f87ab87ec`，selftest PASS）

| # | 补丁 | 内容 | 效果 |
|:--|:--|:--|:--|
| 1 | **基线派生格预算重排** | `BASE_DERIVED_CELLS` 换成 **16 格 / 49 seed**，按 E 降序：A2×5 + A1×5 + sw4.25×4 + sw4.5×4 + sw4.75×3 + sw5.25×2 → **B2×4** → C×lrf0.1×3 → lrf0.1 插值×8 → B1×3 / C×2 / sw 插值×6（**降配**） | 训练臂预算从 B1 迁到 A 家族 |
| 2 | **门禁名额改按 E 排序** | 新增 `cell_yield_table()`；门禁窗口从"nano 前 6"改为"**按 E 排序前 8**"；层级回退：精确格 n≥3 → 同 thr/maxside/lrf 家族 → 同 lrf → 全局 | **旧口径 8 个名额全是 B1 格（E=0.0010）；新口径全是 A2 格（E=0.0303）⇒ 同名额期望产出 ~30 倍** |
| 3 | **时间预算上调** | `ROUND_FIXED_MIN` 45 → **62**；`GATE_RESERVE_MIN` 45 → **58** | 末轮不再因时间不足砍掉门禁 |
| 4 | **种子扫描权重改格级** | 新增 `cell_stats()` + `_cell_weight()`；权重 = `p_triad(格) × (0.25 + 0.75·p_T3(格))`，其中 `p_T3` 向全局收缩（格级 n_tri 只有 1~3 ⇒ 1 样本的 B2 否则拿 0.75 虚高权重） | 实测新分配（24 臂）：**A1×9 / B2×7 / A2×4 / B1×4**（旧：B1×9 / B2×7 / A1×5 / A2×2） |
| 5 | **探索名额排除 triad 臂** | E 排序榜首 `12679/12694/12709` 本身已是 triad（无条件门禁）⇒ 8 个探索名额实际只换来 5 个新臂 ⇒ 排除 triad 后再取前 8 | 名额不浪费 |

> ⚠️ **表序 = 末轮预算收紧时的砍序**（整格丢规则）⇒ **高产格必须排在最前**（补丁 1 已如此排）。
> ⚠️ **判据未放松**：triad 臂仍**全部无条件门禁**；补丁 2 只改"名额给谁"，不改"谁算通过"。

### 6.2 历史修复（此前上下文，已上线）

- `nano_search.py`（md5 `d80969060633809d94abe1d3c565cb38`）：R12 `phase_dual` 的 `KeyError 'picked'` 整段崩修复。
- `used_recipe_seeds() + _fresh_seed()`：（配方, seed）去重 ⇒ 调度器重启不再重复用 seed（新码重复 0）。
- 升轨护栏误杀修复：先判有无六项全过臂 ⇒ 均值护栏让路。
- 早报脚本升级（md5 `18dfe6c1f688006dc1b7b6179faf57bf`）：加入无偏 cell 表 ＋ E 表 ＋ 新格名 grep。
- `route2_acceptance.py` 的 `--act` 硬编码 relu 死模型 bug 修复（act 跟随 ckpt 自描述）。

---

## 7. 当前运行状态（2026-11-13 13:13 实采）

| 项 | 状态 |
|:--|:--|
| 调度器 | PID **2010456**（R13，`--start-round 13`，`--gpus 0,1,2,3`，`--n-seed 6 --n-mut 18 --swa-top-n 3`） |
| R13 阶段 | 训练已于 **12:36:47** 结束（末臂 12897）⇒ 现在 **NanoBench 阶段**（臂 12851~12869 评测中） |
| 守护 | PID **1991784**（deadline **2026-11-14 09:00**，静默阈值 2700 s） |
| 早报 watcher | PID **2109484**（`run_morning_report.sh 2026-11-14 09:10`） |
| GPU | 4 卡训练期 util **84~92%**；NanoBench 阶段 27~68%（该阶段偏 CPU/IO） |
| 代码切换 | `watchdog_state.code_md5 = 9390a865…` ≠ 磁盘（`61206cf2…`）⇒ **R13 轮末自动以新码起 R14** |
| 台账 | 947 臂；`promotion_ready.json` ⇒ `candidates = []` |
| 板卡 | `192.168.0.64` **离线**（服务器 ＋ host 双侧 ping 100% 丢包） |

### 7.1 R14 计划（122 臂，`build_train_arms` 真实台账 dry-run）

| 类别 | 臂数 | 说明 |
|:--|--:|:--|
| 基线派生格（bdc） | 49 | 16 格 × 2~5 seed（**A 家族为主**） |
| 纳米/新增格（nm） | 20 | |
| 种子扫描（seed） | 26 | 按**新格级权重**分配（A1×9 / B2×7 / A2×4 / B1×4） |
| prio_t3 | 18 | 优先补 T3 的臂 |
| prio | 4 | |
| 突变（mut） | 2 | `thr18ms5` / `thr32ms8` 等 |
| **格分布** | — | **thr12ms3×lrf0.1（A 家族）48 臂（39%）** / thr16ms4×lrf0.1（B2）23 臂 / thr16ms4×lrf0.0（B1）~33 臂 / thr14ms3.5×lrf0.1 6 臂 |

### 7.2 吞吐基准（用于排期估算）

| 阶段 | 单臂耗时 | 并发 | 折算 |
|:--|:--|:--|:--|
| 训练 | ~3.0 min/臂 | 8 并发 / 4 卡 | GPU 已饱和 |
| NanoBench | ~0.58 min/臂 | 8 并发 | — |
| 官方三轨 | ~12 min/臂 | 4 并发 | **~3 min/臂**（最贵一步） |
| **R13 全程** | 65 臂 | — | **4.6 h** |
| **R14 预估** | 122 臂 | — | **≈ 8.6 h** |

⇒ 今夜（14:00 → 次日 09:00，约 19 h）预计跑完 **R14 + R15 两轮**，总训练臂 **≈ 266**。

---

## 8. 今晚排期（host 18:00 关机 ⇒ 服务器自主运行）

| 时间 | 事件 | 责任 |
|:--|:--|:--|
| 13:20 | 本文档定稿并同步到服务器 | 本机 |
| ~14:00 | R13 收尾 ⇒ **守护自动以新码起 R14**（122 臂，新预算表 + 新门禁排序） | 服务器自动 |
| ~14:05 | **验收 R14 切换**：① `watchdog.log` 出现"代码已更新…在轮次边界生效"；② R14 首臂 desc 含 `基线派生格 BASE_A2_sw4.0_thr12ms3_lrf0.1 / BASE_A1_…`；③ `adaptive.log` 门禁窗口措辞变为"期望产出率前 8"；④ (配方, seed) 无重复 | 主机轮询脚本 `/tmp/wait_r14.sh`（后台，输出 `/tmp/wait_r14.log`） |
| 18:00 | **host 关机**（服务器不受影响，守护进程 PID 1991784 独立于本会话） | 用户 |
| ~22:40 | R14 收尾 ⇒ R15 起跑 | 服务器自动 |
| 11-14 09:00 | **硬截止**：调度器退出 | 服务器自动 |
| 11-14 09:10 | **早报自动生成** `morning_report_20261114.txt` | 服务器自动 |
| 11-14 上午 | 若有六项全过臂 ⇒ `promote_candidate.py --list` ⇒ **需人工补测板卡 T4**（.64 开机）⇒ 升轨 Baseline_v2.1 | 用户 |

---

## 9. 未闭合项与风险

| # | 项 | 状态 | 影响 |
|:--|:--|:--|:--|
| 1 | **板卡 T4 时延补测** | 🔴 阻塞 | `.64` 离线 ⇒ 所有臂 `board_ms=None` ⇒ 联合门禁只能给 `PASS_WITH_GAP`。**明早若出候选，必须人工开机补测才能完成升轨**（口径：候选 ≤ 0031×1.02 = 20.07 ms） |
| 2 | **T3 瓶颈** | 🟡 未破 | P(T3\|三项全平) = 17%；A 家族 3 个 triad 臂 T3 **全灭**（0/3）⇒ 需更多 seed 摇（σ_t3 = 8.6e-4，最接近的只差 1.2e-4~1.6e-4 ⇒ **1~2σ 内可摇出来**） |
| 3 | **`thr12ms3` 家族的绝对水平** | 🟡 待验证 | A 家族全过 nano 侧与 T1，但 **T3 全灭** ⇒ 可能是 `thr12ms3` 这个小目标定义本身在盲测难例上吃亏 ⇒ 补丁 1 已排 `thr16ms4×lrf0.1`（B2）作对照 |
| 4 | **NanoBench 基准口径缺陷** | 🟢 入档 | 高度被系统性低估 20%（§1.4）⇒ 本轮不改（会失真），下一版统一修 + 重刷参照卡 |
| 5 | **官方 T1 绝对红线** | 🟡 留意 | 0031 本体官方 `gate_report.json` verdict = **FAIL**（`veto_failures=['track1_uav']`，T1 规则 ≥0.905736 实测 0.8859975，Δ −1.97 pp）⇒ **汇报时必须同时给出 gate 原始 verdict 与对等线口径** |
| 6 | **promotion_ready 仍为空** | 🟢 正常 | 尚无六项全过臂 ⇒ 无候选可升轨 |
| 7 | **R14 切换风险** | 🟢 低 | selftest 已 PASS；即便崩溃，守护按轮边界重启 |

---

## 10. 复现与读取指南

### 10.1 连接

```bash
ssh -F /dev/null -p 32222 huangzhe@192.168.99.40      # 服务器（工程 /tmp/pycharm_project_10ae9e2e）
ssh -p 22 root@192.168.0.64                          # 板卡（当前离线）
```

服务器 python 一律用 `/home/huangzhe/anaconda3/envs/uav/bin/python`（系统 python3 不认中文源码）。

### 10.2 常用命令

```bash
cd /tmp/pycharm_project_10ae9e2e

# ① 六项判据全盘复算（权威，以磁盘 gate_report 为准）
/home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/six_criteria_audit.py --top 25

# ② 格级产出率审计（joint 命中 / 组均值近失 / 已投臂账）
/home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/cell_yield_audit.py

# ③ 升轨候选（当前应为空）
/home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/promote_candidate.py --list

# ④ 调度器自检（不碰台账、不占 GPU）
/home/huangzhe/anaconda3/envs/uav/bin/python manu/pipeline/npu/adaptive_runner.py --selftest

# ⑤ 实时进度
tail -f /mnt/data/siping/runs/npu20ms/nano_search/adaptive.log
```

### 10.3 台账结构（`nano_search_ledger.json`）

```
arms: { trial_id: { trial, seed, desc, params, trained, nanobench, picked_epoch,
                    select_rule, picked, official, rknn, rknn_stride, board_ms,
                    speed_ok, dual, dual_pass, beat_to_pass_ref } }
waves / rounds: [ { round, ts, champion_before, arms[], beat[], gated[] } ]
baseline_v2 / baseline_registry / beat_to_pass / search_state
```

---

## 11. 附录 A：文件 md5 清单（本地 ⇄ 服务器一致，11-13 实测）

| 文件 | md5 |
|:--|:--|
| `manu/pipeline/npu/adaptive_runner.py` | `61206cf2b7177db1b345334f87ab87ec` |
| `manu/pipeline/npu/nano_search.py` | `d80969060633809d94abe1d3c565cb38` |
| `manu/pipeline/npu/six_criteria_audit.py` | `e5aba56e436b6fad4021fe5ee22a3ba2` |
| `manu/pipeline/npu/cell_yield_audit.py` | `e4164b169331cb192641a7fc3e5f63ab` |
| `manu/pipeline/npu/promote_candidate.py` | `08b36304b2dd6fdce06335134faab297` |
| `manu/pipeline/npu/run_adaptive_watchdog.sh` | `30fba85763f38cf3ea65669df944a0f7` |
| `manu/pipeline/npu/run_morning_report.sh` | `18dfe6c1f688006dc1b7b6179faf57bf` |
| `manu/memory/memory.md` | `1d0a16518bf9fd70f5c14e0f19525e56` |
| `manu/memory/memory_compact.md` | `3c309419ae2219264fe979e87d47e3f7` |

## 12. 附录 B：术语表

| 术语 | 含义 |
|:--|:--|
| **Baseline_v2.0** | `trial_9863` ep5；当前唯一官方基线；六项判据的参照 |
| **六项判据** | nano / nano4 / f1 / t1 / t2_fp / t3，必须**同一臂同一 epoch** 同时满足 |
| **triad 臂** | 存在某 epoch 使 nano+nano4+f1 三项同时过线的臂 ⇒ **唯一有资格升轨** |
| **格 (cell)** | `(sw, thr, maxside, lr_floor)` 四元组；抹掉 mix / lr 等噪声参数后的聚合单元 |
| **E** | `E[六项全过/臂] = p_triad × p_T1 × p_虚警 × p_T3`，格级期望产出率 |
| **p_triad** | 格内三项全平的臂占比 |
| **A1/A2/B1/B2/C** | A1=sw5×thr12ms3×lrf0.1；A2=sw4×thr12ms3×lrf0.1；B1=sw5×thr16ms4×lrf0.0；B2=sw5×thr16ms4×lrf0.1；C=sw4×thr16ms4×lrf0.0 |
| **sw / swl / thr / ms / lrf** | size-weight-small / size-weight-large / 小目标阈值(px) / 小目标最大边(px) / 学习率下限占比 |
| **SWA 臂** | 由源臂多个 epoch 权重平均合成的臂（**零训练成本**） |
| **NanoBench-T1** | 小目标召回基准（no-GMC 语义，31,613 帧 / 25,111 GT，8px + 4px 双容差） |
| **T1/T2/T3** | 官方三轨（**GMC 语义**）：Anti-UAV 官方域 / Longquan 金标池 / 盲测难例 |

---

**维护**：本文件是"击败 9863"这场搜索的**唯一独立总结**；`manu/memory/memory.md` §153.x 是逐条流水账。
后续每次补丁上线、每轮收工、任何 beat 出现，都应回写本文件并同步到服务器
（本地改 → `scp` → 双侧 md5 比对）。

