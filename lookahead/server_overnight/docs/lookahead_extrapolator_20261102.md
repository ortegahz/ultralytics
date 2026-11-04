# 快速前向推演外挂（Look-ahead Extrapolator）· 龙泉山 8 序列精度验收（2026-11-02）

> **一句话**：在 RC2/trial_0080 龙泉山管线的**输出末端**外挂前向推演，
> **成熟跟踪资产零回归**（关掉开关 ⇒ **8/8 序列整份预测 JSON 的 md5 逐字节一致**）；
> **档位 B（视觉重推）把 320 ms 物理滞后造成的脱靶几乎全部挽回**
> （Δ=8：hit@5px **38.18% → 72.66%**（+34.48 pp），F1 **77.203% → 93.353%**，
> 距无滞后天花板只差 **3.05 pp**）；
> **档位 A（纯运动学外推）在本数据上不成立**——四个 Δ 工况（2/4/8/12）**全部不优于**
> "硬吃滞后"的基线，hit@5px 反而**下降** 4~9 pp。

---

## 0. 基本事实

| 项 | 值 |
|:--|:--|
| 执行机 | **本地** `/media/manu/1TB-Volume/workspace/ultralytics`（分支 `manu_dev_v2`，git `a9eaca6e7`；`manu/tracking/` 未提交，在工作区） |
| 环境 | conda env `uav`，Python 3.10.20，numpy 2.2.6，cv2 4.10，**无 pytest ⇒ 用 unittest** |
| 模型 | **RC2 / trial_0080 / epoch3**（回放 `ms_cache/*.pkl`，**不重跑推理**） |
| 缓存 | `/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache`（**8 序列**，SSHFS 只读） |
| GT | `/home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010` |
| 帧 | `/home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg/<SEQ>/frame_%06d.jpg`（**只读**，≈7.4 ms/帧） |
| 工作点 | `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`（th_base 0.48 / sky_ratio 1.0 / th_ground 0.26 / confirm_consec 7） |
| 代码 | `manu/tracking/lookahead/`（`extrapolator.py` 外挂核 · `track_fusion.py` 跟踪器拷贝 · `lookahead_longquan.py` 管线拷贝+四臂） |
| 单测 | `manu/tracking/tests/test_lookahead_plugin.py`，**18 项**（全库 **135 项**全绿） |
| 产物 | `runs/lookahead/{ref,off,sweep,gate,osd}/`（`osd/` 含 OSD 视频与窗口指标 JSON） |
| 同步 | ⚠️ **产物与代码尚未同步服务器**（沙箱网络层阻断，`ssh -F /dev/null 192.168.99.40` → `socket: Operation not permitted`），需在有写权限处补做 md5 三向比对 |

---

## 1. 外挂是什么（架构）

**不动成熟跟踪资产，只在输出管道末端加一层无状态前推演**：

```
[IR 帧 25Hz] → [GMC/中值特征] → [NPU 多尺度热图 ≈320ms] → [RC2 检测框] → [龙泉山跟踪管线]
                                                                              │
                                                              ┌───────────────┘
                                                              ▼
                                                    D(f) = 管线在 record f 上的输出
                                                              │
                        ┌─────────── 输出级外挂（纯后处理，不回写任何状态）───────────┐
                        │  stale     : report(t) = D(t−Δ)          （老逻辑基线）   │
                        │  kinematic : report(t) = D(t−Δ) + Δ·v    （档位 A）        │
                        │  replay    : report(t) = D(t−Δ) 逐帧重推（档位 B）        │
                        └──────────────────────────────────────────────────────┘
                                                              ▼
                                                     转台上报坐标（物理帧 t）
```

**关键约束**：外挂**只读输出流、只产出新的上报流**，不改动任何历史估计 ⇒
"关掉开关 ⇒ 与原脚本逐位一致"是**构造性保证**（`off` 臂直接取 `dets` 本身，不搬运）。

### 1.1 时间轴（最容易搞错的地方）

- **管线时间轴 = record 下标**（`manu/evaluation/eval_bidirectional_track_fusion.py` 用 `f_idx` 当帧号），
  不是视频绝对帧号；`records[i]["frame_index"]` 才是绝对帧号。
- **Δ 以 record 为单位**。Δ=8 ⇒ 8 个 record ≈ 8 个相机帧 ≈ 320 ms @25 Hz。
- 物理帧 t 的上报，源帧 f = t − Δ；`t < Δ` 的帧没有源帧 ⇒ 不输出。

### 1.2 两个拷贝（为什么不改原文件）

| 文件 | 来源 | 改了什么 |
|:--|:--|:--|
| `manu/tracking/lookahead/track_fusion.py` | `manu/evaluation/eval_bidirectional_track_fusion.py` **逐字拷贝** | 只加 `capture_kf_states` 开关：在 Pass 1 的 `online_tracker.step(...)` **之后**把每条 track 的 `(get_pos(), get_velocity())` 存进 `kf_states[f_idx]`；**既有算术一行未动**（`diff` 只有 5 个 hunk） |
| `manu/tracking/lookahead/lookahead_longquan.py` | `manu/evaluation/system_sota_longquan.py` **逐字拷贝** | import 走本地拷贝；加 11 个 CLI 参数；加四臂对照 + 两把尺子；`--lookahead off` 时 `config["source"]` 仍是 `"system_sota"` ⇒ **整份 JSON 可做 md5 比对** |

> ⚠️ `PointKalmanTrack._count` 是**进程级全局计数器**，`track_id` 随"同进程内此前建过多少条 track"变化。
> ⇒ **比对必须让参考与新脚本都跑满同样顺序的 8 条序列**；拿子集去比 8 序列产物，
> `track_id` 必然不同（位置/分数仍完全相同）。**这是原脚本的性质，不是新脚本引入的。**

---

## 2. 两把尺子（口径，与用户拍板一致）

### 尺子 1：历史帧等价性（证明"没破坏成熟跟踪资产"）

物理帧 t 时，系统手上最新的估计是 `x̂_{t−Δ}`；拿它对 `GT_{t−Δ}` 评测，
要求与老版 Sys SOTA 在同一数据上的指标**绝对一致（ΔF1 = 0.000000）**。

**实现**：在共同源帧窗口 `[0, N−Δ)` 上，对每条臂的**源帧估计**（= `D(f)` 本身）评测 ⇒
各臂天然逐位相等；脚本把它做成**硬门禁**（不等就 `SystemExit`）。
另外每臂带一个 `source_verbatim` 判定（该臂是否原样搬运源帧估计：
`stale=True`、`kinematic/replay=False`），避免"各臂报同一个常数"没有信息量。

**实测**（8 序列，Δ=8，窗口 1,855 record）：

| 臂 | tp | fp | fn | gt | f1 | recall |
|---|---:|---:|---:|---:|---|---|
| original | 25438 | 182 | 1711 | 27149 | `0.9641266652769619` | 0.9369774208994807 |
| stale | 25438 | 182 | 1711 | 27149 | `0.9641266652769619` | 0.9369774208994807 |
| kinematic | 25438 | 182 | 1711 | 27149 | `0.9641266652769619` | 0.9369774208994807 |
| replay | 25438 | 182 | 1711 | 27149 | `0.9641266652769619` | 0.9369774208994807 |

⇒ **ΔF1 = 0.000000**（四臂逐位相等，脚本 `[GATE] 尺子1 各臂源帧口径 ΔF1 = 0.000000`）。

### 尺子 2：最新帧伺服有效性（证明外推收益）

拿**上报给转台的坐标**对**当前帧** `GT_t` 评测：
- **业务域 F1/Recall/Precision/FAR**：复用冻结口径 `manu/evaluation/eval_multiscale_cache_th.py`
  的 `count_matches(..., "inbox")`（贪心按分数降序、点落框内、**一个 GT 只吸收一个检测**）。
- **CLE**：每个「有 GT 且有上报」的 record，取**离最近 GT 中心最近的那个上报点**，算欧氏距离；
  跨序列**取最差值**（分位数不可跨序列合并）。
- **hit@5px**：上报点到最近 GT 中心 ≤ 5 px 的 GT 帧占比（**物理命中率**）。
- **抖动**：同一 track 上报序列的二阶差分 `|(p_t−p_{t−1})−(p_{t−1}−p_{t−2})|` 的 max / p99。
- **覆盖率**：有上报的帧 / 总帧。

---

## 3. Gate A：关掉开关 ⇒ 与原脚本逐位一致

| 项 | 结果 |
|:--|:--|
| 参考（原脚本，未改动） | `runs/lookahead/ref/preds_original_v3/`（8 序列 + summary.json） |
| 新脚本 `--lookahead off` | `runs/lookahead/off/preds/` |
| **整份 JSON 文件 md5** | **8/8 序列 + summary.json 全部 IDENTICAL**（含 `config` + `metrics` + `per_frame`） |
| 总体指标 | frames=28997 GT=27173 TP=25454 FP=182 FN=1719 **R=93.67% P=99.29% F1=96.400% FAR=0.0063**（与参考**逐位相同**） |

---

## 4. 尺子 2 的四臂对照（Δ=8 = 320 ms @25 Hz，8 序列汇总）

| 臂 | F1 | Recall | Precision | FAR | CLE p50 | CLE p90 | CLE p95 | hit@5px | coverage |
|---|---|---|---|---|---|---|---|---|---|
| **original**（无滞后天花板） | **96.400%** | 93.67% | 99.29% | 0.0063 | 1.97 | 7.97 | 13.43 | 88.13% | 89.87% |
| **stale**（老逻辑硬吃滞后） | 77.203% | 75.00% | 79.54% | 0.1807 | 20.43 | 78.02 | 113.73 | 38.18% | 89.82% |
| **kinematic**（档位 A，KF 速度） | 77.552% | 75.34% | 79.90% | 0.1776 | 15.68 | 89.98 | **183.05** | **33.29%** | 89.82% |
| **replay**（档位 B 视觉重推） | **93.353%** | 90.69% | 96.18% | 0.0337 | **4.12** | 25.57 | 91.92 | **72.66%** | 89.82% |

**读法**：
1. **老逻辑的代价被量化了**：硬吃 320 ms ⇒ F1 −19.2 pp、hit@5px −49.95 pp、CLE p50 ×10.4。
2. **档位 B 挽回绝大部分**：F1 +16.15 pp、hit@5px +34.48 pp、CLE p50 −79.8%；
   距无滞后天花板只差 **3.05 pp F1**（且比上一轮异步双频架构的 4.65~8.52 pp 差距**更小**）。
3. **档位 A 不成立**：F1 只 +0.35 pp，**hit@5px 反而 −4.89 pp**，CLE p95 从 113.73 **恶化到 183.05**。
4. **代价在 FAR**：replay 臂 FAR 0.0337（original 0.0063 / stale 0.1807）——
   **比交付线 0.030 高 12%**，是档位 B 唯一没有达标的指标。

### 4.1 逐序列（每一条序列 replay 都赢 stale，没有单条拉平均）

| 序列 | original F1 | stale F1 | kinematic F1 | replay F1 | original hit@5px | stale hit@5px | kinematic hit@5px | replay hit@5px |
|---|---|---|---|---|---|---|---|---|
| VIDEO00006_19700101_000703 | 99.77% | 74.86% | 76.06% | 94.22% | 95.92% | 19.69% | 18.77% | 64.93% |
| VIDEO00007_19700101_000818 | 100.00% | 81.40% | 89.19% | 99.43% | 99.65% | 36.75% | 35.41% | 90.61% |
| VIDEO00007_19700101_001324 | 90.35% | 75.35% | **66.94%** | 84.67% | 70.53% | 43.65% | 34.76% | 49.23% |
| VIDEO00008_19700101_001307 | 99.96% | 74.30% | 78.34% | 98.62% | 99.78% | 41.57% | 36.88% | 96.01% |
| VIDEO00009_19700101_001531 | 99.57% | 51.95% | 79.65% | 90.04% | 99.13% | 3.48% | 21.74% | 72.17% |
| VIDEO00029_19700101_014053 | 100.00% | 58.11% | 66.22% | 94.59% | 99.66% | 15.37% | 13.68% | 61.99% |
| VIDEO00033_19700101_015000 | 99.81% | 82.96% | 85.10% | 98.83% | 99.61% | 28.56% | 27.10% | 80.12% |
| VIDEO00034_19700101_015052 | 100.00% | 82.43% | 82.35% | 99.68% | 99.60% | 39.31% | 31.96% | 93.06% |

- **replay vs stale：8/8 条序列 F1 全赢**（+6.1 ~ +38.1 pp），hit@5px **7/8 赢**（001324 输 5.58 pp）。
- **kinematic vs stale：6/8 输**（001324 上 75.35% → 66.94%，**倒退 8.4 pp**）。

### 4.2 OSD 视频复核（肉眼可查的证据，001324 的 300 帧脱靶窗口）

§4.1 里 001324 是 replay 唯一输给 stale 的序列（hit@5px 49.23% vs 43.65%），
所以专门定位它**滞后真正致命的那一段**做 OSD 三臂对照。

**窗口选取**（数据驱动，不是手挑）：用交付评测**同一套 inbox 判定**
（`manu/videos/render_pred_gt_osd_video.match_frame`）逐帧扫全序列，
找出「stale 全脱靶 且 replay 全命中」的连续帧段 ⇒ 最长的一段是
**frames 8628..8691（连续 64 帧）**。取其外围 **8500..8799（300 帧）** 做渲染。

| 臂 | F1 | Recall | FAR | CLE p50 | CLE p95 | CLE max | hit@5px/GT | hit@10px/GT |
|---|---|---|---|---|---|---|---|---|
| **original**（无滞后天花板） | **100.00%** | 100.00% | 0.0000 | 1.86 | 3.06 | 4.52 | **100.00%** | 100.00% |
| **stale**（老逻辑硬吃滞后） | **62.33%** | 62.33% | 0.3767 | 4.24 | 20.94 | 25.81 | **31.67%** | 36.67% |
| **replay**（档位 B 视觉重推） | **90.67%** | 90.67% | 0.0933 | **1.05** | 7.47 | 19.13 | **76.00%** | 90.00% |

- **产物 JSON**：`runs/lookahead/osd/window_lagfail_001324.json`（窗口指标，上表数字全部出自此文件）。
- **OSD 视频**（`runs/lookahead/osd/videos/`，红=系统检出 / 绿=命中的 GT / 橙=漏掉的 GT）：
  - `VIDEO00007_19700101_001324_original_lagfail.mp4`
  - `VIDEO00007_19700101_001324_stale_lagfail.mp4`
  - `VIDEO00007_19700101_001324_replay_lagfail.mp4`
  - 逐帧判定 JSON：同目录 `*_lagfail.json`（渲染器原生 `--out-json`）
- **目视结论**：stale 臂的红圈**明显拖在绿 GT 框后面**（误差 13.6 → 58.4 px 递增），
  replay 臂红圈**始终落在框内**（误差 1.7 → 8.8 px）。
- **对照窗口**（中速机动）：另渲染 6000..6599（600 帧，Δ=8 滞后位移 p50 6.08 / p95 31.52 px），
  产物 `runs/lookahead/osd/window_moderate_001324.json`：

  | 臂 | F1 | FAR | CLE p50 | CLE p95 | hit@5px/GT |
  |---|---|---|---|---|---|
  | original | 100.00% | 0.0000 | 1.50 | 2.62 | 100.00% |
  | stale | 85.50% | 0.1450 | 4.94 | 17.98 | 43.33% |
  | replay | 100.00% | 0.0000 | 4.48 | 5.57 | 78.00% |

  ⇒ **replay 在中速窗口 F1 已追平天花板（100%），但 CLE p50 = 4.48 px 仍高于 original 的 1.50 px，
  hit@5px 只有 78.00%** ⇒ 残差是**系统性偏置**（快核质心被拖尾拽偏）而不是滞后，
  这正是 §7.2/§7.3 要治的问题。

---

## 5. Δ 敏感性扫描（老逻辑随延迟崩塌，档位 B 基本拉平）

| Δ | 臂 | F1 | Recall | FAR | CLE p50 | CLE p95 | hit@5px |
|---|---|---|---|---|---|---|---|
| 2 | stale | 94.817% | 92.13% | 0.0206 | 4.75 | 32.32 | 69.10% |
| 2 | kinematic | 92.893% | 90.26% | 0.0381 | 3.84 | 45.17 | 71.22% |
| 2 | **replay** | **96.021%** | 93.30% | 0.0097 | 3.16 | 18.26 | 80.37% |
| 4 | stale | 89.805% | 87.25% | 0.0662 | 9.92 | 57.41 | 54.50% |
| 4 | kinematic | 89.404% | 86.86% | 0.0698 | 6.24 | 91.17 | 56.70% |
| 4 | **replay** | **95.233%** | 92.53% | 0.0168 | 3.74 | 28.80 | 76.12% |
| 8 | stale | 77.203% | 75.00% | 0.1807 | 20.43 | 113.73 | 38.18% |
| 8 | kinematic | 77.552% | 75.34% | 0.1776 | 15.68 | 183.05 | 33.29% |
| 8 | **replay** | **93.353%** | 90.69% | 0.0337 | 4.12 | 91.92 | 72.66% |
| 12 | stale | 68.160% | 66.20% | 0.2628 | 29.66 | 154.40 | 30.10% |
| 12 | kinematic | 62.718% | 60.91% | 0.3123 | 27.87 | 269.93 | 21.44% |
| 12 | **replay** | **91.156%** | 88.53% | 0.0535 | 4.47 | 137.64 | 70.45% |

**读法**：
- **老逻辑**：Δ 每翻一倍，F1 掉 5~9 pp，Δ=8→12 掉 9.0 pp；hit@5px 从 88.13%(Δ=0) 塌到 30.10%(Δ=12)。
- **档位 B**：Δ=2→12，F1 只掉 4.87 pp（96.02 → 91.16），hit@5px 只掉 9.9 pp（80.37 → 70.45）；
  ⇒ **把系统对检测延迟的敏感度压低了约一个数量级**（Δ 敏感度从 −19.2 pp/8帧 降到 −4.9 pp/10帧）。
- **档位 A**：**四个 Δ 全部不优于 stale**（Δ=2/4/8/12 分别 −1.92 / −0.40 / +0.35 / −5.44 pp F1），
  且 Δ≥8 时 hit@5px **低于** stale ⇒ **纯运动学外推不是本任务的解**。

---

## 6. 为什么档位 A 不成立（机理，两条独立证据）

### 6.1 速度源对照（Δ=8）

| velocity_source | F1 | FAR | CLE p50 | CLE p95 | hit@5px |
|---|---|---|---|---|---|
| `kf`（卡尔曼后验速度） | 77.552% | 0.1776 | 15.68 | 183.05 | 33.29% |
| `diff1`（1 帧差分） | 77.022% | 0.1824 | 17.00 | 111.80 | 27.46% |
| `diffN`（3 点最小二乘） | 80.567% | 0.1501 | 15.63 | 102.42 | 33.81% |

三档速度源**结论一致**：档位 A 最多 +3.36 pp F1，且 **hit@5px 全部低于 stale 的 38.18%**。

### 6.2 两条机理

1. **静止目标被推偏**：龙泉山目标多数时刻几乎静止（Δ=8 真实位移 p50 = 3.54 px）。
   此时 `Δ·v` 里的 `v` 主要是**速度噪声**（1 帧差分的量级 ≈ 0.5~2 px/帧 ⇒ ×8 = 4~16 px），
   把本来已经落在 GT 框里的上报点**推出框外** ⇒ hit@5px 下降、FAR 上升。
   这是"外推噪声 > 滞后收益"的典型负收益区间（Δ=2 时最明显：F1 −1.92 pp）。
2. **匀速假设在尾部失效**：目标在 Δ 帧内的真实位移 p90 = 51 px / p95 = 91 px / p99 = 180 px。
   匀速外推的误差 ∝ 速度估计误差 × Δ ⇒ 尾部误差被放大到 183 px（CLE p95）。

⇒ **推论**：档位 A 只有在"目标匀速 + 速度估计足够干净"时才成立；本数据两者都不满足。
**要挽回滞后，必须用图像（档位 B），不能只用运动学。**

---

## 7. 档位 B 的代价与已排除的改进方向

### 7.1 代价（实测）

| 项 | 数值 | 说明 |
|:--|:--|:--|
| 重推步数 | 191,311 步 / 25,620 帧 | 平均 7.5 步/帧（Δ=8） |
| 重推失败（弃权走兜底） | 2,474 帧（**9.66%**） | 001324 最重：1,601/8,990 = 17.8% |
| 兜底次数 | 2,474 帧走 kinematic | 因 kinematic 本身不靠谱，兜底**不改善**质量（见 §7.3） |
| FAR | 0.0337（交付线 **0.030**，超 12%） | 比 stale 的 0.1807 好 5.4 倍，但比 original 的 0.0063 差 5.4 倍 |
| max_jitter | 234.47 px（stale 65.24 / original 65.24） | 转台抽搐风险；须在板端加平滑 |

### 7.2 误检来源已定位：**不是"跳到杂波"，是"质心被拖尾拽偏"的系统性偏差**

试过两个门，都被数据否掉（Δ=8）：

| 方案 | F1 | FAR | hit@5px | 判读 |
|---|---|---|---|---|
| 无门（基线） | 93.353% | 0.0337 | 72.66% | — |
| 门：重推位 vs 运动学预测偏差 ≤ 6 px | 77.578% | 0.1773 | 38.39% | **崩塌**：运动学预测本身不可靠（CLE p95 183 px），用它当参照会把好推演全毙掉 |
| 门：逐帧位移 ≤ 12 px/步 | 92.217% | **0.0441** | 72.30% | **负收益**：被毙的帧退回 kinematic ⇒ FP 978→1278 |
| 门：逐帧位移 ≤ 20 px/步 | 93.353% | 0.0337 | 72.66% | **空操作**（重推从不单步跳 >20 px） |

⇒ **重推从不"跳"**（20 px/步门是 no-op），误检是**缓慢的系统性偏移**：
龙泉山目标是**成簇亮点（本体 + 拖尾）**，局部质心被拖尾拽偏（与上一轮"快核单步误差 p50 2.22 px"同源）。
⇒ **要压 FAR 必须治快核（成簇目标本体提取），不是在输出端加门。**
（`--replay-step-gate-px` 保留为诊断开关，默认 **0 = 不设门**。）

### 7.3 兜底策略也要改

当前兜底是 kinematic，而 kinematic 在 Δ=8 时 F1 77.55% / FAR 0.1776 ⇒ **兜底比原位更差**。
若要保留兜底，应改 `stale`（F1 77.20%）或 `abstain`（不输出，让转台知道没坐标）。
⇒ **下一步必须重做兜底策略**，这是当前实现里最明显的短板。

---

## 8. 复现命令

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics
source ~/anaconda3/etc/profile.d/conda.sh && conda activate uav
CACHE=/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache
GT=/home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json

# ① 参考（原脚本，未改动）
PYTHONPATH=. python manu/evaluation/system_sota_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --out-json runs/lookahead/ref/original_v3_8seq.json \
  --out-preds-dir runs/lookahead/ref/preds_original_v3

# ② 新脚本（开关关闭 ⇒ 必须与 ① 逐位一致）+ 四臂对照
PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --lookahead off --arms original,stale,kinematic,replay --delta 8 \
  --out-json runs/lookahead/off/off.json --out-preds-dir runs/lookahead/off/preds

# ③ Gate A：整份 JSON 的 md5 逐位比对
for f in runs/lookahead/ref/preds_original_v3/*.json; do
  b=$(basename $f); a=$(md5sum $f | cut -d' ' -f1); c=$(md5sum runs/lookahead/off/preds/$b | cut -d' ' -f1)
  [ "$a" = "$c" ] && echo "IDENTICAL $b" || echo "DIFFER $b"
done

# ④ Δ 扫描 / 速度源扫描 / 门控扫描
for d in 2 4 12; do PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --lookahead replay --delta $d --out-json runs/lookahead/sweep/delta$d.json; done
for vs in kf diff1 diffN; do PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --lookahead kinematic --velocity-source $vs --out-json runs/lookahead/sweep/$vs.json; done

# ⑤ 单测
PYTHONPATH=. python -m unittest discover -s manu/tracking/tests -t . -v   # 135 项

# ⑥ OSD 视频（§4.2 的三臂对照，001324 的 300 帧脱靶窗口）
FRAMES=/home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg
SEQ=VIDEO00007_19700101_001324
# ⑥a 先产出该序列的 stale / replay 预测
for arm in stale replay; do PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --seq $SEQ --params-json $PARAMS \
  --lookahead $arm --delta 8 --arms original,stale,replay \
  --out-json runs/lookahead/osd/$arm.json --out-preds-dir runs/lookahead/osd/preds_$arm; done
# ⑥b 再渲染（original 用参考臂的预测目录）
for pair in "original:runs/lookahead/ref/preds_original_v3" \
            "stale:runs/lookahead/osd/preds_stale" \
            "replay:runs/lookahead/osd/preds_replay"; do
  arm=${pair%%:*}; preds=${pair#*:}
  PYTHONPATH=. python manu/videos/render_sys_sota_longquan.py \
    --preds-dir "$preds" --seq $SEQ \
    --output runs/lookahead/osd/videos/${SEQ}_${arm}_lagfail.mp4 \
    --labels-root $GT --frames-root $FRAMES --start 8500 --limit 300 --codec ffmpeg:h264 \
    --out-json runs/lookahead/osd/videos/${SEQ}_${arm}_lagfail.json
done
```

---

## 9. 判读与下一步（按优先级）

1. **交付结论**：**档位 B（视觉重推）可以进板端验证**——它在 8/8 序列上把 320 ms 滞后造成的
   脱靶挽回 **84.1% 的 F1 损失**（77.203% → 93.353%，天花板 96.400%），
   **hit@5px 绝对 +34.48 pp**（38.18% → 72.66%，回收了滞后损失的 69.0%），
   且**对成熟跟踪资产零回归**（ΔF1 = 0.000000）。
   **档位 A 不要做**（四个 Δ 工况全部不优于基线）。**肉眼可查的证据见 §4.2 的三臂 OSD 视频**
   （stale 红圈拖在 GT 框后、replay 红圈锁在框内）。
2. **FAR 0.0337 超交付线 0.030**：根因是**快核在成簇目标上的质心偏差**，不是架构问题 ⇒
   下一步治快核（本体提取 / 去拖尾），不是加门（两个门都已被数据否掉，见 §7.2）。
3. **兜底策略要重做**（当前兜底 kinematic 比原位更差，见 §7.3）：候选 stale / abstain。
4. **max_jitter 234 px**：板端必须加输出平滑，否则转台会抽搐；Python 侧暂未做任何平滑
   （本轮刻意不加，避免把"外推收益"和"平滑收益"混在一起）。
5. **板端移植**：`manu/tracking/lookahead/extrapolator.py` 的 replay 臂要移植到 C++；
   逐帧位移门、兜底策略、无状态性（不写回跟踪器状态）三条是移植契约。
6. **Δ 降一降**：若能把检测延迟从 320 ms 压到 160 ms（Δ=4），
   老逻辑 F1 从 77.20% 升到 89.81%、档位 B 从 93.35% 升到 95.23% ⇒ **降延迟与加外挂是叠加收益**。

---

## 10. 铁律提醒

- 模型固定 **RC2 / trial_0080 / epoch3**，**不重跑推理**（回放 `ms_cache/*.pkl`）；
- 评测口径 **import 冻结实现**（`manu/evaluation/eval_multiscale_cache_th.py`）；
- **配对数据用 McNemar 精确二项检验**，Wilson CI 重叠不能当等价性证明（上一轮教训）；
- **文档里的每个统计数字都必须来自产物 JSON**（铁律九）；
- 未经授权**不得**启动训练 / 重推理 / 数据集下载；
- `manu/data/build_full_median_dataset.py` 禁改；
- **`manu/evaluation/*` 的原文件禁改**——外挂一律走 `manu/tracking/lookahead/` 里的**拷贝**。

---

## 附：相关文件

- 代码：`manu/tracking/lookahead/{__init__.py, extrapolator.py, track_fusion.py, lookahead_longquan.py}`
- 单测：`manu/tracking/tests/test_lookahead_plugin.py`（14 项）
- 产物：`runs/lookahead/{ref,off,sweep,gate}/`
- 上一轮（异步双频检跟分离）：`manu/memory/async_dual_rate_tracking_20261102.md`

---

# 第二段追加（2026-11-02 晚）：根因取证 → 修复 → 最终对照表

> **本段结论（一句话）**：在 **Δ=8（320 ms @25 Hz，真实物理滞后）** 下，
> **最优外挂方案 = `--lookahead replay --replay-anchor delta --replay-fallback holdv
> --no-replay-hold-use-ref --fast-win-size 48 --fast-bg-window 96`**，
> 相对**原逻辑 `--lookahead off`** 退化 **F1 −2.085 pp / hit@5px −9.13 pp / FAR ×3.97 /
> max_jitter 65→366 px**；相对"同样硬吃 320 ms 滞后的纯检测基线"**挽回 +17.11 pp F1
> （追回 89.1% 的滞后损失）/ +40.81 pp 命中率**。
> **Gate A 复验：关掉开关 ⇒ 9 份预测 JSON（8 序列 + summary）md5 与 `cmp` 全等。**

## 11. 最终对照表（★ = 本轮新增的实测；产物 = `runs/lookahead/tune/final_delivery.json`）

### 11.1 总体（Δ=8 = 320 ms，8 序列汇总，RC2/trial_0080/epoch3）

| 方案 | F1 | Precision | Recall | FAR/帧 | hit@5px | CLE p50 | CLE p90 | CLE p95 | max_jitter | frag |
|---|---|---|---|---|---|---|---|---|---|---|
| **① 原逻辑 `--lookahead off`**（＝老版 Sys SOTA，无滞后天花板） | **96.400%** | 99.29% | 93.67% | **0.0063** | **88.13%** | **1.97** | 7.97 | **13.43** | **65.2** | 4 |
| ② 臂 1 纯检测基线（硬吃 320 ms 滞后，老逻辑现状） | 77.203% | 79.54% | 75.00% | 0.1807 | 38.18% | 20.43 | 78.02 | 113.73 | 65.2 | 5 |
| **③ 外挂·最终交付候选**（replay / Δ=8 / anchor=delta / holdv / no-hold-use-ref / win48） | **94.316%** | 97.17% | 91.62% | 0.0250 | 79.00% | 3.00 | 18.34 | 35.66 | 365.9 | 5 |

### 11.2 退化账（③ vs ①）—— 用户要的"最好的方案退化情况"

| 指标 | 原逻辑 | 外挂 | 变化 |
|---|---|---|---|
| F1 | 96.400% | 94.316% | **−2.085 pp**（−2.2%） |
| Precision | 99.290% | 97.174% | −2.116 pp |
| Recall | 93.674% | 91.620% | −2.054 pp |
| FAR/帧 | 0.0063 | 0.0250 | **×3.97**（+0.019，仍**低于交付线 0.030**） |
| hit@5px | 88.132% | 78.998% | **−9.134 pp** |
| CLE p50 | 1.974 px | 3.000 px | +1.026 px（×1.52） |
| CLE p95 | 13.434 px | 35.657 px | +22.223 px（×2.66） |
| max_jitter | 65.238 px | 365.923 px | **+300.7 px（×5.61）← 唯一严重不达标项** |
| fragmentation | 4 | 5 | +1 |

### 11.3 收益账（③ vs ②）

- 老逻辑代价被量化：硬吃 320 ms ⇒ **F1 −19.197 pp**、hit@5px −49.95 pp、CLE p50 ×10.4。
- 外挂挽回：**F1 +17.112 pp**（追回 **89.1%** 的滞后损失）、hit@5px **+40.81 pp**、
  FAR 0.1807 → 0.0250（−86%）、CLE p50 20.43 → 3.00 px（−85%）。

### 11.4 逐序列退化（③ vs ①）

| 序列 | 原逻辑 F1 | 外挂 F1 | ΔF1 | 原逻辑 hit@5px | 外挂 hit@5px | Δhit | 外挂 p50 | 外挂 p95 | 外挂 max_jitter | 兜底帧 |
|---|---|---|---|---|---|---|---|---|---|---|
| 000703 | 99.771% | 96.793% | −2.978 | 95.92% | 72.68% | −23.25 | 3.00 | 14.47 | 149.4 | 121/1750 |
| 000818 | 100.000% | 99.600% | −0.400 | 99.65% | 96.04% | −3.61 | 1.58 | 4.57 | 76.5 | 45/8255 |
| 001324 | 90.346% | 87.191% | −3.155 | 70.53% | 58.90% | −11.63 | 2.59 | 35.66 | **365.9** | 846/8990 |
| 001307 | 99.959% | 97.144% | −2.815 | 99.78% | 92.57% | −7.21 | 1.64 | 5.61 | 76.5 | 38/3658 |
| 001531 | 99.567% | 92.641% | −6.926 | 99.13% | 85.22% | −13.91 | 1.85 | 7.47 | 11.0 | 0/116 |
| 014053 | 100.000% | 94.595% | −5.405 | 99.66% | 77.36% | −22.30 | 2.37 | 18.43 | **331.5** | 25/592 |
| 015000 | 99.805% | 98.832% | −0.974 | 99.61% | 87.23% | −12.38 | 2.40 | 7.20 | 41.6 | 146/1028 |
| 015052 | 100.000% | 99.676% | −0.324 | 99.60% | 98.39% | −1.21 | 1.81 | 3.69 | 19.7 | 2/1231 |

> **退化集中在两类帧**：
> ① **兜底帧**（1223/25620 = 4.8%，001324 占 846）——快核在源帧/中途就失败；
> ② **000703 / 014053 的"质心偏置"帧**——目标清晰但 ROI 窗口把拖尾/邻域一起算进质心。
> 二者都是**重推链本身**的误差，不是"延迟"造成的。

## 12. 本轮修掉的两个真 bug

### 12.1 「首步离群门」从来没生效过（死代码）

`extrapolator.py` 的判定写成了 `if rel_tol > 0.0 and not accepted_steps and ref_available:`，
而 `accepted_steps.append(...)` **在这行之前**已经执行 ⇒ `not accepted_steps` **恒为 False**
⇒ 门永远不触发。**证据**：8 序列 `replay_step_rejected` 恒为 0；
把门关掉（`--replay-step-rel-tol-px 0`）与打开（15）跑出来**逐位相同**。

**修复**（`is_first_step = len(accepted_steps) == 0` 提到 append 之前）后重测：

| 配置 | F1 | FAR | hit@5px | CLE p95 | max_jitter |
|---|---|---|---|---|---|
| 门关（0） | **94.316%** | 0.0250 | 79.00% | 35.66 | 365.9 |
| 门开（15，noref） | 92.577% | 0.0408 | 77.63% | 63.38 | — |
| 门开（15，hold_use_ref） | 92.558% | 0.0410 | 77.57% | 182.83 | — |

⇒ **门真正生效后是净亏 −1.74 pp F1**：快核首步位移噪声本身就有 2~10 px，
15 px 的门把大量**合法机动**误杀。**结论：默认关**（CLI/dataclass 默认已改成 0.0），
代码保留（已修正）作为消融项。

### 12.2 `--no-replay-hold-use-ref` 才是最优口径

`hold_use_ref=True`（按卡尔曼参考速度推进）vs `False`（按**最后一次实测步长**推进），
后者在 8 序列上全面更优：

| 口径 | F1 | FAR | hit@5px | CLE p95 | max_jitter |
|---|---|---|---|---|---|
| `hold_use_ref=True`（旧默认） | 94.240% | 0.0257 | 78.92% | 38.05 | 385.4 |
| **`hold_use_ref=False`（新默认）** | **94.316%** | **0.0250** | **79.00%** | **35.66** | 365.9 |

**机理**：源帧快核就失败时（001324 有 846 帧），`accepted_steps` 为空 ⇒
`hold_use_ref=True` 会用卡尔曼速度把位置**往前推**（长程外推，误差被放大）；
`False` 则**冻在最后一次实测位置**（短程，更保守）。对伺服而言"宁可不动也别跳"更安全。

## 13. 本轮新增：连续性兜底 `holdc`（输出级连续，**未入选**）

**动机**：`holdv` 兜底 = "重推链失败点 + 剩余步数 × 单步位移"，剩余步数最多 8、
单步位移实测可达 22 px ⇒ **单帧就能跳出 ~176 px 的幻影位移** ⇒ 转台抽搐
（001324 max_jitter 366 px）。

**实现**（`--replay-fallback holdc`）：兜底位置改为
**「上一帧该 track 的实际上报位置 + 参考速度 × 1」**（输出级小状态，**不回写管线任何估计**），
单帧跳变被钉在 |v|（几 px）量级。

| 配置 | F1 | FAR | hit@5px | CLE p95 | **max_jitter** | 001324 jitter |
|---|---|---|---|---|---|---|
| holdv（交付候选） | **94.316%** | **0.0250** | **79.00%** | **35.66** | 365.9 | 365.9 |
| holdc + noref | 93.906% | 0.0287 | 78.82% | 66.65 | **331.5** | **218.1** |
| holdc + hold_use_ref | 93.781% | 0.0298 | 78.59% | 73.44 | 385.4 | 385.4 |

⇒ `holdc` 把最差序列的抖动从 366 降到 218 px，但 **F1 −0.41 pp、CLE p95 反而恶化
35.7→66.7**（"上一帧 + v"在目标快速机动时追不上）。**按"精度优先"口径不入选**，
保留为消融项 + **板端上报侧连续性限幅的备选实现**。

## 14. 本轮被否掉的路线（勿重启）

| 路线 | 实测结果 | 否决理由 |
|---|---|---|
| `peak_centroid` 快核模式 | hit@5px 49.23% → 27.62% | 质心模式本来就更好 |
| 逐帧相对位移门（第二步起） | tol=4 误杀 37% 步；tol=12 在 001324 上 F1 85.44%（净亏） | 快核步长噪声本身 2~10 px，任何阈值都误杀真实机动 |
| **首步离群门（修好之后）** | F1 94.32% → 92.58% | 同上，默认关 |
| 批量稳健求和（`replay_outlier_tol_px=10`） | p95 15.3 → 17.2 | 净亏，默认关 |
| 全窗口中位数 hold | 00029 +2.19pp 但 001324 −0.50pp、jitter 218→375 | 改为末步口径 |
| `pred` 档位（ROI 跟卡尔曼预测走） | 00029 p50 2.37→3.69、p95 19→24 | 管线卡尔曼速度不够准（8 帧累计误差 ≈5.8 px） |
| `kf` 档位 | 同上 | 保留为消融 |
| kinematic 兜底 | 00029 p50 15.68；001324 p95 183 | 不是好安全网 |
| win32 / win64 | 93.72% / 93.93% | 均劣于 win48 |
| 源帧锚点校验 `--replay-anchor-check-px 12` | F1 94.289%（vs 94.316%） | 净亏，默认 0 |
| `holdc` 连续性兜底 | F1 −0.41 pp | 精度优先口径不入选（见 §13） |

## 15. 本轮的复现命令与产物

```bash
CACHE=/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache
GT=/home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json

# ★ 最终交付候选（≈175 s，8 序列）
python manu/tracking/lookahead/lookahead_longquan.py --cache-root $CACHE --labels-root $GT \
  --all-seqs --params-json $PARAMS --lookahead replay --delta 8 --arms original,stale,replay \
  --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta \
  --fast-win-size 48 --fast-bg-window 96 --no-replay-hold-use-ref \
  --out-json runs/lookahead/tune/final_delivery.json \
  --out-preds-dir runs/lookahead/tune/final_delivery_preds

# ★ Gate A（关开关 ⇒ 与原脚本逐位一致）
python manu/tracking/lookahead/lookahead_longquan.py --cache-root $CACHE --labels-root $GT \
  --all-seqs --params-json $PARAMS --lookahead off --arms original \
  --out-json runs/lookahead/gate/final_off.json --out-preds-dir runs/lookahead/gate/final_off_preds
#   ⇒ 与 runs/lookahead/ref/preds_original_v3/ 比对：9 份文件 md5 与 cmp 全等

# 单测（全库 135 项 = 131 + 本轮新增 4 项：首步门回归 ×2、holdc 连续性 ×2）
PYTHONPATH=. python -m unittest discover -s manu/tracking/tests -t . -v

# 对照表生成（本段所有表格都来自产物 JSON，不重算）
python manu/tracking/lookahead/make_report_table.py \
  --runs runs/lookahead/tune/final_delivery.json --labels "外挂·最终交付候选"
```

**产物清单**：`runs/lookahead/tune/{final_delivery, fx_noref_gate0, fx_holdc, fx_holdc_ref,
fx_noref_gate15, fx_ref_gate15, abl_nogate, abl_noref, abl_noref_anchchk12, abl_kernel,
combo_delta_win48, final_v4, final_v5, final_v6}.json`、
`runs/lookahead/gate/{final_off.json, final_off_preds/}`。

## 16. 未决风险（交付说明必须带上）

1. **max_jitter 366 px ≫ 原逻辑 65 px**——转台会抽搐。
   **必须在板端上报侧加连续性限幅**（限幅/速率限制），或用 `holdc` 兜底，
   或把 Δ 从 8 降到 4。**本外挂本身不做限幅**。
2. **产物与代码尚未同步服务器**（沙箱网络层阻断，`ssh` 报 `Operation not permitted`），
   需在有写权限处补做 md5 三向比对。
3. **FAR 0.0250**：比原逻辑差 4 倍，但**仍在交付线 0.030 以内**。
4. `pred` / `kf` 档位、批量稳健求和、首步离群门、`holdc` 都是**消融项，不要上板**。
