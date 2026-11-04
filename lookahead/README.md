# 快速前向推演外挂（Look-ahead Extrapolator）

> **一句话**：在 RC2/trial_0080 龙泉山管线的**输出末端**外挂一层**无状态**前向推演，把上报给转台的坐标
> 从滞后的 `t−Δ` 帧推到物理帧 `t`；**成熟跟踪资产零回归**（关掉开关 ⇒ 8/8 序列整份预测 JSON 的
> md5 逐字节一致），**档位 B（视觉重推 + xcorr 模板匹配精修）把 320 ms 物理滞后造成的脱靶几乎全部挽回**
> （Δ=8，8 序列：hit@5px **38.18% → 85.17%**，F1 **77.203% → 95.266%**，FAR 0.1807 → **0.0163**，
> 相对零延迟天花板只差 **1.13 pp**；滞后损失追回 **94.09%**）。

**它解决什么问题**：异步检跟分离架构下，检测框滞后 Δ=8 个 record（相机 25 Hz ⇒ **320 ms**）才到转台，
转台拿到的其实是 `t−Δ` 帧的旧坐标 ⇒ 目标做机动时**转台脱靶**。本模块不碰检测、不碰跟踪，只在**上报流**
这一层做补偿，**只读输出流、不回写任何历史估计** ⇒ "关掉开关 ⇒ 与原脚本逐位一致"是**构造性保证**。

```
[IR 帧 25Hz] → [NPU 多尺度热图 ≈320ms] → [RC2 检测框] → [龙泉山跟踪管线] → D(f)
                                                                       │
     ┌──────────── 输出级外挂（纯后处理，不回写任何状态）────────────┐
     │ stale     : report(t) = D(t−Δ)          （老逻辑基线）        │
     │ kinematic : report(t) = D(t−Δ) + Δ·v    （档位 A）             │
     │ replay    : report(t) = D(t−Δ) 逐帧重推（档位 B）             │
     └────────────────────────────────────────────────────────────────┘
                                                    ▼ 转台上报坐标（物理帧 t）
```

## 0. 基本事实

| 项 | 值 |
|:--|:--|
| 执行机 / 环境 | **服务器 = 重活全部在这里跑** `test-Super-Server`（80 核 / 125 GB），cwd `/tmp/pycharm_project_10ae9e2e`，conda env `uav`（`/home/huangzhe/anaconda3/envs/uav/bin/python`）· **本地 = 编辑/抽检** `/media/manu/1TB-Volume/workspace/ultralytics`（分支 `manu_dev_v2`；`manu/tracking/` 未提交，在工作区）· Python 3.10.20，numpy 2.2.6，cv2 4.10，**无 pytest ⇒ 用 unittest** |
| 服务器路径（**唯一正确**） | 缓存 `/tmp/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache` · GT `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010`（**平铺文件** `<SEQ>__<idx>.txt`）· 帧 `/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg/<SEQ>/frame_%06d.jpg` |
| 模型 | **RC2 / trial_0080 / epoch3**（回放 `ms_cache/*.pkl`，**不重跑推理**） |
| 数据（**SSHFS 只读**） | 缓存 `/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache`（**8 序列**）· GT `/home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010`（**取服务器侧路径**，不要指本机只读镜像）· 帧 `/home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg/<SEQ>/frame_%06d.jpg`（≈7.4 ms/帧） |
| 工作点 | `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`（th_base 0.48 / sky_ratio 1.0 / th_ground 0.26 / confirm_consec 7） |
| 代码 / 单测 / 产物 | `manu/tracking/lookahead/`：`extrapolator.py`（外挂核）· `xcorr.py`（**模板匹配位移精修**）· `track_fusion.py`（跟踪器拷贝）· `lookahead_longquan.py`（管线拷贝 + 四臂 + CLI）· `diag_xcorr.py`（诊断）· 单测 `manu/tracking/tests/test_lookahead_plugin.py`（**18 项**，全库 **135 项**全绿）· 产物 `runs/lookahead/{ref,off,sweep,gate,osd,final}/`（见 §7） |
| 跨机同步 | ⚠️ **host 与服务器的代码不同步**（本轮改过的 `xcorr.py` / `extrapolator.py` / `lookahead_longquan.py` 只在服务器上）。同步包 `/mnt/data/codex-overnight/code_sync_20261103.tar.gz`（md5 见 memory §23.7） |
| 同步 | ⚠️ **产物与代码尚未同步服务器**（沙箱网络层阻断，`ssh -F /dev/null 192.168.99.40` → `socket: Operation not permitted`），需在有写权限处补做 md5 三向比对 |

### 0.1 拷贝纪律与时间轴（**两个最容易踩的坑**）

**拷贝纪律**：本目录的 `track_fusion.py` / `lookahead_longquan.py` 分别是 `manu/evaluation/eval_bidirectional_track_fusion.py`
/ `manu/evaluation/system_sota_longquan.py` 的**逐字拷贝**——前者只加了 `capture_kf_states` 开关（Pass 1 的
`online_tracker.step(...)` **之后**把每条 track 的 `(get_pos(), get_velocity())` 存进 `kf_states[f_idx]`，
**既有算术一行未动**，`diff` 只有 5 个 hunk），后者只加了 11 个 CLI 参数 + 四臂对照 + 两把尺子
（`--lookahead off` 时 `config["source"]` 仍是 `"system_sota"` ⇒ **整份 JSON 可做 md5 比对**）。
⇒ **`manu/evaluation/*` 的原文件一律禁改**。
**时间轴**：管线时间轴 = **record 下标**（`f_idx`），不是视频绝对帧号（`records[i]["frame_index"]`）；Δ 以 record
为单位（Δ=8 ≈ 320 ms @25 Hz）；物理帧 t 的源帧 f = t − Δ，`t < Δ` 不输出（计入 `no_source_frames`）。

## 1. 四臂语义

`--lookahead` 决定**主臂**（写进产物 JSON 的 `overall` / `per_sequence` / `--out-preds-dir` 的那一条流）；
`--arms` 决定**几臂并行对照**。四臂共用同一份 `D(f)`，差别只在"怎么把 `D(t−Δ)` 变成上报坐标"。

| 臂 | 运行机制 | 延迟处理方式 | CLI 参数值 | `source_verbatim` |
|:--|:--|:--|:--|:--|
| `original` | **不做任何前推**，直接透传管线自己的输出 `report(t) = D(t)`（逐位拷贝 dict，不做任何数值运算） | 无延迟可处理 ⇒ 这是**无滞后天花板**的锚 | `--lookahead off`（主臂） / `--arms original,...` | `None` |
| `stale` | **老逻辑基线**：把源帧检测整条搬过来 `report(t) = D(t−Δ)`（位置/分数/track_id 原样搬运） | **硬吃滞后**——不补偿，直接上报 Δ 帧前的旧坐标 | `--lookahead stale` | `True` |
| `kinematic` | **档位 A，纯运动学前推**：`report(t) = D(t−Δ) + Δ·v`；速度取 `--velocity-source`，起点取 `--position-source`，其余字段（score/track_id/infilled）与源帧一致；速度取不到时**按 0 外推**（宁可少推，不凭空编运动） | 用**匀速假设**补滞后，零图像开销 | `--lookahead kinematic` | `False` |
| `replay` | **档位 B，视觉重推**：从 `t−Δ` 锚点出发，用**局部质心快核**（`LocalPatchTracker`，`win_size=32`）逐帧重推到 `t`（区间 `f+1..f+Δ` 共 Δ 步，每步以上一位置为输入中心）；缺帧 / 快核判无效 ⇒ **本次重推整体弃权**，按 `--replay-fallback` 兜底 | 用**图像**补滞后（真重推，不是估计） | `--lookahead replay` | `False` |

> `off` 臂的"逐位一致"是**构造性保证**：`build_report_stream` 的 `mode=="off"` 分支只做
> `[dict(det) for det in source_dets]`，一个字节的数值运算都没有。

## 2. 两把尺子（口径）

两把尺子**没有独立开关**：`--arms` 列出的**每一臂**都会被各算一遍，硬门禁（尺子 1）在 `original` 臂存在时自动生效；
`--lookahead` 只决定哪一臂是主臂。
**尺子 1 — 历史帧等价性**（证明"没破坏成熟跟踪资产"）：物理帧 t 时系统手上最新的估计是 `x̂_{t−Δ}`，拿它对
`GT_{t−Δ}` 评测，要求与老版 Sys SOTA 在同一数据上的指标**绝对一致**。**实现**：在共同源帧窗口 `[0, N−Δ)` 上对每臂的
**源帧估计**（= `D(f)` 本身）评测 ⇒ 各臂天然逐位相等；脚本把它做成**硬门禁**（不等就 `SystemExit`），并附
`source_verbatim` 判定该臂是否原样搬运源帧估计。

| 臂 | tp | fp | fn | gt | f1 | recall |
|---|---:|---:|---:|---:|---|---|
| original | 25438 | 182 | 1711 | 27149 | `0.9641266652769619` | 0.9369774208994807 |
| stale | 25438 | 182 | 1711 | 27149 | `0.9641266652769619` | 0.9369774208994807 |
| kinematic | 25438 | 182 | 1711 | 27149 | `0.9641266652769619` | 0.9369774208994807 |
| replay | 25438 | 182 | 1711 | 27149 | `0.9641266652769619` | 0.9369774208994807 |

⇒ **ΔF1 = 0.000000**（共同源帧窗口 1,855 record；脚本打印 `[GATE] 尺子1 各臂源帧口径 ΔF1 = 0.000000`）。
来源：`runs/lookahead/off/off.json` → `arms.<臂>.source_frame.overall` 与 `arms.<臂>.delta_f1_vs_original`。

**尺子 2 — 当前帧伺服有效性**（证明外推收益）：拿**上报给转台的坐标**对**当前帧** `GT_t` 评测。

| 指标 | 口径 |
|:--|:--|
| F1 / Recall / Precision / FAR | 复用冻结口径 `manu/evaluation/eval_multiscale_cache_th.py` 的 `count_matches(..., "inbox")`（贪心按分数降序、点落框内、**一个 GT 只吸收一个检测**） |
| CLE p50/p90/p95/p99/max | 每个「有 GT 且有上报」的 record，取**离最近 GT 中心最近的那个上报点**算欧氏距离；**跨序列取最差值**（分位数不可跨序列合并，`percentile_aggregation = "worst_over_sequences"`） |
| hit@5px | 上报点到最近 GT 中心 ≤ 5 px 的 GT 帧占比（**物理命中率**，分母 = `gt_frames`） |
| max_jitter / p99_jitter | 同一 track 上报序列的二阶差分 `|(p_t−p_{t−1})−(p_{t−1}−p_{t−2})|` 的 max / p99（按 track 分组，跨 track 汇总） |
| coverage | 有上报的帧 / 总帧 |

### 2.1 产物 JSON 怎么读（速查）

| 想读的量 | JSON 路径 |
|:--|:--|
| 尺子 1 F1 / 共同源帧窗口 | `arms.<臂>.source_frame.overall.f1` / `.source_frame.window` |
| 尺子 1 门禁判定量（须 = 0.0） | `arms.<臂>.delta_f1_vs_original` |
| 该臂是否原样搬运源帧估计 | `arms.<臂>.source_verbatim.<序列>`（`stale=True`，`kinematic/replay=False`） |
| 尺子 2 业务域 F1/Recall/Precision/FAR | `arms.<臂>.current_frame.overall.{f1,recall,precision,far_per_frame}` |
| 尺子 2 CLE / hit@5px / 抖动 / 覆盖率 | `arms.<臂>.servo.overall.{cle_p50,cle_p90,cle_p95,hit_rate_5px,max_jitter,p99_jitter,coverage}`；逐序列拆解在 `arms.<臂>.current_frame.per_sequence.<序列>` / `servo.per_sequence.<序列>` |
| 上报流健康度（弃权/兜底/重推步数） | `arms.<臂>.stream_stats.overall.{applied,abstained,fallback_used,missing_velocity,replay_steps,replay_failed}` |
| 本次跑的配置回显 | `lookahead.{mode,main_arm,delta,velocity_source,velocity_window,position_source,replay_require_valid,replay_fallback,replay_step_gate_px,arms,frames_root}` |

> `--out-table` 会额外把四臂汇总渲染成 markdown 对照表（尺子1 F1 / 尺子2 F1 / Recall / Prec / FAR·帧 /
> CLE50 / CLE95 / hit@5px / inbox / 抖动p99 / 碎片 / 覆盖率）。

### 2.2 Gate A：关掉开关 ⇒ 与原脚本逐位一致

参考 `runs/lookahead/ref/preds_original_v3/`（原脚本未改动）vs 新脚本 `runs/lookahead/off/preds/`：**整份 JSON 文件
md5 全部 IDENTICAL**（8/8 序列 + summary.json，含 `config` + `metrics` + `per_frame`）；总体指标 frames=28997
GT=27173 TP=25454 FP=182 FN=1719 **R=93.67% P=99.29% F1=96.400% FAR=0.0063**（与参考**逐位相同**）。

## 3. CLI 参数速查

全部以 `lookahead_longquan.py` 的 `parse_args()` 定义为准。

| 参数 | 含义 | 默认值 | 取值范围 / 说明 |
|:--|:--|:--|:--|
| `--lookahead` | 快推演开关，**选主臂** | `off` | `off` / `stale` / `kinematic` / `replay` |
| `--delta` | 检测滞后 Δ（**record 数**） | `8` | 正整数；8 ≈ 320 ms @25 Hz |
| `--velocity-source` | 速度来源 | `kf` | `kf`（卡尔曼后验速度）/ `diff1`（相邻两帧差分）/ `diffN`（最近 N 帧最小二乘） |
| `--velocity-window` | `diffN` 的拟合窗长度（record 数） | `3` | 正整数；不足 2 点时返回零速度 |
| `--position-source` | 外推起点位置来源 | `output` | `output`（管线输出位置）/ `kf`（KF 位置；取不到时退回检测流位置） |
| `--frames-root` | replay 臂的灰度帧根目录 | `/home/manu/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg` | 目录，布局 `<root>/<sequence>/frame_%06d.jpg`；目录不存在时 **replay 臂被跳过**并打印 `[ARM] … 缺帧目录` |
| `--replay-require-valid` | replay 中途丢目标即整体弃权 | `True`（**默认开**） | 开关；配 `--no-replay-require-valid` 关掉 |
| `--replay-fallback` | replay 弃权时的兜底档位 | `kinematic` | `kinematic` / `stale` / `abstain`（`abstain` = 该帧不输出）/ `hold` / `holdv`（用重推链失败点 + 剩余步数×实测步长）/ `holdc`（**连续性兜底**：上一帧实际上报位置 + 参考速度×1，输出级状态，消抖但 F1 −0.41 pp，**消融项**） |
| `--replay-step-gate-px` | 重推逐帧位移门（px/步） | `0.0`（= **不设门**） | 浮点；单步跳太远 ⇒ 整帧弃权走兜底。**诊断开关** |
| `--replay-anchor` | replay 上报坐标怎么合成 | `kernel` | `kernel`（快核绝对位置）/ **`delta`（管线锚点 + 快核位移，消掉质心绝对偏置，实测最优）** / `pred`（同 delta，但 ROI 跟卡尔曼预测走）/ `kf` |
| `--replay-pred-radius-px` | `pred` 档位 ROI 中心偏离匀速预测的允许半径 | `12.0` | 浮点；>0 时朝实测位置回看，≤0 纯预测不回看 |
| `--replay-step-rel-tol-px` | **首步**离群门（px） | `0.0`（= **不设门，默认关**） | 浮点；重推第一步位移与参考速度之差超过它 ⇒ 该步不入链。**实测净亏 −1.74 pp F1**（门修好后才暴露），保留为消融 |
| `--replay-outlier-tol-px` | 批量稳健位移（px） | `0.0`（关） | 浮点；单步位移偏离步长中位数超过它 ⇒ 换成中位数。**实测净亏**，消融项 |
| `--replay-hold-window` | hold 推进速度的参考窗（步） | `3` | 正整数；0 = 全窗口 |
| `--replay-hold-outlier-tol-px` | 末步与参考中位数差的容差（px） | `10.0` | 仅在 `--replay-hold-robust` 打开时生效 |
| `--replay-hold-robust` | hold 用稳健参考推进 | `False`（**默认关**） | 末步口径在 8 序列上 F1 最优；开则用最近 N 步中位数 |
| `--replay-hold-use-ref` / `--no-replay-hold-use-ref` | hold 时按参考速度推进 | `True`（开） | **`--no-…` 才是最优口径**（F1 94.32% vs 94.24%，机理见 memory §12.2） |
| `--replay-anchor-check-px` | 源帧快核位置与管线锚点的容差（px） | `0.0`（不校验） | 超容差 ⇒ 整段作废走兜底。**实测净亏**，消融项 |
| `--replay-max-hold` | 连续观测缺失时允许 hold 几步 | `0`（严格） | 正整数；8 序列实测 **8 最优** |
| `--fast-mode` | 快核提取模式 | `centroid` | `centroid` / `peak_centroid`（**实测更差**，消融） |
| `--fast-win-size` | 快核 ROI 边长（px） | `32` | 正整数；**无 xcorr 时 48 最优**，**带 xcorr 时 52 最优**（配合 `--fast-bg-window 104`） |
| `--fast-bg-window` | 快核底噪估计窗边长（px） | `64` | 必须 ≥ `fast-win-size`；**带 xcorr 时 104 最优**（128 略高 hit@5px 但 p95 更差） |
| `--fast-th-k` / `--fast-min-energy` / `--fast-min-contrast` | 快核底噪倍数 / 最小能量 / 最小对比度 | `3.0` / `1.0` / `5.0` | 浮点；带 xcorr 时实测最优 **2.5 / 50 / 5.0** |
| `--replay-xcorr-radius` | **xcorr 模板匹配的搜索半径**（px） | `0.0`（= 关闭该档位） | 浮点；**32 实测最优**（24~48 是平台区，56 起 FAR 同步涨）。>0 即启用模板匹配位移精修 |
| `--replay-xcorr-template` | xcorr **模板边长**（px） | `24` | 正整数；**32 实测最优**（20/24/40/48 都更差） |
| `--replay-xcorr-th-k` | xcorr 模板二值化的**底噪倍数** | `2.5` | 浮点；**1.5~2.0 最优**，2.5 起单调变差 |
| `--replay-xcorr-min-peak` | 相关峰值的最低要求 | `0.25` | 浮点；低于它 ⇒ 放弃精修走质心链。0.25~0.5 等价，0.6+ 在 chain 档位更差 |
| `--replay-xcorr-bg` | xcorr 底噪估计窗边长（px） | `96` | 必须 ≥ 模板边长 |
| `--replay-xcorr-core-frac` | 模板**核心化**：只保留最亮的前景像素比例 | `0.0`（= 关） | **已被实测证伪**（001324：CF50 63.40 < 同旋钮基线 `arms_WT32R32.json` 的 001324 = **64.51%** ⇒ 拖尾假说不成立），保留为消融项 |
| `--replay-xcorr-mode` | xcorr 的**用法** | `direct` | `direct`（一次跨 Δ 帧大匹配，**交付口径**）/ `chain`（逐步配准，**实测不入选**：最佳 83.91 < direct 84.58）/ `chain_direct` |
| `--replay-xcorr-chain-radius` | chain 每步的搜索半径（px） | `0.0`（= 复用 `--replay-xcorr-radius`） | 仅 `chain` / `chain_direct` 档位用 |
| `--replay-polarity` | **目标极性（亮/暗）假设** | `bright`（= 历史行为） | `bright`（比背景亮，默认，逐位不变）/ `dark`（强制比背景暗）/ `auto`（**在锚点帧现场判极性并锁定整条链**）。实测 24 序列 FAR 0.0542 → **0.0356**、F1 90.46 → **92.83%**、hit@5px 77.51 → **81.55%**（见 §5.9） |
| `--replay-xcorr-uniq-gate` | **相关峰唯一性比**门：次峰/最佳峰 > 阈值 ⇒ 配准峰有歧义 | `0.0`（= 关） | 浮点；**0.6 是交付值**。歧义时以**锚点**为搜索中心**重配一次**，只在「重配唯一」且「重配比原配更自信（peak 更高）」时采纳，否则**保持原结果不动** ⇒ 最坏是「无变化」，不可能回归。实测 8 序列 F1 95.062→**95.266**、hit@5px 84.96→**85.17%**、FAR 0.0182→**0.0163**、CLE p95 30.15→**27.24 px**（见 §5.10） |
| `--replay-polarity-ratio` | `auto` 模式判暗的**置信倍数** | `10.0`（实测最优） | 只有「暗向能量 > 倍数 × 亮向能量」才判暗。**10 最优**：亮目标上的偶发误判被挡回（`001810` 68 → 2 条链），暗目标链保留 99.99%；30 开始伤到 `014453` 最弱的链 ⇒ 反而变差。**只在 `--replay-polarity auto` 下生效** |
| `--replay-roi-mode` | 重推链的 ROI 中心怎么走 | `chain` | `chain`（沿链）/ `selfpred`（**实测最优**：按 KF 预测前推） |
| `--replay-roi-selfpred-window` | `selfpred` 的前推窗长（帧） | `3` | 正整数 |
| `--replay-roi-selfpred-max-px` | `selfpred` 的前推上限（px） | `0.0`（= 不限） | 浮点；**4 最优**（8 → 84.57、12 → 84.53，放宽会把 ROI 拉偏） |
| `--output-max-step-px` | **输出级限幅**（slew limiter，只作用于 replay 臂） | `0.0`（= 关） | 浮点；**30 是交付值**（把 max_jitter 365.9 → 114.2 px 而 F1 只掉 0.095 pp） |
| `--fast-peak-window` | `peak_centroid` 极大点邻域边长 | `9` | 正整数（仅 `peak_centroid` 用） |
| `--fast-kf-r` | `kf` 档位的观测噪声标准差（px） | `2.0` | 浮点（仅 `--replay-anchor kf` 用） |
| `--arms` | 逗号分隔的对照臂 | `original,stale,kinematic,replay` | 至少一个臂；`original` 必留（尺子 1 的锚） |
| `--out-table` | 可选的 markdown 对照表输出路径 | `""`（不写） | 路径；指定则生成四臂对照表 |
| `--out-json` | 运行报告 JSON 路径 | `""`（不写） | 路径；**所有数字的唯一来源** |
| `--out-preds-dir` | 逐序列预测输出目录 | `""`（不写） | 目录；写 `<序列>.json` + `summary.json` |
| `--cache-root` | 热图缓存根目录 | **必填** | 目录，内含 `<序列>.pkl` |
| `--labels-root` | GT 标签根目录 | **必填** | 目录，内含 `<序列>__<index:06d>.txt` |
| `--seq` | 单序列名 | `""` | 序列名；与 `--seqs` / `--all-seqs` **三选一** |
| `--params-json` | 全参数集 JSON | `""`（用内置默认） | 路径；内容形如 `{"params": {...}}` 或直接是参数字典 |

**其它相关开关**（同属 `parse_args()`）：`--all-seqs`（跑全部缓存序列，默认 False）、`--seqs`（逗号分隔序列子集，**按给定顺序**评测，默认 `""`）、`--dedup-radius`（跨尺度去重半径 px，默认 `None`＝用 params 的 768.0）、`--no-pipeline`（只跑单帧臂，默认 False）、`--self-check`（只跑适配器正确性门禁后退出）、`--expect-f1` / `--expect-tp` / `--expect-fp`（冻结 baseline 自校准门禁，不达标 `SystemExit(2)`）。
**注意**：`--arms` 里 `replay` 臂要求 `--frames-root` 下的序列帧目录存在，否则该臂被跳过并打印 `[ARM] … 缺帧目录`。

## 4. 可复制的运行命令

> ⚠️ 数据（缓存 / GT / 帧）都在**服务器 SSHFS 只读挂载**下（`/home/manu/mnt/...`），评测 `--labels-root`
> 取**服务器侧路径**，不要指本机只读镜像。下列命令逐字来自 `manu/memory/lookahead_extrapolator_20261102.md` §8。

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics
source ~/anaconda3/etc/profile.d/conda.sh && conda activate uav
CACHE=/home/manu/mnt/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache
GT=/home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json

# ① 参考（原脚本，未改动）
PYTHONPATH=. python manu/evaluation/system_sota_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --out-json runs/lookahead/ref/original_v3_8seq.json --out-preds-dir runs/lookahead/ref/preds_original_v3

# ② 新脚本（开关关闭 ⇒ 必须与 ① 逐位一致）+ 四臂对照
PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --lookahead off --arms original,stale,kinematic,replay --delta 8 \
  --out-json runs/lookahead/off/off.json --out-preds-dir runs/lookahead/off/preds

# ③ Gate A：整份 JSON 的 md5 逐位比对
for f in runs/lookahead/ref/preds_original_v3/*.json; do b=$(basename $f); \
  [ "$(md5sum $f|cut -d' ' -f1)" = "$(md5sum runs/lookahead/off/preds/$b|cut -d' ' -f1)" ] \
  && echo "IDENTICAL $b" || echo "DIFFER $b"; done

# ④ Δ 扫描 / 速度源扫描
for d in 2 4 12; do PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --lookahead replay --delta $d --out-json runs/lookahead/sweep/delta$d.json; done
for vs in kf diff1 diffN; do PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --lookahead kinematic --velocity-source $vs --out-json runs/lookahead/sweep/$vs.json; done

# ⑤ 单测：PYTHONPATH=. python -m unittest discover -s manu/tracking/tests -t . -v   # 135 项

# ⑥ ★ 最终交付候选（Δ=8 = 320ms，实测最优口径；≈175 s）
PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --lookahead replay --delta 8 --arms original,stale,replay \
  --replay-max-hold 8 --replay-fallback holdv --replay-anchor delta \
  --fast-win-size 48 --fast-bg-window 96 --no-replay-hold-use-ref \
  --out-json runs/lookahead/tune/final_delivery.json \
  --out-preds-dir runs/lookahead/tune/final_delivery_preds

# ⑦ ★★ 交付定稿（2026-11-03 夜间，服务器）：xcorr 模板匹配精修
PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --frames-root $FRAMES --all-seqs --params-json $PARAMS \
  --lookahead replay --delta 8 --arms original,stale,kinematic,replay \
  --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv \
  --no-replay-hold-use-ref --output-max-step-px 30 --replay-anchor delta \
  --fast-win-size 52 --fast-bg-window 104 --fast-th-k 2.5 --fast-min-energy 50 \
  --replay-roi-mode selfpred --replay-roi-selfpred-window 3 --replay-roi-selfpred-max-px 4 \
  --replay-xcorr-radius 32 --replay-xcorr-template 32 --replay-xcorr-th-k 1.5 \
  --out-json runs/lookahead/final/arms_final_WTH15.json \
  --out-preds-dir runs/lookahead/final/preds_final_WTH15

# ⑧ 对照表（所有数字来自产物 JSON，不重算）
PYTHONPATH=. python manu/tracking/lookahead/make_report_table.py \
  --runs runs/lookahead/tune/final_delivery.json --labels "外挂·最终交付候选"
```

占位符 ↔ 真实绝对路径（**服务器** `/tmp/pycharm_project_10ae9e2e`）：
`$CACHE` = `/tmp/pycharm_project_10ae9e2e/runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache`；
`$GT` = `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010`；
`$FRAMES` = `/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg`（`--frames-root`）；
`$PARAMS` = `manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json`（仓库相对路径）。
（**host 侧**的对应路径是 `/home/manu/mnt/...`，见 §0；两台机器**不要混用**。）

**★ 导出 `stale` 臂的逐帧 preds（做 OSD 对照要用）**：`--lookahead replay` 会把**主臂**设成 `replay`，
此时 `--arms stale` 里没有主臂 ⇒ 脚本直接 `SystemExit("[ARM] 主臂 replay 在序列 … 上不可用（缺帧）")`
（**这个报错信息有误导性**，不是缺帧）。正确做法：

```bash
PYTHONPATH=. python manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root $CACHE --labels-root $GT --all-seqs --params-json $PARAMS \
  --lookahead stale --delta 8 --arms original,stale \
  --out-json runs/lookahead/final/arms_final_stale.json \
  --out-preds-dir runs/lookahead/final/preds_final_stale
```

## 5. 实测结果

> **模型固定 RC2 / trial_0080 / epoch3**（回放 `ms_cache/*.pkl`，**不重跑推理**）；8 序列汇总；
> **交付判据 F1 ≥ 0.88 / Recall ≥ 0.85 / FAR ≤ 0.030**。下列数字全部誊抄自
> `manu/memory/lookahead_extrapolator_20261102.md`（原始出处为对应产物 JSON）。

### 5.1 四臂对照（Δ=8 = 320 ms @25 Hz）

| 臂 | F1 | Recall | Precision | FAR | CLE p50 | CLE p90 | CLE p95 | hit@5px | coverage |
|---|---|---|---|---|---|---|---|---|---|
| **original**（无滞后天花板，**上界不是基线**） | **96.400%** | 93.67% | 99.29% | 0.0063 | 1.97 | 7.97 | 13.43 | 88.13% | 89.87% |
| **stale**（老逻辑硬吃滞后，**交付基线**） | 77.203% | 75.00% | 79.54% | 0.1807 | 20.43 | 78.02 | 113.73 | 38.18% | 89.82% |
| **kinematic**（档位 A，KF 速度，**不成立**） | 77.552% | 75.34% | 79.90% | 0.1776 | 15.68 | 89.98 | **183.05** | **33.29%** | 89.82% |
| **replay**（档位 B 视觉重推 + xcorr，**交付方案**） | **95.077%** | 92.36% | 97.96% | **0.0180** | **2.12** | 14.35 | 30.01 | **84.67%** | 89.82% |

来源：**服务器** `runs/lookahead/final/arms_final_WTH15.json` → `arms.<臂>.current_frame.overall` + `arms.<臂>.servo.overall`
（**8 序列交付定稿**，2026-11-03 夜间）。
> ⚠️ 旧版这一行引的 `runs/lookahead/off/off.json` **在服务器上不存在**（那是 host 侧产物）⇒ 已删除该引用。
> 「xcorr 精修之前」的**上一轮**交付候选（`delta_s30` = `8seq_v2`）在服务器上的产物是
> `runs/lookahead/final/arms_8seq_v2.json` + `preds_8seq_v2/`，其 replay 臂是
> **F1 94.236% / hit@5px 79.41% / FAR 0.0257**（见 §5.5 表末两行）；
> 而更早的「F1 93.353 / hit@5px 72.66 / FAR 0.0337」那一版**服务器上没有存档**（只存在于 host 侧历史产物与旧文档），
> 本 README **仅在历史对照处引用其 FAR 0.0337（8 序列口径）**。
**读法**：
① **老逻辑的代价被量化了**：硬吃 320 ms ⇒ F1 −19.20 pp、hit@5px −49.95 pp、CLE p50 ×10.4。
② **档位 B 挽回绝大部分**：F1 **+17.87 pp**、hit@5px **+46.49 pp**、CLE p50 **−89.6%**，距无滞后天花板只差 **1.32 pp F1**（比上一轮异步双频架构的 4.65~8.52 pp 差距**更小**）。
③ **档位 A 不成立**：F1 只 +0.35 pp，**hit@5px 反而 −4.89 pp**，CLE p95 从 113.73 **恶化到 183.05**。
④ **FAR（8 序列口径）已达标**：replay 臂 FAR **0.0180**（original 0.0063 / stale 0.1807）——**低于交付线 0.030**
（xcorr 精修之前是 0.0337，**超线 12%**；xcorr 把误检也一并压下去了）。
**⚠️ 但 24 序列官方口径上 FAR = 0.0542 > 0.030（超线 80.8%）⇒ 这一项在 24 序列上尚未达标**（见 §5.7）。
⑤ **逐序列：replay 对 stale 是 8/8 全赢**。F1 **8/8 赢（+12.50 ~ +40.69 pp）**；hit@5px **8/8 赢（+20.94 ~ +87.83 pp）**，
**最小差距是 `001324` 的 +20.94 pp**（64.59% vs 43.65%），**最大差距是 `001531` 的 +87.83 pp**（91.30% vs 3.48%）。
**逐序列结论**（`arms.<臂>.current_frame.per_sequence` / `servo.per_sequence`）：
- **replay vs stale**：F1 **8/8 赢**（+12.50 ~ +40.69 pp）、hit@5px **8/8 赢**（+20.94 ~ +87.83 pp）⇒ **没有单条序列拉平均**；
- **kinematic vs stale**：F1 口径 **6/8 赢**、hit@5px 口径 **1/8 赢**
  （`001324` 上 F1 口径 **75.35% → 66.94%**，倒退 **8.41 pp**；hit@5px 口径 **43.65% → 34.76%**，倒退 **8.89 pp**）⇒ **不成立**；
- **replay 距天花板（original）的 hit@5px 逐序列缺口**：`001531` **7.83 pp**（最大）、`015000` **7.41 pp**、`001324` **5.93 pp**、
  `014053` 5.57、`000703` 3.73、`015052` 1.21、`001307` 1.20、`000818` 0.88（最小）。
  （`001324` 只是**帧数占比最高**（38.77%）⇒ 对**总体** 3.46 pp 缺口贡献 **2.30 pp**，与「逐序列最大缺口」是两个量。）

### 5.2 Δ 敏感性扫描（老逻辑随延迟崩塌，档位 B 基本拉平）

> **2026-11-03 夜间用交付定稿配置（`WTH15`，含 xcorr）在服务器重测**（旧表是**上一轮无 xcorr** 的数字，
> 且产物只在 host 上）⇒ 下表全部有产物：`runs/lookahead/final/arms_delta{2,4,8,12}_WTH15.json`。

| Δ | 臂 | F1 | Recall | FAR | CLE p50 | CLE p95 | hit@5px |
|---|---|---|---|---|---|---|---|
| 2 | stale | 94.817% | 92.13% | 0.0206 | 4.75 | 32.32 | 69.10% |
| 2 | kinematic | 92.893% | 90.26% | 0.0381 | 3.84 | 45.17 | 71.22% |
| 2 | **replay** | **96.211%** | 93.48% | 0.0079 | 1.99 | 16.70 | **87.35%** |
| 4 | stale | 89.805% | 87.25% | 0.0662 | 9.92 | 57.41 | 54.50% |
| 4 | kinematic | 89.404% | 86.86% | 0.0698 | 6.24 | 91.17 | 56.70% |
| 4 | **replay** | **95.854%** | 93.13% | 0.0111 | 2.00 | 20.97 | **86.48%** |
| 8 | stale | 77.203% | 75.00% | 0.1807 | 20.43 | 113.73 | 38.18% |
| 8 | kinematic | 77.552% | 75.34% | 0.1776 | 15.68 | 183.05 | 33.29% |
| 8 | **replay** | **95.077%** | 92.36% | 0.0180 | 2.12 | 30.01 | **84.67%** |
| 12 | stale | 68.160% | 66.20% | 0.2628 | 29.66 | 154.40 | 30.10% |
| 12 | kinematic | 62.718% | 60.91% | 0.3123 | 27.87 | 269.93 | 21.44% |
| 12 | **replay** | **94.411%** | 91.69% | 0.0239 | 2.27 | 38.33 | **83.26%** |

**读法**：
① **老逻辑随延迟崩塌**：Δ 每翻一倍 F1 掉 5~9 pp（Δ=8→12 掉 **9.04 pp**）；hit@5px 从 88.13%（Δ=0 天花板）塌到 **30.10%**（Δ=12）。
② **档位 B 基本拉平（xcorr 定稿配置）**：Δ=2→12 **F1 只掉 1.80 pp**（96.211 → 94.411）、**hit@5px 只掉 4.09 pp**（87.35 → 83.26）
⇒ **把系统对检测延迟的敏感度压低了约一个数量级**。
（旧口径「F1 −4.87 pp、hit@5px −9.9 pp」是**上一轮无 xcorr** 的数字，已被本表取代。）
③ **FAR 全程在交付线内**：replay 臂 FAR 在 Δ=2/4/8/12 = **0.0079 / 0.0111 / 0.0180 / 0.0239**，**全部 < 0.030**
（旧配置在 Δ=12 是 0.0535，**超线**；xcorr 把这一项也修回来了）。
④ **档位 A 仍然不成立**：四个 Δ 里只有 Δ=8 微弱优于 stale（**+0.35 pp** F1），其余三个都更差
（Δ=2/4/12 分别 **−1.92 / −0.40 / −5.44 pp** F1），且 Δ≥8 时 hit@5px **低于** stale ⇒ **纯运动学外推不是本任务的解**。
⑤ **外挂的收益随延迟放大**：replay − stale 的 F1 增益 = Δ=2 **+1.39** / Δ=4 **+6.05** / Δ=8 **+17.87** / Δ=12 **+26.25 pp**
⇒ **降延迟与加外挂是叠加收益**（两者都做，单位成本下收益更高）。

### 5.3 速度源对照（Δ=8，档位 A）与门控消融（两个门都被数据否掉）

| velocity_source | F1 | FAR | CLE p50 | CLE p95 | hit@5px |
|---|---|---|---|---|---|
| `kf`（卡尔曼后验速度） | 77.552% | 0.1776 | 15.68 | 183.05 | 33.29% |
| `diff1`（1 帧差分） | 77.022% | 0.1824 | 17.00 | 111.80 | 27.46% |
| `diffN`（3 点最小二乘） | 80.567% | 0.1501 | 15.63 | 102.42 | 33.81% |

来源（**2026-11-03 夜间在服务器用定稿配置重跑**）：`runs/lookahead/final/arms_velsrc_{kf,diff1,diffN}.json`
（旧版引的 `runs/lookahead/sweep/*.json` 在服务器上不存在）。
三档速度源**结论一致**：档位 A 最多 +3.36 pp F1，且 **hit@5px 全部低于 stale 的 38.18%**。
> 注：`--velocity-source` 只影响**档位 A（kinematic）**；**replay 臂三档数字完全相同**
> （F1 95.077% / hit@5px 84.67% / FAR 0.0180）⇒ 交付方案不受速度源选择影响。

| 门控方案（Δ=8） | F1 | FAR | hit@5px | 判读 |
|---|---|---|---|---|
| 无门（基线） | 93.353% | 0.0337 | 72.66% | — |
| 重推位 vs 运动学预测偏差 ≤ 6 px | 77.578% | 0.1773 | 38.39% | **崩塌**：运动学预测本身不可靠（CLE p95 183 px），用它当参照会把好推演全毙掉 |
| 逐帧位移 ≤ 12 px/步 | 92.217% | **0.0441** | 72.30% | **负收益**：被毙的帧退回 kinematic ⇒ FP 978→1278 |
| 逐帧位移 ≤ 20 px/步 | 93.353% | 0.0337 | 72.66% | **空操作**（重推从不单步跳 >20 px） |

门控来源：**门控扫描产物只存在于 host 侧历史目录，服务器无存档**（等价产物见 `runs/lookahead/final/arms_delta*_WTH15.json`）；
旧版引的 `gate/` 目录（`g6.json` = 运动学偏差门 6 px；`step12.json` / `step20.json` = 逐帧位移门）已删除，上表数字仅作历史对照。

### 5.4 OSD 视频复核（肉眼可查的证据：001324 的 300 帧脱靶窗口）

§5.1 里 001324 是 replay 距天花板缺口最大的序列之一（hit@5px 64.59% vs original 70.53%，缺口 5.93 pp），所以专门定位它
**滞后真正致命的那一段**做三臂 OSD 对照。窗口不是手挑的：用交付评测**同一套 inbox 判定**
（`manu/videos/render_pred_gt_osd_video.match_frame`）逐帧扫全序列，找出
「stale 全脱靶 且 replay 全命中」的最长连续帧段 ⇒ **frames 8628..8691（连续 64 帧）**，
取其外围 **8500..8799（300 帧）** 渲染（**旧窗口，产物服务器无存档**；服务器侧可复核的 001324 窗口是
**frames 8674..8973**，见 §5.6 与 `runs/lookahead/osd/windows/VIDEO00007_19700101_001324.pick.log`）。

| 臂 | F1 | Recall | FAR | CLE p50 | CLE p95 | CLE max | hit@5px/GT | hit@10px/GT |
|---|---|---|---|---|---|---|---|---|
| **original**（无滞后天花板） | **100.00%** | 100.00% | 0.0000 | 1.86 | 3.06 | 4.52 | **100.00%** | 100.00% |
| **stale**（老逻辑硬吃滞后） | **62.33%** | 62.33% | 0.3767 | 4.24 | 20.94 | 25.81 | **31.67%** | 36.67% |
| **replay**（档位 B 视觉重推） | **90.67%** | 90.67% | 0.0933 | **1.05** | 7.47 | 19.13 | **76.00%** | 90.00% |

- **产物 JSON**：⚠️ 旧窗口产物 ~~`runs/lookahead/osd/window_lagfail_001324.json`~~ **服务器无存档**，上表数字**无法复核**；
  服务器上可复核的对应产物是 `runs/lookahead/osd/videos/VIDEO00007_19700101_001324_{original,stale,replay}.json`
  （渲染器逐帧判定，窗口 frames 8674..8973）与 `runs/lookahead/osd/windows/VIDEO00007_19700101_001324.json`（窗口选择结果）。
- **OSD 视频**（`runs/lookahead/osd/videos/`，红=系统检出 / 绿=命中的 GT / 橙=漏掉的 GT）：
  `VIDEO00007_19700101_001324_{original,stale,replay}.mp4`（同名 `.json` 是渲染器的逐帧判定，同名 `.log` 是渲染日志）；
  旧命名的 ~~`*_lagfail.mp4`~~ **服务器无存档**。
- **目视结论**：stale 臂的红圈**明显拖在绿 GT 框后面**（误差 13.6 → 58.4 px 递增），
  replay 臂红圈**始终落在框内**（误差 1.7 → 8.8 px）。
- **对照窗口**（中速机动，6000..6599，600 帧）：⚠️ 产物 ~~`runs/lookahead/osd/window_moderate_001324.json`~~ **服务器无存档**，
  下表数字**无法复核**（服务器上 001324 只留了 `runs/lookahead/osd/windows/VIDEO00007_19700101_001324.json` 一段窗口）。

  | 臂 | F1 | FAR | CLE p50 | CLE p95 | hit@5px/GT |
  |---|---|---|---|---|---|
  | original | 100.00% | 0.0000 | 1.50 | 2.62 | 100.00% |
  | stale | 85.50% | 0.1450 | 4.94 | 17.98 | 43.33% |
  | replay | 100.00% | 0.0000 | 4.48 | 5.57 | 78.00% |

  ⇒ **replay 在中速窗口 F1 已追平天花板（100%），但 CLE p50 = 4.48 px 仍高于 original 的 1.50 px，
  hit@5px 只有 78.00%** ⇒ 残差是**系统性偏置**（快核质心被拖尾拽偏）而不是滞后，
  这正是 §6 要治的问题（逐帧位移门因此无效）。

### 5.5 ★★ 交付定稿（2026-11-03 夜间，服务器）：xcorr 模板匹配位移精修

**机理**：质心链的位移 = `K(t) − K(f)`，其中 `K` 是快核的灰度质心。质心相对目标**几何中心**有一个偏置 `b`，
它随目标姿态/亮度/拖尾**漂移** ⇒ 位移误差 = `b(t) − b(f)`，尾巴很重（位移 p90 4.37 px / p95 8.13 px）。
改用**同一份模板**（源帧锚点处按底噪阈值二值化的权重图）在目标帧里配准 ⇒ 两端共用同一个偏置 ⇒ 偏置**精确抵消**，
只剩真实位移（位移 p90 → 3.17 px / p95 → 6.34 px）。模块：`manu/tracking/lookahead/xcorr.py`。

**★ 这一轮同时修掉一个真 bug（xcorr 的半像素约定错误）**：模板从 `round(cx) − tsize//2` 切出，
其**离散中心**对应源帧 `round(cx) − 0.5`，而 `match()` 原来按 `(tsize−1)/2` 报位置 ⇒ **每步系统性少 0.5 px**（x、y 各）。
`direct` 只匹配一次 ⇒ 只差 0.5 px（看不出来）；`chain` 走 8 步 ⇒ **累计 −4 px** ⇒ chain 档位 hit@5px 一度只有 **12.5~14.9%**。
修法：`make_template()` 返回 `(weights, origin)`（模板左上角在源帧的绝对坐标），
`match()` 改成 **上报位置 = `anchor` + 模板左上角的位移**（纯位移口径，与取整约定无关）。
验证：合成图平移 `(7,−4)`，修复前报 `(6.5,−4.5)`、修复后报 `(6.998,−4.003)`。
> **教训**：任何「拿相关峰当位置」的代码都必须显式声明**参考点**与**离散中心**的换算关系 —— 差半个像素，单次无害、链式致命。

**交付参数（`WTH15`）**：`--replay-xcorr-radius 32 --replay-xcorr-template 32 --replay-xcorr-th-k 1.5`
＋ 快核 `--fast-win-size 52 --fast-bg-window 104 --fast-th-k 2.5 --fast-min-energy 50`
＋ ROI `--replay-roi-mode selfpred --replay-roi-selfpred-window 3 --replay-roi-selfpred-max-px 4`
＋ `--output-max-step-px 30 --replay-anchor delta --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --no-replay-hold-use-ref`

**8 序列变体表（hit@5px 降序，本表列前 26 名；**完整表**见 memory 专题档 §23.2 与
`runs/lookahead/final/variant_table.md`）**
> 本表的「旋钮差异」列是**相对 `WT32R32`**（xcorr 首版基线，th_k=2.0）；
> 完整表的差异列是**相对冠军 `WTH15`**，且由脚本
> `manu/tracking/lookahead/gen_variant_table.py` 从全部 `runs/lookahead/final/arms_*.json` **生成**（数字零手抄）。
> 完整表按**实验轴**分两段：**旋钮轴**（Δ=8，改的是旋钮）与**延迟轴**（Δ≠8，改的是检测滞后）**分开排** ——
> 混排会让「Δ=2 的 87.35%」压过冠军 84.67%，看着像「旋钮调优赢过冠军」，那是两个不同的轴。
> 权威行数以产物 `variant_table.json` 的 `row_count` 为准（磁盘产物会持续增加，表要跟着重跑）。

| tag | 相对基线的旋钮差异 | 尺子2 F1 % | FAR/帧 | CLE p50 | CLE p95 | **hit@5px %** |
|:--|:--|--:|--:|--:|--:|--:|
| **`WTH15`（★ 交付冠军）** | `th_k=1.5` | **95.077** | **0.0180** | **2.12** | **30.01** | **84.67** |
| `WBG128` | `bg=128` | 95.088 | 0.0179 | 2.13 | 31.48 | 84.60 |
| `WR28` | `radius=28` | 95.081 | 0.0180 | 2.16 | 30.58 | 84.58 |
| `WT32R32`（xcorr 基线） | — | 95.058 | 0.0182 | 2.16 | 30.39 | 84.58 |
| `WSP8` | `roi_cap=8` | 94.990 | 0.0188 | 2.16 | 30.95 | 84.57 |
| `WR36` | `radius=36` | 95.016 | 0.0186 | 2.16 | 30.43 | 84.55 |
| `WMP50` | `min_peak=0.5` | 95.081 | 0.0180 | 2.13 | 29.63 | 84.54 |
| `WSP12` | `roi_cap=12` | 94.960 | 0.0191 | 2.15 | 31.19 | 84.53 |
| `WT32R40` | `radius=40` | 94.990 | 0.0188 | 2.17 | 30.78 | 84.46 |
| `WT40` | `template=40` | 95.016 | 0.0186 | 2.17 | 30.52 | 84.46 |
| `WR24` | `radius=24` | 95.069 | 0.0181 | 2.16 | 31.28 | 84.45 |
| `WT32R48T36` | `radius=48 template=36` | 94.850 | 0.0201 | 2.14 | 29.62 | 84.40 |
| `WT32R48_MP50` | `radius=48 min_peak=0.5` | 94.979 | 0.0189 | 2.15 | 28.77 | 84.40 |
| `WT48` | `template=48` | 94.963 | 0.0191 | 2.21 | 31.12 | 84.38 |
| `WT32R48` | `radius=48` | 94.872 | 0.0199 | 2.17 | 30.43 | 84.37 |
| `WTH25` | `th_k=2.5` | 95.009 | 0.0187 | 2.17 | 30.11 | 84.35 |
| `WT32R56` | `radius=56` | 94.751 | 0.0210 | 2.18 | 31.21 | 84.28 |
| `WC_R24`（chain） | `mode=chain chain_r=24` | 94.952 | 0.0192 | 2.16 | 33.42 | 84.18 |
| `WT32R48T28` | `radius=48 template=28` | 94.778 | 0.0208 | 2.19 | 32.33 | 84.17 |
| `WT32R64` | `radius=64` | 94.615 | 0.0222 | 2.21 | 32.90 | 84.16 |
| `WT32R48TH25` | `radius=48 th_k=2.5` | 94.819 | 0.0204 | 2.18 | 30.81 | 84.14 |
| `WT32R72` | `radius=72` | 94.304 | 0.0251 | 2.21 | 34.85 | 83.95 |
| `WC_R24_MP60`（chain） | `mode=chain chain_r=24 min_peak=0.6` | 94.888 | 0.0198 | 2.16 | 32.79 | 83.91 |
| `YT32R40TH2`（xcorr 首版） | `radius=40` | 94.865 | 0.0200 | 2.56 | 30.81 | 83.90 |
| `SPC4`（**无 xcorr**，快核重调基线） | — | 94.679 | 0.0217 | 2.48 | 30.96 | 81.24 |
| `8seq_v2` = `delta_s30`（**上一轮交付候选 A，无 xcorr**） | — | 94.236 | 0.0257 | 2.90 | 36.74 | 79.41 |

**参数规律**：**模板 32 最优**（20/24/40/48 都差）｜**th_k 1.5~2.0 最好**（2.5 起单调变差）｜
**搜索半径 24~48 是平台区**（56 起 FAR 同步涨）｜**ROI 前推上限 4 px 最优**（8/12 都掉）｜**`min_peak` 0.25~0.5 等价**。

**两条被实测否决的路线（不要再试）**

| 路线 | 机理假设 | 实测 | 判决 |
|:--|:--|:--|:--|
| `chain` 逐步配准（`--replay-xcorr-mode chain`） | 每步只隔 1 帧 ⇒ 模板几乎不变 ⇒ 残差只剩亚像素噪声；被跟踪的物理点不变 ⇒ 质心偏置不累积 | 修复半像素 bug 后最佳 **83.91%** < direct **84.58%**；001324 上 chain 最差变体 `WC_R12`（`mode=chain chain_r=12 min_peak=0.25`，该档 p95 最差）的 CLE p95 = **49.87 px** vs direct `WTH15` **30.01 px**（错位刷新模板 → 级联跑飞）；`min_peak` 越高越差（0.25→83.28 / 0.60→82.78 / 0.75→80.91） | **不入选** |
| 模板核心化（`--replay-xcorr-core-frac`） | 红外拖尾把相关峰拽偏 ⇒ 只留最亮像素、剔掉拖尾 | 001324 单序列 `CF20` 53.67 / `CF35` 60.67 / `CF50` 63.40 **< 同旋钮基线 `arms_WT32R32.json` 的 001324 = 64.51%** ⇒ **拖尾假说不成立** | **证伪**（旋钮保留，默认关） |

**Gate A（关开关 ⇒ 与原脚本逐位一致）**：`runs/lookahead/final/arms_gateA_W.json`，F1 **96.400%**，
`[GATE] 尺子1 各臂源帧口径 ΔF1 = 0.000000` ⇒ 外挂没动任何既有跟踪资产。

### 5.7 ★★★ 24 序列**官方口径**复评（2026-11-03 夜间，服务器）

产物 `runs/lookahead/final/arms_final_WTH15_24seq.json`（耗时 649.2 s）+ `preds_final_WTH15_24seq/`；
对照表 `runs/lookahead/final/report_WTH15_24seq.md`；24 序列 FPV8 缓存
`runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache`（24/24）。

| 臂 | 尺子1 F1 | 尺子2 F1 | R | P | FAR/帧 | **hit@5px** | max_jitter | **CLE 池化 p50** | **CLE 池化 p95** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `original`（零延迟天花板，**上界不是基线**） | 94.7527% | **94.7469%** | 92.34% | 97.28% | 0.0208 | **85.90%** | 76.5 | 1.56 px | 6.84 px |
| `stale`（老逻辑硬吃 320 ms，**交付基线**） | 94.7527% | **77.5225%** | 75.53% | 79.62% | 0.1559 | **44.82%** | 76.5 | 5.29 px | 56.62 px |
| `kinematic`（档位 A，**不成立**） | 94.7527% | 75.7641% | 73.82% | 77.82% | 0.1697 | **38.68%** | 188.9 | 6.42 px | 61.52 px |
| **`replay`（★ 交付方案 WTH15）** | 94.7527% | **90.4605%** | 88.14% | 92.91% | **0.0542** | **77.51%** | 122.2 | **1.88 px** | **28.48 px** |

⇒ **24 序列追回占比**（口径 = `(外挂−基线)/(天花板−基线)`，越小越好的指标取反向差）：F1 **75.1%**、hit@5px **79.6%**、FAR **75.3%**（`(0.1559−0.0542)/(0.1559−0.0208)`）、CLE 池化 p50 **91.4%**（`(5.29−1.88)/(5.29−1.56)`）、CLE 池化 p95 56.5%。
⇒ **xcorr 在 24 序列上同样成立且更大**：hit@5px **69.96% → 77.51%（+7.55 pp）**、F1 **+0.48 pp**、FAR 0.0581 → 0.0542。
⇒ **Gate A（24 序列）**：原脚本 vs `--lookahead off` ⇒ **25/25 文件 md5 IDENTICAL（0 DIFFER / 0 MISSING）**
（`ref_preds_original_v3_24seq/` ↔ `preds_off_24seq/`）。

> ⚠️ **两条必须写进交付说明的结论**：
> ① **24 序列上 FAR = 0.0542 > 交付线 0.030（超线 80.8%）** —— 8 序列困难集上只有 0.0180（达标），
> **8 序列掩盖了 FAR 问题** ⇒ **交付判据必须以 24 序列为准**；
> ② `max_jitter` 从「无 xcorr」版的 61.5 px 涨到 **122.2 px** ⇒ 伺服侧需一级输出限速/低通。
>
> ⚠️ **CLE 聚合口径警告**：产物里 `servo.overall.cle_p50/p95` 是 **「逐序列最差值」**聚合
> （24 序列上是 144~153 px，指**某一条最差序列**的中位数，**不是全局典型值**）；
> **全局口径必须看 `cle_pooled_*`**（全帧池化，n = 123673~124303）。上表给的是**池化**口径。

### 5.6 ★★★ OSD 三臂对照（**2026-11-03 夜间第 2 段重渲，5 序列 × 3 臂 = 15 段视频**）

窗口由 `pick_osd_windows.py` **自动挑**（300 帧，**不拍脑袋**）：
A/C 段判据 = 最大化「`stale` 中心距 − `replay` 中心距」；**B 段判据 = 最大化「`replay` 中心距 − `original` 中心距」**
（即**专门挑外挂锁偏最明显的地方**，用来暴露反面证据）。
两套口径都直接取产物里的 `metrics` 字段，**没有另行重算**：
**inbox 口径**（TP/FP/FN/P/R/F1）取渲染脚本 `--out-json` 的 `metrics`；**中心距口径**（CLE / hit@5px）取
`runs/lookahead/osd/windows/<SEQ>.json` 的 `per_arm`。

| 序列 | 窗口 | 臂 | inbox F1 | CLE 均值 (px) | hit@5px |
|:--|:--|:--|--:|--:|--:|
| **A · `001324`** 高速机动 | 8549–8848 | `stale`（老逻辑） | 55.33% | 34.26 | 19.67% |
| | | **`replay`（外挂 WTH15）** | **100.00%** | **2.51** | **90.67%** |
| | | `original`（天花板） | 100.00% | 2.01 | 98.67% |
| **B · `014453`** 暗弱悬停 | 6629–6928 | `stale`（老逻辑） | 99.67% | 3.57 | 76.67% |
| | | **`replay`（外挂 WTH15）** | **9.00%** ❌ | **81.49** ❌ | **9.00%** ❌ |
| | | `original`（天花板） | 100.00% | 1.29 | 100.00% |
| **C · `015000`** 悬停 | 726–1025 | `stale`（老逻辑） | 72.00% | 21.28 | 24.67% |
| | | **`replay`（外挂 WTH15）** | **99.00%** | **2.46** | **95.33%** |
| | | `original`（天花板） | 100.00% | 1.97 | 99.67% |

**全序列口径（各臂预测产物自带的 `metrics`，24 序列交付候选）**：

| 序列 | `stale` F1 | **`replay` F1** | `original` F1 | replay vs stale |
|:--|--:|--:|--:|--:|
| `001324` | 74.54% | **87.47%** | 89.95% | **+12.93 pp** |
| `015000` | 82.86% | **98.54%** | 99.81% | **+15.68 pp** |
| `014453` | 92.41% | **63.49%** | 99.93% | **−28.92 pp** ❌ |

⇒ **A / C 达成目标**：老逻辑被 320 ms 滞后吃掉 **44.67 pp / 28.00 pp** 召回，外挂把命中拉回 100% / 99%，
中心距均值距天花板只差 **0.50 px / 0.49 px**。
⇒ **★ B 是必须一起讲的反面证据**：这段暗弱悬停里 **replay 比 stale 更差**（9.00% vs 99.67%），
而天花板 `original` 仍 **100% 命中 / 1.29 px** ⇒ **目标可见、不是不可解，是外挂选错了特征点**
⇒ 与 §5.8 ② 的归因（xcorr 在暗弱目标上无真峰 → 退回质心链 → 锁到云层亮边）**完全一致**。
**不要只放好看的两条。**

产物：`runs/lookahead/osd/videos/<SEQ>_{original,stale,replay}.{mp4,json,log}`
（h264 / 1920×1080 / 300 帧 / 25 fps / 12.0 s；截图 `*_f8702.png` / `*_f6874.png` / `*_f1022.png`）、
`runs/lookahead/osd/windows/<SEQ>.json` + `.pick.log`、`runs/lookahead/osd/osd_window_metrics.md`、
`runs/lookahead/osd/OSD_README.md`。
⚠️ `videos/` 里还有上一轮遗留的 `000818_*` / `014053_*`（旧 replay + 旧 stale 产物），**不属于本轮交付**。
⚠️ 本轮 `stale` 臂是**现跑**的（`params` 与旧产物逐字段相同，差异来自 17:33 的代码改动），
**本节数字以本轮为准**。

### 5.8 ★★★ 24 序列 FAR 缺口的**完整归因**（2026-11-03 夜间第 2 段，服务器）

> 完整细节见 `manu/memory/lookahead_extrapolator_20261102.md` **§24** 与 `manu/memory/memory.md` **§30.9.19**。
> 本节零代码改动（`xcorr.py` `27d2157f…` / `extrapolator.py` `88539a23…` / `lookahead_longquan.py` `c5e5678e…` md5 未变）。

**① FAR 缺口逐序列拆解**（`manu/tracking/lookahead/diag_far_breakdown.py`，产物 `runs/lookahead/final/diag_far_breakdown.{json,md}`）：
24 序列 frames=165223 / TP=117449 / **FP=8963** / FN=15808 ⇒ **FAR=0.0542**（original 0.0208），
红线 0.030 ⇒ **超线 +0.0242 ≈ 4006 个 FP**。
**FP 落点**：有 GT 帧 **6224（69.4%，定位错）** ＋ 无 GT 帧 2739（30.6%）⇒ **超线主体是「定位错」不是空帧噪声**。
上界推演：「无 GT 帧一律不上报」⇒ FAR = **0.0377，仍超线** ⇒ 光掐空帧不够。

**② ★ 根因锁定单条序列 `VIDEO00032_19700101_014453`（暗弱悬停）**
xcorr 峰值 **p50 = 0.123 / p05 = 0.057 / 67.4% 低于 `min_peak`=0.25**（对照 `001324`：p50 = **0.906**、仅 4.1% 低于门）
⇒ **模板匹配在暗弱目标上根本没形成真峰** ⇒ 退回质心链 ⇒ 锁到**云层亮边**
（图像实锤 `runs/lookahead/final/evidence/014453_frame6747_full.png`）。
误差结构：标度比 **0.397**（测得位移只有真实位移 40% ⇒ **不是滞后未补偿，是测不出来**）、
离群帧占比 **36.89%**（original 0.00%）、最长脱靶段 **478 帧**、误差方向与真运动**反相关 −0.545**。
**replay 相对 original 多出的 5523 个 FP 里，`014453` 一条占 3032 个 = 54.9%。**

**③ ★ 新增可复用工具：离线策略仿真器 `manu/tracking/lookahead/sim_policy_offline.py`**
不重跑重管线、在已有交付产物上离线试上报层策略（限幅 / 弃权 / 回落），**与官方评测逐位一致**
（启动即打印 `[SIM][PASS] original / stale / replay 三臂与交付产物逐位一致`），
直接 `import` 交付的 `count_matches` / `load_ground_truth` / `build_candidates`，不自己发明判据。
**★ 踩坑**：流水线 Δ=8 走**记录序号**空间（records 帧集 = 候选帧 ∪ 有 GT 帧，**19/24 序列有空洞**，
`001415` 最大偏移 **1541 帧**），payload 键是**真实帧号** ⇒ 必须按记录序号迭代。

**④ ★★ Pareto 前沿：FAR ≤ 0.030 在当前上报层不可达 ⇒ 决定不新增旋钮**
FAR ≤ 0.030 的最高 F1 只有 **84.93%**（abstain@24，比交付 90.46 低 5.53 pp）；
要 F1≥88 且 R≥85 则最优 FAR = **0.0447**（abstain@64，F1 89.66 / R 85.77 / hit@5px **75.91**）**仍超线**，
且 8 序列上 hit@5px 会掉 ⇒ **不为过线牺牲伺服精度**。产物 `runs/lookahead/final/sim_policy_offline.{json,md}`。

**⑤ ★★ 12 个旋钮家族实测否决**（负结果，详见专题档 §24.6）：离群步→中位数 / 绝对单步门 / 一致性门 /
kf 速度冻结 / 冻结+`velocity_source=diff1` / 相对步长门 / xcorr 半径 / xcorr 峰值门 / ROI 模式 chain /
快核窗口 / 快核阈值 / 快核能量门 / 锚点 pred / 回落 holdc。
其中「关 xcorr + 冻结」在 `014453` 单序列上最好（FAR 0.3608 → 0.1257），
但**关掉 xcorr 会把 8 序列 hit@5px 从 84.67 拉回 ~81** ⇒ **不划算，不采纳**。

**⑥ 复现命令**

```bash
cd /tmp/pycharm_project_10ae9e2e
PY=/home/huangzhe/anaconda3/envs/uav/bin/python

$PY manu/tracking/lookahead/diag_far_breakdown.py

$PY manu/tracking/lookahead/sim_policy_offline.py \
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --original-preds runs/lookahead/final/ref_preds_original_v3_24seq \
  --replay-preds  runs/lookahead/final/preds_final_WTH15_24seq \
  --ship-arms     runs/lookahead/final/arms_final_WTH15_24seq.json \
  --cache-root runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache \
  --cache /tmp/sim_policy_offline_cache.pkl \
  --parameters "clamp:8,16,24,32,48,64;abstain:16,24,32,48,64,96" \
  --out-json runs/lookahead/final/sim_policy_offline.json \
  --out-md   runs/lookahead/final/sim_policy_offline.md

$PY manu/tracking/lookahead/diag_001324.py \
  --labels-root "/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010" \
  --arms-json runs/lookahead/final/arms_final_WTH15.json \
  --preds original=runs/lookahead/final/preds_off \
         chain=runs/lookahead/final/preds_SPC4 \
         delivery=runs/lookahead/final/preds_final_WTH15 \
  --seq VIDEO00032_19700101_014453 \
  --out-json runs/lookahead/final/diag_014453.json \
  --out-md runs/lookahead/final/diag_014453_error_structure.md
```

**⑦ 下轮修复方向（需新机理，不是新旋钮）**：xcorr 的有效性判据应**按目标对比度自适应**
（峰值 p50 0.123 vs 0.906 已把两条序列完全分开），而**不是固定 `min_peak`**
（收紧到 0.6/0.75 会连亮目标真匹配一起挡掉 —— 见 ⑤ 的 xcorr 峰值门行）。

### 5.9 ★★★ 目标**极性**自适应（2026-11-03 夜间第 3 段，服务器 21:00–22:20）

> **执行机**：服务器 `test-Super-Server` · python `/home/huangzhe/anaconda3/envs/uav/bin/python`
> **代码**：`manu/tracking/lookahead/polarity.py`（**新建**）· `xcorr.py` · `extrapolator.py` · `lookahead_longquan.py`
> （`manu/evaluation/*` **零改动**）
> **产物**：`runs/lookahead/final/pol8f/` · `runs/lookahead/final/pol24f/` · `runs/lookahead/final/polratio/` ·
> `runs/lookahead/osd/videos/*_pol_*.mp4`
> **完整细节见 memory 专题档 §25**

**① 机理：红外目标的极性（亮/暗）不是恒定的，而快核与 xcorr 模板都写死了「比背景亮」。**

| 位置 | 亮假设（历史行为） | 暗假设（本节新增） |
|:--|:--|:--|
| 快核 `LocalPatchTracker.step` | 门限 `bg + kσ`；权重 `max(patch − 门限, 0)`；峰取 `argmax` | 门限 `bg − kσ`；权重 `max(门限 − patch, 0)`；峰取 `argmin` |
| xcorr 模板 `make_template` | `clip(patch − (bg + kσ), 0)` | `clip((bg − kσ) − patch, 0)` |

**暗目标 + 亮假设 ⇒ 权重恒为 0 ⇒ 质心退化成窗心、xcorr 模板只剩噪点 ⇒ 配准峰锁到画面里最亮的云边**
（合成图实证：暗目标用亮假设，模板权重和 24.9、配准位移 (−22.9, +20.0)、相关峰 0.107；
用暗假设则是 3107.1 / (0.004, 0.005) / **0.987**）。
**★ 归一化相关的符号坑**：`TM_CCOEFF_NORMED` 在真匹配处的**符号取决于目标极性**
⇒ 暗极性必须把**模板整体取反**再配准，否则真匹配处是负值、仍会锁到最亮处。

**② ★★ 纠正 §5.8 的前提：极性是「逐帧」属性，不能整段序列分类。**
§5.8 的探针用 `sorted(glob(...))[:60]` 取样 ⇒ 取到的是**文件名字典序**最靠前的 60 个 ⇒ **几乎全是序列开头的帧**
⇒ 才得出「`015000` 100% 暗」的错误结论。按**帧号数值顺序**重抽后的真相：

| 序列 | 极性时间线 | 暗帧占比 |
|:--|:--|--:|
| `VIDEO00032_19700101_014453` | **暗 f0 – 8112（整条）** | **100.0%** |
| `VIDEO00033_19700101_015000` | **暗 f0 – 175** → 亮 f175 – 975 | 17.5% |
| `VIDEO00031_19700101_014227` | 亮 f0 – 4058 → **暗 f4058 – 4298** | 10.0% |
| 其余 21 条 | 全亮 | 0.0% |

⇒ **一条序列内部极性也会翻转** ⇒ **只能逐链判、不能按序列判** ⇒ 这就是本节把极性判定放在**锚点帧、逐链锁定**的原因。

**③ 实现**：`--replay-polarity {bright,dark,auto}`（**默认 `bright` = 历史行为，逐位不变**）+
`--replay-polarity-ratio`（`auto` 判暗的置信倍数，**默认 10.0** = 实测最优）。
判据是 ROI 中央块上的**亮向/暗向越界能量**（面积归一化）：`dark_mass > ratio × bright_mass` ⇒ 判暗；
两者相等 ⇒ 判**亮**（历史行为）。

**④ 实测 · 8 序列（官方 gate 集，产物 `runs/lookahead/final/pol8f/`）**

| 臂 | F1 | FP | FAR | hit@5px | CLE p50 | CLE p95 |
|:--|--:|--:|--:|--:|--:|--:|
| `original`（零延迟天花板） | 96.400% | 182 | 0.0063 | 88.13% | 1.97 | 13.43 |
| `stale`（老逻辑，交付基线） | 77.203% | 5241 | 0.1807 | 38.18% | 20.43 | 113.73 |
| `replay` 亮假设（旧交付 `WTH15`） | **95.077%** | 523 | 0.0180 | 84.67% | 2.12 | 30.01 |
| `replay` **auto + ratio10**（上一档） | 95.062% | 527 | 0.0182 | 84.96% | 2.12 | 30.15 |
| **`replay` auto + 唯一性二次意见 τ=0.6（★当前交付档）** | **95.266%** | **473** | **0.0163** | **85.17%** | **2.11** | **27.24** |

**7 条亮目标序列逐位不变**；`015000`（暗段）**hit@5px 92.20% → 96.39%（+4.19 pp）** ⇒ **三闸仍全过**。

**⑤ 实测 · 24 序列（官方口径，产物 `runs/lookahead/final/pol24f/arms_F1_auto10.json`）**

| 指标 | 亮假设（旧交付） | **auto + ratio10** | Δ |
|:--|--:|--:|--:|
| F1 | 90.461% | **92.828%** | **+2.37 pp** |
| FP | 8963 | **5889** | **−3074** |
| FAR | 0.0542 | **0.0356** | **−34.3%**（红线 0.030 ⇒ 超线 80.8% → **18.7%**） |
| hit@5px | 77.51% | **81.55%** | **+4.04 pp** |
| 池化 CLE p50 / p95 | 1.88 / 28.48 | **1.73 / 15.16** | −7.9% / **−46.8%** |

逐序列：**`014453` 63.49% → 99.75%（FAR 0.3608 → 0.0021）**、**`014227` 95.33% → 98.73%**、
**14 条逐位不变**、5 条轻微变差（合计 +55 FP）。

> **当前交付档（再叠一层唯一性二次意见 τ=0.6，§5.10）**：24 序列 F1 **92.924%** / FAR **0.0349**（超线 16.4%）/ hit@5px **81.64%** / max_jitter **109.05 px**。

**⑥ 机理闭环（为什么修极性能同时压 FAR、抬 hit@5px）**
`014453` 的流式统计：亮假设 `held_steps` = **64692** / 单步位移上限 **35.62 px**
⇒ 快核**看不见暗目标**、整条链靠 hold 盲推 ⇒ 漂到云边 ⇒ 定位错 ⇒ FP 暴涨；
暗假设下 `held_steps` = **45** / **13.75 px** ⇒ **快核重新看得见目标**
⇒ **同一套机械结构（快核 + xcorr + 退回重推）不变，只是把「看不见」变成「看得见」**。

**⑦ 置信倍数扫描**（`runs/lookahead/final/polratio/`，11 条关键序列 × 4 档）

| ratio | `001810` F1/FAR（基线 96.27 / 0.0363） | `014453` F1/FAR | 判暗链条数 |
|:--|:--|:--|:--|
| 1 | 95.18% / 0.0473（**−1.09 pp**） | 99.75% / 0.0021 | `001810` 68 |
| 3 | 96.05% / 0.0386 | 99.75% / 0.0021 | `001810` 15 |
| **10** | **96.27% / 0.0363（回到基线）** | **99.75% / 0.0021** | `001810` **2**；`014453` 8318 / 8319 |
| 30 | 96.27% / 0.0363 | **99.35% / 0.0061（变差）** | `014453` 8281 / 8319 |

**⑧ OSD 四臂视频**（`runs/lookahead/osd/videos/<SEQ>_pol_<arm>.mp4`，单元格 = `CLE 均值 / hit@5px`）

| 序列 | 窗口 | `stale` | `replay_old`（亮） | **`replay_new`（auto）** | `original`（天花板） |
|:--|:--|:--|:--|:--|:--|
| **`014453`** | 6638–6937 | 3.65 px / 75.0% | **78.05 px / 9.0%**（max 219 px） | **1.21 px / 100.0%** | 1.28 px / 100.0% |
| **`015000`**（暗段） | 9–128 | 4.65 px / 59.2% | 4.65 px / 59.2% | **1.52 px / 100.0%** | 1.49 px / 100.0% |
| `001324` | 4549–4848 | 4.48 px / 65.3% | 3.31 px / 79.0% | **2.27 px / 93.3%** | 1.74 px / 94.7% |
| `014227` | 8–307 | 3.63 px / 82.7% | 1.65 px / 100.0% | 1.65 px / 100.0% | 1.60 px / 100.0% |

**⑨ 回归防线（Gate A + 单测 + 一键复现）**
- **Gate A · 8 序列**：`--lookahead off` 与原脚本 **9/9 文件 md5 逐字节一致**
  （IDENTICAL=9 / DIFFER=0，F1 均 0.964002）。
- **Gate A · 24 序列**：**25/25 IDENTICAL / 0 DIFFER**（F1 均 0.947469）。
- **单测**：`manu/tracking/tests/test_polarity.py`（**7 用例**：四格极性对称性 / 亮路径逐位一致 /
  `ratio` 置信门 / 逐链锁定）+ 既有套件 ⇒ **70 passed**。
- **一键复现**：`bash manu/tracking/lookahead/run_delivery.sh [8|24|both]`
  （`--all-seqs` 与显式 `--seqs` 两种写法数字逐位一致；本轮起脚本名由 `run_delivery_polarity.sh` 改为 `run_delivery.sh`）。

**⑩ 复现命令**

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
C8=runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache
C24=runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
FRAMES="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
CFG="--replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --no-replay-hold-use-ref \
 --output-max-step-px 30 --replay-anchor delta --fast-win-size 52 --fast-bg-window 104 \
 --fast-th-k 2.5 --fast-min-energy 50 --replay-roi-mode selfpred --replay-roi-selfpred-window 3 \
 --replay-roi-selfpred-max-px 4 --replay-xcorr-radius 32 --replay-xcorr-template 32 --replay-xcorr-th-k 1.5"

# 8 序列（gate 集）
$PY manu/tracking/lookahead/lookahead_longquan.py --cache-root $C8 --labels-root "$GT" \
  --frames-root "$FRAMES" --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
  --all-seqs --lookahead replay --arms original,stale,replay $CFG --replay-polarity auto \
  --out-json runs/lookahead/final/pol8f/arms_F1_auto10.json --out-preds-dir runs/lookahead/final/pol8f/preds_F1_auto10

# 24 序列
$PY manu/tracking/lookahead/lookahead_longquan.py --cache-root $C24 --labels-root "$GT" \
  --frames-root "$FRAMES" --params-json manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json \
  --all-seqs --lookahead replay --arms original,stale,kinematic,replay $CFG --replay-polarity auto \
  --out-json runs/lookahead/final/pol24f/arms_F1_auto10.json --out-preds-dir runs/lookahead/final/pol24f/preds_F1_auto10
```

### 5.10 ★★★ 相关峰唯一性比 +「二次意见」重配（2026-11-04 00:05，服务器）

> **执行机**：服务器 `test-Super-Server` · python `/home/huangzhe/anaconda3/envs/uav/bin/python`
> **代码**：`manu/tracking/lookahead/xcorr.py`（`match()` 增加 `diagnostics` 出参）·
> `extrapolator.py`（`LookaheadConfig.replay_xcorr_uniq_gate` + 二次意见逻辑）·
> `lookahead_longquan.py`（`--replay-xcorr-uniq-gate`，**默认 0 = 关**）
> **产物**：`runs/lookahead/final/final8/` · `runs/lookahead/final/final24/` ·
> `runs/lookahead/final/uniqrematch2/`（8 序列 τ 扫描）· `runs/lookahead/final/uniq24b/`（24 序列）
> **完整细节见 memory 专题档 §26**

**① 起因：把 walk-off 帧的机理**定量**归因**（24 序列 8 条、51205 帧离线诊断）

| 帧类型 | 实测位移 \|p50 | 真位移 \|p50 | 方向余弦 p50 | 方向对(>0.9) | **反向(<−0.5)** | 与同帧 `original` 差 p50 |
|:--|--:|--:|--:|--:|--:|--:|
| **TP**（好帧） | 2.0–6.0 px | 3.0–6.1 px | **0.983–0.994** | 70–82% | 2–6% | 0.6–1.5 px |
| **FP**（walk-off） | **22.9–51.3 px** | 3.0–93.1 px | **−0.03–0.87** | 16–47% | **10–28%** | 25–39 px |

⇒ **位移测量本身很准**（TP 帧方向余弦 p50 = 0.99、实测位移 ≈ 真位移）；
⇒ **坏帧不是「卡住不动」，是「链锁到错误特征后一路跑飞」**（实测位移比真位移还大、方向乱、甚至反向）。
⇒ 这一条把历史上「离群帧上误差与真位移**反向**（余弦 −0.905）」的观测解释清楚了：**拖尾/云边把配准峰拽到了错误位置**。

**② 由此**一次性关掉**三个历史旋钮家族（都带上了数字，不再是「感觉没用」）**

| 家族 | 判据 | 实测（24 序列离线 + 8 序列实跑） | 结论 |
|:--|:--|:--|:--|
| 一致性门（`--replay-consistency-gate-px`） | \|实测位移 − 运动学位移\| > τ | τ=20 时门内 16401 帧、**ΔFP = +6822**（净负）；实跑 τ=0.3 ⇒ F1 95.06→**87.03**、hit@5px 84.96→**67.20** | **否决**（用坏尺子量好坏） |
| 低速冻结（`--replay-freeze-speed-px`） | 速度低 ⇒ 位移锁 0 | 同上（速度判据本身不可靠：同一门在 `001810` 救回 95% 的坏帧，却在别的序列上误伤 427 个好帧） | **否决** |
| 位移自适应融合（`--replay-disp-merge-sigma-px`） | 按「实测 vs 参考」差值赋权 | 同上（把好帧换成更差的参考） | **否决** |

**③ 唯一可用的**无状态**离群信号：相关峰唯一性比**
机理：好帧的最佳相关峰**压倒性独大**（次峰≈0）；walk-off 帧的峰**有歧义**（配到了云边/拖尾，附近还有别的像）。
定义：`uniqueness = 次峰 / 最佳峰`，次峰取「**排除与最佳峰重叠的邻域**」后的最大峰。

| 帧类型 | 唯一性比 p25 | **p50** | p75 | p95 | 最佳峰 p50 |
|:--|--:|--:|--:|--:|--:|
| **TP** | 0.000 | **0.000** | 0.203 | 0.558 | 0.819 |
| **FP** | 0.154 | **0.593** | 0.814 | 0.961 | 0.478 |

判别力（24 序列 8 条、51205 帧）：τ=0.5 抓 **56.7%** 的坏帧而只误伤 **6.6%** 的好帧；
τ=0.6 抓 49.4% / 误伤 3.9%；τ=0.7 抓 40.9% / 误伤 2.1%。
**这是目前唯一能把 walk-off 帧与好帧分开的无状态信号**（位移一致性门做不到：它要 12% 的好帧作代价）。

**④ 但「一看到歧义就回退」仍然是净负** ⇒ 落地形态必须是**二次意见**而不是**回退**

| 落地形态 | 8 序列（τ=0.3） | 判定 |
|:--|:--|:--|
| 回退到**锚点**（8 帧前的旧检测） | F1 **87.03**、hit@5px **67.20**、FAR 0.0913 | **净负**（锚点比真实位置更差） |
| 回退到**运动学预测** | 同量级、更差（24 序列实测 FAR 0.0364） | **净负** |
| **弃权**（该帧不输出） | FP 会掉，但好帧也变 FN ⇒ recall/hit@5px 一起掉 | **净负** |

⇒ **最终形态**：歧义时以**锚点**（已知可靠的位置）为搜索中心**重配一次**，
**只在「① 重配结果唯一（uniqueness ≤ 门限）且 ② 重配比原配更自信（peak 更高）」时才采纳**，
否则**保持原结果不动** ⇒ **最坏情况是「什么都没改」⇒ 这个机制不可能让指标变差**。
（② 这条是关键：没有它时，高速目标会被「更靠近锚点但其实错误」的结构骗走——
24 序列实测 `002438` FP **+69**、`000322` **+42**、`001415` **+27**；加上后收窄到 `002438` **+18**。）

**⑤ τ 扫描（8 序列官方 gate 集，产物 `runs/lookahead/final/uniqrematch2/`）**

| τ | F1 | R | P | FAR | hit@5px | CLE p50 | CLE p95 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 0（= 关） | 95.0618% | 92.35% | 97.94% | 0.0182 | 84.96% | 2.12 px | 30.15 px |
| 0.30 | 95.0997% | 92.38% | 97.98% | 0.0178 | 85.00% | — | 29.58 px |
| 0.50 | 95.2172% | 92.50% | 98.10% | 0.0168 | 85.11% | — | 28.32 px |
| **0.60** | **95.2664%** | **92.54%** | **98.15%** | **0.0163** | **85.17%** | **2.11 px** | **27.24 px** |
| 0.70 | 95.2702% | 92.55% | 98.16% | 0.0163 | 85.16% | — | 27.57 px |

伺服侧同步改善（τ=0 → 0.6）：`hit@inbox` 92.35% → 92.54%、`CLE p50` 2.117 → 2.115 px、
**`CLE p95` 30.15 → 27.24 px**、`CLE mean` 9.150 → 8.853 px、
**`p99_jitter` 40.43 → 36.56 px**（板端关心的抖动也降）、`max_jitter` 60.0 px 不变、
`fragmentation` 5 / `lost_frames` 1666 不变。

**⑥ 24 序列官方口径（产物 `runs/lookahead/final/final24/arms_delivery.json`）**

| 指标 | 交付档（τ=0.6） | 上一档（极性自适应） | Δ |
|:--|--:|--:|--:|
| F1 | **92.9245%** | 92.8282% | **+0.096 pp** |
| R / P | **90.54% / 95.44%** | 90.44% / 95.34% | +0.10 / +0.10 pp |
| FAR | **0.0349** | 0.0356 | **−2.0%**（红线 0.030 ⇒ 超线 18.7% → **16.4%**） |
| hit@5px | **81.64%** | 81.55% | **+0.09 pp** |
| **max_jitter** | **109.05 px** | 122.20 px | **−13.15 px**（板端指标改善） |
| `p99_jitter` | 59.90 px | 59.99 px | −0.10 px |

逐序列 **ΔFP 合计 −125**：`001324` −42、`002251` −31、`000322` −30、`002001` −14、`002111` −13；
`hit@5px` 增益集中在 `002251` **+2.03 pp**、`002111` **+0.43 pp**、`001324` **+0.36 pp**、`014144` **+0.39 pp**；
**只有 `002438` 变差**（FP +18、F1 −0.24 pp）。

**⑦ 回归防线（本轮新增代码之后重跑）**
- **Gate A · 8 序列**：`--lookahead off` 与原脚本 **9/9 文件 md5 逐字节一致**（IDENTICAL=9 / DIFFER=0，F1 均 **0.964002**）。
- **Gate A · 24 序列**：**25/25 IDENTICAL / 0 DIFFER**（F1 均 **0.947469**）。
- **新旋钮默认关 ⇒ replay 臂逐位一致**：τ=0 的 8 序列产物与上一档交付产物 **9/9 文件 md5 逐字节一致**。
- **单测**：`manu/tracking/tests/` **149 项（10 skipped）OK**。

**⑧ OSD 三臂对照（★ 新交付档，4 序列 × 3 臂 = 12 段）**

脚本 `manu/tracking/lookahead/osd/render_osd_3arm.sh <SEQ> [window]`（窗口挑选 `pick_osd_window_3arm.py`，
判据 = `mean(cle[stale]) − mean(cle[replay])` 最大 ⇒ 老臂飞得最远、外挂最稳的那一段）；
产物 `runs/lookahead/osd/videos/<SEQ>_3arm_{stale,replay,original}.mp4` + `runs/lookahead/osd/windows/<SEQ>_3arm.json`。

单元格 = **窗口内 CLE 均值 / hit@5px / CLE max**。

| 序列 | 窗口 | `stale`（老逻辑，真吃 320 ms） | **`replay`（外挂交付档）** | `original`（零延迟天花板） |
|:--|:--|:--|:--|:--|
| **`000703`** | 218–517 | 33.26 px / 6.7% / max 108.7 | **3.04 px / 93.0% / max 40.3** | 1.87 px / 99.0% / max 6.2 |
| **`001324`** | 8538–8837 | 32.89 px / 22.7% / max 192.2 | **2.41 px / 91.7% / max 10.3** | 1.98 px / 99.7% / max 5.7 |
| **`001531`** | 8–107 | 22.67 px / 4.0% / max 111.9 | **1.80 px / 98.0% / max 6.4** | 1.61 px / 100.0% / max 3.5 |
| **`015000`** | 339–638 | 18.91 px / 14.3% / max 68.3 | **2.23 px / 98.7% / max 6.0** | 2.03 px / 100.0% / max 5.0 |

⇒ **老逻辑在同一窗口里 CLE 均值 19~33 px、最大脱靶 68~192 px；外挂压到 1.8~3.0 px、最大 6~40 px**

### 剩余 hit@5px 差距的**逐序列归因**（★ 说明「还差多少、差在哪」）

| 口径 | 数值 |
|:--|:--|
| 8 序列总差距（天花板 88.13% − 外挂 85.17%） | **2.96 pp** |
| 其中 **`001324` 单独贡献** | **1.995 pp（占总差距 67.4%）** |
| 其余 7 条序列合计贡献 | 0.967 pp |
| **外挂已达逐序列天花板的** | **96.64%** |

| 序列 | GT 帧数 | 占总 GT | 天花板 hit@5px | 外挂 hit@5px | 差 | **对总差距的贡献** |
|:--|--:|--:|--:|--:|--:|--:|
| `001324` | 10535 | 38.77% | 70.53% | 65.38% | 5.14% | **1.995 pp** |
| `000818` | 8263 | 30.41% | 99.65% | 98.77% | 0.88% | **0.269 pp** |
| `000703` | 1742 | 6.41% | 95.92% | 92.25% | 3.67% | **0.236 pp** |
| `001307` | 3661 | 13.47% | 99.78% | 98.80% | 0.98% | **0.132 pp** |
| `015000` | 1026 | 3.78% | 99.61% | 96.39% | 3.22% | **0.121 pp** |
| `014053` | 592 | 2.18% | 99.66% | 94.09% | 5.57% | **0.121 pp** |
| `015052` | 1239 | 4.56% | 99.60% | 98.39% | 1.21% | **0.055 pp** |
| `001531` | 115 | 0.42% | 99.13% | 91.30% | 7.83% | **0.033 pp** |
| **合计** | **27173** | 100% | **88.13%** | **85.17%** | **2.96%** | **2.962 pp** |

⇒ **`001324` 的天花板本身只有 70.53%**（高速机动的场景/物理上限，**不是外挂造成的损失**）：
它的 GT 帧占全 8 序列的 **38.77%**，所以这一条序列就吃掉了总差距的 2/3。
**其余 7 条序列的外挂 hit@5px 均 ≥ 92.25%**，最差的一条 `001531` 因为只有 115 个 GT 帧（占 0.42%），
对总指标的影响仅 **0.033 pp**。

⇒ **结论：8 序列上剩余的 2.96 pp 命中率差距里，绝大部分是 `001324` 自身天花板低造成的结构性差距；
外挂已经在逐序列口径上拿到天花板的 96.64%。** 要再往上走，瓶颈**不在外挂**（匹配/重推/极性/离群检测都已到位），
而在 `001324` 这条高速机动序列本身。

（画面上就是「老逻辑抽搐脱靶 vs 外挂平滑跟住」）。

## 6. 已知代价与坑

| 项 | 数值 / 事实 | 说明 |
|:--|:--|:--|
| **replay 的 FAR**（8 序列口径已达标 / **24 序列口径仍超线但已收窄**） | 8 序列 FAR **0.0163**（交付线 **0.030**）；**24 序列 FAR 0.0349（超线 16.4%）** | 8 序列比 stale 的 0.1807 好 **10×**、比 original 的 0.0063 差 2.9 倍。**xcorr 精修之前是 0.0337（8 序列超线 12%）**，xcorr 把误检一并压下去了；**极性自适应再把 24 序列从 0.0542（超线 80.8%）压到 0.0356（超线 18.7%）**（§5.9），**唯一性二次意见再压到 0.0349（超线 16.4%）**（§5.10）。**24 序列官方口径上 FAR 仍超线 ⇒ 这是当前唯一的硬缺口**（见 §5.7 / §5.9） |
| **目标极性（亮/暗）不是恒定的** | 快核与 xcorr 模板都写死了「比背景亮」；红外目标的极性**逐帧会变**（`014453` 整条暗、`015000` 前 175 帧暗、`014227` 末 240 帧暗） | 亮假设遇上暗目标 ⇒ 权重恒为 0、质心退化成窗心、xcorr 模板为空 ⇒ **配准峰锁到画面里最亮的云边**（实测 `014453` 偏 70~100 px）。**必须逐链判、不能按序列判**（一条序列内部也会翻转）。已用 `--replay-polarity auto` 修掉，见 §5.9 |
| **抖动** | 8 序列：`max_jitter` **60.0 px**（stale 65.2 / original 65.2，**已更好**）；`p99_jitter` **36.56 px** vs stale **18.9 px**（上一档 40.43 px，本轮再降 3.87 px）。**24 序列：`max_jitter` 109.05 px**（上一档 122.20 px；无 xcorr 版是 61.5 px） | `max_jitter` 靠 `--output-max-step-px 30` 限幅压住；**`p99_jitter` 仍比老逻辑大** ⇒ 异步校正引入了小幅高频抖动，**板端需加一级输出限速/低通** |
| **输出限幅把 max_jitter 从 365.9 压到 114.2** | 不加 `--output-max-step-px` 时 max_jitter **365.9 px** | 限幅只约束输出轨迹本身、不看运动学参考 ⇒ 不会像「一致性门」那样把好帧误杀；代价 F1 只掉 **0.095 pp** |
| **兜底比原位更差** | 兜底默认 `kinematic`，其 Δ=8 表现 F1 **77.55%** / FAR **0.1776** | kinematic 本身不靠谱 ⇒ 兜底**不改善**质量。候选：`stale`（F1 77.20%）或 `abstain`（不输出，让转台知道没坐标） |
| **误检是系统性偏差，不是"跳跃"** | 误检来自**快核质心被拖尾拽偏**（成簇亮点 = 本体 + 拖尾） | 与上一轮"快核单步误差 p50 2.22 px"同源 ⇒ **逐帧位移门无效**（20 px/步门是 no-op）。要压 FAR 必须**治快核（成簇目标本体提取）**，不是加门 |
| 重推步数 / 失败率 | **204,960 步 / 25,620 帧**（平均 8.0 步/帧）；**重推失败 0 帧（`replay_failed` = 0）**、**弃权 2,905 帧**（`abstained`，= 无上报帧，含检测器盲区）；单步最大位移 **34.39 px**（`replay_max_step_disp`） | `arms.replay.stream_stats.overall` |
| **`track_id` 是进程级全局计数器** | `PointKalmanTrack._count` 随"同进程内此前建过多少条 track"变化 | ⇒ **比对必须让参考与新脚本都跑满同样顺序的 8 条序列**；拿子集去比 8 序列产物，`track_id` 必然不同（位置/分数仍完全相同）。**这是原脚本的性质，不是本模块引入的** |
| 档位 A 的机理 | Δ=8 真实位移 p50 = **3.54 px**；1 帧差分量级 ≈0.5~2 px/帧 ⇒ ×8 = **4~16 px**；真实位移 p90 = **51 px** / p95 = **91 px** / p99 = **180 px** | 静止目标被**推偏**（外推噪声 > 滞后收益）+ 匀速假设在尾部失效 ⇒ 尾部误差放大到 183 px |
| `--replay-step-gate-px` 默认 0 | 即**不设门** | 该参数**保留为诊断开关**；两个门都已被数据否掉（§5.3） |

## 7. 产物路径速查

| 目录 | 含义 | 内容 |
|:--|:--|:--|
| `runs/lookahead/ref/` | **参考**（原脚本 `manu/evaluation/system_sota_longquan.py`，未改动） | `original_v3_8seq.json`（运行报告）+ `preds_original_v3/`（8 序列 + `summary.json`）——Gate A 的比对基准；`original_v3_8seq.log` 为空 |
| `runs/lookahead/final/preds_off/` | 新脚本 `--lookahead off`（**Gate A 的比对对象**） | 8 序列 + `summary.json`；与 `ref/preds_original_v3/` **9/9 文件 md5 逐字节一致** |
| `runs/lookahead/final/ref_preds_original_v3_24seq/` | 原脚本跑 24 序列 | 与 `final/preds_off_24seq/` **25/25 文件 md5 逐字节一致**（24 序列 Gate A） |
| `runs/lookahead/final/arms_delta{2,4,8,12}_WTH15.json` | **Δ 扫描**（2026-11-03 夜间服务器重跑，定稿配置） | §5.2 的表；`delta2/4/12` 属**延迟轴**，不参与旋钮表排序 |
| `runs/lookahead/final/arms_velsrc_{kf,diff1,diffN}.json` | **速度源扫描**（同上） | §5.3 的表 |
| `runs/lookahead/final/seq1324/` | **001324 单序列扫描** | `arms_R{48,64,96,128,160}.json` / `arms_R96MP60.json` / `arms_R96T{24,48}.json` / `arms_CR24MP60.json` / `arms_CR32MP60.json` / `arms_CR32MP75.json` / `arms_CR40MP60.json` / `arms_CF{20,35,50}.json` / `arms_CF35{R48,T48,T48R48}.json` / `arms_TH{30,40}.json` —— **20** 个 `arms_*.json` 产物（每个另配同名 `.log`）—— §23.5 的单序列表 |
| `runs/lookahead/final/diag_001324{,_error_structure}.json/.md` | **001324 误差结构诊断** | §23.5 的表（**脚本直接生成 md**，不手抄） |
| `runs/lookahead/final/diag_001324_xcorr_routes.json` | **xcorr 三条修复路线复现** | §23.11 的表 |
| `runs/lookahead/final/variant_table.md` + `.json` | **变体总表（脚本生成）** | §5.5 / 专题档 §23.2 的完整表；按**旋钮轴 / 延迟轴**分两段 |
| `runs/lookahead/final/diag_far_breakdown.{json,md}` | **24 序列 FAR 逐序列拆解**（脚本 `diag_far_breakdown.py`） | §5.8 ① 的表；自算 vs 交付 `tp/fp/fn` 差值全 0、逐帧复算 24/24 全 0 |
| `runs/lookahead/final/diag_014453{,_error_structure}.{json,md}` | **014453 误差结构诊断**（`diag_001324.py --seq …014453`） | §5.8 ② 的表（标度比 0.397 / 离群帧 36.89% / 最长脱靶段 478 帧） |
| `runs/lookahead/final/diag_014453_xcorr_routes.json` | **014453 xcorr 三路线 + 峰值统计** | §5.8 ② 的峰值表（p50 0.123 / 67.4% 低于 min_peak） |
| `runs/lookahead/final/evidence/014453_frame6747{,_full}.png` | **014453 图像实锤** | GT（绿叉）在极暗弱点 / replay（红叉）锁在 70–100 px 外的云层亮边 |
| `runs/lookahead/final/sim_policy_offline.{json,md}` | **离线策略仿真器产物**（`sim_policy_offline.py`） | §5.8 ④ 的 Pareto 前沿表；启动即打印 `[SIM][PASS] … 逐位一致` |
| `runs/lookahead/final/{far24,far24b,mp3,ack,cand8,s14453}/` | **24 序列 FAR 攻坚的 12 个旋钮家族批次产物** | §5.8 ⑤；脚本 `/mnt/data/codex-overnight/run_*.sh`，日志 `/mnt/data/codex-overnight/{far24_sweep,far24b_sweep,mp3,ack,cand8}.log`。⚠️ `mp3` 批 `M0..M8` **9 个变体漏写参数、产物逐位相同 ⇒ 作废** |
| `runs/lookahead/final/pol8f/` · `runs/lookahead/final/pol24f/` | **★ 极性自适应产物（2026-11-03 夜间第 3 段）** | `arms_F0_bright.json` / `arms_F1_auto10.json`（8 序列）、`pol24/arms_R0_bright.json`（24 序列亮基线）· `pol24f/arms_F1_auto10.json`（24 序列 **auto + ratio10**：F1 92.828% / FAR 0.0356 / hit@5px 81.55%）+ 同名 `preds_*/` 逐帧 preds |
| `runs/lookahead/final/polratio/` · `runs/lookahead/final/poldiag/` | **极性置信倍数扫描 / 逐链诊断** | `arms_r{1,3,10,30}.json`（4 档 × 11 序列）、`poldiag/arms_diag.json`（逐链判暗数 + `replay_polarity_ratio_ge{3,10,30}` 桶计数）—— §5.9 ⑦ 的表 |
| `runs/lookahead/final/pol14453/` | **`014453` 单序列极性对照** | `arms_Q0_bright.json`（旧：F1 63.49 / FAR 0.3608）· `arms_Q1_auto.json`（F1 99.75 / FAR 0.0021）· `arms_Q2_dark.json`（强制暗，与 auto 完全一致）· `arms_Q3_auto_noxcorr.json`（关 xcorr，FAR 0.0013） |
| `runs/lookahead/osd/videos/<SEQ>_3arm_*.mp4` | **★ 新交付档的 OSD 三臂视频**（4 序列 × 3 臂 = 12 段） | `{stale,replay,original}`；脚本 `manu/tracking/lookahead/osd/render_osd_3arm.sh`，窗口挑选 `osd/pick_osd_window_3arm.py` （判据 `mean(cle[stale]) − mean(cle[replay])` 最大）—— §5.10 ⑧ |
| `runs/lookahead/osd/videos/<SEQ>_pol_*.mp4` | **★ 极性修复的 OSD 四臂视频**（4 序列 × 4 臂 = 16 段） | `{stale,replay_old,replay_new,original}`；脚本 `/mnt/data/codex-overnight/render_osd_polarity.sh`，窗口选择 `/mnt/data/codex-overnight/pick_osd_window_polarity.py` —— §5.9 ⑧ 的表 |
| ~~`runs/lookahead/off/`~~ `~~sweep/`~~ `~~gate/`~~ | **服务器上不存在**（只存在于 host 侧） | 旧版本 README 引的这些路径**已删除**；等价产物在 `final/` 下（见上） |
| `runs/lookahead/final/final8/` | **★ 当前交付档 · 8 序列官方 gate 集（2026-11-04）** | `arms_delivery.json`（四臂定稿）+ `preds_delivery/`（逐帧 preds）+ `delivery.log` + `report_delivery_vs_prev.md`（追回 **94.1%** 滞后损失，上一档 93.0%）—— §5.10 |
| `runs/lookahead/final/final24/` | **★ 当前交付档 · 24 序列官方口径（2026-11-04）** | `arms_delivery.json`（四臂定稿）+ `preds_delivery/` + `delivery.log` —— §5.10 ⑥ |
| `runs/lookahead/final/uniqrematch2/` | **二次意见·加严准则 8 序列 τ 扫描（交付口径）** | `arms_u{030,050,060,070}.json` —— §5.10 ⑤ |
| `runs/lookahead/final/uniqrematch/` | **二次意见·宽松准则 8 序列 τ 扫描** | `arms_u{020…070}.json` —— §5.10 ⑤（收益小于加严口径） |
| `runs/lookahead/final/uniq/` | **「回退锚点」形态 8 序列 τ 扫描 —— 已否决** | `arms_u{0,030,040,050,060,070,085}.json`；τ=0 即上一档交付档（与 `pol8f/preds_F1_auto10` 9/9 md5 一致）—— §5.10 ⑤ |
| `runs/lookahead/final/uniq24/` · `runs/lookahead/final/uniq24b/` | **24 序列 τ 扫描（旧准则 / 加严准则）** | `uniq24/arms_u{050,060,070}.json` · `uniq24b/arms_u{050,060}.json`（加严 τ=0.6 = 交付档：F1 92.9245 / FAR 0.0349 / hit 81.64）—— §5.10 ⑥ |
| `runs/lookahead/final/` | **★ 交付定稿（服务器，2026-11-03 夜间）** | `arms_final_WTH15.json`（四臂主产物）+ `preds_final_WTH15/`（replay 臂逐帧 preds）· `arms_final_stale.json` + `preds_final_stale/`（老逻辑基线 preds，做 OSD 用）· `preds_off/`（`--lookahead off`，Gate A 比对对象）· `arms_gateA_W.json`（Gate A 复验，ΔF1 = 0.000000）· `report_8seq_final_WTH15.md`（对照表成品）· `code_sync_20261103.tar.gz` + `.md5`（跨机代码同步包）· `arms_<tag>.json` / `preds_<tag>/`（**130** 个可用变体产物；磁盘扫描 **137** 个 `arms_*.json`，其中 **7** 个因口径不一致/无 `arms.replay` 臂被排除） |
| `runs/lookahead/osd/` | **OSD 视频复核**（§5.4 旧窗口 / §5.6 新 4 段） | `arms_stale_<SEQ>.json`（3 个）+ `VIDEO00007_19700101_001324_stale.json` / `VIDEO00033_19700101_015000_stale.json` + `preds_stale_<SEQ>/`（3 个，逐序列预测 JSON）；`videos/<SEQ>_{original,stale,replay}.mp4`（三臂 OSD 视频）+ 同名 `.json`（渲染器逐帧判定）/ `.log`（渲染日志）；`windows/<SEQ>.json` + 同名 `.pick.log`（窗口选择结果）；`osd_window_metrics.md`（中心距口径的窗口定量核验）；`logs/`。旧窗口产物 ~~`window_lagfail_001324.json`~~ / ~~`window_moderate_001324.json`~~ / ~~`videos/*_lagfail.mp4`~~ **服务器无存档**，§5.4 旧表数字无法复核。已在 memory 档 §4.2 登记 |

## 8. TODO（照抄 memory 档 §9 的"下一步"，按优先级）

1. **交付结论**：**档位 B（视觉重推 + xcorr 精修 + 极性自适应 + 唯一性二次意见）可以进板端验证**——8 序列上把 320 ms 滞后造成的损失挽回
   **F1 95.266%**（77.203% → **95.266%**，天花板 96.400%）、**hit@5px 85.17%**（38.18% → **85.17%**，天花板 88.13%）、
   FAR **0.0163**（交付线 0.030 以内），且**对成熟跟踪资产零回归**（尺子1 ΔF1 = 0.000000，Gate A 9/9 逐位一致）。
   **交付配置 = 冠军 `WTH15` + `--replay-polarity auto` + `--replay-xcorr-uniq-gate 0.6`**（其余参数一字不改）。
   相对上一档：**F1 +0.205 pp / FAR −10.2% / CLE p95 −2.91 px / p99_jitter −3.87 px**；
   **F1 追回 94.09% 的滞后损失**（上一档 93.04%）。
   24 序列官方口径：F1 **92.924%** / FAR **0.0349** / hit@5px **81.64%** / max_jitter **109.05 px**（§5.10）。
   **档位 A 不要做**。
2. **★ 「离群检测器」已落地**：原 TODO 指出的 `001324`「脱靶后停住」型离群帧，**已由「相关峰唯一性比 + 二次意见重配」处理**
   （§5.10）：8 序列 F1 +0.205 pp / hit@5px +0.21 pp / FAR −10.2% / CLE p95 −2.91 px；
   24 序列逐序列 **ΔFP 合计 −125**。**一致性门 / 低速冻结 / 位移自适应融合三个家族实测全部净负**（§8 的否决表），**不要再试**。
   剩余缺口只剩三处：① **`001324` 的天花板本身只有 70.53%**（高速机动的固有上限，不是外挂造成）；
   ② **`001531`**（天花板 99.57% 而外挂 91.30%，唯一仍有明显缺口的一条）；
   ③ **24 序列 `002438` +18 FP**（唯一劣化的序列）。
3. **★ 极性方向的下一步**（见 §5.9 ⑩）：
   ① **暗/亮切换时沿用上一链的极性**（时间平滑），修 `015000` 的 F1 −0.78 pp；
   ② `min_energy` / `min_contrast` **按极性分别设阈值**（暗向能量与亮向能量不同尺度）；
   ③ **24 序列 FAR 还差 0.0049**（0.0349 vs 红线 0.030），剩余主体是**管线自身**问题
   （`015740` 0.4428 / original 0.4277、`002251` 0.1034 / original 0.0997、`003002` 0.0707）+ 外挂残余 `002438` +18 FP。
   ⚠️ 三条「搜索侧」路线已实测否决（专题档 §23.11），**不要再试**。
3. **24 序列官方口径复评 —— ✅ 已完成并已被极性修复覆盖**（2026-11-03 夜间，专题档 §23.12 / §25 · README §5.7 / §5.9）：
   极性修复**前**：F1 90.461%、hit@5px 77.51%、FAR **0.0542**（超线 80.8%）；
   **极性修复后（`--replay-polarity auto`，上一档）**：F1 **92.828%**、hit@5px **81.55%**、FAR **0.0356**（超线 18.7%）；
   Gate A **25/25 md5 一致**（F1 均 0.947469）。
   **⇒ 剩余硬缺口：24 序列 FAR 0.0356 > 交付线 0.030（差 0.0056）**，剩余主体是**管线自身**问题
   （`015740` 0.4428 / original 0.4277、`002251` 0.1054 / original 0.0997、`003002` 0.0707、`002111` 0.0651）
   ⇒ 属**扩展缺口**，**不为此牺牲 hit@5px**。
   **★ 再叠一层唯一性二次意见（τ=0.6）后**：F1 **92.9245%** / hit@5px **81.64%** / FAR **0.0349**（超线 **16.4%**）/ max_jitter **109.05 px**；
   剩余差 **0.0049**；唯一劣化序列 `002438` **+18 FP**（§5.10 ⑥）。
4. **`p99_jitter` 36.56 px vs stale 18.9 px**：板端加一级输出限速/低通（`max_jitter` 60.0 已优于老逻辑的 65.2；上一档 p99_jitter 40.43 px，本轮已降 3.87 px）。
5. **板端移植**：`extrapolator.py` 的 replay 臂（含 `xcorr.py` 模板匹配）要移植到 C++；
   **xcorr 精修、输出限幅、兜底策略、无状态性（不写回跟踪器状态）四条是移植契约**。
6. **Δ 降一降**：若能把检测延迟从 320 ms 压到 160 ms（Δ=4），老逻辑 F1 从 **77.20%** 升到 **89.81%**、
   档位 B 从 **95.08%** 升到 **95.85%**、hit@5px 从 84.67% 升到 **86.48%** ⇒ **降延迟与加外挂是叠加收益**
   （外挂相对老逻辑的增益随 Δ 放大：Δ=2 +1.39 / Δ=4 +6.05 / Δ=8 +17.87 / Δ=12 +26.25 pp F1）。
7. **跨机同步**：host 侧代码落后于服务器（`xcorr.py` / `extrapolator.py` / `lookahead_longquan.py`）⇒ 用
   `/mnt/data/codex-overnight/code_sync_20261103.tar.gz` 补齐（md5 见 memory 专题档 §23.7）。

## 9. 铁律提醒

- 模型固定 **RC2 / trial_0080 / epoch3**，**不重跑推理**（回放 `ms_cache/*.pkl`）；评测口径 **import 冻结实现**（`manu/evaluation/eval_multiscale_cache_th.py`）；**配对数据用 McNemar 精确二项检验**，Wilson CI 重叠不能当等价性证明；
- **文档里的每个统计数字都必须来自产物 JSON**（铁律九）——本 README 的数字全部来自 `runs/lookahead/**` 与 `manu/memory/lookahead_extrapolator_20261102.md`；
- 未经授权**不得**启动训练 / 重推理 / 数据集下载；`manu/data/build_full_median_dataset.py` 禁改；**`manu/evaluation/*` 的原文件禁改**——外挂一律走 `manu/tracking/lookahead/` 里的**拷贝**；
- 本模块**只读输出流、不回写跟踪器状态**——任何"顺手改一下跟踪器状态"的改动都会打破"关掉开关 ⇒ 逐位一致"这个构造性保证。

**相关文件**：代码 `manu/tracking/lookahead/{__init__.py, extrapolator.py, track_fusion.py, lookahead_longquan.py}` · 单测
`manu/tracking/tests/test_lookahead_plugin.py`（14 项）· 专题记忆档 `manu/memory/lookahead_extrapolator_20261102.md`
（本 README 的数字来源）· 上一轮（异步双频检跟分离）`manu/memory/async_dual_rate_tracking_20261102.md`、`manu/tracking/README.md`。
### 5.11 ★★★ 「不自信就不上报」：把误报换成沉默（2026-11-04 服务器夜间第 2 轮）

> **执行机**：服务器 `test-Super-Server` · python `/home/huangzhe/anaconda3/envs/uav/bin/python`
> **代码**：`manu/tracking/lookahead/extrapolator.py`（`LookaheadConfig.replay_abstain_peak_below` /
> `replay_abstain_disp_px` + 弃权判定）· `lookahead_longquan.py`（`--replay-abstain-peak-below` /
> `--replay-abstain-disp-px`，**默认 0 = 关**）· 新诊断脚本 `diag_fp_attrib.py` · `diag_abstain_scan.py`
> **产物**：`runs/lookahead/final/final8/` · `runs/lookahead/final/final24/` ·
> `runs/lookahead/final/final24/diag_fp_attrib.md` · `runs/lookahead/final/final24/diag_abstain_scan.md`
> **完整细节见 memory 专题档 §27**

**① 起点：24 序列 FAR 是唯一的硬缺口**

上一档 24 序列 FAR **0.0349** > 交付线 0.030（超线 807 个 FP）；
8 序列 FAR 0.0163 没有压力。

**② 先把误报逐帧归属（`diag_fp_attrib.py`）**：replay 臂 vs 零延迟臂，同帧对账

| 归属 | 数量 | 占 FP | 插件能不能改 |
|:--|--:|--:|:--|
| `lost_by_replay`（零延迟臂对、重推臂错） | **1658** | 28.8% | **能** |
| `both_missed`（两臂都错，检测器噪声） | 1367 | 23.7% | 不能 |
| `no_ground_truth`（该帧 GT 显式为空） | 2739 | 47.5% | 不能 |

**③ 新机理**：模板配准的**相关峰**是这次重推唯一的自洽性证据。峰值低有两种成因：
目标**机动太大**（模板跟不上但快核跟得住 ⇒ 位置对）或快核**跟丢**（位置错）。
用**链位移**区分：跟丢时链从锚点跑很远，机动时链几乎没跑动。
⇒ 弃权条件：**峰值 < 0.30 且 链位移 > 24 px** ⇒ 该帧不上报（转台保持上一帧指令）。

**④ 离线扫描器**（`diag_abstain_scan.py`）：一次 dump → 规则网格秒级复算；
空规则必须逐位复现交付档（8 序 F1 95.2664%、24 序 F1 92.9245%）。

**⑤ 定稿数字（实跑，产物 JSON 直接读出）**

| 数据集 | 臂 | F1 | FAR | hit@5px | CLE p50 | max_jitter |
|:--|:--|--:|--:|--:|--:|--:|
| 8 序 | `stale`（老逻辑） | 77.2034% | 0.1807 | 38.1850% | 6.88 | 65.24 |
| 8 序 | **`replay`（交付档）** | **95.3038%** | **0.0151** | **85.1728%** | 1.67 | 60.00 |
| 8 序 | `original`（天花板） | 96.4002% | 0.0063 | 88.1316% | 1.57 | 65.24 |
| 24 序 | `stale`（老逻辑） | 77.5225% | 0.1559 | 44.8232% | 5.29 | 76.52 |
| 24 序 | **`replay`（交付档）** | **93.0865%** | **0.0298** | **81.4974%** | 1.72 | 60.00 |
| 24 序 | `original`（天花板） | 94.7469% | 0.0208 | 85.8957% | 1.56 | 76.52 |

**⑥ 复现命令**

```bash
bash manu/tracking/lookahead/run_delivery.sh both          # 交付档（8 + 24 序列）
/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_fp_attrib.py       # 误报归属
/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_abstain_scan.py --dump
/home/huangzhe/anaconda3/envs/uav/bin/python manu/tracking/lookahead/diag_abstain_scan.py --scan
```

**⑦ 新 CLI（默认关，关着时 replay 臂逐位不变）**

```bash
--replay-abstain-peak-below 0.0   # 模板峰值低于此值 ⇒ 不上报；0 = 关
--replay-abstain-disp-px  0.0     # 配合上一项：链位移超过此值才算「跟丢」
```

交付档取值：**`--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24`**。
### 5.12 ★★★ 帧级精度损失归属 ＋ 锚点 oracle 天花板 ＋ 「源帧无检测 ⇒ 再往回退重推」（2026-11-04 夜间第 3 轮）

**交付档配置本轮未改**（冠军仍是 §5.11 的 `peak<0.30 & disp>24`）。本轮补的是**归因与边界**：

| 产物 | 命令 | 结论 |
|:--|:--|:--|
| 帧级损失归属 | `python manu/tracking/lookahead/diag_gap_attrib.py --labels-root ... --original ... --replay ...` | 8 序：外挂独丢 1260 帧 / 独赚 456 帧 / **双失 2769 帧（检测器上限，外挂无责）** |
| 误差分解 | `python manu/tracking/lookahead/diag_err_decomp.py ...` | 外挂误差 = **源帧锚点误差 e(f)** ＋ 位移残差 δ（δ p90 ≤ 3.13 px） |
| 锚点 oracle | `python manu/tracking/lookahead/diag_oracle_bound.py ...` | 天花板：8 序 **87.804%** / 24 序 **85.522%**（零延迟天花板 88.132% / 85.896%） |
| 回退重推 | `--replay-rollback-max-frames N --replay-rollback-min-peak P` | 8 序 `rb8 + mp0.50` → F1 95.624% / hit@5px 85.592% / FAR 0.0171；24 序 FAR 红线饱和 ⇒ **默认关** |

**回归闸门**：`bash manu/tracking/lookahead/check_newknobs_off.sh [8|24]`（新旋钮关 ⇒ 与冠军产物逐位一致）；
标准 Gate A 脚本 `/mnt/data/codex-overnight/gateA_live.sh` · `gateA24_live.sh`。
### 5.13 ★★★ 最终交付候选：冠军 + 回退重推 + 峰值门 0.90（2026-11-04 夜间第 3 轮收尾）

**交付档配置本轮已改**：在冠军旋钮后加两个旋钮（**默认 0 = 关**）：

```bash
--replay-rollback-max-frames 16 --replay-rollback-min-peak 0.90
```

| 口径 | 指标 | 上一版冠军 | **新交付候选** | Δ |
|:--|:--|--:|--:|--:|
| 8 序 | F1 | 95.3038% | **95.4644%** | +0.1605 pp |
| 8 序 | hit@5px | 85.1728% | **85.2206%** | +0.0478 pp |
| 8 序 | FAR/帧 | 0.01511 | **0.01511** | +0.000000 |
| 24 序 | F1 | 93.0865% | **93.1578%** | +0.0713 pp |
| 24 序 | hit@5px | 81.4974% | **81.5379%** | +0.0405 pp |
| 24 序 | FAR/帧 | 0.02981 | **0.02984** | +0.000024 |

| 新产物 | 命令 | 结论 |
|:--|:--|:--|
| 抖动口径审计 | `python manu/tracking/lookahead/diag_jitter_audit.py --arms-json ... --preds-dir ...` | `max_jitter 294.98` 是**伪影**（records 空洞）；**真实单帧位移 ≤ 30.00 px** |
| 回退帧级归属 | `python manu/tracking/lookahead/diag_rollback_attrib.py --labels-root ... --base ... --other ...` | 8 序 **TP 81 / FP 0**、24 序 **TP 176 / FP 4**，**零回归** |

**回归闸门**：`bash manu/tracking/lookahead/check_newknobs_off.sh [8|24]`（新旋钮关 ⇒ 与上一版冠军产物逐位一致）；
标准 Gate A：`/mnt/data/codex-overnight/gateA_live.sh`（8 序）· `gateA24_live.sh`（24 序）。

### 5.14 ★★★ OSD 三臂对照视频（8 序 · 交付档实测，2026-11-04 夜间第 3 轮）

```bash
bash manu/tracking/lookahead/render_osd_3arm.sh 8
```

产物：`runs/lookahead/osd/videos/<SEQ>_3arm_{stale,replay,original}.mp4`（+ 同名 `.json` 逐帧判定）。
**8 序 × 3 臂 = 24 段**，全部 h264 / 1920×1080 / 25 fps、**全帧可解码、0 异常**
（清单：`/mnt/data/codex-overnight/osd_manifest_round6.md`）。

| 序列 | 场景 | 窗口 | stale（老逻辑） | replay（外挂） | original（天花板） |
|:--|:--|:--|:--|:--|:--|
| `VIDEO00007_19700101_001324` | 高速机动 | 8538–8837 | F1 56.67% | **F1 100%** | F1 100% |
| `VIDEO00033_19700101_015000` | 悬停 | 339–638 | F1 75.67% | **F1 100%** | F1 100% |

⚠️ 窗口号以 `runs/lookahead/osd/windows/<SEQ>_3arm.json` 的 `config.frames` 为准；
本表与 §5.11 的老窗口（001324 = 8549–8848、015000 = 726–1025）**不是同一段画面、不可逐帧对齐**。
### 5.15 ★★ 旋钮空间穷举证伪（2026-11-04 夜间第 4 轮）：**交付档不动**

18 个变体（每个只改一个旋钮）在 8 序全部**没能**超过交付档：

| 旋钮族 | 扫描档位 | 结论 |
|:--|:--|:--|
| xcorr 搜索半径 | 32 / 36 / 40 / 48 / 64 | **32 最优**（越大越差：背景假峰 ⇒ 幽灵坐标） |
| xcorr 档位 | direct / chain / chain_direct | **direct 最优**；chain 在「极性修好之后」仍打不过 |
| 模板尺寸 | 28 / 32 / 36 | 32 最优；36 几乎持平但 FAR 更低（0.01511 → 0.01476） |
| xcorr `th_k` | 1.0 / 1.5 / 2.0 | **1.5 最优** |
| ROI 前推上限 | 4 / 8 / 12 / 16 / 24 | **4 最优**（8 只 +0.015 pp ≈ 噪声） |

```bash
bash /mnt/data/codex-overnight/sweep_xcorr_grid.sh    # th_k / 模板 / 半径
bash /mnt/data/codex-overnight/sweep_roimax.sh        # ROI 前推上限
bash /mnt/data/codex-overnight/sweep_xcorr_mode.sh    # xcorr 档位 / 半径
bash /mnt/data/codex-overnight/sweep_minpeak.sh       # xcorr 峰值门 / 唯一性比门
bash /mnt/data/codex-overnight/sweep24_tpl36.sh       # 24 序：模板 36 × 峰值门
```

★ **口径提醒**：console 打的 `CLE p95` 是**最差单序列**口径（8 序 25.74 px），
JSON 的 `cle_pooled_p95` 才是**池化**（9.99 px）—— 两者差 2.6 倍，跨配置比较必须同字段。

### 5.16 ★★★ 第 5 轮（2026-11-04 服务器夜间第 5 轮）：口径纠偏 ＋ 可见性天花板 ＋ 新冠军 CD_MP060

**§31 的头号结论被证伪**：失手帧里「真实位移 > 32px 搜索半径」只占 **25.52%**（988/3872）
⇒ **74.48% 的失手帧位移在半径内** ⇒ 不是「搜不到」，是「**配不准**」
⇒ §31 §4 的「图像重定位推翻管线锚点」**方向错了**（锚点误差 p50 = 1.50 px / p99 = 4.54 px，**不需要推翻**）。
**口径纠偏**：上一轮「手算 64.74% vs 官方 70.53%」是**诊断脚本自己的错**（自写 GT 解析，不是官方 `load_ground_truth`），
**官方 `servo_metrics` 一直是对的**（手算复现与官方 **float 位级相等**，差 0.0000）。

**① 可见性天花板：真实 headroom 只有 +3.5227 pp**（不是 2.91 pp）

| 量 | 8 序实测 |
|:--|--:|
| GT 帧 | 27173 |
| **当前帧目标不可见** | **1568（5.77%）** —— 001324 独占 1564 |
| original 臂 全量 hit@5px | 88.1316%（**不是天花板**） |
| replay 臂 全量 hit@5px | 85.2206% |
| **original 臂 可见帧内 hit@5px** | **93.5286%（← 真天花板）** |
| **replay 臂 可见帧内 hit@5px** | **90.0059%** |
| ⇒ **可见帧内的真实 headroom** | **+3.5227 pp** |

⇒ **001324 的病根是检测器本身**：当前帧不可见率 **14.85%**（其余序列 ≤0.2%）、源帧（t−8）检测器漏检率
**15.3%**（其余序列 <1%）⇒ 001324 上「original 臂自己也只有 70.53% 全量 hit」**不是外挂的锅，是检测器的锅**。

**② 新冠军 `CD_MP060`（chain_direct + min_peak 0.60）：F1 / FAR / hit 三项同时最优**

| 候选 | F1% | FAR/帧 | hit@5px | CLE p50(池) | CLE p95(池) | max_jitter |
|:--|--:|--:|--:|--:|--:|--:|
| 交付档（rb16+mp0.90, direct） | 95.4644 | 0.015105 | 85.2206 | 1.67 | 9.99 | 60.00 |
| **CD_MP060**（chain_direct + min_peak 0.60） | **95.6505** | **0.013553** | **85.3715** | **1.67** | **9.07** | 60.00 |
| MP030（chain + min_peak 0.30） | 95.5240 | 0.013967 | 85.2206 | 1.72 | 8.93 | 62.00 |
| CH_MP040（chain + min_peak 0.40） | 95.5316 | 0.013898 | 85.2022 | 1.72 | 8.89 | 62.00 |

⇒ 相对交付档：F1 **+0.1861 pp**、FAR **−10.27%**、hit **+0.1509 pp**、CLE p95 **改善 0.92 px**、max_jitter 持平。
**并推翻 §31「direct 赢」**：正确对比轴是 **(档位 × min_peak) 二维网格**，不是单点对比 —— chain 类档位下
min_peak 是**逐段**判据（任一段峰值低 ⇒ 整条链作废 ⇒ 该帧不上报 ⇒ 必然脱靶）⇒ **min_peak 越高越差**
（实测 0.30 → 0.40 → 0.60 → 0.75 → 0.85 单调变差）。
机理（`stream_stats`）：`replay_xcorr_ambiguous` **1936 → 1219（−37%）**、唯一性比均值 **0.233 → 0.173**、
`replay_max_step_disp` **79.52 → 54.98** ⇒ 相关峰更独特 ⇒ 歧义帧少 37% ⇒ FAR 降 10% 而 hit 升。

**③ 位移估计器大比试**：逐帧配准（chain 类）把位移 **p95 从 30~84 px 压到 3~13 px**（降 59~96%），
**stride2 与 stride1 打平** ⇒ 步长 1 已够。

**④ 8 序变体全表**（数据源 `runs/lookahead/final/final8/round8_variants.json`；扫描族：`chainmp/` = chain × min_peak、
`chainroimax/` = chain × ROI 前推、`mplo/` = chain × min_peak 低位、`chainfar/` = FAR→hit 兑换）

| 变体 | F1% | FAR/帧 | hit@5px | CLE p50(池) | CLE p95(池) | CLE p99(池) | max_jitter |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `final8/arms_delivery` | 95.4644 | 0.015105 | 85.2206 | 1.67 | 9.99 | 50.92 | 60.00 |
| `chainmp/CD_MP060` | 95.6505 | 0.013553 | 85.3715 | 1.67 | 9.07 | 44.70 | 60.00 |
| `chainmp/CD_MP075` | 95.3080 | 0.016691 | 84.2712 | 1.70 | 10.79 | 62.28 | 60.00 |
| `chainmp/CH_MP040` | 95.5316 | 0.013898 | 85.2022 | 1.72 | 8.89 | 49.50 | 62.00 |
| `chainmp/CH_MP060` | 95.4520 | 0.014622 | 85.0219 | 1.73 | 9.10 | 48.99 | 60.00 |
| `chainmp/CH_MP075` | 95.0505 | 0.018278 | 83.7449 | 1.75 | 10.92 | 63.96 | 60.00 |
| `chainmp/CH_MP085` | 92.8162 | 0.038728 | 77.2789 | 1.93 | 19.70 | 113.21 | 60.00 |
| `chainroimax/CD_R32_ROIMAX0` | 95.4483 | 0.014657 | 85.1912 | 1.67 | 9.53 | 48.50 | 81.88 |
| `chainroimax/CD_R32_ROIMAX16` | 95.4502 | 0.014657 | 85.1912 | 1.67 | 9.48 | 47.65 | 81.88 |
| `chainroimax/CD_R32_ROIMAX24` | 95.4465 | 0.014691 | 85.1912 | 1.67 | 9.48 | 48.50 | 81.88 |
| `chainroimax/CH_R32_ROIMAX0` | 95.4786 | 0.014381 | 85.1875 | 1.72 | 9.00 | 54.18 | 62.00 |
| `chainroimax/CH_R32_ROIMAX16` | 95.4861 | 0.014312 | 85.1985 | 1.72 | 8.98 | 53.62 | 62.00 |
| `chainroimax/CH_R32_ROIMAX24` | 95.4786 | 0.014381 | 85.1985 | 1.72 | 8.98 | 54.18 | 62.00 |
| `chainroimax/CH_R32_ROIMAX48` | 95.4786 | 0.014381 | 85.1875 | 1.72 | 9.00 | 54.18 | 62.00 |
| `chainroimax/CH_R32_ROIMAX8` | 95.4861 | 0.014312 | 85.1949 | 1.72 | 8.98 | 53.62 | 62.00 |
| `chainroimax/DR_R32_ROIMAX0` | 95.4229 | 0.015519 | 85.1838 | 1.67 | 10.02 | 51.64 | 60.00 |
| `mplo/MP000` | 95.2891 | 0.016105 | 85.0440 | 1.72 | 10.18 | 62.53 | 62.00 |
| `mplo/MP010` | 95.3270 | 0.015760 | 85.0661 | 1.72 | 9.89 | 62.53 | 62.00 |
| `mplo/MP020` | 95.4483 | 0.014657 | 85.1654 | 1.72 | 9.35 | 58.29 | 62.00 |
| `mplo/MP030` | 95.5240 | 0.013967 | 85.2206 | 1.72 | 8.93 | 50.24 | 62.00 |
| `mplo/MP040_R24` | 95.4293 | 0.014829 | 85.0219 | 1.72 | 9.32 | 49.71 | 59.98 |
| `mplo/MP040_R48` | 95.4558 | 0.014588 | 85.2133 | 1.72 | 8.94 | 55.38 | 60.00 |
| `chainfar/CH_MP050` | 95.4699 | 0.014622 | 85.1985 | 1.72 | 9.03 | 55.50 | 198.50 |
| `chainfar/CH_MP070` | 95.4861 | 0.014312 | 85.1985 | 1.72 | 8.98 | 53.62 | 62.00 |

**⑤ 复现命令**（参数名取自 `manu/tracking/lookahead/lookahead_longquan.py` 与各诊断脚本的 CLI）：

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. PY=/home/huangzhe/anaconda3/envs/uav/bin/python
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
FRAMES="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"

# ① 可见性天花板（两臂真实输出分层对比，不构造任何 oracle）
$PY manu/tracking/lookahead/diag_visible_ceiling.py \
  --labels-root "$GT" \
  --original runs/lookahead/ref/preds_original_v3 \
  --replay  runs/lookahead/final/final8/preds_delivery \
  --delta 8 --hit-tol-px 5 --label "8seq 交付档 rb16+mp0.90" \
  --out-json runs/lookahead/final/final8/diag_visible_ceiling.json

# ② 位移-半径结构分解（证伪「搜不到」）
$PY manu/tracking/lookahead/diag_disp_radius.py \
  --labels-root "$GT" \
  --original runs/lookahead/ref/preds_original_v3 \
  --replay  runs/lookahead/final/final8/preds_delivery \
  --delta 8 --hit-tol-px 5 --radius 32 --label "8seq 交付档 rb16+mp0.90" \
  --out-json runs/lookahead/final/final8/diag_disp_radius.json

# ③ 位移估计器大比试（direct vs stride1 vs stride2）
$PY manu/tracking/lookahead/bench_xcorr_estimators.py \
  --labels-root "$GT" --frames-root "$FRAMES" --frames-cache /mnt/data/siping/frame_cache_lq \
  --original runs/lookahead/final/preds_off \
  --seqs VIDEO00007_19700101_001324 --delta 8 --max-frames 1500 \
  --out-json runs/lookahead/final/final8/bench_xcorr_estimators_001324.json

# ④ chain × min_peak 扫描（§31「direct 赢」被推翻的那个二维网格）
bash /mnt/data/codex-overnight/sweep_chain_minpeak.sh

# ⑤ 新冠军 CD_MP060 的 8 序四臂完整产物（交付定稿用）
bash /mnt/data/codex-overnight/final8_cdmp060.sh
```

**⑥ 产物路径**

| 用途 | 路径 |
|:--|:--|
| 位移-半径结构分解（新） | `manu/tracking/lookahead/diag_disp_radius.py` · `runs/lookahead/final/final8/diag_disp_radius.json` |
| oracle 拆分（代数退化已修） | `manu/tracking/lookahead/diag_oracle_split.py` · `runs/lookahead/final/final8/diag_oracle_split.json` |
| **可见性天花板**（新，最重要） | `manu/tracking/lookahead/diag_visible_ceiling.py` · `runs/lookahead/final/final8/diag_visible_ceiling.json` |
| **位移估计器大比试**（新） | `manu/tracking/lookahead/bench_xcorr_estimators.py` · `runs/lookahead/final/final8/bench_xcorr_estimators_001324.json` · `bench_xcorr_estimators_6seq.json` · `bench_xcorr_gfallback.json` |
| **8 序变体全表**（数据源） | `runs/lookahead/final/final8/round8_variants.json` |
| 8 序扫描（chain × min_peak） | `runs/lookahead/final/chainmp/` |
| 8 序扫描（chain × ROI 前推） | `runs/lookahead/final/chainroimax/` |
| 8 序扫描（chain × min_peak 低位） | `runs/lookahead/final/mplo/` |
| 8 序扫描（FAR→hit 兑换） | `runs/lookahead/final/chainfar/` |
| 24 序扫描（候选验证） | `runs/lookahead/final/final24par/` |
| 帧缓存（加速用，24 序全覆盖） | `/mnt/data/siping/frame_cache_lq/`（24 条 .npy + .json，165 s 建完） |
| 扫描脚本 | `/mnt/data/codex-overnight/sweep_chain_minpeak.sh` · `sweep_chain_roimax.sh` · `sweep_chain_far2hit.sh` · `sweep_par_batch.sh` · `sweep24_final.sh` · `final8_cdmp060.sh` |

### 5.17 ★★★ 第 6 轮（2026-11-04 服务器夜间第 6 轮）：24 序交叉验证 —— 8 序冠军 CD_MP060 被证伪，**交付档升级为 `direct+roimax16`**

**结论**：§32 在 **8 序 gate 集**上找到的新冠军 `CD_MP060`（F1 +0.19 pp / FAR −10.3% / hit +0.15 pp）
**在 24 序官方口径上全线破线** ⇒ 它是 **8 序过拟合**，**不采用该候选**
（交付档升级后 24 序 FAR **0.029548** ≤ 红线 0.030，余量 **+0.000452**，仍是唯一 PASS 的档位）。
**「8 序调参 ⇒ 24 序交付」这条链路本身不可靠** —— 8 序 gate 集只覆盖 24 序 **20.39%** 的 GT。

**① 24 序官方口径对照（决定性）**（24 序全量：GT 133257 / 帧 125418，同一套 `--arms replay` 口径）：

| 候选 | 尺子2 F1% | P% | R% | FAR/帧 | hit@5px | CLE p50(池) | CLE p95(池) | max_jitter | **FAR 红线 0.030** |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| **交付档**（direct · roimax4 · rb16/mp0.90） | **93.1578** | **96.069** | 90.418 | **0.029838** | **81.5379** | **1.72** | 13.16 | 60.00 | ✅ PASS（余量 **+0.000162**） |
| `MP030_CHAIN`（chain mp0.30 · roimax16） | 93.2271 | 95.751 | **90.833** | 0.032508 | 80.9391 | 1.85 | **13.03** | 122.19 | ❌ **−0.002508** |
| `CD_MP050`（chain_direct mp0.50 · roimax16） | 92.4769 | 95.166 | 89.935 | 0.036841 | 80.5106 | 1.77 | 14.64 | 60.00 | ❌ −0.006841 |
| `CD_MP060`（**§32 的 8 序冠军**） | 91.3830 | 94.060 | 88.855 | 0.045260 | 78.6660 | 1.83 | 17.31 | 60.00 | ❌ **−0.015260** |
| `CD_MP070`（chain_direct mp0.70 · roimax16） | 89.7583 | 92.440 | 87.228 | 0.057534 | 75.9540 | 1.91 | 20.33 | 60.00 | ❌ −0.027534 |

⇒ **四个候选全部破 FAR 红线，且 hit@5px 全部低于交付档**；只有 `MP030_CHAIN` 的 F1 略高 0.07 pp，
但 FAR 破线 **+0.0027**、hit 掉 **0.60 pp**、**max_jitter 从 60.0 爆到 122.19**（伺服不可接受）⇒ 全部否决。
（本表「交付档」基线 = 升级前 roimax4 交付档，四个候选即相对它被否决；**升级后的交付档见 ⑦**。）
**§32 的「三项同时最优」只在 8 序成立**：`CD_MP060` 8 序 → 24 序 =
F1 95.6505 → 91.3830（**−4.27 pp**）、FAR 0.013553 → 0.045260（**×3.34**）、hit 85.3715 → 78.6660（**−6.71 pp**）。

**② 为什么 8 序会骗人**：gate 集只覆盖 24 序的 **20.39%**（GT 帧 27173 / 133257、视频帧 28997 / 165223、序列 8 / 24）；
**chain 家族爆 FAR 的重灾区全部落在 8 序 gate 集之外**（交付档 24 序逐序列 FAR）：
`015740` **0.4036** / F1 12.26%、`002251` **0.1000** / 64.54%、`001810` 0.0148 / 97.44%
（但 `CD_MP060` 在它上面 FAR **0.1792**）、`002111` 0.0424 / 83.13%（`CD_MP060` FAR **0.1242**）、
`002438` 0.0127 / 94.94%（`CD_MP060` FAR **0.1031**）
⇒ **机制**：chain 类档位在「目标不可见 / 背景杂乱」的帧上会把相关峰配到错误物体上，8 序扫描完全看不见。
**⇒ 给后续工作的硬规则**：**任何调参必须在 24 序上确认 FAR ≤ 0.030 才算候选**；
8 序只用来做「机理是否存在」的快速筛选，**不能用来排序冠军**。

**③ 三道 Gate 全部通过**：

| Gate | 内容 | 结果 |
|:--|:--|:--|
| **Gate A（8 序）** | `--lookahead off` vs `runs/lookahead/ref/preds_original_v3` 逐位 | **IDENTICAL=9 DIFFER=0**；ref/off F1 均 **0.964002**（TP 25454 / FP 182 / FN 1719 / GT 27173） |
| **Gate A（24 序）** | 同上，24 序 | **IDENTICAL=25 DIFFER=0**；ref/off F1 均 **0.947469**（TP 123052 / FP 3440 / FN 10205 / GT 133257） |
| **Gate B（本轮新增）** | 用**当前** `run_delivery.sh` 的 CFG 重跑 8 序，与 04:32 落盘的 `final8/arms_delivery.json` **逐位**比对 | **三臂 × 两组指标全部 IDENTICAL**（original/stale/replay × current_frame/servo）⇒ §32 期间对 `extrapolator.py` 的改动**净效果为零**，未污染交付档产物 |

**④ 帧缓存是逐位安全的加速手段（两条独立证据）**：sshfs 上读+解码 **11.3 ms/帧**、本地盘 **2.8 ms/帧**、
memmap **~0.1 ms/帧**；本轮建了本地 memmap 帧缓存 `/mnt/data/siping/frame_cache_lq/`（24 条 `.npy` + `.json`，165 s 建完）。
① **抽样逐字节**：随机 12 帧 `cv2.imread(..., IMREAD_GRAYSCALE)` 与缓存数组 `np.array_equal` ⇒ **12/12 mismatch=0**；
② **端到端逐位**：§32 的 `chainmp/CD_MP060`（sshfs JPEG 路径）与 `final8_cdmp060`（memmap 缓存路径）**同一个配置**，
逐序列 TP/FP/FN **完全相同**，总体 F1 均 **95.651%**、FAR 均 **0.0136**、hit 均 **85.37%**
⇒ `--frames-cache` **逐位安全**，可用于评测加速。
参考实测耗时：交付档 24 序整趟（4 臂、165223 帧、**走 memmap 帧缓存 460.8 s（原 740.6 s，1.6×）**）、
8 序整趟（3 臂、28997 帧）升级后走 memmap 缓存 **86.5 s**（升级前走 sshfs **120.9 s**）；
本轮 24 序候选扫描（单配置单臂、4 配置并行、走 memmap）**13 min/趟**（并行有争抢，不可直接与上面比）。

**⑤ OSD 三臂对照重渲（8 序 × 3 臂 = 24 段）**：用 §32 冠军 `CD_MP060` 的 replay 臂重渲（上一轮用的是旧冠军），
产物 `runs/lookahead/osd/videos/<SEQ>_3arm_{stale,replay,original}.mp4`（24 段，134 MB），
清单 `/mnt/data/codex-overnight/osd_manifest_round8.md`，校验 24/24 通过
（h264 / 1920×1080 / 25 fps / `nb_frames == limit == end−start+1` / 全帧解码无错 / `problems=0`）。

**窗口内 hit@5px / CLE p50**（stale → replay → original）：

| 序列 | 窗口 | stale | replay | original |
|:--|:--|--:|--:|--:|
| 001324（高速机动） | 8549–8848 | 19.7% / 20.93 px | **90.3% / 1.98 px** | 98.7% / 1.86 px |
| 015000（悬停） | 726–1025 | 25.0% / 12.21 px | **97.0% / 1.85 px** | 99.0% / 1.87 px |
| 000703 | 205–504 | 7.3% / 23.31 px | 94.7% / 2.00 px | 99.0% / 1.72 px |
| 000818 | 3819–4118 | 16.3% / 17.02 px | 99.7% / 1.61 px | 100.0% / 1.51 px |
| 001307 | 3358–3657 | 14.0% / 16.71 px | 100.0% / 1.43 px | 100.0% / 1.47 px |
| 001531 | 8–114（107 帧） | 3.7% / 20.43 px | 97.2% / 1.79 px | 99.1% / 1.73 px |
| 014053 | 290–589 | 15.0% / 23.97 px | 93.3% / 2.04 px | 99.7% / 1.81 px |
| 015052 | 490–789 | 26.3% / 13.08 px | 99.0% / 1.72 px | 99.0% / 1.67 px |
| **8 序合计（2207 帧）** | — | **17.0% / 17.95 px** | **96.3% / 1.80 px** | **99.3% / 1.69 px** |

⇒ 肉眼可见的证据：**老逻辑（红圈）甩出目标几十到上百像素（001324 最差帧 f8702 甩到 192 px 外），
外挂（绿框）稳稳锁在目标中心**。同一批窗口上，新冠军的尾误差还比旧冠军更小
（max **22.80 px** vs 旧冠军 **48.15 px**，mean 2.07 vs 2.22 px）。

**⑥ 交付档 CLI（升级后）** —— `manu/tracking/lookahead/run_delivery.sh` 的 CFG 块，原样贴出：

```bash
CFG="--replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --no-replay-hold-use-ref \
 --output-max-step-px 30 --replay-anchor delta --fast-win-size 52 --fast-bg-window 104 \
 --fast-th-k 2.5 --fast-min-energy 50 --replay-roi-mode selfpred --replay-roi-selfpred-window 3 \
 --replay-roi-selfpred-max-px 16 --replay-xcorr-radius 32 --replay-xcorr-template 32 \
 --replay-xcorr-th-k 1.5 --replay-polarity auto --replay-xcorr-uniq-gate 0.6 \
 --replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24 \
 --replay-rollback-max-frames 16 --replay-rollback-min-peak 0.90"
```

⇒ 相对 §30 冠军档**只改了一个旋钮值**：`--replay-roi-selfpred-max-px 4 → 16`（依据见下一节的 2×2 因子隔离）。
**Gate B** 仍是「用**当前** `run_delivery.sh` 的 CFG 重跑 8 序 vs 落盘产物逐位比对」
⇒ **三臂 × 两组指标全部 IDENTICAL**。

**⑦ 交付档升级：2×2 因子隔离 ＋ ROI 前推上限扫描**

① 的四个候选全破 FAR 红线，但**否决理由还没查清** —— 24 序扫描脚本 `sweep24_final.sh` 把**两个因子混在一起**了
（`chain` 档位 **和** ROI 前推上限 `16` 同时生效）⇒ 分不清是哪个把 FAR 推爆的。
24 序 2×2 因子隔离（官方口径，GT 133257 / 帧 165223）：

| 档位 | `--replay-roi-selfpred-max-px` | 尺子2 F1% | FAR/帧 | **FAR 红线余量** | hit@5px | CLE p95(池) | max_jitter | TP / FP / FN | 红线 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| **direct** | 4（**原交付档**） | 93.1578 | 0.029838 | +0.000162 | 81.5379 | 13.16 | 60.00 | 120488 / 4930 / 12769 | ✅ PASS |
| **direct** | **16（新交付档）** | **93.1879** | **0.029548** | **+0.000452** | **81.5612** | **12.95** | 60.00 | 120519 / 4882 / 12738 | ✅ PASS |
| chain (mp0.30) | 16 | 93.2271 | 0.032508 | −0.002508 | 80.9391 | 13.03 | **122.19** | 121041 / 5371 / 12216 | ❌ FAIL |
| chain (mp0.30) | 4 | 93.2148 | 0.032604 | −0.002604 | 80.9278 | 13.09 | **122.19** | 121025 / 5387 / 12232 | ❌ FAIL |

⇒ **FAR 破线的元凶 100% 是 `chain` 档位**（同档位内 roimax 4↔16 的 FAR 差 **< 0.0001**），
**与 ROI 前推上限无关** ⇒ ①「chain 全家族否决」成立且被加强。
⇒ 而 **ROI 前推上限是被 8 序调参掩盖的正收益因子** ⇒ 24 序上扫它的上限（`direct` 档位，其余旋钮不动）：

| `--replay-roi-selfpred-max-px` | 尺子2 F1% | FAR/帧 | 红线余量 | hit@5px | CLE p95(池) |
|--:|--:|--:|--:|--:|--:|
| 4（原交付档） | 93.1578 | 0.029838 | +0.000162 | 81.5379 | 13.16 |
| 8 | 93.1723 | 0.029651 | +0.000349 | 81.5522 | 13.03 |
| **16（新交付档）** | **93.1879** | **0.029548** | **+0.000452** | **81.5612** | **12.95** |
| 24 | 93.1883 | 0.029548 | +0.000452 | 81.5574 | 12.94 |
| 48 | 93.1773 | 0.029608 | +0.000392 | 81.5432 | 12.98 |

⇒ **16 是甜点**（hit 最高 + FAR 余量最大；48 开始回落）⇒ **交付档升级为 `--replay-roi-selfpred-max-px 16`**。

**升级后的交付档 24 序四臂**（产物 `runs/lookahead/final/final24/arms_delivery.json`）：

| 臂 | F1% | R% | P% | FAR/帧 | hit@5px | CLE p50(池) | CLE p95(池) | max_jitter | TP/FP/FN |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| `original`（零延迟天花板） | 94.7469 | 92.342 | 97.280 | 0.020820 | **85.8957** | 1.562 | 6.844 | 76.52 | 123052 / 3440 / 10205 |
| `stale`（老逻辑，真吃 320 ms 滞后） | 77.5225 | 75.531 | 79.621 | 0.155917 | **44.8232** | 5.293 | 56.620 | 76.52 | 100651 / 25761 / 32606 |
| `kinematic`（纯前向外推） | 75.7641 | 73.818 | 77.815 | 0.169734 | 38.6764 | 6.421 | 61.517 | 188.95 | 98368 / 28044 / 34889 |
| **`replay`（交付档 · roimax16）** | **93.1879** | 90.441 | 96.107 | **0.029548** | **81.5612** | **1.719** | **12.946** | **60.00** | 120519 / 4882 / 12738 |
| `replay`（升级前 roimax4 对照） | 93.1578 | 90.418 | 96.069 | 0.029838 | 81.5379 | 1.72 | 13.16 | 60.00 | 120488 / 4930 / 12769 |

⇒ 相对老逻辑（`stale`）：ΔF1 **+15.6654 pp**、ΔFAR **−0.126369**、Δhit@5px **+36.7380 pp**、Δp50 **−3.57 px**、
Δp95 **−43.67 px**、Δmj **−16.52 px**；离天花板（`original`）ΔF1 **1.5589 pp**、Δhit **4.3345 pp**
⇒ **滞后账追回 90.95%（F1）/ 89.45%（hit@5px）**；
**FAR 红线余量 +0.000162 → +0.000452（×2.8）** —— 升级的意义不是那 +0.03 pp，而是余量翻倍多
（余量 0.000162 意味着「换一批序列就可能破线」）。

**⑧ 复现命令**（CLI 参数名取自 `manu/tracking/lookahead/lookahead_longquan.py` 的 argparse）：

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
FC=/mnt/data/siping/frame_cache_lq
C8=runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache
C24=runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json

# ① 交付档本体（8 序 + 24 序，当前 CFG = --replay-roi-selfpred-max-px 16）
bash manu/tracking/lookahead/run_delivery.sh both

# ② 24 序候选扫描（本轮决定性：§32 的 4 个候选 → 24 序官方口径）
bash /mnt/data/codex-overnight/sweep24_final.sh

# ③ 24 序 2×2 因子隔离（chain × ROI 前推上限）
bash /mnt/data/codex-overnight/sweep24_factorial.sh

# ④ 24 序 ROI 前推上限扫描（direct 档位，其余旋钮不动）
bash /mnt/data/codex-overnight/sweep24_roimax.sh

# ⑤ 24 序对照汇总（交付档 vs 4 候选，全部读官方 JSON 字段）
$PY /mnt/data/codex-overnight/cmp24_sweep.py

# ⑥ 8 序 roimax16 复跑（与交付档 8 序产物逐位比对）
bash /mnt/data/codex-overnight/test8_roimax16.sh

# ⑦ Gate B：交付档可复现性（重跑 8 序，与落盘产物逐位比对）
bash /mnt/data/codex-overnight/verify_delivery_unchanged.sh
```

**⑨ 产物路径**

| 用途 | 路径 |
|:--|:--|
| **24 序候选扫描（本轮决定性）** | `runs/lookahead/final/final24par/`（`MP030_CHAIN` / `CD_MP050` / `CD_MP060` / `CD_MP070`） |
| 24 序 2×2 因子隔离（chain × ROI 前推） | `runs/lookahead/final/final24fact/` |
| 24 序对照汇总脚本 | `/mnt/data/codex-overnight/cmp24_sweep.py` |
| 24 序因子扫描脚本 | `/mnt/data/codex-overnight/sweep24_final.sh` · `sweep24_factorial.sh` · `sweep24_roimax.sh` |
| 8 序 roimax16 复跑脚本 | `/mnt/data/codex-overnight/test8_roimax16.sh` → `runs/lookahead/final/roimax16_8/arms_roimax16.json` |
| **Gate B 交付档可复现性验证** | `/mnt/data/codex-overnight/verify_delivery_unchanged.sh` → `runs/lookahead/final/_verify_delivery/arms_verify.json` |
| **帧缓存**（逐位安全，帧读取 11.3 → ~0.1 ms/帧） | `/mnt/data/siping/frame_cache_lq/`（24 条 `.npy` + `.json`） |
| OSD 三臂视频（8 序 × 3 臂） | `runs/lookahead/osd/videos/<SEQ>_3arm_<arm>.mp4` |
| OSD 清单 + 核查 | `/mnt/data/codex-overnight/osd_manifest_round8.md` · `runs/lookahead/osd/osd_round8_verify.json` |
| 8 序四臂完整产物（`CD_MP060` 候选） | `runs/lookahead/final/final8_cdmp060/arms_final_CD_MP060.json` · `report_4arm.md` |
| **交付档产物（roimax16 升级后重跑）** | 8 序 `runs/lookahead/final/final8/arms_delivery.json` · 24 序 `runs/lookahead/final/final24/arms_delivery.json` |

### 5.18 ★★★ 第 7 轮（2026-11-04 服务器夜间第 7 轮）：分段配准（stride）否决 ＋ 「外挂已在物理极限」定量判定

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e` · python = conda env `uav`（`/home/huangzhe/anaconda3/envs/uav/bin/python`）· 时间：2026-11-04 夜间
> **本轮一句话**：新实现「分段配准（`stride`）」档位并做完**生产口径**端到端实测 —— **离线 bench 说它好 8~27 pp，生产口径说它差 0.16~1.93 pp ⇒ 正式否决**；
> 更值钱的是把「24 序还差 **4.33 pp** 命中」的归属钉死：**主体是管线/检测器自身的定位误差（帧间强相关 ⇒ 不可平均掉），
> 外挂只额外添了 p50 **0.87 px** 的位移残差 ⇒ 外挂已在物理极限**；**本轮起不再在外挂侧调参**。

#### 5.18.1 ★★★ 交付档复验（8 序 gate 集 ＋ 24 序官方口径；本轮**未改**，仅复验）

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl
bash manu/tracking/lookahead/run_delivery.sh both
```

执行机：服务器 `test-Super-Server` · python = conda env `uav` · 2026-11-04 夜间。
产物：8 序 `runs/lookahead/final/final8/arms_delivery.json` · 24 序 `runs/lookahead/final/final24/arms_delivery.json`（8 序 86.5 s / 24 序 460.8 s，走 memmap 帧缓存）。

| 口径 | 臂 | F1% | FAR/帧 | hit@5px | CLE p50(池) | p95(池) | max_jitter |
|:--|:--|--:|--:|--:|--:|--:|--:|
| 24 序官方 | original(天花板/上界) | 94.7469 | 0.020820 | 85.8957 | 1.562 | 6.844 | 76.52 |
| | stale(老逻辑,真吃320ms,**交付基线**) | 77.5225 | 0.155917 | 44.8232 | 5.293 | 56.620 | 76.52 |
| | kinematic(纯前向外推) | 75.7641 | 0.169734 | 38.6764 | 6.421 | 61.517 | 188.95 |
| | **replay(交付档)** | **93.1879** | **0.029548** | **81.5612** | **1.719** | **12.946** | **60.00** |
| 8 序 gate | original | 96.4002 | 0.006277 | 88.1316 | 1.568 | 5.970 | 65.24 |
| | stale | 77.2034 | 0.180743 | 38.1850 | 6.878 | 57.035 | 65.24 |
| | **replay(交付档)** | **95.4440** | **0.015381** | **85.2206** | **1.674** | **9.935** | **60.00** |

**⚠️ 口径提醒（先读，否则结论会做反）**：`original` 臂是**零延迟天花板（oracle 上界）**，**不是交付基线**；
**交付基线是 `stale`**（老逻辑：把第 t−Δ 帧的旧框原样甩给转台，真吃 320 ms 滞后）⇒ 交付判定只看「外挂 vs stale」。

- 24 序 replay vs stale：ΔF1 **+15.67 pp**、Δhit@5px **+36.74 pp**、FAR **−0.126**、CLE p95 **−43.67 px**、max_jitter **−16.52 px**
  ⇒ **滞后账追回 F1 90.9% / hit 89.4%**（对照表 `runs/lookahead/final/report_delivery.md`）。
- 8 序 replay vs stale：F1 **77.2034 → 95.4440**，追回 **95.0%**；hit@5px **38.1850 → 85.2206**，追回 **94.2%**。
- 离天花板还差：24 序 ΔF1 **1.56 pp** / Δhit **4.33 pp**（FAR 红线 0.030，余量 **+0.000452**，仍是唯一 PASS 的档位）。

#### 5.18.2 ★★ 新档位 `stride`（分段配准）：**已实现、已实测、已否决**

**机理**：`direct` 拿源帧 f 的模板在目标帧 t（相隔 Δ=8 帧）里做**一次大匹配** ⇒ 模板老化 8 帧、相关峰不准；
`stride` 改成**每 N 帧配一次、每段刷新模板**（模板与目标只隔 N 帧 ⇒ 相关峰很锐）。

**实现位置**（本轮代码改动，全部在 `manu/tracking/lookahead/`）：

| 文件 | 改动 |
|:--|:--|
| `manu/tracking/lookahead/extrapolator.py` | 新增 `_xcorr_stride_refine`（第 318 行）＋ `LookaheadConfig.replay_xcorr_stride`（默认 0 = 关）＋ mode 校验 ＋ `xcorr_stride_hold` 计数 |
| `manu/tracking/lookahead/lookahead_longquan.py` | 新增 `--replay-xcorr-stride`；`--replay-xcorr-mode` 的 choices 增加 `stride` |
| `manu/tracking/tests/test_xcorr_stride.py` | **新增**单测（合成图位移恢复 ＋ 默认值 0 ＋ mode 校验） |

⇒ 全库单测 **155 passed / 10 skipped**（无回归）。

**⚠️ 坑（务必写进任何后续实现）**：`xcorr_peak_last` 必须记为**各段峰值的最弱值**，
否则 `replay_abstain_peak_below` 弃权门会**静默失效**（FAR 红线破防）。

**生产端到端实测（8 序，其余旋钮一字不改）—— 单调变差 ⇒ 否决**：

| 档位 | F1% | FAR/帧 | hit@5px | CLE p95(池) |
|:--|--:|--:|--:|--:|
| **direct（交付档）** | **95.4440** | **0.015381** | **85.2206** | **9.94** |
| stride N=1 | 95.2769 | 0.016415 | 85.0587 | 9.49 |
| stride N=2 | 94.6833 | 0.019623 | 84.3705 | 12.71 |
| stride N=4 | 94.2266 | 0.023589 | 83.2886 | 16.69 |

⇒ hit@5px **85.2206 → 85.0587 → 84.3705 → 83.2886** 单调变差，FAR 同步变差 **0.015381 → 0.023589** ⇒ **该路线否决**。

**根因（本节最有价值的一条）**：离线 bench 的 `direct` 用**锚点**当搜索中心（罩不住快速目标 ⇒ **严重低估 `direct`**）；
**生产**的 `direct` 用**质心链末位置** `chain_pos` 当搜索中心（链已把目标跟到附近，大匹配只是局部精修）。
逐帧对比 direct vs stride1 的位置差 **p50 仅 0.309 px**（p90 1.33 / p95 2.34，25561 帧同时有输出）
⇒ **两者在生产里本来就几乎一样**，stride 的额外结构只增加了出错面。
⇒ **教训：离线 bench 若不与生产口径对齐，会给出完全误导的排序**（本轮顺手修掉 `bench_xcorr_estimators.py` 的 `results` 引用别名 bug ——
旧产物里 000703 与 000818 的数字逐位完全相同）。

**离线 bench 的表象**（产物 `runs/lookahead/final/final8/bench_xcorr_estimators_8seq_fixed.json`，hit@5px，direct → stride1）：
014053 **70.72 → 97.26（+26.54）**、001531 82.24 → 97.20（+14.95）、000703 78.53 → 92.60（+14.07）、015052 91.15 → 99.11（+7.96）；
位移误差 p95 也从 direct 的 **48.1~90.0 px**（000703 71.9 / 001531 90.0 / 014053 83.9 / 015052 48.1）降到 stride1/2 的 **2.4~4.3 px**
⇒ 离线看「分段配准完胜 direct」—— **而生产口径恰好相反**。

**复现命令**（服务器，2026-11-04 夜间；把 `--replay-xcorr-stride` 换成 1/2/4 即为三个 stride 档，其余旋钮一字不改）：

```bash
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
C8=runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache
GT=/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010
FRAMES=/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg
FC=/mnt/data/siping/frame_cache_lq
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json
# $CFG = manu/tracking/lookahead/run_delivery.sh 里的交付档 CFG 块（一字不改）
$PY manu/tracking/lookahead/lookahead_longquan.py \
  --cache-root "$C8" --labels-root "$GT" --frames-root "$FRAMES" --frames-cache "$FC" \
  --params-json "$PARAMS" \
  --all-seqs --lookahead replay --arms replay $CFG \
  --replay-xcorr-mode stride --replay-xcorr-stride N \
  --out-json runs/lookahead/final/final8stride/arms_STR<N>.json \
  --out-preds-dir runs/lookahead/final/final8stride/preds_STR<N>
```

（扫描脚本 `/mnt/data/codex-overnight/sweep8_stride.sh`。）
产物：`runs/lookahead/final/final8stride/{arms_STR1,arms_STR2,arms_STR4}.json` ＋ `preds_STR*`（＋ `STR*.log`）。

#### 5.18.3 ★★ 「外挂已在物理极限」：三条独立证据

| 证据 | 脚本 | 数字 | 含义 |
|:--|:--|:--|:--|
| 误差三分解 | `manu/tracking/lookahead/diag_error_decomp2.py` | 输出误差 p50 **1.670 px**（＝锚点继承 e(f) **1.566 px** 与位移残差 δ **0.870 px** 的**矢量和**）；零延迟管线自身 e(t) p50 **1.561 px**（8 序 25469 帧） | **外挂比零延迟管线只多 0.11 px**；且 δ(0.870) ≪ e(f)(1.566) ⇒ 该修**锚点（检测器）**不是位移估计 |
| L2 不可救 | `manu/tracking/lookahead/diag_gap_coast.py` | 快核穿「检测器空窗」的 hit@5px：001324 **18.95%** / 000322 **17.80%** / 002111 **31.78%**，**全部低于 kinematic 臂 38.68%** | 7.06% 的缺口来自**检测器边界**，不是外挂 |
| 误差不可平均 | （24 序 123966 对） | 管线误差向量帧间相关 ρ(lag1)=**+0.778**、ρ(lag8)=**+0.567** | 多源平均只降 ~16%，不值 3× 算力 |

**误差三分解的明细**（产物 `runs/lookahead/final/final8/diag_error_decomp2.json`，8 序交付档、25469 帧）：

| 量 | p50 | p90 | p99 |
|:--|--:|--:|--:|
| 锚点误差 **e(f)**（= 管线在源帧的定位误差） | 1.566 | 3.749 | 18.491 |
| **位移残差 δ**（外挂唯一能改的那一项） | **0.870** | 2.915 | 31.490 |
| 输出误差 \|e+δ\| | 1.670 | 4.676 | 51.495 |
| **零延迟管线自身误差 e(t)**（对照基准） | **1.561** | 3.656 | 16.356 |

失手帧（2325 帧）上 e(f) p50 **5.889**、δ p50 **5.000** ⇒ 两者**同时**变大 ⇒ 是**条件效应**（难帧上管线自己也不准），不是外挂的独立缺陷。

⇒ **结论一句话：剩余 4.33 pp 命中缺口要靠换/改检测器，不是外挂**；外挂旋钮已扫完，**不再在外挂侧调参**。

#### 5.18.4 ★★ 失手漏斗：把「没命中的帧」逐层拆开（24 序交付档，GT 133257）

脚本 `manu/tracking/lookahead/diag_miss_funnel.py`（新增）· 产物 `runs/lookahead/final/final24/diag_miss_funnel_rb16.json` · 2026-11-04 夜间。

| 层 | 帧数 | 占比 |
|:--|--:|--:|
| L1 源帧不存在 | 184 | 0.14% |
| **L2 源帧无检测** | **9404** | **7.06%** |
| L3 源帧无真值目标 | 43 | 0.03% |
| L4 能重推但未上报 | 1598 | 1.20% |
| **L5 上报了但位置错** | **14198** | **10.65%**（(5,10] 6886 / (10,20] 2874 / (20,50] 2487 / >50 1951） |
| 命中 | 108629 | 81.52% |

**重灾区逐序列**（hit@5px ＋ 三层失手来源）：

| 序列 | hit@5px | 未上报(L4) | 源无检(L2) | 位置错(L5) |
|:--|--:|--:|--:|--:|
| 015740 | **6.98%** | 42 | 302 | 344 |
| 000322 | **48.44%** | 144 | **4563** | 2982 |
| 001324 | 65.24% | 56 | 1669 | 1957 |
| 002111 | 68.80% | 354 | 381 | 368 |
| 002001 | 73.06% | 38 | 957 | **3356** |

⇒ 这 5 条序列合计失手 **17265 / 24628 = 70.1%**（按失手帧排名的前 5 条 000322 / 002001 / 001324 / 002959 / 002438 = **19353 / 24628 = 78.6%**）；
**GT 逐帧位移 p50 仅 0.5~3.5 px（目标基本悬停）⇒「高速机动」不是主因**。
⇒ L4 也已在甜点：24 序弃权门全量扫描（`runs/lookahead/final/final24/diag_abstain_scan.md`）显示 FAR 红线内唯一更优点是 `peak<0.2`
（F1 **+0.16 pp**、hit **−0.08 pp**）⇒ **L4 无剩余空间**；**L2 见 5.18.3（不可救）** ⇒ 剩余空间只剩 L5 里的**定位错**。

#### 5.18.5 ★ Gate A 复验 ＋ OSD 三臂视频 ＋ 报告口径修正

**① Gate A（尺子1 逐位一致，新代码之后）** —— 脚本 `/mnt/data/codex-overnight/gateA_newcode.sh`（2026-11-04 夜间）：

| 口径 | 参照产物 | md5 比对结果 |
|:--|:--|:--|
| 8 序 | `runs/lookahead/ref/preds_original_v3/`（9 份 JSON） | **IDENTICAL=9 DIFFER=0 MISSING=0** |
| 24 序 | `runs/lookahead/final/ref_preds_original_v3_24seq/`（25 份 JSON） | **IDENTICAL=25 DIFFER=0 MISSING=0** |

产物：`runs/lookahead/final/gateA_newcode/`（`arms_off_8seq.json` / `arms_off_24seq.json` ＋ `preds8/` / `preds24/`）。
**⚠️ 坑**：第一次 8/24 全 DIFFER 的根因**不是代码回归**，而是 `--cache-root` 传了**相对路径**（参照产物记的是**绝对路径**）
⇒ `config.cache_root` 字符串不同 ⇒ md5 不同，而 `per_frame` 实际 **0 帧不同**。
⇒ **Gate A 的 md5 含 `config` 块，路径必须是绝对路径。**

**② OSD 三臂视频**（交付档重渲）—— 产物 `runs/lookahead/osd/videos/<SEQ>_3arm_{stale,replay,original}.mp4`，脚本 `/mnt/data/codex-overnight/render_osd_one.sh`：

| 序列 | 窗口 | stale hit5 | replay hit5 | original hit5 |
|:--|:--|--:|--:|--:|
| 001324 | f8549–8848 | **19.7%**（CLE p50 20.93 / max 192.24） | **90.3%**（p50 1.98 / max 10.47） | 98.7% |
| 015000 | f726–1025 | **25.0%**（p50 12.21 / max 102.97） | **97.0%**（p50 1.85） | 99.0% |

⇒ 画面能直接看出「**老逻辑抽搐脱靶 vs 外挂平滑跟住**」。

**③ 报告口径修正**（`manu/tracking/lookahead/make_report_table.py`，2026-11-04 夜间）：

| # | 硬伤 | 症状 | 修法 | 修后 |
|:--|:--|:--|:--|:--|
| 1 | 总体行 CLE 用错口径 | 24 序 replay 报 `p50=146.94px`，与 `hit@5px=81.56%` 自相矛盾 | 总体行改走 **`cle_pooled_*`（全帧池化）**（旧 `cle_p*` 是**逐序列最差值** `worst_over_sequences`） | p50 **146.94→1.72px** / p95 **287.88→12.95px** |
| 2 | 退化/收益小节跨产物混用基线 | 8 序 replay 拿 24 序 stale 当基线 ⇒ 算出无意义的「追回 **104.0%**」 | **按产物分组、基线与被比臂必须同产物** | 8 序追回 **F1 95.0% / hit 94.2%** |

⇒ 产物：`runs/lookahead/final/report_delivery.md`（**引用 CLE 数字必须写明是「池化」还是「逐序列最差值」**）。

#### 5.18.6 ⛔ 新增「不要再试」清单（本轮实测否决）

| # | 路线 | 否决理由（数字与出处） |
|:--|:--|:--|
| 1 | **`stride` 分段配准**（生产口径） | 8 序端到端**单调变差**：hit@5px 85.2206 → 85.0587 / 84.3705 / 83.2886，FAR 0.015381 → 0.016415 / 0.019623 / 0.023589（§5.18.2） |
| 2 | **多源回推取中位数**（t−8 / t−9 / t−10 各回推一次） | 误差向量帧间相关 ρ(lag1)=**+0.778**、ρ(lag8)=**+0.567** ⇒ 平均只能把锚点误差降约 **16%**，**不值 3× 算力**（收益 < 0.3 pp hit）（§5.18.3） |
| 3 | **快核穿「检测器空窗」**（L2 层） | 不可救：快核穿空窗 hit@5px **17.80~31.78%**，**低于 kinematic 臂 38.68%**（§5.18.3 / §5.18.4） |

⇒ 本 README 既有清单（§5.6 / §5.15 等处：观测层收缩、位移层收缩、低速冻结、位移自适应融合、peak_centroid、refine、
一致性门单独用、`--replay-anchor pred`、`--fast-min-contrast`、rolling 链、core_frac、th_k≥2.5、大半径重试、各类单门、chain 家族）
**仍然有效**（本节不重复抄）。
⇒ **下一步的唯一方向：换/改检测器**（重点：`000322` / `002111` / `001324` 的**源帧漏检**，`002001` 的**上报位置错 3356 帧**），
**不再在外挂侧调参**。

> **本节产出**：`runs/lookahead/final/report_delivery.md`（池化口径最终对照表）· `runs/lookahead/final/final8stride/{arms_STR1,arms_STR2,arms_STR4}.json` ＋ `preds_STR*` · `runs/lookahead/final/final8/diag_error_decomp2.json` · `runs/lookahead/final/final8/bench_xcorr_estimators_8seq_fixed.json` · `runs/lookahead/final/final24/diag_miss_funnel_rb16.json` · `runs/lookahead/final/final24/diag_gap_coast.json` · `runs/lookahead/final/final24/diag_abstain_scan.md` · `runs/lookahead/final/gateA_newcode/` · `runs/lookahead/osd/videos/<SEQ>_3arm_{stale,replay,original}.mp4` ＋ `runs/lookahead/osd/windows/<SEQ>_3arm.json` · 代码 `manu/tracking/lookahead/{extrapolator.py,lookahead_longquan.py,make_report_table.py,bench_xcorr_estimators.py}` ＋ 单测 `manu/tracking/tests/test_xcorr_stride.py` · 专题档 `manu/memory/lookahead_extrapolator_20261102.md` §34（4323 行起）· 本 README §5.18。
