# 板端移植契约 · 快速前向推演外挂（Look-ahead Extrapolator）

> **执行机**：服务器 `test-Super-Server`（80 核 / 125 GB）· cwd `/tmp/pycharm_project_10ae9e2e`
> **python**：`/home/huangzhe/anaconda3/envs/uav/bin/python`（conda env `uav`，Python 3.10.20 / numpy 2.2.6 / cv2 4.10）
> **本文档时间**：2026-11-03 夜间（服务器第 3 段时段 21:00–22:20 之后）
> **代码基线**（md5 已于 2026-11-03 夜间复核，与 `manu/memory/handoff_20261103_server_round2.md` §六 一致）：

| 文件 | md5 | 角色 |
|:--|:--|:--|
| `manu/tracking/lookahead/polarity.py` | `2cd585ad87bad8438f6ecfc7272f0553` | 极性自适应（新建） |
| `manu/tracking/lookahead/xcorr.py` | `56bcc444a250988ebab57a377434d7c1` | 模板匹配位移精修 |
| `manu/tracking/lookahead/extrapolator.py` | `b0e010c2c326241392e85e1407f33985` | 外挂主逻辑 |
| `manu/tracking/lookahead/lookahead_longquan.py` | `5bc5ec136d7c02eac51639134833b09a` | CLI / 管线拷贝 |
| `manu/evaluation/system_sota_longquan.py` | `3361e7b70e76d3a92346fd8fd9879dad` | **零改动**（原始管线） |

> **事实来源**：本文件所有数字均来自仓库产物 JSON（`runs/lookahead/final/**`）或 `manu/memory/lookahead_extrapolator_20261102.md` §25；
> 口径复算与 Gate A md5 比对已在本机独立复核（2026-11-03 夜间）。**不含任何推测值**。

---

## 0. 定位：外挂是什么、不做什么

| 项 | 内容 |
|:--|:--|
| **是什么** | 挂在既有管线**输出末端**的一层**无状态**前向推演：物理时刻 `t` 手上只有检测框 `D(t−Δ)`（Δ=8 record ≈ **320 ms** @25 Hz），外挂用**帧缓存**把它**逐帧重推 8 步**到 `t`，得到当前帧坐标交给转台伺服 |
| **不做什么** | 不碰检测、不碰跟踪、不碰卡尔曼；**不回写任何跟踪器状态**；不改 `manu/evaluation/*`（拷贝纪律，原文件零改动） |
| **四臂语义** | `off`（原样搬运，历史行为）· `stale`（`report(t)=D(t−Δ)`，老逻辑基线）· `kinematic`（`D(t−Δ)+Δ·v`，档位 A，**已否决**）· `replay`（视觉重推，档位 B，**交付方案**） |
| **交付配置** | 冠军 `WTH15`（xcorr 模板 32 / 半径 32 / th_k 1.5 + 快核 win52/bg104/th_k 2.5/energy 50 + ROI selfpred(3, ≤4 px) + `--output-max-step-px 30 --replay-anchor delta --replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --no-replay-hold-use-ref`）**＋ `--replay-polarity auto`**（其余参数一字不改） |
| **拷贝纪律** | `manu/tracking/lookahead/` 下 `track_fusion.py` / `lookahead_longquan.py` 是 `manu/evaluation/eval_bidirectional_track_fusion.py` / `system_sota_longquan.py` 的**逐字拷贝**。移植到板端时**以本目录为准**，不要去改 `manu/evaluation/` |

---

## 1. 接口契约

**输入** ① 检测流 `D(f)`：`list[list[dict]]`，索引 = **record 下标**，每 dict 含 `pos (x, y)` / `score` / `track_id` / `infilled`，坐标口径 = **列 x / 行 y**、绝对像素；
② 帧缓存 `FrameSource`：协议 `get(frame_index) -> np.ndarray | None`，按 **record 下标**取**灰度整帧**（640×512 uint8），越界/缺帧返回 `None` ⇒ 该链作废；
③ 标定/参数 `LookaheadConfig`（见 §4）；④ 速度参考 `kf_states`（`dict[int, dict[int, tuple[pos, vel]]]`，管线侧提供，外挂**只读**）。

**输出** ① 上报流 `report(t)`：`list[list[dict]]`，长度 == 帧数，元素 = 检测 dict（`pos` / `score` / `track_id` 沿用源帧），`t < Δ` 的帧**不输出**；
② 计数 `stats`：`applied` / `abstained`（两者互补，合计 == 帧数）/ `fallback_used` / `replay_steps` / `replay_failed` / `replay_max_step_disp` / `output_step_limited` / `output_step_clamped`。

**不变量清单（板端实现必须逐条满足）**

1. **`mode=="off"` ⇒ 逐位透传**：`report[t] == D(t)`，只拷 dict，**不做任何数值运算**。
2. **只读输出流、不回写跟踪器状态**：外挂看不到、也改不了 KF / track 的任何内部量。
3. **极性逐链锁定**：极性在**锚点帧**（检测框中心，误差 ~2 px）判一次，锁死整条链。
4. **xcorr 上报纯位移口径**：`上报位置 = anchor + 模板左上角的位移`（见 §3.1）。
5. **输出限幅只约束输出轨迹本身**，不看运动学参考。
6. **关掉任一新旋钮 ⇒ 逐位等价于历史行为**（默认值全为「关」）。

---

## 2. 板端资源约束（硬约束，来自产物）

| 项 | 数值 | 来源 / 说明 |
|:--|:--|:--|
| **重推步数** | 每输出帧 **1 次重推 × Δ=8 步** | 8 序列 `replay_steps = 204,960` / `applied = 25,620` ⇒ 平均 **8.0 步/帧**；24 序列 `1,011,296` / `126,412` ⇒ 同为 8.0 |
| **单帧读帧数** | 重推区间 `f+1..f+Δ` **全部落在帧缓存窗口内 ⇒ 每帧只读一次** | `SequenceFrameSource` / `CachedFrameSource` 的 LRU 容量 = `max(2, Δ+2)` = **10 帧** ⇒ **板端必须维持 ≥ Δ+2 = 10 帧的灰度环形缓冲** |
| **帧尺寸 / 缓冲内存** | 640 × 512 uint8；10 帧 = **3.28 MB** | `arms_F1_auto10.json → config.img_w/img_h` |
| **xcorr 算力** | `direct` 档每帧**只配准 1 次**：搜索窗 `2r+tsize = 96`、模板 `32×32` ⇒ `matchTemplate` 输出 65×65 ≈ **4.3 M MAC/帧**；**唯一性二次意见**在歧义帧上**再加 1 次**（τ=0.6 时 **1698/24944 = 6.8%** 的匹配）⇒ 平均 **+6.8% MAC** | 板端无压力；**不要开 `chain`（8 次/帧，见 §8）** |
| **快核算力** | 每步 1 次 `52×52` ROI + `104×104` 背景窗，×8 步/帧 | 纯 numpy/OpenCV |
| **失败率** | 8 序列 `replay_failed = 0` / 25,620 帧；24 序列 `replay_failed = 0` / 126,412 帧 | 重推**从不整段失败**；弃权 2,905 / 33,282 帧是**检测器盲区**（`abstained`），不是外挂失败 |
| **单步位移上限** | 8 序列 `replay_max_step_disp = 34.39 px`；24 序列 `64.98 px` | 板端需容忍的链内跳变 |
| **抖动** | 8 序列 `max_jitter = 60.0 px`（stale/original 65.2）· `p99_jitter = 36.6 px`（stale 18.9）；24 序列 `max_jitter = 109.05 px` · `p99_jitter = 59.9 px`（stale 18.7） | **`max_jitter` 已优于老逻辑，但 `p99_jitter` 仍比老逻辑大 ⇒ 板端必须再加一级输出限速 / 低通**（README §8.4） |
| **已知板端缺口** | 24 序列 FAR **0.0349 > 交付红线 0.030**（超线 16.4%，差 0.0049） | 剩余主体是**管线自身**（`015740` 0.4428 / original 0.4277、`002251` 0.1054 / original 0.0997、`003002` 0.0707、`002111` 0.0651）⇒ **属扩展缺口，不为它牺牲 hit@5px** |

---

## 3. 六条移植契约（正文）

### 3.1 契约一 · xcorr 模板匹配位移精修

**是什么**：质心链的位移 = `K(t) − K(f)`（`K` = 快核灰度质心）。质心相对目标**几何中心**有偏置 `b`，随姿态/亮度/拖尾漂移 ⇒ **位移误差 = `b(t) − b(f)`**，尾部很重（位移 p90 **4.37 px** / p95 **8.13 px**）。
改用**同一份模板**（源帧锚点处按底噪阈值二值化的权重图）在目标帧里配准 ⇒ 两端共用同一个偏置 ⇒ **偏置精确抵消**，只剩真实位移（位移 p90 → **3.17 px** / p95 → **6.34 px**；同批 24,503 帧 hit@5px **91.36% → 94.02%**）。

**为什么**：这是「质心偏置的**变化量**」而不是「配准不准」的问题 ⇒ 换搜索策略（半径 / 中心 / 峰值门）都治不了，只有**换度量**能治。

**移植要点**（接口签名代替源码）：

```
make_template(img, xy, tsize=24, bg_size=96, th_k=2.5, core_frac=0.0, polarity=+1)
    -> (weights: float32[t,t], origin: (float, float))   # origin = 模板【左上角】在源帧的绝对坐标
                                                      # 权重恒非负（越大 = 越像目标本体）
match(dst_img, template, anchor, center, radius=48, tsize=24, origin=None, polarity=+1)
    -> (displacement: float64[2], peak: float)
    # 上报位置 = anchor + 模板左上角的位移（纯位移口径）；亚像素 = 抛物线插值（x/y 各一维）
```

**⚠️ 坑 1（今天定位的真 bug，价值最高）—— 半像素约定错误**：`make_template` 从 `round(cx) − tsize//2` 切模板 ⇒ 模板的**离散中心**对应源帧 **`round(cx) − 0.5`**；而旧 `match()` 按 `(tsize−1)/2` 上报位置 ⇒ **每步系统性少 0.5 px（x、y 各）**。
`direct` 只匹配一次 ⇒ 只差 0.5 px（看不出来）；**链式配准 8 步 ⇒ 累计 −4 px** ⇒ `chain` 档位 hit@5px 一度只有 **12.5~14.9%**。
**修法**：`make_template` 返回 `(weights, origin)`；`match()` 上报位置 = **`anchor` + 模板左上角的位移**（与取整约定无关）。
**合成图验证**：平移 `(7, −4)` ⇒ 修复后报 **(6.998, −4.003)**，修复前报 **(6.5, −4.5)**（2026-11-03 夜间本机独立复核：修复后 (6.996, −4.005)，差 ≤0.002 px；回归防线 `manu/tracking/tests/test_xcorr_halfpixel.py`）。

**旋钮与交付值**（默认值见 §4）：`--replay-xcorr-radius` **32**（24~48 平台区，56 起 FAR 同步涨：`WT32R56` F1 94.751 / FAR 0.0210）· `--replay-xcorr-template` **32**（20/24/40/48 都更差：`WT40` 95.016、`WT48` 94.963）· `--replay-xcorr-th-k` **1.5**（1.5~2.0 最优，**2.5 起单调变差**：`WTH25` 95.009 / 84.35%）· `--replay-xcorr-min-peak` 0.25（0.25~0.5 等价：`WMP50` 84.54%；**0.6+ 反而挡掉真匹配**）· `--replay-xcorr-core-frac` **0（关，已被证伪）** · `--replay-xcorr-bg` 96（默认，须 ≥ 模板边长）· `--replay-xcorr-mode` `direct`（`chain` 不入选，见 §8）。

**验收口径**：单次配准位移必须等于**真位移**（合成图 ±0.05 px）；8 步链式累积位移必须 ≈ `8 × step`，**与模板边长奇偶无关**。

### 3.2 契约二 · 目标极性自适应（亮 / 暗）

**是什么**：红外目标相对背景的**极性（亮/暗）会逐帧翻转**。而快核与 xcorr 模板原本都**写死「比背景亮」**：

| 位置 | 亮假设（历史行为） | 暗假设（本条新增） |
|:--|:--|:--|
| 快核 `LocalPatchTracker.step` | 门限 `bg + kσ`；权重 `max(patch − 门限, 0)`；峰取 `argmax`；对比度 `峰 − bg` | 门限 `bg − kσ`；权重 `max(门限 − patch, 0)`；峰取 `argmin`；对比度 `bg − 峰` |
| xcorr 模板 `make_template` | `clip(patch − (bg + kσ), 0)` | `clip((bg − kσ) − patch, 0)` |

**为什么**：暗目标 + 亮假设 ⇒ **权重恒为 0** ⇒ 质心退化成窗心、xcorr 模板为空（只剩噪点）⇒ **配准峰锁到画面里最亮的云边**。
合成图实证：暗目标用亮假设 ⇒ 模板权重和 **24.9**、配准位移 **(−22.932, +20.008)**、相关峰 **0.1069** ⇒ 锁到旁边的亮云；用暗假设 ⇒ 权重和 **3107.1**、位移 **(0.004, 0.005)**、峰值 **0.9872** ✓。实测 `014453` 上旧 replay 偏到 **70~100 px** 外。

**⚠️ 坑 2 —— 归一化相关的符号坑**：`TM_CCOEFF_NORMED` 在真匹配处的**符号取决于目标极性**（暗目标的权重图在目标处为正、目标帧灰度在那里为低 ⇒ 零均值化后两者**反号**）⇒ 暗目标真匹配处的相关值是**负的**；若仍取极大 ⇒ 锁到最亮处。
**修法：暗极性时把模板整体取反再配准**（`corr(−T, I) = −corr(T, I)`，真匹配处即变成正极大）。
板端注意：C++ 侧 `cv::matchTemplate` 的 `TM_CCOEFF_NORMED` 行为与 OpenCV 一致，但**取极值的符号策略必须自己按极性切换**。

**⚠️ 坑 3 —— 极性是「逐帧」属性，不能按序列分类**：旧探针用 `sorted(glob(...))[:60]` 取样 ⇒ 取到的是**文件名字典序**最靠前的 60 个（`__0, __1, __10, __100, __1000, …`）⇒ **几乎全是序列开头的帧** ⇒ 误判「`015000` 100% 暗」。
按**帧号数值顺序**重抽后的真相：

| 序列 | 极性时间线 | 暗帧占比 |
|:--|:--|--:|
| `VIDEO00032_19700101_014453` | **暗 f0 – 8112（整条）** | **100.0%** |
| `VIDEO00033_19700101_015000` | **暗 f0 – 175** → 亮 f175 – 975 | 17.5% |
| `VIDEO00031_19700101_014227` | 亮 f0 – 4058 → **暗 f4058 – 4298** | 10.0% |
| `VIDEO00012_19700101_002251` | 亮 → **暗 f266 – 304** → 亮 f304 – 1482 | 2.5% |
| 其余 20 条 | 全亮 | 0.0% |

⇒ **一条序列内部极性也会翻转** ⇒ **必须逐链判、不能按序列判** ⇒ 极性判定必须放在**锚点帧、逐链锁定**。

**移植要点**（接口签名）：

```
polarity_masses(frame, xy, tracker) -> (bright_mass, dark_mass)   # 面积归一化的越界能量，core = ROI 正中央奇数边长块
    bright_mass = mean(max(core − (bg + kσ), 0))   ;   dark_mass = mean(max((bg − kσ) − core, 0))
estimate_polarity(frame, xy, tracker) -> +1 | −1                     # 相等 ⇒ +1（历史行为）
PolarityTracker(polarity="bright"|"dark"|"auto", dark_ratio=10.0)
    .lock_from(frame, xy) -> int                                    # auto 模式：锚点帧判一次并锁定
    .step(frame, center_xy) -> TrackerResult                        # 亮 ⇒ 调父类；暗 ⇒ 镜像实现
```

**判据**：`dark_mass > ratio × bright_mass` ⇒ 判暗（`ratio` 默认 **10.0**）；`bg` / `σ` 必须用**与快核完全相同**的「大底噪窗 median + MAD」口径（`bg_window = 104`）。
**为什么 ratio = 10**（产物 `runs/lookahead/final/polratio/arms_r{1,3,10,30}.json`，11 条关键序列，2026-11-03 21:32–21:56）：`001810` F1/FAR = r1 **95.18% / 0.0473**（−1.09 pp）· r3 96.05% / 0.0386 · **r10 96.27% / 0.0363（完全回到基线）** · r30 96.27% / 0.0363；`014453` = r1/3/10 **99.75% / 0.0021** · **r30 99.35% / 0.0061（变差）**。
⇒ r=10 把「亮目标上的偶发误判」挡回去（`001810` 误判链 68 → **2**），一条暗链都不牺牲；r=30 开始伤到 `014453` 最弱的 38 条链。

**验收口径**：① 暗目标 + 暗极性 ⇒ 配准位移 ≈ 0 且峰值 > 0.8；② 暗目标 + 亮极性 ⇒ **不得**配到旁边的亮云；③ `PolarityTracker` 在亮假设下与 `LocalPatchTracker` **逐位一致**；④ `dark_ratio` 置信门：亮斑在任何倍率下都判**亮**。

### 3.3 契约三 · 输出级限幅（slew limiter）

**是什么**：`--output-max-step-px`（**默认 0 = 关**，只作用于 **replay 臂**，stale / kinematic 基线不受污染）。
**为什么**：转台看到的是**帧间位移**。重推链的 ROI 偶尔抓住旁边杂波 ⇒ 某帧跳 190 px、下一帧又回来 ⇒ **孤立单帧飞跳** ⇒ 二阶差分 331 px ⇒ 伺服抽搐。限幅**不看运动学参考、只约束输出轨迹本身** ⇒ 不会像「一致性门」那样把好帧误杀成运动学预测（实测一致性门在 `000703` 上 F1 98.0% → 79.4%、`014053` 上 95.1% → 68.8%）。
**规则**（`extrapolator._apply_output_slew_limit`）：参考位置 = 同一 `track_id` **最近一次**的实际上报位置（**跨缺帧保留**：缺帧期间参考位置不更新 ⇒ 重捕获时也按速率限幅追过去）；本帧输出 = 参考位置 + `clip(原始输出 − 参考位置, output_max_step_px)`；`output_max_step_px = 0` ⇒ 函数**直接 return**，既有行为完全可复现。
**移植要点**：限幅器是**输出级**的、有状态的（按 `track_id` 存最后一次上报位置）——**这是全模块唯一有状态的部件**，它存的是**外挂自己的**输出轨迹状态，不违反「不回写跟踪器状态」；但**必须按 track 分桶**，且**缺帧不更新参考位置**。
**验收口径 / 实测**：上一轮配置（`delta_s30`，2026-11-03 上午，memory §18.3/§18.4）**max_jitter 365.9 px → 114.2 px**，F1 只掉 **0.095 pp**（94.3307% → 94.2360%），逐序列 7/8 已到理论界 `2×S = 60 px`、只有 `001324` 残留 114.2 px；**当前交付配置**下 8 序列 `max_jitter = 60.0 px`（stale/original 65.2，已优于老逻辑）、**24 序列 `max_jitter = 122.2 px`**（无 xcorr 版 61.5 px）。

### 3.4 契约四 · 兜底策略

**是什么**：`--replay-fallback`（默认 `kinematic`，**交付值 `holdv`**）。**真实取值集合是六个**：`kinematic`（回退运动学外推）· `stale`（原样搬运 `D(t−Δ)`）· **`abstain`（该帧不输出 ⇒ 让转台知道「这一帧没有坐标」）** · `hold`（用重推途中最后一个有效位置）· **`holdv`（`hold` + 按最后实测的单步位移补齐剩余步数，交付值）** · `holdc`（连续性兜底：上一帧实际上报位置 + 参考速度 ×1，**消融项**，F1 −0.41 pp）。

**为什么**：`kinematic` 本身**不靠谱**——8 序列 kinematic 臂 **F1 77.552% / FAR 0.1776 / hit@5px 33.29%**（**比 stale 的 77.203% 只高 0.35 pp，而 hit@5px 反而低 4.89 pp**；CLE p95 从 113.73 恶化到 **183.05**）；24 序列 kinematic 臂 **F1 75.764% / FAR 0.1697 / hit@5px 38.68%**（**比 stale 的 77.523% 更差**）⇒ **兜底不改善质量**。交付值取 `holdv`（当前路径上 `holdv` 与 `holdc` 逐位相同，`holdc` 是空操作）。

**移植要点**：兜底语义必须与「**该帧是否输出**」解耦（`abstain` = 不输出，不是输出一个假坐标）。板端若伺服要求「每帧必有坐标」，`abstain` 会产生**空洞** ⇒ 需在板端定义空洞策略（本契约不代做决定）。
**验收口径**：`abstained` 与 `applied` 互补且合计 == 帧数；`fallback_used` 记录触发兜底的帧数。

### 3.5 契约五 · 无状态性（构造性保证）

**是什么**：本模块**只读输出流、不回写跟踪器状态**。代码层面的根据：`build_report_stream` 的 `mode=="off"` 分支只做 `[dict(det) for det in source_dets]`——**一个字节的数值运算都没有**。
**为什么**：这是「**关掉开关 ⇒ 与原脚本逐位一致**」这条硬验收项的**唯一**保证机制。任何「顺手改一下跟踪器状态」「顺手加个平滑」「顺手复用一下 KF 速度修正」的改动，都会打破它 ⇒ 成熟跟踪资产零回归无法自证。
**移植要点（不变量）**：① 新旋钮的**默认值必须是「关」**（`0` / `False` / 历史行为），保证 `--lookahead off` 路径**一行都不进**；② 不得引入跨帧的隐式全局状态（唯一例外是 §3.3 的**输出级限幅器**，它只存**外挂自己的**输出轨迹状态且默认关）；③ 不得修改 `manu/evaluation/*`（拷贝纪律 ⇒ 板端也保持「外挂代码与管线代码物理分离」）；④ `track_id` 是**进程级全局计数器**（`PointKalmanTrack._count`，`track_fusion.py:81`）⇒ **比对必须让参考与新实现都跑满同样顺序的同样序列**，否则 `track_id` 必然不同（**位置/分数仍完全相同**）——**这是原脚本的性质，不是本模块引入的**。
**验收口径（两把尺子，见 §5）**：关开关后 **ΔF1 = 0.000000** 且 **md5 逐字节一致**。

### 3.6 契约六 · 相关峰唯一性二次意见（★ 新增，2026-11-04）

**是什么**：xcorr 配准时除了最佳峰，还统计**次峰/最佳峰**的比值 `uniqueness`（次峰取「排除与最佳峰重叠邻域」后的最大峰）。
当 `uniqueness > τ` 时判该次配准**歧义**，此时**以锚点为搜索中心重配一次**，**只在同时满足三条时采纳重配结果**：
① 重配**唯一**（`uniqueness2 ≤ τ`）；② 重配**比原配更自信**（`peak2 > peak`）；③ 重配峰值仍达标（`min_peak`）。
否则**保持原结果不动**。

**为什么需要它**：walk-off（链锁到错误特征后一路跑飞）的帧，其配准峰**次峰同样很高**（FP 帧 uniqueness p50 = **0.593**，p75 0.814）；
好帧 uniqueness p50 = **0.000**（p75 0.203）⇒ 这是目前**唯一**能把坏帧与好帧分开的**无状态**信号
（一致性门 / 低速冻结 / 位移自适应融合三个家族实测全部净负，见 §8）。

**为什么是「二次意见」而不是「回退锚点」**：
- 「一看到歧义就回退锚点」实测**灾难性**（8 序列 τ=0.3：F1 95.062 → **87.03**、hit@5px 84.96 → **67.20**、FAR 0.0182 → **0.0913**）
  ——锚点本身比 8 帧前的真实位置更差。
- 「二次意见」是**单调安全**的：只在「更唯一 **且** 更自信」时才替换旧结果 ⇒ **最坏情况是「什么都没改」** ⇒ 该机制**不可能**让指标变差。
- 条件 ②（`peak2 > peak`）**不能省**：省掉时高速目标会被「更靠近锚点但错误」的结构骗走
  （24 序列 τ=0.5：002438 FP **+69**、000322 +42、001415 +27）；加严后收窄到 002438 **+18**。

**移植要点（不变量）**：
1. **默认 τ = 0.0（关）** ⇒ 关开关路径**一行都不进**（满足 §3.5 构造性保证）。
2. 次峰必须**排除与最佳峰重叠的邻域**再取，否则最优峰自身的旁瓣会被当成次峰 ⇒ uniqueness 恒高、门形同虚设。
3. 重配的搜索中心是**锚点**（已知可靠的位置），半径**沿用** `--replay-xcorr-radius`，模板**沿用**同一份（不要重切）。
4. 采纳判定**三条缺一不可**（唯一 / 更自信 / 峰值达标）；**不要**为了「多救几帧」放宽 ②。
5. 统计计数器：`xcorr_ambiguous` / `xcorr_uniq_n` / `xcorr_uniq_sum` / `xcorr_uniq2_sum` / `xcorr_rematch_applied` / `xcorr_rematch_rejected`，
   落在 `arms.<臂>.stream_stats.overall` ⇒ 板端移植后应能复现同一组计数。

**板端算力代价**：τ=0.6 时 **1698/24944 = 6.8%** 的匹配要多做**一次** `matchTemplate`
（每次 65×65 输出 ≈ 4.3 M MAC）⇒ 平均 **+6.8% MAC**，**不是 8 次/帧**（那是 `chain` 档位，已否决，见 §8）。

**验收口径**：τ=0 时与上一档交付产物 **9/9 文件 md5 逐字节一致**（`runs/lookahead/final/uniq/preds_u0` vs `pol8f/preds_F1_auto10`）⇒ 证明关闭时零副作用。


### 3.8 契约八 · 「源帧无检测 ⇒ 再往回退重推」（可选件，**交付档未启用**）★ 新增，2026-11-04 夜间第 3 轮

**是什么**：正常路径下源帧是 `t−Δ`。当**源帧检测器漏检**（该帧没有任何输出）时，
外挂**没有锚点** ⇒ 该帧必然不上报 ⇒ 必然脱靶。
本契约让外挂**再往回退一点**：退到**最近一次有检测**的帧当锚点，多走几步把链重推回来。

**为什么它是「可选件」而不是交付档**：24 序列 FAR 已**饱和**（交付档 0.0298，红线 0.0300，余量只剩 0.0002），
回退链会在「目标早已飞走」的帧上产出**幽灵坐标** ⇒ FAR 破线。
实测：8 序 `rb8 + mp0.50` 是净收益（F1 +0.32 pp、hit@5px +0.42 pp、FAR 仍只有 0.0171），
但 24 序所有回退档 FAR 都 > 0.030 ⇒ **交付档不启用**。

**两个旋钮（默认关 ⇒ 逐位不变）**：

| 参数 | 默认 | 交付取值 | 含义 |
|:--|--:|--:|:--|
| `--replay-rollback-max-frames` | 0（关） | **0（未启用）** | 源帧无检测时，最多往回退多少帧找锚点 |
| `--replay-rollback-min-peak` | 0.0（关） | **0.0（未启用）** | 回退链的相关峰值置信门：峰值低于此值该帧不上报 |

**移植要点（不变量）**：
1. 默认两个旋钮都是 0 ⇒ 关开关路径**一行都不进**（满足 §3.5 构造性保证）。
2. 回退后**重推步数变多** ⇒ 运动学参考位移（`anchor + Δ·v`）必须按**真实步数**算，
   否则 hold / 一致性门会拿 Δ 帧的参考去比一条更长的链（板端用 `dataclasses.replace(cfg, delta=真实步数)` 实现）。
3. 回退链的峰值置信门**只对回退链生效**，不影响正常链（判据是「这条链是不是回退出来的」，
   不是「全局阈值」）。
4. 板端算力代价：**零新增图像运算**（回退只改「用哪一帧当锚点」；峰值读的是已经在算的 `xcorr_peak_last`）。

**验收口径**：两个旋钮为 0 时与**交付档冠军产物**逐位一致 ——
8 序 **IDENTICAL=9 / DIFFER=0**、24 序 **IDENTICAL=25 / DIFFER=0**（`per_frame` + `metrics`）。
（复核脚本：`bash manu/tracking/lookahead/check_newknobs_off.sh [8|24]`）

**风险条目**：**R18 / R19** 见 §7。

---

## 4. CLI 参数真值表（2026-11-03 夜间 `grep` 核实）

> 真值来源：`manu/tracking/lookahead/lookahead_longquan.py` 的 `add_argument` 字符串 + `extrapolator.LookaheadConfig` 字段默认值，**两者已逐条对齐**。
> **所有默认值都是「关」或历史行为** ⇒ 满足 §3.5 的构造性保证。

| 参数 | 默认值 | 交付值 | | 参数 | 默认值 | 交付值 |
|:--|:--|:--|:--|:--|:--|:--|
| `--lookahead` | `off` | `replay` | | `--replay-roi-mode` | `chain` | **`selfpred`** |
| `--delta` | `8` | 8 | | `--replay-roi-selfpred-window` | `3` | 3 |
| `--velocity-source` | `kf` | `kf` | | `--replay-roi-selfpred-max-px` | `0.0` | 4.0 |
| `--velocity-window` | `3` | 3 | | `--replay-xcorr-radius` | **`0.0`（关）** | **32** |
| `--position-source` | `output` | `output` | | `--replay-xcorr-template` | `24` | **32** |
| `--frames-root` | `/home/manu/mnt/.../frames_ir_jpg` | 服务器侧 `/mnt/data/siping/.../frames_ir_jpg` | | `--replay-xcorr-th-k` | `2.5` | **1.5** |
| `--frames-cache` | `""` | — | | `--replay-xcorr-min-peak` | `0.25` | 0.25 |
| `--replay-require-valid` | `True` | `True` | | `--replay-xcorr-bg` | `96` | 96 |
| `--replay-fallback` | `kinematic` | **`holdv`** | | `--replay-xcorr-core-frac` | **`0.0`（关）** | 0.0 |
| `--replay-max-hold` | `0` | 8 | | `--replay-xcorr-mode` | `direct` | `direct` |
| `--replay-step-gate-px` | `0.0` | 0.0 | | `--replay-xcorr-chain-radius` | `0.0` | 0.0 |
| `--replay-anchor` | `kernel` | **`delta`** | | `--replay-polarity` | **`bright`（历史行为）** | **`auto`** |
| `--replay-anchor-check-px` | `0.0` | 0.0 | | `--replay-polarity-ratio` | `10.0` | 10.0 |
| `--replay-pred-radius-px` | `12.0` | 12.0 | | `--output-max-step-px` | **`0.0`（关）** | **30** |
| `--replay-mode` | `perframe` | `perframe` | | `--replay-consistency-gate-px` | `0.0` | 0.0 |
| `--replay-step-rel-tol-px` | `0.0` | 0.0 | | `--replay-outlier-tol-px` | `0.0` | 0.0 |
| `--replay-hold-window` | `3` | 3 | | `--replay-hold-outlier-tol-px` | `10.0` | 10.0 |
| `--replay-hold-robust` | `False` | False | | `--replay-hold-use-ref` | `True` | **False**（`--no-…`） |
| `--replay-obs-shrink-alpha` | `1.0`（关） | 1.0 | | `--replay-disp-shrink-alpha` | `1.0`（关） | 1.0 |
| `--replay-disp-merge-sigma-px` | `0.0`（关） | 0.0 | | `--replay-freeze-speed-px` | `0.0`（关） | 0.0 |
| `--fast-mode` | `centroid` | `centroid` | | `--fast-win-size` | `32` | **52** |
| `--fast-bg-window` | `64` | **104** | | `--fast-th-k` | `3.0` | **2.5** |
| `--fast-min-energy` | `1.0` | **50** | | `--fast-min-contrast` | `5.0` | 5.0 |
| `--fast-peak-window` | `9` | 9 | | `--fast-kf-r` | `2.0` | 2.0 |
| `--replay-xcorr-uniq-gate` | **`0.0`（关）** | **0.6** |
| `--replay-abstain-peak-below` | **`0.0`（关）** | **0.30** | | `--replay-abstain-disp-px` | **`0.0`（关）** | **24** |
| `--replay-rollback-max-frames` | **`0`（关）** | **0（未启用）** | | `--replay-rollback-min-peak` | **`0.0`（关）** | **0.0（未启用）** |

> ⚠️ `--replay-fallback` 的**真实取值集合**是 `kinematic | stale | abstain | hold | holdv | holdc`（六个，不是四个）。

---

## 5. 验收口径（两把尺子）

### 尺子 1 · 历史帧等价性（证明「没破坏成熟跟踪资产」）

物理帧 `t` 时系统手上最新的估计是 `x̂_{t−Δ}`，拿它对 `GT_{t−Δ}` 评测，要求与老版 Sys SOTA 在同一数据上**绝对一致**。
实现：在共同源帧窗口 `[0, N−Δ)` 上对每臂的**源帧估计**（= `D(f)` 本身）评测 ⇒ 各臂天然逐位相等；脚本做成**硬门禁**（不等就 `SystemExit`）。

| 口径 | 实测（2026-11-03 夜间本机独立复核） |
|:--|:--|
| 各臂源帧口径 ΔF1 | **0.000000**（8 序列四臂同值 `0.964127`；24 序列同值 `0.947527`） |
| Gate A · 8 序列 md5 | `runs/lookahead/ref/preds_original_v3/` vs `runs/lookahead/final/preds_gateA/` ⇒ **9/9 IDENTICAL / 0 DIFFER**（F1 均 **0.964002**） |
| Gate A · 24 序列 md5 | `runs/lookahead/final/ref_preds_original_v3_24seq/` vs `runs/lookahead/final/preds_off_24seq/` ⇒ **25/25 IDENTICAL / 0 DIFFER**（F1 均 **0.947469**） |
| 单测 | `PYTHONPATH=. python -m unittest discover -s manu/tracking/tests -t .` ⇒ **149 项（10 skipped），OK**（2026-11-03 夜间本机复核） |

### 尺子 2 · 当前帧伺服有效性（证明外推收益）

| 指标 | 口径 |
|:--|:--|
| F1 / Recall / Precision / FAR | 冻结口径 `count_matches(..., "inbox")`（贪心按分数降序、点落框内、一个 GT 只吸收一个检测） |
| CLE p50/p90/p95/p99/max | 每个「有 GT 且有上报」的 record，取**离最近 GT 中心最近的那个上报点**算欧氏距离；**跨序列取最差值**（`percentile_aggregation = "worst_over_sequences"`）。**另有池化口径** `cle_pooled_*`（`pooled_over_all_frames`）—— 引用时必须连口径一起写 |
| hit@5px | 上报点到最近 GT 中心 ≤ 5 px 的 GT 帧占比（分母 = `gt_frames`，**物理命中率**） |
| max_jitter / p99_jitter | 同一 track 上报序列二阶差分 `abs((p_t − p_{t−1}) − (p_{t−1} − p_{t−2}))` 的 max / p99（按 track 分组） |
| coverage | 有上报的帧 / 总帧 |

**产物 JSON 路径**：`arms.<臂>.source_frame.overall.f1` / `.delta_f1_vs_original` / `.current_frame.overall.{f1,recall,precision,far_per_frame}` / `.servo.overall.{cle_p50,cle_p95,hit_rate_5px,max_jitter,p99_jitter,coverage}` / `.stream_stats.overall.*`。

---

## 6. 实测数字（全部来自产物 JSON，2026-11-03 夜间本机复核）

### 6.1 · 8 序列（官方 gate 集）—— 产物 `runs/lookahead/final/final8/arms_delivery.json`（耗时 119.6 s；上一档 `pol8f/arms_F1_auto10.json`）

| 臂 | F1 | FP | FAR | hit@5px | CLE p50 | CLE p95 | max_jitter | p99_jitter |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `original`（零延迟天花板） | **96.400%** | 182 | 0.0063 | 88.13% | 1.97 | 13.43 | 65.2 | 19.0 |
| `stale`（老逻辑，交付基线） | 77.203% | 5241 | 0.1807 | 38.18% | 20.43 | 113.73 | 65.2 | 18.9 |
| `replay` 亮假设（旧交付 `WTH15`） | 95.077% | 523 | 0.0180 | 84.67% | 2.12 | 30.01 | 60.0 | 41.3 |
| **`replay` auto + ratio10（上一档）** | 95.062% | 527 | 0.0182 | 84.96% | 2.12 | 30.15 | 60.0 | 40.4 |
| **`replay` auto + 唯一性二次意见 τ=0.6（★新交付）** | **95.266%** | **473** | **0.0163** | **85.17%** | **2.11** | **27.24** | 60.0 | **36.6** |

交付臂**池化** CLE：p50 **1.675** / p95 **10.443**（`cle_pooled_*`）。7 条亮目标序列**逐位不变**；`015000`（暗段）**hit@5px 92.20% → 96.39%（+4.19 pp）**，F1 98.54% → 97.76%（−0.78 pp）、FAR 0.0140 → 0.0210。

### 6.2 · 24 序列（官方口径）—— 产物 `runs/lookahead/final/final24/arms_delivery.json`（耗时 697.0 s；上一档 `pol24f/arms_F1_auto10.json`）

| 指标 | 亮假设（旧交付）`pol24/arms_R0_bright.json` | **auto + ratio10（★交付）** | Δ |
|:--|--:|--:|--:|
| F1 | 90.461% | **92.924%** | **+2.46 pp** |
| FP | 8963 | **5889** | **−3074** |
| FAR | 0.0542 | **0.0349** | **−35.6%**（红线 0.030 ⇒ 超线 80.8% → **16.4%**） |
| hit@5px | 77.51% | **81.64%** | **+4.13 pp** |
| 池化 CLE p50 / p95 | 1.882 / 28.476 | **1.728 / 15.160** | −8.2% / **−46.8%** |
| max_jitter / p99_jitter | 122.2 / 60.0 | **109.05** / 59.9 | **−13.15 px** |
| 交付线 | F1 ≥ 0.88 · Recall ≥ 0.85 · **FAR ≤ 0.030** | FAR **未达标**（超线 16.4%，差 0.0049） | — |

逐序列亮 → auto：`014453` 63.49% → **99.75%**（FAR 0.3608 → **0.0021**）、`014227` 95.33% → **98.73%**、14 条**逐位不变**、5 条轻微变差（合计 +55 FP）。

### 6.3 · `014453` 单序列 —— 产物 `runs/lookahead/final/pol14453/arms_Q{0,1,2,3}*.json`（21:00）

| 变体 | F1 | FP | FAR | hit@5px | CLE p50 / p95 |
|:--|--:|--:|--:|--:|--:|
| `original`（天花板） | 99.93% | 3 | 0.0004 | 99.02% | 1.19 / 3.13 |
| `replay` 亮假设（旧交付） | **63.49%** | **3035** | **0.3608** | **36.04%** | 7.92 / 105.71 |
| `replay` **auto** | **99.75%** | 18 | **0.0021** | **98.51%** | 1.18 / 3.13 |
| `replay` **dark**（强制） | 99.75% | 18 | 0.0021 | 98.51% | 1.18 / 3.13 |
| `replay` auto + **关 xcorr** | 99.83% | 11 | 0.0013 | 98.80% | 1.21 / 3.21 |

**OSD 窗口级证据**（`runs/lookahead/osd/windows/VIDEO00032_19700101_014453_polarity.json`，窗口 **6638–6937**、300 帧）：`stale` 3.65 px / 75.0% · `replay_old`（亮假设）**78.05 px / 9.0%** · **`replay_new`（auto）1.21 px / 100.0%** · `original` 1.28 px / 100.0%。
同批：`015000` 暗段 4.65 px / 59.2% → **1.52 px / 100.0%**；`001324` 4.48 / 65.3% → 3.31 / 79.0%（旧）→ **2.27 / 93.3%**（新），天花板 1.74 / 94.7%；`014227` 3.63 / 82.7% → 1.65 / 100.0%。

### 6.4 · 机理闭环（为什么修极性能同时压 FAR、抬 hit@5px）

`014453` 流式统计（亮 → auto）：`held_steps` **64692 → 45**；`replay_max_step_disp` **35.62 → 13.75 px**。
⇒ 亮假设下快核**看不见暗目标**（`min_energy` 门永不过 ⇒ 整条链靠 hold 盲推）⇒ 漂到云边 ⇒ 定位错 ⇒ FP 暴涨；暗假设下**快核重新看得见目标** ⇒ **同一套机械结构（快核 + xcorr + 退回重推）不变，只是把「看不见」变成「看得见」**。

---

## 7. 移植风险清单（逐条会踩的坑）

| # | 坑 | 症状 | 规避 |
|:--|:--|:--|:--|
| R1 | **半像素约定**（§3.1 坑 1） | 模板离散中心 ≠ anchor 物理点 ⇒ 每步少 0.5 px、链式 8 步累计 **−4 px** | `make_template` 返回 `origin`；上报 = `anchor + 模板左上角位移`；板端单测必须覆盖**奇/偶 tsize** 两种情况 |
| R2 | **归一化相关的符号**（§3.2 坑 2） | 暗目标真匹配处相关值为**负** ⇒ 取极大锁到最亮云边 | 暗极性时**模板整体取反**；板端必须按极性切换极值策略 |
| R3 | **极性逐帧翻转**（§3.2 坑 3） | 按序列分类 ⇒ 一条序列内翻转时全错 | **逐链判、逐链锁定**；判据用 ROI 中央块的亮/暗向越界能量，`dark_mass > ratio × bright_mass`（ratio = 10） |
| R4 | **抽样偏差**（§3.2 坑 3） | `sorted(glob(...))[:60]` 取到字典序最靠前 ⇒ 误判「`015000` 100% 暗」 | 任何按序列取样的统计**必须按帧号数值顺序** |
| R5 | **`track_id` 是进程级全局计数器** | 拿子集去比 8 序列产物 ⇒ `track_id` 必然不同（误报回归） | 比对时**两边跑满同样顺序的同样序列**；位置/分数仍应完全相同 |
| R6 | **关开关必须逐位一致** | 顺手加平滑 / 顺手用 KF 速度修正 ⇒ 破坏构造性保证 | 新旋钮默认全「关」；`off` 分支只拷 dict；Gate A md5 进 CI |
| R7 | **OpenCV `matchTemplate` 归一化相关的符号** | 与 R2 同源，但板端 C++ 侧要自己实现极性切换 | 板端单测：暗目标必须配到 ≈0 位移且峰值 > 0.8 |
| R8 | **`min_peak` 取舍** | 太松 ⇒ 噪声被当目标（**已被证伪**：真匹配 peak p50 = 0.906、p05 = 0.352，低于 0.25 只占 4.10%）；太紧 ⇒ **挡掉真匹配**（0.6/0.75 让 `014453` FAR 0.0836 / 0.2593） | 保持 **0.25**（0.25~0.5 等价）；**不要**收紧 |
| R9 | **`th_k` 取舍** | 模板二值化底噪倍数：**1.5~2.0 最优**，2.5 起单调变差（`WTH25` F1 95.009 / hit 84.35%） | 交付用 **1.5**；**不要**调到 ≥ 2.5 |
| R10 | **输出级限幅跨缺帧语义** | 缺帧期间若更新参考位置 ⇒ 跨缺口重捕获原样放行 ⇒ 二阶差分顶到 100 px 以上 | 参考位置**跨缺帧保留不更新** |
| R11 | **`abstain` 产生空洞** | 兜底走 `abstain` ⇒ 该帧无坐标 | 板端必须定义空洞策略（本契约不代做决定） |
| R12 | **p99_jitter 仍劣于老逻辑** | 8 序列 36.6 px vs stale 18.9 px；24 序列 59.9 px vs 18.7 px | **板端加一级输出限速 / 低通**（README §8.4） |
| R13 | **24 序列 FAR 仍超线** | 0.0349 > 0.030（差 0.0049，超线 16.4%） | 剩余主体是**管线自身**问题 ⇒ 属扩展缺口，**不要**为它牺牲 hit@5px、**不要**新增旋钮 |
| R15 | **唯一性次峰必须排除最优峰邻域** | 不排除 ⇒ 最优峰旁瓣被当次峰 ⇒ uniqueness 恒高、门形同虚设 | 次峰在「排除与最佳峰重叠邻域」后再取；板端必须覆盖「两个等高峰」的合成用例 |
| R14 | **拷贝纪律** | 改 `manu/evaluation/*` ⇒ 与原脚本不可比 | 只改 `manu/tracking/lookahead/`；`manu/evaluation/system_sota_longquan.py` md5 `3361e7b70e76d3a92346fd8fd9879dad` 保持不变 |
| R16 | **弃权导致召回下降**（转台长时间收不到坐标） | 阈值太松会把好帧也弃掉 | 交付阈值由 8/24 序列双口径扫描确定（peak < 0.30 且位移 > 24 px）；单测锁住「低峰 + 静止链不弃权」 |
| R17 | **目标持续机动**时链位移一直很大 ⇒ 弃权条件可能长期成立 | 高速机动序列（`001324`）实测低峰帧链位移 p50 = 0 px，未触发；极端机动需复核 | 实跑 hit@5px 未掉（见 §6）；板端把该旋钮做成**可调**并保留关闭位 |
| R18 | **回退链产出「幽灵坐标」**（目标早已飞走，模板配准锁到背景结构） ⇒ FAR 破线 | 24 序列 FAR 红线 0.030 已饱和（交付档 0.0298），任何回退档都超线 | 回退 + 峰值置信门作为**默认关**的可选件；交付档不启用；板端保留关闭位 |
| R19 | **回退后重推步数变多**，运动学参考位移若仍按 Δ 算会低估外推量 | hold / 一致性门拿 Δ 帧参考去比更长的链 ⇒ 兜底位置偏 | 板端必须按**真实步数**重算参考位移（`replace(cfg, delta=真实步数)`）

---

## 8. 已实测否决、**不要再试**的旋钮家族（负结果资产）

> **本轮（2026-11-04）新增否决**：**唯一性门的「回退锚点 / 运动学 / 弃权」形态** —— 只保留「二次意见」形态。
> 实测（8 序列 `runs/lookahead/final/uniq/`）：τ=0.3 时 F1 95.062 → **87.03**、hit@5px 84.96 → **67.20**、FAR 0.0182 → **0.0913**；τ=0.85 才回升到 F1 95.03 ⇒ **全档净负**。

> 来源：memory 专题档 §23.11 / §24.6 · README §5.5 / §5.9 · 产物 `runs/lookahead/final/{far24,far24b,mp3,ack,cand8,s14453}/`。**板端移植时不要重跑这些路线。**

| 家族 | 实测判决（关键数字） |
|:--|:--|
| **`chain` 档位**（逐步配准） | 修复半像素 bug 后最佳 **83.91%** < direct **84.58%**；`WC_R12` 的 CLE p95 = **49.87 px** vs direct **30.01 px**（错位刷新模板 ⇒ 级联跑飞）；`min_peak` 越高越差（0.25→83.28 / 0.60→82.78 / 0.75→80.91） |
| **`chain_direct`** / **rolling 链** | rolling 链数值发散：`CLE p95 = 3.87e+10 px`、`max_jitter = 6.94e+13 px`（F1 72.491%） |
| **模板 `core_frac` 核心化** | `001324` 单序列 `CF20` 53.67 / `CF35` 60.67 / `CF50` 63.40 **< 同旋钮基线 `arms_WT32R32.json` 的 001324 = 64.51%** ⇒ 拖尾假说不成立，**证伪** |
| **`th_k ≥ 2.5`** | `WTH25` F1 95.009 / hit@5px 84.35%，单调劣于 1.5 |
| **换运动学搜索中心** | 理想单次 xcorr 以运动学预测为中心 **73.20%** vs 以链位置为中心 **78.08%**（−4.88 pp）；CLE p95 34.53 → **100.98 px**；两段式 **78.09%**（+0.01 pp，无收益） |
| **大半径重试**（r64/96/128） | 78.19 / 78.08 / 77.74% **全部 ≤ r32**；CLE p95 34.53 → 51.42 → 77.92 → **100.95 px** |
| **纯峰值门**（`min_peak` 提高） | 001324 上 τ=0.4 → 78.90%（+0.82 pp），但 8 序列 `min_peak 0.5` → **84.54%** < 0.25 → **84.58%** ⇒ 否决 |
| **观测层收缩**（`replay_obs_shrink_alpha`） | 会把链内 ROI 中心也拽向预测 ⇒ 机动目标跟不上（`001324` F1 **−10.8 pp**） |
| **位移层收缩**（`replay_disp_shrink_alpha`） | 实测净亏 |
| **低速冻结**（`replay_freeze_speed_px`） | 3 序列压力集 FAR 仍 0.284；8 序列上恶化（`replay_freeze_speed_px 2 + diff1` ⇒ FAR **0.0389**、hit@5px 掉 2.97 pp） |
| **位移自适应融合**（`replay_disp_merge_sigma_px`） | 实测净亏 |
| **`peak_centroid`**（`--fast-mode`） | 实测更差 |
| **`refine` 档位**（`--replay-mode refine`） | 不入选 |
| **一致性门单独用**（`replay_consistency_gate_px`） | 门生效（rej 422/164 帧）但 `000703` F1 98.0% → **79.4%**、`014053` 95.1% → **68.8%** ⇒ 运动学参考本身不可靠（用坏尺子量好坏） |
| **`--replay-anchor pred`** | `014453` F1 **66.03%** / FAR **0.3358** ⇒ 否决 |
| **`--fast-min-contrast`** | 实测无增益 ⇒ 否决 |
| **降低 `replay_xcorr_min_peak`**（0.02~0.20） | 真匹配峰值普遍很高（p50 0.906），放松只是放噪声进来 ⇒ 否决 |
| **降低 `replay_xcorr_th_k`**（0.75 / 1.0） | 模板变噪、峰不稳 ⇒ 否决 |
| **`--replay-step-rel-tol-px`**（首步离群门） | 门真正生效后 F1 **94.32% → 92.58%**（净亏 1.74 pp），p95 35.7 → 63.4 ⇒ 默认关 |
| **`--replay-outlier-tol-px` / `--replay-anchor-check-px` / `--replay-step-gate-px`** | 实测净亏或无实质增益 ⇒ 默认关，保留为消融开关 |

> ⚠️ **批次事故留档**（memory §24.6 C）：`mp3` 批 `M0..M8` 9 个变体**全部漏写参数** ⇒ 9 个产物**逐位相同** ⇒ 该批作废。
> **教训：变体批次必须回源核对「产物 ≠ 冠军」，否则会把「参数没生效」当成「参数无效」。**

---

## 9. 复现命令（服务器 `test-Super-Server`，2026-11-03 夜间）

```bash
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/huangzhe/anaconda3/envs/uav/bin/python
C8=runs/optuna_mix_ratio/trials/trial_0080/gate/epoch3/ms_cache
C24=runs/optuna_mix_ratio/trials/trial_0080/lq_eval_fpv8/ms_cache
GT="/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010"
FRAMES="/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg"
PARAMS=manu/evaluation/system_sota_longquan_trial0080_delivery_v3_params.json

# ★ 一键复现交付档（8 + 24 序列）
bash manu/tracking/lookahead/run_delivery.sh both

# 单测（149 项，10 skipped）
$PY -m unittest discover -s manu/tracking/tests -t . -v

# Gate A：关开关 ⇒ 与原脚本逐字节一致
for f in runs/lookahead/ref/preds_original_v3/*.json; do b=$(basename $f); \
  [ "$(md5sum $f|cut -d' ' -f1)" = "$(md5sum runs/lookahead/final/preds_gateA/$b|cut -d' ' -f1)" ] \
  && echo "IDENTICAL $b" || echo "DIFFER $b"; done
```

**交付档参数原文**（`run_delivery.sh` 内的 `$CFG`，一字不改）：

```
--replay-mode perframe --replay-max-hold 8 --replay-fallback holdv --no-replay-hold-use-ref
--output-max-step-px 30 --replay-anchor delta --fast-win-size 52 --fast-bg-window 104
--fast-th-k 2.5 --fast-min-energy 50 --replay-roi-mode selfpred --replay-roi-selfpred-window 3
--replay-roi-selfpred-max-px 4 --replay-xcorr-radius 32 --replay-xcorr-template 32
--replay-xcorr-th-k 1.5 --replay-polarity auto --replay-xcorr-uniq-gate 0.6
--replay-abstain-peak-below 0.30 --replay-abstain-disp-px 24
```

**产物路径**：`runs/lookahead/final/pol8f/arms_F1_auto10.json` + `preds_F1_auto10/`（8 序列）· `runs/lookahead/final/pol24f/arms_F1_auto10.json` + `preds_F1_auto10/`（24 序列）· `polratio/arms_r{1,3,10,30}.json`（置信倍数扫描）· `pol14453/arms_Q{0,1,2,3}*.json`（`014453` 单序列）· `poldiag/arms_diag.json`（逐链诊断）· `runs/lookahead/osd/videos/<SEQ>_pol_{stale,replay_old,replay_new,original}.mp4`（16 段）。

---

## 10. 板端移植 TODO（源自仓库，非本文新增）

1. **`p99_jitter` 仍比老逻辑大**（8 序列 36.6 vs 18.9 px；24 序列 59.9 vs 18.7 px）⇒ **板端加一级输出限速 / 低通**。
2. **24 序列 FAR 0.0349 vs 红线 0.030**（差 0.0049，超线 16.4%）⇒ 剩余主体是**管线自身**问题，属扩展缺口。
3. **极性方向的两个小补丁**（memory §25.12）：① 暗/亮**切换**时沿用上一链极性（时间平滑）⇒ 修 `015000` F1 −0.78 pp；② `min_energy` / `min_contrast` **按极性分别设阈值**。
4. **降检测延迟**：Δ 8→4 可让老逻辑 F1 77.20% → 89.81%、档位 B 95.08% → 95.85%（降延迟与加外挂是**叠加收益**）。
5. **001324 重捕获**（交付臂 `|err| > 15 px` 帧占 9.65%，离群帧误差方向与真位移**反向**，余弦 −0.905）⇒ 需要**离群检测器**（与 KF 速度方向一致性门 / 路径平滑门 / 相关峰唯一性比），**不是更好的匹配** ⇒ 预期 +2.3 pp。


### 3.7 契约七 · 「不自信就不上报」（弃权）★ 新增，2026-11-04 夜间第 2 轮

**是什么**：重推链每次上报坐标前，先看**这次配准有多自信**。判据是两个**已经在算**的量：
模板相关峰 `peak`、以及**链位移**（快核链从源帧锚点跑出去多远 `|上报位置 − 源帧锚点|`）。
当 **`peak < replay_abstain_peak_below` 且 `链位移 > replay_abstain_disp_px`** ⇒ **本帧该 track 不上报**，
转台保持上一帧指令（等价于「这一帧我没把握，别动」）。

**为什么需要它**：FAR（误报率）是交付红线（≤ 0.030），上一档 24 序列 FAR 0.0349 超线。
但逐帧归属实测（`diag_fp_attrib.py`）：5764 个误报里，只有 1658 个是「重推跟丢」造成的
（1367 个是检测器自身噪声、2739 个是**该帧 GT 显式为空**的标注稀疏）。
把这 1658 个从「报一个错坐标」改成「**不报**」，FAR 就回到红线内。

**为什么峰值低就等于「位置是错的」—— 为什么必须再看链位移**：
模板峰值低有两种成因，**必须区分**：
- (a) 目标**机动太大** ⇒ 模板跟不上，但快核跟得住 ⇒ **位置其实是对的**；
- (b) 快核**跟丢** ⇒ **位置是错的**。
用**链位移**区分：跟丢时链从锚点跑很远（`001415` 低峰帧 p50 = **80 px**），
机动时链几乎没跑动（`001324` 低峰帧 p50 = **0 px**）⇒ **低峰 + 大位移 = 跟丢**。

**移植要点（不变量）**：
1. **默认两个旋钮都是 0.0（关）** ⇒ 关开关路径**一行都不进**（满足 §3.5 构造性保证）。
2. **两个条件是「与」**，不是「或」：只看峰值会误伤高速机动帧（实测 `peak<0.2` 单门会让 `001324` 命中率崩塌）；
   只看位移会把「目标真的在快速移动」误判成跟丢。
3. **判据必须读重推链已经在算的量**，**不要**新增图像运算（本契约的板端代价 = **零**）。
4. **弃权 ⇒ 该帧无坐标**（与兜底策略 `abstain` 同语义）⇒ 板端的空洞策略（见 R11）同样适用。
5. 统计计数器 `replay_xcorr_abstained` 落在 `arms.<臂>.stream_stats.overall` ⇒ 板端移植后应能复现同一组计数。

**板端算力代价**：**零**。峰值与链位移都是重推链**已经在算**的量，判据只是一次比较 + 一次 `continue`。

**验收口径**：两个旋钮为 0 时与**上一档冠军产物**逐字节一致 ——
8 序 **IDENTICAL=9 / DIFFER=0**（`runs/lookahead/final/noguard8/preds_delivery` vs `uniqrematch2/preds_u060`）、
24 序 **IDENTICAL=25 / DIFFER=0**（`noguard24/preds_delivery` vs `uniq24b/preds_u060`）⇒ 证明关闭时零副作用。
（复核脚本：`bash manu/tracking/lookahead/check_off_bitidentical.sh`）

**风险条目**：**R16 / R17** 见 §7。

**新 CLI（默认关 ⇒ 历史行为逐位不变）**：

| 参数 | 默认 | 交付取值 | 含义 |
|:--|--:|--:|:--|
| `--replay-abstain-peak-below` | 0.0（关） | 0.30 | 模板相关峰低于此值 ⇒ 进入弃权判定 |
| `--replay-abstain-disp-px` | 0.0 | 24 | 配合上一项：链位移（px）超过此值才算「跟丢」；0 = 不看位移 |
