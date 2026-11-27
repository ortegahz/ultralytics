# deep 深化退化轴 · 物理前提验证（合成图，独立验证）

> 验证人：独立验证员　|　日期：2026-11-27　|　对象：`iqa_v2` deep 档位（blur σ≤8 / motion L≤15 / contrast α≥0.05）
> 方法：**零真实数据集**。全部指标与退化用**仓库现成实现**，只读不改。
> 复现脚本：`/tmp/deep_sanity/{synth,verify,analyze,diagnostic}.py`（临时目录，不入库）

---

## 0. 一句话结论

**deep 档位的物理前提「部分成立」**：`blur` 与 `contrast` 两族确实把目标特征物理摧毁
（最重档 `target_lap_var` 只剩 clean 的 **0.43%~9.49%**），但

- **`motion` 族在最重档 L=15 仍塌不下去**（小/中目标剩 **20.7%~36.0%**，超过 20% 红线）⇒ P2 FAIL；
- **`target_lap_var` 的逐档单调性有 3 处反升**（+1.51%~+2.38%）⇒ P1 FAIL；
- **deep 新增的两个 contrast 档位（α=0.10 / 0.05）把 `local_scr` 的单调性打破**（小目标反而升高），
  机理已定位为 `apply_degradation` 的 **uint8 量化**，不是 SCR 定义问题。

**但更要紧的是：本报告用已冻结的基线轮产物做交叉验证后发现，基线轮曲线塌不下来的真正原因
不是退化上限太低。** 实测基线最重档的 `target_lap_var` **早已塌到 clean 的 1.33% / 15.05% / 2.32%**
（blur / motion / contrast），**而 xcorr 跟住率仍停在 90.84% / 92.74% / 98.88%**。
⇒ **指标轴早就塌了，塌不下来的是跟住率**；把上限从 σ=4 抬到 σ=8 对**指标轴几乎没有增量**
（blur/contrast 两族已到底），对**跟住率轴**按基线曲线外推也只有约 **1~2 pp**，
**远不足以让 deep 轮规格 A1（最重档跟住率 < 50%）成立**。

> ⚠️ 因此：**本轮（deep）大概率仍然白跑**，但根因与上一轮写的不一样 —— 不是「轴不够狠」，
> 而是「**跟踪器对已经塌掉的画质不敏感**」。详见 §7 与 §9。

---
## 1. 方法与口径

### 1.1 用到的仓库实现（全部只读）

| 用途 | 调用 |
|---|---|
| 退化 | `degradation_4d.apply_degradation(frame, family, level, angle_deg)` |
| 窗口退化 | `degradation_4d.apply_window_degradation(frames, condition, angle_deg)` |
| 运动角度 | `degradation_4d.motion_angle_for(window_key, seed=42)` |
| 条件表 | `degradation_4d.build_conditions("deep")` + `DEEP_*` / baseline 常量 |
| v2 指标 | `target_roi_metrics.compute_roi_metrics(frame, cx, cy, w, h)` |

**只用 v2 目标中心指标**：`target_lap_var` / `target_local_contrast` / `local_scr`。
**未使用任何 v1 全图指标**（`global_contrast` / `global_lap_var` 一律不参与判定）。

### 1.2 合成图

- 320×320 灰度；背景 = 低频热斑 + 中频纹理（逐帧共享）+ 逐帧独立热噪声（σ=1.8）；
- 目标三种尺寸量级：**小 ≈8×8 / 中 ≈20×20 / 大 ≈60×60 px**；
- 极性：**白热**（峰值 +70）/ **黑热**（峰值 −55）；
- 形状：**高斯斑**（主网格，对齐真实红外点目标形态；σ = size/2.2）
  与 **方斑**（对照，硬边）；
- 窗口 5 帧（目标 ±1 px 抖动），指标取**首帧** —— 与 `run_eval_v2.py` 的
  `compute_roi_metrics(degraded[0], first.cx, first.cy, w, h)` 同口径；
- 运动角度取 `motion_angle_for("synth/<shape>/<size>/<polarity>")`，另做 8 个固定角度扫描。

### 1.3 判据口径

| 判据 | 操作化 |
|---|---|
| P1 单调性 | 逐档 `target_lap_var` 不增；违例判定 `b > a + 1e-6·max(\|a\|,\|b\|) + 1e-12` |
| P2 崩溃性 | 最重档 `target_lap_var` / clean ≤ **20%** |
| P3 对比度坍缩 | α=0.05 的 `target_local_contrast` 相对 clean 塌陷 ≥90% **且** 绝对值 ≤0.05 |
| P4 无极性 | 黑热集合与白热集合行为一致（`local_scr`/`target_local_contrast` 用绝对差） |
| P5 档位超集 | deep ⊋ baseline（三族各自）＋ `build_conditions("deep")` == 26、clean 最前 |

### 1.4 附加诊断（不参与 PASS/FAIL 判定，只用于解释机理）

- **背景地板分解**：同 seed 造一张**无目标**窗口，测同一退化下的 `target_lap_var` ⇒
  把「目标信号」与「核心区背景噪声/纹理」分开；
- **浮点域对照**：验证 `local_scr` 在对比度族的变化是否全部来自 uint8 量化。

---
## 2. P1　单调性：`target_lap_var` 逐档不增

### 2.1 判定

> ### ❌ P1 = FAIL
> 主网格（高斯斑）18 条「尺寸 × 极性 × 族」序列中 **2 条出现「不增反升」**，共 3 个档位对。

### 2.2 违例点名（主网格 · 高斯斑）

| 尺寸 | 极性 | 族 | 违例档位对 | 实测值 | 相对增幅 | 地板校正后是否仍违例 |
|---|---|---|---|---|---|---|
| **medium** | **blackhot** | **motion** | `motion_len9` → `motion_len11` | 101.447 → 103.310 | **+1.836%** | 是（信号 73.44 → 78.25） |
| **large** | **blackhot** | **blur** | `blur_sigma3.0` → `blur_sigma4.0` | 13.429 → 13.632 | **+1.513%** | 是（信号 4.78 → 5.48） |
| **large** | **blackhot** | **blur** | `blur_sigma7.0` → `blur_sigma8.0` | 11.982 → 12.266 | **+2.376%** | 是（信号 4.95 → 5.08） |

其余 16/18 条序列**严格逐档不增**；`contrast` 族 **6/6 全单调**。

### 2.3 方斑对照（硬边目标）—— 违例被放大一个量级

| 尺寸 | 极性 | 族 | 违例档位对 | 相对增幅 |
|---|---|---|---|---|
| medium | whitehot | blur | `blur_sigma0.6` → `blur_sigma1.0` | **+93.42%** |
| medium | whitehot | motion | `motion_len5` → `motion_len7` | **+24.57%** |
| medium | whitehot | contrast | `contrast_a0.15` → `contrast_a0.10` | **+38.64%** |
| medium | blackhot | blur | `blur_sigma0.6` → `blur_sigma1.0` | **+89.11%** |
| medium | blackhot | motion | `motion_len5` → `motion_len7` | **+13.84%** |
| large | whitehot | blur | `blur_sigma0.6` → `blur_sigma1.0` | **+84.60%** |
| large | whitehot | motion | `motion_len5` → `motion_len7` | **+22.62%** |
| large | blackhot | blur | `blur_sigma0.6` → `blur_sigma1.0` | **+80.48%** |
| large | blackhot | motion | `motion_len5` → `motion_len7` | **+21.66%** |

**机理**：硬边目标把 Laplacian 能量集中在 1 像素上，模糊不是"抹掉"而是"摊开"⇒ 单调性病态。
仓库自己的 `iqa_v2/test_iqa_v2.py::synthetic_ir_frame` 注释已写明这一点
（"若画成硬边矩形…任何模糊都会把这条能量摊开 ⇒ `target_lap_var` 反而变大 ⇒ 单调性断言病态"），
本报告**独立复现**了该现象。真实红外点目标是高斯型，**主网格口径可信**。

### 2.4 P1 附带的两个口径问题（必须写进 deep 轮的口径纪律）

**(a) 逐帧复制的测量噪声 ≫ 1e-6 容差。**
同一窗口 5 帧各自成一条序列时，违例对共 290 个，相对增幅中位 **+4.75%**、p90 **+16.84%**、
max **+43.91%**；违例集中在**中段**（blur 0.6→3.0、motion 3→11、contrast 0.8→0.25），
在塌到底的尾段反而为 0。
⇒ 含义：**「逐档严格不增」在单帧口径成立、在复本口径不可达**。
真实流水线只算首帧（`run_eval_v2.py:533`），所以本报告按首帧口径判定；
但 deep 轮若要用单调性做守卫，**容差必须显式写成"相对测量噪声"量级，不能写 1e-6**。

**(b) 大目标的 `target_lap_var` 测的是背景，不是目标。**
无目标背景窗口的同档 `target_lap_var`（地板）与目标窗口的实测值对比：

| 尺寸 | clean 目标实测 / 背景地板 | 最重档 实测 / 地板 / 目标信号 | 信号占比 |
|---|---|---|---|
| small (8) | 3059 / 236.9 | blur 15.16 / 11.73 / 3.43 | 22.6% |
| medium (20) | 422.8 / 312.6 | blur 40.13 / 6.40 / 33.73 | 84.1% |
| **large (60)** | **277.1 / 266.2** | blur 13.06 / 6.47 / 6.59 | **50.5%** |
| **large (60)** | **282.2 / 279.0** | motion 32.54 / 25.04 / 7.50 | **23.1%** |

大目标（60×60 高斯斑）clean 时目标只贡献 **277.1 − 266.2 = 10.9**（3.8%），
`target_lap_var` 几乎全部来自核心区里的背景纹理；σ=0.6→1.0 一步就掉到 7.7%（表 1B），
此后 σ=1.0~8.0 只在 **4.3%~5.6%** 之间小幅移动（动态范围 1.3 pp）。
⇒ **大目标窗口对锐度维度几乎没有信息量**，与 `REPORT_V2_FINAL.md` §5.3
「锐度维度…被大目标窗口的高 `lap_var` 基线摊平」的现象**同源**。
这 3 条 P1 违例全部落在 black-hot（大目标 + 黑热）格上就是这个原因。

---
