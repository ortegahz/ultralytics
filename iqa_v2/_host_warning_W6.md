
---

## W6 🔴🔴 膝点提取缺「退化曲线守卫」⇒ 平曲线上的**数值噪声**被当成膝点，且被双算法门禁误判为 `knee_reliable=1`

> 投递时间：2026-11-26 17:35（host 侧独立复算，用的是你 17:29 的冒烟产物 `iqa_v2/output_smoke/` + `/tmp/iqa_v2_cache_smoke/degrade_detail.csv`）
> ⚠️ **这条必须在写报告之前处理**：你 17:33 已经把全量流水线拉起来了（PID 2521796），**那一次跑出来的 `knee_thresholds_v2.csv` / `constraint_lines_v2.csv` 的三条线全部作废**，修复后要**重跑膝点阶段 + 重跑报告阶段**（eval 阶段的明细不用重跑）。

### 一、证据链（可复现）

1. `extract_knee` 里的守卫只有两条：`xs.size < 4` 和 `span > 0`（算 delta 用）。
   **没有任何「跟住率是否真的塌掉」的判断**。
2. `kneedle_knee` 的守卫是 `y_span <= 0`，但实际 `y_span` 常常是 **`1.1102230246251565e-16`**（浮点噪声，不是 0）⇒ **守卫放行**。
   随后 `yn = (curve_y - curve_y.min()) / y_span` = 除以 1e-16 ⇒ 数值爆炸 ⇒ `argmax|yn-xn|` 落到**任意**一个箱。
3. `pchip_knee` 在平曲线上二阶导≈0 ⇒ `argmax|second|` 同样取**任意**稠密点。
4. 两个「任意点」离得很近 ⇒ `delta_pct_of_span` 很小 ⇒ **被判定 `knee_reliable=1`**。
   实测 `contrast/local_scr/kcf_legacy/common_good`：`delta_pct_of_span = 0.0366%` ⇒ reliable=1。

### 二、host 侧复算出来的真实曲线（12 分箱，n=72 行）

`contrast / local_scr / kcf_legacy / common_good`：

```
x(=local_scr 中位) = 30.27 31.06 31.54 32.97 33.53 33.85 34.07 35.07 35.52 36.40 37.50 38.36
y(=跟住率)        = 0.986 0.986 0.986 0.986 0.986 0.986 0.986 0.986 0.986 0.986 0.986 0.986
y_span = 1.11e-16  ← 跟住率从头到尾没动，这条曲线上根本不存在“崩溃拐点”
```

`blur / local_scr / kcf_legacy / common_good`：12 个箱 **y 全部 = 1.000**，`y_span = 0`（这条被 kneedle 挡下了 ⇒ reliable=0，但 pchip 仍吐了 11.2331）。
`blur / target_lap_var`：y 全部 = 1.000，`y_span = 0`，pchip 吐 823.2189。
`contrast / target_lap_var`：y 全部 = 0.986，pchip/kneedle 吐 1563.05。

⇒ 你现在 `constraint_lines_v2.csv` 里的三条线
（`local_scr=31.0596`、`target_lap_var=1563.047`、`target_local_contrast=0.18337`）
**全部来自这种「跟住率从未塌掉」的曲线**，三个数都不是可交付的物理拐点。

### 三、为什么 contrast 族会这样（数学原因）

任务书 §四.2 明确要求「退化强度递增 ⇒ 局部画质指标单调下降」这条**单调性假设**。
对 `contrast` 族（`I' = αI + (1-α)μ`）：

- `local_scr = P90(|I_core − μ_ring|) / (σ_ring + 1e-5)` ⇒ 分子分母**同时乘 α** ⇒ **数学上不变**；
- `target_local_contrast` 分母含 `μ_ring`，同样近似不变；
- 而 XCorr / KCF 都是**归一化**匹配 ⇒ 对整体对比度缩放近乎免疫 ⇒ **跟住率不随 α 塌**。

host 侧实测的「窗口内指标相对极差」也印证了这一点：

| 退化族 | local_scr | target_local_contrast | target_lap_var |
|:--|--:|--:|--:|
| blur | 63.8% | 45.0% | 94.2% |
| motion | 32.9% | 18.5% | 68.5% |
| **contrast** | **8.2%** | 76.0% | 96.5% |

⇒ **contrast 族对 `local_scr` 基本不承载信息**，这一族的 `local_scr` 膝点**无物理意义**。
（旧报告 `REPORT_V2_ANTIUAV.md` §3.1 的 `contrast/local_scr = 3.20777` 是同一个坑的旧版本产物，**不要**再继承它。）

### 四、必须做的修复

1. **成功率塌陷守卫**（`extract_knee`）：`y_span <= max(1e-3, 0.05)`（或等价相对容差）⇒ 判「曲线退化」，
   `knee_reliable = 0`、`knee_x_final = nan`，**不要**让 1e-16 这种噪声通过。
2. **单调性守卫**（按 `(族, 指标)` 判）：用明细里「退化档位 → 指标中位」检查是否单调下降；
   不成立 ⇒ 该 `(族, 指标)` 组合标 `curve_degenerate = 1` 并**排除出约束线候选**。
   （这与旧报告 §3.5 对 JPEG 族的处理是同一类判断。）
3. **约束线候选条件**改为三者同时满足：`knee_reliable == 1` **且** `curve_degenerate == 0` **且** 成功率跨度达标。
   候选为空 ⇒ 该指标**没有可用约束线**，报告里要**明写「本轮该指标未取得可用红线」**，不许硬凑一个数。
4. **x_at_y50 在平曲线上无意义**（当前实现返回曲线首点）⇒ 退化曲线时也要置 nan。
5. **报告里必须写明**：contrast 族对归一化跟踪器近乎无效 ⇒ 其膝点不可用；
   以及「双算法一致」**不能**用来证明曲线有效（两条算法在平曲线上会一起给噪声点）。
6. 修复后**重跑膝点阶段与报告阶段**（eval 明细不必重跑），并在 `REPORT_V2_FINAL.md` 里给新数字与旧基线
   （819.55 / 2.69 / 3.21）的**差异做归因说明**。

### 五、顺带提醒（同一批冒烟产物里实测到的）

- 冒烟池只有 **12 个窗口**（`common_good ∩ calib` = 1 条序列），任何族都不会真的塌 ⇒ 冒烟阶段的数字**一律不可引用**。
- 全量跑完请先看 `knee_thresholds_v2.csv` 里 **`clean_success_rate`** 与各族**最重档位的跟住率**：
  blur σ=4.0 / motion L=11 / contrast α=0.15 三档必须有**真实的成功率下降**，否则“物理崩溃拐点”这个说法没有支撑。
