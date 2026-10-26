# 红外弱小无人机检测项目 · 精简背景（5 分钟上手卡）

> **用途**：新 Session / 外部模型 5 分钟建立上下文。**详细证据看 `manu/memory/memory.md`（886 行主档）**。
> **更新**：2026-10-23 · **瘦身版**：原 2,166 行全量已归档到 `manu/memory/archive/05_memory_compact_全量_20261022.md`（md5 `eaf91829a58d4abbd499e3d7bc2711a0`，逐字节一致）。
> **配套**：方法论 SOP 在 `manu/skills/`（4 个 Skill），**不在本文件重复**。

---

## 0. 30 秒摘要

红外弱小无人机检测（机载 + 固定点位）。三条线：**A 算法主线已冻结**（"改模型"路线 8 负 0 胜，突破只走系统级后处理）；**B 业务域交付已达标**（龙泉山 F1 90.605%）；**C 领域迁移路线 B 已拿到比基线更好的模型**（**103 trial 收尾**，10 个候选严格过线，最佳 U 比 RC1 高 +0.024975；收尾后复查无新候选超过它）。

**⭐ RC2 已正式交杖**（2026-10-22 10:40）：配置 = `w2.5 + lr_scale 2.2702`，上线权重 `trial_0080/weights/epoch3.pt`，**U 中位数 +0.048817**（3 次独立跑），比 RC1/Ep3 高 **+0.021238**。

**当前最关键事实**：混训增益的主导因子是 **Head `lr_scale`**（corr=+0.824），**不是混合比例**（决定性受控 A/B：同配比 w2.5，lr_scale=1.0 → U +0.0218，lr_scale=2.27 → U +0.0488）。"真正的 1:4" 不是最优，**w2.0~w2.5** 才是甜点区。

**RK3588 PTQ 量化（2026-10-23 全量三轨，int8 vs PT float，同帧/同 GT/同阈值 th=0.30）**：
Anti-UAV **−0.0189**（与历史 trial-0474 的 −0.0197 互证 = 复现）/ 黄金池 **−0.0394** / 001324 盲测 **−0.0795**；
机理 = **削峰（强目标，漏检）与抬底（弱小目标，虚警）两种相反失效模式**；阈值重标定**逐轨给**（位移 **0 / −0.10 / +0.05**，**不是通用 +0.10**）；
🔴 **输入 dtype 决定 `input_set` 延时**：只有 toolkit 自己全图 int8 量化能让输入停在 INT8（**0.34 ms**），
`--no-quant` / hybrid / **QDQ ONNX + `build(False)`（QAT）** 三条路径输入全是 FLOAT16（**45.41 ms**，板端 int8 **13.16 fps** vs fp16 **6.48 fps**）
⇒ **混合精度 / 混合 QAT 产物不是延时意义上的 int8 模型**。详见 §8 导航行。

**RK3588 PTQ 2026-10-23 下午（配置轴 / 校准集 / QAT 三连判）**：
**① PTQ 配置轴全部封死**（dtype 只有 `w8a8`、algorithm 只有 `normal`、`layer` 更差、`group{SIZE}` 平台死路、hybrid 延时出局、`optimization_level` 与 `flash_attention` **逐位零影响**）；
**② 校准集轴不是一条能单独兑现的杠杆**（v2 家族与原 `set_B/C/D` **全部泄漏评测帧**；干净集复评：子集口径确有收益但**一维此消彼长**，
**全量只验证过 set_6** ⇒ 黄金池 ≈+0.0000、001324 **−0.0064**；⚠️ **set_4 / set_5 的子集收益尚未全量，是否成立未知**）；
**③ QAT 判定为负**（rk3588 + toolkit 2.3.2 上**不能兼得"比 int8 基线更准"与"INT8 输入的延时"** ⇒ **维持 toolkit 原生 int8 + 逐轨阈值重标定 T3 0.45 / T2 0.30 / T1 0.50**）。
⚠️ 三条方法论闸门：**校准集先过泄漏自检** / **子集 F1 不能选型** / **热图余弦不能当选型判据**。详见 §3.6 与 §8 导航行。

---

## 1. 三条工作线状态

| 线 | 状态 | 关键数字 |
|:--|:--|:--|
| **A. 算法主线**（Anti-UAV 官方域） | ✅ 冻结 | 底模 Trial 0474 单帧 F1 **0.906361**（TP 21,643 / FP 1,004） |
| **B. 业务域**（龙泉山 24 序列） | 🔥 **RC2 已刷新，但 FAR 触线** | 冻结基线 F1 90.605% → **RC2(trial_0080) F1 94.007%（+3.40pp）/ R 93.143%（+7.26pp）**，pass 序列 **17→21 条**；但 **FAR 4.047% > 3.0% 上限 ❌**（FP 4,919→6,687，增量几乎全来自 `002251` 一条） |
| **C. 领域迁移**（路线 B 全帧对等混训） | ✅ **已收口** | **103 trial 结算** → **10 个严格过线候选**，最佳 `trial_0080` U=**+0.052554**（比 RC1/Ep3 高 **+0.024975**） |

**C 线子状态**：判别标准 ✅ / "真正的 1:4" 口径 ✅ / 超参搜索 ✅（2026-10-22 12:32 收尾）/ 上线候选 ✅ 7 个 / **权杖交接 ⏳ 暂定**（RC2=trial_0080，待 #89/#90 多种子确认）/ 位级复现对照 ⏳ #88 已入队。

---

## 2. 当前擂主与候选（**唯一真值**）

### 2.1 擂主与参照系

| 项 | 值 |
|:--|:--|
| **RC1**（路线 B 5-ep 运行的第 3 epoch 权重） | `runs/mixed_ft_route_b/RC1/epoch3.pt`，md5 `a845a95b77ac8762b7d033b64f61425e` |
| RC1 六点曲线 | `runs/mixed_ft_route_b/RC1/RC1.json` → `full_cycle_curve` |
| RC1 三轨（F1/FP） | T1 0.901918/941 · T2 0.998979/31 · T3 0.871684/461 |
| **RC1/Ep3 的 U（及格线）** | **+0.027579** |
| RC1/Ep5 的 U（切 5 ep 时换用它） | +0.026260 |
| 冻结基线（红线绝对参照） | T1 0.905736/1014 · T2 0.988436/26 · T3 0.837331/590 |

### 2.2 ⭐ RC2（**已正式交杖**，2026-10-22 10:40 多种子确认）

| 项 | 值 |
|:--|:--|
| **RC2 配置** | `mix_spec=w2.5`（实测 1:2.52）+ **`lr_scale=2.2702`** + 其余同 trial_0080 |
| **RC2 上线权重** | `runs/optuna_mix_ratio/trials/trial_0080/weights/epoch3.pt`（md5 `0f03ada97bd3903fa6097c7b31627d9c`） |
| **U 中位数（3 次独立跑）** | **+0.048817** = median(+0.052554, +0.048817, +0.048197)，极差 0.004357 ⇒ **≥ +0.045 门槛，已交权杖** |
| 比 RC1/Ep3 高 | **+0.021238** |
| 三次跑的 T1 F1 | 0.901910 / 0.899420 / 0.900168（RC1 = 0.901918 ⇒ **源域基本零退让**） |

### 2.2 上线候选（**严格**过一级红线 且 U > RC1/Ep3，共 7 个）

| trial | mix | 1:x | U | U−RC1 | T1 F1 | T1 FP | T2 F1 | T2 FP | T3 F1 | T3 FP | lr_scale |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **trial_0080** | w2.5 | 2.52 | **+0.052554** | **+0.024975** | 0.901910 | 1051 | 0.998949 | 33 | 0.897694 | 482 | 2.270 |
| trial_0082 | w6.0 | 6.07 | +0.037358 | +0.009779 | 0.900845 | 1050 | 0.999039 | 30 | 0.883615 | 448 | 2.773 |
| trial_0037 | w3.5 | 3.54 | +0.034749 | +0.007170 | 0.901830 | 970 | 0.998949 | 33 | 0.880089 | 495 | 1.466 |
| trial_0070 | w2.5 | 2.52 | +0.033556 | +0.005977 | 0.901582 | 1128 | 0.998949 | 33 | 0.879516 | 490 | 1.445 |
| trial_0028 | w3.5 | 3.48 | +0.031692 | +0.004113 | 0.903499 | 1057 | 0.998859 | 36 | 0.874404 | 560 | 1.355 |
| trial_0063 | w2.5 | 2.47 | +0.029854 | +0.002275 | 0.900828 | 1046 | 0.999009 | 31 | 0.876669 | 486 | 1.393 |
| trial_0065 | w8.0 | 8.05 | +0.027669 | +0.000090 | 0.901661 | 1108 | 0.998949 | 32 | 0.872931 | 530 | 1.805 |

- 权重：`runs/optuna_mix_ratio/trials/trial_XXXX/weights/epoch3.pt`（21.3 MB/个，epoch1/2/3 全留）；门禁报告在 `.../gate/epoch3/gate_report.json`
- **trial_0080 超参**：`mix_spec=w2.5`（实测 1:2.524）· `lr_scale=2.2702` · `warmup_epochs=0.588` · `weight_decay=2.6906e-04` · `focal_beta=2.75` · `offset_weight=0.40` · `mosaic=0.15` · `scale=0.10` · `translate=0.05` · `clip_max_norm=5.65` · `freeze_bn=0`
- 备份：`/mnt/data/siping/backup/mix_ratio_search_20261022/`（163 MB）
- ⚠️ 另有 8 个**只过放宽通道**但 U 超 RC1 的（trial_0061 +0.057861 最高）——**源域有退让 ⇒ 不选那批**

---

## 3. 关键结论速查

### 3.1 分档统计（**最终 104 行 / 103 trial 结算**，U 重算不信 CSV 列）

| mix | n | U 中位数 | U 最好 | 中位−擂主 |
|:--|--:|--:|--:|--:|
| **w2.5** | 35 | **+0.040633** | +0.053722 | **+0.013054** |
| **w2.0** | 8 | **+0.039976** | **+0.057861** | **+0.012397** |
| w6.0 | 3 | +0.037358 | +0.037780 | +0.009779 |
| w1.5 | 5 | +0.032240 | +0.047631 | +0.004661 |
| w3.5 | 9 | +0.030069 | +0.039274 | +0.002490 |
| w3.0 | 7 | +0.028851 | +0.041612 | +0.001272 |
| w5.0 | 8 | +0.027138 | +0.044525 | −0.000441 |
| w4.0 | 10 | +0.024738 | +0.046743 | −0.002841 |
| w8.0 | 6 | +0.023933 | +0.029824 | −0.003646 |
| natural | 11 | +0.021395 | +0.042524 | −0.006184 |
| w1.0 | 2 | +0.016853 | +0.045522 | −0.010726 |

⇒ **w2.5 / w2.0 并列前二**，都落在锁定先验空间 w2.0~w2.5 内；**"真正的 1:4"（w4.0）排第 9**，低于 RC1/Ep3。
⇒ **锁死 RC1 原值超参后（唯一变量=比例）各档 U 全部挤在 +0.0139~+0.0220 且都低于 RC1 ⇒ 比例零效应**。

### 3.2 三条被数据证伪的"执念"（**不要再犯**）

| 执念 | 判决 | 证据 |
|:--|:--|:--|
| "比例是唯一命门" | ❌ | ① `corr(U, lr_scale)=+0.824` ② ⭐**干净受控 A/B（唯一变量=比例）**：锁死 RC1 原值超参（lr_scale=1.0）后各档 U 中位数 w4.0 +0.0220 / w2.5 +0.0218 / natural +0.0206 / w2.0 +0.0142 ⇒ **全部低于 RC1 自身且彼此几乎无差别 ⇒ 比例零效应** ③ ⭐**决定性对照**：同配比 w2.5，lr_scale=1.0 → U +0.0218；lr_scale=2.27 → U +0.0488（**差 +0.027**）。⚠️ 统计脚本里"比例效应显著"的读数是**被 lr_scale 混淆**的 |
| "1:4 必然最好" | ❌ | w2.0 +0.0445 > w2.5 +0.0413 > **w4.0 +0.0247** > natural +0.0214 |
| "单点复现假象" | ✅ 破除 | RC1 **自身**相邻 epoch 的 T3 F1 就在 0.8469~0.8749（跨度 0.028）、T1 FP 941~1232（跨度 291）摆动 ⇒ 单点当"应该复现到的值"必误判。⇒ **所有上线决策必须多种子确认**（RC2 即按此办理） |
| "位级复现一定能对上" | ⚠️ **逐项未达成、聚合等效** | `trial_0088`（natural + seed=0 + workers=16 + 量 Ep3）U **+0.021395**，逐项判 **NOT_REPRO**；但 `RC1原值超参+natural` 组（n=4）中位数 +0.020636、极差 0.004674、与 RC1 差 −0.006943 ⇒ **差/离散度 = 1.49 < 1 ⇒ 落在自身离散度内，属随机波动** ⇒ 不能断言存在系统性口径差。⚠️ 该次由**旧调度器进程**执行，`--workers` 实测仍是 8 ⇒ 严格说是**近似位级复现** |

### 3.3 锁定先验空间（新场景直接复用）

配比 **w2.0~w2.5** ＋ Head `lr_scale` **2.0~2.8** ＋ 骨干**锁死**（`b0~b4` / G1_frozen 74,848 参数 + eval 保护 BN）＋ 严格通道为上位门槛。

### 3.4 口径分界（**三个独立分界，勿混**）

| 分界 | 起点 | 含义 |
|:--|:--|:--|
| 调度口径 | `trial_0008` 起 | cosine 周期 5 ep；之前 3 ep 周期，末 epoch lr 仅 RC1 的 55% |
| 红线口径 | `trial_0011` 起 | 一级红线双通道（严格 + 受控放宽） |
| 测量 epoch | 切 5 ep 则 `trial_0012` 起 | 量的变 Ep5 ⇒ 参照系换 `RC1_EPOCH5` |

### 3.5 复现状态（**未结**）

trial_0000/0012/0014 三个 natural 对照 U 分别 +0.018219 / +0.022893 / +0.019878，比 RC1 低约 −0.008（差/离散度 1.65，未达显著）⇒ 残余来源候选：**dataloader workers 16→8**、种子。**最终判据 = #88 位级复现（seed=0 + workers=16 + 量 Ep3）**。
⚠️ 昨天的"位级复现"**没真正生效**——运行中的调度器是旧代码，`user_attrs` 覆盖逻辑是当晚才加进文件的（**改文件不会改运行中的进程**）。

### 3.6 ⭐ RK3588 PTQ：2026-10-23 下午的六条硬结论（详见 `PTQ_RK3588_int8_报告.md` §5 / §19~§24）

| # | 结论 | 口径 / 产物 |
|:--|:--|:--|
| 1 | **PTQ 配置轴全部封死**：`quantized_dtype` 只有 `w8a8`；`quantized_algorithm` 只有 `normal`；`quantized_method=layer` 更差（cos_hm 0.96791 vs 0.97309）；`group{SIZE}` 平台死路（合法 SIZE ∈ [32,64,96,128,160,192,224,256]）；`quantized_hybrid_level=1` 延时出局；`optimization_level=0/2` 与 `enable_flash_attention=True` **逐位零影响** | PC 模拟器 100 帧；`screen_olv0/olv2/fa/layer/hyb1_diag` |
| 2 | 🔴 **校准集轴不是一条能单独兑现的杠杆**：v2 家族（set_1/2/3）与原 `set_B/C/D` **全部泄漏评测帧**；干净集复评：**子集口径确有收益但一维此消彼长**（宽↔窄量化范围取舍），**全量只验证过 set_6** ⇒ 黄金池 ≈+0.0000、001324 **−0.0064**；⚠️ **set_4 / set_5 的子集收益尚未全量，是否成立未知** | 板端全量 17,187 / 11,810 帧；`full_scale_sweep_set6/`、`27_check_calib_leak.py` |
| 3 | ⭐ **输入 dtype 硬闸门**：只有「toolkit 自己全图 int8 量化」能让图输入停在 **INT8**（`inputs_set` **0.34 ms** / 20,400 KB）；`--no-quant`、hybrid 1/2/3、QDQ+`build(False)` 全是 **FLOAT16**（45.41 ms / 105,600 KB）⇒ **判定法：开 verbose 构建后 grep 日志的 `InputOperator` 一行** | 板端 + 构建日志 |
| 4 | ⭐ **QAT 最终判定（负结果）**：rk3588 + toolkit 2.3.2 上 **QAT 不能兼得"比 int8 基线更准"与"INT8 输入的延时"** ⇒ **维持 toolkit 原生 int8 + 逐轨阈值重标定（T3 0.45 / T2 0.30 / T1 0.50）** | 板端三轨；`QAT_RK3588_交付报告.md` |
| 5 | 🔴 **两处更正**：(a) `mean_values/std_values` **必须给**（不传 ⇒ 255 倍输入 ⇒ null control 判死）；(b) uint8 图输入 ONNX 在 build 期被拒（`Not Support Dtype: 2`，2.3.2 依旧） | 板端 null control + toolkit 报错原文 |
| 6 | **环境事实**：toolkit build **非 bit-reproducible**（同配方三个 md5，数值完全一致 ⇒ 引用 `.rknn` 必须带 md5）；板卡 `/mnt/manu` 的 NFS 挂载**不随工作站重启存活** | `models/`、`sim25/smoke_base/` |

⚠️ **子集 F1 不是选型依据**：100 帧 → 2000 帧 → 全量，同类错误第 3 次（v2set3 +0.050 → 板端 0.0000；set_6 黄金池 +0.020 → 全量 +0.0000）。
⚠️ **热图余弦不能当选型判据**（hyb1 的 cos_hm 最好但 F1 不占优；`layer` 的 cos 更差却 F1 更高）。
⚠️ **报选型结论必须写明口径**：固定 `th=0.30` 与"各臂各自最优阈值"会给出**相反的赢家**（§19.4）。

### 3.7 ⭐ RK3588 板端 f16 multi-scale **拼图**（Canvas Mosaic）推理 —— 龙泉山两个 val 轨对照（2026-10-26）

**结论：拼图方案可行，且 f16 的代价约等于 0。**（no track / 单帧臂；official val 全量 28,997 帧）

| arm | 计算图 | T2 黄金池 (7 seq / 17,187 帧) | T3 盲测 (1 seq / 11,810 帧) |
|:--|:--|--:|--:|
| A | PC f32 逐尺度（3 次前向，冻结基准） | **0.998949** (TP 16636 / FP 33 / FN 2) | **0.897694** (TP 8972 / FP 482 / FN 1563) |
| B | PC f32 拼图（1 次前向） | 0.999790 (TP 16636 / FP 5 / FN 2) | 0.895025 (TP 8735 / FP 249 / FN 1800) |
| **C** | **板端 f16 拼图（1 次前向）** | **0.999790** (TP 16636 / FP 5 / FN 2) | **0.895555** (TP 8733 / FP 235 / FN 1802) |

- **f16 代价 ≈ 0**：T2 上 **B→C = +0.000000（逐位相同）**，T3 上 **+0.000529**。
  ⇒ **A→C 的差是「打包」的账，不是「量化」的账**（T3 的 −0.002139 = 打包 −0.002668 + 量化 +0.000529）。
- **T2 上拼图反而比逐尺度高 +0.000840**（FP 33→5）；T3 上打包只掉 0.0027，却保住了多尺度增益（拼图口径 +0.0227）。
- **NPU 时间不变**：`rknn_run` 100.259 ms（拼图）== 单尺度 101~102 ms；**拼图装配（RGA ×3）仅 1.095 ms/帧**；
  全量 28,997 帧 **141.413 ms/帧 / 7.071 fps**。
- ⚠️ **口径守卫**：`th_base=0.3` + `dedup_radius=768` 必须从 gate 产物 `config.params` **现场抽取**；
  用脚本默认值（0.22/32）得到的是 **0.848330**，**不可与 0.897694 并列**。
  且该评测**每帧基本只留一个最高分候选**（dedup 半径 768 > 帧内任意两点距离）⇒ 不是常规多框平均精度。
- 🔴 **踩坑**：反投影若把「320 宽热图里的窗口」当紧凑小图（packed stride），**tag1/tag2 会读串**，
  而 tag0 跨满全宽不受影响 ⇒ **症状只在小尺度出现，极易被误判成 f16 损失**。

📌 **三条可复用方法论**（专题档 §7）：① **跨计算图的差值必须先拆成「同图对照」**（否则报的是差值、不是量化损失）；
② **口径参数按轨取**（`th_base=0.3`+`dedup_radius=768` 从各轨 gate 产物现场抽取，脚本默认值不可用）；
③ **「窗口」≠「紧凑图」**（反投影在 320 宽热图上裁子矩形必须传对 `src_stride`，否则**只有小尺度 tile 出错**）。

详见 `manu/memory/MOSAIC_RK3588_20261026.md` §2/§3/§5/**§7 方法论沉淀** 与 `manu/pipeline/npu/MOSAIC_RK3588_REPORT.md`；
错误归因与证伪记录见 `manu/memory/falsified_archive.md` **§三十九**。

---

## 4. 数据与 GT（**唯一真值**）

| 项 | 值 |
|:--|:--|
| 🔴 **业务域 GT（服务器绝对路径）** | `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010` |
| 本机 SSHFS 只读路径 | `/home/manu/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010` |
| ⚠️ 评测 `--labels-root` 取**服务器侧路径**，**不要**指本机 SSHFS | |
| 原始帧 | `.../龙泉山/frames_ir_jpg`（24 序列，`<SEQ>/frame_%06d.jpg`） |
| 🔴 **GT 分母 ≠ 文件数** | 目录 165,224 = **文件数**（165,223 标签 + `classes.txt`）；**GT 分母 = 133,279 行**（标注框行数）⇒ 用错指标系统性偏低 ~20% |
| 人工复核底稿 | `manu/docs/dataset/long-quan-tk.md`（24 条：16 pass / 8 un-pass + 7 条 `<ignore>`） |
| 真缺口 3 条 | `003002` 77.815 / `001324` 84.357 / `002001` 87.877 |
| 龙泉山 official（训练用） | `/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/`（train 13,918 / val 28,997） |
| Anti-UAV（大盘回放底座） | `/mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml`（train 43,008 / val 31,613） |
| 底模 | `runs/optuna_p0_nas/trial_0474/weights/best.pt` |
| 基线臂缓存 | `runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl` |

---

## 5. 环境与常用命令

### 5.1 环境

| 项 | 值 |
|:--|:--|
| **服务器** | `ssh -p 32222 huangzhe@192.168.99.40` → `cd /tmp/pycharm_project_10ae9e2e` |
| conda | `source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav`（Python 3.10.20；**服务器上 `python` 无 3 后缀**） |
| **SSHFS 代码镜像（写入即落服务器）** | `/home/manu/mnt/pycharm_project_10ae9e2e` |
| 本地仓库 | `/media/manu/1TB-Volume/workspace/ultralytics`（本地只有 `python3`，3.14.4，无 pytest） |
| 服务器 `/tmp` | ext4 `/dev/nvme0n1p2`（3.4T，**非 tmpfs**，重启不清空但写满会丢）⇒ 关键产物另存 `/mnt/data` |
| **RK3588 板卡** | `ssh -p 22 root@192.168.0.64`（root 明文口令，**不得写入对外材料**） |

### 5.2 高频命令（**执行机：服务器**，项目目录 `/tmp/pycharm_project_10ae9e2e`）

```bash
# 查搜索状态 / 选型报告
python manu/training/status_mix_ratio_search.py --top 20
python manu/training/status_mix_ratio_search.py --strict-only
python manu/training/status_mix_ratio_search.py --all-calibers

# 分档统计对账（bootstrap CI / 效应量 / 三轨分解）
python manu/training/analyze_mix_spec_results.py --boot 4000

# 复现判决（EXACT / BAND / NOT_REPRO）
python manu/training/check_control_repro.py --all

# 投放对照 trial
python manu/training/enqueue_control_trials.py --specs natural --exact
python manu/training/enqueue_control_trials.py --from-trial trial_0080 --count 2

# 门禁补测（⚠️ 不得与训练同卡）
python manu/training/auto_epoch_gate_check.py --weights <trial>/weights/epoch3.pt \
  --out-dir <trial>/gate/epoch3_retry --gpus 1 --device 3 --infer-gpu 3 --cache-gpus 3 \
  --baseline-cache runs/mixed_ft_route_b/baseline/uav_cache_deP0.pkl \
  --skip-longquan --infer-batch 4

# 三侧 md5 同步判据（铁律〇）
md5sum /media/manu/1TB-Volume/workspace/ultralytics/<f> \
       /home/manu/mnt/pycharm_project_10ae9e2e/<f>
ssh -p 32222 huangzhe@192.168.99.40 "md5sum /tmp/pycharm_project_10ae9e2e/<f>"
```

### 5.3 记忆体系自检（本地）

```bash
python3 manu/memory/validate_nav_table.py    # 归档完整性 + 59 条导航逐条解析
```

---

## 6. 铁律速览（**正文 `manu/memory/rules.md`，1,282 行，必读**）

| # | 铁律 | 一句话 |
|:--|:--|:--|
| 〇 | 本地优先 | 先改本地再同步；**严禁直接改服务器代码**；同步判据**只能靠 md5 比对** |
| 〇之二 | 基线冻结 | `manu/data/build_full_median_dataset.py` **严禁修改** |
| 〇之三 | 任务边界 | 只做用户交代的事；**禁止自启动未授权实验** |
| 一 | 禁擅自启动重活 | 训练/推理/重型探测需明确授权 |
| 二 | 缓存精简 | pkl 等 cache 控制在 ~100MB 级 |
| 三 | 本地改动自动同步 | 本地改完无感同步到服务器映射目录 |
| 四 | 事实实时入档 | 实验事实与阶段结论实时写进 memory 体系 |
| 五 | 受控 A/B | 重大非继承性实验**严禁越级对比 SOTA**，必须同源受控 A/B |
| 六 | 禁解冻底模 | 组件扩容/微调走受控渐进残差规范 |
| 七 | 主档 + 精简档同步 | 本文件 + `memory.md` + 归档三侧 md5 一致 |
| 八 | 并行委派 | 可并行的独立子任务必须委派子 Agent |
| 九 | 可复现记录 | 结论必带**脚本相对路径 + 完整命令 + 执行机 + 产物路径**；否定性结论**必须量中间量** |
| ★ | 评估不得与训练同卡 | 单卡同时只允许"训练"或"评估"二者之一（OOM 事故归纳） |

---

## 7. 已踩过的坑（**勿重试**）

1. `args.max-hours` 在 f-string 里被解析成 `args.max - hours` ⇒ 用 `args.max_hours`。
2. 门禁 `--max-track2-fp` / `--min-track1-f1` 是**绝对阈值**（默认 36 / 0.905736），传增量会误判 ⇒ 传 `GATE_TRACK2_FP_ABS_MAX=36` / `GATE_TRACK1_F1_ABS_MIN=0.900736`。
3. Optuna RDB `trial_id`(1 起) vs `Trial.number`(0 起) **差 1** ⇒ 自愈必须 `study.tell(t.number, state=FAIL)`；`get_trials(deepcopy=False)` 返回 **FrozenTrial**（不能直接传 `tell`）。
4. `enqueue_trial` 参数**须经 `suggest_*` 取值才生效**；Optuna 4.9 的 `skip_if_exists=True` **会跳过重复参数组合** ⇒ 必须 `skip_if_exists=False`。
5. **同一 Optuna study 不能扩搜索空间**（`CategoricalDistribution does not support dynamic value space.`）⇒ 正交实验须新建 study。
6. 官方 `build_dataloader` 的 `fusion-dilate` 默认 9，门禁用 32 ⇒ 必须显式传。
7. torch 1.13 无 `torch.amp.GradScaler/autocast` ⇒ 用 `torch.cuda.amp`。
8. 内联 `python -c` 的 f-string 嵌套引号会被 shell 吃掉 ⇒ **落文件执行**。
9. `pkill -f <pattern>` 会误杀 SSH shell ⇒ 落文件 .sh 按 PID 精确 kill；远程 `cmd && cmd &` 会整链后台化 ⇒ 用 .sh 脚本。
10. `Path.glob` 默认路径写错**静默返回空**；查进程数用 `ps -eo args | grep -c`（**不用 `pgrep -f`**）。
11. `multiscale_heatmap_cache.py:403` 的 `frame_index` 是排序位置非原始帧号；`eval_multiscale_cache_th.py:100` 只认 `{seq}__{digits}.txt`。
12. `auto_epoch_gate_check.py` 不带 `--strict-gate` 时 rc 恒 0，但 `gate_report.json` 仍无条件写出。
13. **调度器 `[WAIT]` 收尾块只等不收割** ⇒ 截止时刻在跑的 trial **不会写 `optuna_summary.csv` 行**，而三个分析脚本**全都只读该 CSV** ⇒ 最后一批会从统计视野里凭空消失。**实测已发生**：2026-10-22 截止时 `trial_0100/0101/0102` 跑完但未进 CSV（CSV 100 行 vs 已结算 102 个）。**兜底工具**：`manu/training/backfill_summary_rows.py`（以 worker 写的 `trial_result.json` 为准，用调度器自己的 `write_summary_row()` 补行，自动备份原 CSV）。⚠️ 该工具在 Optuna 4.x 下踩过 `FrozenTrial` **没有 `.name` 属性**的坑（须用 `getattr(t,"name",None)`）。
14. **运行中的进程不读新代码** ⇒ 改了 `optuna_mix_ratio_search.py` 也不会生效，必须重启进程。
15. `load_hm_model` 的 ABLATION-P0 GATE：不去 P0 时未真正丢弃 29 个 p0 张量会 RuntimeError。
16. 旧 `runs/ms_cache_v2` / `runs/ms_cache_nop0` 的 002001 是 18,014 帧旧特征，不可复用。
17. 🔴 **校准集泄漏**：`calib/set_B,C,D` 与 `calib2/set_1,2,3` **全部含评测帧**（`27_check_calib_leak.py` 实测：226/240/216/500/300/245 路径命中）⇒ **任何校准集对比先过自检**；四集里**只有 `set_A` 干净** ⇒ 旧结论「A/B/C/D 极差 0.048」依据被污染。
18. 🔴 **子集 F1 不能选型**（同类错误第 3 次：100 帧 → 2000 帧 → 全量）；**固定阈值 vs 各自最优阈值会给出相反赢家** ⇒ 报结论必须写明口径。
19. 🔴 **热图余弦不能当选型判据**（hyb1 的 cos_hm 最好但 F1 不占优；`layer` 的 cos 更差却 F1 更高）。
20. 🔴 **Q/DQ ONNX 必须传 `mean_values/std_values`**（旧记录说"不得传"是错的）：不传 ⇒ 板端 uint8 缓冲区被加宽成 FLOAT16（0..255）且图里无 `/255` ⇒ **255 倍输入** ⇒ null control 报 `FATAL: NPU output independent of input`。⚠️ 不是"重复归一化"（mean/std 管**输入张量**，Q/DQ 管**激活 clip**）。
21. **uint8 图输入 ONNX 在 build 期被拒**（`Not Support Dtype: 2` / `Missing quantization parameters for images`，2.3.0 / 2.3.2 依旧）。
22. **toolkit build 不是 bit-reproducible**（同配方三个 md5，数值完全一致）⇒ 引用 `.rknn` 必须带 md5；PC 模拟器 build 必须落**独立目录**，否则板端报 `model md5 MISMATCH` 中止整轮。
23. **板卡 `/mnt/manu` 的 NFS 挂载不随工作站重启存活**（板端无 fstab 项）⇒ 每次工作站重启手动 `mount -t nfs 192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu -o nolock,soft,timeo=5,retrans=2`。
24. **toolkit 的算子表（含 `InputOperator` 一行）只在 verbose 时打印** ⇒ 不开 `--verbose-build` 就判不了输入 dtype；knobs 汇总表还会把臂的 json 认成 `no_json` ⇒ **读数认臂内 json/log**。

---

## 8. 导航（**去哪儿找什么**）

| 你要 | 去哪儿 |
|:--|:--|
| 当前状态 / 擂主 / 最新结论 / §导航表 | `manu/memory/memory.md`（**1,140 行**主档，§11.5 为最新） |
| 完整历史（按年代切 5 卷，零丢失） | `manu/memory/archive/00~04_*.md`；compact 全量 `05_memory_compact_全量_20261022.md` |
| 铁律正文 | `manu/memory/rules.md`（1,282 行） |
| 失败与证伪实验全景 | `manu/memory/falsified_archive.md` |
| **"怎么做"（SOP）** | **`manu/skills/`**：`domain-data-curator` / `mixed-ft-orchestrator` / `three-track-gatekeeper` / `hparam-search-operator`（索引见 `manu/skills/REGISTRY.md`；⚠️ 运行时强制 **kebab-case**，下划线名会被 lint 拒绝） |
| 板卡 / NPU / OpenCL / RKNN 专题 | `manu/memory/heatmap_npu_trial0474.md`、`rknn_npu_board_run.md`、`opencl_fused_rk3588.md`、`cpp_port_rk3588.md`、`gmc_ocl_board_run.md` |
| **RK3588 PTQ 量化（int8 vs PT float 全量三轨 + 2026-10-23 下午的配置轴/泄漏闸门/QAT 判定）** | **`manu/memory/PTQ_RK3588_int8_报告.md`**（§5 **配置轴全表** / §9 全量三轨 / **§19 校准集泄漏闸门 + 干净集复评（分层口径）** / **§20 输入 dtype 硬闸门** / **§21 QAT 最终判定** / §22 两处更正 / §23 环境事实 / §24 th=0.25 对齐）；正文 `manu/memory/archive/07_PTQ量化调优_§五十六.md` 的「§五十七 夜间补录（2026-10-23）」与「**§五十八 夜间补录（2026-10-23 下午）**」；主档 `manu/memory/memory.md` §11~**§11.5** |
| **QAT 路线最终判定（负结果，2026-10-23）** | **`manu/memory/QAT_RK3588_交付报告.md`**（§1 三条构建事实 / §5~§7 板端三轨 / **§10.2 a2f 爆炸四刀定位** / §10.4 结论）；产物目录 `manu/memory/qat_trial_0080_report/` |
| **RK3588 板端 f16 multi-scale 拼图（Canvas Mosaic）推理 + 与 PC f32 逐口径对照（2026-10-26，龙泉山两个 val 轨 / 28,997 帧）** | **`manu/memory/MOSAIC_RK3588_20261026.md`**（§1 交付物 / **§2 六条硬结论（两轨对照表）** / §3 口径（**th_base=0.3 + dedup_radius=768 必须从各轨 gate 产物现场抽取**、每帧一个候选、像素契约、特征溯源闭环）/ §4 复现命令 / **§5 踩坑清单（stride bug / io-mem-nchw / 计时重复计数）** / §6 环境事实 / **§7 可复用方法论沉淀**）；正文 `manu/pipeline/npu/MOSAIC_RK3588_REPORT.md`；**错误归因与证伪记录 `manu/memory/falsified_archive.md` §三十九**；主档 `manu/memory/memory.md` §11.6；RGA 能力基线 `manu/pipeline/npu/RGA_RK3588_NOTES.md`；代码 `manu/pipeline/npu/rknn_mosaic_heatmap.cpp` + `ptq/35~38` |
| 业务域 / 难例 / 数据集 | `manu/memory/hard_cases.md`、`spatio_temporal_track.md`、`manu/docs/dataset/long-quan-tk.md` |
| 双专家 MoE / 渐进残差 / 容量 A/B | `manu/memory/dual_expert_moe_adapter.md`、`progressive_residual_training.md`、`model_scale_ab_test.md` |
| 标注工具 | `manu/memory/web_annotate_tool.md` |

**本地专用脚本**（需本机 SSH 密钥，**不同步到服务器**）：
- `manu/diagnostics/final_mix_search_report.sh` —— 搜索收口报告：一条命令出「调度器状态 / 严格过线候选 / 分档对局 / bootstrap 对账 / 复现判决 / 候选权重 md5 与备份交叉核验」
- `manu/diagnostics/monitor_mix_search.sh` —— 只读监控快照（追加写 `/tmp/mix_search_monitor.log`）

---

## 9. 维护规则

1. 本文件只写"活"的东西；**历史进 `archive/` 或专题档**。
2. 写结论必带：脚本相对路径 + 完整命令 + 执行机 + 产物路径。
3. 改动后必做：本地 → SSHFS → 服务器 **三侧 md5 比对**。
4. **Skill 化判据**："**怎么做**" ⇒ `manu/skills/`；"**做过什么/当前是什么**" ⇒ 本文件 / `memory.md` / 专题档。
5. 归档分卷**只增不改**；引用一律写 §编号 + 主档导航表。
6. 主档瘦身后，跑 `python3 manu/memory/validate_nav_table.py` 确认 59 条导航仍可解析。

**⚠️ 已知缺陷已实锤（2026-10-22 收尾）**：调度器 `[WAIT]` 块只等不收割 ⇒ 截止时刻跑完的 `trial_0100/0101/0102` **确实没写进 CSV**（CSV 100 行 vs 已结算 102 个），必须靠 `manu/training/backfill_summary_rows.py` 回收；最后一个 `trial_0103` 侥幸赶在超时前完成（门禁 1123.7 s，U=+0.032853，PASS，**已计入**）⇒ **本次没真丢 trial**，但丢行风险已确认存在。
