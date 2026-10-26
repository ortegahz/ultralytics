# RK3588 int8 量化交付报告（trial_0080 / PTQ）

> 任务：不启动（或并行探索）QAT，通过**校准集分布设计 + RKNN 量化算法探索**，为 RK3588 找到量化损失最小的 int8 模型。
> **PC 的 PyTorch(float) 版本是唯一精度参照**；板端 = RK3588 NPU 真机；同帧、同解码器、同 GT、同阈值 ⇒ 逐级可归因。
> 生成时间：2026-10-22 18:15（工作站 18:30 关机前的落盘版）；**2026-10-23 由服务器自主链路补全 §8 选型 / §9 全量三轨**（三侧 md5 一致）。
> ⚠️ **本报告有两处结论在 2026-10-23 被全量三轨纠正**：「阈值位移 +0.10」是 **001324 2000 帧子集**的局部结论，不是通用规律（§3.1 / §9.4）；「量化损失 100% 在虚警」只对 001324 子集成立，Anti-UAV 全量是**反向的削峰**（§9.3）。

---

## 0. 结论摘要

1. **int8 零检出的真因已坐实并修复**：`kl_divergence` 把 heatmap 输出张量量程剪窄到 **0.2399 < 判定阈值 0.3** ⇒ 全帧 0 检出。换 `normal` 即解（量程上限 0.9661）。
2. ⚠️ **（2026-10-23 全量三轨后已限定口径，见 §9.3）** 「量化损失 100% 在虚警」**只对 001324 2000 帧子集成立**（子集内 int8 TP 比 PT 还高 77 个、FP 从 171 涨到 527）；**Anti-UAV 全量是反向的"削峰"**：int8 比 PT **少 643 个 TP**、只多 109 个 FP。
3. ⚠️ **（2026-10-23 全量三轨后已纠正，见 §3.1 / §9.4）** 「量化把最优检测阈值推移 +0.10」**只对 001324 的 2000 帧子集成立**：全量三轨实测 Anti-UAV 位移 **0**、黄金池 **−0.10（反向）**、001324 **+0.05** ⇒ **阈值重标定是逐轨的，不是全局规律**。子集口径下阈值重标定把 int8 从 0.6448 拉到 **0.7090（+0.0642）**，量化损失从 −0.1036 收窄到 **−0.0379（回收 63%）**。
4. **但剩余 0.0379 阈值救不回来**（001324 2000 帧子集，th=0.60 时 int8 仍 FP 97 而 fp16 只有 11）⇒ 量化确实造出额外假峰 ⇒ **QAT 是唯一解**。
5. **PTQ 算法轴在 RK3588 上封闭**（实测）：只支持 `w8a8`；`w8a16`/`w4a16` 平台不支持 ⇒ 官方 `gdq` 出局；`mmse` 实测 ETA 4h08m ⇒ 放弃。
6. **QAT 技术路线已解锁**：探针判定 **`QDQ_ADOPTED`** —— toolkit 2.3.2 会采纳 ONNX 里 Q/DQ 的 scale/zero_point 并带到板端（5/5 案例，scale 逐位一致）。
7. **部署含义（2026-10-23 全量三轨修正）**：板端 int8 **不得沿用 PC 的 `th_base=0.30`**；但重标定目标**逐轨不同** ⇒ Anti-UAV **0.35** / 黄金池 **0.55** / 001324 **0.55**（§9.1）⇒ **不存在"通用 +0.10"**。
8. 🔴 **新增硬闸门（2026-10-23）**：**输入 dtype 决定 `input_set` 延时** —— 只有 toolkit 自己全图 int8 量化才能让图输入停在 INT8（`inputs_set` **0.34 ms**）；`--no-quant` / `quantized_hybrid_level=1/2/3` / **QDQ ONNX + `build(do_quantization=False)`** 三条路径都会把 `InputOperator` 变成 **FLOAT16**（`inputs_set` **45.41 ms**，内部内存 105,600 KB）⇒ **混合精度 / 混合 QAT 产物不是延时意义上的 int8 模型**（§9.5 / **§20**）。
9. ⭐ **配置轴全表（2026-10-23 下午，§5）**：`quantized_dtype` 只有 `w8a8`、`quantized_algorithm` 只有 `normal`、`quantized_method=layer` 更差、`group{SIZE}` 平台死路、`quantized_hybrid_level=1` 延时出局、`optimization_level=0/2` 与 `enable_flash_attention=True` **逐位零影响** ⇒ **除校准集外，PTQ 配置轴全部封死**。
10. 🔴 **校准集轴最终判定（§19，分层口径）**：**「校准集 v2 优于基线」被推翻**（v2 家族整族泄漏评测帧）；原「A/B/C/D 极差 0.048 ⇒ 校准集轴无效」的**依据有污染**（B/C/D 全泄漏，只有 set_A 干净）⇒ 用**干净**集复评：**校准集轴不是一条能单独兑现的杠杆** —— 子集口径上确有收益，但收益是**一维此消彼长**（宽↔窄量化范围的取舍）；**在唯一被全量验证的干净集 set_6 上被证伪**（黄金池全量 ≈+0.0000、001324 全量 −0.0064）⇒ **不能单独当选型杠杆**。⚠️ **set_4 / set_5 的子集收益尚未做全量验证，是否成立目前未知**（只能走全量口径判定，不能引用子集数字）。
11. ⭐ **输入 dtype 硬闸门独立成节（§20）**：附**零板卡成本的判定法** —— 构建开 toolkit verbose 后 grep 构建日志里的 `InputOperator <dtype>` 一行即可（`verbose=False` 时该表不打印）。
12. ⭐ **QAT 最终判定（负结果，§21）**：在 rk3588 + toolkit 2.3.2 上，QAT **不能**兼得「比 int8 基线更准」与「INT8 输入的延时」⇒ **维持 toolkit 原生 int8 + 逐轨阈值重标定**（T3 0.45 / T2 0.30 / T1 0.50）；详见 `manu/memory/QAT_RK3588_交付报告.md`。

---

## 1. 阶梯链（PT → ONNX → f16 → int8）

100 敏感帧，th=0.3，topk=100。脚本 `manu/pipeline/npu/ptq/05_ladder_verification.py`，
产物 `/mnt/data/siping/ptq_work/ladder/ladder_report.json`。

| step | F1 | ΔF1 | cos_hm | mse_hm | cos_off |
|---|---|---|---|---|---|
| L0_PT_p0_on | 0.7662 | — | — | — | — |
| L1_PT_p0_off | 0.7817 | **+0.0156** | 0.995429 | 7.446e-07 | 0.999065 |
| L2_ONNX_p0_off | 0.7817 | +0.0000 | **1.000000** | **2.026e-11** | **1.000000** |
| BOARD_f16 | 0.7739 | −0.0079 | 0.985614 | 1.753e-06 | 0.999850 |
| BOARD_int8(normal) | 0.7156 | **−0.0583** | 0.954663 | 5.891e-06 | 0.972562 |

- 累计对 L0：f16 **+0.0077**、int8 **−0.0506**。
- **ONNX 导出链无损**：`cos_hm=1.000000 / mse=2.026e-11`。
- **去 P0 优于带 P0**（+0.0156），与 PC 模拟器同向（p0_on F1 0.6724 vs p0_off 最高 0.7800）。

## 2. int8 零检出根因：输出量程削顶（与输入 dtype 无关）

| 标定集 | 算法 | out[0] scale | 量程上限 | 判定 |
|---|---|---|---|---|
| Set-A (BMP) | kl_divergence | 0.00094072 | 0.2399 | ❌ 全帧零检出 |
| trial-0474 原配方 (JPEG) | kl_divergence | 0.00092659 | 0.2363 | ❌ 同上 |
| Set-A (BMP) | **normal** | 0.00378871 | 0.9661 | ✅ 活 |
| Set-B (BMP) | normal | 0.00364057 | 0.9283 | ✅ 活 |

板端阈值探针（同一 kl 模型，500 帧 001324）：`th=0.3 → 0/500 检出；th=0.1 → 389/500；th=0.05 → 431/500`，
且 `score_max` **恒等于 0.2399 = 255×scale**（硬削顶指纹）。
**机理**（`08_calib_peak_range_analysis.py`）：热图分布极重尾 —— 99.97% 质量在 0.24 以下，目标峰（0.90~0.97）只占 0.0005% ⇒
KL/熵裁剪把尾巴当噪声丢掉，min/max 才保留。

**已作废的两条旧结论**：① 「输入 dtype 陷阱 ⇒ 正解是 uint8 图输入 ONNX」——**证伪**（toolkit 2.3.2 `abort rc=134 Not Support Dtype: 2`，
`--no-quant`/hybrid 同样崩；uint8 图与 float 图虽逐位等价但建不出模型）；② 「kl 臂排序 C>A>D>B」——**作废**（kl 臂模型是死的）。

## 3. ⭐ 量化损失的真实结构与阈值重标定

> ⚠️ **本节的"阈值推移 +0.10"结论已在 2026-10-23 全量三轨后被纠正,见 §3.1 与 §9。**
> 下面的数字来自 001324 的 **2000 帧子集**,它只在**该难序列**上成立;在 Anti-UAV 全量轨上位移为 **0**,在黄金池轨上甚至是 **−0.10(反向)**。

板端 001324 2000 帧，**同一批 TSV 离线重扫**（`manu/pipeline/npu/ptq/19_board_threshold_sweep.py`）：

| 臂 | th=0.30 | 0.35 | 0.40 | **0.45** | 0.50 | 0.55 | 0.60 | 0.70 |
|---|---|---|---|---|---|---|---|---|
| int8(normal) F1 | 0.6448 | 0.6826 | 0.6951 | **0.7090** | 0.6980 | 0.6858 | 0.6635 | 0.5962 |
| int8(normal) FP | 527 | 363 | 276 | **198** | 151 | 109 | 97 | 75 |
| f16 F1 | 0.7395 | **0.7469** | 0.7400 | 0.7218 | 0.7012 | 0.6744 | 0.6484 | 0.5895 |
| f16 FP | 184 | 109 | 67 | 42 | 26 | 18 | 11 | 7 |

（PT float @0.3：F1 0.7484 / TP 1282 / FP 171 / FN 520。）

- **量化损失 100% 在虚警**：int8 的 TP=1359 **高于** fp16 的 1273，FP 却是 527 vs 184。
- **量化把最优阈值推移 +0.10**（int8 0.45 / fp16 0.35）⇒ **板端 int8 上线不能沿用 0.3**。
- 阈值重标定：int8 **0.6448 → 0.7090（+0.0642）**；量化损失 **−0.1036 → −0.0379（回收 63%）**。
- **阈值救不回来的部分**：th=0.60 时 int8 FP 97 vs fp16 FP 11 ⇒ 量化造出的额外假峰 = QAT 的靶子。

### 3.1 🔴 纠正：「阈值位移 +0.10」不是通用规律（2026-10-23 全量三轨实测）

上表（以及 §0.3、§11.1、archive §五十六）里的「量化把最优阈值推移 +0.10」**只对 001324 的 2000 帧子集成立**。
把同一套阈值扫描（`19_board_threshold_sweep.py`）跑到**全量三轨**（每轨 int8 与 PT float 各扫 9 个阈值 0.30~0.70）后：

| 轨 | 帧数 | int8 最优 th | PT float 最优 th | **最优阈值位移** | 结论 |
|---|--:|--:|--:|--:|---|
| T1 Anti-UAV val | 31,613 | 0.35 | 0.35 | **0** | int8 峰值被"削"但最优工作点不动 |
| T2 黄金池 | 17,187 | 0.55 | 0.65 | **−0.10** | **方向相反**（int8 反而更早到峰） |
| T3 001324 盲测 | 11,810 | 0.55 | 0.50 | **+0.05** | 位移为正，且收益最大 |

⇒ **阈值重标定是逐轨的，必须逐轨给，不得写成"通用 +0.10"**。数据见 §9.1 / §9.4，产物
`/mnt/data/siping/ptq_work/full_scale_sweep/{anti,golden,blind}_sweep.json`。

## 4. 板端 2000 帧 Tier2 全表（001324，th=0.3，topk=100）

| 臂 | F1 | R | P | TP | FP | FN |
|---|---|---|---|---|---|---|
| PT float | 0.7484 | 0.6498 | 0.8823 | 1282 | 171 | 520 |
| f16（`p0_off_A_none`） | 0.7395 | 0.6410 | 0.8737 | 1273 | 184 | 529 |
| int8 normal set-A | 0.6448 | 0.5835 | 0.7206 | 1359 | 527 | 443 |
| int8 normal 预 letterbox | 0.6426 | 0.5835 | 0.7150 | 1370 | 546 | 432 |
| int8 normal set-B | 0.5970 | 0.5365 | 0.6728 | 1308 | 636 | 494 |
| int8 normal set-C | 0.6407 | 0.5753 | 0.7228 | 1330 | 510 | 472 |
| int8 normal set-D | 0.6371 | 0.5738 | 0.7163 | 1338 | 530 | 464 |

黄金池 1197 帧：PT 0.9955 / f16 0.9849 / int8-A 0.9420 / int8-lb 0.9436。
⇒ ~~**A/B/C/D 极差仅 0.048 ⇒ IR 分位数分层维度基本无效；一阶杠杆是算法与阈值，不是校准集。**~~
> 🔴 **（2026-10-23 下午纠正，见 §19.2 / §19.3）** 上面那句的**依据有污染**：实测 `calib/set_B`、`set_C`、`set_D`
> **全部泄漏评测帧**（路径命中 226 / 240 / 216，共享 001324 与黄金池 000703），**只有 `set_A` 干净**
> ⇒ 泄漏把 B/C/D 往评测帧的激活分布上拉，于是四集"看起来差不多"。
> ⇒ 该结论后来由**干净集 + 全量口径**独立证实（**在唯一被全量验证的干净集 set_6 上**校准集轴**无实质收益**，§19.3），但**不得再引用 B/C/D 作依据**；
> ⚠️ 该证实**只覆盖 set_6**（set_4 / set_5 尚未做全量，结论范围见 §19.3 第 (3) 条）。

## 5. ⭐ PTQ 配置轴全表：rk3588 + toolkit 2.3.2 上**全部封死**（实测，勿重试）

> **2026-10-23 下午扩写**：本节从"算法轴封闭"扩成**完整配置轴判决表**，逐项都有实测证据。
> 工具：`manu/pipeline/npu/ptq/02_batch_ptq_pc_screening.py`
> （新增 `--quant-method / --quant-hybrid-level / --auto-hybrid-cos / --auto-hybrid-euc / --optimization-level / --enable-flash-attention / --verbose-build`）、
> `run_ptq_knobs.sh`、`run_ptq_knobs2.sh`。
> 产物：`/mnt/data/siping/ptq_work/{screen_base,screen_v2b,screen_layer,screen_hyb1_diag,screen_olv0,screen_olv2,screen_fa,screen_grp32,screen_grp64}/`；
> 日志：`logs/run_ptq_knobs.log`、`logs/run_ptq_knobs2.log`、`logs/knob_*.log`。

| 旋钮 | 判定 | 证据 |
|---|---|---|
| `quantized_dtype` | **只有 `w8a8`** | `w8a16` / `w4a16` → `The quantized_dtype = 'w8a16' not support in 'rk3588'!`（2.3.2 用 `rknn.config()` 直探）⇒ **`group{SIZE}` 随之不可用** |
| `quantized_algorithm` | **只有 `normal`** | `kl_divergence` 削顶（输出上限 **0.2399** < 判定阈值 0.3，§2）；`mmse` ETA **4h08m**；`gdq` → `The quantized_algorithm = 'gdq' only support in ['w4a16', 'w8a16']!` |
| `quantized_method=layer`（per-tensor） | ❌ **更差** | 100 帧 PC 模拟器：cos_hm **0.9679118699845649** vs channel **0.9730922927692202**；mse **4.037e-06** vs **3.566e-06**（`screen_layer/screen_p0_off.json`）⇒ **保留 `channel`** |
| `quantized_method=group{SIZE}` | 🚫 **平台死路** | `The quantize_method = 'group32' only support when quantized_dtype = 'w4a16'`；合法 SIZE ∈ **[32, 64, 96, 128, 160, 192, 224, 256]**（`group4` / `group512` → `Invalid quantized_method ... where {SIZE} is in [...]`） |
| `quantized_hybrid_level=1` | 🚫 **死路（延时）** | 2.3.2 复验：构建算子表 `InputOperator **FLOAT16**  CPU  (1,3,640,640)`；`Total Internal Memory Size: 105600KB`（= fp16）；板端 `inputs_set` **45.41 ms**（与 2026-10-16 archive/02 §28 在 2.3.0 上的结论一致） |
| `auto_hybrid_cos_thresh` / `auto_hybrid_euc_thresh` | 🚫 依赖 hybrid ⇒ **同死** | 2.3.2 的有效 hybrid 值是 `quantized_hybrid_level=1`（**0 = 关闭**），不是 8/16 |
| `optimization_level=0` / `=2` | ➖ **零影响** | cos_hm **0.9730922927692202** / cos_min **0.751060131865189** / mse **3.566024651078652e-06** / F1 **0.7298578199052133**（TP 77 / FP 17）—— 与基线**逐位相同**；只有 `.rknn` 字节 md5 不同（olv0 `d6d985fb…` / olv2 `b1dcdd77…` / base `75d84fcd…`） |
| `enable_flash_attention=True` | ➖ **零影响** | 同上**逐位相同**（`screen_fa` 的 `.rknn` md5 `220895b3…`） |
| 校准集 A/B/C/D | ⚠️ **原依据已污染** | `set_B/C/D` **泄漏**评测帧（§19.2）⇒ 「极差仅 0.048」不能作为"校准集轴无效"的依据 |
| **校准集（干净集 set_4/5/6/7/8/9）** | ⚠️ **子集口径有收益、全量口径不兑现** | **全量**口径（**唯一被全量验证的干净集 = set_6**）：黄金池 **≈+0.0000**、001324 **−0.0064**、Anti-UAV 待跑（§19.3）⇒ **不能单独当选型杠杆**；⚠️ **set_4 / set_5 的子集收益尚未做全量，是否成立未知**（§19.3 第 (3) 条） |

⇒ **配置轴全封死**：除了「校准集」这一个**模型级**变量，PTQ 的**每一个配置旋钮**都已被实测判死；
而「校准集」在**干净集 + 全量口径**下也**无实质收益**（**仅覆盖唯一被全量验证的 set_6**；set_4 / set_5 尚未全量，§19.3）⇒ **PTQ 侧已无杠杆**。

> ⚠️ **汇总表的已知假阴性**：`logs/run_ptq_knobs2.log` 里 `screen_fa / screen_olv0 / screen_olv2` 显示 `no_json / 0.00000`
> —— 那是**汇总器按 tag 取 json 文件名**没对上（这些臂的 json 落成 `p0_off_A_normal.json`），
> **每个臂自己的 `.log` 里内嵌了完整指标**（`screen_fa/p0_off_A_normal.log` 内嵌 json：cos_hm 0.9730922927692202 / F1 0.7298578199052133）。
> ⇒ **读数认「臂内 json / 臂内 log」，不要认汇总表。**

### 5.1 保留入口与零板卡判定法

- 保留待用入口：`rknn.hybrid_quantization_step1(proposal=True)`（官方逐层敏感度建议）+ `step2`（生成混合量化模型）
  —— 但**混合路径的图输入必为 FLOAT16**（§9.5 / §20）⇒ 延时上不划算。
- ⭐ **零板卡成本的配置轴判定法（新做法，2026-10-23 下午）**：构建时开 toolkit verbose，
  构建日志的算子表里有 `D RKNN: … InputOperator  <dtype>  CPU  (1,3,640,640)` 一行，**直接 grep 即可**；
  `02_batch_ptq_pc_screening.py` 已实现（`--verbose-build` + 报告字段 `input_op_dtype`）。
  ⚠️ `verbose=False` 时该表**不打印**（实测 0 次命中）⇒ 不开 verbose 就判不了输入 dtype。
- 平台约束的直探方法：`rknn.config(target_platform="rk3588", **cfg)` 直接触发 `ValueError`，无需 build；
  本轮用它把 `w8a16/w4a16/gdq/group{4,32,64,96,256,512}` 的报错原文一次收齐（见上表证据列）。

## 6. QAT 路线解锁：`QDQ_ADOPTED`

`manu/pipeline/npu/ptq/17_probe_qdq_import.py`（三侧 md5 `145408c66681defb22c8391d82ba9517`），
产物 `/mnt/data/siping/ptq_work/qdq_probe/qdq_probe.json`：

- **toolkit 2.3.2 采纳 ONNX 的 Q/DQ `scale`/`zero_point`**，一路带到板端输出张量（5/5 案例）。
- 硬约束：Q/DQ ONNX **必须** `build(do_quantization=False)`（与 `do_quantization=True` 同给 ⇒ `rc=3 / ret=-1`）。
- 🔴 **更正（2026-10-23 下午，见 §22.1）**：~~**不能**再传 `mean_values`/`std_values`~~ —— **这条是错的**：
  **Q/DQ ONNX 必须给** `mean=[[0,0,0]]` / `std=[[255,255,255]]`。
  不给时板端驱动上传的 uint8 缓冲区被 runtime 直接加宽成 FLOAT16（0..255）、图里没有 `/255`
  ⇒ 网络看到 255 倍大的输入 ⇒ 饱和 ⇒ **null control 直接判死**（`FATAL: NPU output independent of input`）。
  ⚠️ 这**不是**「重复归一化」：`mean_values/std_values` 描述**输入张量**，Q/DQ 描述**激活 clip**，两者正交。
- torch.ao 端到端已验证：`prepare_fx`+`convert_fx`(per-channel) → `export(opset 13)` → `load_onnx` → `build(False)` →
  板端 `out[0] type=INT8 zp=-91 scale=0.00544954`（ONNX 写 `scale=0.005449536256492138, zp=37` ⇒ **scale 逐位一致**）。
- 板端 I/O 契约：**1 输入** `[1,640,640,3]` uint8 NHWC + **2 输出** `[1,1,320,320]`、`[1,2,320,320]`。

## 7. 两个变体对比（带 P0 / 去 P0）—— p0_on 在板端 int8 完全失效

| 变体 | PC 模拟器 F1 (100 敏感帧) | cos_hm | mse_hm | 板端 int8 (001324 2000 帧) | out[0] scale | 量程上限 |
|---|---|---|---|---|---|---|
| p0_off（去 P0）set_3 | **0.7800** | 0.9734 | 3.44e-06 | F1 0.6448 / FP 527 | 0.00378871 | 0.9661 |
| p0_off（去 P0）set_D | 0.7290 | 0.9797 | 2.19e-06 | F1 0.6371 / FP 530 | — | — |
| **p0_on（带 P0）** | 0.6724 | 0.9712 | 4.30e-06 | **ZERO_DETECTIONS（0 检出）** | **0.00380537** | **0.9704** |

- ⇒ **去 P0 明显更优**（模拟器 +0.1076 F1），与阶梯链 L1（+0.0156）同向。
- ⇒ **p0_on 的 int8 模型在板端完全失效，但量程正常**（scale 0.00380537 ⇒ 可表示 0.9704，远高于阈值 0.3），
  ⇒ **不是"输出量程削顶"那一类死模型**，而是**板端特有失效**（同一模型在 PC 模拟器上 F1=0.6724 正常）。
  ⇒ 该失效**不能用 PTQ 算法/校准集解释**，需单独排查 P0 分支在板端的量化行为（明日项）。
- 阶梯链镜像链（p0_on）的板端两级因此**缺位**，05 脚本优雅降级并在报告注明。

## 8. 选型（2026-10-23 由服务器链路补全）

规则（写死于 `16_collect_selection.py` + `20_finalize_and_launch.py`，非人工判断）：
- **REJECT**：out ceiling（255×out[0] scale）≤ 0.30 的模型（量程削顶 ⇒ F1 是坏模型假象，不是量化结果）。
- 主键：**黄金池（axis2）在自身最优阈值下的 F1**（部署口径）。
- 平局 1：001324/2000 帧最优阈值 F1；平局 2：th=0.30 的黄金池 F1；平局 3：out ceiling；平局 4：名字。
- 产物：`/mnt/data/siping/ptq_work/selection/selection_table.json`、`best_model.json`、
  `/mnt/data/siping/ptq_work/final_selection/final_choice.json`。

**实测结果**（扫轴：axis1 = `001324_sub2000` 2000 帧 / axis2 = `golden_pool_sub1200` 1197 帧；产物 `final_choice.json`，落盘 2026-10-22 17:53）：

| 模型 | 板端模型 md5 | axis1 最优 th / F1 | axis2 最优 th / F1 | 判定 |
|---|---|---|---|---|
| `p0_off_A_none`（**fp16 臂**） | `1a3faf2f9459cad8…` | 0.35 / 0.746898 | **0.30 / 0.984916** | 🏆 **规则 winner**（axis2 最优 F1 最高） |
| `p0_off_A_normal`（**int8 臂**，Set-A + `normal`） | `cedaeb74d066a64b…` | 0.45 / 0.708957 | 0.35 / 0.942821 | **int8 臂最优**；**本次全量三轨跑的就是它** |
| `p0_on_A_normal`（带 P0） | — | 无数据 | 无数据 | ❌ 板端 **0 检出**（TP 0 / FP 0 / FN 1802）⇒ 该臂被排除 |
| `p0_off_{1,2,3}_normal`（候选集 v2） | — | `null` | `null` | ⚠️ 选型扫描**未产出数据** ⇒ 本轮未参与定案 |

- `rejected: []` —— **没有任何臂触发 REJECT**（`out_ceiling` 全为 `null`，即选型阶段没有采到 `255×scale`）。
- 🔴 **规则与任务意图的错配（如实记录）**：选型规则的主键是「黄金池最优 F1」，它天然偏向精度更高的 **fp16 臂**
  ⇒ 规则 winner = `p0_off_A_none`（fp16），而任务目标是「找量化损失最小的 **int8** 模型」。
  **该规则没有写 int8-only 约束**；本轮全量三轨按原计划跑 **int8 臂** `p0_off_A_normal`
  （`status.json.model` = `/mnt/data/siping/ptq_work/screen_n/p0_off_A_normal.rknn`，md5 `cedaeb74d066a64ba2c82003993defb1`）。
  ⇒ 后续若要复用该选型规则，**必须先补一条"候选臂限定为 int8"的硬约束**，否则 winner 永远是 fp16。

## 9. 全量三轨（2026-10-23 补全：int8 vs PT float 全量对照）

三轨 = T1 Anti-UAV val 31,613 / T2 黄金池 17,187 / T3 盲测 001324 11,810 = **60,610 帧**，只跑一次。
驱动 `manu/pipeline/npu/ptq/14_run_full_scale_nofs.py`（服务器 tar 流分块，不依赖工作站 NFS），
采集阈值固定 **0.30**（与本文全部既有实验同口径、可直接对账）。
产物：`/mnt/data/siping/ptq_work/full_scale_int8/{blind,golden,anti}_board.tsv` + `{anti,golden,blind}_report.json` + `status.json`；
阈值曲线：`/mnt/data/siping/ptq_work/full_scale_sweep/{anti,golden,blind}_sweep.json` + `full_scale_sweep.json`。

### 9.1 全量三轨对照（int8 vs PT float，**同帧 / 同解码器 / 同 GT / 同阈值 th=0.30 / topk=100**）

| 轨 | 帧数 | GT | int8@0.30 F1 | PT@0.30 F1 | **量化损失@0.30** | int8 最优 F1@th | PT 最优 F1@th | 最优处损失 | 最优阈值位移 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **T1 Anti-UAV val** | 31,613 | 25,111 | 0.8603 | 0.8791 | **−0.0189** | 0.8606@0.35 | 0.8839@0.35 | −0.0232 | **0** |
| **T2 黄金池** | 17,187 | 16,639 | 0.9558 | 0.9952 | **−0.0394** | 0.9570@0.55 | 0.9971@0.65 | −0.0401 | **−0.10（反向）** |
| **T3 001324 盲测** | 11,810 | 10,535 | 0.6664 | 0.7458 | **−0.0795** | 0.7581@0.55 | 0.7950@0.50 | −0.0370 | **+0.05** |

- **GT 数** = TP + FN（两臂各自算，分母相同）：Anti-UAV 20,810+4,301 = 21,453+3,658 = **25,111**；
  黄金池 16,628+11 = 16,612+27 = **16,639**；001324 8,100+2,435 = 7,887+2,648 = **10,535**。
- **覆盖与完整性**（`status.json` + 三份 `*_report.json`）：**60,610 帧 / 123 块 / 0 失败 / 0 缺失 GT**
  （`done_chunks 123/123`、`errors: []`、每轨 `n_missing_gt: 0`、`chunks_failed: []`）。
- **时间窗**：watchdog 18:43:14 启动 → 首块完成 **18:44:01** → 末块完成 **20:33:44**，`status.json` 最后更新 **20:34:12** ⇒ **约 1 h 50 min**。
- **int8 模型三轨同源**：`/mnt/data/siping/ptq_work/screen_n/p0_off_A_normal.rknn`，md5 `cedaeb74d066a64ba2c82003993defb1`。
- **T1 明细（th=0.30）**：int8 TP **20,810** / FP **1,229** / FN **4,301**；PT float TP **21,453** / FP **1,120** / FN **3,658**。
- **T2 明细（th=0.30）**：int8 TP **16,628** / FP **763** / FN **11**；PT float TP **16,612** / FP **66** / FN **27**。
- **T3 明细（th=0.30）**：int8 TP **8,100** / FP **2,838** / FN **2,435**；PT float TP **7,887** / FP **1,364** / FN **2,648**。

### 9.2 与历史 trial-0474 对齐：−0.0189 是**复现**，不是新发现

历史记录：`manu/memory/heatmap_npu_trial0474.md` 与 `manu/memory/archive/02_流水线与量化主线_§二十~§三十五.md` §二十 / §二十六.1
（去 P0 四臂，Anti-UAV val 全量 **31,613 帧 / 25,111 GT**）：

| 臂 | 峰值 th | F1 | TP | FP |
|---|--:|--:|--:|--:|
| 臂1 PC-FP32 **+P0**（冻结基准） | 0.25 | 0.902316 | 21,601 | 1,167 |
| 臂2 PC-FP32 **−P0** | 0.25 | 0.901935 | 21,586 | 1,169 |
| 臂3 **NPU-int8 −P0** | 0.20 | 0.882231 | 20,818 | 1,265 |
| 臂4 **NPU-fp16 −P0** | 0.25 | 0.901401 | 21,548 | 1,151 |

- 历史 int8 量化损失：**峰值口径 −0.019704**（0.901935 − 0.882231，两臂各取自己的最优阈值）；
  **同点口径 th=0.25 为 −0.020986**（0.901935 vs 0.880949）；fp16 相对 PC-FP32 仅 **−0.000533**（−38 TP），挽回率 97.3%。
- 本次 Anti-UAV 全量的 GT 数（TP+FN = **25,111**）与帧数（**31,613**）**与之完全一致** ⇒
  本次 th=0.30 **同点口径** 的量化损失 **−0.0189**，与历史 −0.0197（峰值口径）/ −0.0210（同点 th=0.25 口径）**互相印证**。
- 🔴 **口径红线（写死）**：量化损失**只能**取 int8 与 PT float 在**同帧、同解码器、同 GT、同阈值**下的差值；
  **禁止**拿 int8 绝对 F1 去比历史 float 绝对 F1（两者阈值/模型代际都不同）。
- 复现命令见 §11（阈值扫描对全量 TSV 离线重扫，零板端成本）。

### 9.3 量化损失的机理：**两种方向相反的失效模式**

| 轨 | 目标形态 | 相对 PT float（th=0.30，同帧同 GT） | 失效模式 | 业务表现 |
|---|---|---|---|---|
| **T1 Anti-UAV**（强目标，峰值 0.90~0.97） | 峰值高 | **少 643 个 TP**、只多 **109** 个 FP | **削峰** | 噪声把高峰削低 ⇒ **漏检** |
| **T3 001324**（红外弱小目标，峰值 0.35~0.60） | 峰值弱 | **多 213 个 TP**、多 **1,474** 个 FP | **抬底** | 噪声把弱目标与背景一起抬起 ⇒ **虚警** |
| T2 黄金池（目标密集、强） | 峰值高、召回近饱和 | 多 16 个 TP、多 697 个 FP | 抬底为主 | 虚警（F1 仍 0.9558） |

- **T3 的同一现象在 001324 的 2000 帧子集上更强**：int8 TP **1,359** vs PT float **1,282** ⇒ **多 77 个 TP**；
  FP **527** vs **171** ⇒ **多 356 个 FP**（产物 `/mnt/data/siping/ptq_work/logs/tier2_fixed.log`）。
- **结论**：量化噪声是**绝对量级的加性误差** ⇒ 对峰值 **0.95** 的强目标**抬不动**（表现为削峰/漏检）；
  对峰值 **0.35~0.60** 的弱小目标**足以改写判定**（表现为抬底/虚警）。
  ⇒ **同一份 int8 模型在不同轨上会朝相反方向劣化，不能用单一轨的结论推广到全场景**。
- **支撑量化（昨日已测）**：热图 **99.97% 的质量在 0.24 以下**，决定检出的目标峰（**0.90~0.97**）只占 **0.0005%** 的质量
  ⇒ 正是"重尾分布 + 绝对量级加性噪声"这条机理的热图侧证据（§2、`08_calib_peak_range_analysis.py`）。
- **与历史 DJI 结论互证**：本次 Anti-UAV 全量里 `DJI_0051_2` **TP=2 / FN=576**（578 帧，召回 **0.35%**）、
  `DJI_0175_2` **TP=57 / FN=391**（448 帧，召回 **12.7%**）⇒ DJI 序列在 int8 下近乎全漏；
  历史 trial-0474「剔除 DJI 后召回 0.8955→0.8727（−2.27 pp），配对 McNemar **p=0.0129** 显著」**方向一致**。
  ⚠️ 两次的 ex-DJI 帧集定义不同（历史 550 帧 / 440 GT，本次 2 条序列 1,026 GT），**只做方向互证、不做数值对账**。

### 9.4 🔴 阈值重标定是**逐轨**的，不是全局规律（纠正 2026-10-22 的"+0.10"）

| 轨 | int8 最优 th | PT float 最优 th | **位移** | th=0.30 → 最优 的 F1 变化 | 量化损失@0.30 → @最优 |
|---|--:|--:|--:|--:|--:|
| T1 Anti-UAV | 0.35 | 0.35 | **0** | 0.8603 → 0.8606（+0.0003） | −0.0189 → −0.0232 |
| T2 黄金池 | 0.55 | 0.65 | **−0.10（反向）** | 0.9558 → 0.9570（+0.0012） | −0.0394 → −0.0401 |
| T3 001324 | 0.55 | 0.50 | **+0.05** | 0.6664 → **0.7581**（+0.0917） | **−0.0795 → −0.0370（回收 53%）** |

⇒ 2026-10-22 写下的「通用阈值位移 +0.10」**只对 001324 的 2000 帧子集成立**，**不得推广**：
全量口径下位移 = **0 / −0.10 / +0.05**，**逐轨给**。
⇒ 收益只在 **T3（弱小目标轨）** 上显著（回收 **53%** 的量化损失）；T1/T2 重标定几乎无收益（+0.0003 / +0.0012）。

### 9.5 🔴 新增硬闸门：**输入 dtype 决定 `input_set` 延时**（2026-10-23 实测）

| 路径 | 图的输入算子 | `inputs_set` | NPU 内部内存 | 备注 |
|---|---|--:|--:|---|
| **toolkit 自己全图 int8 量化**（本轮 `p0_off_A_normal`） | **INT8**（`type=2`，1 B/px，1,228,800 B） | **0.34 ms** | **20,400 KB** | ✅ 唯一能保住 int8 输入的路径 |
| `--no-quant`（fp16 模型） | **FLOAT16**（`type=1`，2 B/px，2,457,600 B） | **45.41 ms** | 105,600 KB | 小核标量逐元素展宽：45.41 ms ÷ 1,228,800 = **37 ns/元素**（≈74 cycles @2 GHz） |
| `quantized_hybrid_level=1/2/3` | **FLOAT16** | 45.41 ms | 105,600 KB | 2026-10-16 实测，**三档全 FLOAT16** |
| **QDQ ONNX + `build(do_quantization=False)`（QAT 路线）** | **FLOAT16** | 45.41 ms | 105,600 KB | **2026-10-23 由 QAT 臂构建日志实测确认**（见下） |
| 同上但把展宽挪到大核 | FLOAT16 | **14.77 ms** | 105,600 KB | 展宽是 CPU 标量循环，给大核更快但仍不是 0.34 ms |

- **QAT 臂的原始证据（2026-10-23 09:12~09:13）**：产物 `/mnt/data/siping/ptq_work/logs/qat_build_arms.log`，
  **8 个臂全部**打出
  `D RKNN: 0  InputOperator  FLOAT16  CPU  (1,3,640,640)  InputOperator:images`
  （臂名：`a0_qat_float` / `a1_qat_maxclip` / `a1b_base_maxclip` / `a2_qat_learned`，各带 `_nomeanstd` 变体）。
  ⇒ **QAT 路线产出的模型，输入张量是 FLOAT16，不是 int8。**
- **根因**（`manu/memory/archive/02_流水线与量化主线_§二十~§三十五.md` §二十八）：toolkit 把
  「**图的输入张量 dtype**」与「**计算精度**」绑死 ⇒ `--no-quant` ⇒ 输入 FLOAT16；全图量化 ⇒ 输入 int8；
  混合 ⇒ 它选中间态，把**输入**判成 float16 而只量化部分中间层 ⇒ **没有开关能让输入停在 int8 而图内混精度**。
- 🔴 **含义**：**混合精度 / 混合 QAT 产物即使精度更好，也不是延时意义上的 int8 模型** ——
  int8 相对 fp16 的核心延时优势就是 `inputs_set` **0.34 ms vs 45.41 ms**。
  板端受控实测（同一 manifest `001324_sub2000`、同批 BMP、同一脚本 `04_run_board_fast_eval.py`，
  产物 `/mnt/data/siping/ptq_work/logs/tier2_fixed.log`）：
  **int8 2000 帧 152.0 s = 13.16 fps** vs **fp16 2000 帧 308.6 s = 6.48 fps** ⇒ int8 **2.03×**。
- 证据出处：本地 `manu/docs/experiments/2026-10-16_hybrid_precision_int8_input.md`（§3.2 表格、§1 表）与
  `manu/memory/archive/02_流水线与量化主线_§二十~§三十五.md` §二十八（§28.2 / §28.4 / §28.5）。
- ⭐ **2026-10-23 下午 2.3.2 复验 + 独立成节**：本表的完整版（含 `quantized_hybrid_level=1/2/3` 三档全 FLOAT16、
  内部内存 105,600 KB、板端帧率 13.16 / 6.48 / 5.64 fps、以及**零板卡成本的判定法**）见 **§20「输入 dtype 硬闸门」**。

## 10. 交付物清单（`manu/pipeline/npu/ptq/`，三侧 md5 一致）

`ptq_common.py` + 01 校准集 / 02 PTQ 矩阵 / 03 快速清单 / 04 板端快速评测 / 05 阶梯校验 / 06 参照臂锚点 /
07 uint8≡float 等价 / 08 标定峰值分析 / 09 板端阈值探针 / 10 标定峰值探针 / 11 标准集变体 /
12 标定范围 A/B / 13 预 letterbox 标定集 / 14 无 NFS 全量驱动 / 15 夜班选型扫描 / 16 选型收集器 /
17 QDQ 导入探针 / 19 板端多阈值扫描 / 20 定案+自主启动全量。

## 11. 复现命令（服务器 `ssh -p 32222 huangzhe@192.168.99.40`）

```bash
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav   # 训练/评测
cd /tmp/pycharm_project_10ae9e2e

# 板端 2000 帧 Tier2
python manu/pipeline/npu/ptq/04_run_board_fast_eval.py \
  --models /mnt/data/siping/ptq_work/screen_n/p0_off_A_normal.rknn \
  --manifest /mnt/data/siping/ptq_work/fast/001324_sub2000.json --variant p0_off \
  --out /mnt/data/siping/ptq_work/tier2 --th 0.3 --topk 100 --board-models-dir /mnt/manu/ptq/models

# 阶梯链
python manu/pipeline/npu/ptq/05_ladder_verification.py \
  --frames /mnt/data/siping/ptq_work/fast/tier1_sensitive.json \
  --board-model f16=/mnt/data/siping/ptq_work/screen/p0_off_A_none.rknn \
  --board-model int8=/mnt/data/siping/ptq_work/screen_n/p0_off_A_normal.rknn \
  --out /mnt/data/siping/ptq_work/ladder --th 0.3 --topk 100 --board-models-dir /mnt/manu/ptq/models

# 离线阈值扫描（零板端成本）
python manu/pipeline/npu/ptq/19_board_threshold_sweep.py \
  --manifest /mnt/data/siping/ptq_work/fast/001324_sub2000.json \
  --tsv-dir /tmp/ptq_board_out/p0_off_A_normal \
  --thresholds "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70" --skip-pt \
  --out /mnt/data/siping/ptq_work/sweep_up/p0_off_A_normal

# 选型扫描（板端 2 ctx 并行）
bash manu/pipeline/npu/ptq/15_night_selection.sh /mnt/data/siping/ptq_work/selection 18:24

# 定案 + 自主启动全量（工作站可关机）
python manu/pipeline/npu/ptq/20_finalize_and_launch.py \
  --selection-dir /mnt/data/siping/ptq_work/selection --work /mnt/data/siping/ptq_work --pt-device cuda:1

# ⭐ 全量三轨（2026-10-22 18:43 起跑，18:44→20:34，约 1h50m，123 块）
#   产物：/mnt/data/siping/ptq_work/full_scale_int8/{anti,golden,blind}_board.tsv + *_report.json + status.json
python manu/pipeline/npu/ptq/14_run_full_scale_nofs.py \
  --work /mnt/data/siping/ptq_work --model /mnt/data/siping/ptq_work/screen_n/p0_off_A_normal.rknn \
  --out /mnt/data/siping/ptq_work/full_scale_int8 --th 0.3 --topk 100

# ⭐ 全量 TSV 阈值扫描 + PT float 参照臂（2026-10-23，零板端成本，产出 §9.1/§9.4 的曲线）
#   产物：/mnt/data/siping/ptq_work/full_scale_sweep/{anti,golden,blind}_sweep.json + full_scale_sweep.json
python manu/pipeline/npu/ptq/19_board_threshold_sweep.py \
  --manifest /mnt/data/siping/ptq_work/fast/001324_sub2000.json \
  --tsv-dir /mnt/data/siping/ptq_work/full_scale_int8 \
  --thresholds "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70" \
  --out /mnt/data/siping/ptq_work/full_scale_sweep
```

## 12. 踩过的坑（勿重试）

| 坑 | 结论 |
|---|---|
| toolkit 2.5.0 | 必崩（SDPA `IndexError`）⇒ 固定 **2.3.2** |
| toolkit 2.3.2 + uint8 图输入 | `abort rc=134 Not Support Dtype: 2` ⇒ **该路线封死** |
| `kl_divergence` | 输出量程削顶 ⇒ 死模型 |
| `mmse` | ETA 4h08m ⇒ 放弃 |
| `w8a16` / `gdq` | rk3588 不支持 ⇒ 封死 |
| 模拟器喂 float32 | cos 0.108 ⇒ 一律 uint8 NHWC |
| 板端 TSV | **每检测一个 TAB 列**（只读第 1 列会把 FP/Recall 算错） |
| `while read < $LIST` 里跑 python/scp | 继承 stdin 吃掉清单 ⇒ 每个 axis 只跑 1 个模型 |
| 板卡 ctx 计数 | 必须 `ps -eo comm`，`ps -eo args` 会把 bash 包装一起数 |
| 板端工作目录 | 必须 per-run（PID 隔离），否则并发互相删 TSV |
| 板卡 FUSE/sshfs | 内核 5.10 拒绝 fuse2 ⇒ 走 SSH tar 流 |
| `ps -eo args \| grep "<脚本名>"` | 会匹配到自己的 ssh 命令行 ⇒ 用 `name[_]pattern` 转义 |
| anti 轨 GT | 标签在 `uav_gmc_median/labels/val`（同名同格式）⇒ 已加 `ANTI_GT_ROOT` |
| 选型规则 winner 未必是 int8 | 主键是「黄金池最优 F1」且**没写 int8-only 约束** ⇒ winner = fp16 臂 `p0_off_A_none`（§8）⇒ 复用前必须补该约束 |
| 🔴 **校准集泄漏** | `calib/set_B,C,D` 与 `calib2/set_1,2,3` **全部泄漏评测帧**（`27_check_calib_leak.py` 实测）⇒ 校准集对比**必须先自检**（§19.2） |
| 🔴 **子集 F1 不是选型依据** | 同类错误第 3 次（100 帧 → 2000 帧 → 全量）：v2set3 模拟器 +0.050 → 板端 0.0000；set_6 黄金池子集 +0.020 → 全量 +0.0000（§19.3 / §19.5） |
| 🔴 **选型口径会改变赢家** | 固定 `th=0.30` 下 set_6 在 001324 是 0.6585 vs 基线 0.6448（像"三轨全胜"）；各臂**自身最优阈值**下是 0.6915 vs 0.7090（更差）⇒ 报结论必须写明口径（§19.3） |
| 🔴 **热图余弦不能当选型判据** | hyb1 的 cos_hm 最好但 F1 不占优（TP −58）；`layer` 的 cos 更差却 F1 更高（§19.5） |
| **工具链 build 不是 bit-reproducible** | 同一配方三次构建得到三个不同 md5（`75d84fcd…`/`1de9cb3b…`/`cedaeb74…`），但**数值完全一致** ⇒ 引用 `.rknn` 必须带 md5，数值结论不受影响（§23） |
| **板卡 `/mnt/manu` 的 NFS 挂载不持久** | 板端无 fstab 项 ⇒ 每次工作站重启都要手动 `mount -t nfs 192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu -o nolock,soft,timeo=5,retrans=2`（§23） |
| **knobs 汇总表假阴性** | `run_ptq_knobs2.log` 把 `screen_fa/olv0/olv2` 显示成 `no_json`，但臂内 log 有完整指标 ⇒ 认**臂内 json/log**（§5） |
| **`verbose=False` 看不到输入 dtype** | toolkit 的算子表（含 `InputOperator` 一行）**只在 verbose 时打印** ⇒ 不开 verbose 就判不了 INT8/FLOAT16（§20） |

## 13. 待补（工作站 18:30 关机 → 服务器自主链路补全）

> **2026-10-23 更新**：第 1~3 项**已完成并写入 §8 / §9**，划掉；第 4 项**部分完成**（QAT 臂已构建，但精度结果未并入本报告）。

1. ~~选型结果：`/mnt/data/siping/ptq_work/final_selection/final_choice.json`。~~ ✅ **已完成 → §8**
   （winner = `p0_off_A_none`（fp16，规则缺 int8-only 约束）；int8 臂最优 = `p0_off_A_normal`，全量三轨跑的就是它）。
2. ~~全量三轨结果：`/mnt/data/siping/ptq_work/full_scale/`（三轨 TSV + `status.json` + 报告）。~~ ✅ **已完成 → §9.1**
   （⚠️ 实际路径是 **`/mnt/data/siping/ptq_work/full_scale_int8/`**，不是 `full_scale/`；
   60,610 帧 / 123 块 / 0 失败 / 0 缺失 GT，18:44→20:34 约 1h50m）。
3. ~~全量口径的阈值曲线：对全量 TSV 跑 `19_board_threshold_sweep.py`。~~ ✅ **已完成 → §9.1 / §9.4**
   （产物 `/mnt/data/siping/ptq_work/full_scale_sweep/{anti,golden,blind}_sweep.json`；并据此**纠正**了「通用 +0.10」）。
4. QAT（LSQ 微调 + QDQ 导出）结果并入最终模型评估。 ✅ **已完成 → §21**（**负结果**）
   —— 8 个 QAT 臂已构建（`/mnt/data/siping/ptq_work/qat/trial_0080/rknn/`，日志 `logs/qat_build_arms.log`，**输入算子全 FLOAT16**），
   板端三轨已评：**T3 输给 int8 基线**（0.6863@0.45 vs 0.7090@0.45）、**T2 赢**（0.9509 vs 0.9428）、**T1 打平**；
   `a2f`（唯一 INT8 输入的 QAT 臂）**虚警爆炸 5~8 倍**，根因＝QAT 把 heatmap 头搬坏（§21）。
   ⏳ 仍缺：**QAT 臂的全量三轨**（板端 ≈5.8 h，板卡被别的 harness 占用）。
5. ~~**QAT 臂要 int8 输入怎么办**：QAT 走 float 图输入 ⇒ 输入算子 FLOAT16（§9.5）；
   uint8 图输入被 toolkit 拒（`Not Support Dtype: 2`）⇒ **"QAT + int8 输入"这条路尚未打通**。~~
   ✅ **已定位并给出唯一可行方向 → §21.4**：把 LSQ 从 3 个头部位点扩到全网络（约 85 个 conv），
   让 Q/DQ 覆盖主体，`build(do_quantization=False)` 才能既保住 INT8 输入又不让主体退化成 fp16。
6. （2026-10-23 下午新增）**干净集 set_6 的 Anti-UAV 全量轨**：⏳ **未跑**（`full_scale_r2` 仍在采 anti 轨，产物同目录）。
7. （2026-10-23 下午新增）**th=0.25 与历史 trial-0474 的全量同阈对齐**：⏳ **未做**
   —— 需板端以 collect-th ≤ 0.20 重采一次全量 TSV（§24），属板卡资源。


---

## 14. 🔴🔴 2026-10-23 新增硬红线：**v2 校准集家族（set_1/2/3）含评测帧泄漏，其 T2/T3 指标无效**

> 发现人：本轮 int8 精度攻关（服务器侧核查 `calib2/calib2_manifest.json` + 各 `set_*_sources.txt` 的序列分布）。
> 影响面：2026-10-23 之前所有**用 set_1 / set_2 / set_3 校准的 int8 臂**在 **T2 黄金池** 与 **T3 001324** 上的读数。

### 14.1 证据（校准帧的来源序列 vs 评测轨的序列）

| 校准集 | 来源构成（`set_*_sources.txt`） | 泄漏判定 |
|:--|:--|:--|
| **set_1**（v2-1 强目标组） | **120×001324**（T3 盲测轨）+ **114×000818**（T2）+ 10×000703（T2）+ 3×015052（T2）+ 2×001531（T2）+ 1×015000（T2） | 🔴 **250/250 帧全部来自评测轨** |
| **set_2**（v2-2 0474 复刻+强目标） | 200×train（✔）+ 15×001324 + 15×000818 + 14×000703 + 3×015052 + 2×001531 | 🔴 **50 帧泄漏**（T2/T3） |
| **set_3**（v2-3 混合组） | 85×train（✔）+ 85×set_A_bmp（Anti-UAV train ✔）+ **20×001324 + 20×000818 + 20×000703 + 7×015052** | 🔴 **67 帧泄漏**（T2/T3） |
| **set_A**（现任基线） | 125 Anti-UAV **train** + 125 龙泉山 official **train** | ✅ **干净** |

⇒ 量化器的激活范围**就是由标定帧的极值决定的** ⇒ 用评测帧标定 ⇒ 该帧的激活分布被"见过" ⇒
**T2/T3 上的 int8 F1 偏乐观**，且**不可用于跨臂比较**。

### 14.2 连带作废的结论

| 结论 | 原读数 | 判决 |
|:--|:--|:--|
| 「set_3 比 set_A 好 +0.050 F1」（PC 模拟器 100 敏感帧） | 0.7800 vs 0.7299 | ❌ **作废**（敏感帧含 001324/015052，而 set_3 校准过它们） |
| 「set_3 是最佳校准集」 | — | ❌ **作废**（T2/T3 泄漏；T1 未测） |
| 「v2 校准集值得上板」 | — | ⚠️ **前提失效**：板上 T3-2000 实测 set_3 与 set_A **完全相同**（best F1 均 0.7090@0.45；@0.30 为 0.6505 vs 0.6448）⇒ 模拟器上的 +0.050 **不迁移** |
| 全量三轨（60,610 帧）int8 数字 | T1 0.8603 / T2 0.9558 / T3 0.6664 @0.30 | ✅ **仍有效**（用的是干净的 set_A） |
| 选型 winner（`p0_off_A_none` fp16 臂） | — | ✅ **仍有效**（fp16 无标定集，不存在泄漏） |

### 14.3 新增操作红线

1. **校准帧只能来自训练侧**（Anti-UAV train / 龙泉山 official train / 其它训练数据）。
   **禁止**使用 `$W/bmp/**`（= val 侧的 BMP 帧目录）里的任何帧做标定。
2. 任何新校准集必须**落盘来源序列清单**并与三个评测轨的序列名单做交集检查（脚本化，见 §14.4）。
3. **筛选指标只能上板取**（见 §14.5），PC 模拟器不可用于排序。

### 14.4 泄漏自检（一行判据）

```bash
# 校准集 vs 三个评测轨：交集必须为空
python manu/pipeline/npu/ptq/27_check_calib_leak.py \
  --calib $W/calib2/set_3.txt --calib-sources $W/calib2/set_3_sources.txt \
  --manifests $W/fast/anti_uav_sub2500.json $W/fast/001324_sub2000.json $W/fast/golden_pool_sub1200.json
```

### 14.5 筛选纪律：排序必须上板，模拟器只作诊断

| 口径 | set_3 vs set_A 差值 | 判定 |
|:--|--:|:--|
| PC 模拟器 100 敏感帧 | **+0.050** | ❌ 不迁移（且含泄漏） |
| **板卡 001324 2000 帧（最优阈值）** | **0.000**（均 0.7090@0.45） | ✅ **板卡是唯一可信的排序口径** |

板卡很便宜：**2000 帧 = 153 s、2500 帧 = 190 s**（13 fps）⇒ 一次 5 臂 × 2 轨 ≈ 29 min。
⇒ 筛选脚本：`manu/pipeline/npu/ptq/26_board_screening_round.sh`（等板卡空闲 → 两轨扫 → 排名）。


---

## 15. 2026-10-23 下午：板卡筛选的三个结论 + 流水线两个真 bug

### 15.1 ✅ T1 板卡筛选结果（`anti_uav_sub2500`，2500 帧，collect_th=0.05，全阈值曲线）

| 臂 | best F1 | @0.30 F1 | TP / FP |
|:--|:--|--:|:--|
| `p0_off_A_normal`（**现任基线**，set_A + channel） | 0.8606 @0.30 | 0.8606 | 1639 / **94** |
| `p0_off_3_normal`（set_3 + channel） | **0.8666 @0.35** | 0.8618 | 1618 / **67（−29%）** |
| `p0_off_3_normal_layer`（set_3 + **layer**/per-tensor） | 0.8571 @0.30 | 0.8571 | 1629 / 95 |

⇒ **峰值覆盖型校准集（v2 家族）在 T1 上有真实收益**（T1 = Anti-UAV val，**v2 家族的泄漏只涉及龙泉山序列，对 T1 无影响**）：
**FP −29%、best F1 +0.006**，代价是 TP −21（削峰确实被缓解了一部分）。
⇒ **`quantized_method='layer'` 在 set_3 上更差**（0.8571 vs 0.8666）⇒ **粒度轴收敛到 `channel`**。
⇒ 与失败机理吻合：T1 的量化损失是**削峰→漏检**（少 643 TP），而峰值覆盖的校准集正是为"量化器要见过高���"设计的。

### 15.2 🔴 三个必须写进纪律的结论

| # | 结论 | 证据 |
|:--|:--|:--|
| 1 | **PC 模拟器 100 帧的端到端 F1 不可用于排序** | 模拟器：set_3 0.7800 vs set_A 0.7299（+0.050）；**板卡 2000 帧：两者完全相同**（best F1 均 0.7090@0.45） |
| 2 | **校准集必须只含训练侧帧** | set_1/2/3 的来源序列与三个评测轨**重叠**（§14.1）⇒ 其 T2/T3 读数无效；`27_check_calib_leak.py` 可一键自检 |
| 3 | **筛选必须同时覆盖 T1 与 T3** | T1 的失效是削峰、T3 是抬底，方向相反；只看 T3 会把"修 T1"的臂误判为无效 |

### 15.3 🔴 流水线里的两个真 bug（本轮踩到并已修）

| bug | 症状 | 根因 | 修法 |
|:--|:--|:--|:--|
| **板卡帧路径映射只覆盖龙泉山池** | T1 清单上板后 **每条序列 0/118 帧、0 检出**，却"成功"返回（59 fps = 根本没跑） | 清单行不在 `C.BMP_ROOT` 下时回退到**服务端绝对路径**，板卡读不到 | `04_run_board_fast_eval.py` 增加第二个池映射：`$W/pcfeat_bmp/images` → `/mnt/manu/pcfeat_bmp/images` |
| **模型投递依赖工作站 NFS** | 新模型上板 md5 不匹配（板卡上是别的会话的旧版） | 板卡的 `/mnt/manu` 挂的是**工作站**的导出，服务器写不到 | **服务器 scp 到板卡本地盘 `/userdata/ptq_models/`**，板卡 `--board-models-dir /userdata/ptq_models`（板卡 `/` 只剩 1.4 GB，`/userdata` 有 12 GB） |

⚠️ 两个坑都是"**看起来成功、其实没跑**"的形态 ⇒ 与铁律 §四·9g 同一族：**必须核对帧数**，不能只看 rc=0。

### 15.4 本轮新增工具（仓库内，三侧 md5 一致）

| 工具 | 作用 |
|:--|:--|
| `manu/pipeline/npu/ptq/26_screening_round.sh` | 一轮板卡筛选：等板卡空闲（**锁目录互斥**，不用 `ps|grep` 自匹配）→ 逐清单跑 `19_board_threshold_sweep.py` → 排名表 |
| `manu/pipeline/npu/ptq/26_screening_round.sh` | 服务器端一轮筛选（构建 + PC 模拟器 100 帧）；⚠️ 臂串格式 `set:quant_method:algo` |
| `manu/pipeline/npu/ptq/27_check_calib_leak.py` | 校准集泄漏自检（路径 + 序列双查） |
| `manu/pipeline/npu/ptq/28_make_calib_strong_train.py` | **无泄漏**的强目标校准集（只用 train 侧；ONNX Runtime 探 float 峰值 + top-N 选峰 + 域多样性 + 峰值缓存） |


---

## 16. 2026-10-23 下午续：**校准集轴的完整三轨板卡对照**（无泄漏版）

> 口径：`anti_uav_sub2500`(2500) / `001324_sub2000`(2000) / `golden_pool_sub1200`(1200)，
> 板端 collect_th=0.05，全阈值曲线取各自最优。同一批子集、同一二进制、同一 GT。
> 🔴 **（口径指针，见 §19.3）** 本节**全部是子集口径**（2500 / 2000 / 1197 帧）的数字；
> 唯一被**全量**复评过的干净集是 **set_6**，其子集收益在全量上**不兑现**（T2 持平、T3 −0.0064）⇒
> **本节的 set_4 / set_5 子集收益尚未做全量，是否成立未知**，不得直接当作选型依据。

| 臂 | 校准集 | T1 Anti-UAV | T3 001324 盲测 | T2 黄金池 | 三轨均值 |
|:--|:--|:--|:--|:--|--:|
| `p0_off_A_normal`（**现任基线**） | set_A（125 Anti-UAV train + 125 official train） | 0.8606@0.30 | **0.7090@0.45** | 0.9428@0.35 | **0.8374** |
| `p0_off_3_normal` | set_3（**含泄漏**） | 0.8666@0.35 | 0.7090@0.45 | — | — |
| `p0_off_4_normal`（新） | set_4 = **150 张 Anti-UAV train 强目标**（峰值 top-150，无泄漏） | 0.8603@0.30 | 0.6935@0.40 | **0.9578@0.35** | 0.8372 |
| `p0_off_5_normal`（新） | set_5 = **150 张双域 train 帧**（Anti-UAV train + official train，按 float 峰值排序，无泄漏） | **0.8692@0.35** | 0.6891@0.45 | 0.9508@0.30 | 0.8364 |
| `p0_off_3_normal_layer` | set_3 + per-tensor 权重 | 0.8571@0.30 | — | — | — |

### 16.1 三条结论

1. **峰值覆盖的收益是真的，但不是来自"峰值"**：无泄漏的 set_5（T1 **0.8692**，比基线 +0.0086、FP 94→70）
   甚至**优于泄漏版 set_3（0.8666）** ⇒ 收益**不是泄漏产物**。
2. **但它是"此消彼长"的一维取舍**：set_5 拿 T1/T2 的 +0.008 换来 **T3 的 −0.0199**；
   set_4 拿 T2 的 +0.015 换来 T3 的 −0.0155。
   ⇒ 机理一致：量化范围是**一维**的（宽↔窄），宽了弱目标底噪被抬起（T3 虚警），窄了强目标被削（T1 漏检）。
3. **域多样性 > 峰值覆盖**：set_4（纯 Anti-UAV 强目标）T1 只有 +0.0003；
   set_5（双域）T1 +0.0086 ⇒ **增益主要来自"校准分布覆盖两个域"，峰值覆盖是次要因素**。

### 16.2 🔴 交付口径的关键框架（决定选谁）

| 轨 | 角色 | 对交付的意义 |
|:--|:--|:--|
| **T1 Anti-UAV val** | **源域回归护栏**（模型源自 Anti-UAV） | **不能掉**（掉了就是源域退化），但**涨了对龙泉山交付没有直接价值** |
| **T2 黄金池 / T3 001324 盲测** | **业务域（龙泉山）** | **交付只看这两轨** |

⇒ 因此 **set_5 虽然 T1 最好，却把业务轨 T3 拉低 −0.0199 ⇒ 不能直接替换现任基线**。
⇒ 真正要的模型是**三轨都不低于基线**（T1 ≥ 0.8606、T3 ≥ 0.7090、T2 ≥ 0.9428）。
⇒ 下一候选 **set_6 = set_4 ∪ set_5（300 帧，双域 + 峰值覆盖，无泄漏，零重算）**：
把 set_4 的 T2 优势（0.9578）与 set_5 的 T1 优势（0.8692）合到一起，看能否**三轨同时不低于基线**。

### 16.3 已排除/已封死的 PTQ 旋钮（本日实测汇总）

| 旋钮 | 判决 | 证据 |
|:--|:--|:--|
| `quantized_algorithm='kl_divergence'` | ☠️ 死 | 端到端 F1 **0.0000**（A/B/C/D 四集全灭）；cos_hm 0.84–0.85 vs normal 0.973 ⇒ 不是阈值问题，是图本身崩了 |
| `quantized_algorithm='mmse'` | ⏱ 不划算 | 构建 ETA 4 h 08m（toolkit 逐层优化） |
| `quantized_method='layer'`（per-tensor） | ❌ 更差 | set_3 上 T1 0.8571 vs channel 0.8666 |
| `quantized_method='group{32..256}'` | 🚫 平台不支持 | `only support when quantized_dtype='w4a16'`，而 rk3588 不支持 w4a16 |
| `quantized_hybrid_level=1` | 🚫 延时出局 | 输入算子 FLOAT16 ⇒ `inputs_set` 45 ms（vs int8 的 0.34 ms）⇒ 违反硬闸门 |
| `optimization_level` 0/2/3 / `enable_flash_attention` | ➖ 无差异 | 三臂 F1 与 cos_hm 完全相同 |
| `bias_correction` / `act_equalization`（toolkit 的 PC 级后处理） | 🚫 API 不存在 | rknn-toolkit2 2.3.2 的 `RKNN` 类**没有**这两个方法（只有 .so 模块，未暴露） |
| **校准集** | ⚠️ **曾是唯一还有信号的轴，但依据已污染；子集口径有收益、全量口径不兑现** | 🔴 见 **§19**：v2 家族整族泄漏；原 A/B/C/D 的 B/C/D 也泄漏；**全量**口径（**唯一被全量验证的干净集 = set_6**）黄金池 ≈+0.0000、001324 −0.0064 ⇒ **不能单独当选型杠杆**；⚠️ set_4 / set_5 的子集收益**尚未全量验证**（§19.3 第 (3) 条） |


---

## 17. ⭐ 2026-10-23：**set_6 = set_4 ∪ set_5（300 帧，无泄漏）三轨全胜** —— 新候选冠军

> 🔴 **（2026-10-23 下午补注，见 §19.3 / §19.4）** 本节的"三轨全胜"是**固定 `th=0.30` 口径**下的结论，
> 且基于**子集**（2500 / 2000 / 1197 帧）。换成「**各臂各自最优阈值**」口径 + **全量**（31,613 / 11,810 / 17,187 帧）：
> - 001324：set_6 **0.6915@0.40** vs 基线 **0.7090@0.45** ⇒ **反而更差 −0.0175**；
> - 黄金池：set_6 **0.9570@0.55** vs 基线 **0.9570@0.55** ⇒ **≈持平（精确 −0.00007）**；
> - Anti-UAV：**待跑**（`full_scale_r2` 仍在采）。
> ⇒ **"新候选冠军"不成立**（在**唯一被全量验证的干净集 set_6** 上校准集轴**无实质收益**，§19.3）；本节保留作为**口径差异的活教材**。
> ⚠️ **子集 vs 全量的口径差**：本节 set_4 / set_5 / set_6 的收益**全是子集口径**；全量只验证过 set_6，**set_4 / set_5 尚未做全量**（§19.3 第 (3) 条）。

### 17.1 板卡三轨对照（部署阈值 0.30 口径，子集）

| 臂 | 校准集 | T1 Anti-UAV | T3 001324 | T2 黄金池 | 均值 |
|:--|:--|--:|--:|--:|--:|
| `p0_off_A_normal`（现任基线） | set_A 250 | 0.8606 | 0.6448 | 0.9420 | 0.8158 |
| `p0_off_4_normal` | set_4 150（Anti-UAV 强目标） | 0.8603 | 0.6490 | 0.9570 | 0.8221 |
| `p0_off_5_normal` | set_5 150（双域，峰值排序） | 0.8674 | 0.6444 | 0.9508 | 0.8209 |
| **`p0_off_6_normal`** | **set_6 = set_4 ∪ set_5（300）** | **0.8609** | **0.6585** | **0.9619** | **0.8271（+0.0113）** |

⇒ **三轨同时不低于基线，T3 +0.0137、T2 +0.0199、T1 持平（+0.0003）**。
⇒ 机理：**单域或单一切法的集都会"此消彼长"，把"双域 + 峰值覆盖"合起来才同时吃到两边**：
  - set_4（纯 Anti-UAV 强目标）把 T2 顶到 0.9570 但 T3 掉到 0.6490；
  - set_5（双域）把 T1 顶到 0.8674 但 T3 掉到 0.6444；
  - set_6（两者并集）⇒ **T1/T2 保住，T3 反而最高**（0.6585）⇒ 量化范围在三个域之间取得了更好的折中。

### 17.2 set_6 的构成与可复现性

- 帧源：**全部来自 train 侧**（Anti-UAV train + 龙泉山 official train），
  **零评测帧**（`27_check_calib_leak.py` ⇒ PASS，0 路径命中 / 0 序列命中）。
- 构成：`set_4`（150 张 Anti-UAV train 里 float 峰值最高的一批）+ `set_5`（150 张双域里 float 峰值最高的一批）。
- **零特征重算**：两个集合的 BMP 早已存在，set_6 只是它们的并集（310→300，去重后 300 帧）。
- 峰值探测：ONNX Runtime 跑 float ONNX（`28_make_calib_strong_train.py`），
  训练侧峰值分布 **p50=0.7238 / p95=0.9010 / max=0.9430**
  ⇒ ⚠️ **训练侧的峰值比 val 侧低**（v2 家族用的 0.954 阈值在 train 侧选不中任何帧），
  所以强目标集必须按**相对排序**（top-N）而不是绝对阈值来取。

### 17.3 下一步（定案路径）

1. **全量三轨板卡跑**（60,610 帧，约 1 h 50 min）+ 阈值扫描（零板卡成本）
   ⇒ 得到 set_6 与基线在**同一口径全量**上的三轨表。
2. 若全量口径同样三轨不劣 ⇒ **set_6 替换现任 int8 基线**，并把逐轨部署阈值写进交付参数。


---

## 18. 2026-10-23 下午续：#1 auto_hybrid / #4 业务域集 / #5–#8 的结果

### 18.1 ✅ **auto_hybrid 的输入算子是 INT8**（与 `quantized_hybrid_level` 不同）

`02_batch_ptq_pc_screening.py` 新增 `--auto-hybrid`（走 `rknn.build(auto_hybrid=True)`），
构建日志给出决定性一行：

```
W build: The default input dtype of 'images' is changed from 'float32' to 'int8' in rknn model for performance!
```

⇒ **auto_hybrid 不违反"输入必须 INT8"的硬闸门**（`quantized_hybrid_level=1` 的模型输入是 FLOAT16，2026-10-16 实测），
⇒ 它是 PTQ 轴上**唯一还没在板上验过的"混合精度"入口**。
⇒ 但它在 PC 模拟器上**明显变差**：set_6 + auto_hybrid 的 F1 **0.6444** vs set_6 基线 **0.7512**
（损伤集中在 001324：TP 19 / FP 30 / FN 23），**差 −0.107 远超模拟器噪声** ⇒ 需板卡裁决。

### 18.2 #4 业务域定向校准集（**纯龙泉山 official train，零泄漏**）

| 集 | 构成 | PC 模拟器 cos_hm | cos_hm_min | mse_hm | F1 |
|:--|:--|--:|--:|--:|--:|
| set_6（冠军候选，双域+峰值） | 300 | 0.9754 | 0.7846 | 3.15e-06 | 0.7512 |
| **set_8**（**纯龙泉山 official train**，72 强目标 + 100 多样性，172 帧） | 172 | 0.9792 | 0.6893 | 2.30e-06 | 0.7429 |
| **set_9**（龙泉山 official + 更多峰值覆盖，126 强 + 50 多样，176 帧） | 176 | **0.9823** | **0.8312** | **1.88e-06** | 0.7440 |

⇒ **set_9 的张量指标是全场最好**（mse_hm 1.88e-06，优于 set_6 的 3.15e-06）。
⚠️ **训练侧峰值分布**（ONNX Runtime 探 float ONNX）：
  - Anti-UAV train 池：p50=0.7238 / p95=0.9010 / max=0.9430
  - **龙泉山 official train 池：p50=0.7719 / p95=0.9194 / max=0.9545**（业务域的峰更强）
  ⇒ v2 家族用的绝对阈值 **0.954** 在 train 侧只命中个位数帧 ⇒ **强目标集必须按相对 top-N 取**。

### 18.3 #5–#8：三个干净的"无影响"结论（省掉板卡时间）

| 项 | 结论 | 证据 |
|:--|:--|:--|
| **#5 `compress_weight`** | ➖ **无影响** | cos_hm / mse_hm / F1 与 set_6 基线**逐位相同**（0.9754 / 3.15e-06 / 0.7512） |
| **#6 校准清单顺序** | ➖ **无影响** | 同一批帧打乱顺序重建 ⇒ 指标**逐位相同** ⇒ **标定与清单顺序无关**；此前"同配方重建 md5 不同"只是构建非确定性，**不影响精度** |
| **#8 `enable_flash_attention`** | ➖ **无影响** | 与 set_6 基线相同（且此前在 set_A 上也无效） |
| **#7 校准集规模** | ⏳ 在建（100 / 150 帧 vs 300 帧） | 构建中 |

⇒ 这三条既然在 PC 模拟器上与基线**逐位相同**，就**不必占板卡**（板卡是稀缺资源）。

---

## 19. ⭐ 2026-10-23 下午：校准集**泄漏闸门** + 干净集复评 ⇒ 校准集轴**不是一条能单独兑现的杠杆**

> 口径：板端，同帧 / 同解码器 / 同 GT / topk=100；子集与全量分别标注。
> 归档正文：`manu/memory/archive/07_PTQ量化调优_§五十六.md` 的「§五十八 夜间补录（2026-10-23 下午）」。
> ⚠️ **结论范围**：**全量只验证过 set_6**（唯一被全量复评的干净集）；set_4 / set_5 仍是**子集口径**，见 §19.3 第 (3) 条。

### 19.1 🔴 校准集泄漏闸门：`27_check_calib_leak.py` 全表（2026-10-23 下午实测）

工具：`manu/pipeline/npu/ptq/27_check_calib_leak.py`（**路径 + 序列双查**，服务器 `conda activate rknn`）：

```bash
W=/mnt/data/siping/ptq_work
python manu/pipeline/npu/ptq/27_check_calib_leak.py \
  --calib $W/calib4/set_6.txt --calib-sources $W/calib4/set_6_sources.txt \
  --manifests $W/fast/anti_uav_sub2500.json $W/fast/001324_sub2000.json \
            $W/fast/golden_pool_sub1200.json
```

| 校准集 | 帧数 | 判定 | 路径命中 | 共享评测序列 |
|---|--:|:--|--:|---|
| `calib/set_A` | 250 | ✅ **PASS** | 0 | — |
| `calib/set_B` | 250 | 🔴 **FAIL LEAKED** | 226 | 001324 |
| `calib/set_C` | 250 | 🔴 **FAIL LEAKED** | 240 | 001324 + 黄金池 000703 |
| `calib/set_D` | 250 | 🔴 **FAIL LEAKED** | 216 | 001324 + 黄金池 000703 |
| `calib2/set_1` | 250 | 🔴 **FAIL LEAKED** | 500 | 001324 + 5 条黄金池序列 |
| `calib2/set_2` | 250 | 🔴 **FAIL LEAKED** | 300 | 001324 + 5 条黄金池序列 |
| `calib2/set_3` | 250 | 🔴 **FAIL LEAKED** | 245 | 001324 + 6 条黄金池序列（000703/000818/001307/001531/015000/015052） |
| `calib4/set_4` | 150 | ✅ **PASS** | 0 | — |
| `calib4/set_5` | 150 | ✅ **PASS** | 0 | — |
| `calib4/set_6` | 300 | ✅ **PASS** | 0 | — |
| `calib4/set_6shuf` | 300 | ✅ **PASS** | 0 | — |
| `calib4/set_6_100` | 100 | ✅ **PASS** | 0 | — |
| `calib4/set_6_150` | 150 | ✅ **PASS** | 0 | — |
| `calib4/set_7` | 310 | ✅ **PASS** | 0 | — |
| `calib4/set_8` | 172 | ✅ **PASS** | 0 | — |
| `calib4/set_9` | 176 | ✅ **PASS** | 0 | — |

> `[paths]` 命中数 = **校准清单 + 来源清单两条表一起数**（同一帧在两条表里各记一次）。
> 机理：量化器的激活范围**就是**校准帧的激活分布 ⇒ 校准集见过评测帧 ⇒ 该轨 int8 F1 偏乐观、**臂与臂不可比**。

### 19.2 🔴 两条被推翻的结论（必须显式更正）

| # | 原结论 | 判决 | 依据 |
|:--|:--|:--|:--|
| 1 | 「校准集 v2（set_1/2/3）优于基线」 | ❌ **作废** | v2 家族**整族泄漏**（19.1）⇒ 模拟器上的 +0.050 是**泄漏假象**；板端真值：set_3 与 set_A 在 001324 上**完全一致**（都 0.7090@0.45） |
| 2 | 「A/B/C/D 极差仅 0.048 ⇒ 校准集轴无效」 | ⚠️ **方向成立，但原依据被污染** | 四集里**只有 set_A 干净**，B/C/D 全泄漏 ⇒ 泄漏把 B/C/D 往评测帧分布上拉，于是四集"看起来差不多"。⇒ 必须改用**干净集**重述（19.3），且**必须写明口径**（全量只验证过 set_6，set_4 / set_5 待全量） |

### 19.3 ⭐ 干净集复评：校准集轴的收益**只到"子集口径"为止**（分层判定，**不得越级泛化**）

**（1）子集口径（板端，干净集）** —— set_4 / set_5 / set_6 在**各自擅长的轨**上确有收益，
且**不是泄漏产物**（三者均 PASS 干净，`27_check_calib_leak.py`），但每一份收益都在**另一轨上被抵消**
（"一维此消彼长" = 宽↔窄量化范围的取舍）：

| 校准集 | 擅长的轨（子集口径） | 被抵消的另一轨（子集口径） |
|---|---|---|
| set_5 | T1 **0.8692@0.35**（基线 0.8606，**+0.0086**，FP 94→70） | T3 **0.6891@0.45**（基线 0.7090，**−0.0199**） |
| set_4 | T2 **0.9578@0.35**（基线 0.9428，**+0.0150**） | T3 **0.6935@0.40**（基线 0.7090，**−0.0155**） |
| set_6 | T2 **0.9628@0.40**（基线 0.9428，**+0.0200**） | T3 **0.6915@0.40**（基线 0.7090，**−0.0175**） |

原始子集全表（板端，`26_board_screening_round.sh` 产物 `screen_board_r{2,3,4,5}/<manifest>/<manifest>_threshold_sweep.json`，
各臂在**自身最优阈值**下的 F1）：

| 校准集（全部 PASS 干净） | 001324/2000 | anti_uav_sub2500/2500 | golden_pool_sub1200/1197 |
|---|--:|--:|--:|
| **set_A**（基线） | **0.7090@0.45** | 0.8606@0.30 | 0.9428@0.35 |
| set_4 | 0.6935@0.40 | 0.8603@0.30 | **0.9578@0.35** |
| set_5 | 0.6891@0.45 | **0.8692@0.35** | 0.9508@0.30 |
| set_6 | 0.6915@0.40 | 0.8609@0.30 | **0.9628@0.40** |
| set_7 | 0.6968@0.45 | 0.8622@0.35 | 0.9444@0.35 |

**（2）全量口径（板端，**唯一被全量验证过的干净集 = set_6**，int8 vs 同一 PT float 参照臂，同帧同 GT，9 个阈值离线重打分，
产物 `/mnt/data/siping/ptq_work/full_scale_sweep_set6/full_scale_sweep.json`，
模型 `screen_r2/p0_off_6_normal.rknn` md5 `bc252858d528c8217b7e81fcb24f1df9`）**：

| 轨 | 帧数 | 基线 set_A @0.30 | **set_6 @0.30** | 基线最优 | **set_6 最优** | **Δ(各自最优)** |
|---|--:|--:|--:|--:|--:|--:|
| **T3 blind 001324** | 11,810 | 0.6664 | 0.6674 | 0.7581@0.55 | **0.7517@0.55** | **−0.0064** |
| **T2 黄金池** | 17,187 | 0.9558 | 0.9562 | 0.9570@0.55 | **0.9570@0.55** | **≈0.0000**（精确 −0.00007） |
| **T1 Anti-UAV** | 31,613 | 0.8603 | **待跑**（`full_scale_r2` 仍在采 anti 轨） | 0.8606@0.35 | 待跑 | 待跑 |

⇒ **全量口径读数**：T3 blind 11,810 帧 **0.7517@0.55** vs 基线 **0.7581@0.55**（**−0.0064**）；
T2 黄金池 17,187 帧 **0.9570@0.55** vs 基线 **0.9570@0.55**（**−0.00007**）⇒ **子集收益不迁移**。

复现命令（服务器 `conda activate rknn`，**纯离线重打分，不占板卡**）：

```bash
python -u manu/pipeline/npu/ptq/23_full_scale_threshold_sweep.py \
  --run-dir /mnt/data/siping/ptq_work/full_scale_r2 \
  --bmp-root /mnt/data/siping/ptq_work/bmp \
  --anti-root /mnt/data/siping/ptq_work/pcfeat_bmp/images \
  --out /mnt/data/siping/ptq_work/full_scale_sweep_set6 \
  --tracks blind golden \
  --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70
```

**（3）⭐ 因此可下的结论（**分两句，不要合成一句**）**：

1. 「**校准集轴不是一条能单独兑现的杠杆**：它在子集口径上确有收益，但收益是**一维此消彼长**（宽↔窄量化范围的取舍），
   且在唯一被全量验证的干净集 set_6 上被证伪（T2 持平、T3 变差）。」
2. 「**set_4 / set_5 的子集收益是否在全量口径成立，目前未知**（尚未做全量）⇒ 若要判定它们，只能走全量口径，**不能引用子集数字**。」

⇒ **「黄金池 +0.020」是子集假象**（1197 帧 0.9628 vs 0.9428 → 全量 17,187 帧 0.9570 vs 0.9570）。
> （注：set_4 / set_5 的子集收益尚未做全量验证，不能一并断言为假象。）

### 19.4 🔴 选型**口径本身**会改变赢家（必须写明用的是哪个口径）

| 口径 | 001324：set_6 vs set_A | 黄金池：set_6 vs set_A | 结论 |
|---|---|---|---|
| 固定 `th=0.30`（§17.1，子集） | **0.6585 vs 0.6448**（+0.0137） | 0.9619 vs 0.9420（+0.0199） | 看起来「三轨全胜」 |
| 各臂**自身最优阈值**（子集） | 0.6915 vs 0.7090（**−0.0175**） | 0.9628 vs 0.9428（+0.0200） | 001324 反而更差 |
| 各臂**自身最优阈值**（**全量**） | **0.7517 vs 0.7581（−0.0064）** | **0.9570 vs 0.9570（≈0）** | **两边都不赢** |

⇒ 报任何选型结论，**必须同时写出口径**（固定阈值 / 各自最优；子集 / 全量），否则结论不可复现、不可对账。

### 19.5 ⚠️ 三条方法论闸门（**项目级规则，从此强制**）

| # | 闸门 | 触发过的真实翻车 | 落地动作 |
|:--|:--|:--|:--|
| 1 | **任何校准集对比必须先过 `27_check_calib_leak.py`** | v2 家族（set_1/2/3）整族泄漏；原 A/B/C/D 的 B/C/D 泄漏 ⇒ 一度把"四集极差 0.048"当成"校准集轴无效"的证据 | 校准集**必须**落盘 `set_N.txt` + `set_N_sources.txt`，建完立刻自检；**FAIL 的集不准进任何对比** |
| 2 | **子集 F1 不是选型依据，必须全量复核** | 同一类错误第 3 次出现：100 帧 → 2000 帧 → 全量。v2set3（泄漏）模拟器 +0.050 → 板端 0.0000；set_6 黄金池子集 +0.020 → 全量 +0.0000 | 板端候选排序一律用**全量**（60,610 帧）；子集只用于**诊断**与**排期** |
| 3 | **热图余弦（cos_hm）不能当选型判据** | hyb1 在 001324 的 cos_hm 最好（**0.97378** vs base **0.95638**）但 F1 不占优（召回 TP **−58**）；`layer` 的 cos 更差却 F1 更高 | 排序用**端到端 F1**；cos/mse 只用于**诊断量化器健康度** |

> 附：PC 模拟器本身的保真度已实测（**方法层结论仍成立**）—— 模拟器 vs 板端 ΔF1 ≤ 0.005
> （001324 int8：板端 0.6448 / 模拟器 0.6402；fp16：0.7395 / 0.7398）；
> 模拟器的 PT float 臂与板端 PyTorch 参照臂**逐项完全相同**（001324：F1 0.7484 / TP 1282 / FP 171 / FN 520；
> 黄金池 0.9955 / 1110 / 3 / 4）。
> ⇒ **模拟器足以做相对排序，绝对数字仍以板端为准**；剩余差异来自 NPU 定点算术 vs CPU 仿真（阈值附近的峰值翻转）。

### 19.6 ⚠️ 留档（教训）：本日 14:xx 的「校准集 v2 优于基线」已被推翻

- **当时的说法**：本日白天用 PC 模拟器做受控对照，得到「校准集 v2 `set_3` 在 001324 上 F1 0.7027@0.45 > 基线 0.6928@0.45」，
  并据此写下「校准集 v2 是唯一活着的模型级变量」。
- **推翻理由**：该对照**整个建立在泄漏校准集上**（`set_3` 245 路径命中、共享 001324 + 6 条黄金池序列）⇒ 数字本身没错，
  但**不构成"优于干净基线"的证据**；板端真值是 set_3 与 set_A 在 001324 上**完全相同**（都 0.7090@0.45）。
- **教训**：**任何校准集对比必须先过泄漏自检**（闸门 1）；**子集/模拟器的"收益"必须上板端全量复核**（闸门 2）。

---

## 20. ⭐ 输入 dtype 硬闸门（独立成节，2026-10-23 下午 2.3.2 复验）

> §9.5 是首次记录；本节是**独立、完整、可直接引用的版本**（含零板卡判定法与板端帧率）。

**只有「toolkit 自己全图 int8 量化」这一条路径能让图的输入保持 INT8。**

| 路径 | 图的输入算子 | 板端 `inputs_set` | NPU 内部内存 | 板端帧率 |
|---|---|--:|--:|--:|
| **toolkit 全图 int8**（`p0_off_A_normal` / `a2f_qatw_toolkit_int8`） | **INT8**（1 B/px） | **0.34 ms** | **20,400 KB** | **13.16 fps**（001324 2000 帧 152.0 s） |
| `--no-quant`（fp16 模型） | **FLOAT16**（2 B/px） | 45.41 ms | 105,600 KB | 6.48 fps（308.6 s） |
| `quantized_hybrid_level=1/2/3` | **FLOAT16** | 45.41 ms | 105,600 KB | — |
| **QDQ ONNX + `build(do_quantization=False)`**（QAT 的 A0/A1/A2/A1b） | **FLOAT16** | 45.41 ms | 105,600 KB | 5.64 fps |

- **根因**（`archive/02` §28 已记，本轮 2.3.2 复验）：toolkit 把「**图的输入张量 dtype**」与「**计算精度**」绑死
  ⇒ `--no-quant` ⇒ 输入 FLOAT16；全图量化 ⇒ 输入 INT8；混合 ⇒ 它选中间态，把**输入**判成 float16 而只量化部分中间层
  ⇒ **没有开关能让输入停在 int8 而图内混精度**。
- **含义**：**混合精度 / 混合 QAT 产物即使精度更好，也不是延时意义上的 int8 模型**
  —— int8 相对 fp16 的核心延时优势就是 `inputs_set` **0.34 ms vs 45.41 ms**（≈134×）。
- ⭐ **零板卡成本的判定方法（2026-10-23 新做法）**：构建时开 toolkit verbose，
  构建日志的算子表里有 `D RKNN: … InputOperator  <dtype>  CPU  (1,3,640,640)` 一行，**直接 grep 即可**：
  ```
  D RKNN: [10:15:43.379] 0    InputOperator      INT8     CPU    \   (1,3,640,640)   InputOperator:images
  D RKNN: [10:21:27.381] 0    InputOperator      FLOAT16  CPU    \   (1,3,640,640)   InputOperator:images
  ```
  `02_batch_ptq_pc_screening.py` 已实现（`--verbose-build` + 报告字段 `input_op_dtype`）。
  ⚠️ **`verbose=False` 时该表不打印**（实测 0 次命中）⇒ 不开 verbose 就判不了输入 dtype。
- 板端帧率出处：同一 manifest `001324_sub2000`、同批 BMP、同一脚本 `04_run_board_fast_eval.py`，
  产物 `/mnt/data/siping/ptq_work/logs/tier2_fixed.log`：**int8 2000 帧 152.0 s = 13.16 fps** vs **fp16 2000 帧 308.6 s = 6.48 fps**（int8 **2.03×**）。

---

## 21. ⭐ QAT 路线的最终判定（**负结果**，2026-10-23）

> **完整正文见 `manu/memory/QAT_RK3588_交付报告.md`**（QAT worker 产出）；本节只给结论与对本报告的影响，不复制全文。
> 服务器侧产物：`/mnt/data/siping/ptq_work/qat/trial_0080/report/qat_final_report.{md,json}`、`FINAL_TABLES.txt`。

**在 rk3588 + toolkit 2.3.2 上，QAT 不能兼得「比 int8 基线更准」与「INT8 输入的延时」。**

- **三轨对照（板端，同口径）**：
  T3 `001324_sub2000` int8 基线 **0.7090@0.45**（量化损失 0.0247）**胜** A2 **0.6863@0.45**（0.0473），差 **0.0227**；
  T2 黄金池 **A2 0.9509@0.25**（0.0446）**胜** int8 基线 0.9428@0.35（0.0527），领先 0.0081；T1 tier1/100 **打平**（都 0.8352）。
- **阈值重标定对基线的收益（0.6448→0.7090 = +0.0642）是对 A2 收益（0.6552→0.6863 = +0.0311）的两倍。**
- **`a2f_qatw_toolkit_int8`（QAT 权重走 toolkit 原生全 int8）是唯一 InputOperator=INT8 的 QAT 产物**
  （149 算子全 INT8、算子表与基线逐项相同），但**精度爆炸**：板端 @0.30 FP T1 20→127 / T2 68→700 / T3 527→4433。
- **a2f 爆炸的四刀定位（根因＝权重/训练问题，不是量化问题）**：
  1. ORT 跑 **float** ONNX 已爆：检出比 QAT/base = tier1 ×4.12、黄金池 ×2.94、001324 **×6.55**（7847→51373）；
     001324 @0.30 FP 171→**2582**（×15.1）、F1 0.7484→**0.3671** ⇒ **排除「量化范围被离群值撑爆」**；
  2. 两张 ONNX 算子直方图一致（Conv 85 / Mul 93 / Sigmoid 77 / Softmax 1 / MaxPool 3 / Resize 5…），仅 Constant 64→61 ⇒ **排除「导错图」**；
  3. RKNN 算子表逐项相同（149 算子全 INT8、输入 INT8）⇒ **排除「构建/导出 bug」**；
  4. 权重级 diff：494 张量 **0 个 key/shape 问题**、66 个冻结未动；
     **`head.heatmap.0.weight` rel_l2=0.266、均值 +0.004326→+0.006635（+53%）**
     ⇒ **根因＝QAT 把 heatmap 头搬坏**（logit 正偏置 → sigmoid 饱和 → 过自信 → 固定阈值下 FP 爆炸）。
- **训练损失变好但部署指标变差**：`train_loss` 0.3487→**0.3243**，而 F1@0.30 0.7817→0.5304
  ⇒ focal+offset 不约束热图绝对标度 ⇒ **训练损失不得作为 QAT 有效的证据**。
- **另一条负面结论**：**QAT checkpoint 与它学到的 clip 绑死** —— QAT 权重单独跑 fp16 得 0.3595，
  交给 toolkit 自己的全 int8 标定得 0.2574，**只有连同 `*.qdq_meta.json` 的 scale 一起发布（A2）才可用**。

### 21.4 部署建议与下一步

- **部署建议不变：维持 toolkit 原生 int8 + 逐轨阈值重标定**（T3 **0.45** / T2 **0.30** / T1 **0.50**）。
- **下一步最高价值**：把 LSQ 从 **3 个头部位点**扩到**全网络**（约 **85 个 conv**），让 Q/DQ 覆盖主体，
  `build(do_quantization=False)` 才能**既保住 INT8 输入又不让主体退化成 fp16**。
- ⏳ **未完成**：QAT 臂的**全量集合**（T3 11810 / T2 17187 / T1 anti 31613）没跑（板端 ≈5.8 h，板卡被别的 harness 占用）。

---

## 22. 🔴 本轮对既有记忆的**两处更正**

1. **`mean_values`/`std_values` 必须给** —— 与旧记录「Q/DQ ONNX 不得传 mean/std」**相反**（更正 §6 与 `archive/02`）。
   实测：不传 mean/std 时，板端驱动上传的 uint8 缓冲区被 runtime 直接加宽成 FLOAT16（0..255），图里没有 `/255`
   ⇒ 网络看到 **255 倍大**的输入 ⇒ 饱和 ⇒ **null control 直接判死**（`FATAL: NPU output independent of input`）；
   给了 `mean=[[0,0,0]]` / `std=[[255,255,255]]` 则 RC=0 且 null control 显示 `heatmap CHANGED`。
   **这更正了 `17_probe_qdq_import.py` 探针的一处误读**：探针里 null control 报 `IDENTICAL(BAD)`，
   当时归因为「那帧没无人机」，实际是**缺 mean/std 造成的 255 倍输入错误**；
   探针的 `QDQ_ADOPTED` 结论**仍成立**（它读的是板端 `out[N]` 的 zp/scale，与输入数值无关），但探针**没测过** mean/std 这条轴。
   ⚠️ 这**不是**「重复归一化」：`mean_values/std_values` 描述的是**输入张量**的量化（uint8 缓冲区 = x/255），
   与 Q/DQ 描述的**激活** clip 是**正交**的两件事。
2. **uint8 图输入的 ONNX 在 build 期被拒**：`Not Support Dtype: 2` / `Missing quantization parameters for images`
   （与 2026-10-16 在 2.3.0 上的记录一致，**2.3.2 依旧**）。
   `22_export_qdq_onnx.py` 新增 `--uint8-input`（默认 **0**，保留 FLOAT 输入）；uint8 版本仍可导出，仅作留档。

---

## 23. 环境事实（2026-10-23 下午）

1. **工具链 build 不是 bit-reproducible**：同一配方三次构建得到
   `75d84fcd47d9a081da8790ce6c41e3ae`（`models/p0_off_A_normal_rebuild.rknn`）/
   `1de9cb3b16d8fa1312914de81b985408`（`sim25/smoke_base/rknn/p0_off_A_normal.rknn`）/
   `cedaeb74d066a64ba2c82003993defb1`（`models/p0_off_A_normal.rknn`）**三个不同 md5**，
   但**数值完全一致**（100 帧先验逐项复现：base @0.30 TP 77 / FP 17 / F1 0.7299）
   ⇒ **引用 `.rknn` 时必须带 md5，但数值结论不受影响**。
   ⚠️ 副作用：PC 模拟器若 build 到**与板端同一个目录**，板端驱动会报 `model md5 MISMATCH` 并中止整轮评测
   ⇒ 模拟器的 build 输出必须落到独立目录（如 `pc_sim_rknn/`）。
2. **板卡 `/mnt/manu` 的 NFS 挂载不随工作站重启存活**（板端**无 fstab 项**）⇒ 每次工作站重启都要手动：
   ```
   mount -t nfs 192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu -o nolock,soft,timeo=5,retrans=2
   ```

---

## 24. th=0.25 与历史 trial-0474 的同阈对齐（**子集口径**，未闭合）

- 板端全量 TSV 是在 **th=0.30** 采的 ⇒ **th=0.25 无法从现有 TSV 重算**（0.25~0.30 段分数没采到）。
- PC 模拟器（`25_pc_sim_threshold_sweep.py`，collect-th=0.05）在 Anti-UAV 轨按序列均匀抽样
  **2,493 帧（31,613 帧的 7.9%，24 序列，1,982 GT）**：

| th | int8 F1（模拟器） | PT float F1（模拟器） | Δ |
|---|---|---|---|
| 0.20 | 0.8506 | 0.8608 | −0.0103 |
| **0.25** | **0.8590** | **0.8732** | **−0.0142** |
| 0.30 | 0.8631 | 0.8836 | −0.0204 |

- 历史 trial-0474 **全量** @0.25 同点口径 = **−0.020986**；本轮全量板端 @0.30 = **−0.0189**。
  ⇒ **−0.0142 与 −0.0210 同量级但不相等（差 +0.0068）**。
- 归因三选一**未拆开**：(a) 子集只占 7.9%（但板端同子集 @0.30 = **0.8606** vs 板端全量 @0.30 = **0.8603**
  ⇒ 采样代表性高，**不是主因**）；(b) 模拟器 vs 真 NPU（ΔF1 ≤ 0.005）；(c) **模型代际差**（trial_0080 vs trial-0474）。
- 🔴 **明确声明**：th=0.25 与历史的**全量**同阈对齐必须由板端以 **collect-th ≤ 0.20** 重采一次全量 TSV
  ⇒ **属板卡资源，不在本轮范围**。
- 产物：`/mnt/data/siping/ptq_work/sim25/pt_anti/`、`sim25/base_anti/`（各 `anti_uav_sub2500_pcsim_threshold_sweep.json`）、
  `sim25/board_anti_subset_curve.json`（板端同子集曲线）。


---

## 19. 🔴🔴🔴 2026-10-23 最重要的方法论教训：**子集不能代表全量轨**（子集口径的收益全是假象）

### 19.1 事故经过

1. 板卡筛选跑在**子集**上（T3 2000 帧 / T2 1200 帧 / T1 2500 帧）：
   `set_6`（双域+峰值覆盖，无泄漏）在 T3 子集上 @0.30 是 **0.6585** vs 基线 **0.6448** ⇒ **+0.0137**；
   T2 +0.0199；T1 +0.0003 ⇒ 三轨均值 **+0.0113**，看起来是明确的胜利。
2. 随后 `set_6` 跑了**全量 60,610 帧**（123/123 块，0 错误），并用全量清单重扫：
   **T3 0.7517@0.55、T2 0.9570@0.55、T1 0.8608@0.40** ⇒ vs 基线 **0.7581 / 0.9570 / 0.8606**
   ⇒ **T3 −0.0064、T2 ±0.0000、T1 +0.0002** ⇒ **校准集轴的净收益归零**。

### 19.2 为什么会差这么多：**扫描脚本按 manifest 的行数打分**

`19_board_threshold_sweep.py --tsv-dir` 是**拿 manifest 的 rows 去 TSV 里查检测**，
⇒ 用 `fast/001324_sub2000.json`（2000 行）去扫**全量 11,810 帧的 TSV** ⇒ 得到的是**子集分数**，
但它长得像"全量 TSV 的分数" ⇒ **两列口径不同却并排放在一张表里**。
⇒ 同一天里，"set_6 的 T3 = 0.6915"这个数就是这么来的（子集 manifest 打分）。

### 19.3 三条硬规矩（写进流程）

| # | 规矩 | 落地 |
|:--|:--|:--|
| 1 | **任何跨轨/跨臂比较，必须用同一套 manifest** | 比较前先确认两边的 manifest 行数一致 |
| 2 | **"全量 TSV + 子集 manifest" ≠ 全量分数** | 要全量分数就用全量 manifest（`30_make_full_manifests.py` 已生成：11,810 / 17,187 / 31,613） |
| 3 | **子集只能做"同帧筛选"（同一批帧上比多个臂），不能做"结论"** | 子集差异 < 0.02 一律视为噪声；只有 |Δ| 很大（如 auto_hybrid 的 −0.107）才值得全量复核 |

### 19.4 连带作废/修正

| 结论 | 判决 |
|:--|:--|
| 「set_6 比基线好 +0.0113（三轨）」 | ❌ **作废**（子集口径）；全量口径 **T3 −0.0064 / T2 ±0 / T1 +0.0002** |
| 「set_4 / set_5 的 +0.02 级优势」 | ⚠️ **只在子集成立**，未在全量验证 ⇒ 按 §19.3 视为未证实 |
| 「set_3 与 set_A 在板上完全相同」（另一个会话，2000 帧子集） | ⚠️ 结论方向与全量一致（等效），但数字只代表子集 |
| 基线的全量三轨最优 F1（0.8606 / 0.9570 / 0.7581） | ✅ **经全量清单重扫复核，数字有效**（与原记录一致） |


---

## 20. ⭐ 2026-10-23 **PTQ 轴最终结论表**（RK3588 / int8 / 全量口径）

> 执行机：服务器 `192.168.99.40` + 板卡 `192.168.0.64`；
> 口径：三个 val 轨全量（**11,810 / 17,187 / 31,613 = 60,610 帧**，123/123 块，0 失败），
> 阈值扫描用**全量 manifest**（`30_make_full_manifests.py`）。

### 20.1 旋钮逐条判决

| 旋钮 | 判决 | 实测证据 |
|:--|:--|:--|
| `quantized_algorithm='normal'` | ✅ **唯一可用**（现任） | 全量三轨 **T1 0.8608 / T2 0.9570 / T3 0.7517**（基线 set_A：0.8606 / 0.9570 / 0.7581） |
| `quantized_algorithm='kl_divergence'` | ☠️ **死** | 端到端 F1 **0.0000**（A/B/C/D 四集全灭）；cos_hm 0.84–0.85 vs normal 0.973 ⇒ **模型本身崩了，不是阈值问题** |
| `quantized_algorithm='mmse'` | ⏱ **不可行** | toolkit 逐层优化，ETA **4 h 08 m**/臂 |
| `quantized_method='channel'` | ✅ 最优（现任） | set_3 上 0.8666 vs `layer` 0.8571 |
| `quantized_method='layer'` | ❌ 更差 | 同上 |
| `quantized_method='group{32..256}'` | 🚫 平台不支持 | 报 `only support when quantized_dtype='w4a16'`，rk3588 不支持 w4a16 |
| `quantized_hybrid_level=1` | 🚫 **时延出局**（硬闸门） | 输入算子 **FLOAT16** ⇒ `inputs_set` 45 ms（int8 是 0.34 ms） |
| **`auto_hybrid`（2.3.2 新特性）** | 🚫 **精度出局** | **输入算子 INT8 ✔ 合规**（与 hybrid_level 不同！），但三轨子集均值 **0.8095** vs set_6 的 0.8384；T1 **0.8279 vs 0.8609** ⇒ **差 0.033–0.067，超出噪声** |
| `optimization_level` 0/2/3 | ➖ 无差异 | 三臂 F1 / cos_hm 完全相同 |
| `enable_flash_attention` | ➖ 无差异 | 同上（且此前在 set_A 上亦无效） |
| `compress_weight` | ➖ 无差异 | 与基线**逐位相同** |
| **校准清单顺序** | ➖ 无影响 | 同一批帧打乱顺序重建 ⇒ 指标**逐位相同** ⇒ 此前"同配方重建 md5 不同"只是**构建非确定性**，不影响精度 |
| **校准集规模** 100/150/300 | ➖ 不敏感 | F1 0.6968/0.7512/0.7512（子集）⇒ 150 帧已够 |
| **校准集内容**（唯一有信号） | ⚠️ **全量口径归零** | 子集 +0.0113（set_6）→ **全量 T3 −0.0064 / T2 ±0.0000 / T1 +0.0002** ⇒ **无净收益**；机理：量化范围是一维取舍（宽↔窄） |
| └ 业务域定向集 set_8（纯龙泉山 official train） | ❌ 也不行 | **全量 T3（11,810 帧）0.7494@0.50 vs 基线 0.7581@0.55 ⇒ −0.0087**（子集上它 T1 最好 0.8682，但 T3/T2 都差） |
| 校准集**泄漏** | ☠️ **红线** | set_1 **250/250 帧是评测帧**、set_3 67 帧 ⇒ v2 家族 T2/T3 数字**作废** |
| `bias_correction` / `act_equalization` | ⏳ **技术可行、本轮未试** | toolkit 2.3.2 的 `.so` 可直接 import；入口已定位（`rknn.rknn_base.ir` / `.quant_tab` + `BC.Session` / `AE.Act_Equalization`） |
| QAT（只学 scale、冻结权重） | 🚫 **被闸门挡住** | toolkit 2.3.2 对 **uint8 图输入** `abort rc=134 Not Support Dtype: 2`；QDQ 臂输入算子实测 **FLOAT16** ⇒ 违反 25 fps 硬闸门 |

### 20.2 PTQ 之后剩下的差距（int8 vs PT float，各自最优阈值）

| 轨 | PT float | int8（最好） | 缺口 |
|:--|--:|--:|--:|
| T1 Anti-UAV | 0.8839@0.35 | 0.8608@0.40 | **−0.023** |
| T2 黄金池 | 0.9971@0.65 | 0.9570@0.55 | **−0.040** |
| T3 001324 | 0.7950@0.50 | 0.7581@0.55 | **−0.037** |

⇒ 这 0.023–0.040 **不是校准能补的**（校准轴已到底）⇒ 只能靠**量化感知的训练**（LSQ/只学 scale），
⇒ 而它要求 **INT8 图输入**，toolkit 2.3.2 做不到 ⇒ **两条出路**：
1. **换 toolkit 版本**（2.5.0 会崩 SDPA，但 `enable_flash_attention=False` 可绕；2.5.0 是否修好 uint8 图输入未测）；
2. **接受 int8 现状**，把交付精度交给**系统级后处理**（业务域真正的精度杠杆：去 P0/时空滤波/去重/逐轨阈值）。

### 20.3 交付建议（当前最优解）

**模型 = `p0_off_A_normal`（set_A，现任基线，md5 `cedaeb74d066a64ba2c82003993dfb1`，输入 INT8、`inputs_set` 0.34 ms）**
**逐轨部署阈值 = T1 0.35 / T2 0.55 / T3 0.55** ⇒ 全量三轨 **0.8606 / 0.9570 / 0.7581**。

### 20.4 今天新增的工具（仓库内，三侧 md5 一致）

| 工具 | 作用 |
|:--|:--|
| `26_board_screening_round.sh` | 板卡筛选一轮（锁目录互斥 ⇒ 不用 `ps|grep` 自匹配；等板卡空闲；两/三轨跑 + 排名表） |
| `26_screening_round.sh` | 服务器一轮筛选（构建 + PC 模拟器 100 帧）；臂串格式 `set:quant_method:algo` |
| `27_check_calib_leak.py` | 校准集泄漏自检（路径 + 序列双查） |
| `28_make_calib_strong_train.py` | 无泄漏强目标校准集（train 侧 + ONNX Runtime 探 float 峰值 + top-N + 峰值缓存） |
| `29_finalize_round.sh` | 收尾链：等全量跑 → 全量清单阈值扫描 → 上板筛候选 → 汇总表 |
| `30_make_full_manifests.py` | **全量轨清单**（11,810 / 17,187 / 31,613）⇒ 修掉"子集 manifest 打分全量 TSV"的口径错误 |


---

## 21. 🔴🔴🔴 2026-10-23 晚：**0.8977 与 0.7950 的 0.10 差 = 「多尺度」vs「单尺度」**（不是 P0、不是量化）

> 起因：用户质疑「微调后的模型是 ~0.898，你用的 0.7950 是什么？」⇒ 逐层核查后定位。

### 21.1 三方都不是：臂相同、模型相同、特征逐像素相同

| 核查项 | 结论 | 证据 |
|:--|:--|:--|
| 臂 | **都是 p0_off** | 门禁脚本三轨都传 `--ablate-p0`（`auto_epoch_gate_check.py:242/274/396`）；门禁基线注释也写"去 P0 口径" |
| 模型 | **都是 trial_0080** | 门禁候选臂 `finetune_peak` = 0.901910；全量 PT 臂加载 `trials/trial_0080/weights/epoch3.pt`（`ablate_p0=True`） |
| 特征语义 | **都是 official** | 全量 PT 臂的 BMP 与 `longquanshan_ft_gmc_median_official` 的同帧 **逐像素完全相同**（mean=0.00, max=0）；与 FPV 树差 mean=0.72/max=63 |
| GT | 相同 | 两边 T3 GT 分母都是 10,535、11,810 帧 |
| **输入尺度** | 🔴 **不同！** | 门禁走 `multiscale_heatmap_cache.py --scales 160,80` ⇒ **多尺度（640+160+80，逐像素 `np.maximum` 融合）**；上板与全量 PT 臂是**单尺度 ONNX（只跑 640×640 一次）** |

⇒ **0.8977（多尺度 float）− 0.7950（单尺度 float）= 0.1027，就是"多尺度 vs 单尺度"。**

### 21.2 三段口径的正确拆解（覆盖之前两版结论）

| 段 | T1 | T2 | T3 | 说明 |
|:--|--:|--:|--:|:--|
| ① 多尺度 float（门禁/候选表/业务交付） | 0.9019 | 0.9989 | **0.8977** | 业务交付 94.747% 系统级 F1 就建在这一层 |
| ② 单尺度 float（= 上板图的架构能力） | 0.8791 | 0.9952 | **0.7950** | 去掉多尺度要付 **0.103（T3）** |
| ③ 单尺度 **int8**（当前 NPU 部署形态） | 0.8603 | 0.9558 | **0.7581** | 量化再付 0.019~0.039 |

⇒ **真正的精度杠杆排序**：
1. **多尺度 → 单尺度：−0.103（T3）** ← 比整个 PTQ 轴（−0.02~−0.04）**大 3 倍**；
2. 量化（float→int8）：−0.019 / −0.039 / −0.037；
3. 校准集：全量口径 **±0.000**（已探到底）。

### 21.3 两条作废的旧结论（本日实测推翻）

| 旧结论 | 作废理由 |
|:--|:--|
| 「砍掉 P0 highway 掉 0.103」 | ❌ 那是**多尺度→单尺度**的差。**P0 在 official 语义 T1 上只值 0.0004**（0474 四臂：带P0 0.902316 / 去P0 0.901935）；FPV 口径的 int8 子集上 p0_on 0.6999 **<** p0_off 0.7090 ⇒ **P0 无收益** |
| 「带 P0 的 int8 板端 0 检出」 | ❌ **板端实测完全正常**（INT8 输入 ✔、null control ✔、12/12 帧有检出、坐标与基线一致）⇒ 那是 10-22 那次选型跑的假象（板端上下文污染） |

### 21.4 由此产生的方向性结论（对"要不要重训"的回答）

**要重训，但重训的目标不是修量化，而是"把多尺度折叠进单尺度"**：

- 业务侧的精度是**多尺度**给的（0.8977 / 0.9989），而 NPU 移植线目前只能跑**单尺度**（0.7581 / 0.9558）⇒
  ⇒ **NPU 线与业务交付之间的 0.10 差距，根因是"多尺度没上板"，不是量化。**
- 两条可行路径（成本差异巨大）：
  1. **把多尺度做进 NPU 的一次前向**：多尺度分支（640/160/80）作为**一个图里的三条支路**共享一次 `rknn_run`
     ⇒ 计算量 ×3 但**提交/调度开销 ×1** ⇒ 对 25 fps 比"3 次串行前向"友好得多（3 次串行 = 3× 的 102 ms ⇒ ~3.5 fps）；
  2. **重训/蒸馏一个"单尺度等价"模型**：训练时用多尺度 teacher 监督单尺度 student，
     ⇒ 部署形态不变（单尺度 int8，`inputs_set` 0.34 ms）⇒ **最省板卡**；
     ⇒ 代价：要重训，且要验证"单尺度能追到多尺度的多少"。
- ⇒ **建议的优先级**：PTQ 已到底（收益 ±0.00），**下一个战场是"多尺度"**，不是量化。
