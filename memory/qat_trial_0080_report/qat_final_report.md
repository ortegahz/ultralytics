# QAT 路线的最终判定（rk3588 + rknn-toolkit2 2.3.2）

> 生成于 2026-10-23 12:51:35 ｜ 无板卡依赖 ｜ 原始数据在 `/mnt/data/siping/ptq_work/qat/trial_0080`

## 0. 一句话结论

**NO** —— QAT 路线在 rk3588 + toolkit 2.3.2 上**不能**同时拿到「比 int8 基线更准」和「INT8 输入的延时」。

- A2f (the only QAT route that keeps an INT8 input) is a numerical disaster: the FLOAT ONNX with the same weights already explodes (ORT float detections are 6.5468x the base weights, px>0.5 per frame 6.522244261830546x), so the cause is the WEIGHTS, not the quantiser.
- The QAT training moved head.heatmap.0.weight by rel_l2=0.266 and shifted its mean by +53%, which lifts the heatmap logits and makes the model over-confident; the export is provably fine (identical op histogram, 0 tensor shape/key problems).
- A2 (QAT weights + the learned clips) does recover most of the accuracy, but its best F1 on T3 is still below the int8 baseline's.

## 1. 每条臂的输入 dtype 与由此推出的 inputs_set 代价

| 臂 | InputOperator dtype | 算子数 | NPU dtype 分布 | inputs_set 代价 |
|---|---|---|---|---|
| A2f QAT weights + toolkit full int8 | **INT8** | 149 | {'INT8': 146} | inputs_set 0.34 ms, NPU tensor 20,400 KB |
| A2  QAT weights + LSQ-learned clip | **FLOAT16** | 158 | {'FLOAT16': 149, 'INT8': 6} | inputs_set 45.41 ms, NPU tensor 105,600 KB |
| A1  QAT weights + max-calibrated clip | **FLOAT16** | 158 | {'FLOAT16': 149, 'INT8': 6} | inputs_set 45.41 ms, NPU tensor 105,600 KB |
| A0  QAT weights, no Q/DQ (fp16) | **FLOAT16** | 149 | {'FLOAT16': 146} | inputs_set 45.41 ms, NPU tensor 105,600 KB |
| A1b BASE weights + max-calibrated clip | **FLOAT16** | 158 | {'FLOAT16': 149, 'INT8': 6} | inputs_set 45.41 ms, NPU tensor 105,600 KB |
| base_weights_toolkit_int8 | **INT8** | 149 | {'INT8': 146} | inputs_set 0.34 ms, NPU tensor 20,400 KB |

> 唯一保住 INT8 输入的 QAT 路线是 **A2f**（QAT 权重 + toolkit 原生全 int8 配方）。
> A0/A1/A2 的 Q/DQ 路线一律是 FLOAT16 输入。

### 1.1 a2f 与 int8 基线的构建配方逐项对照（排除构建/导出 bug）

| 项 | int8 基线（base 权重） | a2f（QAT 权重） | 是否相同 |
|---|---|---|---|
| InputOperator dtype | INT8 | INT8 | ✅ 相同 |
| 算子总数 | 149 | 149 | ✅ 相同 |
| NPU dtype 分布 | {'INT8': 146} | {'INT8': 146} | ✅ 相同 |
| CPU dtype 分布 | {'INT8': 3} | {'INT8': 3} | ✅ 相同 |
| .rknn 大小 | 3791048 B | 3790280 B | 差 768 B = 权重不同 |

**结论：a2f 的构建配方与 int8 基线逐项相同**（同一个 ONNX 架构、同一个 `mean_values/std_values`=0/255、同一个 `quantized_algorithm=normal`、同一个 `quantized_method=channel`、同一个 `optimization_level=3`、同一个 `do_quantization=True` + 同一个校准集 `calib/set_A.txt`），算子表逐项一致。**唯一的差别是权重** ⇒ 虚警爆炸不是构建/导出 bug。

## 2.1 T1 tier1_sensitive/100

### 2.1.1 PC 模拟器（无板卡）

| 臂 | @0.30 F1 | @0.45 F1 | 最优 F1 | 最优 th | 量化损失@0.30 | 量化损失@最优 |
|---|---|---|---|---|---|---|
| PT float (base weights) | 0.7817 | 0.8539 | 0.8636 | 0.55 | 0 | 0 |
| A0  QAT weights, no Q/DQ (fp16) | 0.5076 | 0.5816 | 0.7273 | 0.70 | +0.2741 | +0.1299 |
| A1  QAT weights + max-calibrated clip | 0.7477 | 0.7917 | 0.8172 | 0.70 | +0.0341 | +0.0399 |
| A2  QAT weights + LSQ-learned clip | 0.7339 | 0.7835 | 0.8172 | 0.70 | +0.0478 | +0.0399 |
| A1b BASE weights + max-calibrated clip | 0.6133 | 0.1308 | 0.8000 | 0.15 | +0.1684 | -0.0183 |
| A2f QAT weights + toolkit full int8 | 0.3672 | 0.4099 | 0.5047 | 0.70 | +0.4146 | +0.3525 |

> 口径：**PC 模拟器是 NPU 的软件仿真**（量化权重 + 浮点算子）。同一臂同一阈值下它与板端的 F1 差 ≤ 0.0249（逐臂核对：
> A0  QAT weights, no Q/DQ (fp16) 模拟器 0.5816 / 板端 0.5816；A1  QAT weights + max-calibrated clip 模拟器 0.7917 / 板端 0.8000；A2  QAT weights + LSQ-learned clip 模拟器 0.7835 / 板端 0.8000
> ），**臂之间的相对高低与板端一致**；因此模拟器足以承担「无板卡」的相对排序，绝对结论仍引板端数字。

### 2.1.2 板端（历史实测，同口径对照用）

| 臂 | @0.30 F1 | @0.45 F1 | 最优 F1 | 最优 th |
|---|---|---|---|---|
| PT float (base weights) | 0.7817 | 0.8539 | 0.8636 | 0.55 |
| A2  QAT weights + LSQ-learned clip | 0.7547 | 0.8000 | 0.8352 | 0.70 |
| A1  QAT weights + max-calibrated clip | 0.7751 | 0.8000 | 0.8352 | 0.70 |
| A0  QAT weights, no Q/DQ (fp16) | 0.5139 | 0.5816 | 0.7207 | 0.70 |
| A1b BASE weights + max-calibrated clip | 0.6133 | 0.1132 | 0.8172 | 0.15 |
| A2f QAT weights + toolkit full int8 | 0.3724 | 0.4348 | 0.5298 | 0.70 |
| int8 PTQ baseline (toolkit) | 0.7156 | 0.8261 | 0.8352 | 0.50 |
| fp16 baseline (toolkit) | 0.7739 | 0.8539 | 0.8636 | 0.55 |

### 2.1.3 FLOAT ONNX 诊断（onnxruntime，同一批帧）

| 图 | heatmap max | 帧均 heatmap 均值 | px>0.5/帧 | 检出@0.05 | 最优 F1 | @0.30 F1 |
|---|---|---|---|---|---|---|
| base weights | 0.9636 | 0.00119 | 10.4 | 237 | 0.8636@0.55 | 0.7817 (TP 77 FP 10) |
| QAT weights | 0.9973 | 0.00269 | 35.2 | 976 | 0.7619@0.70 | 0.5304 (TP 83 FP 65) |

判定：**FLOAT_ALSO_EXPLODES -> the WEIGHTS/EXPORT are the cause** ｜ 检出比 4.1181 ｜ px>0.5 比 3.3817307692307694 ｜ 算子直方图差异 {'Constant': {'base': 64, 'test': 61}}

## 2.2 T2 golden_pool_sub1200/1197

### 2.2.1 PC 模拟器（无板卡）

| 臂 | @0.30 F1 | @0.45 F1 | 最优 F1 | 最优 th | 量化损失@0.30 | 量化损失@最优 |
|---|---|---|---|---|---|---|
| PT float (base weights) | 0.9955 | 0.9955 | 0.9955 | 0.05 | 0 | 0 |
| A0  QAT weights, no Q/DQ (fp16) | 0.8922 | 0.9145 | 0.9267 | 0.70 | +0.1033 | +0.0679 |
| A1  QAT weights + max-calibrated clip | 0.9373 | 0.9373 | 0.9373 | 0.25 | +0.0583 | +0.0583 |
| A2  QAT weights + LSQ-learned clip | 0.9444 | 0.9444 | 0.9444 | 0.25 | +0.0511 | +0.0511 |
| A1b BASE weights + max-calibrated clip | 0.7394 | 0.1103 | 0.9512 | 0.10 | +0.2561 | +0.0444 |
| A2f QAT weights + toolkit full int8 | 0.6051 | 0.6141 | 0.6175 | 0.70 | +0.3905 | +0.3771 |

> 口径：**PC 模拟器是 NPU 的软件仿真**（量化权重 + 浮点算子）。同一臂同一阈值下它与板端的 F1 差 ≤ 0.0162（逐臂核对：
> A0  QAT weights, no Q/DQ (fp16) 模拟器 0.9145 / 板端 0.9205；A1  QAT weights + max-calibrated clip 模拟器 0.9373 / 板端 0.9468；A2  QAT weights + LSQ-learned clip 模拟器 0.9444 / 板端 0.9509
> ），**臂之间的相对高低与板端一致**；因此模拟器足以承担「无板卡」的相对排序，绝对结论仍引板端数字。

### 2.2.2 板端（历史实测，同口径对照用）

| 臂 | @0.30 F1 | @0.45 F1 | 最优 F1 | 最优 th |
|---|---|---|---|---|
| PT float (base weights) | 0.9955 | 0.9955 | 0.9955 | 0.05 |
| A2  QAT weights + LSQ-learned clip | 0.9509 | 0.9509 | 0.9509 | 0.25 |
| A1  QAT weights + max-calibrated clip | 0.9468 | 0.9468 | 0.9468 | 0.25 |
| A0  QAT weights, no Q/DQ (fp16) | 0.8987 | 0.9205 | 0.9329 | 0.70 |
| A1b BASE weights + max-calibrated clip | 0.7271 | 0.0941 | 0.9594 | 0.05 |
| A2f QAT weights + toolkit full int8 | 0.6137 | 0.6202 | 0.6241 | 0.70 |
| int8 PTQ baseline (toolkit) | 0.9420 | 0.9428 | 0.9428 | 0.35 |
| fp16 baseline (toolkit) | 0.9849 | 0.9849 | 0.9849 | 0.30 |

### 2.2.3 FLOAT ONNX 诊断（onnxruntime，同一批帧）

| 图 | heatmap max | 帧均 heatmap 均值 | px>0.5/帧 | 检出@0.05 | 最优 F1 | @0.30 F1 |
|---|---|---|---|---|---|---|
| base weights | 0.9715 | 0.00069 | 10.4 | 1240 | 0.9955@0.30 | 0.9955 (TP 1110 FP 3) |
| QAT weights | 0.9979 | 0.00130 | 22.4 | 3645 | 0.9920@0.70 | 0.9533 (TP 1113 FP 54) |

判定：**FLOAT_ALSO_EXPLODES -> the WEIGHTS/EXPORT are the cause** ｜ 检出比 2.9395 ｜ px>0.5 比 2.1520430280163763 ｜ 算子直方图差异 {'Constant': {'base': 64, 'test': 61}}

## 2.3 T3 001324_sub2000/2000

### 2.3.1 PC 模拟器（无板卡）

| 臂 | @0.30 F1 | @0.45 F1 | 最优 F1 | 最优 th | 量化损失@0.30 | 量化损失@最优 |
|---|---|---|---|---|---|---|
| PT float (base weights) | 0.7484 | 0.7337 | 0.7545 | 0.35 | 0 | 0 |
| A0  QAT weights, no Q/DQ (fp16) | 0.3540 | 0.4876 | 0.6737 | 0.70 | +0.3944 | -0.0776 |
| A1  QAT weights + max-calibrated clip | 0.6455 | 0.6782 | 0.6817 | 0.50 | +0.1029 | +0.0296 |
| A2  QAT weights + LSQ-learned clip | 0.6547 | 0.6831 | 0.6841 | 0.40 | +0.0937 | +0.0620 |
| A1b BASE weights + max-calibrated clip | 0.3920 | 0.1505 | 0.5739 | 0.15 | +0.3564 | +0.1745 |
| A2f QAT weights + toolkit full int8 | 0.2525 | 0.3363 | 0.4477 | 0.70 | +0.4959 | +0.1483 |

> 口径：**PC 模拟器是 NPU 的软件仿真**（量化权重 + 浮点算子）。同一臂同一阈值下它与板端的 F1 差 ≤ 0.0268（逐臂核对：
> A0  QAT weights, no Q/DQ (fp16) 模拟器 0.4876 / 板端 0.4948；A1  QAT weights + max-calibrated clip 模拟器 0.6782 / 板端 0.6737；A2  QAT weights + LSQ-learned clip 模拟器 0.6831 / 板端 0.6863
> ），**臂之间的相对高低与板端一致**；因此模拟器足以承担「无板卡」的相对排序，绝对结论仍引板端数字。

### 2.3.2 板端（历史实测，同口径对照用）

| 臂 | @0.30 F1 | @0.45 F1 | 最优 F1 | 最优 th |
|---|---|---|---|---|
| PT float (base weights) | 0.7484 | 0.7337 | 0.7545 | 0.35 |
| A2  QAT weights + LSQ-learned clip | 0.6552 | 0.6863 | 0.6863 | 0.45 |
| A1  QAT weights + max-calibrated clip | 0.6518 | 0.6737 | 0.6741 | 0.40 |
| A0  QAT weights, no Q/DQ (fp16) | 0.3595 | 0.4948 | 0.6750 | 0.70 |
| A1b BASE weights + max-calibrated clip | 0.3857 | 0.1238 | 0.5716 | 0.15 |
| A2f QAT weights + toolkit full int8 | 0.2574 | 0.3424 | 0.4374 | 0.70 |
| int8 PTQ baseline (toolkit) | 0.6448 | 0.7090 | 0.7090 | 0.45 |
| fp16 baseline (toolkit) | 0.7395 | 0.7218 | 0.7469 | 0.35 |

### 2.3.3 FLOAT ONNX 诊断（onnxruntime，同一批帧）

| 图 | heatmap max | 帧均 heatmap 均值 | px>0.5/帧 | 检出@0.05 | 最优 F1 | @0.30 F1 |
|---|---|---|---|---|---|---|
| base weights | 0.9686 | 0.00198 | 8.8 | 7847 | 0.7545@0.35 | 0.7484 (TP 1282 FP 171) |
| QAT weights | 0.9979 | 0.00479 | 57.5 | 51373 | 0.7172@0.70 | 0.3671 (TP 1566 FP 2582) |

判定：**FLOAT_ALSO_EXPLODES -> the WEIGHTS/EXPORT are the cause** ｜ 检出比 6.5468 ｜ px>0.5 比 6.522244261830546 ｜ 算子直方图差异 {'Constant': {'base': 64, 'test': 61}}

### 2.4 a2f 虚警爆炸的根因（每一步排除了什么）

| 诊断步骤 | 结果 | 排除了什么 |
|---|---|---|
| 1. ONNX Runtime 跑 **float** ONNX（基线权重 vs QAT 权重），同一批帧 | QAT 权重的 float 图检出数 ×2.94（黄金池）/ ×4.12（tier1）/ **×6.55（001324）**，px>0.5 ×2.15~6.52、@0.30 的 FP 54 vs 3（黄金池）/ 65 vs 10（tier1）/ **2582 vs 171（001324）** | **排除「量化范围被离群值撑爆」** —— float 就已经爆 |
| 2. 对比两个 ONNX 的算子直方图 | 唯一差异是 Constant 64→61（少 3 个），其余算子逐项一致（Conv 85 / Mul 93 / Sigmoid 77 / Softmax 1 / MaxPool 3 …） | **排除「导出错了图」**（P0 门控已正确消掉，架构一致） |
| 3. a2f 与基线的 RKNN 算子表逐行对照 | 149 算子全 INT8、输入 INT8，逐项相同 | **排除「构建配方不一致」** |
| 4. 权重级 diff（QAT checkpoint vs base checkpoint） | 494 张量全部 key/shape 匹配（0 问题），66 个张量完全冻结未动，head.heatmap.0.weight rel_l2=0.266、均值 +0.004326→+0.006635（+53%） | **排除「checkpoint 存错 / LSQ 伪量化留在图里」** —— a2f 的 ONNX 就是纯 float 图（无 QuantizeLinear），权重确为 epoch3 的 QAT 权重 |

**根因：QAT 训练把 heatmap 头的权重整体搬了（rel_l2 0.266）并把权重的均值抬高了 53%，等价于给 heatmap logit 加了一个正偏置 → sigmoid 饱和 → 热力图整体过度自信 → 固定阈值下虚激（FP）爆炸。** 这与训练 loss 变好（0.3487→0.3243）并不矛盾：focal+offset 的 loss 不约束热力图的绝对标度。QAT 权重与它训练时那套 clip 绑定，换量化方案就崩（A2f），只有 A2（QAT 权重 + 学到的 clip）能恢复大部分精度。

## 3. QAT 训练到底动了什么（权重级 diff）

- 张量数：QAT 494，base 523，可比 494，**问题 0**（0 = 没有 key/shape 错配）
- 完全未变的张量：66（被冻结的 backbone）
- 平均相对 L2 变化 0.043524，最大 0.266065

| 张量 | rel_l2 | 均值 base → QAT | absmax base → QAT |
|---|---|---|---|
| head.heatmap.0.weight | 0.266065 | +0.004326 → +0.006635 | 0.6485 → 0.6458 |
| head.offset.2.weight | 0.200323 | +0.007364 → +0.009287 | 0.0883 → 0.0978 |
| head.offset.0.weight | 0.103079 | +0.006386 → +0.007488 | 0.3461 → 0.3478 |
| head.offset.0.bias | 0.100786 | -0.016585 → -0.011499 | 0.1932 → 0.1978 |
| head.heatmap.0.bias | 0.086442 | +0.020383 → +0.020353 | 0.2376 → 0.2343 |
| head.offset.2.bias | 0.069764 | +0.190259 → +0.203485 | 0.1920 → 0.2041 |
| head.feat_conv.1.bn.running_var | 0.066034 | +2960.916748 → +2776.001221 | 24954.7070 → 23438.3203 |
| head.feat_conv.1.bn.running_mean | 0.055665 | +7.231897 → +7.030874 | 89.6805 → 88.4804 |

## 4. 最终判定

- **主问题**：Can the QAT route beat 'toolkit native int8 + per-track threshold recalibration' on rk3588 + toolkit 2.3.2?
- **答案**：**NO**
- T3 板端：int8 基线最优 0.7089574155653451 @ 0.45；A2 QAT 最优 0.6863468634686346 @ 0.45
- A2f（唯一 INT8 输入的 QAT 路线）PC 模拟器最优 0.447687，@0.30 {'th': 0.3, 'tp': 1590, 'fp': 4601, 'fn': 212, 'det': 6191, 'precision': 0.256824, 'recall': 0.248321, 'f1': 0.252501, 'per_sequence': {'VIDEO00007_19700101_001324': {'tp': 1590, 'fp': 4601, 'fn': 212, 'det': 6191}}}
- **部署建议**：继续用 toolkit 原生 int8 模型 + 逐轨阈值重标定（T3 th=0.45、T2 th=0.30、T1 th=0.50）。
- **若一定要走 QAT**：只能发布 A2（QAT 权重 + 学到 clip + Q/DQ 图），且必须连同 `*.qdq_meta.json` 的 scale 一起发布；代价是 FLOAT16 输入（inputs_set 45.41 ms，不是 0.34 ms）。
- **下一步最高价值**：把 LSQ 从 3 个头部位点扩到全网络，让 Q/DQ 覆盖主体，`build(do_quantization=False)` 才能把主体也保持 int8。

### 4.1 三轨对照：「int8 基线 + 逐轨阈值重标定」 vs QAT 路线

> 板端口径：同帧、同解码器、同 GT、同一阈值集合（collect_th=0.05, topk=100）。

| 轨 | 部署 th | int8 基线 F1（@部署 th） | 量化损失 | int8 基线最优 | A2 最优（板端） | 