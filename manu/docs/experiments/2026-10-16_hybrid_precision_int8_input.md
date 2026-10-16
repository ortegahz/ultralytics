# 混合精度模型（int8 输入 + 敏感层 fp16）可行性调查：RKNN-Toolkit2 2.3.0 上不可达

- 日期：2026-10-16
- 目标：输入保持 int8（`inputs_set` 与 int8 模型一致 ≈ 0.34 ms），对量化损失大的层用 fp16，
  期望 **F1 ≈ 纯 fp16（0.901401 / R 85.81%）** 且 **速度显著优于 fp16**
- 结论：**该组合在 rknn-toolkit2 2.3.0 上构建不出来。** 四条独立路径全部实测封死。
  但"逐层量化损失"本身做出来了，而且它指向的层非常集中。

---

## 1. 起点：两个极端都不可接受

| 模型 | 大小 | 输入 attr | `inputs_set` | `rknn_run` | F1 | R |
|---|--:|---|--:|--:|--:|--:|
| int8 | 3,799,304 B | `type=2` INT8，1228800 B (1 B/px) | **0.34 ms** | 61.56 ms | 0.882231 | 82.90% |
| fp16 | 6,327,393 B | `type=1` FLOAT16，2457600 B (2 B/px) | **45.41 ms** | 111.46 ms | 0.901401 | 85.81% |

`inputs_set` 的 133× 差距来自**输入张量的字节宽度**：int8 图输入原样上传，fp16 图输入要由运行时
在用户态展宽全部 640×640×3 个元素（45.41 ms ÷ 1,228,800 = 37 ns/元素 ≈ 74 cycles）。
所以"输入保持 int8"不是一个偏好，是硬约束。

---

## 2. 逐层量化损失（你要的那部分，做出来了）

### 2.1 toolkit 自己的逐层分析

`hybrid_quantization_step1(proposal=True)` 成功（`proposal_dataset_size=1`），产出
`trial0474_hm_nop0.quantization.cfg`（97,662 B），其中 `custom_quantize_layers` 给出
**76 个应保持 float16 的张量**，每行还带一个逐层分数：

| 最低分（最差） | 张量 | 分数 |
|---|---|--:|
| 1 | `/model/b1/act/Mul_output_0` | 0.977864 |
| 2 | `/model/b0/act/Mul_output_0` | 0.988178 |
| 3 | `/model/Concat_5_output_0` | 0.996347 |
| 4 | `/model/b2/Concat_output_0` | 0.996348 |
| 5 | `/model/b2/Slice_output_0` | 0.996394 |
| 6 | `/model/b2/cv1/act/Mul_output_0` | 0.996704 |
| 7 | `/model/b2/cv2/act/Mul_output_0` | 0.997915 |
| 8 | `/model/head/offset/offset.1/Relu_output_0` | 0.997898 |

分数范围 0.977864–0.999718（有分数的 51 条，中位 0.999273）。
76 条按算子分：Conv 29、Mul 24、Concat 5、Add 4、Slice 3、MaxPool 2、其余 9 条。

### 2.2 我自己独立算的逐层 SQNR（交叉验证）

`analyze_quant_loss_per_layer.py`：把 426 个中间张量全部暴露成图输出，用
onnxruntime 跑 12 张标定图，对每个张量按 toolkit 的配置（激活 per-tensor 非对称仿射 int8）
做 fake-quant，与 fp32 参考比 SQNR / NRMSE。

| 分位 | SQNR |
|---|--:|
| p1 | **27.7 dB** |
| p10 | 34.4 dB |
| 中位 | 40.3 dB |

损失最大的实体张量（非 Constant）：

| SQNR | NRMSE | 张量 |
|--:|--:|---|
| **27.24 dB** | 0.0434 | `/model/b2/cv2/conv/Conv` |
| 28.73 dB | 0.0366 | `/model/b10/m/m.0/attn/Softmax` |
| 29.98 dB | 0.0317 | `/model/head/Clip` |
| 29.98 dB | 0.0317 | `/model/head/Sigmoid` |
| 30.16 dB | 0.0311 | `/model/b2/cv1/conv/Conv` |
| 30.40 dB | 0.0302 | `/model/c19/m.0/cv3/conv/Conv` |
| 30.55 dB | 0.0297 | `/model/b2/m.0/cv2/conv/Conv` |
| 30.97 dB | 0.0283 | `/model/fuse_p2/conv/Conv` |

> 表中排第 1、2 的两条是 `Constant` 节点，`vmin == vmax`，fake-quant 退化导致 NRMSE=1，
> 是度量假象不是真损失，已在分析时识别。

### 2.3 两套方法互相印证

toolkit 的最低分集中在 `/model/b1/act/Mul`、`/model/b0/act/Mul`、`/model/b2/*`；
我独立算的最低 SQNR 也集中在 `/model/b2/*`（27.2–30.6 dB），其次是 attention 的
`Softmax`（28.7 dB）与 head 的 `Sigmoid`/`Clip`（30.0 dB）。

**结论：量化损失高度集中在 `b2`（stem 与颈部的衔接段）**，这与 int8 臂掉召回（82.90% < 85%）
的业务判据失败在方向上一致。若 toolkit 支持按名保 fp16，`b2` + head 输出 + attn Softmax
就是第一优先的保护对象。

---

## 3. 为什么建不出来：四条路逐一实测

### 3.1 两步式混合量化 API —— 通了，且产出真正的混合模型

⚠️ **更正**：本文件初版把 `step2` 的崩溃判成 toolkit 缺陷，**那是我的调用错误**。
`hybrid_quantization_step2(model_input, data_input, model_quantization_cfg)` 的
`model_input` 文档写的是 "Model file path"，必须传 **step1 产出的 `.model` 文件**；
我最初传的是 ONNX，于是走进了另一条代码路径并抛
`KeyError '/model/head/feat_conv/feat_conv.1/act/Sigmoid_output_0'`。

用正确参数重跑：

```sh
python hm_convert_hybrid_rknn.py --mode step2 \
  --onnx  hybrid/trial0474_hm_nop0.model \
  --cfg   hybrid/trial0474_hm_nop0.quantization.cfg \
  --data-input hybrid/trial0474_hm_nop0.data \
  --out   models/trial0474_hm_nop0_hyb_cfg.rknn
```

**rc=0**，产出 `trial0474_hm_nop0_hyb_cfg.rknn`，5,740,974 B，md5 `b3cd7508e27b72ecca0ac7726864d797`。
算子表确认它是真混合：

| 张量 | dtype |
|---|---|
| `model.head.heatmap.0.weight` | **INT8** |
| `model.head.heatmap.2.weight`（ConvSigmoid） | **INT8** |
| `model.head.offset.0.weight` | **FLOAT16** |
| `model.head.offset.2.weight` | **FLOAT16** |

全模型张量构成：**43 INT8 / 30 FLOAT16 / 41 FLOAT**。

`step1` 的 `proposal_dataset_size` 也有讲究：传 50（toolkit 默认 1）会走
`hybrid_proposal.expand_batch` 并在 `infer_shapes` 抛
`KeyError '/model/b10/m/m.0/attn/Split_output_0'`；传 1 则 rc=0。
**批大小 >1 的 proposal 在这个图上是坏的。**

> 但这个混合模型的 `InputOperator` 仍然是 **FLOAT16**（见 §3.2、§5），
> 所以它并没有解决本任务的第一条硬约束。

### 3.2 `quantized_hybrid_level` —— 三个级别全部给出 FLOAT16 输入

走 `config(quantized_hybrid_level=N)` + 普通 `build()`，即产出 int8/fp16 两臂的同一条路径：

| 档 | 大小 | md5 | 输入算子 | NPU 内部内存 | INT8 输出算子 |
|---|--:|---|---|--:|--:|
| L1 | 5,740,974 B | `c43c94a519f2` | **FLOAT16** | 105,600 KB | 1（heatmap） |
| L2 | 5,740,974 B | `2e87dc887760` | **FLOAT16** | 105,600 KB | 1 |
| L3 | 4,483,182 B | `8a347afca515` | **FLOAT16** | 105,600 KB | 1 |

三档的输入算子全是 FLOAT16、内部内存全是 105,600 KB——**和 fp16 模型完全一样**
（int8 只有 20,400 KB）。板上实测也确认了 L1：`input attr: type=1 n_elems=1228800 size=2457600`。

⇒ **toolkit 的混合量化会把输入一起变成 float16，`inputs_set` 的 45 ms 一分不省，
直接违背目标的第一条硬约束。** 这不是配置问题，是三条级别全如此。

### 3.3 UINT8 图输入的 ONNX —— 这是唯一真正的硬墙

`make_uint8_input_onnx.py` 把图输入改成 UINT8 并把 `Cast(u8→f32)` + `Div(/255)`
作为图内前两个节点（这是让"输入 int8、归一化在图内"唯一可行的入口）。

**等价性已逐位验证**：与原 float ONNX 在 3 组随机输入上，所有输出
`max_abs_diff = 0.000e+00` —— 网络数学上完全没变，只是输入管线搬进了图内。

| 阶段 | 配置 | 结果 |
|---|---|---|
| `step1` | proposal, size=1 | **rc=0**，产出 `u8b.quantization.cfg`（97,458 B）——**step1 接受 uint8 图输入** |
| `step2` | `.model`/`.data`/cfg（正确参数） | **abort**：`rknn::RKNNInitRNNConst` → `Not Support Dtype: 2`（dtype 2 = INT8） |
| `build` | 纯 int8 | **abort**，同一处 |
| `build` | `--no-quant` | **abort**，同一处（证明与量化无关） |
| `build` | `quantized_hybrid_level=1` | **abort**，同一处 |

**关键点**：step1 走完了全部标定（12/13 轮）、图融合、quant_optimizer，uint8 图输入全程无事；
**只有最后生成可执行模型的那一步会 abort**。也就是说 toolkit 能"理解"这个图、
能给量化参数，但产不出带 uint8 输入的 .rknn。

（途中修掉两个自己的坑：注入的 `/model/pre/scale` 一开始是 0 维标量，已改为 shape (1)；
以及 `TensorProto.UINT8` 是 **2** 不是 1（1 是 FLOAT），导致自动探测 `mean/std` 的逻辑
静默失效、模型会被除两次 255。两处都已修正并留下注释。）

### 3.4 `quantized_dtype='w8a16'` —— toolkit 自己拒绝

`w8a16`（int8 权重 + int16 激活）本是最有希望的折中（输入可保持 1 B/px），
但 toolkit 直接拒绝：

```
config: The quantized_dtype = 'w8a16' not support in 'rk3588'!
```

（`rknn-toolkit2` 二进制里可选值只有 `w8a8 / w8a16 / w16a16 / w4a8 / w4a16`，
但 rk3588 平台只接受 `w8a8`。）

### 3.5 `custom_string` 混合量化 DSL —— 2.3.0 已移除

在 `rknn/api/*.so` 与 `rknn/api/lib/*.so` 里全量搜 `hybrid_quantization` 关键字，
除 Python 封装的方法名外**没有任何 DSL 字符串**。这是 Toolkit 1.x 的功能，2.3.0 没有。

---

## 4. 汇总：能做出混合模型，但输入必然是 fp16

| 路径 | 混合模型？ | int8 输入？ |
|---|---|---|
| 两步式 `step1` + `step2`（**正确参数**） | ✅ 43 INT8 / 30 FLOAT16 | ❌ `InputOperator FLOAT16` |
| `quantized_hybrid_level=1/2/3` | ✅ | ❌ 三档全 FLOAT16，内部内存 105,600 KB（= fp16） |
| UINT8 图输入 ONNX + `step1` | ✅ 能给量化参数 | — step1 rc=0 |
| UINT8 图输入 ONNX + `step2` / `build` | ❌ | ❌ `RKNNInitRNNConst: Not Support Dtype: 2` |
| `quantized_dtype='w8a16'` | — | ❌ rk3588 平台不支持 |
| `custom_string` 混合 DSL | — | ❌ 2.3.0 已移除 |

**结论修正**：混合模型**是做得出来的**（前面判"toolkit 缺陷"是我的调用错误，已更正）。
真正做不到的是**让混合模型的输入停在 int8**。

**根本原因**：toolkit 把"图的输入张量 dtype"和"计算精度"绑在一起。
`--no-quant` ⇒ 输入 FLOAT16；量化开启 ⇒ 全图 int8；混合 ⇒ 它选中间态，
把**输入**判成 float16 而只量化**部分中间层**。
而要绕过它自己做 uint8 图输入，toolkit 能理解这个图（step1 全部标定跑完、给出量化参数），
却在最后生成可执行模型那一步 abort。

---

## 5. 已产出可用的东西

| 产物 | 位置 | 说明 |
|---|---|---|
| toolkit 逐层建议 cfg | `rknn_work/hm/hybrid/trial0474_hm_nop0.quantization.cfg` | 76 个 fp16 张量 + 逐层分数 |
| 独立 SQNR 表 | `rknn_work/hm/hybrid/quant_loss_per_layer.csv` | 426 张量，含 SQNR/NRMSE/范围 |
| 混合模型 L1/L2/L3 | `rknn_work/hm/models/trial0474_hm_nop0_hybridL{1,2,3}.rknn` | 可直接上板，**但输入是 fp16** |
| uint8 输入 ONNX | `rknn_work/hm/hybrid/trial0474_hm_nop0_u8b.onnx` | 逐位等价，但 toolkit 拒收 |
| 转换脚本 | `manu/pipeline/npu/hm_convert_hybrid_rknn.py` | step1 / step2 / build 三模式 |
| 逐层分析 | `manu/pipeline/npu/analyze_quant_loss_per_layer.py` | 可复现 |
| 输入改写 | `manu/pipeline/npu/make_uint8_input_onnx.py` | 含等价性验证路径 |
| cfg 补丁工具 | `manu/pipeline/npu/patch_hybrid_cfg.py` | step2 绕过尝试（失败，但可复现） |

### 如果必须拿到"输入 int8 + 混合精度"，剩下唯一可行方向

不走 toolkit 的混合量化，而是**自己在 ONNX 层把敏感段隔离出来**：
用 `quantized_hybrid_level` 拿不到的东西，可以通过
「把 int8 图的输入保持 uint8，让 toolkit 自己插输入量化（std_values）」+ 
「在 ONNX 里把 `b2`/head 段包成不可量化子图」来实现。代价是需要一个 toolkit 支持的
"子图豁免"机制——目前**没有证据表明 2.3.0 有**，所以这是一条需要先验证可行性的研究路线，
不是可以直接开做的工程任务。

另一条更稳的路是**换 toolkit 版本**：本轮所有失败都发生在 toolkit 侧
（`quant_optimizer`、`RKNNInitRNNConst`、平台级 `w8a16` 限制）。
Rockchip 后续版本对 hybrid quantization 有重写，如果业务上必须做混合精度，
先在更新的 toolkit 上验证 §2.1 那份 76 层清单能否直接喂给 step2，是性价比最高的下一步。

---

## 6. 未完成：板卡依赖项

混合模型 L1/L3 的**上板速度测量被污染作废**，原因是一个已知的操作陷阱：

早先 core(0,2) 失败实验（v2c02）的进程没有随实验结束退出，仍占着一个 NPU 上下文。
加上新跑的两个 worker = 3 个上下文，正好越过 RK3588 的 2 上下文上限，于是：

```
E RKNN: failed to submit!, op id: 144, op name: Add:/model/c19/m.0/m/m.0/Add, int status: 0x100
rk_iommu fdab9000.iommu: Page fault ... of type write
RKNPU: job abort, ret: -22 / RKNPU: soft reset, num: 6
```

随后板卡掉线（100% 丢包），需断电恢复。

⚠️ **操作铁律（新增）**：板卡上任何一次 demo 测量前，必须先确认没有残留的
`e2e_par2_demo` 进程——一个僵尸上下文足以让下一次测量产出 `failed to submit` + IOMMU
page fault，而**表面上进程还在跑、行数还在涨**，不主动查进程表就会把污染数据当成结果。

### 待板卡恢复后要补的测量

1. 混合 L1 / L3 的 `inputs_set` / `rknn_run` / 帧率（板上 `input attr` 已确认为 FLOAT16）
2. 混合 L1 / L3 的逐帧输出 vs fp16 基线
3. L1 / L3 的 24 序列全量 F1（走 `run_npu_pcfeat_board_generic.sh`，与 §26 同口径）
