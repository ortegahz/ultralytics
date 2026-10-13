# ONNX → RKNN 重新量化（自建 rknn 环境）

> **结论（2026-10-13 实测）：从 SDK 自带的 `yolov5s.onnx` 重新量化出的 `.rknn`
> 与官方 `yolov5s-640-640.rknn` 检出结果一致——5 个框全部匹配，类别全同，IoU 0.85~0.94。**
> 完整对账见第七节。

## 一、交付物

| 文件 | 作用 |
|:---|:---|
| `convert_yolov5_to_rknn.py` | 转换脚本（config 对齐官方，量化开启，带产物校验） |
| `compare_detections.py` | 两个模型检出结果的**逐框 IoU 对账**工具 |
| `fix_wheel_version.py` | 把 SDK 那份非 PEP440 的 wheel 重打包 |
| `clear_execstack.py` | 清除 `.so` 的 `PT_GNU_STACK` 可执行位（免 patchelf） |
| `yolov5s_rknn230.rknn` | **本轮产出的新模型**（8.38 MB，INT8，MD5 `4b8459fbc6c2c1ad5ba938200ee4433e`） |

## 二、环境（不落根分区）

conda 默认装在 `/home/manu/anaconda3`，而根分区 `/dev/sda1` **只剩 15 GB / 93% 已用**。
两个 rknn 环境都建在 1 TB 卷上：

```
/media/manu/1TB-Volume/conda-envs/rknn     # toolkit 1.4.0（SDK 自带 wheel，产出被 NPU 拒绝，见第六节）
/media/manu/1TB-Volume/conda-envs/rknn230  # toolkit 2.3.0（可用，与板卡 runtime 同 commit）
```

工作区：`/media/manu/1TB-Volume/rknn_work/`（wheels / models / 中间产物）

## 三、源模型与目标模型不是同一个

| | `examples/onnx/yolov5/yolov5s.onnx` | demo 自带的 `yolov5s-640-640.rknn` |
|:---|:---|:---|
| 输入 | `[1,3,640,640]` **float32 NCHW** | `[1,640,640,3]` uint8 NHWC（转换后） |
| 输出 | **4 维** `[1,255,80,80]` / `[1,255,40,40]` / `[1,255,20,20]` | **5 维** `[1,3,85,80,80]` / `[1,3,85,40,40]` / `[1,3,85,20,20]` |
| 来源 | — | 由 `yolov5s_rm_transpose.onnx` 转换 |

255 = 3 anchors × 85 channels。**两种布局在解码索引上完全等价**，因为：

```
(PROP_BOX_SIZE * a + c) * grid_len + i * grid_w + j      # 对 [1,255,H,W] 和 [1,3,85,H,W] 都成立
```

所以官方 demo 的 `postprocess.cc` 不改一行就能吃下新模型；
`rknn_npu_smoke.cpp` 原本硬断言 5 维，已改成**从末尾两维推导网格并断言形状**（回归已验证）。

## 四、🔴 核心：toolkit 版本必须与板卡 runtime 对齐

板卡实测：

```
librknrt version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
driver version:   0.9.8
```

用 **toolkit 1.4.0**（SDK `packages/` 里唯一附带的那份）转换出来的模型，
`rknn_init` 成功、张量属性也读得出来，但**每一次推理都在提交阶段被 NPU 拒绝**：

```
E RKNN: failed to submit!, op id: 145, op name: OutputOperator:328, flags: 0x5, ...
        If using rknn, update to the latest toolkit2 and runtime from: ...
```

**症状极具误导性**——它不报错退出，而是每次超时约 6.4 s 后返回，
输出缓冲区**从未被填充**，于是：

| 指标 | 真实值 | 被误读成 |
|:---|---:|---|
| 平均单次推理 | 6395 ms | 「模型很慢」 |
| 检出分数 | 全部 ≈ 0.9976 | 「模型很自信」 |
| 检出框 | 退化、坐标乱七八糟 | 「框算错了」 |

⚠️ **教训：「输出非零 + 看起来有结构」不构成模型可用的证据。**
必须同时看**单次耗时**是否落在量级内——6395 ms vs 正常的 23 ms，本身就是最强的异常信号。

换成 **toolkit 2.3.0**（PyPI，与板卡 runtime **同 commit `c949ad889d`**）后一次通过：

```
I librknnc version: 2.3.0 (c949ad889d@2024-11-07T11:39:30)   ← 与板卡 runtime 同一 commit
[OK] wrote yolov5s_rknn230.rknn  (8.38 MB, magic RKNN)
```

**⇒ 以后做任何 RKNN 转换，第一件事是核对 toolkit 版本与板卡 `librknnrt` 的 commit hash 是否一致。**

## 五、装环境时踩到的四个坑

| # | 现象 | 根因 | 解法 |
|:---|:---|:---|:---|
| 1 | `InvalidVersion: '1.4.0-22dcfef4'` | SDK 的 wheel 文件名与 METADATA 都把 git hash 当成 **pre-release** 段，PEP 440 只允许 `+local` | `fix_wheel_version.py` 重写 METADATA 与 dist-info 目录名 |
| 2 | `ModuleNotFoundError: torch` | toolkit 2.x 的 `rknn/api/rknn_utils.py` **确实 import torch**，无法绕开 | 装 **CPU-only** wheel：`--index-url https://download.pytorch.org/whl/cpu` |
| 3 | pip 卡在 `nvidia_cuda_cupti_cu12` 下载 | 默认索引会拉整套 CUDA 包，而**本机无 GPU**，且大文件下载不稳 | `--index-url .../whl/cpu` 或 `--no-deps` 后手动补 ONNX 路径依赖 |
| 4 | `cannot enable executable stack as shared object requires` | onnxruntime 的 `.so` 带 `PT_GNU_STACK` **RWX**，被系统加载器拒绝 | `clear_execstack.py` 清掉 `PF_X`（等价 `patchelf --clear-execstack`，但免依赖） |

另有一条 **opset/IR 版本**的坑，只在 toolkit 1.4.0 那条路上出现：

- onnxruntime 1.12 最高支持 **IR 8**，而 onnxoptimizer 把模型提到 **IR 9**
  → 降级 `onnx==1.9.0` + `onnxruntime==1.10.0`（toolkit 1.4.0 的钉死版本）
- 换成 toolkit 2.3.0 后 onnxoptimizer 把模型提到 **opset 19**，而 onnxruntime 1.12 只保证到 **opset 17**
  → 保持 `onnx==1.14.1`（toolkit 2.x 需要），只把 **onnxruntime 升到 1.17.3**

> 规律：**onnx（写）/ onnxruntime（读）必须同时覆盖对方产出的版本**，
> 只调一边会在下一个阶段换个报错冒出来。

## 六、复现步骤

```bash
# 1) 环境（1 TB 卷，不碰根分区）
conda create -y -p /media/manu/1TB-Volume/conda-envs/rknn230 python=3.8
P=/media/manu/1TB-Volume/conda-envs/rknn230
$P/bin/pip install --no-deps rknn-toolkit2==2.3.0
$P/bin/pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
$P/bin/pip install numpy==1.23.5 protobuf==4.25.4 opencv-python==4.8.1.78 \
                      onnx==1.14.1 onnxruntime==1.17.3 scipy==1.10.1 psutil==5.9.0 \
                      tqdm==4.64.0 ruamel.yaml==0.17.21 flatbuffers==1.12 requests==2.27.1 \
                      fast-histogram
python3 clear_execstack.py --scan $P/lib/python3.8/site-packages/onnxruntime

# 2) 转换
cd /media/manu/1TB-Volume/rknn_work
$P/bin/python convert_yolov5_to_rknn.py --out models/yolov5s_rknn230.rknn

# 3) 上板（数据走 NFS）
cp models/yolov5s_rknn230.rknn /home/manu/nfs/npu/demo/model/RK3588/
cd ../../workspace/ultralytics/manu/pipeline/opencl/board
RK3588_PASSWORD=*** python3 board_run.py --exec \
  'cd /mnt/manu/npu/demo && LD_LIBRARY_PATH=/root/rknn_demo/lib \
   /root/rknn_demo/rknn_yolov5_demo model/RK3588/yolov5s_rknn230.rknn bus.jpg'

# 4) 逐框对账
python3 compare_detections.py det_old.txt det_new.txt 0.5
```

## 七、结果对账（官方 demo，`bus.jpg` 640×640）

**官方 `yolov5s-640-640.rknn`（基线）**
```
person @ (114 235 212 527)  0.819099
person @ (210 242 284 509)  0.814970
person @ (479 235 561 520)  0.790311
bus     @ (99  141 557 445)  0.693320
person  @ (78  338 122 520)  0.404960
average run 23.428400 ms
```

**本轮新量化 `yolov5s_rknn230.rknn`**
```
person @ (211 246 283 515)  0.810206
person @ (472 233 560 521)  0.809694
person @ (111 237 204 540)  0.783034
bus     @ (87  134 559 460)  0.769140
person  @ (79  333 121 517)  0.459860
average run 26.236600 ms
```

`compare_detections.py` 逐框配对（IoU ≥ 0.5）：

| 类别 A→B | IoU | A (l,t,r,b) | B (l,t,r,b) | 分数 A/B | 最大角点偏差 |
|:---|---:|:---|:---|:---|---:|
| person → person | 0.850 | (114,235,212,527) | (111,237,204,540) | 0.819/0.783 | 13 px |
| person → person | 0.938 | (210,242,284,509) | (211,246,283,515) | 0.815/0.810 | 6 px |
| person → person | 0.901 | (479,235,561,520) | (472,233,560,521) | 0.790/0.810 | 7 px |
| **bus → bus** | **0.905** | (99,141,557,445) | (87,134,559,460) | 0.693/0.769 | 15 px |
| person → person | 0.915 | (78,338,122,520) | (79,333,121,517) | 0.405/0.460 | 5 px |

**5/5 匹配，类别 5/5 相同，IoU 0.850~0.938。**

### 残差来源（已归因，不是「差不多就行」）

1. **源 ONNX 不同**：基线由 `yolov5s_rm_transpose.onnx` 转换，本轮用 `examples/onnx/yolov5/yolov5s.onnx`，
   权重导出路径不同 ⇒ 框本身就不是同一组数。
2. **量化标定**：基线由未知版本/标定集生成；本轮严格按 SDK 的 `mean=0 / std=255` + `target_platform=rk3588`，
   标定集是 SDK 自带的**单张 `bus.jpg`**（官方脚本同样只给一张）。单张标定不是有代表性的标定集，
   这也是 bus 的置信度从 0.693 变成 0.769 的主要原因。
3. **推理时机**：均值 26.24 ms vs 23.43 ms；`rknn_run` 23.94 ms vs 24.20 ms 基本持平，
   差距主要在 `outputs_get`（1.72 ms vs 0.47 ms）——4 维 NCHW 输出需要驱动做一次布局转换。

### ⚠️ 单张标定的局限（必须记下来）

`dataset.txt` 里只有一张图。用**被评测的那张图本身**做标定，等于用测试集调参，
得到的分数偏乐观、跨图泛化性未知。**若要把这条链路用于真正的部署模型，
标定集必须换成几十到上百张有代表性的场景图**，且不能再用 bus.jpg 做标定。

## 七之二、留出集验证（把上面那个局限补上）

第七节只在 `bus.jpg` 上验证过，而它同时是标定图 —— **那张结论本身也可能被标定泄漏污染**。
为排除这一点，从 COCO `val2017` 随机抽了 **120 张做标定 + 10 张留出做评测**（互不相交），
重新量化出第三个模型，与前两个一并对比。

```
标定集：/media/manu/1TB-Volume/rknn_work/calib/coco_cal/   （120 张，随机种子 20261013）
留出集：/media/manu/1TB-Volume/rknn_work/calib/coco_test/  （10 张，未参与标定）
新模型：yolov5s_rknn230_coco120.rknn
```

脚本新增 `--calib-dir` / `--calib-list` 以支持自定义标定集：

```bash
python3 convert_yolov5_to_rknn.py \
        --calib-dir  calib/coco_cal \
        --calib-list calib/calib_coco120.txt \
        --out models/yolov5s_rknn230_coco120.rknn
```

### 7.2.1 三模型在 10 张留出图上的汇总

| 模型 | 标定集 | 检出总数 | 与官方匹配 | 平均 IoU | 同类别 |
|:---|:---|---:|---:|---:|---:|
| `yolov5s-640-640`（官方） | 未知 | 17 | — | — | — |
| `yolov5s_rknn230` | **单张 bus.jpg** | 21 | 16/17 (94.1%) | 0.810 | 14/17 (82.4%) |
| `yolov5s_rknn230_coco120` | **120 张 COCO** | 21 | 16/17 (94.1%) | 0.798 | 14/17 (82.4%) |

**两个重新量化的模型结果几乎完全一致**（94.1%，IoU 0.810 vs 0.798，类别同为 14/17）。

⇒ **与官方模型的差异主要来自源 ONNX 不同，而不是标定集质量。**
把标定集从 1 张扩到 120 张并没有改善一致性，说明这是**源权重**（`yolov5s.onnx` vs
`yolov5s_rm_transpose.onnx`）带来的，**不可能靠调标定抹平**。

两个新模型都比官方模型**多检出 4 个框**（21 vs 17），即在本任务上**略偏敏感**。

### 7.2.2 分歧全部集中在决策边界

全部分歧逐条列出：

```
[CLASS DIFF] 000000049810  official:         dog s=0.152 (56,147,559,419)
                        new     :         cat s=0.513 (117,154,628,422)   IoU=0.748
[CLASS DIFF] 000000258793  official:      person s=0.283 (486,232,500,252)
                        new     :     bicycle s=0.357 (482,233,501,252)   IoU=0.709
[UNMATCHED] 000000258793  official:      person s=0.220 (482,232,494,250)   （12×18 px 极小框）
[NEW ONLY  ] 共 5 个，分数 0.124 ~ 0.455
```

按置信度 / 框尺寸分层后规律非常干净：

| 分层 | 一致率 |
|:---|---:|
| 官方分数 `[0.25, 0.40)` | 1/2（50%） |
| 官方分数 `[0.40, 0.60)` | 4/4（**100%**） |
| 官方分数 `[0.60, 1.00]` | 5/5（**100%**） |
| 框最长边 `< 50 px` | 2/4（50%） |
| 框最长边 `50–150 px` | 7/7（**100%**） |
| 框最长边 `> 150 px` | 5/6（83%） |

**⇒ 分数 ≥ 0.40 且框 ≥ 50 px 的检出，两个模型在类别与位置（IoU ≥ 0.5）上完全一致；
所有分歧都发生在阈值附近的低置信或极小框上。**
唯一「丢失」的框是 12×18 px、置信度 0.220 的小人框。

> ⚠️ **样本量诚实声明**：留出集只有 10 张、官方模型共 17 个检出。
> 上述比例适合说明「分歧在哪」，**不足以支撑精度指标的统计结论**。

### 7.2.3 🔴 扩到 300 张后，上面两条结论都被推翻（2026-10-13 补测）

因为上一节自己写了「10 张不足以支撑结论」，于是用 **300 张留出图（与 120 张标定互不相交）**
重跑。参考模型共 **1369 个检出**，逐框 IoU 贪心匹配：

| 候选模型 | 标定集 | 检出数 | 参考框匹配率 (IoU≥0.5) | 95% CI | 平均 IoU | 匹配框同类率 |
|:---|:---|---:|---:|:---|---:|---:|
| `yolov5s_rknn230` | 单张 bus.jpg | 1622 | **1157/1369 = 84.5%** | [82.5, 86.3] | 0.862 | 94.7% |
| `yolov5s_rknn230_coco120` | 120 张 COCO | 1629 | **1151/1369 = 84.1%** | [82.0, 85.9] | 0.854 | 94.5% |

**🔴 更正一：10 张图给出的「94.1% 匹配」是偏乐观的估计，真实值约 84.5%。**
小样本把匹配率抬高了约 10 个百分点 —— 这正是本项目铁律「±0.0002 F1 噪声下限」的同类陷阱，
只不过这里发生在**比例统计**上。

**🔴 更正二：「所有分歧都集中在决策边界」在小样本上成立，在 1369 个框上不成立。**
分层后同类率其实相当均匀：

| 按参考框分数分层 | 同类率 |
|:---|---:|
| [0.25, 0.40) | 94.5% |
| [0.40, 0.60) | 96.6% |
| [0.60, 0.80) | 98.9% |
| [0.80, 1.00] | 100.0% |

| 按参考框最长边分层 | 同类率 |
|:---|---:|
| < 25 px | 100.0% |
| 25–50 px | 95.8% |
| 50–150 px | 96.5% |
| > 150 px | 93.3% |

**置信度只带来约 5 个百分点的提升，不存在「干净的边界效应」。**
17 个检出时看起来像「分歧全在低置信区」，1369 个检出时只是**回归到均值**。
按框尺寸看，大框（>150 px）反而略低（93.3%），与小框假设相反。

**✅ 存活且被加强的结论：标定集大小不影响与官方模型的一致度。**
1 张标定 → 84.5%，120 张标定 → 84.1%，差异远小于置信区间半宽（±1.9%）。
**与官方模型的差异来自源 ONNX 不同，不是标定质量。**

**🔴 新发现：重新量化的两个模型都系统性多检出约 18–19%**（1622 / 1629 vs 1369），
且多出的 465 / 478 个框是**候选模型独有**的。这不是噪声，是**一致的敏感性差异**：
在同样阈值下，重新量化出来的模型判正更松。

复现：

```bash
python3 agree_stats.py /home/manu/nfs/npu/h300_out ref \
        yolov5s_rknn230 yolov5s_rknn230_coco120
```

> ⚠️ 本节 300 张 / 1369 框的结论已带 Wilson 95% 置信区间，可用于工程判断；
> 但**若要判定「哪个模型更准」，还需要人工标注的 GT 做精确率/召回率评测**，
> 本节的全部指标都只是**两个模型之间的一致性**，不是精度。

## 八、对项目主线的意义

- 「从 ONNX 到 RKNN 再上板推理」的**完整闭环已在本机跑通**，不再依赖 Rockchip 现成模型。
- 这意味着 **Trial 0474 可以走同一条路**：服务器侧用 rknn-toolkit2 2.3.0 转 `.rknn`，
  板卡侧用 `rknn_npu_smoke` 做推理与分段计时。
- ⚠️ 但注意：Trial 0474 是**三通道 heatmap 检测器**（输出 `[H,W,1]` 类热图，不是 yolov5 的 85 通道头），
  **后处理不能照搬本文的 yolov5 解码**，只能复用「转换 + 部署 + 计时」这一段。