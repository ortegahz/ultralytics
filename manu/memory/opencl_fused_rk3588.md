# OpenCL 融合内核移植 —— RK3588 (2026-10-09)

## 结论一句话

21 元素排序网络**完备证明正确**（109 次比较，2^21 全枚举）；融合内核 **Ch0 逐位精确**（4345 万像素 Max|Diff| = 0，证明无坐标反转 / 通道错位 / 系统性漂移），Ch1/Ch2 MAE 0.0007/0.0003（闸门 0.05，富余 68×），仅 ~600/4340 万像素超到 Max|Diff| ≤ 7。

---

## 1. 排序网络：完备证明，不是抽样

```
tools/sortnet_verify.c
生成 Batchers 32-lane → sentinel-drop 到 21 → 全枚举验证 → 保证明剪枝
PASS exhaustive 0/1: 2097152 / 2097152, 112 comparators
prune round 1 -> 109 comparators
PASS exhaustive 0/1 after prune: 109
```

**验收标准写的「91 次比较」复现不出来。** 已核实：任何我能构造并**证明**的方案都落在 109。Batchers 奇偶归并排序的公开伪码**只在 n 为 2 的幂时成立**——n 为奇数时 `sort(lo+m, m)` 丢掉最后一格，且归并递归会走到 `CE(lo, lo+r)` 且 `lo+r >= n`（静默越界写）。`tools/dbg_sortnet.c` 逐条演示了这两点，同时证明生成器本身在 P ≤ 16 的所有 2 的幂上都正确、sentinel-drop 构造在 n = 3..21 上全部正确。

sentinel-drop 是**严格正确**的，不是启发式：n 以外的 lane 是 +∞ 哨兵；归纳可证 lane j ≥ n 只会接收 `max(a_i, a_j)`，而 `a_j = +∞` 时 max = +∞，所以这些 lane 永远不装真实值，所有触及它们的比较子对 0..20 都是 no-op。

### 完备验证抓到的 3 个真缺陷（全部表现为「时好时坏」）

1. `unsigned char a[N]` 声明在 `#pragma omp parallel for` 循环体**外** → 所有线程共享。
2. reduction 变量在 `#pragma omp critical` 内 `bad |= 1` → 偶发丢失更新，约 1/200 次调用假 FAIL。
3. **剪枝恢复不是真 undo**：把 `saved` 追加到 `g_net[g_nce]` 会在删除点留洞，并把 `saved` 放到下一轮左移当作普通元素读取的位置——`g_nce` 仍是 112 但网络已被污染，于是「最终有效性检查」在一个从未真正损坏的网络上失败。修法是**右移撤销**。

> 教训：一个用来给内核背书的工具本身出现间歇性失败，比工具报错危险得多。

---

## 2. OpenCV 语义：全部实测，未靠记忆

上一阶段 11 个缺陷全部来自「读源码/凭记忆」。本阶段每条算术决策都由探针给出。

| 事实 | 证据 |
|---|---|
| dst→src，**无半像素偏移** | `probe_interp` 36 模型扫描：`(x,y)` 99.58% 精确；`(x+0.5,y+0.5)` 仅 0.79%，MAE 39.1 |
| 位置量化到 **1/32，四舍五入半上取整**；输出同样半上取整 | `probe_interp2` 阶跃 oracle：33 级阶梯；phi=1/64（phi·32 恰为 0.5）实测取 1/32 → 排除 round-half-even。round32 65/65 vs trunc32 33/65 |
| **`CLK_FILTER_LINEAR` 在 PoCL 上不可用** | 实测：Ch0（本该是逐位拷贝）Max\|Diff\|=34、MAE 1.74；改手动 2×2 后 Max\|Diff\|=0、MAE 0.000 |
| 边界规则 = **两个 tap 各自反射**（模型 B） | `probe_border` 96000 个越界 tap：A（反射基址再取 base+1）错 29.0%，B 错 0.0% |
| 坐标表达式形式**无关紧要** | `probe_coord` 5 种结合/FMA 次序，600 万真实样本，极差 ≤ 0.0004% |

**边界这条踩了坑**：我一度认为 OpenCV 是「先反射基址再取 base+1」（模型 A），把反射搬进内核、pad 降到固定 2，结果 640 宽序列 Ch1 Max|Diff| 从 7 恶化到 **239**。`probe_border` 一跑就证明 B 才对，回退后立刻恢复。

---

## 3. 最终精度（8 序列 / 136 帧 / 4345 万像素，双分辨率 512×512 与 512×640）

```
channel |   Max|Diff| |     MAE |  最差单帧 MAE
  Ch0   |           0 |   0.00000 |         0.00000
  Ch1   |           7 |   0.00073 |         0.00160
  Ch2   |           6 |   0.00033 |         0.00088
|d| 直方图
channel |      0 |     1 |    2 |   3 | 4-6 | >=7
  Ch0   | 43450368 |     0 |    0 |   0 |   0 |    0
  Ch1   | 43420339 | 29323 |  457 | 127 | 116 |    6
  Ch2   | 43436478 | 13792 |   79 |  11 |   8 |    0
```

**Ch0 逐位精确是承重结论**：它是当前帧的纯拷贝，一次性验证 padding、通道序、ring 索引、平面排布、坐标约定。任何坐标反转 / 通道错位 / 系统性漂移都会立刻打崩 Ch0。

残余 ~600 个像素（1.4e-7）超到 ≤ 7，**已定量排除**：
- 不在边界（实测 d_edge 13–212）
- fp64 坐标不改变结果（`--fp64-coord 1`）
- 5 种 float32 表达式差异 ≤ 0.0004%
→ 是 OpenCV 增量式定点映射表与任何直接仿射求值的舍入分歧，在 1/32 格 tie 上翻格。复刻它要照搬 OpenCV 的建表过程，不值得。

**要压到 Max|Diff| ≤ 1 需要双精度浮点（double-float）坐标运算**：每轴每采样约多 12 个 float32 指令，ALU 成本约翻倍，换 4340 万像素里 13 个像素的变化。RK3588 上不划算——但这是显式决策，不是默认忽略。

---

## 4. RK3588 适配（已写进代码形态，不只是文档）

1. **21 个具名标量 `v0..v20`，绝不用 `v[k]`**——循环索引会让编译器假定索引动态，把数组溢出到 private memory，而 Mali 用**全局显存**模拟 private。排序网络因此以 `CAS(v3, v11)` 形式拼接。
2. **只用 float**，内核无 double；Mali-G610 无可用 FP64。仿射求逆在宿主 double 完成（与 OpenCV 一致）。`--fp64-coord` 仅供诊断。
3. 无 local memory / barrier / atomic；global size `(W,H)`，local size 传 NULL，不硬编 work-group。
4. **22 个独立 `image2d_t` 而非一个 `image2d_array_t`**：ring 按指针轮转，每步只有新帧过总线；array 布局每输出帧要重写全部 21 层（512×512 下 5.5 MB vs 256 KB）——在 RK3588 上比内核本身要省的算力还贵。
5. **kernel 参数地址空间规则**（编译器报错学到的）：image/sampler 参数**不能**带限定符（`"parameter may not be qualified with an address space"`）；指针参数**必须**带（`"pointer arguments ... must reside in __global, __constant or __local"`）。
6. 不用 `-cl-mad-enable` / `-cl-fast-relaxed-math`——内核数值要对着 CPU 黄金参考验证，放开重结合会让验证失去意义。
7. **`CL_R8` 优先、`CL_RGBA` 兜底**，`pick_gray_format()` 实测并打印。Mali 支持 CL_R8（1/4 带宽）；PoCL 6.0 拒绝 CL_R/CL_R8 只收 CL_RGBA。

### 关于 OpenMP

只有 `sortnet_verify.c` 用，且是 `#ifdef` 保护、用途是把 2^21 验证在工作站多核上展开。它是**构建期证明工具**，唯一产物是 `sortnet_generated.inc`。RK3588 BSP（Cortex-A76/A55，无 OpenMP 运行时）根本不编译这个 TU。运行期路径（`ocl_fused_check.cpp` + `ocl_host.h`）完全不用线程。

---

## 5. 本机环境缺陷（已在 `ocl_host.h` 用 `#ifndef` 兜底，Mali 厂商头优先）

1. **`dpkg -L opencl-headers` 一个头文件都没有**（只有 doc）。`/usr/include/CL` 来自 `opencl-c-headers 3.0~2025.07.22`，它声明了 OpenCL 2.0 的**函数**却漏掉了全部 2.0 **枚举量**：`CL_R8`、`CL_IMAGE_OBJECT_2D`、`CL_SAMPLES_UINT`、`CL_MEM_OBJECT_*` 都没有，且 `clCreateSampler` 是 1.0 的五参数签名。
2. **`clCreateImage` 在此 ICD 上恒返回 `CL_INVALID_IMAGE_DESCRIPTOR`**（连 `clCreateImage2D` 能接受的格式也是），因为头文件的 `cl_image_desc` 布局与 ICD 分发不一致。宿主先试 2.0 入口（正确栈上它会赢），回退 `clCreateImage2D`，并用作用域 pragma 保持零废弃警告。
3. **陷阱：`CL_NONE == 0 == CL_FALSE`，而 `0` 同时是属性数组终止符**。写 `CL_SAMPLER_MIP_FILTER_MODE, CL_NONE` 会被读成「属性后紧跟列表结束」，所有运行时都返回 `CL_INVALID_VALUE`。`CL_NONE` 是默认值，**省略该属性**既正确又是唯一能拼出来的写法。

---

## 6. 我自己写的两个 bug（都不是内核问题，但一度误导）

1. **对账工具 CPU 参考 `cv::Mat ch2(W, H, ...)` 把 rows/cols 写反** → 512×512 正方形时被掩盖，640 宽序列直接内存越界 + 未初始化，表现为 Ch2 Max|Diff| = 255。
2. **`probe_coord` 的 tap 守卫 `y0 < 1` 把所有 y0=0 的行判成失配** → 报告 Max|Diff| = 145，结论作废；修好后 600 万样本 Max|Diff| = 2，与 GPU 侧一致。

> 这两条与上一阶段 11 个缺陷同源：**验证工具自身的错误会被当成被验证对象的错误**。两个工具都加了断言/守卫，但结论必须靠交叉验证（GPU 直测 ↔ 独立 CPU 探针）才敢采信。

---

## 7. 端到端集成已完成（2026-10-09 续）

`manu/pipeline/cpp/gmc_stream_ocl.cpp` = `gmc_stream.cpp` 的超集。CPU 保留 Shi-Tomasi / LK / RANSAC / 锚点网格 / 连乘，**只有 warp + median + 三通道组装这一段移到 GPU**。

### fork 保真是被证明的，不是假设的

`--fused cpu` 与 `gmc_stream` 的 MD5 完全一致（`5e0d336b971948f631640dc983acfc23`，`01_4485_1167-2666` / limit 60 / anchor-step 2）。**在这一点被验证之前，本文件此前所有 GPU 数字都不该被采信。**

`--fused both` 的两个臂消费**同一个 `mats`**（每 push 只算一次拟合），因此 A/B 不可能靠「两边输入不同」蒙混过关。

### 端到端 A/B（`--anchor-step 2`，`--limit 120`）

```
序列                              Ch0                        Ch1                        Ch2
01_4485_1167-2666  0 / 0.00000 / 100.0000%    7 / 0.00034 / 99.9738%   6 / 0.00013 / 99.9890%
01_1751_0250-1750  0 / 0.00000 / 100.0000%    8 / 0.00096 / 99.9170%   8 / 0.00031 / 99.9700%
--mode nogmc       0 / 0.00000 / 100.0000%    0 / 0.00000 / 100.0000%  0 / 0.00000 / 100.0000%
```

（Max|Diff| / MAE / 逐位相同率）

**`nogmc` 那行（W = IDENTITY）是能给出的最强陈述**：身份变换下融合内核与 CPU 参考**逐位相同**，三通道全帧 100%。一次性确认 padding、ring 索引、通道序、平面排布、坐标约定全链路正确。非身份行复现了隔离 checker 的数字，说明集成自身没引入新误差。

### 时延（ms/帧）

```
序列                        fit   CPU尾段(warp+median)  GPU尾段  upload   KERN    read
01_4485_1167-2666         24.6      72.1               153.4    0.036  118.1   0.347
01_1751_0250-1750         35.5      83.2               114.0    0.024  112.6   0.325
01_1751_0250-1750 nogmc    0.0      39.4               115.5    0.034  114.4   0.342
```

**PoCL 上 GPU 臂比 CPU 臂慢 1.3~2.9 倍，这是预期结果而非缺陷**：PoCL 是 CPU 模拟，kernel 与 CPU 臂抢同样 12 个核；而且手动 2×2 混合每个采样要 4 次 NEAREST 纹理读，真 GPU 的 `CLK_FILTER_LINEAR` 只要 1 次。

**能迁移到 RK3588 的是传输列，不是 kernel 列**：upload 0.024~0.036 ms、readback 0.325~0.347 ms 是总线流量，对任何独立 GPU 都是真实数字；112~118 ms 是 CPU 模拟的属性，对 Mali-G610 不具参考性。**按 memory_compact 既有铁律，PoCL 的任何时延都不得进入嵌入式预算。**

### 埋点时抓到的 2 个测量缺陷

1. **`clEnqueueNDRangeKernel` 在「入队」时就返回。** 围绕它计时得到 0.04 ms，而随后的**阻塞式**回读报 116 ms —— 计算被藏在同步里，第一版结论「回读主导」**完全是错的**。修法是 enqueue 后立刻 `clFinish`，在回读之前就把计算与传输分开；随后两个独立测量一致（118.086 vs 118.081 ms）。教科书做法 `CL_PROFILING_COMMAND_START/END` 在这套栈上不可用：`clCreateCommandQueueWithProperties` 对 `CL_QUEUE_PROFILING_ENABLE` 返回 `CL_INVALID_VALUE`，且 `clCreateEvent` 在本机 `cl.h` 里根本没声明。
2. **取模环不是内容环。** 设备槽位原先按 `绝对索引 % N` 键入并做「已驻留？」检查。冷启动时 `frame_at_lag()` 把所有越过首帧的 lag 钳到 `first_`，于是**当前帧与被钳的历史帧可以同余**；取模版本看到「槽位已含 key K」就跳过上传，留住上一帧内容。症状：**Ch0 有 31% 像素错、最大偏 80 灰阶，预热后消失**。现按**内容身份**分配槽位，使该碰撞不可表示。这正是「Ch0 必须逐位精确」这道闸门存在的意义。

---

## 8. 下一步（未做）
2. **RK3588 上实测重测 `--hw-linear`**：PoCL 的线性过滤不可用不代表 Mali 的不可用（规格要求 `CLK_FILTER_LINEAR`，但必须实测决定）。
3. **分段计时**：PoCL 是 CPU 模拟，时延对嵌入式预算无参考价值。必须以 RK3588 为准，沿用 C++ 基线的 fit / warp+median / total 分段。
4. **CL_R8 vs CL_RGBA 的带宽实测**：22 层 × pad 57 的显存占用在 RK3588 上是否可接受。
5. 若最终仍要求严格 Max|Diff| ≤ 1，需上 double-float 坐标运算，并重新评估 ALU 代价。