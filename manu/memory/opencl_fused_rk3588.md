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

0. **【2026-10-09 已交付，待板卡实跑】环境探针 `manu/pipeline/opencl/board/`**：在移植任何融合内核之前，先确认板卡上究竟有没有可用的 OpenCL 栈。`board_cl_probe.cpp` 通过 **dlopen** 依次尝试 `libOpenCL.so.1` / `libOpenCL.so` / `libmali-vendor.so[.1]` / `libpocl.so.2`，打印 ICD 配置、平台、设备（含 extensions）、跑一个平凡 kernel 并逐元素校验，输出 `PROBE RESULT: PASS/FAIL`。已用 buildroot 工具链交叉编译出 **28KB aarch64 ELF**（仅依赖 `libdl/libstdc++/libm/libgcc_s/libc`），host 原生自测 **PASS / exit 0**。
   - **为什么用 dlopen 而非 `-lOpenCL`**：交叉工具链 sysroot 里**没有任何 OpenCL**，SDK 也只有 buildroot 配方没有预编译库，链接期依赖根本无法满足；改成运行期决定后，「没有 OpenCL」变成一条干净的报告而不是链接错误，还能顺带报出究竟是 ICD loader 还是直连 Mali 驱动。
   - **未完成**：投递到 NFS 需用户执行一次 `sudo`（`/mnt/manu` 为 `root:root 0755`，本机 `manu` 无写权限，见 `rules.md` 1b）；板卡登录本机无 `sshpass`，运行命令须用户手工执行。**板卡上的真实 OpenCL 栈尚属未知**，下面第 2~4 项都以此为前置。
1. ~~在 x86 上验证 Mali 特性~~ —— 不可行，Mali 行为必须以真机为准（见第 2 条）。
2. ~~**RK3588 上实测重测 `--hw-linear`**~~ ✅ **已实测，答案是否定的**（见 8.2）。
3. **分段计时**：⚠️ 板卡约束（实测）：`clGetEventProfilingInfo` **返回失败**，事件级 profiling 不可用 ⇒ 只能给 `clEnqueue + clFinish` 墙钟**上界**。且 Mali **首次启动某 kernel 会现场编译**，单次数字必须区分首次与稳态（探针实测 0.475 ms → 0.344 ms）。
4. ~~**CL_R8 vs CL_RGBA 的带宽实测**~~ ✅ **已实测**：Mali 接受 `{CL_R, CL_UNORM_INT8}`，22 张 padded 图仅 **10.5 MiB**（CL_RGBA 为 42 MiB，4 倍）；`clCreateImage`(OpenCL 2.0) 在 Mali 上正常工作（见 8.2）。
5. ~~若最终仍要求严格 Max|Diff| ≤ 1，需上 double-float 坐标运算~~ 🔴 **该方案在真机上不成立**：Mali-G610 的 extensions 实测**不含 `cl_khr_fp64`**（有 `cl_khr_fp16`）。真要更高精度只能走 `cl_khr_fp16` 混合精度或整数定点，**不可假定 double 可用**。

### 8.1 真机 OpenCL 栈实测结论（2026-10-09）

探针在板卡上 **`PROBE RESULT: PASS` / exit 0**，4096 元素逐个校验全对。

- **ICD 是目录式**：`/etc/OpenCL/vendors/` 是**目录**（ARM 参考实现），内含 `mali.icd`（19 B，内容 `libMaliOpenCL.so.1`），**不是** Khronos 的单文件布局。⚠️ `fopen()` 对目录会成功而首次读失败（EISDIR），naive 实现会误报成「ICD 为空」——本项目首版探针正踩此坑，已修（见 `falsified_archive.md` 第三十四节）。
- ⚠️ **`mali.icd` 指向的 `libMaliOpenCL.so.1` 实际并不存在**；真正生效的驱动库是 `/usr/lib/aarch64-linux-gnu/libmali.so.1.9.0`，来自 Debian 包 `libmali-valhall-g610-g13p0-x11-gbm`。`mali.icd` 是 2020-07-29 的遗留文件。⇒ **排查 OpenCL 问题时不要相信 `.icd` 指向的文件名，直接 `find / -name 'libmali*'`**。
- **加载路径**：`libOpenCL.so.1`（ARM 自己的 ICD loader，`/usr/lib/aarch64-linux-gnu/libOpenCL.so.1.0.0`，34,808 B，2017-04-05）→ 加载 `libmali.so.1`。**Khronos 的 `ocl-icd` 并未安装**。
- **设备**：ARM Platform / **Mali-G610 r0p0**，`OpenCL 3.0 v1.g13p0-01eac0.68603db295fbf2c59ac6b927fdfb1c32`，**FULL_PROFILE**，`OpenCL C 3.0` 同版本；**4 compute units**、max work group **1024**、global mem **7902.1 MiB**、local mem **32 KiB**。内核侧 DDK 为 `g18p0-01eac0`（userspace `g13p0-01eac0`，同为 `01eac0` 构建）。
- **与本项目相关的扩展**：`cl_khr_fp16` ✅、`cl_khr_image2d_from_buffer` ✅、`cl_khr_egl_image` ✅、`cl_khr_suggested_local_work_size` ✅、`cl_khr_command_buffer` ✅、完整 `cl_khr_subgroup*` 系列 ✅；**`cl_khr_fp64` ❌ 缺失**。
- 🔴 **事件级 profiling 不可用**（见第 3 条）。
- ⚠️ **NPU 与 GPU 是两回事**：板上 `/dev/rknpu` **不存在**，`librknnrt.so` 虽已安装但无设备节点 ⇒ **RKNN/NPU 路线在当前内核下走不通**，详见 `rules.md` 1b。**OpenCL 走 GPU，与 rknpu 无关，探针 PASS 不代表 NPU 可用。**
### 8.2 🔴 真机精度验证定论（2026-10-09，Mali-G610 r0p0 / OpenCL 3.0 v1.g13p0-01eac0）

交付 `manu/pipeline/opencl/board/`：`gen_fused_case.cpp`（x86 + 真实 OpenCV 生成用例）与
`board_fused_accuracy.cpp`（板卡，无 OpenCV，dlopen）。在真机跑通 **两个采样臂**，判决如下。

**数据**：Anti-UAV `01_4485_1167-2666`，640×512，**真实 GMC**（Shi-Tomasi 600/0.01/4/bs3 →
LK → `estimateAffinePartial2D(RANSAC, 3.0)`，与 `gmc_stream.cpp` 同原语），3 个 case（t=43/44/45），
每 lag 位移 0.95~1.17 px，**pad=67**（真实运动所需，远超 x86 合成用例），padded 774×646。

**CPU 基准**：在 x86 用真实 `cv::warpAffine(..., INTER_LINEAR, BORDER_REFLECT)` 算好后随包发到板卡。
**理由：无任何 arm64 OpenCV**（工具链 sysroot、厂商 SDK、板卡本身都没有），在板卡重实现 OpenCV
插值等于验证「我自己的重实现」而非 kernel，构成循环论证。

**判决一：`FUSED_USE_HW_LINEAR` 必须为 0（manual 双线性）。**

| 臂 | Ch0 | Ch1 | Ch2 | 判决 |
| :--- | :--- | :--- | :--- | :--- |
| hw-linear (`=1`) | Max\|Diff\|=**130** MAE=0.285~0.288，精确率仅 88.5% | Max\|Diff\|=98~129 MAE=0.307~0.421 | Max\|Diff\|=128~133 MAE=0.315~0.330 | **FAIL** |
| manual (`=0`) | Max\|Diff\|=**0** MAE=0 **100.00% 精确** | Max\|Diff\|=5~6 MAE=0.0011~0.0012 | Max\|Diff\|=1~4 MAE=0.00074~0.00078 | **PASS** |

**决定性证据是 Ch0**：它是对当前帧的**整数坐标直读**，任何正确的插值实现都必须逐位还原，
却错了 130 ⇒ **Mali 的 `CLK_FILTER_LINEAR` 连整数坐标都不还原原值**（采样器按 texel 中心做
2×2 混合，与 kernel 假设的「整数坐标即精确命中」不符）。manual 臂走 `CLK_FILTER_NEAREST` +
手写 2×2，绕开该行为，Ch0 立刻 100% 精确。
> x86 侧 `manual` 臂的 Ch1/Ch2（Max 5~6 / MAE 0.0005）与本项目记录的 x86 基线（Max 6~8 /
> MAE 0.0003~0.0010）一致，说明 harness 本身正确，真机差异来自硬件而非工具。

**判决二：Mali 支持 1 字节/像素格式，CL_R8 vs CL_RGBA 项结案。**
`{CL_R, CL_UNORM_INT8}` 被接受 ⇒ 22 张 padded 图 **10.5 MiB**；若用 CL_RGBA 则 **42 MiB**（4 倍）。
`clCreateImage`（OpenCL 2.0）在 Mali 上**正常工作**，无需退回 `clCreateImage2D`。

**判决三：真机时延（首个可信的板卡数字）**

| 臂 | ms/帧 |
| :--- | :--- |
| hw-linear | 4.17 / 4.38 / 4.17 |
| manual | 4.81 / 3.86 / 3.71 |

⚠️ 三重限定，引用时必须一并给出：**① `clEnqueue + clFinish` 墙钟上界**（事件级 profiling 在
本驱动上不可用，取不到 device-side 时间）；**② 含 22 张图的 upload + readback 全程**，不是纯 kernel；
**③ 首次启动某 kernel 含 JIT 现场编译**，须区分首次与稳态。
同批数据在 PoCL 上是 100~850 ms ⇒ 真机 GPU 相对 CPU 模拟约 **25~30×**。
