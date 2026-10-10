# gmc_stream_ocl 在 RK3588 板卡完整运行：BMP 方案、跨架构精度与性能

> 2026-10-10 实测。全部数字来自板卡 `evm3588` 与 x86 同参数对照，非估算。

## 1. 结论摘要

**`gmc_stream_ocl.cpp` 原样交叉编译后在板卡跑通了完整链路**：读 BMP 序列 → GMC fit（CPU）→ fused OpenCL kernel（GPU）→ Ch0/Ch1/Ch2 输出。**未修改任何一行源代码**（`.bmp` 早已在其扩展名白名单内，`--kernel-dir` 本就是运行时参数）。

| 项 | 结果 |
| :--- | :--- |
| 跨架构精度（板卡 vs **x86 `gmc_stream.cpp` 纯 CPU 权威基线**，60 帧） | ✅ **PASS**，Ch0/Ch1 **Max\|Diff\|=0 全位精确**；Ch2 仅 1 帧 1 像素差 1 级 |
| 板卡帧率（含 CPU fit+warp+median） | **6.39 fps**（156.5 ms/帧） |
| 若只走 GPU 融合尾段 | **44.0 fps**（fit 17.3 + gpu 5.4 ms） |
| GPU 加速比（尾段） | **24.6×**（CPU 132.4 ms → GPU 5.4 ms） |
| 板卡 CPU vs x86 CPU（同源码单线程） | **慢 2.19×**（150.4 vs 68.7 ms） |

## 2. 为什么走 BMP（而不是 JPEG）

板卡 apt 只有 OpenCV 4.2.0，比 x86 的 4.10.0 低 8 个版本；交叉 sysroot 里 **334 个 `.so` 全是 gconv 字符集模块**，无任何 zlib/libjpeg/libpng 头文件。

🔴 **关键实测纠正**：我一度以为「x86 参考用 OpenCV 内建 libjpeg-turbo 3.1.2，用源码树自带的同版本即可同源」。**这是错的。**

```
ldd libopencv_imgcodecs.so.410 | grep jpeg
  → libjpeg.so.8 => /usr/lib/x86_64-linux-gnu/libjpeg.so.8
```

x86 包实际链接的是**系统 libjpeg-turbo 2.1.5**，build info 里那个 `build-libjpeg-turbo (ver 3.1.2-70)` 是 Debian 打包残留的配置串，与运行时链接的库无关。而 OpenCV 源码树自带的是 **3.0.3**，板卡系统是 **1.5.2（.so.62）/ 2.0.3（.so.8）**——**四者没有一对是匹配的**。

**BMP 彻底绕开这个问题**：
- OpenCV 的 BMP 编解码器（`modules/imgcodecs/src/grfmt_bmp.cpp`）由 `file(GLOB ... grfmt*.cpp)` **无条件编译**，只 include 自己的两个头文件，**零外部依赖**；
- BMP 无压缩，解码就是一次 memcpy 加 54 字节文件头，**不存在有损解码器的版本分歧**；
- 实测 x86 端往返无损：`bit-identical to source: True`，640×512 单帧 328,758 字节（裸数据 327,680，仅多 1,078 字节头）。

⇒ 序列在 x86 转成 BMP，两端读**同一份字节**，像素必然一致。60 帧共 19.7 MB，走 NFS。

**新增陷阱**：`imgcodecs` 一旦进 `BUILD_LIST`，`grfmt_jpeg2000_openjpeg.cpp` 也被无条件编译，而 OpenCV 自带的 OpenJPEG **没有**链进静态库 → 链接期报一堆 `undefined reference to opj_*`。必须显式 `-DWITH_OPENJPEG=OFF`。（这个坑只在**调用 `imread` 的程序**链接时暴露，模块列表里看不到。）

## 3. 链接期 OpenCL 可行（推翻了 memory 旧记录）

`ocl_host.h` 是**链接期**依赖 `libOpenCL.so.1`，而 memory 旧记录说「板卡 ICD 指向的 `libMaliOpenCL.so.1` 不存在」。

✅ **实测推翻**：板卡**有** ICD loader `/usr/lib/aarch64-linux-gnu/libOpenCL.so.1`（34,808 字节，2017 年），且在 ldconfig 里。虽然 `/etc/OpenCL/vendors/mali.icd` 指向的 `libMaliOpenCL.so.1` 确实不存在，但 **loader 本身能找到真正的驱动**。

链接期探针在板卡实跑：
```
arm_release_ver: g13p0-01eac0, rk_so_ver: 6
platform=ARM Platform context=OK
```

⇒ **`gmc_stream_ocl.cpp` 无需改成 dlopen 即可移植**。之前 `board_fused_accuracy` 用 dlopen 是因为它不能假定任何东西，现在证明 loader 可用。
（做法：把板卡 loader 的 .so 取到 x86 本地 `/media/manu/1TB-Volume/rk3588/boardlibs/` 供链接期 `-lOpenCL`，**不拷到板卡**；运行期由板卡自己的 ldconfig 解析。）

## 4. 跨架构精度对比（这是本轮最重要的结果）

### 4.1 对齐到 CPU 权威基线（真正的「与 gmc_stream.cpp 相同功能」检验）

⚠️ **首轮只做了「板卡 `gmc_stream_ocl` vs x86 `gmc_stream_ocl`」，那不够**——两端跑的是同一族二进制，跨的只是 OpenCL 后端与架构，**从未与 CPU-only 的 `gmc_stream.cpp` 对账**。目标要求「与 gmc_stream.cpp 相同功能」，基线必须是那个纯 CPU 权威实现。

补跑 x86 `gmc_stream.cpp`（已确认为纯 CPU：0 处 OpenCL 引用），同一份 BMP、同一组参数：

**板卡（Mali-G610 + 交叉编译 OpenCV）vs x86 `gmc_stream.cpp`（纯 CPU 权威参考）**：

| 通道 | Max\|Diff\| | MAE | 位精确 % | 超门限像素 | 判决 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Ch0 | **0** | 0.000000 | 100.0000% | 0 | **PASS** |
| Ch1 | **0** | 0.000000 | 100.0000% | 0 | **PASS** |
| Ch2 | 1 | 0.000000 | 100.0000% | 0 | **PASS** |

**59/60 帧三通道完全位精确**；唯一差异是第 41 帧 Ch2 的**单个像素**（`y=2 x=206`：x86=96，板卡=97）。
🔴 **该差异是孤立的**：前后帧（40、42）完全一致。若上游 fit 系统性漂移，差异会持续累积并逐帧扩大。⇒ 这是**单像素舍入边界抖动**（该像素 Ch2 值恰落在 96.5 附近，两端 float 路径末位差异导致 `cv::round` 取向不同），**不是系统性分歧**。
transform 矩阵 `(21,6) float32`：**126/126 条目全等**，Max\|Diff\|=0。

**顺带的交叉验证**：x86 上 `gmc_stream.cpp`（纯 CPU）与 `gmc_stream_ocl` 的 MD5 完全相同（`4b8b589ec44ed82ee7aaffc74a8d7d2e`）⇒ 两个实现互为印证。

⇒ **三条链路闭合**：`板卡 GPU` ≈ `x86 gmc_stream.cpp (CPU)` ≈ `x86 gmc_stream_ocl`，且板卡与两者的差异都只有那 1 个像素。

### 4.2 与逐算子测试的差异（重要）

`opencv_parity` 逐算子比较时 `warp_dst` 有 **11/19.3M** 像素差 1~2 级；放到完整流水线里只剩 **1 个**像素浮出。⇒ **逐算子是更严格的门，端到端通过并不代表每个算子位精确。**

## 5. 性能（每项重复 3 次取中位数，640×512，单线程）

完整矩阵——**四条配置**，含 CPU 权威基线与板卡纯 CPU 臂，才能把「架构差异」和「CPU vs GPU 差异」分开看：

| 配置 | fit | warp | median | GPU 尾段 | total | fps |
| :--- | --- | --- | --- | --- | --- | --- |
| **板卡 `gmc_stream_ocl` GPU 臂** | 17.34 | 67.37 | 67.23 | **5.38** | **156.50** | **6.39** |
| 板卡 `gmc_stream_ocl` CPU 臂 | 21.51 | 67.69 | 64.73 | — | 150.36 | 6.65 |
| x86 `gmc_stream_ocl`（PoCL） | 7.49 | 27.04 | 36.34 | 106.63 | 176.52 | 5.67 |
| **x86 `gmc_stream.cpp`（纯 CPU 权威）** | 10.15 | 27.20 | 33.03 | — | **68.69** | **14.56** |

### 三条可用结论

1. **GPU 融合尾段加速 24.6×**：板卡上 CPU 尾段 132.4 ms → GPU 5.38 ms，其中 kernel 本体 **3.84 ms/帧**。
2. 🔴 **板卡 CPU 比 x86 CPU 慢 2.19×**（150.4 vs 68.7 ms）。这是同一份源码、同一份输入、单线程下的纯架构差异：x86 有 AVX2/AVX512 运行时 dispatch，板卡只有 NEON。**这是选择 Mali GPU 融合的现实依据**——CPU 路径在板卡上并不划算。
3. 🔴 **x86 上 GPU 比 CPU 慢（0.59×）**，因为那里是 **PoCL 用 CPU 模拟**。同二进制在两块硬件上 106.6 ms vs 5.4 ms，**差 19.8 倍**，再次坐实 memory 铁律「PoCL/CPU 模拟时延严禁当作板卡性能」。

### 瓶颈分析

板卡 156.5 ms/帧的构成：**CPU 的 warp+median 占 86.1%**（134.6 ms），fit 占 11.1%，**GPU 只占 3.4%**（5.4 ms）。

⇒ **当前瓶颈是 CPU 的 warp+median，不是 GPU，也不是 fit。**
- 只用 GPU 融合尾段：`fit 17.3 + gpu 5.4 = 22.7 ms` → **44.0 fps**
- 但只要 median 还在 CPU 上，就回不到这个数——**median 是下一处该动的地方**（21 元素选择，67 ms）。

⚠️ **单线程说明**：`cv::setNumThreads(1)` 是位精确性的硬性要求（`gmc_stream_ocl.cpp:631` 明确写了多线程下 RANSAC 与 LK 不可复现），所以 6.4 fps 是**正确性优先**的数字，**不是性能上限**。板卡 8 核（4×A76 + 4×A55），若能安全放开线程，median 和 warp 的并行收益最大。

⚠️ **`total` 与分段和不等**：`total` 是 `push()` 整个调用的墙钟（`gmc_stream_ocl.cpp:780` 从 `mark0` 起算），含 `imread`、状态维护与 CSV 写盘。板卡差 4.6 ms、x86 差 105.6 ms（**PoCL 初始化**）。引用时必须说明是哪一种，否则会得出「x86 外围开销是板卡 23 倍」这种误导结论。

## 6. 交付物

`manu/pipeline/opencl/board/`：

| 文件 | 作用 |
| :--- | :--- |
| `build_gmc_ocl_arm64.sh` | **交叉编译 `gmc_stream_ocl.cpp`**（含链接期 OpenCL 的全部说明） |
| `convert_seq_to_bmp.py` | x86 侧 JPEG 序列 → BMP，自带往返无损校验 |
| `compare_board_vs_x86.py` | 跨架构逐通道对比，含分歧起点定位 |
| `build_opencv_arm64.sh` | 加了 `imgcodecs` + `-DWITH_OPENJPEG=OFF` |
| `build_opencv_parity.sh` | 双端 parity 构建（链接顺序需含 `-lopencv_imgcodecs`） |

`manu/pipeline/cpp/CMakeLists.txt`：新增 `gmc_stream_ocl` 目标（与 CPU-only 的 `gmc_stream` 分开，后者不需要 OpenCL）。

产物：
- arm64 二进制 `out/gmc_stream_ocl_arm64`（8,103,600 字节，静态链 OpenCV，动态依赖仅板卡已有的 `libOpenCL.so.1` + 系统库）
- x86 参考 `build-ocl/gmc_stream_ocl`
- 板卡 `/mnt/manu/ocvparity_exec/{gmc_ocl, kernels/}`

数据（NFS，`/mnt/manu/ocvparity/`）：`bmp/` 60 帧、`dump_x86/`、`dump_arm64/` 各 57 MB。

## 7. 遗留未解

1. **未跑多序列**。仅 1 个序列 60 帧（`01_4485_1167-2666`）。
2. **单线程**。`cv::setNumThreads(1)` 是位精确性的硬性要求（`gmc_stream_ocl.cpp:631` 明确写了多线程下 RANSAC 和 LK 不可复现），所以 6.4 fps 是**正确性优先**的数字，不是性能上限。若要提帧率需先解决可复现性。
3. **未测 `--mode nogmc` 对照臂**，也未测 `--anchor-step 2`（21 锚点深度 0）配置。
4. **`out-dir`（写 JPG）未验证**——arm64 的 imgcodecs 没编 JPEG 编码器，会失败。精度对比走的是 `--dump-dir` 的 `.npy`，不受影响。
5. **单像素舍入抖动的根因未定位**：Ch2 第 41 帧那 1 个像素走的是 `max(0, I - B)` 之后 `astype(uint8)` 的舍入，具体是 LK 末位还是 `cv::max/min` 的 SIMD 路径差异，未做消融。
