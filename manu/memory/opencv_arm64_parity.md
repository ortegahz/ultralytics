# OpenCV 4.10.0 arm64 交叉编译与跨架构一致性验证

> 2026-10-09 实测，非推测。全部结论在 RK3588 板卡 `evm3588` 上实跑得到。

## 1. 结论摘要

**arm64 OpenCV 4.10.0 交叉编译成功，并已在板卡实跑通过。与 x86 参考版的跨架构一致性实测结果：**

| 算子 | 判决 | 实测差异（59 帧 × 19,333,120 元素） |
| :--- | :--- | :--- |
| `cv::resize` INTER_LINEAR | **位精确** | 0 |
| `cv::goodFeaturesToTrack`（Shi-Tomasi） | **位精确** | 0 |
| `cv::calcOpticalFlowPyrLK` status 标志 | **位精确** | 0 |
| LK 输入点 `p0` | **位精确** | 0 |
| `cv::estimateAffinePartial2D` RANSAC 内点掩码 | **位精确** | 0 |
| LK 输出亚像素位置 `p1` | 差异 | max **1.06e-4 像素**，mean 2.65e-7 |
| LK min-Eig 误差 `err` | 差异 | max 2.08e-3，mean 2.49e-6 |
| RANSAC 矩阵 `affine_M`（CV_64F） | 差异 | max **1.64e-5**，mean 2.07e-7 |
| `fit_H`（float32 + ds 缩放） | 差异 | max 3.27e-5，mean 4.13e-7 |
| `cv::warpAffine` INTER_LINEAR REFLECT | **基本位精确** | 56/59 帧完全一致；**11/19,333,120 像素不同（0.000057%）**，最大 2/255 |
| 21 元素 uint8 median | **基本位精确** | 38/39 帧完全一致；**1/12,779,520 像素不同（0.000008%）**，最大 1/255 |

⇒ **工程判决：arm64 OpenCV 4.10.0 可用于特征通道。** RANSAC 内点掩码 59/59 帧位精确，说明 GMC 的判定（哪些点算内点）在两架构上完全一致；差异停留在 LK 亚像素坐标的 float32 末位，衰减到 8-bit warp 输出时只剩万分之 0.57 的像素差 1~2 个灰度级。

## 2. 构建了什么

| 项 | 值 |
| :--- | :--- |
| 源码 worktree | `/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64`（`git worktree` 自 `4.10.0` tag，HEAD `71d3237a09`） |
| 用户原有 5.x 树 | `/media/manu/1TB-Volume/workspace/opencv`（**未被动过**） |
| 构建目录 | `/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-build` |
| 安装前缀 | `/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install` |
| 模块 | `core imgproc features2d video calib3d flann`（`BUILD_LIST`） |
| 第三方 | **全关**：IPP / OpenCL / FFMPEG / GTK / QT / JPEG / PNG / TIFF / WEBP / OPENEXR / JASPER / LAPACK / EIGEN / TBB / ITT |
| 链接形态 | **静态库**（`BUILD_SHARED_LIBS=OFF`），板卡无需装任何库 |
| 构建耗时 | **约 4 分钟**（`-j4`） |

产物：
```
libopencv_core.a        7,040,352
libopencv_imgproc.a     7,336,702
libopencv_calib3d.a     5,524,480
libopencv_features2d.a  1,667,774
libopencv_flann.a       1,370,614
libopencv_video.a       1,006,278
libtegra_hal.a          1,133,824   (carotene NEON SIMD)
libzlib.a                 160,084
```

## 3. 为什么只建 6 个模块

交叉 sysroot 有 334 个 `.so`，但**全是 gconv 字符集模块**（`ANSI_X3.110.so`、`ARMSCII-8.so` …），**没有 zlib/libpng/libjpeg/libtiff 的头文件**。`imgcodecs` 无法直接构建。

因此精度测试**用原始像素随包下发**（x86 侧解码一次），从根上绕开 `imgcodecs`。这也是为什么 `gen_opencv_parity_case`（x86 专用，需要 imgcodecs）与 `opencv_parity`（双端通用，不需要）拆成两个程序。

顺带证实：板卡运行时**有** `libjpeg.so.62` / `libpng16.so` / `libtiff.so.5`，头文件有 `zlib.h` / `png.h`，但**缺 `jpeglib.h` / `tiffio.h`**。

## 4. 三个真实构建陷阱（都会静默坑人）

### 4.1 CMake 4.x 直接拒绝 OpenCV 4.10

系统 CMake 是 **4.2.3**。OpenCV 4.10.0 顶层 `CMakeLists.txt:19` 是 `cmake_minimum_required(VERSION 3.1 FATAL_ERROR)`，CMake ≥ 4.0 对低于 3.5 的兼容级别**硬报错**。

⇒ 解压 CMake **3.31.6** 到 `/media/manu/1TB-Volume/workspace/tools/` 使用。**没有改 OpenCV 源码**，保持 worktree 与 4.10.0 tag 逐字节一致（改了会污染与 x86 参考的可比性）。

### 4.2 链接缺 `carotene_o4t::split*` —— 实际库名叫 `tegra_hal`

`opencv_core` 无条件引用 carotene（NEON SIMD）的 `split3/split4` 等 HAL 入口。报错信息指向 `carotene_o4t::` 命名空间，看起来像缺第三方依赖，**但 OpenCV 把这个目标的库名定为 `tegra_hal`**（NVIDIA Tegra 时代遗留），符号却命名空间化为 `carotene_o4t`。

它装在 **`lib/opencv4/3rdparty/libtegra_hal.a`**，不是 `lib/`——因为 `ocv_install_target` 把第三方目标统一路由到 `opencv4/3rdparty/`。651 个 carotene 符号。

⇒ 链接加 `-L$INSTALL/lib/opencv4/3rdparty -ltegra_hal`。

### 4.3 链接缺 `gzopen/gzread` —— 需要 `-lzlib`

即使 `WITH_JPEG=OFF`，`core/src/persistence.cpp`（FileStorage 的 `.gz` XML 支持）**无条件**引用 zlib 符号。库文件名是 `libzlib.a`，所以是 `-lzlib` 而非 `-lz`。

## 5. 浮点契约：`-ffp-contract=off` 必须保留，但我原先的推理是错的

构建全程施加 `-ffp-contract=off -fno-fast-math`（项目铁律）。

**我当时推理**：「x86 基线无 FMA，所以 x86 即使 `-ffp-contract=fast` 也等价于 off；arm64 有 `fmadd` 会融合，所以必须显式关掉。」

**这个推理只有一半对，而且实测证明是错的。** 见 §6。

实测到的 x86 参考版 build info：
- 编译器 **gcc 15.2.0**，flags 里**没有** `-ffp-contract=off`（用 GCC 默认 `fast`）
- `Baseline: SSE SSE2`（基线无 FMA，这部分我的推理对）
- **但有 runtime dispatch**：`AVX2 (34 files): + ... FMA3 ...`、`AVX512_SKX (5 files): + ... FMA3 ...`
- 运行时 `getCPUFeaturesLine()` = `SSE SSE2 *SSE4.1 *SSE4.2 *FP16 *AVX *AVX2 *AVX512-SKX?` ⇒ **实际执行的是带 FMA3 的 AVX2/AVX512 路径**

⇒ 所以 x86 参考版**确实会用到 FMA**，基线无 FMA 不代表执行路径无 FMA。这一点我原先没看到 dispatch 层。

## 6. FMA 假设被实测证伪 🔴

**实验**：在同一台 x86 上用 `OPENCV_CPU_DISABLE=AVX2,AVX512_SKX,FP16,SSE4_1,SSE4_2` 关掉所有带 FMA3 的 dispatch 层，只留 AVX 路径，重跑同样 59 帧。

**结果：13 个 stage 全部 `IDENTICAL`，max|diff| = 0，59/59 帧位精确。**

⇒ **LK 的跨架构差异与 FMA 无关。** 同架构下开/关 FMA3 输出完全相同，证明 OpenCV 的 LK 代码路径在这两个 dispatch 之间没有数值差别。

（注意 `OPENCV_CPU_DISABLE=FMA3` 是**无效**的——FMA3 不是独立 dispatch 名，只是 AVX2 的子特性。必须禁用 `AVX2` 本身。禁用后 `getCPUFeaturesLine` 显示 `*AVX2?`，`?` 后缀表示已禁用。）

**真实原因**（未能进一步定位到单条指令，属诚实记录）：x86 与 arm64 走的是**架构特定的 SIMD 实现**（x86 SSE/AVX intrinsics vs aarch64 NEON/carotene intrinsics，迭代中的求和顺序与中间量精度不同），叠加**编译器版本差 6 年**（gcc 15.2.0 vs gcc 9.3.0，auto-vectorizer 决策不同）。这两项都不可消除。

## 7. 🔴 自我更正：23 帧样本给出了错误结论

**第一轮只用 23 帧**，得到 `warp_dst 23/23 IDENTICAL`、`median_out 3/3 IDENTICAL`，我当时据此判断「LK 的差异没有传播到 warp 输出」。

**扩到 60 帧（59 次 fit）后这个结论被推翻**：

| stage | 23 帧（错） | 59 帧（真） |
| :--- | :--- | :--- |
| `warp_dst` | 23/23 IDENTICAL | **56/59 IDENTICAL**，3 帧有差异 |
| `median_out` | 3/3 IDENTICAL | **38/39 IDENTICAL**，1 帧有差异 |

差异极稀疏（warp 11/19.3M 像素，median 1/12.8M 像素），23 帧（7.5M 像素）**恰好没采样到**这 4 帧。

⇒ **教训：稀疏事件的"零观测"不等于"不存在"。** 任何"全一致"的判决都必须附带样本量，且稀疏差异要用**跨全部元素的绝对计数**报告，不能只报"逐帧是否一致"。

⇒ 差异定位到具体帧：`warp_dst` 帧 26（2 px）、帧 51（1 px）、帧 56（8 px）；`median_out` 帧 33（1 px）。

## 8. ULP 指标在近零处会骗人

`affine_M` 报出 `max 72934002268285 ulp`，看起来像灾难，实际 `max|diff| = 1.64e-5`。

原因：ULP 是「相邻可表示数的间隔」。在 0 附近 float64 的间隔极小，两个都接近 0 的数可以差极小的绝对值却有天大的 ULP。

⇒ `compare_opencv_parity.py` 现在会自动标注这类行，并要求**以 `max|diff|` 为准**。`fit_H`（135850 ulp / 3.27e-5）、`lk_curr`（1160 ulp / 1.06e-4）同理。

## 9. 板卡 ABI 兼容性（实测）

| 项 | 值 |
| :--- | :--- |
| 板卡 glibc | **2.31**（Ubuntu 20.04.6，`ldd --version`） |
| 交叉 sysroot glibc | **2.29**（`sysroot/lib/libc-2.29.so`） |
| 方向 | ✅ 正确：低版本 sysroot 编的可在高版本运行时上跑（glibc 向后兼容） |
| 实测 | `ldd` 全部解析，`libpthread.so.0` / `libm.so.6` / `libc.so.6` 无 `not found` |

三套交叉前缀实测：
- `aarch64-linux-gcc`：只有 `aarch64-linux-gcc`（无 `-gnu-`），**无 sysroot**
- **`aarch64-rockchip-linux-gnu-`：✅ 采用**（唯一有 sysroot 的）
- `aarch64-rockchip930-linux-gnu-`：**无 sysroot 目录**，能链接但不提供 glibc

⚠️ 注意 summary 里「板卡对应 `aarch64-rockchip930-`」这条**是错的**（来自早期推断）。实测 `aarch64-rockchip930-linux-gnu` 连 sysroot 目录都没有。已更正于 `rules.md`。

## 10. arm64 侧运行时自述（板卡实跑输出）

```
OpenCV version : 4.10.0
numThreads     : 1
CPU count      : 8
CPU features   : NEON FP16 *NEON_DOTPROD *NEON_FP16
RNG state      : 0xffffffff
Target         : Linux aarch64
3rdparty       : zlib tegra_hal
Custom HAL     : carotene (ver 0.0.1, Auto detected)
Modules built  : calib3d core features2d flann imgproc video
C flags        : -ffp-contract=off -fno-fast-math ... -O3 -DNDEBUG
```

**`RNG state 0xffffffff` 与 x86 参考完全一致** ⇒ 两端 RANSAC 消耗的随机序列对齐，这是 `affine_inl`（内点掩码）59/59 位精确的前提。

⚠️ x86 参考带 `-flto=auto`，arm64 交叉编译**未开 LTO**（OpenCV 默认不开）。这是又一个不可控差异，未单独消融。

## 11. 交付物

`manu/pipeline/opencl/board/`：

| 文件 | 作用 |
| :--- | :--- |
| `build_opencv_arm64.sh` | OpenCV 4.10.0 arm64 交叉编译（含版本守卫，硬拒非 4.10.0） |
| `rk3588-aarch64-toolchain.cmake` | toolchain 文件，携带浮点契约与 sysroot 隔离说明 |
| `gen_opencv_parity_case.cpp` | x86 专用：JPEG → raw 灰度帧 |
| `opencv_parity.cpp` | 双端通用：复刻 `gmc_stream_ocl.cpp` 语义，逐算子 dump |
| `build_opencv_parity.sh` | 双端构建 |
| `compare_opencv_parity.py` | 逐算子对比 + ULP 假象标注（**需 anaconda 的 numpy**） |
| `push_and_run.py` | scp 推送 + SSH 执行（**检查 scp 退出码**） |

数据与产物全在 NFS `/mnt/manu/ocvparity/`（`in2/` 60 帧输入、`out_x86b/`、`out_x86b_base/`、`out_arm64b/`，共约 202 MB）。

⚠️ **自我更正：NFS 目录物理上仍然占板卡磁盘。** 我最初写「板卡本地磁盘零占用」是**错的**。`/mnt/manu` 与 `/` 是**同一个文件系统**（`df -h / /mnt/manu` 两行完全相同，均为 `/dev/root`），本轮实测板卡可用空间从 **1.7G → 1.5G**。

走 NFS 的真实收益是**省掉传输**而不是省磁盘：
- x86 直接在 `/home/manu/mnt/nfs/ocvparity/` 读板卡的对比结果，**不必把几十 MB dump 再 scp 回 x86**；
- 板卡也不必为了「把结果送回去」而在本机再存一份；
- 对比每轮重跑时，NFS 只需增量写入，scp 方案则每次都要重传全部 dump。

⇒ 但**体积仍需估算**（202 MB 已吃掉 1.7G 余量的 12%）。这一事实 `rules.md` 1b 早已记录（「模型与数据集放 `/mnt/manu` 前先估算 GiB」），本轮新写的文档一度与它不一致，已按 memory 铁律更正。

**板卡当前占用**：`/mnt/manu/ocvparity` 202M（NFS 数据）+ `/mnt/manu/ocvparity_exec` 7.5M（唯一的可执行文件，必须在本地盘，因为经 NFS 执行必报 `Text file busy`）。

⚠️ `/home/manu/mnt/nfs` 在 x86 侧需 `/home/manu/anaconda3/bin/python` 跑对比脚本——系统 `python3` 无 numpy。

## 12. 遗留未解

1. **LK 亚像素差异的根因未定位到指令级**。已排除 FMA。剩余候选：架构特定 SIMD 实现 + gcc 版本差（6 年）。若要进一步定位，需在 arm64 侧开 `-ffp-contract=fast` 做对照，代价是违反项目浮点契约。
2. **`affine_M` 的 ULP 指标失真**已通过标注绕过，未改成相对误差。
3. **未测 `-march` 提升**：当前 `-mcpu=cortex-a55`（OpenCV 默认保守选择，兼容全部 8 核）。若改用 A76 或 `armv8.2-a+fp16` 可能有更快路径，也可能改变数值。
4. **仅 1 个序列 60 帧**。跨序列泛化性未验证（用户此前 pending 的"多序列、两种分辨率、数十 case"统计仍未做）。
