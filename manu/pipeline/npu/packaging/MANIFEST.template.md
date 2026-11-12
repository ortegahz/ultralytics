> ℹ️ **这份文件怎么来的**：交付物 `MANIFEST.md` 由工程里的
> `{{SRC_ROOT}}/packaging/MANIFEST.template.md` 生成（占位符由脚本按实况填充；
> `MD5_TABLE` 标记之间的表由脚本自动生成；模板漏改一个，脚本会当场报错退出）。
> **请勿手改交付物。**
>
> ℹ️ **命令里的绝对路径（`/media/manu/...`）是作者机器上的**：换成你自己的包目录即可。

# MANIFEST.md —— 嵌入式组集成交付清单

> 目标板卡**不是**我们手上这块 RK3588 开发板。本清单把交付物分成
> **板卡无关**（直接可用）与**板卡相关**（必须由集成方按自己板子确认）两类。

---

## A. 包内容总览

包根目录：`manu/pipeline/npu/build/`（仓库内路径
`{{PKG_ROOT}}/`）。**交付给集成方的目录名是 `mosaic_sdk/`**（本机
`{{NFS_ROOT}}/mosaic_sdk/`，板卡侧 `/mnt/manu/mosaic_sdk/`），**内容与 `build/` 一致**。

### 交付到板卡后的目录结构

```
/mnt/manu/mosaic_sdk/
├── README.md / MANIFEST.md / build_lib.sh / build_demo.sh / copy_to_nfs.sh
├── src/
│   ├── npu/    mosaic_pipeline.cpp/.h, mosaic_core.h, mosaic_post_gpu.h,
│   │           mosaic_api_test.cpp, mosaic_client_example.cpp,
│   │           mosaic_async_client_example.cpp,
│   │           mosaic_contract_test.cpp, rknn_mosaic_e2e.cpp
│   ├── cpp/    gmc_stream_ocl.cpp, md5.h
│   └── opencl/host/   ocl_host.h, post_local_max.cl
├── include/    rknn_api.h, stb_image.h, stb_image_resize.h, stb_image_write.h,
│               rga/×18, CL/×20
├── runtime/kernels/    warp_median_fused.cl, sortnet_generated.inc, gmc_feature_gpu.cl
├── out/        mosaic_pipeline.o, libmosaic_pipeline.so,
│               mosaic_api_test, mosaic_client_example,
│               mosaic_async_client_example,
│               mosaic_contract_test, rknn_mosaic_e2e
└── 兼容别名（相对符号链接）：
    mosaic_api_test       -> out/mosaic_api_test
    mosaic_async_client_example -> out/mosaic_async_client_example
    mosaic_contract_test  -> out/mosaic_contract_test
    libmosaic_pipeline.so -> out/libmosaic_pipeline.so
    mosaic_pipeline.h     -> src/npu/mosaic_pipeline.h
    kernels               -> runtime/kernels
```

* `mosaic_sdk/` 是 `build/` 的**整树镜像**：源＝包目录下的每个普通文件，
  目标＝**同名相对路径**（`src/`、`include/`、`runtime/kernels/`、`out/`
  整树照搬）。
* 顶层那 5 个**兼容别名是符号链接**（不是副本），分别指向 `out/` 与 `src/npu/`，
  为的是**旧的平铺命令**（`./mosaic_api_test --kernel-dir ./kernels`）继续可用。
* `out/` 的 6 个产物在交付包里**已预编译**（RK3588 / aarch64），嵌入式组
  **不必**先编译即可直接跑；确实要重编时在交付目录原地编（见 §B、§F）。
* ⚠️ **本包不携带任何第三方 .so/.a**（**没有第三方依赖目录**）：OpenCV /
  RKNN Runtime / RGA / OpenCL 全部**动态链接板卡自带的库**（见 §C 的
  **板卡依赖矩阵**）。库本身以**共享库** `out/libmosaic_pipeline.so` 交付
  （SONAME = `libmosaic_pipeline.so`），五个测试程序用
  `-lmosaic_pipeline` + `-Wl,-rpath,'$ORIGIN'` 链接它 ⇒ 整包可搬、可直接跑。

<!-- MD5_TABLE:BEGIN -->
（本段由 `make_build_dir.sh` 自动生成，勿手改）
<!-- MD5_TABLE:END -->

**关于 md5 基线**：

* `src/**`、`include/**`、`runtime/kernels/**` 的 md5 是**固定基线**——可用来核对
  交付是否被改动。
* `out/` 段**刻意不列 md5**：产物是**构建产物**，随工具链/编译路径变化，
  列进来只会误导人。需要核对产物时手动 `md5sum out/*`。
  本版交付的是**共享库** `out/libmosaic_pipeline.so`（不再是 `.a` 归档）
  ⇒ **没有 `ar` 归档时间戳那种「每次构建都变」的问题**，在**同一个包根路径**
  下重编，`.so` 与五个可执行文件的 md5 都是稳定的。
* 🔴 **但产物 md5 与「包根的绝对路径」绑定**：编译时 `__FILE__` 把路径烤进二进制
  ⇒ **集成方在自己的目录重编，md5 会与 README 顶部那张表不同**（**行为完全一样**，
  已实测：换路径编出来的 `mosaic_client_example` 在板卡上 6 帧 12 检出、检测框
  逐字段一致）。想要**跨机器一致**的 md5，用**相对路径**编（README「md5 的三个
  已知差异」一节给了完整命令，已实测两个不同目录编出来逐字节相同）。
* `MANIFEST.md` 自身的 md5 无法自含（自引用），请单独用 `md5sum MANIFEST.md` 核对。

---

## B. 板卡无关的部分（直接可用，拿到就能编）

| 类别 | 文件 | 说明 |
|---|---|---|
| 库实现 | `src/npu/mosaic_pipeline.cpp` | 库的编译单元 |
| 公开 ABI | `src/npu/mosaic_pipeline.h` | 客户写代码时 include 它（C ABI） |
| 内部头 | `src/npu/mosaic_core.h` | GMC / RGA / RKNN 胶水层 |
| 内部头 | `src/npu/mosaic_post_gpu.h` | GPU 后处理路径（头内已内嵌同一份 kernel 源码） |
| 引擎 | `src/cpp/gmc_stream_ocl.cpp` | GMC/OpenCL 引擎 |
| 工具 | `src/cpp/md5.h` | MD5 实现 |
| OpenCL host | `src/opencl/host/ocl_host.h` | OpenCL 封装 |
| OpenCL 副本 | `src/opencl/host/post_local_max.cl` | GPU 后处理 kernel 的**可读副本**（运行时**不需要**） |
| 示例 demo | `src/npu/mosaic_api_test.cpp` | 只调公开 C ABI 的示例 |
| 最小集成示范 | `src/npu/mosaic_client_example.cpp` | **嵌入方照着抄的最简调用骨架**（open / set_post_mode / push / collect / destroy） |
| 多线程生产-消费模板 | `src/npu/mosaic_async_client_example.cpp` | **工程级集成骨架**：相机线程不等算法 + Worker 严格一推一收 + 业务方自己的回调 |
| 契约测试 | `src/npu/mosaic_contract_test.cpp` | 库的验收门禁 |
| 端到端测试程序 | `src/npu/rknn_mosaic_e2e.cpp` | **吞吐/延时测量**（12.9 fps 那个数就是它测的），只调公开 C ABI |
| 第三方头 | `include/rknn_api.h` | RKNN API |
| 第三方头 | `include/stb_image.h`、`include/stb_image_resize.h`、`include/stb_image_write.h` | stb 图像库 |
| 第三方头 | `include/rga/*`（18 个） | RGA API 头 |
| 第三方头 | `include/CL/*`（20 个） | Khronos OpenCL 头 |
| **kernel 资产** | `runtime/kernels/warp_median_fused.cl` | **运行时必需** |
| **kernel 资产** | `runtime/kernels/sortnet_generated.inc` | **运行时必需** |
| kernel 资产 | `runtime/kernels/gmc_feature_gpu.cl` | 可选（仅探针用，库不加载） |
| 构建脚本 | `build_lib.sh` / `build_demo.sh` | 交叉编译库 / 测试程序 |
| 一键脚本 | `copy_to_nfs.sh` | 编译 + 整树拷贝到 NFS + md5 回验 +（可选）板卡验证 |
| 文档 | `README.md` / `MANIFEST.md` | 使用说明 / 本清单 |

**交付包包含源码（`src/**`）、第三方头（`include/**`）、kernel 资产
（`runtime/kernels/**`）、构建脚本（`build_lib.sh` / `build_demo.sh`）与预编译
产物（`out/**`）** ⇒ 集成方**可以在交付目录原地重编**（构建脚本按脚本自身
所在目录定位，不依赖仓库里的其它文件）：

```bash
cd <交付目录>            # 例如 /mnt/manu/mosaic_sdk
bash build_lib.sh        # → out/libmosaic_pipeline.so（共享库）
bash build_demo.sh       # → out/mosaic_api_test, out/mosaic_client_example,
#                            out/mosaic_contract_test, out/rknn_mosaic_e2e
```

> ⚠️ **重编需要集成方提供两样东西**：
>
> ① **交叉编译器**（编译器没法随包发）：
>
> ```bash
> export RK3588_TOOLCHAIN=/path/to/toolchain-root   # 可用 ':' 分隔多个候选，按顺序取第一个能用的
> export CROSS_CXX=/path/to/aarch64-linux-gnu-g++   # 可选：直接指定编译器
> export CROSS_AR=/path/to/aarch64-linux-gnu-ar     # 可选：直接指定 ar
> ```
>
> ② **板卡自带的第三方库**（本包**不携带任何第三方 .so/.a**，见 §C 的
> **板卡依赖矩阵**）：
>
> ```bash
> export OPENCV_INCLUDE_DIR=<板卡 OpenCV 头目录>      # 板卡默认 /usr/include/opencv4
> export OPENCV_LIB_DIR=<板卡 OpenCV 动态库目录>      # 板卡默认 /usr/lib/aarch64-linux-gnu
> export BOARD_LIBS=<板卡 libOpenCL / librga 目录>    # 板卡默认 /usr/lib/aarch64-linux-gnu
> export RT_LIB_DIR=<板卡 librknnrt 目录>             # 板卡默认 /usr/lib
> ```
>
> ⚠️ **交叉编译在宿主机上做时，上面这些路径指的是「目标板卡的文件系统」**：
> 宿主机的 `/usr` 是 x86 的，**绝对不能直接用**（会污染构建）⇒
> 要先把板卡的那份（头 + `.so`）取回宿主机，再把变量指过去。

**源码与头文件不含任何板卡私有路径**——`src/**`、`include/**`、
`runtime/kernels/**` 里只有相对 include 与公开 API 声明，换一块 RK3588 板子，
只要 BSP 里有同版本的交叉工具链与**同版本的 OpenCV**，照样能编译。

> ⚠️ **kernel 目录的相对布局不能改**：源码用相对 include
> （`mosaic_pipeline.cpp` → `../cpp/gmc_stream_ocl.cpp` → `../opencl/host/ocl_host.h`），
> 所以 `src/` 下的三层目录结构必须原样保留。

### 运行时 kernel 的真实要求（重要）

库在运行时**只加载 2 个** kernel 文件（`gmc_stream_ocl.cpp` 的 `FusedGpu` 构造：

```cpp
const std::string ksrc = ocl::read_file(kernel_dir + "/warp_median_fused.cl");
const std::string isrc = ocl::read_file(kernel_dir + "/sortnet_generated.inc");
```

| 文件 | 库运行时是否必需 | 说明 |
|---|:--:|---|
| `warp_median_fused.cl` | ✅ **必需** | 加载失败 ⇒ 建流水线直接失败 |
| `sortnet_generated.inc` | ✅ **必需** | 加载失败 ⇒ 建流水线直接失败 |
| `gmc_feature_gpu.cl` | ❌ 不需要 | 库**不加载**它；只有独立探针 `gmc_gpu_probe` 用 |

⇒ 目标板卡的 `--kernel-dir` 指向的目录里，**至少要有前两个**。

---

## C. 板卡相关的部分（第三方运行库由板卡提供；模型与帧数据仍由集成方自备）

**下面「我们这边用的」那一列是我们验证过的版本，供对照。**

> ⚠️ **本轮策略变更（2026-11-11）**：交付包**不再携带任何第三方 .so/.a**
> ⇒ OpenCV / RKNN Runtime / RGA / OpenCL **全部动态链接板卡自带的库**
> （嵌入式组的要求：直接用板卡上原来就有的 `.so`）。
> 「要不要把某个库装到你们板子上」仍然是集成方自己的决定
> （板卡 BSP 里通常已经有了）。

### 板卡依赖矩阵（本包**不携带任何第三方 .so/.a**）

| 库 | 板卡路径 | SONAME | 版本 |
|:--|:--|:--|:--|
| OpenCV core/imgproc/imgcodecs/calib3d/features2d/flann/video | `/usr/lib/aarch64-linux-gnu/libopencv_*.so.406` | `libopencv_core.so.406` 等 | **OpenCV 4.6.0** |
| OpenCV 头 | `/usr/include/opencv4` | — | 4.6.0 |
| RKNN Runtime | `/usr/lib/librknnrt.so` | `librknnrt.so` | **2.2.0** |
| RGA | `/usr/lib/aarch64-linux-gnu/librga.so.2` | `librga.so.2` | rga_api 1.10.0_[9] |
| OpenCL | `/usr/lib/aarch64-linux-gnu/libOpenCL.so.1` | `libOpenCL.so.1` | OpenCL 3.0（Mali-G610） |

* **编译期**：需要板卡 OpenCV 的**头**（`/usr/include/opencv4`，含 `opencv2/`）。
  ⚠️ **头与运行库必须同版本**：本包用板卡 **4.6.0** 的头去链板卡 **4.6.0** 的
  `.so.406` ⇒ **不要拿 4.8 的运行库配 4.6 的头**（OpenCV 次版本会改 C++ ABI）。
* **链接期**：需要那 7 个 `libopencv_*.so.406` + `librknnrt.so` + `librga.so.2`
  + `libOpenCL.so.1`（用 `OPENCV_INCLUDE_DIR` / `OPENCV_LIB_DIR` / `BOARD_LIBS` /
  `RT_LIB_DIR` 定位；默认值就是板卡的系统路径）。
* **运行期**：上面这些**全部由板卡 `ldconfig` 解析**（板上 `ldconfig -p` 已实测
  全部可解析）⇒ **不需要设 `LD_LIBRARY_PATH`**。
  唯一需要 `-rpath` 的是**本包自己的** `libmosaic_pipeline.so`：五个测试程序用
  `-Wl,-rpath,'$ORIGIN'` 找**同目录**的那一份 ⇒ **整包可搬**。

> ⚠️ **交叉编译在宿主机上做时，上表那些路径指的是「目标板卡的文件系统」**：
> 宿主机的 `/usr` 是 **x86 的**，**绝对不能直接用**（会拿 x86 的头/库去编
> aarch64 ⇒ 编译失败或产出坏二进制）。
> ⇒ **先把板卡的那份取回宿主机**，再把变量指过去。

| 项 | 我们这边用的（仅供对照） | 集成方要做什么 |
|:--|:--|:--|
| `librknnrt.so` | RKNN Runtime **2.2.0**，板卡路径 `/usr/lib/librknnrt.so`，SONAME `librknnrt.so` | **由板卡提供**（本包不带）。运行时用你们板子 BSP 里的那份；**版本必须匹配 `include/rknn_api.h`**（我们验证的是 2.2.0）。若板子上没有 ⇒ 从 RKNN BSP 里取一份装上 |
| `librga.so.2` | 板卡路径 `/usr/lib/aarch64-linux-gnu/librga.so.2`，SONAME `librga.so.2`，`rga_api version 1.10.0_[9]` | **由板卡提供**（本包不带）。其 API 头 `include/rga/*` 已随包交付 |
| `libOpenCL.so.1` | 板卡路径 `/usr/lib/aarch64-linux-gnu/libOpenCL.so.1`，SONAME `libOpenCL.so.1` | **由板卡提供**（本包不带）。用板卡自己的 OpenCL 驱动库（Mali-G610 / OpenCL 3.0） |
| OpenCV | 板卡路径 `/usr/lib/aarch64-linux-gnu/libopencv_*.so.406`，**4.6.0** | **由板卡提供**（本包不带）。⚠️ **头与库必须同版本** |
| 模型 `.rknn` | `{{NFS_ROOT}}/ptq/models/p0_off_A_none.rknn`，6,326,305 B，md5 `1a3faf2f9459cad8a389ffa81810234c` | 可直接用我们这份，也可换他们自己的模型。**换模型的形状约束见下表**（板卡实测） |
| 帧数据 | `VIDEO00007_19700101_001324`（640×512 BMP，命名 `frame_%06d.bmp`，共 11,810 帧），本板路径 `/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324` | 他们自己的相机流（同样须是 640×512 灰度、单通道） |

> **交叉编译器**仍属于对方环境：需要 aarch64 交叉工具链（我们用的是
> gcc-buildroot 9.3.0）。构建脚本里用 `RK3588_TOOLCHAIN`（可用 `:` 分隔多个候选）
> 或 `CROSS_CXX` 覆盖即可，不用改脚本。
>
> **OpenCV 也不属于本包**：它由**板卡**提供（头 `/usr/include/opencv4` +
> 运行库 `.so.406`）⇒ 编译期用 `OPENCV_INCLUDE_DIR` / `OPENCV_LIB_DIR` 指过去
> （默认值即板卡系统路径）。

### 换模型时的形状约束（板卡实测，非推测）

板卡上跑 `mosaic_api_test` 时库自己打印的实际张量：

```
[i] input 640x640x3 fmt=NHWC type=FLOAT16 zp=0 scale=1.00000000
[i] out[0] dims=[1,1,320,320] fmt=NCHW type=FLOAT16
[i] out[1] dims=[1,2,320,320] fmt=NCHW type=FLOAT16
[i] heatmap grid 320x320 (canvas 640x640, stride 2)
```

| 张量 | 形状 | 代码里的约束 |
|---|---|---|
| 输入 | **640×640×3，fp16（NHWC）** | 必须。库按 640×640 画布切 3 块 tile 再拼回 |
| 输出 0 | **[1,1,H,W]**（实测 320×320） | **硬校验**：`mosaic_core.h` 里 `dims[0]!=1 \|\| dims[1]!=1` 直接 `[FATAL] out[0] is not [1,1,H,W]` 退出。**不满足 ⇒ 建流水线就失败** |
| 输出 1 | 实测 **[1,2,320,320]** | 代码只要求 `n_output >= 1`，这一路**不参与硬校验** |

> ⚠️ **注意**：不是「两个输出都是 [1,1,320,320]」。实测第二路是 **2 通道**
> `[1,2,320,320]`。换模型时若第二路的通道数变了（1↔2），后处理语义要跟着改。

---

---

## D. ⚠️ 绑核参数是板卡特定的（换板子必改）

我们预设的旋钮是**按这块板子的核拓扑排的**：RK3588 = 4×A76(核 4-7) +
4×A55(核 0-3)。预设值写在 `mosaic_pipeline_config_preset()` 里
（`MOSAIC_PRESET_REALTIME` / `MOSAIC_PRESET_THROUGHPUT`）：

| 字段 | 我们的预设值 | 含义 |
|---|---|---|
| `cpu_affinity` | `"0-7"` | 全部 8 个核 |
| `gmc_affinity_group` | `"4-7"` | GMC 特征计算 → A76 大核 |
| `pre_affinity` | `"0-3"` | 预处理 → A55 小核 |
| `npu_stage_affinity` | `"6-7"` | NPU 阶段 → 2 个 A76 大核 |
| `gmc_anchor_threads` | `4` | GMC 锚点线程数 |
| `npu_workers` | `2` | NPU 上下文数（板子恰好支持 2 个） |

⇒ **换板子必须按他们自己的核数 / 拓扑重设**，照抄会绑到错误的核（或绑不到，
`sched_setaffinity` 失败 ⇒ 静默退化成不绑核，性能掉下来但不会报错）。

**改法**：`mosaic_pipeline_config_preset(&cfg, MOSAIC_PRESET_REALTIME)` 之后，
在 `mosaic_pipeline_config_set_paths()` 之后**直接覆盖**这几个字段：

```cpp
MosaicPipelineConfig cfg;
memset(&cfg, 0, sizeof(cfg));
mosaic_pipeline_config_preset(&cfg, MOSAIC_PRESET_REALTIME);
mosaic_pipeline_config_set_paths(&cfg, model, kernel_dir, sequence);

/* ↓ 换板子必须重设这一段 ↓ */
cfg.cpu_affinity       = "0-3";   /* 他们的全部核 */
cfg.gmc_affinity_group = "0-1";   /* 他们的性能核 */
cfg.pre_affinity       = "2-3";
cfg.npu_stage_affinity = "0-1";
cfg.gmc_anchor_threads = 2;
cfg.npu_workers        = 2;       /* 若他们的 NPU 只支持 1 个上下文 ⇒ 改 1 */
/* ↑ 改完再 open ↑ */
```

> 注意：`MOSAIC_PRESET_REALTIME` 与 `MOSAIC_PRESET_THROUGHPUT` 的**唯一**区别是
> `buffer_a_depth`（3 vs 64）；绑核值两者相同。`MOSAIC_PRESET_BASELINE` 不装任何
> 旋钮（就是冻结的默认值）。

---

## E. 运行时依赖（`readelf -d` 实测的 DT_NEEDED）

`out/libmosaic_pipeline.so`（共享库，SONAME = `libmosaic_pipeline.so`）的动态依赖：

```
libopencv_imgcodecs.so.406   ← 板卡自带的 OpenCV 4.6.0
libopencv_calib3d.so.406     ← 同上
libopencv_features2d.so.406  ← 同上
libopencv_flann.so.406       ← 同上
libopencv_video.so.406       ← 同上
libopencv_imgproc.so.406     ← 同上
libopencv_core.so.406        ← 同上
libOpenCL.so.1               ← 板卡自带的 OpenCL
librknnrt.so                 ← 板卡自带的 RKNN Runtime
librga.so.2                  ← 板卡自带的 RGA
libpthread.so.0
libdl.so.2
libstdc++.so.6
libm.so.6
libgcc_s.so.1
libc.so.6
```

五个可执行文件（`out/mosaic_api_test` / `out/mosaic_client_example` /
`out/mosaic_async_client_example` / `out/mosaic_contract_test` / `out/rknn_mosaic_e2e`）
的动态依赖 = **上面这份 +
`libmosaic_pipeline.so`**，并且带 **`RPATH = $ORIGIN`**
⇒ 运行时从**自己所在目录**找那份 `.so` ⇒ **整包可搬**，
**不需要设 `LD_LIBRARY_PATH`**。

* **OpenCV 是动态链接的**（7 个 `libopencv_*.so.406`）⇒ **目标板子必须装
  OpenCV 4.6.0**（本包**不带** OpenCV，也不带任何其它第三方 `.so`）。
  ⚠️ **头与运行库必须同版本**（本包用板卡 4.6.0 的头链 4.6.0 的 `.so.406`）。
* 链接的是**板卡自己的** `librknnrt.so` / `librga.so.2` / `libOpenCL.so.1`
  （DT_NEEDED 是裸 SONAME），所以目标板只要 BSP 里有这些同名库就能跑。
* OpenCV 的**传递依赖**（libjpeg/libpng/libtiff/libgdal/…）**只在板卡运行时
  由 `ldconfig` 解析** ⇒ 交叉链接时不必搬（链接行带
  `-Wl,--allow-shlib-undefined`）。
* 产物**不含 `libc++`**（已核验），只用板子标准的 libstdc++。
* 用的是 **glibc**（不是 musl）⇒ 目标板要是 musl/alpine，得换工具链重编。

---

## F. 集成步骤（5 步）

```bash
# 1) 拿包、解包
cd /path/to/manu/pipeline/npu/build
#    ⚠️ 交付给集成方的是同内容的 mosaic_sdk/（本机 {{NFS_ROOT}}/mosaic_sdk，
#       板卡侧 /mnt/manu/mosaic_sdk），目录结构见 §A

# 2) 编译库（交付包里已预编译，可跳过）
bash build_lib.sh          # → out/libmosaic_pipeline.so（共享库）

# 3) 编译测试程序（链接上面那个 .so；交付包里已预编译，可跳过）
bash build_demo.sh         # → out/mosaic_api_test, out/mosaic_client_example,
#                              out/mosaic_contract_test, out/rknn_mosaic_e2e

# 4) 拷贝到目标板：整树镜像 + 递归 md5 回验
bash copy_to_nfs.sh        # → {{NFS_ROOT}}/mosaic_sdk/
#    手工拷等价于：把 build/ 下的每个普通文件按【同名相对路径】拷进交付目录
#    （src/、include/、runtime/kernels/、out/ 整树照搬），再按 §A 建 5 个兼容别名
#    ⚠️ --kernel-dir 必须指向 <交付目录>/runtime/kernels（新布局），
#       不是旧的平铺 kernels/

# 5) 在目标板上跑（路径按他们的布局改）
./out/mosaic_api_test \
  --model      /path/to/model.rknn \
  --kernel-dir <交付目录>/runtime/kernels \
  --raw-root   /path/to/frames \
  --sequence   THEIR_SEQUENCE \
  --limit 12

#    契约测试用同样的参数
./out/mosaic_contract_test \
  --model      /path/to/model.rknn \
  --kernel-dir <交付目录>/runtime/kernels \
  --raw-root   /path/to/frames \
  --sequence   THEIR_SEQUENCE \
  --limit 12

#    端到端吞吐/延时测量（只调公开 C ABI；--out 必填，TSV 落在 <out>/）
./out/rknn_mosaic_e2e \
  --model      /path/to/model.rknn \
  --kernel-dir <交付目录>/runtime/kernels \
  --raw-root   /path/to/frames \
  --sequence   THEIR_SEQUENCE \
  --out        /path/to/out \
  --decoupled-stages --buffer-a-depth 3
```

**板卡侧权威运行命令**（照抄即可）：

```
/mnt/manu/mosaic_sdk/out/mosaic_api_test \
  --kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels \
  --model ... --raw-root ... --sequence ... --limit 12
```

**换板子必改的两处**（其余照搬）：

1. **构建期**：`RK3588_TOOLCHAIN` / `OPENCV_INCLUDE_DIR` / `OPENCV_LIB_DIR` /
   `BOARD_LIBS` / `RT_LIB_DIR` 五个环境变量指向**他们板子 BSP** 的交叉工具链与
   **板卡自带的** OpenCV 头 / 动态库 / RKNN Runtime / RGA / OpenCL 目录
   （脚本里的**默认值就是板卡的系统路径**，见 §B / §C）。
   ⚠️ 交叉编译在宿主机上做时，这些路径指的是**板卡的文件系统** ——
   宿主机的 `/usr` 是 x86 的，**绝对不能直接用** ⇒ 先把板卡那份取回宿主机。
2. **运行期**：**绑核参数必须重设**（见 §D），否则会绑错核而不报错。

验收门禁：先跑 `./out/mosaic_contract_test`（我们这块板实测 **196 passed / 0 failed**），
过了再跑 `./out/mosaic_api_test`。

吞吐声明的复现：`./out/rknn_mosaic_e2e` 用 **decoupled + buffer-a-depth 3 +
sparse 后处理**，在我们这块板上实测 **12.868 fps / 6,084 交付帧 / p50 313.6 ms**
（11,810 帧全序列），精度与官方黄金基线 **ΔF1 = +0.000000**。
