# MANIFEST.md —— 嵌入式组集成交付清单

> 目标板卡**不是**我们手上这块 RK3588 开发板。本清单把交付物分成
> **板卡无关**（直接可用）与**板卡相关**（必须由集成方按自己板子确认）两类。

---

## A. 包内容总览

包根目录：`manu/pipeline/npu/build/`（仓库内路径
`/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu/build/`）。

<!-- MD5_TABLE:BEGIN -->
（本段由 `make_build_dir.sh` 自动生成，勿手改）
<!-- MD5_TABLE:END -->

**关于 md5 基线**：

* `src/**`、`include/**`、`runtime/kernels/**` 的 md5 是**固定基线**——可用来核对
  交付是否被改动。
* `out/` 段**刻意不列 md5**：`ar` 会把成员时间戳写进归档索引，
  `libmosaic_pipeline.a` 的 md5 **每次构建都变**，列进来只会误导人。
  需要核对产物时手动 `md5sum out/*`：`mosaic_pipeline.o` 与两个可执行文件的
  md5 是稳定的，`.a` 的不是。
* `MANIFEST.md` 自身的 md5 无法自含（自引用），请单独用 `md5sum MANIFEST.md` 核对。

---

## B. 板卡无关的部分（直接可用，拿到就能编）

| 类别 | 文件 | 说明 |
|---|---|---|
| 库实现 | `src/npu/mosaic_pipeline.cpp` | 库的编译单元 |
| 公开 ABI | `src/npu/mosaic_pipeline.h` | 客户写代码时 include 它（C ABI） |
| 内部头 | `src/npu/mosaic_core.h` | GMC / RGA / RKNN 胶水层 |
| 引擎 | `src/cpp/gmc_stream_ocl.cpp` | GMC/OpenCL 引擎 |
| 工具 | `src/cpp/md5.h` | MD5 实现 |
| OpenCL host | `src/opencl/host/ocl_host.h` | OpenCL 封装 |
| 示例 demo | `src/npu/mosaic_api_test.cpp` | 只调公开 C ABI 的示例 |
| 契约测试 | `src/npu/mosaic_contract_test.cpp` | 库的验收门禁 |
| 第三方头 | `include/rknn_api.h`、`include/stb_image*.h` | RKNN API + stb 图像库 |
| 第三方头 | `include/rga/*`（18 个） | RGA API 头 |
| 第三方头 | `include/CL/*`（20 个） | Khronos OpenCL 头 |
| **kernel 资产** | `runtime/kernels/warp_median_fused.cl` | **运行时必需** |
| **kernel 资产** | `runtime/kernels/sortnet_generated.inc` | **运行时必需** |
| kernel 资产 | `runtime/kernels/gmc_feature_gpu.cl` | 可选（仅探针用，库不加载） |
| 构建脚本 | `build_lib.sh` / `build_demo.sh` | 交叉编译库 / 测试程序 |
| 一键脚本 | `copy_to_nfs.sh` | 编译 + 拷贝到 NFS + md5 回验 +（可选）板卡验证 |
| 文档 | `README.md` / `MANIFEST.md` | 使用说明 / 本清单 |

**这些文件不含任何板卡私有路径**——换一块 RK3588 板子，只要 BSP 里有同版本的
交叉工具链与 OpenCV 安装，照样能编译。

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

## C. 板卡相关的部分（必须由集成方按自己板子提供）

**下面「我们这边用的」那一列只是我们验证过的版本 + md5，供对照；
这些文件不在交付包内，也不得直接当作依赖包塞给对方。**

| 项 | 我们这边用的（仅供对照） | 集成方要做什么 |
|:--|:--|:--|
| `librknnrt.so` | RKNN Runtime **2.3.0**（`c949ad889d@2024-11-07`），`/run/media/manu/SSD960-LOCAL/nfs/npu/librknnrt.so`，7,259,064 B，md5 `56e425dcc7b59ab0e1aec685b178b99f`，SONAME `librknnrt.so` | 用**他们自己板子 BSP** 的同版本 runtime（2.3.0）。版本必须匹配 `include/rknn_api.h` |
| `librga.so.2` | `/run/media/manu/SSD960-LOCAL/nfs/npu/librga.so.2.1.0`，192,712 B，md5 `7136ae5469eb217f068ef3efc6f26d53`，SONAME `librga.so.2`，`rga_api version 1.10.0_[9]` | 同上，取他们 BSP 的 librga。其 API 头 `include/rga/*` 已随包交付 |
| `libOpenCL.so.1` | `/media/manu/1TB-Volume/rk3588/boardlibs/libOpenCL.so.1.0.0`，34,808 B，md5 `e9036376b6f8f59e30522eb43fe410cd`，SONAME `libOpenCL.so.1`（OpenCL 1.0–2.0） | 同上，取他们板子的 OpenCL 驱动库 |
| 模型 `.rknn` | `/run/media/manu/SSD960-LOCAL/nfs/ptq/models/p0_off_A_none.rknn`，6,326,305 B，md5 `1a3faf2f9459cad8a389ffa81810234c` | 可直接用我们这份，也可换他们自己的模型。**换模型的形状约束见下表**（板卡实测） |
| 帧数据 | `VIDEO00007_19700101_001324`（640×512 BMP，命名 `frame_%06d.bmp`，共 11,810 帧），本板路径 `/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324` | 他们自己的相机流（同样须是 640×512 灰度、单通道） |

> **交叉编译器与 OpenCV** 也属于对方环境：需要 aarch64 交叉工具链
> （gcc-buildroot 9.3.0）+ 交叉编译的 OpenCV 4.10.0 安装树。构建脚本里用
> `RK3588_TOOLCHAIN` / `OPENCV_INSTALL` / `BOARD_LIBS` / `RT_LIB_DIR` 四个环境变量
> 覆盖即可，不用改脚本。

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

`out/mosaic_api_test` / `out/mosaic_contract_test` 的动态依赖：

```
librknnrt.so      ← 板卡 BSP 的 RKNN Runtime
librga.so.2       ← 板卡 BSP 的 RGA
libOpenCL.so.1    ← 板卡 BSP 的 OpenCL
libpthread.so.0
libdl.so.2
libstdc++.so.6
libm.so.6
libgcc_s.so.1
libc.so.6
```

* **OpenCV 是静态链进去的**（`-lopencv_*` + `-ltegra_hal` + `-lzlib`，全部来自
  交叉编译的 OpenCV 4.10.0 安装树）⇒ **目标板子不需要装 OpenCV**。
* 链接的是**板卡自己的** `librknnrt.so` 与 `librga.so.2`（DT_NEEDED 是裸 SONAME），
  所以目标板只要 BSP 里有这两个同名库就能跑，不用我们这份。
* 产物**不含 `libc++`**（已核验），只用板子标准的 libstdc++。
* 用的是 **glibc**（不是 musl）⇒ 目标板要是 musl/alpine，得换工具链重编。

---

## F. 集成步骤（5 步）

```bash
# 1) 拿包、解包
cd /path/to/manu/pipeline/npu/build

# 2) 编译库
bash build_lib.sh          # → out/libmosaic_pipeline.a

# 3) 编译测试程序（链接上面那个 .a）
bash build_demo.sh         # → out/mosaic_api_test, out/mosaic_contract_test

# 4) 拷贝到目标板（或直接留在本地跑）
#    按 README「NFS 交付目录内容」一节的命令，把
#    mosaic_api_test / mosaic_contract_test / libmosaic_pipeline.a /
#    mosaic_pipeline.h / kernels/ / README.md / MANIFEST.md 一起放好

# 5) 在目标板上跑（路径按他们的布局改）
./mosaic_api_test \
  --model      /path/to/model.rknn \
  --kernel-dir /path/to/kernels \
  --raw-root   /path/to/frames \
  --sequence   THEIR_SEQUENCE \
  --limit 12
```

**换板子必改的两处**（其余照搬）：

1. **构建期**：`RK3588_TOOLCHAIN` / `OPENCV_INSTALL` / `BOARD_LIBS` / `RT_LIB_DIR`
   四个环境变量指向**他们板子 BSP** 的交叉工具链、OpenCV 安装树、板卡库目录。
2. **运行期**：**绑核参数必须重设**（见 §D），否则会绑错核而不报错。

验收门禁：先跑 `mosaic_contract_test`（我们这块板实测 **196 passed / 0 failed**），
过了再跑 `mosaic_api_test`。
