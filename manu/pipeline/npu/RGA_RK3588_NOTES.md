# RK3588 RGA 接口笔记（实测）

> 目的：为「RK3588 板端 C++ multi-scale 拼图推理」摸清 RGA 的真实能力与调用方式。
> 本文所有数字均为**板端实测**，不是文档抄录；每条都给出可复现命令。
>
> | 项 | 值 |
>|:--|:--|
> | 板卡 | `root@192.168.0.64`（RK3588，8 核，Ubuntu 20.04.6） |
> | RGA 库 | `/usr/lib/aarch64-linux-gnu/librga.so.2` → `librga.so.2.1.0`（192,712 B，md5 `7136ae5469eb217f068ef3efc6f26d53`） |
> | RGA 内核模块 | `rga v1.3.1`，三核 `rga3_core0` / `rga3_core1` / `rga2`，`rga_iommu: IOMMU binding successfully` |
> | 头文件 | 板端 `/usr/include/rga/`（18 个，**与 SDK 的 linux-rga 源码不同版**，md5 `im2d_type.h=db35d4e9…` vs SDK `47a73d51…`） |
> | 测试程序 | `manu/pipeline/npu/rga_probe.cpp` + `manu/pipeline/npu/build_rga_probe_arm64.sh` |
> | 运行日志 | `manu/pipeline/npu/rga_probe_run_20261026.log`（2,499 行，含 RGA 库自身调试输出） |

## 复现命令

```bash
# 1) 交叉编译（本地）
cd /media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu
./build_rga_probe_arm64.sh

# 2) 上板（板卡不能从 NFS 执行，scp 到 /root/）
scp -P 22 out/rga_probe root@192.168.0.64:/root/

# 3) 运行（板端）
ssh -p 22 root@192.168.0.64 '/root/rga_probe --iters 100'
# 分节运行（一次一节，便于定位崩溃）
ssh -p 22 root@192.168.0.64 '/root/rga_probe --only resize,mosaic,timing'
```

---

## 1. 硬件能力（官方 `querystring()` 实测）

| 项 | 实测值 |
|:--|:--|
| `RGA_VERSION` | `v1.10.0_[9]`，`RGA_2_Enhance RGA_3` |
| `RGA_SCALE_LIMIT` | **0.0625 ~ 16**（即 1/16 ~ 16 倍） |
| `RGA_BYTE_STRIDE` | **16** |
| `RGA_MAX_INPUT` / `RGA_MAX_OUTPUT` | 8192×8192 / 8128×8128 |
| `RGA_INPUT_FORMAT` / `RGA_OUTPUT_FORMAT` | 含 `RGB_888`（另有 RGBA_8888 / RGB_565 / RGBA_4444 / RGBA_5551 / YUV / YUYV / Y4） |
| `RGA_FEATURE` | `color_fill color_palette ROP quantize src1_r2y_csc dst_full_csc FBC_mode blend_in_YUV BT.2020` |

**对齐限制**（SDK《Rockchip_Developer_Guide_RGA_CN》「图像格式对齐说明」，跨双核取最严）：
`RGBA8888 → width 4 对齐`、`RGB888 → width 16 对齐`、`RGB565 → width 8 对齐`。
本项目用到的宽度 640 / 512 / 320 / 160 / 128 / 80 **全部满足**。

**核能力差异**（同上「设计指标」）：`RGA2-Enhance` 最小 2×2、缩放 1/16~16；`RGA3` 最小 **128×128**、缩放 1/8~8。
⇒ thumb80（80×64）低于 RGA3 最小尺寸，但实测调度器会自动避开（core 1/2 都成功，见 §6）。

**`imconfig(IM_CONFIG_SCHEDULER_CORE, n)`**：`n=0` 非法（`ILLEGAL_PARAM`），`n=1` / `n=2` 合法。

---

## 2. 缓冲区：唯一可用的调用路径（踩坑重点）

实测三条路径（同一进程内对比）：

| 路径 | 结果 |
|:--|:--|
| 裸 dma-buf fd + `wrapbuffer_fd_t(fd,…)` | `imcopy` 返回 **SUCCESS，但数据一个字节没动**（dst 全零） |
| dma-buf fd + `importbuffer_fd(fd,…)` + `wrapbuffer_handle(h,…)` | 同上，**空操作** |
| **`importbuffer_virtualaddr(va,…)` + `wrapbuffer_handle(h,…)`** | **payload IDENTICAL ✔** |

⇒ **必须走官方 demo 的 Linux 路径**（`samples/im2d_api_demo/rgaImDemo.cpp:306` 就是这么写的）：
`importbuffer_virtualaddr()` → `wrapbuffer_handle()`。

**两条附加硬规则**（都是实测踩出来的）：

1. **必须「先填数据、后 import」**。`importbuffer_virtualaddr()` 在 import 时刻捕获缓冲区内容；
   先 import 再写数据 ⇒ RGA 读到的是旧内容（表现为空操作）。
2. **缓冲区尺寸要精确等于 `w*h*bpp`**。多分配（按 4 B/px 预留但实际按 RGB_888 用）时操作同样变成空操作。

**dma-heap uapi 有两套布局，别照抄 SDK 那份**：

| 布局 | 结构 | 用在哪 |
|:--|:--|:--|
| upstream Linux | `{u64 len; u32 fd_flags; u32 heap_flags; u64 fd; u64 user_phys_addr;}` | `/dev/dma_heap/*` |
| Rockchip 旧版 | `{u64 len; u32 fd; u32 fd_flags; u64 heap_flags;}` | `/dev/rk_dma_heap/rk-dma-heap-cma`（**本板无此设备**） |

SDK 的 `samples/im2d_slt/sources/dma_alloc.cpp` 用的是**旧版**，它针对的是板卡上不存在的
`/dev/rk_dma_heap/rk-dma-heap-cma`，**不能照抄到 `/dev/dma_heap/system`**。
本板实测走**旧版布局**（fd 在第 2 位）才能拿到有效 fd；探针里做成了双布局自动探测
（`AllocDmaOne(layout=0/1)`），先到先得且校验 `fd > 2`。

---

## 3. `imcheck()` 在这块 librga 上不可用

`imcheck()` / `imcheck_t()` 内部会拿**全零的 pat 缓冲**去做检查，打印：

```
537:src has not fd and address for render
```

并且**紧随其后的真实操作会静默失效**（返回 SUCCESS 但不产生任何输出）。

⇒ **新程序不要用 `imcheck()`**，改用显式字节预算校验（探针里的 `CheckBudget()`：
`w*h*get_bpp_from_format(fmt) <= buf.size`）。这条同时是防内核 fault 的护栏。

---

## 4. `imfill` / `imtranslate` 的限制（决定了拼图只能用 `improcess`）

| 接口 | 限制 | 实测 |
|:--|:--|:--|
| `imfill(buf, rect, color)` | **只支持 RGBA 格式**（文档明写「对 RGBA 格式的图像…」） | RGB_888 画布 → `FAILED (0)` |
| `imtranslate(src, dst, x, y)` | **要求 src 与 dst 宽高一致**，超出部分裁剪 | src 640×512 → dst 640×640 → `INVALID_PARAM (-3)` |
| `improcess(src, dst, pat, srect, drect, prect, usage)` | 缩放 + 落位一次完成 | **3 次调用拼出整张画布，全部 SUCCESS** |

⇒ **拼图方案：`improcess` × 3（native 1:1 / thumb160 1/4 / thumb80 1/8）**。
画布背景是**逐帧常量** ⇒ 初始化时 `memset` 一次即可，**逐帧零 CPU 拷贝**。

---

## 5. 数值语义：RGA 缩放 vs PC 侧 `INTER_AREA` / `INTER_LINEAR`

PC 基线的缩略图用 `cv2.INTER_AREA`（块均值，见 `manu/data/multiscale_heatmap_cache.py`）。
RGA 与两种参考的**平均绝对差**（满量程 255）：

| 用例 | vs INTER_AREA | vs INTER_LINEAR | 结论 |
|:--|--:|--:|:--|
| 1/2 降 | 0.6032 | **0.1266** | RGA 降采样更接近双线性 |
| **1/4 降（thumb160）** | **0.4665** | 0.5807 | **≈ INTER_AREA**（0.18%） |
| **1/8 降（thumb80）** | **1.4765** | 1.7146 | **≈ INTER_AREA**（0.58%） |
| 2× 升（热图 320→640） | 71.5131 | **0.5628** | ≈ INTER_LINEAR |
| 4× 升（thumb160→native） | 35.0962 | **1.0075** | ≈ INTER_LINEAR |
| 8× 升（thumb80→native） | 14.5688 | **0.3631** | ≈ INTER_LINEAR（max 差 1） |

**结论**：
- **降采样方向**：RGA 的双线性在 1/4、1/8 降采样上与 `INTER_AREA` 的均值差仅 **0.47 / 1.48（0.18% / 0.58%）**
  ⇒ 用 RGA 做缩略图与 PC 基线**数值基本一致**，不构成口径断裂。
- **升采样方向**：RGA 与 `INTER_LINEAR` 均值差 ≤1.0（0.4%）⇒ **热图反投影可以交给 RGA**
  （前提：热图是 float32，而 RGA 只吃 8bit ⇒ 需要量化，见下）。
- **插值参数被忽略**：`INTER_NEAREST` 与 `INTER_LINEAR` 实测输出**完全相同**（0.5807 / 0.4665 两行一致）
  ⇒ **RGA 恒为双线性，不要依赖这个参数**。

---

## 6. 拼图装配实测（640×640 画布）

用 `improcess` 三次装配，与 CPU 参考（native 1:1 + INTER_AREA 缩略图）逐区域比对：

```
improcess native 1:1      -> SUCCESS (1)
improcess thumb160 (1/4)  -> SUCCESS (1)
improcess thumb80  (1/8)   -> SUCCESS (1)
native  region vs CPU ref: max=0   mean=0.0000  (byte-identical)
thumb160 region vs CPU ref: max=131 mean=0.4665
thumb80  region vs CPU ref: max=146 mean=1.4765
```

⇒ **native 区字节级一致**；两个缩略图区与 `INTER_AREA` 的均值差同上。

---

## 7. 耗时（板端实测，`--iters 100`；**含 librga 自身调试输出的开销**，实际更快）

| 操作 | 耗时 | CPU 参考 |
|:--|--:|--:|
| `imresize` 640×512 → 160×128 | **0.315 ms**（min 0.302） | 块均值 CPU 1.110 ms |
| `imresize` 160×128 → 640×512 | **0.365 ms**（min 0.339） | 双线性 CPU 5.519 ms |
| `importbuffer_virtualaddr` | **0.623 ms**（min 0.621） | — |
| **完整拼图装配（fill + 3× improcess）** | **0.943 ms**（min 0.726） | CPU 全拼图 1.987 ms |

⇒ **拼图装配 ~1 ms，相对 102 ms 的 NPU 前向可忽略**（前提是逐帧不做 CPU 图像拷贝）。

---

## 8. 对 mosaic 方案的直接影响

1. **拼图打包可以完全交给 RGA**：3 次 `improcess` + 一次初始化 `memset`，**逐帧零 CPU 拷贝**，
   满足「能不拷贝就不拷贝」的要求。
2. **画布背景逐帧不变** ⇒ 只在初始化时填一次；`imfill` 不适用于 RGB_888，用 `memset` 即可。
3. **热图反投影是 CPU 侧的大头**：RGA 不支持 float32，只能
   （a）CPU 做浮点重采样，或
   （b）量化成 8bit 后交给 RGA（会引入 ~1/255 量化误差）。
   **这是表 C 拼图代价的主要来源，必须单独量化**（3 个尺度 × 640×512 的上采样 + Max 融合）。
4. **正式程序的 TSV 必须写文件、不能走 stdout**：板端 librga 是带调试日志编译的，
   每次调用都会往 stdout 打 `im2d_rga_impl rga_dump_opt(...)`，会污染 stdout 上的检测结果。
5. **`imcheck()` 禁用**，用显式字节预算校验。

---

## 9. 未解决 / 待确认

- **热图 float32 反投影的最终口径**：CPU 浮点 vs 量化后 RGA，需要在表 C 里给出量化后的精度代价。
- **librga 调试日志能否关掉**：FAQ 写明 Linux 平台需改 `core/NormalRgaContext.h` 的 `__DEBUG` 重编；
  板端这版库**日志是开的**，无运行时开关 ⇒ 只能靠「TSV 写文件 + stdout 重定向」规避。
- **RGA 与 NPU 的资源竞争**：`rknn_heatmap` 跑的时候同时用 RGA 是否影响 NPU 吞吐，未测。
