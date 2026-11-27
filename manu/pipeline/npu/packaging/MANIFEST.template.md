> ℹ️ **这份文件怎么来的**：交付物 `MANIFEST.md` 由工程里的
> `{{SRC_ROOT}}/packaging/MANIFEST.template.md` 生成（占位符由脚本按实况填充；
> `MD5_TABLE` 标记之间的表由脚本自动生成；模板漏改一个，脚本会当场报错退出）。
> **请勿手改交付物。**
>
> ℹ️ **命令里的绝对路径（`/media/manu/...`）是作者机器上的**：换成你自己的包目录即可。

# MANIFEST.md —— 嵌入式组集成交付清单

> **基准板卡 = Orange Pi 5**（`orangepicm5`，RK3588 / Debian 12 Bookworm /
> aarch64 / glibc 2.36 / 板载 gcc 12.2.0），ssh `root@192.168.1.115`。
> 本清单里的全部实测数据都来自它。本清单把交付物分成
> **板卡无关**（直接可用）与**板卡相关**（必须由集成方按自己板子确认）两类。

---

## A. 包内容总览

包根目录：`manu/pipeline/npu/build/`（仓库内路径
`{{PKG_ROOT}}/`）。**交付给集成方的目录名是 `mosaic_sdk/`**（本机
`{{NFS_ROOT}}/mosaic_sdk/`），**内容与 `build/` 逐文件一致**。
⚠️ **落盘位置由集成方自选**：旧板把它挂在 NFS 的 `/mnt/manu/mosaic_sdk/`，
但**新板（Orange Pi 5）上没有 `/mnt/manu` 这个挂载点** ⇒ 不要再照抄那个路径。

### 交付到板卡后的目录结构

```
mosaic_sdk/
├── README.md / MANIFEST.md / VERSION / build_lib.sh / build_demo.sh /
│   build_probes.sh / build_board_native.sh / copy_to_nfs.sh / fetch_board_libs.sh
│   fetch_opencv_headers.sh / known_hosts
├── src/
│   ├── npu/    mosaic_pipeline.cpp/.h, mosaic_core.h, mosaic_post_gpu.h,
│   │           mosaic_track_fusion.cpp/.h, mosaic_paths.h, stat_compat.c,
│   │           mosaic_api_test.cpp, mosaic_client_example.cpp,
│   │           mosaic_async_client_example.cpp,
│   │           mosaic_contract_test.cpp, rknn_mosaic_e2e.cpp,
│   │           mosaic_cadence_probe.cpp, mosaic_track_dump.cpp,
│   │           mosaic_tracked_cb_example.cpp
│   ├── cpp/    gmc_stream_ocl.cpp, md5.h
│   └── opencl/host/   ocl_host.h, post_local_max.cl
├── include/    rknn_api.h, stb_image.h, stb_image_resize.h, stb_image_write.h,
│               rga/×18, CL/×20,
│               opencv4/×211  ← **OpenCV 4.8 头**（板上只有 4.6 头 ⇒ 必须用这份）
├── runtime/kernels/    warp_median_fused.cl, sortnet_generated.inc, gmc_feature_gpu.cl
├── models/    p0_off_A_none.rknn（**RKNN 模型，随包交付**）
├── out/        mosaic_pipeline.o, mosaic_track_fusion.o, stat_compat.o,
│               libmosaic_pipeline.so, libmosaic_pipeline.a,
│               mosaic_api_test, mosaic_api_test_static,
│               mosaic_client_example,
│               mosaic_async_client_example,
│               mosaic_contract_test, rknn_mosaic_e2e,
│               mosaic_cadence_probe, mosaic_track_dump,
│               mosaic_tracked_cb_example
└── 兼容别名（相对符号链接）：
    mosaic_api_test       -> out/mosaic_api_test
    mosaic_async_client_example -> out/mosaic_async_client_example
    mosaic_contract_test  -> out/mosaic_contract_test
    libmosaic_pipeline.so -> out/libmosaic_pipeline.so
    mosaic_pipeline.h     -> src/npu/mosaic_pipeline.h
    kernels               -> runtime/kernels
```

* `mosaic_sdk/` 是 `build/` 的**整树镜像**：源＝包目录下的每个普通文件，
  目标＝**同名相对路径**（`src/`、`include/`、`runtime/kernels/`、`models/`、
  `out/` 整树照搬）。
* 顶层那 5 个**兼容别名是符号链接**（不是副本），分别指向 `out/` 与 `src/npu/`，
  为的是**旧的平铺命令**（`./mosaic_api_test --kernel-dir ./kernels`）继续可用。
* `out/` 的 {{OUT_COUNT}} 个产物在交付包里**已预编译**（RK3588 / aarch64），嵌入式组
  **不必**先编译即可直接跑；确实要重编时在交付目录原地编（见 §B、§F）。
* ℹ️ **`out/mosaic_tracked_cb_example`（v10.7 新增产物）**：**跟踪结果回调示范可执行文件**
  —— 注册 `mosaic_pipeline_set_tracked_callback` 拿**一目标一框**的跟踪结果，
  并与原始候选框做**同帧对照**。它用的是**纯追加**的新公开接口
  （公开 `mosaic_pipeline_*` 接口 v10.7 时 `34 → 35`、v1.0.0 再 `35 → 36`，
  `MOSAIC_API_VERSION` **仍是 10**，既有接口一字未改）。
  源码在 `src/npu/mosaic_tracked_cb_example.cpp`。
* ⚠️ **本包不携带任何第三方 .so/.a**（**没有第三方依赖目录**）：OpenCV /
  RKNN Runtime / RGA / OpenCL 全部**动态链接板卡自带的库**（见 §C 的
  **板卡依赖矩阵**）。库本身**同时**以**共享库** `out/libmosaic_pipeline.so`
  （SONAME = `libmosaic_pipeline.so`）**与静态库** `out/libmosaic_pipeline.a`
  **两种形式**交付（**两者内容等价**，同源同对象文件）⇒ 客户按需取用；
  六个测试程序用 `-lmosaic_pipeline` + `-Wl,-rpath,'$ORIGIN'` 链接共享库
  ⇒ 整包可搬、可直接跑。
  ℹ️ **「不携带任何第三方 .so/.a」说的是第三方依赖**（OpenCV / RKNN Runtime /
  RGA / OpenCL）；**我们自己的 `out/libmosaic_pipeline.a` 是本包产物，不在此列**。
* ℹ️ **但头文件是要随包发布的**：新板（Orange Pi 5）上 OpenCV **运行库是 4.8**
  （`libopencv_*.so.408`），**头却只有 4.6**（`/usr/include/opencv4`）⇒
  头/库 ABI 错配 ⇒ **4.8 头（`include/opencv4/`，211 个文件）随包发布**，
  集成方在板上**可离线构建**（详见 §C）。
* ✅ **本包自带 RKNN 模型** `models/p0_off_A_none.rknn`（6,326,305 B，md5
  `1a3faf2f9459cad8a389ffa81810234c`）。**它不是第三方依赖**，是我们自己的
  训练产物 ⇒ 不违反「不带第三方 .so/.a」这条约束。
  以前包里**一个 .rknn 都没有**，文档却让客户用**我们内部 NFS 的路径**
  ⇒ 客户照抄命令必然失败；现已改为包内路径。
  ⚠️ 模型在 NFS 上是 **soft 挂载**，**板卡直读会被读坏** ⇒ 板上使用前仍须
  先 `cat` 到板卡 `/tmp`。

### 🏷️ 版本（当前 = `{{SDK_VERSION}}`）

> **本交付包当前版本 = `{{SDK_VERSION}}`**（**第一次打标**）。这个版本号由
> `packaging/make_build_dir.sh` 从 `src/npu/mosaic_pipeline.h` 的 `MOSAIC_SDK_VERSION`
> 宏**自动读出**再回填到本节 ⇒ 与包根的 `VERSION` 文件**同源**，不会各写各的。

**三个载体：分别是哪个版本、什么时候用哪个**

| 载体 | 在哪 | 说的是哪个版本 | 什么时候用 |
|---|---|---|---|
| `MOSAIC_SDK_VERSION` 宏 | `src/npu/mosaic_pipeline.h` | **头文件 / 编译期**的版本（客户代码是照着它编的） | 编译期分支（配 `MOSAIC_SDK_VERSION_MAJOR` / `_MINOR` / `_PATCH`）、给自己的日志打版本 |
| `mosaic_pipeline_sdk_version(char* buf, int32_t cap)` | 公开 C ABI（**v1.0.0 新增**） | **实际加载的那个 `.so`** 的版本（板上跑的到底是哪一版） | **现场排查**：板卡上跑的到底是哪一版 |
| `VERSION` 文件 | `<交付包根>/VERSION` | **这个交付包**的版本（内容就是一行 `1.0.0`） | 脚本 / 人工**一眼可查**（`cat VERSION`） |

* `mosaic_pipeline_sdk_version()` 的语义：把**库内编译进去**的版本串写进 `buf`
  并返回写入长度（**不含**结尾 `NUL`）；`buf == NULL` 或 `cap <= 0` 返回 `-1`；
  `cap` 不够时**截断但仍 `NUL` 结尾**；**对 `NULL` 流水线安全**
  （版本属于**库**，不属于流水线）。
* 🟢 **三处必须一致**：`MOSAIC_SDK_VERSION` 宏 == 运行时调用返回的字符串
  == `VERSION` 文件的内容。
  🔴 **宏与运行时调用返回的字符串不同 ⇒ 头文件与库不是同一次构建**
  ⇒ **重新编译**（用交付包里的头重编，别拿旧头凑）。
* **版本规则（本项目的规矩）**：**主版本（第一位）只由用户决定**
  （**没有用户明确指示不许自己动**）；**次版本（第二位）= 大改动**
  （破坏性行为变更 / 新的交付参数组 / 公开接口重构）；
  **修订号（第三位）= 小改动**（新增可选接口 / 修复 / 探针 / 文档 / 测试）。
  ⇒ 当前 `{{SDK_VERSION}}` ⇒ 后续**小改动走 `1.0.x`**、**大改动走 `1.x.0`**。

**版本历史**

| 版本 | 日期 | 内容 |
|---|---|---|
| `v1.0.0` | 2026-11-27 | 首个打标版本：v10.7 跟踪结果回调 `mosaic_pipeline_set_tracked_callback`（纯追加，公开接口 34→35）+ 版本号机制本身（公开接口 35→36）+ 契约 **223 passed / 0 failed**（ematech 板卡实测）+ 板端运行时回读 `mosaic_pipeline_sdk_version()` = `1.0.0` |

### 新板（Orange Pi 5）上的实际落地布局

| 路径 | 内容 | 说明 |
|---|---|---|
| `/home/orangepi/app` | 线上 app 可执行文件（90 MB） | `DT_NEEDED` 含 `libmosaic_pipeline.so`，`DT_RPATH` 含 `$ORIGIN` ⇒ 找同目录那份 |
| `/home/orangepi/libmosaic_pipeline.so` | **线上在用的 SDK 共享库**（= 本包 `out/libmosaic_pipeline.so`，**{{LIB_SO_SIZE}} B**，md5 `{{LIB_SO_MD5}}`） | ✅ **2026-11-25 已按集成方授权替换线上库**；旧库留 `*.bak.20261125` 回滚备份 ⇒ 详见下面的「线上库已替换」一行 |
| `/home/orangepi/runtime/kernels/` | `warp_median_fused.cl` / `sortnet_generated.inc` / `gmc_feature_gpu.cl` | 库运行时**只加载前两个** |
| `/home/orangepi/runtime/ir_uav_det.rknn` | 线上模型 | 线上 app 用的 RKNN 模型 |
| `/home/orangepi/runtime/lq_ir_640x512.mp4` | 线上测试视频 | — |
| `/home/orangepi/lib/` | OpenCV **4.8 运行库副本** | app 自带的一份 4.8 |
| `/root/bts/` | **测试台** | `p0_off_A_none.rknn`（md5 `1a3faf2f9459cad8a389ffa81810234c`）、`frames/VIDEO00007_19700101_001324/`（**11,810** 帧 640×512 BMP）、`frames3x/`（35,430 帧）、`kernels/` |

磁盘：根分区 28 G，已用 39%（17 G 可用）。

### 板卡实测记录（Orange Pi 5 `orangepicm5` @ 192.168.1.115，2026-11-25）

> 板上 OpenCV **运行库是 4.8**（`/usr/lib/libopencv_*.so.408`），
> **头只有 4.6**（`/usr/include/opencv4`）⇒ 本包**随包发布 4.8 头**。
> 板上 **OpenCL 只有 Debian 的 ICD loader**（`libOpenCL.so.1`，**没有** `libOpenCL.so`
> 开发软链）⇒ `-lOpenCL` 必然链不出来，构建脚本已改成**按绝对路径**链接那个 `.so.1`。
> **RGA 有两份不同的 `librga.so`**（系统版 `/usr/lib/aarch64-linux-gnu/librga.so`
> vs `/home/orangepi/rockit/librga.so`）⇒ 必须锁**系统版**（SONAME `librga.so.2`）。

| 项 | 结果 |
|---|---|
| `out/` 包内自带 `.so` 在新板上跑 60 帧 | 检测 TSV md5 **`ceea5fe670a0e996eae0dbffc8c22c11`**（60 帧 / 168 检出 / 0 丢帧） |
| 共享版 / 静态版 / 零参数版三份对照 | 三份检测 TSV 的 md5 **全部相同** ⇒ 共享库版、静态库版、零参数调用三者行为等价 |
| `mosaic_contract_test` | **217 passed / 0 failed** |
| `mosaic_client_example`（零参数） | 21 帧 52 目标，rc=0 |
| `mosaic_async_client_example`（零参数） | 120 帧进 / 326 目标出 |
| `ldd` | 全部解析成功，**无 `not found`** |
| 线上库**已替换**（2026-11-25，集成方授权；替换时 app **处于停止状态**） | 板上 `/home/orangepi/libmosaic_pipeline.so`（**app 实际加载点**）、`/usr/lib/`、`/root/bts/` 三处均已换成本包这份（md5 `{{LIB_SO_MD5}}`）；**回滚备份 = `*.bak.20261125a`**（md5 `a819d6e4dc5c10bdc79a67362e79bf31`，= 被替换掉的那一版）；⚠️ 板上**原有**的 `*.bak.20261125`（md5 `c7682e3c…` / `08c8d76d…`）**不是**本次这一版的备份，**别搞混**；`/root/mosaic-test/old/` 下的更早副本**刻意保留未动**。替换方式 = 同目录写临时文件再 `mv` 原子替换（**不动运行中进程的 inode**）。**替换后复验（全部实测）**：`ldd /home/orangepi/app` **无 `not found`**；`nm -D --defined-only` 行数 **335 → 335**、全局 `T` 符号 **127 → 128**、公开 `mosaic_pipeline_*` **34 → 35**（**v10.7 当时的实测值**；v1.0.0 又纯追加了 `mosaic_pipeline_sdk_version` ⇒ 现行公开接口 **36**）；**丢失 1 个内部 C++ 符号** `MosaicPipeline::RunFusionStep(const std::vector<mosaic::TrackFusionCand>&)`（该类在公开头里是**不透明类型** ⇒ 客户端**不可能**引用；app `nm -D --undefined-only` 实测**无引用**）；**检测 TSV md5 仍是 `ceea5fe670a0e996eae0dbffc8c22c11`（= 改造前基线，逐字节相同）**；契约 **217 passed / 0 failed**；`--mode tracked-only`（只注册跟踪回调）**60 帧 / 60 框 / 单帧最多 1 框 / 帧内 `track_id` 重复 0 帧**。⚠️ **app 实跑未验证**（替换时 app 未运行，**未擅自启动客户 app**）|
| **v10.7 纯追加**（跟踪结果回调） | 新增公开接口 **`mosaic_pipeline_set_tracked_callback`**（**跟踪结果回调**，**一目标一框**）⇒ 公开 `mosaic_pipeline_*` 接口 `34 → 35`（**v10.7 当时**）、`.so` 导出符号总数 `335 → 335`（**净增 0**：新增 1 个**公开**符号，同时内部 C++ 符号 `MosaicPipeline::RunFusionStep(...)` 因签名由 const-ref 改 rvalue-ref 而**下线** —— 该类在公开头里是**不透明类型**（`typedef struct MosaicPipeline MosaicPipeline;`）⇒ 客户端**不可能**引用它，app 无影响）、`MOSAIC_API_VERSION` **仍是 10**（**既有公开接口一字未改**）；契约断言 `199 → 217`（新增 **18** 条：**13** 条跟踪回调（注册/退回/切换/队列关闭/tracked-only）+ **5** 条 `flush()` 生命周期 / 派发顺序（`flush()` 返回后**不再有任何跟踪回调触发** ⇒ 嵌入式可「`flush()`→`destroy()`」连写，**绝不会**回调访问已销毁流水线的野指针）；交付包新增产物 `out/mosaic_tracked_cb_example`；**两板实测**（Orange Pi 5 `192.168.1.115` + ematech `192.168.0.64`，2026-11-25）：**只注册跟踪回调一个即可**拿到「一目标一框」（`--mode tracked-only` ⇒ tracked 回调 **60 帧 / 60 框 / 单帧最多 1 框**、**帧内 `track_id` 重复 0 帧**、`mosaic_pipeline_collect()` 返回 **-1** 即原始流未缓冲）⇒ **两个回调不是必须同时注册**（第二个只为**同帧对照**）；契约 **217 passed / 0 failed** |
| **v1.0.0 纯追加**（版本号机制） | 新增公开接口 **`mosaic_pipeline_sdk_version`**（`int32_t mosaic_pipeline_sdk_version(char* buf, int32_t cap)`：把**库内编译进去**的版本串写进 `buf` 并返回写入长度（不含结尾 `NUL`）；`buf == NULL` 或 `cap <= 0` 返回 `-1`；`cap` 不够时**截断但仍 `NUL` 结尾**；**对 `NULL` 流水线安全** —— 版本属于**库**不属于流水线）⇒ 公开 `mosaic_pipeline_*` 接口 **`35 → 36`**、`MOSAIC_API_VERSION` **仍是 10**（**纯追加，既有公开接口语义一字未改**）；公开头新增 `MOSAIC_SDK_VERSION` / `MOSAIC_SDK_VERSION_MAJOR` / `MOSAIC_SDK_VERSION_MINOR` / `MOSAIC_SDK_VERSION_PATCH`（= `"1.0.0"` / 1 / 0 / 0）；交付包根目录新增 **`VERSION`** 文件（内容 = `1.0.0`，由 `make_build_dir.sh` 从头文件的 `MOSAIC_SDK_VERSION` 宏**自动生成**）；**版本规则**：**主版本（第一位）只由用户决定**（没有用户明确指示不许自己动）、**次版本（第二位）= 大改动**（破坏性行为变更 / 新的交付参数组 / 公开接口重构）、**修订号（第三位）= 小改动**（新增可选接口 / 修复 / 探针 / 文档 / 测试）⇒ 当前 `1.0.0`，后续小改动走 `1.0.x`、大改动走 `1.x.0`；⚠️ **`.so` 导出符号总数等 `nm -D` 实测数字本次未复验**（`.so` 尚未重编）⇒ 上面的 `335 → 335` / `127 → 128` 等历史数字**保持原样不动**；契约 **217 passed / 0 failed** |
| glibc 兼容 | 本 SDK 产物最高需求 `GLIBC_2.29`（buildroot 9.3.0）≤ 板上 2.36 ⇒ 可上板 |
| **板卡原生重编**（`build_board_native.sh`，板载 g++ **12.2.0**） | ✅ **`BUILD_BOARD_NATIVE_OK`**：用**包内 4.8 头** + 板卡 4.8 运行库编出 `.so`，NEEDED = 7×`libopencv_*.so.408` + `libOpenCL.so.1` + `librga.so.2` + `librknnrt.so`（**`-lOpenCL` / `-lrga` 全程未用**）；该原生产物跑 60 帧的检测 TSV md5 **仍是 `ceea5fe670a0e996eae0dbffc8c22c11`**（= 改造前基线，与交叉编译产物**逐字节相同**）、契约 **217 passed / 0 failed** ⇒ **集成方在这块板上就能自己重编，不用回宿主机** |
| RGA ABI | `sizeof(rga_buffer_t)=96`、`offsetof(handle)=92`、handle 回读 `0x12345678` 一致 ⇒ 与本 SDK 的假设一致 |

#### 旧板实测记录（`evm3588` @ 192.168.0.64 / `192.168.0.211`，**均已作废**）

> ⚠️ **下面这些数字来自旧板，只作历史追溯用** —— 旧板已下线，
> **不要**再拿它当验收基线。当前基线是上面那张 Orange Pi 5 的表。

> 旧板自带 **OpenCV 4.8**（`libopencv_core.so.408`）⇒ 板卡上原生重编 / 实测的是 **ocv48 变体**；
> 另两个变体（ocv46 / ocv42）的 `.so` 是宿主交叉编译产物，与 ocv48 版**行为逐字节等价**
> （下面两行 md5 对三个变体通用），差异只在**运行时参数回读行**是否多打两个字段。

| 项 | 结果 |
|---|---|
| `out/` 包内自带 `.so` + 探针跑 2500 帧 | `ship_td_fused.tsv` md5 `5a317dab22bb28497e93b3db77927270`、`ship_td_cands.tsv` md5 `0e511a4c66a7f2842c79c0a9f73fe613` —— 与板上独立跑的参照**逐字节相同** |
| 运行时参数回读 | `th_base=0.4800 th_salvage=0.0600 th_ground=0.2600 sky_ratio=1.000 match_dist=10.00 tags=0x7 … max_age_phys=-1 r_slope=5.00 alpha_decay=1.000 confirm_consec=2 confirm_gap_tol=8`（**9 字段全在**） |
| 旧 `.so` vs 板端重编 `.so` | 同 250 帧：候选流 md5 `959963cad3d9d520d01d7b84e123b834`、融合 md5 `b1e8eb4401fb6dcd518d827cdec2a2dc` —— **两版完全相同 ⇒ 行为逐字节等价**；差异仅在回读行多打 `th_salvage` / `match_dist` |
| ABI 兼容性 | 旧 `.so` 读新结构体的**全部已知字段值都正确**（`th_base/th_ground/sky_ratio/r_slope/alpha_decay/confirm_consec/confirm_gap_tol/max_age_phys`）⇒ **结构体无错位** |
| 跨尺度候选合并半径（2026-11-23 起**内置在库里**，默认 **20 px**，可用带 16~24） | 板端 2500 帧窗口：半径 `20` ⇒ F1 `0.6930591` / FAR `0.0488`；半径 `0` ⇒ F1 `0.5689473` / FAR `0.3832` ⇒ **合并必须开**。⚠️ **不可照搬服务器的 768 px**（板端尺度不同，会丢次要目标） |
| 跟踪的开关（2026-11-23） | 跟踪**默认开启**、参数**内置** ⇒ 调用侧不需要新增调用；开关只有环境变量 `MOSAIC_TRACK=0`（关）/ `MOSAIC_TRACK_MERGE_RADIUS=<px>`（合并半径）/ `MOSAIC_TRACK_QUEUE_MAX=<n>`（队列上限，默认 4096）。启动日志必须出现 `[i] candidate dedup ON: radius=20.0 px (cross-scale merge)` |
| 静态库 `libmosaic_pipeline.a`（2026-11-23 新增，与共享库同源同对象文件） | 板端 60 帧窗口：用**链静态库**编出的 `mosaic_api_test_static` 跑出的检测 TSV 与**链共享库**的 `mosaic_api_test` **逐字节相同**（md5 相同）⇒ 静态库与共享库**行为等价**；静态程序 **NEEDED 里没有** `libmosaic_pipeline.so`（运行时不需要我们的 .so） |

<!-- MD5_TABLE:BEGIN -->
（本段由 `make_build_dir.sh` 自动生成，勿手改）
<!-- MD5_TABLE:END -->

> ℹ️ **当前基线（新板 Orange Pi 5）**：包内自带 `.so` 在新板上跑 60 帧，
> 检测 TSV md5 = **`ceea5fe670a0e996eae0dbffc8c22c11`**（60 帧 / 168 检出 / 0 丢帧）；
> 共享库版 / 静态库版 / 零参数版三份 md5 **完全相同**。
>
> ℹ️ **历史口径（旧板，已作废）**：`ship_td_fused.tsv` / `ship_td_cands.tsv` 那两个 md5
> 是旧板 2500 帧窗口的记录。2026-11-23 起新交付的 `.so` **默认把跨尺度候选合并半径
> 设成 20 px**（旧记录里那一行是**合并没开**（半径 0）时跑出来的）⇒ 旧口径下
> `ship_td_fused.tsv` md5 **应为 `51c4ef144a172f829c2d40e05ebd89e1`**（r=20 融合输出），
> 而 `ship_td_cands.tsv` md5 **仍是 `0e511a4c66a7f2842c79c0a9f73fe613`**
> —— **候选流不动**（合并只发生在**喂给跟踪器之前**，`mosaic_pipeline_collect()` 的原始
> 候选逐字节不变，认证的检测 TSV md5 不受影响）。
> 想复现旧口径：设 `MOSAIC_TRACK_MERGE_RADIUS=0` 再跑。

**关于 md5 基线**：

* `src/**`、`include/**`、`runtime/kernels/**` 的 md5 是**固定基线**——可用来核对
  交付是否被改动。
* `out/` 段**刻意不列 md5**：产物是**构建产物**，随工具链/编译路径变化，
  列进来只会误导人。需要核对产物时手动 `md5sum out/*`。
  本次交付的共享库 md5 = `{{LIB_SO_MD5}}`，静态库 md5 = `{{LIB_A_MD5}}`。
  本版**同时**交付**共享库** `out/libmosaic_pipeline.so` **与静态库**
  `out/libmosaic_pipeline.a`（两者内容等价）⇒ **没有 `ar` 归档时间戳那种
  「每次构建都变」的问题**（`.a` 用 `ar rcsD` 的 deterministic 模式 ⇒
  归档里不留 uid/gid/mtime），在**同一个包根路径**下重编，`.so`、`.a`
  与六个可执行文件的 md5 都是稳定的。
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
| 时空内核 | `src/npu/mosaic_track_fusion.cpp` | 帧间（时空）后处理内核，独立编译单元；**必须**带 `-ffp-contract=off` 才能与 Python 参考逐位一致 |
| 时空内核头 | `src/npu/mosaic_track_fusion.h` | 时空后处理的参数结构与引擎声明（库内部使用） |
| 内部头 | `src/npu/mosaic_core.h` | GMC / RGA / RKNN 胶水层 |
| 内部头 | `src/npu/mosaic_post_gpu.h` | GPU 后处理路径（头内已内嵌同一份 kernel 源码） |
| **默认路径解析头** | `src/npu/mosaic_paths.h` | **板卡无关的默认路径回落**（demo / 探针用）：`env` → **交付包内** → **板卡部署路径** → 旧 NFS 兜底；**包根只剥一级**（exe 在 `<pkg>/out/` ⇒ 包根 = `<pkg>`）⇒ **整包丢到板卡任意目录都能直接跑**，不依赖任何绝对路径 |
| glibc 垫片 | `src/npu/stat_compat.c` | `stat` hidden symbol 垫片（交叉编译产物必需；编出 `out/stat_compat.o`，**故意不在 `.a` 里** —— 那会把静态库锁死在一个 glibc 大版本上） |
| 引擎 | `src/cpp/gmc_stream_ocl.cpp` | GMC/OpenCL 引擎 |
| 工具 | `src/cpp/md5.h` | MD5 实现 |
| OpenCL host | `src/opencl/host/ocl_host.h` | OpenCL 封装 |
| OpenCL 副本 | `src/opencl/host/post_local_max.cl` | GPU 后处理 kernel 的**可读副本**（运行时**不需要**） |
| 示例 demo | `src/npu/mosaic_api_test.cpp` | 只调公开 C ABI 的示例 |
| 最小集成示范 | `src/npu/mosaic_client_example.cpp` | **嵌入方照着抄的最简调用骨架**（open / set_post_mode / push / collect / destroy） |
| 多线程生产-消费模板 | `src/npu/mosaic_async_client_example.cpp` | **工程级集成骨架**：相机线程不等算法 + Worker 严格一推一收 + 业务方自己的回调 |
| 跟踪结果回调示范 | `src/npu/mosaic_tracked_cb_example.cpp` | **跟踪结果回调示范（v10.7）**：注册 `mosaic_pipeline_set_tracked_callback` 拿**一目标一框**的跟踪结果，并与原始候选框做同帧对照 |
| 契约测试 | `src/npu/mosaic_contract_test.cpp` | 库的验收门禁 |
| 端到端测试程序 | `src/npu/rknn_mosaic_e2e.cpp` | **吞吐/延时测量**（12.9 fps 那个数就是它测的），只调公开 C ABI |
| 第三方头 | `include/rknn_api.h` | RKNN API |
| 第三方头 | `include/stb_image.h`、`include/stb_image_resize.h`、`include/stb_image_write.h` | stb 图像库 |
| 第三方头 | `include/rga/*`（18 个） | RGA API 头 |
| 第三方头 | `include/CL/*`（20 个） | Khronos OpenCL 头 |
| **第三方头** | `include/opencv4/*`（**211 个**） | **OpenCV 4.8 头**（BSD 许可，可再分发）。⚠️ **板上只有 4.6 头**（`/usr/include/opencv4`）⇒ **必须用这份**，否则头/库 ABI 错配（详见 §C） |
| **kernel 资产** | `runtime/kernels/warp_median_fused.cl` | **运行时必需** |
| **kernel 资产** | `runtime/kernels/sortnet_generated.inc` | **运行时必需** |
| kernel 资产 | `runtime/kernels/gmc_feature_gpu.cl` | 可选（仅探针用，库不加载） |
| 构建脚本 | `build_lib.sh` / `build_demo.sh` | 交叉编译库 / 测试程序（**宿主机**上跑，用交叉工具链） |
| 板卡原生构建脚本 | `build_board_native.sh` | **在板卡上**用板载 gcc 12.2.0 全量编译库 + 两个验证探针 |
| 探针脚本 | `build_probes.sh` | 只编两个板卡验证探针（`mosaic_cadence_probe` / `mosaic_track_dump`） |
| 取库脚本 | `fetch_board_libs.sh` | 把**板卡自带的**库与头取回宿主机（ssh 打包 → tar → 解包），并打印构建要用的路径变量 |
| 取头脚本 | `fetch_opencv_headers.sh` | 取**版本严格匹配**的 OpenCV 头（`--src` / `--board` / `--expect-abi` / `--askpass`），版本不符即报错退出 |
| 一键脚本 | `copy_to_nfs.sh` | 编译 + 整树拷贝到 NFS + md5 回验 +（可选）板卡验证 |
| 主机密钥 | `known_hosts` | 板卡主机密钥基线（ssh 严格校验用；板卡换密钥会直接失败，这是故意的） |
| 文档 | `README.md` / `MANIFEST.md` | 使用说明 / 本清单 |
| **版本文件** | `VERSION` | **本交付包的版本号**（内容 = `1.0.0`，一行）；由 `make_build_dir.sh` 从 `src/npu/mosaic_pipeline.h` 的 `MOSAIC_SDK_VERSION` 宏**自动生成** ⇒ 与头文件、运行时 `mosaic_pipeline_sdk_version()` **同源** |

**交付包包含源码（`src/**`）、第三方头（`include/**`）、kernel 资产
（`runtime/kernels/**`）、构建脚本（`build_lib.sh` / `build_demo.sh`）与预编译
产物（`out/**`）** ⇒ 集成方**可以在交付目录原地重编**（构建脚本按脚本自身
所在目录定位，不依赖仓库里的其它文件）：

```bash
cd <交付目录>            # 交付目录名 mosaic_sdk/，落盘位置自选
bash build_lib.sh        # → out/libmosaic_pipeline.so（共享库）
                         #   out/libmosaic_pipeline.a（静态库；**同一批 .o**
                         #   再打一次 ar 归档** —— 不需要额外的构建命令）
bash build_demo.sh       # → out/mosaic_api_test, out/mosaic_client_example,
#                            out/mosaic_contract_test, out/rknn_mosaic_e2e
                         #   out/mosaic_tracked_cb_example（**跟踪结果回调示范**）
                         #   out/mosaic_api_test_static（**链 .a 的静态链接版**）
bash build_probes.sh     # → out/mosaic_cadence_probe, out/mosaic_track_dump
                         #   （两个板卡验证探针；**没有任何别的脚本会编它们**）
```

> 🟢 **板上原生重编**（新板推荐路径）：把整包丢到板卡上（任意目录，例如
> `/home/orangepi/mosaic_sdk/`），然后
>
> ```bash
> cd /home/orangepi/mosaic_sdk
> bash build_board_native.sh   # 用**板载 gcc 12.2.0** 编库 + 两个探针
> ```
>
> 该脚本会**自己处理新板的两个坑**：① OpenCL 没有 `libOpenCL.so` 开发软链
> ⇒ 改成按**绝对路径**链接 `libOpenCL.so.1`；② RGA 有两份不同的 `librga.so`
> ⇒ 用**绝对路径**锁系统那份（`/usr/lib/aarch64-linux-gnu/librga.so`，
> SONAME `librga.so.2`），**不用 `-lrga`**（`-L` 顺序会把 Rockchip BSP 那份挑走）。
> 链接行另加 `-Wl,--no-as-needed`（Debian 的 gcc 默认 `--as-needed`）。
> ⚠️ **板上原生构建的头必须用 `include/opencv4`（4.8）** —— 板上 `/usr/include/opencv4`
> 那份是 **4.6**，配 4.8 运行库就是 **ABI 错配**（见 §C）。

> ℹ️ **静态库由 `build_lib.sh` 一并产出**（同一批 `.o` 用 `ar` 打包）⇒
> **不需要额外的构建命令**，也不需要改任何脚本 ⇒ **客户要哪种库自己选**。
> `build_demo.sh` 会**同时**产出**共享链接版**（`out/mosaic_api_test`）与
> **静态链接版**（`out/mosaic_api_test_static`）两个 demo ⇒ 静态库「能不能链通、
> 能不能跑」当场就有活证据（两者运行输出**逐字节一致**）。
> ⚠️ **静态链接必须按路径写 `out/libmosaic_pipeline.a`，不能再写 `-lmosaic_pipeline`**
> （链接器在 `-L` 路径下**优先挑 `.so`** ⇒ 写了 `-l` 会静默链成共享版）；
> `out/stat_compat.o`（glibc<2.33 的 `stat` hidden symbol 垫片）**要单独带上**
> —— 它**故意不在 `.a` 里**（那会把 `.a` 锁死在一个 glibc 大版本上）。

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
> ② **板卡自带的第三方库**（本包**不携带任何第三方** .so/.a，见 §C 的
> **板卡依赖矩阵**；**我们自己的 `out/libmosaic_pipeline.a` 是本包产物，不在此列**）：
>
> ```bash
> export OPENCV_ABI=408                                # 运行库的 SONAME 后缀（408⇒4.8 / 406⇒4.6 / 410⇒4.10），默认 408
> export OPENCV_VERSION=4.8                            # 期望的**头**版本，默认由 OPENCV_ABI 推导
> export OPENCV_INCLUDE_DIR=<OpenCV 4.8 头目录>        # 默认**包内** include/opencv4（板上那份是 4.6，不能用）
> export OPENCV_LIB_DIR=<OpenCV 4.8 运行库目录>        # 板卡默认 /usr/lib（4.8 在这里，不在 multiarch 目录）
> export BOARD_LIBS=<板卡 libOpenCL / librga 目录>      # 板卡默认 /usr/lib/aarch64-linux-gnu
> export RT_LIB_DIR=<板卡 librknnrt 目录>               # 板卡默认 /usr/lib
> ```
>
> ℹ️ **OpenCV 头不用集成方自己找** —— 本包 `include/opencv4/` 里就是 **4.8 头**
> （211 个文件），`OPENCV_INCLUDE_DIR` 的默认值已经是它 ⇒ **板上可离线构建**。
> 需要换一份头时用包内 `fetch_opencv_headers.sh`（版本不符即报错退出）。
>
> ⚠️ **交叉编译在宿主机上做时，上面这些路径指的是「目标板卡的文件系统」**：
> 宿主机的 `/usr` 是 x86 的，**绝对不能直接用**（会污染构建）⇒
> 要先把板卡的那份（头 + `.so`）取回宿主机，再把变量指过去。

**源码与头文件不含任何板卡私有路径**——`src/**`、`include/**`、
`runtime/kernels/**` 里只有相对 include 与公开 API 声明，换一块 RK3588 板子，
只要 BSP 里有同版本的交叉工具链与**同版本的 OpenCV 头**，照样能编译。

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

## C. 板卡相关的部分（第三方运行库由板卡提供；OpenCV 头由本包随包发布；模型与帧数据仍由集成方自备）

**下面「我们这边用的」那一列是我们在基准板（Orange Pi 5 @ 192.168.1.115）上验证过的版本，供对照。**

> ⚠️ **策略（2026-11-11 起，2026-11-25 按新板修订）**：交付包**不再携带任何第三方** .so/.a
> ⇒ OpenCV / RKNN Runtime / RGA / OpenCL **全部动态链接板卡自带的库**
> （嵌入式组的要求：直接用板卡上原来就有的 `.so`）。
> 「要不要把某个库装到你们板子上」仍然是集成方自己的决定。
> **唯一的例外是「头」**：新板（Orange Pi 5）上 OpenCV **只有 4.6 头**，而运行库是 4.8
> ⇒ 头/库 ABI 错配 ⇒ **4.8 头随包发布**到 `include/opencv4/`（见下）。

### 板卡依赖矩阵（本包**不携带任何第三方** .so/.a；**OpenCV 头除外**）

| 库 | 板卡路径 | SONAME | 版本 |
|:--|:--|:--|:--|
| **OpenCV core/imgproc/imgcodecs/calib3d/features2d/flann/video** | **`/usr/lib/libopencv_*.so.408`** | `libopencv_core.so.408` 等 | **OpenCV 4.8** ← **本 SDK 链的就是这套** |
| OpenCV（Debian 系统包，**本 SDK 不链它**） | `/usr/lib/aarch64-linux-gnu/libopencv_*.so.406`（dpkg `libopencv-core406` 等） | `libopencv_core.so.406` 等 | 4.6 —— ⚠️ **选错就是事故**，见下 |
| **OpenCV 头** | 板上**只有 4.6**：`/usr/include/opencv4`（`CV_VERSION_MINOR 6`） | — | 4.6 ⇒ **不能用**；**用本包 `include/opencv4/`（4.8，211 个文件）** |
| RKNN Runtime | `/usr/lib/librknnrt.so` | `librknnrt.so` | 版本号本次**未复核**（我们历来验证的是 **2.2.0**） |
| RGA | `/usr/lib/aarch64-linux-gnu/librga.so`（→ `librga.so.2`，另有 `librga.so.2.1.0`） | `librga.so.2` | 版本号本次**未复核**（旧板记录：rga_api 1.10.0_[9]）；**ABI 已实测通过**（见 §A） |
| OpenCL（ICD loader） | `/usr/lib/aarch64-linux-gnu/libOpenCL.so.1` | `libOpenCL.so.1` | OpenCL 3.0（Mali-G610 r0p0 / 4 CU） |
| OpenCL（真正的 GPU 实现） | `/usr/lib/aarch64-linux-gnu/mali/libMaliOpenCL.so.1`（由 `/etc/OpenCL/vendors/mali.icd` 指定） | `libMaliOpenCL.so.1` | 链接期**不需要**它（ICD loader 运行时去读 `.icd`） |

#### ⚠️ 三个必须知道的板卡坑

1. **OpenCV 板上并存两套，选错就是事故**：**4.8 运行库**（`/usr/lib`，本 SDK 与线上
   app 链的那套）+ **4.6 系统包**（`/usr/lib/aarch64-linux-gnu`，Debian 的
   `libopencv-core406`）。⇒ 运行库目录**按 ABI（SONAME 后缀）选**，**不按
   「哪个目录存在」选**：`OPENCV_ABI=408`（默认）时只认 `libopencv_core.so.408`
   ⇒ 落到 `/usr/lib`。⚠️ 若误选 4.6 那套，**同一进程会同时装进 4.6 与 4.8 两套
   OpenCV**，`cv::Mat` 这类 C++ 类型跨 `.so` 边界串味 ⇒ 事故（**不报错**，只是行为诡异）。
   构建脚本的 `cv_module_ok()` 带 SONAME 守卫，**选错直接报错退出**。
2. **头只有 4.6，运行库却是 4.8** ⇒ 头/库 ABI 错配（OpenCV 次版本会改 C++ ABI：
   `cv::Mat` 布局 / 内联函数）⇒ **4.8 头必须外部提供**。
   本包已把 **4.8 头随包发布**（`include/opencv4/`，211 个文件，BSD 许可）⇒
   **集成方在板上可离线构建**。构建脚本会**当场校验头版本**
   （`cv_version_of()` 读 `CV_VERSION_MAJOR/MINOR`），与 `OPENCV_ABI` 对不上 ⇒ 报错退出。
   要换一份头时用包内 `fetch_opencv_headers.sh`（`--src` / `--board` /
   `--expect-abi 408`），版本不符即报错退出。
3. **OpenCL 只有 ICD loader，没有开发软链** ⇒ **`-lOpenCL` 必然链不出来**
   ⇒ 构建脚本在找不到 `libOpenCL.so` 时**退化成按绝对路径**链接 `libOpenCL.so.1`
   （链接器取文件里的 SONAME ⇒ 产物 `NEEDED` 仍是 `libOpenCL.so.1`，与线上 app 一致）。
   ⚠️ **RGA 也有两份不同的 `librga.so`**（系统版 `/usr/lib/aarch64-linux-gnu/librga.so`
   vs `/home/orangepi/rockit/librga.so`）⇒ 必须用**绝对路径**锁**系统版**，
   **不要写 `-lrga`**（`-L` 顺序会挑到 Rockchip BSP 那份）。

* **编译期**：需要 OpenCV **4.8** 的头。默认用**包内** `include/opencv4/`；
  板上 `/usr/include/opencv4` 那份是 4.6，**不能**拿来配 4.8 运行库。
* **链接期**：需要那 7 个 `libopencv_*.so.408` + `librknnrt.so` + `librga.so.2`
  + `libOpenCL.so.1`（用 `OPENCV_ABI` / `OPENCV_INCLUDE_DIR` / `OPENCV_LIB_DIR` /
  `BOARD_LIBS` / `RT_LIB_DIR` 定位；默认值就是上面的板卡系统路径）。
* **运行期**：上面这些**全部由板卡 `ldconfig` 解析**（新板上 `ldd` 实测全部可解析、
  **无 `not found`**）⇒ **不需要设 `LD_LIBRARY_PATH`**。
  唯一需要 `-rpath` 的是**本包自己的** `libmosaic_pipeline.so`：六个测试程序用
  `-Wl,-rpath,'$ORIGIN'` 找**同目录**的那一份 ⇒ **整包可搬**。

> ⚠️ **交叉编译在宿主机上做时，上表那些路径指的是「目标板卡的文件系统」**：
> 宿主机的 `/usr` 是 **x86 的**，**绝对不能直接用**（会拿 x86 的头/库去编
> aarch64 ⇒ 编译失败或产出坏二进制）。
> ⇒ **先把板卡的那份取回宿主机**，再把变量指过去。

| 项 | 我们这边用的（仅供对照） | 集成方要做什么 |
|:--|:--|:--|
| `librknnrt.so` | 板卡路径 `/usr/lib/librknnrt.so`，SONAME `librknnrt.so`（版本号本次**未复核**；我们历来验证的是 **2.2.0**） | **由板卡提供**（本包不带）。运行时用你们板子 BSP 里的那份；**版本必须匹配 `include/rknn_api.h`**（我们验证的是 2.2.0）。若板子上没有 ⇒ 从 RKNN BSP 里取一份装上 |
| `librga.so.2` | 板卡路径 `/usr/lib/aarch64-linux-gnu/librga.so`，SONAME `librga.so.2`（版本号本次**未复核**；旧板记录 `rga_api version 1.10.0_[9]`）；⚠️ `/home/orangepi/rockit/librga.so` 是**另一份**（md5 不同），线上 app 解析到的是**系统版** | **由板卡提供**（本包不带）。其 API 头 `include/rga/*` 已随包交付。⚠️ **必须用绝对路径锁系统那份**，不要写 `-lrga` |
| `libOpenCL.so.1` | 板卡路径 `/usr/lib/aarch64-linux-gnu/libOpenCL.so.1`，SONAME `libOpenCL.so.1`；⚠️ **板上没有 `libOpenCL.so` 开发软链** | **由板卡提供**（本包不带）。用板卡自己的 OpenCL 驱动库（Mali-G610 r0p0 / OpenCL 3.0 / 4 CU）；⚠️ **链接期不要写 `-lOpenCL`**，用绝对路径链 `.so.1` |
| OpenCV | **运行库** = `/usr/lib/libopencv_*.so.408`（**4.8**）；**头** = 本包 `include/opencv4/`（**4.8**） | **运行库由板卡提供**（本包不带 `.so`）；**头用本包随包发布的那份**。⚠️ **头与库必须同版本** —— 板上 `/usr/include/opencv4` 那份 **4.6 不能用** |
| 模型 `.rknn` | ✅ **本包自带** `models/p0_off_A_none.rknn`，6,326,305 B，md5 `1a3faf2f9459cad8a389ffa81810234c`（fp16，输入 640×640×3，输出 2×[1,1,320,320]） | ✅ **直接用包内这份**（板上路径 `<交付目录>/models/p0_off_A_none.rknn`；⚠️ NFS soft 挂载直读会被读坏 ⇒ 板上使用前先 `cat` 到 `/tmp`），也可换他们自己的模型。**换模型的形状约束见下表**（板卡实测） |
| 帧数据 | `VIDEO00007_19700101_001324`（640×512 BMP，命名 `frame_%06d.bmp`，共 11,810 帧），新板测试台路径 `/root/bts/frames/VIDEO00007_19700101_001324` | 他们自己的相机流（同样须是 640×512 灰度、单通道） |

> **交叉编译器**仍属于对方环境：需要 aarch64 交叉工具链（我们用的是
> gcc-buildroot 9.3.0）。构建脚本里用 `RK3588_TOOLCHAIN`（可用 `:` 分隔多个候选）
> 或 `CROSS_CXX` 覆盖即可，不用改脚本。
>
> **OpenCV 的运行库也不属于本包**：它由**板卡**提供（4.8 运行库在 `/usr/lib`）
> ⇒ 编译期用 `OPENCV_ABI` / `OPENCV_INCLUDE_DIR` / `OPENCV_LIB_DIR` 指过去
> （默认值即上表的板卡系统路径 / 包内头）。
> **但 OpenCV 的「头」随包发布**（`include/opencv4/`，4.8，211 个文件）——
> 板上那份是 4.6，**不能**拿来配 4.8 运行库。

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

> ℹ️ **基准板 Orange Pi 5 也是 RK3588**（同款 4×A76 + 4×A55 拓扑）⇒ 下面的预设值
> 在新板上**同样成立**；但**换到别的板子**（或换了 BSP 的核数/大小核划分）时，
> 仍**必须**按新板的拓扑重设 —— 绑错核**不报错**，只是性能悄悄掉下去。

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
libopencv_imgcodecs.so.408   ← 板卡自带的 OpenCV **4.8**
libopencv_calib3d.so.408     ← 同上
libopencv_features2d.so.408  ← 同上
libopencv_flann.so.408       ← 同上
libopencv_video.so.408       ← 同上
libopencv_imgproc.so.408     ← 同上
libopencv_core.so.408        ← 同上
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

六个可执行文件（`out/mosaic_api_test` / `out/mosaic_client_example` /
`out/mosaic_async_client_example` / `out/mosaic_contract_test` /
`out/mosaic_tracked_cb_example` / `out/rknn_mosaic_e2e`）
的动态依赖 = **上面这份 +
`libmosaic_pipeline.so`**，并且带 **`RPATH = $ORIGIN`**
⇒ 运行时从**自己所在目录**找那份 `.so` ⇒ **整包可搬**，
**不需要设 `LD_LIBRARY_PATH`**。

（**例外**：`out/mosaic_api_test_static` 是**链静态库 `.a`** 编出来的 ⇒ 它的
`DT_NEEDED` 里**没有** `libmosaic_pipeline.so` ⇒ **运行时不需要**我们的 `.so`；
但**上面那份第三方依赖它一条不缺** —— OpenCV / RKNN Runtime / RGA / OpenCL
仍然是**动态链接板卡自带的库**（静态链接只影响**我们自己的**那一层）。）

* **OpenCV 是动态链接的**（7 个 `libopencv_*.so.408`）⇒ **目标板子必须装
  OpenCV 4.8 运行库**（本包**不带**任何第三方 `.so`；**只带 4.8 的头**）。
  ⚠️ **头与运行库必须同版本**：本包用 **4.8 头**（包内 `include/opencv4/`）
  链 **4.8 运行库**（板上 `/usr/lib`）⇒ 板上那份 **4.6 头不能配**。
  ⚠️ 板上同时还有一套 **4.6 运行库**（`/usr/lib/aarch64-linux-gnu`）——
  **本 SDK 不链它**，也**不要**在同一个进程里同时加载两套 OpenCV（见 §C 的坑 1）。
* 链接的是**板卡自己的** `librknnrt.so` / `librga.so.2` / `libOpenCL.so.1`
  （DT_NEEDED 是裸 SONAME），所以目标板只要 BSP 里有这些同名库就能跑。
  新板实测 `ldd` **全部解析成功、无 `not found`**。
* OpenCV 的**传递依赖**（libjpeg/libpng/libtiff/libgdal/…）**只在板卡运行时
  由 `ldconfig` 解析** ⇒ 交叉链接时不必搬（链接行带
  `-Wl,--allow-shlib-undefined`）。
* 产物**不含 `libc++`**（已核验），只用板子标准的 libstdc++。
* 用的是 **glibc**（不是 musl）⇒ 目标板要是 musl/alpine，得换工具链重编。
  基准板 Orange Pi 5 是 **glibc 2.36**（Debian 12），而本 SDK 产物最高需求
  **`GLIBC_2.29`**（buildroot 9.3.0）⇒ **可上板**（`__xstat@GLIBC_2.17`
  在 2.36 上仍导出，实测 `dlopen` 正常）。

---

---

## F0. 🟢 集成前必读：跟踪已内置、**默认开启**（2026-11-23）

* **时空跟踪现在由库自己建**：`mosaic_pipeline_create()`（`mosaic_pipeline_open()` 内部
  也会走它）时用 **2026-11-20 交付工作点 `cc2_md10`** 建好融合引擎
  （`th_base 0.48` / `th_salvage 0.06` / `th_ground 0.26` / `sky_ratio 1.0` /
  `confirm_consec 2` / `confirm_gap_tol 8` / `r_slope 5.0` / `match_dist 10.0` /
  `max_age_phys -1` / `min_hits 3`），并把**跨尺度候选合并半径 20 px** 一并内置
  （板端可用带 16~24 px；⚠️ **不要照搬服务器的 768 px** —— 板端尺度不同，会丢次要目标）。
* ⇒ **调用侧不需要新增任何调用，也不需要新的公开接口**；
  **公开接口集合回到 2026-11-20 交付版**（`MOSAIC_API_VERSION` 仍然是 `10`）。
* **取跟踪结果**（两个接口都是老接口，**没有新增公开符号**）：
  * `mosaic_pipeline_collect_tracked()` —— **在线**，逐帧，与 `collect()` 一一对应
    （同一个帧号）；队列要排空（上限 `MOSAIC_TRACK_QUEUE_MAX`，默认 4096 帧）。
  * `mosaic_pipeline_collect_fused()` —— **离线**，`mosaic_pipeline_flush()` 之后才有，
    取完返回 `-1`。
  * `mosaic_pipeline_collect()` 语义**不变**（仍返回原始候选流 ⇒ 认证的检测 TSV md5 不动）。
* **版本握手（只影响「要改参数」的客户）**：`mosaic_pipeline_set_track_fusion()` 会校验
  `cfg.version`，**对不上返回 `-2`**，时空融合**静默不生效**（不崩、只是没开）。
  ⇒ **要改参数**的客户必须用本包 `src/npu/mosaic_pipeline.h` **重编 app**
  （**只换 `.so` 不换头** ⇒ 覆盖失败）。
  `MOSAIC_TRACK_FUSION_VERSION` = 2，比 v1 多 `max_age_phys` / `r_slope` / `alpha_decay` /
  `use_physical_frame_id`（默认值见 `mosaic_pipeline_track_fusion_defaults()`，
  唯一真源在 C++ 内核里）。
  **不碰这个接口的客户不需要重编**（跟踪在库内默认开着）。
* **环境开关（标定 / A-B 用，正常集成不用碰）**：
  * `MOSAIC_TRACK=0`（或 `off` / `false`）⇒ 启动时关掉跟踪；
  * `MOSAIC_TRACK_MERGE_RADIUS=<px>` ⇒ 覆盖合并半径（默认 20；`0` = 关闭合并）；
  * `MOSAIC_TRACK_QUEUE_MAX=<n>` ⇒ 覆盖跟踪队列上限（默认 4096）。
* **验收证据**：板端启动日志**必须**出现
  `[i] candidate dedup ON: radius=20.0 px (cross-scale merge)`；
  出现 `[warn] candidate dedup OFF (radius=0)` 说明**合并没开**（FAR 会飙）。
* 帧号语义：`mosaic_pipeline_push_frame(..., client_frame_id)` 请传**物理帧号**（丢帧就跳号）。

## F. 集成步骤（5 步）

```bash
# 1) 拿包、解包
cd /path/to/manu/pipeline/npu/build
#    ⚠️ 交付给集成方的是同内容的 mosaic_sdk/（本机 {{NFS_ROOT}}/mosaic_sdk），
#       板卡侧**落盘位置自选**（新板上没有 /mnt/manu 挂载点），目录结构见 §A

# 2) 编译库（交付包里已预编译，可跳过）
bash build_lib.sh          # → out/libmosaic_pipeline.so（共享库）
#                           out/libmosaic_pipeline.a（静态库；**同一批 .o** 打的归档，
#                             **不需要额外的构建命令**）

# 3) 编译测试程序（链接上面那个 .so；交付包里已预编译，可跳过）
bash build_demo.sh         # → out/mosaic_api_test, out/mosaic_client_example,
#                              out/mosaic_contract_test, out/rknn_mosaic_e2e
#                              out/mosaic_tracked_cb_example（**跟踪结果回调示范**）
#                           out/mosaic_api_test_static（**链 .a 的静态链接版**：
#                             共享链接版与静态链接版**同时**产出）

# 3') 或者：在**板卡上原生重编**（用板载 gcc 12.2.0；
#     OpenCL / RGA 走绝对路径链接，见 §C 的坑 3）
bash build_board_native.sh # → out/libmosaic_pipeline.so + 两个验证探针

# 4) 拷贝到目标板：整树镜像 + 递归 md5 回验
bash copy_to_nfs.sh        # → {{NFS_ROOT}}/mosaic_sdk/
#    手工拷等价于：把 build/ 下的每个普通文件按【同名相对路径】拷进交付目录
#    （src/、include/、runtime/kernels/、out/ 整树照搬），再按 §A 建 5 个兼容别名
#    ⚠️ --kernel-dir 必须指向 <交付目录>/runtime/kernels（新布局），
#       不是旧的平铺 kernels/
#    ⚠️ 新板上没有 /mnt/manu 挂载点 ⇒ 拷到板上时用 --board-dest 指真实目录

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

**基准板（Orange Pi 5）侧权威运行命令**（照抄即可）：

```
# 板卡部署布局：库在 /home/orangepi，kernel 在 /home/orangepi/runtime/kernels，
# 帧数据在测试台 /root/bts/frames
/home/orangepi/mosaic_sdk/out/mosaic_api_test \
  --kernel-dir /home/orangepi/runtime/kernels \
  --model      /home/orangepi/runtime/ir_uav_det.rknn \
  --raw-root   /root/bts/frames \
  --sequence   VIDEO00007_19700101_001324 \
  --limit 12
```

> 🟢 **零参数也能跑**：demo / 探针的默认路径走 `src/npu/mosaic_paths.h` 的
> **四级回落** —— `env`（`MOSAIC_KERNEL_DIR` / `MOSAIC_MODEL` / `MOSAIC_RAW_ROOT` /
> `MOSAIC_FRAMES_DIR`）→ **交付包内**（`<pkg>/runtime/kernels`、`<pkg>/models/…`，
> 包根只剥一级）→ **板卡部署路径**（`/home/orangepi/runtime/kernels`、
> `/root/bts/frames`）→ 旧 NFS 兜底（`/mnt/manu/…`）。
> ⇒ **整包丢到板卡任意目录都能直接跑**，不依赖任何绝对路径；
> **命令行参数优先级永远最高**。

**换板子必改的三处**（其余照搬）：

1. **构建期**：`OPENCV_ABI`（运行库 SONAME 后缀，默认 **408** ⇒ 4.8）/
   `OPENCV_VERSION`（期望的头版本，默认 **4.8**）/ `OPENCV_INCLUDE_DIR`（4.8 头，
   默认**包内** `include/opencv4`）/ `OPENCV_LIB_DIR`（4.8 运行库，默认 `/usr/lib`）/
   `BOARD_LIBS`（OpenCL + RGA，默认 `/usr/lib/aarch64-linux-gnu`）/
   `RT_LIB_DIR`（RKNN Runtime，默认 `/usr/lib`）/ `RK3588_TOOLCHAIN`（交叉工具链）。
   换成**他们板子 BSP** 的路径（脚本里的**默认值就是基准板的系统路径**，见 §B / §C）。
   ⚠️ 交叉编译在宿主机上做时，这些路径指的是**板卡的文件系统** ——
   宿主机的 `/usr` 是 x86 的，**绝对不能直接用** ⇒ 先把板卡那份取回宿主机。
2. **运行期**：**绑核参数必须重设**（见 §D），否则会绑错核而不报错。
3. **路径**：kernel 目录 / 模型 / 帧数据的绝对路径（见上面 `mosaic_paths.h` 的
   四级回落，或直接传命令行参数 / 环境变量）。

验收门禁：先跑 `./out/mosaic_contract_test`（基准板实测 **217 passed / 0 failed**），
过了再跑 `./out/mosaic_api_test`。

吞吐声明的复现：`./out/rknn_mosaic_e2e` 用 **decoupled + buffer-a-depth 3 +
sparse 后处理**，在**旧板**（`evm3588` @ 192.168.0.64）实测 **12.868 fps /
6,084 交付帧 / p50 313.6 ms**（11,810 帧全序列），精度与官方黄金基线
**ΔF1 = +0.000000**。⚠️ **新板（Orange Pi 5）上这个吞吐数尚未复测** —— 本轮在
新板上验证的是**正确性**（检测 TSV md5 / 契约测试 / `ldd`），不是吞吐。
