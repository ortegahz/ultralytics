> ℹ️ **这份文件怎么来的**：交付物 `README.md` 由工程里的
> `manu/pipeline/npu/packaging/README.template.md` 生成（占位符由脚本按实况填充；
> 模板漏改一个，脚本会当场报错，不会产出带双花括号的文档）。**请勿手改交付物**。
>
> ℹ️ **命令里的绝对路径（`/media/manu/...`）是作者机器上的**：把 `cd <路径>` 换成
> 你自己的包目录即可 —— **所有构建脚本都按脚本自身所在目录定位，拷到哪个目录都能编**。
>
> ℹ️ **第三方运行库不再随包，但第三方头随包**：本包**不携带任何第三方** .so/.a
> （**我们自己的 `out/libmosaic_pipeline.a` 是本包产物，不在此列**）—— OpenCV / RKNN Runtime /
> RGA / OpenCL 全部**动态链接板卡自带的库**（详见下面的「板卡依赖矩阵」）。
> ⇒ 编译期需要 **OpenCV 4.8 的头**：**本包现随包发布一份**（`include/opencv4/`，
>   211 个文件）—— 目标板（Orange Pi 5 / Debian 12）上**只有 4.6 的头**
>   （`/usr/include/opencv4`），**不能**拿来配 4.8 的运行库（同一进程会装进两套
>   OpenCV、`cv::Mat` 跨 .so 边界串味 ⇒ 事故）；没有随包头时用包内
>   `fetch_opencv_headers.sh` 取一份（版本不符即 die）。
> ⇒ 链接期需要板卡上那 7 个 `libopencv_*.so.408` + `librknnrt.so` + `librga.so.2` +
>   `libOpenCL.so.1`；⚠ **板上没有 `libOpenCL.so` 开发软链**（只有 Debian 的 ICD loader
>   `libOpenCL.so.1`）⇒ **`-lOpenCL` 在板上必然链接失败**，须按**绝对路径**链那个
>   `.so.1`（三个构建脚本都已自动这么做：`build_lib.sh` / `build_demo.sh` 的
>   `resolve_opencl_link()`、`build_board_native.sh` 的 `OCL_LINK`）；
>   **运行期全部由板卡 `ldconfig` 解析，不需要设 `LD_LIBRARY_PATH`**。
> ⚠️ 交叉编译在**宿主机**上做时，这些路径指的是**目标板卡的文件系统** ——
> 宿主机上的 `/usr` 是 x86 的，**绝对不能直接用**（会污染构建）⇒ 要先把板卡的
> 那份取回宿主机再指过去（用 `OPENCV_INCLUDE_DIR` / `OPENCV_LIB_DIR` /
> `BOARD_LIBS` / `RT_LIB_DIR` 指定）。
> **交叉编译器不在包里**（编译器没法随包发），用 `RK3588_TOOLCHAIN` 指定
> （可用 `:` 分隔多个候选，按顺序取第一个真能用的），或用 `CROSS_CXX` 直接指定编译器。

# mosaic_pipeline SDK —— 独立构建包（RK3588 / aarch64）

本目录是一个**可独立交付的源码包**：库源码、示例 demo、契约测试、第三方头文件全部
拷在这里面，可以**脱离仓库单独编译**（第三方**运行库**不在包里 —— 用板卡自带的，
见「板卡依赖矩阵」）。

```
包根目录（仓库内是 build/，交付给集成方时是 mosaic_sdk/）
├── README.md          ← 你正在看的这份（四条命令）  ★由 packaging/README.template.md 生成
├── MANIFEST.md        ← 交付清单（md5 表自动生成）★由 packaging/MANIFEST.template.md 生成
├── build_lib.sh       ← 只编译库
├── build_demo.sh      ← 只编译测试程序
├── build_probes.sh    ← 只编译两个板卡验证探针
├── build_board_native.sh ← **板卡原生**全量编译（板载 gcc + ABI 校验 + 绝对路径链 OpenCL/RGA）
├── copy_to_nfs.sh     ← 一键整树拷贝到 NFS（可选带板卡验证）
├── fetch_board_libs.sh  ← 把板卡自带的库/头取回宿主机（交叉编译用）
├── fetch_opencv_headers.sh ← 取 **OpenCV 4.8 头**（板上只有 4.6 的头 ⇒ 必须外部提供）
├── known_hosts        ← 板卡主机密钥基线（ssh 严格校验用；板卡换密钥会直接失败，这是故意的）
├── src/
│   ├── npu/           mosaic_pipeline.{cpp,h}, mosaic_core.h, mosaic_post_gpu.h,
│   │                  mosaic_api_test.cpp, mosaic_contract_test.cpp,
│   │                  mosaic_client_example.cpp, rknn_mosaic_e2e.cpp,
│   │                  mosaic_paths.h（**板卡无关的默认路径解析头**）
│   ├── cpp/           gmc_stream_ocl.cpp, md5.h
│   └── opencl/host/   ocl_host.h, post_local_max.cl（kernel 可读副本，运行时不需要）
├── include/           rknn_api.h, stb_image*.h, rga/ + CL/,
│                      opencv4/（**OpenCV 4.8 头，211 个文件，随包发布**）
├── runtime/kernels/   warp_median_fused.cl, sortnet_generated.inc（运行时必需）,
│                      gmc_feature_gpu.cl（可选，库不加载）
├── models/            ← **本包自带的 RKNN 模型（来源）**：p0_off_A_none.rknn（6,326,305 B）
│                      （板上先 cat 到 /tmp/p0_off_A_none.rknn 再用）
└── out/               ← 产物落在这里
    ├── mosaic_pipeline.o
    ├── mosaic_track_fusion.o
    ├── stat_compat.o               ← **buildroot glibc 2.29 `stat` hidden symbol 垫片**（铁律十八）
    ├── libmosaic_pipeline.so      ← **共享库**（SONAME=libmosaic_pipeline.so）
    ├── libmosaic_pipeline.a       ← **静态库**（等价内容；客户要静态链接时用它）
    ├── mosaic_api_test
    ├── mosaic_api_test_static     ← **静态链接版的示例 demo**（链 `.a` 编出来）
    ├── mosaic_client_example
    ├── mosaic_async_client_example
    ├── mosaic_contract_test
    ├── rknn_mosaic_e2e
    ├── mosaic_cadence_probe       ← **板卡验证探针**（本地工具，见下）
    ├── mosaic_track_dump          ← **板卡验证探针**（本地工具，见下）
    └── mosaic_tracked_cb_example  ← **跟踪结果回调示范**（v10.7，见下）
```

> ℹ️ **本包不携带任何第三方 .so/.a**（**没有第三方依赖目录**）：
> OpenCV / RKNN Runtime / RGA / OpenCL 全部**动态链接板卡自带的库**
> （见「板卡依赖矩阵」）。

> **交付到 NFS 时是整棵树一起拷**（不是只拷几个产物文件）：板卡侧看到的目录
> 结构与这里一一对应——`src/ include/ runtime/ models/ out/` + 顶层 10 个文件
> （`README.md` / `MANIFEST.md` / `build_lib.sh` / `build_demo.sh` /
> `build_probes.sh` / `build_board_native.sh` / `copy_to_nfs.sh` /
> `fetch_board_libs.sh` / `fetch_opencv_headers.sh` / `known_hosts`）。

交付包磁盘上共 **{{TOTAL_COUNT}}** 个文件 = 受管 **{{FILE_COUNT}}** 个
（源码 **{{SRC_COUNT}}** 个（`src/`）+ 第三方头 **{{HDR_COUNT}}** 个
（`include/`）+ kernel 资产 **{{KERNEL_COUNT}}** 个（`runtime/`）
+ 模型 **1** 个（`models/`）+ 顶层 9 个文件（含本 README）
+ `MANIFEST.md` 自身）+ `out/` 编译产物 **{{OUT_COUNT}}** 个。

产物（**全部 aarch64**；可执行文件与共享库是 ELF，`out/libmosaic_pipeline.a` 是目标文件归档）：

| 文件 | 是什么 |
|---|---|
| `out/libmosaic_pipeline.so` | **库（共享库）**：GMC/OpenCL + RGA mosaic + RKNN fp16 + 反投影，公开 C ABI 在 `src/npu/mosaic_pipeline.h`；SONAME = `libmosaic_pipeline.so`。**同包内另有静态库 `out/libmosaic_pipeline.a`（等价内容，客户要静态链接时用它）** |
| `out/libmosaic_pipeline.a` | **静态库**（客户要静态链接时用这个；内容与同名 `.so` 等价＝`mosaic_pipeline.o` + `mosaic_track_fusion.o`；**不含** `stat_compat.o` 垫片，链接可执行文件时仍需另行带上） |
| `out/mosaic_api_test` | **给客户的示例 demo**：只调公开 C ABI（open / push / collect / destroy） |
| `out/mosaic_api_test_static` | **静态链接版的示例 demo**（`mosaic_api_test` 链 `.a` 编出来；**静态库可用性的活证据**；运行输出与链 `.so` 的版本逐字节一致） |
| `out/mosaic_client_example` | **最小嵌入示范**（160 行）：只调公开 C ABI（open / set_post_mode / push / collect / destroy），零参数直接跑，嵌入式组照着抄的骨架 |
| `out/mosaic_async_client_example` | **多线程生产-消费 + 回调模板**：相机线程（25 FPS）不等算法、Worker 严格「一推一收」、结果走业务方自己的回调；**嵌入式组真实工程集成的骨架** |
| `out/mosaic_contract_test` | **库的契约测试**（验收门禁） |
| `out/rknn_mosaic_e2e` | **端到端测试程序**：测吞吐/延时（fps / 延时分位就是它测的），只调公开 C ABI |
| `out/mosaic_cadence_probe` | **板卡验证探针**（本地工具，不属于集成面）：逐帧推 / 收，量节奏（fps / Δn / 间隔）与时空融合参数回读 |
| `out/mosaic_track_dump` | **板卡验证探针**（本地工具，不属于集成面）：逐帧导出候选流 / 融合结果 TSV，用于离线对 GT 打分 |
| `out/mosaic_tracked_cb_example` | **跟踪结果回调示范（v10.7）**：注册 `set_tracked_callback` 拿**一目标一框**的跟踪结果，并与原始候选框**同帧对照** |

> 🔴 **两个探针的源码在 `src/npu/`**（`mosaic_cadence_probe.cpp` /
> `mosaic_track_dump.cpp`），**探针走的是 `.so` 里内置的交付工作点**
> （`th_base 0.48` / `th_salvage 0.06` / `th_ground 0.26` / `sky_ratio 1.0` /
> `confirm_consec 2` / `confirm_gap_tol 8` / `r_slope 5.0` / `match_dist 10.0` /
> `max_age_phys -1`）⇒ **探针与客户 app 走的是同一条路径**（2026-11-23 起跟踪内置在库里）。
> 探针只额外覆盖两个工具侧开关（`tag_mask` / `reorder_window`），
> 合并半径通过 `MOSAIC_TRACK_MERGE_RADIUS` 交给库（对应 `--cand-dedup-radius`）。
> 这两个探针**零 OpenCV 依赖**（`readelf -d` 实测 NEEDED 里没有 `libopencv_*`）
> ⇒ **ocv46 / ocv48 / ocv42 三个变体通用**。

> ℹ️ **六个**测试程序（外加上面那两个板卡验证探针）用 `-lmosaic_pipeline` +
> `-Wl,-rpath,'$ORIGIN'` 链接**同目录**的 `out/libmosaic_pipeline.so`
> ⇒ **整包搬到任何目录都能直接跑**，不用设
> `LD_LIBRARY_PATH`（`readelf -d` 实测 RUNPATH = `$ORIGIN`）。
>
> ℹ️ **唯一的例外是 `out/mosaic_api_test_static`**：它是**链静态库 `.a`** 编出来的
> （**不**用 `-lmosaic_pipeline`）⇒ 它的 `DT_NEEDED` 里**没有**
> `libmosaic_pipeline.so` ⇒ **运行时不需要**我们的 `.so`。
>
> ⚠️ **动态库搜索顺序**：`RUNPATH($ORIGIN)` 排在 `LD_LIBRARY_PATH` **之后**
> ⇒ 若 `LD_LIBRARY_PATH` 里另有同名 `libmosaic_pipeline.so`，**它会盖掉本目录这份**
> （实测踩过：`LD_LIBRARY_PATH` 含 `/root/mosaic` 时，探针实际加载的是 `/root/mosaic`
> 那份而不是 `out/` 这份 ⇒ 两个变体跑出同一份输出，差点误判成「等价」）。
> 验收时用 `LD_TRACE_LOADED_OBJECTS=1 <探针>` 实证实际加载的是哪一个。

当前产物的 md5（**本次构建的快照**；`.so` **没有** `ar` 归档时间戳那种
「每次构建都变」的问题；`.a` 用 `ar rcsD`（deterministic：归档里不留
uid/gid/mtime）⇒ **同样稳定**。详见文末「md5 的三个已知差异」）：

| 文件 | md5 |
|---|---|
| `out/mosaic_pipeline.o` | `{{OBJ_MD5}}` |
| `out/mosaic_track_fusion.o` | `{{TRACK_FUSION_OBJ_MD5}}` |
| `out/stat_compat.o` | `{{STAT_COMPAT_OBJ_MD5}}` |
| `out/libmosaic_pipeline.so` | `{{LIB_SO_MD5}}` |
| `out/libmosaic_pipeline.a` | `{{LIB_A_MD5}}` |
| `out/mosaic_api_test` | `{{API_TEST_MD5}}` |
| `out/mosaic_api_test_static` | `{{API_TEST_STATIC_MD5}}` |
| `out/mosaic_client_example` | `{{CLIENT_EXAMPLE_MD5}}` |
| `out/mosaic_async_client_example` | `{{ASYNC_EXAMPLE_MD5}}` |
| `out/mosaic_contract_test` | `{{CONTRACT_TEST_MD5}}` |
| `out/rknn_mosaic_e2e` | `{{E2E_MD5}}` |
| `out/mosaic_cadence_probe` | `{{CADENCE_PROBE_MD5}}` |
| `out/mosaic_track_dump` | `{{TRACK_DUMP_MD5}}` |
| `out/mosaic_tracked_cb_example` | `{{TRACKED_CB_EXAMPLE_MD5}}` |

> ℹ️ **本表读的是「当前磁盘上已有的产物」**（`make_build_dir.sh` 不编译，只**读**
> `$DEST/out/*` 已存在的产物算 md5）⇒ **十四行都有基线**：产物**已经随交付包一起**
> 落在 `out/` 里（`out/` 整棵树会一起拷到 NFS），不是「包里没有」。
>
> ℹ️ **`out/mosaic_tracked_cb_example` 是 v10.7 新增的产物**：**跟踪结果回调示范** ——
> 注册 `set_tracked_callback` 拿**一目标一框**的跟踪结果，并与原始候选框**同帧对照**。
>
> **若某个产物还没编出来**，对应行会显示占位文本「（尚未编译）」而**不是留空**
> ⇒ 编完重跑一次 `make_build_dir.sh` 就会回填。
>
> 🔴 **这张表的 md5 与「包根的绝对路径」绑定**：编译时 `__FILE__` 把路径烤进了
> 二进制 ⇒ **你在自己目录重编出来的 md5 会和这张表不同**（**行为完全一样**，
> 已实测）。这不是包坏了。想拿**跨机器一致**的 md5，见文末「md5 的三个已知差异」
> 一节（相对路径编）。
> ⚠️ 另外它**不是跨版本恒定值**：改了源码，md5 就会变（表中这一行随包更新）。
>
> 🔴 **`out/` 里的 `.so` 与两个 `.o` 的构建方式决定它的 md5**：可能是宿主机
> `build_lib.sh` 交叉编译的产物，也可能是**在 RK3588 板卡上原生重编**的
> （板上有 `aarch64-linux-gnu-g++`，源码 = 本包 `src/npu/` 的同一版）。
> 后者会把**板卡路径**烤进二进制（`__FILE__`）⇒ **md5 与交叉编译版不同**，
> 但**行为逐字节等价**（实测：两版 `.so` 在同一段 250 帧上，候选流与融合结果的
> md5 **完全相同**，见 MANIFEST「板卡实测记录」）。
> 两个**探针**是板上原生编的（`RUNPATH=$ORIGIN`、**零 OpenCV 依赖** ⇒ 三个变体通用）。
> 想拿到与上表一致的 md5，请在自己目录跑 `build_lib.sh` / 重编探针。
>
> ℹ️ **本版交付的是共享库 `.so`**：`.so` 里没有 `ar` 的归档
> 时间戳 ⇒ **同一个包根路径**下重编，md5 是**稳定**的。
> 🔬 **「行为完全一样」的验证方式（2026-11-11 实测）**：在两个不同路径各编一份
> `libmosaic_pipeline.so`，二者 **符号表逐条完全一致**、
> **除构建路径外所有字符串字面量完全一致**（差异只来自 `assert` 宏里的
> `__FILE__`），且**上板跑全序列的逐帧特征 md5 与检测 TSV 逐字节相同**
> ⇒ 差异**纯属路径字符串**，不触及任何行为。

---

## 🧩 本包自带的 RKNN 模型

交付包里**自带一个可以直接跑的 RKNN 模型**，客户**不用**再去别处找模型：

| 项 | 值 |
|---|---|
| **来源**：包内相对路径 | `models/p0_off_A_none.rknn` |
| **来源**：板卡上（交付目录 `/mnt/manu/mosaic_sdk`） | `/mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn` |
| **落点**：板上使用（唯一能安全读 6.3 MB 模型的位置） | `/tmp/p0_off_A_none.rknn` |
| 字节数 | **6,326,305 B**（约 6.3 MB） |
| md5 | `1a3faf2f9459cad8a389ffa81810234c` |
| 量化 | fp16 |
| 输入 | 640×640×3 |
| 输出 | 2×[1,1,320,320] |

* **可以直接用**：把它交给 `--model`（`mosaic_api_test` / `mosaic_contract_test` /
  `mosaic_async_client_example`）或 `mosaic_client_example` 的**第 2 个位置参数**即可。
* **也可以换成客户自己的模型** —— 但**形状约束不变**：**fp16、输入 640×640×3、
  输出 2×[1,1,320,320]**（否则后处理的反投影对不上，检测框会全错）。
* 🔴 **必须先 `cat` 到板卡 `/tmp` 再用**：交付目录本身也在 NFS **soft 挂载**上
  ⇒ librknnrt 从它上面直读 `.rknn` 会被**读坏**（症状是**模型加载失败**，
  不是 SIGILL）⇒ 处置见「必读的坑」。
* ℹ️ **板上使用的落点是 `/tmp/p0_off_A_none.rknn`，四个测试程序的模型内置默认值
  也是它** ⇒ **零参数跑的前提是先把模型 `cat` 到 `/tmp`**（见「零参数运行」）。

```bash
# 模型先落到板卡 /tmp（tmpfs）并对账：md5 期望值 1a3faf2f9459cad8a389ffa81810234c
ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn > /tmp/p0_off_A_none.rknn \
  && md5sum /tmp/p0_off_A_none.rknn \
  && cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
  && cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce \
  && chmod +x /tmp/mce \
  && /tmp/mce \
     /root/bts/frames/VIDEO00007_19700101_001324 \
     /tmp/p0_off_A_none.rknn \
     /home/orangepi/runtime/kernels \
     21'
```

---

## 板卡依赖矩阵（本包**不携带任何第三方** .so/.a，但**随包发布第三方头**）

嵌入式组的要求：**直接用板卡上原来就有的 .so** ⇒ 交付包里**没有任何第三方
`.so` / `.a`**，全部动态链接板卡自带库。
⚠️ **但头文件随包**（`include/opencv4/` = OpenCV **4.8** 头，211 个文件）——
原因见下表第三行。

| 库 | 板卡路径 | SONAME | 版本 |
|:--|:--|:--|:--|
| OpenCV core/imgproc/imgcodecs/calib3d/features2d/flann/video | `/usr/lib/libopencv_*.so.408` | `libopencv_core.so.408` 等 | **OpenCV 4.8**（app 与本 SDK 链的**就是这套**） |
| OpenCV 头（**板上只有 4.6 的头**） | `/usr/include/opencv4`（`CV_VERSION_MINOR 6`） | — | 4.6 —— ⚠️ **不能**拿来配 4.8 的运行库 |
| OpenCV 头（**随包发布，本 SDK 用这份**） | `include/opencv4/`（包根下） | — | **4.8**（211 个文件；板上没有 ⇒ 外部提供） |
| RKNN Runtime | `/usr/lib/librknnrt.so` | `librknnrt.so` | **2.2.0** |
| RGA | `/usr/lib/aarch64-linux-gnu/librga.so` | `librga.so.2` | rga_api 1.10.0_[9] |
| OpenCL（ICD loader） | `/usr/lib/aarch64-linux-gnu/libOpenCL.so.1` | `libOpenCL.so.1` | OpenCL 3.0（Mali-G610 r0p0 / 4 CU） |
| OpenCL（真实现，Mali） | `/usr/lib/aarch64-linux-gnu/mali/libMaliOpenCL.so.1`（由 `/etc/OpenCL/vendors/mali.icd` 指定） | `libMaliOpenCL.so.1` | 链接期**不需要**它 |

> ℹ️ **Debian 系统包那套 4.6 也在板上**（`/usr/lib/aarch64-linux-gnu/libopencv_*.so.406`，
> dpkg `libopencv-core406` 等）⇒ **板上同时存在两套 OpenCV**：
> **4.8 在 `/usr/lib`**（本 SDK 用它）、**4.6 在 multiarch 目录**（系统包，本 SDK 不用）。
> ⇒ 运行库目录**按 ABI 后缀选**，不按「哪个目录存在」选；
> ⇒ 头版本与运行库 ABI **当场校验**（不符即构建失败退出）。

**三个阶段分别需要什么**：

* **编译期** —— 需要 **OpenCV 4.8 的头**（含 `opencv2/`）。
  ⚠️ **头与运行库必须同版本**：本包用**随包发布的 4.8 头**去链板卡 **4.8** 的
  `.so.408` ⇒ **绝对不要拿板上 `/usr/include/opencv4` 的 4.6 头配 4.8 运行库**
  （OpenCV 次版本会改 C++ ABI ⇒ 同一进程装进两套 OpenCV、`cv::Mat` 跨 `.so`
  边界串味 ⇒ 事故）。构建脚本会**当场校验**头版本 ↔ 运行库 ABI，不符即 die。
* **链接期** —— 需要那 7 个 `libopencv_*.so.408` + `librknnrt.so` + `librga.so.2`
  + `libOpenCL.so.1`（`build_lib.sh` / `build_demo.sh` 用下面这 4 个变量定位它们）。
  ⚠️ **板上没有 `libOpenCL.so` 开发软链**（只有 Debian 的 ICD loader `.so.1`）
  ⇒ **`-lOpenCL` 在板上必然链接失败** ⇒ 脚本按**绝对路径**链那个 `.so.1`
  （链接器取文件里的 SONAME ⇒ 产物 `DT_NEEDED` 写的是 `libOpenCL.so.1`，
  与「假如 `-lOpenCL` 能链上」的结果**完全一样** —— 换句话说：**只有链接行的
  写法变了，产物里记录的东西没变**）。
  ⚠️ **RGA 同样按绝对路径链** `/usr/lib/aarch64-linux-gnu/librga.so`
  （**不用 `-lrga`**：`/usr/lib` 里另有一份**不同的** BSP 版 `librga.so`，
  `-L` 顺序会挑到它；板上 `/home/orangepi/rockit/librga.so` 是第三份，app 不用它）。
* **运行期** —— 上面这些**全部由板卡 `ldconfig` 解析**
  （板上 `ldconfig -p` 已实测全部可解析）⇒ **不需要设 `LD_LIBRARY_PATH`**。
  唯一需要 `-rpath` 的是**本包自己的** `libmosaic_pipeline.so`：六个测试程序用
  `-Wl,-rpath,'$ORIGIN'` 找**同目录**的那一份 ⇒ **整包可搬**。
  （**例外**：`out/mosaic_api_test_static` 是**静态链接**版 ⇒ 它的 `DT_NEEDED` 里
  **没有**我们的 `.so` ⇒ **运行时不需要** `libmosaic_pipeline.so`；
  但上面那 7 个 OpenCV `.so` + RKNN Runtime / RGA / OpenCL **仍然要板卡自带**。）

> ⚠️ **交叉编译在宿主机上做时，上表那些路径指的是「目标板卡的文件系统」**：
> 宿主机的 `/usr` 是 **x86 的**，**绝对不能直接用**（会拿 x86 的头/库去编 aarch64
> ⇒ 编译失败或产出坏二进制）。
> ⇒ **先把板卡的那份取回宿主机**，再把变量指过去：
>
> ```bash
> export OPENCV_INCLUDE_DIR=<随包发布的 4.8 头目录>          # 如 $DEST/include/opencv4
> export OPENCV_LIB_DIR=<板卡-root>/usr/lib                 # 板卡 libopencv_*.so.408（4.8）
> export BOARD_LIBS=<板卡-root>/usr/lib/aarch64-linux-gnu   # libOpenCL.so.1 / librga.so
> export RT_LIB_DIR=<板卡-root>/usr/lib                     # librknnrt.so
> ```
>
> 变量名与 `build_lib.sh` / `build_demo.sh` 里的**完全一致**；不设时：
> **头**默认取**包内 `include/opencv4`**（随包发布的 4.8 头），
> **运行库**默认按 ABI 探测（`/usr/lib` → `/usr/lib/aarch64-linux-gnu` → `/usr/lib64`
> → `/root/mosaic/opencv` → `/home/orangepi/lib`，**只接受含 `libopencv_core.so.408`
> 的目录**）。头目录也可以用包内 `fetch_opencv_headers.sh` 取一份（`--src` / `--board`
> / `--expect-abi` / `--askpass`；版本不符即 die）。
> `EXTRA_LIB_DIRS`（可选）用于放 OpenCV 的**传递依赖**目录（libjpeg/libpng/
> libtiff/libgdal/…）—— 那些只在板卡运行时由 `ldconfig` 解析，交叉链接时**不必**
> 搬（链接行带 `-Wl,--allow-shlib-undefined`）。

---

## 本板实测结果（2026-11-25，板卡 `192.168.1.115` / Orange Pi 5）

> ℹ️ **本节数字全部来自 2026-11-25 在新板（Orange Pi 5 / Debian 12 / glibc 2.36）
> 上的实测**（旧板 `192.168.0.211` = Orange Pi CM5 与 `192.168.0.64` **均已作废**）。
> 测试台在板卡 `/root/bts/`：帧 `/root/bts/frames/VIDEO00007_19700101_001324`
> （**11,810** 帧 640×512 BMP）、模型 `/root/bts/p0_off_A_none.rknn`
> （md5 `1a3faf2f9459cad8a389ffa81810234c`）、kernel `/root/bts/kernels`。

### 上板门禁（全过）

| 门禁 | 结果 |
|:--|:--|
| 契约测试 `mosaic_contract_test`（含环境体检） | **`[RESULT] 212 passed, 0 failed`**，EXIT=0 |
| 最小集成示范 `mosaic_client_example`（**零参数**） | 21 帧 / **52 目标** / EXIT=0 |
| 多线程模板 `mosaic_async_client_example`（**零参数**） | 120 帧进 / 120 帧出 / **326 目标** / EXIT=0 |
| 两个探针 `mosaic_track_dump` / `mosaic_cadence_probe`（**零参数**） | EXIT=0 |
| **板卡自带库门禁** | ① `ldd` 全部解析成功、**无 `not found`**（板卡 `ldconfig` 全部解析）② `libmosaic_pipeline.so` 由 **`$ORIGIN`** 解析 ③ **不设 `LD_LIBRARY_PATH`** 也跑通 |

### 🔑 位级基线（**新板交付的硬指标**）

| 项 | 实测值 |
|:--|:--|
| 检测 TSV md5 | **`ceea5fe670a0e996eae0dbffc8c22c11`**（60 帧 / **168 检出** / 丢帧 0） |
| 三种构建形态的 TSV md5 | **静态版 / 共享版 / 零参数版三份 md5 全部相同** ⇒ 静态/共享两态逐位等价 |
| 与适配前对比 | **与适配前逐字节一致**（换板换 ABI **没有**动一个字节的检测结果） |
| `out/libmosaic_pipeline.so` | **{{LIB_SO_SIZE}} B**，md5 `{{LIB_SO_MD5}}`；`DT_NEEDED` = 7× `libopencv_*.so.408` + `libOpenCL.so.1` + `librknnrt.so` + `librga.so.2` |
| `out/libmosaic_pipeline.a` | **801,020 B**，md5 `b86391619f1d9e92973c1b4a1e6520d7`（**与上一轮交付逐字节相同** —— 静态库与板无关） |

⇒ **新板是逐位可复现的**：头随包（4.8）+ 运行库 4.8 严格配套 ⇒ 不再有
「换了 OpenCV 版本就不再逐位一致」的问题。

### e2e 全序列实测（11,810 帧，CANON28 权威参数：decoupled + D=3 + sparse）

> ⚠️ **下表是旧板（`192.168.0.211`，Orange Pi CM5，**已作废**）的历史实测**，
> 用来给出性能量级；**新板（Orange Pi 5）尚未复跑这条全序列 e2e**，
> 所以这里**不填新板的 fps**（不编造未实测的数字）。
> 新板已实测的是上文的**位级基线**与**上板门禁**。

| 指标 | 旧板实测（`192.168.0.211`，已作废） | 更早一板参考 |
|:--|:--|:--|
| fps | **8.564** | 12.811 |
| 交付帧 / GMC 组 | 4051 / 4553 | 6057 / 7122 |
| 延时 p50 / p95 / p99 | 344.9 / 594.4 / 683.2 ms | 314.5 / 444.4 / 463.2 ms |
| `rknn_run` p50 | 156.3 ms | ~104–113 ms |
| `gmc_fit` p50 | 64.2 ms | ~51.8–58.3 ms |
| 丢帧 / VOIDED | **0 / 0** | 0 / 0 |
| 墙钟 | 473.0 s | 472.8 s |

🔴 **fps 偏低（8.564 vs 12.811）的根因是板子热节流，不是 SDK**：
旧板负载中实测 **SoC 82–86 °C**（被动降频点 **75 °C**）、
`thermal-devfreq-0 cur=7 max=7`（冷却设备满档）、
**GPU 被压到最低 300 MHz**（可用到 1000 MHz）—— GMC 跑在 Mali-G610 上，直接被拖慢 60%。
⇒ **建议：改善散热，或把 GPU 频率固定到更高档。** 三种库组合实测 fps 都在 6.5–8.6，
说明与用哪套库无关；同一份二进制在**冷板**上就是 8.564 fps。

> ℹ️ 上表的 fps **随板卡温度浮动**（热节流 ⇒ 6.5–8.6）：墙钟恒为 473 s
> （= `--pace-ms 40` 的节拍下限），变的是「同样时间里能产出多少帧」。
> ⇒ **比 fps 更有意义的是「丢帧 0 / VOIDED 0」**：流水线没有丢任何一帧。

### 🔴 本版交付包含一个**死锁修复**（务必知道）

用交付包复跑 e2e 时抓到过一次**真死锁**：喂完 11,810 帧后卡在收尾，
**13 个线程全 `futex_wait`**、CPU 零增长、**一条错误日志都没有**，
e2e 自带的 `--watchdog-ms` 也看不见它（watchdog 住在**调用方**，而卡点在**库内部**的 `join()`）。

**根因**：`mosaic_pipeline_flush()` 的**丢唤醒**——退出标志在没有持有**等待方互斥量**的
情况下「置标志 + `notify_all()`」，等待方就可能错过这一次通知而永睡
（5 处退出标志全有窗口：`gmc_done` 有 3 个等待方 `m_feat`/`m_art`/`BufferA::m`、
`feat_done`→`m_heat`、`flushed`→`m_out`）。

**本包已修**：标志改为在**等待方持有的互斥量**下发布再 notify
⇒ 只有「已挂上被唤醒」或「之后拿锁时谓词已为真」两种情况，**没有第三种**。

**复验**：修复后 e2e **EXIT=0** 跑完（上表即它的实测值）；且
**行为逐位中性**——与修复前的输出比，共同帧 **1,830 帧特征 md5 全部逐位相同**、
**1,313 帧检测载荷逐字节全同**；官方校验器 **ΔF1 仍 = +0.000000**。
⇒ **修复只动同步，不动任何计算。**

🔒 **同类已彻底封口**：复核时发现 `set_error()`（**NPU 报错路径**）也是「置 `abort` 标志 +
`notify_all()`」而不持等待方的锁 ⇒ 通知照样可能丢 ⇒ **库内 9 处等待已全部改为带 200 ms
超时的 `wait_for`**（公开 `mosaic_pipeline_collect()` 另加显式循环，避免「谓词为假」被
误判成 aborted 而误报 -1）。
⇒ **本库不再存在任何无超时的内部等待** ⇒ 「通知丢失 ⇒ 线程永睡 ⇒ join 永挂 ⇒
一条错误日志都没有」这一类静默死锁被消除（e2e 的 watchdog 看不见库内 hang 的问题随之不成立）。
复验：e2e **EXIT=0**、**行为逐位中性**、官方 **ΔF1 仍 = +0.000000**、fps 8.310（在跑间抖动范围内）。


### ✅ 取舍：新板**头随包** ⇒ 「逐位可复现」反而保住了

| 组合 | OpenCV | RKNN Runtime | 与基准**逐位相同** | ΔF1（官方校验器） |
|:--|:--|:--|--:|:--|
| **本包默认（新板 `192.168.1.115`）** | **4.8**（头随包 `include/opencv4/` + 板卡 `.so.408`） | 板卡 2.2.0 | 检测 TSV md5 **`ceea5fe670a0e996eae0dbffc8c22c11`**，**与适配前逐字节一致** ✅ | — |
| 对照（旧板口径，**已作废**） | 板卡 **4.6**（头与库都是板上那套） | 板卡 2.2.0 | 特征 990/3742（**26.5 %**） | **+0.000000** |
| 对照（旧板口径，**已作废**） | 交叉编译 **4.10** | 板卡 2.2.0 | 特征 **3535/3535（100 %）** ✅ | +0.000000 |

* **旧板那次取舍（历史）**：当时板卡上只有 OpenCV **4.6 的头**，用它去编就得配 4.6 的
  运行库 ⇒ 官方校验器实测 `arm@S` 与 `ref@S` 的 **TP / FP / FN / Recall / Precision /
  F1 全等**（**ΔF1 = +0.000000**，换 OpenCV 只改最低位：框坐标差 ~0.04 px、area ±1、
  peak ~5e-4，**不改变任何 TP/FP/FN 判定**），但**特征 md5 逐位相同率只有 26.5 %**
  ⇒ 校验器的 `VERDICT` 判据（`different == 0`）会报 **FAIL**。
  那**不是精度退化**，是「换了 OpenCV 版本 ⇒ 不再逐位可复现」。
* **新板没有这个取舍了**：目标板（Orange Pi 5 / Debian 12）上**只有 4.6 的头**
  （`/usr/include/opencv4`），但**运行库是 4.8** ⇒ 本包**把 4.8 的头随包发布**
  （`include/opencv4/`），头与运行库**严格配套** ⇒
  **实测 TSV md5 与适配前逐字节相同** ⇒ 「必须与基准逐位一致」的门禁**直接满足**。
* ⇒ **新板的取舍只剩一条**：包里多了 211 个 OpenCV 头文件（第三方 BSD 许可），
  换来「头/库版本永远配套、换板不用重取头」。
  ⚠️ 反过来，**换到只有别的次版本的板子**时（例如板上是 4.6 运行库），
  就必须用 `OPENCV_ABI=406` 换成配套的头（`fetch_opencv_headers.sh --expect-abi 406`），
  **头与库不配套 ⇒ 同一进程装进两套 OpenCV ⇒ 事故**。

### ✅ 位级门禁该拿谁当参照？（**2026-11-25 更新**：换板后参照已重建）

> ⚠️ 历史上换过三块板，**前两块均已作废**：
> `192.168.0.64`（Ubuntu 20.04，交叉编译 OpenCV 4.10，2026-10-28 产出「认证基准」
> `A_gate_step2_nodrop`）→ `192.168.0.211`（Orange Pi CM5 / Debian 12，2026-11-10 换上，
> 11-12 重建过一次 golden）→ **`192.168.1.115`（Orange Pi 5 / Debian 12，**当前目标板**）**。
> ⇒ **拿作废板上的基准跟本包产物比，位级必然对不上**，这不是本包的缺陷。

| 参照 | 产自 | 与本包产物逐位相同率 | 官方校验器判定 |
|:--|:--|--:|:--|
| 🔴 旧「认证基准」`A_gate_step2_nodrop` | **作废板**（`192.168.0.64`，OpenCV 4.10） | **19.97 %** | ❌ FAIL（ΔF1 −0.000103） |
| 🟡 中间板 golden（11-12 重建） | **作废板**（`192.168.0.211`，板卡自带 OpenCV 4.6 / RKNN 2.2.0） | **100.00 %**（共同帧 **11806/11806**） | ✅ **PASS**（ΔF1 = **+0.000000**） |
| ✅ **当前板基线**（11-25 实测） | **当前板 `192.168.1.115`**（OpenCV 4.8：头随包 + 板卡 `.so.408` / RKNN 2.2.0） | **检测 TSV md5 `ceea5fe670a0e996eae0dbffc8c22c11`，与适配前逐字节一致**；静态版 / 共享版 / 零参数版三份**全同** | ✅ **PASS**（与适配前逐字节相同 ⇒ TP/FP/FN/R/P/F1 全等） |

⇒ **位级门禁请用「当前板（`192.168.1.115`）的基线」当参照**，不要用任何作废板上的。
⇒ 重建方法（板卡上，约 52 min）：
  ```bash
  ./rknn_mosaic_e2e --raw-root <帧目录> --sequence <序列名> \
    --model <模型.rknn> --kernel-dir <kernels 目录> --out <输出目录> \
    --anchor-step 2 --npu-workers 2 --gmc-workers 2 --post-workers 1 \
    --input-queue 8 --jitter-depth 8 \
    --gmc-fit-corners 600 --gmc-fit-win 15 --gmc-fit-level 2 --gmc-fit-iters 20
  ```
  ⚠️ 用 `setsid nohup` 脱离运行：**TSV 只在跑完后落盘**，`timeout` 杀进程会拿不到可比对的产物。

### ⚠️ 「2 帧」是**帧集合边界**，不是「有 2 帧算不出来」

opportunistic 引擎忙不过来时会「取最新帧、跳过中间的」，**跳哪 2 帧每轮都不同**
（实测：e2e 路径跳 `000001/000002`，异步 `push-all` 路径跳 `000199/000241`）。
⇒ 两条路径的产物**各有 11808 帧、但不是同一批**；
⇒ **裁到交集帧**（两臂都产出了的那些）再比 ⇒ **逐位 100%**（11806/11806）。
⇒ 两轮合起来覆盖**全部 11810 帧**，**没有哪一帧是算不出来的**。

### 🔁 跨环境（换板 / 换库）时的等价判据

位级 md5 只在「同一块板 + 同一套库」下成立。换板后请用**容差判据**（实测值，已实测通过）：

| 指标 | 判据 | 实测（作废板基准 vs 当时产物） |
|:--|:--|:--|
| 检出数变化率 | ≤ 1 % | **0.56 %** ✅ |
| 坐标容差覆盖率（\|Δcx\|,\|Δcy\| ≤ 0.5 px 的帧占比） | ≥ 99 % | **99.48 %** ✅ |
| 每帧最大坐标差 | —— | 中位 **0.0000 px**、p95 0.11 px、p99 0.31 px |
| ΔF1 | ≥ −1.6e-4 | **−0.000103** ✅ |

⇒ **亚像素抖动 ≠ 精度退化**（TP/FP/FN 判定一个都没变）。
ℹ️ **当前板不需要用容差判据**：头随包 + 运行库严格配套 ⇒
**换板后检测 TSV md5 与适配前逐字节相同**（见上文「位级基线」）。

---

### 🔴 板卡 RGA 驱动会**偶发内核 oops**（**嵌入式组必读**）

> ℹ️ **观测记录来自已作废的 Orange Pi CM5**（`192.168.0.211`，Debian 12 / 内核
> `5.10.160-rockchip-rk3588`）—— 保留在文档里是因为**当前目标板
> `192.168.1.115`（Orange Pi 5）用的是同一代 Rockchip RGA 驱动栈、同一 glibc 2.36
> 发行版**，这个坑**有可能复现**，集成方必须知道怎么查。
> 当前板 2026-11-25 适配实测（契约测试 212 passed + 上板门禁 + 零参数 demo + 两个探针）
> **未触发**该 oops。

**现象**：在已作废的板卡 `192.168.0.211`（Orange Pi CM5 / Debian 12 / 内核 `5.10.160-rockchip-rk3588`）上
跑本包的测试程序时，`dmesg` 偶发出现：

```
rga_mm: This is not a buffer that has been imported, handle = -1
rga: buffer[0] mm release buffer failed! handle = -1
Unable to handle kernel paging request at virtual address ffffff8000000000
Internal error: Oops: 96000146 [#N] SMP
CPU: 7 PID: 18287 Comm: mosaic_contract Tainted: G D  5.10.160-rockchip-rk3588
  __clean_dcache_area_poc / iommu_dma_sync_single_for_device /
  dma_sync_single_for_device / rga_mm_sync_dma_sg_for_device /
  rga_mm_get_buffer / rga_mm_get_channel_handle_info /
  rga_mm_map_job_info / rga_job_commit / rga_request_commit
```

⇒ **这是内核态崩溃，不是本包的用户态 bug**：崩在 Rockchip RGA 驱动的
`rga_mm_get_buffer()` 里，它拿到 `handle = -1` 后**没有做空指针检查**就去做
`dma_sync_single_for_device`，于是解引用 `ffffff8000000000`。

**触发方确实是本包的二进制**（dmesg 里的 `Comm:` 就是 `rknn_mosaic_e2e` /
`mosaic_contract`），**后果**是被打死的那个进程会**永久死锁**
（实测：契约测试跑了 7.5 min 只用掉 13 s CPU 就再也醒不过来，线程全卡在
`futex_wait_queue_me` / `do_sys_poll`）。

**三块板的差异（这才是根因）**：

| | 作废板（`192.168.0.64`） | 作废板（`192.168.0.211`） | **当前板（`192.168.1.115`）** |
|:--|:--|:--|:--|
| 型号 | RK3588 开发板 | Orange Pi CM5 | **Orange Pi 5** |
| 系统 | Ubuntu 20.04.6 | Orange Pi 1.0.0 Bookworm / Debian 12 | **Debian 12 (Bookworm) / aarch64** |
| glibc | —— | —— | **2.36**（`GLIBC_2.36-9+deb12u4`） |
| 板载 gcc | —— | —— | **12.2.0**（Debian） |
| 内核 | —— | `5.10.160-rockchip-rk3588` | **`5.10.160-rockchip-rk3588`** |
| 内核 RGA 驱动 | **`rga v1.3.1`** | **`rockchip_rga` `v1.3.0`** | 同代驱动栈（**具体小版本未复核**） |
| RGA 库 | `librga.so.2` md5 `7136ae54…`（192,712 B） | `librga.so.2` md5 `32c4b08f…`（200,896 B） | `/usr/lib/aarch64-linux-gnu/librga.so`（SONAME **`librga.so.2`**）；⚠ 另有 `/home/orangepi/rockit/librga.so` 是**不同的**一份，app 不用它 |
| RGA API 版本 | `1.10.0_[9]` | `1.10.0_[9]` | `1.10.0_[9]`（头随包 `include/rga/`，**相同**） |
| **RGA ABI 探针** | 未跑 | 未跑 | ✅ **通过**（`sizeof(rga_buffer_t)=96`、`offsetof(handle)=92`、handle 回读 `0x12345678` 一致 ⇒ 与本 SDK 的假设一致） |
| 是否 oops | 未观测到 | **2 次 / 约 1.5 h** | **本轮适配实测未触发**（契约 + 门禁 + demo + 探针全过） |

⇒ **用户态 API 版本相同，是内核 RGA 驱动版本不同**（v1.3.1 → v1.3.0，中间那块反而更旧）。
**这属于板卡/厂商侧的驱动缺陷，本包无法在用户态彻底消除。**
⇒ **当前板（Orange Pi 5）的 RGA ABI 与本 SDK 假设一致（探针通过）**，
但驱动栈同代 ⇒ **集成方仍应按下面的方法做 oops 计数巡检**。

**本包已做的两件事**：

1. **修掉一处无效 API 用法**：`mosaic_core.h` 的 `ComposeFrame()` 原先把
   `s->pat`（从未 import，`handle` 恒为 `-1`）以 `wrapbuffer_handle(-1, …)`
   传给 `improcess()`。现已改为会话创建时 import 一次。
   **该改动对输出零影响**：同一份 e2e 二进制分别挂新旧两版库跑 300 帧，
   `feature_md5.txt` 与 `<seq>_mosaic.tsv` **逐字节相同**
   （md5 `015649b0…` / `b1882de0…`）。
   ⚠️ 但它**不是** oops 的根因 —— 实测两次运行的 `not a buffer that has been
   imported` 计数**完全相同**（各 6 次），说明该内核报错与本包的 `pat` 无关。
2. **给交付门禁加硬超时**：`copy_to_nfs.sh --verify` 现在每个板卡测试都套
   `timeout -k 10 ${BOARD_TEST_TIMEOUT:-600}`。没有它的话，一次 oops 就会让
   `--verify` **无限期挂死**且日志无任何线索（只会看到某个 `[1/5]` 之后再无输出）。
   超时会打印 `[timeout] …` 并强杀后重试一次。

**嵌入式组/平台方需要做的**：

- 升级目标板的 **RGA 内核驱动**（v1.3.0 → v1.3.1 或厂商修订版），或
  向 Orange Pi 镜像维护方报这个 oops（**当前目标板 `192.168.1.115` 用的是同代
  驱动栈，这个坑有可能复现**）；
- 排查现场用：`dmesg | grep -c 'Internal error: Oops'`（**计数是否增加**就是判据），
  以及 `dmesg | grep -c 'not a buffer that has been imported'`；
- 上板自检脚本里建议加一条「**跑之前/跑之后各记一次 oops 计数**」，
  与本项目 §20.8 第 6 条「每臂跑完后核对 `dmesg` 的 `rga: buffer[0] mm release
  buffer failed!` 计数」是同一条思路 —— 那条指标当时被记为「疑似跨进程资源泄漏」，
  **现在已知它与内核 oops 同源**。

---

## ① 单独编译库

```bash
cd {{PKG_ROOT}}
bash build_lib.sh
```

产物：`out/mosaic_pipeline.o` + `out/libmosaic_pipeline.so`（**共享库**，
SONAME = `libmosaic_pipeline.so`）。

**等价的一行命令**（不依赖脚本，可直接粘贴）：

```bash
cd {{PKG_ROOT}} && \
TC={{TOOLCHAIN}} && \
mkdir -p out && \
"$TC/bin/aarch64-rockchip-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations -DCL_TARGET_OPENCL_VERSION=300 \
  -fPIC -Iinclude -Iinclude/rga -Isrc/cpp -Isrc/opencl/host -I{{OPENCV_INCLUDE_DIR}} \
  -c src/npu/mosaic_pipeline.cpp -o out/mosaic_pipeline.o && \
rm -f out/libmosaic_pipeline.so && \
"$TC/bin/aarch64-rockchip-linux-gnu-g++" -shared -Wl,-soname,libmosaic_pipeline.so \
  -Wl,--allow-shlib-undefined \
  -o out/libmosaic_pipeline.so out/mosaic_pipeline.o \
  -L{{RT_LIB_DIR}} -L{{BOARD_LIBS}} -L{{OPENCV_LIB_DIR}} \
  -Wl,-rpath-link,{{RT_LIB_DIR}} -Wl,-rpath-link,{{BOARD_LIBS}} -Wl,-rpath-link,{{OPENCV_LIB_DIR}} \
  -Wl,--no-as-needed \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core \
  {{BOARD_LIBS}}/libOpenCL.so.1 {{BOARD_LIBS}}/librga.so \
  -lrknnrt -lpthread -ldl -lm
```

> 📌 `-fPIC` 与 `-shared` 是共享库的必需项；`-Wl,-soname,libmosaic_pipeline.so`
> 决定集成方按哪个名字链接；`-Wl,--allow-shlib-undefined` 允许 OpenCV 的传递依赖
> （libjpeg/libpng/libgdal/…）留到**板卡运行时**由 `ldconfig` 解析。
>
> 🔴 **`-lOpenCL` 在新板上必然链接失败**（板上只有 Debian 的 ICD loader
> `libOpenCL.so.1`，**没有 `libOpenCL.so` 开发软链**）⇒ 上面那行用
> **绝对路径** `{{BOARD_LIBS}}/libOpenCL.so.1` 参与链接。链接器取的是文件里的
> **SONAME**（就是 `libOpenCL.so.1`）⇒ 产物的 `DT_NEEDED` 仍写
> `libOpenCL.so.1`，与 app 链的那份一致。
> ⚠️ 板上的**真 GPU 实现**是 `/usr/lib/aarch64-linux-gnu/mali/libMaliOpenCL.so.1`
> （由 `/etc/OpenCL/vendors/mali.icd` 指定）—— 那是**运行期**的事，链接期不需要它。
>
> 🔴 **同理不能用 `-lrga`**：`/usr/lib` 里另有一份**不同的** BSP 版 `librga.so`，
> 而 OpenCV 4.8 也在 `/usr/lib` ⇒ `-L` 顺序一不小心就会挑到 BSP 那份
> （app 的 `ldd` 解析到的是 multiarch 那份 `/usr/lib/aarch64-linux-gnu/librga.so`
> ⇒ SONAME `librga.so.2`）⇒ 上面用**绝对路径** `{{BOARD_LIBS}}/librga.so` 锁死。
> ℹ️ **三个脚本在这一点上不完全一样**（别踩）：**`build_board_native.sh`（板卡原生）
> 已经按绝对路径锁 multiarch 那份**；`build_lib.sh` / `build_demo.sh`（宿主机交叉编译）
> 的链接行仍写 `-lrga` ⇒ **在板上跑这两个脚本时**，要保证 `-L` 的顺序先命中
> `/usr/lib/aarch64-linux-gnu`（或把 `BOARD_LIBS` 指到 multiarch 那份的拷贝）。
>
> 📌 `-Wl,--no-as-needed` 不能省：Debian 的 gcc **默认 `--as-needed`**，会把
> 「链了但没直接用到」的 `libopencv_features2d` / `libopencv_flann` 从 `NEEDED`
> 里丢掉 ⇒ 交付产物的 `NEEDED` 集合（7 个 OpenCV 模块）就会与 app 链的对不上。
>
> 📌 **OpenCV 的版本由 `OPENCV_ABI` 决定**（`build_lib.sh` / `build_demo.sh` /
> `build_board_native.sh` **三者同名同义**，默认 **408** = 4.8）：
> `OPENCV_VERSION` 默认由 ABI 推导（408→4.8 / 406→4.6 / 410→4.10），
> `OPENCV_INCLUDE_DIR` 默认**包内 `include/opencv4`**（4.8 头），
> `OPENCV_LIB_DIR` 默认按 ABI 自动探测（`/usr/lib` → `/usr/lib/aarch64-linux-gnu`
> → `/usr/lib64` → `/root/mosaic/opencv` → `/home/orangepi/lib`，
> **只接受含 `libopencv_core.so.$OPENCV_ABI` 的目录**），
> `BOARD_LIBS` 默认 `/usr/lib/aarch64-linux-gnu`，`RT_LIB_DIR` 默认 `/usr/lib`。
> 脚本还会做**头版本 ↔ ABI 校验**：头版本与 `OPENCV_ABI` 对不上就**当场 die**
> 并给出可操作指引（头版本按 `opencv2/core/version.hpp` 读，
> 库则按**精确的 `.so.$ABI`** + **SONAME 守卫**判定，避免 `.so` 软链指向
> 另一个次版本时被误判为「有」）。
>
> 📌 这里的 `-I{{OPENCV_INCLUDE_DIR}}` / `-L{{OPENCV_LIB_DIR}}` 指向
> **OpenCV 4.8 的头与库**（默认 = **包内 `include/opencv4`** 与 `/usr/lib`）——
> ⚠️ **板上 `/usr/include/opencv4` 只有 4.6 的头**，拿它配 4.8 的库就是
> **ABI 错配**（编译期结构体布局按 4.6 头、运行期按 4.8 库解释 ⇒ 踩内存）。
> **交叉编译时这些路径指的是板卡的文件系统**，宿主机的 `/usr` 是 x86 的，
> **绝对不能直接用**（见「板卡依赖矩阵」）。
> 📌 `{{BOARD_LIBS}}` 在板上就是 `/usr/lib/aarch64-linux-gnu`
> （⇒ 上面两个绝对路径即 `/usr/lib/aarch64-linux-gnu/libOpenCL.so.1` 与
> `/usr/lib/aarch64-linux-gnu/librga.so`）；
> **宿主机交叉编译**时把它指向**从板卡取回来的那份**（`fetch_board_libs.sh`），
> 否则链的是宿主机的 x86 库。
> `TC` 是**唯一还要你提供的东西**（交叉编译器）—— 换编译器就改 `TC=` 那一行，
> 或改用 `RK3588_TOOLCHAIN` / `CROSS_CXX` 环境变量让脚本去找。

---

## ② 单独编译测试程序

依赖 ① 产出的 `out/libmosaic_pipeline.so`（链接的就是它）与
`out/libmosaic_pipeline.a`（静态链接版 demo 链的是它）。

```bash
cd {{PKG_ROOT}}
bash build_demo.sh
```

产物：`out/mosaic_api_test`、`out/mosaic_client_example`、
`out/mosaic_async_client_example`、`out/mosaic_contract_test`、`out/rknn_mosaic_e2e`，
外加**静态链接版**的 `out/mosaic_api_test_static`（它链的是 `out/libmosaic_pipeline.a`
⇒ **静态库能不能链通、能不能跑**的活证据；**运行输出与链 `.so` 的
`out/mosaic_api_test` 逐字节一致**）。

**等价的一行命令**（示例 demo；把源文件换成 `src/npu/mosaic_contract_test.cpp`、
产物换成 `out/mosaic_contract_test` 即得契约测试）：

```bash
cd {{PKG_ROOT}} && \
TC={{TOOLCHAIN}} && \
mkdir -p out && \
"$TC/bin/aarch64-rockchip-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations -DCL_TARGET_OPENCL_VERSION=300 \
  -Iinclude -Iinclude/rga -Isrc/cpp -Isrc/opencl/host -I{{OPENCV_INCLUDE_DIR}} \
  src/npu/mosaic_api_test.cpp -o out/mosaic_api_test \
  -Lout -lmosaic_pipeline -Wl,-rpath,'$ORIGIN' \
  -L{{RT_LIB_DIR}} -L{{BOARD_LIBS}} -L{{OPENCV_LIB_DIR}} \
  -Wl,-rpath-link,{{RT_LIB_DIR}} -Wl,-rpath-link,{{BOARD_LIBS}} -Wl,-rpath-link,{{OPENCV_LIB_DIR}} \
  -Wl,--allow-shlib-undefined \
  -Wl,--no-as-needed \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core \
  {{BOARD_LIBS}}/libOpenCL.so.1 {{BOARD_LIBS}}/librga.so \
  -lrknnrt -lpthread -ldl -lm
```

> 📌 `-Lout -lmosaic_pipeline` 链接**同目录**的共享库，
> `-Wl,-rpath,'$ORIGIN'` 让它**运行时**也从同目录找 ⇒ **整包可搬**，
> 不需要设 `LD_LIBRARY_PATH`。
>
> 📌 其余依赖路径全是**板卡的系统路径**（`{{OPENCV_LIB_DIR}}` / `{{RT_LIB_DIR}}` /
> `{{BOARD_LIBS}}`）—— 板卡上都有，**不在包里**（见「板卡依赖矩阵」）。

---

## ③ 把整包拷到 NFS 挂载点

交付包是**源码 + 预编译产物**的整棵树，所以这一步拷的是**整棵树**：源码 +
第三方头 + kernel 资产 + 编译产物 + 脚本 + 文档（不再是拷几个产物文件）。

**推荐：一键脚本**（整树镜像 + 递归 md5 回验 + 建兼容别名）：

```bash
bash {{PKG_ROOT}}/copy_to_nfs.sh
```

**手写等价命令**（脚本不可用时）：

```bash
SRC={{PKG_ROOT}}
DST={{NFS_ROOT}}/mosaic_sdk
mkdir -p "$DST"

# ① 先删掉【上一版遗留的平铺实文件/实目录】—— cp -a 是「合并」语义，
#    不会删；不删就会留下**旧二进制**，让人误以为在跑新版
#    （**更早**的版本是 .a 归档 + 第三方依赖目录；本版是 .so + .a ⇒ 旧目录都要清，
#     别把两版的产物混在同一棵树上）
rm -f  "$DST/mosaic_api_test" "$DST/mosaic_contract_test" \
       "$DST"/libmosaic_pipeline.* "$DST/mosaic_pipeline.h"
rm -rf "$DST/kernels" "$DST/third_party"

# ② 整树镜像
cp -a "$SRC/src" "$SRC/include" "$SRC/runtime" "$SRC/models" "$SRC/out" "$DST/"
cp -a "$SRC/README.md" "$SRC/MANIFEST.md" \
      "$SRC/build_lib.sh" "$SRC/build_demo.sh" "$SRC/copy_to_nfs.sh" \
      "$SRC/known_hosts" "$DST/"
chmod +x "$DST/build_lib.sh" "$DST/build_demo.sh" "$DST/copy_to_nfs.sh" \
         "$DST/out/mosaic_api_test" "$DST/out/mosaic_client_example" \
         "$DST/out/mosaic_async_client_example" \
         "$DST/out/mosaic_contract_test" "$DST/out/rknn_mosaic_e2e"

# ③ 建 6 个兼容符号链接（**相对路径**），旧命令才继续可用
( cd "$DST" && \
  ln -sfn out/mosaic_api_test      mosaic_api_test && \
  ln -sfn out/mosaic_async_client_example mosaic_async_client_example && \
  ln -sfn out/mosaic_contract_test mosaic_contract_test && \
  ln -sfn out/libmosaic_pipeline.so libmosaic_pipeline.so && \
  ln -sfn src/npu/mosaic_pipeline.h mosaic_pipeline.h && \
  ln -sfn runtime/kernels           kernels )
```

> ⚠️ **别省 ① 和 ③**：少了 ① 会留下**上一版的旧二进制**（跑的是旧程序却以为是新版）；
> 少了 ③ 则「旧写法仍然可用」这句话不成立。
> **更省心的做法是用 `bash {{PKG_ROOT}}/copy_to_nfs.sh`**（它三步全做，还带递归 md5 回验）。

> **拷完必须验这两样**，缺一个就说明交付包是残的：
>
> ```bash
> ls {{NFS_ROOT}}/mosaic_sdk/src/npu/mosaic_api_test.cpp   # 源码在（客户能参考、能重编）
> ls {{NFS_ROOT}}/mosaic_sdk/out/mosaic_api_test          # 产物在
> ```

**挂载点对应关系**：NFS 在本机是 `{{NFS_ROOT}}`，在板卡上是
`/mnt/manu` ⇒ 拷过去之后，板卡从 `/mnt/manu/mosaic_sdk/` 就能看到这整棵树。

---

## ④ 运行命令（板卡 `{{BOARD}}:{{BOARD_PORT}}`）

```bash
# ① 模型先落到板卡 /tmp（tmpfs）并对账（md5 期望值 1a3faf2f9459cad8a389ffa81810234c）
# ② .so 与可执行文件也 cat 到 /tmp（$ORIGIN 要同目录）
ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn > /tmp/p0_off_A_none.rknn \
  && md5sum /tmp/p0_off_A_none.rknn \
  && cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
  && cat /mnt/manu/mosaic_sdk/out/mosaic_api_test > /tmp/mosaic_api_test \
  && chmod +x /tmp/mosaic_api_test \
  && /tmp/mosaic_api_test \
  --raw-root   /root/bts/frames \
  --sequence   VIDEO00007_19700101_001324 \
  --model      /tmp/p0_off_A_none.rknn \
  --kernel-dir /home/orangepi/runtime/kernels \
  --limit 12'
```

> ⚠️ **本板（`192.168.1.115`）上没有 `/mnt/manu` 这个挂载点** ——
> `/mnt/manu/mosaic_sdk` 是**NFS 交付落点**（嵌入式组那边把整包挂 NFS，板卡侧
> 通过 `/mnt/manu/...` 访问）；**本板实测**用的是板卡自己的布局：
> kernel = `/home/orangepi/runtime/kernels`、帧 = `/root/bts/frames`、
> 模型 = `/root/bts/p0_off_A_none.rknn`（先 `cat` 到 `/tmp`）。
> ⇒ 上面命令里的 `--raw-root` / `--kernel-dir` 换成本板路径即可；
> **不给这两个参数**也行（程序走 `mosaic_paths.h` 的回落，会自动命中本板路径）。

> ⚠️ **共享库带来的新要求**：测试程序用 `-rpath '$ORIGIN'` 找**同目录**的
> `libmosaic_pipeline.so` ⇒ **`.so` 必须和可执行文件放在同一个目录**。
> （**例外**：`out/mosaic_api_test_static` 是**静态链接**版 ⇒ **不用** `cat` 我们的 `.so`。）
> 上面的命令**模型 / `.so` / 可执行文件都 `cat` 到 `/tmp`** ⇒ `$ORIGIN` = `/tmp` ⇒ 能找到。
> （只拷可执行文件、不拷 `.so` ⇒ 程序起不来，报 `libmosaic_pipeline.so: cannot
> open shared object file`。）
> ℹ️ **模型同理**：它有 6.3 MB，从 NFS soft 挂载直读会被读坏 ⇒ **必须先 `cat`
> 到 `/tmp`**，`--model` 指的是 **`/tmp/p0_off_A_none.rknn`**（不是包内路径）。
>
> ℹ️ **为什么要绕这一圈（`cat` 到 `/tmp` 再跑）**：板卡上的 `/mnt/manu` 是
> **NFS soft 挂载（软挂载）** ⇒ 从它上面**直接执行**超过 1 MB 的二进制会被读坏
> （旧版静态链接 OpenCV 时实测 8.3 MB 的 `out/mosaic_api_test` 直接跑 ⇒
> **SIGILL(132)**）。
> 铁律：**>1 MB 的二进制一律先 `cat` 到板卡 `/tmp`（tmpfs）再执行**，
> `chmod +x` 后再跑。详见「必读的坑」。
> ⚠️ 本版改成共享库后，产物**全部 < 1 MB**（最大的 `out/libmosaic_pipeline.so`
> 约 417 KB，六个可执行文件**都远小于 1 MB**）⇒ 这条铁律**当前不再触发**；
> 但下面的命令**仍然 `cat` 到 `/tmp`**（稳妥，而且 `.so` 与可执行文件本来就要
> 待在一起）。
>
> **兼容别名**：交付根目录留了指向 `out/` 与 `runtime/kernels` 的相对符号链接，
> 所以旧写法 `/mnt/manu/mosaic_sdk/mosaic_api_test --kernel-dir /mnt/manu/mosaic_sdk/kernels`
> **仍然可用**（`kernels` 是 `runtime/kernels` 的别名）。新代码一律用上面
> `out/` + `runtime/kernels` 的写法。

---

## 注意事项

* **契约测试**：`out/mosaic_contract_test`（同样参数，同样**先落到 `/tmp` 再跑**，
  **`.so` 一起拷**）：

  ```bash
  ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn > /tmp/p0_off_A_none.rknn \
    && md5sum /tmp/p0_off_A_none.rknn \
    && cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
    && cat /mnt/manu/mosaic_sdk/out/mosaic_contract_test > /tmp/mosaic_contract_test \
    && chmod +x /tmp/mosaic_contract_test \
    && /tmp/mosaic_contract_test --raw-root /root/bts/frames \
    --sequence VIDEO00007_19700101_001324 \
    --model /tmp/p0_off_A_none.rknn \
    --kernel-dir /home/orangepi/runtime/kernels --limit 60'
  ```

  板卡实测 **217 passed / 0 failed**（2026-11-25，本板 `192.168.1.115`；
  更早一轮在作废板 `192.168.0.211` 上是 196 passed / 0 failed）。
  它是库的验收门禁——客户环境跑通这个，
  说明库的行为是符合契约的。
* **demo 的全部选项与默认值**（`src/npu/mosaic_api_test.cpp`）：

  | 选项 | 默认值 | 说明 |
  |---|---|---|
  | `--raw-root` | `mosaic_paths::FramesRoot()`（`MOSAIC_RAW_ROOT` → `/root/bts/frames` → `/mnt/manu/lq_ir_bmp`） | 帧根目录，只用于环境体检和拼帧目录 |
  | `--sequence` | `VIDEO00007_19700101_001324` | 序列名，帧目录是 `<raw-root>/<sequence>/frame_%06d.bmp` |
  | `--model` | `/tmp/p0_off_A_none.rknn` | RKNN 模型路径（**板上落点**；模型**来源**是本包 `models/p0_off_A_none.rknn`，须先 `cat` 到 `/tmp`） |
  | `--kernel-dir` | `mosaic_paths::KernelDir()`（`MOSAIC_KERNEL_DIR` → `<包根>/runtime/kernels` → `/home/orangepi/runtime/kernels` → …） | OpenCL kernel 目录 |
  | `--limit` | `60` | 最多跑多少帧（`0` = 不限） |
  | `--no-env-gate` | 关（默认开体检） | 跳过开库前的环境体检 |
  | `--post-mode` | `sparse` | 后处理算法：`sparse`（默认）/ `dense` / `verify` / `gpu` |
  | `--dump-tsv` | 不导出 | 把逐帧检测导出成 e2e 同格式的 TSV |

  不带参数也能直接跑（默认值由 `mosaic_paths.h` 逐级回落 ⇒ 命中本板布局）。
* **后处理有四种算法（v10.3）**，由 `--post-mode` 或环境变量 `MOSAIC_POST_MODE` 选：

  | 模式 | 板卡实测（RK3588，600 帧） | 全序列门禁 VERDICT | 说明 |
  |---|---|---|---|
  | **`sparse`（默认）** | **0.531 ms/帧**，σ 0.219，最坏帧 1.037 | **PASS** | CPU NEON 稀疏路径：只求值「可能过阈值」的像素。**输出与 `dense` 逐比特一致**（全序列上 6034 个共同帧的检测框逐条相同、feature md5 7113/7113 相同），已认证的历史数字不需要重跑 |
  | `dense` | 37.856 ms/帧，σ 0.728，最坏帧 40.665 | PASS | 参考实现：全图双线性上采样 + 全图 8 连通域。留作对照与退路 |
  | `verify` | ≈ dense 的两倍 | — | 诊断用：同一帧同时跑 dense 与 sparse 并逐字段比对 |
  | `gpu` | 稳态 0.857 ms/帧（σ 0.039） | **FAIL** | OpenCL 局部极大路径（跑在 Mali 上）。**不是 dense 的即插即用替代**（检测流 5666 条不同、假警率高 50 %），且有一次性 OpenCL 初始化尖刺（首帧约 54 ms） |

  换成参考实现：`MOSAIC_POST_MODE=dense` 或 `--post-mode dense`。
* **GMC 的拟合分辨率**（环境变量 `MOSAIC_GMC_DOWNSCALE`，1..4，默认 **2**）：
  640×512 的特征帧在**做角点检测与 LK 拟合之前**先按这个倍数缩小，
  拟合出的变换再放大回原生像素。

  | 值 | 拟合分辨率 | 板卡实测（GMC 拟合耗时，600 帧） | 精度（600 帧） |
  |---|---|---|---|
  | **2（默认 / 冻结值）** | **320×256** | 均值 64.624 ms｜p95 144.609 ms｜max 170.112 ms | 基准：TP 398 / FP 19 / FN 169，F1 **0.808943** |
  | **4** | **160×128** | 均值 **21.195 ms**（3.05×）｜p95 **46.084 ms**｜max **50.984 ms** | TP 398 / FP 21 / FN 169，F1 **0.807302**（ΔF1 −0.0016，召回不变） |

  ⚠️ **分辨率变了，变换就变了，检测结果也会变** —— 上面这行数是本项目在龙泉山序列上实测的，
  换数据必须重测。默认值 2 精确复现已认证的交付行为（检测 TSV 的 md5 与认证默认完全相同），
  随时可以退回。
  程序化用法：`mosaic_pipeline_config_set_gmc_downscale(&cfg, 4)`。
* **GMC 的锚点步长**（环境变量 `MOSAIC_GMC_ANCHOR_STEP`，2..64 且必须是 2 的倍数，默认 **2**）：
  决定 21 帧窗口里**多少个 lag 各自独立拟合**。
  `max_lag = window(21) × stride_step(2) = 42`；锚点从 stride 起、每次 +anchor_step、到 max_lag 为止，
  再补上 max_lag 本身（与引擎的 `anchor_grid` 同构）。

  | 值 | 锚点 lag | 锚点个数 | 板卡实测（GMC 拟合，600 帧 / e2e 全量） | 精度（600 帧 / 全序列同帧集合） |
  |---|---|---|---|---|
  | **2（默认 / 冻结值）** | 2,4,…,42 | **21** | 64.565 ms（2.09× 的基准） | 基准：F1 0.808943，门禁 **PASS**（逐位） |
  | **10** | 2,12,22,32,42 | **5** | **30.821 ms（2.09×）**；e2e 端到端 fps 12.819 → **15.471（+20.7%）** | F1 0.808511（ΔF1 −0.0004）；全序列同帧集合 **ΔF1 +0.000365**，但门禁 **FAIL**（特征不再逐位相同） |

  ⚠️ **两条代价**：
  ① **最坏的一帧 246.9 → 491.6 ms（翻倍，越过板卡 250 ms 的 stall 设计点）** ——
     合成拟合会把锚点误差级联放大。**单帧最坏延时有硬预算的场景不要用 10**。
  ② **Buffer A 丢弃 1109 → 2951**：GMC 快了 ⇒ 推给 NPU 的帧更多 ⇒ NPU 被打爆（利用率 96.8 %）
     ⇒ 净交付帧只 +20.7 %，不是拟合帧的 +43 %。
  程序化用法：`mosaic_pipeline_config_set_gmc_anchor_step(&cfg, 10)`；e2e 侧用 `--anchor-step 10`。
* **NFS 上不能直接执行二进制**（板卡 NFS 是 **soft 挂载（软挂载）**）：不是「慢」的问题，
  是**会跑坏** —— >1 MB 的可执行文件从 `/mnt/manu/...` 直接跑 ⇒ **SIGILL(132)**
  （内容被读坏）。处置：先 `cat` 到板卡 `/tmp`（tmpfs）再跑——
  **共享库版本要把 `.so` 一起 `cat` 过去**（`-rpath '$ORIGIN'` 要求同目录）：

  ```bash
  ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
    && cat /mnt/manu/mosaic_sdk/out/mosaic_api_test > /tmp/mosaic_api_test \
    && chmod +x /tmp/mosaic_api_test && /tmp/mosaic_api_test ...'
  ```

  ⚠️ 本版产物**全部 < 1 MB**（共享库 ~417 KB，六个可执行文件**都远小于 1 MB**）⇒ 这条
  铁律当前不触发；上面仍 `cat` 到 `/tmp` 是稳妥起见 + `.so` 必须同目录。

  详见「必读的坑」。
* **板卡根分区 `/root` 的 ext4 已损坏**：**不要往 `/root` scp 大文件**。本包走
  NFS，不受影响。
* **必须「一推一收」**：流水线是「机会主义最新帧」调度，一次只推一帧；
  一次推多帧只会回最后一帧的结果，前面的帧永远等不到 → `collect` 会永久阻塞。

---

## 嵌入集成：怎么配置、怎么调用

需要 **4 个调用** + **1 行后处理模式配置**：`mosaic_pipeline_open` / `push` /
`collect` / `destroy`，外加 `mosaic_pipeline_set_post_mode`。
头文件是 `src/npu/mosaic_pipeline.h`（顶层另有一个指向它的兼容别名），链接的是
**共享库** `out/libmosaic_pipeline.so`（`-lmosaic_pipeline`）；
**同包内另有等价的静态库** `out/libmosaic_pipeline.a`
（两种链接方式见下文「🔗 两种链接方式」一节）。

> 🟢 **2026-11-23 起：时空跟踪已经内置在库里，而且默认开启** —— 交付参数组与
> 跨尺度候选合并半径（20 px）都由库在建流水线时**自己**设好 ⇒
> **调用侧不需要新增任何一行跟踪代码，也不需要任何新的接口**。
> 跟踪结果从 `mosaic_pipeline_collect_tracked()`（在线）/ `mosaic_pipeline_collect_fused()`
> （离线，`flush` 之后）取 —— 详见下一节。

### 🟢 跟踪已内置、默认开启（2026-11-23）：**你的调用代码一行都不用改**

**以前要做什么**：客户得自己调 `mosaic_pipeline_set_track_fusion()` 把时空融合**打开**并
配 9 个字段，另外还得把**跨尺度候选合并半径**设成 20 px。后者当时是个「要客户来调的
旋钮」，为此在公开头里**新加**了一个 setter —— 那会**改掉客户已经集成的公开接口**
⇒ 该 setter 已**从公开接口移除**，公开接口集合回到 **2026-11-20 交付版**。

**现在要做什么**：**什么都不用做**。库在 `mosaic_pipeline_create()`（`mosaic_pipeline_open()`
内部也会走它）时**自己建好**融合引擎，参数就是交付工作点 `cc2_md10`：

| 组 | 参数 | 交付值 |
|---|---|---|
| 检测阈值 | `th_base` | `0.48` |
| 检测阈值 | `th_salvage` | `0.06` |
| 检测阈值 | `th_ground` | `0.26` |
| 检测阈值 | `sky_ratio` | `1.0` |
| 跟踪器 | `confirm_consec` | `2` |
| 跟踪器 | `confirm_gap_tol` | `8` |
| 跟踪器 | `r_slope` | `5.0` |
| 跟踪器 | `match_dist` | `10.0` |
| 跟踪器 | `max_age_phys` | `-1`（=「不设」⇒ legacy 语义） |
| 跟踪器 | `min_hits` | `3` |

* 参数档：`manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json`，
  `params_md5 = a1157e013ebb496731be805c72c95dd8`。
* 🟢 **板上实机回读**（Orange Pi 5 `192.168.1.115`，2026-11-25，跑
  `out/mosaic_cadence_probe --fusion 1` 打印库内生效参数）：
  `th_base=0.4800 th_salvage=0.0600 th_ground=0.2600 sky_ratio=1.000 match_dist=10.00
  max_age_phys=-1 r_slope=5.00 confirm_consec=2 confirm_gap_tol=8`
  ⇒ **与上表逐项一致** ⇒ **集成方不需要手动传这 8~10 个参数**；
  即便要显式覆盖，`MosaicTrackFusionConfig` 的**成员默认值**本身就已经是交付值。
* **跨尺度候选合并半径 = 20 px**（可用带 **16 ~ 24 px**），**已内置**。
  ⚠️ **不要照搬服务器的 768 px** —— 板端的图像尺度不同，768 px 会让每帧只剩 1 个候选、
  **次要目标全丢**。板端 2500 帧窗口实测：半径 `0` ⇒ F1 `0.5689473` / FAR `0.3832`；
  半径 `20`（交付值）⇒ F1 `0.6930591` / FAR `0.0488`
  （服务器同口径参考：F1 `0.6918947` / FAR `0.0552`）⇒ **合并半径必须开**。

**三条不变量（重要）**

1. **`mosaic_pipeline_collect()` 的语义没变**：它返回的**仍是原始候选流**（回投影后的
   框），**逐字节不变** ⇒ 已认证的检测 TSV md5 不会动。跟踪结果**另开一条通道**。
2. **跟踪结果从这两个接口取**（它们是 2026-11-20 就有的老接口，**没有新增公开符号**）：

   | 接口 | 时机 | 拿到什么 | 返回 `-1` 的含义 |
   |---|---|---|---|
   | `mosaic_pipeline_collect_tracked(p, &fid, out, n)` | **在线**，逐帧 | 该帧**被观测到**的已确认轨迹（不含插值帧） | 跟踪被关 / 中止 / 已 flush 排空 |
   | `mosaic_pipeline_collect_fused(p, &fid, out, n)` | **离线**，**`flush()` 之后** | 终版结果（两遍平滑 + 连续帧确认） | 还没 flush，或已经取完 |

   * `collect_tracked` 与 `collect` **一一对应**：同一次「一推一收」里，`collect` 返回
     第 N 帧的检测框，紧接着的 `collect_tracked` 返回**同一个帧号 N** 的跟踪结果
     （`frame_id` 就是你自己传进来的物理帧号）。
   * ⚠️ **`collect_tracked` 的队列要排空**：不取它就会一直涨（库内上限
     `MOSAIC_TRACK_QUEUE_MAX`，默认 **4096** 帧；超了会**丢最旧的帧**并打一行 `[warn]`）。
     只用检测结果、永远不取跟踪结果的客户**不用管这条**。
   * `collect_fused` 走**物理帧号**（相机跳号的地方是空洞，跳过不返回）。
   * 🟢 **v10.7 起：回调通道里**也有**跟踪结果了**（**纯追加**：`MOSAIC_API_VERSION` 仍是
     `10`，**既有接口一个都没改**）。跟踪结果回调与原始结果回调
     （`mosaic_pipeline_set_result_callback()` / `..._ex()`）**并列**，两者可同时注册：
     * 回调签名：`typedef void (*MosaicTrackedCallback)(void* user, int64_t frame_id, const MosaicTrackedDet* dets, int32_t n_dets);`
     * 注册：`mosaic_pipeline_set_tracked_callback(p, on_tracked, my_user);`
     * **每个已确认目标一条**（**跨尺度的重复框已在库内合并**）⇒ **一目标一框**。
     * **切换语义**（与结果回调同一套）：
       - 注册（非 `NULL`）⇒ 跟踪结果**不再进** `mosaic_pipeline_collect_tracked()` 的队列
         （否则那条队列**每帧涨一条、永不排空**）；
       - 传 `NULL` 移除回调 ⇒ 跟踪结果**退回**队列（`collect_tracked()` 照旧拿得到）；
       - 切换的瞬间，**两条路径之间 pending 的帧会互搬** ⇒ **不丢帧**。
     * ✅ **只要「一目标一框」的跟踪结果 ⇒ 只注册 `set_tracked_callback` 一个就够了**
       （**两个回调不是必须同时注册**；2026-11-25 在 **Orange Pi 5（192.168.1.115）**
       与 **ematech（192.168.0.64）** 两块板**实测通过**：`--mode tracked-only`
       ⇒ tracked 回调 **60 帧 / 60 框 / 单帧最多 1 框**、**帧内 `track_id` 重复 0 帧**）。
       * ⚠️ 代价：只注册跟踪回调时**原始候选流不再缓冲**，
         `mosaic_pipeline_collect()` **返回 `-1`**（stderr **告警一次**）。
       * 只有当你**还想同时拿原始候选流**时，才需要**两个都注册** ——
         「两个都注册」在本 SDK 里是为了**同帧对照**，**不是**跟踪结果的必要条件。
     * ⚠️ **回调跑在库自己的派发线程上**（与结果回调同一个线程）：回调里**不要**调阻塞接口
       （`collect` / `collect_tracked`）、**不要** `destroy` 流水线；
       `flush()` 会**等所有已派发的回调执行完**再返回，`destroy()` 会 **join 派发线程**
       ⇒ **不存在回调访问野指针**。
3. **公开接口集合 = 2026-11-20 交付版 + v10.7 追加的 1 个**
   （`MOSAIC_API_VERSION` 仍然是 `10`）：
   `open` / `set_post_mode` / `set_result_callback` /
   **`set_tracked_callback`（v10.7 新增）** / `push`（或 `push_frame`）/
   `collect` / `collect_tracked` / `collect_fused` / `flush` / `destroy` ——
   **跟踪的打开与配置仍然不需要任何新接口**；
   v10.7 **只追加**了 `set_tracked_callback`，**既有接口一个都没改**
   （`mosaic_pipeline_collect_tracked()` 的语义与返回值逐字节不变）。

### 🟩 嵌入式接「一目标一框」：**改 1 个函数名 + 1 个结构体类型**（v10.7）

> **为什么嵌入式原来一个目标身上画了两三个框**：嵌入式拿到的通常是**原始候选框**
> （`mosaic_pipeline_set_result_callback()` 送的那一路）⇒ **同一个红外目标在多个尺度上
> 各有一个连通域候选** ⇒ **一目标多框**。**跟踪结果**（时空融合 + 跨尺度合并 +
> 连续帧确认）才是**一目标一框**。

**三步（其余调用一行都不用改）**：

```cpp
/* ① 结构体类型：const MosaicDet*  ->  const MosaicTrackedDet*（多了 track_id / infilled） */
static void on_tracked(void* user, int64_t frame_id,
                       const MosaicTrackedDet* dets, int32_t n_dets) {
    for (int32_t i = 0; i < n_dets; ++i) {
        /* 一个 track_id 一个框  =>  一个目标一个框（画框是你自己的活） */
        my_draw_box(dets[i].x, dets[i].y, dets[i].score, dets[i].track_id);
    }
}

/* ② 函数名：mosaic_pipeline_set_result_callback  ->  mosaic_pipeline_set_tracked_callback */
mosaic_pipeline_set_tracked_callback(p, on_tracked, my_user);

/* ③ push / collect / destroy 一行都不用改 */
```

* ✅ **完整可运行版**：`out/mosaic_tracked_cb_example`（源码
  `src/npu/mosaic_tracked_cb_example.cpp`）—— 它**同时注册两路回调**（原始 + 跟踪）
  做**同帧对照**，并核对「同一帧 tracked 的框数 ≤ raw 的框数」且「同一帧内 `track_id`
  不重复」（重复 = 一个目标画了多个框 = 没做到位）。
  * 🟢 **`--mode tracked-only`**：**只注册跟踪回调**（不注册原始回调），
    用来复现**嵌入式真实做法**并证明「两个回调不是必须同时注册」；
    此时 `mosaic_pipeline_collect()` 返回 **`-1`**（原始流未缓冲，**预期行为**）。
    * `--mode both`（默认）：两路都注册 ⇒ 同帧对照 raw vs tracked。
* ⚠️ **回调里不要调阻塞接口**（`collect` / `collect_tracked`）、**不要** `destroy` 流水线
  ⇒ 要做慢活（画框 / IO）就自己往队列里塞，别在回调里干。
* ℹ️ **契约测试新增 18 条用例**钉住这个语义（`mosaic_contract_test`：
  **217 passed / 0 failed**；199 旧 + **13 新**（跟踪回调：注册/退回/切换/队列关闭/tracked-only）
  + **5 新**（`flush()` 生命周期 + 派发顺序：见下））。
* 🟢 **`flush()` 生命周期已钉进契约测试**：`flush()` 返回后**再也不会有跟踪回调触发**
  ⇒ 嵌入式可以「`flush()` → `destroy()`」连写，**绝不会**出现回调访问已销毁流水线的野指针。
  （契约断言：`flush: no tracked callback fires after flush() returned` /
  `flush: the tracked callback ran before flush() returned`。）
* 🟢 **派发顺序已钉进契约测试**：跟踪回调在**库自己的派发线程**上触发，
  **按帧轴顺序**交付（`tracked callback: the frames are delivered in the frame-axis order`），
  与原始结果回调**同一个线程模型** ⇒ 嵌入式画框不会拿到乱序的帧。

**🟩 三个环境变量开关（只用于标定 / A-B，正常集成不用碰）**

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `MOSAIC_TRACK` | **开** | `0` / `off` / `false` ⇒ **启动时把跟踪关掉**（用来跟旧版本做 A/B） |
| `MOSAIC_TRACK_MERGE_RADIUS` | **20**（px） | 覆盖**跨尺度候选合并半径**（`0` = 关闭合并）；板端可用带 16~24 |
| `MOSAIC_TRACK_QUEUE_MAX` | **4096** | `collect_tracked` 队列的帧数上限（超了丢最旧的并打 `[warn]`） |

**🔴 验收证据：板端启动日志**

```
[i] candidate dedup ON: radius=20.0 px (cross-scale merge)
```

* 这一行**必须出现** ⇒ 合并半径是 20 px、跟踪是开的。
* 看到 `[warn] candidate dedup OFF (radius=0)` ⇒ **合并没开**（候选流没合并就喂给了
  跟踪器 ⇒ 同一个目标在多个尺度上会变成多条轨迹，FAR 会飙）。
* 看到 `[i] the track fusion is OFF (MOSAIC_TRACK=0)` ⇒ 跟踪被环境变量关了。

**要改参数怎么办（可选）**

只有在**想改**某个跟踪参数时才需要调这个接口（**默认不用**）：

```cpp
MosaicTrackFusionConfig tf;
mosaic_pipeline_track_fusion_defaults(&tf);   // 冻结默认值（唯一真源在 C++ 内核里）
tf.match_dist = 12.0;                          // 只改你要改的那一个
if (mosaic_pipeline_set_track_fusion(p, &tf) != 0) {
    fprintf(stderr, "设置时空融合失败（多半是 .h 与 .so 版本不匹配）\n");
}
```

* ⚠️ **版本握手（只影响「要改参数」的客户）**：
  `mosaic_pipeline_set_track_fusion()` 会**校验** `cfg.version`，
  **对不上直接返回 `-2`**，融合**静默不生效**（**不崩、不打日志、只是没开**）。
  `MOSAIC_TRACK_FUSION_VERSION` 现在是 **2**（相对 1 多了 `max_age_phys` / `r_slope` /
  `alpha_decay` / `use_physical_frame_id`）⇒ **要改参数，就必须用本包里的
  `src/npu/mosaic_pipeline.h` 重编你的 app**（**只换 `.so` 不换头** ⇒ 覆盖返回 `-2`）。
  * ✅ **完全不碰这个接口的客户不受影响**：跟踪在库内默认开着，**不需要重编**
    就能拿到交付级跟踪结果。
* 传 `NULL` ⇒ **关掉**跟踪（等价于 `MOSAIC_TRACK=0`）。
* ⚠️ **帧号语义**：`mosaic_pipeline_push_frame(p, buf, w, h, client_frame_id)` 的
  `client_frame_id` 请传**相机的物理帧号**（中间丢帧就跳号）⇒ 引擎据此算 Δn（物理帧间隔），
  关联门限/存活期/分数衰减都按真实间隔走。
  传稠密计数器也能跑（退化成 Δn≡1 的老行为，不会错），但**跳帧期间的机动会被门限吃掉**。

### 最短路径（TL;DR）

**你要写的代码就是下面这几句** —— 配置只有「模型 + kernel + 档位」三个字段：

```cpp
#include "mosaic_pipeline.h"

MosaicOpenOptions opt = {};
opt.version     = MOSAIC_API_VERSION;
opt.preset      = MOSAIC_PRESET_REALTIME;   // 最优档：3 级流水线 + D=3
opt.model_path  = model.c_str();
opt.kernel_dir  = kernel.c_str();
// opt.sequence / opt.raw_root 留空：自己喂帧的客户端没有帧目录可指，
// 留空 = 跳过环境体检（**不是**体检失败），流水线照常全套跑起来。
opt.check_env   = 1;

char err[512] = {0};
MosaicPipeline* p = mosaic_pipeline_open(&opt, err, sizeof(err));
if (!p) { fprintf(stderr, "%s\n", err); return 1; }

/* 唯一一行「最优配置」：把后处理钉死在 sparse（库默认也是它，写出来是为了
   不受 MOSAIC_POST_MODE 环境变量干扰，让「我们跑的是最优路径」在代码里有据可查） */
if (mosaic_pipeline_set_post_mode(p, MOSAIC_POST_SPARSE) != 0) {
    fprintf(stderr, "设置 post_mode 失败\n");
    mosaic_pipeline_destroy(p);      // 必须销毁：RKNN context 上限只有 2，泄漏会拖垮后续实例
    return 1;
}

mosaic_pipeline_push(p, gray, w, h);                       // 一帧灰度图推进去
MosaicDet dets[128]; int64_t fid = -1;
int32_t nd = mosaic_pipeline_collect(p, &fid, dets, 128);   // 检测框取回来

/* 跟踪结果（跟踪已内置、默认开启，参数不用你配）：
   collect_tracked = 在线逐帧，和上面那次 collect 一一对应（同一个 fid） */
MosaicTrackedDet tdets[128]; int64_t tfid = -1;
int32_t ntd = mosaic_pipeline_collect_tracked(p, &tfid, tdets, 128);

/* 终版（离线）结果：必须先 flush —— 平滑器要看完整段序列。
   逐帧取，取到返回 -1 为止（flush 之后才有） */
mosaic_pipeline_flush(p);
MosaicTrackedDet fdets[128]; int64_t ffid = -1;
for (;;) {
    int32_t nfd = mosaic_pipeline_collect_fused(p, &ffid, fdets, 128);
    if (nfd < 0) break;                 // 取完了
    /* fdets[0..nfd-1] 是这一帧的终版结果，ffid 是它的帧号 */
}

mosaic_pipeline_destroy(p);
```

> `dets[128]`：库每帧最多保留 100 个连通域（`max_detections`），缓冲区给 128
> ⇒ **不会被静默截断**。
>
> 🟢 `collect_tracked` / `collect_fused` 是**跟踪结果的唯一出口**；只想要检测框的
> 客户**可以不调它们**（`collect` 行为完全不变）—— 但调了就**必须排空**，
> 否则队列会涨（上限 `MOSAIC_TRACK_QUEUE_MAX`，默认 4096 帧，超了丢最旧的并打 `[warn]`）。

**三条必须记住的规矩**（照抄下面的示范程序就不会踩坑）：

1. **一推一收**：一次只推一帧。流水线是「机会主义最新帧」调度，一次推多帧只会回
   **最后一帧**的结果，前面的帧永远等不到 ⇒ `collect` 会**永久阻塞**。
2. **`collect` 是阻塞的**：一直等到这一帧算完才返回。
3. **帧尺寸要一致**：本 SDK 吃 640×512 的 8-bit 灰度图（`w*h` 字节）。

**帧从哪来都行**：摄像头 buffer、文件、网络 —— 你手里有一张 `uint8_t` 灰度图就能推。
包里还有一份**能直接跑的最简示范**：`src/npu/mosaic_client_example.cpp`
（编出来是 `out/mosaic_client_example`，**160 行**）—— 就是上面这段的完整版，
嵌入式组照着抄、或直接改它，最快。

### 上板运行这个最小集成示范

**先 `cat` 到板卡 `/tmp` 再执行**，**模型和共享库都要一起拷**（`-rpath '$ORIGIN'`
要求 `.so` 与可执行文件同目录；板卡 NFS 是 **soft 挂载（软挂载）**，>1 MB 的
二进制直接在 `/mnt/manu/...` 上跑 ⇒ **SIGILL(132)**；6.3 MB 的模型从 soft 挂载
直读也会被读坏 ⇒ **模型同样必须先落到 `/tmp`**，见「必读的坑」）：

```bash
# 模型先落到板卡 /tmp 并对账（md5 期望值 1a3faf2f9459cad8a389ffa81810234c）
ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn > /tmp/p0_off_A_none.rknn \
  && md5sum /tmp/p0_off_A_none.rknn \
  && cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
  && cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce \
  && chmod +x /tmp/mce \
  && /tmp/mce \
  /root/bts/frames/VIDEO00007_19700101_001324 \
  /tmp/p0_off_A_none.rknn \
  /home/orangepi/runtime/kernels \
  21'
```

ℹ️ 上面用的是**旧的四个位置参数**写法（**仍然兼容**）；新写法见下一节的
5 个开关（`--frames-dir` / `--model` / `--kernel-dir` / `--seq` / `--frames`）。
⚠️ **新板上没有 `/mnt/manu` 这个挂载点** ⇒ 帧与 kernel 必须按本板布局给
（帧 = `/root/bts/frames/...`，kernel = `/home/orangepi/runtime/kernels`）。

**零参数直接跑也行**（**零参数时程序走 `src/npu/mosaic_paths.h` 的板卡无关回落**
⇒ 本板命中 kernel = `/home/orangepi/runtime/kernels`、帧 =
`/root/bts/frames/VIDEO00007_19700101_001324`；
**模型的默认值仍然是 `/tmp/p0_off_A_none.rknn`** ⇒ **模型仍然要先 `cat` 到 `/tmp`**）：

```bash
ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn > /tmp/p0_off_A_none.rknn \
  && md5sum /tmp/p0_off_A_none.rknn \
  && cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
  && cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce \
  && chmod +x /tmp/mce && /tmp/mce'
```

输出长这样（21 帧 / 52 个目标）：

```
帧 0：2 个目标
    #0 tag=0 cx=310.6 cy=251.8 w=36 h=14 area=346 peak=0.839
    #1 tag=1 cx=305.4 cy=249.0 w=60 h=40 area=1790 peak=0.730
...
共 21 帧，52 个目标
```

### `mosaic_client_example` 的参数（**5 个开关**，旧的四个位置参数仍兼容）

**开关写法（推荐）**：

| 开关 | 含义 | 默认值（不给就取 `mosaic_paths.h` 的回落值） |
|---|---|---|
| `--frames-dir D` | 帧目录（里面是 `frame_%06d.bmp`） | 帧根目录 + 序列名（见下表） |
| `--model M` | RKNN 模型路径（fp16，输入 640×640×3，输出 2×[1,1,320,320]） | `/tmp/p0_off_A_none.rknn`（**板上落点**；模型**来源**是本包 `models/p0_off_A_none.rknn`，须先 `cat` 到 `/tmp`） |
| `--kernel-dir K` | OpenCL kernel 目录 | 见下表 |
| `--seq S` | 序列名（只有 `--frames-dir` **没给**时才用到） | `VIDEO00007_19700101_001324` |
| `--frames N` | 最多喂多少帧 | `21`（= GMC 的一个完整历史窗口） |

**旧写法仍兼容**（`[帧目录] [模型路径] [kernel 目录] [帧数]`，**按位置解析**）：

```bash
mosaic_client_example /path/frames /path/model.rknn /path/kernels 21
```

⚠️ **开关优先级高于位置参数**：同一个值两边都给了，**以 `--xxx` 为准**。

🔴 **零参数可跑**：不给任何参数时，程序走 **`src/npu/mosaic_paths.h`**
 的**板卡无关回落**（`env` → **交付包内** → **板卡部署路径** → 旧 NFS 兜底），
 **不再硬编码旧板的 NFS 路径**（早期版本写死 `/mnt/manu/...`，换到新板上
 必然 `[OCL FATAL] cannot open .../warp_median_fused.cl`）：

| 项 | 回落顺序（谁先命中用谁） | 本板（`192.168.1.115`）实际命中 |
|---|---|---|
| kernel 目录 | `MOSAIC_KERNEL_DIR` → `<包根>/runtime/kernels` → `/home/orangepi/runtime/kernels` → `/root/kernels` → `/root/bts/kernels` → `/mnt/manu/mosaic_sdk/runtime/kernels` | `/home/orangepi/runtime/kernels`（板卡部署路径） |
| 模型 | `MOSAIC_MODEL` → `/tmp/p0_off_A_none.rknn` → `<包根>/models/p0_off_A_none.rknn` | `/tmp/p0_off_A_none.rknn`（先 `cat` 上去） |
| 帧根目录 | `MOSAIC_RAW_ROOT` → `/root/bts/frames` → `/mnt/manu/lq_ir_bmp` | `/root/bts/frames`（本板测试台，11,810 帧 640×512 BMP） |

ℹ️ **包根只剥一级**：可执行文件在 `<包根>/out/` 下 ⇒ 包根 = 它的上一级 ⇒
**整包丢到板卡任意目录都能直接跑**，不依赖任何绝对路径。
⚠️ **`--frames-dir` 与 `--seq` 的关系**：帧目录 = 帧根目录 + 序列名；
 只有在 `--frames-dir` **没给**时 `--seq` 才起作用。

ℹ️ **模型（默认 `/tmp/p0_off_A_none.rknn`）已随包交付**（来源 = 包内
`models/p0_off_A_none.rknn`，板上先 `cat` 到 `/tmp/p0_off_A_none.rknn`）
⇒ **显式覆盖时也传 `/tmp/p0_off_A_none.rknn`**（不是包内路径 ——
 那条在 NFS soft 挂载上，直读会被读坏）。

**为什么默认喂 21 帧**：GMC 用 21 帧历史做运动补偿。喂不满也能出结果（早期帧的
历史会被 clamp 到最早那一帧，运动估计退化），但**喂满 21 帧才是认证质量的拟合**。
本目录共有 11,810 帧，所以默认只喂 21 帧。

**错误契约**（**不静默降级**）：

| 情况 | stderr | 退出码 |
|---|---|---|
| 未知开关（以 `-` 开头但不是上面 5 个） | `[FATAL] 未知参数 <arg>` + 用法提示 | 2 |
| 开关缺参数值 | `[FATAL] <开关> 缺少参数值` + 用法提示 | 2 |
| 位置参数超过 4 个 | `[FATAL] 多余的位置参数 <arg>` + 用法提示 | 2 |
| 帧目录里没有帧 | `[FATAL] <目录> 下没有帧` | 1 |
| 某一帧读不出来 | `[FATAL] 读不出 <路径>` | 1 |
| 建流水线失败 | `[FATAL] 建流水线失败：<err>`（err 是人类可读的原因） | 1 |

### 🔀 多线程生产-消费 + 回调：`mosaic_async_client_example`（**真实工程集成的骨架**）

> 上面那个 `mosaic_client_example` 是**单线程**原型，用来把 ABI 讲清楚、验证连通性。
> 真实业务里**没有人会把相机取帧和阻塞式推理写在同一个线程**——按单线程写法推帧，
> 必然踩中上面写明的第一条规矩（*推多帧 ⇒ `collect` 永久阻塞*）。
> 这一节给的是**工程级模板**：相机硬实时（25 FPS）与算法软实时（8~13 FPS）解耦。

**线程模型**

```text
[相机线程 Producer]              [算法线程 Worker]           [业务方]
每 40 ms 产出一帧                  取「最新」的一帧              OSD / 伺服 / 推流
feed_frame()  ────────────────>   push() → collect() ───────> callback()
```

**⭐ 库新增的 `mosaic_pipeline_push_frame()`：把「这是第几帧」交给库，让库告诉你结果属于哪一帧**

库的推理速度（8~12 fps）**远低于**相机（25 fps），而且机会主义调度器**自己决定算哪几帧**
⇒ 调用方**既不能假设结果按 push 顺序回来，也不能靠数个数** ⇒ OSD / 伺服 / 推流
必须知道「这个检测框属于我推的第几帧」，否则会画在错误的画面上。

```c
/* 推帧时带上自己的帧号 */
mosaic_pipeline_push_frame(p, gray, w, h, my_frame_id);
/* 取结果时，frame_id 原样带回你的帧号 */
int64_t fid = -1;
mosaic_pipeline_collect(p, &fid, dets, 128);   /* fid == my_frame_id */
```

* ✅ **客户端帧号原样回传**（`collect` 的返回值、结果回调的 `frame_id` 都是它）。
* ✅ **客户端帧号可以稀疏、可以非零** —— 库**不会**改写它。
* ✅ **库内部的稠密编号保持独立**：GMC 的 21 帧历史锚点和「相机流空洞检测」用的仍是它
  ⇒ **客户端跳帧不会让引擎误报 HOLE、不会拉伸时间窗、不会掉精度**。
  （这正是「内部 id」与「客户端 id」必须**解耦**的原因：合成一个，客户端跳一帧
  就会静默毁掉精度。）
* ✅ **老接口一字未改**：`mosaic_pipeline_push()` 仍在，行为、返回值、ABI 全部不变
  （它就是 `push_frame` 的「客户端没有自己的帧号」版本）。**既有代码不用改一行。**
* 契约测试新增 3 条用例钉住这个语义（`mosaic_contract_test`：**217 passed / 0 failed**）。

**三条设计规则（照抄就不会踩坑）**

1. **相机线程绝不阻塞**：`feed_frame()` 只做一次 `memcpy` + 一次 `notify_one`；
   帧数据在 `feed_frame()` 里就被**深拷贝**了一份 ⇒ 相机可以把 DMA buffer
   立刻还给驱动复用，不会有野指针，也不会有撕裂。
   ⚠️ **但应用层不丢帧**：算法跟不上时丢帧由**SDK 内部**做（丢队列里最老的帧），
   **不要**在应用层做「最新帧覆盖」—— 那会打散 GMC 的 21 帧物理历史（见下面的硬规矩）。
2. **Worker 严格「一推一收」**：`push()` 一次就 `collect()` 一次，且两个动作在
   **同一个线程**里 ⇒ 从代码结构上杜绝「推多帧 ⇒ `collect` 永久阻塞」。
3. **帧号是物理帧号**：`mosaic_pipeline_push_frame()` 把客户端帧号绑在这帧上，
   库**原样带回** ⇒ 输出里不可能出现 `-1`；库内部另有自己的稠密编号
   （给 GMC 历史做锚点、检测相机流空洞），**与客户端帧号解耦**。

**三种运行模式（`--mode`）**

| 模式 | 应用层行为 | SDK 侧丢帧 | 用途 |
|:--|:--|:--|:--|
| `serial` | **单线程** push→collect 成对执行 | 不丢（每帧都算） | **参照基线**：语义 == 老 demo `mosaic_client_example`，用来证明多线程路径不改变任何一个数字 |
| `push-all`（**默认**） | 多线程 Producer **带反压**（`--inflight N`）⇒ 帧 **100% 连续**送入 SDK | 不丢（反压让 SDK 跟得上） | **精度验收 / Golden 对标**：输出与 `serial` **逐位相同**，大盘 F1 与认证 golden **逐项全等** |
| `app-drop` | 多线程 Producer **不带反压**（相机线程一秒都不等） | 由 SDK 内部丢（丢队列里**最老**的帧） | **在线生产**：帧照进 SDK（历史保持物理连续），推理慢时由 SDK 自己抽帧 |

**📊 覆盖率：精度对齐 ≠ 覆盖率对齐（这条必须让集成方知道）**

「被处理的帧逐位相同」**不等于**「每一帧都有结果」：

| 口径 | `app-drop`（3000 帧喂入） | `push-all`（3000 帧喂入） |
|:--|:--|:--|
| 产出结果的帧 | **1,009（≈34 %）** | **2,998（≈99.9 %）** |
| 没产出结果的帧 | 1,991 帧**没有框** | 2 帧没有框 |
| 被处理帧的逐位 | ✅ 与参照**全等** | ✅ 与参照**全等** |

⇒ **官方校验器口径**（只比 arm 真正产出结果的帧）下两者 **ΔF1 = 0**；
⇒ 但**按全序列算 F1**（未产出结果的帧记漏检），丢帧模式**一定会掉** ——
校验器看不到这个差异，是因为那些帧根本不在它的子集里。

⇒ **两条口径对应两种用法**：

* **精度验收 / Golden 对标** ⇒ `--mode push-all --inflight 1`：
  反压到 1 帧 ⇒ 严格一推一收 ⇒ GMC 永不忙 ⇒ **覆盖率 100 %**
  （**已实测**：喂入 3000 帧 → 产出 **3000 帧**，零跳过；速率 3.1 fps，
  代价是相机线程被算法节奏拖住，验收模式无所谓）。
* **在线生产（延迟优先）** ⇒ `--mode app-drop`：相机线程一秒都不等，
  丢帧交给 SDK 内部；**接受那部分帧没有框**。

> 🔴 **最重要的一条规矩（务必照抄）**：
> **应用层绝不跳帧**。每一帧都必须 `push` 进 SDK，丢帧只允许发生在**推理阶段**。
>
> ❌ **反例（本模板已废止）**：在**应用层**做「最新帧覆盖」（容量 1 的 drop queue）。
> 那会让喂进 SDK 的时间轴从 40 ms 变成 120 ms ⇒ GMC 的 **21 帧物理历史被打散** ⇒
> 特征静默漂移 ⇒ F1 出现万分之八量级的波动。⚠️ 而且 **SDK 察觉不到**：
> 它的内部号是连续的发号计数，看不出物理帧号跳了 ⇒ 这种劣化**没有任何告警**。
>
> ✅ **正解**：应用层每帧都 push ⇒ SDK 内部丢帧时丢的是队列里**最老**的帧 ⇒
> **GMC 拿到的输入序列仍然物理连续** ⇒ 前处理历史窗口不破，精度不掉。
> **抽帧只能在推理阶段做，不能在前处理（历史窗口）阶段做。**

> ⚠️ `push-all` **必须**限流（`--inflight`，默认 4，与 `rknn_mosaic_e2e` 同口径）：
> 流水线中间的队列是有界的，不限流会把 GMC→NPU→POST 之间的队列灌爆，
> 结果卡在出口出不来（表现为「特征算完了但 `collect` 永远阻塞」）。
> 反压的等待谓词**含停机信号** ⇒ 关机时 Producer 立刻退出，`producer.join()` 不会挂死。

**上板实测**（板卡 `{{BOARD}}`，25 FPS 相机，序列 `VIDEO00007_19700101_001324`）

> ℹ️ **本小节下面三张表的数字是上一轮在作废板 `192.168.0.211`（Orange Pi CM5）
> 上实测的**（板卡 OpenCV 4.6 那套环境）；新板 `192.168.1.115` 上只复测了
> **零参数可跑**（120 帧进 / 120 帧出 / 326 目标，见「本板实测结果」一节）。
> 本小节保留旧数据是为了说明**多线程模板的语义**（推帧/收集/回调链路），
> 那部分结论与板卡无关。

| 模式 | 相机喂入 | SDK 收到 | 产出结果 | 库内队列溢出 | 退出码 |
|:--|--:|--:|--:|--:|:--|
| `serial` | 300 | 300 | **300** | 0 | ✅ 0 |
| `push-all` | 300 | 300 | 298 | 0 | ✅ 0 |
| `push-all` | 3,000 | 3,000 | 2,998 | 0 | ✅ 0 |
| `app-drop` | 3,000 | 3,000 | 1,009 | 0 | ✅ 0 |
| `app-drop` | 60（交付门禁第 5 道） | 60 | 9 | 0 | ✅ 0 |

⇒ **三种模式都零死锁、退出码严格 0**；相机帧率 > 算法帧率时是**丢帧而不是堆积** ⇒ 延迟不累积。

**逐位对齐（这是本模板最硬的一条验收）**

| 对比 | 交集 | 特征 md5 逐位相同 | 检测载荷逐字节相同 |
|:--|--:|--:|--:|
| `serial` vs `push-all` | 298 帧 | **298 / 298** | **298 / 298** |
| `serial` vs `app-drop` | 209 帧 | **209 / 209** | **160 / 160** |
| 新 `.so` vs 旧 `.so`（`serial`） | 300 帧 | **300 / 300** | **300 / 300** |

⇒ **「不管 work 快慢，被处理的帧都严格与参照对齐」是逐位证明的**：
`push-all`（反压）与 `app-drop`（无反压、SDK 内部抽帧）**被处理的帧**与单线程参照
**一个字节都不差**；`app-drop` 少算的那些帧**不在交集里**（被 SDK 推理阶段跳过）。
⇒ 落盘的 `feature_md5_phys.txt` / `_mosaic.tsv` / `frame_map.tsv` 里 **没有任何 `-1` 条目**。

**精度（服务器官方校验器 `load_and_align_subset.py`，只比「实际处理过的帧」）**

| 模式 | 子集 | `arm@S` TP/FP/FN | `arm@S` F1 | `ref@S` F1 | ΔF1 |
|:--|--:|:--|--:|--:|--:|
| `push-all` | 1,998 帧 | 1027 / 62 / 938 | 0.672561 | 0.672781 | **−0.000220**（TP/FN/Recall **全等**，只差 1 个 FP） |
| **`push-all`**（**三方同子集**） | **218 帧** | **119 / 5 / 96** | **0.702065** | **0.702065** | **+0.000000（TP/FP/FN/R/P/F1 逐项全等）** |
| `app-drop` | 518 帧 | 336 / 25 / 143 | 0.800000 | 0.780124 | +0.019876 |
| `app-drop`（三方同子集） | 218 帧 | 128 / 9 / 87 | 0.727273 | 0.702065 | +0.025208 |

⇒ **`push-all` 在 218 帧子集上与认证 golden 逐项完全一致** ⇒ **多线程模板的推帧/收集/回调
链路与单线程参考行为等价**（代码正确性铁证）。`app-drop` 位级几乎全不同，是因为
**跳帧 push 打散了 GMC 的 21 帧物理历史**（设计后果，不是 bug）；它在本序列上 F1 反而略高，
但**不能据此说它更好**——只能说**宏观精度没有退化**。**要与基线逐帧对齐 ⇒ 用 `push-all`。**

位级（同一子集内逐帧 md5 / 检测载荷相等）：`push-all` 1998 帧里 md5 相同 381（19.1 %）、
检测载荷相同 470，**检出组数不一致只有 3/1528 帧**；`app-drop` 因为 GMC 的物理历史
被打散，md5 几乎全不同（518 帧里 1 帧）但**数值差异极小**（质心差 ~0.1 像素、peak 差 ~0.0007）。
⇒ 位级差异来自**运行库版本差**（那轮逐位比对是在**作废板** `192.168.0.211` 上做的，
用的是板卡自带的 OpenCV **4.6**，而认证 golden 是另一套版本编出来的 ——
与 `rknn_mosaic_e2e` 同结论），**F1 宏观一致**。
ℹ️ **新板（`192.168.1.115`）上这个前提已经消失**：运行库是 **4.8**、
头随包（**4.8**，严格配套）⇒ 本包交付产物的检测 TSV 与基线**逐字节相同**
（md5 `ceea5fe670a0e996eae0dbffc8c22c11`），**不再有「换板就掉逐位」的问题**。

⚠️ **硬规矩**：`push-all` **必须**限流（`--inflight`，默认 4，与 `rknn_mosaic_e2e` 同口径）。
不限流会把 GMC→NPU→POST 之间的队列灌爆，症状是「特征算完了但 `collect()` 永远阻塞」
（零错误日志、所有线程 `futex_wait`）。

**怎么跑**

```bash
# ⚠️ 必须先落板卡 /tmp：板卡 NFS 是 soft 挂载，>1 MB 的二进制直跑会 SIGILL(132)，
#    6.3 MB 的模型直读也会被读坏 ⇒ 模型同样先 cat 到 /tmp（md5 期望值 1a3faf2f9459cad8a389ffa81810234c）
ssh -p {{BOARD_PORT}} root@{{BOARD}} 'cat /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn > /tmp/p0_off_A_none.rknn \
  && md5sum /tmp/p0_off_A_none.rknn \
  && cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
  && cat /mnt/manu/mosaic_sdk/out/mosaic_async_client_example > /tmp/mace \
  && chmod +x /tmp/mace \
  && /tmp/mace \
    --frames-dir /root/bts/frames/VIDEO00007_19700101_001324 \
    --model /tmp/p0_off_A_none.rknn \
    --kernel-dir /home/orangepi/runtime/kernels \
    --out /tmp/mosaic_async_out --frames 3000 --fps 25 --mode app-drop'
```

参数：`--frames-dir` / `--model` / `--kernel-dir` / `--out` / `--seq` / `--fps` /
`--frames` / `--mode`（`serial` | `push-all` | `app-drop`）/ `--inflight`。
输出：`--out` 目录里落 `<seq>_mosaic.tsv`（检测结果，stem 带**物理**帧号）、
`feature_md5.txt`（SDK 内部号）、`feature_md5_phys.txt`（**物理**帧号，★离线比对用这一份）、
`frame_map.tsv`（内部号 → 物理帧号）。

> 📏 **文件规模**：整个文件 370 行（代码 292 行），其中**真正要照抄的只有
> `AsyncMosaicRunner` 这一个类（约 125 行）** —— 相机线程只要调它的
> `feed_frame()`，结果从你自己的回调里拿。其余是**演示与验证**用的：帧目录
> 扫描、25 FPS 节奏模拟、TSV/指纹落盘（离线精度比对用）。

### 后处理模式（post mode）配置

四种模式，枚举定义在 `src/npu/mosaic_pipeline.h:227-230`：

| 模式 | 值 | 含义 |
|---|---|---|
| `MOSAIC_POST_DENSE` | 0 | 参考实现：全图回投 + 8 连通域 |
| `MOSAIC_POST_SPARSE` | 1 | **默认值**：CPU 稀疏路径（NEON），与 DENSE **逐比特一致**（host 906 例 / 板卡 600 帧 / 零不一致，检测 TSV md5 相同），耗时 37.856 → 0.531 ms/帧、最坏帧 40.665 → 1.037 ms |
| `MOSAIC_POST_VERIFY` | 2 | 诊断用：同帧同时跑 DENSE + SPARSE 逐字段比对，**不是生产模式**（成本约 DENSE 两倍） |
| `MOSAIC_POST_GPU` | 3 | OpenCL 局部极大值，跑 Mali，稳态 0.857 ms/帧，但**与 DENSE 不逐比特等价**（ΔF1 −0.0081，定位判定翻转 0 帧）→ 切过去必须重测精度 |

**要分两层看**：

- **库层只有两条旋钮** —— ① `mosaic_pipeline_set_post_mode()` 与
  ② 环境变量 `MOSAIC_POST_MODE`，**① 覆盖 ②**（`set_post_mode` 在
  `open`/`create` 之后调用）。
- **应用层多一条** —— `out/mosaic_api_test` **解析 `--post-mode`，再经由 ① 落地**
  ⇒ 在这个程序里，**命令行实际优先级最高**。
  （`out/mosaic_client_example` 是给嵌入方照抄的**最简**示范：**不做**这层解析，
  直接在代码里写死 ① ⇒ 只剩「库层 ① 覆盖 ②」两条旋钮，路径最短、最不容易配错。）

| 旋钮 | 生效时机 | 优先级 |
|:--|:--|:--|
| ① `mosaic_pipeline_set_post_mode(p, MOSAIC_POST_SPARSE)` | **运行时**（atomic），写进代码 | **库层最高**（覆盖 ②）：已在处理中的那一帧按它**开始时**的模式跑完，下一帧才用新模式 ⇒ 不用停机、不用重建流水线。非法值返回 `-2`（合法 0..3） |
| ② 环境变量 `MOSAIC_POST_MODE=sparse`（`dense\|0` / `sparse\|1` / `verify\|2` / `gpu\|3`） | `mosaic_pipeline_open` / `create` 时读**一次** | **库层最低**（被 ① 覆盖）；**拼错只 `[WARN]` 并保持 sparse**（设计如此：打错字绝不能悄悄改变算法） |
| ③ `--post-mode` 命令行（**应用层**：只有 `out/mosaic_api_test` 解析后转调 ①） | 程序启动时解析 → 立刻经由 ① 写入 | **应用层最高**（经由 ① 覆盖 ① 的默认值与 ②） |

> `mosaic_pipeline_open` 这套 API **本身没有** post_mode 字段（`MosaicOpenOptions`
> 布局冻结）⇒ **库层只有 ① 和 ②**，没有第三条库级旋钮。

**关键事实：默认就是 SPARSE**（`src/npu/mosaic_pipeline.cpp:400` 的
`std::atomic<int> post_mode{1}` + `:3307-3308` 的 `int mode = 1`）。TL;DR 里把
post mode 写出来是为了**防环境变量误设**，不是「必须」。
因为 sparse 与 dense **逐比特一致**，**已认证的历史精度数字不需要重跑**（TSV 与
md5 都不动），延时税直接消失（端到端延时 p50 −14.2 %）。

### 上板自检

启动时 stderr 会打一行 `[i] post mode = sparse` ⇒ 拿它确认当前跑的是哪个模式；
`mosaic_pipeline_set_post_mode()` 切换后还会再打一行 `[i] post mode -> …`。

在线核对后处理耗时用 `mosaic_pipeline_get_post_timing(p, &t)`（给的是**每帧**
后处理耗时的**分布**，不是累计均值）：

```cpp
MosaicPostTiming t = {};
if (mosaic_pipeline_get_post_timing(p, &t) == 0) {
    fprintf(stderr, "post: frames=%d mean=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f ms\n",
            t.frames, t.mean_ms, t.p50_ms, t.p95_ms, t.p99_ms, t.max_ms);
}
```

`t` 的字段：`frames` / `total_ms` / `mean_ms` / `stddev_ms` / `min_ms` / `p50_ms` /
`p90_ms` / `p95_ms` / `p99_ms` / `max_ms`。

### 编译与链接

**包里的源码 + 板卡自带的库**（`cd` 到交付目录即可，**不依赖仓库路径**）：

```bash
cd {{PKG_ROOT}} && \
TC={{TOOLCHAIN}} && \
mkdir -p out && \
"$TC/bin/aarch64-rockchip-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations -DCL_TARGET_OPENCL_VERSION=300 \
  -Iinclude -Iinclude/rga -Isrc/cpp -Isrc/opencl/host -I{{OPENCV_INCLUDE_DIR}} \
  src/npu/mosaic_client_example.cpp -o out/mosaic_client_example \
  -Lout -lmosaic_pipeline -Wl,-rpath,'$ORIGIN' \
  -L{{RT_LIB_DIR}} -L{{BOARD_LIBS}} -L{{OPENCV_LIB_DIR}} \
  -Wl,-rpath-link,{{RT_LIB_DIR}} -Wl,-rpath-link,{{BOARD_LIBS}} -Wl,-rpath-link,{{OPENCV_LIB_DIR}} \
  -Wl,--allow-shlib-undefined \
  -Wl,--no-as-needed \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core \
  {{BOARD_LIBS}}/libOpenCL.so.1 {{BOARD_LIBS}}/librga.so \
  -lrknnrt -lpthread -ldl -lm
```

> 📌 **第三方运行库不在包里**（本包**不携带任何第三方** .so/.a）：OpenCV / RKNN
> Runtime / RGA / OpenCL **全部链接板卡自带的库**（`{{OPENCV_LIB_DIR}}` /
> `{{RT_LIB_DIR}}` / `{{BOARD_LIBS}}`，板卡默认 `/usr/lib/aarch64-linux-gnu` 与
> `/usr/lib`）—— 见「板卡依赖矩阵」。
> 你要自己提供的是：**交叉编译器**（`TC=` 那一行）+ **板卡那份 OpenCV 头**
> （`{{OPENCV_INCLUDE_DIR}}`）。

* **链接对象是 `out/libmosaic_pipeline.so`**（共享库；先跑 `bash build_lib.sh` 产出），
  头文件是 `src/npu/mosaic_pipeline.h`。
  链接行是 `-Lout -lmosaic_pipeline -Wl,-rpath,'$ORIGIN'` ⇒ **整包可搬**。
* **`-I` 列表与包内 `build_demo.sh` 逐字一致**（`-Iinclude -Iinclude/rga
  -Isrc/cpp -Isrc/opencl/host -I{{OPENCV_INCLUDE_DIR}}`）——
  **不要加 `-I/usr/include`**：宿主机的头是 x86 的，会污染交叉编译。
* **自己的代码**：把源文件换成你自己的 `.cpp` 即可。头文件写
  `#include "mosaic_pipeline.h"`；**包内那几个程序是靠在 `src/npu/` 里「同目录
  优先」找到头文件的**，所以命令行里没有 `-Isrc/npu` —— **你的代码不在 `src/npu/`
  旁边就要自己加 `-Isrc/npu`**。
* ⚠️ **md5 与「包在哪个目录」有关**（行为无关）。编译时 `mosaic_core.h` 的
  `assert(...)` 与 `gmc_stream_ocl.cpp` / `ocl_host.h` 的日志宏会把 `__FILE__`
  的**路径字符串**嵌进 `.rodata` ⇒ **包根绝对路径一变，字节就变**
  （实测包根从 `/media/manu/.../build` 换到 `/tmp/xxx/pkgB`：
  `.o` 差 **224 字节**、六个可执行文件 md5 全变）。
  ⚠️ **字节差不是固定值** —— 它等于那几个嵌入路径串的**长度差**，
  换个长度的包根，差值就不同。
  ⇒ **你用自己的路径重编出来的产物，md5 不会等于本 README 顶部那张表** ——
  **这是预期的，不是包坏了**。板卡上跑起来**行为完全一样**
  （已实测：换路径编出来的 `mosaic_client_example` 在板卡上 6 帧 12 检出，
  检测框与基线**逐字段一致**）。
  ⇒ 想要**跨机器一致的 md5**，用**相对路径**编（库和 demo 都得相对，见文末
  「md5 的三个已知差异」一节）：那样编出来的产物**与目录无关**。

### 🔗 两种链接方式：共享库 `.so` vs 静态库 `.a`

本包**同时**交付两种库，**内容等价**（同源、同对象文件）⇒ **客户要哪种都有**：

| | 共享库 `out/libmosaic_pipeline.so` | 静态库 `out/libmosaic_pipeline.a` |
|---|---|---|
| 链接写法 | `-Lout -lmosaic_pipeline -Wl,-rpath,'$ORIGIN'` | `out/libmosaic_pipeline.a`（**按路径写全**） |
| 运行时还要不要我们的 `.so` | **要**（`-rpath '$ORIGIN'` ⇒ 找同目录那份） | **不要**（库代码已进可执行文件） |
| 板卡自带的第三方库 | OpenCV / RKNN Runtime / RGA / OpenCL | **同样要**（这些是第三方动态库，本包不打包） |
| 交付形态 | 换 `.so` 文件即换版本（业务 app 不用重编） | 单文件交付（app 与库合成一个二进制） |

**共享链接**（默认，绝大多数场景用这个）：

```bash
cd {{PKG_ROOT}} && \
TC={{TOOLCHAIN}} && \
mkdir -p out && \
"$TC/bin/aarch64-rockchip-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations -DCL_TARGET_OPENCL_VERSION=300 \
  -Iinclude -Iinclude/rga -Isrc/cpp -Isrc/opencl/host -I{{OPENCV_INCLUDE_DIR}} \
  src/npu/mosaic_api_test.cpp -o out/mosaic_api_test \
  -Lout -lmosaic_pipeline -Wl,-rpath,'$ORIGIN' out/stat_compat.o \
  -L{{RT_LIB_DIR}} -L{{BOARD_LIBS}} -L{{OPENCV_LIB_DIR}} \
  -Wl,-rpath-link,{{RT_LIB_DIR}} -Wl,-rpath-link,{{BOARD_LIBS}} -Wl,-rpath-link,{{OPENCV_LIB_DIR}} \
  -Wl,--allow-shlib-undefined \
  -Wl,--no-as-needed \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core \
  {{BOARD_LIBS}}/libOpenCL.so.1 {{BOARD_LIBS}}/librga.so \
  -lrknnrt -lpthread -ldl -lm
```

**静态链接**：

```bash
cd {{PKG_ROOT}} && \
TC={{TOOLCHAIN}} && \
mkdir -p out && \
"$TC/bin/aarch64-rockchip-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations -DCL_TARGET_OPENCL_VERSION=300 \
  -Iinclude -Iinclude/rga -Isrc/cpp -Isrc/opencl/host -I{{OPENCV_INCLUDE_DIR}} \
  src/npu/mosaic_api_test.cpp -o out/mosaic_api_test_static \
  out/libmosaic_pipeline.a out/stat_compat.o \
  -L{{RT_LIB_DIR}} -L{{BOARD_LIBS}} -L{{OPENCV_LIB_DIR}} \
  -Wl,-rpath-link,{{RT_LIB_DIR}} -Wl,-rpath-link,{{BOARD_LIBS}} -Wl,-rpath-link,{{OPENCV_LIB_DIR}} \
  -Wl,--allow-shlib-undefined \
  -Wl,--no-as-needed \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core \
  {{BOARD_LIBS}}/libOpenCL.so.1 {{BOARD_LIBS}}/librga.so \
  -lrknnrt -lpthread -ldl -lm
```

> 🔴 **静态链接必须按路径写 `out/libmosaic_pipeline.a`，不能再写 `-lmosaic_pipeline`**：
> 链接器在 `-L` 路径下**优先挑 `libmosaic_pipeline.so`** ⇒ 写了 `-l` 就会**静默链成共享版**
> （产物看着正常，`DT_NEEDED` 里却还有我们的 `.so`）⇒「静态链接验证」变成一句空话。
> 同理 **`out/stat_compat.o` 不要重复给**（重复给会撞重复符号）。
>
> 🔴 **`out/stat_compat.o` 两种链接方式都要带上**：
> glibc **2.29~2.32** 只导出 `__xstat(ver, path, buf)`，**不导出公开符号 `stat`**
> （公开的 `stat` 只在 `libc_nonshared.a` 里以 **hidden** 形式存在）；
> 而板卡上的 OpenCV 引用的是**公开符号 `stat`** ⇒ 不带这个垫片，**链可执行文件必炸**：
>
> ```text
> ld: .../libc_nonshared.a(stat.oS): hidden symbol `stat' in ...
>    is referenced by DSO
> ld: final link failed: Bad value
> ```
>
> （**链共享库时不报** —— `build_lib.sh` 带 `-Wl,--allow-shlib-undefined`；
> **只有链可执行文件才会撞上**。）
> `stat_compat.o` 定义公开符号 `stat` 并转调 `__xstat(1, ...)`（`_STAT_VER`）
> ⇒ **纯静态对象、无运行时依赖**（不需要 `LD_PRELOAD`，也不需要随包带任何 `.so`）。
> 板卡 glibc **≥ 2.33** 时 `stat` 本身就是真公开符号 ⇒ 那块板**不需要**这个垫片
> （带上也只是多一个未使用的定义，不影响行为）⇒ **按目标板的 glibc 自己决定带不带**。
> 包里也带了源码 `src/npu/stat_compat.c`，可以自己重编。
>
> ℹ️ **`.a` 里为什么故意不含 `stat_compat.o`**：那个垫片转发的是 glibc 的 `__xstat`
> （glibc 2.33 起已移除）⇒ 把它打进**预编译归档**，等于给这份 `.a` 锁死一个
> glibc 大版本 ⇒ 目标板 glibc ≥ 2.33 时直接链接失败 ⇒ 由集成方按自己板子决定。

### ✅ 选哪个？（共享库 vs 静态库）

* **默认用共享库 `.so`** —— 要**动态升级**（只换 `libmosaic_pipeline.so` 就能换版本，
  业务 app 不用重编）、多套程序复用同一份库、要跟着板卡实际情况调库 ⇒ 选 `.so`。
* **用静态库 `.a`** —— 要**单文件交付**（不想在运行环境里带我们的 `.so`）、
  目标板 glibc 与交叉工具链**完全一致**、或做**整包烧录固化** ⇒ 选 `.a`。
* **两者行为逐字节一致**（同源同对象文件：`mosaic_pipeline.o` + `mosaic_track_fusion.o`）
  ⇒ **选哪个都不改变检测结果**，只是部署形态不同。
  板端实测：`out/mosaic_api_test_static`（链 `.a`）与 `out/mosaic_api_test`（链 `.so`）
  跑同一段 60 帧窗口，**检测 TSV 逐字节相同（md5 相同）**；
  静态版的 `DT_NEEDED` 里**没有** `libmosaic_pipeline.so`
  ⇒ **运行时不需要我们的 `.so`**。

### 必读的坑

* **必须「一推一收」**：一次只推一帧。流水线是「机会主义最新帧」调度，一次推多帧
  只会回**最后一帧**的结果，前面的帧永远等不到 ⇒ `collect` 会**永久阻塞**。
* **`collect` 会阻塞**，直到下一帧算完 —— 它不是「取已经算好的结果」。
* **板卡 `/root` 的 ext4 已损坏**：不要往 `/root` scp 大文件（>1 MB 会被写坏）；
  中转只走 `/tmp`（tmpfs）。
* **板卡 NFS 是 soft 挂载（软挂载）⇒ >1 MB 的二进制不能直接在 `/mnt/manu/...` 上执行**：
  必须先 `cat` 到板卡 `/tmp`（tmpfs）再跑。这是**传输/缓存损坏**，不是算法 bug。
  * ⚠️ **共享库版本：`cat` 时要把 `libmosaic_pipeline.so` 一起拷过去**
    （`-rpath '$ORIGIN'` ⇒ `.so` 必须和可执行文件同目录）。
  * ℹ️ 本版产物**全部 < 1 MB**（共享库 ~417 KB，六个可执行文件**都远小于 1 MB**）
    ⇒ 这条铁律**当前不触发**；下面的 `cat` 流程仍照做（稳妥 + `.so` 要同目录）。
  * **现象**（旧版静态链接 OpenCV 时）：直接执行 8.3 MB 的 `out/mosaic_api_test` /
    `out/mosaic_client_example` ⇒ **SIGILL(132)**；板卡 `dmesg` 出现
    `RKNPU job timeout` + `soft reset` ⇒ **看着像 NPU 驱动中毒，其实不是**
    （NPU 没问题，是**喂进去的二进制被读坏了**）。
  * **判定口诀**：**「同尺寸、不同内容」= 传输/缓存损坏，不是算法 bug**。
    对比一下就分得清（实测坏掉的副本与原件**尺寸一样、md5 不同**，差 3741 字节）：

    ```bash
    ssh -p {{BOARD_PORT}} {{BOARD}} 'md5sum /mnt/manu/mosaic_sdk/out/mosaic_client_example'
    ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce && md5sum /tmp/mce'
    ```

    两行 md5 不同 ⇒ 就是软挂载把内容读坏了 ⇒ **跟模型、跟后处理、跟算法都没关系**，
    别再去查 NPU / RKNN / 后处理了。
  * **处置**：`cat` 到板卡 `/tmp`（tmpfs）+ `chmod +x` 再跑（可选再 `md5sum` 对账）：

    ```bash
    ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
      && cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce \
      && chmod +x /tmp/mce \
      && /tmp/mce'
    ```
  * ⚠️ **模型文件也受同一条铁律约束**：`.rknn` 模型（**本包自带**，来源 =
    包内 `models/p0_off_A_none.rknn`，板上 =
    `/mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn`，**6,326,305 B**）同样是
    librknnrt 从 **soft 挂载**上直读的 ⇒ 也会被读坏。症状不同：**模型加载失败**
    （不是 SIGILL）。
    稳妥做法是**模型也落 `/tmp`**（实测落盘后 md5 与源一致、结果逐位相同）：
    **板上落点固定是 `/tmp/p0_off_A_none.rknn`**（四个测试程序的模型内置默认值
    也是它 ⇒ 不传 `--model` 时同样要求模型已经在 `/tmp`）：

    ```bash
    ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn > /tmp/p0_off_A_none.rknn \
      && md5sum /tmp/p0_off_A_none.rknn \
      && cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
      && cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce \
      && chmod +x /tmp/mce \
      && /tmp/mce \
         /root/bts/frames/VIDEO00007_19700101_001324 \
         /tmp/p0_off_A_none.rknn \
         /home/orangepi/runtime/kernels \
         21'
    ```

    ⚠️ **模型已随包交付（`models/`），但「先落 `/tmp`」这一步不能省**：交付目录
    本身也在 NFS soft 挂载上 ⇒ 同一条直读约束照样适用。
    ℹ️ **`md5sum /tmp/p0_off_A_none.rknn` 的期望值** =
    `1a3faf2f9459cad8a389ffa81810234c`（与包内 `models/p0_off_A_none.rknn`
    逐位一致才算传输没坏）。

    （注意 `mosaic_client_example` 的参数是**自己的 5 个开关**
    —— `--frames-dir` / `--model` / `--kernel-dir` / `--seq` / `--frames`，
    **不是** `mosaic_api_test` 那种命名开关；
    旧的四个位置参数写法**仍然兼容**，详见「`mosaic_client_example` 的参数」一节。）

    （OpenCL kernel 是 12–25 KB 的小文本文件，**不受**这条规则约束。）
* **`MosaicOpenOptions` 的结构体布局是冻结的**（v10.1 之后只增不改，
  `MOSAIC_API_VERSION` 保持 10）：**没有 `post_mode` 字段** ⇒ 只能走 setter（①）
  或环境变量（②），不要往这个结构体里塞字段。
* **实时流场景**：`opt.check_env = 1` **且** `opt.raw_root` 非空时，库会先跑环境体检，
  **不过就直接返回 NULL**（`mosaic_pipeline.cpp` 的 `mosaic_pipeline_open` 里那段）。
  接实时流（**自己没有帧目录**）的客户端请把 `opt.raw_root` 传 **NULL**，
  或把 `opt.check_env` 置 **0**。
* **`preset` 是 `int32_t` 不是枚举类型**：C 枚举的大小是编译器定义的，**按值传
  枚举是 ABI 地雷** —— 直接传 `MOSAIC_PRESET_REALTIME` 这个常量，不要自己定义
  枚举再传。

---

## 关于产物 md5 与仓库产物不一致（**预期内**）

本包内的产物与仓库 `manu/pipeline/npu/out/` 里的同名产物 **md5 不同**。原因是
**编译时用的源文件绝对路径不同**：`mosaic_core.h` 里大量 `assert(...)` 会把
`__FILE__` 的路径字符串**嵌进 .rodata**，路径一变，字节就变。

**判定标准是「行为一致」，不是「md5 相同」。** 已用以下方式核实两者行为一致：

* 把两边二进制里的路径字符串归一化后，**字符串表集合逐条完全相同**
  （这是**旧版**（`.a` 归档那次比对）的结果：库 1150 条、
  `mosaic_api_test` 21232 条、`mosaic_contract_test` 21457 条，
  **只差 repo 的 = 0 条，只差 pkg 的 = 0 条**）；**本版改成共享库后机制不变**
  —— `__FILE__` 的路径字符串照旧嵌在 `.rodata` 里；
* `mosaic_pipeline.o` 只大 32 字节，正好是 3 个嵌入路径的长度差
  （⚠️ 这个差值**不是固定值** —— 它等于嵌入路径串的长度差，
  两边路径长度一变，差值就变；上表数字是**当时那次比对的实测值**）；
* `DT_NEEDED` 一致：`libopencv_{core,imgproc,imgcodecs,calib3d,features2d,flann,video}.so.408`
  + `libOpenCL.so.1` + `librknnrt.so` + `librga.so.2`（**全部板卡自带**）、
  无 `libc++`。

两边的编译参数与已认证的 `build_mosaic_pipeline_arm64.sh` **逐字一致**，只有路径
换成了包内路径，所以行为等价。

### md5 的三个已知差异（都不是 bug）

1. **`.so` 的 md5 是稳定的（本版已修掉旧版那个坑）**：旧版交付的是 `.a` **归档**，
   `ar` 会把成员的时间戳写进归档索引 ⇒ 那个 `.a` **每次构建 md5 都变**，
   **不能当回归基线**。
   本版交付的是**共享库 `.so`** ⇒ **没有归档索引** ⇒ **同一个包根路径**下重跑
   `bash build_lib.sh`，`out/libmosaic_pipeline.so` 的 md5 **不变**
   （`.o` 与六个可执行文件同样稳定）。
   ℹ️ **本版同时交付的静态库 `.a` 也是稳定的**：`build_lib.sh` 用的是
   `ar rcsD`（**deterministic**：归档索引里不留 uid/gid/mtime）⇒ 同一份源码
   重复构建，`out/libmosaic_pipeline.a` 的 md5 **不变** ⇒ **可以当回归基线**。
   ⚠️ 但 md5 仍然**绑定「包根的绝对路径」**（见第 3 条）。
2. **脚本构建 vs 一行命令构建，md5 也不同**：脚本用**绝对路径**编译
   （`$HERE/src/npu/mosaic_pipeline.cpp`），README 的一行命令用**相对路径**
   （`src/npu/mosaic_pipeline.cpp`）。`assert()` 嵌的是 `__FILE__`，路径长度
   不同 ⇒ `.o` 差 **272 字节**（实测；**这个差值不是固定值** —— 它等于嵌入路径串
   的长度差）。**行为一样**，只是字节不同。
3. 🔴 **换个目录重编 ⇒ 六个可执行文件的 md5 全变**（**这一条最容易让人误判**）：
   `build_lib.sh` / `build_demo.sh` 用**绝对路径**编译，路径里含**包根目录**；
   `__FILE__` 把这条路径烤进 `.rodata` ⇒ **包放在哪儿，字节就取决于哪儿**。
   * 实测：包根 `/media/manu/.../npu/build` ⇒ `mosaic_client_example`
     `76bc31df…`；同一份包放到 `/tmp/xxx/pkgB` 重编 ⇒ `a63a6995…`，
     `.o` 差 **224 字节**（同样，**差值随路径长度变**），**行为完全一样**
     （板卡上 6 帧 12 检出、检测框逐字段一致）。
   * ⇒ **你在自己机器上重编出来的 md5 不会等于本 README 顶部那张表。这是预期的，
     不是包坏了。**
   * ⇒ **想要跨机器一致的 md5**，把**库和 demo 都用相对路径**编（在包根下）：

     ```bash
     cd <交付目录>
     # 库（相对路径）
     "$TC/bin/aarch64-rockchip-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra \
       -Wno-unused-parameter -Wno-unused-function -ffp-contract=off -fno-fast-math \
       -fno-unsafe-math-optimizations -DCL_TARGET_OPENCL_VERSION=300 -fPIC \
       -Iinclude -Iinclude/rga -Isrc/cpp -Isrc/opencl/host -I{{OPENCV_INCLUDE_DIR}} \
       -c src/npu/mosaic_pipeline.cpp -o out/mosaic_pipeline.o
     "$TC/bin/aarch64-rockchip-linux-gnu-g++" -shared -Wl,-soname,libmosaic_pipeline.so \
       -Wl,--allow-shlib-undefined \
       -o out/libmosaic_pipeline.so out/mosaic_pipeline.o \
       -L{{RT_LIB_DIR}} -L{{BOARD_LIBS}} -L{{OPENCV_LIB_DIR}} \
       -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann \
       -lopencv_video -lopencv_imgproc -lopencv_core \
       -Wl,--no-as-needed \
       {{BOARD_LIBS}}/libOpenCL.so.1 {{BOARD_LIBS}}/librga.so \
       -lrknnrt -lpthread -ldl -lm
     # demo（相对路径）
     "$TC/bin/aarch64-rockchip-linux-gnu-g++" ... 同样的 -I 列表 ... \
       src/npu/mosaic_client_example.cpp -o out/mosaic_client_example \
       -Lout -lmosaic_pipeline -Wl,-rpath,'$ORIGIN' \
       -L{{RT_LIB_DIR}} -L{{BOARD_LIBS}} -L{{OPENCV_LIB_DIR}} \
       -Wl,--allow-shlib-undefined \
       -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann \
       -lopencv_video -lopencv_imgproc -lopencv_core \
       -Wl,--no-as-needed \
       {{BOARD_LIBS}}/libOpenCL.so.1 {{BOARD_LIBS}}/librga.so \
       -lrknnrt -lpthread -ldl -lm
     ```

     ⇒ 相对路径编出来的产物**与目录无关**，可以真正当跨机器基线用。
     （代价：它与本包顶部那张 md5 表**不是同一个值** —— 那张表是绝对路径编出来的。）

要拿可复现的基线，请用 `out/mosaic_pipeline.o`、`out/libmosaic_pipeline.so`
或六个可执行文件的 md5，**并确保编译时的包根路径不变**；或者按第 3 条改用相对
路径编，换取跨机器一致。

---

## 交付给嵌入式组

本包要交给**嵌入式组**集成。**目标板卡不是我们手上这块 RK3588 开发板**，所以
必须分清「板卡无关」和「板卡相关」两部分：

* **板卡无关（直接可用）**：全部源码（`src/**`）、全部头文件（`include/**`）、
  **OpenCL kernel 资产（`runtime/kernels/**`）**、两个构建脚本
  （`build_lib.sh` / `build_demo.sh`）、预编译产物（`out/**`）、
  **自带的 RKNN 模型（`models/p0_off_A_none.rknn`）**。
  这些文件**不含任何板卡私有路径**，换一个 RK3588 板子照样能编。
* **板卡相关（必须由他们自己板子提供/确认）**：
  * 模型 `.rknn` —— **已随包交付**：**来源** = 包内 `models/p0_off_A_none.rknn`
    （板上 = `/mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn`），**落点** =
    `/tmp/p0_off_A_none.rknn`（**必须先 `cat` 到板卡 `/tmp`** —— 从 NFS soft
    挂载直读会被读坏）⇒ **可以直接用我们这份**；也可以换他们自己的
    （但**换模型必须保持输入 640×640×3 fp16、输出 2×[1,1,320,320]**，
    否则后处理不匹配）。
  * **绑核参数** —— 我们预设的旋钮是**按这块板子的核拓扑排的，换板子必须重设**。
    详见 `MANIFEST.md` §D。
  * **板卡自带的第三方动态库** —— 本包**不携带任何第三方** .so/.a
    （**我们自己的 `out/libmosaic_pipeline.a` 是本包产物，不在此列**）
    （`librknnrt.so` / `librga.so.2` / `libOpenCL.so.1` / 7 个
    `libopencv_*.so.408` **全部由板卡提供**）⇒
    * **编译期**：需要 **OpenCV 4.8 的头**（**随包发布**在 `include/opencv4/`，
      板上 `/usr/include/opencv4` **只有 4.6 的头**，不能拿来配 4.8 的库）
      + 上面那些 `.so`
      （用 `OPENCV_INCLUDE_DIR` / `OPENCV_LIB_DIR` / `BOARD_LIBS` / `RT_LIB_DIR`
      指过去；交叉编译时它们指的是**板卡的文件系统**，见「板卡依赖矩阵」）。
    * **运行期**：全部由板卡 `ldconfig` 解析 ⇒ **不需要设 `LD_LIBRARY_PATH`**。
    * ⚠️ **版本必须匹配**：RKNN Runtime 的版本要与包内 `include/rknn_api.h` 对得上；
      OpenCV 的**头与运行库必须同版本**（本包是 **4.8**：头随包 `include/opencv4/`、
      运行库 `libopencv_*.so.408` 在板卡 `/usr/lib`）。
      ⚠️ 板上 `/usr/include/opencv4` 是 **4.6** 的头 ⇒ 拿它配 4.8 的库会**ABI 错配**；
      头不够时用 `packaging/fetch_opencv_headers.sh` 取一份 4.8 头。

> **完整交付清单（含每个文件的 md5 / 字节数 / 用途 / 板卡相关性）见
> [`MANIFEST.md`](MANIFEST.md)。**

> 🧩 **给嵌入式组的两个示例，选哪个？**
> * `src/npu/mosaic_client_example.cpp`（160 行）—— **单线程最小原型**，用来把
>   ABI 讲清楚、验证连通性。**照它写真实业务会踩「推多帧 ⇒ collect 永久阻塞」**。
> * `src/npu/mosaic_async_client_example.cpp` —— **多线程生产-消费 + 回调模板**，
>   相机线程（25 FPS）不等算法、Worker 严格「一推一收」、结果走业务方自己的回调。
>   **真实工程集成照这个来**，细节见上文「🔀 多线程生产-消费 + 回调」一节。

## 零参数运行

6 个 demo（`mosaic_api_test` / `mosaic_contract_test` / `mosaic_client_example` /
`mosaic_async_client_example` / `mosaic_track_dump` / `mosaic_cadence_probe`）
**全部接了 `src/npu/mosaic_paths.h`** ⇒ **默认路径不再是硬编码的旧板 NFS 路径**，
而是按 `env` → **交付包内** → **板卡部署路径** → 旧 NFS 兜底逐级回落。
本板（`192.168.1.115`）上零参数跑命中的是：

| 项 | 回落顺序（谁先命中用谁） | 本板命中 | 本板是否存在 |
|---|---|---|:--:|
| 模型 | `MOSAIC_MODEL` → `/tmp/p0_off_A_none.rknn` → `<包根>/models/p0_off_A_none.rknn` | `/tmp/p0_off_A_none.rknn` | ✅（先 `cat` 到 `/tmp` 之后） |
| kernel 目录 | `MOSAIC_KERNEL_DIR` → `<包根>/runtime/kernels` → `/home/orangepi/runtime/kernels` → `/root/kernels` → `/root/bts/kernels` → `/mnt/manu/mosaic_sdk/runtime/kernels` | `/home/orangepi/runtime/kernels` | ✅ |
| 帧目录 | `MOSAIC_RAW_ROOT` → `/root/bts/frames` → `/mnt/manu/lq_ir_bmp`（再拼序列名 `VIDEO00007_19700101_001324`） | `/root/bts/frames/VIDEO00007_19700101_001324` | ✅（11,810 帧 640×512 BMP） |

⚠️ **旧口径对照**：早期版本把这三项**硬编码**成 `/mnt/manu/lq_ir_bmp/...` 与
`/mnt/manu/mosaic_sdk/runtime/kernels` ⇒ 换到新板（**没有 `/mnt/manu` 挂载点**）
必然 `[OCL FATAL] cannot open .../warp_median_fused.cl` ⇒ **现在用
`mosaic_paths.h` 就没有这个问题**。

所以**在本板**（上面三项都就位）**不带任何参数就能直接跑**：

```bash
ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/models/p0_off_A_none.rknn > /tmp/p0_off_A_none.rknn \
  && md5sum /tmp/p0_off_A_none.rknn \
  && cat /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so > /tmp/libmosaic_pipeline.so \
  && cat /mnt/manu/mosaic_sdk/out/mosaic_api_test > /tmp/mosaic_api_test \
  && chmod +x /tmp/mosaic_api_test \
  && /tmp/mosaic_api_test --limit 12'
```

> **仍然要 `cat` 到 `/tmp` 再跑**：板卡 NFS 是 **soft 挂载（软挂载）**，>1 MB 的二进制
> 直接在 `/mnt/manu/...` 上执行 ⇒ **SIGILL(132)**（见「必读的坑」）。
> ⚠️ **共享库要一起 `cat`**（`-rpath '$ORIGIN'` 要求 `.so` 与可执行文件同目录）。
> ℹ️ 本版产物**全部 < 1 MB** ⇒ 这条铁律当前不触发，上面仍 `cat` 是稳妥起见。
> 🔴 **模型也要 `cat`**：demo 的模型内置默认值是 `/tmp/p0_off_A_none.rknn`
> ⇒ **零参数跑的前提是模型已经在板卡 `/tmp`**（上面命令的第一步就是它）。
>
> **换板子时这三个默认路径大概率不同**（模型放哪、kernel 放哪、帧从哪来，都
> 取决于他们的系统布局）⇒ 用 `--model` / `--kernel-dir` / `--frames-dir` /
> `--seq` 显式覆盖，或设 `MOSAIC_MODEL` / `MOSAIC_KERNEL_DIR` /
> `MOSAIC_RAW_ROOT` / `MOSAIC_FRAMES_DIR` 环境变量，**别照抄默认值**。
> ℹ️ **模型用本包自带的那个**：先从包内 `models/p0_off_A_none.rknn` `cat` 到
> `/tmp/p0_off_A_none.rknn`，再 `--model /tmp/p0_off_A_none.rknn`。
>
> ⚠️ **`<包根>/runtime/kernels` 那一档才是交付包自带的 kernel**
> （可执行文件在 `<包根>/out/` 下 ⇒ 包根 = 它的上一级 ⇒ **整包丢到板卡任意目录
> 都能直接跑**，不依赖任何绝对路径）；
> `/home/orangepi/runtime/kernels` 那一档是**板卡部署路径**（本板命中它）。

## 源码在哪 / 怎么自己重编

交付目录是**源码包**：源码、第三方头、OpenCL kernel、构建脚本全在里面，
**不依赖本仓库**，放到哪都能原地重编（第三方**运行库**不在包里 —— 用板卡自带的，
见「板卡依赖矩阵」）。

| 我要改什么 | 在哪 |
|---|---|
| 示例 demo（跑流水线拿检测框） | `src/npu/mosaic_api_test.cpp` |
| 最小嵌入示范（嵌入式组照着抄的那份） | `src/npu/mosaic_client_example.cpp` |
| 契约测试（验收门禁） | `src/npu/mosaic_contract_test.cpp` |
| 端到端测试程序（测吞吐/延时，fps / 延时分位就是它测的） | `src/npu/rknn_mosaic_e2e.cpp` |
| 库实现 | `src/npu/mosaic_pipeline.cpp` |
| 公开 C ABI 头（自己写程序就 include 它） | `src/npu/mosaic_pipeline.h` |
| **板卡无关的默认路径解析头**（demo / 探针的路径回落就是它） | `src/npu/mosaic_paths.h` |
| GMC / OpenCL 引擎 | `src/cpp/gmc_stream_ocl.cpp` |
| OpenCL host 封装 | `src/opencl/host/ocl_host.h` |

原地重编（**构建脚本按脚本自身所在目录定位，拷到哪个目录都能编**）：

```bash
cd <交付目录>                # 例：cd /mnt/manu/mosaic_sdk
bash build_lib.sh            # -> out/mosaic_pipeline.o, out/libmosaic_pipeline.so（共享库）,
                              #    out/libmosaic_pipeline.a（静态库：同一批 .o 用 ar 打的归档）
bash build_demo.sh           # -> out/mosaic_api_test, out/mosaic_client_example,
                              #    out/mosaic_async_client_example, out/mosaic_contract_test, out/rknn_mosaic_e2e,
                              #    out/mosaic_api_test_static（**链 .a 的静态链接版**）
bash build_board_native.sh   # **板卡原生**全量编译（板载 gcc 12.2.0）：
> ✅ **2026-11-25 在本板实测通过**（`192.168.1.115`，g++ 12.2.0）：脚本以 `BUILD_BOARD_NATIVE_OK` 收尾，
> 编出的 `.so` NEEDED = 7×`libopencv_*.so.408` + `libOpenCL.so.1` + `librga.so.2` + `librknnrt.so`
> （**全程没用 `-lOpenCL` / `-lrga`**，所以板上没有 OpenCL 开发软链也不影响），
> 该原生产物跑 60 帧的检测 TSV md5 **仍是 `ceea5fe670a0e996eae0dbffc8c22c11`**（与交叉编译产物**逐字节相同**），
> 契约 **217 passed / 0 failed** ⇒ **这块板上可以原地重编，不需要回宿主机**。
                              #    ABI 锁定 + 头版本校验 + 绝对路径链 OpenCL/RGA + -Wl,--no-as-needed
                              #    ⇒ 在新板（Orange Pi 5 / Debian 12）上直接编，不用交叉工具链
```

> ℹ️ **静态库不需要额外的构建命令**：`bash build_lib.sh` **一并**产出
> `out/libmosaic_pipeline.a`（与 `.so` 用的是**同一批 `.o`**，只是再打一次 `ar` 归档）；
> `bash build_demo.sh` 会**同时**产出**共享链接版**（`out/mosaic_api_test`）与
> **静态链接版**（`out/mosaic_api_test_static`）两个 demo。
> ⇒ 想要哪种库，客户**自己选**即可，**不用改任何构建脚本**。

> **交叉编译器**（编译器没法随包发）要用这个指定：
>
> ```bash
> export RK3588_TOOLCHAIN=/path/to/toolchain-root   # 可以用 ':' 分隔多个候选，按顺序取第一个能用的
> # 或者更直接：
> export CROSS_CXX=/path/to/aarch64-linux-gnu-g++   # 可选：直接指定编译器
> export CROSS_AR=/path/to/aarch64-linux-gnu-ar     # 可选：直接指定 ar
> ```
>
> **第三方库不在包里**（本包**不携带任何第三方** .so/.a），要用板卡自带的：
>
> | 依赖 | 板卡位置 | 什么时候需要 | 定位它的环境变量 |
> |---|---|---|---|
> | OpenCV **头**（**4.8**，**随包发布**在 `include/opencv4/`） | 包内 `include/opencv4`（板上 `/usr/include/opencv4` **只有 4.6 的头**，不能配 4.8 的库） | 编译期 | `OPENCV_INCLUDE_DIR` |
> | OpenCV **动态库**（7 个 `.so.408`） | `/usr/lib` | 链接期（运行期由 ldconfig 解析） | `OPENCV_LIB_DIR` |
> | 板卡 `libOpenCL.so.1` / `librga.so.2` | `/usr/lib/aarch64-linux-gnu` | 链接期 + 运行期 | `BOARD_LIBS` |
> | 板卡 `librknnrt.so` | `/usr/lib` | 链接期 + 运行期 | `RT_LIB_DIR` |
>
> ⚠️ **交叉编译在宿主机上做时，上面这些路径指的是「目标板卡的文件系统」**：
> 宿主机的 `/usr` 是 x86 的，**绝对不能直接用** ⇒ 先把板卡的那份取回宿主机，
> 再把变量指过去（见「板卡依赖矩阵」）。
>
> 第三方**头**（`include/`：rknn_api.h / stb_image*.h / rga/ / CL/）已随包，
> OpenCL kernel 资产在 `runtime/kernels/`（运行时 `--kernel-dir` 指这里）。
>
> ⚠️ **原地重编后，本 README 顶部的「产物 md5」表会过期**（那是打包时写进去的，
> 交付态的包里没有生成它的脚本）。自检办法：`md5sum out/*` 与表里比对。
> 本版交付的是**共享库 `.so`** ⇒ **没有 `ar` 归档时间戳那种「每次编译都变」的
> 问题**（同一个包根路径下 md5 稳定）；**同批 `.o` 打的静态库 `.a` 用
> `ar rcsD`（deterministic）⇒ md5 同样稳定**；但 md5 仍与「包根的绝对路径」绑定
> （见文末「md5 的三个已知差异」第 3 条）。
>
> ⚠️ **换了编译器 / 换了 OpenCV 版本 ⇒ 产物 md5 会变**（不是 bug）。
> 上面那张 md5 表对应的是作者机器上那套工具链（gcc 9.3.0 buildroot +
> **OpenCV 4.8 的头与库**：头 = 随包发布的 `include/opencv4/`、库 = 板卡
> `/usr/lib` 的 `libopencv_*.so.408`）。要复现它，用同一套工具链 + 同一套 4.8 头与库。
> ⚠️ **`OPENCV_ABI` 是这套 ABI 的唯一真相来源**（默认 **408**）：
> `build_lib.sh` / `build_demo.sh` / `build_board_native.sh` 三者**同名同义**，
> 会校验「头版本 ↔ `OPENCV_ABI`」是否配套，不配套就**当场 die** 并给出指引。

---

## NFS 交付目录内容

`{{NFS_ROOT}}/mosaic_sdk/`（板卡侧 = `/mnt/manu/mosaic_sdk/`）是**整棵源码包**：

```
{{NFS_ROOT}}/mosaic_sdk/
├── README.md                 ← 本文件
├── MANIFEST.md               ← 交付清单
├── build_lib.sh              ← 编译库（+x）
├── build_demo.sh             ← 编译 demo / 契约测试 / 端到端测试程序（+x）
├── build_probes.sh           ← 编译两个板卡验证探针（+x）
├── build_board_native.sh     ← **板卡原生**全量编译（板载 gcc + ABI 校验 + 绝对路径链 OpenCL/RGA）（+x）
├── copy_to_nfs.sh            ← 一键整树拷贝（+x）
├── fetch_board_libs.sh       ← 把板卡自带的库/头取回宿主机（交叉编译用）（+x）
├── fetch_opencv_headers.sh   ← 取 **OpenCV 4.8 头**（板上只有 4.6 的头 ⇒ 必须外部提供）（+x）
├── known_hosts               ← 板卡主机密钥基线（ssh 严格校验用；换密钥直接失败）
├── src/                      ← 全部源码（可参考、可改、可原地重编）
│   ├── npu/    mosaic_pipeline.{cpp,h}, mosaic_core.h, mosaic_post_gpu.h,
│   │           mosaic_api_test.cpp, mosaic_client_example.cpp,
│   │           mosaic_contract_test.cpp, rknn_mosaic_e2e.cpp,
│   │           mosaic_paths.h（**板卡无关的默认路径解析头**）
│   ├── cpp/    gmc_stream_ocl.cpp, md5.h
│   └── opencl/host/   ocl_host.h, post_local_max.cl
├── include/                  ← 第三方头：rknn_api.h, stb_image*.h, rga/×18, CL/×20,
│                              opencv4/（**OpenCV 4.8 头，211 个文件，随包发布**）
├── runtime/kernels/          ← OpenCL kernel 资产
│   ├── warp_median_fused.cl     ← 运行时必需
│   ├── sortnet_generated.inc    ← 运行时必需
│   └── gmc_feature_gpu.cl       ← 可选（探针用，库不加载）
├── models/                  ← **本包自带的 RKNN 模型（来源）**：p0_off_A_none.rknn
│                              （板上先 cat 到 /tmp/p0_off_A_none.rknn 再用）
├── out/                      ← 编译产物
│   ├── mosaic_pipeline.o          ← 库的主目标文件
│   ├── mosaic_track_fusion.o      ← 时空融合内核目标文件
│   ├── stat_compat.o              ← glibc<2.33 `stat` hidden symbol 垫片（链可执行文件时带上）
│   ├── libmosaic_pipeline.so      ← **共享库**（SONAME=libmosaic_pipeline.so）
│   ├── libmosaic_pipeline.a       ← **静态库**（同源同对象文件；客户要静态链接时用它）
│   ├── mosaic_api_test            ← 示例 demo 可执行文件
│   ├── mosaic_api_test_static     ← **静态链接版**示例 demo（链 `.a` 编出来）
│   ├── mosaic_client_example      ← 最小嵌入示范可执行文件
│   ├── mosaic_async_client_example← 多线程生产-消费模板
│   ├── mosaic_contract_test       ← 契约测试可执行文件
│   ├── rknn_mosaic_e2e            ← 端到端测试程序（吞吐/延时测量）
│   ├── mosaic_cadence_probe       ← 板卡验证探针（本地工具）
│   └── mosaic_track_dump          ← 板卡验证探针（本地工具）
└── 兼容别名（相对符号链接）
    mosaic_api_test       -> out/mosaic_api_test
    mosaic_contract_test  -> out/mosaic_contract_test
    libmosaic_pipeline.so -> out/libmosaic_pipeline.so
    mosaic_pipeline.h     -> src/npu/mosaic_pipeline.h
    kernels               -> runtime/kernels
```

> ℹ️ **没有 `third_party/`**：本包**不携带任何第三方** .so/.a，
> OpenCV / RKNN Runtime / RGA / OpenCL 全部**动态链接板卡自带的库**
> （见「板卡依赖矩阵」）。
> ℹ️ **但第三方「头」随包**：`include/` 下有 rknn_api.h / stb_image*.h /
> rga/×18 / CL/×20 **+ `opencv4/`（4.8 头 211 个）** ——
> 板上 `/usr/include/opencv4` **只有 4.6 的头**，配 4.8 的库会 ABI 错配 ⇒
> **4.8 的头必须外部提供**，本包直接把它带上了（离线可构建）。

拷贝命令（**由 Leader / 使用者执行**）：整树镜像

```bash
SRC={{PKG_ROOT}}
DST={{NFS_ROOT}}/mosaic_sdk
mkdir -p "$DST"

# ① 先删掉【上一版遗留的平铺实文件/实目录】—— cp -a 是「合并」语义，
#    不会删；不删就会留下**旧二进制**，让人误以为在跑新版
#    （**更早**的版本是 .a 归档 + 第三方依赖目录；本版是 .so + .a ⇒ 旧目录都要清，
#     别把两版的产物混在同一棵树上）
rm -f  "$DST/mosaic_api_test" "$DST/mosaic_contract_test" \
       "$DST"/libmosaic_pipeline.* "$DST/mosaic_pipeline.h"
rm -rf "$DST/kernels" "$DST/third_party"

# ② 整树镜像
cp -a "$SRC/src" "$SRC/include" "$SRC/runtime" "$SRC/out" "$DST/"
cp -a "$SRC/README.md" "$SRC/MANIFEST.md" \
      "$SRC/build_lib.sh" "$SRC/build_demo.sh" "$SRC/build_probes.sh" \
      "$SRC/build_board_native.sh" "$SRC/copy_to_nfs.sh" \
      "$SRC/fetch_board_libs.sh" "$SRC/fetch_opencv_headers.sh" \
      "$SRC/known_hosts" "$DST/"
chmod +x "$DST/build_lib.sh" "$DST/build_demo.sh" "$DST/build_probes.sh" \
         "$DST/build_board_native.sh" "$DST/copy_to_nfs.sh" \
         "$DST/fetch_board_libs.sh" "$DST/fetch_opencv_headers.sh" \
         "$DST/out/mosaic_api_test" "$DST/out/mosaic_client_example" \
         "$DST/out/mosaic_async_client_example" \
         "$DST/out/mosaic_contract_test" "$DST/out/rknn_mosaic_e2e"

# ③ 建 6 个兼容符号链接（**相对路径**），旧命令才继续可用
( cd "$DST" && \
  ln -sfn out/mosaic_api_test      mosaic_api_test && \
  ln -sfn out/mosaic_async_client_example mosaic_async_client_example && \
  ln -sfn out/mosaic_contract_test mosaic_contract_test && \
  ln -sfn out/libmosaic_pipeline.so libmosaic_pipeline.so && \
  ln -sfn src/npu/mosaic_pipeline.h mosaic_pipeline.h && \
  ln -sfn runtime/kernels           kernels )
```

> ⚠️ **别省 ① 和 ③**：少了 ① 会留下**上一版的旧二进制**（跑的是旧程序却以为是新版）；
> 少了 ③ 则「旧写法仍然可用」这句话不成立。
> **更省心的做法是用 `bash {{PKG_ROOT}}/copy_to_nfs.sh`**（它三步全做，还带递归 md5 回验）。

> **板卡上的 `--kernel-dir` 指向 `runtime/kernels`**：
> `--kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels`
> （旧写法 `/mnt/manu/mosaic_sdk/kernels` 是**兼容别名**，仍然可用）。
> ℹ️ **本板（`192.168.1.115`）没有 `/mnt/manu` 挂载点** ⇒ 本板用
> `--kernel-dir /home/orangepi/runtime/kernels`（或干脆不给，让
> `mosaic_paths.h` 自动回落）。

---

## 一键拷贝

拷贝这一步也有一个脚本，**编译 + 拷贝 + 上板验证一条命令**：

```bash
cd {{PKG_ROOT}}
bash copy_to_nfs.sh --build --verify      # 编译 + 拷贝 + 上板验证，一条命令
```

`copy_to_nfs.sh` 会做这些事：**整树镜像**到 NFS（`src/` `include/` `runtime/`
`models/` `out/` + 顶层 10 个文件，不是零散拷几个产物）→ **递归 md5 回验**
（NFS 写坏了立刻报错）→ 在交付根目录建好指向 `out/` 与 `runtime/kernels` 的
**相对符号链接**（兼容别名）→ 打印目标清单、板卡侧路径和运行命令。

| 变体 | 命令 | 说明 |
|:--|:--|:--|
| 只拷贝 | `bash copy_to_nfs.sh` | 用已有的 `out/` 产物，跳过编译 |
| 编译 + 拷贝 | `bash copy_to_nfs.sh --build` | 拷贝前先跑 `build_lib.sh` + `build_demo.sh` |
| 换 NFS 目录 | `bash copy_to_nfs.sh --nfs /tmp/test_nfs` | 覆盖 NFS 根目录（**不碰真 NFS**），用于试跑 |
| 换板卡 | `bash copy_to_nfs.sh --board root@1.2.3.4:2222` | 覆盖 ssh 目标与端口 |
| 换板卡落点 | `bash copy_to_nfs.sh --board-dest /mnt/manu/mosaic_sdk` | 板卡侧交付目录（NFS 根不可写、需经板卡中转时用它） |
| 上板验证 | `bash copy_to_nfs.sh --verify` | 拷贝后 ssh 到板卡**串行**跑：契约测试（`--limit 60`）→ demo（`--limit 6`）→ 最小集成示范（`--limit 6`），**任一非 0 退出即失败** |

> ⚠️ **NFS 根只读时**：本机 NFS 根不可写 ⇒ 脚本把整树**先经板卡 `/tmp`（tmpfs）
> 中转**，再落到板卡上的交付目录（`--board-dest` 指定）。
>
> ⚠️ **绝不能往板卡 `/root` 写大文件**：板卡根分区 `/root` 的 ext4 已损坏，
> >1 MB 的二进制会被写坏。中转只走 `/tmp`（tmpfs），用完自动清理；
> `--verify` 只**执行**已有的程序，**不往板卡 `/root` 写任何东西**。
>
> ℹ️ `--verify` 的三个测试**任一失败会自动重试一次**（等 5 s）并打印 `[retry]`：
> 板卡上的 NFS 是 **soft 挂载**，刚写完 25 MB 产物就立刻读帧可能撞上瞬时 I/O 错误
> （帧文件明明在，OpenCV 却报 `can't open/read file`）——**重试仍失败才是真回归**。
>
> ℹ️ **主机密钥是钉死的**：交付包带一份 `known_hosts`（板卡的 ed25519/rsa/ecdsa 公钥），
> 脚本用 `StrictHostKeyChecking=yes` 校验 ⇒ 板卡换密钥会**直接失败**（这是故意的）。

---

## 这个 build/ 是怎么来的（打包流程）

`build/` 是**生成物**，不是手写的。任何时候想刷新它，跑这个脚本就行：

```bash
# 从工程里把交付用的全部文件整理进 build/（可反复重跑）
bash {{SRC_ROOT}}/packaging/make_build_dir.sh
```

它会做四件事：

1. 把工程里的**全部交付文件**拷进 `build/`：`src/npu/`（库 + demo + 契约测试）、
   `src/cpp/`、`src/opencl/host/`、`include/`（第三方头）、`runtime/kernels/`
   （OpenCL kernel 资产）、`models/`（**自带的 RKNN 模型**）、三个脚本 + 本 README。
   （**不拷任何第三方 .so/.a** —— 那些改用板卡自带的库。）
2. **删掉陈旧文件**：`build/src`、`build/include`、`build/runtime` 下不再需要的
   旧文件会被清掉（逐个打印 `[stale] 已删除 …`）；
   **旧包残留的第三方依赖目录会被整棵删掉**（打印 `[stale] 已删除整个目录 …`）。
3. 按 `packaging/MANIFEST.template.md` 生成 `build/MANIFEST.md`（md5 表自动生成）。
4. **回验**：表内每个 md5 与磁盘对一遍、表外不许有漏登记的文件。

**规则**：

* **`build/` 里不要手改**——下一次重跑会把你改的东西冲掉。
* 要改源码就改**工程里的原件**（`manu/pipeline/npu/mosaic_pipeline.cpp` 等），
  然后重跑本脚本刷新 `build/`。
* 要改脚本/文档就改 **`packaging/` 里的原件**（`build_lib.sh` / `build_demo.sh` /
  `copy_to_nfs.sh` / `README.md` / `MANIFEST.template.md`），再重跑本脚本。
* 本脚本**不编译、不拷到 NFS、不 ssh**：编译用 `build/build_lib.sh` +
  `build/build_demo.sh`，拷贝用 `build/copy_to_nfs.sh`。

> 第三方头（`include/`）来自 `manu/pipeline/npu/.build-include/`。若该目录不存在，
> 先跑一次 `MOSAIC_NO_DEPLOY=1 bash build_mosaic_pipeline_arm64.sh` 让它自动 stage。

---

## 唯一源在哪（给维护者）

* **`manu/pipeline/npu/packaging/` 是唯一源，且被 git 跟踪**：
  `make_build_dir.sh`、`README.template.md`、`MANIFEST.template.md`、
  `build_lib.sh`、`build_demo.sh`、`copy_to_nfs.sh`。
* **`manu/pipeline/npu/build/` 是生成物**，被仓库 `.gitignore` 的 `build/` 规则
  忽略（不会进 git），由一条命令重建：

  ```bash
  bash {{SRC_ROOT}}/packaging/make_build_dir.sh
  ```

* 要改**交付内容**，改 `packaging/` 里的原件，然后重跑上面那条命令刷新
  `build/`；**不要**直接改 `build/` 里的东西——下一次重跑会冲掉。
