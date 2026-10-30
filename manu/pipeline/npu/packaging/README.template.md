> ⚠️ **本文件是模板，不是交付物本身。**
> 交付物 `build/README.md` 由
> `{{SRC_ROOT}}/packaging/make_build_dir.sh` 从本文件生成，**请勿手改生成物**。
> 本文件里的占位符（双花括号形式）由脚本按实况填充；模板若漏改一个，
> 脚本会检查到残留的双花括号并直接报错退出，不会产出一份带占位符的文档。

# mosaic_pipeline SDK —— 独立构建包（RK3588 / aarch64）

本目录是一个**自包含的交付包**：库源码、示例 demo、契约测试、第三方头文件全部
拷在这里面，可以**脱离仓库单独编译**。

```
build/
├── README.md          ← 你正在看的这份（四条命令）  ★由 packaging/README.template.md 生成
├── MANIFEST.md        ← 交付清单（md5 表自动生成）★由 packaging/MANIFEST.template.md 生成
├── build_lib.sh       ← 只编译库
├── build_demo.sh      ← 只编译测试程序
├── copy_to_nfs.sh     ← 一键拷贝到 NFS（可选带板卡验证）
├── src/
│   ├── npu/           mosaic_pipeline.{cpp,h}, mosaic_core.h,
│   │                  mosaic_api_test.cpp, mosaic_contract_test.cpp
│   ├── cpp/           gmc_stream_ocl.cpp, md5.h
│   └── opencl/host/   ocl_host.h
├── include/           rknn_api.h, stb_image*.h, rga/ + CL/
├── runtime/kernels/   warp_median_fused.cl, sortnet_generated.inc（运行时必需）,
│                      gmc_feature_gpu.cl（可选，库不加载）
└── out/               ← 产物落在这里
```

交付包共 **{{FILE_COUNT}}** 个文件：源码 **{{SRC_COUNT}}** 个（`src/`）、
第三方头 **{{HDR_COUNT}}** 个（`include/`）、kernel 资产 **{{KERNEL_COUNT}}** 个
（`runtime/`），另加 `out/` 的编译产物。

产物（全部 aarch64 ELF）：

| 文件 | 是什么 |
|---|---|
| `out/libmosaic_pipeline.a` | **库**：GMC/OpenCL + RGA mosaic + RKNN fp16 + 反投影，公开 C ABI 在 `src/npu/mosaic_pipeline.h` |
| `out/mosaic_api_test` | **给客户的示例 demo**：只调公开 C ABI（open / push / collect / destroy） |
| `out/mosaic_contract_test` | **库的契约测试**（验收门禁） |

当前产物的 md5（**本次构建的快照**；`libmosaic_pipeline.a` 每次构建都会变，
见文末「md5 的两个已知假差异」）：

| 文件 | md5 |
|---|---|
| `out/mosaic_pipeline.o` | `{{OBJ_MD5}}` |
| `out/libmosaic_pipeline.a` | `{{LIB_A_MD5}}` |
| `out/mosaic_api_test` | `{{API_TEST_MD5}}` |
| `out/mosaic_contract_test` | `{{CONTRACT_TEST_MD5}}` |

---

## ① 单独编译库

```bash
cd {{PKG_ROOT}}
bash build_lib.sh
```

产物：`out/mosaic_pipeline.o` + `out/libmosaic_pipeline.a`。

**等价的一行命令**（不依赖脚本，可直接粘贴）：

```bash
cd {{PKG_ROOT}} && \
TC={{TOOLCHAIN}} && \
INSTALL={{OPENCV_INSTALL}} && \
mkdir -p out && \
"$TC/bin/aarch64-rockchip-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations -DCL_TARGET_OPENCL_VERSION=300 \
  -Iinclude -Iinclude/rga -Isrc/cpp -Isrc/opencl/host -I"$INSTALL/include/opencv4" \
  -c src/npu/mosaic_pipeline.cpp -o out/mosaic_pipeline.o && \
rm -f out/libmosaic_pipeline.a && \
"$TC/bin/aarch64-rockchip-linux-gnu-ar" rcs out/libmosaic_pipeline.a out/mosaic_pipeline.o
```

---

## ② 单独编译测试程序

依赖 ① 产出的 `out/libmosaic_pipeline.a`（链接的就是它）。

```bash
cd {{PKG_ROOT}}
bash build_demo.sh
```

产物：`out/mosaic_api_test`、`out/mosaic_contract_test`。

**等价的一行命令**（示例 demo；把源文件换成 `src/npu/mosaic_contract_test.cpp`、
产物换成 `out/mosaic_contract_test` 即得契约测试）：

```bash
cd {{PKG_ROOT}} && \
TC={{TOOLCHAIN}} && \
INSTALL={{OPENCV_INSTALL}} && \
LIBDIR={{BOARD_LIBS}} && \
RT_LIB_DIR={{RT_LIB_DIR}} && \
mkdir -p out && \
"$TC/bin/aarch64-rockchip-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
  -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations -DCL_TARGET_OPENCL_VERSION=300 \
  -Iinclude -Iinclude/rga -Isrc/cpp -Isrc/opencl/host -I"$INSTALL/include/opencv4" \
  src/npu/mosaic_api_test.cpp out/libmosaic_pipeline.a -o out/mosaic_api_test \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$INSTALL/lib/opencv4/3rdparty" \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core -ltegra_hal -lzlib -lOpenCL -lpthread -ldl -lm
```

---

## ③ 把编译产物拷到 NFS 挂载点

```bash
mkdir -p {{NFS_ROOT}}/mosaic_sdk
cp -f {{PKG_ROOT}}/out/mosaic_api_test \
      {{PKG_ROOT}}/out/mosaic_contract_test \
      {{PKG_ROOT}}/out/libmosaic_pipeline.a \
      {{PKG_ROOT}}/src/npu/mosaic_pipeline.h \
      {{NFS_ROOT}}/mosaic_sdk/
```

> `mosaic_pipeline.h` 一起拷过去，是为了让客户能自己写代码链接这个库。

**挂载点对应关系**：NFS 在本机是 `{{NFS_ROOT}}`，在板卡上是
`/mnt/manu` ⇒ 拷过去之后，板卡从 `/mnt/manu/mosaic_sdk/` 就能看到这些文件。

---

## ④ 运行命令（板卡 `{{BOARD}}:{{BOARD_PORT}}`）

```bash
ssh -p {{BOARD_PORT}} {{BOARD}} '/mnt/manu/mosaic_sdk/mosaic_api_test \
  --raw-root   /mnt/manu/lq_ir_bmp \
  --sequence   VIDEO00007_19700101_001324 \
  --model      /root/psearch/models/p0_off_A_none.rknn \
  --kernel-dir /root/psearch/kernels \
  --limit 12'
```

---

## 注意事项

* **契约测试**：`/mnt/manu/mosaic_sdk/mosaic_contract_test`（同样参数）。
  板卡实测 **196 passed / 0 failed**。它是库的验收门禁——客户环境跑通这个，
  说明库的行为是符合契约的。
* **demo 的全部选项与默认值**（`src/npu/mosaic_api_test.cpp`）：

  | 选项 | 默认值 | 说明 |
  |---|---|---|
  | `--raw-root` | `/mnt/manu/lq_ir_bmp` | 帧根目录，只用于环境体检和拼帧目录 |
  | `--sequence` | `VIDEO00007_19700101_001324` | 序列名，帧目录是 `<raw-root>/<sequence>/frame_%06d.bmp` |
  | `--model` | `/mnt/manu/ptq/models/p0_off_A_none.rknn` | RKNN 模型路径 |
  | `--kernel-dir` | `/root/kernels` | OpenCL kernel 目录 |
  | `--limit` | `60` | 最多跑多少帧（`0` = 不限） |
  | `--no-env-gate` | 关（默认开体检） | 跳过开库前的环境体检 |
  | `--post-mode` | `sparse` | 后处理算法：`sparse`（默认）/ `dense` / `verify` / `gpu` |
  | `--dump-tsv` | 不导出 | 把逐帧检测导出成 e2e 同格式的 TSV |

  不带参数也能直接跑（默认值就是板卡上那一套）。
* **后处理有四种算法（v10.3）**，由 `--post-mode` 或环境变量 `MOSAIC_POST_MODE` 选：

  | 模式 | 板卡实测（RK3588，600 帧） | 全序列门禁 VERDICT | 说明 |
  |---|---|---|---|
  | **`sparse`（默认）** | **0.531 ms/帧**，σ 0.219，最坏帧 1.037 | **PASS** | CPU NEON 稀疏路径：只求值「可能过阈值」的像素。**输出与 `dense` 逐比特一致**（全序列上 6034 个共同帧的检测框逐条相同、feature md5 7113/7113 相同），已认证的历史数字不需要重跑 |
  | `dense` | 37.856 ms/帧，σ 0.728，最坏帧 40.665 | PASS | 参考实现：全图双线性上采样 + 全图 8 连通域。留作对照与退路 |
  | `verify` | ≈ dense 的两倍 | — | 诊断用：同一帧同时跑 dense 与 sparse 并逐字段比对 |
  | `gpu` | 稳态 0.857 ms/帧（σ 0.039） | **FAIL** | OpenCL 局部极大路径（跑在 Mali 上）。**不是 dense 的即插即用替代**（检测流 5666 条不同、假警率高 50 %），且有一次性 OpenCL 初始化尖刺（首帧约 54 ms） |

  换成参考实现：`MOSAIC_POST_MODE=dense` 或 `--post-mode dense`。
* **NFS 上直接执行慢**：先拷到本地再跑——
  `cp /mnt/manu/mosaic_sdk/mosaic_api_test /tmp/ && /tmp/mosaic_api_test ...`。
* **板卡根分区 `/root` 的 ext4 已损坏**：**不要往 `/root` scp 大文件**。本包走
  NFS，不受影响。
* **必须「一推一收」**：流水线是「机会主义最新帧」调度，一次只推一帧；
  一次推多帧只会回最后一帧的结果，前面的帧永远等不到 → `collect` 会永久阻塞。

---

## 客户自己写程序怎么用

只需要 **4 个调用**：`mosaic_pipeline_open` / `push` / `collect` / `destroy`。

链接 `libmosaic_pipeline.a`，头文件用 `mosaic_pipeline.h`：

```cpp
#include "mosaic_pipeline.h"

MosaicOpenOptions opt = {};
opt.version = MOSAIC_API_VERSION;
opt.preset  = MOSAIC_PRESET_REALTIME;
opt.model_path = model.c_str();
opt.kernel_dir = kernel.c_str();
opt.sequence   = seq.c_str();
opt.raw_root   = root.c_str();
opt.check_env  = 1;

char err[512] = {0};
MosaicPipeline* p = mosaic_pipeline_open(&opt, err, sizeof(err));
if (!p) { fprintf(stderr, "%s\n", err); return 1; }

mosaic_pipeline_push(p, gray, w, h);              // 一帧进去
MosaicDet dets[64]; int64_t fid = -1;
int32_t nd = mosaic_pipeline_collect(p, &fid, dets, 64);   // 检测框出来
mosaic_pipeline_destroy(p);
```

`mosaic_pipeline.h` 在 `src/npu/`，已经随 ③ 一起拷到 NFS 挂载点了。

---

## 关于产物 md5 与仓库产物不一致（**预期内**）

本包内的产物与仓库 `manu/pipeline/npu/out/` 里的同名产物 **md5 不同**。原因是
**编译时用的源文件绝对路径不同**：`mosaic_core.h` 里大量 `assert(...)` 会把
`__FILE__` 的路径字符串**嵌进 .rodata**，路径一变，字节就变。

**判定标准是「行为一致」，不是「md5 相同」。** 已用以下方式核实两者行为一致：

* 把两边二进制里的路径字符串归一化后，**字符串表集合逐条完全相同**
  （`libmosaic_pipeline.a` 1150 条、`mosaic_api_test` 21232 条、
  `mosaic_contract_test` 21457 条，**只差 repo 的 = 0 条，只差 pkg 的 = 0 条**）；
* `mosaic_pipeline.o` 只大 32 字节，正好是 3 个嵌入路径的长度差；
* `DT_NEEDED` 一致：`librknnrt.so` + `librga.so.2`（板卡自己的运行时）、
  无 `libc++`。

两边的编译参数与已认证的 `build_mosaic_pipeline_arm64.sh` **逐字一致**，只有路径
换成了包内路径，所以行为等价。

### md5 的两个已知「假差异」（都不是 bug）

1. **`.a` 每次构建 md5 都变**：`ar` 会把成员的时间戳写进归档索引，所以
   `libmosaic_pipeline.a` **不是可复现构建**——同一个包、同样的源、同一台机器，
   重新跑一次 `build_lib.sh`，md5 就会变。**别拿它当回归基线。**
   （实测：`6ec51fc0…` → `800c2f31…`，而 `.o` 与两个可执行文件的 md5
   始终不变。）
2. **脚本构建 vs 一行命令构建，md5 也不同**：脚本用**绝对路径**编译
   （`$HERE/src/npu/mosaic_pipeline.cpp`），README 的一行命令用**相对路径**
   （`src/npu/mosaic_pipeline.cpp`）。`assert()` 嵌的是 `__FILE__`，路径长度
   不同 ⇒ `.o` 差 208 字节。**行为一样**，只是字节不同。

要拿可复现的基线，请用 `out/mosaic_pipeline.o` 或两个可执行文件的 md5。

---

## 交付给嵌入式组

本包要交给**嵌入式组**集成。**目标板卡不是我们手上这块 RK3588 开发板**，所以
必须分清「板卡无关」和「板卡相关」两部分：

* **板卡无关（直接可用）**：全部源码（`src/**`）、全部头文件（`include/**`）、
  **OpenCL kernel 资产（`runtime/kernels/**`）**、两个构建脚本（`build_lib.sh` /
  `build_demo.sh`）。这些文件**不含任何板卡私有路径**，换一个 RK3588 板子照样能编。
* **板卡相关（必须由他们自己板子提供/确认）**：
  * `librknnrt.so`、`librga.so.2`、`libOpenCL.so.1` —— **必须用他们自己板子 BSP
    的版本**，不能直接用我们这份。我们验证过的版本 + md5 只写在
    `MANIFEST.md` §C 里**供对照**。
  * 模型 `.rknn` —— 可以直接用我们这份，也可以换他们自己的（但**换模型必须
    保持输入 640×640×3 fp16、输出 2×[1,1,320,320]**，否则后处理不匹配）。
  * **绑核参数** —— 我们预设的旋钮是**按这块板子的核拓扑排的，换板子必须重设**。
    详见 `MANIFEST.md` §D。

> **完整交付清单（含每个文件的 md5 / 字节数 / 用途 / 板卡相关性）见
> [`MANIFEST.md`](MANIFEST.md)。**

## 零参数运行

板卡上 demo 的**默认路径在本板全部存在**：

| 项 | demo 默认值 | 本板是否存在 |
|---|---|:--:|
| 模型 | `/mnt/manu/ptq/models/p0_off_A_none.rknn` | ✅ |
| kernel 目录 | `/root/kernels` | ✅ |
| 帧目录 | `/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324` | ✅ |

所以**不带任何参数就能直接跑**：

```bash
ssh -p {{BOARD_PORT}} {{BOARD}} '/mnt/manu/mosaic_sdk/mosaic_api_test --limit 12'
```

> **换板子时这三个默认路径大概率不同**（模型放哪、kernel 放哪、帧从哪来，都
> 取决于他们的系统布局）⇒ 用 `--model` / `--kernel-dir` / `--raw-root` /
> `--sequence` 显式覆盖，别照抄默认值。
>
> 注：上面零参数运行依赖的是**本板已有的 `/root/kernels`**。如果按下面
> 「NFS 交付目录内容」把 kernel 放到 `/mnt/manu/mosaic_sdk/kernels/`，那就必须
> 显式加 `--kernel-dir /mnt/manu/mosaic_sdk/kernels`。

## NFS 交付目录内容

`{{NFS_ROOT}}/mosaic_sdk/`（板卡侧 = `/mnt/manu/mosaic_sdk/`）
应当包含这些：

```
{{NFS_ROOT}}/mosaic_sdk/
├── mosaic_api_test          ← 示例 demo 可执行文件
├── mosaic_contract_test     ← 契约测试可执行文件
├── libmosaic_pipeline.a     ← 静态库
├── mosaic_pipeline.h        ← 公开 C ABI 头（客户自己写代码要用）
├── kernels/
│   ├── warp_median_fused.cl     ← 运行时必需
│   ├── sortnet_generated.inc    ← 运行时必需
│   └── gmc_feature_gpu.cl       ← 可选（探针用，库不加载）
├── README.md                ← 本文件
└── MANIFEST.md              ← 交付清单
```

拷贝命令（**由 Leader / 使用者执行**）：

```bash
SRC={{PKG_ROOT}}
DST={{NFS_ROOT}}/mosaic_sdk
mkdir -p "$DST/kernels"
cp -f "$SRC/out/mosaic_api_test" "$SRC/out/mosaic_contract_test" \
      "$SRC/out/libmosaic_pipeline.a" "$SRC/src/npu/mosaic_pipeline.h" \
      "$SRC/README.md" "$SRC/MANIFEST.md" "$DST/"
cp -f "$SRC/runtime/kernels/gmc_feature_gpu.cl" \
      "$SRC/runtime/kernels/warp_median_fused.cl" \
      "$SRC/runtime/kernels/sortnet_generated.inc" "$DST/kernels/"
```

> **板卡上的 `--kernel-dir` 要指向放 kernel 的那个目录**：如果按上面这样放，
> 就是 `--kernel-dir /mnt/manu/mosaic_sdk/kernels`。

---

## 一键拷贝

拷贝这一步也有一个脚本，**编译 + 拷贝 + 上板验证一条命令**：

```bash
cd {{PKG_ROOT}}
bash copy_to_nfs.sh --build --verify      # 编译 + 拷贝 + 上板验证，一条命令
```

`copy_to_nfs.sh` 会做这些事：拷贝 10 个文件到 NFS（产物 + 头文件 + kernel 资产 +
README + MANIFEST + 脚本自身）→ **逐文件 md5 回验**（NFS 写坏了立刻报错）→
打印目标清单、板卡侧路径和运行命令。

| 变体 | 命令 | 说明 |
|:--|:--|:--|
| 只拷贝 | `bash copy_to_nfs.sh` | 用已有的 `out/` 产物，跳过编译 |
| 编译 + 拷贝 | `bash copy_to_nfs.sh --build` | 拷贝前先跑 `build_lib.sh` + `build_demo.sh` |
| 换 NFS 目录 | `bash copy_to_nfs.sh --nfs /tmp/test_nfs` | 覆盖 NFS 根目录（**不碰真 NFS**），用于试跑 |
| 换板卡 | `bash copy_to_nfs.sh --board root@1.2.3.4:2222` | 覆盖 ssh 目标与端口 |
| 上板验证 | `bash copy_to_nfs.sh --verify` | 拷贝后 ssh 到板卡跑一次 demo（零参数，`--limit 6`） |

> ⚠️ 脚本**绝不 scp / rsync 到板卡**：板卡根分区 `/root` 的 ext4 已损坏，
> >1 MB 的二进制会被写坏。`--verify` 只在 NFS 上**执行**已有的程序，
> **不往板卡写任何东西**。

---

## 这个 build/ 是怎么来的（打包流程）

`build/` 是**生成物**，不是手写的。任何时候想刷新它，跑这个脚本就行：

```bash
# 从工程里把交付用的全部依赖文件整理进 build/（可反复重跑）
bash {{SRC_ROOT}}/packaging/make_build_dir.sh
```

它会做四件事：

1. 把工程里的**全部依赖文件**拷进 `build/`：`src/npu/`（库 + demo + 契约测试）、
   `src/cpp/`、`src/opencl/host/`、`include/`（第三方头）、`runtime/kernels/`
   （OpenCL kernel 资产）、三个脚本 + 本 README。
2. **删掉陈旧文件**：`build/src`、`build/include`、`build/runtime` 下不再需要的
   旧文件会被清掉（逐个打印 `[stale] 已删除 …`）。
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
