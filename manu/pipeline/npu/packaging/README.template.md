> ℹ️ **这份文件怎么来的**：交付物 `README.md` 由工程里的
> `manu/pipeline/npu/packaging/README.template.md` 生成（占位符由脚本按实况填充；
> 模板漏改一个，脚本会当场报错，不会产出带双花括号的文档）。**请勿手改交付物**。
>
> ℹ️ **命令里的绝对路径（`/media/manu/...`）是作者机器上的**：把 `cd <路径>` 换成
> 你自己的包目录即可 —— **所有构建脚本都按脚本自身所在目录定位，拷到哪个目录都能编**；
> 工具链 / OpenCV / 板卡运行时库的位置用 `RK3588_TOOLCHAIN` / `OPENCV_INSTALL` /
> `BOARD_LIBS` / `RT_LIB_DIR` 四个环境变量覆盖。

# mosaic_pipeline SDK —— 独立构建包（RK3588 / aarch64）

本目录是一个**自包含的交付包**：库源码、示例 demo、契约测试、第三方头文件全部
拷在这里面，可以**脱离仓库单独编译**。

```
包根目录（仓库内是 build/，交付给集成方时是 mosaic_sdk/）
├── README.md          ← 你正在看的这份（四条命令）  ★由 packaging/README.template.md 生成
├── MANIFEST.md        ← 交付清单（md5 表自动生成）★由 packaging/MANIFEST.template.md 生成
├── build_lib.sh       ← 只编译库
├── build_demo.sh      ← 只编译测试程序
├── copy_to_nfs.sh     ← 一键整树拷贝到 NFS（可选带板卡验证）
├── known_hosts        ← 板卡主机密钥基线（ssh 严格校验用；板卡换密钥会直接失败，这是故意的）
├── src/
│   ├── npu/           mosaic_pipeline.{cpp,h}, mosaic_core.h, mosaic_post_gpu.h,
│   │                  mosaic_api_test.cpp, mosaic_contract_test.cpp,
│   │                  mosaic_client_example.cpp, rknn_mosaic_e2e.cpp
│   ├── cpp/           gmc_stream_ocl.cpp, md5.h
│   └── opencl/host/   ocl_host.h, post_local_max.cl（kernel 可读副本，运行时不需要）
├── include/           rknn_api.h, stb_image*.h, rga/ + CL/
├── runtime/kernels/   warp_median_fused.cl, sortnet_generated.inc（运行时必需）,
│                      gmc_feature_gpu.cl（可选，库不加载）
└── out/               ← 产物落在这里
    ├── mosaic_pipeline.o
    ├── libmosaic_pipeline.a
    ├── mosaic_api_test
    ├── mosaic_client_example
    ├── mosaic_contract_test
    └── rknn_mosaic_e2e
```

> **交付到 NFS 时是整棵树一起拷**（不是只拷几个产物文件）：板卡侧看到的目录
> 结构与这里一一对应——`src/ include/ runtime/ out/` + 顶层 6 个文件
> （`README.md` / `MANIFEST.md` / `build_lib.sh` / `build_demo.sh` /
> `copy_to_nfs.sh` / `known_hosts`）。

交付包磁盘上共 **{{TOTAL_COUNT}}** 个文件 = 受管 **{{FILE_COUNT}}** 个
（源码 **{{SRC_COUNT}}** 个（`src/`）+ 第三方头 **{{HDR_COUNT}}** 个
（`include/`）+ kernel 资产 **{{KERNEL_COUNT}}** 个（`runtime/`）+ 顶层 6 个文件
+ `MANIFEST.md` 自身）+ `out/` 编译产物 **{{OUT_COUNT}}** 个。

产物（全部 aarch64 ELF）：

| 文件 | 是什么 |
|---|---|
| `out/libmosaic_pipeline.a` | **库**：GMC/OpenCL + RGA mosaic + RKNN fp16 + 反投影，公开 C ABI 在 `src/npu/mosaic_pipeline.h` |
| `out/mosaic_api_test` | **给客户的示例 demo**：只调公开 C ABI（open / push / collect / destroy） |
| `out/mosaic_client_example` | **最小嵌入示范**（160 行）：只调公开 C ABI（open / set_post_mode / push / collect / destroy），零参数直接跑，嵌入式组照着抄的骨架 |
| `out/mosaic_contract_test` | **库的契约测试**（验收门禁） |
| `out/rknn_mosaic_e2e` | **端到端测试程序**：测吞吐/延时（12.9 fps 那个数就是它测的），只调公开 C ABI |

当前产物的 md5（**本次构建的快照**；`libmosaic_pipeline.a` 每次构建都会变，
见文末「md5 的两个已知假差异」）：

| 文件 | md5 |
|---|---|
| `out/mosaic_pipeline.o` | `{{OBJ_MD5}}` |
| `out/libmosaic_pipeline.a` | `{{LIB_A_MD5}}` |
| `out/mosaic_api_test` | `{{API_TEST_MD5}}` |
| `out/mosaic_client_example` | `{{CLIENT_EXAMPLE_MD5}}` |
| `out/mosaic_contract_test` | `{{CONTRACT_TEST_MD5}}` |

> ℹ️ **本表读的是「当前磁盘上已有的产物」**（`make_build_dir.sh` 不编译，只**读**
> `$DEST/out/*` 已存在的产物算 md5）⇒ **五行都有基线**：产物**已经随交付包一起**
> 落在 `out/` 里（`out/` 整棵树会一起拷到 NFS），不是「包里没有」。
>
> **若某个产物还没编出来**，对应行会显示占位文本「（尚未编译）」而**不是留空**
> ⇒ 编完重跑一次 `make_build_dir.sh` 就会回填。
>
> `out/mosaic_client_example` 的 md5 可以当基线用：这个示范程序源码里**没有
> `assert()`** ⇒ `__FILE__` 路径长度不影响字节 ⇒ **同一份源码**用相对路径编与用
> `build_demo.sh` 的绝对路径编**逐字节相同**（见「编译与链接」一节）。
> ⚠️ 但它**不是跨版本恒定值**：改了示范程序源码，md5 就会变（表中这一行随包更新）。

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

产物：`out/mosaic_api_test`、`out/mosaic_client_example`、`out/mosaic_contract_test`、
`out/rknn_mosaic_e2e`。

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

## ③ 把整包拷到 NFS 挂载点

交付包是**自包含源码包**，所以这一步拷的是**整棵树**：源码 + 第三方头 +
kernel 资产 + 编译产物 + 脚本 + 文档（不再是拷几个产物文件）。

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
rm -f  "$DST/mosaic_api_test" "$DST/mosaic_contract_test" \
       "$DST/libmosaic_pipeline.a" "$DST/mosaic_pipeline.h"
rm -rf "$DST/kernels"

# ② 整树镜像
cp -a "$SRC/src" "$SRC/include" "$SRC/runtime" "$SRC/out" "$DST/"
cp -a "$SRC/README.md" "$SRC/MANIFEST.md" \
      "$SRC/build_lib.sh" "$SRC/build_demo.sh" "$SRC/copy_to_nfs.sh" \
      "$SRC/known_hosts" "$DST/"
chmod +x "$DST/build_lib.sh" "$DST/build_demo.sh" "$DST/copy_to_nfs.sh" \
         "$DST/out/mosaic_api_test" "$DST/out/mosaic_client_example" \
         "$DST/out/mosaic_contract_test" "$DST/out/rknn_mosaic_e2e"

# ③ 建 5 个兼容符号链接（**相对路径**），旧命令才继续可用
( cd "$DST" && \
  ln -sfn out/mosaic_api_test      mosaic_api_test && \
  ln -sfn out/mosaic_contract_test mosaic_contract_test && \
  ln -sfn out/libmosaic_pipeline.a libmosaic_pipeline.a && \
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
ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/mosaic_api_test > /tmp/mosaic_api_test \
  && chmod +x /tmp/mosaic_api_test \
  && /tmp/mosaic_api_test \
  --raw-root   /mnt/manu/lq_ir_bmp \
  --sequence   VIDEO00007_19700101_001324 \
  --model      /mnt/manu/ptq/models/p0_off_A_none.rknn \
  --kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels \
  --limit 12'
```

> ⚠️ **为什么要绕这一圈（`cat` 到 `/tmp` 再跑）**：板卡上的 `/mnt/manu` 是
> **NFS soft 挂载（软挂载）** ⇒ 从它上面**直接执行**超过 1 MB 的二进制会被读坏
> （实测 8.3 MB 的 `out/mosaic_api_test` 直接跑 ⇒ **SIGILL(132)**）。
> 铁律：**>1 MB 的二进制一律先 `cat` 到板卡 `/tmp`（tmpfs）再执行**，
> `chmod +x` 后再跑。详见「必读的坑」。
>
> **兼容别名**：交付根目录留了指向 `out/` 与 `runtime/kernels` 的相对符号链接，
> 所以旧写法 `/mnt/manu/mosaic_sdk/mosaic_api_test --kernel-dir /mnt/manu/mosaic_sdk/kernels`
> **仍然可用**（`kernels` 是 `runtime/kernels` 的别名）。新代码一律用上面
> `out/` + `runtime/kernels` 的写法。

---

## 注意事项

* **契约测试**：`out/mosaic_contract_test`（同样参数，同样**先落到 `/tmp` 再跑**）：

  ```bash
  ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/mosaic_contract_test > /tmp/mosaic_contract_test \
    && chmod +x /tmp/mosaic_contract_test \
    && /tmp/mosaic_contract_test --raw-root /mnt/manu/lq_ir_bmp \
    --sequence VIDEO00007_19700101_001324 \
    --model /mnt/manu/ptq/models/p0_off_A_none.rknn \
    --kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels --limit 60'
  ```

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

  ```bash
  ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/mosaic_api_test > /tmp/mosaic_api_test \
    && chmod +x /tmp/mosaic_api_test && /tmp/mosaic_api_test ...'
  ```

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
`out/libmosaic_pipeline.a`。

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
mosaic_pipeline_destroy(p);
```

> `dets[128]`：库每帧最多保留 100 个连通域（`max_detections`），缓冲区给 128
> ⇒ **不会被静默截断**。

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

**必须先 `cat` 到板卡 `/tmp` 再执行**（板卡 NFS 是 **soft 挂载（软挂载）**，>1 MB 的
二进制直接在 `/mnt/manu/...` 上跑 ⇒ **SIGILL(132)**，见「必读的坑」）：

```bash
ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce \
  && chmod +x /tmp/mce \
  && /tmp/mce \
  /mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324 \
  /mnt/manu/ptq/models/p0_off_A_none.rknn \
  /mnt/manu/mosaic_sdk/runtime/kernels \
  21'
```

**零参数直接跑也行**（上面四个位置参数全是程序内置的默认值）：

```bash
ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce \
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

### `mosaic_client_example` 的参数（**四个位置参数**，没有开关）

| 位置 | 含义 | 默认值 |
|---|---|---|
| 1 | 帧目录（里面是 `frame_%06d.bmp`） | `/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324` |
| 2 | RKNN 模型路径（fp16，输入 640×640×3，输出 2×[1,1,320,320]） | `/mnt/manu/ptq/models/p0_off_A_none.rknn` |
| 3 | OpenCL kernel 目录 | `/mnt/manu/mosaic_sdk/runtime/kernels` |
| 4 | 最多喂多少帧 | `21`（= GMC 的一个完整历史窗口） |

⚠️ **前三个默认值全是本板私有路径** ⇒ **换板子必须显式覆盖**，
否则不是找不到帧就是找不到模型。

**为什么默认喂 21 帧**：GMC 用 21 帧历史做运动补偿。喂不满也能出结果（早期帧的
历史会被 clamp 到最早那一帧，运动估计退化），但**喂满 21 帧才是认证质量的拟合**。
本目录共有 11,810 帧，所以默认只喂 21 帧。

**错误契约**（**不静默降级**）：

| 情况 | stderr | 退出码 |
|---|---|---|
| 参数个数超过 4 | 用法提示 | 2 |
| 帧目录里没有帧 | `[FATAL] <目录> 下没有帧` | 1 |
| 某一帧读不出来 | `[FATAL] 读不出 <路径>` | 1 |
| 建流水线失败 | `[FATAL] 建流水线失败：<err>`（err 是人类可读的原因） | 1 |

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

**只用交付包里的东西**（`cd` 到交付目录即可，**不依赖仓库路径**）：

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
  src/npu/mosaic_client_example.cpp out/libmosaic_pipeline.a -o out/mosaic_client_example \
  -L"$RT_LIB_DIR" -lrknnrt -lrga -Wl,-rpath-link,"$RT_LIB_DIR" \
  -L"$LIBDIR" -L"$INSTALL/lib" -L"$INSTALL/lib/opencv4/3rdparty" \
  -lopencv_imgcodecs -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_video \
  -lopencv_imgproc -lopencv_core -ltegra_hal -lzlib -lOpenCL -lpthread -ldl -lm
```

* **链接对象是 `out/libmosaic_pipeline.a`**（先跑 `bash build_lib.sh` 产出），
  头文件是 `src/npu/mosaic_pipeline.h`。
* **`-I` 列表与包内 `build_demo.sh` 逐字一致**（`-Iinclude -Iinclude/rga
  -Isrc/cpp -Isrc/opencl/host -I$INSTALL/include/opencv4`）——
  **不要加 `-I/usr/include`**：宿主机的头是 x86 的，会污染交叉编译。
* **自己的代码**：把源文件换成你自己的 `.cpp` 即可。头文件写
  `#include "mosaic_pipeline.h"`；**包内那几个程序是靠在 `src/npu/` 里「同目录
  优先」找到头文件的**，所以命令行里没有 `-Isrc/npu` —— **你的代码不在 `src/npu/`
  旁边就要自己加 `-Isrc/npu`**。
* ⚠️ 上面这行用**相对路径**编，包内 `build_demo.sh` 用**绝对路径**编 ⇒
  `assert()` 嵌的 `__FILE__` 路径长度不同 ⇒ **库对象 `out/mosaic_pipeline.o`**
  差 208 字节：**行为一样、字节不同**（已知「假差异」，详见文末「md5 的两个已知
  「假差异」」）。
  **本示范程序源码里没有 `assert()` ⇒ 不受此影响 ⇒ 两种编法逐字节相同**
  （**已实测**：用上面这行相对路径命令编出来的产物，与 `build_demo.sh` 用绝对路径
  编出来的**完全一致** ⇒ 拿示范程序的 md5 做基线是安全的；
  它的 md5 就是上面「产物 md5」表里 `out/mosaic_client_example` 那一行）。
  ⇒ 但你**自己写的代码**如果用了 `assert()`，就会踩这个「假差异」：
  相对路径与绝对路径编出来的二进制 md5 不同，**行为完全一样**。

### 必读的坑

* **必须「一推一收」**：一次只推一帧。流水线是「机会主义最新帧」调度，一次推多帧
  只会回**最后一帧**的结果，前面的帧永远等不到 ⇒ `collect` 会**永久阻塞**。
* **`collect` 会阻塞**，直到下一帧算完 —— 它不是「取已经算好的结果」。
* **板卡 `/root` 的 ext4 已损坏**：不要往 `/root` scp 大文件（>1 MB 会被写坏）；
  中转只走 `/tmp`（tmpfs）。
* **板卡 NFS 是 soft 挂载（软挂载）⇒ >1 MB 的二进制不能直接在 `/mnt/manu/...` 上执行**：
  必须先 `cat` 到板卡 `/tmp`（tmpfs）再跑。这是**传输/缓存损坏**，不是算法 bug。
  * **现象**：直接执行 8.3 MB 的 `out/mosaic_api_test` / `out/mosaic_client_example`
    ⇒ **SIGILL(132)**；板卡 `dmesg` 出现 `RKNPU job timeout` + `soft reset`
    ⇒ **看着像 NPU 驱动中毒，其实不是**（NPU 没问题，是**喂进去的二进制被读坏了**）。
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
    ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce \
      && chmod +x /tmp/mce \
      && /tmp/mce'
    ```
  * ⚠️ **模型文件也受同一条铁律约束**：`--model` 指向的 `.rknn`（本板是
    `/mnt/manu/ptq/models/p0_off_A_none.rknn`，**6.3 MB**）同样是 librknnrt 从
    **soft 挂载**上直读的 ⇒ 也会被读坏。症状不同：**模型加载失败**（不是 SIGILL）。
    稳妥做法是**模型也落 `/tmp`**（实测落盘后 md5 与源一致、结果逐位相同）：

    ```bash
    ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/ptq/models/p0_off_A_none.rknn > /tmp/model.rknn \
      && cat /mnt/manu/mosaic_sdk/out/mosaic_client_example > /tmp/mce \
      && chmod +x /tmp/mce \
      && /tmp/mce \
         /mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324 \
         /tmp/model.rknn \
         /mnt/manu/mosaic_sdk/runtime/kernels \
         21'
    ```

    （注意 `mosaic_client_example` 用的是**四个位置参数**：帧目录 / 模型 /
    kernel 目录 / 帧数 —— 它**不是** `mosaic_api_test` 那种命名开关。）

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
| kernel 目录 | `/root/kernels` | ✅（仅本板碰巧有，见下） |
| 帧目录 | `/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324` | ✅ |

所以**不带任何参数就能直接跑**：

```bash
ssh -p {{BOARD_PORT}} {{BOARD}} 'cat /mnt/manu/mosaic_sdk/out/mosaic_api_test > /tmp/mosaic_api_test \
  && chmod +x /tmp/mosaic_api_test \
  && /tmp/mosaic_api_test --limit 12'
```

> **仍然要 `cat` 到 `/tmp` 再跑**：板卡 NFS 是 **soft 挂载（软挂载）**，>1 MB 的二进制
> 直接在 `/mnt/manu/...` 上执行 ⇒ **SIGILL(132)**（见「必读的坑」）。
>
> **换板子时这三个默认路径大概率不同**（模型放哪、kernel 放哪、帧从哪来，都
> 取决于他们的系统布局）⇒ 用 `--model` / `--kernel-dir` / `--raw-root` /
> `--sequence` 显式覆盖，别照抄默认值。
>
> ⚠️ **`--kernel-dir` 的默认值 `/root/kernels` 是这块开发板上碰巧存在的路径**，
> **不是交付包的一部分**。交付包自带 kernel（`runtime/kernels/`），在交付目录
> 里跑就应当显式写 `--kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels`。

## 源码在哪 / 怎么自己重编

交付目录是**自包含源码包**：源码、第三方头、OpenCL kernel、构建脚本全在里面，
**不依赖本仓库**，放到哪都能原地重编。

| 我要改什么 | 在哪 |
|---|---|
| 示例 demo（跑流水线拿检测框） | `src/npu/mosaic_api_test.cpp` |
| 最小嵌入示范（嵌入式组照着抄的那份） | `src/npu/mosaic_client_example.cpp` |
| 契约测试（验收门禁） | `src/npu/mosaic_contract_test.cpp` |
| 端到端测试程序（测吞吐/延时，12.9 fps 那个数就是它测的） | `src/npu/rknn_mosaic_e2e.cpp` |
| 库实现 | `src/npu/mosaic_pipeline.cpp` |
| 公开 C ABI 头（自己写程序就 include 它） | `src/npu/mosaic_pipeline.h` |
| GMC / OpenCL 引擎 | `src/cpp/gmc_stream_ocl.cpp` |
| OpenCL host 封装 | `src/opencl/host/ocl_host.h` |

原地重编（**构建脚本按脚本自身所在目录定位，拷到哪个目录都能编**）：

```bash
cd <交付目录>                # 例：cd /mnt/manu/mosaic_sdk
bash build_lib.sh            # -> out/mosaic_pipeline.o, out/libmosaic_pipeline.a
bash build_demo.sh           # -> out/mosaic_api_test, out/mosaic_client_example, out/mosaic_contract_test, out/rknn_mosaic_e2e
```

> 第三方头（`include/`：rknn_api.h / stb_image*.h / rga/ / CL/）已随包，
> OpenCL kernel 资产在 `runtime/kernels/`（运行时 `--kernel-dir` 指这里）。
>
> ⚠️ **原地重编后，本 README 顶部的「产物 md5」表会过期**（那是打包时写进去的，
> 交付态的包里没有生成它的脚本）。自检办法：`md5sum out/*` 与表里比对。
> 已知 `out/libmosaic_pipeline.a` **每次编译都会变**（`ar` 写成员时间戳），
> 拿它当回归基线没意义；三个可执行文件在源码不变时是**逐位稳定**的。
> 交叉工具链 / OpenCV / 板卡运行时库的位置用环境变量改：
> `RK3588_TOOLCHAIN`、`OPENCV_INSTALL`、`BOARD_LIBS`、`RT_LIB_DIR`。

---

## NFS 交付目录内容

`{{NFS_ROOT}}/mosaic_sdk/`（板卡侧 = `/mnt/manu/mosaic_sdk/`）是**整棵源码包**：

```
{{NFS_ROOT}}/mosaic_sdk/
├── README.md                 ← 本文件
├── MANIFEST.md               ← 交付清单
├── build_lib.sh              ← 编译库（+x）
├── build_demo.sh             ← 编译 demo / 契约测试 / 端到端测试程序（+x）
├── copy_to_nfs.sh            ← 一键整树拷贝（+x）
├── known_hosts               ← 板卡主机密钥基线（ssh 严格校验用；换密钥直接失败）
├── src/                      ← 全部源码（可参考、可改、可原地重编）
│   ├── npu/    mosaic_pipeline.{cpp,h}, mosaic_core.h, mosaic_post_gpu.h,
│   │           mosaic_api_test.cpp, mosaic_client_example.cpp,
│   │           mosaic_contract_test.cpp, rknn_mosaic_e2e.cpp
│   ├── cpp/    gmc_stream_ocl.cpp, md5.h
│   └── opencl/host/   ocl_host.h, post_local_max.cl
├── include/                  ← 第三方头：rknn_api.h, stb_image*.h, rga/×18, CL/×20
├── runtime/kernels/          ← OpenCL kernel 资产
│   ├── warp_median_fused.cl     ← 运行时必需
│   ├── sortnet_generated.inc    ← 运行时必需
│   └── gmc_feature_gpu.cl       ← 可选（探针用，库不加载）
├── out/                      ← 编译产物
│   ├── mosaic_pipeline.o
│   ├── libmosaic_pipeline.a
│   ├── mosaic_api_test           ← 示例 demo 可执行文件
│   ├── mosaic_client_example     ← 最小嵌入示范可执行文件
│   ├── mosaic_contract_test      ← 契约测试可执行文件
│   └── rknn_mosaic_e2e           ← 端到端测试程序（吞吐/延时测量）
└── 兼容别名（相对符号链接）
    mosaic_api_test       -> out/mosaic_api_test
    mosaic_contract_test  -> out/mosaic_contract_test
    libmosaic_pipeline.a  -> out/libmosaic_pipeline.a
    mosaic_pipeline.h     -> src/npu/mosaic_pipeline.h
    kernels               -> runtime/kernels
```

拷贝命令（**由 Leader / 使用者执行**）：整树镜像

```bash
SRC={{PKG_ROOT}}
DST={{NFS_ROOT}}/mosaic_sdk
mkdir -p "$DST"

# ① 先删掉【上一版遗留的平铺实文件/实目录】—— cp -a 是「合并」语义，
#    不会删；不删就会留下**旧二进制**，让人误以为在跑新版
rm -f  "$DST/mosaic_api_test" "$DST/mosaic_contract_test" \
       "$DST/libmosaic_pipeline.a" "$DST/mosaic_pipeline.h"
rm -rf "$DST/kernels"

# ② 整树镜像
cp -a "$SRC/src" "$SRC/include" "$SRC/runtime" "$SRC/out" "$DST/"
cp -a "$SRC/README.md" "$SRC/MANIFEST.md" \
      "$SRC/build_lib.sh" "$SRC/build_demo.sh" "$SRC/copy_to_nfs.sh" \
      "$SRC/known_hosts" "$DST/"
chmod +x "$DST/build_lib.sh" "$DST/build_demo.sh" "$DST/copy_to_nfs.sh" \
         "$DST/out/mosaic_api_test" "$DST/out/mosaic_client_example" \
         "$DST/out/mosaic_contract_test" "$DST/out/rknn_mosaic_e2e"

# ③ 建 5 个兼容符号链接（**相对路径**），旧命令才继续可用
( cd "$DST" && \
  ln -sfn out/mosaic_api_test      mosaic_api_test && \
  ln -sfn out/mosaic_contract_test mosaic_contract_test && \
  ln -sfn out/libmosaic_pipeline.a libmosaic_pipeline.a && \
  ln -sfn src/npu/mosaic_pipeline.h mosaic_pipeline.h && \
  ln -sfn runtime/kernels           kernels )
```

> ⚠️ **别省 ① 和 ③**：少了 ① 会留下**上一版的旧二进制**（跑的是旧程序却以为是新版）；
> 少了 ③ 则「旧写法仍然可用」这句话不成立。
> **更省心的做法是用 `bash {{PKG_ROOT}}/copy_to_nfs.sh`**（它三步全做，还带递归 md5 回验）。

> **板卡上的 `--kernel-dir` 指向 `runtime/kernels`**：
> `--kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels`
> （旧写法 `/mnt/manu/mosaic_sdk/kernels` 是**兼容别名**，仍然可用）。

---

## 一键拷贝

拷贝这一步也有一个脚本，**编译 + 拷贝 + 上板验证一条命令**：

```bash
cd {{PKG_ROOT}}
bash copy_to_nfs.sh --build --verify      # 编译 + 拷贝 + 上板验证，一条命令
```

`copy_to_nfs.sh` 会做这些事：**整树镜像**到 NFS（`src/` `include/` `runtime/`
`out/` + 顶层 6 个文件，不是零散拷几个产物）→ **递归 md5 回验**
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
