# e2e 并行流水线：速度与精度对齐最终报告

**日期**：2026-10-15 · **序列** `34_1`（537 帧）· **状态**：精度逐位一致，速度 2.45×

---

## 0. 一句话结论

三级流水线（GMC ∥ letterbox ∥ NPU）在**特征、heatmap+offset 张量、检出结果三层全部与原版逐字节一致**，
检出数恒为 547，坏例分支全程未触发，GMC **计算管线**（fit+warp+median）全程 ≤40 ms
（**该口径不含 BMP 读写**，见 §2.1 口径说明）；
速度从原版 **5.74 fps 提到 14.07 fps，累计 2.45×**。

**唯一未完成项**：`rknn 60.77 ms` 内部构成（硬件推理 vs 张量上传/回读）尚未拆开，
已改好代码但未上板运行，**不得声称已定位**。

---

## 1. 实验环境与对象

### 1.1 硬件与系统

| 项 | 值 |
|---|---|
| 板卡 | RK3588 `evm3588`，`ssh -p 22 root@192.168.0.64` |
| 内核 | Linux 5.10.160 aarch64 / Ubuntu 20.04.6 LTS |
| CPU | 8 核：**cpu0-3 `capacity=405` @1.80 GHz（A55 小核）**、**cpu4-7 `capacity=1024` @2.352 GHz（A76 大核）** |
| GPU | Mali-G610 r0p0，4 CU，max work-group 1024，7902 MiB global |
| OpenCL | 3.0 v1.g13p0-01eac0，`arm_release_ver g13p0-01eac0`，`rk_so_ver 6` |
| 存储 | host `192.168.0.115` NFS 导出 `/run/media/manu/SSD960-LOCAL/nfs`，板卡挂 `/mnt/manu` |

> **CPU 拓扑是本轮最关键的硬件事实**：算力最重的 GMC 必须绑大核（4-7）。
> 默认策略曾把小核当快核用，导致 GMC p50 从 27 ms 恶化到 93 ms。详见 §6.2。

### 1.2 模型与内核产物

| 文件 | md5 | 备注 |
|---|---|---|
| `/mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn` | `ef302edb1b5aff1d376204bb7ad77550` | 3,799,304 bytes，fnv1a32=`70a6edc7`，CPU 算子 0 |
| `/root/kernels/warp_median_fused.cl` | `b764f0a85b9af50773ef1fdbf8e9b295` | GMC OpenCL 融合核 |
| `/root/kernels/sortnet_generated.inc` | `1642b63a2c7605d3bec807e3825b6995` | 生成的中值排序表 |

### 1.3 板卡二进制

| 文件 | md5 | 说明 |
|---|---|---|
| `/root/gmc_stream_ocl` | `aa99da811356694d66a0bce6de108b92` | 原版 GMC（与本地重编**逐字节一致**，证明只加 main guard 是语义空操作） |
| `/root/rknn_heatmap_smoke` | `170379506adb4aeaef2f5a5f8ce65cc2` | 原版 NPU 段（同上） |
| `/root/e2e_par_demo` | **`d314f84ecb1c2711d69d5cb302cdd0e6`** | **产出本报告全部主数字的二进制**，推送时间 17:11:59 |

### 1.4 运行参数（四档完全一致）

```
th=0.2500  stride=2  topk=100  anchor_step=10  fused=gpu
opencv=4.10.0   window=21  stride_step=2  downscale=2  mode=tree
ring_depth=43  lags=2..42
FIT-GATE: shift<=0.25 x frame, scale in [0.50,2.00]
letterbox 512x512 -> 640x640 (r=1.250000, pad l=0 t=0 r=0 b=0)
heatmap 320x320, offset 320x320, implied stride 2
序列 34_1 / 537 帧 / 预热 100 帧剔除 / 有界队列 --depth 6
```

三级线程绑核：`feature → cpu 4-5`，`letterbox → cpu 6-7`，`npu → cpu 0-3`。

---

## 2. 速度结果

### 2.1 四级总表

数字全部取自各档 `run.log`，非估算。**帧时间 = 1000 / FPS**。

| # | 配置 | 产物目录 | FPS | 墙钟 | 帧时间 | GMC p50 | GMC p95 | GMC max | ≤40 ms |
|:--|---|---|--:|--:|--:|--:|--:|--:|---|
| 1 | 原版（串行 + 落盘） | `io_serial` | 5.74 | 93.61 s | 174.3 ms | 27.52 | 28.39 | 28.83 | **PASS** |
| 2 | 去中间落盘（串行） | `v2_noio_serial` | 8.16 | 65.84 s | 122.5 ms | 27.71 | 28.56 | 28.99 | **PASS** |
| 3 | 去落盘 + 并行 2 级 | `v2_noio_parallel` | 10.15 | 52.91 s | 98.5 ms | 27.40 | 28.10 | 30.71 | **PASS** |
| 4 | **去落盘 + 并行 3 级** | `split_lb` | **14.07** | **38.18 s** | **71.1 ms** | 27.93 | 28.57 | 32.10 | **PASS** |

**累计 5.74 → 14.07 fps = 2.45×。**

> **⚠️ 40 ms 判据的口径（初版未标清，已补）**：表中「≤40 ms」判的是
> **计算管线本体**，即 `gmc_stream_ocl.cpp:1585` 的 `t_total`（fit + warp + median），
> 判定式在 `e2e_par_demo.cpp:484` 为 `p95<=40 && max<=40`。**不含 BMP 读写**。
>
> 含 I/O 的 GMC wall（见 §2.3）**超过 40ms**：第 1 档 avg 46.18 / max 51.30 ms，
> 第 4 档 avg 33.19 / max 37.32 ms。
> 即：**CPU 特征段的纯计算满足 40ms 约束；把 BMP 编解码算进去则只有并行档（去落盘）满足。**
> 这与原始需求「特征提取部分 fit+warp 保证不超过 40ms」的表述一致（该表述指计算量），
> 但若按「CPU 特征段端到端含 I/O」解读，则只有第 2–4 档达标。

### 2.2 提速分解（逐级归因）

| 步骤 | 增益 | 机制 |
|---|--:|---|
| 5.74 → 8.16 | **+42%** | 去掉中间落盘。每帧省下 load 8.38 ms + write 29.55 ms ≈ 38 ms 纯 I/O |
| 8.16 → 10.15 | **+24%** | GMC 与 NPU 两级重叠。GMC 32.05 / NPU 86.32，串行等价 118.37 → 实际 98.5 |
| 10.15 → 14.07 | **+39%** | letterbox 拆成独立一级。NPU 线程从 94.45 降到 66.99 |

### 2.3 各级墙钟与重叠效率

| 档 | GMC wall avg | NPU wall avg | 串行等价 | 实际帧时间 | 重叠效率 |
|---|--:|--:|--:|--:|--:|
| 1 原版 | 46.18 | 119.21 | 165.39 | 174.3 | 0.949× |
| 2 去落盘 | 32.05 | 86.32 | 118.37 | 122.5 | 0.965× |
| 3 并行 2 级 | 31.60 | 94.45 | 126.04 | 98.5 | **1.279×** |
| 4 并行 3 级 | 33.19 | 66.99 | 100.18 | 71.1 | **1.409×** |

三级版 letterbox 独占线程实测：`avg 12.23 / p50 12.35 / p95 12.36 / max 21.58 ms`。

### 2.4 瓶颈与天花板

- 当前帧时间 71.1 ms，NPU 线程 66.99 ms 占 **94%**。
- 理论上限 `max(33.19, 12.23, 66.99) = 66.99 ms` → **14.93 fps**，当前重叠效率 94%。
- **硬地板** `rknn 60.77 + dequant 2.75 + peaks 0.35 = 63.87 ms` → **15.7 fps**。
- 再往下只能动 `rknn` 本身。**该内部构成尚未定位**（见 §7）。

---

## 3. NPU 段分阶段归因（ms/帧）

| 阶段 | 1 原版 | 2 去落盘 | 3 并行2级 | 4 并行3级 |
|---|--:|--:|--:|--:|
| load（stbi_load 读文件 + BMP 解码） | 8.38 | 0 | 0 | 0 |
| letterbox（512→640 双线性） | 28.38 | 25.10 | 28.60 | **0（已拆出）** |
| **rknn（inputs_set + run + outputs_get）** | 55.79 | 56.88 | 59.96 | **60.77** |
| dequant | 2.72 | 2.72 | 2.75 | 2.75 |
| peaks | 0.34 | 0.35 | 0.35 | 0.35 |
| write（heatmap + det TSV → NFS） | **29.55** | 0 | 0 | 0 |

**两个关键观察**

1. **落盘路径每帧纯 I/O ≈ 37.9 ms**（load 8.38 + write 29.55），占 NPU 段的 32%。
   这就是「原版 5.74 fps 不可信」的直接来源——瓶颈不在算，在 NFS 往返。
2. **letterbox 单独跑只要 12.23 ms，比挂在 NPU 线程上的 28.60 ms 快 2.3 倍。**
   差额是争抢（内存带宽 + cache），不是算法本身。
   ⇒ **"某段代码慢"要先分清是算法慢还是被抢慢**，方法：挪到独占线程再测一次。

---

## 4. 精度对齐：五条独立证据

### 4.1 汇总

| # | 证据 | 对象 | 结果 |
|:--|---|---|---|
| 1 | 特征 BMP 聚合 md5 | `ref` / `io_serial` / `io_parallel` / `e2e`，各 537 张 | 四档全 **`f47f91356b6fb9aaae1c2cf9c8260e24`** |
| 2 | GMC 序列 md5 | 6 档全部 | **`5caa3c6fc72eb7fad4dd5dfd08ebf4ad`** |
| 3 | heatmap + offset 张量 | `ref` vs `io_serial` | 535 帧 × 2 = **1070/1070 逐字节一致** |
| 3 | heatmap + offset 张量 | `ref` vs `io_parallel` | 535 帧 × 2 = **1070/1070 逐字节一致** |
| 4 | 检出 TSV | `ref` vs 两档 | 535 可比帧，**0 行差异**，均 `detections=547` |
| 5 | 链式哈希 | 6 档 | **`d5e81977`**（并经独立实现重算，见 §4.3） |

**证据 1 的聚合 md5 定义**（初版未给定义，已补）：`f47f9135…` =
先对每张 BMP 单独算 md5 取十六进制串，按路径排序后把 537 条十六进制串**拼接**再算一次 md5。
等价写法：

```sh
find $d/feat -name '*.bmp' | sort | xargs md5sum | awk '{print $1}' | md5sum
```

若改为按帧名排序、只聚合哈希流（与目录前缀无关，更便于跨目录引用），四档同为
`546f211fd6d8caaf308b74461b11a340`。**两个值都只依赖帧内容本身，换算法口径会得到不同的字面量，
引用时必须连定义一起给。**

**坏例逻辑**：`fitgate=0 chain=0 fitdeadline=0 cornercap=0`，四档全部未触发。
`--fit-deadline-ms` 保持 **0**（确定性模式），GMC 输出与机器负载无关。

> **⚠️ 统计口径披露：上表所有分布统计（GMC p50/p95/max、NPU 各段均值、哈希链）
> 都剔除了前 100 帧预热**（`e2e_par_demo.cpp:470`，日志打印为 `warmup dropped: 100`）。
> 逐字节比对与链式哈希则覆盖**全部 537 帧**，未剔除。
> 因此「GMC max 32.10 ms」指预热后的最大值，不含前 100 帧。

### 4.2 帧偏移必须是 `ref = serial + 1`，且是唯一解

`ref` 由原版 shell demo 产出、**从第 3 帧开始**（`frame_34_1__000003_hm.bin`，535 帧）；
`e2e_par_demo` 从第 0 帧起（`34_1_34_1_000000_hm.bin`，537 帧）。帧名不同。

扫 0~4 五个候选偏移：

| 偏移假设 | `io_serial` vs `ref` | `io_parallel` vs `ref` |
|---|--:|--:|
| `ref = serial + 0` | 1068 / 1068 不一致 | 1068 / 1068 不一致 |
| **`ref = serial + 1`** | **0 / 1070 不一致** | **0 / 1070 不一致** |
| `ref = serial + 2` | 1070 / 1070 不一致 | 1070 / 1070 不一致 |
| `ref = serial + 3` | 1070 / 1070 不一致 | 1070 / 1070 不一致 |
| `ref = serial + 4` | 1068 / 1068 不一致 | 1068 / 1068 不一致 |

只有 +1 为零差异，邻位全错 ⇒ **对齐是真解，不是退化匹配**。

> **教训（已升级为方法论）**：比对前先扫偏移确定唯一解，再看该解是否为零差异。
> 只看"某个偏移下是 0"不够——全零文件在任何偏移下都是 0。

### 4.3 哈希链把两组证据接上

三级档用 `--no-io` 不落任何字节，精度证据只有链值 `d5e81977`；
而落盘档（已证 == `ref`）当时**没开 `--hash`**，日志里没有链值。⇒ 两组各证一半，中间是断的。

搭桥办法：用**独立实现**重算。按 `e2e_core.cpp:485-487` 的规则
（初值 2166136261，逐帧 `h ^= fnv1a(hm); h ^= fnv1a(off)`，FNV-1a 定义见
`rknn_heatmap_smoke.cpp:75`）直接读落盘档 `.bin`：

```
io_serial   (帧 0..536, 537 帧): d5e81977   (659,865,600 字节)
io_parallel (帧 0..536, 537 帧): d5e81977   (659,865,600 字节)
noio_serial / noio_parallel / split_lb / v2_noio_parallel  日志链值: d5e81977
```

**⇒ 三级档 == 落盘档 == `ref`，传递可证。**

FNV-1a 是字节级串行链，659 MB 用 Python 跑不动（纯 Python 逐字节循环需 ~10⁷ 秒），
故写 C 重算（`/tmp/fnvchain.c`，`gcc -O2 -o fnvchain fnvchain.c`，
用法 `fnvchain <hm目录> <文件名前缀> <首帧> <末帧>`）。

### 4.4 为什么绕开 BMP 不会偷换通道

模型实际看到的是 `[Ch2, Ch1, Ch0]`：`cv::imwrite` 按 BGR 存 BMP，`stb_image` 读回给的是 RGB。
若把内存平面直接喂给推理通道而**不复现这次翻转**，每一张 heatmap 都会变。

内存路（`feature_stream_next_rgb`）原样复现了翻转，
因此 §4.1 证据 5 的链值与文件路完全一致 ⇒ **通道序已对齐**。

---

## 5. 设计要点

### 5.1 三级流水线结构

```
GMC 线程 (cpu 4-5)            letterbox 线程 (cpu 6-7)         NPU 线程 (cpu 0-3)
─────────────────────          ───────────────────────         ──────────────────
读 raw bmp                    取 rgb buffer                  dequant 2.75
fit   ─┐                      letterbox 512→640              rknn_inputs_set
warp   ├→ fit+warp+median     (12.23 ms)                     rknn_run        60.77
median ┘  27.96 ms                                             rknn_outputs_get
        ↓                                                        peaks         0.35
   RGB buffer ──→ [depth 6 有界队列] ──────────────────────→         ↓
                                                              detections → TSV
```

### 5.2 三个使等价性「白送」的设计

1. **只加编译开关 + 产物 md5 自证**：任何「A/B 必须等价」的改造都不重写算法，
   只加 `#ifndef ..._NO_MAIN` 包裹 main，用重编二进制的 md5 一致性证明代码没变。
   `gmc_stream_ocl` / `rknn_heatmap_smoke` 均按此法验证过。
2. **批量路径收成流式薄壳**：`extract_features` 循环 `feature_stream_next`，
   `run_infer` 循环 `infer_one`；帧体抽成共享的 `infer_frame()`
   （文件路 / 内存路 / preboxed 路都进同一函数）。
   ⇒ 串行版与并行版跑同一份逐帧代码，一致是构造性的，不是碰巧。
3. **纯搬运/纯计算的无状态阶段挪线程不需重验精度**（与设计 1 同源）：
   三级版调的是同一个 `letterbox()`、同样参数，只换线程，链值一致即可证。

### 5.3 缓存

`--depth` 有界队列，默认 6 帧。无界时生产者会把整个序列的 512×512×3 BMP 攥在内存里。

---

## 6. 本轮修正的四个问题

前三个是**比对方法**问题（数据从未变），第四个是**记录错误**。

### 6.1 帧名偏移使裸 `diff -rq` 必报全不一致

见 §4.2。裸 `diff -rq` 把「名字对不上」当成内容不一致，1070/1070 全错。

### 6.2 哈希链与落盘档之间证据断裂

见 §4.3。已用独立实现搭桥闭合。

### 6.3 一个作废的产物目录，和一个**我原先判错了**的目录

产物目录 `/mnt/manu/npu/out/e2e_verify/` 里：

| 目录 | 处置 | 依据 |
|---|---|---|
| `noio_serial` / `noio_parallel` | **作废** | **修 `--no-io` 位置 bug 之前的旧运行**。头部写 `io=NONE`，但 breakdown 里 `write` 仍是 25.37 / 25.18 ms，FPS 被低估成 6.59 / 8.16。表内用修复后的 `v2_noio_serial`(8.16) / `v2_noio_parallel`(10.15)。 |
| `e2e/` | **保留，且是第 4 条独立佐证** | ⚠️ 本报告初版曾写「所有偏移下都对不上（1068/1068）」——**该结论是错的，系比对脚本的偏移方向写反所致**。独立复核重扫后：`e2e/` 在 `ref = serial + 1` 下 **1070/1070 逐字节一致**、检出 TSV 0 行差异、特征聚合 md5 同为 `f47f9135…`、链值 `d5e81977`，与 `io_serial` / `io_parallel` **完全相同**。 |

> **判据：头部声明的参数与 breakdown 里的数字必须自洽，不自洽的目录直接作废。**
> （`e2e/` 不违反这条——它是 §21 `e2e_demo` 的产物，配置自洽，只是二进制早于本轮。）

**教训**：我第一次判 `e2e/` 不匹配时，用的脚本把偏移方向写反了（`e2e[k]` vs `ref[k+off]`，
而正确对齐是 `serial(k-off)` vs `ref(k)`），于是在 0~4 五个候选里**没有一个测到正确对齐**，
却据此下了「对不上」的结论。⇒ **扫偏移时必须先确认扫描方向覆盖了真解，
否则「全都错」和「没测对」是无法区分的。**

### 6.4 产出二进制 md5 记错（本次复核查出）

此前记录的「产出二进制 md5 = `5599bcde…`」**是错的**。
板上 `/root/e2e_par_demo` 最后一次推送为 **17:11:59**，实际 md5 **`d314f84ecb1c2711d69d5cb302cdd0e6`**；
而 `io_serial`(17:50:58) ~ `split_lb`(18:07:29) 全部在该推送之后
⇒ **本报告全部主数字由 `d314f84e…` 产出**。

| 产物 | md5 | 是否上过板 |
|---|---|---|
| 板上 `/root/e2e_par_demo` | `d314f84ecb1c2711d69d5cb302cdd0e6` | 是，产出全部主数字 |
| 本地 `out/e2e_par_demo`（新版，含 rknn 三段拆分） | `cfc750ad70e2b8626938367c1e1ed824` | **否，从未上过板** |

---

## 7. 未完成项

### 7.1 `rknn 60.77 ms` 内部构成未定位

**现状**：代码已改好，尚未上板运行。

- `e2e_core.cpp:413` 原来把 `rknn_inputs_set` / `rknn_run` / `rknn_outputs_get` 加总成一个 `rknn_ms`，拆不开。
- 已改为三段各自计时（新增 `inputs_set_ms` / `run_ms` / `outputs_get_ms`），
  `rknn_ms` **保留作三者之和**以兼容旧报告口径。
- `--rknn-perf`（`RKNN_FLAG_COLLECT_PERF_MASK` + `RKNN_QUERY_PERF_RUN`）路线**判定不可用**：
  第一次拿到 `run_duration` = 36280.88 ms（实为累计值）；改成差分后得 744.49 ms/帧，
  远超 rknn wall 69.15 ms，且 `upload+readback` 为负数（−675 ms）。该 flag 还会拖慢运行（14.07 → 13.16 fps）。

**源码 md5**（本地 = host 镜像，已双侧校验）：

| 文件 | md5 |
|---|---|
| `e2e_core.h` | `846bba0a70d31d4d204f123e12b617f5` |
| `e2e_core.cpp` | `7ab66dc57431fd96d1ba534afd364612` |
| `e2e_par_demo.cpp` | `3a0e7e05b6ededa0317627b0ccf839cf` |
| `out/e2e_par_demo` | `cfc750ad70e2b8626938367c1e1ed824` |

**下一步**：推 `cfc750ad…` 上板，跑一次三级档，读 `inputs_set / rknn_run / outputs_get` 三段。
若大头在 `outputs_get`（每帧约 1.2 MB 回读），有优化余地；
若在 `rknn_run` 本身，那是模型/驱动的地板。
**需逐次授权（铁律一），本轮未执行。**

---

## 8. 限定（引用前必读）

1. **只测 `34_1` 一个序列、537 帧**，每档**单次运行、未重复取方差**。
   原版档两次运行实测 5.78 / 5.74 fps；本报告统一取 `io_serial/run.log` 的 5.74。
   其余 23 序列、`--fused cpu`、更深缓存均未覆盖。
2. **速度依赖 NFS 写吞吐**：原版档的 write 29.55 ms 打在 NFS 上。
   真实产品若写本地存储，该项显著下降、FPS 更高；反之若 heatmap 必须落盘，
   则无 I/O 档不可用，2/3/4 档不成立。
3. **三级档未在板上落文件做逐位复核**，靠链值传递证明（链值已用独立实现重算验证，见 §4.3）。
4. **`letterbox` 若改算法会改变像素**，必须重新走一遍逐位验证。
   本轮只换线程不换算法，故未重验。
5. **`--fit-deadline-ms` 必须保持 0**，否则 GMC 输出会变成 load-dependent，
   并行化将破坏确定性。
6. **NPU 并行度上限是 2 不是 3**：第 3 个并发上下文触发
   `RKNPU: job timeout ... elapsed time: 6232015us` → soft reset →
   `BUG: spinlock bad magic on CPU#2, kswapd0/146`，需手动断电恢复。
   P≥4 时吞吐数字是假的（9 进程只有 2 个真产出 60 帧）。

---

## 9. 交付物与复现

### 9.1 文件

| 路径 | md5 | 行数 |
|---|---|--:|
| `manu/pipeline/npu/e2e_core.h` | `846bba0a70d31d4d204f123e12b617f5` | — |
| `manu/pipeline/npu/e2e_core.cpp` | `7ab66dc57431fd96d1ba534afd364612` | — |
| `manu/pipeline/npu/e2e_par_demo.cpp` | `3a0e7e05b6ededa0317627b0ccf839cf` | — |
| `manu/pipeline/npu/e2e_demo.cpp` | 串行 demo（未改） | — |
| `manu/pipeline/npu/build_e2e_demo_arm64.sh` | 合并两条构建链 | — |
| `manu/pipeline/npu/cmp_e2e_outputs.py` | 逐字节比对（内置 PUSH_OFFSET 帧号映射） | — |
| `manu/docs/experiments/2026-10-15_e2e_par_pipeline.md` | 实验记录 §1–§8 | 405 |
| 本报告 | `manu/docs/experiments/2026-10-15_e2e_par_final_report.md` | — |
| `manu/memory/memory.md` | §22 / §23 / **§24** | — |
| `manu/memory/memory_compact.md` | §18 / §19 / **§20** | — |

全部源码与文档已按铁律零同步到 host 镜像 `/home/manu/mnt/pycharm_project_10ae9e2e/`，md5 双侧一致。

### 9.2 复现命令

```sh
# 三级流水线（产出 14.07 fps 的配置）
/root/e2e_par_demo --raw-root raw --sequence 34_1 --out-root OUT \
  --model /mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn \
  --kernel-dir /root/kernels --fused gpu --anchor-step 10 --th 0.25 \
  --mode parallel --no-io --hash --split-lb --depth 6 \
  --cpu-feat 4-5 --cpu-lb 6-7 --cpu-npu 0-3

# 串行基线（产出 5.74 fps）
/root/e2e_par_demo ... --mode parallel            # 2 级（去掉 --split-lb）
/root/e2e_par_demo ... --mode serial              # 串行
```

新增参数：`--split-lb`（letterbox 独立成一级）、`--cpu-lb`（letterbox 线程绑核，默认 6-7）、
`--depth`（有界队列深度）、`--warmup`、`--rknn-perf`（**已判定不可用**）。

### 9.3 精度比对命令（host 上执行，NFS 就地可读）

```sh
V=/run/media/manu/SSD960-LOCAL/nfs/npu/out/e2e_verify

# 帧偏移扫描，唯一解 = ref = serial + 1
python3 - <<'PY'
import os,hashlib,re
base="/run/media/manu/SSD960-LOCAL/nfs/npu/out/e2e_verify"
def md5(p):
    h=hashlib.md5()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()
ki=lambda n:int(re.search(r'(\d{6})_hm\.bin$',n).group(1))
ref=sorted(ki(n) for n in os.listdir(f"{base}/ref/hm") if n.endswith("_hm.bin"))
for s in ("io_serial","io_parallel"):
    sk=set(ki(n) for n in os.listdir(f"{base}/{s}/hm") if n.endswith("_hm.bin"))
    for off in range(5):
        common=[k for k in ref if (k-off) in sk]
        bad=sum(1 for k in common for suf in("hm","off")
                if md5(f"{base}/{s}/hm/34_1_34_1_{k-off:06d}_{suf}.bin")
                   !=md5(f"{base}/ref/hm/frame_34_1__{k:06d}_{suf}.bin"))
        print(f"{s}(帧k) vs ref(帧k+{off}): 张量={len(common)*2} 不一致={bad}")
PY

# 检出 TSV（同一映射）
python3 - <<'PY'
base="/run/media/manu/SSD960-LOCAL/nfs/npu/out/e2e_verify"
rows=lambda p:{f[0]:"\t".join(f[1:]) for f in (l.rstrip("\n").split("\t") for l in open(p)) if len(f)>=2}
ref=rows(f"{base}/ref/npu_det.tsv")
for s in ("io_serial","io_parallel"):
    t=rows(f"{base}/{s}/npu_det.tsv")
    bad=sum(1 for k,v in ref.items()
            if (kn:=f"34_1_{int(k.split('__')[1][:6])-1:06d}.bmp") in t and t[kn]!=v)
    print(s,"det 不一致帧数 =",bad)
PY

# 链式哈希独立重算
gcc -O2 -o /tmp/fnvchain /tmp/fnvchain.c
/tmp/fnvchain $V/io_serial/hm   34_1_34_1_ 0 536   # -> d5e81977
/tmp/fnvchain $V/io_parallel/hm 34_1_34_1_ 0 536   # -> d5e81977
```

> 比对在 host 上做而非板卡：板卡是 busybox ash（无 `$(k-1)` 算术），
> 且 NFS 就在 host 本地可见，不必过 ssh 逐帧调用。

---

## 10. 最终判定

| 判定项 | 结论 |
|---|---|
| 与原版特征通道一致？ | **是**，537/537 逐字节一致 |
| 与原版 heatmap 一致？ | **是**，1070/1070 逐字节一致（偏移经扫描确认为唯一解） |
| 与原版检出结果一致？ | **是**，535 可比帧 0 行差异，均 547 检出 |
| 坏例逻辑保留？ | **是**，`fitgate/chain/fitdeadline/cornercap` 全 0，四档一致 |
| GMC ≤40 ms？ | **是**，四级 p95 ≤28.57、max ≤32.10 ms |
| 提速？ | **2.45×**（5.74 → 14.07 fps） |
| `rknn` 内部构成定位？ | **否，未完成**，不得声称已定位 |
