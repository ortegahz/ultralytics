# `manu/pipeline/npu/` — RK3588 NPU 推理打通

> 结论（2026-10-13 板卡 `evm3588` 实测）：**RK3588 NPU 可用，RKNN 推理链路完整跑通。**
> 这推翻了此前「NPU 驱动未使能、RKNN 路线走不通」的记录——详见本文第三节。

## 1. 交付物

| 文件 | 作用 |
|:---|:---|
| `rknn_npu_smoke.cpp` | 最小 NPU 推理测试程序：init → query → inputs_set → run → outputs_get → 解码 → OSD → 分段 perf + 空对照 |
| `build_rknn_smoke_arm64.sh` | 交叉编译（aarch64，链接板卡版 `librknnrt.so`） |
| `run_npu_smoke.sh` | 一键 build + scp 部署 + 运行 |
| `build_sdk_demo_arm64.sh` | **交叉编译 Rockchip 官方 `rknn_yolov5_demo`**（参照臂） |
| `run_sdk_demo.sh` | 部署并运行官方 demo，产出参照框与参照 OSD |
| `rk3588-aarch64-npu-toolchain.cmake` | 给官方 demo 用的最小交叉工具链 |

## 2. 为什么不用 SDK 的 `rknn_yolov5_demo` 直接跑

参考例子在
`external/rknpu2/examples/rknn_yolov5_demo/`，但它的 `CMakeLists.txt` 依赖两个板卡上都没有的东西：

- **OpenCV C++ SDK** —— 板卡只有 pip 的 `cv2.abi3.so` 自包含模块，没有头文件 / 可链接库 / `.pc`
- **librga** —— 板卡上没有

所以本程序只依赖 **librknnrt + SDK 自带的 header-only `stb_image.h` / `stb_image_resize.h`**，
JPEG 解码与缩放全部自己完成，板卡不需要装任何额外东西。

模型和测试图直接用 SDK 自带的：
`external/rknpu2/examples/rknn_yolov5_demo/model/RK3588/yolov5s-640-640.rknn` 与 `model/bus.jpg`。

## 3. 🔴 关键更正：NPU 驱动一直都在

旧记录称「`/dev/rknpu` 不存在、`/proc/devices` 无 rknpu、dmesg 无 rknpu 行 ⇒ 内核未使能 rknpu 驱动」。
**这条是错的。** 实测：

```
[    3.771924] RKNPU fdab0000.npu: Adding to iommu group 0
[    3.774235] [drm] Initialized rknpu 0.9.8 20240828 for fdab0000.npu on minor 1
$ readlink -f /sys/class/drm/card1/device/driver
/sys/bus/platform/drivers/RKNPU
```

- 驱动 **RKNPU 0.9.8** 正常开机初始化，**注册在 DRM 子系统上**，所以设备节点是
  `/dev/dri/card1`（渲染节点 `/dev/dri/renderD129`），**不是** `/dev/rknpu`。
- 因此 `ls /dev/rknpu` 与 `grep rknpu /proc/devices` 全部落空 —— 这两个检查对 0.9.x 架构**必然失败**，
  用它们判断驱动是否存在会得到假阴性。
- ⚠️ 同理，「`dmesg` 无 rknpu 行」也是错的：板上 `dmesg | grep -c rknpu` = **16**。
- `lsmod` 查不到 rknpu 也属正常：它是**内建驱动**，不走模块。

**正确的探测方式**：`ls /sys/class/drm/`，看 `cardN` 的 `device/driver` 软链是否指向 `.../drivers/RKNPU`。

## 4. runtime 版本：必须链板卡那份，不要链 SDK 那份

| 位置 | 版本 | 说明 |
|:---|:---|:---|
| 板卡 `/usr/lib/librknnrt.so` | **2.3.0** (2024-11-07) | 实际会被加载的那一份，SONAME = `librknnrt.so` |
| 板卡 `/usr/lib/librknnrt.so_bak` | 2.3.2 (2025-04-09) | 备份，未启用 |
| SDK `runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so` | **1.4.0** (2022-09-09) | 🔴 **过期，禁止链接** |

SDK 自带的头文件 `rknn_api.h` 是 2.x 的（含 `RKNN_QUERY_PERF_RUN`、`RKNN_QUERY_NATIVE_*`），
但同一个目录下的 `.so` 却是 1.4.0 —— **头文件和库不同源**。
本项目做法：把板卡的 `.so` 经 NFS 取回 host 做链接目标，并在构建脚本里断言
`DT_NEEDED` 里出现的是纯 SONAME `librknnrt.so` 而不是绝对路径。

## 5. 实测输出（`evm3588`，2026-10-13）

```
[2/6] rknn_init
  [ OK ] rknn_init succeeded
  runtime api_version = 2.3.0 (c949ad889d@2024-11-07T11:35:33)
  driver_version     = 0.9.8
  n_input = 1, n_output = 3
  in [0] dims=[1,640,640,3] fmt=NHWC type=INT8 zp=-128 scale=0.003922
  out[0] dims=[1,3,85,80,80]  out[1] dims=[1,3,85,40,40]  out[2] dims=[1,3,85,20,20]
[4/6] input pixels: min=0 max=255 mean=87.68
[5/6] [ OK ] inputs_set + run + outputs_get
  determinism: run#1=4a1fc4a6 run#2=4a1fc4a6 -> IDENTICAL
[5b] null control: original out=4a1fc4a6  inverted out=871a5b52 -> DIFFER (input path is live)
[6/6] raw candidates (thr 0.25) = 55 -> after NMS = 5
      #0 person  score=0.822  box=(212.2, 242.2, 73.4, 267.0)
      #1 person  score=0.819  box=(111.4, 234.6, 106.8, 291.5)
      #2 person  score=0.788  box=(478.3, 232.3, 81.6, 291.5)
      #3 bus     score=0.677  box=(100.6, 143.1, 458.0, 304.0)
      #4 person  score=0.406  box=(78.9, 337.4, 43.8, 181.4)
```

检测结果与 `bus.jpg` 内容一致（1 辆 bus + 多人），**NPU 推理结果正确**。

## 6. 时延与分段 perf（640×640 yolov5s，NPU 满频 1 GHz）

板卡实测（10 warmup + 30 稳态帧，`--loops 30 --warmup 10`）：

```
host-side submit->fetch total: mean=25.101 ms  median=25.041  min=24.958  p95=25.694  p99=25.847  max=25.847
throughput = 39.84 fps

stage breakdown (all measured on the board, wall clock):
  inputs_set     mean=  0.266  median= 0.264  min=0.262  p95=0.278  max=0.285   (1.1% of total)
  rknn_run       mean= 24.362  median=24.306  min=24.228 p95=24.909 max=25.051  (97.1% of total)
  outputs_get    mean=  0.473  median= 0.468  min=0.459  p95=0.516  max=0.526   (1.9% of total)

one-shot:
  preprocess     13.075 ms  (JPEG decode + resize to model input, one shot)
```

| 口径 | 数值 | 说明 |
|:---|---:|:---|
| **`rknn_run`** | **24.36 ms** | **真正该拿去做的预算数字**，占 97.1% |
| `inputs_set` | 0.27 ms | 上传 1.2 MB 输入的转换/拷贝 |
| `outputs_get` | 0.47 ms | 回传 2,142,000 B（≈2.1 MB）输出 |
| 端到端 submit→fetch | 25.10 ms | 含全部 memcpy，**不可当 NPU 算子耗时** |
| `preprocess` | 13.08 ms | 一次性 JPEG 解码+缩放（不计入稳态循环） |

- 🔴 **`RKNN_FLAG_COLLECT_PERF_MASK` 让推理慢 54%**（25.10 → 38.82 ms）。
  **报真实吞吐必须关掉它**（即不加 `--perf`）；只有要读 NPU 侧 `run_duration` 才开。
- NPU 频率实测 `cur_freq = 1000000000`（可用区间 300M~1000M），**不在降频**。
- NPU load：`Core0 46%, Core1 0%, Core2 0%` —— 三个核心没跑满，是后续可挖的性能空间。

## 6b. OSD 可视化

`--osd <path>` 会在原图上绘制检测框 + 类别/置信度标签 + 顶部信息条，并写出 JPEG。
绘制用自带的 5×7 点阵字体与矩形函数，**不依赖 OpenCV，也不依赖任何字体库**。

板卡实测输出 `/mnt/manu/npu/out/bus_osd.jpg`（640×640，q92）：黄色框精准罩住公交车，
四个 person 框分别对应画面中的行人，**目视与数值结果一致**。

> ⚠️ 板卡上目录必须先存在且对 host 可写，否则 `stbi_write_jpg` 静默失败：
> `mkdir -p /mnt/manu/npu/out && chown 1000:1000 /mnt/manu/npu/out`

## 7. 🔴 踩坑：检测框「整体往左上歪」——被减了两次 w/2

**症状**：框画出来一律偏上偏左，宽度也偏小，像缩放没变换回去。

**根因**：`process()` 解码的最后一步已经做了 center→left 转换：

```c
box_x = (box_x + j) * stride;   // 中心
box_w = box_w * box_w * anchor;
box_x -= (box_w / 2.0);          // ← 已经是 LEFT 了
```

我的 `decode_head` 照抄了这一行，所以 `Det` 里存的**本来就是 left/top**。
但绘制时又按「中心点」处理，**再减了一次 w/2** ⇒ 整框左上偏移半个身位。

**为什么没立刻发现**：数值完全自洽——框仍然是矩形、仍与目标部分重叠、置信度也不变，
只是整体平移。**这类错误不会触发任何异常**，只能靠「与已知正确的实现逐框对账」发现。

**修法**：把 `Det` 的字段从 `x/y/w/h` 改成显式的 `left/top/right/bottom`，
并在调用点提供 `w()`/`h()` 访问器。**四角坐标在语义上无歧义，中心+w/h 有歧义**，
这正是本项目 `rules.md`「坐标空间契约」那条铁律的又一个实例。

### 7.1 参照基准：把 SDK demo 本身跑起来

为避免「照抄源码 = 正确」的循环论证，本目录额外把 **Rockchip 官方的 `rknn_yolov5_demo`
交叉编译并跑通**，作为逐框对账的参照臂：

```bash
RK3588_PASSWORD=*** ./run_sdk_demo.sh
```

官方 demo 在板卡上的输出（`model/RK3588/yolov5s-640-640.rknn` + `bus.jpg`）：

```
person @ (114 235 212 527)  0.819099
person @ (210 242 284 509)  0.814970
person @ (479 235 561 520)  0.790311
bus     @ (99  141 557 445)  0.693320
person  @ (78  338 122 520)  0.404960
loop count = 10 , average run  23.428400 ms
```

与 `rknn_npu_smoke` 修正后逐框对比：

| 检测 | `rknn_npu_smoke` (l,t,r,b) | SDK demo (l,t,r,b) | 偏差 |
|:---|:---|:---|:---:|
| bus | 100.6, 143.1, 558.6, 447.2 | 99, 141, 557, 445 | ~2 px |
| person | 212.2, 242.2, 285.6, 509.1 | 210, 242, 284, 509 | ~2 px |
| person | 111.4, 234.6, 218.2, 526.1 | 114, 235, 212, 527 | ~1–6 px |
| person | 478.3, 232.3, 559.9, 523.8 | 479, 235, 561, 520 | ~1–4 px |
| person | 78.9, 337.4, 122.7, 518.8 | 78, 338, 122, 520 | ~1 px |

**全部在个位数像素内**。残差来源明确：本程序用 **stb_image** 解 JPEG，
官方 demo 用 **OpenCV 3.4.5 / libjpeg-turbo**，两者像素不完全一致
（正是 `falsified_archive.md` 记录的「四者无一匹配」那类问题）。
⇒ **这不是解码 bug，是解码器差异**，两个 OSD 图目视完全一致。

### 7.2 官方 demo 的编译要点（均实测）

| 项 | 结论 |
|:---|:---|
| CMake 4.2.3 拒绝 `cmake_minimum_required(3.4.1)` | 加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`，**不改 SDK 源码** |
| armhf/aarch64 选择 | CMakeLists 用 `CMAKE_C_COMPILER MATCHES "aarch64"` 判断 ⇒ 交叉前缀必须含 `aarch64` |
| OpenCV | SDK 自带 aarch64 预编译 **3.4.5 静态库**，板卡零安装 |
| librknnrt | 二进制按 **SONAME** 链接 ⇒ 运行时解析到板卡 **2.3.0**（demo 自报 `sdk version: 2.3.0` 已验证） |
| 🔴 `make install` | **故意不执行**：会把 SDK 那份 **1.4.0** librknnrt 拷到二进制旁边，遮蔽板卡 2.3.0 |
| 模型路径 | demo 硬编码 `model/RK3588/...` 与 `./model/coco_80_labels_list.txt`（相对 cwd） |

### 7.3 其余三个静默失效的坑

1. **`stbir_resize_uint8` 的第 9 个参数是 `num_channels`，不是 filter。**
   easy API 没有 filter 参数（内部自动选），而 `STBIR_FILTER_DEFAULT == 0`。
   误传 filter ⇒ 传了 0 通道 ⇒ **一个字节都没写**，目标缓冲区保持全 0。
   图是纯黑，模型照常跑、输出照样非零且逐次完全一致 —— **唯一的症状就是「检不出东西」**。
2. **它的返回值 `1 == 成功`、`0 == 失败`**（与直觉相反）。写反了会把正确的 resize 判成失败。
3. **输出是 5 维 `[1,3,85,H,W]`（`N,A,C,H,W`），不是 SDK demo 假设的 `[1,255,H,W]`。**
   把 `dims[2]` 当网格高度会得到 `gh=85`，整段解码变成噪声。本程序改为**断言形状**而不是信任它。

> 上述四点都不是「读代码能看出来」的：1 和 2 是猜错签名与返回语义，3 是猜错张量布局，
> 4 是坐标语义二义性导致的重复变换。最终靠四件事定位：
> **空对照（反相输入必须改变输出）、输入缓冲 min/max 打印、全 buffer 峰值通道定位、
> 以及把 SDK 官方 demo 跑起来做逐框对账。**

### 7.4 ⚠️ NFS 目录缓存会骗你

host 侧把模型移进 `model/RK3588/` 后立刻在板卡上跑，demo 报
`Open file ... failed`；但同一时刻板卡 `stat` 能读到完整 7,945,926 字节。
**NFS 目录项缓存陈旧**导致第一次失败，重跑即成功。
⇒ **NFS 上的「文件不存在」要先在板卡侧复核一遍再下结论**，尤其在 host 刚做过 `mv` 之后。

## 8. 用法

```bash
# 一次性准备（板卡侧建好 host 可写的数据目录）
ssh root@192.168.0.64 'mkdir -p /mnt/manu/npu/data && chown 1000:1000 /mnt/manu/npu/data'

# build + 部署 + 跑
RK3588_PASSWORD=*** ./run_npu_smoke.sh
RK3588_PASSWORD=*** ./run_npu_smoke.sh --perf          # 附带 NPU 自报耗时（慢 54%）

# 出 OSD 图（需要板卡上 /mnt/manu/npu/out 已存在且对 host 可写）
ssh root@192.168.0.64 'mkdir -p /mnt/manu/npu/out && chown 1000:1000 /mnt/manu/npu/out'
RK3588_PASSWORD=*** ./run_npu_smoke.sh --loops 30 --osd /mnt/manu/npu/out/bus_osd.jpg
# host 侧直接取回：/home/manu/nfs/npu/out/bus_osd.jpg

# 模型/换图：放到 host 的 /home/manu/nfs/npu/data/ 即可，板卡侧无需改动
```

注意 `board_run.py` 的 `--exec` **只吃单行命令**，多行命令会打断 pty 匹配而静默返回空输出。

## 9. 对项目主线的意义

- RK3588 **NPU 硬件与驱动、runtime 三件套全部就绪**，此前「RKNN 路线走不通」这一阻塞**解除**。
- 真正的阻塞换成了 **Trial 0474 没有任何 NPU 部署产物**（`runs/` 不在同步链路内），
  且 rknn-toolkit2 是 x86 侧 Python 包，转换需在训练服务器上做。
- 本目录证明了转换-部署-推理的**工具链闭环**，可作为把冻结底模搬上 NPU 的起点。