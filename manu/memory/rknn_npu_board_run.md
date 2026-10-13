# RK3588 NPU 推理打通（rknn_yolov5_demo 参考 + 自写最小测试程序）

> **2026-10-13 板卡 `evm3588` 实测。本专题一次性解除两条长期阻塞，并更正一条错误结论。**
> 代码与工具：`manu/pipeline/npu/`（`rknn_npu_smoke.cpp` / `build_rknn_smoke_arm64.sh` / `run_npu_smoke.sh`）。

---

## 一、判决摘要

| 项 | 结论 |
|:---|:---|
| **RK3588 NPU** | ✅ **可用**。驱动 RKNPU **0.9.8**（20240828）正常初始化，runtime **librknnrt 2.3.0** |
| **RKNN 推理链路** | ✅ `init → inputs_set → run → outputs_get` 全通，输出正确（bus.jpg 检出 1 bus + 4 person，**OSD 目视二次确认**） |
| 🔴 **更正** | **「NPU 驱动未使能、RKNN 路线走不通」是错的**，见第二节 |
| 🔴 **更正** | 「板卡 `/dev/rknpu` 不存在」不是缺陷，是 **DRM 架构的正常表现** |
| ✅ **解除** | NFS 挂载阻塞已解除，方向确认为「板卡挂 host」且**现已生效** |
| 🔴 **新阻塞** | 换成 **Trial 0474 无任何 NPU 部署产物**（老结论仍然成立） |

---

## 二、🔴 更正一：NPU 驱动一直都在，旧判据全部是假阴性

旧记录（2026-10-09）：「`/dev/rknpu` 不存在、`/proc/devices` 无 rknpu 条目、`dmesg` 无 rknpu 行
⇒ 当前内核未使能 rknpu 驱动，RKNN 路线走不通」。**这一整条链是错的。**

实测：

```
[    3.771924] RKNPU fdab0000.npu: Adding to iommu group 0
[    3.773582] RKNPU fdab0000.npu: rknpu iommu is enabled, using iommu mode
[    3.774235] [drm] Initialized rknpu 0.9.8 20240828 for fdab0000.npu on minor 1
$ readlink -f /sys/class/drm/card1/device/driver
/sys/bus/platform/drivers/RKNPU
$ dmesg | grep -c rknpu
16
```

- 驱动注册在 **DRM 子系统**（`[drm] Initialized rknpu ... on minor 1`），
  因此设备节点是 **`/dev/dri/card1`**（渲染节点 `/dev/dri/renderD129`），**不是 `/dev/rknpu`**。
- ⇒ 旧判据里三条命令（`ls /dev/rknpu`、`grep rknpu /proc/devices`、`lsmod | grep rknpu`）
  **对 0.9.x 架构必然全部失败**，却据此得出「驱动缺失」的结论。
- `lsmod` 查不到 rknpu 也正常：它是**内建驱动**，不走可加载模块（与 Mali 同理，见 `rules.md` 二·1b）。

**⇒ 今后判断 NPU 是否存在，唯一正确的探针**：

```bash
readlink -f /sys/class/drm/card*/device/driver    # 出现 .../drivers/RKNPU 即为 RKNPU
```

**方法论教训**：这是「探测手段选错 ⇒ 假阴性 ⇒ 阻塞了一条主线」的典型。
本项目此前据此把 NPU 标记为「走不通」，实际只是用 1.x 时代的探针去测 0.9.x 的板子。
**教训升级：断言「某硬件不可用」之前，必须先证明探测手段对目标版本是有效的**，
而不是「探针没报错 + 没找到」就下结论。

---

## 三、🔴 更正二：SDK 自带的 `librknnrt.so` 是过期版本，禁止链接

| 位置 | 版本 | 可否使用 |
|:---|:---|:---|
| 板卡 `/usr/lib/librknnrt.so` | **2.3.0**（2024-11-07，7.26 MB） | ✅ 实际被加载，SONAME = `librknnrt.so` |
| 板卡 `/usr/lib/librknnrt.so_bak` | 2.3.2（2025-04-09，7.73 MB） | 备份，未启用 |
| SDK `runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so` | **1.4.0**（2022-09-09，3.57 MB） | 🔴 **过期，禁止链接** |

🔴 **头文件与库不同源**：SDK 的 `rknn_api.h` 是 **2.x** 的（含 `RKNN_QUERY_PERF_RUN`、
`RKNN_QUERY_NATIVE_*`、`rknn_perf_run`），但同一目录下的 `.so` 却是 **1.4.0**。
`build info ≠ 运行时链接的库` 这条教训（`falsified_archive.md` 第三十七节）在 SDK 上同样成立。

**本项目做法**：把板卡那份 `.so` 经 NFS 取回 host 当链接目标
（`cp /usr/lib/librknnrt.so /mnt/manu/npu/`，MD5 `56e425dcc7b59ab0e1aec685b178b99f`），
并在构建脚本里**断言 `DT_NEEDED` 里出现的是纯 SONAME `librknnrt.so` 而不是绝对路径** ——
否则会去找 SDK 目录下那份 1.4.0。

---

## 四、为什么不直接跑 SDK 的 `rknn_yolov5_demo`

该 demo 的 `CMakeLists.txt` 依赖两个板卡都没有的库：

- **OpenCV C++ SDK**：板卡只有 pip 的 `cv2.abi3.so` 自包含模块，无头文件 / 无可链接库 / 无 `.pc`
- **librga**：板卡上没有

⇒ 自写最小程序，**只链接 librknnrt**，JPEG 解码与缩放用 SDK 自带的 header-only
`stb_image.h` + `stb_image_resize.h`，板卡零安装依赖。
模型/图直接复用 SDK 的 `model/RK3588/yolov5s-640-640.rknn` 与 `model/bus.jpg`。

模型预处理契约（取自 SDK 的 `convert_rknn_demo/yolov5/onnx2rknn.py`）：
`mean_values=[[0,0,0]], std_values=[[255,255,255]]` ⇒ `(x-0)/255` 已烘进图，
**喂 uint8 RGB NHWC 即可**；SDK demo 也是直接 resize 到模型输入（无 letterbox），本程序保持一致。

---

## 五、实测输出与时延

### 5.1 正确性

```
runtime api_version = 2.3.0   driver_version = 0.9.8   n_input=1 n_output=3
in [0] dims=[1,640,640,3] fmt=NHWC type=INT8 zp=-128 scale=0.003922
out[0] dims=[1,3,85,80,80]  out[1] dims=[1,3,85,40,40]  out[2] dims=[1,3,85,20,20]
detections: person 0.822 / person 0.819 / person 0.788 / bus 0.677 / person 0.406
```

与 `bus.jpg` 内容一致 ⇒ **NPU 推理结果正确**。

### 5.2 分段 perf（板卡实测，10 warmup + 30 稳态帧）

```
host-side submit->fetch total: mean=25.101 ms  median=25.041  min=24.958  p95=25.694  p99=25.847  max=25.847
throughput = 39.84 fps
  inputs_set     mean=  0.266  median= 0.264  min=0.262  p95=0.278  max=0.285   (1.1%)
  rknn_run       mean= 24.362  median=24.306  min=24.228 p95=24.909 max=25.051  (97.1%)
  outputs_get    mean=  0.473  median= 0.468  min=0.459  p95=0.516  max=0.526   (1.9%)
one-shot: preprocess 13.075 ms
```

- 🔴 **进嵌入式预算应取 `rknn_run` = 24.36 ms**，不是端到端 25.10 ms；
  两者差值（0.74 ms）全是 `inputs_set` + `outputs_get` 的驱动拷贝，与 NPU 算力无关。
- 🔴 **`RKNN_FLAG_COLLECT_PERF_MASK` 让推理慢 54%**（25.10 → 38.82 ms）。
  **报真实吞吐必须关掉它**；只有要读 NPU 侧 `run_duration` 时才开。
- `outputs_get` 0.47 ms 对应回传 **2,142,000 B**（≈4.5 GB/s 有效带宽）—— 输出体积是实打实的传输成本，
  后续接自研模型时可考虑降低输出头数量或量化位宽来省这块。
- NPU 频率实测 `cur_freq = 1000000000`（可用区间 300M~1000M）⇒ **不在降频**。
- NPU load：`Core0 46%, Core1 0%, Core2 0%` ⇒ **三核未跑满**，是后续可挖的空间。
- ⚠️ 端到端口径**含约 2.1 MB 输出 memcpy**，**不得当作纯 NPU 算子耗时**。

### 5.3 OSD 可视化

`--osd <path>` 在原图上画框 + 类别/置信度 + 顶部信息条，输出 JPEG。
绘制使用自带的 5×7 点阵字体与矩形函数，**不依赖 OpenCV 与任何字体库**。

实测产物 `/mnt/manu/npu/out/bus_osd.jpg`（640×640 q92）：黄色框精准罩住公交车，
四个 person 框分别落在画面中的行人身上，**目视与数值结果一致** ⇒ NPU 推理正确性获得视觉二次确认。

⚠️ 板卡上输出目录必须先存在且对 host 可写，否则 `stbi_write_jpg` **静默返回 0 失败**：
`mkdir -p /mnt/manu/npu/out && chown 1000:1000 /mnt/manu/npu/out`（本轮踩过一次）。

---

## 六、踩坑（六个，全是静默失效）

> 四个都不抛异常、不改指标、只给「看起来有点不对」的答案。靠四件工具定位：
> 空对照（反相输入必须改变输出）、输入缓冲 min/max 打印、全 buffer 峰值通道定位、
> **以及把 SDK 官方 demo 跑起来做逐框对账**。

### 6.1 `stbir_resize_uint8` 第 9 个参数是 `num_channels`，不是 filter

```c
// 本版本 easy API 的真实签名（stb_image_resize.h:227）
int stbir_resize_uint8(const unsigned char* in, int iw, int ih, int in_stride,
                       unsigned char* out, int ow, int oh, int out_stride,
                       int num_channels);      // ← 没有 filter 参数
```

误把 `STBIR_FILTER_DEFAULT` 当第 9 个参数传 ⇒ **传了 0 通道** ⇒ **一个字节都没写**，
目标缓冲区保持全 0。后果：图是纯黑 → 模型照常跑 → 输出非零且**逐次逐字节完全一致**，
**唯一症状就是「检不出任何东西」**。这就是「确定性 + 非零输出 ≠ 结果正确」的活教材。

### 6.2 它的返回值 `1 == 成功`、`0 == 失败`

`stbir__resize_allocated` 里全部失败分支 `return 0`，唯一成功路径 `return 1`。
按直觉写成 `if (rc != 0) FAIL` 会把正确的 resize 判成失败 —— 本轮真的先写反了一次。

### 6.3 🔴 输出是 5 维 `[N, A, C, H, W] = [1,3,85,H,W]`，不是 SDK demo 假设的 `[1,255,H,W]`

SDK 的 `postprocess.cc` 用的索引式 `(PROP_BOX_SIZE * a + c) * grid_len + i * grid_w + j`
对 `[1,3,85,H,W]` 同样成立（3×85=255，展平后等价），**但维度下标不同**：
把 `dims[2]` 当网格高度会得到 `gh=85` ⇒ 整段解码是噪声。
本程序改为**断言形状**（`n_dims==5 && dims[1]==3 && dims[2]==85`），不匹配就跳过并告警。

### 6.4 🔴🔴 检测框「整体往左上歪」＝ center→left 被做了两次

**症状**：框一律偏上偏左、宽度也偏小，像尺度没变换回去。

**根因**：`process()` 解码最后一步**已经**做了 center→left：

```c
box_x = (box_x + j) * stride;    // 中心
box_w = box_w * box_w * anchor;
box_x -= (box_w / 2.0);           // ← 这之后已经是 LEFT
```

我的 `decode_head` 照抄了这一行，所以 `Det` 里存的**本来就是 left/top**；
但绘制时又按「中心点」处理，**再减一次 w/2** ⇒ 整框左上偏移半个身位。

**为什么没有任何异常信号**：框仍是矩形、仍与目标部分重叠、置信度不变，只是整体平移。
**这类错误不抛异常、不改指标，只能靠「与已知正确实现逐框对账」发现。**

**修法**：`Det` 字段由 `x/y/w/h` 改为显式 **`left/top/right/bottom`**，
并提供 `w()`/`h()` 访问器。**四角坐标语义无歧义，「中心+w/h」有歧义** ——
这是 `rules.md` 二·4「坐标空间契约」铁律的又一实例。

### 6.5 参照基准：把 SDK 官方 demo 本身跑起来

为避免「照抄源码 = 正确」的循环论证，已交叉编译并运行 **Rockchip 官方
`rknn_yolov5_demo`**（`manu/pipeline/npu/{build,run}_sdk_demo*.sh`），作为逐框对账的参照臂。

官方 demo 板卡输出：

```
person @ (114 235 212 527)  0.819099
person @ (210 242 284 509)  0.814970
person @ (479 235 561 520)  0.790311
bus     @ (99  141 557 445)  0.693320
person  @ (78  338 122 520)  0.404960
loop count = 10 , average run  23.428400 ms
```

逐框对账（本程序 vs 官方 demo，l/t/r/b）：

| 检测 | 本程序 | SDK demo | 偏差 |
|:---|:---|:---|:---:|
| bus | 100.6, 143.1, 558.6, 447.2 | 99, 141, 557, 445 | ~2 px |
| person | 212.2, 242.2, 285.6, 509.1 | 210, 242, 284, 509 | ~2 px |
| person | 111.4, 234.6, 218.2, 526.1 | 114, 235, 212, 527 | ~1–6 px |
| person | 478.3, 232.3, 559.9, 523.8 | 479, 235, 561, 520 | ~1–4 px |
| person | 78.9, 337.4, 122.7, 518.8 | 78, 338, 122, 520 | ~1 px |

**全部在个位数像素内**。残差来源明确：本程序用 **stb_image** 解 JPEG，
官方 demo 用 **OpenCV 3.4.5 / libjpeg-turbo**，像素不完全一致
（同 `falsified_archive.md` 记录的「四者无一匹配」类问题）。
⇒ **不是解码 bug，是解码器差异**；两张 OSD 目视一致。

官方 demo 的编译要点（均实测）：

| 项 | 结论 |
|:---|:---|
| CMake 4.2.3 拒绝 `cmake_minimum_required(3.4.1)` | 加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`，**不改 SDK 源码** |
| armhf/aarch64 选择 | CMakeLists 用 `CMAKE_C_COMPILER MATCHES "aarch64"` 判断 ⇒ 交叉前缀必须含 `aarch64` |
| OpenCV | SDK 自带 aarch64 预编译 **3.4.5 静态库**，板卡零安装 |
| librknnrt | 按 **SONAME** 链接 ⇒ 运行时解析到板卡 **2.3.0**（demo 自报 `sdk version: 2.3.0` 已验证） |
| 🔴 `make install` | **故意不执行**：会把 SDK 那份 **1.4.0** 拷到二进制旁，遮蔽板卡 2.3.0 |
| 模型路径 | demo 硬编码 `model/RK3588/...` 与 `./model/coco_80_labels_list.txt`（相对 cwd） |

### 6.6 ⚠️ NFS 目录缓存会给出假的「文件不存在」

host 侧把模型 `mv` 进 `model/RK3588/` 后立刻在板卡跑，demo 报 `Open file ... failed`；
但同一时刻板卡 `stat` 能读到完整 7,945,926 字节。**NFS 目录项缓存陈旧**，重跑即成功。
⇒ **NFS 上的「文件不存在」必须先在板卡侧复核再下结论**，尤其在 host 刚做过 `mv` 之后。
（这是本项目「静默失效优先于崩溃」的又一例：不报错，只是给错答案。）

### 6.7 ⚠️ 同一目录在板卡侧与 host 侧是两���路径

`板卡:/mnt/manu/npu/demo` ≡ `host:/home/manu/nfs/npu/demo`。
脚本里曾把板卡路径直接用于 host 侧 `cp`，报 `Permission denied`。
⇒ `rules.md` 1b-0 的「同路径语义已变」在此具体化为：**同一份数据，两个根路径，写脚本时必须分别定义**。

---

## 七、验收闸门（写进了程序，不可绕过）

1. **`rknn_init` 必须成功**，并打印 runtime / driver 版本。
2. **输入缓冲区 min/max 打印**；`max == 0` 直接 `FAIL` 退出 —— 全零输入会让一切下游数字失去意义。
3. **空对照**：同一模型喂反相图，输出哈希**必须不同**，否则判定「输入没进图」，结论一律不可引用。
4. **确定性**：同输入连跑两次，输出哈希必须相同。
5. **字节数自洽**：`3 * 85 * H * W` 必须等于 `out[i].size`。
6. **形状断言**：见 6.3。
7. **shape 解码自检**：obj 平面最大值与过阈单元数一并打印，区分「阈值/尺度错」与「模型真没看见」。

> 第 3 条正是 `rules.md` 三·7「A/B 判定前必须先立空对照」的应用：
> 没有它，一次「输出非零且稳定」的假通过会被当成 NPU 可用。

---

## 八、NFS 相关的两条实测结论（补充 `rules.md` 1b-0）

1. ✅ **「板卡挂 host」方向已生效**（此前 memory 记为「导出表未刷新、mount 应当失败」）：
   ```
   192.168.0.115:/home/manu/nfs on /mnt/manu type nfs4 (rw,relatime,vers=4.0,
   rsize=1048576,wsize=1048576,namlen=255,soft,proto=tcp,timeo=5,retrans=2,
   sec=sys,clientaddr=192.168.0.64,local_lock=none,addr=192.168.0.115)
   ```
   ⇒ 用户已执行 `exportfs -ra`，该阻塞**解除**。
2. ⚠️ **UID 映射风险已实测坐实**：板卡 mount 未指定 `uid=`/`gid=` 且导出带 `no_root_squash`
   ⇒ 板卡 root 创建的文件在 host 上属 **`root:root`**，`manu` 无写权限。
   **对策**：板卡侧建目录后 `chown 1000:1000`（host 用户 UID/GID 均为 1000），
   本轮用 `/mnt/manu/npu/data` 承载模型与图像，host 侧可正常读写。

---

## 九、存储分工（本轮实际用量）

| 内容 | 位置 | 体积 | 理由 |
|:---|:---|---:|---|
| 可执行文件 | 板卡本地 `/root/rknn_npu_smoke` | 148 KB | 经 NFS 执行会 `Text file busy`（旧方向实测） |
| 模型 `.rknn` | `/mnt/manu/npu/data/`（NFS=host） | 7.9 MB | 不占板卡盘 |
| 测试图 `.jpg` | `/mnt/manu/npu/data/`（NFS=host） | 181 KB | 同上 |
| 链接用 `librknnrt.so` | host `/home/manu/nfs/npu/` | 7.3 MB | 只在 host 侧用，**不推板卡** |

板卡本地盘新增 **仅 148 KB**（`/` 仍有 ~1.7 GB 可用），符合「板卡空间有限，数据一律走 NFS」。