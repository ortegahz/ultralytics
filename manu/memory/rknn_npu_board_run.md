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

`板卡:/mnt/manu/npu/demo` ≡ `host:/run/media/manu/SSD960-LOCAL/nfs/npu/demo`（**2026-10-14 换址后；旧记 `/home/manu/nfs/npu/demo` 已作废**）。
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

1. ✅ **「板卡挂 host」方向已生效，且导出目录已于 2026-10-14 换到 SSD960-LOCAL 大盘**（此前 memory 记为「导出表未刷新、mount 应当失败」）：
   ```
   192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs on /mnt/manu type nfs4 (rw,relatime,vers=4.0,
   rsize=1048576,wsize=1048576,namlen=255,soft,proto=tcp,timeo=5,retrans=2,
   sec=sys,clientaddr=192.168.0.64,local_lock=none,addr=192.168.0.115)
   ```
   ⇒ 用户已执行 `exportfs -ra`，该阻塞**解除**；**2026-10-14 复测仍生效**，`df -h /mnt/manu` = `880G / 54G used / 782G avail / 7%`（旧记的 `/home/manu/nfs` 已从 `/etc/exports` 与内核导出表移除，仅剩遗留拷贝）。
2. ⚠️ **UID 映射风险已实测坐实**（2026-10-14 探针复现）：板卡 mount 未指定 `uid=`/`gid=` 且导出带 `no_root_squash`
   ⇒ 板卡 root 创建的文件在 host 上属 **`root:root`**，`manu` 无写权限。
   **对策**：板卡侧建目录后 `chown 1000:1000`（host 用户 UID/GID 均为 1000），
   本轮用 `/mnt/manu/npu/data` 承载模型与图像，host 侧可正常读写。

---

## 九、ONNX → RKNN 重新量化闭环（2026-10-13）

**判决**：从 SDK 自带 `examples/onnx/yolov5/yolov5s.onnx` 重新量化的 `.rknn`
（`yolov5s_rknn230.rknn`，8.38 MB，MD5 `4b8459fbc6c2c1ad5ba938200ee4433e`）
**与官方 `yolov5s-640-640.rknn` 检出结果一致：5/5 框匹配，类别 5/5 相同，IoU 0.850~0.938。**

代码与文档：`manu/pipeline/npu/convert/`（转换脚本、逐框对账工具、wheel/exec-stack 修补工具）
与 `manu/pipeline/npu/CONVERT.md`。

### 9.1 🔴🔴 核心：toolkit 版本必须与板卡 runtime 的 commit 对齐

用 **toolkit 1.4.0**（SDK `packages/` 里唯一附带的那份）转出的模型，
`rknn_init` 成功、张量属性也读得出来，但**每次推理都在提交阶段被 NPU 拒绝**：

```
E RKNN: failed to submit!, op id: 145, op name: OutputOperator:328, flags: 0x5, ...
```

**症状极具误导性**——不报错退出，而是每次超时 ~6.4 s 后返回，输出缓冲区**从未被填充**：

| 指标 | 真实值 | 会被误读成 |
|:---|---:|---|
| 平均单次推理 | 6395 ms | 「模型慢」 |
| 检出分数 | 全部 ≈ 0.9976 | 「模型很自信」 |
| 检出框 | 退化、坐标乱 | 「框算错了」 |

⇒ **教训：「输出非零 + 看起来有结构」不构成模型可用的证据。**
必须同时看**单次耗时量级**：6395 ms vs 正常 23 ms 本身就是最强异常信号。

换 **toolkit 2.3.0**（PyPI，与板卡 runtime **同 commit `c949ad889d`**）后一次通过：

```
I librknnc version: 2.3.0 (c949ad889d@2024-11-07T11:39:30)
```

**⇒ 铁律：任何 RKNN 转换，第一件事是核对 toolkit 与板卡 `librknnrt` 的 commit hash。**

### 9.2 源 ONNX 与官方模型不是同一个

| | `examples/onnx/yolov5/yolov5s.onnx` | demo 自带 `yolov5s-640-640.rknn` |
|:---|:---|:---|
| 输入 | `[1,3,640,640]` float32 NCHW | `[1,640,640,3]` uint8 NHWC |
| 输出 | **4 维** `[1,255,80,80]` 等 | **5 维** `[1,3,85,80,80]` 等 |

255 = 3 anchors × 85。**两种布局在解码索引上完全等价**（`3*85 == 255`），
故官方 `postprocess.cc` 不改一行即可吃下新模型；
`rknn_npu_smoke.cpp` 已改为**从末尾两维推导网格 + 断言形状**（原硬断言 5 维），回归已验证。

### 9.3 装环境踩到的坑（均实测）

| # | 现象 | 根因 | 解法 |
|:---|:---|:---|:---|
| 1 | `InvalidVersion: '1.4.0-22dcfef4'` | SDK wheel 把 git hash 当 **pre-release** 段，PEP 440 只允许 `+local` | `fix_wheel_version.py` 重写 METADATA + dist-info 目录名 |
| 2 | `No module named torch` | toolkit 2.x 的 `rknn_utils.py` **确实 import torch**，绕不开 | 装 **CPU-only**：`--index-url .../whl/cpu` |
| 3 | pip 卡在 `nvidia_cuda_cupti_cu12` | 默认索引拉整套 CUDA，而**本机无 GPU** | CPU index 或 `--no-deps` 后补 ONNX 路径依赖 |
| 4 | `cannot enable executable stack` | onnxruntime 的 `.so` 带 `PT_GNU_STACK` **RWX** | `clear_execstack.py` 清 `PF_X`（免 patchelf） |

另：**onnx（写）与 onnxruntime（读）必须同时覆盖对方的版本**。
1.4.0 那条路上 onnxoptimizer 提到 IR 9 而 ORT 1.12 只到 8 ⇒ 降级 onnx 1.9.0 + ORT 1.10.0；
2.3.0 这条路上提到 opset 19 而 ORT 1.12 只保证到 17 ⇒ 保持 onnx 1.14.1，只把 **ORT 升到 1.17.3**。

### 9.4 对账结果

| 类别 A→B | IoU | A (l,t,r,b) | B (l,t,r,b) | 分数 A/B | 最大角点偏差 |
|:---|---:|:---|:---|:---|---:|
| person → person | 0.850 | (114,235,212,527) | (111,237,204,540) | 0.819/0.783 | 13 px |
| person → person | 0.938 | (210,242,284,509) | (211,246,283,515) | 0.815/0.810 | 6 px |
| person → person | 0.901 | (479,235,561,520) | (472,233,560,521) | 0.790/0.810 | 7 px |
| **bus → bus** | **0.905** | (99,141,557,445) | (87,134,559,460) | 0.693/0.769 | 15 px |
| person → person | 0.915 | (78,338,122,520) | (79,333,121,517) | 0.405/0.460 | 5 px |

**残差已归因**（不是「差不多就行」）：①源 ONNX 不同（基线来自 `yolov5s_rm_transpose.onnx`）；
②量化标定不同（bus 置信度 0.693→0.769 的主因）；③`rknn_run` 23.94 vs 24.20 ms 基本持平，
差距在 `outputs_get`（1.72 vs 0.47 ms，4 维 NCHW 输出需驱动做布局转换）。

### 9.5 🔴 标定集规模的实测结论（不是猜的）

第 9.4 节的验证只用 `bus.jpg`，而它同时也是标定图 —— **那张结论本身可能被标定泄漏污染**。
为排除这一点，从 COCO `val2017` 随机抽 **120 张标定 + 10 张留出**（互不相交，随机种子 20261013）
重新量化出第三个模型，三方对比：

| 模型 | 标定集 | 检出 | 与官方匹配 | 平均 IoU | 同类别 |
|:---|:---|---:|---:|---:|---:|
| `yolov5s-640-640`（官方） | 未知 | 17 | — | — | — |
| `yolov5s_rknn230` | **单张 bus.jpg** | 21 | 16/17 (94.1%) | 0.810 | 14/17 (82.4%) |
| `yolov5s_rknn230_coco120` | **120 张 COCO** | 21 | 16/17 (94.1%) | 0.798 | 14/17 (82.4%) |

**标定集从 1 张扩到 120 张，对「与官方模型的一致度」几乎没有改善**
⇒ **与官方模型的差异主要来自源 ONNX 不同**（`yolov5s.onnx` vs `yolov5s_rm_transpose.onnx`），
**不是标定质量**，靠调标定**抹不平**。

### 9.6 🔴🔴 扩到 300 张后，9.5 的两个结论都被推翻（本节最重要）

9.5 自己写了「10 张不足以支撑结论」，于是用 **300 张留出图**（与 120 张标定互不相交）重跑。
参考模型共 **1369 个检出**：

| 候选模型 | 标定集 | 检出数 | 参考框匹配率 (IoU≥0.5) | 95% CI | 平均 IoU | 匹配框同类率 |
|:---|:---|---:|---:|:---|---:|---:|
| `yolov5s_rknn230` | 单张 bus.jpg | 1622 | **1157/1369 = 84.5%** | [82.5, 86.3] | 0.862 | 94.7% |
| `yolov5s_rknn230_coco120` | 120 张 COCO | 1629 | **1151/1369 = 84.1%** | [82.0, 85.9] | 0.854 | 94.5% |

**🔴 更正一：10 张图的「94.1% 匹配」是偏乐观估计，真实值约 84.5%** ——
小样本把比例统计抬高了约 **10 个百分点**。

**🔴 更正二：「所有分歧都集中在决策边界」在小样本上成立，在 1369 框上不成立。**
分层后同类率相当均匀（按分数 94.5 / 96.6 / 98.9 / 100.0%），
**置信度只带来约 5 个百分点提升，不存在干净的边界效应**；
按尺寸看大框（>150 px）反而略低（93.3%），与小框假设相反。
**17 个检出时看着像「边界效应」，1369 个检出时只是回归到均值。**

**✅ 存活并被加强：标定集大小不影响与官方模型的一致度。**
84.5% vs 84.1%，差异远小于置信区间半宽 ±1.9%。

**🔴 新发现：两个重量化模型都系统性多检出约 18–19%**（1622 / 1629 vs 1369），
多出的 465 / 478 框是候选模型独有的 —— **不是噪声，是一致的敏感性差异**。

⚠️ 上述全部指标都是**两模型之间的一致性**，**不是精度**。
要判定「哪个更准」仍需人工标注 GT 做 P/R 评测。

### 9.7 ⚠️ 标定集仍必须换（理由与 9.5 初稿不同）

用 COCO 自然图标定出的量化参数**不代表本项目真正要部署的红外弱小无人机域**。
（至于「扩大标定集能否缩小与官方模型的差距」，实测答案是**不能**，原因见 9.5。）

### 9.8 环境位置（不落根分区）

`2026-10-13` 实测根分区 `/dev/sda1` 仅剩 15 GB / 93% 已用（**2026-10-14 NFS 交换区换址到 SSD960-LOCAL 大盘后该红线已解除**，`782G` 可用），conda 默认在根分区，故两个环境都建在 1 TB 卷：

```
/media/manu/1TB-Volume/conda-envs/rknn       # toolkit 1.4.0（产出被 NPU 拒绝，留作证据）
/media/manu/1TB-Volume/conda-envs/rknn230    # toolkit 2.3.0（可用）
/media/manu/1TB-Volume/rknn_work/            # wheels / models / 中间产物
```

### 9.7 对 Trial 0474 的意义与限制

「ONNX → RKNN → 板卡推理」的闭环已在本机跑通，不再依赖 Rockchip 现成模型。
但 **Trial 0474 是三通道 heatmap 检测器**（输出 `[H,W,1]` 类热图，不是 yolov5 的 85 通道头），
**后处理不能照搬 yolov5 解码**，只能复用「转换 + 部署 + 计时」这一段。

## 十、存储分工（本轮实际用量）

| 内容 | 位置 | 体积 | 理由 |
|:---|:---|---:|---|
| 可执行文件 | 板卡本地 `/root/rknn_npu_smoke` | 148 KB | 经 NFS 执行会 `Text file busy`（旧方向实测） |
| 模型 `.rknn` | `/mnt/manu/npu/data/`（NFS=host） | 7.9 MB | 不占板卡盘 |
| 测试图 `.jpg` | `/mnt/manu/npu/data/`（NFS=host） | 181 KB | 同上 |
| 链接用 `librknnrt.so` | host `/run/media/manu/SSD960-LOCAL/nfs/npu/` | 7.3 MB | 只在 host 侧用，**不推板卡** |

板卡本地盘新增 **仅 148 KB**（`/` 仍有 ~1.7 GB 可用），符合「板卡空间有限，数据一律走 NFS」。

## 十一、【2026-10-13 追加】Trial 0474 热图移植时本文件两条教训的复发与补充

> 本节为**追加**，上文一字未改。完整记录见专档 `manu/memory/heatmap_npu_trial0474.md`，
> 操作手册见 `manu/pipeline/npu/CONVERT_HM.md`。

**1. 🔴「非零输出即成功」再次复发，但这次不是版本问题。**
本文件的 2d 教训是「用错版本转出的模型会输出退化结果却从不报错」。热图这轮遇到的是**另一种形状的同一病**：

| 闸门 | 本轮做法 | 为什么必须有 |
|:---|:---|:---|
| **空白输入拒答** | 全零输入下模型**拒绝给出**结果 | 恒真的输出通路与「模型可用」在数值上不可区分 |
| **空对照** | 输入取反 ⇒ **两个输出张量的哈希都改变** | 输出与输入无关的模型，也能输出「结构完整」的结果 |
| **确定性** | 两次运行**逐字节一致** | 非确定输出无法作为证据引用 |

**2. 🔴「toolkit 版本必须与板卡 runtime 对齐」再次复发，且这次版本明明是对齐的。**
本文件 §9.1 / CONVERT.md 第四节记的是 toolkit **1.4.0** 产出被拒。本轮 **toolkit 2.3.0 与 runtime 2.3.0
（同 commit `c949ad889d`）完全对齐**，`rknn_init` 成功、张量属性也读得出来，
**但同一份 ONNX 的 fp16 构建（6,739,809 B）在第一次推理就提交失败**：

```
failed to submit!, op id: 8, op name: Conv:/model/p0_highway/p0_dw/Conv, flags: 0x1, ...
```

**int8 构建（4,093,832 B）运行完全正常。** ⇒ **版本对齐是必要条件、不是充分条件**：
**每个构建产物都要自己上过板**（新增 `rules.md` §四·2f）。
⚠️ **报错文本相似 ≠ 成因相同**：这与 §9.1 的 `OutputOperator` 是两次不同故障，归因前必须先比 op name / flags / 版本。

**3. 时延结构对比：同一个板子、同一个 runtime，两条链路的瓶颈完全不同。**

| 链路 | `rknn_run` | 预处理 | 形态 |
|:---|---:|---:|:---|
| **yolov5（本文件第十节）** | 24.4 ms（占 97%），端到端 25.1 ms / 39.8 fps | **约 13 ms** | I/O + 预处理是主要成本 |
| **Trial 0474 热图（本轮）** | **151.341 ms（占 99.3%）**，端到端 **152.395 ms / 6.6 fps** | letterbox+解码在二进制内 **0.46–0.51 ms** | **纯算力主导** |

⇒ **两条链路的时延结论不可互相外推**；评估新模型时先测瓶颈占比，再决定优化对象
（与 `rules.md` §四·9c「优化前先实测瓶颈占比」同源）。

**4. ⚠️ NPU 节点提醒（再次强调，本轮实测复核）**：驱动 RKNPU **0.9.8** 以 **DRM 子系统**注册，
真实节点是 **`/dev/dri/card1`**，**不是 `/dev/rknpu`**；
`/proc/devices`、`lsmod`、`ls /dev/rknpu` 三条查法给出的仍是**假阴性**，
**看到它们「查不到」时不要据此宣布 NPU 不可用**。