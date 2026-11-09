# 标准 YOLO26n 在 RK3588 NPU 上的基准（本项目 NPU 结构优化的参照点）

> **产出日期**：2026-11-05（晚间）
> **执行机**：host（导出/转换）= 本机 `/media/manu/1TB-Volume/workspace/ultralytics`；
> **板卡** = `root@192.168.0.64:22`（RK3588，板载 librknnrt 2.3.0，驱动 0.9.8）。
> **工具链**：`/media/manu/1TB-Volume/conda-envs/rknn230`（rknn-toolkit2 2.3.0，onnx 1.14.1，torch 2.4.1 CPU）。
> **授权口径**：用户在本轮对话明确要求「看看在板卡上的 npu 推理时间是多少」，故本轮的板卡跑分属**逐次授权**范围（铁律一）；
> 未跑任何训练/评测/缓存提取脚本。

---

## 一、这一节为什么存在

本项目的检测器是 `YOLO26HeatmapDetector`（YOLO26 + P2 Neck + **320×320 热图头**，21.48 G MAC，1.79 M 参数），
它在板卡上的 `rknn_run` 是 **55.501 ms**（单尺度探针）/ **108.202 ms**（mosaic e2e 口径）。

但**「标准 YOLO26n」本身的 NPU 时间从来没有被测过**。没有这个参照点，任何结构改动都只能靠"感觉应该更快"来判断：
到底是网络结构的问题，还是**算子布局 / 搬运 / 量化 / 调度**的问题？这条基准就是为了把这个问题变成可测量的。

---

## 二、被测对象：必须是「标准的」官方 YOLO26n

| 项 | 值 | 证据 |
|:--|:--|:--|
| 权重 | Ultralytics 官方 `yolo26n.pt`，5,544,453 B，**md5 `cf3cca69f04cf639bafdeb2644bd0843`** | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt` |
| 结构 | `scale=n`，`end2end=True`，`reg_max=1`，`nc=80`，**参数 2,572,280** | `ultralytics/cfg/models/26/yolo26.yaml`；实测加载确认 |
| 拓扑 | **P3/8 – P4/16 – P5/32**（`yolo26.yaml`），非本项目用的 P2 变体（`yolo26-p2.yaml`） | 同上 |
| 规模 | 6.1 GFLOPs ≈ **3.05 G MAC** @640×640 | yaml 注释 |

⚠️ 本项目 `heatmap_model.py` 里的 `YOLO26HeatmapDetector` 与官方 `yolo26n.yaml` **不是同一个网络**：
本项目多了 **P2 支路（stride 4，160×160）** 与 **stride 2 的 P1 头（320×320 输出）**。
**"标准 YOLO26n" 的参照点必须是官方权重 + 官方 P3-P5 拓扑**，否则测的是自定义结构，不是参照系。

---

## 三、导出与量化：严格走 Ultralytics 自己的 RKNN 规则

### 3.1 关键决策：关掉 `end2end` 分支（不是自己拍脑袋）

`ultralytics/engine/exporter.py:681-684`：

```python
if fmt in {"rknn", "ncnn", "executorch", "paddle", "imx", "edgetpu", "qnn"}:
    # Disable end2end branch for certain export formats as they does not support topk
    model.end2end = False
```

⇒ **RKNN 没有 topk，Ultralytics 官方对 RKNN 导出就是关掉 end2end 的**。
⇒ 导出图 = 经典 one2many 检测头：raw `(B, 4+nc, N)`，**解码 + NMS 在 host 侧做**。
⇒ 这也正是 Rockchip SDK `rknn_yolov5_demo` 的接线方式（host 侧 NMS）。

> **副作用（已验证无害）**：`one2one_cv2/cv3` 是 `cv2/cv3` 的 `copy.deepcopy`（`ultralytics/nn/modules/head.py:140-141`），
> 两个分支层数与 MAC 完全相同 ⇒ **关不关 end2end 不改变 NPU 时间口径**。

### 3.2 关键决策：套 `_NormalizeCoords`（RKNN int8 官方路径）

`ultralytics/engine/exporter.py:1096-1104` 对 `rknn + int8 + detect + 非 end2end` 会套
`ultralytics/utils/export/engine.py:17` 的 `_NormalizeCoords`：把 4 个框通道除以输入尺寸。
原因：像素级坐标的动态范围会被 per-tensor int8 压扁，除到 [0,1] 能保住相对精度。
⇒ **板端 C++ 读到的框是归一化的，乘回 640 才是画布像素**。

### 3.3 产物

| 产物 | 路径 | 大小 | md5 |
|:--|:--|--:|:--|
| 官方权重 | `/media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.pt` | 5,544,453 B | `cf3cca69f04cf639bafdeb2644bd0843` |
| ONNX | `/media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.onnx` | 9.81 MB | `da583d351317c1cc8bfa9b03a2203467` |
| RKNN int8 | `/media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n_int8.rknn` | 4,687,888 B | `56e27e5f600f50d8ee0cecc86b188c9e` |
| 构建日志 | `/media/manu/1TB-Volume/rknn_work/yolo26n/logs/build_yolo26n_int8.log` | — | — |

ONNX 契约：`input images (1,3,640,640)` → `output0 (1,84,8400)`；
84 = 4 框 + 80 类；8400 = 6400(P3) + 1600(P4) + 400(P5)（P3 在前）。
`onnx.checker` 通过。

### 3.4 量化配置与校准集

* 复用项目既有转换器 `manu/pipeline/npu/hm_convert_rknn.py`（不改它，只换 `--onnx/--out`），
  配置 = `mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]`、`target_platform=rk3588`、`optimization_level=3`
  ⇒ 图消费 **uint8 RGB**，RKNN 做 `(x-0)/255`。
* 校准集 = **200 张真实红外特征图**（`uav_gmc_median/images/train`，跨 199 个序列，确定性分层抽样）。
  ⚠️ **该清单与项目既有 `hm/calib_train_features.txt` 逐行相同**（已 diff 验证）⇒ 校准口径与热图模型**完全受控一致**。
* ⚠️ **校准数据是 3 通道特征图（I_t / 差分 / 21 帧中位残差），不是 COCO 彩色图**。
  这对本任务**无影响**（量化尺度只影响数值不影响时延），但若日后要论"标准 yolo26n 在 COCO 上的精度"，必须换 COCO 校准集重转。

### 3.5 上板闸门（铁律§四·9e / 9g）

| 检查项 | 结果 |
|:--|:--|
| `fallback` | **0** |
| `unsupport` | **0** |
| `exceed` | **0** |
| `Transpose will fallback to CPU` | **0** |
| CPU 算子数 | **2**（`InputOperator` + `OutputOperator`，即 I/O 边界，非计算算子） |
| Total Internal Memory | 4400 KB |
| Total Weight Memory | 2818.44 KB |

---

## 四、板端实测（本节核心数字）

板端属性（`rknn_query` 自报）：
```
api_version=2.3.0 (c949ad889d@2024-11-07)  driver_version=0.9.8
in : n_dims=4 dims=[1 640 640 3] fmt=1(NHWC) type=2(INT8) zp=-128 scale=0.0039215689 (=1/255) size=1228800
out: n_dims=3 dims=[1 84 8400]    fmt=3       type=2(INT8) zp=-128 scale=0.0043071685 n_elems=705600
```

### 4.1 主结果（`bus.jpg` 640×640，warmup 10 / loops 30，core_mask=AUTO）

| 阶段 | avg | median | p95 | worst |
|:--|--:|--:|--:|--:|
| `inputs_set` | **0.215 ms** | | | |
| **`rknn_run`** | **23.658 ms** | 23.728 | 24.397 | 24.523 |
| `outputs_get` | 0.572 ms | | | |
| **NPU 合计** | **24.445 ms ⇒ 40.91 fps** | | | |
| 仅 `rknn_run` | ⇒ **42.27 fps** | | | |

### 4.2 ⭐ `pass_through=1` 是一个**巨大**的、免费的赢面

| 输入递交方式 | `inputs_set` | `rknn_run` | NPU 合计 | 输出 hash |
|:--|--:|--:|--:|:--|
| `pass_through=0`（runtime 自己重量化 + 转置 1.2 MB） | **15.054 ms** | 25.135 ms | 40.751 ms（24.54 fps） | `37e3649f7b3cd889` |
| `pass_through=1`（host 直接把已量化码交给 NPU） | **0.213 ms** | 23.950 ms | **24.734 ms（40.43 fps）** | `37e3649f7b3cd889` |

⇒ **`inputs_set` 降 15.054 → 0.213 ms（71×）**，且**输出 hash 逐字节相同** ⇒ 数值上完全等价。
⇒ 板端例程默认已设为 `pass_through=1`。

### 4.3 与本项目热图模型的对照（同一块板、同一 runtime）

| 模型 | 拓扑 | 规模 | `rknn_run` | 端到端 fps | 倍数 |
|:--|:--|:--|--:|--:|--:|
| **标准 YOLO26n（本节）** | P3-P5，输出 80×80 | 6.1 GFLOPs ≈ 3.05 G MAC | **23.658 ms** | 40.91 fps（NPU 合计） | **1.00×** |
| `trial_0474` 热图（单尺度探针口径） | P2 + P1 头，输出 320×320 | 21.48 G MAC | 55.501 ms | 17.7 fps | **2.35× 慢** |
| `trial_0474` 热图（mosaic e2e 口径） | 同上 + 拼图三尺度 | 21.48 G MAC | 108.202 ms | 12.507 fps | **4.58× 慢** |

⇒ **同一个底座下，"把大分辨率热图头换成标准 P3-P5 检测头" 换来 2.35×（同口径）/ 4.58×（e2e 口径）的 NPU 提速。**
⇒ ⚠️ 但 **6.1 GFLOPs 只跑出 ~258 GOPS**（6.1 GOP ÷ 23.658 ms），相对 RK3588 三核 NPU 的标称 int8 吞吐（~6 TOPS）只有 **~4%**。
⇒ **这个 4% 是本节最重要的未结问题**：要么是板卡/驱动的天花板，要么是这张图的算子排布不友好。**必须用参照点裁决，不能靠猜。**

### 4.4 正确性证据（不是"看着像"）

| 检查 | 结果 |
|:--|:--|
| 确定性（同输入两次输出） | **PASS**（hash `d1c3ff787724cc61` = `d1c3ff787724cc61`） |
| 反相图必须改变输出 | **PASS**（`d1c3ff787724cc61` → `3413af5f3e0842b7`） |
| 输出字节数自洽 | 705,600 = 84 × 8400 ✓ |
| host 参照（onnxruntime，同 letterbox + 同解码 + 同 NMS） | `bus.jpg` → **cls=5(bus) 0.8971** + 4×cls=0(person) |
| 板端 int8 同图 | **5 个检出**（与 host 同数量） |

host 参照脚本：`manu/pipeline/npu/yolo26n_host_ref.py`（可 `--compare board.tsv` 逐框对账）。

---

## 五、🔴 踩坑记录（每一条都花过时间，且都与"结构优化"直接相关）

### 5.1 🔴 框通道是 **(cx, cy, w, h)**，不是 xyxy

第一版板端后处理按 xyxy 读 ⇒ 出现 `x1 > x2` 的倒挂框。
根因：`Detect.decode_bboxes(..., xywh: bool = True)`（`ultralytics/nn/modules/head.py:227`）**默认就是 xywh**，
`_get_decode_boxes`（`head.py:210`）调用时没覆盖 ⇒ `dist2bbox` 返回 **中心 + 尺寸**。
⇒ 导出图的 4 个框通道是 `(cx, cy, w, h)`（画布像素），`_NormalizeCoords` 再除以 640。
⇒ **正确解码**：`x1 = cx - w/2`，`x2 = cx + w/2`（y 同理）。
⚠️ 这类错误在红外图上**完全看不出来**（本来就没检出）⇒ 必须在 COCO 图上做 host↔板端对账。

### 5.2 🔴 `--layout nchw` 会让 runtime **core dump**

runtime 对该模型只接受 **NHWC** 宿主缓冲（`in_attr.fmt = RKNN_TENSOR_NHWC`）；
声明 NCHW ⇒ 直接 `Bus error (core dumped)`。
⇒ **本模型只能用 NHWC**；要省搬运就靠 `pass_through=1`（见 4.2），不要动 layout。

### 5.3 🔴 `rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2)` **把板卡驱动搞挂了**

```c
rknn_set_core_mask(ctx, 7);   // 返回 0（成功）……然后 6 秒后：
```
```
RKNPU: job: ..., mask: 0x7, commit elapse time: 6096934us, timeout: 6000000us
RKNPU: failed to wait job ...
RKNPU: job timeout, flags: 0x0
RKNPU: soft reset, num: 6
```
⇒ 之后**每一次** `rknn_init` 都是 **exit 132（SIGILL，Illegal instruction）**，板卡必须**物理断电重启**才能恢复。
⇒ 这与 `falsified_archive.md` §49.3 记载的运维事实**完全一致**（"板卡上的 e2e 调用次数是稀缺资源"）。
⇒ **结论：这块板不要在运行时切 core mask。** 想要多核只能在**构建期**想办法
（但 rknn-toolkit2 2.3.0 的 Python API 里 `core_mask` 只属于 `init_runtime()`，**没有构建期 core_mask**）。

### 5.4 ⚠️ `build()` 没有 `core_mask` 参数

`rknn.api.RKNN.build(self, do_quantization=True, dataset=None, rknn_batch_size=None)`；
`core_mask` 在 `init_runtime(...)` 里，而 `init_runtime` 只在"host 直连设备"时才有意义。
⇒ 想换核只能走 `config(single_core_mode=...)`（默认 False = 多核 capable）。

### 5.5 ⚠️ int8 输入缓冲装的是**码**，不是值

`scale = 1/255, zp = -128` ⇒ 宿主字节 `x` 的码是 `x + zp`（即 `x - 128`）。
**逐字节照抄是错的**（`0x80` 会被解成 −128 而不是 0）⇒ 板端例程用通用仿射式
`code = round(x / 255 / scale) + zp` 计算。

### 5.6 ⚠️ SDK 里**没有 yolov8 例子**

`external/rknpu2/examples/` 下只有 `rknn_yolov5_demo` / `rknn_mobilenet_demo` / `rknn_api_demo` 等，**无 yolov8**。
⇒ 本项目的"参考 SDK 的 yolov8 例子"落到 **`rknn_yolov5_demo`** 上（API 接线方式完全一致）。
⚠️ 且该 demo **拉伸不 letterbox**（用 RGA 改长宽比）⇒ 检测任务必须自己写 letterbox。

---

## 六、可复制命令（铁律九）

### 6.1 导出 ONNX

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics
export PYTHONPATH=/media/manu/1TB-Volume/workspace/ultralytics
/media/manu/1TB-Volume/conda-envs/rknn230/bin/python manu/pipeline/npu/yolo26n_export_onnx.py \
  --weights /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.pt \
  --out     /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.onnx \
  --imgsz 640 --opset 12
```

### 6.2 转 RKNN int8（复用项目既有转换器）

```bash
/media/manu/1TB-Volume/conda-envs/rknn230/bin/python manu/pipeline/npu/hm_convert_rknn.py \
  --onnx /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.onnx \
  --out  /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n_int8.rknn \
  --calib-split train --calib-n 200 \
  --calib-list /media/manu/1TB-Volume/rknn_work/yolo26n/calib_train_features.txt \
  2>&1 | tee /media/manu/1TB-Volume/rknn_work/yolo26n/logs/build_yolo26n_int8.log

# 🔴 上板闸门（必须重定向日志，否则 grep 空文件会静默判过）
test -s /media/manu/1TB-Volume/rknn_work/yolo26n/logs/build_yolo26n_int8.log || { echo "❌ 日志缺失"; exit 1; }
grep -ic 'fallback'  /media/manu/1TB-Volume/rknn_work/yolo26n/logs/build_yolo26n_int8.log   # 期望 0
grep -ic 'unsupport' /media/manu/1TB-Volume/rknn_work/yolo26n/logs/build_yolo26n_int8.log   # 期望 0
```

### 6.3 构建尖刺 A/B 臂（core/sram/compress 旋钮）

```bash
/media/manu/1TB-Volume/conda-envs/rknn230/bin/python manu/pipeline/npu/yolo26n_speed_ab.py \
  --onnx /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.onnx \
  --outdir /media/manu/1TB-Volume/rknn_work/yolo26n/arms \
  --calib-list /media/manu/1TB-Volume/rknn_work/yolo26n/calib_train_features.txt
# 产物：arms/yolo26n_{baseline,sram,compress,sram_compress,nbuf_high,sram_nbuf}.rknn + arms/manifest.json
```

### 6.4 交叉编译板端例程

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu
bash build_rknn_yolo26n_arm64.sh
# 产物：out/rknn_yolo26n_smoke（aarch64，DT_NEEDED 必须是纯 librknnrt.so）
```

### 6.5 部署 + 跑分

```bash
# 二进制走 SSH 落板卡本地盘（禁 NFS，否则 ETXTBSY / SIGILL）
scp -P 22 out/rknn_yolo26n_smoke root@192.168.0.64:/root/
ssh -p 22 root@192.168.0.64 'md5sum /root/rknn_yolo26n_smoke'   # 与本机 md5 比对

# 模型走 NFS：host /run/media/manu/SSD960-LOCAL/nfs/npu/ == 板卡 /mnt/manu/npu/
cp -f /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n_int8.rknn \
      /run/media/manu/SSD960-LOCAL/nfs/npu/data/yolo26n_int8.rknn

ssh -p 22 root@192.168.0.64 \
 'cd /mnt/manu/npu && LD_LIBRARY_PATH=/usr/lib timeout 200 /root/rknn_yolo26n_smoke \
    /mnt/manu/npu/data/yolo26n_int8.rknn /mnt/manu/npu/data/bus.jpg \
    --loops 30 --warmup 10 --th 0.25 --pass-through 1 \
    --detections-out /mnt/manu/npu/out/yolo26n_bus.tsv --osd /mnt/manu/npu/out/yolo26n_bus.jpg'
```

### 6.6 host ↔ 板端对账

```bash
/media/manu/1TB-Volume/conda-envs/rknn230/bin/python manu/pipeline/npu/yolo26n_host_ref.py \
  --onnx /media/manu/1TB-Volume/rknn_work/yolo26n/yolo26n.onnx \
  --image /run/media/manu/SSD960-LOCAL/nfs/npu/data/bus.jpg --th 0.25 \
  --compare /run/media/manu/SSD960-LOCAL/nfs/npu/out/yolo26n_bus.tsv
```

---

## 七、产物与脚本清单

| 类别 | 路径 |
|:--|:--|
| 导出脚本（新增） | `manu/pipeline/npu/yolo26n_export_onnx.py` |
| 尖刺 A/B 脚本（新增） | `manu/pipeline/npu/yolo26n_speed_ab.py` |
| host 参照脚本（新增） | `manu/pipeline/npu/yolo26n_host_ref.py` |
| 板端例程（新增） | `manu/pipeline/npu/rknn_yolo26n_smoke.cpp` |
| 板端例程构建脚本（新增） | `manu/pipeline/npu/build_rknn_yolo26n_arm64.sh` |
| 转换器（复用，未改） | `manu/pipeline/npu/hm_convert_rknn.py` |
| 模型 | `/media/manu/1TB-Volume/rknn_work/yolo26n/{yolo26n.pt,yolo26n.onnx,yolo26n_int8.rknn}` |
| A/B 臂 | `/media/manu/1TB-Volume/rknn_work/yolo26n/arms/` + `manifest.json` |
| 板卡侧模型 | `/mnt/manu/npu/data/yolo26n_int8.rknn`（host: `/run/media/manu/SSD960-LOCAL/nfs/npu/data/`） |
| 板卡侧二进制 | `/root/rknn_yolo26n_smoke`（md5 `830a5c97d7a43dee94408bc94886e40d`） |

---

## 八、未结问题（下一轮的第一优先级）

1. **~4% NPU 利用率**：23.658 ms 跑 6.1 GFLOPs ⇒ ~258 GOPS，相对标称 ~6 TOPS 只有 ~4%。
   **必须用 Rockchip 官方 demo（`rknn_yolov5_demo` / `rknn_mobilenet_demo`）在同一块板上取参照点**，才能判定：
   是"板卡/驱动天花板"还是"这张图排布不友好"。⚠️ 该参照测量因 §5.3 的驱动挂死而**未完成**，需**板卡物理断电重启后**补做。
2. **`rknn_run` 里到底装了什么**：23.658 ms 中，卷积本身占多少？量化 requantize 的 `Add/Sub/Mul` 算子占多少？
   `RKNN_QUERY_PERF_DETAIL` 可以逐层归因（但 `--perf` 标志会**显著变慢**，需要另开一组对照）。
3. **尖刺 A/B 臂待测**：`sram_prefer` / `compress_weight` / `nbuf_prefer` 是否能把这 23.658 ms 压下来（臂已建好，待板卡恢复）。
4. **口径提醒**：本节的 23.658 ms 是**单图单次**口径；本项目 mosaic e2e 口径含拼图/多尺度，**两者不可直接相减**。

---

## 九、⭐⭐⭐ 板卡 NPU 时延的**成本模型**（本节最有工程价值的产出）

### 9.1 数据来源（全部可复核）

* **逐层 NPU 周期 / DDR 周期 / 读写字节（RW）**：RKNN 构建日志 `Network Layer Information Table` 里
  `Cycles(DDR/NPU/Total)` 与 `RW(KB)` 两列**逐层累加**。
  - 标准 YOLO26n：`/media/manu/1TB-Volume/rknn_work/yolo26n/logs/build_yolo26n_int8.log`
  - 本项目热图模型 6 个 head 变体：`/media/manu/1TB-Volume/rknn_work/hm/merge_head/build_{base,dw,late,late32,latedw,latedw32}_int8.log`
* **实测 `rknn_run`（int8，640×640，单尺度探针口径）**：`archive/02_流水线与量化主线_§二十~§三十五.md` §二十九~§三十二
  （base 55.501 / dw 49.860 / late 36.657 / late32 34.089 / latedw 35.255 / latedw32 34.302 ms）＋ 本节 yolo26n 23.658 ms。

### 9.2 逐层归因（标准 YOLO26n，182 个算子，CPU 算子 2 个）

| 算子类型 | 个数 | NPU 周期 | 占 NPU 周期 | DDR 周期 |
|:--|--:|--:|--:|--:|
| `ConvExSwish` | 87 | 4,978,800 | **95.67%** | 1,730,296 |
| `Conv` | 11 | 187,200 | 3.60% | 101,834 |
| `ConvAdd` | 4 | 38,400 | 0.74% | 34,634 |
| `Concat`/`Split`/`Add`/`Reshape`/`MaxPool`/`Resize`/`Sigmoid`/`SDPAttention` 等 | 78 | 0 | 0 | 0 |

* **NPU 周期合计 = 5,204,400** ⇒ 板卡 NPU 跑满 1 GHz 时**纯算力只需 5.20 ms**。
* 最贵的单层：**Stem 卷积（3→16，s2，640×640 输入）= 921,600 周期（占 17.7%）**、P2 的 3×3（16→32@160×160）= 230,400。
* `RW` 合计 = **44.8 MB**（对比热图 base 的 113.5 MB、dw 的 144.6 MB）。

### 9.3 ⭐ 拟合出的时延模型（7 个数据点，全部来自实测）

对 7 个模型（1 个标准 YOLO26n + 6 个本项目热图变体）做最小二乘：

```
rknn_run(ms) ≈ 13.94  +  0.902 × NPU周期(百万)  +  0.143 × RW(MB)
```

| 模型 | NPU 周期(百万) | RW(MB) | 预测(ms) | 实测(ms) | 残差 |
|:--|--:|--:|--:|--:|--:|
| **标准 YOLO26n** | 5.20 | 44.8 | 25.03 | **23.658** | +1.37 |
| 热图 base | 28.25 | 113.5 | 55.63 | 55.501 | +0.13 |
| 热图 dw | 17.60 | 144.6 | 50.47 | 49.860 | +0.61 |
| 热图 late | 12.43 | 77.6 | 36.23 | 36.657 | −0.42 |
| 热图 late32 | 9.90 | 73.6 | 33.38 | 34.089 | −0.71 |
| 热图 latedw | 9.77 | 85.2 | 34.92 | 35.255 | −0.34 |
| 热图 latedw32 | 9.36 | 79.0 | 33.66 | 34.302 | −0.64 |

⇒ **7/7 点残差 ≤ 1.37 ms**（6 个热图变体 ≤ 0.72 ms）。

**三项的工程含义**：

1. **~13.9 ms 的固定单次推理开销**（runtime/driver/dispatch/同步）。
   ⇒ **这块板上任何单帧 NPU 推理的下限就是 ~14 ms**；"标准 YOLO26n 只要 3~6 ms"在**这块板**上**不可能**（哪怕算力为 0）。
2. **0.902 ms / 百万 NPU 周期**（≈ 1 MAC 阵列周期 ≈ 1.11 GHz 等效）。
3. **0.143 ms / MB 读写流量**（带宽税）。

### 9.4 ⭐ 对"网络结构优化"这条线的直接改写

用该模型拆解**本项目热图模型 base 的 55.501 ms**：

| 项 | 值 | 占比 | 能不能靠改网络压掉 |
|:--|--:|--:|:--|
| 固定开销 | 13.94 ms | 25% | ❌ 不能（要靠流水/双缓冲/换调用方式） |
| 算力 | 28.25 Mcycle × 0.902 = **25.5 ms** | 46% | ✅ 能（砍 MAC / 砍通道 / 砍层数） |
| 带宽 | 113.5 MB × 0.143 = **16.2 ms** | 29% | ✅ 能（**砍大张量 = 降 RW**，这正是 `late`/`late32` 的作用） |

⇒ **本项目的"搬运量才是关键"这条经验法则，第一次有了机制与数量**：
`sram_prefer` / `compress_weight` 这类构建旋钮动的是**带宽项**；
`late` / `late32` / 骨干瘦身动的是**算力项 + 带宽项**；
而**固定开销 13.9 ms 谁都动不了**（它属于调用方式，不属于网络）。
⇒ **可动的最大空间 = 55.5 − 13.9 ≈ 41.6 ms**（这才是结构优化的真实天花板）。

### 9.5 标准 YOLO26n 自身的拆解（23.658 ms）

| 项 | 值 | 占比 |
|:--|--:|--:|
| 固定开销 | ~13.9 ms | **59%** |
| 算力 | 5.20 Mcycle × 0.902 = **4.7 ms** | 20% |
| 带宽 | 44.8 MB × 0.143 = **6.4 ms** | 27% |

⇒ **即使网络再缩一半，yolo26n 也只能到 ~15 ms**（固定开销 + 算力/带宽减半）。
⇒ **该板的"每帧 NPU 时间"存在 ~14 ms 硬地板** ⇒ 任何"再快一点"的目标必须先回答"能不能绕开每次 rknn_run 的固定开销"
（多路并发、异步 `RKNN_FLAG_ASYNC_MASK`、把固定开销摊到流水线里）。

### 9.6 ⚠️ 本模型的可信度边界（必须随数字一起引用）

* **样本只有 7 个**，其中 **6 个属于同一模型族**（同骨干、只换 head）⇒ 拟合能很好地描述"**换 head**"的方向，
  对"换整个网络"的外推力较弱（yolo26n 那个点残差最大 +1.37 ms，正是这一点）。
* **固定项的构成未分解**：13.9 ms 里有多少是 runtime/driver 的 job 提交开销、多少是 182 个算子的 dispatch 开销、
  多少是同步等待，**尚未区分**。区分它需要板端逐层 perf dump（`RKNN_QUERY_PERF_DETAIL`），
  ⚠️ 而 `--perf` 标志本身会显著变慢，**必须另开一组对照**，不能直接拿带 perf 的数字当时延。
* **口径**：本模型只覆盖**单尺度 640×640、int8、单图单次**；
  mosaic e2e 的 108.202 ms **不在拟合集内**（拼图三尺度是另一种输入排布）。
* 复算脚本（本节表格可一键重算）：
  `python3 - <<'EOF'` 见本节 9.1 的解析正则；逐层解析用
  `re.compile(r'^D RKNN: \[[0-9:.]+\]\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+.*?\s(\d+)/(\d+)/(\d+)\s+(\d+)\s+(\S+)\s*$')`。

---

## 十、⭐⭐⭐ 参照点裁决：**这块板的 NPU 真实吞吐是 ~256–282 GOPS**（用 Rockchip 官方 demo 对照）

### 10.1 一次打包会话测完（2026-11-05 17:02）

入口：`manu/pipeline/npu/run_yolo26n_ref_board.sh`（宿主驱动，自动构建/暂存/部署/md5 校验/单次板端会话/回读制表）
配套：`manu/pipeline/npu/print_yolo26n_ref_table.sh`（回读 + 对比表，支持 `--raw`）

板端会话脚本：`manu/pipeline/npu/out/measure_yolo26n_ref_board.sh`（由宿主脚本生成后部署）
结果原始文件：`<NFS>/yolo26n_ref/ref_results.txt`（host: `/run/media/manu/SSD960-LOCAL/nfs/npu/yolo26n_ref/ref_results.txt`）

| 模型 | 来源 | ms/帧 | GFLOPs | 隐含 GOPS |
|:--|:--|--:|--:|--:|
| **mobilenet_v1**（224×224） | SDK 自带模型 ＋ `rknn_bench_generic` | **2.020** | 0.57 | **282** |
| **yolov5s_rknn230**（640×640） | **Rockchip 自带 `rknn_yolov5_demo`** | **26.311** | 7.2 | **274** |
| **yolo26n_int8_stock**（640×640） | 本项目 `rknn_yolo26n_smoke` | **23.805** | 6.1 | **256** |

⇒ **三个独立参照点落在 256–282 GOPS**，而 RK3588 标称 int8 是 **~6 TOPS** ⇒ **本板有效利用率只有 ~4.5%**。
⇒ ⭐ **本项目的标准 YOLO26n（23.805 ms）与 Rockchip 官方 demo 的 YOLOv5s（26.311 ms）完全同量级**
⇒ **"标准 YOLO26n 应当只要 3~6 ms"在这块板上不成立**：那需要 1000–2000 GOPS，本板做不到。

### 10.2 构建旋钮 A/B：**全都不是杠杆**（实测，同一块板同一会话）

| 臂 | 构建配置 | `rknn_run` | 相对 baseline |
|:--|:--|--:|--:|
| `baseline` | 默认（optimization_level=3） | 23.624 ms | — |
| `sram` | `sram_prefer=True` | **23.404 ms** | **−0.93%** |
| `compress` | `compress_weight=True` | 23.755 ms | +0.55% |
| `sram_compress` | 两者同时 | 23.500 ms | −0.53% |

⇒ **全部落在 ±1% 的run-to-run 噪声内（23.404–23.805 ms，跨度 1.7%）** ⇒
`sram_prefer` / `compress_weight` / `nbuf_prefer` **对本模型没有任何可测收益**。
⇒ 与 §9 的成本模型一致：**这 23.8 ms 里 59% 是固定开销（13.9 ms）+ 20% 算力 + 27% 带宽**，
而这张图**既不缺 SRAM 也不缺权重带宽**（RW 44.8 MB，权重 2.8 MB）⇒ 旋钮无处发力。
⇒ **这条否证结论的中间量**：RW 总量 44.8 MB（base/ dw/ late 对照：113.5 / 144.6 / 77.6 MB）、
Weight Mem 2818.44 KB（`compress_weight` 只降到 2816.5 KB）、
Internal Mem 4400 KB（`sram_prefer` 不改变分配量，只改 operand 落位）。

### 10.3 🔴🔴 **板卡自带 `/usr/lib/librknnrt.so` 是坏的**（重要运维事实）

| 文件 | md5 | 位置 | 行为 |
|:--|:--|:--|:--|
| 板卡自带 | `9cb5f17b971f8cdf00ca539d727e47e9` | `/usr/lib/librknnrt.so`（mtime 2026-02-05） | 🔴 **在 `rknn_run` 内 SIGILL（exit 132）** |
| 项目已知可用 | `56e425dcc7b59ab0e1aec685b178b99f` | `<NFS>/npu/librknnrt.so` | ✅ 完全正常 |
| 板卡备份 | `a37ee1d5d664c79836bf6e35b7ef6289` | `/usr/lib/librknnrt.so_bak`（= 2.3.2） | ✅ 正常（`api_version=2.3.2`） |

* 两份 2.3.0 **同版本号、同大小 7,259,064 B，但有 20,439 字节不同**（首个差异在 offset 737281，是整段代码）。
* ⭐ **修法不需要物理断电**：把已知可用的库放到 `LD_LIBRARY_PATH` 前面即可：
  `LD_LIBRARY_PATH=<dir>:/usr/lib`，其中 `<dir>/librknnrt.so` = md5 `56e425dc…`。
  实测：`23.586 ms`（2.3.0）/ `24.101 ms`（2.3.2）⇒ 与出事前的 23.658 ms 一致（差 0.3%）。
* ⚠️ **判别口径**：`exit 132` 有**两个**成因，必须分开：
  ① **库的构建不对**（用好库即可排除）；
  ② **NPU 真被软复位卡死**（`RKNPU: job timeout` → `soft reset`）⇒ 才需要物理断电。
  ⇒ 排查顺序：**先换库，再考虑断电**。
* ⚠️ 本轮正是 `rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2)` 触发了 `job timeout` → `soft reset`（见 §5.3），
  **切勿在运行时切 core mask**。

### 10.4 ⚠️ **校准集决定类别分数，不只决定量化尺度**

用**红外特征图**（`uav_gmc_median` 的 3 通道特征）校准一个 **COCO 预训练**模型 ⇒ 板端类别分数**坍缩**：

| | host（float32 ONNX） | 板端 int8（红外校准） |
|:--|:--|:--|
| 检出数 | 5 | 5 ✅ |
| 框位置 | bus `[85.7,136.6,554.3,445.1]`、person×4 | bus `[71.7,133.7,551.3,445.2]`、person×4 ✅ **逐框对得上（Δ≈5–13 px）** |
| 类别分数 | 0.8971 / 0.8566 / 0.8448 / 0.8283 / 0.5578 | **0.6676 ×4 + 0.4996**（坍缩到几乎同一个量化码） |

⇒ **框是对的、分数是坏的** ⇒ 说明**解码契约（`(cx,cy,w,h)`、通道步长 8400、量化 zp/scale）全部正确**，
问题**只在校准分布**。
⇒ ⚠️ **"标准 YOLO26n" 的基线必须用 COCO 校准集**（项目已有 `/media/manu/1TB-Volume/rknn_work/calib/calib_coco120.txt`，120 张 COCO 图）。
⇒ ⭐ 而本项目自己的热图模型**必须**用红外特征校准（`hm_convert_rknn.py` 的注释已写明理由）⇒
**校准集不是"随便挑一批图"，它是模型的一部分**。

### 10.5 板卡↔host 逐框对账工具

`manu/pipeline/npu/yolo26n_host_ref.py --compare <board.tsv>`
⚠️ **该比较器按索引配对，而两侧的排序不同**（板端 int8 分数坍缩后排序变了）⇒ 报出��`worst box delta = 405 px` 的**假警报**。
⇒ **正确用法**：按框坐标匹配（IoU）或先按类别+框排序再配对。**结论以"检出数 + 框集合"为准，不要以"逐索引差"为准。**

---

## 十一、🔴 顺带修掉的两个工具级坑（都不是我本轮造的，但会静默毁掉实验）

### 11.1 🔴 `hm_convert_rknn.py` **会忽略 `--calib-list`**（已修）

```python
calib = build_calib_list(args.calib_split, args.calib_n, Path(args.calib_list))
```

`build_calib_list()` **总是**从 `uav_gmc_median/images/<split>` 重新生成清单并**覆盖** `--calib-list` 指向的文件。
⇒ **传 `--calib-list` 毫无作用**，而且**会把那个文件改掉**。
⇒ 本轮第一次"用 COCO 校准集重建"因此**静默地又用了红外图**，而且**把 `calib_coco120.txt` 覆盖成了红外清单**
（已按 `coco_cal/*.jpg` 重建为 120 行全路径清单）。

**已修（向后兼容的加法式改动）**：新增 `--calib-list-only` ⇒ **逐字使用** `--calib-list`，
并**校验每一行文件存在**（缺一即 `FATAL` 退出，不静默继续）：

```bash
/media/manu/1TB-Volume/conda-envs/rknn230/bin/python manu/pipeline/npu/hm_convert_rknn.py \
  --onnx <x.onnx> --out <x.rknn> \
  --calib-list /media/manu/1TB-Volume/rknn_work/calib/calib_coco120.txt --calib-list-only
```

⇒ **教训（铁律〇之五「链式脚本可证伪」）**：一个**命令行参数名存在、但实现里被静默忽略**，
比"参数不存在"危险得多 —— 日志里明明打印了 `[calib] list -> <你的文件>`，看起来一切正常。
⇒ **凡是"参数看起来生效了"的开关，都要从产物或日志里反向验证一次**。

### 11.2 🔴 `nbuf_prefer` 只接受 **bool**（不是字符串）

```python
rknn.config(nbuf_prefer="high_perf")
# ValueError: The value of nbuf_prefer only support <class 'bool'> !
```

⇒ 正确写法：`nbuf_prefer=True`。
⇒ 同理 `sram_prefer=True`（bool）、`compress_weight=True`（bool）。

### 11.3 ⚠️ `rknn-toolkit2 2.3.0` 的 Python API **没有构建期 core_mask**

`build(self, do_quantization=True, dataset=None, rknn_batch_size=None)` —— **无** `core_mask`。
`core_mask` 只属于 `init_runtime()`（只对"host 直连设备"有意义）。
⇒ 想让模型跨 3 核拆分**在本工具链里做不到**；运行时 `rknn_set_core_mask(0_1_2)` 又会搞挂驱动（§5.3）。
⇒ **结论：本板单模型只能吃 1 核**（这与 §10.1 的 ~256–282 GOPS ≈ 单核实测能力相符）。

---

## 十二、⭐ 最终一轮（2026-11-05 17:1x）：全部变体一次会话测完 + **端到端数值验证通过**

### 12.1 一次会话的最终表（全部 7 个 yolo26n 变体 + 2 个参照点）

| 模型 | 来源 | ms/帧 | GFLOPs | 隐含 GOPS |
|:--|:--|--:|--:|--:|
| **mobilenet_v1**（224×224） | SDK 自带模型 ＋ `rknn_bench_generic` | **2.018** | 0.57 | **282** |
| **yolov5s_rknn230**（640×640） | **Rockchip 自带 `rknn_yolov5_demo`** | **26.207** | 7.2 | **275** |
| **yolo26n_int8_stock**（640×640） | 本项目 `rknn_yolo26n_smoke` | **23.699** | 6.1 | **257** |
| `arm_baseline` | 默认构建 | 23.785 | 6.1 | 256 |
| `arm_coco_calib` | **COCO 校准集重建** | 23.637 | 6.1 | 258 |
| `arm_compress` | `compress_weight=True` | 23.722 | 6.1 | 257 |
| `arm_nbuf` | `nbuf_prefer=True` | 23.680 | 6.1 | 258 |
| `arm_sram` | `sram_prefer=True` | 23.563 | 6.1 | 259 |
| `arm_sram_compress` | 两者 | 23.540 | 6.1 | 259 |
| `arm_sram_nbuf` | 两者 | 23.734 | 6.1 | 257 |

⇒ **7 个 yolo26n 变体跨度 23.540–23.785 ms = 1.04%** ⇒ **所有构建旋钮都是噪声**（`sram`/`compress`/`nbuf` 及其组合）。
⇒ **COCO 校准版与红外校准版时延完全一致**（23.637 vs 23.699，差 0.26%）⇒ **校准集不改变时延，只改变数值保真度**（§10.4 已证）。

### 12.2 ⭐⭐ **端到端数值验证通过**（板端 int8 vs host float32 ONNX，同图同 letterbox 同解码同 NMS）

`bus.jpg`（COCO 校准版，md5 `731e3c3063e22cf782cfee3011bd1ca8`）：

| # | 板端 int8 | host float32 ONNX | Δscore | Δbox(px) |
|:--|:--|:--|--:|--:|
| 1 | cls=**5**(bus) 0.8567 `[79.50,135.70,561.99,445.48]` | cls=5 0.8971 `[85.74,136.56,554.27,445.11]` | 0.040 | ≤6.2 |
| 2 | cls=0 0.8396 `[115.14,235.76,224.80,526.35]` | cls=0 0.8566 `[108.13,235.56,223.77,534.42]` | 0.017 | ≤8.1 |
| 3 | cls=0 0.8096 `[213.83,241.24,285.11,520.87]` | cls=0 0.8448 `[212.10,241.41,285.20,510.34]` | 0.035 | ≤10.5 |
| 4 | cls=0 0.7796 `[472.89,224.80,552.39,515.38]` | cls=0 0.8283 `[477.03,229.20,560.88,521.70]` | 0.049 | ≤8.5 |
| 5 | cls=0 0.5483 `[78.13,334.45,113.77,515.38]` | cls=0 0.5578 `[79.03,329.11,116.57,515.52]` | 0.010 | ≤5.4 |

⇒ **检出数、类别、顺序全对；分数差 ≤0.049（int8 量化正常量级）；框差 ≤10.5 px** ⇒
**整条链路（ONNX 导出 → RKNN 量化 → 板端 int8 → 自写后处理）已被逐框验证**。
⇒ **COCO 校准把 §10.4 的分数坍缩彻底修掉**（0.6676×4 → 0.8567/0.8396/0.8096/0.7796/0.5483）
⇒ **再次确认：坍缩的唯一原因是校准分布，不是解码契约**。

### 12.3 ⭐⭐ 一句话结论（供后续任务直接引用）

> **标准 YOLO26n（官方权重、官方 P3-P5 拓扑、int8、640×640）在这块 RK3588 上 `rknn_run` = 23.7 ms。**
> **本项目热图模型 = 55.5 ms（单尺度）/ 108.2 ms（mosaic e2e）⇒ 标准 P3-P5 检测头比本项目结构快 2.35× / 4.58×。**
> **但这块板的 NPU 有效吞吐只有 ~256–282 GOPS（标称 6 TOPS 的 ~4.5%），Rockchip 官方 demo 的 YOLOv5s 同样要 26.2 ms**
> ⇒ **"标准 YOLO26n 应当 3~6 ms"在板上不成立**，那需要 1000–2000 GOPS。
> **且单次推理有 ~13.9 ms 固定开销** ⇒ 该板"每帧 NPU 时间"的硬地板就在 ~14 ms。

---

## 十三、⭐⭐⭐ 频率锁与激活函数的实测裁决（2026-11-05 17:2x–17:4x）

### 13.1 本板的频率配置：**杠杆已经全额用尽**

| 项 | 当前值 | 状态 |
|:--|:--|:--|
| NPU devfreq | `cur_freq=1000000000`（1 GHz）= `max_freq`，governor=`performance` | ✅ 已锁最高 |
| NPU debugfs 实际频率 | `/sys/kernel/debug/rknpu/freq = 1000000000`，`volt=800000`，`power=on` | ✅ 确实跑在 1 GHz |
| **DDR devfreq (`dmc`)** | `cur_freq=2112000000`（2112 MHz）= `max_freq`，`target_freq` 同值，governor=`performance` | ✅ 已锁最高 |
| CPU | cpu0 @1800 MHz(max)、cpu4 @**2352 MHz**(max)，8 核全 `performance` | ✅ 最高 |
| 温控 | SoC 39.8 °C / NPU 39.8 °C / bigcore 38.8 °C | ✅ 远未到限频点 |

**与社区说法的三处差异**：
1. **本板 DDR 峰值是 2112 MHz，不是 1848 MHz**（RK3588 LPDDR4 速度档不同；可用档 528 / 1068 / 1560 / 2112）。
2. **DDR 已连续保持在 2112 MHz 约 8 小时 34 分**（`trans_stat` 当前行累计 30,792,366 ms，与 uptime 吻合；528 MHz 只出现过 17.46 s）
   ⇒ **"常年缩在 528 MHz（28%）"在这块板上不成立**。
3. **本板没有 `/sys/kernel/debug/ddr_freq/` 节点** ⇒ 社区那套分域 `ddr_freq` 调试接口不存在，
   "锁 ddr_freq" 这件事**已由 devfreq 的 `performance` governor 自动完成**。

### 13.2 DDR 频率敏感度 A/B（NPU 恒 1000 MHz）：**复现了社区现象，但收益已经拿到了**

| DDR 频率 | mobilenet | yolo26n | vs 最高 |
|:--|--:|--:|--:|
| **2112 MHz（=max）** | 2.036 | **23.590** | — |
| 1560 MHz | 2.054 (+0.9%) | 24.153 (+2.4%) | |
| 1068 MHz | 2.091 (+2.7%) | 25.962 (+10.1%) | |
| **528 MHz（社区说的"未锁"档）** | 2.870 (+41%) | **33.863 (+43.5%)** | **+10.3 ms** |

⇒ **DDR 从 2112 掉到 528 只让 yolo26n 慢 10.3 ms** ⇒ 与社区报的「24 ms → 16.6 ms」**同源同量级**。
⇒ **本板已锁最高 ⇒ 这 10 ms 的收益早已拿到，没有剩余空间。**

### 13.3 NPU 频率 A/B（DDR 恒 2112 MHz）：**把"固定开销"和"stall"分开了**

| NPU 频率 | mobilenet | yolo26n |
|:--|--:|--:|
| 1000 MHz | 2.012 | 23.585 |
| 500 MHz | 3.168 | 37.830 |
| 300 MHz | 4.979 | 57.823 |
| 1000 MHz（恢复） | 2.000 | 23.871 |

拟合 `time = a + b/f`：

| 模型 | a = 频率无关部分 | b/f = 随频率缩放部分（@1 GHz） |
|:--|--:|--:|
| yolo26n | **8.91 ms** | **14.67 ms** |
| mobilenet | 0.74 ms | 1.27 ms |

⇒ 随频率缩放部分里，周期表算出的**纯卷积算力只有 5.20 ms**
⇒ **其余 ~9.5 ms 是 stall（等 DDR）** —— 与 §13.2 的 DDR A/B（+10.3 ms）**互相印证** ✓

### 13.4 ⭐⭐⭐ **修正后的 23.6 ms 拆解（三个独立来源互证）**

| 分量 | 值 | 占比 | 能不能靠改网络压掉 |
|:--|--:|--:|:--|
| runtime / dispatch / 同步（**与 NPU、DDR 频率都无关**） | **~8.9 ms** | 38% | ❌ 碰不到（属调用方式） |
| NPU 纯卷积算力（周期表 5.204 Mcycle） | **~5.2 ms** | 22% | ✅ 砍 MAC / 通道 / 层数 |
| **DDR 等待（搬运量）** | **~9.5–10.3 ms** | **41%** | ✅ **砍大张量 = 降 RW（最大单项）** |

交叉验证热图模型 base：RW 113.5 MB（yolo26n 44.8 MB，2.5×）⇒ DDR-wait ≈ 24 ms，算力 25.5 ms，开销 8.9 ms ⇒ 合计 ≈ 58 ms ≈ **实测 55.501 ms** ✓

### 13.5 🔴🔴🔴 **SiLU→ReLU 是 39% 的免费午餐，而 RKNN 的周期表给它定价为零**

| 变体（stock YOLO26n，同会话） | `rknn_run` @1000 MHz | @500 MHz | @300 MHz | 拟合 a / b |
|:--|--:|--:|--:|:--|
| **SiLU（baseline）** | 23.814 | 37.655 | 57.765 | a=**9.26** / b=14.55 |
| **ReLU 全换** | **14.330（−39.8%）** | 22.799 | 35.915 | a=**5.08** / b=9.25 |
| **ReLU 只换 neck+head（70/105 模块）** | **19.040（−20.0%）** | | | |

**机理（关键）**：
* SiLU 在 NPU 上被融合成 Rockchip 的 **`ConvExSwish`** 算子；ReLU 不会。
* ⚠️ **RKNN 周期表对两者给的是完全相同的 NPU 周期（5,204,400）** ⇒ **周期模型根本没给激活函数定价**
  ⇒ **§9 的成本模型在"激活函数"这一维上是瞎的**（算力项 0.902 ms/Mcycle 对 ConvExSwish 系统性低估）。
* 收益**同时**落在两项上：频率无关项 9.26→5.08 ms（**dispatch/调度**），随频率项 14.55→9.25 ms（**真实执行**）。
* 对本项目热图模型**同样成立**：base/late/late32 三个臂的算子表里都有 **304 处 `ConvExSwish`**。

⚠️ **纪律**：
* 这是 **非等价改写**（改的是 SiLU 训练出来的权重），**只能当时延探针，绝不能出精度结论**
  （与项目里 `late`/`dw` 探针的用法一致）。
* 本项目**唯一可用 scope 是 `trainable`**（b5..b10 + neck + head）：`b0..b4` 是官方 SiLU 权重的冻结底模，
  换掉它的激活就等于改变冻结底模的行为（铁律六）。本文件已给 `hm_export_onnx.py` 加了
  `--act {silu,relu}` + `--act-scope {all,trainable}`。
* **必须重训才能恢复精度**：这是"改结构"，不是"改参数"，因此**必须走受控 A/B**（铁律五）。

### 13.6 ⭐⭐ **本项目热图模型的同场实测（2026-11-05 17:4x，历史 6 臂 + 本轮 4 探针一次会话）**

板卡 `/root/merge_head/*_int8.rknn`（2026-10-16 建）与本轮新探针，同一场、同一图、同一 binary：

| 臂 | 今天同场 `rknn_run` | 历史值(10-16) | 差 | vs 现状 | 精度风险 | 需重训 |
|:--|--:|--:|--:|--:|:--|:--|
| `base`（现状） | 55.640 | 55.501 | +0.25% | — | — | — |
| `dw` | 49.474 | 49.860 | −0.77% | | | |
| `late` | 36.592 | 36.657 | −0.18% | −34.2% | **高**（砍 P1 分辨率 320→160，精度**从未实测**） | 是 |
| `late32` | 33.903 | 34.089 | −0.55% | −39.1% | 同上 | 是 |
| `latedw` | 35.009 | 35.255 | −0.70% | | | |
| `latedw32` | 34.257 | 34.302 | −0.13% | | | |
| `silu_probe`（本轮重建基线） | 55.668 | — | **vs hist base 差 0.05%** | — | — | — |
| **`relu_all`** | **44.371** | — | | **−20.3%** | 中（动冻结底模 ⇒ 违反铁律六） | 是 |
| **`relu_trainable`** | **47.733** | — | | **−14.2%** | **低**（只换激活，分辨率/感受野不动） | 是 |
| **`late + relu_trainable`** | **36.581** | — | | −34.2% | 高 | 是 | ← **比 `late` 还快 0.011 ms = 无增益** |

⇒ **历史 6 臂今天全部复现（±0.8%）** ⇒ 板卡状态与 10-16 当天一致，**历史数据可信**；
⇒ **重建基线与历史 base 差 0.05%** ⇒ 本轮重建路径与项目原路径同源；
⇒ 🔴 **两条杠杆互斥、不叠加**：**ReLU 的收益来自"高分辨率下的 SiLU 税"，`late` 把 320×320 砍掉后税就没了**
⇒ **决策口径**：
* 要**低风险**提速（不动分辨率、不动感受野）⇒ **`base + relu_trainable` = 47.73 ms（−14.2%）**
* 要**极致** ⇒ **`late` = 36.59 ms**（且**不要再叠 ReLU**，实测无增益）
* ⚠️ 两者都**必须重训**（非等价改写）⇒ **必须走受控 A/B**（铁律五）；精度结论一个都还没有。

### 13.7 本轮对项目代码的两处**加法式**改动（向后兼容）

* `manu/pipeline/npu/hm_export_onnx.py`：新增 `--act {silu,relu}` + `--act-scope {all,trainable}`
  （`trainable` = b5..b10 + neck + head；**b0..b4 保持 SiLU**，遵守铁律六的底模冻结）。
* `manu/pipeline/npu/yolo26n_export_onnx.py`：新增 `--act` + `--act-scope {all,neck_head}`（标准 yolo26n 用）。
* 两者默认行为**完全不变**（默认 `silu`/`all`，等价于改动前的输出）。
* ⚠️ 探针臂导出时必须加 `--skip-check`（位精确自检会因为数值改变而失败，那是预期的）。
