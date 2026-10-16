# ONNX → RKNN int8 转换与板卡部署（Trial 0474 热图检测器）

> **面向操作者的一页式手册。** yolov5 版见同目录 `CONVERT.md`；**两者不可混抄**
> （后处理、输出张量语义、标定策略全都不同）。
> 结论与全部实测数字见 `manu/memory/heatmap_npu_trial0474.md`。

## 一、结论（2026-10-14 实测）

**当前交付版 = `trial0474_hm_nop0_int8.rknn`（v3，删 P0 Highway）**

| 项 | 结果 |
|:---|:---|
| **主推 `trial0474_hm_nop0_int8.rknn`（v3）** | **3,799,304 B**，md5 `ef302edb1b5aff1d376204bb7ad77550`，板卡 `rknn_init` ✅，**0 个 CPU 计算算子** ✅ |
| v3 单帧端到端 | **56.7 ms → 17.7 fps**（`rknn_run` **55.5 / 55.7 ms**，NPU cycles **28,250,400**，Conv MAC **21.54 G**） |
| 首轮起点 `trial0474_hm_int8.rknn`（v1） | **4,093,832 B**，md5 `0e2f8ae5c47912bfb9e6a38c4dec0db8`，**152.9 ms → 6.5 fps**（`rknn_run` 占 99.2%，计算主导；1860 次独立复验 152.758 ms，相对差 0.10%，`rules.md` §三·10）。⚠️ **含 1 个 CPU 计算算子、2 条 fallback 警告，已被 v2/v3 取代，仅作起点存档** |
| 中间版 `trial0474_hm_dw_int8.rknn`（v2，带 p0） | 4,168,968 B，**103.7 ms → 9.5 fps**；CPU 算子已清零，但被 v3 取代 |
| **加速归因（两笔独立的账，代价相当）** | ① `nn.PixelUnshuffle` 改写成 depthwise conv ⇒ 清掉 **1 个 CPU 计算算子**：**151.2 → 103.7 ms（−47.5 ms）**；② 删 P0 Highway：**103.7 → 55.5 ms（−48.2 ms）**。⚠️ **只查回退会漏掉后一半**；v2→v3 MAC 只降 6.8%（23.11 → 21.54 G）却降时延 46.5%，因为 p0 那几层是 **640×640 / C=16** 的卷积，**同时踩中 NPU「大分辨率搬运」与「窄通道空转」两个最差工况**（原理见 `manu/memory/heatmap_npu_trial0474.md`） |
| P0 可删的依据 | 底模 Trial 22 **100% 物理冻结**（1,790,531 参数），p0 支路仅 **1,954 参数**零初始化冷启动、**Step 0 绝对数学恒等**；训练收益仅召回 **+0.15pp**，收敛后 F1 反而略降（0.9057 → 0.9055）。600 帧配对检验：带 p0 vs 去 p0 **不一致帧 1/600，McNemar p = 1.0000 ⇒ 无可归因精度代价** |
| `trial0474_hm_fp16.rknn` | **6,739,809 B**，**带 P0 的图** fp16 在板卡提交失败 ⇒ 别拿它当对照臂（详见第六节）。⚠️ **2026-10-16 限定：该结论只对「图里含 `p0_highway` 支路」成立** —— 去 P0 的 fp16（`trial0474_hm_nop0_fp16.rknn` md5 `07c893e4…`）在同板实测可提交、可全量跑通 31,613 帧。失败算子是 `Conv:/model/p0_highway/p0_dw/Conv`，在 P0 支路内部 |
| 诚实性闸门 | 空白输入拒答 ✅ / 空对照（输入取反 ⇒ 两个输出哈希都变）✅ / 两次运行逐字节一致 ✅ / **时钟自检 `[ok]`** ✅（50 ms `usleep` 实测 50.137 / 50.080 / 50.124 / 50.133 ms） |
| int8 的真实代价（v1 口径，结论未被 v2/v3 推翻） | 剔除 DJI 后召回 **0.8955 → 0.8727（−2.27 pp）**，配对 McNemar **p = 0.0129 ⇒ 统计显著**；但**检出位置分歧 0/600**，代价全在阈值附近的峰值 |

## 二、环境（**不要**装进根分区）

根分区 `/dev/sda1` 仅剩 **15 GB / 93% 已用**；板卡 `/` 也只剩 **1.7 GB**。

```
P=/media/manu/1TB-Volume/conda-envs/rknn230   # rknn-toolkit2 2.3.0（与板卡 runtime 同 commit c949ad889d）
W=/media/manu/1TB-Volume/rknn_work/hm         # 中间产物 / 标定清单 / 参考臂输出
```

⚠️ **toolkit 版本必须与板卡 `librknnrt` 对齐**，本轮是 **2.3.0**。
`conda-envs/rknn`（1.4.0）那条路的产出已被 NPU 拒绝，**留作证据，不要再用**（`CONVERT.md` 第四节）。
⚠️ 模型侧脚本（`hm_export_onnx.py` / `hm_reference.py`）需要仓库的 torch + ultralytics 环境；
`$P` 里装的是 **CPU-only torch wheel**，转换够用，跑模型前请自行切到训练服务器环境。

## 三、十一个脚本各自干什么

> 本节只列**转换 / 推理 / 对账**链路上的脚本。端到端 demo 另有 4 个
> （`hm_index_probe.py`、`hm_demo_report.py`、`hm_demo_osd.py`、`run_hm_demo.sh`），
> 见 **第十节**。

| 脚本 | 职责 | 关键参数 |
|:---|:---|:---|
| **`hm_common.py`** | **唯一的一份定义**：letterbox 几何、`extract_peaks_np`、`gt_from_label`。**不是脚本，是被 import 的库** —— 板端 C++ 里的 letterbox/抽峰是它的 1:1 移植，两边必须一起改 | — |
| `hm_export_onnx.py` | `best.pt` → 双输出 ONNX；**断言 523/523 state-dict 张量匹配**（据此定 `scale='n'`），再用 onnxruntime 对一张真实特征 JPEG 交叉核对 | `--weights --out --opset 12 --check-image --tol-hm 1e-4 --tol-off 1e-3 --skip-check` **`--ablate-p0`** |
| `hm_convert_rknn.py` | ONNX → `.rknn`（int8，`--no-quant` 出 fp16 臂）；**内置标定集抽样与清单落盘** | `--onnx --out --calib-split train --calib-n 200 --calib-list --no-quant` |
| `hm_reference.py` | **CPU/PyTorch 权威臂**：出 `<stem>_hm.npy / _off.npy`、`detections.json`、以及 TP/FP/FN。**模型不在 eval 态直接拒绝运行** | `--frames <list> --out <dir> --th 0.25 --stride 2 --topk 100 --dist 8 --no-gt --no-dump-tensors` **`--ablate-p0`** |
| `hm_compare_npu_cpu.py` | 三件互不相干的事一次查清：① 板端 C++ 解码 vs host numpy 解码；② NPU vs CPU 张量逐元素；③ NPU 检出对 GT（P/R 各带 Wilson CI） | `--ref <dir> --npu <dir> --tag int8 --th 0.25 --stride 2 --dist 8` |
| `hm_seq_stats.py` | 读 `detections.json` 出**分序列 + 总体** TP/FP/FN；每个比例都带绝对数与 Wilson 区间；**`n_gt < 100` 的序列标 `low-n`**（`rules.md` §三·6b）；按 recall **升序**排，最差的在最前 | `<detections.json> [--sort recall\|f1\|seq]` |
| `hm_osd.py` | 出 OSD 图（单帧 / 连帧条带）。**每个 panel 都标它来自 NPU 还是 CPU**，NPU 张量缺失时直接说缺，不偷画 CPU 结果 | `--mode single\|sequence --image --ref --npu --tag --th --stride --out [--zoom --cols --frames]` |
| **`hm_pixelshuffle.py`** | `nn.PixelUnshuffle` 的 **depthwise-conv 逐位等价替身**（v2 清 CPU 算子的关键）。`--verify` 自检四个形状 **(1,16,640,640) / (1,3,320,320) / (2,16,128,96) / (1,64,8,8) 全部 `torch.equal` EXACT**。⚠️ 两个坑：one-hot 必须落**通道对角线** `w[o,o,i,j]=1`；输出通道索引必须 `c·f²+i·f+j`（**写反了 C=1 时完全正确，测不出来**） | `--verify` |
| **`hm_flops.py`** | 逐层 MAC + 分辨率/通道分布统计。⚠️ **禁止用 `onnx.shape_inference` 估计算量**，本工程实测差 **11 倍**（2.31 G vs 真值 25.5 G），必须用 RKNN 算子表（`rules.md` §三·4b） | `--weights --imgsz 640` |
| **`hm_struct_variant.py`** | 结构变体生成器（`stride` / `head_k3` / `imgsz` / `p0` 四维）。⚠️ **已导出的变体 int8 多数未建，不是已交付项**（见第五节末表） | `--weights --variant <name> --out --outdir --list` |
| **`hm_score_board_det.py`** | 板端 `--detections-out` 的 TSV ↔ CPU 权威臂 `detections.json` 对账器，出同样的 TP/FP/FN。**600 帧对账走它，不要走 dump**（见第七节） | `--tsv <板端tsv> --ref <ref目录> --dist 8 [--out <json>]` |

⚠️ **`hm_export_onnx.py --ablate-p0` 与 `manu/inference/cache_trial0474_inferences.py --ablate-p0` 导出的 ONNX 字节级一致（MD5 相同 `028f23c655f421fe61a5e43575715fca`）** ⇒ 参考臂与上板模型必然同源，不用担心对账对的是两套网络。

配套（不在上表，但部署必需）：

| 文件 | 作用 |
|:---|:---|
| `rknn_heatmap_smoke.cpp` | 板端程序：解码 JPEG → letterbox → `rknn_init/inputs_set/run/outputs_get` → C++ 抽峰 → `--dump` 原始张量 + 检出。**免 OpenCV、免 RGA**（板卡两者都没有）。⚠️ **原版只在批帧数 ≤ 12 时打印检测结果**，批量模式什么都不输出；已加 **`--detections-out <path>`**（每帧一行紧凑 TSV） |
| `build_rknn_heatmap_arm64.sh` | 交叉编译上表程序（见第五节） |

## 四、标定政策（**转换里唯一最要紧的决定**）

**交付主线（复制即可，v3 去 p0）**：

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu

# 1) 导出 ONNX（训练后移除 P0 Highway）
$P/bin/python hm_export_onnx.py --weights <best.pt> \
    --out $W/models/trial0474_hm_nop0.onnx --opset 12 --check-image --ablate-p0
#    → 7,210,756 B，md5 028f23c655f421fe61a5e43575715fca

# 2) 转 int8（标定集参数见下方政策，别改）
#    ⚠️ hm_convert_rknn.py 自己不写日志文件，必须用重定向把构建输出留下来，
#    否则第 3 步的闸门会去 grep 一个不存在的文件、**静默判过**。
mkdir -p $W/models
$P/bin/python hm_convert_rknn.py --onnx $W/models/trial0474_hm_nop0.onnx \
    --out $W/models/trial0474_hm_nop0_int8.rknn \
    --calib-split train --calib-n 200 --calib-list $W/calib_train_features.txt \
    2>&1 | tee $W/logs/build_nop0_final.log
#    → 3,799,304 B，md5 ef302edb1b5aff1d376204bb7ad77550

# 3) 🔴 构建后立刻查 CPU 回退（rules.md §四·9e 交付硬闸门）
L=$W/logs/build_nop0_final.log
test -s "$L" || { echo "❌ 构建日志缺失，闸门无法判定"; exit 1; }   # 防空文件误判通过
grep -ic 'fallback' "$L"                        # 期望 0
grep -c 'Transpose will fallback to CPU' "$L"   # 期望 0
grep -oE '^[DWI] RKNN: \[[0-9:.]+\] [0-9]+ +[A-Za-z0-9_./]+ +[A-Z0-9]+ +(CPU|NPU|GPU)' "$L" \
  | awk '{print $NF}' | sort | uniq -c          # 期望：CPU 仅 3（InputOperator + 2×OutputOperator）
```

⚠️ **`--ablate-p0` 两边必须同时开**：本流程的 `hm_export_onnx.py --ablate-p0` 与
`manu/inference/cache_trial0474_inferences.py --ablate-p0`（`cache_trial0474_inferences.py:128-148`，
含 `module.` / `model.model.` / `model.` 键名归一化与 `[ABLATE-P0 GATE]` 硬闸门）
**导出的 ONNX 字节级一致，MD5 相同 `028f23c655f421fe61a5e43575715fca`** ⇒ 参考臂与上板模型必然同源。
⚠️ 参考臂也**必须**用他们自己的 `--ablate-p0`，不要拿别的脚本凑。

`hm_convert_rknn.py` 的默认值就是政策本身（交付主线见上），这里复述理由以便你改参数时知道在破坏什么：

1. **标定集必须是「已建好的三通道特征 JPEG」**（`uav_gmc_median/images/{split}`），
   Ch0 = 原始灰度、Ch1 = GMC 短差、Ch2 = 21 帧中值残差。
   **喂原始 IR 会让量化器在一个模型永远见不到的输入分布上标定**，
   纸面上量化得好、检测得好，实际悄悄丢掉热图头赖以生存的次阈值脉冲。
2. **`--calib-split` 默认 `train`，不许改成 `val`。** 量化本身不是指标，但在评测集上标定是不必要的污染。
3. **分层抽样、确定性、可审计**：**200 张跨 199 个不同序列**轮转取（排序 + 固定步长），
   清单落盘 `$W/calib_train_features.txt`（200 行）。**这份清单是产物，不是临时文件 —— 必须随模型一起归档。**
4. **量化参数 `mean 0 / std 255`**：图直接吃 uint8、由 RKNN 自己除 255，
   与 PyTorch 侧 `uint8/255` **逐比特等价** ⇒ host 与板卡看到同一组数，差异只可能来自 NPU。
5. ⚠️ 本条**只对本模型的部署精度成立**。`rknn_npu_board_run.md` §9.6 的「标定集大小不影响一致度」
   测的是**两个模型之间的一致性**，**不是同一个命题，不可互相引用**（`rules.md` §四·9）。

### 已导出但 int8 未建的结构变体（**供后续选型，不是已交付项**）

用 `hm_struct_variant.py --variant <name>` 生成，**目前只到 ONNX，没有一条建过 int8、没有一条上过板**：

| 变体 | stride | head | GMAC | heatmap |
|:---|:---:|:---:|---:|:---|
| `no_p0` | 2 | 3×3 | 21.54 | 320×320 |
| `no_p0_head1x1` | 2 | 1×1 | 11.39 | 320×320 |
| `s4` | 4 | 3×3 | 10.04 | 160×160 |
| `s4_head1x1` | 4 | 1×1 | 6.68 | 160×160 |
| `in320_s4_head1x1` / `in320_s2_head1x1` | 4 / 2 | 1×1 | — | 输入 320×320 |

⚠️ **10 ms 目标尚未验证可达**：最激进的 `s4_head1x1` 为 6.68 G MAC，按当前 21.54 G ↔ 55.5 ms
粗略外推约 **17 ms**，大概率不够。⚠️ **线性外推已被证伪（v2→v3 MAC 降 6.8% 而时延降 46.5%），必须实测**。
⚠️ **`p0_highway` 与 stride 4 结构不兼容**：highway 内部只下采样一次（640→320），残差恒为 stride 2，
无法与 160×160 的 neck 相加 ⇒ **该支路把整网锁死在 stride 2**。

## 五、交叉编译

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu
./build_rknn_heatmap_arm64.sh          # → out/rknn_heatmap_smoke
```

脚本已内建的约束（不要绕过）：

- 链**板卡自己的** `librknnrt.so`（`RT_LIB_DIR` 默认 `/run/media/manu/SSD960-LOCAL/nfs/npu`），
  断言 `DT_NEEDED` 是**纯 SONAME** `librknnrt.so`，**绝不用** SDK 里那份 1.4.0 的；
- `-ffp-contract=off` + `-fno-fast-math` + `-fno-unsafe-math-optimizations`（浮点契约）；
- 头文件私有暂存到 `.build-include/`，**不 `-I/usr/include`**；
- 板端无 OpenCV、无 RGA ⇒ JPEG 解码用 SDK 自带的 `stb_image`。

## 六、部署与运行

```bash
# 二进制走 SSH（走 NFS 会 Text file busy / ETXTBSY）
scp out/rknn_heatmap_smoke root@192.168.0.64:/root/
# 模型与帧走 NFS：host 的 /run/media/manu/SSD960-LOCAL/nfs/npu/ == 板卡的 /mnt/manu/npu/
#   （模型与帧清单在 data/，对账输出在 out/；板卡 / 只剩 1.7 GB，别把数据集搬上板卡）
ls -l /run/media/manu/SSD960-LOCAL/nfs/npu/data/trial0474_hm_nop0_int8.rknn /run/media/manu/SSD960-LOCAL/nfs/npu/data/hmval_list.txt

# 单帧 + 计时（--loops/--warmup 决定计时口径；--detections-out 出紧凑 TSV）
RK3588_PASSWORD=*** python3 ../../opencl/board/board_run.py --exec \
  'cd /mnt/manu/npu && LD_LIBRARY_PATH=/usr/lib \
   /root/rknn_heatmap_smoke data/trial0474_hm_nop0_int8.rknn data/frames/<stem>.jpg \
    --th 0.25 --stride 2 --loops 30 --warmup 10 \
    --detections-out out/nop0/<stem>.tsv --tag <stem>'

# 批量（帧清单一行一个）—— 600 帧对账走 --detections-out，不要走 --dump
... /root/rknn_heatmap_smoke data/trial0474_hm_nop0_int8.rknn --batch data/hmval_list.txt \
    --th 0.25 --stride 2 --detections-out out/nop0_det600.tsv
```

🔴 **口令纪律**：`RK3588_PASSWORD` 环境变量传入，**不落任何文件**（不进脚本、不进文档、不进仓库）。

**板端程序用法**（`--help` 未打印的信息不要写进验收步骤）：

```
rknn_heatmap_smoke <model.rknn> <feature.jpg> [--th F] [--stride N] [--topk N]
                   [--loops N] [--warmup N] [--perf] [--dump DIR] [--tag NAME]
                   [--detections-out PATH]
rknn_heatmap_smoke <model.rknn> --batch <list.txt> [--th F] [--stride N]
                   [--dump DIR] [--detections-out PATH]
```

⚠️ **`--detections-out` 存在的理由**：原版二进制**只在批帧数 ≤ 12 时才打印检测结果**，
批量模式什么都不输出，逼得对账只能走 `--dump`；而 dump 经 NFS 约 **2 帧/分钟**
⇒ **600 帧需 583 分钟**。加了 TSV 之后 **600 帧几分钟完成**，检出结果完全等价。

⚠️ **stride 变体必须显式传 `--stride`**：`p0_highway` 与 stride 4 不兼容（见第四节末表），
若上板 stride-4 变体而漏传，smoke 二进制默认 `--stride 2` 会报
`input/grid ratio 4 does not match --stride 2`。

⚠️ **`--perf` 会打开 `RKNN_FLAG_COLLECT_PERF_MASK`，`rknn_run` 会显著变慢** ——
**报性能数字时不要带 `--perf`**（yolov5 轮实测带该标志慢 54%）。

## 七、对账与出图

```bash
# CPU 权威臂（必须与板端同一帧集、同一 th、同一 8 px 容差；v3 臂要加 --ablate-p0）
$P/bin/python hm_reference.py --frames $W/frames_val600.txt --out $W/ref_val600 \
    --th 0.25 --stride 2 --ablate-p0

# 分序列统计（分母 < 100 的序列只作定位，不作结论）
$P/bin/python hm_seq_stats.py $W/ref_val600/detections.json

# ① 检出级对账（600 帧走这条：板端 --detections-out 的 TSV ↔ CPU 权威臂）
$P/bin/python hm_score_board_det.py --tsv <板端 nop0_val600.tsv> \
    --ref $W/ref_val600 --dist 8

# ② 张量级对账（只对少量代表帧做，别对 600 帧跑）
$P/bin/python hm_compare_npu_cpu.py --ref $W/ref_val600 --npu <板端 dump 目录> --tag int8

# OSD 目视二次确认
$P/bin/python hm_osd.py --mode single --image <feature.jpg> --ref $W/ref_val600 \
    --npu <板端 dump 目录> --tag <stem> --out hm_single_npu.png
```

⚠️ **600 帧不要默认走 `--dump`**：dump 原始张量经 NFS 约 **2 帧/分钟 ⇒ 600 帧需 583 分钟**，
且占 1.2 GB。走 `--detections-out` 的紧凑 TSV（每帧一行）**几分钟完成**，TP/FP/FN 完全等价。
`hm_compare_npu_cpu.py` 留给**张量级**逐元素核对，用在少数代表帧上。

**读数纪律**：任何比例都必须以 `分子/分母 + Wilson 95% CI` 的形式出现（`rules.md` §三·6b）；
**F1 不是比例，不要给它配区间**，改报 P、R 的区间。工具已经这么印了，别手工简化。
⚠️ 但**「两臂区间重叠 ⇒ 不显著」对配对数据无效**，显著性必须走配对检验（`rules.md` §三·6c）。

## 八、故障排查表

| # | 现象 | 根因 | 解法 |
|:---|:---|:---|:---|
| 1 | ONNX 与 PyTorch 差得离谱（heatmap max diff **0.2801**、offset **1.677**），像是导出坏了 | `torch.onnx.export` **返回后模型仍在 train 模式**，BN 走批统计 | **在导出之前抓参考值**，或导出后 `model.eval()` 再比。同口径下应为 **5.07e-7 / 2.48e-5**（`rules.md` §三·10） |
| 2 | 指标全崩但只有纵向偏（recall 31/158 = 0.1962 → 144/158 = 0.9114） | 原始 `.txt` 标签按**源尺寸**归一，不是画布 | 换算必须是 `source_norm * src_hw * scale + pad`。⚠️ dataloader 的 `batch["bboxes"]` 是**画布归一**，两套标签源不可混用（`rules.md` §四·2e） |
| 3 | `rknn_init` 成功、第一次推理就 `failed to submit!, op id: 8, op name: Conv:/model/p0_highway/p0_dw/Conv` | **带 P0 的那份 ONNX**，其 fp16（`--no-quant`）构建在 RK3588 上提交失败，int8 正常 | **对照臂必须自己上过板**。本轮只能拿 int8 当可运行的对照臂（`rules.md` §四·2f）。⚠️ **2026-10-16 更新**：失败算子落在 `p0_highway` 支路内，**去 P0 的 fp16 实测完全正常**（见 `manu/docs/experiments/2026-10-16_npu_fp16_vs_int8.md`）。⇒ 本条的适用边界是「图含 p0 支路」，不是「这块板子不支持 fp16」 |
| 4 | 输出看着「有结构」但整体错 | 空白输入也照样给结果 | 空对照闸门：输入取反 ⇒ 两个输出张量哈希**都必须改变**；再测两次运行是否逐字节一致 |
| 5 | 模型跑起来但慢得离谱 / 分数全挤在 0.99 附近 | toolkit 与板卡 runtime 版本不对齐 | 核对 commit hash（本轮均为 `c949ad889d` / 2.3.0）。**注意报错文本相似但成因不同**（1.4.0 的 `OutputOperator` 失败 vs 本行的 `Conv` 失败） |
| 6 | 板端报 `Text file busy` | 可执行文件经 NFS 执行 | 二进制走 SSH，**数据/模型**才走 NFS |
| 7 | 板端报找不到 `/dev/rknpu` 就断言 NPU 不可用 | 驱动以 **DRM 子系统**注册，真实节点是 **`/dev/dri/card1`** | 看 `/dev/dri/card1`；`/proc/devices`、`lsmod` 查不到属正常 |
| 8 | 目标明明在图上却匹配不上（只在纵向偏） | 用了**画布坐标**当**源图坐标** | 所有打印必须显式标注坐标系；`hm_osd.py` 已把 `centre_x=` / `centre_y=` 同时按源图与画布两套打印 |
| 9 | NPU 与 CPU 的 recall 差看起来很大 | **DJI 两序列（512×512）本就召不回**，会主导任何含 DJI 帧的差值 | 先分层/剔除再看差；这批 FN 是**模型域弱点，不是移植代价** |
| 10 | 方形的 512×512 源图上，怎么调阈值都对不齐 | letterbox 里写了 `r = min(r, 1.0)`，于是方形图被**填 114 灰边**而不是**上采样到 640**。而 ultralytics val 走 `load_image(rect_mode=True)`，**先把长边缩到 imgsz**，方形图本来就要**上采样** | **删掉 `r` 的 1.0 上限**（`hm_common.py` 与板端 C++ **两处都要改**）。影响 val 的 **4682/31613 = 14.8%** 帧；修后 CPU 臂 recall 0.8143 → 0.8204 |
| 11 | 两臂 Wilson 区间大幅重叠，就写下「差异不显著」 | **两臂跑的是同一批帧，是配对数据**；区间重叠只对**独立样本**有意义 | 改用**配对 McNemar**（只看不一致帧）。本例 ex-DJI 下 12:2，**p = 0.0129 ⇒ 显著**，初版结论被推翻（`rules.md` §三·6c） |
| 12 | 板端印出 `preprocess 544817000000.00 ms` / `latency ... 0.0 fps`，看着像"板卡慢到爆" | **不是板卡慢，是计时器坏了**：`clock_gettime` + 单位换算写错，把**纳秒当成了毫秒再乘 1000**（实际是**皮秒**） | 换 `std::chrono::steady_clock` + `duration<double, std::milli>`。⚠️ **必看 `[1] clock sanity` 那行**：50 ms `usleep` 差值不落在 49–52 ms 就打印 `[BAD -- durations below are meaningless]`，**这时的时延数字一律不算数** |
| 13 | `--dump <目录>` 跑完 `rc=1`，前面几步看着都正常 | 目标目录在板卡侧不存在（NFS 目录要先建） | 先 `mkdir -p` 再跑；`board_run.py --exec` 只吃单行，用 `&&` 串起来 |
| 14 | build 日志出现 `Transpose will fallback to CPU, because input shape has exceeded the max limit, height(4) * width(102400) = 409600, required product no larger than 16384!`，模型比预期**慢一倍**（151.2 ms） | `nn.PixelUnshuffle` 的 **permute**：f=2 在 (1,16,640,640) 上被折成 **(1,16,4,102400)** 的 Reshape，随后的 Transpose 触发上限 | **depthwise conv 等价改写**（`hm_pixelshuffle.py`，四形状 `torch.equal` EXACT）。⚠️ 上限卡在**空间乘积 `H*W ≤ 16384`**（约 128×128）⇒ **151.2 → 103.7 ms**（`rules.md` §四·9e） |
| 15 | 按 14 改成 **slice + stack + reshape** 后**更慢了**，CPU 算子 4 → 10 | 改写虽然逐位精确，但**空间乘积上限与张量怎么拼无关**，仍落到同样的 409600；且**多出 6 个 CPU Slice** | 🔴 **改写张量表达式救不了布局类回退，必须改写成算术**（depthwise conv）。别在这条路上继续试拼法 |
| 16 | 板端报 `input/grid ratio 4 does not match --stride 2` | 上板的是 **stride-4 变体**，但**没显式传 `--stride`**，smoke 二进制默认 `--stride 2` | **显式传 `--stride 4`**。⚠️ 另注：`p0_highway` 内部只下采样一次（640→320），残差恒为 stride 2，**与 stride 4 结构不兼容**，该支路把整网锁死在 stride 2 |
| 17 | 600 帧对账**跑不动** / NFS 写不动 | dump 原始张量约 **2 帧/分钟 ⇒ 600 帧需 583 分钟**，占 1.2 GB；且原二进制**只在批帧数 ≤ 12 时打印检测结果**，批量模式什么都不输出 | 板端加 **`--detections-out <path>`**（每帧一行紧凑 TSV），host 侧用 **`hm_score_board_det.py --tsv <tsv> --ref <ref目录> --dist 8`** ⇒ **几分钟完成**，TP/FP/FN 等价 |

## 九、交付前自检清单

**每一条都要能当场指出证据行/证据文件；打不了勾的不许交付。**

**A. 🔴 0 个 CPU 计算算子（`rules.md` §四·9e 硬闸门）**

- [ ] build 日志搜 `fallback` ⇒ **0 条**
- [ ] build 日志搜 `Transpose will fallback to CPU` ⇒ **0 条**
- [ ] 算子表里落 CPU 的**计算算子数 = 0**（`InputOperator` / `OutputOperator` 属 I/O 边界，**不计入**）

```bash
L=/media/manu/1TB-Volume/rknn_work/hm/logs/build_nop0_final.log   # ← 改成你本次构建实际留下的日志
test -s "$L" || { echo "❌ 日志不存在/为空，闸门无法判定"; exit 1; }   # 否则 grep 会静默判过
grep -oE '^[DWI] RKNN: \[[0-9:.]+\] [0-9]+ +[A-Za-z0-9_./]+ +[A-Z0-9]+ +(CPU|NPU|GPU)' "$L" \
  | awk '{print $NF}' | sort | uniq -c
# 期望：CPU 仅 3（InputOperator + 2×OutputOperator），其余全部 NPU
```

⚠️ **`rknn_init` 成功 ≠ 全在 NPU 上**。v1 臂就是反例：`rknn_init` ✅ 但 1 个 CPU 计算算子 + 2 条 fallback。
⚠️ **计数口径**：`fallback` 警告条数与算子表的 CPU 计算算子数**必须同时过**——
v1 的 **2 条** Transpose 超限警告被 toolkit **合并进了同一个 `Reshape` 节点**，
所以「2 条警告」对应的是**1 个** CPU 计算算子，不是 2 个。

**B. 🔴 时钟自检 `[ok]`**

- [ ] `[1] clock sanity` 落在 **49–52 ms**（50 ms `usleep` 实测 **50.137 / 50.080 / 50.124 / 50.133 ms**）
- [ ] 该行**没有** `[BAD -- durations below are meaningless]`

⚠️ 出现 `[BAD]` 时，**本次所有时延数字一律不算数**，重跑后再谈性能。

**C. 诚实性闸门（四道，全过）**

- [ ] **空白输入拒答** —— 喂全黑/全零输入，不能照样给出「有结构」的输出
- [ ] **空对照** —— 输入取反 ⇒ **两个**输出张量哈希**都改变**（只看一个不算数）
- [ ] **两次运行逐字节一致** —— 同一输入跑两次，输出哈希相同
- [ ] **时钟自检 `[ok]`**（同 B）

**D. 读数纪律（`rules.md` §三·6b / §三·6c）**

- [ ] **每个比例都写成 `分子/分母 + Wilson 95% CI`**，没有裸小数
- [ ] **F1 没有配区间** —— 它不是比例，改报 P、R 的区间
- [ ] **`n_gt < 100` 的序列标 `low-n`**，只作定位不作结论
- [ ] **两臂对比走配对检验（McNemar）**，不靠「Wilson 区间重叠」判显著性
      （本轮：带 p0 vs 去 p0 **不一致帧 1/600，p = 1.0000 ⇒ 无可归因精度代价**）
- [ ] 报性能数字的运行**没带 `--perf`**（会打开 `RKNN_FLAG_COLLECT_PERF_MASK`，显著变慢）

**E. 交付物与文档一致性**

- [ ] 上板模型 = 结论里主推的那个（当前为 `trial0474_hm_nop0_int8.rknn`），md5 对得上
- [ ] 参考臂用了**它自己的** `--ablate-p0`，ONNX md5 与上板那条**相同**
      （`028f23c655f421fe61a5e43575715fca`）
- [ ] **标定清单 `$W/calib_train_features.txt` 随模型归档**（产物，不是临时文件）
- [ ] 手册里**没有把 v1（`trial0474_hm_int8.rknn`）当成交付版**的表述
- [ ] **未上板的候选（如 `s4_head1x1`）没有写成结论**，明确标注「仅导出 ONNX、int8 未建」

## 十、端到端串行 demo（原始红外 → GMC 特征 → NPU 检测）

前九节的链路都从**已冻结的三通道特征**起步；本章往前再接一段 `gmc_stream_ocl`，
把**原始红外序列**一路走到检测 —— 目前唯一一条**从像素到检出全程都在板端**的通路。
原理与取舍见 `manu/memory/heatmap_npu_trial0474.md`。

### 10.1 链路与设计取舍

```
原始红外序列/视频 --host转BMP--> NFS
  --board--> gmc_stream_ocl（CPU fit + Mali-G610 融合尾）→ 三通道特征 BMP
  --board--> rknn_heatmap_smoke（RKNN int8 去 p0）→ 检测 TSV
  --host--> hm_demo_report.py → 三层精度对齐 + 速度报告
```

⚠️ **刻意串行、不融合二进制**：两段都是**已认证的现成二进制**。
代价是多一轮特征 I/O（512×512×3 BMP ≈ **786 KB/帧**）；
收益是**每段各有独立可归因的计时器**、且**其中一段可单独替换而不动另一段**。
**不要为了"少一次落盘"去合并两个二进制** —— 合并后 speed 报告就不再可归因。

### 10.2 一条命令的完整用法

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/npu

export RK3588_PASSWORD=...          # 只经环境变量传入，绝不落盘

# 有配对 GT 的序列 → 三层精度对齐 + 速度
./run_hm_demo.sh --sequence 34_1

# 视频（本项目里的 .mp4 都是审片渲染、不是干净输入，无配对 GT）→ 只报速度与检出分布
./run_hm_demo.sh --video /path/clip.mp4 --seq-name clip --frames 120
```

| 选项 / 环境变量 | 作用 |
|:---|:---|
| `--sequence <seq>` | 原始序列目录名（走冻结特征对齐，**有 GT 时用这个**） |
| `--video <path>` `--seq-name <name>` `--frames <n>` | 视频输入；`--frames 0` = 不限；`--seq-name` 缺省取文件名 |
| `--limit <n>` | GMC 侧只跑前 n 帧（调试用；正式跑必须 0） |
| `--th <f>` | 检出阈值，默认 **0.25**，必须与 CPU 权威臂一致 |
| `--skip-build` | 跳过两个交叉编译步骤 |
| `OUT_ROOT` | host 侧 CPU 臂与报告的落盘根，默认 `/media/manu/1TB-Volume/rknn_work/hm` |
| `RK3588_PASSWORD` | **只走环境变量**；不设脚本直接退出 |

⚠️ 脚本第 6 步会检查 `$OUT_ROOT/<seq>/ref_cpu/detections.json`；
**没有就在 stdout 打印训练服务器上的参考命令并跳过 CPU 臂**，
此时第 7/8 步要拿精度结论就必须先补这个文件。

### 10.3 手工分步版（脚本失败时照这个走）

```bash
PY=/media/manu/1TB-Volume/conda-envs/rknn230/bin/python
BR=/media/manu/1TB-Volume/workspace/ultralytics/manu/pipeline/opencl/board/board_run.py
SEQ=34_1
NFS_DIR=/run/media/manu/SSD960-LOCAL/nfs/npu/out/demo/$SEQ   # host 侧新导出 == 板卡的 /mnt/manu/npu/out/demo/$SEQ（旧 /home/manu/nfs 已不导出）
BOARD_DIR=/mnt/manu/npu/out/demo/$SEQ
FROZEN=/home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val
WORK=/media/manu/1TB-Volume/rknn_work/hm/$SEQ

# ① host：原始帧 → 8bit 灰度 BMP（放进板卡可见的 NFS）
mkdir -p $NFS_DIR/raw/$SEQ
$PY - <<'PY'
import os, cv2
src = "/home/manu/mnt/data/siping/datasets/manu/anti-uav/train/34_1"
dst = "/run/media/manu/SSD960-LOCAL/nfs/npu/out/demo/34_1/raw/34_1"
for f in sorted(os.listdir(src)):
    if not f.lower().endswith((".png", ".jpg", ".bmp", ".tif")):
        continue
    im = cv2.imread(os.path.join(src, f), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise SystemExit(f"[ERROR] unreadable {f}")
    cv2.imwrite(os.path.join(dst, os.path.splitext(f)[0] + ".bmp"), im)
PY
```

⚠️ **必须是 BMP**：板端交叉构建的 OpenCV **只有内置 BMP 编解码**（不带 libpng/libjpeg），
喂 PNG 会在写文件时直接报 `could not find a writer`。

```bash
# ② board：二进制与两个 OpenCL kernel 全部走 SSH（走 NFS 会 Text file busy）
python3 $BR --put .../opencl/board/out/gmc_stream_ocl_arm64:/root/gmc_stream_ocl
python3 $BR --put .../npu/out/rknn_heatmap_smoke:/root/rknn_heatmap_smoke
python3 $BR --put .../opencl/kernels/warp_median_fused.cl:/root/kernels/warp_median_fused.cl
python3 $BR --put .../opencl/kernels/sortnet_generated.inc:/root/kernels/sortnet_generated.inc

# ③ board：GMC 阶段（原始红外 → 三通道特征 BMP + 分阶段账本）
python3 $BR --exec "
cd $BOARD_DIR && mkdir -p feat &&
/root/gmc_stream_ocl --raw-root raw --sequence $SEQ --out-dir feat --out-ext bmp \
  --fused gpu --kernel-dir /root/kernels --timing --profile-csv gmc_profile.csv \
  > gmc_run.log 2>&1
echo GMC_RC=\$? ; grep -E 'END-TO-END|budget|MD5-SUM' gmc_run.log"

# ⑤ board：NPU 阶段（特征 → RKNN int8 去 p0 → 检测 TSV）
python3 $BR --exec "
cd /mnt/manu/npu && /root/rknn_heatmap_smoke data/trial0474_hm_nop0_int8.rknn \
  --batch $BOARD_DIR/feat_named/list_board.txt --th 0.25 --stride 2 \
  --detections-out $BOARD_DIR/npu_det.tsv > $BOARD_DIR/npu_run.log 2> $BOARD_DIR/npu_run.err
echo NPU_RC=\$? ; wc -l $BOARD_DIR/npu_det.tsv"    # ← 产物 0 行即失败，必须追查（§四·9g）
```

**④ host 映射**（放在 ③ 和 ⑤ 之间，板卡按 push 索引命名输出、对账按数据集 id 找 GT）：

```bash
$PY - "$NFS_DIR/feat" "$NFS_DIR/feat_named" "$BOARD_DIR/feat_named" "$SEQ" "$FROZEN" <<'PY'
import os, sys
from pathlib import Path
feat, named, board_named, seq, frozen = sys.argv[1:6]
ids = sorted(int(Path(p).stem.split("__")[-1])
             for p in os.listdir(frozen) if p.startswith(seq + "__"))
os.makedirs(named, exist_ok=True)
for fid in ids:                       # ⚠️ dataset_id = push_index + 1（实测，见 hm_index_probe.py）
    d = os.path.join(named, f"{seq}__{fid:06d}.bmp")
    if os.path.lexists(d):
        os.unlink(d)
    os.symlink(f"../feat/{seq}_{fid - 1:06d}.bmp", d)   # ⚠️ 必须相对路径
with open(os.path.join(named, "list_board.txt"), "w") as f:
    f.write("\n".join(f"{board_named}/{seq}__{i:06d}.bmp" for i in ids) + "\n")
PY
```

⚠️ 两条**实测踩过的硬约束**：
- **`dataset_id = push_index + 1`**（`hm_index_probe.py` 逐像素比对定出来的，**不要改成 0**）；
- **symlink 必须是相对路径** `../feat/<seq>_%06d.bmp`，**清单必须写板卡侧前缀** `/mnt/manu/npu/...`。
  写成 host 的 `/run/media/manu/SSD960-LOCAL/nfs/...` 或绝对 symlink，板卡上 `stbi_load` 会**静默失败**、产物 0 字节。

```bash
# ⑥ host：检出级对账 + 三层精度对齐 + 速度（参数见 10.4）
$PY hm_score_board_det.py --tsv $NFS_DIR/npu_det.tsv --ref $WORK/ref_cpu --dist 8 \
    --out $WORK/board_perframe.json
$PY hm_demo_report.py --sequence $SEQ --speed-only --tsv $NFS_DIR/npu_det.tsv     # 无 GT 时只报速度
```

### 10.4 🔴 三层精度对齐法

**为什么必须分层**：把「板端端到端 vs CPU」压成**一个** recall 数，就把两个含义完全不同的
偏差来源混在一起了。三层各回答一个问题：

```bash
# L1 GMC 层：板端特征 vs 冻结数据集特征，逐像素（535 帧全比）
$PY hm_demo_report.py --sequence 34_1 \
  --board-feat $NFS_DIR/feat --frozen-feat $FROZEN \
  --ref $WORK/ref_cpu --perframe $WORK/board_perframe.json \
  --tsv $NFS_DIR/npu_det.tsv --gmc-csv $NFS_DIR/gmc_profile.csv \
  --gmc-log $NFS_DIR/gmc_run.log --npu-log $NFS_DIR/npu_run.log --push-offset 1

# L2 检出层：两臂各自对 GT（比例带分子分母 + Wilson 95% CI，F1 不附区间）
$PY hm_score_board_det.py --tsv $NFS_DIR/npu_det.tsv --ref $WORK/ref_cpu --dist 8

# L3 配对层：同批帧配对 McNemar（L1/L2/L3 由上面那条 hm_demo_report.py 一次出全）
```

| 层 | 回答的问题 | 偏差大说明什么 |
|:---|:---|:---|
| **L1 GMC 层** | 板端特征 vs 冻结特征，逐像素差多少 | 偏差大 ⇒ **预处理缺陷**（流式引擎 / Mali 融合尾）。偏差小 ⇒ **L2 的差异不能归因于特征合成** |
| **L2 检出层** | 两臂各自对 GT 的 P/R/F1 | 业务真正关心的量；用 `分子/分母 + Wilson 95% CI`，**F1 不是比例，不附区间** |
| **L3 配对层** | 同一批帧上两臂的不一致 | **非显著的 recall 差不得报为回归**（`rules.md` §三·6c）；「Wilson 区间重叠」不是显著性判据 |

### 10.5 已跑通的实测样例（序列 34_1，535 帧，模型 `trial0474_hm_nop0_int8.rknn`）

**L2 检出层（GT = 503）**：

| 臂 | TP | FP | FN | precision | recall | F1 |
|:---|---:|---:|---:|:---|:---|---:|
| CPU 权威臂（冻结特征 + PyTorch 去 p0） | 502 | 72 | 1 | 502/574 = 0.8746 [0.8450, 0.8992] | 502/503 = 0.9980 [0.9888, 0.9996] | 0.9322 |
| 板端端到端（Mali GMC + RKNN int8 去 p0） | 496 | 49 | 7 | 496/545 = 0.9101 [0.8831, 0.9313] | 496/503 = 0.9861 [0.9716, 0.9932] | 0.9466 |

**L3 配对层**：不一致 **6/535 = 0.0112**，NPU 独中 **0** / CPU 独中 **6**，
McNemar **p = 0.0312 ⇒ 显著**（不是"不显著所以无差异"，是**单侧劣势且显著**）。

⚠️ **这 6 帧全部是 canvas x ≤ 80.63 的贴边目标**（全序列 GT canvas x 中位数 **352.5**）。
这是**待业务裁决的弱点**，**不是**已判定的阻塞项 —— 板端链路在正常视野内的目标上
precision 不降反升（0.8746 → 0.9101）。

**L1 GMC 层**：逐帧 `max|diff|` p50 = **3** / p95 = **7** / 全局 max = **17**（uint8 满量程 255），
平均 `mean|diff|` = **1.275e-02** ⇒ **高度一致但非逐位**；这是流式引擎与离线 builder 的**既有**差异
（同量级亦见于 host Python 引擎），其下游后果由 L3 定量给出。

**速度（串行相加，两段各自计时）**：

| 段 | p50 | 备注 |
|:---|---:|:---|
| GMC 特征阶段（512×512） | **31.918 ms** | 40 ms 红线 p99 **32.9 ms ⇒ PASS**；`resid` p50 **0.033 ≈ 0** ⇒ 账可信 |
| NPU 阶段 | **56.934 ms** | `rknn_run` **55.958 ms 占 98%**，时钟自检 `[ok]` |
| **端到端** | **88.852 ms → 11.25 fps** | = GMC + NPU，**两段相加，未做流水重叠** |

🔴 **分辨率对照（实时性结论必须绑定分辨率，`rules.md` §四·9h）**：
同一二进制、同一板卡，输入换成 **1024×576** 时 **GMC fit p50 从 23.8 ms 涨到 73.4 ms**、
40 ms 红线 **FAIL（p99 91.4 ms）**，而 **NPU 阶段恒定 56.6 ms 不受影响**（模型输入恒为 640×640）。
⇒ **预处理随分辨率近似线性增长，NPU 不受影响**；**禁止拿 512×512 的时延外推到高分辨率**。
⚠️ 已知问题待修：短片段下 `[S-E2E]` 那一行的 GMC p50 仍按未封顶的 100 帧冷启动计算，
会与上方分阶段表（已封顶为 `总帧数/5`）的 GMC p50 不一致；引用端到端数时以分阶段表的口径为准。

### 10.6 ⚠️ 本轮踩的坑

| # | 现象 | 根因 / 解法 |
|:---|:---|:---|
| 1 | GMC 跑完报 `could not find a writer` | 板端 OpenCV **无 PNG 写入器**（交叉构建不带 libpng/libjpeg）。精度对齐**必须 `--out-ext bmp`**（无损） |
| 2 | 产物 0 字节 / 0 行，日志末尾没有摘要 | 清单里写了 host 绝对路径或写了绝对 symlink ⇒ `stbi_load` 失败后**静默 return 1**。**产物 0 行必须当失败追查日志全文**（`rules.md` §四·9g） |
| 3 | 图内中文全渲染成 `?????`，**且报错为零** | `cv2.putText` 是 **Hershey 字库，画不了中文**。**图内文本一律 ASCII，中文走 stdout** |
| 4 | 坐标反算出负值 | letterbox canvas→源图是 **`(canvas - src*scale)/2`**，不是 `(src*scale - src)/2`。⚠️ **这种错只有画出来才看得见** |
| 5 | 短序列上"剔除 100 帧冷启动"让 120 帧只剩 20 帧统计 | 报告工具已把 warmup 降为上限 `总帧数/5` 并**打印实际 n**；看到 `⚠️ 仅有 N 帧` 就按该样本量看待结论 |
| 6 | 管道里没报错却也没报告 | **不要用 `grep`/`sed` 过滤报告输出来判成败**，会把 traceback 一起吞掉。用报告工具自带的 **`--speed-only`**（`rules.md` §四·9g） |

### 10.7 产物清单

```
manu/pipeline/npu/out/demo34_1/   报告 report_34_1.txt + 逐帧对账 board_perframe.json
                                  + GMC 账本 gmc_profile.csv / gmc_run.log
                                  + 检测 TSV npu_det.tsv（535 行）+ 两张 OSD
manu/pipeline/npu/out/lqreview/   视频路径的速度报告 speed_lqreview.txt（无 GT，只报速度）
```

运行期的实际工作目录是 host 的 `$OUT_ROOT/<seq>/` 与板卡可见的
`/run/media/manu/SSD960-LOCAL/nfs/npu/out/demo/<seq>/`（== 板卡的 `/mnt/manu/npu/out/demo/<seq>/`）；
上面两行是**归档快照**。

OSD 用 **`hm_demo_osd.py`**（**直接吃 `--detections-out` 的 TSV**）：

```bash
$PY hm_demo_osd.py --frame 34_1__000346 \
  --feat-dir /home/manu/mnt/data/siping/datasets/manu/uav_gmc_median/images/val \
  --tsv out/demo34_1/npu_det.tsv \
  --ref $WORK/ref_cpu/detections.json \
  --out out/osd_34_1__000346.png
```

⚠️ **`hm_osd.py` 不能用于这条链路**：它只吃板端张量 dump（`_hm.bin`/`_off.bin`，
经 NFS 约 2 帧/分钟）；demo 路径走的是 TSV，对得上。

原理与全部实测数字见 `manu/memory/heatmap_npu_trial0474.md`；判据出处见 `manu/memory/rules.md` §三·6b / §三·6c / §三·10 / §三·4b / §四·2e / §四·2f / §四·9 / §四·9e / §四·9g / §四·9h。