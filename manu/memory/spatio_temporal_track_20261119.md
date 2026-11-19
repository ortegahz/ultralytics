# 时空融合（trial0080_th045）移植 + 板卡节拍实测 — 2026-11-19

> 本文件是 §180 的详细版：数据表、复现命令、坑。
> 主线结论在 `memory.md` 的 §180。

---

## 0. 一句话

时空后处理（Pass1/2/3）已完成 **C++ 移植并与 Python 参考实现逐位对齐（bit-exact）**；
**scipy 的纯 Python 重写已与 scipy 1.15.3 本体差分验证 100% 一致**；
板卡实测确认 **加这套后处理对帧率无影响（+0.015%）**，
并给出了 **RT 参数搜索要用的真实更新间隔：123 ms / Δn≈3.08 帧（npu_workers=1 口径）**。

---

## 1. 🔴 scipy 纯 Python 重写的对齐验收（用户提问："跟库对齐了吗？"）

**结论：对齐了，100%。**（2026-11-19 亲手复跑，不是转述）

| 项 | 值 |
|:--|:--|
| 对照库 | `scipy.optimize.linear_sum_assignment`（scipy **1.15.3**，numpy 2.2.6） |
| 被测 | `manu/evaluation/lsap_reference.py::linear_sum_assignment_ref`（**不 import scipy**） |
| 验收脚本 | `manu/evaluation/check_lsap_reference.py`（seed=20261119） |
| 用例数 | **4854**（27 定型 + 2000 随机 + 400 大矩阵，各 × float32/float64 两种 dtype） |
| 覆盖格点 | **371,058** |
| 不一致 | **0** |
| 一致率 | **100.000000%** |
| 异常一致性 | 3/3 OK（NaN / -inf / 全 +inf，报错文案逐字相同） |
| 反向验证 | 故意注入 tie-break bug ⇒ 一致率掉到 **67.49%** ⇒ 证明这套用例**对 tie-break 敏感**（不是"随便找个最优解"那种假一致） |

**比对口径**：不仅比「配对集合」，还比 **总代价**（`math.fsum`，顺序无关的正确舍入）；
scipy 的 tie-break 规则本身是契约的一部分（`shortestPathCosts[j] < lowest ||
(shortestPathCosts[j]==lowest && row4col[j]==-1)`），去掉它一致率会掉 32.5%。

---

## 2. 🟢 三端逐位对齐（Python 参考 / C++ host / C++ aarch64 板卡）

| 项 | 值 |
|:--|:--|
| 样本 | 板卡序列 `VIDEO00007_19700101_001324` 的 **400 帧** |
| 候选 | **1764** 条（4.41 条/帧，`--thr 0.02` 导出） |
| 输出检测 | **859** 条 |
| 三端 MD5 | `bf82d8c37cc1a7c89f54d74a3fce5440`（Python 参考 == C++ host == C++ aarch64 板卡，**三份相同**） |
| 逐字段比对 | `compare_fused.py`：离散项不一致 **0** / 数值项不一致 **0** / `max|Δpos| = 0` / `max|Δscore| = 0` |

**冻结参数（v1 / th_base=0.45）**：`manu/evaluation/system_sota_longquan_trial0080_delivery_params.json`
（另有 v3 档 `..._v3_params.json`：th_base 0.48 / th_ground 0.26 / sky_ratio 1.0 / confirm_consec 7 —— **尚未用 v3 复验**）。

### 2.1 🔴 坑：**编译标志决定位级一致性**（本次亲自踩到并复验）

同一份源码、同一份候选流（MD5 `e7f6aef24c66c416348fb98b6039cc30`，板卡与宿主一致）：

| 编译标志 | 板卡 aarch64 输出 MD5 | 结论 |
|:--|:--|:--|
| `-O2`（aarch64 默认 `-ffp-contract=fast`） | `7fa730d3d30f8eec392f3363d64ece1a` | ❌ 与 Python 参考**不一致**（FMA 合并改了舍入） |
| `-O2 -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations` | `bf82d8c37cc1a7c89f54d74a3fce5440` | ✅ 与 Python 参考 / x86 host **完全相同** |

⇒ **任何临时重编都必须带上 `FP_FLAGS`**；`mosaic_sdk/build_lib.sh` 已经带了（第 132 行 `FP_FLAGS=...`，
第 192 行 `LIB_CXXFLAGS` 引用），所以交付构建本身没问题；**但 `build_lib.sh` 目前只编
`src/npu/mosaic_pipeline.cpp` ⇒ 接入时空融合后必须把 `mosaic_track_fusion.cpp` 加进去**。

**C++ 移植文件（SDK 内）**
- `rk3588_app_base/mosaic_sdk/src/npu/mosaic_track_fusion.h`（零第三方依赖）
- `rk3588_app_base/mosaic_sdk/src/npu/mosaic_track_fusion.cpp`（866 行：Pass1 卡尔曼 + Pass2 缝合/插补/剪枝 + Pass3 连续确认 + scipy 等价 LSAP）

---

## 3. ⭐ 板卡实测：检测器 → 跟踪器的更新间隔与帧率波动（本次主任务）

### 3.1 测法（工具：`rk3588_app_base/mosaic_sdk/src/npu/mosaic_cadence_probe.cpp`，板卡原生编译）

- **与嵌入式同构**：生产者线程按源帧率 `mosaic_pipeline_push_frame(带自己的帧号)`（相机帧号），
  消费者线程阻塞 `mosaic_pipeline_collect()`。
- 记录每个结果的 **(到达时刻, 帧号, 检测数)** ⇒
  - 结果间隔 = 跟踪器拿到的**物理更新间隔**；
  - 帧号步长 **Δn** = 跟踪器看到的**源帧数间隔**（= 两次更新之间被丢掉的源帧数）。
- 探针只用 SDK 的 .so + 自带 BMP 读取器，**零 OpenCV 依赖**。
- 宿主端分析脚本 `src/npu/analyze_cadence.py`（重算间隔/Δn 分布/乱序计数）。

**环境**：ematech 开发板 `192.168.0.64`（EVM3588 / Ubuntu 20.04 / kernel 5.10.160 / 8 核 / 7.9 G），
模型 `/tmp/p0_off_A_none.rknn`（trial_0080），序列 `VIDEO00007_19700101_001324`（640×512 IR BMP），
**`npu_workers=1`**（本板 2 个 NPU context 会触发 RKNN 6 s 超时 → NPU soft reset，见 memory §179.4）。

### 3.2 主结果表

| 配置 | preset | 源帧率 | 输出 fps | 间隔 mean | p50 | p90 | p95 | p99 | max | **Δn mean** | Δn 分布 |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| rt25 | realtime | 25 | **8.13** | 123.0 | 119.4 | 127.7 | 152.5 | 178.4 | 185.6 | **3.04** | 1:2% 2:20% 3:55% 4:18% 5:4% |
| F | realtime | 25 (30 s) | 8.08 | 123.8 | 120.3 | 132.3 | 155.1 | 170.3 | 172.8 | **3.08** | 1:2% 2:13% 3:63% 4:17% 5:4% |
| B | realtime | 30 | 8.13 | 123.1 | 118.8 | 135.7 | 161.2 | 172.3 | 189.7 | **3.65** | 1:1% 2:13% 3:29% 4:38% 5:16% 6:3% |
| E | throughput | 25 | **8.41** | 118.9 | 118.7 | 121.6 | 122.3 | 125.6 | 126.0 | **2.43** | 1:16% 2:38% 3:33% 4:11% 5:1% |
| D | baseline | 25 | 4.28 | 233.9 | 160.5 | 441.5 | 476.0 | 530.9 | 533.4 | 5.96 | 1:1% … 13:4% |
| C2 | realtime | 不限速(30 s) | 6.57 | 152.2 | 143.6 | 205.1 | 217.2 | 240.5 | 267.1 | — | 相机 4027 fps 灌满队列 |

### 3.3 分级耗时（realtime 档，600 帧 @25 fps）

| 阶段 | 指标 | 数值 |
|:--|:--|:--|
| **NPU（关键路径）** | rknn_run p50 / p95 / p99 / max | **118 / 121 / 125 / 128 ms** |
| GMC 拟合 | p50 / p95 / p99 / max | 17.5~36 / 128~131 / 155~158 / 178~191 ms（**长尾是抖的主因**） |
| 后处理（回投影） | mean / stddev / p99 / max | **0.58 / 0.26 / 1.14 / 1.29 ms** |
| 丢帧 | camera-queue / features / detections | **0 / 0 / 0**（`fallback summary: clean`） |
| 调度 | opportunistic scheduler 跳过 | 600 帧里 **375 组被算、225 帧被跳**（引擎忙时抓最新帧） |

**读法**：输出周期 123 ms ≈ NPU 的 118 ms ⇒ **NPU 是瓶颈**；
realtime/throughput 档靠 `decoupled_stages=1` + 绑核把 GMC 与 NPU 解耦，两者重叠；
baseline 档三段串行 ⇒ 4.3 fps 且长尾恐怖（p95 476 ms，GPU upload 64.8 s 计入关键路径）。

**帧号无乱序**：所有 6 组配置里「后到结果帧号更小」计数 = **0**
⇒ 嵌入式日志里 `57218→57220→57221→57224→57222→57225→57228` 的乱序来自 **2 个 NPU worker**（本板跑不了）。

### 3.4 ⭐ 加时空后处理的时间成本（RT 搜索要用的常数项）

板卡实测（工具：`src/npu/track_fusion_bench.cpp`）：

| 项 | 板卡 aarch64 | 宿主 x86-64 |
|:--|:--|:--|
| Pass1 `Step()`（每帧，400 帧候选 / 4.41 候选·帧） | mean **8.1 µs** / p99 19.8 µs / max 111 µs | mean 1.3 µs / p99 2.3 µs |
| Pass1 `Step()`（2000 帧候选 / 10.73 候选·帧） | mean 3.2 µs / p99 17.2 µs / max 35.9 µs | — |
| 在线输出 `OnlineFused()` | mean 0.37 µs / max 9.9 µs | mean 0.09 µs |
| `Finalize()`（Pass2+3，整段 400 帧一次性） | 0.61 ms | 0.23 ms |

> ⚠️ 上表是**严格 FP 标志**（`-ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations`）下测的，
> 与 bit-exact 那份二进制是同一份编译产物（见 §2.1 的坑）。

**结论（可以直接写进仿真器）**
- 相对帧周期 123 ms：**+0.015%**；
- 相对现有后处理段 0.58 ms/帧：**+3%**；
- ⇒ **加时空后处理不会改变帧率，也不会改变跟踪器的更新间隔**；参数搜索里它是常数开销，可直接忽略。
- `Finalize()` 是**批处理**口径（离线参考实现）；RT 路径按在线口径走 `Step()+OnlineFused()`。

---

## 4. 🔴 与既有文档假设的差异（影响仿真口径，必须先对齐）

用户给的仿真设计文档假设板卡 **~14 fps / 70 ms / Δn≈1.78**。本板实测是 **8.1 fps / 123 ms / Δn≈3.08**：

| 项 | 文档假设 | 本板实测（npu_workers=1） | 差因 |
|:--|:--|:--|:--|
| 板卡吞吐 | ~14 fps（70 ms） | **8.1 fps（123 ms）** | 文档的 14 fps 对应 **CANON28 的 2 个 NPU worker**；本板 2 context 会 NPU soft reset（§179.4），只能 1 个 ⇒ 吞吐减半 |
| 输入:处理比 | 25/14 ≈ 1.78 帧 | 25/8.13 ≈ **3.08 帧** | 同上 |
| 常规 Δn 分布 | {1:35%, 2:50%, 3:15%} | **{1:2%, 2:20%, 3:55%, 4:18%, 5:4%}** | 同上 |
| 高压 Δn 分布 | {1:15%, 2:55%, 3:30%}（11.6 fps） | 30 fps 源：{1:1%, 2:13%, 3:29%, 4:38%, 5:16%, 6:3%}（8.13 fps） | 同上 |
| 乱序注入 | 8% 概率交换相邻帧号 | 实测 npu_workers=1 **无乱序**（乱序来自 2 worker） | 乱序项只在 2 worker 口径下有意义 |

**给参数搜索的建议（不改搜索空间，只改缩放）**
1. `dt` 一律写成 **Δn 帧**单位（物理帧差），不写毫秒；
   `dt_phys = 0.04 s × Δn`（25 fps 基准）。
2. 搜索空间保留两套工况：
   - **口径 A（实测，npu_workers=1）**：Δn ≈ {1:2%, 2:20%, 3:55%, 4:18%, 5:4%}，更新间隔 123 ms；
   - **口径 B（文档，2 NPU worker）**：Δn ≈ {1:35%, 2:50%, 3:15%}，更新间隔 70 ms。
   两套只差 Δn 缩放，`min_hits` / `max_age` / `R_base` / `R_slope` / `alpha` 的**搜索区间不变**。
3. `max_age` 用 **帧**（源帧数）还是用 **更新次数** 必须先定死：本实现里 `max_age=3` 是「**更新次数**」
   （两次更新之间），不是源帧数 ⇒ 换算：3 次更新 ≈ 3×Δn ≈ 9 个源帧（Δn=3.08 时）。
   → **待用户 RT 指令时确认口径**（见 §6 待办）。
4. `alpha` 衰减按 `score*alpha^Δn + curr_score` 走（文档口径），与本实现当前
   `score = 0.65*score + 0.35*new_score`（常数混合）**不一致** ⇒ RT 集成时要改。

---

## 5. 工具与产物清单（本次新增/复用）

| 文件 | 作用 |
|:--|:--|
| `rk3588_app_base/mosaic_sdk/src/npu/mosaic_cadence_probe.cpp` | **新增**：节拍探针（输出节拍 + Δn + 分级耗时 + 丢帧计数） |
| `rk3588_app_base/mosaic_sdk/src/npu/analyze_cadence.py` | **新增**：宿主端结果 TSV 分析（间隔/Δn/乱序） |
| `rk3588_app_base/mosaic_sdk/src/npu/track_fusion_bench.cpp` | **新增**：融合引擎逐帧耗时 bench |
| `rk3588_app_base/mosaic_sdk/src/npu/mosaic_track_fusion.{h,cpp}` | 时空融合内核（上一轮新增） |
| `rk3588_app_base/mosaic_sdk/src/npu/track_fusion_align_test.cpp` | 精度对齐工具（上一轮新增） |
| `rk3588_app_base/mosaic_sdk/src/npu/mosaic_det_dump.cpp` | 候选流导出（上一轮新增） |
| `ultralytics/manu/evaluation/lsap_reference.py` | scipy LSAP 的纯 Python 移植（不 import scipy） |
| `ultralytics/manu/evaluation/check_lsap_reference.py` | 差分验收脚本（100%） |
| `ultralytics/manu/evaluation/sdk_track_fusion_ref.py` | 候选流 → 融合结果（复用生产 `run_system_sota`） |
| `ultralytics/manu/evaluation/compare_fused.py` | 两份 fused.tsv 逐帧逐字段比对 |

**板卡上的产物**：`/root/mosaic_cadence_probe`、`/root/track_fusion_bench`、`/tmp/cadence_{rt25,B,C2,D,E,F}.tsv`、
`/tmp/cands400.tsv`、`/tmp/cands2000.tsv`（宿主副本 `/tmp/cadence/`）。
**宿主分析输出**：`/tmp/cadence/runs1.log`、`/tmp/cadence/runs2.log`。

**复现命令（板卡）**
```bash
# 节拍（realtime 交付档，25 fps 源）
LD_PRELOAD=/root/mosaic/libstatcompat.so \
LD_LIBRARY_PATH=/root/mosaic/opencv:/root/mosaic:/lib:/lib/aarch64-linux-gnu \
/root/mosaic_cadence_probe --preset realtime --src-fps 25 --frames 600 --warmup 40 \
    --npu-workers 1 --out /tmp/cadence_rt25.tsv
# 融合耗时
/root/track_fusion_bench --cands /tmp/cands400.tsv --reps 5
```

---

## 6. 待办 / 风险（**等用户 RT 指令**，本轮不动）

1. **RT pipeline 接入**（`mosaic_pipeline_set_track_fusion` / `collect_tracked`）尚未做：
   设计稿见 memory §180；接入点 `post_loop()`，pipeline 级单例 + mutex。
2. **候选池口径**：Python 参考用**多尺度 merge 池**（`merge_scales`，dedup_radius=768），
   SDK 是**单尺度 640×512 CCL + tag 0/1/2** ⇒ th045 的阈值是在 Python 池上标定的，
   SDK 侧**必须重新标定**。⚠️ 参考实现的 `points` 缓存真实坐标空间（是否 1920×1080 原图空间）
   **尚未最终确认** ⇒ 不要臆断。
3. **DT/关联门限口径**：`max_age` 按更新次数还是源帧数？`alpha` 用常数混合还是 `alpha^Δn`？
4. **v3 参数档**（th_base 0.48 / th_ground 0.26 / sky_ratio 1.0 / confirm_consec 7）尚未复验三端对齐。
5. **2 NPU worker 的 14 fps 口径**无法在本板复现（会 NPU soft reset，需物理断电）；
   若要按文档口径搜索，只能用服务器仿真按 §4 的口径 B 造帧号序列。
6. **板卡负载铁律**：本轮板卡只跑了单路探针（8 核，load ≈ 1.1~1.4），
   未开多进程压测；host 侧 load 1.35 / available 4979 MB ⇒ 🟡 ⇒ 全程未开 sub-agent。

---

## 7. 🟢 2026-11-19 追加：SDK 接入完成 + A/B 耗时/帧率波动实测

> 主档见 `memory.md` **§181**；原始数据 `manu/memory/cadence_ab_20261119/`。

### 7.1 SDK 接入（已落地，板卡已装）

```
mosaic_pipeline_track_fusion_defaults(cfg)   // 冻结默认值（单一真源 = C++ 内核）
mosaic_pipeline_set_track_fusion(p, cfg)     // 指针入参；NULL = 关
mosaic_pipeline_collect_tracked(p, &fid, out, n)  // 在线逐帧
mosaic_pipeline_collect_fused(p, &fid, out, n)    // 离线（flush 后）
```

板卡复验（400 帧）：`cands` MD5 `e7f6aef2…` / `tracked` MD5 `4bb83aca…` / `fused` MD5 `bf82d8c3…`
⇒ 与 Python 参考**逐位一致**（`compare_fused.py --strict` PASS）。

### 7.2 复现命令（板卡 192.168.0.64）

```bash
# 编译（板卡原生 aarch64；注意 -ffp-contract=off 必须带）
g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations \
    -fPIC -I/root/sdk/include -I/root/sdk/include/rga -I/root/sdk/src/cpp \
    -I/root/sdk/src/opencl/host -I/root/cv480inc -c /root/sdk/src/npu/mosaic_pipeline.cpp
# ... 同标志编 mosaic_track_fusion.cpp，再 -shared -Wl,-soname,libmosaic_pipeline.so 链接

# A/B 节拍（240 s、25 fps 源、npu_workers=1、realtime、thr=0.02）
export LD_PRELOAD=/root/mosaic/libstatcompat.so
export LD_LIBRARY_PATH=/root/mosaic/opencv:/root/mosaic:/lib:/lib/aarch64-linux-gnu
/root/mosaic_cadence_probe --dir /mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324 \
  --model /tmp/p0_off_A_none.rknn --kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels \
  --seq VIDEO00007_19700101_001324 --duration 240 --src-fps 25 --warmup 40 --frames 0 \
  --npu-workers 1 --thr 0.02 --preset realtime --fusion 1 --reorder-window 0 \
  --out /tmp/cad_on.tsv
# 加 --fusion 0 / --reorder-window 2 得到另外两臂
```

### 7.3 实测结论（供 RT 参数搜索直接引用）

| 臂 | fps | 间隔 mean/p50/p95/p99 (ms) | Δn 均值 | 后处理段 mean (ms) |
|:--|--:|:--|--:|--:|
| 融合关 | 8.058 | 124.17 / 119.33 / 162.50 / 209.73 | 3.102 | 2.156 |
| 融合开 | 8.033 | 124.57 / 119.63 / 162.00 / 210.87 | 3.112 | 2.065 |
| 融合开 + reorder2 | 8.058 | 124.17 / 119.51 / 158.07 / 207.65 | 3.102 | 2.113 |

* **加时空后处理对帧率与更新间隔零影响**（融合自身 8 µs / 更新，帧周期 124 ms）。
* 跟踪器更新间隔 = 检测输出间隔（逐项相同）⇒ 融合不引入额外延迟。
* Δn 分布实测：`1:2.2% 2:19.3% 3:50.7% 4:22.1% 5:4.4% 6:1.2%`（均值 3.11）⇒ 仿真**必须保留 1..7 的长尾**。
* 文档口径（2 NPU worker，Δn≈1.80/2.15）与本板实测差 **1.73×**（吞吐比），搜索时只改缩放。

### 7.4 🔴 搜索前必须先统一的口径（引擎现状）

引擎 `Step()` 收的是**稠密更新序号**，运动模型固定 **dt = 1 帧**，`max_age`/`min_hits`/`alpha` 衰减
都按**更新**计 ⇒ 与"物理帧口径"差 Δn 倍（实测 ~3.1）。
可选修法：(a) 引擎加 `dt`（Python 参考同步改，保持逐位对齐）；(b) 仿真侧按 Δn 折算参数（快但失真）。

### 7.5 🟢 独立短视频对齐（2026-11-19 补）

换一条视频（`VIDEO00009_19700101_001531_640x512.mp4`，7.64 s / 191 帧），**从交付 mp4 现场解码**成
640×512 灰度 BMP（BMP 头与 `lq_ir_bmp` 完全同格式），在板卡 0.64 上跑融合管线：

```
191 帧 / 645 候选 / 214 融合检测
```

**四路输出 MD5 全同（`93a7d5ec119ce2d52d0745ccccbe9f64`）**：
Python 参考（离线/在线）、板卡 C++（`collect_fused` / `collect_tracked`）、
板卡 gcc 9.4 编译的 align 工具、**交付工具链 gcc 9.3 交叉编译**的 align 工具。
`compare_fused.py --strict` → **PASS (identical)**。

复现：
```bash
# 1) mp4 -> BMP（host，cv2）
python mp4_to_bmp.py <in.mp4> <out_dir>            # 640x512 灰度 BMP
# 2) 送到板卡 /tmp，跑管线（板卡）
./mosaic_track_dump --dir /tmp/vid09_bmp --model /tmp/p0_off_A_none.rknn \
  --kernel-dir /mnt/manu/mosaic_sdk/runtime/kernels --frames 191 --fusion 1 --thr 0.02 \
  --out-prefix /tmp/v09
# 3) host 上跑 Python 参考 + 严格比对
python sdk_track_fusion_ref.py --cands /tmp/v09/v09_cands.tsv \
  --params system_sota_longquan_trial0080_delivery_params.json --out /tmp/v09/fused_py.tsv
python sdk_track_online_ref.py --cands /tmp/v09/v09_cands.tsv \
  --params system_sota_longquan_trial0080_delivery_params.json --out /tmp/v09/online_py.tsv
python compare_fused.py --a /tmp/v09/fused_py.tsv --b /tmp/v09/v09_fused.tsv --strict
```
原始 TSV：`manu/memory/cadence_ab_20261119/vid09_align/`。

---

## 8. 🟢 2026-11-19 追加：Δn 口径统一 + 超参搜索 + C++ 落地 + 板卡实跑逐位对齐

> 主档见 `memory.md` **§183**（§183.1~§183.11）。本节只留「时空后处理」视角的要点。

### 8.1 结论一句话

板卡实测 **8.03 fps / Δn 均值 3.11**（源恒定 25 fps）下，**冻结参数走物理帧轴 F1 = 0.00%**
（`confirm_consec=6` + `confirm_gap_tol=0` 的「逐帧连续」假设把轨迹全杀光）；
改成 **`confirm_consec=2` + `confirm_gap_tol=6` + `max_age_phys=10` + `r_slope=3.5` + `alpha_decay=1.0`**
（`match_dist` **保持 16.0** 不动）后，24 序全量复核 **s1 94.69 / s2 94.24 / s3 92.44 / s4 94.77**，
对比逐帧上限 **95.33** ⇒ **−0.65 pp**，FAR **0.0281** 仍在 0.03 红线内 ⇒ **可接受，已 apply C++**。

### 8.2 引擎侧新增语义（`mosaic_track_fusion.{h,cpp}`）

| 参数 | 含义 | 新默认 |
|:--|:--|:--|
| `max_age_phys` | 航迹保留的**物理帧数**（`fid - last_hit_fid > max_age_phys` 即销毁） | 10 |
| `r_slope` | 每多跳 1 帧，关联半径放宽 `r_slope` px（`R = R_base + r_slope·(Δn−1)`） | 3.5 |
| `alpha_decay` | 单帧(40 ms)置信度保留率，按 `Δn` 衰减 | 1.0（不衰减） |
| `use_physical_frame_id` | 引擎是否吃**物理帧号**（否则按稠密更新计数） | 1 |

配套：`Predict(dt = 0.04·Δn)` 状态外推、乱序守卫（`fid <= last_fid` 丢弃）、
`mosaic_pipeline` 侧 **`FusionAxisPicker`**（物理帧号优先；迟到帧原样喂由引擎丢弃；
`id<0` 或同 id 连续 3 次 ⇒ 退回稠密轴）。

### 8.3 板卡实跑逐位对齐（0.64 / OpenCV 4.8 / aarch64）

* 本板**不能跑 4.6 交付包** ⇒ 板上原生重编（g++ 9.4 + `/root/cv480inc` + `/root/mosaic/opencv`）
  ⇒ `libmosaic_pipeline.so` md5 `96fb8e98…`（已装 `/root/mosaic/`）。
* **稠密臂**（VIDEO00007 前 400 帧）：1764 候选 / **890** 融合检测；
  板卡 fused md5 **与 Python 参考同为 `d5af6923…`** ⇒ 逐位一致。
  （`cands.tsv` md5 `e7f6aef2…`，与 §7.1 记录相同 ⇒ 检测通路未被改动。）
* **跳帧臂**（按实测 Δn 推 400 次更新，物理帧号 0…1202）：5340 候选 / **1084** 融合检测；
  `compare_fused.py --strict` **PASS(identical)**（离散 0、数值 0、max\|Δpos\|=0、max\|Δscore\|=0）。
  ⚠️ 唯一差异是**格式**：板卡 `collect_fused` **不输出空帧**（652 行）而 Python 每帧都写（1203 行）；
  去掉空帧行后两边 652 行**逐位相同**（检测数都是 1084）。
* 🛠 新增 `mosaic_track_dump --ids <file>`：每帧的 `client_frame_id` 可指定 ⇒ 板卡上复现跳帧；
  `cands.tsv` 帧号列随之写物理帧号 ⇒ 可直接喂 `sdk_track_fusion_ref.py --physical-frame-index`。

### 8.4 交付包（4.6 口径，给嵌入式组）

* `out/libmosaic_pipeline.so` md5 **`0baaaaecd12af82606601ce685384e80`**（NEEDED 全 `libopencv_*.so.406`）。
* 落点：NFS `/run/media/manu/SSD960-LOCAL/nfs/mosaic_sdk`（76 文件）+ `rk3588_app_base/mosaic_sdk/`。
* 🔴🔴 **版本握手**：`MOSAIC_TRACK_FUSION_VERSION` **1 → 2** ⇒ 嵌入方**必须用新 `mosaic_pipeline.h` 重编**，
  否则 `mosaic_pipeline_set_track_fusion()` 返回 **`-2`**，融合**静默不生效**（不崩，只是没开）。

### 8.5 待办

* ⬜ 真 th045 底座复跑（服务器现存 24 序缓存是 `cs_t0043`；trial_0080 的 th045 缓存已不存在，需 GPU，本轮未启动）。
* ⬜ **RT pipeline 参数搜索**（时空后处理/跟踪器寻优）：**等用户 RT 指令**再做。
* ⬜ 候选池口径（Python 多尺度 merge 池 vs SDK 单尺度 640×512 CCL）：仍未标定。
