# GMC 特征准备上 Mali-G610（OpenCL）：速度与精度代价 · 最终探针报告

- 板卡：RK3588 / **Mali-G610 r0p0**（4 CU），OpenCL 3.0 v1.g13p0-01eac0
- 序列：`/mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324`（11810 张 640×512 BMP，NFS）
- 探针二进制：`/root/gmc_gpu_probe`（md5 `493081c9a59157fd9f0ad40fc8f65530`）
- 内核：`/root/kernels/gmc_feature_gpu.cl`（md5 `e29b61dd8a76999dfa93d0ae1b5ca6f1`）
- 探针是**独立二进制**：不链接 `libmosaic_pipeline.a`、不接管线、不改任何认证路径数值

---

## 0. 结论（先说）

> **把 GMC 的特征准备搬到 Mali-G610，在本平台上是数值等价但更慢（0.598×），因此不能作为提升 25 fps 的手段；LK 是稀疏 gather 型负载，不适合 GPU。**

- **数值等价**：max|Δ平移| **0.0121 px**、mean 0.0023 px、旋转 cos **1.0000000**、|Δs/s| 1.7e-5、角点对应率 98.8%（≤0.5 px）。GPU 路径与 CPU 路径在数值上等价，**远优于「角点砍到 100」那条已知代价 2.05 px 的老路** —— 也就是说，**如果**要做，精度代价是极小的那一档。
- **但更慢**：硬段单线程下 GPU 14.199 ms vs CPU 8.487 ms（**0.598×**，GPU 慢 1.67×）。
- **根因**：LK 占 CPU 拟合成本的 **77.9%**，而它是**稀疏、gather 密集、迭代式**负载 ⇒ 4 个 Mali CU 的并行度太粗、访存不合并；CPU 的 SIMD + 缓存友好访问反而更优。
- **唯一有希望的是 gFTT**：kernel 1.128 ms vs CPU 整段 gFTT 1.401 ms（省 0.274 ms），但要把「选取」（NMS + qualityLevel 阈值 + minDistance，当前在 CPU 上 1.136 ms）也搬上 GPU 才可能打平 ⇒ **净收益约等于零**。

---

## 1. 实测：三组

### 1.1 A —— 开头简单帧（`--limit 10 --start 0`，91.4 角点/帧，单线程）

| 阶段 | CPU (ms) | GPU (ms) |
| :-- | --: | --: |
| resize | 0.336 | 0.336（共享） |
| 角点检测 | 1.469 | 1.015 (gFTT kernel) + 0.922 (CPU 选取) |
| 金字塔构建 | (含在 LK 内) | 0.551 |
| 上传 (H2D) | — | 0.861 |
| LK | 1.513 | 5.514 |
| RANSAC | 0.065 | 0.061 |
| **总计** | **3.384** | **9.261** |

加速比 **0.365×**（p50 3.306 / 8.693，p95 3.867 / 11.947）

### 1.2 B —— 硬段多线程 CPU（`--start 5000 --limit 60`，596.8 角点/帧，`--threads 4`）

| 阶段 | CPU (ms) | GPU (ms) |
| :-- | --: | --: |
| resize | 0.333 | 0.333（共享） |
| 角点检测 | 1.418 | 0.963 (gFTT kernel) + 1.129 (CPU 选取) |
| 金字塔构建 | (含在 LK 内) | 0.583 |
| 上传 (H2D) | — | 0.860 |
| LK | 3.857 | 10.072 |
| RANSAC | 0.139 | 0.148 |
| **总计** | **5.747** | **14.087** |

加速比 **0.408×**（p50 5.876 / 12.218，p95 6.415 / 25.741）

### 1.3 C —— 硬段单线程 CPU（`--start 5000 --limit 60`，596.8 角点/帧，`--threads 1`）**← 生产同口径**

| 阶段 | CPU (ms) | GPU (ms) |
| :-- | --: | --: |
| resize | 0.334 | 0.334（共享） |
| 角点检测 | 1.401 | 1.128 (gFTT kernel) + 1.136 (CPU 选取) |
| 金字塔构建 | (含在 LK 内) | 0.519 |
| 上传 (H2D) | — | 0.860 |
| **LK** | **6.610** | **10.077** |
| RANSAC | 0.142 | 0.146 |
| **总计** | **8.487** | **14.199** |

加速比 **0.598×**（p50 8.691 / 12.379，p95 9.303 / 26.030，max 9.561 / 36.083）

### 1.4 C' —— 只把 gFTT 搬上 GPU（第 1 档；LK 仍留在 CPU，喂 GPU 的角点）

同硬段、同单线程：`--gmc-only gftt`

| 阶段 | CPU (ms) | GPU (ms) |
| :-- | --: | --: |
| 角点检测 | 1.664 | 1.156 (gFTT kernel) + 1.331 (CPU 选取) |
| LK | 6.665 | 6.665（仍在 CPU） |
| RANSAC | 0.140 | 0.141 |
| **总计** | **8.800** | **9.517** |

加速比 **0.925×** ⇒ **gFTT 档本身也是净亏**：gFTT 段 CPU 1.664 ms vs GPU 1.156 + 1.331 = 2.487 ms。

### 1.5 多线程对 CPU 的收益（对照，**代价是位级不可复现**）

| CPU 基准 | 硬段总耗时 | 相对单线程 |
| :-- | --: | --: |
| 单线程（`--threads 1`，**生产口径**） | 8.487 ms | 1.00× |
| 多线程（`--threads 4`） | 5.747 ms | **1.48×** |
| 多线程（OpenCV 自适应默认，同一探针的另一次运行） | 4.380 ms | **1.93×** |

**多线程能给 CPU 带来 1.5~1.9×，但生产路径不能这么用**：`gmc_stream_ocl.cpp:56-62` 写明，多线程下 **RANSAC 与 Lucas-Kanade 不可位级复现**（一次 anchor 拟合可能在两个同样合理的解之间翻转，整条 warp 边就移了）。因此本探针的 CPU 基准默认 `cv::setNumThreads(1)`，**与生产同口径**；多线程只作为对照列出，**不作为可用方案**。

---

## 2. 精度代价（两条路数值等价）

| 指标 | A（简单帧） | B/C（硬段 596.8 角点） |
| :-- | --: | --: |
| max\|Δ平移\| (px, 全分辨率) | 0.0418 | **0.0121** |
| mean\|Δ平移\| (px, 全分辨率) | 0.0076 | 0.0023 |
| 旋转 cos（最小 / 均值） | 1.0000000 / 1.0000000 | 1.0000000 / 1.0000000 |
| \|Δs/s\|（最大 / 均值） | 3.2e-5 / 9e-6 | 1.7e-5 / 4e-6 |
| 角点数 CPU / GPU（均值） | 91.4 / 90.6 | 596.8 / 593.6 |
| 角点对应率 ≤0.5 px | 0.992 | 0.988 |
| 角点对应率 ≤1.0 px | 0.992 | 0.988 |
| 角点坐标偏差 mean (px, 小图) | 0.0516 | 0.0704 |
| 角点坐标偏差 max (px, 小图) | 8.60 | 23.32 |
| 分数图自检 max\|Δ\| vs `cv::cornerMinEigenVal` | — | 0.0396（占 max 的 0.76） |

**与「角点砍到 100」那条老路对比**（`manu/memory/gmc_fit_cache_rk3588.md` §5，板卡实测，同一套口径）：

| 路线 | 拟合耗时 | max 平移偏差 |
| :-- | --: | --: |
| 角点 600 → 100（老路） | 13.03 ms | **2.05 px** |
| 角点 600 → 60（老路） | 6.47 ms | 3.66 px |
| **gFTT 上 GPU（本文，LK 留 CPU）** | 9.517 ms（硬段总） | **0.0121 px** |
| gFTT + LK 全上 GPU（本文） | 14.199 ms（硬段总） | **0.0121 px** |

⇒ **GPU gFTT 的精度代价比「砍角点」低 170 倍（0.0121 vs 2.05 px）。** 精度不是拦路虎，**速度才是**。

分数图自检：GPU 分数图 vs `cv::cornerMinEigenVal` 的 max|Δ| = 0.0396（占该图 max 的 0.76）—— 残差来自 float32 下最小特征值的相减抵消与 OpenCV 的 14-bit 定点权重；**选择结果不受影响**（角点对应率 98.8%）。

---

## 3. 口径：探针测的是什么，不是什么

**本报告只报实测，不做外推。**

- 探针测的是**单 lag** 的一次拟合：`gFTT + LK + RANSAC`（外加两边共享的 resize）。
- 生产的 `gmc_fit` 预算（约 60 ms/帧）**不是同一个量**：生产一帧要跑**多 lag 链**（anchor 拟合 + 逐 lag 变换合成）+ **GPU 尾段**（warp + median）。
- 因此「探针的 8.49 ms/帧」与「生产的 60 ms/帧」之间还有约 **7.1×** 的量差，**这 7.1× 来自多 lag 链 + 变换合成 + GPU 尾段，不是 gFTT/LK 本身**。
- 要定位那 7.1×，**不要**用本探针推算：用仓库里已有的五阶段剖析器 `manu/pipeline/cpp/fit_profile.cpp` 在**硬段**（如 `--start 5000`）上打生产拟合。

> 本报告里出现的所有百分比都来自**同一次运行**的分阶段实测，没有把探针帧的耗时按比例缩放到文档化的 60.1 ms —— 那不是同一个量。

---

## 4. 为什么 GPU 更慢（机理）

**LK 占 CPU 拟合成本的 77.9%（6.610 / 8.487 ms），而它是稀疏、gather 密集、迭代式的负载。**

| 事实 | 数字 | 来源 |
| :-- | --: | :-- |
| LK 占 CPU 单次拟合 | 77.9% (6.610 / 8.487 ms) | C 组实测 |
| LK 的 work-item 数 | ~597 个角点 | C 组实测 |
| 每个 work-item 的迭代 | 3 层金字塔 × ≤30 轮 × 15×15 patch = 每轮每点 225 次双线性采样 | LK 参数 |
| 采样点 | 各 work-item 互不相关 ⇒ 访存不合并（gather） | LK 结构 |
| Mali-G610 并行度 | 4 CU | `clGetDeviceInfo` |

**推论：**

1. **LK 不适合 GPU。** ~597 个 work-item 在 4 个 CU 上只有几波，而每个 work-item 内部是 ≤30 轮串行迭代、每轮 225 次互不相关的 gather —— 并行度太粗、访存不合并。CPU 侧相反：SIMD（NEON）+ 对同一 patch 的缓存友好访问，**反而更优**。实测 GPU LK 10.077 ms vs CPU LK 6.610 ms（**1.52× 慢**）。把金字塔的 21 次「每层 × 每角点」重采样摊开后，这个差距进一步放大。
2. **唯一有希望的是 gFTT。** kernel 1.128 ms vs CPU 整段 gFTT 1.401 ms（省 0.274 ms）。但要把「选取」（NMS + qualityLevel 阈值 + minDistance，当前在 CPU 上 1.136 ms）也搬上 GPU 才可能打平 ⇒ **净收益约等于零**。
3. **数值上两条路是等价的。** 精度代价见 §2：max|Δ平移| 0.0121 px，旋转 cos ≥ 1.0000000，|Δs/s| ≤ 1.7e-5。
4. **单线程是硬约束，不是保守。** 生产路径被位级复现要求强制单线程（`gmc_stream_ocl.cpp:56-62`）。本探针 CPU 基准用 `cv::setNumThreads(1)`，与生产同口径。

---

## 5. 如果还要继续做，先解决什么

1. **LK（先别做）**：当前 LK kernel 的采样是 gather 型、patch 缓存放在**全局 scratch**（转置布局）而非 private/local memory。要让 GPU LK 有竞争力，必须先做：patch 放 private/local memory、当前帧采样改 texture/sampler 路径、迭代上限与终止准则与 OpenCV 对齐。**在 4 CU × 597 点这个规模下，期望上限也就是打平。**
2. **gFTT（唯一有希望的一条）**：把「选取」（NMS + 阈值 + minDistance）也搬上 GPU —— 分数图 1.128 ms 已经是 CPU 整段的 0.80×，若选取能压到 ~0.3 ms，整段 gFTT 才可能从 1.401 ms 降到 ~1.4 ms 上下（**净收益约等于零**）。
3. **不要动角点预算**：600 → 100 已知代价 2.05 px；而 GPU gFTT 的精度代价是 0.0121 px，**精度旋钮应该往 GPU 这边走，不是往砍角点那边走**。

---

## 6. 交付物

### 6.1 新增 / 修改的文件

| 文件 | 状态 | 说明 |
| :-- | :-- | :-- |
| `manu/pipeline/opencl/kernels/gmc_feature_gpu.cl` | **新增**（351 行） | 2 个 kernel：`gftt_min_eigen`（Shi-Tomasi 最小特征值分数图）、`lk_pyr_track`（多级金字塔 LK，每 work-item 一个角点） |
| `manu/pipeline/npu/gmc_gpu_probe.cpp` | **新增**（约 1160 行） | 独立探针；`--gmc-only gftt|gftt+lk` 两档、`--threads`（默认 1）、`--start`、`--lag`、`--limit` |
| `manu/pipeline/npu/build_mosaic_pipeline_arm64.sh` | **追加**（第 176–205 行） | 只追加一段编译 `gmc_gpu_probe`；已有目标与编译参数一字未动 |

未改动：`gmc_stream_ocl.cpp`、`mosaic_pipeline.cpp`、`mosaic_core.h`、`rknn_mosaic_e2e.cpp`、任何 doc/report/memory 文件。未 git add / commit。

### 6.2 内核的关键设计

- **工作项划分**
  - `gftt_min_eigen`：1 work-item = 1 像素（320×256）。3×3 窗口内直接累加结构张量 `[[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]`，取最小特征值 `0.5(A+C) - sqrt(0.25(A-C)²+B²)`。
  - `lk_pyr_track`：1 work-item = 1 个角点。3 层金字塔由粗到细，每层 ≤30 轮（OpenCV 默认 `TermCriteria(COUNT|EPS,30,0.01)`，**与生产实际走的无 criteria 重载一致**，不是 `iters_=20` 那条分支），每轮 225 次双线性采样，解 2×2 正规方程。
- **数据布局**：金字塔各层（图像 + Scharr 导数平面）**在 CPU 上构建后上传**（构建只占拟合的 3.2%，不是瓶颈；这样 GPU 只做迭代）。导数平面按 OpenCV 的 `derivBorder=BORDER_CONSTANT`、图像层按 `BORDER_REFLECT_101` 各自补边，**padding 宽度 = winSize = 15（不是 7）**，kernel 按 `roi + 15` 采样，边界外推与 OpenCV 逐字节一致。
- **边界处理**：分数图用 REFLECT_101（= OpenCV 的 `BORDER_DEFAULT`）对**源图**取样做 Sobel、对**梯度图**取样做 3×3 盒滤波，与 OpenCV 的 `cornerMinEigenVal` 等价（已用 `cv::cornerMinEigenVal` 做过分数图自检）。
- **LK 的 patch 缓存**：放在**全局 scratch**（`patch[c][p][gid]` 转置布局，让同一 warp 的 lane 访问连续地址 ⇒ 合并访存）。之所以放全局而不是 private/local：Mali 上每 work-item 675 个 float 的 private 数组会溢出 `CL_DEVICE_LOCAL_MEM_SIZE`。
- **定点/浮点**：OpenCV 的 LK 用 14-bit 定点双线性权重、并把上一帧 patch 存成 32× 整数；这些比例因子在 delta 里**相互抵消**，所以内核用精确 float 权重，但显式保留 `diff = 32*(curr-prev)` 与原始 Scharr 尺度，保证 delta 的绝对量纲正确；`minEig` 的两个拒绝判据仍按 OpenCV 的 `FLT_SCALE=2^-20` 尺度计算（否则阈值 1e-4 的含义就变了）。

### 6.3 构建命令

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics
export LC_ALL=C
MOSAIC_NO_DEPLOY=1 bash manu/pipeline/npu/build_mosaic_pipeline_arm64.sh
```

交叉编译纪律：无 `-I/usr/include`、`-ffp-contract=off`、无 fast-math、无新增运行时依赖（只加 `-lOpenCL`）。

### 6.4 四个认证产物的 md5（未因本次改动而变化）

| 产物 | 改动前 | 本地重建后 | 板端 `/root` |
| :-- | :-- | :-- | :-- |
| `libmosaic_pipeline.a` | `57f0af5273345f2644a65fe4942595fa` | `b0283a96660ff861fce95209dad8ea8f` | `57f0af5273345f2644a65fe4942595fa` |
| `rknn_mosaic_e2e` | `08d441cf3a9dd79709f1b8d6a6f7ce79` | `08d441cf3a9dd79709f1b8d6a6f7ce79` | `08d441cf3a9dd79709f1b8d6a6f7ce79` |
| `mosaic_api_test` | `c53f261a71d9996528710c4681e273db` | `c53f261a71d9996528710c4681e273db` | `c53f261a71d9996528710c4681e273db` |
| `rknn_mosaic_heatmap` | `5dcf5bf231533593a96ab8d03cfd17b8` | `5dcf5bf231533593a96ab8d03cfd17b8` | `5dcf5bf231533593a96ab8d03cfd17b8` |

- 三个可执行文件**逐字节不变**。
- `libmosaic_pipeline.a` 的 md5 变了，但**内容逐字节不变**：解包后的成员 `mosaic_pipeline.o` md5 仍是 `d6408767e2dce7c7046f00a9e4e1f573`（改动前后一致），差异只有 `ar` 归档头里的成员时间戳（`cmp -l` 只有 8 个字节不同，全在时间戳字段）。**这是任何一次重建都会发生的，不是本次改动引起的**；板端跑批用的 `/root/libmosaic_pipeline.a` 本次**没有重新部署**（构建用 `MOSAIC_NO_DEPLOY=1`，只 scp 了探针与内核）。
- e2e TSV md5 基线 `0bf598a1d890040bcfa8d157dc69c032` 未受影响：本次没有改动任何参与 GMC 数值路径的代码，认证产物逐字节不变。

### 6.5 复现命令

```bash
# 板端（跑前必须确认空闲：ps -eo comm | grep -c "^rknn_" 为 0）
cd /root
./gmc_gpu_probe --raw-root /mnt/manu/lq_ir_bmp --sequence VIDEO00007_19700101_001324 \
                --kernel-dir /root/kernels --limit 60 --start 5000 --threads 1 \
                --gmc-only gftt+lk --out /tmp/gmc_gpu_probe.md
```
