# 红外弱小无人机检测项目精简背景

> 用途：向外部高性能模型提供项目上下文。详细实验证据以 `manu/memory/memory.md` 及 `manu/memory/` 专题文档为准。
> 更新时间：2026-10-09（龙泉山工程判据 + RK3588 板卡）

## 1. 项目目标

面向低空安全场景，研究复杂背景下 1~3 像素红外弱小无人机检测。核心评价集为 Anti-UAV 红外数据，采用防数据泄漏的场景划分和全量统一评测。

统一评测规模：24 个验证序列、31,613 帧、25,111 个 GT，匹配标准为 Distance <= 8.0 px。

## 2. 当前最高基线

### 单帧模型：Trial 0474

- 权重：`runs/optuna_p0_nas/trial_0474/weights/best.pt`
- 架构：YOLO26 Heatmap Detector，P0-NAS 微架构 `standard_dw + pixel_unshuffle + diff_only (depth=2)`
- 输入契约：`[I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]`
- 其中 GMC 在原始灰度帧上估计运动并对齐历史帧，`B_t` 为 21 帧时域中值背景
- 单帧指标：Recall 86.19%、Precision 95.57%、F1 0.9064、TP 21,643、FP 1,004、FAR 0.0318/帧
- Trial 0474 是当前永久冻结的单帧基线

### 系统级方案

Trial 0474 + 双向时空平滑 + 碎片缝合 + 高确信插补 + 刚性坏点剪枝：

- In-BBox：Recall 90.22%、Precision 94.05%、F1 0.9209、TP 22,654、FP 1,434
- 严格距离口径：F1 约 0.9194
- 交付参数：`th_base=0.22`、`th_salvage=0.06`、`th_ground=0.35`、`stitch_gap=4`、`infill_gap=3`、`min_hits_infill=5`、`min_rigid_disp=2.0`、`max_rigid_var=0.50`、`min_hits_prune=8`

## 3. 关键演进

1. 公司数据摸底后确认场景多样性和极小目标样本不足，转向公开 Anti-UAV 红外数据。
2. 建立 5-Fold 防泄漏划分和统一 24 序列评测大盘。
3. YOLO26-P2 Bbox 基线 `trial_0028`：F1 0.7631。
4. Anchor-free 高分辨率 Heatmap 范式 `uav_gpu23_heatmap`：F1 0.8331。
5. GMC + 21 帧中值背景建模显著提升输入质量，形成 Trial 22：F1 0.9057。
6. P0-NAS 搜索产出 Trial 0474：单帧 F1 0.9064。
7. 双向时空后处理和刚性坏点剪枝将系统级 F1 提升至 0.9209。
8. 自定义 DataLoader 曾造成样本腰斩和指标漂移，已修复并由官方 DataLoader 零漂移复现。

## 4. 四项代表性失败实验

1. **Tubelet 时序注意力**：空间坐标未对齐，历史目标能量无法注入；F1=0.8954，门控收敛为微负。
2. **浅层多尺度时序 NAS**：58 组搜索均停留在 F1=0.9063~0.9064，未超过 Trial 0474，说明短时局部卷积已触及信息增益天花板。
3. **点流 DP-TBD**：48 组搜索最佳 F1=0.9057；低阈值候选中的白噪声被时空串联，FP 增至 2,369，低于系统基线。
4. **亚像素积分能量守恒 Focal Loss**：单帧 Recall 短暂升至 86.32%，但破坏热图峰值曲率和时间连续性，系统级净丢 223 个真值，正式停止检测头损失微调。

其他已证伪路线：原始 3-Frame 堆叠、Hybrid Corr、960 免训练跨尺度推理、IRSTD-UNet 从零训练、PixelShuffle 直接微调、Top-Hat 零训练推理、Soft-IoU、TTA 热图平均、全参数高斯松弛微调、形态学侧支无差别微调。

## 5. 当前技术定论

- 1~3 像素点目标更适合高分辨率 Heatmap 点回归，不适合传统 Bbox 回归。
- GMC 必须在原始灰度图上估计，不能在差分图上估计。
- 输入管道永久锁定三通道语义，不得擅自改变通道顺序或破坏灰度直流基底。
- 训练和推理尺度必须物理对齐，不能直接免训练放大推理。
- 单帧模型和底模 Trial 0474 已冻结；后续突破优先从系统级时空物理后处理和候选池分析入手。
- 任何重大非继承性改动必须执行同源 Twin A/B 冷启动受控协议。
- 任何侧支实验必须底模 100% 冻结、零初始化恒等门控，并先完成 Epoch 0 零回归校验。
- 任何新实验必须使用官方 DataLoader，训练样本 43,008、验证样本 31,613，且基线需浮点级复现。
- 涉及混合分辨率数据时，原生图像坐标和 640x640 Letterbox 坐标严禁混用；不得硬编码 `imgsz=640` 进行原生坐标采样。
- 修改已有多通道数据时，已有通道必须读取官方数据集图像，禁止从 raw 重新生成导致基线漂移。

## 6. 领域迁移 Bad Case（当前业务数据发现）

以下 8 类问题是业务域落地中的核心失效模式；所列工程解法除已有输入契约外均需 A/B 和定量验证，不能直接视为已证实收益。

1. 黑热模式：`(I_t-B_t)^+` 截断暗目标响应。优先在原始灰度帧级执行 `I'=255-I`，再重新生成 GMC+21 帧中值特征；不能只反转已有通道或 HM。无元数据时动态极性识别仍待验证。
2. 悬停目标：超过 21 帧后可能被中值背景吸收。候选方案是高置信航迹驻留保护和预测位置局部背景冻结，必须防止固定坏点被锁定。
3. 近距离大目标 Heatmap 多峰：机体热部件产生多个峰。候选方案是按空间距离和连通能量聚类，以加权质心合并。
4. 近距离大目标 Bbox 漏检：差分自抵消和辐射纹理断裂。候选方案是原始灰度 320x320 Bbox 旁路，或 Track 层按同速和相对距离稳定性组装点目标。
5. NUC 温漂与固定条纹：FPN 破坏 GMC 和中值背景。候选方案是轻量列去条纹和适度 CLAHE，需验证不破坏原域基线。
6. 镜头水雾/油污：光学散射使峰值降低。候选方案是降低双门限并提高轨迹连续性约束，禁止单独降门限。
7. 吊舱大动态机动：GMC 在大视角转动、变焦和视差下失效。候选方案是按残差 MSE/内点率健康度门控，异常时静音差分和中值通道，降级纯灰度输入。
8. 野外复杂热杂波：树木、岩石和地表热斑产生持续伪峰。候选方案是按净位移/总路程比 `R` 做曲折度剪枝，并结合刚性位移约束。
9. 下沉入地/入城：从天空进入建筑、道路和高温地表后，局部背景方差暴增、GMC亚像素残差在强边缘产生 `(I_t-B_t)^+` 毛刺，目标可能被压低、粘连或使航迹中断。**首要任务是确认临界段热图/Ch1/Ch2属于完全无峰、低于 `th_base=0.22` 的残存峰，还是被边缘峰吸引**；候选方向为已确认航迹的地表ROI软保护、边缘感知残差抑制、受控运动管道积分和GMC异常降级，均待A/B验证。新增业务分桶指标：Recall、FAR、连续漏检长度、航迹中断率、恢复延迟、误保护率。

迁移验证统一保留 Trial 0474 对照，统计 Recall、Precision、FAR、连续漏检长度和轨迹误删率，并通过 Anti-UAV 原始验证集不可退化闸门。

### 新增定性结果：近距离大目标直接 160 下采样

- 原始 640x512 三通道特征直接推理时，近距离大目标存在热图能量摊平、峰值低于 `th_base=0.22` 的尺度失配问题。
- “下采样再放大回原尺寸”没有明显收益；改为整张 `[I_t, GMC_Diff, Median_Residual]` 同步缩小到 160 px 后，直接通道感知 Padding 到 640x640 输入 Trial 0474。
- Ch0 用边缘灰度均值填充，Ch1/Ch2 严格填 0，避免通道语义错误和 Padding 阶跃边缘。
- `VIDEO00005_19700101_002959` 视频中近距离无人机出现明显、稳定的定性检出，支持“尺度/感受野失配”假设；目前没有 GT 定量指标，暂不能宣称 Recall/F1 提升。
- 当前定位：近距离大目标 P1 候选旁路，需与原始分支做人工 GT A/B，统计检出率、连续漏检长度、FAR、中心偏差和峰值响应。

### Final_Labels 人工终审标注集（2026-10-09）

- 交付标注已升级为 `龙泉山/label/Final_Labels`，**取代** `pre_label/part1 + part2` 试标注。
- **扁平目录，无 part1/part2**：单层 `{sequence}__{index:06d}.txt`，共 **165,223 个**标签 + `classes.txt`（仅 `airplane`，单类 class 0），**24 个序列**，索引 `0..N-1` 连续，与 `frames_ir_jpg/<seq>/frame_%06d.jpg` 一一对应。
- 非空标签 **132,570（80.2%）**，格式 `class cx cy w h`，归一化到原生 640×512，class id 全为 0。
- ⚠️ **数据缺陷：132,592 行 vs 132,570 非空文件 ⇒ 22 个文件含完全重复行**（例 `VIDEO00002_19700101_001415__009082.txt`）；且**全库无任何一帧存在两个不同目标**。按行数统计目标数前必须去重，否则虚高。
- 新增 `manu/videos/render_gt_osd_video.py`：读扁平 Final_Labels + frames_ir_jpg，按 `--seq` 渲染「全帧 OSD + 局部放大」双联 GT 复核视频；`--list` 列出全部 24 序列；内置去重并打印 `duplicate_rows_dropped`；选中范围内标签找不到帧即 `RuntimeError`，不做静默兜底。

### 多尺度 Trial 0474 热图 + pkl cache（2026-10-09 `manu/data/multiscale_heatmap_cache.py`）

- **模型只有一个：冻结 Trial 0474**，各尺度跑同一模型；**不含 YOLO26 BBox**（YOLO 伪标签分支是 `preannotate_hm_bbox.py` 那条独立路线）。
- **融合口径对齐 `probe_heatmap_regions_video.py --fusion-row`**：各尺度热图先逆映射回**原生 640×512 网格** → `np.maximum.reduce` → 按 `main_threshold`(0.22) 二值化 → 膨胀 `fusion_dilate`(9) → 8 连通域 → `area >= min_area`(4)。逐尺度行**不膨胀**，用 `region_threshold`(0.06)。
- 每个检出 `(cx, cy, w, h, area, peak, sum)`，score = 连通域热图峰值，**按 sum 降序**（与视频脚本一致，非按峰值）。
- **Cache CSR 布局**：`frame_index`/`offset`/`points`(**float32**)/`sizes`/`scores`(f16)/`sums`/`area`(i32)/`tag`(尺度下标，**255=融合行**)，配 `iter_frames(cache)`。
- **多卡靠 `--shard i/n`**；每序列独立写 `{seq}.pkl`，无共享 manifest。
- ⚠️ 落盘序 `[I_t, diff, median]` 必须**每尺度** `[..., ::-1]` 转成模型序 `[median, diff, I_t]`；`copyMakeBorder(value=114)` 只填第 0 通道；`glob("__frame_*.jpg")` 返回 0，须写 `"*__frame_*.jpg"`。
- **尺度量化误差**（热图往返）：原生 0.0px、160 尺度 1.0px、**80 尺度 4.5px**（1 输入像素 = 8 原生像素，固有），仍 < 8px 匹配容差。

### RK3588 板卡到手 + NFS 打通（2026-10-09）

- 登录：**`ssh -p 22 root@192.168.0.64`，密码 `ematech`**（明文 root 口令，禁止写入汇报材料/代码仓库；完整规范见 `rules.md` 第二节 1b）。
- **板卡 NFS 已实测在线**（本机侧挂载）：`sudo mount -t nfs 192.168.0.64:/mnt/manu /home/manu/mnt/nfs -o nolock` ⇒ 生效为 `nfs4 vers=4.2 hard proto=tcp`；SSH 横幅 `OpenSSH_8.2p1 Ubuntu-4ubuntu0.12` ⇒ 板卡是 **Ubuntu base 的 arm64 Linux**。x86 ↔ 板卡从此有直接传产物的通道。
- RK3588 从「纯 x86 侧 C++ 移植 + Golden Reference」推进到「有真机可验」；`cpp_port_rk3588.md` 的两项未决（**L3 全量位精确**、**真实板卡时延**）有了执行载体。
- ⚠️ **板卡磁盘仅剩 1.7G 可用（16G 总量 / 14G 已用 / 89%）** —— 转 rknn、装 runtime、拷模型前必须先 `df -h /` 并给 GiB 估算（同铁律二的缓存体积红线）。
- ⚠️ **NFS 不能替代代码同步**：只暴露板卡 `/mnt/manu`，代码仓库不在其内；铁律三的自动同步仍只覆盖 x86 挂载。
- ⚠️ **板卡上没有任何 `runs/` 权重**（`runs/` 从未纳入同步链路），板卡推理必须把权重与数据双端就位列为前置检查。
- ⚠️ **PoCL/CPU 模拟时延严禁当作板卡性能**。

### 板卡 OpenCL 环境探针（2026-10-09，详见 `opencl_fused_rk3588.md` 第八节）

- **交付**：`manu/pipeline/opencl/board/`（`board_cl_probe.cpp` + `build_board_probe.sh` + `deploy_and_run.py` + README）。枚举平台/设备、解析 ICD、跑平凡 kernel 并逐元素校验。**已在板卡实跑 `PROBE RESULT: PASS` / exit 0**，传输前后 MD5 一致（`ec641df354e998b21e51a22ee016159f`）。
- **设计要点**：用 **dlopen** 而非 `-lOpenCL`——工具链 sysroot 无任何 OpenCL、SDK 只有 buildroot 配方，链接期依赖无法满足；改为运行期决定后，「无 OpenCL」变成干净报告而非链接错误，并能报出是 ICD loader 还是直连 Mali。
- ✅ **板卡 `evm3588` 实测**：Ubuntu 20.04.6 LTS，内核 `5.10.160`；8 核 = 4×A76 + 4×A55；**RAM 7.7 GiB**（⚠️ memory 里「16G/14G/1.7G」是**磁盘**，且 `/mnt/manu` 与 `/` 同一文件系统）；GPU 已就绪（`/dev/mali0`，内核内建驱动，`lsmod` 查不到属正常）。
- 🔴 **板卡 OpenCV C++ SDK 专项探测（`board_opencv_probe`，双证据一致 = 无）**：**板卡有完整编译器**（`gcc/g++ 9.4.0`、`make`、`/usr/local/bin/cmake`、`pkg-config`，**可原生编译**）；但 OpenCV C++ SDK **不存在**——`dpkg` opencv 包 0 个、`pkg-config opencv4/opencv` not found、全盘扫**头文件 0 / `libopencv*.so` 0**、`dlopen libopencv_core.so` 失败，**且实际 `g++ -lopencv_core` 两次均报 `fatal error: opencv2/core.hpp: No such file or directory`**。
  ⚠️ **`import cv2` 成功 ≠ 装了 OpenCV**：4.12.0 是 pip wheel 的 `cv2.abi3.so` 自包含模块，不带头文件/可链接库/`.pc`。
  ✅ **但 apt 有候选 `libopencv-dev 4.2.0+dfsg-5`** ——⚠️ **远低于 x86 侧 4.10.0**，直接装会破坏「调用同一批 OpenCV C++ 原语」的位精确前提，装前须做等价性验证。
  - ✅ **已解决：自行交叉编译 arm64 OpenCV 4.10.0，板卡实跑通过（2026-10-09）**，不再依赖 apt 的 4.2.0。
    - **产物**：`/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64-install`（静态库，模块 `core imgproc features2d video calib3d flann`，全关 IPP/OCL/FFMPEG/图像编解码/LAPACK/TBB），**约 4 分钟**编译完成；源码 worktree `/media/manu/1TB-Volume/workspace/opencv-4.10.0-arm64`（用户原有 5.x 树未动）。
    - **跨架构一致性实测（59 帧 × 19,333,120 元素，逐算子 dump 对比）**：`resize` / `goodFeaturesToTrack` / LK status / LK 输入点 / **RANSAC 内点掩码** = **全部位精确**；LK 亚像素位置 max **1.06e-4 像素**、RANSAC 矩阵 max **1.64e-5**；`warpAffine` **56/59 帧位精确**，19.3M 像素中**仅 11 个不同（0.000057%）**、最大 2/255；21 元素 median **38/39 帧位精确**，12.8M 中**仅 1 个不同**。
      ⇒ **判决：arm64 OpenCV 可用于特征通道**；RANSAC 判定在两架构完全一致，差异停在 float32 末位。
    - 🔴 **FMA 归因假设被实测证伪**：同架构 x86 上 `OPENCV_CPU_DISABLE=AVX2,AVX512_SKX,FP16,SSE4_1,SSE4_2` 关掉带 FMA3 的 dispatch 后，**13 个 stage 全部 59/59 位精确** ⇒ LK 差异与 FMA 无关，真实来源是**架构特定 SIMD 实现（x86 SSE/AVX vs aarch64 NEON/carotene）+ 编译器版本差 6 年（gcc 15.2.0 vs gcc 9.3.0）**，不可消除。
      ⚠️ 顺带更正我自己的错误推理：x86 参考版 `Baseline: SSE SSE2` 无 FMA，**但有 runtime dispatch 实际走带 FMA3 的 AVX2/AVX512 路径**，基线无 FMA ≠ 执行路径无 FMA。`OPENCV_CPU_DISABLE=FMA3` 无效（FMA3 不是独立 dispatch 名）。
    - 🔴 **样本量教训**：首轮只跑 23 帧，`warp_dst`/`median_out` 恰好全一致，我据此判「差异未传播」；**扩到 59 帧后被推翻**（warp 有 3 帧、median 有 1 帧存在差异）。稀疏差异必须用**跨全部元素的绝对计数**报告，**「零观测」≠「不存在」**。
    - **三个静默构建坑**：① 系统 CMake 4.2.3 会**硬拒** OpenCV 4.10 的 `cmake_minimum_required(3.1)`，须用 3.31.6（不改源码以保持与 tag 逐字节一致）；② 链接报 `carotene_o4t::split*` 未定义，实际库名是 **`tegra_hal`**，装在 `lib/opencv4/3rdparty/`（**不在 `lib/`**）；③ 即使 `WITH_JPEG=OFF`，`persistence.cpp` 仍引用 zlib，须加 **`-lzlib`**（文件名 `libzlib.a`）。
    - ⚠️ **交叉前缀更正**：板卡 glibc **2.31**，sysroot **2.29**；`aarch64-rockchip930-linux-gnu-` **无 sysroot 目录**，早期「板卡对应 rockchip930-」的推断是**错的**，应用 `aarch64-rockchip-linux-gnu-`。
    - 详见 `memory/opencv_arm64_parity.md`；工具在 `pipeline/opencl/board/`（`build_opencv_arm64.sh`、`rk3588-aarch64-toolchain.cmake`、`opencv_parity.cpp`、`gen_opencv_parity_case.cpp`、`compare_opencv_parity.py`、`push_and_run.py`）。
    - ⚠️ 跑对比脚本要用 **`/home/manu/anaconda3/bin/python`**（系统 `python3` 无 numpy）。
    - ✅ **2026-10-10：`gmc_stream_ocl.cpp` 原样交叉编译后在板卡跑通完整链路**（读 BMP → GMC fit → fused kernel → Ch0/Ch1/Ch2，**未改任何源码**）。
      **跨架构精度 PASS**：板卡 vs **x86 `gmc_stream.cpp` 纯 CPU 权威基线**同参数同输入，**Ch0/Ch1 Max|Diff|=0 全位精确**，Ch2 仅第 41 帧 1 个像素差 1 级（前后帧完全一致 ⇒ 舍入抖动非系统性漂移）；transform 矩阵 126/126 全等。**三条链路闭合**：`板卡 GPU` ≈ `x86 gmc_stream.cpp` ≈ `x86 gmc_stream_ocl`；x86 上两实现 MD5 相同互为印证。
      **性能**（重复 3 次中位数，四配置）：板卡 GPU 臂 total **156.50 ms（6.39 fps）**；板卡 CPU 臂 150.36；x86 OCL(PoCL) 176.52；**x86 `gmc_stream.cpp` 纯 CPU 68.69（14.56 fps）**。GPU 加速 **24.6×**（132.4→5.38，kernel 3.84）；只用 GPU 尾段可达 **44 fps**。
      🔴 **板卡 CPU 比 x86 CPU 慢 2.19×**（同源码/同输入/单线程）⇒ CPU 路径在板卡不划算，这是选 GPU 融合的现实依据。
      🔴 **x86 上 GPU 比 CPU 慢 0.59×**（PoCL 模拟），同二进制两硬件差 **19.8 倍** —— 再次证实 PoCL 结果不可当板卡性能。
      🔴 **当前瓶颈是 CPU 不是 GPU**：CPU 占 97%、GPU 仅 3.4%。
    - 🔴 **三条推翻旧认知的事实**：① **x86 参考实际链接系统 libjpeg-turbo 2.1.5**（非内建 3.1.2），源码树 3.0.3、板卡 1.5.2/2.0.3 —— **四者无一匹配** ⇒ 改走 **BMP**（OpenCV 内置、零外部依赖、无压缩无解码分歧，实测往返无损）；② **板卡链接期 OpenCL 可用**（`libOpenCL.so.1` 34 KB 在 ldconfig 里，探针 `context=OK`），旧记录「无 loader」不准确，`ocl_host.h` 无需改 dlopen；③ **`imgcodecs` 进 BUILD_LIST 必须加 `-DWITH_OPENJPEG=OFF`**，否则链接期报 `opj_*` undefined。
    - ⚠️ **逐算子比端到端更严格**：`opencv_parity` 里 `warp_dst` 差 11/19.3M 像素，端到端跑完只剩 **1 个**像素浮出。
    - 🔴 **本轮推翻了 `opencl_fused_rk3588.md` 的两条旧结论**：① 「无任何 arm64 OpenCV」已不成立；② 「fit 从未在板卡测过」已解决，且旧的「5 倍分歧」获裁决——**板卡实测 fit 仅 17.3 ms/帧，不是折算的 30~41 ms**；③ **真正瓶颈是 CPU 的 warp+median（86.1%），不是 fit（11.1%）也不是 GPU（3.4%）**。
    - ⚠️ **四个可复用的坑**（详见 `falsified_archive.md` 第三十七节）：**build info ≠ 运行时链接的库**（x86 实际是 turbo 2.1.5，非配置串里的 3.1.2）；**依赖版本对不上时换格式而非硬凑版本**（改 BMP）；**新增模块后必须有真调 API 的链接测试**（`imgcodecs` 的 OpenJPEG 只在链接时炸）；**`.icd` 坏 ≠ ICD 机制不可用**（loader 一直在，只是它指向的库不在）。
    - 详见 `memory/gmc_ocl_board_run.md`。
- ✅ **Mali OpenCL 实测可用**：`Mali-G610 r0p0`，OpenCL 3.0 `v1.g13p0-01eac0`，FULL_PROFILE，**4 CU**、max WG 1024、global 7902.1 MiB、local 32 KiB。**ICD 是目录式**（ARM 布局，`mali.icd`），且 **`.icd` 指向的 `libMaliOpenCL.so.1` 根本不存在**，真正驱动是 `libmali.so.1.9.0`（包 `libmali-valhall-g610-g13p0-x11-gbm`）——排查时直接 `find`，别信 `.icd`。
- 🔴 **真机推翻三条原计划**：①**无 `cl_khr_fp64`** ⇒ double-float 回退作废（有 `cl_khr_fp16`）；②**事件级 profiling 不可用** ⇒ 板卡取不到 device-side 时间，只能给 `clEnqueue+clFinish` 墙钟**上界**；③Mali **首次启动某 kernel 会现场编译**（探针 0.475 ms → 二次 0.344 ms），单次数字必须区分首次/稳态。可用扩展：`cl_khr_image2d_from_buffer`、`cl_khr_egl_image`、`cl_khr_suggested_local_work_size`、完整 subgroup 系列。
- 🔴 **NPU 当前不可用**：`/dev/rknpu` 不存在、`/proc/devices` 无 rknpu 条目、`dmesg` 无 rknpu 行；`/usr/lib/librknnrt.so`（7.26 MB）**已装但无设备节点** ⇒ 当前内核未使能 rknpu 驱动，**RKNN 路线走不通**。⚠️ **OpenCL 走 GPU 与 NPU 无关，探针 PASS 不代表 NPU 可用。**
- ✅ **板卡命令现可由 Agent 自动执行**：本机虽无 `sshpass`，但 **`pexpect` 可用**，已封装 `deploy_and_run.py`（base64 单会话传输 + 部署 + 摸底 + 运行，`--exec` 任意查询）；密码不入文件，由 `RK3588_PASSWORD` 环境变量提供。**取代**「无法非交互登录板卡」的限制。
- **四个编译期查不出的坑**（已入 `falsified_archive.md` 第三十四节）：① `CL_SUCCESS==0` 使 `!rc` 把成功读成失败；② `clEnqueueNDRangeKernel` 真实 ABI 顺序是 `(offset, size, local)`，三参数同型无从校验；③ 未检查的 `clSetKernelArg` 返回值被 **PoCL 误报为 `-52 CL_INVALID_GLOBAL_WORK_SIZE`** 而非 `-51`；④ **ICD 目录/文件布局混淆**——`fopen` 目录会成功而首读 EISDIR ⇒ 伪报「ICD 为空」，与真实故障无法区分。

### 🔴 真机精度验证定论（2026-10-09，Mali-G610，详见 `opencl_fused_rk3588.md` 8.2）

交付 `manu/pipeline/opencl/board/{gen_fused_case.cpp, board_fused_accuracy.cpp, fused_case_format.h}`：
x86 用**真实 OpenCV** 算 CPU 基准（**当时**全链路无任何 arm64 OpenCV ⚠️**该前提 2026-10-10 已不成立**，见本节后续条目），在板卡重实现 OpenCV 等于自证），
板卡无 OpenCV、只 dlopen OpenCL，跑**两个采样臂**。

- **判决一：`FUSED_USE_HW_LINEAR` 必须为 0（manual 双线性）。** 数据：Anti-UAV `01_4485_1167-2666` 640×512、**真实 GMC**、3 case、pad=67。
  - `hw-linear`：**FAIL**（Ch0 Max|Diff|=**130**、MAE 0.285、精确率仅 88.5%；Ch1/Ch2 MAE 0.31~0.42）
  - `manual`：**PASS**（**Ch0 Max|Diff|=0、100.00% 精确**；Ch1 Max|Diff|=5~6、MAE 0.0011；Ch2 Max|Diff|=1~4、MAE 0.00075）
  - **决定性证据是 Ch0**：整数坐标直读本应逐位还原却错 130 ⇒ **Mali 的 `CLK_FILTER_LINEAR` 连整数坐标都不还原原值**。
  - manual 臂 Ch1/Ch2 与已记录的 x86 基线一致 ⇒ 工具正确，差异来自硬件。
- **判决二：Mali 支持 1 字节/像素**，`{CL_R, CL_UNORM_INT8}` 可用 ⇒ 22 张 padded 图 **10.5 MiB**（CL_RGBA 为 42 MiB）；`clCreateImage`(2.0) 在 Mali 正常工作。**「CL_R8 vs CL_RGBA 带宽」项结案。**
- ⚠️ **判决三（2026-10-09 已更正）**：首轮「3.7~4.8 ms」**只圈了 kernel**（upload 在计时窗口之前、readback 在之后），**不是尾段总成本**。三段实测（manual 臂）：**upload 2.72~3.33 + kernel 3.87~5.71 + readback 0.11~0.12 = 合计 6.70~9.15 ms/帧**，upload 约占 40%。同批 PoCL 合计 131~135 ms ⇒ 真机约 **15~20×**。
   🔴 **本节合计仍不含 GMC fit**（mats 由 x86 下发）。⚠️ **此处对 fit 的判断已被 2026-10-10 实测推翻**（当时无 arm64 OpenCV；fit 实测仅 **17.34 ms/帧**，非折算的 30~41 ms）。真实瓶颈是 **CPU 的 warp+median（86.1%）**，非 fit 非 GPU。墙钟均为**上界**（事件级 profiling 不可用），首次含 JIT。
- 🔴 **顺带证伪 `ocl_host.h` 两个规格常量**：`CL_R8=0x10D0` 实为 `CL_SNORM_INT8`（非法 channel_order，正确是 `{CL_R,CL_UNORM_INT8}`）；`CL_IMAGE_OBJECT_2D=0x10F0` 实为 `CL_MEM_OBJECT_BUFFER`（应为 `0x10F1`）。**在 x86 被双重掩盖**（PoCL 拒绝→退 CL_RGBA；`clCreateImage` 失败→走不读 `image_desc` 的 `clCreateImage2D`），**上 Mali 两条同时生效**。
- **可执行文件禁止经 NFS 投递**：x86 写入后在板卡执行报 `Text file busy`，换新名也无效 ⇒ **二进制走 SSH、数据走 NFS**。
- 另有四个坑已入 `falsified_archive.md` 第三十五节：`/*__SORTNET__*/` 未拼接导致 **Ch2 静默错误而 Ch0/Ch1 完美**（已加 FATAL 闸门）、上传未按 `nch` 扩展行距致越界、`clCreateSampler` 是 **5 参数**（漏 `normalized_coords` 即参数左移崩溃）、`clSetKernelArgSampler` 在 ocl-icd 上不存在（改用 `clSetKernelArg` 传 `cl_sampler`）。
- **本地交叉编译器**：`/media/manu/1TB-Volume/rk3588/rk3588_cross_toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu`；**SDK 资料**：`/media/manu/1TB-Volume/rk3588/rk3588_sdk`。二者只放 x86 本地，严禁拷到板卡（磁盘仅剩 1.7G）；换工具链**不得绕过** `-ffp-contract=off` / 拒绝 fast-math 的浮点契约。

## 7. 领域迁移阶段收口边界

- Trial 0474 单帧 F1=0.9064；Trial 0474 加双向平滑、插补和刚性剪枝的系统 In-BBox F1=0.9209，不能混写为底模指标。
- 近距大目标直接 160 下采样 + Ch0 灰度均值/114 Padding、Ch1/Ch2 填 0，已获得稳定定性正向结果；双尺度 0.5x/0.25x、主从 NMS、`sum>=15` 能量兜底尚未完成定量 A/B，不能宣称彻底解决。
- 黑热正确路径是原始灰度逐帧反相后重新计算 GMC+21 帧中值；已有定性验证，但人工 GT 定量指标仍待补齐，无元数据动态极性识别仍属候选。
- GMC 健康度门控、NUC 列去条纹、CLAHE、曲折度剪枝、恶劣天气动态门限均为待固化防御项，尚无完整业务域定量回归。
- 所有迁移方案必须与原始 Trial 0474 做同一连续帧 A/B，并通过 Anti-UAV 原始验证集不可退化闸门。

### 待验证方案 A：原生 1:1 双拼多尺度输入

- 640x640 画布上部保留原始 640x512 三通道特征，完全不缩放，保护 1~3 像素目标。
- 画布下部左侧放同一特征的严格等比 0.25x 版本，即 160x128，用于压缩近距大目标；其余区域按通道填充：Ch0 灰度用 114/边缘均值，Ch1/Ch2 填 0。
- 区域 A 坐标保持原生；区域 B 坐标按缩放和 y 偏移反算。该方案目前只是无畸变设计，尚未完成 Trial 0474 接入、双拼训练、融合和 GT 定量验证。
- 若配合 Wh Head，Wh 只能在标注中心稀疏监督，底模保持冻结；Wh 误差和融合效果需独立报告。

### Wh/BBox 尺寸回归结果

- 已完成冻结 Trial 0474 的轻量 WhHead 四卡训练，仅新增约 13,954 个可训练参数；底模、Heatmap、Offset 和 BN 状态保持冻结。
- Epoch 0 与训练结束均完成 SOTA 对账，原始单帧指标保持 `Recall=0.861853 / Precision=0.955665 / F1=0.906338`，说明尺寸支路没有破坏原始检测器。
- 10 Epoch 后 `train_wh=0.05120`、`val_wh=0.05316`；该损失是 log(w/h) 中心点 SmoothL1，不是像素误差。
- 业务视频上估计 bbox 基本可用，但精度低于专用 YOLO Bbox；适合多峰吸收、尺度融合和 Track 关联，不应视为替代 Bbox 检测器。
- 后续以 mAP50-95、宽高像素 MAE、绝对/相对误差中位数及误差覆盖率共同评估。

### 无监督运动解耦试跑记录（2026-09-29）

测试缓存已成功构建，单卡 Optuna Trial 能启动并被正常剪枝；单卡 Trial 已完整跑通20 Epoch，无崩溃/OOM；最终 fitness 约0.063（无监督物理先验分数，不等于Recall/F1）。显著图通过 mean_target=0.05、barrier=2.0、bias=-2.8、Top-K峰值保持和梯度裁剪稳定。stride=12下龙泉山仅13,770窗口，正式缓存前需决定减小stride扩充，或接受约4.5万总量。训练已改为三阶段：先训练EgoMotionNet，再冻结相机分支训练显著分支，最后低学习率联合微调；评估改为有效batch均值，只有塌陷比例超过50%才判定Trial失败。Anti-UAV验证链路已新增：训练用原始无标签灰度序列，官方val用uav_gmc_median的BGR通道0作为I_t并读取labels/val GT；支持checkpoint、阈值扫描、Distance<=8px指标和热图可视化。首次10 Epoch单Trial结果：最佳fitness=0.063099，但官方val最佳阈值th=0.08仅TP=10、FP=211,384、FN=21,858，Recall=0.0457%、Precision=0.00473%、F1=0.0000857。结论是当前无监督显著性未学到可用无人机响应，fitness不能代表检测性能；禁止进入4096 Trial，后续需GT弱监督/显式前景约束或冻结Trial0474受控A/B。

### ST-BgNet 无监督背景重构与 GMC-Net 立项（2026-09-30）

- 新增 `manu/models/st_bgnet.py`：`STBgNet` 参数量 19,609，三阶段盲孔设计（网络只看 `W(I_{t-6}), W(I_{t-2}), W(I_{t-1})`，重建目标为 `I_t`），损失为 Charbonnier + 非对称正残差松弛（系数 0.05）+ TV。
- 数据划分已纠偏为：三源 `anti-uav` / `fpv_data` / `frames_ir_jpg` **全部只做训练**；**官方 Anti-UAV `images/val` 单独做验证**。严禁再对三源随机切分。
- 三条工程教训：①几何契约必须走 YOLO26 风格等比例 letterbox 到 640×640、padding=114，**严禁 resize 拉伸**（会把 512×512 横向拉变形）；②640×640 四帧预对齐缓存实测约 **500 GiB，已手工删除**，任何新缓存必须先给 GiB 估算；③在线 GMC 仍是当前唯一路径，已把特征估计降采样提到 1/8、特征点降到 120，四卡 batch=16 约 5~7 batch/s，CPU 为瓶颈。
- **GMC-Net 待定实验**（详见 `manu/memory/gmc_net_pending.md`）：以 4 点位移参数化 + 可微 DLT 回归单应性替代 OpenCV GMC，目标 30K~60K 参数、NPU 延迟 < 0.8 ms，攻克 CPU 算力黑洞与大晃动内点率崩塌导致 `Ch1` 撕裂引爆 FP；配套免标注真实帧评测（集合 A 大晃动失效集 / 集合 B 稳态强纹理集，指标 CCD 闭环漂移 < 2.0 px、ELE 边缘泄漏能量）。**当前仅设计定稿、代码未实现、零实测指标。**
**口径漂移已定位并修复**：根因是 `cache_trial0474_inferences.py` 把坐标存成 **float16**（letterbox 量级 256~640 处 float16 间隔 0.25~0.5 px，8 px 容差下足以推翻边界帧，致 TP −3 / FP +3）。改为 float32 + 权威抽峰 `0.08/80` 后 **`[SELF-CHECK] PASS`：TP 21,643 / FP 1,004 / GT 25,111 完全对齐**，F1 0.9063607 与标称 0.9064 仅差 −3.9e-05（四位舍入）。此前记录的 `0.9062` 作废；「抽峰深度」归因已被证伪。**权威产出脚本是 `optuna_p0_nas_distributed.py`**，不是 `train_p0_residual_highway.py`（后者 P0 默认架构不同）；memory 原记的 `val_heatmap_resolution.py` 亦不成立（未开 `use_p0_highway`，无法加载该 checkpoint）。
- **Tree 锚点级联 GMC 已判决（2026-10-08，详见 `manu/memory/gmc_net_pending.md` 第八~九、十二节）**：`W(.)` 只在锚点网格 `{2,12,22,32,42}` 直接拟合，其余 lag 由缓存 stride 步连乘（深度 ≤4），**拟合 22 → 6.19/帧（省 71.9%），但 warp 仍 22 次/帧未减少**。**权威口径三臂定稿**：native `F1 0.906361 / R 86.19% / P 95.57% / TP 21643 / FP 1004`；**tree `F1 0.906050（ΔF1 −0.000311）/ R 86.18% / P 95.51% / TP 21641 / FP 1018`**；no-GMC `F1 0.900945（ΔF1 −0.005417）`。**tree 把 GMC 精度代价压掉 94.3%**，dF1 跨 10 门限 6 正 4 负（不可区分），而 no-GMC 10 门限全负（真实退化）；tree 的缺口几乎全部来自 `DJI_0051_2` 一条，`wg2022_ir_052_split_08` dTP −74→**0**。> **作废**：旧口径 `0.9062 / 0.9060 / 0.9009` 与 `TP 21640/21639/21504` 全部作废（float16 缓存伪值，详见第十一节）；`0.9064` 是 `trial_0474/results.csv` epoch 3 冻结标称，引用须注明。> **禁止入汇报**：「几何误差 <0.2px」从未测量；「消灭 75% 算力」只有拟合省了、warp 未动。> **空对照臂角色已被流式引擎 `G2=0` 取代**，ctrl 可取消。> **RK3588 真正阻塞项是 Trial 0474 无任何 NPU 部署产物**（全仓唯一 onnx 属已止损的 YOLO26s 路线）。
- **anchor_step 端点与【阻塞中】的时延不自洽（2026-10-08，详见 `gmc_net_pending.md` 第十三节）**：**精确等价只在 `--anchor-step 2`**（锚点=每个偶数 lag、零连乘），已由流式引擎直接对冻结 native 集验证 `G0/G1/G2` 全 0、21 个锚点 lag 逐个 diff 全 0（范围 1 序列 / 198 帧 / 640×512，**非全量铁证**）⇒ step=2 指标即 **F1 0.906361 / R 86.19% / P 95.57%**，`ΔF1=0.000000`。**「短基线连乘 == 长基线拟合」不成立** —— 合成代数 bit-exact 不能当估计等价的证据。`anchor_step∈{4,6,8}` 未测（从 10 到 6 仅多花 ~3.6% 算力、深度由 4 降到 2）。> ⚠️ **本线唯一阻塞项**：时延测量差 **5 倍**（由两个 `anchor_step` 反解 **9.06 ms/次拟合** vs `stage timing` 实测 **1.80 ms/次**），机制未查明，**两套数字均禁止作为嵌入式预算依据**。> 报「等价」必须同时给出序列数 / 帧数 / 分辨率（本项目两次把局部验证称作全量）。
- **RK3588 嵌入式移植启动（2026-10-08）**：新增部署侧驱动目标，把冻结 Trial 0474 搬到 RK3588 NPU。首个决策问题即「GMC 是**替换**还是**仅加速**」，No-GMC 全量 F1 掉分是其定量输入（三臂权威口径见上）。`--stage timing` 修正了早期抽检的采样缺陷（原 8 帧全落在序列起点钳制区、`ci−2×21<0`，不代表稳态），改为每序列跨全序列均匀抽样并剔除前 2 个窗口。**代码已就绪；RK3588 真正阻塞项是 Trial 0474 无任何 NPU 部署产物**（全仓唯一 onnx 属已止损的 YOLO26s 路线）。
- **C++ 移植 L2 端到端位精确已通过（2026-10-08）**：`manu/pipeline/cpp/gmc_stream.cpp` + 验证器 `manu/pipeline/verify_cpp_port.py`（详见 `manu/memory/cpp_port_rk3588.md`）。**`--anchor-step 2` 臂在双分辨率 2 序列各 200 帧上：G1 400/400 与 Python 引擎逐位一致、G2 396/396 与冻结 `uav_gmc_median` 逐字节相同、`[MATS] max|diff|=0`（各 21 lag）、二进制自报 `fits/frm=21.00 compose/frm=0.00`（结构上印证零连乘）。即 C++ 输出 == Trial 0474 取得 F1 0.906361 所用的特征集，按构造继承 SOTA 单帧精度。** 口径：覆盖 **2/24 序列、400/31,613 帧（1.27%）**，报告自带 `[SCOPE] LOCAL check`，**是局部证据不是全量铁证**，L3 未做。**step 10 臂 G1 亦 400/400 逐位一致，但 step 10 ≠ SOTA 特征**（逐帧 400 帧中 354 帧不同，两序列均自第 23 帧起分歧，第 0~22 帧为冷启动钳制区全同），与 tree F1 0.906050 / ΔF1 −0.000311 一致；「非逐字节相同」须区分「复现 step10 精确」与「step10 等于 SOTA」两件事。
- **第四套时延数据（2026-10-08，仍不可进预算）**：C++ `--timing` 在 **Debug `-O0`** 下 median 占 **58~64%**（`DJI_0051_2` 512×512：fit 101.0 / warp 22.0 / median 216.3 / total 339.6 ms；`wg2022_ir_052_split_08` 640×512：137.9 / 40.3 / 251.0 / 429.5 ms）。瓶颈是 `gmc_stream.cpp:530-534` 逐像素 `std::nth_element` 标量循环，**加速优先级 median ≫ warp > fit**（warp 超线性 1.83x）。`fit` 折算 4.8~6.6 ms/次，落在反解 9.06 与实测 1.80 之间，**但 -O0 与优化构建不可比，无法裁决 13.4 的 5 倍分歧，该项仍是本线唯一阻塞项**。
- **两个工程教训**：①`CMakeLists.txt` 曾写 `${CMAKE_CXX_FLAGS_<CONFIG>}`（`<` 是非法变量名字符）⇒ CMake 语法错误、CLion 完全配不出来，而**逐配置槽位恰是唯一能抓到发行版往 `CMAKE_CXX_FLAGS_DEBUG` 注入 `-ffast-math` 的入口**，即浮点守卫从未生效；已修为 `string(TOUPPER ...)`，并用**反向注入测试**（注入 `-ffast-math` 必须 FATAL_ERROR）证明守卫会失败。②编译产物只在**本地** checkout（`manu/pipeline/cpp/cmake-build-debug/`），服务器挂载上**没有** `gmc_stream`，故 `--cpp-bin` 必须给绝对路径；数据集在 `/home/manu/mnt/data/...` 且序列位于 `anti-uav/train/<seq>/`。
- **OpenCL 融合内核已端到端接入流水线（2026-10-09，详见 `manu/memory/opencl_fused_rk3588.md`）**：`manu/pipeline/cpp/gmc_stream_ocl.cpp` 是 `gmc_stream.cpp` 的超集，CPU 保留 Shi-Tomasi/LK/RANSAC/锚点网格/连乘，**只有 warp + 21 帧中值 + 三通道组装这一段移到 GPU**。**fork 保真已被证明**：`--fused cpu` 的 MD5 与 `gmc_stream` 完全一致（`5e0d336b971948f631640dc983acfc23`），在此之前所有 GPU 数字都不可采信。`--fused both` 两臂消费同一批拟合（每 push 只算一次），排除「两边输入不同」的假通过。**端到端 A/B（`--anchor-step 2`，`--limit 120`）：Ch0 三序列均 100.0000% 逐位精确；Ch1 Max|Diff|=7~8 / MAE 0.0003~0.0010；Ch2 Max|Diff|=6~8 / MAE 0.0001~0.0003。** `--mode nogmc`（W=IDENTITY）下**三通道全帧 100% 逐位相同**——身份变换是最强的全链路正确性证明（padding/ring 索引/通道序/平面排布/坐标约定）。**排序网络 109 次比较，全枚举 2^21 = 2,097,152 个 0/1 输入完备验证通过**（Knuth 0/1 原理，非抽样）⇒ **验收标准写的「91 次比较」复现不出来**，根因是公开的 Batcher 奇偶归并伪码只在 n 为 2 的幂时成立。> **PoCL 上 GPU 臂比 CPU 臂慢 1.3~2.9 倍，这是预期结果不是缺陷**（PoCL 是 CPU 模拟，与 CPU 臂抢同 12 核；且手动 2×2 每采样 4 次 NEAREST 读 vs 真 GPU 1 次滤波读）。**能迁移到 RK3588 的只有传输列**：upload 0.024~0.036 ms、readback 0.325~0.347 ms 是总线流量为真实数字；kernel 112~118 ms 是 CPU 模拟属性，**不得进入嵌入式预算**。> **埋点抓到的两个测量缺陷**：①`clEnqueueNDRangeKernel` 在入队即返回，围它计时得 0.04 ms 而阻塞回读报 116 ms——计算藏在同步里，第一版「回读主导」结论完全错误，须 enqueue 后立刻 `clFinish`；②设备槽位按 `索引 % N` 键入时，冷启动被钳帧与当前帧可同余→跳过上传→**Ch0 31% 像素错、偏 80 灰阶、预热后消失**，现按内容身份分配槽位。

### 龙泉山域工程交付判据与结论（2026-10-09，详见 `memory.md` 第 5d 节）

- **判据**：F1 ≥ **88.0%** 且 Recall ≥ **85.0%**，且工作门限下虚警受控。依据项目内已成文的 `manu/diagnostics/audit_substandard_cases.py:78-81`（`--min-f1 88.0`/`--min-recall 85.0`/`--min-prec 90.0`/`--max-pure-bg-far 0.030`，三者为**与关系**）；⚠️ 该线原为 Anti-UAV **系统级**所设，不可直接套到单帧候选池。
- **实测全局**（24 序列 / 165,223 帧 / 132,570 GT，dist 8px，多尺度 r=64）：**F1 73.58% / R 69.06% / P 78.74% / FAR 0.1496** ⇒ **三指标全部不达标，Recall 线零通过**。
- **multi-scale > single-scale 已确证**：单尺度 640 F1 66.698% → 多尺度 r=64 F1 **73.581%**，**TP +17,321 / FN −17,321 / Recall +13.07**，最优 th 同为 0.30，唯一变量是尺度集合。合并半径是真超参（r=0→32 F1 52.29%→72.98%；**r=0 裸并集 FP 达 64,392 直接崩盘 ⇒ 多尺度必须配跨尺度去重**）。
- ⚠️ **FAR 0.1496/帧 = 军工光电合格线(0.030)的 5 倍**，是最大工程障碍，必须接双向平滑/插补/刚性剪枝。
- ⚠️ **8px 容差对本域偏严**：GT 中位框 49×19px，8px 已达中位最短边的 42%（p10 小框的 80%），而 8px 是照 Anti-UAV 1~3px 弱点定的。同一批预测换 inbox 口径 F1 76.47%→82.53% ⇒ **73.58% 是保守低估**。两口径都报，**严禁静默换口径**。
- **FN 必须拆解**：`fn_absent`（附近无检出=前端召回上限）/ `fn_compete`（有检出被更高分抢走=合并代价）——判断半径是否过大的唯一依据。
- **7 条 UN-PASS**（seqBestF1 远高于全局 F1 者=工作点/预处理可解；本身就低者=能力缺口）：
  `00006_000322`(63.97%，头号事故 FN 6,847) / `00008_002001`(85.16%) / `00012_002251`(**45%→97.11%@th0.62**，纯门限问题) / `00032_014453`(**64%→89.42%@th0.08**，黑热截断) / `00007_001324`(83.32%) / `00003_002111`(80.86%) / `00036_015740`(24.03%，短序列塌陷)。
- ⚠️ `00032` 的 **Precision 97.46% + Recall 47.10%** 组合与 memory §九.4 记录的黑热截断特征高度吻合 ⇒ **疑似预处理问题而非标注问题**，解法是 `I'=255-I` 反相重算特征，**不是打回标注员**。

## 8. 下一步研发方向

1. 以 Trial 0474 作为冻结单帧基线，统一所有实验口径。
2. 按 P0 优先验证黑热极性对齐和 GMC 健康度门控；按 P1/P2 分层验证其余业务 Bad Case 解法。
3. 分析系统级候选池覆盖、轨迹碎片、刚性坏点和虚警来源。
4. 对典型难例进行前端漏检与后处理误删分层诊断。
5. 新方案必须先完成零回归校验，再进行全盘指标比较。
6. 交互执行规则：每次因报错、代码修复或参数修改要求重跑，必须重新提供包含工作目录、环境变量和全部参数的完整可复制命令，不能只要求用户重跑上一条命令。

## 8. 详细资料导航

- `manu/memory/memory.md`：项目总索引、完整演进路线和当前结论
- `manu/memory/rules.md`：实验、数据、缓存、服务器和同步铁律
- `manu/memory/gmc_median_sota.md`：GMC、21 帧中值和 Trial 0474
- `manu/memory/falsified_archive.md`：失败实验与工程避坑档案
- `manu/memory/spatio_temporal_track.md`：双向时空平滑系统
- `manu/memory/hard_cases.md`：难例诊断与攻坚记录
