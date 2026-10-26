# §五十六　RK3588 PTQ 量化分布调优（2026-10-22 晚，新任务）

> 目标：不启动 QAT，用**校准集分布设计 + RKNN-Toolkit2 量化算法探索**，找到使红外弱小目标
> （盲测死穴 `001324`）量化损失最小的 PTQ INT8 模型。**PC 的 PT(PyTorch) 版本是唯一精度参照**。

### 十六.1 三个推翻初始假设的实测结论

| 初始假设 | 实测结论 | 证据 |
|---|---|---|
| trial_0080 是 YOLO 检测器，需改 C++ | **它是热图模型**（`YOLO26HeatmapDetector`，ckpt 带 `p0_kwargs`、`model` 是 `b0.conv.*` state_dict、训练脚本 import `HeatmapLoss`）⇒ 现有 `hm_export_onnx.py → hm_convert_rknn.py → rknn_heatmap_smoke` **零改动可用** | ckpt 结构 + `manu/training/optuna_mix_ratio_search.py` |
| rknn-toolkit2 用最新的 2.5.0 | **2.5.0 必崩**：`rules/matmul.py:1539` 的 SDPA 融合规则 `IndexError`，`enable_flash_attention=False` 也崩 ⇒ 必须用 **2.3.2**（与板端 `librknnrt_2.3.2.so` 同版） | 2.5.0 两次失败 / 2.3.2 成功 |
| 门禁基线是带 P0 的 | **三轨全部 `--ablate-p0`**（`auto_epoch_gate_check.py` 第 242/274/396 行）⇒ memory 里的 T1/T2/T3 数字都是 **p0_off** 数字，只有 p0_off 能直接对账 | 源码 + memory §3 |

### 十六.2 阶梯校验链（逐级可归因，用户指定）

```
L0  PT   p0_on   float32  PC      原始模型（带 P0）
L1  PT   p0_off  float32  PC      去 P0            Δ_P0   = L1-L0
L2  ONNX p0_off  float32  PC      导出保真        Δ_onnx  = L2-L1  (实测 5.5e-7)
L2' ONNX p0_on  float32  PC      （镜像级）
L3  RKNN f16     float16  板端    float16 臂       Δ_f16   = L3-L2
L4  RKNN int8    int8     板端    量化臂           Δ_int8  = L4-L3
```
每级同帧同解码器（`extract_peaks_np` 是板端 C++ `extract_peaks` 的 1:1 numpy 移植）同 GT，
⇒ 相邻两级之差**只**归因于那一步变化。脚本：`manu/pipeline/npu/ptq/05_ladder_verification.py`。

### 十六.3 🔴 参照臂锚点（**用户要求：用 BMP 复现 memory 里记录的精度**）

两重独立证明，全部通过：

| 证明 | 方法 | 结果 |
|---|---|---|
| **像素级** | 对 4,000 帧抽样，`cv2.imread(BMP)` 必须 **逐像素** 等于 `cv2.imread(JPG)` | **mismatches=0, max_abs_diff=0** |
| **端到端** | 把 BMP **软链成** `<seq>__frame_%06d.jpg`（OpenCV 按文件头而非扩展名选解码器）⇒ 现有 `multiscale_heatmap_cache.py` **零改动**读 BMP 重建缓存，与门禁存储缓存 `gate/epoch3/ms_cache/001531.pkl` 逐帧比对 | **identical_frames = 191/191** |

⇒ **BMP 路径与 memory 记录路径完全等价**，后续所有数字可归因到量化本身。

### 十六.4 🔴 像素契约（跨 PC/板卡的唯一真值）

```
磁盘（特征图）      [ I_t , GMC_diff , median_residual ]   ← build_full_median_dataset.py 写出
模型输入（forward） [ median_residual , GMC_diff , I_t ]   ← 训练时的顺序
板端 NPU 输入      uint8 NHWC，字节序 = reverse(cv2.imread(file))
```
关键恒等式：**`stbi_load(file) == reverse(cv2.imread(file))`**（stb_image 固定 RGB，OpenCV 固定 BGR），
而训练管线做的是 `cv2.imread` 后 `[::-1]`。所以只要 BMP 用
`cv2.imwrite(dst, cv2.imread(src))` 写出，**板端喂 NPU 的字节与 PC 喂 PyTorch 的张量逐字节相同**。
中间放 JPEG ⇒ libjpeg-turbo 与 stb_image 解码不一致 ⇒ 量化损失被解码差异污染 ⇒ **全部中间产物一律 BMP**。

**归一化**：`mean_values=[[0,0,0]] / std_values=[[255,255,255]]` ⇒ 图内烘焙 `(x-0)/255`，NPU 吃**原始 uint8 RGB**；
PyTorch 吃 `uint8/255`。两者数值相同 ⇒ 差异只可能来自 NPU。

### 十六.5 校准集矩阵（4 组 × 250，全部 BMP）

| 组 | 构成 | 落地分布 |
|---|---|---|
| Set-A 随机基准 | Anti-UAV train 125 + 龙泉山 official train 125（均匀） | 103 序列 |
| Set-B 暗弱难例偏置 | 001324+003002 暗弱刚体穿行 175 + 开阔天空 75 | 001324×226 / 003002×24 |
| Set-C 高动态杂波 | 复杂地表/建筑/公路/山林边缘反光 175 + 开阔天空 75 | 000703×165 / 001324×75 |
| Set-D 黄金三元态平衡 | 极暗弱刚体 85 + 高频强杂波 85 + 开阔标准空域 80 | 001324×141 / 000703×75 |

分层判据**自适应**：IR 均值约 96/255，先前按 "<40 判暗" 的绝对阈值选不出任何帧 ⇒ 改为**分位数**
（暗弱 = 有目标 ∧ IR ≤ Q25 ∧ 对比度 ≤ 中位；杂波 = 有目标 ∧ 对比度 ≥ Q90；天空 = 顶行带均匀且亮 ∧ 无目标）。
`n_gt` 来自 GT 标签，**"刚体穿行" = 该帧有目标**。阈值写进 manifest 可审计。

### 十六.6 PTQ 矩阵与筛选（p0_off，PC 模拟器）

| 配置 | cos_hm | cos_min | 构建耗时 |
|---|---|---|---|
| Set-C `kl_divergence` | **0.852816** | 0.728241 | 537 s |
| Set-A `kl_divergence` | 0.851756 | 0.727437 | 576 s |
| Set-D `kl_divergence` | 0.850386 | 0.717341 | 540 s |
| Set-B `kl_divergence` | 0.843241 | 0.731533 | 555 s |

⇒ **校准集选择对 int8 量化损失的影响很小**（4 组极差仅 0.0096）——**这是本轮最重要的实证结论**：
瓶颈不在校准集分布，而在 int8 本身的算子量化。`normal` / `mmse` 臂明日补。

### 十六.7 🔴 工具链硬事实（全部实测，别再踩）

| 事实 | 证据 |
|---|---|
| `load_rknn` 载入的模型**不支持 PC 模拟器** ⇒ 构建与模拟必须**同进程** | `init_runtime: RKNN model that loaded by 'load_rknn' not support inference on the simulator` |
| 模拟器输入必须 **uint8 NHWC** | 6 组对照探针：`mean/std+uint8` ⇒ cos **0.999999**；`mean/std+float32` ⇒ **0.108**（缩了 255 倍）；`无mean/std+float32` ⇒ 0.999999；NHWC/NCHW **无差别** |
| int8 构建 **~9 min/配置**，f16 **~4 s** | 实测 537~576 s vs 4~10 s |
| `mmse` 吃内存（toolkit 警告 *"will consume a lot of memory"*）⇒ 必须限并发 | 15 GB RSS |
| 板端吞吐 **6.2~6.5 fps**（单进程，200 帧/32.4 s；NFS 读只占 16 ms/帧 ⇒ 瓶颈在 NPU） | 板端实测 |
| 板卡本地盘只剩 1.6 GB（16 GB 用满 90%）⇒ BMP 只能放 NFS | `df -h /` |
| 服务器**无法挂** host 的 NFS（非 root），但能 ping 通板卡 ⇒ 走 `ssh` 别名 `ptq-board`（**只装公钥，口令不出服务器**） | 实测 |
| 内联 `python -c` 的 f-string 同引号嵌套在 Python 3.10 会 **SyntaxError** ⇒ 一律落文件 | 踩了 3 次 |
| `&` 会把 `cd ... && cmd` **整链后台化** ⇒ 远程一律用 `.sh` | 踩了 2 次 |
| 局部函数不能被 multiprocessing pool pickle ⇒ worker 必须是**模块级** | 踩了 2 次 |

### 十六.7 🔴🔴 int8 模型上板**全帧零检出**：真因 = 输出量程被削顶（**本节早先的"输入 dtype 陷阱"结论已作废**）

**症状**：int8 模型在板端跑得动、帧数全对，但 **每一帧都是 0 个检出**（F1=0.0000 / FP=0 / FN=2000），
而同一批 BMP 上 PT 参照臂正常（001324 F1=0.7484、黄金池 F1=0.9955）。

**板端日志**：
```
input dims=[1,640,640,3] fmt=NHWC type=INT8
out[0] dims=[1,1,320,320] type=INT8 zp=-128 scale=0.00094072
```
`zp=-128` ⇒ 可表示区间是 `[0, 255×0.00094072] = [0, 0.2399]`。
**真因不是输入 dtype，而是输出张量的量程只有 0.2399 < 判定阈值 0.3 ⇒ 真实峰值（0.85~0.97）全被削到阈值之下。**

**三条实测证据（互相独立）**：

1. **板端阈值探针**（同一模型 500 帧 001324，脚本 `09_board_threshold_probe.sh`）：
   | th | 有检出帧 | 总检测 | score_max |
   |---|---|---|---|
   | 0.30 | **0** / 500 | 0 | — |
   | 0.10 | 389 / 500 | 8529 | **0.2399** |
   | 0.05 | 431 / 500 | 10768 | **0.2399** |
   `score_max` **恒等于 255×scale** = 张量天花板 ⇒ **硬削顶**的指纹；降阈值即恢复检出 ⇒ **模型没坏，是量程被剪**。
2. **2×2 交叉验证**（唯一变量 = 算法；两个标定集 × 两个算法）：
   | 标定集 | 算法 | out[0] scale | 天花板 | 板端 th=0.3 |
   |---|---|---|---|---|
   | Set-A (BMP) | kl_divergence | 0.00094072 | 0.2399 | 0/60 帧，最高分 0.0000 |
   | trial-0474 原配方 (JPEG) | kl_divergence | 0.00092659 | 0.2363 | 同上 |
   | Set-A (BMP) | **normal** | **0.00378871** | **0.9661** | **60/60 帧，最高分 0.9093** |
   | Set-B (BMP) | **normal** | 0.00364057 | 0.9283 | — |
   ⇒ **算法是决定性变量，标定集不是**。
3. **机理量化**（脚本 `08_calib_peak_range_analysis.py`）：heatmap 逐像素分布**极度重尾** ——
   **99.97% 的质量在 0.24 以下**，决定检出的目标峰（0.90~0.97）只占 **0.0005%** 的质量。
   KL/熵型裁剪按"最小 KL 散度"选截断点，把这条尾巴当噪声丢掉（量级上可忽略）⇒ 量程止于 ≈0.24；
   min/max（`normal`）保留尾巴 ⇒ 量程到 0.97。
   min/max 预测器 `scale = max/255` 被 **6 个独立数据点**验证（set A 误差 0.7%、trial-0474 误差 1.4%）。

**反证参照**：冻结的 trial-0474 清单（200 张 Anti-UAV train JPG，float 峰值 0.94416）预测 `scale=0.0037026`，
板端实测 **0.00365272**（差 1.4%）⇒ **trial-0474 当年就是 min/max 量程**；
本轮换成 `kl_divergence` 才把范围砍到 1/4 ⇒ **不是本模型的问题，是算法选错了**。

**⇒ 结论**：`kl_divergence` 在本图（稀疏高饱和热图）上**不可用**；`normal` 可用（`mmse` 待测）。
**通用判据（写进报告与构建流程）**：任何 int8 模型上板前先读 `out[0] zp/scale`，
**`255×scale` 必须 > 判定阈值**（本项目 0.3；理想 ≈ 真实帧峰值 0.97），否则该模型是"死模型"，
它的 F1=0 是**量程缺陷**而不是量化损失，**不可当成量化结果仲裁**。

**⚠️ 已作废的两条早先结论**：
1. 「float32 输入 ONNX 与板端 uint8 喂法不兼容 ⇒ 正解是 uint8 图输入 ONNX」——**已证伪**：
   `make_uint8_input_onnx.py` 生成的 uint8 图输入 ONNX 在 toolkit 2.3.2 上 **abort rc=134 `Not Support Dtype: 2`**
   （`--no-quant` 与 hybrid 同样崩 ⇒ 与量化无关），与 `archive/02 §28.4` 已记录的死路一致 ⇒ **该路线封死**。
   另实测 uint8 图与 float 图**逐位等价**（100 帧 max_abs_diff=0.0，脚本 `07_verify_u8_vs_float_onnx.py`），
   但等价也建不出模型。
2. 「kl 臂排序 `C>A>D>B` ⇒ 校准集影响很小」——**排序作废**（kl 臂模型是死的），
   必须用 `normal` 臂重测（见 十六.10）。

### 十六.8 🔴 板卡访问服务器 SSD（工作站 18:30 关机后的全量评测通路）

工作站（NFS 导出方）18:30 关机 ⇒ 板端 `/mnt/manu/ptq/bmp` 消失。**服务器不关机且持有全部帧** ⇒ 三条路：

| 方案 | 状态 | 依据 |
|---|---|---|
| **SSH tar 流分块**（服务器推 / 板卡拉） | ✅ **采用** | 板卡 root 生成 `/root/.ssh/id_ed25519_ptq_server`，公钥写进**服务器 `huangzhe` 账号自己的** `~/.ssh/authorized_keys`（**零 su**）；实测 tar 流 **84 MB/s** |
| FUSE / sshfs 挂载 | ❌ **死路** | 板端 sshfs 3.6.0 是 fuse2 构建，内核 5.10.160 拒绝 libfuse2 挂载（`fusermount: old style mounting not supported`）；focal 无 fuse3 版 sshfs（jammy 的要 glibc 2.34，focal 是 2.31） |
| 服务器 NFS 导出给板卡 | ❌ 不可行 | 需要服务器 root（`exportfs` + `/etc/exports` + nfs-server）⇒ 违反"无 su"约束 |

配套事实：
- **板卡有外网**，坏的是 systemd-resolved 的 DNS 上游（`/etc/resolv.conf` → `127.0.0.53` 不通）⇒
  已在 `/etc/systemd/resolved.conf` 固化 `DNS=223.5.5.5 119.29.29.29`（备份 `.bak_20261022`）后 `apt` 可用。
- 板端 `stbi_load` **支持 PNG**（实测）⇒ 传输用无损 PNG：像素与 BMP 逐字节相同，体积 1/3.2（950 KB→293 KB）。
- **板卡磁盘**：`/dev/root` 只剩 1.6 GB（不可用）；**`/userdata` 12 GB 可用** ⇒ 暂存必须用 `/userdata`，
  且**分块推→跑→删**保持峰值 <1 GB。
- **服务器已持有全部 60,610 帧**：官方 val 28,997 BMP（`ptq_work/bmp/`）+ Anti-UAV val 31,613 BMP
  （`ptq_work/pcfeat_bmp/images/`，从工作站 NFS 一次性灌入 29 GB）⇒ 工作站关机后全量三轨照跑。
- ⚠️ **像素不同源**：工作站 `pcfeat_bmp` 的 BMP 与服务器源 JPG 经 cv2 转出的 BMP **md5 不同**
  （`4775c6a2…` vs `78e68ce7…`）⇒ 必须用**服务器那份**作为唯一真源，否则 Track1 数字会换一批像素。

### 十六.9 🔴 阶梯校验链（用户指定：带P0 → 去P0 → ONNX → f16 → int8，逐级可追溯）

100 敏感帧，th=0.3、topk=100，板端与 PC 用同一批 BMP、同一套峰值解码、同一套 GT 匹配：

| step | F1 | ΔF1 | ΔRecall | ΔFP | cos_hm | mse_hm | cos_off |
|---|---|---|---|---|---|---|---|
| L0_PT_p0_on | 0.7662 | — | — | — | — | — | — |
| L1_PT_p0_off | 0.7817 | **+0.0156** | +0.0125 | −2 | 0.995429 | 7.446e-07 | 0.999065 |
| L2_ONNX_p0_off | 0.7817 | +0.0000 | +0.0000 | +0 | **1.000000** | **2.026e-11** | **1.000000** |
| BOARD_f16 | 0.7739 | −0.0079 | −0.0063 | +1 | 0.985614 | 1.753e-06 | 0.999850 |
| BOARD_int8(normal) | 0.7156 | **−0.0583** | −0.0437 | +9 | 0.954663 | 5.891e-06 | 0.972562 |

累计对 L0（PT 带P0）：BOARD_f16 dF1=+0.0077、BOARD_int8 dF1=**−0.0506**。
镜像链 `L0_PT_p0_on → L2p_ONNX_p0_on → 板端` 同样出表（p0_on 板端模型缺位时优雅降级并注明）。

**读法**：
- **去 P0 反而 +1.56 pp** ⇒ P0 高速路对本图**有害**（不是"有用但有损"）。
- **ONNX 导出零损失**（cos=1.000000 / mse=2.0e-11）⇒ 导出链无误差。
- f16 板端 **−0.79 pp**；int8(normal) 板端 **−5.83 pp**（100 帧口径；2000 帧口径是 −10.36 pp，见 十六.10）。

### 十六.10 🔴 量化算法 × 校准集矩阵（`normal` 臂，板端 2000 帧 001324，th=0.3、topk=100）

PT 参照臂：F1=**0.7484**（R 0.6498 / P 0.8823 / FP 171 / FN 520）。

| 模型 | out[0] scale | 天花板 | F1 | R | P | FP | ΔF1 vs PT |
|---|---|---|---|---|---|---|---|
| f16（`p0_off_A_none`） | — | — | 0.7395 | 0.6410 | 0.8737 | 184 | **−0.0089** |
| int8 normal **Set-A** | 0.00378871 | 0.9661 | **0.6448** | 0.5835 | 0.7206 | 527 | −0.1036 |
| int8 normal Set-A（预 letterbox 标定） | 0.00381651 | 0.9733 | 0.6426 | 0.5835 | 0.7150 | 546 | −0.1058 |
| int8 normal Set-B | 0.00364057 | 0.9283 | 0.5970 | 0.5365 | 0.6728 | 636 | −0.1514 |
| int8 normal Set-C | 0.00375555 | 0.9575 | 0.6407 | 0.5753 | 0.7228 | 510 | −0.1077 |
| int8 normal Set-D | 0.00375555 | 0.9575 | （同上批次） | | | | |

**两条硬结论**：
1. **f16 近乎无损（−0.0089），int8 损失几乎全在虚警**（FP 171→527/636，召回只掉 4~11 pp）
   ⇒ int8 的激活量化噪声把**边缘背景峰**推过 0.3 ⇒ 这是热图模型做 int8 的本质难点。
2. **IR 分位数分层的四个校准集差异极小**（F1 极差 0.048，模拟器口径 0.057）⇒
   **"按 IR 统计分层"这个设计维度基本无效**；一阶杠杆是**算法**，不是图片集。
   候选集 v2（`calib2/set_{1,2,3}.txt`：强目标组 / train 复刻组 / 混合组，峰值 max 0.97629）
   用来检验"峰值覆盖"这一维是否还有剩余空间。

**板端 2000 帧口径的警告**：板端二进制**只跑单尺度**，而业务交付基线是多尺度（scales 0/160/80），
⇒ 板端绝对 F1 **不能**与 memory 里的 90.605% / 94.007% 直接比较；量化损失只能取**同尺度同阈值下的差值**。

### 十六.11 交付物（`manu/pipeline/npu/ptq/`）

| 文件 | 作用 |
|---|---|
| `ptq_common.py` | 共享契约（像素契约 / 通道序 / 阈值 / 峰值解码 / 自检闸 `selftest()`） |
| `01_build_calibration_sets.py` | 4 组 × 250 张 BMP 校准集（显式存在性检查 + `assert len(lines)==250`） |
| `02_batch_ptq_pc_screening.py` | PTQ 矩阵构建 + PC 模拟器逐层/输出级余弦与 MSE + JSON 报告 + 排名 |
| `03_make_fast_eval_manifest.py` | `001324_sub2000.json` / `golden_pool_sub1200.json` + 全量 BMP 铺量（28,997 帧 / 27 GB） |
| `04_run_board_fast_eval.py` | 板端 NPU 驱动 + **PT 同口径** P/R/F1/FP + 仲裁 |
| `05_ladder_verification.py` | 阶梯链逐级可归因（板端 `--dump` 原始张量 + `--dump-input` 回灌） |
| `06_verify_reference_anchor.py` | 参照臂锚点（像素级 + 端到端 191/191） |
| `07_verify_u8_vs_float_onnx.py` | uint8 图 ≡ float 图**逐位等价**验证（100 帧 max_abs_diff=0.0）——结论：该图**建不出模型**（abort） |
| `08_calib_peak_range_analysis.py` | 标定集峰值分布 / 尾部质量占比 / min-max 量程预测（6 点验证） |
| `09_board_threshold_probe.sh` | **板端阈值扫描探针**（判"死模型"vs"真量化损失"的决定性工具） |
| `10_probe_calib_peak.py` | 标定帧在 float ONNX 上的峰值探针（含通道序对照） |
| `11_make_calib_variants.py` | 标定集变体（BMP↔PNG↔trial-0474 原配方清单） |
| `12_calib_range_ab.sh` | **标定范围 A/B**（同 ONNX、同算法、只换标定集 ⇒ 读各自 out[0] scale） |
| `13_make_letterboxed_calib.py` | **预 letterbox 标定集**（标定分布 ≡ 推理分布，逐字节对齐） |
| `14_run_full_scale_nofs.py` | **无 NFS 全量驱动**（SSH tar 流分块推→跑→删，峰值 <1 GB，可续跑、逐块校验） |
| `15_night_selection.sh` + `16_collect_selection.py` | **两轴选型扫描**（T3 诊断轴 / T2 选型轴）+ 确定性选型规则与汇总表 |
| `17_probe_qdq_import.py` | **QDQ 导入探针**（决定 QAT 的激活侧收益能否到达板端） |
| `run_ptq_pipeline.sh` | 一键驱动（5 段）+ **关键常量说明（mean/std ↔ YOLO 预处理对应）** |

### 十六.12 本轮数据产物

- ONNX：`runs/onnx/trial_0080_hm_nop0.onnx`（`adf14c5b9af11c01dcdd023f2ffc3fb8`）/ `trial_0080_hm_p0.onnx`（`725d92efa3caf16d486044f58d9ffd22`）
  / uint8 输入版 `trial_0080_hm_{nop0,p0}_u8.onnx`（**能建图但建不出模型**）
- 校准集：`/mnt/data/siping/ptq_work/calib/`（set_A..D.txt 指向 BMP）；变体 `calib2/`（set_1..3 = 候选 v2、set_t0474、png/）；
  预 letterbox `calib3/lb_A/`
- 帧全集（**服务器为准**）：`/mnt/data/siping/ptq_work/bmp/`（官方 val 28,997）
  + `/mnt/data/siping/ptq_work/pcfeat_bmp/images/`（Anti-UAV val 31,613，从工作站 NFS 一次性灌入 29 GB）
- RKNN 矩阵：`screen/`（f16+kl，旧）/ `screen_n/`（normal×A..D）/ `screen_v2/`（normal×候选 v2）/ `screen_m/`（mmse）/ `screen_on/`（p0_on）
- 锚点报告：`/mnt/data/siping/ptq_work/anchor/anchor_report.json`
- 板端 Tier2：`/mnt/data/siping/ptq_work/tier2/`；阶梯：`/mnt/data/siping/ptq_work/ladder/ladder_report.json`；
  选型：`/mnt/data/siping/ptq_work/selection/selection_table.json` + `best_model.json`
- 峰值分析：`/mnt/data/siping/ptq_work/peak_analysis/{peak_stats,candidate_stats,pool_scan}.json`

---

## §五十六 晚间补录（2026-10-22 17:00–18:30）：量化损失的真实结构 + PTQ 路线封死 + QAT 解锁

> 本节是 `manu/memory/memory.md` §11.1 的归档正文。全部结论均带脚本相对路径、可复制命令、执行机与产物路径。

### 1. 阶梯链全表（100 敏感帧，th=0.3，topk=100）

| step | F1 | ΔF1 | cos_hm | mse_hm | cos_off |
|---|---|---|---|---|---|
| L0_PT_p0_on | 0.7662 | — | — | — | — |
| L1_PT_p0_off | 0.7817 | +0.0156 | 0.995429 | 7.446e-07 | 0.999065 |
| L2_ONNX_p0_off | 0.7817 | +0.0000 | 1.000000 | 2.026e-11 | 1.000000 |
| BOARD_f16 | 0.7739 | −0.0079 | 0.985614 | 1.753e-06 | 0.999850 |
| BOARD_int8(normal) | 0.7156 | −0.0583 | 0.954663 | 5.891e-06 | 0.972562 |

⇒ 累计对 L0：f16 **+0.0077**、int8 **−0.0506**。ONNX 与 PT 的数值差 `2.026e-11` ⇒ **导出链无损**。
产物：`/mnt/data/siping/ptq_work/ladder/ladder_report.json`。

### 2. 量化损失 100% 在虚警，且大部分可由阈值重标定回收

板端 001324 2000 帧，**同一批 TSV 离线重扫**（脚本 `manu/pipeline/npu/ptq/19_board_threshold_sweep.py`）：

| 臂 | th=0.30 | 0.35 | 0.40 | **0.45** | 0.50 | 0.55 | 0.60 | 0.70 |
|---|---|---|---|---|---|---|---|---|
| int8(normal) F1 | 0.6448 | 0.6826 | 0.6951 | **0.7090** | 0.6980 | 0.6858 | 0.6635 | 0.5962 |
| int8(normal) FP | 527 | 363 | 276 | **198** | 151 | 109 | 97 | 75 |
| f16 F1 | 0.7395 | **0.7469** | 0.7400 | 0.7218 | 0.7012 | 0.6744 | 0.6484 | 0.5895 |
| f16 FP | 184 | 109 | 67 | 42 | 26 | 18 | 11 | 7 |

（PT float @0.3 = 0.7484 / FP 171 / TP 1282 / FN 520。）

- ⇒ **量化把最优检测阈值推移了 +0.10**（int8 最优 0.45，f16 最优 0.35）。
- ⇒ 阈值重标定把 int8 从 0.6448 拉到 **0.7090（+0.0642）**；量化损失从 −0.1036 收窄到 **−0.0379（回收 63%）**。
- ⇒ **剩余 0.0379 阈值救不回来**：th=0.60 时 int8 仍 FP 97 而 f16 只有 11 ⇒ 量化确实造出额外假峰（= QAT 的靶子）。
- ⇒ **部署含义：板端 int8 不得沿用 PC 的 `th_base=0.30`**，必须重标定到 ~0.45。
- ⇒ **选型主键因此改为"部署口径"**（各模型在自己最优阈值下的 F1），同时仍报 th=0.30 的同口径量化损失。

复现命令（服务器）：
```bash
python manu/pipeline/npu/ptq/19_board_threshold_sweep.py \
  --manifest /mnt/data/siping/ptq_work/fast/001324_sub2000.json \
  --tsv-dir /tmp/ptq_board_out/p0_off_A_normal \
  --thresholds "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70" --skip-pt \
  --out /mnt/data/siping/ptq_work/sweep_up/p0_off_A_normal
```

### 3. 板端 2000 帧 Tier2 全表（001324，th=0.3，topk=100）

| 臂 | F1 | R | P | TP | FP | FN |
|---|---|---|---|---|---|---|
| PT float | 0.7484 | 0.6498 | 0.8823 | 1282 | 171 | 520 |
| f16（`p0_off_A_none`） | 0.7395 | 0.6410 | 0.8737 | 1273 | 184 | 529 |
| int8 normal set-A | 0.6448 | 0.5835 | 0.7206 | 1359 | 527 | 443 |
| int8 normal 预 letterbox | 0.6426 | 0.5835 | 0.7150 | 1370 | 546 | 432 |
| int8 normal set-B | 0.5970 | 0.5365 | 0.6728 | 1308 | 636 | 494 |
| int8 normal set-C | 0.6407 | 0.5753 | 0.7228 | 1330 | 510 | 472 |
| int8 normal set-D | 0.6371 | 0.5738 | 0.7163 | 1338 | 530 | 464 |

黄金池 1197 帧：PT 0.9955、f16 0.9849、int8-A 0.9420、int8-lb 0.9436。
⇒ **A/B/C/D 极差仅 0.048 ⇒ IR 分位数分层维度基本无效；一阶杠杆是算法/阈值，不是校准集。**

### 4. PTQ 算法轴在 RK3588 上封闭（实测封死，勿重试）

- toolkit 2.3.2 在 rk3588 上**只有 `w8a8`** 可用：
  `w8a16`/`w4a16` → `The quantized_dtype = 'w8a16' not support in 'rk3588'!`；
  `w4a8`/`w4a4`/`w8a4`/`w16a8`/`w16a16` → `Invalid quantized_dtype ... expect in ['w8a8','w8a16','w16a16i','w16a16i_dfp','w4a16']`。
- ⇒ **`gdq`（官方 Gradient-based Distillation Quantization）随之出局**（它只支持 w4a16/w8a16）。
- `mmse`：实测 151 层中 9 层用 29 分钟、**ETA 4h08m**、15.1 GB RSS ⇒ **放弃该臂**（原写"mmse 待测"作废）。
- ⇒ 算法轴实际只剩 {normal ✅、kl_divergence ❌削顶、mmse ⏱超时} ⇒ **PTQ 内部无出路**。
- 保留待用入口：`rknn.hybrid_quantization_step1(dataset=..., proposal=True)`（官方逐层敏感度建议，指定层留 float16）
  + `rknn.hybrid_quantization_step2(model, data, cfg)`（按 cfg 生成混合量化模型）。

### 5. `QDQ_ADOPTED` ⇒ QAT 技术路线解锁

脚本 `manu/pipeline/npu/ptq/17_probe_qdq_import.py`（三侧 md5 `145408c66681defb22c8391d82ba9517`），
产物 `/mnt/data/siping/ptq_work/qdq_probe/qdq_probe.json`：

- **rknn-toolkit2 2.3.2 会采纳 ONNX 里 Q/DQ 节点的 `scale`/`zero_point`**，并一路带到板端输出张量（5/5 案例）。
- 两条硬约束：**Q/DQ ONNX 必须 `build(do_quantization=False)`**（同时 `do_quantization=True` ⇒ `build rc=3 / ret=-1` 直接失败）；
  **不能**再传 `mean_values`/`std_values`。
- torch.ao 路线端到端验证：`prepare_fx`+`convert_fx`(per-channel) → `export(opset 13)` → `load_onnx` → `build(False)`
  → 板端 `out[0] type=INT8 zp=-91 scale=0.00544954`（ONNX 写 `scale=0.005449536256492138, zp=37` ⇒ **scale 逐位一致**，
  uint8 zp 37 被重定心到 int8 −91，量程等价）。
- 板端 I/O 契约：**1 输入** `[1,640,640,3]` uint8 NHWC + **2 输出** `[1,1,320,320]` heatmap、`[1,2,320,320]` offset。
- ⇒ **QAT 学到的 clip 能精确落到板端，激活侧收益不会被 toolkit 丢掉。**

### 6. `anti` 轨 GT 根曾整轨指错（已修）

- `GT_ROOT` 指向龙泉山标签树；Anti-UAV 标签实际在
  `/mnt/data/siping/datasets/manu/uav_gmc_median/labels/val`，**命名 `{seq}__{%06d}.txt`、内容 `0 cx cy w h`，与龙泉山完全同格式**
  ⇒ 31,613 帧**全部算 FP**、该轨指标不可采信。
- 修法：`ptq_common.ANTI_GT_ROOT` + `gt_path(seq, fid, root=None)`（向后兼容，2 参调用行为不变）
  + `14_run_full_scale_nofs.py --gt-root [TRACK=]PATH`（per-track，默认 blind/golden→龙泉山、anti→Anti-UAV）。
- 实测：`gt_files=31613/31613`（原 `0/31613`）；40 帧真板端跑 **F1=1.0000 R=1.0000 P=1.0000 TP=40 FP=0 FN=0**
  （canvas→source 误差 0.22 px < `DIST_THR=8` ⇒ 不是退化匹配）。

### 7. 15 脚本的两个实测坑（都会静默丢数据）

- **`while read < $LIST` 循环里跑 `python`/`scp` 会继承 stdin 吃掉清单剩余行** ⇒ 每个 axis 只跑 1 个模型就 EOF。
  必须 `< /dev/null`。
- **板卡 ctx 计数必须按 `comm` 而非 `args`**：板端 demo 由 `ssh "bash -c '/root/rknn_heatmap_smoke ...'"` 启动，
  `ps -eo args | grep -c` 把 **bash 包装 + 真实 demo 一起数**（1 个 ctx 数成 2）⇒ 闸误判"板卡满"并**静默跳过模型**。
  正确口径：`ps -eo comm | grep -c "^rknn_heatmap"`（comm 截断 15 字符）。
- 另：`ps -eo args | grep "<脚本名>"` 会**匹配到本机 ssh 命令行自身** ⇒ 必须用 `night[_]selection` 这类方括号转义。

### 8. 工作站 18:30 关机不影响主链路

`manu/pipeline/npu/ptq/20_finalize_and_launch.py`（服务器端自主，`setsid nohup` 脱离会话）完成：
「等 15 选型扫描 → 对每个候选离线扫阈值 → 按写死的部署口径规则定案 → `final_choice.json` → 启动 14 全量三轨」。
14 自带 flock 锁、断点续跑、逐块校验、失败隔离；PT 参照臂走 `cuda:1` 让开 QAT 训练的 `cuda:0`。
全量采集阈值固定 **0.30**（与本节全部既有实验同口径、可直接对账），部署阈值结论来自选型阶段的曲线。
服务器是**全部 60,610 帧的唯一真源**（工作站 `pcfeat_bmp` 与服务器源 JPG→BMP 的 md5 不同：`4775c6a2…` vs `78e68ce7…`）。

### 9. 明早待办

1. 读 `final_selection/final_choice.json` + `full_scale/status.json` 与三轨报告 ⇒ 补全量三轨表。
2. 板端多阈值离线扫全量 TSV ⇒ 给出全量口径下的最优部署阈值。
3. QAT（LSQ 微调 + QDQ 导出）结果并入最终模型评估；QDQ ONNX 必须 `build(do_quantization=False)`、不传 mean/std。
4. 补 `p0_on` 变体的板端两级阶梯（镜像链）。
5. 若需纯 int8 之外的中间档，评估 `hybrid_quantization_step1` 的逐层敏感度 proposal。

---

## §五十七 夜间补录（2026-10-23）：全量三轨 int8 vs PT float 对照 + 量化损失机理 + 输入 dtype 硬闸门

> 本节把 2026-10-22 夜里的"待补"三项（选型 / 全量三轨 / 全量阈值曲线）**全部结清**，
> 并**纠正**上一节的两处过强表述：「阈值位移 +0.10」与「量化损失 100% 在虚警」。
> 全部数字取自服务器产物，未做二次加工；口径一律为**同帧 / 同解码器 / 同 GT / 同阈值 / topk=100**。
> 汇总版：`manu/memory/PTQ_RK3588_int8_报告.md` §8~§9（2026-10-23 补全）。

### 1. 选型结果（结清「明早待办 1」）

产物：`/mnt/data/siping/ptq_work/final_selection/final_choice.json`（落盘 2026-10-22 17:53）、
`/mnt/data/siping/ptq_work/selection/{selection_table.json,best_model.json,model_list.txt}`。
扫轴：axis1 = `001324_sub2000`（2000 帧）/ axis2 = `golden_pool_sub1200`（1197 帧）。

| 模型 | 板端模型 md5 | axis1 最优 th / F1 | axis2 最优 th / F1 | 判定 |
|---|---|---|---|---|
| `p0_off_A_none`（**fp16 臂**） | `1a3faf2f9459cad8a389ffa81810234c` | 0.35 / 0.746898 | **0.30 / 0.984916** | 🏆 **规则 winner** |
| `p0_off_A_normal`（**int8 臂**，Set-A + `normal`） | `cedaeb74d066a64ba2c82003993defb1` | 0.45 / 0.708957 | 0.35 / 0.942821 | **int8 臂最优**；全量三轨跑的就是它 |
| `p0_on_A_normal`（带 P0） | — | 无数据 | 无数据 | ❌ 板端 **0 检出**（TP 0 / FP 0 / FN 1802） |
| `p0_off_{1,2,3}_normal`（候选集 v2） | — | `null` | `null` | ⚠️ 扫描未产出数据 ⇒ 未参与定案 |

- `rejected: []`：**没有臂触发 REJECT**（`out_ceiling` 全 `null` ⇒ 选型阶段没采到 `255×scale`）。
- 🔴 **规则缺陷（如实记录）**：选型规则主键是「黄金池最优 F1」，**没有 int8-only 约束** ⇒ winner 永远是 fp16 臂。
  本轮全量三轨按原计划跑 **int8 臂** `p0_off_A_normal`（`full_scale_int8/status.json.model`）⇒
  **复用该规则前必须补"候选臂限定为 int8"的硬约束**。

### 2. 全量三轨对照（结清「明早待办 1~2」）

采集：`manu/pipeline/npu/ptq/14_run_full_scale_nofs.py`，int8 模型
`/mnt/data/siping/ptq_work/screen_n/p0_off_A_normal.rknn`（md5 `cedaeb74d066a64ba2c82003993defb1`），
采集阈值固定 **0.30**。
> ⚠️ **路径更正**：实际产物目录是 **`/mnt/data/siping/ptq_work/full_scale_int8/`**（不是上文 §9 待办里写的 `full_scale/`）。
产物：`/mnt/data/siping/ptq_work/full_scale_int8/{anti,golden,blind}_board.tsv` +
`{anti,golden,blind}_report.json` + `status.json`；
阈值曲线：`/mnt/data/siping/ptq_work/full_scale_sweep/{anti,golden,blind}_sweep.json` + `full_scale_sweep.json`。

| 轨 | 帧数 | GT | int8@0.30 F1 | PT@0.30 F1 | **量化损失@0.30** | int8 最优 F1@th | PT 最优 F1@th | 最优处损失 | 最优阈值位移 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **T1 Anti-UAV val** | 31,613 | 25,111 | 0.8603 | 0.8791 | **−0.0189** | 0.8606@0.35 | 0.8839@0.35 | −0.0232 | **0** |
| **T2 黄金池** | 17,187 | 16,639 | 0.9558 | 0.9952 | **−0.0394** | 0.9570@0.55 | 0.9971@0.65 | −0.0401 | **−0.10（反向）** |
| **T3 001324 盲测** | 11,810 | 10,535 | 0.6664 | 0.7458 | **−0.0795** | 0.7581@0.55 | 0.7950@0.50 | −0.0370 | **+0.05** |

- **完整性**：**60,610 帧 / 123 块 / 0 失败 / 0 缺失 GT**（`done_chunks 123/123`、`errors: []`、
  每轨 `n_missing_gt: 0`、`chunks_failed: []`）。
- **时间窗**：watchdog 18:43:14 启动 → 首块完成 **18:44:01** → 末块完成 **20:33:44** → `status.json` 最后更新 **20:34:12**
  ⇒ **约 1 h 50 min**。
- **GT 分母**（= TP+FN，两臂一致）：Anti-UAV 20,810+4,301 = 21,453+3,658 = **25,111**；
  黄金池 16,628+11 = 16,612+27 = **16,639**；001324 8,100+2,435 = 7,887+2,648 = **10,535**。
- **th=0.30 明细**：T1 int8 TP 20,810 / FP 1,229 / FN 4,301，PT TP 21,453 / FP 1,120 / FN 3,658；
  T2 int8 TP 16,628 / FP 763 / FN 11，PT TP 16,612 / FP 66 / FN 27；
  T3 int8 TP 8,100 / FP 2,838 / FN 2,435，PT TP 7,887 / FP 1,364 / FN 2,648。

### 3. 与历史 trial-0474 对齐：**−0.0189 是复现，不是新发现**

历史来源：`manu/memory/heatmap_npu_trial0474.md`、`manu/memory/archive/02_流水线与量化主线_§二十~§三十五.md` §二十 / §二十六.1
（去 P0 四臂，Anti-UAV val 全量 **31,613 帧 / 25,111 GT**）：

| 臂 | 峰值 th | F1 | TP | FP |
|---|--:|--:|--:|--:|
| 臂1 PC-FP32 **+P0**（冻结基准） | 0.25 | 0.902316 | 21,601 | 1,167 |
| 臂2 PC-FP32 **−P0** | 0.25 | 0.901935 | 21,586 | 1,169 |
| 臂3 **NPU-int8 −P0** | 0.20 | 0.882231 | 20,818 | 1,265 |
| 臂4 **NPU-fp16 −P0** | 0.25 | 0.901401 | 21,548 | 1,151 |

- 历史 int8 量化损失：**峰值口径 −0.019704**（0.901935 − 0.882231，各取自己的最优阈值）；
  **同点口径 th=0.25 为 −0.020986**（0.901935 vs 0.880949）；fp16 相对 PC-FP32 仅 **−0.000533**（−38 TP），挽回 97.3%。
- 本次 Anti-UAV 的帧数（31,613）与 GT 数（**25,111**）**与之完全一致** ⇒ th=0.30 **同点口径** 的
  **−0.0189** 与历史 −0.0197（峰值口径）/ −0.0210（同点 th=0.25 口径）**互相印证**。
- 🔴 **口径红线**：量化损失**只能**取 int8 与 PT float 在**同帧、同解码器、同 GT、同阈值**下的差值；
  **禁止**拿 int8 绝对 F1 去比历史 float 绝对 F1。

### 4. 量化损失的机理：**两种方向相反的失效模式**

| 轨 | 目标形态 | 相对 PT float（th=0.30，同帧同 GT） | 失效模式 | 业务表现 |
|---|---|---|---|---|
| **T1 Anti-UAV**（强目标，峰值 0.90~0.97） | 峰值高 | **少 643 个 TP**、只多 **109** 个 FP | **削峰** | 高峰被削低 ⇒ **漏检** |
| **T3 001324**（红外弱小目标，峰值 0.35~0.60） | 峰值弱 | **多 213 个 TP**、多 **1,474** 个 FP | **抬底** | 弱目标与背景一起被抬起 ⇒ **虚警** |
| T2 黄金池（密集强目标） | 峰值高、召回近饱和 | 多 16 个 TP、多 697 个 FP | 抬底为主 | 虚警（F1 仍 0.9558） |

- 同一现象在 **001324 2000 帧子集**上更强：int8 TP 1,359 vs PT float 1,282 ⇒ **多 77 个 TP**；
  FP 527 vs 171 ⇒ **多 356 个 FP**（产物 `/mnt/data/siping/ptq_work/logs/tier2_fixed.log`）。
- **机理**：量化噪声是**绝对量级的加性误差** ⇒ 对峰值 **0.95** 的强目标**抬不动**（⇒ 削峰/漏检）；
  对峰值 **0.35~0.60** 的弱小目标**足以改写判定**（⇒ 抬底/虚警）。
- **热图侧支撑（昨日已测）**：**99.97% 的质量在 0.24 以下**，决定检出的目标峰（**0.90~0.97**）只占 **0.0005%** 的质量
  ⇒ 重尾分布 + 绝对量级加性噪声，正是上面两条失效模式的成因。
- **与历史 DJI 结论互证**：本次 Anti-UAV 全量 `DJI_0051_2` **TP=2 / FN=576**（578 帧，召回 **0.35%**）、
  `DJI_0175_2` **TP=57 / FN=391**（448 帧，召回 **12.7%**）⇒ DJI 序列在 int8 下近乎全漏；
  历史 trial-0474「剔除 DJI 后召回 0.8955→0.8727（−2.27 pp），配对 McNemar **p=0.0129** 显著」**方向一致**。
  ⚠️ 两次 ex-DJI 帧集定义不同（历史 550 帧 / 440 GT，本次 2 序列 1,026 GT）⇒ **只做方向互证、不做数值对账**。

### 5. 🔴 纠正：「阈值位移 +0.10」**只对 001324 的 2000 帧子集成立**

| 轨 | int8 最优 th | PT float 最优 th | **位移** | th=0.30 → 最优 的 F1 | 量化损失@0.30 → @最优 |
|---|--:|--:|--:|--:|--:|
| T1 Anti-UAV | 0.35 | 0.35 | **0** | 0.8603 → 0.8606（+0.0003） | −0.0189 → −0.0232 |
| T2 黄金池 | 0.55 | 0.65 | **−0.10（反向）** | 0.9558 → 0.9570（+0.0012） | −0.0394 → −0.0401 |
| T3 001324 | 0.55 | 0.50 | **+0.05** | 0.6664 → **0.7581**（+0.0917） | **−0.0795 → −0.0370（回收 53%）** |

⇒ **阈值重标定是逐轨的，不是全局规律**；2026-10-22 写下的「通用 +0.10」**不得推广**。
收益只在 **T3（弱小目标轨）** 显著（回收 **53%**）；T1/T2 重标定几乎无收益。
⇒ 同时纠正上一节「量化损失 100% 在虚警」：**只在 001324 子集成立**，Anti-UAV 全量是**反向的削峰**。

### 6. 🔴 新增硬闸门：**输入 dtype 决定 `input_set` 延时**（2026-10-23）

| 路径 | 图的输入算子 | `inputs_set` | NPU 内部内存 |
|---|---|--:|--:|
| **toolkit 自己全图 int8 量化**（本轮 `p0_off_A_normal`） | **INT8**（`type=2`，1 B/px，1,228,800 B） | **0.34 ms** | **20,400 KB** |
| `--no-quant`（fp16 模型） | **FLOAT16**（`type=1`，2 B/px，2,457,600 B） | **45.41 ms** | 105,600 KB |
| `quantized_hybrid_level=1/2/3`（2026-10-16 实测） | **FLOAT16** | 45.41 ms | 105,600 KB |
| **QDQ ONNX + `build(do_quantization=False)`（QAT 路线）** | **FLOAT16** | 45.41 ms | 105,600 KB |
| 同上但展宽挪到大核 | FLOAT16 | **14.77 ms** | 105,600 KB |

- **45.41 ms ÷ 1,228,800 = 37 ns/元素（≈74 cycles @2 GHz）** ⇒ 是**小核上的标量逐元素展宽**，不是 NPU 计算。
- **QAT 臂的原始证据**：服务器 `/mnt/data/siping/ptq_work/logs/qat_build_arms.log`（2026-10-23 09:12~09:13），
  **8 个臂全部**打出
  `D RKNN: 0  InputOperator  FLOAT16  CPU  (1,3,640,640)  InputOperator:images`
  （`a0_qat_float` / `a1_qat_maxclip` / `a1b_base_maxclip` / `a2_qat_learned`，各带 `_nomeanstd` 变体）
  ⇒ **QAT 路线的模型输入张量是 FLOAT16，不是 int8。**
- **根因**：`manu/memory/archive/02_流水线与量化主线_§二十~§三十五.md` §二十八 —— toolkit 把
  「图的输入张量 dtype」与「计算精度」绑死 ⇒ **没有开关能让输入停在 int8 而图内混精度**。
- 🔴 **含义**：**混合精度 / 混合 QAT 产物即使精度更好，也不是延时意义上的 int8 模型**（int8 的核心延时优势
  就是 `inputs_set` **0.34 ms vs 45.41 ms**）。板端受控实测（同一 manifest `001324_sub2000`、同批 BMP、
  同一脚本 `04_run_board_fast_eval.py`，产物 `logs/tier2_fixed.log`）：
  **int8 2000 帧 152.0 s = 13.16 fps** vs **fp16 2000 帧 308.6 s = 6.48 fps** ⇒ int8 **2.03×**。

### 7. 复现命令（执行机：服务器 `ssh -p 32222 huangzhe@192.168.99.40`）

```bash
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
cd /tmp/pycharm_project_10ae9e2e

# 全量三轨（18:43 起跑 → 20:34 完成）
python manu/pipeline/npu/ptq/14_run_full_scale_nofs.py \
  --work /mnt/data/siping/ptq_work --model /mnt/data/siping/ptq_work/screen_n/p0_off_A_normal.rknn \
  --out /mnt/data/siping/ptq_work/full_scale_int8 --th 0.3 --topk 100

# 全量 TSV 阈值扫描 + PT float 参照臂（零板端成本）
python manu/pipeline/npu/ptq/19_board_threshold_sweep.py \
  --manifest /mnt/data/siping/ptq_work/fast/001324_sub2000.json \
  --tsv-dir /mnt/data/siping/ptq_work/full_scale_int8 \
  --thresholds "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70" \
  --out /mnt/data/siping/ptq_work/full_scale_sweep

# 选型结果（读结论用）
cat /mnt/data/siping/ptq_work/final_selection/final_choice.json

# QAT 臂输入算子自证（8 个臂全 FLOAT16）
grep "InputOperator" /mnt/data/siping/ptq_work/logs/qat_build_arms.log
```

### 8. 本节结清情况

| 上节「明早待办」 | 状态 |
|---|---|
| 1. 读 `final_choice.json` + `status.json` 与三轨报告 ⇒ 补全量三轨表 | ✅ 本节 §1 / §2 |
| 2. 板端多阈值离线扫全量 TSV ⇒ 全量口径最优部署阈值 | ✅ 本节 §2 / §5（**并纠正"+0.10"**） |
| 3. QAT 结果并入最终模型评估 | ⏳ **未完成** —— 8 臂已建（`qat/trial_0080/rknn/`），但输入算子全 FLOAT16（本节 §6）⇒ 精度评估前必须先裁决「要不要为 int8 输入重新导出图」 |
| 4. 补 `p0_on` 变体的板端两级阶梯（镜像链） | ⏳ 未做（本轮 `p0_on` 板端 0 检出，见 §1） |
| 5. 评估 `hybrid_quantization_step1` 逐层敏感度 proposal | ⏳ 未做（且 §6 已证明：混合路径输入必为 FLOAT16 ⇒ 延时上不划算） |
| （新增）QAT + int8 输入 | ⏳ **前置硬闸门**：toolkit 2.3.2 对 uint8 图输入 `abort rc=134 Not Support Dtype: 2`（§2 / 十六.7）⇒ 尚未打通 |

---

## §五十八 夜间补录（2026-10-23 下午）：配置轴全封死 + 校准集泄漏闸门 + 干净集复评 + QAT 最终判定

> 本节是同日白天的续录（服务器自主链路，板卡零占用）。**本节推翻了同日白天的两条中间结论**：
> ① 「校准集 v2（set_3）优于基线」；② 「A/B/C/D 极差仅 0.048 ⇒ 校准集轴无效」的**原依据**。
> 全部数字取自服务器产物，**口径一律标注**（板端 / PC 模拟器 / 子集 / 全量）。
> 汇总版：`manu/memory/PTQ_RK3588_int8_报告.md` §5 / §19~§24；主档 `manu/memory/memory.md` §11.5。

### B1. ⭐ PTQ 配置轴全表：rk3588 + toolkit 2.3.2 上**全部封死**（实测，勿重试）

证据来源：`02_batch_ptq_pc_screening.py`（新增 `--quant-method/--quant-hybrid-level/--auto-hybrid-cos/--auto-hybrid-euc/
--optimization-level/--enable-flash-attention/--verbose-build`）+ `run_ptq_knobs.sh` / `run_ptq_knobs2.sh`；
产物 `screen_base / screen_v2b / screen_layer / screen_hyb1_diag / screen_olv0 / screen_olv2 / screen_fa /
screen_grp32 / screen_grp64`；日志 `logs/run_ptq_knobs.log`、`logs/run_ptq_knobs2.log`、`logs/knob_*.log`。

| 旋钮 | 判定 | 证据 |
|---|---|---|
| `quantized_dtype` | **只有 `w8a8`** | `w8a16`/`w4a16` → `The quantized_dtype = 'w8a16' not support in 'rk3588'!`（2.3.2 用 `rknn.config()` 直探）⇒ `group{SIZE}` 随之不可用 |
| `quantized_algorithm` | **只有 `normal`** | `kl_divergence` 削顶（输出上限 0.2399 < 判定阈值 0.3）；`mmse` ETA 4h08m；`gdq` → `The quantized_algorithm = 'gdq' only support in ['w4a16', 'w8a16']!` |
| `quantized_method=layer`（per-tensor） | ❌ **更差** | 100 帧 PC 模拟器：cos_hm **0.9679118699845649** vs channel **0.9730922927692202**；mse **4.037e-06** vs **3.566e-06**（`screen_layer/screen_p0_off.json`）⇒ 保留 `channel` |
| `quantized_method=group{SIZE}` | 🚫 **死路** | `The quantize_method = 'group32' only support when quantized_dtype = 'w4a16'`；合法 SIZE ∈ **[32,64,96,128,160,192,224,256]**（`group4`/`group512` → `Invalid quantized_method ... where {SIZE} is in [...]`） |
| `quantized_hybrid_level=1` | 🚫 **死路（延时）** | 2.3.2 复验：构建算子表 `InputOperator **FLOAT16** (1,3,640,640)`；`Total Internal Memory Size: 105600KB`（= fp16）；板端 `inputs_set` **45.41 ms**（与 2026-10-16 §28 在 2.3.0 上的结论一致） |
| `auto_hybrid_cos_thresh` / `auto_hybrid_euc_thresh` | 🚫 依赖 hybrid ⇒ **同死** | 2.3.2 的有效 hybrid 值是 `quantized_hybrid_level=1`（**0 = 关闭**），不是 8/16 |
| `optimization_level=0` / `=2` | ➖ **零影响** | cos_hm **0.9730922927692202** / cos_min **0.751060131865189** / mse **3.566024651078652e-06** / F1 **0.7298578199052133**（TP 77 / FP 17）—— 与基线**逐位相同**，只有 `.rknn` 字节 md5 不同（olv0 `d6d985fb…` / olv2 `b1dcdd77…` / base `75d84fcd…`） |
| `enable_flash_attention=True` | ➖ **零影响** | 同上**逐位相同**（`screen_fa` 的 `.rknn` md5 `220895b3…`） |
| 校准集 A/B/C/D | ⚠️ **原依据已污染** | B/C/D **泄漏**（见 B2）⇒ 「极差仅 0.048」不能作为"校准集轴无效"的依据；已用**干净**集复评（**全量只验证过 set_6**，见 B3） |
| **校准集（干净集 set_4/5/6/7/8/9）** | ⚠️ **子集口径有收益、全量口径不兑现** | 全量口径（**唯一被全量验证的干净集 = set_6**）：golden **≈+0.0000**、blind **−0.0064**、anti 待跑（见 B3）⇒ **不能单独当选型杠杆**；⚠️ **set_4 / set_5 的子集收益尚未做全量，是否成立未知**（B3 第 (3) 条） |

> ⚠️ 汇总表的已知假阴性：`logs/run_ptq_knobs2.log` 里 `screen_fa / screen_olv0 / screen_olv2` 显示 `no_json / 0.00000`
> 那是**汇总器按 tag 取 json 文件名**没对上（这些臂的 json 落成 `p0_off_A_normal.json`），
> **每个臂自己的 `.log` 里有完整指标**（`screen_fa/p0_off_A_normal.log` 内嵌 json：cos_hm 0.9730922927692202 / F1 0.7298578199052133）。
> ⇒ 读数请认**臂内 json / 臂内 log**，不要认汇总表。

### B2. 🔴 校准集泄漏闸门：v2 家族整族泄漏，且原 A/B/C/D 的 B/C/D 也泄漏

工具：`manu/pipeline/npu/ptq/27_check_calib_leak.py`（路径 + 序列双查，服务器 `conda activate rknn`）：

```bash
W=/mnt/data/siping/ptq_work
python manu/pipeline/npu/ptq/27_check_calib_leak.py \
  --calib $W/calib4/set_6.txt --calib-sources $W/calib4/set_6_sources.txt \
  --manifests $W/fast/anti_uav_sub2500.json $W/fast/001324_sub2000.json \
            $W/fast/golden_pool_sub1200.json
```

**2026-10-23 下午实测全表**（`[paths]` 命中数 = 校准清单 + 来源清单两条表一起数）：

| 校准集 | 帧数 | 判定 | 路径命中 | 共享评测序列 |
|---|--:|:--|--:|---|
| `calib/set_A` | 250 | ✅ **PASS** | 0 | 0 |
| `calib/set_B` | 250 | 🔴 **FAIL LEAKED** | 226 | 001324 |
| `calib/set_C` | 250 | 🔴 **FAIL LEAKED** | 240 | 001324 + 000703 |
| `calib/set_D` | 250 | 🔴 **FAIL LEAKED** | 216 | 001324 + 000703 |
| `calib2/set_1` | 250 | 🔴 **FAIL LEAKED** | 500 | 001324 + 5 条黄金池序列 |
| `calib2/set_2` | 250 | 🔴 **FAIL LEAKED** | 300 | 001324 + 5 条黄金池序列 |
| `calib2/set_3` | 250 | 🔴 **FAIL LEAKED** | 245 | 001324 + 6 条黄金池序列（000703/000818/001307/001531/015000/015052） |
| `calib4/set_4` | 150 | ✅ **PASS** | 0 | 0 |
| `calib4/set_5` | 150 | ✅ **PASS** | 0 | 0 |
| `calib4/set_6` | 300 | ✅ **PASS** | 0 | 0 |
| `calib4/set_6shuf` | 300 | ✅ **PASS** | 0 | 0 |
| `calib4/set_6_100` | 100 | ✅ **PASS** | 0 | 0 |
| `calib4/set_6_150` | 150 | ✅ **PASS** | 0 | 0 |
| `calib4/set_7` | 310 | ✅ **PASS** | 0 | 0 |
| `calib4/set_8` | 172 | ✅ **PASS** | 0 | 0 |
| `calib4/set_9` | 176 | ✅ **PASS** | 0 | 0 |

⇒ **两条被推翻的结论**：
1. 「校准集 v2 家族（set_1/2/3）优于基线」⇒ **作废**（整族泄漏；量化器的激活范围就是校准帧的激活分布，见过评测帧 ⇒ 该轨 F1 乐观、臂与臂不可比）。
2. 「A/B/C/D 极差仅 0.048 ⇒ 校准集轴无效」⇒ **结论方向后来被干净集复评独立证实（仅覆盖 set_6），但原依据（B/C/D）有污染** ⇒ 必须以干净集为准重述（见 B3）。

### B3. ⭐ 干净集复评：校准集轴**不是一条能单独兑现的杠杆**（分层判定，**不得越级泛化**）

**（1）子集口径（板端，干净集）** —— set_4 / set_5 / set_6 在**各自擅长的轨**上确有收益，且**不是泄漏产物**
（全部 PASS 干净，`27_check_calib_leak.py`），但每一份收益都在**另一轨上被抵消**（"一维此消彼长" = 宽↔窄量化范围的取舍）：

| 校准集 | 擅长的轨（子集口径） | 被抵消的另一轨（子集口径） |
|---|---|---|
| set_5 | T1 **0.8692@0.35**（基线 0.8606，**+0.0086**，FP 94→70） | T3 **0.6891@0.45**（基线 0.7090，**−0.0199**） |
| set_4 | T2 **0.9578@0.35**（基线 0.9428，**+0.0150**） | T3 **0.6935@0.40**（基线 0.7090，**−0.0155**） |
| set_6 | T2 **0.9628@0.40**（基线 0.9428，**+0.0200**） | T3 **0.6915@0.40**（基线 0.7090，**−0.0175**） |

原始子集全表（板端，`26_board_screening_round.sh` 产物，`screen_board_r{2,3,4,5}/<manifest>/<manifest>_threshold_sweep.json`，
各臂在**自身最优阈值**下的 F1）—— 这一层的"收益"在全量上会缩水：

| 校准集（全部 PASS） | 001324/2000 | anti_uav_sub2500/2500 | golden_pool_sub1200/1197 |
|---|--:|--:|--:|
| **set_A**（基线） | **0.7090@0.45** | 0.8606@0.30 | 0.9428@0.35 |
| set_4 | 0.6935@0.40 | 0.8603@0.30 | **0.9578@0.35** |
| set_5 | 0.6891@0.45 | **0.8692@0.35** | 0.9508@0.30 |
| set_6 | 0.6915@0.40 | 0.8609@0.30 | **0.9628@0.40** |
| set_7 | 0.6968@0.45 | 0.8622@0.35 | 0.9444@0.35 |

**（2）全量口径（板端，**唯一被全量验证过的干净集 = set_6**，int8 vs 同一 PT float 参照臂，同帧同 GT，9 个阈值离线重打分，
产物 `/mnt/data/siping/ptq_work/full_scale_sweep_set6/full_scale_sweep.json`，
模型 `screen_r2/p0_off_6_normal.rknn` md5 `bc252858d528c8217b7e81fcb24f1df9`）**：

| 轨 | 帧数 | 基线 set_A @0.30 | **set_6 @0.30** | 基线最优 | **set_6 最优** | **Δ(各自最优)** |
|---|--:|--:|--:|--:|--:|--:|
| **T3 blind 001324** | 11,810 | 0.6664 | 0.6674 | 0.7581@0.55 | **0.7517@0.55** | **−0.0064** |
| **T2 黄金池** | 17,187 | 0.9558 | 0.9562 | 0.9570@0.55 | **0.9570@0.55** | **≈0.0000**（精确 −0.00007） |
| **T1 Anti-UAV** | 31,613 | 0.8603 | **待跑**（`full_scale_r2` 的 anti 轨仍在采集，产物同目录） | 0.8606@0.35 | 待跑 | 待跑 |

复现命令（服务器 `conda activate rknn`，**纯离线重打分，不占板卡**）：

```bash
python -u manu/pipeline/npu/ptq/23_full_scale_threshold_sweep.py \
  --run-dir /mnt/data/siping/ptq_work/full_scale_r2 \
  --bmp-root /mnt/data/siping/ptq_work/bmp \
  --anti-root /mnt/data/siping/ptq_work/pcfeat_bmp/images \
  --out /mnt/data/siping/ptq_work/full_scale_sweep_set6 \
  --tracks blind golden \
  --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70
```

⇒ **全量口径读数**：T3 blind 11,810 帧 **0.7517@0.55** vs 基线 **0.7581@0.55**（**−0.0064**）；
T2 黄金池 17,187 帧 **0.9570@0.55** vs 基线 **0.9570@0.55**（**−0.00007**）⇒ **子集收益不迁移**。

**（3）⭐ 因此可下的结论（分两句，不要合成一句）**：
1. 「**校准集轴不是一条能单独兑现的杠杆**：它在子集口径上确有收益，但收益是**一维此消彼长**（宽↔窄量化范围的取舍），
   且在唯一被全量验证的干净集 set_6 上被证伪（T2 持平、T3 变差）。」
2. 「**set_4 / set_5 的子集收益是否在全量口径成立，目前未知**（尚未做全量）⇒ 若要判定它们，只能走全量口径，不能引用子集数字。」

⇒ **「黄金池 +0.020」是子集假象**（1197 帧 0.9628 vs 0.9428 → 全量 17,187 帧变成 0.9570 vs 0.9570）。
> （注：set_4 / set_5 的子集收益尚未做全量验证，不能一并断言为假象。）
⇒ **选型口径本身会改变赢家**：固定 `th=0.30` 口径下 set_6 在 001324 是 0.6585 vs 基线 0.6448（**看起来三轨全胜**），
但"各臂各自最优阈值"口径下 001324 是 0.6915 vs 0.7090（**反而更差**）⇒ 报选型结论必须写明用的是哪个口径。

### B4. ⚠️ 三条方法论闸门（**项目级规则，从此强制**）

| # | 闸门 | 触发过的真实翻车 | 落地动作 |
|:--|:--|:--|:--|
| 1 | **任何校准集对比必须先过 `27_check_calib_leak.py`** | v2 家族（set_1/2/3）整族泄漏；原 A/B/C/D 的 B/C/D 泄漏 ⇒ 一度把"四集极差 0.048"当成"校准集轴无效"的证据 | 校准集**必须**落盘 `set_N.txt` + `set_N_sources.txt`，建完立刻跑自检，**FAIL 的集不准进任何对比** |
| 2 | **子集 F1 不是选型依据，必须全量复核** | 同一类错误第 3 次出现：100 帧 → 2000 帧 → 全量。v2set3（泄漏）模拟器 +0.050 → 板端 0.0000；set_6 黄金池子集 +0.020 → 全量 +0.0000 | 板端候选排序一律用**全量**（60,610 帧）；子集只用于**诊断**与**排期** |
| 3 | **热图余弦（cos_hm）不能当选型判据** | hyb1 在 001324 的 cos_hm 最好（0.97378 vs base 0.95638）但 F1 不占优（召回 TP −58）；`layer` 的 cos 更差却 F1 更高 | 排序用**端到端 F1**；cos/mse 只用于**诊断量化器健康度** |

> 附：PC 模拟器本身的保真度已实测（**方法层结论仍成立**）—— 模拟器 vs 板端 ΔF1 ≤ 0.005
> （001324 int8：板端 0.6448 / 模拟器 0.6402；fp16：0.7395 / 0.7398）；
> 模拟器的 PT float 臂与板端 PyTorch 参照臂**逐项完全相同**（001324：F1 0.7484 / TP 1282 / FP 171 / FN 520；
> 黄金池 0.9955 / 1110 / 3 / 4）。
> ⇒ **模拟器足以做相对排序，绝对数字仍以板端为准**；剩余差异来自 NPU 定点算术 vs CPU 仿真（阈值附近的峰值翻转）。

### B5. ⭐ 输入 dtype 硬闸门（独立成节，2.3.2 复验 + 零板卡判定法）

**只有「toolkit 自己全图 int8 量化」这一条路径能让图的输入保持 INT8。**

| 路径 | 图的输入算子 | 板端 `inputs_set` | NPU 内部内存 | 板端帧率 |
|---|---|--:|--:|--:|
| **toolkit 全图 int8**（`p0_off_A_normal` / `a2f`） | **INT8** | **0.34 ms** | 20,400 KB | **13.16 fps**（001324 2000 帧 152.0 s） |
| `--no-quant`（fp16） | FLOAT16 | 45.41 ms | 105,600 KB | 6.48 fps（308.6 s） |
| `quantized_hybrid_level=1/2/3` | FLOAT16 | 45.41 ms | 105,600 KB | — |
| **QDQ ONNX + `build(do_quantization=False)`**（= QAT 的 A0/A1/A2/A1b） | FLOAT16 | 45.41 ms | 105,600 KB | 5.64 fps |

- 根因（`archive/02` §28 已记，2.3.2 复验）：toolkit 把「**图的输入张量 dtype**」与「**计算精度**」绑死
  ⇒ 没有开关能让输入停在 int8 而图内混精度。
- ⭐ **零板卡成本的判定方法（新做法）**：构建时开 toolkit verbose，构建日志的算子表里有
  `D RKNN: … InputOperator  <dtype>  CPU  (1,3,640,640)` 一行，**直接 grep 即可**；
  `02_batch_ptq_pc_screening.py` 已实现（`--verbose-build` + 报告字段 `input_op_dtype`）。
  ⚠️ `verbose=False` 时该表**不打印**（实测 0 次命中）⇒ 不开 verbose 就判不了。
- 板端帧率出处：`logs/tier2_fixed.log`（同一 manifest `001324_sub2000`、同批 BMP、同一脚本 `04_run_board_fast_eval.py`）。

### B6. ⭐ QAT 路线的最终判定（**负结果**，详见 `manu/memory/QAT_RK3588_交付报告.md`）

**在 rk3588 + toolkit 2.3.2 上，QAT 不能兼得「比 int8 基线更准」与「INT8 输入的延时」。**

- **三轨对照（板端，同口径）**：T3 `001324_sub2000` int8 基线 **0.7090@0.45**（量化损失 0.0247）**胜** A2 **0.6863@0.45**（0.0473）0.0227；
  T2 黄金池 **A2 0.9509@0.25**（0.0446）**胜** int8 基线 0.9428@0.35（0.0527）0.0081；T1 tier1/100 **打平**（都 0.8352）。
- **阈值重标定对基线的收益（0.6448→0.7090 = +0.0642）是对 A2 收益（0.6552→0.6863 = +0.0311）的两倍。**
- `a2f_qatw_toolkit_int8`（QAT 权重走 toolkit 原生全 int8）是**唯一 InputOperator=INT8 的 QAT 产物**
  （149 算子全 INT8、算子表与基线逐项相同），但**精度爆炸**：板端 @0.30 FP T1 20→127 / T2 68→700 / T3 527→4433。
- **a2f 爆炸的四刀定位（根因＝权重/训练问题，不是量化问题）**：
  1. ORT 跑 **float** ONNX 已爆：检出比 QAT/base = tier1 ×4.12、黄金池 ×2.94、001324 **×6.55**（7847→51373）；
     001324 @0.30 FP 171→**2582**（×15.1）、F1 0.7484→**0.3671** ⇒ **排除「量化范围被离群值撑爆」**；
  2. 两张 ONNX 算子直方图一致（Conv 85 / Mul 93 / Sigmoid 77 / Softmax 1 / MaxPool 3 / Resize 5…），仅 Constant 64→61 ⇒ **排除「导错图」**；
  3. RKNN 算子表逐项相同（149 算子全 INT8、输入 INT8）⇒ **排除「构建/导出 bug」**；
  4. 权重级 diff：494 张量 **0 个 key/shape 问题**、66 个冻结未动；
     **`head.heatmap.0.weight` rel_l2=0.266、均值 +0.004326→+0.006635（+53%）**
     ⇒ **根因＝QAT 把 heatmap 头搬坏（logit 正偏置 → sigmoid 饱和 → 过自信 → 固定阈值下 FP 爆炸）**。
- **训练损失变好但部署指标变差**：`train_loss` 0.3487→**0.3243**，而 F1@0.30 0.7817→0.5304。
  focal+offset 不约束热图绝对标度 ⇒ **训练损失不得作为 QAT 有效的证据**。
- 另一条负面结论：**QAT checkpoint 与它学到的 clip 绑死** —— QAT 权重单独跑 fp16 得 0.3595，
  交给 toolkit 自己的全 int8 标定得 0.2574，只有连同 `*.qdq_meta.json` 的 scale 一起发布（A2）才可用。
- **部署建议：维持 toolkit 原生 int8 + 逐轨阈值重标定**（T3 0.45 / T2 0.30 / T1 0.50）。
- **下一步最高价值**：把 LSQ 从 3 个头部位点扩到全网络（约 85 个 conv），让 Q/DQ 覆盖主体，
  `build(do_quantization=False)` 才能既保住 INT8 输入又不让主体退化成 fp16。
- 未完成：QAT 臂的**全量集合**（T3 11810 / T2 17187 / T1 anti 31613）没跑（板端 ≈5.8 h，且板卡被别的 harness 占用）。

### B7. 🔴 本轮对既有记忆的**两处更正**

1. **`mean_values`/`std_values` 必须给** —— 与旧记录「Q/DQ ONNX 不得传 mean/std」**相反**。
   实测：不传 mean/std 时，板端驱动上传的 uint8 缓冲区被 runtime 直接加宽成 FLOAT16（0..255），图里没有 `/255`
   ⇒ 网络看到 255 倍大的输入 ⇒ 饱和 ⇒ **null control 直接判死**（`FATAL: NPU output independent of input`）；
   给了 `mean=[[0,0,0]]` / `std=[[255,255,255]]` 则 RC=0 且 null control 显示 `heatmap CHANGED`。
   **更正了 `17_probe_qdq_import.py` 探针的一处误读**：探针里 null control 报 `IDENTICAL(BAD)`，
   当时归因为「那帧没无人机」，实际是缺 mean/std 造成的 255 倍输入错误；
   探针的 `QDQ_ADOPTED` 结论**仍成立**（它读板端 `out[N]` 的 zp/scale，与输入数值无关）。
   ⚠️ 这**不是**「重复归一化」：mean/std 描述的是**输入张量**，Q/DQ 描述的是**激活 clip**，两者正交。
2. **uint8 图输入的 ONNX 在 build 期被拒**：`Not Support Dtype: 2` / `Missing quantization parameters for images`
   （与 2026-10-16 在 2.3.0 上的记录一致，2.3.2 依旧）。`22_export_qdq_onnx.py` 新增 `--uint8-input`（默认 **0**）。

### B8. 环境事实（2 条）

1. **工具链 build 不是 bit-reproducible**：同一配方三次构建得到 `75d84fcd…` / `1de9cb3b…` / `cedaeb74…`
   三个不同 md5（`models/p0_off_A_normal_rebuild.rknn` / `sim25/smoke_base/rknn/p0_off_A_normal.rknn` / `models/p0_off_A_normal.rknn`），
   但**数值完全一致**（100 帧先验逐项复现：base @0.30 TP 77 / FP 17 / F1 0.7299）⇒
   **引用 `.rknn` 必须带 md5，但数值结论不受影响**。
2. **板卡 `/mnt/manu` 的 NFS 挂载不随工作站重启存活**（板端无 fstab 项）⇒ 每次工作站重启都要手动
   `mount -t nfs 192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu -o nolock,soft,timeo=5,retrans=2`。

### B9. th=0.25 与历史 trial-0474 的同阈对齐（**子集口径**，未闭合）

- 板端全量 TSV 是在 **th=0.30** 采的 ⇒ **th=0.25 无法从现有 TSV 重算**（0.25~0.30 段分数没采到）。
- PC 模拟器（`25_pc_sim_threshold_sweep.py`，collect-th=0.05）在 Anti-UAV 轨按序列均匀抽样
  **2,493 帧（31,613 帧的 7.9%，24 序列，1,982 GT）**：

| th | int8 F1（模拟器） | PT float F1（模拟器） | Δ |
|---|---|---|---|
| 0.20 | 0.8506 | 0.8608 | −0.0103 |
| **0.25** | **0.8590** | **0.8732** | **−0.0142** |
| 0.30 | 0.8631 | 0.8836 | −0.0204 |

- 历史 trial-0474 **全量** @0.25 同点口径 = **−0.020986**；本轮全量板端 @0.30 = −0.0189。
  ⇒ **−0.0142 与 −0.0210 同量级但不相等（差 +0.0068）**。
- 归因三选一**未拆开**：(a) 子集只占 7.9%（但板端同子集 @0.30 = 0.8606 vs 板端全量 @0.30 = 0.8603 ⇒ 采样代表性高，**不是主因**）；
  (b) 模拟器 vs 真 NPU（ΔF1 ≤ 0.005）；(c) **模型代际差**（trial_0080 vs trial-0474）。
- 🔴 **明确声明**：th=0.25 与历史的**全量**同阈对齐必须由板端以 collect-th ≤ 0.20 重采一次全量 TSV
  ⇒ **属板卡资源，不在本轮范围**。

### B10. 本节结清情况 / 待补

| 项 | 状态 |
|---|---|
| PTQ 配置轴全表（dtype / algorithm / method / hybrid / optimization_level / flash_attention） | ✅ 本节 B1 |
| 校准集泄漏自检（16 个集全查） | ✅ 本节 B2 |
| 干净集复评（校准集轴最终判定，**全量只验证过 set_6**） | ✅ blind + golden（B3）；⏳ **anti 全量待跑**（`full_scale_r2` 仍在采）；⏳ **set_4 / set_5 的子集收益尚未做全量，是否成立未知**（B3 第 (3) 条） |
| 输入 dtype 硬闸门独立成节 + 零板卡判定法 | ✅ 本节 B5 |
| QAT 最终判定 | ✅ 本节 B6（详见 `QAT_RK3588_交付报告.md`） |
| 两处更正 | ✅ 本节 B7 |
| 环境事实 | ✅ 本节 B8 |
| th=0.25 全量同阈对齐 | ⏳ **未做**（需板端重采全量 TSV，板卡被别的 harness 占用） |
| QAT 臂的全量三轨 | ⏳ **未做**（板端 ≈5.8 h，板卡被别的 harness 占用） |
