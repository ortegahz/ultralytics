# 视频质量指标挖掘：好视频组 vs 坏视频组（2026-11-19）

> **执行机**：`huangzhe@192.168.99.40:32222`（主机名 `test-Super-Server`，本机直跑，**未用 GPU**）
> **解释器**：`/home/huangzhe/anaconda3/envs/uav/bin/python`（Python 3.10.20；numpy 1.26.4 / scipy 1.15.3 / opencv 4.10.0）
> **工作目录**：`/tmp/pycharm_project_10ae9e2e`（= `/mnt/data/siping/runs/npu20ms/nano_search/repo`）
> **脚本目录**：`manu/video_quality/`　**产物目录**：`manu/video_quality/out/`（逐帧指标）、`manu/video_quality/report/`（对比与图表）
> **对外汇报件**：`manu/video_quality/report/REPORT_视频质量指标.md`

---

> **📌 图表落点说明（铁律二十五，2026-11-24）**：**代码产生的中间产物一律出仓库目录** —— 本档正文与结论表
> （`.md` / `.csv` / `.json`）仍随仓库入库，**图表（`.png`）本身不再入库**。
> 全部图表的 canonical 落点 =
> `/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/`。
> 该目录 **1:1 镜像仓库相对路径**（把仓库相对路径原样拼到 `repo-artifacts/` 后面即命中 ⇒ **可原路还原**），
> 图表也可由 `manu/video_quality/vq_*.py` 整批重生成。

## 一、结论（事实层）

**跨两个模态真正共同的失效特征是「糊 + 灰 + 压」，不是「晃」。**

- 跨模态一致成立的 **3 个有效指标**：**全局对比度 `global_contrast`、边缘密度 `edge_density`、拉普拉斯清晰度方差 `lap_var`**。
- 红外模态另有 **2 个有效指标**：**块效应 `blockiness`、灰度熵 `entropy`**（可见光上区分倍数 <1.2，不作红线）。
- 合计 **5 个有效指标** ⇒ **8 条核心红线**（IR 5 条 + VIS 3 条）+ **2 条可见光弱红线** = **第三节表里的 10 条阈值**。
- **红线分级口径（先过候选集、再分级）**：候选集 = 2.1/2.2 表里判定为「✅ 有效」的指标（**不含「✅ 有效（描述性）」**；窗口级判别力 ≥ 0.70、显著性成立、**方向正确**、两组中位数拉开 > 1.2×）；**在候选集内**按阈值处平衡准确率分级：≥ 0.90 = 核心红线（拒绝接入），< 0.90 = 弱红线（只告警）。**红外「平均亮度」只作描述性统计、不进候选集**——亮度随场景/昼夜变化，不是画质缺陷，即使阈值处平衡准确率有 1.000 也不做红线。**注意**：VIS「帧间位移速率」「帧间缩放(%)」「帧间位移」「帧间最大位移」视频级平衡准确率 0.958/0.958/1.000/0.958，窗口级判别力只有 0.637/0.638/0.665/0.680、**没进候选集** ⇒ 不进红线（红外的位移口径不进红线是**方向相反**，属另一个原因）。
- **「晃」要分模态看**：红外的坏片比航拍好片**更稳**（方向相反）；可见光的坏片整段确实**更晃**（`shift_frac` 3.93×、视频级完全可分），但窗口级判别力只有 0.665、未达 0.70 门槛，只作参考不进红线。
- **参考方法脚本的 4 个指标已全部落地实测**：帧间位移 P95、**帧间最大位移 `shift_frac_max`（max 口径）**、拉普拉斯清晰度 P5、红外粗糙度 ρ。`shift_frac_max` 由已存的逐帧 `shift_frac` 数组按 1 秒窗口取最大派生（`METRIC_SOURCE` 映射，**无需重跑批跑**）：IR 判别力 0.794（方向相反）、VIS 判别力 0.680（未达 0.70）、OSD 0.661 ⇒ **两个位移口径都不进红线**，结论与 P95 口径一致。
  **口径提醒**：报告 2.4 故障窗表里 `max` 行的「好组全局」取的是**整段视频最大值**（单帧离群敏感）、「坏组尾窗」只有 125 帧 ⇒ **倍数被放大、不可直接对外引用**；**视频级（2.1/2.2 表）用的是「各 1 秒窗口最大值的中位数」**，那才是稳定口径。
- **粗糙度 ρ 两个模态都不可用**：产物口径 `expected_direction=+1`（值越高越差），实测**两模态都是坏组更低**（`separation=bad_low`）⇒ 按 2.5 判据**两个模态都属「方向相反」**；**表象差别在判别力门槛**：VIS ρ 判别力 0.966 达标（2.2 表判「⚠️ 方向相反·不可作判据」）、IR ρ 判别力只有 0.694、**未达 0.70 门槛**（2.1 表只判「⚠️ 弱」，故未进 2.5 清单）。**两模态都不进红线**：ρ 测的是「糊」不是「脏」（越低 = 梯度能量越少 = 越糊），与已进入红线的清晰度 / 灰度类指标（`lap_var`/`edge_density`/`global_contrast`/`entropy`）**信息冗余**。
- **弱红线只能当提示**：阈值处平衡准确率 0.708、**TPR 仅 0.50**（2 条坏 VIS 里每条弱红线只抓到 1 条：`120BT-04-可见光.mp4` 的 `illum_jump` 0.001587 < 0.001680、`stripe_idx_row` 32.99 < 33.53，均不触发）⇒ **本批能拒绝坏片完全靠 3 条核心红线全部越线**。

**判定规则（分级，不可简化成「任一条越线即拒」）**：

| 条件 | 动作 | 本批 25 条回溯结果 |
| :--- | :--- | :--- |
| 该模态**全部核心红线**都越线 | 拒绝接入 | 坏组命中 **4/4**，好组误拒 **0/18** |
| 命中 **≥1 条**核心红线（未全部） | 人工复核（**不自动放行**） | 好组 1 条、坏组 0 条 |
| 仅命中**弱红线** | 通过，但记录并抽检 | 好组 2 条（12 条好 VIS 里 2 条，正常误报，不阻断接入） |
| 一条不越 | 通过 | 好组 15 条 |

## 二、数据构成（先把话说清楚）

- **好视频组**：18 条源视频（红外 6 + 可见光 12），30 fps 裸 H.264。
- **坏视频组**：真正可比的**源视频只有 4 条**（红外 2 + 可见光 2，25 fps mp4）；另 3 条是**操作员桌面录屏**（Windows 桌面 + 地面站 UI），不参与统计。
- 规模：25 条片、**307,910 帧**，批跑 `rc=0`，25/25 `status=ok`（`manu/video_quality/out/run_manifest.json`）。

### 三个必须记下的坑

1. **坏组 IR 有大面积黑边（pillarbox）**：不裁边直接算，黑边硬边缘会把拉普拉斯/梯度能量抬高，**把坏片「测成」清晰**。必须先检测裁掉再缩放到宽 640。
2. **文件名不可信**：`130CT-02可见光_OSDT.mp4` 名字写可见光，实测是红外/可见光来回切换（**81.0% 时长是可见光**，**290/358** 个 5 秒采样点判为彩色；**首个彩色采样点在第 195.0 s** ⇒ 开头约 3 分钟是红外灰度）。模态按**全片像素饱和度多数表决**判定（`vq_run_server.py` 的 `probe_modality`，fps=1/5 采样），**不看文件名**。上述数字**已落盘**：逐片 `out/130CT-02可见光_OSDT.json` 的 `modality_vis_ratio` / `modality_samples` / `modality_coloured` / `modality_first_vis_seconds`（由 `vq_reclassify.py` 写入，报告里的占比是**从该产物读出来的**，不是写死的）。
3. **`osd/` 下 3 个「挂飞」文件实为桌面录屏**，不是视频源。证据图：`/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/evidence_挂飞录屏实为桌面录屏_t300s.png`。

## 三、核心数字（全部可在 `report/comparison.csv` 对上）

> **p 值口径**：精确置换检验为**双侧**（按 |AUC − 0.5| 判极端）。红外 6+2 只有 C(8,2)=28 种标签分配，
> **双侧 p 的最小可能值就是 2/28 = 0.0714** ⇒ 红外侧永远拿不到 p<0.05，显著性由窗口级 cluster CI 承担。
> **视频级统计量 = 「典型秒」**：逐帧 → 1 秒窗内取分位 → 全片中位数（长度无关，避免 343 帧与 50k 帧不可比）。

### 红外 IR（好 6 / 坏 2；窗口 409 vs 3,222）

| 指标 | 好中位 | 坏中位 | 倍数 | 坏组在 | 视频级判别力 | 窗口级判别力 | 双侧 p | 红线阈值 | 留一法 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `blockiness` | 1.046 | 2.051 | 1.96× | 高 | 1.000 | 0.991 | 0.0714 | 1.631 | 0.917 |
| `entropy` | 7.374 | 5.246 | 1.41× | 低 | 1.000 | 0.971 | 0.0714 | 6.417 | 0.750 |
| `global_contrast` | 0.7732 | 0.2897 | 2.67× | 低 | 1.000 | 0.930 | 0.0714 | 0.4693 | 0.750 |
| `edge_density` | 0.05528 | 0.007018 | 7.88× | 低 | 0.917 | 0.802 | 0.1429 | 0.02016 | 0.917 |
| `lap_var` | 104.3 | 38.28 | 2.73× | 低 | 0.917 | 0.729 | 0.1429 | 57.32 | 0.667 |

### 可见光 VIS（好 12 / 坏 2；窗口 703 vs 3,140）

| 指标 | 好中位 | 坏中位 | 倍数 | 坏组在 | 视频级判别力 | 窗口级判别力 | 双侧 p | 红线阈值 | 留一法 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `lap_var` | 2299 | 44.73 | **51.4×（表观）** | 低 | 1.000 | 0.990 | 0.0220 | 534 | 1.000 |
| `edge_density` | 0.1207 | 0.007725 | 15.6× | 低 | 1.000 | 0.952 | 0.0220 | 0.03637 | 1.000 |
| `global_contrast` | 0.6708 | 0.3413 | 1.97× | 低 | 1.000 | 0.903 | 0.0220 | 0.4756 | 1.000 |
| `illum_jump`（弱） | 0.001124 | 0.003803 | 3.38× | 高 | 0.958 | 0.759 | 0.0440 | 0.00168 | 0.750 |
| `stripe_idx_row`（弱） | 27.15 | 39.22 | 1.44× | 高 | 0.917 | 0.762 | 0.0879 | 33.53 | 0.708 |

> **阈值算法（方向感知，可当场复算）**：坏组偏高型取 (好组 P90 + 坏组 P10)/2，例如块效应 = (1.347 + 1.916)/2 = **1.631**；
> 坏组偏低型取 (好组 P10 + 坏组 P90)/2，例如灰度熵 = (6.972 + 5.862)/2 = **6.417**。
> **留一法**是把每条片轮流拿掉再重算阈值与平衡准确率取最差值（外推下界）。**红外三条核心留一后只剩 0.667~0.750**，
> 原因是坏组只有 2 条 ⇒ **这是「上线前必须扩样重标」的直接证据**。

## 四、否定性结论（能分开 ≠ 能用；必须量中间量）

### 4.1 方向相反，不能作红线（13 个「指标对但方向反」的组合）

| 模态 | 指标 | 好中位 | 坏中位 | 窗口级判别力 | 原因 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| IR | `stripe_idx_row` | 46.26 | 12.59 | 0.953 | 条纹指数随画面变糊而**下降** |
| IR | `shift_rate` | 0.0476 | 0.00706 | 0.825 | 坏片是机头朝地/悬停，**比航拍好片更稳** |
| IR | `shift_frac` | 0.001587 | 2.824e-04 | 0.808 | 同上 |
| IR | `shift_frac_max` | 0.001676 | 3.554e-04 | 0.794 | 同上：最大位移也是坏组更小 |
| IR | `col_bias` | 0.1063 | 0.06814 | 0.765 | 方向相反 |
| IR | `noise_sigma` | 0.3353 | 0.138 | 0.764 | 糊图高频少 → 实测噪声更小；按 4.3 节折算回原生尺度后差距 4.5×，方向仍是「坏组更小」 |
| IR | `row_bias` | 0.1555 | 0.09521 | 0.757 | 方向相反 |
| VIS | `snr_db` | 39.13 | 54.5 | 0.975 | 糊图更平滑 → 信噪比**反而更高** |
| VIS | `noise_sigma` | 1.304 | 0.2283 | 0.974 | 糊图高频少 → 实测噪声更小；按 4.3 节折算回原生尺度后方向不变 |
| VIS | `sat_ratio` | 0.009864 | 0 | 0.972 | 坏片更灰更暗 → 近白像素更少 |
| VIS | `grad_energy_ratio` | 0.1497 | 0.03604 | 0.966 | ρ 不是「脏」而是「糊」 |
| VIS | `dog_snr_max` | 86.02 | 182.5 | 0.735 | 糊图上最像目标的区域对比度反而更高 |
| VIS | `dog_snr_top8` | 81.27 | 169.9 | 0.729 | 同上 |

### 4.2 完全无区分度（窗口级判别力 ≈ 0.5，直接剔除）

IR：`stripe_idx_col` 0.545、`flow_resid_frac` 0.512、`scale_pct` 0.504、`sat_ratio` 0.501；
VIS：`flow_med_frac` 0.546、`dark_ratio` 0.530。

### 4.3 归一化画幅的尺度偏置（**已实测**，`report/resample_bias.txt`，脚本 `manu/video_quality/vq_bias.py`）

缩放到宽 640 会**整体抬高**依赖像素尺度的指标（实测 = canonical640 / 原生分辨率，每条片 120 帧）：

| 片单（组/模态/内容框宽） | 噪声 sigma | 图像信噪比 | 梯度能量比 rho | 拉普拉斯清晰度 |
| :--- | :--- | :--- | :--- | :--- |
| good / ir / VIDEO00007（640 宽） | 1.00× | 1.000× | 1.00× | 1.00× |
| good / ir / VIDEO00008（640 宽） | 1.00× | 1.000× | 1.00× | 1.00× |
| good / vis / VIDEO00011（1920 宽） | 2.67× | 0.828× | 1.91× | 7.21× |
| good / vis / VIDEO00015（1920 宽） | 2.51× | 0.842× | 1.83× | 7.20× |
| good / vis / VIDEO00019（1920 宽） | 2.41× | 0.846× | 1.77× | 6.97× |
| bad / ir / 120BT-04-红外02（1280 宽） | 1.93× | 0.917× | 1.81× | 2.01× |
| bad / ir / 130CT-02红外_IRT（1280 宽） | 1.84× | 0.920× | 1.81× | 2.07× |
| bad / vis / 120BT-04-可见光（1920 宽） | 2.69× | 0.851× | 2.46× | 6.01× |
| bad / vis / 130CT-02可见光_OSDT（1280 宽） | 1.85× | 0.919× | 1.81× | 2.08× |

**五个结论（都有数）**：

1. **红外侧偏置是保守的 ⇒ 表里的差距被低估**：坏组被抬高 2.01~2.07×、好组不缩放（1.00×）。把抬升折算回原生尺度后，
   **红外拉普拉斯清晰度的真实差距是 5.6×**（表里写的 2.73×）⇒ **真实差距比表中更大，结论方向不会因偏置反转**。
2. **可见光侧偏置方向相反 ⇒ 表里的倍数要打折**：好组 VIS 是 1920 宽（抬高 6.97~7.21×），
   而坏组 `130CT-02可见光_OSDT.mp4` 内容框只有 1280 宽（只抬高 2.08×，其内容框是在开头红外时段检测到的），
   **两组被抬高的幅度不一样**。折算回与好组相同的原生尺度后，**可见光清晰度差距约 29×**（表里写的 51.4×）
   ⇒ **差距依然远大于 1、结论不变，但不能拿 51× 这个数字对外说**。
3. **σ / ρ / SNR 被剔除的理由是「方向」而不是「偏置」**：折算回原生尺度后，**红外噪声 σ 的差距是 4.5× 且方向仍是「坏组更小」**
   （好 0.3353 vs 坏 0.138）⇒ **方向反转是真实的，不是归一化造成的假象**。
4. **可见光 29× 是保守下界**：`130CT-02可见光_OSDT.mp4` 的偏置实测只取开头 120 帧（≈4.8 s），实测落在它的**红外时段**
   （与纯红外片 `130CT-02红外_IRT` 的原生值逐项重合：`grad_energy_ratio` 0.00792/0.00792、`noise_sigma` 0.04207/0.04209、
   `lap_var` 18.97/19.06）⇒ **它的 2.08× 抬升系数其实是红外内容系数**；若改按好组可见光内容系数（6.97~7.21×）折算，
   差距会**更大**（≈45×）。**折算口径**：好组侧取 3 条实测好片系数的**均值**、坏组侧**逐片去偏后取中位数**（好组改中位数口径为 29.13×）。
5. **偏置实测覆盖 9/25 条片**（好 IR 2/6、好 VIS 3/12、坏组 4/4），其余按同内容框宽度外推。

## 五、回溯判定里的边界情况（必须一起汇报）

- 好组 `VIDEO00007_19700101_012038IR.h264` 命中 **3/5** 条红外核心红线（`lap_var` 37.49、`edge_density` 0.01203，
  **低于坏组最大值 49.44 / 0.01209**），落到「人工复核」档。
- 坏组 `130CT-02红外_IRT.mp4` 命中 **5/5**，落到「拒绝接入」档。
- **这两片在清晰度/边缘密度上是真的重叠的** ⇒ 无参考全图指标**分不清「画质差」与「场景没内容」**
  （该好片大概率拍到大面积均匀天空/地面）。
- ⇒ **红线只适合做首轮筛查，不能作为自动拒绝的唯一依据；「人工复核」这一档必须保留。**
- **核心红线在坏片故障窗（最后 5 秒）里照样全部越线**（`report/tail_compare.csv`）⇒ 同一套阈值既可用于接入前体检，
  也可用于在线实时告警。

## 六、复现命令（可直接复制粘贴）

```bash
# 执行机：huangzhe@192.168.99.40:32222    工作目录：/tmp/pycharm_project_10ae9e2e
cd /tmp/pycharm_project_10ae9e2e
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mplcfg
#    产物根（服务器 canonical = /mnt/data/siping/runs/；本机跑时覆盖到 workspace/cache/ultralytics/）
#      VQ_PROD=/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/video_quality
VQ_PROD="${VQ_PROD:-/mnt/data/siping/runs/video_quality}"

# 1) 逐帧指标提取（25 条片；4 worker；cv2.setNumThreads(1)；ffmpeg -threads 1；60s 看门狗 + 20% 内存刹车）
bash manu/video_quality/vq_launch_server.sh 4
#    日志：manu/video_quality/out/run.log（结束标志 === done rc=0；逐片 [DONE] 行）

# 1b) 模态重分类（按像素饱和度重判 ir/vis，并把采样占比与首个彩色采样点落盘）
#     批跑用的就是同一套 probe_modality，但产物里必须带上 modality_vis_ratio /
#     modality_samples / modality_coloured / modality_first_vis_seconds 四个字段，
#     否则报告里的「81.0% 可见光」「开头约 3 分钟是红外灰度」无处可查。
python manu/video_quality/vq_reclassify.py --metrics-dir manu/video_quality/out
#     幂等：重复执行输出 "reclassified 0 clips"；同时把最终分组同步进
#     out/run_manifest.json 的 modality_final（批跑初判 modality 保留为历史记录）

# 2) 好/坏对比 + AUC + 阈值挖掘 + 出图（~1.5 min）
python manu/video_quality/vq_compare.py \
  --metrics-dir manu/video_quality/out --report-dir manu/video_quality/report

# 3) 统计严谨性：精确置换检验（双侧）+ clip/cluster bootstrap CI + 红线阈值 + LOCO（~1 min）
python manu/video_quality/vq_stats.py \
  --metrics-dir manu/video_quality/out --out manu/video_quality/report

# 4) 跟丢切片对比（坏片故障窗 vs 好片全局，~1 min）
python manu/video_quality/vq_tail.py \
  --metrics-dir manu/video_quality/out --out manu/video_quality/report --slice-seconds 5.0

# 5) 归一化尺度偏置实测（每条片 120 帧，原生分辨率 vs 640 归一化各算一次，~1 min）
python manu/video_quality/vq_bias.py \
  --clip "bad IR  120BT-04-红外02=/mnt/data/xxxu/纵横挂飞/120BT-04-红外02.mp4" \
  --clip "bad IR  130CT-02红外_IRT=/mnt/data/xxxu/纵横挂飞/130CT-02红外_IRT.mp4" \
  --clip "good IR VIDEO00007=/mnt/data/xxxu/朗基天香20260903/VIDEO00007_19700101_012038IR.h264" \
  --clip "good IR VIDEO00008=/mnt/data/xxxu/朗基天香20260903/VIDEO00008_19700101_012152IR.h264" \
  --clip "bad VIS 130CT-02可见光_OSDT=/mnt/data/xxxu/纵横挂飞/osd/130CT-02可见光_OSDT.mp4" \
  --clip "bad VIS 120BT-04-可见光=/mnt/data/xxxu/纵横挂飞/osd/120BT-04-可见光.mp4" \
  --clip "good VIS VIDEO00011=/mnt/data/xxxu/朗基天香20260903/VIDEO00011_19700101_010155VIS.h264" \
  --clip "good VIS VIDEO00015=/mnt/data/xxxu/朗基天香20260903/VIDEO00015_19700101_010552VIS.h264" \
  --clip "good VIS VIDEO00019=/mnt/data/xxxu/朗基天香20260903/VIDEO00019_19700101_011317VIS.h264" \
  --samples 120 --out manu/video_quality/report/resample_bias.txt

# 6) 生成中文汇报件
python manu/video_quality/vq_report.py \
  --metrics-dir manu/video_quality/out \
  --report-dir manu/video_quality/report --out manu/video_quality/report/REPORT_视频质量指标.md
#    报告里的模态占比是从 out/130CT-02可见光_OSDT.json 读出来的（不写死）

# 7) 独立复算验收（AUC 用秩和法重算，不 import 上面两个脚本）
python manu/video_quality/vq_verify.py \
  --metrics-dir manu/video_quality/out --report-dir manu/video_quality/report \
  --memory manu/memory/video_quality_metrics_20261119.md

# 8) 机械验收（27 项检查，完成判据）
python manu/video_quality/vq_acceptance.py \
  --metrics-dir manu/video_quality/out --report-dir manu/video_quality/report \
  --memory manu/memory/video_quality_metrics_20261119.md

# 9) 🔴 图表出仓库（铁律二十五 · 2026-11-24 补充裁决）
#    PNG 属「可由 vq_*.py 整批重生成」的中间产物 ⇒ **中间产物不许留在仓库目录**。
#    跑完立刻迁到产物根：服务器 ${VQ_PROD}/report；本机（覆盖 VQ_PROD 后）= workspace/cache/ultralytics/...
#    仓库只留报告正文 REPORT_视频质量指标.md 与结论级表（comparison/stats/tail_compare/knee_* 等）。
mkdir -p "${VQ_PROD}/report"
mv -f manu/video_quality/report/*.png "${VQ_PROD}/report/" 2>/dev/null || true
```

## 七、产物清单

| 产物 | 路径 | 说明 |
| :--- | :--- | :--- |
| 中文汇报件 | `manu/video_quality/report/REPORT_视频质量指标.md` | 可直接发领导 |
| 对比主表 | `manu/video_quality/report/comparison.csv` / `.json` | 78 行（26 指标 × 3 模态） |
| 统计表 | `manu/video_quality/report/stats.csv` / `.json` | 双侧精确置换 p、cluster CI、红线、LOCO |
| 尾窗对比 | `manu/video_quality/report/tail_compare.csv` / `.json`、`tail_slices.csv` | 故障窗 vs 好片全局 |
| 逐片指标 | `manu/video_quality/out/*.json` + `*.npz` + `run_manifest.json` | 25 条 × 逐帧数组；逐片 json 另含模态探测字段 `modality_vis_ratio` / `modality_samples` / `modality_coloured` / `modality_first_vis_seconds` / `modality_probe_seconds`。**三个模态字段的口径互不相同，别混用**：逐片 json 的 `modality` = **像素探测结果**（3 条桌面录屏是彩色画面，故也记 `vis`）、逐片 json 的 `kind` = `source`/`osd`，**统计分组靠 `kind` 过滤**（`vq_stats.PAIRS`：ir/vis 两个组都带 `kind=="source"`，osd 组按 `kind=="osd"` 取，故桌面录屏的 `modality` 记什么都不影响统计）；`run_manifest.json` 的 `modality` = **批跑初判**（历史记录，**永不改写**）、`modality_final` = **最终分析分组**（osd 片记 `osd`） |
| 图表 | `/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/chart_metrics_{ir,vis,osd}.png`、`/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/chart_auc_{ir,vis,osd}.png`、`/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/chart_timeseries_{ir,vis,osd}.png` | 🔴 图表不入库（铁律二十五）：落 cache 的 repo-artifacts 镜像目录，1:1 镜像仓库相对路径，可原路还原 |
| 桌面录屏证据 | `/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/evidence_挂飞录屏实为桌面录屏_t300s.png` | 同上 |
| 尺度偏置实测 | `manu/video_quality/report/resample_bias.txt` | 由 `vq_bias.py` 生成 |

**产物 md5（2026-11-20 重新逐个对过现产物；点时值，以现算为准）**：
`report/comparison.csv` `91d8fb72b150f0c71e3f436f92b06f05`；
`report/stats.csv` `699fca02f1cbb8beb3d750387f4cc781`；
`report/tail_compare.csv` `8c5ea848d5f2b7c6347fdb7630dc1e53`；
`out/run_manifest.json` `f8baaaf9a43f07d83325859a37d043c3`；
`report/REPORT_视频质量指标.md` `5ca9666f544abcfeb930b68c84e4b1b3`。

> 复算命令（在工程根目录执行，路径相对 `manu/video_quality/`）：
> ```bash
> md5sum report/comparison.csv report/stats.csv report/tail_compare.csv \
>        out/run_manifest.json "report/REPORT_视频质量指标.md"
> ```
> ⚠️ 这张表曾在 2026-11-19 写下时就是**错的**（3 个 csv 的 md5 记的是更早一版），已按现产物更正；今后任何重跑 `vq_compare.py` / `vq_stats.py` / `vq_tail.py` / `vq_report.py` / `vq_reclassify.py` 都会改变对应 md5，**改完必须重算这一行**。

## 八、局限（必须一起汇报）

1. **坏组源视频只有 4 条**（IR 2 + VIS 2，另有 3 条桌面录屏不算源视频），视频级统计量置信区间宽；**可见光只有 2 条坏样本，可见光侧结论只算「假设级」**；**第三节的阈值与 3.1 的分级规则都是在这 4 条坏片上标定的，上线前必须扩样重标**（红外三条核心红线留一法只剩 0.667~0.750）。
2. **没有 GT 框，未算真值局部 SCR**：两组视频目录内不存在任何 `.txt/.json/.yaml/.xml` 标注或初始框文件，参考方法的「|μt-μb|/σb」**无法按原样执行**；改用 **DoG 响应 ÷ 噪声 σ**（`dog_snr_max`/`dog_snr_top8`）作无框代理。**归一化位移（位移/目标尺寸）同样因缺框未算** —— 补齐标注后**必须重算**。
3. **`noise_sigma` 只在同模态内部可比**：Immerkær 类估计在可见光上会把纹理当噪声，不能跨模态比较。
4. **`130CT-02可见光_OSDT.mp4` 混有 19.0% 红外帧**（81.0% 可见光；按全片时长多数表决归入 VIS）；若只取可见光时段，VIS 红线会更保守（需重跑提取验证）。
5. **可见光两侧的归一化尺度并不统一**：该片裁剪后内容框 1280 宽、好组 VIS 1920 宽，抬升系数差 3 倍 ⇒ **可见光清晰度的表观 51.4× 里约 1.7× 来自这个尺度差**，折算回原生尺度约 29×。**接入前必须统一裁剪/缩放策略，否则可见光阈值不可迁移。** 该片偏置实测只取开头 120 帧、落在其红外时段，故 2.08× 是红外内容系数，按可见光内容系数折算差距会更大（≈45×）⇒ **29× 是保守下界**（见 4.3 第 4 条）。
6. **相关性而非因果**：给出的是「哪些画质指标能预测视频不可用」，不等于「该指标就是跟丢的原因」。
7. **全图无参考指标分不清「画质差」与「场景没内容」**（见第五节）。
8. **坏组源文件本身有损坏**（实测 mp4 box 与包解析，2026-11-20 复核）：`120BT-04-红外02.mp4`（237,060,096 B）是 fragmented mp4，**最后一个 mdat 声明 676,711 B、实际只有 671,506 B ⇒ 文件尾被截 5,205 字节**；损坏起点是最后一个包的 offset **237,055,843**（文件的 **99.998%**），该包 NAL 长度头声明 **4343 B**、到 EOF 只剩 **4249 B** ⇒ 报 `Invalid NAL unit size (4343 > 4249)`；ffmpeg 的 `partial file` 报错位置 `0xe21405e` = 237,060,190（EOF 之后 94 B）。**模态结论不受影响**：该片 duration = 1434.0 s，模态探测的 **287** 个采样点**覆盖全程、没有提前截断**，全判为红外（`modality_vis_ratio=0.0`、饱和度稳定在 [21,28]）；但**逐帧指标是在受损码流上算的**（`frames_decoded=35,848`，理论 35,850 帧，**尾部丢了 2 帧**），属**源数据质量问题、不是算法问题**，上线前应要求重新导出一份。另：**短片的模态采样点很少**（`VIDEO00011_19700101_012742IR` **2** 点、`VIDEO00013` **3** 点、`VIDEO00014` **5** 点、`VIDEO00020_...VIS` **11** 点、`VIDEO00018_...VIS` **18** 点），**这些片的模态占比统计置信度很低**，引用时必须带样本数（均已落盘在各逐片 json 的 `modality_samples` / `modality_coloured`）。

## 九、下一步（尚未做，需用户确认后再动）

- 用交付模型（`trial_0080` / RC2）在两组视频上实测 F1 / 跟踪丢失率，把画质指标与算法失效直接挂钩。
- 补 GT 框后重算局部 SCR 与归一化位移。
- 对越线视频做后处理增强（更激进阈值 / 多帧融合 / 重检测）能否救回，作为算法侧对照实验。
- **扩样**：多收几条坏视频（尤其可见光），重新标定红线阈值与分级规则。
- **统一分析尺度**：接入前固定裁剪与缩放策略，使阈值在生产环境可迁移。

---

**关联专题档**：`manu/memory/nogmc_focus_20261102.md`；**主档** `manu/memory/memory.md` §260；**精简版** `manu/memory/memory_compact.md` §260。
**耐久登记点**：主档/精简档的 §260 **可能被并行任务的整档重写覆盖**（本档已实测被覆盖两次）；
**权威入口是本专题档 + `manu/memory/.slim_memory.md` §7 的登记行**（`.slim_memory.md` 不被并行任务改写）。
§260 被覆盖后的**恢复命令**（备份块已随工程落盘，不依赖 `/tmp`）：

```bash
cd /tmp/pycharm_project_10ae9e2e
cat manu/memory/.vq_sec260_main.md    >> manu/memory/memory.md   # 主档 §260
cp  manu/memory/memory.md             memory.md                  # 根副本保持逐字节一致
cat manu/memory/.vq_sec260_compact.md >> manu/memory/memory_compact.md   # 精简档 §260
```

（`.vq_sec260_main.md` / `.vq_sec260_compact.md` 是本条 §260 的完整正文快照，**只追加、不覆盖**，重复执行会产生重复条目，恢复后请确认 `grep -c "§260" manu/memory/memory.md` 为 2。）

---

# 七、全图任意点选可跟踪性（空间可跟踪性）—— 2026-11-20

> **对外汇报件对应章节**：`manu/video_quality/report/REPORT_视频质量指标.md` **第八节**
> **主档登记**：§269 ｜ **新增脚本**：`manu/video_quality/vq_grid*.py` + `test_vq_grid_lk.py`
> **新增产物**：`manu/video_quality/grid_out/`（25 片 × json/npz/rows.csv，36 MB）
> ＋ `manu/video_quality/report/grid_{summary,windows,compare,crosstab,sensitivity}.csv`
> ＋ `report/grid_thresholds.json` ＋ `/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/chart_grid_*.png`（6 张）

## 7.1 为什么做这一节（业务动机）

真实交互是**操作员在画面任意位置点选目标**，而不是「画面里预设了某个特定目标」。
⇒ 评估核心从「能不能找到某个特定目标」变成 **Spatial Trackability**：
**随机点 N 个位置，有多大比例能被稳定锁住并跨帧跟住？**
这个设定**彻底摆脱了对检测框 / GT 的依赖**，可以全自动跑完整批视频。

## 7.2 方法（`vq_grid.py`）

- 帧准备与既有指标流水线**完全一致**：黑边（pillarbox）裁切 → 缩放到宽 640 → 灰度
  ⇒ 网格格覆盖场景比例在每条片里相同，可与 `comparison.csv` / `windows.csv` 逐秒对齐。
- **48 宫格模拟点击**：6 行 × 8 列，格 32×32 px，搜索半径 32 px，`gap = 5` 帧，每秒窗口 2 对。
- **单格判「可跟踪」需同时满足**：
  1. 格内标准差 ≥ 3（有纹理，否则提不到特征）；
  2. **前向** t→t+5：32×32 模板在 64×64 窗内 NCC ≥ 0.65；
  3. **回环** t+5→t：闭环误差 ‖点击点 − 回环点‖ ≤ 3 px。
- **两个匹配域**：`raw`（原始灰度）与 `dog`（DoG 带通，σ **1.5/4.0**，**另加峰值唯一性检验**
  以拒绝「自相似/马赛克伪影」这种匹配得上一致但锁错结构的假阳性）。
- **纹理地板 DET**：单格拉普拉斯方差 ≥ τ_det，**τ_det 只用同模态好片的 P10 标定**
  （IR **4.03** / VIS **82.04**）⇒ **不用坏片标定**，避免循环论证。

## 7.3 三个指标

| 指标 | 定义 | 业务含义 |
| --- | --- | --- |
| **TGR** | 48 格中通过双向闭环检验的比例 | 随便点，多大比例能被锁住 |
| **DET** | 48 格中有足够细节的比例 | 点下去能不能提取到特征 |
| **UGR** | 同时「有细节」**且**「能锁住」的格数占比 | **真正可用的操作位（主指标）** |

## 7.4 主结果（25 条片 / **26,490** 条逐秒评估）

分级沿用 §3.1 红线规则（**不看本节结果**，避免自证）：

| 分级 | 条数 | TGR(raw) | TGR(DoG) | DET | **UGR 中位** | UGR 区间 | 无纹理格(48) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 通过 | 15 | 0.917 | 0.896 | 0.969 | **0.833** | 0.521~0.979 | 2 |
| 通过（弱红线告警） | 5 | 0.583 | 0.563 | 0.552 | **0.250** | 0.208~0.875 | 1 |
| 人工复核 | 1 | 0.458 | 0.396 | 0.500 | **0.292** | — | 17 |
| 拒绝接入 | 4 | 0.281 | 0.219 | 0.281 | **0.063** | 0.000~0.333 | 12 |

- **AUC（好片为正类）**：IR `tgr_raw`/`tgr_dog` 视频级 **1.000**、窗口级 0.883/0.865；
  VIS `det_ratio`/`ugr` 视频级 **1.000**、窗口级 0.929/0.896；VIS `lambda_min_p50` 视频级 **1.000**。
- **红线窗口 vs 干净窗口**（核心红线全越线的窗口为正类）：`tgr_dog` AUC = **IR 0.008 / VIS 0.129**
  ⇒ 越线秒的可用率系统性落在最底部。
- **空间分布**（`/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/chart_grid_heatmap.png`）：好片**整幅均匀**（IR 每格 0.58~0.86、VIS 0.66~0.79）；
  坏片**整幅都低**（IR 0.21~0.29、VIS 0.07~0.23）⇒ 坏片是**系统性不可用**，不是「只是没拍到目标」。
- **阈值敏感性**：τ_det 扫 0.25×~8×（**32 倍**），好片 UGR 只从 0.875 降到 0.583，
  **坏片始终 ≤ 0.083**，分离度不塌。

## 7.5 🔴 一条被实测推翻的旧假设（否定性结论必须量中间量）

原 `vq_grid.py` 的 docstring 写着「raw 域对失焦是盲的，坏片 `120BT-04-可见光.mp4` 的 raw TGR ≈ 0.98」。
**实测推翻**：该片 raw TGR = **0.833**，DoG TGR = **0.771**（DoG 域**不是**决定性过滤器）。
真正把它拒掉的是**纹理地板**：48 格里只有 **29.2%** 有足够拉普拉斯能量 ⇒ **UGR = 0.125**。
⇒ **两种失效模式互补，缺一不可**：
- 「**锐利但锁不住**」（`130CT-02红外_IRT`：DET 0.917 / DoG TGR 0.354 / UGR 0.333）→ 由**闭环检验**抓住；
- 「**整幅平滑**」（`120BT-04-可见光`：DoG TGR 0.771 / DET 0.292）→ 由**纹理地板**抓住。
⇒ 对外交付的主指标必须是 **UGR**（两个条件在**同一格**同时成立），不是 TGR。
（docstring 已按实测数字改写，`vq_grid.py` 顶部。）

## 7.6 解掉 §5 的局限 7（好片拍到干净天空 → 全图清晰度偏低 ⇒ 误报）

`VIDEO00007_19700101_012038IR.h264`（好片、天空多）：48 格里 **17 格完全无纹理**，
但仍有 **29.2%** 的格位满足闭环可锁 ⇒ **正确归到「人工复核」而非「拒绝接入」**；
而 4 条拒绝接入片里 **2 条 UGR = 0.000**、另 2 条整幅每格仅 0.07~0.23。
⇒ **网格化统计可显著降低 3.1 边界情况的误报率**。

## 7.7 口径边界（必须一起说）

UGR 衡量的是**画面本身**有没有可供跟踪的局部结构，**不等于**「实际跟踪器在这条片上的成功率」
（跟踪器还有初始化、模型、搜索策略等自身边界）。本节回答的是**画质这一侧的天花板**：
**UGR ≈ 0 时是画质把路堵死了，任何跟踪器都救不回来。**

## 7.8 复现命令（执行机 `huangzhe@192.168.99.40:32222`，conda env `uav`）

```bash
cd /tmp/pycharm_project_10ae9e2e
#    产物根（服务器 canonical = /mnt/data/siping/runs/；本机跑时覆盖到 workspace/cache/ultralytics/）
VQ_PROD="${VQ_PROD:-/mnt/data/siping/runs/video_quality}"
# 1) 逐片提取网格可跟踪性（25 条片，4 worker，线程钉 1，60s 看门狗 + 20% 内存自刹车）
bash manu/video_quality/vq_grid_launch_server.sh 4
#    产物：manu/video_quality/grid_out/<片名>.{json,npz,rows.csv} + run_manifest.json
# 2) 与质量红线对照 + 出图
python manu/video_quality/vq_grid_compare.py \
  --grid-dir manu/video_quality/grid_out \
  --windows-csv manu/video_quality/report/windows.csv \
  --report-dir manu/video_quality/report
python manu/video_quality/vq_grid_charts.py \
  --grid-dir manu/video_quality/grid_out \
  --report-dir manu/video_quality/report
# 3) LK 光流对照实验（方法无关性；自检不需要视频）
python manu/video_quality/test_vq_grid_lk.py
bash manu/video_quality/vq_grid_lk_launch_server.sh 4
# 4) 🔴 图表出仓库（同 §六 第 9 步）：PNG 迁到产物根，仓库只留报告正文与结论级表
mkdir -p "${VQ_PROD}/report"
mv -f manu/video_quality/report/*.png "${VQ_PROD}/report/" 2>/dev/null || true
```

## 7.9 🔴 工程坑（踩过，勿重犯）

1. **`np.load` 读 `.npz` 是惰性的**：每访问一次 `archive[key]` 就**重新解压整个数组**。
   逐窗循环里读 6 个数组 ⇒ 4 MB 的数组被解压几千次 ⇒ 对比分析从 **45 s 变成 30~60 min**。
   ⇒ 已加 `vq_grid.LazyArrays`（解压一次、缓存复用），三个脚本统一用它。
2. **`cv2.calcOpticalFlowPyrLK` 在 `OPTFLOW_USE_INITIAL_FLOW` 下返回的是绝对坐标，不是增量**，
   且**会原地改写 `nextPts` 数组**。
   ⇒ LK 第一版把绝对坐标当增量用，实测出现 `lk_disp_p50 = 459.5 px`（= 网格中位 x 坐标）
   与 `lk_fb_err_p50 = 1378.9 px`（> 640 px 画幅宽度，物理上不可能）⇒ **25 条片全部 `trackable = 0`**。
   ⇒ 修法：`predicted = forward`（绝对）、回程初值用 `points`、`returned = backward`、
   `error = ‖backward − points‖`、`displacement = ‖forward − points‖`，**所有传入数组 `.copy()`**。
   ⇒ 自检（合成帧，已知位移 (+7,−4)）：`disp_p50 = 8.062 px`（= √65）、`fb_err_p50 = 0.000`、
   真实平移对 `trackable = 1.000`、**无关噪声对 `trackable = 0.000`**（负对照）。
3. **LK 的 `MAX_FLOW_FROM_GUESS = 30 px` 是拿全局相位相关位移当参照**；
   有明显变焦/旋转/视差时，真正可锁的格子也可能被误判为 diverged ⇒ 该常数是可调旋钮。

## 7.10 🔴 LK 光流对照实验（方法选择不是中性的）

同一套 6×8 网格、同一批 25 条片、同样的帧准备，把匹配域换成 **Lucas-Kanade**（前后向误差 ≤ 1 px）整批重跑，
产物 `manu/video_quality/grid_lk_out/`。

| 口径 | 模板匹配 TGR(DoG) | LK 光流 |
| --- | --- | --- |
| IR 源片 AUC（好 6 / 坏 2） | **1.000** | **1.000** |
| VIS 源片 AUC（好 12 / 坏 2） | **0.875** | **0.625** |
| 全部源片 AUC（好 18 / 坏 4） | **0.944** | **0.812** |
| 好片组中位（18 条源片） | 0.875 | 0.875 |
| 坏片组中位（4 条源片） | 0.219 | 0.260 |
| 22 条源片两法 Spearman | — | **0.897** |
| 25 条片逐条 \|Δ\| ≤ 0.05 | — | **17/25**（中位 \|Δ\| = 0.021） |

**两个结论**：
1. ✅ 「好片可跟、坏片不可跟」**不是模板匹配的实现 artifact**（IR 侧两法 AUC 都是 1.000，好片侧数字几乎重合）；
2. 🔴 **但「坏片不可跟」的严格程度依赖算法** —— VIS 侧 LK 只有 0.625。
   原因可解释：**LK 跟的是平滑明暗梯度，而失焦画面里这种梯度依然连续**
   （重焦芦苇特写 `120BT-04-可见光.mp4`：TGR(DoG) 0.771 vs **LK 0.917**）
   ⇒ **对外汇报必须写明用的哪个方法**；本报告主表用**模板匹配 + 峰值唯一性 + 纹理地板**
   （更保守、更贴近真实跟踪器），LK 作为**对照与边界说明**一并给出，不隐藏。

**自检（不依赖视频）**：`python manu/video_quality/test_vq_grid_lk.py`
合成帧已知位移 (+7, −4) px ⇒ 实测 `disp_p50 = 8.062 px`（= √65，精确）、`fb_err_p50 = 0.000 px`、
真实平移对可跟踪率 **1.000**、**无关噪声对（负对照）0.000**。

## 7.11 🔴 第四个坑：断点续跑会捡起上一轮的陈旧产物

上一轮 LK 被中断时残留的 JSON（**旧 schema、1130 px 回环误差**）被续跑逻辑当成「已完成」跳过
⇒ 结果里混进一份**旧数据**（`TS130CT-02挂飞.mp4`），若不核对 schema 会**静默污染整批结论**。
⇒ 已给 `vq_grid_lk_launch_server.sh` 的续跑逻辑加 **schema 完整性 + 物理合理性双重校验**
（键齐 + 回环误差与位移 < 画幅宽度），并单片重跑核对了该片；**全 25 条片复核 0 suspect**。

---

# 八、点选式单目标跟踪（SOT）实测：画质 → 「点得准跟得住」合格线 —— 2026-11-20

> **主档登记**：`manu/memory/memory.md` **§284**（已完成）；`memory_compact.md` §284、`.slim_memory.md` §7 登记行同批落档
> **汇报件落点**：`manu/video_quality/report/REPORT_视频质量指标.md` **第九节**（+ 附录 C 逐片明细），并在 **§一** 补上「画质合格线」最终答案、**§四 / §五** 同步更新（原先写的「因果需靠对照实验来证」已过时，现由第八、九节给出）
> **新增脚本**：`manu/video_quality/vq_sot_link.py`（探针）、`manu/video_quality/vq_sot_link_driver.py`（批跑驱动）、
> `manu/video_quality/vq_sot_compare.py`（聚合 + 合格线反解）、`manu/video_quality/vq_sot_verify.py`（独立复算验收）
> **新增产物**：服务器 `/tmp/sot_out/`（22 条源片）、`/tmp/sot_out_osd/`（3 条 OSD）、
> `/tmp/sot_report_long/`、`/tmp/sot_report_short/`；
> 本地 `manu/video_quality/sot_out/`（25 个 JSON，gitignore）
> ＋ `manu/video_quality/report/sot_{clips,groups,discrimination,thresholds,tau_det,windows}.csv`
> ＋ `report/sot_{thresholds,discrimination,groups,tau_det}_short.csv`
> **执行机**：`huangzhe@192.168.99.40:32222`（conda env `uav`，工作目录 `/tmp/pycharm_project_a7d966e6`）
> ｜**未用 GPU**｜**未碰板卡**｜**纯 CPU**
> **md5 双侧一致**：`vq_sot_link.py` `ebc6aa0980d4951765ce95237376803d` ｜
> `vq_sot_link_driver.py` `b1de859720c63b617bb01363a9eb183c` ｜
> `vq_sot_compare.py` `36b31298355d5139a2f614bb7e9d0c1d` ｜
> `vq_sot_verify.py` `3e57e2bfc2c8e23374ca84f2799dce74`

## 8.1 为什么推翻上一版「反无人机检测器」路线

**2026-11-20 用户纠偏：这批测试视频里根本没有无人机。**
上一版（`vq_track_link.py` / `vq_track_link_driver.py` / `vq_track_link_launch.sh`）
拿**反无人机检测器**去跑**无无人机**的视频：

| 上一版现象 | 真实解释 | 后果 |
| --- | --- | --- |
| 好片测出 `7.5%~9.0%` 候选帧、`10~15 帧`连续航迹 | 检测器把**楼角 / 路灯 / 树梢 / 高对比度地物**当成疑似弱小目标，地物纹理清晰 ⇒ GMC 补偿后被后端时空关联当成连续轨迹留下 | 测到的不是「目标可跟踪性」，而是**背景虚警（虚警）率** |
| 坏片测出 **0 航迹**、候选帧 `0.167%` | 画面发灰发糊把背景地物的锐利边缘全抹平 ⇒ 检测器**连背景虚警都触发不了** | 「坏片零航迹」**不能证明**「画质差 ⇒ 跟不住」的因果（糊图只是**抑制了虚警**） |

⇒ 只要领导反问一句「**这批测试视频里原本有无人机吗？**」，整版因果论证当场崩
⇒ **上一版数据全部作废、不得进正式报告**；批跑进程已 kill，产物 `/tmp/vq_track_out` 已删。

**真实业务场景**：前端不靠检测器自动起批，而是**操作员在画面任意位置点一下**，
由**单目标跟踪器（SOT）**锁住那个局部特征块去跟
⇒ 评估对象必须从「检测器能不能找到目标」换成「**点下去能不能锁住并跟住**」，与业务交互一一对应。

## 8.2 方法：48 宫格模拟点击 → 两族 SOT → 双时程

脚本：`manu/video_quality/vq_sot_link.py`（探针）＋ `manu/video_quality/vq_sot_link_driver.py`（批跑驱动）。

- **模拟点击**：48 宫格（**6×8**），**每 1 秒一个 epoch**，每个 epoch **旋转取 16 格**（避免总盯同一批格子）。
- **点击模式交替**（`--click-mode alternating`）：**偶 epoch 点格心** / **奇 epoch 点格内 Shi-Tomasi 最优角点**
  ⇒ 同时覆盖「粗点（任意位置）」与「细点（最好抓的特征点）」两种操作员手感。
- **两族跟踪核**（**旁路检测器**，全程不加载任何检测模型）：

  | 臂 | 实现 | 性质 |
  | --- | --- | --- |
  | **xcorr 臂** | `manu/tracking/lookahead/xcorr.py::match` —— 逐帧在**上一帧位置**重切模板再配准（亚像素 + 峰值唯一性诊断） | 通用模板核，**不要求目标外形** |
  | **patch 臂** | `manu/tracking/lookahead/polarity.py::PolarityTracker` —— 产品快核（LocalPatchTracker + 亮/暗自适应） | 产品同款快核，要求「紧凑亮/暗斑块」 |

- **两个时程**：`short` = **5 帧**（对齐 §271 的 `gap=5`，**按帧数不是秒**）／ `long` = **1 秒**。
- **判「跟住」门限**：**NCC ≥ 0.65 且双向闭环误差 ≤ 3 px**（与 §271 同一套门限）。
- **纹理地板 DET**：单格**拉普拉斯方差 ≥ τ_det** 才算「有纹理」；
  **τ_det 只用好片 P10 标定**（§271 同一纪律 ⇒ **不用坏片标定**，避免循环论证）
  ⇒ **IR 4.3279（n=6448 好片点击）/ VIS 97.6087（n=11056）**；§271 旧值为 IR 4.0285 / VIS 82.0397。
  （`sot_tau_det.csv` 的 `n_clicks` 列 = IR 3232 / VIS 5600，是**实际参与标定的格心点击子集**；
  上表的 n 是好片**全部**点击。）
- **主指标 UGR** =（**有纹理** ∧ **跟住** ∧ **闭环成立**）的点击占比 / **全部点击**。

**规模**：**22 条源片**（好片 18 + 坏源片 4）＋ **3 条桌面录屏参考组**；
**211,312 次模拟点击**；**13,207 个逐秒窗口**。
批跑用时：22 条源片 **567.8 s**（22 worker 并行，墙钟＝最长片）、3 条 OSD **753.3 s**。

**帧准备**：ffmpeg `rawvideo` 灰度管道**顺序解码**
（🔴 不用 OpenCV `seek`，见 8.8 坑 6）；帧准备与 §260 流水线一致
（pillarbox 裁切 → 缩放到宽 640 → 灰度）⇒ 数字可与 `windows.csv` 逐秒对齐。

## 8.3 主结果：好片 / 坏片 / 桌面录屏 三组

p50 口径（源：`manu/video_quality/report/sot_groups.csv` / `sot_groups_short.csv`）：

| 组 | 类型 | 点选 | 窗口数 | short_lock | short_UGR | long_lock | long_UGR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 好片 | 源视频 | 格心 | 552 | 0.875 | 0.8125 | 0.4375 | 0.375 |
| 好片 | 源视频 | 角点 | 542 | 0.875 | 0.875 | 0.4375 | 0.4375 |
| 坏片 | 源视频 | 格心 | 3179 | 0.1875 | **0.000** | 0.000 | **0.000** |
| 坏片 | 源视频 | 角点 | 3179 | 0.1875 | **0.000** | 0.000 | **0.000** |
| 桌面录屏 | 参考组 | 格心 | 2878 | 0.4375 | 0.1875 | 0.25 | 0.1875 |
| 桌面录屏 | 参考组 | 角点 | 2877 | 0.5 | 0.3125 | 0.3125 | 0.25 |

- **好片 vs 坏片 UGR 完全分离**：好片 short_UGR **0.8125 / 0.875**，坏片 short/long UGR **全部 0.000**。
- 🔴 **坏片 UGR 是被纹理地板压到 0.000 的** —— 坏片 `120BT-04-可见光` 虽然 blur，
  但 **NCC 照样匹配得上**（TGR 0.833 / DET 0.292）⇒ **不加纹理地板它会伪装成最好片**（见 8.8 坑 4）。
- **桌面录屏（OSD）夹在中间**（short_UGR 0.1875 / 0.3125）——
  它是**参考组不是源视频**（§7.4 已知：`TS130CT-02挂飞.mp4` 实为桌面录屏），**不得混进好/坏统计**。

### 🟢 patch 臂（产品快核）几乎全灭 —— 一个对业务很有用的发现

| 组 | 点选 | patch short_lock_mean | patch short_UGR p50 | patch long_UGR p50 |
| --- | --- | --- | --- | --- |
| 好片 | 格心 | 0.014 | 0.000 | 0.000 |
| 好片 | 角点 | 0.024 | 0.000 | 0.000 |
| 坏片 / 桌面录屏 | 全部 | ≤ 0.010 | 0.000 | 0.000 |

⇒ **patch 臂所有组 short/long UGR p50 一律 0.000**，好片 short_lock_mean 也只有 **0.014~0.024**。
**根因**：产品快核要求「**紧凑的亮/暗斑块**」，对**任意纹理**几乎锁不住。
⇒ **业务结论**：操作员「点选」必须**点中目标本体**，**点背景纹理锁不住**
（这也再次说明：**无目标视频测不了「跟踪无人机」本身**）。

## 8.4 判别力：哪些画质指标能预测「跟得住」

源片（好片为正类；源：`sot_discrimination.csv` / `sot_discrimination_short.csv`）：

| 指标 | AUC(指标分组) | Spearman(指标 vs 跟住率) | 好片中位 | 坏片中位 | AUC_score(long) | AUC_score(short) |
| --- | --- | --- | --- | --- | --- | --- |
| global_contrast | 0.919 | +0.541 | 0.6618 | 0.3070 | 0.841 | 0.899 |
| edge_density | 0.902 | +0.738 | 0.06668 | 0.002143 | 0.841 | 0.899 |
| lap_var | 0.896 | +0.737 | 997.4 | 31.69 | 0.841 | 0.899 |
| blockiness | 0.894 | −0.522 | 1.031 | 1.778 | 0.841 | 0.899 |
| entropy | 0.874 | +0.569 | 6.937 | 5.610 | 0.841 | 0.899 |
| dog_snr_max | 0.333 | −0.411 | 124 | 226.5 | 0.841 | 0.899 |
| snr_db | 0.134 | −0.713 | 43.31 | 56.9 | 0.841 | 0.899 |
| noise_sigma | 0.111 | +0.718 | 0.8118 | 0.1398 | 0.841 | 0.899 |

⇒ **与 §260 的「5 个有效指标」完全一致**：跨模态 = `lap_var` / `edge_density` / `global_contrast`，
IR 独有 = `blockiness` / `entropy`。
🔴 **`snr_db` / `noise_sigma` / `dog_snr_max` 对「跟得住」无区分度**
（AUC 0.111~0.333，方向还互相打架）⇒ 汇报时**不要再拿 SNR 类指标当画质合格线**。

## 8.5 画质合格线（按 IR / VIS 分列的准入线）

> 🔴 **本节测的是「任意点选位置的纹理可跟踪性」，不是「跟踪无人机本身的成功率」**：测试片里**没有无人机**，
> 且用的**不是**交付模型（`trial_0080` / RC2）。
> 「跟得住」的判定是**逐秒窗口二元化** —— 该秒 **UGR ≥ 0.5**（有纹理 ∧ 跟住 ∧ 双向闭环成立）⇒ 该秒记「跟得住」。
> 两个时程：`short` = **5 帧**、`long` = **1 秒**（=30 帧），**两个时程的线不能混用**。

脚本 `manu/video_quality/vq_sot_compare.py`（阈值搜索 + 方向按 `METRIC_DIRECTION` 统一）。

### 8.5.1 三条线的定义（`vq_sot_compare.py::admission_line`，**全部按 scope 分开推导**）

| 线 | 定义 | 用途 |
| --- | --- | --- |
| **① 准入线** | 同时满足「线上跟住率 ≥ 90%」**且**「坏片误放率 ≤ 1%」的**最低**那条 | 上线准入；**无解时必须明写「无解」，不许用别的线冒充** |
| **② 推荐线** | 在「坏片误放率 ≤ 1%」的候选里**线上跟住率最高**的那条 | 要「保证跟得住」时的宣传口径（用好片覆盖率换保证） |
| **③ 最低安全线** | 在「坏片误放率 ≤ 1%」的候选里**阈值最低**的那条 | 准入预筛（好片覆盖率最高） |
| **参考线** | gap 口径 `raw_threshold` | 历史上已落档、与 §271 互证的那一组，**只作参考** |

- **线上跟住率** = `good_TPR` = P(跟得住 ｜ 指标达标, **好片**)　—— 达标方向由 `METRIC_DIRECTION` 定：
  `blockiness` / `noise_sigma` 声明为**越低越好**（`≤ 线`），其余**越高越好**（`≥ 线`）。
- **坏片误放率** = `bad_FPR` = P(跟得住 ｜ 指标达标, **坏源片**)。
- **好片覆盖率** = 好片窗口中达标的比例。
- 🔴 **上一版的「混池覆盖率口径」已作废**（旧表里 `lap_var` / `edge_density` / `entropy` / `global_contrast` /
  `blockiness` 那五条**混池**线）—— 混池线实质是 VIS 专属线，理由与实测见 8.5.4 / 8.5.5。
- 本节主表口径 = **① 准入线**；🔴 混池（`scope=all`）的线**只是参考**，见 8.5.4。

### 8.5.2 短时程（5 帧）—— 准入线（**按 IR / VIS 分列**）

源文件：`manu/video_quality/report/sot_thresholds_ir_short.csv`（IR）、`manu/video_quality/report/sot_thresholds_vis_short.csv`（VIS）。

| 画质指标 | **IR 准入线** | IR 线上跟住率 | IR 坏片误放率 | IR 好片覆盖 | **VIS 准入线** | VIS 线上跟住率 | VIS 坏片误放率 | VIS 好片覆盖 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `global_contrast` | **≥ 0.7767** | 0.9816 | 0 | 0.4045 | **≥ 0.7034** | 0.9357 | 0.00641 | 0.2026 |
| `blockiness` | **≤ 1.145** | 0.9677 | 0 | 0.7692 | **无解** | — | — | — |
| `entropy` | **≥ 7.371** | 0.9802 | 0 | 0.5012 | **无解** | — | — | — |
| `lap_var` | **无解** | — | — | — | **≥ 2790** | 1 | 0 | 0.2402 |
| `edge_density` | **无解** | — | — | — | **无解** | — | — | — |
| `snr_db` | **无解** | — | — | — | **无解** | — | — | — |
| `noise_sigma` | **无解** | — | — | — | **无解** | — | — | — |
| `dog_snr_max` | **无解** | — | — | — | **无解** | — | — | — |

> **短时程（5 帧）有真线**：IR **3 条**（`global_contrast` / `blockiness` / `entropy`）、VIS **2 条**（`lap_var` / `global_contrast`）。
> 🔴 `snr_db` / `noise_sigma` / `dog_snr_max` 在 **IR / VIS 两侧都无准入线**（AUC 仅 0.111~0.333）⇒ **不得当合格线**。

**附：②推荐线 / ③最低安全线（短时程，备选口径，不能当准入线）**

| 画质指标 | IR ②推荐线 | IR ②跟住率 / 好片覆盖 | IR ③最低安全线 | IR ③跟住率 / 好片覆盖 | VIS ②推荐线 | VIS ②跟住率 / 好片覆盖 | VIS ③最低安全线 | VIS ③跟住率 / 好片覆盖 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `global_contrast` | ≥ 0.8026 | 1 / 0.2308 | ≥ 0.7767 | 0.9816 / 0.4045 | ≥ 0.7377 | 1 / 0.04052 | ≥ 0.7034 | 0.9357 / 0.2026 |
| `blockiness` | ≤ 1.038 | 1 / 0.33 | ≤ 1.145 | 0.9677 / 0.7692 | 无解 | — | 无解 | — |
| `entropy` | ≥ 7.407 | 0.9917 / 0.3002 | ≥ 7.371 | 0.9802 / 0.5012 | 无解 | — | 无解 | — |
| `lap_var` | 无解 | — | 无解 | — | ≥ 2790 | 1 / 0.2402 | ≥ 2790 | 1 / 0.2402 |
| `edge_density` | 无解 | — | 无解 | — | 无解 | — | 无解 | — |
| `snr_db` | ≥ 56.23 | 0.2769 / 0.1613 | ≥ 56.23 | 0.2769 / 0.1613 | 无解 | — | 无解 | — |
| `noise_sigma` | 无解 | — | 无解 | — | 无解 | — | 无解 | — |
| `dog_snr_max` | 无解 | — | 无解 | — | 无解 | — | 无解 | — |

> ②③ 两线的**坏片误放率按定义 ≤ 1%**；「无解」= 连「坏片误放率 ≤ 1%」的候选都没有。
> ②推荐线是「用好片覆盖率换保证」：短时程 IR `global_contrast` / `blockiness` 与 VIS `lap_var` / `global_contrast`
> 的线上跟住率都能压到 **1**，代价是好片覆盖掉到 0.04052~0.33。
> 🔴 `snr_db` / `noise_sigma` / `dog_snr_max` 三项**方向本身相反、AUC 仅 0.111~0.333**，列出来只为留痕、**不得使用**。

### 8.5.3 长时程（1 秒）—— 准入线（**按 IR / VIS 分列**）

源文件：`manu/video_quality/report/sot_thresholds_ir.csv`（IR）、`manu/video_quality/report/sot_thresholds_vis.csv`（VIS）。

| 画质指标 | **IR 准入线** | IR 线上跟住率 | IR 坏片误放率 | IR 好片覆盖 | **VIS 准入线** | VIS 线上跟住率 | VIS 坏片误放率 | VIS 好片覆盖 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `global_contrast` | **无解** | — | — | — | **无解** | — | — | — |
| `blockiness` | **≤ 1.032** | 0.9059 | 0 | 0.2109 | **无解** | — | — | — |
| `entropy` | **无解** | — | — | — | **无解** | — | — | — |
| `lap_var` | **无解** | — | — | — | **无解** | — | — | — |
| `edge_density` | **无解** | — | — | — | **无解** | — | — | — |
| `snr_db` | **无解** | — | — | — | **无解** | — | — | — |
| `noise_sigma` | **无解** | — | — | — | **无解** | — | — | — |
| `dog_snr_max` | **无解** | — | — | — | **无解** | — | — | — |

> **长时程（1 秒）几乎没有真线**：只有红外 `blockiness` 这一条成立，可见光**全部无解**。
> 这不是「没算出来」，是「**算出来就是没有**」（见 8.5.7）。

**附：②推荐线 / ③最低安全线（长时程，备选口径，不能当准入线）**

| 画质指标 | IR ②推荐线 | IR ②跟住率 / 好片覆盖 | IR ③最低安全线 | IR ③跟住率 / 好片覆盖 | VIS ②推荐线 | VIS ②跟住率 / 好片覆盖 | VIS ③最低安全线 | VIS ③跟住率 / 好片覆盖 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `global_contrast` | ≥ 0.7539 | 0.6364 / 0.6005 | ≥ 0.7539 | 0.6364 / 0.6005 | ≥ 0.706 | 0.344 / 0.1809 | ≥ 0.697 | 0.3375 / 0.2315 |
| `blockiness` | ≤ 1.024 | 1 / 0.09181 | ≤ 1.145 | 0.6581 / 0.7692 | 无解 | — | 无解 | — |
| `entropy` | ≥ 7.355 | 0.7297 / 0.5509 | ≥ 7.302 | 0.6767 / 0.6601 | 无解 | — | 无解 | — |
| `lap_var` | 无解 | — | 无解 | — | ≥ 2384 | 0.6586 / 0.4197 | ≥ 1386 | 0.5548 / 0.6599 |
| `edge_density` | 无解 | — | 无解 | — | ≥ 0.1976 | 0.8201 / 0.2012 | ≥ 0.1976 | 0.8201 / 0.2012 |
| `snr_db` | ≥ 48.08 | 0.5025 / 0.9876 | ≥ 48.08 | 0.5025 / 0.9876 | 无解 | — | 无解 | — |
| `noise_sigma` | ≤ 0.3848 | 0.5052 / 0.9578 | ≤ 0.3928 | 0.505 / 0.9876 | 无解 | — | 无解 | — |
| `dog_snr_max` | ≥ 220.3 | 0.5414 / 0.7196 | ≥ 149.5 | 0.5151 / 0.9876 | 无解 | — | 无解 | — |

> 🔴 长时程的 ②推荐线**没有一个能把线上跟住率压到 0.9 以上**（IR 最高只有 `blockiness` 的 1，好片覆盖同时掉到 0.09181）
> ⇒ 长时程**没有可用的单指标单线**；`snr_db` / `noise_sigma` / `dog_snr_max` 同样**只留痕、不得使用**。

### 8.5.4 混池（`scope=all`）线 —— **仅参考**

源文件：`manu/video_quality/report/sot_thresholds_short.csv`（短时程）、`manu/video_quality/report/sot_thresholds.csv`（长时程）。

| 画质指标 | 短时程混池准入线 | 短线上跟住率 | 短好片覆盖 | 长时程混池准入线 | 长线上跟住率 | 长好片覆盖 |
| --- | --- | --- | --- | --- | --- | --- |
| `global_contrast` | ≥ 0.7778 | 0.9805 | 0.1408 | **无解** | — | — |
| `blockiness` | **无解** | — | — | **无解** | — | — |
| `entropy` | ≥ 7.374 | 0.9848 | 0.1801 | **无解** | — | — |
| `lap_var` | ≥ 2793 | 1 | 0.1499 | **无解** | — | — |
| `edge_density` | **无解** | — | — | **无解** | — | — |
| `snr_db` | **无解** | — | — | **无解** | — | — |
| `noise_sigma` | **无解** | — | — | **无解** | — | — |
| `dog_snr_max` | **无解** | — | — | **无解** | — | — |

🔴 **混池线的局限（必须一起说）**：红外与可见光的指标**量纲差一个数量级**
（好片 `lap_var` 中位 **IR 100 vs VIS 2102**）⇒ 混池线实质上是 **VIS 专属线**，
拿它去卡红外，会把**好红外片整批判成不合格**（实测见 8.5.5）。
⇒ **混池线只能当参考，正式用必须按 IR / VIS 分列**（8.5.2 / 8.5.3）。

### 8.5.5 跨模态迁移实测（把一条线套到另一个模态，会误杀多少好片）

脚本 `manu/video_quality/vq_sot_transfer.py`；产物 `manu/video_quality/report/sot_admission_transfer.csv`。
口径：好片 = `group=good` 且 `kind=source` 的逐秒窗口；「误杀」= 好片窗口**不满足**该线。

短时程**混池（`scope=all`）线**拿去卡另一个模态 ⇒ 好片被误杀比例：

| 混池线 | 用在红外 | 用在可见光 |
| --- | --- | --- |
| `lap_var ≥ 2793` | **100.0%**（403/403 窗口全被误杀） | 76.3% |
| `global_contrast ≥ 0.7778` | 61.8% | **100.0%** |
| `entropy ≥ 7.374` | 51.1% | **100.0%** |

跨模态迁移（IR 线卡 VIS / VIS 线卡 IR）**普遍误杀 36%~100% 好片**；其中 `IR blockiness ≤ 1.145` 卡可见光
只误杀 **2.6%** ⇒ `blockiness` 是五个有效指标里**唯一能跨模态用**的。

**一句话结论**：**混池线不是「偏严」而是「用错模态」** —— `lap_var` / `global_contrast` / `entropy` 的混池线
会把好红外片**整批判不合格**（`lap_var` 误杀 **100%**）；**唯一能跨模态用的是 `blockiness`**。

复现（执行机 `huangzhe@192.168.99.40:32222`，conda env `uav`，工作目录 `/tmp/pycharm_project_a7d966e6`）：

```bash
cd /tmp/pycharm_project_a7d966e6 && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
python manu/video_quality/vq_sot_transfer.py --windows manu/video_quality/report/sot_windows.csv \
  --thresholds-dir manu/video_quality/report --out manu/video_quality/report/sot_admission_transfer.csv
```

### 8.5.6 gap 口径参考线（**只作参考**，已落档、与 §271 互证）

| 画质指标 | 短时程（5 帧）参考线 | 短线上 / 线下（均值） | 短 gap | 长时程（1 秒）参考线 | 长线上 / 线下（均值） | 长 gap |
| --- | --- | --- | --- | --- | --- | --- |
| `edge_density` | ≥ 0.04763 | 0.764 / 0.127 | **0.637** | ≥ 0.19466 | 0.641 / 0.139 | **0.502** |
| `entropy` | ≥ 7.37783 | 0.899 / 0.265 | 0.634 | ≥ 7.37248 | 0.592 / 0.138 | 0.454 |
| `lap_var` | ≥ 2421.08 | 0.867 / 0.257 | 0.610 | ≥ 2423.88 | 0.556 / 0.134 | 0.422 |
| `global_contrast` | ≥ 0.77465 | 0.835 / 0.266 | 0.570 | ≥ 0.77465 | 0.512 / 0.140 | 0.372 |
| `blockiness` | ≤ 1.04981 | 0.714 / 0.182 | 0.532 | ≤ 0.87452 | 0.659 / 0.147 | **0.512** |
| `snr_db` | **无区分度** | — | — | **无区分度** | — | — |
| `noise_sigma` | **无区分度** | — | — | **无区分度** | — | — |
| `dog_snr_max` | **无区分度** | — | — | **无区分度** | — | — |

> gap 口径取「线以上均值 − 线以下均值」最大的那条线（**最强分界线**），
> **它不是准入线**：既不保证「线上跟住率 ≥ 90%」，也不保证「坏片误放率 ≤ 1%」
> ⇒ **只作参考**（历史上已落档、与 §271 互证的那一组），**不许拿来当准入线**。

### 8.5.7 准入线存在性统计：**48 个组合里只有 9 条真线**

- **总数**：48 个（时程 × scope × 指标）组合中，**只有 9 个真的同时满足「线上跟住率 ≥ 90%」且「坏片误放率 ≤ 1%」**。
- **短时程（5 帧）**：IR **3** 条 + VIS **2** 条 + 混池 **3** 条 = **8 条 —— 有真线**。
- **长时程（1 秒）**：IR **1** 条（`blockiness`）+ VIS **0** 条 + 混池 **0** 条 = **1 条 —— 几乎没有真线**。
- **原因**：链式跟踪**每帧重切模板、误差累积** ⇒ 好片**逐秒 UGR 方差很大**；
  把线压到「线上跟住率 ≥ 90%」只能落在好片 **P90 附近** ⇒ 好片覆盖率必然只剩 **10~40%**。
- 🔴 **这是真实结论，不是脚本 bug**：无准入线的组合一律明写「**无解**」，**FALLBACK 不许藏**
  （绝不用 ②推荐线 / ③最低安全线 / gap 参考线冒充准入线）。

### 8.5.8 一句话给领导

**用短时程（5 帧）口径** —— 红外卡 **`global_contrast ≥ 0.7767`** / **`blockiness ≤ 1.145`** / **`entropy ≥ 7.371`**；
可见光卡 **`lap_var ≥ 2790`** / **`global_contrast ≥ 0.7034`**；
🔴 `snr_db` / `noise_sigma` / `dog_snr_max` **一律不能当合格线**（无准入线、AUC 仅 0.111~0.333）。
**长时程（1 秒）没有可用的单指标单线**（仅红外 `blockiness` 一条勉强成立，可见光全部**无解**）⇒ **以短时程为准**。

### 8.5.9 复现（本小节全部数字）

执行机 `huangzhe@192.168.99.40:32222`，conda env `uav`，工作目录 `/tmp/pycharm_project_a7d966e6`（纯 CPU，**未用 GPU**）。

```bash
ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_a7d966e6 && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
# 1) 聚合 + 合格线反解（长 / 短两个时程各跑一次）
python manu/video_quality/vq_sot_compare.py --sot-dir /tmp/sot_out,/tmp/sot_out_osd \
  --windows manu/video_quality/report/windows.csv --grid-summary manu/video_quality/report/grid_summary.csv \
  --out-dir /tmp/sot_report_long --score xcorr_long_ugr
python manu/video_quality/vq_sot_compare.py --sot-dir /tmp/sot_out,/tmp/sot_out_osd \
  --windows manu/video_quality/report/windows.csv --grid-summary manu/video_quality/report/grid_summary.csv \
  --out-dir /tmp/sot_report_short --score xcorr_short_ugr
# 2) 跨模态迁移检验（见 8.5.5）
python manu/video_quality/vq_sot_transfer.py \
  --windows manu/video_quality/report/sot_windows.csv \
  --thresholds-dir manu/video_quality/report \
  --out manu/video_quality/report/sot_admission_transfer.csv
```

产物（落在 `manu/video_quality/report/`）：
`manu/video_quality/report/sot_thresholds{,_ir,_vis}.csv`（长时程）、
`manu/video_quality/report/sot_thresholds{,_ir,_vis}_short.csv`（短时程）；
跨模态迁移实测见 8.5.5（`manu/video_quality/report/sot_admission_transfer.csv`）。

脚本 md5（2026-11-20 服务器侧实测；`vq_sot_compare.py` 本轮新增 `admission_line` 三条线输出、
修复 `KeyError: 'good_tpr'`，故与 §七 记录的旧 md5 不同）：
`vq_sot_link.py` `ebc6aa0980d4951765ce95237376803d` ｜
`vq_sot_link_driver.py` `b1de859720c63b617bb01363a9eb183c` ｜
`vq_sot_compare.py` `11053de3050a8964b650664082306e44` ｜
`vq_sot_verify.py` `3e57e2bfc2c8e23374ca84f2799dce74` ｜
`vq_sot_transfer.py` `39ae769d525b55d96aa830458ebf03a4`。

## 8.6 与 §271 的交叉互证

`manu/video_quality/vq_sot_verify.py` 的 **CHECK 1** 把**逐片 SOT short_lock**（链式跟踪实测）
与 **§271 grid `tgr_raw`**（暴力 NCC 搜索）逐片对照（**25 条片全部列出**）：

- 平均 **|gap| = 0.082**，**worst = 0.208**（容差 0.12 ⇒ OK）。

⇒ **两套完全独立的实现**（§271 = 逐格暴力 NCC 搜索；本节 = 链式 SOT 跟踪）**互为佐证**：
「好片点得准跟得住、坏片点不准跟不住」**不是某个匹配器的实现 artifact**。

## 8.7 局限（必须一起汇报）

1. 🔴 **无目标视频只能测「纹理可跟踪性」，测不了「跟踪无人机本身」** ——
   本节素材里**没有无人机**，因此「点得准跟得住」指的是**画面局部纹理能不能被 SOT 锁住**，
   **不等于**「操作员点中无人机后跟得住」。结论的适用范围必须写清。
2. 🔴 **patch 快核点不中目标就锁不住** —— 产品快核要求「紧凑亮/暗斑块」，
   对任意纹理 short/long UGR 一律 **0.000** ⇒ 业务上「点选」必须点中目标本体，不能点背景纹理。
3. 🔴 **OSD 是桌面录屏参考组，不是源视频** —— 3 条桌面录屏只作对照，不得混进「好片 / 坏片」统计。
4. **UGR 是画质侧的**天花板：它衡量画面本身有没有可供跟踪的局部结构，
   **不等于**「实际跟踪器在这条片上的成功率」（跟踪器还有初始化、模型、搜索策略等自身边界）
   —— 与 §7.7 同一条纪律。
5. **本节全部指标为离线批跑结果**，未接入线上实时链路做在线分位数核对。

## 8.8 🔴 本轮踩过的 8 个坑（勿重犯）

| # | 症状 | 根因 | 修法 |
| --- | --- | --- | --- |
| 1 | 好片「有连续航迹」、坏片「零航迹」，看起来完美支持「画质差 ⇒ 跟不住」 | **检测器路线逻辑错误**：视频里**根本没有无人机** ⇒ 好片「航迹」= **背景虚警**、坏片「零航迹」= 糊图**抑制**了虚警 | 用户纠偏 ⇒ **整版数据作废**、路线整体推翻，改走「模拟点击 → SOT」 |
| 2 | 闭环误差动辄上百 px，好片也被判失锁 | 闭环误差用了 `\|回配点 − 点击点\|`（被**相机运动**主导） | 改用**往返误差** `\|前向位移 + 回配位移\|`（对相机运动不敏感） |
| 3 | 明明 NCC 还挺高，却大面积判失锁 | 失锁判定取了**精确中心**的 NCC（**1 px 抖动就误判**） | 改为在跟踪位置 **±8 px 内取最佳值**（`ncc_best`） |
| 4 | 坏片 `120BT-04-可见光`（blur）UGR 竟然不低，甚至**伪装成最好片**（TGR 0.833 / DET 0.292） | **缺纹理地板 DET**：糊图上 NCC 照样匹配得上 | 加**纹理地板**（单格拉普拉斯方差 ≥ τ_det），τ_det **只用好片 P10** 标定 |
| 5 | 短时程口径与 §271 对不上 | 短时程按**秒**换算（25 fps ⇒ 变成 4 帧） | 短时程**按帧数 5**（对齐 §271 的 `gap=5`） |
| 6 | 读到的帧全是第 0 帧 | **OpenCV `CAP_PROP_POS_FRAMES` 在裸 `.h264` 上根本不移动**（POS_FRAMES 停在 0） | 顺序解码改走 **ffmpeg `rawvideo` 灰度管道**，**不用 OpenCV seek** |
| 7 | 聚合结果 **UGR 全 0** | **键名两套**：点击级 `horizons` 键是 `"5"/"30"`，窗口级聚合才是 `"h5"/"h30"` ⇒ 聚合脚本查 `"h5"` 全查空 | 统一键名映射（冒烟时才发现 ⇒ **每次改 schema 必须先冒烟**） |
| 8 | 验收脚本报 **PASS** 但其实什么都没验 | `--sot-dir a,b` **未被拆开** ⇒ clips 为空 ⇒ 各项检查空转仍报 PASS | 加**硬保护**（无输入直接 `rc=2` 拒绝出 PASS）；另外**覆盖率合格线必须用中位数不用均值**（坏片样本会把均值拖到目标以下 ⇒「明明该有交叉却报无交叉」） |

## 8.9 复现命令（可直接复制）

执行机 `huangzhe@192.168.99.40:32222`，conda env `uav`，工作目录 `/tmp/pycharm_project_a7d966e6`。

```bash
ssh -F /dev/null -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 32222 huangzhe@192.168.99.40
cd /tmp/pycharm_project_a7d966e6 && source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
# 1) 逐片模拟点击 + 链式 SOT 跟踪（作业表 = TSV：路径 分组 kind 模态）
python manu/video_quality/vq_sot_link_driver.py --jobs /tmp/sot_jobs.tsv --out-dir /tmp/sot_out --workers 22 --threads 4 --click-mode alternating
python manu/video_quality/vq_sot_link_driver.py --jobs /tmp/sot_jobs_osd.tsv --out-dir /tmp/sot_out_osd --workers 3 --threads 4 --click-mode alternating
# 2) 聚合 + 合格线反解（长/短两个口径各跑一次）
python manu/video_quality/vq_sot_compare.py --sot-dir /tmp/sot_out,/tmp/sot_out_osd \
  --windows manu/video_quality/report/windows.csv --grid-summary manu/video_quality/report/grid_summary.csv \
  --out-dir /tmp/sot_report_long --score xcorr_long_ugr
python manu/video_quality/vq_sot_compare.py --sot-dir /tmp/sot_out,/tmp/sot_out_osd \
  --windows manu/video_quality/report/windows.csv --grid-summary manu/video_quality/report/grid_summary.csv \
  --out-dir /tmp/sot_report_short --score xcorr_short_ugr
# 3) 独立复算验收（不 import compare；空输入直接 rc=2）
python manu/video_quality/vq_sot_verify.py --sot-dir /tmp/sot_out,/tmp/sot_out_osd \
  --windows manu/video_quality/report/windows.csv --grid-summary manu/video_quality/report/grid_summary.csv \
  --thresholds /tmp/sot_report_long/sot_thresholds.csv
```

**独立复算验收结果**：**PASS**（CHECK 0 τ_det 复现 / CHECK 1 逐片对照 §271 / CHECK 2 窗口对齐 /
CHECK 4 分组独立重算 / CHECK 5 合格线复现 `thr=0.7746 == 0.7746`）。

---

本节全部产物与脚本**未提交（按铁律〇之三之二）**，仅落工作区。

---

## 九、物理崩溃拐点（Knee Point）标定 —— 第二子目标（2026-11-20 追加，只追加不改历史）

> 📌 **关系**：本节是**叠加**在 §一~§八 之上的**第二子目标**，**不替换**任何既有结论。
> §八的网格版、§九 的点选式 SOT 实测、**按 IR / VIS 分列的准入线**一律保留。
> 本节做的是**给合格线补一条不依赖坏片的物理依据**，再用**未参与标定的真实坏片**盲测，把**更稳的阈值**并回 §9.5 / §一。

### 9.1 方法（五步）

1. **抽母本**：18 条好源片各抽 **12 秒**（用 `report/windows.csv` 挑「质量最接近该片中位数」的窗口起点，选母本**不需要解码**），母本统一 **CRF 18**。
2. **三组退化**（每条母本**独立**施加，贴合真实病因）：高斯模糊 σ **0.5~4.0**（8 档，打 `lap_var`/`edge_density`）／向灰 128 收拢 **0.80~0.15**（6 档，打 `global_contrast`/`entropy`）／H.264 **CRF 23~44**（7 档，打 `blockiness`）⇒ **396 个变体**（18 片 × 22 档，含母本）。
3. **双路同步测量**：**X**＝退化后实测画质（**复用** `vq_metrics.analyze_video` + `vq_stats.window_values` 原口径：裁黑边 → 640px 归一 → 逐 1 秒窗口 P05/P95）；**Y**＝SOT 锁定率（**复用** `vq_sot_link_driver.py` 探针）；两路按**同一变体、同一批逐 1 秒窗口**对齐。
4. **分模态提取膝点**：IR 5 项 / VIS 3 项；先**保序回归（isotonic）单调化**去噪 → **PCHIP** 平滑 → ①**Kneedle**（X/Y 归一到 [0,1]，取**到首尾连线距离最大**的点）②**PCHIP 二阶导绝对值最大**处，两法交叉验证；另记**第三锚点**：拟合曲线上锁定率**降到 0.5** 的 X；两法不一致取**更靠近好片一侧**者（宁可多留好片）。
5. **盲测**：**4 条真实坏片**（IR 2 + VIS 2，**未参与标定**）+ **18 条好片**，逐 1 秒窗口 + 逐条片两个粒度，新旧线**同一批数据、同一口径**。

### 9.2 🔴 口径偏离（必须写明）

目标书写「所有退化都在 CRF 18 母本这一层之上施加」；**本实现采用单次有损编码**（每个变体**直接从原始源**解码 → 可选滤镜 → **只编码一次**）。
理由：若改成「母本(CRF18) → 滤镜 → 再编码」，每个变体**多一代压缩**，CRF 梯度就混入「二次压缩」这一额外变量，**膝点的物理含义被污染**；
单次编码下**同一梯度族内只有一个自变量**，曲线才可解释。

### 9.3 🔴 本轮修掉的 4 个真 bug（都是「读数看着像那么回事、其实取到曲线端点」的类型）

1. **Kneedle 符号反了**：原实现 `argmax(xn - yn)`，在饱和 S 曲线上会取到**曲线端点**；正确的是 `argmax(|yn - xn|)`（标准 `kneed` 语义：到首尾连线距离最大）。
2. **「越小越好」的指标曲线是递减的**（`blockiness`）：Kneedle 与 PCHIP 都假设 X/Y 同向递增 ⇒ 直接跑会取到端点；**必须先把 X 取负**统一成「越大越好」。
3. **取负时必须同步反转 Y**，否则 X/Y 配对被打乱（修前 `blockiness` 膝点算出 1.672（最差端），修后 1.047）。
4. **片级判定曾用「失败标志」算过线率 ⇒ 方向反了**：正确是**用「过线标志」，过线率 < 50% ⇒ 该片判不合格**。

### 9.4 🔴 其余已踩坑（勿重犯）

- **裸 `.h264` 输入侧 `-ss` 快进无效**（只出 262 字节），必须用**输出侧 `-ss`**；
- **ffmpeg 4.2 一路输入多路输出时，输出侧 `-t` 会把后面的输出越截越短**（22 路时只有第 1 路拿满 12 s）⇒ **必须用 `-frames:v N`**；
- **`windows.csv` 的 `file` 列带扩展名**，母本匹配必须用 **basename**；
- **τ_det 纹理地板必须用好片点击标定**（`--tau-dir /tmp/sot_out`），直接用退化片标会得到空集 ⇒ **UGR 全 0**；
- **SOT `summary["video"]` 带扩展名**，与 `variant_id`（stem）不一致 ⇒ 需 `Path(video_path).name` 映射；
- **合并 click_mode 时第一行不能丢**（alternating 模式逐窗 center/corner 交替，每窗口只有一行 click_mode）。

### 9.5 🎯 核心结论（盲测：4 条未参与标定的真实坏片 + 18 条好片）

- 🔴 **旧口径的「准入线 / 分位线」是多指标 AND，每个阈值又都取在好片分位上 ⇒ 联合覆盖率被乘性压垮**：
  片级把 **18 条好片里的 16~18 条**判成不合格（短·IR 准入线组合 **4/6** + 短·VIS **12/12**；§9.5.8 分位线组合 **18/18**）。
  **这在旧口径里是「好片覆盖 10.2%」的另一个说法，此前没被单独点出来。**
- 🟢 **新膝点线（短时程 5 帧）**：
  - **红外 IR**：`global_contrast ≥ 0.5779` **AND** `entropy ≥ 7.1147` ⇒ 坏片拦截 **2/2**、好片误杀 **1/6**、**过线好片跟住率 96.86%**；
  - **可见光 VIS**：`lap_var ≥ 306.47` ⇒ 坏片拦截 **2/2**、好片误杀 **0/12**、**过线好片跟住率 91.99%**。
- 🎯 **最刺眼的一例**：旧 VIS `lap_var ≥ 2790` 会误杀 **10/12** 条好片；换成物理膝点 `lap_var ≥ 306.47` 后 **好片 0/12 被误杀、坏片 2/2 全拦**。
- 🟢 **物理交叉验证**：红外 `blockiness` 的三个锚点 Kneedle **1.188** / PCHIP **1.047** / 锁定率 0.5 交点 **1.141** **把旧准入线 1.145 夹在中间** ⇒ **旧准入线有独立物理依据**。
- 🔴 **长时程不采用**（过线好片跟住率只有 40%~63%，明显劣于短时程）⇒ 沿用 §9.5.8 的「上线以短时程（5 帧）为准」，**不改**。
- 📌 **两套线并存、用途分开**：旧线＝**充分条件式**（过线基本必能跟，但几乎不可用）；膝点线＝**必要条件式**（低于线基本跟不住，**过线不等于一定跟得住**）。
  ⇒ **上线逐秒告警 / 拒接用膝点线（可用性优先）**；**要求必然跟得住的强保证场景用旧准入线**。

### 9.6 ⚠️ 诚实口径（铁律不变）

- 本节测的是「**任意点选位置的纹理可跟踪性**」，**不是**「跟踪无人机本身的成功率」（测试片**没有无人机**、用的**不是**交付模型 `trial_0080` / RC2）；
- **IR 灰度域与 VIS 彩色域量纲不同，阈值必须分模态**；混池线只作参考、**跨模态禁用**；
- **gap 线只是「最优分界线」，不是准入线**；
- **坏源片只有 4 条 ⇒「坏片拦截 100%」属小样本结论，上线前必须扩样重标**；
- **OSD 桌面录屏是参考组**，不与源片混算。

### 9.7 产物 / 脚本 / 验收

- **产物**（`manu/video_quality/report/`）：`degrade_windows.csv`（4730 行）、`degrade_lock.csv`（3938 行）、`degrade_curve.csv`（992 行）、
  `knee_thresholds.csv`（16 行）、`knee_blindtest.csv`（127 条线）、`knee_combos.csv`（134 条线级组合）、
  `/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/chart_degrade_knee_{ir,vis}_{metric}.png`（8 张）；中间产物在服务器 `/tmp/vq_degrade/`（**396 个退化视频**，体积大，**不回传本地**）。
- **脚本**：`manu/video_quality/vq_degrade_prepare.py`（md5 `619dd401f07630e51cf7ef2bb504f6be`）、
  `vq_degrade_knee.py`（md5 `ff836a1c1b9bcde70e39390b20e0f4eb`，六相 measure/curve/knee/blindtest/combo/charts）、
  `vq_degrade_report.py`（md5 `bcba6322522e91ca06161533285e20e6`，生成 §9.5.9 小节，数字全部从 CSV 现场读）。
- **执行机**：`huangzhe@192.168.99.40:32222`，conda env `uav`，工作目录 `/tmp/pycharm_project_a7d966e6`；
  **可复制命令见汇报件 §9.9 的「5-1 ~ 5-8」**。
- **验收**：`vq_sot_verify.py` → **[PASS] all checks passed**；`vq_sot_admission_verify.py` 长/短 → **[RESULT] PASS**；
  汇报件 **38 张 markdown 表列数自检全一致**。

### 9.8 🔴 脚本 md5 更正（历史记录保留，勿改历史）

- **`vq_sot_compare.py` 由 `11053de3050a8964b650664082306e44` 更新为 `560e41a39a5c3dcc03546894d0335870`**：
  根因修复 = `calibrate_tau` **不筛 `click_mode`**，导致 τ_det 元数据写错（原先误写 "(center clicks)" 与 3232/5600，现为 **6448/11056**）；
  已重跑长/短两个时程，`sot_tau_det.csv` 的 md5 = `ad9b39e4f16cb5f662c61d4a7f917041`。
- **`vq_sot_transfer.py` 已删除**：与 `vq_sot_admission_verify.py` 的 **CHECK E** 功能重复，且两者写**同名产物** `sot_admission_transfer.csv` 会互相覆盖
  ⇒ **跨模态迁移体检由 `vq_sot_admission_verify.py` CHECK E 唯一承担**。

---

## §287.1 补充 · 膝点可靠性（`knee_reliable`）修正 —— IR 采用线改用「两个可靠膝点」（2026-11-23）

🔴 **背景（不改历史，只更正）**：§287 记录的 IR 采用线是 `global_contrast ≥ 0.5779` AND `entropy ≥ 7.1147`。
本轮收尾审计发现 **`knee_thresholds.csv` 里的 `knee_reliable` 标志从未被写进报告**，而 §287 自己定的可靠性规则是
（`vq_degrade_knee.py`，`KNEE_AGREE_PCT = 15`）：**Kneedle 与 PCHIP 两个独立算法给出的膝点之差 ≤ 15% 量程 ⇒
`knee_reliable=1` 可作准入线；> 15% 量程 ⇒ 曲线噪声过大，膝点只作「物理参照」、不作准入线**。
红外 `entropy` 的两法差是 **25.4% 量程**（Kneedle **6.332** / PCHIP **7.115**）⇒ `knee_reliable=0`
⇒ **按本任务自己的规则，它本不该进采用线**。这是 §287 的口径缺口，不是 §287 的方法被推翻。

✅ **修正后的采用线（短时程 5 帧）**：

| 模态 | 采用线 | 组件可靠性 | 坏片拦截(片) | 好片误杀(片) | 过线好片跟住率 | 坏片拦截(窗) | 好片误杀(窗) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **IR** | `global_contrast ≥ 0.5779` **AND** `lap_var ≥ 6.493` | 两个组件均 `knee_reliable=1`（两法差 0.0% / 0.0%） | **100%（2/2）** | **16.67%（1/6）** | **95.97%** | 88.79% | 26.05% |
| **VIS** | `lap_var ≥ 306.47` | `knee_reliable=1`（两法差 7.1%） | **100%（2/2）** | **0%（0/12）** | **91.99%** | 92.89% | 0.58% |

- **片级表现与 §287 原 IR 线完全相同**（坏片 2/2 全拦、好片误杀 1/6）；窗级：坏片拦截 **88.79%**（原 97.70%、
  因为去掉了一个被过线判据放行的松阈值）、好片误杀 **26.05%**（原 28.78%）、过线好片跟住率 **95.97%**（原 96.86%）。
- 🔶 **原 IR 线降级为「备选加强线」（报告表 D-2，默认不启用）**：`global_contrast ≥ 0.5779 AND entropy ≥ 7.1147`
  —— **窗级**坏片拦截更高（**97.70%** vs 88.79%）、**片级**与采用线完全相同，代价是含一个
  **只作物理参照**的膝点（`entropy` 两法差 25.4%）⇒ 业务上若「宁可多拦坏片窗口」可启用，**默认不启用**。
- 🔴 **其余结论一条没动**：9.5.1~9.5.8 的 IR/VIS 分列准入线、混池线「只作参考、跨模态禁用」、gap 线「只作分界线
  不是准入线」、长时程不采用（跟住率 40~63%）、**两套线并存**（旧线＝充分条件式／膝点线＝必要条件式）、
  诚实口径四条（任意点选纹理可跟踪性 ≠ 跟踪无人机成功率；IR/VIS 必须分模态；gap 线只作分界线；坏片仅 4 条属小样本）
  **全部保留**；仅新增诚实口径**第 8 条**：`knee_reliable` 是选线的**一票否决项**。

📌 **本轮改动清单**：
- 脚本 `manu/video_quality/vq_degrade_report.py`：表 A 增「膝点可靠性」列 + 可靠性判据说明；`ADOPTED` 改为
  `("short","ir"): ["global_contrast","lap_var"]`；新增 `ALT_LINES`（备选加强线）⇒ 报告新增**表 D-2** 与
  「选线过程留痕」段；诚实口径新增第 8 条。
- 报告 `manu/video_quality/report/REPORT_视频质量指标.md`：**§一** 与 **§9.5.9** 同步（§9.5.9 整节由脚本生成）。
- 复现命令（可复制）：
  `cd /run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics && \
   python3 manu/video_quality/vq_degrade_report.py --variants manu/video_quality/report/degrade_variants.csv \
   --out /tmp/section_959.md --apply`
  （**幂等**：连跑两次报告 md5 均为 `542729c1c4ce16808077f71bfde9a8d0`）。
- 执行机：**本地主机** `/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/ultralytics`；
  服务器 `192.168.99.40:32222` 的 `/tmp/vq_degrade/variants.csv` 与本地 `report/degrade_variants.csv`
  **md5 一致**（`0be9515c45f37b37bf8496bbe716e599`）⇒ 本地重跑与服务器口径等价。
- 产物：`manu/video_quality/report/{REPORT_视频质量指标.md,knee_thresholds.csv,knee_combos.csv,knee_manifest.md5}`。
- 验收：`md5sum -c manu/video_quality/report/knee_manifest.md5` → **42/42 OK**；
  `vq_sot_verify.py` 本地 **[PASS]**；`vq_sot_admission_verify.py` 短时程本地 **[RESULT] PASS**。
- 📌 **补记（2026-11-25 晚；主档 `memory.md` §310 七）**：清单随后扩到 **90 条**（补入 KCF 派生表与 PDF），
  其中 4 条记忆文件是**「活基线」**——记忆库改动后 `md5sum -c` 一度只过 **86/90**，已刷新并在清单头写明刷新时机
  ⇒ 现 **90/90 OK**；仓库内 PNG **0 张**、报告图链 **37/37** 命中。
- 🔴 **未提交（按铁律〇之三之二）**，仅落工作区。

## 十、KCF 版重做 —— 换 OpenCV contrib KCF 跟踪器跑同一套「画质 → 跟踪」因果链（2026-11-24）

> 📌 本节只追加、不改 §八/§九 的任何历史结论；主档见 `manu/memory/memory.md` **§308**，
> 报告全文见 `manu/video_quality/report/REPORT_视频质量指标.md` **§9.10.1~§9.10.11**。

### 10.1 为什么是 OpenCV contrib 的 KCF（而不是自己写一份）

- 🔴 **OpenCV 主仓库没有 KCF**：`opencv/modules/video/src/tracking/` 只有 MIL/MedianFlow/GOTURN/Nano/Vit，
  `grep -ril kcf` 零命中 ⇒ 官方 `opencv-python` 里 **KCF 不存在**。
- KCF 实现在 **opencv_contrib 的 `modules/tracking`** ⇒ Python 入口 **`cv2.legacy.TrackerKCF_create`**。
- 服务器 `opencv-python 4.10.0.84` 无 `cv2.legacy` ⇒ 装 **`opencv-contrib-python==4.10.0.84`** 到
  **隔离目录** `/tmp/pylibs_cvcontrib`（`pip install --no-deps --target`），跑时 `PYTHONPATH` 注入，
  **不污染共享 conda env `uav`**；📌 **产品若要上 KCF，依赖必须一起带**。
- contrib `TrackerKCF` 实为 **FSR/KCF 变体**（高斯核 + 傅里叶域滤波 + 在线模型更新；HOG / CN-PCA 特征；
  **半分辨率** `resize=1`；内部 `detect_thresh=0.5`；`sigma=0.2`、`lambda=1e-4`、`interp_factor=0.075`）。
- 🔴 **改参数的坑**：`TrackerKCF::read()` **先重置默认值、只读 YAML 里出现的键** ⇒ 必须用
  **`tracker.read(fs.root())`**（`fs.getFirstTopLevelNode()` 会踩 `isMap()` 断言）。
- 🔴 **`detect_thresh=0.5` 是全部失锁的唯一来源**：关掉后同批点击原本第 4~20 帧断的链条全部跟满 30 帧
  ⇒ 失锁是**外观置信度门限**，不是几何伪影。

### 10.2 两臂设计 + 协议一致性

- 两个臂：`kcf`（contrib 默认参数）与 `kcf_nt`（`detect_thresh=0.0`）⇒ 膝点主臂用 `kcf_nt`，`kcf` 臂做敏感性对照。
- 脚本：`manu/tracking/lookahead/kcf_cv.py`（**新增**，薄适配层，缺 contrib **显式报错**）；
  `manu/video_quality/vq_sot_link.py` / `vq_sot_link_driver.py` / `vq_sot_compare.py` / `vq_sot_verify.py` /
  `vq_sot_admission_verify.py` / `vq_degrade_knee.py` 全部新增 `--arms`（**默认 `xcorr,patch` ⇒ 行为不变**）；
  自研 numpy 版 `manu/tracking/lookahead/kcf.py` **已废弃删除**。
- ✅ **协议一致 22/22**：22 条源片的点击点 `(window,cell,click_mode,x,y,tex_*)` 与 xcorr 版**逐位相同**
  ⇒ 两家族画的是同一条退化曲线。
- ✅ **零漂移回归**：默认臂与旧产物**逐字节一致**（与服务器既有 22 份 JSON 相同，忽略 `elapsed_s`）
  ⇒ **§八/§九 的任何数字都没被本轮改动**。

### 10.3 KCF 版结果（数字全部从 `manu/video_quality/report/kcf_*.csv` 现场读）

**① 跟踪分 AUC（好片为正类，混池）**：短 **0.878634**（xcorr 0.898653）/ 长 **0.860606**（xcorr 0.841294）；
指标侧 AUC 与家族无关（global_contrast 0.918928 / edge_density 0.901649 / lap_var 0.896152 /
blockiness 0.894445 / entropy 0.874495）；Spearman KCF **0.45~0.74**（xcorr 0.45~0.83）。

**② 合格线（4 条真实坏片硬切）**：短 IR `global_contrast ≥ 0.802616` / `blockiness ≤ 1.07442` / `entropy ≥ 7.37059`；
短 VIS `lap_var ≥ 2384.46` / `global_contrast ≥ 0.737747`；长 IR `blockiness ≤ 1.040557`；
长 VIS `lap_var ≥ 3357.21`（覆盖 3.0%）/ `edge_density ≥ 0.197635`（覆盖 20.1%）。
🔴 长时程 VIS AND 组合后**好片 0/691** ⇒ **「长时程 VIS 无可用单指标单线」不变**。

**③ 膝点（396 个退化变体，X=退化后实测画质 / Y=SOT 锁定率）**：短 IR global_contrast **0.577855**（xcorr 逐位相同）、
lap_var **6.918433**、edge_density 0.004803；长 IR global_contrast **0.57763**、lap_var **7.843251**（逐位相同）、
edge_density **0.007345**（逐位相同）；短 VIS lap_var **270.794141**；
🔴 长 VIS lap_var **270.794141 vs xcorr 171.384126（+58% 不一致）**，且 global_contrast / edge_density 在 KCF 侧
`knee_reliable=0` ⇒ **长时程 VIS 继续按「不采用」处理**，只作物理参照。

**④ 盲测（4 坏 + 18 好）**：KCF 与 xcorr **逐位相同** —— 短/长 VIS `lap_var ≥ 270.794` 片级 **0% / 100%**；
短/长 IR `global_contrast ≥ 0.577855` 片级 **16.667% / 100%**（窗级 26.055% / 88.789%）；
🔴 IR `lap_var` 单独用 **0% / 0%（无区分度）**，必须与 `global_contrast` 组合。

**⑤ 敏感性（默认参数臂 `kcf`）**：五格与 `kcf_nt` **逐位相同**，仅长 IR global_contrast 0.566708 vs 0.57763（−1.9%）。

### 10.4 🔴 本轮唯一的 FAIL（必须一起汇报）

`vq_sot_verify.py` CHECK 1「与 §271 网格法逐片一致性」：xcorr 平均差 **0.082（worst 0.208，PASS，容差 0.12）**，
**KCF 0.161（FAIL）**。该检查是**跨实现一致性**、**不是**阈值推导正确性；其余 verify 检查在 KCF 臂全 PASS，
`vq_sot_admission_verify.py` 长/短双口径 **[RESULT] PASS**。
⇒ **不同跟踪器家族的绝对锁定率不可互换** ⇒ **KCF 数字只在 KCF 家族内自洽**，与 xcorr **并列呈现、绝不合并**；
对外只报「同一指标、两个家族给出同一条线」的定性结论 + 各自的具体线值。

### 10.5 产物 / 复现 / 诚实口径

- **产物**：`manu/video_quality/report/kcf_knee_thresholds.csv`、`kcf_knee_blindtest.csv`、`kcf_knee_combos.csv`、
  `kcf_degrade_curve.csv`、`kcf_default_*.csv`；聚合目录 `report/sot_kcf_short/`、`report/sot_kcf_long/`；
  图 `/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/chart_degrade_knee_kcf_{ir,vis}_*.png`（8 张）+ `/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/cache/ultralytics/repo-artifacts/manu/video_quality/report/chart_degrade_knee_kcfdefault_*`（8 张）。
- **复现（服务器 `192.168.99.40:32222`，`/tmp/pycharm_project_a7d966e6`）**：
  ```bash
  source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
  export PYTHONPATH=/tmp/pylibs_cvcontrib:/tmp/pycharm_project_a7d966e6:/tmp/pycharm_project_a7d966e6/manu/video_quality
  ```
  命令清单见报告 §9.10.11。
- **诚实口径（八条）**：①测的是「任意点选纹理可跟踪性」≠「跟踪无人机成功率」；②IR/VIS 分模态、混池线禁用；
  ③gap 线只作分界线；④坏源片仅 4 条属小样本；⑤膝点线是必要条件非充分条件；⑥`knee_reliable` 一票否决；
  ⑦**跨家族绝对锁定率不可互换（KCF verify FAIL）**；⑧KCF 是 contrib 的 FSR/KCF 变体且半分辨率处理。
- 🔴 **未提交（铁律〇之三之二）**，仅落工作区。

### 10.6 结论口径收口：「方法无关性（Method-Agnostic）」及其**边界**（2026-11-24 追加）

**一句话结论**：**像素互相关（xcorr）**与**经典相关滤波（KCF）**两类原理完全不同的跟踪器
**各自独立**反推出的**被采用的两条线几乎逐位复现**、**盲测坏片拦截率 100% 相同**
⇒ 这条红线卡的是**图像本身的物理信息底线**，不是某一个跟踪器的实现细节。

| 时程/模态/指标 | xcorr 膝点 | **KCF 膝点** | 判定 |
| --- | --- | --- | --- |
| 短 IR global_contrast | 0.577855 | **0.577855** | ✅ 逐位相同 |
| 短 IR lap_var | 6.493031 | **6.918433** | ✅ +6.5% 同量级 |
| 短 VIS lap_var | 306.473878 | **270.794141** | ✅ −11.6% 同量级 |
| 长 IR global_contrast | 0.573831 | **0.57763** | ✅ +0.7% 同量级 |
| 长 IR lap_var / edge_density | 7.843251 / 0.007345 | **7.843251 / 0.007345** | ✅ 逐位相同 |
| 长 VIS lap_var | 171.384126 | 270.794141 | 🔴 **+58% 不一致 ⇒ 不采用** |
| 长 VIS global_contrast / edge_density | 0.27881 / 0.078302（reliable=1） | 0.501745 / 0.089019（**reliable=0**） | 🔴 **不一致 ⇒ 不采用** |

**🔴 边界三条**：①方法无关性**只成立于被采用的两条线**（IR 短/长、VIS 短）；VIS 长时程两家族不一致，
但该格**本来就按「不采用」处理**（xcorr 侧 AND 组合好片 0/691）⇒ 不构成对结论的挑战；
②「一致」= **线位一致**，≠ **绝对锁定率一致**（跨家族不可合并，见 10.4）；
③坏源片仅 4 条 ⇒ 「坏片拦截 100%」仍属**小样本结论**，上线前须扩样重标。

**✅ 补跑（口径缺口闭合）**：KCF 臂 `vq_sot_verify.py` 首轮只跑了长口径就被 FAIL 截断，
短口径已补跑 ⇒ **mean 0.161 / worst 0.384（与长口径同一数值）**，CHECK 2/4/5 全 PASS
⇒ 🔴 **该 FAIL 是 KCF 家族与 §271 网格法的系统性刻度差，不是单一口径偶发**。

**📦 部署口径**：线上画质门禁**不跑跟踪器**（只算全图指标）⇒ **无需引入 `opencv-contrib`**；
但**产品跟踪链路若要用 OpenCV KCF**，编译/打包必须带 **`opencv_contrib` 的 `tracking` 模块**。

**报告对应**：`REPORT_视频质量指标.md` **§9.10.12**（含可直接引用的汇报话术）。
🔴 **未提交（铁律〇之三之二）**，仅落工作区。

### 10.7 📘 《图像解读手册》PDF（40 张图逐张讲清楚，2026-11-25 追加）

**做了什么**：把产物区散落的 **40 张 PNG** 自动合成**一本可交付的图解手册 PDF**（**50 页 / 3.7 MB**，不入库），
每张图独占一页，统一三段「**画的是什么 / 看图要点 / 一句话结论**」，膝点页另附**该格确切数值与来源 CSV**。

| 项 | 值 |
|:--|:--|
| 脚本 | `manu/video_quality/vq_chart_doc.py`（入库，纯 matplotlib，**不需要 cv2 / 视频解码**） |
| 复现 | `python3 manu/video_quality/vq_chart_doc.py`（**本地主机**，约 24 s） |
| 产物 | `cache/ultralytics/repo-artifacts/manu/video_quality/report/图像解读手册_40张图.pdf` |
| 输入 | 40 张 PNG + `knee_thresholds.csv` + `kcf_knee_thresholds.csv` + `kcf_default_knee_thresholds.csv` |

**🔴 排版三坑（勿重犯）**：① `**加粗**` 标记外泄 ⇒ 须**分段感知渲染**（`split("**")` → 换行携带加粗状态 → 按 run 切 `weight`）
且**必须注册 Noto CJK Bold 字面**（只注册 Regular 会静默退化成常规字重）；
② 图文空隙过大 ⇒ **先排正文算总高 → 再定图带 → 按实际 `draw_h` 把图注与正文贴到图下方**；
③ 页脚溢出 ⇒ **文件名 + md5 挪到图注行**，页脚只留短 FOOTER。

**✅ 验收**：`pdftotext` ⇒ `**` 残留 **0**；`pdfimages -list` ⇒ 内嵌 **40 张**；
图注 **40/40 文件名命中 + 40/40 md5 与磁盘逐位一致**；脚本内置溢出告警 ⇒ **0 条**。
**报告对应**：`REPORT_视频质量指标.md` **§六·附**。
🔴 **未提交（铁律〇之三之二）**，仅落工作区。

### 10.8 📘 红外画质评估 **v2 · Target-Centric**（目标中心 ROI，2026-11-26 追加）

**为什么做 v2**：§八~§10.7 测的都是「画面**背景**任意位置点一下能不能锁住」（48 宫格模拟点击）；
领导明确指出业务痛点是「**特定业务目标（无人机）本身**能不能跟得住」
⇒ v2 把度量窗口**收缩到目标本体 + 紧邻背景环**，只用本地 Anti-UAV 的**红外**模态。

| 项 | 值 |
|:--|:--|
| 代码 | `iqa_v2/`：`dataset_anti_uav.py`（红外单目标加载 + 5 帧窗口 + 尾帧产出）· `target_roi_metrics.py`（双区域 ROI 算子 + 指标方向表）· `degradation_4d.py`（4 维退化）· `run_eval_v2.py`（三臂跟踪 + 膝点 + 尾部 + 报告）· `verify_recalc_v2.py`（独立复算）· `test_iqa_v2.py`（机械门禁）· `make_manifest_v2.py`（清单） |
| 数据集 | `/mnt/data/siping/datasets/manu/anti-uav`（**仅 IR**，可见光不打开）223 序列 / 0 剔除 / **2442 窗口** / 小目标 **42.2%** |
| 执行机 | `huangzhe@192.168.99.40:32222`（conda env `uav`；KCF 需 `PYTHONPATH=/tmp/pylibs_cvcontrib`） |
| 复现 | `PYTHONPATH=/tmp/pylibs_cvcontrib python3 iqa_v2/run_eval_v2.py --phase all --workers 24`（**9.7 分钟** / 58,608 行） |
| 报告 | `iqa_v2/REPORT_V2_ANTIUAV.md`（入库） |

**双区域 ROI 算子**：核心区 `max(9, w) × max(9, h)` 贴目标中心；背景环 `max(48, 3·max(w,h))` 且**挖除**核心区
外扩 2 px ⇒ 两区交集**恒为 0**（机检断言）。指标：`local_scr` / `target_local_contrast` / `target_lap_var` /
`target_grad_mean`（核心）+ `surround_std` / `surround_edge_density` / `surround_blockiness`（环）
+ `global_contrast` / `global_lap_var`（全图对照）。

**🔴 contrib legacy KCF 的三条实现事实**：① `update()` **严格滞后一帧**（实测真值 120/128/136/144/152
⇒ `update(帧1..5)` 依次返回 120/128/136/144/152）⇒ 必须喂**尾帧 t+5** 才能报出「第 t+4 帧的位置」，
否则 KCF 臂被「一帧的运动量」系统性罚掉、**膝点曲线整段带偏**；
② `detect_thresh` **确实生效**（0.0 正常；0.99 ⇒ 响应峰达不到阈值 ⇒ 返回退化 (0,0) 矩形）；
③ 初始化框必须**精确落在目标本体**（偏 12 px ⇒ 锁到局部、以 12 px 固定偏差跟着跑，像「矩形不更新」）。

**✅ 约束线（各族膝点取最小，两族并列不合并）**：`target_lap_var` **819.551**（两族 0.00%）、
`target_grad_mean` 27.1303（0.00%）、`surround_std` 2.36438（0.00%）、`surround_blockiness` 0.620585（0.00%）、
`global_contrast` 0.242857（0.00%）；有差：`local_scr` 3.20777 vs 2.69153（16.09%）、
`target_local_contrast` 0.170531 vs 0.126782（25.65%）、`global_lap_var` 41.63 vs 9.11919（78.09%）。

**✅ 核心结论（零合成退化，真实红外尾部 10%/15%）**：跟丢预测 AUC
—— 局部 ROI `target_local_contrast` **0.7970** / `local_scr` 0.6938 **>** 全图对照
`global_lap_var` 0.6044 / `global_contrast` 0.5831；**CSRT 臂上全图指标低于随机（0.4070 / 0.4697）**。
尾部 10% 跟住率 `target_local_contrast` **53.7% vs 87.9%（−34.2 pp）**。
⇒ **「全图随机采点」测背景纹理，「目标中心 ROI」才测业务目标可跟性**——v2 假设成立。

**🔴 JPEG 族在红外上的物理特殊性（实测推翻任务书假设）**：`target_lap_var` **基本持平**（−2.0%，
DCT 振铃把高频塞回 9×9 核心区，抵消了背景抹噪的损失）、`surround_blockiness` **不升反降**（−16.8%，
低对比、噪声主导的红外背景被 JPEG **先抹噪**，块内梯度掉得比块边界梯度更快）、
`global_contrast` **+2.4%（方向相反）**；只有 `local_scr` 单调下降（−8.9%）⇒
**「块效应」在红外上根本不是 JPEG 的表征**，jpeg 族膝点**不能当码率下限**读；
码率信息损失要用 **MAE（退化帧 vs 原帧）** 量化（实测 12/12 个真实窗口逐档单调递增）。
已固化成机检用例 `test_jpeg_on_ir_is_not_a_sharpness_or_blockiness_axis`。

**🔴 今天修掉的 7 个真 bug**：`_clamp_rect` 中心/左上角语义错配（致 xcorr 跟住率 0%）/ `Tracker.init`
返回 `None` 被判失败（CSRT 恒失败）/ KCF 滞后一帧未补偿 / 相关性方向判据反了（画质类应正相关）/
Laplacian 核与边界口径（`ksize=3` 是**对角核**；`BORDER_DEFAULT` ≡ numpy `pad("reflect")`）/
`verify_knee` 误用 `r["tracker"]` / **测试 `OUTPUT_DIR` 路径错 ⇒ 数据洁净度门禁一直「跳过」**。

**✅ 验收**：独立复算 ROI 最大误差 **3.553e-15**、膝点误差 **0.000e+00**（容差 1e-6）；
`pytest` **25 passed**；`iqa_v2/output/manifest_v2.md5` **14/14 OK**（明细走绝对路径登记、不入库）。
🔴 **口径声明**：阈值用 **Anti-UAV 开源红外**标定，**只验证方法**；正式上板红线待内部 58 条业务 GT 回收后同流水线重标。
🔴 **未提交（铁律〇之三之二）**，仅落工作区。

### 10.9 📘 红外画质评估 **v2 · 深轮（deep round）· 两条红线**（2026-11-27 追加）

**为什么做深轮**：§10.8 的基线轮只到三族退化轴的**基础档位**，膝点只落在「画质—跟踪性能分界点」，
且**只有一条对外红线**（`target_local_contrast < 0.1891664`）、锐度维度 `target_lap_var` 因守卫未过而**作废**
⇒ 深轮把**三族退化轴的上限抬高**后重跑整条 v2 流水线，看能否拿到**更硬的弯折点**与**第二条对外红线**。
🔴 **本轮唯一的变量是「退化档位上限」**；守卫门槛与全部口径**一字未动**（放宽门槛被规格书定性为错误做法）。

| 项 | 值 |
|:--|:--|
| a-priori 规格 | `iqa_v2/DEEP_AXIS_SPEC_20261127.md`（**写在任何深轮结果落盘之前** ⇒ 判据不是照着答案写的） |
| 机器判定器 | `iqa_v2/acceptance_deep.py`（**22 条判定** = A 4 + B 6 + C 4 + D 8；**纯标准库**，host 直接跑；**只吃产物 CSV、不采信报告正文**） |
| **退化轴抬高（唯一变量）** | `blur` σ 上限 **4.0 → 8.0**（10 档：0.6/1/1.5/2.2/3/4/5/6/7/8）· `motion` 核长上限 **11 → 15**（7 档：3/5/7/9/11/13/15）· `contrast` α 下限 **0.15 → 0.05**（8 档：0.80/0.65/0.50/0.35/0.25/0.15/0.10/0.05；🔴 **α 越小越重**）⇒ 共 **25** 档 |
| 守卫门槛（**一字未动**） | `CURVE_MIN_SPAN = 0.05` · `KNEE_AGREE_PCT = 15.0` · `N_BINS` · 跟住容差 `max(5, 0.5·max(w,h))` · 基线档位表 |
| 规模 | Pass A 窗口 **2442**（不变）· 仿真池 **895**（不变）· 退化条件 **17 → 25** · 明细 **17,657 → 24,817 行 / 47 列** |
| 产物 / 报告 | `iqa_v2/output_deep/`（**独立于基线轮的 `iqa_v2/output/`**）· `iqa_v2/REPORT_V2_DEEP.md` · `iqa_v2/REPORT_V2_DEEP_ACCEPTANCE.md` |
| 执行机 | `huangzhe@192.168.99.40:32222`（conda env `uav`；KCF 需 `PYTHONPATH=/tmp/pylibs_cvcontrib`） |
| **复现命令位置** | 跑一轮：`iqa_v2/DEEP_ROUND_RUNBOOK_20261127.md` §二「唯一合法的深轮命令」= `python iqa_v2/run_eval_v2.py --conditions-profile deep`；验一轮（host，纯标准库）：主档 **§320** §六·四 ⇒ `python3 iqa_v2/acceptance_deep.py --deep-dir iqa_v2/output_deep --baseline-dir iqa_v2/output --code-dir iqa_v2 --report iqa_v2/REPORT_V2_DEEP.md --acceptance iqa_v2/REPORT_V2_DEEP_ACCEPTANCE.md --gates-json iqa_v2/_gates_deep_final_20261127.json --json-out /tmp/final_deep_verdicts.json`（加 `--strict` ⇒ 存在非 PASS 判定时退出码 1，可当门禁用） |

**✅ a-priori 判据 A1~A4 实测结果**（判据先写死，结果由判定器从 CSV 现场重算）：

| 编号 | 判据 | 判定 | 一句话证据 |
|:--|:--|:--:|:--|
| **A1** | deep 轴下**最重档**跟住率**显著低于 50%**（曲线真塌，不是「弯折」） | 🔴 **FAIL** | **无一组合跌破 50%**；最低三档 blur/xcorr **87.9%**、motion/xcorr **89.0%**、blur/kcf_legacy **91.2%** |
| **A2** | deep 轴下 **KCF 臂**是否也过守卫（blur/motion 达 A 级） | 🟡 **PARTIAL** | 过守卫：`target_local_contrast/blur`、`target_local_contrast/contrast`、`target_lap_var/blur`；**未过**：`target_local_contrast/motion`、`target_lap_var/motion` |
| **A3** | `target_lap_var` 是否拿到有效红线 | 🟢 **PASS** | 线位 **220.61511928541879**（决定族/臂 blur/xcorr；分箱跨度 **14.81 pp**、逐档跨度 **10.87 pp**、单调、`knee_reliable = 1`） |
| **A4** | `local_scr` 盲测区分间隔 ≥ 50 pp | 🔴 **FAIL** | 本轮 `line_value = nan`；参照阈值下区分间隔仅 **2.00 pp**（坏片拦截 **100.00%** vs 真实好片错误豁免 **98.00%**） |

> 🔴 **4 条主判据里 1 条通过（A3）、1 条部分通过（A2）、2 条不通过（A1、A4）**；
> **6 条反向判据 B1~B6 全 PASS**（门槛不许动 / 盲测集零参与阈值搜索 / 池子不变 / 线位取最严 / 不得比基线更松 / 档位表与规格逐位一致）
> ⇒ **本轮不得被表述成「深轮成功 / 达标」**。

**✅ 当前两条对外红线**（数值取 7 位有效数字）：

| 指标 | **对外红线** | 决定族 / 决定臂 | XCorr 侧 | KCF 侧 | CSRT 侧（只作参照） | **跨臂线位差** |
|:--|:--|:--|--:|--:|--:|--:|
| `target_local_contrast` | **< 0.1005622** | blur / `kcf_legacy` | 0.21857865 | **0.10056222** | 0.05885589491877143 | **53.99%** |
| `target_lap_var` | **< 220.6151** | blur / `xcorr` | **220.61512** | 359.85049 | 141.7701370550641 | **38.69%** |
| `local_scr` | **无有效红线（nan）· 不采用** | — / — | nan | nan | nan | nan |

🔴 **取值规则**（跨家族纪律）：对外红线 = 通过全部守卫的候选里**每族取最小值**（最严的一族说了算）；
**绝不跨臂求平均**（`0.1005622159` ≠ 平均 `0.1595704333`；`220.6151193` ≠ 平均 `290.2328029`）；
**CSRT 是同族对照臂，只作参照，不参与取值**。
🔴 **与基线轮对照**：`target_local_contrast` 深轮 **↓ 46.9%** 比基线 `0.18916642395250532` **更严** ⇒ 直接采纳 deep；
`target_lap_var` 基线**无数值（nan）** ⇒ 深轮是新线（相对任务书基准 819.55 **↓ 73.1%**），不构成放宽；
`local_scr` 深轮无线位 ⇒ **不放宽**（基线线位在形式上仍有效，但对外可用性另受 A4 约束）。
🔴 **`local_scr` 不采用的理由**：本轮**连有效红线都没有**（`line_value = nan`）；参照读数 **8.0293609** 下
真实素材区分间隔仅 **2.00 pp**、好片段留出集**错误豁免率 36.05%** ⇒ **不进对外免责声明**。

**🔴 三条硬限定（对外必须与正文同时给出，缺一条就是超发）**：

1. 🔴 **弯折点 ≠ 崩溃点**：两条线都是「**画质—跟踪性能分界点 / 弯折点**」，**不是**「**跟踪崩溃点**」——
   在净化底座上即使退化加到 σ=8 / L=15 / α=0.05，最重档跟住率仍有 **87.9% / 89.0% / 97.7%**，曲线**从未塌下来**（A1 不通过）。
   🔴 根因：整帧级模糊与拖影**打不塌**跟住率（跟踪器在大目标上有强形状先验）⇒ 要塌得下去得用**目标级**退化
   ⇒ **「加档位」不是通用药**，它解决的是**指标轴区分度**，不是**跟住率塌陷**。
2. 🔴 **算法无关性仍不成立**：两条线是「**单臂反解 + 独立盲测背书**」——`target_local_contrast` 由 **KCF** 臂反解、
   `target_lap_var` 由 **XCorr** 臂反解，两族线位相差 **53.99% / 38.69%** ⇒「XCorr ↔ KCF 算法无关性」**本轮未复现**（A2 部分通过）。
   🔴 口径澄清：**53.99% / 38.69% 是「跨臂线位差」**，**不是**「双算法偏差」——`KNEE_AGREE_PCT = 15%` 判的是
   **同一条曲线**上 Kneedle vs PCHIP 两种取点法的偏差，**跨臂线位差根本不设门禁**；
   变化轨迹：基线轮是「KCF 根本没有线」（不可比），深轮是「两臂都有线，但线位互相矛盾」
   ⇒ 从「无法互证」变成「**互证失败**」，算法无关性**并没有被建立起来**。
3. 🔴 **盲测样本极小**：真实坏片 / 真实好片对照**各只有 2 条**（坏片 `120BT-04-红外02`、`130CT-02红外_IRT`；
   好片 `VIDEO00007`、`VIDEO00008`）⇒ **小样本级观测，不可外推**到其它机芯或其它气象条件；
   且真实好片对照段是**无目标的停车场航拍**（有车、有行人，**没有无人机**；DoG 只能拾到地面纹理）
   ⇒ 🔴 `local_scr` 区分不开的根因**在素材、不在深度**，要救它必须换素材（**有目标**的真实好片对照），不是继续压轴。

**✅ 盲测与三分法回归**（B2：盲测集全程零参与阈值搜索）：

| 指标 | 真实坏片拦截（248 帧） | 真实好片错误豁免（300 帧） | **区分间隔** | 好片段留出集错误豁免率（380 窗） | 一般片段临界敏感度（773 窗） | 跟丢片段拦截率（394 窗） |
|:--|--:|--:|--:|--:|--:|--:|
| `target_local_contrast` | **100.00%** | **2.33%** | **97.67 pp** ✅ | **4.47%**（17/380） | **14.62%**（113/773） | **23.86%**（94/394） |
| `target_lap_var` | **97.18%** | **0.00%** | **97.18 pp** ✅ | **0.79%**（3/380） | **2.33%**（18/773） | **5.84%**（23/394） |
| `local_scr`（参照读数，**非红线**） | **100.00%** | **98.00%** | **2.00 pp** ❌ | **36.05%**（137/380） | **57.05%**（441/773） | **72.84%**（287/394） |

🔴 两轮 `blind_real_v2.csv` md5 **逐字节一致**（`420d7337cb0a5d9b3dda553de5830744`），两条约束线**只**用合成侧
`knee_thresholds_v2.csv` 就能逐位复算 ⇒ 线位**不可能**是由真实素材反推出来的。
🔴 跟丢片段拦截率明显低于真实坏片 —— **这是设计意图**：跟丢片段里相当一部分是「**算法侧失锁**」（快速机动 / 干扰 / 目标出框），
画质并不低 ⇒ 红线只应拦**物理侧不可解释**的失锁，否则会把算法缺陷也洗成免责项。

**✅ 验收**：host `pytest iqa_v2/ -q` **152 passed / 0 skipped / 0 failed**（服务器 **129 passed / 2 skipped / 0 failed**）；
`verify_recalc_v2.py --samples 120` ROI 抽样 120 行最大绝对误差 **6.821e-13**、膝点复算 21 个 **0.000e+00**、
曲线有效性复算 162 条不一致 **0** 条；`audit_final.py` 对深轮与基线轮均 **PASS 41 / FAIL 0 / WARN 0**；
`md5sum -c iqa_v2/output_deep/manifest_deep.md5` rc=0（host **OK 34 / FAILED 0**）；
`canonical_digest.py verify` rc=0（`md5:82fb447b38160379cde55722e792fe12`）；
`compare_rounds.py --strict` rc=0 ⇒ `target_local_contrast` **DEEP_STRICTER**、`target_lap_var` **DEEP_NEW**、
`local_scr` **DEEP_NONE** ⇒ **无更松线位**；判定器最终 **PASS 19 / PARTIAL 1 / FAIL 2**
（22 条里剩的 2 条 FAIL 就是 **A1 / A4** 这两条**如实记录的实测结论**，不是门禁缺陷）。
🔴 基线轮**冻结未改**：`iqa_v2/output/manifest_v2.md5` 生成前后 md5 均为 `a72bc7fa5c267b853943416fb2670820`。
🔴 **口径声明**：阈值仍用 **Anti-UAV 开源红外**标定，**只验证方法有效**；
正式上板红线待内部 **58 条**红外业务 GT 序列回收后，用**同一套流水线**重标（**不是**「这就是正式上板红线」）。
🔴 **未决**：A1 的真问题（曲线塌不下来）本轮未解决（需要**目标级**退化）；
`local_scr` 的真问题（素材里没有目标）本轮未解决；`manu/video_quality/` 的 v1 代码**一行未动**。
🔴 **未提交（铁律〇之三之二）**，仅落工作区。
