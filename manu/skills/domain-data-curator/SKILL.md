---
name: domain-data-curator
description: |
  把新场景（沙漠 / 海面 / 低空密林等）的红外无人机外场数据，规范成仓库标准的
  「官方语义 3 通道 GMC 特征数据集 + train / anchor / blind / ignore 四类划分 + 可对账的 GT」。
  覆盖人工定界、特征基底固化（通道逐像素对齐 + sha256 抽样）、数据划分防护网、GT 分母对账。
  When the user asks to 接入新场景数据 / 人工定界 / 划 ignore 和 blind / 建特征数据集 /
  校验数据划分 / 对一下 GT 分母 / 新场景指标明显偏低或偏高 / 把沙漠或海面数据并进训练，
  load this skill before touching anything under manu/data/ for that scene。
  Near-misses（这些不要用本 skill）：纯训练调度与混训比例设计 → mixed-ft-orchestrator；
  只做原始视频解码成 JPG 而不涉及特征语义与划分 → 直接用 manu/data 下的预处理脚本；
  修改已有 GT 标注内容本身（标签修复是独立任务，不是规范化）。
  本 skill 只做数据侧规范化，不负责判定模型好坏，也不负责上线选型。
descriptions:
  zh-Hans: 新场景红外无人机数据的领域规范化 SOP：人工定界、官方语义特征基底固化、四类划分防护网与 GT 分母对账。
displayNames:
  zh-Hans: 领域数据规范化
---

# Domain Data Curator

## Inputs to collect

动手前先确认下面每一项；缺项就去问用户或去读 `manu/memory/memory.md`，不要用假设值硬上。

- **场景标识与原始帧根目录**（服务器绝对路径）。原始帧必须已经落成 JPG，目录形如
  `/mnt/data/siping/datasets/manu/<scene>/frames_ir_jpg`。原始视频解码不在本 skill 范围内。
- **业务域合并 GT 目录**（服务器绝对路径）。命名 `{seq}__{idx:06d}.txt`，每行 ≥ 5 个字段
  （`cls cx cy w h`）。龙泉山当前唯一真值是
  `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010`。
- **序列清单**：序列名、帧数、是否已有人工复核结论。已有复核结论就直接复用结论，不要重看视频。
- **目标产物根目录**（`--out-root`）。放在 `/mnt/data/siping/datasets/manu/` 下，
  命名 `<scene>_ft_gmc_median_official/`，避免和历史 `fpv` 口径缓存混在一起。
- **本轮是否要 train 组**：要参与混训就产出 train 组；只做评测就只建 val 组。
- **特征来源策略**：`rebuild`（从原始帧重算，权威口径）还是 `copy`（复用既有特征缓存，
  前提是已证明语义一致 —— 见 Procedure 步骤 5）。
- **是否沿用既有 split config 作为模板**：龙泉山模板是 `manu/data/longquan_ft_split.json`。

## Procedure

### 0. 确认执行机与不可动的基座

本 skill 的绝大多数命令在**服务器**（`ssh -p 32222 huangzhe@192.168.99.40`）执行，
因为数据在服务器盘 `/mnt/data` 上；本机只有 SSHFS 只读镜像
`/home/manu/mnt/data/...` 和代码镜像 `/home/manu/mnt/pycharm_project_10ae9e2e`（写代码即落服务器）。

先按下列顺序建立环境（服务器上 `python` 无 3 后缀，conda 环境名为 `uav`）：

```bash
# 服务器
ssh -p 32222 huangzhe@192.168.99.40
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
cd /tmp/pycharm_project_10ae9e2e
export PYTHONPATH=.
python -V   # 期望 3.10.20
```

三条不可动的基座，碰到就绕开，不要改：

- `manu/data/build_full_median_dataset.py` 是特征基底的**权威实现**，只调用或只读。
  改动会让已冻结的 baseline 数值失去可比性。
- 训练侧一律走 YOLO 官方 `build_yolo_dataset` + `build_dataloader`，
  不要另写 OpenCV 图像处理分支，不要改 `letterbox_gray`
  （定义在 `manu/data/preannotate_hm_bbox.py`，被 `manu/data/multiscale_heatmap_cache.py` 引用）。
- 新场景的改动先落在本地仓库 `/media/manu/1TB-Volume/workspace/ultralytics/`，
  再同步到服务器 `/tmp/pycharm_project_10ae9e2e`；同步是否成功**只用 md5 比对**判定。

### 1. 人工定界：把新场景切成 Ignore / Blind / Train+Anchor

这一步是外场数据接入的第一道卡口，目标 30 分钟内完成。判据只有三条，按顺序过：

1. **硬件非均匀性条纹（硬件 FPN）与非防空视角** → 划入 `ignore` 免责集。
   FPN 条纹是传感器固定缺陷，强行微调会让卷积核对纵向条纹过拟合；
   非防空视角（俯视公路、汽车、近距离人头）属于域外严重漂移，不是防空反无人机的常规视场。
2. **挑 1 条典型死穴** → 划入 `blind` 难例，做独立盲测集。
   典型死穴是"目标落复杂背景 + 机身变黑 + 召回低"，这类样本用来看门禁是否真的在做事。
3. **其余全部** → 划入 `train` 与 `anchor`（黄金池）。
   train 组要覆盖典型失效机理：核心攻坚（暗刚体正切片 + 地表高频负切片）、
   负样本挖掘、运动补偿；anchor 组是质量达标的标准空域基准。

人工复核底稿写成 `manu/docs/dataset/<scene>-tk.md`，格式对齐龙泉山底稿
`manu/docs/dataset/long-quan-tk.md`：文件分 `# pass` 与 `# unpass` 两段，
每行是 `[pass]|[ignore] <SEQ>_F1-<val>.mp4 - <原因>`。
龙泉山实测量级是 24 条序列 = 16 条 pass + 8 条 un-pass，其中 7 行带 `<ignore>` 前缀，
剩下 3 条是真实缺口序列：`003002`（F1 77.815）、`001324`（F1 84.357）、`002001`（F1 87.877）。

### 2. 把定界结果写成 split config JSON

复制 `manu/data/longquan_ft_split.json` 的结构改写，不要自创字段名 —— 下游脚本按这套字段名读。

顶层键：`version` / `created` / `author` / `purpose` / `rationale` / `sampling` / `feature_spec` /
`groups` / `baseline_reference`。

`baseline_reference` 至少要能给出冻结基线数值，其中两个键是自检脚本的硬依赖：
`golden_pool_single_frame.gt` 与 `blind_001324_single_frame.gt`
（键名沿用龙泉山；新场景把"golden pool 7 条的 GT 行数"和"blind 难例 1 条的 GT 行数"填进去）。

`groups` 下三个组，各带 `description` + `sequences`：

| 组 | 序列 role 取值 | 用途 |
|:--|:--|:--|
| `val` | `golden_pool`、`blind_hardcase` | 黄金守护池 + 独立盲测（龙泉山：7 + 1 = 8 条） |
| `train` | `core_positive_mining`、`negative_mining`、`motion_compensation` | 微调养料（龙泉山：9 条） |
| `ignore` | 只写 `sequence` + `reason` | 直接排除出训练与常规验证闭环（龙泉山：7 条） |

`sampling` 至少写 `val_stride` / `val_start` / `train_stride` / `train_start`：
龙泉山是 val 步长 1 起点 0（全帧，含第 0 帧）、train 步长 5 起点 0。
Anti-UAV 参考集的 train 是步长 5 **起点 6**、val 步长 1 起点 3；要完全复刻就传 `--train-start 6`。

`feature_spec` 写死五项（`window` / `stride_step` / `downscale` / `gmc_semantics` / `naming`）：

- `window` = 21（中值窗口）、`stride_step` = 2、`downscale` = 2。
- `gmc_semantics` = **逐 lag 直接拟合**（`compute_affine(h, t)`），与 `manu/data/build_full_median_dataset.py` 一致；
  仿射连乘（FPV）变体属于历史口径，不要在新场景启用。
- `naming` = `<sequence>__frame_<真值帧号:06d>.jpg`，与同名 `.txt` 1:1 镜像。

### 3. 构建官方语义特征数据集

```bash
# 服务器（cd /tmp/pycharm_project_10ae9e2e 后执行）
python manu/data/build_longquan_ft_dataset.py \
  --split-config  manu/data/<scene>_ft_split.json \
  --raw-root       /mnt/data/siping/datasets/manu/<scene>/frames_ir_jpg \
  --labels-root   /mnt/data/siping/datasets/manu/<scene>/label/Final_Labels_merged_<日期> \
  --out-root      /mnt/data/siping/datasets/manu/<scene>_ft_gmc_median_official \
  --gmc-semantics official \
  --feature-mode  rebuild \
  --workers 16 \
  --split-chunks 4
```

关键点：

- **`--gmc-semantics official` 必须显式传**。该参数默认值是 `fpv`
  （`manu/data/build_longquan_ft_dataset.py:467`），传默认会得到仿射连乘语义的缓存，
  与 Anti-UAV 官方集不同源，跨场景指标不可比。
- `--feature-mode rebuild` 从原始帧重算，是权威口径；`copy` 复用既有缓存，
  必须先跑完步骤 5 的语义对齐验证再允许。
- `--split-chunks N` 把每条序列切成 N 块并行，不改变任何一帧的计算结果
  （特征所需上下文 t-1 / t-2 / 中值窗口 t-2k 始终按**绝对帧号**从原始帧读取）。
- `--dry-run` 先跑一遍只打印计划与统计，确认序列数与步长符合预期再落盘。
- `--manifest-only` 只按磁盘真值重写 `data.yaml` + `split_manifest.json`，不做图像计算；
  适合先把划分接线正确再开算力。
- `--seqs` 只构建指定序列，用于单序列试跑定位问题。

`--groups` 默认 `val,train`，`ignore` 组不构建 —— ignore 是免责集，不是数据。

### 4. 数据集自检与 GT 对账

```bash
# 服务器
python manu/data/check_longquan_ft_dataset.py \
  --dataset-root /mnt/data/siping/datasets/manu/<scene>_ft_gmc_median_official \
  --split-config manu/data/<scene>_ft_split.json \
  --labels-root  /mnt/data/siping/datasets/manu/<scene>/label/Final_Labels_merged_<日期> \
  --sample-features 40 \
  --out-json /mnt/data/siping/datasets/manu/<scene>_ft_gmc_median_official/check_report.json
```

自检覆盖 8 项（`manu/data/check_longquan_ft_dataset.py:4-15`）：
结构（images 与 labels 严格 1:1 镜像）、划分（train/val 互斥、ignore 序列完全缺席、三组全覆盖全部配置序列）、
采样（val 帧号连续 0..N-1 含第 0 帧、train 帧号等差 step=5）、GT 对账、标签合法性
（5 字段、类 id 合法、cx/cy/w/h ∈ [0,1]、w/h > 0）、特征健康度（抽样 3 通道、尺寸、通道互异）、
源一致性（抽样比对数据集标签与合并真值源文件）、data.yaml 可解析。

GT 对账是硬指标：脚本从 split config 的 `baseline_reference` 读期望值
（`manu/data/check_longquan_ft_dataset.py:100-103`），
按角色分组求和 —— `role == "golden_pool"` 的序列 GT 之和必须等于
`baseline_reference.golden_pool_single_frame.gt`，
`role == "blind_hardcase"` 的序列 GT 之和必须等于
`baseline_reference.blind_001324_single_frame.gt`。

⇒ **新场景的 split config 必须提供这两个键**（键名沿用龙泉山，值填新场景自己的冻结 GT：
golden pool 7 条的 GT 之和、blind 难例 1 条的 GT）。
缺键会直接 KeyError；有键但对不上，说明标签链路错了，不要靠调阈值绕过去。

#### 🔴 GT 分母 ≠ 文件数

这是最容易算错、且错了会让指标**系统性偏低约 20%** 的地方。

龙泉山业务域 GT 目录 `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010`
实测目录项 **165,224** = 165,223 个标签文件 + 1 个 `classes.txt`；
但**评测分母 GT = 133,279 行**，即标注框行数（每行 ≥ 5 字段，去重前）。

165,223 / 133,279 ≈ 1.24 —— 拿文件数当分母，指标会被压低约 20%。
新场景对账时也要按**标注框行数**统计，不要按文件数：

```bash
# 服务器：统计 ≥5 字段的标注框行数（去重前）
# 用 "打印一行再 wc -l" 而不是 "awk 累加 n"：xargs 会把上万个文件拆成多次 awk 调用，
# 累加变量在每次调用间会重置，导致少算。
LAB=/mnt/data/siping/datasets/manu/<scene>/label/Final_Labels_merged_<日期>
find "$LAB" -maxdepth 1 -name '*.txt' ! -name 'classes.txt' -print0 \
  | xargs -0 -r awk 'NF>=5 {print FILENAME":"FNR}' \
  | wc -l
```

### 5. 特征基底固化：通道序 + 逐像素对齐 + sha256 抽样

目标是**零协变量偏移**：同一条原始帧，在任何机器、任何脚本里生成的特征必须逐字节一致。

先确认通道语义与模型输入通道序一致（磁盘序 `[I_t, |I_t - W(I_{t-2})|, (I_t - B_t)^+]`，
模型输入序 `[median_residual, gmc_diff, I_t]`，推理时要翻转）：

```bash
# 服务器
python manu/data/probe_channel_order_final.py \
  --data /mnt/data/siping/datasets/manu/uav_gmc_median/data.yaml \
  --frames 4 --batch 4 --imgsz 640 \
  --out-json /mnt/data/siping/datasets/manu/channel_order_probe.json
```

再证明"新场景的 GMC 语义与官方集逐像素一致"（这一步是 `copy` 模式的前置条件）：

```bash
# 服务器
python manu/data/verify_antiuav_gmc_parity.py \
  --raw-root      /mnt/data/siping/datasets/manu/<scene>/frames_ir_jpg \
  --canonical-root /mnt/data/siping/datasets/manu/uav_gmc_median/images/val \
  --sample 25 --window 21 --stride-step 2 --downscale 2 \
  --out-json /mnt/data/siping/datasets/manu/<scene>_gmc_parity.json
```

该脚本逐字节 sha256 比对（`manu/data/verify_antiuav_gmc_parity.py:134`），
要求与官方产物 125/125 逐字节全等。

已有特征缓存要落到新数据集上时，用 `verify_longquan_ft_features.py` 做 sha256 逐文件比对：

```bash
# 服务器
python manu/data/verify_longquan_ft_features.py \
  --dataset-root /mnt/data/siping/datasets/manu/<scene>_ft_gmc_median_official \
  --cache-root   /mnt/data/siping/datasets/manu/<scene>/features \
  --split-config manu/data/<scene>_ft_split.json \
  --groups val,train \
  --out-json /mnt/data/siping/datasets/manu/<scene>_features_verify.json
```

要追某个具体帧的数值来源（怀疑通道错位或窗口错位）时用：

```bash
# 服务器
python manu/data/verify_longquan_feature_source.py \
  --raw-root      /mnt/data/siping/datasets/manu/<scene>/frames_ir_jpg \
  --features-root /mnt/data/siping/datasets/manu/<scene>/features \
  --sequence <SEQ> --frames 100,101,102 \
  --window 21 --stride-step 2 --downscale 2 \
  --out-json /mnt/data/siping/datasets/manu/<scene>_feature_source.json
```

再核一遍"落盘特征 == 训练时在线算的特征"（防止离线缓存与在线增强漂移）：

```bash
# 服务器
python manu/data/verify_onthefly_features.py \
  --raw-root /mnt/data/siping/datasets/manu/<scene>/frames_ir_jpg \
  --features-root /mnt/data/siping/datasets/manu/<scene>_ft_gmc_median_official/images/train \
  --sequence <SEQ> --n-frames 200 --stride 5 \
  --median-window 21 --temporal-stride 2 --downscale 2 \
  --out-json /mnt/data/siping/datasets/manu/<scene>_onthefly_verify.json
```

### 6. 划分防护网

- **步长配置写进 split config，并在构建时显式落盘**：`sampling` 段 + `split_manifest.json`
  一起留档；事后只靠目录结构反推步长会把 val/train 弄混。
- **GT 标签用纯数字命名**。`manu/evaluation/eval_multiscale_cache_th.py:100` 的 `load_ground_truth`
  取 `name[len(prefix):-4]` 后判 `isdigit()`（第 110 行），不通过就 `continue`
  —— 只认 `{seq}__{digits}.txt`：如果中间多出 `frame_` 之类字段，
  该文件会被**整条跳过**（静默清零，不报错）。
  业务域 GT 沿用 `{seq}__{idx:06d}.txt`；数据集内的 `{seq}__frame_{idx:06d}.jpg` 是特征文件名，
  不是评测 GT 文件名，两者不要混。
- **`Path.glob` 路径写错会静默返回空列表**。任何 glob / listdir 之后，先断言非空再往下走，
  否则后面所有统计都会是 0 且不报错：

```python
from pathlib import Path
files = sorted(Path(root).glob("*.jpg"))
assert files, f"glob 返回空：检查路径 {root}"
```

- **`frame_index` 是排序位置，不是原始帧号**。`manu/data/multiscale_heatmap_cache.py:403`
  用 `frame_index = start + index` 记录排序后的位置；把它当原始帧号回查 GT 会整体错位。
- **门禁/评估的 `--fusion-dilate` 要显式传 32**。`manu/data/multiscale_heatmap_cache.py:69`
  的默认值是 9，门禁脚本 `manu/training/auto_epoch_gate_check.py:382` 显式传的是 32；
  自己手写缓存脚本时漏传就会用 9，阈值与膨胀半径一起错。

### 7. 评测链路接线

- `--labels-root` 取**服务器侧路径**，不要用本机 SSHFS 路径
  `/home/manu/mnt/data/siping/datasets/manu/...`；SSHFS 路径在评测里会引入 I/O 抖动，
  且不同机器上的绝对路径不一致会让缓存与报告无法复用。
- 评测脚本：`manu/evaluation/eval_multiscale_cache_th.py`（认 `{seq}__{digits}.txt`）、
  `manu/evaluation/system_sota_longquan.py`（`--cache-root` / `--labels-root` / `--seqs` / `--out-json`）。

```bash
# 服务器
python manu/evaluation/system_sota_longquan.py \
  --cache-root  /mnt/data/siping/datasets/manu/<scene>_ft_gmc_median_official/cache \
  --labels-root /mnt/data/siping/datasets/manu/<scene>/label/Final_Labels_merged_<日期> \
  --seqs <SEQ1>,<SEQ2> \
  --out-json /mnt/data/siping/datasets/manu/<scene>_sota.json
```

- **GPU 铁律**：通道序探针、特征缓存、门禁/评估**不得与正在训练的 trial 同卡**；
  单卡同时只允许"训练"或"评估"二者之一。先用 `nvidia-smi` 确认目标卡空闲再起评估进程，
  否则会 OOM 并连带打断训练。

- **执行机搞混**：本 skill 的命令全部针对 Linux 服务器（`ssh` / `conda` / `find` / `awk`）。
  Windows（win32）客户端上只做本地仓库的代码编辑与草稿，**不要**在 win32 上执行这些命令；
  本机的 SSHFS 路径 `/home/manu/mnt/...` 只用于读代码镜像，不作为评测 `--labels-root`。

## Output contract

完成后应当能交出下面这几项，缺一项就说明流程没走完：

1. **人工复核底稿**：`manu/docs/dataset/<scene>-tk.md`，`# pass` / `# unpass` 两段，
   每行 `[pass]|[ignore] <SEQ>_F1-<val>.mp4 - <原因>`。
2. **split config**：`manu/data/<scene>_ft_split.json`，含 `groups` / `sampling` / `feature_spec` /
   `baseline_reference` 四段可用内容。
3. **数据集根目录** `/mnt/data/siping/datasets/manu/<scene>_ft_gmc_median_official/`，
   含 `data.yaml`、`images/{train,val}`、`labels/{train,val}`、`split_manifest.json`、
   `check_report.json`。
4. **自检报告**：`check_report.json` 全绿，且 GT 对账行数与 `baseline_reference` 一致；
   外加通道序探针 JSON、GMC parity JSON（sha256 逐字节全等）、在线-离线一致性 JSON。
5. **一句话交接**：新场景的 val/train 帧数、GT 行数、golden pool 与 blind 各多少条序列，
   以及"1 条 blind 难例是哪一条"。混训接入参数交给 `mixed-ft-orchestrator`。

## Failure handling

- **glob / listdir 返回空**：先别继续。把路径、挂载、权限逐个查一遍，
  并在代码里加 `assert files, ...`。空列表一路走到底，得到的全是 0 且不报错，
  这是最常见的"数据莫名其妙为空"根因。
- **GT 行数与 `baseline_reference` 对不上**：优先怀疑标签链路（命名、字段数、去重），
  不要调评测阈值去凑。命名必须纯数字；`frame_` 之类字段会被 `isdigit()` 静默跳过。
- **指标比预期低 ~20%**：先查分母是不是被当成了文件数。分母是标注框行数。
- **`--labels-root` 指向了 SSHFS 路径**：改回服务器侧 `/mnt/data/...` 路径再跑。
- **构建报"样本数异常 / 立即中止"**：那是满额护栏（龙泉山 13,918 + Anti-UAV 43,008），
  说明上游数据集不完整或 data.yaml 指错。先修数据，不要绕过护栏。
- **特征与官方集 parity 不是逐字节全等**：不要用 `copy` 模式；回到 `--feature-mode rebuild`，
  并用 `verify_longquan_feature_source.py` 逐帧追数值来源。
- **门禁/评估 OOM**：多半是评估进程和训练 trial 抢了同一张卡。先停评估，
  用 `nvidia-smi` 换到空闲卡再起；不要靠减 batch 掩盖冲突。
- **必须回报用户而不是硬上的情况**：新场景没有任何一条序列能同时满足
  "有目标 + 常规防空视场 + 信噪比可用"（全段黑热 / 纯 FPN / 域外视角）；
  或者 GT 标注质量不足以支撑"盲测"这个角色；
  或者新场景特征 parity 与官方集不一致且定位不到原因。
  这三种情况继续往下走只会产出不可比的指标，应当带着已完成的证据回报用户并给出建议。

## Examples

### 例 1（canonical，已真实执行）：龙泉山 official 特征集构建 + 自检

输入：
- split config：`manu/data/longquan_ft_split.json`（val 8 条 = 7 golden_pool + 1 blind_hardcase
  `VIDEO00007_19700101_001324`；train 9 条；ignore 7 条）
- 原始帧根：`/mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg`
- GT 根：`/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010`
- 产物根：`/mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/`

执行：

```bash
# 服务器
cd /tmp/pycharm_project_10ae9e2e
source /home/huangzhe/anaconda3/etc/profile.d/conda.sh && conda activate uav
export PYTHONPATH=.

python manu/data/build_longquan_ft_dataset.py \
  --split-config  manu/data/longquan_ft_split.json \
  --raw-root       /mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg \
  --labels-root   /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
  --out-root      /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official \
  --gmc-semantics official \
  --feature-mode  rebuild \
  --workers 16 --split-chunks 4

python manu/data/check_longquan_ft_dataset.py \
  --dataset-root /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official \
  --split-config manu/data/longquan_ft_split.json \
  --labels-root  /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
  --sample-features 40 \
  --out-json /mnt/data/siping/datasets/manu/longquanshan_ft_gmc_median_official/check_report.json
```

输出（实测值）：
- 特征集 `train` 13,918 帧 / `val` 28,997 帧（golden pool 17,187 帧 + 001324 11,810 帧）。
- GT 对账：golden pool **16,638** 行、001324 **10,535** 行，与 `baseline_reference` 完全一致。
- 底稿 `manu/docs/dataset/long-quan-tk.md`：24 条序列 = 16 pass + 8 un-pass，
  其中 7 条带 `<ignore>`（3 条 FPN 条纹、2 条全段黑热、1 条极端恶劣复合、1 条域外漂移），
  真实缺口 3 条 = `003002` / `001324` / `002001`。

### 例 2：把新场景（例如沙漠）并进混训前的最小动作

输入：新场景已按步骤 1~4 产出 `/mnt/data/siping/datasets/manu/desert_ft_gmc_median_official/`，
但 train 组帧数远小于 Anti-UAV 大盘。

执行：
1. 用 `manu/data/check_longquan_ft_dataset.py` 跑一遍新场景（脚本参数名与场景无关，
   `--dataset-root` / `--split-config` / `--labels-root` 指到新场景即可）。
2. 在 mix 侧用 `--longquan-data` 指向新场景的 `data.yaml`；
   但要注意 `manu/training/optuna_mix_ratio_search.py:102` 的满额护栏
   `LQ_EXPECT = 13918 / UAV_EXPECT = 43008` 是按龙泉山 official 写死的，
   换场景必须同步改护栏常数，否则会在 `[DATA-GATE]` 直接中止 —— 这是刻意的中止，不是 bug。

输出：新场景的 `data.yaml` + `check_report.json` 就绪，且已明确记录"护栏常数是否需要改、改成多少"。
