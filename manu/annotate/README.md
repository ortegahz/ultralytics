# 视频逐帧标注工具 (Web Video Annotator)

递归扫描原始文件夹里的所有视频，用**多尺度 Trial 0474 热图 SOTA** 做预标注，然后在浏览器里
像 LabelMe 那样逐帧查看、调整、保存 YOLO 格式标注。核心目标是**把逐帧标注的时间压下来**。

```
manu/annotate/
  server.py        零依赖 HTTP 服务（只用 http.server + cv2/numpy）
  dataset.py       递归发现视频/帧目录 + 缓存解码
  labels.py        YOLO 标签读写、原子保存、去重、复核状态、时间线统计
  jobs.py          预标注任务调度（SSH 到训练服务器跑推理）
  worker_infer.py  多尺度 Trial 0474 推理 worker（跑在服务器 uav 环境）
  static/          Canvas 前端
  selftest.py      端到端自测（36 项）
```

## 复用已有预标注（推荐先这样起步）

项目里已经有跑好的 SOTA 预标注 `龙泉山/preannot_v1`（24 序列 / 165,223 帧，与帧数一一对应）。
用 `--prelabel-source` **原地只读挂载**，不用把 16 万个文件拷一遍：

```bash
--prelabel-source /home/manu/mnt/data/siping/datasets/manu/龙泉山/preannot_v1/labels/train \
--prelabel-class 1
```

- `preannot_v1` 是混合集：`0 = uav_bbox`（YOLO26 bbox 分支）、`1 = uav_hm_coarse`（Trial 0474 热图分支）。
  `--prelabel-class 1` 只取热图分支，与本工具的口径一致。
- 被选中的类别在展示与采纳时**统一重映射为 class 0**，因为标注集是单类；否则采纳一条候选
  会静默写成 class 1。
- 外部树**只读**：本工具只 `read`，人工修改永远落不到它上面。自有 `prelabels/` 优先于外部树。
- 两种布局都认：`<root>/<seq_id>/<帧号>.txt` 与 `<root>/<seq_id>__<帧号>.txt`（后者即 `preannot_v1` 的布局）。
- 外部候选没有分数边车文件，界面标注为「导入」，且不受阈值滑块过滤影响（缺失分数视为「始终显示」）。

⚠️ **口径差异**：`preannot_v1` 是旧预标注流程的产物（`hm_conf 0.06`、含 bbox 分支），
与本工具 worker 的多尺度融合（阈值 0.22 + 膨胀 9 + 连通域面积 ≥ 4）**不是同一口径**。
拿它起步是为了立刻有候选可用；要换成工具自己的多尺度口径，仍需点一次「预标注」重跑。

## 快速开始

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics

# 1) 启动服务（--root 会被递归扫描）
/home/manu/anaconda3/bin/python -m manu.annotate.server \
    --root /path/to/原始视频文件夹 \
    --workspace /home/manu/mnt/pycharm_project_10ae9e2e/manu/annotate_workspace \
    --port 8777

# 2) 浏览器打开
#    http://127.0.0.1:8777
```

`--workspace` 默认取 `<root>/../annotate_workspace`。**预标注需要 workspace 位于 SSHFS 挂载内**
（`/home/manu/mnt/data` 或 `/home/manu/mnt/pycharm_project_10ae9e2e`），否则训练服务器看不到它，
接口会直接报出这个原因而不是静默失败。

## 部署在 GPU 服务器上（推荐：取帧和推理同机）

服务也可以直接跑在训练服务器 `192.168.99.40` 上，数据和代码那边都有：

```bash
cd /tmp/pycharm_project_10ae9e2e          # 服务器上的仓库路径
~/anaconda3/envs/uav/bin/python -m manu.annotate.server \
    --root /mnt/data/siping/datasets/manu/龙泉山/frames_ir_jpg \
    --workspace /tmp/pycharm_project_10ae9e2e/manu/annotate_workspace \
    --prelabel-source /mnt/data/siping/datasets/manu/龙泉山/preannot_v1/labels/train \
    --prelabel-class 1 \
    --host 0.0.0.0 --port 8777
# 浏览器打开 http://192.168.99.40:8777
```

注意服务器上的路径前缀和本机**不同**：`/home/manu/mnt/data` ↔ `/mnt/data`，
`/home/manu/mnt/pycharm_project_10ae9e2e` ↔ `/tmp/pycharm_project_10ae9e2e`。

此时预标注走 `--inference-mode auto`（默认）的自动分支：`jobs.py` 比对 SSH 目标和本机网卡地址，
认出「自己就是目标服务器」后改为**就地 `bash -lc` 起 worker**，不再发 SSH。原因是服务器上没有存着自己的
私钥，`ssh 自己 → 自己` 会被拒（`Permission denied (publickey)`），按钮会点了没反应。
同一个开关也顺带关掉了路径翻译——本机上 `/mnt/data/...` 不匹配任何工作机前缀，硬套 `PATH_MAP`
会把每个序列都误判成「不可见」。

自动判断依据是网卡地址而不是 `socket.gethostbyname(hostname)`：后者在 Debian 下返回 `127.0.1.1`，
永远匹配不上真实 IP。需要手动钉死时用 `--inference-mode ssh` 或 `--inference-mode local`。

## 为什么推理默认在远端

本机 `uav` 环境是 `torch 1.13.1 + numpy 2.2.6`，张量互操作直接抛
`RuntimeError: Numpy is not available`；本机也没有任何 Web 框架。因此：

- 服务端只用 `http.server`，任何解释器都能起；
- 服务跑在别处时，推理一律通过 SSH 在训练服务器 `huangzhe@192.168.99.40:32222` 的 `uav` 环境执行
  （服务器 numpy 1.26.4 + torch 1.13.1 + CUDA 正常，4×RTX 4090 D）；
- 服务跑在服务器上时，同一个 `uav` 环境就地使用；
- **模型只在你点「预标注」按钮时才加载**，符合铁律一。

`jobs.py` 里的 `PATH_MAP` 负责本机路径 → 服务器路径的翻译，扩展挂载点改这一处即可。

## 预标注口径

严格对齐 `manu/data/multiscale_heatmap_cache.py`，不重新推导：

| 环节 | 取值 |
| :--- | :--- |
| 输入通道 | 磁盘序 `[I_t, GMC diff, median residual]` → 模型序 `[..., ::-1]` |
| GMC | `FastGMCEstimator(downscale=2)`，在**原始灰度**上估计 |
| 中值背景 | 21 帧窗口 × 步长 2 |
| 尺度 | 原生 letterbox + 160 + 80 |
| 融合 | 各尺度热图逆映射回原生网格 → `np.maximum` → 阈值 0.22 → 膨胀 9 → 8 连通域 → 面积 ≥ 4 |
| 排序 | 按连通域热图积分降序（与缓存脚本一致），截断到 100 |

候选写入 `prelabels/<seq_id>/`，**与人工标签 `labels/<seq_id>/` 完全分离**，重跑预标注不会覆盖任何人工修改。
YOLO 没有分数列，所以 worker 额外写 `prelabels/<seq_id>/scores.json` 存每个候选的峰值，供界面按阈值过滤。

## 提升标注效率的机制

视频标注的瓶颈不是画框，而是「在 16000 帧里找到还没看的帧」和「反复画同一个目标」。对应设计：

| 机制 | 作用 |
| :--- | :--- |
| **航迹传播 `P`** | 用上一帧测得的速度外推选中框 N 帧；遇到已有标注自动停住，一次请求批量写入 |
| **航迹插补 `I`** | 在当前帧与下一个已标注帧之间线性插值 |
| **接受候选 `空格`** | 一键采纳本帧模型候选 → 保存 → 自动前进 |
| **跳到未复核 `F`** | 沿时间线找下一个未复核帧，已完成的直接跳过 |
| **快进 `⇧A/⇧D`** | 10 帧步进，粗定位段落 |
| **候选阈值 `[` `]`** | 弱候选一键隐藏，只留可信目标 |
| **复核状态分离** | 只有显式操作才置「已复核」，导航不算 ⇒ 复核进度条是真实工作队列 |
| **时间线三色** | 蓝=模型候选，绿=已标注，深绿=已复核，一眼看出剩余工作量 |
| **帧预取 + 双 Canvas** | 图片只在换帧时重绘，拖拽框只重绘叠加层；滚动/切帧无等待 |
| **服务端 JPEG 环缓存** | 回退一帧是缓存命中（实测冷读 266 帧/s，命中瞬时） |
| **深链 `?seq=&f=`** | 把某一帧的争议直接用 URL 发给同事 |

## 快捷键

| 键 | 功能 | 键 | 功能 |
| :--- | :--- | :--- | :--- |
| `A`/`D` `←`/`→` | 上/下一帧（`⇧` 为 10 帧） | `N` | 新建框 |
| `空格` | 接受候选并前进 | `Del` | 删除选中框 |
| `Enter` | 保存并前进 | `P` / `I` | 传播 / 插补 |
| `F` | 跳到未复核帧 | `C` | 复制上一帧 |
| `Tab` | 显隐候选 | `[` `]` | 候选阈值 |
| `Home`/`End` | 首/末帧 | `0` `+` `-` | 适应 / 缩放 |
| `Ctrl+Z`/`Ctrl+Y` | 撤销 / 重做 | `Ctrl+S` | 强制保存 |

鼠标：拖拽空白处新建（双击进入绘制模式）、拖框体移动、拖 8 个控制点缩放、点框选中。
`Ctrl`/`Shift` + 滚轮缩放，普通滚轮平移。

## 浏览器交互自测

服务端与推理管线的测试覆盖不到「鼠标/键盘交互」这一层，所以另有一个在**真实浏览器里跑**的交互测试：
`static/uitest.html` 用同源 iframe 载入真实应用，通过 `window.__annotate` 句柄驱动它，
再用 `chrome --headless --dump-dom` 取回结果。

```bash
python -m manu.annotate.server --root <视频目录> --workspace <一次性工作区> --port 8793 &
google-chrome --headless=new --disable-gpu --no-sandbox --window-size=1500,900 \
  --virtual-time-budget=60000 --dump-dom \
  "http://127.0.0.1:8793/static/uitest.html?seq=<seq_id>&f=<frame>" 2>/dev/null \
  | sed -n '/TOTAL/,/<\/pre>/p' | sed 's/<[^>]*>//g'
```

> ⚠️ **务必给它一个一次性 `--workspace`。** 这个测试会真的走保存、传播、插补和候选采纳，
> 在正式工作区上跑会写进十几帧假的标注。测完把该工作区删掉即可。

> ⚠️ **序列至少 61 帧。** 导航用例断言的是第 20 帧附近的绝对下标，`C` 复制用例要访问到第 60 帧
> 才能验证「上一帧不在内存缓存里」这条真实路径。序列太短时这些下标会被夹取，失败信息看起来
> 像应用 bug，其实是测试前提不成立。现在开头会有一条显式长度检查，不满足就直接给出原因。

用例会根据候选是否带分数自动分支，两种来源都要覆盖到：

- **worker 自己产出的 `prelabels/<seq>/`**：有 `scores.json`，测试验证按分数做阈值过滤；
- **外部只读挂载的树**（如 `preannot_v1`）：只有纯 YOLO 文本、没有分数，UI 按兜底分 1 处理，
  测试验证「任何阈值都不过滤」这一设计行为。

覆盖 65 项：新建/规范化/删除框、命中测试与 8 个控制点、撤销重做、候选阈值过滤、
**接受候选→落盘→前进**、保存读回、航迹传播、插补、复制上帧、帧跳转与越界夹取、缩放与适应窗口、
跳未复核帧，以及**真实 KeyboardEvent 驱动的全部快捷键**（`A/D/←/→/⇧/Home/End/0/N/Esc/Delete/
Ctrl+Z/Ctrl+Y/Tab/[/]/+/-/C/F/空格`）。

🔴 **这一层抓到两个致命 bug**，单测与截图都发现不了：

1. **`acceptProposals()` 在干净帧上完全不落盘。** 它调 `saveFrame()` 时没带 `force`，
   而 `saveFrame()` 在 `dirty === false` 时**直接 return** ⇒ 在「本帧只有模型候选、标注员没改过」
   这一最常见场景下，**候选只留在内存，既不落盘也不前进，且无任何报错**。
   这正是「用 SOTA 预标注加速标注」的核心价值路径。已改为 `force: true`。
   共性教训：**「看起来设了 dirty」不等于「一定会走保存分支」**。
2. **`C`（复制上一帧）在跳帧后失效。** 它只查内存里的 `recentLabels`（只存本次访问过的帧），
   标注员跳到远处再按 `C`，即使上一帧**磁盘上有标签**也只会提示「上一帧没有框」。
   已改为缓存未命中时回源服务端读取。

## 导出成交付用的扁平 GT 布局

工具内部写 `labels/<seq_id>/<帧号>.txt`（按序列隔离，便于并发与分片），
但项目交付用的是**扁平** `{sequence}__{index:06d}.txt`、**每帧一个文件**（空文件表示无目标）——
这正是 `audit_substandard_cases.py`、`render_gt_osd_video.py`、`merge_patch_labels.py` 读的那套。
用导出命令转换：

```bash
# 全量导出（--root 只用来取帧数，把未标注帧补成空文件）
python -m manu.annotate.export_labels --workspace WS --out NEW_GT --root DATASET_ROOT

# 只改了某几个序列：以现有交付 GT 为底，补丁式合并（其余序列原样保留）
python -m manu.annotate.export_labels --workspace WS --out NEW_GT --base EXISTING_GT --link
```

- 导出目录**必须不存在**，否则直接报错退出——避免半份导出被当成完整交付。
- `--base` 模式下，工作区里有标注的帧覆盖 base，其余帧（含其他序列与 `classes.txt`）原样保留。
- `--link` 走硬链接复用未改动文件。⚠️ **硬链接要求 `--out` 与 `--base` 在同一文件系统**：
  跨文件系统会退化为复制（16.5 万文件约 2 分钟），不会报错，只是慢。

**在 GPU 服务器 `192.168.99.40` 上导出时，两条路径不在同一个盘**（实测）：

| 路径 | 设备 |
| :--- | :--- |
| `/tmp/pycharm_project_10ae9e2e/manu/annotate_workspace` | `/dev/nvme0n1p2` |
| `/mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010` | `/dev/sda` |

所以 `--out` 要放在 base 那一侧才吃得到硬链接：

```bash
cd /tmp/pycharm_project_10ae9e2e && ~/anaconda3/envs/uav/bin/python -m manu.annotate.export_labels \
  --workspace /tmp/pycharm_project_10ae9e2e/manu/annotate_workspace \
  --base    /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261010 \
  --out     /mnt/data/siping/datasets/manu/龙泉山/label/Final_Labels_merged_20261011 \
  --link
```

（在本机上跑则相反：base 走 SSHFS，把 `--out` 也放到 `/home/manu/mnt/data/...` 下即可。）

- 导出结果可被本工具当 `--prelabel-source` 直接读回（扁平布局已支持），无需再转换。

## 标签格式

标准 YOLO，每帧一个 `.txt`，`class cx cy w h` 归一化：

```
0 0.682460 0.010579 0.105299 0.026382
```

- `labels/<seq_id>/<帧号:06d>.txt` — 人工标注（唯一真值来源）
- `prelabels/<seq_id>/<帧号:06d>.txt` — 模型候选，永不被人工编辑覆盖
- `review/<seq_id>.json` — 复核状态

与现有 `龙泉山/label/Final_Labels_merged_20261010` 格式完全一致，解析器已实测可直接读取。

**出界框处理**：写入时按「与画面求交」裁剪，并丢弃**可见面积 < 10%** 的框。
只做裁剪会把飞离画面的目标变成焊在边框上的一条细边条，恰好污染航迹离开的那几帧；
完全在画面外的框会被直接丢弃。人工在界面里画的框本身已被限制在画面内，不受影响。

## 性能实测

| 项 | 实测 |
| :--- | :--- |
| 真实数据集扫描 | 24 序列 / 165,223 帧，约 1.5 s |
| 帧读取（SSHFS 冷读） | **265.8 帧/s**（142 KB/帧），缓存命中瞬时 |
| `summary()`（点一次序列就跑） | 12,460 帧规模 **3.31 s → 0.07 s**（见下） |
| 标签往返精度 | 像素级无损（`x1=100,x2=112` 精确还原） |

⚠️ `summary()` 的「总框数」需要读全部标签文件，是唯一随序列长度线性增长的部分，
12k 帧实测 3.3 s。因此**默认不算**，界面先渲染时间线、再在后台单独请求
（`/api/seq/<id>/summary?boxes=1`）。前提是 `format_yolo` 在写入时就折叠重复行，
使磁盘行数等于框数，无需重新解析。

## 复用已有预标注

若要的是「把现成 YOLO 结果搬进工作区」，上面的 `--prelabel-source` 是更好的方式（不复制、只读）。
确需物化一份副本时：

```bash
SRC=/home/manu/mnt/data/siping/datasets/manu/龙泉山/preannot_v1/labels/train
DST=/home/manu/mnt/pycharm_project_10ae9e2e/manu/annotate_workspace/prelabels
mkdir -p "$DST"
for d in "$SRC"/*/; do seq=$(basename "$d"); mkdir -p "$DST/$seq"; cp "$d"*.txt "$DST/$seq/" 2>/dev/null; done
```

## 自测

```bash
# 完整（含推理 worker 管线，需要 torch）
/home/manu/anaconda3/envs/uav/bin/python -m manu.annotate.selftest

# 仅服务端/标签/外部树（无需 torch）
/home/manu/anaconda3/bin/python -m manu.annotate.selftest
```

合成一段含双运动目标的视频 + 一个帧目录，跑通扫描 / 解码 / YOLO 往返 / 去重 / 出界裁剪与可见性丢弃 /
标签存储 / 复核状态 / 时间线 RLE / **预标注任务队列（提交→派发→完成、并发上限、远程命令内容）** /
外部只读候选树 / **推理 worker 管线** / HTTP 全接口。

**96 项断言**（uav 环境）/ 68 项（base 环境自动跳过 worker 段）。

其中 **推理 worker 管线**只把「模型前向」换成注入的合成热图，其余全部走真实代码：
逐帧解码、GMC + 中值特征、**冻结通道序（模型序 = 磁盘序反转）逐通道比对**、多尺度画布、
热图逆映射、融合阈值/膨胀/连通域、整段 `main()` 的分块与历史承接、YOLO 与 `scores.json` 落盘。
这样能在**不加载任何模型、不占 GPU** 的前提下证明这段代码真的能跑通。

⚠️ 三条方法论教训（都是这次写断言时踩的）：
1. `transpose(2,0,1)` 把 `(H,W,C)` 变成 `(C,H,W)` **并不转置 H/W**，参考面是 `a[:,:,k]` 而非 `a[:,:,k].T`；
   错误断言在平滑红外数据上**会偶然通过**（转置后恰好相等），只有随机值才暴露它。
2. 膨胀只长 mask 不长数值 ⇒ 4×4 的 0.9 热斑经 9×9 膨胀后**框是 12×12 但积分仍是 14.4**（16 个真像素 × 0.9）。
3. 断言写错时要先确认是**断言错**还是**代码错**——三次失败里有三次是我的断言写错，代码本身是对的。

## 已知边界

- `scan_root` 跳过名为 `labels` / `annotations` 的目录，避免把工作区当成序列扫进来；
  若原始数据里恰好有同名目录需改名。
- 视频随机大跳（> 240 帧）会触发一次 `CAP_PROP_POS_FRAMES` seek；
  少数容器不支持该 seek 时退化为从头 `grab` 前进，较慢但结果正确。
- 预标注要求源文件在服务器侧可见；纯本机视频可人工标注，但没有候选。
- 前后帧各自被裁成最小外接框，**不会**沿用上一帧的框尺寸——刻意选择，
  避免目标缓慢形变时尺寸被锁死在旧值上。
