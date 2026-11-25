# PointAnnotator — 目标中心点逐帧标注工具

桌面版（`tkinter` + `OpenCV`）中心点标注器，替代网页版 `manu/annotate`。

* 标注对象是**目标中心点**，不是检测框
* 帧状态是 **`pos` / `neg` / `ign`** 三态，而不是「有/无标签」
* 人工标注与模型预推坐标**双轨存储**，误操作后一键无损回退
* 一次改动一次落盘，崩溃不丢数据

---

## 1. 运行

```bash
# 依赖：Python 3.10+、opencv-python、numpy（tkinter 走标准库）
pip install opencv-python numpy

# 指定视频目录
python point_annotator.py /path/to/videos

# 不传目录 -> 弹出图形化文件夹选择框
python point_annotator.py

# 导入模型预推点作为底稿
python point_annotator.py /path/to/videos \
    --preannot-dir /path/to/preannot_rc2_tracked
```

支持的视频扩展名：`.mp4` `.avi` `.mkv` `.mov`（按文件名排序）。

### 命令行参数

| 参数 | 说明 |
| --- | --- |
| `video_dir` | 视频目录，可省略（省略则弹选择框） |
| `--preannot-dir` | 模型预推点目录，可重复给多个 |
| `--coord-space` | 跟踪记录 JSON 的坐标空间：`auto`（默认）/ `full` / `native` |
| `--native-size` | 模型原生网格 `W,H`，默认 `640,512` |
| `--crop` | 裁剪契约 `left,top,width,height`，默认 `320,0,1280,1024` |
| `--min-score` | 导入预推点时的置信度下限（读 `scores.json` 边车），默认 `0` = 全留 |
| `--self-test` | 跑无窗口自检后退出 |

---

## 2. 快捷键

| 按键 / 鼠标 | 功能 |
| --- | --- |
| **鼠标左键** | 把当前激活点移到光标处（该帧无点则新建首个目标点并置为 `pos`） |
| **`w` / `a` / `s` / `d`** | 激活点 上 / 左 / 下 / 右 微调 **1 px**（自动钳位在图像边界内） |
| **`W` / `A` / `S` / `D`** | 激活点步进 **5 px**（应对相机快速转动） |
| **`e` / `q`** | 下一帧 / 上一帧（按当前推进策略填充未编辑帧） |
| **`E` / `Q`**（Shift） | 快进 10 帧 / 后退 10 帧 |
| **`Space`** | 播放 / 暂停视频预览 |
| **`1`** | 采用上一帧标注（`t-1`） |
| **`2`** | 采用当前模型预推（`pred`）——**误操作的无损恢复入口** |
| **`Tab`** | 在 `t-1` 与 `pred` 之间一键切换 |
| **`n`** | 当前帧标记为 **硬负样本 `neg`** 并自动跳下一帧 |
| **`i`** | 当前帧标记为 **整帧忽略 `ign`** 并自动跳下一帧 |
| **`p`** | 在光标处新增一个目标点（多目标场景） |
| **`c`** | 删除当前激活点 |
| **`r`** | 切换默认推进模式（`Prev[1]` ↔ `Model[2]`） |
| **`t`** | 切换局部亮度重心吸附（默认关闭） |
| **`v`** | 循环切换显示滤镜（RAW → CLAHE → INFERNO） |
| **`[` / `]`** | 目录中的上一条 / 下一条视频 |
| **`h`** | 开关画面中央的快捷键说明浮层 |
| **`ESC`**（或 **`x`**） | 保存并退出程序 |

> **切换到英文输入法（`Switch to ENG IME`）**：界面右上角常驻提醒。单字母快捷键在中文输入法下会被吞掉。

### 双模式逐帧推进（`R` 键切换）

| 模式 | 按 `e` 推进到下一帧时，若该帧尚未人工编辑 | 适用 |
| --- | --- | --- |
| **`Prev[1]`**（默认） | 继承当前帧的 `pts` | 大目标防跳动、悬停片段 |
| **`Model[2]`** | 采用该帧的 `pred_pts` | 常规移动小目标 |

两种模式都只写 `pts`、不动 `pred_pts`，继承出来的帧 `edited` 仍为 `false`，所以继承链能一直延续。

### 质心吸附（`T` 键，默认关闭）

开启后左键点击会在 **7×7 邻域**内计算「局部背景减除后的亮度质心」并吸附过去；
按住 **`Alt` + 左键** 临时禁用吸附（不改变 `T` 的开关状态）。

> 默认关闭是有意的：避免大目标点击机身中心时被吸偏到单个电机高热点上。

---

## 3. 画面

* **同屏三层标记叠加**
  1. **上一帧确认位置**（`t-1` 残影）：暗灰十字 `#A0A0A0`
  2. **当前帧模型预推**（`pred_pts`）：青蓝空心圆 `#00C8FF`
  3. **当前生效标注**（`pts`）：亮绿圆环十字准星 `#00FF00`，当前激活点高亮
* **右上角 8× 像素级局部放大镜**：固定 `248 × 248` 窗口，取原图 `31 × 31` 邻域，
  `INTER_NEAREST` 保留像素网格；焦点优先级 = 激活标注点 → 模型预推点 → 鼠标位置。
  贴边时邻域自动裁剪并用边缘复制补齐，不会越界报错。
* **显示滤镜**（`v` 键）：原始画面 / CLAHE 局部直方图均衡 / Inferno 伪彩，
  **只影响显示**，不改原始像素、不改标注数据。
* **顶部 HUD**：视频序号/总数、帧号/总帧数、帧状态、推进模式、吸附开关、显示滤镜、
  **修饰键通道状态**（`MOD:X11` / `MOD:WIN32` / `MOD:NONE`），
  以及 `Switch to ENG IME` 提醒。
  修饰键通道读不到时（`MOD:NONE`）会额外显示红色 `! NO MODIFIER KEYS`——
  此时 `W/A/S/D`（5 px）与 `E/Q`（±10 帧）会退化成 1 px / 单帧。
* **鼠标坐标换算**：OpenCV 的 Qt / Win32 / GTK 后端在回调之前就已经把光标位置换算成
  **显示图像的像素坐标**（Qt 见 `window_QT.cpp` 的 `DefaultViewPort::icvmouseProcessing`：
  `floor(view_coord / ratioX)`，`ratioX = view_width / image_width`），
  所以回调拿到的 `(x, y)` **直接就是画布像素**，不能再按窗口尺寸二次缩放。
  画布 → 原图再经一次 letterbox 偏移换算（`_to_video`），落点即原图像素。

---

## 4. 数据格式

### 落盘位置

```
<视频目录>/annotations/<视频同名>.json
```

标注目录与视频目录同级，**原始视频目录不被污染**。

### JSON 结构

```json
{
  "video": "VIDEO00033_19700101_003501_osd.mp4",
  "resolution": [1920, 1080],
  "frames": {
    "142": {
      "state": "ign",
      "pts": [],
      "pred_pts": [[840, 412]],
      "edited": true
    },
    "310": {
      "state": "pos",
      "pts": [[845, 418]],
      "pred_pts": [[847, 420]],
      "edited": true
    },
    "311": {
      "state": "neg",
      "pts": [],
      "pred_pts": [[1120, 380]],
      "edited": true
    }
  }
}
```

| 字段 | 含义 |
| --- | --- |
| `state` | 帧三态：`pos` 有效正样本 / `neg` 人工核验的硬负样本 / `ign` 整帧忽略（**默认状态**） |
| `pts` | 当前生效标注的**中心点**（原图像素坐标） |
| `pred_pts` | 原始模型预推点，**只读不改**；误操作后按 `2` 键可无损恢复 |
| `edited` | 该帧是否被人动过。自动继承与模型底稿都保持 `false` |

写入是原子的（临时文件 + `os.replace`），断电/崩溃不会留下半截 JSON。

### 加载优先级（断点续标）

1. `annotations/<stem>.json` —— 本工具自己的产出，**优先**，用于断点续标；
2. `<视频目录>/<stem>.json` —— 模型预推理输出，作为初始底稿导入；
3. `--preannot-dir` 指向的目录。

载入后停在**第一帧还没被人动过**的位置（全部编辑过则停在末帧）。

### 支持的预推点来源

| 布局 | 目录形态 | 说明 |
| --- | --- | --- |
| **逐帧 YOLO 树** | `<root>/<sequence>/<frame:06d>.txt` + `scores.json` | 每行一个 YOLO 框（`class cx cy w h`，归一化），其**中心 × 原图宽高**即跟踪点 |
| **底稿 JSON** | `<root>/<sequence>.json` | 本工具自己的 JSON 格式：`pts` / `pred_pts` / `state` / `edited` **全部保留**（人工改过的帧以 `pts` 为准） |
| **跟踪记录 JSON** | `<root>/<sequence>.json` | `{"sequence", "per_frame": {"52": [{"pos": [x, y], "score": ...}]}}`，坐标在模型原生网格，按 `--crop` 契约换算回原图 |

`纵横AI数据采集_preannot_rc2_tracked` 属于第一种：目录名与**原始视频**同名
（`纵横AI数据采集_mp4/VIDEO00033_19700101_003501.mp4`），
树里是**归一化坐标 × 1920×1080 = 原始视频像素**（已在真实视频上核对：
树点与目标峰值平均只差 `(+3.7, +1.0) px`）。

> ⚠️ **预推点是「原始视频」坐标系**。OSD 版视频（`..._osd.mp4`）的画面被重新排版过
> （相机区 = 原图裁剪 `(320,0,1280,1024)` 缩放，右侧是放大镜面板），
> 树点画上去会偏 200～300 px。工具检测到这种情况会**在启动日志里告警**，
> 但**不做自动坐标变换**（各序列的 OSD 版式并不一致）。
> **要按预推点标注，请用原始视频目录**：
>
> ```bash
> python point_annotator.py /path/to/纵横AI数据采集_mp4 \
>     --preannot-dir /path/to/纵横AI数据采集_preannot_rc2_tracked
> ```

---

## 5. 自检

```bash
python point_annotator.py --self-test
```

覆盖规范点名的全部边界条件：第 0 帧按 `1`/`q`、末帧按 `e`、无预推数据时按 `2`、
坐标贴边时的放大镜跨界裁剪、微调钳位、每次状态变更的落盘与重载还原、
`2` 键无损恢复、两种推进策略、三态机、预推点导入与坐标换算、随机 seek 的像素级精确性、
鼠标坐标映射（回调坐标不得二次缩放、映射结果与窗口尺寸无关、黑边处加点被钳到图内）。

另外还覆盖：修饰键通道与 Shift 判定（X11 位图按 **keycode** 取位）、Alt 临时禁吸附、
底稿越界帧号丢弃与 `state`/`edited` 保留、`neg` 帧不会被自带点提升成 `pos`、
`.tmp` 残骸清理（只删 10 分钟前的，不误删别的实例正在写的）、
置信度过滤（空行不吃分数）、播放的片尾重播与手动换帧同步、
帧数未知时的换帧与越界裁剪、滤镜结果缓存、切视频时缓存失效、GTK 带修饰键键码的解码。

当前结果：**PASS 101 / FAIL 0**。

---

## 6. 在 Windows 上跑起来

### 6.1 方式 A：直接用 Python 跑（自己调试用）

1. 装 **Python 3.10 或更高**（python.org 的安装包，**务必勾选 `tcl/tk and IDLE`**，
   目录选择框 `tkinter.filedialog` 靠它；勾了才带 tkinter）
2. 装依赖（tkinter 走标准库，不用 pip）：

```bat
python -V
python -m pip install opencv-python numpy
```

3. 启动：

```bat
:: 指定视频目录
python point_annotator.py D:\videos

:: 带模型预推点（纵横数据）
python point_annotator.py D:\videos --preannot-dir D:\preannot_rc2_tracked

:: 不带目录参数 -> 弹 tkinter 文件夹选择框
python point_annotator.py
```

4. 先自检（**无窗口**，几秒就完，用来确认依赖和 tkinter 都正常）：

```bat
python point_annotator.py --self-test
```

### 6.2 方式 B：打包成 exe（发给标注员，目标机不需要 Python）

```bat
:: 在 Windows 上执行（需要联网装 PyInstaller）
build_exe.bat
```

产出 `dist\PointAnnotator\PointAnnotator.exe`。**整个 `dist\PointAnnotator` 目录**
（exe + `_internal\`）一起拷走，缺了 `_internal` 就起不来。

* 用 `--onedir` 而不是 `--onefile`：单文件模式每次启动都要解压整个运行时到临时目录，
  慢（十几秒）且**易被杀软误报**。
* 双击 exe 等价于「不带参数启动」⇒ 会弹目录选择框。
* 也可以 `PointAnnotator.exe D:\videos --preannot-dir D:\preannot_rc2_tracked` 直接带参启动。

### 6.3 方式 C：便携 Python + `run.bat`

```bat
run.bat D:\videos
```

`run.bat` 优先用同目录 `python\python.exe`，没有就退回 PATH 里的 `python`，两个都没有就报错退出。
**它依赖 Python，不是免安装方案**；`build_exe.bat` 也不会把它打进 `dist\`。

### 6.4 Windows 侧常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| **exe 双击后什么都没发生** | `--windowed` 打包**没有控制台**，出错也不会弹窗 | 看 exe 同目录的 `point_annotator.log`（启动信息、落盘失败、自检结果都在里面） |
| **exe 的自检看不到输出** | 同上，`--self-test` 的结果只写日志 | `type point_annotator.log` 看最后一行 `PASS n / FAIL m` |
| **没弹目录选择框** | 装 Python 时没勾 `tcl/tk`，tkinter 缺失 | 源码跑：重装 Python 并勾上 `tcl/tk and IDLE`；exe：**打包机上**也要有 tcl/tk，否则重新 `build_exe.bat`；或命令行直接传视频目录 |
| **杀毒软件报毒 / 拦启动** | PyInstaller 产物常见误报 | 把整个 `dist\PointAnnotator` 目录加进 Windows Defender「排除项」 |
| **窗口模糊、字发虚** | 高 DPI 缩放 | 不影响功能；把窗口拉大或改显示缩放比例 |
| **按键没反应** | 中文输入法吃了单字母键 | 切英文输入法（界面右上角常驻 `Switch to ENG IME` 提醒） |
| **标注存不进去** | 视频目录只读 | 拷到可写目录；落盘失败**界面不报警**，只在 `point_annotator.log` 里留一行 |

### 6.5 Windows 上的后端差异（实测结论）

* 官方 `opencv-python` 的 Windows wheel 走 **Win32 后端**（`WM_CHAR` + `VK<<16`），
  方向键等特殊键不会误触发微调，**也没有** Qt 后端「F 键和字母键同码」的歧义。
* Win32 后端的 `WM_CHAR` **带大小写** ⇒ Caps Lock 开着时 `e` 会变成 `E`。
  正常情况（`MOD:WIN32`）Shift 读的是真实按键状态，Caps Lock 无影响；
  只有在 `MOD:NONE`（读不到修饰键）时才会退化成「按字符大小写判 Shift」。

### 6.6 🔴 `.bat` 文件的编码纪律（Windows 上真踩过的坑）

**规则：`.bat` 一律写成「纯 ASCII + CRLF」，中文提示一律放到独立的 UTF-8 文本里用 `type` 显示。**

原因（2026-11-18 用户在 Windows 上实测报 `'锛?echo' 不是内部或外部命令`）：

1. **cmd.exe 按 ANSI（中文系统是 GBK/936）读 `.bat`**，不是 UTF-8 ⇒ 批处理里
   直接写中文，`echo` 会被前面的中文字节截断成乱码命令。
2. **`chcp 65001` 之后 cmd 按「字节偏移」定位自己**，而 UTF-8 中文是多字节
   ⇒ 代码页切换后文件位置错位，后面的行被切成碎片
   （实测表现：`'t_annotator.py' 不是内部或外部命令`、`'k' 不是内部或外部命令`）。
3. **`&&` 写在 `if ... ( )` 括号块里**会被判成语法错（`此时不应有 &&`）⇒
   一律改成 `where xxx >nul 2>nul` + `if not errorlevel 1 goto :label` 的直线写法。
4. **LF 换行**在带 `goto` 的批处理里也不保险 ⇒ 一律 CRLF。

因此本仓库的 `build_exe.bat` / `run.bat` 已改成纯 ASCII + CRLF，
中文提示改由 `msg_cn.txt`（UTF-8 无 BOM）配合 `chcp 65001` + `type` 输出。
**改这两个 `.bat` 时请沿用该纪律。**

---

## 7. 把交付包拆成 N 份（发给多个标注员）

`split_delivery.py` 把一份 Windows 交付包按「等量」拆成 N 份，每份**自包含**
（工具 + 免装 Python 的 exe + 课件 + Python 安装包 + 自己的视频与预标注），
发给不同的标注员并行标注。

```bash
# 只看分配方案，不落盘
python3 manu/point_annotator/split_delivery.py \
    --master <母包目录> --out-dir <输出目录> --parts 3 --plan

# 真正生成
python3 manu/point_annotator/split_delivery.py \
    --master <母包目录> --out-dir <输出目录> --parts 3

# 验收（结构 / 配对 / 逐字节 md5 / 等量 / 无重复无遗漏）
python3 manu/point_annotator/verify_split.py \
    --master <母包目录> --parts-dir <输出目录>
```

### 「等量」按什么算

**不是按条数**（视频长短能差十几倍），**也不只是按总帧数**
（预标注覆盖率可能从 0% 到 90% 不等 ⇒ 没底稿的帧要手工落点，有底稿的帧只需核对）。
本工具同时对齐三个指标：

| 指标 | 含义 |
| --- | --- |
| 总帧数 | 每帧都要过一遍的总量 |
| 手工帧数 | 视频帧数 − 预标注覆盖帧数 = 必须人工落点的帧 |
| 工作量当量 | 手工帧 × 1.0 + 有底稿帧 × 0.35（核对/微调的折算） |

分配算法 = **LPT 播种**（工作量降序进当前最轻的一份）+ **局部搜索**
（单条挪动 / 两条对调，直到三份的最大相对偏差最小）。

### 产物

* 每份一个目录 `<前缀>N/`，含母包的全部骨架（`.bat` / `msg_cn.txt` / `README_先看这个.md` /
  `tool/` / `docs/` / `python-3.10.0-amd64.exe`），**按份改写** `README` 与 `msg_cn.txt`；
* 每份的 `videos/annotations/` 建**空目录** ⇒ 标注员直接从第一条开始标；
* **不带**：母包的试跑草稿、`tool/build/`（PyInstaller 中间态）、旧 `*.log`；
* 每份一份 `_split_manifest.md5`（逐文件 md5）⇒ `md5sum -c` 自校验；
* 输出目录根有一份《交付说明.md》（清单、等量数据、发放与回收流程、复验命令）。

### 两个坑

1. **帧数别用 `ffprobe -count_frames`**（逐帧解码，几百条视频要十几分钟）⇒
   用容器表里的 `nb_frames`（毫秒级）；
2. **`.bat` 是 CRLF + 纯 ASCII**（见 §6）⇒ 改写时**按行拆（保留行尾）再插回**，
   按 `\n` 找锚点必然落空。
