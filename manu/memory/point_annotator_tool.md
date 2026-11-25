# 目标中心点标注工具 PointAnnotator（桌面版）

> 2026-11-16 新增。**替代网页版 `manu/annotate`** 做「目标中心点」标注：标注对象从检测框换成中心点，
> 帧状态从「有/无标签」换成 `pos / neg / ign` 三态，人工标注与模型预推坐标**双轨存储**。
> 交付物在 `manu/point_annotator/`（`point_annotator.py` + `build_exe.bat` + `run.bat` + `README.md`）。

## 0. 一句话结论

工具已建成，**自检 59/59 通过**，并已在**真实纵横数据**上端到端验证：
从 `纵横AI数据采集_preannot_rc2_tracked` 读入 **1068 帧**预推点，与磁盘 YOLO 树**逐帧核对 0 处不一致**，
叠加层渲染经像素级抽查确认（青蓝预推圈 / 亮绿生效点 / 暗灰 t-1 残影 三层同时在场）。

---

## 1. 需求来源与定位

用户给的规格书要求：Python 3.10+ / opencv-python / numpy / 标准库 tkinter+json+pathlib，
Ubuntu 与 Windows **零差异运行**，Windows 侧用 PyInstaller `--onedir --windowed` 分发。

⇒ 与网页版的分工：

| | 网页版 `manu/annotate` | 桌面版 `PointAnnotator` |
| :--- | :--- | :--- |
| 标注对象 | 检测框（YOLO 5 列） | **中心点** `[[x, y], ...]` |
| 帧状态 | 「有标签 / 无标签」+ 复核位 | **`pos` / `neg` / `ign` 三态** |
| 候选与标注 | 候选树 `prelabels/` + 人工树 `labels/` 分开 | **同一条记录内 `pts` / `pred_pts` 双轨** |
| 分发 | 服务器上跑、浏览器访问 | **Windows 单机 exe**，拷贝即用 |

---

## 2. 架构（单文件内聚）

`manu/point_annotator/point_annotator.py`（单文件，约 1685 行）：

| 组件 | 职责 |
| :--- | :--- |
| `ModifierReader` | 读实时 Shift/Alt/Ctrl（ctypes 调 Win32 `GetAsyncKeyState` / Xlib `XQueryPointer`） |
| `AnnotationStore` | 逐帧记录 + 三态机 + 双轨坐标 + 序列化 |
| `save_json()` | 原子落盘（临时文件 + `os.replace`） |
| `PreAnnotationLoader` | 导入模型预推点（三种落盘布局） |
| `VideoReader` | 随机访问帧读取（seek 像素级精确） |
| `PointAnnotator` | 渲染（三层叠加 + 8× 放大镜 + HUD + 帮助浮层）+ 交互 + 落盘 |

**零新增第三方依赖**：`tkinter` 只用于「选择目录」弹窗和读屏幕尺寸；画面渲染走 OpenCV。

---

## 3. 数据契约

### 落盘位置

```
<视频目录>/annotations/<视频同名>.json
```

### JSON 结构（与规格书逐字一致）

```json
{
  "video": "VIDEO00033_19700101_003501_osd.mp4",
  "resolution": [1920, 1080],
  "frames": {
    "142": {"state": "ign", "pts": [], "pred_pts": [[840, 412]], "edited": true},
    "310": {"state": "pos", "pts": [[845, 418]], "pred_pts": [[847, 420]], "edited": true},
    "311": {"state": "neg", "pts": [], "pred_pts": [[1120, 380]], "edited": true}
  }
}
```

* `state`：`pos` 有效正样本 / `neg` 人工核验的硬负样本 / `ign` 整帧忽略（**默认状态**）
* `pts`：当前生效标注（中心点，原图像素）
* `pred_pts`：原始模型预推点，**只读不改** ⇒ 误操作后按 `2` 键**无损恢复**
* `edited`：该帧是否被人动过（自动继承 / 模型底稿都保持 `false`）

### 三条自定的语义（规格书未写死，这里定死）

1. **点被清空 ⇒ 状态回 `ign`，不自动升 `neg`**。`neg` 的语义是「人工核验过这是硬负样本」，
   删除一个点不构成这个断言 ⇒ `neg` **只由 `n` 键产生**。
2. **继承只在当前帧有点时发生**。当前帧是 `neg`/`ign`（无点）时，下一帧保持自己的模型底稿，
   否则会把下一帧的模型预推点抹成空。
3. **模型底稿导入 ⇒ `pts` 与 `pred_pts` 同时写入，`edited=False`**。
   这样「空格/方向键推进」不会把底稿抹掉，而 `2` 键随时能拉回模型原始落点。

---

## 4. 预推点导入：规格书与磁盘现实的差异

规格书写的是「检查视频同级目录下的 `<stem>.json`」。**磁盘上实际有三种形态**，工具全部支持：

| 布局 | 形态 | 坐标空间 |
| :--- | :--- | :--- |
| **逐帧 YOLO 树** | `<root>/<seq>/<frame:06d>.txt` + `scores.json` | **原图像素**（框中心即跟踪点） |
| **底稿 JSON** | `<root>/<seq>.json`，`{"video","resolution","frames":{...}}` | 原图像素 |
| **跟踪记录 JSON** | `<root>/<seq>.json`，`{"sequence","per_frame":{"52":[{"pos":[x,y],"score":...}]}}` | **模型原生网格 640×512** |

⇒ 跟踪记录走 `--crop 320,0,1280,1024` 契约换算：`full_x = 320 + native_x * 1280/640`。
**该换算已被两个独立产物交叉验证**：`preds_track/VIDEO00033….json` 的 `pos=[24.859, 200.265]`
⇒ 换算得 `(369.72, 400.53)`，与 `preannot_rc2_tracked/…/000052.txt` 的
`0 0.192563 0.370852 …` 反解出的 `(369.72, 400.52)` **一致到 0.01 px**。

`_osd` 后缀会自动去掉去匹配原始序列目录（OSD 视频 `…_osd.mp4` ↔ 原始序列 `…`）。

---

## 5. 三个必须记住的坑（本轮实测踩出来的）

### 5.1 🔴 `XQueryPointer` 段错误的真凶是**探针自己**，不是 X11

第一次自检直接 **SIGSEGV**（`EXIT=139`），栈顶在 `ModifierReader._x11_mask`。
当时判定成「X11 查询在受限环境里不安全」，于是加了子进程探针 —— 结果**整条 X11 路线被静默关掉**，
Ubuntu 上 `W/A/S/D`（5 px）与 `E/Q`（±10 帧）全部失效，HUD 也不提示，**属于静默功能缺失**。

第二轮复查（用真实出参重跑探针）结论反了：

```
XQueryPointer(display, win, root_ret, child_ret, root_x, root_y, win_x, win_y, mask_ret)
                                          ↑ 6 个出参，全是「必写」的指针
原探针：lib.XQueryPointer(d, r, None, None, None, None, None, None, byref(m))
                                        ↑ 传 NULL ⇒ Xlib 往空指针里写 ⇒ SIGSEGV
新探针：给全部出参传真实缓冲区 ⇒ ok=1, pos=(1280,729)，不崩
```

⇒ **不是 X11 不能用，是探针把出参写成了 NULL**。

修法（两处一起改）：

1. **改用 `XQueryKeymap`**：它只要求调用方给一个 32 字节缓冲区，没有「必写出参」这种坑；
2. 探针同步换成 `XQueryKeymap` + `create_string_buffer(32)`。

> 沉淀：**探针本身也会骗人**。子进程探针只能证明「我写的调用方式」安全，
> 不能证明「这个 API 不安全」——被探针判死的功能要先查探针，再查 API。
> 另外：**功能静默失效比崩溃更危险**，修饰键通道起不来时 HUD 必须显式告警
> （现在 HUD 右上有 `! NO MODIFIER KEYS` 红字，状态位显示 `MOD:X11` / `MOD:WIN32` / `MOD:NONE`）。

#### 🔴🔴 第三轮才发现：`XQueryKeymap` 的位图是**按 keycode 索引**的

改用 `XQueryKeymap` 之后，第三轮审计又抓到：**「bit0=Shift」这个理解是错的**。
X11 协议原文（QueryKeymap）：

> "Byte N (from 0) contains the bits for keys **8N to 8N + 7** with the least significant bit
> in the byte representing key 8N."

⇒ 位图按 **keycode** 索引，不是修饰键掩码；而 X 的 keycode **从 8 开始**
（本机实测 `XDisplayKeycodes` → `min_keycode = 8`）⇒ **第 0 字节恒为 0x00**。
按「第 0 字节 bit0=Shift」去读 ⇒ 整条通道**永远返回「无修饰键」**，
但 HUD 还显示 `MOD:X11`（通道"在"却读不到东西）⇒ 又是一次**静默失效**。

实测（本机 :0，XTest 合成按下 Shift_L）：

```
min_keycode = 8
Shift_L: keysym=0xFFE1 -> keycode=50 (byte 6 bit 2)
按下 Shift_L 后位图非零字节: {6: '0x04'}     # byte6 bit2 = keycode 50 ✓
按「第 0 字节当掩码」读: shift=False         # ✗ 永远读不到
按 keycode 读:            shift=True          # ✓
```

正确修法（**修饰键 keycode 由键盘布局决定，不能写死**）：

```python
lib.XKeysymToKeycode.restype = ctypes.c_ubyte
code = lib.XKeysymToKeycode(display, 0xFFE1)   # Shift_L
pressed = raw[code // 8] >> (code % 8) & 1      # byte=keycode//8, bit=keycode%8
```

Shift_L/R=0xFFE1/0xFFE2、Control_L/R=0xFFE3/0xFFE4、Alt_L/R=0xFFE9/0xFFEA。
改完用 XTest 端到端复核（合成按/放三个修饰键，`state()` 三个都跟着变）：

```
Shift_L(keycode 50):    按下时 state=(True, False, False)   松开后 (False, False, False)
Control_L(keycode 37):  按下时 state=(False, False, True)   松开后 (False, False, False)
Alt_L(keycode 64):      按下时 state=(False, True, False)   松开后 (False, False, False)
```

自检里也补了**变异可红**的回归项：造一张「Shift 键码置位、第 0 字节为 0」的位图塞进
`XQueryKeymap`，断言 `state()` 必须读出 Shift ⇒ 把实现改回「第 0 字节当掩码」立刻 FAIL。

> 沉淀：**X11 协议文档要读原文，不能凭印象**——
> 「`XQueryKeymap` 返回修饰键掩码」是个流传很广的错误理解（那是 `XQueryPointer` 的 `mask`）。

### 5.2 🔴 OpenCV 各后端的键码约定不同，方向键会误触发微调

读 `opencv/modules/highgui/src/window_Qt.cpp` 与 `window_w32.cpp` 的源码确认：

* **Windows**：普通键走 `WM_CHAR` ⇒ ASCII（Shift+e ⇒ `'E'`）；方向键 / F 键 / Home / End 走
  `keyCode = (int)(message.wParam << 16)` ⇒ **低字节是 0**。
* **Qt**（`window_QT.cpp` `CvWindow::keyPressEvent`）：可翻译键走 `QTest::keyToAscii` ⇒ ASCII
  ——**不带修饰键**（实测 `keyToAscii(Qt::Key_A) == 97`，Shift+w 与 w 同码）；
  方向键走 `evnt->nativeVirtualKey()`（X11 keysym，`0xff51` 之类）。
* **GTK**（`window_gtk.cpp` `icvOnKeyPress`）：`code |= event->state << 16` ⇒
  **低 16 位是键值，高 16 位是 GDK 修饰键掩码**（Shift=0x1 / Ctrl=0x4 / Alt=0x8）；
  特殊键的 keyval `>= 0xff00`。

⇒ 统一在 `_decode_key()` 归一：

| 判据 | 归属 | 处理 |
| :--- | :--- | :--- |
| `raw >= 0x01000000` | Qt 原始键码 | Esc / Tab 认，其余挡掉 |
| `raw >= 0x10000` 且低 16 位 == 0 | Windows `VK<<16` 特殊键 | 挡掉（方向键 / F 键） |
| `raw >= 0x10000` 且低 16 位 `>= 0xff00` | GDK / X11 特殊键 | 挡掉 |
| `raw >= 0x10000` 且低 16 位是 ASCII | **GTK 带修饰键的普通键** | 取低位当字符，**高位掩码当 Shift/Alt/Ctrl** |

⚠️ 旧实现把 `raw >= 0x10000` 一律挡掉 ⇒ **GTK 后端下 `Shift+w` 整个按键被丢弃**。
现在 GTK 的修饰键直接来自键码高位（比轮询更准），只有其余后端才回落到操作系统查询。
自检里有专门的回归项（`_decode_key` / `_key_modifiers` 各 6 条）。

### 5.3 ⚠️ 断言写错 vs 代码写错（又一次）

自测里「Model 模式不覆盖已有标注」这条一开始红了：断言查的是**第 8 帧**，而推进是 6→7，
**第 8 帧本来就没被碰过**。是断言错，不是代码错——与 `web_annotate_tool.md` §26.4 记的是同一类。

> 沉淀：**测试失败先判是断言错还是代码错**；连续两次「修复」后仍失败，就回头查测试前提。

### 5.4 🔴 OpenCV 的鼠标回调给的**已经是图像坐标**，别再二次缩放

初版 `handle_mouse` 写了这么一段：

```python
rect = cv2.getWindowImageRect(WINDOW)          # 想拿窗口尺寸算比例
canvas_x = x / (rect[2] / self.canvas_w)        # ← 错：这是二次换算
```

读 `opencv/modules/highgui/src/window_QT.cpp`（4.10.0）原文：

```cpp
void DefaultViewPort::icvmouseProcessing(QPointF pt, int cv_event, int flags)
{
    //to convert mouse coordinate
    qreal pfx, pfy;
    matrixWorld_inv.map(pt.x(),pt.y(),&pfx,&pfy);
    mouseCoordinate.rx()=floor(pfx/ratioX);      // ratioX = width() / image2Draw_mat->cols
    ...
}
```

⇒ **OpenCV 在回调之前就把光标位置换算成显示图像的像素坐标了**（Win32 / GTK 后端同理），
回调拿到的 `(x, y)` 直接就是画布像素。而 `draw2D` 把图像拉伸铺满 viewport ⇒ 两边自洽。

本机实测反例：画布 1280×720，Qt 视口只有 **1175×661**（HiDPI 缩放），
`cv2.getWindowImageRect` 返回的也是这个**全局屏幕矩形** `(693,399,1175,661)`——
拿它的宽高当缩放因子 ⇒ **约 9% 的系统性偏差**（点 500 实际落到 544）。

⇒ 修法：`canvas_x, canvas_y = float(x), float(y)`，直接用；画布→原图的 letterbox 偏移
由 `_to_video()` 单独负责。**坐标链一共三层，别搞混**：

```
OpenCV 回调(画布像素) → _to_video()(letterbox 偏移) → 原图像素
                                                    ↘ --crop 契约(native→full)
```

> 沉淀：**GUI 库里凡是「回调坐标」的语义，先读后端源码再动手**——
> 直觉（回调给的是窗口坐标）在这套 API 上是错的，而且错得很隐蔽（差 9%）。

### 5.5 ✅ ESC 在 Qt 后端**不会** abort（用 ctypes 直接调 Qt5Test 判决）

担心 `QTest::keyToAscii(Qt::Key_Escape)` 在 wheel 自带的 Qt5Test 里 SIGABRT，
于是用 `ctypes` 直接调库里的符号判决（**子进程里调**，崩了也不影响主进程）：

```python
lib = ctypes.CDLL(".../opencv_contrib_python.libs/libQt5Test-d435aae7.so.5.15.13")
fn = getattr(lib, "_ZN14QtOpenCVPython5QTest10keyToAsciiENS_2Qt3KeyE")   # 注意去掉 @@Qt_5
fn.restype, fn.argtypes = ctypes.c_int, [ctypes.c_int]
```

| 键 | 返回 | 退出码 |
| :--- | :--- | :--- |
| `Qt::Key_Escape` (0x01000000) | **27** | 0 |
| `Qt::Key_Tab` (0x01000001) | 9 | 0 |
| `Qt::Key_Space` (0x20) | 32 | 0 |
| `Qt::Key_A` (0x41) | **97**（小写！） | 0 |

⇒ **ESC 安全**（`isTranslatableKey()` 把 Esc 归为可翻译键 ⇒ 走 `keyToAscii` ⇒ 27），
后端也**不会**把 ESC 吞掉。同时这一栏再次证明 **Qt 的 `keyToAscii` 不带修饰键**
（`Key_A` 恒为小写 97 ⇒ `Shift+w` 与 `w` 在 Qt 后端不可区分 ⇒ 必须另找修饰键通道）。

两个附带收获：
* `dlsym` 要用**不带 `@@Qt_5` 版本后缀**的符号名，带了会 `AttributeError: undefined symbol`；
* ESC 之外仍加了一个 `x` 键作为退出备份（后端吞键时还能退出）。

### 5.6 🔴 预推点坐标是**原始视频**的坐标系，OSD 版视频对不上

`纵横AI数据采集_preannot_rc2_tracked` 的 YOLO 树里是**归一化**坐标，
乘原图宽高（1920×1080）就是原图像素。用**原始视频** `纵横AI数据采集_mp4/VIDEO00033_….mp4`
实测：树点与该帧的局部对比度峰**平均只差 (+3.7, +1.0) px**（最大 13 px，纯跟踪抖动）
⇒ **树 = 原始视频坐标系** 确认。

但 OSD 版视频 `纵横AI数据采集_osd_rc2_tracked_th020_20261110/…_osd.mp4` 的画面是
**重新排版**过的：相机区只占左边约 1300 px，右边是放大镜面板。
用相位相关（`cv2.phaseCorrelate`）对齐实测：

| 假设 | 残差 | 响应 |
| :--- | :--- | :--- |
| OSD 相机区 = 原图裁剪 `(320,0,1280,1024)` 拉到 1300×1080 | **(1.5,1.0) ～ (5,4) px** | 0.14 ～ 0.55 |
| OSD 相机区 = 原图整幅拉到 1300×1080 | (−375, 262) ～ (98, 47) px | ≈ 0 |

⇒ OSD 相机区 = 原图**裁剪 (320,0,1280,1024)** 缩放而成（正好是模型的 crop 契约）。
**树点是原图坐标，直接画到 OSD 视频上会偏 200～300 px**（绿点落在空天上，目标在左下角）。

⚠️ 而且**这个版式不是全序列一致**：`VIDEO00028` 用同一假设对齐残差高达 (81, 268)，
`VIDEO00045` / `VIDEO00069` 才是一致的那套 ⇒ **不能写死一个 OSD 坐标变换**。

⇒ 定死两条规矩：

1. **要按预推点标注，就用原始视频目录**（`纵横AI数据采集_mp4`，与树目录**同名**，无需别名匹配）；
2. 工具匹配到 **`_osd` 别名**（树目录名 = 视频名去掉 `_osd`）时，**显式告警**：
   「底稿来自原始视频坐标系，而当前放的是 OSD 版视频：OSD 画面的相机区是原图裁剪缩放过的，
   预推点不会落在目标上；要按预推点标注，请改用原始视频目录」。

> 沉淀：**坐标契约要用「另一个独立产物」交叉验证**——
> 树（归一化）与 `preds_track/*.json`（模型原生网格 640×512）
> 经 `--crop 320,0,1280,1024` 换算后**逐像素相等**，才敢认定坐标系。

### 5.7 🔴 给渲染结果加缓存 ⇒ 缓存键必须包含「数据来源的世代」

CLAHE 在 1080p 上单次 ~32 ms，主循环每 15 ms 重绘一次 ⇒ 不缓存会卡成幻灯片，
于是加了 `_filtered_frame()` 的 `(帧号, 滤镜)` 缓存。

**结果换视频时踩了坑**：新视频的续标帧号常常也是 0 ⇒ 缓存键 `(0, 1)` 与旧视频完全相同
⇒ **切视频后画布上还是上一条视频的帧**（自检里是渲染到别的帧，肉眼很难发现）。

修法：`_open_video()` 里显式把缓存打掉（`self._filtered_key = None`）。
回归项写成「切视频后 `before is not after`」——**实测把修复去掉后该项立刻 FAIL**，确认它真能兜住。

> 沉淀：**缓存键要能区分「同键不同源」**。凡是「换了个数据源但索引从 0 重新开始」的场景，
> 缓存一定要显式失效，并在自检里留一条能真的红掉的回归项。

---

## 6. 验证结果（全部实测）

### 6.1 自检 101/101

```bash
cd /media/manu/1TB-Volume/workspace/ultralytics/manu/point_annotator
/home/manu/anaconda3/envs/uav/bin/python point_annotator.py --self-test
# [self-test] PASS 101  FAIL 0
```

覆盖规格书 §5 点名的全部边界条件 + 若干追加项：

| # | 用例 | 断言 |
| :--- | :--- | :--- |
| 1 | 第 0 帧按 `1` / `q` / `2` | 不抛异常、不越界 |
| 2 | 末帧按 `e` / `E` | 不前进 |
| 3 | 无预推数据时按 `2` | 不炸、`pred_pts` 保持为空 |
| 4 | 坐标贴四角 | 放大镜取样不跨界 |
| 5 | 贴边微调 | 被钳位在图像边界内 |
| 6 | 状态变更落盘 + 重载 | `pts`/`pred_pts`/`state`/`edited` 全部还原 |
| 7 | 重开视频 | 从 `annotations/<stem>.json` 恢复 |
| 8 | 误操作后按 `2` | **无损恢复模型预推点**，`pred_pts` 未被改动 |
| 9 | 两种推进策略 | Prev 继承 / Model 采用预推 / 都不动 `pred_pts` / 继承帧仍 `edited=False` |
| 10 | 三态机 | `n`→`neg`、`i`→`ign`、点空→`ign` 而非 `neg` |
| 11 | 预推点导入 | YOLO 树中心换算、跟踪 JSON 的 native→full 换算 |
| 12 | 随机 seek | 与逐帧解码**像素级一致** |
| 13 | 三种显示滤镜 + 放大镜渲染 | 均不抛异常 |
| 14 | 键码归一 | ESC/Tab/Space/方向键分类正确，方向键不误触发微调 |
| 15 | 鼠标坐标映射 | 回调坐标原样当画布像素；画布↔原图往返落点误差 ≤ 1 px |
| 16 | 映射与窗口尺寸无关 | 篡改 `getWindowImageRect` 返回值后映射结果不变（防二次缩放回归） |
| 17 | 黑边处加点 | `p` 键在画面黑边上新增的点被钳到图像边界内 |
| 18 | 修饰键通道 | 状态返回 `(shift, alt, ctrl)`；X11 环境下通道**可用**；Shift 判定「OS 优先、字符大小写兜底」 |
| 19 | Alt 临时禁吸附 | 鼠标事件带 `CV_EVENT_FLAG_ALTKEY` 时不吸附，不带时照常吸附 |
| 20 | 底稿导入 | 越界帧号**不写进标注**；底稿的 `neg` 状态与 `edited` 标记**被保留**；`.tmp` 残骸被清理 |
| 21 | 置信度过滤 | 低于 `--min-score` 的框被过滤，达到下限的框保留 |
| 22 | 播放 | 片尾再按空格**从头播**；手动换帧后播放原点跟着走 |
| 23 | 帧数未知 | `CAP_PROP_FRAME_COUNT<=0` 时靠解码判断边界，仍能前进 |
| 24 | 滤镜缓存 | 同一帧的滤镜结果**只算一次**（CLAHE 1080p 单次 ~32 ms，不缓存会卡成幻灯片） |
| 25 | GTK 键码 | `keyval \| (state<<16)` 的普通键不再被整体丢掉；GDK 特殊键仍挡掉；高位掩码里的 Shift/Alt/Ctrl 能读出 |
| 26 | 切视频 | 滤镜缓存的键 `(帧号, 滤镜)` 会与新视频撞车 ⇒ 切视频必须让缓存失效（回归项验证过：去掉修复即 FAIL） |
| 27 | `neg` 底稿 | 底稿里自带点的 `neg` 帧导入后**仍是 neg、点为空**（不会被 `set_pts` 提升成 pos） |
| 28 | 帧数未知 | 越界帧号照样丢弃（二分探测真实帧数）；片尾再按空格也从头播 |
| 29 | 置信度索引 | `.txt` 里的空行**不吃掉框的置信度**（分数按有效框序号对齐） |
| 30 | X11 位图 | 位图按 **keycode** 取位（造位图注入，验证 `state()` 读得出 Shift） |
| 31 | `.tmp` 清理 | 陈旧（>10 min）的残骸被删；**另一个实例正在写的新鲜 tmp 不被误删** |

### 6.2 真实数据端到端（纵横）

```bash
# 视频：纵横AI数据采集_osd_rc2_tracked_th020_20261110（58 条 _osd.mp4，1920×1080，3152 帧）
# 预推点：纵横AI数据采集_preannot_rc2_tracked（24/58 序列目录，每序列 1068 帧 .txt + scores.json）
```

* 读入 **1068 帧**预推点，帧号 52…3151；
* 与磁盘 YOLO 树**逐帧核对 1068/1068，0 处不一致**；
* 叠加层渲染像素级抽查：青蓝预推圈 20 px、亮绿生效点 223 px、暗灰残影 68 px ⇒ **三层同时在场**；
* 绿点质心 `(295.02, 263.97)` == 点位 `(443,396)` 的画布映射 `(295,264)` ⇒ **坐标映射零偏差**。

### 6.2b 鼠标映射链的端到端复核（同一份真实数据）

```
[1] 逐帧核对: 树上 1068 帧 / 工具载入 1068 帧 / 不一致 0 / 多余 0
[2] 叠加层像素: 青蓝预推圈 20 / 亮绿生效点 223 / 暗灰残影 68   ← 三层同时在场
[3] 绿点质心 (295.02, 263.97) vs 点位画布映射 (295, 264)        ← 零偏差
[4] 画布(295,264) → 鼠标原图坐标 (442.5, 396.0) vs 真值 [443, 396]
[5] 落盘 231 499 bytes / 1068 帧，顶层键 {frames, resolution, video}
```

### 6.2b2 全量 58 序列逐帧核对（预推点导入路径的端到端保证）

```
preannot 序列数：58
[OK ] VIDEO00028_19700101_002947        帧数=3069 树帧= 441 不一致=0 越界=0 别名匹配=False
[OK ] VIDEO00032_19700101_003358        帧数=3154 树帧=2000 不一致=0 越界=0 别名匹配=False
...（58 行全 OK）
[OK ] VIDEO00087_19700101_003101        帧数=1741 树帧=1741 不一致=0 越界=0 别名匹配=False

核对序列 58 条，异常 0 条 []
```

⇒ **58/58 序列**的预推点导入与磁盘 YOLO 树**逐帧完全一致**，无越界帧号；
树帧数从 19 到 3140 不等（有检出的帧才有 `.txt`），全部对上。
（用**原始视频**目录核对 ⇒ 视频名与树目录名**同名**，`_base_stem` 别名匹配不参与。）

### 6.2c 原始视频工作流复核（`纵横AI数据采集_mp4`，树点确实落在目标上）

```bash
# 视频：纵横AI数据采集_mp4/VIDEO00033_19700101_003501.mp4（1920×1080，3152 帧，与树目录同名）
# 预推点：纵横AI数据采集_preannot_rc2_tracked/VIDEO00033_19700101_003501
```

| 帧 | 树点（原图像素） | 该帧局部对比度峰 | 偏移 |
| :--- | :--- | :--- | :--- |
| 52 | (370, 401) | (383, 402) | (+13, +1) |
| 55 | (443, 396) | (449, 399) | (+6, +3) |
| 60 | (549, 389) | (552, 389) | (+3, 0) |
| 120 | (1592, 355) | (1599, 353) | (+7, −2) |
| 3000 | (702, 249) | (703, 252) | (+1, +3) |

**平均偏移 (+3.7, +1.0) px** ⇒ 树点就是目标中心（残差是跟踪抖动，不是坐标错位）。
```

### 6.3 落盘产物结构（真实跑一遍）

```
annotations/VIDEO00033_19700101_003501_osd.json   231 386 bytes / 1068 帧
  "52": {"state":"neg","pts":[],"pred_pts":[[370,401]],"edited":true}
  "53": {"state":"ign","pts":[],"pred_pts":[[401,399]],"edited":true}
  "54": {"state":"pos","pts":[[424,398]],"pred_pts":[[424,398]],"edited":false}
```

---

## 7. 复现命令

```bash
# 1) 自检（无窗口，任何环境可跑）
cd /media/manu/1TB-Volume/workspace/ultralytics/manu/point_annotator
/home/manu/anaconda3/envs/uav/bin/python point_annotator.py --self-test

# 2) 本机 → 服务器镜像同步（镜像在本机是**只读**挂载，cp 必须由用户在沙箱外执行）
cp -r /media/manu/1TB-Volume/workspace/ultralytics/manu/point_annotator \
      /home/manu/mnt/pycharm_project_10ae9e2e/manu/
md5sum manu/point_annotator/point_annotator.py            # 两侧比对 md5

# 3) 真实数据（服务器上跑）
ssh -p 32222 huangzhe@192.168.99.40
source ~/anaconda3/etc/profile.d/conda.sh && conda activate uav
cd /tmp/pycharm_project_10ae9e2e
python manu/point_annotator/point_annotator.py \
    /mnt/data/siping/datasets/manu/纵横AI数据采集_osd_rc2_tracked_th020_20261110 \
    --preannot-dir /mnt/data/siping/datasets/manu/纵横AI数据采集_preannot_rc2_tracked
```

> ⚠️ 本工具是**桌面单机程序**，没有 `--host` / `--port` 之类的服务参数（那是从网页版串过来的），
> CLI 只有：`video_dir` `--preannot-dir` `--coord-space` `--native-size` `--crop`
> `--min-score` `--self-test`。

> ⚠️ 标注结果写进 **视频目录** 的 `annotations/` 子目录 ⇒ 跑在服务器上时会把标注写到
> `/mnt/data/...`（**服务器数据集目录，默认只读**）⇒ **要么把视频目录挂到可写区**，
> 要么先 `cp` 一份 OSD 视频到可写目录再标。**别直接往 `/mnt/data` 写标注。**

---

## 8. 遗留与边界

1. **Alt 临时禁用吸附**：优先用鼠标事件自带的 `CV_EVENT_FLAG_ALTKEY`（三个后端都会填），
   拿不到才回落到轮询；X11 通道现在走 `XQueryKeymap`（已实测可用）。
2. **Shift 检测**：Linux 走 `XQueryKeymap`、Windows 走 `GetAsyncKeyState`；
   两者都拿不到时才靠字符大小写（Caps Lock 开启时可能把 1 px 判成 5 px）。
3. **放大镜跟随的是「激活点 → 预推点 → 鼠标」的焦点优先级**；没有任何焦点时不画放大镜。
4. **播放模式不做帧填充也不落盘**（纯预览）。
5. **`W`/`A`/`S`/`D` 的 5px 步进不跨越图像边界**（与 1px 同样钳位）。
6. **Windows 打包未在本机实测**（本机无 PyInstaller / 无 Windows）；`.bat` 已按 CRLF + UTF-8
   （`chcp 65001`）写，`--onedir` 非 `--onefile`。
   `--windowed` 打包后 `sys.stdout is None` ⇒ 所有消息统一走 `log()`（stdout + exe 同目录
   `point_annotator.log`），别直接 `print`。
7. **OSD 版视频的坐标错位已确认但没做自动变换**（版式非全序列一致，见 §5.6）⇒
   工具只**告警**，不做硬编码映射；按预推点标注请用原始视频目录。
8. **播放到片尾后按空格会从头播**；播放中手动换帧会把播放原点重同步到当前帧（不再被拉回）。
9. **Qt 后端跑在 Windows 上时，F 键和字母键无法区分**：
   `nativeVirtualKey()` 在 Windows 上返回 VK 码（F1=`0x70`='p'、F9=`0x78`='x'），
   与 `keyToAscii` 的字母同码 ⇒ 按 F1 会触发 `p`、按 F9 会触发 `x`。
   官方 `opencv-python` 的 Windows wheel 用的是 **Win32 后端**（`WM_CHAR` + `VK<<16`，无此歧义），
   只有自编译带 Qt 的 Windows 版才会遇到；Linux(X11) 下 nativeVirtualKey 返回 keysym（`0xff51`），无此问题。

---

## 9. 标注员培训 PPT 分页脚本（2026-11-16 追加）

| 文件 | 说明 |
| :--- | :--- |
| `manu/point_annotator/标注培训PPT_分页内容.md` | **15 页精简版**培训课件（正式交付）：每页只有「页面内容 / ⚠️ 注意 / 现场演示」，**只讲操作与快捷键** |
| `manu/point_annotator/标注培训PPT_分页内容_详版23页.md` | 详版讲稿稿（**留档，不上 PPT**）：23 页，每页含「页面内容 / 讲稿 / 现场演示 / 配图建议」 |

* 面向**标注员**（无算法背景），讲解 35 分钟 + 实操 10 分钟。
* **用户 2026-11-16 明确要求：课件最多 15 页、聚焦操作与快捷键，原理/数据格式/目录结构这些前因后果由培训时口头补充，不进 PPT** ⇒ 正式版按此重写（原 23 页版降级为留档详版）。
* 内容全部按**源码真实行为**写，不臆造：快捷键全表、帧三态语义、`pts`/`pred_pts` 双轨与 `2` 键无损回退、
  推进模式 `Prev[1]`/`Model[2]` 语义、HUD 字段与 `MOD:NONE` 后果、三层叠加标记配色、放大镜 248×248/8×、
  吸附 7×7 与 `Alt` 临时禁用、滤镜只影响显示、落盘路径与原子写、加载优先级与断点续标、纵横数据的正确目录搭配。
* **只写规划文件，不生成 PPT**（用户 2026-11-16 明确要求）。
* 核对方式：静态比对 `point_annotator.py` + `README.md` + 独立 reviewer 复核，**未改任何生产代码**，
  自检基线仍为 101/101。

### 9.1 写课件时挖出来的 11 条**行为事实**（评审复核逐条回源码确认，已回写进课件）

> 这批都是「按字面理解会踩坑」的点，README 里没写清楚，**标注员培训必须讲**。

1. **`2` 键不是无条件无损**：`_adopt_pred()` 无条件 `set_pts(index, pred_pts)`；该帧**没有预推点**时
   ⇒ 清空当前点，且 `set_pts` 空点会把 `pos` 降回 `ign`（自检 `:1802` 固化此行为）。
2. **预推点树不是逐帧全覆盖**（实测磁盘）：`VIDEO00033` 3152 帧只有 **1068** 帧有 `.txt`，
   `VIDEO00028` 3069 帧只有 **441** 帧 ⇒ 后半段视频没有任何青蓝圈，`Model[2]` 等价于失效。
3. **`Model[2]` 模式按 `n`/`i` 后，下一帧会被自动置成 POS**：`_mark` → `_advance(1)` → model 分支
   把 `pred_pts(target)` 写进 `pts`（`:984`）⇒ 状态变 `pos`，画面上凭空多一个绿圈。
4. **删光最后一个点 ⇒ POS 变 IGN（不是 NEG）**：`_delete_active` → `set_pts` 空点 → 回默认态。
5. **已标 NEG 的帧再按 `1`/`2`/左键 ⇒ 翻成 POS**：`set_pts` 里「点非空 ⇒ state=pos」。
6. **断点续标只认 `edited` 帧**：`_resume_index` 返回第一个 `not edited` 的帧；
   `_advance` 的自动继承走 `set_pts(target, current)` 且 `edited` 缺省为 `None` ⇒ **继承/跳过的帧仍是 False**
   ⇒ 重开会退回很靠前，不是「上次位置」。
7. **落盘失败是静默的**：`_save` 只往 `point_annotator.log` 写一行 `标注落盘失败`，界面无任何提示；
   `annotations` 目录 mkdir 也是 `except OSError: pass`。只读目录 ⇒ 标了等于没标。
8. **`MOD:NONE` 时 `_shift_down` 回退到字符大小写** ⇒ Caps Lock 开着会把 1px 判成 5px；
   只有 Qt 后端（`keyToAscii` 不带修饰键）才真退化成 1px / ±1 帧。
9. **树来源载入后 `pts == pred_pts`**：`_read_tree` 给 `pts=points, pred_pts=[]`，
   载入时 `predicted = record.get("pred_pts") or points` ⇒ 初始绿圈**画在青蓝之上、把青蓝盖住**，
   把点拖偏才看得出两个圈。
10. **`run.bat` 不在 exe 包里**：`build_exe.bat` 只打包 `point_annotator.py`；
    `run.bat` 需要同目录 `python\python.exe` 或 PATH 里的 python ⇒ 「免装 Python」只对 exe 成立。
11. **自检断言条数随环境变**：`suite.*` 调用点 81 处，其中 X11 分支在循环里多断言几条
    ⇒ 「101/101」只在 X11 通道可用时成立（无 DISPLAY / Windows 会少几条）。

---

## 10. Windows 侧运行手册（2026-11-18 新增）

> 起因：用户要求「在 Windows 上跑这个工具」。README §6 已从「Windows 分发」重写为完整运行手册
> （方式 A 直跑 / 方式 B 打包 exe / 方式 C 便携 Python + `run.bat` + 高频坑表 + 后端差异）。

### 10.1 🔴 本轮修掉的坑：`--self-test` 在 windowed exe 里输出被吞

* 现象：PyInstaller `--windowed`（`build_exe.bat` 用的就是它）下 `sys.stdout is None`，
  而 `run_self_test()` 原来用**裸 `print()`** ⇒ 结果一行都不出；
  「在目标机上跑一次 `--self-test`」恰恰是 Windows 侧**唯一**可用的验收手段。
* 修法：自检两行 `print` 改成走已有的 `log()`（stdout 有则打印 + 始终写 exe 同目录
  `point_annotator.log`），**零新增依赖、零行为变更**（控制台下的输出完全一样）。
* 验证：① `python point_annotator.py --self-test` → `PASS 101 / FAIL 0`；
  ② **用 `sys.stdout=None; sys.stderr=None` + `runpy` 模拟 windowed** → 日志尾部出现
  `[self-test] PASS 101  FAIL 0`（修复前这一行根本不会出现）；
  ③ `ruff check` / `ruff format --check` 全绿。

### 10.2 Windows 侧必须记住的几条

1. **装 Python 必须勾 `tcl/tk and IDLE`**：目录选择框是 `tkinter.filedialog`，
   缺 tkinter ⇒ 不弹框（退化成命令行输入），且 `screen_size()` 退回 1920×1080。
2. **exe 是 windowed 的，没有控制台**：启动信息、落盘失败、自检结果**全在 exe 同目录的
   `point_annotator.log`**；「双击没反应」第一件事就是看这个日志。
3. **`dist\PointAnnotator` 整个目录都要拷走**（含 `_internal\`），缺了就起不来。
4. **`run.bat` 依赖 Python**（同目录 `python\python.exe` 或 PATH），**不是免安装方案**，
   而且 `build_exe.bat` 不会把它打进 `dist\`。
5. **官方 Windows wheel 走 Win32 后端**（`WM_CHAR` + `VK<<16`）：方向键不会误触发微调，
   也**没有** Qt 后端「F 键与字母键同码」的歧义；但 `WM_CHAR` 带大小写，
   ⇒ `MOD:NONE` 时 Caps Lock 会把 1px 判成 5px（§9.1 第 8 条同源）。
6. **杀软误报**：PyInstaller 产物的常态，把 `dist\PointAnnotator` 加进 Defender 排除项。

---

## 11. Windows 交付包（2026-11-18 落地）

**位置**：`/run/media/manu/3061aec8-10bf-45ee-abd2-03ec85b7f084/workspace/PointAnnotator_Windows/`
（⚠️ **2026-11-24 修正**：外接卷已改挂到 `/run/media/manu/3061aec8-…/`，旧路径 `/media/manu/1TB-Volume/…` **已不存在**（`/media/manu` 下无任何条目）⇒ 按旧路径找会找不到包。
⚠️ 历史记录里 `/media/manu/1TB-Volume` 是 2026-11-18 当时的挂载点，卷顶层 `ro`、只有 `workspace/` 子目录 `rw` ⇒ 包只能落在 `workspace/` 下）

```
PointAnnotator_Windows\
├─ 1_自检.bat / 2_开始标注.bat / 3_打包exe给标注员.bat   （chcp 65001 + python|py 回退 + pause）
├─ README_先看这个.md        （Windows 运行手册：Linux 命令 → Windows 路径对照表）
├─ tool\                     （point_annotator.py + build_exe.bat + run.bat + README.md）
├─ docs\                     （15 页精简版课件 + 23 页详版讲稿稿）
├─ videos\                   （58 条原始视频，可写副本；标注落 videos\annotations\）
└─ preannot_rc2_tracked\     （58 序列，96232 个文件，只读）
```

* **Linux 命令 → Windows 的映射**（用户已用 Linux 命令跑通）：
  `python point_annotator.py <视频目录> --preannot-dir <预推点目录>`
  ⇒ `python tool\point_annotator.py videos --preannot-dir preannot_rc2_tracked`
  （已包成 `2_开始标注.bat`，双击即可）
* **拷贝完整性已实测**：58/58 视频 md5 全一致 · 预推点 **96232** 个文件路径全一致 ·
  抽样 88 个（每序列 1 个 txt + 全部 `scores.json`）md5 全一致。
* ⚠️ **拷贝时发现**：`纵横AI数据采集_mp4/annotations/` 里本来就有 **2 个 json**
  （`VIDEO00028` / `VIDEO00031`，服务器侧用工具试跑留下的草稿，非人工标注），
  随视频一起被拷进了 `videos\annotations\`。工具会优先读它们 ⇒ 启动即停在旧草稿上。
  已在包内 README 写明「想从零开始标就先删掉这两个 json」。
  （本轮验证写盘时误改写了其中一份，**已按字节从只读源还原，md5 双侧一致**。）
* **exe 只能在 Windows 上打包**（Linux 无法交叉编译出 Windows 可执行文件）⇒ 包里给源码 + `.bat`。

### 11.1 🔴 `.bat` 在 Windows 上炸掉的根因（用户实测报 `'锛?echo' 不是内部或外部命令`）

**现象**（2026-11-18，用户在 `Z:\workspace\PointAnnotator_Windows` 跑 `1_自检.bat`）：

```
'锛?echo' 不是内部或外部命令，也不是可运行的程序
't_annotator.py' 不是内部或外部命令，也不是可运行的程序
'k' 不是内部或外部命令，也不是可运行的程序
此时不应有 &&。
```

**根因（四条，不是玄学）**：

1. **cmd.exe 按 ANSI（中文系统 = GBK/936）读 `.bat`**，不是 UTF-8
   ⇒ 批处理里直接写中文，`echo` 被前面的中文字节截断成乱码命令（`锛?echo`）。
2. **`chcp 65001` 之后 cmd 按「字节偏移」定位自己**，而 UTF-8 中文是多字节
   ⇒ 代码页切换后文件位置错位，后面的行被切成碎片（`t_annotator.py`、`k` 都是被切碎的残片）。
3. **`&&` 写在 `if ... ( )` 括号块里**会被判语法错（`此时不应有 &&`）。
4. **LF 换行**在带 `goto` 的批处理里不保险（我们第一版就是这么写的）。

**修法（已落地，7 个 `.bat` 全部通过静态校验）**：

* **纪律：`.bat` 一律「纯 ASCII + CRLF」；中文提示放独立 UTF-8（无 BOM）文本，
  脚本里 `chcp 65001 >nul` + `type msg_cn.txt` 输出。**
* 判定 Python：`where python >nul 2>nul` + `if not errorlevel 1 goto :label`，
  再退 `py`；**全程不用括号块、不用 `&&`**。
* 静态校验（自写脚本，7/7 全绿）：纯 ASCII、全 CRLF、`goto`/`call` 目标齐全、
  括号块内无 `&&`、`call` 的 bat 文件真实存在。
* repo 侧 `manu/point_annotator/{build_exe.bat,run.bat}` **同样重写**（原来也是 UTF-8+中文，
  一样会炸），并把纪律写进 `README.md` §6.6。
* 包内新增 `msg_cn.txt` / `msg_build_cn.txt`（UTF-8 无 BOM）。

---

## 12. 三份并行交付（2026-11-25 新增；主档 `manu/memory/memory.md` §315）

> 需求：母包 `PointAnnotator_Windows/` 里是**全部 58 条视频 + 58 个序列的预推点**，
> 要拆成**等量三份**发给 3 位标注员（视频与配套素材一起走）。

### 12.1 「等量」的口径（关键裁决）

| 口径 | 能不能用 | 原因 |
| --- | --- | --- |
| 按**条数**平分 | ❌ | 58 条视频长短差 **13 倍**（243 帧 ~ 3,154 帧）⇒ 19 条与 20 条的活儿能差 40% |
| 只按**总帧数** | ❌ | 预标注覆盖率**极不均**：2 条序列 **0 覆盖**、`VIDEO00031` 只有 19/3150 帧（1%）、`VIDEO00062` 有 2820/3148 帧（90%）⇒ 没底稿的帧要**手工落点**，有底稿的帧只需**核对/微调** |
| **总帧数 + 手工帧数 + 工作量当量** 三项同时对齐 | ✅ | 本轮采用。「手工帧」= 视频帧数 − 预标注覆盖帧数；「工作量」= 手工帧 × 1.0 + 有底稿帧 × 0.35 |

分配算法 = **LPT 播种**（工作量降序进当前最轻的一份）+ **局部搜索**
（单条挪动 / 两条对调，只要三份最大相对偏差变小就接受）。

### 12.2 结果

| 份 | 目录 | 条数 | 总帧 | 手工帧 | 底稿帧 | 工作量 | 体积 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `PointAnnotator_Windows_标注员1` | 19 | 58,080 | 25,689 | 32,391 | 37,026 | 238.0 MB |
| 2 | `PointAnnotator_Windows_标注员2` | 19 | 58,266 | 26,073 | 32,193 | 37,341 | 238.9 MB |
| 3 | `PointAnnotator_Windows_标注员3` | 20 | 57,435 | 25,845 | 31,590 | 36,902 | 235.7 MB |

**最大偏差：总帧 0.8% ｜ 手工帧 0.8% ｜ 工作量 0.7%**（条数 19/19/20）。
第 3 份含 2 条**无预推点**序列（`VIDEO00057` 449 帧、`VIDEO00060` 243 帧，合计占总量 0.4%）⇒ 需纯手工，已计入口径。

### 12.3 每份自包含 + 干净

* 带齐：`tool/`（源码 + 免装 Python 的 `dist/PointAnnotator/` exe）、`docs/` 课件、
  `python-3.10.0-amd64.exe`、三个 `.bat`、`README_先看这个.md`（按份改写）、`msg_cn.txt`（按份改写）
  ⇒ **拷到 Windows 上双击 `1_自检.bat` → `2_开始标注.bat` 即可**；
* **不带**：母包 `videos/annotations/` 的 Linux 试跑草稿（免得被当成人工标注接着标）、
  `tool/build/`（PyInstaller 中间态 17 MB）、旧 `*.log`；
* 每份的 `videos/annotations/` 建的是**空目录**；母包**原封未动**。

### 12.4 验收（三重独立复验）

| 验收项 | 结果 |
| --- | --- |
| `verify_split.py`（结构 / 配对 / 逐字节 md5 / 等量 / 无重复无遗漏） | **67 项全绿** |
| 每份 `_split_manifest.md5`（逐文件 md5） | 32,526 / 32,328 / 31,727 条**全 OK** |
| 每份 `tool/point_annotator.py --self-test` | 三份全 **`PASS 101  FAIL 0`** |
| 预标注端到端加载 | **56/58**（另 2 条本就没有底稿） |
| 三份互不重叠 / 母包全覆盖 | 重复 **0**、遗漏 **0**（58/58） |

### 12.5 回收与合并

标注员只需回传 `videos\annotations\` 整个目录；合并时按**视频同名**放回母包
`PointAnnotator_Windows/videos/annotations/` ⇒ 三份视频名互不重叠，**直接并集，不会互相覆盖**。

### 12.6 复现（工具入库）

```bash
# 拆分（--plan 只看方案不落盘）
python3 manu/point_annotator/split_delivery.py \
  --master <母包> --out-dir <输出目录> --parts 3 --frame-cache /tmp/vq_frames.json
# 验收（67 项）
python3 manu/point_annotator/verify_split.py --master <母包> --parts-dir <输出目录>
```

### 12.7 两个踩过的坑

1. **帧数别用 `ffprobe -count_frames`**（逐帧解码，58 条 1080p 要十几分钟）⇒ 用容器表里的
   `nb_frames`，毫秒级；
2. **`.bat` 是 CRLF + 纯 ASCII**（见 §11）⇒ 改写时**按行拆（保留行尾）再插回**，
   按 `\n` 找锚点必然落空。
