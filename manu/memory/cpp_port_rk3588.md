# C++ 移植：RK3588 Golden Reference（2026-10-08 立项，L2 已通过 / L3 未做）

> **状态（2026-10-08 更新）：端到端位精确已在真实数据上验证通过（L2）。**
> 真实 OpenCV **4.10.0** 编译成功。在**两种分辨率**上完成端到端验证：
> `DJI_0051_2`（**512×512**）与 `wg2022_ir_052_split_08`（**512×640**），各 200 帧。
> `--anchor-step 2` 臂：G1 **400/400 帧与 Python 引擎逐位一致**、
> G2 **396/396 文件与冻结 SOTA 数据集逐字节相同**、`[MATS] max|diff|=0`（各 21 lag）。
> `--anchor-step 10` 臂：G1 **400/400** 亦逐位一致（但 step10 ≠ SOTA 特征，见第六节）。
> **即「step 2 时输出与 SOTA 特征完全一致」已在两种分辨率上证实。**
> 24 序列全量（L3）**未执行**：覆盖 2/24 序列、400/31,613 帧（1.27%），是局部证据。
> **旧口径 F1/时延数字仍为 Python 侧引用，非 C++ 侧结论；Debug(-O0) 时延禁止进预算。**
> 服务器编译与真实数据验证由用户手工执行（铁律一）。

---

## 一、立项动机

`gmc_net_pending.md` 第九~十三节把嵌入式移植的第一个决策问题逼到了墙角：

- No-GMC 全量 F1 **0.900945**（ΔF1 −0.005417），真实退化 ⇒ **GMC 不能删**；
- tree 锚点级联 F1 **0.906050**（ΔF1 −0.000311），拟合 22 → 6.19/帧（省 71.9%），
  但 **warp 仍是 22 次/帧未减少** ⇒ 「省算力」这句话只覆盖了一半；
- 第十节的流式引擎已被证明与离线 tree 特征**逐字节相同**（G0/G1/G2 全 0），
  是 RK3588 的 Golden Reference —— 但它目前是 **Python**。

RK3588 NPU 要跑的是 C++/C 侧代码。因此需要一个**与 Python 逐位一致**的 C++ 实现，
否则「Python 侧 F1 0.906050」这个数字无法合法地搬到 NPU 上。

---

## 二、核心工程判断：为什么这个移植能做到逐位一致

**Python 参考实现本身也不是算法的重新实现。**
`cv2` 是 OpenCV C++ 翻译单元的薄绑定；本项目依赖的全部原语都位于 `libopencv_*` 的编译 C++ 代码中：

| 原语 | OpenCV 模块 | 本项目参数 |
| :--- | :--- | :--- |
| Shi-Tomasi 角点 | `video`/`features2d` | `maxCorners=600, qualityLevel=0.01, minDistance=4, blockSize=3` |
| 金字塔 Lucas-Kanade | `video/tracking` | `winSize=(15,15), maxLevel=2`，其余默认 |
| RANSAC 部分仿射 | `calib3d` | `method=RANSAC, ransacReprojThreshold=3.0`，其余默认 |
| `warpAffine` | `imgproc` | `INTER_LINEAR, BORDER_REFLECT` |
| `resize`（downscale=2） | `imgproc` | `INTER_LINEAR` |

⇒ 移植只要**调用同一批 C++ 入口、传同样参数、同调用顺序、同 OpenCV 构建、单线程**，
原语结果就是同一份机器码。**逐位一致性来自这个结构事实，而不是来自"写得小心"。**

真正需要移植的是 Python 叠加在原语之上的**编排层**：

| 编排内容 | Python 出处 |
| :--- | :--- |
| 绝对帧号索引的环形缓冲 | `OnlineFeaturePipeline._ring` / `_frame_at_lag` |
| 锚点网格 + 有界链式合成 | `_transforms` / `anchor_grid` |
| float64 齐次合成 + 有限性守卫 | `_compose` / `_to33` |
| 奇数长度中值 + float32 裁剪 | `np.median` / `np.clip` |
| 通道装配顺序 | `np.stack` |

---

## 三、浮点契约（本次移植新增的硬约束）

1. **`-ffp-contract=off`**（CMake 强制，配置阶段校验）。
   否则编译器会把 `_compose` 里的 `acc + a*b` 融合成 FMA，舍入从两次变一次。
   **这是「微小但非零偏差」最可能的成因**，且量级会随链深放大。
2. **禁止 `-Ofast` / `-ffast-math`**。它们开启重结合，而 `_compose` 的 k 升序点积对顺序敏感。
   CMake 在检测到 `-ffast-math` 时**直接 `FATAL_ERROR`**。
3. **OpenCV 必须与 Python 侧同一构建**，双方 `setNumThreads(1)`。
   多线程下 RANSAC 与 LK **不是逐位可复现的**：锚点拟合会在两个同样合理的解之间翻转，
   整条 warped 边缘随之位移。二进制启动时打印 `opencv=`，必须与 `cv2.__version__` 对照。
4. **RANSAC 消耗 OpenCV 全局 RNG**（`theRNG()`）。默认两侧发出相同抽取序列因而一致；
   `--rng-seed-per-fit` 是加固开关，**启用后必须在 Python 侧同样启用**，
   且**不再复现冻结特征集**（冻结集是无重置构建的）。
5. **中值取奇数个（21）⇒ 是选择不是平均**，任何实现都精确。
   若改成偶数长度就必须去匹配 numpy 的并列与溢出规则——这是移植时最容易悄悄破坏的一点。

---

## 四、`anchor_step` 语义（复用第九、十三节结论）

| 配置 | 锚点 | 合成深度 | 拟合/帧 | 与冻结 native 的关系 |
| :--- | :--- | ---: | ---: | :--- |
| `--anchor-step 10` | `{2,12,22,32,42}` | 4 | 6.19 | **不等价**，F1 0.906050（引用值） |
| `--anchor-step 2` | `{2,4,…,42}` | **0** | 21.00 | **精确等价**，即 SOTA 特征本身 |

`anchor_step=2` 是与冻结 native 精确等价的**唯一**设置（每个 lag 都做长基线直接拟合，零连乘）。
它同时充当本移植的**空对照臂**：把被测变量退化为恒等（合成深度 0），
此时 C++ 输出在数学上必须等于基线，与基线的差**即纯移植漂移**，无法与特征设计混淆。

### 覆盖 A/B 两臂

`ab_test_chained_gmc.py` 的决策问题是「GMC 是替换还是仅加速」，其 timing stage 会同跑两臂。
本移植用 `--mode` 对齐：`tree`（`W(.)` = 拟合 GMC）与 `nogmc`（`W(.)` = `IDENTITY`，
0 拟合 0 warp，严格对应 `build_nogmc_median_dataset.py --mode nogmc`：
`Ch1 = |I_t − I_{t−2}|` 无 warp、`Ch2` 用未 warp 的 21 帧历史取中值）。

> nogmc 臂**不得**被当作"已验证的降本方案"：全量 F1 0.900945（ΔF1 −0.005417）是真实退化，
> 该数字来自 Python 侧全量评测，**C++ 侧未做 nogmc 臂的精度评测**（L2 只测了 tree 的
> step 2 与 step 10 两臂）。

---

## 五、构建与验收命令（由用户在服务器执行）

```bash
cd /home/manu/mnt/pycharm_project_10ae9e2e

# 探测 C++ 头文件是否可用（Python cv2 能用不代表头文件在）
ls /usr/include/opencv4/opencv2/core.hpp 2>/dev/null \
  || ls /usr/local/include/opencv4/opencv2/core.hpp 2>/dev/null \
  || echo "NEED_OPENCV_DEV"     # 需要时 sudo apt-get install -y libopencv-dev

cmake -S manu/pipeline/cpp -B manu/pipeline/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build manu/pipeline/cpp/build -j
```

主验收（step=2 复现 SOTA 特征）：

```bash
cd /home/manu/mnt/pycharm_project_10ae9e2e
mkdir -p runs/cpp_port
PYTHONPATH=. python manu/pipeline/verify_cpp_port.py \
    --cpp-bin manu/pipeline/cpp/build/gmc_stream \
    --raw-root /mnt/data/siping/datasets/manu/anti-uav \
    --sequence wg2022_ir_052_split_08 --limit 200 --anchor-step 2 \
    --frozen-root /mnt/data/siping/datasets/manu/uav_gmc_median \
    --cpp-out-dir runs/cpp_port/g2_jpg --md5-out runs/cpp_port/cpp_step2.md5 \
    --dump-dir runs/cpp_port/dump_step2 --timing \
    --out-json runs/cpp_port/verify_step2.json
```

**通过判据（缺一不可）**：`[G1] PASS` 200/200 帧 md5 全等；`[G2]` 与冻结集文件逐字节相同；
`[MATS] max|diff| = 0.000e+00`。完整命令集（step=10 臂、24 序列全量臂）见 `manu/pipeline/cpp/README.md`。

---

## 六、验证进展分级（2026-10-08）

**必须按级引用，不得跨级。** 下面是实测到哪一级，不是预期能到哪一级。

| 级别 | 内容 | 状态 | 证据 |
| :--- | :--- | :--- | :--- |
| **L0 编译** | `gmc_stream.cpp` 在项目浮点选项下编译 | **已通过** | `-fsyntax-only` 与 `-c` 均 exit 0，产出 122 KB 目标文件，22 个导出符号 |
| **L1 单元** | 移植**自己写的**标量逻辑 vs Python | **已通过** | `test_port_units.py`：compose 20000/20000、Ch2 中值 60000/60000、anchor_grid 9/9、natural_less 8/8、md5 8/8，全部逐位一致 |
| **L1' 闸门自测** | 验证器自身的映射与对齐 | **已通过** | `test_verify_helpers.py` 15/15 |
| **L1'' 编排** | `push()` 的寻址与合成顺序 | **已通过** | `test_orchestration.py`：21/21 lag 解析到正确帧 + **变异检查通过** |
| **L2 端到端** | 真实数据整条管道 vs Python 引擎 | **已通过** | **双分辨率**（512×512 / 512×640）各 200 帧：`anchor_step=2` G1 **400/400**、G2 **396/396 逐字节**、`[MATS] max\|diff\|=0`；`anchor_step=10` G1 **400/400** |
| **L3 全量** | 24 序列 / 31,613 帧 | **未执行** | 局部覆盖 **2/24 序列、400/31,613 帧（1.27%）**；双分辨率 + 各 200 帧已足以判定移植路径，但**不是全量铁证** |

### L1 为何能成立、边界在哪

- L1 覆盖的是**移植自己写的**部分：链式合成、锚点网格、帧序比较器、中值+裁剪、md5 指纹。
- L1 **不覆盖** OpenCV 原语（Shi-Tomasi / LK / RANSAC / warpAffine / resize）——
  它们是 Python 侧**已经在调用的同一份 C++ 代码**，等价性由构造保证，不由测试保证。
- L1 **不覆盖** 管道装配顺序、RNG 抽取序列、OpenCV 构建一致性。**这三项只能由 L2 证明。**
- L1 用的编译选项与 CMake 完全一致（`-O3 -ffp-contract=off -fno-fast-math`），
  所以「compose 逐位一致」这一结论**只在这些选项下成立**。

### 第 12 个缺陷：`CMakeLists.txt` 的浮点守卫**从未生效**，且挡住了 CLion 配置（2026-10-08 已修）

这是**唯一一个会让整套守卫形同虚设**的缺陷，必须单独记录：

```cmake
# 修复前 CMakeLists.txt:66 —— 把 <CONFIG> 当成了占位符
set(_all_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_<CONFIG>}")
```

- `<` 是非法变量名字符 ⇒ **CMake 语法错误**，配置阶段直接失败，
  CLion 完全无法加载该工程（`Invalid character ('<') in a variable name`）。
- 即便能解析，**逐配置变量槽位也正是该守卫唯一能抓到「发行版往 `CMAKE_CXX_FLAGS_DEBUG`
  注入 `-ffast-math`」的入口**，而它是唯一失效的那个。
- 同行下方 `foreach(_f IN LISTS COMPILE_OPTIONS)` 遍历的是**未定义变量**
  （真正的目标是 target property，在其后用 `get_target_property` 取），**纯死代码**。

修复（已落地并双向验证）：

```cmake
string(TOUPPER "${CMAKE_BUILD_TYPE}" _cfg)
set(_all_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_cfg}} ${CMAKE_CXX_FLAGS_RELEASE}")
```

**验证**：正常配置打印 `Floating-point guard: OK (ffp-contract=off present, no fast-math)`；
且注入 `-DCMAKE_CXX_FLAGS=-ffast-math` 与 `-DCMAKE_CXX_FLAGS_DEBUG="-O2 -ffast-math"`
**两者都会 FATAL_ERROR**。守卫的「会失败」被证明，不是恒真装饰。

> **方法论沉淀**：一个守卫若从未被真正执行过，其文档化的保护力是**零**，
> 且比没有守卫更危险——它会让人误以为浮点契约已被强制。
> 凡「配置阶段自检」类断言，必须有一条**反向注入测试**证明它会失败。

### 位精确性对优化等级不敏感（-O0 与 -O3 均通过）

一个超出预期的正向结论：

| 构建 | 编译行实测 | 覆盖 | 结果 |
| :--- | :--- | :--- | :--- |
| **L1 单元** | `-O3 -ffp-contract=off -fno-fast-math -fno-unsafe-math-optimizations` | 自有标量逻辑 | 逐位一致 |
| **L2 端到端** | `-g -std=c++17 -ffp-contract=off …`（**无 `-O`，即 `-O0`**） | 整条管道 | 逐位一致 |

即 L1 在 `-O3`、L2 在 `-O0` 下**都**与 Python 逐位一致。
这与浮点契约的设计意图一致（`-ffp-contract=off` 禁止 FMA 收缩；
中值/裁剪是整数与精确浮点路径），并说明**位精确性不依赖优化等级**。

⚠️ 但这只对**精度**成立，**不适用于时延**：`-O0` 下的耗时不可用于性能判断（见第六节末）。

### step 10 臂：C++ 同样位精确，但它 ≠ SOTA 特征（更正流传说法）

- `verify_2res_s10.json`：**PASS，G1 400/400 逐字节一致**
  （`DJI_0051_2` md5 `aeffe02f7af7995a819326e55fb17ad4`、
  `wg2022_ir_052_split_08` md5 `75c804b726cafd929358873b60117f05`）。
  ⇒ **C++ 移植对两种 `anchor_step` 都位精确**，不只零连乘那一条。
- **但 step 10 与 step 2 的特征不同**：逐帧比对两份 md5，**400 帧中 354 帧不同**，
  两条序列**均自第 23 帧起分歧**，第 0~22 帧全同。
  第 0~22 帧相同是冷启动钳制区（历史帧不足，链式合成输入相同）。
- 因此准确表述是：**C++ 精确复现 step 10 本身；但 step 10 ≠ SOTA 冻结特征**
  （与第九/十二节 tree F1 0.906050、ΔF1 −0.000311 一致）。
  对 step 10 臂报 `G2 not applicable` 属预期设计，不是失败。
- ⚠️ 顺带暴露**第 13 个缺陷（未修）**：`runs/cpp_port/g2_full/` 与 `g2_multi/` 里
  的 JPG **文件名不含序列名**（`000000.jpg`…`001466.jpg`），
  而 `--out-dir` 代码路径已按第 10 个缺陷的修复加了序列名前缀。
  这批历史产物来自修复前的旧二进制，**且多序列共用一个 out-dir 时会互相覆盖**
  ⇒ 该目录下的 G2 证据**不可引用**。新跑务必用带序列名前缀的产物目录。

### L2 口径如实标注（第三次「局部≠全量」风险面）

| 判据 | 覆盖 | 占全量 |
| :--- | :--- | :---: |
| 序列数 | **2 / 24** | 8.3% |
| G1 帧数 | **400 / 31,613** | 1.27% |
| G2 帧数 | **396 / 31,613** | 1.25% |

`verify_step2.json` 自带 `[SCOPE] … LOCAL check, not full-dataset evidence`。
**G2 的 198/198 而非 200/200 是正确的**（冻结集清单自 `__000003` 起，前 2 帧不在清单内）。
引用「C++ 位精确复现 SOTA 特征」时**必须同时给出序列数 / 帧数 / 分辨率**。

### 第 10、11 个缺陷：运行期才暴露的两个（已修）

10. **`--out-dir` 的 JPG 文件名不带序列名**，只有 push 序号。
    多序列共用一个 `--out-dir` 时，**每个序列都会覆盖上一个序列的同名文件**，
    G2 随后会把「最后一个序列的文件」拿去和「第一个序列的冻结帧」比对——
    又是一个**自信的错误结论**，且在全量跑批时才发作。
    `--dump-dir` 一直带序列前缀（`%s_%06zu`），唯独 `--out-dir` 漏了，不一致才暴露了它。
    已改为 `<seq>_%06zu.jpg`，验证器查名同步更新。
11. **计时输出单位错 1000 倍**：`t_fit` 等累加器本来就是毫秒
    （`duration<double, std::milli>`），打印时又乘了 `1e3`，于是「ms」标签下印的是**微秒**
    （实测 `total=230094 ms/frame`，真实值 **230.09 ms**）。
    本项目时延本就有 5 倍不自洽的历史争议，再注入一个 1000 倍单位错误会让嵌入式预算彻底失真。
    8 处已修正；修正后 `step=10` 实测 **125.08 ms/帧**（fit 39.64 / warp 38.89 / median 50.32），
    与 memory 记录的 native 183.32 ms 同量级，量级合理。

### 第 8、9 个缺陷：只有真实 OpenCV 才能暴露的两个（已修）

**这两个是手写桩头文件的直接代价，也是本轮最重要的一条方法论教训。**

L0/L1 全程用**我凭记忆写的** OpenCV 桩头文件编译。它顺利通过，却掩盖了两个真实错误：

8. **`using cv::uchar;` 在 OpenCV 4.10 中不存在** —— `uchar` 是全局 typedef，不在 `cv` 命名空间。
   桩把它写进了 `cv`，于是错误被固化。
9. **`warpAffine` 参数顺序写反** —— 真实签名是
   `warpAffine(src, dst, M, dsize, flags, borderMode)`，**矩阵在 `dsize` 之前**；
   我按桩的顺序写成了 `(src, dst, dsize, M, ...)`。

装上 `libopencv-dev` 用真 OpenCV 4.10.0 一编译，两处立刻报错。

> **这条必须写进铁律：手写的桩头文件只能证明「代码自洽」，不能证明「API 调用正确」。**
> 我此前声称「桩能证明调用形态正确」——**这句话被这两个错误证伪了**。
> 桩是从记忆生成的，记忆错了桩就会把错误固化，而且比无桩更危险：
> 它给出一个虚假的通过信号。任何跨库移植的桩都必须与真实头文件**逐行比对签名**，
> 并且**桩编译通过不能替代真实编译**。

### 编译验证抓到的三个真实缺陷（已修）

这三个都是**编译能通过就不可能暴露、肉眼也容易漏**的错误，靠桩编译与单元对拍抓出来：

1. **`Matx33d` 未引入作用域** —— 只写了 `using cv::Matx23f`，漏了 `Matx33d`，
   而 `to33()` / `compose()` 都在用它。真编译必然失败。
2. **`ring_depth_` 不存在** —— `push()` 里写成了下划线结尾，实际成员是**方法** `ring_depth()`。
3. **`natural_less` 的分词器不是 `re.split` 的忠实移植** —— 本项目最危险的一个。
   `re.split(r"(\d+)", stem)` 因**带捕获组**，总会返回每段数字前后的文本，**包括空串**
   （`"000001"` → `['', '000001', '']`）。原实现丢弃空文本段，token 序列因此不同，
   **排序结果会变**。已改为严格镜像 `re.split`。

> 第 3 条的教训值得单列：**「同前缀文件排序一致」会掩盖这个 bug**，因为真实序列目录内
> 文件命名模式统一，token 对齐恰好掩盖了差异。它只在混排不同模式时暴露——
> 而「只在别人的数据上不暴露」的错误，正是必须在合成用例上主动对拍的原因。

### 验证器自身抓到的第四个缺陷（已修）——本轮最危险的一个

`verify_cpp_port.py::build_push_to_manifest` 里**字典方向反了**：

```python
list_to_frame = {li: fi for fi, li in idx_map.items()}   # 键是「列表下标」
li = list_to_frame.get(fi)                              # 却用「帧号」去查
```

`idx_map` 本来就是 **帧号 → 列表下标**，方向正好够用，却被反转了一次。
合成树上实测 **10 帧只映射上 4 帧，且键是帧号而非 push 下标**。

**后果不是报错，而是一个自信的错误结论**：G2 会在错误的帧上比对，
照样打印整齐的表格和「零漂移」。这正是该函数 docstring 自己警告的
「猜名正是『报告零漂移其实在比不同帧』的成因」——而它自己就犯了这个错。

已修（直接用 `idx_map.get(fi)`），并新增 `test_verify_helpers.py` 固化 15 项断言，
其中**第 7 项是回归守卫**、第 8~13 项覆盖 `[MATS]` 的帧/序列对齐：重现那段反转逻辑并断言它**不等于**正确映射。

**同一处又发现两个 `[MATS]` 对齐错误（静默、方向相同）**：

4. **C++ 在第 0 帧落盘 `_mats.npy`，Python 却读 `pipe.last_mats`（最后一帧）** ——
   即使**单序列也永远比错帧**。`last_mats` 的字面意思就是「上一次」，
   而 dump 的是 `push index 0`。
5. **`last_mats` 是循环内共享变量，却拿 `sequence[0]` 去比** ——
   多序列（全量 24 序列）时变成「首序列的 C++ 矩阵 vs 末序列的 Python 矩阵」。
   单序列全对、全量全错，属于**只在正式跑批时发作**的典型缺陷。

两处都已修：`python_side()` 改为在 `i == 0` 时抓取并**逐序列**存放，
`[MATS]` 改为逐序列比对并把 `mats` 纳入 JSON 报告。

> 由此得到一条应写进铁律的结论：**闸门脚本本身也必须被测试**。
> 闸门的失败模式是「自信地给出错误答案」，比被测代码出错更危险，
> 因为它的输出正是决策依据。本项目此前所有闸门都默认「写得对就不用测」。
>
> 更具体的补充：**「单序列调试全对」不等于闸门正确**。
> 第 4 条在单序列下也错、第 5 条只在多序列下错，两者都必须在
> **单序列 + 多序列**两种配置下各测一次。只跑一种配置等于没测。

### 编排测试：为什么必须跑，不能靠读

本次移植共抓到 **7 个真实缺陷，全部靠运行发现，靠阅读一个都没抓到**。
其中最危险的一类历史上就出现过：**环形缓冲按 lag 索引、而每一帧又是它自己那次 push 的当前帧**，
导致所有历史帧被取到一半的 lag（`falsified_archive.md` 第 32.2 条「本轮最严重」）。

`test_orchestration.py` 用**带标记的假 OpenCV** 在两侧解耦运行：
帧 k 的像素(0,0) 置为 k+1，`goodFeaturesToTrack` 记住该标记，
`estimateAffinePartial2D` 返回 `[[1,0,marker],[0,1,0]]`，`--downscale 1` 保证标记不被缩放吞掉。
于是 `mats[lag].tx` 就是「lag 解析到了哪一帧」的直接指纹：

- **锚点 lag**  → 应等于 `marker(t − lag)`
- **合成 lag**  → 应等于 `marker(t − below) + Σ marker(t − lag_abs)`（同时验证**合成顺序**）

默认 45 帧、`--mats-frame 44`，使 **21 个 lag 中 20 个落在真实帧**（仅最大 lag 仍命中冷启动钳制），
所以既验证了寻址，也验证了冷启动钳制。结果 **21/21 全对**。

> **原语是假的，这个测试只证明寻址与合成顺序，绝不证明 OpenCV 等价性。**
> 端到端结论仍然只属于 `verify_cpp_port.py` 的 G1/G2。

### 变异检查：测试必须被证明「有牙齿」

一个永远通过的测试证明不了任何东西。因此本测试把**历史上的环形缓冲 bug 原样重新注入**
（`idx % depth` 改成 `lag % depth`），然后断言构建**必须失败**：

```
[CASE 3] PASS -- mutation detected: reintroducing the lag-indexed ring makes 20 lags fail.
```

**测试必须能被证伪。** 新增任何测试时都应回答一个问题：
「把被测行为改坏，它会不会失败？」不会，就是装饰品。

### 编译验证的边界

L0 用**忠实于真实 OpenCV 4.x 签名的桩头文件**完成（本地无 OpenCV C++ 头文件）。
桩能证明语法、类型、调用形态正确，**不能**证明链接期与真实 OpenCV 一致；
真实编译仍须在服务器执行。

### 本线当前状态与未决项

**已确立**：移植路径可行；位精确性有结构性依据（同一批 OpenCV C++ 原语）而非依赖运气；
编译通过（**含第 12 个浮点守卫缺陷已修**）；移植自有逻辑已对拍证明与 Python 逐位一致；
**L2 端到端位精确已在双分辨率真实数据上通过**；位精确性对优化等级不敏感（-O0/-O3 均可）；
编译选项、验证闸门、风险表已固化。

**未决项**：

1. **L3 未验证**：全量铁证仍需 24 序列 / 31,613 帧跑完。局部 2 序列 400 帧只是局部
   （覆盖 1.27%）。**这是当前最大的证据缺口。**
2. **时延无可用数字**：Debug(-O0) 实测 `median` 占 58~64%（详见
   `gmc_net_pending.md` 14.1），`fit` 折算 4.8~6.6 ms/次。
   优化构建（`-O3`）的同口径分段计时**尚未测量**，本工具任何 ms 数字
   **禁止**进入嵌入式预算。
3. 与第十一节同一个未解项：时延测量历史性差 5 倍（反解 9.06 ms/次 vs 实测 1.80 ms/次）。
   按第十三节 13.4 的教训，**本工具报出的任何 ms 数字都必须与
   `ab_test_chained_gmc.py --stage timing` 交叉验证后才可引用**，
   不一致时禁止取其一、禁止取平均。
4. **第 13 个缺陷待修**：`g2_full/` / `g2_multi/` 历史产物文件名不含序列名
   （多序列共用 out-dir 会互相覆盖），该批 G2 证据不可引用。

### 交付物

| 文件 | 职责 |
| :--- | :--- |
| `manu/pipeline/cpp/gmc_stream.cpp` | C++ 流式引擎（含分段计时与 md5 指纹） |
| `manu/pipeline/cpp/CMakeLists.txt` | 锁定浮点编译选项；检测 fast-math 即失败 |
| `manu/pipeline/cpp/md5.h` | MD5 + RFC 1321 三向量启动自检 |
| `manu/pipeline/cpp/test_port_units.py` | **L1 离线单元验证**（无需数据集/显卡，秒级） |
| `manu/pipeline/cpp/test_verify_helpers.py` | **验证器自身的单元测试**（映射 + stdout + MATS 对齐，15 项） |
| `manu/pipeline/cpp/test_orchestration.py` | **编排寻址测试** + 变异检查（假 OpenCV，秒级） |
| `manu/pipeline/cpp/README.md` | 位精确契约、构建命令、验收判据、风险表 |
| `manu/pipeline/verify_cpp_port.py` | **L2/L3 验证器**：G1 逐位 md5 / G2 冻结集逐字节 / `[MATS]` / G0 空对照 |

**已同步至** `/home/manu/mnt/pycharm_project_10ae9e2e/manu/pipeline/`（md5 校验逐字节一致）。

### 验证纪律

- **G1** 比在**编码前数组**上（md5/帧）。严禁拿 decoded JPG 判严格阈值：
  三通道 JPEG 是 4:2:0 色度下采样 + 量化，那种差异度量的是存储格式而非特征。
- **G2** 只对 `anchor_step=2` 开放。step=10 对照冻结集只会得到无意义的 −0.0003。
- **每个闸门必须打印它实际比较了多少样本**；比较数为 0 时显示 `SKIP`，**绝不显示 `PASS`**。
- 验证器**不复现算法**，直接驱动既有 Python 引擎，避免出现第二个逐渐走偏的 Python 实现。
- 冻结集 push index → manifest 文件名映射复用**冻结 builder 自己的** `natural_key` + `idx_map`，
  不做名称猜测——猜名正是「报告零漂移其实在比不同帧」的成因。