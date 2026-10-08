# C++ 移植版流式特征引擎（RK3588 / Golden Reference）

`gmc_stream` 是 `manu/pipeline/streaming_feature_pipeline.py`（`OnlineFeaturePipeline`）的 C++ 移植，
目标是与 Python 版**逐位一致**的特征输出。

```
Ch0 = I_t
Ch1 = |I_t - W(I_{t-2})|                        (lag 2 精确，绝不取 1 或 4)
Ch2 = (I_t - B_t)^+,  B_t = median{W(I_{t-2k})}, k = 1..21
输出 = (3, H, W) uint8，通道序 [Ch0, Ch1, Ch2]
```

---

## 1. 为什么这个移植能做到逐位一致

关键点：**Python 参考实现本身也不是算法的重新实现**。`cv2` 只是 OpenCV C++ 翻译单元的薄绑定——
Shi-Tomasi、金字塔 Lucas-Kanade、RANSAC 部分仿射、`warpAffine`、`resize` 全部位于 `libopencv_*`
的编译 C++ 代码中。

所以本移植**不重写这些原语**，而是：

- 调用**同一批 C++ 入口函数**（`cv::goodFeaturesToTrack` / `cv::calcOpticalFlowPyrLK` /
  `cv::estimateAffinePartial2D` / `cv::warpAffine` / `cv::resize`），
- 传入**逐字相同的参数**，
- 以**相同的调用顺序**执行（RNG 消耗顺序因此也相同），
- 运行在**同一个 OpenCV 构建**上，且双方 `setNumThreads(1)`。

⇒ 原语结果是同一份机器码，逐位相同。

真正被移植的是 Python 在原语之上叠加的**编排层**：

| 移植内容 | 对应 Python |
| :--- | :--- |
| 绝对帧号索引的环形缓冲（绝不列表移位、绝不按 lag 索引） | `_ring` / `_frame_at_lag` |
| 锚点网格 + 有界链式合成 | `_transforms` / `anchor_grid` |
| float64 齐次合成 + 有限性守卫 | `_compose` / `_to33` |
| 奇数长度中值 + float32 裁剪 | `np.median` / `np.clip` |
| 通道装配顺序 | `np.stack` |

---

## 2. 浮点契约（违反任何一条都会破坏逐位一致）

1. **`-ffp-contract=off`**（已在 `CMakeLists.txt` 强制）。否则编译器会把 `acc + a*b` 融合成 FMA，
   舍入从两次变一次。这是「微小但非零偏差」最可能的成因。
2. **禁止 `-Ofast` / `-ffast-math`**。它们开启重结合，而 `_compose` 对求和顺序敏感。
3. **OpenCV 必须与 Python 侧同一构建**，且双方单线程。多线程下 RANSAC 与 LK 不是逐位可复现的——
   锚点拟合会在两个同样合理的解之间翻转，整条 warped 边缘随之位移。
4. **RANSAC 消耗 OpenCV 全局 RNG**（`theRNG()`）。默认两侧发出相同的抽取序列，因此一致。
   `--rng-seed-per-fit` 会在每次拟合前重置种子以对抗无关的 RNG 消耗，
   但**启用它就必须在 Python 侧同样启用**，且**它不再复现冻结特征集**（冻结集是无重置构建的）。
5. **中值取奇数个样本（21）**，因此是「选择」而非「平均」，任何实现都精确。
   若为偶数个则变成平均，必须改为匹配 numpy 的并列与溢出规则。

---

## 3. `anchor_step` 语义

| 配置 | 锚点 | 合成深度 | 拟合/帧 | 含义 |
| :--- | :--- | ---: | ---: | :--- |
| `--anchor-step 10`（默认） | `{2,12,22,32,42}` | 4 | 5 + 1.19 | **交付配置**，F1 0.906050，dF1 −0.000311 |
| `--anchor-step 2` | `{2,4,…,42}` | **0** | 21 | **精确等价臂**；每个 lag 都做长基线直接拟合 |

`--anchor-step 2` 是**唯一**与冻结 native（SOTA）管线精确等价的配置，因为每个 lag 都独立拟合、
零连乘。**用它对照 `uav_gmc_median` 即可证明本移植复现了 SOTA 特征。**

`--anchor-step ∈ {4,6,8}` 未测（用约 3.6% 算力换「可能 ΔF1 = 0」的保险，见 `gmc_net_pending.md` 13.3）。

### A/B 的另一臂：`--mode nogmc`

`manu/evaluation/ab_test_chained_gmc.py` 的决策问题是「GMC 是**替换**还是**仅加速**」，
其 timing stage 会同时跑 native 与 no-GMC 两臂。本移植用 `--mode` 覆盖这两臂：

| `--mode` | 含义 | 拟合/帧 | warp/帧 |
| :--- | :--- | ---: | ---: |
| `tree`（默认） | `W(.)` = 拟合出的 GMC | 6.19（step=10）/ 21（step=2） | 22 |
| `nogmc` | `W(.)` = `IDENTITY` | **0** | **0** |

`nogmc` 臂严格对应 `build_nogmc_median_dataset.py --mode nogmc`：
`Ch1 = |I_t − I_{t−2}|`（无 warp），`Ch2` 用**未 warp** 的 21 帧历史取中值。
它用于与 Python 的 nogmc 特征集做同样的 G1 逐位对拍，
以及复现 A/B 的成本对比（`ab_test_chained_gmc.py --stage timing`）。

> nogmc 臂**不得**被当作"已验证的降本方案"：全量 F1 0.900945（ΔF1 −0.005417）是真实退化，
> 该数字来自 Python 侧全量评测，C++ 侧零实测。

---

## 4. 构建

### 4.0 先跑离线单元验证（秒级，不需要数据集 / 显卡 / 已编译的二进制）

```bash
cd /home/manu/mnt/pycharm_project_10ae9e2e
python manu/pipeline/cpp/test_port_units.py
```

它直接检验**本移植自己写的逻辑**与 Python 是否逐位一致，并把三段函数从 `gmc_stream.cpp`
**原文抽出**编译（花括号配对提取），因此测试不可能与实际交付的代码走样。

| 测试项 | 样本 | 覆盖什么 |
| :--- | ---: | :--- |
| `compose()` vs numpy | 20,000 | float64 齐次链式合成 + 有限性守卫 |
| Ch2 中值+裁剪 vs numpy | 60,000 | `np.median(21×uint8)` → `clip` → uint8 |
| `anchor_grid()` | 9 组 | 锚点集合 |
| `natural_less()` | 8 种命名模式 | 帧序（须镜像 `re.split(r"(\d+)", stem)`） |
| `md5` vs `hashlib` | 8 组 | G1 闸门的指纹本身 |

任何一项 FAIL 或 SKIP 都会显式列出，**SKIP 绝不计为 PASS**。

> 它**不**覆盖 OpenCV 原语（Shi-Tomasi / LK / RANSAC / warpAffine / resize）——
> 那些是 Python 侧已在调用的同一份 C++ 代码，等价性由构造保证。
> 也**不**覆盖管道装配顺序、RNG 抽取序列、OpenCV 构建一致性——**这三项只能由 §5 的 G1/G2 在真实数据上证明。**

前置：需要 `g++` 与 numpy。若机器没有 OpenCV 头文件，`compose()` 一项会显示 SKIP；
装好 §4.1 的 `libopencv-dev` 后加 `--opencv-include /usr/include/opencv4` 即可全项通过。

### 4.1 编译

前置：需要 **OpenCV C++ 开发头文件**（`opencv2/core.hpp` 等）与 CMake。
服务器上的 Python `cv2` 能用不代表 C++ 头文件在，先定位：

```bash
cd /home/manu/mnt/pycharm_project_10ae9e2e

# 1) 找 OpenCV 安装前缀（两种方式任选其一种到结果即可）
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -iE "^\s+(Installation|CMAKE_INSTALL_PREFIX)" -A2
pkg-config --modversion opencv4 2>/dev/null && pkg-config --cflags --libs opencv4

# 2) 确认 C++ 头文件确实存在（这才是能否编译的关键）
ls /usr/include/opencv4/opencv2/core.hpp 2>/dev/null \
  || ls /usr/local/include/opencv4/opencv2/core.hpp 2>/dev/null \
  || echo "NEED_OPENCV_DEV"
```

若输出 `NEED_OPENCV_DEV`，需要装开发包（需要 sudo 时请你在服务器上执行）：

```bash
# Debian/Ubuntu
sudo apt-get install -y libopencv-dev
# 或按上一步 getBuildInformation 里的 CMAKE_INSTALL_PREFIX 找到的前缀，手工补 headers
```

然后构建：

```bash
cd /home/manu/mnt/pycharm_project_10ae9e2e
cmake -S manu/pipeline/cpp -B manu/pipeline/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build manu/pipeline/cpp/build -j
```

配置阶段会打印 OpenCV 版本，并在 `-ffp-contract=off` 缺失或检测到 `-ffast-math` 时**直接报错退出**——
这是刻意的：编译选项被吞掉时位精确性就已经没有保证了。

---

## 5. 验收

### 5.0 ✅ 已验证结论（2026-10-08，真实 OpenCV 4.10.0）

在**两种分辨率**上完成端到端验证，Python 侧使用 `envs/uav`（**cv2 必须与 C++ 侧同为 4.10.0**）：

| 序列 | 分辨率 | G1（C++ vs Python 编码前数组） | G2（重编码文件 vs 冻结 SOTA） | `[MATS]` |
| :--- | :--- | :--- | :--- | :--- |
| `DJI_0051_2` | **512×512** | 200/200 逐位一致 | **198/198 逐字节相同** | max\|diff\| = 0 |
| `wg2022_ir_052_split_08` | **512×640** | 200/200 逐位一致 | **198/198 逐字节相同** | max\|diff\| = 0 |

`--anchor-step 2` 臂合计 **G1 400/400、G2 396/396**，即：

> **`--anchor-step 2` 的输出与冻结 SOTA 特征集 `uav_gmc_median` 完全一致（逐字节）。**

`--anchor-step 10`（交付配置）在同样两分辨率上 G1 400/400 逐位一致。

> **复现命令必须使用 `envs/uav`**，否则 G1 会假失败：
> 用 cv2 5.0.0 的解释器跑，输出与 cv2 4.10.0 的 C++/冻结集全部不同
> （`[G1]` 0/200，`[MATS]` 仍为 0），而 G2 依然 198/198 —— 这个组合本身就是判别依据。



### 5.1 主验收：step=2 必须复现 SOTA 特征

```bash
cd /home/manu/mnt/pycharm_project_10ae9e2e
mkdir -p runs/cpp_port

PYTHONPATH=. python manu/pipeline/verify_cpp_port.py \
    --cpp-bin manu/pipeline/cpp/build/gmc_stream \
    --raw-root /mnt/data/siping/datasets/manu/anti-uav \
    --sequence wg2022_ir_052_split_08 \
    --limit 200 \
    --anchor-step 2 \
    --frozen-root /mnt/data/siping/datasets/manu/uav_gmc_median \
    --cpp-out-dir runs/cpp_port/g2_jpg \
    --md5-out runs/cpp_port/cpp_step2.md5 \
    --dump-dir runs/cpp_port/dump_step2 \
    --timing \
    --out-json runs/cpp_port/verify_step2.json
```

**判据（缺一不可）：**

| 闸门 | 必须 | 含义 |
| :--- | :--- | :--- |
| `[G1]` | `PASS`，200/200 帧 md5 全等 | C++ 与 Python 引擎逐位一致（比在编码前数组上） |
| `[G2]` | 与冻结集比对的文件 **逐字节相同** | step=2 臂复现 SOTA 特征 |
| `[MATS]` | `max|diff| = 0.000e+00` | 每个 lag 的变换矩阵都一致 |
| `[G0]` | 深度 0 ⇒ 本臂**就是** native 管线 | 非零残差只能是移植漂移，与特征设计无关 |

### 5.2 交付配置 step=10

```bash
cd /home/manu/mnt/pycharm_project_10ae9e2e
PYTHONPATH=. python manu/pipeline/verify_cpp_port.py \
    --cpp-bin manu/pipeline/cpp/build/gmc_stream \
    --raw-root /mnt/data/siping/datasets/manu/anti-uav \
    --sequence wg2022_ir_052_split_08 \
    --limit 200 \
    --anchor-step 10 \
    --md5-out runs/cpp_port/cpp_step10.md5 \
    --dump-dir runs/cpp_port/dump_step10 \
    --timing \
    --out-json runs/cpp_port/verify_step10.json
```

此臂 `[G1]` 必须 PASS（它只证明 C++ == Python）。**它不得对照冻结 native 集**——
`anchor_step=10` 有 4 层合成，与 native 不是精确等价，对照会得到无意义的 −0.0003。

### 5.3 全量证据（可选，24 序列）

上面两次是**局部检查**。要把结论写进汇报，必须跑全量：

```bash
cd /home/manu/mnt/pycharm_project_10ae9e2e
SEQS=$(python3 - <<'PY'
from pathlib import Path
import re
d = Path("/mnt/data/siping/datasets/manu/uav_gmc_median/images/val")
seqs = {re.split(r"___|__", p.stem)[0] for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png"}}
print(",".join(sorted(seqs)))
PY
)
ARGS=$(echo "$SEQS" | tr ',' ' ' | sed 's/^/--sequence /' | tr '\n' ' ')

PYTHONPATH=. python manu/pipeline/verify_cpp_port.py \
    --cpp-bin manu/pipeline/cpp/build/gmc_stream \
    --raw-root /mnt/data/siping/datasets/manu/anti-uav \
    $ARGS --limit 0 --anchor-step 2 \
    --frozen-root /mnt/data/siping/datasets/manu/uav_gmc_median \
    --cpp-out-dir runs/cpp_port/g2_full --md5-out runs/cpp_port/cpp_full.md5 \
    --out-json runs/cpp_port/verify_full.json
```

全量判据：31,613 帧 `[G1]` 全等，且 `[G2]` 比对帧数 = 31,613、逐字节相同文件数 = 31,613。

> **纪律**：任何「等价 / 一致」的表述都必须同时给出**序列数 / 帧数 / 分辨率**。
> 本项目已经两次把局部验证称作全量（`ls | head -200` 按文件名排序只覆盖单序列）。

---

## 6. 直接运行（不经 Python 验证器）

```bash
cd /home/manu/mnt/pycharm_project_10ae9e2e
./manu/pipeline/cpp/build/gmc_stream \
    --raw-root /mnt/data/siping/datasets/manu/anti-uav \
    --sequence wg2022_ir_052_split_08 \
    --limit 200 \
    --anchor-step 2 \
    --timing \
    --md5-out runs/cpp_port/direct.md5 \
    --dump-dir runs/cpp_port/direct_npy
```

输出：
- `[SEQ] … frames=N md5=…` 每序列 md5（可直接与验证器打印的 Python md5 对照）
- `[MD5-SUM] …` 全部序列合并 md5
- `--timing` 给出 **fit / warp / median 分段**耗时，并附 `fits/frm`、`warps/frm` 次数列

> **时延数字纪律**：本工具提供的是**第三套**独立测量。按 `gmc_net_pending.md` 13.4 的教训，
> 同一次 `push()` 的成本在不同工具里曾相差 5 倍（反解 9.06 ms/次拟合 vs 实测 1.80 ms/次）。
> **任何进入嵌入式预算的时延数字必须至少来自两套独立工具**，不一致时禁止取其一或取平均。
> 无论本工具报多少，都要与 `ab_test_chained_gmc.py --stage timing` 交叉验证后再引用。

---

## 7. 内建 md5 自检

`md5.h` 里的 MD5 在 `main()` 开头跑 RFC 1321 的三条标准向量自检
（`""`、`"abc"`、80 字节跨块输入），**不通过直接退出码 3 且拒绝运行**。

理由：指纹实现本身的 bug 绝不能被误读成特征不一致。

---

## 8. 已知风险

| 风险 | 后果 | 对策 |
| :--- | :--- | :--- |
| 服务器只有 OpenCV Python 轮子、没有 C++ 头文件 | 无法编译 | §4 已给探测与安装命令 |
| 两侧 OpenCV 版本不同 | 原语结果可能不同 | 二进制会打印 `opencv=`，与 Python 侧 `cv2.__version__` 对照 |
| 工具链默认注入 `-ffast-math` | 链式合成结果漂移 | CMake 显式关掉并在配置阶段报错 |
| RANSAC 因 RNG 状态不同翻转解 | 锚点矩阵大偏差 | 先看 `[MATS]`：非零即定位到估计器/输入，而非合成层 |
| 误把 step=10 与冻结集对照 | 得到无意义的 −0.0003 | 验证器只对 step=2 开放 G2 |