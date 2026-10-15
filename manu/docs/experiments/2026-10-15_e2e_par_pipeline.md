# 并行流水线 e2e demo：GMC 与 NPU 重叠，输出逐位一致，1.76× 提速

日期：2026-10-15
目标：把特征提取（fit + warp + median，CPU 侧）与 NPU 推理**并行**，
中间加缓存 buffer；CPU 侧保留全部逻辑与 bad case handle 且守住 40 ms；NPU 全力做。
验证口径与前一轮一致：一个短序列，特征通道与 heatmap 完全一致。

---

## 1. 结论

序列 **`34_1`**（537 帧），`anchor_step=10 / --fused gpu / th=0.25`，预热 100 帧剔除。

| 配置 | FPS | 墙钟 | GMC p50 | GMC p95 | GMC max | ≤40 ms |
|---|--:|--:|--:|--:|--:|---|
| 原版（串行 + 落盘） | 5.78 | 93.0 s | 27.33 | 28.11 | 28.65 | **PASS** |
| 去中间落盘（串行） | 8.16 | 65.8 s | 27.71 | 28.56 | 28.99 | **PASS** |
| **去中间落盘 + 并行流水线** | **10.15** | **52.9 s** | 27.40 | 28.10 | 30.71 | **PASS** |

**累计提速 1.76×（5.78 → 10.15 fps），40 ms 约束全程守住。**

拆解：去掉中间落盘贡献 +41%（5.78 → 8.16），并行重叠再贡献 +24%（8.16 → 10.15）。

### 1.1 等价性（三重独立证明）

| 证明 | 内容 | 结果 |
|---|---|---|
| 逐字节（文件路） | shell demo 基准 ref vs 并行版：三通道特征 | **537 / 537** |
| 逐字节（文件路） | heatmap + offset 张量 | **1070 / 1070** |
| 哈希链（跨 I/O 边界） | 文件路 vs 无 I/O 路，逐帧 heatmap+offset 链式 fnv1a | **三档全部 `d5e81977`** |

GMC 特征序列 md5 四档均为 `5caa3c6fc72eb7fad4dd5dfd08ebf4ad`，
`fitgate=0 chain=0 fitdeadline=0 cornercap=0` —— **bad case handle 全部保留且未触发**。

第三行是最关键的一条：它证明**绕开 BMP 落盘没有偷换通道**（见 §3.2），
且不落一个字节就能证明与文件路逐位相同。

---

## 2. NPU 分阶段归因（ms/帧）

| 阶段 | 文件路 | 无 I/O 串行 | 无 I/O 并行 |
|---|--:|--:|--:|
| load（stbi_load 读文件+BMP 解码） | 8.46 | **0.00** | **0.00** |
| letterbox（512→640 双线性） | 27.51 | 25.10 | 28.60 |
| **rknn（inputs_set+run+outputs_get）** | 55.32 | 56.88 | **59.96** |
| dequant | 2.73 | 2.72 | 2.75 |
| peaks | 0.35 | 0.35 | 0.35 |
| **write（写 heatmap/det 到 NFS）** | **27.39** | **0.00** | **0.00** |
| NPU 合计 | 115.44 | 86.32 | 94.45 |

⇒ 带落盘时 **每帧 35.85 ms 纯粹花在 I/O 上**（load 8.46 + write 27.39），占 NPU 段的 31%。
这正是"5.78 fps 不可信"的来源：每帧还要往 NFS 写 2 × 1.2 MB 的 heatmap 文件。

剩下的真实计算瓶颈是 **rknn 59.96 ms**，其次是 **letterbox 28.60 ms**
（`rknn_heatmap_smoke.cpp:193` 的 float 双线性放大，640×640×3）。

---

## 3. 设计要点

### 3.1 结构

```
e2e_core.h     可复用 API（含流式接口）
e2e_core.cpp   #include 两个生产源（main 编译掉），提供批量 + 流式两套入口
e2e_demo.cpp   串行 demo
e2e_par_demo.cpp  串行 / 并行同一个二进制（--mode），共用同一套计时
```

生产者线程（main）跑 GMC，消费者线程跑 NPU，中间是有界队列（`--depth`，默认 6）。

**批量路径被收成流式路径的薄壳**（`extract_features` 循环 `feature_stream_next`，
`run_infer` 循环 `infer_one`），帧体也抽成共享的 `infer_frame()`：
文件路与内存路都进同一个函数。⇒ 串行版和并行版跑的是同一份逐帧代码，
"输出一致"是构造性的，不是重跑验证出来的。

两个 demo 编进**同一个二进制**、由**同一套计时代码**测量，
所以 FPS 对比不可能是两个不同测量口径的产物。

### 3.2 为什么绕开 BMP 要手动复现通道序

`cv::imwrite` 按 BGR 存 BMP，`stbi_load` 读回给的是 RGB。
所以**模型实际看到的是特征管线的 `[Ch2, Ch1, Ch0]`**，不是 `[Ch0, Ch1, Ch2]`。

内存路径必须原样复现这个翻转，否则每张 heatmap 都会变。
`feature_stream_next_rgb()` 就是照这个做的，并用哈希链证明等价。

### 3.3 CPU 绑定：小核和大核不是一回事（关键）

| CPU | 架构 | capacity | max_freq |
|---|---|--:|--:|
| cpu0-3 | A55 | 405 | 1.8 GHz |
| cpu4-7 | A76 | **1024** | 2.352 GHz |

特征提取是算力最重的一段，**必须绑大核**。第一次默认绑成 `cpu-feat 0-3`，
结果 GMC p50 从 27 ms 变成 93 ms（3.4× 劣化），而代码看起来完全正确。

最终默认：`--cpu-feat 4-7`（大核给 GMC）、`--cpu-npu 0-3`（NPU 线程大部分阻塞在
`rknn_run`，小核够用）、`--npu-nice 5`（按 tid 降优先级，见 §5.2）。

绑定到位后，并行版的 GMC p95 从 62.05 ms 回落到 **28.10 ms**，
40 ms 预算从"被抢占打断"变回"稳定满足"。

---

## 4. 脚本与环境

| 脚本 | md5 |
|---|---|
| `manu/pipeline/npu/e2e_core.h` | `188aa40881a0d86686e27aeb011a14ec` |
| `manu/pipeline/npu/e2e_core.cpp` | `bd4dfc7147d6f1d187f6376f40f0e59b` |
| `manu/pipeline/npu/e2e_par_demo.cpp` | `98da3be028f30115d92c4d1fb8754931` |
| `manu/pipeline/npu/e2e_demo.cpp` | `948d92eb6cbea6f13f1db12ed4e2e091` |
| `manu/pipeline/npu/build_e2e_demo_arm64.sh` | `c15145ab5e73323fdb9bf26bd08d0d74` |

板卡：`/root/e2e_par_demo` md5 `59c1713f7525ac9fb20a8004a8a4fae6`、
`/root/e2e_demo` md5 `d644809d7f772a9944e97f4217113f0e`（均为 AArch64）。

### 4.1 可复制命令

```sh
cd manu/pipeline/npu && ./build_e2e_demo_arm64.sh          # 编出两个二进制
scp -P 22 out/e2e_par_demo out/e2e_demo root@192.168.0.64:/root/

# 原版（串行 + 落盘）
ssh -p 22 root@192.168.0.64 'cd /mnt/manu/npu/out/demo/34_1 && /root/e2e_par_demo \
  --raw-root raw --sequence 34_1 --out-root /mnt/manu/npu/out/e2e_verify/io_serial \
  --model /mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn \
  --kernel-dir /root/kernels --fused gpu --anchor-step 10 --th 0.25 \
  --mode serial --hash'

# 最快（无中间落盘 + 并行流水线）
ssh -p 22 root@192.168.0.64 'cd /mnt/manu/npu/out/demo/34_1 && /root/e2e_par_demo \
  --raw-root raw --sequence 34_1 --out-root /mnt/manu/npu/out/e2e_verify/noio_parallel \
  --model /mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn \
  --kernel-dir /root/kernels --fused gpu --anchor-step 10 --th 0.25 \
  --mode parallel --no-io --hash --depth 6'
```

`--hash` 打开逐帧 heatmap+offset 链式哈希（`[HASH]` 行），用于跨 I/O 边界的等价性证明。
`--cpu-feat / --cpu-npu / --npu-nice / --depth / --warmup` 可调。

---

## 5. 过程中的四个 bug（都是"看起来对"的类型）

### 5.1 帧名丢目录前缀

改用构造函数构建 `FeatureStream` 时，漏掉了给帧名拼 `seq_dir` 的一行，
`cv::imread` 拿到裸文件名 → `[FATAL] failed to read 000001.bmp`。

### 5.2 `setpriority(PRIO_PROCESS, 0, n)` 降的是整个进程

想给 NPU 线程降优先级，但 `who=0` + `PRIO_PROCESS` 指的是**当前进程**，
于是 GMC 线程也被降了优先级：**p50 27 ms → 91 ms**，代码看起来完全正确。
改为按 tid：`setpriority(PRIO_PROCESS, syscall(SYS_gettid), n)`。

⇒ **任何 per-thread 调优 API 的 `who=0` 都要确认语义**，它经常指"整个进程"。

### 5.3 双重析构

`feature_stream_close()` 里还留着 placement-new 时代的显式
`s->pipe.~OnlineFeaturePipeline()`，改成构造函数版本后 `delete s` 又析构一次
→ `free(): double free detected in tcache 2`，且只在 stdout 重定向到文件时才显形。
更糟的是 stdout 块缓冲把崩溃前的日志全吞了，定位多花了时间
⇒ 已对两个 demo 都加 `setvbuf(stdout, nullptr, _IOLBF, 0)`。

### 5.4 `--no-io` 写在错误的位置

`if (no_io) dump_heatmap = 0;` 放在了 `if (dump_heatmap) ip.dump_dir = ...` **之后**，
所以"不落盘"模式其实还在写 heatmap：NPU 的 `write` 仍是 25 ms/帧，
FPS 只从 5.74 走到 6.59，几乎看不出区别。修正后 write 归零、FPS 跳到 8.16。

⇒ **配置覆盖必须早于它要影响的那次组装**，否则"关掉了"只是看起来关掉了。

---

## 6. ⚠️ 三条限定

1. **只测了 `34_1` 一个序列、537 帧**，且全程 `anchor_step=10 / --fused gpu / th=0.25`、
   单进程双线程。其余 23 序列、`--fused cpu`、更深缓存均未覆盖。
2. **10.15 fps 依赖 NFS 状态**：产物落在 NFS 上，写吞吐会随网络变化。
   真实产品若写本地存储，`write` 那 27.39 ms 会显著下降，FPS 会更高；
   反之若 heatmap 必须落盘，则无 I/O 路不可用。
3. **`letterbox` 的 28.60 ms 是下一个优化点**（float 双线性放大 640×640×3，
   在 `rknn_heatmap_smoke.cpp:193`）。本轮未动它——动它会改变像素、
   进而可能改变 heatmap，必须重新走一遍逐位验证。

---

## 7. 追加：三级流水线（letterbox 独立成一级）

### 7.1 为什么做

§2 的归因显示 NPU 线程 94.45 ms 里只有 59.96 ms 是硬件，**28.60 ms 是挤在同一条线程上的
CPU letterbox**——硬件在那 28.6 ms 里空转。目标要求「NPU 推理部分全力做」，
把 letterbox 挪走属于目标范围内，不是额外加需求。

### 7.2 结果（序列 34_1，537 帧，`--split-lb`）

| 配置 | FPS | 帧时间 | GMC p50 | GMC max | ≤40ms | hash |
|---|--:|--:|--:|--:|---|---|
| 原版（串行 + 落盘） | 5.74 | 174.3 ms | 27.52 | 28.83 | PASS | `d5e81977` |
| 去落盘（串行） | 8.16 | 122.5 ms | 27.71 | 28.99 | PASS | `d5e81977` |
| 去落盘 + 并行 2 级 | 10.15 | 98.5 ms | 27.40 | 30.71 | PASS | `d5e81977` |
| **去落盘 + 并行 3 级** | **14.07** | **71.1 ms** | 27.93 | 32.10 | **PASS** | **`d5e81977`** |

**累计 2.45×（5.74 → 14.07 fps）。**

> §7.2 原表首行写 5.78 / 172.8 ms，那是另一次运行。复核时按 `io_serial/run.log`
> 的**实测值 5.74 / 93.61 s / 174.3 ms** 取数；单次运行、未重复取方差，两值同量级。

三级各段（ms/帧）：

| 级 | 线程 | 耗时 |
|---|---|--:|
| GMC | cpu 4-5 | 33.19（含 imread + RGB 组装 + 入队） |
| **letterbox** | **cpu 6-7** | **12.23**（之前在 NPU 线程上是 28.60） |
| NPU | cpu 0-3 | 66.99（rknn 60.77 + dequant 2.75 + peaks 0.35 + ~3 开销） |

### 7.3 两点值得记住

1. **letterbox 单独跑只要 12.23 ms，比它挂在 NPU 线程上的 28.60 ms 快 2.3 倍。**
   差额是争抢（内存带宽 + cache），不是算法本身。
   ⇒ **"某段代码慢"要先分清是算法慢还是被抢慢**，方法：把它挪到独占线程再测一次。
2. **等价性是白送的**：三级路径调用的是**同一个 `letterbox()`、同样的参数**，
   只是换了线程。hash `d5e81977` 与两级、文件模式三者一致，可证。

### 7.4 现在的瓶颈与天花板

- 当前帧时间 71.1 ms，NPU 线程 66.99 ms，占 94%。
- 理论上限 = max(33.19, 12.23, 66.99) = 66.99 ms → **14.93 fps**，当前重叠效率 94%。
- **硬地板 = rknn 60.77 + dequant 2.75 + peaks 0.35 = 63.87 ms → 15.7 fps。**
- 再往下只能动 `rknn` 本身：它现在是 `inputs_set + rknn_run + outputs_get` 三者合并统计，
  加 `RKNN_FLAG_COLLECT_PERF_MASK` 可拆开。若大头是每帧 1.2 MB 的 `outputs_get`，有优化余地；
  若是 `rknn_run` 本身，那是模型/驱动的地板。
  **本轮未测，勿声称已定位。**

### 7.5 用法

```sh
/root/e2e_par_demo --raw-root raw --sequence 34_1 --out-root OUT \
  --model /mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn \
  --kernel-dir /root/kernels --fused gpu --anchor-step 10 --th 0.25 \
  --mode parallel --no-io --hash --split-lb --depth 6 \
  --cpu-feat 4-5 --cpu-lb 6-7 --cpu-npu 0-3
```

新增参数：`--split-lb`（letterbox 独立成一级）、`--cpu-lb`（letterbox 线程绑核，默认 6-7）。
`--cpu-feat` 默认改为 `4-5`（三级时两个大核组各给一级），`--cpu-npu` 默认 `0-3`。

---

## 8. 复核：精度对齐的最终核验（不采信历史结论，重新逐位跑一遍）

### 8.1 为什么复核

§1.1 的等价性结论当时是用 `cmp_e2e_outputs.py` 得到的。交付最终报告时按裸 `diff -rq`
重比，**1070/1070 报不一致**。有可能是代码变了，也有可能是比对方法错了。
不能靠"记得当时验过"交差，于是重做——结论是**数据没变，是比对方法错了**（见 §8.3）。

### 8.2 复核结果（全部一致）

| 证据 | 对象 | 结果 |
|---|---|---|
| 特征 BMP 聚合 md5 | `ref` / `io_serial` / `io_parallel` 各 537 张 | 三档全 `f47f91356b6fb9aaae1c2cf9c8260e24` |
| GMC 序列 md5 | 6 档全部 | `5caa3c6fc72eb7fad4dd5dfd08ebf4ad` |
| heatmap+offset 张量 | `ref` vs `io_serial` | 535 帧 × 2 = **1070/1070 逐字节一致** |
| heatmap+offset 张量 | `ref` vs `io_parallel` | 535 帧 × 2 = **1070/1070 逐字节一致** |
| 检出 TSV | `ref` vs `io_serial` / `io_parallel` | 535 可比帧，**0 行差异**，均 `detections=547` |
| bad case handle | 6 档全部 | `fitgate=0 chain=0 fitdeadline=0 cornercap=0` |

### 8.3 坑一：帧名偏移，`diff -rq` 必然全报不一致

`ref` 由 shell demo 产出、**从第 3 帧开始**（`frame_34_1__000003_hm.bin`，共 535 帧）；
`e2e_par_demo` 从第 0 帧起（`34_1_34_1_000000_hm.bin`，共 537 帧）。
两者帧名不同，裸 `diff -rq` 把「名字对不上」当成内容不一致 → 1070/1070 全错。

正确映射是 **`ref = serial + 1`**。直接 `sed` 去猜偏移不可靠，**扫一遍 0~4 五个候选**：

| 偏移假设 | io_serial vs ref | io_parallel vs ref |
|---|--:|--:|
| ref = serial + 0 | 1068 / 1068 不一致 | 1068 / 1068 不一致 |
| **ref = serial + 1** | **0 / 1070 不一致** | **0 / 1070 不一致** |
| ref = serial + 2 | 1070 / 1070 不一致 | 1070 / 1070 不一致 |
| ref = serial + 3 | 1070 / 1070 不一致 | 1070 / 1070 不一致 |
| ref = serial + 4 | 1068 / 1068 不一致 | 1068 / 1068 不一致 |

只有 +1 一档为 0，邻位全错 ⇒ **对齐是真解，不是退化匹配**。

> 方法论：**比对前先扫偏移确定唯一解，再看唯一解是否为零差异**。
> 只看"某个偏移下是 0"不够——全零文件在任何偏移下都是 0。

### 8.4 坑二：哈希链没有和落盘档接上（补上了）

三级档 `split_lb` / `v2_noio_parallel` 用 `--no-io`，不落任何字节，精度证据只有链值 `d5e81977`；
而落盘档 `io_serial` / `io_parallel`（已证 == `ref`）当时**没开 `--hash`**，日志里没有链值。
⇒ 两组证据各证一半，中间是断的。

用独立实现重算来搭桥。按 `e2e_core.cpp:485-487` 的规则
（初值 2166136261，逐帧 `h ^= fnv1a(hm); h ^= fnv1a(off)`，`fnv1a` 见
`rknn_heatmap_smoke.cpp:75`），直接读落盘档的 `.bin` 重算：

```
io_serial  (帧 0..536, 537 帧): d5e81977   (659,865,600 字节)
io_parallel(帧 0..536, 537 帧): d5e81977   (659,865,600 字节)
noio_serial / noio_parallel / split_lb / v2_noio_parallel  日志链值: d5e81977
```

**桥搭上了：三级档的链值与已证 == ref 的落盘档逐位相同 ⇒ 三级档 == `ref`，传递可证。**

复算脚本 `/tmp/fnvchain.c`（FNV 是字节级串行链，659 MB 用 Python 跑不动，故写 C）：
`gcc -O2 -o fnvchain fnvchain.c`，用法 `fnvchain <hm目录> <文件名前缀> <首帧> <末帧>`。

### 8.5 排除的误导产物

产物目录 `/mnt/manu/npu/out/e2e_verify/` 里有两个**不能进结论**的目录：

1. `noio_serial` / `noio_parallel` —— **修 `--no-io` 位置 bug 之前的旧运行**。
   头部写 `io=NONE`，但 `NPU breakdown` 里 `write` 仍是 25.37 / 25.18 ms，
   FPS 被低估成 6.59 / 8.16。表内用的是修复后的 `v2_noio_serial`（8.16）/ `v2_noio_parallel`（10.15）。
2. `e2e/` —— 更早一轮 `e2e_demo` 的产物，与 `ref` 在**所有偏移下都对不上**（1068/1068），
   模型/配置与本轮不同。**不参与本轮结论**。

判据：**头部声明的参数与 breakdown 里的数字必须自洽**，不自洽的目录直接作废。

### 8.6 可复制完整命令与环境

环境：host `192.168.0.115`（NFS 导出 `/run/media/manu/SSD960-LOCAL/nfs`）；
板卡 `ssh -p 22 root@192.168.0.64`（RK3588，8 核 A55×4@1.8G + A76×4@2.352G）；
比对在 host 上做（板卡 busybox ash 无 `$(k-1)`，且 NFS 就地可读，不用过 ssh 逐帧）。

```sh
V=/run/media/manu/SSD960-LOCAL/nfs/npu/out/e2e_verify
ls $V                      # 产物目录清单

# 帧偏移扫描（唯一解 = ref = serial + 1，见 §8.3）
python3 - <<'PY'
import os,hashlib,re
base="/run/media/manu/SSD960-LOCAL/nfs/npu/out/e2e_verify"
def md5(p):
    h=hashlib.md5()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()
ki=lambda n:int(re.search(r'(\d{6})_hm\.bin$',n).group(1))
ref=sorted(ki(n) for n in os.listdir(f"{base}/ref/hm") if n.endswith("_hm.bin"))
for s in ("io_serial","io_parallel"):
    sk=set(ki(n) for n in os.listdir(f"{base}/{s}/hm") if n.endswith("_hm.bin"))
    for off in range(5):
        common=[k for k in ref if (k-off) in sk]
        bad=sum(1 for k in common for suf in("hm","off")
                if md5(f"{base}/{s}/hm/34_1_34_1_{k-off:06d}_{suf}.bin")
                   !=md5(f"{base}/ref/hm/frame_34_1__{k:06d}_{suf}.bin"))
        print(f"{s}(帧k) vs ref(帧k+{off}): 张量={len(common)*2} 不一致={bad}")
PY

# 检出 TSV（同一映射）
python3 - <<'PY'
base="/run/media/manu/SSD960-LOCAL/nfs/npu/out/e2e_verify"
rows=lambda p:{f[0]:"\t".join(f[1:]) for f in (l.rstrip("\n").split("\t") for l in open(p)) if len(f)>=2}
ref=rows(f"{base}/ref/npu_det.tsv")
for s in ("io_serial","io_parallel"):
    t=rows(f"{base}/{s}/npu_det.tsv")
    bad=sum(1 for k,v in ref.items()
            if (kn:=f"34_1_{int(k.split('__')[1][:6])-1:06d}.bmp") in t and t[kn]!=v)
    print(s,"det 不一致帧数 =",bad)
PY

# 链式哈希独立重算（搭 §8.4 的桥）
gcc -O2 -o /tmp/fnvchain /tmp/fnvchain.c
/tmp/fnvchain $V/io_serial/hm   34_1_34_1_ 0 536   # -> d5e81977
/tmp/fnvchain $V/io_parallel/hm 34_1_34_1_ 0 536   # -> d5e81977
```

### 8.7 rknn 三段拆分（已改，**尚未上板跑**）

`e2e_core.cpp:413` 原来把 `rknn_inputs_set / rknn_run / rknn_outputs_get` 加总成一个
`rknn_ms`，拆不开。已改为三段各自计时（`inputs_set_ms` / `run_ms` / `outputs_get_ms`），
`rknn_ms` 保留作三者之和以兼容旧报告口径。`--rknn-perf`（`RKNN_FLAG_COLLECT_PERF_MASK`）
路线判定为不可用：两次尝试读数都不合理，且该 flag 会拖慢运行（14.07 → 13.16 fps）。

| 源码 | 本地 md5 | host 镜像 md5 |
|---|---|---|
| `e2e_core.h` | `846bba0a70d31d4d204f123e12b617f5` | 同左 |
| `e2e_core.cpp` | `7ab66dc57431fd96d1ba534afd364612` | 同左 |
| `e2e_par_demo.cpp` | `3a0e7e05b6ededa0317627b0ccf839cf` | 同左 |
| `out/e2e_par_demo`（新编） | `cfc750ad70e2b8626938367c1e1ed824` | — |

**复核更正**：本轮此前记录的「产出二进制 md5 = `5599bcde…`」是**错的**。板上
`/root/e2e_par_demo` 最后一次推送时间为 **17:11:59**，实际 md5 为
**`d314f84ecb1c2711d69d5cb302cdd0e6`**；而 `io_serial`(17:50:58) ~ `split_lb`(18:07:29)
全部在该推送之后，故 **§7/§8 的全部数字均由 `d314f84e…` 产出**。
新编二进制 `out/e2e_par_demo` md5 `cfc750ad70e2b8626938367c1e1ed824`（18:14:55 本地）
**从未上过板**。

`rknn 60.77 ms` 内部构成**仍未定位**，勿声称已定位。

### 8.8 本节新增限定

1. **只测 `34_1` 537 帧、每档单次运行**，未重复取方差；原版档两次运行 5.78 / 5.74 fps。
2. **三级档未在板上落文件做逐位复核**，靠链值传递证明（链值已用独立实现重算验证）。
3. 复核暴露的三个坑（偏移、证据链断裂、误导产物目录）都是**比对方法**问题，不是代码问题；
   但它们说明：等价性结论必须附带**可复现的比对命令**，否则无法被重新验证。
