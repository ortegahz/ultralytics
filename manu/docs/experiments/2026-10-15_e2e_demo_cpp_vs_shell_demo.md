# e2e_demo：单进程纯 C++ 端到端 demo，与 shell demo 输出逐字节一致

日期：2026-10-15
目标：不用 `run_hm_demo.sh` 的 shell 编排，单写一个 `e2e_demo.cpp`（纯 C++ 执行），
在同一个短序列上验证两个 demo 的输出**完全一致**——三通道特征 + 推理 heatmap。

---

## 1. 结论

序列 **`34_1`**（537 帧，真实 val 序列，有 GT）。

| 产物 | 一致数 | 结果 |
|---|--:|---|
| 三通道特征 BMP（537 张） | **537 / 537** | 逐字节一致 |
| heatmap + offset 张量（535 帧 × 2，float32 `.bin`） | **1070 / 1070** | 逐字节一致 |
| 检出 TSV（535 行） | **535 / 535** | 逐行一致 |

两边的 GMC 特征序列 md5 完全相同：`5caa3c6fc72eb7fad4dd5dfd08ebf4ad`
且两边回退计数均为 0（`fitgate=0 chain=0 fitdeadline=0 cornercap=0`）。

检出总数 ref 545 / e2e 547，差 2 恰为 e2e 多出的 push 0、1 两帧——参考的 NPU 阶段从
id 3 起（见 §4 帧号映射），e2e 从 push 0 起，多出的两帧没有对照，非差异。

**「完全一致」由构造保证，不是事后凑出来的**：e2e_demo 不重新实现任何算法，
它 `#include` 两个生产源（main 被编译掉）并调用同一批类与函数。

---

## 2. 做法：核心代码复用，而不是重写

```
e2e_core.h     可复用 API 声明（后续并行 demo 直接调）
e2e_core.cpp   API 实现：#include 两个生产源，main 编译掉
e2e_demo.cpp   纯 C++ 驱动，串行调两个 API
```

`e2e_core.cpp` 复用：

- 阶段 A（`../cpp/gmc_stream_ocl.cpp`）：`OnlineFeaturePipeline` / `FusedGpu` /
  `find_sequence_folder` / `list_dir` / `mkdir_p`
- 阶段 B（`rknn_heatmap_smoke.cpp`）：`letterbox` / `run_once` / `dequant` /
  `extract_peaks` / `Ctx` / `write_bin` / `fnv1a`

对既有生产文件**只加了一行 `#ifndef ..._NO_MAIN` 包裹 main**，函数体一字未改。

### 2.1 guard 重构被证明是语义空操作

加 guard 后重新交叉编译两个原二进制，与板上**原有那份**逐字节一致：

| 二进制 | 重编后 md5 | 板上原版 md5 | |
|---|---|---|---|
| `rknn_heatmap_smoke` | `170379506adb4aeaef2f5a5f8ce65cc2` | `170379506adb4aeaef2f5a5f8ce65cc2` | 一致 |
| `gmc_stream_ocl_arm64` | `aa99da811356694d66a0bce6de108b92` | `aa99da811356694d66a0bce6de108b92` | 一致 |

这是最强的一条：编译器产出的机器码没变，所以「同一份代码」不是说法而是事实。

---

## 3. 为什么 shell demo 需要改名、e2e_demo 不需要

`run_hm_demo.sh` 用两个独立进程串起来，中间必须由 host 侧 python 建符号链接改名：
GMC 按 **push 序号**命名，NPU 要按**数据集帧号**读。两个进程必须就文件名达成一致。

e2e_demo 是单进程，阶段 A 直接把特征路径交阶段 B，**这个协调步骤结构性地消失了**。
所以两边的帧号不是一一对应，比对时必须显式映射，否则会得到假的「一致」或假的「差异」。

---

## 4. 帧号映射（PUSH_OFFSET=1）

| | 命名 | 覆盖范围 |
|---|---|---|
| 参考 GMC 产出 | `feat/34_1_<push:06d>.bmp` | push 0..536 |
| 参考改名后 | `feat_named/34_1__<id:06d>.bmp`，`id = push + 1` | id 3..537 |
| 参考 NPU 阶段 | 读 id 3..537 | = push **2..536**，535 帧 |
| e2e_demo NPU 阶段 | 读 `feat/34_1_<push:06d>.bmp` | push **0..536**，537 帧 |

⇒ **参考 id N ≡ e2e push N−1**。比对脚本 `cmp_e2e_outputs.py` 显式做这个换算，
并把只有单边存在的帧单独列为 note 而不是悄悄补齐。

---

## 5. 脚本与环境

| 脚本 | md5 |
|---|---|
| `manu/pipeline/npu/e2e_core.h` | `620e40e2dffef0a95a3735fc142347cb` |
| `manu/pipeline/npu/e2e_core.cpp` | `1ec64a76375dea41a7c24fb63e485554` |
| `manu/pipeline/npu/e2e_demo.cpp` | `83d4ed506fcc540d9d621fa8ec03fb48` |
| `manu/pipeline/npu/build_e2e_demo_arm64.sh` | `0e1bf424236f42ed5d42c0f43f83c3f6` |
| `manu/pipeline/npu/cmp_e2e_outputs.py` | `0a726d9b70bdb572402b84fdaff9fbc5` |
| `manu/pipeline/npu/rknn_heatmap_smoke.cpp`（仅加 guard） | `ca6cf48023ea45629e57f08679a45f9d` |
| `manu/pipeline/cpp/gmc_stream_ocl.cpp`（仅加 guard） | `ecdd435b6b1f4fd22ca64ea59188fb36` |

板卡二进制：`/root/e2e_demo` md5 `3a321482ec090489f734dfac80c0aeeb`（AArch64）

环境：板卡 `ssh -p 22 root@192.168.0.64`（RK3588 / Mali-G610，8 CPU 核），
模型 `/mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn` md5 `ef302edb1b5aff1d376204bb7ad77550`，
OpenCL kernel `/root/kernels/`，交叉工具链
`gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu`，
OpenCV 4.10.0 arm64 交叉构建，librknnrt.so 用板端 2.3.0。

### 5.1 可复制命令

```sh
# ---- 构建 + 部署 ----
cd manu/pipeline/npu && ./build_e2e_demo_arm64.sh
scp -P 22 out/e2e_demo root@192.168.0.64:/root/e2e_demo

# ---- 跑 e2e_demo（单进程，纯 C++）----
ssh -p 22 root@192.168.0.64 '
cd /mnt/manu/npu/out/demo/34_1
/root/e2e_demo --raw-root raw --sequence 34_1 \
  --out-root /mnt/manu/npu/out/e2e_verify/e2e \
  --model /mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn \
  --kernel-dir /root/kernels --fused gpu --anchor-step 10 --th 0.25'

# ---- 逐字节比对 ----
python3 manu/pipeline/npu/cmp_e2e_outputs.py \
  --ref-root  /run/media/manu/SSD960-LOCAL/nfs/npu/out/e2e_verify/ref \
  --e2e-root  /run/media/manu/SSD960-LOCAL/nfs/npu/out/e2e_verify/e2e \
  --sequence 34_1 --ref-tag frame --e2e-tag 34_1
```

产物目录：`<out-root>/feat/`（三通道特征 BMP）、`<out-root>/hm/`（`*_hm.bin` / `*_off.bin`）、
`<out-root>/npu_det.tsv`。

---

## 6. 为后续并行 demo 预留的能力

`e2ecore::extract_features()` 返回该序列写出的特征路径列表（push 顺序）；
`e2ecore::run_infer()` 接受任意 `frames` 子集。因此：

- 并行 GMC：多个 worker 各自 `extract_features` 不同序列，或按帧切分
- 并行 NPU：每个 worker `run_infer({..., frames = shard})`

**已知的硬约束**（memory §二十·20.3 实测）：RK3588 的 NPU 并行度上限是 **2**，
第 3 个并发上下文会撞 RKNPU 6 秒 job timeout → 驱动 soft reset → 在飞上下文全部陪葬。
6 / 9 进程不但不快，吞吐还会劣化 3 倍。设计并行 demo 时必须记住这条。

---

## 7. 过程中的两个事故（勿复现）

1. **Mali 驱动锁泄漏导致板卡卡死。** 先前做 NPU 并行度扫描时触发了 RKNPU 软复位与
   `BUG: spinlock bad magic on CPU#2, kswapd0/146`（debug kernel oops），此后任何打开
   `/dev/mali0` 的进程都永久阻塞在 `rwsem_down_write_slowpath`（D 状态，`kill -9` 无效，
   load average 88）。**必须断电重启才能恢复**；`reboot` 命令在该状态下不可靠，
   最终是手动断电解决的。重启后 NFS 挂载会丢，需手工重挂：
   ```sh
   mount -t nfs4 -o rw,vers=4.0,soft,proto=tcp,timeo=5,retrans=2 \
     192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu
   ```
2. **比对脚本自身有过两个 bug，都制造了假信号**，已修：
   - 先把参考的 `_hm.bin` 拿去和 e2e 的 `_off.bin` 比（offset 张量是 heatmap 的两倍大，
     409600 vs 819200），报出 535/1070「全不一致」，实际全一致；
   - 解析帧号时漏剥 `.bmp` 后缀，`int('000003.bmp')` 直接抛异常。
   ⇒ 比对脚本本身必须有「同种类比同种类」的显式断言，否则第一个跑出来的数字都不能信。
