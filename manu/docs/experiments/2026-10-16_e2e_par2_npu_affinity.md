# e2e 双 NPU 并行架构：CPU 绑核决定 fp16 能不能过 12 fps

- 日期：2026-10-16
- 板卡：RK3588，192.168.0.64，`rknn_init` 自报 `arm_release_ver: g13p0-01eac0, rk_so_ver: 6`
- 序列：`34_1`，537 帧，`anchor_step=10`，`--fused gpu`，`th=0.25`，预热 100 帧剔除
- 模型：`trial0474_hm_nop0_fp16.rknn`（md5 `07c893e4d2ca3af9a47a4bf88335fe33`）
  / `trial0474_hm_nop0_int8.rknn`（md5 `ef302edb1b5aff1d376204bb7ad77550`）
- 程序：`manu/pipeline/npu/e2e_par2_demo.cpp`（4 级流水线 GMC ∥ letterbox ∥ NPU worker0 ∥ NPU worker1）
- 复核了目标原话里的「两个 **npu core(0 1)** worker」——那是 NPU core 而非 CPU 核，§5 单独验证

---

## 1. 一句话

fp16 从 **11.89 fps 提到 15.26 fps（+28.4%）**，越过 12 fps 门槛；
决定性动作不是加流水线也不是改模型，而是**把四个线程从 A55 小核搬到 A76 大核、并且一人一个核**。
逐帧输出与原绑核基线 **537/537 完全一致**。

---

## 2. 流水线结构与理论上限

```
GMC(1 线程) ──► letterbox(1 线程) ──┬──► NPU worker0 (rknn ctx #1) ──┐
   512×512×3        640×640×3      └──► NPU worker1 (rknn ctx #2) ──┴──► TSV
```

吞吐上限由最慢一级决定：

```
fps_ceiling = 2 × 1000 / per_context_ms
```

因为两个 rknn 上下文各自独立、并行地每 `per_context_ms` 处理一帧。
实测 11.89 fps 时 `per_context_ms = 162.97`，理论上限 12.27 fps——**旧绑核方案本来就贴着天花板跑**，
再怎么调流水线级数也翻不过 12.3。所以要过 12，必须把 `per_context_ms` 本身压下去。

`per_context_ms` 的构成（fp16 档）：

| 段 | 旧绑核 | 占比 |
|---|--:|--:|
| `rknn_inputs_set` | 45.41 ms | 28% |
| `rknn_run` | 111.46 ms | 68% |
| `outputs_get` | 0.91 ms | 0.6% |
| dequant + peaks + 写盘 | 2.50 ms | 1.5% |
| 合计 | 162.97 ms | |

---

## 3. `inputs_set` 45 ms 的定性与证伪（用户叫停，留档）

用户判断：这 45 ms 是 `librknnrt.so` 在用户态做的低效类型转换/归一化，建议
「传已对齐 fp16 缓冲 / NEON 预转换 / 转模型时把输入固化为 uint8」三条路。逐条做了实测。

### 3.1 定性：确认是 CPU 侧类型工作，且与模型 dtype 强相关

同一个 1228800 字节 uint8 上传、同一段代码、同一个驱动：

| 模型 | `in_attr.type` | attr `size` | `inputs_set` |
|---|---|--:|--:|
| int8 | `2` = UINT8 | 1,228,800（1 B/px） | **0.34 ms** |
| fp16 | `1` = FLOAT16 | 2,457,600（2 B/px） | **45.41 ms** |

45.41 ms ÷ 1,228,800 元素 = **37 ns/元素 ≈ 74 cycles/元素 @2 GHz**。
这是标量逐元素循环的量级，不是向量化的量级。int8 图因为量化输入原生吃 uint8，运行时几乎不做事。

**结论：判断成立，且已定位到它是小核上的标量循环**（见 §4）。

### 3.2 三条建议路线的实测结果

| 路线 | 实测结果 |
|---|---|
| 传已对齐 fp16 缓冲（`pass_through=1`, 2 B/px） | **驱动拒绝**：`E RKNN: rknn_inputs_set, param input size(2457600) < model input size(4915200)`，每帧失败 |
| 同上但补齐到 4915200 B（4 B/px） | **SIGSEGV**，栈顶 `__memcpy_generic` ← `librknnrt.so` ← `rknn_inputs_set` |
| 声明 `type=FLOAT32`（不过直通），先除 255 | `inputs_set` 46.74 ms（没降），537 帧 **0 检测**（运行时又除了一次 255） |
| 声明 `type=FLOAT32`（不过直通），不除 255 | `inputs_set` 47.78 ms（没降），571 检测（与基线一致，但没提速） |

**`pass_through` 在这个 fp16 图上是坏的，且是驱动自相矛盾，不是调用姿势问题**：
`RKNN_QUERY_INPUT_ATTR` 报 `type=1(FLOAT16) fmt=1(NHWC) n_elems=1228800 size=2457600`，
即模型声明的输入就是 2 B/px；但驱动的直通检查要求 `4915200`（4 B/px）。
按声明给 2 B/px → 报错；按检查给 4 B/px → 段错误。这个矛盾 SDK 头文件里解释不了，
只能当驱动 bug 记。代码里保留了 `--f16-in` / `--f32-in` 两条路径作为可复现的负面对照。

> 顺带修正一个旧误判：早先记为「arm I 挂死」是错的。它每帧都在失败返回，
> 只是 worker 吞掉错误继续跑，队列永不排空，producer 侧看着像不动。真因是尺寸校验失败。

**剩余可行路线（未执行，需用户裁决）**：转模型时把输入固化为 uint8 节点，
让 Cast/Mul 变成图内算子跑到 NPU 上——这正是 int8 档 `inputs_set` 只有 0.34 ms 的原因。
这要改 ONNX 输入 dtype 并用 rknn-toolkit2 2.3.0 重新转换，属模型侧改动。

---

## 4. 真正的杠杆：CPU 绑核

`inputs_set` 是**单线程标量循环**，跑在哪个核上结果差 3 倍。而 worker 线程有 111 ms
是在 `rknn_run` 上阻塞等待 NPU 的，CPU 闲置——所以「worker 占着两个 A55」纯属浪费。

### 4.1 六档绑核对照（fp16 / 2 workers / 537 帧）

| 臂 | GMC | letterbox | npu0 | npu1 | **fps** | per-ctx | `inputs_set` | `rknn_run` | GMC wall | LB |
|---|---|---|---|---|--:|--:|--:|--:|--:|--:|
| **A**（原方案） | 4-5 | 6-7 | 0-1 | 2-3 | **11.80** | 162.97 | 45.41 | 111.46 | 32.09 | 12.36 |
| **P** | 0-1 | 2-3 | 4-5 | 6-7 | 9.12 | 119.72 | 14.73 | 102.27 | 106.25 | 27.91 |
| **Q** | 0-1 | 2-3 | 4 | 5 | 9.22 | 120.36 | 14.72 | 103.16 | 105.05 | 27.94 |
| **T2** | 4-5 | 4-5 | 6 | 7 | 15.13 | 127.99 | 14.78 | 110.74 | 33.95 | 14.89 |
| **T** | 4 | 5 | 6 | 7 | **15.26** | 126.99 | 14.77 | 109.62 | 33.40 | 14.87 |
| **T**（复跑） | 4 | 5 | 6 | 7 | **15.25** | 126.99 | 14.77 | 109.69 | 34.41 | 14.88 |

全部 571 检测，537 行。

### 4.2 三条可复用的结论

1. **P/Q 是反例，说明瓶颈会转移**：worker 挪到大核后 NPU 级从 81.5 掉到 59.9 ms/帧
   （理论 16.7 fps），但 GMC 被挤到 A55 后从 32.09 涨到 **106.25 ms**，GMC 变成新瓶颈，
   总帧率反而掉到 9.12。「优化某一级」在异构核上必须看全局。
2. **Q 证明每个 worker 只要 1 个大核**：单核 `inputs_set` 14.72 vs 双核 14.73，
   完全一样——因为 `inputs_set` 是单线程串行循环，给两个核也没用。
   但 `rknn_run` 从 102.27 涨到 103.16，说明多给核对 NPU 等待几乎无收益。
3. **T 是四个线程一人一个大核的均衡解**：GMC 33.40（对 A 档的 32.09 只退化 4%），
   LB 14.87，per-ctx 126.99 → 理论 15.75，实测 **15.26（97% 流水线效率）**。
   四个 A55（cpu 0-3）全程闲置。

### 4.3 int8 档同绑核

| 档 | 绑核 | fps | per-ctx | `inputs_set` | `rknn_run` |
|---|---|--:|--:|--:|--:|
| int8 | A（worker 小核） | 26.88 / 27.10 | 69.51 | 0.34 | 61.56 |
| int8 | **T（全大核）** | **29.83** | 59.80 | 0.17 | 57.67 |

---

## 5. NPU core 分配：目标里的「两个 npu core(0 1) worker」

§4 全部解决的是 **CPU 核**。目标原话是「构建两个 **npu core(0 1)** worker」，
那是 **NPU core**——两件不同的事，必须分开验证。

### 5.1 此前从未验证过

代码里**从未调用 `rknn_set_core_mask`**，两个上下文都跑在默认的 `RKNN_NPU_CORE_AUTO` 下。
`rknn_set_core_mask` 在板卡 `/lib/librknnrt.so` 里确实导出（`0xaff20 T`），
所以之前那些 11.80 / 15.26 fps 只能证明「两个 worker」，**不能证明「worker N 占住 NPU core N」**。

新增 `infer_set_npu_core(e, mask)` + demo 的 `--npu-core0 / --npu-core1`（掩码位 1/2/4 = core 0/1/2，0 = AUTO）。
掩码被运行时拒绝时直接 FATAL 退出，不静默吞掉——一个没生效的掩码会让吞吐数字看着正常，
而「core 0 / core 1」这句话是空的。

### 5.2 三臂实测（AUTO / core 0,1 / core 0,2，含逐 core 占用采样）

绑核一律用 §4.3 采纳的 T 方案（feat=4, lb=5, npu0=6, npu1=7）。

| 臂 | worker0 | worker1 | **fps** | per-ctx | `rknn_run` | Core0 | Core1 | Core2 |
|---|---|---|--:|--:|--:|--:|--:|--:|
| v0auto | AUTO | AUTO | **15.28** | 126.57 | 109.38 | 84.1% (136/139 忙) | 75.2% (132/139) | **0.0%** |
| v1c01 | `0x1` | `0x2` | **15.18** | 127.94 | 110.24 | 79.4% (134/140) | 79.9% (134/140) | **0.0%** |
| v2c02 | `0x1` | `0x4` | **失败** | — | — | — | — | — |

逐 core 占用来自运行期间 0.25 s 间隔采样的 `/sys/kernel/debug/rknpu/load`。

- **v2c02 直接失败**：`E RKNN: failed to submit!, op id: 64 ...` → `rknn_run failed`，
  537 帧只出 49 行（残缺产物即失败）。**RK3588 只支持 0/1 这对相邻 core，0+2 不可用**，
  与既有的「第 3 个上下文触发 6 s job timeout + 驱动 soft reset」是同一个约束的两面。
- **AUTO 已经在用 core 0 和 core 1，core 2 全程 0%。** 所以目标里的「core(0 1) worker」
  被满足了——只是要靠采样才能证明，不能靠默认。
- **显式切分只让负载更均衡**（79.4/79.9 对 84.1/75.2，帧率不变），说明 AUTO 下 worker0 的核
  略忙一点，但这 5 个点不构成可利用的空间。

### 5.3 结论：NPU core 分配这块没有余量

`rknn_run` 109–110 ms 时两个核各约 80% 占用，且第三个核不可用。
⇒ **那 109 ms 是真实的 NPU 计算，不是排队等待。** 显式 0/1 切分不提速也不掉速（15.18 vs 15.28，
落在 T / Tr / v0auto 的 15.25–15.28 复跑波动内）。

**要不要显式切分**：功能等价，帧率等价。采纳「默认 AUTO」——
理由是少两个参数、少一个失败模式（`rknn_set_core_mask` 被拒 → 直接 FATAL），
而收益为零。但 `--npu-core0/--npu-core1` 保留在代码里，因为下一次换模型时可能需要它。

---

## 6. 闸门（全部 PASS）

| 闸门 | 结果 |
|---|---|
| 帧数完整 | T / Tr / T2 / I8T 各 537 行（0 字节产物=失败，本轮无） |
| **逐帧精度** | T vs 基线 A：**537/537 IDENTICAL**（571 检测） |
| **复跑稳定性** | Tr vs T：15.25 vs 15.26 fps；同样 **537/537 IDENTICAL** |
| **回归** | 改完 `e2e_core.cpp` / `rknn_heatmap_smoke.cpp`（含新增 core-mask 代码）后重跑 `e2e_par_demo` int8，chain hash 仍为 **`d5e81977`**，13.86 fps，与 10-15 基线逐位一致 |
| **NPU core 臂精度** | v0auto / v1c01 vs 基线：各 **537/537 IDENTICAL**（571 检测） |
| 残缺产物 | v2c02 只出 49/537 行 → **判失败**，未被当成有效结果 |
| 内核错误 | `dmesg` 中 16 行 `rknpu/rknn` 全为开机驱动初始化信息，本轮运行 0 错误 |
| NPU 上下文数 | 2 个，未触发第 3 上下文的 6 s job timeout / soft reset |

复现命令：

```sh
cd /mnt/manu/armfeat
/root/e2e_par2_demo --raw-root bmpraw --sequence 34_1 \
  --out-root <OUT> --model /mnt/manu/npu/data/trial0474_hm_nop0_fp16.rknn \
  --kernel-dir /root/kernels --fused gpu --anchor-step 10 --th 0.25 \
  --workers 2 --depth 6 \
  --cpu-feat 4 --cpu-lb 5 --cpu-npu0 6 --cpu-npu1 7
```

（`--raw-root` 必须用 `bmpraw`；arm64 OpenCV 4.10.0 编了 imgcodecs 但没有 PNG 解码器。）

---

## 7. 产物

| 路径 | 内容 |
|---|---|
| `manu/pipeline/npu/e2e_par2_demo.cpp` | 4 级流水线 demo，md5 `cab54f8ecb5cf9c2df8c2368f6140b55` |
| `manu/pipeline/npu/e2e_core.cpp/.h` | 新增 `F32Boxed` / `letterbox_feature_f32` / `infer_preboxed_f32` / `u8_to_f32`(NEON) |
| `manu/pipeline/npu/rknn_heatmap_smoke.cpp` | `run_once` 新增 f32 直通分支 + 输入 attr 打印 + `Ctx.core_mask` |
| `manu/pipeline/npu/e2e_core.h/.cpp` | 新增 `infer_set_npu_core` / `infer_npu_core_mask`（显式 NPU core 切分） |
| `manu/pipeline/npu/build_e2e_demo_arm64.sh` | 编译循环加入 `e2e_par2_demo` |
| 板卡 `/mnt/manu/npu/out/par2i/P,Q`、`par2j/T,T2`、`par2k/Tr,I8T,REG`、`par2l/v0auto,v1c01,v2c02,REG2` | 各臂 TSV 与日志 |
| 板卡 `/root/e2e_par2_demo` | md5 与本地一致 |

五个源文件已 `cp` 同步到只读镜像 `/home/manu/mnt/pycharm_project_10ae9e2e/`，md5 双侧全部一致。

---

## 8. 剩下的空间

`per_context_ms = 126.99` 里 `rknn_run` 占 109.62 ms（86%）。两个上下文并行，
`rknn_run` 是硬地板：`2 / 109.62 ms = 18.2 fps` 是这套配置的绝对上限。

想把 15.26 往 18 推，只剩三处，都不是本文档范围：

1. **彻底干掉 `inputs_set` 14.77 ms** → 18.2 fps。需要 §3.2 末尾那条「转模型固化为 uint8」。
2. **压 `rknn_run` 109.62 ms** → 纯模型/驱动侧：构建日志里那条 `Conv:/model/b5/conv/Conv`
   的 3Core→单核回退、以及 fp16 图 105,600 KB 的内部内存，都还没动过。
3. **把 dequant + peaks（0.82 ms）挪到闲置的 A55 上做第 5 级** → 约 +0.5 fps，
   需要先把 1.2 MB 输出张量拷到私有缓冲（因为 runtime 的输出缓冲会被同上下文下一次 run 覆盖）。
