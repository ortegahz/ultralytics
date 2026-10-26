# fp16 模型的输入路径：零拷贝实验的**否定结论** + 真正可用的杠杆（CPU 绑核）

> 日期：**2026-10-23** · 执行机：**板卡 RK3588 `192.168.0.64`**（runtime `api=2.3.0`，driver 0.9.8）
> 模型：`/userdata/ptq_models/p0_off_A_none.rknn`（trial_0080 的 f16/`--no-quant` 版，6,326,305 B）
> 代码：`manu/pipeline/npu/rknn_heatmap_smoke.cpp`（全部 **opt-in**；默认路径与已认证旧二进制**逐位一致**）
> 产物：板卡 NFS `/mnt/manu/ptq_iomem/` · 流水线产物 `/mnt/manu/ptq_iomem/e2e_{old,new,iomem,iomemf16}/`

---

## 0. 结论（先说否定的那条）

> **在 RK3588 + librknnrt 2.3.x 上，驱动的 `rknn_inputs_set`（uint8 NHWC 上传）就是最优输入路径。**
> 主机侧零拷贝（`rknn_set_io_mem`）**能把可见的 `inputs_set` 从 11–34 ms 压到 0.78 ms，
> 但驱动随之把"暂存输入"挪进 `rknn_run`，代价 +13.6 ms（fp16）/ +6.2 ms（int8）
> ⇒ **整帧净亏 3.2 ms（fp16）/ 6.2 ms（int8）**。
>
> **真正能省下来的杠杆只有一个：CPU 绑核** —— `inputs_set` **45.41 ms → 11.15 ms（4.1×）**，
> 零成本、零精度影响、流水线里本来就有（T 档绑核）。

⚠️ 本文件早期版本只比较了"输入侧"这一个指标，写过"43×"的结论——**那是错的**，
因为驱动把输入暂存的开销挪进了 `rknn_run`。本版是**更正后**的结论，证据在 §6–§8。

---

## 1. 背景：fp16 的 `inputs_set` 为什么贵

同图同码，2026-10-16 实测（`docs/experiments/2026-10-16_e2e_par2_npu_affinity.md` §3.1）：

| 模型 | `in_attr.type` | attr size | `inputs_set` |
|:--|:--|--:|--:|
| int8 | `UINT8` | 1,228,800（1 B/px） | **0.34 ms** |
| fp16 | `FLOAT16` | 2,457,600（2 B/px） | **45.41 ms**（A55）/ 11.1 ms（A76） |

机理：fp16 图的输入张量是 FLOAT16，运行时必须在用户态把 640×640×3 个元素**逐个**展宽。

## 2. 试过的五条替代路径（全部实测，全部**没有**比驱动更快）

| # | 路径 | 输入侧 | `rknn_run` | NPU 段合计 | 判定 |
|:--|:--|--:|--:|--:|:--|
| 0 | **驱动 uint8 NHWC 上传（基线）** | **11.15 ms** | 102.07 ms | **113.2 ms** | ✅ **最优** |
| 1 | `rknn_set_io_mem` 零拷贝（NHWC） | **0.78 ms** | **115.59 ms** | 116.4 ms | ❌ 净亏 3.2 ms |
| 2 | `rknn_set_io_mem` 零拷贝（NCHW 布局） | — | — | — | ❌ 驱动**忽略数据**：输出与输入无关 ⇒ FATAL |
| 3 | `type=FLOAT16` 类型上传（我们 NEON 转好的 fp16） | 13.28 ms | 102.25 ms | 115.5 ms | ❌ 比 uint8 路径还慢 2.1 ms |
| 4 | `pass_through=1` 直传 fp16 | — | — | — | ❌ 驱动按 4 B/px 校验：报 `param input size(2457600) < model input size(4915200)`；补齐 4 B/px ⇒ **SIGSEGV** |
| 5 | `type=FLOAT32` 类型上传（10-16 试过） | 46.7–47.8 ms | — | — | ❌ 没降；先除 255 ⇒ 0 检出 |

### 2.1 为什么零拷贝反而更慢（中间量，不是猜测）

- 零拷贝把输入**暂存进 NPU 内存**这件事从驱动的 `inputs_set` 挪到了 `rknn_run`，
  而那条路子的带宽只有 **≈180 MB/s**（2.4 MB / 13.6 ms）。
- 代价**按字节线性**：int8 输入 1.2 MB ⇒ +6.2 ms；fp16 输入 2.4 MB ⇒ +13.6 ms。
  ⇒ 这是**一次慢速拷贝**，不是 NPU 算力。
- 驱动的 uint8 路径是"边转换边写进 NPU 的输入缓冲"（一趟），所以反而最省。

## 3. 机理核对：输入张量的数值空间（`--conv-div`）

板端 `RKNN_QUERY_INPUT_ATTR`（f16 模型）：

```
input dims=[1,640,640,3] fmt=NHWC type=FLOAT16
input attr extra: zp=0 scale=1.0000000000 size=2457600 size_with_stride=2457600 w_stride=640
```

**`scale = 1.0`** ⇒ 输入张量里放的是**原始 0..255**，`÷255` 是图内的算子。
实测两侧吻合：`--conv-div 1` 与基线 TSV **md5 相同**；`--conv-div 255`（预先除过）**12 帧全灭**。

## 4. 真正可用的杠杆：CPU 绑核（**4.1×，零成本**）

`inputs_set` 的展宽是**单线程标量循环**，跑在小核 A55（1.8 GHz）上是 45.41 ms，
搬到 A76（2.35 GHz）是 **11.15 ms**；给两个核与给一个核**完全一样**（14.72 vs 14.73）。

| 核 | `inputs_set` |
|:--|--:|
| A55（cpu 0–3, 1.8 GHz） | 45.41 ms |
| **A76（cpu 4–7, 2.35 GHz）** | **11.15 ms** |

⇒ **给板端评测脚本加 `--cpu-affinity 6`（或 `taskset -c 6`）**，
`rknn_heatmap_smoke` 与 `e2e_par2_demo` 都已支持（后者本来就有 T 档绑核）。

## 5. 顺带修掉的两个真 bug

1. **`e2e_core.cpp` 的 `letterbox_feature_f16` 是错的**：非 div255 分支写的是
   `(*dst)[i] = (uint16_t)boxed[i];` —— 把**原始字节当成 fp16 位模式**。
   已改为调用共享的 `letterbox_to_f16()`（逐位一致，见 §7）。
2. **fp16 输入张量存的是"码"不是"值"**（int8 图）：`type=INT8, zp=-128, scale=1/255`
   ⇒ 主机字节 x 的码是 `x + zp = x − 128`；驱动做的是**字节级重解释**。
   ⇒ 零拷贝时**原样 memcpy 是错的**（`0x80` 解成 −128 而不是 0），
   实测后果：**检测 TSV md5 不同**。修正为 `dst[i] = (int8_t)(src[i] + zp)`。

## 6. 单进程实测（12 帧，f16 模型，绑 A76，空闲板卡）

| 臂 | 输入路径 | 输入侧 | `rknn_run` | NPU 合计 | 检测 md5 |
|:--|:--|--:|--:|--:|:--|
| A 基线 | 驱动 uint8 NHWC | **11.147** | 102.07 | **113.2** | `58c7cbd3…` |
| A' 基线不绑核 | 同上 | **33.567** | 101.68 | 135.2 | 同上 |
| B 零拷贝 | `set_io_mem` + 主机 NEON | **0.780** | 115.59 | 116.4 | 同上 |
| C 零拷贝（融合 letterbox） | 同上，canvas 直写 fp16 | 0（并入 letterbox 1.03） | 115.47 | 116.5 | 同上 |
| D FLOAT16 类型上传 | 我们的 NEON fp16 + typed 上传 | 13.281 | 102.25 | 115.5 | 同上 |
| E int8 基线 | 驱动 | 0.118 | 55.34 | **55.5** | — |
| F int8 零拷贝 | `set_io_mem` | 0.078 | **61.56** | 61.6 | 同上 |

⇒ **B/C/D/F 全部不优于 A**（C 的 letterbox 1.03 ms 里已含转换）。

## 7. 融合 letterbox（`letterbox_to_f16`）：**逐位一致，但当前无收益**

- 目标：省掉 1.2 MB 的 uint8 中间缓冲（letterbox 直接写 fp16）。
- 正确性：`--selftest-lb` 逐位对照 `letterbox()` + `u8->f16`，
  **三种几何（512×512 放大 / 640×512 加边 / 501×377）全部 PASS（0 mismatch）**。
- 收益：**驱动路径要的就是 uint8 NHWC ⇒ 融合对主路径无用**；
  只在零拷贝路径下有意义，而零拷贝本身净亏 ⇒ **整条路不推广**。
- 保留价值：一个**逐位一致、带自检**的 fp16 画布生成器（未来若换 runtime/驱动可能重新有用）。

## 8. 流水线端到端（4 级：GMC ∥ letterbox ∥ NPU0 ∥ NPU1，序列 34_1，537 帧）

| 臂 | `inputs_set` | `rknn_run` | 每 ctx | **FPS** | 检测 | 链哈希 |
|:--|--:|--:|--:|--:|--:|:--|
| **旧二进制（已认证）** | 14.72 | 108.51 | 125.8 | **15.33** | 571 | `90f11c5b` / `b3b34ee5` |
| 新二进制（默认路径） | 14.74 | 109.31 | 126.6 | **15.32** | 571 | 同上 |
| 新二进制 `--io-mem` | **0.78** | **125.56** | 128.8 | 15.04 | 571 | 同上 |
| 新二进制 `--io-mem --io-mem-f16` | **0.00** | 125.25 | 127.8 | 14.99 | 571 | 同上 |

⇒ 流水线是 **NPU 绑定**的（`rknn_run` 占 88–99%）：输入侧省 14 ms 完全被 `rknn_run`
多吃的 17 ms 吃掉，**帧率不升反降**。
⇒ **推广结论：不推广零拷贝；推广 CPU 绑核**（流水线 T 档已有：feat=4 / lb=5 / npu0=6 / npu1=7）。

## 9. 闸门（全部通过）

| 闸门 | 结果 |
|:--|:--|
| 新二进制**默认路径** vs 已认证旧二进制（fp16） | TSV **md5 相同** `58c7cbd3ad6ac68c846ba2f506885e59` |
| 同上（int8 模型） | TSV **md5 相同** |
| 流水线四臂 | 帧数 537、检测 571、每 worker 链哈希四臂一致 |
| `--selftest-conv` | **PASS（0 mismatch）**，256 取值 × 2 口径 |
| `--selftest-lb` | **PASS（0 mismatch）**，3 种几何 |
| e2e 三支二进制重编译 | 通过 |

## 10. 复现命令（板卡；帧与产物都走 NFS，板卡 `/` 只剩 1.4 GB）

```bash
ssh -p 22 root@192.168.0.64
W=/mnt/manu/ptq_iomem; M=/userdata/ptq_models/p0_off_A_none.rknn; B=/root/rknn_heatmap_smoke_iomem
mkdir -p $W
ls /mnt/manu/pcfeat_bmp/images | grep '^01_4485_1167-2666__' | head -12 | sed 's|^|/mnt/manu/pcfeat_bmp/images/|' > $W/list12.txt

$B $M --selftest-conv
$B $M --selftest-lb
$B $M --bench-fill --io-mem --cpu-affinity 6      # 转换 vs 写设备内存

# 四条路径（绑 A76）
$B $M --batch $W/list12.txt --th 0.25 --topk 100 --cpu-affinity 6                 --detections-out $W/A.tsv
$B $M --batch $W/list12.txt --th 0.25 --topk 100 --cpu-affinity 6 --io-mem         --detections-out $W/B.tsv
$B $M --batch $W/list12.txt --th 0.25 --topk 100 --cpu-affinity 6 --upload-f16      --detections-out $W/D.tsv
md5sum $W/A.tsv $W/B.tsv $W/D.tsv        # 必须相同（数值等价）

# 流水线（板卡）
cd /mnt/manu/armfeat
/root/e2e_par2_demo_iomem --raw-root bmpraw --sequence 34_1 \
  --model /mnt/manu/npu/data/trial0474_hm_nop0_fp16.rknn --kernel-dir /root/kernels \
  --fused gpu --anchor-step 10 --th 0.25 --workers 2 --depth 6 \
  --cpu-feat 4 --cpu-lb 5 --cpu-npu0 6 --cpu-npu1 7 --out-root /mnt/manu/ptq_iomem/e2e_new
```

## 11. 本轮踩的坑（NEON 位运算 + 语义，勿重犯）

1. **`vcgtq_u32` / `vceqq_u32` 返回全 1 掩码（0xFFFFFFFF），不是 0/1** ⇒ 当布尔用会得到
   `inc = 0xFFFFFFFF`（加 1 变减 1，fp16 差 1 ULP）。必须 `& 1` 归一。
2. **`vreinterpretq_u16_u32()` 把 4 个 u32 当成 8 个 u16**（高低半字交错）⇒ 存出来全是乱码。
   要用 `vmovn_u32()`（narrow 取低 16 bit）打包。
3. **指数位不能从"已掩掉指数的尾数"上取**：`mag = x & 0x7FFFFF` 之后再 `>>23` 永远是 0。
4. **LUT 里 n=0 必须特判**：`0 − 0x1C000` 无符号下溢 ⇒ 0x4000（fp16 的 2.0）。
5. `movi v.4s, #imm, msl #8` 在 objdump 里显示易误读（`#0x3, msl #8` 实际是 `0x3FF`）——
   验证常量要**在板上跑一次**看实际值。
6. **比较"输入侧"这一个指标会得出错误结论**（本轮踩过）：必须看 **NPU 段合计 / 整帧**。
7. **零拷贝之后必须重新填回输入**：null control 会把反相画布留在绑定张量里 ⇒ 首帧检测会错。

## 12. ✅ 已推广：CPU 绑核（用 `taskset`，不需要换板端二进制）

**改动**（2026-10-23 已落地，三侧 md5 一致）：
- `manu/pipeline/npu/ptq/04_run_board_fast_eval.py`：板端命令加 `taskset -c <cpu>`，
  新增 `--board-cpu`（默认 **6** = A76 大核；空串关闭）。
- `manu/pipeline/npu/ptq/14_run_full_scale_nofs.py`：同上。

**为什么用 `taskset` 而不是二进制新参数**：已认证的板端二进制
`/root/rknn_heatmap_smoke` **不需要替换**（替换会与另一个会话正在跑的任务冲突），
`taskset` 对它直接生效 ⇒ 零风险。

**板端实测（f16 模型 `p0_off_A_none`，12 帧，已认证二进制）**：

| | `inputs_set` | `rknn_run` | NPU 合计 | FPS | 检测 md5 |
|:--|--:|--:|--:|--:|:--|
| 不绑核 | **30.847 ms** | 103.17 | 136.89 | 7.3 | `58c7cbd3…` |
| **`taskset -c 6`** | **11.118 ms** | 102.14 | 114.36 | **8.7** | **同上（逐位一致）** |

⇒ **整帧 136.9 → 114.4 ms（+19% fps），零精度影响**。

## 13. 保留但**不推广**的东西（全部 opt-in，默认路径已验逐位不变）

- `--io-mem` / `--io-mem-f16`（零拷贝）：**净亏**，只作为负面对照保留。
- `--upload-f16`（FLOAT16 类型上传）：比 uint8 路径慢 2.1 ms。
- `--upload-nchw`（NCHW 上传）：驱动忽略数据 ⇒ 输出与输入无关。
- `--io-mem-nchw`：驱动忽略数据。
- `letterbox_to_f16()`：逐位一致、带自检的 fp16 画布生成器（当前无收益，留给未来的 runtime 替换）。
