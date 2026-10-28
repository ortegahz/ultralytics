# 双缓存（输入 FIFO + 计算快照）验收报告

> 生成时间 2026-10-28 · 执行机：RK3588 板卡 `root@192.168.0.64` + 本地主机
> 被验收对象：`manu/pipeline/cpp/gmc_stream_ocl.cpp` 的 **GMC 计算快照**（第二级缓存）与
> `manu/pipeline/npu/mosaic_pipeline.cpp` 的 **相机侧输入 FIFO 深度**（第一级缓存）
> 关联产物：`manu/pipeline/npu/cmp_snapshot_board.sh`、`manu/pipeline/cpp/gmc_snapshot_census/`

---

## 0. 结论与验收边界（先说清楚「没验什么」）

| 判据 | 结果 | 证据 |
|:--|:--|:--|
| ① 计算快照与基线**逐帧特征指纹一致**（零丢帧、板卡、600 帧） | **PASS** | §4.2：step10 与 step2 两个配置各 **600/600 逐位相同，diff=0** |
| ② 快照**覆盖了每一次历史读取**（结构保证成立） | **PASS** | §4.3：`[SNAPSHOT] … not served from the private snapshot = 0`（两个配置 × 两个引擎） |
| ③ 快照拷贝开销可接受 | **PASS** | §4.3：板卡 **0.92 ~ 1.86 ms/帧** |
| ④ 读集是**实测**的而非推导的 | **PASS** | §2：5 组配置插桩取证，稳态读集恒为 22 帧 |
| ⑤ 第一级 FIFO 深度放开后**丢帧归零** | **PASS** | §5：`dropped=0`、`gmc_history_gaps=0`（D0/D1 臂，服务器 @S 打分） |
| ⑥ **快照版的端到端 F1（服务器 @S 打分）** | ❌ **未做** | §6.1 —— **这是本次验收唯一缺失项** |
| ⑦ **有洞序列（相机丢帧）下快照版 vs 基线的逐位对比** | ❌ **未做** | §6.2 |
| ⑧ **带 tracker 的时序链口径** | ❌ **未做** | §6.3 |

> **一句话**：快照「不改变任何一个比特」这件事**已被板卡 600/600 逐位证明**；
> 但**「快照版的端到端 F1 与基线一致」还没有一个服务器 @S 分数**，这是欠的。

---

## 1. 被验收的实现

### 1.1 第一级：相机侧输入 FIFO

| 项 | 内容 |
|:--|:--|
| 位置 | `manu/pipeline/npu/mosaic_pipeline.cpp:1203` |
| 改动 | `{"input_queue", &cfg_s.input_queue, 1, 8192, 8}`（原上限 **64**） |
| 语义 | 相机 push 队列 `in_q` 的深度。满了丢最老（`gmc_input_drop_policy=0`）。**它是整条链上唯一的丢帧入口** |
| 为什么放开 | 深度决定能吸收多少**产能差**；64 帧在 25 fps 下只有 2.56 s 缓冲。放开上限让深度可以按「相机−GMC 产能差 × 抖动裕度」来定，而不是被硬钳在 64 |
| 验收证据 | §5：D0（1024）/ D1（4096）两臂实测 `dropped=0`、`gmc_history_gaps=0` |

### 1.2 第二级：计算快照（私有历史拷贝）

| 项 | 内容 |
|:--|:--|
| 位置 | `manu/pipeline/cpp/gmc_stream_ocl.cpp` |
| 新增成员 | `snap_`（`std::vector<Mat>`，22 槽）、`snap_ok_`（有效位）、`snap_first_`（冷启钳位目标）、`n_snapshot_miss_`（计数器） |
| 新增函数 | `build_snapshot()` :2280 · `copy_snapshot_slot()` :2298 · `snapshot_slot()` :2331 |
| 改造点 | `frame_at_lag()` :2353 —— 由「直接读共享 ring」改为「**快照优先，miss 回退 ring**」 |
| 调用时机 | `process_frame()` :1841 —— 在**任何拟合之前**调用 `build_snapshot()` |
| 落盘统计 | `mosaic_pipeline.cpp:1718-1730` —— 每个引擎收尾时打印 `[SNAPSHOT] …`（走 stderr，**不进公开 ABI 的 timing 结构**） |

**核心语义**（`copy_snapshot_slot()` :2298-2326）：

```cpp
snap_ok_[slot] = 0;                                   // 先置无效
if (idx <= clamp_lo()) return;                        // 冷启钳位 → 走 snap_first_
const size_t s = idx % ring_depth();
if (phys_anchor_ && ring_id_[s] != idx) return;       // ★ 历史有洞 → 槽位留无效
const Mat& src = ring_[s];
if (src.empty()) return;
snap_[slot].create(src.rows, src.cols, src.type());   // 一次性分配，逐帧复用
src.copyTo(snap_[slot]);
snap_ok_[slot] = 1;
```

★ **有洞时槽位被故意留成无效**，于是该次读取会走 `frame_at_lag()` 的原路径 —— 洞检测、`nearest_older` 回退、`n_history_gap_` 计数**全部原样保留**，且 `n_snapshot_miss_` 自增。

⇒ **正确性不依赖快照完整性**：快照只是「把像素换个地方放」，任何未覆盖的读取都自动退回改动前的行为。

---

## 2. 前置取证：读集是量出来的，不是推出来的

快照必须恰好覆盖「本次计算会读的那一组物理帧」。少一个 → 回退 ring（结构保证失效）；多一个 → 白占内存。

**所以先插桩、后实现。** 在 `frame_at_lag()` 里临时记录它**实际服务的物理索引**（不是被请求的索引 —— 冷启钳位和洞回退服务的是**另一帧**），跑完即撤。插桩代码已移除，留档在 `manu/pipeline/cpp/gmc_snapshot_census/`：

| trace 文件 | anchor step | 尾 | 物理锚定 |
|:--|:--|:--|:--|
| `census_anchorstep2.log` | 2 | CPU | off |
| `census_anchorstep10.log` | 10 | CPU | off |
| `census_anchorstep2_physanchor.log` | 2 | CPU | on |
| `census_anchorstep10_physanchor.log` | 10 | CPU | on |
| `census_anchorstep10_gputail.log` | 10 | CPU+GPU | off |

**结果**：稳态下**所有配置读到的物理帧集合完全相同**：

```
{t} ∪ {t−2, t−4, …, t−42}   = 22 帧
```

每帧**读取次数**随配置不同（step2/CPU = 44 次、step10/CPU = 30 次、step10/CPU+GPU = 51 次），
但**集合**一致 —— 因为 GPU 尾与 CPU 尾都无条件遍历 `lags_` 的全部 21 个 lag，与 anchor-step 无关。

**两个光看代码拿不到的数**：

1. **读到的最深帧是 42**（= `max_lag_`）。**从未观察到 44** —— 原设计里担心的「组合拟合会读出 t−44」不成立。
2. **冷启时深 lag 坍缩到钳位目标 `first_`**，早期帧读的是 `{t} ∪ {first_}` ⇒ 快照必须**单独存一份 `first_`**（`snap_first_`）。

复现命令（`manu/pipeline/cpp/gmc_snapshot_census/README.md`）：
```bash
python3 make_synth.py 60 seq
cmake --build build --target gmc_stream_ocl -j
GMC_TRACE_READS=1 ./build/gmc_stream_ocl --raw-root . --sequence seq \
    --limit 60 --anchor-step 10 --fused cpu > census_anchorstep10.log
python3 agg.py census_anchorstep10.log
```

---

## 3. 验收方法与判据

脚本：`manu/pipeline/npu/cmp_snapshot_board.sh`（板卡 `/root/cmp_snapshot_board.sh`）

```bash
export LD_LIBRARY_PATH=/root/psearch/fixlibs
bash /root/cmp_snapshot_board.sh /root/rknn_mosaic_e2e_sweep4 /root/rknn_mosaic_e2e_snap /tmp/snapcmp
```

**三个判据**：

| # | 判据 | 为什么这样定 |
|:--|:--|:--|
| 1 | **逐帧特征指纹一致** | 快照只搬像素不改像素 ⇒ GMC 特征必须逐位相同。`feature_md5.txt` 第 3 列是滚动 MD5（每帧一个） |
| 2 | **`n_snapshot_miss_ == 0`** | 非 0 = 某次历史读取没被私有副本覆盖，回退了共享 ring ⇒ **数字仍对，但结构保证失效**，这必须显式暴露而不是默认通过 |
| 3 | **拷贝开销可接受** | 快照每次计算要拷 22 帧 × 320 KB ≈ 7 MB，必须确认它不吃掉帧预算 |

**为什么必须零丢帧**：基线的历史保证是**调参得来的**（ring 比窗口深 `gmc_ring_slack` 帧，生产者最多领先那么多）。在**丢过帧**的序列上，基线自己就在用 `nearest_older` 读替换帧，两个版本可能合理地不同。所以验收必须跑在**相机队列从未溢出**的序列上 —— 故 `--input-queue 1024`。

---

## 4. 板卡实测结果（原文）

### 4.1 二进制与命令

```
=== binaries ===
c448229d23fa98aefe2ffc1e1cb1045d  /root/rknn_mosaic_e2e_sweep4   ← 基线（改动前同一份源码）
8a5a04e9585194ac1d9022b4a0a72b5f  /root/rknn_mosaic_e2e_snap      ← 快照版
```

每个配置跑两次（基线 / 快照），除二进制外**参数完全相同**：
```
--raw-root /mnt/srv2/lq_ir_bmp --sequence VIDEO00007_19700101_001324 \
--model /root/psearch/models/p0_off_A_none.rknn --kernel-dir /root/psearch/kernels \
--limit 600 --pace-ms 40 --input-queue 1024 --jitter-depth 3 --gmc-ring-slack 256 \
--npu-workers 2 --post-workers 1 --gmc-workers 2 --gmc-phys-anchor 1 --cpu-affinity 4-7
[--anchor-step 10]   （交付配置，5 锚点）
[--anchor-step 2 --gmc-anchor-threads 4 --gmc-affinity-group "0-3;4-7" --cpu-affinity 0-7]（21 锚点等价配置）
```

### 4.2 判据 ①：逐帧指纹

| 配置 | 基线帧数 | 快照帧数 | **逐位相同** | **不同** |
|:--|--:|--:|--:|--:|
| **anchor-step 10**（交付配置） | 600 | 600 | **600** | **0** |
| **anchor-step 2**（21 锚点等价配置） | 600 | 600 | **600** | **0** |

脚本判定原文：
```
  [anchor-step 10] baseline frames=600  snapshot frames=600  identical=600
  [OK] anchor-step 10: 600/600 frames bit-identical
  [anchor-step 2] baseline frames=600  snapshot frames=600  identical=600
  [OK] anchor-step 2: 600/600 frames bit-identical
```

**独立复核**（Leader 于 2026-10-28 13:4x 重新读取板卡留存现场 `/tmp/snapcmp/`）：
```
--- a10 ---   base frames=600  snap frames=600  identical=600  diff=0
--- a2  ---   base frames=600  snap frames=600  identical=600  diff=0
```

### 4.3 判据 ②③④：覆盖计数与拷贝开销

脚本判定原文：
```
[SNAPSHOT] engine 0: history reads not served from the private snapshot = 0  (every history read came from the private snapshot) | snapshot copy cost 0.98 ms/frame
[SNAPSHOT] engine 1: history reads not served from the private snapshot = 0  (every history read came from the private snapshot) | snapshot copy cost 1.05 ms/frame
[OK] anchor-step 10: every history read came from the private snapshot

[SNAPSHOT] engine 0: history reads not served from the private snapshot = 0  (every history read came from the private snapshot) | snapshot copy cost 1.86 ms/frame
[SNAPSHOT] engine 1: history reads not served from the private snapshot = 0  (every history read came from the private snapshot) | snapshot copy cost 0.92 ms/frame
[OK] anchor-step 2: every history read came from the private snapshot

=== RESULT: PASS -- the snapshot is bit-identical to the baseline and covered every read. ===
```

| 配置 | 引擎 0 | 引擎 1 | miss 计数 |
|:--|--:|--:|--:|
| anchor-step 10 | 0.98 ms/帧 | 1.05 ms/帧 | **0 / 0** |
| anchor-step 2 | 1.86 ms/帧 | 0.92 ms/帧 | **0 / 0** |

**开销占比**：相对各配置自身的 `total_wall`（step10 68.2 ms/帧、step2 139.1 ms/帧），拷贝开销约 **1.4% ~ 2.7%**。

---

## 5. 第一级 FIFO 的精度证据（服务器 @S 打分）

| 臂 | 口径 | 帧数 | TP | FP | FN | Recall | Precision | F1 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| **D0**（step10 + FIFO 1024） | arm@S | 6950 | 4810 | 147 | 1200 | 0.800333 | 0.970345 | 0.877177 |
| | ref@S | 6950 | 4812 | 146 | 1198 | 0.800666 | 0.970553 | 0.877462 |
| | **arm@S − ref@S** | | **−2** | **+1** | **+2** | −0.000333 | −0.000208 | **−0.000285** |
| **D1**（step2 + FIFO 4096） | arm@S | 3861 | 2135 | 153 | 1380 | 0.607397 | 0.933129 | 0.735826 |
| | ref@S | 3861 | 2135 | 153 | 1380 | 0.607397 | 0.933129 | 0.735826 |
| | **arm@S − ref@S** | | **+0** | **+0** | **+0** | **+0.000000** | **+0.000000** | **+0.000000** |
| 参照 | ref@fullGT | 11810 | 8640 | 200 | 1895 | 0.820123 | 0.977376 | 0.891871 |

两臂的 `dropped = 0`、`gmc_history_gaps = 0`。**D1 的 TP/FP/FN 与认证零丢帧参照逐项全 0**。

---

## 6. 本次验收**未覆盖**的边界（诚实清单）

### 6.1 快照版从未做端到端 F1 打分 —— **这是唯一缺失项**

板卡验收只把两个二进制跑在 `/tmp/snapcmp/`，**从未发布到 `/mnt/srv2/psearch/results/`**，因此
**服务器打分看门狗从未见过快照版**（本地核查：`ls /mnt/srv2/psearch/results/ | grep -i snap` → 空）。

⇒ 「快照不改变任何一个比特」是**推断出来的**（特征逐位相同 ⇒ NPU 输入相同 ⇒ 检出必然相同），
但**没有一个服务器 F1 数字直接证明它**。补法：快照版跑同一配置 → 发布 → 看门狗打分 → 与基线版的 @S 分数对照。

### 6.2 有洞序列下未做逐位对比

丢帧序列上基线本身就在用 `nearest_older` 读替换帧，两版可能合理地不同。该场景**不在本次验收范围内**。
快照版在洞处的行为是：**槽位留无效 → 回退原 ring 路径**（洞检测/`nearest_older`/gap 计数原样），
即**与基线同路径**，但**未做逐位实测**。

### 6.3 时序链口径未验

所有门禁/对照都是**单帧臂**（`system_sota_longquan.py --no-pipeline`，无 track）。
「零丢帧是硬性必要条件」在**带 tracker 的时序链**口径上仍未被推翻。

### 6.4 拷贝开销在 canonical（21 锚点）下更大

板卡实测：step10 为 0.98/1.05 ms/帧，**step2 为 1.86/0.92 ms/帧**。两个引擎的差异来自
「引擎 0 与引擎 1 在各自的 anchor 网格上落在哪个 lag」，不是随机波动。

### 6.5 降分辨率省内存不可行（只评估，未改）

`downscale=2` 只作用于**拟合段**；而 **GPU 尾**上传全分辨率（640×512→RGBA）、**CPU 尾**在全分辨率
`warpAffine`、`median_residual` 与三通道拼接也是全分辨率 ⇒ 尾端必须拿到全分辨率帧，
**4 倍省内存在不改变尾部分辨率语义的前提下拿不到**。

---

## 7. 本地预验收（x86，60 帧合成序列，基线 vs 快照）

| 配置 | 逐位一致 | snapmiss | 拷贝开销 |
|:--|:--|:--:|:--|
| anchor-step 2 / cpu | 60/60 | 0 | 0.48 ms/帧 |
| anchor-step 2 / both | 60/60 | 0 | 0.45 ms/帧 |
| anchor-step 10 / cpu | 60/60 | 0 | 0.48 ms/帧 |
| anchor-step 10 / both | 60/60 | 0 | 0.48 ms/帧 |

**内存**：每引擎实例 **+7.19 MiB**（22 窗口帧 + 1 钳位目标 = 23 帧 × 320 KB）。
`state_bytes()` 实测 13.44 → 20.62/20.63 MiB（delta +7.18 MiB，吻合）。
`--gmc-workers 2` ⇒ 整机 +14.4 MiB。缓冲区一次性分配、逐帧复用，稳态零分配。

---

## 8. 结论

**双缓存的实现与验收是成立的**：

1. **读集实测确定**（5 组配置插桩取证，稳态恒为 22 帧 `{t} ∪ {t−2…t−42}`，最深 42 从未 44）；
2. **板卡 600/600 逐位一致**（step10 与 step2 两个配置，diff=0）；
3. **覆盖完整**（`n_snapshot_miss_ = 0`，两个配置 × 两个引擎）；
4. **开销可接受**（0.92~1.86 ms/帧，占帧预算 1.4%~2.7%）；
5. **回退语义健全**（洞 → 槽位无效 → 原 ring 路径，正确性不依赖快照完整性）；
6. **第一级 FIFO 的精度收益已由服务器 @S 证明**（D1：ΔTP/ΔFP/ΔFN 全 0）。

**欠一件事**：**快照版的服务器 @S 分数**（§6.1）。补跑一次「同配置、只换二进制、发布到
`/mnt/srv2/psearch/results/`」即可闭环。

---

## 9. 复现方式

```bash
# 板卡：快照位级验收（基线 vs 快照）
scp -P 22 manu/pipeline/npu/out/rknn_mosaic_e2e_sweep4 root@192.168.0.64:/root/
scp -P 22 manu/pipeline/npu/out/rknn_mosaic_e2e_snap      root@192.168.0.64:/root/
scp -P 22 manu/pipeline/npu/cmp_snapshot_board.sh       root@192.168.0.64:/root/
ssh -p 22 root@192.168.0.64 'export LD_LIBRARY_PATH=/root/psearch/fixlibs; \
  bash /root/cmp_snapshot_board.sh /root/rknn_mosaic_e2e_sweep4 /root/rknn_mosaic_e2e_snap /tmp/snapcmp'

# 板卡：快照版的服务器打分（补 §6.1 的缺口）—— 跑完发布到发布目录即可被看门狗拾取
#   /root/rknn_mosaic_e2e_snap ... --out /root/psearch/results/<ARM>
#   cp <ARM>/* /mnt/srv2/psearch/results/<ARM>/
# 服务器：存活帧口径打分
ssh -p 32222 huangzhe@192.168.99.40 'sed -n "/## 2. 精度对照/,/## 4/p" \
  /mnt/data/siping/ptq_work/surviving_frames/<ARM>.md'
```

**产物路径**
- 板卡验收现场：`/tmp/snapcmp/{base,snap}_{a10,a2}/`
- 插桩取证：`manu/pipeline/cpp/gmc_snapshot_census/`（README + 5 份 trace + 复现脚本）
- 二进制：`/root/rknn_mosaic_e2e_sweep4` md5 `c448229d23fa98aefe2ffc1e1cb1045d`（基线）、
  `/root/rknn_mosaic_e2e_snap` md5 `8a5a04e9585194ac1d9022b4a0a72b5f`（快照版）
- 认证二进制全程未被覆盖（7/7 md5 未变）
