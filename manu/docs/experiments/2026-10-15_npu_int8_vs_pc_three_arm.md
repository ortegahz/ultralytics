# NPU int8 去 P0 vs PC SOTA —— 三臂受控消融

日期：2026-10-15
实验目标：PC SOTA 模型配置一字不改，特征仍用 PC 版本，**只替换推理后端**为 ARM NPU int8（去 P0），测性能差。

---

## 1. 结论先行

| 臂 | 说明 | 峰值 th | F1 | P% | R% | TP | FP |
|---|---|--:|--:|--:|--:|--:|--:|
| 臂1 | PC-FP32 **+P0**（冻结基准） | 0.25 | **0.902316** | 94.87 | 86.02 | 21,601 | 1,167 |
| 臂2 | PC-FP32 **−P0** | 0.25 | **0.901935** | 94.86 | 85.96 | 21,586 | 1,169 |
| 臂3 | **NPU-int8 −P0** | **0.20** | **0.882231** | 94.27 | 82.90 | 20,818 | 1,265 |

代价分解（唯一变量受控）：

| 分解项 | 峰值 ΔF1 | ΔTP | 15 档均值 | 区间 | 负档 |
|---|--:|--:|--:|---|--:|
| 去 P0（臂1→臂2） | **−0.000382** | −15 | −0.004296 | [−0.028273, +0.001676] | 11/15 |
| int8 量化（臂2→臂3） | **−0.019704** | −768 | −0.016615 | [−0.026206, −0.001469] | **15/15** |
| **合计（臂1→臂3）** | **−0.020085** | −783 | −0.020911 | [−0.029743, +0.000048] | 14/15 |

固定阈值 th=0.25 同点对比：臂2 0.901935 vs 臂3 0.880949，ΔF1 = **−0.020986**。

**要点**

1. **损失 98% 来自 int8 量化，不来自去 P0。** 去 P0 的 −0.000382 相当于 15 个 TP，而本口径噪声下限是 ±3 TP
   （≈±0.0002），5 倍噪声、但绝对量级可忽略；int8 的 −0.019704 是它的 **51 倍**。
2. **int8 是全档位一致劣化**：15/15 档全负，区间上界也只有 −0.001469，不是某几档的抖动。
3. **NPU 最优阈值从 0.25 漂到 0.20**，int8 量化把分数分布整体压低了，e2e 脚本里固化的 `TH=0.25`
   不是 NPU 模型的最优工作点。
4. **召回是唯一受创项**：R 85.96% → 82.90%（−3.06pt，TP −768）；精度 P 仅 94.86% → 94.27%（−0.59pt）。
   NPU 更保守、更稀疏。
5. 佐证稀疏化：NPU 每帧平均 **0.83** 个检出，**7,823/31,613 = 24.8% 的帧零检出**；PC cache 每帧最多 11 个。

---

## 2. 受控性保证

三臂共用：同一批 PC-BMP 特征（`pcfeat_bmp/images` 31,613 帧）、同一 harvest（`--conf-thresh 0.08 --top-k 80`，
板端 `--th 0.08 --topk 80`）、同一 15 档阈值 0.05~0.75、同一 8px 匹配、GT 一律取自 PC cache。

自校准闸门两道，全部 PASS：

- 对比1/3 以臂1 为 A 臂，硬校验 `F1=0.902316 TP=21601 FP=1167` → PASS（冻结 headline 逐位复现）
- 对比2 以臂2 为 A 臂，期望值从对比1 的 JSON 读取（非手抄）`F1=0.901935 TP=21586 FP=1169` → PASS

每臂均报 `[OK] 严格同集配对: 31613 帧一一对应` 与 `[OK] GT 逐项一致: 合计 GT=25111`。

板卡完整性硬校验（不齐就 exit，不拿残缺数据算 F1）：24/24 序列、总行数 **31,613**、**零失败零重试**。

---

## 3. 执行环境

| 项 | 值 |
|---|---|
| 板卡 | `ssh -p 22 root@192.168.0.64`，evm3588 / RK3588 / Mali-G610，8 CPU 核，NPU governor 锁 1 GHz |
| 板卡二进制 | `/root/rknn_heatmap_smoke` md5 `170379506adb4aeaef2f5a5f8ce65cc2`（AArch64） |
| 模型 | `/mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn` 3,799,304 B，md5 `ef302edb1b5aff1d376204bb7ad77550`，CPU 算子 0 |
| 训练服务器 | `ssh -p 32222 huangzhe@192.168.99.40`，项目 `/tmp/pycharm_project_10ae9e2e` |
| 解释器 | `/home/huangzhe/anaconda3/envs/uav/bin/python`（py3.10.20 / cv2 4.10.0 / numpy 1.26.4） |
| 官方 val | 24 序列 / 31,613 帧 / GT 25,111 |

---

## 4. 脚本与可复制命令

### 4.1 脚本路径

| 脚本 | md5 | 作用 |
|---|---|---|
| `manu/pipeline/npu/run_npu_pcfeat_board.sh` | `7b73facac2b13d65e712374db90124e0` | 板卡 NPU 全量（跳过 GMC，直接吃 PC 特征） |
| `manu/pipeline/npu/bench_npu_parallel.sh` | `c479c2fc11773c90d0a4637c15f8076f` | 并行度扫描 + dmesg 稳定性监控 |
| `manu/diagnostics/npu_tsv_to_cache.py` | `6ad269efe6ac55875249a6c887463fad` | 板端 TSV → cache pkl |
| `manu/evaluation/eval_arm_pc_gmc_ab.py` | （既有，未改） | 两臂 15 档 A/B + 自校准闸门 |
| `manu/evaluation/run_three_arm_eval.sh` | `1b22fa75cd0350c451ad713ecd970b5f` | 三臂编排（3 次两两对比 + 汇总） |
| `manu/evaluation/three_arm_pipeline.sh` | `03293a5c5cf733a6976f557d1817398c` | 端到端收尾（等待→校验→同步→评估） |
| `manu/diagnostics/check_npu_tsv_vs_cache.py` | `ed3b82471e8b363734d4164c1e39b73f` | TSV 坐标校验 |

> 注：`check_npu_tsv_vs_cache.py` 早期交接记录写的是 `26ce1a85...`，**实际是 `ed3b82471e8b...`**，以本表为准。

### 4.2 完整命令

```sh
# ---- 步骤 1：板卡 NPU 全量推理（单进程，24 序列 31,613 帧）----
scp -P 22 manu/pipeline/npu/run_npu_pcfeat_board.sh root@192.168.0.64:/root/run_npu_pcfeat.sh
ssh -p 22 root@192.168.0.64 'chmod +x /root/run_npu_pcfeat.sh; : > /mnt/manu/npu_pcfeat.log; nohup /root/run_npu_pcfeat.sh > /mnt/manu/npu_pcfeat.log 2>&1 &'

# ---- 步骤 2：端到端收尾（校验 24 序列 + 31613 行 → 同步 → 三臂评估）----
sh manu/evaluation/three_arm_pipeline.sh

# ---- 步骤 3（等价，服务器侧手动跑）----
ssh -p 32222 huangzhe@192.168.99.40 'sh /tmp/pycharm_project_10ae9e2e/manu/evaluation/run_three_arm_eval.sh'

# ---- 附：并行度扫描（可选，复现 §5 结论）----
scp -P 22 manu/pipeline/npu/bench_npu_parallel.sh root@192.168.0.64:/root/bench_npu.sh
ssh -p 22 root@192.168.0.64 'chmod +x /root/bench_npu.sh; sh /root/bench_npu.sh'
```

### 4.3 三臂 cache 与报告产物 md5

| 产物 | md5 |
|---|---|
| `runs/gmc_eval/pc_bmp_full_cache.pkl`（臂1） | `59c0a19bc8fa4705d37e51fdd8f4a0c5` |
| `runs/gmc_eval/pc_bmp_nop0_cache.pkl`（臂2） | `8433a526b109f17bbbf1f5bb582c5557` |
| `runs/gmc_eval/arm_npu_nop0_cache.pkl`（臂3） | `928816ccdd0823f70ecef8a03a119567` |
| `runs/three_arm/cmp1_arm1_vs_arm2.json` | `cdee5bca05e5e0173da925ebfc46ecc4` |
| `runs/three_arm/cmp2_arm2_vs_arm3.json` | `f72938d68daddf10dda496fef0b2faeb` |
| `runs/three_arm/cmp3_arm1_vs_arm3.json` | `2e49a78b8271036e52b59289fc01aa44` |

（路径均相对项目根 `/media/manu/1TB-Volume/workspace/ultralytics/`）

---

## 5. 附带发现：RK3588 NPU 并行度硬上限是 2

扫描过程中发现 RK3588 **标称 3 个 NPU 核，但第 3 个并发上下文一提交就撞 RKNPU 6 秒 job timeout**，
驱动 `soft reset` 整个 NPU，在飞上下文全部陪葬（`rc=1` 或 `139`）。干净条件实测（每档冷页缓存）：

| 并发 P | 成功 worker | 真实完成帧 | 墙钟 | 真实吞吐 | dmesg 崩溃 |
|--:|--:|--:|--:|--:|--:|
| 1 | 1/1 | 60 | 6,050 ms | 9.92 f/s | 0 |
| **2** | **2/2** | 120 | 6,110 ms | **19.64 f/s** | 0 |
| 3 | 2/3 | 120 | 5,990 ms | 20.03 f/s | **1** |
| 4 | 2/4 | 120 | 18,940 ms | 6.34 f/s | 0 |
| 6 | 2/6 | 120 | 19,010 ms | 6.31 f/s | 0 |
| 9 | 2/9 | 120 | 19,070 ms | 6.29 f/s | 0 |

dmesg 证据：
```
RKNPU: job timeout ... elapsed time: 6232015us          (驱动看门狗 6,000,000us)
RKNPU: core 2 irq status: 0x0, require mask: 0x300
RKNPU: soft reset, num: 6
BUG: spinlock bad magic on CPU#2, kswapd0/146            (DEBUG kernel，non-fatal oops)
```

要点：

- **6 进程 / 9 进程不可行**，任何 P>2 都不会更快，只会空耗 ~19 s 超时窗口。
- P≥4 时"吞吐"是假的：9 进程只有 2 个真产出 60 帧，其余全废。
- `/sys/class/devfreq/fdab0000.npu/load` 单进程即显示 `100@1000000000Hz`，那是**三核合计**口径，
  **不能**用来判断有没有空核，必须实测吞吐 + 看 dmesg。
- 本轮最终按用户决定走**单进程**（稳优先），全程零失败零重试。

---

## 6. 踩过的坑（避免复现）

1. **benchmark 缺前置存在性检查**：用越界 `BASE` 取到不存在的帧，NPU 立刻 `rc!=0` 退出，脚本照样打印
   「120 帧 / 250ms / 480fps」的**假数据**。已在 `bench_npu_parallel.sh` 加清单存在性校验，并用越界输入
   验证过守卫会 `exit 3` 拦截。
2. **并行基准被污染**：最初那组 1/2/3 进程数据是在全量任务并发跑的条件下测的，不可用，已重测。
3. **自校准闸门此前完全失效**：`run_arm_pc_gmc_full_ab.sh:150-154` 压根没传任何 `--expect-*`。本轮已接上。
4. **`__<k+1>` 偏移不适用于本链路**：该 OFFSET=1 只存在于旧 ARM-GMC 装配路径
   （`run_arm_pc_gmc_full_ab.sh:123`）。板端写 `34_1__000003.bmp`、PC cache 存同名，**无偏移风险**。
5. `run_arm_pc_gmc_full_ab.sh:41` 的 `BOARD_TOTAL=31661` 是错的，实际 **31,613**。
